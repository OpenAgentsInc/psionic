#!/usr/bin/env node
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { dirname, join } from "node:path";

const defaultCorpus = new URL(
  "../fixtures/csm/benchmarks/csm_speech_benchmark_corpus.v1.json",
  import.meta.url,
);

const args = parseArgs(process.argv.slice(2));
const baseUrl = stripTrailingSlash(
  args.url || process.env.PSIONIC_CSM_BENCHMARK_URL || "http://127.0.0.1:8081",
);
const corpusPath = args.corpus || defaultCorpus;
const repeat = Number.parseInt(args.repeat || process.env.PSIONIC_CSM_BENCHMARK_REPEAT || "1", 10);
const timeoutMs = Number.parseInt(
  args.timeoutMs || process.env.PSIONIC_CSM_BENCHMARK_TIMEOUT_MS || "60000",
  10,
);
const outPath =
  args.out ||
  join(new URL("../target/csm-benchmarks", import.meta.url).pathname, `csm-speech-${Date.now()}.json`);

if (!Number.isFinite(repeat) || repeat < 1) {
  throw new Error(`invalid repeat: ${args.repeat}`);
}

const corpus = JSON.parse(await readFile(corpusPath, "utf8"));
const cases = Array.isArray(corpus.cases) ? corpus.cases : [];
if (cases.length === 0) {
  throw new Error(`benchmark corpus has no cases: ${corpusPath}`);
}

const health = await fetchJson(`${baseUrl}/health`, timeoutMs);
const samples = [];
for (const testCase of cases) {
  for (let iteration = 0; iteration < repeat; iteration += 1) {
    samples.push(await runCase(testCase, iteration));
  }
}

const successful = samples.filter((sample) => sample.ok);
const failed = samples.filter((sample) => !sample.ok);
const report = {
  schema_version: "psionic.csm.speech_benchmark_report.v1",
  generated_at: new Date().toISOString(),
  base_url: baseUrl,
  corpus_schema_version: corpus.schema_version,
  repeat,
  health: {
    status: health.status,
    served_backend: health.served_backend,
    execution_engine: health.execution_engine,
    runtime: health.runtime,
  },
  summary: {
    sample_count: samples.length,
    success_count: successful.length,
    failure_count: failed.length,
    success_rate: samples.length === 0 ? 0 : successful.length / samples.length,
    served_backends: unique(successful.map((sample) => sample.served_backend).filter(Boolean)),
    execution_engines: unique(successful.map((sample) => sample.execution_engine).filter(Boolean)),
    accelerated_backends: unique(
      successful.map((sample) => sample.accelerated_backend).filter(Boolean),
    ),
    cpu_fallback_count: successful.filter((sample) => sample.cpu_fallback_reason).length,
    latency_ms: latencySummary(successful.map((sample) => sample.wall_latency_ms)),
    first_audio_latency_ms: latencySummary(
      successful.map((sample) => sample.first_audio_latency_ms).filter(Number.isFinite),
    ),
    full_generation_latency_ms: latencySummary(
      successful.map((sample) => sample.full_generation_latency_ms).filter(Number.isFinite),
    ),
    output_duration_ms: latencySummary(
      successful.map((sample) => sample.output_duration_ms).filter(Number.isFinite),
    ),
  },
  samples,
};

await mkdir(dirname(outPath), { recursive: true });
await writeFile(outPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report.summary, null, 2));
console.log(`wrote ${outPath}`);

if (failed.length > 0) {
  process.exitCode = 1;
}

async function runCase(testCase, iteration) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  const started = performance.now();
  try {
    const response = await fetch(`${baseUrl}/psionic/csm/speech`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        request_id: `bench_${testCase.id}_${iteration}`,
        model: "sesame/csm-1b",
        input: testCase.input,
        voice_profile_id: "openagents/default_female_v1",
        response_format: "wav",
        psionic_csm: {
          max_audio_length_ms: testCase.max_audio_length_ms,
          context_policy: "none",
        },
      }),
    });
    const wallLatency = Math.round(performance.now() - started);
    const body = Buffer.from(await response.arrayBuffer());
    const headers = response.headers;
    const sample = {
      case_id: testCase.id,
      iteration,
      expected_business_outcome: testCase.expected_business_outcome,
      ok: response.ok && body.length > 44 && body.subarray(0, 4).toString("ascii") === "RIFF",
      status: response.status,
      wall_latency_ms: wallLatency,
      wav_bytes: body.length,
      served_backend: headers.get("x-psionic-served-backend"),
      generation_backend: headers.get("x-psionic-generation-backend"),
      execution_engine: headers.get("x-psionic-execution-engine"),
      generation_execution_engine: headers.get("x-psionic-generation-execution-engine"),
      accelerated_backend: headers.get("x-psionic-accelerated-backend"),
      gpu_model: headers.get("x-psionic-gpu-model"),
      cpu_fallback_reason: headers.get("x-psionic-cpu-fallback-reason"),
      first_audio_latency_ms: numberHeader(headers, "x-psionic-first-audio-latency-ms"),
      full_generation_latency_ms: numberHeader(headers, "x-psionic-full-generation-latency-ms"),
      output_duration_ms: numberHeader(headers, "x-psionic-output-duration-ms"),
    };
    if (!sample.ok) {
      sample.error = body.toString("utf8").slice(0, 400);
    }
    return sample;
  } catch (error) {
    return {
      case_id: testCase.id,
      iteration,
      expected_business_outcome: testCase.expected_business_outcome,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
      wall_latency_ms: Math.round(performance.now() - started),
    };
  } finally {
    clearTimeout(timeout);
  }
}

async function fetchJson(url, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      throw new Error(`GET ${url} failed: ${response.status}`);
    }
    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
}

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith("--")) {
      throw new Error(`unexpected argument: ${arg}`);
    }
    const key = arg.slice(2).replace(/-([a-z])/g, (_, char) => char.toUpperCase());
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`missing value for ${arg}`);
    }
    parsed[key] = value;
    index += 1;
  }
  return parsed;
}

function numberHeader(headers, name) {
  const value = headers.get(name);
  if (!value) {
    return null;
  }
  const number = Number.parseFloat(value);
  return Number.isFinite(number) ? number : null;
}

function latencySummary(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (sorted.length === 0) {
    return null;
  }
  return {
    min: sorted[0],
    p50: percentile(sorted, 0.5),
    p95: percentile(sorted, 0.95),
    max: sorted[sorted.length - 1],
  };
}

function percentile(sorted, quantile) {
  const index = Math.min(sorted.length - 1, Math.ceil(sorted.length * quantile) - 1);
  return sorted[index];
}

function unique(values) {
  return [...new Set(values)];
}

function stripTrailingSlash(value) {
  return value.replace(/\/+$/, "");
}
