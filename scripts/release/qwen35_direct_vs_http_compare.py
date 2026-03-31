#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import platform
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
HEADER_NAMES = [
    "x-psionic-backend",
    "x-psionic-execution-mode",
    "x-psionic-execution-engine",
    "x-psionic-batch-posture",
    "x-psionic-scheduling-class",
    "x-psionic-prefill-decode-mode",
    "x-psionic-ttft-ns",
    "x-psionic-itl-ns",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the native qwen35 CUDA direct-engine lane against the "
            "native OpenAI-compatible HTTP lane, with an optional direct "
            "vLLM reference helper."
        )
    )
    parser.add_argument("--psionic-root", default=str(REPO_ROOT))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--direct-repeats", type=int, default=3)
    parser.add_argument("--http-concurrency", default="1,2,4")
    parser.add_argument("--contract", default="greedy_one_sentence")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--allow-direct-fallbacks", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--server-bin", default="")
    parser.add_argument("--bench-bin", default="")
    parser.add_argument("--vllm-model", default="")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    return parser.parse_args()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def repo_relative_path(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def run_command(args: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        args,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed.stdout


def git_revision(repo_root: Path) -> str:
    return run_command(["git", "rev-parse", "HEAD"], cwd=repo_root).strip()


def build_artifacts(repo_root: Path) -> None:
    run_command(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "psionic-serve",
            "--bin",
            "psionic-openai-server",
            "--example",
            "qwen35_cuda_bench",
        ],
        cwd=repo_root,
    )


def parse_concurrency_ladder(raw: str) -> list[int]:
    values = []
    for field in raw.split(","):
        field = field.strip()
        if not field:
            continue
        value = int(field)
        if value < 1:
            raise ValueError("http concurrency values must be at least 1")
        values.append(value)
    if not values:
        raise ValueError("http concurrency ladder cannot be empty")
    return values


def prompt_contract(contract_id: str, max_output_tokens: int) -> dict[str, Any]:
    if contract_id != "greedy_one_sentence":
        raise ValueError(f"unsupported contract `{contract_id}`")
    prompt = "Explain what Psionic is in one sentence."
    return {
        "contract_id": contract_id,
        "user_prompt": prompt,
        "messages": [{"role": "user", "content": prompt}],
        "decode_mode": "greedy",
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "stream": False,
        "max_output_tokens": max_output_tokens,
        "direct_example_args": ["--decode", "greedy"],
    }


def case_headers(headers: Any) -> dict[str, str]:
    output: dict[str, str] = {}
    for name in HEADER_NAMES:
        value = headers.get(name)
        if value is not None:
            output[name] = value
    return output


def ns_header_to_seconds(headers: dict[str, str], name: str) -> float | None:
    value = headers.get(name)
    if value is None:
        return None
    try:
        return int(value) / 1_000_000_000.0
    except ValueError:
        return None


def http_json_request(
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": "Bearer dummy",
            "Content-Type": "application/json",
        },
        method=method,
    )
    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            text = response.read().decode("utf-8")
            ended_at = time.perf_counter()
            return {
                "status": response.status,
                "payload": parse_json_payload(text),
                "headers": case_headers(response.headers),
                "started_at_monotonic": started_at,
                "ended_at_monotonic": ended_at,
            }
    except urllib.error.HTTPError as error:
        text = error.read().decode("utf-8")
        payload_json = parse_json_payload(text)
        ended_at = time.perf_counter()
        return {
            "status": error.code,
            "payload": payload_json,
            "headers": case_headers(error.headers),
            "started_at_monotonic": started_at,
            "ended_at_monotonic": ended_at,
        }


def allocate_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server_ready(base_url: str, timeout_seconds: float) -> float:
    started_at = time.perf_counter()
    deadline = started_at + timeout_seconds
    health_url = f"{base_url.rstrip('/')}/health"
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                if response.status == 200:
                    return time.perf_counter() - started_at
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError(f"server failed health check before timeout: {health_url}")


def fetch_model_id(base_url: str, timeout_seconds: float) -> str:
    response = http_json_request(
        method="GET",
        url=f"{base_url.rstrip('/')}/v1/models",
        payload=None,
        timeout_seconds=timeout_seconds,
    )
    if response["status"] != 200:
        raise RuntimeError(f"/v1/models failed: {response['status']} {response['payload']}")
    data = response["payload"].get("data") or []
    if not data:
        raise RuntimeError("server returned no models from /v1/models")
    model_id = data[0].get("id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"server returned malformed model id payload: {response['payload']}")
    return model_id


def http_body(contract: dict[str, Any], model_id: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": contract["messages"],
        "temperature": contract["temperature"],
        "top_p": contract["top_p"],
        "seed": contract["seed"],
        "max_tokens": contract["max_output_tokens"],
        "stream": contract["stream"],
    }


def parse_json_payload(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"raw_body": text}
    return payload if isinstance(payload, dict) else {"raw_body": payload}


def summarize_http_response(raw: dict[str, Any], request_index: int) -> dict[str, Any]:
    if raw["status"] != 200:
        raise RuntimeError(
            f"chat completions request {request_index} failed with {raw['status']}: {raw['payload']}"
        )
    payload = raw["payload"]
    usage = payload.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    wallclock_s = raw["ended_at_monotonic"] - raw["started_at_monotonic"]
    ttft_s = ns_header_to_seconds(raw["headers"], "x-psionic-ttft-ns")
    itl_s = ns_header_to_seconds(raw["headers"], "x-psionic-itl-ns")
    finish_reason = None
    choices = payload.get("choices") or []
    if choices:
        finish_reason = choices[0].get("finish_reason")
    return {
        "request_index": request_index,
        "http_status": raw["status"],
        "wallclock_s": wallclock_s,
        "ttft_s": ttft_s,
        "itl_s": itl_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tok_s": (completion_tokens / wallclock_s) if completion_tokens and wallclock_s else 0.0,
        "finish_reason": finish_reason,
        "headers": raw["headers"],
        "started_at_monotonic": raw["started_at_monotonic"],
        "ended_at_monotonic": raw["ended_at_monotonic"],
    }


def run_http_request(
    *,
    base_url: str,
    contract: dict[str, Any],
    model_id: str,
    timeout_seconds: float,
    request_index: int,
) -> dict[str, Any]:
    raw = http_json_request(
        method="POST",
        url=f"{base_url.rstrip('/')}/v1/chat/completions",
        payload=http_body(contract, model_id),
        timeout_seconds=timeout_seconds,
    )
    return summarize_http_response(raw, request_index)


def run_http_concurrency_case(
    *,
    base_url: str,
    contract: dict[str, Any],
    model_id: str,
    timeout_seconds: float,
    concurrency: int,
) -> dict[str, Any]:
    barrier = threading.Barrier(concurrency)

    def worker(request_index: int) -> dict[str, Any]:
        barrier.wait()
        return run_http_request(
            base_url=base_url,
            contract=contract,
            model_id=model_id,
            timeout_seconds=timeout_seconds,
            request_index=request_index,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, request_index + 1) for request_index in range(concurrency)]
        requests = [future.result() for future in futures]

    started_at = min(request["started_at_monotonic"] for request in requests)
    ended_at = max(request["ended_at_monotonic"] for request in requests)
    elapsed_s = ended_at - started_at
    total_completion_tokens = sum(request["completion_tokens"] for request in requests)
    scheduling_classes = sorted(
        {
            value
            for request in requests
            if (value := request["headers"].get("x-psionic-scheduling-class")) is not None
        }
    )
    output = {
        "concurrency": concurrency,
        "elapsed_s": elapsed_s,
        "total_completion_tokens": total_completion_tokens,
        "aggregate_tok_s": (total_completion_tokens / elapsed_s) if total_completion_tokens and elapsed_s else 0.0,
        "mean_wallclock_s": mean([request["wallclock_s"] for request in requests]),
        "mean_ttft_s": mean(
            [request["ttft_s"] for request in requests if request["ttft_s"] is not None]
        ),
        "mean_itl_s": mean(
            [request["itl_s"] for request in requests if request["itl_s"] is not None]
        ),
        "scheduling_classes": scheduling_classes,
        "requests": [
            {
                key: value
                for key, value in request.items()
                if key not in {"started_at_monotonic", "ended_at_monotonic"}
            }
            for request in requests
        ],
    }
    return output


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)


def run_direct_engine(
    *,
    bench_bin: Path,
    repo_root: Path,
    model_path: Path,
    contract: dict[str, Any],
    direct_repeats: int,
    allow_direct_fallbacks: bool,
    report_path: Path,
) -> dict[str, Any]:
    args = [
        str(bench_bin),
        "--backend",
        "psionic",
        "--model-path",
        str(model_path),
        "--prompt",
        contract["user_prompt"],
        "--max-output-tokens",
        str(contract["max_output_tokens"]),
        "--repeats",
        str(direct_repeats),
        "--json-out",
        str(report_path),
        *contract["direct_example_args"],
    ]
    if not allow_direct_fallbacks:
        args.append("--require-fallback-free-cuda")
    completed = subprocess.run(
        args,
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"direct-engine bench failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    return {
        "command": args,
        "stdout_tail": completed.stdout.strip().splitlines()[-12:],
        "report": report,
    }


def run_http_compare(
    *,
    server_bin: Path,
    repo_root: Path,
    model_path: Path,
    contract: dict[str, Any],
    timeout_seconds: float,
    host: str,
    port: int,
    concurrency_ladder: list[int],
    report_path: Path,
) -> dict[str, Any]:
    base_url = f"http://{host}:{port}"
    server_log_path = report_path.with_name(f"{report_path.stem}_server.log")
    with server_log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [
                str(server_bin),
                "-m",
                str(model_path),
                "--backend",
                "cuda",
                "--host",
                host,
                "--port",
                str(port),
            ],
            cwd=str(repo_root),
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        try:
            startup_ready_s = wait_for_server_ready(base_url, timeout_seconds)
            model_id = fetch_model_id(base_url, timeout_seconds)
            warmup = run_http_request(
                base_url=base_url,
                contract=contract,
                model_id=model_id,
                timeout_seconds=timeout_seconds,
                request_index=0,
            )
            concurrency_results = [
                run_http_concurrency_case(
                    base_url=base_url,
                    contract=contract,
                    model_id=model_id,
                    timeout_seconds=timeout_seconds,
                    concurrency=concurrency,
                )
                for concurrency in concurrency_ladder
            ]
            return {
                "server_command": [
                    str(server_bin),
                    "-m",
                    str(model_path),
                    "--backend",
                    "cuda",
                    "--host",
                    host,
                    "--port",
                    str(port),
                ],
                "base_url": base_url,
                "model_id": model_id,
                "startup_ready_s": startup_ready_s,
                "warmup": {
                    key: value
                    for key, value in warmup.items()
                    if key not in {"started_at_monotonic", "ended_at_monotonic"}
                },
                "concurrency_results": concurrency_results,
                "server_log_path": str(server_log_path),
            }
        finally:
            terminate_process(process)


def maybe_run_vllm_direct(
    *,
    model: str,
    contract: dict[str, Any],
    concurrency_ladder: list[int],
    gpu_memory_utilization: float,
    max_model_len: int,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    load_started_at = time.perf_counter()
    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False,
        max_model_len=max_model_len,
    )
    load_s = time.perf_counter() - load_started_at
    sampling_params = SamplingParams(
        temperature=contract["temperature"],
        top_p=contract["top_p"],
        seed=contract["seed"],
        max_tokens=contract["max_output_tokens"],
    )
    warmup_started_at = time.perf_counter()
    llm.generate([contract["user_prompt"]], sampling_params)
    warmup_s = time.perf_counter() - warmup_started_at
    results = []
    for concurrency in concurrency_ladder:
        prompts = [contract["user_prompt"] for _ in range(concurrency)]
        started_at = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed_s = time.perf_counter() - started_at
        total_completion_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        results.append(
            {
                "concurrency": concurrency,
                "elapsed_s": elapsed_s,
                "total_completion_tokens": total_completion_tokens,
                "aggregate_tok_s": (
                    total_completion_tokens / elapsed_s
                )
                if total_completion_tokens and elapsed_s
                else 0.0,
                "avg_tokens_per_request": (
                    total_completion_tokens / concurrency if concurrency else 0.0
                ),
            }
        )
    return {
        "benchmark_class": "optional_reference_direct_engine",
        "model": model,
        "load_s": load_s,
        "warmup_s": warmup_s,
        "metric_availability": {
            "ttft_s": False,
            "itl_s": False,
        },
        "results": results,
    }


def host_metadata() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(args.psionic_root).resolve()
    model_path = Path(args.model_path).resolve()
    report_path = Path(args.report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    contract = prompt_contract(args.contract, args.max_output_tokens)
    concurrency_ladder = parse_concurrency_ladder(args.http_concurrency)
    server_bin = (
        Path(args.server_bin).resolve()
        if args.server_bin
        else repo_root / "target/release/psionic-openai-server"
    )
    bench_bin = (
        Path(args.bench_bin).resolve()
        if args.bench_bin
        else repo_root / "target/release/examples/qwen35_cuda_bench"
    )
    if not model_path.is_file():
        raise SystemExit(f"missing model artifact: {model_path}")
    if not args.skip_build:
        build_artifacts(repo_root)
    if not server_bin.is_file():
        raise SystemExit(f"missing server binary: {server_bin}")
    if not bench_bin.is_file():
        raise SystemExit(f"missing direct bench binary: {bench_bin}")

    direct_report_path = report_path.with_name(f"{report_path.stem}_direct_engine.json")
    direct_engine = run_direct_engine(
        bench_bin=bench_bin,
        repo_root=repo_root,
        model_path=model_path,
        contract=contract,
        direct_repeats=args.direct_repeats,
        allow_direct_fallbacks=args.allow_direct_fallbacks,
        report_path=direct_report_path,
    )

    port = args.port if args.port > 0 else allocate_tcp_port()
    http = run_http_compare(
        server_bin=server_bin,
        repo_root=repo_root,
        model_path=model_path,
        contract=contract,
        timeout_seconds=args.timeout_seconds,
        host=args.host,
        port=port,
        concurrency_ladder=concurrency_ladder,
        report_path=report_path,
    )

    reference = None
    if args.vllm_model:
        reference = maybe_run_vllm_direct(
            model=args.vllm_model,
            contract=contract,
            concurrency_ladder=concurrency_ladder,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
        )

    output = {
        "schema_version": 1,
        "report_kind": "qwen35_direct_vs_http_compare",
        "generated_at_utc": now_utc_iso(),
        "psionic_commit": git_revision(repo_root),
        "host": host_metadata(),
        "prompt_contract": {
            **contract,
            "http_concurrency_ladder": concurrency_ladder,
        },
        "publication_gate": {
            "direct_engine_fallback_free_required": not args.allow_direct_fallbacks,
        },
        "artifacts": {
            "combined_report_path": repo_relative_path(repo_root, report_path),
            "direct_engine_report_path": repo_relative_path(repo_root, direct_report_path),
            "server_log_path": repo_relative_path(repo_root, Path(http["server_log_path"])),
        },
        "direct_engine": {
            "benchmark_class": "direct_engine",
            "command": direct_engine["command"],
            "stdout_tail": direct_engine["stdout_tail"],
            "report": direct_engine["report"],
        },
        "http": {
            "benchmark_class": "http",
            **{
                key: value
                for key, value in http.items()
                if key != "server_log_path"
            },
        },
        "reference": {
            "vllm_direct": reference,
        },
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
