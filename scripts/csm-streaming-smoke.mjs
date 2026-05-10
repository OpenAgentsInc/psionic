#!/usr/bin/env node

const baseUrl = (process.env.PSIONIC_CSM_SMOKE_URL || 'http://34.48.128.199:8081').replace(/\/$/, '')
const minChunks = Number(process.env.PSIONIC_CSM_SMOKE_MIN_CHUNKS || 2)
const timeoutMs = Number(process.env.PSIONIC_CSM_SMOKE_TIMEOUT_MS || 120000)

function assert(condition, message) {
  if (!condition) throw new Error(message)
}

async function main() {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  const startedAt = performance.now()

  try {
    const response = await fetch(`${baseUrl}/v1/audio/speech`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({
        model: 'sesame/csm-1b',
        input:
          'Please produce a longer streaming answer so this smoke can observe multiple generated audio windows before terminal metadata arrives.',
        voice_profile_id: 'openagents/default_female_v1',
        response_format: 'wav',
        stream: true,
        stream_format: 'jsonl_base64',
        psionic_csm: {
          max_audio_length_ms: 2000,
          context_policy: 'prompt_profile_only',
        },
      }),
    })

    assert(response.ok, `streaming request failed with HTTP ${response.status}`)
    assert(response.body, 'streaming response did not include a readable body')

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let audioChunks = 0
    let terminalEvents = 0
    let firstAudioAtMs = null
    let terminalAtMs = null
    let totalAudioBytes = 0

    async function handleLine(line) {
      const trimmed = line.trim()
      if (!trimmed) return
      const event = JSON.parse(trimmed)
      if (event.event === 'audio') {
        audioChunks += 1
        totalAudioBytes += Number(event.chunk_bytes || 0)
        firstAudioAtMs ??= Math.round(performance.now() - startedAt)
        assert(typeof event.data_base64 === 'string' && event.data_base64.length > 0, 'audio event missing base64')
      } else if (event.event === 'terminal') {
        terminalEvents += 1
        terminalAtMs = Math.round(performance.now() - startedAt)
      } else if (event.event === 'error') {
        throw new Error(`streaming response returned error event: ${JSON.stringify(event.error || event)}`)
      }
    }

    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let newlineIndex
      while ((newlineIndex = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newlineIndex)
        buffer = buffer.slice(newlineIndex + 1)
        await handleLine(line)
      }
    }
    buffer += decoder.decode()
    await handleLine(buffer)

    const completedAtMs = Math.round(performance.now() - startedAt)
    assert(audioChunks >= minChunks, `expected at least ${minChunks} audio chunks, got ${audioChunks}`)
    assert(terminalEvents === 1, `expected one terminal event, got ${terminalEvents}`)
    assert(firstAudioAtMs !== null, 'no audio event arrived')
    assert(terminalAtMs !== null, 'no terminal event arrived')
    assert(firstAudioAtMs <= terminalAtMs, 'first audio did not arrive before terminal metadata')
    assert(firstAudioAtMs < completedAtMs, 'first audio did not arrive before response completion')

    console.log(
      JSON.stringify(
        {
          system: 'psionic_csm_speech',
          kind: 'generation_time_streaming_smoke',
          base_url: baseUrl,
          audio_chunks: audioChunks,
          terminal_events: terminalEvents,
          total_audio_bytes: totalAudioBytes,
          first_audio_at_ms: firstAudioAtMs,
          terminal_at_ms: terminalAtMs,
          completed_at_ms: completedAtMs,
        },
        null,
        2,
      ),
    )
  } finally {
    clearTimeout(timer)
  }
}

main().catch((error) => {
  console.error(error.message)
  process.exit(1)
})
