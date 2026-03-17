use std::{
    env,
    io::{self, Write},
    path::PathBuf,
    process::ExitCode,
};

use psionic_mlx_audio::{
    MlxAudioClip, MlxAudioConditioning, MlxAudioReferenceRuntime, MlxAudioSpeechRequest,
    MlxSpeechToSpeechRequest, MlxTextToSpeechRequest,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::Usage(message)) => {
            let _ = writeln!(io::stdout(), "{message}");
            ExitCode::SUCCESS
        }
        Err(CliError::Message(message)) => {
            let _ = writeln!(io::stderr(), "{message}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug)]
enum CliError {
    Usage(String),
    Message(String),
}

fn run() -> Result<(), CliError> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        return Err(CliError::Usage(usage()));
    };
    match command.as_str() {
        "synthesize" => run_synthesize(args),
        "speech-to-speech" => run_speech_to_speech(args),
        "inspect-wav" => run_inspect_wav(args),
        "speech-request" => run_speech_request(args),
        "-h" | "--help" => Err(CliError::Usage(usage())),
        other => Err(CliError::Message(format!(
            "unrecognized subcommand `{other}`\n\n{}",
            usage()
        ))),
    }
}

fn run_synthesize(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_synthesize(args).map_err(CliError::Message)?;
    let report = MlxAudioReferenceRuntime::default()
        .synthesize_text(
            &parsed.family,
            &MlxTextToSpeechRequest {
                request_id: String::from("audio-cli-tts"),
                text: parsed.text,
                conditioning: parsed.voice.map(|voice| MlxAudioConditioning::VoiceLabel { voice }),
                sample_rate_hz: parsed.sample_rate_hz,
                stream_chunk_frames: parsed.stream_chunk_frames,
            },
        )
        .map_err(|error| CliError::Message(format!("synthesis failed: {error}")))?;
    report
        .output_clip
        .save_wav(&parsed.wav_out)
        .map_err(|error| CliError::Message(format!("failed to save wav: {error}")))?;
    write_json_output(&report, parsed.json_out).map_err(CliError::Message)
}

fn run_speech_to_speech(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_speech_to_speech(args).map_err(CliError::Message)?;
    let input_clip = MlxAudioClip::load_wav(&parsed.input_wav)
        .map_err(|error| CliError::Message(format!("failed to load wav: {error}")))?;
    let report = MlxAudioReferenceRuntime::default()
        .synthesize_speech_to_speech(
            &parsed.family,
            &MlxSpeechToSpeechRequest {
                request_id: String::from("audio-cli-s2s"),
                input_clip: input_clip.clone(),
                transcript: parsed.transcript,
                conditioning: parsed.reference_wav.map(|path| {
                    MlxAudioConditioning::ReferenceAudio {
                        clip: MlxAudioClip::load_wav(path).expect("validated ref wav"),
                    }
                }),
                stream_chunk_frames: parsed.stream_chunk_frames,
            },
        )
        .map_err(|error| CliError::Message(format!("speech-to-speech failed: {error}")))?;
    report
        .output_clip
        .save_wav(&parsed.wav_out)
        .map_err(|error| CliError::Message(format!("failed to save wav: {error}")))?;
    write_json_output(&report, parsed.json_out).map_err(CliError::Message)
}

fn run_inspect_wav(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_single_path(args, "--input-wav").map_err(CliError::Message)?;
    let clip = MlxAudioClip::load_wav(&parsed.input)
        .map_err(|error| CliError::Message(format!("failed to load wav: {error}")))?;
    write_json_output(
        &serde_json::json!({
            "sample_rate_hz": clip.sample_rate_hz,
            "channels": clip.channels,
            "frames": clip.frames(),
            "duration_ms": clip.duration_ms(),
            "digest": clip.digest(),
        }),
        parsed.json_out,
    )
    .map_err(CliError::Message)
}

fn run_speech_request(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_speech_request(args).map_err(CliError::Message)?;
    let response = MlxAudioReferenceRuntime::default()
        .handle_speech_request(&MlxAudioSpeechRequest {
            model: parsed.family,
            input: parsed.text,
            voice: parsed.voice,
            response_format: String::from("wav"),
            stream: parsed.stream,
        })
        .map_err(|error| CliError::Message(format!("speech request failed: {error}")))?;
    response
        .clip
        .save_wav(&parsed.wav_out)
        .map_err(|error| CliError::Message(format!("failed to save wav: {error}")))?;
    write_json_output(&response, parsed.json_out).map_err(CliError::Message)
}

#[derive(Clone, Debug)]
struct SynthesizeArgs {
    family: String,
    text: String,
    voice: Option<String>,
    sample_rate_hz: u32,
    stream_chunk_frames: usize,
    wav_out: PathBuf,
    json_out: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct SpeechToSpeechArgs {
    family: String,
    input_wav: PathBuf,
    transcript: Option<String>,
    reference_wav: Option<PathBuf>,
    stream_chunk_frames: usize,
    wav_out: PathBuf,
    json_out: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct SinglePathArgs {
    input: PathBuf,
    json_out: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct SpeechRequestArgs {
    family: String,
    text: String,
    voice: Option<String>,
    stream: bool,
    wav_out: PathBuf,
    json_out: Option<PathBuf>,
}

fn parse_synthesize(args: impl IntoIterator<Item = String>) -> Result<SynthesizeArgs, String> {
    let mut family = String::from("kokoro");
    let mut text = None;
    let mut voice = None;
    let mut sample_rate_hz = 16_000;
    let mut stream_chunk_frames = 1_024usize;
    let mut wav_out = None;
    let mut json_out = None;
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--family" => family = next_value(&mut args, "--family")?,
            "--text" => text = Some(next_value(&mut args, "--text")?),
            "--voice" => voice = Some(next_value(&mut args, "--voice")?),
            "--sample-rate-hz" => {
                sample_rate_hz = next_value(&mut args, "--sample-rate-hz")?
                    .parse()
                    .map_err(|error| format!("invalid --sample-rate-hz value: {error}"))?;
            }
            "--stream-chunk-frames" => {
                stream_chunk_frames = next_value(&mut args, "--stream-chunk-frames")?
                    .parse()
                    .map_err(|error| format!("invalid --stream-chunk-frames value: {error}"))?;
            }
            "--wav-out" => wav_out = Some(PathBuf::from(next_value(&mut args, "--wav-out")?)),
            "--json-out" => json_out = Some(PathBuf::from(next_value(&mut args, "--json-out")?)),
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unrecognized argument `{other}`\n\n{}", usage())),
        }
    }
    let Some(text) = text else {
        return Err(format!("missing required `--text`\n\n{}", usage()));
    };
    let Some(wav_out) = wav_out else {
        return Err(format!("missing required `--wav-out`\n\n{}", usage()));
    };
    Ok(SynthesizeArgs {
        family,
        text,
        voice,
        sample_rate_hz,
        stream_chunk_frames,
        wav_out,
        json_out,
    })
}

fn parse_speech_to_speech(
    args: impl IntoIterator<Item = String>,
) -> Result<SpeechToSpeechArgs, String> {
    let mut family = String::from("xtts");
    let mut input_wav = None;
    let mut transcript = None;
    let mut reference_wav = None;
    let mut stream_chunk_frames = 1_024usize;
    let mut wav_out = None;
    let mut json_out = None;
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--family" => family = next_value(&mut args, "--family")?,
            "--input-wav" => input_wav = Some(PathBuf::from(next_value(&mut args, "--input-wav")?)),
            "--transcript" => transcript = Some(next_value(&mut args, "--transcript")?),
            "--reference-wav" => {
                reference_wav = Some(PathBuf::from(next_value(&mut args, "--reference-wav")?))
            }
            "--stream-chunk-frames" => {
                stream_chunk_frames = next_value(&mut args, "--stream-chunk-frames")?
                    .parse()
                    .map_err(|error| format!("invalid --stream-chunk-frames value: {error}"))?;
            }
            "--wav-out" => wav_out = Some(PathBuf::from(next_value(&mut args, "--wav-out")?)),
            "--json-out" => json_out = Some(PathBuf::from(next_value(&mut args, "--json-out")?)),
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unrecognized argument `{other}`\n\n{}", usage())),
        }
    }
    let Some(input_wav) = input_wav else {
        return Err(format!("missing required `--input-wav`\n\n{}", usage()));
    };
    let Some(wav_out) = wav_out else {
        return Err(format!("missing required `--wav-out`\n\n{}", usage()));
    };
    Ok(SpeechToSpeechArgs {
        family,
        input_wav,
        transcript,
        reference_wav,
        stream_chunk_frames,
        wav_out,
        json_out,
    })
}

fn parse_single_path(
    args: impl IntoIterator<Item = String>,
    flag: &str,
) -> Result<SinglePathArgs, String> {
    let mut input = None;
    let mut json_out = None;
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--input-wav" => input = Some(PathBuf::from(next_value(&mut args, "--input-wav")?)),
            "--json-out" => json_out = Some(PathBuf::from(next_value(&mut args, "--json-out")?)),
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unrecognized argument `{other}`\n\n{}", usage())),
        }
    }
    let Some(input) = input else {
        return Err(format!("missing required `{flag}`\n\n{}", usage()));
    };
    Ok(SinglePathArgs { input, json_out })
}

fn parse_speech_request(args: impl IntoIterator<Item = String>) -> Result<SpeechRequestArgs, String> {
    let mut family = String::from("kokoro");
    let mut text = None;
    let mut voice = None;
    let mut stream = false;
    let mut wav_out = None;
    let mut json_out = None;
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--family" => family = next_value(&mut args, "--family")?,
            "--text" => text = Some(next_value(&mut args, "--text")?),
            "--voice" => voice = Some(next_value(&mut args, "--voice")?),
            "--stream" => stream = true,
            "--wav-out" => wav_out = Some(PathBuf::from(next_value(&mut args, "--wav-out")?)),
            "--json-out" => json_out = Some(PathBuf::from(next_value(&mut args, "--json-out")?)),
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unrecognized argument `{other}`\n\n{}", usage())),
        }
    }
    let Some(text) = text else {
        return Err(format!("missing required `--text`\n\n{}", usage()));
    };
    let Some(wav_out) = wav_out else {
        return Err(format!("missing required `--wav-out`\n\n{}", usage()));
    };
    Ok(SpeechRequestArgs {
        family,
        text,
        voice,
        stream,
        wav_out,
        json_out,
    })
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for `{flag}`"))
}

fn write_json_output<T: serde::Serialize>(
    value: &T,
    json_out: Option<PathBuf>,
) -> Result<(), String> {
    let json = serde_json::to_string_pretty(value)
        .map_err(|error| format!("failed to serialize JSON: {error}"))?;
    match json_out {
        Some(path) => std::fs::write(path, format!("{json}\n"))
            .map_err(|error| format!("failed to write JSON output: {error}"))?,
        None => {
            let mut stdout = io::stdout().lock();
            stdout
                .write_all(json.as_bytes())
                .map_err(|error| format!("failed to write JSON output: {error}"))?;
            stdout
                .write_all(b"\n")
                .map_err(|error| format!("failed to terminate JSON output: {error}"))?;
        }
    }
    Ok(())
}

fn usage() -> String {
    String::from(
        "usage:\n  psionic-mlx-audio synthesize [--family <family>] --text <text> [--voice <label>] [--sample-rate-hz <hz>] [--stream-chunk-frames <n>] --wav-out <path> [--json-out <path>]\n  psionic-mlx-audio speech-to-speech [--family <family>] --input-wav <path> [--transcript <text>] [--reference-wav <path>] [--stream-chunk-frames <n>] --wav-out <path> [--json-out <path>]\n  psionic-mlx-audio inspect-wav --input-wav <path> [--json-out <path>]\n  psionic-mlx-audio speech-request [--family <family>] --text <text> [--voice <label>] [--stream] --wav-out <path> [--json-out <path>]",
    )
}

#[cfg(test)]
mod tests {
    use super::{parse_speech_request, parse_synthesize, usage};

    #[test]
    fn parse_synthesize_requires_text() {
        let error = parse_synthesize(["--wav-out", "/tmp/out.wav"].into_iter().map(String::from))
            .expect_err("missing text");
        assert!(error.contains("missing required `--text`"));
        assert!(error.contains(&usage()));
    }

    #[test]
    fn parse_speech_request_accepts_stream_flag() {
        let parsed = parse_speech_request(
            [
                "--text",
                "hello",
                "--stream",
                "--wav-out",
                "/tmp/out.wav",
            ]
            .into_iter()
            .map(String::from),
        )
        .expect("parsed");
        assert!(parsed.stream);
    }
}
