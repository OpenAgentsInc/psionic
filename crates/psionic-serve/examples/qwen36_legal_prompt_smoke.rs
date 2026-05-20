use std::{env, error::Error, io, path::PathBuf};

use psionic_models::{
    run_qwen36_legal_prompt_smoke, write_qwen36_27b_smoke_safetensors, QWEN36_27B_MODEL_ID,
    QWEN36_27B_SMOKE_CONFIG_PATH, QWEN36_27B_SMOKE_SHARD_PATH, QWEN36_27B_SMOKE_TOKENIZER_PATH,
};

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().collect::<Vec<_>>();
    let model =
        optional_flag(&args, "--model").unwrap_or_else(|| String::from(QWEN36_27B_MODEL_ID));
    let prompt = required_flag(&args, "--prompt")?;
    let config = optional_flag(&args, "--config")
        .unwrap_or_else(|| String::from(QWEN36_27B_SMOKE_CONFIG_PATH));
    let tokenizer = optional_flag(&args, "--tokenizer")
        .unwrap_or_else(|| String::from(QWEN36_27B_SMOKE_TOKENIZER_PATH));
    let shard = optional_flag(&args, "--shard")
        .unwrap_or_else(|| String::from(QWEN36_27B_SMOKE_SHARD_PATH));
    let shard_path = PathBuf::from(&shard);
    if !shard_path.exists() {
        write_qwen36_27b_smoke_safetensors(&shard_path)?;
    }

    let report = run_qwen36_legal_prompt_smoke(
        model.as_str(),
        PathBuf::from(prompt),
        PathBuf::from(config),
        PathBuf::from(tokenizer),
        &[shard_path],
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn required_flag(args: &[String], flag: &str) -> Result<String, io::Error> {
    optional_flag(args, flag).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: qwen36_legal_prompt_smoke --model Qwen3.6-27B --prompt fixtures/legal/smoke.prompt [--config path] [--tokenizer path] [--shard path]",
        )
    })
}

fn optional_flag(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find_map(|window| (window[0] == flag).then(|| window[1].clone()))
}
