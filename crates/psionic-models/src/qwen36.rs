use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokenizers::Tokenizer;

use crate::{PromptMessage, PromptMessageRole};

pub const QWEN36_TEMPLATE_ID: &str = "qwen3.6.chat_template.v1";
pub const QWEN36_EMPTY_THINK_BLOCK: &str = "<think>\n\n</think>";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen36ReasoningMode {
    Thinking,
    DirectAnswer,
    MixedExplicit,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Qwen36PromptOptions {
    pub reasoning_mode: Qwen36ReasoningMode,
    pub add_generation_prompt: bool,
    pub emit_empty_think_block: bool,
}

impl Default for Qwen36PromptOptions {
    fn default() -> Self {
        Self {
            reasoning_mode: Qwen36ReasoningMode::DirectAnswer,
            add_generation_prompt: true,
            emit_empty_think_block: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Qwen36RenderedPrompt {
    pub template_id: String,
    pub reasoning_mode: Qwen36ReasoningMode,
    pub text: String,
    pub prompt_hash: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_ids: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Qwen36PromptReceipt {
    pub template_id: String,
    pub reasoning_mode: Qwen36ReasoningMode,
    pub prompt_hash: String,
    pub token_count: usize,
}

impl From<&Qwen36RenderedPrompt> for Qwen36PromptReceipt {
    fn from(prompt: &Qwen36RenderedPrompt) -> Self {
        Self {
            template_id: prompt.template_id.clone(),
            reasoning_mode: prompt.reasoning_mode,
            prompt_hash: prompt.prompt_hash.clone(),
            token_count: prompt.token_ids.len(),
        }
    }
}

#[derive(Debug, Error)]
pub enum Qwen36TemplateError {
    #[error("failed to load Qwen3.6 tokenizer: {0}")]
    TokenizerLoad(String),
    #[error("failed to tokenize Qwen3.6 prompt: {0}")]
    TokenizerEncode(String),
    #[error("invalid Qwen3.6 chat messages: {0}")]
    InvalidMessages(String),
}

pub struct Qwen36PromptRenderer {
    tokenizer: Option<Tokenizer>,
}

impl Qwen36PromptRenderer {
    pub fn without_tokenizer() -> Self {
        Self { tokenizer: None }
    }

    pub fn from_tokenizer_file(path: impl AsRef<Path>) -> Result<Self, Qwen36TemplateError> {
        let tokenizer = Tokenizer::from_file(path.as_ref())
            .map_err(|error| Qwen36TemplateError::TokenizerLoad(error.to_string()))?;
        Ok(Self {
            tokenizer: Some(tokenizer),
        })
    }

    pub fn from_tokenizer_json_bytes(bytes: &[u8]) -> Result<Self, Qwen36TemplateError> {
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|error| Qwen36TemplateError::TokenizerLoad(error.to_string()))?;
        Ok(Self {
            tokenizer: Some(tokenizer),
        })
    }

    pub fn render(
        &self,
        messages: &[PromptMessage],
        options: &Qwen36PromptOptions,
    ) -> Result<Qwen36RenderedPrompt, Qwen36TemplateError> {
        let text = render_qwen36_prompt_text(messages, options)?;
        let prompt_hash = qwen36_prompt_hash(text.as_str(), options.reasoning_mode);
        let token_ids = if let Some(tokenizer) = &self.tokenizer {
            tokenizer
                .encode(text.as_str(), false)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|error| Qwen36TemplateError::TokenizerEncode(error.to_string()))?
        } else {
            Vec::new()
        };
        Ok(Qwen36RenderedPrompt {
            template_id: String::from(QWEN36_TEMPLATE_ID),
            reasoning_mode: options.reasoning_mode,
            text,
            prompt_hash,
            token_ids,
        })
    }
}

pub fn render_qwen36_prompt_text(
    messages: &[PromptMessage],
    options: &Qwen36PromptOptions,
) -> Result<String, Qwen36TemplateError> {
    if messages.is_empty() {
        return Err(Qwen36TemplateError::InvalidMessages(String::from(
            "Qwen3.6 prompt rendering requires at least one message",
        )));
    }
    let mut rendered = String::new();
    let mut saw_non_instruction = false;
    let mut index = 0usize;
    while index < messages.len() {
        let message = &messages[index];
        match message.role {
            PromptMessageRole::System | PromptMessageRole::Developer => {
                if saw_non_instruction {
                    return Err(Qwen36TemplateError::InvalidMessages(String::from(
                        "Qwen3.6 system/developer messages must precede user, assistant, and tool messages",
                    )));
                }
                rendered.push_str("<|im_start|>system\n");
                rendered.push_str(message.content.trim());
                rendered.push_str("<|im_end|>\n");
            }
            PromptMessageRole::User => {
                saw_non_instruction = true;
                rendered.push_str("<|im_start|>user\n");
                rendered.push_str(message.content.trim());
                rendered.push_str("<|im_end|>\n");
            }
            PromptMessageRole::Assistant => {
                saw_non_instruction = true;
                rendered.push_str("<|im_start|>assistant\n");
                rendered
                    .push_str(qwen36_assistant_content(message, options.reasoning_mode)?.as_str());
                rendered.push_str("<|im_end|>\n");
            }
            PromptMessageRole::Tool => {
                saw_non_instruction = true;
                let start = index;
                while index < messages.len() && messages[index].role == PromptMessageRole::Tool {
                    index += 1;
                }
                rendered.push_str("<|im_start|>user\n");
                for tool_message in &messages[start..index] {
                    rendered.push_str("<tool_response>");
                    if let Some(author) = tool_message.author_name.as_deref() {
                        rendered.push_str("\nname: ");
                        rendered.push_str(author);
                    }
                    rendered.push('\n');
                    rendered.push_str(tool_message.content.trim());
                    rendered.push_str("\n</tool_response>\n");
                }
                rendered.push_str("<|im_end|>\n");
                continue;
            }
        }
        index += 1;
    }
    if options.add_generation_prompt {
        rendered.push_str("<|im_start|>assistant\n");
        if options.reasoning_mode == Qwen36ReasoningMode::DirectAnswer
            && options.emit_empty_think_block
        {
            rendered.push_str(QWEN36_EMPTY_THINK_BLOCK);
            rendered.push_str("\n\n");
        }
    }
    Ok(rendered)
}

pub fn qwen36_prompt_hash(text: &str, reasoning_mode: Qwen36ReasoningMode) -> String {
    let mut hasher = Sha256::new();
    hasher.update(QWEN36_TEMPLATE_ID.as_bytes());
    hasher.update(b"|");
    hasher.update(format!("{reasoning_mode:?}").as_bytes());
    hasher.update(b"|");
    hasher.update(text.as_bytes());
    hex::encode(hasher.finalize())
}

fn qwen36_assistant_content(
    message: &PromptMessage,
    mode: Qwen36ReasoningMode,
) -> Result<String, Qwen36TemplateError> {
    if message.content.contains("/think") || message.content.contains("/nothink") {
        return Err(Qwen36TemplateError::InvalidMessages(String::from(
            "Qwen3.6 renderer does not accept /think or /nothink control tokens",
        )));
    }
    let answer_content = strip_qwen36_think_block(message.content.trim());
    let reasoning = message
        .reasoning_content
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    Ok(match mode {
        Qwen36ReasoningMode::Thinking => {
            let mut content = String::new();
            content.push_str("<think>\n");
            content.push_str(reasoning.unwrap_or(""));
            content.push_str("\n</think>\n\n");
            content.push_str(answer_content.trim());
            content
        }
        Qwen36ReasoningMode::DirectAnswer => answer_content.trim().to_string(),
        Qwen36ReasoningMode::MixedExplicit => {
            if let Some(reasoning) = reasoning {
                let mut content = String::new();
                content.push_str("<think>\n");
                content.push_str(reasoning);
                content.push_str("\n</think>\n\n");
                content.push_str(answer_content.trim());
                content
            } else {
                answer_content.trim().to_string()
            }
        }
    })
}

fn strip_qwen36_think_block(content: &str) -> String {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("<think>") {
        return content.to_string();
    }
    if let Some(end) = trimmed.find("</think>") {
        return trimmed[end + "</think>".len()..].trim_start().to_string();
    }
    content.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PromptMessage, PromptMessageRole};

    fn messages() -> Vec<PromptMessage> {
        vec![
            PromptMessage::new(PromptMessageRole::System, "Use tools carefully."),
            PromptMessage::new(PromptMessageRole::User, "Write memo.md."),
            PromptMessage::new(PromptMessageRole::Tool, "{\"ok\":true}")
                .with_author_name("validate_deliverables"),
            PromptMessage::new(PromptMessageRole::Assistant, "Done.")
                .with_reasoning_content("Need to verify memo.md first."),
        ]
    }

    #[test]
    fn qwen36_template_direct_answer_is_stable_and_has_no_fake_reasoning() {
        let renderer = Qwen36PromptRenderer::without_tokenizer();
        let options = Qwen36PromptOptions {
            reasoning_mode: Qwen36ReasoningMode::DirectAnswer,
            add_generation_prompt: true,
            emit_empty_think_block: false,
        };
        let first = renderer.render(&messages(), &options).expect("render");
        let second = renderer.render(&messages(), &options).expect("render");

        assert_eq!(first.text, second.text);
        assert_eq!(first.prompt_hash, second.prompt_hash);
        assert!(!first.text.contains("/think"));
        assert!(!first.text.contains("/nothink"));
        assert!(!first.text.contains("Need to verify memo.md first."));
        assert!(!first.text.contains("<think>"));
        assert!(first.text.contains("<tool_response>"));
    }

    #[test]
    fn qwen36_template_thinking_mode_renders_explicit_reasoning() {
        let renderer = Qwen36PromptRenderer::without_tokenizer();
        let options = Qwen36PromptOptions {
            reasoning_mode: Qwen36ReasoningMode::Thinking,
            add_generation_prompt: false,
            emit_empty_think_block: false,
        };
        let rendered = renderer.render(&messages(), &options).expect("render");

        assert!(
            rendered
                .text
                .contains("<think>\nNeed to verify memo.md first.\n</think>")
        );
        assert!(rendered.text.contains("Done."));
    }

    #[test]
    fn qwen36_template_direct_generation_can_emit_empty_think_block() {
        let renderer = Qwen36PromptRenderer::without_tokenizer();
        let options = Qwen36PromptOptions {
            reasoning_mode: Qwen36ReasoningMode::DirectAnswer,
            add_generation_prompt: true,
            emit_empty_think_block: true,
        };
        let rendered = renderer
            .render(
                &[PromptMessage::new(
                    PromptMessageRole::User,
                    "Answer directly.",
                )],
                &options,
            )
            .expect("render");

        assert!(
            rendered
                .text
                .ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        );
    }

    #[test]
    fn qwen36_template_rejects_old_soft_switch_tokens() {
        let renderer = Qwen36PromptRenderer::without_tokenizer();
        let error = renderer
            .render(
                &[PromptMessage::new(
                    PromptMessageRole::Assistant,
                    "/nothink answer",
                )],
                &Qwen36PromptOptions::default(),
            )
            .expect_err("soft switch tokens are rejected");
        assert!(error.to_string().contains("/think or /nothink"));
    }
}
