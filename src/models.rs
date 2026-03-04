use serde::{Deserialize, Serialize};

pub const DEFAULT_MODEL: &str = "claude-opus-4.6";

pub fn supported_models() -> Vec<&'static str> {
    vec![
        "claude-opus-4.6",
        "claude-opus-4.5",
        "claude-sonnet-4.5",
        "gpt-5.2",
        "gpt-5-mini",
        "gpt-4.1",
        "gpt-4o",
        "o3",
    ]
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub model: Option<String>,
    pub stream: Option<bool>,
    pub include_reasoning: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ModelsListResponse {
    pub object: &'static str,
    pub data: Vec<ModelItem>,
}

#[derive(Debug, Serialize)]
pub struct ModelItem {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u8,
    pub message: AssistantMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ----- SSE streaming response types (OpenAI-compatible) -----

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u8,
    pub delta: ChunkDelta,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

// ----- Anthropic Claude Messages API types -----

#[derive(Debug, Clone, Deserialize)]
pub struct ClaudeMessagesRequest {
    pub model: String,
    pub messages: Vec<ClaudeMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default, deserialize_with = "deserialize_claude_system")]
    pub system: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
}

fn default_max_tokens() -> u32 {
    4096
}

fn deserialize_claude_system<'de, D>(d: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;
    use serde_json::Value;

    let v = Option::<Value>::deserialize(d)?;
    match v {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(Value::Array(arr)) => {
            let text = extract_text_from_blocks(&arr);
            if text.is_empty() {
                Ok(None)
            } else {
                Ok(Some(text))
            }
        }
        Some(_) => Err(de::Error::custom("system must be string or array")),
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClaudeMessage {
    pub role: String,
    #[serde(deserialize_with = "deserialize_claude_content")]
    pub content: String,
}

/// Claude content can be a plain string or an array of content blocks.
/// We normalise both to a single string.
fn deserialize_claude_content<'de, D>(d: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;
    use serde_json::Value;

    let v = Value::deserialize(d)?;
    match v {
        Value::String(s) => Ok(s),
        Value::Array(arr) => Ok(extract_text_from_blocks(&arr)),
        Value::Object(obj) => Ok(extract_text_from_blocks(&[Value::Object(obj)])),
        _ => Err(de::Error::custom("content must be string, object, or array")),
    }
}

fn extract_text_from_blocks(arr: &[serde_json::Value]) -> String {
    use serde_json::Value;

    let mut parts = Vec::new();

    for block in arr {
        if let Some(t) = block.get("text").and_then(Value::as_str)
            && !t.is_empty()
        {
            parts.push(t.to_string());
            continue;
        }

        if let Some("tool_result") = block.get("type").and_then(Value::as_str)
            && let Some(content) = block.get("content")
        {
            match content {
                Value::String(s) if !s.is_empty() => parts.push(s.clone()),
                Value::Array(nested) => {
                    let nested_text = extract_text_from_blocks(nested);
                    if !nested_text.is_empty() {
                        parts.push(nested_text);
                    }
                }
                Value::Object(obj) => {
                    if let Some(text) = obj.get("text").and_then(Value::as_str)
                        && !text.is_empty()
                    {
                        parts.push(text.to_string());
                    }
                }
                _ => {}
            }
            continue;
        }

        if let Some("tool_use") = block.get("type").and_then(Value::as_str) {
            continue;
        }
    }

    parts.join("\n")
}

#[derive(Debug, Serialize)]
pub struct ClaudeMessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub role: &'static str,
    pub content: Vec<ClaudeContentBlock>,
    pub model: String,
    pub stop_reason: &'static str,
    pub usage: ClaudeUsage,
}

#[derive(Debug, Serialize)]
pub struct ClaudeContentBlock {
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct ClaudeUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ----- Claude streaming event types -----

#[derive(Debug, Serialize)]
pub struct ClaudeStreamMessageStart {
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub message: ClaudeStreamMessageMeta,
}

#[derive(Debug, Serialize)]
pub struct ClaudeStreamMessageMeta {
    pub id: String,
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub role: &'static str,
    pub content: Vec<()>,
    pub model: String,
    pub stop_reason: Option<&'static str>,
    pub usage: ClaudeUsage,
}

#[derive(Debug, Serialize)]
pub struct ClaudeStreamContentBlockStart {
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub index: u8,
    pub content_block: ClaudeContentBlock,
}

#[derive(Debug, Serialize)]
pub struct ClaudeStreamContentBlockDelta {
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub index: u8,
    pub delta: ClaudeTextDelta,
}

#[derive(Debug, Serialize)]
pub struct ClaudeTextDelta {
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct ClaudeStreamContentBlockStop {
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub index: u8,
}

#[derive(Debug, Serialize)]
pub struct ClaudeStreamMessageDelta {
    #[serde(rename = "type")]
    pub type_field: &'static str,
    pub delta: ClaudeStopDelta,
    pub usage: ClaudeUsage,
}

#[derive(Debug, Serialize)]
pub struct ClaudeStopDelta {
    pub stop_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ClaudeStreamMessageStop {
    #[serde(rename = "type")]
    pub type_field: &'static str,
}

#[cfg(test)]
mod tests {
    use super::ClaudeMessage;

    #[test]
    fn claude_content_ignores_tool_use_blocks() {
        let msg: ClaudeMessage = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "shell",
                    "input": { "cmd": "pwd" }
                }
            ]
        }))
        .expect("should deserialize ClaudeMessage");

        assert!(msg.content.is_empty(), "tool_use should not be forwarded as prompt text");
    }

    #[test]
    fn claude_content_keeps_only_tool_result_text() {
        let msg: ClaudeMessage = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": {
                        "text": "/home/nonewhite"
                    }
                }
            ]
        }))
        .expect("should deserialize ClaudeMessage");

        assert_eq!(msg.content, "/home/nonewhite");
    }

    #[test]
    fn claude_content_drops_tool_result_object_without_text() {
        let msg: ClaudeMessage = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": {
                        "exit_code": 0,
                        "stdout": "ok"
                    }
                }
            ]
        }))
        .expect("should deserialize ClaudeMessage");

        assert!(
            msg.content.is_empty(),
            "non-text tool_result metadata should not be forwarded"
        );
    }
}
