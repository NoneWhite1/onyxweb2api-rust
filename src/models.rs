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

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub model: Option<String>,
    pub stream: Option<bool>,
    pub include_reasoning: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<OpenAIToolDefinition>>,
    #[serde(default)]
    pub tool_choice: Option<OpenAIToolChoice>,
}

impl ChatCompletionRequest {
    pub fn requested_tool_names(&self) -> Vec<String> {
        self.tools
            .as_ref()
            .map(|tools| {
                tools
                    .iter()
                    .filter_map(|tool| {
                        if let Some(function) = &tool.function {
                            non_empty_trimmed(function.name.as_str())
                        } else if let Some(name) = &tool.name {
                            non_empty_trimmed(name)
                        } else if let Some(tool_type) = &tool.r#type {
                            openai_tool_type_alias(tool_type)
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn forced_tool_name(&self) -> Option<String> {
        let choice = self.tool_choice.as_ref()?;
        match choice {
            OpenAIToolChoice::String(_) => None,
            OpenAIToolChoice::Object(obj) => {
                if let Some(function) = &obj.function {
                    non_empty_trimmed(function.name.as_str())
                } else if let Some(name) = &obj.name {
                    non_empty_trimmed(name)
                } else if let Some(tool_type) = &obj.r#type {
                    openai_tool_type_alias(tool_type)
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIToolDefinition {
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub function: Option<OpenAIFunctionDefinition>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIFunctionDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum OpenAIToolChoice {
    String(String),
    Object(OpenAIToolChoiceObject),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIToolChoiceObject {
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub function: Option<OpenAIFunctionChoice>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIFunctionChoice {
    pub name: String,
}

fn non_empty_trimmed(input: &str) -> Option<String> {
    let value = input.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn openai_tool_type_alias(tool_type: &str) -> Option<String> {
    let normalized = tool_type.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "code_interpreter" | "python" | "python_tool" => Some("python".to_string()),
        "open_url" | "openurl" | "open_url_tool" | "openurltool" => {
            Some("open_url".to_string())
        }
        _ => non_empty_trimmed(tool_type),
    }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<AssistantToolCall>>,
}

#[derive(Debug, Serialize)]
pub struct AssistantToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: AssistantToolCallFunction,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct AssistantToolCallFunction {
    pub name: String,
    pub arguments: String,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<AssistantToolCall>>,
}

// ----- Anthropic Claude Messages API types -----

#[derive(Debug, Clone, Deserialize, Serialize)]
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
    #[serde(default)]
    pub tools: Option<Vec<ClaudeToolDefinition>>,
    #[serde(default)]
    pub tool_choice: Option<ClaudeToolChoice>,
}

impl ClaudeMessagesRequest {
    pub fn requested_tool_names(&self) -> Vec<String> {
        self.tools
            .as_ref()
            .map(|tools| {
                tools
                    .iter()
                    .filter_map(|tool| non_empty_trimmed(tool.name.as_str()))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn forced_tool_name(&self) -> Option<String> {
        let choice = self.tool_choice.as_ref()?;
        match choice {
            ClaudeToolChoice::String(value) => {
                let lower = value.trim().to_ascii_lowercase();
                if lower == "auto" || lower == "any" || lower == "none" || lower == "required" {
                    None
                } else {
                    non_empty_trimmed(value)
                }
            }
            ClaudeToolChoice::Object(choice_obj) => {
                let kind = choice_obj
                    .type_field
                    .as_deref()
                    .map(str::trim)
                    .map(str::to_ascii_lowercase)
                    .unwrap_or_default();

                if kind.is_empty() || kind == "tool" {
                    choice_obj.name.as_ref().and_then(|name| non_empty_trimmed(name))
                } else {
                    None
                }
            }
        }
    }

    pub fn has_tool_result_message(&self) -> bool {
        self.messages.iter().any(|message| message.has_tool_result)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClaudeToolDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ClaudeToolChoice {
    String(String),
    Object(ClaudeToolChoiceObject),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClaudeToolChoiceObject {
    #[serde(default, rename = "type")]
    pub type_field: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
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

#[derive(Debug, Clone, Serialize)]
pub struct ClaudeMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing)]
    pub has_tool_result: bool,
}

impl<'de> Deserialize<'de> for ClaudeMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de;
        use serde_json::Value;

        #[derive(Deserialize)]
        struct RawClaudeMessage {
            role: String,
            content: Value,
        }

        let raw = RawClaudeMessage::deserialize(deserializer)?;
        let (content, has_tool_result) = match raw.content {
            Value::String(s) => (s, false),
            Value::Array(arr) => extract_text_and_flags_from_blocks(&arr),
            Value::Object(obj) => extract_text_and_flags_from_blocks(&[Value::Object(obj)]),
            _ => {
                return Err(de::Error::custom(
                    "content must be string, object, or array",
                ));
            }
        };

        Ok(Self {
            role: raw.role,
            content,
            has_tool_result,
        })
    }
}

fn extract_text_from_blocks(arr: &[serde_json::Value]) -> String {
    extract_text_and_flags_from_blocks(arr).0
}

fn extract_text_and_flags_from_blocks(arr: &[serde_json::Value]) -> (String, bool) {
    use serde_json::Value;

    let mut parts = Vec::new();
    let mut has_tool_result = false;

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
            has_tool_result = true;
            match content {
                Value::String(s) if !s.is_empty() => parts.push(s.clone()),
                Value::Array(nested) => {
                    let (nested_text, nested_has_tool_result) = extract_text_and_flags_from_blocks(nested);
                    has_tool_result |= nested_has_tool_result;
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

    (parts.join("\n"), has_tool_result)
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
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
    use super::{ChatCompletionRequest, ClaudeMessage, ClaudeMessagesRequest};

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
        assert!(
            !msg.has_tool_result,
            "tool_use-only content should not mark tool_result presence"
        );
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
        assert!(
            msg.has_tool_result,
            "tool_result content should mark tool_result presence"
        );
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
        assert!(
            msg.has_tool_result,
            "tool_result blocks should be tracked even if text payload is empty"
        );
    }

    #[test]
    fn claude_request_extracts_tools_and_forced_tool_name() {
        let req: ClaudeMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "claude-opus-4.6",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1024,
            "tools": [
                {"name": "shell", "description": "Run shell", "input_schema": {"type": "object"}},
                {"name": "web_search", "description": "Search the web", "input_schema": {"type": "object"}}
            ],
            "tool_choice": {"type": "tool", "name": "shell"}
        }))
        .expect("should deserialize ClaudeMessagesRequest");

        assert_eq!(req.requested_tool_names(), vec!["shell", "web_search"]);
        assert_eq!(req.forced_tool_name().as_deref(), Some("shell"));
        assert!(!req.has_tool_result_message());
    }

    #[test]
    fn claude_request_detects_tool_result_message_blocks() {
        let req: ClaudeMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "claude-opus-4.6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "ok"
                        }
                    ]
                }
            ],
            "max_tokens": 1024,
            "tools": [
                {"name": "bash", "description": "Run shell", "input_schema": {"type": "object"}}
            ]
        }))
        .expect("should deserialize ClaudeMessagesRequest");

        assert!(req.has_tool_result_message());
    }

    #[test]
    fn chat_completions_request_extracts_openai_forced_tool_name() {
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "description": "Run shell",
                        "parameters": {"type": "object"}
                    }
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "shell"}
            }
        }))
        .expect("should deserialize ChatCompletionRequest");

        assert_eq!(req.requested_tool_names(), vec!["shell"]);
        assert_eq!(req.forced_tool_name().as_deref(), Some("shell"));
    }

    #[test]
    fn chat_completions_request_maps_code_interpreter_type_to_python() {
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "run code"}],
            "tools": [
                {"type": "code_interpreter"}
            ],
            "tool_choice": {"type": "code_interpreter"}
        }))
        .expect("should deserialize ChatCompletionRequest");

        assert_eq!(req.requested_tool_names(), vec!["python"]);
        assert_eq!(req.forced_tool_name().as_deref(), Some("python"));
    }
}
