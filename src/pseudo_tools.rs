use serde_json::Value;

use crate::models::{
    ChatCompletionRequest, ChatMessage, ClaudeMessagesRequest, ClaudeToolChoice,
    ClaudeToolDefinition, OpenAIToolChoice, OpenAIToolChoiceObject, OpenAIToolDefinition,
};

pub const MAX_PROTOCOL_RETRIES: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoiceMode {
    Auto,
    None,
    Required,
    Specific(String),
}

#[derive(Debug, Clone)]
pub struct NormalizedTool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct ToolPromptContext {
    pub tools: Vec<NormalizedTool>,
    pub choice: ToolChoiceMode,
}

impl ToolPromptContext {
    pub fn is_active(&self) -> bool {
        !matches!(self.choice, ToolChoiceMode::None) && !self.tools.is_empty()
    }

    pub fn tool_names(&self) -> Vec<String> {
        self.tools.iter().map(|tool| tool.name.clone()).collect()
    }
}

#[derive(Debug, Clone)]
pub enum ParsedPseudoToolResponse {
    Final {
        content: String,
    },
    Action {
        tool_name: String,
        action_input: Value,
    },
}

#[derive(Debug, Clone)]
pub struct ValidationFailure {
    pub code: &'static str,
    pub message: String,
}

pub fn context_from_openai_request(request: &ChatCompletionRequest) -> ToolPromptContext {
    let tools = request
        .tools
        .as_ref()
        .map(|tools| {
            tools
                .iter()
                .filter_map(normalize_openai_tool)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let choice = request
        .tool_choice
        .as_ref()
        .map(normalize_openai_tool_choice)
        .unwrap_or_else(|| {
            if tools.is_empty() {
                ToolChoiceMode::None
            } else {
                ToolChoiceMode::Auto
            }
        });

    ToolPromptContext { tools, choice }
}

pub fn context_from_claude_request(request: &ClaudeMessagesRequest) -> ToolPromptContext {
    let tools = request
        .tools
        .as_ref()
        .map(|tools| tools.iter().map(normalize_claude_tool).collect::<Vec<_>>())
        .unwrap_or_default();

    let choice = request
        .tool_choice
        .as_ref()
        .map(normalize_claude_tool_choice)
        .unwrap_or_else(|| {
            if tools.is_empty() {
                ToolChoiceMode::None
            } else {
                ToolChoiceMode::Auto
            }
        });

    ToolPromptContext { tools, choice }
}

pub fn should_enable_openai_protocol(
    request: &ChatCompletionRequest,
    context: &ToolPromptContext,
) -> bool {
    request.has_tool_result_message()
        || request.has_assistant_tool_call_message()
        || (!context.tools.is_empty() && !matches!(context.choice, ToolChoiceMode::None))
}

pub fn should_enable_claude_protocol(
    request: &ClaudeMessagesRequest,
    context: &ToolPromptContext,
) -> bool {
    request.has_tool_result_message()
        || (!context.tools.is_empty() && !matches!(context.choice, ToolChoiceMode::None))
}

pub fn prepend_protocol_messages(
    base_messages: &[ChatMessage],
    context: &ToolPromptContext,
    retry_failure: Option<&ValidationFailure>,
) -> Vec<ChatMessage> {
    let mut prompt_messages = Vec::new();

    if let Some(retry_failure) = retry_failure {
        prompt_messages.push(ChatMessage::system(build_retry_prompt(retry_failure)));
    }

    prompt_messages.push(ChatMessage::system(build_protocol_prompt(context)));
    prompt_messages.extend(base_messages.iter().cloned());
    prompt_messages
}

pub fn parse_pseudo_tool_response(
    raw_text: &str,
    context: &ToolPromptContext,
) -> Result<ParsedPseudoToolResponse, ValidationFailure> {
    let trimmed = raw_text.trim();
    if trimmed.is_empty() {
        return Err(ValidationFailure {
            code: "EMPTY_OUTPUT",
            message: "model returned empty output".to_string(),
        });
    }

    if trimmed.contains("```") {
        return Err(ValidationFailure {
            code: "DIRTY_JSON",
            message: "response contained markdown fences".to_string(),
        });
    }

    let value: Value = serde_json::from_str(trimmed).map_err(|err| ValidationFailure {
        code: "NON_JSON",
        message: format!("response was not a single valid JSON object: {err}"),
    })?;

    let Value::Object(object) = value else {
        return Err(ValidationFailure {
            code: "NON_OBJECT_JSON",
            message: "response JSON must be an object".to_string(),
        });
    };

    let has_final = object.contains_key("final");
    let has_action = object.contains_key("action");

    match (has_final, has_action) {
        (true, false) => parse_final_response(&object),
        (false, true) => parse_action_response(&object, context),
        (true, true) => Err(ValidationFailure {
            code: "AMBIGUOUS_OUTPUT",
            message: "response cannot contain both 'final' and 'action'".to_string(),
        }),
        (false, false) => Err(ValidationFailure {
            code: "MISSING_FIELDS",
            message: "response must contain either 'final' or 'action'".to_string(),
        }),
    }
}

pub fn normalize_openai_messages(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    messages
        .iter()
        .map(|message| {
            let content = render_openai_message_for_prompt(message);
            ChatMessage {
                role: message.role.clone(),
                content,
                tool_call_id: None,
                name: None,
                tool_calls: None,
            }
        })
        .collect()
}

fn normalize_openai_tool(tool: &OpenAIToolDefinition) -> Option<NormalizedTool> {
    if let Some(function) = &tool.function {
        return Some(NormalizedTool {
            name: function.name.trim().to_string(),
            description: function.description.clone(),
            input_schema: function.parameters.clone(),
        });
    }

    tool.name.as_ref().map(|name| NormalizedTool {
        name: name.trim().to_string(),
        description: None,
        input_schema: None,
    })
}

fn normalize_claude_tool(tool: &ClaudeToolDefinition) -> NormalizedTool {
    NormalizedTool {
        name: tool.name.trim().to_string(),
        description: tool.description.clone(),
        input_schema: tool.input_schema.clone(),
    }
}

fn normalize_openai_tool_choice(choice: &OpenAIToolChoice) -> ToolChoiceMode {
    match choice {
        OpenAIToolChoice::String(value) => match value.trim().to_ascii_lowercase().as_str() {
            "none" => ToolChoiceMode::None,
            "required" => ToolChoiceMode::Required,
            _ => ToolChoiceMode::Auto,
        },
        OpenAIToolChoice::Object(obj) => normalize_openai_tool_choice_object(obj),
    }
}

fn normalize_openai_tool_choice_object(choice: &OpenAIToolChoiceObject) -> ToolChoiceMode {
    if let Some(function) = &choice.function {
        return ToolChoiceMode::Specific(function.name.trim().to_string());
    }

    if let Some(name) = &choice.name {
        return ToolChoiceMode::Specific(name.trim().to_string());
    }

    match choice
        .r#type
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("none") => ToolChoiceMode::None,
        Some("required") => ToolChoiceMode::Required,
        _ => ToolChoiceMode::Auto,
    }
}

fn normalize_claude_tool_choice(choice: &ClaudeToolChoice) -> ToolChoiceMode {
    match choice {
        ClaudeToolChoice::String(value) => match value.trim().to_ascii_lowercase().as_str() {
            "none" => ToolChoiceMode::None,
            "any" | "required" => ToolChoiceMode::Required,
            _ => ToolChoiceMode::Auto,
        },
        ClaudeToolChoice::Object(object) => {
            if object
                .type_field
                .as_deref()
                .map(str::trim)
                .map(str::to_ascii_lowercase)
                .as_deref()
                == Some("tool")
            {
                if let Some(name) = &object.name {
                    return ToolChoiceMode::Specific(name.trim().to_string());
                }
            }
            ToolChoiceMode::Auto
        }
    }
}

fn build_protocol_prompt(context: &ToolPromptContext) -> String {
    let mut prompt = String::from(
        "You are operating under a strict tool protocol. You must output exactly one JSON object and nothing else. Do not use markdown fences, prose, prefixes, suffixes, or explanations.\n\n",
    );

    prompt.push_str("Allowed response shapes:\n");
    prompt.push_str("1. {\"action\":\"tool_name\",\"action_input\":{...}}\n");
    prompt.push_str("2. {\"final\":\"final answer text\"}\n\n");

    prompt.push_str("Tool choice policy: ");
    match &context.choice {
        ToolChoiceMode::Auto => {
            prompt.push_str("You may either call one tool or return a final answer.\n")
        }
        ToolChoiceMode::None => {
            prompt.push_str("Do not call any tool. You must return a final answer.\n")
        }
        ToolChoiceMode::Required => {
            prompt.push_str("You must call exactly one tool in this response.\n")
        }
        ToolChoiceMode::Specific(name) => prompt.push_str(&format!(
            "You must call the tool named '{name}' in this response.\n"
        )),
    }

    if context.tools.is_empty() {
        prompt.push_str("No tools are available for this request.\n");
    } else {
        prompt.push_str("Available tools:\n");
        for tool in &context.tools {
            prompt.push_str(&format!("- {}\n", tool.name));
            if let Some(description) = &tool.description
                && !description.trim().is_empty()
            {
                prompt.push_str(&format!("  description: {}\n", description.trim()));
            }
            if let Some(schema) = &tool.input_schema {
                let schema_text = serde_json::to_string_pretty(schema)
                    .unwrap_or_else(|_| schema.to_string())
                    .replace('\n', "\n  ");
                prompt.push_str(&format!("  input_schema: {}\n", schema_text));
            }
        }
    }

    prompt.push_str("\nIf prior tool results are present in the conversation, use them. After receiving a tool result, either call another tool or return {\"final\":...}. Never repeat a previous invalid output.\n");
    prompt
}

fn build_retry_prompt(failure: &ValidationFailure) -> String {
    format!(
        "Your previous response was discarded. Failure code: {}. Failure reason: {}. Retry now and return exactly one valid JSON object with no extra text.",
        failure.code, failure.message
    )
}

fn parse_final_response(
    object: &serde_json::Map<String, Value>,
) -> Result<ParsedPseudoToolResponse, ValidationFailure> {
    if object.len() != 1 {
        return Err(ValidationFailure {
            code: "DIRTY_JSON",
            message: "final response may only contain the 'final' field".to_string(),
        });
    }

    let Some(final_text) = object.get("final").and_then(Value::as_str) else {
        return Err(ValidationFailure {
            code: "INVALID_FINAL",
            message: "final must be a string".to_string(),
        });
    };

    let final_text = final_text.trim();
    if final_text.is_empty() {
        return Err(ValidationFailure {
            code: "EMPTY_FINAL",
            message: "final must not be empty".to_string(),
        });
    }

    Ok(ParsedPseudoToolResponse::Final {
        content: final_text.to_string(),
    })
}

fn parse_action_response(
    object: &serde_json::Map<String, Value>,
    context: &ToolPromptContext,
) -> Result<ParsedPseudoToolResponse, ValidationFailure> {
    if object.len() != 2 || !object.contains_key("action_input") {
        return Err(ValidationFailure {
            code: "MISSING_FIELDS",
            message: "action response must contain exactly 'action' and 'action_input'".to_string(),
        });
    }

    let Some(action_name) = object.get("action").and_then(Value::as_str) else {
        return Err(ValidationFailure {
            code: "INVALID_ACTION",
            message: "action must be a string".to_string(),
        });
    };

    let action_name = action_name.trim();
    if action_name.is_empty() {
        return Err(ValidationFailure {
            code: "INVALID_ACTION",
            message: "action must not be empty".to_string(),
        });
    }

    if matches!(context.choice, ToolChoiceMode::None) {
        return Err(ValidationFailure {
            code: "TOOL_NOT_ALLOWED",
            message: "tool use is disabled for this request".to_string(),
        });
    }

    if let ToolChoiceMode::Specific(required_name) = &context.choice
        && required_name != action_name
    {
        return Err(ValidationFailure {
            code: "WRONG_TOOL",
            message: format!(
                "response must call the required tool '{}' instead of '{}'",
                required_name, action_name
            ),
        });
    }

    let Some(action_input) = object.get("action_input") else {
        return Err(ValidationFailure {
            code: "MISSING_FIELDS",
            message: "action_input is required".to_string(),
        });
    };

    if !action_input.is_object() {
        return Err(ValidationFailure {
            code: "INVALID_ACTION_INPUT",
            message: "action_input must be a JSON object".to_string(),
        });
    }

    let Some(tool) = context.tools.iter().find(|tool| tool.name == action_name) else {
        return Err(ValidationFailure {
            code: "UNKNOWN_TOOL",
            message: format!("tool '{action_name}' is not in the allowed tool list"),
        });
    };

    if let Some(schema) = &tool.input_schema {
        validate_schema_subset(action_input, schema)?;
    }

    Ok(ParsedPseudoToolResponse::Action {
        tool_name: action_name.to_string(),
        action_input: action_input.clone(),
    })
}

fn validate_schema_subset(value: &Value, schema: &Value) -> Result<(), ValidationFailure> {
    let Some(schema_object) = schema.as_object() else {
        return Ok(());
    };

    if let Some(enum_values) = schema_object.get("enum").and_then(Value::as_array)
        && !enum_values.iter().any(|candidate| candidate == value)
    {
        return Err(ValidationFailure {
            code: "SCHEMA_INVALID",
            message: format!("value {value} was not in schema enum {enum_values:?}"),
        });
    }

    if let Some(schema_type) = schema_object.get("type") {
        validate_type(value, schema_type)?;
    }

    if let Some(required) = schema_object.get("required").and_then(Value::as_array)
        && let Some(object_value) = value.as_object()
    {
        for key in required.iter().filter_map(Value::as_str) {
            if !object_value.contains_key(key) {
                return Err(ValidationFailure {
                    code: "SCHEMA_INVALID",
                    message: format!("missing required field '{key}'"),
                });
            }
        }
    }

    if let Some(properties) = schema_object.get("properties").and_then(Value::as_object)
        && let Some(object_value) = value.as_object()
    {
        for (key, property_schema) in properties {
            if let Some(property_value) = object_value.get(key) {
                validate_schema_subset(property_value, property_schema)?;
            }
        }
    }

    if let Some(items) = schema_object.get("items")
        && let Some(array_value) = value.as_array()
    {
        for item in array_value {
            validate_schema_subset(item, items)?;
        }
    }

    Ok(())
}

fn validate_type(value: &Value, schema_type: &Value) -> Result<(), ValidationFailure> {
    if let Some(type_name) = schema_type.as_str() {
        return validate_type_name(value, type_name);
    }

    if let Some(type_names) = schema_type.as_array() {
        let mut last_error = None;
        for type_name in type_names.iter().filter_map(Value::as_str) {
            match validate_type_name(value, type_name) {
                Ok(()) => return Ok(()),
                Err(err) => last_error = Some(err),
            }
        }

        return Err(last_error.unwrap_or(ValidationFailure {
            code: "SCHEMA_INVALID",
            message: "no supported type matched".to_string(),
        }));
    }

    Ok(())
}

fn validate_type_name(value: &Value, type_name: &str) -> Result<(), ValidationFailure> {
    let valid = match type_name {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "null" => value.is_null(),
        _ => true,
    };

    if valid {
        Ok(())
    } else {
        Err(ValidationFailure {
            code: "SCHEMA_INVALID",
            message: format!("value {value} did not match type '{type_name}'"),
        })
    }
}

fn render_openai_message_for_prompt(message: &ChatMessage) -> String {
    let mut parts = Vec::new();
    let content = message.content.trim();
    if !content.is_empty() {
        parts.push(content.to_string());
    }

    if let Some(tool_calls) = &message.tool_calls {
        for tool_call in tool_calls {
            let arguments = tool_call.function.arguments.trim();
            let arguments = if arguments.is_empty() {
                "{}"
            } else {
                arguments
            };
            parts.push(format!(
                "[assistant_tool_call id={} name={} input={}]",
                tool_call.id, tool_call.function.name, arguments
            ));
        }
    }

    if message.role.eq_ignore_ascii_case("tool") {
        let tool_call_id = message
            .tool_call_id
            .as_deref()
            .unwrap_or("tool_call_unknown");
        let tool_name = message.name.as_deref().unwrap_or("tool");
        let tool_output = if content.is_empty() {
            "(empty)".to_string()
        } else {
            content.to_string()
        };
        return format!("[tool_result id={tool_call_id} name={tool_name}]\n{tool_output}");
    }

    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        ParsedPseudoToolResponse, ToolChoiceMode, context_from_openai_request,
        parse_pseudo_tool_response, should_enable_openai_protocol,
    };
    use crate::models::{ChatCompletionRequest, ChatMessage};

    fn build_request(value: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(value).expect("request should deserialize")
    }

    #[test]
    fn openai_context_detects_required_tool_mode() {
        let request = build_request(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "description": "Search docs",
                    "parameters": {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}}
                }
            }],
            "tool_choice": "required"
        }));

        let context = context_from_openai_request(&request);
        assert!(should_enable_openai_protocol(&request, &context));
        assert!(matches!(context.choice, ToolChoiceMode::Required));
        assert_eq!(context.tool_names(), vec!["search_docs"]);
    }

    #[test]
    fn parses_valid_action_response() {
        let request = build_request(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "parameters": {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}}
                }
            }]
        }));

        let context = context_from_openai_request(&request);
        let parsed = parse_pseudo_tool_response(
            r#"{"action":"search_docs","action_input":{"query":"hello"}}"#,
            &context,
        )
        .expect("response should parse");

        match parsed {
            ParsedPseudoToolResponse::Action {
                tool_name,
                action_input,
            } => {
                assert_eq!(tool_name, "search_docs");
                assert_eq!(action_input["query"], "hello");
            }
            ParsedPseudoToolResponse::Final { .. } => panic!("expected action"),
        }
    }

    #[test]
    fn rejects_dirty_json_response() {
        let request = build_request(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "search_docs"}}]
        }));

        let context = context_from_openai_request(&request);
        let err = parse_pseudo_tool_response(
            "```json\n{\"action\":\"search_docs\",\"action_input\":{}}\n```",
            &context,
        )
        .expect_err("dirty json should fail");

        assert_eq!(err.code, "DIRTY_JSON");
    }

    #[test]
    fn rejects_missing_required_schema_fields() {
        let request = build_request(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "parameters": {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}}
                }
            }]
        }));

        let context = context_from_openai_request(&request);
        let err =
            parse_pseudo_tool_response(r#"{"action":"search_docs","action_input":{}}"#, &context)
                .expect_err("missing query should fail");

        assert_eq!(err.code, "SCHEMA_INVALID");
    }

    #[test]
    fn normalizes_openai_tool_messages_into_prompt_text() {
        let messages = vec![ChatMessage {
            role: "tool".to_string(),
            content: "tool output".to_string(),
            tool_call_id: Some("call_123".to_string()),
            name: Some("search_docs".to_string()),
            tool_calls: None,
        }];

        let normalized = super::normalize_openai_messages(&messages);
        assert!(
            normalized[0]
                .content
                .contains("[tool_result id=call_123 name=search_docs]")
        );
    }
}
