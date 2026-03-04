use anyhow::{Context, anyhow};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use tokio::sync::mpsc;

use crate::{
    config::Settings,
    models::{ChatMessage, DEFAULT_MODEL},
};

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct OnyxToolMetadata {
    pub id: i64,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub in_code_tool_id: Option<String>,
}

pub async fn fetch_available_tools(
    client: &reqwest::Client,
    settings: &Settings,
    cookie: &str,
) -> anyhow::Result<Vec<OnyxToolMetadata>> {
    let response = client
        .get(format!("{}/api/tool", settings.onyx_base_url))
        .header("Cookie", format!("fastapiusersauth={cookie}"))
        .header("Origin", &settings.onyx_origin_url)
        .header("Referer", &settings.onyx_referer)
        .send()
        .await
        .context("failed to call tool catalog endpoint")?;

    if response.status() == reqwest::StatusCode::UNAUTHORIZED
        || response.status() == reqwest::StatusCode::FORBIDDEN
    {
        return Err(anyhow!("onyx auth failed: {}", response.status().as_u16()));
    }

    if !response.status().is_success() {
        let code = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("onyx tool catalog HTTP {code}: {body}"));
    }

    let payload: Value = response
        .json()
        .await
        .context("invalid tool catalog response json")?;
    parse_tool_catalog_response(payload)
}

pub fn resolve_tool_selection_by_name(
    available_tools: &[OnyxToolMetadata],
    requested_tool_names: &[String],
    forced_tool_name: Option<&str>,
) -> (Option<Vec<i64>>, Option<i64>) {
    let mut lookup = HashMap::<String, i64>::new();

    for tool in available_tools {
        for candidate in [
            tool.name.as_deref(),
            tool.display_name.as_deref(),
            tool.in_code_tool_id.as_deref(),
        ] {
            let Some(name) = candidate else {
                continue;
            };
            for alias in tool_name_aliases(name) {
                lookup.entry(alias).or_insert(tool.id);
            }
        }
    }

    let mut allowed_ids = Vec::new();
    for name in requested_tool_names {
        if let Some(id) = resolve_lookup_id(&lookup, name)
            && !allowed_ids.contains(&id)
        {
            allowed_ids.push(id);
        }
    }

    let forced_tool_id = forced_tool_name.and_then(|name| resolve_lookup_id(&lookup, name));

    if let Some(forced_id) = forced_tool_id
        && !allowed_ids.contains(&forced_id)
    {
        allowed_ids.push(forced_id);
    }

    let allowed_tool_ids = if allowed_ids.is_empty() {
        None
    } else {
        Some(allowed_ids)
    };

    (allowed_tool_ids, forced_tool_id)
}

pub async fn full_chat(
    client: &reqwest::Client,
    settings: &Settings,
    cookie: &str,
    messages: &[ChatMessage],
    model_name: &str,
    allowed_tool_ids: Option<&[i64]>,
    forced_tool_id: Option<i64>,
) -> anyhow::Result<(String, String)> {
    let chat_session_id = create_chat_session(client, settings, cookie).await?;
    let (provider, version) = resolve_model(model_name);
    let payload = build_send_chat_payload(
        messages,
        &chat_session_id,
        &provider,
        &version,
        settings,
        allowed_tool_ids,
        forced_tool_id,
    );

    let text = send_chat_message_text(client, settings, cookie, &payload).await?;
    let (content, thinking) = parse_onyx_stream_text(&text);

    if content.is_empty() && thinking.is_empty() {
        let snippet = text.chars().take(500).collect::<String>();
        return Err(anyhow!("empty upstream response: {snippet}"));
    }

    if should_retry_python_syntax_error(&content) {
        let mut retry_messages = messages.to_vec();
        retry_messages.push(ChatMessage {
            role: "user".to_string(),
            content: "The previous Python sandbox run failed with SyntaxError. Retry once with syntactically valid Python code and return the final answer.".to_string(),
        });

        let retry_payload = build_send_chat_payload(
            &retry_messages,
            &chat_session_id,
            &provider,
            &version,
            settings,
            allowed_tool_ids,
            forced_tool_id,
        );

        if let Ok(retry_text) = send_chat_message_text(client, settings, cookie, &retry_payload).await {
            let (retry_content, retry_thinking) = parse_onyx_stream_text(&retry_text);
            if !(retry_content.is_empty() && retry_thinking.is_empty())
                && !should_retry_python_syntax_error(&retry_content)
            {
                return Ok((retry_content, retry_thinking));
            }
        }
    }

    Ok((content, thinking))
}

async fn send_chat_message_text(
    client: &reqwest::Client,
    settings: &Settings,
    cookie: &str,
    payload: &Value,
) -> anyhow::Result<String> {
    let response = client
        .post(format!("{}/api/chat/send-chat-message", settings.onyx_base_url))
        .header("Cookie", format!("fastapiusersauth={cookie}"))
        .header("Origin", &settings.onyx_origin_url)
        .header("Referer", &settings.onyx_referer)
        .json(payload)
        .send()
        .await
        .context("failed to call send-chat-message")?;

    if response.status() == reqwest::StatusCode::UNAUTHORIZED
        || response.status() == reqwest::StatusCode::FORBIDDEN
    {
        return Err(anyhow!("onyx auth failed: {}", response.status().as_u16()));
    }

    if !response.status().is_success() {
        let code = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("onyx send-chat-message HTTP {code}: {body}"));
    }

    response
        .text()
        .await
        .context("failed to read response body")
}

fn should_retry_python_syntax_error(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    lower.contains("syntaxerror")
        && (lower.contains("python") || lower.contains("code interpreter"))
}

/// SSE event types emitted by streaming_chat
#[derive(Debug)]
pub enum StreamEvent {
    /// role announcement (first chunk)
    Role,
    /// reasoning / thinking content delta
    Reasoning(String),
    /// assistant content delta
    Content(String),
    /// generation complete
    Done,
}

pub fn format_error_chain(err: &anyhow::Error) -> String {
    let chain = err
        .chain()
        .map(|cause| cause.to_string())
        .collect::<Vec<_>>();

    if chain.is_empty() {
        return String::from("unknown error");
    }

    let root_cause = err.root_cause().to_string();
    format!(
        "{} | chain: {} | root_cause: {}",
        chain[0],
        chain.join(" -> "),
        root_cause
    )
}

/// Like `full_chat`, but streams parsed events back via an mpsc channel.
/// The caller receives `StreamEvent` items as they arrive from Onyx.
pub async fn streaming_chat(
    client: &reqwest::Client,
    settings: &Settings,
    cookie: &str,
    messages: &[ChatMessage],
    model_name: &str,
    allowed_tool_ids: Option<&[i64]>,
    forced_tool_id: Option<i64>,
) -> anyhow::Result<mpsc::Receiver<StreamEvent>> {
    let chat_session_id = create_chat_session(client, settings, cookie).await?;
    let (provider, version) = resolve_model(model_name);
    let payload = build_send_chat_payload(
        messages,
        &chat_session_id,
        &provider,
        &version,
        settings,
        allowed_tool_ids,
        forced_tool_id,
    );

    let response = client
        .post(format!("{}/api/chat/send-chat-message", settings.onyx_base_url))
        .header("Cookie", format!("fastapiusersauth={cookie}"))
        .header("Origin", &settings.onyx_origin_url)
        .header("Referer", &settings.onyx_referer)
        .json(&payload)
        .send()
        .await
        .context("failed to call send-chat-message")?;

    if response.status() == reqwest::StatusCode::UNAUTHORIZED
        || response.status() == reqwest::StatusCode::FORBIDDEN
    {
        return Err(anyhow!("onyx auth failed: {}", response.status().as_u16()));
    }

    if !response.status().is_success() {
        let code = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("onyx send-chat-message HTTP {code}: {body}"));
    }

    let (tx, rx) = mpsc::channel::<StreamEvent>(64);

    // Spawn background task reading the byte stream from Onyx
    let byte_stream = response.bytes_stream();
    tokio::spawn(async move {
        let _ = tx.send(StreamEvent::Role).await;

        let mut buffer = String::new();
        tokio::pin!(byte_stream);

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(_) => break,
            };

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete lines
            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim().to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                let line = if let Some(stripped) = line.strip_prefix("data:") {
                    stripped.trim().to_string()
                } else {
                    line
                };

                if line.is_empty() || line == "[DONE]" {
                    continue;
                }

                let Ok(root) = serde_json::from_str::<Value>(&line) else {
                    continue;
                };

                let obj = root.get("obj").unwrap_or(&root);
                let event_type = obj
                    .get("type")
                    .and_then(|v| v.as_str())
                    .or_else(|| root.get("type").and_then(|v| v.as_str()));

                match event_type {
                    Some("reasoning_delta") => {
                        if let Some(r) = obj.get("reasoning").and_then(|v| v.as_str()) {
                            let _ = tx.send(StreamEvent::Reasoning(r.to_string())).await;
                        }
                    }
                    Some("message_delta") => {
                        if let Some(c) = obj.get("content").and_then(|v| v.as_str()) {
                            let _ = tx.send(StreamEvent::Content(c.to_string())).await;
                        }
                    }
                    Some("stop") => {
                        let _ = tx.send(StreamEvent::Done).await;
                        return;
                    }
                    _ => {}
                }
            }
        }

        // If stream ended without an explicit stop event, send Done anyway
        let _ = tx.send(StreamEvent::Done).await;
    });

    Ok(rx)
}

fn parse_onyx_stream_text(text: &str) -> (String, String) {
    let mut thinking = String::new();
    let mut content = String::new();

    for raw_line in text.lines() {
        let mut line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(stripped) = line.strip_prefix("data:") {
            line = stripped.trim();
        }
        if line.is_empty() || line == "[DONE]" {
            continue;
        }

        let parsed: Result<Value, _> = serde_json::from_str(line);
        let Ok(root) = parsed else {
            continue;
        };

        let obj = root.get("obj").unwrap_or(&root);
        let event_type = obj
            .get("type")
            .and_then(|v| v.as_str())
            .or_else(|| root.get("type").and_then(|v| v.as_str()));

        match event_type {
            Some("reasoning_delta") => {
                if let Some(r) = obj.get("reasoning").and_then(|v| v.as_str()) {
                    thinking.push_str(r);
                }
            }
            Some("message_delta") => {
                if let Some(c) = obj.get("content").and_then(|v| v.as_str()) {
                    content.push_str(c);
                }
            }
            Some("stop") => break,
            _ => {}
        }
    }

    (content, thinking)
}

fn build_send_chat_payload(
    messages: &[ChatMessage],
    chat_session_id: &str,
    provider: &str,
    version: &str,
    settings: &Settings,
    allowed_tool_ids: Option<&[i64]>,
    forced_tool_id: Option<i64>,
) -> Value {
    serde_json::json!({
        "message": build_prompt(messages),
        "chat_session_id": chat_session_id,
        "parent_message_id": null,
        "file_descriptors": [],
        "internal_search_filters": {
            "source_type": null,
            "document_set": null,
            "time_cutoff": null,
            "tags": []
        },
        "deep_research": false,
        "allowed_tool_ids": allowed_tool_ids.map(|ids| ids.to_vec()),
        "forced_tool_id": forced_tool_id,
        "llm_override": {
            "temperature": 0.5,
            "model_provider": provider,
            "model_version": version
        },
        "origin": settings.onyx_origin,
    })
}

fn parse_tool_catalog_response(payload: Value) -> anyhow::Result<Vec<OnyxToolMetadata>> {
    match payload {
        Value::Array(_) => serde_json::from_value(payload).context("tool catalog array parse failed"),
        Value::Object(map) => {
            for key in ["tools", "items", "data"] {
                if let Some(value) = map.get(key)
                    && value.is_array()
                {
                    return serde_json::from_value(value.clone())
                        .context("tool catalog object array parse failed");
                }
            }
            Err(anyhow!("unexpected tool catalog response shape"))
        }
        _ => Err(anyhow!("unexpected tool catalog response type")),
    }
}

fn normalize_tool_name(input: &str) -> String {
    input
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

fn tool_name_aliases(input: &str) -> Vec<String> {
    let normalized = normalize_tool_name(input);
    if normalized.is_empty() {
        return Vec::new();
    }

    let mut aliases = vec![normalized.clone()];
    if normalized.ends_with("tool") && normalized.len() > 4 {
        aliases.push(normalized[..normalized.len() - 4].to_string());
    }
    aliases
}

fn resolve_lookup_id(lookup: &HashMap<String, i64>, input: &str) -> Option<i64> {
    for alias in tool_name_aliases(input) {
        if let Some(id) = lookup.get(&alias) {
            return Some(*id);
        }
    }
    None
}

async fn create_chat_session(
    client: &reqwest::Client,
    settings: &Settings,
    cookie: &str,
) -> anyhow::Result<String> {
    let payload = serde_json::json!({
        "persona_id": settings.onyx_persona_id,
        "description": null,
        "project_id": null
    });

    let response = client
        .post(format!("{}/api/chat/create-chat-session", settings.onyx_base_url))
        .header("Cookie", format!("fastapiusersauth={cookie}"))
        .header("Origin", &settings.onyx_origin_url)
        .header("Referer", &settings.onyx_referer)
        .json(&payload)
        .send()
        .await
        .context("failed to call create-chat-session")?;

    if response.status() == reqwest::StatusCode::UNAUTHORIZED
        || response.status() == reqwest::StatusCode::FORBIDDEN
    {
        return Err(anyhow!("onyx auth failed: {}", response.status().as_u16()));
    }

    if !response.status().is_success() {
        let code = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("onyx create-chat-session HTTP {code}: {body}"));
    }

    let data: Value = response
        .json()
        .await
        .context("invalid create-chat-session response json")?;

    data.get("chat_session_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("create-chat-session missing chat_session_id"))
}

pub fn build_prompt(messages: &[ChatMessage]) -> String {
    if messages.is_empty() {
        return String::new();
    }
    messages
        .iter()
        .filter_map(|m| {
            let content = m.content.trim();
            if content.is_empty() {
                None
            } else {
                Some(format!("{}: {}", m.role, content))
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn resolve_model(model_name: &str) -> (String, String) {
    match model_name {
        "claude-opus-4.6" => ("Anthropic".to_string(), "claude-opus-4-6".to_string()),
        "claude-opus-4.5" => ("Anthropic".to_string(), "claude-opus-4-5".to_string()),
        "claude-sonnet-4.5" => ("Anthropic".to_string(), "claude-sonnet-4-5".to_string()),
        "gpt-5.2" => ("OpenAI".to_string(), "gpt-5.2".to_string()),
        "gpt-5-mini" => ("OpenAI".to_string(), "gpt-5-mini".to_string()),
        "gpt-4.1" => ("OpenAI".to_string(), "gpt-4.1".to_string()),
        "gpt-4o" => ("OpenAI".to_string(), "gpt-4o".to_string()),
        "o3" => ("OpenAI".to_string(), "o3".to_string()),
        _ => {
            let parts: Vec<&str> = model_name.split("__").collect();
            if parts.len() == 3 {
                (parts[0].to_string(), parts[2].to_string())
            } else {
                let default = DEFAULT_MODEL;
                resolve_model(default)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::anyhow;
    use crate::models::ChatMessage;
    use serde_json::json;

    use super::{build_prompt, parse_onyx_stream_text};

    #[test]
    fn parses_nested_obj_event_format() {
        let body = r#"{"user_message_id":1}
{"placement":null,"obj":{"type":"reasoning_delta","reasoning":"思考"}}
{"placement":null,"obj":{"type":"message_delta","content":"你好"}}
{"placement":null,"obj":{"type":"stop"}}"#;

        let (content, thinking) = parse_onyx_stream_text(body);
        assert_eq!(content, "你好");
        assert_eq!(thinking, "思考");
    }

    #[test]
    fn parses_data_prefix_and_top_level_format() {
        let body = r#"data: {"type":"reasoning_delta","reasoning":"A"}
data: {"type":"message_delta","content":"B"}
data: {"type":"stop"}
data: [DONE]"#;

        let (content, thinking) = parse_onyx_stream_text(body);
        assert_eq!(content, "B");
        assert_eq!(thinking, "A");
    }

    #[test]
    fn formats_full_error_chain_for_precise_debugging() {
        let err = anyhow!("dns lookup failed").context("failed to call send-chat-message");
        let rendered = super::format_error_chain(&err);

        assert!(
            rendered.contains("failed to call send-chat-message"),
            "should include top-level context"
        );
        assert!(
            rendered.contains("dns lookup failed"),
            "should include root cause"
        );
        assert!(
            rendered.contains("root_cause:"),
            "should explicitly mark root cause"
        );
    }

    #[test]
    fn build_prompt_skips_empty_content_messages() {
        let prompt = build_prompt(&[
            ChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: String::new(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "result text".to_string(),
            },
        ]);

        assert_eq!(prompt, "user: hello\nuser: result text");
    }

    #[test]
    fn resolve_tool_selection_maps_tool_names_and_forced_name() {
        let tools = vec![
            super::OnyxToolMetadata {
                id: 11,
                name: Some("web_search".to_string()),
                display_name: Some("Web Search".to_string()),
                in_code_tool_id: Some("WebSearchTool".to_string()),
            },
            super::OnyxToolMetadata {
                id: 22,
                name: Some("shell".to_string()),
                display_name: Some("Shell".to_string()),
                in_code_tool_id: Some("PythonTool".to_string()),
            },
        ];

        let requested = vec!["web-search".to_string(), "shell".to_string()];
        let (allowed, forced) =
            super::resolve_tool_selection_by_name(&tools, &requested, Some("WebSearchTool"));

        assert_eq!(allowed, Some(vec![11, 22]));
        assert_eq!(forced, Some(11));
    }

    #[test]
    fn resolve_tool_selection_maps_python_alias_to_python_tool() {
        let tools = vec![super::OnyxToolMetadata {
            id: 22,
            name: Some("PythonTool".to_string()),
            display_name: Some("Code Interpreter".to_string()),
            in_code_tool_id: Some("PythonTool".to_string()),
        }];

        let requested = vec!["python".to_string()];
        let (allowed, forced) = super::resolve_tool_selection_by_name(&tools, &requested, Some("python"));

        assert_eq!(allowed, Some(vec![22]));
        assert_eq!(forced, Some(22));
    }

    #[test]
    fn resolve_tool_selection_maps_open_url_alias_to_open_url_tool() {
        let tools = vec![super::OnyxToolMetadata {
            id: 33,
            name: Some("OpenURLTool".to_string()),
            display_name: Some("Open URL".to_string()),
            in_code_tool_id: Some("OpenURLTool".to_string()),
        }];

        let requested = vec!["open_url".to_string()];
        let (allowed, forced) =
            super::resolve_tool_selection_by_name(&tools, &requested, Some("open_url"));

        assert_eq!(allowed, Some(vec![33]));
        assert_eq!(forced, Some(33));
    }

    #[test]
    fn build_send_chat_payload_includes_tool_id_fields() {
        let settings = crate::config::Settings::default();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        }];

        let payload = super::build_send_chat_payload(
            &messages,
            "chat-session-1",
            "Anthropic",
            "claude-opus-4-6",
            &settings,
            Some(&[11, 22]),
            Some(22),
        );

        assert_eq!(payload["allowed_tool_ids"], json!([11, 22]));
        assert_eq!(payload["forced_tool_id"], json!(22));
    }

    #[test]
    fn retry_trigger_detects_python_syntax_error() {
        let msg = "Python execution failed: SyntaxError: invalid syntax";
        assert!(super::should_retry_python_syntax_error(msg));
    }

    #[test]
    fn retry_trigger_ignores_non_syntaxerror_content() {
        let msg = "Python tool completed successfully.";
        assert!(!super::should_retry_python_syntax_error(msg));
    }
}
