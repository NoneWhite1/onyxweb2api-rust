use anyhow::{Context, anyhow};
use futures_util::StreamExt;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::{
    config::Settings,
    models::{ChatMessage, DEFAULT_MODEL},
};

pub async fn full_chat(
    client: &reqwest::Client,
    settings: &Settings,
    cookie: &str,
    messages: &[ChatMessage],
    model_name: &str,
) -> anyhow::Result<(String, String)> {
    let chat_session_id = create_chat_session(client, settings, cookie).await?;
    let (provider, version) = resolve_model(model_name);

    let payload = serde_json::json!({
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
        "forced_tool_id": null,
        "llm_override": {
            "temperature": 0.5,
            "model_provider": provider,
            "model_version": version
        },
        "origin": settings.onyx_origin,
    });

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

    let text = response.text().await.context("failed to read response body")?;
    let (content, thinking) = parse_onyx_stream_text(&text);

    if content.is_empty() && thinking.is_empty() {
        let snippet = text.chars().take(500).collect::<String>();
        return Err(anyhow!("empty upstream response: {snippet}"));
    }

    Ok((content, thinking))
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

/// Like `full_chat`, but streams parsed events back via an mpsc channel.
/// The caller receives `StreamEvent` items as they arrive from Onyx.
pub async fn streaming_chat(
    client: &reqwest::Client,
    settings: &Settings,
    cookie: &str,
    messages: &[ChatMessage],
    model_name: &str,
) -> anyhow::Result<mpsc::Receiver<StreamEvent>> {
    let chat_session_id = create_chat_session(client, settings, cookie).await?;
    let (provider, version) = resolve_model(model_name);

    let payload = serde_json::json!({
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
        "forced_tool_id": null,
        "llm_override": {
            "temperature": 0.5,
            "model_provider": provider,
            "model_version": version
        },
        "origin": settings.onyx_origin,
    });

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
        .map(|m| format!("{}: {}", m.role, m.content))
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
    use super::parse_onyx_stream_text;

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
}
