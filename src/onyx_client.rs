use anyhow::{Context, anyhow};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;

use crate::{
    config::Settings,
    models::{ChatMessage, ClaudeToolDefinition, DEFAULT_MODEL},
};

const MAX_SEND_CHAT_ATTEMPTS: usize = 3;
const SEND_CHAT_RETRY_BACKOFF_MS: u64 = 250;
const MAX_TOOL_CALL_JSON_LEN: usize = 16 * 1024;
const MAX_BASH_COMMAND_LEN: usize = 1024;

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
    let request_payload = serde_json::json!({});
    let response = match client
        .get(format!("{}/api/tool", settings.onyx_base_url))
        .header("Cookie", format!("fastapiusersauth={cookie}"))
        .header("Origin", &settings.onyx_origin_url)
        .header("Referer", &settings.onyx_referer)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(err) => {
            let wrapped = anyhow!(err).context("failed to call tool catalog endpoint");
            append_upstream_error_record(
                settings,
                "tool-catalog",
                "request_send",
                cookie,
                &request_payload,
                None,
                &format_error_chain(&wrapped),
            )
            .await;
            return Err(wrapped);
        }
    };

    if response.status() == reqwest::StatusCode::UNAUTHORIZED
        || response.status() == reqwest::StatusCode::FORBIDDEN
    {
        let err = anyhow!("onyx auth failed: {}", response.status().as_u16());
        append_upstream_error_record(
            settings,
            "tool-catalog",
            "http_auth",
            cookie,
            &request_payload,
            Some(response.status().as_u16()),
            &err.to_string(),
        )
        .await;
        return Err(err);
    }

    if !response.status().is_success() {
        let code = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();
        let err = anyhow!("onyx tool catalog HTTP {code}: {body}");
        append_upstream_audit_record(
            settings,
            "tool-catalog",
            "http_status",
            cookie,
            &request_payload,
            Some(code),
            &serde_json::json!({"error": body}),
        )
        .await;
        append_upstream_error_record(
            settings,
            "tool-catalog",
            "http_status",
            cookie,
            &request_payload,
            Some(code),
            &err.to_string(),
        )
        .await;
        return Err(err);
    }

    let payload: Value = response
        .json()
        .await
        .context("invalid tool catalog response json")?;
    append_upstream_audit_record(
        settings,
        "tool-catalog",
        "response_success",
        cookie,
        &request_payload,
        Some(200),
        &payload,
    )
    .await;
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
    for attempt in 1..=MAX_SEND_CHAT_ATTEMPTS {
        let response = match client
            .post(format!("{}/api/chat/send-chat-message", settings.onyx_base_url))
            .header("Cookie", format!("fastapiusersauth={cookie}"))
            .header("Origin", &settings.onyx_origin_url)
            .header("Referer", &settings.onyx_referer)
            .json(payload)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(err) => {
                let wrapped = anyhow!(err).context("failed to call send-chat-message");
                append_upstream_error_record(
                    settings,
                    "send-chat-message",
                    "request_send",
                    cookie,
                    payload,
                    None,
                    &format_error_chain(&wrapped),
                )
                .await;

                if attempt < MAX_SEND_CHAT_ATTEMPTS && is_retryable_upstream_io_error(&wrapped) {
                    tokio::time::sleep(Duration::from_millis(
                        SEND_CHAT_RETRY_BACKOFF_MS * attempt as u64,
                    ))
                    .await;
                    continue;
                }
                return Err(wrapped);
            }
        };

        if response.status() == reqwest::StatusCode::UNAUTHORIZED
            || response.status() == reqwest::StatusCode::FORBIDDEN
        {
            let err = anyhow!("onyx auth failed: {}", response.status().as_u16());
            append_upstream_audit_record(
                settings,
                "send-chat-message",
                "http_auth",
                cookie,
                payload,
                Some(response.status().as_u16()),
                &serde_json::json!({"error": err.to_string()}),
            )
            .await;
            append_upstream_error_record(
                settings,
                "send-chat-message",
                "http_auth",
                cookie,
                payload,
                Some(response.status().as_u16()),
                &err.to_string(),
            )
            .await;
            return Err(err);
        }

        if !response.status().is_success() {
            let code = response.status().as_u16();
            let body = match response.text().await {
                Ok(text) => text,
                Err(err) => {
                    let wrapped = anyhow!(err).context("failed to read response body");
                    append_upstream_error_record(
                        settings,
                        "send-chat-message",
                        "http_error_body_read",
                        cookie,
                        payload,
                        Some(code),
                        &format_error_chain(&wrapped),
                    )
                    .await;
                    if attempt < MAX_SEND_CHAT_ATTEMPTS
                        && is_retryable_upstream_io_error(&wrapped)
                    {
                        tokio::time::sleep(Duration::from_millis(
                            SEND_CHAT_RETRY_BACKOFF_MS * attempt as u64,
                        ))
                        .await;
                        continue;
                    }
                    return Err(wrapped);
                }
            };

            let err = anyhow!("onyx send-chat-message HTTP {code}: {body}");
            append_upstream_audit_record(
                settings,
                "send-chat-message",
                "http_status",
                cookie,
                payload,
                Some(code),
                &serde_json::json!({"error": body}),
            )
            .await;
            append_upstream_error_record(
                settings,
                "send-chat-message",
                "http_status",
                cookie,
                payload,
                Some(code),
                &err.to_string(),
            )
            .await;
            return Err(err);
        }

        match response.text().await {
            Ok(text) => {
                append_upstream_audit_record(
                    settings,
                    "send-chat-message",
                    "response_success",
                    cookie,
                    payload,
                    Some(200),
                    &serde_json::json!({"body": text}),
                )
                .await;
                return Ok(text);
            }
            Err(err) => {
                let wrapped = anyhow!(err).context("failed to read response body");
                append_upstream_audit_record(
                    settings,
                    "send-chat-message",
                    "response_read",
                    cookie,
                    payload,
                    Some(200),
                    &serde_json::json!({"error": format_error_chain(&wrapped)}),
                )
                .await;
                append_upstream_error_record(
                    settings,
                    "send-chat-message",
                    "response_read",
                    cookie,
                    payload,
                    Some(200),
                    &format_error_chain(&wrapped),
                )
                .await;
                if attempt < MAX_SEND_CHAT_ATTEMPTS && is_retryable_upstream_io_error(&wrapped) {
                    tokio::time::sleep(Duration::from_millis(
                        SEND_CHAT_RETRY_BACKOFF_MS * attempt as u64,
                    ))
                    .await;
                    continue;
                }
                return Err(wrapped);
            }
        }
    }

    Err(anyhow!(
        "failed to call send-chat-message after {} attempts",
        MAX_SEND_CHAT_ATTEMPTS
    ))
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

#[derive(Debug, Serialize)]
struct UpstreamErrorRecord {
    ts_ms: u64,
    endpoint: String,
    stage: String,
    status: Option<u16>,
    error: String,
    cookie_fingerprint: String,
    payload: Value,
}

#[derive(Debug, Serialize)]
struct UpstreamAuditRecord {
    ts_ms: u64,
    endpoint: String,
    stage: String,
    status: Option<u16>,
    cookie_fingerprint: String,
    payload: Value,
    response: Value,
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn mask_cookie(cookie: &str) -> String {
    let len = cookie.chars().count();
    if len <= 8 {
        return "***".to_string();
    }
    let prefix: String = cookie.chars().take(4).collect();
    let suffix: String = cookie
        .chars()
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{prefix}...{suffix}")
}

async fn append_upstream_error_record(
    settings: &Settings,
    endpoint: &str,
    stage: &str,
    cookie: &str,
    payload: &Value,
    status: Option<u16>,
    error: &str,
) {
    let path_value = settings.request_error_log_path.trim();
    if path_value.is_empty() {
        return;
    }

    let record = UpstreamErrorRecord {
        ts_ms: now_unix_ms(),
        endpoint: endpoint.to_string(),
        stage: stage.to_string(),
        status,
        error: error.to_string(),
        cookie_fingerprint: mask_cookie(cookie),
        payload: payload.clone(),
    };

    let path = Path::new(path_value);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
        && let Err(err) = tokio::fs::create_dir_all(parent).await
    {
        tracing::error!(
            log_path = %path.display(),
            error = %err,
            "failed to create upstream error log directory"
        );
        return;
    }

    let serialized = match serde_json::to_string(&record) {
        Ok(s) => s,
        Err(err) => {
            tracing::error!(
                log_path = %path.display(),
                error = %err,
                "failed to serialize upstream error record"
            );
            return;
        }
    };

    match tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await
    {
        Ok(mut file) => {
            if let Err(err) = file
                .write_all(format!("{serialized}\n").as_bytes())
                .await
            {
                tracing::error!(
                    log_path = %path.display(),
                    error = %err,
                    "failed to append upstream error log"
                );
            }
        }
        Err(err) => {
            tracing::error!(
                log_path = %path.display(),
                error = %err,
                "failed to open upstream error log file"
            );
        }
    }
}

async fn append_upstream_audit_record(
    settings: &Settings,
    endpoint: &str,
    stage: &str,
    cookie: &str,
    payload: &Value,
    status: Option<u16>,
    response: &Value,
) {
    let path_value = settings.request_audit_log_path.trim();
    if path_value.is_empty() {
        return;
    }

    let record = UpstreamAuditRecord {
        ts_ms: now_unix_ms(),
        endpoint: endpoint.to_string(),
        stage: stage.to_string(),
        status,
        cookie_fingerprint: mask_cookie(cookie),
        payload: payload.clone(),
        response: response.clone(),
    };

    let path = Path::new(path_value);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
        && let Err(err) = tokio::fs::create_dir_all(parent).await
    {
        tracing::error!(
            log_path = %path.display(),
            error = %err,
            "failed to create upstream audit log directory"
        );
        return;
    }

    let serialized = match serde_json::to_string(&record) {
        Ok(s) => s,
        Err(err) => {
            tracing::error!(
                log_path = %path.display(),
                error = %err,
                "failed to serialize upstream audit record"
            );
            return;
        }
    };

    match tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await
    {
        Ok(mut file) => {
            if let Err(err) = file
                .write_all(format!("{serialized}\n").as_bytes())
                .await
            {
                tracing::error!(
                    log_path = %path.display(),
                    error = %err,
                    "failed to append upstream audit log"
                );
            }
        }
        Err(err) => {
            tracing::error!(
                log_path = %path.display(),
                error = %err,
                "failed to open upstream audit log file"
            );
        }
    }
}

fn is_retryable_upstream_io_error(err: &anyhow::Error) -> bool {
    let chain = format_error_chain(err).to_ascii_lowercase();
    let markers = [
        "unexpected eof",
        "chunk size line",
        "error reading a body from connection",
        "request or response body error",
        "error decoding response body",
        "connection reset",
        "connection closed",
        "broken pipe",
        "timed out",
        "timeout",
    ];
    markers.iter().any(|m| chain.contains(m))
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

    let response = match client
        .post(format!("{}/api/chat/send-chat-message", settings.onyx_base_url))
        .header("Cookie", format!("fastapiusersauth={cookie}"))
        .header("Origin", &settings.onyx_origin_url)
        .header("Referer", &settings.onyx_referer)
        .json(&payload)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(err) => {
            let wrapped = anyhow!(err).context("failed to call send-chat-message");
            append_upstream_error_record(
                settings,
                "send-chat-message",
                "stream_request_send",
                cookie,
                &payload,
                None,
                &format_error_chain(&wrapped),
            )
            .await;
            return Err(wrapped);
        }
    };

    if response.status() == reqwest::StatusCode::UNAUTHORIZED
        || response.status() == reqwest::StatusCode::FORBIDDEN
    {
        let err = anyhow!("onyx auth failed: {}", response.status().as_u16());
        append_upstream_audit_record(
            settings,
            "send-chat-message",
            "stream_http_auth",
            cookie,
            &payload,
            Some(response.status().as_u16()),
            &serde_json::json!({"error": err.to_string()}),
        )
        .await;
        append_upstream_error_record(
            settings,
            "send-chat-message",
            "stream_http_auth",
            cookie,
            &payload,
            Some(response.status().as_u16()),
            &err.to_string(),
        )
        .await;
        return Err(err);
    }

    if !response.status().is_success() {
        let code = response.status().as_u16();
        let body = match response.text().await {
            Ok(text) => text,
            Err(err) => {
                let wrapped = anyhow!(err).context("failed to read response body");
                append_upstream_error_record(
                    settings,
                    "send-chat-message",
                    "stream_http_error_body_read",
                    cookie,
                    &payload,
                    Some(code),
                    &format_error_chain(&wrapped),
                )
                .await;
                return Err(wrapped);
            }
        };
        let err = anyhow!("onyx send-chat-message HTTP {code}: {body}");
        append_upstream_audit_record(
            settings,
            "send-chat-message",
            "stream_http_status",
            cookie,
            &payload,
            Some(code),
            &serde_json::json!({"error": body}),
        )
        .await;
        append_upstream_error_record(
            settings,
            "send-chat-message",
            "stream_http_status",
            cookie,
            &payload,
            Some(code),
            &err.to_string(),
        )
        .await;
        return Err(err);
    }

    let (tx, rx) = mpsc::channel::<StreamEvent>(64);

    // Spawn background task reading the byte stream from Onyx
    let byte_stream = response.bytes_stream();
    let settings_for_stream = settings.clone();
    let payload_for_stream = payload.clone();
    let cookie_for_stream = cookie.to_string();
    tokio::spawn(async move {
        let _ = tx.send(StreamEvent::Role).await;

        let mut buffer = String::new();
        let mut raw_stream_body = String::new();
        tokio::pin!(byte_stream);

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(err) => {
                    let wrapped = anyhow!(err).context("failed to read streaming response body");
                    append_upstream_audit_record(
                        &settings_for_stream,
                        "send-chat-message",
                        "stream_response_read",
                        &cookie_for_stream,
                        &payload_for_stream,
                        Some(200),
                        &serde_json::json!({
                            "body": raw_stream_body,
                            "error": format_error_chain(&wrapped)
                        }),
                    )
                    .await;
                    append_upstream_error_record(
                        &settings_for_stream,
                        "send-chat-message",
                        "stream_response_read",
                        &cookie_for_stream,
                        &payload_for_stream,
                        Some(200),
                        &format_error_chain(&wrapped),
                    )
                    .await;
                    tracing::error!(
                        error = %format_error_chain(&wrapped),
                        "streaming Onyx response body read failed"
                    );
                    break;
                }
            };

            let chunk_text = String::from_utf8_lossy(&chunk).to_string();
            raw_stream_body.push_str(&chunk_text);
            buffer.push_str(&chunk_text);

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
        append_upstream_audit_record(
            &settings_for_stream,
            "send-chat-message",
            "stream_response_complete",
            &cookie_for_stream,
            &payload_for_stream,
            Some(200),
            &serde_json::json!({"body": raw_stream_body}),
        )
        .await;
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

    let response = match client
        .post(format!("{}/api/chat/create-chat-session", settings.onyx_base_url))
        .header("Cookie", format!("fastapiusersauth={cookie}"))
        .header("Origin", &settings.onyx_origin_url)
        .header("Referer", &settings.onyx_referer)
        .json(&payload)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(err) => {
            let wrapped = anyhow!(err).context("failed to call create-chat-session");
            append_upstream_error_record(
                settings,
                "create-chat-session",
                "request_send",
                cookie,
                &payload,
                None,
                &format_error_chain(&wrapped),
            )
            .await;
            return Err(wrapped);
        }
    };

    if response.status() == reqwest::StatusCode::UNAUTHORIZED
        || response.status() == reqwest::StatusCode::FORBIDDEN
    {
        let err = anyhow!("onyx auth failed: {}", response.status().as_u16());
        append_upstream_audit_record(
            settings,
            "create-chat-session",
            "http_auth",
            cookie,
            &payload,
            Some(response.status().as_u16()),
            &serde_json::json!({"error": err.to_string()}),
        )
        .await;
        append_upstream_error_record(
            settings,
            "create-chat-session",
            "http_auth",
            cookie,
            &payload,
            Some(response.status().as_u16()),
            &err.to_string(),
        )
        .await;
        return Err(err);
    }

    if !response.status().is_success() {
        let code = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();
        let err = anyhow!("onyx create-chat-session HTTP {code}: {body}");
        append_upstream_audit_record(
            settings,
            "create-chat-session",
            "http_status",
            cookie,
            &payload,
            Some(code),
            &serde_json::json!({"error": body}),
        )
        .await;
        append_upstream_error_record(
            settings,
            "create-chat-session",
            "http_status",
            cookie,
            &payload,
            Some(code),
            &err.to_string(),
        )
        .await;
        return Err(err);
    }

    let data: Value = response
        .json()
        .await
        .context("invalid create-chat-session response json")?;
    append_upstream_audit_record(
        settings,
        "create-chat-session",
        "response_success",
        cookie,
        &payload,
        Some(200),
        &data,
    )
    .await;

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

// ----- Prompt-based tool calling emulation -----

#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Build a prompt that includes tool definitions as text instructions.
/// When tools are present, inject a system instruction telling the LLM
/// to use `<tool_call>` markers for tool invocations.
pub fn build_prompt_with_tools(
    messages: &[ChatMessage],
    tools: &[ClaudeToolDefinition],
) -> String {
    let mut parts = Vec::new();

    // Inject tool definitions as a system instruction
    if !tools.is_empty() {
        let mut tool_section = String::from(
            "system: You have access to the following tools. When you need to use a tool, you MUST respond with a tool call in this EXACT format (including the XML tags):\n\n\
             <tool_call>\n\
             {\"name\": \"tool_name\", \"arguments\": {\"param1\": \"value1\"}}\n\
             </tool_call>\n\n\
             You can make multiple tool calls by using multiple <tool_call> blocks.\n\
             If you don't need to use any tool, respond normally with text.\n\
             IMPORTANT: Always use the <tool_call> tags, never call tools in any other format.\n\n\
             Available tools:\n",
        );

        for tool in tools {
            tool_section.push_str(&format!("- {}", tool.name));
            if let Some(desc) = &tool.description {
                tool_section.push_str(&format!(": {}", desc));
            }
            tool_section.push('\n');
            if let Some(schema) = &tool.input_schema {
                tool_section.push_str(&format!("  Parameters: {}\n", schema));
            }
        }

        parts.push(tool_section);
    }

    // Format messages, handling tool results specially
    for m in messages {
        let content = m.content.trim();
        if content.is_empty() {
            continue;
        }
        parts.push(format!("{}: {}", m.role, content));
    }

    parts.join("\n")
}

/// Extract tool calls from LLM response text that uses `<tool_call>` markers.
/// Returns a list of parsed tool calls and the remaining text with markers removed.
pub fn extract_tool_calls_from_text(text: &str) -> (Vec<ParsedToolCall>, String) {
    let mut tool_calls = Vec::new();
    let mut remaining = String::new();
    let mut search_from = 0;

    while search_from < text.len() {
        let rest = &text[search_from..];
        if let Some(start_pos) = rest.find("<tool_call>") {
            // Add text before the marker to remaining
            remaining.push_str(&rest[..start_pos]);

            let after_tag = &rest[start_pos + "<tool_call>".len()..];
            if let Some(end_pos) = after_tag.find("</tool_call>") {
                let json_str = after_tag[..end_pos].trim();
                let block_contains_nested_tags = json_str.contains("<tool_call>")
                    || json_str.contains("</tool_call>");
                if !block_contains_nested_tags && let Some(tc) = parse_tool_call_json(json_str) {
                    tool_calls.push(tc);
                } else {
                    // Failed to parse — keep original text
                    remaining.push_str(&rest[start_pos..start_pos + "<tool_call>".len() + end_pos + "</tool_call>".len()]);
                }
                search_from += start_pos + "<tool_call>".len() + end_pos + "</tool_call>".len();
            } else {
                // No closing tag — keep as-is
                remaining.push_str(&rest[start_pos..]);
                search_from = text.len();
            }
        } else {
            remaining.push_str(rest);
            break;
        }
    }

    let remaining = remaining.trim().to_string();
    (tool_calls, remaining)
}

fn value_contains_tool_call_marker(value: &Value) -> bool {
    match value {
        Value::String(s) => s.contains("<tool_call>") || s.contains("</tool_call>"),
        Value::Array(items) => items.iter().any(value_contains_tool_call_marker),
        Value::Object(map) => map.values().any(value_contains_tool_call_marker),
        _ => false,
    }
}

fn parsed_tool_call_is_reasonable(name: &str, arguments: &Value) -> bool {
    match name {
        "bash" => arguments
            .get("command")
            .and_then(Value::as_str)
            .map(|command| {
                command.len() <= MAX_BASH_COMMAND_LEN
                    && !command.contains("<path>")
                    && !command.contains("<content>")
                    && !looks_like_line_annotated_source_dump(command)
            })
            .unwrap_or(false),
        _ => true,
    }
}

fn looks_like_line_annotated_source_dump(input: &str) -> bool {
    let normalized = input.replace("\\n", "\n");
    let mut matched_lines = 0;
    for line in normalized.lines() {
        let trimmed = line.trim_start();
        let digit_count = trimmed.chars().take_while(|c| c.is_ascii_digit()).count();
        if digit_count > 0 {
            let rest = &trimmed[digit_count..];
            if rest.starts_with('#') && rest.contains('|') {
                matched_lines += 1;
                if matched_lines >= 2 {
                    return true;
                }
            }
        }
    }
    false
}

fn parse_tool_call_json(json_str: &str) -> Option<ParsedToolCall> {
    if json_str.len() > MAX_TOOL_CALL_JSON_LEN {
        return None;
    }

    let v: Value = serde_json::from_str(json_str).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    if name.is_empty() {
        return None;
    }
    let arguments = v
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    if value_contains_tool_call_marker(&arguments) {
        return None;
    }
    if !parsed_tool_call_is_reasonable(&name, &arguments) {
        return None;
    }
    Some(ParsedToolCall { name, arguments })
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
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::config::Settings;
    use crate::models::ChatMessage;
    use serde_json::json;

    use super::{
        append_upstream_audit_record, append_upstream_error_record, build_prompt,
        is_retryable_upstream_io_error,
        parse_onyx_stream_text,
    };

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
    fn retryable_upstream_io_error_detects_chunked_unexpected_eof_chain() {
        let err = anyhow!("unexpected EOF during chunk size line")
            .context("error reading a body from connection")
            .context("request or response body error")
            .context("error decoding response body")
            .context("failed to read response body");

        assert!(is_retryable_upstream_io_error(&err));
    }

    #[test]
    fn retryable_upstream_io_error_ignores_auth_failure() {
        let err = anyhow!("onyx auth failed: 401");
        assert!(!is_retryable_upstream_io_error(&err));
    }

    #[tokio::test]
    async fn append_upstream_error_record_writes_jsonl_entry() {
        let mut settings = Settings::default();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rust_proxy_upstream_error_{ts}.jsonl"));
        settings.request_error_log_path = path.to_string_lossy().to_string();

        let payload = json!({"message": "hello", "chat_session_id": "abc"});
        append_upstream_error_record(
            &settings,
            "send-chat-message",
            "response_read",
            "cookie-123456",
            &payload,
            Some(200),
            "failed to read response body",
        )
        .await;

        let content = tokio::fs::read_to_string(&path)
            .await
            .expect("log file should be written");
        assert!(content.contains("\"endpoint\":\"send-chat-message\""));
        assert!(content.contains("\"stage\":\"response_read\""));
        assert!(content.contains("\"payload\":{"));

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn append_upstream_audit_record_writes_jsonl_entry() {
        let mut settings = Settings::default();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rust_proxy_upstream_audit_{ts}.jsonl"));
        settings.request_audit_log_path = path.to_string_lossy().to_string();

        let payload = json!({"message": "hello", "chat_session_id": "abc"});
        let response = json!({"content": "world"});
        append_upstream_audit_record(
            &settings,
            "send-chat-message",
            "response_success",
            "cookie-123456",
            &payload,
            Some(200),
            &response,
        )
        .await;

        let content = tokio::fs::read_to_string(&path)
            .await
            .expect("audit log file should be written");
        assert!(content.contains("\"endpoint\":\"send-chat-message\""));
        assert!(content.contains("\"response\":{"));
        assert!(content.contains("\"content\":\"world\""));

        let _ = tokio::fs::remove_file(&path).await;
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

    #[test]
    fn extract_tool_calls_parses_single_tool_call() {
        let text = "Let me check that for you.\n\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls -la\"}}\n</tool_call>";
        let (calls, remaining) = super::extract_tool_calls_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arguments, json!({"command": "ls -la"}));
        assert_eq!(remaining, "Let me check that for you.");
    }

    #[test]
    fn extract_tool_calls_parses_multiple_tool_calls() {
        let text = "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"pwd\"}}\n</tool_call>\n\n<tool_call>\n{\"name\": \"grep\", \"arguments\": {\"pattern\": \"TODO\", \"path\": \".\"}}\n</tool_call>";
        let (calls, remaining) = super::extract_tool_calls_from_text(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[1].name, "grep");
        assert!(remaining.is_empty());
    }

    #[test]
    fn extract_tool_calls_returns_empty_for_plain_text() {
        let text = "Here is your answer: the file contains 42 lines.";
        let (calls, remaining) = super::extract_tool_calls_from_text(text);
        assert!(calls.is_empty());
        assert_eq!(remaining, text);
    }

    #[test]
    fn extract_tool_calls_handles_malformed_json() {
        let text = "<tool_call>\nnot valid json\n</tool_call>";
        let (calls, remaining) = super::extract_tool_calls_from_text(text);
        assert!(calls.is_empty());
        // malformed content is kept as-is
        assert!(!remaining.is_empty());
    }

    #[test]
    fn extract_tool_calls_ignores_nested_tool_call_markers_inside_arguments() {
        let text = r#"<tool_call>
{"name": "bash", "arguments": {"command": "echo before <tool_call> embedded"}}
</tool_call>"#;
        let (calls, remaining) = super::extract_tool_calls_from_text(text);
        assert!(calls.is_empty(), "nested marker content should not be parsed as a valid tool call");
        assert!(!remaining.is_empty(), "nested marker block should be preserved as plain text");
    }

    #[test]
    fn extract_tool_calls_ignores_oversized_tool_call_blocks() {
        let huge = "x".repeat(20_000);
        let text = format!(
            "<tool_call>\n{{\"name\": \"bash\", \"arguments\": {{\"command\": \"{}\"}}}}\n</tool_call>",
            huge
        );
        let (calls, remaining) = super::extract_tool_calls_from_text(&text);
        assert!(calls.is_empty(), "oversized blocks should be rejected to avoid corrupted extraction");
        assert!(!remaining.is_empty(), "oversized block should remain in plain text output");
    }

    #[test]
    fn extract_tool_calls_rejects_suspiciously_large_bash_command() {
        let polluted = format!(
            "pwd\\\"}} trailing {}",
            "A".repeat(3000)
        );
        let text = format!(
            "<tool_call>\n{{\"name\":\"bash\",\"arguments\":{{\"command\":\"{}\"}}}}\n</tool_call>",
            polluted
        );
        let (calls, remaining) = super::extract_tool_calls_from_text(&text);
        assert!(calls.is_empty(), "oversized bash command payload should be rejected");
        assert!(!remaining.is_empty(), "rejected bash block should remain in plain text output");
    }

    #[test]
    fn extract_tool_calls_rejects_line_annotated_source_dump_in_bash_command() {
        let polluted = "pwd\n722#JV| some rust source\n723#XQ| another line";
        let text = format!(
            "<tool_call>\n{{\"name\":\"bash\",\"arguments\":{{\"command\":\"{}\"}}}}\n</tool_call>",
            polluted.replace('\n', "\\n")
        );
        let (calls, remaining) = super::extract_tool_calls_from_text(&text);
        assert!(calls.is_empty(), "line-annotated source dump should be rejected as bash command");
        assert!(!remaining.is_empty(), "rejected polluted block should remain as plain text");
    }

    #[test]
    fn build_prompt_with_tools_injects_definitions() {
        use crate::models::ClaudeToolDefinition;

        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "list files".to_string(),
            },
        ];
        let tools = vec![
            ClaudeToolDefinition {
                name: "bash".to_string(),
                description: Some("Execute shell commands".to_string()),
                input_schema: Some(json!({"type": "object", "properties": {"command": {"type": "string"}}})),
            },
        ];
        let prompt = super::build_prompt_with_tools(&messages, &tools);
        assert!(prompt.contains("<tool_call>"), "should contain tool_call instruction");
        assert!(prompt.contains("bash"), "should contain tool name");
        assert!(prompt.contains("Execute shell commands"), "should contain description");
        assert!(prompt.contains("user: list files"), "should contain user message");
    }
}
