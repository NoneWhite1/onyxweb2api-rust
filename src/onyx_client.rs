use anyhow::{Context, anyhow};
use futures_util::StreamExt;
use serde::Serialize;
use serde_json::Value;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;

use crate::{
    config::Settings,
    models::{ChatMessage, DEFAULT_MODEL},
    pseudo_tools::strip_noop_trailer,
};

const MAX_SEND_CHAT_ATTEMPTS: usize = 3;
const SEND_CHAT_RETRY_BACKOFF_MS: u64 = 250;
const STREAM_TRAILER_HOLDBACK_CHARS: usize = 512;

pub async fn full_chat(
    client: &reqwest::Client,
    settings: &Settings,
    cookie: &str,
    messages: &[ChatMessage],
    model_name: &str,
) -> anyhow::Result<(String, String)> {
    let chat_session_id = create_chat_session(client, settings, cookie).await?;
    let (provider, version) = resolve_model(model_name);
    let payload =
        build_send_chat_payload(messages, &chat_session_id, &provider, &version, settings);

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
            tool_call_id: None,
            name: None,
            tool_calls: None,
        });

        let retry_payload = build_send_chat_payload(
            &retry_messages,
            &chat_session_id,
            &provider,
            &version,
            settings,
        );

        if let Ok(retry_text) =
            send_chat_message_text(client, settings, cookie, &retry_payload).await
        {
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
            .post(format!(
                "{}/api/chat/send-chat-message",
                settings.onyx_base_url
            ))
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
                    if attempt < MAX_SEND_CHAT_ATTEMPTS && is_retryable_upstream_io_error(&wrapped)
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
            if let Err(err) = file.write_all(format!("{serialized}\n").as_bytes()).await {
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
            if let Err(err) = file.write_all(format!("{serialized}\n").as_bytes()).await {
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
) -> anyhow::Result<mpsc::Receiver<StreamEvent>> {
    let chat_session_id = create_chat_session(client, settings, cookie).await?;
    let (provider, version) = resolve_model(model_name);
    let payload =
        build_send_chat_payload(messages, &chat_session_id, &provider, &version, settings);

    let response = match client
        .post(format!(
            "{}/api/chat/send-chat-message",
            settings.onyx_base_url
        ))
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
        let mut pending_content = String::new();
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
                            pending_content.push_str(c);
                            flush_safe_stream_content(&tx, &mut pending_content).await;
                        }
                    }
                    Some("stop") => {
                        finalize_stream_content(&tx, &mut pending_content).await;
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
        finalize_stream_content(&tx, &mut pending_content).await;
        let _ = tx.send(StreamEvent::Done).await;
    });

    Ok(rx)
}

async fn flush_safe_stream_content(tx: &mpsc::Sender<StreamEvent>, pending_content: &mut String) {
    let char_count = pending_content.chars().count();
    if char_count <= STREAM_TRAILER_HOLDBACK_CHARS {
        return;
    }

    let safe_chars = char_count - STREAM_TRAILER_HOLDBACK_CHARS;
    let split_idx = nth_char_byte_index(pending_content, safe_chars);
    let safe_prefix = pending_content[..split_idx].to_string();
    let safe_prefix = safe_prefix.trim_end_matches(|c: char| c == '\0');
    if !safe_prefix.is_empty() {
        let _ = tx.send(StreamEvent::Content(safe_prefix.to_string())).await;
    }
    *pending_content = pending_content[split_idx..].to_string();
}

async fn finalize_stream_content(tx: &mpsc::Sender<StreamEvent>, pending_content: &mut String) {
    if pending_content.is_empty() {
        return;
    }

    let final_content =
        strip_noop_trailer(pending_content).unwrap_or_else(|| pending_content.clone());
    if !final_content.is_empty() {
        let _ = tx.send(StreamEvent::Content(final_content)).await;
    }
    pending_content.clear();
}

fn nth_char_byte_index(s: &str, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    s.char_indices()
        .nth(n)
        .map(|(idx, _)| idx)
        .unwrap_or(s.len())
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
        "llm_override": {
            "temperature": 0.5,
            "model_provider": provider,
            "model_version": version
        },
        "origin": settings.onyx_origin,
    })
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
        .post(format!(
            "{}/api/chat/create-chat-session",
            settings.onyx_base_url
        ))
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
    use tokio::sync::mpsc;

    use crate::config::Settings;
    use crate::models::ChatMessage;
    use serde_json::json;

    use super::{
        append_upstream_audit_record, append_upstream_error_record, build_prompt,
        is_retryable_upstream_io_error, parse_onyx_stream_text,
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
        let record: serde_json::Value = serde_json::from_str(
            content
                .lines()
                .find(|line| !line.trim().is_empty())
                .unwrap_or("{}"),
        )
        .expect("error log should contain valid json");
        assert_eq!(record["endpoint"], "send-chat-message");
        assert_eq!(record["stage"], "response_read");
        assert_eq!(record["payload"]["chat_session_id"], "abc");

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
        let record: serde_json::Value = serde_json::from_str(
            content
                .lines()
                .find(|line| !line.trim().is_empty())
                .unwrap_or("{}"),
        )
        .expect("audit log should contain valid json");
        assert_eq!(record["endpoint"], "send-chat-message");
        assert_eq!(record["stage"], "response_success");
        assert_eq!(record["response"]["content"], "world");

        let _ = tokio::fs::remove_file(&path).await;
    }

    #[test]
    fn build_prompt_skips_empty_content_messages() {
        let prompt = build_prompt(&[
            ChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
                tool_call_id: None,
                name: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: String::new(),
                tool_call_id: None,
                name: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "result text".to_string(),
                tool_call_id: None,
                name: None,
                tool_calls: None,
            },
        ]);

        assert_eq!(prompt, "user: hello\nuser: result text");
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

    #[tokio::test]
    async fn finalize_stream_content_strips_noop_trailer() {
        let (tx, mut rx) = mpsc::channel(8);
        let mut pending = format!(
            "Visible text.\n<<<ONYX_TOOL_CALL_9F4C2E7A6B1D8C3E5A0F7B2D4C6E8A1F>>><onyx_tool_call_9f4c2e7a6b1d8c3e5a0f7b2d4c6e8a1f>{{\"action\":\"proxy_noop\",\"action_input\":{{\"status\":\"final\"}}}}</onyx_tool_call_9f4c2e7a6b1d8c3e5a0f7b2d4c6e8a1f>"
        );

        super::finalize_stream_content(&tx, &mut pending).await;

        match rx.recv().await {
            Some(super::StreamEvent::Content(text)) => assert_eq!(text, "Visible text."),
            other => panic!("expected content event, got {other:?}"),
        }
        assert!(pending.is_empty());
    }

    #[tokio::test]
    async fn flush_safe_stream_content_holds_back_possible_trailer() {
        let (tx, mut rx) = mpsc::channel(8);
        let mut pending = format!(
            "{}prefix{}",
            "A".repeat(super::STREAM_TRAILER_HOLDBACK_CHARS + 16),
            "<<<ONYX_TOOL_CALL_9F4C2E7A6B1D8C3E5A0F7B2D4C6E8A1F>>>"
        );

        super::flush_safe_stream_content(&tx, &mut pending).await;

        match rx.recv().await {
            Some(super::StreamEvent::Content(text)) => {
                assert!(text.contains(&"A".repeat(16)));
                assert!(!text.contains("<<<ONYX_TOOL_CALL_9F4C2E7A6B1D8C3E5A0F7B2D4C6E8A1F>>>"));
            }
            other => panic!("expected content event, got {other:?}"),
        }
        assert!(pending.contains("<<<ONYX_TOOL_CALL_9F4C2E7A6B1D8C3E5A0F7B2D4C6E8A1F>>>"));
    }
}
