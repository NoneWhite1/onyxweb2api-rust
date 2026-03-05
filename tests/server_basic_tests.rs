use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use serde_json::Value;
use rust_proxy::{
    config::Settings,
    server::{build_router, build_state},
};
use tower::util::ServiceExt;

fn test_router() -> axum::Router {
    let state = build_state(Settings {
        onyx_auth_cookie: "test-cookie-1,test-cookie-2".to_string(),
        api_key: None,
        ..Settings::default()
    })
    .expect("state should build");
    build_router(state)
}

fn test_router_with_api_key() -> axum::Router {
    let state = build_state(Settings {
        onyx_auth_cookie: "test-cookie-1,test-cookie-2".to_string(),
        api_key: Some("secret-key".to_string()),
        ..Settings::default()
    })
    .expect("state should build");
    build_router(state)
}

fn test_router_with_invalid_upstream() -> axum::Router {
    let state = build_state(Settings {
        onyx_base_url: "http://127.0.0.1:9".to_string(),
        onyx_auth_cookie: "test-cookie-1".to_string(),
        request_timeout_secs: 1,
        api_key: None,
        ..Settings::default()
    })
    .expect("state should build");
    build_router(state)
}

#[tokio::test]
async fn health_endpoint_returns_ok() {
    let app = test_router();

    let response = app
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn root_endpoint_returns_ok() {
    let app = test_router();

    let response = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn ui_endpoint_returns_ok() {
    let app = test_router();

    let response = app
        .oneshot(Request::builder().uri("/ui").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn status_endpoint_returns_ok() {
    let app = test_router();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn status_endpoint_requires_api_key_when_configured() {
    let app = test_router_with_api_key();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn status_endpoint_accepts_valid_bearer_api_key() {
    let app = test_router_with_api_key();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/status")
                .header(header::AUTHORIZATION, "Bearer secret-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn v1_models_endpoint_returns_ok() {
    let app = test_router();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn chat_completions_validates_messages() {
    let app = test_router();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"model":"gpt-4o","messages":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn add_cookie_endpoint_returns_ok() {
    let app = test_router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/cookies")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"cookie":"fastapiusersauth=added-cookie-1"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn delete_cookie_endpoint_returns_ok() {
    let app = test_router();

    let list_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/cookies")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(list_resp.status(), StatusCode::OK);

    let body = to_bytes(list_resp.into_body(), usize::MAX).await.unwrap();
    let data: Vec<Value> = serde_json::from_slice(&body).unwrap();
    let fingerprint = data[0]["fingerprint"].as_str().unwrap().to_string();

    let delete_resp = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/api/cookies/{fingerprint}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(delete_resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn chat_completions_stream_no_longer_returns_bad_request() {
    let app = test_router();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"model":"gpt-5.2","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    // Should NOT be BAD_REQUEST (streaming is now implemented)
    // It will be BAD_GATEWAY because there's no real Onyx backend in tests,
    // but that proves the code path progresses past the old stream rejection.
    assert_ne!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn chat_completions_stream_with_api_key() {
    let app = test_router_with_api_key();

    // Without API key → UNAUTHORIZED
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    // With valid API key → not BAD_REQUEST (streaming accepted)
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::AUTHORIZATION, "Bearer secret-key")
                .body(Body::from(
                    r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn chat_completions_stream_validates_empty_messages() {
    let app = test_router();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"model":"gpt-4o","messages":[],"stream":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    // Empty messages should still be rejected with BAD_REQUEST
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn auth_login_page_returns_html() {
    let app = test_router();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/auth/login")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 65536)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&body);
    assert!(text.contains("Get Cookie"), "login page should contain title");
    assert!(text.contains("loginForm"), "login page should contain login form");
}

#[tokio::test]
async fn auth_login_post_validates_upstream_failure() {
    let app = test_router();

    // POST login endpoint — will fail because there's no real Onyx backend,
    // but should attempt the proxy (not just 404 or panic)
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/auth/login")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"email":"test@test.com","password":"testpass"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    // test_router uses "https://cloud.onyx.app" so request reaches real Onyx,
    // which returns 400 for bad creds. Our handler relays as BAD_REQUEST.
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn auth_login_requires_api_key_when_configured() {
    let app = test_router_with_api_key();

    // Without API key
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/auth/login")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"email":"test@test.com","password":"testpass"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    // With valid API key
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/auth/login")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::AUTHORIZATION, "Bearer secret-key")
                .body(Body::from(
                    r#"{"email":"test@test.com","password":"testpass"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    // With valid API key — reaches real Onyx, gets 400 for bad creds
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

// ----- Claude /v1/messages tests -----

#[tokio::test]
async fn claude_messages_validates_empty_messages() {
    let app = test_router();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"model":"claude-opus-4.6","messages":[],"max_tokens":1024}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(response.into_body(), 65536)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["type"], "error");
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn claude_messages_requires_api_key_when_configured() {
    let app = test_router_with_api_key();

    // Without API key
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"model":"gpt-5.2","messages":[{"role":"user","content":"hi"}],"max_tokens":1024}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    // With x-api-key header
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::CONTENT_TYPE, "application/json")
                .header("x-api-key", "secret-key")
                .body(Body::from(
                    r#"{"model":"gpt-5.2","messages":[{"role":"user","content":"hi"}],"max_tokens":1024}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    // Should pass auth and attempt upstream (not UNAUTHORIZED)
    assert_ne!(response.status(), StatusCode::UNAUTHORIZED);

    // With Bearer token (also accepted)
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::AUTHORIZATION, "Bearer secret-key")
                .body(Body::from(
                    r#"{"model":"gpt-5.2","messages":[{"role":"user","content":"hi"}],"max_tokens":1024}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn claude_messages_non_streaming_returns_claude_format() {
    let app = test_router();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"model":"gpt-5.2","messages":[{"role":"user","content":"hello"}],"max_tokens":1024}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    // Non-streaming will attempt upstream call. With real Onyx, it succeeds;
    // in tests (using real cloud.onyx.app but with test-cookie), it may fail with BAD_GATEWAY.
    // The important thing is it doesn't return UNAUTHORIZED or BAD_REQUEST (the request is valid).
    let status = response.status();
    assert_ne!(status, StatusCode::UNAUTHORIZED);
    assert_ne!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn claude_messages_upstream_error_includes_reason_chain() {
    let app = test_router_with_invalid_upstream();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"model":"gpt-5.2","messages":[{"role":"user","content":"hello"}],"max_tokens":1024}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);

    let body = axum::body::to_bytes(response.into_body(), 65536)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let message = json["error"]["message"].as_str().unwrap_or_default();

    assert!(
        message.contains("chain:"),
        "error message should include causal chain: {message}"
    );
    assert!(
        message.contains("root_cause:"),
        "error message should include explicit root cause: {message}"
    );
}

#[tokio::test]
async fn claude_messages_tools_payload_with_system_blocks_does_not_422() {
    let app = test_router_with_invalid_upstream();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{
                        "model":"claude-opus-4.6",
                        "system":[{"type":"text","text":"You are a tool-using assistant."}],
                        "messages":[
                          {
                            "role":"assistant",
                            "content":[
                              {"type":"tool_use","id":"toolu_1","name":"shell","input":{"cmd":"pwd"}}
                            ]
                          },
                          {
                            "role":"user",
                            "content":[
                              {"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"/home/nonewhite"}]}
                            ]
                          }
                        ],
                        "tools":[{"name":"shell","description":"run shell","input_schema":{"type":"object"}}],
                        "max_tokens":1024
                    }"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
}

#[tokio::test]
async fn claude_messages_content_object_shape_does_not_422() {
    let app = test_router_with_invalid_upstream();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{
                        "model":"claude-opus-4.6",
                        "messages":[
                          {
                            "role":"user",
                            "content":{"type":"text","text":"hello from object content"}
                          }
                        ],
                        "max_tokens":1024
                    }"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
}

#[tokio::test]
async fn claude_messages_tool_result_object_content_does_not_422() {
    let app = test_router_with_invalid_upstream();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{
                        "model":"claude-opus-4.6",
                        "messages":[
                          {
                            "role":"assistant",
                            "content":[{"type":"tool_use","id":"toolu_1","name":"shell","input":{"cmd":"pwd"}}]
                          },
                          {
                            "role":"user",
                            "content":[{"type":"tool_result","tool_use_id":"toolu_1","content":{"text":"/home/nonewhite"}}]
                          }
                        ],
                        "max_tokens":1024
                    }"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
}

#[tokio::test]
async fn claude_messages_invalid_json_returns_reason_text() {
    let app = test_router();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{
                        "model":"claude-opus-4.6",
                        "system":{"type":"text","text":"invalid-shape"},
                        "messages":[{"role":"user","content":"hello"}],
                        "max_tokens":1024
                    }"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);

    let body = axum::body::to_bytes(response.into_body(), 65536)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let message = json["error"]["message"].as_str().unwrap_or_default();

    assert!(
        message.contains("invalid request body:"),
        "422 response should include parse reason prefix: {message}"
    );
}

#[tokio::test]
async fn claude_messages_streaming() {
    let app = test_router();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"model":"gpt-5.2","messages":[{"role":"user","content":"hello"}],"max_tokens":1024,"stream":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    // Streaming: should not be BAD_REQUEST (streaming is implemented)
    let status = response.status();
    assert_ne!(status, StatusCode::BAD_REQUEST);
    assert_ne!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn claude_messages_streaming_local_tool_use_returns_tool_use_events() {
    let app = test_router();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{
                        "model":"claude-opus-4.6",
                        "messages":[{"role":"user","content":"Use the bash tool to run: pwd"}],
                        "tools":[{"name":"bash","description":"run shell","input_schema":{"type":"object"}}],
                        "max_tokens":1024,
                        "stream":true
                    }"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), 65536).await.unwrap();
    let sse = String::from_utf8_lossy(&body);

    assert!(sse.contains("event: message_start"), "missing message_start event: {sse}");
    assert!(
        sse.contains("\"type\":\"tool_use\""),
        "missing tool_use content block: {sse}"
    );
    assert!(sse.contains("\"name\":\"bash\""), "missing bash tool name: {sse}");
    assert!(
        sse.contains("\"command\":\"pwd\""),
        "missing tool input command: {sse}"
    );
    assert!(
        sse.contains("\"stop_reason\":\"tool_use\""),
        "missing tool_use stop reason: {sse}"
    );
    assert!(sse.contains("event: message_stop"), "missing message_stop event: {sse}");
}

#[tokio::test]
async fn claude_messages_streaming_write_tool_use_for_english_create_file_prompt() {
    let app = test_router();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{
                        "model":"claude-opus-4.6",
                        "messages":[{"role":"user","content":"Please create 1234.txt with content 1234"}],
                        "tools":[{"name":"write","description":"write file","input_schema":{"type":"object"}}],
                        "max_tokens":1024,
                        "stream":true
                    }"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), 65536).await.unwrap();
    let sse = String::from_utf8_lossy(&body);

    assert!(sse.contains("event: message_start"), "missing message_start event: {sse}");
    assert!(
        sse.contains("\"type\":\"tool_use\""),
        "missing tool_use content block: {sse}"
    );
    assert!(sse.contains("\"name\":\"write\""), "missing write tool name: {sse}");
    assert!(
        sse.contains("\"filePath\":\"1234.txt\""),
        "missing expected filePath in tool input: {sse}"
    );
    assert!(
        sse.contains("\"content\":\"1234\""),
        "missing expected content in tool input: {sse}"
    );
    assert!(
        sse.contains("\"stop_reason\":\"tool_use\""),
        "missing tool_use stop reason: {sse}"
    );
    assert!(sse.contains("event: message_stop"), "missing message_stop event: {sse}");
}

#[tokio::test]
async fn claude_messages_streaming_prefers_write_when_bash_and_write_both_available() {
    let app = test_router();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{
                        "model":"claude-opus-4.6",
                        "messages":[{"role":"user","content":"Please create 1234.txt with content 1234"}],
                        "tools":[
                            {"name":"bash","description":"run shell","input_schema":{"type":"object"}},
                            {"name":"write","description":"write file","input_schema":{"type":"object"}}
                        ],
                        "max_tokens":1024,
                        "stream":true
                    }"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), 65536).await.unwrap();
    let sse = String::from_utf8_lossy(&body);

    assert!(
        sse.contains("\"type\":\"tool_use\""),
        "missing tool_use content block: {sse}"
    );
    assert!(
        sse.contains("\"name\":\"write\""),
        "expected write tool when both write and bash available: {sse}"
    );
    assert!(
        !sse.contains("\"name\":\"bash\""),
        "should not pick bash for this create-file prompt when write is available: {sse}"
    );
    assert!(
        sse.contains("\"stop_reason\":\"tool_use\""),
        "missing tool_use stop reason: {sse}"
    );
}
