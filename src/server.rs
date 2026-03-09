use std::convert::Infallible;
use std::error::Error as StdError;
use std::path::Path as FsPath;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{Path, State},
    http::{HeaderMap, StatusCode, header::AUTHORIZATION},
    response::{Html, IntoResponse, Sse, sse::Event},
    routing::{delete, get, post},
};
use futures_util::stream;
use reqwest::header::SET_COOKIE;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::error;

use crate::{
    config::Settings,
    cookie_manager::{CookieFailureKind, CookieManager},
    models::{
        AssistantMessage, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
        ChatMessage, Choice, ChunkChoice, ChunkDelta, ClaudeContentBlock, ClaudeInputJsonDelta,
        ClaudeMessagesRequest, ClaudeMessagesResponse, ClaudeStopDelta,
        ClaudeStreamContentBlockDelta, ClaudeStreamContentBlockStart, ClaudeStreamContentBlockStop,
        ClaudeStreamMessageDelta, ClaudeStreamMessageMeta, ClaudeStreamMessageStart,
        ClaudeStreamMessageStop, ClaudeTextDelta, ClaudeUsage, DEFAULT_MODEL, ModelItem,
        ModelsListResponse, Usage, supported_models,
    },
    onyx_client::{self, StreamEvent},
    pseudo_tools::{self, ParsedPseudoToolResponse, ToolPromptContext, ValidationFailure},
};

#[derive(Clone)]
pub struct AppState {
    pub settings: Settings,
    pub cookie_manager: Arc<RwLock<CookieManager>>,
    pub rr_counter: Arc<AtomicUsize>,
    pub http_client: reqwest::Client,
    pub started_at_ts: u64,
}

#[derive(Debug, Serialize)]
struct StatusResponse {
    status: &'static str,
    uptime_secs: u64,
    onyx_base_url: String,
    cookie_total: usize,
    cookie_active: usize,
    cookie_exhausted: usize,
}

#[derive(Debug, Serialize)]
struct RefreshResponse {
    total: usize,
    refreshed: usize,
    failed: usize,
}

const COOKIE_REFRESH_INTERVAL_SECS: u64 = 3600;

#[derive(Debug, Deserialize)]
struct AddCookieRequest {
    cookie: String,
}

#[derive(Debug, Deserialize)]
struct LoginRequest {
    email: String,
    password: String,
}

#[derive(Debug, Serialize)]
struct MutationResponse {
    ok: bool,
}

#[derive(Debug, Serialize)]
struct ProxyAuditRecord {
    ts_ms: u64,
    endpoint: String,
    status: u16,
    raw_request_body: Option<String>,
    request: serde_json::Value,
    response: serde_json::Value,
}

pub fn build_state(settings: Settings) -> anyhow::Result<AppState> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(
            settings.request_timeout_secs,
        ))
        .build()?;

    Ok(AppState {
        cookie_manager: Arc::new(RwLock::new(CookieManager::load_or_create(
            &settings.cookie_persist_path,
            &settings.onyx_auth_cookie,
        ))),
        rr_counter: Arc::new(AtomicUsize::new(0)),
        http_client: client,
        started_at_ts: now_ts(),
        settings,
    })
}

fn rotate_cookie_values(values: Vec<String>, start_index: usize) -> Vec<String> {
    if values.is_empty() {
        return values;
    }

    let offset = start_index % values.len();
    if offset == 0 {
        return values;
    }

    let mut rotated = Vec::with_capacity(values.len());
    rotated.extend_from_slice(&values[offset..]);
    rotated.extend_from_slice(&values[..offset]);
    rotated
}

fn next_round_robin_offset(counter: &AtomicUsize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    counter.fetch_add(1, Ordering::Relaxed) % len
}

async fn active_cookie_values_round_robin(state: &AppState) -> Vec<String> {
    let values = {
        let cm = state.cookie_manager.read().await;
        cm.active_cookie_values()
    };
    let offset = next_round_robin_offset(&state.rr_counter, values.len());
    rotate_cookie_values(values, offset)
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(root_handler))
        .route("/health", get(health_handler))
        .route("/ui", get(ui_handler))
        .route("/api/status", get(status_handler))
        .route("/v1/models", get(v1_models_handler))
        .route("/v1/chat/completions", post(v1_chat_completions_handler))
        .route("/v1/messages", post(v1_messages_handler))
        .route(
            "/api/cookies",
            get(cookies_handler).post(add_cookie_handler),
        )
        .route("/api/cookies/{fingerprint}", delete(delete_cookie_handler))
        .route("/api/cookies/refresh", post(refresh_cookies_handler))
        .route(
            "/auth/login",
            get(auth_login_page_handler).post(auth_login_handler),
        )
        .with_state(state)
}

pub fn spawn_cookie_refresh_task(state: AppState) -> JoinHandle<()> {
    spawn_cookie_refresh_task_with_interval(
        state,
        Duration::from_secs(COOKIE_REFRESH_INTERVAL_SECS),
    )
}

fn spawn_cookie_refresh_task_with_interval(state: AppState, interval: Duration) -> JoinHandle<()> {
    tokio::spawn(async move {
        let initial = refresh_all_cookies(&state).await;
        error!(
            endpoint = "/api/cookies/refresh",
            total = initial.total,
            refreshed = initial.refreshed,
            failed = initial.failed,
            "startup cookie refresh completed"
        );

        let mut ticker = tokio::time::interval(interval);
        ticker.tick().await;
        loop {
            ticker.tick().await;
            let result = refresh_all_cookies(&state).await;
            error!(
                endpoint = "/api/cookies/refresh",
                total = result.total,
                refreshed = result.refreshed,
                failed = result.failed,
                "scheduled cookie refresh completed"
            );
        }
    })
}

async fn root_handler() -> Json<serde_json::Value> {
    Json(json!({ "message": "Onyx2OpenAI API is running" }))
}

async fn health_handler() -> Json<serde_json::Value> {
    Json(json!({ "status": "healthy" }))
}

async fn ui_handler() -> Html<&'static str> {
    Html(
        r#"<!doctype html>
        <!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Onyx Proxy Dashboard</title>
  <style>
    :root {
      --bg: #0f1117; --surface: #1a1d27; --surface2: #242736;
      --border: #2e3348; --text: #e1e4ed; --muted: #8b8fa3;
      --primary: #6c72cb; --primary-hover: #7b82d8;
      --green: #34d399; --red: #f87171; --yellow: #fbbf24;
      --radius: 10px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Inter', system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; padding: 24px; }
    .container { max-width: 960px; margin: 0 auto; }
    .header { display: flex; align-items: center; gap: 12px; margin-bottom: 28px; }
    .header h1 { font-size: 22px; font-weight: 600; }
    .header .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:.4; } }
    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap: 14px; margin-bottom: 24px; }
    .stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 18px; }
    .stat-card .label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .6px; margin-bottom: 6px; }
    .stat-card .value { font-size: 26px; font-weight: 700; }
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; margin-bottom: 18px; }
    .card-title { font-size: 14px; font-weight: 600; margin-bottom: 14px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; }
    .input-row { display: flex; gap: 10px; margin-bottom: 14px; }
    input[type=text], input[type=password] { flex: 1; background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; color: var(--text); font-size: 14px; outline: none; transition: border .2s; }
    input:focus { border-color: var(--primary); }
    button { background: var(--primary); color: #fff; border: none; border-radius: 8px; padding: 10px 18px; font-size: 13px; font-weight: 500; cursor: pointer; transition: background .2s, transform .1s; white-space: nowrap; }
    button:hover { background: var(--primary-hover); }
    button:active { transform: scale(.97); }
    button.secondary { background: var(--surface2); border: 1px solid var(--border); color: var(--text); }
    button.secondary:hover { background: var(--border); }
    button.danger { background: transparent; border: 1px solid var(--red); color: var(--red); padding: 6px 12px; font-size: 12px; }
    button.danger:hover { background: var(--red); color: #fff; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { text-align: left; padding: 10px 12px; color: var(--muted); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; border-bottom: 1px solid var(--border); }
    td { padding: 12px; border-bottom: 1px solid var(--border); }
    tr:last-child td { border-bottom: none; }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }
    .badge-ok { background: rgba(52,211,153,.15); color: var(--green); }
    .badge-err { background: rgba(248,113,113,.15); color: var(--red); }
    .mono { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 12px; color: var(--muted); }
    .actions-row { display: flex; gap: 10px; }
    .empty-msg { text-align: center; padding: 32px; color: var(--muted); }
    .toast { position: fixed; bottom: 24px; right: 24px; background: var(--surface2); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px 20px; font-size: 13px; opacity: 0; transition: opacity .3s; pointer-events: none; z-index: 99; }
    .toast.show { opacity: 1; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="dot"></div>
      <h1>Onyx Proxy Dashboard</h1>
    </div>

    <div class="stats">
      <div class="stat-card"><div class="label">Uptime</div><div class="value" id="uptime">-</div></div>
      <div class="stat-card"><div class="label">Active Cookies</div><div class="value" id="active" style="color:var(--green)">-</div></div>
      <div class="stat-card"><div class="label">Exhausted</div><div class="value" id="exhausted" style="color:var(--red)">-</div></div>
      <div class="stat-card"><div class="label">Total</div><div class="value" id="total">-</div></div>
    </div>

    <div class="card">
      <div class="card-title">Authentication</div>
      <div class="input-row">
        <input type="password" id="apiKeyInput" placeholder="API Key (leave empty if not required)" />
        <button id="saveApiKeyBtn" class="secondary">Save Key</button>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Add Cookie</div>
      <div class="input-row">
        <input type="text" id="cookieInput" placeholder="Paste fastapiusersauth cookie value..." />
        <button id="addBtn">Add</button>
      </div>
    </div>

    <div class="card">
      <div class="card-title" style="display:flex;justify-content:space-between;align-items:center;">
        <span>Cookie Pool</span>
        <button id="refreshBtn" class="secondary" style="padding:6px 14px;font-size:12px;">Refresh All</button>
      </div>
      <table>
        <thead><tr><th>Fingerprint</th><th>Preview</th><th>Status</th><th>Last Refresh</th><th>Error</th><th></th></tr></thead>
        <tbody id="cookies"></tbody>
      </table>
      <div class="empty-msg" id="emptyMsg" style="display:none">No cookies configured. Add one above.</div>
    </div>
  </div>

  <div class="toast" id="toast"></div>

  <script>
    function authHeaders(extra = {}) {
      const key = localStorage.getItem('proxy_api_key') || '';
      if (!key) return extra;
      return { ...extra, Authorization: `Bearer ${key}` };
    }

    async function apiFetch(url, options = {}) {
      const headers = authHeaders(options.headers || {});
      const resp = await fetch(url, { ...options, headers });
      if (resp.status === 401) { toast('Unauthorized: check your API key', true); throw new Error('unauthorized'); }
      return resp;
    }

    function toast(msg, isError = false) {
      const el = document.getElementById('toast');
      el.textContent = msg;
      el.style.borderColor = isError ? 'var(--red)' : 'var(--green)';
      el.classList.add('show');
      setTimeout(() => el.classList.remove('show'), 2500);
    }

    function fmtUptime(s) {
      if (s < 60) return s + 's';
      if (s < 3600) return Math.floor(s/60) + 'm ' + (s%60) + 's';
      const h = Math.floor(s/3600);
      return h + 'h ' + Math.floor((s%3600)/60) + 'm';
    }

    function fmtTs(ts) {
      if (!ts) return '-';
      return new Date(ts * 1000).toLocaleString();
    }

    async function load() {
      try {
        const status = await apiFetch('/api/status').then(r => r.json());
        document.getElementById('uptime').textContent = fmtUptime(status.uptime_secs);
        document.getElementById('active').textContent = status.cookie_active;
        document.getElementById('exhausted').textContent = status.cookie_exhausted;
        document.getElementById('total').textContent = status.cookie_total;

        const cookies = await apiFetch('/api/cookies').then(r => r.json());
        const tbody = document.getElementById('cookies');
        const emptyMsg = document.getElementById('emptyMsg');
        tbody.innerHTML = '';
        if (cookies.length === 0) { emptyMsg.style.display = ''; return; }
        emptyMsg.style.display = 'none';

        for (const c of cookies) {
          const tr = document.createElement('tr');
          const badge = c.exhausted
            ? '<span class="badge badge-err">Exhausted</span>'
            : '<span class="badge badge-ok">Active</span>';
          tr.innerHTML = `<td class="mono">${c.fingerprint.slice(0,12)}</td><td class="mono">${c.preview}</td><td>${badge}</td><td>${fmtTs(c.last_refresh_ts)}</td><td style="color:var(--red);font-size:12px;">${c.last_error ?? '-'}</td><td><button class="danger" data-fp="${c.fingerprint}">Delete</button></td>`;
          tbody.appendChild(tr);
        }

        for (const btn of tbody.querySelectorAll('button[data-fp]')) {
          btn.onclick = async () => {
            await apiFetch(`/api/cookies/${btn.dataset.fp}`, { method: 'DELETE' });
            toast('Cookie removed');
            await load();
          };
        }
      } catch (e) { /* auth error already toasted */ }
    }

    document.getElementById('refreshBtn').onclick = async () => {
      document.getElementById('refreshBtn').textContent = 'Refreshing...';
      try {
        const r = await apiFetch('/api/cookies/refresh', { method: 'POST' }).then(r => r.json());
        toast(`Refreshed ${r.refreshed}/${r.total} cookies`);
      } catch(e) {}
      document.getElementById('refreshBtn').textContent = 'Refresh All';
      await load();
    };

    document.getElementById('addBtn').onclick = async () => {
      const input = document.getElementById('cookieInput');
      const cookie = input.value.trim();
      if (!cookie) return;
      const r = await apiFetch('/api/cookies', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cookie })
      }).then(r => r.json());
      input.value = '';
      toast(r.ok ? 'Cookie added' : 'Cookie already exists or invalid');
      await load();
    };

    document.getElementById('saveApiKeyBtn').onclick = async () => {
      const input = document.getElementById('apiKeyInput');
      const key = input.value.trim();
      if (key) {
        localStorage.setItem('proxy_api_key', key);
        toast('API key saved');
      } else {
        localStorage.removeItem('proxy_api_key');
        toast('API key cleared');
      }
      await load();
    };

    document.getElementById('apiKeyInput').value = localStorage.getItem('proxy_api_key') || '';
    load();
    setInterval(load, 3000);
  </script>
</body>
</html>"#,
    )
}

async fn auth_login_page_handler() -> Html<&'static str> {
    Html(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Login — Onyx Proxy</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg: #0f1117; --surface: #1a1d27; --surface2: #242736;
      --border: #2e3348; --text: #e1e4ed; --muted: #8b8fa3;
      --primary: #6c72cb; --primary-hover: #7b82d8;
      --green: #34d399; --red: #f87171;
      --radius: 10px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Inter', system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; align-items: center; justify-content: center; }
    .login-card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 40px; width: 100%; max-width: 420px; }
    .login-card h2 { font-size: 22px; font-weight: 700; margin-bottom: 8px; text-align: center; }
    .login-card .sub { font-size: 13px; color: var(--muted); text-align: center; margin-bottom: 28px; }
    .field { margin-bottom: 16px; }
    .field label { display: block; font-size: 12px; font-weight: 500; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 6px; }
    .field input { width: 100%; background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; color: var(--text); font-size: 14px; outline: none; transition: border .2s; }
    .field input:focus { border-color: var(--primary); }
    .btn { width: 100%; background: var(--primary); color: #fff; border: none; border-radius: 8px; padding: 12px; font-size: 14px; font-weight: 600; cursor: pointer; transition: background .2s, transform .1s; margin-top: 8px; }
    .btn:hover { background: var(--primary-hover); }
    .btn:active { transform: scale(.98); }
    .btn:disabled { opacity: .6; cursor: not-allowed; }
    .divider { display: flex; align-items: center; gap: 12px; margin: 20px 0; font-size: 12px; color: var(--muted); }
    .divider::before, .divider::after { content: ''; flex: 1; height: 1px; background: var(--border); }
    .oauth-btn { width: 100%; display: flex; align-items: center; justify-content: center; gap: 10px; background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-size: 14px; font-weight: 500; cursor: pointer; color: var(--text); text-decoration: none; transition: background .2s; }
    .oauth-btn:hover { background: var(--border); }
    .oauth-btn svg { width: 18px; height: 18px; }
    .msg { margin-top: 14px; padding: 10px 14px; border-radius: 8px; font-size: 13px; display: none; }
    .msg.error { display: block; background: rgba(248,113,113,.12); color: var(--red); border: 1px solid rgba(248,113,113,.2); }
    .msg.success { display: block; background: rgba(52,211,153,.12); color: var(--green); border: 1px solid rgba(52,211,153,.2); }
    .back { display: block; text-align: center; margin-top: 18px; color: var(--muted); font-size: 13px; text-decoration: none; }
    .back:hover { color: var(--text); }
  </style>
</head>
<body>
  <div class="login-card">
    <h2>🔑 Get Cookie</h2>
    <div class="sub">Login to Onyx Cloud to automatically capture your auth cookie.</div>

    <form id="loginForm">
      <div class="field">
        <label>Email</label>
        <input type="email" id="email" placeholder="your@email.com" required autocomplete="email" />
      </div>
      <div class="field">
        <label>Password</label>
        <input type="password" id="password" placeholder="••••••••" required autocomplete="current-password" />
      </div>
      <button type="submit" class="btn" id="loginBtn">Login & Capture Cookie</button>
    </form>

    <div id="msg" class="msg"></div>

    <div class="divider">or</div>

    <a class="oauth-btn" href="https://cloud.onyx.app/auth/login" target="_blank" rel="noopener">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
      Login via Google on Onyx Cloud
    </a>
    <div style="text-align:center;margin-top:8px;font-size:11px;color:var(--muted);">
      After Google login, copy the <code>fastapiusersauth</code> cookie from DevTools and paste it in the dashboard.
    </div>

    <a class="back" href="/ui">← Back to Dashboard</a>
  </div>

  <script>
    const form = document.getElementById('loginForm');
    const msgEl = document.getElementById('msg');
    const btn = document.getElementById('loginBtn');

    function authHeaders(extra = {}) {
      const key = localStorage.getItem('proxy_api_key') || '';
      if (!key) return extra;
      return { ...extra, Authorization: `Bearer ${key}` };
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      msgEl.className = 'msg';
      btn.disabled = true;
      btn.textContent = 'Logging in...';

      try {
        const resp = await fetch('/auth/login', {
          method: 'POST',
          headers: authHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({
            email: document.getElementById('email').value,
            password: document.getElementById('password').value,
          }),
        });
        const data = await resp.json();

        if (resp.ok && data.ok) {
          msgEl.className = 'msg success';
          msgEl.textContent = '✅ Cookie captured! Fingerprint: ' + (data.fingerprint || 'unknown') + '. Redirecting...';
          setTimeout(() => { window.location.href = '/ui'; }, 1500);
        } else {
          msgEl.className = 'msg error';
          msgEl.textContent = data.error || 'Login failed. Check your credentials.';
        }
      } catch (err) {
        msgEl.className = 'msg error';
        msgEl.textContent = 'Network error: ' + err.message;
      } finally {
        btn.disabled = false;
        btn.textContent = 'Login & Capture Cookie';
      }
    });
  </script>
</body>
</html>"#,
    )
}

async fn auth_login_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(payload): Json<LoginRequest>,
) -> impl IntoResponse {
    if let Err(status) = ensure_authorized(&headers, &state) {
        return (status, Json(json!({"error":"unauthorized"}))).into_response();
    }

    // Proxy the login to Onyx Cloud's basic auth endpoint
    let login_result = state
        .http_client
        .post(format!("{}/api/auth/login", state.settings.onyx_base_url))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(format!(
            "username={}&password={}",
            urlencoding::encode(&payload.email),
            urlencoding::encode(&payload.password),
        ))
        .send()
        .await;

    let resp = match login_result {
        Ok(r) => r,
        Err(err) => {
            let reason = format_std_error_chain(&err);
            error!(endpoint = "/auth/login", error = %reason, "onyx login request failed");
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": format!("failed to reach onyx: {reason}")})),
            )
                .into_response();
        }
    };

    let status_code = resp.status();

    // Extract the fastapiusersauth cookie from response
    let cookie_value = extract_cookie_from_headers(resp.headers());

    // Read body for error details
    let body: serde_json::Value = resp.json().await.unwrap_or(json!({}));

    if !status_code.is_success() || cookie_value.is_none() {
        let detail = body
            .get("detail")
            .and_then(|v| v.as_str())
            .unwrap_or("login failed");
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"ok": false, "error": detail})),
        )
            .into_response();
    }

    // Add the captured cookie to the cookie manager
    let cookie_val = cookie_value.unwrap();
    let fingerprint = {
        let mut cm = state.cookie_manager.write().await;
        cm.add_cookie(&cookie_val);
        cm.save();
        crate::cookie_manager::fingerprint(&cookie_val)
    };

    (
        StatusCode::OK,
        Json(json!({
            "ok": true,
            "fingerprint": fingerprint,
            "message": "Cookie captured and added to pool"
        })),
    )
        .into_response()
}

async fn status_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> Result<Json<StatusResponse>, StatusCode> {
    ensure_authorized(&headers, &state)?;
    let cm = state.cookie_manager.read().await;
    let stats = cm.stats();
    Ok(Json(StatusResponse {
        status: "healthy",
        uptime_secs: now_ts().saturating_sub(state.started_at_ts),
        onyx_base_url: state.settings.onyx_base_url.clone(),
        cookie_total: stats.total,
        cookie_active: stats.active,
        cookie_exhausted: stats.exhausted,
    }))
}

async fn v1_models_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> Result<Json<ModelsListResponse>, StatusCode> {
    ensure_authorized(&headers, &state)?;

    let data = supported_models()
        .into_iter()
        .map(|id| ModelItem {
            id: id.to_string(),
            object: "model",
            created: 1_700_000_000,
            owned_by: "onyx",
        })
        .collect::<Vec<_>>();

    Ok(Json(ModelsListResponse {
        object: "list",
        data,
    }))
}

async fn v1_chat_completions_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    raw_body: Bytes,
) -> impl IntoResponse {
    if let Err(status) = ensure_authorized(&headers, &state) {
        return (status, Json(json!({"error":"unauthorized"}))).into_response();
    }

    let raw_request_body = String::from_utf8_lossy(&raw_body).to_string();
    let req: ChatCompletionRequest = match serde_json::from_slice(&raw_body) {
        Ok(req) => req,
        Err(err) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"error":format!("invalid request body: {err}")})),
            )
                .into_response();
        }
    };

    if req.messages.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":"messages must not be empty"})),
        )
            .into_response();
    }

    let request_log =
        serde_json::to_value(&req).unwrap_or_else(|_| json!({"error":"request_serialize_failed"}));
    let chat_messages = pseudo_tools::normalize_openai_messages(&req.messages);
    let tool_context = pseudo_tools::context_from_openai_request(&req);
    let uses_pseudo_tool_protocol =
        pseudo_tools::should_enable_openai_protocol(&req, &tool_context);

    if uses_pseudo_tool_protocol {
        let model = req
            .model
            .clone()
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());
        let include_reasoning = req.include_reasoning.unwrap_or(true);

        if req.stream.unwrap_or(false) {
            return openai_pseudo_tool_stream_response(
                headers,
                state,
                raw_request_body,
                request_log,
                chat_messages,
                tool_context,
                model,
                include_reasoning,
            )
            .await;
        }

        return openai_pseudo_tool_non_stream_response(
            headers,
            state,
            raw_request_body,
            request_log,
            chat_messages,
            tool_context,
            model,
            include_reasoning,
        )
        .await;
    }

    if req.stream.unwrap_or(false) {
        let model = req.model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
        let include_reasoning = req.include_reasoning.unwrap_or(true);

        let cookie_values = active_cookie_values_round_robin(&state).await;

        if cookie_values.is_empty() {
            let response_json = json!({"error":"no cookies configured"});
            append_proxy_audit_record(
                &state.settings,
                "/v1/chat/completions",
                &request_log,
                Some(&raw_request_body),
                StatusCode::BAD_GATEWAY,
                &response_json,
            )
            .await;
            return (StatusCode::BAD_GATEWAY, Json(response_json)).into_response();
        }

        let chat_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = now_ts();
        let model_for_stream = model.clone();

        // Try each cookie until one works
        let mut last_err = String::from("unknown upstream error");
        let mut rx_opt = None;
        let mut _success_cookie: Option<String> = None;

        for cookie in &cookie_values {
            match onyx_client::streaming_chat(
                &state.http_client,
                &state.settings,
                cookie,
                &req.messages,
                &model,
            )
            .await
            {
                Ok(rx) => {
                    let mut cm = state.cookie_manager.write().await;
                    cm.mark_call_success(cookie);
                    cm.save();
                    rx_opt = Some(rx);
                    _success_cookie = Some(cookie.clone());
                    break;
                }
                Err(err) => {
                    let err_msg = onyx_client::format_error_chain(&err);
                    let cookie_fp = crate::cookie_manager::fingerprint(cookie);
                    error!(
                        endpoint = "/v1/chat/completions",
                        mode = "stream",
                        cookie = %cookie_fp,
                        error = %err_msg,
                        "upstream streaming_chat failed"
                    );
                    let mut cm = state.cookie_manager.write().await;
                    mark_cookie_failure(&mut cm, cookie, err_msg.clone());
                    cm.save();
                    last_err = err_msg;
                }
            }
        }

        let Some(rx) = rx_opt else {
            let response_json = json!({"error": format!("upstream failure: {last_err}")});
            append_proxy_audit_record(
                &state.settings,
                "/v1/chat/completions",
                &request_log,
                Some(&raw_request_body),
                StatusCode::BAD_GATEWAY,
                &response_json,
            )
            .await;
            return (StatusCode::BAD_GATEWAY, Json(response_json)).into_response();
        };

        append_proxy_audit_record(
            &state.settings,
            "/v1/chat/completions",
            &request_log,
            Some(&raw_request_body),
            StatusCode::OK,
            &json!({"stream":true,"status":"started"}),
        )
        .await;

        // Build SSE stream from the mpsc receiver
        let sse_stream = stream::unfold(
            (
                rx,
                chat_id,
                model_for_stream,
                created,
                include_reasoning,
                false,
            ),
            |(mut rx, chat_id, model, created, include_reasoning, done)| async move {
                if done {
                    return None;
                }

                match rx.recv().await {
                    Some(StreamEvent::Role) => {
                        let chunk = ChatCompletionChunk {
                            id: chat_id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: Some("assistant"),
                                    content: None,
                                    reasoning_content: None,
                                    tool_calls: None,
                                },
                                finish_reason: None,
                            }],
                        };
                        let data = serde_json::to_string(&chunk).unwrap_or_default();
                        let event = Event::default().data(data);
                        Some((
                            Ok::<_, Infallible>(event),
                            (rx, chat_id, model, created, include_reasoning, false),
                        ))
                    }
                    Some(StreamEvent::Reasoning(text)) => {
                        if !include_reasoning {
                            // Skip reasoning chunks if not requested
                            return Some((
                                Ok::<_, Infallible>(Event::default().data("")),
                                (rx, chat_id, model, created, include_reasoning, false),
                            ));
                        }
                        let chunk = ChatCompletionChunk {
                            id: chat_id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: None,
                                    reasoning_content: Some(text),
                                    tool_calls: None,
                                },
                                finish_reason: None,
                            }],
                        };
                        let data = serde_json::to_string(&chunk).unwrap_or_default();
                        let event = Event::default().data(data);
                        Some((
                            Ok::<_, Infallible>(event),
                            (rx, chat_id, model, created, include_reasoning, false),
                        ))
                    }
                    Some(StreamEvent::Content(text)) => {
                        let chunk = ChatCompletionChunk {
                            id: chat_id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: Some(text),
                                    reasoning_content: None,
                                    tool_calls: None,
                                },
                                finish_reason: None,
                            }],
                        };
                        let data = serde_json::to_string(&chunk).unwrap_or_default();
                        let event = Event::default().data(data);
                        Some((
                            Ok::<_, Infallible>(event),
                            (rx, chat_id, model, created, include_reasoning, false),
                        ))
                    }
                    Some(StreamEvent::Done) | None => {
                        let chunk = ChatCompletionChunk {
                            id: chat_id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: None,
                                    reasoning_content: None,
                                    tool_calls: None,
                                },
                                finish_reason: Some("stop"),
                            }],
                        };
                        let data = serde_json::to_string(&chunk).unwrap_or_default();
                        let stop_event = Event::default().data(data);
                        Some((
                            Ok::<_, Infallible>(stop_event),
                            (rx, chat_id, model, created, include_reasoning, true),
                        ))
                    }
                }
            },
        );

        return Sse::new(sse_stream).into_response();
    }

    let model = req.model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let include_reasoning = req.include_reasoning.unwrap_or(true);

    let cookie_values = active_cookie_values_round_robin(&state).await;

    if cookie_values.is_empty() {
        return (
            StatusCode::BAD_GATEWAY,
            Json(json!({"error":"no cookies configured"})),
        )
            .into_response();
    }

    let mut last_err = String::from("unknown upstream error");
    for cookie in cookie_values {
        match onyx_client::full_chat(
            &state.http_client,
            &state.settings,
            &cookie,
            &req.messages,
            &model,
        )
        .await
        {
            Ok((content, thinking)) => {
                let content = pseudo_tools::strip_noop_trailer(&content).unwrap_or(content);
                let mut cm = state.cookie_manager.write().await;
                cm.mark_call_success(&cookie);
                cm.save();

                let response = ChatCompletionResponse {
                    id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                    object: "chat.completion",
                    created: now_ts(),
                    model,
                    choices: vec![Choice {
                        index: 0,
                        message: AssistantMessage {
                            role: "assistant",
                            content: Some(content),
                            reasoning_content: if include_reasoning && !thinking.is_empty() {
                                Some(thinking)
                            } else {
                                None
                            },
                            tool_calls: None,
                        },
                        finish_reason: "stop",
                    }],
                    usage: Usage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    },
                };
                let response_json = json!(response);
                append_proxy_audit_record(
                    &state.settings,
                    "/v1/chat/completions",
                    &request_log,
                    Some(&raw_request_body),
                    StatusCode::OK,
                    &response_json,
                )
                .await;

                return (StatusCode::OK, Json(response_json)).into_response();
            }
            Err(err) => {
                let err_msg = onyx_client::format_error_chain(&err);
                let cookie_fp = crate::cookie_manager::fingerprint(&cookie);
                error!(
                    endpoint = "/v1/chat/completions",
                    mode = "non_stream",
                    cookie = %cookie_fp,
                    error = %err_msg,
                    "upstream full_chat failed"
                );
                let mut cm = state.cookie_manager.write().await;
                mark_cookie_failure(&mut cm, &cookie, err_msg.clone());
                cm.save();
                last_err = err_msg;
            }
        }
    }

    let response_json = json!({"error": format!("upstream failure: {last_err}")});
    append_proxy_audit_record(
        &state.settings,
        "/v1/chat/completions",
        &request_log,
        Some(&raw_request_body),
        StatusCode::BAD_GATEWAY,
        &response_json,
    )
    .await;
    (StatusCode::BAD_GATEWAY, Json(response_json)).into_response()
}

async fn cookies_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    ensure_authorized(&headers, &state)?;
    let cm = state.cookie_manager.read().await;
    Ok(Json(json!(cm.views())))
}

async fn add_cookie_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(payload): Json<AddCookieRequest>,
) -> Result<Json<MutationResponse>, StatusCode> {
    ensure_authorized(&headers, &state)?;
    let mut cm = state.cookie_manager.write().await;
    let ok = cm.add_cookie(&payload.cookie);
    Ok(Json(MutationResponse { ok }))
}

async fn delete_cookie_handler(
    headers: HeaderMap,
    Path(fingerprint): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<MutationResponse>, StatusCode> {
    ensure_authorized(&headers, &state)?;
    let mut cm = state.cookie_manager.write().await;
    let ok = cm.remove_by_fingerprint(&fingerprint);
    Ok(Json(MutationResponse { ok }))
}

async fn refresh_cookies_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> Result<Json<RefreshResponse>, StatusCode> {
    ensure_authorized(&headers, &state)?;
    Ok(Json(refresh_all_cookies(&state).await))
}

async fn refresh_all_cookies(state: &AppState) -> RefreshResponse {
    let mut refreshed = 0usize;
    let mut failed = 0usize;

    let mut cm = state.cookie_manager.write().await;
    let total = cm.entries_mut().len();
    for entry in cm.entries_mut() {
        let res = state
            .http_client
            .post(format!("{}/api/auth/refresh", state.settings.onyx_base_url))
            .header("Cookie", format!("fastapiusersauth={}", entry.value))
            .send()
            .await;

        match res {
            Ok(resp) if resp.status().is_success() => {
                if let Some(updated) = extract_cookie_from_headers(resp.headers()) {
                    entry.value = updated;
                }
                entry.exhausted = false;
                entry.temporary_failures = 0;
                entry.cooldown_until_ts = None;
                entry.last_error = None;
                entry.last_refresh_ts = Some(now_ts());
                refreshed += 1;
            }
            Ok(resp) => {
                let status = resp.status().as_u16();
                let body = resp.text().await.unwrap_or_default();
                let reason = format!(
                    "refresh failed: http {status}, body: {}",
                    truncate_for_error(&body, 240)
                );
                error!(endpoint = "/api/cookies/refresh", error = %reason, "cookie refresh http failure");
                apply_cookie_refresh_failure(entry, &reason);
                failed += 1;
            }
            Err(err) => {
                let reason = format_std_error_chain(&err);
                error!(endpoint = "/api/cookies/refresh", error = %reason, "cookie refresh transport failure");
                apply_cookie_refresh_failure(entry, &reason);
                failed += 1;
            }
        }
    }

    cm.save();

    RefreshResponse {
        total,
        refreshed,
        failed,
    }
}

fn apply_cookie_refresh_failure(entry: &mut crate::cookie_manager::CookieEntry, reason: &str) {
    let kind = classify_cookie_failure(reason);
    if matches!(kind, CookieFailureKind::Permanent) {
        entry.exhausted = true;
        entry.temporary_failures = 0;
        entry.cooldown_until_ts = None;
    } else {
        entry.temporary_failures = entry.temporary_failures.saturating_add(1);
        if entry.temporary_failures >= 3 {
            entry.cooldown_until_ts = Some(now_ts().saturating_add(120));
            entry.temporary_failures = 0;
        }
    }
    entry.last_error = Some(reason.to_string());
}

/// Anthropic Claude-compatible Messages endpoint.
/// Accepts requests in Claude format and converts to/from Onyx backend.
async fn v1_messages_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    raw_body: Bytes,
) -> impl IntoResponse {
    // Claude uses x-api-key header instead of Bearer token
    if let Some(expected) = &state.settings.api_key {
        let api_key = headers
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .or_else(|| extract_bearer_token(&headers));
        let valid = api_key.map(|k| k == expected.as_str()).unwrap_or(false);
        if !valid {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}})),
            )
                .into_response();
        }
    }

    let raw_request_body = String::from_utf8_lossy(&raw_body).to_string();

    let req = match serde_json::from_slice::<ClaudeMessagesRequest>(&raw_body) {
        Ok(req) => req,
        Err(err) => {
            let reason = err.to_string();
            error!(
                endpoint = "/v1/messages",
                error = %reason,
                "claude request deserialization failed"
            );
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"type":"error","error":{"type":"invalid_request_error","message":format!("invalid request body: {reason}")}})),
            )
                .into_response();
        }
    };

    if req.messages.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"type":"error","error":{"type":"invalid_request_error","message":"messages must not be empty"}})),
        )
            .into_response();
    }

    let request_log =
        serde_json::to_value(&req).unwrap_or_else(|_| json!({"error":"request_serialize_failed"}));

    // Convert Claude messages to our internal ChatMessage format
    let mut chat_messages: Vec<ChatMessage> = Vec::new();
    if let Some(sys) = &req.system {
        chat_messages.push(ChatMessage {
            role: "system".to_string(),
            content: sys.clone(),
            tool_call_id: None,
            name: None,
            tool_calls: None,
        });
    }

    for m in &req.messages {
        chat_messages.push(ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
            tool_call_id: None,
            name: None,
            tool_calls: None,
        });
    }

    let model = req.model.clone();

    let tool_context = pseudo_tools::context_from_claude_request(&req);
    let uses_pseudo_tool_protocol =
        pseudo_tools::should_enable_claude_protocol(&req, &tool_context);

    // Estimate input tokens (rough estimation: 1 token ~= 4 chars)
    let input_tokens = {
        let mut chars = req.system.as_ref().map(|s| s.len()).unwrap_or(0);
        for m in &req.messages {
            chars += m.role.len() + m.content.len() + 4;
        }
        (chars / 4) as u32 + 10 // +10 for structural overhead
    };

    if uses_pseudo_tool_protocol {
        if req.stream.unwrap_or(false) {
            return claude_pseudo_tool_stream_response(
                headers,
                state,
                raw_request_body,
                request_log,
                chat_messages,
                tool_context,
                model,
                input_tokens,
            )
            .await;
        }

        return claude_pseudo_tool_non_stream_response(
            headers,
            state,
            raw_request_body,
            request_log,
            chat_messages,
            tool_context,
            model,
            input_tokens,
        )
        .await;
    }

    let cookie_values = active_cookie_values_round_robin(&state).await;

    if cookie_values.is_empty() {
        let response_json =
            json!({"type":"error","error":{"type":"api_error","message":"no cookies configured"}});
        append_proxy_audit_record(
            &state.settings,
            "/v1/messages",
            &request_log,
            Some(&raw_request_body),
            StatusCode::BAD_GATEWAY,
            &response_json,
        )
        .await;
        return (StatusCode::BAD_GATEWAY, Json(response_json)).into_response();
    }

    // ----- Streaming mode -----
    if req.stream.unwrap_or(false) {
        let msg_id = format!("msg_{}", uuid::Uuid::new_v4().as_simple());
        let model_for_stream = model.clone();

        let mut last_err = String::from("unknown upstream error");
        let mut rx_opt: Option<tokio::sync::mpsc::Receiver<StreamEvent>> = None;

        for cookie in &cookie_values {
            match onyx_client::streaming_chat(
                &state.http_client,
                &state.settings,
                cookie,
                &chat_messages,
                &model,
            )
            .await
            {
                Ok(rx) => {
                    let mut cm = state.cookie_manager.write().await;
                    cm.mark_call_success(cookie);
                    cm.save();
                    rx_opt = Some(rx);
                    break;
                }
                Err(err) => {
                    let err_msg = onyx_client::format_error_chain(&err);
                    let cookie_fp = crate::cookie_manager::fingerprint(cookie);
                    error!(
                        endpoint = "/v1/messages",
                        mode = "stream",
                        cookie = %cookie_fp,
                        error = %err_msg,
                        "claude streaming upstream failed"
                    );
                    let mut cm = state.cookie_manager.write().await;
                    mark_cookie_failure(&mut cm, cookie, err_msg.clone());
                    cm.save();
                    last_err = err_msg;
                }
            }
        }

        let Some(rx) = rx_opt else {
            let response_json = json!({"type":"error","error":{"type":"api_error","message":format!("upstream failure: {last_err}")}});
            append_proxy_audit_record(
                &state.settings,
                "/v1/messages",
                &request_log,
                Some(&raw_request_body),
                StatusCode::BAD_GATEWAY,
                &response_json,
            )
            .await;
            return (StatusCode::BAD_GATEWAY, Json(response_json)).into_response();
        };

        append_proxy_audit_record(
            &state.settings,
            "/v1/messages",
            &request_log,
            Some(&raw_request_body),
            StatusCode::OK,
            &json!({"stream":true,"status":"started"}),
        )
        .await;

        // Build SSE stream for Claude format
        // State: (rx, msg_id, model, phase, output_tokens, input_tokens, buffer)
        // phase: 0=message_start, 1=content_block_start, 2=streaming, 3=content_block_stop, 4=message_delta, 5=message_stop
        let sse_stream = stream::unfold(
            (
                rx,
                msg_id,
                model_for_stream,
                0u8,
                0u32,
                input_tokens,
                None::<StreamEvent>,
            ),
            |(mut rx, msg_id, model, phase, output_tokens, input_tokens, mut event_buffer)| async move {
                match phase {
                    0 => {
                        // Emit message_start
                        let start = ClaudeStreamMessageStart {
                            type_field: "message_start",
                            message: ClaudeStreamMessageMeta {
                                id: msg_id.clone(),
                                type_field: "message",
                                role: "assistant",
                                content: vec![],
                                model: model.clone(),
                                stop_reason: None,
                                usage: ClaudeUsage {
                                    input_tokens,
                                    output_tokens: 0,
                                },
                            },
                        };
                        let data = serde_json::to_string(&start).unwrap_or_default();
                        let event = Event::default().event("message_start").data(data);
                        Some((
                            Ok::<_, Infallible>(event),
                            (rx, msg_id, model, 1, output_tokens, input_tokens, None),
                        ))
                    }
                    1 => {
                        // Emit content_block_start
                        let block_start = ClaudeStreamContentBlockStart {
                            type_field: "content_block_start",
                            index: 0,
                            content_block: ClaudeContentBlock {
                                type_field: "text",
                                text: Some(String::new()),
                                id: None,
                                name: None,
                                input: None,
                            },
                        };
                        let data = serde_json::to_string(&block_start).unwrap_or_default();
                        let event = Event::default().event("content_block_start").data(data);
                        Some((
                            Ok::<_, Infallible>(event),
                            (rx, msg_id, model, 2, output_tokens, input_tokens, None),
                        ))
                    }
                    2 => {
                        // Streaming content_block_delta
                        let event = if let Some(e) = event_buffer.take() {
                            Some(e)
                        } else {
                            rx.recv().await
                        };

                        match event {
                            Some(StreamEvent::Content(text))
                            | Some(StreamEvent::Reasoning(text)) => {
                                let new_tokens = output_tokens + 1;
                                let delta = ClaudeStreamContentBlockDelta {
                                    type_field: "content_block_delta",
                                    index: 0,
                                    delta: serde_json::to_value(ClaudeTextDelta {
                                        type_field: "text_delta",
                                        text,
                                    })
                                    .unwrap_or_else(|_| json!({})),
                                };
                                let data = serde_json::to_string(&delta).unwrap_or_default();
                                let event =
                                    Event::default().event("content_block_delta").data(data);
                                Some((
                                    Ok::<_, Infallible>(event),
                                    (rx, msg_id, model, 2, new_tokens, input_tokens, None),
                                ))
                            }
                            Some(StreamEvent::Role) => {
                                // Skip role event, stay in phase 2 and recv again
                                let event =
                                    Event::default().event("ping").data("{\"type\":\"ping\"}");
                                Some((
                                    Ok::<_, Infallible>(event),
                                    (rx, msg_id, model, 2, output_tokens, input_tokens, None),
                                ))
                            }
                            Some(StreamEvent::Done) | None => {
                                // Transition to content_block_stop
                                let block_stop = ClaudeStreamContentBlockStop {
                                    type_field: "content_block_stop",
                                    index: 0,
                                };
                                let data = serde_json::to_string(&block_stop).unwrap_or_default();
                                let event = Event::default().event("content_block_stop").data(data);
                                Some((
                                    Ok::<_, Infallible>(event),
                                    (rx, msg_id, model, 4, output_tokens, input_tokens, None),
                                ))
                            }
                        }
                    }
                    4 => {
                        // Emit message_delta
                        let msg_delta = ClaudeStreamMessageDelta {
                            type_field: "message_delta",
                            delta: ClaudeStopDelta {
                                stop_reason: "end_turn",
                            },
                            usage: ClaudeUsage {
                                input_tokens,
                                output_tokens,
                            },
                        };
                        let data = serde_json::to_string(&msg_delta).unwrap_or_default();
                        let event = Event::default().event("message_delta").data(data);
                        Some((
                            Ok::<_, Infallible>(event),
                            (rx, msg_id, model, 5, output_tokens, input_tokens, None),
                        ))
                    }
                    5 => {
                        // Emit message_stop
                        let stop = ClaudeStreamMessageStop {
                            type_field: "message_stop",
                        };
                        let data = serde_json::to_string(&stop).unwrap_or_default();
                        let event = Event::default().event("message_stop").data(data);
                        Some((
                            Ok::<_, Infallible>(event),
                            (rx, msg_id, model, 99, output_tokens, input_tokens, None),
                        ))
                    }
                    _ => None, // terminal
                }
            },
        );

        return Sse::new(sse_stream).into_response();
    }

    // ----- Non-streaming mode -----
    let mut last_err = String::from("unknown upstream error");

    for cookie in &cookie_values {
        match onyx_client::full_chat(
            &state.http_client,
            &state.settings,
            cookie,
            &chat_messages,
            &model,
        )
        .await
        {
            Ok((content, _thinking)) => {
                let content = pseudo_tools::strip_noop_trailer(&content).unwrap_or(content);
                let mut cm = state.cookie_manager.write().await;
                cm.mark_call_success(cookie);
                cm.save();

                let content_blocks = vec![ClaudeContentBlock {
                    type_field: "text",
                    text: Some(content),
                    id: None,
                    name: None,
                    input: None,
                }];
                let output_tokens = estimate_claude_output_tokens(&content_blocks);

                let response = ClaudeMessagesResponse {
                    id: format!("msg_{}", uuid::Uuid::new_v4().as_simple()),
                    type_field: "message",
                    role: "assistant",
                    content: content_blocks,
                    model: model.clone(),
                    stop_reason: "end_turn",
                    usage: ClaudeUsage {
                        input_tokens,
                        output_tokens,
                    },
                };
                let response_json = json!(response);
                append_proxy_audit_record(
                    &state.settings,
                    "/v1/messages",
                    &request_log,
                    Some(&raw_request_body),
                    StatusCode::OK,
                    &response_json,
                )
                .await;

                return (StatusCode::OK, Json(response_json)).into_response();
            }
            Err(err) => {
                let err_msg = onyx_client::format_error_chain(&err);
                let cookie_fp = crate::cookie_manager::fingerprint(cookie);
                error!(
                    endpoint = "/v1/messages",
                    mode = "non_stream",
                    cookie = %cookie_fp,
                    error = %err_msg,
                    "claude non-stream upstream failed"
                );
                let mut cm = state.cookie_manager.write().await;
                mark_cookie_failure(&mut cm, cookie, err_msg.clone());
                cm.save();
                last_err = err_msg;
            }
        }
    }

    let response_json = json!({"type":"error","error":{"type":"api_error","message":format!("upstream failure: {last_err}")}});
    append_proxy_audit_record(
        &state.settings,
        "/v1/messages",
        &request_log,
        Some(&raw_request_body),
        StatusCode::BAD_GATEWAY,
        &response_json,
    )
    .await;
    (StatusCode::BAD_GATEWAY, Json(response_json)).into_response()
}

#[derive(Debug, Clone)]
struct PseudoToolRunOutcome {
    response: ParsedPseudoToolResponse,
    thinking: String,
}

async fn openai_pseudo_tool_non_stream_response(
    _headers: HeaderMap,
    state: AppState,
    raw_request_body: String,
    request_log: serde_json::Value,
    chat_messages: Vec<ChatMessage>,
    tool_context: ToolPromptContext,
    model: String,
    include_reasoning: bool,
) -> axum::response::Response {
    let cookie_values = active_cookie_values_round_robin(&state).await;
    if cookie_values.is_empty() {
        let response_json = json!({"error":"no cookies configured"});
        append_proxy_audit_record(
            &state.settings,
            "/v1/chat/completions",
            &request_log,
            Some(&raw_request_body),
            StatusCode::BAD_GATEWAY,
            &response_json,
        )
        .await;
        return (StatusCode::BAD_GATEWAY, Json(response_json)).into_response();
    }

    match run_pseudo_tool_protocol(
        &state,
        &cookie_values,
        &chat_messages,
        &tool_context,
        &model,
    )
    .await
    {
        Ok(outcome) => {
            let finish_reason = match outcome.response {
                ParsedPseudoToolResponse::Final { .. } => "stop",
                ParsedPseudoToolResponse::Action { .. } => "tool_calls",
            };

            let message = build_openai_assistant_message(&outcome, include_reasoning);
            let response = ChatCompletionResponse {
                id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                object: "chat.completion",
                created: now_ts(),
                model,
                choices: vec![Choice {
                    index: 0,
                    message,
                    finish_reason,
                }],
                usage: Usage {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0,
                },
            };
            let response_json = json!(response);
            append_proxy_audit_record(
                &state.settings,
                "/v1/chat/completions",
                &request_log,
                Some(&raw_request_body),
                StatusCode::OK,
                &response_json,
            )
            .await;
            (StatusCode::OK, Json(response_json)).into_response()
        }
        Err(last_err) => {
            let response_json = json!({"error": format!("upstream failure: {last_err}")});
            append_proxy_audit_record(
                &state.settings,
                "/v1/chat/completions",
                &request_log,
                Some(&raw_request_body),
                StatusCode::BAD_GATEWAY,
                &response_json,
            )
            .await;
            (StatusCode::BAD_GATEWAY, Json(response_json)).into_response()
        }
    }
}

async fn openai_pseudo_tool_stream_response(
    _headers: HeaderMap,
    state: AppState,
    raw_request_body: String,
    request_log: serde_json::Value,
    chat_messages: Vec<ChatMessage>,
    tool_context: ToolPromptContext,
    model: String,
    include_reasoning: bool,
) -> axum::response::Response {
    let cookie_values = active_cookie_values_round_robin(&state).await;
    if cookie_values.is_empty() {
        let response_json = json!({"error":"no cookies configured"});
        append_proxy_audit_record(
            &state.settings,
            "/v1/chat/completions",
            &request_log,
            Some(&raw_request_body),
            StatusCode::BAD_GATEWAY,
            &response_json,
        )
        .await;
        return (StatusCode::BAD_GATEWAY, Json(response_json)).into_response();
    }

    match run_pseudo_tool_protocol(
        &state,
        &cookie_values,
        &chat_messages,
        &tool_context,
        &model,
    )
    .await
    {
        Ok(outcome) => {
            append_proxy_audit_record(
                &state.settings,
                "/v1/chat/completions",
                &request_log,
                Some(&raw_request_body),
                StatusCode::OK,
                &json!({"stream":true,"status":"started","pseudo_tool_protocol":true}),
            )
            .await;

            let chat_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
            let created = now_ts();
            let events = build_openai_buffered_stream_events(
                &chat_id,
                &model,
                created,
                include_reasoning,
                &outcome,
            );
            Sse::new(stream::iter(events)).into_response()
        }
        Err(last_err) => {
            let response_json = json!({"error": format!("upstream failure: {last_err}")});
            append_proxy_audit_record(
                &state.settings,
                "/v1/chat/completions",
                &request_log,
                Some(&raw_request_body),
                StatusCode::BAD_GATEWAY,
                &response_json,
            )
            .await;
            (StatusCode::BAD_GATEWAY, Json(response_json)).into_response()
        }
    }
}

async fn claude_pseudo_tool_non_stream_response(
    _headers: HeaderMap,
    state: AppState,
    raw_request_body: String,
    request_log: serde_json::Value,
    chat_messages: Vec<ChatMessage>,
    tool_context: ToolPromptContext,
    model: String,
    input_tokens: u32,
) -> axum::response::Response {
    let cookie_values = active_cookie_values_round_robin(&state).await;
    if cookie_values.is_empty() {
        let response_json =
            json!({"type":"error","error":{"type":"api_error","message":"no cookies configured"}});
        append_proxy_audit_record(
            &state.settings,
            "/v1/messages",
            &request_log,
            Some(&raw_request_body),
            StatusCode::BAD_GATEWAY,
            &response_json,
        )
        .await;
        return (StatusCode::BAD_GATEWAY, Json(response_json)).into_response();
    }

    match run_pseudo_tool_protocol(
        &state,
        &cookie_values,
        &chat_messages,
        &tool_context,
        &model,
    )
    .await
    {
        Ok(outcome) => {
            let response = build_claude_response(&model, input_tokens, &outcome);
            let response_json = json!(response);
            append_proxy_audit_record(
                &state.settings,
                "/v1/messages",
                &request_log,
                Some(&raw_request_body),
                StatusCode::OK,
                &response_json,
            )
            .await;
            (StatusCode::OK, Json(response_json)).into_response()
        }
        Err(last_err) => {
            let response_json = json!({"type":"error","error":{"type":"api_error","message":format!("upstream failure: {last_err}")}});
            append_proxy_audit_record(
                &state.settings,
                "/v1/messages",
                &request_log,
                Some(&raw_request_body),
                StatusCode::BAD_GATEWAY,
                &response_json,
            )
            .await;
            (StatusCode::BAD_GATEWAY, Json(response_json)).into_response()
        }
    }
}

async fn claude_pseudo_tool_stream_response(
    _headers: HeaderMap,
    state: AppState,
    raw_request_body: String,
    request_log: serde_json::Value,
    chat_messages: Vec<ChatMessage>,
    tool_context: ToolPromptContext,
    model: String,
    input_tokens: u32,
) -> axum::response::Response {
    let cookie_values = active_cookie_values_round_robin(&state).await;
    if cookie_values.is_empty() {
        let response_json =
            json!({"type":"error","error":{"type":"api_error","message":"no cookies configured"}});
        append_proxy_audit_record(
            &state.settings,
            "/v1/messages",
            &request_log,
            Some(&raw_request_body),
            StatusCode::BAD_GATEWAY,
            &response_json,
        )
        .await;
        return (StatusCode::BAD_GATEWAY, Json(response_json)).into_response();
    }

    match run_pseudo_tool_protocol(
        &state,
        &cookie_values,
        &chat_messages,
        &tool_context,
        &model,
    )
    .await
    {
        Ok(outcome) => {
            append_proxy_audit_record(
                &state.settings,
                "/v1/messages",
                &request_log,
                Some(&raw_request_body),
                StatusCode::OK,
                &json!({"stream":true,"status":"started","pseudo_tool_protocol":true}),
            )
            .await;

            let message_id = format!("msg_{}", uuid::Uuid::new_v4().as_simple());
            let events =
                build_claude_buffered_stream_events(&message_id, &model, input_tokens, &outcome);
            Sse::new(stream::iter(events)).into_response()
        }
        Err(last_err) => {
            let response_json = json!({"type":"error","error":{"type":"api_error","message":format!("upstream failure: {last_err}")}});
            append_proxy_audit_record(
                &state.settings,
                "/v1/messages",
                &request_log,
                Some(&raw_request_body),
                StatusCode::BAD_GATEWAY,
                &response_json,
            )
            .await;
            (StatusCode::BAD_GATEWAY, Json(response_json)).into_response()
        }
    }
}

async fn run_pseudo_tool_protocol(
    state: &AppState,
    cookie_values: &[String],
    chat_messages: &[ChatMessage],
    tool_context: &ToolPromptContext,
    model: &str,
) -> Result<PseudoToolRunOutcome, String> {
    let mut last_err = String::from("unknown upstream error");

    for cookie in cookie_values {
        let mut retry_failure: Option<ValidationFailure> = None;

        for attempt in 1..=pseudo_tools::MAX_PROTOCOL_RETRIES {
            let injected_messages = pseudo_tools::prepend_protocol_messages(
                chat_messages,
                tool_context,
                retry_failure.as_ref(),
            );

            match onyx_client::full_chat(
                &state.http_client,
                &state.settings,
                cookie,
                &injected_messages,
                model,
            )
            .await
            {
                Ok((content, thinking)) => {
                    match pseudo_tools::parse_pseudo_tool_response(&content, tool_context) {
                        Ok(response) => {
                            let mut cm = state.cookie_manager.write().await;
                            cm.mark_call_success(cookie);
                            cm.save();
                            return Ok(PseudoToolRunOutcome { response, thinking });
                        }
                        Err(err) if attempt < pseudo_tools::MAX_PROTOCOL_RETRIES => {
                            retry_failure = Some(err);
                        }
                        Err(err) => {
                            last_err = format!(
                                "invalid pseudo tool output after {} attempts: {} ({})",
                                pseudo_tools::MAX_PROTOCOL_RETRIES,
                                err.code,
                                err.message
                            );
                            break;
                        }
                    }
                }
                Err(err) => {
                    let err_msg = onyx_client::format_error_chain(&err);
                    let cookie_fp = crate::cookie_manager::fingerprint(cookie);
                    error!(
                        endpoint = "pseudo_tool_protocol",
                        cookie = %cookie_fp,
                        error = %err_msg,
                        "upstream pseudo tool protocol call failed"
                    );
                    let mut cm = state.cookie_manager.write().await;
                    mark_cookie_failure(&mut cm, cookie, err_msg.clone());
                    cm.save();
                    last_err = err_msg;
                    break;
                }
            }
        }
    }

    Err(last_err)
}

fn build_openai_assistant_message(
    outcome: &PseudoToolRunOutcome,
    include_reasoning: bool,
) -> AssistantMessage {
    let reasoning_content = if include_reasoning && !outcome.thinking.is_empty() {
        Some(outcome.thinking.clone())
    } else {
        None
    };

    match &outcome.response {
        ParsedPseudoToolResponse::Final { content } => AssistantMessage {
            role: "assistant",
            content: Some(content.clone()),
            reasoning_content,
            tool_calls: None,
        },
        ParsedPseudoToolResponse::Action {
            preamble_text,
            tool_name,
            action_input,
        } => AssistantMessage {
            role: "assistant",
            content: preamble_text.clone(),
            reasoning_content,
            tool_calls: Some(vec![build_openai_tool_call(tool_name, action_input, None)]),
        },
    }
}

fn build_openai_tool_call(
    tool_name: &str,
    action_input: &serde_json::Value,
    index: Option<u32>,
) -> crate::models::AssistantToolCall {
    crate::models::AssistantToolCall {
        id: format!("call_{}", uuid::Uuid::new_v4().as_simple()),
        kind: "function",
        function: crate::models::AssistantToolCallFunction {
            name: tool_name.to_string(),
            arguments: serde_json::to_string(action_input).unwrap_or_else(|_| String::from("{}")),
        },
        index,
    }
}

fn build_openai_buffered_stream_events(
    chat_id: &str,
    model: &str,
    created: u64,
    include_reasoning: bool,
    outcome: &PseudoToolRunOutcome,
) -> Vec<Result<Event, Infallible>> {
    let mut events = Vec::new();

    let role_chunk = ChatCompletionChunk {
        id: chat_id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: Some("assistant"),
                content: None,
                reasoning_content: None,
                tool_calls: None,
            },
            finish_reason: None,
        }],
    };
    events.push(Ok(
        Event::default().data(serde_json::to_string(&role_chunk).unwrap_or_default())
    ));

    if include_reasoning && !outcome.thinking.is_empty() {
        let reasoning_chunk = ChatCompletionChunk {
            id: chat_id.to_string(),
            object: "chat.completion.chunk",
            created,
            model: model.to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: None,
                    reasoning_content: Some(outcome.thinking.clone()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
        };
        events.push(Ok(
            Event::default().data(serde_json::to_string(&reasoning_chunk).unwrap_or_default())
        ));
    }

    match &outcome.response {
        ParsedPseudoToolResponse::Final { content } => {
            if !content.is_empty() {
                let content_chunk = ChatCompletionChunk {
                    id: chat_id.to_string(),
                    object: "chat.completion.chunk",
                    created,
                    model: model.to_string(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: None,
                            content: Some(content.clone()),
                            reasoning_content: None,
                            tool_calls: None,
                        },
                        finish_reason: None,
                    }],
                };
                events.push(Ok(Event::default()
                    .data(serde_json::to_string(&content_chunk).unwrap_or_default())));
            }

            let done_chunk = ChatCompletionChunk {
                id: chat_id.to_string(),
                object: "chat.completion.chunk",
                created,
                model: model.to_string(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: None,
                        reasoning_content: None,
                        tool_calls: None,
                    },
                    finish_reason: Some("stop"),
                }],
            };
            events.push(Ok(
                Event::default().data(serde_json::to_string(&done_chunk).unwrap_or_default())
            ));
        }
        ParsedPseudoToolResponse::Action {
            preamble_text,
            tool_name,
            action_input,
        } => {
            if let Some(text) = preamble_text.as_ref().filter(|text| !text.is_empty()) {
                let content_chunk = ChatCompletionChunk {
                    id: chat_id.to_string(),
                    object: "chat.completion.chunk",
                    created,
                    model: model.to_string(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: None,
                            content: Some(text.clone()),
                            reasoning_content: None,
                            tool_calls: None,
                        },
                        finish_reason: None,
                    }],
                };
                events.push(Ok(Event::default()
                    .data(serde_json::to_string(&content_chunk).unwrap_or_default())));
            }

            let tool_chunk = ChatCompletionChunk {
                id: chat_id.to_string(),
                object: "chat.completion.chunk",
                created,
                model: model.to_string(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: None,
                        reasoning_content: None,
                        tool_calls: Some(vec![build_openai_tool_call(
                            tool_name,
                            action_input,
                            Some(0),
                        )]),
                    },
                    finish_reason: None,
                }],
            };
            events.push(Ok(
                Event::default().data(serde_json::to_string(&tool_chunk).unwrap_or_default())
            ));

            let done_chunk = ChatCompletionChunk {
                id: chat_id.to_string(),
                object: "chat.completion.chunk",
                created,
                model: model.to_string(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: None,
                        reasoning_content: None,
                        tool_calls: None,
                    },
                    finish_reason: Some("tool_calls"),
                }],
            };
            events.push(Ok(
                Event::default().data(serde_json::to_string(&done_chunk).unwrap_or_default())
            ));
        }
    }

    events
}

fn build_claude_response(
    model: &str,
    input_tokens: u32,
    outcome: &PseudoToolRunOutcome,
) -> ClaudeMessagesResponse {
    let (content, stop_reason) = match &outcome.response {
        ParsedPseudoToolResponse::Final { content } => (
            vec![ClaudeContentBlock {
                type_field: "text",
                text: Some(content.clone()),
                id: None,
                name: None,
                input: None,
            }],
            "end_turn",
        ),
        ParsedPseudoToolResponse::Action {
            preamble_text,
            tool_name,
            action_input,
        } => {
            let mut blocks = Vec::new();
            if let Some(text) = preamble_text.as_ref().filter(|text| !text.is_empty()) {
                blocks.push(ClaudeContentBlock {
                    type_field: "text",
                    text: Some(text.clone()),
                    id: None,
                    name: None,
                    input: None,
                });
            }
            blocks.push(ClaudeContentBlock {
                type_field: "tool_use",
                text: None,
                id: Some(format!("toolu_{}", uuid::Uuid::new_v4().as_simple())),
                name: Some(tool_name.clone()),
                input: Some(action_input.clone()),
            });
            (blocks, "tool_use")
        }
    };

    let output_tokens = estimate_claude_output_tokens(&content);

    ClaudeMessagesResponse {
        id: format!("msg_{}", uuid::Uuid::new_v4().as_simple()),
        type_field: "message",
        role: "assistant",
        content,
        model: model.to_string(),
        stop_reason,
        usage: ClaudeUsage {
            input_tokens,
            output_tokens,
        },
    }
}

fn estimate_claude_output_tokens(content: &[ClaudeContentBlock]) -> u32 {
    let mut chars = 0usize;

    for block in content {
        chars += match block.type_field {
            "text" => block.text.as_deref().map(str::len).unwrap_or(0),
            "tool_use" => {
                let name_len = block.name.as_deref().map(str::len).unwrap_or(0);
                let input_len = block
                    .input
                    .as_ref()
                    .map(|value| value.to_string().len())
                    .unwrap_or(0);
                name_len + input_len
            }
            _ => 0,
        };
    }

    if chars == 0 {
        0
    } else {
        ((chars / 4) as u32).max(1)
    }
}

fn build_claude_tool_use_stream_payloads(
    index: u8,
    tool_use_id: &str,
    tool_name: &str,
    action_input: &serde_json::Value,
) -> Vec<(&'static str, serde_json::Value)> {
    let block_start = ClaudeStreamContentBlockStart {
        type_field: "content_block_start",
        index,
        content_block: ClaudeContentBlock {
            type_field: "tool_use",
            text: None,
            id: Some(tool_use_id.to_string()),
            name: Some(tool_name.to_string()),
            input: Some(json!({})),
        },
    };

    let input_delta = ClaudeStreamContentBlockDelta {
        type_field: "content_block_delta",
        index,
        delta: serde_json::to_value(ClaudeInputJsonDelta {
            type_field: "input_json_delta",
            partial_json: serde_json::to_string(action_input)
                .unwrap_or_else(|_| String::from("{}")),
        })
        .unwrap_or_else(|_| json!({})),
    };

    let block_stop = ClaudeStreamContentBlockStop {
        type_field: "content_block_stop",
        index,
    };

    vec![
        (
            "content_block_start",
            serde_json::to_value(block_start).unwrap_or_else(|_| json!({})),
        ),
        (
            "content_block_delta",
            serde_json::to_value(input_delta).unwrap_or_else(|_| json!({})),
        ),
        (
            "content_block_stop",
            serde_json::to_value(block_stop).unwrap_or_else(|_| json!({})),
        ),
    ]
}

fn build_claude_buffered_stream_events(
    message_id: &str,
    model: &str,
    input_tokens: u32,
    outcome: &PseudoToolRunOutcome,
) -> Vec<Result<Event, Infallible>> {
    let mut events = Vec::new();

    let message_start = ClaudeStreamMessageStart {
        type_field: "message_start",
        message: ClaudeStreamMessageMeta {
            id: message_id.to_string(),
            type_field: "message",
            role: "assistant",
            content: vec![],
            model: model.to_string(),
            stop_reason: None,
            usage: ClaudeUsage {
                input_tokens,
                output_tokens: 0,
            },
        },
    };
    events.push(Ok(Event::default()
        .event("message_start")
        .data(serde_json::to_string(&message_start).unwrap_or_default())));

    match &outcome.response {
        ParsedPseudoToolResponse::Final { content } => {
            let block_start = ClaudeStreamContentBlockStart {
                type_field: "content_block_start",
                index: 0,
                content_block: ClaudeContentBlock {
                    type_field: "text",
                    text: Some(String::new()),
                    id: None,
                    name: None,
                    input: None,
                },
            };
            events.push(Ok(Event::default()
                .event("content_block_start")
                .data(serde_json::to_string(&block_start).unwrap_or_default())));

            if !content.is_empty() {
                let delta = ClaudeStreamContentBlockDelta {
                    type_field: "content_block_delta",
                    index: 0,
                    delta: serde_json::to_value(ClaudeTextDelta {
                        type_field: "text_delta",
                        text: content.clone(),
                    })
                    .unwrap_or_else(|_| json!({})),
                };
                events.push(Ok(Event::default()
                    .event("content_block_delta")
                    .data(serde_json::to_string(&delta).unwrap_or_default())));
            }

            let block_stop = ClaudeStreamContentBlockStop {
                type_field: "content_block_stop",
                index: 0,
            };
            events.push(Ok(Event::default()
                .event("content_block_stop")
                .data(serde_json::to_string(&block_stop).unwrap_or_default())));

            let message_delta = ClaudeStreamMessageDelta {
                type_field: "message_delta",
                delta: ClaudeStopDelta {
                    stop_reason: "end_turn",
                },
                usage: ClaudeUsage {
                    input_tokens,
                    output_tokens: 0,
                },
            };
            events.push(Ok(Event::default()
                .event("message_delta")
                .data(serde_json::to_string(&message_delta).unwrap_or_default())));
        }
        ParsedPseudoToolResponse::Action {
            preamble_text,
            tool_name,
            action_input,
        } => {
            let mut next_index = 0;

            if let Some(text) = preamble_text.as_ref().filter(|text| !text.is_empty()) {
                let text_block_start = ClaudeStreamContentBlockStart {
                    type_field: "content_block_start",
                    index: next_index,
                    content_block: ClaudeContentBlock {
                        type_field: "text",
                        text: Some(String::new()),
                        id: None,
                        name: None,
                        input: None,
                    },
                };
                events.push(Ok(Event::default()
                    .event("content_block_start")
                    .data(serde_json::to_string(&text_block_start).unwrap_or_default())));

                let text_delta = ClaudeStreamContentBlockDelta {
                    type_field: "content_block_delta",
                    index: next_index,
                    delta: serde_json::to_value(ClaudeTextDelta {
                        type_field: "text_delta",
                        text: text.clone(),
                    })
                    .unwrap_or_else(|_| json!({})),
                };
                events.push(Ok(Event::default()
                    .event("content_block_delta")
                    .data(serde_json::to_string(&text_delta).unwrap_or_default())));

                let text_block_stop = ClaudeStreamContentBlockStop {
                    type_field: "content_block_stop",
                    index: next_index,
                };
                events.push(Ok(Event::default()
                    .event("content_block_stop")
                    .data(serde_json::to_string(&text_block_stop).unwrap_or_default())));

                next_index += 1;
            }

            let tool_use_id = format!("toolu_{}", uuid::Uuid::new_v4().as_simple());
            for (event_name, payload) in build_claude_tool_use_stream_payloads(
                next_index,
                &tool_use_id,
                tool_name,
                action_input,
            ) {
                events.push(Ok(Event::default()
                    .event(event_name)
                    .data(serde_json::to_string(&payload).unwrap_or_default())));
            }

            let message_delta = ClaudeStreamMessageDelta {
                type_field: "message_delta",
                delta: ClaudeStopDelta {
                    stop_reason: "tool_use",
                },
                usage: ClaudeUsage {
                    input_tokens,
                    output_tokens: 0,
                },
            };
            events.push(Ok(Event::default()
                .event("message_delta")
                .data(serde_json::to_string(&message_delta).unwrap_or_default())));
        }
    }

    let message_stop = ClaudeStreamMessageStop {
        type_field: "message_stop",
    };
    events.push(Ok(Event::default()
        .event("message_stop")
        .data(serde_json::to_string(&message_stop).unwrap_or_default())));

    events
}

fn ensure_authorized(headers: &HeaderMap, state: &AppState) -> Result<(), StatusCode> {
    match &state.settings.api_key {
        None => Ok(()),
        Some(expected) => {
            let valid = extract_bearer_token(headers)
                .map(|token| token == expected)
                .unwrap_or(false);
            if valid {
                Ok(())
            } else {
                Err(StatusCode::UNAUTHORIZED)
            }
        }
    }
}

fn mark_cookie_failure(cm: &mut CookieManager, cookie: &str, reason: String) {
    let kind = classify_cookie_failure(&reason);
    cm.mark_call_failure(cookie, kind, reason);
}

fn classify_cookie_failure(reason: &str) -> CookieFailureKind {
    let lower = reason.to_ascii_lowercase();

    let permanent_markers = [
        "onyx auth failed: 401",
        "onyx auth failed: 403",
        "http 401",
        "http 403",
        "http 429",
        "rate limit",
        "insufficient_quota",
        "quota exceeded",
        "quota exhausted",
        "usage limit exceeded",
        "usagelimitexceedederror",
        "llm_cost_cents",
        "credit balance",
        "额度耗尽",
    ];

    if permanent_markers
        .iter()
        .any(|marker| lower.contains(marker))
    {
        CookieFailureKind::Permanent
    } else {
        CookieFailureKind::Temporary
    }
}

fn extract_bearer_token(headers: &HeaderMap) -> Option<&str> {
    let auth = headers.get(AUTHORIZATION)?.to_str().ok()?;
    let token = auth.strip_prefix("Bearer ")?;
    if token.trim().is_empty() {
        None
    } else {
        Some(token.trim())
    }
}

fn extract_cookie_from_headers(headers: &reqwest::header::HeaderMap) -> Option<String> {
    for value in headers.get_all(SET_COOKIE) {
        let s = value.to_str().ok()?;
        for part in s.split(';') {
            let item = part.trim();
            if let Some(v) = item.strip_prefix("fastapiusersauth=") {
                return Some(v.to_string());
            }
        }
    }
    None
}

fn format_std_error_chain(err: &(dyn StdError + 'static)) -> String {
    let mut chain = Vec::new();
    chain.push(err.to_string());

    let mut source = err.source();
    while let Some(cause) = source {
        chain.push(cause.to_string());
        source = cause.source();
    }

    if chain.is_empty() {
        return String::from("unknown error");
    }

    let root_cause = chain
        .last()
        .cloned()
        .unwrap_or_else(|| String::from("unknown error"));
    format!(
        "{} | chain: {} | root_cause: {}",
        chain[0],
        chain.join(" -> "),
        root_cause
    )
}

fn truncate_for_error(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    let truncated = input.chars().take(max_chars).collect::<String>();
    format!("{truncated}...")
}

fn now_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

async fn append_proxy_audit_record(
    settings: &Settings,
    endpoint: &str,
    request: &serde_json::Value,
    raw_request_body: Option<&str>,
    status: StatusCode,
    response: &serde_json::Value,
) {
    let path_value = settings.request_audit_log_path.trim();
    if path_value.is_empty() {
        return;
    }

    let path = FsPath::new(path_value);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
        && let Err(err) = tokio::fs::create_dir_all(parent).await
    {
        error!(log_path = %path.display(), error = %err, "failed to create proxy audit log directory");
        return;
    }

    let record = ProxyAuditRecord {
        ts_ms: now_unix_ms(),
        endpoint: endpoint.to_string(),
        status: status.as_u16(),
        raw_request_body: raw_request_body.map(|s| s.to_string()),
        request: request.clone(),
        response: response.clone(),
    };

    let serialized = match serde_json::to_string(&record) {
        Ok(v) => v,
        Err(err) => {
            error!(log_path = %path.display(), error = %err, "failed to serialize proxy audit log");
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
                error!(log_path = %path.display(), error = %err, "failed to append proxy audit log");
            }
        }
        Err(err) => {
            error!(log_path = %path.display(), error = %err, "failed to open proxy audit log file");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PseudoToolRunOutcome, StatusCode, build_claude_response,
        build_claude_tool_use_stream_payloads, build_openai_assistant_message, build_state,
        classify_cookie_failure, next_round_robin_offset, refresh_all_cookies,
        rotate_cookie_values, spawn_cookie_refresh_task_with_interval,
    };
    use crate::config::Settings;
    use crate::cookie_manager::CookieFailureKind;
    use crate::pseudo_tools::ParsedPseudoToolResponse;
    use axum::{
        Router,
        extract::State,
        http::header::{COOKIE, SET_COOKIE},
        response::IntoResponse,
        routing::post,
    };
    use serde_json::json;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    #[test]
    fn classify_cookie_failure_marks_auth_error_as_permanent() {
        let reason = "onyx auth failed: 401 | chain: ...";
        assert_eq!(
            classify_cookie_failure(reason),
            CookieFailureKind::Permanent
        );
    }

    #[test]
    fn classify_cookie_failure_marks_rate_limit_as_permanent() {
        let reason = "onyx send-chat-message HTTP 429: rate limit exceeded";
        assert_eq!(
            classify_cookie_failure(reason),
            CookieFailureKind::Permanent
        );
    }

    #[test]
    fn classify_cookie_failure_marks_usage_limit_exceeded_as_permanent() {
        let reason = "empty upstream response: {\"error\":\"An unexpected error occurred while processing your request. Please try again later.\",\"stack_trace\":\"Traceback ... UsageLimitExceededError: Usage limit exceeded for llm_cost_cents: current usage 847.9482500000003, limit 800.0\"}";
        assert_eq!(
            classify_cookie_failure(reason),
            CookieFailureKind::Permanent
        );
    }

    #[test]
    fn classify_cookie_failure_marks_llm_cost_cents_limit_exceeded_as_permanent() {
        let reason = "empty upstream response: UsageLimitExceededError: Usage limit exceeded for llm_cost_cents";
        assert_eq!(
            classify_cookie_failure(reason),
            CookieFailureKind::Permanent
        );
    }

    #[test]
    fn classify_cookie_failure_marks_transport_error_as_temporary() {
        let reason = "failed to call send-chat-message | chain: connection refused";
        assert_eq!(
            classify_cookie_failure(reason),
            CookieFailureKind::Temporary
        );
    }

    #[test]
    fn rotate_cookie_values_applies_round_robin_offset() {
        let values = vec![
            "cookie-a".to_string(),
            "cookie-b".to_string(),
            "cookie-c".to_string(),
        ];
        let rotated = rotate_cookie_values(values, 1);
        assert_eq!(
            rotated,
            vec![
                "cookie-b".to_string(),
                "cookie-c".to_string(),
                "cookie-a".to_string()
            ]
        );
    }

    #[test]
    fn next_round_robin_offset_cycles_through_indexes() {
        let counter = AtomicUsize::new(0);
        assert_eq!(next_round_robin_offset(&counter, 3), 0);
        assert_eq!(next_round_robin_offset(&counter, 3), 1);
        assert_eq!(next_round_robin_offset(&counter, 3), 2);
        assert_eq!(next_round_robin_offset(&counter, 3), 0);
    }

    #[test]
    fn openai_assistant_message_keeps_text_and_tool_call() {
        let outcome = PseudoToolRunOutcome {
            response: ParsedPseudoToolResponse::Action {
                preamble_text: Some("I will check the directory first.".to_string()),
                tool_name: "bash".to_string(),
                action_input: json!({"command": "ls"}),
            },
            thinking: String::new(),
        };

        let message = build_openai_assistant_message(&outcome, false);
        assert_eq!(
            message.content.as_deref(),
            Some("I will check the directory first.")
        );
        assert_eq!(
            message.tool_calls.as_ref().map(|calls| calls.len()),
            Some(1)
        );
        assert_eq!(
            message.tool_calls.as_ref().unwrap()[0].function.name,
            "bash"
        );
    }

    #[test]
    fn openai_assistant_message_for_final_text_has_no_tool_calls() {
        let outcome = PseudoToolRunOutcome {
            response: ParsedPseudoToolResponse::Final {
                content: "Done editing the file.".to_string(),
            },
            thinking: String::new(),
        };

        let message = build_openai_assistant_message(&outcome, false);
        assert_eq!(message.content.as_deref(), Some("Done editing the file."));
        assert!(message.tool_calls.is_none());
    }

    #[test]
    fn openai_assistant_message_without_preamble_uses_null_content_for_tool_call() {
        let outcome = PseudoToolRunOutcome {
            response: ParsedPseudoToolResponse::Action {
                preamble_text: None,
                tool_name: "bash".to_string(),
                action_input: json!({"command": "ls"}),
            },
            thinking: String::new(),
        };

        let message = build_openai_assistant_message(&outcome, false);
        assert_eq!(message.content, None);
        assert_eq!(
            message.tool_calls.as_ref().map(|calls| calls.len()),
            Some(1)
        );
        assert_eq!(
            message.tool_calls.as_ref().unwrap()[0].function.name,
            "bash"
        );
    }

    #[test]
    fn claude_response_keeps_text_and_tool_use() {
        let outcome = PseudoToolRunOutcome {
            response: ParsedPseudoToolResponse::Action {
                preamble_text: Some("I will check the directory first.".to_string()),
                tool_name: "bash".to_string(),
                action_input: json!({"command": "ls"}),
            },
            thinking: String::new(),
        };

        let response = build_claude_response("claude-sonnet-4.5", 12, &outcome);
        assert_eq!(response.stop_reason, "tool_use");
        assert_eq!(response.content.len(), 2);
        assert_eq!(response.content[0].type_field, "text");
        assert_eq!(
            response.content[0].text.as_deref(),
            Some("I will check the directory first.")
        );
        assert_eq!(response.content[1].type_field, "tool_use");
        assert_eq!(response.content[1].name.as_deref(), Some("bash"));
        assert!(response.usage.output_tokens > 0);
    }

    #[test]
    fn claude_response_for_final_text_has_no_tool_use_block() {
        let outcome = PseudoToolRunOutcome {
            response: ParsedPseudoToolResponse::Final {
                content: "Done editing the file.".to_string(),
            },
            thinking: String::new(),
        };

        let response = build_claude_response("claude-sonnet-4.5", 12, &outcome);
        assert_eq!(response.stop_reason, "end_turn");
        assert_eq!(response.content.len(), 1);
        assert_eq!(response.content[0].type_field, "text");
        assert_eq!(
            response.content[0].text.as_deref(),
            Some("Done editing the file.")
        );
        assert!(response.usage.output_tokens > 0);
    }

    #[test]
    fn claude_tool_use_stream_payload_includes_input_json_delta() {
        let payloads = build_claude_tool_use_stream_payloads(
            0,
            "toolu_123",
            "bash",
            &json!({
                "command": "ls /home/nonewhite/Download/ 2>&1 || echo 'DIR_NOT_FOUND'",
                "description": "Check if Download directory exists"
            }),
        );

        assert_eq!(payloads.len(), 3);
        assert_eq!(payloads[0].0, "content_block_start");
        assert_eq!(payloads[1].0, "content_block_delta");
        assert_eq!(payloads[2].0, "content_block_stop");

        assert_eq!(payloads[0].1["content_block"]["type"], "tool_use");
        assert_eq!(payloads[0].1["content_block"]["name"], "bash");
        assert_eq!(payloads[0].1["content_block"]["input"], json!({}));

        assert_eq!(payloads[1].1["delta"]["type"], "input_json_delta");
        let partial_json = payloads[1].1["delta"]["partial_json"]
            .as_str()
            .expect("partial_json should be a string");
        let parsed: serde_json::Value =
            serde_json::from_str(partial_json).expect("partial_json should be valid JSON");
        assert_eq!(
            parsed["command"],
            "ls /home/nonewhite/Download/ 2>&1 || echo 'DIR_NOT_FOUND'"
        );
        assert_eq!(parsed["description"], "Check if Download directory exists");
    }

    async fn spawn_refresh_server(refreshed_cookie: &'static str) -> (String, oneshot::Sender<()>) {
        async fn refresh_handler(
            State(cookie_value): State<&'static str>,
            headers: axum::http::HeaderMap,
        ) -> impl IntoResponse {
            let cookie_header = headers
                .get(COOKIE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default()
                .to_string();

            if !cookie_header.contains("fastapiusersauth=") {
                return (StatusCode::UNAUTHORIZED, "missing auth cookie").into_response();
            }

            (
                StatusCode::OK,
                [(
                    SET_COOKIE,
                    format!("fastapiusersauth={cookie_value}; Path=/; HttpOnly"),
                )],
                "ok",
            )
                .into_response()
        }

        let app = Router::new()
            .route("/api/auth/refresh", post(refresh_handler))
            .with_state(refreshed_cookie);

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind refresh test server");
        let addr = listener.local_addr().expect("local addr");
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("refresh test server should run");
        });

        (format!("http://{}", addr), shutdown_tx)
    }

    #[tokio::test]
    async fn refresh_all_cookies_updates_stored_cookie_value() {
        let (base_url, shutdown_tx) = spawn_refresh_server("refreshed-cookie").await;
        let cookie_file = std::env::temp_dir().join(format!(
            "rust-proxy-cookie-refresh-test-{}.json",
            uuid::Uuid::new_v4()
        ));
        let state = build_state(Settings {
            onyx_base_url: base_url,
            onyx_auth_cookie: "stale-cookie".to_string(),
            api_key: None,
            cookie_persist_path: cookie_file.to_string_lossy().to_string(),
            ..Settings::default()
        })
        .expect("state should build");

        let result = refresh_all_cookies(&state).await;
        assert_eq!(result.total, 1);
        assert_eq!(result.refreshed, 1);
        assert_eq!(result.failed, 0);

        let mut cm = state.cookie_manager.write().await;
        let entries = cm.entries_mut();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].value, "refreshed-cookie");
        assert!(entries[0].last_refresh_ts.is_some());

        let _ = shutdown_tx.send(());
        let _ = std::fs::remove_file(cookie_file);
    }

    #[tokio::test]
    async fn auto_refresh_task_runs_immediately_on_startup() {
        let (base_url, shutdown_tx) = spawn_refresh_server("startup-refreshed-cookie").await;
        let cookie_file = std::env::temp_dir().join(format!(
            "rust-proxy-cookie-refresh-startup-test-{}.json",
            uuid::Uuid::new_v4()
        ));
        let state = build_state(Settings {
            onyx_base_url: base_url,
            onyx_auth_cookie: "startup-stale-cookie".to_string(),
            api_key: None,
            cookie_persist_path: cookie_file.to_string_lossy().to_string(),
            ..Settings::default()
        })
        .expect("state should build");

        let handle =
            spawn_cookie_refresh_task_with_interval(state.clone(), Duration::from_secs(3600));
        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut cm = state.cookie_manager.write().await;
        let entries = cm.entries_mut();
        assert_eq!(entries[0].value, "startup-refreshed-cookie");
        assert!(entries[0].last_refresh_ts.is_some());
        drop(cm);

        handle.abort();
        let _ = shutdown_tx.send(());
        let _ = std::fs::remove_file(cookie_file);
    }
}
