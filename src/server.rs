use std::convert::Infallible;
use std::error::Error as StdError;
use std::path::Path as FsPath;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    Json, Router,
    extract::{Path, State, rejection::JsonRejection},
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
use tracing::error;

use crate::{
    config::Settings,
    cookie_manager::{CookieFailureKind, CookieManager},
    models::{
        AssistantMessage, AssistantToolCall, AssistantToolCallFunction, ChatCompletionChunk,
        ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, ChunkChoice,
        ChunkDelta, ClaudeContentBlock,
        ClaudeMessagesRequest, ClaudeMessagesResponse, ClaudeStreamContentBlockDelta,
        ClaudeStreamContentBlockStart, ClaudeStreamContentBlockStop,
        ClaudeStreamMessageDelta, ClaudeStreamMessageStart, ClaudeStreamMessageStop,
        ClaudeStreamMessageMeta, ClaudeStopDelta, ClaudeTextDelta, ClaudeUsage,
        ModelItem, ModelsListResponse, Usage, DEFAULT_MODEL, supported_models,
    },
    onyx_client::{self, StreamEvent},
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
    request: serde_json::Value,
    response: serde_json::Value,
}

pub fn build_state(settings: Settings) -> anyhow::Result<AppState> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(settings.request_timeout_secs))
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
        .route("/api/cookies", get(cookies_handler).post(add_cookie_handler))
        .route("/api/cookies/{fingerprint}", delete(delete_cookie_handler))
        .route("/api/cookies/refresh", post(refresh_cookies_handler))
        .route("/auth/login", get(auth_login_page_handler).post(auth_login_handler))
        .with_state(state)
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
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    if let Err(status) = ensure_authorized(&headers, &state) {
        return (status, Json(json!({"error":"unauthorized"}))).into_response();
    }

    if req.messages.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":"messages must not be empty"})),
        )
            .into_response();
    }

    let request_log = serde_json::to_value(&req).unwrap_or_else(|_| json!({"error":"request_serialize_failed"}));

    let requested_tool_names = req.requested_tool_names();
    let forced_tool_name = req.forced_tool_name();

    if let Some(tool_call) = maybe_build_local_tool_call_from_chat_request(&req) {
        if req.stream.unwrap_or(false) {
            let chat_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
            let created = now_ts();
            let model = req.model.clone().unwrap_or_else(|| DEFAULT_MODEL.to_string());
            let tool_call_id = format!("call_{}", uuid::Uuid::new_v4().simple());

            let sse_stream = stream::unfold(0u8, move |phase| {
                let chat_id = chat_id.clone();
                let model = model.clone();
                let tool_call_id = tool_call_id.clone();
                let tool_name = tool_call.name.clone();
                let tool_args = tool_call.arguments.clone();
                async move {
                    match phase {
                        0 => {
                            // Chunk 1: Role and initial tool call block
                            let chunk = ChatCompletionChunk {
                                id: chat_id,
                                object: "chat.completion.chunk",
                                created,
                                model,
                                choices: vec![ChunkChoice {
                                    index: 0,
                                    delta: ChunkDelta {
                                        role: Some("assistant"),
                                        content: None,
                                        reasoning_content: None,
                                        tool_calls: Some(vec![AssistantToolCall {
                                            id: tool_call_id,
                                            kind: "function",
                                            function: AssistantToolCallFunction {
                                                name: tool_name,
                                                arguments: String::new(),
                                            },
                                            index: Some(0),
                                        }]),
                                    },
                                    finish_reason: None,
                                }],
                            };
                            let event = Event::default().data(serde_json::to_string(&chunk).unwrap());
                            Some((Ok::<_, Infallible>(event), 1))
                        }
                        1 => {
                            // Chunk 2: Arguments delta
                            let chunk = ChatCompletionChunk {
                                id: chat_id,
                                object: "chat.completion.chunk",
                                created,
                                model,
                                choices: vec![ChunkChoice {
                                    index: 0,
                                    delta: ChunkDelta {
                                        role: None,
                                        content: None,
                                        reasoning_content: None,
                                        tool_calls: Some(vec![AssistantToolCall {
                                            id: String::new(), // ID only required in first chunk
                                            kind: "function",
                                            function: AssistantToolCallFunction {
                                                name: String::new(),
                                                arguments: tool_args,
                                            },
                                            index: Some(0),
                                        }]),
                                    },
                                    finish_reason: None,
                                }],
                            };
                            let event = Event::default().data(serde_json::to_string(&chunk).unwrap());
                            Some((Ok::<_, Infallible>(event), 2))
                        }
                        2 => {
                            // Chunk 3: Stop reason
                            let chunk = ChatCompletionChunk {
                                id: chat_id,
                                object: "chat.completion.chunk",
                                created,
                                model,
                                choices: vec![ChunkChoice {
                                    index: 0,
                                    delta: ChunkDelta { role: None, content: None, reasoning_content: None, tool_calls: None },
                                    finish_reason: Some("tool_calls"),
                                }],
                            };
                            let event = Event::default().data(serde_json::to_string(&chunk).unwrap());
                            Some((Ok::<_, Infallible>(event), 99))
                        }
                        _ => None,
                    }
                }
            });
            return Sse::new(sse_stream).into_response();
        } else {
            let response = ChatCompletionResponse {
                id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                object: "chat.completion",
                created: now_ts(),
                model: req.model.clone().unwrap_or_else(|| DEFAULT_MODEL.to_string()),
                choices: vec![Choice {
                    index: 0,
                    message: AssistantMessage {
                        role: "assistant",
                        content: String::new(), // Should ideally be None, but kept for minimal change if Serialize allows empty string. Standard is null.
                        reasoning_content: None,
                        tool_calls: Some(vec![AssistantToolCall {
                            id: format!("call_{}", uuid::Uuid::new_v4().simple()),
                            kind: "function",
                            function: AssistantToolCallFunction {
                                name: tool_call.name,
                                arguments: tool_call.arguments,
                            },
                            index: Some(0),
                        }]),
                    },
                    finish_reason: "tool_calls",
                }],
                usage: Usage { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
            };
            let response_json = json!(response);
            append_proxy_audit_record(
                &state.settings,
                "/v1/chat/completions",
                &request_log,
                StatusCode::OK,
                &response_json,
            )
            .await;
            return (StatusCode::OK, Json(response_json)).into_response();
        }
        }
    if !req.stream.unwrap_or(false)
        && let Some(tool_call) = maybe_build_local_tool_call_from_chat_request(&req)
    {
        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion",
            created: now_ts(),
            model: req.model.clone().unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            choices: vec![Choice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: String::new(),
                    reasoning_content: None,
                    tool_calls: Some(vec![AssistantToolCall {
                        id: format!("call_{}", uuid::Uuid::new_v4().simple()),
                        kind: "function",
                        function: AssistantToolCallFunction {
                            name: tool_call.name,
                            arguments: tool_call.arguments,
                        },
                        index: Some(0),
                    }]),
                },
                finish_reason: "tool_calls",
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
            StatusCode::OK,
            &response_json,
        )
        .await;

        return (StatusCode::OK, Json(response_json)).into_response();
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
                StatusCode::BAD_GATEWAY,
                &response_json,
            )
            .await;
            return (
                StatusCode::BAD_GATEWAY,
                Json(response_json),
            )
                .into_response();
        }

        let chat_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = now_ts();
        let model_for_stream = model.clone();

        // Try each cookie until one works
        let mut last_err = String::from("unknown upstream error");
        let mut rx_opt = None;
        let mut _success_cookie: Option<String> = None;

        for cookie in &cookie_values {
            let (allowed_tool_ids, forced_tool_id) =
                match resolve_tool_selection_for_cookie(
                    &state.http_client,
                    &state.settings,
                    cookie,
                    &requested_tool_names,
                    forced_tool_name.as_deref(),
                )
                .await
                {
                    Ok(selection) => selection,
                    Err(err) => {
                        let err_msg = onyx_client::format_error_chain(&err);
                        let cookie_fp = crate::cookie_manager::fingerprint(cookie);
                        error!(
                            endpoint = "/v1/chat/completions",
                            mode = "stream",
                            cookie = %cookie_fp,
                            error = %err_msg,
                            "tool translation failed"
                        );
                        let mut cm = state.cookie_manager.write().await;
                        mark_cookie_failure(&mut cm, cookie, err_msg.clone());
                        cm.save();
                        last_err = err_msg;
                        continue;
                    }
                };

            match onyx_client::streaming_chat(
                &state.http_client,
                &state.settings,
                cookie,
                &req.messages,
                &model,
                allowed_tool_ids.as_deref(),
                forced_tool_id,
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
                StatusCode::BAD_GATEWAY,
                &response_json,
            )
            .await;
            return (
                StatusCode::BAD_GATEWAY,
                Json(response_json),
            )
                .into_response();
        };

        append_proxy_audit_record(
            &state.settings,
            "/v1/chat/completions",
            &request_log,
            StatusCode::OK,
            &json!({"stream":true,"status":"started"}),
        )
        .await;

        // Build SSE stream from the mpsc receiver
        let sse_stream = stream::unfold(
            (rx, chat_id, model_for_stream, created, include_reasoning, false),
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
                        Some((Ok::<_, Infallible>(event), (rx, chat_id, model, created, include_reasoning, false)))
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
                        Some((Ok::<_, Infallible>(event), (rx, chat_id, model, created, include_reasoning, false)))
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
                        Some((Ok::<_, Infallible>(event), (rx, chat_id, model, created, include_reasoning, false)))
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
                        Some((Ok::<_, Infallible>(stop_event), (rx, chat_id, model, created, include_reasoning, true)))
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
        let (allowed_tool_ids, forced_tool_id) = match resolve_tool_selection_for_cookie(
            &state.http_client,
            &state.settings,
            &cookie,
            &requested_tool_names,
            forced_tool_name.as_deref(),
        )
        .await
        {
            Ok(selection) => selection,
            Err(err) => {
                let err_msg = onyx_client::format_error_chain(&err);
                let cookie_fp = crate::cookie_manager::fingerprint(&cookie);
                error!(
                    endpoint = "/v1/chat/completions",
                    mode = "non_stream",
                    cookie = %cookie_fp,
                    error = %err_msg,
                    "tool translation failed"
                );
                let mut cm = state.cookie_manager.write().await;
                mark_cookie_failure(&mut cm, &cookie, err_msg.clone());
                cm.save();
                last_err = err_msg;
                continue;
            }
        };

        match onyx_client::full_chat(
            &state.http_client,
            &state.settings,
            &cookie,
            &req.messages,
            &model,
            allowed_tool_ids.as_deref(),
            forced_tool_id,
        )
        .await {
            Ok((content, thinking)) => {
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
                            content,
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
        StatusCode::BAD_GATEWAY,
        &response_json,
    )
    .await;
    (
        StatusCode::BAD_GATEWAY,
        Json(response_json),
    )
        .into_response()
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
                let kind = classify_cookie_failure(&reason);
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
                entry.last_error = Some(reason);
                failed += 1;
            }
            Err(err) => {
                let reason = format_std_error_chain(&err);
                error!(endpoint = "/api/cookies/refresh", error = %reason, "cookie refresh transport failure");
                let kind = classify_cookie_failure(&reason);
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
                entry.last_error = Some(reason);
                failed += 1;
            }
        }
    }

    cm.save();

    Ok(Json(RefreshResponse {
        total,
        refreshed,
        failed,
    }))
}

/// Anthropic Claude-compatible Messages endpoint.
/// Accepts requests in Claude format and converts to/from Onyx backend.
async fn v1_messages_handler(
    headers: HeaderMap,
    State(state): State<AppState>,
    payload: Result<Json<ClaudeMessagesRequest>, JsonRejection>,
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

    let req = match payload {
        Ok(Json(req)) => req,
        Err(rej) => {
            let reason = rej.body_text();
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

    let request_log = serde_json::to_value(&req).unwrap_or_else(|_| json!({"error":"request_serialize_failed"}));

    // Convert Claude messages to our internal ChatMessage format
    let mut chat_messages: Vec<ChatMessage> = Vec::new();
    if let Some(sys) = &req.system {
        chat_messages.push(ChatMessage {
            role: "system".to_string(),
            content: sys.clone(),
        });
    }

    // When tools are present, inject tool definitions as a system instruction
    let has_tools = req.tools.as_ref().is_some_and(|t| !t.is_empty());
    if has_tools {
        let tools = req.tools.as_ref().unwrap();
        let tool_instruction = build_tool_injection_prompt(tools);
        chat_messages.push(ChatMessage {
            role: "system".to_string(),
            content: tool_instruction,
        });
    }

    for m in &req.messages {
        chat_messages.push(ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        });
    }

    let model = req.model.clone();
    let requested_tool_names = req.requested_tool_names();
    let forced_tool_name = req.forced_tool_name();

    // Estimate input tokens (rough estimation: 1 token ~= 4 chars)
    let input_tokens = {
        let mut chars = req.system.as_ref().map(|s| s.len()).unwrap_or(0);
        for m in &req.messages {
            chars += m.role.len() + m.content.len() + 4;
        }
        (chars / 4) as u32 + 10 // +10 for structural overhead
    };

    if req.stream.unwrap_or(false)
        && let Some(tool_call) = maybe_build_local_tool_call_from_claude_request(&req)
    {
        let tool_input: serde_json::Value =
            serde_json::from_str(&tool_call.arguments).unwrap_or_else(|_| json!({}));
        let tool_input_json = serde_json::to_string(&tool_input).unwrap_or_else(|_| "{}".to_string());
        let msg_id = format!("msg_{}", uuid::Uuid::new_v4().as_simple());
        let model_for_stream = model.clone();
        let tool_use_id = format!("toolu_{}", uuid::Uuid::new_v4().simple());
        let tool_name = tool_call.name;

        let sse_stream = stream::unfold(0u8, move |phase| {
            let msg_id = msg_id.clone();
            let model = model_for_stream.clone();
            let tool_use_id = tool_use_id.clone();
            let tool_name = tool_name.clone();
            let tool_input_json = tool_input_json.clone();
            async move {
                match phase {
                    0 => {
                        let start = ClaudeStreamMessageStart {
                            type_field: "message_start",
                            message: ClaudeStreamMessageMeta {
                                id: msg_id,
                                type_field: "message",
                                role: "assistant",
                                content: vec![],
                                model,
                                stop_reason: None,
                                usage: ClaudeUsage {
                                    input_tokens,
                                    output_tokens: 0,
                                },
                            },
                        };
                        let data = serde_json::to_string(&start).unwrap_or_default();
                        let event = Event::default().event("message_start").data(data);
                        Some((Ok::<_, Infallible>(event), 1))
                    }
                    1 => {
                        let block_start = ClaudeStreamContentBlockStart {
                            type_field: "content_block_start",
                            index: 0,
                            content_block: ClaudeContentBlock {
                                type_field: "tool_use",
                                text: None,
                                id: Some(tool_use_id),
                                name: Some(tool_name),
                                input: Some(json!({})),
                            },
                        };
                        let data = serde_json::to_string(&block_start).unwrap_or_default();
                        let event = Event::default().event("content_block_start").data(data);
                        Some((Ok::<_, Infallible>(event), 2))
                    }
                    2 => {
                        let data = serde_json::to_string(&json!({
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": tool_input_json,
                            }
                        }))
                        .unwrap_or_default();
                        let event = Event::default().event("content_block_delta").data(data);
                        Some((Ok::<_, Infallible>(event), 3))
                    }
                    3 => {
                        let block_stop = ClaudeStreamContentBlockStop {
                            type_field: "content_block_stop",
                            index: 0,
                        };
                        let data = serde_json::to_string(&block_stop).unwrap_or_default();
                        let event = Event::default().event("content_block_stop").data(data);
                        Some((Ok::<_, Infallible>(event), 4))
                    }
                    4 => {
                        let msg_delta = ClaudeStreamMessageDelta {
                            type_field: "message_delta",
                            delta: ClaudeStopDelta {
                                stop_reason: "tool_use",
                            },
                            usage: ClaudeUsage {
                                input_tokens,
                                output_tokens: 0,
                            },
                        };
                        let data = serde_json::to_string(&msg_delta).unwrap_or_default();
                        let event = Event::default().event("message_delta").data(data);
                        Some((Ok::<_, Infallible>(event), 5))
                    }
                    5 => {
                        let stop = ClaudeStreamMessageStop {
                            type_field: "message_stop",
                        };
                        let data = serde_json::to_string(&stop).unwrap_or_default();
                        let event = Event::default().event("message_stop").data(data);
                        Some((Ok::<_, Infallible>(event), 99))
                    }
                    _ => None,
                }
            }
        });

        return Sse::new(sse_stream).into_response();
    }

    if !req.stream.unwrap_or(false)
        && let Some(tool_call) = maybe_build_local_tool_call_from_claude_request(&req)
    {
        let tool_input: serde_json::Value =
            serde_json::from_str(&tool_call.arguments).unwrap_or_else(|_| json!({}));
        let response = ClaudeMessagesResponse {
            id: format!("msg_{}", uuid::Uuid::new_v4().as_simple()),
            type_field: "message",
            role: "assistant",
            content: vec![ClaudeContentBlock {
                type_field: "tool_use",
                text: None,
                id: Some(format!("toolu_{}", uuid::Uuid::new_v4().simple())),
                name: Some(tool_call.name),
                input: Some(tool_input),
            }],
            model: model.clone(),
            stop_reason: "tool_use",
            usage: ClaudeUsage {
                input_tokens,
                output_tokens: 0,
            },
        };
        let response_json = json!(response);
        append_proxy_audit_record(
            &state.settings,
            "/v1/messages",
            &request_log,
            StatusCode::OK,
            &response_json,
        )
        .await;
        return (StatusCode::OK, Json(response_json)).into_response();
    }

    let cookie_values = active_cookie_values_round_robin(&state).await;

    if cookie_values.is_empty() {
        let response_json = json!({"type":"error","error":{"type":"api_error","message":"no cookies configured"}});
        append_proxy_audit_record(
            &state.settings,
            "/v1/messages",
            &request_log,
            StatusCode::BAD_GATEWAY,
            &response_json,
        )
        .await;
        return (
            StatusCode::BAD_GATEWAY,
            Json(response_json),
        )
            .into_response();
    }

    // ----- Streaming mode -----
    if req.stream.unwrap_or(false) {
        // When tools are present, use full_chat + post-process to detect tool_calls.
        // This sacrifices streaming latency but ensures reliable tool_call detection.
        if has_tools {
            append_proxy_audit_record(
                &state.settings,
                "/v1/messages",
                &request_log,
                StatusCode::OK,
                &json!({"stream":true,"status":"started","mode":"tool-aware"}),
            )
            .await;
            return claude_stream_with_tools(
                &state, &cookie_values, &chat_messages, &model,
                &requested_tool_names, forced_tool_name.as_deref(), input_tokens,
            ).await;
        }

        let msg_id = format!("msg_{}", uuid::Uuid::new_v4().as_simple());
        let model_for_stream = model.clone();

        let mut last_err = String::from("unknown upstream error");
        let mut rx_opt: Option<tokio::sync::mpsc::Receiver<StreamEvent>> = None;

        for cookie in &cookie_values {
            let (allowed_tool_ids, forced_tool_id) =
                match resolve_tool_selection_for_cookie(
                    &state.http_client,
                    &state.settings,
                    cookie,
                    &requested_tool_names,
                    forced_tool_name.as_deref(),
                )
                .await
                {
                    Ok(selection) => selection,
                    Err(err) => {
                        let err_msg = onyx_client::format_error_chain(&err);
                        let cookie_fp = crate::cookie_manager::fingerprint(cookie);
                        error!(
                            endpoint = "/v1/messages",
                            mode = "stream",
                            cookie = %cookie_fp,
                            error = %err_msg,
                            "tool translation failed"
                        );
                        let mut cm = state.cookie_manager.write().await;
                        mark_cookie_failure(&mut cm, cookie, err_msg.clone());
                        cm.save();
                        last_err = err_msg;
                        continue;
                    }
                };

            match onyx_client::streaming_chat(
                &state.http_client,
                &state.settings,
                cookie,
                &chat_messages,
                &model,
                allowed_tool_ids.as_deref(),
                forced_tool_id,
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
                StatusCode::BAD_GATEWAY,
                &response_json,
            )
            .await;
            return (
                StatusCode::BAD_GATEWAY,
                Json(response_json),
            )
                .into_response();
        };

        append_proxy_audit_record(
            &state.settings,
            "/v1/messages",
            &request_log,
            StatusCode::OK,
            &json!({"stream":true,"status":"started"}),
        )
        .await;

        // Build SSE stream for Claude format
        // State: (rx, msg_id, model, phase, output_tokens, input_tokens, buffer)
        // phase: 0=message_start, 1=content_block_start, 2=streaming, 3=content_block_stop, 4=message_delta, 5=message_stop
        let sse_stream = stream::unfold(
            (rx, msg_id, model_for_stream, 0u8, 0u32, input_tokens, None::<StreamEvent>),
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
                        Some((Ok::<_, Infallible>(event), (rx, msg_id, model, 1, output_tokens, input_tokens, None)))
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
                        Some((Ok::<_, Infallible>(event), (rx, msg_id, model, 2, output_tokens, input_tokens, None)))
                    }
                    2 => {
                        // Streaming content_block_delta
                        let event = if let Some(e) = event_buffer.take() {
                            Some(e)
                        } else {
                            rx.recv().await
                        };

                        match event {
                            Some(StreamEvent::Content(text)) | Some(StreamEvent::Reasoning(text)) => {
                                let new_tokens = output_tokens + 1;
                                let delta = ClaudeStreamContentBlockDelta {
                                    type_field: "content_block_delta",
                                    index: 0,
                                    delta: ClaudeTextDelta {
                                        type_field: "text_delta",
                                        text,
                                    },
                                };
                                let data = serde_json::to_string(&delta).unwrap_or_default();
                                let event = Event::default().event("content_block_delta").data(data);
                                Some((Ok::<_, Infallible>(event), (rx, msg_id, model, 2, new_tokens, input_tokens, None)))
                            }
                            Some(StreamEvent::Role) => {
                                // Skip role event, stay in phase 2 and recv again
                                let event = Event::default().event("ping").data("{\"type\":\"ping\"}");
                                Some((Ok::<_, Infallible>(event), (rx, msg_id, model, 2, output_tokens, input_tokens, None)))
                            }
                            Some(StreamEvent::Done) | None => {
                                // Transition to content_block_stop
                                let block_stop = ClaudeStreamContentBlockStop {
                                    type_field: "content_block_stop",
                                    index: 0,
                                };
                                let data = serde_json::to_string(&block_stop).unwrap_or_default();
                                let event = Event::default().event("content_block_stop").data(data);
                                Some((Ok::<_, Infallible>(event), (rx, msg_id, model, 4, output_tokens, input_tokens, None)))
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
                        Some((Ok::<_, Infallible>(event), (rx, msg_id, model, 5, output_tokens, input_tokens, None)))
                    }
                    5 => {
                        // Emit message_stop
                        let stop = ClaudeStreamMessageStop {
                            type_field: "message_stop",
                        };
                        let data = serde_json::to_string(&stop).unwrap_or_default();
                        let event = Event::default().event("message_stop").data(data);
                        Some((Ok::<_, Infallible>(event), (rx, msg_id, model, 99, output_tokens, input_tokens, None)))
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
        let (allowed_tool_ids, forced_tool_id) = match resolve_tool_selection_for_cookie(
            &state.http_client,
            &state.settings,
            cookie,
            &requested_tool_names,
            forced_tool_name.as_deref(),
        )
        .await
        {
            Ok(selection) => selection,
            Err(err) => {
                let err_msg = onyx_client::format_error_chain(&err);
                let cookie_fp = crate::cookie_manager::fingerprint(cookie);
                error!(
                    endpoint = "/v1/messages",
                    mode = "non_stream",
                    cookie = %cookie_fp,
                    error = %err_msg,
                    "tool translation failed"
                );
                let mut cm = state.cookie_manager.write().await;
                mark_cookie_failure(&mut cm, cookie, err_msg.clone());
                cm.save();
                last_err = err_msg;
                continue;
            }
        };

        match onyx_client::full_chat(
            &state.http_client,
            &state.settings,
            cookie,
            &chat_messages,
            &model,
            allowed_tool_ids.as_deref(),
            forced_tool_id,
        )
        .await
        {
            Ok((content, _thinking)) => {
                let mut cm = state.cookie_manager.write().await;
                cm.mark_call_success(cookie);
                cm.save();

                // Check for tool_calls in the response text
                let (tool_calls, remaining_text) = if has_tools {
                    onyx_client::extract_tool_calls_from_text(&content)
                } else {
                    (vec![], content.clone())
                };

                if !tool_calls.is_empty() {
                    // Build content blocks: optional text block + tool_use blocks
                    let mut blocks = Vec::new();
                    if !remaining_text.is_empty() {
                        blocks.push(ClaudeContentBlock {
                            type_field: "text",
                            text: Some(remaining_text),
                            id: None,
                            name: None,
                            input: None,
                        });
                    }
                    for tc in &tool_calls {
                        blocks.push(ClaudeContentBlock {
                            type_field: "tool_use",
                            text: None,
                            id: Some(format!("toolu_{}", uuid::Uuid::new_v4().simple())),
                            name: Some(tc.name.clone()),
                            input: Some(tc.arguments.clone()),
                        });
                    }

        let response = ClaudeMessagesResponse {
                        id: format!("msg_{}", uuid::Uuid::new_v4().as_simple()),
                        type_field: "message",
                        role: "assistant",
                        content: blocks,
                        model: model.clone(),
                        stop_reason: "tool_use",
                        usage: ClaudeUsage {
                            input_tokens,
                            output_tokens: 0,
                        },
                    };
                    let response_json = json!(response);
                    append_proxy_audit_record(
                        &state.settings,
                        "/v1/messages",
                        &request_log,
                        StatusCode::OK,
                        &response_json,
                    )
                    .await;
                    return (StatusCode::OK, Json(response_json)).into_response();
                }

                let response = ClaudeMessagesResponse {
                    id: format!("msg_{}", uuid::Uuid::new_v4().as_simple()),
                    type_field: "message",
                    role: "assistant",
                    content: vec![ClaudeContentBlock {
                        type_field: "text",
                        text: Some(content),
                        id: None,
                        name: None,
                        input: None,
                    }],
                    model: model.clone(),
                    stop_reason: "end_turn",
                    usage: ClaudeUsage {
                        input_tokens,
                        output_tokens: 0,
                    },
                };
                let response_json = json!(response);
                append_proxy_audit_record(
                    &state.settings,
                    "/v1/messages",
                    &request_log,
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
        StatusCode::BAD_GATEWAY,
        &response_json,
    )
    .await;
    (
        StatusCode::BAD_GATEWAY,
        Json(response_json),
    )
        .into_response()
}

/// Build a system prompt that instructs the LLM to use `<tool_call>` markers.
fn build_tool_injection_prompt(tools: &[crate::models::ClaudeToolDefinition]) -> String {
    let mut s = String::from(
        "You have access to the following tools. When you need to use a tool, you MUST respond with a tool call in this EXACT format (including the XML tags):\n\n\
         <tool_call>\n\
         {\"name\": \"tool_name\", \"arguments\": {\"param1\": \"value1\"}}\n\
         </tool_call>\n\n\
         You can make multiple tool calls by using multiple <tool_call> blocks.\n\
         If you don't need to use any tool, respond normally with text.\n\
         IMPORTANT: Always use the <tool_call> tags, never call tools in any other format.\n\n\
         Available tools:\n",
    );

    for tool in tools {
        s.push_str(&format!("- {}", tool.name));
        if let Some(desc) = &tool.description {
            s.push_str(&format!(": {}", desc));
        }
        s.push('\n');
        if let Some(schema) = &tool.input_schema {
            s.push_str(&format!("  Parameters: {}\n", schema));
        }
    }

    s
}

/// Handle Claude streaming requests when tools are present.
/// Collects the full response via `full_chat()`, parses tool_calls,
/// then emits SSE events in the correct Claude streaming format.
async fn claude_stream_with_tools(
    state: &AppState,
    cookie_values: &[String],
    chat_messages: &[ChatMessage],
    model: &str,
    requested_tool_names: &[String],
    forced_tool_name: Option<&str>,
    input_tokens: u32,
) -> axum::response::Response {
    let msg_id = format!("msg_{}", uuid::Uuid::new_v4().as_simple());
    let model_owned = model.to_string();
    let state = state.clone();
    let cookie_values = cookie_values.to_vec();
    let chat_messages = chat_messages.to_vec();
    let requested_tool_names = requested_tool_names.to_vec();
    let forced_tool_name = forced_tool_name.map(|s| s.to_string());

    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(64);
    tokio::spawn(async move {
        let _ = tx
            .send(immediate_message_start_event(
                msg_id.clone(),
                model_owned.clone(),
                input_tokens,
            ))
            .await;

        let mut last_err = String::from("unknown upstream error");

        for cookie in cookie_values {
            let (allowed_tool_ids, forced_tool_id) =
                match resolve_tool_selection_for_cookie(
                    &state.http_client,
                    &state.settings,
                    &cookie,
                    &requested_tool_names,
                    forced_tool_name.as_deref(),
                )
                .await
                {
                    Ok(selection) => selection,
                    Err(err) => {
                        let err_msg = onyx_client::format_error_chain(&err);
                        let cookie_fp = crate::cookie_manager::fingerprint(&cookie);
                        error!(
                            endpoint = "/v1/messages",
                            mode = "stream+tools",
                            cookie = %cookie_fp,
                            error = %err_msg,
                            "tool translation failed"
                        );
                        let mut cm = state.cookie_manager.write().await;
                        mark_cookie_failure(&mut cm, &cookie, err_msg.clone());
                        cm.save();
                        last_err = err_msg;
                        continue;
                    }
                };

            match onyx_client::full_chat(
                &state.http_client,
                &state.settings,
                &cookie,
                &chat_messages,
                &model_owned,
                allowed_tool_ids.as_deref(),
                forced_tool_id,
            )
            .await
            {
                Ok((content, _thinking)) => {
                    let mut cm = state.cookie_manager.write().await;
                    cm.mark_call_success(&cookie);
                    cm.save();

                    let (tool_calls, remaining_text) =
                        onyx_client::extract_tool_calls_from_text(&content);

                    let events = if !tool_calls.is_empty() {
                        build_claude_tool_use_followup_events(
                            input_tokens,
                            remaining_text,
                            tool_calls,
                        )
                    } else {
                        build_claude_text_followup_events(input_tokens, content)
                    };

                    for event in events {
                        if tx.send(event).await.is_err() {
                            return;
                        }
                    }
                    return;
                }
                Err(err) => {
                    let err_msg = onyx_client::format_error_chain(&err);
                    let cookie_fp = crate::cookie_manager::fingerprint(&cookie);
                    error!(
                        endpoint = "/v1/messages",
                        mode = "stream+tools",
                        cookie = %cookie_fp,
                        error = %err_msg,
                        "claude stream+tools upstream failed"
                    );
                    let mut cm = state.cookie_manager.write().await;
                    mark_cookie_failure(&mut cm, &cookie, err_msg.clone());
                    cm.save();
                    last_err = err_msg;
                }
            }
        }

        for event in build_claude_text_followup_events(
            input_tokens,
            format!("upstream failure: {last_err}"),
        ) {
            if tx.send(event).await.is_err() {
                return;
            }
        }
    });

    let sse_stream = stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|event| (Ok::<_, Infallible>(event), rx))
    });
    Sse::new(sse_stream).into_response()
}

fn immediate_message_start_event(msg_id: String, model: String, input_tokens: u32) -> Event {
    Event::default().event("message_start").data(
        serde_json::to_string(&ClaudeStreamMessageStart {
            type_field: "message_start",
            message: ClaudeStreamMessageMeta {
                id: msg_id,
                type_field: "message",
                role: "assistant",
                content: vec![],
                model,
                stop_reason: None,
                usage: ClaudeUsage {
                    input_tokens,
                    output_tokens: 0,
                },
            },
        })
        .unwrap_or_default(),
    )
}

fn build_claude_tool_use_followup_events(
    input_tokens: u32,
    remaining_text: String,
    tool_calls: Vec<onyx_client::ParsedToolCall>,
) -> Vec<Event> {
    let mut events: Vec<Event> = Vec::new();
    let mut block_index: u8 = 0;

    if !remaining_text.is_empty() {
        events.push(
            Event::default().event("content_block_start").data(
                serde_json::to_string(&ClaudeStreamContentBlockStart {
                    type_field: "content_block_start",
                    index: block_index,
                    content_block: ClaudeContentBlock {
                        type_field: "text",
                        text: Some(String::new()),
                        id: None,
                        name: None,
                        input: None,
                    },
                })
                .unwrap_or_default(),
            ),
        );
        events.push(
            Event::default().event("content_block_delta").data(
                serde_json::to_string(&ClaudeStreamContentBlockDelta {
                    type_field: "content_block_delta",
                    index: block_index,
                    delta: ClaudeTextDelta {
                        type_field: "text_delta",
                        text: remaining_text,
                    },
                })
                .unwrap_or_default(),
            ),
        );
        events.push(
            Event::default().event("content_block_stop").data(
                serde_json::to_string(&ClaudeStreamContentBlockStop {
                    type_field: "content_block_stop",
                    index: block_index,
                })
                .unwrap_or_default(),
            ),
        );
        block_index += 1;
    }

    for tc in &tool_calls {
        let tool_use_id = format!("toolu_{}", uuid::Uuid::new_v4().simple());
        let tool_input_json = serde_json::to_string(&tc.arguments).unwrap_or_else(|_| "{}".to_string());

        events.push(
            Event::default().event("content_block_start").data(
                serde_json::to_string(&ClaudeStreamContentBlockStart {
                    type_field: "content_block_start",
                    index: block_index,
                    content_block: ClaudeContentBlock {
                        type_field: "tool_use",
                        text: None,
                        id: Some(tool_use_id),
                        name: Some(tc.name.clone()),
                        input: Some(json!({})),
                    },
                })
                .unwrap_or_default(),
            ),
        );
        events.push(
            Event::default().event("content_block_delta").data(
                serde_json::to_string(&json!({
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": tool_input_json,
                    }
                }))
                .unwrap_or_default(),
            ),
        );
        events.push(
            Event::default().event("content_block_stop").data(
                serde_json::to_string(&ClaudeStreamContentBlockStop {
                    type_field: "content_block_stop",
                    index: block_index,
                })
                .unwrap_or_default(),
            ),
        );
        block_index += 1;
    }

    events.push(
        Event::default().event("message_delta").data(
            serde_json::to_string(&ClaudeStreamMessageDelta {
                type_field: "message_delta",
                delta: ClaudeStopDelta {
                    stop_reason: "tool_use",
                },
                usage: ClaudeUsage {
                    input_tokens,
                    output_tokens: 0,
                },
            })
            .unwrap_or_default(),
        ),
    );
    events.push(
        Event::default().event("message_stop").data(
            serde_json::to_string(&ClaudeStreamMessageStop {
                type_field: "message_stop",
            })
            .unwrap_or_default(),
        ),
    );
    events
}

fn build_claude_text_followup_events(input_tokens: u32, content: String) -> Vec<Event> {
    let mut events: Vec<Event> = Vec::new();
    events.push(Event::default().event("content_block_start").data(
        serde_json::to_string(&ClaudeStreamContentBlockStart {
            type_field: "content_block_start",
            index: 0,
            content_block: ClaudeContentBlock {
                type_field: "text",
                text: Some(String::new()),
                id: None,
                name: None,
                input: None,
            },
        })
        .unwrap_or_default(),
    ));

    events.push(Event::default().event("content_block_delta").data(
        serde_json::to_string(&ClaudeStreamContentBlockDelta {
            type_field: "content_block_delta",
            index: 0,
            delta: ClaudeTextDelta {
                type_field: "text_delta",
                text: content,
            },
        })
        .unwrap_or_default(),
    ));

    events.push(Event::default().event("content_block_stop").data(
        serde_json::to_string(&ClaudeStreamContentBlockStop {
            type_field: "content_block_stop",
            index: 0,
        })
        .unwrap_or_default(),
    ));

    events.push(Event::default().event("message_delta").data(
        serde_json::to_string(&ClaudeStreamMessageDelta {
            type_field: "message_delta",
            delta: ClaudeStopDelta {
                stop_reason: "end_turn",
            },
            usage: ClaudeUsage {
                input_tokens,
                output_tokens: 0,
            },
        })
        .unwrap_or_default(),
    ));

    events.push(Event::default().event("message_stop").data(
        serde_json::to_string(&ClaudeStreamMessageStop {
            type_field: "message_stop",
        })
        .unwrap_or_default(),
    ));

    events
}

async fn resolve_tool_selection_for_cookie(
    http_client: &reqwest::Client,
    settings: &Settings,
    cookie: &str,
    requested_tool_names: &[String],
    forced_tool_name: Option<&str>,
) -> anyhow::Result<(Option<Vec<i64>>, Option<i64>)> {
    if requested_tool_names.is_empty() && forced_tool_name.is_none() {
        return Ok((None, None));
    }

    let available_tools = onyx_client::fetch_available_tools(http_client, settings, cookie).await?;
    Ok(onyx_client::resolve_tool_selection_by_name(
        &available_tools,
        requested_tool_names,
        forced_tool_name,
    ))
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

    if permanent_markers.iter().any(|marker| lower.contains(marker)) {
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

    let root_cause = chain.last().cloned().unwrap_or_else(|| String::from("unknown error"));
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

#[derive(Debug)]
struct LocalToolCall {
    name: String,
    arguments: String,
}

fn maybe_build_local_tool_call_from_chat_request(req: &ChatCompletionRequest) -> Option<LocalToolCall> {
    let tool_names = req.requested_tool_names();
    let user_message = req
        .messages
        .iter()
        .rev()
        .find(|msg| msg.role.eq_ignore_ascii_case("user"))
        .map(|msg| msg.content.as_str())?;
    maybe_build_local_tool_call(&tool_names, user_message)
}

fn maybe_build_local_tool_call_from_claude_request(
    req: &ClaudeMessagesRequest,
) -> Option<LocalToolCall> {
    if req.has_tool_result_message() {
        return None;
    }

    let tool_names = req.requested_tool_names();
    let user_message = req
        .messages
        .iter()
        .rev()
        .find(|msg| msg.role.eq_ignore_ascii_case("user"))
        .map(|msg| msg.content.as_str())?;
    maybe_build_local_tool_call(&tool_names, user_message)
}

fn maybe_build_local_tool_call(tool_names: &[String], user_message: &str) -> Option<LocalToolCall> {
    let supports_bash = tool_names.iter().any(|name| name.eq_ignore_ascii_case("bash"));
    let supports_write = tool_names
        .iter()
        .any(|name| name.eq_ignore_ascii_case("write") || name.eq_ignore_ascii_case("write_file"));

    if supports_bash
        && let Some(command) = extract_bash_command_from_text(user_message)
    {
        return Some(LocalToolCall {
            name: "bash".to_string(),
            arguments: json!({"command": command}).to_string(),
        });
    }

    let path = extract_file_path_from_text(user_message)?;
    let content = extract_write_content_from_text(user_message)?;

    if supports_write {
        return Some(LocalToolCall {
            name: "write".to_string(),
            arguments: json!({"filePath": path, "content": content}).to_string(),
        });
    }

    if supports_bash {
        let parent = std::path::Path::new(&path)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .filter(|p| !p.is_empty())
            .unwrap_or_else(|| ".".to_string());

        let command = format!(
            "mkdir -p \"{}\" && printf %s \"{}\" > \"{}\"",
            escape_shell_double_quotes(&parent),
            escape_shell_double_quotes(&content),
            escape_shell_double_quotes(&path)
        );
        return Some(LocalToolCall {
            name: "bash".to_string(),
            arguments: json!({"command": command}).to_string(),
        });
    }

    None
}

fn extract_bash_command_from_text(input: &str) -> Option<String> {
    for marker in ["run:", "run：", "command:", "command：", "cmd:", "cmd："] {
        if let Some(pos) = input.to_ascii_lowercase().find(marker) {
            let start = pos + marker.len();
            let command = input[start..]
                .trim()
                .trim_matches('`')
                .trim_matches('"')
                .trim_matches('\'')
                .trim();
            if !command.is_empty() {
                return Some(command.to_string());
            }
        }
    }

    None
}

fn extract_file_path_from_text(input: &str) -> Option<String> {
    if let Some(start) = input.find("/home/") {
        let mut candidate = String::new();
        for ch in input[start..].chars() {
            if ch.is_ascii_alphanumeric() || matches!(ch, '/' | '.' | '_' | '-') {
                candidate.push(ch);
            } else {
                break;
            }
        }

        if candidate.contains('.') && !candidate.ends_with('/') {
            return Some(candidate);
        }

        let dir = if candidate.ends_with('/') {
            candidate
        } else if let Some(pos) = candidate.rfind('/') {
            candidate[..=pos].to_string()
        } else {
            return None;
        };

        let filename = extract_filename_from_text(input)?;
        return Some(format!("{dir}{filename}"));
    }

    extract_filename_from_text(input)
}

fn extract_filename_from_text(input: &str) -> Option<String> {
    let mut current = String::new();
    let mut candidates = Vec::new();
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
            current.push(ch);
        } else {
            if current.contains('.') {
                candidates.push(current.clone());
            }
            current.clear();
        }
    }
    if current.contains('.') {
        candidates.push(current);
    }
    candidates.into_iter().find(|name| !name.starts_with('/'))
}

fn extract_write_content_from_text(input: &str) -> Option<String> {
    if let Some(start) = input.find("写入") {
        let content = input[start + "写入".len()..].trim();
        if !content.is_empty() {
            return Some(content.trim_matches('"').trim_matches('\'').to_string());
        }
    }

    let lower = input.to_ascii_lowercase();
    for marker in ["with content", "content"] {
        if let Some(pos) = lower.find(marker) {
            let raw = input[pos + marker.len()..].trim();
            let cleaned = raw
                .trim_start_matches([':', '：', '=', '-', ' '])
                .trim_start_matches("is ")
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .trim();
            if !cleaned.is_empty() {
                return Some(cleaned.to_string());
            }
        }
    }

    None
}

fn escape_shell_double_quotes(input: &str) -> String {
    input.replace('\\', "\\\\").replace('"', "\\\"")
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
        classify_cookie_failure, immediate_message_start_event,
        maybe_build_local_tool_call_from_chat_request,
        maybe_build_local_tool_call_from_claude_request, next_round_robin_offset,
        rotate_cookie_values,
    };
    use crate::cookie_manager::CookieFailureKind;
    use crate::models::{ChatCompletionRequest, ClaudeMessagesRequest};
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn classify_cookie_failure_marks_auth_error_as_permanent() {
        let reason = "onyx auth failed: 401 | chain: ...";
        assert_eq!(classify_cookie_failure(reason), CookieFailureKind::Permanent);
    }

    #[test]
    fn classify_cookie_failure_marks_rate_limit_as_permanent() {
        let reason = "onyx send-chat-message HTTP 429: rate limit exceeded";
        assert_eq!(classify_cookie_failure(reason), CookieFailureKind::Permanent);
    }

    #[test]
    fn classify_cookie_failure_marks_usage_limit_exceeded_as_permanent() {
        let reason = "empty upstream response: {\"error\":\"An unexpected error occurred while processing your request. Please try again later.\",\"stack_trace\":\"Traceback ... UsageLimitExceededError: Usage limit exceeded for llm_cost_cents: current usage 847.9482500000003, limit 800.0\"}";
        assert_eq!(classify_cookie_failure(reason), CookieFailureKind::Permanent);
    }

    #[test]
    fn classify_cookie_failure_marks_llm_cost_cents_limit_exceeded_as_permanent() {
        let reason = "empty upstream response: UsageLimitExceededError: Usage limit exceeded for llm_cost_cents";
        assert_eq!(classify_cookie_failure(reason), CookieFailureKind::Permanent);
    }

    #[test]
    fn classify_cookie_failure_marks_transport_error_as_temporary() {
        let reason = "failed to call send-chat-message | chain: connection refused";
        assert_eq!(classify_cookie_failure(reason), CookieFailureKind::Temporary);
    }

    #[test]
    fn local_bash_tool_call_is_generated_for_create_and_write_request() {
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "user", "content": "帮我在 /home/nonewhite/Download/1234.txt 中写入 123"}
            ],
            "tools": [
                {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}
            ],
            "stream": false
        }))
        .expect("request should deserialize");

        let tool_call = maybe_build_local_tool_call_from_chat_request(&req).expect("tool call expected");
        assert_eq!(tool_call.name, "bash");
        assert!(tool_call.arguments.contains("/home/nonewhite/Download/1234.txt"));
        assert!(tool_call.arguments.contains("printf %s \\\"123\\\""));
    }

    #[test]
    fn local_bash_tool_call_handles_no_whitespace_chinese_path_prompt() {
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "user", "content": "在/home/nonewhite/Download/中创建一个1234.txt文件并且在里面写入123"}
            ],
            "tools": [
                {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}
            ],
            "stream": false
        }))
        .expect("request should deserialize");

        let tool_call = maybe_build_local_tool_call_from_chat_request(&req).expect("tool call expected");
        assert_eq!(tool_call.name, "bash");
        assert!(tool_call.arguments.contains("/home/nonewhite/Download/1234.txt"));
    }

    #[test]
    fn claude_messages_can_generate_local_tool_use_call() {
        let req: ClaudeMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "user", "content": "帮我在 /home/nonewhite/Download/1234.txt 中写入 123"}
            ],
            "tools": [
                {"name": "bash", "description": "run shell", "input_schema": {"type": "object"}}
            ],
            "max_tokens": 1024,
            "stream": false
        }))
        .expect("request should deserialize");

        let tool_call = maybe_build_local_tool_call_from_claude_request(&req).expect("tool call expected");
        assert_eq!(tool_call.name, "bash");
        assert!(tool_call.arguments.contains("/home/nonewhite/Download/1234.txt"));
    }

    #[test]
    fn claude_messages_single_bash_tool_request_generates_command_tool_call() {
        let req: ClaudeMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "user", "content": "Use the bash tool to run: pwd"}
            ],
            "tools": [
                {"name": "bash", "description": "run shell", "input_schema": {"type": "object"}}
            ],
            "max_tokens": 1024,
            "stream": false
        }))
        .expect("request should deserialize");

        let tool_call = maybe_build_local_tool_call_from_claude_request(&req).expect("tool call expected");
        assert_eq!(tool_call.name, "bash");
        assert!(tool_call.arguments.contains("\"command\":\"pwd\""));
    }

    #[test]
    fn claude_messages_english_create_file_prefers_write_tool_when_available() {
        let req: ClaudeMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "user", "content": "Please create 1234.txt with content 1234"}
            ],
            "tools": [
                {"name": "bash", "description": "run shell", "input_schema": {"type": "object"}},
                {"name": "write", "description": "write file", "input_schema": {"type": "object"}}
            ],
            "max_tokens": 1024,
            "stream": false
        }))
        .expect("request should deserialize");

        let tool_call = maybe_build_local_tool_call_from_claude_request(&req).expect("tool call expected");
        assert_eq!(tool_call.name, "write");
        assert!(tool_call.arguments.contains("\"filePath\":\"1234.txt\""));
        assert!(tool_call.arguments.contains("\"content\":\"1234\""));
    }

    #[test]
    fn claude_messages_with_tool_result_does_not_generate_local_tool_call() {
        let req: ClaudeMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "claude-opus-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "run: pwd"
                        }
                    ]
                }
            ],
            "tools": [
                {"name": "bash", "description": "run shell", "input_schema": {"type": "object"}}
            ],
            "max_tokens": 1024,
            "stream": true
        }))
        .expect("request should deserialize");

        let tool_call = maybe_build_local_tool_call_from_claude_request(&req);
        assert!(
            tool_call.is_none(),
            "local tool call must be skipped when request already contains tool_result blocks"
        );
    }

    #[test]
    fn rotate_cookie_values_applies_round_robin_offset() {
        let values = vec!["cookie-a".to_string(), "cookie-b".to_string(), "cookie-c".to_string()];
        let rotated = rotate_cookie_values(values, 1);
        assert_eq!(
            rotated,
            vec!["cookie-b".to_string(), "cookie-c".to_string(), "cookie-a".to_string()]
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
    fn immediate_message_start_event_uses_expected_claude_shape() {
        let event = immediate_message_start_event("msg_123".to_string(), "claude-opus-4.6".to_string(), 42);
        let rendered = format!("{:?}", event);
        assert!(rendered.contains("message_start"));
        assert!(rendered.contains("msg_123"));
        assert!(rendered.contains("claude-opus-4.6"));
    }
}
