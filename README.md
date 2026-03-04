# OnyxWeb2API — Rust Proxy

Convert [Onyx Cloud](https://cloud.onyx.app/) web chat into **OpenAI** and **Anthropic Claude** compatible API endpoints.

Use any OpenAI/Claude-compatible client (NextChat, LobeChat, Continue, Cursor, etc.) with your Onyx Cloud account — no official API key required.

## Features

- **OpenAI-compatible API** — `POST /v1/chat/completions` (streaming + non-streaming)
- **Claude-compatible API** — `POST /v1/messages` (streaming + non-streaming)
- **Multi-cookie pool** — round-robin across multiple Onyx accounts for load balancing
- **Cookie management dashboard** — add, remove, monitor cookie status via web UI
- **Auto cookie capture** — login with email/password to automatically capture auth cookies
- **SSE streaming** — real-time token-by-token streaming for both API formats
- **API key protection** — optional Bearer token / x-api-key authentication

## Supported Models

| Model | Provider |
|-------|----------|
| `gpt-5.2` | OpenAI |
| `gpt-5-mini` | OpenAI |
| `gpt-4.1` | OpenAI |
| `gpt-4o` | OpenAI |
| `o3` | OpenAI |
| `claude-opus-4.6` | Anthropic |
| `claude-opus-4.5` | Anthropic |
| `claude-sonnet-4.5` | Anthropic |

## Quick Start

### 1. Get your Onyx auth cookie

**Option A — Email/Password (automatic):**

Navigate to `http://your-server:8897/auth/login` and login with your Onyx Cloud credentials. The cookie is captured automatically.

**Option B — Browser DevTools (manual):**

1. Login to [cloud.onyx.app](https://cloud.onyx.app/)
2. Open DevTools → Application → Cookies
3. Copy the value of `fastapiusersauth`

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set ONYX_AUTH_COOKIE to your cookie value
```

### 3. Run

**From source:**
```bash
cargo build --release
./target/release/rust-proxy
```

**With environment variables:**
```bash
ONYX_AUTH_COOKIE=your-cookie PORT=8897 ./rust-proxy
```

### 4. Use

**OpenAI format:**
```bash
curl http://localhost:8897/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.2",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

**Claude format:**
```bash
curl http://localhost:8897/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-opus-4.6",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 1024,
    "stream": true
  }'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat (stream + non-stream) |
| `/v1/messages` | POST | Claude-compatible messages (stream + non-stream) |
| `/v1/models` | GET | List available models |
| `/ui` | GET | Dashboard — cookie management, status overview |
| `/auth/login` | GET/POST | Login page + email/password cookie capture |
| `/api/cookies` | GET/POST | Manage cookie pool (list / add) |
| `/api/cookies/:fingerprint` | DELETE | Remove a cookie by fingerprint |
| `/api/status` | GET | Proxy status and cookie health |
| `/health` | GET | Health check |

## Client Configuration

### NextChat / LobeChat / ChatBox

- **API Base URL:** `http://your-server:8897/v1`
- **API Key:** your configured `API_KEY` (or leave empty if not set)
- **Model:** `gpt-5.2`, `claude-opus-4.6`, etc.

### Continue / Cursor / OpenCode

```json
{
  "apiBase": "http://your-server:8897/v1",
  "apiKey": "your-api-key",
  "model": "gpt-5.2"
}
```

### Claude-compatible clients

- **API Base URL:** `http://your-server:8897`
- **API Key header:** `x-api-key: your-api-key`
- **Endpoint:** `/v1/messages`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ONYX_BASE_URL` | `https://cloud.onyx.app` | Onyx Cloud base URL |
| `ONYX_AUTH_COOKIE` | *(required)* | Auth cookie(s), comma-separated |
| `ONYX_PERSONA_ID` | `0` | Persona ID (0 = default) |
| `ONYX_ORIGIN` | `webapp` | Origin field for API payload |
| `ONYX_ORIGIN_URL` | `$ONYX_BASE_URL` | HTTP Origin header value |
| `ONYX_REFERER` | `https://cloud.onyx.app/app` | HTTP Referer header |
| `API_KEY` | *(none)* | Optional API key to protect endpoints |
| `PORT` | `8897` | Server port |
| `LOG_LEVEL` | `info` | Log level (trace/debug/info/warn/error) |
| `REQUEST_TIMEOUT` | `300` | Request timeout in seconds |

## Building

```bash
# Debug build
cargo build

# Release build (optimized, ~5MB stripped)
cargo build --release
strip target/release/rust-proxy

# Run tests
cargo test
```

## Architecture

```
Client (NextChat, Cursor, etc.)
  │
  ├── POST /v1/chat/completions  (OpenAI format)
  ├── POST /v1/messages          (Claude format)
  │
  ▼
┌─────────────────────┐
│   Rust Proxy (Axum)  │
│                     │
│  ┌── Auth check ──┐ │
│  ┌── Cookie pool ─┐ │  ← Round-robin multi-account
│  ┌── Model map ───┐ │  ← gpt-5.2 → ("OpenAI","gpt-5.2")
│  ┌── SSE bridge ──┐ │  ← Onyx SSE → OpenAI/Claude SSE
└────��────────────────┘
  │
  ▼
cloud.onyx.app (Onyx Cloud backend)
```

## License

MIT
