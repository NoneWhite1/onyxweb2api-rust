use std::env;

#[derive(Debug, Clone)]
pub struct Settings {
    pub host: String,
    pub onyx_base_url: String,
    pub onyx_auth_cookie: String,
    pub onyx_persona_id: i64,
    pub onyx_origin: String,
    pub onyx_origin_url: String,
    pub onyx_referer: String,
    pub api_key: Option<String>,
    pub port: u16,
    pub log_level: String,
    pub request_timeout_secs: u64,
    pub cookie_persist_path: String,
    pub request_error_log_path: String,
}

impl Settings {
    pub fn from_env() -> Self {
        let _ = dotenvy::dotenv();

        let onyx_base_url =
            env::var("ONYX_BASE_URL").unwrap_or_else(|_| "https://cloud.onyx.app".to_string());
        let host = env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
        let onyx_auth_cookie = env::var("ONYX_AUTH_COOKIE").unwrap_or_default();
        let onyx_persona_id = env::var("ONYX_PERSONA_ID")
            .ok()
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(0);
        let onyx_origin = env::var("ONYX_ORIGIN").unwrap_or_else(|_| "webapp".to_string());
        let onyx_origin_url = env::var("ONYX_ORIGIN_URL").unwrap_or_else(|_| onyx_base_url.clone());
        let onyx_referer =
            env::var("ONYX_REFERER").unwrap_or_else(|_| "https://cloud.onyx.app/app".to_string());
        let api_key = env::var("API_KEY")
            .ok()
            .and_then(|v| if v.is_empty() { None } else { Some(v) });
        let port = env::var("PORT")
            .ok()
            .and_then(|v| v.parse::<u16>().ok())
            .unwrap_or(8897);
        let log_level = env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string());
        let request_timeout_secs = env::var("REQUEST_TIMEOUT")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(300);
        let cookie_persist_path =
            env::var("COOKIE_PERSIST_PATH").unwrap_or_else(|_| "cookies.json".to_string());
        let request_error_log_path = env::var("REQUEST_ERROR_LOG_PATH")
            .unwrap_or_else(|_| "request_error_records.jsonl".to_string());

        Self {
            host,
            onyx_base_url,
            onyx_auth_cookie,
            onyx_persona_id,
            onyx_origin,
            onyx_origin_url,
            onyx_referer,
            api_key,
            port,
            log_level,
            request_timeout_secs,
            cookie_persist_path,
            request_error_log_path,
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self::from_env()
    }
}
