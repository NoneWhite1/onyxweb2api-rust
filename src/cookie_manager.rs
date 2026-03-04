use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CookieEntry {
    pub value: String,
    pub fingerprint: String,
    #[serde(default)]
    pub exhausted: bool,
    #[serde(default)]
    pub temporary_failures: u32,
    pub last_refresh_ts: Option<u64>,
    #[serde(default)]
    pub cooldown_until_ts: Option<u64>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CookieFailureKind {
    Temporary,
    Permanent,
}

#[derive(Debug, Clone, Serialize)]
pub struct CookieView {
    pub fingerprint: String,
    pub preview: String,
    pub exhausted: bool,
    pub last_refresh_ts: Option<u64>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CookieStats {
    pub total: usize,
    pub active: usize,
    pub exhausted: usize,
}

#[derive(Debug)]
pub struct CookieManager {
    cookies: Vec<CookieEntry>,
    persist_path: Option<PathBuf>,
}

impl Default for CookieManager {
    fn default() -> Self {
        Self {
            cookies: Vec::new(),
            persist_path: None,
        }
    }
}

impl CookieManager {
    pub fn load_or_create(persist_path: &str, env_cookies: &str) -> Self {
        let path = PathBuf::from(persist_path);
        let mut manager = if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(data) => {
                    let cookies: Vec<CookieEntry> = serde_json::from_str(&data).unwrap_or_default();
                    Self {
                        cookies,
                        persist_path: Some(path.clone()),
                    }
                }
                Err(_) => Self {
                    cookies: Vec::new(),
                    persist_path: Some(path.clone()),
                },
            }
        } else {
            Self {
                cookies: Vec::new(),
                persist_path: Some(path.clone()),
            }
        };

        // Merge env cookies (add any not already persisted)
        let existing_values: HashSet<String> =
            manager.cookies.iter().map(|c| c.value.clone()).collect();
        for part in env_cookies.split(',') {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value = extract_fastapiusersauth(trimmed).unwrap_or_else(|| trimmed.to_string());
            if value.is_empty() || existing_values.contains(&value) {
                continue;
            }
            manager.cookies.push(CookieEntry {
                fingerprint: fingerprint(&value),
                value,
                exhausted: false,
                temporary_failures: 0,
                last_refresh_ts: None,
                cooldown_until_ts: None,
                last_error: None,
            });
        }

        manager.save();
        manager
    }

    pub fn from_auth_cookie(raw: &str) -> Self {
        let mut seen = HashSet::new();
        let mut cookies = Vec::new();

        for part in raw.split(',') {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value = extract_fastapiusersauth(trimmed).unwrap_or_else(|| trimmed.to_string());
            if value.is_empty() || seen.contains(&value) {
                continue;
            }
            seen.insert(value.clone());
            cookies.push(CookieEntry {
                fingerprint: fingerprint(&value),
                value,
                exhausted: false,
                temporary_failures: 0,
                last_refresh_ts: None,
                cooldown_until_ts: None,
                last_error: None,
            });
        }

        Self {
            cookies,
            persist_path: None,
        }
    }

    pub fn stats(&self) -> CookieStats {
        let exhausted = self.cookies.iter().filter(|c| c.exhausted).count();
        let now = now_ts();
        let cooling_down = self
            .cookies
            .iter()
            .filter(|c| !c.exhausted && c.cooldown_until_ts.is_some_and(|until| until > now))
            .count();
        CookieStats {
            total: self.cookies.len(),
            exhausted,
            active: self
                .cookies
                .len()
                .saturating_sub(exhausted)
                .saturating_sub(cooling_down),
        }
    }

    pub fn views(&self) -> Vec<CookieView> {
        self.cookies
            .iter()
            .map(|c| CookieView {
                fingerprint: c.fingerprint.clone(),
                preview: preview_cookie(&c.value),
                exhausted: c.exhausted,
                last_refresh_ts: c.last_refresh_ts,
                last_error: c.last_error.clone(),
            })
            .collect()
    }

    pub fn entries_mut(&mut self) -> &mut [CookieEntry] {
        &mut self.cookies
    }

    pub fn active_cookie_values(&self) -> Vec<String> {
        let now = now_ts();
        let mut active: Vec<String> = self
            .cookies
            .iter()
            .filter(|c| !c.exhausted)
            .filter(|c| c.cooldown_until_ts.is_none_or(|until| until <= now))
            .map(|c| c.value.clone())
            .collect();
        if active.is_empty() {
            active = self.cookies.iter().map(|c| c.value.clone()).collect();
        }
        active
    }

    pub fn mark_call_success(&mut self, cookie_value: &str) {
        if let Some(entry) = self.cookies.iter_mut().find(|c| c.value == cookie_value) {
            entry.exhausted = false;
            entry.temporary_failures = 0;
            entry.cooldown_until_ts = None;
            entry.last_error = None;
        }
    }

    pub fn mark_call_failure(
        &mut self,
        cookie_value: &str,
        kind: CookieFailureKind,
        error: String,
    ) {
        const TEMP_FAILURE_THRESHOLD: u32 = 3;
        const TEMP_FAILURE_COOLDOWN_SECS: u64 = 120;

        if let Some(entry) = self.cookies.iter_mut().find(|c| c.value == cookie_value) {
            match kind {
                CookieFailureKind::Permanent => {
                    entry.exhausted = true;
                    entry.temporary_failures = 0;
                    entry.cooldown_until_ts = None;
                }
                CookieFailureKind::Temporary => {
                    entry.temporary_failures = entry.temporary_failures.saturating_add(1);
                    if entry.temporary_failures >= TEMP_FAILURE_THRESHOLD {
                        entry.cooldown_until_ts =
                            Some(now_ts().saturating_add(TEMP_FAILURE_COOLDOWN_SECS));
                        entry.temporary_failures = 0;
                    }
                }
            }
            entry.last_error = Some(error);
        }
    }

    pub fn add_cookie(&mut self, raw: &str) -> bool {
        let value = extract_fastapiusersauth(raw).unwrap_or_else(|| raw.trim().to_string());
        if value.is_empty() || self.cookies.iter().any(|c| c.value == value) {
            return false;
        }

        self.cookies.push(CookieEntry {
            fingerprint: fingerprint(&value),
            value,
            exhausted: false,
            temporary_failures: 0,
            last_refresh_ts: None,
            cooldown_until_ts: None,
            last_error: None,
        });
        self.save();
        true
    }

    pub fn remove_by_fingerprint(&mut self, fingerprint: &str) -> bool {
        if let Some(idx) = self
            .cookies
            .iter()
            .position(|c| c.fingerprint == fingerprint)
        {
            self.cookies.remove(idx);
            self.save();
            return true;
        }
        false
    }

    pub fn save(&self) {
        if let Some(path) = &self.persist_path {
            if let Ok(data) = serde_json::to_string_pretty(&self.cookies) {
                let _ = std::fs::write(path, data);
            }
        }
    }
}

fn extract_fastapiusersauth(input: &str) -> Option<String> {
    if !input.contains('=') {
        return None;
    }
    for segment in input.split(';') {
        let item = segment.trim();
        if let Some(value) = item.strip_prefix("fastapiusersauth=") {
            return Some(value.trim().to_string());
        }
    }
    None
}

fn preview_cookie(value: &str) -> String {
    if value.len() <= 10 {
        return "***".to_string();
    }
    format!("{}...{}", &value[..6], &value[value.len() - 4..])
}

pub fn fingerprint(value: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn now_ts() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::{CookieEntry, CookieFailureKind, CookieManager};

    fn manager_with_cookie(value: &str) -> CookieManager {
        CookieManager {
            cookies: vec![CookieEntry {
                value: value.to_string(),
                fingerprint: super::fingerprint(value),
                exhausted: false,
                temporary_failures: 0,
                last_refresh_ts: None,
                cooldown_until_ts: None,
                last_error: None,
            }],
            persist_path: None,
        }
    }

    #[test]
    fn temporary_failure_does_not_exhaust_cookie() {
        let mut manager = manager_with_cookie("cookie-A");
        manager.mark_call_failure(
            "cookie-A",
            CookieFailureKind::Temporary,
            "upstream 500".to_string(),
        );

        assert!(!manager.entries_mut()[0].exhausted);
        assert_eq!(manager.entries_mut()[0].temporary_failures, 1);
    }

    #[test]
    fn permanent_failure_exhausts_cookie_immediately() {
        let mut manager = manager_with_cookie("cookie-A");
        manager.mark_call_failure(
            "cookie-A",
            CookieFailureKind::Permanent,
            "onyx auth failed: 401".to_string(),
        );

        assert!(manager.entries_mut()[0].exhausted);
    }
}
