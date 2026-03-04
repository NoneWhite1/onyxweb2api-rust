pub mod config;
pub mod cookie_manager;
pub mod models;
pub mod onyx_client;
pub mod server;

#[cfg(test)]
mod tests {
    use crate::config::Settings;

    #[test]
    fn settings_defaults_load() {
        let s = Settings::default();
        assert_eq!(s.onyx_base_url, "https://cloud.onyx.app");
    }
}
