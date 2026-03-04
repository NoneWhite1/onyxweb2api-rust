use rust_proxy::config::Settings;
use rust_proxy::server::{build_router, build_state};
use tokio::net::TcpListener;
use tracing::info;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let settings = Settings::default();
    let addr = format!("{}:{}", settings.host, settings.port);
    let listener = TcpListener::bind(&addr)
        .await
        .expect("failed to bind tcp listener");

    info!("rust-proxy listening on http://{}", addr);
    info!("onyx base url: {}", settings.onyx_base_url);

    let state = build_state(settings).expect("failed to build app state");

    axum::serve(listener, build_router(state))
        .await
        .expect("axum server error");
}
