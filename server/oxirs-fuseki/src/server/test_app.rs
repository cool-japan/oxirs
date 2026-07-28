//! Public test app builder for HTTP-level parity tests.
//!
//! `build_jena_router` exposes the subset of routes that map to Apache
//! Jena Fuseki's HTTP surface — SPARQL query/update, Graph Store
//! Protocol, bulk upload, RDF Patch, and the `$/...` admin endpoints.
//!
//! Both flat (`/sparql`, `/update`, `/data`, `/upload`, `/patch`) and
//! dataset-prefixed (`/{dataset}/sparql`, `/{dataset}/update`,
//! `/{dataset}/data`, `/{dataset}/upload`, `/{dataset}/patch`) routes
//! are wired so the router responds to the URL conventions used by
//! Jena Fuseki. The dataset segment is currently parsed but not
//! plumbed into the underlying single-dataset store; this matches the
//! behaviour of OxiRS Fuseki running in single-dataset mode.

use crate::handlers;
use crate::server::AppState;
use axum::{
    routing::{get, post},
    Router,
};
use oxirs_core::rdf_store::ConcreteStore;
use std::sync::Arc;

/// Build an axum `Router` that serves the Jena-Fuseki-equivalent HTTP
/// surface for parity testing.
///
/// `state` is the application state. `concrete` is the underlying
/// `ConcreteStore` used by GSP / upload / patch handlers, which take
/// `Arc<ConcreteStore>` directly rather than the multi-dataset
/// [`crate::store::Store`] wrapper.
///
/// The router exposes:
/// - `GET|POST /sparql` and `GET|POST /{dataset}/sparql` for SPARQL queries
/// - `POST /update` and `POST /{dataset}/update` for SPARQL updates
/// - `GET|HEAD|PUT|POST|DELETE|OPTIONS /data` and `/{dataset}/data`
///   for the Graph Store Protocol
/// - `POST /upload` and `POST /{dataset}/upload` for bulk uploads
/// - `POST /patch` and `POST /{dataset}/patch` for RDF Patch
/// - `GET /$/ping`, `GET /$/server`, `GET /$/stats`, `GET /$/datasets`
///   for the standard Jena admin endpoints
pub fn build_jena_router(state: Arc<AppState>, concrete: Arc<ConcreteStore>) -> Router {
    let sparql_routes: Router = Router::new()
        .route(
            "/sparql",
            get(handlers::query_handler_get).post(handlers::query_handler_post),
        )
        .route(
            "/{dataset}/sparql",
            get(handlers::query_handler_get).post(handlers::query_handler_post),
        )
        .route(
            "/{dataset}/query",
            get(handlers::query_handler_get).post(handlers::query_handler_post),
        )
        .route("/update", post(handlers::sparql_refactored::update_handler))
        .route(
            "/{dataset}/update",
            post(handlers::sparql_refactored::update_handler),
        )
        .with_state(state.clone());

    let admin_routes: Router = Router::new()
        .route("/$/ping", get(jena_ping_handler))
        .route("/$/server", get(handlers::admin::server_info))
        .route("/$/stats", get(handlers::admin::server_stats))
        .route("/$/datasets", get(handlers::admin::list_datasets))
        .with_state(state);

    let store_routes: Router = Router::new()
        .route(
            "/data",
            get(handlers::gsp::read::handle_gsp_get::<ConcreteStore>)
                .head(handlers::gsp::read::handle_gsp_head::<ConcreteStore>)
                .put(handlers::gsp::write::handle_gsp_put::<ConcreteStore>)
                .post(handlers::gsp::write::handle_gsp_post::<ConcreteStore>)
                .delete(handlers::gsp::write::handle_gsp_delete::<ConcreteStore>)
                .options(handlers::gsp::read::handle_gsp_options),
        )
        .route(
            "/{dataset}/data",
            get(handlers::gsp::read::handle_gsp_get::<ConcreteStore>)
                .head(handlers::gsp::read::handle_gsp_head::<ConcreteStore>)
                .put(handlers::gsp::write::handle_gsp_put::<ConcreteStore>)
                .post(handlers::gsp::write::handle_gsp_post::<ConcreteStore>)
                .delete(handlers::gsp::write::handle_gsp_delete::<ConcreteStore>)
                .options(handlers::gsp::read::handle_gsp_options),
        )
        .route(
            "/upload",
            post(handlers::upload::handle_upload::<ConcreteStore>),
        )
        .route(
            "/{dataset}/upload",
            post(handlers::upload::handle_upload::<ConcreteStore>),
        )
        .route(
            "/patch",
            post(handlers::patch::handle_patch::<ConcreteStore>),
        )
        .route(
            "/{dataset}/patch",
            post(handlers::patch::handle_patch::<ConcreteStore>),
        )
        .with_state(concrete);

    Router::new()
        .merge(sparql_routes)
        .merge(admin_routes)
        .merge(store_routes)
}

/// Lightweight `$/ping` handler. Apache Jena Fuseki returns the date in
/// ISO-8601 format with a `text/plain` body.
async fn jena_ping_handler() -> impl axum::response::IntoResponse {
    use axum::http::header::CONTENT_TYPE;
    let body = chrono::Utc::now().to_rfc3339();
    ([(CONTENT_TYPE, "text/plain; charset=utf-8")], body)
}

/// Build a baseline `AppState` with all optional services disabled.
///
/// This is the smallest `AppState` that the public handlers will accept.
/// It uses `Store::new()` for in-memory storage and supplies empty
/// service handles. Useful for parity tests that need only the routing
/// + handler logic, not the production middleware stack.
pub fn build_minimal_app_state(
    store: crate::store::Store,
    config: crate::config::ServerConfig,
) -> AppState {
    AppState {
        store,
        config,
        auth_service: None,
        metrics_service: None,
        performance_service: None,
        query_optimizer: None,
        subscription_manager: None,
        federation_manager: None,
        streaming_manager: None,
        concurrency_manager: None,
        memory_manager: None,
        batch_executor: None,
        stream_manager: None,
        dataset_manager: None,
        api_key_service: None,
        security_auditor: None,
        ddos_protector: None,
        load_balancer: None,
        edge_cache_manager: None,
        performance_profiler: None,
        notification_manager: None,
        backup_manager: None,
        recovery_manager: None,
        disaster_recovery: None,
        certificate_rotation: None,
        http2_manager: None,
        http3_manager: None,
        adaptive_execution_engine: None,
        rebac_manager: None,
        prefix_store: Arc::new(handlers::PrefixStore::new()),
        task_manager: Arc::new(handlers::TaskManager::new()),
        request_logger: Arc::new(handlers::RequestLogger::new()),
        startup_time: std::time::Instant::now(),
        system_monitor: Arc::new(parking_lot::Mutex::new(sysinfo::System::new())),
        audit_logger: Arc::new(oxirs_core::audit::InMemoryAuditLogger::new()),
        sparql_cache: None,
        #[cfg(feature = "rate-limit")]
        rate_limiter: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ServerConfig;
    use crate::store::Store;

    #[test]
    fn test_build_minimal_app_state() {
        let store = Store::new().expect("store");
        let config = ServerConfig::default();
        let state = build_minimal_app_state(store, config);
        assert!(state.metrics_service.is_none());
        assert!(state.auth_service.is_none());
    }

    #[tokio::test]
    async fn test_build_jena_router_constructable() {
        let store = Store::new().expect("store");
        let config = ServerConfig::default();
        let state = Arc::new(build_minimal_app_state(store, config));
        let concrete = Arc::new(ConcreteStore::new().expect("concrete"));
        let _router = build_jena_router(state, concrete);
        // Router was built without panic.
    }
}
