//! RESTful and GraphQL API endpoints for embedding services
//!
//! This module provides production-ready HTTP APIs for embedding generation,
//! model management, and batch processing capabilities.
//!
//! ## Module Structure
//!
//! - **config**: API configuration and state management
//! - **types**: Request/response types and data structures
//! - **graphql**: GraphQL schema and resolvers
//! - **handlers**: HTTP endpoint handlers
//! - **routes**: API route definitions
//! - **helpers**: Utility functions

pub mod config;
pub mod graphql;
pub mod handlers;
pub mod helpers;
pub mod routes;
pub mod types;

// Re-export main types and functions
pub use config::*;
#[cfg(feature = "graphql")]
pub use graphql::create_schema;
pub use routes::create_router;
pub use types::*;

/// Start the API server
#[cfg(feature = "api-server")]
pub async fn start_server(state: ApiState) -> anyhow::Result<()> {
    use axum::Router;
    use std::net::SocketAddr;
    use tower::ServiceBuilder;
    use tower_http::{cors::CorsLayer, timeout::TimeoutLayer, trace::TraceLayer};
    use tracing::info;

    let app = create_router(state.clone()).layer(
        ServiceBuilder::new()
            .layer(TraceLayer::new_for_http())
            .layer(TimeoutLayer::new(std::time::Duration::from_secs(
                state.config.request_timeout_secs,
            )))
            .layer(if state.config.enable_cors {
                CorsLayer::permissive()
            } else {
                CorsLayer::new()
            }),
    );

    let addr = SocketAddr::from(([127, 0, 0, 1], state.config.port));
    info!("API server starting on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
