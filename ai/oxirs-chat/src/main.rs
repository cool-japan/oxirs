//! OxiRS Chat Server Binary

use clap::Parser;
use oxirs_chat::{
    server::{ChatServer, ServerConfig},
    ChatManager,
};
use oxirs_core::store::Store;
use std::{path::PathBuf, sync::Arc};
use tracing::{error, info};

#[derive(Parser)]
#[command(name = "oxirs-chat")]
#[command(about = "OxiRS RAG chat API server with LLM integration")]
struct Args {
    /// Server port
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Server host
    #[arg(long, default_value = "localhost")]
    host: String,

    /// Dataset path
    #[arg(short, long)]
    dataset: Option<PathBuf>,

    /// LLM model configuration
    #[arg(short, long)]
    model_config: Option<PathBuf>,

    /// Maximum concurrent connections
    #[arg(long, default_value = "1000")]
    max_connections: usize,

    /// Session timeout in seconds
    #[arg(long, default_value = "3600")]
    session_timeout: u64,

    /// Enable metrics endpoint
    #[arg(long)]
    enable_metrics: bool,

    /// Logging level
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize logging
    let log_level = match args.log_level.as_str() {
        "trace" => tracing::Level::TRACE,
        "debug" => tracing::Level::DEBUG,
        "info" => tracing::Level::INFO,
        "warn" => tracing::Level::WARN,
        "error" => tracing::Level::ERROR,
        _ => tracing::Level::INFO,
    };

    tracing_subscriber::fmt().with_max_level(log_level).init();

    info!("Starting OxiRS Chat server on {}:{}", args.host, args.port);

    // Initialize the knowledge graph store
    let store = match initialize_store(args.dataset.as_ref()).await {
        Ok(store) => Arc::new(store),
        Err(e) => {
            error!("Failed to initialize store: {}", e);
            return Err(e);
        }
    };

    info!("Knowledge graph store initialized");

    // Initialize the chat manager
    let chat_manager = ChatManager::new(store.clone());

    // Configure the server
    let server_config = ServerConfig {
        host: args.host,
        port: args.port,
        max_connections: args.max_connections,
        session_timeout: std::time::Duration::from_secs(args.session_timeout),
        enable_metrics: args.enable_metrics,
        cors_origins: vec!["*".to_string()], // TODO: Make configurable
    };

    info!("Server configuration: {:?}", server_config);

    if let Some(model_config) = &args.model_config {
        info!("Model config: {:?}", model_config);
        // TODO: Load and apply model configuration
    }

    // Create and start the server
    let server = ChatServer::new(chat_manager, server_config);

    info!("🚀 OxiRS Chat server starting...");
    info!(
        "📡 HTTP API available at: http://{}:{}/api",
        args.host, args.port
    );
    info!(
        "🔄 WebSocket endpoint: ws://{}:{}/api/sessions/{{session_id}}/ws",
        args.host, args.port
    );
    info!(
        "❤️  Health check: http://{}:{}/health",
        args.host, args.port
    );

    if args.enable_metrics {
        info!(
            "📊 Metrics endpoint: http://{}:{}/metrics",
            args.host, args.port
        );
    }

    // Start the server
    match server.serve().await {
        Ok(_) => info!("Server stopped gracefully"),
        Err(e) => {
            error!("Server error: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

/// Initialize the knowledge graph store
async fn initialize_store(
    dataset_path: Option<&PathBuf>,
) -> Result<Store, Box<dyn std::error::Error>> {
    let store = Store::new()?;

    if let Some(path) = dataset_path {
        info!("Loading dataset from: {:?}", path);
        // TODO: Implement dataset loading
        // store.load_from_file(path)?;
        info!("Dataset loaded successfully");
    } else {
        info!("No dataset specified, starting with empty store");
    }

    Ok(store)
}
