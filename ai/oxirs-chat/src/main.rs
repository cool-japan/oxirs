//! OxiRS Chat Server Binary

use clap::Parser;
use oxirs_chat::{
    server::{ChatServer, ServerConfig},
    ChatManager,
};
use oxirs_core::{parser::RdfFormat, GraphName, Literal, NamedNode, Quad, Store, Triple};
use std::{path::PathBuf, sync::Arc};
use tracing::{error, info, warn};

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

    /// Session persistence path
    #[arg(long)]
    persistence_path: Option<PathBuf>,
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
    let chat_manager = match &args.persistence_path {
        Some(path) => {
            info!("Enabling session persistence at: {:?}", path);
            match ChatManager::with_persistence(store.clone(), path).await {
                Ok(manager) => Arc::new(manager),
                Err(e) => {
                    error!("Failed to initialize chat manager with persistence: {}", e);
                    return Err(format!(
                        "Failed to initialize chat manager with persistence: {}",
                        e
                    )
                    .into());
                }
            }
        }
        None => {
            info!("Session persistence disabled");
            match ChatManager::new(store.clone()).await {
                Ok(manager) => Arc::new(manager),
                Err(e) => {
                    error!("Failed to initialize chat manager: {}", e);
                    return Err(format!("Failed to initialize chat manager: {}", e).into());
                }
            }
        }
    };

    // Store host and port for later use
    let host = args.host.clone();
    let port = args.port;

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

    // Clone chat_manager before moving it
    let chat_manager_clone = chat_manager.clone();

    // Create and start the server
    let server = ChatServer::new(chat_manager, server_config);

    info!("🚀 OxiRS Chat server starting...");
    info!("📡 HTTP API available at: http://{}:{}/api", host, port);
    info!(
        "🔄 WebSocket endpoint: ws://{}:{}/api/sessions/{{session_id}}/ws",
        host, port
    );
    info!("❤️  Health check: http://{}:{}/health", host, port);

    if args.enable_metrics {
        info!("📊 Metrics endpoint: http://{}:{}/metrics", host, port);
    }

    if args.persistence_path.is_some() {
        info!("💾 Session persistence enabled");
    }

    // Set up graceful shutdown
    let chat_manager_for_shutdown = chat_manager_clone.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for Ctrl+C");
        info!("Received shutdown signal, saving sessions...");
        if let Err(e) = chat_manager_for_shutdown.save_all_sessions().await {
            error!("Failed to save sessions on shutdown: {}", e);
        }
        info!("Sessions saved, shutting down...");
        std::process::exit(0);
    });

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
    let mut store = Store::new()?;

    if let Some(path) = dataset_path {
        info!("Loading dataset from: {:?}", path);

        // Determine file format from extension
        let format = if let Some(extension) = path.extension().and_then(|s| s.to_str()) {
            match extension.to_lowercase().as_str() {
                "nt" | "ntriples" => RdfFormat::NTriples,
                "ttl" | "turtle" => RdfFormat::Turtle,
                "rdf" | "xml" => RdfFormat::RdfXml,
                "n3" => RdfFormat::Turtle, // N3 not supported, use Turtle
                "jsonld" | "json-ld" => RdfFormat::JsonLd,
                _ => {
                    warn!(
                        "Unknown file extension '{}', defaulting to Turtle",
                        extension
                    );
                    RdfFormat::Turtle
                }
            }
        } else {
            warn!("No file extension found, defaulting to Turtle");
            RdfFormat::Turtle
        };

        // Load the dataset
        match std::fs::read_to_string(path) {
            Ok(_content) => {
                info!("File read successfully, format: {:?}", format);

                // TODO: Implement parsing from string with format detection
                // For now, we'll use the sample data below
                warn!("Dataset parsing from file not yet implemented, using sample data");
            }
            Err(e) => {
                error!("Failed to read dataset file: {}", e);
                return Err(format!("Failed to read dataset file: {}", e).into());
            }
        }
    } else {
        info!("No dataset specified, starting with empty store");

        // Add some sample triples for demonstration
        info!("Adding sample triples for demonstration...");
        add_sample_data(&mut store)?;
    }

    Ok(store)
}

/// Add sample RDF data for demonstration when no dataset is provided
fn add_sample_data(store: &mut Store) -> Result<(), Box<dyn std::error::Error>> {
    let sample_triples = vec![
        // Person data
        Triple::new(
            NamedNode::new("http://example.org/person/alice")?,
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?,
            NamedNode::new("http://xmlns.com/foaf/0.1/Person")?,
        ),
        Triple::new(
            NamedNode::new("http://example.org/person/alice")?,
            NamedNode::new("http://xmlns.com/foaf/0.1/name")?,
            Literal::new_simple_literal("Alice Smith"),
        ),
        Triple::new(
            NamedNode::new("http://example.org/person/alice")?,
            NamedNode::new("http://example.org/age")?,
            Literal::new_typed_literal(
                "30",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?,
            ),
        ),
        // Organization data
        Triple::new(
            NamedNode::new("http://example.org/org/acme")?,
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?,
            NamedNode::new("http://xmlns.com/foaf/0.1/Organization")?,
        ),
        Triple::new(
            NamedNode::new("http://example.org/org/acme")?,
            NamedNode::new("http://xmlns.com/foaf/0.1/name")?,
            Literal::new_simple_literal("ACME Corporation"),
        ),
        // Relationship data
        Triple::new(
            NamedNode::new("http://example.org/person/alice")?,
            NamedNode::new("http://example.org/worksFor")?,
            NamedNode::new("http://example.org/org/acme")?,
        ),
    ];

    let mut triples_added = 0;
    for triple in sample_triples {
        let quad = Quad::new(
            triple.subject().clone(),
            triple.predicate().clone(),
            triple.object().clone(),
            GraphName::DefaultGraph,
        );

        if let Err(e) = store.insert_quad(quad) {
            warn!("Failed to insert sample triple: {}", e);
        } else {
            triples_added += 1;
        }
    }

    info!("Added {} sample triples", triples_added);
    Ok(())
}
