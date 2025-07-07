//! OxiRS Chat Server Binary

use clap::Parser;
use oxirs_chat::{
    server::{ChatServer, ServerConfig},
    ChatConfig, OxiRSChat,
};
use oxirs_core::{format::RdfFormat, ConcreteStore, GraphName, Literal, NamedNode, Quad, Triple};
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

    /// CORS allowed origins (comma-separated list). Use "*" for any origin.
    #[arg(long, default_value = "*")]
    cors_origins: String,
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

    // Load and prepare model configuration
    let llm_config = if let Some(model_config_path) = &args.model_config {
        info!("Loading model configuration from: {:?}", model_config_path);
        match load_llm_config(model_config_path).await {
            Ok(config) => {
                info!("Successfully loaded model configuration");
                Some(config)
            }
            Err(e) => {
                error!("Failed to load model configuration: {}", e);
                warn!("Using default model configuration");
                None
            }
        }
    } else {
        info!("No model configuration specified, using defaults");
        None
    };

    // Initialize OxiRS Chat instance with LLM configuration
    let chat_instance = {
        info!("Initializing OxiRS Chat with advanced AI capabilities");
        let chat_config = ChatConfig::default();
        match OxiRSChat::new_with_llm_config(chat_config, store.clone(), llm_config).await {
            Ok(chat) => Arc::new(chat),
            Err(e) => {
                error!("Failed to initialize OxiRS Chat: {}", e);
                return Err(format!("Failed to initialize OxiRS Chat: {}", e).into());
            }
        }
    };

    // Store host and port for later use
    let host = args.host.clone();
    let port = args.port;

    // Parse CORS origins from command line argument
    let cors_origins: Vec<String> = args
        .cors_origins
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    // Configure the server
    let server_config = ServerConfig {
        host: args.host,
        port: args.port,
        max_connections: args.max_connections,
        session_timeout: std::time::Duration::from_secs(args.session_timeout),
        enable_metrics: args.enable_metrics,
        cors_origins,
    };

    info!("Server configuration: {:?}", server_config);

    // Load existing sessions if persistence path is provided
    if let Some(ref persistence_path) = args.persistence_path {
        info!("Loading existing sessions from {:?}", persistence_path);
        match chat_instance.load_sessions(persistence_path).await {
            Ok(count) => {
                info!("Loaded {} existing sessions", count);
            }
            Err(e) => {
                warn!("Failed to load existing sessions: {}", e);
            }
        }
    }

    // Clone chat_instance before moving it
    let chat_instance_clone = chat_instance.clone();

    // Create and start the server
    let server = ChatServer::new(chat_instance, server_config);

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

    // Set up graceful shutdown with session saving
    let chat_instance_for_shutdown = chat_instance_clone.clone();
    let persistence_path_for_shutdown = args.persistence_path.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for Ctrl+C");
        info!("Received shutdown signal, saving sessions...");

        if let Some(persistence_path) = persistence_path_for_shutdown {
            match chat_instance_for_shutdown
                .save_sessions(&persistence_path)
                .await
            {
                Ok(count) => {
                    info!(
                        "Successfully saved {} sessions to {:?}",
                        count, persistence_path
                    );
                }
                Err(e) => {
                    error!("Failed to save sessions: {}", e);
                }
            }
        } else {
            info!("No persistence path configured, sessions will not be saved");
        }

        info!("Graceful shutdown complete");
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
) -> Result<ConcreteStore, Box<dyn std::error::Error>> {
    let mut store = ConcreteStore::new()?;

    if let Some(path) = dataset_path {
        info!("Loading dataset from: {:?}", path);

        // Determine file format from extension
        let format = if let Some(extension) = path.extension().and_then(|s| s.to_str()) {
            match extension.to_lowercase().as_str() {
                "nt" | "ntriples" => RdfFormat::NTriples,
                "ttl" | "turtle" => RdfFormat::Turtle,
                "rdf" | "xml" => RdfFormat::RdfXml,
                "n3" => RdfFormat::Turtle, // N3 not supported, use Turtle
                "jsonld" | "json-ld" => {
                    // Use default JSON-LD profile
                    use oxirs_core::format::JsonLdProfileSet;
                    RdfFormat::JsonLd {
                        profile: JsonLdProfileSet::empty(),
                    }
                }
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
            Ok(content) => {
                info!("File read successfully, format: {:?}", format);

                // Parse the RDF content using oxirs-core
                info!("Parsing RDF data from file...");
                match parse_rdf_content(&content, format, &mut store) {
                    Ok(count) => {
                        info!(
                            "Successfully parsed and loaded {} triples from dataset",
                            count
                        );
                    }
                    Err(e) => {
                        error!("Failed to parse RDF data: {}", e);
                        warn!("Adding sample data instead due to parsing error");
                        add_sample_data(&mut store)?;
                    }
                }
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

/// Load LLM configuration from file
async fn load_llm_config(
    config_path: &PathBuf,
) -> Result<oxirs_chat::llm::LLMConfig, Box<dyn std::error::Error>> {
    // Read the configuration file
    let config_content = std::fs::read_to_string(config_path)?;

    // Determine file format from extension
    let config = if let Some(extension) = config_path.extension().and_then(|s| s.to_str()) {
        match extension.to_lowercase().as_str() {
            "toml" => toml::from_str(&config_content)?,
            "json" => serde_json::from_str(&config_content)?,
            "yaml" | "yml" => serde_yaml::from_str(&config_content)?,
            _ => {
                warn!(
                    "Unknown config file extension '{}', trying TOML format",
                    extension
                );
                toml::from_str(&config_content)?
            }
        }
    } else {
        // Default to TOML if no extension
        toml::from_str(&config_content)?
    };

    Ok(config)
}

/// Parse RDF content from string using specified format
fn parse_rdf_content(
    content: &str,
    format: RdfFormat,
    store: &mut ConcreteStore,
) -> Result<usize, Box<dyn std::error::Error>> {
    use oxirs_core::format::parser::simple;
    // std::io::Cursor removed - unused import

    let mut count = 0;

    match format {
        RdfFormat::Turtle => {
            let triples = simple::parse_turtle(content)?;
            for triple in triples {
                let quad = Quad::new(
                    triple.subject().clone(),
                    triple.predicate().clone(),
                    triple.object().clone(),
                    GraphName::DefaultGraph,
                );
                store.insert_quad(quad)?;
                count += 1;
            }
        }
        RdfFormat::NTriples => {
            let triples = simple::parse_ntriples(content)?;
            for triple in triples {
                let quad = Quad::new(
                    triple.subject().clone(),
                    triple.predicate().clone(),
                    triple.object().clone(),
                    GraphName::DefaultGraph,
                );
                store.insert_quad(quad)?;
                count += 1;
            }
        }
        RdfFormat::RdfXml => {
            // Use generic parser for RDF/XML
            use oxirs_core::format::RdfParser;
            let parser = RdfParser::new(oxirs_core::format::RdfFormat::RdfXml);
            let quads: Result<Vec<_>, _> = parser.for_slice(content.as_bytes()).collect();

            match quads {
                Ok(quads) => {
                    for quad in quads {
                        store.insert_quad(quad)?;
                        count += 1;
                    }
                }
                Err(e) => {
                    return Err(format!("Failed to parse RDF/XML: {}", e).into());
                }
            }
        }
        RdfFormat::JsonLd { .. } => {
            // Use generic parser for JSON-LD
            use oxirs_core::format::RdfParser;
            let parser = RdfParser::new(format);
            let quads: Result<Vec<_>, _> = parser.for_slice(content.as_bytes()).collect();

            match quads {
                Ok(quads) => {
                    for quad in quads {
                        store.insert_quad(quad)?;
                        count += 1;
                    }
                }
                Err(e) => {
                    return Err(format!("Failed to parse JSON-LD: {}", e).into());
                }
            }
        }
        _ => {
            // For other formats, use the generic parser
            use oxirs_core::format::RdfParser;
            let format_debug = format!("{:?}", format); // Capture debug representation before move
            let parser = RdfParser::new(format);
            let quads: Result<Vec<_>, _> = parser.for_slice(content.as_bytes()).collect();

            match quads {
                Ok(quads) => {
                    for quad in quads {
                        store.insert_quad(quad)?;
                        count += 1;
                    }
                }
                Err(e) => {
                    return Err(
                        format!("Failed to parse RDF format {}: {}", format_debug, e).into(),
                    );
                }
            }
        }
    }

    Ok(count)
}

/// Add sample RDF data for demonstration when no dataset is provided
fn add_sample_data(store: &mut ConcreteStore) -> Result<(), Box<dyn std::error::Error>> {
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
