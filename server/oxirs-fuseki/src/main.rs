//! OxiRS Fuseki Server Binary

use clap::Parser;
use oxirs_fuseki::Server;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "oxirs-fuseki")]
#[command(about = "OxiRS SPARQL server with Fuseki compatibility")]
struct Args {
    /// Server port
    #[arg(short, long, default_value = "3030")]
    port: u16,

    /// Server host
    #[arg(long, default_value = "localhost")]
    host: String,

    /// Dataset configuration file
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Dataset storage path
    #[arg(short, long)]
    dataset: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Install the Pure Rust crypto provider as the process default (idempotent).
    let _ = rustls::crypto::CryptoProvider::install_default((*oxitls::pure_provider()).clone());

    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Use config file if provided, otherwise use CLI args. When a config file is
    // given, the FULL config (datasets, per-dataset `read_only`, security, …) is
    // threaded into the builder so it reaches `AppState`; previously only host/port
    // were extracted and the rest — including every `read_only` flag — was dropped.
    let mut builder = Server::builder();
    if let Some(config_path) = args.config {
        use oxirs_fuseki::config::ServerConfig;
        let config = ServerConfig::from_file(config_path)?;
        builder = builder
            .host(config.server.host.clone())
            .port(config.server.port)
            .config(config);
    } else {
        builder = builder.host(args.host).port(args.port);
    }

    if let Some(dataset_path) = args.dataset {
        builder = builder.dataset_path(dataset_path.to_string_lossy());
    }

    let server = builder.build().await?;
    server.run().await?;

    Ok(())
}
