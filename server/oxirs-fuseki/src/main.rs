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
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Use config file if provided, otherwise use CLI args
    let (host, port) = if let Some(config_path) = args.config {
        use oxirs_fuseki::config::ServerConfig;
        let config = ServerConfig::from_file(config_path)?;
        (config.server.host, config.server.port)
    } else {
        (args.host, args.port)
    };

    let mut builder = Server::builder().port(port).host(host);

    if let Some(dataset_path) = args.dataset {
        builder = builder.dataset_path(dataset_path.to_string_lossy());
    }

    let server = builder.build().await?;
    server.run().await?;

    Ok(())
}
