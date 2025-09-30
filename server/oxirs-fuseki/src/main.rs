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

    let mut builder = Server::builder().port(args.port).host(args.host);

    if let Some(dataset_path) = args.dataset {
        builder = builder.dataset_path(dataset_path.to_string_lossy());
    }

    let server = builder.build().await?;
    server.run().await?;

    Ok(())
}
