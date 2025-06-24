//! OxiRS Chat Server Binary

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "oxirs-chat")]
#[command(about = "OxiRS RAG chat API server")]
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
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    
    let args = Args::parse();
    
    println!("Starting OxiRS Chat server on {}:{}", args.host, args.port);
    
    if let Some(dataset) = &args.dataset {
        println!("Dataset: {:?}", dataset);
    }
    
    if let Some(model_config) = &args.model_config {
        println!("Model config: {:?}", model_config);
    }
    
    // TODO: Implement chat server startup
    println!("Chat server would start here...");
    
    Ok(())
}