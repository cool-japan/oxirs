//! Oxide CLI Binary

use clap::Parser;
use oxide::{Cli, run};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    run(cli).await
}