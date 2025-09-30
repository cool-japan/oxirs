//! OxiRS CLI Binary

use clap::Parser;
use oxirs::{run, Cli};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    run(cli).await
}
