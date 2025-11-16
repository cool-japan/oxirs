//! OxiRS CLI Binary

use clap::Parser;
use oxirs::cli::AliasManager;
use oxirs::{run, Cli};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get command-line arguments
    let args: Vec<String> = env::args().collect();

    // Expand aliases before parsing
    let expanded_args = match AliasManager::new() {
        Ok(manager) => manager.expand_args(args),
        Err(_) => {
            // If alias manager fails to initialize, just use original args
            args
        }
    };

    // Parse CLI with expanded arguments
    let cli = Cli::parse_from(expanded_args);

    run(cli).await
}
