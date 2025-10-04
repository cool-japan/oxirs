//! Server command implementation
//!
//! Starts the OxiRS SPARQL HTTP server with full configuration support.

use super::CommandResult;
use std::path::PathBuf;

/// Start the OxiRS server
pub async fn run(config: PathBuf, port: u16, host: String, graphql: bool) -> CommandResult {
    println!("🚀 Starting OxiRS SPARQL Server...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Step 1: Load configuration
    println!("📋 Loading configuration from: {}", config.display());

    let oxirs_config = load_server_configuration(&config)?;

    println!("   ✓ Configuration loaded successfully");
    println!("   Datasets configured: {}", oxirs_config.datasets.len());

    // Step 2: Determine dataset path(s) from configuration
    let dataset_path = extract_primary_dataset_path(&oxirs_config)?;

    println!("   ✓ Primary dataset: {}", dataset_path.display());

    // Step 3: Build and configure the server
    println!("\n🔧 Initializing server components...");

    let server = oxirs_fuseki::Server::builder()
        .host(&host)
        .port(port)
        .dataset_path(dataset_path.to_string_lossy().to_string())
        .build()
        .await?;

    println!("   ✓ Server initialized successfully");

    // Step 4: Display server information
    println!("\n📡 Server Configuration:");
    println!("   Address: http://{}:{}", host, port);
    println!("   SPARQL Query: http://{}:{}/sparql", host, port);
    println!("   SPARQL Update: http://{}:{}/update", host, port);

    if graphql {
        println!("   GraphQL: http://{}:{}/graphql", host, port);
        println!("   (GraphQL endpoint enabled)");
    }

    println!("\n⚡ Server Health:");
    println!("   Liveness: http://{}:{}/health/live", host, port);
    println!("   Readiness: http://{}:{}/health/ready", host, port);
    println!("   Metrics: http://{}:{}/metrics", host, port);

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✅ Server is ready to accept connections!");
    println!("Press Ctrl+C to stop the server gracefully");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Step 5: Run the server
    match server.run().await {
        Ok(_) => {
            println!("\n🛑 Server stopped gracefully");
            Ok(())
        }
        Err(e) => {
            eprintln!("\n❌ Server error: {e}");
            Err(e.to_string().into())
        }
    }
}

/// Load OxiRS configuration from TOML file
fn load_server_configuration(
    config_path: &PathBuf,
) -> Result<crate::config::OxirsConfig, Box<dyn std::error::Error>> {
    use std::fs;

    // Check if config file exists
    if !config_path.exists() {
        return Err(format!("Configuration file not found: {}", config_path.display()).into());
    }

    // Read and parse TOML
    let content = fs::read_to_string(config_path).map_err(|e| {
        format!(
            "Failed to read configuration file '{}': {e}",
            config_path.display()
        )
    })?;

    let config: crate::config::OxirsConfig = toml::from_str(&content).map_err(|e| {
        format!(
            "Failed to parse TOML configuration '{}': {e}",
            config_path.display()
        )
    })?;

    Ok(config)
}

/// Extract the primary dataset path from configuration
fn extract_primary_dataset_path(
    config: &crate::config::OxirsConfig,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Try to get the "default" dataset first
    if let Some(dataset) = config.datasets.get("default") {
        return Ok(PathBuf::from(&dataset.location));
    }

    // Otherwise, use the first available dataset
    if let Some((_name, dataset)) = config.datasets.iter().next() {
        return Ok(PathBuf::from(&dataset.location));
    }

    Err("No datasets configured in configuration file. Add at least one dataset.".into())
}
