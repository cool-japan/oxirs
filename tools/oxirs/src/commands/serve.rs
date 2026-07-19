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

    // Otherwise, deterministically pick the alphabetically-first dataset
    // name. `config.datasets` is a `HashMap`, whose iteration order is
    // randomized per-process, so picking `.iter().next()` directly would
    // make server startup select a different dataset across runs whenever
    // more than one dataset is configured without an explicit "default".
    if let Some(name) = config.datasets.keys().min() {
        // Safe: `name` was just obtained from `config.datasets.keys()`.
        if let Some(dataset) = config.datasets.get(name) {
            return Ok(PathBuf::from(&dataset.location));
        }
    }

    Err("No datasets configured in configuration file. Add at least one dataset.".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DatasetConfig, OxirsConfig};
    use std::collections::HashMap;

    fn dataset_config(location: &str) -> DatasetConfig {
        let toml_str = format!(
            r#"
            dataset_type = "memory"
            location = "{location}"
            "#
        );
        toml::from_str(&toml_str).expect("valid minimal DatasetConfig TOML")
    }

    #[test]
    fn extract_primary_dataset_path_prefers_explicit_default() {
        let mut datasets = HashMap::new();
        datasets.insert("zeta".to_string(), dataset_config("/data/zeta"));
        datasets.insert("default".to_string(), dataset_config("/data/default"));
        datasets.insert("alpha".to_string(), dataset_config("/data/alpha"));

        let config = OxirsConfig {
            datasets,
            ..Default::default()
        };

        let path = extract_primary_dataset_path(&config).expect("dataset should be found");
        assert_eq!(path, PathBuf::from("/data/default"));
    }

    #[test]
    fn extract_primary_dataset_path_is_deterministic_without_default() {
        let mut datasets = HashMap::new();
        datasets.insert("zeta".to_string(), dataset_config("/data/zeta"));
        datasets.insert("alpha".to_string(), dataset_config("/data/alpha"));
        datasets.insert("mid".to_string(), dataset_config("/data/mid"));

        let config = OxirsConfig {
            datasets,
            ..Default::default()
        };

        // Regardless of HashMap iteration order, the alphabetically-first
        // key ("alpha") must always be selected.
        for _ in 0..20 {
            let path = extract_primary_dataset_path(&config).expect("dataset should be found");
            assert_eq!(path, PathBuf::from("/data/alpha"));
        }
    }

    #[test]
    fn extract_primary_dataset_path_errors_when_empty() {
        let config = OxirsConfig::default();
        assert!(extract_primary_dataset_path(&config).is_err());
    }
}
