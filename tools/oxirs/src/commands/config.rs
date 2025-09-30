//! Configuration management command

use super::CommandResult;
use crate::ConfigAction;
use std::fs;

/// Run configuration management command
pub async fn run(action: ConfigAction) -> CommandResult {
    match action {
        ConfigAction::Init { output } => {
            println!("Generating default configuration at {}", output.display());

            if output.exists() {
                return Err(
                    format!("Configuration file '{}' already exists", output.display()).into(),
                );
            }

            let default_config = create_default_server_config()?;
            fs::write(&output, default_config)?;

            println!("Default configuration generated successfully");
            println!("Edit the file to customize your OxiRS deployment");
        }
        ConfigAction::Validate { config } => {
            println!("Validating configuration at {}", config.display());

            if !config.exists() {
                return Err(format!("Configuration file '{}' not found", config.display()).into());
            }

            // Read and parse TOML
            let content = fs::read_to_string(&config)?;
            let _parsed: toml::Value = toml::from_str(&content)?;

            println!("Configuration is valid ✓");
        }
        ConfigAction::Show { config } => {
            if let Some(config_path) = config {
                println!("Showing configuration from: {}", config_path.display());

                if !config_path.exists() {
                    return Err(format!(
                        "Configuration file '{}' not found",
                        config_path.display()
                    )
                    .into());
                }

                let content = fs::read_to_string(config_path)?;
                println!("---");
                println!("{content}");
                println!("---");
            } else {
                println!("Showing default configuration:");
                println!("---");
                println!("{}", create_default_server_config()?);
                println!("---");
            }
        }
    }
    Ok(())
}

/// Create default server configuration TOML
fn create_default_server_config() -> Result<String, Box<dyn std::error::Error>> {
    let config = r#"# OxiRS Server Configuration
# Documentation: https://github.com/cool-japan/oxirs

[server]
# Server binding configuration
host = "localhost"
port = 3030
cors_enabled = true

# Maximum request size (in bytes)
max_request_size = "16MB"

# Request timeout (in seconds)
request_timeout = 30

[datasets]
# Default dataset configuration
# Multiple datasets can be defined here

[datasets.default]
name = "default"
format = "tdb2"
location = "./data/default"
description = "Default dataset"

# Query endpoint configuration
[datasets.default.endpoints]
query = "/default/sparql"
update = "/default/update"
graph_store = "/default/data"

# Optional GraphQL endpoint
graphql = "/default/graphql"
graphiql = "/default/graphiql"

[logging]
# Logging configuration
level = "info"
format = "json"

# Log file (optional, logs to stdout if not specified)
# file = "./logs/oxirs.log"

[security]
# Security configuration (optional)
auth_enabled = false

# CORS configuration
[security.cors]
allow_origins = ["*"]
allow_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
allow_headers = ["*"]

[features]
# Optional features
reasoning = false
validation = false
text_search = false
vector_search = false

# Performance tuning
[performance]
query_timeout = 300  # seconds
max_concurrent_queries = 100
cache_size = "1GB"

# Monitoring and metrics
[monitoring]
metrics_enabled = true
metrics_endpoint = "/metrics"
health_check_endpoint = "/health"
"#;

    Ok(config.to_string())
}
