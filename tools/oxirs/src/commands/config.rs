//! Configuration management command

use super::CommandResult;
use crate::config::{validate_config_strict, OxirsConfig};
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

            // Ensure output directory exists
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }

            // Generate default configuration using ConfigManager
            let config = OxirsConfig::default();
            let toml_content = toml::to_string_pretty(&config)
                .map_err(|e| format!("Failed to serialize configuration: {e}"))?;

            // Add header comments
            let header = r#"# OxiRS Configuration File
# Documentation: https://github.com/cool-japan/oxirs
#
# This configuration supports:
# - Multiple datasets with independent settings
# - Profile-based configuration (dev, staging, prod)
# - Environment variable substitution
# - Server, logging, and security settings
#
# To use profiles, create config.dev.toml, config.staging.toml, etc.
# Use --profile flag to select: oxirs --profile dev serve ...

"#;
            let full_content = format!("{}{}", header, toml_content);

            fs::write(&output, full_content)?;

            println!("✓ Configuration generated successfully");
            println!();
            println!("Next steps:");
            println!("  1. Edit {} to customize settings", output.display());
            println!("  2. Validate: oxirs config validate {}", output.display());
            println!("  3. View: oxirs config show {}", output.display());
            println!();
            println!("For profile-based configuration:");
            println!("  - Create config.dev.toml, config.prod.toml, etc.");
            println!("  - Use --profile flag: oxirs --profile dev serve ...");
        }
        ConfigAction::Validate { config } => {
            println!("Validating configuration at {}", config.display());
            println!();

            if !config.exists() {
                return Err(format!("Configuration file '{}' not found", config.display()).into());
            }

            // Read and parse TOML
            let content = fs::read_to_string(&config)?;
            let parsed_config: OxirsConfig =
                toml::from_str(&content).map_err(|e| format!("Invalid TOML syntax: {e}"))?;

            // Perform comprehensive validation
            let config_dir = config.parent();
            match validate_config_strict(&parsed_config, config_dir) {
                Ok(_) => {
                    println!("✓ Configuration is valid");
                    println!();
                    println!("Configuration summary:");
                    println!("  Datasets: {}", parsed_config.datasets.len());
                    println!(
                        "  Server: {}:{}",
                        parsed_config.server.host, parsed_config.server.port
                    );
                    println!("  Default format: {}", parsed_config.general.default_format);
                    println!("  Log level: {}", parsed_config.general.log_level);

                    if parsed_config.server.auth.enabled {
                        println!(
                            "  Authentication: enabled ({:?})",
                            parsed_config.server.auth.method
                        );
                    }

                    if parsed_config.server.cors.enabled {
                        println!("  CORS: enabled");
                    }

                    if parsed_config.server.enable_graphql {
                        println!(
                            "  GraphQL: enabled at {}",
                            parsed_config.server.graphql_path
                        );
                    }

                    // List datasets
                    if !parsed_config.datasets.is_empty() {
                        println!();
                        println!("Configured datasets:");
                        for (name, dataset) in &parsed_config.datasets {
                            println!(
                                "  • {} ({}): {}",
                                name, dataset.dataset_type, dataset.location
                            );
                        }
                    }
                }
                Err(e) => {
                    println!("✗ Configuration validation failed");
                    println!();
                    return Err(e);
                }
            }
        }
        ConfigAction::Show { config } => {
            if let Some(config_path) = config {
                println!("Configuration from: {}", config_path.display());
                println!();

                if !config_path.exists() {
                    return Err(format!(
                        "Configuration file '{}' not found",
                        config_path.display()
                    )
                    .into());
                }

                // Parse and validate before showing
                let content = fs::read_to_string(&config_path)?;
                let parsed_config: OxirsConfig = toml::from_str(&content)?;

                // Validate (non-strict)
                if let Err(e) = crate::config::validate_config(&parsed_config, config_path.parent())
                {
                    println!("⚠️  Warning: Configuration has validation errors");
                    println!("{}", e);
                    println!();
                }

                println!("{}", "=".repeat(60));
                println!("{content}");
                println!("{}", "=".repeat(60));
            } else {
                // Show default configuration
                println!("Default Configuration Template:");
                println!();
                let default_config = OxirsConfig::default();
                let toml_content = toml::to_string_pretty(&default_config)
                    .map_err(|e| format!("Failed to serialize configuration: {e}"))?;
                println!("{}", "=".repeat(60));
                println!("{toml_content}");
                println!("{}", "=".repeat(60));
                println!();
                println!("To generate a configuration file:");
                println!("  oxirs config init oxirs.toml");
            }
        }
    }
    Ok(())
}
