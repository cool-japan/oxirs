//! CLI configuration management
//!
//! This module provides:
//! - Configuration file parsing (TOML)
//! - Profile management (dev, staging, prod)
//! - Environment variable overrides
//! - Dataset configuration loading

pub mod manager;
pub mod secrets;

pub use manager::{
    AuthConfig, ConfigManager, CorsConfig, DatasetConfig, GeneralConfig, OxirsConfig, QueryConfig,
    RiotConfig, ServerConfig, TdbConfig, ToolsConfig, ValidationConfig,
};
pub use secrets::{SecretBackend, SecretManager};

/// Alias for backward compatibility
pub type Config = OxirsConfig;

/// CLI-specific configuration (legacy compatibility)
pub struct CliConfig {
    pub default_dataset: Option<String>,
    pub default_format: String,
    pub server_defaults: ServerDefaults,
}

/// Default server settings (legacy compatibility)
pub struct ServerDefaults {
    pub host: String,
    pub port: u16,
    pub enable_graphql: bool,
}

impl Default for CliConfig {
    fn default() -> Self {
        CliConfig {
            default_dataset: None,
            default_format: "turtle".to_string(),
            server_defaults: ServerDefaults {
                host: "localhost".to_string(),
                port: 3030,
                enable_graphql: false,
            },
        }
    }
}

/// Convert from new config to legacy format
impl From<&OxirsConfig> for CliConfig {
    fn from(config: &OxirsConfig) -> Self {
        // Get default dataset - use the first configured dataset or None
        let default_dataset = config.datasets.keys().next().cloned();

        CliConfig {
            default_dataset,
            default_format: config.general.default_format.clone(),
            server_defaults: ServerDefaults {
                host: config.server.host.clone(),
                port: config.server.port,
                enable_graphql: config.server.enable_graphql,
            },
        }
    }
}

// Dataset configuration helper functions

use crate::cli::error::{CliError, CliResult};
use std::path::{Path, PathBuf};

/// Load dataset configuration from a directory containing oxirs.toml
///
/// This function loads the dataset configuration from an oxirs.toml file
/// in the specified directory and returns the storage path for the dataset.
///
/// # Arguments
///
/// * `dataset_dir` - Directory containing oxirs.toml file
///
/// # Returns
///
/// PathBuf to the dataset storage location
///
/// # Errors
///
/// Returns error if:
/// - oxirs.toml file doesn't exist
/// - TOML parsing fails
/// - Dataset configuration is missing or invalid
pub fn load_dataset_from_config<P: AsRef<Path>>(dataset_dir: P) -> CliResult<PathBuf> {
    let dataset_dir = dataset_dir.as_ref();
    let config_path = dataset_dir.join("oxirs.toml");

    if !config_path.exists() {
        return Err(CliError::config_error(format!(
            "Configuration file '{}' not found",
            config_path.display()
        )));
    }

    // Read and parse the TOML file
    let content = std::fs::read_to_string(&config_path).map_err(|e| {
        CliError::config_error(format!(
            "Failed to read config file '{}': {e}",
            config_path.display()
        ))
    })?;

    let config: OxirsConfig = toml::from_str(&content).map_err(|e| {
        CliError::config_error(format!(
            "Failed to parse TOML in '{}': {e}",
            config_path.display()
        ))
    })?;

    // Extract the default dataset configuration
    // First try "default" dataset, then first available dataset
    let dataset_config = config
        .datasets
        .get("default")
        .or_else(|| config.datasets.values().next())
        .ok_or_else(|| {
            CliError::config_error(format!(
                "No dataset configuration found in '{}'",
                config_path.display()
            ))
        })?;

    // Parse the location path
    let storage_path = PathBuf::from(&dataset_config.location);

    // If the path is relative, make it relative to the dataset directory
    let storage_path = if storage_path.is_absolute() {
        storage_path
    } else {
        dataset_dir.join(storage_path)
    };

    Ok(storage_path)
}

/// Load a specific named dataset from configuration
///
/// # Arguments
///
/// * `dataset_dir` - Directory containing oxirs.toml file
/// * `dataset_name` - Name of the dataset to load
///
/// # Returns
///
/// Tuple of (storage_path, dataset_config)
pub fn load_named_dataset<P: AsRef<Path>>(
    dataset_dir: P,
    dataset_name: &str,
) -> CliResult<(PathBuf, DatasetConfig)> {
    let dataset_dir = dataset_dir.as_ref();
    let config_path = dataset_dir.join("oxirs.toml");

    if !config_path.exists() {
        return Err(CliError::config_error(format!(
            "Configuration file '{}' not found",
            config_path.display()
        )));
    }

    let content = std::fs::read_to_string(&config_path).map_err(|e| {
        CliError::config_error(format!(
            "Failed to read config file '{}': {e}",
            config_path.display()
        ))
    })?;

    let config: OxirsConfig = toml::from_str(&content).map_err(|e| {
        CliError::config_error(format!(
            "Failed to parse TOML in '{}': {e}",
            config_path.display()
        ))
    })?;

    let dataset_config = config
        .datasets
        .get(dataset_name)
        .ok_or_else(|| {
            CliError::config_error(format!(
                "Dataset '{dataset_name}' not found in configuration"
            ))
        })?
        .clone();

    let storage_path = PathBuf::from(&dataset_config.location);
    let storage_path = if storage_path.is_absolute() {
        storage_path
    } else {
        dataset_dir.join(storage_path)
    };

    Ok((storage_path, dataset_config))
}

#[cfg(test)]
mod config_tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_load_dataset_from_config() -> CliResult<()> {
        let temp_dir = std::env::temp_dir().join(format!("oxirs_test_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&temp_dir)?;

        // Create test oxirs.toml
        let config_content = r#"
[datasets.default]
name = "default"
location = "./data/default"
dataset_type = "tdb2"
read_only = false
"#;

        fs::write(temp_dir.join("oxirs.toml"), config_content)?;

        let storage_path = load_dataset_from_config(&temp_dir)?;
        assert!(storage_path.ends_with("data/default"));

        // Cleanup
        fs::remove_dir_all(&temp_dir).ok();

        Ok(())
    }

    #[test]
    fn test_load_named_dataset() -> CliResult<()> {
        let temp_dir = std::env::temp_dir().join(format!("oxirs_test_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&temp_dir)?;

        let config_content = r#"
[datasets.test_db]
name = "test_db"
location = "/absolute/path/to/db"
dataset_type = "tdb2"
read_only = true
"#;

        fs::write(temp_dir.join("oxirs.toml"), config_content)?;

        let (storage_path, config) = load_named_dataset(&temp_dir, "test_db")?;
        assert_eq!(storage_path, PathBuf::from("/absolute/path/to/db"));
        assert_eq!(config.dataset_type, "tdb2");
        assert!(config.read_only);

        fs::remove_dir_all(&temp_dir).ok();

        Ok(())
    }

    #[test]
    fn test_missing_config_file() {
        let temp_dir = std::env::temp_dir().join(format!("oxirs_test_{}", uuid::Uuid::new_v4()));
        let result = load_dataset_from_config(&temp_dir);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Configuration file"));
    }

    #[test]
    fn test_invalid_toml() -> std::io::Result<()> {
        let temp_dir = std::env::temp_dir().join(format!("oxirs_test_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&temp_dir)?;

        fs::write(temp_dir.join("oxirs.toml"), "invalid toml [[")?;

        let result = load_dataset_from_config(&temp_dir);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("parse TOML"));

        fs::remove_dir_all(&temp_dir).ok();
        Ok(())
    }

    #[test]
    fn test_missing_dataset() -> std::io::Result<()> {
        let temp_dir = std::env::temp_dir().join(format!("oxirs_test_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&temp_dir)?;

        fs::write(
            temp_dir.join("oxirs.toml"),
            "[server]\nhost = \"localhost\"",
        )?;

        let result = load_dataset_from_config(&temp_dir);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No dataset configuration"));

        fs::remove_dir_all(&temp_dir).ok();
        Ok(())
    }
}
