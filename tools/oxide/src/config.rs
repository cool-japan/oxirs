//! CLI configuration management

pub mod manager;
pub mod secrets;

pub use manager::{ConfigManager, OxideConfig};
pub use secrets::{SecretBackend, SecretManager};

/// Alias for backward compatibility
pub type Config = OxideConfig;

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
impl From<&OxideConfig> for CliConfig {
    fn from(config: &OxideConfig) -> Self {
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
