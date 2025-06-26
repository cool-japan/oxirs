//! CLI configuration management

pub mod manager;
pub mod secrets;

pub use manager::{ConfigManager, OxideConfig};
pub use secrets::{SecretManager, SecretBackend};

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
        CliConfig {
            default_dataset: None, // TODO: Get from datasets
            default_format: config.general.default_format.clone(),
            server_defaults: ServerDefaults {
                host: config.server.host.clone(),
                port: config.server.port,
                enable_graphql: false, // TODO: Get from server config
            },
        }
    }
}
