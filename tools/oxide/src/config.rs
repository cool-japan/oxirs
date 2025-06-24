//! CLI configuration management

/// CLI-specific configuration
pub struct CliConfig {
    pub default_dataset: Option<String>,
    pub default_format: String,
    pub server_defaults: ServerDefaults,
}

/// Default server settings
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