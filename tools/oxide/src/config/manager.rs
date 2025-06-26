//! Configuration management with profiles and environment detection
//!
//! Provides a hierarchical configuration system with profiles, environment
//! variables, and automatic configuration discovery.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;
use toml;

use crate::cli::error::{CliError, CliResult};

/// Configuration manager with profile support
pub struct ConfigManager {
    /// Base configuration directory
    config_dir: PathBuf,
    /// Current profile
    active_profile: String,
    /// Loaded configurations by profile
    configs: HashMap<String, OxideConfig>,
}

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OxideConfig {
    /// General settings
    #[serde(default)]
    pub general: GeneralConfig,
    
    /// Server settings
    #[serde(default)]
    pub server: ServerConfig,
    
    /// Dataset configurations
    #[serde(default)]
    pub datasets: HashMap<String, DatasetConfig>,
    
    /// Tool-specific settings
    #[serde(default)]
    pub tools: ToolsConfig,
    
    /// Environment-specific overrides
    #[serde(default)]
    pub env: HashMap<String, toml::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Default RDF format
    #[serde(default = "default_format")]
    pub default_format: String,
    
    /// Default output directory
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
    
    /// Enable progress bars
    #[serde(default = "default_true")]
    pub show_progress: bool,
    
    /// Enable colored output
    #[serde(default = "default_true")]
    pub colored_output: bool,
    
    /// Default timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    
    /// Log level
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Default host
    #[serde(default = "default_host")]
    pub host: String,
    
    /// Default port
    #[serde(default = "default_port")]
    pub port: u16,
    
    /// Enable admin interface
    #[serde(default)]
    pub admin_enabled: bool,
    
    /// CORS settings
    #[serde(default)]
    pub cors: CorsConfig,
    
    /// Authentication settings
    #[serde(default)]
    pub auth: AuthConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CorsConfig {
    pub enabled: bool,
    pub allowed_origins: Vec<String>,
    pub allowed_methods: Vec<String>,
    pub allowed_headers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuthConfig {
    pub enabled: bool,
    pub method: Option<String>, // basic, jwt, oauth
    pub config: HashMap<String, toml::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Dataset type (tdb2, memory, remote)
    pub dataset_type: String,
    
    /// Location (path or URL)
    pub location: String,
    
    /// Read-only mode
    #[serde(default)]
    pub read_only: bool,
    
    /// Dataset-specific options
    #[serde(default)]
    pub options: HashMap<String, toml::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolsConfig {
    /// RDF I/O settings
    #[serde(default)]
    pub riot: RiotConfig,
    
    /// Query settings
    #[serde(default)]
    pub query: QueryConfig,
    
    /// TDB settings
    #[serde(default)]
    pub tdb: TdbConfig,
    
    /// Validation settings
    #[serde(default)]
    pub validation: ValidationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RiotConfig {
    pub strict_mode: bool,
    pub base_uri: Option<String>,
    pub pretty_print: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryConfig {
    pub timeout: Option<u64>,
    pub optimize: bool,
    pub explain: bool,
    pub result_limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TdbConfig {
    pub cache_size: Option<usize>,
    pub file_mode: Option<String>,
    pub sync_mode: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidationConfig {
    pub abort_on_error: bool,
    pub max_errors: Option<usize>,
    pub report_format: Option<String>,
}

impl ConfigManager {
    /// Create a new configuration manager
    pub fn new() -> CliResult<Self> {
        let config_dir = Self::get_config_dir()?;
        
        Ok(Self {
            config_dir,
            active_profile: "default".to_string(),
            configs: HashMap::new(),
        })
    }

    /// Get the configuration directory
    fn get_config_dir() -> CliResult<PathBuf> {
        // Check environment variable first
        if let Ok(dir) = std::env::var("OXIDE_CONFIG_DIR") {
            return Ok(PathBuf::from(dir));
        }

        // Use platform-specific config directory
        dirs::config_dir()
            .map(|p| p.join("oxide"))
            .ok_or_else(|| CliError::config_error("Cannot determine config directory"))
    }

    /// Load configuration for a profile
    pub fn load_profile(&mut self, profile: &str) -> CliResult<&OxideConfig> {
        if self.configs.contains_key(profile) {
            return Ok(&self.configs[profile]);
        }

        let config = self.load_config_cascade(profile)?;
        self.configs.insert(profile.to_string(), config);
        self.active_profile = profile.to_string();
        
        Ok(&self.configs[profile])
    }

    /// Load configuration with cascade (defaults -> profile -> env -> cli)
    fn load_config_cascade(&self, profile: &str) -> CliResult<OxideConfig> {
        // Start with defaults
        let mut config = OxideConfig::default();

        // Load global config if exists
        let global_path = self.config_dir.join("config.toml");
        if global_path.exists() {
            let global_config = self.load_config_file(&global_path)?;
            config = self.merge_configs(config, global_config);
        }

        // Load profile-specific config if exists
        if profile != "default" {
            let profile_path = self.config_dir.join(format!("config.{}.toml", profile));
            if profile_path.exists() {
                let profile_config = self.load_config_file(&profile_path)?;
                config = self.merge_configs(config, profile_config);
            }
        }

        // Apply environment variable overrides
        config = self.apply_env_overrides(config)?;

        Ok(config)
    }

    /// Load a configuration file
    fn load_config_file(&self, path: &Path) -> CliResult<OxideConfig> {
        let content = fs::read_to_string(path)
            .map_err(|e| CliError::config_error(format!("Cannot read config file: {}", e)))?;
        
        toml::from_str(&content)
            .map_err(|e| CliError::config_error(format!("Invalid TOML in config file: {}", e)))
    }

    /// Merge two configurations (right overwrites left)
    fn merge_configs(&self, mut base: OxideConfig, overlay: OxideConfig) -> OxideConfig {
        // Merge general settings
        if overlay.general.default_format != default_format() {
            base.general.default_format = overlay.general.default_format;
        }
        if overlay.general.output_dir.is_some() {
            base.general.output_dir = overlay.general.output_dir;
        }
        
        // Merge server settings
        if overlay.server.host != default_host() {
            base.server.host = overlay.server.host;
        }
        if overlay.server.port != default_port() {
            base.server.port = overlay.server.port;
        }
        
        // Merge datasets
        base.datasets.extend(overlay.datasets);
        
        // Deep merge tools config
        base.tools = overlay.tools;
        
        base
    }

    /// Apply environment variable overrides
    fn apply_env_overrides(&self, mut config: OxideConfig) -> CliResult<OxideConfig> {
        // OXIDE_DEFAULT_FORMAT
        if let Ok(format) = std::env::var("OXIDE_DEFAULT_FORMAT") {
            config.general.default_format = format;
        }

        // OXIDE_OUTPUT_DIR
        if let Ok(dir) = std::env::var("OXIDE_OUTPUT_DIR") {
            config.general.output_dir = Some(PathBuf::from(dir));
        }

        // OXIDE_NO_COLOR
        if std::env::var("OXIDE_NO_COLOR").is_ok() || std::env::var("NO_COLOR").is_ok() {
            config.general.colored_output = false;
        }

        // OXIDE_SERVER_HOST
        if let Ok(host) = std::env::var("OXIDE_SERVER_HOST") {
            config.server.host = host;
        }

        // OXIDE_SERVER_PORT
        if let Ok(port_str) = std::env::var("OXIDE_SERVER_PORT") {
            if let Ok(port) = port_str.parse::<u16>() {
                config.server.port = port;
            }
        }

        Ok(config)
    }

    /// Get active configuration
    pub fn get_config(&self) -> CliResult<&OxideConfig> {
        self.configs.get(&self.active_profile)
            .ok_or_else(|| CliError::config_error("No configuration loaded"))
    }

    /// Save configuration to file
    pub fn save_config(&self, config: &OxideConfig, profile: Option<&str>) -> CliResult<()> {
        let profile = profile.unwrap_or(&self.active_profile);
        
        // Ensure config directory exists
        fs::create_dir_all(&self.config_dir)
            .map_err(|e| CliError::config_error(format!("Cannot create config directory: {}", e)))?;

        let path = if profile == "default" {
            self.config_dir.join("config.toml")
        } else {
            self.config_dir.join(format!("config.{}.toml", profile))
        };

        let content = toml::to_string_pretty(config)
            .map_err(|e| CliError::config_error(format!("Cannot serialize config: {}", e)))?;

        fs::write(&path, content)
            .map_err(|e| CliError::config_error(format!("Cannot write config file: {}", e)))?;

        Ok(())
    }

    /// List available profiles
    pub fn list_profiles(&self) -> CliResult<Vec<String>> {
        let mut profiles = vec!["default".to_string()];

        if self.config_dir.exists() {
            for entry in fs::read_dir(&self.config_dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("config.") && name.ends_with(".toml") {
                        let profile = name
                            .strip_prefix("config.")
                            .and_then(|n| n.strip_suffix(".toml"))
                            .unwrap_or("");
                        
                        if !profile.is_empty() {
                            profiles.push(profile.to_string());
                        }
                    }
                }
            }
        }

        profiles.sort();
        profiles.dedup();
        Ok(profiles)
    }

    /// Generate a default configuration file
    pub fn generate_default_config(&self) -> CliResult<()> {
        let config = OxideConfig::default();
        self.save_config(&config, Some("default"))
    }
}

impl Default for OxideConfig {
    fn default() -> Self {
        Self {
            general: GeneralConfig::default(),
            server: ServerConfig::default(),
            datasets: HashMap::new(),
            tools: ToolsConfig::default(),
            env: HashMap::new(),
        }
    }
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            default_format: default_format(),
            output_dir: None,
            show_progress: default_true(),
            colored_output: default_true(),
            timeout: default_timeout(),
            log_level: default_log_level(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            admin_enabled: false,
            cors: CorsConfig::default(),
            auth: AuthConfig::default(),
        }
    }
}

// Default value functions for serde
fn default_format() -> String { "turtle".to_string() }
fn default_true() -> bool { true }
fn default_timeout() -> u64 { 30 }
fn default_log_level() -> String { "info".to_string() }
fn default_host() -> String { "localhost".to_string() }
fn default_port() -> u16 { 3030 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OxideConfig::default();
        assert_eq!(config.general.default_format, "turtle");
        assert_eq!(config.server.host, "localhost");
        assert_eq!(config.server.port, 3030);
    }

    #[test]
    fn test_config_serialization() {
        let config = OxideConfig::default();
        let toml_str = toml::to_string(&config).unwrap();
        assert!(toml_str.contains("[general]"));
        assert!(toml_str.contains("[server]"));
    }

    #[test]
    fn test_env_override() {
        std::env::set_var("OXIDE_DEFAULT_FORMAT", "ntriples");
        std::env::set_var("OXIDE_SERVER_PORT", "8080");

        let manager = ConfigManager::new().unwrap();
        let config = manager.apply_env_overrides(OxideConfig::default()).unwrap();

        assert_eq!(config.general.default_format, "ntriples");
        assert_eq!(config.server.port, 8080);

        // Clean up
        std::env::remove_var("OXIDE_DEFAULT_FORMAT");
        std::env::remove_var("OXIDE_SERVER_PORT");
    }
}