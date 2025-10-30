//! Configuration validation module
//!
//! Provides comprehensive validation for OxiRS configuration files including:
//! - Schema validation
//! - Required field checking
//! - Path existence verification
//! - Dataset configuration validation
//! - Security configuration validation

use crate::cli::error::{CliError, CliResult};
use crate::config::{DatasetConfig, OxirsConfig};
use std::path::{Path, PathBuf};

/// Validation error types
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// Missing required field
    MissingField { field: String, section: String },
    /// Invalid field value
    InvalidValue {
        field: String,
        value: String,
        reason: String,
    },
    /// Path does not exist
    PathNotFound { path: PathBuf, field: String },
    /// Invalid format or type
    InvalidFormat { field: String, expected: String },
    /// Security validation failure
    SecurityError { reason: String },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::MissingField { field, section } => {
                write!(
                    f,
                    "Missing required field '{field}' in section '[{section}]'"
                )
            }
            ValidationError::InvalidValue {
                field,
                value,
                reason,
            } => {
                write!(f, "Invalid value '{value}' for field '{field}': {reason}")
            }
            ValidationError::PathNotFound { path, field } => {
                write!(
                    f,
                    "Path '{}' specified in field '{field}' does not exist",
                    path.display()
                )
            }
            ValidationError::InvalidFormat { field, expected } => {
                write!(f, "Invalid format for field '{field}': expected {expected}")
            }
            ValidationError::SecurityError { reason } => {
                write!(f, "Security configuration error: {reason}")
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Configuration validator
pub struct ConfigValidator {
    errors: Vec<ValidationError>,
    warnings: Vec<String>,
    strict_mode: bool,
}

impl ConfigValidator {
    /// Create a new validator
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
            strict_mode: false,
        }
    }

    /// Enable strict validation mode
    pub fn with_strict_mode(mut self) -> Self {
        self.strict_mode = true;
        self
    }

    /// Validate a complete configuration
    pub fn validate(&mut self, config: &OxirsConfig, config_dir: Option<&Path>) -> CliResult<()> {
        // Validate general settings
        self.validate_general(&config.general);

        // Validate server settings
        self.validate_server(&config.server);

        // Validate datasets
        for (name, dataset) in &config.datasets {
            self.validate_dataset(name, dataset, config_dir);
        }

        // Validate tools configuration
        self.validate_tools(&config.tools);

        // Check if there are any errors
        if !self.errors.is_empty() {
            let error_messages: Vec<String> = self.errors.iter().map(|e| e.to_string()).collect();
            return Err(CliError::config_error(format!(
                "Configuration validation failed:\n  - {}",
                error_messages.join("\n  - ")
            )));
        }

        Ok(())
    }

    /// Validate general configuration
    fn validate_general(&mut self, general: &crate::config::GeneralConfig) {
        // Validate default format
        let valid_formats = [
            "turtle", "ntriples", "nquads", "trig", "rdfxml", "jsonld", "n3",
        ];
        if !valid_formats.contains(&general.default_format.as_str()) {
            self.errors.push(ValidationError::InvalidValue {
                field: "general.default_format".to_string(),
                value: general.default_format.clone(),
                reason: format!("Must be one of: {}", valid_formats.join(", ")),
            });
        }

        // Validate output directory if specified
        if let Some(ref output_dir) = general.output_dir {
            if self.strict_mode && !output_dir.exists() {
                self.warnings.push(format!(
                    "Output directory '{}' does not exist and will be created if needed",
                    output_dir.display()
                ));
            }
        }

        // Validate timeout
        if general.timeout == 0 {
            self.errors.push(ValidationError::InvalidValue {
                field: "general.timeout".to_string(),
                value: general.timeout.to_string(),
                reason: "Timeout must be greater than 0".to_string(),
            });
        }

        if general.timeout > 3600 {
            self.warnings.push(format!(
                "Timeout value {} seconds is very high (>1 hour)",
                general.timeout
            ));
        }

        // Validate log level
        let valid_log_levels = ["error", "warn", "info", "debug", "trace"];
        if !valid_log_levels.contains(&general.log_level.to_lowercase().as_str()) {
            self.errors.push(ValidationError::InvalidValue {
                field: "general.log_level".to_string(),
                value: general.log_level.clone(),
                reason: format!("Must be one of: {}", valid_log_levels.join(", ")),
            });
        }
    }

    /// Validate server configuration
    fn validate_server(&mut self, server: &crate::config::ServerConfig) {
        // Validate host
        if server.host.trim().is_empty() {
            self.errors.push(ValidationError::InvalidValue {
                field: "server.host".to_string(),
                value: server.host.clone(),
                reason: "Host cannot be empty".to_string(),
            });
        }

        // Validate port
        if server.port == 0 {
            self.errors.push(ValidationError::InvalidValue {
                field: "server.port".to_string(),
                value: server.port.to_string(),
                reason: "Port must be greater than 0".to_string(),
            });
        }

        if server.port < 1024 && !cfg!(windows) {
            self.warnings.push(format!(
                "Port {} is a privileged port (requires root/admin on Unix systems)",
                server.port
            ));
        }

        // Validate GraphQL path
        if !server.graphql_path.starts_with('/') {
            self.errors.push(ValidationError::InvalidValue {
                field: "server.graphql_path".to_string(),
                value: server.graphql_path.clone(),
                reason: "Path must start with '/'".to_string(),
            });
        }

        // Validate authentication configuration
        if server.auth.enabled {
            if let Some(ref method) = server.auth.method {
                let valid_methods = ["basic", "bearer", "jwt", "oauth2"];
                if !valid_methods.contains(&method.to_lowercase().as_str()) {
                    self.errors.push(ValidationError::InvalidValue {
                        field: "server.auth.method".to_string(),
                        value: method.clone(),
                        reason: format!("Must be one of: {}", valid_methods.join(", ")),
                    });
                }
            } else {
                self.errors.push(ValidationError::MissingField {
                    field: "method".to_string(),
                    section: "server.auth".to_string(),
                });
            }
        }

        // Validate CORS configuration
        if server.cors.enabled && server.cors.allowed_origins.is_empty() {
            self.warnings
                .push("CORS is enabled but no allowed_origins specified".to_string());
        }
    }

    /// Validate dataset configuration
    fn validate_dataset(&mut self, name: &str, dataset: &DatasetConfig, config_dir: Option<&Path>) {
        // Validate dataset type
        let valid_types = ["tdb2", "memory", "remote"];
        if !valid_types.contains(&dataset.dataset_type.as_str()) {
            self.errors.push(ValidationError::InvalidValue {
                field: format!("datasets.{name}.dataset_type"),
                value: dataset.dataset_type.clone(),
                reason: format!("Must be one of: {}", valid_types.join(", ")),
            });
        }

        // Validate location
        if dataset.location.trim().is_empty() {
            self.errors.push(ValidationError::MissingField {
                field: "location".to_string(),
                section: format!("datasets.{name}"),
            });
            return;
        }

        // For local datasets, check if path exists or can be created
        if dataset.dataset_type == "tdb2" || dataset.dataset_type == "memory" {
            let location_path = PathBuf::from(&dataset.location);

            // Resolve relative paths
            let absolute_path = if location_path.is_absolute() {
                location_path
            } else if let Some(config_dir) = config_dir {
                config_dir.join(location_path)
            } else {
                location_path
            };

            if self.strict_mode && !absolute_path.exists() {
                if dataset.read_only {
                    self.errors.push(ValidationError::PathNotFound {
                        path: absolute_path,
                        field: format!("datasets.{name}.location"),
                    });
                } else {
                    self.warnings.push(format!(
                        "Dataset location '{}' does not exist and will be created",
                        absolute_path.display()
                    ));
                }
            }
        }

        // For remote datasets, validate URL
        if dataset.dataset_type == "remote"
            && !dataset.location.starts_with("http://")
            && !dataset.location.starts_with("https://")
        {
            self.errors.push(ValidationError::InvalidValue {
                field: format!("datasets.{name}.location"),
                value: dataset.location.clone(),
                reason: "Remote dataset location must be a valid HTTP(S) URL".to_string(),
            });
        }
    }

    /// Validate tools configuration
    fn validate_tools(&mut self, tools: &crate::config::ToolsConfig) {
        // Validate query configuration
        if let Some(timeout) = tools.query.timeout {
            if timeout == 0 {
                self.errors.push(ValidationError::InvalidValue {
                    field: "tools.query.timeout".to_string(),
                    value: timeout.to_string(),
                    reason: "Query timeout must be greater than 0".to_string(),
                });
            }
        }

        if let Some(limit) = tools.query.result_limit {
            if limit == 0 {
                self.warnings.push(
                    "Query result limit is set to 0 (no results will be returned)".to_string(),
                );
            }
        }

        // Validate TDB configuration
        if let Some(cache_size) = tools.tdb.cache_size {
            if cache_size == 0 {
                self.errors.push(ValidationError::InvalidValue {
                    field: "tools.tdb.cache_size".to_string(),
                    value: cache_size.to_string(),
                    reason: "Cache size must be greater than 0".to_string(),
                });
            }
        }

        // Validate validation configuration
        if let Some(max_errors) = tools.validation.max_errors {
            if max_errors == 0 {
                self.warnings.push(
                    "Validation max_errors is set to 0 (validation will stop immediately)"
                        .to_string(),
                );
            }
        }
    }

    /// Get all validation errors
    pub fn errors(&self) -> &[ValidationError] {
        &self.errors
    }

    /// Get all validation warnings
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

impl Default for ConfigValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Quick validation function for convenience
pub fn validate_config(config: &OxirsConfig, config_dir: Option<&Path>) -> CliResult<()> {
    let mut validator = ConfigValidator::new();
    validator.validate(config, config_dir)?;

    // Print warnings if any
    if !validator.warnings().is_empty() {
        eprintln!("Configuration warnings:");
        for warning in validator.warnings() {
            eprintln!("  ⚠️  {warning}");
        }
    }

    Ok(())
}

/// Validate configuration in strict mode
pub fn validate_config_strict(config: &OxirsConfig, config_dir: Option<&Path>) -> CliResult<()> {
    let mut validator = ConfigValidator::new().with_strict_mode();
    validator.validate(config, config_dir)?;

    // Print warnings if any
    if !validator.warnings().is_empty() {
        eprintln!("Configuration warnings:");
        for warning in validator.warnings() {
            eprintln!("  ⚠️  {warning}");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OxirsConfig;

    #[test]
    fn test_valid_config() {
        let config = OxirsConfig::default();
        let mut validator = ConfigValidator::new();
        assert!(validator.validate(&config, None).is_ok());
        assert!(validator.is_valid());
    }

    #[test]
    fn test_invalid_format() {
        let mut config = OxirsConfig::default();
        config.general.default_format = "invalid".to_string();

        let mut validator = ConfigValidator::new();
        assert!(validator.validate(&config, None).is_err());
        assert!(!validator.is_valid());
        assert_eq!(validator.errors().len(), 1);
    }

    #[test]
    fn test_invalid_port() {
        let mut config = OxirsConfig::default();
        config.server.port = 0;

        let mut validator = ConfigValidator::new();
        assert!(validator.validate(&config, None).is_err());
        assert!(!validator.is_valid());
    }

    #[test]
    fn test_missing_dataset_location() {
        let mut config = OxirsConfig::default();
        let dataset = DatasetConfig {
            dataset_type: "tdb2".to_string(),
            location: "".to_string(),
            read_only: false,
            options: Default::default(),
        };
        config.datasets.insert("test".to_string(), dataset);

        let mut validator = ConfigValidator::new();
        assert!(validator.validate(&config, None).is_err());
    }

    #[test]
    fn test_warnings() {
        let mut config = OxirsConfig::default();
        config.server.port = 80; // Privileged port

        let mut validator = ConfigValidator::new();
        validator.validate(&config, None).ok();

        #[cfg(not(windows))]
        {
            assert!(!validator.warnings().is_empty());
        }
    }

    #[test]
    fn test_invalid_log_level() {
        let mut config = OxirsConfig::default();
        config.general.log_level = "invalid".to_string();

        let mut validator = ConfigValidator::new();
        assert!(validator.validate(&config, None).is_err());
    }

    #[test]
    fn test_zero_timeout() {
        let mut config = OxirsConfig::default();
        config.general.timeout = 0;

        let mut validator = ConfigValidator::new();
        assert!(validator.validate(&config, None).is_err());
    }
}
