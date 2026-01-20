//! Command alias system for the OxiRS CLI
//!
//! Provides a flexible alias system that allows users to create shortcuts
//! for frequently used commands.
//!
//! ## Features
//! - Store aliases in user config directory
//! - Expand aliases before command parsing
//! - Support for command arguments in aliases
//! - Built-in alias management commands
//!
//! ## Example Usage
//! ```bash
//! # Add an alias
//! oxirs alias add q "query"
//! oxirs alias add qi "query --format json"
//!
//! # Use the alias
//! oxirs q mykg "SELECT * WHERE { ?s ?p ?o }"
//! oxirs qi mykg "SELECT * WHERE { ?s ?p ?o }"
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Alias configuration structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AliasConfig {
    /// Map of alias name to command expansion
    pub aliases: HashMap<String, String>,
}

/// Alias manager for loading, saving, and expanding aliases
pub struct AliasManager {
    config_path: PathBuf,
    config: AliasConfig,
}

impl AliasManager {
    /// Create a new alias manager
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config_path = Self::get_config_path(None)?;
        let config = Self::load_config(&config_path)?;

        Ok(Self {
            config_path,
            config,
        })
    }

    /// Create a new alias manager with custom config directory (for testing)
    #[cfg(test)]
    fn with_config_dir(config_dir: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let config_path = Self::get_config_path(Some(config_dir))?;
        let config = Self::load_config(&config_path)?;

        Ok(Self {
            config_path,
            config,
        })
    }

    /// Get the path to the alias configuration file
    fn get_config_path(custom_dir: Option<PathBuf>) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let config_dir = if let Some(dir) = custom_dir {
            dir
        } else {
            dirs::config_dir()
                .ok_or("Could not determine config directory")?
                .join("oxirs")
        };

        // Create config directory if it doesn't exist
        fs::create_dir_all(&config_dir)?;

        Ok(config_dir.join("aliases.toml"))
    }

    /// Load alias configuration from file
    fn load_config(path: &PathBuf) -> Result<AliasConfig, Box<dyn std::error::Error>> {
        if !path.exists() {
            // Create default config with some helpful aliases
            let default_config = AliasConfig {
                aliases: Self::default_aliases(),
            };

            let toml = toml::to_string_pretty(&default_config)?;
            fs::write(path, toml)?;

            Ok(default_config)
        } else {
            let content = fs::read_to_string(path)?;
            let config: AliasConfig = toml::from_str(&content)?;
            Ok(config)
        }
    }

    /// Get default aliases
    fn default_aliases() -> HashMap<String, String> {
        let mut aliases = HashMap::new();

        // Common shortcuts
        aliases.insert("q".to_string(), "query".to_string());
        aliases.insert("i".to_string(), "import".to_string());
        aliases.insert("e".to_string(), "export".to_string());
        aliases.insert("inter".to_string(), "interactive".to_string());
        aliases.insert("bench".to_string(), "benchmark".to_string());
        aliases.insert("perf".to_string(), "performance".to_string());

        // Query format shortcuts
        aliases.insert("qj".to_string(), "query --format json".to_string());
        aliases.insert("qc".to_string(), "query --format csv".to_string());
        aliases.insert("qt".to_string(), "query --format table".to_string());

        // Import/Export shortcuts
        aliases.insert("itt".to_string(), "import --format turtle".to_string());
        aliases.insert("int".to_string(), "import --format ntriples".to_string());
        aliases.insert("ijl".to_string(), "import --format jsonld".to_string());

        aliases
    }

    /// Save the current configuration to file
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let toml = toml::to_string_pretty(&self.config)?;
        fs::write(&self.config_path, toml)?;
        Ok(())
    }

    /// Add a new alias
    pub fn add_alias(
        &mut self,
        name: String,
        command: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Validate alias name (no spaces, valid identifier)
        if name.contains(' ') {
            return Err("Alias name cannot contain spaces".into());
        }

        if name.is_empty() {
            return Err("Alias name cannot be empty".into());
        }

        // Check if it conflicts with existing command names
        let reserved = vec![
            "init",
            "import",
            "export",
            "query",
            "update",
            "serve",
            "interactive",
            "benchmark",
            "performance",
            "migrate",
            "config",
            "aspect",
            "aas",
            "package",
            "explain",
            "templates",
            "cache",
            "history",
            "cicd",
            "alias",
            "help",
            "version",
        ];

        if reserved.contains(&name.as_str()) {
            return Err(format!(
                "Cannot create alias '{}': conflicts with existing command",
                name
            )
            .into());
        }

        self.config.aliases.insert(name, command);
        self.save()?;
        Ok(())
    }

    /// Remove an alias
    pub fn remove_alias(&mut self, name: &str) -> Result<bool, Box<dyn std::error::Error>> {
        let removed = self.config.aliases.remove(name).is_some();
        if removed {
            self.save()?;
        }
        Ok(removed)
    }

    /// Get an alias expansion
    pub fn get_alias(&self, name: &str) -> Option<&String> {
        self.config.aliases.get(name)
    }

    /// List all aliases
    pub fn list_aliases(&self) -> &HashMap<String, String> {
        &self.config.aliases
    }

    /// Expand aliases in command-line arguments
    pub fn expand_args(&self, args: Vec<String>) -> Vec<String> {
        if args.is_empty() {
            return args;
        }

        // First arg is the program name, skip it
        let mut result = vec![args[0].clone()];

        if args.len() < 2 {
            return args;
        }

        // Check if the first command arg is an alias
        let command = &args[1];

        if let Some(expansion) = self.get_alias(command) {
            // Expand the alias
            let expanded_parts: Vec<String> =
                shlex::split(expansion).unwrap_or_else(|| vec![expansion.clone()]);

            result.extend(expanded_parts);

            // Add remaining arguments
            result.extend(args[2..].iter().cloned());
        } else {
            // No alias, return original args
            return args;
        }

        result
    }

    /// Clear all aliases (reset to defaults)
    pub fn reset_to_defaults(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.config.aliases = Self::default_aliases();
        self.save()?;
        Ok(())
    }
}

impl Default for AliasManager {
    fn default() -> Self {
        Self::new().expect("Failed to create alias manager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_manager() -> (AliasManager, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let manager = AliasManager::with_config_dir(temp_dir.path().to_path_buf()).unwrap();
        (manager, temp_dir)
    }

    #[test]
    fn test_default_aliases() {
        let aliases = AliasManager::default_aliases();
        assert_eq!(aliases.get("q"), Some(&"query".to_string()));
        assert_eq!(aliases.get("i"), Some(&"import".to_string()));
        assert_eq!(aliases.get("e"), Some(&"export".to_string()));
    }

    #[test]
    fn test_expand_simple_alias() {
        let (mut manager, _temp_dir) = create_test_manager();
        manager
            .config
            .aliases
            .insert("q".to_string(), "query".to_string());

        let args = vec![
            "oxirs".to_string(),
            "q".to_string(),
            "mykg".to_string(),
            "SELECT * WHERE { ?s ?p ?o }".to_string(),
        ];

        let expanded = manager.expand_args(args);
        assert_eq!(expanded[1], "query");
        assert_eq!(expanded[2], "mykg");
    }

    #[test]
    fn test_expand_alias_with_flags() {
        let (mut manager, _temp_dir) = create_test_manager();
        manager
            .config
            .aliases
            .insert("qj".to_string(), "query --format json".to_string());

        let args = vec![
            "oxirs".to_string(),
            "qj".to_string(),
            "mykg".to_string(),
            "SELECT * WHERE { ?s ?p ?o }".to_string(),
        ];

        let expanded = manager.expand_args(args);
        assert_eq!(expanded[1], "query");
        assert_eq!(expanded[2], "--format");
        assert_eq!(expanded[3], "json");
        assert_eq!(expanded[4], "mykg");
    }

    #[test]
    fn test_no_expansion_when_not_alias() {
        let (manager, _temp_dir) = create_test_manager();

        let args = vec!["oxirs".to_string(), "query".to_string(), "mykg".to_string()];

        let expanded = manager.expand_args(args.clone());
        assert_eq!(expanded, args);
    }

    #[test]
    fn test_add_alias_validation() {
        let (mut manager, _temp_dir) = create_test_manager();

        // Should fail: spaces in name
        assert!(manager
            .add_alias("my alias".to_string(), "query".to_string())
            .is_err());

        // Should fail: empty name
        assert!(manager
            .add_alias("".to_string(), "query".to_string())
            .is_err());

        // Should fail: reserved command name
        assert!(manager
            .add_alias("query".to_string(), "something".to_string())
            .is_err());

        // Should succeed
        assert!(manager
            .add_alias("myq".to_string(), "query --format json".to_string())
            .is_ok());
    }

    #[test]
    fn test_remove_alias() {
        let (mut manager, _temp_dir) = create_test_manager();
        manager
            .config
            .aliases
            .insert("test".to_string(), "query".to_string());

        assert!(manager.remove_alias("test").unwrap());
        assert!(!manager.remove_alias("nonexistent").unwrap());
    }
}
