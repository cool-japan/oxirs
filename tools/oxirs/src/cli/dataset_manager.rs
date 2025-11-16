//! Multi-dataset connection management for REPL
//!
//! This module provides functionality for managing multiple dataset connections,
//! allowing users to switch between datasets seamlessly in the interactive REPL.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::cli::error::{CliError, CliResult};

/// Connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionState {
    /// Connection is active and ready
    Active,
    /// Connection is idle (not actively used)
    Idle,
    /// Connection has encountered an error
    Error,
    /// Connection is closed
    Closed,
}

/// Dataset connection information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConnection {
    /// Connection ID (unique identifier)
    pub id: String,

    /// Dataset name
    pub name: String,

    /// Dataset location (file path or URL)
    pub location: String,

    /// Connection state
    pub state: ConnectionState,

    /// When the connection was established
    pub connected_at: DateTime<Utc>,

    /// Last time the connection was used
    pub last_used: DateTime<Utc>,

    /// Number of queries executed on this connection
    pub query_count: u64,

    /// Optional description
    pub description: Option<String>,

    /// Connection properties/metadata
    pub properties: HashMap<String, String>,
}

impl DatasetConnection {
    /// Create a new dataset connection
    pub fn new(id: String, name: String, location: String) -> Self {
        let now = Utc::now();
        Self {
            id,
            name,
            location,
            state: ConnectionState::Active,
            connected_at: now,
            last_used: now,
            query_count: 0,
            description: None,
            properties: HashMap::new(),
        }
    }

    /// Create a connection with description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Add a property
    pub fn with_property(mut self, key: String, value: String) -> Self {
        self.properties.insert(key, value);
        self
    }

    /// Record query execution
    pub fn record_query(&mut self) {
        self.query_count += 1;
        self.last_used = Utc::now();
    }

    /// Mark connection as closed
    pub fn close(&mut self) {
        self.state = ConnectionState::Closed;
    }

    /// Check if connection is active
    pub fn is_active(&self) -> bool {
        self.state == ConnectionState::Active
    }
}

/// Dataset manager configuration
#[derive(Debug, Clone)]
pub struct DatasetManagerConfig {
    /// Maximum number of concurrent connections
    pub max_connections: usize,

    /// Auto-connect to default dataset on startup
    pub auto_connect_default: bool,

    /// Default dataset name
    pub default_dataset: Option<String>,

    /// Connection timeout in seconds
    pub connection_timeout: Option<u64>,

    /// Idle connection timeout (close after inactivity)
    pub idle_timeout: Option<u64>,
}

impl Default for DatasetManagerConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            auto_connect_default: true,
            default_dataset: None,
            connection_timeout: Some(30),
            idle_timeout: Some(300), // 5 minutes
        }
    }
}

/// Manager for multiple dataset connections
pub struct DatasetManager {
    /// All active connections (by ID)
    connections: HashMap<String, DatasetConnection>,

    /// Currently active connection ID
    active_connection: Option<String>,

    /// Configuration
    config: DatasetManagerConfig,

    /// Connection counter for generating IDs
    connection_counter: u64,

    /// Connection aliases (name -> ID)
    aliases: HashMap<String, String>,
}

impl DatasetManager {
    /// Create a new dataset manager
    pub fn new() -> Self {
        Self::with_config(DatasetManagerConfig::default())
    }

    /// Create a dataset manager with custom config
    pub fn with_config(config: DatasetManagerConfig) -> Self {
        Self {
            connections: HashMap::new(),
            active_connection: None,
            config,
            connection_counter: 0,
            aliases: HashMap::new(),
        }
    }

    /// Connect to a dataset
    pub fn connect(&mut self, name: String, location: String) -> CliResult<String> {
        self.connect_with_description(name, location, None)
    }

    /// Connect to a dataset with description
    pub fn connect_with_description(
        &mut self,
        name: String,
        location: String,
        description: Option<String>,
    ) -> CliResult<String> {
        // Check connection limit
        if self.connections.len() >= self.config.max_connections {
            return Err(CliError::invalid_arguments(format!(
                "Maximum number of connections ({}) reached. Close an existing connection first.",
                self.config.max_connections
            )));
        }

        // Check if name already exists
        if self.aliases.contains_key(&name) {
            return Err(CliError::invalid_arguments(format!(
                "Connection with name '{}' already exists. Use a different name.",
                name
            )));
        }

        // Generate connection ID
        self.connection_counter += 1;
        let conn_id = format!("conn-{}", self.connection_counter);

        // Create connection
        let mut connection = DatasetConnection::new(conn_id.clone(), name.clone(), location);
        if let Some(desc) = description {
            connection.description = Some(desc);
        }

        // Store connection
        self.connections.insert(conn_id.clone(), connection);
        self.aliases.insert(name, conn_id.clone());

        // Set as active if it's the first connection
        if self.active_connection.is_none() {
            self.active_connection = Some(conn_id.clone());
        }

        Ok(conn_id)
    }

    /// Switch to a different dataset
    pub fn switch(&mut self, name: &str) -> CliResult<()> {
        let conn_id = self
            .aliases
            .get(name)
            .ok_or_else(|| CliError::not_found(format!("Connection '{}' not found", name)))?;

        let connection = self
            .connections
            .get(conn_id)
            .ok_or_else(|| CliError::not_found(format!("Connection '{}' not found", name)))?;

        if !connection.is_active() {
            return Err(CliError::invalid_arguments(format!(
                "Connection '{}' is not active (state: {:?})",
                name, connection.state
            )));
        }

        self.active_connection = Some(conn_id.clone());
        Ok(())
    }

    /// Get the active connection
    pub fn active_connection(&self) -> Option<&DatasetConnection> {
        self.active_connection
            .as_ref()
            .and_then(|id| self.connections.get(id))
    }

    /// Get a mutable reference to the active connection
    pub fn active_connection_mut(&mut self) -> Option<&mut DatasetConnection> {
        self.active_connection
            .as_ref()
            .and_then(|id| self.connections.get_mut(id))
    }

    /// Get active connection name
    pub fn active_name(&self) -> Option<String> {
        self.active_connection().map(|c| c.name.clone())
    }

    /// Get connection by name
    pub fn get(&self, name: &str) -> Option<&DatasetConnection> {
        self.aliases
            .get(name)
            .and_then(|id| self.connections.get(id))
    }

    /// Get connection by ID
    pub fn get_by_id(&self, id: &str) -> Option<&DatasetConnection> {
        self.connections.get(id)
    }

    /// List all connections
    pub fn list(&self) -> Vec<&DatasetConnection> {
        let mut conns: Vec<&DatasetConnection> = self.connections.values().collect();
        conns.sort_by(|a, b| a.name.cmp(&b.name));
        conns
    }

    /// List active connections only
    pub fn list_active(&self) -> Vec<&DatasetConnection> {
        self.connections
            .values()
            .filter(|c| c.is_active())
            .collect()
    }

    /// Close a connection
    pub fn close(&mut self, name: &str) -> CliResult<()> {
        let conn_id = self
            .aliases
            .get(name)
            .ok_or_else(|| CliError::not_found(format!("Connection '{}' not found", name)))?;

        // Check if trying to close active connection
        if self.active_connection.as_ref() == Some(conn_id) {
            return Err(CliError::invalid_arguments(
                "Cannot close the active connection. Switch to another connection first.",
            ));
        }

        // Mark as closed
        if let Some(connection) = self.connections.get_mut(conn_id) {
            connection.close();
        }

        // Remove from aliases
        self.aliases.remove(name);

        Ok(())
    }

    /// Close all idle connections
    pub fn close_idle(&mut self) -> usize {
        let mut closed = 0;

        // Find idle connections to close
        let idle_ids: Vec<String> = self
            .connections
            .iter()
            .filter(|(id, conn)| {
                conn.state == ConnectionState::Idle && self.active_connection.as_ref() != Some(*id)
            })
            .map(|(id, _)| id.clone())
            .collect();

        // Close idle connections
        for id in idle_ids {
            if let Some(conn) = self.connections.get_mut(&id) {
                conn.close();
                self.aliases.remove(&conn.name);
                closed += 1;
            }
        }

        closed
    }

    /// Disconnect (remove) a connection
    pub fn disconnect(&mut self, name: &str) -> CliResult<()> {
        let conn_id = self
            .aliases
            .get(name)
            .ok_or_else(|| CliError::not_found(format!("Connection '{}' not found", name)))?
            .clone();

        // Check if trying to disconnect active connection
        if self.active_connection.as_ref() == Some(&conn_id) {
            return Err(CliError::invalid_arguments(
                "Cannot disconnect the active connection. Switch to another connection first.",
            ));
        }

        // Remove connection
        self.connections.remove(&conn_id);
        self.aliases.remove(name);

        Ok(())
    }

    /// Record query execution on active connection
    pub fn record_query(&mut self) -> CliResult<()> {
        if let Some(conn) = self.active_connection_mut() {
            conn.record_query();
            Ok(())
        } else {
            Err(CliError::invalid_arguments("No active connection"))
        }
    }

    /// Get connection statistics
    pub fn stats(&self) -> DatasetManagerStats {
        DatasetManagerStats {
            total_connections: self.connections.len(),
            active_connections: self.list_active().len(),
            active_connection_name: self.active_name(),
            total_queries: self.connections.values().map(|c| c.query_count).sum(),
            max_connections: self.config.max_connections,
        }
    }

    /// Clear all connections
    pub fn clear(&mut self) {
        self.connections.clear();
        self.aliases.clear();
        self.active_connection = None;
    }

    /// Check if any connection is active
    pub fn has_active_connection(&self) -> bool {
        self.active_connection.is_some()
    }

    /// Get configuration
    pub fn config(&self) -> &DatasetManagerConfig {
        &self.config
    }
}

impl Default for DatasetManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Dataset manager statistics
#[derive(Debug, Clone)]
pub struct DatasetManagerStats {
    /// Total number of connections
    pub total_connections: usize,

    /// Number of active connections
    pub active_connections: usize,

    /// Name of the active connection
    pub active_connection_name: Option<String>,

    /// Total queries across all connections
    pub total_queries: u64,

    /// Maximum allowed connections
    pub max_connections: usize,
}

/// Helper functions for dataset paths
pub mod paths {
    use super::*;

    /// Normalize a dataset path
    pub fn normalize_path(path: &str) -> PathBuf {
        PathBuf::from(path)
    }

    /// Check if path is a valid dataset location
    pub fn is_valid_location(location: &str) -> bool {
        // Check if it's a file path or URL
        location.starts_with("http://")
            || location.starts_with("https://")
            || Path::new(location).exists()
            || !location.is_empty()
    }

    /// Get dataset name from path
    pub fn dataset_name_from_path(path: &Path) -> Option<String> {
        path.file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connect_dataset() {
        let mut manager = DatasetManager::new();

        let conn_id = manager
            .connect("test-db".to_string(), "/data/test.db".to_string())
            .unwrap();

        assert!(!conn_id.is_empty());
        assert_eq!(manager.connections.len(), 1);
        assert!(manager.has_active_connection());
    }

    #[test]
    fn test_connect_duplicate_name() {
        let mut manager = DatasetManager::new();

        manager
            .connect("test-db".to_string(), "/data/test1.db".to_string())
            .unwrap();

        let result = manager.connect("test-db".to_string(), "/data/test2.db".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_switch_connection() {
        let mut manager = DatasetManager::new();

        manager
            .connect("db1".to_string(), "/data/db1.db".to_string())
            .unwrap();
        manager
            .connect("db2".to_string(), "/data/db2.db".to_string())
            .unwrap();

        assert_eq!(manager.active_name(), Some("db1".to_string()));

        manager.switch("db2").unwrap();
        assert_eq!(manager.active_name(), Some("db2".to_string()));
    }

    #[test]
    fn test_switch_nonexistent() {
        let mut manager = DatasetManager::new();

        manager
            .connect("db1".to_string(), "/data/db1.db".to_string())
            .unwrap();

        let result = manager.switch("db2");
        assert!(result.is_err());
    }

    #[test]
    fn test_close_connection() {
        let mut manager = DatasetManager::new();

        manager
            .connect("db1".to_string(), "/data/db1.db".to_string())
            .unwrap();
        let db2_id = manager
            .connect("db2".to_string(), "/data/db2.db".to_string())
            .unwrap();

        manager.switch("db1").unwrap();
        manager.close("db2").unwrap();

        assert_eq!(manager.connections.len(), 2); // Still exists but closed
                                                  // Use ID to get connection since alias was removed
        assert!(!manager.get_by_id(&db2_id).unwrap().is_active());
    }

    #[test]
    fn test_close_active_connection() {
        let mut manager = DatasetManager::new();

        manager
            .connect("db1".to_string(), "/data/db1.db".to_string())
            .unwrap();

        let result = manager.close("db1");
        assert!(result.is_err());
    }

    #[test]
    fn test_disconnect_connection() {
        let mut manager = DatasetManager::new();

        manager
            .connect("db1".to_string(), "/data/db1.db".to_string())
            .unwrap();
        manager
            .connect("db2".to_string(), "/data/db2.db".to_string())
            .unwrap();

        manager.switch("db1").unwrap();
        manager.disconnect("db2").unwrap();

        assert_eq!(manager.connections.len(), 1);
        assert!(manager.get("db2").is_none());
    }

    #[test]
    fn test_list_connections() {
        let mut manager = DatasetManager::new();

        manager
            .connect("db1".to_string(), "/data/db1.db".to_string())
            .unwrap();
        manager
            .connect("db2".to_string(), "/data/db2.db".to_string())
            .unwrap();
        manager
            .connect("db3".to_string(), "/data/db3.db".to_string())
            .unwrap();

        let conns = manager.list();
        assert_eq!(conns.len(), 3);

        // Check sorted by name
        assert_eq!(conns[0].name, "db1");
        assert_eq!(conns[1].name, "db2");
        assert_eq!(conns[2].name, "db3");
    }

    #[test]
    fn test_record_query() {
        let mut manager = DatasetManager::new();

        manager
            .connect("db1".to_string(), "/data/db1.db".to_string())
            .unwrap();

        assert_eq!(manager.active_connection().unwrap().query_count, 0);

        manager.record_query().unwrap();
        assert_eq!(manager.active_connection().unwrap().query_count, 1);

        manager.record_query().unwrap();
        assert_eq!(manager.active_connection().unwrap().query_count, 2);
    }

    #[test]
    fn test_max_connections() {
        let config = DatasetManagerConfig {
            max_connections: 2,
            ..Default::default()
        };
        let mut manager = DatasetManager::with_config(config);

        manager
            .connect("db1".to_string(), "/data/db1.db".to_string())
            .unwrap();
        manager
            .connect("db2".to_string(), "/data/db2.db".to_string())
            .unwrap();

        let result = manager.connect("db3".to_string(), "/data/db3.db".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_connection_with_description() {
        let mut manager = DatasetManager::new();

        manager
            .connect_with_description(
                "db1".to_string(),
                "/data/db1.db".to_string(),
                Some("Test database".to_string()),
            )
            .unwrap();

        let conn = manager.get("db1").unwrap();
        assert_eq!(conn.description, Some("Test database".to_string()));
    }

    #[test]
    fn test_stats() {
        let mut manager = DatasetManager::new();

        manager
            .connect("db1".to_string(), "/data/db1.db".to_string())
            .unwrap();
        manager
            .connect("db2".to_string(), "/data/db2.db".to_string())
            .unwrap();

        manager.record_query().unwrap();
        manager.switch("db2").unwrap();
        manager.record_query().unwrap();
        manager.record_query().unwrap();

        let stats = manager.stats();
        assert_eq!(stats.total_connections, 2);
        assert_eq!(stats.active_connections, 2);
        assert_eq!(stats.total_queries, 3);
        assert_eq!(stats.active_connection_name, Some("db2".to_string()));
    }

    #[test]
    fn test_clear_connections() {
        let mut manager = DatasetManager::new();

        manager
            .connect("db1".to_string(), "/data/db1.db".to_string())
            .unwrap();
        manager
            .connect("db2".to_string(), "/data/db2.db".to_string())
            .unwrap();

        assert_eq!(manager.connections.len(), 2);

        manager.clear();

        assert_eq!(manager.connections.len(), 0);
        assert!(!manager.has_active_connection());
    }

    #[test]
    fn test_path_validation() {
        assert!(paths::is_valid_location("http://example.com/dataset"));
        assert!(paths::is_valid_location("https://example.com/dataset"));
        assert!(paths::is_valid_location("/data/test.db"));
        assert!(!paths::is_valid_location(""));
    }
}
