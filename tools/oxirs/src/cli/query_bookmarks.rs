//! Query bookmark management for SPARQL REPL
//!
//! This module provides functionality for saving, managing, and retrieving
//! frequently-used SPARQL queries with metadata.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::cli::error::{CliError, CliResult};

/// A saved SPARQL query with metadata
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryBookmark {
    /// Unique name for the bookmark
    pub name: String,

    /// The SPARQL query text
    pub query: String,

    /// Optional description
    pub description: Option<String>,

    /// Tags for categorization and search
    pub tags: Vec<String>,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last update timestamp
    pub updated_at: DateTime<Utc>,

    /// Number of times this bookmark has been used
    pub use_count: u64,

    /// Last time this bookmark was used
    pub last_used: Option<DateTime<Utc>>,
}

impl QueryBookmark {
    /// Create a new bookmark
    pub fn new(name: String, query: String) -> Self {
        let now = Utc::now();
        Self {
            name,
            query,
            description: None,
            tags: Vec::new(),
            created_at: now,
            updated_at: now,
            use_count: 0,
            last_used: None,
        }
    }

    /// Set description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Update the query text
    pub fn update_query(&mut self, query: String) {
        self.query = query;
        self.updated_at = Utc::now();
    }

    /// Record usage of this bookmark
    pub fn record_usage(&mut self) {
        self.use_count += 1;
        self.last_used = Some(Utc::now());
    }
}

/// Configuration for bookmark storage
#[derive(Debug, Clone)]
pub struct BookmarkConfig {
    /// Path to the bookmarks file
    pub storage_path: PathBuf,

    /// Auto-save on changes
    pub auto_save: bool,
}

impl Default for BookmarkConfig {
    fn default() -> Self {
        let storage_path = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("oxirs")
            .join("bookmarks.json");

        Self {
            storage_path,
            auto_save: true,
        }
    }
}

impl BookmarkConfig {
    /// Create a new config with custom storage path
    pub fn with_path(path: PathBuf) -> Self {
        Self {
            storage_path: path,
            auto_save: true,
        }
    }

    /// Enable or disable auto-save
    pub fn auto_save(mut self, enabled: bool) -> Self {
        self.auto_save = enabled;
        self
    }
}

/// Manager for query bookmarks
pub struct BookmarkManager {
    bookmarks: HashMap<String, QueryBookmark>,
    config: BookmarkConfig,
    dirty: bool,
}

impl BookmarkManager {
    /// Create a new bookmark manager with default config
    pub fn new() -> CliResult<Self> {
        Self::with_config(BookmarkConfig::default())
    }

    /// Create a new bookmark manager with custom config
    pub fn with_config(config: BookmarkConfig) -> CliResult<Self> {
        let mut manager = Self {
            bookmarks: HashMap::new(),
            config,
            dirty: false,
        };

        // Try to load existing bookmarks
        if manager.config.storage_path.exists() {
            manager.load()?;
        }

        Ok(manager)
    }

    /// Save a new bookmark
    pub fn save(&mut self, name: String, query: String) -> CliResult<()> {
        if self.bookmarks.contains_key(&name) {
            return Err(CliError::invalid_arguments(format!(
                "Bookmark '{}' already exists. Use update() to modify it.",
                name
            )));
        }

        let bookmark = QueryBookmark::new(name.clone(), query);
        self.bookmarks.insert(name, bookmark);
        self.dirty = true;

        if self.config.auto_save {
            self.persist()?;
        }

        Ok(())
    }

    /// Save a bookmark with metadata
    pub fn save_with_metadata(
        &mut self,
        name: String,
        query: String,
        description: Option<String>,
        tags: Vec<String>,
    ) -> CliResult<()> {
        if self.bookmarks.contains_key(&name) {
            return Err(CliError::invalid_arguments(format!(
                "Bookmark '{}' already exists. Use update() to modify it.",
                name
            )));
        }

        let mut bookmark = QueryBookmark::new(name.clone(), query);
        if let Some(desc) = description {
            bookmark.description = Some(desc);
        }
        bookmark.tags = tags;

        self.bookmarks.insert(name, bookmark);
        self.dirty = true;

        if self.config.auto_save {
            self.persist()?;
        }

        Ok(())
    }

    /// Update an existing bookmark
    pub fn update(
        &mut self,
        name: &str,
        query: Option<String>,
        description: Option<String>,
        tags: Option<Vec<String>>,
    ) -> CliResult<()> {
        let bookmark = self
            .bookmarks
            .get_mut(name)
            .ok_or_else(|| CliError::not_found(format!("Bookmark '{}' not found", name)))?;

        if let Some(q) = query {
            bookmark.update_query(q);
        }

        if let Some(desc) = description {
            bookmark.description = Some(desc);
            bookmark.updated_at = Utc::now();
        }

        if let Some(t) = tags {
            bookmark.tags = t;
            bookmark.updated_at = Utc::now();
        }

        self.dirty = true;

        if self.config.auto_save {
            self.persist()?;
        }

        Ok(())
    }

    /// Delete a bookmark
    pub fn delete(&mut self, name: &str) -> CliResult<()> {
        self.bookmarks
            .remove(name)
            .ok_or_else(|| CliError::not_found(format!("Bookmark '{}' not found", name)))?;

        self.dirty = true;

        if self.config.auto_save {
            self.persist()?;
        }

        Ok(())
    }

    /// Get a bookmark by name and record usage
    pub fn get(&mut self, name: &str) -> CliResult<QueryBookmark> {
        let bookmark = self
            .bookmarks
            .get_mut(name)
            .ok_or_else(|| CliError::not_found(format!("Bookmark '{}' not found", name)))?;

        bookmark.record_usage();
        self.dirty = true;
        let result = bookmark.clone();

        if self.config.auto_save {
            self.persist()?;
        }

        Ok(result)
    }

    /// Get query text by bookmark name
    pub fn get_query(&mut self, name: &str) -> CliResult<String> {
        Ok(self.get(name)?.query)
    }

    /// List all bookmarks
    pub fn list(&self) -> Vec<&QueryBookmark> {
        let mut bookmarks: Vec<&QueryBookmark> = self.bookmarks.values().collect();
        bookmarks.sort_by(|a, b| a.name.cmp(&b.name));
        bookmarks
    }

    /// List bookmarks sorted by usage count (most used first)
    pub fn list_by_usage(&self) -> Vec<&QueryBookmark> {
        let mut bookmarks: Vec<&QueryBookmark> = self.bookmarks.values().collect();
        bookmarks.sort_by(|a, b| b.use_count.cmp(&a.use_count));
        bookmarks
    }

    /// List bookmarks sorted by last update (newest first)
    pub fn list_by_updated(&self) -> Vec<&QueryBookmark> {
        let mut bookmarks: Vec<&QueryBookmark> = self.bookmarks.values().collect();
        bookmarks.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        bookmarks
    }

    /// Search bookmarks by name, tags, or content
    pub fn search(&self, query: &str) -> Vec<&QueryBookmark> {
        let query_lower = query.to_lowercase();

        self.bookmarks
            .values()
            .filter(|bookmark| {
                // Search in name
                bookmark.name.to_lowercase().contains(&query_lower)
                    // Search in description
                    || bookmark.description.as_ref()
                        .map(|d| d.to_lowercase().contains(&query_lower))
                        .unwrap_or(false)
                    // Search in tags
                    || bookmark.tags.iter()
                        .any(|t| t.to_lowercase().contains(&query_lower))
                    // Search in query text
                    || bookmark.query.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    /// Filter bookmarks by tag
    pub fn filter_by_tag(&self, tag: &str) -> Vec<&QueryBookmark> {
        self.bookmarks
            .values()
            .filter(|bookmark| bookmark.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Get all unique tags
    pub fn get_all_tags(&self) -> Vec<String> {
        let mut tags: Vec<String> = self
            .bookmarks
            .values()
            .flat_map(|b| b.tags.iter().cloned())
            .collect();
        tags.sort();
        tags.dedup();
        tags
    }

    /// Load bookmarks from storage
    pub fn load(&mut self) -> CliResult<()> {
        let content = fs::read_to_string(&self.config.storage_path).map_err(CliError::io_error)?;

        self.bookmarks = serde_json::from_str(&content).map_err(|e| {
            CliError::serialization_error(format!("Failed to parse bookmarks: {}", e))
        })?;

        self.dirty = false;
        Ok(())
    }

    /// Persist bookmarks to storage
    pub fn persist(&mut self) -> CliResult<()> {
        if !self.dirty {
            return Ok(());
        }

        // Ensure directory exists
        if let Some(parent) = self.config.storage_path.parent() {
            fs::create_dir_all(parent).map_err(CliError::io_error)?;
        }

        let json = serde_json::to_string_pretty(&self.bookmarks).map_err(|e| {
            CliError::serialization_error(format!("Failed to serialize bookmarks: {}", e))
        })?;

        let mut file = fs::File::create(&self.config.storage_path).map_err(CliError::io_error)?;

        file.write_all(json.as_bytes())
            .map_err(CliError::io_error)?;

        self.dirty = false;
        Ok(())
    }

    /// Import bookmarks from a file
    pub fn import(&mut self, path: &Path, merge: bool) -> CliResult<usize> {
        let content = fs::read_to_string(path).map_err(CliError::io_error)?;

        let imported: HashMap<String, QueryBookmark> =
            serde_json::from_str(&content).map_err(|e| {
                CliError::serialization_error(format!("Failed to parse import file: {}", e))
            })?;

        let count = imported.len();

        if merge {
            // Merge imported bookmarks, keeping existing ones
            for (name, bookmark) in imported {
                self.bookmarks.entry(name).or_insert(bookmark);
            }
        } else {
            // Replace all bookmarks
            self.bookmarks = imported;
        }

        self.dirty = true;

        if self.config.auto_save {
            self.persist()?;
        }

        Ok(count)
    }

    /// Export bookmarks to a file
    pub fn export(&self, path: &Path) -> CliResult<()> {
        let json = serde_json::to_string_pretty(&self.bookmarks).map_err(|e| {
            CliError::serialization_error(format!("Failed to serialize bookmarks: {}", e))
        })?;

        let mut file = fs::File::create(path).map_err(CliError::io_error)?;

        file.write_all(json.as_bytes())
            .map_err(CliError::io_error)?;

        Ok(())
    }

    /// Get the number of bookmarks
    pub fn count(&self) -> usize {
        self.bookmarks.len()
    }

    /// Check if a bookmark exists
    pub fn exists(&self, name: &str) -> bool {
        self.bookmarks.contains_key(name)
    }

    /// Clear all bookmarks
    pub fn clear(&mut self) -> CliResult<()> {
        self.bookmarks.clear();
        self.dirty = true;

        if self.config.auto_save {
            self.persist()?;
        }

        Ok(())
    }
}

impl Default for BookmarkManager {
    fn default() -> Self {
        Self::new().expect("Failed to create default BookmarkManager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn create_test_manager() -> BookmarkManager {
        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join(format!("test_bookmarks_{}.json", Utc::now().timestamp()));
        let config = BookmarkConfig::with_path(test_file).auto_save(false);
        BookmarkManager::with_config(config).unwrap()
    }

    #[test]
    fn test_create_bookmark() {
        let mut manager = create_test_manager();

        manager
            .save(
                "test".to_string(),
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
            )
            .unwrap();

        assert_eq!(manager.count(), 1);
        assert!(manager.exists("test"));
    }

    #[test]
    fn test_save_duplicate_bookmark() {
        let mut manager = create_test_manager();

        manager
            .save(
                "test".to_string(),
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
            )
            .unwrap();
        let result = manager.save(
            "test".to_string(),
            "SELECT * WHERE { ?s ?p ?o }".to_string(),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_update_bookmark() {
        let mut manager = create_test_manager();

        manager
            .save(
                "test".to_string(),
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
            )
            .unwrap();
        manager
            .update(
                "test",
                Some("SELECT ?s WHERE { ?s ?p ?o }".to_string()),
                None,
                None,
            )
            .unwrap();

        let bookmark = manager.get("test").unwrap();
        assert_eq!(bookmark.query, "SELECT ?s WHERE { ?s ?p ?o }");
    }

    #[test]
    fn test_delete_bookmark() {
        let mut manager = create_test_manager();

        manager
            .save(
                "test".to_string(),
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
            )
            .unwrap();
        assert_eq!(manager.count(), 1);

        manager.delete("test").unwrap();
        assert_eq!(manager.count(), 0);
    }

    #[test]
    fn test_get_bookmark() {
        let mut manager = create_test_manager();

        manager
            .save(
                "test".to_string(),
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
            )
            .unwrap();
        let query = manager.get_query("test").unwrap();

        assert_eq!(query, "SELECT * WHERE { ?s ?p ?o }");
    }

    #[test]
    fn test_list_bookmarks() {
        let mut manager = create_test_manager();

        manager
            .save(
                "test1".to_string(),
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
            )
            .unwrap();
        manager
            .save(
                "test2".to_string(),
                "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
            )
            .unwrap();

        let bookmarks = manager.list();
        assert_eq!(bookmarks.len(), 2);
    }

    #[test]
    fn test_search_bookmarks() {
        let mut manager = create_test_manager();

        manager
            .save_with_metadata(
                "test1".to_string(),
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
                Some("Get all triples".to_string()),
                vec!["select".to_string(), "basic".to_string()],
            )
            .unwrap();

        manager
            .save_with_metadata(
                "test2".to_string(),
                "INSERT DATA { <x> <y> <z> }".to_string(),
                Some("Insert triple".to_string()),
                vec!["insert".to_string(), "update".to_string()],
            )
            .unwrap();

        let results = manager.search("select");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "test1");

        let results = manager.search("INSERT");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "test2");
    }

    #[test]
    fn test_filter_by_tag() {
        let mut manager = create_test_manager();

        manager
            .save_with_metadata(
                "test1".to_string(),
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
                None,
                vec!["basic".to_string()],
            )
            .unwrap();

        manager
            .save_with_metadata(
                "test2".to_string(),
                "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
                None,
                vec!["advanced".to_string()],
            )
            .unwrap();

        let results = manager.filter_by_tag("basic");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "test1");
    }

    #[test]
    fn test_get_all_tags() {
        let mut manager = create_test_manager();

        manager
            .save_with_metadata(
                "test1".to_string(),
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
                None,
                vec!["select".to_string(), "basic".to_string()],
            )
            .unwrap();

        manager
            .save_with_metadata(
                "test2".to_string(),
                "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
                None,
                vec!["select".to_string(), "advanced".to_string()],
            )
            .unwrap();

        let tags = manager.get_all_tags();
        assert_eq!(tags, vec!["advanced", "basic", "select"]);
    }

    #[test]
    fn test_usage_tracking() {
        let mut manager = create_test_manager();

        manager
            .save(
                "test".to_string(),
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
            )
            .unwrap();

        assert_eq!(manager.bookmarks.get("test").unwrap().use_count, 0);

        manager.get("test").unwrap();
        assert_eq!(manager.bookmarks.get("test").unwrap().use_count, 1);

        manager.get("test").unwrap();
        assert_eq!(manager.bookmarks.get("test").unwrap().use_count, 2);
    }

    #[test]
    fn test_list_by_usage() {
        let mut manager = create_test_manager();

        manager
            .save(
                "test1".to_string(),
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
            )
            .unwrap();
        manager
            .save(
                "test2".to_string(),
                "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
            )
            .unwrap();

        manager.get("test2").unwrap();
        manager.get("test2").unwrap();
        manager.get("test1").unwrap();

        let bookmarks = manager.list_by_usage();
        assert_eq!(bookmarks[0].name, "test2");
        assert_eq!(bookmarks[1].name, "test1");
    }

    #[test]
    fn test_persist_and_load() {
        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join(format!("test_persist_{}.json", Utc::now().timestamp()));

        {
            let mut manager = BookmarkManager::with_config(
                BookmarkConfig::with_path(test_file.clone()).auto_save(false),
            )
            .unwrap();

            manager
                .save(
                    "test".to_string(),
                    "SELECT * WHERE { ?s ?p ?o }".to_string(),
                )
                .unwrap();
            manager.persist().unwrap();
        }

        {
            let mut manager = BookmarkManager::with_config(
                BookmarkConfig::with_path(test_file.clone()).auto_save(false),
            )
            .unwrap();

            assert_eq!(manager.count(), 1);
            let query = manager.get_query("test").unwrap();
            assert_eq!(query, "SELECT * WHERE { ?s ?p ?o }");
        }

        // Cleanup
        let _ = fs::remove_file(test_file);
    }

    #[test]
    fn test_import_export() {
        let temp_dir = env::temp_dir();
        let export_file = temp_dir.join(format!("test_export_{}.json", Utc::now().timestamp()));

        let mut manager1 = create_test_manager();
        manager1
            .save(
                "test1".to_string(),
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
            )
            .unwrap();
        manager1
            .save(
                "test2".to_string(),
                "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
            )
            .unwrap();

        manager1.export(&export_file).unwrap();

        let mut manager2 = create_test_manager();
        let count = manager2.import(&export_file, false).unwrap();

        assert_eq!(count, 2);
        assert_eq!(manager2.count(), 2);
        assert!(manager2.exists("test1"));
        assert!(manager2.exists("test2"));

        // Cleanup
        let _ = fs::remove_file(export_file);
    }

    #[test]
    fn test_clear_bookmarks() {
        let mut manager = create_test_manager();

        manager
            .save(
                "test1".to_string(),
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
            )
            .unwrap();
        manager
            .save(
                "test2".to_string(),
                "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
            )
            .unwrap();

        assert_eq!(manager.count(), 2);

        manager.clear().unwrap();
        assert_eq!(manager.count(), 0);
    }
}
