//! Query history management
//!
//! Track, list, and replay SPARQL queries

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Query history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// Unique entry ID
    pub id: usize,
    /// Timestamp when query was executed
    pub timestamp: DateTime<Utc>,
    /// Dataset name
    pub dataset: String,
    /// SPARQL query text
    pub query: String,
    /// Execution time in milliseconds
    pub execution_time_ms: Option<f64>,
    /// Result count (number of solutions)
    pub result_count: Option<usize>,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Query history manager
pub struct QueryHistory {
    /// History entries
    entries: Vec<HistoryEntry>,
    /// Path to history file
    history_file: PathBuf,
    /// Maximum number of entries to keep
    max_entries: usize,
}

impl QueryHistory {
    /// Create new query history manager
    pub fn new(history_file: PathBuf, max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            history_file,
            max_entries,
        }
    }

    /// Get default history file path
    pub fn default_history_file() -> PathBuf {
        let mut path = dirs::data_dir().unwrap_or_else(|| PathBuf::from("."));
        path.push("oxirs");
        path.push("query_history.json");
        path
    }

    /// Load history from file
    pub fn load(&mut self) -> Result<()> {
        if !self.history_file.exists() {
            return Ok(());
        }

        let content = fs::read_to_string(&self.history_file)
            .with_context(|| format!("Failed to read history file: {:?}", self.history_file))?;

        self.entries =
            serde_json::from_str(&content).with_context(|| "Failed to parse history file")?;

        Ok(())
    }

    /// Save history to file
    pub fn save(&self) -> Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = self.history_file.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create history directory: {:?}", parent))?;
        }

        let content = serde_json::to_string_pretty(&self.entries)
            .with_context(|| "Failed to serialize history")?;

        fs::write(&self.history_file, content)
            .with_context(|| format!("Failed to write history file: {:?}", self.history_file))?;

        Ok(())
    }

    /// Add query to history
    pub fn add_entry(&mut self, entry: HistoryEntry) -> Result<()> {
        self.entries.push(entry);

        // Trim to max entries
        if self.entries.len() > self.max_entries {
            self.entries.drain(0..self.entries.len() - self.max_entries);
        }

        // Renumber IDs
        for (i, entry) in self.entries.iter_mut().enumerate() {
            entry.id = i + 1;
        }

        self.save()
    }

    /// Get all entries
    pub fn entries(&self) -> &[HistoryEntry] {
        &self.entries
    }

    /// Get entry by ID
    pub fn get_entry(&self, id: usize) -> Option<&HistoryEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Clear all history
    pub fn clear(&mut self) -> Result<()> {
        self.entries.clear();
        self.save()
    }

    /// Search history by query text
    pub fn search(&self, query_text: &str) -> Vec<&HistoryEntry> {
        self.entries
            .iter()
            .filter(|e| e.query.contains(query_text))
            .collect()
    }

    /// Get recent entries
    pub fn recent(&self, n: usize) -> Vec<&HistoryEntry> {
        let start = if self.entries.len() > n {
            self.entries.len() - n
        } else {
            0
        };
        self.entries[start..].iter().collect()
    }
}

/// Query history commands
pub mod commands {
    use super::*;

    /// List query history
    pub async fn list_command(limit: Option<usize>, dataset: Option<String>) -> Result<()> {
        let mut history = QueryHistory::new(QueryHistory::default_history_file(), 1000);
        history.load()?;

        let entries: Vec<&HistoryEntry> = if let Some(ds) = dataset {
            history
                .entries()
                .iter()
                .filter(|e| e.dataset == ds)
                .collect()
        } else {
            history.entries().iter().collect()
        };

        if entries.is_empty() {
            println!("📜 No query history found");
            return Ok(());
        }

        println!("📜 Query History\n");

        let limit = limit.unwrap_or(20);
        let display_entries: Vec<&&HistoryEntry> = entries.iter().rev().take(limit).collect();

        for entry in display_entries.iter().rev() {
            let status = if entry.success { "✅" } else { "❌" };
            let time = entry.timestamp.format("%Y-%m-%d %H:%M:%S");

            println!(
                "{} #{} - {} | Dataset: {}",
                status, entry.id, time, entry.dataset
            );

            // Show truncated query
            let query_preview = if entry.query.len() > 80 {
                format!("{}...", &entry.query[..77])
            } else {
                entry.query.clone()
            };
            println!("   Query: {}", query_preview);

            if let Some(exec_time) = entry.execution_time_ms {
                println!("   Execution: {:.2}ms", exec_time);
            }

            if let Some(count) = entry.result_count {
                println!("   Results: {} solutions", count);
            }

            if let Some(error) = &entry.error {
                println!("   Error: {}", error);
            }

            println!();
        }

        println!("Total: {} queries", entries.len());
        println!("\n💡 Use 'oxirs history show <id>' to see full query");
        println!("💡 Use 'oxirs history replay <id>' to re-execute a query");

        Ok(())
    }

    /// Show full query details
    pub async fn show_command(id: usize) -> Result<()> {
        let mut history = QueryHistory::new(QueryHistory::default_history_file(), 1000);
        history.load()?;

        let entry = history
            .get_entry(id)
            .ok_or_else(|| anyhow::anyhow!("Query #{} not found in history", id))?;

        println!("📋 Query Details\n");
        println!("ID: #{}", entry.id);
        println!("Timestamp: {}", entry.timestamp.format("%Y-%m-%d %H:%M:%S"));
        println!("Dataset: {}", entry.dataset);
        println!(
            "Status: {}",
            if entry.success {
                "Success ✅"
            } else {
                "Failed ❌"
            }
        );

        if let Some(exec_time) = entry.execution_time_ms {
            println!("Execution Time: {:.2}ms", exec_time);
        }

        if let Some(count) = entry.result_count {
            println!("Result Count: {} solutions", count);
        }

        if let Some(error) = &entry.error {
            println!("Error: {}", error);
        }

        println!(
            "\nQuery:\n{}\n{}\n{}\n",
            "─".repeat(80),
            entry.query,
            "─".repeat(80)
        );

        Ok(())
    }

    /// Replay a query from history
    pub async fn replay_command(id: usize, output: Option<String>) -> Result<()> {
        let mut history = QueryHistory::new(QueryHistory::default_history_file(), 1000);
        history.load()?;

        let entry = history
            .get_entry(id)
            .ok_or_else(|| anyhow::anyhow!("Query #{} not found in history", id))?;

        println!("🔄 Replaying Query #{}\n", id);
        println!("Dataset: {}", entry.dataset);
        println!("Query: {}\n", entry.query);

        // Re-execute the query using the query command
        let output_format = output.unwrap_or_else(|| "table".to_string());
        crate::commands::query::run(
            entry.dataset.clone(),
            entry.query.clone(),
            false,
            output_format,
        )
        .await?;

        Ok(())
    }

    /// Search query history
    pub async fn search_command(query_text: String) -> Result<()> {
        let mut history = QueryHistory::new(QueryHistory::default_history_file(), 1000);
        history.load()?;

        let results = history.search(&query_text);

        if results.is_empty() {
            println!("🔍 No queries found matching '{}'", query_text);
            return Ok(());
        }

        println!("🔍 Search Results for '{}'\n", query_text);

        for entry in results.iter().rev() {
            let status = if entry.success { "✅" } else { "❌" };
            let time = entry.timestamp.format("%Y-%m-%d %H:%M:%S");

            println!(
                "{} #{} - {} | Dataset: {}",
                status, entry.id, time, entry.dataset
            );

            // Highlight matching text
            let query_preview = if entry.query.len() > 80 {
                format!("{}...", &entry.query[..77])
            } else {
                entry.query.clone()
            };
            println!("   Query: {}", query_preview);
            println!();
        }

        println!("Found: {} matches", results.len());

        Ok(())
    }

    /// Clear query history
    pub async fn clear_command() -> Result<()> {
        let mut history = QueryHistory::new(QueryHistory::default_history_file(), 1000);
        history.load()?;

        let count = history.entries().len();
        history.clear()?;

        println!("✅ Query history cleared");
        println!("   Removed {} entries", count);

        Ok(())
    }

    /// Show history statistics
    pub async fn stats_command() -> Result<()> {
        let mut history = QueryHistory::new(QueryHistory::default_history_file(), 1000);
        history.load()?;

        let total = history.entries().len();
        let successful = history.entries().iter().filter(|e| e.success).count();
        let failed = total - successful;

        // Calculate average execution time
        let avg_exec_time: f64 = history
            .entries()
            .iter()
            .filter_map(|e| e.execution_time_ms)
            .sum::<f64>()
            / history
                .entries()
                .iter()
                .filter(|e| e.execution_time_ms.is_some())
                .count() as f64;

        // Count by dataset
        let mut dataset_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for entry in history.entries() {
            *dataset_counts.entry(entry.dataset.clone()).or_insert(0) += 1;
        }

        println!("📊 Query History Statistics\n");
        println!("Total Queries: {}", total);
        println!(
            "Successful: {} ({:.1}%)",
            successful,
            (successful as f64 / total as f64) * 100.0
        );
        println!(
            "Failed: {} ({:.1}%)",
            failed,
            (failed as f64 / total as f64) * 100.0
        );
        println!("Average Execution Time: {:.2}ms", avg_exec_time);
        println!();

        if !dataset_counts.is_empty() {
            println!("Queries by Dataset:");
            let mut sorted_datasets: Vec<_> = dataset_counts.iter().collect();
            sorted_datasets.sort_by(|a, b| b.1.cmp(a.1));
            for (dataset, count) in sorted_datasets.iter().take(10) {
                println!("  {} - {} queries", dataset, count);
            }
        }

        Ok(())
    }
}

/// Record a query execution in history
pub fn record_query(
    dataset: &str,
    query: &str,
    execution_time_ms: Option<f64>,
    result_count: Option<usize>,
    success: bool,
    error: Option<String>,
) -> Result<()> {
    let mut history = QueryHistory::new(QueryHistory::default_history_file(), 1000);
    history.load().ok(); // Ignore errors on first run

    let entry = HistoryEntry {
        id: 0, // Will be renumbered when added
        timestamp: Utc::now(),
        dataset: dataset.to_string(),
        query: query.to_string(),
        execution_time_ms,
        result_count,
        success,
        error,
    };

    history.add_entry(entry)?;

    Ok(())
}
