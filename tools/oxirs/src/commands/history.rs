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

    /// Get analytics for query history
    pub fn analytics(&self) -> HistoryAnalytics {
        let total = self.entries.len();
        let successful = self.entries.iter().filter(|e| e.success).count();
        let failed = total - successful;

        let avg_execution_time = if successful > 0 {
            let sum: f64 = self
                .entries
                .iter()
                .filter(|e| e.success && e.execution_time_ms.is_some())
                .map(|e| e.execution_time_ms.unwrap())
                .sum();
            sum / successful as f64
        } else {
            0.0
        };

        // Find slowest queries
        let mut sorted_by_time: Vec<&HistoryEntry> = self
            .entries
            .iter()
            .filter(|e| e.success && e.execution_time_ms.is_some())
            .collect();
        sorted_by_time.sort_by(|a, b| {
            b.execution_time_ms
                .partial_cmp(&a.execution_time_ms)
                .unwrap()
        });
        let slowest_queries: Vec<SlowQuery> = sorted_by_time
            .iter()
            .take(10)
            .map(|e| SlowQuery {
                query: e.query.clone(),
                execution_time_ms: e.execution_time_ms.unwrap(),
                result_count: e.result_count,
            })
            .collect();

        // Dataset usage statistics
        let mut dataset_usage: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for entry in &self.entries {
            *dataset_usage.entry(entry.dataset.clone()).or_insert(0) += 1;
        }

        // Query type distribution (SELECT, ASK, CONSTRUCT, DESCRIBE)
        let mut query_types: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for entry in &self.entries {
            let query_type = extract_query_type(&entry.query);
            *query_types.entry(query_type).or_insert(0) += 1;
        }

        // Common error patterns
        let mut error_patterns: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for entry in self.entries.iter().filter(|e| !e.success) {
            if let Some(error) = &entry.error {
                let pattern = classify_error(error);
                *error_patterns.entry(pattern).or_insert(0) += 1;
            }
        }

        HistoryAnalytics {
            total_queries: total,
            successful_queries: successful,
            failed_queries: failed,
            success_rate: if total > 0 {
                (successful as f64 / total as f64) * 100.0
            } else {
                0.0
            },
            avg_execution_time_ms: avg_execution_time,
            slowest_queries,
            dataset_usage,
            query_types,
            error_patterns,
        }
    }
}

/// Analytics data for query history
#[derive(Debug, Serialize, Deserialize)]
pub struct HistoryAnalytics {
    pub total_queries: usize,
    pub successful_queries: usize,
    pub failed_queries: usize,
    pub success_rate: f64,
    pub avg_execution_time_ms: f64,
    pub slowest_queries: Vec<SlowQuery>,
    pub dataset_usage: std::collections::HashMap<String, usize>,
    pub query_types: std::collections::HashMap<String, usize>,
    pub error_patterns: std::collections::HashMap<String, usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SlowQuery {
    pub query: String,
    pub execution_time_ms: f64,
    pub result_count: Option<usize>,
}

/// Extract query type from SPARQL query
fn extract_query_type(query: &str) -> String {
    let query_upper = query.to_uppercase();
    let lines: Vec<&str> = query_upper.lines().collect();

    for line in lines {
        let trimmed = line.trim();

        // Skip comments and PREFIX declarations
        if trimmed.starts_with('#') || trimmed.starts_with("PREFIX") {
            continue;
        }

        // Detect query type
        if trimmed.contains("SELECT") {
            return "SELECT".to_string();
        } else if trimmed.contains("ASK") {
            return "ASK".to_string();
        } else if trimmed.contains("CONSTRUCT") {
            return "CONSTRUCT".to_string();
        } else if trimmed.contains("DESCRIBE") {
            return "DESCRIBE".to_string();
        } else if trimmed.contains("INSERT") || trimmed.contains("DELETE") {
            return "UPDATE".to_string();
        }
    }

    "UNKNOWN".to_string()
}

/// Classify error into common patterns
fn classify_error(error: &str) -> String {
    let error_lower = error.to_lowercase();

    if error_lower.contains("syntax") || error_lower.contains("parse") {
        "Syntax Error".to_string()
    } else if error_lower.contains("timeout") {
        "Timeout".to_string()
    } else if error_lower.contains("not found") {
        "Not Found".to_string()
    } else if error_lower.contains("permission") || error_lower.contains("denied") {
        "Permission Denied".to_string()
    } else if error_lower.contains("connection") || error_lower.contains("network") {
        "Connection Error".to_string()
    } else if error_lower.contains("memory") || error_lower.contains("out of") {
        "Resource Exhaustion".to_string()
    } else {
        "Other Error".to_string()
    }
}

/// Query history commands
pub mod commands {
    use super::*;
    use colored::Colorize;

    /// Display query history analytics
    pub async fn analytics_command(dataset: Option<String>) -> Result<()> {
        let mut history = QueryHistory::new(QueryHistory::default_history_file(), 1000);
        history.load()?;

        // Filter by dataset if specified
        let filtered_entries: Vec<HistoryEntry> = if let Some(ds) = &dataset {
            history
                .entries()
                .iter()
                .filter(|e| &e.dataset == ds)
                .cloned()
                .collect()
        } else {
            history.entries().to_vec()
        };

        if filtered_entries.is_empty() {
            println!("📊 No query history found");
            return Ok(());
        }

        // Create temporary history for analytics
        let mut temp_history = QueryHistory::new(PathBuf::new(), 0);
        temp_history.entries = filtered_entries;
        let analytics = temp_history.analytics();

        println!("{}", "📊 Query History Analytics\n".bold());

        // Overall statistics
        println!("{}", "Overall Statistics:".bold());
        println!("  Total Queries: {}", analytics.total_queries);
        println!(
            "  Successful: {} ({}%)",
            analytics.successful_queries.to_string().green(),
            format!("{:.1}", analytics.success_rate).green()
        );
        println!(
            "  Failed: {} ({}%)",
            analytics.failed_queries.to_string().red(),
            format!("{:.1}", 100.0 - analytics.success_rate).red()
        );
        println!(
            "  Avg Execution Time: {:.2}ms",
            analytics.avg_execution_time_ms
        );
        println!();

        // Query type distribution
        if !analytics.query_types.is_empty() {
            println!("{}", "Query Type Distribution:".bold());
            for (qtype, count) in &analytics.query_types {
                let percentage = (*count as f64 / analytics.total_queries as f64) * 100.0;
                println!("  {}: {} ({:.1}%)", qtype, count, percentage);
            }
            println!();
        }

        // Dataset usage
        if analytics.dataset_usage.len() > 1 {
            println!("{}", "Dataset Usage:".bold());
            let mut sorted_datasets: Vec<_> = analytics.dataset_usage.iter().collect();
            sorted_datasets.sort_by(|a, b| b.1.cmp(a.1));
            for (dataset, count) in sorted_datasets.iter().take(10) {
                let percentage = (**count as f64 / analytics.total_queries as f64) * 100.0;
                println!("  {}: {} queries ({:.1}%)", dataset, count, percentage);
            }
            println!();
        }

        // Slowest queries
        if !analytics.slowest_queries.is_empty() {
            println!("{}", "Top 5 Slowest Queries:".bold());
            for (i, slow_query) in analytics.slowest_queries.iter().take(5).enumerate() {
                println!(
                    "  {}. {:.2}ms",
                    i + 1,
                    slow_query.execution_time_ms.to_string().yellow()
                );
                let preview = if slow_query.query.len() > 60 {
                    format!("{}...", &slow_query.query[..57])
                } else {
                    slow_query.query.clone()
                };
                println!("     {}", preview);
                if let Some(count) = slow_query.result_count {
                    println!("     Results: {} solutions", count);
                }
            }
            println!();
        }

        // Error patterns
        if !analytics.error_patterns.is_empty() {
            println!("{}", "Common Error Patterns:".bold());
            let mut sorted_errors: Vec<_> = analytics.error_patterns.iter().collect();
            sorted_errors.sort_by(|a, b| b.1.cmp(a.1));
            for (pattern, count) in sorted_errors {
                let percentage = (*count as f64 / analytics.failed_queries as f64) * 100.0;
                println!("  {}: {} ({:.1}%)", pattern.red(), count, percentage);
            }
            println!();
        }

        // Recommendations
        println!("{}", "💡 Recommendations:".bold());
        if analytics.avg_execution_time_ms > 1000.0 {
            println!("  • Consider optimizing slow queries (avg >1s)");
            println!("  • Use LIMIT clauses to reduce result set sizes");
            println!("  • Add indexes for frequently queried properties");
        }
        if analytics.success_rate < 80.0 {
            println!("  • Review and fix common error patterns");
            println!("  • Validate queries with: oxirs qparse <query>");
        }
        if let Some((most_common_error, _)) = analytics
            .error_patterns
            .iter()
            .max_by_key(|(_, count)| *count)
        {
            println!("  • Most common error: {}", most_common_error);
            println!("    Consider addressing this systematically");
        }

        Ok(())
    }

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
