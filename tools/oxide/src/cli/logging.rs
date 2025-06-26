//! Structured logging with JSON support
//!
//! Provides comprehensive logging with structured output, performance tracking,
//! and multiple output formats.

use serde::{Serialize, Deserialize};
use std::time::Instant;
use std::collections::HashMap;
use tracing::Level;
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    registry::LookupSpan,
    Layer, Registry,
};
use chrono::{DateTime, Utc};

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Output format (text, json, pretty)
    pub format: LogFormat,
    /// Include timestamps
    pub timestamps: bool,
    /// Include source location (file:line)
    pub source_location: bool,
    /// Include thread IDs
    pub thread_ids: bool,
    /// Performance logging threshold (ms)
    pub perf_threshold_ms: Option<u64>,
    /// Log file path (if any)
    pub file: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum LogFormat {
    Text,
    Json,
    Pretty,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: LogFormat::Text,
            timestamps: true,
            source_location: false,
            thread_ids: false,
            perf_threshold_ms: Some(1000),
            file: None,
        }
    }
}

/// Initialize logging system
pub fn init_logging(config: &LogConfig) -> Result<(), Box<dyn std::error::Error>> {
    let level = match config.level.to_lowercase().as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };

    let registry = Registry::default();
    
    // Console output layer
    let console_layer = match config.format {
        LogFormat::Text => {
            let layer = fmt::layer()
                .with_target(config.source_location)
                .with_thread_ids(config.thread_ids)
                .with_span_events(FmtSpan::CLOSE);
            
            if config.timestamps {
                layer.boxed()
            } else {
                layer.without_time().boxed()
            }
        }
        LogFormat::Json => {
            let layer = fmt::layer()
                .json()
                .with_target(config.source_location)
                .with_thread_ids(config.thread_ids);
            
            if config.timestamps {
                layer.boxed()
            } else {
                layer.without_time().boxed()
            }
        }
        LogFormat::Pretty => {
            let layer = fmt::layer()
                .pretty()
                .with_target(config.source_location)
                .with_thread_ids(config.thread_ids);
            
            if config.timestamps {
                layer.boxed()
            } else {
                layer.without_time().boxed()
            }
        }
    };
    
    let subscriber = registry
        .with(console_layer)
        .with(tracing_subscriber::filter::LevelFilter::from_level(level));
    
    // Add file layer if configured
    if let Some(ref file_path) = config.file {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(file_path)?;
        
        let file_layer = fmt::layer()
            .json()
            .with_writer(std::sync::Mutex::new(file));
        
        let subscriber = subscriber.with(file_layer);
        tracing::subscriber::set_global_default(subscriber)?;
    } else {
        tracing::subscriber::set_global_default(subscriber)?;
    }
    
    Ok(())
}

/// Structured log event
#[derive(Debug, Serialize, Deserialize)]
pub struct LogEvent {
    pub timestamp: DateTime<Utc>,
    pub level: String,
    pub message: String,
    pub target: String,
    pub fields: HashMap<String, serde_json::Value>,
    pub span: Option<SpanInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SpanInfo {
    pub name: String,
    pub target: String,
    pub fields: HashMap<String, serde_json::Value>,
}

/// Performance tracking
pub struct PerfLogger {
    operation: String,
    start: Instant,
    metadata: HashMap<String, serde_json::Value>,
}

impl PerfLogger {
    /// Start timing an operation
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            start: Instant::now(),
            metadata: HashMap::new(),
        }
    }
    
    /// Add metadata to the performance log
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Serialize) {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.metadata.insert(key.into(), json_value);
        }
    }
    
    /// Complete the operation and log if exceeds threshold
    pub fn complete(self, threshold_ms: Option<u64>) {
        let duration = self.start.elapsed();
        let duration_ms = duration.as_millis() as u64;
        
        if let Some(threshold) = threshold_ms {
            if duration_ms >= threshold {
                tracing::warn!(
                    operation = %self.operation,
                    duration_ms = duration_ms,
                    metadata = ?self.metadata,
                    "Slow operation detected"
                );
            }
        }
        
        tracing::debug!(
            operation = %self.operation,
            duration_ms = duration_ms,
            metadata = ?self.metadata,
            "Operation completed"
        );
    }
}

/// Command execution logger
pub struct CommandLogger {
    command: String,
    args: Vec<String>,
    start: Instant,
}

impl CommandLogger {
    pub fn new(command: impl Into<String>, args: Vec<String>) -> Self {
        let command = command.into();
        
        tracing::info!(
            command = %command,
            args = ?args,
            "Command started"
        );
        
        Self {
            command,
            args,
            start: Instant::now(),
        }
    }
    
    pub fn success(self) {
        let duration = self.start.elapsed();
        
        tracing::info!(
            command = %self.command,
            args = ?self.args,
            duration_ms = duration.as_millis() as u64,
            status = "success",
            "Command completed"
        );
    }
    
    pub fn error(self, error: &dyn std::error::Error) {
        let duration = self.start.elapsed();
        
        tracing::error!(
            command = %self.command,
            args = ?self.args,
            duration_ms = duration.as_millis() as u64,
            status = "error",
            error = %error,
            "Command failed"
        );
    }
}

/// Query execution logger
pub struct QueryLogger {
    query_type: String,
    dataset: String,
    start: Instant,
    metadata: HashMap<String, serde_json::Value>,
}

impl QueryLogger {
    pub fn new(query_type: impl Into<String>, dataset: impl Into<String>) -> Self {
        Self {
            query_type: query_type.into(),
            dataset: dataset.into(),
            start: Instant::now(),
            metadata: HashMap::new(),
        }
    }
    
    pub fn add_query_text(&mut self, query: &str) {
        // Log first 200 chars of query
        let preview = if query.len() > 200 {
            format!("{}...", &query[..200])
        } else {
            query.to_string()
        };
        
        self.metadata.insert("query_preview".to_string(), serde_json::Value::String(preview));
        self.metadata.insert("query_length".to_string(), serde_json::Value::Number(query.len().into()));
    }
    
    pub fn complete(self, result_count: usize) {
        let duration = self.start.elapsed();
        
        tracing::info!(
            query_type = %self.query_type,
            dataset = %self.dataset,
            duration_ms = duration.as_millis() as u64,
            result_count = result_count,
            metadata = ?self.metadata,
            "Query executed"
        );
    }
    
    pub fn error(self, error: &str) {
        let duration = self.start.elapsed();
        
        tracing::error!(
            query_type = %self.query_type,
            dataset = %self.dataset,
            duration_ms = duration.as_millis() as u64,
            error = %error,
            metadata = ?self.metadata,
            "Query failed"
        );
    }
}

/// Data operation logger
pub struct DataLogger {
    operation: String,
    dataset: String,
    start: Instant,
    bytes_processed: u64,
    items_processed: u64,
}

impl DataLogger {
    pub fn new(operation: impl Into<String>, dataset: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            dataset: dataset.into(),
            start: Instant::now(),
            bytes_processed: 0,
            items_processed: 0,
        }
    }
    
    pub fn update_progress(&mut self, bytes: u64, items: u64) {
        self.bytes_processed = bytes;
        self.items_processed = items;
    }
    
    pub fn complete(self) {
        let duration = self.start.elapsed();
        let throughput_mbps = if duration.as_secs() > 0 {
            (self.bytes_processed as f64 / 1_048_576.0) / duration.as_secs_f64()
        } else {
            0.0
        };
        
        tracing::info!(
            operation = %self.operation,
            dataset = %self.dataset,
            duration_ms = duration.as_millis() as u64,
            bytes_processed = self.bytes_processed,
            items_processed = self.items_processed,
            throughput_mbps = throughput_mbps,
            "Data operation completed"
        );
    }
}

/// Log analysis utilities
pub mod analysis {
    use super::*;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    
    /// Analyze log file for patterns
    pub fn analyze_log_file(path: &str) -> Result<LogAnalysis, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        let mut analysis = LogAnalysis::default();
        
        for line in reader.lines() {
            let line = line?;
            
            // Try to parse as JSON
            if let Ok(event) = serde_json::from_str::<LogEvent>(&line) {
                analysis.process_event(event);
            }
        }
        
        Ok(analysis)
    }
    
    #[derive(Default, Serialize)]
    pub struct LogAnalysis {
        pub total_events: usize,
        pub events_by_level: HashMap<String, usize>,
        pub slow_operations: Vec<SlowOperation>,
        pub errors: Vec<ErrorSummary>,
        pub command_stats: HashMap<String, CommandStats>,
    }
    
    #[derive(Serialize)]
    pub struct SlowOperation {
        pub operation: String,
        pub duration_ms: u64,
        pub timestamp: DateTime<Utc>,
    }
    
    #[derive(Serialize)]
    pub struct ErrorSummary {
        pub message: String,
        pub count: usize,
        pub first_seen: DateTime<Utc>,
        pub last_seen: DateTime<Utc>,
    }
    
    #[derive(Default, Serialize)]
    pub struct CommandStats {
        pub count: usize,
        pub success_count: usize,
        pub error_count: usize,
        pub avg_duration_ms: u64,
    }
    
    impl LogAnalysis {
        fn process_event(&mut self, event: LogEvent) {
            self.total_events += 1;
            
            *self.events_by_level.entry(event.level.clone()).or_insert(0) += 1;
            
            // Extract slow operations
            if event.level == "WARN" && event.message.contains("Slow operation") {
                if let Some(duration) = event.fields.get("duration_ms").and_then(|v| v.as_u64()) {
                    self.slow_operations.push(SlowOperation {
                        operation: event.fields.get("operation")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string(),
                        duration_ms: duration,
                        timestamp: event.timestamp,
                    });
                }
            }
            
            // Track errors
            if event.level == "ERROR" {
                // Implementation for error tracking
            }
            
            // Track command statistics
            if event.message.contains("Command") {
                // Implementation for command stats
            }
        }
    }
}

/// Development mode logging helpers
pub mod dev {
    use super::*;
    
    /// Enable debug logging for specific modules
    pub fn enable_debug_for_modules(modules: &[&str]) {
        // This would be implemented with tracing's dynamic filtering
        for module in modules {
            tracing::debug!("Enabling debug logging for module: {}", module);
        }
    }
    
    /// Log a debug table
    pub fn debug_table(title: &str, headers: &[&str], rows: Vec<Vec<String>>) {
        use prettytable::{Table, Row, Cell};
        
        let mut table = Table::new();
        table.add_row(Row::new(headers.iter().map(|h| Cell::new(h)).collect()));
        
        for row in rows {
            table.add_row(Row::new(row.iter().map(|c| Cell::new(c)).collect()));
        }
        
        tracing::debug!("{}\n{}", title, table);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[test]
    fn test_log_config_default() {
        let config = LogConfig::default();
        assert_eq!(config.level, "info");
        assert_eq!(config.format, LogFormat::Text);
        assert!(config.timestamps);
    }
    
    #[test]
    fn test_perf_logger() {
        let mut logger = PerfLogger::new("test_operation");
        logger.add_metadata("items", 100);
        logger.add_metadata("dataset", "test_db");
        
        // Simulate some work
        std::thread::sleep(Duration::from_millis(10));
        
        logger.complete(Some(5)); // 5ms threshold
    }
    
    #[test]
    fn test_command_logger() {
        let logger = CommandLogger::new("query", vec!["mydb".to_string(), "SELECT * WHERE { ?s ?p ?o }".to_string()]);
        
        // Simulate command execution
        std::thread::sleep(Duration::from_millis(10));
        
        logger.success();
    }
}