//! Performance profiling support for TTL parsing
//!
//! This module provides profiling capabilities to track and analyze performance
//! of RDF parsing operations. It helps identify bottlenecks and optimize parsing.
//!
//! # Features
//!
//! - **Throughput Metrics**: Measure triples/second and MB/second
//! - **Duration Tracking**: Track total parsing time
//! - **Resource Counting**: Count triples, tokens, and bytes processed
//! - **Human-Readable Reports**: Generate formatted statistics
//!
//! # Example: Basic Profiling
//!
//! ```rust
//! use oxirs_ttl::profiling::ParsingStats;
//!
//! let mut stats = ParsingStats::new();
//! stats.start();
//!
//! // ... perform parsing ...
//! stats.triple_count = 1000;
//! stats.bytes_processed = 50_000;
//!
//! stats.stop();
//!
//! println!("Throughput: {:.0} triples/s", stats.triples_per_second());
//! println!("Speed: {:.2} MB/s", stats.mb_per_second());
//! ```
//!
//! # Example: Profiling with TtlProfiler
//!
//! ```rust
//! use oxirs_ttl::TtlProfiler;
//! use oxirs_ttl::turtle::TurtleParser;
//! use oxirs_ttl::Parser;
//! use std::io::Cursor;
//!
//! let data = r#"
//! @prefix ex: <http://example.org/> .
//! ex:s ex:p "o" .
//! "#;
//!
//! let mut profiler = TtlProfiler::new();
//! profiler.start();
//!
//! let parser = TurtleParser::new();
//! for result in parser.for_reader(Cursor::new(data)) {
//!     let _triple = result?;
//!     profiler.record_triple();
//! }
//!
//! profiler.stop();
//! println!("{}", profiler.report());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Example: Complete Report
//!
//! ```rust
//! use oxirs_ttl::profiling::ParsingStats;
//! use std::time::Duration;
//!
//! let mut stats = ParsingStats::new();
//! stats.triple_count = 100_000;
//! stats.bytes_processed = 5_000_000;
//! stats.total_duration = Duration::from_secs(2);
//!
//! // Generate detailed report
//! let report = stats.report();
//! println!("{}", report);
//! ```

use std::time::{Duration, Instant};

/// Profiling statistics for parsing operations
///
/// Tracks performance metrics including duration, throughput, and resource usage.
///
/// # Example
///
/// ```rust
/// use oxirs_ttl::profiling::ParsingStats;
///
/// let mut stats = ParsingStats::new();
/// stats.start();
/// // ... parsing work ...
/// stats.triple_count = 1000;
/// stats.stop();
///
/// println!("Parsed {} triples in {:.2}s",
///          stats.triple_count,
///          stats.total_duration.as_secs_f64());
/// ```
#[derive(Debug, Clone, Default)]
pub struct ParsingStats {
    /// Total parsing time
    pub total_duration: Duration,
    /// Number of triples parsed
    pub triple_count: usize,
    /// Number of tokens processed
    pub token_count: usize,
    /// Bytes processed
    pub bytes_processed: usize,
    /// Parse start time
    pub start_time: Option<Instant>,
    /// Parse end time
    pub end_time: Option<Instant>,
}

impl ParsingStats {
    /// Create new parsing statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Start profiling
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Stop profiling
    pub fn stop(&mut self) {
        self.end_time = Some(Instant::now());
        if let (Some(start), Some(end)) = (self.start_time, self.end_time) {
            self.total_duration = end.duration_since(start);
        }
    }

    /// Get parsing throughput (triples per second)
    pub fn triples_per_second(&self) -> f64 {
        if self.total_duration.as_secs_f64() > 0.0 {
            self.triple_count as f64 / self.total_duration.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get bytes per second
    pub fn bytes_per_second(&self) -> f64 {
        if self.total_duration.as_secs_f64() > 0.0 {
            self.bytes_processed as f64 / self.total_duration.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get MB per second
    pub fn mb_per_second(&self) -> f64 {
        self.bytes_per_second() / (1024.0 * 1024.0)
    }

    /// Generate a human-readable report
    pub fn report(&self) -> String {
        format!(
            "Parsing Statistics:\n\
             - Duration: {:.3}s\n\
             - Triples: {}\n\
             - Tokens: {}\n\
             - Bytes: {} ({:.2} MB)\n\
             - Throughput: {:.0} triples/s\n\
             - Speed: {:.2} MB/s",
            self.total_duration.as_secs_f64(),
            self.triple_count,
            self.token_count,
            self.bytes_processed,
            self.bytes_processed as f64 / (1024.0 * 1024.0),
            self.triples_per_second(),
            self.mb_per_second()
        )
    }
}

/// Performance profiler for TTL parsing
pub struct TtlProfiler {
    /// Parsing statistics
    pub stats: ParsingStats,
    /// Whether profiling is enabled
    enabled: bool,
}

impl TtlProfiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            stats: ParsingStats::new(),
            enabled: true,
        }
    }

    /// Create a disabled profiler (no overhead)
    pub fn disabled() -> Self {
        Self {
            stats: ParsingStats::new(),
            enabled: false,
        }
    }

    /// Enable profiling
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable profiling
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Start profiling
    pub fn start(&mut self) {
        if self.enabled {
            self.stats.start();
        }
    }

    /// Stop profiling
    pub fn stop(&mut self) {
        if self.enabled {
            self.stats.stop();
        }
    }

    /// Record a triple
    pub fn record_triple(&mut self) {
        if self.enabled {
            self.stats.triple_count += 1;
        }
    }

    /// Record multiple triples
    pub fn record_triples(&mut self, count: usize) {
        if self.enabled {
            self.stats.triple_count += count;
        }
    }

    /// Record a token
    pub fn record_token(&mut self) {
        if self.enabled {
            self.stats.token_count += 1;
        }
    }

    /// Record bytes processed
    pub fn record_bytes(&mut self, bytes: usize) {
        if self.enabled {
            self.stats.bytes_processed += bytes;
        }
    }

    /// Get the statistics
    pub fn statistics(&self) -> &ParsingStats {
        &self.stats
    }

    /// Generate a report
    pub fn report(&self) -> String {
        self.stats.report()
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.stats = ParsingStats::new();
    }
}

impl Default for TtlProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_parsing_stats() {
        let mut stats = ParsingStats::new();
        stats.triple_count = 1000;
        stats.bytes_processed = 50000;
        stats.start();
        thread::sleep(Duration::from_millis(100));
        stats.stop();

        assert!(stats.total_duration.as_millis() >= 100);
        assert!(stats.triples_per_second() > 0.0);
        assert!(stats.bytes_per_second() > 0.0);

        let report = stats.report();
        assert!(report.contains("1000"));
        assert!(report.contains("triples/s"));
    }

    #[test]
    fn test_profiler() {
        let mut profiler = TtlProfiler::new();
        assert!(profiler.is_enabled());

        profiler.start();
        profiler.record_triples(100);
        profiler.record_bytes(5000);
        profiler.stop();

        assert_eq!(profiler.statistics().triple_count, 100);
        assert_eq!(profiler.statistics().bytes_processed, 5000);
    }

    #[test]
    fn test_disabled_profiler() {
        let mut profiler = TtlProfiler::disabled();
        assert!(!profiler.is_enabled());

        profiler.start();
        profiler.record_triples(100);
        profiler.stop();

        // When disabled, stats are not recorded (no overhead)
        assert_eq!(profiler.statistics().triple_count, 0);
        assert_eq!(profiler.statistics().total_duration.as_secs(), 0);
    }

    #[test]
    fn test_enable_disable() {
        let mut profiler = TtlProfiler::new();

        profiler.disable();
        assert!(!profiler.is_enabled());

        profiler.enable();
        assert!(profiler.is_enabled());
    }

    #[test]
    fn test_reset() {
        let mut profiler = TtlProfiler::new();
        profiler.record_triples(50);
        profiler.record_bytes(1000);

        assert_eq!(profiler.statistics().triple_count, 50);

        profiler.reset();
        assert_eq!(profiler.statistics().triple_count, 0);
        assert_eq!(profiler.statistics().bytes_processed, 0);
    }
}
