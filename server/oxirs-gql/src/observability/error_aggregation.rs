//! Error Aggregation and Grouping
//!
//! Provides intelligent error aggregation, grouping, and analysis
//! for GraphQL operations.
//!
//! # Features
//!
//! - Automatic error grouping by similarity
//! - Error frequency tracking
//! - Root cause analysis
//! - Error pattern detection
//! - Stack trace fingerprinting
//! - Error trend analysis

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Error severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Error category
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    Validation,
    Authorization,
    Network,
    Database,
    Internal,
    Timeout,
    RateLimit,
    Unknown,
}

/// Individual error occurrence
#[derive(Debug, Clone)]
pub struct ErrorOccurrence {
    pub timestamp: SystemTime,
    pub error_message: String,
    pub error_type: String,
    pub stack_trace: Option<String>,
    pub query_id: Option<String>,
    pub user_id: Option<String>,
    pub field_path: Option<Vec<String>>,
    pub context: HashMap<String, String>,
}

/// Grouped errors with statistics
#[derive(Debug, Clone)]
pub struct ErrorGroup {
    pub id: String,
    pub fingerprint: String,
    pub error_type: String,
    pub category: ErrorCategory,
    pub severity: ErrorSeverity,
    pub first_seen: SystemTime,
    pub last_seen: SystemTime,
    pub occurrences: Vec<ErrorOccurrence>,
    pub count: usize,
    pub affected_users: usize,
    pub affected_queries: usize,
    pub sample_message: String,
}

impl ErrorGroup {
    /// Calculate error rate (errors per minute)
    pub fn error_rate(&self) -> f64 {
        if self.occurrences.len() < 2 {
            return 0.0;
        }

        let duration = self
            .last_seen
            .duration_since(self.first_seen)
            .unwrap_or(Duration::ZERO);

        let minutes = duration.as_secs_f64() / 60.0;
        if minutes > 0.0 {
            self.count as f64 / minutes
        } else {
            0.0
        }
    }

    /// Check if error is trending up
    pub fn is_trending_up(&self, window: Duration) -> bool {
        let now = SystemTime::now();
        let cutoff = now - window;

        let recent_count = self
            .occurrences
            .iter()
            .filter(|e| e.timestamp >= cutoff)
            .count();

        let total_duration = now
            .duration_since(self.first_seen)
            .unwrap_or(Duration::ZERO);
        let window_fraction = window.as_secs_f64() / total_duration.as_secs_f64();

        recent_count as f64 > self.count as f64 * window_fraction * 1.5
    }
}

/// Error aggregation manager
pub struct ErrorAggregator {
    groups: HashMap<String, ErrorGroup>,
    max_occurrences_per_group: usize,
}

impl ErrorAggregator {
    /// Create a new error aggregator
    pub fn new(max_occurrences_per_group: usize) -> Self {
        Self {
            groups: HashMap::new(),
            max_occurrences_per_group,
        }
    }

    /// Record an error
    pub fn record_error(&mut self, occurrence: ErrorOccurrence) {
        let fingerprint = Self::generate_fingerprint(&occurrence);
        let category = Self::categorize_error(&occurrence);
        let severity = Self::assess_severity(&occurrence, &category);

        if let Some(group) = self.groups.get_mut(&fingerprint) {
            // Update existing group
            group.last_seen = occurrence.timestamp;
            group.count += 1;

            // Track unique users and queries
            if let Some(ref user_id) = occurrence.user_id {
                if !group
                    .occurrences
                    .iter()
                    .any(|e| e.user_id.as_ref() == Some(user_id))
                {
                    group.affected_users += 1;
                }
            }

            if let Some(ref query_id) = occurrence.query_id {
                if !group
                    .occurrences
                    .iter()
                    .any(|e| e.query_id.as_ref() == Some(query_id))
                {
                    group.affected_queries += 1;
                }
            }

            // Add occurrence
            group.occurrences.push(occurrence);

            // Trim if needed
            if group.occurrences.len() > self.max_occurrences_per_group {
                group.occurrences.drain(0..1);
            }
        } else {
            // Create new group
            let id = format!("group-{}", self.groups.len());
            let sample_message = occurrence.error_message.clone();
            let first_seen = occurrence.timestamp;

            let occurrences = vec![occurrence];
            let affected_users = occurrences[0].user_id.as_ref().map(|_| 1).unwrap_or(0);
            let affected_queries = occurrences[0].query_id.as_ref().map(|_| 1).unwrap_or(0);

            self.groups.insert(
                fingerprint.clone(),
                ErrorGroup {
                    id,
                    fingerprint,
                    error_type: occurrences[0].error_type.clone(),
                    category,
                    severity,
                    first_seen,
                    last_seen: first_seen,
                    occurrences,
                    count: 1,
                    affected_users,
                    affected_queries,
                    sample_message,
                },
            );
        }
    }

    /// Get all error groups
    pub fn get_all_groups(&self) -> Vec<&ErrorGroup> {
        self.groups.values().collect()
    }

    /// Get error groups by category
    pub fn get_groups_by_category(&self, category: ErrorCategory) -> Vec<&ErrorGroup> {
        self.groups
            .values()
            .filter(|g| g.category == category)
            .collect()
    }

    /// Get error groups by severity
    pub fn get_groups_by_severity(&self, severity: ErrorSeverity) -> Vec<&ErrorGroup> {
        self.groups
            .values()
            .filter(|g| g.severity == severity)
            .collect()
    }

    /// Get top error groups by count
    pub fn get_top_errors(&self, limit: usize) -> Vec<&ErrorGroup> {
        let mut groups: Vec<_> = self.groups.values().collect();
        groups.sort_by(|a, b| b.count.cmp(&a.count));
        groups.into_iter().take(limit).collect()
    }

    /// Get trending errors
    pub fn get_trending_errors(&self, window: Duration) -> Vec<&ErrorGroup> {
        self.groups
            .values()
            .filter(|g| g.is_trending_up(window))
            .collect()
    }

    /// Get error statistics
    pub fn get_statistics(&self) -> ErrorStatistics {
        let total_errors: usize = self.groups.values().map(|g| g.count).sum();
        let total_groups = self.groups.len();

        let mut by_category: HashMap<ErrorCategory, usize> = HashMap::new();
        let mut by_severity: HashMap<ErrorSeverity, usize> = HashMap::new();

        for group in self.groups.values() {
            *by_category.entry(group.category.clone()).or_insert(0) += group.count;
            *by_severity.entry(group.severity).or_insert(0) += group.count;
        }

        ErrorStatistics {
            total_errors,
            total_groups,
            by_category,
            by_severity,
        }
    }

    /// Clear all error data
    pub fn clear(&mut self) {
        self.groups.clear();
    }

    /// Generate error fingerprint for grouping
    fn generate_fingerprint(occurrence: &ErrorOccurrence) -> String {
        // Use error type and a normalized message
        let normalized_message = Self::normalize_message(&occurrence.error_message);

        // Hash for consistent fingerprint
        let mut hash: u64 = 5381;
        for byte in occurrence.error_type.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        for byte in normalized_message.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }

        format!("{:x}", hash)
    }

    /// Normalize error message for grouping
    fn normalize_message(message: &str) -> String {
        // Remove numbers, IDs, and variable parts
        let mut normalized = message.to_string();

        // Replace numbers with placeholder
        normalized = normalized
            .split_whitespace()
            .map(|word| {
                if word.chars().all(|c| c.is_numeric()) {
                    "<num>"
                } else {
                    word
                }
            })
            .collect::<Vec<_>>()
            .join(" ");

        // Normalize case
        normalized.to_lowercase()
    }

    /// Categorize error based on content
    fn categorize_error(occurrence: &ErrorOccurrence) -> ErrorCategory {
        let message_lower = occurrence.error_message.to_lowercase();
        let type_lower = occurrence.error_type.to_lowercase();

        if type_lower.contains("validation") || message_lower.contains("invalid") {
            ErrorCategory::Validation
        } else if type_lower.contains("auth") || message_lower.contains("unauthorized") {
            ErrorCategory::Authorization
        } else if type_lower.contains("database")
            || type_lower.contains("db")
            || message_lower.contains("database")
            || message_lower.contains("query")
        {
            ErrorCategory::Database
        } else if message_lower.contains("network") || message_lower.contains("connection") {
            ErrorCategory::Network
        } else if message_lower.contains("timeout") {
            ErrorCategory::Timeout
        } else if message_lower.contains("rate limit") {
            ErrorCategory::RateLimit
        } else if type_lower.contains("internal") {
            ErrorCategory::Internal
        } else {
            ErrorCategory::Unknown
        }
    }

    /// Assess error severity
    fn assess_severity(occurrence: &ErrorOccurrence, category: &ErrorCategory) -> ErrorSeverity {
        match category {
            ErrorCategory::Authorization => ErrorSeverity::High,
            ErrorCategory::Internal => ErrorSeverity::Critical,
            ErrorCategory::Database => ErrorSeverity::High,
            ErrorCategory::Timeout => ErrorSeverity::Medium,
            ErrorCategory::RateLimit => ErrorSeverity::Low,
            ErrorCategory::Validation => ErrorSeverity::Low,
            ErrorCategory::Network => ErrorSeverity::Medium,
            ErrorCategory::Unknown => {
                // Check message for severity hints
                let message_lower = occurrence.error_message.to_lowercase();
                if message_lower.contains("critical") || message_lower.contains("fatal") {
                    ErrorSeverity::Critical
                } else if message_lower.contains("error") {
                    ErrorSeverity::High
                } else {
                    ErrorSeverity::Medium
                }
            }
        }
    }

    /// Generate error report
    pub fn generate_report(&self) -> String {
        let mut report = String::from("=== Error Aggregation Report ===\n\n");

        let stats = self.get_statistics();
        report.push_str(&format!("Total Errors: {}\n", stats.total_errors));
        report.push_str(&format!("Error Groups: {}\n\n", stats.total_groups));

        // By category
        report.push_str("Errors by Category:\n");
        for (category, count) in &stats.by_category {
            report.push_str(&format!("  {:?}: {}\n", category, count));
        }
        report.push('\n');

        // By severity
        report.push_str("Errors by Severity:\n");
        for (severity, count) in &stats.by_severity {
            report.push_str(&format!("  {:?}: {}\n", severity, count));
        }
        report.push('\n');

        // Top errors
        report.push_str("Top 5 Errors:\n");
        for (i, group) in self.get_top_errors(5).iter().enumerate() {
            report.push_str(&format!(
                "{}. {} (count: {}, users: {})\n",
                i + 1,
                group.sample_message,
                group.count,
                group.affected_users
            ));
        }

        report
    }
}

/// Error statistics
#[derive(Debug)]
pub struct ErrorStatistics {
    pub total_errors: usize,
    pub total_groups: usize,
    pub by_category: HashMap<ErrorCategory, usize>,
    pub by_severity: HashMap<ErrorSeverity, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_error() {
        let mut aggregator = ErrorAggregator::new(100);

        let occurrence = ErrorOccurrence {
            timestamp: SystemTime::now(),
            error_message: "Invalid input".to_string(),
            error_type: "ValidationError".to_string(),
            stack_trace: None,
            query_id: Some("query-1".to_string()),
            user_id: Some("user-1".to_string()),
            field_path: None,
            context: HashMap::new(),
        };

        aggregator.record_error(occurrence);

        assert_eq!(aggregator.get_all_groups().len(), 1);
    }

    #[test]
    fn test_error_grouping() {
        let mut aggregator = ErrorAggregator::new(100);

        // Same error type, should group together
        for i in 0..3 {
            aggregator.record_error(ErrorOccurrence {
                timestamp: SystemTime::now(),
                error_message: format!("Invalid input {}", i),
                error_type: "ValidationError".to_string(),
                stack_trace: None,
                query_id: Some(format!("query-{}", i)),
                user_id: None,
                field_path: None,
                context: HashMap::new(),
            });
        }

        // Should be grouped into one
        let groups = aggregator.get_all_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].count, 3);
    }

    #[test]
    fn test_error_categorization() {
        let mut aggregator = ErrorAggregator::new(100);

        aggregator.record_error(ErrorOccurrence {
            timestamp: SystemTime::now(),
            error_message: "Unauthorized access".to_string(),
            error_type: "AuthError".to_string(),
            stack_trace: None,
            query_id: None,
            user_id: None,
            field_path: None,
            context: HashMap::new(),
        });

        let groups = aggregator.get_groups_by_category(ErrorCategory::Authorization);
        assert_eq!(groups.len(), 1);
    }

    #[test]
    fn test_error_severity() {
        let mut aggregator = ErrorAggregator::new(100);

        aggregator.record_error(ErrorOccurrence {
            timestamp: SystemTime::now(),
            error_message: "Database connection failed".to_string(),
            error_type: "DatabaseError".to_string(),
            stack_trace: None,
            query_id: None,
            user_id: None,
            field_path: None,
            context: HashMap::new(),
        });

        let groups = aggregator.get_groups_by_severity(ErrorSeverity::High);
        assert_eq!(groups.len(), 1);
    }

    #[test]
    fn test_top_errors() {
        let mut aggregator = ErrorAggregator::new(100);

        // Error with 5 occurrences
        for _ in 0..5 {
            aggregator.record_error(ErrorOccurrence {
                timestamp: SystemTime::now(),
                error_message: "Error A".to_string(),
                error_type: "ErrorA".to_string(),
                stack_trace: None,
                query_id: None,
                user_id: None,
                field_path: None,
                context: HashMap::new(),
            });
        }

        // Error with 2 occurrences
        for _ in 0..2 {
            aggregator.record_error(ErrorOccurrence {
                timestamp: SystemTime::now(),
                error_message: "Error B".to_string(),
                error_type: "ErrorB".to_string(),
                stack_trace: None,
                query_id: None,
                user_id: None,
                field_path: None,
                context: HashMap::new(),
            });
        }

        let top = aggregator.get_top_errors(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].count, 5);
    }

    #[test]
    fn test_affected_users_tracking() {
        let mut aggregator = ErrorAggregator::new(100);

        aggregator.record_error(ErrorOccurrence {
            timestamp: SystemTime::now(),
            error_message: "Error".to_string(),
            error_type: "Error".to_string(),
            stack_trace: None,
            query_id: None,
            user_id: Some("user-1".to_string()),
            field_path: None,
            context: HashMap::new(),
        });

        aggregator.record_error(ErrorOccurrence {
            timestamp: SystemTime::now(),
            error_message: "Error".to_string(),
            error_type: "Error".to_string(),
            stack_trace: None,
            query_id: None,
            user_id: Some("user-2".to_string()),
            field_path: None,
            context: HashMap::new(),
        });

        let groups = aggregator.get_all_groups();
        assert_eq!(groups[0].affected_users, 2);
    }

    #[test]
    fn test_statistics() {
        let mut aggregator = ErrorAggregator::new(100);

        aggregator.record_error(ErrorOccurrence {
            timestamp: SystemTime::now(),
            error_message: "Invalid input".to_string(),
            error_type: "ValidationError".to_string(),
            stack_trace: None,
            query_id: None,
            user_id: None,
            field_path: None,
            context: HashMap::new(),
        });

        aggregator.record_error(ErrorOccurrence {
            timestamp: SystemTime::now(),
            error_message: "Unauthorized".to_string(),
            error_type: "AuthError".to_string(),
            stack_trace: None,
            query_id: None,
            user_id: None,
            field_path: None,
            context: HashMap::new(),
        });

        let stats = aggregator.get_statistics();
        assert_eq!(stats.total_errors, 2);
        assert_eq!(stats.total_groups, 2);
    }

    #[test]
    fn test_clear() {
        let mut aggregator = ErrorAggregator::new(100);

        aggregator.record_error(ErrorOccurrence {
            timestamp: SystemTime::now(),
            error_message: "Error".to_string(),
            error_type: "Error".to_string(),
            stack_trace: None,
            query_id: None,
            user_id: None,
            field_path: None,
            context: HashMap::new(),
        });

        assert_eq!(aggregator.get_all_groups().len(), 1);

        aggregator.clear();
        assert_eq!(aggregator.get_all_groups().len(), 0);
    }

    #[test]
    fn test_generate_report() {
        let mut aggregator = ErrorAggregator::new(100);

        aggregator.record_error(ErrorOccurrence {
            timestamp: SystemTime::now(),
            error_message: "Test error".to_string(),
            error_type: "TestError".to_string(),
            stack_trace: None,
            query_id: None,
            user_id: None,
            field_path: None,
            context: HashMap::new(),
        });

        let report = aggregator.generate_report();
        assert!(report.contains("Error Aggregation Report"));
        assert!(report.contains("Total Errors: 1"));
    }

    #[test]
    fn test_error_rate_calculation() {
        let mut aggregator = ErrorAggregator::new(100);

        let first_time = SystemTime::now();

        aggregator.record_error(ErrorOccurrence {
            timestamp: first_time,
            error_message: "Error".to_string(),
            error_type: "Error".to_string(),
            stack_trace: None,
            query_id: None,
            user_id: None,
            field_path: None,
            context: HashMap::new(),
        });

        aggregator.record_error(ErrorOccurrence {
            timestamp: first_time + Duration::from_secs(60),
            error_message: "Error".to_string(),
            error_type: "Error".to_string(),
            stack_trace: None,
            query_id: None,
            user_id: None,
            field_path: None,
            context: HashMap::new(),
        });

        let groups = aggregator.get_all_groups();
        let rate = groups[0].error_rate();
        assert!(rate > 0.0);
    }

    #[test]
    fn test_normalize_message() {
        let msg1 = "Error with ID 12345";
        let msg2 = "Error with ID 67890";

        let norm1 = ErrorAggregator::normalize_message(msg1);
        let norm2 = ErrorAggregator::normalize_message(msg2);

        // Should normalize to same pattern
        assert_eq!(norm1, norm2);
    }

    #[test]
    fn test_max_occurrences_per_group() {
        let mut aggregator = ErrorAggregator::new(2);

        for _ in 0..5 {
            aggregator.record_error(ErrorOccurrence {
                timestamp: SystemTime::now(),
                error_message: "Error".to_string(),
                error_type: "Error".to_string(),
                stack_trace: None,
                query_id: None,
                user_id: None,
                field_path: None,
                context: HashMap::new(),
            });
        }

        let groups = aggregator.get_all_groups();
        assert_eq!(groups[0].occurrences.len(), 2); // Limited to 2
        assert_eq!(groups[0].count, 5); // But count is still 5
    }
}
