//! Advanced GraphQL features
//!
//! This module provides advanced features like query optimization, introspection,
//! validation, and performance enhancements.

// Re-export advanced features
pub use crate::optimizer::*;
pub use crate::introspection::*;
pub use crate::validation::*;

/// Feature components
pub mod components {
    //! Individual feature components
    
    /// Query optimization and caching
    pub mod optimization {
        pub use crate::optimizer::*;
    }
    
    /// GraphQL introspection
    pub mod introspection {
        pub use crate::introspection::*;
    }
    
    /// Query validation and security
    pub mod validation {
        pub use crate::validation::*;
    }
}

/// Performance monitoring and metrics
pub mod performance {
    use std::time::{Duration, Instant};
    use std::collections::HashMap;
    
    /// Query performance metrics
    #[derive(Debug, Clone)]
    pub struct QueryMetrics {
        pub query_hash: String,
        pub execution_time: Duration,
        pub complexity_score: usize,
        pub depth: usize,
        pub cache_hit: bool,
        pub timestamp: Instant,
    }
    
    impl QueryMetrics {
        pub fn new(query_hash: String, execution_time: Duration, complexity_score: usize, depth: usize, cache_hit: bool) -> Self {
            Self {
                query_hash,
                execution_time,
                complexity_score,
                depth,
                cache_hit,
                timestamp: Instant::now(),
            }
        }
    }
    
    /// Performance tracker for collecting metrics
    #[derive(Debug, Default)]
    pub struct PerformanceTracker {
        metrics: Vec<QueryMetrics>,
        slow_queries: Vec<QueryMetrics>,
        slow_query_threshold: Duration,
    }
    
    impl PerformanceTracker {
        pub fn new() -> Self {
            Self {
                metrics: Vec::new(),
                slow_queries: Vec::new(),
                slow_query_threshold: Duration::from_millis(1000), // 1 second
            }
        }
        
        pub fn with_slow_query_threshold(mut self, threshold: Duration) -> Self {
            self.slow_query_threshold = threshold;
            self
        }
        
        pub fn record_query(&mut self, metrics: QueryMetrics) {
            if metrics.execution_time > self.slow_query_threshold {
                self.slow_queries.push(metrics.clone());
            }
            self.metrics.push(metrics);
        }
        
        pub fn get_average_execution_time(&self) -> Option<Duration> {
            if self.metrics.is_empty() {
                return None;
            }
            
            let total: Duration = self.metrics.iter().map(|m| m.execution_time).sum();
            Some(total / self.metrics.len() as u32)
        }
        
        pub fn get_cache_hit_rate(&self) -> f64 {
            if self.metrics.is_empty() {
                return 0.0;
            }
            
            let cache_hits = self.metrics.iter().filter(|m| m.cache_hit).count();
            cache_hits as f64 / self.metrics.len() as f64
        }
        
        pub fn get_slow_queries(&self) -> &[QueryMetrics] {
            &self.slow_queries
        }
        
        pub fn get_complexity_distribution(&self) -> HashMap<usize, usize> {
            let mut distribution = HashMap::new();
            
            for metric in &self.metrics {
                let bucket = (metric.complexity_score / 100) * 100; // Group by hundreds
                *distribution.entry(bucket).or_insert(0) += 1;
            }
            
            distribution
        }
    }
}

/// Security utilities
pub mod security {
    use std::collections::HashSet;
    use anyhow::Result;
    
    /// Common GraphQL security patterns
    pub struct SecurityPatterns;
    
    impl SecurityPatterns {
        /// Check if query contains potentially dangerous patterns
        pub fn has_dangerous_patterns(query: &str) -> bool {
            let dangerous_keywords = [
                "__schema",
                "__type", 
                "mutation",
                "subscription",
                "introspection"
            ];
            
            let query_lower = query.to_lowercase();
            dangerous_keywords.iter().any(|&keyword| query_lower.contains(keyword))
        }
        
        /// Extract operation names from a query
        pub fn extract_operation_names(query: &str) -> Vec<String> {
            // This is a simplified implementation
            // In practice, you'd want to parse the query properly
            let mut operations = Vec::new();
            
            for line in query.lines() {
                let line = line.trim();
                if line.starts_with("query ") || line.starts_with("mutation ") || line.starts_with("subscription ") {
                    if let Some(name_start) = line.find(' ') {
                        if let Some(name_end) = line[name_start + 1..].find([' ', '(', '{']) {
                            let name = &line[name_start + 1..name_start + 1 + name_end];
                            if !name.is_empty() {
                                operations.push(name.to_string());
                            }
                        }
                    }
                }
            }
            
            operations
        }
        
        /// Get recommended security settings for different environments
        pub fn recommended_validation_config(environment: Environment) -> crate::validation::ValidationConfig {
            match environment {
                Environment::Development => crate::validation::ValidationConfig {
                    max_depth: 15,
                    max_complexity: 2000,
                    max_aliases: 100,
                    max_root_fields: 50,
                    disable_introspection: false,
                    max_fragments: 100,
                    ..Default::default()
                },
                Environment::Production => crate::validation::ValidationConfig {
                    max_depth: 8,
                    max_complexity: 1000,
                    max_aliases: 20,
                    max_root_fields: 15,
                    disable_introspection: true,
                    max_fragments: 25,
                    ..Default::default()
                },
                Environment::Testing => crate::validation::ValidationConfig {
                    max_depth: 20,
                    max_complexity: 5000,
                    max_aliases: 200,
                    max_root_fields: 100,
                    disable_introspection: false,
                    max_fragments: 200,
                    ..Default::default()
                },
            }
        }
    }
    
    /// Deployment environment
    #[derive(Debug, Clone, Copy)]
    pub enum Environment {
        Development,
        Production,
        Testing,
    }
}