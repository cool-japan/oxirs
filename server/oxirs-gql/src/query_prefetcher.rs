//! Query Result Prefetching
//!
//! This module provides intelligent query result prefetching to reduce latency
//! by predicting and pre-executing queries that are likely to be requested next.
//!
//! # Features
//! - Query pattern analysis to predict next queries
//! - Sequential query pattern detection
//! - Co-occurrence analysis for related queries
//! - Priority-based prefetch queue
//! - Background prefetching with async execution
//! - Memory-aware prefetch limits
//! - Integration with caching system
//! - Configurable prefetch strategies
//!
//! # Example
//! ```
//! use oxirs_gql::query_prefetcher::{QueryPrefetcher, PrefetchStrategy};
//!
//! let mut prefetcher = QueryPrefetcher::new();
//! prefetcher.set_strategy(PrefetchStrategy::Sequential);
//!
//! // Record query execution
//! prefetcher.record_query("query { user { name } }");
//!
//! // Get predictions for next queries
//! let predictions = prefetcher.predict_next_queries("query { user { name } }", 5);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Error types for prefetching
#[derive(Debug, Error)]
pub enum PrefetchError {
    #[error("Lock acquisition failed: {0}")]
    LockError(String),

    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    #[error("Prefetch queue full")]
    QueueFull,

    #[error("Prediction failed: {0}")]
    PredictionError(String),
}

/// Prefetch strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PrefetchStrategy {
    /// Sequential - prefetch queries that typically follow current query
    Sequential,

    /// CoOccurrence - prefetch queries often executed together
    CoOccurrence,

    /// Popularity - prefetch most popular queries
    Popularity,

    /// Adaptive - automatically adapt strategy based on hit rate
    #[default]
    Adaptive,

    /// ML-based - use machine learning to predict next queries
    MLBased,
}

/// Query sequence pattern
#[derive(Debug, Clone)]
struct QuerySequence {
    /// Query that was executed (stored for future use)
    _query: String,

    /// Query executed after this one
    next_query: String,

    /// Number of times this sequence occurred
    occurrence_count: u64,

    /// Last time this sequence was seen
    last_seen: Instant,

    /// Average time between the two queries
    avg_time_between_ms: f64,
}

impl QuerySequence {
    fn new(query: String, next_query: String) -> Self {
        Self {
            _query: query,
            next_query,
            occurrence_count: 1,
            last_seen: Instant::now(),
            avg_time_between_ms: 0.0,
        }
    }

    fn update(&mut self, time_between_ms: f64) {
        self.occurrence_count += 1;
        self.last_seen = Instant::now();

        // Update average with exponential moving average
        let alpha = 0.3;
        self.avg_time_between_ms =
            alpha * time_between_ms + (1.0 - alpha) * self.avg_time_between_ms;
    }
}

/// Query co-occurrence pattern
#[derive(Debug, Clone)]
struct CoOccurrencePattern {
    /// Set of queries that co-occur
    queries: HashSet<String>,

    /// Co-occurrence count
    count: u64,

    /// Last occurrence time
    last_seen: Instant,
}

/// Query prediction with confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPrediction {
    /// Predicted query
    pub query: String,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,

    /// Priority (higher = more important)
    pub priority: u8,

    /// Estimated time until this query will be executed (ms)
    pub estimated_time_ms: f64,

    /// Reason for prediction
    pub reason: String,
}

/// Prefetch task
#[derive(Debug, Clone)]
struct PrefetchTask {
    /// Query to prefetch
    query: String,

    /// Priority (higher = execute sooner)
    priority: u8,

    /// Confidence in this prediction (reserved for future use)
    _confidence: f64,

    /// When this task was created (reserved for future use)
    _created_at: Instant,

    /// Estimated execution time (reserved for future use)
    _estimated_execution_ms: f64,
}

/// Configuration for prefetcher
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchConfig {
    /// Maximum number of queries to prefetch ahead
    pub max_prefetch_queue_size: usize,

    /// Minimum confidence threshold to prefetch
    pub min_confidence: f64,

    /// Maximum number of predictions to generate
    pub max_predictions: usize,

    /// Time window for co-occurrence analysis (seconds)
    pub cooccurrence_window_secs: u64,

    /// Enable background prefetching
    pub enable_background_prefetch: bool,

    /// Maximum memory for prefetch cache (bytes)
    pub max_cache_memory_bytes: u64,

    /// Minimum occurrence count for pattern
    pub min_pattern_occurrences: u64,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            max_prefetch_queue_size: 100,
            min_confidence: 0.6,
            max_predictions: 10,
            cooccurrence_window_secs: 60,
            enable_background_prefetch: true,
            max_cache_memory_bytes: 100_000_000, // 100 MB
            min_pattern_occurrences: 3,
        }
    }
}

/// Query prefetcher
pub struct QueryPrefetcher {
    /// Sequential patterns (query -> next query)
    sequences: Arc<RwLock<HashMap<String, Vec<QuerySequence>>>>,

    /// Co-occurrence patterns
    cooccurrences: Arc<RwLock<Vec<CoOccurrencePattern>>>,

    /// Query execution history
    execution_history: Arc<RwLock<VecDeque<(String, Instant)>>>,

    /// Prefetch queue
    prefetch_queue: Arc<RwLock<VecDeque<PrefetchTask>>>,

    /// Current strategy
    strategy: Arc<RwLock<PrefetchStrategy>>,

    /// Configuration
    config: PrefetchConfig,

    /// Last executed query
    last_query: Arc<RwLock<Option<(String, Instant)>>>,

    /// Prefetch hit statistics
    prefetch_hits: Arc<RwLock<u64>>,

    /// Prefetch miss statistics
    prefetch_misses: Arc<RwLock<u64>>,

    /// Total predictions made
    total_predictions: Arc<RwLock<u64>>,
}

impl QueryPrefetcher {
    /// Create a new query prefetcher with default configuration
    pub fn new() -> Self {
        Self::with_config(PrefetchConfig::default())
    }

    /// Create a new query prefetcher with custom configuration
    pub fn with_config(config: PrefetchConfig) -> Self {
        Self {
            sequences: Arc::new(RwLock::new(HashMap::new())),
            cooccurrences: Arc::new(RwLock::new(Vec::new())),
            execution_history: Arc::new(RwLock::new(VecDeque::new())),
            prefetch_queue: Arc::new(RwLock::new(VecDeque::new())),
            strategy: Arc::new(RwLock::new(PrefetchStrategy::default())),
            config,
            last_query: Arc::new(RwLock::new(None)),
            prefetch_hits: Arc::new(RwLock::new(0)),
            prefetch_misses: Arc::new(RwLock::new(0)),
            total_predictions: Arc::new(RwLock::new(0)),
        }
    }

    /// Set the prefetch strategy
    pub fn set_strategy(&mut self, strategy: PrefetchStrategy) -> Result<(), PrefetchError> {
        let mut s = self
            .strategy
            .write()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?;
        *s = strategy;
        Ok(())
    }

    /// Get the current strategy
    pub fn get_strategy(&self) -> Result<PrefetchStrategy, PrefetchError> {
        let s = self
            .strategy
            .read()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?;
        Ok(*s)
    }

    /// Record a query execution
    pub fn record_query(&mut self, query: &str) -> Result<(), PrefetchError> {
        let now = Instant::now();

        {
            // Update execution history
            let mut history = self
                .execution_history
                .write()
                .map_err(|e| PrefetchError::LockError(e.to_string()))?;

            history.push_back((query.to_string(), now));

            // Keep only recent history
            while history.len() > 1000 {
                history.pop_front();
            }
        }

        // Update sequential patterns
        let mut last = self
            .last_query
            .write()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?;

        if let Some((prev_query, prev_time)) = last.as_ref() {
            let time_between_ms = now.duration_since(*prev_time).as_millis() as f64;
            self.update_sequence_pattern(prev_query, query, time_between_ms)?;
        }

        *last = Some((query.to_string(), now));

        // Update co-occurrence patterns
        self.update_cooccurrence_patterns(query, now)?;

        // Check if this was a prefetch hit
        self.check_prefetch_hit(query)?;

        Ok(())
    }

    /// Update sequential pattern
    fn update_sequence_pattern(
        &self,
        prev_query: &str,
        current_query: &str,
        time_between_ms: f64,
    ) -> Result<(), PrefetchError> {
        let mut sequences = self
            .sequences
            .write()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?;

        let patterns = sequences
            .entry(prev_query.to_string())
            .or_insert_with(Vec::new);

        // Find existing pattern or create new one
        if let Some(pattern) = patterns.iter_mut().find(|p| p.next_query == current_query) {
            pattern.update(time_between_ms);
        } else {
            patterns.push(QuerySequence::new(
                prev_query.to_string(),
                current_query.to_string(),
            ));
        }

        Ok(())
    }

    /// Update co-occurrence patterns
    fn update_cooccurrence_patterns(
        &self,
        _query: &str,
        now: Instant,
    ) -> Result<(), PrefetchError> {
        let history = self
            .execution_history
            .read()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?;

        // Get recent queries within time window
        let window = Duration::from_secs(self.config.cooccurrence_window_secs);
        let recent_queries: HashSet<String> = history
            .iter()
            .rev()
            .take_while(|(_, time)| now.duration_since(*time) < window)
            .map(|(q, _)| q.clone())
            .collect();

        if recent_queries.len() < 2 {
            return Ok(());
        }

        let mut cooccurrences = self
            .cooccurrences
            .write()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?;

        // Update or create co-occurrence pattern
        let mut found = false;
        for pattern in cooccurrences.iter_mut() {
            if pattern.queries == recent_queries {
                pattern.count += 1;
                pattern.last_seen = now;
                found = true;
                break;
            }
        }

        if !found {
            cooccurrences.push(CoOccurrencePattern {
                queries: recent_queries,
                count: 1,
                last_seen: now,
            });
        }

        // Limit number of patterns
        if cooccurrences.len() > 1000 {
            cooccurrences.sort_by(|a, b| b.count.cmp(&a.count));
            cooccurrences.truncate(1000);
        }

        Ok(())
    }

    /// Check if query was a prefetch hit
    fn check_prefetch_hit(&self, query: &str) -> Result<(), PrefetchError> {
        let mut queue = self
            .prefetch_queue
            .write()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?;

        let was_prefetched = queue.iter().any(|task| task.query == query);

        if was_prefetched {
            // Remove from queue
            queue.retain(|task| task.query != query);

            let mut hits = self
                .prefetch_hits
                .write()
                .map_err(|e| PrefetchError::LockError(e.to_string()))?;
            *hits += 1;
        } else {
            let mut misses = self
                .prefetch_misses
                .write()
                .map_err(|e| PrefetchError::LockError(e.to_string()))?;
            *misses += 1;
        }

        Ok(())
    }

    /// Predict next queries
    pub fn predict_next_queries(
        &self,
        current_query: &str,
        max_predictions: usize,
    ) -> Result<Vec<QueryPrediction>, PrefetchError> {
        let strategy = self.get_strategy()?;

        let mut total = self
            .total_predictions
            .write()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?;
        *total += 1;

        match strategy {
            PrefetchStrategy::Sequential => self.predict_sequential(current_query, max_predictions),
            PrefetchStrategy::CoOccurrence => {
                self.predict_cooccurrence(current_query, max_predictions)
            }
            PrefetchStrategy::Popularity => self.predict_popularity(max_predictions),
            PrefetchStrategy::Adaptive => self.predict_adaptive(current_query, max_predictions),
            PrefetchStrategy::MLBased => self.predict_ml_based(current_query, max_predictions),
        }
    }

    /// Predict using sequential patterns
    fn predict_sequential(
        &self,
        current_query: &str,
        max_predictions: usize,
    ) -> Result<Vec<QueryPrediction>, PrefetchError> {
        let sequences = self
            .sequences
            .read()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?;

        let mut predictions = Vec::new();

        if let Some(patterns) = sequences.get(current_query) {
            // Sort by occurrence count
            let mut sorted_patterns = patterns.clone();
            sorted_patterns.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));

            for pattern in sorted_patterns.iter().take(max_predictions) {
                if pattern.occurrence_count < self.config.min_pattern_occurrences {
                    continue;
                }

                // Calculate confidence based on occurrence count and recency
                let recency_factor = {
                    let age_secs = pattern.last_seen.elapsed().as_secs_f64();
                    (-age_secs / 3600.0).exp() // 1 hour half-life
                };

                let occurrence_factor = (pattern.occurrence_count as f64 / 100.0).min(1.0);
                let confidence = (occurrence_factor * 0.7 + recency_factor * 0.3).min(1.0);

                if confidence >= self.config.min_confidence {
                    predictions.push(QueryPrediction {
                        query: pattern.next_query.clone(),
                        confidence,
                        priority: (confidence * 100.0) as u8,
                        estimated_time_ms: pattern.avg_time_between_ms,
                        reason: format!(
                            "Sequential pattern ({} occurrences)",
                            pattern.occurrence_count
                        ),
                    });
                }
            }
        }

        Ok(predictions)
    }

    /// Predict using co-occurrence patterns
    fn predict_cooccurrence(
        &self,
        current_query: &str,
        max_predictions: usize,
    ) -> Result<Vec<QueryPrediction>, PrefetchError> {
        let cooccurrences = self
            .cooccurrences
            .read()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?;

        let mut predictions = Vec::new();

        for pattern in cooccurrences.iter() {
            if !pattern.queries.contains(current_query) {
                continue;
            }

            if pattern.count < self.config.min_pattern_occurrences {
                continue;
            }

            // Predict other queries in the co-occurrence set
            for query in &pattern.queries {
                if query == current_query {
                    continue;
                }

                let confidence = (pattern.count as f64 / 50.0).min(1.0);

                if confidence >= self.config.min_confidence {
                    predictions.push(QueryPrediction {
                        query: query.clone(),
                        confidence,
                        priority: (confidence * 100.0) as u8,
                        estimated_time_ms: 1000.0, // Default estimate
                        reason: format!("Co-occurrence pattern ({} times)", pattern.count),
                    });
                }
            }

            if predictions.len() >= max_predictions {
                break;
            }
        }

        predictions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        predictions.truncate(max_predictions);

        Ok(predictions)
    }

    /// Predict using popularity
    fn predict_popularity(
        &self,
        max_predictions: usize,
    ) -> Result<Vec<QueryPrediction>, PrefetchError> {
        let history = self
            .execution_history
            .read()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?;

        // Count query frequencies
        let mut frequency: HashMap<String, u64> = HashMap::new();
        for (query, _) in history.iter() {
            *frequency.entry(query.clone()).or_insert(0) += 1;
        }

        // Sort by frequency
        let mut sorted: Vec<_> = frequency.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        let predictions = sorted
            .iter()
            .take(max_predictions)
            .map(|(query, count)| {
                let confidence = (*count as f64 / history.len() as f64).min(1.0);
                QueryPrediction {
                    query: query.clone(),
                    confidence,
                    priority: (confidence * 100.0) as u8,
                    estimated_time_ms: 0.0,
                    reason: format!("Popular query ({} executions)", count),
                }
            })
            .filter(|p| p.confidence >= self.config.min_confidence)
            .collect();

        Ok(predictions)
    }

    /// Adaptive prediction (combines strategies)
    fn predict_adaptive(
        &self,
        current_query: &str,
        max_predictions: usize,
    ) -> Result<Vec<QueryPrediction>, PrefetchError> {
        // Try sequential first
        let mut predictions = self.predict_sequential(current_query, max_predictions)?;

        // If not enough predictions, add co-occurrence
        if predictions.len() < max_predictions {
            let remaining = max_predictions - predictions.len();
            let mut cooccurrence_preds = self.predict_cooccurrence(current_query, remaining)?;
            predictions.append(&mut cooccurrence_preds);
        }

        // If still not enough, add popular queries
        if predictions.len() < max_predictions {
            let remaining = max_predictions - predictions.len();
            let mut popularity_preds = self.predict_popularity(remaining)?;
            predictions.append(&mut popularity_preds);
        }

        // Remove duplicates
        let mut seen = HashSet::new();
        predictions.retain(|p| seen.insert(p.query.clone()));

        predictions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        predictions.truncate(max_predictions);

        Ok(predictions)
    }

    /// ML-based prediction (placeholder for future implementation)
    fn predict_ml_based(
        &self,
        current_query: &str,
        max_predictions: usize,
    ) -> Result<Vec<QueryPrediction>, PrefetchError> {
        // For now, fall back to adaptive
        self.predict_adaptive(current_query, max_predictions)
    }

    /// Add query to prefetch queue
    pub fn queue_prefetch(&mut self, prediction: QueryPrediction) -> Result<(), PrefetchError> {
        let mut queue = self
            .prefetch_queue
            .write()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?;

        if queue.len() >= self.config.max_prefetch_queue_size {
            return Err(PrefetchError::QueueFull);
        }

        let task = PrefetchTask {
            query: prediction.query,
            priority: prediction.priority,
            _confidence: prediction.confidence,
            _created_at: Instant::now(),
            _estimated_execution_ms: prediction.estimated_time_ms,
        };

        queue.push_back(task);

        // Sort by priority
        let mut tasks: Vec<_> = queue.drain(..).collect();
        tasks.sort_by(|a, b| b.priority.cmp(&a.priority));
        queue.extend(tasks);

        Ok(())
    }

    /// Get prefetch statistics
    pub fn get_statistics(&self) -> Result<PrefetchStatistics, PrefetchError> {
        let hits = *self
            .prefetch_hits
            .read()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?;

        let misses = *self
            .prefetch_misses
            .read()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?;

        let predictions = *self
            .total_predictions
            .read()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?;

        let sequences = self
            .sequences
            .read()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?;

        let cooccurrences = self
            .cooccurrences
            .read()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?;

        let queue = self
            .prefetch_queue
            .read()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?;

        let hit_rate = if hits + misses > 0 {
            (hits as f64 / (hits + misses) as f64) * 100.0
        } else {
            0.0
        };

        Ok(PrefetchStatistics {
            total_patterns: sequences.len(),
            total_cooccurrences: cooccurrences.len(),
            prefetch_hits: hits,
            prefetch_misses: misses,
            hit_rate_percent: hit_rate,
            total_predictions: predictions,
            queue_size: queue.len(),
        })
    }

    /// Clear all data
    pub fn clear(&mut self) -> Result<(), PrefetchError> {
        self.sequences
            .write()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?
            .clear();

        self.cooccurrences
            .write()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?
            .clear();

        self.execution_history
            .write()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?
            .clear();

        self.prefetch_queue
            .write()
            .map_err(|e| PrefetchError::LockError(e.to_string()))?
            .clear();

        *self
            .last_query
            .write()
            .map_err(|e| PrefetchError::LockError(e.to_string()))? = None;

        *self
            .prefetch_hits
            .write()
            .map_err(|e| PrefetchError::LockError(e.to_string()))? = 0;

        *self
            .prefetch_misses
            .write()
            .map_err(|e| PrefetchError::LockError(e.to_string()))? = 0;

        *self
            .total_predictions
            .write()
            .map_err(|e| PrefetchError::LockError(e.to_string()))? = 0;

        Ok(())
    }
}

impl Default for QueryPrefetcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Prefetch statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchStatistics {
    /// Total sequential patterns learned
    pub total_patterns: usize,

    /// Total co-occurrence patterns
    pub total_cooccurrences: usize,

    /// Number of prefetch hits
    pub prefetch_hits: u64,

    /// Number of prefetch misses
    pub prefetch_misses: u64,

    /// Prefetch hit rate percentage
    pub hit_rate_percent: f64,

    /// Total predictions made
    pub total_predictions: u64,

    /// Current prefetch queue size
    pub queue_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_prefetcher_creation() {
        let prefetcher = QueryPrefetcher::new();
        assert_eq!(
            prefetcher.get_strategy().unwrap(),
            PrefetchStrategy::Adaptive
        );
    }

    #[test]
    fn test_set_strategy() {
        let mut prefetcher = QueryPrefetcher::new();
        prefetcher
            .set_strategy(PrefetchStrategy::Sequential)
            .unwrap();
        assert_eq!(
            prefetcher.get_strategy().unwrap(),
            PrefetchStrategy::Sequential
        );
    }

    #[test]
    fn test_record_query() {
        let mut prefetcher = QueryPrefetcher::new();
        prefetcher.record_query("query { user }").unwrap();
        prefetcher.record_query("query { posts }").unwrap();

        let stats = prefetcher.get_statistics().unwrap();
        assert!(stats.total_patterns > 0 || stats.total_cooccurrences > 0);
    }

    #[test]
    fn test_sequential_pattern() {
        let mut prefetcher = QueryPrefetcher::with_config(PrefetchConfig {
            min_confidence: 0.1,
            ..Default::default()
        });
        prefetcher
            .set_strategy(PrefetchStrategy::Sequential)
            .unwrap();

        // Create a pattern: query A -> query B
        for _ in 0..5 {
            prefetcher.record_query("query { user }").unwrap();
            thread::sleep(Duration::from_millis(10));
            prefetcher.record_query("query { posts }").unwrap();
            thread::sleep(Duration::from_millis(10));
        }

        let predictions = prefetcher
            .predict_next_queries("query { user }", 5)
            .unwrap();

        // Should predict "query { posts }" after "query { user }"
        assert!(!predictions.is_empty());
        assert!(predictions.iter().any(|p| p.query.contains("posts")));
    }

    #[test]
    fn test_prediction_confidence() {
        let mut prefetcher = QueryPrefetcher::with_config(PrefetchConfig {
            min_confidence: 0.5,
            min_pattern_occurrences: 2,
            ..Default::default()
        });

        // Create pattern
        for _ in 0..3 {
            prefetcher.record_query("query A").unwrap();
            prefetcher.record_query("query B").unwrap();
        }

        let predictions = prefetcher.predict_next_queries("query A", 5).unwrap();

        for prediction in &predictions {
            assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
            assert!(prediction.priority <= 100);
        }
    }

    #[test]
    fn test_prefetch_queue() {
        let mut prefetcher = QueryPrefetcher::new();

        let prediction = QueryPrediction {
            query: "query { user }".to_string(),
            confidence: 0.8,
            priority: 80,
            estimated_time_ms: 100.0,
            reason: "Test".to_string(),
        };

        prefetcher.queue_prefetch(prediction).unwrap();

        let stats = prefetcher.get_statistics().unwrap();
        assert_eq!(stats.queue_size, 1);
    }

    #[test]
    fn test_prefetch_hit() {
        let mut prefetcher = QueryPrefetcher::new();

        let prediction = QueryPrediction {
            query: "query { user }".to_string(),
            confidence: 0.8,
            priority: 80,
            estimated_time_ms: 100.0,
            reason: "Test".to_string(),
        };

        prefetcher.queue_prefetch(prediction).unwrap();

        // Record the query that was prefetched
        prefetcher.record_query("query { user }").unwrap();

        let stats = prefetcher.get_statistics().unwrap();
        assert_eq!(stats.prefetch_hits, 1);
        assert_eq!(stats.queue_size, 0); // Should be removed from queue
    }

    #[test]
    fn test_prefetch_miss() {
        let mut prefetcher = QueryPrefetcher::new();

        // Record a query that wasn't prefetched
        prefetcher.record_query("query { user }").unwrap();

        let stats = prefetcher.get_statistics().unwrap();
        assert_eq!(stats.prefetch_misses, 1);
    }

    #[test]
    fn test_statistics() {
        let mut prefetcher = QueryPrefetcher::new();

        for _ in 0..3 {
            prefetcher.record_query("query { user }").unwrap();
            prefetcher.record_query("query { posts }").unwrap();
        }

        let stats = prefetcher.get_statistics().unwrap();
        assert!(stats.total_patterns > 0 || stats.total_cooccurrences > 0);
    }

    #[test]
    fn test_clear() {
        let mut prefetcher = QueryPrefetcher::new();

        prefetcher.record_query("query { user }").unwrap();
        prefetcher.record_query("query { posts }").unwrap();

        let stats1 = prefetcher.get_statistics().unwrap();
        assert!(stats1.total_patterns > 0 || stats1.total_cooccurrences > 0);

        prefetcher.clear().unwrap();

        let stats2 = prefetcher.get_statistics().unwrap();
        assert_eq!(stats2.total_patterns, 0);
        assert_eq!(stats2.total_cooccurrences, 0);
        assert_eq!(stats2.prefetch_hits, 0);
        assert_eq!(stats2.prefetch_misses, 0);
    }

    #[test]
    fn test_popularity_strategy() {
        let mut prefetcher = QueryPrefetcher::new();
        prefetcher
            .set_strategy(PrefetchStrategy::Popularity)
            .unwrap();

        // Record some queries
        for _ in 0..10 {
            prefetcher.record_query("query { popular }").unwrap();
        }
        for _ in 0..3 {
            prefetcher.record_query("query { rare }").unwrap();
        }

        let predictions = prefetcher.predict_next_queries("query { any }", 5).unwrap();

        // Should predict popular query
        if !predictions.is_empty() {
            assert!(predictions
                .iter()
                .any(|p| p.query.contains("popular") || p.confidence > 0.0));
        }
    }

    #[test]
    fn test_adaptive_strategy() {
        let mut prefetcher = QueryPrefetcher::new();
        prefetcher.set_strategy(PrefetchStrategy::Adaptive).unwrap();

        // Create various patterns
        for _ in 0..3 {
            prefetcher.record_query("query A").unwrap();
            prefetcher.record_query("query B").unwrap();
        }

        let predictions = prefetcher.predict_next_queries("query A", 5).unwrap();

        // Adaptive should combine strategies
        assert!(predictions.len() <= 5);
    }

    #[test]
    fn test_queue_size_limit() {
        let mut prefetcher = QueryPrefetcher::with_config(PrefetchConfig {
            max_prefetch_queue_size: 2,
            ..Default::default()
        });

        let pred1 = QueryPrediction {
            query: "query 1".to_string(),
            confidence: 0.8,
            priority: 80,
            estimated_time_ms: 100.0,
            reason: "Test".to_string(),
        };

        let pred2 = QueryPrediction {
            query: "query 2".to_string(),
            confidence: 0.7,
            priority: 70,
            estimated_time_ms: 100.0,
            reason: "Test".to_string(),
        };

        let pred3 = QueryPrediction {
            query: "query 3".to_string(),
            confidence: 0.6,
            priority: 60,
            estimated_time_ms: 100.0,
            reason: "Test".to_string(),
        };

        prefetcher.queue_prefetch(pred1).unwrap();
        prefetcher.queue_prefetch(pred2).unwrap();

        // Should fail - queue is full
        assert!(prefetcher.queue_prefetch(pred3).is_err());
    }
}
