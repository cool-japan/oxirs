//! Time-series optimization for temporal RDF
//!
//! This module provides specialized storage for temporal RDF data,
//! optimizing for time-based queries and temporal reasoning.

use crate::model::{Literal, NamedNode, Term, Triple, TriplePattern};
use crate::OxirsError;
use chrono::{DateTime, Duration, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ops::Bound;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Temporal storage configuration
#[derive(Debug, Clone)]
pub struct TemporalConfig {
    /// Base path for temporal data
    pub path: PathBuf,
    /// Time bucket duration
    pub bucket_duration: Duration,
    /// Retention policy
    pub retention: RetentionPolicy,
    /// Indexing strategy
    pub indexing: TemporalIndexing,
    /// Enable temporal compression
    pub compression: bool,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        TemporalConfig {
            path: PathBuf::from("/var/oxirs/temporal"),
            bucket_duration: Duration::hours(1),
            retention: RetentionPolicy::Days(365),
            indexing: TemporalIndexing::default(),
            compression: true,
        }
    }
}

/// Retention policy for temporal data
#[derive(Clone)]
pub enum RetentionPolicy {
    /// Keep data forever
    Forever,
    /// Keep data for N days
    Days(u32),
    /// Keep data for N months
    Months(u32),
    /// Keep last N versions
    Versions(u32),
    /// Custom policy function
    Custom(Arc<dyn Fn(&TemporalTriple) -> bool + Send + Sync>),
}

impl std::fmt::Debug for RetentionPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RetentionPolicy::Forever => write!(f, "Forever"),
            RetentionPolicy::Days(n) => write!(f, "Days({})", n),
            RetentionPolicy::Months(n) => write!(f, "Months({})", n),
            RetentionPolicy::Versions(n) => write!(f, "Versions({})", n),
            RetentionPolicy::Custom(_) => write!(f, "Custom(<function>)"),
        }
    }
}

/// Temporal indexing strategy
#[derive(Debug, Clone)]
pub struct TemporalIndexing {
    /// Index by time intervals
    pub interval_index: bool,
    /// Index by entity history
    pub entity_index: bool,
    /// Index by change events
    pub change_index: bool,
    /// Enable Allen interval relations
    pub allen_relations: bool,
}

impl Default for TemporalIndexing {
    fn default() -> Self {
        TemporalIndexing {
            interval_index: true,
            entity_index: true,
            change_index: true,
            allen_relations: false,
        }
    }
}

/// Temporal triple with time metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalTriple {
    /// The base triple
    pub triple: Triple,
    /// Valid time start
    pub valid_from: DateTime<Utc>,
    /// Valid time end (None means currently valid)
    pub valid_to: Option<DateTime<Utc>>,
    /// Transaction time
    pub transaction_time: DateTime<Utc>,
    /// Additional temporal metadata
    pub metadata: TemporalMetadata,
}

/// Temporal metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalMetadata {
    /// Certainty factor (0.0 - 1.0)
    pub certainty: Option<f64>,
    /// Provenance information
    pub provenance: Option<String>,
    /// Is this a predicted/inferred value
    pub predicted: bool,
    /// Temporal granularity
    pub granularity: TemporalGranularity,
}

/// Temporal granularity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalGranularity {
    Nanosecond,
    Microsecond,
    Millisecond,
    Second,
    Minute,
    Hour,
    Day,
    Month,
    Year,
}

/// Temporal storage engine
pub struct TemporalStorage {
    config: TemporalConfig,
    /// Time-bucketed storage
    buckets: Arc<RwLock<BTreeMap<DateTime<Utc>, Bucket>>>,
    /// Interval index
    interval_index: Arc<RwLock<IntervalIndex>>,
    /// Entity history index
    entity_index: Arc<RwLock<EntityIndex>>,
    /// Change event index
    change_index: Arc<RwLock<ChangeIndex>>,
    /// Statistics
    stats: Arc<RwLock<TemporalStats>>,
}

/// Time bucket for efficient temporal storage
struct Bucket {
    /// Start time of bucket
    start_time: DateTime<Utc>,
    /// Triples in this bucket
    triples: Vec<TemporalTriple>,
    /// Bucket statistics
    stats: BucketStats,
}

/// Bucket statistics
#[derive(Debug, Default)]
struct BucketStats {
    triple_count: usize,
    compressed_size: Option<usize>,
    last_access: DateTime<Utc>,
}

/// Interval index for temporal queries
struct IntervalIndex {
    /// Interval tree for efficient range queries
    intervals: IntervalTree<DateTime<Utc>, TemporalTriple>,
}

/// Entity history index
struct EntityIndex {
    /// Entity URI to history mapping
    entity_history: HashMap<String, EntityHistory>,
}

/// Entity history
#[derive(Debug, Clone)]
struct EntityHistory {
    /// Chronological list of states
    states: BTreeMap<DateTime<Utc>, EntityState>,
    /// Change events
    changes: Vec<ChangeEvent>,
}

/// Entity state at a point in time
#[derive(Debug, Clone)]
struct EntityState {
    /// Properties at this time
    properties: HashMap<String, Vec<Literal>>,
    /// Relationships at this time
    relationships: HashMap<String, Vec<String>>,
}

/// Change event
#[derive(Debug, Clone)]
struct ChangeEvent {
    /// Time of change
    timestamp: DateTime<Utc>,
    /// Type of change
    change_type: ChangeType,
    /// Changed property/relationship
    property: String,
    /// Old value
    old_value: Option<Term>,
    /// New value
    new_value: Option<Term>,
}

/// Type of change
#[derive(Debug, Clone)]
enum ChangeType {
    Insert,
    Update,
    Delete,
}

/// Change event index
struct ChangeIndex {
    /// Recent changes queue
    recent_changes: VecDeque<ChangeEvent>,
    /// Changes by property
    property_changes: HashMap<String, Vec<ChangeEvent>>,
}

/// Temporal statistics
#[derive(Debug, Default)]
struct TemporalStats {
    total_triples: u64,
    active_triples: u64,
    historical_triples: u64,
    total_buckets: u64,
    compression_ratio: f64,
    avg_query_time_ms: f64,
}

/// Interval tree for temporal queries (simplified placeholder)
struct IntervalTree<K, V> {
    _key: std::marker::PhantomData<K>,
    _value: std::marker::PhantomData<V>,
}

impl<K, V> IntervalTree<K, V> {
    fn new() -> Self {
        IntervalTree {
            _key: std::marker::PhantomData,
            _value: std::marker::PhantomData,
        }
    }
}

impl TemporalStorage {
    /// Create new temporal storage
    pub async fn new(config: TemporalConfig) -> Result<Self, OxirsError> {
        std::fs::create_dir_all(&config.path)?;

        Ok(TemporalStorage {
            config,
            buckets: Arc::new(RwLock::new(BTreeMap::new())),
            interval_index: Arc::new(RwLock::new(IntervalIndex {
                intervals: IntervalTree::new(),
            })),
            entity_index: Arc::new(RwLock::new(EntityIndex {
                entity_history: HashMap::new(),
            })),
            change_index: Arc::new(RwLock::new(ChangeIndex {
                recent_changes: VecDeque::with_capacity(10000),
                property_changes: HashMap::new(),
            })),
            stats: Arc::new(RwLock::new(TemporalStats::default())),
        })
    }

    /// Store a temporal triple
    pub async fn store_temporal(
        &self,
        triple: Triple,
        valid_from: DateTime<Utc>,
        valid_to: Option<DateTime<Utc>>,
        metadata: Option<TemporalMetadata>,
    ) -> Result<(), OxirsError> {
        let temporal_triple = TemporalTriple {
            triple: triple.clone(),
            valid_from,
            valid_to,
            transaction_time: Utc::now(),
            metadata: metadata.unwrap_or_else(|| TemporalMetadata {
                certainty: None,
                provenance: None,
                predicted: false,
                granularity: TemporalGranularity::Second,
            }),
        };

        // Determine bucket
        let bucket_time = self.get_bucket_time(valid_from);

        // Store in bucket
        {
            let mut buckets = self.buckets.write().await;
            let bucket = buckets.entry(bucket_time).or_insert_with(|| Bucket {
                start_time: bucket_time,
                triples: Vec::new(),
                stats: BucketStats::default(),
            });

            bucket.triples.push(temporal_triple.clone());
            bucket.stats.triple_count += 1;
            bucket.stats.last_access = Utc::now();
        }

        // Update indexes
        if self.config.indexing.entity_index {
            self.update_entity_index(&temporal_triple).await?;
        }

        if self.config.indexing.change_index {
            self.update_change_index(&temporal_triple).await?;
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_triples += 1;
        if valid_to.is_none() {
            stats.active_triples += 1;
        } else {
            stats.historical_triples += 1;
        }

        Ok(())
    }

    /// Query triples at a specific time point
    pub async fn query_at_time(
        &self,
        pattern: &TriplePattern,
        time: DateTime<Utc>,
    ) -> Result<Vec<Triple>, OxirsError> {
        let mut results = Vec::new();

        // Search relevant buckets
        let buckets = self.buckets.read().await;
        for (_, bucket) in buckets.iter() {
            for temporal in &bucket.triples {
                // Check if triple is valid at the given time
                if temporal.valid_from <= time {
                    if let Some(valid_to) = temporal.valid_to {
                        if valid_to < time {
                            continue;
                        }
                    }

                    // Check if pattern matches
                    if pattern.matches(&temporal.triple) {
                        results.push(temporal.triple.clone());
                    }
                }
            }
        }

        Ok(results)
    }

    /// Query triples within a time range
    pub async fn query_time_range(
        &self,
        pattern: &TriplePattern,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<TemporalTriple>, OxirsError> {
        let mut results = Vec::new();

        // Get relevant buckets
        let start_bucket = self.get_bucket_time(start);
        let end_bucket = self.get_bucket_time(end);

        let buckets = self.buckets.read().await;
        let range = buckets.range((Bound::Included(start_bucket), Bound::Included(end_bucket)));

        for (_, bucket) in range {
            for temporal in &bucket.triples {
                // Check temporal overlap
                if temporal.valid_from <= end {
                    if let Some(valid_to) = temporal.valid_to {
                        if valid_to < start {
                            continue;
                        }
                    }

                    // Check pattern match
                    if pattern.matches(&temporal.triple) {
                        results.push(temporal.clone());
                    }
                }
            }
        }

        Ok(results)
    }

    /// Get entity history
    pub async fn get_entity_history(
        &self,
        entity_uri: &str,
    ) -> Result<Option<EntityHistory>, OxirsError> {
        let entity_index = self.entity_index.read().await;
        Ok(entity_index.entity_history.get(entity_uri).cloned())
    }

    /// Get recent changes
    pub async fn get_recent_changes(&self, limit: usize) -> Result<Vec<ChangeEvent>, OxirsError> {
        let change_index = self.change_index.read().await;
        Ok(change_index
            .recent_changes
            .iter()
            .take(limit)
            .cloned()
            .collect())
    }

    /// Perform temporal reasoning
    pub async fn temporal_reason(
        &self,
        query: TemporalQuery,
    ) -> Result<TemporalResult, OxirsError> {
        match query {
            TemporalQuery::AllenRelation {
                triple1,
                triple2,
                relation,
            } => {
                // Implement Allen's interval algebra
                Ok(TemporalResult::Boolean(false)) // Placeholder
            }
            TemporalQuery::TemporalPath {
                start,
                end,
                predicate,
                max_hops,
            } => {
                // Find temporal paths between entities
                Ok(TemporalResult::Paths(Vec::new())) // Placeholder
            }
            TemporalQuery::ChangeDetection {
                entity,
                property,
                threshold,
            } => {
                // Detect significant changes
                Ok(TemporalResult::Changes(Vec::new())) // Placeholder
            }
            TemporalQuery::TrendAnalysis {
                entity,
                property,
                window,
            } => {
                // Analyze trends over time
                Ok(TemporalResult::Trend(TrendData::default())) // Placeholder
            }
        }
    }

    /// Apply retention policy
    pub async fn apply_retention(&self) -> Result<usize, OxirsError> {
        let mut removed = 0;
        let now = Utc::now();

        let mut buckets = self.buckets.write().await;
        let mut to_remove = Vec::new();

        for (bucket_time, bucket) in buckets.iter_mut() {
            match &self.config.retention {
                RetentionPolicy::Days(days) => {
                    let cutoff = now - Duration::days(*days as i64);
                    if *bucket_time < cutoff {
                        to_remove.push(*bucket_time);
                        removed += bucket.triples.len();
                    }
                }
                RetentionPolicy::Months(months) => {
                    let cutoff = now - Duration::days((*months as i64) * 30);
                    if *bucket_time < cutoff {
                        to_remove.push(*bucket_time);
                        removed += bucket.triples.len();
                    }
                }
                _ => {} // Other policies not implemented in this example
            }
        }

        for bucket_time in to_remove {
            buckets.remove(&bucket_time);
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_triples = stats.total_triples.saturating_sub(removed as u64);

        Ok(removed)
    }

    /// Get bucket time for a given timestamp
    fn get_bucket_time(&self, time: DateTime<Utc>) -> DateTime<Utc> {
        let bucket_seconds = self.config.bucket_duration.num_seconds();
        let timestamp = time.timestamp();
        let bucket_timestamp = (timestamp / bucket_seconds) * bucket_seconds;
        DateTime::from_timestamp(bucket_timestamp, 0).unwrap()
    }

    /// Update entity index
    async fn update_entity_index(&self, temporal: &TemporalTriple) -> Result<(), OxirsError> {
        let mut entity_index = self.entity_index.write().await;

        // Extract entity URI from subject
        let entity_uri = match temporal.triple.subject() {
            crate::model::Subject::NamedNode(nn) => nn.as_str().to_string(),
            _ => return Ok(()), // Skip non-URI subjects
        };

        let history = entity_index
            .entity_history
            .entry(entity_uri)
            .or_insert_with(|| EntityHistory {
                states: BTreeMap::new(),
                changes: Vec::new(),
            });

        // Update entity state
        let state = history
            .states
            .entry(temporal.valid_from)
            .or_insert_with(|| EntityState {
                properties: HashMap::new(),
                relationships: HashMap::new(),
            });

        // Add property or relationship
        let predicate_uri = match temporal.triple.predicate() {
            crate::model::Predicate::NamedNode(nn) => nn.as_str(),
            crate::model::Predicate::Variable(v) => v.as_str(),
        };
        match temporal.triple.object() {
            crate::model::Object::Literal(lit) => {
                state
                    .properties
                    .entry(predicate_uri.to_string())
                    .or_insert_with(Vec::new)
                    .push(lit.clone());
            }
            crate::model::Object::NamedNode(nn) => {
                state
                    .relationships
                    .entry(predicate_uri.to_string())
                    .or_insert_with(Vec::new)
                    .push(nn.as_str().to_string());
            }
            _ => {}
        }

        Ok(())
    }

    /// Update change index
    async fn update_change_index(&self, temporal: &TemporalTriple) -> Result<(), OxirsError> {
        let mut change_index = self.change_index.write().await;

        let change = ChangeEvent {
            timestamp: temporal.valid_from,
            change_type: ChangeType::Insert,
            property: match temporal.triple.predicate() {
                crate::model::Predicate::NamedNode(nn) => nn.as_str(),
                crate::model::Predicate::Variable(v) => v.as_str(),
            }
            .to_string(),
            old_value: None,
            new_value: Some(Term::from_object(temporal.triple.object())),
        };

        // Add to recent changes
        change_index.recent_changes.push_front(change.clone());
        if change_index.recent_changes.len() > 10000 {
            change_index.recent_changes.pop_back();
        }

        // Index by property
        change_index
            .property_changes
            .entry(change.property.clone())
            .or_insert_with(Vec::new)
            .push(change);

        Ok(())
    }
}

/// Temporal query types
#[derive(Debug, Clone)]
pub enum TemporalQuery {
    /// Allen interval relation query
    AllenRelation {
        triple1: TemporalTriple,
        triple2: TemporalTriple,
        relation: AllenRelation,
    },
    /// Temporal path query
    TemporalPath {
        start: String,
        end: String,
        predicate: Option<String>,
        max_hops: usize,
    },
    /// Change detection query
    ChangeDetection {
        entity: String,
        property: String,
        threshold: f64,
    },
    /// Trend analysis query
    TrendAnalysis {
        entity: String,
        property: String,
        window: Duration,
    },
}

/// Allen's interval relations
#[derive(Debug, Clone)]
pub enum AllenRelation {
    Before,
    After,
    Meets,
    MetBy,
    Overlaps,
    OverlappedBy,
    Starts,
    StartedBy,
    During,
    Contains,
    Finishes,
    FinishedBy,
    Equals,
}

/// Temporal query result
#[derive(Debug)]
pub enum TemporalResult {
    Boolean(bool),
    Paths(Vec<Vec<TemporalTriple>>),
    Changes(Vec<ChangeEvent>),
    Trend(TrendData),
}

/// Trend analysis data
#[derive(Debug, Default)]
pub struct TrendData {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub predictions: Vec<(DateTime<Utc>, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_temporal_storage() {
        let config = TemporalConfig {
            path: PathBuf::from("/tmp/oxirs_temporal_test"),
            ..Default::default()
        };

        let storage = TemporalStorage::new(config).await.unwrap();

        // Create temporal triple
        let triple = Triple::new(
            NamedNode::new("http://example.org/person1").unwrap(),
            NamedNode::new("http://example.org/age").unwrap(),
            crate::model::Object::Literal(Literal::new("25")),
        );

        let valid_from = Utc::now() - Duration::days(365);
        let valid_to = Some(Utc::now() - Duration::days(180));

        // Store temporal triple
        storage
            .store_temporal(triple.clone(), valid_from, valid_to, None)
            .await
            .unwrap();

        // Query at a time when triple was valid
        let query_time = Utc::now() - Duration::days(270);
        let pattern = TriplePattern::new(
            Some(crate::model::SubjectPattern::NamedNode(
                NamedNode::new("http://example.org/person1").unwrap(),
            )),
            None,
            None,
        );

        let results = storage.query_at_time(&pattern, query_time).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], triple);

        // Query at current time (should be empty as triple is no longer valid)
        let current_results = storage.query_at_time(&pattern, Utc::now()).await.unwrap();
        assert_eq!(current_results.len(), 0);
    }

    #[tokio::test]
    async fn test_entity_history() {
        let config = TemporalConfig {
            path: PathBuf::from("/tmp/oxirs_temporal_history"),
            ..Default::default()
        };

        let storage = TemporalStorage::new(config).await.unwrap();

        let entity = "http://example.org/person1";

        // Store multiple versions of age
        for age in 20..=25 {
            let triple = Triple::new(
                NamedNode::new(entity).unwrap(),
                NamedNode::new("http://example.org/age").unwrap(),
                crate::model::Object::Literal(Literal::new(age.to_string())),
            );

            let valid_from = Utc::now() - Duration::days((26 - age) as i64 * 365);
            storage
                .store_temporal(triple, valid_from, None, None)
                .await
                .unwrap();
        }

        // Get entity history
        let history = storage.get_entity_history(entity).await.unwrap();
        assert!(history.is_some());

        let history = history.unwrap();
        assert_eq!(history.states.len(), 6);
    }
}
