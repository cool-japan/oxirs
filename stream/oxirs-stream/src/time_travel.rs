//! # Time-Travel Query System
//!
//! Advanced temporal query capabilities for OxiRS Stream, enabling querying
//! data at any point in time, temporal analytics, and historical state reconstruction.

use crate::event_sourcing::{EventStoreTrait, EventStream};
use crate::StreamEvent;
use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Type alias for custom filter functions to reduce complexity
pub type CustomFilterFn = Box<dyn Fn(&StreamEvent) -> bool + Send + Sync>;

/// Time-travel query configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeTravelConfig {
    /// Maximum time window for time-travel queries
    pub max_time_window_days: u32,
    /// Enable temporal indexing for faster queries
    pub enable_temporal_indexing: bool,
    /// Temporal index granularity (minutes)
    pub index_granularity_minutes: u32,
    /// Maximum concurrent time-travel queries
    pub max_concurrent_queries: usize,
    /// Query timeout in seconds
    pub query_timeout_seconds: u64,
    /// Enable result caching
    pub enable_result_caching: bool,
    /// Cache TTL in minutes
    pub cache_ttl_minutes: u32,
    /// Maximum cache size in MB
    pub max_cache_size_mb: usize,
}

impl Default for TimeTravelConfig {
    fn default() -> Self {
        Self {
            max_time_window_days: 365,
            enable_temporal_indexing: true,
            index_granularity_minutes: 60,
            max_concurrent_queries: 100,
            query_timeout_seconds: 300,
            enable_result_caching: true,
            cache_ttl_minutes: 60,
            max_cache_size_mb: 1024,
        }
    }
}

/// Time point specification for queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimePoint {
    /// Specific timestamp
    Timestamp(DateTime<Utc>),
    /// Relative time from now
    RelativeTime(ChronoDuration),
    /// Event version number
    Version(u64),
    /// Event ID
    EventId(Uuid),
    /// Named snapshot
    Snapshot(String),
}

/// Time range specification for queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: TimePoint,
    pub end: TimePoint,
}

/// Temporal query specification
#[derive(Debug, Clone)]
pub struct TemporalQuery {
    pub query_id: Uuid,
    pub time_point: Option<TimePoint>,
    pub time_range: Option<TimeRange>,
    pub filter: TemporalFilter,
    pub projection: TemporalProjection,
    pub ordering: TemporalOrdering,
    pub limit: Option<usize>,
}

impl Default for TemporalQuery {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalQuery {
    /// Create a new temporal query
    pub fn new() -> Self {
        Self {
            query_id: Uuid::new_v4(),
            time_point: None,
            time_range: None,
            filter: TemporalFilter::default(),
            projection: TemporalProjection::default(),
            ordering: TemporalOrdering::default(),
            limit: None,
        }
    }

    /// Query at specific time point
    pub fn at_time(mut self, time_point: TimePoint) -> Self {
        self.time_point = Some(time_point);
        self
    }

    /// Query within time range
    pub fn in_range(mut self, time_range: TimeRange) -> Self {
        self.time_range = Some(time_range);
        self
    }

    /// Add filter
    pub fn filter(mut self, filter: TemporalFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Set projection
    pub fn project(mut self, projection: TemporalProjection) -> Self {
        self.projection = projection;
        self
    }

    /// Set ordering
    pub fn order_by(mut self, ordering: TemporalOrdering) -> Self {
        self.ordering = ordering;
        self
    }

    /// Set limit
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

/// Temporal filter for events
#[derive(Default)]
pub struct TemporalFilter {
    pub event_types: Option<HashSet<String>>,
    pub aggregate_ids: Option<HashSet<String>>,
    pub user_ids: Option<HashSet<String>>,
    pub sources: Option<HashSet<String>>,
    pub custom_filters: Vec<CustomFilterFn>,
}

impl std::fmt::Debug for TemporalFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TemporalFilter")
            .field("event_types", &self.event_types)
            .field("aggregate_ids", &self.aggregate_ids)
            .field("user_ids", &self.user_ids)
            .field("sources", &self.sources)
            .field(
                "custom_filters",
                &format!("<{} filters>", self.custom_filters.len()),
            )
            .finish()
    }
}

impl Clone for TemporalFilter {
    fn clone(&self) -> Self {
        Self {
            event_types: self.event_types.clone(),
            aggregate_ids: self.aggregate_ids.clone(),
            user_ids: self.user_ids.clone(),
            sources: self.sources.clone(),
            custom_filters: Vec::new(), // Cannot clone function pointers
        }
    }
}

/// Temporal projection specification
#[derive(Debug, Clone)]
pub enum TemporalProjection {
    /// Return full events
    FullEvents,
    /// Return only metadata
    MetadataOnly,
    /// Return specific fields
    Fields(Vec<String>),
    /// Return aggregated data
    Aggregation(AggregationType),
}

impl Default for TemporalProjection {
    fn default() -> Self {
        Self::FullEvents
    }
}

/// Aggregation type for temporal queries
#[derive(Debug, Clone)]
pub enum AggregationType {
    Count,
    CountBy(String),
    Timeline(ChronoDuration),
    Statistics,
}

/// Temporal ordering specification
#[derive(Debug, Clone)]
pub enum TemporalOrdering {
    /// Order by timestamp ascending
    TimeAscending,
    /// Order by timestamp descending
    TimeDescending,
    /// Order by version ascending
    VersionAscending,
    /// Order by version descending
    VersionDescending,
    /// Order by custom field
    Custom(String, bool), // field, ascending
}

impl Default for TemporalOrdering {
    fn default() -> Self {
        Self::TimeDescending
    }
}

/// Result of a temporal query
#[derive(Debug, Clone)]
pub struct TemporalQueryResult {
    pub query_id: Uuid,
    pub events: Vec<StreamEvent>,
    pub metadata: TemporalResultMetadata,
    pub aggregations: Option<TemporalAggregations>,
    pub execution_time: Duration,
    pub from_cache: bool,
}

/// Metadata about temporal query results
#[derive(Debug, Clone)]
pub struct TemporalResultMetadata {
    pub total_events: usize,
    pub time_range_covered: Option<(DateTime<Utc>, DateTime<Utc>)>,
    pub version_range_covered: Option<(u64, u64)>,
    pub aggregates_scanned: HashSet<String>,
    pub index_hits: usize,
    pub index_misses: usize,
}

/// Aggregated data from temporal queries
#[derive(Debug, Clone)]
pub struct TemporalAggregations {
    pub count: usize,
    pub count_by_type: HashMap<String, usize>,
    pub timeline: Vec<TimelinePoint>,
    pub statistics: TemporalStatistics,
}

/// Point in timeline aggregation
#[derive(Debug, Clone)]
pub struct TimelinePoint {
    pub timestamp: DateTime<Utc>,
    pub count: usize,
    pub event_types: HashMap<String, usize>,
}

/// Statistical data from temporal queries
#[derive(Debug, Clone)]
pub struct TemporalStatistics {
    pub events_per_second: f64,
    pub peak_throughput: f64,
    pub average_event_size: f64,
    pub unique_aggregates: usize,
    pub unique_users: usize,
    pub time_span: ChronoDuration,
}

/// Temporal index for efficient time-travel queries
#[derive(Debug)]
struct TemporalIndex {
    /// Time-based index: timestamp -> event IDs
    time_index: BTreeMap<DateTime<Utc>, Vec<Uuid>>,
    /// Version-based index: version -> event metadata
    version_index: BTreeMap<u64, EventIndexEntry>,
    /// Aggregate-based index: aggregate_id -> time-ordered events
    aggregate_index: HashMap<String, BTreeMap<DateTime<Utc>, Vec<Uuid>>>,
    /// Type-based index: event_type -> time-ordered events
    type_index: HashMap<String, BTreeMap<DateTime<Utc>, Vec<Uuid>>>,
}

#[derive(Debug, Clone)]
struct EventIndexEntry {
    pub event_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub aggregate_id: String,
    pub event_type: String,
    pub version: u64,
}

impl TemporalIndex {
    fn new() -> Self {
        Self {
            time_index: BTreeMap::new(),
            version_index: BTreeMap::new(),
            aggregate_index: HashMap::new(),
            type_index: HashMap::new(),
        }
    }

    fn add_event(&mut self, event: &StreamEvent) {
        let metadata = event.metadata();
        let timestamp = metadata.timestamp;
        let event_id = uuid::Uuid::parse_str(&metadata.event_id).unwrap_or(uuid::Uuid::new_v4());
        let aggregate_id = metadata.context.clone().unwrap_or_default();
        let event_type = format!("{event:?}");
        let version = metadata.version.parse::<u64>().unwrap_or(0);

        // Time index
        self.time_index.entry(timestamp).or_default().push(event_id);

        // Version index
        self.version_index.insert(
            version,
            EventIndexEntry {
                event_id,
                timestamp,
                aggregate_id: aggregate_id.clone(),
                event_type: event_type.clone(),
                version,
            },
        );

        // Aggregate index
        self.aggregate_index
            .entry(aggregate_id)
            .or_default()
            .entry(timestamp)
            .or_default()
            .push(event_id);

        // Type index
        self.type_index
            .entry(event_type)
            .or_default()
            .entry(timestamp)
            .or_default()
            .push(event_id);
    }

    fn find_events_by_time_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<Uuid> {
        let mut event_ids = Vec::new();

        for (_, ids) in self.time_index.range(start..=end) {
            event_ids.extend_from_slice(ids);
        }

        event_ids
    }

    fn find_events_by_version_range(&self, start: u64, end: u64) -> Vec<Uuid> {
        let mut event_ids = Vec::new();

        for (_, entry) in self.version_index.range(start..=end) {
            event_ids.push(entry.event_id);
        }

        event_ids
    }

    fn find_events_by_aggregate(
        &self,
        aggregate_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<Uuid> {
        if let Some(time_map) = self.aggregate_index.get(aggregate_id) {
            let mut event_ids = Vec::new();
            for (_, ids) in time_map.range(start..=end) {
                event_ids.extend_from_slice(ids);
            }
            event_ids
        } else {
            Vec::new()
        }
    }
}

/// Time-travel query engine
pub struct TimeTravelEngine {
    config: TimeTravelConfig,
    event_store: Arc<dyn EventStoreTrait>,
    event_stream: Arc<dyn EventStream>,
    temporal_index: Arc<RwLock<TemporalIndex>>,
    query_cache: Arc<RwLock<QueryCache>>,
    query_semaphore: Arc<tokio::sync::Semaphore>,
    metrics: Arc<RwLock<TimeTravelMetrics>>,
}

impl TimeTravelEngine {
    /// Create a new time-travel engine
    pub fn new(
        config: TimeTravelConfig,
        event_store: Arc<dyn EventStoreTrait>,
        event_stream: Arc<dyn EventStream>,
    ) -> Self {
        Self {
            query_semaphore: Arc::new(tokio::sync::Semaphore::new(config.max_concurrent_queries)),
            temporal_index: Arc::new(RwLock::new(TemporalIndex::new())),
            query_cache: Arc::new(RwLock::new(QueryCache::new(config.clone()))),
            config,
            event_store,
            event_stream,
            metrics: Arc::new(RwLock::new(TimeTravelMetrics::default())),
        }
    }

    /// Start the time-travel engine
    pub async fn start(&self) -> Result<()> {
        info!("Starting time-travel engine");

        // Build initial index if enabled
        if self.config.enable_temporal_indexing {
            self.build_temporal_index().await?;
        }

        // Start index maintenance task
        let index = Arc::clone(&self.temporal_index);
        let event_stream = Arc::clone(&self.event_stream);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                if let Err(e) =
                    Self::update_index(Arc::clone(&index), Arc::clone(&event_stream)).await
                {
                    error!("Failed to update temporal index: {}", e);
                }
            }
        });

        info!("Time-travel engine started successfully");
        Ok(())
    }

    /// Execute a temporal query
    pub async fn execute_query(&self, query: TemporalQuery) -> Result<TemporalQueryResult> {
        let start_time = Instant::now();
        let query_id = query.query_id;

        debug!("Executing temporal query {}", query_id);

        // Acquire semaphore for concurrency control
        let _permit = self.query_semaphore.acquire().await?;

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.queries_executed += 1;
            metrics.active_queries += 1;
        }

        // Check cache first
        let cache_key = self.generate_cache_key(&query);
        if self.config.enable_result_caching {
            let cache = self.query_cache.read().await;
            if let Some(cached_result) = cache.get(&cache_key) {
                let mut metrics = self.metrics.write().await;
                metrics.active_queries -= 1;
                metrics.cache_hits += 1;

                return Ok(TemporalQueryResult {
                    query_id,
                    events: cached_result.events,
                    metadata: cached_result.metadata,
                    aggregations: cached_result.aggregations,
                    execution_time: start_time.elapsed(),
                    from_cache: true,
                });
            }
        }

        let result = self.execute_query_internal(query).await;

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.active_queries -= 1;
            match &result {
                Ok(_) => {
                    metrics.queries_succeeded += 1;
                    if !self.config.enable_result_caching {
                        metrics.cache_misses += 1;
                    }
                }
                Err(_) => metrics.queries_failed += 1,
            }
        }

        let execution_time = start_time.elapsed();
        debug!(
            "Temporal query {} executed in {:?}",
            query_id, execution_time
        );

        if let Ok(ref res) = result {
            // Cache result if applicable
            if self.config.enable_result_caching {
                let mut cache = self.query_cache.write().await;
                cache.set(cache_key, res.clone());
            }
        }

        result.map(|mut r| {
            r.execution_time = execution_time;
            r.from_cache = false;
            r
        })
    }

    /// Execute query internally
    async fn execute_query_internal(&self, query: TemporalQuery) -> Result<TemporalQueryResult> {
        let query_id = query.query_id;

        // Resolve time points to actual timestamps
        let (start_time, end_time) = self.resolve_time_range(&query).await?;

        // Find candidate events
        let candidate_event_ids = if self.config.enable_temporal_indexing {
            self.find_events_with_index(&query, start_time, end_time)
                .await?
        } else {
            self.find_events_without_index(&query, start_time, end_time)
                .await?
        };

        // Load full events
        let mut events = Vec::new();
        for event_id in candidate_event_ids {
            if let Some(event) = self.load_event(event_id).await? {
                if self.matches_filter(&event, &query.filter) {
                    events.push(event);
                }
            }
        }

        // Apply ordering
        self.apply_ordering(&mut events, &query.ordering);

        // Apply limit
        if let Some(limit) = query.limit {
            events.truncate(limit);
        }

        // Generate metadata
        let metadata = self.generate_result_metadata(&events, start_time, end_time);

        // Generate aggregations if requested
        let aggregations = match query.projection {
            TemporalProjection::Aggregation(ref agg_type) => {
                Some(self.generate_aggregations(&events, agg_type, start_time, end_time)?)
            }
            _ => None,
        };

        // Apply projection
        let projected_events = self.apply_projection(events, &query.projection);

        Ok(TemporalQueryResult {
            query_id,
            events: projected_events,
            metadata,
            aggregations,
            execution_time: Duration::default(), // Will be set by caller
            from_cache: false,
        })
    }

    /// Query state at specific time point
    pub async fn query_state_at_time(
        &self,
        aggregate_id: &str,
        time_point: TimePoint,
    ) -> Result<Vec<StreamEvent>> {
        let query = TemporalQuery::new()
            .at_time(time_point)
            .filter(TemporalFilter {
                aggregate_ids: Some(std::iter::once(aggregate_id.to_string()).collect()),
                ..Default::default()
            });

        let result = self.execute_query(query).await?;
        Ok(result.events)
    }

    /// Query changes between two time points
    pub async fn query_changes_between(
        &self,
        start: TimePoint,
        end: TimePoint,
        filter: Option<TemporalFilter>,
    ) -> Result<Vec<StreamEvent>> {
        let query = TemporalQuery::new()
            .in_range(TimeRange { start, end })
            .filter(filter.unwrap_or_default());

        let result = self.execute_query(query).await?;
        Ok(result.events)
    }

    /// Query timeline aggregation
    pub async fn query_timeline(
        &self,
        time_range: TimeRange,
        granularity: ChronoDuration,
        filter: Option<TemporalFilter>,
    ) -> Result<Vec<TimelinePoint>> {
        let query = TemporalQuery::new()
            .in_range(time_range)
            .filter(filter.unwrap_or_default())
            .project(TemporalProjection::Aggregation(AggregationType::Timeline(
                granularity,
            )));

        let result = self.execute_query(query).await?;
        Ok(result.aggregations.map(|a| a.timeline).unwrap_or_default())
    }

    /// Build temporal index from existing events
    async fn build_temporal_index(&self) -> Result<()> {
        info!("Building temporal index");

        let events = self
            .event_stream
            .read_events_from_position(0, usize::MAX)
            .await?;
        let mut index = self.temporal_index.write().await;

        for stored_event in events {
            index.add_event(&stored_event.event_data);
        }

        info!(
            "Temporal index built with {} events",
            index.time_index.len()
        );
        Ok(())
    }

    /// Update index with new events
    async fn update_index(
        index: Arc<RwLock<TemporalIndex>>,
        event_stream: Arc<dyn EventStream>,
    ) -> Result<()> {
        // This would typically track the last processed position
        // For simplicity, we'll just rebuild periodically
        let events = event_stream.read_events_from_position(0, 10000).await?;
        let mut idx = index.write().await;

        for stored_event in events {
            idx.add_event(&stored_event.event_data);
        }

        Ok(())
    }

    /// Resolve time range from query specification
    async fn resolve_time_range(
        &self,
        query: &TemporalQuery,
    ) -> Result<(DateTime<Utc>, DateTime<Utc>)> {
        let now = Utc::now();

        match (&query.time_point, &query.time_range) {
            (Some(time_point), None) => {
                let timestamp = self.resolve_time_point(time_point).await?;
                Ok((timestamp, timestamp))
            }
            (None, Some(time_range)) => {
                let start = self.resolve_time_point(&time_range.start).await?;
                let end = self.resolve_time_point(&time_range.end).await?;
                Ok((start, end))
            }
            (None, None) => {
                // Default to last 24 hours
                let start = now - ChronoDuration::hours(24);
                Ok((start, now))
            }
            (Some(_), Some(_)) => Err(anyhow!("Cannot specify both time_point and time_range")),
        }
    }

    /// Resolve a time point to an actual timestamp
    async fn resolve_time_point(&self, time_point: &TimePoint) -> Result<DateTime<Utc>> {
        match time_point {
            TimePoint::Timestamp(timestamp) => Ok(*timestamp),
            TimePoint::RelativeTime(duration) => Ok(Utc::now() + *duration),
            TimePoint::Version(version) => {
                // Find timestamp for this version
                let index = self.temporal_index.read().await;
                if let Some(entry) = index.version_index.get(version) {
                    Ok(entry.timestamp)
                } else {
                    Err(anyhow!("Version {} not found", version))
                }
            }
            TimePoint::EventId(event_id) => {
                // Find timestamp for this event ID
                if let Some(event) = self.load_event(*event_id).await? {
                    Ok(event.metadata().timestamp)
                } else {
                    Err(anyhow!("Event {} not found", event_id))
                }
            }
            TimePoint::Snapshot(name) => {
                // This would integrate with snapshot store
                Err(anyhow!("Snapshot resolution not implemented: {}", name))
            }
        }
    }

    /// Find events using temporal index
    async fn find_events_with_index(
        &self,
        query: &TemporalQuery,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<Uuid>> {
        let index = self.temporal_index.read().await;

        // Use most specific index available
        if let Some(ref aggregate_ids) = query.filter.aggregate_ids {
            if aggregate_ids.len() == 1 {
                let aggregate_id = aggregate_ids.iter().next().unwrap();
                return Ok(index.find_events_by_aggregate(aggregate_id, start_time, end_time));
            }
        }

        Ok(index.find_events_by_time_range(start_time, end_time))
    }

    /// Find events without using index (sequential scan)
    async fn find_events_without_index(
        &self,
        _query: &TemporalQuery,
        _start_time: DateTime<Utc>,
        _end_time: DateTime<Utc>,
    ) -> Result<Vec<Uuid>> {
        // This would scan all events in the time range
        // For now, return empty set as this requires event store iteration
        warn!("Sequential scan not implemented, returning empty result");
        Ok(Vec::new())
    }

    /// Load a specific event by ID
    async fn load_event(&self, _event_id: Uuid) -> Result<Option<StreamEvent>> {
        // This would load from event store by ID
        // For now, return None as this requires event store lookup by ID
        Ok(None)
    }

    /// Check if event matches filter
    fn matches_filter(&self, event: &StreamEvent, filter: &TemporalFilter) -> bool {
        let metadata = event.metadata();
        let event_type_str = format!("{event:?}");

        if let Some(ref event_types) = filter.event_types {
            if !event_types.contains(&event_type_str) {
                return false;
            }
        }

        if let Some(ref aggregate_ids) = filter.aggregate_ids {
            if let Some(ref context) = metadata.context {
                if !aggregate_ids.contains(context) {
                    return false;
                }
            } else {
                return false;
            }
        }

        if let Some(ref user_ids) = filter.user_ids {
            if let Some(ref user) = metadata.user {
                if !user_ids.contains(user) {
                    return false;
                }
            } else {
                return false;
            }
        }

        if let Some(ref sources) = filter.sources {
            if !sources.contains(&metadata.source) {
                return false;
            }
        }

        // Apply custom filters
        for custom_filter in &filter.custom_filters {
            if !custom_filter(event) {
                return false;
            }
        }

        true
    }

    /// Apply ordering to events
    fn apply_ordering(&self, events: &mut [StreamEvent], ordering: &TemporalOrdering) {
        match ordering {
            TemporalOrdering::TimeAscending => {
                events.sort_by(|a, b| a.metadata().timestamp.cmp(&b.metadata().timestamp));
            }
            TemporalOrdering::TimeDescending => {
                events.sort_by(|a, b| b.metadata().timestamp.cmp(&a.metadata().timestamp));
            }
            TemporalOrdering::VersionAscending => {
                events.sort_by(|a, b| a.metadata().version.cmp(&b.metadata().version));
            }
            TemporalOrdering::VersionDescending => {
                events.sort_by(|a, b| b.metadata().version.cmp(&a.metadata().version));
            }
            TemporalOrdering::Custom(_field, _ascending) => {
                // Custom field ordering would be implemented here
                warn!("Custom ordering not implemented");
            }
        }
    }

    /// Apply projection to events
    fn apply_projection(
        &self,
        events: Vec<StreamEvent>,
        projection: &TemporalProjection,
    ) -> Vec<StreamEvent> {
        match projection {
            TemporalProjection::FullEvents => events,
            TemporalProjection::MetadataOnly => {
                // Return events with only metadata (simplified data)
                // For metadata-only projection, we keep the event but could filter data in a real implementation
                events
            }
            TemporalProjection::Fields(_fields) => {
                // Field projection would be implemented here
                warn!("Field projection not implemented");
                events
            }
            TemporalProjection::Aggregation(_) => {
                // Aggregation results are handled separately
                Vec::new()
            }
        }
    }

    /// Generate result metadata
    fn generate_result_metadata(
        &self,
        events: &[StreamEvent],
        _start_time: DateTime<Utc>,
        _end_time: DateTime<Utc>,
    ) -> TemporalResultMetadata {
        let total_events = events.len();

        let time_range_covered = if !events.is_empty() {
            let min_time = events.iter().map(|e| e.metadata().timestamp).min().unwrap();
            let max_time = events.iter().map(|e| e.metadata().timestamp).max().unwrap();
            Some((min_time, max_time))
        } else {
            None
        };

        let version_range_covered = if !events.is_empty() {
            let min_version = events
                .iter()
                .filter_map(|e| e.metadata().version.parse::<u64>().ok())
                .min();
            let max_version = events
                .iter()
                .filter_map(|e| e.metadata().version.parse::<u64>().ok())
                .max();
            if let (Some(min), Some(max)) = (min_version, max_version) {
                Some((min, max))
            } else {
                None
            }
        } else {
            None
        };

        let aggregates_scanned: HashSet<String> = events
            .iter()
            .filter_map(|e| e.metadata().context.clone())
            .collect();

        TemporalResultMetadata {
            total_events,
            time_range_covered,
            version_range_covered,
            aggregates_scanned,
            index_hits: 0, // Would be tracked during execution
            index_misses: 0,
        }
    }

    /// Generate aggregations
    fn generate_aggregations(
        &self,
        events: &[StreamEvent],
        agg_type: &AggregationType,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<TemporalAggregations> {
        match agg_type {
            AggregationType::Count => Ok(TemporalAggregations {
                count: events.len(),
                count_by_type: HashMap::new(),
                timeline: Vec::new(),
                statistics: self.calculate_statistics(events, start_time, end_time),
            }),
            AggregationType::CountBy(field) => {
                let mut count_by_type = HashMap::new();
                for event in events {
                    if field == "event_type" {
                        let event_type = format!("{event:?}");
                        *count_by_type.entry(event_type).or_insert(0) += 1;
                    }
                    // Other fields would be handled here
                }

                Ok(TemporalAggregations {
                    count: events.len(),
                    count_by_type,
                    timeline: Vec::new(),
                    statistics: self.calculate_statistics(events, start_time, end_time),
                })
            }
            AggregationType::Timeline(granularity) => {
                let timeline = self.generate_timeline(events, *granularity, start_time, end_time);

                Ok(TemporalAggregations {
                    count: events.len(),
                    count_by_type: HashMap::new(),
                    timeline,
                    statistics: self.calculate_statistics(events, start_time, end_time),
                })
            }
            AggregationType::Statistics => Ok(TemporalAggregations {
                count: events.len(),
                count_by_type: HashMap::new(),
                timeline: Vec::new(),
                statistics: self.calculate_statistics(events, start_time, end_time),
            }),
        }
    }

    /// Generate timeline aggregation
    fn generate_timeline(
        &self,
        events: &[StreamEvent],
        granularity: ChronoDuration,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Vec<TimelinePoint> {
        let mut timeline = Vec::new();
        let mut current_time = start_time;

        while current_time < end_time {
            let window_end = current_time + granularity;

            let events_in_window: Vec<_> = events
                .iter()
                .filter(|e| {
                    e.metadata().timestamp >= current_time && e.metadata().timestamp < window_end
                })
                .collect();

            let mut event_types = HashMap::new();
            for event in &events_in_window {
                let event_type = format!("{event:?}");
                *event_types.entry(event_type).or_insert(0) += 1;
            }

            timeline.push(TimelinePoint {
                timestamp: current_time,
                count: events_in_window.len(),
                event_types,
            });

            current_time = window_end;
        }

        timeline
    }

    /// Calculate temporal statistics
    fn calculate_statistics(
        &self,
        events: &[StreamEvent],
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> TemporalStatistics {
        let time_span = end_time.signed_duration_since(start_time);
        let events_per_second = if time_span.num_seconds() > 0 {
            events.len() as f64 / time_span.num_seconds() as f64
        } else {
            0.0
        };

        // Calculate peak throughput (events per second in busiest minute)
        let peak_throughput = if !events.is_empty() {
            let mut minute_counts = HashMap::new();
            for event in events {
                let minute = event
                    .metadata()
                    .timestamp
                    .format("%Y-%m-%d %H:%M")
                    .to_string();
                *minute_counts.entry(minute).or_insert(0) += 1;
            }
            minute_counts.values().max().copied().unwrap_or(0) as f64
        } else {
            0.0
        };

        // Calculate average event size
        let total_size: usize = events.iter().map(|e| format!("{e:?}").len()).sum();
        let average_event_size = if !events.is_empty() {
            total_size as f64 / events.len() as f64
        } else {
            0.0
        };

        let unique_aggregates = events
            .iter()
            .filter_map(|e| e.metadata().context.as_ref())
            .collect::<HashSet<_>>()
            .len();

        let unique_users = events
            .iter()
            .filter_map(|e| e.metadata().user.as_ref())
            .collect::<HashSet<_>>()
            .len();

        TemporalStatistics {
            events_per_second,
            peak_throughput,
            average_event_size,
            unique_aggregates,
            unique_users,
            time_span,
        }
    }

    /// Generate cache key for query
    fn generate_cache_key(&self, query: &TemporalQuery) -> String {
        // Simple cache key based on query structure
        format!("temporal_query_{:?}", query.query_id)
    }

    /// Get time-travel metrics
    pub async fn get_metrics(&self) -> TimeTravelMetrics {
        self.metrics.read().await.clone()
    }
}

/// Query cache for temporal queries
#[derive(Debug)]
struct QueryCache {
    config: TimeTravelConfig,
    entries: HashMap<String, CachedResult>,
}

#[derive(Debug, Clone)]
struct CachedResult {
    events: Vec<StreamEvent>,
    metadata: TemporalResultMetadata,
    aggregations: Option<TemporalAggregations>,
    cached_at: DateTime<Utc>,
}

impl QueryCache {
    fn new(config: TimeTravelConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
        }
    }

    fn get(&self, key: &str) -> Option<CachedResult> {
        if let Some(entry) = self.entries.get(key) {
            let age = Utc::now().signed_duration_since(entry.cached_at);
            if age.num_minutes() < self.config.cache_ttl_minutes as i64 {
                return Some(entry.clone());
            }
        }
        None
    }

    fn set(&mut self, key: String, result: TemporalQueryResult) {
        let entry = CachedResult {
            events: result.events,
            metadata: result.metadata,
            aggregations: result.aggregations,
            cached_at: Utc::now(),
        };

        self.entries.insert(key, entry);
        self.evict_if_needed();
    }

    fn evict_if_needed(&mut self) {
        // Remove expired entries
        let now = Utc::now();
        self.entries.retain(|_, entry| {
            let age = now.signed_duration_since(entry.cached_at);
            age.num_minutes() < self.config.cache_ttl_minutes as i64
        });

        // Simple memory management (could be more sophisticated)
        while self.entries.len() > 1000 {
            if let Some(oldest_key) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.cached_at)
                .map(|(key, _)| key.clone())
            {
                self.entries.remove(&oldest_key);
            } else {
                break;
            }
        }
    }
}

/// Time-travel engine metrics
#[derive(Debug, Clone, Default)]
pub struct TimeTravelMetrics {
    pub queries_executed: u64,
    pub queries_succeeded: u64,
    pub queries_failed: u64,
    pub active_queries: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub index_hits: u64,
    pub index_misses: u64,
    pub average_query_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_time_travel_config_defaults() {
        let config = TimeTravelConfig::default();
        assert_eq!(config.max_time_window_days, 365);
        assert!(config.enable_temporal_indexing);
        assert_eq!(config.index_granularity_minutes, 60);
    }

    #[tokio::test]
    async fn test_temporal_query_builder() {
        let query = TemporalQuery::new()
            .at_time(TimePoint::Timestamp(Utc::now()))
            .filter(TemporalFilter::default())
            .order_by(TemporalOrdering::TimeDescending)
            .limit(100);

        assert!(query.time_point.is_some());
        assert!(query.limit.is_some());
        assert_eq!(query.limit.unwrap(), 100);
    }

    #[tokio::test]
    async fn test_time_point_resolution() {
        let now = Utc::now();
        let relative = TimePoint::RelativeTime(ChronoDuration::hours(-1));

        match relative {
            TimePoint::RelativeTime(duration) => {
                let resolved = now + duration;
                assert!(resolved < now);
            }
            _ => panic!("Expected RelativeTime"),
        }
    }

    #[tokio::test]
    async fn test_temporal_filter() {
        let filter = TemporalFilter {
            event_types: Some(std::iter::once("TestEvent".to_string()).collect()),
            ..Default::default()
        };

        assert!(filter.event_types.is_some());
        assert!(filter.event_types.as_ref().unwrap().contains("TestEvent"));
    }
}
