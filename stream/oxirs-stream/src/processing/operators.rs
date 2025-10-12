//! Advanced Stream Operators
//!
//! This module provides functional stream processing operators:
//! - map: Transform events
//! - filter: Select events based on predicate
//! - flatMap: Transform and flatten events
//! - partition: Split streams based on criteria
//! - reduce: Aggregate events
//! - scan: Stateful transformation
//! - distinct: Remove duplicates
//! - throttle: Rate limiting
//! - debounce: Event coalescing

use crate::StreamEvent;
use anyhow::Result;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

/// Stream operator trait for composable transformations
#[async_trait::async_trait]
pub trait StreamOperator: Send + Sync {
    /// Apply operator to an event
    async fn apply(&mut self, event: StreamEvent) -> Result<Vec<StreamEvent>>;

    /// Get operator statistics
    fn stats(&self) -> OperatorStats;

    /// Reset operator state
    fn reset(&mut self);
}

/// Operator statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct OperatorStats {
    pub events_processed: u64,
    pub events_emitted: u64,
    pub events_filtered: u64,
    pub processing_time_ms: f64,
    pub errors: u64,
}

/// Map operator - transforms each event
pub struct MapOperator<F>
where
    F: Fn(StreamEvent) -> Result<StreamEvent> + Send + Sync,
{
    transform: Arc<F>,
    stats: OperatorStats,
}

impl<F> MapOperator<F>
where
    F: Fn(StreamEvent) -> Result<StreamEvent> + Send + Sync,
{
    pub fn new(transform: F) -> Self {
        Self {
            transform: Arc::new(transform),
            stats: OperatorStats::default(),
        }
    }
}

#[async_trait::async_trait]
impl<F> StreamOperator for MapOperator<F>
where
    F: Fn(StreamEvent) -> Result<StreamEvent> + Send + Sync,
{
    async fn apply(&mut self, event: StreamEvent) -> Result<Vec<StreamEvent>> {
        let start = std::time::Instant::now();

        self.stats.events_processed += 1;

        match (self.transform)(event) {
            Ok(transformed) => {
                self.stats.events_emitted += 1;
                self.stats.processing_time_ms += start.elapsed().as_secs_f64() * 1000.0;
                Ok(vec![transformed])
            }
            Err(e) => {
                self.stats.errors += 1;
                Err(e)
            }
        }
    }

    fn stats(&self) -> OperatorStats {
        self.stats.clone()
    }

    fn reset(&mut self) {
        self.stats = OperatorStats::default();
    }
}

/// Filter operator - selects events based on predicate
pub struct FilterOperator<F>
where
    F: Fn(&StreamEvent) -> bool + Send + Sync,
{
    predicate: Arc<F>,
    stats: OperatorStats,
}

impl<F> FilterOperator<F>
where
    F: Fn(&StreamEvent) -> bool + Send + Sync,
{
    pub fn new(predicate: F) -> Self {
        Self {
            predicate: Arc::new(predicate),
            stats: OperatorStats::default(),
        }
    }
}

#[async_trait::async_trait]
impl<F> StreamOperator for FilterOperator<F>
where
    F: Fn(&StreamEvent) -> bool + Send + Sync,
{
    async fn apply(&mut self, event: StreamEvent) -> Result<Vec<StreamEvent>> {
        let start = std::time::Instant::now();

        self.stats.events_processed += 1;

        if (self.predicate)(&event) {
            self.stats.events_emitted += 1;
            self.stats.processing_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            Ok(vec![event])
        } else {
            self.stats.events_filtered += 1;
            self.stats.processing_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            Ok(vec![])
        }
    }

    fn stats(&self) -> OperatorStats {
        self.stats.clone()
    }

    fn reset(&mut self) {
        self.stats = OperatorStats::default();
    }
}

/// FlatMap operator - transforms and flattens events
pub struct FlatMapOperator<F>
where
    F: Fn(StreamEvent) -> Result<Vec<StreamEvent>> + Send + Sync,
{
    transform: Arc<F>,
    stats: OperatorStats,
}

impl<F> FlatMapOperator<F>
where
    F: Fn(StreamEvent) -> Result<Vec<StreamEvent>> + Send + Sync,
{
    pub fn new(transform: F) -> Self {
        Self {
            transform: Arc::new(transform),
            stats: OperatorStats::default(),
        }
    }
}

#[async_trait::async_trait]
impl<F> StreamOperator for FlatMapOperator<F>
where
    F: Fn(StreamEvent) -> Result<Vec<StreamEvent>> + Send + Sync,
{
    async fn apply(&mut self, event: StreamEvent) -> Result<Vec<StreamEvent>> {
        let start = std::time::Instant::now();

        self.stats.events_processed += 1;

        match (self.transform)(event) {
            Ok(events) => {
                self.stats.events_emitted += events.len() as u64;
                self.stats.processing_time_ms += start.elapsed().as_secs_f64() * 1000.0;
                Ok(events)
            }
            Err(e) => {
                self.stats.errors += 1;
                Err(e)
            }
        }
    }

    fn stats(&self) -> OperatorStats {
        self.stats.clone()
    }

    fn reset(&mut self) {
        self.stats = OperatorStats::default();
    }
}

/// Partition operator - splits stream into multiple partitions
pub struct PartitionOperator<F>
where
    F: Fn(&StreamEvent) -> usize + Send + Sync,
{
    partition_fn: Arc<F>,
    num_partitions: usize,
    partition_buffers: Vec<VecDeque<StreamEvent>>,
    stats: OperatorStats,
}

impl<F> PartitionOperator<F>
where
    F: Fn(&StreamEvent) -> usize + Send + Sync,
{
    pub fn new(partition_fn: F, num_partitions: usize) -> Self {
        Self {
            partition_fn: Arc::new(partition_fn),
            num_partitions,
            partition_buffers: vec![VecDeque::new(); num_partitions],
            stats: OperatorStats::default(),
        }
    }

    pub fn get_partition(&mut self, partition_id: usize) -> Option<Vec<StreamEvent>> {
        if partition_id >= self.num_partitions {
            return None;
        }

        let events: Vec<_> = self.partition_buffers[partition_id].drain(..).collect();
        Some(events)
    }
}

#[async_trait::async_trait]
impl<F> StreamOperator for PartitionOperator<F>
where
    F: Fn(&StreamEvent) -> usize + Send + Sync,
{
    async fn apply(&mut self, event: StreamEvent) -> Result<Vec<StreamEvent>> {
        let start = std::time::Instant::now();

        self.stats.events_processed += 1;

        let partition_id = (self.partition_fn)(&event) % self.num_partitions;
        self.partition_buffers[partition_id].push_back(event.clone());

        self.stats.events_emitted += 1;
        self.stats.processing_time_ms += start.elapsed().as_secs_f64() * 1000.0;

        Ok(vec![event])
    }

    fn stats(&self) -> OperatorStats {
        self.stats.clone()
    }

    fn reset(&mut self) {
        self.stats = OperatorStats::default();
        for buffer in &mut self.partition_buffers {
            buffer.clear();
        }
    }
}

/// Distinct operator - removes duplicate events
pub struct DistinctOperator {
    seen: HashSet<String>,
    key_extractor: Arc<dyn Fn(&StreamEvent) -> String + Send + Sync>,
    stats: OperatorStats,
}

impl DistinctOperator {
    pub fn new<F>(key_extractor: F) -> Self
    where
        F: Fn(&StreamEvent) -> String + Send + Sync + 'static,
    {
        Self {
            seen: HashSet::new(),
            key_extractor: Arc::new(key_extractor),
            stats: OperatorStats::default(),
        }
    }
}

#[async_trait::async_trait]
impl StreamOperator for DistinctOperator {
    async fn apply(&mut self, event: StreamEvent) -> Result<Vec<StreamEvent>> {
        let start = std::time::Instant::now();

        self.stats.events_processed += 1;

        let key = (self.key_extractor)(&event);

        if self.seen.insert(key) {
            self.stats.events_emitted += 1;
            self.stats.processing_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            Ok(vec![event])
        } else {
            self.stats.events_filtered += 1;
            self.stats.processing_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            Ok(vec![])
        }
    }

    fn stats(&self) -> OperatorStats {
        self.stats.clone()
    }

    fn reset(&mut self) {
        self.stats = OperatorStats::default();
        self.seen.clear();
    }
}

/// Throttle operator - rate limiting
pub struct ThrottleOperator {
    interval: ChronoDuration,
    last_emit: Option<DateTime<Utc>>,
    stats: OperatorStats,
}

impl ThrottleOperator {
    pub fn new(interval: ChronoDuration) -> Self {
        Self {
            interval,
            last_emit: None,
            stats: OperatorStats::default(),
        }
    }
}

#[async_trait::async_trait]
impl StreamOperator for ThrottleOperator {
    async fn apply(&mut self, event: StreamEvent) -> Result<Vec<StreamEvent>> {
        let start = std::time::Instant::now();

        self.stats.events_processed += 1;

        let now = Utc::now();

        let should_emit = match self.last_emit {
            None => true,
            Some(last) => now - last >= self.interval,
        };

        if should_emit {
            self.last_emit = Some(now);
            self.stats.events_emitted += 1;
            self.stats.processing_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            Ok(vec![event])
        } else {
            self.stats.events_filtered += 1;
            self.stats.processing_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            Ok(vec![])
        }
    }

    fn stats(&self) -> OperatorStats {
        self.stats.clone()
    }

    fn reset(&mut self) {
        self.stats = OperatorStats::default();
        self.last_emit = None;
    }
}

/// Debounce operator - event coalescing
pub struct DebounceOperator {
    delay: ChronoDuration,
    pending: Option<(StreamEvent, DateTime<Utc>)>,
    stats: OperatorStats,
}

impl DebounceOperator {
    pub fn new(delay: ChronoDuration) -> Self {
        Self {
            delay,
            pending: None,
            stats: OperatorStats::default(),
        }
    }

    pub async fn flush(&mut self) -> Result<Vec<StreamEvent>> {
        if let Some((event, _)) = self.pending.take() {
            self.stats.events_emitted += 1;
            Ok(vec![event])
        } else {
            Ok(vec![])
        }
    }
}

#[async_trait::async_trait]
impl StreamOperator for DebounceOperator {
    async fn apply(&mut self, event: StreamEvent) -> Result<Vec<StreamEvent>> {
        let start = std::time::Instant::now();

        self.stats.events_processed += 1;

        let now = Utc::now();

        // Check if we should emit the pending event
        let mut to_emit = vec![];
        if let Some((pending_event, pending_time)) = &self.pending {
            if now - *pending_time >= self.delay {
                to_emit.push(pending_event.clone());
                self.stats.events_emitted += 1;
            }
        }

        // Update pending event
        self.pending = Some((event, now));

        self.stats.processing_time_ms += start.elapsed().as_secs_f64() * 1000.0;

        Ok(to_emit)
    }

    fn stats(&self) -> OperatorStats {
        self.stats.clone()
    }

    fn reset(&mut self) {
        self.stats = OperatorStats::default();
        self.pending = None;
    }
}

/// Reduce operator - stateful aggregation
pub struct ReduceOperator<F, S>
where
    F: Fn(&mut S, StreamEvent) -> Result<()> + Send + Sync,
    S: Clone + Send + Sync,
{
    reducer: Arc<F>,
    state: S,
    stats: OperatorStats,
}

impl<F, S> ReduceOperator<F, S>
where
    F: Fn(&mut S, StreamEvent) -> Result<()> + Send + Sync,
    S: Clone + Send + Sync,
{
    pub fn new(initial_state: S, reducer: F) -> Self {
        Self {
            reducer: Arc::new(reducer),
            state: initial_state,
            stats: OperatorStats::default(),
        }
    }

    pub fn get_state(&self) -> &S {
        &self.state
    }
}

#[async_trait::async_trait]
impl<F, S> StreamOperator for ReduceOperator<F, S>
where
    F: Fn(&mut S, StreamEvent) -> Result<()> + Send + Sync,
    S: Clone + Send + Sync,
{
    async fn apply(&mut self, event: StreamEvent) -> Result<Vec<StreamEvent>> {
        let start = std::time::Instant::now();

        self.stats.events_processed += 1;

        match (self.reducer)(&mut self.state, event.clone()) {
            Ok(_) => {
                self.stats.events_emitted += 1;
                self.stats.processing_time_ms += start.elapsed().as_secs_f64() * 1000.0;
                Ok(vec![event])
            }
            Err(e) => {
                self.stats.errors += 1;
                Err(e)
            }
        }
    }

    fn stats(&self) -> OperatorStats {
        self.stats.clone()
    }

    fn reset(&mut self) {
        self.stats = OperatorStats::default();
    }
}

/// Stream operator pipeline for chaining operations
pub struct OperatorPipeline {
    operators: Vec<Box<dyn StreamOperator>>,
    stats: PipelineStats,
}

#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    pub total_events_in: u64,
    pub total_events_out: u64,
    pub total_processing_time_ms: f64,
    pub operator_stats: Vec<OperatorStats>,
}

impl OperatorPipeline {
    pub fn new() -> Self {
        Self {
            operators: Vec::new(),
            stats: PipelineStats::default(),
        }
    }

    pub fn add_operator(&mut self, operator: Box<dyn StreamOperator>) {
        self.operators.push(operator);
    }

    pub async fn process(&mut self, event: StreamEvent) -> Result<Vec<StreamEvent>> {
        let start = std::time::Instant::now();

        self.stats.total_events_in += 1;

        let mut current_events = vec![event];

        for operator in &mut self.operators {
            let mut next_events = Vec::new();
            for evt in current_events {
                match operator.apply(evt).await {
                    Ok(mut events) => next_events.append(&mut events),
                    Err(e) => return Err(e),
                }
            }
            current_events = next_events;
        }

        self.stats.total_events_out += current_events.len() as u64;
        self.stats.total_processing_time_ms += start.elapsed().as_secs_f64() * 1000.0;

        Ok(current_events)
    }

    pub fn stats(&self) -> PipelineStats {
        let mut stats = self.stats.clone();
        stats.operator_stats = self.operators.iter().map(|op| op.stats()).collect();
        stats
    }

    pub fn reset(&mut self) {
        self.stats = PipelineStats::default();
        for operator in &mut self.operators {
            operator.reset();
        }
    }
}

impl Default for OperatorPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating operator pipelines fluently
pub struct PipelineBuilder {
    pipeline: OperatorPipeline,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            pipeline: OperatorPipeline::new(),
        }
    }

    pub fn map<F>(mut self, transform: F) -> Self
    where
        F: Fn(StreamEvent) -> Result<StreamEvent> + Send + Sync + 'static,
    {
        self.pipeline
            .add_operator(Box::new(MapOperator::new(transform)));
        self
    }

    pub fn filter<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&StreamEvent) -> bool + Send + Sync + 'static,
    {
        self.pipeline
            .add_operator(Box::new(FilterOperator::new(predicate)));
        self
    }

    pub fn flat_map<F>(mut self, transform: F) -> Self
    where
        F: Fn(StreamEvent) -> Result<Vec<StreamEvent>> + Send + Sync + 'static,
    {
        self.pipeline
            .add_operator(Box::new(FlatMapOperator::new(transform)));
        self
    }

    pub fn distinct<F>(mut self, key_extractor: F) -> Self
    where
        F: Fn(&StreamEvent) -> String + Send + Sync + 'static,
    {
        self.pipeline
            .add_operator(Box::new(DistinctOperator::new(key_extractor)));
        self
    }

    pub fn throttle(mut self, interval: ChronoDuration) -> Self {
        self.pipeline
            .add_operator(Box::new(ThrottleOperator::new(interval)));
        self
    }

    pub fn debounce(mut self, delay: ChronoDuration) -> Self {
        self.pipeline
            .add_operator(Box::new(DebounceOperator::new(delay)));
        self
    }

    pub fn build(self) -> OperatorPipeline {
        self.pipeline
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventMetadata;

    fn create_test_event(subject: &str) -> StreamEvent {
        StreamEvent::TripleAdded {
            subject: subject.to_string(),
            predicate: "test".to_string(),
            object: "value".to_string(),
            graph: None,
            metadata: EventMetadata::default(),
        }
    }

    #[tokio::test]
    async fn test_map_operator() {
        let mut operator = MapOperator::new(|mut event| {
            if let StreamEvent::TripleAdded { ref mut object, .. } = event {
                *object = "transformed".to_string();
            }
            Ok(event)
        });

        let event = create_test_event("test");
        let results = operator.apply(event).await.unwrap();

        assert_eq!(results.len(), 1);
        if let StreamEvent::TripleAdded { object, .. } = &results[0] {
            assert_eq!(object, "transformed");
        }
    }

    #[tokio::test]
    async fn test_filter_operator() {
        let mut operator = FilterOperator::new(|event| {
            if let StreamEvent::TripleAdded { subject, .. } = event {
                subject == "keep"
            } else {
                false
            }
        });

        let event1 = create_test_event("keep");
        let event2 = create_test_event("drop");

        assert_eq!(operator.apply(event1).await.unwrap().len(), 1);
        assert_eq!(operator.apply(event2).await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_pipeline() {
        let mut pipeline = PipelineBuilder::new()
            .filter(|event| {
                if let StreamEvent::TripleAdded { subject, .. } = event {
                    subject.starts_with("test")
                } else {
                    false
                }
            })
            .map(|mut event| {
                if let StreamEvent::TripleAdded { ref mut object, .. } = event {
                    *object = format!("{}_transformed", object);
                }
                Ok(event)
            })
            .build();

        let event = create_test_event("test_subject");
        let results = pipeline.process(event).await.unwrap();

        assert_eq!(results.len(), 1);
        if let StreamEvent::TripleAdded { object, .. } = &results[0] {
            assert_eq!(object, "value_transformed");
        }
    }
}
