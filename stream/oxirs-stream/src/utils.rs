//! # Stream Utilities
//!
//! Utility functions and helpers for common stream operations.

use crate::event::StreamEvent;
use crate::{Stream, StreamConfig};
use anyhow::Result;
use std::time::Duration;

/// Batch processor for processing multiple events efficiently
pub struct BatchProcessor {
    batch_size: usize,
    timeout: Duration,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(batch_size: usize, timeout: Duration) -> Self {
        Self {
            batch_size,
            timeout,
        }
    }

    /// Process events in batches with a callback
    pub async fn process<F, Fut>(&self, stream: &mut Stream, mut callback: F) -> Result<usize>
    where
        F: FnMut(Vec<StreamEvent>) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        let mut batch = Vec::with_capacity(self.batch_size);
        let mut total_processed = 0;
        let start = tokio::time::Instant::now();

        loop {
            match tokio::time::timeout(self.timeout, stream.consume()).await {
                Ok(Ok(Some(event))) => {
                    batch.push(event);

                    if batch.len() >= self.batch_size {
                        callback(std::mem::take(&mut batch)).await?;
                        total_processed += self.batch_size;
                    }
                }
                Ok(Ok(None)) => {
                    // No more events, process remaining batch
                    if !batch.is_empty() {
                        let count = batch.len();
                        callback(std::mem::take(&mut batch)).await?;
                        total_processed += count;
                    }
                    break;
                }
                Ok(Err(e)) => {
                    return Err(e);
                }
                Err(_) => {
                    // Timeout - process what we have
                    if !batch.is_empty() {
                        let count = batch.len();
                        callback(std::mem::take(&mut batch)).await?;
                        total_processed += count;
                    }

                    // Check if we should continue or stop
                    if start.elapsed() > self.timeout * 2 {
                        break;
                    }
                }
            }
        }

        Ok(total_processed)
    }
}

/// Type alias for event predicate functions
type EventPredicate = Box<dyn Fn(&StreamEvent) -> bool + Send + Sync>;

/// Event filter builder for creating complex event filters
pub struct EventFilter {
    predicates: Vec<EventPredicate>,
}

impl EventFilter {
    /// Create a new event filter
    pub fn new() -> Self {
        Self {
            predicates: Vec::new(),
        }
    }

    /// Add a predicate to the filter
    pub fn add_predicate<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&StreamEvent) -> bool + Send + Sync + 'static,
    {
        self.predicates.push(Box::new(predicate));
        self
    }

    /// Filter events by subject pattern
    pub fn by_subject(self, pattern: String) -> Self {
        self.add_predicate(move |event| match event {
            StreamEvent::TripleAdded { subject, .. } => subject.contains(&pattern),
            StreamEvent::TripleRemoved { subject, .. } => subject.contains(&pattern),
            _ => false,
        })
    }

    /// Filter events by predicate pattern
    pub fn by_predicate(self, pattern: String) -> Self {
        self.add_predicate(move |event| match event {
            StreamEvent::TripleAdded { predicate, .. } => predicate.contains(&pattern),
            StreamEvent::TripleRemoved { predicate, .. } => predicate.contains(&pattern),
            _ => false,
        })
    }

    /// Filter events by graph
    pub fn by_graph(self, graph_name: String) -> Self {
        self.add_predicate(move |event| match event {
            StreamEvent::TripleAdded { graph, .. } => {
                graph.as_ref().is_some_and(|g| g == &graph_name)
            }
            StreamEvent::TripleRemoved { graph, .. } => {
                graph.as_ref().is_some_and(|g| g == &graph_name)
            }
            _ => false,
        })
    }

    /// Test if an event matches all predicates
    pub fn matches(&self, event: &StreamEvent) -> bool {
        self.predicates.iter().all(|predicate| predicate(event))
    }

    /// Filter a batch of events
    pub fn filter_batch(&self, events: Vec<StreamEvent>) -> Vec<StreamEvent> {
        events.into_iter().filter(|e| self.matches(e)).collect()
    }
}

impl Default for EventFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Stream statistics aggregator
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    pub total_events: u64,
    pub events_per_second: f64,
    pub avg_event_size: u64,
    pub total_bytes: u64,
    pub error_count: u64,
    pub start_time: Option<std::time::Instant>,
}

impl StreamStats {
    /// Create a new stream statistics aggregator
    pub fn new() -> Self {
        Self {
            start_time: Some(std::time::Instant::now()),
            ..Default::default()
        }
    }

    /// Record an event
    pub fn record_event(&mut self, event_size: u64) {
        self.total_events += 1;
        self.total_bytes += event_size;

        if let Some(start) = self.start_time {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                self.events_per_second = self.total_events as f64 / elapsed;
            }
        }

        if self.total_events > 0 {
            self.avg_event_size = self.total_bytes / self.total_events;
        }
    }

    /// Record an error
    pub fn record_error(&mut self) {
        self.error_count += 1;
    }

    /// Get the error rate
    pub fn error_rate(&self) -> f64 {
        if self.total_events == 0 {
            return 0.0;
        }
        self.error_count as f64 / self.total_events as f64
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Stream multiplexer for consuming from multiple streams
pub struct StreamMultiplexer {
    streams: Vec<Stream>,
}

impl StreamMultiplexer {
    /// Create a new stream multiplexer
    pub fn new(streams: Vec<Stream>) -> Self {
        Self { streams }
    }

    /// Consume from all streams round-robin
    pub async fn consume_round_robin(&mut self) -> Result<Option<StreamEvent>> {
        for stream in &mut self.streams {
            if let Some(event) = stream.consume().await? {
                return Ok(Some(event));
            }
        }
        Ok(None)
    }

    /// Consume from all streams in parallel and return the first available event
    pub async fn consume_first_available(&mut self) -> Result<Option<StreamEvent>> {
        use futures::future::select_all;

        let futures: Vec<_> = self
            .streams
            .iter_mut()
            .map(|stream| Box::pin(stream.consume()))
            .collect();

        if futures.is_empty() {
            return Ok(None);
        }

        let (result, _index, _remaining) = select_all(futures).await;
        result
    }

    /// Get the number of streams
    pub fn len(&self) -> usize {
        self.streams.len()
    }

    /// Check if the multiplexer is empty
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }
}

/// Helper to create a stream with sensible defaults for development
pub async fn create_dev_stream(topic: &str) -> Result<Stream> {
    let config = StreamConfig::development(topic);
    Stream::new(config).await
}

/// Helper to create a stream with production settings
pub async fn create_prod_stream(topic: &str) -> Result<Stream> {
    let config = StreamConfig::production(topic);
    Stream::new(config).await
}

/// Simple rate limiter for controlling event publishing rate
pub struct SimpleRateLimiter {
    permits_per_second: u64,
    last_refill: tokio::time::Instant,
    available_permits: u64,
}

impl SimpleRateLimiter {
    /// Create a new rate limiter
    pub fn new(permits_per_second: u64) -> Self {
        Self {
            permits_per_second,
            last_refill: tokio::time::Instant::now(),
            available_permits: permits_per_second,
        }
    }

    /// Acquire a permit, blocking if necessary
    pub async fn acquire(&mut self) -> Result<()> {
        loop {
            // Refill permits based on elapsed time
            let now = tokio::time::Instant::now();
            let elapsed = now.duration_since(self.last_refill);
            let new_permits = (elapsed.as_secs_f64() * self.permits_per_second as f64) as u64;

            if new_permits > 0 {
                self.available_permits =
                    (self.available_permits + new_permits).min(self.permits_per_second);
                self.last_refill = now;
            }

            if self.available_permits > 0 {
                self.available_permits -= 1;
                return Ok(());
            }

            // Wait before checking again
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

/// Event sampler for sampling events at a specified rate
pub struct EventSampler {
    sample_rate: f64,
    count: u64,
}

impl EventSampler {
    /// Create a new event sampler
    ///
    /// # Arguments
    /// * `sample_rate` - Fraction of events to keep (0.0 to 1.0)
    pub fn new(sample_rate: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&sample_rate),
            "Sample rate must be between 0 and 1"
        );
        Self {
            sample_rate,
            count: 0,
        }
    }

    /// Check if the current event should be sampled
    pub fn should_sample(&mut self) -> bool {
        self.count += 1;

        if self.sample_rate >= 1.0 {
            return true;
        }

        if self.sample_rate <= 0.0 {
            return false;
        }

        // Deterministic sampling based on count
        (self.count as f64 * self.sample_rate).fract() < self.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_stats() {
        let mut stats = StreamStats::new();

        stats.record_event(100);
        stats.record_event(200);
        stats.record_event(300);

        assert_eq!(stats.total_events, 3);
        assert_eq!(stats.total_bytes, 600);
        assert_eq!(stats.avg_event_size, 200);
    }

    #[test]
    fn test_event_filter() {
        use crate::EventMetadata;
        use std::collections::HashMap;

        let filter = EventFilter::new().by_subject("example.org".to_string());

        let event = StreamEvent::TripleAdded {
            subject: "http://example.org/test".to_string(),
            predicate: "http://example.org/prop".to_string(),
            object: "value".to_string(),
            graph: None,
            metadata: EventMetadata {
                event_id: "test".to_string(),
                timestamp: chrono::Utc::now(),
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        };

        assert!(filter.matches(&event));
    }

    #[test]
    fn test_event_sampler() {
        let mut sampler = EventSampler::new(0.5);

        let mut sampled = 0;
        for _ in 0..1000 {
            if sampler.should_sample() {
                sampled += 1;
            }
        }

        // Should be approximately 500 (50% sampling)
        assert!((450..=550).contains(&sampled), "Sampled {sampled} events");
    }

    #[tokio::test]
    async fn test_simple_rate_limiter() {
        let mut limiter = SimpleRateLimiter::new(10); // 10 permits per second

        let start = tokio::time::Instant::now();

        for _ in 0..5 {
            limiter.acquire().await.unwrap();
        }

        let elapsed = start.elapsed();

        // Should complete almost instantly for 5 permits
        assert!(elapsed < Duration::from_millis(100));
    }
}
