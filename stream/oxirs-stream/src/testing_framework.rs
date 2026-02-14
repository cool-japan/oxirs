//! # Stream Testing Framework
//!
//! This module provides a comprehensive testing framework for stream
//! applications, enabling developers to write reliable and maintainable tests.
//!
//! ## Features
//! - Test harness for stream applications
//! - Mock streams and event generators
//! - Time manipulation for testing windows
//! - Assertions for stream output
//! - Performance testing utilities
//! - Test fixtures and builders
//! - Test report generation
//!
//! ## Example
//! ```rust,ignore
//! use oxirs_stream::testing_framework::*;
//!
//! #[tokio::test]
//! async fn test_stream_processing() {
//!     let harness = TestHarness::builder()
//!         .with_mock_clock()
//!         .with_event_generator(EventGenerator::sequential(100))
//!         .build()
//!         .await?;
//!
//!     harness.push_events(vec![/* events */]).await;
//!     harness.advance_time(Duration::from_secs(60)).await;
//!
//!     assert_stream_output!(harness, contains(expected));
//! }
//! ```

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info};
use uuid::Uuid;

use crate::event::{EventMetadata, StreamEvent};

/// Configuration for the test harness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestHarnessConfig {
    /// Use mock clock for time manipulation
    pub use_mock_clock: bool,
    /// Initial mock time
    pub initial_time: Option<DateTime<Utc>>,
    /// Event buffer size
    pub event_buffer_size: usize,
    /// Timeout for assertions
    pub assertion_timeout: Duration,
    /// Enable verbose logging
    pub verbose: bool,
    /// Capture all events for inspection
    pub capture_events: bool,
    /// Maximum events to capture
    pub max_captured_events: usize,
    /// Enable performance metrics
    pub enable_metrics: bool,
}

impl Default for TestHarnessConfig {
    fn default() -> Self {
        Self {
            use_mock_clock: true,
            initial_time: None,
            event_buffer_size: 10000,
            assertion_timeout: Duration::from_secs(10),
            verbose: false,
            capture_events: true,
            max_captured_events: 100000,
            enable_metrics: true,
        }
    }
}

/// Mock clock for time manipulation in tests
pub struct MockClock {
    /// Current time
    current_time: Arc<RwLock<DateTime<Utc>>>,
    /// Time advancement listeners
    listeners: Arc<RwLock<Vec<mpsc::Sender<DateTime<Utc>>>>>,
}

impl MockClock {
    /// Create a new mock clock
    pub fn new(initial_time: DateTime<Utc>) -> Self {
        Self {
            current_time: Arc::new(RwLock::new(initial_time)),
            listeners: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Get current time
    pub async fn now(&self) -> DateTime<Utc> {
        *self.current_time.read().await
    }

    /// Advance time by duration
    pub async fn advance(&self, duration: Duration) {
        let mut time = self.current_time.write().await;
        *time += chrono::Duration::from_std(duration).unwrap_or_default();

        let new_time = *time;
        drop(time);

        // Notify listeners
        let listeners = self.listeners.read().await;
        for listener in listeners.iter() {
            let _ = listener.send(new_time).await;
        }
    }

    /// Set time to specific value
    pub async fn set_time(&self, time: DateTime<Utc>) {
        let mut current = self.current_time.write().await;
        *current = time;

        drop(current);

        let listeners = self.listeners.read().await;
        for listener in listeners.iter() {
            let _ = listener.send(time).await;
        }
    }

    /// Subscribe to time changes
    pub async fn subscribe(&self) -> mpsc::Receiver<DateTime<Utc>> {
        let (tx, rx) = mpsc::channel(100);
        let mut listeners = self.listeners.write().await;
        listeners.push(tx);
        rx
    }
}

/// Event generator for creating test events
pub struct EventGenerator {
    /// Generator type
    generator_type: GeneratorType,
    /// Event counter
    counter: AtomicU64,
    /// Configuration
    config: GeneratorConfig,
}

/// Generator configuration
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Source name for events
    pub source: String,
    /// Event properties template
    pub properties: HashMap<String, String>,
    /// Timestamp increment
    pub timestamp_increment: Duration,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            source: "test_generator".to_string(),
            properties: HashMap::new(),
            timestamp_increment: Duration::from_millis(100),
        }
    }
}

/// Types of event generators
#[derive(Debug, Clone)]
pub enum GeneratorType {
    /// Sequential integer events
    Sequential { start: u64, step: u64 },
    /// Random events
    Random { min: f64, max: f64 },
    /// Cyclic pattern
    Cyclic { pattern: Vec<f64>, index: usize },
    /// Gaussian distribution
    Gaussian { mean: f64, stddev: f64 },
    /// Custom generator function
    Custom,
}

impl EventGenerator {
    /// Create a sequential generator
    pub fn sequential(_count: u64) -> Self {
        Self {
            generator_type: GeneratorType::Sequential { start: 0, step: 1 },
            counter: AtomicU64::new(0),
            config: GeneratorConfig::default(),
        }
    }

    /// Create a random generator
    pub fn random(min: f64, max: f64) -> Self {
        Self {
            generator_type: GeneratorType::Random { min, max },
            counter: AtomicU64::new(0),
            config: GeneratorConfig::default(),
        }
    }

    /// Create a cyclic generator
    pub fn cyclic(pattern: Vec<f64>) -> Self {
        Self {
            generator_type: GeneratorType::Cyclic { pattern, index: 0 },
            counter: AtomicU64::new(0),
            config: GeneratorConfig::default(),
        }
    }

    /// Create a gaussian generator
    pub fn gaussian(mean: f64, stddev: f64) -> Self {
        Self {
            generator_type: GeneratorType::Gaussian { mean, stddev },
            counter: AtomicU64::new(0),
            config: GeneratorConfig::default(),
        }
    }

    /// Set source name
    pub fn with_source(mut self, source: String) -> Self {
        self.config.source = source;
        self
    }

    /// Set properties
    pub fn with_properties(mut self, properties: HashMap<String, String>) -> Self {
        self.config.properties = properties;
        self
    }

    /// Generate next event
    pub fn next_event(&self, timestamp: DateTime<Utc>) -> StreamEvent {
        let count = self.counter.fetch_add(1, Ordering::SeqCst);

        let value = match &self.generator_type {
            GeneratorType::Sequential { start, step } => {
                format!("{}", start + count * step)
            }
            GeneratorType::Random { min, max } => {
                let range = max - min;
                let value = min + (count as f64 % 1000.0) / 1000.0 * range;
                format!("{:.2}", value)
            }
            GeneratorType::Cyclic { pattern, .. } => {
                let index = count as usize % pattern.len();
                format!("{:.2}", pattern[index])
            }
            GeneratorType::Gaussian { mean, stddev } => {
                // Simple approximation
                let value = mean + (count as f64 % 10.0 - 5.0) * stddev / 5.0;
                format!("{:.2}", value)
            }
            GeneratorType::Custom => {
                format!("{}", count)
            }
        };

        let metadata = EventMetadata {
            event_id: Uuid::new_v4().to_string(),
            timestamp,
            source: self.config.source.clone(),
            user: None,
            context: Some(format!("test_event_{}", count)),
            caused_by: None,
            version: "1.0".to_string(),
            properties: self.config.properties.clone(),
            checksum: None,
        };

        // Use TripleAdded as a test event type
        StreamEvent::TripleAdded {
            subject: format!("test:subject_{}", count),
            predicate: "test:predicate".to_string(),
            object: value,
            graph: None,
            metadata,
        }
    }

    /// Generate batch of events
    pub fn generate_batch(&self, count: usize, start_time: DateTime<Utc>) -> Vec<StreamEvent> {
        let mut events = Vec::with_capacity(count);
        let mut time = start_time;

        for _ in 0..count {
            events.push(self.next_event(time));
            time += chrono::Duration::from_std(self.config.timestamp_increment).unwrap_or_default();
        }

        events
    }
}

/// Test harness for stream testing
pub struct TestHarness {
    /// Configuration
    config: TestHarnessConfig,
    /// Mock clock
    clock: Arc<MockClock>,
    /// Event generator
    generator: Option<Arc<EventGenerator>>,
    /// Input events channel
    input_tx: mpsc::Sender<StreamEvent>,
    /// Input events receiver
    input_rx: Arc<RwLock<mpsc::Receiver<StreamEvent>>>,
    /// Output events
    output_events: Arc<RwLock<VecDeque<StreamEvent>>>,
    /// Captured events for inspection
    captured_events: Arc<RwLock<Vec<CapturedEvent>>>,
    /// Test metrics
    metrics: Arc<RwLock<TestMetrics>>,
    /// Assertions
    assertions: Arc<RwLock<Vec<Assertion>>>,
}

/// Captured event with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedEvent {
    /// Original event
    pub event: StreamEvent,
    /// Capture time
    pub captured_at: DateTime<Utc>,
    /// Processing time
    pub processing_time: Option<Duration>,
    /// Source
    pub source: String,
}

/// Test metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TestMetrics {
    /// Total events pushed
    pub events_pushed: u64,
    /// Total events received
    pub events_received: u64,
    /// Total assertions
    pub total_assertions: u64,
    /// Passed assertions
    pub passed_assertions: u64,
    /// Failed assertions
    pub failed_assertions: u64,
    /// Average processing time
    pub avg_processing_time_us: f64,
    /// Max processing time
    pub max_processing_time_us: u64,
    /// Test duration
    pub test_duration: Duration,
    /// Memory usage
    pub memory_usage_bytes: usize,
}

/// Assertion for testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assertion {
    /// Assertion type
    pub assertion_type: AssertionType,
    /// Expected value
    pub expected: String,
    /// Actual value
    pub actual: Option<String>,
    /// Result
    pub passed: bool,
    /// Error message
    pub error_message: Option<String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Assertion types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AssertionType {
    /// Assert event count
    EventCount,
    /// Assert event contains
    Contains,
    /// Assert event order
    Order,
    /// Assert no events
    NoEvents,
    /// Assert event property
    Property,
    /// Assert within duration
    WithinDuration,
    /// Assert performance
    Performance,
    /// Custom assertion
    Custom(String),
}

/// Test harness builder
pub struct TestHarnessBuilder {
    config: TestHarnessConfig,
    generator: Option<EventGenerator>,
}

impl TestHarnessBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: TestHarnessConfig::default(),
            generator: None,
        }
    }

    /// Use mock clock
    pub fn with_mock_clock(mut self) -> Self {
        self.config.use_mock_clock = true;
        self
    }

    /// Set initial time
    pub fn with_initial_time(mut self, time: DateTime<Utc>) -> Self {
        self.config.initial_time = Some(time);
        self
    }

    /// Set event buffer size
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.config.event_buffer_size = size;
        self
    }

    /// Set assertion timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.assertion_timeout = timeout;
        self
    }

    /// Enable verbose logging
    pub fn verbose(mut self) -> Self {
        self.config.verbose = true;
        self
    }

    /// Set event generator
    pub fn with_event_generator(mut self, generator: EventGenerator) -> Self {
        self.generator = Some(generator);
        self
    }

    /// Build the test harness
    pub async fn build(self) -> Result<TestHarness> {
        let initial_time = self.config.initial_time.unwrap_or_else(Utc::now);
        let clock = Arc::new(MockClock::new(initial_time));

        let (input_tx, input_rx) = mpsc::channel(self.config.event_buffer_size);

        let harness = TestHarness {
            config: self.config,
            clock,
            generator: self.generator.map(Arc::new),
            input_tx,
            input_rx: Arc::new(RwLock::new(input_rx)),
            output_events: Arc::new(RwLock::new(VecDeque::new())),
            captured_events: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(TestMetrics::default())),
            assertions: Arc::new(RwLock::new(Vec::new())),
        };

        if harness.config.verbose {
            info!("Test harness created with config: {:?}", harness.config);
        }

        Ok(harness)
    }
}

impl Default for TestHarnessBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TestHarness {
    /// Create a new test harness builder
    pub fn builder() -> TestHarnessBuilder {
        TestHarnessBuilder::new()
    }

    /// Get current mock time
    pub async fn now(&self) -> DateTime<Utc> {
        self.clock.now().await
    }

    /// Advance mock time
    pub async fn advance_time(&self, duration: Duration) {
        if self.config.verbose {
            debug!("Advancing time by {:?}", duration);
        }
        self.clock.advance(duration).await;
    }

    /// Set mock time
    pub async fn set_time(&self, time: DateTime<Utc>) {
        if self.config.verbose {
            debug!("Setting time to {:?}", time);
        }
        self.clock.set_time(time).await;
    }

    /// Push a single event
    pub async fn push_event(&self, event: StreamEvent) -> Result<()> {
        self.input_tx
            .send(event.clone())
            .await
            .map_err(|e| anyhow!("Failed to push event: {}", e))?;

        if self.config.capture_events {
            let mut captured = self.captured_events.write().await;
            if captured.len() < self.config.max_captured_events {
                captured.push(CapturedEvent {
                    event,
                    captured_at: self.clock.now().await,
                    processing_time: None,
                    source: "input".to_string(),
                });
            }
        }

        let mut metrics = self.metrics.write().await;
        metrics.events_pushed += 1;

        Ok(())
    }

    /// Push multiple events
    pub async fn push_events(&self, events: Vec<StreamEvent>) -> Result<()> {
        for event in events {
            self.push_event(event).await?;
        }
        Ok(())
    }

    /// Generate and push events
    pub async fn generate_events(&self, count: usize) -> Result<()> {
        if let Some(generator) = &self.generator {
            let time = self.clock.now().await;
            let events = generator.generate_batch(count, time);
            self.push_events(events).await
        } else {
            Err(anyhow!("No event generator configured"))
        }
    }

    /// Add output event (called by stream processor)
    pub async fn add_output(&self, event: StreamEvent) {
        let mut output = self.output_events.write().await;
        output.push_back(event.clone());

        if self.config.capture_events {
            let mut captured = self.captured_events.write().await;
            if captured.len() < self.config.max_captured_events {
                captured.push(CapturedEvent {
                    event,
                    captured_at: self.clock.now().await,
                    processing_time: None,
                    source: "output".to_string(),
                });
            }
        }

        let mut metrics = self.metrics.write().await;
        metrics.events_received += 1;
    }

    /// Get output events
    pub async fn get_output(&self) -> Vec<StreamEvent> {
        let output = self.output_events.read().await;
        output.iter().cloned().collect()
    }

    /// Clear output events
    pub async fn clear_output(&self) {
        let mut output = self.output_events.write().await;
        output.clear();
    }

    /// Get captured events
    pub async fn get_captured_events(&self) -> Vec<CapturedEvent> {
        let captured = self.captured_events.read().await;
        captured.clone()
    }

    /// Assert event count
    pub async fn assert_event_count(&self, expected: usize) -> Result<()> {
        let output = self.output_events.read().await;
        let actual = output.len();

        let passed = actual == expected;
        let error_message = if passed {
            None
        } else {
            Some(format!("Expected {} events, got {}", expected, actual))
        };

        let assertion = Assertion {
            assertion_type: AssertionType::EventCount,
            expected: expected.to_string(),
            actual: Some(actual.to_string()),
            passed,
            error_message: error_message.clone(),
            timestamp: self.clock.now().await,
        };

        let mut assertions = self.assertions.write().await;
        assertions.push(assertion);

        let mut metrics = self.metrics.write().await;
        metrics.total_assertions += 1;
        if passed {
            metrics.passed_assertions += 1;
        } else {
            metrics.failed_assertions += 1;
        }

        if passed {
            Ok(())
        } else {
            Err(anyhow!(
                error_message.expect("error_message should be set when assertion fails")
            ))
        }
    }

    /// Assert output contains event
    pub async fn assert_contains(&self, predicate: impl Fn(&StreamEvent) -> bool) -> Result<()> {
        let output = self.output_events.read().await;
        let found = output.iter().any(predicate);

        let passed = found;
        let error_message = if passed {
            None
        } else {
            Some("Expected event not found in output".to_string())
        };

        let assertion = Assertion {
            assertion_type: AssertionType::Contains,
            expected: "matching event".to_string(),
            actual: Some(format!("{} events checked", output.len())),
            passed,
            error_message: error_message.clone(),
            timestamp: self.clock.now().await,
        };

        let mut assertions = self.assertions.write().await;
        assertions.push(assertion);

        let mut metrics = self.metrics.write().await;
        metrics.total_assertions += 1;
        if passed {
            metrics.passed_assertions += 1;
        } else {
            metrics.failed_assertions += 1;
        }

        if passed {
            Ok(())
        } else {
            Err(anyhow!(
                error_message.expect("error_message should be set when assertion fails")
            ))
        }
    }

    /// Assert no output events
    pub async fn assert_no_events(&self) -> Result<()> {
        self.assert_event_count(0).await
    }

    /// Assert events within duration
    pub async fn assert_within(
        &self,
        duration: Duration,
        condition: impl Fn(&[StreamEvent]) -> bool,
    ) -> Result<()> {
        let start = Instant::now();

        while start.elapsed() < duration {
            let output = self.output_events.read().await;
            let events: Vec<_> = output.iter().cloned().collect();
            drop(output);

            if condition(&events) {
                let assertion = Assertion {
                    assertion_type: AssertionType::WithinDuration,
                    expected: format!("condition within {:?}", duration),
                    actual: Some(format!("satisfied after {:?}", start.elapsed())),
                    passed: true,
                    error_message: None,
                    timestamp: self.clock.now().await,
                };

                let mut assertions = self.assertions.write().await;
                assertions.push(assertion);

                let mut metrics = self.metrics.write().await;
                metrics.total_assertions += 1;
                metrics.passed_assertions += 1;

                return Ok(());
            }

            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let error_message = format!("Condition not satisfied within {:?}", duration);

        let assertion = Assertion {
            assertion_type: AssertionType::WithinDuration,
            expected: format!("condition within {:?}", duration),
            actual: Some("timeout".to_string()),
            passed: false,
            error_message: Some(error_message.clone()),
            timestamp: self.clock.now().await,
        };

        let mut assertions = self.assertions.write().await;
        assertions.push(assertion);

        let mut metrics = self.metrics.write().await;
        metrics.total_assertions += 1;
        metrics.failed_assertions += 1;

        Err(anyhow!(error_message))
    }

    /// Assert performance metric
    pub async fn assert_performance(
        &self,
        metric: PerformanceMetric,
        threshold: f64,
    ) -> Result<()> {
        let metrics = self.metrics.read().await;

        let (actual, passed) = match metric {
            PerformanceMetric::AvgLatency => (
                metrics.avg_processing_time_us,
                metrics.avg_processing_time_us <= threshold,
            ),
            PerformanceMetric::MaxLatency => (
                metrics.max_processing_time_us as f64,
                metrics.max_processing_time_us as f64 <= threshold,
            ),
            PerformanceMetric::Throughput => {
                let throughput = if metrics.test_duration.as_secs_f64() > 0.0 {
                    metrics.events_received as f64 / metrics.test_duration.as_secs_f64()
                } else {
                    0.0
                };
                (throughput, throughput >= threshold)
            }
        };

        drop(metrics);

        let error_message = if passed {
            None
        } else {
            Some(format!(
                "{:?} {} does not meet threshold {}",
                metric, actual, threshold
            ))
        };

        let assertion = Assertion {
            assertion_type: AssertionType::Performance,
            expected: format!("{:?} <= {}", metric, threshold),
            actual: Some(actual.to_string()),
            passed,
            error_message: error_message.clone(),
            timestamp: self.clock.now().await,
        };

        let mut assertions = self.assertions.write().await;
        assertions.push(assertion);

        let mut metrics = self.metrics.write().await;
        metrics.total_assertions += 1;
        if passed {
            metrics.passed_assertions += 1;
        } else {
            metrics.failed_assertions += 1;
        }

        if passed {
            Ok(())
        } else {
            Err(anyhow!(
                error_message.expect("error_message should be set when assertion fails")
            ))
        }
    }

    /// Get test metrics
    pub async fn get_metrics(&self) -> TestMetrics {
        self.metrics.read().await.clone()
    }

    /// Get all assertions
    pub async fn get_assertions(&self) -> Vec<Assertion> {
        self.assertions.read().await.clone()
    }

    /// Generate test report
    pub async fn generate_report(&self) -> TestReport {
        let metrics = self.metrics.read().await;
        let assertions = self.assertions.read().await;
        let captured = self.captured_events.read().await;

        TestReport {
            test_name: "stream_test".to_string(),
            status: if metrics.failed_assertions == 0 {
                TestStatus::Passed
            } else {
                TestStatus::Failed
            },
            metrics: metrics.clone(),
            assertions: assertions.clone(),
            event_count: captured.len(),
            generated_at: Utc::now(),
        }
    }

    /// Reset harness state
    pub async fn reset(&self) {
        self.output_events.write().await.clear();
        self.captured_events.write().await.clear();
        *self.metrics.write().await = TestMetrics::default();
        self.assertions.write().await.clear();

        if self.config.verbose {
            info!("Test harness reset");
        }
    }
}

/// Performance metrics
#[derive(Debug, Clone, Copy)]
pub enum PerformanceMetric {
    /// Average latency in microseconds
    AvgLatency,
    /// Maximum latency in microseconds
    MaxLatency,
    /// Throughput in events per second
    Throughput,
}

/// Test report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestReport {
    /// Test name
    pub test_name: String,
    /// Test status
    pub status: TestStatus,
    /// Test metrics
    pub metrics: TestMetrics,
    /// Assertions
    pub assertions: Vec<Assertion>,
    /// Total event count
    pub event_count: usize,
    /// Report generation time
    pub generated_at: DateTime<Utc>,
}

/// Test status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TestStatus {
    Passed,
    Failed,
    Skipped,
    Error,
}

impl TestReport {
    /// Convert to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| anyhow!("JSON error: {}", e))
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("\n=== Test Report: {} ===", self.test_name);
        println!("Status: {:?}", self.status);
        println!("Events pushed: {}", self.metrics.events_pushed);
        println!("Events received: {}", self.metrics.events_received);
        println!(
            "Assertions: {}/{} passed",
            self.metrics.passed_assertions, self.metrics.total_assertions
        );
        if self.metrics.total_assertions > 0 && self.metrics.failed_assertions > 0 {
            println!("Failed assertions:");
            for assertion in &self.assertions {
                if !assertion.passed {
                    println!(
                        "  - {:?}: {}",
                        assertion.assertion_type,
                        assertion.error_message.clone().unwrap_or_default()
                    );
                }
            }
        }
        println!("========================\n");
    }
}

/// Test fixture for common test scenarios
pub struct TestFixture {
    /// Name
    pub name: String,
    /// Setup events
    pub setup_events: Vec<StreamEvent>,
    /// Expected outputs
    pub expected_outputs: Vec<StreamEvent>,
    /// Time advancement
    pub time_advance: Option<Duration>,
}

impl TestFixture {
    /// Create a new fixture
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            setup_events: Vec::new(),
            expected_outputs: Vec::new(),
            time_advance: None,
        }
    }

    /// Add setup event
    pub fn with_input(mut self, event: StreamEvent) -> Self {
        self.setup_events.push(event);
        self
    }

    /// Add expected output
    pub fn expect_output(mut self, event: StreamEvent) -> Self {
        self.expected_outputs.push(event);
        self
    }

    /// Set time advancement
    pub fn advance_time(mut self, duration: Duration) -> Self {
        self.time_advance = Some(duration);
        self
    }

    /// Run fixture with harness
    pub async fn run(&self, harness: &TestHarness) -> Result<()> {
        // Push setup events
        harness.push_events(self.setup_events.clone()).await?;

        // Advance time if configured
        if let Some(duration) = self.time_advance {
            harness.advance_time(duration).await;
        }

        // Verify outputs
        harness
            .assert_event_count(self.expected_outputs.len())
            .await?;

        Ok(())
    }
}

/// Event predicate function type
type EventPredicate = Box<dyn Fn(&StreamEvent) -> bool + Send + Sync>;

/// Event matcher for assertions
pub struct EventMatcher {
    conditions: Vec<EventPredicate>,
}

impl EventMatcher {
    /// Create a new matcher
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
        }
    }

    /// Add condition
    pub fn with_condition<F>(mut self, condition: F) -> Self
    where
        F: Fn(&StreamEvent) -> bool + Send + Sync + 'static,
    {
        self.conditions.push(Box::new(condition));
        self
    }

    /// Match triple added events
    pub fn triple_added(mut self) -> Self {
        self.conditions
            .push(Box::new(|e| matches!(e, StreamEvent::TripleAdded { .. })));
        self
    }

    /// Match triple removed events
    pub fn triple_removed(mut self) -> Self {
        self.conditions
            .push(Box::new(|e| matches!(e, StreamEvent::TripleRemoved { .. })));
        self
    }

    /// Match events by source
    pub fn with_source(mut self, source: &str) -> Self {
        let source = source.to_string();
        self.conditions.push(Box::new(move |e| match e {
            StreamEvent::TripleAdded { metadata, .. }
            | StreamEvent::TripleRemoved { metadata, .. }
            | StreamEvent::GraphCreated { metadata, .. }
            | StreamEvent::GraphDeleted { metadata, .. }
            | StreamEvent::TransactionBegin { metadata, .. }
            | StreamEvent::TransactionCommit { metadata, .. }
            | StreamEvent::TransactionAbort { metadata, .. }
            | StreamEvent::Heartbeat { metadata, .. } => metadata.source == source,
            _ => false,
        }));
        self
    }

    /// Match SPARQL update events
    pub fn sparql_update(mut self) -> Self {
        self.conditions
            .push(Box::new(|e| matches!(e, StreamEvent::SparqlUpdate { .. })));
        self
    }

    /// Match heartbeat events
    pub fn heartbeat(mut self) -> Self {
        self.conditions
            .push(Box::new(|e| matches!(e, StreamEvent::Heartbeat { .. })));
        self
    }

    /// Check if event matches all conditions
    pub fn matches(&self, event: &StreamEvent) -> bool {
        self.conditions.iter().all(|c| c(event))
    }
}

impl Default for EventMatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Macros for common assertions
#[macro_export]
macro_rules! assert_stream_output {
    ($harness:expr, count($expected:expr)) => {
        $harness.assert_event_count($expected).await
    };
    ($harness:expr, contains($predicate:expr)) => {
        $harness.assert_contains($predicate).await
    };
    ($harness:expr, empty) => {
        $harness.assert_no_events().await
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_harness_builder() {
        let harness = TestHarness::builder()
            .with_mock_clock()
            .with_buffer_size(1000)
            .with_timeout(Duration::from_secs(5))
            .build()
            .await
            .unwrap();

        assert!(harness.config.use_mock_clock);
        assert_eq!(harness.config.event_buffer_size, 1000);
    }

    #[tokio::test]
    async fn test_mock_clock() {
        let clock = MockClock::new(Utc::now());
        let initial = clock.now().await;

        clock.advance(Duration::from_secs(60)).await;
        let after = clock.now().await;

        let diff = (after - initial).num_seconds();
        assert_eq!(diff, 60);
    }

    #[tokio::test]
    async fn test_event_generator() {
        let generator = EventGenerator::sequential(10);
        let time = Utc::now();

        let events = generator.generate_batch(5, time);
        assert_eq!(events.len(), 5);
    }

    #[tokio::test]
    async fn test_push_events() {
        let harness = TestHarness::builder().build().await.unwrap();

        let metadata = EventMetadata {
            event_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            source: "test".to_string(),
            user: None,
            context: None,
            caused_by: None,
            version: "1.0".to_string(),
            properties: HashMap::new(),
            checksum: None,
        };

        let event = StreamEvent::TripleAdded {
            subject: "test:subject".to_string(),
            predicate: "test:predicate".to_string(),
            object: "value1".to_string(),
            graph: None,
            metadata,
        };

        harness.push_event(event).await.unwrap();

        let metrics = harness.get_metrics().await;
        assert_eq!(metrics.events_pushed, 1);
    }

    #[tokio::test]
    async fn test_assert_event_count() {
        let harness = TestHarness::builder().build().await.unwrap();

        // Should pass with 0 events
        harness.assert_event_count(0).await.unwrap();

        // Add an output event
        let metadata = EventMetadata {
            event_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            source: "test".to_string(),
            user: None,
            context: None,
            caused_by: None,
            version: "1.0".to_string(),
            properties: HashMap::new(),
            checksum: None,
        };

        let event = StreamEvent::TripleAdded {
            subject: "test:subject".to_string(),
            predicate: "test:predicate".to_string(),
            object: "value1".to_string(),
            graph: None,
            metadata,
        };

        harness.add_output(event).await;

        // Should pass with 1 event
        harness.assert_event_count(1).await.unwrap();

        // Should fail with wrong count
        assert!(harness.assert_event_count(2).await.is_err());
    }

    #[tokio::test]
    async fn test_assert_contains() {
        let harness = TestHarness::builder().build().await.unwrap();

        let metadata = EventMetadata {
            event_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            source: "test".to_string(),
            user: None,
            context: None,
            caused_by: None,
            version: "1.0".to_string(),
            properties: HashMap::new(),
            checksum: None,
        };

        let event = StreamEvent::TripleAdded {
            subject: "test:subject".to_string(),
            predicate: "test:predicate".to_string(),
            object: "value42".to_string(),
            graph: None,
            metadata,
        };

        harness.add_output(event).await;

        // Should find triple added event
        harness.assert_contains(|e| {
            matches!(e, StreamEvent::TripleAdded { subject, .. } if subject == "test:subject")
        }).await.unwrap();

        // Should not find non-existent event
        assert!(harness
            .assert_contains(|e| {
                matches!(e, StreamEvent::TripleAdded { subject, .. } if subject == "other:subject")
            })
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_event_matcher() {
        let matcher = EventMatcher::new().triple_added();

        let metadata = EventMetadata {
            event_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            source: "test".to_string(),
            user: None,
            context: None,
            caused_by: None,
            version: "1.0".to_string(),
            properties: HashMap::new(),
            checksum: None,
        };

        let event = StreamEvent::TripleAdded {
            subject: "test:subject".to_string(),
            predicate: "test:predicate".to_string(),
            object: "value".to_string(),
            graph: None,
            metadata,
        };

        assert!(matcher.matches(&event));
    }

    #[tokio::test]
    async fn test_generate_report() {
        let harness = TestHarness::builder().build().await.unwrap();

        harness.assert_event_count(0).await.unwrap();

        let report = harness.generate_report().await;
        assert_eq!(report.status, TestStatus::Passed);
        assert_eq!(report.metrics.total_assertions, 1);
        assert_eq!(report.metrics.passed_assertions, 1);
    }

    #[tokio::test]
    async fn test_fixture() {
        let harness = TestHarness::builder().build().await.unwrap();

        let fixture = TestFixture::new("basic_test").advance_time(Duration::from_secs(60));

        // Should pass with no inputs/outputs
        fixture.run(&harness).await.unwrap();
    }

    #[tokio::test]
    async fn test_harness_reset() {
        let harness = TestHarness::builder().build().await.unwrap();

        let metadata = EventMetadata {
            event_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            source: "test".to_string(),
            user: None,
            context: None,
            caused_by: None,
            version: "1.0".to_string(),
            properties: HashMap::new(),
            checksum: None,
        };

        let event = StreamEvent::TripleAdded {
            subject: "test:subject".to_string(),
            predicate: "test:predicate".to_string(),
            object: "value".to_string(),
            graph: None,
            metadata,
        };

        harness.add_output(event).await;
        harness.assert_event_count(1).await.unwrap();

        harness.reset().await;

        let metrics = harness.get_metrics().await;
        assert_eq!(metrics.events_received, 0);
        assert_eq!(metrics.total_assertions, 0);

        harness.assert_event_count(0).await.unwrap();
    }

    #[tokio::test]
    async fn test_captured_events() {
        let harness = TestHarness::builder().build().await.unwrap();

        let metadata = EventMetadata {
            event_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            source: "test".to_string(),
            user: None,
            context: None,
            caused_by: None,
            version: "1.0".to_string(),
            properties: HashMap::new(),
            checksum: None,
        };

        let event = StreamEvent::TripleAdded {
            subject: "test:subject".to_string(),
            predicate: "test:predicate".to_string(),
            object: "value".to_string(),
            graph: None,
            metadata,
        };

        harness.push_event(event).await.unwrap();

        let captured = harness.get_captured_events().await;
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].source, "input");
    }

    #[tokio::test]
    async fn test_time_advancement() {
        let initial = Utc::now();
        let harness = TestHarness::builder()
            .with_mock_clock()
            .with_initial_time(initial)
            .build()
            .await
            .unwrap();

        assert_eq!(harness.now().await, initial);

        harness.advance_time(Duration::from_secs(3600)).await;
        let after = harness.now().await;
        let diff = (after - initial).num_seconds();
        assert_eq!(diff, 3600);
    }
}
