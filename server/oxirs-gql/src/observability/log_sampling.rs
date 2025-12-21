//! Advanced Log Sampling Strategies
//!
//! Provides intelligent log sampling to reduce volume while maintaining visibility
//! into important events. Supports multiple sampling strategies and composite samplers.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use super::structured_logging::LogLevel;

/// Sampling decision for a log entry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingDecision {
    /// Record this log entry
    Record,
    /// Drop this log entry
    Drop,
}

/// Context for sampling decisions
#[derive(Debug, Clone)]
pub struct SamplingContext {
    /// Log level
    pub level: LogLevel,
    /// Operation name
    pub operation: Option<String>,
    /// Request ID
    pub request_id: Option<String>,
    /// Whether this log is for an error
    pub is_error: bool,
    /// Priority (0-10, higher is more important)
    pub priority: u8,
    /// Trace ID for correlation
    pub trace_id: Option<String>,
    /// User ID
    pub user_id: Option<String>,
    /// Tags for filtering
    pub tags: HashMap<String, String>,
}

impl SamplingContext {
    /// Create a new sampling context
    pub fn new(level: LogLevel) -> Self {
        Self {
            level,
            operation: None,
            request_id: None,
            is_error: false,
            priority: 5,
            trace_id: None,
            user_id: None,
            tags: HashMap::new(),
        }
    }

    /// Set operation name
    pub fn with_operation(mut self, operation: String) -> Self {
        self.operation = Some(operation);
        self
    }

    /// Set request ID
    pub fn with_request_id(mut self, request_id: String) -> Self {
        self.request_id = Some(request_id);
        self
    }

    /// Mark as error
    pub fn with_error(mut self, is_error: bool) -> Self {
        self.is_error = is_error;
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority.min(10);
        self
    }

    /// Set trace ID
    pub fn with_trace_id(mut self, trace_id: String) -> Self {
        self.trace_id = Some(trace_id);
        self
    }

    /// Set user ID
    pub fn with_user_id(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }

    /// Add tag
    pub fn with_tag(mut self, key: String, value: String) -> Self {
        self.tags.insert(key, value);
        self
    }
}

/// Trait for log sampling strategies
pub trait LogSampler: Send + Sync {
    /// Determine whether to sample this log entry
    fn should_sample(&self, context: &SamplingContext) -> SamplingDecision;

    /// Reset sampler state
    fn reset(&self) {}

    /// Get sampler statistics
    fn get_statistics(&self) -> SamplerStatistics {
        SamplerStatistics::default()
    }
}

/// Sampler statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SamplerStatistics {
    /// Total logs evaluated
    pub total_evaluated: u64,
    /// Total logs recorded
    pub total_recorded: u64,
    /// Total logs dropped
    pub total_dropped: u64,
    /// Sampling rate (0.0 to 1.0)
    pub sampling_rate: f64,
    /// Last reset time
    pub last_reset: Option<u64>,
}

impl SamplerStatistics {
    /// Calculate sampling rate
    pub fn calculate_rate(&mut self) {
        if self.total_evaluated > 0 {
            self.sampling_rate = self.total_recorded as f64 / self.total_evaluated as f64;
        } else {
            self.sampling_rate = 0.0;
        }
    }
}

/// Always sample - records all logs
#[derive(Debug, Clone)]
pub struct AlwaysSampler;

impl LogSampler for AlwaysSampler {
    fn should_sample(&self, _context: &SamplingContext) -> SamplingDecision {
        SamplingDecision::Record
    }
}

/// Never sample - drops all logs
#[derive(Debug, Clone)]
pub struct NeverSampler;

impl LogSampler for NeverSampler {
    fn should_sample(&self, _context: &SamplingContext) -> SamplingDecision {
        SamplingDecision::Drop
    }
}

/// Probabilistic sampler - samples based on a probability
#[derive(Debug)]
pub struct ProbabilisticSampler {
    /// Sampling rate (0.0 to 1.0)
    rate: f64,
    /// Statistics
    stats: Arc<RwLock<SamplerStatistics>>,
}

impl ProbabilisticSampler {
    /// Create a new probabilistic sampler
    pub fn new(rate: f64) -> Self {
        Self {
            rate: rate.clamp(0.0, 1.0),
            stats: Arc::new(RwLock::new(SamplerStatistics::default())),
        }
    }
}

impl LogSampler for ProbabilisticSampler {
    fn should_sample(&self, _context: &SamplingContext) -> SamplingDecision {
        let random_value = fastrand::f64();
        if random_value < self.rate {
            SamplingDecision::Record
        } else {
            SamplingDecision::Drop
        }
    }

    fn get_statistics(&self) -> SamplerStatistics {
        // This is a sync method, so we can't await here
        // Return a default for now - in real implementation, use try_read
        match self.stats.try_read() {
            Ok(stats) => stats.clone(),
            Err(_) => SamplerStatistics::default(),
        }
    }
}

/// Rate-limited sampler - limits logs per time window
#[derive(Debug)]
pub struct RateLimitedSampler {
    /// Maximum logs per window
    max_logs_per_window: usize,
    /// Time window duration
    window_duration: Duration,
    /// Log timestamps
    log_timestamps: Arc<RwLock<VecDeque<Instant>>>,
    /// Statistics
    stats: Arc<RwLock<SamplerStatistics>>,
}

impl RateLimitedSampler {
    /// Create a new rate-limited sampler
    pub fn new(max_logs_per_window: usize, window_duration: Duration) -> Self {
        Self {
            max_logs_per_window,
            window_duration,
            log_timestamps: Arc::new(RwLock::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(SamplerStatistics::default())),
        }
    }
}

impl LogSampler for RateLimitedSampler {
    fn should_sample(&self, _context: &SamplingContext) -> SamplingDecision {
        // Note: This is a simplified sync implementation
        // In a real async context, you'd use tokio::task::block_in_place
        // For now, we'll use a simple try_write approach
        if let Ok(mut timestamps) = self.log_timestamps.try_write() {
            let now = Instant::now();

            // Remove old timestamps
            while let Some(&front) = timestamps.front() {
                if now.duration_since(front) > self.window_duration {
                    timestamps.pop_front();
                } else {
                    break;
                }
            }

            // Check if we can add another log
            if timestamps.len() < self.max_logs_per_window {
                timestamps.push_back(now);
                return SamplingDecision::Record;
            }
        }

        SamplingDecision::Drop
    }

    fn reset(&self) {
        if let Ok(mut timestamps) = self.log_timestamps.try_write() {
            timestamps.clear();
        }
        if let Ok(mut stats) = self.stats.try_write() {
            *stats = SamplerStatistics::default();
        }
    }

    fn get_statistics(&self) -> SamplerStatistics {
        match self.stats.try_read() {
            Ok(stats) => stats.clone(),
            Err(_) => SamplerStatistics::default(),
        }
    }
}

/// Priority-based sampler - samples based on priority levels
#[derive(Debug)]
pub struct PriorityBasedSampler {
    /// Minimum priority to sample (0-10)
    min_priority: u8,
    /// Priority-specific rates
    priority_rates: HashMap<u8, f64>,
    /// Statistics
    stats: Arc<RwLock<SamplerStatistics>>,
}

impl PriorityBasedSampler {
    /// Create a new priority-based sampler
    pub fn new(min_priority: u8) -> Self {
        Self {
            min_priority: min_priority.min(10),
            priority_rates: HashMap::new(),
            stats: Arc::new(RwLock::new(SamplerStatistics::default())),
        }
    }

    /// Set sampling rate for a specific priority
    pub fn with_priority_rate(mut self, priority: u8, rate: f64) -> Self {
        self.priority_rates.insert(priority, rate.clamp(0.0, 1.0));
        self
    }
}

impl LogSampler for PriorityBasedSampler {
    fn should_sample(&self, context: &SamplingContext) -> SamplingDecision {
        // Always sample high priority logs
        if context.priority >= self.min_priority {
            // Check for priority-specific rate
            if let Some(&rate) = self.priority_rates.get(&context.priority) {
                if fastrand::f64() < rate {
                    return SamplingDecision::Record;
                } else {
                    return SamplingDecision::Drop;
                }
            }
            return SamplingDecision::Record;
        }

        SamplingDecision::Drop
    }

    fn get_statistics(&self) -> SamplerStatistics {
        match self.stats.try_read() {
            Ok(stats) => stats.clone(),
            Err(_) => SamplerStatistics::default(),
        }
    }
}

/// Error-aware sampler - always samples errors, probabilistic for others
#[derive(Debug)]
pub struct ErrorAwareSampler {
    /// Sampling rate for non-error logs
    normal_rate: f64,
    /// Always sample errors
    always_sample_errors: bool,
    /// Statistics
    stats: Arc<RwLock<SamplerStatistics>>,
}

impl ErrorAwareSampler {
    /// Create a new error-aware sampler
    pub fn new(normal_rate: f64, always_sample_errors: bool) -> Self {
        Self {
            normal_rate: normal_rate.clamp(0.0, 1.0),
            always_sample_errors,
            stats: Arc::new(RwLock::new(SamplerStatistics::default())),
        }
    }
}

impl LogSampler for ErrorAwareSampler {
    fn should_sample(&self, context: &SamplingContext) -> SamplingDecision {
        // Always sample errors if configured
        if self.always_sample_errors && context.is_error {
            return SamplingDecision::Record;
        }

        // Sample high-level logs (Error, Warn) more aggressively
        let rate = match context.level {
            LogLevel::Error | LogLevel::Warn => 1.0,
            _ => self.normal_rate,
        };

        if fastrand::f64() < rate {
            SamplingDecision::Record
        } else {
            SamplingDecision::Drop
        }
    }

    fn get_statistics(&self) -> SamplerStatistics {
        match self.stats.try_read() {
            Ok(stats) => stats.clone(),
            Err(_) => SamplerStatistics::default(),
        }
    }
}

/// Tail sampler - buffers logs and decides based on final outcome
#[derive(Debug)]
pub struct TailSampler {
    /// Buffer size
    buffer_size: usize,
    /// Buffered contexts by trace ID
    buffer: Arc<RwLock<HashMap<String, Vec<SamplingContext>>>>,
    /// Statistics
    stats: Arc<RwLock<SamplerStatistics>>,
}

impl TailSampler {
    /// Create a new tail sampler
    pub fn new(buffer_size: usize) -> Self {
        Self {
            buffer_size,
            buffer: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(SamplerStatistics::default())),
        }
    }

    /// Make a decision for a trace
    pub async fn decide_trace(&self, trace_id: &str) -> Vec<SamplingDecision> {
        let buffer = self.buffer.read().await;
        if let Some(contexts) = buffer.get(trace_id) {
            // If any log in the trace is an error, sample entire trace
            let has_error = contexts.iter().any(|ctx| ctx.is_error);
            let decision = if has_error {
                SamplingDecision::Record
            } else {
                SamplingDecision::Drop
            };

            vec![decision; contexts.len()]
        } else {
            vec![]
        }
    }
}

impl LogSampler for TailSampler {
    fn should_sample(&self, context: &SamplingContext) -> SamplingDecision {
        // For tail sampling, we initially buffer everything
        // and make decisions later based on trace completion
        if let Some(trace_id) = &context.trace_id {
            if let Ok(mut buffer) = self.buffer.try_write() {
                buffer
                    .entry(trace_id.clone())
                    .or_insert_with(Vec::new)
                    .push(context.clone());

                // Limit buffer size
                if buffer.len() > self.buffer_size {
                    // Remove oldest trace
                    if let Some(oldest_key) = buffer.keys().next().cloned() {
                        buffer.remove(&oldest_key);
                    }
                }
            }
        }

        // Initially, we assume we'll record (decision made later)
        SamplingDecision::Record
    }

    fn reset(&self) {
        if let Ok(mut buffer) = self.buffer.try_write() {
            buffer.clear();
        }
        if let Ok(mut stats) = self.stats.try_write() {
            *stats = SamplerStatistics::default();
        }
    }

    fn get_statistics(&self) -> SamplerStatistics {
        match self.stats.try_read() {
            Ok(stats) => stats.clone(),
            Err(_) => SamplerStatistics::default(),
        }
    }
}

/// Adaptive sampler - adjusts rate based on system load
#[derive(Debug)]
pub struct AdaptiveSampler {
    /// Base sampling rate
    base_rate: f64,
    /// Current sampling rate (adjusted dynamically)
    current_rate: Arc<RwLock<f64>>,
    /// Target logs per second
    target_logs_per_second: f64,
    /// Recent log count
    recent_logs: Arc<RwLock<VecDeque<Instant>>>,
    /// Statistics
    stats: Arc<RwLock<SamplerStatistics>>,
}

impl AdaptiveSampler {
    /// Create a new adaptive sampler
    pub fn new(base_rate: f64, target_logs_per_second: f64) -> Self {
        Self {
            base_rate: base_rate.clamp(0.0, 1.0),
            current_rate: Arc::new(RwLock::new(base_rate.clamp(0.0, 1.0))),
            target_logs_per_second,
            recent_logs: Arc::new(RwLock::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(SamplerStatistics::default())),
        }
    }

    /// Adjust sampling rate based on load
    pub async fn adjust_rate(&self) {
        let mut logs = self.recent_logs.write().await;
        let now = Instant::now();

        // Remove logs older than 1 second
        while let Some(&front) = logs.front() {
            if now.duration_since(front) > Duration::from_secs(1) {
                logs.pop_front();
            } else {
                break;
            }
        }

        let current_logs_per_second = logs.len() as f64;

        // Adjust rate
        let mut current_rate = self.current_rate.write().await;
        if current_logs_per_second > self.target_logs_per_second {
            // Decrease rate
            *current_rate = (*current_rate * 0.9).max(0.01);
        } else if current_logs_per_second < self.target_logs_per_second * 0.8 {
            // Increase rate
            *current_rate = (*current_rate * 1.1).min(1.0);
        }
    }

    /// Get current sampling rate
    pub async fn get_current_rate(&self) -> f64 {
        *self.current_rate.read().await
    }
}

impl LogSampler for AdaptiveSampler {
    fn should_sample(&self, _context: &SamplingContext) -> SamplingDecision {
        if let Ok(logs) = self.recent_logs.try_write() {
            // Track this log
            let mut logs = logs;
            logs.push_back(Instant::now());

            // Use current rate
            if let Ok(current_rate) = self.current_rate.try_read() {
                if fastrand::f64() < *current_rate {
                    return SamplingDecision::Record;
                }
            }
        }

        SamplingDecision::Drop
    }

    fn reset(&self) {
        if let Ok(mut logs) = self.recent_logs.try_write() {
            logs.clear();
        }
        if let Ok(mut current_rate) = self.current_rate.try_write() {
            *current_rate = self.base_rate;
        }
        if let Ok(mut stats) = self.stats.try_write() {
            *stats = SamplerStatistics::default();
        }
    }

    fn get_statistics(&self) -> SamplerStatistics {
        match self.stats.try_read() {
            Ok(stats) => stats.clone(),
            Err(_) => SamplerStatistics::default(),
        }
    }
}

/// Composite sampler - combines multiple samplers
#[derive(Debug, Clone, Copy)]
pub enum CompositeStrategy {
    /// All samplers must agree to record
    All,
    /// Any sampler can decide to record
    Any,
    /// Use first sampler that returns Record
    FirstMatch,
}

pub struct CompositeSampler {
    /// Sampling strategy
    strategy: CompositeStrategy,
    /// Child samplers
    samplers: Vec<Arc<dyn LogSampler>>,
}

impl CompositeSampler {
    /// Create a new composite sampler
    pub fn new(strategy: CompositeStrategy) -> Self {
        Self {
            strategy,
            samplers: Vec::new(),
        }
    }

    /// Add a sampler
    pub fn add_sampler(mut self, sampler: Arc<dyn LogSampler>) -> Self {
        self.samplers.push(sampler);
        self
    }
}

impl LogSampler for CompositeSampler {
    fn should_sample(&self, context: &SamplingContext) -> SamplingDecision {
        match self.strategy {
            CompositeStrategy::All => {
                // All samplers must agree to record
                for sampler in &self.samplers {
                    if sampler.should_sample(context) == SamplingDecision::Drop {
                        return SamplingDecision::Drop;
                    }
                }
                SamplingDecision::Record
            }
            CompositeStrategy::Any => {
                // Any sampler can decide to record
                for sampler in &self.samplers {
                    if sampler.should_sample(context) == SamplingDecision::Record {
                        return SamplingDecision::Record;
                    }
                }
                SamplingDecision::Drop
            }
            CompositeStrategy::FirstMatch => {
                // Use first sampler that returns Record
                for sampler in &self.samplers {
                    let decision = sampler.should_sample(context);
                    if decision == SamplingDecision::Record {
                        return decision;
                    }
                }
                SamplingDecision::Drop
            }
        }
    }

    fn reset(&self) {
        for sampler in &self.samplers {
            sampler.reset();
        }
    }
}

/// Log sampling manager
pub struct LogSamplingManager {
    /// Default sampler
    default_sampler: Arc<dyn LogSampler>,
    /// Named samplers for specific use cases
    named_samplers: HashMap<String, Arc<dyn LogSampler>>,
}

impl LogSamplingManager {
    /// Create a new log sampling manager
    pub fn new(default_sampler: Arc<dyn LogSampler>) -> Self {
        Self {
            default_sampler,
            named_samplers: HashMap::new(),
        }
    }

    /// Register a named sampler
    pub fn register_sampler(&mut self, name: String, sampler: Arc<dyn LogSampler>) {
        self.named_samplers.insert(name, sampler);
    }

    /// Get sampler by name
    pub fn get_sampler(&self, name: &str) -> Option<&Arc<dyn LogSampler>> {
        self.named_samplers.get(name)
    }

    /// Use default sampler
    pub fn should_sample(&self, context: &SamplingContext) -> SamplingDecision {
        self.default_sampler.should_sample(context)
    }

    /// Use named sampler
    pub fn should_sample_with(&self, name: &str, context: &SamplingContext) -> SamplingDecision {
        if let Some(sampler) = self.named_samplers.get(name) {
            sampler.should_sample(context)
        } else {
            self.default_sampler.should_sample(context)
        }
    }

    /// Reset all samplers
    pub fn reset_all(&self) {
        self.default_sampler.reset();
        for sampler in self.named_samplers.values() {
            sampler.reset();
        }
    }
}

impl Default for LogSamplingManager {
    fn default() -> Self {
        Self::new(Arc::new(AlwaysSampler))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_always_sampler() {
        let sampler = AlwaysSampler;
        let context = SamplingContext::new(LogLevel::Info);

        assert_eq!(sampler.should_sample(&context), SamplingDecision::Record);
    }

    #[test]
    fn test_never_sampler() {
        let sampler = NeverSampler;
        let context = SamplingContext::new(LogLevel::Info);

        assert_eq!(sampler.should_sample(&context), SamplingDecision::Drop);
    }

    #[test]
    fn test_probabilistic_sampler() {
        let sampler = ProbabilisticSampler::new(0.5);
        let context = SamplingContext::new(LogLevel::Info);

        // Run multiple times to test randomness
        let mut records = 0;
        let mut _drops = 0;

        for _ in 0..100 {
            match sampler.should_sample(&context) {
                SamplingDecision::Record => records += 1,
                SamplingDecision::Drop => _drops += 1,
            }
        }

        // With 50% rate, we expect roughly 50/50 split (with some variance)
        assert!(records > 30 && records < 70);
        assert!(_drops > 30 && _drops < 70);
    }

    #[test]
    fn test_probabilistic_sampler_zero_rate() {
        let sampler = ProbabilisticSampler::new(0.0);
        let context = SamplingContext::new(LogLevel::Info);

        for _ in 0..10 {
            assert_eq!(sampler.should_sample(&context), SamplingDecision::Drop);
        }
    }

    #[test]
    fn test_probabilistic_sampler_full_rate() {
        let sampler = ProbabilisticSampler::new(1.0);
        let context = SamplingContext::new(LogLevel::Info);

        for _ in 0..10 {
            assert_eq!(sampler.should_sample(&context), SamplingDecision::Record);
        }
    }

    #[test]
    fn test_rate_limited_sampler() {
        let sampler = RateLimitedSampler::new(5, Duration::from_secs(1));
        let context = SamplingContext::new(LogLevel::Info);

        // First 5 should be recorded
        for _ in 0..5 {
            assert_eq!(sampler.should_sample(&context), SamplingDecision::Record);
        }

        // Next ones should be dropped
        for _ in 0..3 {
            assert_eq!(sampler.should_sample(&context), SamplingDecision::Drop);
        }
    }

    #[test]
    fn test_priority_based_sampler() {
        let sampler = PriorityBasedSampler::new(7);
        let high_priority = SamplingContext::new(LogLevel::Info).with_priority(8);
        let low_priority = SamplingContext::new(LogLevel::Info).with_priority(5);

        assert_eq!(
            sampler.should_sample(&high_priority),
            SamplingDecision::Record
        );
        assert_eq!(sampler.should_sample(&low_priority), SamplingDecision::Drop);
    }

    #[test]
    fn test_priority_based_sampler_with_rates() {
        let sampler = PriorityBasedSampler::new(5).with_priority_rate(7, 0.5);

        let context = SamplingContext::new(LogLevel::Info).with_priority(7);

        // Run multiple times
        let mut records = 0;
        let mut _drops = 0;

        for _ in 0..100 {
            match sampler.should_sample(&context) {
                SamplingDecision::Record => records += 1,
                SamplingDecision::Drop => _drops += 1,
            }
        }

        // Should be roughly 50/50
        assert!(records > 30 && records < 70);
    }

    #[test]
    fn test_error_aware_sampler() {
        let sampler = ErrorAwareSampler::new(0.1, true);

        let error_context = SamplingContext::new(LogLevel::Error).with_error(true);
        let normal_context = SamplingContext::new(LogLevel::Info);

        // Errors should always be sampled
        assert_eq!(
            sampler.should_sample(&error_context),
            SamplingDecision::Record
        );

        // Normal logs sampled at 10%
        let mut records = 0;
        for _ in 0..100 {
            if sampler.should_sample(&normal_context) == SamplingDecision::Record {
                records += 1;
            }
        }

        // Should be roughly 10% (with variance)
        assert!(records > 0 && records < 30);
    }

    #[test]
    fn test_error_aware_sampler_log_levels() {
        let sampler = ErrorAwareSampler::new(0.1, false);

        let error_context = SamplingContext::new(LogLevel::Error);
        let warn_context = SamplingContext::new(LogLevel::Warn);

        // Error and Warn should always be sampled (100% rate)
        assert_eq!(
            sampler.should_sample(&error_context),
            SamplingDecision::Record
        );
        assert_eq!(
            sampler.should_sample(&warn_context),
            SamplingDecision::Record
        );
    }

    #[test]
    fn test_tail_sampler() {
        let sampler = TailSampler::new(100);

        let context = SamplingContext::new(LogLevel::Info)
            .with_trace_id("trace-123".to_string())
            .with_error(false);

        // Should buffer the log
        assert_eq!(sampler.should_sample(&context), SamplingDecision::Record);
    }

    #[test]
    fn test_adaptive_sampler() {
        let sampler = AdaptiveSampler::new(0.5, 100.0);
        let context = SamplingContext::new(LogLevel::Info);

        // Run multiple times
        let mut records = 0;
        for _ in 0..100 {
            if sampler.should_sample(&context) == SamplingDecision::Record {
                records += 1;
            }
        }

        // Should sample some logs
        assert!(records > 0);
    }

    #[test]
    fn test_composite_sampler_all() {
        let sampler = CompositeSampler::new(CompositeStrategy::All)
            .add_sampler(Arc::new(AlwaysSampler))
            .add_sampler(Arc::new(ProbabilisticSampler::new(0.5)));

        let context = SamplingContext::new(LogLevel::Info);

        // Run multiple times - should respect the 50% rate
        let mut records = 0;
        for _ in 0..100 {
            if sampler.should_sample(&context) == SamplingDecision::Record {
                records += 1;
            }
        }

        assert!(records > 30 && records < 70);
    }

    #[test]
    fn test_composite_sampler_any() {
        let sampler = CompositeSampler::new(CompositeStrategy::Any)
            .add_sampler(Arc::new(AlwaysSampler))
            .add_sampler(Arc::new(NeverSampler));

        let context = SamplingContext::new(LogLevel::Info);

        // Should always record (AlwaysSampler wins)
        assert_eq!(sampler.should_sample(&context), SamplingDecision::Record);
    }

    #[test]
    fn test_composite_sampler_first_match() {
        let sampler = CompositeSampler::new(CompositeStrategy::FirstMatch)
            .add_sampler(Arc::new(NeverSampler))
            .add_sampler(Arc::new(AlwaysSampler));

        let context = SamplingContext::new(LogLevel::Info);

        // Should record (AlwaysSampler is first to return Record)
        assert_eq!(sampler.should_sample(&context), SamplingDecision::Record);
    }

    #[test]
    fn test_sampling_context_builder() {
        let context = SamplingContext::new(LogLevel::Info)
            .with_operation("query".to_string())
            .with_request_id("req-123".to_string())
            .with_error(true)
            .with_priority(8)
            .with_trace_id("trace-456".to_string())
            .with_user_id("user-789".to_string())
            .with_tag("env".to_string(), "prod".to_string());

        assert_eq!(context.operation, Some("query".to_string()));
        assert_eq!(context.request_id, Some("req-123".to_string()));
        assert!(context.is_error);
        assert_eq!(context.priority, 8);
        assert_eq!(context.trace_id, Some("trace-456".to_string()));
        assert_eq!(context.user_id, Some("user-789".to_string()));
        assert_eq!(context.tags.get("env"), Some(&"prod".to_string()));
    }

    #[test]
    fn test_log_sampling_manager() {
        let mut manager = LogSamplingManager::new(Arc::new(AlwaysSampler));

        manager.register_sampler("never".to_string(), Arc::new(NeverSampler));
        manager.register_sampler(
            "probabilistic".to_string(),
            Arc::new(ProbabilisticSampler::new(0.5)),
        );

        let context = SamplingContext::new(LogLevel::Info);

        // Default sampler (AlwaysSampler)
        assert_eq!(manager.should_sample(&context), SamplingDecision::Record);

        // Never sampler
        assert_eq!(
            manager.should_sample_with("never", &context),
            SamplingDecision::Drop
        );

        // Unknown sampler (falls back to default)
        assert_eq!(
            manager.should_sample_with("unknown", &context),
            SamplingDecision::Record
        );
    }

    #[test]
    fn test_rate_limited_sampler_reset() {
        let sampler = RateLimitedSampler::new(3, Duration::from_secs(1));
        let context = SamplingContext::new(LogLevel::Info);

        // Fill up the limit
        for _ in 0..3 {
            sampler.should_sample(&context);
        }

        // Should be at limit
        assert_eq!(sampler.should_sample(&context), SamplingDecision::Drop);

        // Reset
        sampler.reset();

        // Should work again
        assert_eq!(sampler.should_sample(&context), SamplingDecision::Record);
    }

    #[test]
    fn test_sampler_statistics() {
        let mut stats = SamplerStatistics {
            total_evaluated: 100,
            total_recorded: 75,
            total_dropped: 25,
            ..Default::default()
        };

        stats.calculate_rate();

        assert_eq!(stats.sampling_rate, 0.75);
    }

    #[test]
    fn test_priority_clamping() {
        let context = SamplingContext::new(LogLevel::Info).with_priority(15);

        // Priority should be clamped to 10
        assert_eq!(context.priority, 10);
    }

    #[test]
    fn test_probabilistic_rate_clamping() {
        let sampler_high = ProbabilisticSampler::new(1.5);
        let sampler_low = ProbabilisticSampler::new(-0.5);

        // Rates should be clamped to [0.0, 1.0]
        assert!(sampler_high.rate <= 1.0);
        assert!(sampler_low.rate >= 0.0);
    }
}
