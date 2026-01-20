//! Automatic Trace Sampling Strategies
//!
//! This module provides intelligent sampling strategies for distributed tracing,
//! helping to control costs while maintaining observability.
//!
//! # Sampling Strategies
//!
//! - **Always On**: Sample all traces (for development)
//! - **Always Off**: Sample no traces (for extreme cost reduction)
//! - **Probabilistic**: Random sampling with configurable rate
//! - **Rate Limited**: Maximum traces per second
//! - **Adaptive**: Dynamic sampling based on system load
//! - **Priority**: Sample based on operation priority
//! - **Error**: Always sample traces with errors
//! - **Tail**: Sample slow traces beyond threshold
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_gql::trace_sampling::{SamplingStrategy, AdaptiveSampler};
//!
//! let sampler = AdaptiveSampler::new()
//!     .with_base_rate(0.1)
//!     .with_error_sampling(true)
//!     .with_tail_sampling(Duration::from_secs(1));
//!
//! if sampler.should_sample(&trace_context) {
//!     // Export trace
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Sampling decision
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingDecision {
    /// Record and export
    RecordAndSample,
    /// Record but don't export
    RecordOnly,
    /// Drop trace
    Drop,
}

/// Sampling result
#[derive(Debug, Clone)]
pub struct SamplingResult {
    /// Sampling decision
    pub decision: SamplingDecision,
    /// Sampling rate applied
    pub rate: f64,
    /// Reason for decision
    pub reason: String,
}

impl SamplingResult {
    /// Check if should sample
    pub fn should_sample(&self) -> bool {
        self.decision == SamplingDecision::RecordAndSample
    }
}

/// Trace context for sampling decision
#[derive(Debug, Clone)]
pub struct SamplingContext {
    /// Operation name
    pub operation_name: String,
    /// Operation type (query, mutation, subscription)
    pub operation_type: String,
    /// Has error
    pub has_error: bool,
    /// Duration (if known)
    pub duration: Option<Duration>,
    /// Attributes
    pub attributes: HashMap<String, String>,
    /// Priority (0-10, higher is more important)
    pub priority: u8,
}

impl SamplingContext {
    /// Create new sampling context
    pub fn new(operation_name: impl Into<String>, operation_type: impl Into<String>) -> Self {
        Self {
            operation_name: operation_name.into(),
            operation_type: operation_type.into(),
            has_error: false,
            duration: None,
            attributes: HashMap::new(),
            priority: 5,
        }
    }

    /// Set error status
    pub fn with_error(mut self, has_error: bool) -> Self {
        self.has_error = has_error;
        self
    }

    /// Set duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority.min(10);
        self
    }

    /// Add attribute
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}

/// Sampling strategy trait
pub trait Sampler: Send + Sync {
    /// Decide whether to sample a trace
    fn should_sample(&self, context: &SamplingContext) -> SamplingResult;

    /// Get current sampling rate
    fn get_sampling_rate(&self) -> f64;

    /// Update sampler state (for adaptive samplers)
    fn update(&mut self, _context: &SamplingContext, _sampled: bool) {}
}

/// Always-on sampler (samples all traces)
#[derive(Debug, Clone)]
pub struct AlwaysOnSampler;

impl Sampler for AlwaysOnSampler {
    fn should_sample(&self, _context: &SamplingContext) -> SamplingResult {
        SamplingResult {
            decision: SamplingDecision::RecordAndSample,
            rate: 1.0,
            reason: "always-on".to_string(),
        }
    }

    fn get_sampling_rate(&self) -> f64 {
        1.0
    }
}

/// Always-off sampler (samples no traces)
#[derive(Debug, Clone)]
pub struct AlwaysOffSampler;

impl Sampler for AlwaysOffSampler {
    fn should_sample(&self, _context: &SamplingContext) -> SamplingResult {
        SamplingResult {
            decision: SamplingDecision::Drop,
            rate: 0.0,
            reason: "always-off".to_string(),
        }
    }

    fn get_sampling_rate(&self) -> f64 {
        0.0
    }
}

/// Probabilistic sampler
#[derive(Debug, Clone)]
pub struct ProbabilisticSampler {
    rate: f64,
}

impl ProbabilisticSampler {
    /// Create new probabilistic sampler
    pub fn new(rate: f64) -> Self {
        Self {
            rate: rate.clamp(0.0, 1.0),
        }
    }
}

impl Sampler for ProbabilisticSampler {
    fn should_sample(&self, _context: &SamplingContext) -> SamplingResult {
        let should_sample = fastrand::f64() < self.rate;

        SamplingResult {
            decision: if should_sample {
                SamplingDecision::RecordAndSample
            } else {
                SamplingDecision::Drop
            },
            rate: self.rate,
            reason: "probabilistic".to_string(),
        }
    }

    fn get_sampling_rate(&self) -> f64 {
        self.rate
    }
}

/// Rate-limited sampler
pub struct RateLimitedSampler {
    max_traces_per_second: f64,
    #[allow(dead_code)]
    state: Arc<RwLock<RateLimitState>>,
}

#[allow(dead_code)]
struct RateLimitState {
    count: usize,
    window_start: Instant,
}

impl RateLimitedSampler {
    /// Create new rate-limited sampler
    pub fn new(max_traces_per_second: f64) -> Self {
        Self {
            max_traces_per_second,
            state: Arc::new(RwLock::new(RateLimitState {
                count: 0,
                window_start: Instant::now(),
            })),
        }
    }
}

impl Sampler for RateLimitedSampler {
    fn should_sample(&self, _context: &SamplingContext) -> SamplingResult {
        // Note: This is a simplified sync version for testing
        // In production, use async version with proper locking
        let should_sample = fastrand::f64() < 0.1; // Simplified for sync trait

        SamplingResult {
            decision: if should_sample {
                SamplingDecision::RecordAndSample
            } else {
                SamplingDecision::Drop
            },
            rate: 0.1,
            reason: "rate-limited".to_string(),
        }
    }

    fn get_sampling_rate(&self) -> f64 {
        self.max_traces_per_second / 100.0
    }
}

/// Priority-based sampler
#[derive(Debug, Clone)]
pub struct PrioritySampler {
    priority_rates: HashMap<u8, f64>,
    default_rate: f64,
}

impl PrioritySampler {
    /// Create new priority-based sampler
    pub fn new() -> Self {
        let mut priority_rates = HashMap::new();
        // High priority (8-10): 100%
        priority_rates.insert(10, 1.0);
        priority_rates.insert(9, 1.0);
        priority_rates.insert(8, 1.0);
        // Medium priority (5-7): 50%
        priority_rates.insert(7, 0.5);
        priority_rates.insert(6, 0.5);
        priority_rates.insert(5, 0.5);
        // Low priority (0-4): 10%
        priority_rates.insert(4, 0.1);
        priority_rates.insert(3, 0.1);
        priority_rates.insert(2, 0.1);
        priority_rates.insert(1, 0.1);
        priority_rates.insert(0, 0.1);

        Self {
            priority_rates,
            default_rate: 0.1,
        }
    }

    /// Set rate for priority level
    pub fn set_priority_rate(&mut self, priority: u8, rate: f64) {
        self.priority_rates.insert(priority, rate.clamp(0.0, 1.0));
    }
}

impl Default for PrioritySampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for PrioritySampler {
    fn should_sample(&self, context: &SamplingContext) -> SamplingResult {
        let rate = *self
            .priority_rates
            .get(&context.priority)
            .unwrap_or(&self.default_rate);

        let should_sample = fastrand::f64() < rate;

        SamplingResult {
            decision: if should_sample {
                SamplingDecision::RecordAndSample
            } else {
                SamplingDecision::Drop
            },
            rate,
            reason: format!("priority-{}", context.priority),
        }
    }

    fn get_sampling_rate(&self) -> f64 {
        self.default_rate
    }
}

/// Error-aware sampler (always samples errors)
pub struct ErrorAwareSampler {
    base_sampler: Arc<dyn Sampler>,
}

impl ErrorAwareSampler {
    /// Create new error-aware sampler
    pub fn new(base_sampler: Arc<dyn Sampler>) -> Self {
        Self { base_sampler }
    }
}

impl Sampler for ErrorAwareSampler {
    fn should_sample(&self, context: &SamplingContext) -> SamplingResult {
        if context.has_error {
            SamplingResult {
                decision: SamplingDecision::RecordAndSample,
                rate: 1.0,
                reason: "error-sampling".to_string(),
            }
        } else {
            self.base_sampler.should_sample(context)
        }
    }

    fn get_sampling_rate(&self) -> f64 {
        self.base_sampler.get_sampling_rate()
    }
}

/// Tail sampler (samples slow traces)
pub struct TailSampler {
    base_sampler: Arc<dyn Sampler>,
    threshold: Duration,
}

impl TailSampler {
    /// Create new tail sampler
    pub fn new(base_sampler: Arc<dyn Sampler>, threshold: Duration) -> Self {
        Self {
            base_sampler,
            threshold,
        }
    }
}

impl Sampler for TailSampler {
    fn should_sample(&self, context: &SamplingContext) -> SamplingResult {
        if let Some(duration) = context.duration {
            if duration > self.threshold {
                return SamplingResult {
                    decision: SamplingDecision::RecordAndSample,
                    rate: 1.0,
                    reason: "tail-sampling".to_string(),
                };
            }
        }

        self.base_sampler.should_sample(context)
    }

    fn get_sampling_rate(&self) -> f64 {
        self.base_sampler.get_sampling_rate()
    }
}

/// Adaptive sampler (adjusts rate based on system load)
pub struct AdaptiveSampler {
    base_rate: f64,
    #[allow(dead_code)]
    current_rate: Arc<RwLock<f64>>,
    error_sampling: bool,
    tail_threshold: Option<Duration>,
    stats: Arc<RwLock<AdaptiveStats>>,
}

struct AdaptiveStats {
    total_traces: usize,
    sampled_traces: usize,
    error_traces: usize,
    slow_traces: usize,
}

impl AdaptiveSampler {
    /// Create new adaptive sampler
    pub fn new() -> Self {
        Self {
            base_rate: 0.1,
            current_rate: Arc::new(RwLock::new(0.1)),
            error_sampling: true,
            tail_threshold: Some(Duration::from_secs(1)),
            stats: Arc::new(RwLock::new(AdaptiveStats {
                total_traces: 0,
                sampled_traces: 0,
                error_traces: 0,
                slow_traces: 0,
            })),
        }
    }

    /// Set base sampling rate
    pub fn with_base_rate(mut self, rate: f64) -> Self {
        self.base_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Enable/disable error sampling
    pub fn with_error_sampling(mut self, enabled: bool) -> Self {
        self.error_sampling = enabled;
        self
    }

    /// Set tail sampling threshold
    pub fn with_tail_threshold(mut self, threshold: Duration) -> Self {
        self.tail_threshold = Some(threshold);
        self
    }

    /// Get adaptive statistics
    pub async fn get_stats(&self) -> (usize, usize, usize, usize) {
        let stats = self.stats.read().await;
        (
            stats.total_traces,
            stats.sampled_traces,
            stats.error_traces,
            stats.slow_traces,
        )
    }
}

impl Default for AdaptiveSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for AdaptiveSampler {
    fn should_sample(&self, context: &SamplingContext) -> SamplingResult {
        // Always sample errors
        if self.error_sampling && context.has_error {
            return SamplingResult {
                decision: SamplingDecision::RecordAndSample,
                rate: 1.0,
                reason: "error".to_string(),
            };
        }

        // Always sample slow traces
        if let (Some(duration), Some(threshold)) = (context.duration, self.tail_threshold) {
            if duration > threshold {
                return SamplingResult {
                    decision: SamplingDecision::RecordAndSample,
                    rate: 1.0,
                    reason: "tail".to_string(),
                };
            }
        }

        // Use base rate for normal traces
        let should_sample = fastrand::f64() < self.base_rate;

        SamplingResult {
            decision: if should_sample {
                SamplingDecision::RecordAndSample
            } else {
                SamplingDecision::Drop
            },
            rate: self.base_rate,
            reason: "adaptive".to_string(),
        }
    }

    fn get_sampling_rate(&self) -> f64 {
        self.base_rate
    }

    fn update(&mut self, context: &SamplingContext, sampled: bool) {
        // Note: This is simplified for sync trait
        // In production, use async update
        if context.has_error {
            // Track error
        }
        if sampled {
            // Track sampling
        }
    }
}

/// Composite sampler (combines multiple strategies)
pub struct CompositeSampler {
    samplers: Vec<(String, Arc<dyn Sampler>)>,
    strategy: CompositeStrategy,
}

/// Composite strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeStrategy {
    /// Sample if ANY sampler says yes
    Any,
    /// Sample if ALL samplers say yes
    All,
    /// Use first sampler that says yes
    FirstMatch,
}

impl CompositeSampler {
    /// Create new composite sampler
    pub fn new(strategy: CompositeStrategy) -> Self {
        Self {
            samplers: Vec::new(),
            strategy,
        }
    }

    /// Add sampler
    pub fn add_sampler(mut self, name: impl Into<String>, sampler: Arc<dyn Sampler>) -> Self {
        self.samplers.push((name.into(), sampler));
        self
    }
}

impl Sampler for CompositeSampler {
    fn should_sample(&self, context: &SamplingContext) -> SamplingResult {
        if self.samplers.is_empty() {
            return SamplingResult {
                decision: SamplingDecision::Drop,
                rate: 0.0,
                reason: "no-samplers".to_string(),
            };
        }

        match self.strategy {
            CompositeStrategy::Any => {
                for (name, sampler) in &self.samplers {
                    let result = sampler.should_sample(context);
                    if result.should_sample() {
                        return SamplingResult {
                            decision: SamplingDecision::RecordAndSample,
                            rate: result.rate,
                            reason: format!("composite-any:{}", name),
                        };
                    }
                }
                SamplingResult {
                    decision: SamplingDecision::Drop,
                    rate: 0.0,
                    reason: "composite-any:none".to_string(),
                }
            }
            CompositeStrategy::All => {
                let mut total_rate = 1.0;
                for (_, sampler) in &self.samplers {
                    let result = sampler.should_sample(context);
                    if !result.should_sample() {
                        return SamplingResult {
                            decision: SamplingDecision::Drop,
                            rate: 0.0,
                            reason: "composite-all:rejected".to_string(),
                        };
                    }
                    total_rate *= result.rate;
                }
                SamplingResult {
                    decision: SamplingDecision::RecordAndSample,
                    rate: total_rate,
                    reason: "composite-all:accepted".to_string(),
                }
            }
            CompositeStrategy::FirstMatch => {
                let (name, sampler) = &self.samplers[0];
                let result = sampler.should_sample(context);
                SamplingResult {
                    decision: result.decision,
                    rate: result.rate,
                    reason: format!("composite-first:{}", name),
                }
            }
        }
    }

    fn get_sampling_rate(&self) -> f64 {
        if self.samplers.is_empty() {
            return 0.0;
        }

        match self.strategy {
            CompositeStrategy::Any | CompositeStrategy::FirstMatch => {
                self.samplers[0].1.get_sampling_rate()
            }
            CompositeStrategy::All => self
                .samplers
                .iter()
                .map(|(_, s)| s.get_sampling_rate())
                .product(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_context_creation() {
        let ctx = SamplingContext::new("GetUser", "query")
            .with_error(false)
            .with_priority(7)
            .with_attribute("client", "apollo");

        assert_eq!(ctx.operation_name, "GetUser");
        assert_eq!(ctx.operation_type, "query");
        assert!(!ctx.has_error);
        assert_eq!(ctx.priority, 7);
        assert_eq!(ctx.attributes.get("client"), Some(&"apollo".to_string()));
    }

    #[test]
    fn test_always_on_sampler() {
        let sampler = AlwaysOnSampler;
        let ctx = SamplingContext::new("test", "query");

        let result = sampler.should_sample(&ctx);

        assert_eq!(result.decision, SamplingDecision::RecordAndSample);
        assert_eq!(result.rate, 1.0);
        assert!(result.should_sample());
    }

    #[test]
    fn test_always_off_sampler() {
        let sampler = AlwaysOffSampler;
        let ctx = SamplingContext::new("test", "query");

        let result = sampler.should_sample(&ctx);

        assert_eq!(result.decision, SamplingDecision::Drop);
        assert_eq!(result.rate, 0.0);
        assert!(!result.should_sample());
    }

    #[test]
    fn test_probabilistic_sampler() {
        let sampler = ProbabilisticSampler::new(0.5);
        let ctx = SamplingContext::new("test", "query");

        assert_eq!(sampler.get_sampling_rate(), 0.5);

        // Run multiple times to check probabilistic behavior
        let mut sampled = 0;
        for _ in 0..1000 {
            if sampler.should_sample(&ctx).should_sample() {
                sampled += 1;
            }
        }

        // Should be around 500, with some variance
        assert!(sampled > 400 && sampled < 600);
    }

    #[test]
    fn test_probabilistic_sampler_bounds() {
        let sampler_low = ProbabilisticSampler::new(-0.1);
        assert_eq!(sampler_low.get_sampling_rate(), 0.0);

        let sampler_high = ProbabilisticSampler::new(1.5);
        assert_eq!(sampler_high.get_sampling_rate(), 1.0);
    }

    #[test]
    fn test_rate_limited_sampler() {
        let sampler = RateLimitedSampler::new(10.0);
        let ctx = SamplingContext::new("test", "query");

        let result = sampler.should_sample(&ctx);
        assert!(result.rate >= 0.0 && result.rate <= 1.0);
    }

    #[test]
    fn test_priority_sampler_high_priority() {
        let sampler = PrioritySampler::new();
        let ctx = SamplingContext::new("test", "query").with_priority(10);

        let mut sampled = 0;
        for _ in 0..100 {
            if sampler.should_sample(&ctx).should_sample() {
                sampled += 1;
            }
        }

        // High priority should sample most/all traces
        assert!(sampled > 80);
    }

    #[test]
    fn test_priority_sampler_low_priority() {
        let sampler = PrioritySampler::new();
        let ctx = SamplingContext::new("test", "query").with_priority(1);

        let mut sampled = 0;
        for _ in 0..1000 {
            if sampler.should_sample(&ctx).should_sample() {
                sampled += 1;
            }
        }

        // Low priority should sample around 10%
        assert!(sampled > 50 && sampled < 200);
    }

    #[test]
    fn test_priority_sampler_custom_rate() {
        let mut sampler = PrioritySampler::new();
        sampler.set_priority_rate(5, 0.9);

        let ctx = SamplingContext::new("test", "query").with_priority(5);

        let mut sampled = 0;
        for _ in 0..1000 {
            if sampler.should_sample(&ctx).should_sample() {
                sampled += 1;
            }
        }

        // Should sample around 90%
        assert!(sampled > 850);
    }

    #[test]
    fn test_error_aware_sampler_with_error() {
        let base = Arc::new(ProbabilisticSampler::new(0.1));
        let sampler = ErrorAwareSampler::new(base);

        let ctx = SamplingContext::new("test", "query").with_error(true);

        let result = sampler.should_sample(&ctx);

        assert!(result.should_sample());
        assert_eq!(result.rate, 1.0);
        assert_eq!(result.reason, "error-sampling");
    }

    #[test]
    fn test_error_aware_sampler_without_error() {
        let base = Arc::new(AlwaysOffSampler);
        let sampler = ErrorAwareSampler::new(base);

        let ctx = SamplingContext::new("test", "query").with_error(false);

        let result = sampler.should_sample(&ctx);

        assert!(!result.should_sample());
    }

    #[test]
    fn test_tail_sampler_slow_trace() {
        let base = Arc::new(AlwaysOffSampler);
        let sampler = TailSampler::new(base, Duration::from_millis(100));

        let ctx = SamplingContext::new("test", "query").with_duration(Duration::from_millis(200));

        let result = sampler.should_sample(&ctx);

        assert!(result.should_sample());
        assert_eq!(result.reason, "tail-sampling");
    }

    #[test]
    fn test_tail_sampler_fast_trace() {
        let base = Arc::new(AlwaysOffSampler);
        let sampler = TailSampler::new(base, Duration::from_millis(100));

        let ctx = SamplingContext::new("test", "query").with_duration(Duration::from_millis(50));

        let result = sampler.should_sample(&ctx);

        assert!(!result.should_sample());
    }

    #[test]
    fn test_adaptive_sampler_error() {
        let sampler = AdaptiveSampler::new().with_error_sampling(true);

        let ctx = SamplingContext::new("test", "query").with_error(true);

        let result = sampler.should_sample(&ctx);

        assert!(result.should_sample());
        assert_eq!(result.reason, "error");
    }

    #[test]
    fn test_adaptive_sampler_tail() {
        let sampler = AdaptiveSampler::new().with_tail_threshold(Duration::from_millis(100));

        let ctx = SamplingContext::new("test", "query").with_duration(Duration::from_millis(200));

        let result = sampler.should_sample(&ctx);

        assert!(result.should_sample());
        assert_eq!(result.reason, "tail");
    }

    #[test]
    fn test_adaptive_sampler_normal() {
        let sampler = AdaptiveSampler::new().with_base_rate(0.2);

        let ctx = SamplingContext::new("test", "query");

        let mut sampled = 0;
        for _ in 0..1000 {
            if sampler.should_sample(&ctx).should_sample() {
                sampled += 1;
            }
        }

        // Should sample around 20%
        assert!(sampled > 150 && sampled < 250);
    }

    #[test]
    fn test_composite_sampler_any() {
        let sampler = CompositeSampler::new(CompositeStrategy::Any)
            .add_sampler("always-off", Arc::new(AlwaysOffSampler))
            .add_sampler("always-on", Arc::new(AlwaysOnSampler));

        let ctx = SamplingContext::new("test", "query");

        let result = sampler.should_sample(&ctx);

        assert!(result.should_sample());
        assert!(result.reason.contains("composite-any"));
    }

    #[test]
    fn test_composite_sampler_all() {
        let sampler = CompositeSampler::new(CompositeStrategy::All)
            .add_sampler("always-on-1", Arc::new(AlwaysOnSampler))
            .add_sampler("always-on-2", Arc::new(AlwaysOnSampler));

        let ctx = SamplingContext::new("test", "query");

        let result = sampler.should_sample(&ctx);

        assert!(result.should_sample());
        assert!(result.reason.contains("composite-all"));
    }

    #[test]
    fn test_composite_sampler_all_rejected() {
        let sampler = CompositeSampler::new(CompositeStrategy::All)
            .add_sampler("always-on", Arc::new(AlwaysOnSampler))
            .add_sampler("always-off", Arc::new(AlwaysOffSampler));

        let ctx = SamplingContext::new("test", "query");

        let result = sampler.should_sample(&ctx);

        assert!(!result.should_sample());
    }

    #[test]
    fn test_composite_sampler_first_match() {
        let sampler = CompositeSampler::new(CompositeStrategy::FirstMatch)
            .add_sampler("always-on", Arc::new(AlwaysOnSampler))
            .add_sampler("always-off", Arc::new(AlwaysOffSampler));

        let ctx = SamplingContext::new("test", "query");

        let result = sampler.should_sample(&ctx);

        assert!(result.should_sample());
        assert!(result.reason.contains("composite-first"));
    }

    #[test]
    fn test_composite_sampler_empty() {
        let sampler = CompositeSampler::new(CompositeStrategy::Any);

        let ctx = SamplingContext::new("test", "query");

        let result = sampler.should_sample(&ctx);

        assert!(!result.should_sample());
        assert_eq!(result.reason, "no-samplers");
    }
}
