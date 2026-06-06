//! Per-endpoint running statistics using Exponential Weighted Moving Average (EWMA).

use std::time::Instant;

/// Per-endpoint running statistics.
///
/// All latency/error figures are maintained as exponential weighted moving
/// averages (EWMA) so that recent observations receive higher weight.
#[derive(Debug, Clone)]
pub struct EndpointStats {
    /// EWMA of request latency in milliseconds.
    pub ewma_latency_ms: f64,

    /// EWMA error rate in [0.0, 1.0].
    /// 0.0 = no recent errors; 1.0 = all recent requests failed.
    pub ewma_error_rate: f64,

    /// Total queries routed to this endpoint since creation.
    pub total_queries: u64,

    /// Total errors observed on this endpoint since creation.
    pub total_errors: u64,

    /// Timestamp of the most recent observation (success or failure).
    pub last_seen: Instant,
}

impl EndpointStats {
    /// Initialise with neutral defaults: 100 ms latency, 0 % error rate.
    pub fn new() -> Self {
        Self {
            ewma_latency_ms: 100.0,
            ewma_error_rate: 0.0,
            total_queries: 0,
            total_errors: 0,
            last_seen: Instant::now(),
        }
    }

    /// Record a successful query round-trip.
    ///
    /// Updates EWMA latency using the supplied `alpha` decay factor and
    /// gently drives the error rate toward zero.
    ///
    /// # Arguments
    /// * `latency_ms` — observed round-trip latency in milliseconds.
    /// * `alpha` — EWMA decay factor in (0, 1).  Larger values weight recent
    ///   observations more heavily.
    pub fn update_success(&mut self, latency_ms: f64, alpha: f64) {
        // EWMA update: new = alpha * observation + (1 - alpha) * current
        self.ewma_latency_ms = alpha * latency_ms + (1.0 - alpha) * self.ewma_latency_ms;
        // Drive error rate toward 0 on success
        self.ewma_error_rate *= 1.0 - alpha;
        self.total_queries += 1;
        self.last_seen = Instant::now();
    }

    /// Record a query failure.
    ///
    /// Updates EWMA error rate toward 1.0 using `alpha`.  Latency is left
    /// unchanged (failure duration is often a timeout, not representative).
    ///
    /// # Arguments
    /// * `alpha` — EWMA decay factor in (0, 1).
    pub fn update_failure(&mut self, alpha: f64) {
        // Drive error rate toward 1.0 on failure
        self.ewma_error_rate = alpha + (1.0 - alpha) * self.ewma_error_rate;
        self.total_queries += 1;
        self.total_errors += 1;
        self.last_seen = Instant::now();
    }

    /// Derived composite availability score in (0, 1].
    ///
    /// Formula: `(1 − error_rate) / (1 + latency_ms / 1000)`
    ///
    /// A healthy endpoint with zero errors and 0 ms latency returns 1.0.
    /// A high-error or high-latency endpoint approaches 0.0.
    pub fn availability_score(&self) -> f64 {
        (1.0 - self.ewma_error_rate) / (1.0 + self.ewma_latency_ms / 1000.0)
    }
}

impl Default for EndpointStats {
    fn default() -> Self {
        Self::new()
    }
}
