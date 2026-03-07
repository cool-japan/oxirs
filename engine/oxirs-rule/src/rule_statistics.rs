//! Rule execution statistics and profiling.
//!
//! This module provides per-rule and aggregate statistics collection for
//! rule-based reasoning engines. Statistics include activation counts,
//! cumulative execution time, and inference throughput.

use std::collections::HashMap;

/// Per-rule execution statistics.
///
/// Tracks how many times a rule has fired, the cumulative CPU time spent,
/// the number of new inferences produced, and the timestamp of the last
/// activation.
#[derive(Debug, Clone, Default)]
pub struct RuleStats {
    /// Unique identifier for the rule.
    pub rule_id: String,
    /// Total number of times this rule has been activated.
    pub activation_count: u64,
    /// Cumulative execution duration across all activations, in microseconds.
    pub total_duration_us: u64,
    /// Total number of new facts / inferences produced by this rule.
    pub inferences_produced: u64,
    /// Timestamp (in microseconds since epoch or since start) of the last activation.
    pub last_activation_us: u64,
}

impl RuleStats {
    /// Creates a new `RuleStats` instance for the given rule ID.
    pub fn new(rule_id: impl Into<String>) -> Self {
        Self {
            rule_id: rule_id.into(),
            ..Default::default()
        }
    }

    /// Returns the average duration per activation in microseconds.
    ///
    /// Returns `0.0` if the rule has never been activated.
    pub fn avg_duration_us(&self) -> f64 {
        if self.activation_count == 0 {
            0.0
        } else {
            self.total_duration_us as f64 / self.activation_count as f64
        }
    }

    /// Returns the inference throughput in inferences per second.
    ///
    /// Calculated as `inferences_produced / (total_duration_us / 1_000_000)`.
    /// Returns `0.0` if total duration is zero.
    pub fn throughput_per_sec(&self) -> f64 {
        if self.total_duration_us == 0 {
            0.0
        } else {
            let secs = self.total_duration_us as f64 / 1_000_000.0;
            self.inferences_produced as f64 / secs
        }
    }

    /// Records a single activation, updating cumulative counters.
    pub fn record_activation(&mut self, duration_us: u64, inferences: u64, timestamp_us: u64) {
        self.activation_count += 1;
        self.total_duration_us += duration_us;
        self.inferences_produced += inferences;
        self.last_activation_us = timestamp_us;
    }
}

/// Aggregate statistics collector across all rules in a rule engine.
///
/// Provides methods to record individual rule activations, query per-rule
/// statistics, and compute aggregate summaries such as top rules by
/// activation count or cumulative duration.
pub struct RuleStatisticsCollector {
    stats: HashMap<String, RuleStats>,
    total_cycles: u64,
}

impl RuleStatisticsCollector {
    /// Creates a new, empty statistics collector.
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            total_cycles: 0,
        }
    }

    /// Records a single rule activation.
    ///
    /// If the rule has not been seen before, a new entry is created.
    /// The `timestamp_us` can be any monotonic timestamp (e.g., microseconds
    /// since engine start).
    pub fn record(&mut self, rule_id: &str, duration_us: u64, inferences: u64) {
        let entry = self
            .stats
            .entry(rule_id.to_string())
            .or_insert_with(|| RuleStats::new(rule_id));
        // Use activation_count as a simple monotonic timestamp proxy.
        let ts = entry.activation_count;
        entry.record_activation(duration_us, inferences, ts);
    }

    /// Increments the rule-engine cycle counter by one.
    pub fn increment_cycle(&mut self) {
        self.total_cycles += 1;
    }

    /// Returns a reference to the statistics for a specific rule, or `None`.
    pub fn get(&self, rule_id: &str) -> Option<&RuleStats> {
        self.stats.get(rule_id)
    }

    /// Returns references to all collected rule statistics.
    pub fn all(&self) -> Vec<&RuleStats> {
        self.stats.values().collect()
    }

    /// Returns the top `n` rules by activation count (descending).
    pub fn top_by_activations(&self, n: usize) -> Vec<&RuleStats> {
        let mut sorted: Vec<&RuleStats> = self.stats.values().collect();
        sorted.sort_by(|a, b| b.activation_count.cmp(&a.activation_count));
        sorted.truncate(n);
        sorted
    }

    /// Returns the top `n` rules by total cumulative duration (descending).
    pub fn top_by_duration(&self, n: usize) -> Vec<&RuleStats> {
        let mut sorted: Vec<&RuleStats> = self.stats.values().collect();
        sorted.sort_by(|a, b| b.total_duration_us.cmp(&a.total_duration_us));
        sorted.truncate(n);
        sorted
    }

    /// Returns the sum of all inferences produced across all rules.
    pub fn total_inferences(&self) -> u64 {
        self.stats.values().map(|s| s.inferences_produced).sum()
    }

    /// Returns the number of rule-engine reasoning cycles performed.
    pub fn total_cycles(&self) -> u64 {
        self.total_cycles
    }

    /// Resets all statistics and the cycle counter to zero.
    pub fn reset(&mut self) {
        self.stats.clear();
        self.total_cycles = 0;
    }

    /// Returns the number of distinct rules tracked.
    pub fn rule_count(&self) -> usize {
        self.stats.len()
    }

    /// Returns `true` if no statistics have been recorded.
    pub fn is_empty(&self) -> bool {
        self.stats.is_empty()
    }

    /// Returns the rule with the highest activation count, or `None` if empty.
    pub fn hottest_rule(&self) -> Option<&RuleStats> {
        self.stats.values().max_by_key(|s| s.activation_count)
    }

    /// Returns the total cumulative duration across all rules in microseconds.
    pub fn total_duration_us(&self) -> u64 {
        self.stats.values().map(|s| s.total_duration_us).sum()
    }
}

impl Default for RuleStatisticsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_collector() -> RuleStatisticsCollector {
        RuleStatisticsCollector::new()
    }

    // --- RuleStats unit tests ---

    #[test]
    fn test_rule_stats_default() {
        let s = RuleStats::default();
        assert_eq!(s.activation_count, 0);
        assert_eq!(s.total_duration_us, 0);
        assert_eq!(s.inferences_produced, 0);
    }

    #[test]
    fn test_rule_stats_new() {
        let s = RuleStats::new("rule-1");
        assert_eq!(s.rule_id, "rule-1");
        assert_eq!(s.activation_count, 0);
    }

    #[test]
    fn test_avg_duration_zero_activations() {
        let s = RuleStats::new("r");
        assert_eq!(s.avg_duration_us(), 0.0);
    }

    #[test]
    fn test_avg_duration_single_activation() {
        let mut s = RuleStats::new("r");
        s.record_activation(100, 5, 0);
        assert!((s.avg_duration_us() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_avg_duration_multiple_activations() {
        let mut s = RuleStats::new("r");
        s.record_activation(200, 1, 0);
        s.record_activation(400, 1, 1);
        // avg = (200+400)/2 = 300
        assert!((s.avg_duration_us() - 300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_throughput_zero_duration() {
        let mut s = RuleStats::new("r");
        s.activation_count = 1;
        s.inferences_produced = 100;
        // total_duration_us = 0 → throughput = 0.0
        assert_eq!(s.throughput_per_sec(), 0.0);
    }

    #[test]
    fn test_throughput_non_zero() {
        let mut s = RuleStats::new("r");
        s.record_activation(1_000_000, 50, 0); // 1 second, 50 inferences
                                               // throughput = 50 / 1.0 = 50.0
        assert!((s.throughput_per_sec() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_throughput_half_second() {
        let mut s = RuleStats::new("r");
        s.record_activation(500_000, 100, 0); // 0.5 seconds, 100 inferences
                                              // throughput = 100 / 0.5 = 200.0
        assert!((s.throughput_per_sec() - 200.0).abs() < 1e-4);
    }

    #[test]
    fn test_record_activation_increments_all_fields() {
        let mut s = RuleStats::new("r");
        s.record_activation(50, 3, 999);
        assert_eq!(s.activation_count, 1);
        assert_eq!(s.total_duration_us, 50);
        assert_eq!(s.inferences_produced, 3);
        assert_eq!(s.last_activation_us, 999);
    }

    // --- RuleStatisticsCollector tests ---

    #[test]
    fn test_new_is_empty() {
        let c = make_collector();
        assert!(c.is_empty());
        assert_eq!(c.rule_count(), 0);
        assert_eq!(c.total_inferences(), 0);
        assert_eq!(c.total_cycles(), 0);
    }

    #[test]
    fn test_record_single_rule() {
        let mut c = make_collector();
        c.record("rule-a", 100, 5);
        let s = c.get("rule-a").expect("should exist");
        assert_eq!(s.activation_count, 1);
        assert_eq!(s.total_duration_us, 100);
        assert_eq!(s.inferences_produced, 5);
    }

    #[test]
    fn test_record_same_rule_twice() {
        let mut c = make_collector();
        c.record("rule-a", 100, 5);
        c.record("rule-a", 200, 10);
        let s = c.get("rule-a").expect("should exist");
        assert_eq!(s.activation_count, 2);
        assert_eq!(s.total_duration_us, 300);
        assert_eq!(s.inferences_produced, 15);
    }

    #[test]
    fn test_record_multiple_rules() {
        let mut c = make_collector();
        c.record("rule-a", 100, 5);
        c.record("rule-b", 200, 10);
        c.record("rule-c", 300, 15);
        assert_eq!(c.rule_count(), 3);
    }

    #[test]
    fn test_get_nonexistent_rule() {
        let c = make_collector();
        assert!(c.get("no-such-rule").is_none());
    }

    #[test]
    fn test_all_returns_all_rules() {
        let mut c = make_collector();
        c.record("r1", 10, 1);
        c.record("r2", 20, 2);
        let all = c.all();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_all_empty() {
        let c = make_collector();
        assert!(c.all().is_empty());
    }

    #[test]
    fn test_top_by_activations_ordering() {
        let mut c = make_collector();
        c.record("slow", 100, 1);
        c.record("slow", 100, 1); // 2 activations
        c.record("fast", 50, 1); // 1 activation
        let top = c.top_by_activations(2);
        assert_eq!(top[0].rule_id, "slow");
    }

    #[test]
    fn test_top_by_activations_limit() {
        let mut c = make_collector();
        for i in 0..10u64 {
            c.record(&format!("rule-{}", i), 10 * i, i);
        }
        let top = c.top_by_activations(3);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_top_by_activations_n_larger_than_count() {
        let mut c = make_collector();
        c.record("only", 10, 1);
        let top = c.top_by_activations(100);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_top_by_duration_ordering() {
        let mut c = make_collector();
        c.record("cheap", 10, 1);
        c.record("expensive", 9999, 1);
        let top = c.top_by_duration(2);
        assert_eq!(top[0].rule_id, "expensive");
    }

    #[test]
    fn test_top_by_duration_limit() {
        let mut c = make_collector();
        c.record("r1", 100, 1);
        c.record("r2", 200, 1);
        c.record("r3", 300, 1);
        let top = c.top_by_duration(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].rule_id, "r3");
    }

    #[test]
    fn test_total_inferences_sum() {
        let mut c = make_collector();
        c.record("r1", 10, 3);
        c.record("r2", 20, 7);
        c.record("r1", 10, 5); // additional for r1
        assert_eq!(c.total_inferences(), 15);
    }

    #[test]
    fn test_total_inferences_empty() {
        let c = make_collector();
        assert_eq!(c.total_inferences(), 0);
    }

    #[test]
    fn test_increment_cycle() {
        let mut c = make_collector();
        c.increment_cycle();
        c.increment_cycle();
        assert_eq!(c.total_cycles(), 2);
    }

    #[test]
    fn test_increment_cycle_many() {
        let mut c = make_collector();
        for _ in 0..100 {
            c.increment_cycle();
        }
        assert_eq!(c.total_cycles(), 100);
    }

    #[test]
    fn test_reset_clears_all() {
        let mut c = make_collector();
        c.record("r1", 100, 5);
        c.record("r2", 200, 10);
        c.increment_cycle();
        c.increment_cycle();
        c.reset();
        assert!(c.is_empty());
        assert_eq!(c.total_cycles(), 0);
        assert_eq!(c.total_inferences(), 0);
    }

    #[test]
    fn test_reset_then_reuse() {
        let mut c = make_collector();
        c.record("r1", 100, 5);
        c.reset();
        c.record("r2", 50, 3);
        assert_eq!(c.rule_count(), 1);
        assert_eq!(c.total_inferences(), 3);
    }

    #[test]
    fn test_hottest_rule_empty() {
        let c = make_collector();
        assert!(c.hottest_rule().is_none());
    }

    #[test]
    fn test_hottest_rule_single() {
        let mut c = make_collector();
        c.record("only", 10, 1);
        let h = c.hottest_rule().expect("should have hottest");
        assert_eq!(h.rule_id, "only");
    }

    #[test]
    fn test_hottest_rule_multiple() {
        let mut c = make_collector();
        c.record("cold", 10, 1);
        c.record("hot", 100, 1);
        c.record("hot", 100, 1);
        c.record("hot", 100, 1);
        let h = c.hottest_rule().expect("must exist");
        assert_eq!(h.rule_id, "hot");
    }

    #[test]
    fn test_total_duration_us() {
        let mut c = make_collector();
        c.record("r1", 150, 1);
        c.record("r2", 250, 1);
        assert_eq!(c.total_duration_us(), 400);
    }

    #[test]
    fn test_default_constructor() {
        let c = RuleStatisticsCollector::default();
        assert!(c.is_empty());
    }

    #[test]
    fn test_top_by_activations_empty() {
        let c = make_collector();
        assert!(c.top_by_activations(5).is_empty());
    }

    #[test]
    fn test_top_by_duration_empty() {
        let c = make_collector();
        assert!(c.top_by_duration(5).is_empty());
    }

    #[test]
    fn test_avg_duration_three_activations() {
        let mut s = RuleStats::new("r");
        s.record_activation(100, 0, 0);
        s.record_activation(200, 0, 1);
        s.record_activation(300, 0, 2);
        // avg = 600/3 = 200
        assert!((s.avg_duration_us() - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_throughput_multiple_activations() {
        let mut s = RuleStats::new("r");
        s.record_activation(1_000_000, 30, 0); // 1 sec → 30/sec
        s.record_activation(1_000_000, 70, 1); // 1 sec → 70/sec
                                               // total: 2 sec, 100 inferences → 50/sec
        assert!((s.throughput_per_sec() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_rule_count_after_multiple_records() {
        let mut c = make_collector();
        c.record("a", 10, 1);
        c.record("b", 20, 2);
        c.record("a", 30, 3); // same rule again
        assert_eq!(c.rule_count(), 2);
    }

    #[test]
    fn test_record_zero_inferences() {
        let mut c = make_collector();
        c.record("r", 100, 0);
        let s = c.get("r").expect("must exist");
        assert_eq!(s.inferences_produced, 0);
        assert_eq!(s.activation_count, 1);
    }

    #[test]
    fn test_record_zero_duration() {
        let mut c = make_collector();
        c.record("r", 0, 5);
        let s = c.get("r").expect("must exist");
        assert_eq!(s.total_duration_us, 0);
        assert_eq!(s.avg_duration_us(), 0.0);
    }

    #[test]
    fn test_all_returns_correct_rule_ids() {
        let mut c = make_collector();
        c.record("rule-alpha", 10, 1);
        c.record("rule-beta", 20, 2);
        let ids: std::collections::HashSet<&str> =
            c.all().iter().map(|s| s.rule_id.as_str()).collect();
        assert!(ids.contains("rule-alpha"));
        assert!(ids.contains("rule-beta"));
    }

    #[test]
    fn test_top_by_activations_ties() {
        let mut c = make_collector();
        c.record("r1", 10, 1);
        c.record("r2", 10, 1);
        // Both have 1 activation — top 1 returns one of them
        let top = c.top_by_activations(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_top_by_duration_zero_elements_requested() {
        let mut c = make_collector();
        c.record("r1", 100, 1);
        assert!(c.top_by_duration(0).is_empty());
    }

    #[test]
    fn test_total_duration_empty() {
        let c = make_collector();
        assert_eq!(c.total_duration_us(), 0);
    }

    #[test]
    fn test_hottest_rule_after_reset() {
        let mut c = make_collector();
        c.record("hot", 10, 1);
        c.record("hot", 10, 1);
        c.reset();
        assert!(c.hottest_rule().is_none());
    }

    #[test]
    fn test_cycles_reset_to_zero() {
        let mut c = make_collector();
        c.increment_cycle();
        c.increment_cycle();
        c.reset();
        assert_eq!(c.total_cycles(), 0);
    }
}
