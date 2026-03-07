//! Time-series alerting rules engine.
//!
//! An `AlertEngine` holds a set of `AlertRule`s.  Each time a metric
//! value is observed via `AlertEngine::evaluate`, the engine checks
//! threshold conditions, manages the pending/firing lifecycle (including
//! `duration_ms` hold-time), and emits `AlertEvent`s.

use std::collections::HashMap;

// ── Threshold operation ───────────────────────────────────────────────────────

/// Comparison operator for a threshold check.
#[derive(Debug, Clone, PartialEq)]
pub enum ThresholdOp {
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
    Equal,
    NotEqual,
}

/// Evaluate `value <op> threshold`.
fn threshold_check(value: f64, op: &ThresholdOp, threshold: f64) -> bool {
    match op {
        ThresholdOp::GreaterThan => value > threshold,
        ThresholdOp::LessThan => value < threshold,
        ThresholdOp::GreaterOrEqual => value >= threshold,
        ThresholdOp::LessOrEqual => value <= threshold,
        ThresholdOp::Equal => (value - threshold).abs() < f64::EPSILON,
        ThresholdOp::NotEqual => (value - threshold).abs() >= f64::EPSILON,
    }
}

// ── Severity ──────────────────────────────────────────────────────────────────

/// Alert severity level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

// ── AlertRule ─────────────────────────────────────────────────────────────────

/// A single alerting rule.
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Unique identifier for this rule.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Metric name this rule watches.
    pub metric: String,
    /// How to compare the observed value to `threshold`.
    pub op: ThresholdOp,
    /// The comparison threshold.
    pub threshold: f64,
    /// How long (ms) the condition must hold before the alert fires.
    /// Set to `0` for immediate firing.
    pub duration_ms: u64,
    /// Severity of this alert.
    pub severity: AlertSeverity,
}

impl AlertRule {
    /// Build a new rule.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        metric: impl Into<String>,
        op: ThresholdOp,
        threshold: f64,
        duration_ms: u64,
        severity: AlertSeverity,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            metric: metric.into(),
            op,
            threshold,
            duration_ms,
            severity,
        }
    }
}

// ── AlertState ────────────────────────────────────────────────────────────────

/// Lifecycle state of a rule, attached to the engine.
#[derive(Debug, Clone, PartialEq)]
pub enum AlertState {
    /// Condition not met (or cleared after firing).
    Inactive,
    /// Condition met but hold-time not yet elapsed.  Contains the timestamp
    /// (ms) when the condition first became true.
    Pending(u64),
    /// Alert is actively firing.  Contains the timestamp (ms) when it started
    /// firing.
    Firing(u64),
}

// ── AlertEvent ────────────────────────────────────────────────────────────────

/// Produced every time a rule transitions state.
#[derive(Debug, Clone)]
pub struct AlertEvent {
    /// ID of the rule that produced this event.
    pub rule_id: String,
    /// The metric value that triggered the event.
    pub value: f64,
    /// New state of the rule after this evaluation.
    pub state: AlertState,
    /// Unix timestamp (ms) of the evaluation.
    pub timestamp: u64,
}

// ── AlertEngine ───────────────────────────────────────────────────────────────

/// Manages alert rules and their current lifecycle states.
pub struct AlertEngine {
    rules: HashMap<String, AlertRule>,
    states: HashMap<String, AlertState>,
    history: Vec<AlertEvent>,
}

impl AlertEngine {
    /// Create a new empty engine.
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            states: HashMap::new(),
            history: Vec::new(),
        }
    }

    /// Register an alert rule.  Overwrites any existing rule with the same ID.
    pub fn add_rule(&mut self, rule: AlertRule) {
        self.states
            .entry(rule.id.clone())
            .or_insert(AlertState::Inactive);
        self.rules.insert(rule.id.clone(), rule);
    }

    /// Evaluate all rules that watch `metric` with the given `value` at `now_ms`.
    ///
    /// Returns the list of state-change events produced this tick.
    pub fn evaluate(&mut self, metric: &str, value: f64, now_ms: u64) -> Vec<AlertEvent> {
        // Collect matching rule IDs first to avoid borrow-checker issues.
        let rule_ids: Vec<String> = self
            .rules
            .values()
            .filter(|r| r.metric == metric)
            .map(|r| r.id.clone())
            .collect();

        let mut events = Vec::new();

        for rule_id in rule_ids {
            let rule = match self.rules.get(&rule_id) {
                Some(r) => r.clone(),
                None => continue,
            };
            let current_state = self
                .states
                .get(&rule_id)
                .cloned()
                .unwrap_or(AlertState::Inactive);

            let condition_met = threshold_check(value, &rule.op, rule.threshold);

            let new_state = match &current_state {
                AlertState::Inactive => {
                    if condition_met {
                        if rule.duration_ms == 0 {
                            AlertState::Firing(now_ms)
                        } else {
                            AlertState::Pending(now_ms)
                        }
                    } else {
                        AlertState::Inactive
                    }
                }
                AlertState::Pending(since) => {
                    if condition_met {
                        let elapsed = now_ms.saturating_sub(*since);
                        if elapsed >= rule.duration_ms {
                            AlertState::Firing(*since)
                        } else {
                            AlertState::Pending(*since)
                        }
                    } else {
                        // Condition no longer met — reset.
                        AlertState::Inactive
                    }
                }
                AlertState::Firing(since) => {
                    if condition_met {
                        AlertState::Firing(*since) // still firing
                    } else {
                        AlertState::Inactive // resolved
                    }
                }
            };

            if new_state != current_state {
                let event = AlertEvent {
                    rule_id: rule_id.clone(),
                    value,
                    state: new_state.clone(),
                    timestamp: now_ms,
                };
                events.push(event.clone());
                self.history.push(event);
                self.states.insert(rule_id, new_state);
            }
        }

        events
    }

    /// Return references to all rules whose current state is `Firing`.
    pub fn firing_alerts(&self) -> Vec<(&AlertRule, &AlertState)> {
        self.rules
            .values()
            .filter_map(|rule| {
                let state = self.states.get(&rule.id)?;
                if matches!(state, AlertState::Firing(_)) {
                    Some((rule, state))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return the current state of a specific rule.
    pub fn rule_state(&self, rule_id: &str) -> Option<&AlertState> {
        self.states.get(rule_id)
    }

    /// Full event history (all state changes since engine creation).
    pub fn history(&self) -> &[AlertEvent] {
        &self.history
    }

    /// Reset a rule to `Inactive` and clear its pending/firing state.
    pub fn clear_rule(&mut self, rule_id: &str) {
        self.states
            .insert(rule_id.to_string(), AlertState::Inactive);
    }

    /// Number of rules registered.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

impl Default for AlertEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn gt_rule(id: &str, metric: &str, threshold: f64) -> AlertRule {
        AlertRule::new(
            id,
            "test-rule",
            metric,
            ThresholdOp::GreaterThan,
            threshold,
            0, // immediate
            AlertSeverity::Warning,
        )
    }

    fn gt_rule_with_duration(id: &str, metric: &str, threshold: f64, dur_ms: u64) -> AlertRule {
        AlertRule::new(
            id,
            "dur-rule",
            metric,
            ThresholdOp::GreaterThan,
            threshold,
            dur_ms,
            AlertSeverity::Critical,
        )
    }

    // ── threshold_check ───────────────────────────────────────────────────────

    #[test]
    fn test_threshold_greater_than_true() {
        assert!(threshold_check(10.0, &ThresholdOp::GreaterThan, 5.0));
    }

    #[test]
    fn test_threshold_greater_than_false() {
        assert!(!threshold_check(3.0, &ThresholdOp::GreaterThan, 5.0));
    }

    #[test]
    fn test_threshold_less_than() {
        assert!(threshold_check(1.0, &ThresholdOp::LessThan, 5.0));
    }

    #[test]
    fn test_threshold_greater_or_equal_at_boundary() {
        assert!(threshold_check(5.0, &ThresholdOp::GreaterOrEqual, 5.0));
    }

    #[test]
    fn test_threshold_less_or_equal_at_boundary() {
        assert!(threshold_check(5.0, &ThresholdOp::LessOrEqual, 5.0));
    }

    #[test]
    fn test_threshold_equal() {
        assert!(threshold_check(7.0, &ThresholdOp::Equal, 7.0));
    }

    #[test]
    fn test_threshold_not_equal() {
        assert!(threshold_check(8.0, &ThresholdOp::NotEqual, 7.0));
    }

    #[test]
    fn test_threshold_not_equal_false_when_same() {
        assert!(!threshold_check(7.0, &ThresholdOp::NotEqual, 7.0));
    }

    // ── add_rule / rule_count ─────────────────────────────────────────────────

    #[test]
    fn test_new_engine_empty() {
        let e = AlertEngine::new();
        assert_eq!(e.rule_count(), 0);
    }

    #[test]
    fn test_add_rule_increments_count() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 80.0));
        assert_eq!(e.rule_count(), 1);
    }

    #[test]
    fn test_add_multiple_rules() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 80.0));
        e.add_rule(gt_rule("r2", "mem", 90.0));
        assert_eq!(e.rule_count(), 2);
    }

    #[test]
    fn test_new_rule_state_inactive() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 80.0));
        assert_eq!(e.rule_state("r1"), Some(&AlertState::Inactive));
    }

    // ── evaluate → immediate fire ──────────────────────────────────────────────

    #[test]
    fn test_evaluate_below_threshold_no_event() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 80.0));
        let events = e.evaluate("cpu", 50.0, 1000);
        assert!(events.is_empty());
    }

    #[test]
    fn test_evaluate_above_threshold_fires_immediately() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 80.0));
        let events = e.evaluate("cpu", 90.0, 1000);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].state, AlertState::Firing(_)));
    }

    #[test]
    fn test_evaluate_fires_state_changes() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 80.0));
        e.evaluate("cpu", 90.0, 1000);
        assert!(matches!(e.rule_state("r1"), Some(AlertState::Firing(_))));
    }

    #[test]
    fn test_evaluate_resolved_after_below_threshold() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 80.0));
        e.evaluate("cpu", 90.0, 1000); // fires
        let events = e.evaluate("cpu", 50.0, 2000); // resolves
        assert!(!events.is_empty());
        assert_eq!(e.rule_state("r1"), Some(&AlertState::Inactive));
    }

    #[test]
    fn test_evaluate_no_duplicate_event_while_firing() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 80.0));
        e.evaluate("cpu", 90.0, 1000);
        let events = e.evaluate("cpu", 90.0, 2000); // still above threshold
                                                    // State did not change (already Firing) — no new event.
        assert!(events.is_empty());
    }

    // ── evaluate → pending / duration ─────────────────────────────────────────

    #[test]
    fn test_evaluate_pending_before_duration() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule_with_duration("r1", "cpu", 80.0, 5000));
        e.evaluate("cpu", 90.0, 1000); // transitions to Pending
        assert!(matches!(e.rule_state("r1"), Some(AlertState::Pending(_))));
    }

    #[test]
    fn test_evaluate_fires_after_duration() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule_with_duration("r1", "cpu", 80.0, 5000));
        e.evaluate("cpu", 90.0, 1000); // Pending at t=1000
        e.evaluate("cpu", 90.0, 6001); // 5001ms elapsed → Firing
        assert!(matches!(e.rule_state("r1"), Some(AlertState::Firing(_))));
    }

    #[test]
    fn test_evaluate_pending_reset_if_condition_clears() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule_with_duration("r1", "cpu", 80.0, 5000));
        e.evaluate("cpu", 90.0, 1000); // Pending
        e.evaluate("cpu", 50.0, 2000); // Condition clears → Inactive
        assert_eq!(e.rule_state("r1"), Some(&AlertState::Inactive));
    }

    // ── firing_alerts ─────────────────────────────────────────────────────────

    #[test]
    fn test_firing_alerts_empty_initially() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 80.0));
        assert!(e.firing_alerts().is_empty());
    }

    #[test]
    fn test_firing_alerts_contains_firing_rule() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 80.0));
        e.evaluate("cpu", 90.0, 1000);
        let fa = e.firing_alerts();
        assert_eq!(fa.len(), 1);
        assert_eq!(fa[0].0.id, "r1");
    }

    #[test]
    fn test_firing_alerts_excludes_inactive() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 80.0));
        e.add_rule(gt_rule("r2", "mem", 90.0));
        e.evaluate("cpu", 90.0, 1000); // r1 fires
                                       // r2 never exceeds threshold
        let fa = e.firing_alerts();
        assert_eq!(fa.len(), 1);
    }

    // ── history ───────────────────────────────────────────────────────────────

    #[test]
    fn test_history_empty_initially() {
        let e = AlertEngine::new();
        assert!(e.history().is_empty());
    }

    #[test]
    fn test_history_grows_with_events() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 80.0));
        e.evaluate("cpu", 90.0, 1000);
        e.evaluate("cpu", 50.0, 2000);
        assert_eq!(e.history().len(), 2);
    }

    #[test]
    fn test_history_event_has_value() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 80.0));
        e.evaluate("cpu", 99.0, 1000);
        let ev = &e.history()[0];
        assert!((ev.value - 99.0).abs() < 0.001);
    }

    // ── clear_rule ────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_rule_resets_to_inactive() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 80.0));
        e.evaluate("cpu", 90.0, 1000);
        e.clear_rule("r1");
        assert_eq!(e.rule_state("r1"), Some(&AlertState::Inactive));
    }

    #[test]
    fn test_clear_nonexistent_rule_noop() {
        let mut e = AlertEngine::new();
        e.clear_rule("nonexistent"); // should not panic
    }

    // ── unrelated metric ──────────────────────────────────────────────────────

    #[test]
    fn test_unrelated_metric_no_events() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 80.0));
        let events = e.evaluate("memory", 99.0, 1000); // wrong metric
        assert!(events.is_empty());
    }

    // ── default ───────────────────────────────────────────────────────────────

    #[test]
    fn test_default_engine_empty() {
        let e = AlertEngine::default();
        assert_eq!(e.rule_count(), 0);
    }

    // ── severity preserved ────────────────────────────────────────────────────

    #[test]
    fn test_severity_critical_preserved() {
        let rule = AlertRule::new(
            "r1",
            "n",
            "m",
            ThresholdOp::GreaterThan,
            1.0,
            0,
            AlertSeverity::Critical,
        );
        assert_eq!(rule.severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_severity_info_preserved() {
        let rule = AlertRule::new(
            "r1",
            "n",
            "m",
            ThresholdOp::GreaterThan,
            1.0,
            0,
            AlertSeverity::Info,
        );
        assert_eq!(rule.severity, AlertSeverity::Info);
    }

    // ── multiple rules same metric ────────────────────────────────────────────

    #[test]
    fn test_multiple_rules_same_metric_both_fire() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 70.0));
        e.add_rule(gt_rule("r2", "cpu", 60.0));
        let events = e.evaluate("cpu", 90.0, 1000);
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_multiple_rules_partial_fire() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 95.0)); // not exceeded
        e.add_rule(gt_rule("r2", "cpu", 60.0)); // exceeded
        let events = e.evaluate("cpu", 80.0, 1000);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].rule_id, "r2");
    }

    // ── Additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_threshold_less_or_equal_below() {
        assert!(threshold_check(3.0, &ThresholdOp::LessOrEqual, 5.0));
    }

    #[test]
    fn test_threshold_greater_or_equal_above() {
        assert!(threshold_check(6.0, &ThresholdOp::GreaterOrEqual, 5.0));
    }

    #[test]
    fn test_alert_rule_fields() {
        let rule = gt_rule("r1", "metric_x", 42.0);
        assert_eq!(rule.id, "r1");
        assert_eq!(rule.metric, "metric_x");
        assert!((rule.threshold - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_alert_state_firing_contains_timestamp() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 80.0));
        e.evaluate("cpu", 90.0, 5555);
        match e.rule_state("r1") {
            Some(AlertState::Firing(ts)) => assert_eq!(*ts, 5555),
            _ => panic!("expected Firing"),
        }
    }

    #[test]
    fn test_alert_state_pending_contains_timestamp() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule_with_duration("r1", "cpu", 80.0, 5000));
        e.evaluate("cpu", 90.0, 1234);
        match e.rule_state("r1") {
            Some(AlertState::Pending(ts)) => assert_eq!(*ts, 1234),
            _ => panic!("expected Pending"),
        }
    }

    #[test]
    fn test_event_rule_id_matches() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("my_rule", "mem", 50.0));
        let events = e.evaluate("mem", 60.0, 1000);
        assert_eq!(events[0].rule_id, "my_rule");
    }

    #[test]
    fn test_history_records_resolution_event() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 80.0));
        e.evaluate("cpu", 90.0, 1000); // fires
        e.evaluate("cpu", 50.0, 2000); // resolves
                                       // history should have: Firing event + Inactive event
        assert_eq!(e.history().len(), 2);
        assert!(matches!(e.history()[1].state, AlertState::Inactive));
    }

    #[test]
    fn test_rule_count_after_overwrite() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 80.0));
        e.add_rule(gt_rule("r1", "cpu", 90.0)); // overwrite
        assert_eq!(e.rule_count(), 1);
    }

    #[test]
    fn test_evaluate_no_rules_no_events() {
        let mut e = AlertEngine::new();
        let events = e.evaluate("cpu", 99.0, 1000);
        assert!(events.is_empty());
    }

    #[test]
    fn test_firing_alerts_clears_after_resolve() {
        let mut e = AlertEngine::new();
        e.add_rule(gt_rule("r1", "cpu", 80.0));
        e.evaluate("cpu", 90.0, 1000); // fires
        e.evaluate("cpu", 50.0, 2000); // resolves
        assert!(e.firing_alerts().is_empty());
    }

    #[test]
    fn test_threshold_check_not_equal_different() {
        assert!(threshold_check(5.1, &ThresholdOp::NotEqual, 5.0));
    }
}
