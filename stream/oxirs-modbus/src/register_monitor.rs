//! # Modbus Register Monitor
//!
//! Monitors Modbus registers against configurable threshold conditions and
//! emits alert events with cooldown support to avoid alert storms.
//!
//! ## Features
//!
//! - Multiple threshold conditions: Above, Below, Equal, NotEqual, InRange, OutOfRange, Changed
//! - Per-rule cooldown to suppress repeated alerts
//! - Supports Coil, Discrete Input, Holding Register, and Input Register address types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ─────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────

/// Errors returned by the register monitor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MonitorError {
    /// A rule with this ID already exists.
    DuplicateRuleId(String),
    /// No rule found with the given ID.
    RuleNotFound(String),
}

impl fmt::Display for MonitorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateRuleId(id) => write!(f, "duplicate monitor rule id: {id}"),
            Self::RuleNotFound(id) => write!(f, "monitor rule not found: {id}"),
        }
    }
}

impl std::error::Error for MonitorError {}

// ─────────────────────────────────────────────
// Register type / address
// ─────────────────────────────────────────────

/// Modbus register type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegisterType {
    Coil,
    DiscreteInput,
    HoldingRegister,
    InputRegister,
}

/// A fully qualified Modbus register address.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegisterAddress {
    pub register_type: RegisterType,
    pub address: u16,
}

// ─────────────────────────────────────────────
// Threshold condition
// ─────────────────────────────────────────────

/// Condition evaluated against the register value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdCondition {
    /// Trigger when value > threshold.
    Above(f64),
    /// Trigger when value < threshold.
    Below(f64),
    /// Trigger when value == threshold (within floating-point epsilon).
    Equal(f64),
    /// Trigger when value != threshold.
    NotEqual(f64),
    /// Trigger when `low <= value <= high`.
    InRange(f64, f64),
    /// Trigger when `value < low || value > high`.
    OutOfRange(f64, f64),
    /// Trigger when the value is different from the last observed value.
    Changed,
}

// ─────────────────────────────────────────────
// Monitor rule
// ─────────────────────────────────────────────

/// A single monitoring rule applied to one register address.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorRule {
    /// Unique rule identifier.
    pub id: String,
    /// Register to watch.
    pub address: RegisterAddress,
    /// Condition that triggers an alert.
    pub condition: ThresholdCondition,
    /// Minimum milliseconds between consecutive alerts for this rule.
    pub cooldown_ms: u64,
}

// ─────────────────────────────────────────────
// Alert event
// ─────────────────────────────────────────────

/// An alert event emitted when a monitor rule condition is satisfied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvent {
    /// ID of the rule that triggered.
    pub rule_id: String,
    /// Timestamp in milliseconds when the alert was raised.
    pub timestamp_ms: u64,
    /// Value observed before this update (if available).
    pub previous_value: f64,
    /// Value that triggered the alert.
    pub current_value: f64,
    /// Human-readable description.
    pub message: String,
}

// ─────────────────────────────────────────────
// Register monitor
// ─────────────────────────────────────────────

/// Monitors a set of Modbus registers for threshold violations.
#[derive(Debug, Default)]
pub struct RegisterMonitor {
    /// All registered monitoring rules keyed by rule id.
    rules: HashMap<String, MonitorRule>,
    /// Last known value per raw address (u16 key).
    last_values: HashMap<u16, f64>,
    /// Timestamp of the last alert per rule id.
    last_alert_ms: HashMap<String, u64>,
}

impl RegisterMonitor {
    /// Create an empty monitor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a monitoring rule.
    ///
    /// Returns `Err(DuplicateRuleId)` if a rule with the same id already exists.
    pub fn add_rule(&mut self, rule: MonitorRule) -> Result<(), MonitorError> {
        if self.rules.contains_key(&rule.id) {
            return Err(MonitorError::DuplicateRuleId(rule.id.clone()));
        }
        self.rules.insert(rule.id.clone(), rule);
        Ok(())
    }

    /// Remove a monitoring rule by id.
    ///
    /// Returns `Err(RuleNotFound)` if no such rule exists.
    pub fn remove_rule(&mut self, id: &str) -> Result<(), MonitorError> {
        if self.rules.remove(id).is_none() {
            return Err(MonitorError::RuleNotFound(id.to_string()));
        }
        self.last_alert_ms.remove(id);
        Ok(())
    }

    /// Update the value of a register at address `address` at time `now_ms`.
    ///
    /// Returns all `AlertEvent`s for rules whose condition is satisfied and
    /// whose cooldown has elapsed.
    pub fn update_value(&mut self, address: u16, value: f64, now_ms: u64) -> Vec<AlertEvent> {
        let prev = self.last_values.get(&address).copied();
        self.last_values.insert(address, value);

        let prev_val = prev.unwrap_or(0.0);

        // Collect matching rule ids first (avoid borrow issues).
        let matching_ids: Vec<String> = self
            .rules
            .values()
            .filter(|r| r.address.address == address)
            .filter(|r| Self::evaluate_condition(&r.condition, prev, value))
            .map(|r| r.id.clone())
            .collect();

        let mut events = Vec::new();

        for id in matching_ids {
            let last_opt = self.last_alert_ms.get(&id).copied();
            let rule = match self.rules.get(&id) {
                Some(r) => r,
                None => continue,
            };
            // Allow the first alert through (no previous alert recorded),
            // or if the cooldown has elapsed since the last alert.
            let cooldown_ok = match last_opt {
                None => true,
                Some(last) => now_ms.saturating_sub(last) >= rule.cooldown_ms,
            };
            if cooldown_ok {
                let message = format!(
                    "Rule '{}': address {} value {value} triggered condition {:?}",
                    id, address, rule.condition
                );
                events.push(AlertEvent {
                    rule_id: id.clone(),
                    timestamp_ms: now_ms,
                    previous_value: prev_val,
                    current_value: value,
                    message,
                });
                self.last_alert_ms.insert(id, now_ms);
            }
        }

        events
    }

    /// Evaluate a threshold condition against the previous and current values.
    pub fn evaluate_condition(
        condition: &ThresholdCondition,
        prev: Option<f64>,
        current: f64,
    ) -> bool {
        match condition {
            ThresholdCondition::Above(t) => current > *t,
            ThresholdCondition::Below(t) => current < *t,
            ThresholdCondition::Equal(t) => (current - t).abs() < f64::EPSILON,
            ThresholdCondition::NotEqual(t) => (current - t).abs() >= f64::EPSILON,
            ThresholdCondition::InRange(lo, hi) => current >= *lo && current <= *hi,
            ThresholdCondition::OutOfRange(lo, hi) => current < *lo || current > *hi,
            ThresholdCondition::Changed => match prev {
                Some(p) => (current - p).abs() >= f64::EPSILON,
                None => false, // No previous value → no change
            },
        }
    }

    /// Return all currently active rules.
    pub fn active_rules(&self) -> Vec<&MonitorRule> {
        self.rules.values().collect()
    }

    /// Return the last known value for a raw register address, if any.
    pub fn current_value(&self, address: u16) -> Option<f64> {
        self.last_values.get(&address).copied()
    }

    /// Clear all stored register values and alert timestamps.
    pub fn clear_history(&mut self) {
        self.last_values.clear();
        self.last_alert_ms.clear();
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn holding(address: u16) -> RegisterAddress {
        RegisterAddress {
            register_type: RegisterType::HoldingRegister,
            address,
        }
    }

    fn rule(id: &str, address: u16, cond: ThresholdCondition, cooldown: u64) -> MonitorRule {
        MonitorRule {
            id: id.to_string(),
            address: holding(address),
            condition: cond,
            cooldown_ms: cooldown,
        }
    }

    // ── add_rule ───────────────────────────────────────────────────────

    #[test]
    fn test_add_rule_success() {
        let mut mon = RegisterMonitor::new();
        assert!(mon
            .add_rule(rule("r1", 100, ThresholdCondition::Above(50.0), 0))
            .is_ok());
    }

    #[test]
    fn test_add_rule_duplicate_error() {
        let mut mon = RegisterMonitor::new();
        mon.add_rule(rule("r1", 100, ThresholdCondition::Above(50.0), 0))
            .expect("should succeed");
        let res = mon.add_rule(rule("r1", 100, ThresholdCondition::Below(10.0), 0));
        assert_eq!(res, Err(MonitorError::DuplicateRuleId("r1".to_string())));
    }

    // ── remove_rule ────────────────────────────────────────────────────

    #[test]
    fn test_remove_rule_success() {
        let mut mon = RegisterMonitor::new();
        mon.add_rule(rule("r1", 100, ThresholdCondition::Above(50.0), 0))
            .expect("should succeed");
        assert!(mon.remove_rule("r1").is_ok());
        assert!(mon.rules.is_empty());
    }

    #[test]
    fn test_remove_rule_not_found() {
        let mut mon = RegisterMonitor::new();
        let res = mon.remove_rule("nonexistent");
        assert_eq!(
            res,
            Err(MonitorError::RuleNotFound("nonexistent".to_string()))
        );
    }

    // ── evaluate_condition ─────────────────────────────────────────────

    #[test]
    fn test_cond_above_true() {
        assert!(RegisterMonitor::evaluate_condition(
            &ThresholdCondition::Above(10.0),
            None,
            11.0
        ));
    }

    #[test]
    fn test_cond_above_false() {
        assert!(!RegisterMonitor::evaluate_condition(
            &ThresholdCondition::Above(10.0),
            None,
            9.0
        ));
    }

    #[test]
    fn test_cond_below_true() {
        assert!(RegisterMonitor::evaluate_condition(
            &ThresholdCondition::Below(5.0),
            None,
            4.0
        ));
    }

    #[test]
    fn test_cond_equal_true() {
        assert!(RegisterMonitor::evaluate_condition(
            &ThresholdCondition::Equal(3.0),
            None,
            3.0
        ));
    }

    #[test]
    fn test_cond_not_equal_true() {
        assert!(RegisterMonitor::evaluate_condition(
            &ThresholdCondition::NotEqual(5.0),
            None,
            6.0
        ));
    }

    #[test]
    fn test_cond_in_range_true() {
        assert!(RegisterMonitor::evaluate_condition(
            &ThresholdCondition::InRange(1.0, 10.0),
            None,
            5.0
        ));
    }

    #[test]
    fn test_cond_in_range_boundary() {
        assert!(RegisterMonitor::evaluate_condition(
            &ThresholdCondition::InRange(1.0, 10.0),
            None,
            1.0
        ));
        assert!(RegisterMonitor::evaluate_condition(
            &ThresholdCondition::InRange(1.0, 10.0),
            None,
            10.0
        ));
    }

    #[test]
    fn test_cond_out_of_range_true() {
        assert!(RegisterMonitor::evaluate_condition(
            &ThresholdCondition::OutOfRange(1.0, 10.0),
            None,
            11.0
        ));
        assert!(RegisterMonitor::evaluate_condition(
            &ThresholdCondition::OutOfRange(1.0, 10.0),
            None,
            0.0
        ));
    }

    #[test]
    fn test_cond_changed_true() {
        assert!(RegisterMonitor::evaluate_condition(
            &ThresholdCondition::Changed,
            Some(1.0),
            2.0
        ));
    }

    #[test]
    fn test_cond_changed_false_same_value() {
        assert!(!RegisterMonitor::evaluate_condition(
            &ThresholdCondition::Changed,
            Some(5.0),
            5.0
        ));
    }

    #[test]
    fn test_cond_changed_false_no_prev() {
        assert!(!RegisterMonitor::evaluate_condition(
            &ThresholdCondition::Changed,
            None,
            5.0
        ));
    }

    // ── update_value ───────────────────────────────────────────────────

    #[test]
    fn test_update_value_triggers_alert() {
        let mut mon = RegisterMonitor::new();
        mon.add_rule(rule("r1", 100, ThresholdCondition::Above(50.0), 0))
            .expect("should succeed");
        let events = mon.update_value(100, 75.0, 1000);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].rule_id, "r1");
        assert!((events[0].current_value - 75.0).abs() < 1e-9);
    }

    #[test]
    fn test_update_value_no_alert_below_threshold() {
        let mut mon = RegisterMonitor::new();
        mon.add_rule(rule("r1", 100, ThresholdCondition::Above(50.0), 0))
            .expect("should succeed");
        let events = mon.update_value(100, 30.0, 1000);
        assert!(events.is_empty());
    }

    #[test]
    fn test_update_value_cooldown_suppresses() {
        let mut mon = RegisterMonitor::new();
        mon.add_rule(rule("r1", 100, ThresholdCondition::Above(10.0), 5000))
            .expect("should succeed");
        let ev1 = mon.update_value(100, 20.0, 1000);
        assert_eq!(ev1.len(), 1);
        let ev2 = mon.update_value(100, 25.0, 2000); // 1000 ms elapsed < 5000 ms cooldown
        assert!(ev2.is_empty());
    }

    #[test]
    fn test_update_value_cooldown_expires() {
        let mut mon = RegisterMonitor::new();
        mon.add_rule(rule("r1", 100, ThresholdCondition::Above(10.0), 1000))
            .expect("should succeed");
        mon.update_value(100, 20.0, 0);
        let events = mon.update_value(100, 30.0, 2000); // 2000 ms > 1000 ms cooldown
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_update_value_different_address_no_alert() {
        let mut mon = RegisterMonitor::new();
        mon.add_rule(rule("r1", 100, ThresholdCondition::Above(10.0), 0))
            .expect("should succeed");
        let events = mon.update_value(200, 50.0, 1000); // address 200, rule watches 100
        assert!(events.is_empty());
    }

    #[test]
    fn test_update_value_multiple_rules_same_address() {
        let mut mon = RegisterMonitor::new();
        mon.add_rule(rule("above", 100, ThresholdCondition::Above(10.0), 0))
            .expect("should succeed");
        mon.add_rule(rule("below", 100, ThresholdCondition::Below(100.0), 0))
            .expect("should succeed");
        let events = mon.update_value(100, 50.0, 1000);
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_update_value_changed_condition() {
        let mut mon = RegisterMonitor::new();
        mon.add_rule(rule("changed", 100, ThresholdCondition::Changed, 0))
            .expect("should succeed");
        // First update — no prev value, so no alert
        let ev1 = mon.update_value(100, 1.0, 1000);
        assert!(ev1.is_empty());
        // Second update — value changed
        let ev2 = mon.update_value(100, 2.0, 2000);
        assert_eq!(ev2.len(), 1);
    }

    // ── active_rules ───────────────────────────────────────────────────

    #[test]
    fn test_active_rules_empty() {
        let mon = RegisterMonitor::new();
        assert!(mon.active_rules().is_empty());
    }

    #[test]
    fn test_active_rules_returns_all() {
        let mut mon = RegisterMonitor::new();
        mon.add_rule(rule("r1", 100, ThresholdCondition::Above(1.0), 0))
            .expect("should succeed");
        mon.add_rule(rule("r2", 200, ThresholdCondition::Below(0.0), 0))
            .expect("should succeed");
        assert_eq!(mon.active_rules().len(), 2);
    }

    // ── current_value ──────────────────────────────────────────────────

    #[test]
    fn test_current_value_none_before_update() {
        let mon = RegisterMonitor::new();
        assert!(mon.current_value(100).is_none());
    }

    #[test]
    fn test_current_value_after_update() {
        let mut mon = RegisterMonitor::new();
        mon.update_value(100, 42.0, 1000);
        assert!((mon.current_value(100).unwrap_or(0.0) - 42.0).abs() < 1e-9);
    }

    // ── clear_history ──────────────────────────────────────────────────

    #[test]
    fn test_clear_history_removes_values() {
        let mut mon = RegisterMonitor::new();
        mon.update_value(100, 42.0, 1000);
        mon.clear_history();
        assert!(mon.current_value(100).is_none());
    }

    #[test]
    fn test_clear_history_resets_cooldowns() {
        let mut mon = RegisterMonitor::new();
        mon.add_rule(rule("r1", 100, ThresholdCondition::Above(10.0), 99999))
            .expect("should succeed");
        mon.update_value(100, 50.0, 0); // alert at t=0
        mon.clear_history(); // reset cooldown state
        let events = mon.update_value(100, 60.0, 1); // should fire again
        assert_eq!(events.len(), 1);
    }

    // ── multiple register types ────────────────────────────────────────

    #[test]
    fn test_coil_register_rule() {
        let mut mon = RegisterMonitor::new();
        mon.add_rule(MonitorRule {
            id: "coil1".to_string(),
            address: RegisterAddress {
                register_type: RegisterType::Coil,
                address: 5,
            },
            condition: ThresholdCondition::Equal(1.0),
            cooldown_ms: 0,
        })
        .expect("should succeed");
        let events = mon.update_value(5, 1.0, 100);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_monitor_error_display() {
        let e = MonitorError::DuplicateRuleId("abc".to_string());
        assert!(e.to_string().contains("abc"));
        let e2 = MonitorError::RuleNotFound("xyz".to_string());
        assert!(e2.to_string().contains("xyz"));
    }
}
