//! Modbus alarm and event management.
//!
//! Allows defining alarm rules based on Modbus register values, evaluating
//! incoming register readings, and tracking the lifecycle of triggered alarms
//! from active through acknowledged to cleared.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Severity ordering: Emergency is highest (0), Debug is lowest (7)
// ---------------------------------------------------------------------------

/// The severity level of an alarm, following syslog conventions.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlarmSeverity {
    /// System is unusable.
    Emergency = 0,
    /// Action must be taken immediately.
    Alert = 1,
    /// Critical conditions.
    Critical = 2,
    /// Error conditions.
    Error = 3,
    /// Warning conditions.
    Warning = 4,
    /// Normal but significant condition.
    Notice = 5,
    /// Informational messages.
    Info = 6,
    /// Debug-level messages.
    Debug = 7,
}

/// The current lifecycle state of an alarm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlarmState {
    /// The alarm is active and has not yet been acknowledged.
    Active,
    /// The alarm has been acknowledged by an operator.
    Acknowledged,
    /// The condition has been cleared and the alarm is closed.
    Cleared,
}

/// A live alarm instance.
#[derive(Debug, Clone)]
pub struct Alarm {
    /// Unique identifier of this alarm instance.
    pub id: u64,
    /// Modbus register address that triggered the alarm.
    pub register: u16,
    /// Optional bit position within the register (for bit-level rules).
    pub bit: Option<u8>,
    /// Human-readable description.
    pub description: String,
    /// Severity classification.
    pub severity: AlarmSeverity,
    /// Current lifecycle state.
    pub state: AlarmState,
    /// Millisecond timestamp when the alarm was first raised.
    pub triggered_at: u64,
    /// Millisecond timestamp when the alarm was acknowledged, if applicable.
    pub acknowledged_at: Option<u64>,
    /// Millisecond timestamp when the alarm was cleared, if applicable.
    pub cleared_at: Option<u64>,
}

/// A rule that maps a register value (or bit) to an alarm condition.
#[derive(Debug, Clone)]
pub struct AlarmRule {
    /// Unique identifier of this rule.
    pub id: u64,
    /// Modbus register address to monitor.
    pub register: u16,
    /// Optional bit index within the register (bits 0–15).
    pub bit: Option<u8>,
    /// The raw register value (or bit value: 0/1) that triggers the alarm.
    pub trigger_value: u16,
    /// Human-readable description of the alarm condition.
    pub description: String,
    /// Severity to assign when the alarm fires.
    pub severity: AlarmSeverity,
}

/// Manages Modbus alarm rules, active alarms, and alarm history.
pub struct AlarmManager {
    rules: HashMap<u64, AlarmRule>,
    active_alarms: HashMap<u64, Alarm>,
    history: Vec<Alarm>,
    next_id: u64,
}

impl Default for AlarmManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AlarmManager {
    /// Create a new, empty `AlarmManager`.
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            active_alarms: HashMap::new(),
            history: Vec::new(),
            next_id: 1,
        }
    }

    /// Add an alarm rule to the manager.
    pub fn add_rule(&mut self, rule: AlarmRule) {
        self.rules.insert(rule.id, rule);
    }

    /// Evaluate a register reading and trigger alarms as appropriate.
    ///
    /// Matches every rule whose `register` equals the supplied `register`
    /// address.  For bit-level rules, extracts the relevant bit from `value`.
    /// For whole-register rules, compares `value` directly to
    /// `rule.trigger_value`.
    ///
    /// Already-active alarms for a rule are not duplicated.
    ///
    /// Returns a `Vec` of newly triggered alarm IDs.
    pub fn evaluate(&mut self, register: u16, value: u16, now_ms: u64) -> Vec<u64> {
        let mut triggered = Vec::new();

        // Collect matching rules first to avoid borrow conflicts
        let matching_rules: Vec<AlarmRule> = self
            .rules
            .values()
            .filter(|r| r.register == register)
            .cloned()
            .collect();

        for rule in matching_rules {
            let effective_value = match rule.bit {
                Some(bit_pos) => (value >> bit_pos) & 1,
                None => value,
            };

            if effective_value != rule.trigger_value {
                continue;
            }

            // Skip if an active alarm for this rule already exists
            let already_active = self
                .active_alarms
                .values()
                .any(|a| a.register == rule.register && a.bit == rule.bit);
            if already_active {
                continue;
            }

            let alarm_id = self.next_id;
            self.next_id += 1;

            let alarm = Alarm {
                id: alarm_id,
                register: rule.register,
                bit: rule.bit,
                description: rule.description.clone(),
                severity: rule.severity.clone(),
                state: AlarmState::Active,
                triggered_at: now_ms,
                acknowledged_at: None,
                cleared_at: None,
            };

            self.active_alarms.insert(alarm_id, alarm);
            triggered.push(alarm_id);
        }

        triggered
    }

    /// Acknowledge an active alarm.
    ///
    /// Returns `true` if the alarm was found and is in the `Active` state;
    /// `false` otherwise.
    pub fn acknowledge(&mut self, alarm_id: u64, now_ms: u64) -> bool {
        if let Some(alarm) = self.active_alarms.get_mut(&alarm_id) {
            if alarm.state == AlarmState::Active {
                alarm.state = AlarmState::Acknowledged;
                alarm.acknowledged_at = Some(now_ms);
                return true;
            }
        }
        false
    }

    /// Clear an active or acknowledged alarm, moving it to history.
    ///
    /// Returns `true` if the alarm was found and successfully cleared;
    /// `false` otherwise.
    pub fn clear(&mut self, alarm_id: u64, now_ms: u64) -> bool {
        if let Some(mut alarm) = self.active_alarms.remove(&alarm_id) {
            alarm.state = AlarmState::Cleared;
            alarm.cleared_at = Some(now_ms);
            self.history.push(alarm);
            true
        } else {
            false
        }
    }

    /// Number of currently active (not yet cleared) alarms.
    pub fn active_count(&self) -> usize {
        self.active_alarms.len()
    }

    /// Return references to all active alarms with the given severity.
    pub fn active_by_severity(&self, severity: AlarmSeverity) -> Vec<&Alarm> {
        self.active_alarms
            .values()
            .filter(|a| a.severity == severity)
            .collect()
    }

    /// Number of alarms that have been cleared (history size).
    pub fn history_count(&self) -> usize {
        self.history.len()
    }

    /// Return the highest (lowest ordinal) severity among active alarms,
    /// or `None` if there are no active alarms.
    pub fn highest_active_severity(&self) -> Option<&AlarmSeverity> {
        self.active_alarms
            .values()
            .map(|a| &a.severity)
            .min_by_key(|s| *s as &AlarmSeverity)
    }

    /// Borrow the full slice of alarm history.
    pub fn history(&self) -> &[Alarm] {
        &self.history
    }

    /// Borrow the active alarms map.
    pub fn active_alarms(&self) -> &HashMap<u64, Alarm> {
        &self.active_alarms
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rule(id: u64, register: u16, trigger: u16, sev: AlarmSeverity) -> AlarmRule {
        AlarmRule {
            id,
            register,
            bit: None,
            trigger_value: trigger,
            description: format!("rule-{id}"),
            severity: sev,
        }
    }

    fn make_bit_rule(
        id: u64,
        register: u16,
        bit: u8,
        trigger: u16,
        sev: AlarmSeverity,
    ) -> AlarmRule {
        AlarmRule {
            id,
            register,
            bit: Some(bit),
            trigger_value: trigger,
            description: format!("bit-rule-{id}"),
            severity: sev,
        }
    }

    // --- new / add_rule ---

    #[test]
    fn test_new_starts_empty() {
        let mgr = AlarmManager::new();
        assert_eq!(mgr.active_count(), 0);
        assert_eq!(mgr.history_count(), 0);
    }

    #[test]
    fn test_add_rule_stores_rule() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 100, 1, AlarmSeverity::Warning));
        // Indirectly verify by evaluating
        let ids = mgr.evaluate(100, 1, 1000);
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_default_same_as_new() {
        let mgr = AlarmManager::default();
        assert_eq!(mgr.active_count(), 0);
    }

    // --- evaluate ---

    #[test]
    fn test_evaluate_triggers_alarm() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 200, 0xFFFF, AlarmSeverity::Error));
        let ids = mgr.evaluate(200, 0xFFFF, 1000);
        assert_eq!(ids.len(), 1);
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn test_evaluate_no_match_on_different_register() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 200, 1, AlarmSeverity::Warning));
        let ids = mgr.evaluate(201, 1, 1000);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_evaluate_no_trigger_on_wrong_value() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 100, 5, AlarmSeverity::Info));
        let ids = mgr.evaluate(100, 6, 1000);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_evaluate_bit_rule_triggers_on_bit_set() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_bit_rule(1, 300, 3, 1, AlarmSeverity::Critical));
        // Bit 3 of 0b00001000 = 1
        let ids = mgr.evaluate(300, 0b0000_1000, 1000);
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_evaluate_bit_rule_no_trigger_when_bit_clear() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_bit_rule(1, 300, 3, 1, AlarmSeverity::Critical));
        let ids = mgr.evaluate(300, 0b0000_0000, 1000);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_evaluate_no_duplicate_active_alarms() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 100, 1, AlarmSeverity::Warning));
        mgr.evaluate(100, 1, 1000);
        let ids = mgr.evaluate(100, 1, 2000);
        // Second evaluation should not add a duplicate
        assert!(ids.is_empty());
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn test_evaluate_multiple_rules_same_register() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 100, 1, AlarmSeverity::Warning));
        mgr.add_rule(make_bit_rule(2, 100, 0, 1, AlarmSeverity::Error));
        // value = 1 triggers both the whole-register rule (100==1) and bit-0 rule (bit0==1)
        let ids = mgr.evaluate(100, 1, 1000);
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_evaluate_returns_alarm_id() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 50, 0, AlarmSeverity::Debug));
        let ids = mgr.evaluate(50, 0, 1000);
        assert_eq!(ids.len(), 1);
        let id = ids[0];
        assert!(mgr.active_alarms().contains_key(&id));
    }

    // --- acknowledge ---

    #[test]
    fn test_acknowledge_active_alarm() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 100, 1, AlarmSeverity::Warning));
        let ids = mgr.evaluate(100, 1, 1000);
        let id = ids[0];
        let result = mgr.acknowledge(id, 2000);
        assert!(result);
        assert_eq!(mgr.active_alarms()[&id].state, AlarmState::Acknowledged);
        assert_eq!(mgr.active_alarms()[&id].acknowledged_at, Some(2000));
    }

    #[test]
    fn test_acknowledge_unknown_id_returns_false() {
        let mut mgr = AlarmManager::new();
        assert!(!mgr.acknowledge(9999, 1000));
    }

    #[test]
    fn test_acknowledge_twice_returns_false_second_time() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 100, 1, AlarmSeverity::Warning));
        let ids = mgr.evaluate(100, 1, 1000);
        let id = ids[0];
        mgr.acknowledge(id, 2000);
        // Already acknowledged — second call should fail
        assert!(!mgr.acknowledge(id, 3000));
    }

    // --- clear ---

    #[test]
    fn test_clear_active_alarm_moves_to_history() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 100, 1, AlarmSeverity::Warning));
        let ids = mgr.evaluate(100, 1, 1000);
        let id = ids[0];
        let result = mgr.clear(id, 3000);
        assert!(result);
        assert_eq!(mgr.active_count(), 0);
        assert_eq!(mgr.history_count(), 1);
        assert_eq!(mgr.history()[0].cleared_at, Some(3000));
    }

    #[test]
    fn test_clear_acknowledged_alarm_moves_to_history() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 100, 1, AlarmSeverity::Warning));
        let ids = mgr.evaluate(100, 1, 1000);
        let id = ids[0];
        mgr.acknowledge(id, 2000);
        assert!(mgr.clear(id, 3000));
        assert_eq!(mgr.history_count(), 1);
    }

    #[test]
    fn test_clear_unknown_id_returns_false() {
        let mut mgr = AlarmManager::new();
        assert!(!mgr.clear(9999, 1000));
    }

    #[test]
    fn test_clear_then_re_trigger() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 100, 1, AlarmSeverity::Warning));
        let ids = mgr.evaluate(100, 1, 1000);
        mgr.clear(ids[0], 2000);
        // After clearing, the next matching evaluation should trigger a new alarm
        let ids2 = mgr.evaluate(100, 1, 3000);
        assert_eq!(ids2.len(), 1);
        assert_eq!(mgr.active_count(), 1);
    }

    // --- active_by_severity ---

    #[test]
    fn test_active_by_severity_returns_matching() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 100, 1, AlarmSeverity::Warning));
        mgr.add_rule(make_rule(2, 200, 1, AlarmSeverity::Error));
        mgr.evaluate(100, 1, 1000);
        mgr.evaluate(200, 1, 1000);
        let warnings = mgr.active_by_severity(AlarmSeverity::Warning);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].severity, AlarmSeverity::Warning);
    }

    #[test]
    fn test_active_by_severity_empty() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 100, 1, AlarmSeverity::Warning));
        mgr.evaluate(100, 1, 1000);
        let errors = mgr.active_by_severity(AlarmSeverity::Error);
        assert!(errors.is_empty());
    }

    // --- highest_active_severity ---

    #[test]
    fn test_highest_active_severity_none_when_empty() {
        let mgr = AlarmManager::new();
        assert!(mgr.highest_active_severity().is_none());
    }

    #[test]
    fn test_highest_active_severity_single() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 100, 1, AlarmSeverity::Warning));
        mgr.evaluate(100, 1, 1000);
        assert_eq!(mgr.highest_active_severity(), Some(&AlarmSeverity::Warning));
    }

    #[test]
    fn test_highest_active_severity_picks_most_severe() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 100, 1, AlarmSeverity::Warning));
        mgr.add_rule(make_rule(2, 200, 1, AlarmSeverity::Critical));
        mgr.add_rule(make_rule(3, 300, 1, AlarmSeverity::Info));
        mgr.evaluate(100, 1, 1000);
        mgr.evaluate(200, 1, 1000);
        mgr.evaluate(300, 1, 1000);
        // Critical (ordinal 2) < Warning (4) < Info (6) → Critical is highest
        assert_eq!(
            mgr.highest_active_severity(),
            Some(&AlarmSeverity::Critical)
        );
    }

    // --- severity ordering ---

    #[test]
    fn test_severity_ordering() {
        assert!(AlarmSeverity::Emergency < AlarmSeverity::Alert);
        assert!(AlarmSeverity::Alert < AlarmSeverity::Critical);
        assert!(AlarmSeverity::Critical < AlarmSeverity::Error);
        assert!(AlarmSeverity::Error < AlarmSeverity::Warning);
        assert!(AlarmSeverity::Warning < AlarmSeverity::Notice);
        assert!(AlarmSeverity::Notice < AlarmSeverity::Info);
        assert!(AlarmSeverity::Info < AlarmSeverity::Debug);
    }

    // --- alarm state ---

    #[test]
    fn test_alarm_state_eq() {
        assert_eq!(AlarmState::Active, AlarmState::Active);
        assert_ne!(AlarmState::Active, AlarmState::Acknowledged);
        assert_ne!(AlarmState::Acknowledged, AlarmState::Cleared);
    }

    #[test]
    fn test_alarm_triggered_at_stored() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 100, 1, AlarmSeverity::Warning));
        let ids = mgr.evaluate(100, 1, 42_000);
        let alarm = &mgr.active_alarms()[&ids[0]];
        assert_eq!(alarm.triggered_at, 42_000);
    }

    #[test]
    fn test_alarm_bit_stored() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_bit_rule(1, 10, 5, 1, AlarmSeverity::Notice));
        let ids = mgr.evaluate(10, 1 << 5, 1000);
        let alarm = &mgr.active_alarms()[&ids[0]];
        assert_eq!(alarm.bit, Some(5));
    }

    // --- history ---

    #[test]
    fn test_history_starts_empty() {
        let mgr = AlarmManager::new();
        assert_eq!(mgr.history_count(), 0);
        assert!(mgr.history().is_empty());
    }

    #[test]
    fn test_history_accumulates() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 100, 1, AlarmSeverity::Warning));
        let ids = mgr.evaluate(100, 1, 1000);
        mgr.clear(ids[0], 2000);
        mgr.evaluate(100, 1, 3000);
        let ids2 = mgr.active_alarms().keys().cloned().collect::<Vec<_>>();
        mgr.clear(ids2[0], 4000);
        assert_eq!(mgr.history_count(), 2);
    }

    // --- additional coverage ---

    #[test]
    fn test_evaluate_zero_value_trigger() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 50, 0, AlarmSeverity::Emergency));
        let ids = mgr.evaluate(50, 0, 100);
        assert_eq!(ids.len(), 1);
        assert_eq!(
            mgr.active_alarms()[&ids[0]].severity,
            AlarmSeverity::Emergency
        );
    }

    #[test]
    fn test_evaluate_max_value_trigger() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 10, 0xFFFF, AlarmSeverity::Alert));
        let ids = mgr.evaluate(10, 0xFFFF, 500);
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_evaluate_bit_rule_high_bit() {
        let mut mgr = AlarmManager::new();
        // bit 15 (MSB of u16)
        mgr.add_rule(make_bit_rule(1, 20, 15, 1, AlarmSeverity::Debug));
        let ids = mgr.evaluate(20, 0x8000, 100);
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_evaluate_bit_rule_trigger_zero_value() {
        let mut mgr = AlarmManager::new();
        // trigger when bit 2 is 0
        mgr.add_rule(make_bit_rule(1, 30, 2, 0, AlarmSeverity::Notice));
        // 0b0000_0000 → bit 2 == 0 → trigger
        let ids = mgr.evaluate(30, 0b0000_0000, 100);
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_evaluate_increments_id_each_alarm() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 10, 1, AlarmSeverity::Info));
        mgr.add_rule(make_rule(2, 20, 1, AlarmSeverity::Info));
        let ids1 = mgr.evaluate(10, 1, 100);
        let ids2 = mgr.evaluate(20, 1, 200);
        assert_ne!(ids1[0], ids2[0]);
    }

    #[test]
    fn test_acknowledge_stores_timestamp() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 100, 1, AlarmSeverity::Warning));
        let ids = mgr.evaluate(100, 1, 1000);
        mgr.acknowledge(ids[0], 9999);
        assert_eq!(mgr.active_alarms()[&ids[0]].acknowledged_at, Some(9999));
    }

    #[test]
    fn test_clear_sets_cleared_state_in_history() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 100, 1, AlarmSeverity::Warning));
        let ids = mgr.evaluate(100, 1, 100);
        mgr.clear(ids[0], 500);
        assert_eq!(mgr.history()[0].state, AlarmState::Cleared);
    }

    #[test]
    fn test_alarm_description_stored() {
        let mut mgr = AlarmManager::new();
        let rule = AlarmRule {
            id: 1,
            register: 10,
            bit: None,
            trigger_value: 7,
            description: "overvoltage detected".to_string(),
            severity: AlarmSeverity::Critical,
        };
        mgr.add_rule(rule);
        let ids = mgr.evaluate(10, 7, 100);
        assert_eq!(
            mgr.active_alarms()[&ids[0]].description,
            "overvoltage detected"
        );
    }

    #[test]
    fn test_active_by_severity_multiple_alarms_same_severity() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 10, 1, AlarmSeverity::Error));
        mgr.add_rule(make_rule(2, 20, 1, AlarmSeverity::Error));
        mgr.evaluate(10, 1, 100);
        mgr.evaluate(20, 1, 200);
        let errs = mgr.active_by_severity(AlarmSeverity::Error);
        assert_eq!(errs.len(), 2);
    }

    #[test]
    fn test_highest_severity_after_clear() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 10, 1, AlarmSeverity::Emergency));
        mgr.add_rule(make_rule(2, 20, 1, AlarmSeverity::Debug));
        let ids1 = mgr.evaluate(10, 1, 100);
        mgr.evaluate(20, 1, 200);
        // Clear the Emergency alarm
        mgr.clear(ids1[0], 300);
        // Now only Debug remains
        assert_eq!(mgr.highest_active_severity(), Some(&AlarmSeverity::Debug));
    }

    #[test]
    fn test_add_rule_overwrites_same_id() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 100, 5, AlarmSeverity::Warning));
        // Overwrite rule 1 with a new trigger value
        mgr.add_rule(make_rule(1, 100, 9, AlarmSeverity::Error));
        // value 5 should NOT trigger (old rule overwritten)
        assert!(mgr.evaluate(100, 5, 100).is_empty());
        // value 9 SHOULD trigger
        assert_eq!(mgr.evaluate(100, 9, 200).len(), 1);
    }

    #[test]
    fn test_active_count_decrements_after_clear() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 10, 1, AlarmSeverity::Warning));
        mgr.add_rule(make_rule(2, 20, 1, AlarmSeverity::Error));
        let ids1 = mgr.evaluate(10, 1, 100);
        mgr.evaluate(20, 1, 200);
        assert_eq!(mgr.active_count(), 2);
        mgr.clear(ids1[0], 300);
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn test_no_alarms_without_rules() {
        let mut mgr = AlarmManager::new();
        let ids = mgr.evaluate(99, 0xABCD, 100);
        assert!(ids.is_empty());
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_severity_eq() {
        assert_eq!(AlarmSeverity::Critical, AlarmSeverity::Critical);
        assert_ne!(AlarmSeverity::Warning, AlarmSeverity::Error);
    }

    #[test]
    fn test_history_cleared_at_stored() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 5, 1, AlarmSeverity::Info));
        let ids = mgr.evaluate(5, 1, 1000);
        mgr.clear(ids[0], 7777);
        assert_eq!(mgr.history()[0].cleared_at, Some(7777));
    }

    #[test]
    fn test_history_alarm_register_preserved() {
        let mut mgr = AlarmManager::new();
        mgr.add_rule(make_rule(1, 42, 1, AlarmSeverity::Notice));
        let ids = mgr.evaluate(42, 1, 100);
        mgr.clear(ids[0], 200);
        assert_eq!(mgr.history()[0].register, 42);
    }
}
