//! Adaptive Modbus polling strategy (Fixed, Adaptive, OnChange, OnDemand).
//!
//! `PollingStrategy` tracks per-register state and decides when each register
//! should next be polled.  The adaptive mode automatically shortens or lengthens
//! the polling interval based on how often the register's value changes.

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// Selects one of four polling strategies.
#[derive(Debug, Clone, PartialEq)]
pub enum PollingMode {
    /// Poll every `interval_ms` milliseconds.
    Fixed { interval_ms: u64 },
    /// Vary the interval between `min_ms` and `max_ms` depending on change
    /// activity.  Intervals shrink when the register changes; they grow when
    /// it stays constant.  A value is considered "changed" when its
    /// difference from the previous reading exceeds `change_threshold`.
    Adaptive {
        min_ms: u64,
        max_ms: u64,
        change_threshold: f64,
    },
    /// Poll only when the value changes; apply `debounce_ms` to avoid
    /// flooding on noisy signals.
    OnChange { debounce_ms: u64 },
    /// Do not schedule automatic polls — only poll when explicitly requested.
    OnDemand,
}

/// The outcome of a single poll.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PollResult {
    /// Register address.
    pub register: u16,
    /// Raw register value returned by the device.
    pub value: u16,
    /// Epoch-millis timestamp of the poll.
    pub timestamp_ms: u64,
    /// Whether the value differs from the last recorded value.
    pub changed: bool,
}

impl PollResult {
    /// Convenience constructor.
    pub fn new(register: u16, value: u16, timestamp_ms: u64, changed: bool) -> Self {
        Self {
            register,
            value,
            timestamp_ms,
            changed,
        }
    }
}

/// Per-register runtime state tracked by the strategy.
#[derive(Debug, Clone)]
pub struct PollingState {
    /// Register address.
    pub register: u16,
    /// Last observed value, if any.
    pub last_value: Option<u16>,
    /// Epoch-millis of the most recent poll.
    pub last_poll_ms: u64,
    /// Total number of polls conducted for this register.
    pub poll_count: u64,
    /// Number of times the value changed since tracking began.
    pub change_count: u64,
    /// Current adaptive interval (only meaningful in `Adaptive` mode).
    current_interval_ms: u64,
}

impl PollingState {
    fn new(register: u16, initial_interval_ms: u64) -> Self {
        Self {
            register,
            last_value: None,
            last_poll_ms: 0,
            poll_count: 0,
            change_count: 0,
            current_interval_ms: initial_interval_ms,
        }
    }
}

/// Manages polling schedules and state for a collection of Modbus registers.
#[derive(Debug)]
pub struct PollingStrategy {
    mode: PollingMode,
    states: HashMap<u16, PollingState>,
}

impl PollingStrategy {
    /// Create a strategy with a given mode.  No registers are tracked until
    /// the first call to `should_poll` or `record_poll`.
    pub fn new(mode: PollingMode) -> Self {
        Self {
            mode,
            states: HashMap::new(),
        }
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Return the default interval for this mode (used when creating new state).
    fn default_interval(&self) -> u64 {
        match &self.mode {
            PollingMode::Fixed { interval_ms } => *interval_ms,
            PollingMode::Adaptive { min_ms, .. } => *min_ms,
            PollingMode::OnChange { debounce_ms } => *debounce_ms,
            PollingMode::OnDemand => u64::MAX,
        }
    }

    /// Lazily initialise per-register state.
    fn ensure_state(&mut self, register: u16) {
        let interval = self.default_interval();
        self.states
            .entry(register)
            .or_insert_with(|| PollingState::new(register, interval));
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// Return `true` if `register` is due for a poll at `now_ms`.
    pub fn should_poll(&mut self, register: u16, now_ms: u64) -> bool {
        self.ensure_state(register);
        let state = self.states.get(&register).expect("state just ensured");
        let next = self.next_poll_time_from_state(state, now_ms);
        now_ms >= next
    }

    /// Record the result of a poll and update internal state.
    pub fn record_poll(&mut self, result: PollResult) {
        self.ensure_state(result.register);
        let state = self
            .states
            .get_mut(&result.register)
            .expect("state just ensured");

        let changed = match state.last_value {
            None => true,
            Some(prev) => {
                let delta = (result.value as f64 - prev as f64).abs();
                match &self.mode {
                    PollingMode::Adaptive {
                        change_threshold, ..
                    } => delta > *change_threshold,
                    _ => result.value != prev,
                }
            }
        };

        // Adjust adaptive interval.
        if let PollingMode::Adaptive { min_ms, max_ms, .. } = &self.mode {
            if changed {
                // Value is volatile: speed up polling.
                state.current_interval_ms = (state.current_interval_ms / 2).max(*min_ms);
            } else {
                // Value is stable: slow down polling.
                state.current_interval_ms =
                    (state.current_interval_ms.saturating_mul(2)).min(*max_ms);
            }
        }

        state.last_value = Some(result.value);
        state.last_poll_ms = result.timestamp_ms;
        state.poll_count += 1;
        if changed {
            state.change_count += 1;
        }
    }

    /// Return the earliest epoch-millis at which `register` should be polled next.
    pub fn next_poll_time(&self, register: u16, now_ms: u64) -> u64 {
        match self.states.get(&register) {
            Some(state) => self.next_poll_time_from_state(state, now_ms),
            None => {
                // Register not yet seen — poll immediately.
                now_ms
            }
        }
    }

    fn next_poll_time_from_state(&self, state: &PollingState, _now_ms: u64) -> u64 {
        match &self.mode {
            PollingMode::Fixed { interval_ms } => {
                if state.poll_count == 0 {
                    0
                } else {
                    state.last_poll_ms + interval_ms
                }
            }
            PollingMode::Adaptive { .. } => {
                if state.poll_count == 0 {
                    0
                } else {
                    state.last_poll_ms + state.current_interval_ms
                }
            }
            PollingMode::OnChange { debounce_ms } => {
                if state.poll_count == 0 {
                    0
                } else {
                    state.last_poll_ms + debounce_ms
                }
            }
            PollingMode::OnDemand => u64::MAX,
        }
    }

    /// Return the change rate for a register (changes per poll, 0.0–1.0).
    ///
    /// Returns 0.0 if the register has never been polled.
    pub fn change_rate(&self, register: u16) -> f64 {
        match self.states.get(&register) {
            Some(state) if state.poll_count > 0 => {
                state.change_count as f64 / state.poll_count as f64
            }
            _ => 0.0,
        }
    }

    /// Return the total poll count for a register.
    pub fn poll_count(&self, register: u16) -> u64 {
        self.states
            .get(&register)
            .map(|s| s.poll_count)
            .unwrap_or(0)
    }

    /// Return all tracked register addresses.
    pub fn registers(&self) -> Vec<u16> {
        let mut regs: Vec<u16> = self.states.keys().copied().collect();
        regs.sort_unstable();
        regs
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed(interval_ms: u64) -> PollingMode {
        PollingMode::Fixed { interval_ms }
    }

    fn adaptive(min: u64, max: u64, threshold: f64) -> PollingMode {
        PollingMode::Adaptive {
            min_ms: min,
            max_ms: max,
            change_threshold: threshold,
        }
    }

    fn on_change(debounce: u64) -> PollingMode {
        PollingMode::OnChange {
            debounce_ms: debounce,
        }
    }

    fn make_result(reg: u16, value: u16, ts: u64) -> PollResult {
        PollResult::new(reg, value, ts, false)
    }

    // ── PollResult ────────────────────────────────────────────────────────────

    #[test]
    fn test_poll_result_new() {
        let r = PollResult::new(40001, 1234, 999, true);
        assert_eq!(r.register, 40001);
        assert_eq!(r.value, 1234);
        assert_eq!(r.timestamp_ms, 999);
        assert!(r.changed);
    }

    // ── Fixed mode ────────────────────────────────────────────────────────────

    #[test]
    fn test_fixed_first_poll_always_ready() {
        let mut s = PollingStrategy::new(fixed(1000));
        assert!(s.should_poll(0, 0));
    }

    #[test]
    fn test_fixed_not_ready_before_interval() {
        let mut s = PollingStrategy::new(fixed(1000));
        s.record_poll(make_result(0, 10, 500));
        assert!(!s.should_poll(0, 999));
    }

    #[test]
    fn test_fixed_ready_after_interval() {
        let mut s = PollingStrategy::new(fixed(1000));
        s.record_poll(make_result(0, 10, 0));
        assert!(s.should_poll(0, 1000));
    }

    #[test]
    fn test_fixed_ready_well_after_interval() {
        let mut s = PollingStrategy::new(fixed(500));
        s.record_poll(make_result(1, 100, 1000));
        assert!(s.should_poll(1, 2000));
    }

    #[test]
    fn test_fixed_multiple_registers_independent() {
        let mut s = PollingStrategy::new(fixed(1000));
        s.record_poll(make_result(0, 5, 0));
        s.record_poll(make_result(1, 6, 500));

        assert!(s.should_poll(0, 1000));
        assert!(!s.should_poll(1, 1000)); // not yet due
    }

    #[test]
    fn test_fixed_poll_count_increments() {
        let mut s = PollingStrategy::new(fixed(100));
        s.record_poll(make_result(0, 1, 0));
        s.record_poll(make_result(0, 1, 100));
        assert_eq!(s.poll_count(0), 2);
    }

    #[test]
    fn test_fixed_change_detection() {
        let mut s = PollingStrategy::new(fixed(100));
        s.record_poll(make_result(0, 10, 0));
        s.record_poll(make_result(0, 20, 100));
        // The value changed, so change_count should be > 0
        assert!(s.change_rate(0) > 0.0);
    }

    #[test]
    fn test_fixed_no_change_rate_zero() {
        let mut s = PollingStrategy::new(fixed(100));
        s.record_poll(make_result(0, 10, 0));
        s.record_poll(make_result(0, 10, 100));
        s.record_poll(make_result(0, 10, 200));
        // First poll always counts as "changed" (no previous value)
        assert_eq!(s.change_rate(0), 1.0 / 3.0);
    }

    // ── Adaptive mode ─────────────────────────────────────────────────────────

    #[test]
    fn test_adaptive_first_poll_ready() {
        let mut s = PollingStrategy::new(adaptive(100, 5000, 0.0));
        assert!(s.should_poll(0, 0));
    }

    #[test]
    fn test_adaptive_interval_decreases_on_change() {
        let mut s = PollingStrategy::new(adaptive(100, 5000, 0.0));
        // Record an initial poll then two changing polls to drive the interval down
        s.record_poll(make_result(0, 10, 0));
        let state_before = s.states[&0].current_interval_ms;
        s.record_poll(make_result(0, 20, state_before));
        let state_after = s.states[&0].current_interval_ms;
        assert!(state_after <= state_before);
    }

    #[test]
    fn test_adaptive_interval_increases_on_no_change() {
        let mut s = PollingStrategy::new(adaptive(100, 8000, 0.0));
        s.record_poll(make_result(0, 10, 0));
        let init = s.states[&0].current_interval_ms;
        // Same value three times
        s.record_poll(make_result(0, 10, init));
        s.record_poll(make_result(0, 10, init * 2));
        let after = s.states[&0].current_interval_ms;
        // The interval should have grown or stayed at max
        assert!(after >= init);
    }

    #[test]
    fn test_adaptive_interval_bounded_by_min() {
        let mut s = PollingStrategy::new(adaptive(500, 10000, 0.0));
        // Many changes should not drive interval below min
        for i in 0..20_u64 {
            s.record_poll(make_result(0, i as u16, i * 10));
        }
        assert!(s.states[&0].current_interval_ms >= 500);
    }

    #[test]
    fn test_adaptive_interval_bounded_by_max() {
        let mut s = PollingStrategy::new(adaptive(100, 2000, 0.0));
        // Many stable readings should not exceed max
        s.record_poll(make_result(0, 5, 0));
        for i in 1..=20_u64 {
            s.record_poll(make_result(0, 5, i * 2000));
        }
        assert!(s.states[&0].current_interval_ms <= 2000);
    }

    #[test]
    fn test_adaptive_change_threshold_respected() {
        // Only changes > 5.0 are considered actual changes
        let mut s = PollingStrategy::new(adaptive(100, 5000, 5.0));
        s.record_poll(make_result(0, 100, 0));
        // Small delta — should NOT be considered a change
        s.record_poll(make_result(0, 103, 100)); // delta = 3 < 5
                                                 // With a large delta:
        s.record_poll(make_result(0, 120, 200)); // delta = 17 > 5
                                                 // Two of three polls "changed" in our logic:
                                                 // 1st poll: always changed (no prev)
                                                 // 2nd poll: not changed (3 < 5)
                                                 // 3rd poll: changed (17 > 5)
        let rate = s.change_rate(0);
        assert!((rate - 2.0 / 3.0).abs() < 1e-9);
    }

    // ── OnChange mode ─────────────────────────────────────────────────────────

    #[test]
    fn test_on_change_first_poll_ready() {
        let mut s = PollingStrategy::new(on_change(50));
        assert!(s.should_poll(0, 0));
    }

    #[test]
    fn test_on_change_debounce_respected() {
        let mut s = PollingStrategy::new(on_change(200));
        s.record_poll(make_result(0, 5, 1000));
        assert!(!s.should_poll(0, 1050));
        assert!(s.should_poll(0, 1200));
    }

    #[test]
    fn test_on_change_poll_count() {
        let mut s = PollingStrategy::new(on_change(100));
        for i in 0..5_u64 {
            s.record_poll(make_result(0, i as u16, i * 200));
        }
        assert_eq!(s.poll_count(0), 5);
    }

    // ── OnDemand mode ─────────────────────────────────────────────────────────

    #[test]
    fn test_on_demand_never_ready_automatically() {
        let mut s = PollingStrategy::new(PollingMode::OnDemand);
        s.record_poll(make_result(0, 1, 0));
        // next_poll_time should be u64::MAX
        assert_eq!(s.next_poll_time(0, u64::MAX - 1), u64::MAX);
        // should_poll checks now_ms >= next, so u64::MAX would be required
        assert!(!s.should_poll(0, u64::MAX - 1));
    }

    #[test]
    fn test_on_demand_record_poll_works() {
        let mut s = PollingStrategy::new(PollingMode::OnDemand);
        s.record_poll(make_result(5, 42, 100));
        assert_eq!(s.poll_count(5), 1);
    }

    // ── change_rate ───────────────────────────────────────────────────────────

    #[test]
    fn test_change_rate_unknown_register_is_zero() {
        let s = PollingStrategy::new(fixed(100));
        assert_eq!(s.change_rate(999), 0.0);
    }

    #[test]
    fn test_change_rate_all_changes() {
        let mut s = PollingStrategy::new(fixed(10));
        for i in 0..5_u16 {
            s.record_poll(make_result(0, i, u64::from(i) * 10));
        }
        // All polls registered a change (first has no prev, rest changed value)
        assert_eq!(s.change_rate(0), 1.0);
    }

    // ── registers() ───────────────────────────────────────────────────────────

    #[test]
    fn test_registers_empty_initially() {
        let s = PollingStrategy::new(fixed(100));
        assert!(s.registers().is_empty());
    }

    #[test]
    fn test_registers_sorted() {
        let mut s = PollingStrategy::new(fixed(100));
        s.record_poll(make_result(300, 0, 0));
        s.record_poll(make_result(100, 0, 0));
        s.record_poll(make_result(200, 0, 0));
        assert_eq!(s.registers(), vec![100, 200, 300]);
    }

    #[test]
    fn test_registers_after_should_poll() {
        let mut s = PollingStrategy::new(fixed(100));
        let _ = s.should_poll(42, 0); // initialises state
        assert!(s.registers().contains(&42));
    }

    // ── next_poll_time ────────────────────────────────────────────────────────

    #[test]
    fn test_next_poll_time_unknown_register_is_now() {
        let s = PollingStrategy::new(fixed(1000));
        assert_eq!(s.next_poll_time(99, 500), 500);
    }

    #[test]
    fn test_next_poll_time_fixed_after_first_poll() {
        let mut s = PollingStrategy::new(fixed(1000));
        s.record_poll(make_result(0, 5, 2000));
        assert_eq!(s.next_poll_time(0, 3000), 3000);
    }

    #[test]
    fn test_next_poll_time_on_demand_is_max() {
        let mut s = PollingStrategy::new(PollingMode::OnDemand);
        s.record_poll(make_result(0, 0, 0));
        assert_eq!(s.next_poll_time(0, 0), u64::MAX);
    }

    // ── poll_count() ──────────────────────────────────────────────────────────

    #[test]
    fn test_poll_count_unknown_is_zero() {
        let s = PollingStrategy::new(fixed(100));
        assert_eq!(s.poll_count(77), 0);
    }

    // ── PollingMode clone / PartialEq ─────────────────────────────────────────

    #[test]
    fn test_polling_mode_equality() {
        assert_eq!(fixed(500), fixed(500));
        assert_ne!(fixed(500), fixed(1000));
    }

    #[test]
    fn test_polling_mode_clone() {
        let m = adaptive(100, 5000, 1.0);
        let m2 = m.clone();
        assert_eq!(m, m2);
    }

    // ── PollingState ──────────────────────────────────────────────────────────

    #[test]
    fn test_polling_state_initial_values() {
        let mut s = PollingStrategy::new(fixed(500));
        let _ = s.should_poll(0, 0); // triggers state creation
        let state = &s.states[&0];
        assert_eq!(state.register, 0);
        assert!(state.last_value.is_none());
        assert_eq!(state.poll_count, 0);
        assert_eq!(state.change_count, 0);
    }

    #[test]
    fn test_polling_state_updates_on_record() {
        let mut s = PollingStrategy::new(fixed(100));
        s.record_poll(make_result(10, 77, 1000));
        let state = &s.states[&10];
        assert_eq!(state.last_value, Some(77));
        assert_eq!(state.last_poll_ms, 1000);
        assert_eq!(state.poll_count, 1);
    }
}
