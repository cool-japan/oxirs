//! CAN bus error counting and state management.
//!
//! Implements TEC (Transmit Error Counter) and REC (Receive Error Counter)
//! tracking with the CAN specification's four-state error state machine:
//! Error Active → Error Warning → Error Passive → Bus Off.
//!
//! Provides error frame classification (bit, stuff, form, CRC, ACK),
//! bus recovery sequencing, configurable warning thresholds, and bus-load
//! estimation from the error rate.

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// CAN error types as defined in ISO 11898-1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CanErrorType {
    /// Bit error: transmitted bit differs from monitored bit.
    BitError,
    /// Stuff error: more than 5 consecutive identical bits.
    StuffError,
    /// Form error: fixed-form bit field violation.
    FormError,
    /// CRC error: received CRC does not match calculated CRC.
    CrcError,
    /// Acknowledgement error: no dominant bit detected in ACK slot.
    AckError,
}

impl CanErrorType {
    /// Return a human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            CanErrorType::BitError => "Bit Error",
            CanErrorType::StuffError => "Stuff Error",
            CanErrorType::FormError => "Form Error",
            CanErrorType::CrcError => "CRC Error",
            CanErrorType::AckError => "ACK Error",
        }
    }
}

/// CAN error state machine states per ISO 11898-1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorState {
    /// TEC ≤ 127 and REC ≤ 127. Node participates normally.
    ErrorActive,
    /// TEC or REC in [96, 127]. Warning threshold exceeded (non-standard
    /// but widely supported). Still participates, but software should alert.
    ErrorWarning,
    /// TEC > 127 or REC > 127. Node sends passive error flags.
    ErrorPassive,
    /// TEC ≥ 256. Node is disconnected from the bus.
    BusOff,
}

impl ErrorState {
    /// `true` if the node is allowed to transmit frames.
    pub fn can_transmit(&self) -> bool {
        matches!(
            self,
            ErrorState::ErrorActive | ErrorState::ErrorWarning | ErrorState::ErrorPassive
        )
    }

    /// `true` if the node is fully disconnected.
    pub fn is_bus_off(&self) -> bool {
        matches!(self, ErrorState::BusOff)
    }
}

/// A logged state transition.
#[derive(Debug, Clone)]
pub struct StateTransition {
    /// Timestamp (epoch milliseconds) when the transition occurred.
    pub timestamp_ms: u64,
    /// Previous error state.
    pub from: ErrorState,
    /// New error state.
    pub to: ErrorState,
    /// TEC at the time of transition.
    pub tec: u16,
    /// REC at the time of transition.
    pub rec: u16,
}

/// An error frame record.
#[derive(Debug, Clone)]
pub struct ErrorFrame {
    /// Timestamp (epoch milliseconds).
    pub timestamp_ms: u64,
    /// The error type detected.
    pub error_type: CanErrorType,
    /// Whether this error occurred during transmission (`true`) or reception (`false`).
    pub is_transmit: bool,
}

/// Aggregate error statistics.
#[derive(Debug, Clone, Default)]
pub struct ErrorStats {
    /// Total error frames processed.
    pub total_errors: u64,
    /// Per-error-type counts.
    pub per_type: HashMap<CanErrorType, u64>,
    /// Number of times the node entered Bus Off state.
    pub bus_off_count: u64,
    /// Number of successful bus recoveries.
    pub recovery_count: u64,
    /// Number of state transitions logged.
    pub transitions_logged: u64,
    /// Total transmit errors.
    pub transmit_errors: u64,
    /// Total receive errors.
    pub receive_errors: u64,
}

/// Configuration for the error counter.
#[derive(Debug, Clone)]
pub struct ErrorCounterConfig {
    /// Warning threshold for TEC/REC (default: 96 per CAN spec).
    pub warning_threshold: u16,
    /// Bus Off threshold for TEC (default: 256 per CAN spec).
    pub bus_off_threshold: u16,
    /// Number of 11-recessive-bit sequences needed for bus recovery (default: 128).
    pub recovery_sequences: u32,
    /// Maximum number of state transitions to retain in the log.
    pub max_transition_log: usize,
}

impl Default for ErrorCounterConfig {
    fn default() -> Self {
        Self {
            warning_threshold: 96,
            bus_off_threshold: 256,
            recovery_sequences: 128,
            max_transition_log: 1_000,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ErrorCounter
// ──────────────────────────────────────────────────────────────────────────────

/// CAN bus error counter implementing the ISO 11898-1 error state machine.
pub struct ErrorCounter {
    config: ErrorCounterConfig,
    /// Transmit Error Counter (TEC).
    tec: u16,
    /// Receive Error Counter (REC).
    rec: u16,
    /// Current error state.
    state: ErrorState,
    /// Number of recovery sequences received so far (while in Bus Off).
    recovery_progress: u32,
    /// State transition log.
    transitions: Vec<StateTransition>,
    /// Aggregate statistics.
    stats: ErrorStats,
    /// Timestamps of recent errors for rate calculation.
    recent_error_timestamps: Vec<u64>,
}

impl ErrorCounter {
    /// Create a new error counter with the given configuration.
    pub fn new(config: ErrorCounterConfig) -> Self {
        Self {
            config,
            tec: 0,
            rec: 0,
            state: ErrorState::ErrorActive,
            recovery_progress: 0,
            transitions: Vec::new(),
            stats: ErrorStats::default(),
            recent_error_timestamps: Vec::new(),
        }
    }

    /// Create an error counter with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ErrorCounterConfig::default())
    }

    /// Current Transmit Error Counter value.
    pub fn tec(&self) -> u16 {
        self.tec
    }

    /// Current Receive Error Counter value.
    pub fn rec(&self) -> u16 {
        self.rec
    }

    /// Current error state.
    pub fn state(&self) -> ErrorState {
        self.state
    }

    /// Return the aggregate error statistics.
    pub fn stats(&self) -> &ErrorStats {
        &self.stats
    }

    /// Return the state transition log.
    pub fn transitions(&self) -> &[StateTransition] {
        &self.transitions
    }

    /// Recovery progress as a fraction [0.0, 1.0].
    pub fn recovery_progress(&self) -> f64 {
        if self.config.recovery_sequences == 0 {
            return 1.0;
        }
        self.recovery_progress as f64 / self.config.recovery_sequences as f64
    }

    /// Process an error frame: update counters and state machine.
    pub fn process_error(&mut self, frame: &ErrorFrame) {
        self.stats.total_errors += 1;
        *self.stats.per_type.entry(frame.error_type).or_insert(0) += 1;
        self.recent_error_timestamps.push(frame.timestamp_ms);

        if frame.is_transmit {
            self.stats.transmit_errors += 1;
            self.increment_tec(frame);
        } else {
            self.stats.receive_errors += 1;
            self.increment_rec(frame);
        }

        self.update_state(frame.timestamp_ms);
    }

    /// Record a successful transmission (decrements TEC by 1).
    pub fn record_successful_transmit(&mut self, timestamp_ms: u64) {
        if self.tec > 0 {
            self.tec -= 1;
        }
        self.update_state(timestamp_ms);
    }

    /// Record a successful reception (decrements REC by 1, minimum 0).
    pub fn record_successful_receive(&mut self, timestamp_ms: u64) {
        if self.rec > 0 {
            self.rec -= 1;
        }
        self.update_state(timestamp_ms);
    }

    /// Signal that one 11-recessive-bit sequence has been observed during
    /// Bus Off recovery. After `recovery_sequences` such sequences, the
    /// node returns to Error Active.
    pub fn signal_recovery_sequence(&mut self, timestamp_ms: u64) {
        if self.state != ErrorState::BusOff {
            return;
        }
        self.recovery_progress += 1;
        if self.recovery_progress >= self.config.recovery_sequences {
            // Bus recovery complete
            self.tec = 0;
            self.rec = 0;
            self.recovery_progress = 0;
            self.stats.recovery_count += 1;
            let old_state = self.state;
            self.state = ErrorState::ErrorActive;
            self.log_transition(timestamp_ms, old_state, self.state);
        }
    }

    /// Estimate the error rate (errors per second) over a time window.
    ///
    /// `window_ms` defines how far back in time to look.
    pub fn error_rate(&self, window_ms: u64) -> f64 {
        if self.recent_error_timestamps.is_empty() || window_ms == 0 {
            return 0.0;
        }
        let latest = self.recent_error_timestamps.last().copied().unwrap_or(0);
        let cutoff = latest.saturating_sub(window_ms);
        let count = self
            .recent_error_timestamps
            .iter()
            .filter(|&&t| t > cutoff)
            .count();
        (count as f64 * 1000.0) / window_ms as f64
    }

    /// Estimate bus load percentage from the error rate.
    ///
    /// Assumes each error frame occupies approximately 20 bit-times on a
    /// 500 kbps bus (nominal CAN). Returns a value in [0.0, 100.0].
    pub fn estimated_bus_load_percent(&self, window_ms: u64, bus_speed_bps: u64) -> f64 {
        if bus_speed_bps == 0 || window_ms == 0 {
            return 0.0;
        }
        let error_rate = self.error_rate(window_ms);
        let error_bits_per_second = error_rate * 20.0; // ~20 bit-times per error frame
        let load = (error_bits_per_second / bus_speed_bps as f64) * 100.0;
        load.min(100.0)
    }

    /// Reset all counters and state to initial values.
    pub fn reset(&mut self) {
        self.tec = 0;
        self.rec = 0;
        self.state = ErrorState::ErrorActive;
        self.recovery_progress = 0;
        self.transitions.clear();
        self.stats = ErrorStats::default();
        self.recent_error_timestamps.clear();
    }

    /// Clear the transition log while keeping counters.
    pub fn clear_transition_log(&mut self) {
        self.transitions.clear();
    }

    // ── Private ──────────────────────────────────────────────────────────────

    /// Increment TEC based on error type (CAN spec rules).
    fn increment_tec(&mut self, frame: &ErrorFrame) {
        let increment = match frame.error_type {
            CanErrorType::BitError | CanErrorType::AckError => 8,
            CanErrorType::StuffError | CanErrorType::FormError | CanErrorType::CrcError => 8,
        };
        self.tec = self.tec.saturating_add(increment);
    }

    /// Increment REC based on error type (CAN spec rules).
    fn increment_rec(&mut self, frame: &ErrorFrame) {
        let increment = match frame.error_type {
            CanErrorType::BitError => 1,
            CanErrorType::StuffError | CanErrorType::FormError => 1,
            CanErrorType::CrcError => 8,
            CanErrorType::AckError => 1,
        };
        self.rec = self.rec.saturating_add(increment);
    }

    /// Re-evaluate the error state after counter changes.
    fn update_state(&mut self, timestamp_ms: u64) {
        let old_state = self.state;
        let new_state = self.compute_state();

        if new_state != old_state {
            if new_state == ErrorState::BusOff {
                self.stats.bus_off_count += 1;
                self.recovery_progress = 0;
            }
            self.state = new_state;
            self.log_transition(timestamp_ms, old_state, new_state);
        }
    }

    /// Compute the current state from TEC and REC values.
    fn compute_state(&self) -> ErrorState {
        if self.tec >= self.config.bus_off_threshold {
            return ErrorState::BusOff;
        }
        if self.tec > 127 || self.rec > 127 {
            return ErrorState::ErrorPassive;
        }
        if self.tec >= self.config.warning_threshold || self.rec >= self.config.warning_threshold {
            return ErrorState::ErrorWarning;
        }
        ErrorState::ErrorActive
    }

    /// Append a state transition to the log (bounded).
    fn log_transition(&mut self, timestamp_ms: u64, from: ErrorState, to: ErrorState) {
        self.stats.transitions_logged += 1;
        if self.transitions.len() >= self.config.max_transition_log {
            self.transitions.remove(0);
        }
        self.transitions.push(StateTransition {
            timestamp_ms,
            from,
            to,
            tec: self.tec,
            rec: self.rec,
        });
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tx_error(error_type: CanErrorType, timestamp_ms: u64) -> ErrorFrame {
        ErrorFrame {
            timestamp_ms,
            error_type,
            is_transmit: true,
        }
    }

    fn rx_error(error_type: CanErrorType, timestamp_ms: u64) -> ErrorFrame {
        ErrorFrame {
            timestamp_ms,
            error_type,
            is_transmit: false,
        }
    }

    // ── CanErrorType ─────────────────────────────────────────────────────────

    #[test]
    fn test_error_type_labels() {
        assert_eq!(CanErrorType::BitError.label(), "Bit Error");
        assert_eq!(CanErrorType::StuffError.label(), "Stuff Error");
        assert_eq!(CanErrorType::FormError.label(), "Form Error");
        assert_eq!(CanErrorType::CrcError.label(), "CRC Error");
        assert_eq!(CanErrorType::AckError.label(), "ACK Error");
    }

    #[test]
    fn test_error_type_eq_hash() {
        let a = CanErrorType::BitError;
        let b = CanErrorType::BitError;
        assert_eq!(a, b);

        let mut map = HashMap::new();
        map.insert(CanErrorType::CrcError, 42);
        assert_eq!(map.get(&CanErrorType::CrcError), Some(&42));
    }

    // ── ErrorState ───────────────────────────────────────────────────────────

    #[test]
    fn test_error_state_can_transmit() {
        assert!(ErrorState::ErrorActive.can_transmit());
        assert!(ErrorState::ErrorWarning.can_transmit());
        assert!(ErrorState::ErrorPassive.can_transmit());
        assert!(!ErrorState::BusOff.can_transmit());
    }

    #[test]
    fn test_error_state_is_bus_off() {
        assert!(!ErrorState::ErrorActive.is_bus_off());
        assert!(!ErrorState::ErrorWarning.is_bus_off());
        assert!(!ErrorState::ErrorPassive.is_bus_off());
        assert!(ErrorState::BusOff.is_bus_off());
    }

    // ── ErrorCounter creation ────────────────────────────────────────────────

    #[test]
    fn test_counter_creation() {
        let counter = ErrorCounter::with_defaults();
        assert_eq!(counter.tec(), 0);
        assert_eq!(counter.rec(), 0);
        assert_eq!(counter.state(), ErrorState::ErrorActive);
    }

    #[test]
    fn test_counter_custom_config() {
        let cfg = ErrorCounterConfig {
            warning_threshold: 50,
            bus_off_threshold: 200,
            recovery_sequences: 64,
            max_transition_log: 500,
        };
        let counter = ErrorCounter::new(cfg);
        assert_eq!(counter.state(), ErrorState::ErrorActive);
    }

    // ── TEC increments (transmit errors) ─────────────────────────────────────

    #[test]
    fn test_tec_increment_bit_error() {
        let mut counter = ErrorCounter::with_defaults();
        counter.process_error(&tx_error(CanErrorType::BitError, 100));
        assert_eq!(counter.tec(), 8);
    }

    #[test]
    fn test_tec_increment_stuff_error() {
        let mut counter = ErrorCounter::with_defaults();
        counter.process_error(&tx_error(CanErrorType::StuffError, 100));
        assert_eq!(counter.tec(), 8);
    }

    #[test]
    fn test_tec_increment_form_error() {
        let mut counter = ErrorCounter::with_defaults();
        counter.process_error(&tx_error(CanErrorType::FormError, 100));
        assert_eq!(counter.tec(), 8);
    }

    #[test]
    fn test_tec_increment_crc_error() {
        let mut counter = ErrorCounter::with_defaults();
        counter.process_error(&tx_error(CanErrorType::CrcError, 100));
        assert_eq!(counter.tec(), 8);
    }

    #[test]
    fn test_tec_increment_ack_error() {
        let mut counter = ErrorCounter::with_defaults();
        counter.process_error(&tx_error(CanErrorType::AckError, 100));
        assert_eq!(counter.tec(), 8);
    }

    // ── REC increments (receive errors) ──────────────────────────────────────

    #[test]
    fn test_rec_increment_bit_error() {
        let mut counter = ErrorCounter::with_defaults();
        counter.process_error(&rx_error(CanErrorType::BitError, 100));
        assert_eq!(counter.rec(), 1);
    }

    #[test]
    fn test_rec_increment_crc_error() {
        let mut counter = ErrorCounter::with_defaults();
        counter.process_error(&rx_error(CanErrorType::CrcError, 100));
        assert_eq!(counter.rec(), 8);
    }

    #[test]
    fn test_rec_increment_stuff_error() {
        let mut counter = ErrorCounter::with_defaults();
        counter.process_error(&rx_error(CanErrorType::StuffError, 100));
        assert_eq!(counter.rec(), 1);
    }

    // ── State transitions ────────────────────────────────────────────────────

    #[test]
    fn test_error_active_to_warning() {
        let mut counter = ErrorCounter::with_defaults();
        // 12 transmit bit errors → TEC = 96 → Warning threshold
        for i in 0..12 {
            counter.process_error(&tx_error(CanErrorType::BitError, i * 10));
        }
        assert_eq!(counter.tec(), 96);
        assert_eq!(counter.state(), ErrorState::ErrorWarning);
    }

    #[test]
    fn test_warning_to_error_passive() {
        let mut counter = ErrorCounter::with_defaults();
        // 16 transmit errors → TEC = 128 → Error Passive
        for i in 0..16 {
            counter.process_error(&tx_error(CanErrorType::BitError, i * 10));
        }
        assert_eq!(counter.tec(), 128);
        assert_eq!(counter.state(), ErrorState::ErrorPassive);
    }

    #[test]
    fn test_error_passive_to_bus_off() {
        let mut counter = ErrorCounter::with_defaults();
        // 32 transmit errors → TEC = 256 → Bus Off
        for i in 0..32 {
            counter.process_error(&tx_error(CanErrorType::BitError, i * 10));
        }
        assert_eq!(counter.tec(), 256);
        assert_eq!(counter.state(), ErrorState::BusOff);
    }

    #[test]
    fn test_rec_causes_warning() {
        let mut counter = ErrorCounter::with_defaults();
        // 96 receive CRC errors → REC = 768 (but only 12 needed for 96)
        // Actually REC increments by 8 for CRC, so 12 → REC=96
        for i in 0..12 {
            counter.process_error(&rx_error(CanErrorType::CrcError, i * 10));
        }
        assert_eq!(counter.rec(), 96);
        assert_eq!(counter.state(), ErrorState::ErrorWarning);
    }

    #[test]
    fn test_transition_logging() {
        let mut counter = ErrorCounter::with_defaults();
        // Go from Active → Warning
        for i in 0..12 {
            counter.process_error(&tx_error(CanErrorType::BitError, i * 10));
        }
        assert!(!counter.transitions().is_empty());
        let last = counter
            .transitions()
            .last()
            .expect("should have transition");
        assert_eq!(last.from, ErrorState::ErrorActive);
        assert_eq!(last.to, ErrorState::ErrorWarning);
    }

    #[test]
    fn test_transition_log_bounded() {
        let cfg = ErrorCounterConfig {
            max_transition_log: 2,
            ..Default::default()
        };
        let mut counter = ErrorCounter::new(cfg);
        // Force multiple transitions
        for i in 0..100 {
            counter.process_error(&tx_error(CanErrorType::BitError, i * 10));
        }
        assert!(counter.transitions().len() <= 2);
    }

    // ── Successful transmit/receive ──────────────────────────────────────────

    #[test]
    fn test_successful_transmit_decrements_tec() {
        let mut counter = ErrorCounter::with_defaults();
        counter.process_error(&tx_error(CanErrorType::BitError, 100));
        assert_eq!(counter.tec(), 8);
        counter.record_successful_transmit(200);
        assert_eq!(counter.tec(), 7);
    }

    #[test]
    fn test_successful_transmit_at_zero() {
        let mut counter = ErrorCounter::with_defaults();
        counter.record_successful_transmit(100);
        assert_eq!(counter.tec(), 0); // doesn't go negative
    }

    #[test]
    fn test_successful_receive_decrements_rec() {
        let mut counter = ErrorCounter::with_defaults();
        counter.process_error(&rx_error(CanErrorType::CrcError, 100));
        assert_eq!(counter.rec(), 8);
        counter.record_successful_receive(200);
        assert_eq!(counter.rec(), 7);
    }

    #[test]
    fn test_successful_receive_at_zero() {
        let mut counter = ErrorCounter::with_defaults();
        counter.record_successful_receive(100);
        assert_eq!(counter.rec(), 0);
    }

    // ── Bus Off recovery ─────────────────────────────────────────────────────

    #[test]
    fn test_bus_off_recovery_full() {
        let mut counter = ErrorCounter::with_defaults();
        // Drive to Bus Off
        for i in 0..32 {
            counter.process_error(&tx_error(CanErrorType::BitError, i * 10));
        }
        assert_eq!(counter.state(), ErrorState::BusOff);

        // 128 recovery sequences
        for i in 0..128 {
            counter.signal_recovery_sequence(1000 + i);
        }
        assert_eq!(counter.state(), ErrorState::ErrorActive);
        assert_eq!(counter.tec(), 0);
        assert_eq!(counter.rec(), 0);
        assert_eq!(counter.stats().recovery_count, 1);
    }

    #[test]
    fn test_recovery_progress() {
        let mut counter = ErrorCounter::with_defaults();
        for i in 0..32 {
            counter.process_error(&tx_error(CanErrorType::BitError, i * 10));
        }
        assert_eq!(counter.state(), ErrorState::BusOff);

        for i in 0..64 {
            counter.signal_recovery_sequence(1000 + i);
        }
        let progress = counter.recovery_progress();
        assert!((progress - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_recovery_sequence_ignored_when_not_bus_off() {
        let mut counter = ErrorCounter::with_defaults();
        counter.signal_recovery_sequence(100);
        assert_eq!(counter.state(), ErrorState::ErrorActive);
    }

    // ── Error rate ───────────────────────────────────────────────────────────

    #[test]
    fn test_error_rate() {
        let mut counter = ErrorCounter::with_defaults();
        for i in 0..10 {
            counter.process_error(&tx_error(CanErrorType::BitError, i * 100));
        }
        // 10 errors in 900ms window
        let rate = counter.error_rate(1000);
        assert!(rate > 0.0);
    }

    #[test]
    fn test_error_rate_empty() {
        let counter = ErrorCounter::with_defaults();
        assert!((counter.error_rate(1000) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_error_rate_zero_window() {
        let mut counter = ErrorCounter::with_defaults();
        counter.process_error(&tx_error(CanErrorType::BitError, 100));
        assert!((counter.error_rate(0) - 0.0).abs() < f64::EPSILON);
    }

    // ── Bus load estimation ──────────────────────────────────────────────────

    #[test]
    fn test_bus_load_estimation() {
        let mut counter = ErrorCounter::with_defaults();
        for i in 0..100 {
            counter.process_error(&tx_error(CanErrorType::BitError, i));
        }
        let load = counter.estimated_bus_load_percent(1000, 500_000);
        assert!((0.0..=100.0).contains(&load));
    }

    #[test]
    fn test_bus_load_zero_speed() {
        let counter = ErrorCounter::with_defaults();
        assert!((counter.estimated_bus_load_percent(1000, 0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bus_load_zero_window() {
        let counter = ErrorCounter::with_defaults();
        assert!((counter.estimated_bus_load_percent(0, 500_000) - 0.0).abs() < f64::EPSILON);
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_total_errors() {
        let mut counter = ErrorCounter::with_defaults();
        counter.process_error(&tx_error(CanErrorType::BitError, 100));
        counter.process_error(&rx_error(CanErrorType::CrcError, 200));
        assert_eq!(counter.stats().total_errors, 2);
    }

    #[test]
    fn test_stats_per_type() {
        let mut counter = ErrorCounter::with_defaults();
        counter.process_error(&tx_error(CanErrorType::BitError, 100));
        counter.process_error(&tx_error(CanErrorType::BitError, 200));
        counter.process_error(&rx_error(CanErrorType::CrcError, 300));
        assert_eq!(
            counter.stats().per_type.get(&CanErrorType::BitError),
            Some(&2)
        );
        assert_eq!(
            counter.stats().per_type.get(&CanErrorType::CrcError),
            Some(&1)
        );
    }

    #[test]
    fn test_stats_transmit_receive_split() {
        let mut counter = ErrorCounter::with_defaults();
        counter.process_error(&tx_error(CanErrorType::BitError, 100));
        counter.process_error(&tx_error(CanErrorType::AckError, 200));
        counter.process_error(&rx_error(CanErrorType::CrcError, 300));
        assert_eq!(counter.stats().transmit_errors, 2);
        assert_eq!(counter.stats().receive_errors, 1);
    }

    #[test]
    fn test_stats_bus_off_count() {
        let mut counter = ErrorCounter::with_defaults();
        for i in 0..32 {
            counter.process_error(&tx_error(CanErrorType::BitError, i * 10));
        }
        assert_eq!(counter.stats().bus_off_count, 1);
    }

    #[test]
    fn test_stats_default() {
        let s = ErrorStats::default();
        assert_eq!(s.total_errors, 0);
        assert!(s.per_type.is_empty());
        assert_eq!(s.bus_off_count, 0);
    }

    // ── Reset ────────────────────────────────────────────────────────────────

    #[test]
    fn test_reset() {
        let mut counter = ErrorCounter::with_defaults();
        for i in 0..10 {
            counter.process_error(&tx_error(CanErrorType::BitError, i * 10));
        }
        counter.reset();
        assert_eq!(counter.tec(), 0);
        assert_eq!(counter.rec(), 0);
        assert_eq!(counter.state(), ErrorState::ErrorActive);
        assert!(counter.transitions().is_empty());
        assert_eq!(counter.stats().total_errors, 0);
    }

    #[test]
    fn test_clear_transition_log() {
        let mut counter = ErrorCounter::with_defaults();
        for i in 0..20 {
            counter.process_error(&tx_error(CanErrorType::BitError, i * 10));
        }
        let had_transitions = !counter.transitions().is_empty();
        counter.clear_transition_log();
        assert!(counter.transitions().is_empty());
        assert!(had_transitions);
    }

    // ── Config defaults ──────────────────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = ErrorCounterConfig::default();
        assert_eq!(cfg.warning_threshold, 96);
        assert_eq!(cfg.bus_off_threshold, 256);
        assert_eq!(cfg.recovery_sequences, 128);
        assert_eq!(cfg.max_transition_log, 1_000);
    }

    // ── Custom warning threshold ─────────────────────────────────────────────

    #[test]
    fn test_custom_warning_threshold() {
        let cfg = ErrorCounterConfig {
            warning_threshold: 40,
            ..Default::default()
        };
        let mut counter = ErrorCounter::new(cfg);
        // 5 tx errors → TEC = 40 → Warning at threshold 40
        for i in 0..5 {
            counter.process_error(&tx_error(CanErrorType::BitError, i * 10));
        }
        assert_eq!(counter.tec(), 40);
        assert_eq!(counter.state(), ErrorState::ErrorWarning);
    }

    // ── StateTransition clone ────────────────────────────────────────────────

    #[test]
    fn test_state_transition_clone() {
        let t = StateTransition {
            timestamp_ms: 100,
            from: ErrorState::ErrorActive,
            to: ErrorState::ErrorWarning,
            tec: 96,
            rec: 0,
        };
        let t2 = t.clone();
        assert_eq!(t2.timestamp_ms, 100);
        assert_eq!(t2.from, ErrorState::ErrorActive);
        assert_eq!(t2.to, ErrorState::ErrorWarning);
    }

    // ── ErrorFrame clone ─────────────────────────────────────────────────────

    #[test]
    fn test_error_frame_clone() {
        let f = ErrorFrame {
            timestamp_ms: 100,
            error_type: CanErrorType::BitError,
            is_transmit: true,
        };
        let f2 = f.clone();
        assert_eq!(f2.timestamp_ms, 100);
        assert!(f2.is_transmit);
    }

    // ── Recovery after Bus Off resets counters ────────────────────────────────

    #[test]
    fn test_recovery_resets_counters() {
        let mut counter = ErrorCounter::with_defaults();
        // Drive to Bus Off
        for i in 0..32 {
            counter.process_error(&tx_error(CanErrorType::BitError, i * 10));
        }
        // Add some REC
        // Note: once Bus Off, further errors still accumulate
        assert!(counter.tec() >= 256);

        // Full recovery
        for i in 0..128 {
            counter.signal_recovery_sequence(1000 + i);
        }
        assert_eq!(counter.tec(), 0);
        assert_eq!(counter.rec(), 0);
    }

    // ── Multiple Bus Off / Recovery cycles ───────────────────────────────────

    #[test]
    fn test_multiple_bus_off_recovery_cycles() {
        let mut counter = ErrorCounter::with_defaults();

        for _cycle in 0..3 {
            // Drive to Bus Off
            for i in 0..32 {
                counter.process_error(&tx_error(CanErrorType::BitError, i * 10));
            }
            assert_eq!(counter.state(), ErrorState::BusOff);

            // Recover
            for i in 0..128 {
                counter.signal_recovery_sequence(2000 + i);
            }
            assert_eq!(counter.state(), ErrorState::ErrorActive);
        }
        assert_eq!(counter.stats().bus_off_count, 3);
        assert_eq!(counter.stats().recovery_count, 3);
    }
}
