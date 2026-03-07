//! Register value validation for Modbus devices.
//!
//! Provides range checking, data type validation, scaling factor application,
//! unit conversion, alarm threshold checking (HH/H/L/LL), rate-of-change
//! detection, dead-band filtering, and configurable validation rules.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Modbus register data type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegisterDataType {
    /// Signed 16-bit integer (single register).
    Int16,
    /// Unsigned 16-bit integer (single register).
    Uint16,
    /// Signed 32-bit integer (two registers).
    Int32,
    /// Unsigned 32-bit integer (two registers).
    Uint32,
    /// IEEE 754 32-bit float (two registers).
    Float32,
    /// Boolean (bit from a register).
    Bool,
}

impl RegisterDataType {
    /// Number of 16-bit registers occupied by this data type.
    pub fn register_count(&self) -> usize {
        match self {
            RegisterDataType::Int16 | RegisterDataType::Uint16 | RegisterDataType::Bool => 1,
            RegisterDataType::Int32 | RegisterDataType::Uint32 | RegisterDataType::Float32 => 2,
        }
    }
}

/// Scaling model: `engineering_value = raw * scale + offset`.
#[derive(Debug, Clone)]
pub struct ScalingConfig {
    /// Multiplicative factor.
    pub scale: f64,
    /// Additive offset.
    pub offset: f64,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            offset: 0.0,
        }
    }
}

impl ScalingConfig {
    /// Apply the scaling formula.
    pub fn apply(&self, raw: f64) -> f64 {
        raw * self.scale + self.offset
    }

    /// Reverse the scaling formula: `raw = (eng - offset) / scale`.
    pub fn reverse(&self, eng: f64) -> Option<f64> {
        if self.scale.abs() < f64::EPSILON {
            return None;
        }
        Some((eng - self.offset) / self.scale)
    }
}

/// Alarm thresholds following ISA-18.2 conventions.
#[derive(Debug, Clone, Default)]
pub struct AlarmThresholds {
    /// High-high threshold (critical).
    pub high_high: Option<f64>,
    /// High threshold (warning).
    pub high: Option<f64>,
    /// Low threshold (warning).
    pub low: Option<f64>,
    /// Low-low threshold (critical).
    pub low_low: Option<f64>,
}

/// Alarm severity levels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlarmLevel {
    /// Value exceeds high-high threshold.
    HighHigh,
    /// Value exceeds high threshold.
    High,
    /// Value is below low threshold.
    Low,
    /// Value is below low-low threshold.
    LowLow,
}

/// Rate-of-change configuration.
#[derive(Debug, Clone)]
pub struct RocConfig {
    /// Maximum allowed rate of change per second.
    pub max_rate_per_sec: f64,
}

/// Dead-band configuration.
#[derive(Debug, Clone)]
pub struct DeadBandConfig {
    /// Minimum absolute change required to consider the value as changed.
    pub absolute: Option<f64>,
    /// Minimum percentage change required (0.0 - 100.0).
    pub percentage: Option<f64>,
}

/// A validation rule for a specific register.
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Register address.
    pub address: u16,
    /// Human-readable name.
    pub name: String,
    /// Expected data type.
    pub data_type: RegisterDataType,
    /// Minimum valid raw value.
    pub min_raw: Option<f64>,
    /// Maximum valid raw value.
    pub max_raw: Option<f64>,
    /// Scaling configuration.
    pub scaling: ScalingConfig,
    /// Engineering unit string (e.g., "degC", "bar", "rpm").
    pub unit: String,
    /// Alarm thresholds (applied to engineering value).
    pub alarms: AlarmThresholds,
    /// Rate-of-change limit.
    pub roc: Option<RocConfig>,
    /// Dead-band filter.
    pub dead_band: Option<DeadBandConfig>,
}

/// Result of validating a single register value.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Register address.
    pub address: u16,
    /// Rule name.
    pub name: String,
    /// Raw value as f64.
    pub raw_value: f64,
    /// Engineering value after scaling.
    pub engineering_value: f64,
    /// Engineering unit.
    pub unit: String,
    /// Whether the value is within the raw range.
    pub in_range: bool,
    /// Active alarm levels (may be empty).
    pub alarms: Vec<AlarmLevel>,
    /// Whether the rate-of-change limit was exceeded.
    pub roc_exceeded: bool,
    /// Whether the value passed the dead-band filter (i.e., changed enough).
    pub dead_band_passed: bool,
    /// Overall validity: in_range && no alarms && !roc_exceeded.
    pub is_valid: bool,
}

/// Errors from validation operations.
#[derive(Debug)]
pub enum ValidatorError {
    /// No rule found for the given register address.
    NoRule(u16),
    /// Data type mismatch.
    TypeMismatch(String),
    /// Invalid configuration.
    InvalidConfig(String),
}

impl std::fmt::Display for ValidatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidatorError::NoRule(addr) => write!(f, "no validation rule for register {addr}"),
            ValidatorError::TypeMismatch(msg) => write!(f, "type mismatch: {msg}"),
            ValidatorError::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for ValidatorError {}

// ─────────────────────────────────────────────────────────────────────────────
// RegisterValidator
// ─────────────────────────────────────────────────────────────────────────────

/// Validates Modbus register values against configurable rules.
pub struct RegisterValidator {
    /// Rules keyed by register address.
    rules: HashMap<u16, ValidationRule>,
    /// Last known engineering values (for rate-of-change and dead-band).
    last_values: HashMap<u16, f64>,
    /// Last timestamp per register (ms) for rate-of-change calculation.
    last_timestamps: HashMap<u16, u64>,
    /// Validation statistics.
    total_validated: u64,
    total_failures: u64,
    total_alarms: u64,
}

impl Default for RegisterValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl RegisterValidator {
    /// Create a new empty validator.
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            last_values: HashMap::new(),
            last_timestamps: HashMap::new(),
            total_validated: 0,
            total_failures: 0,
            total_alarms: 0,
        }
    }

    /// Add a validation rule.
    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.rules.insert(rule.address, rule);
    }

    /// Remove a rule by register address.
    pub fn remove_rule(&mut self, address: u16) -> bool {
        self.rules.remove(&address).is_some()
    }

    /// Get a rule by register address.
    pub fn get_rule(&self, address: u16) -> Option<&ValidationRule> {
        self.rules.get(&address)
    }

    /// Number of configured rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Total validations performed.
    pub fn total_validated(&self) -> u64 {
        self.total_validated
    }

    /// Total validation failures.
    pub fn total_failures(&self) -> u64 {
        self.total_failures
    }

    /// Total alarm triggers.
    pub fn total_alarms(&self) -> u64 {
        self.total_alarms
    }

    // ─── Raw Value Conversion ────────────────────────────────────────────

    /// Convert a raw u16 value to f64 based on data type.
    pub fn raw_to_f64(data_type: &RegisterDataType, raw: u16) -> f64 {
        match data_type {
            RegisterDataType::Int16 => raw as i16 as f64,
            RegisterDataType::Uint16 => raw as f64,
            RegisterDataType::Bool => {
                if raw != 0 {
                    1.0
                } else {
                    0.0
                }
            }
            // For 32-bit types, a single u16 is only the high or low word.
            // In production this would need two registers; here we treat it as u16 value.
            RegisterDataType::Int32 => raw as i16 as f64,
            RegisterDataType::Uint32 => raw as f64,
            RegisterDataType::Float32 => raw as f64,
        }
    }

    /// Convert a pair of u16 registers to f64 for 32-bit types.
    pub fn raw_pair_to_f64(data_type: &RegisterDataType, high: u16, low: u16) -> f64 {
        let combined = ((high as u32) << 16) | (low as u32);
        match data_type {
            RegisterDataType::Int32 => combined as i32 as f64,
            RegisterDataType::Uint32 => combined as f64,
            RegisterDataType::Float32 => f32::from_bits(combined) as f64,
            _ => high as f64,
        }
    }

    // ─── Validation ──────────────────────────────────────────────────────

    /// Validate a single register value.
    ///
    /// `raw_value` is the f64 representation of the raw register value.
    /// `now_ms` is the current timestamp for rate-of-change calculation.
    pub fn validate(
        &mut self,
        address: u16,
        raw_value: f64,
        now_ms: u64,
    ) -> Result<ValidationResult, ValidatorError> {
        let rule = self
            .rules
            .get(&address)
            .ok_or(ValidatorError::NoRule(address))?
            .clone();

        self.total_validated += 1;

        // Apply scaling
        let eng_value = rule.scaling.apply(raw_value);

        // Range check
        let in_range = self.check_range(raw_value, &rule);

        // Alarm check
        let alarms = self.check_alarms(eng_value, &rule.alarms);
        if !alarms.is_empty() {
            self.total_alarms += alarms.len() as u64;
        }

        // Rate-of-change check
        let roc_exceeded = self.check_roc(address, eng_value, now_ms, &rule);

        // Dead-band check
        let dead_band_passed = self.check_dead_band(address, eng_value, &rule);

        // Update last known values
        self.last_values.insert(address, eng_value);
        self.last_timestamps.insert(address, now_ms);

        let is_valid = in_range && alarms.is_empty() && !roc_exceeded;
        if !is_valid {
            self.total_failures += 1;
        }

        Ok(ValidationResult {
            address,
            name: rule.name.clone(),
            raw_value,
            engineering_value: eng_value,
            unit: rule.unit.clone(),
            in_range,
            alarms,
            roc_exceeded,
            dead_band_passed,
            is_valid,
        })
    }

    /// Validate a register value from raw u16.
    pub fn validate_u16(
        &mut self,
        address: u16,
        raw: u16,
        now_ms: u64,
    ) -> Result<ValidationResult, ValidatorError> {
        let rule = self
            .rules
            .get(&address)
            .ok_or(ValidatorError::NoRule(address))?;
        let raw_f64 = Self::raw_to_f64(&rule.data_type, raw);
        self.validate(address, raw_f64, now_ms)
    }

    /// Batch validate multiple registers.
    pub fn validate_batch(
        &mut self,
        readings: &[(u16, f64)],
        now_ms: u64,
    ) -> Vec<Result<ValidationResult, ValidatorError>> {
        readings
            .iter()
            .map(|&(addr, val)| self.validate(addr, val, now_ms))
            .collect()
    }

    // ─── Private helpers ─────────────────────────────────────────────────

    fn check_range(&self, raw_value: f64, rule: &ValidationRule) -> bool {
        if let Some(min) = rule.min_raw {
            if raw_value < min {
                return false;
            }
        }
        if let Some(max) = rule.max_raw {
            if raw_value > max {
                return false;
            }
        }
        true
    }

    fn check_alarms(&self, eng_value: f64, thresholds: &AlarmThresholds) -> Vec<AlarmLevel> {
        let mut alarms = Vec::new();

        if let Some(hh) = thresholds.high_high {
            if eng_value >= hh {
                alarms.push(AlarmLevel::HighHigh);
            }
        }
        if let Some(h) = thresholds.high {
            if eng_value >= h && !alarms.contains(&AlarmLevel::HighHigh) {
                alarms.push(AlarmLevel::High);
            }
        }
        if let Some(ll) = thresholds.low_low {
            if eng_value <= ll {
                alarms.push(AlarmLevel::LowLow);
            }
        }
        if let Some(l) = thresholds.low {
            if eng_value <= l && !alarms.contains(&AlarmLevel::LowLow) {
                alarms.push(AlarmLevel::Low);
            }
        }

        alarms
    }

    fn check_roc(&self, address: u16, eng_value: f64, now_ms: u64, rule: &ValidationRule) -> bool {
        let roc_config = match &rule.roc {
            Some(c) => c,
            None => return false,
        };

        let prev_value = match self.last_values.get(&address) {
            Some(v) => *v,
            None => return false,
        };
        let prev_ts = match self.last_timestamps.get(&address) {
            Some(t) => *t,
            None => return false,
        };

        let dt_sec = (now_ms.saturating_sub(prev_ts)) as f64 / 1000.0;
        if dt_sec < f64::EPSILON {
            return false;
        }

        let rate = (eng_value - prev_value).abs() / dt_sec;
        rate > roc_config.max_rate_per_sec
    }

    fn check_dead_band(&self, address: u16, eng_value: f64, rule: &ValidationRule) -> bool {
        let db_config = match &rule.dead_band {
            Some(c) => c,
            None => return true, // no dead-band => always passes
        };

        let prev_value = match self.last_values.get(&address) {
            Some(v) => *v,
            None => return true, // first reading always passes
        };

        let abs_change = (eng_value - prev_value).abs();

        // Check absolute dead-band.
        if let Some(abs_db) = db_config.absolute {
            if abs_change >= abs_db {
                return true;
            }
        }

        // Check percentage dead-band.
        if let Some(pct_db) = db_config.percentage {
            if prev_value.abs() > f64::EPSILON {
                let pct_change = (abs_change / prev_value.abs()) * 100.0;
                if pct_change >= pct_db {
                    return true;
                }
            }
        }

        false
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rule(address: u16, name: &str) -> ValidationRule {
        ValidationRule {
            address,
            name: name.to_string(),
            data_type: RegisterDataType::Uint16,
            min_raw: None,
            max_raw: None,
            scaling: ScalingConfig::default(),
            unit: "units".to_string(),
            alarms: AlarmThresholds::default(),
            roc: None,
            dead_band: None,
        }
    }

    fn make_validator() -> RegisterValidator {
        RegisterValidator::new()
    }

    // ── Data Type Tests ──────────────────────────────────────────────────

    #[test]
    fn test_register_count_16bit() {
        assert_eq!(RegisterDataType::Int16.register_count(), 1);
        assert_eq!(RegisterDataType::Uint16.register_count(), 1);
        assert_eq!(RegisterDataType::Bool.register_count(), 1);
    }

    #[test]
    fn test_register_count_32bit() {
        assert_eq!(RegisterDataType::Int32.register_count(), 2);
        assert_eq!(RegisterDataType::Uint32.register_count(), 2);
        assert_eq!(RegisterDataType::Float32.register_count(), 2);
    }

    #[test]
    fn test_raw_to_f64_uint16() {
        assert!(
            (RegisterValidator::raw_to_f64(&RegisterDataType::Uint16, 1000) - 1000.0).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_raw_to_f64_int16() {
        // 0xFFFF as i16 = -1
        assert!(
            (RegisterValidator::raw_to_f64(&RegisterDataType::Int16, 0xFFFF) - (-1.0)).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_raw_to_f64_bool() {
        assert!(
            (RegisterValidator::raw_to_f64(&RegisterDataType::Bool, 1) - 1.0).abs() < f64::EPSILON
        );
        assert!(
            (RegisterValidator::raw_to_f64(&RegisterDataType::Bool, 0) - 0.0).abs() < f64::EPSILON
        );
    }

    #[test]
    fn test_raw_pair_to_f64_int32() {
        // 0x00010000 = 65536 as i32
        let val = RegisterValidator::raw_pair_to_f64(&RegisterDataType::Int32, 1, 0);
        assert!((val - 65536.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_raw_pair_to_f64_uint32() {
        let val = RegisterValidator::raw_pair_to_f64(&RegisterDataType::Uint32, 0, 100);
        assert!((val - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_raw_pair_to_f64_float32() {
        // IEEE 754: 0x41200000 = 10.0f32
        let val = RegisterValidator::raw_pair_to_f64(&RegisterDataType::Float32, 0x4120, 0x0000);
        assert!((val - 10.0).abs() < 0.01);
    }

    // ── Scaling Tests ────────────────────────────────────────────────────

    #[test]
    fn test_scaling_identity() {
        let s = ScalingConfig::default();
        assert!((s.apply(42.0) - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scaling_apply() {
        let s = ScalingConfig {
            scale: 0.1,
            offset: -40.0,
        };
        // 250 * 0.1 + (-40) = -15.0
        assert!((s.apply(250.0) - (-15.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scaling_reverse() {
        let s = ScalingConfig {
            scale: 0.1,
            offset: -40.0,
        };
        let eng = -15.0;
        let raw = s.reverse(eng);
        assert!(raw.is_some());
        assert!((raw.expect("raw") - 250.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scaling_reverse_zero_scale() {
        let s = ScalingConfig {
            scale: 0.0,
            offset: 10.0,
        };
        assert!(s.reverse(5.0).is_none());
    }

    // ── Rule Management Tests ────────────────────────────────────────────

    #[test]
    fn test_add_rule() {
        let mut v = make_validator();
        v.add_rule(make_rule(40001, "temperature"));
        assert_eq!(v.rule_count(), 1);
    }

    #[test]
    fn test_get_rule() {
        let mut v = make_validator();
        v.add_rule(make_rule(40001, "temperature"));
        assert!(v.get_rule(40001).is_some());
        assert!(v.get_rule(40002).is_none());
    }

    #[test]
    fn test_remove_rule() {
        let mut v = make_validator();
        v.add_rule(make_rule(40001, "temperature"));
        assert!(v.remove_rule(40001));
        assert_eq!(v.rule_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_rule() {
        let mut v = make_validator();
        assert!(!v.remove_rule(99));
    }

    // ── Range Validation Tests ───────────────────────────────────────────

    #[test]
    fn test_validate_in_range() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.min_raw = Some(0.0);
        rule.max_raw = Some(1000.0);
        v.add_rule(rule);

        let result = v.validate(40001, 500.0, 0);
        assert!(result.is_ok());
        let r = result.expect("result");
        assert!(r.in_range);
        assert!(r.is_valid);
    }

    #[test]
    fn test_validate_below_min() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.min_raw = Some(100.0);
        v.add_rule(rule);

        let r = v.validate(40001, 50.0, 0).expect("result");
        assert!(!r.in_range);
        assert!(!r.is_valid);
    }

    #[test]
    fn test_validate_above_max() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.max_raw = Some(500.0);
        v.add_rule(rule);

        let r = v.validate(40001, 600.0, 0).expect("result");
        assert!(!r.in_range);
        assert!(!r.is_valid);
    }

    #[test]
    fn test_validate_no_range() {
        let mut v = make_validator();
        v.add_rule(make_rule(40001, "temp"));
        let r = v.validate(40001, 99999.0, 0).expect("result");
        assert!(r.in_range);
    }

    // ── Alarm Threshold Tests ────────────────────────────────────────────

    #[test]
    fn test_alarm_high_high() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.alarms.high_high = Some(100.0);
        v.add_rule(rule);

        let r = v.validate(40001, 105.0, 0).expect("result");
        assert!(r.alarms.contains(&AlarmLevel::HighHigh));
        assert!(!r.is_valid);
    }

    #[test]
    fn test_alarm_high() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.alarms.high = Some(80.0);
        v.add_rule(rule);

        let r = v.validate(40001, 85.0, 0).expect("result");
        assert!(r.alarms.contains(&AlarmLevel::High));
    }

    #[test]
    fn test_alarm_low() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.alarms.low = Some(10.0);
        v.add_rule(rule);

        let r = v.validate(40001, 5.0, 0).expect("result");
        assert!(r.alarms.contains(&AlarmLevel::Low));
    }

    #[test]
    fn test_alarm_low_low() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.alarms.low_low = Some(-10.0);
        v.add_rule(rule);

        let r = v.validate(40001, -15.0, 0).expect("result");
        assert!(r.alarms.contains(&AlarmLevel::LowLow));
    }

    #[test]
    fn test_alarm_high_high_suppresses_high() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.alarms.high = Some(80.0);
        rule.alarms.high_high = Some(100.0);
        v.add_rule(rule);

        let r = v.validate(40001, 110.0, 0).expect("result");
        assert!(r.alarms.contains(&AlarmLevel::HighHigh));
        assert!(!r.alarms.contains(&AlarmLevel::High));
    }

    #[test]
    fn test_no_alarms_in_normal_range() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.alarms.high = Some(80.0);
        rule.alarms.low = Some(10.0);
        v.add_rule(rule);

        let r = v.validate(40001, 50.0, 0).expect("result");
        assert!(r.alarms.is_empty());
    }

    // ── Rate-of-Change Tests ─────────────────────────────────────────────

    #[test]
    fn test_roc_not_exceeded() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.roc = Some(RocConfig {
            max_rate_per_sec: 10.0,
        });
        v.add_rule(rule);

        v.validate(40001, 50.0, 0).ok();
        let r = v.validate(40001, 55.0, 1000).expect("result");
        // 5 units / 1 sec = 5/s < 10/s
        assert!(!r.roc_exceeded);
    }

    #[test]
    fn test_roc_exceeded() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.roc = Some(RocConfig {
            max_rate_per_sec: 10.0,
        });
        v.add_rule(rule);

        v.validate(40001, 50.0, 0).ok();
        let r = v.validate(40001, 100.0, 1000).expect("result");
        // 50 units / 1 sec = 50/s > 10/s
        assert!(r.roc_exceeded);
    }

    #[test]
    fn test_roc_first_reading_not_exceeded() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.roc = Some(RocConfig {
            max_rate_per_sec: 1.0,
        });
        v.add_rule(rule);

        let r = v.validate(40001, 1000.0, 0).expect("result");
        assert!(!r.roc_exceeded);
    }

    #[test]
    fn test_roc_no_config() {
        let mut v = make_validator();
        v.add_rule(make_rule(40001, "temp"));
        v.validate(40001, 0.0, 0).ok();
        let r = v.validate(40001, 10000.0, 1000).expect("result");
        assert!(!r.roc_exceeded);
    }

    // ── Dead-Band Tests ──────────────────────────────────────────────────

    #[test]
    fn test_dead_band_absolute_passes() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.dead_band = Some(DeadBandConfig {
            absolute: Some(5.0),
            percentage: None,
        });
        v.add_rule(rule);

        v.validate(40001, 50.0, 0).ok();
        let r = v.validate(40001, 56.0, 1000).expect("result");
        // |56 - 50| = 6 >= 5
        assert!(r.dead_band_passed);
    }

    #[test]
    fn test_dead_band_absolute_filtered() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.dead_band = Some(DeadBandConfig {
            absolute: Some(5.0),
            percentage: None,
        });
        v.add_rule(rule);

        v.validate(40001, 50.0, 0).ok();
        let r = v.validate(40001, 52.0, 1000).expect("result");
        // |52 - 50| = 2 < 5
        assert!(!r.dead_band_passed);
    }

    #[test]
    fn test_dead_band_percentage_passes() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.dead_band = Some(DeadBandConfig {
            absolute: None,
            percentage: Some(10.0),
        });
        v.add_rule(rule);

        v.validate(40001, 100.0, 0).ok();
        let r = v.validate(40001, 115.0, 1000).expect("result");
        // |115 - 100| / 100 * 100 = 15% >= 10%
        assert!(r.dead_band_passed);
    }

    #[test]
    fn test_dead_band_percentage_filtered() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.dead_band = Some(DeadBandConfig {
            absolute: None,
            percentage: Some(10.0),
        });
        v.add_rule(rule);

        v.validate(40001, 100.0, 0).ok();
        let r = v.validate(40001, 105.0, 1000).expect("result");
        // 5% < 10%
        assert!(!r.dead_band_passed);
    }

    #[test]
    fn test_dead_band_first_reading_passes() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.dead_band = Some(DeadBandConfig {
            absolute: Some(100.0),
            percentage: None,
        });
        v.add_rule(rule);

        let r = v.validate(40001, 1.0, 0).expect("result");
        assert!(r.dead_band_passed);
    }

    #[test]
    fn test_dead_band_none_always_passes() {
        let mut v = make_validator();
        v.add_rule(make_rule(40001, "temp"));
        v.validate(40001, 50.0, 0).ok();
        let r = v.validate(40001, 50.001, 1000).expect("result");
        assert!(r.dead_band_passed);
    }

    // ── Scaling with Validation Tests ────────────────────────────────────

    #[test]
    fn test_scaling_applied_to_engineering_value() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.scaling = ScalingConfig {
            scale: 0.1,
            offset: -40.0,
        };
        v.add_rule(rule);

        let r = v.validate(40001, 500.0, 0).expect("result");
        // 500 * 0.1 + (-40) = 10.0
        assert!((r.engineering_value - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_alarm_applied_to_engineering_value() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.scaling = ScalingConfig {
            scale: 0.1,
            offset: 0.0,
        };
        rule.alarms.high = Some(50.0);
        v.add_rule(rule);

        // raw 600 => eng 60.0 >= 50.0 => High alarm
        let r = v.validate(40001, 600.0, 0).expect("result");
        assert!(r.alarms.contains(&AlarmLevel::High));
    }

    // ── validate_u16 Tests ───────────────────────────────────────────────

    #[test]
    fn test_validate_u16() {
        let mut v = make_validator();
        v.add_rule(make_rule(40001, "temp"));
        let r = v.validate_u16(40001, 500, 0).expect("result");
        assert!((r.raw_value - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validate_u16_no_rule() {
        let mut v = make_validator();
        assert!(v.validate_u16(40001, 500, 0).is_err());
    }

    // ── Batch Validation Tests ───────────────────────────────────────────

    #[test]
    fn test_validate_batch() {
        let mut v = make_validator();
        v.add_rule(make_rule(40001, "temp"));
        v.add_rule(make_rule(40002, "pressure"));
        let results = v.validate_batch(&[(40001, 50.0), (40002, 75.0)], 0);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
    }

    #[test]
    fn test_validate_batch_with_missing_rule() {
        let mut v = make_validator();
        v.add_rule(make_rule(40001, "temp"));
        let results = v.validate_batch(&[(40001, 50.0), (40099, 75.0)], 0);
        assert!(results[0].is_ok());
        assert!(results[1].is_err());
    }

    // ── Statistics Tests ─────────────────────────────────────────────────

    #[test]
    fn test_statistics_counters() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.max_raw = Some(100.0);
        rule.alarms.high = Some(80.0);
        v.add_rule(rule);

        v.validate(40001, 50.0, 0).ok(); // valid
        v.validate(40001, 200.0, 1000).ok(); // out of range + alarm
        assert_eq!(v.total_validated(), 2);
        assert!(v.total_failures() >= 1);
    }

    #[test]
    fn test_alarm_counter() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.alarms.high = Some(80.0);
        v.add_rule(rule);

        v.validate(40001, 90.0, 0).ok();
        assert!(v.total_alarms() >= 1);
    }

    // ── Error Display Tests ──────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e = ValidatorError::NoRule(40001);
        assert!(format!("{e}").contains("40001"));
        let e = ValidatorError::TypeMismatch("bad type".to_string());
        assert!(format!("{e}").contains("bad type"));
        let e = ValidatorError::InvalidConfig("missing field".to_string());
        assert!(format!("{e}").contains("missing field"));
    }

    // ── Default Trait Tests ──────────────────────────────────────────────

    #[test]
    fn test_default_validator() {
        let v = RegisterValidator::default();
        assert_eq!(v.rule_count(), 0);
    }

    #[test]
    fn test_default_scaling() {
        let s = ScalingConfig::default();
        assert!((s.scale - 1.0).abs() < f64::EPSILON);
        assert!((s.offset - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_default_alarms() {
        let a = AlarmThresholds::default();
        assert!(a.high_high.is_none());
        assert!(a.high.is_none());
        assert!(a.low.is_none());
        assert!(a.low_low.is_none());
    }

    // ── Edge Cases ───────────────────────────────────────────────────────

    #[test]
    fn test_validate_no_rule_error() {
        let mut v = make_validator();
        assert!(v.validate(40001, 50.0, 0).is_err());
    }

    #[test]
    fn test_roc_zero_time_delta() {
        let mut v = make_validator();
        let mut rule = make_rule(40001, "temp");
        rule.roc = Some(RocConfig {
            max_rate_per_sec: 1.0,
        });
        v.add_rule(rule);

        v.validate(40001, 50.0, 1000).ok();
        let r = v.validate(40001, 99999.0, 1000).expect("result");
        // dt = 0 => no roc check
        assert!(!r.roc_exceeded);
    }
}
