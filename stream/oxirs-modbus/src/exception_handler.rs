//! Modbus exception code processing and retry logic.
//!
//! Provides parsing of Modbus exception responses, classification of exception
//! codes (retryable vs. fatal), exponential-backoff retry planning, and a
//! rolling history of exceptions per device.

use std::collections::HashMap;
use std::fmt;

// ── Exception codes ───────────────────────────────────────────────────────────

/// Standard Modbus exception codes as defined in Modbus Application Protocol V1.1b3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ModbusExceptionCode {
    /// Function code received is not supported.
    IllegalFunction = 0x01,
    /// Data address is not allowed for the requested function.
    IllegalDataAddress = 0x02,
    /// Value in the data field is not an acceptable quantity.
    IllegalDataValue = 0x03,
    /// Unrecoverable error occurred while processing the request.
    ServerDeviceFailure = 0x04,
    /// Server has accepted the request and is processing it.
    Acknowledge = 0x05,
    /// Server is busy processing a previous request.
    ServerDeviceBusy = 0x06,
    /// Memory parity error detected during extended file access.
    MemoryParityError = 0x08,
    /// Gateway path is not available.
    GatewayPathUnavailable = 0x0A,
    /// Target device failed to respond (gateway relay timeout).
    GatewayTargetDeviceFailedToRespond = 0x0B,
}

impl ModbusExceptionCode {
    /// Parse a raw byte into a known exception code, returning `None` for unknown values.
    pub fn from_u8(code: u8) -> Option<Self> {
        match code {
            0x01 => Some(Self::IllegalFunction),
            0x02 => Some(Self::IllegalDataAddress),
            0x03 => Some(Self::IllegalDataValue),
            0x04 => Some(Self::ServerDeviceFailure),
            0x05 => Some(Self::Acknowledge),
            0x06 => Some(Self::ServerDeviceBusy),
            0x08 => Some(Self::MemoryParityError),
            0x0A => Some(Self::GatewayPathUnavailable),
            0x0B => Some(Self::GatewayTargetDeviceFailedToRespond),
            _ => None,
        }
    }

    /// Human-readable description of the exception.
    pub fn description(&self) -> &'static str {
        match self {
            Self::IllegalFunction => "Illegal Function",
            Self::IllegalDataAddress => "Illegal Data Address",
            Self::IllegalDataValue => "Illegal Data Value",
            Self::ServerDeviceFailure => "Server Device Failure",
            Self::Acknowledge => "Acknowledge (processing)",
            Self::ServerDeviceBusy => "Server Device Busy",
            Self::MemoryParityError => "Memory Parity Error",
            Self::GatewayPathUnavailable => "Gateway Path Unavailable",
            Self::GatewayTargetDeviceFailedToRespond => "Gateway Target Device Failed to Respond",
        }
    }

    /// Returns `true` if the operation should be retried after this exception.
    ///
    /// Retryable codes: `Acknowledge`, `ServerDeviceBusy`,
    /// `GatewayPathUnavailable`, `GatewayTargetDeviceFailedToRespond`.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Acknowledge
                | Self::ServerDeviceBusy
                | Self::GatewayPathUnavailable
                | Self::GatewayTargetDeviceFailedToRespond
        )
    }

    /// Returns `true` for non-retryable (fatal) exceptions.
    pub fn is_fatal(&self) -> bool {
        !self.is_retryable()
    }
}

// ── Exception record ──────────────────────────────────────────────────────────

/// A parsed Modbus exception together with context metadata.
#[derive(Debug, Clone)]
pub struct ModbusException {
    /// The function code from the original request (error bit cleared).
    pub function_code: u8,
    /// The decoded exception code.
    pub exception_code: ModbusExceptionCode,
    /// Modbus device / unit ID.
    pub device_id: u8,
    /// Unix-like millisecond timestamp (set by the caller; may be 0 in tests).
    pub timestamp_ms: u64,
}

// ── Retry configuration ───────────────────────────────────────────────────────

/// Configuration for exponential-backoff retry behaviour.
#[derive(Debug, Clone, Copy)]
pub struct RetryConfig {
    /// Maximum number of retry attempts before giving up.
    pub max_retries: u32,
    /// Delay in milliseconds before the first retry.
    pub initial_delay_ms: u64,
    /// Multiplicative factor applied to the delay on each successive retry.
    pub backoff_multiplier: f64,
    /// Upper bound on the computed delay in milliseconds.
    pub max_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig {
            max_retries: 3,
            initial_delay_ms: 100,
            backoff_multiplier: 2.0,
            max_delay_ms: 5_000,
        }
    }
}

// ── Retry plan ────────────────────────────────────────────────────────────────

/// Decision and timing produced by `ExceptionHandler::should_retry`.
#[derive(Debug, Clone)]
pub struct RetryPlan {
    /// Whether the operation should be retried.
    pub should_retry: bool,
    /// Which retry number this plan corresponds to (`0` = first attempt).
    pub retry_count: u32,
    /// How many milliseconds to wait before the next attempt.
    pub delay_ms: u64,
    /// Human-readable explanation of the decision.
    pub reason: String,
}

// ── Handler ───────────────────────────────────────────────────────────────────

/// Processes Modbus exceptions, plans retries, and maintains a rolling history.
pub struct ExceptionHandler {
    retry_config: RetryConfig,
    exception_history: Vec<ModbusException>,
    max_history: usize,
}

impl ExceptionHandler {
    /// Create a handler with default history limit (1 000 entries).
    pub fn new(config: RetryConfig) -> Self {
        Self::with_max_history(config, 1_000)
    }

    /// Create a handler with an explicit history limit.
    pub fn with_max_history(config: RetryConfig, max_history: usize) -> Self {
        Self {
            retry_config: config,
            exception_history: Vec::new(),
            max_history,
        }
    }

    /// Append an exception to the history, evicting the oldest entry when the
    /// limit is reached.
    pub fn record(&mut self, exc: ModbusException) {
        if self.exception_history.len() >= self.max_history {
            self.exception_history.remove(0);
        }
        self.exception_history.push(exc);
    }

    /// Decide whether the operation should be retried after `exc` on attempt
    /// number `attempt` (0-based: 0 = first call, i.e. before any retry).
    pub fn should_retry(&self, exc: &ModbusException, attempt: u32) -> RetryPlan {
        if !exc.exception_code.is_retryable() {
            return RetryPlan {
                should_retry: false,
                retry_count: attempt,
                delay_ms: 0,
                reason: format!(
                    "Exception {} is fatal and not retryable",
                    exc.exception_code.description()
                ),
            };
        }

        if attempt >= self.retry_config.max_retries {
            return RetryPlan {
                should_retry: false,
                retry_count: attempt,
                delay_ms: 0,
                reason: format!("Max retries ({}) reached", self.retry_config.max_retries),
            };
        }

        let delay_ms = self.compute_delay_ms(attempt);
        RetryPlan {
            should_retry: true,
            retry_count: attempt,
            delay_ms,
            reason: format!(
                "Retryable exception '{}'; retry {} of {}",
                exc.exception_code.description(),
                attempt + 1,
                self.retry_config.max_retries
            ),
        }
    }

    /// Compute the delay before retry number `attempt` (0-based).
    ///
    /// `delay = min(initial_delay * multiplier^attempt, max_delay)`
    pub fn compute_delay_ms(&self, attempt: u32) -> u64 {
        let factor = self.retry_config.backoff_multiplier.powi(attempt as i32);
        let delay = self.retry_config.initial_delay_ms as f64 * factor;
        delay.min(self.retry_config.max_delay_ms as f64) as u64
    }

    /// Immutable view of the full exception history.
    pub fn history(&self) -> &[ModbusException] {
        &self.exception_history
    }

    /// All historical exceptions for a given device ID.
    pub fn history_for_device(&self, device_id: u8) -> Vec<&ModbusException> {
        self.exception_history
            .iter()
            .filter(|e| e.device_id == device_id)
            .collect()
    }

    /// Count of each exception code seen in the history.
    pub fn exception_count_by_code(&self) -> HashMap<ModbusExceptionCode, usize> {
        let mut map: HashMap<ModbusExceptionCode, usize> = HashMap::new();
        for exc in &self.exception_history {
            *map.entry(exc.exception_code).or_insert(0) += 1;
        }
        map
    }

    /// The most recently recorded exception, or `None` if history is empty.
    pub fn most_recent(&self) -> Option<&ModbusException> {
        self.exception_history.last()
    }

    /// Remove all entries from the history.
    pub fn clear_history(&mut self) {
        self.exception_history.clear();
    }

    /// Parse a raw Modbus response into a `ModbusException`.
    ///
    /// A Modbus exception response has the format:
    ///   `[function_code | 0x80, exception_code, ...]`
    ///
    /// Returns `None` if the response is shorter than 2 bytes, if the high bit
    /// of the first byte is not set, or if the exception code is unrecognised.
    pub fn parse_exception_response(device_id: u8, response: &[u8]) -> Option<ModbusException> {
        if response.len() < 2 {
            return None;
        }

        let first_byte = response[0];
        // High bit must be set to indicate an exception.
        if first_byte & 0x80 == 0 {
            return None;
        }

        let function_code = first_byte & 0x7F;
        let exception_code = ModbusExceptionCode::from_u8(response[1])?;

        Some(ModbusException {
            function_code,
            exception_code,
            device_id,
            timestamp_ms: 0,
        })
    }
}

// ── Display for error ─────────────────────────────────────────────────────────

impl fmt::Display for ModbusExceptionCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (0x{:02X})", self.description(), *self as u8)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_handler() -> ExceptionHandler {
        ExceptionHandler::new(RetryConfig::default())
    }

    fn make_exc(code: ModbusExceptionCode) -> ModbusException {
        ModbusException {
            function_code: 0x03,
            exception_code: code,
            device_id: 1,
            timestamp_ms: 0,
        }
    }

    // ── from_u8 ───────────────────────────────────────────────────────────────

    #[test]
    fn test_from_u8_illegal_function() {
        assert_eq!(
            ModbusExceptionCode::from_u8(0x01),
            Some(ModbusExceptionCode::IllegalFunction)
        );
    }

    #[test]
    fn test_from_u8_illegal_data_address() {
        assert_eq!(
            ModbusExceptionCode::from_u8(0x02),
            Some(ModbusExceptionCode::IllegalDataAddress)
        );
    }

    #[test]
    fn test_from_u8_illegal_data_value() {
        assert_eq!(
            ModbusExceptionCode::from_u8(0x03),
            Some(ModbusExceptionCode::IllegalDataValue)
        );
    }

    #[test]
    fn test_from_u8_server_device_failure() {
        assert_eq!(
            ModbusExceptionCode::from_u8(0x04),
            Some(ModbusExceptionCode::ServerDeviceFailure)
        );
    }

    #[test]
    fn test_from_u8_acknowledge() {
        assert_eq!(
            ModbusExceptionCode::from_u8(0x05),
            Some(ModbusExceptionCode::Acknowledge)
        );
    }

    #[test]
    fn test_from_u8_server_device_busy() {
        assert_eq!(
            ModbusExceptionCode::from_u8(0x06),
            Some(ModbusExceptionCode::ServerDeviceBusy)
        );
    }

    #[test]
    fn test_from_u8_memory_parity_error() {
        assert_eq!(
            ModbusExceptionCode::from_u8(0x08),
            Some(ModbusExceptionCode::MemoryParityError)
        );
    }

    #[test]
    fn test_from_u8_gateway_path_unavailable() {
        assert_eq!(
            ModbusExceptionCode::from_u8(0x0A),
            Some(ModbusExceptionCode::GatewayPathUnavailable)
        );
    }

    #[test]
    fn test_from_u8_gateway_target_failed() {
        assert_eq!(
            ModbusExceptionCode::from_u8(0x0B),
            Some(ModbusExceptionCode::GatewayTargetDeviceFailedToRespond)
        );
    }

    #[test]
    fn test_from_u8_unknown_returns_none() {
        assert!(ModbusExceptionCode::from_u8(0x07).is_none());
        assert!(ModbusExceptionCode::from_u8(0x00).is_none());
        assert!(ModbusExceptionCode::from_u8(0xFF).is_none());
    }

    // ── is_retryable ──────────────────────────────────────────────────────────

    #[test]
    fn test_acknowledge_is_retryable() {
        assert!(ModbusExceptionCode::Acknowledge.is_retryable());
    }

    #[test]
    fn test_server_device_busy_is_retryable() {
        assert!(ModbusExceptionCode::ServerDeviceBusy.is_retryable());
    }

    #[test]
    fn test_gateway_path_unavailable_is_retryable() {
        assert!(ModbusExceptionCode::GatewayPathUnavailable.is_retryable());
    }

    #[test]
    fn test_gateway_target_device_failed_is_retryable() {
        assert!(ModbusExceptionCode::GatewayTargetDeviceFailedToRespond.is_retryable());
    }

    #[test]
    fn test_illegal_function_is_not_retryable() {
        assert!(!ModbusExceptionCode::IllegalFunction.is_retryable());
    }

    #[test]
    fn test_illegal_data_address_is_not_retryable() {
        assert!(!ModbusExceptionCode::IllegalDataAddress.is_retryable());
    }

    #[test]
    fn test_illegal_data_value_is_not_retryable() {
        assert!(!ModbusExceptionCode::IllegalDataValue.is_retryable());
    }

    #[test]
    fn test_server_device_failure_is_not_retryable() {
        assert!(!ModbusExceptionCode::ServerDeviceFailure.is_retryable());
    }

    #[test]
    fn test_memory_parity_error_is_not_retryable() {
        assert!(!ModbusExceptionCode::MemoryParityError.is_retryable());
    }

    // ── is_fatal ──────────────────────────────────────────────────────────────

    #[test]
    fn test_is_fatal_for_non_retryable_codes() {
        assert!(ModbusExceptionCode::IllegalFunction.is_fatal());
        assert!(ModbusExceptionCode::ServerDeviceFailure.is_fatal());
        assert!(ModbusExceptionCode::MemoryParityError.is_fatal());
    }

    #[test]
    fn test_is_not_fatal_for_retryable_codes() {
        assert!(!ModbusExceptionCode::Acknowledge.is_fatal());
        assert!(!ModbusExceptionCode::ServerDeviceBusy.is_fatal());
    }

    // ── description ───────────────────────────────────────────────────────────

    #[test]
    fn test_description_non_empty_for_all_codes() {
        let codes = [
            ModbusExceptionCode::IllegalFunction,
            ModbusExceptionCode::IllegalDataAddress,
            ModbusExceptionCode::IllegalDataValue,
            ModbusExceptionCode::ServerDeviceFailure,
            ModbusExceptionCode::Acknowledge,
            ModbusExceptionCode::ServerDeviceBusy,
            ModbusExceptionCode::MemoryParityError,
            ModbusExceptionCode::GatewayPathUnavailable,
            ModbusExceptionCode::GatewayTargetDeviceFailedToRespond,
        ];
        for code in codes {
            assert!(
                !code.description().is_empty(),
                "empty description for {code:?}"
            );
        }
    }

    // ── should_retry ─────────────────────────────────────────────────────────

    #[test]
    fn test_should_retry_retryable_within_max() {
        let h = default_handler();
        let exc = make_exc(ModbusExceptionCode::ServerDeviceBusy);
        let plan = h.should_retry(&exc, 0);
        assert!(plan.should_retry);
    }

    #[test]
    fn test_should_retry_non_retryable_always_false() {
        let h = default_handler();
        let exc = make_exc(ModbusExceptionCode::IllegalFunction);
        let plan = h.should_retry(&exc, 0);
        assert!(!plan.should_retry);
    }

    #[test]
    fn test_should_retry_at_max_retries_returns_false() {
        let h = default_handler(); // max_retries = 3
        let exc = make_exc(ModbusExceptionCode::ServerDeviceBusy);
        let plan = h.should_retry(&exc, 3);
        assert!(!plan.should_retry);
    }

    #[test]
    fn test_should_retry_before_max_retries_returns_true() {
        let h = default_handler();
        let exc = make_exc(ModbusExceptionCode::Acknowledge);
        for attempt in 0..3 {
            let plan = h.should_retry(&exc, attempt);
            assert!(plan.should_retry, "should retry at attempt {attempt}");
        }
    }

    #[test]
    fn test_should_retry_plan_contains_delay() {
        let h = default_handler();
        let exc = make_exc(ModbusExceptionCode::GatewayPathUnavailable);
        let plan = h.should_retry(&exc, 0);
        assert!(plan.delay_ms > 0);
    }

    #[test]
    fn test_should_retry_reason_non_empty() {
        let h = default_handler();
        let exc = make_exc(ModbusExceptionCode::ServerDeviceBusy);
        let plan = h.should_retry(&exc, 0);
        assert!(!plan.reason.is_empty());
    }

    // ── compute_delay_ms ─────────────────────────────────────────────────────

    #[test]
    fn test_compute_delay_exponential_growth() {
        let h = default_handler(); // initial=100, mult=2.0
        assert_eq!(h.compute_delay_ms(0), 100);
        assert_eq!(h.compute_delay_ms(1), 200);
        assert_eq!(h.compute_delay_ms(2), 400);
        assert_eq!(h.compute_delay_ms(3), 800);
    }

    #[test]
    fn test_compute_delay_capped_at_max() {
        let cfg = RetryConfig {
            max_retries: 10,
            initial_delay_ms: 100,
            backoff_multiplier: 2.0,
            max_delay_ms: 500,
        };
        let h = ExceptionHandler::new(cfg);
        // 100 * 2^4 = 1600 > 500 → should clamp.
        assert_eq!(h.compute_delay_ms(4), 500);
        assert_eq!(h.compute_delay_ms(10), 500);
    }

    #[test]
    fn test_compute_delay_attempt_zero_is_initial() {
        let h = default_handler();
        assert_eq!(h.compute_delay_ms(0), 100);
    }

    // ── record / history ─────────────────────────────────────────────────────

    #[test]
    fn test_record_adds_to_history() {
        let mut h = default_handler();
        h.record(make_exc(ModbusExceptionCode::Acknowledge));
        assert_eq!(h.history().len(), 1);
    }

    #[test]
    fn test_record_multiple_entries() {
        let mut h = default_handler();
        for _ in 0..5 {
            h.record(make_exc(ModbusExceptionCode::ServerDeviceBusy));
        }
        assert_eq!(h.history().len(), 5);
    }

    #[test]
    fn test_history_evicts_oldest_when_full() {
        let mut h = ExceptionHandler::with_max_history(RetryConfig::default(), 3);
        for i in 0..4u8 {
            let exc = ModbusException {
                function_code: i,
                exception_code: ModbusExceptionCode::IllegalFunction,
                device_id: i,
                timestamp_ms: i as u64,
            };
            h.record(exc);
        }
        assert_eq!(h.history().len(), 3);
        // The oldest entry (device_id=0) should have been evicted.
        assert_eq!(h.history()[0].device_id, 1);
    }

    // ── history_for_device ───────────────────────────────────────────────────

    #[test]
    fn test_history_for_device_filters_correctly() {
        let mut h = default_handler();
        for device in [1u8, 2, 1, 3, 1] {
            let exc = ModbusException {
                function_code: 0x03,
                exception_code: ModbusExceptionCode::ServerDeviceFailure,
                device_id: device,
                timestamp_ms: 0,
            };
            h.record(exc);
        }
        assert_eq!(h.history_for_device(1).len(), 3);
        assert_eq!(h.history_for_device(2).len(), 1);
        assert_eq!(h.history_for_device(99).len(), 0);
    }

    // ── exception_count_by_code ───────────────────────────────────────────────

    #[test]
    fn test_exception_count_by_code() {
        let mut h = default_handler();
        h.record(make_exc(ModbusExceptionCode::IllegalFunction));
        h.record(make_exc(ModbusExceptionCode::IllegalFunction));
        h.record(make_exc(ModbusExceptionCode::ServerDeviceBusy));
        let counts = h.exception_count_by_code();
        assert_eq!(counts[&ModbusExceptionCode::IllegalFunction], 2);
        assert_eq!(counts[&ModbusExceptionCode::ServerDeviceBusy], 1);
    }

    #[test]
    fn test_exception_count_empty_history() {
        let h = default_handler();
        assert!(h.exception_count_by_code().is_empty());
    }

    // ── most_recent ───────────────────────────────────────────────────────────

    #[test]
    fn test_most_recent_returns_last_recorded() {
        let mut h = default_handler();
        h.record(make_exc(ModbusExceptionCode::Acknowledge));
        let last = ModbusException {
            function_code: 0x10,
            exception_code: ModbusExceptionCode::ServerDeviceBusy,
            device_id: 5,
            timestamp_ms: 999,
        };
        h.record(last.clone());
        let recent = h.most_recent().expect("should have most recent");
        assert_eq!(recent.exception_code, ModbusExceptionCode::ServerDeviceBusy);
        assert_eq!(recent.device_id, 5);
    }

    #[test]
    fn test_most_recent_empty_returns_none() {
        let h = default_handler();
        assert!(h.most_recent().is_none());
    }

    // ── clear_history ─────────────────────────────────────────────────────────

    #[test]
    fn test_clear_history_empties_history() {
        let mut h = default_handler();
        h.record(make_exc(ModbusExceptionCode::Acknowledge));
        h.record(make_exc(ModbusExceptionCode::Acknowledge));
        h.clear_history();
        assert!(h.history().is_empty());
    }

    // ── parse_exception_response ──────────────────────────────────────────────

    #[test]
    fn test_parse_exception_response_valid() {
        // FC 03 exception with code 0x02 (IllegalDataAddress)
        let response = [0x83u8, 0x02]; // 0x83 = 0x80 | 0x03
        let exc = ExceptionHandler::parse_exception_response(1, &response).expect("should parse");
        assert_eq!(exc.function_code, 0x03);
        assert_eq!(exc.exception_code, ModbusExceptionCode::IllegalDataAddress);
        assert_eq!(exc.device_id, 1);
    }

    #[test]
    fn test_parse_exception_response_non_exception_byte() {
        // High bit not set → normal response, not an exception.
        let response = [0x03u8, 0x02];
        assert!(ExceptionHandler::parse_exception_response(1, &response).is_none());
    }

    #[test]
    fn test_parse_exception_response_too_short() {
        let response = [0x83u8];
        assert!(ExceptionHandler::parse_exception_response(1, &response).is_none());
    }

    #[test]
    fn test_parse_exception_response_empty() {
        assert!(ExceptionHandler::parse_exception_response(1, &[]).is_none());
    }

    #[test]
    fn test_parse_exception_response_unknown_code_returns_none() {
        let response = [0x83u8, 0x07]; // 0x07 is not a known exception code
        assert!(ExceptionHandler::parse_exception_response(1, &response).is_none());
    }

    #[test]
    fn test_parse_exception_response_all_valid_codes() {
        let code_pairs: &[(u8, ModbusExceptionCode)] = &[
            (0x01, ModbusExceptionCode::IllegalFunction),
            (0x02, ModbusExceptionCode::IllegalDataAddress),
            (0x03, ModbusExceptionCode::IllegalDataValue),
            (0x04, ModbusExceptionCode::ServerDeviceFailure),
            (0x05, ModbusExceptionCode::Acknowledge),
            (0x06, ModbusExceptionCode::ServerDeviceBusy),
            (0x08, ModbusExceptionCode::MemoryParityError),
            (0x0A, ModbusExceptionCode::GatewayPathUnavailable),
            (
                0x0B,
                ModbusExceptionCode::GatewayTargetDeviceFailedToRespond,
            ),
        ];
        for (raw_code, expected) in code_pairs {
            let response = [0x80 | 0x01, *raw_code];
            let exc = ExceptionHandler::parse_exception_response(42, &response)
                .unwrap_or_else(|| panic!("parse failed for code 0x{raw_code:02X}"));
            assert_eq!(&exc.exception_code, expected);
        }
    }

    // ── display ───────────────────────────────────────────────────────────────

    #[test]
    fn test_display_includes_hex_code() {
        let s = format!("{}", ModbusExceptionCode::Acknowledge);
        assert!(s.contains("0x05"));
    }

    // ── retry_plan fields ─────────────────────────────────────────────────────

    #[test]
    fn test_retry_plan_retry_count_matches_attempt() {
        let h = default_handler();
        let exc = make_exc(ModbusExceptionCode::ServerDeviceBusy);
        let plan = h.should_retry(&exc, 2);
        assert_eq!(plan.retry_count, 2);
    }

    // ── edge: custom config ───────────────────────────────────────────────────

    #[test]
    fn test_custom_config_one_retry() {
        let cfg = RetryConfig {
            max_retries: 1,
            initial_delay_ms: 50,
            backoff_multiplier: 3.0,
            max_delay_ms: 10_000,
        };
        let h = ExceptionHandler::new(cfg);
        let exc = make_exc(ModbusExceptionCode::GatewayPathUnavailable);
        assert!(h.should_retry(&exc, 0).should_retry);
        assert!(!h.should_retry(&exc, 1).should_retry);
    }

    #[test]
    fn test_compute_delay_multiplier_three() {
        let cfg = RetryConfig {
            max_retries: 10,
            initial_delay_ms: 10,
            backoff_multiplier: 3.0,
            max_delay_ms: 100_000,
        };
        let h = ExceptionHandler::new(cfg);
        assert_eq!(h.compute_delay_ms(0), 10);
        assert_eq!(h.compute_delay_ms(1), 30);
        assert_eq!(h.compute_delay_ms(2), 90);
        assert_eq!(h.compute_delay_ms(3), 270);
    }
}
