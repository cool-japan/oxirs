//! # Frame Validator
//!
//! CAN frame integrity and DLC (Data Length Code) validation.
//!
//! Validates CAN 2.0 frames according to the following rules:
//! - Standard frames: 11-bit ID ≤ 0x7FF
//! - Extended frames: 29-bit ID ≤ 0x1FFFFFFF
//! - DLC must be in 0–8
//! - `data.len()` must equal `dlc`
//! - RTR (Remote Transmission Request) frames must not carry data bytes

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

/// Maximum 11-bit standard CAN identifier.
pub const STANDARD_ID_MAX: u32 = 0x7FF;

/// Maximum 29-bit extended CAN identifier.
pub const EXTENDED_ID_MAX: u32 = 0x1FFF_FFFF;

/// Maximum DLC value for CAN 2.0 (8 bytes).
pub const MAX_DLC: u8 = 8;

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// A CAN 2.0 frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanFrame {
    /// Frame identifier (11-bit for standard, 29-bit for extended).
    pub id: u32,
    /// Data Length Code — the number of data bytes the frame declares it contains.
    pub dlc: u8,
    /// Actual data payload.
    pub data: Vec<u8>,
    /// `true` for 29-bit extended frame format.
    pub is_extended: bool,
    /// `true` for Remote Transmission Request frames.
    pub is_rtr: bool,
}

impl CanFrame {
    /// Create a standard data frame.
    pub fn standard(id: u32, data: Vec<u8>) -> Self {
        let dlc = data.len() as u8;
        Self {
            id,
            dlc,
            data,
            is_extended: false,
            is_rtr: false,
        }
    }

    /// Create an extended data frame.
    pub fn extended(id: u32, data: Vec<u8>) -> Self {
        let dlc = data.len() as u8;
        Self {
            id,
            dlc,
            data,
            is_extended: true,
            is_rtr: false,
        }
    }

    /// Create a standard RTR frame.
    pub fn rtr_standard(id: u32, dlc: u8) -> Self {
        Self {
            id,
            dlc,
            data: vec![],
            is_extended: false,
            is_rtr: true,
        }
    }

    /// Create an extended RTR frame.
    pub fn rtr_extended(id: u32, dlc: u8) -> Self {
        Self {
            id,
            dlc,
            data: vec![],
            is_extended: true,
            is_rtr: true,
        }
    }
}

/// A particular integrity violation detected in a frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameViolation {
    /// The declared DLC does not match the actual number of data bytes.
    DlcMismatch {
        /// Declared DLC.
        dlc: u8,
        /// Actual `data.len()`.
        actual_len: u8,
    },
    /// The DLC value exceeds 8, which is invalid in CAN 2.0.
    InvalidDlc(u8),
    /// An extended-format ID exceeds 29 bits (> 0x1FFFFFFF).
    ExtendedIdTooLarge(u32),
    /// A standard-format ID exceeds 11 bits (> 0x7FF).
    StandardIdTooLarge(u32),
    /// The data payload exceeds 8 bytes regardless of DLC.
    DataLengthExceeded(u8),
    /// An RTR frame carries non-empty data.
    RtrWithData,
    /// Reserved for implementation-specific invalid frame ID conditions.
    InvalidFrameId,
}

impl std::fmt::Display for FrameViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameViolation::DlcMismatch { dlc, actual_len } => {
                write!(f, "DLC mismatch: declared {}, actual {}", dlc, actual_len)
            }
            FrameViolation::InvalidDlc(dlc) => {
                write!(f, "invalid DLC: {} (max {})", dlc, MAX_DLC)
            }
            FrameViolation::ExtendedIdTooLarge(id) => {
                write!(f, "extended ID 0x{:08X} exceeds 29-bit limit", id)
            }
            FrameViolation::StandardIdTooLarge(id) => {
                write!(f, "standard ID 0x{:03X} exceeds 11-bit limit", id)
            }
            FrameViolation::DataLengthExceeded(len) => {
                write!(f, "data length {} exceeds CAN 2.0 maximum of 8", len)
            }
            FrameViolation::RtrWithData => {
                write!(f, "RTR frame carries non-empty data")
            }
            FrameViolation::InvalidFrameId => {
                write!(f, "invalid frame ID")
            }
        }
    }
}

/// The outcome of validating a single frame.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// The frame's identifier.
    pub frame_id: u32,
    /// `true` when no violations were detected.
    pub valid: bool,
    /// Zero or more violations found.
    pub violations: Vec<FrameViolation>,
}

impl ValidationResult {
    fn new_valid(frame_id: u32) -> Self {
        Self {
            frame_id,
            valid: true,
            violations: vec![],
        }
    }

    fn new_invalid(frame_id: u32, violations: Vec<FrameViolation>) -> Self {
        Self {
            frame_id,
            valid: false,
            violations,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// FrameValidator
// ────────────────────────────────────────────────────────────────────────────

/// Stateless CAN 2.0 frame integrity validator.
///
/// # Example
/// ```rust
/// use oxirs_canbus::frame_validator::{CanFrame, FrameValidator};
///
/// let frame = CanFrame::standard(0x123, vec![0xDE, 0xAD]);
/// let result = FrameValidator::validate(&frame);
/// assert!(result.valid);
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct FrameValidator;

impl FrameValidator {
    /// Create a new validator (stateless; provided for conventional constructor pattern).
    pub fn new() -> Self {
        Self
    }

    /// Validate a single CAN frame and return a [`ValidationResult`].
    ///
    /// The following checks are performed in order; all applicable violations
    /// are collected (the validator does not stop at the first error):
    ///
    /// 1. ID range (standard ≤ 0x7FF, extended ≤ 0x1FFFFFFF).
    /// 2. DLC ≤ 8.
    /// 3. `data.len()` ≤ 8.
    /// 4. `data.len()` == DLC (only when both are individually valid).
    /// 5. RTR frame must have empty data.
    pub fn validate(frame: &CanFrame) -> ValidationResult {
        let mut violations: Vec<FrameViolation> = Vec::new();

        // Check 1: ID range
        if frame.is_extended {
            if frame.id > EXTENDED_ID_MAX {
                violations.push(FrameViolation::ExtendedIdTooLarge(frame.id));
            }
        } else if frame.id > STANDARD_ID_MAX {
            violations.push(FrameViolation::StandardIdTooLarge(frame.id));
        }

        // Check 2: DLC validity
        let dlc_valid = frame.dlc <= MAX_DLC;
        if !dlc_valid {
            violations.push(FrameViolation::InvalidDlc(frame.dlc));
        }

        // Check 3: actual data length
        let actual_len = frame.data.len();
        let data_len_valid = actual_len <= MAX_DLC as usize;
        if !data_len_valid {
            violations.push(FrameViolation::DataLengthExceeded(actual_len as u8));
        }

        // Check 4: DLC matches actual data length (only when both are individually legal,
        // and only for data frames — RTR frames declare requested length in DLC but
        // carry no data payload).
        if !frame.is_rtr && dlc_valid && data_len_valid && actual_len != frame.dlc as usize {
            violations.push(FrameViolation::DlcMismatch {
                dlc: frame.dlc,
                actual_len: actual_len as u8,
            });
        }

        // Check 5: RTR must not carry data
        if frame.is_rtr && !frame.data.is_empty() {
            violations.push(FrameViolation::RtrWithData);
        }

        if violations.is_empty() {
            ValidationResult::new_valid(frame.id)
        } else {
            ValidationResult::new_invalid(frame.id, violations)
        }
    }

    /// Validate a slice of frames and return one result per frame.
    pub fn validate_batch(frames: &[CanFrame]) -> Vec<ValidationResult> {
        frames.iter().map(Self::validate).collect()
    }

    /// Count valid results in a batch.
    pub fn valid_count(results: &[ValidationResult]) -> usize {
        results.iter().filter(|r| r.valid).count()
    }

    /// Count invalid results in a batch.
    pub fn invalid_count(results: &[ValidationResult]) -> usize {
        results.iter().filter(|r| !r.valid).count()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Valid standard frames ─────────────────────────────────────────────

    #[test]
    fn test_valid_standard_frame_zero_id() {
        let frame = CanFrame::standard(0x000, vec![]);
        let r = FrameValidator::validate(&frame);
        assert!(r.valid);
        assert!(r.violations.is_empty());
    }

    #[test]
    fn test_valid_standard_frame_max_id() {
        let frame = CanFrame::standard(STANDARD_ID_MAX, vec![1, 2, 3]);
        let r = FrameValidator::validate(&frame);
        assert!(r.valid);
    }

    #[test]
    fn test_valid_standard_frame_8_bytes() {
        let frame = CanFrame::standard(0x100, vec![0u8; 8]);
        let r = FrameValidator::validate(&frame);
        assert!(r.valid);
    }

    #[test]
    fn test_valid_standard_frame_empty_data() {
        let frame = CanFrame::standard(0x123, vec![]);
        let r = FrameValidator::validate(&frame);
        assert!(r.valid);
        assert_eq!(r.frame_id, 0x123);
    }

    // ── Valid extended frames ─────────────────────────────────────────────

    #[test]
    fn test_valid_extended_frame_zero_id() {
        let frame = CanFrame::extended(0x00000000, vec![]);
        let r = FrameValidator::validate(&frame);
        assert!(r.valid);
    }

    #[test]
    fn test_valid_extended_frame_max_id() {
        let frame = CanFrame::extended(EXTENDED_ID_MAX, vec![0xAA, 0xBB]);
        let r = FrameValidator::validate(&frame);
        assert!(r.valid);
    }

    #[test]
    fn test_valid_extended_frame_8_bytes() {
        let frame = CanFrame::extended(0x12345678, vec![0u8; 8]);
        let r = FrameValidator::validate(&frame);
        assert!(r.valid);
    }

    // ── Standard ID too large ─────────────────────────────────────────────

    #[test]
    fn test_standard_id_just_above_limit() {
        let frame = CanFrame::standard(0x800, vec![]);
        let r = FrameValidator::validate(&frame);
        assert!(!r.valid);
        assert!(r
            .violations
            .iter()
            .any(|v| matches!(v, FrameViolation::StandardIdTooLarge(0x800))));
    }

    #[test]
    fn test_standard_id_far_above_limit() {
        let frame = CanFrame::standard(0xFFFF, vec![]);
        let r = FrameValidator::validate(&frame);
        assert!(!r.valid);
        assert!(r
            .violations
            .iter()
            .any(|v| matches!(v, FrameViolation::StandardIdTooLarge(_))));
    }

    // ── Extended ID too large ─────────────────────────────────────────────

    #[test]
    fn test_extended_id_just_above_limit() {
        let id = EXTENDED_ID_MAX + 1;
        let frame = CanFrame::extended(id, vec![]);
        let r = FrameValidator::validate(&frame);
        assert!(!r.valid);
        assert!(r
            .violations
            .iter()
            .any(|v| matches!(v, FrameViolation::ExtendedIdTooLarge(_))));
    }

    #[test]
    fn test_extended_id_max_u32() {
        let frame = CanFrame::extended(u32::MAX, vec![]);
        let r = FrameValidator::validate(&frame);
        assert!(!r.valid);
    }

    // ── DLC mismatch ──────────────────────────────────────────────────────

    #[test]
    fn test_dlc_mismatch_too_high() {
        let mut frame = CanFrame::standard(0x100, vec![1, 2]);
        frame.dlc = 5; // declared 5, actual 2
        let r = FrameValidator::validate(&frame);
        assert!(!r.valid);
        assert!(r.violations.iter().any(|v| matches!(
            v,
            FrameViolation::DlcMismatch {
                dlc: 5,
                actual_len: 2
            }
        )));
    }

    #[test]
    fn test_dlc_mismatch_too_low() {
        let mut frame = CanFrame::standard(0x100, vec![1, 2, 3, 4]);
        frame.dlc = 1; // declared 1, actual 4
        let r = FrameValidator::validate(&frame);
        assert!(!r.valid);
        assert!(r.violations.iter().any(|v| {
            matches!(
                v,
                FrameViolation::DlcMismatch {
                    dlc: 1,
                    actual_len: 4
                }
            )
        }));
    }

    // ── Invalid DLC ───────────────────────────────────────────────────────

    #[test]
    fn test_invalid_dlc_9() {
        let frame = CanFrame {
            id: 0x100,
            dlc: 9,
            data: vec![0u8; 9],
            is_extended: false,
            is_rtr: false,
        };
        let r = FrameValidator::validate(&frame);
        assert!(!r.valid);
        assert!(r
            .violations
            .iter()
            .any(|v| matches!(v, FrameViolation::InvalidDlc(9))));
    }

    #[test]
    fn test_invalid_dlc_255() {
        let frame = CanFrame {
            id: 0x100,
            dlc: 255,
            data: vec![],
            is_extended: false,
            is_rtr: false,
        };
        let r = FrameValidator::validate(&frame);
        assert!(!r.valid);
        assert!(r
            .violations
            .iter()
            .any(|v| matches!(v, FrameViolation::InvalidDlc(255))));
    }

    // ── Data length exceeded ──────────────────────────────────────────────

    #[test]
    fn test_data_length_exceeded_9_bytes() {
        let frame = CanFrame {
            id: 0x100,
            dlc: 8,
            data: vec![0u8; 9],
            is_extended: false,
            is_rtr: false,
        };
        let r = FrameValidator::validate(&frame);
        assert!(!r.valid);
        assert!(r
            .violations
            .iter()
            .any(|v| matches!(v, FrameViolation::DataLengthExceeded(9))));
    }

    // ── RTR with data ─────────────────────────────────────────────────────

    #[test]
    fn test_rtr_with_data_is_invalid() {
        let frame = CanFrame {
            id: 0x100,
            dlc: 1,
            data: vec![0xAA],
            is_extended: false,
            is_rtr: true,
        };
        let r = FrameValidator::validate(&frame);
        assert!(!r.valid);
        assert!(r.violations.contains(&FrameViolation::RtrWithData));
    }

    #[test]
    fn test_rtr_without_data_is_valid() {
        let frame = CanFrame::rtr_standard(0x200, 4);
        let r = FrameValidator::validate(&frame);
        assert!(r.valid, "RTR frame with empty data should be valid");
    }

    #[test]
    fn test_extended_rtr_without_data_is_valid() {
        let frame = CanFrame::rtr_extended(0x1234ABCD, 0);
        let r = FrameValidator::validate(&frame);
        assert!(r.valid);
    }

    // ── DLC 0 is valid ────────────────────────────────────────────────────

    #[test]
    fn test_dlc_zero_is_valid() {
        let frame = CanFrame::standard(0x111, vec![]);
        let r = FrameValidator::validate(&frame);
        assert!(r.valid);
    }

    // ── validate_batch ────────────────────────────────────────────────────

    #[test]
    fn test_validate_batch_all_valid() {
        let frames = vec![
            CanFrame::standard(0x100, vec![1, 2]),
            CanFrame::extended(0x12345, vec![0xAA]),
            CanFrame::standard(0x7FF, vec![]),
        ];
        let results = FrameValidator::validate_batch(&frames);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.valid));
    }

    #[test]
    fn test_validate_batch_mix() {
        let frames = vec![
            CanFrame::standard(0x100, vec![1]),      // valid
            CanFrame::standard(0xFFF, vec![]),       // invalid: ID too large
            CanFrame::extended(0x1FFF_FFFF, vec![]), // valid
        ];
        let results = FrameValidator::validate_batch(&frames);
        assert_eq!(FrameValidator::valid_count(&results), 2);
        assert_eq!(FrameValidator::invalid_count(&results), 1);
    }

    #[test]
    fn test_validate_batch_empty_slice() {
        let results = FrameValidator::validate_batch(&[]);
        assert!(results.is_empty());
    }

    // ── valid_count / invalid_count ───────────────────────────────────────

    #[test]
    fn test_valid_count_zero() {
        let r = ValidationResult::new_invalid(0, vec![FrameViolation::InvalidFrameId]);
        assert_eq!(FrameValidator::valid_count(&[r]), 0);
    }

    #[test]
    fn test_invalid_count_zero() {
        let r = ValidationResult::new_valid(0x100);
        assert_eq!(FrameValidator::invalid_count(&[r]), 0);
    }

    #[test]
    fn test_counts_match_total() {
        let frames: Vec<CanFrame> = (0..10)
            .map(|i| {
                if i % 2 == 0 {
                    CanFrame::standard(0x100, vec![i as u8]) // valid
                } else {
                    CanFrame::standard(0xFFFF, vec![]) // invalid: ID too large
                }
            })
            .collect();
        let results = FrameValidator::validate_batch(&frames);
        assert_eq!(
            FrameValidator::valid_count(&results) + FrameValidator::invalid_count(&results),
            results.len()
        );
    }

    // ── Multiple violations ───────────────────────────────────────────────

    #[test]
    fn test_multiple_violations_collected() {
        // Standard ID too large AND data > 8 bytes AND RTR with data
        let frame = CanFrame {
            id: 0xFFFF,         // standard ID too large
            dlc: 9,             // invalid DLC
            data: vec![0u8; 9], // too long AND rtr with data
            is_extended: false,
            is_rtr: true,
        };
        let r = FrameValidator::validate(&frame);
        assert!(!r.valid);
        assert!(r.violations.len() >= 2);
    }

    // ── frame_id reflected in result ──────────────────────────────────────

    #[test]
    fn test_frame_id_reflected_in_valid_result() {
        let frame = CanFrame::standard(0x5A5, vec![0x11]);
        let r = FrameValidator::validate(&frame);
        assert_eq!(r.frame_id, 0x5A5);
    }

    #[test]
    fn test_frame_id_reflected_in_invalid_result() {
        let frame = CanFrame::standard(0xBEEF, vec![]);
        let r = FrameValidator::validate(&frame);
        assert_eq!(r.frame_id, 0xBEEF);
        assert!(!r.valid);
    }

    // ── FrameViolation Display ────────────────────────────────────────────

    #[test]
    fn test_violation_display_dlc_mismatch() {
        let v = FrameViolation::DlcMismatch {
            dlc: 4,
            actual_len: 2,
        };
        let s = v.to_string();
        assert!(s.contains("4") && s.contains("2"));
    }

    #[test]
    fn test_violation_display_rtr_with_data() {
        let v = FrameViolation::RtrWithData;
        assert!(
            v.to_string().contains("RTR")
                || v.to_string().contains("rtr")
                || v.to_string().contains("data")
        );
    }

    // ── CanFrame constructors ─────────────────────────────────────────────

    #[test]
    fn test_can_frame_standard_sets_dlc() {
        let frame = CanFrame::standard(0x100, vec![1, 2, 3]);
        assert_eq!(frame.dlc, 3);
        assert!(!frame.is_extended);
        assert!(!frame.is_rtr);
    }

    #[test]
    fn test_can_frame_extended_sets_dlc() {
        let frame = CanFrame::extended(0x1FFFF, vec![0xAA, 0xBB]);
        assert_eq!(frame.dlc, 2);
        assert!(frame.is_extended);
    }

    #[test]
    fn test_can_frame_rtr_standard_empty_data() {
        let frame = CanFrame::rtr_standard(0x100, 4);
        assert!(frame.is_rtr);
        assert!(frame.data.is_empty());
        assert_eq!(frame.dlc, 4);
    }
}
