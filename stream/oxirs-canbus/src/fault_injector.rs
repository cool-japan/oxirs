//! CAN bus fault injection for testing.
//!
//! `FaultInjector` applies configurable fault rules to CAN frames using a
//! deterministic LCG random number generator so that tests are reproducible.

use std::collections::HashMap;

/// The type of fault to inject.
#[derive(Debug, Clone, PartialEq)]
pub enum FaultType {
    /// Flip a single bit in the data payload.
    BitFlip { bit_position: u8 },
    /// Truncate the payload to a shorter length.
    Truncate { new_length: u8 },
    /// Add a simulated transmission delay (metadata only).
    Delay { extra_ms: u64 },
    /// Drop the frame entirely (inject() returns empty vec).
    Drop,
    /// Return the frame twice.
    Duplicate,
    /// Replace the CAN frame ID.
    CorruptId { new_id: u32 },
    /// Overwrite a single byte in the data payload.
    DataCorrupt { byte_offset: u8, new_value: u8 },
}

/// A rule that triggers a fault on matching CAN frames.
#[derive(Debug, Clone)]
pub struct FaultRule {
    /// Unique rule identifier.
    pub id: u32,
    /// Mask applied to `frame_id & frame_id_mask == frame_id_mask` → match all.
    pub frame_id_mask: u32,
    /// The fault to apply when this rule matches.
    pub fault: FaultType,
    /// Probability [0.0, 1.0] that the fault is applied on a given match.
    pub probability: f64,
    /// Maximum number of times this rule may fire; `None` = unlimited.
    pub max_applications: Option<usize>,
}

impl FaultRule {
    /// Create a new rule that always fires (`probability = 1.0`, unlimited).
    pub fn always(id: u32, frame_id_mask: u32, fault: FaultType) -> Self {
        Self {
            id,
            frame_id_mask,
            fault,
            probability: 1.0,
            max_applications: None,
        }
    }
}

/// A frame after (possible) fault injection.
#[derive(Debug, Clone, PartialEq)]
pub struct InjectedFrame {
    pub original_id: u32,
    pub original_data: Vec<u8>,
    pub modified_id: u32,
    pub modified_data: Vec<u8>,
    pub dropped: bool,
    pub duplicated: bool,
    /// Extra delay in ms introduced by a `Delay` fault.
    pub extra_delay_ms: u64,
}

impl InjectedFrame {
    fn passthrough(id: u32, data: &[u8]) -> Self {
        Self {
            original_id: id,
            original_data: data.to_vec(),
            modified_id: id,
            modified_data: data.to_vec(),
            dropped: false,
            duplicated: false,
            extra_delay_ms: 0,
        }
    }
}

/// CAN bus fault injector.
pub struct FaultInjector {
    rules: Vec<FaultRule>,
    application_counts: HashMap<u32, usize>,
    seed: u64,
}

impl FaultInjector {
    /// Create a new injector with the given LCG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rules: Vec::new(),
            application_counts: HashMap::new(),
            seed,
        }
    }

    /// Add a fault rule. Rules are evaluated in insertion order; the first
    /// matching rule whose probability fires is applied.
    pub fn add_rule(&mut self, rule: FaultRule) {
        self.rules.push(rule);
    }

    /// Inject faults into a CAN frame.
    ///
    /// Returns:
    /// - `vec![]` if the frame is dropped.
    /// - `vec![modified]` for a single (possibly mutated) frame.
    /// - `vec![frame, frame]` for duplicated frames.
    pub fn inject(&mut self, frame_id: u32, data: &[u8]) -> Vec<InjectedFrame> {
        for rule in self.rules.iter() {
            // Check mask: all bits set in mask must be set in frame_id
            if frame_id & rule.frame_id_mask != rule.frame_id_mask {
                continue;
            }
            // Check max_applications
            let count = self.application_counts.get(&rule.id).copied().unwrap_or(0);
            if let Some(max) = rule.max_applications {
                if count >= max {
                    continue;
                }
            }
            // Probabilistic gate
            let p = self.lcg_next();
            if p >= rule.probability {
                continue;
            }

            // Fire the rule
            *self.application_counts.entry(rule.id).or_insert(0) += 1;

            let fault = rule.fault.clone();
            return Self::apply_fault(frame_id, data, fault);
        }

        // No rule fired — pass through
        vec![InjectedFrame::passthrough(frame_id, data)]
    }

    /// Number of rules registered.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Total number of times any rule has fired across all calls.
    pub fn total_applications(&self) -> usize {
        self.application_counts.values().sum()
    }

    /// Reset all per-rule application counts (probabilities are preserved).
    pub fn reset_counts(&mut self) {
        self.application_counts.clear();
    }

    /// Simple LCG: returns a float in [0.0, 1.0).
    fn lcg_next(&mut self) -> f64 {
        // Numerical Recipes LCG parameters
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Use upper 32 bits to get a uniform float
        let upper = (self.seed >> 32) as u32;
        upper as f64 / (u32::MAX as f64 + 1.0)
    }

    fn apply_fault(frame_id: u32, data: &[u8], fault: FaultType) -> Vec<InjectedFrame> {
        let mut frame = InjectedFrame::passthrough(frame_id, data);
        match fault {
            FaultType::Drop => {
                frame.dropped = true;
                return vec![];
            }
            FaultType::Duplicate => {
                frame.duplicated = true;
                return vec![frame.clone(), frame];
            }
            FaultType::BitFlip { bit_position } => {
                if !frame.modified_data.is_empty() {
                    let byte_idx = (bit_position / 8) as usize % frame.modified_data.len();
                    let bit = bit_position % 8;
                    frame.modified_data[byte_idx] ^= 1 << bit;
                }
            }
            FaultType::Truncate { new_length } => {
                let new_len = (new_length as usize).min(frame.modified_data.len());
                frame.modified_data.truncate(new_len);
            }
            FaultType::Delay { extra_ms } => {
                frame.extra_delay_ms = extra_ms;
            }
            FaultType::CorruptId { new_id } => {
                frame.modified_id = new_id;
            }
            FaultType::DataCorrupt {
                byte_offset,
                new_value,
            } => {
                let idx = byte_offset as usize;
                if idx < frame.modified_data.len() {
                    frame.modified_data[idx] = new_value;
                }
            }
        }
        vec![frame]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn injector() -> FaultInjector {
        FaultInjector::new(42)
    }

    // --- FaultInjector basic ---------------------------------------------

    #[test]
    fn test_new_no_rules() {
        let fi = injector();
        assert_eq!(fi.rule_count(), 0);
        assert_eq!(fi.total_applications(), 0);
    }

    #[test]
    fn test_add_rule_increases_count() {
        let mut fi = injector();
        fi.add_rule(FaultRule::always(1, 0x000, FaultType::Drop));
        assert_eq!(fi.rule_count(), 1);
        fi.add_rule(FaultRule::always(2, 0x000, FaultType::Duplicate));
        assert_eq!(fi.rule_count(), 2);
    }

    // --- Passthrough (no matching rule) -----------------------------------

    #[test]
    fn test_inject_no_rules_passthrough() {
        let mut fi = injector();
        let frames = fi.inject(0x100, &[1, 2, 3]);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].modified_data, vec![1, 2, 3]);
        assert_eq!(frames[0].modified_id, 0x100);
        assert!(!frames[0].dropped);
        assert!(!frames[0].duplicated);
    }

    #[test]
    fn test_inject_mask_no_match_passthrough() {
        let mut fi = injector();
        // mask=0x7FF requires all lower 11 bits set; frame_id=0x100 won't match 0x200 mask
        fi.add_rule(FaultRule::always(1, 0x200, FaultType::Drop));
        let frames = fi.inject(0x100, &[0xAA]);
        assert_eq!(frames.len(), 1);
        assert!(!frames[0].dropped);
    }

    // --- Drop ------------------------------------------------------------

    #[test]
    fn test_drop_returns_empty() {
        let mut fi = injector();
        fi.add_rule(FaultRule::always(1, 0x000, FaultType::Drop));
        let frames = fi.inject(0x123, &[1, 2]);
        assert!(frames.is_empty());
    }

    #[test]
    fn test_drop_increments_count() {
        let mut fi = injector();
        fi.add_rule(FaultRule::always(1, 0x000, FaultType::Drop));
        fi.inject(0x100, &[]);
        assert_eq!(fi.total_applications(), 1);
    }

    // --- Duplicate -------------------------------------------------------

    #[test]
    fn test_duplicate_returns_two() {
        let mut fi = injector();
        fi.add_rule(FaultRule::always(1, 0x000, FaultType::Duplicate));
        let frames = fi.inject(0x100, &[7, 8]);
        assert_eq!(frames.len(), 2);
        assert!(frames[0].duplicated);
        assert!(frames[1].duplicated);
    }

    #[test]
    fn test_duplicate_both_identical() {
        let mut fi = injector();
        fi.add_rule(FaultRule::always(1, 0x000, FaultType::Duplicate));
        let frames = fi.inject(0x200, &[0xAB, 0xCD]);
        assert_eq!(frames[0], frames[1]);
    }

    // --- BitFlip ---------------------------------------------------------

    #[test]
    fn test_bit_flip_bit_zero() {
        let mut fi = injector();
        fi.add_rule(FaultRule::always(1, 0x000, FaultType::BitFlip { bit_position: 0 }));
        let frames = fi.inject(0x100, &[0xFF]);
        assert_eq!(frames[0].modified_data[0], 0xFE); // bit 0 flipped
    }

    #[test]
    fn test_bit_flip_bit_seven() {
        let mut fi = injector();
        fi.add_rule(FaultRule::always(1, 0x000, FaultType::BitFlip { bit_position: 7 }));
        let frames = fi.inject(0x100, &[0x00]);
        assert_eq!(frames[0].modified_data[0], 0x80);
    }

    #[test]
    fn test_bit_flip_empty_data_no_panic() {
        let mut fi = injector();
        fi.add_rule(FaultRule::always(1, 0x000, FaultType::BitFlip { bit_position: 0 }));
        let frames = fi.inject(0x100, &[]);
        assert_eq!(frames.len(), 1);
        assert!(frames[0].modified_data.is_empty());
    }

    #[test]
    fn test_bit_flip_wraps_to_valid_byte() {
        let mut fi = injector();
        // bit_position=8 maps to byte 1 if available; wrap to 0 if only 1 byte
        fi.add_rule(FaultRule::always(1, 0x000, FaultType::BitFlip { bit_position: 8 }));
        let frames = fi.inject(0x100, &[0xFF]); // only 1 byte, 8/8=1 wraps to 0
        assert_eq!(frames[0].modified_data[0], 0xFE);
    }

    // --- Truncate --------------------------------------------------------

    #[test]
    fn test_truncate_reduces_length() {
        let mut fi = injector();
        fi.add_rule(FaultRule::always(1, 0x000, FaultType::Truncate { new_length: 3 }));
        let frames = fi.inject(0x100, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(frames[0].modified_data.len(), 3);
    }

    #[test]
    fn test_truncate_to_zero() {
        let mut fi = injector();
        fi.add_rule(FaultRule::always(1, 0x000, FaultType::Truncate { new_length: 0 }));
        let frames = fi.inject(0x100, &[1, 2, 3]);
        assert!(frames[0].modified_data.is_empty());
    }

    #[test]
    fn test_truncate_larger_than_data_clamped() {
        let mut fi = injector();
        fi.add_rule(FaultRule::always(1, 0x000, FaultType::Truncate { new_length: 100 }));
        let frames = fi.inject(0x100, &[1, 2, 3]);
        assert_eq!(frames[0].modified_data.len(), 3);
    }

    // --- Delay -----------------------------------------------------------

    #[test]
    fn test_delay_sets_extra_ms() {
        let mut fi = injector();
        fi.add_rule(FaultRule::always(1, 0x000, FaultType::Delay { extra_ms: 50 }));
        let frames = fi.inject(0x100, &[0xAA]);
        assert_eq!(frames[0].extra_delay_ms, 50);
        assert_eq!(frames[0].modified_data, vec![0xAA]);
    }

    // --- CorruptId -------------------------------------------------------

    #[test]
    fn test_corrupt_id_changes_id() {
        let mut fi = injector();
        fi.add_rule(FaultRule::always(1, 0x000, FaultType::CorruptId { new_id: 0x7FF }));
        let frames = fi.inject(0x100, &[1]);
        assert_eq!(frames[0].original_id, 0x100);
        assert_eq!(frames[0].modified_id, 0x7FF);
    }

    // --- DataCorrupt -----------------------------------------------------

    #[test]
    fn test_data_corrupt_byte() {
        let mut fi = injector();
        fi.add_rule(FaultRule::always(1, 0x000, FaultType::DataCorrupt { byte_offset: 2, new_value: 0xAB }));
        let frames = fi.inject(0x100, &[0, 1, 2, 3]);
        assert_eq!(frames[0].modified_data[2], 0xAB);
        assert_eq!(frames[0].modified_data[0], 0);
        assert_eq!(frames[0].modified_data[1], 1);
    }

    #[test]
    fn test_data_corrupt_out_of_bounds_no_change() {
        let mut fi = injector();
        fi.add_rule(FaultRule::always(1, 0x000, FaultType::DataCorrupt { byte_offset: 10, new_value: 0xFF }));
        let frames = fi.inject(0x100, &[1, 2, 3]);
        assert_eq!(frames[0].modified_data, vec![1, 2, 3]);
    }

    // --- max_applications ------------------------------------------------

    #[test]
    fn test_max_applications_limit() {
        let mut fi = injector();
        fi.add_rule(FaultRule {
            id: 1,
            frame_id_mask: 0,
            fault: FaultType::Drop,
            probability: 1.0,
            max_applications: Some(2),
        });
        // First two: dropped
        assert!(fi.inject(0x100, &[1]).is_empty());
        assert!(fi.inject(0x100, &[1]).is_empty());
        // Third: passthrough
        let frames = fi.inject(0x100, &[1]);
        assert_eq!(frames.len(), 1);
    }

    #[test]
    fn test_total_applications_counted() {
        let mut fi = injector();
        fi.add_rule(FaultRule::always(1, 0x000, FaultType::Drop));
        fi.inject(0x100, &[]);
        fi.inject(0x100, &[]);
        assert_eq!(fi.total_applications(), 2);
    }

    // --- reset_counts ----------------------------------------------------

    #[test]
    fn test_reset_counts() {
        let mut fi = injector();
        fi.add_rule(FaultRule {
            id: 1,
            frame_id_mask: 0,
            fault: FaultType::Drop,
            probability: 1.0,
            max_applications: Some(1),
        });
        fi.inject(0x100, &[]);
        assert_eq!(fi.total_applications(), 1);
        fi.reset_counts();
        assert_eq!(fi.total_applications(), 0);
        // Rule fires again after reset
        assert!(fi.inject(0x100, &[]).is_empty());
    }

    // --- FaultRule::always -----------------------------------------------

    #[test]
    fn test_fault_rule_always_fields() {
        let r = FaultRule::always(42, 0x7FF, FaultType::Delay { extra_ms: 10 });
        assert_eq!(r.id, 42);
        assert_eq!(r.frame_id_mask, 0x7FF);
        assert_eq!(r.probability, 1.0);
        assert!(r.max_applications.is_none());
    }

    // --- InjectedFrame passthrough fields ---------------------------------

    #[test]
    fn test_passthrough_fields() {
        let f = InjectedFrame::passthrough(0x1FF, &[0xAA, 0xBB]);
        assert_eq!(f.original_id, 0x1FF);
        assert_eq!(f.modified_id, 0x1FF);
        assert_eq!(f.original_data, vec![0xAA, 0xBB]);
        assert_eq!(f.modified_data, vec![0xAA, 0xBB]);
        assert!(!f.dropped);
        assert!(!f.duplicated);
        assert_eq!(f.extra_delay_ms, 0);
    }

    // --- Mask matching ---------------------------------------------------

    #[test]
    fn test_mask_zero_matches_all() {
        let mut fi = injector();
        fi.add_rule(FaultRule::always(1, 0x000, FaultType::Drop));
        assert!(fi.inject(0x000, &[]).is_empty());
        fi.reset_counts();
        assert!(fi.inject(0x7FF, &[]).is_empty());
    }

    #[test]
    fn test_mask_exact_match() {
        let mut fi = injector();
        fi.add_rule(FaultRule::always(1, 0x700, FaultType::Drop));
        // 0x700 & 0x700 == 0x700 ✓
        assert!(fi.inject(0x700, &[]).is_empty());
        // 0x600 & 0x700 = 0x600 ≠ 0x700 → passthrough
        let frames = fi.inject(0x600, &[1]);
        assert_eq!(frames.len(), 1);
    }
}
