//! # CAN Frame Filter
//!
//! Provides composable filtering rules for CAN frames, including exact-ID matching,
//! mask-based matching, range matching, data-byte conditions, DLC bounds, and logical
//! combinators (And / Or / Not). A `FilterBank` applies named `FilterSet`s in order
//! and returns an `Accept`, `Reject`, or `Tag` action.

// ─────────────────────────────────────────────────────────────────────────────
// Frame identity types
// ─────────────────────────────────────────────────────────────────────────────

/// CAN identifier type — standard (11-bit) or extended (29-bit).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
pub enum FrameIdType {
    /// Standard 11-bit CAN identifier (0x000–0x7FF).
    Standard(u16),
    /// Extended 29-bit CAN identifier (0x00000000–0x1FFFFFFF).
    Extended(u32),
}

impl FrameIdType {
    /// Return the raw identifier value as a `u32`.
    pub fn value(self) -> u32 {
        match self {
            FrameIdType::Standard(id) => id as u32,
            FrameIdType::Extended(id) => id,
        }
    }
}

/// A single CAN data or remote frame.
#[derive(Clone, Debug)]
pub struct CanFrame {
    /// CAN identifier (standard or extended).
    pub id: FrameIdType,
    /// Frame payload bytes (0–8 bytes for CAN 2.0, or more for CAN FD).
    pub data: Vec<u8>,
    /// Data Length Code — number of payload bytes declared in the frame header.
    pub dlc: u8,
    /// `true` for a Remote Transmission Request (RTR) frame.
    pub is_remote: bool,
    /// Capture timestamp in milliseconds.
    pub timestamp_ms: u64,
}

impl CanFrame {
    /// Return the numeric value of the frame identifier.
    pub fn id_value(&self) -> u32 {
        self.id.value()
    }

    /// Convenience constructor.
    pub fn new(id: FrameIdType, data: Vec<u8>, timestamp_ms: u64) -> Self {
        let dlc = data.len().min(255) as u8;
        Self {
            id,
            data,
            dlc,
            is_remote: false,
            timestamp_ms,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Filter rules
// ─────────────────────────────────────────────────────────────────────────────

/// A composable filter rule for CAN frames.
#[derive(Clone, Debug)]
pub enum FilterRule {
    /// Match a specific CAN identifier exactly.
    ExactId(FrameIdType),
    /// Match if `(frame_id & mask) == (id & mask)`.
    IdMask { id: u32, mask: u32 },
    /// Match if the frame id falls within `[start, end]` (inclusive).
    IdRange { start: u32, end: u32 },
    /// Match if `data[offset] == value` (frame must have at least `offset + 1` bytes).
    DataByte { offset: u8, value: u8 },
    /// Match if `(data[offset] & mask) == (value & mask)`.
    DataMask { offset: u8, value: u8, mask: u8 },
    /// Match if `frame.dlc >= min`.
    DlcMin(u8),
    /// Match if `frame.dlc <= max`.
    DlcMax(u8),
    /// Match if `frame.is_remote == flag`.
    RemoteFrame(bool),
    /// Match if both sub-rules match.
    And(Box<FilterRule>, Box<FilterRule>),
    /// Match if either sub-rule matches.
    Or(Box<FilterRule>, Box<FilterRule>),
    /// Match if the sub-rule does NOT match.
    Not(Box<FilterRule>),
}

// ─────────────────────────────────────────────────────────────────────────────
// Filter evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluates `FilterRule`s against `CanFrame`s.
#[derive(Debug, Default)]
pub struct FrameFilter;

impl FrameFilter {
    /// Create a new `FrameFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Evaluate whether `rule` matches `frame`.
    pub fn matches(rule: &FilterRule, frame: &CanFrame) -> bool {
        match rule {
            FilterRule::ExactId(id) => frame.id == *id,

            FilterRule::IdMask { id, mask } => (frame.id_value() & mask) == (id & mask),

            FilterRule::IdRange { start, end } => {
                let v = frame.id_value();
                v >= *start && v <= *end
            }

            FilterRule::DataByte { offset, value } => {
                frame.data.get(*offset as usize).copied() == Some(*value)
            }

            FilterRule::DataMask {
                offset,
                value,
                mask,
            } => frame
                .data
                .get(*offset as usize)
                .map(|b| (b & mask) == (value & mask))
                .unwrap_or(false),

            FilterRule::DlcMin(min) => frame.dlc >= *min,
            FilterRule::DlcMax(max) => frame.dlc <= *max,
            FilterRule::RemoteFrame(flag) => frame.is_remote == *flag,

            FilterRule::And(a, b) => Self::matches(a, frame) && Self::matches(b, frame),
            FilterRule::Or(a, b) => Self::matches(a, frame) || Self::matches(b, frame),
            FilterRule::Not(inner) => !Self::matches(inner, frame),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Filter sets and bank
// ─────────────────────────────────────────────────────────────────────────────

/// Action to take when a filter set matches a frame.
#[derive(Clone, Debug, PartialEq)]
pub enum FilterAction {
    /// Pass the frame through.
    Accept,
    /// Drop the frame.
    Reject,
    /// Pass the frame and attach a string tag.
    Tag(String),
}

/// A named collection of filter rules evaluated as a conjunction (all must match).
#[derive(Clone, Debug)]
pub struct FilterSet {
    /// Descriptive name for this filter set.
    pub name: String,
    /// Rules — all must match for the set to fire.
    pub rules: Vec<FilterRule>,
    /// Action taken when this set matches a frame.
    pub action: FilterAction,
}

impl FilterSet {
    /// Create a new filter set.
    pub fn new(name: impl Into<String>, rules: Vec<FilterRule>, action: FilterAction) -> Self {
        Self {
            name: name.into(),
            rules,
            action,
        }
    }

    /// Return `true` if all rules in this set match `frame`.
    pub fn matches(&self, frame: &CanFrame) -> bool {
        self.rules.iter().all(|r| FrameFilter::matches(r, frame))
    }
}

/// Statistics from processing a batch of frames through a filter bank.
#[derive(Clone, Debug, Default)]
pub struct FilterStats {
    /// Total frames presented.
    pub total: usize,
    /// Frames that received an `Accept` action.
    pub accepted: usize,
    /// Frames that received a `Reject` action.
    pub rejected: usize,
    /// Frames that received a `Tag` action.
    pub tagged: usize,
}

/// A bank of `FilterSet`s applied in order; the first matching set wins.
///
/// Frames that match no set are `Accept`ed by default.
#[derive(Debug, Default)]
pub struct FilterBank {
    /// Ordered list of filter sets.
    pub filters: Vec<FilterSet>,
}

impl FilterBank {
    /// Create an empty filter bank.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a filter set to the bank.
    pub fn add_filter(&mut self, set: FilterSet) {
        self.filters.push(set);
    }

    /// Determine the action for `frame` by scanning filter sets in order.
    ///
    /// The first matching set's action is returned. If no set matches, `Accept` is the default.
    pub fn apply(&self, frame: &CanFrame) -> FilterAction {
        for set in &self.filters {
            if set.matches(frame) {
                return set.action.clone();
            }
        }
        FilterAction::Accept
    }

    /// Return only those frames whose action is `Accept`.
    pub fn filter_frames<'a>(&self, frames: &'a [CanFrame]) -> Vec<&'a CanFrame> {
        frames
            .iter()
            .filter(|f| matches!(self.apply(f), FilterAction::Accept))
            .collect()
    }

    /// Return all frames paired with their optional tag string.
    ///
    /// `Accept`ed frames get `None`; `Tag`ged frames get `Some(tag)`; `Reject`ed frames
    /// are still included with `None` (callers can check the action separately if needed).
    pub fn tag_frames<'a>(&self, frames: &'a [CanFrame]) -> Vec<(&'a CanFrame, Option<String>)> {
        frames
            .iter()
            .map(|f| {
                let tag = match self.apply(f) {
                    FilterAction::Tag(t) => Some(t),
                    _ => None,
                };
                (f, tag)
            })
            .collect()
    }

    /// Compute statistics over a batch of frames.
    pub fn statistics(&self, frames: &[CanFrame]) -> FilterStats {
        let mut stats = FilterStats {
            total: frames.len(),
            ..Default::default()
        };
        for frame in frames {
            match self.apply(frame) {
                FilterAction::Accept => stats.accepted += 1,
                FilterAction::Reject => stats.rejected += 1,
                FilterAction::Tag(_) => stats.tagged += 1,
            }
        }
        stats
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn std_frame(id: u16, data: &[u8]) -> CanFrame {
        CanFrame::new(FrameIdType::Standard(id), data.to_vec(), 0)
    }

    fn ext_frame(id: u32, data: &[u8]) -> CanFrame {
        CanFrame::new(FrameIdType::Extended(id), data.to_vec(), 0)
    }

    // ── FrameIdType ──────────────────────────────────────────────────────────

    #[test]
    fn test_standard_id_value() {
        assert_eq!(FrameIdType::Standard(0x1FF).value(), 0x1FF);
    }

    #[test]
    fn test_extended_id_value() {
        assert_eq!(FrameIdType::Extended(0x1FFFFFFF).value(), 0x1FFFFFFF);
    }

    #[test]
    fn test_frame_id_copy_clone() {
        let id = FrameIdType::Standard(0x100);
        let id2 = id;
        assert_eq!(id, id2);
    }

    // ── CanFrame ─────────────────────────────────────────────────────────────

    #[test]
    fn test_frame_id_value_std() {
        let f = std_frame(0x7FF, &[]);
        assert_eq!(f.id_value(), 0x7FF);
    }

    #[test]
    fn test_frame_id_value_ext() {
        let f = ext_frame(0x1A2B3C4, &[]);
        assert_eq!(f.id_value(), 0x1A2B3C4);
    }

    #[test]
    fn test_frame_dlc_set_from_data_len() {
        let f = std_frame(0x100, &[1, 2, 3]);
        assert_eq!(f.dlc, 3);
    }

    // ── FilterRule::ExactId ──────────────────────────────────────────────────

    #[test]
    fn test_exact_id_match() {
        let f = std_frame(0x200, &[]);
        assert!(FrameFilter::matches(
            &FilterRule::ExactId(FrameIdType::Standard(0x200)),
            &f
        ));
    }

    #[test]
    fn test_exact_id_no_match() {
        let f = std_frame(0x200, &[]);
        assert!(!FrameFilter::matches(
            &FilterRule::ExactId(FrameIdType::Standard(0x201)),
            &f
        ));
    }

    #[test]
    fn test_exact_id_type_mismatch() {
        let f = std_frame(0x100, &[]);
        // Standard vs Extended with same numeric value
        assert!(!FrameFilter::matches(
            &FilterRule::ExactId(FrameIdType::Extended(0x100)),
            &f
        ));
    }

    // ── FilterRule::IdMask ───────────────────────────────────────────────────

    #[test]
    fn test_id_mask_match() {
        let f = std_frame(0x1F0, &[]);
        assert!(FrameFilter::matches(
            &FilterRule::IdMask {
                id: 0x100,
                mask: 0x700
            },
            &f
        ));
    }

    #[test]
    fn test_id_mask_no_match() {
        let f = std_frame(0x200, &[]);
        assert!(!FrameFilter::matches(
            &FilterRule::IdMask {
                id: 0x100,
                mask: 0x700
            },
            &f
        ));
    }

    // ── FilterRule::IdRange ──────────────────────────────────────────────────

    #[test]
    fn test_id_range_inclusive_start() {
        let f = std_frame(0x100, &[]);
        assert!(FrameFilter::matches(
            &FilterRule::IdRange {
                start: 0x100,
                end: 0x1FF
            },
            &f
        ));
    }

    #[test]
    fn test_id_range_inclusive_end() {
        let f = std_frame(0x1FF, &[]);
        assert!(FrameFilter::matches(
            &FilterRule::IdRange {
                start: 0x100,
                end: 0x1FF
            },
            &f
        ));
    }

    #[test]
    fn test_id_range_outside() {
        let f = std_frame(0x200, &[]);
        assert!(!FrameFilter::matches(
            &FilterRule::IdRange {
                start: 0x100,
                end: 0x1FF
            },
            &f
        ));
    }

    // ── FilterRule::DataByte ─────────────────────────────────────────────────

    #[test]
    fn test_data_byte_match() {
        let f = std_frame(0x100, &[0x00, 0xAB]);
        assert!(FrameFilter::matches(
            &FilterRule::DataByte {
                offset: 1,
                value: 0xAB
            },
            &f
        ));
    }

    #[test]
    fn test_data_byte_no_match() {
        let f = std_frame(0x100, &[0x00, 0xAB]);
        assert!(!FrameFilter::matches(
            &FilterRule::DataByte {
                offset: 1,
                value: 0xCC
            },
            &f
        ));
    }

    #[test]
    fn test_data_byte_out_of_bounds() {
        let f = std_frame(0x100, &[0x01]);
        assert!(!FrameFilter::matches(
            &FilterRule::DataByte {
                offset: 5,
                value: 0x01
            },
            &f
        ));
    }

    // ── FilterRule::DataMask ─────────────────────────────────────────────────

    #[test]
    fn test_data_mask_match() {
        let f = std_frame(0x100, &[0b1010_0011]);
        // mask=0xF0: upper nibble must be 0b1010_xxxx
        assert!(FrameFilter::matches(
            &FilterRule::DataMask {
                offset: 0,
                value: 0b1010_0000,
                mask: 0xF0
            },
            &f
        ));
    }

    #[test]
    fn test_data_mask_no_match() {
        let f = std_frame(0x100, &[0b1111_0000]);
        assert!(!FrameFilter::matches(
            &FilterRule::DataMask {
                offset: 0,
                value: 0b1010_0000,
                mask: 0xF0
            },
            &f
        ));
    }

    // ── FilterRule::DlcMin / DlcMax ──────────────────────────────────────────

    #[test]
    fn test_dlc_min_pass() {
        let f = std_frame(0x100, &[1, 2, 3]);
        assert!(FrameFilter::matches(&FilterRule::DlcMin(3), &f));
    }

    #[test]
    fn test_dlc_min_fail() {
        let f = std_frame(0x100, &[1, 2]);
        assert!(!FrameFilter::matches(&FilterRule::DlcMin(3), &f));
    }

    #[test]
    fn test_dlc_max_pass() {
        let f = std_frame(0x100, &[1, 2]);
        assert!(FrameFilter::matches(&FilterRule::DlcMax(4), &f));
    }

    #[test]
    fn test_dlc_max_fail() {
        let f = std_frame(0x100, &[1, 2, 3, 4, 5]);
        assert!(!FrameFilter::matches(&FilterRule::DlcMax(4), &f));
    }

    // ── FilterRule::RemoteFrame ──────────────────────────────────────────────

    #[test]
    fn test_remote_frame_true() {
        let mut f = std_frame(0x100, &[]);
        f.is_remote = true;
        assert!(FrameFilter::matches(&FilterRule::RemoteFrame(true), &f));
    }

    #[test]
    fn test_remote_frame_false() {
        let f = std_frame(0x100, &[]);
        assert!(FrameFilter::matches(&FilterRule::RemoteFrame(false), &f));
    }

    // ── Combinators ──────────────────────────────────────────────────────────

    #[test]
    fn test_and_both_match() {
        let f = std_frame(0x100, &[0xAB]);
        let rule = FilterRule::And(
            Box::new(FilterRule::ExactId(FrameIdType::Standard(0x100))),
            Box::new(FilterRule::DataByte {
                offset: 0,
                value: 0xAB,
            }),
        );
        assert!(FrameFilter::matches(&rule, &f));
    }

    #[test]
    fn test_and_one_fails() {
        let f = std_frame(0x100, &[0x00]);
        let rule = FilterRule::And(
            Box::new(FilterRule::ExactId(FrameIdType::Standard(0x100))),
            Box::new(FilterRule::DataByte {
                offset: 0,
                value: 0xAB,
            }),
        );
        assert!(!FrameFilter::matches(&rule, &f));
    }

    #[test]
    fn test_or_one_matches() {
        let f = std_frame(0x200, &[]);
        let rule = FilterRule::Or(
            Box::new(FilterRule::ExactId(FrameIdType::Standard(0x100))),
            Box::new(FilterRule::ExactId(FrameIdType::Standard(0x200))),
        );
        assert!(FrameFilter::matches(&rule, &f));
    }

    #[test]
    fn test_or_none_matches() {
        let f = std_frame(0x300, &[]);
        let rule = FilterRule::Or(
            Box::new(FilterRule::ExactId(FrameIdType::Standard(0x100))),
            Box::new(FilterRule::ExactId(FrameIdType::Standard(0x200))),
        );
        assert!(!FrameFilter::matches(&rule, &f));
    }

    #[test]
    fn test_not_inverts() {
        let f = std_frame(0x100, &[]);
        let rule = FilterRule::Not(Box::new(FilterRule::ExactId(FrameIdType::Standard(0x100))));
        assert!(!FrameFilter::matches(&rule, &f));
    }

    #[test]
    fn test_not_passes_when_inner_fails() {
        let f = std_frame(0x200, &[]);
        let rule = FilterRule::Not(Box::new(FilterRule::ExactId(FrameIdType::Standard(0x100))));
        assert!(FrameFilter::matches(&rule, &f));
    }

    // ── FilterBank ───────────────────────────────────────────────────────────

    #[test]
    fn test_filter_bank_default_accept() {
        let bank = FilterBank::new();
        let f = std_frame(0x100, &[]);
        assert_eq!(bank.apply(&f), FilterAction::Accept);
    }

    #[test]
    fn test_filter_bank_reject_rule() {
        let mut bank = FilterBank::new();
        bank.add_filter(FilterSet::new(
            "block",
            vec![FilterRule::ExactId(FrameIdType::Standard(0x100))],
            FilterAction::Reject,
        ));
        let f = std_frame(0x100, &[]);
        assert_eq!(bank.apply(&f), FilterAction::Reject);
    }

    #[test]
    fn test_filter_bank_tag_rule() {
        let mut bank = FilterBank::new();
        bank.add_filter(FilterSet::new(
            "engine",
            vec![FilterRule::IdRange {
                start: 0x0A0,
                end: 0x0AF,
            }],
            FilterAction::Tag("engine".into()),
        ));
        let f = std_frame(0x0A5, &[]);
        assert_eq!(bank.apply(&f), FilterAction::Tag("engine".into()));
    }

    #[test]
    fn test_filter_bank_first_match_wins() {
        let mut bank = FilterBank::new();
        bank.add_filter(FilterSet::new(
            "first",
            vec![FilterRule::DlcMin(0)],
            FilterAction::Tag("first".into()),
        ));
        bank.add_filter(FilterSet::new(
            "second",
            vec![FilterRule::DlcMin(0)],
            FilterAction::Reject,
        ));
        let f = std_frame(0x100, &[]);
        assert_eq!(bank.apply(&f), FilterAction::Tag("first".into()));
    }

    #[test]
    fn test_filter_frames_only_accepted() {
        let mut bank = FilterBank::new();
        bank.add_filter(FilterSet::new(
            "block_0x100",
            vec![FilterRule::ExactId(FrameIdType::Standard(0x100))],
            FilterAction::Reject,
        ));
        let frames = vec![std_frame(0x100, &[]), std_frame(0x200, &[])];
        let accepted = bank.filter_frames(&frames);
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0].id_value(), 0x200);
    }

    #[test]
    fn test_tag_frames_returns_all_with_tags() {
        let mut bank = FilterBank::new();
        bank.add_filter(FilterSet::new(
            "tag_0x100",
            vec![FilterRule::ExactId(FrameIdType::Standard(0x100))],
            FilterAction::Tag("important".into()),
        ));
        let frames = vec![std_frame(0x100, &[]), std_frame(0x200, &[])];
        let tagged = bank.tag_frames(&frames);
        assert_eq!(tagged.len(), 2);
        assert_eq!(tagged[0].1, Some("important".into()));
        assert_eq!(tagged[1].1, None);
    }

    #[test]
    fn test_statistics_counts() {
        let mut bank = FilterBank::new();
        bank.add_filter(FilterSet::new(
            "reject_0x100",
            vec![FilterRule::ExactId(FrameIdType::Standard(0x100))],
            FilterAction::Reject,
        ));
        bank.add_filter(FilterSet::new(
            "tag_0x200",
            vec![FilterRule::ExactId(FrameIdType::Standard(0x200))],
            FilterAction::Tag("t".into()),
        ));
        let frames = vec![
            std_frame(0x100, &[]),
            std_frame(0x200, &[]),
            std_frame(0x300, &[]),
        ];
        let stats = bank.statistics(&frames);
        assert_eq!(stats.total, 3);
        assert_eq!(stats.rejected, 1);
        assert_eq!(stats.tagged, 1);
        assert_eq!(stats.accepted, 1);
    }

    // ── FilterSet ────────────────────────────────────────────────────────────

    #[test]
    fn test_filter_set_all_rules_must_match() {
        let set = FilterSet::new(
            "combo",
            vec![
                FilterRule::ExactId(FrameIdType::Standard(0x100)),
                FilterRule::DlcMin(2),
            ],
            FilterAction::Accept,
        );
        let good = std_frame(0x100, &[1, 2]);
        let bad = std_frame(0x100, &[1]); // dlc < 2
        assert!(set.matches(&good));
        assert!(!set.matches(&bad));
    }

    #[test]
    fn test_filter_set_empty_rules_always_matches() {
        let set = FilterSet::new("empty", vec![], FilterAction::Accept);
        let f = std_frame(0x7FF, &[]);
        assert!(set.matches(&f));
    }

    #[test]
    fn test_filter_stats_empty_frames() {
        let bank = FilterBank::new();
        let stats = bank.statistics(&[]);
        assert_eq!(stats.total, 0);
        assert_eq!(stats.accepted, 0);
    }

    #[test]
    fn test_nested_and_or_not() {
        // Rule: (id == 0x100 OR id == 0x200) AND NOT remote
        let rule = FilterRule::And(
            Box::new(FilterRule::Or(
                Box::new(FilterRule::ExactId(FrameIdType::Standard(0x100))),
                Box::new(FilterRule::ExactId(FrameIdType::Standard(0x200))),
            )),
            Box::new(FilterRule::Not(Box::new(FilterRule::RemoteFrame(true)))),
        );
        let data_frame = std_frame(0x100, &[]);
        let mut remote_frame = std_frame(0x100, &[]);
        remote_frame.is_remote = true;
        let other_frame = std_frame(0x300, &[]);

        assert!(FrameFilter::matches(&rule, &data_frame));
        assert!(!FrameFilter::matches(&rule, &remote_frame));
        assert!(!FrameFilter::matches(&rule, &other_frame));
    }

    #[test]
    fn test_filter_bank_multiple_add_filter() {
        let mut bank = FilterBank::new();
        bank.add_filter(FilterSet::new(
            "f1",
            vec![],
            FilterAction::Tag("first".into()),
        ));
        bank.add_filter(FilterSet::new("f2", vec![], FilterAction::Reject));
        assert_eq!(bank.filters.len(), 2);
    }

    // Additional coverage tests

    #[test]
    fn test_id_mask_full_match() {
        let f = ext_frame(0x18FEF100, &[]);
        // J1939-style: PGN mask 0x03FFFF00
        let rule = FilterRule::IdMask {
            id: 0x18FEF100,
            mask: 0x1FFFFFFF,
        };
        assert!(FrameFilter::matches(&rule, &f));
    }

    #[test]
    fn test_data_mask_out_of_bounds() {
        let f = std_frame(0x100, &[]);
        let rule = FilterRule::DataMask {
            offset: 5,
            value: 0,
            mask: 0xFF,
        };
        assert!(!FrameFilter::matches(&rule, &f));
    }

    #[test]
    fn test_filter_frames_all_rejected() {
        let mut bank = FilterBank::new();
        bank.add_filter(FilterSet::new("block_all", vec![], FilterAction::Reject));
        let frames = vec![std_frame(0x100, &[]), std_frame(0x200, &[])];
        let accepted = bank.filter_frames(&frames);
        assert!(accepted.is_empty());
    }

    #[test]
    fn test_filter_frames_all_accepted() {
        let bank = FilterBank::new();
        let frames = vec![std_frame(0x100, &[]), std_frame(0x200, &[])];
        let accepted = bank.filter_frames(&frames);
        assert_eq!(accepted.len(), 2);
    }

    #[test]
    fn test_filter_action_clone_eq() {
        let a = FilterAction::Tag("hello".into());
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn test_frame_filter_new() {
        let _ff = FrameFilter::new();
    }

    #[test]
    fn test_filter_bank_new_empty() {
        let bank = FilterBank::new();
        assert!(bank.filters.is_empty());
    }

    #[test]
    fn test_stats_all_tagged() {
        let mut bank = FilterBank::new();
        bank.add_filter(FilterSet::new(
            "tag_all",
            vec![],
            FilterAction::Tag("x".into()),
        ));
        let frames: Vec<CanFrame> = (0..5).map(|i| std_frame(i as u16, &[])).collect();
        let stats = bank.statistics(&frames);
        assert_eq!(stats.tagged, 5);
        assert_eq!(stats.accepted, 0);
        assert_eq!(stats.rejected, 0);
    }
}
