//! CAN-to-MQTT/HTTP gateway bridge (in-memory simulation).
//!
//! Provides rule-based routing of CAN frames to MQTT topics, HTTP endpoints,
//! or in-memory channels. Each rule specifies a CAN-ID mask/filter, a
//! transformation operation, and a target.  Processed frames are logged in
//! memory for inspection in tests.

use std::collections::HashMap;
use std::fmt;

// ── CAN frame ────────────────────────────────────────────────────────────────

/// A CAN data frame.
#[derive(Debug, Clone, PartialEq)]
pub struct CanFrame {
    /// 11-bit (standard) or 29-bit (extended) CAN identifier.
    pub id: u32,
    /// Payload data, 0-8 bytes.
    pub data: Vec<u8>,
    /// `true` when the frame uses a 29-bit identifier.
    pub is_extended: bool,
    /// `true` for a remote-frame request (no data).
    pub is_remote: bool,
    /// Millisecond timestamp (set by caller; may be 0 in tests).
    pub timestamp_ms: u64,
}

// ── Bridge target ────────────────────────────────────────────────────────────

/// Where a matched frame should be forwarded.
#[derive(Debug, Clone, PartialEq)]
pub enum BridgeTarget {
    /// MQTT broker topic with quality-of-service level.
    Mqtt { topic: String, qos: u8 },
    /// HTTP endpoint with method (`"POST"` or `"PUT"`).
    Http { url: String, method: String },
    /// In-memory named channel (useful for tests).
    Internal { channel: String },
}

// ── Transform operations ──────────────────────────────────────────────────────

/// How the CAN frame payload should be transformed before forwarding.
#[derive(Debug, Clone)]
pub enum TransformOp {
    /// Forward raw bytes as a hex string (`"0A 1B 2C"`).
    Raw,
    /// Extract named fields from byte ranges.
    Json {
        /// Maps field name → (byte offset, byte count).
        field_map: HashMap<String, (usize, usize)>,
    },
    /// Decode one integer field and apply a linear scale.
    Scale {
        byte_offset: usize,
        byte_count: usize,
        factor: f64,
        offset: f64,
    },
    /// Expand a template using `{byte_N}` placeholders (0-indexed).
    Template(String),
}

// ── Bridge rule ───────────────────────────────────────────────────────────────

/// A routing rule: match → transform → forward.
#[derive(Debug, Clone)]
pub struct BridgeRule {
    /// Unique identifier for this rule.
    pub id: String,
    /// Frame matches when `(frame.id & can_id_mask) == can_id_filter`.
    pub can_id_mask: u32,
    /// See `can_id_mask`.
    pub can_id_filter: u32,
    /// Where to forward the transformed payload.
    pub target: BridgeTarget,
    /// How to transform the frame payload.
    pub transform: TransformOp,
    /// When `false` the rule is skipped.
    pub enabled: bool,
}

// ── Bridge message ────────────────────────────────────────────────────────────

/// A record of one forwarded message produced by a matched rule.
#[derive(Debug, Clone)]
pub struct BridgeMessage {
    /// ID of the rule that produced this message.
    pub rule_id: String,
    /// Target the message was forwarded to.
    pub target: BridgeTarget,
    /// Transformed payload as a UTF-8 string.
    pub payload: String,
    /// CAN frame ID of the source frame.
    pub source_frame_id: u32,
    /// Millisecond timestamp copied from the source frame.
    pub timestamp_ms: u64,
}

// ── Error ─────────────────────────────────────────────────────────────────────

/// Errors produced by the gateway bridge.
#[derive(Debug)]
pub enum BridgeError {
    /// A rule with the given ID was not found.
    RuleNotFound(String),
    /// The transformation failed.
    TransformFailed(String),
    /// The CAN frame is invalid.
    InvalidFrame(String),
    /// A rule with the same ID already exists.
    DuplicateRuleId(String),
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BridgeError::RuleNotFound(id) => write!(f, "rule not found: {id}"),
            BridgeError::TransformFailed(msg) => write!(f, "transform failed: {msg}"),
            BridgeError::InvalidFrame(msg) => write!(f, "invalid frame: {msg}"),
            BridgeError::DuplicateRuleId(id) => write!(f, "duplicate rule id: {id}"),
        }
    }
}

impl std::error::Error for BridgeError {}

// ── Gateway bridge ────────────────────────────────────────────────────────────

/// Routes CAN frames to MQTT/HTTP/internal targets using configurable rules.
pub struct GatewayBridge {
    rules: Vec<BridgeRule>,
    message_log: Vec<BridgeMessage>,
    max_log_size: usize,
    frames_processed: u64,
    messages_forwarded: u64,
}

impl GatewayBridge {
    /// Create a bridge with an unlimited message log.
    pub fn new() -> Self {
        Self::with_max_log(usize::MAX)
    }

    /// Create a bridge with a bounded message log.
    pub fn with_max_log(max_log_size: usize) -> Self {
        Self {
            rules: Vec::new(),
            message_log: Vec::new(),
            max_log_size,
            frames_processed: 0,
            messages_forwarded: 0,
        }
    }

    /// Add a routing rule.
    ///
    /// Returns `Err(BridgeError::DuplicateRuleId)` if a rule with the same ID
    /// already exists.
    pub fn add_rule(&mut self, rule: BridgeRule) -> Result<(), BridgeError> {
        if self.rules.iter().any(|r| r.id == rule.id) {
            return Err(BridgeError::DuplicateRuleId(rule.id.clone()));
        }
        self.rules.push(rule);
        Ok(())
    }

    /// Remove a rule by ID.
    pub fn remove_rule(&mut self, rule_id: &str) -> Result<(), BridgeError> {
        let pos = self
            .rules
            .iter()
            .position(|r| r.id == rule_id)
            .ok_or_else(|| BridgeError::RuleNotFound(rule_id.to_string()))?;
        self.rules.remove(pos);
        Ok(())
    }

    /// Enable a rule.
    pub fn enable_rule(&mut self, rule_id: &str) -> Result<(), BridgeError> {
        self.rules
            .iter_mut()
            .find(|r| r.id == rule_id)
            .ok_or_else(|| BridgeError::RuleNotFound(rule_id.to_string()))
            .map(|r| r.enabled = true)
    }

    /// Disable a rule.
    pub fn disable_rule(&mut self, rule_id: &str) -> Result<(), BridgeError> {
        self.rules
            .iter_mut()
            .find(|r| r.id == rule_id)
            .ok_or_else(|| BridgeError::RuleNotFound(rule_id.to_string()))
            .map(|r| r.enabled = false)
    }

    /// Process one CAN frame: match all enabled rules, transform, and log.
    ///
    /// Returns the list of messages that were produced.
    pub fn process_frame(&mut self, frame: &CanFrame) -> Result<Vec<BridgeMessage>, BridgeError> {
        if frame.data.len() > 8 {
            return Err(BridgeError::InvalidFrame(format!(
                "CAN frame data too long: {} bytes (max 8)",
                frame.data.len()
            )));
        }

        self.frames_processed += 1;

        let mut produced: Vec<BridgeMessage> = Vec::new();

        // Collect matching rules first to avoid borrow conflicts.
        let matching: Vec<(String, BridgeTarget, TransformOp)> = self
            .rules
            .iter()
            .filter(|r| r.enabled && (frame.id & r.can_id_mask) == r.can_id_filter)
            .map(|r| (r.id.clone(), r.target.clone(), r.transform.clone()))
            .collect();

        for (rule_id, target, transform) in matching {
            let payload = Self::apply_transform(frame, &transform)?;
            let msg = BridgeMessage {
                rule_id,
                target: target.clone(),
                payload,
                source_frame_id: frame.id,
                timestamp_ms: frame.timestamp_ms,
            };

            // Respect log size limit.
            if self.message_log.len() >= self.max_log_size {
                self.message_log.remove(0);
            }
            self.message_log.push(msg.clone());
            self.messages_forwarded += 1;
            produced.push(msg);
        }

        Ok(produced)
    }

    /// Immutable view of the message log.
    pub fn message_log(&self) -> &[BridgeMessage] {
        &self.message_log
    }

    /// Total number of CAN frames processed.
    pub fn frames_processed(&self) -> u64 {
        self.frames_processed
    }

    /// Total number of messages forwarded (one per rule match).
    pub fn messages_forwarded(&self) -> u64 {
        self.messages_forwarded
    }

    /// Number of rules currently registered.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Clear the message log without resetting counters.
    pub fn clear_log(&mut self) {
        self.message_log.clear();
    }

    // ── Transform helpers ──────────────────────────────────────────────────────

    /// Apply `op` to `frame.data`, producing a UTF-8 payload string.
    fn apply_transform(frame: &CanFrame, op: &TransformOp) -> Result<String, BridgeError> {
        match op {
            TransformOp::Raw => Ok(Self::bytes_to_hex(&frame.data)),

            TransformOp::Json { field_map } => {
                let mut map: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
                // Sort field names for deterministic output.
                let mut names: Vec<&String> = field_map.keys().collect();
                names.sort();
                for name in names {
                    let (offset, count) = field_map[name];
                    let end = offset + count;
                    if end > frame.data.len() {
                        return Err(BridgeError::TransformFailed(format!(
                            "field '{name}': byte range {offset}..{end} out of bounds (data len {})",
                            frame.data.len()
                        )));
                    }
                    let bytes = &frame.data[offset..end];
                    // Decode as big-endian unsigned integer.
                    let mut value: u64 = 0;
                    for b in bytes {
                        value = (value << 8) | *b as u64;
                    }
                    map.insert(name.clone(), serde_json::Value::Number(value.into()));
                }
                serde_json::to_string(&map).map_err(|e| BridgeError::TransformFailed(e.to_string()))
            }

            TransformOp::Scale {
                byte_offset,
                byte_count,
                factor,
                offset,
            } => {
                let end = byte_offset + byte_count;
                if end > frame.data.len() {
                    return Err(BridgeError::TransformFailed(format!(
                        "scale: byte range {byte_offset}..{end} out of bounds (data len {})",
                        frame.data.len()
                    )));
                }
                let bytes = &frame.data[*byte_offset..end];
                let mut raw: u64 = 0;
                for b in bytes {
                    raw = (raw << 8) | *b as u64;
                }
                let result = raw as f64 * factor + offset;
                Ok(result.to_string())
            }

            TransformOp::Template(template) => {
                let mut result = template.clone();
                for (i, byte) in frame.data.iter().enumerate() {
                    let placeholder = format!("{{byte_{i}}}");
                    result = result.replace(&placeholder, &byte.to_string());
                }
                Ok(result)
            }
        }
    }

    /// Format bytes as an upper-case hex string separated by spaces.
    fn bytes_to_hex(data: &[u8]) -> String {
        data.iter()
            .map(|b| format!("{b:02X}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl Default for GatewayBridge {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn frame(id: u32, data: Vec<u8>) -> CanFrame {
        CanFrame {
            id,
            data,
            is_extended: false,
            is_remote: false,
            timestamp_ms: 0,
        }
    }

    fn mqtt_target(topic: &str) -> BridgeTarget {
        BridgeTarget::Mqtt {
            topic: topic.to_string(),
            qos: 0,
        }
    }

    fn internal_target(channel: &str) -> BridgeTarget {
        BridgeTarget::Internal {
            channel: channel.to_string(),
        }
    }

    fn raw_rule(id: &str, mask: u32, filter: u32) -> BridgeRule {
        BridgeRule {
            id: id.to_string(),
            can_id_mask: mask,
            can_id_filter: filter,
            target: internal_target("test"),
            transform: TransformOp::Raw,
            enabled: true,
        }
    }

    // ── add_rule ──────────────────────────────────────────────────────────────

    #[test]
    fn test_add_rule_increases_count() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("r1", 0xFFFF, 0x100))
            .expect("should succeed");
        assert_eq!(b.rule_count(), 1);
    }

    #[test]
    fn test_add_rule_duplicate_id_error() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("r1", 0xFFFF, 0x100))
            .expect("should succeed");
        let result = b.add_rule(raw_rule("r1", 0xFFFF, 0x200));
        assert!(matches!(result, Err(BridgeError::DuplicateRuleId(_))));
    }

    // ── remove_rule ───────────────────────────────────────────────────────────

    #[test]
    fn test_remove_rule_decreases_count() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("r1", 0xFFFF, 0x100))
            .expect("should succeed");
        b.remove_rule("r1").expect("should succeed");
        assert_eq!(b.rule_count(), 0);
    }

    #[test]
    fn test_remove_rule_not_found_error() {
        let mut b = GatewayBridge::new();
        assert!(matches!(
            b.remove_rule("ghost"),
            Err(BridgeError::RuleNotFound(_))
        ));
    }

    // ── enable / disable ──────────────────────────────────────────────────────

    #[test]
    fn test_disable_rule_stops_matching() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("r1", 0xFFFF, 0x100))
            .expect("should succeed");
        b.disable_rule("r1").expect("should succeed");
        let msgs = b
            .process_frame(&frame(0x100, vec![1, 2]))
            .expect("should succeed");
        assert!(msgs.is_empty());
    }

    #[test]
    fn test_enable_rule_resumes_matching() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("r1", 0xFFFF, 0x100))
            .expect("should succeed");
        b.disable_rule("r1").expect("should succeed");
        b.enable_rule("r1").expect("should succeed");
        let msgs = b
            .process_frame(&frame(0x100, vec![1]))
            .expect("should succeed");
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn test_enable_rule_not_found_error() {
        let mut b = GatewayBridge::new();
        assert!(matches!(
            b.enable_rule("nope"),
            Err(BridgeError::RuleNotFound(_))
        ));
    }

    #[test]
    fn test_disable_rule_not_found_error() {
        let mut b = GatewayBridge::new();
        assert!(matches!(
            b.disable_rule("nope"),
            Err(BridgeError::RuleNotFound(_))
        ));
    }

    // ── process_frame – matching ──────────────────────────────────────────────

    #[test]
    fn test_process_frame_single_matching_rule() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("r1", 0xFFFF, 0x200))
            .expect("should succeed");
        let msgs = b
            .process_frame(&frame(0x200, vec![0xAB]))
            .expect("should succeed");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].rule_id, "r1");
    }

    #[test]
    fn test_process_frame_multiple_matching_rules() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("r1", 0xFFFF, 0x300))
            .expect("should succeed");
        b.add_rule(raw_rule("r2", 0xFFFF, 0x300))
            .expect("should succeed");
        let msgs = b
            .process_frame(&frame(0x300, vec![1]))
            .expect("should succeed");
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn test_process_frame_no_matching_rules() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("r1", 0xFFFF, 0x100))
            .expect("should succeed");
        let msgs = b
            .process_frame(&frame(0x200, vec![1]))
            .expect("should succeed");
        assert!(msgs.is_empty());
    }

    // ── mask/filter matching ──────────────────────────────────────────────────

    #[test]
    fn test_mask_filter_exact_match() {
        let mut b = GatewayBridge::new();
        // mask=0xFFFF → exact match on lower 16 bits.
        b.add_rule(raw_rule("r1", 0xFFFF, 0x100))
            .expect("should succeed");
        assert_eq!(
            b.process_frame(&frame(0x100, vec![]))
                .expect("should succeed")
                .len(),
            1
        );
        assert_eq!(
            b.process_frame(&frame(0x101, vec![]))
                .expect("should succeed")
                .len(),
            0
        );
    }

    #[test]
    fn test_mask_filter_group_match() {
        let mut b = GatewayBridge::new();
        // mask=0xFF00 → match any frame with upper byte = 0x01.
        b.add_rule(raw_rule("r1", 0xFF00, 0x0100))
            .expect("should succeed");
        assert_eq!(
            b.process_frame(&frame(0x0142, vec![]))
                .expect("should succeed")
                .len(),
            1
        );
        assert_eq!(
            b.process_frame(&frame(0x0200, vec![]))
                .expect("should succeed")
                .len(),
            0
        );
    }

    #[test]
    fn test_mask_zero_matches_all() {
        let mut b = GatewayBridge::new();
        // mask=0 → (frame.id & 0) == 0 is always true.
        b.add_rule(raw_rule("catch_all", 0, 0))
            .expect("should succeed");
        for id in [0x000, 0x100, 0x7FF, 0x1FFF_FFFF] {
            let msgs = b.process_frame(&frame(id, vec![])).expect("should succeed");
            assert_eq!(msgs.len(), 1, "frame ID {id:#X} should match catch-all");
        }
    }

    // ── counters ──────────────────────────────────────────────────────────────

    #[test]
    fn test_frames_processed_counter() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("r1", 0xFFFF, 0x100))
            .expect("should succeed");
        b.process_frame(&frame(0x100, vec![]))
            .expect("should succeed");
        b.process_frame(&frame(0x200, vec![]))
            .expect("should succeed");
        assert_eq!(b.frames_processed(), 2);
    }

    #[test]
    fn test_messages_forwarded_counter() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("r1", 0xFFFF, 0x100))
            .expect("should succeed");
        b.add_rule(raw_rule("r2", 0xFFFF, 0x100))
            .expect("should succeed");
        b.process_frame(&frame(0x100, vec![]))
            .expect("should succeed");
        assert_eq!(b.messages_forwarded(), 2);
    }

    // ── Raw transform ─────────────────────────────────────────────────────────

    #[test]
    fn test_raw_transform_hex_output() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("r1", 0xFFFF, 0x10))
            .expect("should succeed");
        let msgs = b
            .process_frame(&frame(0x10, vec![0x0A, 0x1B, 0x2C]))
            .expect("should succeed");
        assert_eq!(msgs[0].payload, "0A 1B 2C");
    }

    #[test]
    fn test_raw_transform_empty_data() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("r1", 0xFFFF, 0x10))
            .expect("should succeed");
        let msgs = b
            .process_frame(&frame(0x10, vec![]))
            .expect("should succeed");
        assert_eq!(msgs[0].payload, "");
    }

    // ── Scale transform ───────────────────────────────────────────────────────

    #[test]
    fn test_scale_transform_basic() {
        let mut b = GatewayBridge::new();
        let rule = BridgeRule {
            id: "temp".into(),
            can_id_mask: 0xFFFF,
            can_id_filter: 0x0A0,
            target: mqtt_target("sensors/temp"),
            transform: TransformOp::Scale {
                byte_offset: 0,
                byte_count: 2,
                factor: 0.1,
                offset: -40.0,
            },
            enabled: true,
        };
        b.add_rule(rule).expect("should succeed");
        // Raw bytes [0x01, 0xF4] = 500 decimal → 500 * 0.1 - 40 = 10.0
        let msgs = b
            .process_frame(&frame(0x0A0, vec![0x01, 0xF4]))
            .expect("should succeed");
        let value: f64 = msgs[0].payload.parse().expect("should be float string");
        assert!((value - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_scale_transform_out_of_bounds_error() {
        let f_frame = frame(0x0A0, vec![0x01]); // only 1 byte
        let op = TransformOp::Scale {
            byte_offset: 0,
            byte_count: 2, // needs 2 bytes
            factor: 1.0,
            offset: 0.0,
        };
        let result = GatewayBridge::apply_transform(&f_frame, &op);
        assert!(result.is_err());
    }

    // ── JSON transform ────────────────────────────────────────────────────────

    #[test]
    fn test_json_transform_single_field() {
        let mut fields = HashMap::new();
        fields.insert("rpm".to_string(), (0usize, 2usize));
        let op = TransformOp::Json { field_map: fields };
        // [0x0E, 0x10] = 3600 decimal
        let f = frame(0, vec![0x0E, 0x10]);
        let result = GatewayBridge::apply_transform(&f, &op).expect("should succeed");
        assert!(result.contains("3600"), "expected 3600 in {result}");
    }

    #[test]
    fn test_json_transform_multiple_fields() {
        let mut fields = HashMap::new();
        fields.insert("a".to_string(), (0usize, 1usize));
        fields.insert("b".to_string(), (1usize, 1usize));
        let op = TransformOp::Json { field_map: fields };
        let f = frame(0, vec![0x01, 0x02]);
        let result = GatewayBridge::apply_transform(&f, &op).expect("should succeed");
        // JSON object with both keys.
        assert!(result.contains("\"a\""));
        assert!(result.contains("\"b\""));
    }

    #[test]
    fn test_json_transform_out_of_bounds_error() {
        let mut fields = HashMap::new();
        fields.insert("x".to_string(), (0usize, 4usize));
        let op = TransformOp::Json { field_map: fields };
        let f = frame(0, vec![0x01]); // only 1 byte
        assert!(GatewayBridge::apply_transform(&f, &op).is_err());
    }

    // ── Template transform ────────────────────────────────────────────────────

    #[test]
    fn test_template_transform_replaces_placeholders() {
        let op = TransformOp::Template("{byte_0} and {byte_1}".to_string());
        let f = frame(0, vec![10, 20]);
        let result = GatewayBridge::apply_transform(&f, &op).expect("should succeed");
        assert_eq!(result, "10 and 20");
    }

    #[test]
    fn test_template_transform_no_placeholders() {
        let op = TransformOp::Template("static payload".to_string());
        let f = frame(0, vec![1, 2, 3]);
        let result = GatewayBridge::apply_transform(&f, &op).expect("should succeed");
        assert_eq!(result, "static payload");
    }

    // ── message log ───────────────────────────────────────────────────────────

    #[test]
    fn test_message_log_records_messages() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("r1", 0xFFFF, 0x10))
            .expect("should succeed");
        b.process_frame(&frame(0x10, vec![0xFF]))
            .expect("should succeed");
        assert_eq!(b.message_log().len(), 1);
    }

    #[test]
    fn test_clear_log() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("r1", 0xFFFF, 0x10))
            .expect("should succeed");
        b.process_frame(&frame(0x10, vec![]))
            .expect("should succeed");
        b.clear_log();
        assert!(b.message_log().is_empty());
    }

    #[test]
    fn test_log_overflow_drops_oldest() {
        let mut b = GatewayBridge::with_max_log(2);
        b.add_rule(raw_rule("r1", 0, 0)).expect("should succeed");
        b.process_frame(&frame(1, vec![])).expect("should succeed"); // log: [m1]
        b.process_frame(&frame(2, vec![])).expect("should succeed"); // log: [m1, m2]
        b.process_frame(&frame(3, vec![])).expect("should succeed"); // log: [m2, m3] — m1 dropped
        assert_eq!(b.message_log().len(), 2);
        assert_eq!(b.message_log()[0].source_frame_id, 2);
        assert_eq!(b.message_log()[1].source_frame_id, 3);
    }

    // ── source frame id in message ─────────────────────────────────────────────

    #[test]
    fn test_message_contains_source_frame_id() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("r1", 0xFFFF, 0x42))
            .expect("should succeed");
        let msgs = b
            .process_frame(&frame(0x42, vec![]))
            .expect("should succeed");
        assert_eq!(msgs[0].source_frame_id, 0x42);
    }

    // ── target preserved in message ────────────────────────────────────────────

    #[test]
    fn test_message_target_matches_rule_target() {
        let mut b = GatewayBridge::new();
        let rule = BridgeRule {
            id: "r1".into(),
            can_id_mask: 0xFFFF,
            can_id_filter: 0x10,
            target: mqtt_target("can/raw"),
            transform: TransformOp::Raw,
            enabled: true,
        };
        b.add_rule(rule).expect("should succeed");
        let msgs = b
            .process_frame(&frame(0x10, vec![]))
            .expect("should succeed");
        assert_eq!(msgs[0].target, mqtt_target("can/raw"));
    }

    // ── bytes_to_hex ───────────────────────────────────────────────────────────

    #[test]
    fn test_bytes_to_hex_single_byte() {
        assert_eq!(GatewayBridge::bytes_to_hex(&[0xAB]), "AB");
    }

    #[test]
    fn test_bytes_to_hex_multiple_bytes() {
        assert_eq!(GatewayBridge::bytes_to_hex(&[0x0A, 0x1B, 0xFF]), "0A 1B FF");
    }

    #[test]
    fn test_bytes_to_hex_empty() {
        assert_eq!(GatewayBridge::bytes_to_hex(&[]), "");
    }

    // ── invalid frame ─────────────────────────────────────────────────────────

    #[test]
    fn test_process_frame_too_long_data_error() {
        let mut b = GatewayBridge::new();
        let bad_frame = CanFrame {
            id: 0x100,
            data: vec![0u8; 9], // 9 bytes > 8
            is_extended: false,
            is_remote: false,
            timestamp_ms: 0,
        };
        assert!(matches!(
            b.process_frame(&bad_frame),
            Err(BridgeError::InvalidFrame(_))
        ));
    }

    // ── error display ─────────────────────────────────────────────────────────

    #[test]
    fn test_bridge_error_display_rule_not_found() {
        let e = BridgeError::RuleNotFound("r99".into());
        assert!(e.to_string().contains("r99"));
    }

    #[test]
    fn test_bridge_error_display_duplicate_rule_id() {
        let e = BridgeError::DuplicateRuleId("r1".into());
        assert!(e.to_string().contains("r1"));
    }

    #[test]
    fn test_bridge_error_display_transform_failed() {
        let e = BridgeError::TransformFailed("oops".into());
        assert!(e.to_string().contains("oops"));
    }

    // ── default ────────────────────────────────────────────────────────────────

    #[test]
    fn test_default_bridge_has_no_rules() {
        let b = GatewayBridge::default();
        assert_eq!(b.rule_count(), 0);
    }

    // ── disabled rule not triggered ────────────────────────────────────────────

    #[test]
    fn test_disabled_rule_not_triggered_for_matching_id() {
        let mut b = GatewayBridge::new();
        let mut rule = raw_rule("r1", 0xFFFF, 0x55);
        rule.enabled = false;
        b.add_rule(rule).expect("should succeed");
        let msgs = b
            .process_frame(&frame(0x55, vec![]))
            .expect("should succeed");
        assert!(msgs.is_empty());
    }

    // ── HTTP target ────────────────────────────────────────────────────────────

    #[test]
    fn test_http_target_forwarded() {
        let mut b = GatewayBridge::new();
        let rule = BridgeRule {
            id: "http_rule".into(),
            can_id_mask: 0xFFFF,
            can_id_filter: 0xAA,
            target: BridgeTarget::Http {
                url: "http://localhost/api".into(),
                method: "POST".into(),
            },
            transform: TransformOp::Raw,
            enabled: true,
        };
        b.add_rule(rule).expect("should succeed");
        let msgs = b
            .process_frame(&frame(0xAA, vec![0x01]))
            .expect("should succeed");
        assert_eq!(msgs.len(), 1);
        assert!(matches!(msgs[0].target, BridgeTarget::Http { .. }));
    }

    // ── extended ID frames ─────────────────────────────────────────────────────

    #[test]
    fn test_extended_id_frame_matched() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("ext", 0x1FFF_FFFF, 0x1234_5678))
            .expect("should succeed");
        let ext_frame = CanFrame {
            id: 0x1234_5678,
            data: vec![],
            is_extended: true,
            is_remote: false,
            timestamp_ms: 0,
        };
        let msgs = b.process_frame(&ext_frame).expect("should succeed");
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn test_remote_frame_processed() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("r1", 0xFFFF, 0x100))
            .expect("should succeed");
        let remote = CanFrame {
            id: 0x100,
            data: vec![],
            is_remote: true,
            is_extended: false,
            timestamp_ms: 42,
        };
        let msgs = b.process_frame(&remote).expect("should succeed");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].timestamp_ms, 42);
    }

    // ── rule id in message ─────────────────────────────────────────────────────

    #[test]
    fn test_message_rule_id_matches() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("unique-rule-id", 0xFFFF, 0x77))
            .expect("should succeed");
        let msgs = b
            .process_frame(&frame(0x77, vec![]))
            .expect("should succeed");
        assert_eq!(msgs[0].rule_id, "unique-rule-id");
    }

    // ── add multiple rules different filters ──────────────────────────────────

    #[test]
    fn test_multiple_rules_different_filters_both_match() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("r1", 0xFFFF, 0x100))
            .expect("should succeed");
        b.add_rule(raw_rule("r2", 0xFFFF, 0x200))
            .expect("should succeed");
        assert_eq!(
            b.process_frame(&frame(0x100, vec![]))
                .expect("should succeed")
                .len(),
            1
        );
        assert_eq!(
            b.process_frame(&frame(0x200, vec![]))
                .expect("should succeed")
                .len(),
            1
        );
    }

    #[test]
    fn test_messages_cleared_on_clear_log() {
        let mut b = GatewayBridge::new();
        b.add_rule(raw_rule("r1", 0, 0)).expect("should succeed");
        b.process_frame(&frame(0x1, vec![]))
            .expect("should succeed");
        b.process_frame(&frame(0x2, vec![]))
            .expect("should succeed");
        assert_eq!(b.message_log().len(), 2);
        b.clear_log();
        assert_eq!(b.message_log().len(), 0);
    }
}
