//! Stream message routing engine.
//!
//! Provides content-based, topic-based, and header-based message routing with
//! round-robin distribution, priority evaluation, regex pattern matching, and
//! dead-letter routing for unroutable messages.

use std::collections::{HashMap, VecDeque};

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// The kind of matching a routing rule uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Match when a payload field equals a specific value.
    ContentBased { field: String, value: String },
    /// Match when the topic name equals a literal string.
    TopicExact(String),
    /// Match when the topic name matches a regex pattern.
    TopicRegex(String),
    /// Match when a header key equals a specific value.
    HeaderBased { key: String, value: String },
    /// Distribute messages round-robin across destinations.
    RoundRobin(Vec<String>),
}

/// A single routing rule with priority and target destination.
#[derive(Debug, Clone)]
pub struct RoutingRule {
    /// Unique identifier for the rule.
    pub id: String,
    /// Evaluation priority (higher is evaluated first).
    pub priority: u32,
    /// The matching strategy.
    pub strategy: RoutingStrategy,
    /// Target destination name (topic, queue, etc.).
    pub destination: String,
    /// Whether the rule is currently active.
    pub enabled: bool,
}

/// A message to be routed.
#[derive(Debug, Clone)]
pub struct RoutableMessage {
    /// Unique message identifier.
    pub id: String,
    /// The topic the message was published to.
    pub topic: String,
    /// Key-value headers attached to the message.
    pub headers: HashMap<String, String>,
    /// Key-value payload fields (simplified for routing decisions).
    pub payload_fields: HashMap<String, String>,
    /// Raw payload bytes.
    pub payload: Vec<u8>,
}

/// The outcome of routing a single message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingOutcome {
    /// Successfully routed to a destination.
    Routed {
        destination: String,
        rule_id: String,
    },
    /// No matching rule — sent to DLQ.
    DeadLettered,
}

/// Aggregate routing statistics.
#[derive(Debug, Clone, Default)]
pub struct RoutingStats {
    /// Total messages evaluated.
    pub total_evaluated: u64,
    /// Messages successfully routed.
    pub total_routed: u64,
    /// Messages sent to DLQ (no matching rule).
    pub total_dead_lettered: u64,
    /// Per-destination message counts.
    pub per_destination: HashMap<String, u64>,
    /// Per-rule match counts.
    pub per_rule: HashMap<String, u64>,
}

/// Configuration for the [`StreamRouter`].
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Maximum number of dead-lettered messages to retain.
    pub dlq_capacity: usize,
    /// Whether to enable the dead-letter queue.
    pub enable_dlq: bool,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            dlq_capacity: 10_000,
            enable_dlq: true,
        }
    }
}

/// Router error types.
#[derive(Debug)]
pub enum RouterError {
    /// A rule with this ID already exists.
    DuplicateRuleId(String),
    /// The referenced rule ID was not found.
    RuleNotFound(String),
    /// The DLQ is full.
    DlqFull,
    /// Invalid regex pattern.
    InvalidRegex(String),
}

impl std::fmt::Display for RouterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RouterError::DuplicateRuleId(id) => write!(f, "duplicate rule id: {}", id),
            RouterError::RuleNotFound(id) => write!(f, "rule not found: {}", id),
            RouterError::DlqFull => write!(f, "dead letter queue is full"),
            RouterError::InvalidRegex(pat) => write!(f, "invalid regex pattern: {}", pat),
        }
    }
}

impl std::error::Error for RouterError {}

// ──────────────────────────────────────────────────────────────────────────────
// StreamRouter
// ──────────────────────────────────────────────────────────────────────────────

/// Stream message router with content-, topic-, and header-based routing,
/// round-robin distribution, priority ordering, and a dead-letter queue.
pub struct StreamRouter {
    config: RouterConfig,
    /// Rules sorted by descending priority (highest first).
    rules: Vec<RoutingRule>,
    /// Dead-lettered messages.
    dlq: VecDeque<RoutableMessage>,
    /// Running statistics.
    stats: RoutingStats,
    /// Round-robin counters keyed by rule ID.
    round_robin_counters: HashMap<String, usize>,
}

impl StreamRouter {
    /// Create a new router with the given configuration.
    pub fn new(config: RouterConfig) -> Self {
        Self {
            config,
            rules: Vec::new(),
            dlq: VecDeque::new(),
            stats: RoutingStats::default(),
            round_robin_counters: HashMap::new(),
        }
    }

    /// Create a router with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(RouterConfig::default())
    }

    /// Add a routing rule. Rules are kept sorted by descending priority.
    pub fn add_rule(&mut self, rule: RoutingRule) -> Result<(), RouterError> {
        if self.rules.iter().any(|r| r.id == rule.id) {
            return Err(RouterError::DuplicateRuleId(rule.id));
        }
        // Validate regex patterns eagerly
        if let RoutingStrategy::TopicRegex(ref pat) = rule.strategy {
            Self::compile_regex(pat)?;
        }
        self.rules.push(rule);
        self.rules.sort_by_key(|b| std::cmp::Reverse(b.priority));
        Ok(())
    }

    /// Remove a routing rule by ID.
    pub fn remove_rule(&mut self, rule_id: &str) -> Result<RoutingRule, RouterError> {
        let idx = self
            .rules
            .iter()
            .position(|r| r.id == rule_id)
            .ok_or_else(|| RouterError::RuleNotFound(rule_id.to_string()))?;
        let removed = self.rules.remove(idx);
        self.round_robin_counters.remove(rule_id);
        Ok(removed)
    }

    /// Update an existing rule in place.
    pub fn update_rule(&mut self, rule: RoutingRule) -> Result<(), RouterError> {
        let idx = self
            .rules
            .iter()
            .position(|r| r.id == rule.id)
            .ok_or_else(|| RouterError::RuleNotFound(rule.id.clone()))?;
        // Validate regex patterns eagerly
        if let RoutingStrategy::TopicRegex(ref pat) = rule.strategy {
            Self::compile_regex(pat)?;
        }
        self.rules[idx] = rule;
        self.rules.sort_by_key(|b| std::cmp::Reverse(b.priority));
        Ok(())
    }

    /// Enable or disable a rule.
    pub fn set_rule_enabled(&mut self, rule_id: &str, enabled: bool) -> Result<(), RouterError> {
        let rule = self
            .rules
            .iter_mut()
            .find(|r| r.id == rule_id)
            .ok_or_else(|| RouterError::RuleNotFound(rule_id.to_string()))?;
        rule.enabled = enabled;
        Ok(())
    }

    /// Return a snapshot of current routing statistics.
    pub fn stats(&self) -> &RoutingStats {
        &self.stats
    }

    /// Return the number of active (enabled) rules.
    pub fn active_rule_count(&self) -> usize {
        self.rules.iter().filter(|r| r.enabled).count()
    }

    /// Return total number of rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Return the dead-letter queue contents.
    pub fn dlq(&self) -> &VecDeque<RoutableMessage> {
        &self.dlq
    }

    /// Pop the oldest dead-lettered message.
    pub fn pop_dlq(&mut self) -> Option<RoutableMessage> {
        self.dlq.pop_front()
    }

    /// Clear all dead-lettered messages.
    pub fn clear_dlq(&mut self) {
        self.dlq.clear();
    }

    /// Route a single message through the rule chain.
    ///
    /// The first matching enabled rule (by priority) wins. If no rule matches,
    /// the message is dead-lettered (if DLQ is enabled).
    pub fn route(&mut self, message: RoutableMessage) -> RoutingOutcome {
        self.stats.total_evaluated += 1;

        // Find the matching rule index first (immutable pass), then update
        // counters and resolve destination (mutable pass) to satisfy the borrow checker.
        let matched_idx = self
            .rules
            .iter()
            .position(|rule| rule.enabled && Self::matches_rule_static(rule, &message));

        if let Some(idx) = matched_idx {
            let rule_id = self.rules[idx].id.clone();
            let destination = self.resolve_destination_by_index(idx);
            self.stats.total_routed += 1;
            *self
                .stats
                .per_destination
                .entry(destination.clone())
                .or_insert(0) += 1;
            *self.stats.per_rule.entry(rule_id.clone()).or_insert(0) += 1;
            return RoutingOutcome::Routed {
                destination,
                rule_id,
            };
        }

        // No rule matched — dead letter
        self.stats.total_dead_lettered += 1;
        if self.config.enable_dlq {
            if self.dlq.len() >= self.config.dlq_capacity {
                // Evict the oldest entry to make room
                self.dlq.pop_front();
            }
            self.dlq.push_back(message);
        }
        RoutingOutcome::DeadLettered
    }

    /// Route a batch of messages. Returns one outcome per message.
    pub fn route_batch(&mut self, messages: Vec<RoutableMessage>) -> Vec<RoutingOutcome> {
        messages.into_iter().map(|m| self.route(m)).collect()
    }

    /// Reset statistics counters to zero.
    pub fn reset_stats(&mut self) {
        self.stats = RoutingStats::default();
    }

    /// Return all rule IDs in priority order (highest first).
    pub fn rule_ids(&self) -> Vec<&str> {
        self.rules.iter().map(|r| r.id.as_str()).collect()
    }

    /// Return a reference to a rule by ID, if it exists.
    pub fn get_rule(&self, rule_id: &str) -> Option<&RoutingRule> {
        self.rules.iter().find(|r| r.id == rule_id)
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Test whether a rule matches a given message (static, no &self needed).
    fn matches_rule_static(rule: &RoutingRule, message: &RoutableMessage) -> bool {
        match &rule.strategy {
            RoutingStrategy::ContentBased { field, value } => {
                message.payload_fields.get(field) == Some(value)
            }
            RoutingStrategy::TopicExact(topic) => message.topic == *topic,
            RoutingStrategy::TopicRegex(pattern) => Self::regex_matches(pattern, &message.topic),
            RoutingStrategy::HeaderBased { key, value } => message.headers.get(key) == Some(value),
            RoutingStrategy::RoundRobin(destinations) => !destinations.is_empty(),
        }
    }

    /// Resolve the actual destination string by rule index. For round-robin, pick the next target.
    fn resolve_destination_by_index(&mut self, idx: usize) -> String {
        let rule = &self.rules[idx];
        match &rule.strategy {
            RoutingStrategy::RoundRobin(destinations) if !destinations.is_empty() => {
                let rule_id = rule.id.clone();
                let counter = self.round_robin_counters.entry(rule_id).or_insert(0);
                let dest_idx = *counter % destinations.len();
                *counter = counter.wrapping_add(1);
                destinations[dest_idx].clone()
            }
            _ => rule.destination.clone(),
        }
    }

    /// Compile a regex pattern, returning an error if invalid.
    fn compile_regex(pattern: &str) -> Result<(), RouterError> {
        // Simple regex-like matching: we support `*` as wildcard and `^`/`$` anchors.
        // For full regex we validate the pattern manually.
        if pattern.is_empty() {
            return Err(RouterError::InvalidRegex("empty pattern".to_string()));
        }
        // Check for unbalanced parentheses
        let mut depth: i32 = 0;
        for ch in pattern.chars() {
            match ch {
                '(' => depth += 1,
                ')' => depth -= 1,
                _ => {}
            }
            if depth < 0 {
                return Err(RouterError::InvalidRegex(
                    "unbalanced parentheses".to_string(),
                ));
            }
        }
        if depth != 0 {
            return Err(RouterError::InvalidRegex(
                "unbalanced parentheses".to_string(),
            ));
        }
        Ok(())
    }

    /// Simple regex-like topic matching.
    ///
    /// Supported syntax:
    /// - `*` matches any sequence of characters (equivalent to `.*`)
    /// - Literal characters match themselves
    /// - `^` and `$` anchor to start/end
    fn regex_matches(pattern: &str, input: &str) -> bool {
        // Convert our simplified pattern to segment-based matching
        let anchored_start = pattern.starts_with('^');
        let anchored_end = pattern.ends_with('$');

        let trimmed = pattern.trim_start_matches('^').trim_end_matches('$');

        if trimmed.is_empty() {
            return input.is_empty();
        }

        // Split on `*` to get literal segments
        let segments: Vec<&str> = trimmed.split('*').collect();

        if segments.len() == 1 {
            // No wildcards
            let seg = segments[0];
            if anchored_start && anchored_end {
                return input == seg;
            }
            if anchored_start {
                return input.starts_with(seg);
            }
            if anchored_end {
                return input.ends_with(seg);
            }
            return input.contains(seg);
        }

        // Multiple segments separated by wildcards
        let mut pos = 0usize;

        for (i, seg) in segments.iter().enumerate() {
            if seg.is_empty() {
                continue;
            }
            if i == 0 && anchored_start {
                if !input.starts_with(seg) {
                    return false;
                }
                pos = seg.len();
                continue;
            }

            match input[pos..].find(seg) {
                Some(found) => {
                    pos += found + seg.len();
                }
                None => return false,
            }
        }

        if anchored_end {
            let last_seg = segments.last().unwrap_or(&"");
            if last_seg.is_empty() {
                return true; // pattern ends with `*`
            }
            return input.ends_with(last_seg);
        }

        true
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_message(id: &str, topic: &str) -> RoutableMessage {
        RoutableMessage {
            id: id.to_string(),
            topic: topic.to_string(),
            headers: HashMap::new(),
            payload_fields: HashMap::new(),
            payload: Vec::new(),
        }
    }

    fn make_message_with_header(id: &str, topic: &str, key: &str, val: &str) -> RoutableMessage {
        let mut msg = make_message(id, topic);
        msg.headers.insert(key.to_string(), val.to_string());
        msg
    }

    fn make_message_with_field(id: &str, topic: &str, key: &str, val: &str) -> RoutableMessage {
        let mut msg = make_message(id, topic);
        msg.payload_fields.insert(key.to_string(), val.to_string());
        msg
    }

    // ── Basic construction ───────────────────────────────────────────────────

    #[test]
    fn test_router_creation_defaults() {
        let router = StreamRouter::with_defaults();
        assert_eq!(router.rule_count(), 0);
        assert_eq!(router.active_rule_count(), 0);
        assert_eq!(router.stats().total_evaluated, 0);
    }

    #[test]
    fn test_router_creation_custom_config() {
        let cfg = RouterConfig {
            dlq_capacity: 500,
            enable_dlq: false,
        };
        let router = StreamRouter::new(cfg);
        assert_eq!(router.rule_count(), 0);
        assert!(router.dlq().is_empty());
    }

    // ── Rule management ──────────────────────────────────────────────────────

    #[test]
    fn test_add_rule() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::TopicExact("orders".to_string()),
            destination: "order-queue".to_string(),
            enabled: true,
        };
        assert!(router.add_rule(rule).is_ok());
        assert_eq!(router.rule_count(), 1);
    }

    #[test]
    fn test_add_duplicate_rule() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::TopicExact("orders".to_string()),
            destination: "q".to_string(),
            enabled: true,
        };
        assert!(router.add_rule(rule.clone()).is_ok());
        assert!(router.add_rule(rule).is_err());
    }

    #[test]
    fn test_remove_rule() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::TopicExact("x".to_string()),
            destination: "y".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();
        let removed = router.remove_rule("r1");
        assert!(removed.is_ok());
        assert_eq!(router.rule_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_rule() {
        let mut router = StreamRouter::with_defaults();
        assert!(router.remove_rule("nope").is_err());
    }

    #[test]
    fn test_update_rule() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 5,
            strategy: RoutingStrategy::TopicExact("a".to_string()),
            destination: "dest1".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();

        let updated = RoutingRule {
            id: "r1".to_string(),
            priority: 20,
            strategy: RoutingStrategy::TopicExact("b".to_string()),
            destination: "dest2".to_string(),
            enabled: true,
        };
        assert!(router.update_rule(updated).is_ok());
        assert_eq!(
            router.get_rule("r1").map(|r| &r.destination),
            Some(&"dest2".to_string())
        );
    }

    #[test]
    fn test_update_nonexistent_rule() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "nope".to_string(),
            priority: 1,
            strategy: RoutingStrategy::TopicExact("x".to_string()),
            destination: "y".to_string(),
            enabled: true,
        };
        assert!(router.update_rule(rule).is_err());
    }

    #[test]
    fn test_enable_disable_rule() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 1,
            strategy: RoutingStrategy::TopicExact("x".to_string()),
            destination: "y".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();
        assert_eq!(router.active_rule_count(), 1);

        router.set_rule_enabled("r1", false).ok();
        assert_eq!(router.active_rule_count(), 0);

        router.set_rule_enabled("r1", true).ok();
        assert_eq!(router.active_rule_count(), 1);
    }

    #[test]
    fn test_enable_nonexistent_rule() {
        let mut router = StreamRouter::with_defaults();
        assert!(router.set_rule_enabled("nope", true).is_err());
    }

    // ── Topic-exact routing ──────────────────────────────────────────────────

    #[test]
    fn test_topic_exact_match() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::TopicExact("orders".to_string()),
            destination: "order-queue".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();

        let msg = make_message("m1", "orders");
        let outcome = router.route(msg);
        assert_eq!(
            outcome,
            RoutingOutcome::Routed {
                destination: "order-queue".to_string(),
                rule_id: "r1".to_string()
            }
        );
    }

    #[test]
    fn test_topic_exact_no_match() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::TopicExact("orders".to_string()),
            destination: "order-queue".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();

        let msg = make_message("m1", "payments");
        let outcome = router.route(msg);
        assert_eq!(outcome, RoutingOutcome::DeadLettered);
    }

    // ── Topic-regex routing ──────────────────────────────────────────────────

    #[test]
    fn test_topic_regex_wildcard() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::TopicRegex("orders.*".to_string()),
            destination: "all-orders".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();

        let msg = make_message("m1", "orders.us");
        assert!(matches!(router.route(msg), RoutingOutcome::Routed { .. }));
    }

    #[test]
    fn test_topic_regex_anchored() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::TopicRegex("^orders$".to_string()),
            destination: "exact-orders".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();

        let match_msg = make_message("m1", "orders");
        assert!(matches!(
            router.route(match_msg),
            RoutingOutcome::Routed { .. }
        ));

        let no_match = make_message("m2", "orders.eu");
        assert_eq!(router.route(no_match), RoutingOutcome::DeadLettered);
    }

    #[test]
    fn test_topic_regex_contains() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::TopicRegex("ship".to_string()),
            destination: "shipping".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();

        let msg = make_message("m1", "order-shipping-us");
        assert!(matches!(router.route(msg), RoutingOutcome::Routed { .. }));
    }

    #[test]
    fn test_invalid_regex_rejected() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::TopicRegex("(".to_string()),
            destination: "dest".to_string(),
            enabled: true,
        };
        assert!(router.add_rule(rule).is_err());
    }

    #[test]
    fn test_empty_regex_rejected() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::TopicRegex(String::new()),
            destination: "dest".to_string(),
            enabled: true,
        };
        assert!(router.add_rule(rule).is_err());
    }

    // ── Content-based routing ────────────────────────────────────────────────

    #[test]
    fn test_content_based_match() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::ContentBased {
                field: "type".to_string(),
                value: "order".to_string(),
            },
            destination: "order-queue".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();

        let msg = make_message_with_field("m1", "events", "type", "order");
        assert!(matches!(router.route(msg), RoutingOutcome::Routed { .. }));
    }

    #[test]
    fn test_content_based_no_match() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::ContentBased {
                field: "type".to_string(),
                value: "order".to_string(),
            },
            destination: "order-queue".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();

        let msg = make_message_with_field("m1", "events", "type", "payment");
        assert_eq!(router.route(msg), RoutingOutcome::DeadLettered);
    }

    #[test]
    fn test_content_based_missing_field() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::ContentBased {
                field: "region".to_string(),
                value: "us".to_string(),
            },
            destination: "us-queue".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();

        let msg = make_message("m1", "events");
        assert_eq!(router.route(msg), RoutingOutcome::DeadLettered);
    }

    // ── Header-based routing ─────────────────────────────────────────────────

    #[test]
    fn test_header_based_match() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::HeaderBased {
                key: "X-Priority".to_string(),
                value: "high".to_string(),
            },
            destination: "priority-queue".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();

        let msg = make_message_with_header("m1", "events", "X-Priority", "high");
        assert!(matches!(router.route(msg), RoutingOutcome::Routed { .. }));
    }

    #[test]
    fn test_header_based_no_match_wrong_value() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::HeaderBased {
                key: "X-Priority".to_string(),
                value: "high".to_string(),
            },
            destination: "priority-queue".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();

        let msg = make_message_with_header("m1", "events", "X-Priority", "low");
        assert_eq!(router.route(msg), RoutingOutcome::DeadLettered);
    }

    #[test]
    fn test_header_based_missing_header() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::HeaderBased {
                key: "X-Priority".to_string(),
                value: "high".to_string(),
            },
            destination: "priority-queue".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();

        let msg = make_message("m1", "events");
        assert_eq!(router.route(msg), RoutingOutcome::DeadLettered);
    }

    // ── Round-robin routing ──────────────────────────────────────────────────

    #[test]
    fn test_round_robin_distribution() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "rr1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::RoundRobin(vec![
                "dest-a".to_string(),
                "dest-b".to_string(),
                "dest-c".to_string(),
            ]),
            destination: String::new(), // not used for RR
            enabled: true,
        };
        router.add_rule(rule).ok();

        let o1 = router.route(make_message("m1", "any"));
        let o2 = router.route(make_message("m2", "any"));
        let o3 = router.route(make_message("m3", "any"));
        let o4 = router.route(make_message("m4", "any"));

        assert_eq!(
            o1,
            RoutingOutcome::Routed {
                destination: "dest-a".to_string(),
                rule_id: "rr1".to_string()
            }
        );
        assert_eq!(
            o2,
            RoutingOutcome::Routed {
                destination: "dest-b".to_string(),
                rule_id: "rr1".to_string()
            }
        );
        assert_eq!(
            o3,
            RoutingOutcome::Routed {
                destination: "dest-c".to_string(),
                rule_id: "rr1".to_string()
            }
        );
        assert_eq!(
            o4,
            RoutingOutcome::Routed {
                destination: "dest-a".to_string(),
                rule_id: "rr1".to_string()
            }
        );
    }

    #[test]
    fn test_round_robin_empty_destinations() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "rr1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::RoundRobin(vec![]),
            destination: String::new(),
            enabled: true,
        };
        router.add_rule(rule).ok();

        let outcome = router.route(make_message("m1", "any"));
        assert_eq!(outcome, RoutingOutcome::DeadLettered);
    }

    // ── Priority ordering ────────────────────────────────────────────────────

    #[test]
    fn test_priority_ordering() {
        let mut router = StreamRouter::with_defaults();

        // Low priority
        let low = RoutingRule {
            id: "low".to_string(),
            priority: 1,
            strategy: RoutingStrategy::TopicExact("orders".to_string()),
            destination: "low-queue".to_string(),
            enabled: true,
        };
        // High priority
        let high = RoutingRule {
            id: "high".to_string(),
            priority: 100,
            strategy: RoutingStrategy::TopicExact("orders".to_string()),
            destination: "high-queue".to_string(),
            enabled: true,
        };

        // Add low first, then high — high should still win
        router.add_rule(low).ok();
        router.add_rule(high).ok();

        let outcome = router.route(make_message("m1", "orders"));
        assert_eq!(
            outcome,
            RoutingOutcome::Routed {
                destination: "high-queue".to_string(),
                rule_id: "high".to_string()
            }
        );
    }

    #[test]
    fn test_disabled_rule_skipped_in_priority() {
        let mut router = StreamRouter::with_defaults();
        let high = RoutingRule {
            id: "high".to_string(),
            priority: 100,
            strategy: RoutingStrategy::TopicExact("orders".to_string()),
            destination: "high-queue".to_string(),
            enabled: false, // disabled
        };
        let low = RoutingRule {
            id: "low".to_string(),
            priority: 1,
            strategy: RoutingStrategy::TopicExact("orders".to_string()),
            destination: "low-queue".to_string(),
            enabled: true,
        };
        router.add_rule(high).ok();
        router.add_rule(low).ok();

        let outcome = router.route(make_message("m1", "orders"));
        assert_eq!(
            outcome,
            RoutingOutcome::Routed {
                destination: "low-queue".to_string(),
                rule_id: "low".to_string()
            }
        );
    }

    // ── Dead letter queue ────────────────────────────────────────────────────

    #[test]
    fn test_dlq_receives_unroutable() {
        let mut router = StreamRouter::with_defaults();
        let msg = make_message("m1", "unknown-topic");
        let outcome = router.route(msg);
        assert_eq!(outcome, RoutingOutcome::DeadLettered);
        assert_eq!(router.dlq().len(), 1);
    }

    #[test]
    fn test_dlq_disabled() {
        let cfg = RouterConfig {
            dlq_capacity: 100,
            enable_dlq: false,
        };
        let mut router = StreamRouter::new(cfg);
        let msg = make_message("m1", "unknown");
        router.route(msg);
        assert!(router.dlq().is_empty());
    }

    #[test]
    fn test_dlq_capacity_eviction() {
        let cfg = RouterConfig {
            dlq_capacity: 2,
            enable_dlq: true,
        };
        let mut router = StreamRouter::new(cfg);
        router.route(make_message("m1", "x"));
        router.route(make_message("m2", "x"));
        router.route(make_message("m3", "x"));
        assert_eq!(router.dlq().len(), 2);
        // m1 should have been evicted
        assert_eq!(router.dlq().front().map(|m| m.id.as_str()), Some("m2"));
    }

    #[test]
    fn test_pop_dlq() {
        let mut router = StreamRouter::with_defaults();
        router.route(make_message("m1", "x"));
        router.route(make_message("m2", "x"));
        let popped = router.pop_dlq();
        assert_eq!(popped.map(|m| m.id), Some("m1".to_string()));
        assert_eq!(router.dlq().len(), 1);
    }

    #[test]
    fn test_clear_dlq() {
        let mut router = StreamRouter::with_defaults();
        router.route(make_message("m1", "x"));
        router.route(make_message("m2", "x"));
        router.clear_dlq();
        assert!(router.dlq().is_empty());
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_tracking() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::TopicExact("orders".to_string()),
            destination: "q".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();

        router.route(make_message("m1", "orders"));
        router.route(make_message("m2", "orders"));
        router.route(make_message("m3", "unknown"));

        assert_eq!(router.stats().total_evaluated, 3);
        assert_eq!(router.stats().total_routed, 2);
        assert_eq!(router.stats().total_dead_lettered, 1);
        assert_eq!(router.stats().per_destination.get("q"), Some(&2));
        assert_eq!(router.stats().per_rule.get("r1"), Some(&2));
    }

    #[test]
    fn test_reset_stats() {
        let mut router = StreamRouter::with_defaults();
        router.route(make_message("m1", "x"));
        router.reset_stats();
        assert_eq!(router.stats().total_evaluated, 0);
        assert_eq!(router.stats().total_routed, 0);
        assert_eq!(router.stats().total_dead_lettered, 0);
    }

    // ── Batch routing ────────────────────────────────────────────────────────

    #[test]
    fn test_batch_routing() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::TopicExact("orders".to_string()),
            destination: "q".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();

        let messages = vec![
            make_message("m1", "orders"),
            make_message("m2", "payments"),
            make_message("m3", "orders"),
        ];
        let outcomes = router.route_batch(messages);
        assert_eq!(outcomes.len(), 3);
        assert!(matches!(outcomes[0], RoutingOutcome::Routed { .. }));
        assert_eq!(outcomes[1], RoutingOutcome::DeadLettered);
        assert!(matches!(outcomes[2], RoutingOutcome::Routed { .. }));
    }

    // ── Rule accessors ───────────────────────────────────────────────────────

    #[test]
    fn test_rule_ids_in_priority_order() {
        let mut router = StreamRouter::with_defaults();
        let r1 = RoutingRule {
            id: "low".to_string(),
            priority: 1,
            strategy: RoutingStrategy::TopicExact("a".to_string()),
            destination: "d".to_string(),
            enabled: true,
        };
        let r2 = RoutingRule {
            id: "high".to_string(),
            priority: 100,
            strategy: RoutingStrategy::TopicExact("b".to_string()),
            destination: "d".to_string(),
            enabled: true,
        };
        router.add_rule(r1).ok();
        router.add_rule(r2).ok();
        let ids = router.rule_ids();
        assert_eq!(ids, vec!["high", "low"]);
    }

    #[test]
    fn test_get_rule() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 5,
            strategy: RoutingStrategy::TopicExact("x".to_string()),
            destination: "y".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();
        assert!(router.get_rule("r1").is_some());
        assert!(router.get_rule("nope").is_none());
    }

    // ── Mixed strategy tests ─────────────────────────────────────────────────

    #[test]
    fn test_multiple_strategies() {
        let mut router = StreamRouter::with_defaults();
        let topic_rule = RoutingRule {
            id: "topic".to_string(),
            priority: 5,
            strategy: RoutingStrategy::TopicExact("orders".to_string()),
            destination: "topic-dest".to_string(),
            enabled: true,
        };
        let header_rule = RoutingRule {
            id: "header".to_string(),
            priority: 10,
            strategy: RoutingStrategy::HeaderBased {
                key: "X-VIP".to_string(),
                value: "true".to_string(),
            },
            destination: "vip-dest".to_string(),
            enabled: true,
        };
        router.add_rule(topic_rule).ok();
        router.add_rule(header_rule).ok();

        // Header rule has higher priority and matches
        let msg = make_message_with_header("m1", "orders", "X-VIP", "true");
        let outcome = router.route(msg);
        assert_eq!(
            outcome,
            RoutingOutcome::Routed {
                destination: "vip-dest".to_string(),
                rule_id: "header".to_string()
            }
        );
    }

    #[test]
    fn test_fallthrough_to_lower_priority() {
        let mut router = StreamRouter::with_defaults();
        let header_rule = RoutingRule {
            id: "header".to_string(),
            priority: 10,
            strategy: RoutingStrategy::HeaderBased {
                key: "X-VIP".to_string(),
                value: "true".to_string(),
            },
            destination: "vip-dest".to_string(),
            enabled: true,
        };
        let topic_rule = RoutingRule {
            id: "topic".to_string(),
            priority: 1,
            strategy: RoutingStrategy::TopicExact("orders".to_string()),
            destination: "topic-dest".to_string(),
            enabled: true,
        };
        router.add_rule(header_rule).ok();
        router.add_rule(topic_rule).ok();

        // No VIP header, falls through to topic match
        let msg = make_message("m1", "orders");
        let outcome = router.route(msg);
        assert_eq!(
            outcome,
            RoutingOutcome::Routed {
                destination: "topic-dest".to_string(),
                rule_id: "topic".to_string()
            }
        );
    }

    // ── Regex edge cases ─────────────────────────────────────────────────────

    #[test]
    fn test_regex_start_anchor() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::TopicRegex("^orders".to_string()),
            destination: "d".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();

        assert!(matches!(
            router.route(make_message("m", "orders.us")),
            RoutingOutcome::Routed { .. }
        ));
        assert_eq!(
            router.route(make_message("m", "pre-orders")),
            RoutingOutcome::DeadLettered
        );
    }

    #[test]
    fn test_regex_end_anchor() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::TopicRegex("orders$".to_string()),
            destination: "d".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();

        assert!(matches!(
            router.route(make_message("m", "all-orders")),
            RoutingOutcome::Routed { .. }
        ));
        assert_eq!(
            router.route(make_message("m", "orders-new")),
            RoutingOutcome::DeadLettered
        );
    }

    #[test]
    fn test_regex_wildcard_middle() {
        let mut router = StreamRouter::with_defaults();
        let rule = RoutingRule {
            id: "r1".to_string(),
            priority: 10,
            strategy: RoutingStrategy::TopicRegex("^order*done$".to_string()),
            destination: "d".to_string(),
            enabled: true,
        };
        router.add_rule(rule).ok();

        assert!(matches!(
            router.route(make_message("m", "order-processing-done")),
            RoutingOutcome::Routed { .. }
        ));
        assert!(matches!(
            router.route(make_message("m", "orderdone")),
            RoutingOutcome::Routed { .. }
        ));
    }

    // ── Router error display ─────────────────────────────────────────────────

    #[test]
    fn test_router_error_display() {
        let e1 = RouterError::DuplicateRuleId("r1".to_string());
        assert!(format!("{}", e1).contains("r1"));

        let e2 = RouterError::RuleNotFound("r2".to_string());
        assert!(format!("{}", e2).contains("r2"));

        let e3 = RouterError::DlqFull;
        assert!(format!("{}", e3).contains("full"));

        let e4 = RouterError::InvalidRegex("bad".to_string());
        assert!(format!("{}", e4).contains("bad"));
    }

    #[test]
    fn test_router_error_is_error() {
        let e: Box<dyn std::error::Error> = Box::new(RouterError::DuplicateRuleId("x".to_string()));
        assert!(!e.to_string().is_empty());
    }

    // ── Routing stats default ────────────────────────────────────────────────

    #[test]
    fn test_routing_stats_default() {
        let stats = RoutingStats::default();
        assert_eq!(stats.total_evaluated, 0);
        assert_eq!(stats.total_routed, 0);
        assert_eq!(stats.total_dead_lettered, 0);
        assert!(stats.per_destination.is_empty());
        assert!(stats.per_rule.is_empty());
    }

    // ── Config default ───────────────────────────────────────────────────────

    #[test]
    fn test_router_config_default() {
        let cfg = RouterConfig::default();
        assert_eq!(cfg.dlq_capacity, 10_000);
        assert!(cfg.enable_dlq);
    }

    // ── RoutingStrategy clone + eq ───────────────────────────────────────────

    #[test]
    fn test_routing_strategy_clone_eq() {
        let s1 = RoutingStrategy::TopicExact("x".to_string());
        let s2 = s1.clone();
        assert_eq!(s1, s2);

        let s3 = RoutingStrategy::ContentBased {
            field: "a".to_string(),
            value: "b".to_string(),
        };
        let s4 = s3.clone();
        assert_eq!(s3, s4);

        let s5 = RoutingStrategy::HeaderBased {
            key: "k".to_string(),
            value: "v".to_string(),
        };
        assert_eq!(s5.clone(), s5);

        let s6 = RoutingStrategy::RoundRobin(vec!["a".to_string()]);
        assert_eq!(s6.clone(), s6);

        let s7 = RoutingStrategy::TopicRegex("pat".to_string());
        assert_eq!(s7.clone(), s7);
    }

    // ── RoutingOutcome ───────────────────────────────────────────────────────

    #[test]
    fn test_routing_outcome_dead_lettered_eq() {
        assert_eq!(RoutingOutcome::DeadLettered, RoutingOutcome::DeadLettered);
    }

    #[test]
    fn test_routing_outcome_routed_eq() {
        let a = RoutingOutcome::Routed {
            destination: "x".to_string(),
            rule_id: "r".to_string(),
        };
        let b = a.clone();
        assert_eq!(a, b);
    }
}
