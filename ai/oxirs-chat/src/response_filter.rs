//! Response content filtering and moderation.
//!
//! `ResponseFilter` evaluates a text string against an ordered list of
//! `FilterRule` entries and returns a `FilterResult` describing the action
//! taken. Rules are evaluated in priority order (lowest priority number first);
//! the first matching rule whose action is not `Allow` stops evaluation.

/// The action to take when a rule matches.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterAction {
    /// Pass the text through unchanged.
    Allow,
    /// Reject the text with an explanation.
    Block(String),
    /// Replace matching portions with `[REDACTED]` and allow the cleaned text.
    Redact(String),
}

/// A single filter rule.
#[derive(Debug, Clone)]
pub struct FilterRule {
    /// Unique identifier for this rule.
    pub id: String,
    /// Pattern (substring) to match against the input text.
    pub pattern: String,
    /// Action to take on a match.
    pub action: FilterAction,
    /// Evaluation order: lower numbers are evaluated first.
    pub priority: i32,
}

impl FilterRule {
    /// Construct a new rule.
    pub fn new(
        id: impl Into<String>,
        pattern: impl Into<String>,
        action: FilterAction,
        priority: i32,
    ) -> Self {
        Self {
            id: id.into(),
            pattern: pattern.into(),
            action,
            priority,
        }
    }
}

/// Result of applying the filter to a text.
#[derive(Debug, Clone)]
pub struct FilterResult {
    /// Whether the text is allowed through.
    pub allowed: bool,
    /// The action that was applied (if any rule matched).
    pub action_taken: Option<FilterAction>,
    /// The (possibly redacted) output text. `None` if the text was blocked.
    pub filtered_text: Option<String>,
    /// ID of the rule that fired, if any.
    pub matched_rule: Option<String>,
}

impl FilterResult {
    fn allow(text: String) -> Self {
        Self {
            allowed: true,
            action_taken: Some(FilterAction::Allow),
            filtered_text: Some(text),
            matched_rule: None,
        }
    }

    fn block(reason: String, rule_id: String) -> Self {
        Self {
            allowed: false,
            action_taken: Some(FilterAction::Block(reason)),
            filtered_text: None,
            matched_rule: Some(rule_id),
        }
    }

    fn redact(text: String, label: String, rule_id: String) -> Self {
        Self {
            allowed: true,
            action_taken: Some(FilterAction::Redact(label)),
            filtered_text: Some(text),
            matched_rule: Some(rule_id),
        }
    }
}

/// Response content filter.
pub struct ResponseFilter {
    rules: Vec<FilterRule>,
}

impl ResponseFilter {
    /// Create an empty filter (all text passes through by default).
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a rule. Rules are sorted by priority before evaluation.
    pub fn add_rule(&mut self, rule: FilterRule) {
        self.rules.push(rule);
        self.rules.sort_by_key(|r| r.priority);
    }

    /// Remove a rule by ID. Returns `true` if a rule was removed.
    pub fn remove_rule(&mut self, id: &str) -> bool {
        let before = self.rules.len();
        self.rules.retain(|r| r.id != id);
        self.rules.len() < before
    }

    /// Apply the filter to `text`. Evaluates rules in priority order.
    pub fn filter(&self, text: &str) -> FilterResult {
        for rule in &self.rules {
            if rule.pattern.is_empty() {
                continue;
            }
            if Self::contains_pattern(text, &rule.pattern) {
                return match &rule.action {
                    FilterAction::Allow => FilterResult::allow(text.to_string()),
                    FilterAction::Block(reason) => {
                        FilterResult::block(reason.clone(), rule.id.clone())
                    }
                    FilterAction::Redact(label) => {
                        let cleaned = Self::redact_all(text, &[rule.pattern.clone()]);
                        FilterResult::redact(cleaned, label.clone(), rule.id.clone())
                    }
                };
            }
        }
        // Default: allow unchanged
        FilterResult::allow(text.to_string())
    }

    /// Replace all occurrences of each pattern in `text` with `[REDACTED]`.
    pub fn redact_all(text: &str, patterns: &[String]) -> String {
        let mut result = text.to_string();
        for pattern in patterns {
            if !pattern.is_empty() {
                result = result.replace(pattern.as_str(), "[REDACTED]");
            }
        }
        result
    }

    /// Return `true` if `text` contains `pattern` as a substring (case-sensitive).
    pub fn contains_pattern(text: &str, pattern: &str) -> bool {
        text.contains(pattern)
    }

    /// Number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Collect the patterns of all `Block` rules.
    pub fn blocked_phrases(&self) -> Vec<&str> {
        self.rules
            .iter()
            .filter(|r| matches!(&r.action, FilterAction::Block(_)))
            .map(|r| r.pattern.as_str())
            .collect()
    }
}

impl Default for ResponseFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn block_rule(id: &str, pattern: &str, priority: i32) -> FilterRule {
        FilterRule::new(id, pattern, FilterAction::Block("blocked".into()), priority)
    }

    fn redact_rule(id: &str, pattern: &str, priority: i32) -> FilterRule {
        FilterRule::new(id, pattern, FilterAction::Redact("pii".into()), priority)
    }

    fn allow_rule(id: &str, pattern: &str, priority: i32) -> FilterRule {
        FilterRule::new(id, pattern, FilterAction::Allow, priority)
    }

    // --- FilterAction eq --------------------------------------------------

    #[test]
    fn test_filter_action_eq() {
        assert_eq!(FilterAction::Allow, FilterAction::Allow);
        assert_eq!(
            FilterAction::Block("x".into()),
            FilterAction::Block("x".into())
        );
        assert_ne!(
            FilterAction::Block("x".into()),
            FilterAction::Block("y".into())
        );
    }

    // --- ResponseFilter::new / default ------------------------------------

    #[test]
    fn test_new_empty() {
        let f = ResponseFilter::new();
        assert_eq!(f.rule_count(), 0);
    }

    #[test]
    fn test_default_same_as_new() {
        let f = ResponseFilter::default();
        assert_eq!(f.rule_count(), 0);
    }

    // --- add_rule / rule_count -------------------------------------------

    #[test]
    fn test_add_rule_increases_count() {
        let mut f = ResponseFilter::new();
        f.add_rule(block_rule("r1", "bad", 1));
        assert_eq!(f.rule_count(), 1);
    }

    #[test]
    fn test_add_multiple_rules() {
        let mut f = ResponseFilter::new();
        f.add_rule(block_rule("r1", "bad", 1));
        f.add_rule(block_rule("r2", "evil", 2));
        assert_eq!(f.rule_count(), 2);
    }

    #[test]
    fn test_rules_sorted_by_priority() {
        let mut f = ResponseFilter::new();
        f.add_rule(block_rule("r_low", "low", 10));
        f.add_rule(block_rule("r_high", "high", 1));
        // First rule should be r_high (priority=1)
        assert_eq!(f.rules[0].id, "r_high");
    }

    // --- remove_rule -----------------------------------------------------

    #[test]
    fn test_remove_existing_rule() {
        let mut f = ResponseFilter::new();
        f.add_rule(block_rule("r1", "bad", 1));
        let removed = f.remove_rule("r1");
        assert!(removed);
        assert_eq!(f.rule_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_rule() {
        let mut f = ResponseFilter::new();
        let removed = f.remove_rule("nope");
        assert!(!removed);
    }

    #[test]
    fn test_remove_one_of_two() {
        let mut f = ResponseFilter::new();
        f.add_rule(block_rule("r1", "bad", 1));
        f.add_rule(block_rule("r2", "evil", 2));
        f.remove_rule("r1");
        assert_eq!(f.rule_count(), 1);
        assert_eq!(f.rules[0].id, "r2");
    }

    // --- filter: Allow ---------------------------------------------------

    #[test]
    fn test_filter_no_rules_allows() {
        let f = ResponseFilter::new();
        let result = f.filter("Hello world");
        assert!(result.allowed);
        assert_eq!(result.filtered_text.as_deref(), Some("Hello world"));
    }

    #[test]
    fn test_filter_no_match_allows() {
        let mut f = ResponseFilter::new();
        f.add_rule(block_rule("r1", "bad", 1));
        let result = f.filter("This is fine");
        assert!(result.allowed);
        assert!(result.matched_rule.is_none());
    }

    #[test]
    fn test_filter_allow_rule_matches() {
        let mut f = ResponseFilter::new();
        f.add_rule(allow_rule("r1", "hello", 1));
        let result = f.filter("hello world");
        assert!(result.allowed);
    }

    // --- filter: Block ---------------------------------------------------

    #[test]
    fn test_filter_block_match() {
        let mut f = ResponseFilter::new();
        f.add_rule(block_rule("r1", "offensive", 1));
        let result = f.filter("This is offensive text");
        assert!(!result.allowed);
        assert!(result.filtered_text.is_none());
        assert_eq!(result.matched_rule.as_deref(), Some("r1"));
    }

    #[test]
    fn test_filter_block_reason_preserved() {
        let mut f = ResponseFilter::new();
        f.add_rule(FilterRule::new(
            "r1",
            "hate",
            FilterAction::Block("hate speech detected".into()),
            1,
        ));
        let result = f.filter("hate speech example");
        if let Some(FilterAction::Block(reason)) = &result.action_taken {
            assert_eq!(reason, "hate speech detected");
        } else {
            panic!("expected Block action");
        }
    }

    // --- filter: Redact --------------------------------------------------

    #[test]
    fn test_filter_redact_match() {
        let mut f = ResponseFilter::new();
        f.add_rule(redact_rule("r1", "secret", 1));
        let result = f.filter("This contains a secret value");
        assert!(result.allowed);
        assert_eq!(
            result.filtered_text.as_deref(),
            Some("This contains a [REDACTED] value")
        );
    }

    #[test]
    fn test_filter_redact_rule_id_recorded() {
        let mut f = ResponseFilter::new();
        f.add_rule(redact_rule("pii_rule", "email@example.com", 1));
        let result = f.filter("Contact email@example.com for info");
        assert_eq!(result.matched_rule.as_deref(), Some("pii_rule"));
    }

    // --- priority ordering -----------------------------------------------

    #[test]
    fn test_priority_block_before_redact() {
        let mut f = ResponseFilter::new();
        // Lower number = higher priority
        f.add_rule(block_rule("block_first", "bad", 1));
        f.add_rule(redact_rule("redact_second", "bad", 2));
        let result = f.filter("bad word here");
        assert!(!result.allowed); // block rule fires first
        assert_eq!(result.matched_rule.as_deref(), Some("block_first"));
    }

    #[test]
    fn test_priority_redact_before_block() {
        let mut f = ResponseFilter::new();
        f.add_rule(redact_rule("redact_first", "bad", 1));
        f.add_rule(block_rule("block_second", "bad", 2));
        let result = f.filter("bad word here");
        assert!(result.allowed); // redact rule fires first
        assert_eq!(result.matched_rule.as_deref(), Some("redact_first"));
    }

    // --- redact_all ------------------------------------------------------

    #[test]
    fn test_redact_all_single_pattern() {
        let result =
            ResponseFilter::redact_all("My SSN is 123-45-6789", &["123-45-6789".to_string()]);
        assert_eq!(result, "My SSN is [REDACTED]");
    }

    #[test]
    fn test_redact_all_multiple_patterns() {
        let result = ResponseFilter::redact_all(
            "name: Alice, phone: 555-1234",
            &["Alice".to_string(), "555-1234".to_string()],
        );
        assert_eq!(result, "name: [REDACTED], phone: [REDACTED]");
    }

    #[test]
    fn test_redact_all_no_match() {
        let result = ResponseFilter::redact_all("clean text", &["xyz".to_string()]);
        assert_eq!(result, "clean text");
    }

    #[test]
    fn test_redact_all_empty_patterns() {
        let result = ResponseFilter::redact_all("text here", &[]);
        assert_eq!(result, "text here");
    }

    #[test]
    fn test_redact_all_multiple_occurrences() {
        let result = ResponseFilter::redact_all("bad bad bad", &["bad".to_string()]);
        assert_eq!(result, "[REDACTED] [REDACTED] [REDACTED]");
    }

    // --- contains_pattern ------------------------------------------------

    #[test]
    fn test_contains_pattern_found() {
        assert!(ResponseFilter::contains_pattern("hello world", "world"));
    }

    #[test]
    fn test_contains_pattern_not_found() {
        assert!(!ResponseFilter::contains_pattern("hello world", "xyz"));
    }

    #[test]
    fn test_contains_pattern_empty_text() {
        assert!(!ResponseFilter::contains_pattern("", "pattern"));
    }

    #[test]
    fn test_contains_pattern_empty_pattern() {
        // empty pattern always matches in str::contains
        assert!(ResponseFilter::contains_pattern("text", ""));
    }

    #[test]
    fn test_contains_pattern_case_sensitive() {
        assert!(!ResponseFilter::contains_pattern("Hello", "hello"));
    }

    // --- blocked_phrases -------------------------------------------------

    #[test]
    fn test_blocked_phrases_empty() {
        let f = ResponseFilter::new();
        assert!(f.blocked_phrases().is_empty());
    }

    #[test]
    fn test_blocked_phrases_only_block_rules() {
        let mut f = ResponseFilter::new();
        f.add_rule(block_rule("b1", "hate", 1));
        f.add_rule(redact_rule("r1", "email", 2));
        f.add_rule(allow_rule("a1", "good", 3));
        let phrases = f.blocked_phrases();
        assert_eq!(phrases, vec!["hate"]);
    }

    #[test]
    fn test_blocked_phrases_multiple() {
        let mut f = ResponseFilter::new();
        f.add_rule(block_rule("b1", "bad", 1));
        f.add_rule(block_rule("b2", "evil", 2));
        let phrases = f.blocked_phrases();
        assert_eq!(phrases.len(), 2);
        assert!(phrases.contains(&"bad"));
        assert!(phrases.contains(&"evil"));
    }

    // --- FilterResult helper constructors --------------------------------

    #[test]
    fn test_filter_result_allow() {
        let r = FilterResult::allow("text".into());
        assert!(r.allowed);
        assert!(r.matched_rule.is_none());
    }

    #[test]
    fn test_filter_result_block() {
        let r = FilterResult::block("reason".into(), "rule1".into());
        assert!(!r.allowed);
        assert!(r.filtered_text.is_none());
        assert_eq!(r.matched_rule.as_deref(), Some("rule1"));
    }

    #[test]
    fn test_filter_result_redact() {
        let r = FilterResult::redact("clean".into(), "label".into(), "r1".into());
        assert!(r.allowed);
        assert_eq!(r.filtered_text.as_deref(), Some("clean"));
        assert_eq!(r.matched_rule.as_deref(), Some("r1"));
    }

    // --- FilterRule::new -------------------------------------------------

    #[test]
    fn test_filter_rule_new() {
        let r = FilterRule::new("my-rule", "pattern", FilterAction::Allow, 5);
        assert_eq!(r.id, "my-rule");
        assert_eq!(r.pattern, "pattern");
        assert_eq!(r.priority, 5);
    }

    // --- Empty pattern edge case ----------------------------------------

    #[test]
    fn test_empty_pattern_rule_skipped() {
        let mut f = ResponseFilter::new();
        f.add_rule(FilterRule::new(
            "r1",
            "",
            FilterAction::Block("blocked".into()),
            1,
        ));
        let result = f.filter("any text");
        // Empty-pattern rule is skipped → default allow
        assert!(result.allowed);
    }
}
