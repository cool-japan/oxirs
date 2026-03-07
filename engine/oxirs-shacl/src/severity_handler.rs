/// SHACL severity level handling.
///
/// Provides a `SeverityHandler` that manages the three standard SHACL severity
/// levels (sh:Info, sh:Warning, sh:Violation) plus user-defined custom levels.
/// Supports severity-based filtering, aggregation, threshold configuration,
/// escalation rules, and statistics.
use std::collections::HashMap;

// ── Standard severity levels ─────────────────────────────────────────────────

/// The three SHACL-standard severity levels plus a custom variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SeverityLevel {
    /// sh:Info — informational notice.
    Info,
    /// sh:Warning — advisory issue.
    Warning,
    /// sh:Violation — data does not conform.
    Violation,
    /// A user-defined severity level.
    Custom(CustomSeverity),
}

impl SeverityLevel {
    /// Return the numeric rank: higher values are more severe.
    pub fn rank(&self) -> u32 {
        match self {
            Self::Info => 100,
            Self::Warning => 200,
            Self::Violation => 300,
            Self::Custom(c) => c.rank,
        }
    }

    /// Return the IRI of this severity level.
    pub fn iri(&self) -> &str {
        match self {
            Self::Info => "http://www.w3.org/ns/shacl#Info",
            Self::Warning => "http://www.w3.org/ns/shacl#Warning",
            Self::Violation => "http://www.w3.org/ns/shacl#Violation",
            Self::Custom(c) => &c.iri,
        }
    }

    /// Return a short label for display.
    pub fn label(&self) -> &str {
        match self {
            Self::Info => "Info",
            Self::Warning => "Warning",
            Self::Violation => "Violation",
            Self::Custom(c) => &c.label,
        }
    }

    /// Parse from a SHACL IRI string (e.g. `"http://www.w3.org/ns/shacl#Warning"`).
    pub fn from_iri(iri: &str) -> Option<Self> {
        match iri {
            "http://www.w3.org/ns/shacl#Info" | "sh:Info" => Some(Self::Info),
            "http://www.w3.org/ns/shacl#Warning" | "sh:Warning" => Some(Self::Warning),
            "http://www.w3.org/ns/shacl#Violation" | "sh:Violation" => Some(Self::Violation),
            _ => None,
        }
    }

    /// Returns `true` if this level is at least as severe as `threshold`.
    pub fn meets_threshold(&self, threshold: &SeverityLevel) -> bool {
        self.rank() >= threshold.rank()
    }
}

impl std::fmt::Display for SeverityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

impl PartialOrd for SeverityLevel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SeverityLevel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank().cmp(&other.rank())
    }
}

// ── Custom severity ──────────────────────────────────────────────────────────

/// A user-defined severity level.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CustomSeverity {
    /// Full IRI of the severity.
    pub iri: String,
    /// Short display label.
    pub label: String,
    /// Numeric rank (higher = more severe).
    pub rank: u32,
}

impl CustomSeverity {
    /// Create a new custom severity.
    pub fn new(iri: impl Into<String>, label: impl Into<String>, rank: u32) -> Self {
        Self {
            iri: iri.into(),
            label: label.into(),
            rank,
        }
    }
}

// ── Validation result entry ──────────────────────────────────────────────────

/// A single validation result annotated with severity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeverityResult {
    /// The severity of this result.
    pub severity: SeverityLevel,
    /// The focus node IRI.
    pub focus_node: String,
    /// The constraint that was evaluated.
    pub constraint: String,
    /// Human-readable message.
    pub message: String,
    /// Source shape IRI, if known.
    pub source_shape: Option<String>,
    /// The value that caused the violation, if any.
    pub value: Option<String>,
}

impl SeverityResult {
    /// Create a new result.
    pub fn new(
        severity: SeverityLevel,
        focus_node: impl Into<String>,
        constraint: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            focus_node: focus_node.into(),
            constraint: constraint.into(),
            message: message.into(),
            source_shape: None,
            value: None,
        }
    }

    /// Set the source shape.
    pub fn with_source_shape(mut self, shape: impl Into<String>) -> Self {
        self.source_shape = Some(shape.into());
        self
    }

    /// Set the offending value.
    pub fn with_value(mut self, value: impl Into<String>) -> Self {
        self.value = Some(value.into());
        self
    }
}

// ── Severity statistics ──────────────────────────────────────────────────────

/// Per-level count of results.
#[derive(Debug, Clone, Default)]
pub struct SeverityStatistics {
    /// Number of Info-level results.
    pub info_count: usize,
    /// Number of Warning-level results.
    pub warning_count: usize,
    /// Number of Violation-level results.
    pub violation_count: usize,
    /// Counts of custom severity levels (IRI -> count).
    pub custom_counts: HashMap<String, usize>,
    /// Total result count.
    pub total: usize,
}

impl SeverityStatistics {
    /// Increment the count for a given severity level.
    pub fn record(&mut self, level: &SeverityLevel) {
        self.total += 1;
        match level {
            SeverityLevel::Info => self.info_count += 1,
            SeverityLevel::Warning => self.warning_count += 1,
            SeverityLevel::Violation => self.violation_count += 1,
            SeverityLevel::Custom(c) => {
                *self.custom_counts.entry(c.iri.clone()).or_insert(0) += 1;
            }
        }
    }

    /// Returns `true` when there are zero results at or above `Violation`.
    pub fn conforms(&self) -> bool {
        self.violation_count == 0
    }
}

// ── Escalation rule ──────────────────────────────────────────────────────────

/// A rule that upgrades severity based on context (e.g. constraint type or
/// focus node pattern).
#[derive(Debug, Clone)]
pub struct EscalationRule {
    /// Human-readable name for this rule.
    pub name: String,
    /// Original severity that triggers escalation.
    pub from_severity: SeverityLevel,
    /// The severity to escalate to.
    pub to_severity: SeverityLevel,
    /// Optional constraint pattern: only escalate if the constraint string contains this.
    pub constraint_pattern: Option<String>,
    /// Optional focus-node pattern: only escalate if the focus node contains this.
    pub focus_node_pattern: Option<String>,
}

impl EscalationRule {
    /// Create a new escalation rule.
    pub fn new(name: impl Into<String>, from: SeverityLevel, to: SeverityLevel) -> Self {
        Self {
            name: name.into(),
            from_severity: from,
            to_severity: to,
            constraint_pattern: None,
            focus_node_pattern: None,
        }
    }

    /// Restrict escalation to results whose constraint contains `pattern`.
    pub fn with_constraint_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.constraint_pattern = Some(pattern.into());
        self
    }

    /// Restrict escalation to results whose focus node contains `pattern`.
    pub fn with_focus_node_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.focus_node_pattern = Some(pattern.into());
        self
    }

    /// Check whether this rule applies to the given result and, if so,
    /// return the escalated severity.
    pub fn apply(&self, result: &SeverityResult) -> Option<SeverityLevel> {
        if result.severity != self.from_severity {
            return None;
        }
        if let Some(ref cp) = self.constraint_pattern {
            if !result.constraint.contains(cp.as_str()) {
                return None;
            }
        }
        if let Some(ref fp) = self.focus_node_pattern {
            if !result.focus_node.contains(fp.as_str()) {
                return None;
            }
        }
        Some(self.to_severity.clone())
    }
}

// ── Severity handler ─────────────────────────────────────────────────────────

/// Manages severity levels, filtering, aggregation, and escalation.
pub struct SeverityHandler {
    /// Custom severity levels registered with this handler.
    custom_levels: Vec<CustomSeverity>,
    /// The minimum severity threshold for keeping results.
    threshold: SeverityLevel,
    /// The severity above which validation is considered failed.
    fail_threshold: SeverityLevel,
    /// Escalation rules applied in order.
    escalation_rules: Vec<EscalationRule>,
}

impl SeverityHandler {
    /// Create a handler with default settings (threshold = Info, fail = Violation).
    pub fn new() -> Self {
        Self {
            custom_levels: Vec::new(),
            threshold: SeverityLevel::Info,
            fail_threshold: SeverityLevel::Violation,
            escalation_rules: Vec::new(),
        }
    }

    /// Set the minimum severity threshold; results below this are excluded.
    pub fn with_threshold(mut self, threshold: SeverityLevel) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the failure threshold; if any result meets or exceeds this, the
    /// validation is considered failed.
    pub fn with_fail_threshold(mut self, threshold: SeverityLevel) -> Self {
        self.fail_threshold = threshold;
        self
    }

    /// Register a custom severity level.
    pub fn register_custom(&mut self, custom: CustomSeverity) {
        self.custom_levels.push(custom);
    }

    /// Add an escalation rule.
    pub fn add_escalation_rule(&mut self, rule: EscalationRule) {
        self.escalation_rules.push(rule);
    }

    /// Return all registered custom severity levels.
    pub fn custom_levels(&self) -> &[CustomSeverity] {
        &self.custom_levels
    }

    /// Look up a severity level by IRI, including custom levels.
    pub fn resolve_severity(&self, iri: &str) -> Option<SeverityLevel> {
        if let Some(std) = SeverityLevel::from_iri(iri) {
            return Some(std);
        }
        for c in &self.custom_levels {
            if c.iri == iri {
                return Some(SeverityLevel::Custom(c.clone()));
            }
        }
        None
    }

    // ── Filtering ────────────────────────────────────────────────────────────

    /// Filter a list of results to keep only those meeting the threshold.
    pub fn filter_by_threshold(&self, results: &[SeverityResult]) -> Vec<SeverityResult> {
        results
            .iter()
            .filter(|r| r.severity.meets_threshold(&self.threshold))
            .cloned()
            .collect()
    }

    /// Filter results to keep only a specific severity.
    pub fn filter_by_level(
        results: &[SeverityResult],
        level: &SeverityLevel,
    ) -> Vec<SeverityResult> {
        results
            .iter()
            .filter(|r| r.severity == *level)
            .cloned()
            .collect()
    }

    // ── Aggregation ─────────────────────────────────────────────────────────

    /// Compute severity statistics for a set of results.
    pub fn statistics(results: &[SeverityResult]) -> SeverityStatistics {
        let mut stats = SeverityStatistics::default();
        for r in results {
            stats.record(&r.severity);
        }
        stats
    }

    /// Aggregate results from multiple validation runs: combine into a single
    /// list, de-duplicating by `(focus_node, constraint)`.
    pub fn aggregate(runs: &[Vec<SeverityResult>]) -> Vec<SeverityResult> {
        let mut seen = std::collections::HashSet::new();
        let mut aggregated = Vec::new();
        for run in runs {
            for r in run {
                let key = (r.focus_node.clone(), r.constraint.clone());
                if seen.insert(key) {
                    aggregated.push(r.clone());
                }
            }
        }
        aggregated
    }

    /// Return the highest severity found in the results.
    pub fn max_severity(results: &[SeverityResult]) -> Option<SeverityLevel> {
        results.iter().map(|r| r.severity.clone()).max()
    }

    // ── Validation pass/fail ────────────────────────────────────────────────

    /// Determine whether validation passed: no result meets or exceeds the
    /// fail threshold.
    pub fn validation_passes(&self, results: &[SeverityResult]) -> bool {
        !results
            .iter()
            .any(|r| r.severity.meets_threshold(&self.fail_threshold))
    }

    // ── Escalation ──────────────────────────────────────────────────────────

    /// Apply escalation rules to a set of results, returning a new list with
    /// escalated severities where applicable.
    pub fn apply_escalations(&self, results: &[SeverityResult]) -> Vec<SeverityResult> {
        results
            .iter()
            .map(|r| {
                let mut escalated = r.clone();
                for rule in &self.escalation_rules {
                    if let Some(new_sev) = rule.apply(&escalated) {
                        escalated.severity = new_sev;
                        break; // apply only the first matching rule
                    }
                }
                escalated
            })
            .collect()
    }

    // ── Sorting ─────────────────────────────────────────────────────────────

    /// Sort results by severity (highest first).
    pub fn sort_by_severity(results: &mut [SeverityResult]) {
        results.sort_by(|a, b| b.severity.cmp(&a.severity));
    }
}

impl Default for SeverityHandler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn info_result(node: &str, constraint: &str, msg: &str) -> SeverityResult {
        SeverityResult::new(SeverityLevel::Info, node, constraint, msg)
    }

    fn warning_result(node: &str, constraint: &str, msg: &str) -> SeverityResult {
        SeverityResult::new(SeverityLevel::Warning, node, constraint, msg)
    }

    fn violation_result(node: &str, constraint: &str, msg: &str) -> SeverityResult {
        SeverityResult::new(SeverityLevel::Violation, node, constraint, msg)
    }

    // ── SeverityLevel ───────────────────────────────────────────────────────

    #[test]
    fn test_severity_rank_ordering() {
        assert!(SeverityLevel::Info < SeverityLevel::Warning);
        assert!(SeverityLevel::Warning < SeverityLevel::Violation);
    }

    #[test]
    fn test_severity_iri() {
        assert_eq!(SeverityLevel::Info.iri(), "http://www.w3.org/ns/shacl#Info");
        assert_eq!(
            SeverityLevel::Warning.iri(),
            "http://www.w3.org/ns/shacl#Warning"
        );
        assert_eq!(
            SeverityLevel::Violation.iri(),
            "http://www.w3.org/ns/shacl#Violation"
        );
    }

    #[test]
    fn test_severity_label() {
        assert_eq!(SeverityLevel::Info.label(), "Info");
        assert_eq!(SeverityLevel::Warning.label(), "Warning");
        assert_eq!(SeverityLevel::Violation.label(), "Violation");
    }

    #[test]
    fn test_severity_from_iri() {
        assert_eq!(
            SeverityLevel::from_iri("http://www.w3.org/ns/shacl#Info"),
            Some(SeverityLevel::Info)
        );
        assert_eq!(
            SeverityLevel::from_iri("sh:Warning"),
            Some(SeverityLevel::Warning)
        );
        assert_eq!(
            SeverityLevel::from_iri("sh:Violation"),
            Some(SeverityLevel::Violation)
        );
        assert_eq!(SeverityLevel::from_iri("unknown"), None);
    }

    #[test]
    fn test_severity_meets_threshold() {
        assert!(SeverityLevel::Violation.meets_threshold(&SeverityLevel::Info));
        assert!(SeverityLevel::Warning.meets_threshold(&SeverityLevel::Warning));
        assert!(!SeverityLevel::Info.meets_threshold(&SeverityLevel::Warning));
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(SeverityLevel::Info.to_string(), "Info");
        assert_eq!(SeverityLevel::Warning.to_string(), "Warning");
        assert_eq!(SeverityLevel::Violation.to_string(), "Violation");
    }

    // ── Custom severity ─────────────────────────────────────────────────────

    #[test]
    fn test_custom_severity() {
        let custom = CustomSeverity::new("http://example.org/Critical", "Critical", 400);
        let level = SeverityLevel::Custom(custom);
        assert_eq!(level.rank(), 400);
        assert_eq!(level.label(), "Critical");
        assert!(level > SeverityLevel::Violation);
    }

    #[test]
    fn test_custom_severity_iri() {
        let custom = CustomSeverity::new("http://example.org/Low", "Low", 50);
        let level = SeverityLevel::Custom(custom);
        assert_eq!(level.iri(), "http://example.org/Low");
    }

    // ── SeverityResult ──────────────────────────────────────────────────────

    #[test]
    fn test_severity_result_new() {
        let r = info_result("node1", "sh:minCount", "too few");
        assert_eq!(r.severity, SeverityLevel::Info);
        assert_eq!(r.focus_node, "node1");
        assert_eq!(r.constraint, "sh:minCount");
        assert_eq!(r.message, "too few");
        assert!(r.source_shape.is_none());
        assert!(r.value.is_none());
    }

    #[test]
    fn test_severity_result_with_source_shape() {
        let r = warning_result("n", "c", "m").with_source_shape("shape1");
        assert_eq!(r.source_shape, Some("shape1".into()));
    }

    #[test]
    fn test_severity_result_with_value() {
        let r = violation_result("n", "c", "m").with_value("42");
        assert_eq!(r.value, Some("42".into()));
    }

    // ── SeverityStatistics ──────────────────────────────────────────────────

    #[test]
    fn test_statistics_empty() {
        let stats = SeverityStatistics::default();
        assert_eq!(stats.total, 0);
        assert!(stats.conforms());
    }

    #[test]
    fn test_statistics_record() {
        let mut stats = SeverityStatistics::default();
        stats.record(&SeverityLevel::Info);
        stats.record(&SeverityLevel::Warning);
        stats.record(&SeverityLevel::Violation);
        assert_eq!(stats.info_count, 1);
        assert_eq!(stats.warning_count, 1);
        assert_eq!(stats.violation_count, 1);
        assert_eq!(stats.total, 3);
        assert!(!stats.conforms());
    }

    #[test]
    fn test_statistics_custom() {
        let mut stats = SeverityStatistics::default();
        let c = CustomSeverity::new("urn:test", "Test", 150);
        stats.record(&SeverityLevel::Custom(c.clone()));
        stats.record(&SeverityLevel::Custom(c));
        assert_eq!(stats.custom_counts.get("urn:test"), Some(&2));
        assert_eq!(stats.total, 2);
        assert!(stats.conforms());
    }

    // ── SeverityHandler: filtering ──────────────────────────────────────────

    #[test]
    fn test_filter_by_threshold_default() {
        let handler = SeverityHandler::new();
        let results = vec![
            info_result("n1", "c1", "m1"),
            warning_result("n2", "c2", "m2"),
            violation_result("n3", "c3", "m3"),
        ];
        // Default threshold is Info; all pass.
        let filtered = handler.filter_by_threshold(&results);
        assert_eq!(filtered.len(), 3);
    }

    #[test]
    fn test_filter_by_threshold_warning() {
        let handler = SeverityHandler::new().with_threshold(SeverityLevel::Warning);
        let results = vec![
            info_result("n1", "c1", "m1"),
            warning_result("n2", "c2", "m2"),
            violation_result("n3", "c3", "m3"),
        ];
        let filtered = handler.filter_by_threshold(&results);
        assert_eq!(filtered.len(), 2);
        assert!(filtered
            .iter()
            .all(|r| r.severity >= SeverityLevel::Warning));
    }

    #[test]
    fn test_filter_by_threshold_violation() {
        let handler = SeverityHandler::new().with_threshold(SeverityLevel::Violation);
        let results = vec![
            info_result("n1", "c1", "m1"),
            warning_result("n2", "c2", "m2"),
            violation_result("n3", "c3", "m3"),
        ];
        let filtered = handler.filter_by_threshold(&results);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_filter_by_level() {
        let results = vec![
            info_result("n1", "c1", "m1"),
            warning_result("n2", "c2", "m2"),
            warning_result("n3", "c3", "m3"),
            violation_result("n4", "c4", "m4"),
        ];
        let warnings = SeverityHandler::filter_by_level(&results, &SeverityLevel::Warning);
        assert_eq!(warnings.len(), 2);
    }

    // ── Aggregation ─────────────────────────────────────────────────────────

    #[test]
    fn test_aggregate_deduplicates() {
        let run1 = vec![
            info_result("n1", "c1", "m1"),
            warning_result("n2", "c2", "m2"),
        ];
        let run2 = vec![
            info_result("n1", "c1", "m1"), // duplicate
            violation_result("n3", "c3", "m3"),
        ];
        let aggregated = SeverityHandler::aggregate(&[run1, run2]);
        assert_eq!(aggregated.len(), 3);
    }

    #[test]
    fn test_aggregate_empty() {
        let aggregated = SeverityHandler::aggregate(&[]);
        assert!(aggregated.is_empty());
    }

    // ── Max severity ────────────────────────────────────────────────────────

    #[test]
    fn test_max_severity() {
        let results = vec![
            info_result("n1", "c1", "m1"),
            warning_result("n2", "c2", "m2"),
        ];
        assert_eq!(
            SeverityHandler::max_severity(&results),
            Some(SeverityLevel::Warning)
        );
    }

    #[test]
    fn test_max_severity_empty() {
        let results: Vec<SeverityResult> = vec![];
        assert!(SeverityHandler::max_severity(&results).is_none());
    }

    // ── Validation pass/fail ────────────────────────────────────────────────

    #[test]
    fn test_validation_passes_no_violations() {
        let handler = SeverityHandler::new();
        let results = vec![
            info_result("n1", "c1", "m1"),
            warning_result("n2", "c2", "m2"),
        ];
        assert!(handler.validation_passes(&results));
    }

    #[test]
    fn test_validation_fails_with_violation() {
        let handler = SeverityHandler::new();
        let results = vec![
            info_result("n1", "c1", "m1"),
            violation_result("n2", "c2", "m2"),
        ];
        assert!(!handler.validation_passes(&results));
    }

    #[test]
    fn test_validation_custom_fail_threshold() {
        let handler = SeverityHandler::new().with_fail_threshold(SeverityLevel::Warning);
        let results = vec![warning_result("n1", "c1", "m1")];
        assert!(!handler.validation_passes(&results));
    }

    // ── Custom severity registration ────────────────────────────────────────

    #[test]
    fn test_register_custom_severity() {
        let mut handler = SeverityHandler::new();
        let custom = CustomSeverity::new("urn:critical", "Critical", 400);
        handler.register_custom(custom);
        assert_eq!(handler.custom_levels().len(), 1);
    }

    #[test]
    fn test_resolve_standard_severity() {
        let handler = SeverityHandler::new();
        assert_eq!(
            handler.resolve_severity("sh:Info"),
            Some(SeverityLevel::Info)
        );
    }

    #[test]
    fn test_resolve_custom_severity() {
        let mut handler = SeverityHandler::new();
        handler.register_custom(CustomSeverity::new("urn:critical", "Critical", 400));
        let resolved = handler.resolve_severity("urn:critical");
        assert!(resolved.is_some());
        assert_eq!(resolved.expect("is some").rank(), 400);
    }

    #[test]
    fn test_resolve_unknown_severity() {
        let handler = SeverityHandler::new();
        assert!(handler.resolve_severity("urn:unknown").is_none());
    }

    // ── Escalation rules ────────────────────────────────────────────────────

    #[test]
    fn test_escalation_basic() {
        let mut handler = SeverityHandler::new();
        handler.add_escalation_rule(EscalationRule::new(
            "warn-to-violation",
            SeverityLevel::Warning,
            SeverityLevel::Violation,
        ));
        let results = vec![warning_result("n1", "c1", "m1")];
        let escalated = handler.apply_escalations(&results);
        assert_eq!(escalated[0].severity, SeverityLevel::Violation);
    }

    #[test]
    fn test_escalation_no_match() {
        let mut handler = SeverityHandler::new();
        handler.add_escalation_rule(EscalationRule::new(
            "warn-to-violation",
            SeverityLevel::Warning,
            SeverityLevel::Violation,
        ));
        let results = vec![info_result("n1", "c1", "m1")];
        let escalated = handler.apply_escalations(&results);
        assert_eq!(escalated[0].severity, SeverityLevel::Info); // not escalated
    }

    #[test]
    fn test_escalation_with_constraint_pattern() {
        let mut handler = SeverityHandler::new();
        handler.add_escalation_rule(
            EscalationRule::new(
                "warn-mincount",
                SeverityLevel::Warning,
                SeverityLevel::Violation,
            )
            .with_constraint_pattern("minCount"),
        );
        let r1 = warning_result("n1", "sh:minCount", "m1");
        let r2 = warning_result("n2", "sh:maxCount", "m2");
        let escalated = handler.apply_escalations(&[r1, r2]);
        assert_eq!(escalated[0].severity, SeverityLevel::Violation); // escalated
        assert_eq!(escalated[1].severity, SeverityLevel::Warning); // not escalated
    }

    #[test]
    fn test_escalation_with_focus_node_pattern() {
        let mut handler = SeverityHandler::new();
        handler.add_escalation_rule(
            EscalationRule::new(
                "critical-nodes",
                SeverityLevel::Info,
                SeverityLevel::Warning,
            )
            .with_focus_node_pattern("critical"),
        );
        let r1 = info_result("critical-node-1", "c1", "m1");
        let r2 = info_result("normal-node-2", "c2", "m2");
        let escalated = handler.apply_escalations(&[r1, r2]);
        assert_eq!(escalated[0].severity, SeverityLevel::Warning); // escalated
        assert_eq!(escalated[1].severity, SeverityLevel::Info); // not escalated
    }

    // ── Sorting ─────────────────────────────────────────────────────────────

    #[test]
    fn test_sort_by_severity() {
        let mut results = vec![
            info_result("n1", "c1", "m1"),
            violation_result("n2", "c2", "m2"),
            warning_result("n3", "c3", "m3"),
        ];
        SeverityHandler::sort_by_severity(&mut results);
        assert_eq!(results[0].severity, SeverityLevel::Violation);
        assert_eq!(results[1].severity, SeverityLevel::Warning);
        assert_eq!(results[2].severity, SeverityLevel::Info);
    }

    // ── Statistics via handler ───────────────────────────────────────────────

    #[test]
    fn test_handler_statistics() {
        let results = vec![
            info_result("n1", "c1", "m1"),
            warning_result("n2", "c2", "m2"),
            violation_result("n3", "c3", "m3"),
            violation_result("n4", "c4", "m4"),
        ];
        let stats = SeverityHandler::statistics(&results);
        assert_eq!(stats.info_count, 1);
        assert_eq!(stats.warning_count, 1);
        assert_eq!(stats.violation_count, 2);
        assert_eq!(stats.total, 4);
        assert!(!stats.conforms());
    }

    // ── Default handler ─────────────────────────────────────────────────────

    #[test]
    fn test_handler_default() {
        let handler = SeverityHandler::default();
        assert!(handler.custom_levels().is_empty());
        assert!(handler.escalation_rules.is_empty());
    }

    // ── EscalationRule builder ──────────────────────────────────────────────

    #[test]
    fn test_escalation_rule_builder() {
        let rule = EscalationRule::new("test", SeverityLevel::Info, SeverityLevel::Warning)
            .with_constraint_pattern("sh:datatype")
            .with_focus_node_pattern("special");
        assert_eq!(rule.name, "test");
        assert_eq!(rule.constraint_pattern, Some("sh:datatype".into()));
        assert_eq!(rule.focus_node_pattern, Some("special".into()));
    }

    #[test]
    fn test_escalation_rule_apply_both_patterns() {
        let rule = EscalationRule::new("test", SeverityLevel::Warning, SeverityLevel::Violation)
            .with_constraint_pattern("minCount")
            .with_focus_node_pattern("node-A");
        let r = warning_result("node-A-1", "sh:minCount", "m");
        assert!(rule.apply(&r).is_some());
        // Mismatch on focus node
        let r2 = warning_result("node-B-1", "sh:minCount", "m");
        assert!(rule.apply(&r2).is_none());
    }

    // ── Edge cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_filter_empty_results() {
        let handler = SeverityHandler::new();
        let filtered = handler.filter_by_threshold(&[]);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_validation_passes_empty() {
        let handler = SeverityHandler::new();
        assert!(handler.validation_passes(&[]));
    }

    #[test]
    fn test_aggregate_single_run() {
        let run = vec![info_result("n1", "c1", "m1")];
        let agg = SeverityHandler::aggregate(&[run]);
        assert_eq!(agg.len(), 1);
    }

    #[test]
    fn test_sort_already_sorted() {
        let mut results = vec![
            violation_result("n1", "c1", "m1"),
            warning_result("n2", "c2", "m2"),
            info_result("n3", "c3", "m3"),
        ];
        SeverityHandler::sort_by_severity(&mut results);
        assert_eq!(results[0].severity, SeverityLevel::Violation);
        assert_eq!(results[2].severity, SeverityLevel::Info);
    }
}
