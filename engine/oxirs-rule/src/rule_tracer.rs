/// Rule execution tracing and debugging.
///
/// Records which rules fired, when, what they produced, and how facts were
/// derived.  Supports trace filtering by rule ID, predicate, or time range,
/// performance profiling per rule, and export to a human-readable text format.
use std::collections::HashMap;

// ── Timestamps ────────────────────────────────────────────────────────────────

/// A monotonic timestamp in nanoseconds (relative to trace start).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TraceTimestamp(pub u64);

impl TraceTimestamp {
    /// Create a timestamp.
    pub fn new(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Value in nanoseconds.
    pub fn as_nanos(&self) -> u64 {
        self.0
    }

    /// Value in microseconds.
    pub fn as_micros(&self) -> u64 {
        self.0 / 1_000
    }

    /// Value in milliseconds.
    pub fn as_millis(&self) -> u64 {
        self.0 / 1_000_000
    }
}

impl std::fmt::Display for TraceTimestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ms = self.as_millis();
        if ms > 0 {
            write!(f, "{ms}ms")
        } else {
            let us = self.as_micros();
            if us > 0 {
                write!(f, "{us}us")
            } else {
                write!(f, "{}ns", self.0)
            }
        }
    }
}

// ── Derived fact ──────────────────────────────────────────────────────────────

/// A fact (triple) produced by rule execution.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DerivedFact {
    /// Subject term.
    pub subject: String,
    /// Predicate term.
    pub predicate: String,
    /// Object term.
    pub object: String,
}

impl DerivedFact {
    /// Create a new derived fact.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
}

impl std::fmt::Display for DerivedFact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} {} {})", self.subject, self.predicate, self.object)
    }
}

// ── Trace entry ───────────────────────────────────────────────────────────────

/// A single trace entry recording one rule firing.
#[derive(Debug, Clone)]
pub struct TraceEntry {
    /// Unique sequence number.
    pub seq: u64,
    /// The rule that fired.
    pub rule_id: String,
    /// When the rule fired.
    pub timestamp: TraceTimestamp,
    /// Duration of this particular firing (nanoseconds).
    pub duration_nanos: u64,
    /// Facts consumed (matched) by this firing.
    pub consumed: Vec<DerivedFact>,
    /// Facts produced (derived) by this firing.
    pub produced: Vec<DerivedFact>,
    /// Derivation depth (0 = base-level firing).
    pub depth: usize,
    /// Parent trace entry sequence number (if this was triggered by another).
    pub parent_seq: Option<u64>,
}

// ── Derivation chain ──────────────────────────────────────────────────────────

/// A derivation chain showing how a fact was derived through successive rules.
#[derive(Debug, Clone)]
pub struct DerivationChain {
    /// The target fact this chain explains.
    pub fact: DerivedFact,
    /// Steps in the chain from axiom to the target fact (earliest first).
    pub steps: Vec<DerivationStep>,
}

/// A single step in a derivation chain.
#[derive(Debug, Clone)]
pub struct DerivationStep {
    /// The rule that produced the fact.
    pub rule_id: String,
    /// The facts consumed by this step.
    pub inputs: Vec<DerivedFact>,
    /// The fact produced by this step.
    pub output: DerivedFact,
    /// The derivation depth.
    pub depth: usize,
}

// ── Trace filter ──────────────────────────────────────────────────────────────

/// Filter criteria for narrowing a trace.
#[derive(Debug, Clone, Default)]
pub struct TraceFilter {
    /// Only show entries for this rule ID.
    pub rule_id: Option<String>,
    /// Only show entries that produced facts with this predicate.
    pub predicate: Option<String>,
    /// Only show entries at or after this timestamp.
    pub from_time: Option<TraceTimestamp>,
    /// Only show entries at or before this timestamp.
    pub to_time: Option<TraceTimestamp>,
    /// Only show entries at this depth or below.
    pub max_depth: Option<usize>,
}

impl TraceFilter {
    /// Create an empty (pass-all) filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by rule ID.
    pub fn with_rule_id(mut self, rule_id: impl Into<String>) -> Self {
        self.rule_id = Some(rule_id.into());
        self
    }

    /// Filter by produced predicate.
    pub fn with_predicate(mut self, predicate: impl Into<String>) -> Self {
        self.predicate = Some(predicate.into());
        self
    }

    /// Filter by time range (from).
    pub fn with_from_time(mut self, ts: TraceTimestamp) -> Self {
        self.from_time = Some(ts);
        self
    }

    /// Filter by time range (to).
    pub fn with_to_time(mut self, ts: TraceTimestamp) -> Self {
        self.to_time = Some(ts);
        self
    }

    /// Filter by maximum derivation depth.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    /// Check whether a `TraceEntry` passes this filter.
    pub fn matches(&self, entry: &TraceEntry) -> bool {
        if let Some(ref rid) = self.rule_id {
            if entry.rule_id != *rid {
                return false;
            }
        }
        if let Some(ref pred) = self.predicate {
            let has_pred = entry.produced.iter().any(|f| f.predicate == *pred);
            if !has_pred {
                return false;
            }
        }
        if let Some(from) = self.from_time {
            if entry.timestamp < from {
                return false;
            }
        }
        if let Some(to) = self.to_time {
            if entry.timestamp > to {
                return false;
            }
        }
        if let Some(max_d) = self.max_depth {
            if entry.depth > max_d {
                return false;
            }
        }
        true
    }
}

// ── Rule performance profile ──────────────────────────────────────────────────

/// Performance statistics for a single rule.
#[derive(Debug, Clone)]
pub struct RuleProfile {
    /// The rule ID.
    pub rule_id: String,
    /// How many times the rule fired.
    pub fire_count: u64,
    /// Total execution time in nanoseconds.
    pub total_nanos: u64,
    /// Minimum single-firing time in nanoseconds.
    pub min_nanos: u64,
    /// Maximum single-firing time in nanoseconds.
    pub max_nanos: u64,
    /// Total facts produced across all firings.
    pub total_produced: u64,
}

impl RuleProfile {
    fn new(rule_id: String) -> Self {
        Self {
            rule_id,
            fire_count: 0,
            total_nanos: 0,
            min_nanos: u64::MAX,
            max_nanos: 0,
            total_produced: 0,
        }
    }

    /// Average execution time per firing (nanoseconds).
    pub fn avg_nanos(&self) -> u64 {
        if self.fire_count == 0 {
            0
        } else {
            self.total_nanos / self.fire_count
        }
    }

    fn record(&mut self, duration_nanos: u64, produced_count: u64) {
        self.fire_count += 1;
        self.total_nanos += duration_nanos;
        if duration_nanos < self.min_nanos {
            self.min_nanos = duration_nanos;
        }
        if duration_nanos > self.max_nanos {
            self.max_nanos = duration_nanos;
        }
        self.total_produced += produced_count;
    }
}

// ── Trace statistics summary ─────────────────────────────────────────────────

/// Summary statistics for an entire trace.
#[derive(Debug, Clone)]
pub struct TraceStatistics {
    /// Total number of rule firings.
    pub total_firings: u64,
    /// Total distinct rules that fired.
    pub distinct_rules: usize,
    /// Total facts derived.
    pub total_facts_derived: u64,
    /// Maximum derivation depth reached.
    pub max_depth: usize,
    /// The rule that fired most often.
    pub most_fired_rule: Option<(String, u64)>,
    /// The longest derivation chain length.
    pub longest_chain: usize,
    /// Total trace time span (nanoseconds).
    pub time_span_nanos: u64,
}

// ── Rule tracer ───────────────────────────────────────────────────────────────

/// Records rule execution traces for debugging and profiling.
pub struct RuleTracer {
    entries: Vec<TraceEntry>,
    next_seq: u64,
    max_depth_limit: Option<usize>,
    profiles: HashMap<String, RuleProfile>,
    /// Maps (predicate, subject, object) -> seq of the first entry that produced it.
    derivation_index: HashMap<(String, String, String), u64>,
}

impl RuleTracer {
    /// Create a new, empty tracer.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_seq: 0,
            max_depth_limit: None,
            profiles: HashMap::new(),
            derivation_index: HashMap::new(),
        }
    }

    /// Create a tracer with a maximum recording depth.  Entries deeper than
    /// this are silently discarded.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth_limit = Some(max_depth);
        self
    }

    /// Record a rule firing.
    ///
    /// Returns the sequence number assigned to this entry, or `None` if the
    /// entry was discarded due to depth limiting.
    #[allow(clippy::too_many_arguments)]
    pub fn record_firing(
        &mut self,
        rule_id: impl Into<String>,
        timestamp: TraceTimestamp,
        duration_nanos: u64,
        consumed: Vec<DerivedFact>,
        produced: Vec<DerivedFact>,
        depth: usize,
        parent_seq: Option<u64>,
    ) -> Option<u64> {
        if let Some(limit) = self.max_depth_limit {
            if depth > limit {
                return None;
            }
        }

        let rule_id = rule_id.into();
        let seq = self.next_seq;
        self.next_seq += 1;

        // Update profiling.
        let profile = self
            .profiles
            .entry(rule_id.clone())
            .or_insert_with(|| RuleProfile::new(rule_id.clone()));
        profile.record(duration_nanos, produced.len() as u64);

        // Update derivation index.
        for fact in &produced {
            let key = (
                fact.predicate.clone(),
                fact.subject.clone(),
                fact.object.clone(),
            );
            self.derivation_index.entry(key).or_insert(seq);
        }

        self.entries.push(TraceEntry {
            seq,
            rule_id,
            timestamp,
            duration_nanos,
            consumed,
            produced,
            depth,
            parent_seq,
        });

        Some(seq)
    }

    /// Return all trace entries.
    pub fn entries(&self) -> &[TraceEntry] {
        &self.entries
    }

    /// Return entries matching a filter.
    pub fn filter(&self, filter: &TraceFilter) -> Vec<&TraceEntry> {
        self.entries.iter().filter(|e| filter.matches(e)).collect()
    }

    /// Return the number of recorded entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if no entries have been recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all trace data.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.next_seq = 0;
        self.profiles.clear();
        self.derivation_index.clear();
    }

    /// Retrieve the trace entry with the given sequence number.
    pub fn get_entry(&self, seq: u64) -> Option<&TraceEntry> {
        self.entries.iter().find(|e| e.seq == seq)
    }

    // ── Derivation chains ────────────────────────────────────────────────────

    /// Build a derivation chain for a given fact, tracing back through parent
    /// entries to the axioms.
    pub fn derivation_chain(&self, fact: &DerivedFact) -> Option<DerivationChain> {
        let key = (
            fact.predicate.clone(),
            fact.subject.clone(),
            fact.object.clone(),
        );
        let start_seq = self.derivation_index.get(&key).copied()?;
        let mut steps = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self.collect_chain(start_seq, &mut steps, &mut visited);
        steps.reverse();
        Some(DerivationChain {
            fact: fact.clone(),
            steps,
        })
    }

    fn collect_chain(
        &self,
        seq: u64,
        steps: &mut Vec<DerivationStep>,
        visited: &mut std::collections::HashSet<u64>,
    ) {
        if !visited.insert(seq) {
            return;
        }
        if let Some(entry) = self.get_entry(seq) {
            for produced in &entry.produced {
                steps.push(DerivationStep {
                    rule_id: entry.rule_id.clone(),
                    inputs: entry.consumed.clone(),
                    output: produced.clone(),
                    depth: entry.depth,
                });
            }
            if let Some(parent) = entry.parent_seq {
                self.collect_chain(parent, steps, visited);
            }
        }
    }

    // ── Performance profiling ────────────────────────────────────────────────

    /// Return the performance profile for a specific rule.
    pub fn profile(&self, rule_id: &str) -> Option<&RuleProfile> {
        self.profiles.get(rule_id)
    }

    /// Return all rule profiles.
    pub fn all_profiles(&self) -> Vec<&RuleProfile> {
        self.profiles.values().collect()
    }

    // ── Statistics ──────────────────────────────────────────────────────────

    /// Compute summary statistics for the entire trace.
    pub fn statistics(&self) -> TraceStatistics {
        let total_firings = self.entries.len() as u64;
        let distinct_rules = self.profiles.len();
        let total_facts_derived: u64 = self.entries.iter().map(|e| e.produced.len() as u64).sum();
        let max_depth = self.entries.iter().map(|e| e.depth).max().unwrap_or(0);

        let most_fired_rule = self
            .profiles
            .values()
            .max_by_key(|p| p.fire_count)
            .map(|p| (p.rule_id.clone(), p.fire_count));

        // Longest derivation chain: for each produced fact, compute chain length.
        let mut longest_chain = 0_usize;
        for entry in &self.entries {
            for fact in &entry.produced {
                if let Some(chain) = self.derivation_chain(fact) {
                    if chain.steps.len() > longest_chain {
                        longest_chain = chain.steps.len();
                    }
                }
            }
        }

        let time_span_nanos = if self.entries.is_empty() {
            0
        } else {
            let min_t = self
                .entries
                .iter()
                .map(|e| e.timestamp.0)
                .min()
                .unwrap_or(0);
            let max_t = self
                .entries
                .iter()
                .map(|e| e.timestamp.0 + e.duration_nanos)
                .max()
                .unwrap_or(0);
            max_t.saturating_sub(min_t)
        };

        TraceStatistics {
            total_firings,
            distinct_rules,
            total_facts_derived,
            max_depth,
            most_fired_rule,
            longest_chain,
            time_span_nanos,
        }
    }

    // ── Export ───────────────────────────────────────────────────────────────

    /// Export the trace to a human-readable text format.
    pub fn export_text(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Rule Execution Trace ===\n\n");

        for entry in &self.entries {
            let _ = std::fmt::Write::write_fmt(
                &mut out,
                format_args!(
                    "[{seq:>4}] rule={rule} t={ts} d={dur}ns depth={depth}",
                    seq = entry.seq,
                    rule = entry.rule_id,
                    ts = entry.timestamp,
                    dur = entry.duration_nanos,
                    depth = entry.depth,
                ),
            );
            if let Some(parent) = entry.parent_seq {
                let _ = std::fmt::Write::write_fmt(&mut out, format_args!(" parent={parent}"));
            }
            out.push('\n');
            for c in &entry.consumed {
                let _ = std::fmt::Write::write_fmt(&mut out, format_args!("  IN:  {c}\n"));
            }
            for p in &entry.produced {
                let _ = std::fmt::Write::write_fmt(&mut out, format_args!("  OUT: {p}\n"));
            }
            out.push('\n');
        }

        // Append profile summary.
        out.push_str("=== Performance Summary ===\n\n");
        let mut profiles: Vec<_> = self.profiles.values().collect();
        profiles.sort_by(|a, b| b.fire_count.cmp(&a.fire_count));
        for p in &profiles {
            let _ = std::fmt::Write::write_fmt(
                &mut out,
                format_args!(
                    "{rule}: fired={cnt} total={total}ns avg={avg}ns produced={prod}\n",
                    rule = p.rule_id,
                    cnt = p.fire_count,
                    total = p.total_nanos,
                    avg = p.avg_nanos(),
                    prod = p.total_produced,
                ),
            );
        }

        out
    }

    /// Export only filtered entries to text.
    pub fn export_filtered_text(&self, filter: &TraceFilter) -> String {
        let filtered = self.filter(filter);
        let mut out = String::new();
        out.push_str("=== Filtered Rule Trace ===\n\n");
        for entry in &filtered {
            let _ = std::fmt::Write::write_fmt(
                &mut out,
                format_args!(
                    "[{seq:>4}] rule={rule} t={ts} depth={depth}\n",
                    seq = entry.seq,
                    rule = entry.rule_id,
                    ts = entry.timestamp,
                    depth = entry.depth,
                ),
            );
            for p in &entry.produced {
                let _ = std::fmt::Write::write_fmt(&mut out, format_args!("  OUT: {p}\n"));
            }
        }
        out
    }
}

impl Default for RuleTracer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fact(s: &str, p: &str, o: &str) -> DerivedFact {
        DerivedFact::new(s, p, o)
    }

    fn ts(nanos: u64) -> TraceTimestamp {
        TraceTimestamp::new(nanos)
    }

    // ── TraceTimestamp ───────────────────────────────────────────────────────

    #[test]
    fn test_timestamp_units() {
        let t = ts(5_000_000);
        assert_eq!(t.as_nanos(), 5_000_000);
        assert_eq!(t.as_micros(), 5_000);
        assert_eq!(t.as_millis(), 5);
    }

    #[test]
    fn test_timestamp_display_ms() {
        let t = ts(3_000_000);
        assert_eq!(t.to_string(), "3ms");
    }

    #[test]
    fn test_timestamp_display_us() {
        let t = ts(500_000);
        assert_eq!(t.to_string(), "500us");
    }

    #[test]
    fn test_timestamp_display_ns() {
        let t = ts(123);
        assert_eq!(t.to_string(), "123ns");
    }

    #[test]
    fn test_timestamp_ordering() {
        assert!(ts(1) < ts(2));
        assert!(ts(100) > ts(50));
        assert_eq!(ts(42), ts(42));
    }

    // ── DerivedFact ─────────────────────────────────────────────────────────

    #[test]
    fn test_derived_fact_new() {
        let f = fact("alice", "parent", "bob");
        assert_eq!(f.subject, "alice");
        assert_eq!(f.predicate, "parent");
        assert_eq!(f.object, "bob");
    }

    #[test]
    fn test_derived_fact_display() {
        let f = fact("alice", "parent", "bob");
        assert_eq!(f.to_string(), "(alice parent bob)");
    }

    #[test]
    fn test_derived_fact_equality() {
        let a = fact("a", "b", "c");
        let b = fact("a", "b", "c");
        let c = fact("a", "b", "d");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ── RuleTracer: basic recording ─────────────────────────────────────────

    #[test]
    fn test_tracer_new_is_empty() {
        let tracer = RuleTracer::new();
        assert!(tracer.is_empty());
        assert_eq!(tracer.len(), 0);
    }

    #[test]
    fn test_record_single_firing() {
        let mut tracer = RuleTracer::new();
        let consumed = vec![fact("alice", "parent", "bob")];
        let produced = vec![fact("alice", "ancestor", "bob")];
        let seq = tracer.record_firing("r1", ts(100), 50, consumed, produced, 0, None);
        assert_eq!(seq, Some(0));
        assert_eq!(tracer.len(), 1);
    }

    #[test]
    fn test_record_multiple_firings() {
        let mut tracer = RuleTracer::new();
        let s1 = tracer.record_firing(
            "r1",
            ts(100),
            50,
            vec![],
            vec![fact("a", "b", "c")],
            0,
            None,
        );
        let s2 = tracer.record_firing(
            "r2",
            ts(200),
            30,
            vec![],
            vec![fact("d", "e", "f")],
            0,
            None,
        );
        assert_eq!(s1, Some(0));
        assert_eq!(s2, Some(1));
        assert_eq!(tracer.len(), 2);
    }

    #[test]
    fn test_record_with_parent() {
        let mut tracer = RuleTracer::new();
        let s1 = tracer.record_firing(
            "r1",
            ts(100),
            50,
            vec![],
            vec![fact("a", "b", "c")],
            0,
            None,
        );
        let s2 = tracer.record_firing(
            "r2",
            ts(200),
            30,
            vec![fact("a", "b", "c")],
            vec![fact("x", "y", "z")],
            1,
            s1,
        );
        assert_eq!(s2, Some(1));
        let entry = tracer.get_entry(1).expect("entry exists");
        assert_eq!(entry.parent_seq, Some(0));
    }

    // ── Depth limiting ──────────────────────────────────────────────────────

    #[test]
    fn test_depth_limit_records_within_limit() {
        let mut tracer = RuleTracer::new().with_max_depth(2);
        let s = tracer.record_firing("r1", ts(100), 10, vec![], vec![], 2, None);
        assert!(s.is_some());
    }

    #[test]
    fn test_depth_limit_discards_beyond_limit() {
        let mut tracer = RuleTracer::new().with_max_depth(2);
        let s = tracer.record_firing("r1", ts(100), 10, vec![], vec![], 3, None);
        assert!(s.is_none());
        assert!(tracer.is_empty());
    }

    // ── Filtering ───────────────────────────────────────────────────────────

    #[test]
    fn test_filter_by_rule_id() {
        let mut tracer = RuleTracer::new();
        tracer.record_firing(
            "r1",
            ts(100),
            10,
            vec![],
            vec![fact("a", "b", "c")],
            0,
            None,
        );
        tracer.record_firing(
            "r2",
            ts(200),
            10,
            vec![],
            vec![fact("d", "e", "f")],
            0,
            None,
        );
        let filter = TraceFilter::new().with_rule_id("r1");
        let filtered = tracer.filter(&filter);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].rule_id, "r1");
    }

    #[test]
    fn test_filter_by_predicate() {
        let mut tracer = RuleTracer::new();
        tracer.record_firing(
            "r1",
            ts(100),
            10,
            vec![],
            vec![fact("a", "parent", "b")],
            0,
            None,
        );
        tracer.record_firing(
            "r2",
            ts(200),
            10,
            vec![],
            vec![fact("c", "ancestor", "d")],
            0,
            None,
        );
        let filter = TraceFilter::new().with_predicate("parent");
        let filtered = tracer.filter(&filter);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_filter_by_time_range() {
        let mut tracer = RuleTracer::new();
        tracer.record_firing("r1", ts(100), 10, vec![], vec![], 0, None);
        tracer.record_firing("r2", ts(500), 10, vec![], vec![], 0, None);
        tracer.record_firing("r3", ts(900), 10, vec![], vec![], 0, None);
        let filter = TraceFilter::new()
            .with_from_time(ts(200))
            .with_to_time(ts(600));
        let filtered = tracer.filter(&filter);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].rule_id, "r2");
    }

    #[test]
    fn test_filter_by_max_depth() {
        let mut tracer = RuleTracer::new();
        tracer.record_firing("r1", ts(100), 10, vec![], vec![], 0, None);
        tracer.record_firing("r2", ts(200), 10, vec![], vec![], 1, None);
        tracer.record_firing("r3", ts(300), 10, vec![], vec![], 2, None);
        let filter = TraceFilter::new().with_max_depth(1);
        let filtered = tracer.filter(&filter);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_filter_combined() {
        let mut tracer = RuleTracer::new();
        tracer.record_firing(
            "r1",
            ts(100),
            10,
            vec![],
            vec![fact("a", "parent", "b")],
            0,
            None,
        );
        tracer.record_firing(
            "r1",
            ts(500),
            10,
            vec![],
            vec![fact("c", "ancestor", "d")],
            1,
            None,
        );
        tracer.record_firing(
            "r2",
            ts(300),
            10,
            vec![],
            vec![fact("e", "parent", "f")],
            0,
            None,
        );
        let filter = TraceFilter::new()
            .with_rule_id("r1")
            .with_predicate("parent");
        let filtered = tracer.filter(&filter);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].timestamp, ts(100));
    }

    #[test]
    fn test_filter_empty_trace() {
        let tracer = RuleTracer::new();
        let filter = TraceFilter::new().with_rule_id("anything");
        let filtered = tracer.filter(&filter);
        assert!(filtered.is_empty());
    }

    // ── Profiling ───────────────────────────────────────────────────────────

    #[test]
    fn test_profile_single_rule() {
        let mut tracer = RuleTracer::new();
        tracer.record_firing(
            "r1",
            ts(100),
            50,
            vec![],
            vec![fact("a", "b", "c")],
            0,
            None,
        );
        tracer.record_firing(
            "r1",
            ts(200),
            30,
            vec![],
            vec![fact("d", "e", "f"), fact("g", "h", "i")],
            0,
            None,
        );
        let profile = tracer.profile("r1").expect("profile exists");
        assert_eq!(profile.fire_count, 2);
        assert_eq!(profile.total_nanos, 80);
        assert_eq!(profile.avg_nanos(), 40);
        assert_eq!(profile.min_nanos, 30);
        assert_eq!(profile.max_nanos, 50);
        assert_eq!(profile.total_produced, 3);
    }

    #[test]
    fn test_profile_nonexistent_rule() {
        let tracer = RuleTracer::new();
        assert!(tracer.profile("nonexistent").is_none());
    }

    #[test]
    fn test_all_profiles() {
        let mut tracer = RuleTracer::new();
        tracer.record_firing("r1", ts(100), 10, vec![], vec![], 0, None);
        tracer.record_firing("r2", ts(200), 20, vec![], vec![], 0, None);
        let profiles = tracer.all_profiles();
        assert_eq!(profiles.len(), 2);
    }

    #[test]
    fn test_profile_avg_zero_firings() {
        let profile = RuleProfile::new("empty".into());
        assert_eq!(profile.avg_nanos(), 0);
    }

    // ── Derivation chains ───────────────────────────────────────────────────

    #[test]
    fn test_derivation_chain_single_step() {
        let mut tracer = RuleTracer::new();
        let consumed = vec![fact("alice", "parent", "bob")];
        let produced = vec![fact("alice", "ancestor", "bob")];
        tracer.record_firing("r1", ts(100), 50, consumed, produced.clone(), 0, None);
        let chain = tracer.derivation_chain(&produced[0]);
        assert!(chain.is_some());
        let c = chain.expect("chain exists");
        assert_eq!(c.steps.len(), 1);
        assert_eq!(c.steps[0].rule_id, "r1");
    }

    #[test]
    fn test_derivation_chain_multi_step() {
        let mut tracer = RuleTracer::new();
        let s1 = tracer.record_firing(
            "r1",
            ts(100),
            50,
            vec![fact("alice", "parent", "bob")],
            vec![fact("alice", "ancestor", "bob")],
            0,
            None,
        );
        tracer.record_firing(
            "r2",
            ts(200),
            30,
            vec![fact("alice", "ancestor", "bob")],
            vec![fact("alice", "grandparent-line", "bob")],
            1,
            s1,
        );
        let chain = tracer.derivation_chain(&fact("alice", "grandparent-line", "bob"));
        assert!(chain.is_some());
        let c = chain.expect("chain exists");
        assert!(c.steps.len() >= 2);
    }

    #[test]
    fn test_derivation_chain_not_found() {
        let tracer = RuleTracer::new();
        let chain = tracer.derivation_chain(&fact("x", "y", "z"));
        assert!(chain.is_none());
    }

    // ── Statistics ──────────────────────────────────────────────────────────

    #[test]
    fn test_statistics_empty() {
        let tracer = RuleTracer::new();
        let stats = tracer.statistics();
        assert_eq!(stats.total_firings, 0);
        assert_eq!(stats.distinct_rules, 0);
        assert_eq!(stats.total_facts_derived, 0);
        assert_eq!(stats.max_depth, 0);
        assert!(stats.most_fired_rule.is_none());
        assert_eq!(stats.longest_chain, 0);
    }

    #[test]
    fn test_statistics_populated() {
        let mut tracer = RuleTracer::new();
        tracer.record_firing(
            "r1",
            ts(100),
            50,
            vec![],
            vec![fact("a", "b", "c")],
            0,
            None,
        );
        tracer.record_firing(
            "r1",
            ts(200),
            30,
            vec![],
            vec![fact("d", "e", "f")],
            0,
            None,
        );
        tracer.record_firing(
            "r2",
            ts(300),
            20,
            vec![],
            vec![fact("g", "h", "i")],
            1,
            None,
        );
        let stats = tracer.statistics();
        assert_eq!(stats.total_firings, 3);
        assert_eq!(stats.distinct_rules, 2);
        assert_eq!(stats.total_facts_derived, 3);
        assert_eq!(stats.max_depth, 1);
        let (most, count) = stats.most_fired_rule.expect("has most fired");
        assert_eq!(most, "r1");
        assert_eq!(count, 2);
    }

    #[test]
    fn test_statistics_time_span() {
        let mut tracer = RuleTracer::new();
        tracer.record_firing("r1", ts(100), 50, vec![], vec![], 0, None);
        tracer.record_firing("r2", ts(500), 100, vec![], vec![], 0, None);
        let stats = tracer.statistics();
        // time_span = max(500+100) - min(100) = 500
        assert_eq!(stats.time_span_nanos, 500);
    }

    // ── Export ───────────────────────────────────────────────────────────────

    #[test]
    fn test_export_text_empty() {
        let tracer = RuleTracer::new();
        let text = tracer.export_text();
        assert!(text.contains("Rule Execution Trace"));
        assert!(text.contains("Performance Summary"));
    }

    #[test]
    fn test_export_text_with_entries() {
        let mut tracer = RuleTracer::new();
        tracer.record_firing(
            "r1",
            ts(1_000_000),
            500,
            vec![],
            vec![fact("a", "b", "c")],
            0,
            None,
        );
        let text = tracer.export_text();
        assert!(text.contains("r1"));
        assert!(text.contains("OUT: (a b c)"));
    }

    #[test]
    fn test_export_filtered_text() {
        let mut tracer = RuleTracer::new();
        tracer.record_firing(
            "r1",
            ts(100),
            10,
            vec![],
            vec![fact("a", "b", "c")],
            0,
            None,
        );
        tracer.record_firing(
            "r2",
            ts(200),
            10,
            vec![],
            vec![fact("d", "e", "f")],
            0,
            None,
        );
        let filter = TraceFilter::new().with_rule_id("r2");
        let text = tracer.export_filtered_text(&filter);
        assert!(text.contains("r2"));
        assert!(!text.contains("[   0]"));
    }

    // ── Clear ───────────────────────────────────────────────────────────────

    #[test]
    fn test_clear() {
        let mut tracer = RuleTracer::new();
        tracer.record_firing(
            "r1",
            ts(100),
            10,
            vec![],
            vec![fact("a", "b", "c")],
            0,
            None,
        );
        assert!(!tracer.is_empty());
        tracer.clear();
        assert!(tracer.is_empty());
        assert!(tracer.all_profiles().is_empty());
    }

    // ── get_entry ───────────────────────────────────────────────────────────

    #[test]
    fn test_get_entry_found() {
        let mut tracer = RuleTracer::new();
        tracer.record_firing("r1", ts(100), 10, vec![], vec![], 0, None);
        assert!(tracer.get_entry(0).is_some());
    }

    #[test]
    fn test_get_entry_not_found() {
        let tracer = RuleTracer::new();
        assert!(tracer.get_entry(99).is_none());
    }

    // ── Default trait ───────────────────────────────────────────────────────

    #[test]
    fn test_default_tracer() {
        let tracer = RuleTracer::default();
        assert!(tracer.is_empty());
    }

    // ── TraceFilter builder ─────────────────────────────────────────────────

    #[test]
    fn test_trace_filter_default_passes_all() {
        let filter = TraceFilter::default();
        let entry = TraceEntry {
            seq: 0,
            rule_id: "r1".into(),
            timestamp: ts(100),
            duration_nanos: 10,
            consumed: vec![],
            produced: vec![],
            depth: 0,
            parent_seq: None,
        };
        assert!(filter.matches(&entry));
    }

    #[test]
    fn test_trace_filter_predicate_no_produced() {
        let filter = TraceFilter::new().with_predicate("parent");
        let entry = TraceEntry {
            seq: 0,
            rule_id: "r1".into(),
            timestamp: ts(100),
            duration_nanos: 10,
            consumed: vec![],
            produced: vec![], // no produced facts
            depth: 0,
            parent_seq: None,
        };
        assert!(!filter.matches(&entry));
    }

    // ── RuleProfile ─────────────────────────────────────────────────────────

    #[test]
    fn test_rule_profile_min_max() {
        let mut tracer = RuleTracer::new();
        tracer.record_firing("r1", ts(100), 10, vec![], vec![], 0, None);
        tracer.record_firing("r1", ts(200), 100, vec![], vec![], 0, None);
        tracer.record_firing("r1", ts(300), 50, vec![], vec![], 0, None);
        let p = tracer.profile("r1").expect("profile exists");
        assert_eq!(p.min_nanos, 10);
        assert_eq!(p.max_nanos, 100);
    }

    // ── Multiple produced facts ─────────────────────────────────────────────

    #[test]
    fn test_multiple_produced_facts_indexed() {
        let mut tracer = RuleTracer::new();
        let produced = vec![fact("a", "type", "Person"), fact("a", "name", "Alice")];
        tracer.record_firing("r1", ts(100), 10, vec![], produced, 0, None);
        // Both facts should be findable.
        assert!(tracer
            .derivation_chain(&fact("a", "type", "Person"))
            .is_some());
        assert!(tracer
            .derivation_chain(&fact("a", "name", "Alice"))
            .is_some());
    }

    // ── Entry data integrity ────────────────────────────────────────────────

    #[test]
    fn test_entry_data_integrity() {
        let mut tracer = RuleTracer::new();
        let consumed = vec![fact("x", "y", "z")];
        let produced = vec![fact("a", "b", "c")];
        tracer.record_firing(
            "test_rule",
            ts(42),
            99,
            consumed.clone(),
            produced.clone(),
            3,
            Some(10),
        );
        let entry = tracer.get_entry(0).expect("entry exists");
        assert_eq!(entry.rule_id, "test_rule");
        assert_eq!(entry.timestamp, ts(42));
        assert_eq!(entry.duration_nanos, 99);
        assert_eq!(entry.consumed, consumed);
        assert_eq!(entry.produced, produced);
        assert_eq!(entry.depth, 3);
        assert_eq!(entry.parent_seq, Some(10));
    }
}
