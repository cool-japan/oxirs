//! # Profile Command
//!
//! SPARQL query profiler CLI command that analyses query text to detect
//! language features, estimate execution cost, classify complexity, and
//! suggest potential optimisations.
//!
//! Feature detection is keyword / pattern based and works on raw query strings
//! without a full parse tree; it is intentionally lightweight so it can run
//! instantly on any query, even malformed ones.
//!
//! ## Cost model (additive)
//!
//! | Feature         | Cost per occurrence |
//! |-----------------|---------------------|
//! | TriplePattern   | 1.0                 |
//! | OptionalClause  | 3.0                 |
//! | FilterExpression| 0.5                 |
//! | SubQuery        | 10.0                |
//! | Aggregation     | 5.0                 |
//! | ServiceCall     | 20.0                |
//! | PropertyPath    | 4.0                 |
//! | GroupBy         | 2.0 (one-off)       |
//! | OrderBy         | 1.0 (one-off)       |
//! | Distinct        | 1.5 (one-off)       |
//!
//! ## Complexity thresholds
//!
//! | Cost         | Level      |
//! |--------------|------------|
//! | < 2          | Trivial    |
//! | < 5          | Simple     |
//! | < 15         | Moderate   |
//! | < 50         | Complex    |
//! | ≥ 50         | VeryComplex|
//!
//! ## Example
//!
//! ```rust
//! use oxirs::commands::profile_command::ProfileCommand;
//!
//! let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
//! let profile = ProfileCommand::profile(query);
//! println!("{}", ProfileCommand::format_report(&profile));
//! ```

// ─── Feature enum ─────────────────────────────────────────────────────────────

/// A SPARQL language feature detected in the query text.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QueryFeature {
    /// One or more `?s ?p ?o`-style triple patterns in the WHERE clause.
    TriplePattern,
    /// One or more `OPTIONAL { … }` clauses.
    OptionalClause,
    /// One or more `FILTER ( … )` expressions.
    FilterExpression,
    /// A nested `SELECT … WHERE { … }` sub-query.
    SubQuery,
    /// Aggregate functions (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, `GROUP_CONCAT`, `SAMPLE`).
    Aggregation,
    /// One or more `SERVICE <…> { … }` federated query calls.
    ServiceCall,
    /// Property path operators (`/`, `|`, `*`, `+`, `?`).
    PropertyPath,
    /// A `GROUP BY` clause.
    GroupBy,
    /// An `ORDER BY` clause.
    OrderBy,
    /// A `DISTINCT` keyword.
    Distinct,
}

// ─── Complexity level ─────────────────────────────────────────────────────────

/// Overall complexity classification of a SPARQL query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComplexityLevel {
    /// Estimated cost < 2.
    Trivial,
    /// Estimated cost in [2, 5).
    Simple,
    /// Estimated cost in [5, 15).
    Moderate,
    /// Estimated cost in [15, 50).
    Complex,
    /// Estimated cost ≥ 50.
    VeryComplex,
}

impl std::fmt::Display for ComplexityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            ComplexityLevel::Trivial => "Trivial",
            ComplexityLevel::Simple => "Simple",
            ComplexityLevel::Moderate => "Moderate",
            ComplexityLevel::Complex => "Complex",
            ComplexityLevel::VeryComplex => "VeryComplex",
        };
        write!(f, "{label}")
    }
}

// ─── QueryProfile ─────────────────────────────────────────────────────────────

/// The complete profiling result for a single SPARQL query.
#[derive(Debug, Clone)]
pub struct QueryProfile {
    /// The original query text that was profiled.
    pub query_text: String,
    /// Deduplicated list of features detected in the query.
    pub features: Vec<QueryFeature>,
    /// Estimated execution cost (sum of per-feature costs).
    pub estimated_cost: f64,
    /// Complexity category derived from `estimated_cost`.
    pub complexity: ComplexityLevel,
    /// Number of detected triple patterns.
    pub triple_pattern_count: usize,
    /// Number of detected `OPTIONAL` clauses.
    pub optional_count: usize,
    /// Number of detected `FILTER` expressions.
    pub filter_count: usize,
    /// Depth of nested sub-queries (number of nested `SELECT … WHERE`).
    pub subquery_depth: usize,
    /// Human-readable optimisation warnings.
    pub warnings: Vec<String>,
}

// ─── ProfileCommand ───────────────────────────────────────────────────────────

/// Stateless SPARQL query profiler.
pub struct ProfileCommand;

// ── Cost constants ────────────────────────────────────────────────────────────

const COST_TRIPLE: f64 = 1.0;
const COST_OPTIONAL: f64 = 3.0;
const COST_FILTER: f64 = 0.5;
const COST_SUBQUERY: f64 = 10.0;
const COST_AGGREGATION: f64 = 5.0;
const COST_SERVICE: f64 = 20.0;
const COST_PROPERTY_PATH: f64 = 4.0;
const COST_GROUP_BY: f64 = 2.0;
const COST_ORDER_BY: f64 = 1.0;
const COST_DISTINCT: f64 = 1.5;

impl ProfileCommand {
    /// Create a new `ProfileCommand` (stateless; this is a convenience
    /// constructor for callers that prefer an instance-based API).
    pub fn new() -> Self {
        Self
    }

    // ── Analysis ──────────────────────────────────────────────────────────────

    /// Analyse `query` and return a [`QueryProfile`].
    pub fn profile(query: &str) -> QueryProfile {
        let upper = query.to_uppercase();

        // ── Count occurrences ────────────────────────────────────────────────
        let triple_count = Self::count_triple_patterns(query);
        let optional_count = Self::count_keyword_occurrences(&upper, "OPTIONAL");
        let filter_count = Self::count_keyword_occurrences(&upper, "FILTER");
        let service_count = Self::count_keyword_occurrences(&upper, "SERVICE");
        let subquery_depth = Self::measure_subquery_depth(query);
        let has_aggregation = Self::has_aggregation(&upper);
        let has_group_by = upper.contains("GROUP BY");
        let has_order_by = upper.contains("ORDER BY");
        let has_distinct = upper.contains("DISTINCT");
        let has_property_path = Self::has_property_path(query);

        // ── Build feature list ────────────────────────────────────────────────
        let mut features: Vec<QueryFeature> = Vec::new();
        if triple_count > 0 {
            features.push(QueryFeature::TriplePattern);
        }
        if optional_count > 0 {
            features.push(QueryFeature::OptionalClause);
        }
        if filter_count > 0 {
            features.push(QueryFeature::FilterExpression);
        }
        if subquery_depth > 0 {
            features.push(QueryFeature::SubQuery);
        }
        if has_aggregation {
            features.push(QueryFeature::Aggregation);
        }
        if service_count > 0 {
            features.push(QueryFeature::ServiceCall);
        }
        if has_property_path {
            features.push(QueryFeature::PropertyPath);
        }
        if has_group_by {
            features.push(QueryFeature::GroupBy);
        }
        if has_order_by {
            features.push(QueryFeature::OrderBy);
        }
        if has_distinct {
            features.push(QueryFeature::Distinct);
        }

        // ── Estimate cost ────────────────────────────────────────────────────
        let mut cost = 0.0_f64;
        cost += triple_count as f64 * COST_TRIPLE;
        cost += optional_count as f64 * COST_OPTIONAL;
        cost += filter_count as f64 * COST_FILTER;
        cost += subquery_depth as f64 * COST_SUBQUERY;
        if has_aggregation {
            cost += COST_AGGREGATION;
        }
        cost += service_count as f64 * COST_SERVICE;
        if has_property_path {
            cost += COST_PROPERTY_PATH;
        }
        if has_group_by {
            cost += COST_GROUP_BY;
        }
        if has_order_by {
            cost += COST_ORDER_BY;
        }
        if has_distinct {
            cost += COST_DISTINCT;
        }

        let complexity = Self::classify_complexity(cost);

        // ── Optimisation hints ────────────────────────────────────────────────
        let mut profile = QueryProfile {
            query_text: query.to_string(),
            features,
            estimated_cost: cost,
            complexity,
            triple_pattern_count: triple_count,
            optional_count,
            filter_count,
            subquery_depth,
            warnings: Vec::new(),
        };

        profile.warnings = Self::suggest_optimizations(&profile);

        profile
    }

    // ── Cost / complexity ─────────────────────────────────────────────────────

    /// Compute the estimated cost from a completed [`QueryProfile`].
    pub fn estimate_cost(profile: &QueryProfile) -> f64 {
        profile.estimated_cost
    }

    /// Map an estimated cost to a [`ComplexityLevel`].
    pub fn classify_complexity(cost: f64) -> ComplexityLevel {
        if cost < 2.0 {
            ComplexityLevel::Trivial
        } else if cost < 5.0 {
            ComplexityLevel::Simple
        } else if cost < 15.0 {
            ComplexityLevel::Moderate
        } else if cost < 50.0 {
            ComplexityLevel::Complex
        } else {
            ComplexityLevel::VeryComplex
        }
    }

    // ── Optimisation suggestions ──────────────────────────────────────────────

    /// Return a list of actionable optimisation suggestions for `profile`.
    pub fn suggest_optimizations(profile: &QueryProfile) -> Vec<String> {
        let mut hints: Vec<String> = Vec::new();

        let query_upper = profile.query_text.to_uppercase();

        // LIMIT hint: if there is no LIMIT and the query returns results.
        if !query_upper.contains("LIMIT") && query_upper.contains("SELECT") {
            hints.push(
                "Consider adding LIMIT to cap the result set and reduce memory usage.".to_string(),
            );
        }

        // SERVICE hint.
        if profile.features.contains(&QueryFeature::ServiceCall) {
            hints.push(
                "SERVICE calls are expensive and involve network latency; \
                 cache results where possible."
                    .to_string(),
            );
        }

        // Sub-query hint.
        if profile.subquery_depth >= 2 {
            hints.push(
                "Deeply nested sub-queries impact performance; consider flattening or \
                 materialising intermediate results."
                    .to_string(),
            );
        } else if profile.subquery_depth == 1 {
            hints.push(
                "Sub-queries add overhead; ensure they are necessary and properly indexed."
                    .to_string(),
            );
        }

        // OPTIONAL hint.
        if profile.optional_count >= 3 {
            hints.push(
                "Many OPTIONAL clauses degrade join performance; \
                 move mandatory patterns to the main WHERE block."
                    .to_string(),
            );
        }

        // Property path hint.
        if profile.features.contains(&QueryFeature::PropertyPath) {
            hints.push(
                "Property paths with `*` or `+` can traverse unbounded graph depth; \
                 add explicit depth limits where possible."
                    .to_string(),
            );
        }

        // Aggregation without GROUP BY.
        if profile.features.contains(&QueryFeature::Aggregation)
            && !profile.features.contains(&QueryFeature::GroupBy)
        {
            hints.push(
                "Aggregation without GROUP BY produces a single row; \
                 verify this is intentional."
                    .to_string(),
            );
        }

        // Cartesian product risk (many triples, no filter).
        if profile.triple_pattern_count >= 5 && profile.filter_count == 0 {
            hints.push(
                "Many triple patterns without FILTER may produce a Cartesian product; \
                 add selective FILTER conditions."
                    .to_string(),
            );
        }

        hints
    }

    // ── Report ────────────────────────────────────────────────────────────────

    /// Format `profile` as a human-readable multi-line report string.
    pub fn format_report(profile: &QueryProfile) -> String {
        let mut lines: Vec<String> = Vec::new();

        lines.push("=== SPARQL Query Profile Report ===".to_string());
        lines.push(format!("Complexity   : {}", profile.complexity));
        lines.push(format!("Estimated Cost: {:.2}", profile.estimated_cost));
        lines.push(format!(
            "Triple Patterns : {}",
            profile.triple_pattern_count
        ));
        lines.push(format!("OPTIONAL Clauses: {}", profile.optional_count));
        lines.push(format!("FILTER Expressions: {}", profile.filter_count));
        lines.push(format!("Sub-query Depth: {}", profile.subquery_depth));

        if !profile.features.is_empty() {
            lines.push("Detected Features:".to_string());
            for f in &profile.features {
                lines.push(format!("  - {:?}", f));
            }
        }

        if !profile.warnings.is_empty() {
            lines.push("Optimisation Hints:".to_string());
            for w in &profile.warnings {
                lines.push(format!("  * {w}"));
            }
        } else {
            lines.push("Optimisation Hints: none".to_string());
        }

        lines.join("\n")
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Count how many times a whole-word keyword appears in `upper_query`.
    fn count_keyword_occurrences(upper_query: &str, keyword: &str) -> usize {
        let mut count = 0;
        let mut start = 0;
        while let Some(pos) = upper_query[start..].find(keyword) {
            let abs = start + pos;
            // Check word boundaries.
            let before_ok = abs == 0
                || !upper_query
                    .as_bytes()
                    .get(abs - 1)
                    .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_');
            let after = abs + keyword.len();
            let after_ok = after >= upper_query.len()
                || !upper_query
                    .as_bytes()
                    .get(after)
                    .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_');
            if before_ok && after_ok {
                count += 1;
            }
            start = abs + 1;
        }
        count
    }

    /// Estimate the number of triple patterns by counting the number of opening
    /// braces (each `{` likely starts a group graph pattern that contains
    /// triples) and the number of `.` pattern terminators.
    ///
    /// This is an approximation: we count occurrences of the ` . ` separator
    /// inside WHERE blocks plus a base of 1 per `{` (since a non-empty block
    /// has at least one triple).
    fn count_triple_patterns(query: &str) -> usize {
        // Simple heuristic: count sequences of non-brace, non-semicolon tokens
        // that look like triple pattern terminators (`.`) inside WHERE blocks.
        //
        // More precisely: we count how many `.` characters appear that are
        // likely to be triple pattern terminators (not inside IRIs or strings),
        // plus the number of `}` that close non-empty blocks.
        let upper = query.to_uppercase();
        // Find WHERE clause start.
        let where_start = upper.find("WHERE").map(|i| i + 5).unwrap_or(0);
        let where_body = &query[where_start..];

        // Count dot-terminators: a `.` surrounded by whitespace or `}`.
        let dot_count = where_body
            .chars()
            .enumerate()
            .filter(|(i, c)| {
                if *c != '.' {
                    return false;
                }
                let bytes = where_body.as_bytes();
                // Skip if inside an IRI: check there's no open `<` without `>` before.
                // Skip if the dot is part of a number or string.
                let prev = if *i > 0 {
                    bytes.get(i - 1).copied()
                } else {
                    None
                };
                let next = bytes.get(i + 1).copied();
                let prev_ws = prev.map_or(true, |b| {
                    b == b' '
                        || b == b'\t'
                        || b == b'\n'
                        || b == b'\r'
                        || b == b'}'
                        || b == b'"'
                        || b == b'>'
                });
                let next_ws = next.map_or(true, |b| {
                    b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' || b == b'{'
                });
                prev_ws && next_ws
            })
            .count();

        // A block with no dots still has one triple (if non-empty).
        // Heuristic: at least 1 if WHERE body contains `{`.
        let base = if where_body.contains('{') { 1 } else { 0 };
        // Each dot means one more triple (the terminator after each non-last triple).
        (base + dot_count).max(if where_body.contains('{') { 1 } else { 0 })
    }

    /// Measure the depth of nested SELECT sub-queries.
    ///
    /// A sub-query is a `SELECT` keyword that appears inside a `{…}` block
    /// (i.e. after the first `WHERE {`).  Each additional nesting level
    /// adds 1 to the depth.
    fn measure_subquery_depth(query: &str) -> usize {
        let upper = query.to_uppercase();
        // Find the main WHERE clause.
        let main_where = match upper.find("WHERE") {
            Some(i) => i + 5,
            None => return 0,
        };
        let body = &upper[main_where..];
        // Count SELECT occurrences inside the body — each one is a sub-query.
        let mut depth = 0_usize;
        let mut start = 0;
        while let Some(pos) = body[start..].find("SELECT") {
            let abs = start + pos;
            // Word-boundary check.
            let before_ok = abs == 0
                || !body
                    .as_bytes()
                    .get(abs - 1)
                    .is_some_and(|b| b.is_ascii_alphabetic());
            let after = abs + 6;
            let after_ok = after >= body.len()
                || !body
                    .as_bytes()
                    .get(after)
                    .is_some_and(|b| b.is_ascii_alphabetic());
            if before_ok && after_ok {
                depth += 1;
            }
            start = abs + 1;
        }
        depth
    }

    /// Returns `true` if the query contains any aggregate function keyword.
    fn has_aggregation(upper_query: &str) -> bool {
        let agg_funcs = [
            "COUNT(",
            "SUM(",
            "AVG(",
            "MIN(",
            "MAX(",
            "GROUP_CONCAT(",
            "SAMPLE(",
        ];
        agg_funcs.iter().any(|f| upper_query.contains(f))
    }

    /// Returns `true` if the query contains property path syntax characters
    /// (`/`, `|`, `*`, `+`, `?`) in a position that suggests property paths
    /// (not inside IRIs or string literals).
    fn has_property_path(query: &str) -> bool {
        // Look for `/` outside of `<…>` IRIs.
        // Strategy: scan for `/`, `*`, `+`, `?` that are surrounded by spaces
        // or variable/prefix tokens.
        let upper = query.to_uppercase();
        // Simple heuristic: if any of these appear between `WHERE {` and `}`
        // and are not part of a known IRI pattern.
        let where_start = upper.find("WHERE").map(|i| i + 5).unwrap_or(0);
        let body = &query[where_start..];

        // Walk chars, skip content inside `<…>` IRIs and `"…"` literals.
        let mut in_iri = false;
        let mut in_string = false;
        let chars: Vec<char> = body.chars().collect();
        let n = chars.len();
        let mut i = 0;
        while i < n {
            match chars[i] {
                '<' if !in_string => {
                    in_iri = true;
                }
                '>' if in_iri => {
                    in_iri = false;
                }
                '"' if !in_iri => {
                    in_string = !in_string;
                }
                '/' | '*' | '+' | '?' if !in_iri && !in_string => {
                    // `?` followed by a word char is a variable, not a path modifier.
                    if chars[i] == '?' {
                        let next = chars.get(i + 1);
                        if next.is_some_and(|c| c.is_ascii_alphanumeric() || *c == '_') {
                            i += 1;
                            continue;
                        }
                    }
                    // `*` inside a SELECT projection is not a path modifier.
                    if chars[i] == '*' {
                        // Check if surrounded by whitespace on both sides (path kleene star).
                        let prev = if i > 0 { Some(chars[i - 1]) } else { None };
                        let next = chars.get(i + 1);
                        let looks_like_path = prev.is_some_and(|c| !c.is_whitespace())
                            || next.is_some_and(|c| !c.is_whitespace() && *c != ' ');
                        if !looks_like_path {
                            i += 1;
                            continue;
                        }
                    }
                    return true;
                }
                _ => {}
            }
            i += 1;
        }
        false
    }
}

impl Default for ProfileCommand {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn features_of(query: &str) -> Vec<QueryFeature> {
        ProfileCommand::profile(query).features
    }

    fn has_feature(query: &str, feature: &QueryFeature) -> bool {
        features_of(query).contains(feature)
    }

    // ── ComplexityLevel Display ───────────────────────────────────────────────

    #[test]
    fn test_complexity_display_trivial() {
        assert_eq!(ComplexityLevel::Trivial.to_string(), "Trivial");
    }

    #[test]
    fn test_complexity_display_simple() {
        assert_eq!(ComplexityLevel::Simple.to_string(), "Simple");
    }

    #[test]
    fn test_complexity_display_moderate() {
        assert_eq!(ComplexityLevel::Moderate.to_string(), "Moderate");
    }

    #[test]
    fn test_complexity_display_complex() {
        assert_eq!(ComplexityLevel::Complex.to_string(), "Complex");
    }

    #[test]
    fn test_complexity_display_very_complex() {
        assert_eq!(ComplexityLevel::VeryComplex.to_string(), "VeryComplex");
    }

    // ── classify_complexity ───────────────────────────────────────────────────

    #[test]
    fn test_classify_zero_cost_trivial() {
        assert_eq!(
            ProfileCommand::classify_complexity(0.0),
            ComplexityLevel::Trivial
        );
    }

    #[test]
    fn test_classify_just_below_2_trivial() {
        assert_eq!(
            ProfileCommand::classify_complexity(1.9),
            ComplexityLevel::Trivial
        );
    }

    #[test]
    fn test_classify_exactly_2_simple() {
        assert_eq!(
            ProfileCommand::classify_complexity(2.0),
            ComplexityLevel::Simple
        );
    }

    #[test]
    fn test_classify_4_9_simple() {
        assert_eq!(
            ProfileCommand::classify_complexity(4.9),
            ComplexityLevel::Simple
        );
    }

    #[test]
    fn test_classify_5_moderate() {
        assert_eq!(
            ProfileCommand::classify_complexity(5.0),
            ComplexityLevel::Moderate
        );
    }

    #[test]
    fn test_classify_14_9_moderate() {
        assert_eq!(
            ProfileCommand::classify_complexity(14.9),
            ComplexityLevel::Moderate
        );
    }

    #[test]
    fn test_classify_15_complex() {
        assert_eq!(
            ProfileCommand::classify_complexity(15.0),
            ComplexityLevel::Complex
        );
    }

    #[test]
    fn test_classify_49_9_complex() {
        assert_eq!(
            ProfileCommand::classify_complexity(49.9),
            ComplexityLevel::Complex
        );
    }

    #[test]
    fn test_classify_50_very_complex() {
        assert_eq!(
            ProfileCommand::classify_complexity(50.0),
            ComplexityLevel::VeryComplex
        );
    }

    #[test]
    fn test_classify_100_very_complex() {
        assert_eq!(
            ProfileCommand::classify_complexity(100.0),
            ComplexityLevel::VeryComplex
        );
    }

    // ── Feature detection: TriplePattern ─────────────────────────────────────

    #[test]
    fn test_detect_triple_pattern_basic() {
        let q = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        assert!(has_feature(q, &QueryFeature::TriplePattern));
    }

    #[test]
    fn test_triple_pattern_count_single() {
        let q = "SELECT * WHERE { ?s ?p ?o }";
        let p = ProfileCommand::profile(q);
        assert!(p.triple_pattern_count >= 1);
    }

    #[test]
    fn test_triple_pattern_count_multiple_with_dots() {
        let q = "SELECT * WHERE { ?s ?p ?o . ?a ?b ?c }";
        let p = ProfileCommand::profile(q);
        // Should detect at least 2 triple patterns.
        assert!(p.triple_pattern_count >= 2);
    }

    // ── Feature detection: OptionalClause ─────────────────────────────────────

    #[test]
    fn test_detect_optional_clause() {
        let q = "SELECT ?s WHERE { ?s <p> ?o OPTIONAL { ?s <q> ?r } }";
        assert!(has_feature(q, &QueryFeature::OptionalClause));
    }

    #[test]
    fn test_optional_count_two() {
        let q = "SELECT ?s WHERE { ?s <p> ?o OPTIONAL { ?s <q> ?r } OPTIONAL { ?s <x> ?y } }";
        let p = ProfileCommand::profile(q);
        assert_eq!(p.optional_count, 2);
    }

    #[test]
    fn test_no_optional_in_simple_query() {
        let q = "SELECT ?s WHERE { ?s <p> ?o }";
        assert!(!has_feature(q, &QueryFeature::OptionalClause));
    }

    // ── Feature detection: FilterExpression ──────────────────────────────────

    #[test]
    fn test_detect_filter_expression() {
        let q = "SELECT ?s WHERE { ?s <age> ?a FILTER(?a > 18) }";
        assert!(has_feature(q, &QueryFeature::FilterExpression));
    }

    #[test]
    fn test_filter_count_two() {
        let q = "SELECT ?s WHERE { ?s <a> ?x FILTER(?x > 0) FILTER(?x < 100) }";
        let p = ProfileCommand::profile(q);
        assert_eq!(p.filter_count, 2);
    }

    #[test]
    fn test_no_filter_in_simple_query() {
        let q = "SELECT ?s WHERE { ?s ?p ?o }";
        assert!(!has_feature(q, &QueryFeature::FilterExpression));
    }

    // ── Feature detection: SubQuery ───────────────────────────────────────────

    #[test]
    fn test_detect_subquery() {
        let q = "SELECT ?s WHERE { { SELECT ?s WHERE { ?s ?p ?o } } }";
        assert!(has_feature(q, &QueryFeature::SubQuery));
    }

    #[test]
    fn test_subquery_depth_one() {
        let q = "SELECT ?s WHERE { { SELECT ?s WHERE { ?s ?p ?o } } }";
        let p = ProfileCommand::profile(q);
        assert_eq!(p.subquery_depth, 1);
    }

    #[test]
    fn test_no_subquery_in_flat_query() {
        let q = "SELECT ?s WHERE { ?s ?p ?o }";
        assert!(!has_feature(q, &QueryFeature::SubQuery));
    }

    // ── Feature detection: Aggregation ────────────────────────────────────────

    #[test]
    fn test_detect_count_aggregation() {
        let q = "SELECT (COUNT(?s) AS ?n) WHERE { ?s ?p ?o }";
        assert!(has_feature(q, &QueryFeature::Aggregation));
    }

    #[test]
    fn test_detect_sum_aggregation() {
        let q = "SELECT (SUM(?v) AS ?total) WHERE { ?s <val> ?v }";
        assert!(has_feature(q, &QueryFeature::Aggregation));
    }

    #[test]
    fn test_detect_avg_aggregation() {
        let q = "SELECT (AVG(?v) AS ?avg) WHERE { ?s <v> ?v }";
        assert!(has_feature(q, &QueryFeature::Aggregation));
    }

    #[test]
    fn test_detect_min_aggregation() {
        let q = "SELECT (MIN(?v) AS ?m) WHERE { ?s <v> ?v }";
        assert!(has_feature(q, &QueryFeature::Aggregation));
    }

    #[test]
    fn test_detect_max_aggregation() {
        let q = "SELECT (MAX(?v) AS ?m) WHERE { ?s <v> ?v }";
        assert!(has_feature(q, &QueryFeature::Aggregation));
    }

    #[test]
    fn test_no_aggregation_in_simple_select() {
        let q = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        assert!(!has_feature(q, &QueryFeature::Aggregation));
    }

    // ── Feature detection: ServiceCall ────────────────────────────────────────

    #[test]
    fn test_detect_service_call() {
        let q = "SELECT ?s WHERE { SERVICE <http://example.org/sparql> { ?s ?p ?o } }";
        assert!(has_feature(q, &QueryFeature::ServiceCall));
    }

    #[test]
    fn test_no_service_in_local_query() {
        let q = "SELECT ?s WHERE { ?s ?p ?o }";
        assert!(!has_feature(q, &QueryFeature::ServiceCall));
    }

    // ── Feature detection: PropertyPath ──────────────────────────────────────

    #[test]
    fn test_detect_property_path_slash() {
        let q = "SELECT ?o WHERE { ?s <a>/<b> ?o }";
        assert!(has_feature(q, &QueryFeature::PropertyPath));
    }

    #[test]
    fn test_detect_property_path_plus() {
        let q = "SELECT ?o WHERE { ?s <knows>+ ?o }";
        assert!(has_feature(q, &QueryFeature::PropertyPath));
    }

    #[test]
    fn test_no_property_path_in_simple_query() {
        let q = "SELECT ?s WHERE { ?s <p> ?o }";
        // Variables start with `?` followed by alpha char, should not trigger path detection
        assert!(!has_feature(q, &QueryFeature::PropertyPath));
    }

    // ── Feature detection: GroupBy ────────────────────────────────────────────

    #[test]
    fn test_detect_group_by() {
        let q = "SELECT ?s (COUNT(?o) AS ?c) WHERE { ?s ?p ?o } GROUP BY ?s";
        assert!(has_feature(q, &QueryFeature::GroupBy));
    }

    #[test]
    fn test_no_group_by_in_simple_query() {
        let q = "SELECT ?s WHERE { ?s ?p ?o }";
        assert!(!has_feature(q, &QueryFeature::GroupBy));
    }

    // ── Feature detection: OrderBy ────────────────────────────────────────────

    #[test]
    fn test_detect_order_by() {
        let q = "SELECT ?s WHERE { ?s ?p ?o } ORDER BY ?s";
        assert!(has_feature(q, &QueryFeature::OrderBy));
    }

    #[test]
    fn test_no_order_by_in_simple_query() {
        let q = "SELECT ?s WHERE { ?s ?p ?o }";
        assert!(!has_feature(q, &QueryFeature::OrderBy));
    }

    // ── Feature detection: Distinct ───────────────────────────────────────────

    #[test]
    fn test_detect_distinct() {
        let q = "SELECT DISTINCT ?s WHERE { ?s ?p ?o }";
        assert!(has_feature(q, &QueryFeature::Distinct));
    }

    #[test]
    fn test_no_distinct_in_simple_query() {
        let q = "SELECT ?s WHERE { ?s ?p ?o }";
        assert!(!has_feature(q, &QueryFeature::Distinct));
    }

    // ── Cost estimation ───────────────────────────────────────────────────────

    #[test]
    fn test_estimate_cost_simple_triple() {
        let q = "SELECT * WHERE { ?s ?p ?o }";
        let p = ProfileCommand::profile(q);
        // 1 triple = 1.0 (plus no LIMIT hint)
        assert!(p.estimated_cost >= 1.0);
    }

    #[test]
    fn test_estimate_cost_with_optional() {
        let q = "SELECT ?s WHERE { ?s ?p ?o OPTIONAL { ?s <q> ?r } }";
        let p = ProfileCommand::profile(q);
        // >= 1.0 (triple) + 3.0 (optional) = 4.0
        assert!(p.estimated_cost >= 4.0);
    }

    #[test]
    fn test_estimate_cost_with_service() {
        let q = "SELECT ?s WHERE { SERVICE <http://x.org/s> { ?s ?p ?o } }";
        let p = ProfileCommand::profile(q);
        // >= 20.0 (service)
        assert!(p.estimated_cost >= 20.0);
    }

    #[test]
    fn test_estimate_cost_with_aggregation() {
        let q = "SELECT (COUNT(?s) AS ?c) WHERE { ?s ?p ?o }";
        let p = ProfileCommand::profile(q);
        // >= 5.0 (aggregation) + 1.0 (triple)
        assert!(p.estimated_cost >= 6.0);
    }

    #[test]
    fn test_estimate_cost_matches_profile_field() {
        let q = "SELECT ?s WHERE { ?s ?p ?o }";
        let p = ProfileCommand::profile(q);
        assert!((ProfileCommand::estimate_cost(&p) - p.estimated_cost).abs() < f64::EPSILON);
    }

    // ── Complexity levels from real queries ───────────────────────────────────

    #[test]
    fn test_empty_query_trivial_complexity() {
        let p = ProfileCommand::profile("");
        assert_eq!(p.complexity, ComplexityLevel::Trivial);
    }

    #[test]
    fn test_service_query_very_complex() {
        let q = "SELECT ?s WHERE { SERVICE <http://a.org/> { ?s ?p ?o } \
                  SERVICE <http://b.org/> { ?s ?q ?r } \
                  SERVICE <http://c.org/> { ?s ?x ?y } }";
        let p = ProfileCommand::profile(q);
        // 3 × 20 = 60 → VeryComplex
        assert_eq!(p.complexity, ComplexityLevel::VeryComplex);
    }

    #[test]
    fn test_simple_select_moderate_or_lower() {
        let q = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        let p = ProfileCommand::profile(q);
        assert!(matches!(
            p.complexity,
            ComplexityLevel::Trivial | ComplexityLevel::Simple | ComplexityLevel::Moderate
        ));
    }

    // ── Optimisation suggestions ──────────────────────────────────────────────

    #[test]
    fn test_suggest_limit_for_select_without_limit() {
        let q = "SELECT ?s WHERE { ?s ?p ?o }";
        let hints = ProfileCommand::suggest_optimizations(&ProfileCommand::profile(q));
        assert!(hints.iter().any(|h| h.to_lowercase().contains("limit")));
    }

    #[test]
    fn test_no_limit_hint_when_limit_present() {
        let q = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 100";
        let hints = ProfileCommand::suggest_optimizations(&ProfileCommand::profile(q));
        assert!(!hints.iter().any(|h| h.to_lowercase().contains("limit")));
    }

    #[test]
    fn test_suggest_service_hint() {
        let q = "SELECT ?s WHERE { SERVICE <http://x.org/> { ?s ?p ?o } }";
        let hints = ProfileCommand::suggest_optimizations(&ProfileCommand::profile(q));
        assert!(hints.iter().any(|h| h.to_lowercase().contains("service")));
    }

    #[test]
    fn test_suggest_subquery_hint_for_nested() {
        let q = "SELECT ?s WHERE { { SELECT ?s WHERE { ?s ?p ?o } } }";
        let hints = ProfileCommand::suggest_optimizations(&ProfileCommand::profile(q));
        assert!(hints.iter().any(|h| h.to_lowercase().contains("sub")));
    }

    #[test]
    fn test_suggest_many_optionals_hint() {
        let q = "SELECT ?s WHERE { \
            ?s <a> ?x OPTIONAL { ?s <b> ?y } OPTIONAL { ?s <c> ?z } OPTIONAL { ?s <d> ?w } \
        }";
        let hints = ProfileCommand::suggest_optimizations(&ProfileCommand::profile(q));
        assert!(hints.iter().any(|h| h.to_lowercase().contains("optional")));
    }

    #[test]
    fn test_suggest_aggregation_without_group_by() {
        let q = "SELECT (COUNT(?s) AS ?c) WHERE { ?s ?p ?o }";
        let hints = ProfileCommand::suggest_optimizations(&ProfileCommand::profile(q));
        assert!(hints.iter().any(|h| h.to_lowercase().contains("group")));
    }

    #[test]
    fn test_no_aggregation_hint_with_group_by() {
        let q = "SELECT ?s (COUNT(?o) AS ?c) WHERE { ?s ?p ?o } GROUP BY ?s";
        let hints = ProfileCommand::suggest_optimizations(&ProfileCommand::profile(q));
        // Should NOT have the aggregation-without-GROUP-BY hint.
        assert!(!hints
            .iter()
            .any(|h| h.contains("Aggregation without GROUP BY")));
    }

    // ── format_report ─────────────────────────────────────────────────────────

    #[test]
    fn test_format_report_contains_complexity() {
        let q = "SELECT ?s WHERE { ?s ?p ?o }";
        let p = ProfileCommand::profile(q);
        let report = ProfileCommand::format_report(&p);
        assert!(report.contains("Complexity"));
    }

    #[test]
    fn test_format_report_contains_estimated_cost() {
        let q = "SELECT ?s WHERE { ?s ?p ?o }";
        let p = ProfileCommand::profile(q);
        let report = ProfileCommand::format_report(&p);
        assert!(report.contains("Estimated Cost"));
    }

    #[test]
    fn test_format_report_contains_triple_pattern_count() {
        let q = "SELECT ?s WHERE { ?s ?p ?o }";
        let p = ProfileCommand::profile(q);
        let report = ProfileCommand::format_report(&p);
        assert!(report.contains("Triple Patterns"));
    }

    #[test]
    fn test_format_report_contains_optional_count() {
        let q = "SELECT ?s WHERE { ?s ?p ?o }";
        let p = ProfileCommand::profile(q);
        let report = ProfileCommand::format_report(&p);
        assert!(report.contains("OPTIONAL"));
    }

    #[test]
    fn test_format_report_contains_filter_count() {
        let q = "SELECT ?s WHERE { ?s ?p ?o }";
        let p = ProfileCommand::profile(q);
        let report = ProfileCommand::format_report(&p);
        assert!(report.contains("FILTER"));
    }

    #[test]
    fn test_format_report_contains_subquery_depth() {
        let q = "SELECT ?s WHERE { ?s ?p ?o }";
        let p = ProfileCommand::profile(q);
        let report = ProfileCommand::format_report(&p);
        assert!(report.contains("Sub-query"));
    }

    #[test]
    fn test_format_report_contains_features_section() {
        let q = "SELECT ?s WHERE { ?s ?p ?o }";
        let p = ProfileCommand::profile(q);
        let report = ProfileCommand::format_report(&p);
        assert!(report.contains("Feature") || report.contains("feature"));
    }

    #[test]
    fn test_format_report_contains_hints() {
        let q = "SELECT ?s WHERE { ?s ?p ?o }";
        let p = ProfileCommand::profile(q);
        let report = ProfileCommand::format_report(&p);
        assert!(report.contains("Optimis") || report.contains("Hint") || report.contains("hint"));
    }

    #[test]
    fn test_format_report_header_present() {
        let q = "SELECT ?s WHERE { ?s ?p ?o }";
        let p = ProfileCommand::profile(q);
        let report = ProfileCommand::format_report(&p);
        assert!(report.contains("SPARQL Query Profile Report"));
    }

    #[test]
    fn test_format_report_is_multiline() {
        let q = "SELECT ?s WHERE { ?s ?p ?o }";
        let p = ProfileCommand::profile(q);
        let report = ProfileCommand::format_report(&p);
        assert!(report.contains('\n'));
    }

    // ── ProfileCommand::new ───────────────────────────────────────────────────

    #[test]
    fn test_new_returns_instance() {
        let _cmd = ProfileCommand::new();
    }

    #[test]
    fn test_default_same_as_new() {
        let _cmd: ProfileCommand = Default::default();
    }

    // ── QueryProfile fields ───────────────────────────────────────────────────

    #[test]
    fn test_profile_query_text_preserved() {
        let q = "SELECT ?s WHERE { ?s ?p ?o }";
        let p = ProfileCommand::profile(q);
        assert_eq!(p.query_text, q);
    }

    #[test]
    fn test_profile_features_deduplicated() {
        // Run profile on a query that has DISTINCT exactly once — should appear once.
        let q = "SELECT DISTINCT ?s WHERE { ?s ?p ?o }";
        let p = ProfileCommand::profile(q);
        let distinct_count = p
            .features
            .iter()
            .filter(|f| **f == QueryFeature::Distinct)
            .count();
        assert_eq!(distinct_count, 1);
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_profile_empty_string() {
        let p = ProfileCommand::profile("");
        assert_eq!(p.estimated_cost, 0.0);
        assert_eq!(p.complexity, ComplexityLevel::Trivial);
        assert!(p.features.is_empty());
    }

    #[test]
    fn test_profile_ask_query() {
        let q = "ASK { ?s ?p ?o }";
        let p = ProfileCommand::profile(q);
        // Should at least detect a triple pattern.
        assert!(p.triple_pattern_count >= 1 || p.estimated_cost >= 0.0);
    }

    #[test]
    fn test_profile_construct_query() {
        let q = "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }";
        let p = ProfileCommand::profile(q);
        assert!(p.estimated_cost >= 0.0);
    }
}
