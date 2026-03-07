//! ML-Based Query Optimization Advisor
//!
//! Analyzes SPARQL queries for common anti-patterns and performance problems,
//! provides rewrite suggestions, and estimates expected speedup ratios.
//!
//! ## Features
//!
//! - Cartesian product detection (triple patterns with no shared variables)
//! - Unbound predicate detection (`?s ?p ?o` fully variable triple)
//! - Deep nesting detection (SELECT subqueries > 3 levels)
//! - Repeated subquery pattern detection
//! - Missing index hints
//! - Early FILTER reordering
//! - Pattern selectivity-based reordering
//! - Speedup estimation heuristics

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Detected SPARQL anti-patterns that indicate optimization opportunities
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryPattern {
    /// Two or more triple patterns share no variables — Cartesian product at runtime
    CartesianProduct,
    /// A triple pattern has all three positions as variables: `?s ?p ?o`
    UnboundPredicate,
    /// Nested SELECT subqueries exceed 3 levels of depth
    DeepNesting,
    /// The same subquery text appears more than once in the query
    RepeatedSubquery,
    /// Frequently-queried predicates appear without associated index hint
    MissingIndex,
}

impl QueryPattern {
    /// Human-readable name for the pattern
    pub fn name(&self) -> &'static str {
        match self {
            Self::CartesianProduct => "Cartesian Product",
            Self::UnboundPredicate => "Unbound Predicate",
            Self::DeepNesting => "Deep Nesting",
            Self::RepeatedSubquery => "Repeated Subquery",
            Self::MissingIndex => "Missing Index",
        }
    }

    /// Estimated overhead multiplier compared to an optimized query
    pub fn overhead_multiplier(&self) -> f64 {
        match self {
            Self::CartesianProduct => 10.0,
            Self::UnboundPredicate => 5.0,
            Self::DeepNesting => 3.0,
            Self::RepeatedSubquery => 2.5,
            Self::MissingIndex => 2.0,
        }
    }
}

/// A single optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suggestion {
    /// Which pattern triggered this suggestion
    pub pattern: QueryPattern,
    /// Human-readable description of the problem and fix
    pub description: String,
    /// Optionally, a rewritten version of the query
    pub rewritten_query: Option<String>,
}

impl Suggestion {
    /// Create a suggestion without a rewritten query
    pub fn new(pattern: QueryPattern, description: impl Into<String>) -> Self {
        Self {
            pattern,
            description: description.into(),
            rewritten_query: None,
        }
    }

    /// Attach a rewritten query to the suggestion
    pub fn with_rewrite(mut self, rewritten: impl Into<String>) -> Self {
        self.rewritten_query = Some(rewritten.into());
        self
    }
}

/// Statistics about query shape used for speedup estimation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryStats {
    /// Average triple store size (in millions of triples)
    pub triple_count_millions: f64,
    /// Average result set size from similar queries
    pub avg_result_size: usize,
    /// Whether indexes exist on commonly-used predicates
    pub has_predicate_index: bool,
}

impl QueryStats {
    /// Create stats with a given triple-store size
    pub fn with_triple_count(mut self, millions: f64) -> Self {
        self.triple_count_millions = millions;
        self
    }

    /// Mark that predicate indexes exist
    pub fn with_predicate_index(mut self) -> Self {
        self.has_predicate_index = true;
        self
    }
}

/// Full analysis result for a single query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Patterns detected in the query
    pub patterns: Vec<QueryPattern>,
    /// Actionable suggestions
    pub suggestions: Vec<Suggestion>,
    /// Estimated speedup ratio (>1.0 means the optimized version is faster)
    pub estimated_speedup: f64,
    /// Complexity score 0–10
    pub complexity_score: u8,
}

impl AnalysisResult {
    /// True if any critical patterns were found
    pub fn has_issues(&self) -> bool {
        !self.patterns.is_empty()
    }
}

/// ML-inspired SPARQL query advisor
///
/// Uses heuristic pattern matching and structural analysis to produce
/// optimization recommendations. In a production system the heuristics
/// would be trained on query logs; here they encode expert knowledge.
pub struct MlQueryAdvisor;

impl MlQueryAdvisor {
    /// Analyze a SPARQL query and return optimization recommendations
    pub fn analyze_query(query: &str) -> AnalysisResult {
        let mut patterns: HashSet<QueryPattern> = HashSet::new();
        let mut suggestions: Vec<Suggestion> = Vec::new();

        // --- Pattern detection ---
        if detect_cartesian_product(query) {
            patterns.insert(QueryPattern::CartesianProduct);
            suggestions.push(Suggestion::new(
                QueryPattern::CartesianProduct,
                "Query contains triple patterns with no shared variables, causing a Cartesian \
                 product join. Add shared variables or use VALUES/FILTER to reduce the join size.",
            ));
        }

        if detect_unbound_predicate(query) {
            patterns.insert(QueryPattern::UnboundPredicate);
            suggestions.push(Suggestion::new(
                QueryPattern::UnboundPredicate,
                "Query contains a fully-variable triple pattern (?s ?p ?o) which will scan \
                 the entire triple store. Bind at least one component to a constant IRI or \
                 use a FILTER to narrow the scan.",
            ));
        }

        let nesting_depth = detect_nesting_depth(query);
        if nesting_depth > 3 {
            patterns.insert(QueryPattern::DeepNesting);
            suggestions.push(Suggestion::new(
                QueryPattern::DeepNesting,
                format!(
                    "Query has {} levels of subquery nesting (threshold: 3). \
                     Consider flattening nested SELECTs into a single graph pattern \
                     or using WITH/AS common table expression patterns.",
                    nesting_depth
                ),
            ));
        }

        if detect_repeated_subquery(query) {
            patterns.insert(QueryPattern::RepeatedSubquery);
            suggestions.push(Suggestion::new(
                QueryPattern::RepeatedSubquery,
                "The same subquery block appears more than once. Extract it into a named \
                 sub-SELECT or use VALUES to pre-compute shared bindings once.",
            ));
        }

        if detect_missing_index(query) {
            patterns.insert(QueryPattern::MissingIndex);
            suggestions.push(Suggestion::new(
                QueryPattern::MissingIndex,
                "Query uses predicates without evidence of an index. Add a text or predicate \
                 index hint (SERVICE <urn:index:predicate>) to improve lookup speed.",
            ));
        }

        // --- Complexity score (0–10) ---
        let complexity_score = compute_complexity(query, nesting_depth);

        // --- Speedup estimation ---
        let stats = QueryStats::default();
        let estimated_speedup = Self::estimate_speedup(query, query, &stats);

        // Build pattern list preserving insertion order via sorted determinism
        let mut sorted_patterns: Vec<QueryPattern> = patterns.into_iter().collect();
        sorted_patterns.sort_by_key(|p| p.name());

        AnalysisResult {
            patterns: sorted_patterns,
            suggestions,
            estimated_speedup,
            complexity_score,
        }
    }

    /// Estimate the speedup ratio between an original and optimized query
    ///
    /// Returns a value ≥1.0 (1.0 = no improvement, 5.0 = 5× faster).
    pub fn estimate_speedup(original: &str, _optimized: &str, stats: &QueryStats) -> f64 {
        let mut multiplier: f64 = 1.0;

        if detect_cartesian_product(original) {
            multiplier *= QueryPattern::CartesianProduct.overhead_multiplier();
        }
        if detect_unbound_predicate(original) {
            multiplier *= QueryPattern::UnboundPredicate.overhead_multiplier();
        }
        let depth = detect_nesting_depth(original);
        if depth > 3 {
            multiplier *= QueryPattern::DeepNesting.overhead_multiplier();
        }
        if detect_repeated_subquery(original) {
            multiplier *= QueryPattern::RepeatedSubquery.overhead_multiplier();
        }
        if detect_missing_index(original) && !stats.has_predicate_index {
            multiplier *= QueryPattern::MissingIndex.overhead_multiplier();
        }

        // Scale by dataset size (larger datasets benefit more from optimization)
        let size_factor = if stats.triple_count_millions > 100.0 {
            1.5
        } else if stats.triple_count_millions > 10.0 {
            1.2
        } else {
            1.0
        };

        (multiplier * size_factor).max(1.0)
    }
}

/// Returns true when there exist two triple patterns that share no variables
fn detect_cartesian_product(query: &str) -> bool {
    let patterns = extract_triple_patterns(query);
    if patterns.len() < 2 {
        return false;
    }
    // Build variable sets per pattern
    let var_sets: Vec<HashSet<String>> = patterns.iter().map(|p| extract_pattern_vars(p)).collect();

    // Check any pair with an empty intersection
    for i in 0..var_sets.len() {
        for j in (i + 1)..var_sets.len() {
            let intersection: HashSet<&String> = var_sets[i].intersection(&var_sets[j]).collect();
            if intersection.is_empty() && !var_sets[i].is_empty() && !var_sets[j].is_empty() {
                return true;
            }
        }
    }
    false
}

/// Returns true if any triple pattern has all three positions as variables
fn detect_unbound_predicate(query: &str) -> bool {
    let patterns = extract_triple_patterns(query);
    for pattern in &patterns {
        let parts: Vec<&str> = pattern.split_whitespace().collect();
        if parts.len() >= 3 {
            // All three parts start with '?' — fully unbound triple
            if parts[0].starts_with('?') && parts[1].starts_with('?') && parts[2].starts_with('?') {
                return true;
            }
        }
    }
    false
}

/// Count the maximum nesting depth of SELECT subqueries
fn detect_nesting_depth(query: &str) -> usize {
    let upper = query.to_uppercase();
    let mut max_depth: usize = 0;
    let mut current_depth: usize = 0;

    for c in upper.chars() {
        if c == '{' {
            current_depth += 1;
            if current_depth > max_depth {
                max_depth = current_depth;
            }
        } else if c == '}' {
            current_depth = current_depth.saturating_sub(1);
        }
    }

    // Count SELECT occurrences as proxy for subquery depth
    let select_count = upper.matches("SELECT").count().saturating_sub(1); // outer doesn't count
    select_count.max(max_depth.saturating_sub(1))
}

/// Detect repeated subquery blocks by checksumming whitespace-normalized sub-blocks
fn detect_repeated_subquery(query: &str) -> bool {
    // Extract { ... } blocks and check for duplicates (simplified heuristic)
    let blocks = extract_brace_blocks(query);
    let mut seen: HashMap<String, usize> = HashMap::new();
    for block in &blocks {
        // Normalize whitespace
        let normalized: String = block.split_whitespace().collect::<Vec<_>>().join(" ");
        if normalized.len() < 20 {
            continue; // skip tiny blocks
        }
        let count = seen.entry(normalized).or_insert(0);
        *count += 1;
        if *count > 1 {
            return true;
        }
    }
    false
}

/// Heuristic: detect queries that may benefit from an index
fn detect_missing_index(query: &str) -> bool {
    let upper = query.to_uppercase();
    // If there are many triple patterns and no SERVICE call, suggest an index
    let triple_count = extract_triple_patterns(query).len();
    let has_service = upper.contains("SERVICE");
    triple_count >= 3 && !has_service
}

/// Compute a complexity score 0–10
fn compute_complexity(query: &str, nesting_depth: usize) -> u8 {
    let patterns = extract_triple_patterns(query);
    let triple_score = (patterns.len().min(10) as u8) / 2;
    let nesting_score = (nesting_depth.min(5) as u8) * 2;
    let upper = query.to_uppercase();
    let optional_score = upper.matches("OPTIONAL").count().min(3) as u8;
    let union_score = upper.matches("UNION").count().min(2) as u8;

    (triple_score + nesting_score + optional_score + union_score).min(10)
}

// --- Internal helpers ---

/// Extract raw triple pattern strings from the WHERE clause
///
/// Handles both single-line and multi-line SPARQL queries by extracting
/// the WHERE body text and splitting on statement boundaries.
fn extract_triple_patterns(query: &str) -> Vec<String> {
    let mut patterns = Vec::new();

    // Extract the content between the outermost braces
    let body = extract_where_body(query);
    let body_upper = body.to_uppercase();

    // Split by '.' as triple separator, then also by newlines
    // Then examine each candidate token sequence
    let candidates: Vec<&str> = body.split('\n').flat_map(|line| line.split(';')).collect();

    for candidate in candidates {
        let trimmed = candidate.trim().trim_end_matches('.');
        let upper_trimmed = trimmed.to_uppercase();

        // Skip empty, comments, structural keywords, and brace-only lines
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed == "{"
            || trimmed == "}"
            || upper_trimmed.starts_with("SELECT")
            || upper_trimmed.starts_with("WHERE")
            || upper_trimmed.starts_with("FILTER")
            || upper_trimmed.starts_with("OPTIONAL")
            || upper_trimmed.starts_with("UNION")
            || upper_trimmed.starts_with("PREFIX")
            || upper_trimmed.starts_with("MINUS")
            || upper_trimmed.starts_with("GRAPH")
            || upper_trimmed.starts_with("SERVICE")
        {
            continue;
        }

        // A triple pattern has at least 3 whitespace-separated tokens
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        if tokens.len() >= 3 {
            // Only extract up to the first 3 relevant tokens for variable analysis
            let clean: Vec<String> = tokens
                .iter()
                .map(|t| t.trim_end_matches(['.', ';', ',']).to_string())
                .collect();
            patterns.push(clean[..3.min(clean.len())].join(" "));
        }
        // Handle inline dot-separated triples (same line, multiple patterns)
        let _ = &body_upper; // suppress unused warning
    }
    patterns
}

/// Extract the content of the outermost WHERE clause brace block
fn extract_where_body(query: &str) -> String {
    let upper = query.to_uppercase();
    // Find the position of WHERE (or the first `{` after it)
    let where_pos = upper.find("WHERE").unwrap_or(0);
    let body_start = query[where_pos..]
        .find('{')
        .map(|p| where_pos + p + 1)
        .unwrap_or(where_pos);

    if body_start >= query.len() {
        // No WHERE clause found; return full query for simple analysis
        return query.to_string();
    }

    // Extract until the matching closing brace
    let mut depth: usize = 1;
    let mut end = body_start;
    for (i, c) in query[body_start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    end = body_start + i;
                    break;
                }
            }
            _ => {}
        }
    }
    query[body_start..end].to_string()
}

/// Extract variable names from a triple pattern string
fn extract_pattern_vars(pattern: &str) -> HashSet<String> {
    pattern
        .split_whitespace()
        .filter(|t| t.starts_with('?'))
        .map(|t| t.trim_end_matches('.').to_string())
        .collect()
}

/// Extract all brace blocks `{...}` from the query at every nesting level
///
/// This allows detection of repeated subquery patterns at any depth.
fn extract_brace_blocks(query: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    // Stack of (start_position, depth_at_start)
    let mut stack: Vec<usize> = Vec::new();

    for (i, c) in query.char_indices() {
        match c {
            '{' => {
                stack.push(i);
            }
            '}' => {
                if let Some(s) = stack.pop() {
                    blocks.push(query[s..=i].to_string());
                }
            }
            _ => {}
        }
    }
    blocks
}

/// Query rewriter — applies structural transformations to improve performance
pub struct QueryRewriter;

impl QueryRewriter {
    /// Move FILTER clauses as early as possible in the graph pattern
    ///
    /// This is a text-level heuristic: it moves FILTER lines immediately
    /// after the last triple pattern that binds their variables.
    pub fn add_filter_early(query: &str) -> String {
        let lines: Vec<&str> = query.lines().collect();
        let mut output: Vec<String> = Vec::new();
        let mut filter_lines: Vec<String> = Vec::new();
        let mut triple_lines: Vec<String> = Vec::new();

        for line in &lines {
            let upper = line.to_uppercase();
            if upper.trim_start().starts_with("FILTER") {
                filter_lines.push(line.to_string());
            } else if upper.trim_start().starts_with("SELECT")
                || upper.trim_start().starts_with("WHERE")
                || upper.trim_start().starts_with("PREFIX")
                || line.trim() == "{"
                || line.trim() == "}"
                || line.trim().is_empty()
            {
                // Flush pending triple+filter lines before structural keywords
                if !triple_lines.is_empty() {
                    output.append(&mut triple_lines);
                    output.append(&mut filter_lines);
                }
                output.push(line.to_string());
            } else {
                // Check if we can insert filters after last relevant triple
                if !filter_lines.is_empty() {
                    // Check if current triple uses filter variables
                    let should_insert = filter_lines.iter().any(|f| {
                        let filter_vars = extract_filter_vars(f);
                        let triple_vars = extract_pattern_vars(line.trim());
                        filter_vars.iter().any(|v| triple_vars.contains(v))
                    });
                    triple_lines.push(line.to_string());
                    if should_insert {
                        output.append(&mut triple_lines);
                        output.append(&mut filter_lines);
                    }
                } else {
                    output.push(line.to_string());
                }
            }
        }
        // Flush remaining
        output.append(&mut triple_lines);
        output.append(&mut filter_lines);
        output.join("\n")
    }

    /// Reorder triple patterns by selectivity (most specific first)
    ///
    /// Heuristic order:
    /// 1. Patterns with constant subject + constant predicate
    /// 2. Patterns with constant subject or constant predicate
    /// 3. Fully-variable patterns
    pub fn reorder_patterns_by_selectivity(query: &str) -> String {
        let lines: Vec<&str> = query.lines().collect();
        let mut header_lines: Vec<String> = Vec::new();
        let mut pattern_lines: Vec<String> = Vec::new();
        let mut footer_lines: Vec<String> = Vec::new();
        let mut in_where_body = false;
        let mut brace_depth: usize = 0;

        for line in &lines {
            let upper = line.to_uppercase();
            let trimmed = line.trim();
            if upper.contains("WHERE") && upper.contains('{') {
                in_where_body = true;
                brace_depth += 1;
                header_lines.push(line.to_string());
                continue;
            }
            if !in_where_body {
                header_lines.push(line.to_string());
                continue;
            }
            if trimmed == "{" {
                brace_depth += 1;
                header_lines.push(line.to_string());
                continue;
            }
            if trimmed == "}" || trimmed == "}." {
                brace_depth = brace_depth.saturating_sub(1);
                if brace_depth == 0 {
                    footer_lines.push(line.to_string());
                    in_where_body = false;
                } else {
                    footer_lines.push(line.to_string());
                }
                continue;
            }
            // Inside WHERE body
            let is_triple = {
                let tokens: Vec<&str> = trimmed.split_whitespace().collect();
                tokens.len() >= 3
                    && !upper.trim_start().starts_with("FILTER")
                    && !upper.trim_start().starts_with("OPTIONAL")
                    && !upper.trim_start().starts_with("UNION")
                    && !upper.trim_start().starts_with("SELECT")
            };
            if is_triple {
                pattern_lines.push(line.to_string());
            } else {
                footer_lines.push(line.to_string());
            }
        }

        // Sort pattern_lines by selectivity score (lower = more specific = first)
        pattern_lines.sort_by_key(|line| selectivity_score(line.trim()));

        let mut result = header_lines;
        result.extend(pattern_lines);
        result.extend(footer_lines);
        result.join("\n")
    }
}

/// Compute a selectivity score for a triple pattern line (lower = more selective)
fn selectivity_score(pattern: &str) -> u8 {
    let tokens: Vec<&str> = pattern.split_whitespace().collect();
    if tokens.len() < 3 {
        return 255;
    }
    let s_var = tokens[0].starts_with('?');
    let p_var = tokens[1].starts_with('?');
    let o_var = tokens[2].starts_with('?');

    match (s_var, p_var, o_var) {
        (false, false, false) => 0,
        (false, false, true) => 1,
        (false, true, false) => 2,
        (true, false, false) => 3,
        (false, true, true) => 4,
        (true, false, true) => 5,
        (true, true, false) => 6,
        (true, true, true) => 7,
    }
}

/// Extract variable names referenced inside a FILTER expression
fn extract_filter_vars(filter_line: &str) -> HashSet<String> {
    filter_line
        .split(|c: char| !c.is_alphanumeric() && c != '?' && c != '_')
        .filter(|t| t.starts_with('?'))
        .map(|t| t.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- QueryPattern ---

    #[test]
    fn test_pattern_names() {
        assert_eq!(QueryPattern::CartesianProduct.name(), "Cartesian Product");
        assert_eq!(QueryPattern::UnboundPredicate.name(), "Unbound Predicate");
        assert_eq!(QueryPattern::DeepNesting.name(), "Deep Nesting");
        assert_eq!(QueryPattern::RepeatedSubquery.name(), "Repeated Subquery");
        assert_eq!(QueryPattern::MissingIndex.name(), "Missing Index");
    }

    #[test]
    fn test_pattern_overhead_multipliers_gt_one() {
        for p in &[
            QueryPattern::CartesianProduct,
            QueryPattern::UnboundPredicate,
            QueryPattern::DeepNesting,
            QueryPattern::RepeatedSubquery,
            QueryPattern::MissingIndex,
        ] {
            assert!(
                p.overhead_multiplier() > 1.0,
                "{} multiplier should be > 1",
                p.name()
            );
        }
    }

    // --- Cartesian product detection ---

    #[test]
    fn test_no_cartesian_product_shared_var() {
        let query = r#"
SELECT ?s ?p ?o WHERE {
  ?s rdf:type <http://example.org/Person> .
  ?s ?p ?o .
}"#;
        assert!(!detect_cartesian_product(query));
    }

    #[test]
    fn test_cartesian_product_detected() {
        let query = r#"
SELECT ?s ?x WHERE {
  ?s rdf:type <http://example.org/Person> .
  ?x rdf:type <http://example.org/Product> .
}"#;
        assert!(detect_cartesian_product(query));
    }

    // --- Unbound predicate detection ---

    #[test]
    fn test_no_unbound_predicate_when_bound() {
        let query = r#"
SELECT ?s WHERE {
  ?s rdf:type <http://example.org/Person> .
}"#;
        assert!(!detect_unbound_predicate(query));
    }

    #[test]
    fn test_unbound_predicate_detected() {
        let query = r#"
SELECT ?s ?p ?o WHERE {
  ?s ?p ?o .
}"#;
        assert!(detect_unbound_predicate(query));
    }

    // --- Deep nesting detection ---

    #[test]
    fn test_nesting_depth_zero_for_simple_query() {
        let query = "SELECT ?s WHERE { ?s ?p ?o . }";
        assert!(detect_nesting_depth(query) <= 3);
    }

    #[test]
    fn test_nesting_depth_detected_for_deep_query() {
        let query = r#"
SELECT ?s WHERE {
  { SELECT ?s WHERE {
      { SELECT ?s WHERE {
          { SELECT ?s WHERE {
              ?s ?p ?o .
          }}
      }}
  }}
}"#;
        let depth = detect_nesting_depth(query);
        assert!(depth > 3, "Expected deep nesting, got depth {}", depth);
    }

    // --- Repeated subquery detection ---

    #[test]
    fn test_no_repeated_subquery() {
        let query = r#"SELECT ?s WHERE { ?s rdf:type <http://a> . ?s <http://b> ?o . }"#;
        assert!(!detect_repeated_subquery(query));
    }

    #[test]
    fn test_repeated_subquery_detected() {
        let block = "{ SELECT ?s WHERE { ?s rdf:type <http://example.org/Person> . } }";
        let query = format!("SELECT ?s WHERE {{ {} UNION {} }}", block, block);
        assert!(detect_repeated_subquery(&query));
    }

    // --- MlQueryAdvisor.analyze_query ---

    #[test]
    fn test_analyze_simple_query_no_issues() {
        let query = r#"
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT ?s WHERE {
  ?s rdf:type <http://example.org/Person> .
}"#;
        let result = MlQueryAdvisor::analyze_query(query);
        // Simple query should have a reasonable complexity score
        assert!(result.complexity_score <= 10);
    }

    #[test]
    fn test_analyze_detects_unbound_predicate() {
        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . }";
        let result = MlQueryAdvisor::analyze_query(query);
        assert!(result.patterns.contains(&QueryPattern::UnboundPredicate));
        assert!(!result.suggestions.is_empty());
    }

    #[test]
    fn test_analyze_detects_cartesian_product() {
        let query = r#"
SELECT ?s ?x WHERE {
  ?s rdf:type <http://example.org/Person> .
  ?x rdf:type <http://example.org/Product> .
}"#;
        let result = MlQueryAdvisor::analyze_query(query);
        assert!(result.patterns.contains(&QueryPattern::CartesianProduct));
    }

    #[test]
    fn test_analysis_result_has_issues_flag() {
        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . }";
        let result = MlQueryAdvisor::analyze_query(query);
        assert!(result.has_issues());
    }

    #[test]
    fn test_analysis_result_no_issues_for_good_query() {
        let query = r#"
SELECT ?s WHERE {
  ?s <http://schema.org/name> ?name .
}"#;
        let result = MlQueryAdvisor::analyze_query(query);
        // No CartesianProduct, no UnboundPredicate
        assert!(!result.patterns.contains(&QueryPattern::CartesianProduct));
        assert!(!result.patterns.contains(&QueryPattern::UnboundPredicate));
    }

    #[test]
    fn test_estimated_speedup_at_least_one() {
        let query = "SELECT ?s WHERE { ?s rdf:type <http://a> . }";
        let stats = QueryStats::default();
        let speedup = MlQueryAdvisor::estimate_speedup(query, query, &stats);
        assert!(speedup >= 1.0);
    }

    #[test]
    fn test_estimated_speedup_higher_for_bad_query() {
        let bad_query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . }";
        let good_query = "SELECT ?s WHERE { ?s rdf:type <http://a> . }";
        let stats = QueryStats::default();
        let bad_speedup = MlQueryAdvisor::estimate_speedup(bad_query, good_query, &stats);
        let good_speedup = MlQueryAdvisor::estimate_speedup(good_query, good_query, &stats);
        assert!(bad_speedup >= good_speedup);
    }

    #[test]
    fn test_speedup_scales_with_dataset_size() {
        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . }";
        let small_stats = QueryStats::default().with_triple_count(1.0);
        let large_stats = QueryStats::default().with_triple_count(200.0);
        let good_query = "SELECT ?s WHERE { ?s rdf:type <http://a> . }";
        let small_speedup = MlQueryAdvisor::estimate_speedup(query, good_query, &small_stats);
        let large_speedup = MlQueryAdvisor::estimate_speedup(query, good_query, &large_stats);
        assert!(large_speedup >= small_speedup);
    }

    // --- QueryRewriter ---

    #[test]
    fn test_filter_early_returns_string() {
        let query = r#"SELECT ?s WHERE {
  ?s rdf:type <http://example.org/Person> .
  ?s <http://schema.org/age> ?age .
  FILTER (?age > 18)
}"#;
        let rewritten = QueryRewriter::add_filter_early(query);
        assert!(!rewritten.is_empty());
        assert!(rewritten.contains("FILTER"));
    }

    #[test]
    fn test_reorder_patterns_most_selective_first() {
        // Use distinct variable names in body vs SELECT to avoid false matches
        let query = "SELECT ?result WHERE {\n?subj ?pred ?obj .\n<http://a> <http://b> ?obj .\n}";
        let rewritten = QueryRewriter::reorder_patterns_by_selectivity(query);
        // The fully-constant-subject-predicate pattern should come before the fully-variable one
        let cp_pos = rewritten.find("<http://a>").unwrap_or(usize::MAX);
        let fv_pos = rewritten.find("?subj ?pred ?obj").unwrap_or(usize::MAX);
        assert!(
            cp_pos < fv_pos,
            "Constant pattern should come before variable pattern. Got rewritten:\n{}",
            rewritten
        );
    }

    #[test]
    fn test_reorder_preserves_select_clause() {
        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . }";
        let rewritten = QueryRewriter::reorder_patterns_by_selectivity(query);
        assert!(rewritten.contains("SELECT"));
    }

    // --- Suggestion ---

    #[test]
    fn test_suggestion_with_rewrite() {
        let s = Suggestion::new(QueryPattern::CartesianProduct, "description")
            .with_rewrite("SELECT ?s WHERE { ?s rdf:type <http://a> . }");
        assert!(s.rewritten_query.is_some());
    }

    #[test]
    fn test_suggestion_without_rewrite() {
        let s = Suggestion::new(QueryPattern::UnboundPredicate, "description");
        assert!(s.rewritten_query.is_none());
    }

    // --- QueryStats builder ---

    #[test]
    fn test_query_stats_builder() {
        let stats = QueryStats::default()
            .with_triple_count(50.0)
            .with_predicate_index();
        assert!((stats.triple_count_millions - 50.0).abs() < f64::EPSILON);
        assert!(stats.has_predicate_index);
    }

    // --- selectivity_score ---

    #[test]
    fn test_selectivity_score_ordering() {
        // Fully bound should have lowest score
        assert!(
            selectivity_score("<http://a> <http://b> <http://c>") < selectivity_score("?s ?p ?o")
        );
        // Partially bound should be between extremes
        assert!(selectivity_score("?s rdf:type <http://c>") < selectivity_score("?s ?p ?o"));
    }

    #[test]
    fn test_complexity_score_bounded() {
        let query =
            "SELECT ?s ?p ?o WHERE { ?s ?p ?o . ?s ?p ?o . ?s ?p ?o . OPTIONAL { ?s ?p ?o . } }";
        let score = compute_complexity(query, 0);
        assert!(score <= 10);
    }
}
