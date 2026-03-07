//! Source Selection Engine for federated SPARQL queries
//!
//! Determines which endpoints can answer which triple patterns using
//! capability-based reasoning, VoID statistics, and cost estimation.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// Capability description of a SPARQL endpoint.
///
/// Captures what data and features an endpoint supports, derived from
/// VoID descriptions, SPARQL introspection, or manual configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointCapabilities {
    /// Unique identifier for this endpoint (e.g. "dbpedia", "wikidata")
    pub endpoint_id: String,
    /// The SPARQL endpoint URL
    pub endpoint_url: String,
    /// Known predicates at this endpoint (from VoID `void:property`)
    pub predicates: HashSet<String>,
    /// Known classes at this endpoint (from VoID `void:class`)
    pub classes: HashSet<String>,
    /// Named graphs available
    pub named_graphs: HashSet<String>,
    /// Estimated triple count (from VoID `void:triples` or sampling)
    pub triple_count: Option<u64>,
    /// Whether this endpoint supports SPARQL 1.1
    pub sparql_11: bool,
    /// Whether this endpoint supports OPTIONAL patterns
    pub supports_optional: bool,
    /// Whether this endpoint supports UNION patterns
    pub supports_union: bool,
    /// Whether this endpoint supports sub-queries
    pub supports_subqueries: bool,
}

impl EndpointCapabilities {
    /// Create a new endpoint capabilities descriptor with default settings.
    pub fn new(id: &str, url: &str) -> Self {
        Self {
            endpoint_id: id.to_string(),
            endpoint_url: url.to_string(),
            predicates: HashSet::new(),
            classes: HashSet::new(),
            named_graphs: HashSet::new(),
            triple_count: None,
            sparql_11: true,
            supports_optional: true,
            supports_union: true,
            supports_subqueries: true,
        }
    }

    /// Check if this endpoint might be able to answer a triple pattern.
    ///
    /// Returns `true` when:
    /// - No capability information is available (optimistic assumption), OR
    /// - The predicate is known to this endpoint, OR
    /// - The object looks like a class IRI that is known to this endpoint
    pub fn can_answer_pattern(
        &self,
        _subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> bool {
        // If we have no capability metadata, be optimistic
        if self.predicates.is_empty() && self.classes.is_empty() {
            return true;
        }

        let pred = match predicate {
            None => return true, // unbound predicate: optimistic
            Some(p) => p,
        };

        // Variable predicate: optimistic
        if pred.starts_with('?') {
            return true;
        }

        // rdf:type: check whether the object class is known
        if pred == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
            match object {
                Some(obj) if !obj.starts_with('?') => {
                    return self.classes.contains(obj);
                }
                _ => return true, // variable object: optimistic
            }
        }

        // Check if the predicate is known
        self.predicates.contains(pred)
    }

    /// Estimate the selectivity of a predicate against this endpoint.
    ///
    /// Returns a value in `[0.0, 1.0]` where lower means more selective
    /// (fewer expected results). Uses triple-count heuristics when
    /// VoID statistics are unavailable.
    pub fn estimate_selectivity(&self, predicate: Option<&str>) -> f64 {
        let pred = match predicate {
            Some(p) if !p.starts_with('?') => p,
            _ => return 1.0, // variable predicate: very unselective
        };

        let total = match self.triple_count {
            Some(t) if t > 0 => t as f64,
            _ => return 0.5, // unknown cardinality: mid-point estimate
        };

        // rdf:type tends to match many triples
        if pred == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
            return (1000_f64 / total).clamp(0.001, 1.0);
        }

        // For known predicates we estimate ~1000 triples on average
        if self.predicates.contains(pred) {
            return (1000_f64 / total).clamp(0.001, 1.0);
        }

        // Unknown predicate at this endpoint: effectively zero selectivity
        0.0
    }
}

/// A triple pattern extracted from a federated SPARQL query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedPattern {
    /// Unique identifier for this pattern within a query
    pub id: String,
    /// Subject term — `None` means a variable
    pub subject: Option<String>,
    /// Predicate term — `None` means a variable
    pub predicate: Option<String>,
    /// Object term — `None` means a variable
    pub object: Option<String>,
    /// Named graph restriction, if any
    pub graph: Option<String>,
    /// Candidate endpoint IDs gathered during query planning
    pub relevant_endpoints: Vec<String>,
}

impl FederatedPattern {
    /// Create a new federated pattern with an explicit ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            subject: None,
            predicate: None,
            object: None,
            graph: None,
            relevant_endpoints: Vec::new(),
        }
    }
}

/// How a subquery pattern is assigned to one or more endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssignmentStrategy {
    /// Send to a single best endpoint (lowest cost, highest confidence)
    SingleSource,
    /// Send to all candidate endpoints and take the union of results
    AllSources,
    /// Probe endpoints in the given priority order; stop at first success
    ProbingOrder(Vec<String>),
}

/// The result of assigning a single federated pattern to endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceAssignment {
    /// The pattern this assignment covers
    pub pattern_id: String,
    /// Endpoint IDs that will execute this pattern
    pub assigned_endpoints: Vec<String>,
    /// Estimated relative cost (lower is better)
    pub estimated_cost: f64,
    /// The strategy chosen for this assignment
    pub strategy: AssignmentStrategy,
}

/// Reasons a source was selected or rejected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionReason {
    /// The endpoint being described
    pub endpoint_id: String,
    /// Whether this endpoint was selected
    pub selected: bool,
    /// Human-readable reason for the selection decision
    pub reason: String,
    /// Confidence/selectivity score associated with this endpoint
    pub confidence: f64,
}

/// Source selection engine.
///
/// Maintains a registry of endpoint capabilities and, given a set of
/// `FederatedPattern`s, decides which endpoints should answer each.
pub struct SourceSelector {
    endpoints: HashMap<String, EndpointCapabilities>,
}

impl Default for SourceSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceSelector {
    /// Create an empty source selector.
    pub fn new() -> Self {
        Self {
            endpoints: HashMap::new(),
        }
    }

    /// Register an endpoint's capabilities.
    pub fn register_endpoint(&mut self, caps: EndpointCapabilities) {
        self.endpoints.insert(caps.endpoint_id.clone(), caps);
    }

    /// Remove an endpoint from the registry.
    pub fn deregister_endpoint(&mut self, endpoint_id: &str) {
        self.endpoints.remove(endpoint_id);
    }

    /// Return all registered endpoint IDs.
    pub fn endpoint_ids(&self) -> Vec<&str> {
        self.endpoints.keys().map(String::as_str).collect()
    }

    /// Select sources for a list of triple patterns.
    ///
    /// For each pattern the algorithm:
    /// 1. Filters capable endpoints using `can_answer_pattern`.
    /// 2. If exactly one endpoint is capable, assigns it exclusively.
    /// 3. If multiple endpoints are capable, chooses between `SingleSource`
    ///    and `AllSources` based on selectivity estimates.
    /// 4. Falls back to `AllSources` across all registered endpoints when
    ///    no capable endpoints are found (open-world assumption).
    pub fn select_sources(&self, patterns: &[FederatedPattern]) -> Vec<SourceAssignment> {
        patterns
            .iter()
            .map(|pattern| self.assign_pattern(pattern))
            .collect()
    }

    /// Assign endpoints to a single pattern.
    fn assign_pattern(&self, pattern: &FederatedPattern) -> SourceAssignment {
        // Determine which registered endpoints are candidates
        let candidates: Vec<&EndpointCapabilities> = if pattern.relevant_endpoints.is_empty() {
            self.endpoints.values().collect()
        } else {
            pattern
                .relevant_endpoints
                .iter()
                .filter_map(|id| self.endpoints.get(id))
                .collect()
        };

        // Filter by capability
        let capable: Vec<&EndpointCapabilities> = candidates
            .into_iter()
            .filter(|ep| {
                ep.can_answer_pattern(
                    pattern.subject.as_deref(),
                    pattern.predicate.as_deref(),
                    pattern.object.as_deref(),
                )
            })
            .collect();

        if capable.is_empty() {
            // Fall back: broadcast to all known endpoints
            let all: Vec<String> = self.endpoints.keys().cloned().collect();
            let cost = self.broadcast_cost(all.len());
            return SourceAssignment {
                pattern_id: pattern.id.clone(),
                assigned_endpoints: all,
                estimated_cost: cost,
                strategy: AssignmentStrategy::AllSources,
            };
        }

        // Exclusive source: only one endpoint can answer
        if capable.len() == 1 {
            let ep = capable[0];
            let cost = self.single_source_cost(ep, pattern);
            return SourceAssignment {
                pattern_id: pattern.id.clone(),
                assigned_endpoints: vec![ep.endpoint_id.clone()],
                estimated_cost: cost,
                strategy: AssignmentStrategy::SingleSource,
            };
        }

        // Multiple capable endpoints: rank by selectivity
        let mut ranked: Vec<(&EndpointCapabilities, f64)> = capable
            .iter()
            .map(|ep| (*ep, ep.estimate_selectivity(pattern.predicate.as_deref())))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // If top endpoint has substantially higher selectivity, use it exclusively
        let top_sel = ranked[0].1;
        let second_sel = ranked[1].1;
        if top_sel > 0.0 && second_sel > 0.0 && (top_sel / second_sel) > 2.0 {
            let best_ep = ranked[0].0;
            let cost = self.single_source_cost(best_ep, pattern);
            return SourceAssignment {
                pattern_id: pattern.id.clone(),
                assigned_endpoints: vec![best_ep.endpoint_id.clone()],
                estimated_cost: cost,
                strategy: AssignmentStrategy::SingleSource,
            };
        }

        // Similar selectivity across endpoints: probe in ranked order
        let order: Vec<String> = ranked
            .iter()
            .map(|(ep, _)| ep.endpoint_id.clone())
            .collect();
        let cost = self.multi_source_cost(&ranked);

        SourceAssignment {
            pattern_id: pattern.id.clone(),
            assigned_endpoints: order.clone(),
            estimated_cost: cost,
            strategy: AssignmentStrategy::ProbingOrder(order),
        }
    }

    /// Estimate total query cost for a set of assignments.
    ///
    /// Uses the critical-path model: the bottleneck is the slowest
    /// assignment, plus a small merging overhead.
    pub fn estimate_query_cost(&self, assignments: &[SourceAssignment]) -> f64 {
        if assignments.is_empty() {
            return 0.0;
        }
        let max_cost = assignments
            .iter()
            .map(|a| a.estimated_cost)
            .fold(0.0_f64, f64::max);
        // 10 % merging overhead per additional assignment
        let merge_overhead = (assignments.len() as f64 - 1.0).max(0.0) * 0.1 * max_cost;
        max_cost + merge_overhead
    }

    // --- cost helpers -------------------------------------------------------

    fn single_source_cost(&self, ep: &EndpointCapabilities, pattern: &FederatedPattern) -> f64 {
        let sel = ep.estimate_selectivity(pattern.predicate.as_deref());
        let base = ep.triple_count.unwrap_or(1_000_000) as f64;
        base * (1.0 - sel.min(0.9999))
    }

    fn multi_source_cost(&self, ranked: &[(&EndpointCapabilities, f64)]) -> f64 {
        ranked
            .iter()
            .map(|(ep, sel)| {
                let base = ep.triple_count.unwrap_or(1_000_000) as f64;
                base * (1.0 - sel.min(0.9999))
            })
            .sum()
    }

    fn broadcast_cost(&self, endpoint_count: usize) -> f64 {
        1_000_000.0 * endpoint_count as f64
    }
}

/// Explain why each endpoint was selected or rejected for a pattern.
///
/// Useful for debugging and exposing federation decisions to operators.
pub fn explain_selection(
    selector: &SourceSelector,
    pattern: &FederatedPattern,
) -> Vec<SelectionReason> {
    selector
        .endpoints
        .values()
        .map(|ep| {
            let selected = ep.can_answer_pattern(
                pattern.subject.as_deref(),
                pattern.predicate.as_deref(),
                pattern.object.as_deref(),
            );
            let confidence = ep.estimate_selectivity(pattern.predicate.as_deref());
            let reason = if selected {
                if ep.predicates.is_empty() && ep.classes.is_empty() {
                    "Selected (no capability metadata – optimistic)".to_string()
                } else {
                    format!("Selected (predicate/class match, selectivity={confidence:.4})")
                }
            } else {
                "Rejected (predicate or class not in endpoint VoID description)".to_string()
            };
            SelectionReason {
                endpoint_id: ep.endpoint_id.clone(),
                selected,
                reason,
                confidence,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_endpoint(id: &str, predicates: &[&str], triples: u64) -> EndpointCapabilities {
        let mut caps = EndpointCapabilities::new(id, &format!("http://{id}.example/sparql"));
        caps.triple_count = Some(triples);
        for p in predicates {
            caps.predicates.insert((*p).to_string());
        }
        caps
    }

    fn make_pattern(id: &str, predicate: Option<&str>) -> FederatedPattern {
        FederatedPattern {
            id: id.to_string(),
            subject: None,
            predicate: predicate.map(str::to_string),
            object: None,
            graph: None,
            relevant_endpoints: vec![],
        }
    }

    #[test]
    fn test_can_answer_pattern_no_metadata() {
        let caps = EndpointCapabilities::new("ep1", "http://ep1/sparql");
        assert!(caps.can_answer_pattern(None, Some("http://ex.org/name"), None));
    }

    #[test]
    fn test_can_answer_pattern_known_predicate() {
        let mut caps = EndpointCapabilities::new("ep1", "http://ep1/sparql");
        caps.predicates.insert("http://ex.org/name".to_string());

        assert!(caps.can_answer_pattern(None, Some("http://ex.org/name"), None));
        assert!(!caps.can_answer_pattern(None, Some("http://ex.org/unknown"), None));
    }

    #[test]
    fn test_can_answer_pattern_variable_predicate() {
        let mut caps = EndpointCapabilities::new("ep1", "http://ep1/sparql");
        caps.predicates.insert("http://ex.org/name".to_string());
        assert!(caps.can_answer_pattern(None, Some("?p"), None));
    }

    #[test]
    fn test_can_answer_rdf_type_class_known() {
        let mut caps = EndpointCapabilities::new("ep1", "http://ep1/sparql");
        caps.classes.insert("http://ex.org/Person".to_string());

        let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
        assert!(caps.can_answer_pattern(None, Some(rdf_type), Some("http://ex.org/Person")));
        assert!(!caps.can_answer_pattern(None, Some(rdf_type), Some("http://ex.org/Unknown")));
    }

    #[test]
    fn test_exclusive_source_selection() {
        let mut selector = SourceSelector::new();
        selector.register_endpoint(make_endpoint("ep1", &["http://ex.org/foo"], 100_000));
        selector.register_endpoint(make_endpoint("ep2", &["http://ex.org/bar"], 200_000));

        let pattern = make_pattern("p1", Some("http://ex.org/foo"));
        let assignments = selector.select_sources(&[pattern]);

        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].assigned_endpoints.len(), 1);
        assert_eq!(assignments[0].assigned_endpoints[0], "ep1");
        assert!(matches!(
            assignments[0].strategy,
            AssignmentStrategy::SingleSource
        ));
    }

    #[test]
    fn test_multi_source_selection() {
        let mut selector = SourceSelector::new();
        selector.register_endpoint(make_endpoint("ep1", &["http://ex.org/name"], 100_000));
        selector.register_endpoint(make_endpoint("ep2", &["http://ex.org/name"], 200_000));

        let pattern = make_pattern("p1", Some("http://ex.org/name"));
        let assignments = selector.select_sources(&[pattern]);

        assert_eq!(assignments.len(), 1);
        assert!(!assignments[0].assigned_endpoints.is_empty());
    }

    #[test]
    fn test_no_capable_endpoint_broadcasts() {
        let mut selector = SourceSelector::new();
        selector.register_endpoint(make_endpoint("ep1", &["http://ex.org/foo"], 100_000));
        selector.register_endpoint(make_endpoint("ep2", &["http://ex.org/bar"], 200_000));

        let pattern = make_pattern("p1", Some("http://ex.org/unknown"));
        let assignments = selector.select_sources(&[pattern]);

        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].assigned_endpoints.len(), 2);
        assert!(matches!(
            assignments[0].strategy,
            AssignmentStrategy::AllSources
        ));
    }

    #[test]
    fn test_estimate_query_cost_multiple_patterns() {
        let mut selector = SourceSelector::new();
        selector.register_endpoint(make_endpoint("ep1", &["http://ex.org/foo"], 100_000));

        let patterns = vec![
            make_pattern("p1", Some("http://ex.org/foo")),
            make_pattern("p2", Some("http://ex.org/foo")),
        ];
        let assignments = selector.select_sources(&patterns);
        let cost = selector.estimate_query_cost(&assignments);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_register_deregister() {
        let mut selector = SourceSelector::new();
        selector.register_endpoint(make_endpoint("ep1", &[], 0));
        assert_eq!(selector.endpoint_ids().len(), 1);

        selector.deregister_endpoint("ep1");
        assert_eq!(selector.endpoint_ids().len(), 0);
    }

    #[test]
    fn test_selectivity_unknown_predicate() {
        let caps = make_endpoint("ep1", &["http://ex.org/known"], 1_000_000);
        assert_eq!(
            caps.estimate_selectivity(Some("http://ex.org/unknown")),
            0.0
        );
    }

    #[test]
    fn test_selectivity_known_predicate() {
        let caps = make_endpoint("ep1", &["http://ex.org/known"], 1_000_000);
        let sel = caps.estimate_selectivity(Some("http://ex.org/known"));
        assert!(sel > 0.0 && sel <= 1.0);
    }

    #[test]
    fn test_explain_selection() {
        let mut selector = SourceSelector::new();
        selector.register_endpoint(make_endpoint("ep1", &["http://ex.org/foo"], 100_000));
        selector.register_endpoint(make_endpoint("ep2", &["http://ex.org/bar"], 200_000));

        let pattern = make_pattern("p1", Some("http://ex.org/foo"));
        let reasons = explain_selection(&selector, &pattern);

        assert_eq!(reasons.len(), 2);
        let ep1_reason = reasons.iter().find(|r| r.endpoint_id == "ep1");
        let ep2_reason = reasons.iter().find(|r| r.endpoint_id == "ep2");
        assert!(ep1_reason.map(|r| r.selected).unwrap_or(false));
        assert!(!ep2_reason.map(|r| r.selected).unwrap_or(true));
    }

    #[test]
    fn test_empty_patterns_returns_empty() {
        let selector = SourceSelector::new();
        let assignments = selector.select_sources(&[]);
        assert!(assignments.is_empty());
    }
}
