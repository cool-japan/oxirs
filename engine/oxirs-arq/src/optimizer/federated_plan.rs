//! Federation-aware query planning pass.
//!
//! This pass rewrites a SPARQL algebra so that triple patterns whose subject,
//! predicate, or object IRI sits inside a *known federated dataset namespace*
//! get wrapped in [`Algebra::Service`] nodes that delegate to the appropriate
//! remote SPARQL endpoint.
//!
//! The pass is dependency-light: it operates against a [`SourceSelectivityProvider`]
//! trait that downstream callers (typically `oxirs-federate`) can implement.
//! This keeps `oxirs-arq` decoupled from the heavy federation runtime while
//! still letting the planner make federation-aware cost choices.
//!
//! # Concept
//!
//! Given a query like:
//!
//! ```sparql
//! SELECT ?s ?o WHERE {
//!     ?s <http://example.com/dbpedia/property> ?o .
//! }
//! ```
//!
//! and a registered federated dataset for the namespace `http://example.com/dbpedia/`,
//! the planner will rewrite the BGP into:
//!
//! ```text
//! Service { endpoint = "https://dbpedia.org/sparql", pattern = Bgp([..]), silent = false }
//! ```
//!
//! At execution time the [`crate::executor::QueryExecutor`] hands such
//! `Service` nodes off to the federation runtime (typically
//! `oxirs-federate::endpoint_client`).
//!
//! # Cost Model
//!
//! The pass also feeds the registered cost model: each `Service` node costs
//! the round-trip overhead of a federated call plus the source's reported
//! selectivity for the BGP.  Adaptive plans can use feedback from
//! [`crate::optimizer::adaptive::AdaptiveStatsStore`] to re-issue patterns
//! against alternate endpoints if a source becomes slow or unreliable.

use std::collections::HashMap;
use std::sync::Arc;

use crate::algebra::{Algebra, Term, TriplePattern};
use oxirs_core::model::NamedNode;

// ─────────────────────────────────────────────────────────────────────────────
// Trait: SourceSelectivityProvider
// ─────────────────────────────────────────────────────────────────────────────

/// Estimated selectivity of a federated triple pattern at a particular source.
///
/// Lower numbers mean *more selective* (fewer matching results); higher numbers
/// mean *less selective* (more matching results).  The selectivity is used by
/// the cost model to choose between alternative federated sources.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FederatedSelectivity {
    /// Estimated number of matching solutions (cardinality).
    pub estimated_cardinality: f64,
    /// Round-trip latency to the source, in milliseconds.
    pub estimated_latency_ms: f64,
    /// Confidence in the estimate, 0.0–1.0.  1.0 means "from VoID metadata",
    /// 0.5 means "from runtime feedback", 0.1 means "no information".
    pub confidence: f64,
}

impl Default for FederatedSelectivity {
    fn default() -> Self {
        // No information: treat as a moderately costly call.
        Self {
            estimated_cardinality: 1_000.0,
            estimated_latency_ms: 100.0,
            confidence: 0.1,
        }
    }
}

/// Bridge trait between the ARQ planner and the federation runtime.
///
/// Implementations describe (a) which IRI namespaces map to which endpoints
/// and (b) how to estimate per-pattern selectivity at each endpoint.
///
/// `oxirs-federate` provides a concrete impl backed by its
/// `source_selector::SourceSelector`; downstream embedders may also write
/// custom implementations to integrate with proprietary federation registries.
pub trait SourceSelectivityProvider: Send + Sync {
    /// Resolve an IRI to a federated endpoint URL, if any.
    ///
    /// A return value of `None` indicates the IRI is local to the current
    /// dataset and should not be federated.
    fn endpoint_for_iri(&self, iri: &NamedNode) -> Option<String>;

    /// Estimate selectivity of a triple pattern at a given endpoint.
    ///
    /// Implementations may consult VoID descriptions, ASK probes, or runtime
    /// feedback.  The default is a uniform "no information" estimate.
    fn pattern_selectivity(&self, endpoint: &str, pattern: &TriplePattern) -> FederatedSelectivity {
        let _ = (endpoint, pattern);
        FederatedSelectivity::default()
    }

    /// Whether the endpoint should use silent semantics (errors swallowed).
    fn silent_default(&self, endpoint: &str) -> bool {
        let _ = endpoint;
        false
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Static / in-memory provider — used for testing and as a basic registration
// surface for embedders without oxirs-federate.
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight in-memory implementation of [`SourceSelectivityProvider`].
///
/// Each registered entry maps an IRI namespace prefix to an endpoint URL with
/// an associated selectivity estimate.  Patterns whose IRIs share the prefix
/// are rewritten to delegate to that endpoint.
#[derive(Debug, Default, Clone)]
pub struct StaticSourceProvider {
    /// `prefix -> (endpoint, selectivity, silent)` entries.
    /// The longest matching prefix wins.
    entries: Vec<StaticSourceEntry>,
}

#[derive(Debug, Clone)]
struct StaticSourceEntry {
    prefix: String,
    endpoint: String,
    selectivity: FederatedSelectivity,
    silent: bool,
}

impl StaticSourceProvider {
    /// Create an empty provider.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a federated namespace.
    pub fn register(
        &mut self,
        prefix: impl Into<String>,
        endpoint: impl Into<String>,
        selectivity: FederatedSelectivity,
    ) -> &mut Self {
        self.entries.push(StaticSourceEntry {
            prefix: prefix.into(),
            endpoint: endpoint.into(),
            selectivity,
            silent: false,
        });
        self.sort_by_specificity();
        self
    }

    /// Register a federated namespace with `SILENT` semantics enabled.
    ///
    /// Errors from the remote endpoint will be swallowed at execution time.
    pub fn register_silent(
        &mut self,
        prefix: impl Into<String>,
        endpoint: impl Into<String>,
        selectivity: FederatedSelectivity,
    ) -> &mut Self {
        self.entries.push(StaticSourceEntry {
            prefix: prefix.into(),
            endpoint: endpoint.into(),
            selectivity,
            silent: true,
        });
        self.sort_by_specificity();
        self
    }

    /// Number of registered namespaces.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the provider is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn sort_by_specificity(&mut self) {
        // Longest prefix first, so longer prefixes take precedence over shorter
        // ones.  Stable sort preserves registration order for equal-length
        // prefixes.
        self.entries
            .sort_by_key(|entry| std::cmp::Reverse(entry.prefix.len()));
    }

    fn lookup(&self, iri: &str) -> Option<&StaticSourceEntry> {
        self.entries.iter().find(|e| iri.starts_with(&e.prefix))
    }
}

impl SourceSelectivityProvider for StaticSourceProvider {
    fn endpoint_for_iri(&self, iri: &NamedNode) -> Option<String> {
        self.lookup(iri.as_str()).map(|e| e.endpoint.clone())
    }

    fn pattern_selectivity(&self, endpoint: &str, pattern: &TriplePattern) -> FederatedSelectivity {
        // Find the entry whose endpoint matches and the pattern shares an IRI
        // prefix.  If multiple entries map to the same endpoint, take the most
        // specific one (longest prefix); this is the first in entries due to
        // sort_by_specificity.
        for entry in &self.entries {
            if entry.endpoint != endpoint {
                continue;
            }
            if pattern_uses_prefix(pattern, &entry.prefix) {
                return entry.selectivity;
            }
        }
        FederatedSelectivity::default()
    }

    fn silent_default(&self, endpoint: &str) -> bool {
        self.entries
            .iter()
            .find(|e| e.endpoint == endpoint)
            .map(|e| e.silent)
            .unwrap_or(false)
    }
}

fn pattern_uses_prefix(pattern: &TriplePattern, prefix: &str) -> bool {
    pattern_term_uses_prefix(&pattern.subject, prefix)
        || pattern_term_uses_prefix(&pattern.predicate, prefix)
        || pattern_term_uses_prefix(&pattern.object, prefix)
}

fn pattern_term_uses_prefix(term: &Term, prefix: &str) -> bool {
    match term {
        Term::Iri(node) => node.as_str().starts_with(prefix),
        _ => false,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FederatedPlanner — the actual rewriting pass
// ─────────────────────────────────────────────────────────────────────────────

/// Cost-based federation-aware query planner.
///
/// Walks an [`Algebra`] tree and rewrites BGPs whose triple patterns refer to
/// known federated datasets into [`Algebra::Service`] nodes.  Patterns that
/// resolve to multiple possible endpoints are routed to the source with the
/// best estimated cost (cardinality × latency / confidence).
pub struct FederatedPlanner {
    provider: Arc<dyn SourceSelectivityProvider>,
    /// Static cost weight balancing latency vs cardinality.  Higher values
    /// penalize slow endpoints more aggressively.
    latency_weight: f64,
}

impl FederatedPlanner {
    /// Create a new planner using the given source provider.
    pub fn new(provider: Arc<dyn SourceSelectivityProvider>) -> Self {
        Self {
            provider,
            latency_weight: 1.0,
        }
    }

    /// Override the latency cost weight (default 1.0).
    pub fn with_latency_weight(mut self, weight: f64) -> Self {
        self.latency_weight = weight;
        self
    }

    /// Apply the federated planning pass to `algebra`.
    ///
    /// Returns the rewritten algebra plus the set of endpoints touched (useful
    /// for downstream cost accounting and observability).
    pub fn plan(&self, algebra: Algebra) -> FederatedPlanOutcome {
        let mut report = PlanReport::default();
        let rewritten = self.rewrite(algebra, &mut report);
        FederatedPlanOutcome {
            algebra: rewritten,
            endpoints_used: report.endpoints,
            patterns_federated: report.patterns_federated,
        }
    }

    fn rewrite(&self, algebra: Algebra, report: &mut PlanReport) -> Algebra {
        match algebra {
            Algebra::Bgp(patterns) => self.rewrite_bgp(patterns, report),
            Algebra::Join { left, right } => Algebra::Join {
                left: Box::new(self.rewrite(*left, report)),
                right: Box::new(self.rewrite(*right, report)),
            },
            Algebra::LeftJoin {
                left,
                right,
                filter,
            } => Algebra::LeftJoin {
                left: Box::new(self.rewrite(*left, report)),
                right: Box::new(self.rewrite(*right, report)),
                filter,
            },
            Algebra::Union { left, right } => Algebra::Union {
                left: Box::new(self.rewrite(*left, report)),
                right: Box::new(self.rewrite(*right, report)),
            },
            Algebra::Filter { pattern, condition } => Algebra::Filter {
                pattern: Box::new(self.rewrite(*pattern, report)),
                condition,
            },
            Algebra::Extend {
                pattern,
                variable,
                expr,
            } => Algebra::Extend {
                pattern: Box::new(self.rewrite(*pattern, report)),
                variable,
                expr,
            },
            Algebra::Minus { left, right } => Algebra::Minus {
                left: Box::new(self.rewrite(*left, report)),
                right: Box::new(self.rewrite(*right, report)),
            },
            // Existing SERVICE nodes are passed through unchanged — we never
            // re-federate something that already names its endpoint.
            Algebra::Service {
                endpoint,
                pattern,
                silent,
            } => {
                if let Term::Iri(node) = &endpoint {
                    report.endpoints.insert(node.as_str().to_string(), 1);
                }
                Algebra::Service {
                    endpoint,
                    pattern,
                    silent,
                }
            }
            Algebra::Graph { graph, pattern } => Algebra::Graph {
                graph,
                pattern: Box::new(self.rewrite(*pattern, report)),
            },
            Algebra::Project { pattern, variables } => Algebra::Project {
                pattern: Box::new(self.rewrite(*pattern, report)),
                variables,
            },
            Algebra::Distinct { pattern } => Algebra::Distinct {
                pattern: Box::new(self.rewrite(*pattern, report)),
            },
            Algebra::Reduced { pattern } => Algebra::Reduced {
                pattern: Box::new(self.rewrite(*pattern, report)),
            },
            Algebra::Slice {
                pattern,
                offset,
                limit,
            } => Algebra::Slice {
                pattern: Box::new(self.rewrite(*pattern, report)),
                offset,
                limit,
            },
            Algebra::OrderBy {
                pattern,
                conditions,
            } => Algebra::OrderBy {
                pattern: Box::new(self.rewrite(*pattern, report)),
                conditions,
            },
            Algebra::Group {
                pattern,
                variables,
                aggregates,
            } => Algebra::Group {
                pattern: Box::new(self.rewrite(*pattern, report)),
                variables,
                aggregates,
            },
            Algebra::Having { pattern, condition } => Algebra::Having {
                pattern: Box::new(self.rewrite(*pattern, report)),
                condition,
            },
            // PropertyPath patterns are kept local — federation of property
            // paths requires endpoint capability negotiation that the
            // SourceSelectivityProvider abstraction does not currently model.
            other => other,
        }
    }

    fn rewrite_bgp(&self, patterns: Vec<TriplePattern>, report: &mut PlanReport) -> Algebra {
        if patterns.is_empty() {
            return Algebra::Bgp(patterns);
        }

        // Group patterns by the endpoint we want to ship them to.  Patterns
        // whose IRIs do not match any registered namespace stay in the
        // `local` bucket.
        let mut local: Vec<TriplePattern> = Vec::new();
        let mut federated: HashMap<String, Vec<TriplePattern>> = HashMap::new();

        for pattern in patterns {
            match self.choose_endpoint(&pattern) {
                Some(endpoint) => {
                    federated.entry(endpoint).or_default().push(pattern);
                }
                None => local.push(pattern),
            }
        }

        if federated.is_empty() {
            // Nothing to federate — pass through.
            return Algebra::Bgp(local);
        }

        // Build the rewritten algebra: chain Joins of (local BGP) ⋈ (per-endpoint
        // SERVICE BGPs).  Order by endpoint cost so the cheapest one is
        // joined first, giving the executor more pruning leverage.
        let mut services: Vec<(String, Vec<TriplePattern>, f64)> = federated
            .into_iter()
            .map(|(endpoint, patterns)| {
                let cost = self.endpoint_cost(&endpoint, &patterns);
                (endpoint, patterns, cost)
            })
            .collect();
        services.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Track patterns_federated for the outcome report.
        report.patterns_federated += services.iter().map(|(_, p, _)| p.len()).sum::<usize>();
        for (endpoint, patterns, _) in &services {
            *report.endpoints.entry(endpoint.clone()).or_insert(0) += patterns.len();
        }

        let mut iter = services.into_iter().map(|(endpoint, patterns, _)| {
            let endpoint_node = NamedNode::new_unchecked(&endpoint);
            let silent = self.provider.silent_default(&endpoint);
            Algebra::Service {
                endpoint: Term::Iri(endpoint_node),
                pattern: Box::new(Algebra::Bgp(patterns)),
                silent,
            }
        });

        // Start with the cheapest service node (or local BGP, if any patterns
        // remained local), then join the rest in cost-ascending order.
        let mut current = if local.is_empty() {
            // SAFETY: we already checked federated.is_empty() above and
            // returned, so iter has at least one element.
            match iter.next() {
                Some(node) => node,
                None => Algebra::Bgp(Vec::new()),
            }
        } else {
            Algebra::Bgp(local)
        };

        for service in iter {
            current = Algebra::Join {
                left: Box::new(current),
                right: Box::new(service),
            };
        }

        current
    }

    fn choose_endpoint(&self, pattern: &TriplePattern) -> Option<String> {
        // Pick whichever IRI in the triple has the most specific endpoint
        // mapping.  Prefer predicate > subject > object since predicates are
        // typically the most discriminating.
        for term in [&pattern.predicate, &pattern.subject, &pattern.object] {
            if let Term::Iri(node) = term {
                if let Some(endpoint) = self.provider.endpoint_for_iri(node) {
                    return Some(endpoint);
                }
            }
        }
        None
    }

    fn endpoint_cost(&self, endpoint: &str, patterns: &[TriplePattern]) -> f64 {
        // Aggregate selectivity over all patterns shipped to this endpoint.
        // Cost = Σ(cardinality) + latency_weight × max(latency).
        let mut total_card = 0.0;
        let mut max_latency = 0.0f64;
        let mut min_confidence = 1.0f64;
        for pattern in patterns {
            let sel = self.provider.pattern_selectivity(endpoint, pattern);
            total_card += sel.estimated_cardinality;
            max_latency = max_latency.max(sel.estimated_latency_ms);
            min_confidence = min_confidence.min(sel.confidence);
        }
        // Adjust by inverse of confidence: low-confidence estimates carry a
        // small penalty so the planner prefers known-good sources.
        let confidence_penalty = if min_confidence > 0.0 {
            1.0 / min_confidence
        } else {
            10.0
        };
        (total_card + self.latency_weight * max_latency) * confidence_penalty.sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Outcome / report
// ─────────────────────────────────────────────────────────────────────────────

/// Result of running [`FederatedPlanner::plan`].
#[derive(Debug, Clone)]
pub struct FederatedPlanOutcome {
    /// The rewritten algebra.
    pub algebra: Algebra,
    /// Map of endpoint URL → number of patterns shipped to it.
    pub endpoints_used: HashMap<String, usize>,
    /// Total number of triple patterns federated (sum across endpoints).
    pub patterns_federated: usize,
}

impl FederatedPlanOutcome {
    /// Whether the rewrite emitted any federated nodes.
    pub fn touched_federation(&self) -> bool {
        self.patterns_federated > 0
    }
}

#[derive(Debug, Default)]
struct PlanReport {
    endpoints: HashMap<String, usize>,
    patterns_federated: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Term, TriplePattern, Variable};

    fn iri(s: &str) -> Term {
        Term::Iri(NamedNode::new_unchecked(s))
    }

    fn var(name: &str) -> Term {
        Term::Variable(Variable::new(name).expect("valid variable name"))
    }

    fn pattern(s: Term, p: Term, o: Term) -> TriplePattern {
        TriplePattern {
            subject: s,
            predicate: p,
            object: o,
        }
    }

    fn dbpedia_provider() -> StaticSourceProvider {
        let mut provider = StaticSourceProvider::new();
        provider.register(
            "http://dbpedia.org/",
            "https://dbpedia.org/sparql",
            FederatedSelectivity {
                estimated_cardinality: 100.0,
                estimated_latency_ms: 80.0,
                confidence: 0.9,
            },
        );
        provider
    }

    #[test]
    fn test_pattern_with_known_predicate_emits_service() {
        let provider = Arc::new(dbpedia_provider());
        let planner = FederatedPlanner::new(provider);

        let alg = Algebra::Bgp(vec![pattern(
            var("s"),
            iri("http://dbpedia.org/property/birthDate"),
            var("o"),
        )]);

        let outcome = planner.plan(alg);
        assert!(outcome.touched_federation());
        assert_eq!(outcome.patterns_federated, 1);
        assert!(outcome
            .endpoints_used
            .contains_key("https://dbpedia.org/sparql"));
        match outcome.algebra {
            Algebra::Service { endpoint, .. } => match endpoint {
                Term::Iri(node) => assert_eq!(node.as_str(), "https://dbpedia.org/sparql"),
                other => panic!("expected IRI endpoint, got {other:?}"),
            },
            other => panic!("expected Service node, got {other:?}"),
        }
    }

    #[test]
    fn test_local_only_pattern_passes_through() {
        let provider = Arc::new(dbpedia_provider());
        let planner = FederatedPlanner::new(provider);

        let alg = Algebra::Bgp(vec![pattern(
            iri("http://example.org/local/alice"),
            iri("http://example.org/local/knows"),
            var("friend"),
        )]);

        let outcome = planner.plan(alg.clone());
        assert!(!outcome.touched_federation());
        assert_eq!(outcome.algebra, alg);
    }

    #[test]
    fn test_mixed_local_and_federated_emits_join() {
        let provider = Arc::new(dbpedia_provider());
        let planner = FederatedPlanner::new(provider);

        let alg = Algebra::Bgp(vec![
            pattern(
                var("s"),
                iri("http://example.org/local/labelOf"),
                var("label"),
            ),
            pattern(
                var("s"),
                iri("http://dbpedia.org/property/birthDate"),
                var("date"),
            ),
        ]);

        let outcome = planner.plan(alg);
        assert!(outcome.touched_federation());
        assert_eq!(outcome.patterns_federated, 1);
        // Result should be Join { local-BGP, Service { ... } }
        match outcome.algebra {
            Algebra::Join { left, right } => {
                assert!(matches!(*left, Algebra::Bgp(_)));
                assert!(matches!(*right, Algebra::Service { .. }));
            }
            other => panic!("expected Join, got {other:?}"),
        }
    }

    #[test]
    fn test_two_federations_join_in_cost_order() {
        let mut provider = StaticSourceProvider::new();
        // Cheaper source: small cardinality, low latency.
        provider.register(
            "http://cheap.example/",
            "https://cheap.example/sparql",
            FederatedSelectivity {
                estimated_cardinality: 10.0,
                estimated_latency_ms: 20.0,
                confidence: 0.9,
            },
        );
        // Expensive source: large cardinality, high latency.
        provider.register(
            "http://pricy.example/",
            "https://pricy.example/sparql",
            FederatedSelectivity {
                estimated_cardinality: 10_000.0,
                estimated_latency_ms: 500.0,
                confidence: 0.9,
            },
        );

        let planner = FederatedPlanner::new(Arc::new(provider));

        let alg = Algebra::Bgp(vec![
            pattern(
                var("s"),
                iri("http://pricy.example/data#predicate"),
                var("o1"),
            ),
            pattern(
                var("s"),
                iri("http://cheap.example/data#predicate"),
                var("o2"),
            ),
        ]);

        let outcome = planner.plan(alg);
        assert_eq!(outcome.patterns_federated, 2);

        // Top-level should be a Join. Cheaper endpoint (lowest cost) should
        // be the left operand.
        match outcome.algebra {
            Algebra::Join { left, right } => {
                let extract_endpoint = |alg: &Algebra| -> Option<String> {
                    if let Algebra::Service {
                        endpoint: Term::Iri(node),
                        ..
                    } = alg
                    {
                        return Some(node.as_str().to_string());
                    }
                    None
                };
                let left_ep = extract_endpoint(&left);
                let right_ep = extract_endpoint(&right);
                assert_eq!(left_ep.as_deref(), Some("https://cheap.example/sparql"));
                assert_eq!(right_ep.as_deref(), Some("https://pricy.example/sparql"));
            }
            other => panic!("expected Join, got {other:?}"),
        }
    }

    #[test]
    fn test_silent_default_propagates() {
        let mut provider = StaticSourceProvider::new();
        provider.register_silent(
            "http://flaky.example/",
            "https://flaky.example/sparql",
            FederatedSelectivity::default(),
        );
        let planner = FederatedPlanner::new(Arc::new(provider));

        let alg = Algebra::Bgp(vec![pattern(
            var("s"),
            iri("http://flaky.example/data#p"),
            var("o"),
        )]);

        let outcome = planner.plan(alg);
        match outcome.algebra {
            Algebra::Service { silent, .. } => assert!(silent),
            other => panic!("expected Service, got {other:?}"),
        }
    }

    #[test]
    fn test_pre_existing_service_passes_through() {
        let provider = Arc::new(dbpedia_provider());
        let planner = FederatedPlanner::new(provider);

        let original = Algebra::Service {
            endpoint: iri("https://other.example/sparql"),
            pattern: Box::new(Algebra::Bgp(vec![pattern(
                var("s"),
                iri("http://example.org/local/p"),
                var("o"),
            )])),
            silent: true,
        };

        let outcome = planner.plan(original.clone());
        assert_eq!(outcome.algebra, original);
        assert!(outcome
            .endpoints_used
            .contains_key("https://other.example/sparql"));
    }

    #[test]
    fn test_recursive_into_filter() {
        let provider = Arc::new(dbpedia_provider());
        let planner = FederatedPlanner::new(provider);

        let inner = Algebra::Bgp(vec![pattern(
            var("s"),
            iri("http://dbpedia.org/property/birthDate"),
            var("o"),
        )]);
        let alg = Algebra::Filter {
            pattern: Box::new(inner),
            condition: crate::algebra::Expression::Variable(
                Variable::new("o").expect("valid variable"),
            ),
        };

        let outcome = planner.plan(alg);
        assert!(outcome.touched_federation());
        assert!(matches!(
            outcome.algebra,
            Algebra::Filter {
                pattern: _,
                condition: _,
            }
        ));
    }

    #[test]
    fn test_static_source_provider_longest_prefix_wins() {
        let mut provider = StaticSourceProvider::new();
        provider.register(
            "http://example.org/",
            "https://wide.example/sparql",
            FederatedSelectivity::default(),
        );
        provider.register(
            "http://example.org/specific/",
            "https://narrow.example/sparql",
            FederatedSelectivity::default(),
        );

        let iri_specific = NamedNode::new_unchecked("http://example.org/specific/foo");
        let iri_wide = NamedNode::new_unchecked("http://example.org/foo");
        assert_eq!(
            provider.endpoint_for_iri(&iri_specific).as_deref(),
            Some("https://narrow.example/sparql")
        );
        assert_eq!(
            provider.endpoint_for_iri(&iri_wide).as_deref(),
            Some("https://wide.example/sparql")
        );
    }

    #[test]
    fn test_outcome_default() {
        let outcome = FederatedPlanOutcome {
            algebra: Algebra::Bgp(vec![]),
            endpoints_used: HashMap::new(),
            patterns_federated: 0,
        };
        assert!(!outcome.touched_federation());
    }
}
