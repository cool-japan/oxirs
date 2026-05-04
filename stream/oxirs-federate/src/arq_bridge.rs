//! Bridge between [`oxirs_arq`] and the [`crate::source_selector::SourceSelector`].
//!
//! This module exposes [`ArqSourceSelectivityProvider`], a thin adapter that
//! wraps a [`SourceSelector`] (or any other in-process registry of federated
//! sources) and implements
//! [`oxirs_arq::optimizer::federated_plan::SourceSelectivityProvider`] so that
//! the ARQ optimizer can consult `oxirs-federate` for federation-aware query
//! rewriting.
//!
//! # Usage
//!
//! 1. Build and populate a [`SourceSelector`] with [`FederatedSource`] entries
//!    that include VoID metadata (`property_partitions` and/or `uri_spaces`).
//! 2. Wrap it in [`ArqSourceSelectivityProvider`].
//! 3. Hand the wrapper to [`oxirs_arq::optimizer::Optimizer::with_federated_planner`].
//!
//! ```no_run
//! use std::sync::Arc;
//! use oxirs_arq::optimizer::{Optimizer, OptimizerConfig};
//! use oxirs_arq::optimizer::federated_plan::SourceSelectivityProvider;
//! use oxirs_federate::arq_bridge::ArqSourceSelectivityProvider;
//! use oxirs_federate::source_selector::SourceSelector;
//!
//! let mut selector = SourceSelector::new();
//! // ... selector.register_source(...);
//! let provider: Arc<dyn SourceSelectivityProvider> =
//!     Arc::new(ArqSourceSelectivityProvider::new(Arc::new(selector)));
//! let _optimizer = Optimizer::new(OptimizerConfig::default())
//!     .with_federated_planner(provider);
//! ```
//!
//! # Snapshot semantics
//!
//! The bridge takes an `Arc<SourceSelector>` snapshot.  Mutations to the
//! underlying selector after wrapping are **not** observed — callers should
//! register all sources before constructing the bridge, or wrap the selector
//! in an `Arc<RwLock<…>>` and build a custom provider.

use std::sync::Arc;

use oxirs_arq::algebra::{Term, TriplePattern as ArqTriplePattern};
use oxirs_arq::optimizer::federated_plan::{FederatedSelectivity, SourceSelectivityProvider};
use oxirs_core::model::NamedNode;

use crate::source_selector::{
    FederatedSource, SourceSelector, TriplePattern as FederateTriplePattern,
};

/// Adapter that exposes a [`SourceSelector`] as a
/// [`SourceSelectivityProvider`] for the ARQ optimizer.
///
/// See the [module-level documentation](crate::arq_bridge) for usage.
pub struct ArqSourceSelectivityProvider {
    selector: Arc<SourceSelector>,
    /// Latency floor in milliseconds.  Used to avoid emitting confidence
    /// scores for endpoints that have never been probed (`avg_latency_ms == 0`).
    latency_floor_ms: f64,
    /// Whether to emit `SILENT` semantics by default for federated SERVICE nodes.
    silent_default: bool,
}

impl ArqSourceSelectivityProvider {
    /// Wrap a [`SourceSelector`] snapshot.
    ///
    /// `selector` is held by `Arc` so multiple optimizers (or threads) can
    /// share the same view.
    pub fn new(selector: Arc<SourceSelector>) -> Self {
        Self {
            selector,
            latency_floor_ms: 50.0,
            silent_default: false,
        }
    }

    /// Set the floor used when an endpoint reports `avg_latency_ms == 0`
    /// (i.e. never observed).  Defaults to 50ms.
    pub fn with_latency_floor_ms(mut self, floor: f64) -> Self {
        self.latency_floor_ms = floor;
        self
    }

    /// Mark all federated calls as `SILENT` by default.
    ///
    /// Useful when the embedder wants federation to fail open (ignore
    /// unreachable endpoints) instead of failing closed.
    pub fn with_silent_default(mut self, silent: bool) -> Self {
        self.silent_default = silent;
        self
    }

    /// Whether the underlying selector has any registered sources.
    pub fn is_empty(&self) -> bool {
        self.selector.source_count() == 0
    }

    /// Borrow the wrapped selector.
    pub fn selector(&self) -> &SourceSelector {
        &self.selector
    }

    /// Find the best-scoring source whose VoID description covers `iri`.
    fn best_source_for_iri(&self, iri: &str) -> Option<&FederatedSource> {
        let mut best: Option<(&FederatedSource, f64)> = None;
        for id in self.selector.source_ids() {
            let Some(source) = self.selector.get_source(&id) else {
                continue;
            };
            if !source.enabled {
                continue;
            }
            if !self.selector.is_eligible(&source.id) {
                continue;
            }
            if !iri_matches_source(iri, source) {
                continue;
            }
            let score = self.selector.score_source(source);
            match best {
                Some((_, best_score)) if score <= best_score => {}
                _ => best = Some((source, score)),
            }
        }
        best.map(|(s, _)| s)
    }

    /// Find the source that owns a given endpoint URL, if any.
    fn source_for_endpoint(&self, endpoint: &str) -> Option<&FederatedSource> {
        for id in self.selector.source_ids() {
            let Some(source) = self.selector.get_source(&id) else {
                continue;
            };
            if source.endpoint_url == endpoint {
                return Some(source);
            }
        }
        None
    }

    /// Convert an ARQ pattern into the federate TriplePattern shape.
    fn convert_pattern(pattern: &ArqTriplePattern) -> FederateTriplePattern {
        FederateTriplePattern {
            subject: term_iri_or_none(&pattern.subject),
            predicate: term_iri_or_none(&pattern.predicate),
            object: term_iri_or_none(&pattern.object),
        }
    }

    /// Compute estimated cardinality from a source's VoID description, falling
    /// back to a sensible default when no metadata is available.
    fn estimate_cardinality(source: &FederatedSource, pattern: &ArqTriplePattern) -> f64 {
        let void = match source.void_description.as_ref() {
            Some(v) => v,
            None => return 1_000.0,
        };

        // Predicate-bound: prefer property partition counts.
        if let Term::Iri(node) = &pattern.predicate {
            if let Some(count) = void.property_partitions.get(node.as_str()) {
                return (*count as f64).max(1.0);
            }
        }
        // Class-typed objects (rdf:type): prefer class partition counts.
        if matches_predicate(
            &pattern.predicate,
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        ) {
            if let Term::Iri(node) = &pattern.object {
                if let Some(count) = void.class_partitions.get(node.as_str()) {
                    return (*count as f64).max(1.0);
                }
            }
        }
        // Otherwise use total triple count, scaled down for a plausibly-rare
        // pattern.  (Without metadata about unbound variables this is a
        // heuristic; the optimizer treats it as low-confidence.)
        ((void.triples as f64) * 0.01).max(1.0)
    }

    /// Confidence: 0.9 with VoID metadata, 0.5 with VoID partial coverage,
    /// 0.1 with no metadata.
    fn estimate_confidence(source: &FederatedSource, pattern: &ArqTriplePattern) -> f64 {
        let Some(void) = source.void_description.as_ref() else {
            return 0.1;
        };
        if let Term::Iri(node) = &pattern.predicate {
            if void.property_partitions.contains_key(node.as_str()) {
                return 0.9;
            }
        }
        if matches_predicate(
            &pattern.predicate,
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        ) {
            if let Term::Iri(node) = &pattern.object {
                if void.class_partitions.contains_key(node.as_str()) {
                    return 0.9;
                }
            }
        }
        // Has VoID but pattern doesn't hit a partition — uri_spaces match only.
        for term in [&pattern.subject, &pattern.predicate, &pattern.object] {
            if let Term::Iri(node) = term {
                if void.covers_iri(node.as_str()) {
                    return 0.5;
                }
            }
        }
        0.1
    }
}

impl SourceSelectivityProvider for ArqSourceSelectivityProvider {
    fn endpoint_for_iri(&self, iri: &NamedNode) -> Option<String> {
        self.best_source_for_iri(iri.as_str())
            .map(|s| s.endpoint_url.clone())
    }

    fn pattern_selectivity(
        &self,
        endpoint: &str,
        pattern: &ArqTriplePattern,
    ) -> FederatedSelectivity {
        let Some(source) = self.source_for_endpoint(endpoint) else {
            // Endpoint not registered — bridge fell out of sync with planner.
            // Return the trait default so the planner doesn't crash.
            return FederatedSelectivity::default();
        };

        // Sanity-check that this source actually claims to answer the pattern.
        let federate_pat = Self::convert_pattern(pattern);
        let _matches = self
            .selector
            .probe_source(&source.id, &federate_pat)
            .map(|r| r.pattern_matched)
            .unwrap_or(false);

        let estimated_cardinality = Self::estimate_cardinality(source, pattern);
        let confidence = Self::estimate_confidence(source, pattern);
        let estimated_latency_ms = if source.metrics.avg_latency_ms > 0.0 {
            source.metrics.avg_latency_ms
        } else {
            self.latency_floor_ms
        };

        FederatedSelectivity {
            estimated_cardinality,
            estimated_latency_ms,
            confidence,
        }
    }

    fn silent_default(&self, endpoint: &str) -> bool {
        // If the endpoint's reachability metric is false, prefer SILENT so
        // optimizer-emitted SERVICE nodes don't fail the entire query.
        if let Some(source) = self.source_for_endpoint(endpoint) {
            if !source.metrics.is_reachable {
                return true;
            }
        }
        self.silent_default
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn term_iri_or_none(term: &Term) -> Option<String> {
    match term {
        Term::Iri(node) => Some(node.as_str().to_string()),
        _ => None,
    }
}

fn matches_predicate(term: &Term, expected: &str) -> bool {
    matches!(term, Term::Iri(node) if node.as_str() == expected)
}

/// Whether the given IRI is "owned" by `source` according to its VoID metadata.
///
/// Matches in priority order:
/// 1. Direct hit in `property_partitions`
/// 2. Direct hit in `class_partitions`
/// 3. Prefix coverage via `uri_spaces`
fn iri_matches_source(iri: &str, source: &FederatedSource) -> bool {
    let Some(void) = source.void_description.as_ref() else {
        return false;
    };
    if void.property_partitions.contains_key(iri) {
        return true;
    }
    if void.class_partitions.contains_key(iri) {
        return true;
    }
    void.covers_iri(iri)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source_selector::{
        SourceCapabilities, SourceMetrics, SparqlVersion, VoidDescription,
    };
    use oxirs_arq::algebra::{TriplePattern, Variable};

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

    fn dbpedia_source() -> FederatedSource {
        let mut void = VoidDescription::new("http://dbpedia.org/dataset", 1_000_000_000);
        void.property_partitions
            .insert("http://dbpedia.org/property/birthDate".to_string(), 100_000);
        void.property_partitions
            .insert("http://dbpedia.org/ontology/Person".to_string(), 5_000_000);
        void.uri_spaces.push("http://dbpedia.org/".to_string());

        FederatedSource {
            id: "dbpedia".to_string(),
            endpoint_url: "https://dbpedia.org/sparql".to_string(),
            label: Some("DBpedia".to_string()),
            capabilities: SourceCapabilities {
                sparql_version: SparqlVersion::V1_1,
                supports_named_graphs: true,
                supports_update: false,
                supports_federation: true,
                supports_text_search: false,
                supports_geosparql: false,
                named_graphs: vec![],
            },
            void_description: Some(void),
            metrics: SourceMetrics {
                avg_latency_ms: 80.0,
                success_rate: 0.99,
                freshness_ms: 60_000,
                queries_processed: 100,
                is_reachable: true,
            },
            priority: 0,
            enabled: true,
        }
    }

    fn wikidata_source() -> FederatedSource {
        let mut void = VoidDescription::new("http://www.wikidata.org/dataset", 10_000_000_000);
        void.property_partitions
            .insert("http://www.wikidata.org/prop/P31".to_string(), 50_000_000);
        void.uri_spaces.push("http://www.wikidata.org/".to_string());

        FederatedSource {
            id: "wikidata".to_string(),
            endpoint_url: "https://query.wikidata.org/sparql".to_string(),
            label: Some("Wikidata".to_string()),
            capabilities: SourceCapabilities::default(),
            void_description: Some(void),
            metrics: SourceMetrics {
                avg_latency_ms: 200.0,
                success_rate: 0.95,
                freshness_ms: 30_000,
                queries_processed: 50,
                is_reachable: true,
            },
            priority: 0,
            enabled: true,
        }
    }

    fn build_selector() -> Arc<SourceSelector> {
        let mut sel = SourceSelector::new();
        sel.register_source(dbpedia_source())
            .expect("register dbpedia");
        sel.register_source(wikidata_source())
            .expect("register wikidata");
        Arc::new(sel)
    }

    #[test]
    fn endpoint_for_iri_routes_to_dbpedia() {
        let provider = ArqSourceSelectivityProvider::new(build_selector());
        let node = NamedNode::new_unchecked("http://dbpedia.org/property/birthDate");
        let endpoint = provider
            .endpoint_for_iri(&node)
            .expect("dbpedia property must route");
        assert_eq!(endpoint, "https://dbpedia.org/sparql");
    }

    #[test]
    fn endpoint_for_iri_routes_to_wikidata() {
        let provider = ArqSourceSelectivityProvider::new(build_selector());
        let node = NamedNode::new_unchecked("http://www.wikidata.org/prop/P31");
        let endpoint = provider
            .endpoint_for_iri(&node)
            .expect("wikidata property must route");
        assert_eq!(endpoint, "https://query.wikidata.org/sparql");
    }

    #[test]
    fn endpoint_for_iri_returns_none_for_unknown_iri() {
        let provider = ArqSourceSelectivityProvider::new(build_selector());
        let node = NamedNode::new_unchecked("http://example.org/local/foo");
        assert!(provider.endpoint_for_iri(&node).is_none());
    }

    #[test]
    fn pattern_selectivity_uses_property_partition_count() {
        let provider = ArqSourceSelectivityProvider::new(build_selector());
        let pat = pattern(
            var("s"),
            iri("http://dbpedia.org/property/birthDate"),
            var("o"),
        );
        let sel = provider.pattern_selectivity("https://dbpedia.org/sparql", &pat);
        assert_eq!(sel.estimated_cardinality, 100_000.0);
        assert_eq!(sel.confidence, 0.9);
        assert_eq!(sel.estimated_latency_ms, 80.0);
    }

    #[test]
    fn pattern_selectivity_falls_back_to_total_triples_when_predicate_unknown() {
        let provider = ArqSourceSelectivityProvider::new(build_selector());
        let pat = pattern(
            var("s"),
            iri("http://dbpedia.org/property/unmapped"),
            var("o"),
        );
        let sel = provider.pattern_selectivity("https://dbpedia.org/sparql", &pat);
        // 1B * 0.01 = 10M
        assert!(sel.estimated_cardinality > 1_000_000.0);
        // uri_spaces covers it → 0.5 confidence
        assert_eq!(sel.confidence, 0.5);
    }

    #[test]
    fn pattern_selectivity_for_unknown_endpoint_returns_default() {
        let provider = ArqSourceSelectivityProvider::new(build_selector());
        let pat = pattern(
            var("s"),
            iri("http://dbpedia.org/property/birthDate"),
            var("o"),
        );
        let sel = provider.pattern_selectivity("https://unknown.example/sparql", &pat);
        let default = FederatedSelectivity::default();
        assert_eq!(sel.estimated_cardinality, default.estimated_cardinality);
        assert_eq!(sel.confidence, default.confidence);
    }

    #[test]
    fn silent_default_returns_true_for_unreachable_source() {
        let mut sel = SourceSelector::new();
        let mut src = dbpedia_source();
        src.metrics.is_reachable = false;
        sel.register_source(src).expect("register dbpedia");
        let provider = ArqSourceSelectivityProvider::new(Arc::new(sel));
        assert!(provider.silent_default("https://dbpedia.org/sparql"));
    }

    #[test]
    fn silent_default_respects_with_silent_default_flag() {
        let provider =
            ArqSourceSelectivityProvider::new(build_selector()).with_silent_default(true);
        // Reachable endpoint → uses configured default.
        assert!(provider.silent_default("https://dbpedia.org/sparql"));
    }

    #[test]
    fn silent_default_false_by_default_for_reachable_source() {
        let provider = ArqSourceSelectivityProvider::new(build_selector());
        assert!(!provider.silent_default("https://dbpedia.org/sparql"));
    }

    #[test]
    fn provider_is_empty_when_selector_has_no_sources() {
        let sel = Arc::new(SourceSelector::new());
        let provider = ArqSourceSelectivityProvider::new(sel);
        assert!(provider.is_empty());
    }

    #[test]
    fn rdf_type_pattern_uses_class_partition_count() {
        let mut void = VoidDescription::new("http://example.org/dataset", 100_000);
        void.class_partitions
            .insert("http://example.org/Person".to_string(), 12_345);
        void.uri_spaces.push("http://example.org/".to_string());

        let mut sel = SourceSelector::new();
        sel.register_source(FederatedSource {
            id: "ex".to_string(),
            endpoint_url: "http://ex.example/sparql".to_string(),
            label: None,
            capabilities: SourceCapabilities::default(),
            void_description: Some(void),
            metrics: SourceMetrics {
                avg_latency_ms: 30.0,
                success_rate: 1.0,
                freshness_ms: 0,
                queries_processed: 0,
                is_reachable: true,
            },
            priority: 0,
            enabled: true,
        })
        .expect("register ex");
        let provider = ArqSourceSelectivityProvider::new(Arc::new(sel));

        let pat = pattern(
            var("s"),
            iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            iri("http://example.org/Person"),
        );
        let selectivity = provider.pattern_selectivity("http://ex.example/sparql", &pat);
        assert_eq!(selectivity.estimated_cardinality, 12_345.0);
        assert_eq!(selectivity.confidence, 0.9);
    }

    #[test]
    fn latency_floor_replaces_zero_latency() {
        let mut src = dbpedia_source();
        src.metrics.avg_latency_ms = 0.0; // never observed
        let mut sel = SourceSelector::new();
        sel.register_source(src).expect("register");
        let provider =
            ArqSourceSelectivityProvider::new(Arc::new(sel)).with_latency_floor_ms(150.0);

        let pat = pattern(
            var("s"),
            iri("http://dbpedia.org/property/birthDate"),
            var("o"),
        );
        let selectivity = provider.pattern_selectivity("https://dbpedia.org/sparql", &pat);
        assert_eq!(selectivity.estimated_latency_ms, 150.0);
    }

    #[test]
    fn excluded_source_not_routed() {
        let mut sel = SourceSelector::new();
        sel.register_source(dbpedia_source()).expect("register");
        sel.exclude_source("dbpedia");
        let provider = ArqSourceSelectivityProvider::new(Arc::new(sel));
        let node = NamedNode::new_unchecked("http://dbpedia.org/property/birthDate");
        // Excluded sources are filtered out.
        assert!(provider.endpoint_for_iri(&node).is_none());
    }

    #[test]
    fn higher_scoring_source_wins_when_two_match() {
        // Two sources both cover the same uri_space; bridge picks the one
        // with the higher score (lower latency, higher reliability).
        let mut fast_src = dbpedia_source();
        fast_src.id = "fast".to_string();
        fast_src.endpoint_url = "https://fast.example/sparql".to_string();
        fast_src.metrics.avg_latency_ms = 10.0;
        fast_src.metrics.success_rate = 0.999;

        let mut slow_src = dbpedia_source();
        slow_src.id = "slow".to_string();
        slow_src.endpoint_url = "https://slow.example/sparql".to_string();
        slow_src.metrics.avg_latency_ms = 5_000.0;
        slow_src.metrics.success_rate = 0.5;

        let mut sel = SourceSelector::new();
        sel.register_source(fast_src).expect("fast");
        sel.register_source(slow_src).expect("slow");
        let provider = ArqSourceSelectivityProvider::new(Arc::new(sel));

        let node = NamedNode::new_unchecked("http://dbpedia.org/property/birthDate");
        let endpoint = provider
            .endpoint_for_iri(&node)
            .expect("a source must route");
        assert_eq!(endpoint, "https://fast.example/sparql");
    }
}
