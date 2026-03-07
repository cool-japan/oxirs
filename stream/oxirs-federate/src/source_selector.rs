//! Federated source selection for SPARQL query routing.
//!
//! Provides ASK-based source probing, VoID description matching, capability-based
//! routing, source ranking by latency/reliability/freshness, multi-source join
//! planning, source exclusion/inclusion lists, cost-based selection, and source
//! metadata caching.

use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// SPARQL version supported by an endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SparqlVersion {
    /// SPARQL 1.0.
    V1_0,
    /// SPARQL 1.1.
    V1_1,
    /// SPARQL 1.2 draft.
    V1_2,
}

/// Capabilities reported by a federated source.
#[derive(Debug, Clone)]
pub struct SourceCapabilities {
    /// Supported SPARQL version.
    pub sparql_version: SparqlVersion,
    /// Whether the endpoint supports named graphs.
    pub supports_named_graphs: bool,
    /// Whether the endpoint supports SPARQL Update.
    pub supports_update: bool,
    /// Whether the endpoint supports SERVICE federation.
    pub supports_federation: bool,
    /// Whether the endpoint supports full-text search extensions.
    pub supports_text_search: bool,
    /// Whether the endpoint supports GeoSPARQL.
    pub supports_geosparql: bool,
    /// Available named graph IRIs.
    pub named_graphs: Vec<String>,
}

impl Default for SourceCapabilities {
    fn default() -> Self {
        Self {
            sparql_version: SparqlVersion::V1_1,
            supports_named_graphs: false,
            supports_update: false,
            supports_federation: false,
            supports_text_search: false,
            supports_geosparql: false,
            named_graphs: Vec::new(),
        }
    }
}

/// VoID (Vocabulary of Interlinked Datasets) description for a source.
#[derive(Debug, Clone)]
pub struct VoidDescription {
    /// Dataset IRI.
    pub dataset_iri: String,
    /// Number of triples in the dataset.
    pub triples: u64,
    /// Number of distinct subjects.
    pub distinct_subjects: u64,
    /// Number of distinct predicates.
    pub distinct_predicates: u64,
    /// Number of distinct objects.
    pub distinct_objects: u64,
    /// Set of vocabulary IRIs (class and property namespaces) the dataset uses.
    pub vocabularies: HashSet<String>,
    /// URI space(s): IRI prefixes that appear in this dataset.
    pub uri_spaces: Vec<String>,
    /// Class partitions: class IRI -> triple count.
    pub class_partitions: HashMap<String, u64>,
    /// Property partitions: property IRI -> triple count.
    pub property_partitions: HashMap<String, u64>,
}

impl VoidDescription {
    /// Create a minimal VoID description.
    pub fn new(dataset_iri: impl Into<String>, triples: u64) -> Self {
        Self {
            dataset_iri: dataset_iri.into(),
            triples,
            distinct_subjects: 0,
            distinct_predicates: 0,
            distinct_objects: 0,
            vocabularies: HashSet::new(),
            uri_spaces: Vec::new(),
            class_partitions: HashMap::new(),
            property_partitions: HashMap::new(),
        }
    }

    /// Check if this dataset contains the given property.
    pub fn has_property(&self, property_iri: &str) -> bool {
        self.property_partitions.contains_key(property_iri)
    }

    /// Check if this dataset contains the given class.
    pub fn has_class(&self, class_iri: &str) -> bool {
        self.class_partitions.contains_key(class_iri)
    }

    /// Check if any URI space covers the given IRI.
    pub fn covers_iri(&self, iri: &str) -> bool {
        self.uri_spaces.iter().any(|prefix| iri.starts_with(prefix))
    }
}

/// Health and performance metrics for a source.
#[derive(Debug, Clone)]
pub struct SourceMetrics {
    /// Average latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Success rate (0.0 - 1.0).
    pub success_rate: f64,
    /// Data freshness: milliseconds since last data update.
    pub freshness_ms: u64,
    /// Number of queries processed.
    pub queries_processed: u64,
    /// Is the source currently reachable?
    pub is_reachable: bool,
}

impl Default for SourceMetrics {
    fn default() -> Self {
        Self {
            avg_latency_ms: 0.0,
            success_rate: 1.0,
            freshness_ms: 0,
            queries_processed: 0,
            is_reachable: true,
        }
    }
}

/// A registered federated source.
#[derive(Debug, Clone)]
pub struct FederatedSource {
    /// Unique source identifier.
    pub id: String,
    /// SPARQL endpoint URL.
    pub endpoint_url: String,
    /// Human-readable label.
    pub label: Option<String>,
    /// Source capabilities.
    pub capabilities: SourceCapabilities,
    /// VoID description, if available.
    pub void_description: Option<VoidDescription>,
    /// Performance metrics.
    pub metrics: SourceMetrics,
    /// Priority (higher = preferred).
    pub priority: i32,
    /// Whether this source is enabled.
    pub enabled: bool,
}

/// Result of an ASK-based probe.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    /// Source ID that was probed.
    pub source_id: String,
    /// Whether the probe was successful.
    pub success: bool,
    /// Latency observed during the probe (ms).
    pub latency_ms: f64,
    /// Whether the triple pattern matched.
    pub pattern_matched: bool,
}

/// A triple pattern used for source selection.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TriplePattern {
    /// Subject (None = variable).
    pub subject: Option<String>,
    /// Predicate (None = variable).
    pub predicate: Option<String>,
    /// Object (None = variable).
    pub object: Option<String>,
}

/// A source selection result: which sources can answer which patterns.
#[derive(Debug, Clone)]
pub struct SelectionResult {
    /// Pattern -> list of selected source IDs, ordered by score.
    pub pattern_sources: HashMap<usize, Vec<String>>,
    /// Per-source scores used in ranking.
    pub source_scores: HashMap<String, f64>,
    /// Estimated total cost.
    pub estimated_cost: f64,
}

/// Cost model weights for source ranking.
#[derive(Debug, Clone)]
pub struct CostWeights {
    /// Weight for latency (lower is better).
    pub latency_weight: f64,
    /// Weight for reliability (higher is better).
    pub reliability_weight: f64,
    /// Weight for data freshness (lower is better).
    pub freshness_weight: f64,
    /// Weight for data coverage (higher is better).
    pub coverage_weight: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            latency_weight: 0.3,
            reliability_weight: 0.3,
            freshness_weight: 0.2,
            coverage_weight: 0.2,
        }
    }
}

/// Cached source metadata entry.
#[derive(Debug, Clone)]
struct CachedMetadata {
    capabilities: SourceCapabilities,
    void_description: Option<VoidDescription>,
    cached_at_ms: u64,
    ttl_ms: u64,
}

impl CachedMetadata {
    fn is_expired(&self, now_ms: u64) -> bool {
        now_ms.saturating_sub(self.cached_at_ms) >= self.ttl_ms
    }
}

/// Errors from source selection operations.
#[derive(Debug)]
pub enum SelectionError {
    /// Source not found.
    SourceNotFound(String),
    /// No sources available for the given pattern.
    NoSourcesAvailable,
    /// Source is excluded.
    SourceExcluded(String),
    /// Invalid pattern.
    InvalidPattern(String),
}

impl std::fmt::Display for SelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectionError::SourceNotFound(id) => write!(f, "source not found: {id}"),
            SelectionError::NoSourcesAvailable => write!(f, "no sources available"),
            SelectionError::SourceExcluded(id) => write!(f, "source is excluded: {id}"),
            SelectionError::InvalidPattern(msg) => write!(f, "invalid pattern: {msg}"),
        }
    }
}

impl std::error::Error for SelectionError {}

// ─────────────────────────────────────────────────────────────────────────────
// SourceSelector
// ─────────────────────────────────────────────────────────────────────────────

/// Federated source selector for SPARQL query routing.
pub struct SourceSelector {
    /// Registered sources.
    sources: HashMap<String, FederatedSource>,
    /// Source exclusion set.
    exclusion_list: HashSet<String>,
    /// Source inclusion set (if non-empty, only these are considered).
    inclusion_list: HashSet<String>,
    /// Cost model weights.
    cost_weights: CostWeights,
    /// Metadata cache: source_id -> cached data.
    metadata_cache: HashMap<String, CachedMetadata>,
    /// Default metadata cache TTL in milliseconds.
    cache_ttl_ms: u64,
}

impl Default for SourceSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceSelector {
    /// Create a new source selector.
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            exclusion_list: HashSet::new(),
            inclusion_list: HashSet::new(),
            cost_weights: CostWeights::default(),
            metadata_cache: HashMap::new(),
            cache_ttl_ms: 300_000, // 5 minutes
        }
    }

    /// Create with custom cost weights.
    pub fn with_weights(weights: CostWeights) -> Self {
        Self {
            cost_weights: weights,
            ..Self::new()
        }
    }

    /// Set the metadata cache TTL.
    pub fn set_cache_ttl_ms(&mut self, ttl: u64) {
        self.cache_ttl_ms = ttl;
    }

    /// Set cost weights.
    pub fn set_weights(&mut self, weights: CostWeights) {
        self.cost_weights = weights;
    }

    /// Get cost weights.
    pub fn weights(&self) -> &CostWeights {
        &self.cost_weights
    }

    // ─── Source Registration ─────────────────────────────────────────────

    /// Register a source.
    pub fn register_source(&mut self, source: FederatedSource) -> Result<(), SelectionError> {
        if source.endpoint_url.is_empty() {
            return Err(SelectionError::InvalidPattern(
                "empty endpoint URL".to_string(),
            ));
        }
        self.sources.insert(source.id.clone(), source);
        Ok(())
    }

    /// Remove a source by ID.
    pub fn remove_source(&mut self, id: &str) -> bool {
        self.metadata_cache.remove(id);
        self.sources.remove(id).is_some()
    }

    /// Get a source by ID.
    pub fn get_source(&self, id: &str) -> Option<&FederatedSource> {
        self.sources.get(id)
    }

    /// Number of registered sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// List all source IDs.
    pub fn source_ids(&self) -> Vec<String> {
        self.sources.keys().cloned().collect()
    }

    // ─── Exclusion / Inclusion Lists ─────────────────────────────────────

    /// Add a source to the exclusion list.
    pub fn exclude_source(&mut self, id: impl Into<String>) {
        self.exclusion_list.insert(id.into());
    }

    /// Remove a source from the exclusion list.
    pub fn unexclude_source(&mut self, id: &str) {
        self.exclusion_list.remove(id);
    }

    /// Set the inclusion list (if non-empty, only these sources are considered).
    pub fn set_inclusion_list(&mut self, ids: Vec<String>) {
        self.inclusion_list = ids.into_iter().collect();
    }

    /// Clear the inclusion list (all sources considered).
    pub fn clear_inclusion_list(&mut self) {
        self.inclusion_list.clear();
    }

    /// Check if a source is eligible (not excluded, and in inclusion list if set).
    pub fn is_eligible(&self, id: &str) -> bool {
        if self.exclusion_list.contains(id) {
            return false;
        }
        if !self.inclusion_list.is_empty() && !self.inclusion_list.contains(id) {
            return false;
        }
        true
    }

    // ─── ASK-based Probing ───────────────────────────────────────────────

    /// Simulate an ASK-based probe against a source using a triple pattern.
    ///
    /// In a real implementation this would send SPARQL ASK queries to the
    /// endpoint. Here we use VoID metadata to estimate whether the source
    /// can answer the pattern.
    pub fn probe_source(
        &self,
        source_id: &str,
        pattern: &TriplePattern,
    ) -> Result<ProbeResult, SelectionError> {
        let source = self
            .sources
            .get(source_id)
            .ok_or_else(|| SelectionError::SourceNotFound(source_id.to_string()))?;

        let pattern_matched = self.pattern_matches_source(pattern, source);

        Ok(ProbeResult {
            source_id: source_id.to_string(),
            success: source.metrics.is_reachable,
            latency_ms: source.metrics.avg_latency_ms,
            pattern_matched,
        })
    }

    /// Probe all eligible sources for a pattern.
    pub fn probe_all(&self, pattern: &TriplePattern) -> Vec<ProbeResult> {
        self.eligible_sources()
            .iter()
            .filter_map(|s| self.probe_source(&s.id, pattern).ok())
            .collect()
    }

    // ─── VoID Description Matching ───────────────────────────────────────

    /// Find sources that match a triple pattern based on VoID descriptions.
    pub fn void_match(&self, pattern: &TriplePattern) -> Vec<String> {
        self.eligible_sources()
            .iter()
            .filter(|s| self.pattern_matches_source(pattern, s))
            .map(|s| s.id.clone())
            .collect()
    }

    // ─── Capability-based Routing ────────────────────────────────────────

    /// Find sources that support a given SPARQL version.
    pub fn sources_with_sparql_version(&self, version: &SparqlVersion) -> Vec<String> {
        self.eligible_sources()
            .iter()
            .filter(|s| &s.capabilities.sparql_version == version)
            .map(|s| s.id.clone())
            .collect()
    }

    /// Find sources that support named graphs.
    pub fn sources_with_named_graphs(&self) -> Vec<String> {
        self.eligible_sources()
            .iter()
            .filter(|s| s.capabilities.supports_named_graphs)
            .map(|s| s.id.clone())
            .collect()
    }

    /// Find sources that support GeoSPARQL.
    pub fn sources_with_geosparql(&self) -> Vec<String> {
        self.eligible_sources()
            .iter()
            .filter(|s| s.capabilities.supports_geosparql)
            .map(|s| s.id.clone())
            .collect()
    }

    // ─── Source Ranking ──────────────────────────────────────────────────

    /// Compute a score for a source (higher is better).
    pub fn score_source(&self, source: &FederatedSource) -> f64 {
        let w = &self.cost_weights;

        // Latency score: lower latency = higher score. Cap at 10000ms.
        let max_latency = 10_000.0f64;
        let latency_score = 1.0 - (source.metrics.avg_latency_ms / max_latency).min(1.0);

        // Reliability score: direct use of success_rate.
        let reliability_score = source.metrics.success_rate;

        // Freshness score: lower freshness_ms = higher score. Cap at 1 hour.
        let max_freshness = 3_600_000.0f64;
        let freshness_score = 1.0 - (source.metrics.freshness_ms as f64 / max_freshness).min(1.0);

        // Coverage score: based on triple count from VoID, normalised.
        let coverage_score = source
            .void_description
            .as_ref()
            .map(|v| (v.triples as f64).ln().max(0.0) / 25.0) // ln(~72 billion) ~ 25
            .unwrap_or(0.5)
            .min(1.0);

        w.latency_weight * latency_score
            + w.reliability_weight * reliability_score
            + w.freshness_weight * freshness_score
            + w.coverage_weight * coverage_score
    }

    /// Rank all eligible sources (highest score first).
    pub fn rank_sources(&self) -> Vec<(String, f64)> {
        let mut ranked: Vec<(String, f64)> = self
            .eligible_sources()
            .iter()
            .map(|s| (s.id.clone(), self.score_source(s)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    // ─── Cost-based Selection ────────────────────────────────────────────

    /// Select the best source(s) for a set of triple patterns.
    ///
    /// For each pattern, ranks matching sources by score and selects the top one.
    pub fn select_sources(
        &self,
        patterns: &[TriplePattern],
    ) -> Result<SelectionResult, SelectionError> {
        let eligible = self.eligible_sources();
        if eligible.is_empty() {
            return Err(SelectionError::NoSourcesAvailable);
        }

        let mut pattern_sources: HashMap<usize, Vec<String>> = HashMap::new();
        let mut source_scores: HashMap<String, f64> = HashMap::new();

        for (idx, pattern) in patterns.iter().enumerate() {
            let mut matches: Vec<(String, f64)> = eligible
                .iter()
                .filter(|s| self.pattern_matches_source(pattern, s))
                .map(|s| (s.id.clone(), self.score_source(s)))
                .collect();

            // If no VoID-based match, consider all eligible sources.
            if matches.is_empty() {
                matches = eligible
                    .iter()
                    .map(|s| (s.id.clone(), self.score_source(s)))
                    .collect();
            }

            matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let selected_ids: Vec<String> = matches.iter().map(|(id, _)| id.clone()).collect();
            for (id, score) in &matches {
                source_scores
                    .entry(id.clone())
                    .and_modify(|s| {
                        if *score > *s {
                            *s = *score;
                        }
                    })
                    .or_insert(*score);
            }
            pattern_sources.insert(idx, selected_ids);
        }

        let estimated_cost = self.estimate_join_cost(&pattern_sources);

        Ok(SelectionResult {
            pattern_sources,
            source_scores,
            estimated_cost,
        })
    }

    // ─── Multi-source Join Planning ──────────────────────────────────────

    /// Estimate the cost of executing patterns across sources.
    ///
    /// Penalises cross-source joins (patterns handled by different sources).
    fn estimate_join_cost(&self, pattern_sources: &HashMap<usize, Vec<String>>) -> f64 {
        let mut cost = 0.0f64;
        let pattern_count = pattern_sources.len();

        // Base cost: number of patterns.
        cost += pattern_count as f64;

        // Cross-source penalty: count pairs of patterns that have no common source.
        let pattern_ids: Vec<usize> = pattern_sources.keys().copied().collect();
        for i in 0..pattern_ids.len() {
            for j in (i + 1)..pattern_ids.len() {
                let sources_i: HashSet<&String> = pattern_sources
                    .get(&pattern_ids[i])
                    .map(|v| v.iter().take(1).collect())
                    .unwrap_or_default();
                let sources_j: HashSet<&String> = pattern_sources
                    .get(&pattern_ids[j])
                    .map(|v| v.iter().take(1).collect())
                    .unwrap_or_default();
                if sources_i.is_disjoint(&sources_j) {
                    cost += 10.0; // cross-source join penalty
                }
            }
        }

        cost
    }

    // ─── Metadata Caching ────────────────────────────────────────────────

    /// Cache metadata for a source.
    pub fn cache_metadata(
        &mut self,
        source_id: impl Into<String>,
        capabilities: SourceCapabilities,
        void_description: Option<VoidDescription>,
        now_ms: u64,
    ) {
        let id = source_id.into();
        self.metadata_cache.insert(
            id,
            CachedMetadata {
                capabilities,
                void_description,
                cached_at_ms: now_ms,
                ttl_ms: self.cache_ttl_ms,
            },
        );
    }

    /// Get cached metadata for a source, if not expired.
    pub fn get_cached_metadata(
        &self,
        source_id: &str,
        now_ms: u64,
    ) -> Option<(&SourceCapabilities, Option<&VoidDescription>)> {
        self.metadata_cache.get(source_id).and_then(|cached| {
            if cached.is_expired(now_ms) {
                None
            } else {
                Some((&cached.capabilities, cached.void_description.as_ref()))
            }
        })
    }

    /// Evict expired metadata cache entries.
    pub fn evict_expired_cache(&mut self, now_ms: u64) -> usize {
        let before = self.metadata_cache.len();
        self.metadata_cache.retain(|_, v| !v.is_expired(now_ms));
        before - self.metadata_cache.len()
    }

    /// Number of cached metadata entries.
    pub fn cache_size(&self) -> usize {
        self.metadata_cache.len()
    }

    // ─── Update Metrics ──────────────────────────────────────────────────

    /// Update metrics for a source.
    pub fn update_metrics(
        &mut self,
        source_id: &str,
        metrics: SourceMetrics,
    ) -> Result<(), SelectionError> {
        let source = self
            .sources
            .get_mut(source_id)
            .ok_or_else(|| SelectionError::SourceNotFound(source_id.to_string()))?;
        source.metrics = metrics;
        Ok(())
    }

    // ─── Private helpers ─────────────────────────────────────────────────

    fn eligible_sources(&self) -> Vec<&FederatedSource> {
        self.sources
            .values()
            .filter(|s| s.enabled && self.is_eligible(&s.id) && s.metrics.is_reachable)
            .collect()
    }

    fn pattern_matches_source(&self, pattern: &TriplePattern, source: &FederatedSource) -> bool {
        if let Some(ref void) = source.void_description {
            // Check predicate against property partitions.
            if let Some(ref pred) = pattern.predicate {
                if void.has_property(pred) {
                    return true;
                }
            }
            // Check subject against URI spaces.
            if let Some(ref subj) = pattern.subject {
                if void.covers_iri(subj) {
                    return true;
                }
            }
            // Check object against URI spaces.
            if let Some(ref obj) = pattern.object {
                if void.covers_iri(obj) {
                    return true;
                }
            }
            // If all components are variables, consider it a match for non-empty datasets.
            if pattern.subject.is_none()
                && pattern.predicate.is_none()
                && pattern.object.is_none()
                && void.triples > 0
            {
                return true;
            }
            false
        } else {
            // No VoID: assume the source may have the data.
            true
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_source(id: &str, url: &str) -> FederatedSource {
        FederatedSource {
            id: id.to_string(),
            endpoint_url: url.to_string(),
            label: Some(format!("Source {id}")),
            capabilities: SourceCapabilities::default(),
            void_description: None,
            metrics: SourceMetrics::default(),
            priority: 0,
            enabled: true,
        }
    }

    fn make_source_with_void(id: &str, predicates: &[&str], triples: u64) -> FederatedSource {
        let mut void = VoidDescription::new(format!("http://example.org/{id}"), triples);
        for p in predicates {
            void.property_partitions.insert(p.to_string(), triples / 2);
        }
        void.uri_spaces.push(format!("http://example.org/{id}/"));
        FederatedSource {
            id: id.to_string(),
            endpoint_url: format!("http://{id}.example.org/sparql"),
            label: None,
            capabilities: SourceCapabilities::default(),
            void_description: Some(void),
            metrics: SourceMetrics {
                avg_latency_ms: 50.0,
                success_rate: 0.99,
                freshness_ms: 1000,
                queries_processed: 100,
                is_reachable: true,
            },
            priority: 0,
            enabled: true,
        }
    }

    fn make_selector() -> SourceSelector {
        SourceSelector::new()
    }

    // ── Registration Tests ───────────────────────────────────────────────

    #[test]
    fn test_register_source() {
        let mut sel = make_selector();
        let s = make_source("s1", "http://s1.example.org/sparql");
        assert!(sel.register_source(s).is_ok());
        assert_eq!(sel.source_count(), 1);
    }

    #[test]
    fn test_register_empty_url_error() {
        let mut sel = make_selector();
        let s = make_source("s1", "");
        assert!(sel.register_source(s).is_err());
    }

    #[test]
    fn test_remove_source() {
        let mut sel = make_selector();
        sel.register_source(make_source("s1", "http://x")).ok();
        assert!(sel.remove_source("s1"));
        assert_eq!(sel.source_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut sel = make_selector();
        assert!(!sel.remove_source("nope"));
    }

    #[test]
    fn test_get_source() {
        let mut sel = make_selector();
        sel.register_source(make_source("s1", "http://x")).ok();
        assert!(sel.get_source("s1").is_some());
        assert!(sel.get_source("nope").is_none());
    }

    #[test]
    fn test_source_ids() {
        let mut sel = make_selector();
        sel.register_source(make_source("s1", "http://x")).ok();
        sel.register_source(make_source("s2", "http://y")).ok();
        let ids = sel.source_ids();
        assert_eq!(ids.len(), 2);
    }

    // ── Exclusion / Inclusion Tests ──────────────────────────────────────

    #[test]
    fn test_exclude_source() {
        let mut sel = make_selector();
        sel.register_source(make_source("s1", "http://x")).ok();
        sel.exclude_source("s1");
        assert!(!sel.is_eligible("s1"));
    }

    #[test]
    fn test_unexclude_source() {
        let mut sel = make_selector();
        sel.exclude_source("s1");
        sel.unexclude_source("s1");
        assert!(sel.is_eligible("s1"));
    }

    #[test]
    fn test_inclusion_list() {
        let mut sel = make_selector();
        sel.register_source(make_source("s1", "http://x")).ok();
        sel.register_source(make_source("s2", "http://y")).ok();
        sel.set_inclusion_list(vec!["s1".to_string()]);
        assert!(sel.is_eligible("s1"));
        assert!(!sel.is_eligible("s2"));
    }

    #[test]
    fn test_clear_inclusion_list() {
        let mut sel = make_selector();
        sel.set_inclusion_list(vec!["s1".to_string()]);
        sel.clear_inclusion_list();
        assert!(sel.is_eligible("s2"));
    }

    #[test]
    fn test_exclusion_overrides_inclusion() {
        let mut sel = make_selector();
        sel.set_inclusion_list(vec!["s1".to_string()]);
        sel.exclude_source("s1");
        assert!(!sel.is_eligible("s1"));
    }

    // ── VoID Description Tests ───────────────────────────────────────────

    #[test]
    fn test_void_has_property() {
        let void = VoidDescription {
            dataset_iri: "http://ds".to_string(),
            triples: 1000,
            distinct_subjects: 0,
            distinct_predicates: 0,
            distinct_objects: 0,
            vocabularies: HashSet::new(),
            uri_spaces: Vec::new(),
            class_partitions: HashMap::new(),
            property_partitions: {
                let mut m = HashMap::new();
                m.insert("http://schema.org/name".to_string(), 500);
                m
            },
        };
        assert!(void.has_property("http://schema.org/name"));
        assert!(!void.has_property("http://schema.org/age"));
    }

    #[test]
    fn test_void_covers_iri() {
        let mut void = VoidDescription::new("http://ds", 1000);
        void.uri_spaces.push("http://example.org/data/".to_string());
        assert!(void.covers_iri("http://example.org/data/person/1"));
        assert!(!void.covers_iri("http://other.org/data/1"));
    }

    #[test]
    fn test_void_match() {
        let mut sel = make_selector();
        sel.register_source(make_source_with_void(
            "s1",
            &["http://schema.org/name"],
            1000,
        ))
        .ok();
        sel.register_source(make_source_with_void(
            "s2",
            &["http://schema.org/age"],
            2000,
        ))
        .ok();

        let pattern = TriplePattern {
            subject: None,
            predicate: Some("http://schema.org/name".to_string()),
            object: None,
        };
        let matches = sel.void_match(&pattern);
        assert!(matches.contains(&"s1".to_string()));
        assert!(!matches.contains(&"s2".to_string()));
    }

    // ── ASK Probing Tests ────────────────────────────────────────────────

    #[test]
    fn test_probe_source() {
        let mut sel = make_selector();
        sel.register_source(make_source_with_void(
            "s1",
            &["http://schema.org/name"],
            1000,
        ))
        .ok();
        let pattern = TriplePattern {
            subject: None,
            predicate: Some("http://schema.org/name".to_string()),
            object: None,
        };
        let result = sel.probe_source("s1", &pattern);
        assert!(result.is_ok());
        let r = result.expect("probe result");
        assert!(r.success);
        assert!(r.pattern_matched);
    }

    #[test]
    fn test_probe_nonexistent() {
        let sel = make_selector();
        let pattern = TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        };
        assert!(sel.probe_source("nope", &pattern).is_err());
    }

    #[test]
    fn test_probe_all() {
        let mut sel = make_selector();
        sel.register_source(make_source_with_void("s1", &["http://p1"], 100))
            .ok();
        sel.register_source(make_source_with_void("s2", &["http://p2"], 200))
            .ok();
        let pattern = TriplePattern {
            subject: None,
            predicate: Some("http://p1".to_string()),
            object: None,
        };
        let results = sel.probe_all(&pattern);
        assert_eq!(results.len(), 2);
    }

    // ── Capability Routing Tests ─────────────────────────────────────────

    #[test]
    fn test_sources_with_sparql_version() {
        let mut sel = make_selector();
        let mut s = make_source("s1", "http://x");
        s.capabilities.sparql_version = SparqlVersion::V1_2;
        sel.register_source(s).ok();
        sel.register_source(make_source("s2", "http://y")).ok();

        let v12 = sel.sources_with_sparql_version(&SparqlVersion::V1_2);
        assert_eq!(v12.len(), 1);
        assert_eq!(v12[0], "s1");
    }

    #[test]
    fn test_sources_with_named_graphs() {
        let mut sel = make_selector();
        let mut s = make_source("s1", "http://x");
        s.capabilities.supports_named_graphs = true;
        sel.register_source(s).ok();

        let ng = sel.sources_with_named_graphs();
        assert_eq!(ng.len(), 1);
    }

    #[test]
    fn test_sources_with_geosparql() {
        let mut sel = make_selector();
        let mut s = make_source("s1", "http://x");
        s.capabilities.supports_geosparql = true;
        sel.register_source(s).ok();

        let geo = sel.sources_with_geosparql();
        assert_eq!(geo.len(), 1);
    }

    // ── Ranking Tests ────────────────────────────────────────────────────

    #[test]
    fn test_score_source_healthy() {
        let sel = make_selector();
        let s = make_source("s1", "http://x");
        let score = sel.score_source(&s);
        // Default metrics: 0ms latency, 1.0 reliability, 0ms freshness
        assert!(score > 0.0);
    }

    #[test]
    fn test_score_source_unhealthy() {
        let sel = make_selector();
        let mut s = make_source("s1", "http://x");
        s.metrics.avg_latency_ms = 10_000.0;
        s.metrics.success_rate = 0.0;
        s.metrics.freshness_ms = 3_600_000;
        let score = sel.score_source(&s);
        // Very low score expected
        assert!(score < 0.5);
    }

    #[test]
    fn test_rank_sources() {
        let mut sel = make_selector();
        let mut fast = make_source("fast", "http://fast");
        fast.metrics.avg_latency_ms = 10.0;
        fast.metrics.success_rate = 0.99;
        sel.register_source(fast).ok();

        let mut slow = make_source("slow", "http://slow");
        slow.metrics.avg_latency_ms = 5000.0;
        slow.metrics.success_rate = 0.5;
        sel.register_source(slow).ok();

        let ranked = sel.rank_sources();
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].0, "fast");
    }

    // ── Cost-based Selection Tests ───────────────────────────────────────

    #[test]
    fn test_select_sources_single_pattern() {
        let mut sel = make_selector();
        sel.register_source(make_source_with_void("s1", &["http://p1"], 100))
            .ok();

        let patterns = vec![TriplePattern {
            subject: None,
            predicate: Some("http://p1".to_string()),
            object: None,
        }];
        let result = sel.select_sources(&patterns);
        assert!(result.is_ok());
        let r = result.expect("selection result");
        assert!(r.pattern_sources.contains_key(&0));
    }

    #[test]
    fn test_select_sources_no_eligible() {
        let sel = make_selector();
        let patterns = vec![TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        }];
        assert!(sel.select_sources(&patterns).is_err());
    }

    #[test]
    fn test_select_sources_cross_join_cost() {
        let mut sel = make_selector();
        sel.register_source(make_source_with_void("s1", &["http://p1"], 100))
            .ok();
        sel.register_source(make_source_with_void("s2", &["http://p2"], 200))
            .ok();

        let patterns = vec![
            TriplePattern {
                subject: None,
                predicate: Some("http://p1".to_string()),
                object: None,
            },
            TriplePattern {
                subject: None,
                predicate: Some("http://p2".to_string()),
                object: None,
            },
        ];
        let result = sel.select_sources(&patterns).expect("selection result");
        // Two patterns with different preferred sources => cross-source join penalty
        assert!(result.estimated_cost >= 2.0);
    }

    // ── Metadata Caching Tests ───────────────────────────────────────────

    #[test]
    fn test_cache_metadata() {
        let mut sel = make_selector();
        sel.cache_metadata("s1", SourceCapabilities::default(), None, 1000);
        assert_eq!(sel.cache_size(), 1);
    }

    #[test]
    fn test_get_cached_metadata() {
        let mut sel = make_selector();
        sel.cache_metadata("s1", SourceCapabilities::default(), None, 1000);
        let cached = sel.get_cached_metadata("s1", 2000);
        assert!(cached.is_some());
    }

    #[test]
    fn test_cached_metadata_expired() {
        let mut sel = make_selector();
        sel.set_cache_ttl_ms(5000);
        sel.cache_metadata("s1", SourceCapabilities::default(), None, 1000);
        // 7000ms > 5000ms TTL
        let cached = sel.get_cached_metadata("s1", 7000);
        assert!(cached.is_none());
    }

    #[test]
    fn test_evict_expired_cache() {
        let mut sel = make_selector();
        sel.set_cache_ttl_ms(5000);
        sel.cache_metadata("s1", SourceCapabilities::default(), None, 0);
        sel.cache_metadata("s2", SourceCapabilities::default(), None, 4000);
        let evicted = sel.evict_expired_cache(6000);
        assert_eq!(evicted, 1); // s1 expired, s2 still valid
        assert_eq!(sel.cache_size(), 1);
    }

    // ── Metrics Update Tests ─────────────────────────────────────────────

    #[test]
    fn test_update_metrics() {
        let mut sel = make_selector();
        sel.register_source(make_source("s1", "http://x")).ok();
        let new_metrics = SourceMetrics {
            avg_latency_ms: 100.0,
            success_rate: 0.95,
            freshness_ms: 5000,
            queries_processed: 50,
            is_reachable: true,
        };
        assert!(sel.update_metrics("s1", new_metrics).is_ok());
        let s = sel.get_source("s1").expect("source");
        assert!((s.metrics.avg_latency_ms - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_metrics_nonexistent() {
        let mut sel = make_selector();
        assert!(sel
            .update_metrics("nope", SourceMetrics::default())
            .is_err());
    }

    // ── Cost Weights Tests ───────────────────────────────────────────────

    #[test]
    fn test_custom_weights() {
        let weights = CostWeights {
            latency_weight: 0.5,
            reliability_weight: 0.2,
            freshness_weight: 0.2,
            coverage_weight: 0.1,
        };
        let sel = SourceSelector::with_weights(weights.clone());
        assert!((sel.weights().latency_weight - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_set_weights() {
        let mut sel = make_selector();
        sel.set_weights(CostWeights {
            latency_weight: 0.8,
            reliability_weight: 0.1,
            freshness_weight: 0.05,
            coverage_weight: 0.05,
        });
        assert!((sel.weights().latency_weight - 0.8).abs() < f64::EPSILON);
    }

    // ── Error Display Tests ──────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e = SelectionError::SourceNotFound("s1".to_string());
        assert!(format!("{e}").contains("s1"));
        let e = SelectionError::NoSourcesAvailable;
        assert!(!format!("{e}").is_empty());
        let e = SelectionError::SourceExcluded("s2".to_string());
        assert!(format!("{e}").contains("s2"));
        let e = SelectionError::InvalidPattern("bad".to_string());
        assert!(format!("{e}").contains("bad"));
    }

    // ── Default Trait Tests ──────────────────────────────────────────────

    #[test]
    fn test_default_selector() {
        let sel = SourceSelector::default();
        assert_eq!(sel.source_count(), 0);
    }

    #[test]
    fn test_default_source_capabilities() {
        let caps = SourceCapabilities::default();
        assert_eq!(caps.sparql_version, SparqlVersion::V1_1);
        assert!(!caps.supports_named_graphs);
    }

    #[test]
    fn test_default_source_metrics() {
        let m = SourceMetrics::default();
        assert!(m.is_reachable);
        assert!((m.success_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_default_cost_weights() {
        let w = CostWeights::default();
        let sum = w.latency_weight + w.reliability_weight + w.freshness_weight + w.coverage_weight;
        assert!((sum - 1.0).abs() < f64::EPSILON);
    }

    // ── All-variable Pattern Matching ────────────────────────────────────

    #[test]
    fn test_all_variable_pattern_matches_nonempty() {
        let mut sel = make_selector();
        sel.register_source(make_source_with_void("s1", &[], 100))
            .ok();
        let pattern = TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        };
        let matches = sel.void_match(&pattern);
        assert!(matches.contains(&"s1".to_string()));
    }

    #[test]
    fn test_disabled_source_excluded() {
        let mut sel = make_selector();
        let mut s = make_source("s1", "http://x");
        s.enabled = false;
        sel.register_source(s).ok();
        let ranked = sel.rank_sources();
        assert!(ranked.is_empty());
    }

    #[test]
    fn test_unreachable_source_excluded() {
        let mut sel = make_selector();
        let mut s = make_source("s1", "http://x");
        s.metrics.is_reachable = false;
        sel.register_source(s).ok();
        let ranked = sel.rank_sources();
        assert!(ranked.is_empty());
    }
}
