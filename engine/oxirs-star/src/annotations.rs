//! Annotation support for RDF-star quoted triples
//!
//! This module provides metadata annotation capabilities for RDF-star triples,
//! allowing additional information to be attached to quoted triples beyond
//! the standard subject-predicate-object structure.

use crate::model::{StarGraph, StarTerm, StarTriple};
use crate::StarResult;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, span, Level};

/// Annotation metadata for a quoted triple
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct TripleAnnotation {
    /// Confidence/certainty score (0.0 to 1.0)
    pub confidence: Option<f64>,

    /// Source or author of the triple
    pub source: Option<String>,

    /// Timestamp when the triple was created/asserted
    pub timestamp: Option<DateTime<Utc>>,

    /// Validity period (start, end)
    pub validity_period: Option<(DateTime<Utc>, DateTime<Utc>)>,

    /// Evidence supporting this triple
    pub evidence: Vec<EvidenceItem>,

    /// Custom metadata as key-value pairs
    pub custom_metadata: HashMap<String, String>,

    /// Provenance chain (for tracking modifications)
    pub provenance: Vec<ProvenanceRecord>,

    /// Quality score or trust level
    pub quality_score: Option<f64>,

    /// Language/locale information
    pub locale: Option<String>,

    /// Version number for this triple
    pub version: Option<u64>,

    /// Nested annotations (annotations on this annotation)
    /// This enables meta-annotations and annotation chains
    pub meta_annotations: Vec<MetaAnnotation>,

    /// Annotation ID for referencing in chains
    pub annotation_id: Option<String>,
}

/// Meta-annotation - an annotation on another annotation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetaAnnotation {
    /// Type of meta-annotation (e.g., "verification", "correction", "endorsement")
    pub annotation_type: String,

    /// The actual annotation data
    pub annotation: TripleAnnotation,

    /// Target annotation ID (if referencing another annotation)
    pub target_id: Option<String>,

    /// Depth in the annotation chain (0 = root)
    pub depth: usize,
}

/// Evidence item supporting a triple assertion
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvidenceItem {
    /// Type of evidence (e.g., "citation", "experiment", "observation")
    pub evidence_type: String,

    /// Reference to the evidence source (IRI, DOI, etc.)
    pub reference: String,

    /// Strength of this evidence (0.0 to 1.0)
    pub strength: f64,

    /// Optional description
    pub description: Option<String>,
}

/// Provenance record tracking triple history
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProvenanceRecord {
    /// Action performed (e.g., "created", "modified", "verified")
    pub action: String,

    /// Agent who performed the action
    pub agent: String,

    /// Timestamp of the action
    pub timestamp: DateTime<Utc>,

    /// Activity context
    pub activity: Option<String>,

    /// Generation method (e.g., "manual", "automatic", "inferred")
    pub method: Option<String>,
}

impl TripleAnnotation {
    /// Create a new empty annotation
    pub fn new() -> Self {
        Self::default()
    }

    /// Create annotation with confidence score
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// Create annotation with source
    pub fn with_source(mut self, source: String) -> Self {
        self.source = Some(source);
        self
    }

    /// Create annotation with timestamp
    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Add evidence item
    pub fn add_evidence(&mut self, evidence: EvidenceItem) {
        self.evidence.push(evidence);
    }

    /// Add provenance record
    pub fn add_provenance(&mut self, record: ProvenanceRecord) {
        self.provenance.push(record);
    }

    /// Add custom metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.custom_metadata.insert(key, value);
    }

    /// Get overall trust score combining confidence, quality, and evidence
    pub fn trust_score(&self) -> f64 {
        let mut score = 0.0;
        let mut weight_sum = 0.0;

        // Confidence contributes 40%
        if let Some(conf) = self.confidence {
            score += conf * 0.4;
            weight_sum += 0.4;
        }

        // Quality score contributes 30%
        if let Some(quality) = self.quality_score {
            score += quality * 0.3;
            weight_sum += 0.3;
        }

        // Evidence strength contributes 30%
        if !self.evidence.is_empty() {
            let avg_evidence: f64 =
                self.evidence.iter().map(|e| e.strength).sum::<f64>() / self.evidence.len() as f64;
            score += avg_evidence * 0.3;
            weight_sum += 0.3;
        }

        if weight_sum > 0.0 {
            score / weight_sum
        } else {
            0.5 // Default neutral score
        }
    }

    /// Check if this triple is currently valid based on validity period
    pub fn is_currently_valid(&self) -> bool {
        if let Some((start, end)) = self.validity_period {
            let now = Utc::now();
            now >= start && now <= end
        } else {
            true // No validity period means always valid
        }
    }

    /// Add a meta-annotation (annotation on this annotation)
    pub fn add_meta_annotation(&mut self, annotation_type: String, annotation: TripleAnnotation) {
        let depth = self.get_max_depth() + 1;
        self.meta_annotations.push(MetaAnnotation {
            annotation_type,
            annotation,
            target_id: self.annotation_id.clone(),
            depth,
        });
    }

    /// Get the maximum depth of nested annotations
    pub fn get_max_depth(&self) -> usize {
        self.meta_annotations
            .iter()
            .map(|meta| meta.annotation.get_max_depth() + 1)
            .max()
            .unwrap_or(0)
    }

    /// Traverse all nested annotations and apply a function
    pub fn traverse_annotations<F>(&self, f: &mut F)
    where
        F: FnMut(&TripleAnnotation, usize),
    {
        self.traverse_annotations_internal(f, 0);
    }

    fn traverse_annotations_internal<F>(&self, f: &mut F, depth: usize)
    where
        F: FnMut(&TripleAnnotation, usize),
    {
        f(self, depth);
        for meta in &self.meta_annotations {
            meta.annotation.traverse_annotations_internal(f, depth + 1);
        }
    }

    /// Propagate trust scores through annotation chain
    /// Child annotations can influence parent trust scores
    pub fn propagated_trust_score(&self) -> f64 {
        let base_score = self.trust_score();

        if self.meta_annotations.is_empty() {
            return base_score;
        }

        // Calculate average endorsement/verification from meta-annotations
        let endorsement_count = self
            .meta_annotations
            .iter()
            .filter(|m| m.annotation_type == "endorsement" || m.annotation_type == "verification")
            .count();

        let correction_count = self
            .meta_annotations
            .iter()
            .filter(|m| m.annotation_type == "correction" || m.annotation_type == "dispute")
            .count();

        // Endorsements increase trust, corrections decrease it
        let adjustment = (endorsement_count as f64 * 0.05) - (correction_count as f64 * 0.1);
        (base_score + adjustment).clamp(0.0, 1.0)
    }

    /// Convert annotation to RDF triples using reification
    pub fn to_rdf_triples(&self, triple_id: &str) -> StarResult<Vec<StarTriple>> {
        let mut triples = Vec::new();
        let stmt_term = StarTerm::iri(triple_id)?;

        // Confidence
        if let Some(conf) = self.confidence {
            triples.push(StarTriple::new(
                stmt_term.clone(),
                StarTerm::iri("http://www.w3.org/ns/prov#confidence")?,
                StarTerm::literal(&conf.to_string())?,
            ));
        }

        // Source
        if let Some(ref source) = self.source {
            triples.push(StarTriple::new(
                stmt_term.clone(),
                StarTerm::iri("http://www.w3.org/ns/prov#hadPrimarySource")?,
                StarTerm::iri(source)?,
            ));
        }

        // Timestamp
        if let Some(ts) = self.timestamp {
            triples.push(StarTriple::new(
                stmt_term.clone(),
                StarTerm::iri("http://www.w3.org/ns/prov#generatedAtTime")?,
                StarTerm::literal(&ts.to_rfc3339())?,
            ));
        }

        // Quality score
        if let Some(quality) = self.quality_score {
            triples.push(StarTriple::new(
                stmt_term.clone(),
                StarTerm::iri("http://example.org/quality")?,
                StarTerm::literal(&quality.to_string())?,
            ));
        }

        // Meta-annotations
        for (idx, meta) in self.meta_annotations.iter().enumerate() {
            let meta_id = format!("{}/meta/{}", triple_id, idx);
            triples.push(StarTriple::new(
                stmt_term.clone(),
                StarTerm::iri("http://www.w3.org/ns/oa#hasAnnotation")?,
                StarTerm::iri(&meta_id)?,
            ));

            triples.push(StarTriple::new(
                StarTerm::iri(&meta_id)?,
                StarTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?,
                StarTerm::iri(&meta.annotation_type)?,
            ));

            // Recursively convert nested annotations
            let nested_triples = meta.annotation.to_rdf_triples(&meta_id)?;
            triples.extend(nested_triples);
        }

        Ok(triples)
    }
}

/// Annotation store managing annotations for quoted triples
pub struct AnnotationStore {
    /// Map from triple hash to annotation
    annotations: HashMap<u64, TripleAnnotation>,

    /// Annotation configuration
    config: AnnotationConfig,
}

/// Configuration for annotation store
#[derive(Debug, Clone)]
pub struct AnnotationConfig {
    /// Enable automatic timestamp generation
    pub auto_timestamp: bool,

    /// Enable automatic provenance tracking
    pub auto_provenance: bool,

    /// Default agent for provenance
    pub default_agent: String,

    /// Maximum annotations to cache
    pub max_cache_size: usize,
}

impl Default for AnnotationConfig {
    fn default() -> Self {
        Self {
            auto_timestamp: true,
            auto_provenance: true,
            default_agent: "oxirs-star".to_string(),
            max_cache_size: 10000,
        }
    }
}

impl AnnotationStore {
    /// Create a new annotation store
    pub fn new() -> Self {
        Self {
            annotations: HashMap::new(),
            config: AnnotationConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: AnnotationConfig) -> Self {
        Self {
            annotations: HashMap::new(),
            config,
        }
    }

    /// Add or update annotation for a triple
    pub fn annotate(
        &mut self,
        triple: &StarTriple,
        mut annotation: TripleAnnotation,
    ) -> StarResult<()> {
        let span = span!(Level::DEBUG, "annotate_triple");
        let _enter = span.enter();

        // Auto-add timestamp if enabled
        if self.config.auto_timestamp && annotation.timestamp.is_none() {
            annotation.timestamp = Some(Utc::now());
        }

        // Auto-add provenance if enabled
        if self.config.auto_provenance {
            annotation.add_provenance(ProvenanceRecord {
                action: "annotated".to_string(),
                agent: self.config.default_agent.clone(),
                timestamp: Utc::now(),
                activity: None,
                method: Some("automatic".to_string()),
            });
        }

        let hash = Self::hash_triple(triple);
        self.annotations.insert(hash, annotation);

        debug!("Added annotation for triple {}", hash);
        Ok(())
    }

    /// Get annotation for a triple
    pub fn get_annotation(&self, triple: &StarTriple) -> Option<&TripleAnnotation> {
        let hash = Self::hash_triple(triple);
        self.annotations.get(&hash)
    }

    /// Get mutable annotation for a triple
    pub fn get_annotation_mut(&mut self, triple: &StarTriple) -> Option<&mut TripleAnnotation> {
        let hash = Self::hash_triple(triple);
        self.annotations.get_mut(&hash)
    }

    /// Remove annotation for a triple
    pub fn remove_annotation(&mut self, triple: &StarTriple) -> Option<TripleAnnotation> {
        let hash = Self::hash_triple(triple);
        self.annotations.remove(&hash)
    }

    /// Export annotations to RDF graph
    pub fn export_to_graph(&self, base_iri: &str) -> StarResult<StarGraph> {
        let span = span!(Level::INFO, "export_annotations_to_graph");
        let _enter = span.enter();

        let mut graph = StarGraph::new();

        for (idx, (_hash, annotation)) in self.annotations.iter().enumerate() {
            let triple_id = format!("{}/annotation/{}", base_iri, idx + 1);
            let annotation_triples = annotation.to_rdf_triples(&triple_id)?;

            for triple in annotation_triples {
                graph.insert(triple)?;
            }
        }

        debug!(
            "Exported {} annotations to RDF graph",
            self.annotations.len()
        );
        Ok(graph)
    }

    /// Filter triples by minimum trust score
    pub fn filter_by_trust_score(&self, triples: &[StarTriple], min_score: f64) -> Vec<StarTriple> {
        triples
            .iter()
            .filter(|triple| {
                if let Some(annotation) = self.get_annotation(triple) {
                    annotation.trust_score() >= min_score
                } else {
                    false // No annotation = no trust data
                }
            })
            .cloned()
            .collect()
    }

    /// Get statistics about annotations
    #[allow(clippy::field_reassign_with_default)]
    pub fn statistics(&self) -> AnnotationStatistics {
        let mut stats = AnnotationStatistics::default();

        stats.total_annotations = self.annotations.len();

        for annotation in self.annotations.values() {
            if annotation.confidence.is_some() {
                stats.with_confidence += 1;
            }
            if annotation.source.is_some() {
                stats.with_source += 1;
            }
            if !annotation.evidence.is_empty() {
                stats.with_evidence += 1;
            }
            if !annotation.provenance.is_empty() {
                stats.with_provenance += 1;
            }
            if !annotation.meta_annotations.is_empty() {
                stats.with_meta_annotations += 1;
            }

            stats.total_evidence += annotation.evidence.len();
            stats.total_provenance_records += annotation.provenance.len();
            stats.total_meta_annotations += annotation.meta_annotations.len();

            let trust = annotation.trust_score();
            stats.avg_trust_score += trust;
            stats.min_trust_score = stats.min_trust_score.min(trust);
            stats.max_trust_score = stats.max_trust_score.max(trust);

            let depth = annotation.get_max_depth();
            stats.max_annotation_depth = stats.max_annotation_depth.max(depth);
        }

        if stats.total_annotations > 0 {
            stats.avg_trust_score /= stats.total_annotations as f64;
        }

        stats
    }

    /// Find annotations by type in meta-annotation chains
    pub fn find_by_meta_type(&self, annotation_type: &str) -> Vec<(u64, &TripleAnnotation)> {
        self.annotations
            .iter()
            .filter_map(|(hash, annotation)| {
                let has_type = annotation
                    .meta_annotations
                    .iter()
                    .any(|meta| meta.annotation_type == annotation_type);
                if has_type {
                    Some((*hash, annotation))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get all annotations with a minimum depth of nested annotations
    pub fn filter_by_min_depth(&self, min_depth: usize) -> Vec<(u64, &TripleAnnotation)> {
        self.annotations
            .iter()
            .filter_map(|(hash, annotation)| {
                if annotation.get_max_depth() >= min_depth {
                    Some((*hash, annotation))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Count total annotations including nested ones
    pub fn count_total_annotations(&self) -> usize {
        let mut count = self.annotations.len();
        for annotation in self.annotations.values() {
            count += Self::count_nested_annotations_recursive(annotation);
        }
        count
    }

    fn count_nested_annotations_recursive(annotation: &TripleAnnotation) -> usize {
        let mut count = annotation.meta_annotations.len();
        for meta in &annotation.meta_annotations {
            count += Self::count_nested_annotations_recursive(&meta.annotation);
        }
        count
    }

    /// Hash a triple for use as a key
    fn hash_triple(triple: &StarTriple) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{:?}", triple).hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for AnnotationStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about annotations in the store
#[derive(Debug, Clone, Default)]
pub struct AnnotationStatistics {
    /// Total number of annotations
    pub total_annotations: usize,

    /// Annotations with confidence scores
    pub with_confidence: usize,

    /// Annotations with source information
    pub with_source: usize,

    /// Annotations with evidence
    pub with_evidence: usize,

    /// Annotations with provenance
    pub with_provenance: usize,

    /// Annotations with meta-annotations
    pub with_meta_annotations: usize,

    /// Total evidence items across all annotations
    pub total_evidence: usize,

    /// Total provenance records
    pub total_provenance_records: usize,

    /// Total meta-annotations
    pub total_meta_annotations: usize,

    /// Maximum depth of annotation chains
    pub max_annotation_depth: usize,

    /// Average trust score
    pub avg_trust_score: f64,

    /// Minimum trust score
    pub min_trust_score: f64,

    /// Maximum trust score
    pub max_trust_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_annotation_creation() {
        let annotation = TripleAnnotation::new()
            .with_confidence(0.95)
            .with_source("http://example.org/source".to_string());

        assert_eq!(annotation.confidence, Some(0.95));
        assert_eq!(
            annotation.source,
            Some("http://example.org/source".to_string())
        );
    }

    #[test]
    fn test_trust_score_calculation() {
        let mut annotation = TripleAnnotation::new().with_confidence(0.8);

        annotation.quality_score = Some(0.9);
        annotation.add_evidence(EvidenceItem {
            evidence_type: "citation".to_string(),
            reference: "http://example.org/paper".to_string(),
            strength: 0.85,
            description: None,
        });

        let trust = annotation.trust_score();
        assert!(trust > 0.8 && trust < 0.9);
    }

    #[test]
    fn test_annotation_store() {
        let mut store = AnnotationStore::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/subject").unwrap(),
            StarTerm::iri("http://example.org/predicate").unwrap(),
            StarTerm::iri("http://example.org/object").unwrap(),
        );

        let annotation = TripleAnnotation::new().with_confidence(0.9);

        store.annotate(&triple, annotation.clone()).unwrap();

        let retrieved = store.get_annotation(&triple);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().confidence, Some(0.9));
    }

    #[test]
    fn test_validity_period() {
        let start = Utc::now();
        let end = start + chrono::Duration::days(30);

        let mut annotation = TripleAnnotation::new();
        annotation.validity_period = Some((start, end));

        assert!(annotation.is_currently_valid());
    }

    #[test]
    fn test_provenance_tracking() {
        let mut annotation = TripleAnnotation::new();

        annotation.add_provenance(ProvenanceRecord {
            action: "created".to_string(),
            agent: "Alice".to_string(),
            timestamp: Utc::now(),
            activity: Some("data_collection".to_string()),
            method: Some("manual".to_string()),
        });

        assert_eq!(annotation.provenance.len(), 1);
        assert_eq!(annotation.provenance[0].agent, "Alice");
    }

    #[test]
    fn test_nested_annotations() {
        let mut base_annotation = TripleAnnotation::new()
            .with_confidence(0.8)
            .with_source("http://example.org/source1".to_string());
        base_annotation.annotation_id = Some("ann-1".to_string());

        // Add a verification meta-annotation
        let verification = TripleAnnotation::new()
            .with_confidence(0.95)
            .with_source("http://example.org/verifier".to_string());

        base_annotation.add_meta_annotation("verification".to_string(), verification);

        assert_eq!(base_annotation.meta_annotations.len(), 1);
        assert_eq!(
            base_annotation.meta_annotations[0].annotation_type,
            "verification"
        );
        assert_eq!(base_annotation.get_max_depth(), 1);
    }

    #[test]
    fn test_deeply_nested_annotations() {
        let mut root = TripleAnnotation::new().with_confidence(0.7);
        root.annotation_id = Some("root".to_string());

        let mut level1 = TripleAnnotation::new().with_confidence(0.8);
        level1.annotation_id = Some("level1".to_string());

        let level2 = TripleAnnotation::new().with_confidence(0.9);

        level1.add_meta_annotation("endorsement".to_string(), level2);
        root.add_meta_annotation("verification".to_string(), level1);

        assert_eq!(root.get_max_depth(), 2);
        assert_eq!(root.meta_annotations.len(), 1);
        assert_eq!(
            root.meta_annotations[0].annotation.meta_annotations.len(),
            1
        );
    }

    #[test]
    fn test_annotation_traversal() {
        let mut root = TripleAnnotation::new().with_confidence(0.7);

        let child1 = TripleAnnotation::new().with_confidence(0.8);
        let child2 = TripleAnnotation::new().with_confidence(0.9);

        root.add_meta_annotation("verification".to_string(), child1);
        root.add_meta_annotation("endorsement".to_string(), child2);

        let mut count = 0;
        root.traverse_annotations(&mut |_ann, _depth| {
            count += 1;
        });

        // Should visit root + 2 children = 3 annotations
        assert_eq!(count, 3);
    }

    #[test]
    fn test_propagated_trust_score() {
        let mut base = TripleAnnotation::new().with_confidence(0.7);
        base.quality_score = Some(0.7);

        let initial_score = base.trust_score();

        // Add endorsements
        let endorsement = TripleAnnotation::new().with_confidence(0.9);
        base.add_meta_annotation("endorsement".to_string(), endorsement);

        let propagated = base.propagated_trust_score();
        // Endorsement should increase the score
        assert!(propagated > initial_score);

        // Add a correction (should decrease score)
        let correction = TripleAnnotation::new().with_confidence(0.5);
        base.add_meta_annotation("correction".to_string(), correction);

        let adjusted = base.propagated_trust_score();
        // Correction should decrease compared to just endorsement
        assert!(adjusted < propagated);
    }

    #[test]
    fn test_annotation_store_with_meta() {
        let mut store = AnnotationStore::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        let mut annotation = TripleAnnotation::new().with_confidence(0.8);
        annotation.annotation_id = Some("ann-1".to_string());

        let meta = TripleAnnotation::new().with_confidence(0.95);
        annotation.add_meta_annotation("verification".to_string(), meta);

        store.annotate(&triple, annotation).unwrap();

        let stats = store.statistics();
        assert_eq!(stats.with_meta_annotations, 1);
        assert_eq!(stats.total_meta_annotations, 1);
        assert_eq!(stats.max_annotation_depth, 1);
    }

    #[test]
    fn test_find_by_meta_type() {
        let mut store = AnnotationStore::new();

        let triple1 = StarTriple::new(
            StarTerm::iri("http://example.org/s1").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        let triple2 = StarTriple::new(
            StarTerm::iri("http://example.org/s2").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        let mut ann1 = TripleAnnotation::new().with_confidence(0.8);
        ann1.add_meta_annotation("verification".to_string(), TripleAnnotation::new());

        let mut ann2 = TripleAnnotation::new().with_confidence(0.9);
        ann2.add_meta_annotation("endorsement".to_string(), TripleAnnotation::new());

        store.annotate(&triple1, ann1).unwrap();
        store.annotate(&triple2, ann2).unwrap();

        let verified = store.find_by_meta_type("verification");
        assert_eq!(verified.len(), 1);

        let endorsed = store.find_by_meta_type("endorsement");
        assert_eq!(endorsed.len(), 1);
    }

    #[test]
    fn test_filter_by_min_depth() {
        let mut store = AnnotationStore::new();

        let triple1 = StarTriple::new(
            StarTerm::iri("http://example.org/s1").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        let triple2 = StarTriple::new(
            StarTerm::iri("http://example.org/s2").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        // Simple annotation (depth 0)
        let ann1 = TripleAnnotation::new().with_confidence(0.8);

        // Nested annotation (depth 1)
        let mut ann2 = TripleAnnotation::new().with_confidence(0.9);
        ann2.add_meta_annotation("verification".to_string(), TripleAnnotation::new());

        store.annotate(&triple1, ann1).unwrap();
        store.annotate(&triple2, ann2).unwrap();

        let deep = store.filter_by_min_depth(1);
        assert_eq!(deep.len(), 1); // Only ann2 has depth >= 1
    }

    #[test]
    fn test_count_total_annotations() {
        let mut store = AnnotationStore::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        let mut root = TripleAnnotation::new().with_confidence(0.7);
        let mut child1 = TripleAnnotation::new().with_confidence(0.8);
        let child2 = TripleAnnotation::new().with_confidence(0.9);

        child1.add_meta_annotation("endorsement".to_string(), child2);
        root.add_meta_annotation("verification".to_string(), child1);

        store.annotate(&triple, root).unwrap();

        // 1 root + 1 child1 + 1 child2 = 3 total
        assert_eq!(store.count_total_annotations(), 3);
    }
}
