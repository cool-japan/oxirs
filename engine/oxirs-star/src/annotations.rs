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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

impl Default for TripleAnnotation {
    fn default() -> Self {
        Self {
            confidence: None,
            source: None,
            timestamp: None,
            validity_period: None,
            evidence: Vec::new(),
            custom_metadata: HashMap::new(),
            provenance: Vec::new(),
            quality_score: None,
            locale: None,
            version: None,
        }
    }
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

            stats.total_evidence += annotation.evidence.len();
            stats.total_provenance_records += annotation.provenance.len();

            let trust = annotation.trust_score();
            stats.avg_trust_score += trust;
            stats.min_trust_score = stats.min_trust_score.min(trust);
            stats.max_trust_score = stats.max_trust_score.max(trust);
        }

        if stats.total_annotations > 0 {
            stats.avg_trust_score /= stats.total_annotations as f64;
        }

        stats
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

    /// Total evidence items across all annotations
    pub total_evidence: usize,

    /// Total provenance records
    pub total_provenance_records: usize,

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
}
