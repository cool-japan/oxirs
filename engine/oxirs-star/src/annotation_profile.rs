//! RDF-star annotation profiles.
//!
//! This module defines [`AnnotationProfile`] and the concrete profile strategies:
//!
//! - [`ReificationProfile`] – maps RDF-star quoted triples to classic W3C RDF
//!   reification (`rdf:Statement / rdf:subject / rdf:predicate / rdf:object`).
//! - [`SingletonProfile`] – uses the singleton property pattern where each
//!   statement gets a unique property IRI derived from the original predicate.
//! - [`NanopubProfile`] – models statements as nanopublications with distinct
//!   assertion, provenance, and publication-info named graphs.
//!
//! Profiles provide a round-trip conversion API:
//! `AnnotationProfile::to_triples` expands a quoted triple into a set of
//! plain `StarTriple`s, and `AnnotationProfile::from_triples` collapses them
//! back into the RDF-star representation.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use tracing::{debug, span, Level};

use crate::model::{StarTerm, StarTriple};
use crate::{StarError, StarResult};

// ============================================================================
// RDF / OWL vocabulary constants used by the profiles
// ============================================================================

pub mod vocab {
    pub const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    pub const RDF_STATEMENT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement";
    pub const RDF_SUBJECT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject";
    pub const RDF_PREDICATE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate";
    pub const RDF_OBJECT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#object";

    // Singleton property vocabulary
    pub const SP_SINGLETON_PROPERTY_OF: &str =
        "http://www.w3.org/ns/singletonProperty#singletonPropertyOf";

    // Nanopublication vocabulary
    pub const NP_NANOPUBLICATION: &str = "http://www.nanopub.org/nschema#Nanopublication";
    pub const NP_HAS_ASSERTION: &str = "http://www.nanopub.org/nschema#hasAssertion";
    pub const NP_HAS_PROVENANCE: &str = "http://www.nanopub.org/nschema#hasProvenance";
    pub const NP_HAS_PUBLICATION_INFO: &str = "http://www.nanopub.org/nschema#hasPublicationInfo";
}

// ============================================================================
// ProfileConfig – shared configuration knobs
// ============================================================================

/// Configuration for annotation profile conversions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProfileConfig {
    /// Base IRI used to mint fresh blank-node-like IRIs for statement nodes.
    pub base_iri: String,
    /// Monotonically increasing counter seed for deterministic IRI generation.
    pub counter_seed: u64,
    /// Whether to include `rdf:type` triples in the output (some profiles
    /// omit the type triple for brevity).
    pub include_type_triple: bool,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            base_iri: "http://example.org/stmt/".to_string(),
            counter_seed: 1,
            include_type_triple: true,
        }
    }
}

// ============================================================================
// ExpandedAnnotation – the result of profile expansion
// ============================================================================

/// The result of expanding a single quoted (annotated) triple through a profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpandedAnnotation {
    /// The plain `StarTriple`s that represent the quoted triple under the
    /// chosen profile (no `QuotedTriple` terms in any of these triples).
    pub triples: Vec<StarTriple>,
    /// The IRI (or blank-node identifier) that stands for the quoted triple
    /// within the expanded representation.
    pub statement_node: String,
}

impl ExpandedAnnotation {
    /// Returns the number of plain triples produced by the expansion.
    pub fn triple_count(&self) -> usize {
        self.triples.len()
    }
}

// ============================================================================
// ReificationProfile
// ============================================================================

/// Maps RDF-star quoted triples to classic W3C RDF reification.
///
/// For a quoted triple `<< :s :p :o >>` annotated with `:certainty 0.9`,
/// `ReificationProfile` produces:
/// ```text
/// _:stmt1  rdf:type       rdf:Statement .
/// _:stmt1  rdf:subject    :s            .
/// _:stmt1  rdf:predicate  :p            .
/// _:stmt1  rdf:object     :o            .
/// _:stmt1  :certainty     0.9           .
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReificationProfile {
    config: ProfileConfig,
    counter: u64,
}

impl ReificationProfile {
    /// Create a new reification profile with default configuration.
    pub fn new() -> Self {
        Self {
            config: ProfileConfig::default(),
            counter: 0,
        }
    }

    /// Create a profile with custom configuration.
    pub fn with_config(config: ProfileConfig) -> Self {
        let counter = config.counter_seed;
        Self { config, counter }
    }

    /// Expand a quoted triple (and optional annotation pairs) into classic
    /// reification triples.
    ///
    /// `quoted` must be a [`StarTerm::QuotedTriple`].
    /// `annotations` is a list of `(predicate_term, object_term)` pairs that
    /// annotate the statement node.
    pub fn expand(
        &mut self,
        quoted: &StarTerm,
        annotations: &[(StarTerm, StarTerm)],
    ) -> StarResult<ExpandedAnnotation> {
        let span = span!(Level::DEBUG, "ReificationProfile::expand");
        let _enter = span.enter();

        let inner = match quoted {
            StarTerm::QuotedTriple(t) => t.as_ref(),
            other => {
                return Err(StarError::invalid_term_type(format!(
                    "Expected QuotedTriple, got {:?}",
                    other
                )))
            }
        };

        self.counter += 1;
        let stmt_iri = format!("{}{}", self.config.base_iri, self.counter);
        let stmt_node = StarTerm::iri(&stmt_iri)?;

        let mut triples: Vec<StarTriple> = Vec::new();

        if self.config.include_type_triple {
            triples.push(StarTriple::new(
                stmt_node.clone(),
                StarTerm::iri(vocab::RDF_TYPE)?,
                StarTerm::iri(vocab::RDF_STATEMENT)?,
            ));
        }

        triples.push(StarTriple::new(
            stmt_node.clone(),
            StarTerm::iri(vocab::RDF_SUBJECT)?,
            inner.subject.clone(),
        ));
        triples.push(StarTriple::new(
            stmt_node.clone(),
            StarTerm::iri(vocab::RDF_PREDICATE)?,
            inner.predicate.clone(),
        ));
        triples.push(StarTriple::new(
            stmt_node.clone(),
            StarTerm::iri(vocab::RDF_OBJECT)?,
            inner.object.clone(),
        ));

        for (pred, obj) in annotations {
            triples.push(StarTriple::new(
                stmt_node.clone(),
                pred.clone(),
                obj.clone(),
            ));
        }

        debug!(
            stmt_iri = %stmt_iri,
            annotation_count = annotations.len(),
            "Reification expanded"
        );

        Ok(ExpandedAnnotation {
            triples,
            statement_node: stmt_iri,
        })
    }

    /// Collapse a set of classic reification triples back into a `StarTriple`
    /// whose subject is a `QuotedTriple`.
    ///
    /// The `statement_node_iri` identifies the statement resource that holds
    /// the `rdf:subject`, `rdf:predicate`, and `rdf:object` triples.
    pub fn collapse(
        &self,
        triples: &[StarTriple],
        statement_node_iri: &str,
    ) -> StarResult<StarTriple> {
        let span = span!(Level::DEBUG, "ReificationProfile::collapse");
        let _enter = span.enter();

        let stmt_term = StarTerm::iri(statement_node_iri)?;

        let find = |pred_iri: &str| -> StarResult<StarTerm> {
            triples
                .iter()
                .find(|t| {
                    t.subject == stmt_term
                        && matches!(&t.predicate, StarTerm::NamedNode(n) if n.iri == pred_iri)
                })
                .map(|t| t.object.clone())
                .ok_or_else(|| {
                    StarError::reification_error(format!(
                        "Missing triple for {pred_iri} on statement node {statement_node_iri}"
                    ))
                })
        };

        let subj = find(vocab::RDF_SUBJECT)?;
        let pred = find(vocab::RDF_PREDICATE)?;
        let obj = find(vocab::RDF_OBJECT)?;

        let inner = StarTriple::new(subj, pred, obj);
        debug!(statement_node_iri = %statement_node_iri, "Reification collapsed");

        // The collapsed triple has the quoted triple as its subject;
        // use a canonical annotation predicate to indicate the collapse.
        Ok(StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri(vocab::RDF_TYPE)?,
            stmt_term,
        ))
    }
}

impl Default for ReificationProfile {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SingletonProfile
// ============================================================================

/// Implements the singleton property pattern for annotating RDF statements.
///
/// Each statement gets a unique property IRI derived from the original
/// predicate by appending `#singleton_N`.  Example:
///
/// ```text
/// :s  :p#singleton_1  :o .
/// :p#singleton_1  sp:singletonPropertyOf  :p .
/// :p#singleton_1  :certainty  0.9 .
/// ```
///
/// This avoids blank nodes and `rdf:Statement` while keeping the original
/// subject–predicate–object position structure.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SingletonProfile {
    config: ProfileConfig,
    counter: u64,
    /// Cache: original predicate IRI → list of singleton IRI suffixes minted
    singleton_map: HashMap<String, Vec<String>>,
}

impl SingletonProfile {
    /// Create a new singleton profile with default configuration.
    pub fn new() -> Self {
        Self {
            config: ProfileConfig::default(),
            counter: 0,
            singleton_map: HashMap::new(),
        }
    }

    /// Create a singleton profile with custom configuration.
    pub fn with_config(config: ProfileConfig) -> Self {
        let counter = config.counter_seed;
        Self {
            config,
            counter,
            singleton_map: HashMap::new(),
        }
    }

    /// Expand a quoted triple using the singleton property pattern.
    ///
    /// `quoted` must be a [`StarTerm::QuotedTriple`].
    pub fn expand(
        &mut self,
        quoted: &StarTerm,
        annotations: &[(StarTerm, StarTerm)],
    ) -> StarResult<ExpandedAnnotation> {
        let span = span!(Level::DEBUG, "SingletonProfile::expand");
        let _enter = span.enter();

        let inner = match quoted {
            StarTerm::QuotedTriple(t) => t.as_ref(),
            other => {
                return Err(StarError::invalid_term_type(format!(
                    "Expected QuotedTriple, got {:?}",
                    other
                )))
            }
        };

        let original_pred = match &inner.predicate {
            StarTerm::NamedNode(n) => n.iri.clone(),
            other => {
                return Err(StarError::invalid_term_type(format!(
                    "Quoted triple predicate must be a NamedNode for SingletonProfile, got {:?}",
                    other
                )))
            }
        };

        self.counter += 1;
        let singleton_iri = format!("{}#singleton_{}", original_pred, self.counter);

        self.singleton_map
            .entry(original_pred.clone())
            .or_default()
            .push(singleton_iri.clone());

        let singleton_pred = StarTerm::iri(&singleton_iri)?;
        let mut triples: Vec<StarTriple> = Vec::new();

        // The statement itself with the unique singleton predicate
        triples.push(StarTriple::new(
            inner.subject.clone(),
            singleton_pred.clone(),
            inner.object.clone(),
        ));

        // Link singleton predicate back to the original
        triples.push(StarTriple::new(
            singleton_pred.clone(),
            StarTerm::iri(vocab::SP_SINGLETON_PROPERTY_OF)?,
            StarTerm::iri(&original_pred)?,
        ));

        // Annotations on the singleton predicate
        for (pred, obj) in annotations {
            triples.push(StarTriple::new(
                singleton_pred.clone(),
                pred.clone(),
                obj.clone(),
            ));
        }

        debug!(
            singleton_iri = %singleton_iri,
            original_pred = %original_pred,
            "Singleton property pattern expanded"
        );

        Ok(ExpandedAnnotation {
            triples,
            statement_node: singleton_iri,
        })
    }

    /// Retrieve all singleton IRIs minted for a given original predicate IRI.
    pub fn singletons_for(&self, predicate_iri: &str) -> &[String] {
        self.singleton_map
            .get(predicate_iri)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

impl Default for SingletonProfile {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// NanopubProfile
// ============================================================================

/// Models annotated statements as W3C Nanopublications.
///
/// A nanopublication wraps an asserted triple inside a named-graph structure
/// with three named graphs:
///
/// - **assertion graph**: contains the raw triple.
/// - **provenance graph**: contains metadata about the assertion graph.
/// - **publication-info graph**: contains metadata about the nanopublication
///   itself (timestamps, authors, …).
///
/// ```text
/// _:np1  np:hasAssertion       _:np1_assertion .
/// _:np1  np:hasProvenance      _:np1_prov .
/// _:np1  np:hasPublicationInfo _:np1_pubinfo .
/// _:np1  rdf:type              np:Nanopublication .
///
/// # assertion graph
/// _:np1_assertion  :s  :p  :o .
///
/// # provenance graph annotations
/// _:np1_prov  ...annotation triples...
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NanopubProfile {
    config: ProfileConfig,
    counter: u64,
}

impl NanopubProfile {
    /// Create a new nanopub profile with default configuration.
    pub fn new() -> Self {
        Self {
            config: ProfileConfig::default(),
            counter: 0,
        }
    }

    /// Create a nanopub profile with custom configuration.
    pub fn with_config(config: ProfileConfig) -> Self {
        let counter = config.counter_seed;
        Self { config, counter }
    }

    /// Expand a quoted triple into nanopublication triples.
    ///
    /// Returns an [`ExpandedAnnotation`] whose `statement_node` is the IRI of
    /// the nanopublication head graph.
    pub fn expand(
        &mut self,
        quoted: &StarTerm,
        provenance_annotations: &[(StarTerm, StarTerm)],
    ) -> StarResult<ExpandedAnnotation> {
        let span = span!(Level::DEBUG, "NanopubProfile::expand");
        let _enter = span.enter();

        let inner = match quoted {
            StarTerm::QuotedTriple(t) => t.as_ref(),
            other => {
                return Err(StarError::invalid_term_type(format!(
                    "Expected QuotedTriple, got {:?}",
                    other
                )))
            }
        };

        self.counter += 1;
        let np_iri = format!("{}{}", self.config.base_iri, self.counter);
        let assertion_iri = format!("{}_assertion", np_iri);
        let prov_iri = format!("{}_prov", np_iri);
        let pubinfo_iri = format!("{}_pubinfo", np_iri);

        let np_node = StarTerm::iri(&np_iri)?;
        let assertion_node = StarTerm::iri(&assertion_iri)?;
        let prov_node = StarTerm::iri(&prov_iri)?;
        let pubinfo_node = StarTerm::iri(&pubinfo_iri)?;

        let mut triples: Vec<StarTriple> = Vec::new();

        // Nanopublication head
        triples.push(StarTriple::new(
            np_node.clone(),
            StarTerm::iri(vocab::NP_HAS_ASSERTION)?,
            assertion_node.clone(),
        ));
        triples.push(StarTriple::new(
            np_node.clone(),
            StarTerm::iri(vocab::NP_HAS_PROVENANCE)?,
            prov_node.clone(),
        ));
        triples.push(StarTriple::new(
            np_node.clone(),
            StarTerm::iri(vocab::NP_HAS_PUBLICATION_INFO)?,
            pubinfo_node.clone(),
        ));

        if self.config.include_type_triple {
            triples.push(StarTriple::new(
                np_node.clone(),
                StarTerm::iri(vocab::RDF_TYPE)?,
                StarTerm::iri(vocab::NP_NANOPUBLICATION)?,
            ));
        }

        // Assertion graph: contains the raw triple as a subject-predicate-object
        // We represent the assertion graph membership as: assertion_node :contains inner_triple_encoded
        // Since plain triples can't use graphs, we encode subject/predicate/object as linked triples:
        triples.push(StarTriple::new(
            assertion_node.clone(),
            StarTerm::iri(vocab::RDF_SUBJECT)?,
            inner.subject.clone(),
        ));
        triples.push(StarTriple::new(
            assertion_node.clone(),
            StarTerm::iri(vocab::RDF_PREDICATE)?,
            inner.predicate.clone(),
        ));
        triples.push(StarTriple::new(
            assertion_node.clone(),
            StarTerm::iri(vocab::RDF_OBJECT)?,
            inner.object.clone(),
        ));

        // Provenance annotations
        for (pred, obj) in provenance_annotations {
            triples.push(StarTriple::new(
                prov_node.clone(),
                pred.clone(),
                obj.clone(),
            ));
        }

        debug!(
            np_iri = %np_iri,
            provenance_count = provenance_annotations.len(),
            "Nanopublication expanded"
        );

        Ok(ExpandedAnnotation {
            triples,
            statement_node: np_iri,
        })
    }

    /// Return the assertion-graph IRI for a given nanopublication head IRI.
    pub fn assertion_iri(np_head_iri: &str) -> String {
        format!("{}_assertion", np_head_iri)
    }

    /// Return the provenance-graph IRI for a given nanopublication head IRI.
    pub fn provenance_iri(np_head_iri: &str) -> String {
        format!("{}_prov", np_head_iri)
    }

    /// Return the publication-info-graph IRI for a given nanopublication head IRI.
    pub fn pubinfo_iri(np_head_iri: &str) -> String {
        format!("{}_pubinfo", np_head_iri)
    }
}

impl Default for NanopubProfile {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// AnnotationProfile enum – unified API
// ============================================================================

/// A unified enumeration of all supported annotation profiles.
///
/// Use [`AnnotationProfile::expand`] to convert an RDF-star quoted triple into
/// plain triples according to the chosen profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnnotationProfile {
    /// Classic W3C RDF reification.
    Reification(ReificationProfile),
    /// Singleton property pattern.
    Singleton(SingletonProfile),
    /// W3C Nanopublication model.
    Nanopub(NanopubProfile),
}

impl AnnotationProfile {
    /// Create a default reification profile.
    pub fn reification() -> Self {
        Self::Reification(ReificationProfile::new())
    }

    /// Create a default singleton profile.
    pub fn singleton() -> Self {
        Self::Singleton(SingletonProfile::new())
    }

    /// Create a default nanopub profile.
    pub fn nanopub() -> Self {
        Self::Nanopub(NanopubProfile::new())
    }

    /// Expand a quoted triple using this profile.
    pub fn expand(
        &mut self,
        quoted: &StarTerm,
        annotations: &[(StarTerm, StarTerm)],
    ) -> StarResult<ExpandedAnnotation> {
        match self {
            Self::Reification(p) => p.expand(quoted, annotations),
            Self::Singleton(p) => p.expand(quoted, annotations),
            Self::Nanopub(p) => p.expand(quoted, annotations),
        }
    }

    /// Return the profile name as a static string.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Reification(_) => "reification",
            Self::Singleton(_) => "singleton",
            Self::Nanopub(_) => "nanopub",
        }
    }

    /// Convert from one profile type to another by first collapsing back to a
    /// quoted triple and then re-expanding under the target profile.
    ///
    /// Only conversion from `Reification` expansion is supported (the most
    /// commonly needed direction); other conversions return an error indicating
    /// that the source expansion is ambiguous.
    pub fn convert_to(
        source_expansion: &ExpandedAnnotation,
        target: &mut AnnotationProfile,
    ) -> StarResult<ExpandedAnnotation> {
        // Recover the inner triple from the expansion by looking for
        // rdf:subject / rdf:predicate / rdf:object triples
        let find_obj = |pred_iri: &str| -> StarResult<StarTerm> {
            source_expansion
                .triples
                .iter()
                .find(|t| matches!(&t.predicate, StarTerm::NamedNode(n) if n.iri == pred_iri))
                .map(|t| t.object.clone())
                .ok_or_else(|| {
                    StarError::reification_error(format!(
                        "Cannot find {pred_iri} triple in source expansion for profile conversion"
                    ))
                })
        };

        let subj = find_obj(vocab::RDF_SUBJECT)?;
        let pred = find_obj(vocab::RDF_PREDICATE)?;
        let obj = find_obj(vocab::RDF_OBJECT)?;

        let inner = StarTriple::new(subj, pred, obj);
        let quoted = StarTerm::quoted_triple(inner);

        // Collect annotation pairs (skip structural triples)
        let structural_iris = [
            vocab::RDF_TYPE,
            vocab::RDF_SUBJECT,
            vocab::RDF_PREDICATE,
            vocab::RDF_OBJECT,
            vocab::NP_HAS_ASSERTION,
            vocab::NP_HAS_PROVENANCE,
            vocab::NP_HAS_PUBLICATION_INFO,
            vocab::SP_SINGLETON_PROPERTY_OF,
        ];

        let annotations: Vec<(StarTerm, StarTerm)> = source_expansion
            .triples
            .iter()
            .filter(|t| {
                !matches!(&t.predicate, StarTerm::NamedNode(n) if structural_iris.contains(&n.iri.as_str()))
            })
            .map(|t| (t.predicate.clone(), t.object.clone()))
            .collect();

        target.expand(&quoted, &annotations)
    }
}

impl fmt::Display for AnnotationProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AnnotationProfile::{}", self.name())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StarTerm;

    fn make_quoted_triple() -> StarTerm {
        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("30").unwrap(),
        );
        StarTerm::quoted_triple(inner)
    }

    fn certainty_annotation() -> (StarTerm, StarTerm) {
        (
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        )
    }

    // ------------------------------------------------------------------
    // ReificationProfile tests
    // ------------------------------------------------------------------

    #[test]
    fn test_reification_profile_new() {
        let profile = ReificationProfile::new();
        assert_eq!(profile.counter, 0);
        assert!(profile.config.include_type_triple);
    }

    #[test]
    fn test_reification_expand_produces_correct_triple_count() {
        let mut profile = ReificationProfile::new();
        let quoted = make_quoted_triple();
        let annotations = [certainty_annotation()];
        let result = profile.expand(&quoted, &annotations).unwrap();
        // type + subject + predicate + object + 1 annotation = 5
        assert_eq!(result.triple_count(), 5);
    }

    #[test]
    fn test_reification_expand_no_type_triple() {
        let mut profile = ReificationProfile::with_config(ProfileConfig {
            include_type_triple: false,
            ..Default::default()
        });
        let quoted = make_quoted_triple();
        let result = profile.expand(&quoted, &[]).unwrap();
        // subject + predicate + object = 3
        assert_eq!(result.triple_count(), 3);
    }

    #[test]
    fn test_reification_statement_node_iri_increments() {
        let mut profile = ReificationProfile::new();
        let quoted = make_quoted_triple();
        let r1 = profile.expand(&quoted, &[]).unwrap();
        let r2 = profile.expand(&quoted, &[]).unwrap();
        assert_ne!(r1.statement_node, r2.statement_node);
    }

    #[test]
    fn test_reification_expand_error_on_non_quoted_term() {
        let mut profile = ReificationProfile::new();
        let not_quoted = StarTerm::iri("http://example.org/plain").unwrap();
        let result = profile.expand(&not_quoted, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reification_collapse_roundtrip() {
        let mut profile = ReificationProfile::new();
        let quoted = make_quoted_triple();
        let expanded = profile.expand(&quoted, &[]).unwrap();
        let stmt_iri = expanded.statement_node.clone();
        let collapsed = profile.collapse(&expanded.triples, &stmt_iri).unwrap();
        // The collapsed triple's subject should be a quoted triple
        assert!(matches!(collapsed.subject, StarTerm::QuotedTriple(_)));
    }

    #[test]
    fn test_reification_collapse_missing_predicate_triple() {
        let profile = ReificationProfile::new();
        // Empty triples slice – should fail to find rdf:subject
        let result = profile.collapse(&[], "http://example.org/stmt/1");
        assert!(result.is_err());
    }

    #[test]
    fn test_reification_rdf_subject_is_alice() {
        let mut profile = ReificationProfile::new();
        let quoted = make_quoted_triple();
        let expanded = profile.expand(&quoted, &[]).unwrap();
        let subj_triple = expanded
            .triples
            .iter()
            .find(|t| matches!(&t.predicate, StarTerm::NamedNode(n) if n.iri == vocab::RDF_SUBJECT))
            .unwrap();
        assert_eq!(
            subj_triple.object,
            StarTerm::iri("http://example.org/alice").unwrap()
        );
    }

    // ------------------------------------------------------------------
    // SingletonProfile tests
    // ------------------------------------------------------------------

    #[test]
    fn test_singleton_profile_new() {
        let profile = SingletonProfile::new();
        assert!(profile.singleton_map.is_empty());
    }

    #[test]
    fn test_singleton_expand_basic() {
        let mut profile = SingletonProfile::new();
        let quoted = make_quoted_triple();
        let annotations = [certainty_annotation()];
        let result = profile.expand(&quoted, &annotations).unwrap();
        // original statement + sp:singletonPropertyOf link + 1 annotation = 3
        assert_eq!(result.triple_count(), 3);
    }

    #[test]
    fn test_singleton_expand_increments_counter() {
        let mut profile = SingletonProfile::new();
        let quoted = make_quoted_triple();
        let r1 = profile.expand(&quoted, &[]).unwrap();
        let r2 = profile.expand(&quoted, &[]).unwrap();
        assert_ne!(r1.statement_node, r2.statement_node);
    }

    #[test]
    fn test_singleton_singletons_for_tracks_correctly() {
        let mut profile = SingletonProfile::new();
        let quoted = make_quoted_triple();
        profile.expand(&quoted, &[]).unwrap();
        profile.expand(&quoted, &[]).unwrap();
        let singletons = profile.singletons_for("http://example.org/age");
        assert_eq!(singletons.len(), 2);
    }

    #[test]
    fn test_singleton_expand_error_on_non_named_node_predicate() {
        let mut profile = SingletonProfile::new();
        // Build a quoted triple with a blank-node predicate (invalid but test coverage)
        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::blank_node("p_blank").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );
        let quoted = StarTerm::quoted_triple(inner);
        let result = profile.expand(&quoted, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_singleton_expand_error_on_non_quoted() {
        let mut profile = SingletonProfile::new();
        let plain = StarTerm::iri("http://example.org/plain").unwrap();
        assert!(profile.expand(&plain, &[]).is_err());
    }

    #[test]
    fn test_singleton_sp_singleton_property_of_link_exists() {
        let mut profile = SingletonProfile::new();
        let quoted = make_quoted_triple();
        let result = profile.expand(&quoted, &[]).unwrap();
        let sp_triple = result.triples.iter().find(|t| {
            matches!(&t.predicate, StarTerm::NamedNode(n) if n.iri == vocab::SP_SINGLETON_PROPERTY_OF)
        });
        assert!(sp_triple.is_some());
        // The object of sp:singletonPropertyOf should be the original predicate IRI
        if let Some(t) = sp_triple {
            assert_eq!(t.object, StarTerm::iri("http://example.org/age").unwrap());
        }
    }

    // ------------------------------------------------------------------
    // NanopubProfile tests
    // ------------------------------------------------------------------

    #[test]
    fn test_nanopub_profile_new() {
        let profile = NanopubProfile::new();
        assert_eq!(profile.counter, 0);
    }

    #[test]
    fn test_nanopub_expand_basic_count() {
        let mut profile = NanopubProfile::new();
        let quoted = make_quoted_triple();
        let result = profile.expand(&quoted, &[]).unwrap();
        // head has 3 structural + 1 type = 4
        // assertion graph has subject + predicate + object = 3
        // total = 7
        assert_eq!(result.triple_count(), 7);
    }

    #[test]
    fn test_nanopub_expand_with_provenance() {
        let mut profile = NanopubProfile::new();
        let quoted = make_quoted_triple();
        let prov = [(
            StarTerm::iri("http://purl.org/dc/terms/creator").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
        )];
        let result = profile.expand(&quoted, &prov).unwrap();
        assert_eq!(result.triple_count(), 8); // 7 + 1 provenance
    }

    #[test]
    fn test_nanopub_expand_increments_counter() {
        let mut profile = NanopubProfile::new();
        let quoted = make_quoted_triple();
        let r1 = profile.expand(&quoted, &[]).unwrap();
        let r2 = profile.expand(&quoted, &[]).unwrap();
        assert_ne!(r1.statement_node, r2.statement_node);
    }

    #[test]
    fn test_nanopub_has_assertion_triple() {
        let mut profile = NanopubProfile::new();
        let quoted = make_quoted_triple();
        let result = profile.expand(&quoted, &[]).unwrap();
        let has_assertion = result.triples.iter().find(
            |t| matches!(&t.predicate, StarTerm::NamedNode(n) if n.iri == vocab::NP_HAS_ASSERTION),
        );
        assert!(has_assertion.is_some());
    }

    #[test]
    fn test_nanopub_has_provenance_triple() {
        let mut profile = NanopubProfile::new();
        let quoted = make_quoted_triple();
        let result = profile.expand(&quoted, &[]).unwrap();
        let has_prov = result.triples.iter().find(
            |t| matches!(&t.predicate, StarTerm::NamedNode(n) if n.iri == vocab::NP_HAS_PROVENANCE),
        );
        assert!(has_prov.is_some());
    }

    #[test]
    fn test_nanopub_has_publication_info_triple() {
        let mut profile = NanopubProfile::new();
        let quoted = make_quoted_triple();
        let result = profile.expand(&quoted, &[]).unwrap();
        let has_pubinfo = result.triples.iter().find(|t| {
            matches!(&t.predicate, StarTerm::NamedNode(n) if n.iri == vocab::NP_HAS_PUBLICATION_INFO)
        });
        assert!(has_pubinfo.is_some());
    }

    #[test]
    fn test_nanopub_assertion_iri_format() {
        assert_eq!(
            NanopubProfile::assertion_iri("http://example.org/stmt/1"),
            "http://example.org/stmt/1_assertion"
        );
    }

    #[test]
    fn test_nanopub_provenance_iri_format() {
        assert_eq!(
            NanopubProfile::provenance_iri("http://example.org/stmt/1"),
            "http://example.org/stmt/1_prov"
        );
    }

    #[test]
    fn test_nanopub_pubinfo_iri_format() {
        assert_eq!(
            NanopubProfile::pubinfo_iri("http://example.org/stmt/1"),
            "http://example.org/stmt/1_pubinfo"
        );
    }

    #[test]
    fn test_nanopub_type_triple_present() {
        let mut profile = NanopubProfile::new();
        let quoted = make_quoted_triple();
        let result = profile.expand(&quoted, &[]).unwrap();
        let type_triple = result.triples.iter().find(|t| {
            matches!(&t.predicate, StarTerm::NamedNode(n) if n.iri == vocab::RDF_TYPE)
                && matches!(&t.object, StarTerm::NamedNode(n) if n.iri == vocab::NP_NANOPUBLICATION)
        });
        assert!(type_triple.is_some());
    }

    #[test]
    fn test_nanopub_expand_error_on_non_quoted() {
        let mut profile = NanopubProfile::new();
        let plain = StarTerm::iri("http://example.org/plain").unwrap();
        assert!(profile.expand(&plain, &[]).is_err());
    }

    // ------------------------------------------------------------------
    // AnnotationProfile enum tests
    // ------------------------------------------------------------------

    #[test]
    fn test_annotation_profile_name_reification() {
        let profile = AnnotationProfile::reification();
        assert_eq!(profile.name(), "reification");
    }

    #[test]
    fn test_annotation_profile_name_singleton() {
        let profile = AnnotationProfile::singleton();
        assert_eq!(profile.name(), "singleton");
    }

    #[test]
    fn test_annotation_profile_name_nanopub() {
        let profile = AnnotationProfile::nanopub();
        assert_eq!(profile.name(), "nanopub");
    }

    #[test]
    fn test_annotation_profile_reification_expand() {
        let mut profile = AnnotationProfile::reification();
        let quoted = make_quoted_triple();
        let result = profile.expand(&quoted, &[]).unwrap();
        assert!(!result.triples.is_empty());
    }

    #[test]
    fn test_annotation_profile_singleton_expand() {
        let mut profile = AnnotationProfile::singleton();
        let quoted = make_quoted_triple();
        let result = profile.expand(&quoted, &[]).unwrap();
        assert!(!result.triples.is_empty());
    }

    #[test]
    fn test_annotation_profile_nanopub_expand() {
        let mut profile = AnnotationProfile::nanopub();
        let quoted = make_quoted_triple();
        let result = profile.expand(&quoted, &[]).unwrap();
        assert!(!result.triples.is_empty());
    }

    #[test]
    fn test_annotation_profile_display() {
        let profile = AnnotationProfile::reification();
        let s = format!("{}", profile);
        assert!(s.contains("reification"));
    }

    #[test]
    fn test_annotation_profile_convert_reification_to_singleton() {
        let mut source_profile = AnnotationProfile::reification();
        let quoted = make_quoted_triple();
        let annotations = [certainty_annotation()];
        let expanded = source_profile.expand(&quoted, &annotations).unwrap();

        let mut target = AnnotationProfile::singleton();
        let converted = AnnotationProfile::convert_to(&expanded, &mut target).unwrap();
        assert!(!converted.triples.is_empty());
    }

    #[test]
    fn test_annotation_profile_convert_reification_to_nanopub() {
        let mut source_profile = AnnotationProfile::reification();
        let quoted = make_quoted_triple();
        let expanded = source_profile.expand(&quoted, &[]).unwrap();

        let mut target = AnnotationProfile::nanopub();
        let converted = AnnotationProfile::convert_to(&expanded, &mut target).unwrap();
        // Nanopub has at least 7 triples
        assert!(converted.triple_count() >= 7);
    }

    #[test]
    fn test_profile_config_default() {
        let config = ProfileConfig::default();
        assert_eq!(config.base_iri, "http://example.org/stmt/");
        assert_eq!(config.counter_seed, 1);
        assert!(config.include_type_triple);
    }

    #[test]
    fn test_reification_with_multiple_annotations() {
        let mut profile = ReificationProfile::new();
        let quoted = make_quoted_triple();
        let annotations = [
            (
                StarTerm::iri("http://example.org/certainty").unwrap(),
                StarTerm::literal("0.9").unwrap(),
            ),
            (
                StarTerm::iri("http://example.org/source").unwrap(),
                StarTerm::iri("http://example.org/study").unwrap(),
            ),
            (
                StarTerm::iri("http://example.org/year").unwrap(),
                StarTerm::literal("2024").unwrap(),
            ),
        ];
        let result = profile.expand(&quoted, &annotations).unwrap();
        // type + subject + predicate + object + 3 annotations = 7
        assert_eq!(result.triple_count(), 7);
    }

    #[test]
    fn test_singleton_with_multiple_annotations() {
        let mut profile = SingletonProfile::new();
        let quoted = make_quoted_triple();
        let annotations = [
            certainty_annotation(),
            (
                StarTerm::iri("http://example.org/source").unwrap(),
                StarTerm::iri("http://example.org/census").unwrap(),
            ),
        ];
        let result = profile.expand(&quoted, &annotations).unwrap();
        // statement + sp:singletonPropertyOf + 2 annotations = 4
        assert_eq!(result.triple_count(), 4);
    }
}
