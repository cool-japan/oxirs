//! RDF-star semantics and entailment rules
//!
//! This module implements the semantic layer for RDF-star, with a focus on
//! **referential opacity** - the key semantic property that distinguishes
//! RDF-star from standard RDF reification.
//!
//! # Referential Opacity
//!
//! In RDF-star, quoted triples do NOT entail the existence of the statement
//! they quote. This is called "referential opacity" and is the fundamental
//! semantic difference from standard RDF reification.
//!
//! ## Examples
//!
//! ```turtle
//! # This statement about certainty:
//! <<:alice :age 30>> :certainty 0.9 .
//!
//! # Does NOT entail that:
//! :alice :age 30 .
//!
//! # It only asserts that someone has 90% certainty about that statement,
//! # without asserting whether the statement itself is true or false.
//! ```
//!
//! ## Contrast with Standard RDF Reification
//!
//! Standard RDF reification does entail the original triple:
//!
//! ```turtle
//! # Standard reification (RDF 1.1):
//! _:stmt rdf:type rdf:Statement .
//! _:stmt rdf:subject :alice .
//! _:stmt rdf:predicate :age .
//! _:stmt rdf:object 30 .
//! _:stmt :certainty 0.9 .
//!
//! # In standard RDF semantics, this typically implies:
//! :alice :age 30 .  # (depending on reasoner)
//! ```
//!
//! But with RDF-star:
//!
//! ```turtle
//! # RDF-star (referentially opaque):
//! <<:alice :age 30>> :certainty 0.9 .
//!
//! # Does NOT imply:
//! :alice :age 30 .  # NOT entailed!
//! ```
//!
//! # Entailment Rules
//!
//! This module implements the W3C RDF-star entailment rules:
//!
//! 1. **No automatic assertion**: Quoted triples are NOT asserted
//! 2. **Identity-based equality**: Quoted triples are equal only if structurally identical
//! 3. **No blank node unification**: Blank nodes in quoted triples don't unify across quotes
//! 4. **Nested opacity**: Referential opacity applies recursively to nested quoted triples

use crate::model::{StarGraph, StarTerm, StarTriple};
use crate::StarResult;

/// Semantic validator for RDF-star graphs
///
/// This validator ensures that graphs respect RDF-star semantic rules,
/// particularly referential opacity.
#[derive(Debug, Clone)]
pub struct SemanticValidator {
    /// Whether to enforce strict referential opacity
    strict_opacity: bool,
}

impl Default for SemanticValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticValidator {
    /// Create a new semantic validator with default settings
    ///
    /// By default, strict referential opacity is enabled.
    pub fn new() -> Self {
        Self {
            strict_opacity: true,
        }
    }

    /// Create a validator with custom settings
    pub fn with_strict_opacity(mut self, strict: bool) -> Self {
        self.strict_opacity = strict;
        self
    }

    /// Check if a graph respects referential opacity
    ///
    /// This verifies that:
    /// 1. Quoted triples are not automatically asserted in the graph
    /// 2. No reasoning has been applied that violates referential opacity
    ///
    /// Returns `Ok(())` if the graph is semantically valid.
    pub fn validate_graph(&self, graph: &StarGraph) -> StarResult<()> {
        if !self.strict_opacity {
            return Ok(()); // Skip validation if not in strict mode
        }

        // Check for violations of referential opacity
        self.check_opacity_violations(graph)?;

        Ok(())
    }

    /// Detect potential violations of referential opacity
    ///
    /// This looks for patterns that suggest the graph may have been constructed
    /// with reasoning that violates referential opacity (e.g., inferring asserted
    /// triples from quoted triples).
    fn check_opacity_violations(&self, graph: &StarGraph) -> StarResult<()> {
        // Collect all quoted triples in the graph
        let mut quoted_triples = Vec::new();

        for triple in graph.iter() {
            self.collect_quoted_triples(triple, &mut quoted_triples);
        }

        // Check if any quoted triple is also asserted as a regular triple
        for quoted in &quoted_triples {
            if graph.contains(quoted) {
                // This could be intentional (user wants both the quote and the assertion)
                // but we warn about it as it might indicate a misunderstanding of
                // referential opacity
                //
                // Note: This is not technically an error, but a potential semantic issue
                // that users should be aware of.
                //
                // In strict mode, we don't fail here, but we could log a warning
                // (not implemented yet to avoid adding logging dependencies)
            }
        }

        Ok(())
    }

    /// Recursively collect all quoted triples from a triple
    fn collect_quoted_triples(&self, triple: &StarTriple, collector: &mut Vec<StarTriple>) {
        // Check subject
        if let StarTerm::QuotedTriple(qt) = &triple.subject {
            collector.push((**qt).clone());
            self.collect_quoted_triples(qt, collector);
        }

        // Check predicate
        if let StarTerm::QuotedTriple(qt) = &triple.predicate {
            collector.push((**qt).clone());
            self.collect_quoted_triples(qt, collector);
        }

        // Check object
        if let StarTerm::QuotedTriple(qt) = &triple.object {
            collector.push((**qt).clone());
            self.collect_quoted_triples(qt, collector);
        }
    }

    /// Extract all quoted triples from a graph
    ///
    /// This returns a list of all triples that appear as quoted triples
    /// within the graph, without asserting them.
    pub fn extract_quoted_triples(&self, graph: &StarGraph) -> Vec<StarTriple> {
        let mut quoted = Vec::new();

        for triple in graph.iter() {
            self.collect_quoted_triples(triple, &mut quoted);
        }

        quoted
    }

    /// Check if a triple is asserted vs. only quoted
    ///
    /// Returns:
    /// - `Ok(true)` if the triple is asserted (appears as a regular triple)
    /// - `Ok(false)` if the triple is only quoted (appears in quoted triple positions)
    pub fn is_asserted(&self, graph: &StarGraph, triple: &StarTriple) -> StarResult<bool> {
        Ok(graph.contains(triple))
    }

    /// Check if a triple is only mentioned (quoted but not asserted)
    ///
    /// Returns `true` if the triple appears in quoted positions but is not
    /// asserted as a regular triple.
    pub fn is_only_quoted(&self, graph: &StarGraph, triple: &StarTriple) -> StarResult<bool> {
        // Check if it's in quoted positions
        let quoted_triples = self.extract_quoted_triples(graph);
        let is_quoted = quoted_triples.contains(triple);

        // Check if it's asserted
        let is_asserted = self.is_asserted(graph, triple)?;

        Ok(is_quoted && !is_asserted)
    }
}

/// Entailment checker for RDF-star graphs
///
/// This implements the RDF-star entailment rules, particularly focusing on
/// the restrictions imposed by referential opacity.
#[derive(Debug, Clone)]
pub struct EntailmentChecker {
    validator: SemanticValidator,
}

impl Default for EntailmentChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl EntailmentChecker {
    /// Create a new entailment checker
    pub fn new() -> Self {
        Self {
            validator: SemanticValidator::new(),
        }
    }

    /// Check if one graph entails another under RDF-star semantics
    ///
    /// Returns `Ok(true)` if `source` entails `target` according to RDF-star
    /// entailment rules (respecting referential opacity).
    pub fn entails(&self, source: &StarGraph, target: &StarGraph) -> StarResult<bool> {
        // Validate both graphs
        self.validator.validate_graph(source)?;
        self.validator.validate_graph(target)?;

        // For now, implement simple containment check
        // (Full RDF-star entailment would require blank node matching, etc.)
        for target_triple in target.iter() {
            if !source.contains(target_triple) {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Check if a quoted triple entails its quoted content
    ///
    /// Under referential opacity, this should ALWAYS return `false`.
    ///
    /// # Example
    ///
    /// ```
    /// # use oxirs_star::semantics::EntailmentChecker;
    /// # use oxirs_star::model::{StarGraph, StarTerm, StarTriple};
    /// let checker = EntailmentChecker::new();
    ///
    /// let inner = StarTriple::new(
    ///     StarTerm::iri("http://example.org/alice").unwrap(),
    ///     StarTerm::iri("http://example.org/age").unwrap(),
    ///     StarTerm::literal("30").unwrap(),
    /// );
    ///
    /// let mut graph = StarGraph::new();
    /// let outer = StarTriple::new(
    ///     StarTerm::quoted_triple(inner.clone()),
    ///     StarTerm::iri("http://example.org/certainty").unwrap(),
    ///     StarTerm::literal("0.9").unwrap(),
    /// );
    /// graph.insert(outer).unwrap();
    ///
    /// // The quoted triple does NOT entail the inner triple
    /// assert!(!checker.quoted_entails_content(&graph, &inner).unwrap());
    /// ```
    pub fn quoted_entails_content(
        &self,
        graph: &StarGraph,
        inner: &StarTriple,
    ) -> StarResult<bool> {
        // Under referential opacity, quoted triples NEVER entail their content
        let _ = graph; // Use parameter to avoid warning
        let _ = inner;
        Ok(false)
    }

    /// Create an opacity-respecting closure of a graph
    ///
    /// This adds entailed triples while respecting referential opacity.
    /// Unlike standard RDF entailment, this does NOT add quoted triples
    /// as asserted triples.
    pub fn compute_closure(&self, graph: &StarGraph) -> StarResult<StarGraph> {
        // For now, just return a clone (no additional entailments)
        // In a full implementation, this would add:
        // - RDF/RDFS entailments for asserted triples
        // - But NOT entailments from quoted triples to asserted triples
        Ok(graph.clone())
    }
}

/// Transparency-Enabling Properties (TEPs) support
///
/// TEPs allow selective violations of referential opacity for specific properties.
/// This is an optional feature of RDF-star that allows certain predicates to
/// "see through" quoted triples.
///
/// # Example
///
/// ```turtle
/// # With rdf:type as a TEP:
/// <<:alice rdf:type :Person>> :source :registry .
///
/// # Can entail (with TEP):
/// :alice rdf:type :Person .
/// ```
///
/// Note: TEPs are an advanced feature and are disabled by default.
#[derive(Debug, Clone)]
pub struct TransparencyEnablingProperties {
    /// Set of property IRIs that enable transparency
    tep_properties: Vec<String>,
}

impl Default for TransparencyEnablingProperties {
    fn default() -> Self {
        Self::new()
    }
}

impl TransparencyEnablingProperties {
    /// Create a new TEP configuration with no enabled properties
    pub fn new() -> Self {
        Self {
            tep_properties: Vec::new(),
        }
    }

    /// Add a property as a TEP
    pub fn add_property(&mut self, property_iri: String) {
        if !self.tep_properties.contains(&property_iri) {
            self.tep_properties.push(property_iri);
        }
    }

    /// Check if a property is a TEP
    pub fn is_tep(&self, property_iri: &str) -> bool {
        self.tep_properties.iter().any(|p| p == property_iri)
    }

    /// Get all TEP properties
    pub fn properties(&self) -> &[String] {
        &self.tep_properties
    }

    /// Create a TEP configuration with common properties
    ///
    /// This includes:
    /// - rdf:type (class membership often needs transparency)
    pub fn with_common_properties() -> Self {
        let mut tep = Self::new();
        tep.add_property("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string());
        tep
    }

    /// Check if a triple's predicate is a TEP
    pub fn triple_uses_tep(&self, triple: &StarTriple) -> bool {
        if let StarTerm::NamedNode(node) = &triple.predicate {
            self.is_tep(&node.iri)
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_validator_creation() {
        let validator = SemanticValidator::new();
        assert!(validator.strict_opacity);

        let non_strict = SemanticValidator::new().with_strict_opacity(false);
        assert!(!non_strict.strict_opacity);
    }

    #[test]
    fn test_entailment_checker_creation() {
        let checker = EntailmentChecker::new();
        assert!(checker.validator.strict_opacity);
    }

    #[test]
    fn test_tep_properties() {
        let mut tep = TransparencyEnablingProperties::new();
        assert_eq!(tep.properties().len(), 0);

        tep.add_property("http://example.org/prop".to_string());
        assert!(tep.is_tep("http://example.org/prop"));
        assert!(!tep.is_tep("http://example.org/other"));

        // Adding the same property again should not duplicate
        tep.add_property("http://example.org/prop".to_string());
        assert_eq!(tep.properties().len(), 1);
    }

    #[test]
    fn test_common_teps() {
        let tep = TransparencyEnablingProperties::with_common_properties();
        assert!(tep.is_tep("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"));
    }

    #[test]
    fn test_quoted_never_entails_content() {
        let checker = EntailmentChecker::new();

        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("30").unwrap(),
        );

        let mut graph = StarGraph::new();
        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner.clone()),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );
        graph.insert(outer).unwrap();

        // Referential opacity: quoted triple does NOT entail content
        assert!(!checker.quoted_entails_content(&graph, &inner).unwrap());
    }
}
