//! RDF mapping structs, entity types, and property mapping types for physics simulations.

// ──────────────────────────────────────────────────────────────────────────────
// Namespace constants
// ──────────────────────────────────────────────────────────────────────────────

pub const NS_SOSA: &str = "http://www.w3.org/ns/sosa/";
pub const NS_SSN: &str = "http://www.w3.org/ns/ssn/";
pub const NS_QUDT: &str = "http://qudt.org/schema/qudt/";
pub const NS_UNIT: &str = "http://qudt.org/vocab/unit/";
pub const NS_EX: &str = "http://oxirs.org/example/physics#";
pub const NS_PHYS: &str = "http://oxirs.org/physics#";
pub const NS_PROV: &str = "http://www.w3.org/ns/prov#";
pub const NS_XSD: &str = "http://www.w3.org/2001/XMLSchema#";
pub const NS_RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
pub const NS_RDFS: &str = "http://www.w3.org/2000/01/rdf-schema#";

// ──────────────────────────────────────────────────────────────────────────────
// RDF triple representation (lightweight, no store dependency)
// ──────────────────────────────────────────────────────────────────────────────

/// A minimal N-Triple-style triple (subject, predicate, object as strings).
#[derive(Debug, Clone, PartialEq)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl Triple {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }

    /// Format as a Turtle triple statement (without trailing `.`).
    pub fn to_turtle_statement(&self) -> String {
        format!("{} {} {}", self.subject, self.predicate, self.object)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Parsed boundary condition (physics-level, no oxirs-core dependency here)
// ──────────────────────────────────────────────────────────────────────────────

/// A boundary condition extracted from RDF.
#[derive(Debug, Clone, PartialEq)]
pub struct RdfBoundaryCondition {
    /// Subject IRI of the boundary condition node.
    pub iri: String,
    /// Condition type (e.g. `"inlet"`, `"wall"`, `"outlet"`).
    pub condition_type: String,
    /// Name of the physical property.
    pub property: String,
    /// Numeric value in SI units.
    pub value: f64,
    /// QUDT unit suffix.
    pub unit: String,
}

/// A material property extracted from RDF.
#[derive(Debug, Clone, PartialEq)]
pub struct RdfMaterialProperty {
    /// Subject IRI of the material node.
    pub iri: String,
    /// Material or property name.
    pub name: String,
    /// Numeric value.
    pub value: f64,
    /// QUDT unit suffix.
    pub unit: String,
    /// Optional description.
    pub description: Option<String>,
}

/// PhysicsToRdf configuration flags.
///
/// Controls which optional annotation families are included in the generated
/// triple set.
#[derive(Debug, Clone)]
pub struct PhysicsToRdfConfig {
    /// Base IRI for generated entities.
    pub base_iri: String,
    /// Include provenance triples (W3C PROV).
    pub include_provenance: bool,
    /// Include digital twin state triples.
    pub include_digital_twin: bool,
    /// Include QUDT unit annotations.
    pub include_units: bool,
}

impl Default for PhysicsToRdfConfig {
    fn default() -> Self {
        Self {
            base_iri: NS_EX.to_string(),
            include_provenance: true,
            include_digital_twin: true,
            include_units: true,
        }
    }
}
