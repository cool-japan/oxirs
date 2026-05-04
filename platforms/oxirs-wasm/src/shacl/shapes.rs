//! SHACL shapes — data structures for NodeShapes and PropertyShapes.

use serde::{Deserialize, Serialize};

/// A constraint that can appear on a `PropertyShape`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum ShaclConstraint {
    /// `sh:minCount` — minimum number of values.
    MinCount(usize),
    /// `sh:maxCount` — maximum number of values.
    MaxCount(usize),
    /// `sh:datatype` — required XSD datatype IRI.
    Datatype(String),
    /// `sh:pattern` — regex pattern the string value must match.
    Pattern(String),
    /// `sh:minInclusive` — minimum numeric value (inclusive).
    MinInclusive(f64),
    /// `sh:maxInclusive` — maximum numeric value (inclusive).
    MaxInclusive(f64),
    /// `sh:class` — required rdf:type IRI of the value node.
    Class(String),
    /// `sh:nodeKind` — required node kind: `"IRI"`, `"Literal"`, `"BlankNode"`.
    NodeKind(NodeKind),
    /// `sh:in` — value must be one of the listed strings.
    In(Vec<String>),
    /// `sh:hasValue` — at least one value must equal this string.
    HasValue(String),
}

/// RDF node kinds for `sh:nodeKind`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    Iri,
    Literal,
    BlankNode,
    BlankNodeOrIri,
    BlankNodeOrLiteral,
    IriOrLiteral,
}

impl std::str::FromStr for NodeKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "IRI" | "sh:IRI" => Ok(Self::Iri),
            "Literal" | "sh:Literal" => Ok(Self::Literal),
            "BlankNode" | "sh:BlankNode" => Ok(Self::BlankNode),
            "BlankNodeOrIRI" | "sh:BlankNodeOrIRI" => Ok(Self::BlankNodeOrIri),
            "BlankNodeOrLiteral" | "sh:BlankNodeOrLiteral" => Ok(Self::BlankNodeOrLiteral),
            "IRIOrLiteral" | "sh:IRIOrLiteral" => Ok(Self::IriOrLiteral),
            _ => Err(format!("unknown NodeKind: {s:?}")),
        }
    }
}

/// A `sh:PropertyShape` — constrains values of a specific property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyShape {
    /// The property IRI being constrained (`sh:path`).
    pub path: String,
    /// Optional human-readable name.
    pub name: Option<String>,
    /// Constraints that apply to values of this property.
    pub constraints: Vec<ShaclConstraint>,
}

impl PropertyShape {
    /// Create a property shape for the given path.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            name: None,
            constraints: Vec::new(),
        }
    }

    /// Add a constraint.
    pub fn with_constraint(mut self, c: ShaclConstraint) -> Self {
        self.constraints.push(c);
        self
    }

    /// Named shape.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// A `sh:NodeShape` — groups constraints that apply to a focus node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeShape {
    /// Shape identifier (IRI).
    pub id: String,
    /// `sh:targetClass` — validates all nodes of this rdf:type.
    pub target_class: Option<String>,
    /// `sh:targetNode` — validates a specific node.
    pub target_node: Option<String>,
    /// Property shapes nested inside this node shape.
    pub properties: Vec<PropertyShape>,
    /// Whether the shape is `sh:closed` (no additional properties allowed).
    pub closed: bool,
}

impl NodeShape {
    /// Create a node shape with the given IRI.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            target_class: None,
            target_node: None,
            properties: Vec::new(),
            closed: false,
        }
    }

    /// Set `sh:targetClass`.
    pub fn targeting_class(mut self, class: impl Into<String>) -> Self {
        self.target_class = Some(class.into());
        self
    }

    /// Add a property shape.
    pub fn with_property(mut self, prop: PropertyShape) -> Self {
        self.properties.push(prop);
        self
    }

    /// Mark the shape as closed.
    pub fn closed(mut self) -> Self {
        self.closed = true;
        self
    }
}
