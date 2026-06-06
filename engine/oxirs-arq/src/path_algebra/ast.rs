/// An IRI reference (URI string) for use in property path expressions.
pub type Iri = String;

/// Direction of traversal in a property path evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathDirection {
    Forward,
    Backward,
}

/// An item in a negated property set: a predicate IRI used either forward or inverse.
///
/// SPARQL 1.2 §18.1.6 — NPS items appear inside `!(p1 | p2 | ^p3 | ...)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NpsItem {
    Forward(Iri),
    Inverse(Iri),
}

/// Full SPARQL 1.2 property path AST.
///
/// Implements the grammar from SPARQL 1.2 §18.1 (property path expressions).
/// The `Box` indirection is required for recursive variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropertyPath {
    /// Direct predicate traversal: `<iri>`
    Link(Iri),
    /// Inverse path: `^p` — traverses the predicate in reverse (object→subject)
    Inverse(Box<PropertyPath>),
    /// Sequence: `p1/p2` — p1 then p2
    Sequence(Box<PropertyPath>, Box<PropertyPath>),
    /// Alternative: `p1|p2` — p1 or p2
    Alternative(Box<PropertyPath>, Box<PropertyPath>),
    /// Kleene star: `p*` — zero or more steps; start node is always included
    ZeroOrMore(Box<PropertyPath>),
    /// One or more steps: `p+` — start node excluded per SPARQL semantics
    OneOrMore(Box<PropertyPath>),
    /// Optional: `p?` — zero or one step
    Optional(Box<PropertyPath>),
    /// Negated property set: `!(iri1 | ^iri2 | ...)` — any predicate NOT in the set
    NegatedPropertySet(Vec<NpsItem>),
}
