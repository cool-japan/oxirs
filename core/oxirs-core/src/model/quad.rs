//! RDF Quad implementation

use crate::model::{
    BlankNode, NamedNode, Object, ObjectRef, Predicate, PredicateRef, Subject, SubjectRef, Triple,
    TripleRef, Variable,
};
use std::fmt;
use std::hash::Hash;

/// Union type for terms that can be graph names
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum GraphName {
    NamedNode(NamedNode),
    BlankNode(BlankNode),
    Variable(Variable),
    DefaultGraph,
}

impl GraphName {
    /// Returns true if this is the default graph
    pub fn is_default_graph(&self) -> bool {
        matches!(self, GraphName::DefaultGraph)
    }
}

impl fmt::Display for GraphName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphName::NamedNode(n) => write!(f, "{n}"),
            GraphName::BlankNode(b) => write!(f, "{b}"),
            GraphName::Variable(v) => write!(f, "{v}"),
            GraphName::DefaultGraph => write!(f, "DEFAULT"),
        }
    }
}

impl AsRef<str> for GraphName {
    fn as_ref(&self) -> &str {
        match self {
            GraphName::NamedNode(n) => n.as_str(),
            GraphName::BlankNode(b) => b.as_str(),
            GraphName::Variable(v) => v.name(),
            GraphName::DefaultGraph => "DEFAULT",
        }
    }
}

impl From<NamedNode> for GraphName {
    fn from(node: NamedNode) -> Self {
        GraphName::NamedNode(node)
    }
}

impl From<BlankNode> for GraphName {
    fn from(node: BlankNode) -> Self {
        GraphName::BlankNode(node)
    }
}

impl From<Variable> for GraphName {
    fn from(variable: Variable) -> Self {
        GraphName::Variable(variable)
    }
}

impl From<crate::model::node::NamedOrBlankNode> for GraphName {
    fn from(node: crate::model::node::NamedOrBlankNode) -> Self {
        match node {
            crate::model::node::NamedOrBlankNode::NamedNode(n) => GraphName::NamedNode(n),
            crate::model::node::NamedOrBlankNode::BlankNode(b) => GraphName::BlankNode(b),
        }
    }
}

/// An RDF Quad
///
/// Represents an RDF statement with subject, predicate, object, and graph name.
/// This is used in RDF datasets where triples can belong to different named graphs.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct Quad {
    subject: Subject,
    predicate: Predicate,
    object: Object,
    graph_name: GraphName,
}

impl Quad {
    /// Creates a new quad
    pub fn new(
        subject: impl Into<Subject>,
        predicate: impl Into<Predicate>,
        object: impl Into<Object>,
        graph_name: impl Into<GraphName>,
    ) -> Self {
        Quad {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            graph_name: graph_name.into(),
        }
    }

    /// Creates a new quad in the default graph
    pub fn new_default_graph(
        subject: impl Into<Subject>,
        predicate: impl Into<Predicate>,
        object: impl Into<Object>,
    ) -> Self {
        Quad {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            graph_name: GraphName::DefaultGraph,
        }
    }

    /// Creates a quad from a triple, placing it in the default graph
    pub fn from_triple(triple: Triple) -> Self {
        let (subject, predicate, object) = triple.into_parts();
        Quad::new_default_graph(subject, predicate, object)
    }

    /// Creates a quad from a triple with a specific graph name
    pub fn from_triple_in_graph(triple: Triple, graph_name: impl Into<GraphName>) -> Self {
        let (subject, predicate, object) = triple.into_parts();
        Quad::new(subject, predicate, object, graph_name)
    }

    /// Returns the subject of this quad
    pub fn subject(&self) -> &Subject {
        &self.subject
    }

    /// Returns the predicate of this quad
    pub fn predicate(&self) -> &Predicate {
        &self.predicate
    }

    /// Returns the object of this quad
    pub fn object(&self) -> &Object {
        &self.object
    }

    /// Returns the graph name of this quad
    pub fn graph_name(&self) -> &GraphName {
        &self.graph_name
    }

    /// Decomposes the quad into its components
    pub fn into_parts(self) -> (Subject, Predicate, Object, GraphName) {
        (self.subject, self.predicate, self.object, self.graph_name)
    }

    /// Converts this quad to a triple, discarding the graph name
    pub fn to_triple(&self) -> Triple {
        Triple::new(
            self.subject.clone(),
            self.predicate.clone(),
            self.object.clone(),
        )
    }

    /// Returns a reference to this quad
    pub fn as_ref(&self) -> QuadRef<'_> {
        QuadRef::from(self)
    }

    /// Returns true if this quad is in the default graph
    pub fn is_default_graph(&self) -> bool {
        matches!(self.graph_name, GraphName::DefaultGraph)
    }

    /// Returns the triple if this quad is in the default graph, None otherwise
    pub fn triple_in_default_graph(&self) -> Option<Triple> {
        if self.is_default_graph() {
            Some(self.to_triple())
        } else {
            None
        }
    }

    /// Returns true if this quad contains any variables
    pub fn has_variables(&self) -> bool {
        matches!(self.subject, Subject::Variable(_))
            || matches!(self.predicate, Predicate::Variable(_))
            || matches!(self.object, Object::Variable(_))
            || matches!(self.graph_name, GraphName::Variable(_))
    }

    /// Returns true if this quad is ground (contains no variables)
    pub fn is_ground(&self) -> bool {
        !self.has_variables()
    }
}

impl fmt::Display for Quad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.graph_name {
            GraphName::DefaultGraph => {
                write!(f, "{} {} {} .", self.subject, self.predicate, self.object)
            }
            graph => write!(
                f,
                "{} {} {} {} .",
                self.subject, self.predicate, self.object, graph
            ),
        }
    }
}

/// A borrowed quad reference for zero-copy operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuadRef<'a> {
    subject: SubjectRef<'a>,
    predicate: PredicateRef<'a>,
    object: ObjectRef<'a>,
    graph_name: GraphNameRef<'a>,
}

impl<'a> QuadRef<'a> {
    /// Creates a new quad reference
    pub fn new(
        subject: SubjectRef<'a>,
        predicate: PredicateRef<'a>,
        object: ObjectRef<'a>,
        graph_name: GraphNameRef<'a>,
    ) -> Self {
        QuadRef {
            subject,
            predicate,
            object,
            graph_name,
        }
    }

    /// Returns the subject
    pub fn subject(&self) -> SubjectRef<'a> {
        self.subject
    }

    /// Returns the predicate
    pub fn predicate(&self) -> PredicateRef<'a> {
        self.predicate
    }

    /// Returns the object
    pub fn object(&self) -> ObjectRef<'a> {
        self.object
    }

    /// Returns the graph name
    pub fn graph_name(&self) -> GraphNameRef<'a> {
        self.graph_name
    }

    /// Converts to an owned quad
    pub fn to_owned(&self) -> Quad {
        Quad {
            subject: self.subject.to_owned(),
            predicate: self.predicate.to_owned(),
            object: self.object.to_owned(),
            graph_name: self.graph_name.to_owned(),
        }
    }

    /// Converts to a triple reference, discarding the graph name
    pub fn to_triple_ref(&self) -> TripleRef<'a> {
        TripleRef::new(self.subject, self.predicate, self.object)
    }

    /// Returns the triple part of this quad (alias for to_triple_ref)
    pub fn triple(&self) -> TripleRef<'a> {
        self.to_triple_ref()
    }

    /// Converts to an owned quad (alias for to_owned)
    pub fn into_owned(self) -> Quad {
        self.to_owned()
    }
}

impl<'a> fmt::Display for QuadRef<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.graph_name {
            GraphNameRef::DefaultGraph => {
                write!(f, "{} {} {} .", self.subject, self.predicate, self.object)
            }
            graph => write!(
                f,
                "{} {} {} {} .",
                self.subject, self.predicate, self.object, graph
            ),
        }
    }
}

/// Borrowed graph name reference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphNameRef<'a> {
    NamedNode(&'a NamedNode),
    BlankNode(&'a BlankNode),
    Variable(&'a Variable),
    DefaultGraph,
}

impl<'a> GraphNameRef<'a> {
    /// Returns true if this is the default graph
    pub fn is_default_graph(&self) -> bool {
        matches!(self, GraphNameRef::DefaultGraph)
    }

    /// Converts to an owned graph name
    pub fn to_owned(&self) -> GraphName {
        match self {
            GraphNameRef::NamedNode(n) => GraphName::NamedNode((*n).clone()),
            GraphNameRef::BlankNode(b) => GraphName::BlankNode((*b).clone()),
            GraphNameRef::Variable(v) => GraphName::Variable((*v).clone()),
            GraphNameRef::DefaultGraph => GraphName::DefaultGraph,
        }
    }
}

impl<'a> fmt::Display for GraphNameRef<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphNameRef::NamedNode(n) => write!(f, "{n}"),
            GraphNameRef::BlankNode(b) => write!(f, "{b}"),
            GraphNameRef::Variable(v) => write!(f, "{v}"),
            GraphNameRef::DefaultGraph => write!(f, "DEFAULT"),
        }
    }
}

// Conversion implementations
impl<'a> From<&'a GraphName> for GraphNameRef<'a> {
    fn from(graph_name: &'a GraphName) -> Self {
        match graph_name {
            GraphName::NamedNode(n) => GraphNameRef::NamedNode(n),
            GraphName::BlankNode(b) => GraphNameRef::BlankNode(b),
            GraphName::Variable(v) => GraphNameRef::Variable(v),
            GraphName::DefaultGraph => GraphNameRef::DefaultGraph,
        }
    }
}

impl<'a> From<&'a Quad> for QuadRef<'a> {
    fn from(quad: &'a Quad) -> Self {
        QuadRef {
            subject: quad.subject().into(),
            predicate: quad.predicate().into(),
            object: quad.object().into(),
            graph_name: quad.graph_name().into(),
        }
    }
}

impl<'a> From<QuadRef<'a>> for Quad {
    fn from(quad_ref: QuadRef<'a>) -> Self {
        quad_ref.to_owned()
    }
}

impl From<Triple> for Quad {
    fn from(triple: Triple) -> Self {
        Quad::from_triple(triple)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode};

    #[test]
    fn test_quad_creation() {
        let subject = NamedNode::new("http://example.org/subject").unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = Literal::new("object");
        let graph = NamedNode::new("http://example.org/graph").unwrap();

        let quad = Quad::new(
            subject.clone(),
            predicate.clone(),
            object.clone(),
            graph.clone(),
        );

        assert!(quad.is_ground());
        assert!(!quad.has_variables());
        assert!(!quad.is_default_graph());
    }

    #[test]
    fn test_quad_default_graph() {
        let subject = NamedNode::new("http://example.org/subject").unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = Literal::new("object");

        let quad = Quad::new_default_graph(subject, predicate, object);

        assert!(quad.is_default_graph());
        assert!(quad.graph_name().is_default_graph());
    }

    #[test]
    fn test_quad_from_triple() {
        let subject = NamedNode::new("http://example.org/subject").unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = Literal::new("object");

        let triple = Triple::new(subject.clone(), predicate.clone(), object.clone());
        let quad = Quad::from_triple(triple.clone());

        assert!(quad.is_default_graph());
        assert_eq!(quad.to_triple().subject(), triple.subject());
        assert_eq!(quad.to_triple().predicate(), triple.predicate());
        assert_eq!(quad.to_triple().object(), triple.object());
    }

    #[test]
    fn test_quad_with_variable() {
        let subject = Variable::new("x").unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = Literal::new("object");
        let graph = Variable::new("g").unwrap();

        let quad = Quad::new(subject, predicate, object, graph);

        assert!(!quad.is_ground());
        assert!(quad.has_variables());
    }

    #[test]
    fn test_quad_ref() {
        let subject = NamedNode::new("http://example.org/s").unwrap();
        let predicate = NamedNode::new("http://example.org/p").unwrap();
        let object = Literal::new("o");
        let graph = NamedNode::new("http://example.org/g").unwrap();

        let quad = Quad::new(subject, predicate, object, graph);
        let quad_ref = QuadRef::from(&quad);
        let quad_owned = quad_ref.to_owned();

        assert_eq!(quad, quad_owned);
    }
}
