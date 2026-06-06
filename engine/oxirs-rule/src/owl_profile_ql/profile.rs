//! OWL 2 QL profile types: simplified ontology axioms and class expressions.
//!
//! These types model a logical OWL 2 axiom in a form suitable for profile compliance
//! checking. They are intentionally decoupled from the broader OWL parser/reasoner
//! types in [`crate::owl`] / [`crate::owl2`] so the profile checker stays focused
//! on the syntactic restrictions defined by the OWL 2 QL specification.

/// A logical OWL 2 axiom (simplified for profile checking).
#[derive(Debug, Clone, PartialEq)]
pub enum OntologyAxiom {
    SubClassOf {
        sub: ClassExpr,
        sup: ClassExpr,
    },
    EquivalentClasses(Vec<ClassExpr>),
    SubObjectPropertyOf {
        sub: String,
        sup: String,
    },
    SubObjectPropertyChain {
        chain: Vec<String>,
        sup: String,
    },
    InverseObjectProperties {
        p1: String,
        p2: String,
    },
    TransitiveObjectProperty(String),
    SymmetricObjectProperty(String),
    AsymmetricObjectProperty(String),
    ReflexiveObjectProperty(String),
    IrreflexiveObjectProperty(String),
    FunctionalObjectProperty(String),
    InverseFunctionalObjectProperty(String),
    DisjointObjectProperties(Vec<String>),
    ObjectPropertyDomain {
        property: String,
        domain: ClassExpr,
    },
    ObjectPropertyRange {
        property: String,
        range: ClassExpr,
    },
    DataPropertyDomain {
        property: String,
        domain: ClassExpr,
    },
    DataPropertyRange {
        property: String,
        range: String,
    },
    FunctionalDataProperty(String),
    ClassAssertion {
        class: ClassExpr,
        individual: String,
    },
    ObjectPropertyAssertion {
        property: String,
        subject: String,
        object: String,
    },
    DataPropertyAssertion {
        property: String,
        subject: String,
        value: String,
    },
    NegativeObjectPropertyAssertion {
        property: String,
        subject: String,
        object: String,
    },
    DisjointUnion {
        class: String,
        classes: Vec<ClassExpr>,
    },
}

/// An OWL 2 class expression (simplified for profile checking).
#[derive(Debug, Clone, PartialEq)]
pub enum ClassExpr {
    Named(String),
    Thing,
    Nothing,
    IntersectionOf(Vec<ClassExpr>),
    UnionOf(Vec<ClassExpr>),
    ComplementOf(Box<ClassExpr>),
    SomeValuesFrom {
        property: String,
        filler: Box<ClassExpr>,
    },
    AllValuesFrom {
        property: String,
        filler: Box<ClassExpr>,
    },
    HasValue {
        property: String,
        individual: String,
    },
    OneOf(Vec<String>),
    MinCardinality {
        n: u32,
        property: String,
        filler: Option<Box<ClassExpr>>,
    },
    MaxCardinality {
        n: u32,
        property: String,
        filler: Option<Box<ClassExpr>>,
    },
    ExactCardinality {
        n: u32,
        property: String,
        filler: Option<Box<ClassExpr>>,
    },
    HasSelf(String),
}

impl ClassExpr {
    /// QL "subClassExpression": atomic class OR `SomeValuesFrom(P owl:Thing)`.
    pub fn is_ql_sub_class_expression(&self) -> bool {
        match self {
            ClassExpr::Named(_) | ClassExpr::Thing | ClassExpr::Nothing => true,
            ClassExpr::SomeValuesFrom { filler, .. } => {
                matches!(filler.as_ref(), ClassExpr::Thing)
            }
            _ => false,
        }
    }

    /// QL "superClassExpression": atomic, intersection of atomic, or
    /// `SomeValuesFrom(P C)` with atomic `C`, or `ComplementOf(subClassExpression)`.
    pub fn is_ql_super_class_expression(&self) -> bool {
        match self {
            ClassExpr::Named(_) | ClassExpr::Thing | ClassExpr::Nothing => true,
            ClassExpr::IntersectionOf(parts) => parts.iter().all(|p| p.is_atomic()),
            ClassExpr::SomeValuesFrom { filler, .. } => filler.is_atomic(),
            ClassExpr::ComplementOf(inner) => inner.is_ql_sub_class_expression(),
            _ => false,
        }
    }

    /// True if this is an atomic class (Named, Thing, or Nothing).
    pub fn is_atomic(&self) -> bool {
        matches!(
            self,
            ClassExpr::Named(_) | ClassExpr::Thing | ClassExpr::Nothing
        )
    }
}
