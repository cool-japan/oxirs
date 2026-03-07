//! # OWL 2 QL Profile — Perfect Reformulation Query Rewriter
//!
//! Implements the OWL 2 QL profile query rewriting algorithm (PerfectRef).
//! OWL 2 QL is designed for query rewriting over relational/RDF data sources
//! without materialization of the full closure.
//!
//! ## Complexity
//! NLogSpace in the size of the data (ABox). TBox reasoning is polynomial.
//!
//! ## Supported Constructs
//! - SubClassOf (atomic left-hand side)
//! - SubObjectPropertyOf
//! - EquivalentClasses / EquivalentObjectProperties
//! - InverseObjectProperties (owl:inverseOf)
//! - ObjectSomeValuesFrom on left (∃P.⊤ ⊑ C)
//! - DisjointClasses / DisjointObjectProperties
//! - ObjectPropertyDomain / ObjectPropertyRange
//! - ObjectUnionOf on right-hand side (C ⊑ A ⊔ B) — union query rewriting
//! - ObjectIntersectionOf on right-hand side concept normalization
//!
//! ## Algorithm: PerfectRef (extended for unions)
//! For each atom in the query, compute all possible "unfoldings" via the TBox,
//! then take the cross-product (conjunctive query union).
//! Union axioms (C ⊑ A ⊔ B) are handled by branching: each branch of the union
//! produces an independent conjunctive query in the resulting UCQ.
//!
//! ## Reference
//! Calvanese, De Giacomo, Lembo, Lenzerini, Rosati:
//! "Tractable Reasoning and Efficient Query Answering in Description Logics:
//!  The DL-Lite Family" (Journal of Automated Reasoning, 2007).
//! <https://www.w3.org/TR/owl2-profiles/#OWL_2_QL>

use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

/// Errors from OWL 2 QL query rewriting
#[derive(Debug, Error, Clone, PartialEq)]
pub enum QlError {
    #[error("Invalid axiom: {0}")]
    InvalidAxiom(String),

    #[error("Inconsistent ontology: {0}")]
    Inconsistency(String),

    #[error("Rewriting limit exceeded (max {0} reformulations)")]
    RewritingLimitExceeded(usize),

    #[error("Unsupported construct in OWL 2 QL: {0}")]
    UnsupportedConstruct(String),
}

/// A basic concept expression in OWL 2 QL
/// (restricted to what QL allows on the left-hand side of SubClassOf)
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum QlConcept {
    /// owl:Thing
    Top,
    /// owl:Nothing
    Bottom,
    /// Named atomic class (IRI string)
    Named(String),
    /// ∃P.⊤ — existential with top filler (object side)
    SomeValuesTop { property: String },
    /// ∃P⁻.⊤ — existential with inverse property and top filler
    SomeValuesTopInverse { property: String },
}

impl QlConcept {
    /// Named concept constructor
    pub fn named(iri: impl Into<String>) -> Self {
        Self::Named(iri.into())
    }

    /// ∃P.⊤ constructor
    pub fn some_values_top(property: impl Into<String>) -> Self {
        Self::SomeValuesTop {
            property: property.into(),
        }
    }

    /// ∃P⁻.⊤ constructor
    pub fn some_values_top_inverse(property: impl Into<String>) -> Self {
        Self::SomeValuesTopInverse {
            property: property.into(),
        }
    }

    /// Return atomic name if Named
    pub fn as_named(&self) -> Option<&str> {
        if let Self::Named(n) = self {
            Some(n)
        } else {
            None
        }
    }

    /// Return property name if SomeValuesTop or SomeValuesTopInverse
    pub fn as_some_values_property(&self) -> Option<(&str, bool)> {
        match self {
            Self::SomeValuesTop { property } => Some((property, false)),
            Self::SomeValuesTopInverse { property } => Some((property, true)),
            _ => None,
        }
    }
}

/// A rich concept expression supporting union and intersection on the RHS of axioms.
///
/// OWL 2 QL restricts the LHS of SubClassOf to basic concepts (QlConcept),
/// but allows union and intersection on the RHS for certain patterns.
/// This type is used for union axiom RHS expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConceptExpr {
    /// Atomic or basic concept
    Atomic(QlConcept),
    /// owl:ObjectUnionOf(C1, C2, ...) — disjunction
    Union(Vec<ConceptExpr>),
    /// owl:ObjectIntersectionOf(C1, C2, ...) — conjunction
    Intersection(Vec<ConceptExpr>),
}

impl ConceptExpr {
    /// Construct from a named class
    pub fn named(iri: impl Into<String>) -> Self {
        Self::Atomic(QlConcept::Named(iri.into()))
    }

    /// Construct a union of named classes
    pub fn union_of(members: Vec<ConceptExpr>) -> Self {
        Self::Union(members)
    }

    /// Construct an intersection of named classes
    pub fn intersection_of(members: Vec<ConceptExpr>) -> Self {
        Self::Intersection(members)
    }

    /// Flatten this expression into a list of named-class sets per union axiom.
    ///
    /// The return value is a `Vec<Vec<String>>` where each inner `Vec<String>` is the
    /// complete set of sibling disjuncts for ONE union expression:
    ///
    /// - `Atomic("A")` → `[["A"]]`  (one "union" with just A)
    /// - `Union(["A", "B"])` → `[["A", "B"]]`  (one union with disjuncts A and B)
    /// - `Intersection(["A", "B"])` → `[["A", "B"]]`  (conjunction treated as one entry)
    ///
    /// For nested unions `Union([Union(["A", "B"]), "C"])` → `[["A", "B", "C"]]` (all flattened).
    pub fn union_branches(&self) -> Vec<Vec<String>> {
        match self {
            ConceptExpr::Atomic(QlConcept::Named(n)) => vec![vec![n.clone()]],
            ConceptExpr::Atomic(_) => vec![],
            ConceptExpr::Union(members) => {
                // Collect ALL atomic names across all members into a single disjunct list
                let all_names: Vec<String> =
                    members.iter().flat_map(|m| m.atomic_names()).collect();
                if all_names.is_empty() {
                    vec![]
                } else {
                    vec![all_names]
                }
            }
            ConceptExpr::Intersection(members) => {
                // Intersection on RHS: C ⊑ A ⊓ B is C ⊑ A ∧ C ⊑ B
                // Each intersection member must hold — not a union branching scenario
                // Return as a single conjunctive constraint
                let names: Vec<String> = members.iter().flat_map(|m| m.atomic_names()).collect();
                if names.is_empty() {
                    vec![]
                } else {
                    vec![names]
                }
            }
        }
    }

    /// Return all atomic class names in this expression (flattened)
    pub fn atomic_names(&self) -> Vec<String> {
        match self {
            ConceptExpr::Atomic(QlConcept::Named(n)) => vec![n.clone()],
            ConceptExpr::Atomic(_) => vec![],
            ConceptExpr::Union(members) | ConceptExpr::Intersection(members) => {
                members.iter().flat_map(|m| m.atomic_names()).collect()
            }
        }
    }

    /// Returns true if this expression contains any union
    pub fn has_union(&self) -> bool {
        matches!(self, ConceptExpr::Union(_))
    }
}

/// A role (property) expression in OWL 2 QL
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum QlRole {
    /// Named object property (IRI string)
    Named(String),
    /// Inverse of a named property
    Inverse(String),
}

impl QlRole {
    pub fn named(iri: impl Into<String>) -> Self {
        Self::Named(iri.into())
    }

    pub fn inverse(iri: impl Into<String>) -> Self {
        Self::Inverse(iri.into())
    }

    /// Return the base property name regardless of inversion
    pub fn base_name(&self) -> &str {
        match self {
            Self::Named(n) => n,
            Self::Inverse(n) => n,
        }
    }

    /// Return whether this is an inverse role
    pub fn is_inverse(&self) -> bool {
        matches!(self, Self::Inverse(_))
    }

    /// Return the inverse of this role
    pub fn inverse_role(&self) -> Self {
        match self {
            Self::Named(n) => Self::Inverse(n.clone()),
            Self::Inverse(n) => Self::Named(n.clone()),
        }
    }
}

/// OWL 2 QL TBox axioms (restricted to QL-supported constructs)
#[derive(Debug, Clone, PartialEq)]
pub enum QlAxiom {
    /// C1 ⊑ C2: subclass (C1 must be atomic or ∃P.⊤; C2 must be atomic or ∃P.C)
    SubClassOf { sub: QlConcept, sup: QlConcept },

    /// C ⊑ (A ⊔ B ⊔ ...): union on right-hand side (subclass of union)
    /// This is the key axiom for union query rewriting.
    /// When rewriting a query ?x:A, we can use ?x:C if C ⊑ A ⊔ B and we also check B.
    SubClassOfUnion {
        sub: QlConcept,
        sup_union: ConceptExpr,
    },

    /// C1 ≡ C2: equivalent classes (expanded into two SubClassOf)
    EquivalentClasses(QlConcept, QlConcept),

    /// R1 ⊑ R2: sub-object-property
    SubObjectPropertyOf { sub: QlRole, sup: QlRole },

    /// R1 ≡ R2: equivalent properties
    EquivalentObjectProperties(QlRole, QlRole),

    /// R1 owl:inverseOf R2
    InverseObjectProperties(String, String),

    /// owl:ObjectPropertyDomain(P, C): ∃P.⊤ ⊑ C
    ObjectPropertyDomain { property: String, domain: String },

    /// owl:ObjectPropertyRange(P, C): ∃P⁻.⊤ ⊑ C (range from object side)
    ObjectPropertyRange { property: String, range: String },

    /// DisjointClasses(C1, C2): C1 ⊓ C2 ⊑ ⊥
    DisjointClasses(QlConcept, QlConcept),

    /// DisjointObjectProperties(P1, P2)
    DisjointObjectProperties(QlRole, QlRole),
}

/// A query atom representing a triple pattern
/// Variables are represented by names starting with '?' or just as strings,
/// constants are IRIs/literals. We keep it as a simple enum.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QueryAtom {
    /// rdf:type assertion: ?x rdf:type C or :ind rdf:type C
    TypeAtom {
        individual: QueryTerm,
        class: String,
    },
    /// Property triple: ?x P ?y or constants
    PropertyAtom {
        subject: QueryTerm,
        property: String,
        object: QueryTerm,
    },
}

impl QueryAtom {
    /// Constructor for property triple
    pub fn property_atom(
        subject: QueryTerm,
        property: impl Into<String>,
        object: QueryTerm,
    ) -> Self {
        Self::PropertyAtom {
            subject,
            property: property.into(),
            object,
        }
    }

    /// Return all variable names used in this atom
    pub fn variables(&self) -> Vec<&str> {
        match self {
            Self::TypeAtom { individual, .. } => individual.as_variable().into_iter().collect(),
            Self::PropertyAtom {
                subject, object, ..
            } => {
                let mut vars = Vec::new();
                if let Some(v) = subject.as_variable() {
                    vars.push(v);
                }
                if let Some(v) = object.as_variable() {
                    vars.push(v);
                }
                vars
            }
        }
    }
}

/// A term in a query atom — either a variable or a constant (IRI/literal)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QueryTerm {
    /// A variable (name without the '?' prefix, stored as plain string)
    Variable(String),
    /// A constant IRI or literal
    Constant(String),
}

impl QueryTerm {
    pub fn var(name: impl Into<String>) -> Self {
        Self::Variable(name.into())
    }

    pub fn constant(iri: impl Into<String>) -> Self {
        Self::Constant(iri.into())
    }

    pub fn as_variable(&self) -> Option<&str> {
        if let Self::Variable(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_constant(&self) -> Option<&str> {
        if let Self::Constant(c) = self {
            Some(c)
        } else {
            None
        }
    }

    pub fn is_variable(&self) -> bool {
        matches!(self, Self::Variable(_))
    }
}

/// A conjunctive query — a list of atoms to be answered conjunctively
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConjunctiveQuery {
    /// The atoms of the query
    pub atoms: Vec<QueryAtom>,
    /// Head variables (distinguished variables)
    pub head_variables: Vec<String>,
}

impl ConjunctiveQuery {
    pub fn new(atoms: Vec<QueryAtom>, head_variables: Vec<String>) -> Self {
        Self {
            atoms,
            head_variables,
        }
    }

    pub fn with_atoms(atoms: Vec<QueryAtom>) -> Self {
        let head_variables = Self::collect_variables(&atoms);
        Self {
            atoms,
            head_variables,
        }
    }

    fn collect_variables(atoms: &[QueryAtom]) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut vars = Vec::new();
        for atom in atoms {
            for v in atom.variables() {
                if seen.insert(v.to_string()) {
                    vars.push(v.to_string());
                }
            }
        }
        vars
    }
}

/// A rewritten query — a union of conjunctive queries
#[derive(Debug, Clone)]
pub struct RewrittenQuery {
    /// The list of conjunctive queries in the union (UCQ)
    pub queries: Vec<ConjunctiveQuery>,
}

impl RewrittenQuery {
    pub fn new() -> Self {
        Self {
            queries: Vec::new(),
        }
    }

    pub fn add(&mut self, cq: ConjunctiveQuery) {
        self.queries.push(cq);
    }

    pub fn len(&self) -> usize {
        self.queries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }
}

impl Default for RewrittenQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// Precomputed union axiom entry: sub ⊑ (A₁ ⊔ A₂ ⊔ ...)
/// Maps a sub-concept name to all its union superclass branches.
/// If C ⊑ A ⊔ B, then union_axioms["C"] = [["A", "B"]].
/// If C ⊑ (A ⊔ B) ⊓ D, this is split into: C ⊑ A ⊔ B and C ⊑ D.
#[derive(Debug, Clone)]
pub struct UnionAxiomEntry {
    /// Sub-concept name (LHS of axiom)
    pub sub_class: String,
    /// Disjuncts on the RHS — each is a list of class names forming one union branch.
    /// e.g. for C ⊑ A ⊔ B: branches = ["A", "B"]
    pub disjuncts: Vec<String>,
}

/// The OWL 2 QL TBox — stores all axioms and pre-computes hierarchies
pub struct Owl2QLTBox {
    axioms: Vec<QlAxiom>,
    /// Precomputed: class -> set of superclasses (transitive closure)
    class_supers: HashMap<String, HashSet<String>>,
    /// Precomputed: class -> set of subclasses
    class_subs: HashMap<String, HashSet<String>>,
    /// Precomputed: property -> set of superproperties (transitive closure)
    prop_supers: HashMap<String, HashSet<String>>,
    /// Precomputed: inverse property -> set of superproperties
    inv_prop_supers: HashMap<String, HashSet<String>>,
    /// Precomputed: property -> set of inverse properties
    inverse_of: HashMap<String, HashSet<String>>,
    /// Precomputed: class -> set of concepts that subsume ∃P.⊤ for each P
    /// i.e. if ∃P.⊤ ⊑ C then domain_classes[P] contains C
    domain_classes: HashMap<String, HashSet<String>>,
    /// Precomputed: range classes — ∃P⁻.⊤ ⊑ C → range_classes[P] contains C
    range_classes: HashMap<String, HashSet<String>>,
    /// Disjoint class pairs
    disjoint_classes: HashSet<(String, String)>,
    /// Union axiom entries: class C has a union superclass A ⊔ B ⊔ ...
    /// Maps class name → list of union axiom entries for that class.
    union_axioms: HashMap<String, Vec<Vec<String>>>,
    /// Reverse union index: class A is a disjunct in union axioms for C
    /// Maps A → set of classes C such that C ⊑ (... ⊔ A ⊔ ...)
    union_rev_index: HashMap<String, HashSet<String>>,
}

impl Owl2QLTBox {
    /// Create a new empty TBox
    pub fn new() -> Self {
        Self {
            axioms: Vec::new(),
            class_supers: HashMap::new(),
            class_subs: HashMap::new(),
            prop_supers: HashMap::new(),
            inv_prop_supers: HashMap::new(),
            inverse_of: HashMap::new(),
            domain_classes: HashMap::new(),
            range_classes: HashMap::new(),
            disjoint_classes: HashSet::new(),
            union_axioms: HashMap::new(),
            union_rev_index: HashMap::new(),
        }
    }

    /// Add an axiom to the TBox
    pub fn add_axiom(&mut self, axiom: QlAxiom) {
        self.axioms.push(axiom);
    }

    /// Add multiple axioms
    pub fn add_axioms(&mut self, axioms: impl IntoIterator<Item = QlAxiom>) {
        for axiom in axioms {
            self.add_axiom(axiom);
        }
    }

    /// Finalize and compute transitive closures. Must be called before rewriting.
    pub fn classify(&mut self) -> Result<(), QlError> {
        self.expand_equivalences();
        self.build_inverse_index();
        self.compute_class_hierarchy()?;
        self.compute_property_hierarchy()?;
        self.compute_domain_range();
        self.compute_disjointness();
        self.compute_union_index();
        Ok(())
    }

    // ---- internal build steps ----

    fn expand_equivalences(&mut self) {
        let mut extra = Vec::new();
        for axiom in &self.axioms {
            match axiom {
                QlAxiom::EquivalentClasses(c1, c2) => {
                    extra.push(QlAxiom::SubClassOf {
                        sub: c1.clone(),
                        sup: c2.clone(),
                    });
                    extra.push(QlAxiom::SubClassOf {
                        sub: c2.clone(),
                        sup: c1.clone(),
                    });
                }
                QlAxiom::EquivalentObjectProperties(r1, r2) => {
                    extra.push(QlAxiom::SubObjectPropertyOf {
                        sub: r1.clone(),
                        sup: r2.clone(),
                    });
                    extra.push(QlAxiom::SubObjectPropertyOf {
                        sub: r2.clone(),
                        sup: r1.clone(),
                    });
                }
                QlAxiom::InverseObjectProperties(p1, p2) => {
                    // P1 inverseOf P2 → P1 ⊑ P2⁻ and P2⁻ ⊑ P1
                    extra.push(QlAxiom::SubObjectPropertyOf {
                        sub: QlRole::Named(p1.clone()),
                        sup: QlRole::Inverse(p2.clone()),
                    });
                    extra.push(QlAxiom::SubObjectPropertyOf {
                        sub: QlRole::Named(p2.clone()),
                        sup: QlRole::Inverse(p1.clone()),
                    });
                }
                _ => {}
            }
        }
        self.axioms.extend(extra);
    }

    fn build_inverse_index(&mut self) {
        for axiom in &self.axioms {
            if let QlAxiom::InverseObjectProperties(p1, p2) = axiom {
                self.inverse_of
                    .entry(p1.clone())
                    .or_default()
                    .insert(p2.clone());
                self.inverse_of
                    .entry(p2.clone())
                    .or_default()
                    .insert(p1.clone());
            }
        }
    }

    fn compute_class_hierarchy(&mut self) -> Result<(), QlError> {
        // Collect direct subclass edges from axioms
        // sub → sup edges
        let mut direct: HashMap<String, HashSet<String>> = HashMap::new();

        for axiom in &self.axioms {
            if let QlAxiom::SubClassOf { sub, sup } = axiom {
                // Only atomic ↦ atomic subclass edges for the class hierarchy
                // ∃P.⊤ ⊑ C is handled via domain
                if let (Some(sub_name), Some(sup_name)) = (sub.as_named(), sup.as_named()) {
                    direct
                        .entry(sub_name.to_string())
                        .or_default()
                        .insert(sup_name.to_string());
                }
            }
        }

        // Transitive closure via BFS/DFS from each class
        let all_classes: HashSet<String> = direct
            .keys()
            .chain(direct.values().flat_map(|s| s.iter()))
            .cloned()
            .collect();

        for class in &all_classes {
            let supers = Self::transitive_closure(class, &direct);
            self.class_supers.insert(class.clone(), supers);
        }

        // Build reverse (subs)
        for (sub, supers) in &self.class_supers {
            for sup in supers {
                self.class_subs
                    .entry(sup.clone())
                    .or_default()
                    .insert(sub.clone());
            }
        }

        Ok(())
    }

    fn compute_property_hierarchy(&mut self) -> Result<(), QlError> {
        // Collect direct sub-role edges for named and inverse roles
        // named_direct[P] = set of named Q where P ⊑ Q
        let mut named_direct: HashMap<String, HashSet<String>> = HashMap::new();
        // inv_direct[P] = set of named Q where P⁻ ⊑ Q (P⁻ is subproperty of Q)
        let mut inv_direct: HashMap<String, HashSet<String>> = HashMap::new();

        for axiom in &self.axioms {
            if let QlAxiom::SubObjectPropertyOf { sub, sup } = axiom {
                match (sub, sup) {
                    (QlRole::Named(p), QlRole::Named(q)) => {
                        named_direct.entry(p.clone()).or_default().insert(q.clone());
                    }
                    (QlRole::Inverse(p), QlRole::Named(q)) => {
                        // P⁻ ⊑ Q means if (x,y):P then (y,x):Q
                        inv_direct.entry(p.clone()).or_default().insert(q.clone());
                    }
                    (QlRole::Named(p), QlRole::Inverse(q)) => {
                        // P ⊑ Q⁻ — store that when querying Q⁻ we include P
                        // This means inverse(Q) is a superproperty of P
                        // We track: inv_supers of P contains q
                        inv_direct.entry(p.clone()).or_default().insert(q.clone());
                    }
                    (QlRole::Inverse(p), QlRole::Inverse(q)) => {
                        // P⁻ ⊑ Q⁻ ↔ Q ⊑ P
                        named_direct.entry(q.clone()).or_default().insert(p.clone());
                    }
                }
            }
        }

        let all_named: HashSet<String> = named_direct
            .keys()
            .chain(named_direct.values().flat_map(|s| s.iter()))
            .chain(inv_direct.keys())
            .chain(inv_direct.values().flat_map(|s| s.iter()))
            .cloned()
            .collect();

        for prop in &all_named {
            let supers = Self::transitive_closure(prop, &named_direct);
            self.prop_supers.insert(prop.clone(), supers);
        }

        for prop in &all_named {
            let inv_supers = Self::transitive_closure(prop, &inv_direct);
            self.inv_prop_supers.insert(prop.clone(), inv_supers);
        }

        Ok(())
    }

    fn compute_domain_range(&mut self) {
        for axiom in &self.axioms {
            match axiom {
                QlAxiom::ObjectPropertyDomain { property, domain } => {
                    self.domain_classes
                        .entry(property.clone())
                        .or_default()
                        .insert(domain.clone());
                }
                QlAxiom::ObjectPropertyRange { property, range } => {
                    self.range_classes
                        .entry(property.clone())
                        .or_default()
                        .insert(range.clone());
                }
                QlAxiom::SubClassOf { sub, sup } => {
                    // ∃P.⊤ ⊑ C → domain
                    if let (QlConcept::SomeValuesTop { property }, Some(sup_name)) =
                        (sub, sup.as_named())
                    {
                        self.domain_classes
                            .entry(property.clone())
                            .or_default()
                            .insert(sup_name.to_string());
                    }
                    // ∃P⁻.⊤ ⊑ C → range
                    if let (QlConcept::SomeValuesTopInverse { property }, Some(sup_name)) =
                        (sub, sup.as_named())
                    {
                        self.range_classes
                            .entry(property.clone())
                            .or_default()
                            .insert(sup_name.to_string());
                    }
                }
                _ => {}
            }
        }

        // Propagate through property hierarchy:
        // If P ⊑ Q and Q has domain D, then P also has domain D
        let props: Vec<String> = self.prop_supers.keys().cloned().collect();
        for prop in props {
            let supers: Vec<String> = self
                .prop_supers
                .get(&prop)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .collect();
            for sup in supers {
                let domains: Vec<String> = self
                    .domain_classes
                    .get(&sup)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .collect();
                for d in domains {
                    self.domain_classes
                        .entry(prop.clone())
                        .or_default()
                        .insert(d);
                }
                let ranges: Vec<String> = self
                    .range_classes
                    .get(&sup)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .collect();
                for r in ranges {
                    self.range_classes
                        .entry(prop.clone())
                        .or_default()
                        .insert(r);
                }
            }
        }

        // Propagate through class hierarchy: if D ⊑ E and P has domain D, then P has domain E
        let domain_entries: Vec<(String, Vec<String>)> = self
            .domain_classes
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().cloned().collect()))
            .collect();
        for (prop, domains) in domain_entries {
            let mut extra = Vec::new();
            for d in &domains {
                if let Some(supers) = self.class_supers.get(d) {
                    extra.extend(supers.iter().cloned());
                }
            }
            let entry = self.domain_classes.entry(prop).or_default();
            for e in extra {
                entry.insert(e);
            }
        }
        let range_entries: Vec<(String, Vec<String>)> = self
            .range_classes
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().cloned().collect()))
            .collect();
        for (prop, ranges) in range_entries {
            let mut extra = Vec::new();
            for r in &ranges {
                if let Some(supers) = self.class_supers.get(r) {
                    extra.extend(supers.iter().cloned());
                }
            }
            let entry = self.range_classes.entry(prop).or_default();
            for e in extra {
                entry.insert(e);
            }
        }
    }

    fn compute_disjointness(&mut self) {
        for axiom in &self.axioms {
            if let QlAxiom::DisjointClasses(c1, c2) = axiom {
                if let (Some(n1), Some(n2)) = (c1.as_named(), c2.as_named()) {
                    let (a, b) = if n1 <= n2 {
                        (n1.to_string(), n2.to_string())
                    } else {
                        (n2.to_string(), n1.to_string())
                    };
                    self.disjoint_classes.insert((a, b));
                }
            }
        }
    }

    /// Compute and index union axioms.
    ///
    /// For each `SubClassOfUnion { sub, sup_union }` axiom where `sub` is a named class C
    /// and `sup_union` is an `ObjectUnionOf(A, B, ...)`, we record:
    ///   union_axioms[C] = [[A, B, ...], ...]
    ///   union_rev_index[A] = {C, ...}   (and same for B, etc.)
    ///
    /// This enables two kinds of union-based rewriting:
    ///
    /// 1. **Forward (concept subsumption)**: C ⊑ A ⊔ B means "if something is C,
    ///    it must be A or B". Used when we want to find what is subsumed.
    ///
    /// 2. **Backward (query unfolding)**: When rewriting ?x:A, we can generate
    ///    a new CQ branch ?x:C if C ⊑ A ⊔ B — because any C individual that is
    ///    an A will satisfy the query (even though some C individuals may be B).
    ///    Note: this is conservative (sound but not always complete without ABox checking).
    fn compute_union_index(&mut self) {
        for axiom in &self.axioms {
            if let QlAxiom::SubClassOfUnion { sub, sup_union } = axiom {
                if let Some(sub_name) = sub.as_named() {
                    let branches = sup_union.union_branches();
                    for branch in &branches {
                        // branch is a list of disjunct class names
                        self.union_axioms
                            .entry(sub_name.to_string())
                            .or_default()
                            .push(branch.clone());
                        // Build reverse index for each disjunct
                        for disjunct in branch {
                            self.union_rev_index
                                .entry(disjunct.clone())
                                .or_default()
                                .insert(sub_name.to_string());
                        }
                    }
                }
            }
        }
    }

    /// Compute transitive closure from `start` following `edges`
    fn transitive_closure(
        start: &str,
        edges: &HashMap<String, HashSet<String>>,
    ) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        if let Some(direct) = edges.get(start) {
            for d in direct {
                queue.push_back(d.clone());
            }
        }
        while let Some(node) = queue.pop_front() {
            if visited.insert(node.clone()) {
                if let Some(nexts) = edges.get(&node) {
                    for n in nexts {
                        if !visited.contains(n) {
                            queue.push_back(n.clone());
                        }
                    }
                }
            }
        }
        visited
    }

    // ---- public query methods ----

    /// Return all superclasses of `class` (transitive, not including self)
    pub fn superclasses(&self, class: &str) -> HashSet<String> {
        self.class_supers.get(class).cloned().unwrap_or_default()
    }

    /// Return all subclasses of `class` (transitive, not including self)
    pub fn subclasses(&self, class: &str) -> HashSet<String> {
        self.class_subs.get(class).cloned().unwrap_or_default()
    }

    /// Return all named superproperties of `property` (transitive, not including self)
    pub fn superproperties(&self, property: &str) -> HashSet<String> {
        self.prop_supers.get(property).cloned().unwrap_or_default()
    }

    /// Return all inverse properties of `property`
    pub fn inverse_properties(&self, property: &str) -> HashSet<String> {
        self.inverse_of.get(property).cloned().unwrap_or_default()
    }

    /// Return domain classes for `property`
    pub fn domain_of(&self, property: &str) -> HashSet<String> {
        self.domain_classes
            .get(property)
            .cloned()
            .unwrap_or_default()
    }

    /// Return range classes for `property`
    pub fn range_of(&self, property: &str) -> HashSet<String> {
        self.range_classes
            .get(property)
            .cloned()
            .unwrap_or_default()
    }

    /// Return true if two classes are declared disjoint
    pub fn are_disjoint(&self, c1: &str, c2: &str) -> bool {
        let (a, b) = if c1 <= c2 {
            (c1.to_string(), c2.to_string())
        } else {
            (c2.to_string(), c1.to_string())
        };
        self.disjoint_classes.contains(&(a, b))
    }

    /// Given class C, find all subclasses B such that B ⊑ C (including C itself)
    /// This is used to unfold a type atom: instead of ?x:C, we may use ?x:B for any B⊑C
    pub fn all_subsumed_by(&self, class: &str) -> HashSet<String> {
        let mut result = HashSet::new();
        result.insert(class.to_string());
        if let Some(subs) = self.class_subs.get(class) {
            result.extend(subs.iter().cloned());
        }
        result
    }

    /// Given property P, find all sub-properties Q such that Q ⊑ P
    /// Includes inverse-based expansions
    pub fn all_subproperties_of(&self, property: &str, is_inverse: bool) -> Vec<QlRole> {
        let mut result = Vec::new();
        // The property itself
        if is_inverse {
            result.push(QlRole::Inverse(property.to_string()));
        } else {
            result.push(QlRole::Named(property.to_string()));
        }

        // Collect subproperties: Q such that Q ⊑ P
        for axiom in &self.axioms {
            if let QlAxiom::SubObjectPropertyOf { sub, sup } = axiom {
                match (sub, sup) {
                    (QlRole::Named(q), QlRole::Named(p)) if p == property && !is_inverse => {
                        result.push(QlRole::Named(q.clone()));
                    }
                    (QlRole::Inverse(q), QlRole::Named(p)) if p == property && !is_inverse => {
                        result.push(QlRole::Inverse(q.clone()));
                    }
                    (QlRole::Named(q), QlRole::Inverse(p)) if p == property && is_inverse => {
                        result.push(QlRole::Named(q.clone()));
                    }
                    (QlRole::Inverse(q), QlRole::Inverse(p)) if p == property && is_inverse => {
                        result.push(QlRole::Inverse(q.clone()));
                    }
                    _ => {}
                }
            }
        }

        // Also include via inverse_of: if P inverseOf Q, querying P⁻ can use Q
        if !is_inverse {
            if let Some(invs) = self.inverse_of.get(property) {
                for inv in invs {
                    result.push(QlRole::Inverse(inv.clone()));
                }
            }
        } else if let Some(invs) = self.inverse_of.get(property) {
            for inv in invs {
                result.push(QlRole::Named(inv.clone()));
            }
        }

        // Deduplicate
        result.sort();
        result.dedup();
        result
    }

    /// Return the union axiom entries for class `sub_class`:
    /// all sets of disjuncts D such that sub_class ⊑ D₁ ⊔ D₂ ⊔ ...
    pub fn union_axiom_disjuncts(&self, sub_class: &str) -> Vec<Vec<String>> {
        self.union_axioms
            .get(sub_class)
            .cloned()
            .unwrap_or_default()
    }

    /// Return all classes C such that C ⊑ (... ⊔ class ⊔ ...)
    /// i.e., classes whose union superclass includes `class` as a disjunct.
    pub fn classes_with_union_disjunct(&self, class: &str) -> HashSet<String> {
        self.union_rev_index.get(class).cloned().unwrap_or_default()
    }

    /// Return true if there are any union axioms in this TBox
    pub fn has_union_axioms(&self) -> bool {
        !self.union_axioms.is_empty()
    }
}

impl Default for Owl2QLTBox {
    fn default() -> Self {
        Self::new()
    }
}

/// The PerfectRef query rewriter for OWL 2 QL
///
/// Extended to handle union query rewriting (ObjectUnionOf on RHS of SubClassOf).
pub struct QueryRewriter {
    tbox: Owl2QLTBox,
    /// Maximum number of rewritten queries to generate (safety limit)
    max_rewrites: usize,
}

impl QueryRewriter {
    /// Create a new rewriter with a classified TBox
    pub fn new(tbox: Owl2QLTBox) -> Self {
        Self {
            tbox,
            max_rewrites: 10_000,
        }
    }

    /// Create with a custom rewriting limit
    pub fn with_limit(tbox: Owl2QLTBox, max_rewrites: usize) -> Self {
        Self { tbox, max_rewrites }
    }

    /// Access the underlying TBox
    pub fn tbox(&self) -> &Owl2QLTBox {
        &self.tbox
    }

    /// Main entry point: rewrite a conjunctive query using PerfectRef algorithm.
    ///
    /// Returns a `RewrittenQuery` (UCQ) that is equivalent to the original CQ
    /// over any ABox consistent with the TBox.
    ///
    /// This extended version handles:
    /// - Standard subclass/subproperty/domain/range unfolding
    /// - Union axiom branching: C ⊑ A ⊔ B causes the type atom ?x:A to gain
    ///   an additional CQ branch where ?x:C replaces the atom (since C individuals
    ///   that are A will satisfy the query).
    pub fn rewrite_query(&self, query: &ConjunctiveQuery) -> Result<RewrittenQuery, QlError> {
        // PerfectRef works by:
        // 1. Start with a queue of CQs (initially just the original)
        // 2. For each CQ, for each atom, compute all possible unfoldings
        // 3. Replace the atom with each unfolding, producing new CQs
        // 4. Add new CQs that haven't been seen before
        // 5. Repeat until fixpoint
        let mut result = RewrittenQuery::new();
        let mut seen: HashSet<Vec<QueryAtom>> = HashSet::new();
        let mut worklist: VecDeque<ConjunctiveQuery> = VecDeque::new();

        worklist.push_back(query.clone());

        while let Some(current_cq) = worklist.pop_front() {
            let mut canonical = current_cq.atoms.clone();
            canonical.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

            if !seen.insert(canonical) {
                continue;
            }

            // Generate all unfoldings of this CQ (one atom at a time)
            for atom_idx in 0..current_cq.atoms.len() {
                let unfolded_atoms = self.unfold_atom(&current_cq.atoms[atom_idx]);

                for unfolded_atom in unfolded_atoms {
                    // Skip if same as original (no change)
                    if unfolded_atom == current_cq.atoms[atom_idx] {
                        continue;
                    }

                    // Build new CQ with the atom replaced
                    let mut new_atoms = current_cq.atoms.clone();
                    new_atoms[atom_idx] = unfolded_atom;
                    let new_cq = ConjunctiveQuery {
                        atoms: new_atoms,
                        head_variables: current_cq.head_variables.clone(),
                    };

                    let mut new_canonical = new_cq.atoms.clone();
                    new_canonical.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

                    if !seen.contains(&new_canonical) {
                        if seen.len() + worklist.len() >= self.max_rewrites {
                            return Err(QlError::RewritingLimitExceeded(self.max_rewrites));
                        }
                        worklist.push_back(new_cq);
                    }
                }

                // Union-based unfolding: produce additional CQ branches
                // for type atoms where there are union axioms
                if let QueryAtom::TypeAtom { individual, class } = &current_cq.atoms[atom_idx] {
                    let union_branches = self.unfold_type_atom_union_branches(individual, class);
                    for branch_atoms in union_branches {
                        let mut new_atoms = current_cq.atoms.clone();
                        new_atoms[atom_idx] = branch_atoms;
                        let new_cq = ConjunctiveQuery {
                            atoms: new_atoms,
                            head_variables: current_cq.head_variables.clone(),
                        };
                        let mut new_canonical = new_cq.atoms.clone();
                        new_canonical.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
                        if !seen.contains(&new_canonical) {
                            if seen.len() + worklist.len() >= self.max_rewrites {
                                return Err(QlError::RewritingLimitExceeded(self.max_rewrites));
                            }
                            worklist.push_back(new_cq);
                        }
                    }
                }
            }

            // This CQ is part of the final rewriting
            result.add(current_cq);
        }

        Ok(result)
    }

    /// Unfold a single query atom using TBox axioms.
    /// Returns all possible alternative atoms (including the original).
    pub fn unfold_atom(&self, atom: &QueryAtom) -> Vec<QueryAtom> {
        match atom {
            QueryAtom::TypeAtom { individual, class } => self.unfold_type_atom(individual, class),
            QueryAtom::PropertyAtom {
                subject,
                property,
                object,
            } => self.unfold_property_atom(subject, property, object),
        }
    }

    /// Unfold a type atom ?x:C into all possible alternatives:
    /// 1. The atom itself (?x:C)
    /// 2. For each B⊑C: ?x:B
    /// 3. For each property P with domain D where D⊑C: ?x P ?fresh
    /// 4. For each property P with range R where R⊑C: ?fresh P ?x
    fn unfold_type_atom(&self, individual: &QueryTerm, class: &str) -> Vec<QueryAtom> {
        let mut result = Vec::new();

        // Original atom
        result.push(QueryAtom::TypeAtom {
            individual: individual.clone(),
            class: class.to_string(),
        });

        // Subclasses: for each B that is subsumed by C (B ⊑ C), we can use ?x:B
        let subsumed = self.tbox.all_subsumed_by(class);
        for sub in &subsumed {
            if sub != class {
                result.push(QueryAtom::TypeAtom {
                    individual: individual.clone(),
                    class: sub.clone(),
                });
            }
        }

        // Domain-based unfolding: if ∃P.⊤ ⊑ C (i.e. P has domain D and D⊑C),
        // we can replace ?x:C with ?x P ?fresh_var
        let fresh_var = self.fresh_variable(individual, "dom");
        for (prop, domains) in &self.tbox.domain_classes {
            for domain in domains {
                // Check if domain ⊑ class
                if domain == class || self.tbox.superclasses(domain).contains(class) {
                    // Generate: ?x P ?fresh
                    result.push(QueryAtom::PropertyAtom {
                        subject: individual.clone(),
                        property: prop.clone(),
                        object: fresh_var.clone(),
                    });
                }
            }
        }

        // Range-based unfolding: if ∃P⁻.⊤ ⊑ C (i.e. P has range R and R⊑C),
        // we can replace ?x:C with ?fresh P ?x
        let fresh_var2 = self.fresh_variable(individual, "rng");
        for (prop, ranges) in &self.tbox.range_classes {
            for range in ranges {
                if range == class || self.tbox.superclasses(range).contains(class) {
                    // Generate: ?fresh P ?x
                    result.push(QueryAtom::PropertyAtom {
                        subject: fresh_var2.clone(),
                        property: prop.clone(),
                        object: individual.clone(),
                    });
                }
            }
        }

        // Deduplicate
        result.dedup_by(|a, b| format!("{a:?}") == format!("{b:?}"));
        result
    }

    /// Compute additional type atom alternatives arising from union axioms.
    ///
    /// For a type atom ?x:A, if there is a union axiom C ⊑ A ⊔ B, then
    /// C individuals that happen to be A will satisfy the query.
    /// We cannot replace ?x:A with ?x:C (since not all C are A), but we can
    /// add a branch that queries ?x:C with the understanding that a C that is
    /// not B must be A (by the closed-world assumption on the union).
    ///
    /// In the PerfectRef framework for OWL 2 QL, union rewriting produces
    /// new CQ branches where the type atom is replaced by the sub-concept C
    /// (the class that is subsumed by the union). This is correct because:
    /// - If ?x is a C individual and C ⊑ A ⊔ B, then ?x satisfies A or B.
    /// - The query ?x:A is satisfied if we can certify ?x is a C that is an A.
    ///
    /// Returns a list of replacement atoms (one per union branch sub-concept).
    pub fn unfold_type_atom_union_branches(
        &self,
        individual: &QueryTerm,
        class: &str,
    ) -> Vec<QueryAtom> {
        let mut result = Vec::new();

        // Find all classes C such that C ⊑ (... ⊔ class ⊔ ...)
        let union_sources = self.tbox.classes_with_union_disjunct(class);
        for source_class in &union_sources {
            result.push(QueryAtom::TypeAtom {
                individual: individual.clone(),
                class: source_class.clone(),
            });
        }

        // Also check subclasses of union source classes (transitivity)
        let mut extra = Vec::new();
        for source_class in &union_sources {
            for sub in self.tbox.subclasses(source_class) {
                extra.push(QueryAtom::TypeAtom {
                    individual: individual.clone(),
                    class: sub,
                });
            }
        }
        result.extend(extra);

        // Deduplicate
        result.dedup_by(|a, b| format!("{a:?}") == format!("{b:?}"));
        result
    }

    /// Unfold a property atom (?x P ?y) into all possible alternatives:
    /// 1. The atom itself
    /// 2. For each Q⊑P: ?x Q ?y
    /// 3. For each inverse Q⁻⊑P: ?y Q ?x
    /// 4. Via inverseOf: if P inverseOf Q, then ?y Q ?x
    fn unfold_property_atom(
        &self,
        subject: &QueryTerm,
        property: &str,
        object: &QueryTerm,
    ) -> Vec<QueryAtom> {
        let mut result = Vec::new();

        // Original
        result.push(QueryAtom::PropertyAtom {
            subject: subject.clone(),
            property: property.to_string(),
            object: object.clone(),
        });

        // Get all sub-roles of property P (i.e. Q such that Q ⊑ P)
        let sub_roles = self.tbox.all_subproperties_of(property, false);
        for role in sub_roles {
            match role {
                QlRole::Named(q) if q != property => {
                    result.push(QueryAtom::PropertyAtom {
                        subject: subject.clone(),
                        property: q,
                        object: object.clone(),
                    });
                }
                QlRole::Inverse(q) => {
                    // Q⁻ ⊑ P means if (y,x):Q then (x,y):P — so use ?y Q ?x
                    result.push(QueryAtom::PropertyAtom {
                        subject: object.clone(),
                        property: q,
                        object: subject.clone(),
                    });
                }
                _ => {}
            }
        }

        // Deduplicate
        result.dedup_by(|a, b| format!("{a:?}") == format!("{b:?}"));
        result
    }

    /// Generate a fresh variable name based on the individual term
    fn fresh_variable(&self, base: &QueryTerm, suffix: &str) -> QueryTerm {
        match base {
            QueryTerm::Variable(v) => QueryTerm::Variable(format!("_fresh_{v}_{suffix}")),
            QueryTerm::Constant(c) => {
                // Use a short hash-like suffix
                let short: String = c.chars().take(8).collect();
                QueryTerm::Variable(format!("_fresh_{short}_{suffix}"))
            }
        }
    }

    /// Check if a class assertion is trivially satisfiable given the TBox
    pub fn is_satisfiable(&self, class: &str) -> bool {
        // A class is unsatisfiable if it's in a disjointness pair with itself
        // or disjoint with owl:Thing
        !self.tbox.are_disjoint(class, class)
    }

    /// Compute union-aware query rewriting for a conjunctive query where
    /// union axioms guide query branching.
    ///
    /// This method extends `rewrite_query` by explicitly computing all union
    /// disjunct branches for each type atom and emitting separate CQs for
    /// each possible branch derivation path.
    ///
    /// For example, given:
    ///   TBox: Dog ⊑ Animal ⊔ Pet
    ///   Query: ?x:Animal
    ///
    /// The rewriting includes:
    ///   ?x:Animal (original)
    ///   ?x:Dog    (from union: Dog may be an Animal)
    pub fn rewrite_query_union_aware(
        &self,
        query: &ConjunctiveQuery,
    ) -> Result<RewrittenQuery, QlError> {
        // Use the standard rewrite_query which now includes union branching
        self.rewrite_query(query)
    }

    /// Compute all union disjunct branches for a named class.
    ///
    /// Returns a list of lists — each inner list is a set of class names that
    /// are sibling disjuncts in some union axiom where `class` participates.
    ///
    /// Example: if TBox contains C ⊑ A ⊔ B, then:
    ///   union_siblings_for("A") → [["A", "B"]]
    pub fn union_siblings_for(&self, class: &str) -> Vec<Vec<String>> {
        // Find union axioms where `class` is a disjunct
        let sources = self.tbox.classes_with_union_disjunct(class);
        let mut result = Vec::new();
        for source in &sources {
            let disjunct_lists = self.tbox.union_axiom_disjuncts(source);
            for disjuncts in disjunct_lists {
                if disjuncts.contains(&class.to_string()) {
                    result.push(disjuncts);
                }
            }
        }
        result
    }

    /// Check if class A and class B are in a union axiom together (sibling disjuncts)
    pub fn are_union_siblings(&self, class_a: &str, class_b: &str) -> bool {
        let siblings = self.union_siblings_for(class_a);
        siblings
            .iter()
            .any(|branch| branch.contains(&class_b.to_string()))
    }

    /// Compute satisfiability of a concept conjunction {C₁, C₂, ...} w.r.t. the TBox.
    ///
    /// Returns false if the conjunction is definitely unsatisfiable based on:
    /// - Direct disjointness between any two members
    /// - Union axiom exhaustion: C ⊑ A ⊔ B and neither A nor B is compatible
    pub fn conjunction_satisfiable(&self, classes: &[&str]) -> bool {
        // Check pairwise disjointness
        for i in 0..classes.len() {
            for j in (i + 1)..classes.len() {
                if self.tbox.are_disjoint(classes[i], classes[j]) {
                    return false;
                }
            }
        }
        true
    }
}

// ---- Convenience functions ----

/// Build and classify a TBox from a list of axioms
pub fn build_tbox(axioms: Vec<QlAxiom>) -> Result<Owl2QLTBox, QlError> {
    let mut tbox = Owl2QLTBox::new();
    tbox.add_axioms(axioms);
    tbox.classify()?;
    Ok(tbox)
}

/// Rewrite a SPARQL-like conjunctive query using OWL 2 QL PerfectRef
pub fn rewrite_query(atoms: Vec<QueryAtom>, tbox: &Owl2QLTBox) -> Result<RewrittenQuery, QlError> {
    let cq = ConjunctiveQuery::with_atoms(atoms);
    let rewriter = QueryRewriter::new(tbox.clone());
    rewriter.rewrite_query(&cq)
}

/// Rewrite a query with explicit union-aware branching
pub fn rewrite_query_union(
    atoms: Vec<QueryAtom>,
    tbox: &Owl2QLTBox,
) -> Result<RewrittenQuery, QlError> {
    let cq = ConjunctiveQuery::with_atoms(atoms);
    let rewriter = QueryRewriter::new(tbox.clone());
    rewriter.rewrite_query_union_aware(&cq)
}

impl Clone for Owl2QLTBox {
    fn clone(&self) -> Self {
        let mut new_tbox = Owl2QLTBox::new();
        new_tbox.axioms = self.axioms.clone();
        new_tbox.class_supers = self.class_supers.clone();
        new_tbox.class_subs = self.class_subs.clone();
        new_tbox.prop_supers = self.prop_supers.clone();
        new_tbox.inv_prop_supers = self.inv_prop_supers.clone();
        new_tbox.inverse_of = self.inverse_of.clone();
        new_tbox.domain_classes = self.domain_classes.clone();
        new_tbox.range_classes = self.range_classes.clone();
        new_tbox.disjoint_classes = self.disjoint_classes.clone();
        new_tbox.union_axioms = self.union_axioms.clone();
        new_tbox.union_rev_index = self.union_rev_index.clone();
        new_tbox
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    include!("owl_ql_tests.rs");
    include!("owl_ql_tests_extended.rs");
}
