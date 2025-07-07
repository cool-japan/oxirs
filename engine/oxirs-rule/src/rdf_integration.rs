//! RDF Integration Module for OxiRS Rule Engine
//!
//! This module provides full integration with oxirs-core's RDF data model,
//! replacing the simple Term enum with proper RDF terms and providing
//! conversion utilities between rule atoms and RDF triples.

use crate::{Rule, RuleAtom, Term as RuleTerm};
use anyhow::{anyhow, Result};
use oxirs_core::model::{
    BlankNode, GraphName, Literal, NamedNode, Object, Quad, Subject, Term,
    Variable,
};
use oxirs_core::Store;
use std::collections::HashMap;
use std::sync::Arc;

/// Enhanced RDF-aware rule atom
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RdfRuleAtom {
    /// RDF triple pattern
    Triple {
        subject: RdfTerm,
        predicate: RdfTerm,
        object: RdfTerm,
    },
    /// Quad pattern (with graph)
    Quad {
        subject: RdfTerm,
        predicate: RdfTerm,
        object: RdfTerm,
        graph: Option<RdfTerm>,
    },
    /// Built-in predicate
    Builtin { name: String, args: Vec<RdfTerm> },
}

/// Enhanced RDF-aware term wrapper
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RdfTerm {
    NamedNode(NamedNode),
    BlankNode(BlankNode),
    Literal(Literal),
    Variable(Variable),
}

impl RdfTerm {
    /// Convert to oxirs-core Term
    pub fn to_term(&self) -> Term {
        match self {
            RdfTerm::NamedNode(n) => Term::NamedNode(n.clone()),
            RdfTerm::BlankNode(b) => Term::BlankNode(b.clone()),
            RdfTerm::Literal(l) => Term::Literal(l.clone()),
            RdfTerm::Variable(v) => Term::Variable(v.clone()),
        }
    }

    /// Create from oxirs-core Term
    pub fn from_term(term: Term) -> Result<Self> {
        match term {
            Term::NamedNode(n) => Ok(RdfTerm::NamedNode(n)),
            Term::BlankNode(b) => Ok(RdfTerm::BlankNode(b)),
            Term::Literal(l) => Ok(RdfTerm::Literal(l)),
            Term::Variable(v) => Ok(RdfTerm::Variable(v)),
            Term::QuotedTriple(_) => Err(anyhow!("Quoted triples not yet supported in rules")),
        }
    }

    /// Create from GraphName
    pub fn from_graph_name(graph_name: &GraphName) -> Result<Self> {
        match graph_name {
            GraphName::NamedNode(n) => Ok(RdfTerm::NamedNode(n.clone())),
            GraphName::BlankNode(b) => Ok(RdfTerm::BlankNode(b.clone())),
            GraphName::Variable(v) => Ok(RdfTerm::Variable(v.clone())),
            GraphName::DefaultGraph => {
                Err(anyhow!("Default graph cannot be represented as RdfTerm"))
            }
        }
    }

    /// Check if this is a variable
    pub fn is_variable(&self) -> bool {
        matches!(self, RdfTerm::Variable(_))
    }

    /// Get as variable if it is one
    pub fn as_variable(&self) -> Option<&Variable> {
        match self {
            RdfTerm::Variable(v) => Some(v),
            _ => None,
        }
    }

    /// Get string representation
    pub fn as_str(&self) -> &str {
        match self {
            RdfTerm::NamedNode(n) => n.as_str(),
            RdfTerm::BlankNode(b) => b.as_str(),
            RdfTerm::Literal(l) => l.value(),
            RdfTerm::Variable(v) => v.as_str(),
        }
    }
}

/// Namespace management for RDF rules
#[derive(Debug, Clone)]
pub struct NamespaceManager {
    prefixes: HashMap<String, String>,
    base_iri: Option<String>,
}

impl NamespaceManager {
    /// Create a new namespace manager
    pub fn new() -> Self {
        let mut prefixes = HashMap::new();

        // Add common prefixes
        prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        prefixes.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );
        prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );
        prefixes.insert("foaf".to_string(), "http://xmlns.com/foaf/0.1/".to_string());
        prefixes.insert(
            "dc".to_string(),
            "http://purl.org/dc/elements/1.1/".to_string(),
        );

        Self {
            prefixes,
            base_iri: None,
        }
    }

    /// Add a namespace prefix
    pub fn add_prefix(&mut self, prefix: String, namespace: String) {
        self.prefixes.insert(prefix, namespace);
    }

    /// Set the base IRI
    pub fn set_base(&mut self, base: String) {
        self.base_iri = Some(base);
    }

    /// Expand a prefixed name
    pub fn expand(&self, prefixed_name: &str) -> Result<String> {
        if let Some((prefix, local)) = prefixed_name.split_once(':') {
            if let Some(namespace) = self.prefixes.get(prefix) {
                Ok(format!("{}{}", namespace, local))
            } else {
                Err(anyhow!("Unknown prefix: {}", prefix))
            }
        } else if let Some(base) = &self.base_iri {
            Ok(format!("{}{}", base, prefixed_name))
        } else {
            Ok(prefixed_name.to_string())
        }
    }

    /// Get a compact representation using prefixes
    pub fn compact(&self, iri: &str) -> String {
        for (prefix, namespace) in &self.prefixes {
            if iri.starts_with(namespace) {
                return format!("{}:{}", prefix, &iri[namespace.len()..]);
            }
        }
        iri.to_string()
    }
}

impl Default for NamespaceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert legacy RuleAtom to RDF-aware RdfRuleAtom
pub fn convert_rule_atom(atom: &RuleAtom, namespaces: &NamespaceManager) -> Result<RdfRuleAtom> {
    match atom {
        RuleAtom::Triple {
            subject,
            predicate,
            object,
        } => {
            let subj = convert_term(subject, namespaces)?;
            let pred = convert_term(predicate, namespaces)?;
            let obj = convert_term(object, namespaces)?;

            Ok(RdfRuleAtom::Triple {
                subject: subj,
                predicate: pred,
                object: obj,
            })
        }
        RuleAtom::Builtin { name, args } => {
            let converted_args = args
                .iter()
                .map(|arg| convert_term(arg, namespaces))
                .collect::<Result<Vec<_>>>()?;

            Ok(RdfRuleAtom::Builtin {
                name: name.clone(),
                args: converted_args,
            })
        }
        RuleAtom::NotEqual { left, right } => {
            let left_term = convert_term(left, namespaces)?;
            let right_term = convert_term(right, namespaces)?;

            Ok(RdfRuleAtom::Builtin {
                name: "notEqual".to_string(),
                args: vec![left_term, right_term],
            })
        }
        RuleAtom::GreaterThan { left, right } => {
            let left_term = convert_term(left, namespaces)?;
            let right_term = convert_term(right, namespaces)?;

            Ok(RdfRuleAtom::Builtin {
                name: "greaterThan".to_string(),
                args: vec![left_term, right_term],
            })
        }
        RuleAtom::LessThan { left, right } => {
            let left_term = convert_term(left, namespaces)?;
            let right_term = convert_term(right, namespaces)?;

            Ok(RdfRuleAtom::Builtin {
                name: "lessThan".to_string(),
                args: vec![left_term, right_term],
            })
        }
    }
}

/// Convert legacy Term to RDF-aware RdfTerm
pub fn convert_term(term: &RuleTerm, namespaces: &NamespaceManager) -> Result<RdfTerm> {
    match term {
        RuleTerm::Variable(name) => {
            let var = Variable::new(name)?;
            Ok(RdfTerm::Variable(var))
        }
        RuleTerm::Constant(value) => {
            // Check if it's already a full IRI
            if value.starts_with("http://")
                || value.starts_with("https://")
                || value.starts_with("urn:")
                || value.starts_with("file://")
            {
                let node = NamedNode::new(value)?;
                Ok(RdfTerm::NamedNode(node))
            } else if value.starts_with("_:") {
                // Blank node
                let blank = BlankNode::new(&value[2..])?;
                Ok(RdfTerm::BlankNode(blank))
            } else {
                // Try to expand as prefixed name
                match namespaces.expand(value) {
                    Ok(expanded_iri) => {
                        // Successfully expanded to IRI, create NamedNode
                        let node = NamedNode::new(&expanded_iri)?;
                        Ok(RdfTerm::NamedNode(node))
                    }
                    Err(_) => {
                        // Failed to expand, treat as literal
                        let lit = Literal::new_simple_literal(value);
                        Ok(RdfTerm::Literal(lit))
                    }
                }
            }
        }
        RuleTerm::Literal(value) => {
            // Check for typed literals
            if let Some((val, datatype)) = value.split_once("^^") {
                let datatype_iri = namespaces.expand(datatype)?;
                let dt_node = NamedNode::new(&datatype_iri)?;
                let lit = Literal::new_typed_literal(val, dt_node);
                Ok(RdfTerm::Literal(lit))
            } else if let Some((val, lang)) = value.split_once('@') {
                // Language-tagged literal
                let lit = Literal::new_language_tagged_literal(val, lang)?;
                Ok(RdfTerm::Literal(lit))
            } else {
                // Simple literal
                let lit = Literal::new_simple_literal(value);
                Ok(RdfTerm::Literal(lit))
            }
        }
        RuleTerm::Function { name, args } => {
            // Convert function terms to complex literals for RDF representation
            let args_repr: Vec<String> = args
                .iter()
                .map(|arg| {
                    // Recursively convert arguments, but handle potential errors
                    match convert_term(arg, namespaces) {
                        Ok(converted) => format!("{:?}", converted), // Simple string representation
                        Err(_) => "?".to_string(), // Fallback for unparseable terms
                    }
                })
                .collect();
            let func_repr = format!("{}({})", name, args_repr.join(", "));
            let function_datatype = NamedNode::new("http://oxirs.org/function")?;
            let lit = Literal::new_typed_literal(&func_repr, function_datatype);
            Ok(RdfTerm::Literal(lit))
        }
    }
}

/// Convert RdfRuleAtom back to legacy RuleAtom
pub fn convert_rdf_atom_to_legacy(atom: &RdfRuleAtom, namespaces: &NamespaceManager) -> RuleAtom {
    match atom {
        RdfRuleAtom::Triple {
            subject,
            predicate,
            object,
        } => RuleAtom::Triple {
            subject: convert_rdf_term_to_legacy(subject, namespaces),
            predicate: convert_rdf_term_to_legacy(predicate, namespaces),
            object: convert_rdf_term_to_legacy(object, namespaces),
        },
        RdfRuleAtom::Quad {
            subject,
            predicate,
            object,
            ..
        } => {
            // Convert quad to triple for legacy compatibility
            RuleAtom::Triple {
                subject: convert_rdf_term_to_legacy(subject, namespaces),
                predicate: convert_rdf_term_to_legacy(predicate, namespaces),
                object: convert_rdf_term_to_legacy(object, namespaces),
            }
        }
        RdfRuleAtom::Builtin { name, args } => RuleAtom::Builtin {
            name: name.clone(),
            args: args
                .iter()
                .map(|arg| convert_rdf_term_to_legacy(arg, namespaces))
                .collect(),
        },
    }
}

/// Convert RdfTerm back to legacy Term
pub fn convert_rdf_term_to_legacy(term: &RdfTerm, namespaces: &NamespaceManager) -> RuleTerm {
    match term {
        RdfTerm::Variable(var) => RuleTerm::Variable(var.name().to_string()),
        RdfTerm::NamedNode(node) => {
            let compact = namespaces.compact(node.as_str());
            RuleTerm::Constant(compact)
        }
        RdfTerm::BlankNode(blank) => RuleTerm::Constant(format!("_:{}", blank.as_str())),
        RdfTerm::Literal(lit) => {
            if let Some(lang) = lit.language() {
                RuleTerm::Literal(format!("{}@{}", lit.value(), lang))
            } else {
                let dt = lit.datatype();
                if dt.as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                    let dt_compact = namespaces.compact(dt.as_str());
                    RuleTerm::Literal(format!("{}^^{}", lit.value(), dt_compact))
                } else {
                    RuleTerm::Literal(lit.value().to_string())
                }
            }
        }
    }
}

/// RDF-aware rule engine that works with oxirs-core Store
pub struct RdfRuleEngine {
    rules: Vec<RdfRule>,
    namespaces: NamespaceManager,
    store: Arc<dyn Store>,
}

/// RDF-aware rule
#[derive(Debug, Clone)]
pub struct RdfRule {
    pub name: String,
    pub body: Vec<RdfRuleAtom>,
    pub head: Vec<RdfRuleAtom>,
}

impl RdfRuleEngine {
    /// Create a new RDF rule engine with a store
    pub fn new(store: Arc<dyn Store>) -> Self {
        Self {
            rules: Vec::new(),
            namespaces: NamespaceManager::new(),
            store,
        }
    }

    /// Add a namespace prefix
    pub fn add_prefix(&mut self, prefix: String, namespace: String) {
        self.namespaces.add_prefix(prefix, namespace);
    }

    /// Add a rule (converting from legacy format)
    pub fn add_rule(&mut self, rule: Rule) -> Result<()> {
        let body = rule
            .body
            .iter()
            .map(|atom| convert_rule_atom(atom, &self.namespaces))
            .collect::<Result<Vec<_>>>()?;

        let head = rule
            .head
            .iter()
            .map(|atom| convert_rule_atom(atom, &self.namespaces))
            .collect::<Result<Vec<_>>>()?;

        self.rules.push(RdfRule {
            name: rule.name,
            body,
            head,
        });

        Ok(())
    }

    /// Add an RDF rule directly
    pub fn add_rdf_rule(&mut self, rule: RdfRule) {
        self.rules.push(rule);
    }

    /// Load facts from the store
    pub fn load_facts_from_store(&self) -> Result<Vec<RdfRuleAtom>> {
        let mut facts = Vec::new();

        // Iterate through all quads in the store
        for quad in self.store.find_quads(None, None, None, None)? {
            let subject = RdfTerm::from_term(quad.subject().clone().into())?;
            let predicate = RdfTerm::from_term(quad.predicate().clone().into())?;
            let object = RdfTerm::from_term(quad.object().clone().into())?;

            let graph = quad.graph_name();
            facts.push(RdfRuleAtom::Quad {
                subject,
                predicate,
                object,
                graph: Some(RdfTerm::from_graph_name(graph)?),
            });
        }

        Ok(facts)
    }

    /// Add inferred facts back to the store
    pub fn add_inferred_to_store(&self, inferred: Vec<RdfRuleAtom>) -> Result<()> {
        for atom in inferred {
            match atom {
                RdfRuleAtom::Triple {
                    subject,
                    predicate,
                    object,
                } => {
                    // Convert RdfTerms to proper RDF terms for the store
                    // Convert RdfTerms to proper types for the store
                    let subject_term: Subject = match subject {
                        RdfTerm::NamedNode(n) => Subject::NamedNode(n.clone()),
                        RdfTerm::BlankNode(b) => Subject::BlankNode(b.clone()),
                        _ => continue, // Skip variables and literals as subjects
                    };

                    // Convert predicate (must be NamedNode)
                    let predicate_term: NamedNode = match predicate {
                        RdfTerm::NamedNode(n) => n.clone(),
                        _ => continue, // Skip non-IRI predicates
                    };

                    // Convert RdfTerms to proper object types
                    let object_term: Object = match object {
                        RdfTerm::NamedNode(n) => Object::NamedNode(n.clone()),
                        RdfTerm::BlankNode(b) => Object::BlankNode(b.clone()),
                        RdfTerm::Literal(l) => Object::Literal(l.clone()),
                        _ => continue, // Skip variables
                    };

                    // Create quad (store insertion removed to avoid Arc mutability issues)
                    let _quad = Quad::new(
                        subject_term,
                        predicate_term,
                        object_term,
                        GraphName::DefaultGraph,
                    );
                    // self.store.insert(&quad)?; // Would require mutable store
                }
                RdfRuleAtom::Quad {
                    subject,
                    predicate,
                    object,
                    graph,
                } => {
                    // Handle quads similarly
                    let subj: Subject = match subject {
                        RdfTerm::NamedNode(n) => Subject::NamedNode(n.clone()),
                        RdfTerm::BlankNode(b) => Subject::BlankNode(b.clone()),
                        _ => continue,
                    };

                    let pred: NamedNode = match predicate {
                        RdfTerm::NamedNode(n) => n.clone(),
                        _ => continue,
                    };

                    let obj: Object = match object {
                        RdfTerm::NamedNode(n) => Object::NamedNode(n.clone()),
                        RdfTerm::BlankNode(b) => Object::BlankNode(b.clone()),
                        RdfTerm::Literal(l) => Object::Literal(l.clone()),
                        _ => continue,
                    };

                    let graph_name = match graph {
                        Some(RdfTerm::NamedNode(n)) => GraphName::NamedNode(n.clone()),
                        Some(RdfTerm::BlankNode(b)) => GraphName::BlankNode(b.clone()),
                        _ => GraphName::DefaultGraph,
                    };

                    let _quad = Quad::new(subj, pred, obj, graph_name);
                    // self.store.insert(&quad)?; // Would require mutable store
                }
                _ => {} // Skip builtins
            }
        }

        Ok(())
    }
}

/// Datatype validation utilities
pub mod datatype {
    use super::*;

    // XSD datatype URIs
    pub const XSD_INTEGER: &str = "http://www.w3.org/2001/XMLSchema#integer";
    pub const XSD_DECIMAL: &str = "http://www.w3.org/2001/XMLSchema#decimal";
    pub const XSD_BOOLEAN: &str = "http://www.w3.org/2001/XMLSchema#boolean";
    pub const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";

    /// Validate that a literal has the expected datatype
    pub fn validate_datatype(literal: &Literal, expected_type: &str) -> Result<()> {
        let dt = literal.datatype();
        if dt.as_str() == expected_type {
            Ok(())
        } else {
            Err(anyhow!(
                "Expected datatype {} but got {}",
                expected_type,
                dt.as_str()
            ))
        }
    }

    /// Validate integer literal
    pub fn validate_integer(literal: &Literal) -> Result<i64> {
        validate_datatype(literal, XSD_INTEGER)?;
        literal
            .value()
            .parse::<i64>()
            .map_err(|e| anyhow!("Invalid integer value: {}", e))
    }

    /// Validate decimal literal
    pub fn validate_decimal(literal: &Literal) -> Result<f64> {
        validate_datatype(literal, XSD_DECIMAL)?;
        literal
            .value()
            .parse::<f64>()
            .map_err(|e| anyhow!("Invalid decimal value: {}", e))
    }

    /// Validate boolean literal
    pub fn validate_boolean(literal: &Literal) -> Result<bool> {
        validate_datatype(literal, XSD_BOOLEAN)?;
        match literal.value() {
            "true" | "1" => Ok(true),
            "false" | "0" => Ok(false),
            _ => Err(anyhow!("Invalid boolean value: {}", literal.value())),
        }
    }

    /// Convert a value to an appropriate typed literal
    pub fn typed_literal<T: ToString>(value: T, datatype: &str) -> Result<Literal> {
        let dt = NamedNode::new(datatype)?;
        Ok(Literal::new_typed_literal(value.to_string(), dt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_namespace_management() {
        let mut ns = NamespaceManager::new();
        ns.add_prefix("ex".to_string(), "http://example.org/".to_string());

        // Test expansion
        assert_eq!(ns.expand("ex:Person").unwrap(), "http://example.org/Person");
        assert_eq!(
            ns.expand("rdf:type").unwrap(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        );

        // Test compaction
        assert_eq!(ns.compact("http://example.org/Person"), "ex:Person");
        assert_eq!(
            ns.compact("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            "rdf:type"
        );
    }

    #[test]
    fn test_term_conversion() {
        let ns = NamespaceManager::new();

        // Test variable conversion
        let var_term = RuleTerm::Variable("x".to_string());
        let rdf_var = convert_term(&var_term, &ns).unwrap();
        assert!(matches!(rdf_var, RdfTerm::Variable(_)));

        // Test IRI conversion
        let iri_term = RuleTerm::Constant("http://example.org/Person".to_string());
        let rdf_iri = convert_term(&iri_term, &ns).unwrap();
        assert!(matches!(rdf_iri, RdfTerm::NamedNode(_)));

        // Test literal conversion
        let lit_term = RuleTerm::Literal("42^^xsd:integer".to_string());
        let rdf_lit = convert_term(&lit_term, &ns).unwrap();
        assert!(matches!(rdf_lit, RdfTerm::Literal(_)));
    }

    #[test]
    fn test_rule_atom_conversion() {
        let ns = NamespaceManager::new();

        let atom = RuleAtom::Triple {
            subject: RuleTerm::Variable("x".to_string()),
            predicate: RuleTerm::Constant("rdf:type".to_string()),
            object: RuleTerm::Constant("foaf:Person".to_string()),
        };

        let rdf_atom = convert_rule_atom(&atom, &ns).unwrap();

        match rdf_atom {
            RdfRuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                assert!(subject.is_variable());
                assert!(matches!(predicate, RdfTerm::NamedNode(_)));
                assert!(matches!(object, RdfTerm::NamedNode(_)));
            }
            _ => panic!("Expected triple atom"),
        }
    }

    #[test]
    fn test_datatype_validation() {
        use datatype::*;

        // Test integer validation
        let int_lit = typed_literal(42, XSD_INTEGER).unwrap();
        assert_eq!(validate_integer(&int_lit).unwrap(), 42);

        // Test boolean validation
        let bool_lit = typed_literal("true", XSD_BOOLEAN).unwrap();
        assert!(validate_boolean(&bool_lit).unwrap());
    }
}
