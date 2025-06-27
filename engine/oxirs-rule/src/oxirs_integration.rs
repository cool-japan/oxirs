//! Integration module for bridging oxirs-rule with oxirs-core RDF types
//!
//! This module provides conversions between the rule engine's internal representations
//! and the standard RDF types from oxirs-core.

use crate::{Term as RuleTerm, RuleAtom};
use oxirs_core::model::{
    Term as RdfTerm, Subject, Predicate, Object, Triple,
    NamedNode, BlankNode, Literal, Variable,
};
use anyhow::{anyhow, Result};
use std::convert::TryFrom;

/// Convert a rule engine Term to an RDF Term
pub fn rule_term_to_rdf_term(rule_term: &RuleTerm) -> Result<RdfTerm> {
    match rule_term {
        RuleTerm::Variable(var) => {
            Ok(RdfTerm::Variable(Variable::new(var)?))
        }
        RuleTerm::Constant(iri) => {
            // Try to parse as IRI first
            if iri.starts_with("http://") || iri.starts_with("https://") || iri.contains(':') {
                Ok(RdfTerm::NamedNode(NamedNode::new(iri)?))
            } else if iri.starts_with("_:") {
                // Blank node
                Ok(RdfTerm::BlankNode(BlankNode::new(&iri[2..])?))
            } else {
                // Treat as string literal
                Ok(RdfTerm::Literal(Literal::new_simple_literal(iri)))
            }
        }
        RuleTerm::Literal(value) => {
            // Check if it's a typed literal (format: "value"^^<datatype>)
            if let Some(type_sep) = value.find("^^") {
                let literal_value = &value[..type_sep].trim_matches('"');
                let datatype_iri = &value[type_sep + 2..].trim_matches('<').trim_matches('>');
                Ok(RdfTerm::Literal(Literal::new_typed_literal(
                    literal_value,
                    NamedNode::new(datatype_iri)?
                )))
            } else if let Some(lang_sep) = value.find('@') {
                // Language-tagged literal
                let literal_value = &value[..lang_sep].trim_matches('"');
                let lang_tag = &value[lang_sep + 1..];
                Ok(RdfTerm::Literal(Literal::new_language_tagged_literal(
                    literal_value,
                    lang_tag,
                )?))
            } else {
                // Simple literal
                Ok(RdfTerm::Literal(Literal::new_simple_literal(value.trim_matches('"'))))
            }
        }
    }
}

/// Convert an RDF Term to a rule engine Term
pub fn rdf_term_to_rule_term(rdf_term: &RdfTerm) -> RuleTerm {
    match rdf_term {
        RdfTerm::Variable(var) => RuleTerm::Variable(var.name().to_string()),
        RdfTerm::NamedNode(node) => RuleTerm::Constant(node.as_str().to_string()),
        RdfTerm::BlankNode(node) => RuleTerm::Constant(format!("_:{}", node.id())),
        RdfTerm::Literal(lit) => {
            // Convert literal to rule engine format
            match lit.datatype() {
                Some(dt) if dt.as_str() != "http://www.w3.org/2001/XMLSchema#string" => {
                    RuleTerm::Literal(format!("\"{}\"^^<{}>", lit.value(), dt.as_str()))
                }
                _ => match lit.language() {
                    Some(lang) => RuleTerm::Literal(format!("\"{}\"@{}", lit.value(), lang)),
                    None => RuleTerm::Literal(format!("\"{}\"", lit.value())),
                }
            }
        }
        RdfTerm::QuotedTriple(_) => {
            // For now, we don't support quoted triples in rules
            RuleTerm::Constant("<<quoted-triple>>".to_string())
        }
    }
}

/// Convert a RuleAtom to an RDF Triple
pub fn rule_atom_to_triple(atom: &RuleAtom) -> Result<Triple> {
    match atom {
        RuleAtom::Triple { subject, predicate, object } => {
            let rdf_subject = rule_term_to_subject(subject)?;
            let rdf_predicate = rule_term_to_predicate(predicate)?;
            let rdf_object = rule_term_to_object(object)?;
            
            Ok(Triple::new(rdf_subject, rdf_predicate, rdf_object))
        }
        RuleAtom::Builtin { .. } => {
            Err(anyhow!("Cannot convert builtin atom to RDF triple"))
        }
    }
}

/// Convert a rule Term to an RDF Subject
fn rule_term_to_subject(term: &RuleTerm) -> Result<Subject> {
    let rdf_term = rule_term_to_rdf_term(term)?;
    match rdf_term {
        RdfTerm::NamedNode(n) => Ok(Subject::NamedNode(n)),
        RdfTerm::BlankNode(b) => Ok(Subject::BlankNode(b)),
        RdfTerm::Variable(v) => Ok(Subject::Variable(v)),
        RdfTerm::QuotedTriple(qt) => Ok(Subject::QuotedTriple(qt)),
        RdfTerm::Literal(_) => Err(anyhow!("Literals cannot be used as subjects")),
    }
}

/// Convert a rule Term to an RDF Predicate
fn rule_term_to_predicate(term: &RuleTerm) -> Result<Predicate> {
    let rdf_term = rule_term_to_rdf_term(term)?;
    match rdf_term {
        RdfTerm::NamedNode(n) => Ok(Predicate::NamedNode(n)),
        RdfTerm::Variable(v) => Ok(Predicate::Variable(v)),
        _ => Err(anyhow!("Only named nodes and variables can be used as predicates")),
    }
}

/// Convert a rule Term to an RDF Object
fn rule_term_to_object(term: &RuleTerm) -> Result<Object> {
    let rdf_term = rule_term_to_rdf_term(term)?;
    match rdf_term {
        RdfTerm::NamedNode(n) => Ok(Object::NamedNode(n)),
        RdfTerm::BlankNode(b) => Ok(Object::BlankNode(b)),
        RdfTerm::Literal(l) => Ok(Object::Literal(l)),
        RdfTerm::Variable(v) => Ok(Object::Variable(v)),
        RdfTerm::QuotedTriple(qt) => Ok(Object::QuotedTriple(qt)),
    }
}

/// Convert an RDF Triple to a RuleAtom
pub fn triple_to_rule_atom(triple: &Triple) -> RuleAtom {
    let subject = subject_to_rule_term(triple.subject());
    let predicate = predicate_to_rule_term(triple.predicate());
    let object = object_to_rule_term(triple.object());
    
    RuleAtom::Triple { subject, predicate, object }
}

/// Convert an RDF Subject to a rule Term
fn subject_to_rule_term(subject: &Subject) -> RuleTerm {
    match subject {
        Subject::NamedNode(n) => RuleTerm::Constant(n.as_str().to_string()),
        Subject::BlankNode(b) => RuleTerm::Constant(format!("_:{}", b.id())),
        Subject::Variable(v) => RuleTerm::Variable(v.name().to_string()),
        Subject::QuotedTriple(_) => RuleTerm::Constant("<<quoted-triple>>".to_string()),
    }
}

/// Convert an RDF Predicate to a rule Term
fn predicate_to_rule_term(predicate: &Predicate) -> RuleTerm {
    match predicate {
        Predicate::NamedNode(n) => RuleTerm::Constant(n.as_str().to_string()),
        Predicate::Variable(v) => RuleTerm::Variable(v.name().to_string()),
    }
}

/// Convert an RDF Object to a rule Term
fn object_to_rule_term(object: &Object) -> RuleTerm {
    match object {
        Object::NamedNode(n) => RuleTerm::Constant(n.as_str().to_string()),
        Object::BlankNode(b) => RuleTerm::Constant(format!("_:{}", b.id())),
        Object::Literal(l) => {
            match l.datatype() {
                Some(dt) if dt.as_str() != "http://www.w3.org/2001/XMLSchema#string" => {
                    RuleTerm::Literal(format!("\"{}\"^^<{}>", l.value(), dt.as_str()))
                }
                _ => match l.language() {
                    Some(lang) => RuleTerm::Literal(format!("\"{}\"@{}", l.value(), lang)),
                    None => RuleTerm::Literal(l.value().to_string()),
                }
            }
        }
        Object::Variable(v) => RuleTerm::Variable(v.name().to_string()),
        Object::QuotedTriple(_) => RuleTerm::Constant("<<quoted-triple>>".to_string()),
    }
}

/// Load RDF data and convert to rule atoms
pub fn load_rdf_data(triples: Vec<Triple>) -> Vec<RuleAtom> {
    triples.iter()
        .map(triple_to_rule_atom)
        .collect()
}

/// Export rule atoms as RDF triples
pub fn export_rule_atoms(atoms: Vec<RuleAtom>) -> Result<Vec<Triple>> {
    atoms.iter()
        .filter_map(|atom| match atom {
            RuleAtom::Triple { .. } => Some(rule_atom_to_triple(atom)),
            RuleAtom::Builtin { .. } => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_conversion() {
        let rule_var = RuleTerm::Variable("x".to_string());
        let rdf_term = rule_term_to_rdf_term(&rule_var).unwrap();
        
        assert!(matches!(rdf_term, RdfTerm::Variable(_)));
        if let RdfTerm::Variable(v) = rdf_term {
            assert_eq!(v.name(), "x");
        }
        
        // Round trip
        let back = rdf_term_to_rule_term(&RdfTerm::Variable(Variable::new("x").unwrap()));
        assert_eq!(back, rule_var);
    }

    #[test]
    fn test_iri_conversion() {
        let rule_iri = RuleTerm::Constant("http://example.org/test".to_string());
        let rdf_term = rule_term_to_rdf_term(&rule_iri).unwrap();
        
        assert!(matches!(rdf_term, RdfTerm::NamedNode(_)));
        if let RdfTerm::NamedNode(n) = &rdf_term {
            assert_eq!(n.as_str(), "http://example.org/test");
        }
        
        // Round trip
        let back = rdf_term_to_rule_term(&rdf_term);
        assert_eq!(back, rule_iri);
    }

    #[test]
    fn test_blank_node_conversion() {
        let rule_blank = RuleTerm::Constant("_:b1".to_string());
        let rdf_term = rule_term_to_rdf_term(&rule_blank).unwrap();
        
        assert!(matches!(rdf_term, RdfTerm::BlankNode(_)));
        if let RdfTerm::BlankNode(b) = &rdf_term {
            assert_eq!(b.id(), "b1");
        }
    }

    #[test]
    fn test_literal_conversion() {
        // Simple literal
        let rule_lit = RuleTerm::Literal("\"hello\"".to_string());
        let rdf_term = rule_term_to_rdf_term(&rule_lit).unwrap();
        
        assert!(matches!(rdf_term, RdfTerm::Literal(_)));
        if let RdfTerm::Literal(l) = &rdf_term {
            assert_eq!(l.value(), "hello");
        }
        
        // Typed literal
        let typed_lit = RuleTerm::Literal("\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>".to_string());
        let rdf_typed = rule_term_to_rdf_term(&typed_lit).unwrap();
        
        if let RdfTerm::Literal(l) = &rdf_typed {
            assert_eq!(l.value(), "42");
            assert!(l.datatype().is_some());
        }
        
        // Language-tagged literal
        let lang_lit = RuleTerm::Literal("\"bonjour\"@fr".to_string());
        let rdf_lang = rule_term_to_rdf_term(&lang_lit).unwrap();
        
        if let RdfTerm::Literal(l) = &rdf_lang {
            assert_eq!(l.value(), "bonjour");
            assert_eq!(l.language(), Some("fr"));
        }
    }

    #[test]
    fn test_triple_conversion() {
        let rule_atom = RuleAtom::Triple {
            subject: RuleTerm::Constant("http://example.org/alice".to_string()),
            predicate: RuleTerm::Constant("http://xmlns.com/foaf/0.1/knows".to_string()),
            object: RuleTerm::Constant("http://example.org/bob".to_string()),
        };
        
        let triple = rule_atom_to_triple(&rule_atom).unwrap();
        assert_eq!(triple.subject().to_string(), "<http://example.org/alice>");
        assert_eq!(triple.predicate().to_string(), "<http://xmlns.com/foaf/0.1/knows>");
        assert_eq!(triple.object().to_string(), "<http://example.org/bob>");
        
        // Round trip
        let back = triple_to_rule_atom(&triple);
        assert_eq!(back, rule_atom);
    }
}