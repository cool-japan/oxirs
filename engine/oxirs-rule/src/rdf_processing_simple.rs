//! Simplified RDF Data Processing Module
//!
//! This module provides basic functionality for processing RDF data
//! and converting it to rule atoms.

use crate::rdf_integration::{NamespaceManager, RdfRuleAtom, RdfTerm};
use crate::{RuleAtom, Term as RuleTerm};
use anyhow::{anyhow, Result};
use oxirs_core::model::{BlankNode, Literal, NamedNode, Quad, Term, Triple, Variable};
use oxirs_core::{OxirsError, Store};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

/// Simple RDF triple for parsing
#[derive(Debug, Clone)]
pub struct SimpleTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub object_type: ObjectType,
}

#[derive(Debug, Clone)]
pub enum ObjectType {
    Iri,
    Literal(Option<String>, Option<String>), // value, lang/datatype
    BlankNode,
}

/// RDF format types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RdfFormat {
    NTriples,
    Turtle,
    RdfXml,
}

/// Simple RDF processor
pub struct SimpleRdfProcessor {
    store: Arc<dyn Store>,
    namespaces: NamespaceManager,
}

impl SimpleRdfProcessor {
    /// Create a new processor
    pub fn new(store: Arc<dyn Store>) -> Self {
        Self {
            store,
            namespaces: NamespaceManager::new(),
        }
    }

    /// Process N-Triples data
    pub fn process_ntriples(&mut self, data: &str) -> Result<Vec<RuleAtom>> {
        let mut atoms = Vec::new();

        for line in data.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Ok(triple) = self.parse_ntriples_line(line) {
                let atom = self.triple_to_rule_atom(&triple)?;
                atoms.push(atom);

                // Note: Store insertion handled separately to avoid Arc mutability issues
            }
        }

        Ok(atoms)
    }

    /// Parse a single N-Triples line
    fn parse_ntriples_line(&self, line: &str) -> Result<SimpleTriple> {
        // Very basic N-Triples parser
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 || !line.ends_with('.') {
            return Err(anyhow!("Invalid N-Triples line"));
        }

        let subject = parts[0].trim_start_matches('<').trim_end_matches('>');
        let predicate = parts[1].trim_start_matches('<').trim_end_matches('>');

        // Parse object
        let object_str = parts[2..parts.len() - 1].join(" ");
        let (object, object_type) = if object_str.starts_with('<') && object_str.ends_with('>') {
            // IRI
            let iri = object_str.trim_start_matches('<').trim_end_matches('>');
            (iri.to_string(), ObjectType::Iri)
        } else if object_str.starts_with('"') {
            // Literal
            self.parse_literal(&object_str)?
        } else if object_str.starts_with("_:") {
            // Blank node
            (object_str[2..].to_string(), ObjectType::BlankNode)
        } else {
            return Err(anyhow!("Invalid object format"));
        };

        Ok(SimpleTriple {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object,
            object_type,
        })
    }

    /// Parse a literal from N-Triples format
    fn parse_literal(&self, literal_str: &str) -> Result<(String, ObjectType)> {
        // Find the closing quote
        let end_quote = literal_str
            .rfind('"')
            .ok_or_else(|| anyhow!("No closing quote"))?;
        let value = &literal_str[1..end_quote];

        // Check for language tag or datatype
        let remainder = &literal_str[end_quote + 1..];
        if remainder.starts_with("@") {
            // Language tag
            let lang = remainder[1..].to_string();
            Ok((
                value.to_string(),
                ObjectType::Literal(Some(value.to_string()), Some(lang)),
            ))
        } else if remainder.starts_with("^^") {
            // Datatype
            let datatype = remainder[2..]
                .trim_start_matches('<')
                .trim_end_matches('>')
                .to_string();
            Ok((
                value.to_string(),
                ObjectType::Literal(Some(value.to_string()), Some(datatype)),
            ))
        } else {
            // Plain literal
            Ok((
                value.to_string(),
                ObjectType::Literal(Some(value.to_string()), None),
            ))
        }
    }

    /// Convert SimpleTriple to RuleAtom
    fn triple_to_rule_atom(&self, triple: &SimpleTriple) -> Result<RuleAtom> {
        let subject = if triple.subject.starts_with("_:") {
            RuleTerm::Constant(format!("_:{}", &triple.subject[2..]))
        } else {
            RuleTerm::Constant(self.namespaces.compact(&triple.subject))
        };

        let predicate = RuleTerm::Constant(self.namespaces.compact(&triple.predicate));

        let object = match &triple.object_type {
            ObjectType::Iri => RuleTerm::Constant(self.namespaces.compact(&triple.object)),
            ObjectType::Literal(Some(value), lang_or_dt) => {
                if let Some(lang_or_dt) = lang_or_dt {
                    if lang_or_dt.starts_with("http://") || lang_or_dt.starts_with("https://") {
                        // Datatype
                        RuleTerm::Literal(format!(
                            "{}^^{}",
                            value,
                            self.namespaces.compact(lang_or_dt)
                        ))
                    } else {
                        // Language
                        RuleTerm::Literal(format!("{}@{}", value, lang_or_dt))
                    }
                } else {
                    RuleTerm::Literal(value.clone())
                }
            }
            ObjectType::BlankNode => RuleTerm::Constant(format!("_:{}", triple.object)),
            _ => RuleTerm::Literal(triple.object.clone()),
        };

        Ok(RuleAtom::Triple {
            subject,
            predicate,
            object,
        })
    }

    /// Add triple to store
    fn add_triple_to_store(&self, triple: &SimpleTriple) -> Result<()> {
        // Create proper RDF terms
        let subject: oxirs_core::Subject = if triple.subject.starts_with("_:") {
            BlankNode::new(&triple.subject[2..])?.into()
        } else {
            NamedNode::new(&triple.subject)?.into()
        };

        let predicate = NamedNode::new(&triple.predicate)?;

        let object: oxirs_core::Object = match &triple.object_type {
            ObjectType::Iri => NamedNode::new(&triple.object)?.into(),
            ObjectType::Literal(Some(value), lang_or_dt) => {
                if let Some(lang_or_dt) = lang_or_dt {
                    if lang_or_dt.starts_with("http://") || lang_or_dt.starts_with("https://") {
                        // Typed literal
                        let dt = NamedNode::new(lang_or_dt)?;
                        Literal::new_typed_literal(value.clone(), dt).into()
                    } else {
                        // Language-tagged literal
                        Literal::new_language_tagged_literal(value.clone(), lang_or_dt)?.into()
                    }
                } else {
                    Literal::new_simple_literal(value.clone()).into()
                }
            }
            ObjectType::BlankNode => BlankNode::new(&triple.object)?.into(),
            _ => Literal::new_simple_literal(triple.object.clone()).into(),
        };

        // Note: Store insertion removed to avoid Arc<Store> mutability issues
        // The caller can handle storing the processed data if needed
        let _quad = Quad::new(
            subject,
            predicate,
            object,
            oxirs_core::model::GraphName::DefaultGraph,
        );
        // self.store.insert(&quad)?; // Would require mutable store
        Ok(())
    }

    /// Process Turtle data (simplified - only handles basic prefixes)
    pub fn process_turtle(&mut self, data: &str) -> Result<Vec<RuleAtom>> {
        let mut atoms = Vec::new();

        // First pass: extract prefixes
        for line in data.lines() {
            let line = line.trim();
            if line.starts_with("@prefix") || line.starts_with("PREFIX") {
                self.parse_prefix_declaration(line)?;
            } else if line.starts_with("@base") || line.starts_with("BASE") {
                self.parse_base_declaration(line)?;
            }
        }

        // Second pass: parse triples (very basic)
        // This is a placeholder - real Turtle parsing is complex
        // For now, we'll just handle simple subject predicate object . patterns

        Ok(atoms)
    }

    /// Parse prefix declaration
    fn parse_prefix_declaration(&mut self, line: &str) -> Result<()> {
        // Simple regex-based parsing
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let prefix = parts[1].trim_end_matches(':');
            let namespace = parts[2]
                .trim_start_matches('<')
                .trim_end_matches('>')
                .trim_end_matches('.');
            self.namespaces
                .add_prefix(prefix.to_string(), namespace.to_string());
        }
        Ok(())
    }

    /// Parse base declaration
    fn parse_base_declaration(&mut self, line: &str) -> Result<()> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            let base = parts[1]
                .trim_start_matches('<')
                .trim_end_matches('>')
                .trim_end_matches('.');
            self.namespaces.set_base(base.to_string());
        }
        Ok(())
    }

    /// Get namespaces
    pub fn namespaces(&self) -> &NamespaceManager {
        &self.namespaces
    }

    /// Get all facts from store as rule atoms
    pub fn get_facts_as_atoms(&self) -> Result<Vec<RuleAtom>> {
        let mut atoms = Vec::new();

        // Iterate through store quads using query_quads
        let quads = self.store.query_quads(None, None, None, None)?;

        for quad in quads {
            // Convert to RuleAtom
            let subject = self.term_to_rule_term(&Term::from_subject(quad.subject()))?;
            let predicate = self.term_to_rule_term(&Term::from_predicate(quad.predicate()))?;
            let object = self.term_to_rule_term(&Term::from_object(quad.object()))?;

            atoms.push(RuleAtom::Triple {
                subject,
                predicate,
                object,
            });
        }

        Ok(atoms)
    }

    /// Convert RDF term to rule term
    fn term_to_rule_term(&self, term: &Term) -> Result<RuleTerm> {
        match term {
            Term::NamedNode(n) => Ok(RuleTerm::Constant(self.namespaces.compact(n.as_str()))),
            Term::BlankNode(b) => Ok(RuleTerm::Constant(format!("_:{}", b.as_str()))),
            Term::Literal(l) => {
                if let Some(lang) = l.language() {
                    Ok(RuleTerm::Literal(format!("{}@{}", l.value(), lang)))
                } else if l.datatype().as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                    Ok(RuleTerm::Literal(format!(
                        "{}^^{}",
                        l.value(),
                        self.namespaces.compact(l.datatype().as_str())
                    )))
                } else {
                    Ok(RuleTerm::Literal(l.value().to_string()))
                }
            }
            Term::Variable(v) => Ok(RuleTerm::Variable(v.name().to_string())),
            Term::QuotedTriple(_) => Err(anyhow!("Quoted triples not supported")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ntriples() {
        let store = Arc::new(Store::new().unwrap());
        let mut processor = SimpleRdfProcessor::new(store);

        let ntriples = r#"
<http://example.org/subject> <http://example.org/predicate> <http://example.org/object> .
<http://example.org/s2> <http://example.org/p2> "literal value" .
<http://example.org/s3> <http://example.org/p3> "hello"@en .
<http://example.org/s4> <http://example.org/p4> "42"^^<http://www.w3.org/2001/XMLSchema#integer> .
_:blank1 <http://example.org/p5> _:blank2 .
"#;

        let atoms = processor.process_ntriples(ntriples).unwrap();
        assert_eq!(atoms.len(), 5);

        // Check first atom
        match &atoms[0] {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                assert!(matches!(subject, RuleTerm::Constant(s) if s.contains("subject")));
                assert!(matches!(predicate, RuleTerm::Constant(p) if p.contains("predicate")));
                assert!(matches!(object, RuleTerm::Constant(o) if o.contains("object")));
            }
            _ => panic!("Expected triple"),
        }

        // Check literal
        match &atoms[1] {
            RuleAtom::Triple { object, .. } => {
                assert!(matches!(object, RuleTerm::Literal(l) if l == "literal value"));
            }
            _ => panic!("Expected triple"),
        }

        // Check language-tagged literal
        match &atoms[2] {
            RuleAtom::Triple { object, .. } => {
                assert!(matches!(object, RuleTerm::Literal(l) if l == "hello@en"));
            }
            _ => panic!("Expected triple"),
        }
    }

    #[test]
    fn test_parse_prefixes() {
        let store = Arc::new(Store::new().unwrap());
        let mut processor = SimpleRdfProcessor::new(store);

        let turtle = r#"
@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@base <http://example.org/base/> .
"#;

        processor.process_turtle(turtle).unwrap();

        assert_eq!(
            processor.namespaces.expand("ex:Person").unwrap(),
            "http://example.org/Person"
        );
        assert_eq!(
            processor.namespaces.expand("foaf:name").unwrap(),
            "http://xmlns.com/foaf/0.1/name"
        );
    }
}
