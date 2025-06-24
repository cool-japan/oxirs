//! Integration Bridge with OxiRS Core
//!
//! This module provides seamless integration between the oxirs-rule engine
//! and oxirs-core RDF model, allowing rules to operate directly on core RDF types.

use crate::{Rule, RuleAtom, Term, RuleEngine};
use anyhow::Result;
use oxirs_core::{Store, Triple, Quad, NamedNode, Literal, Subject, Predicate, Object, GraphName};
use std::collections::HashMap;
use tracing::{debug, info, trace, warn};

/// Integration bridge for connecting oxirs-rule with oxirs-core
#[derive(Debug)]
pub struct RuleIntegration {
    /// Core rule engine
    pub rule_engine: RuleEngine,
    /// Core RDF store
    pub store: Store,
}

impl Default for RuleIntegration {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleIntegration {
    /// Create a new rule integration with an empty store
    pub fn new() -> Self {
        Self {
            rule_engine: RuleEngine::new(),
            store: Store::new().unwrap(),
        }
    }

    /// Create integration with an existing store
    pub fn with_store(store: Store) -> Self {
        Self {
            rule_engine: RuleEngine::new(),
            store,
        }
    }

    /// Add a rule to the engine
    pub fn add_rule(&mut self, rule: Rule) {
        self.rule_engine.add_rule(rule);
    }

    /// Add multiple rules
    pub fn add_rules(&mut self, rules: Vec<Rule>) {
        self.rule_engine.add_rules(rules);
    }

    /// Load facts from the core store into the rule engine
    pub fn load_facts_from_store(&mut self) -> Result<usize> {
        let quads = self.store.iter_quads()?;
        let rule_atoms: Vec<RuleAtom> = quads.into_iter()
            .map(|quad| self.quad_to_rule_atom(&quad))
            .collect();
        
        let fact_count = rule_atoms.len();
        self.rule_engine.add_facts(rule_atoms);
        
        info!("Loaded {} facts from store into rule engine", fact_count);
        Ok(fact_count)
    }

    /// Apply rules and store derived facts back to the core store
    pub fn apply_rules(&mut self) -> Result<usize> {
        // Load current facts from store
        self.load_facts_from_store()?;
        
        // Apply forward chaining
        let derived_facts = self.rule_engine.forward_chain(&[])?;
        
        // Convert derived facts back to core model and store them
        let mut new_fact_count = 0;
        for rule_atom in derived_facts {
            if let Ok(triple) = self.rule_atom_to_triple(&rule_atom) {
                if self.store.insert_triple(triple)? {
                    new_fact_count += 1;
                }
            }
        }
        
        info!("Applied rules and derived {} new facts", new_fact_count);
        Ok(new_fact_count)
    }

    /// Query the rule engine using backward chaining for a goal
    pub fn prove_goal(&mut self, goal_triple: &Triple) -> Result<bool> {
        // Convert triple to rule atom
        let goal_atom = self.triple_to_rule_atom(goal_triple);
        
        // Load current facts
        self.load_facts_from_store()?;
        
        // Attempt to prove the goal
        self.rule_engine.backward_chain(&goal_atom)
    }

    /// Find all solutions for a query pattern
    pub fn query_with_rules(&mut self, 
                           subject: Option<&Subject>,
                           predicate: Option<&Predicate>, 
                           object: Option<&Object>) -> Result<Vec<Triple>> {
        // First get direct matches from store
        let direct_matches = self.store.query_triples(subject, predicate, object)?;
        
        // Apply rules to potentially derive more facts
        self.apply_rules()?;
        
        // Query again after applying rules
        let rule_enhanced_matches = self.store.query_triples(subject, predicate, object)?;
        
        Ok(rule_enhanced_matches)
    }

    /// Get comprehensive statistics about the integration
    pub fn get_integration_stats(&self) -> Result<IntegrationStats> {
        let store_quad_count = self.store.len()?;
        let rule_fact_count = self.rule_engine.get_facts().len();
        let rule_count = self.rule_engine.rules.len();
        
        Ok(IntegrationStats {
            store_quad_count,
            rule_fact_count,
            rule_count,
        })
    }

    /// Convert a core Triple to a RuleAtom
    fn triple_to_rule_atom(&self, triple: &Triple) -> RuleAtom {
        RuleAtom::Triple {
            subject: self.subject_to_term(triple.subject()),
            predicate: self.predicate_to_term(triple.predicate()),
            object: self.object_to_term(triple.object()),
        }
    }

    /// Convert a core Quad to a RuleAtom (ignoring graph for now)
    fn quad_to_rule_atom(&self, quad: &Quad) -> RuleAtom {
        RuleAtom::Triple {
            subject: self.subject_to_term(quad.subject()),
            predicate: self.predicate_to_term(quad.predicate()),
            object: self.object_to_term(quad.object()),
        }
    }

    /// Convert a RuleAtom to a core Triple (if possible)
    fn rule_atom_to_triple(&self, atom: &RuleAtom) -> Result<Triple> {
        match atom {
            RuleAtom::Triple { subject, predicate, object } => {
                let core_subject = self.term_to_subject(subject)?;
                let core_predicate = self.term_to_predicate(predicate)?;
                let core_object = self.term_to_object(object)?;
                
                Ok(Triple::new(core_subject, core_predicate, core_object))
            }
            RuleAtom::Builtin { .. } => {
                Err(anyhow::anyhow!("Cannot convert builtin rule atom to triple"))
            }
        }
    }

    /// Convert core Subject to rule Term
    fn subject_to_term(&self, subject: &Subject) -> Term {
        match subject {
            Subject::NamedNode(node) => Term::Constant(node.as_str().to_string()),
            Subject::BlankNode(node) => Term::Constant(format!("_:{}", node.as_str())),
        }
    }

    /// Convert core Predicate to rule Term
    fn predicate_to_term(&self, predicate: &Predicate) -> Term {
        match predicate {
            Predicate::NamedNode(node) => Term::Constant(node.as_str().to_string()),
        }
    }

    /// Convert core Object to rule Term
    fn object_to_term(&self, object: &Object) -> Term {
        match object {
            Object::NamedNode(node) => Term::Constant(node.as_str().to_string()),
            Object::BlankNode(node) => Term::Constant(format!("_:{}", node.as_str())),
            Object::Literal(literal) => Term::Literal(literal.value().to_string()),
        }
    }

    /// Convert rule Term to core Subject
    fn term_to_subject(&self, term: &Term) -> Result<Subject> {
        match term {
            Term::Constant(value) => {
                if value.starts_with("_:") {
                    // Blank node
                    Ok(Subject::BlankNode(oxirs_core::BlankNode::new(&value[2..])?))
                } else {
                    // Named node
                    Ok(Subject::NamedNode(NamedNode::new(value)?))
                }
            }
            Term::Variable(_) => {
                Err(anyhow::anyhow!("Cannot convert unbound variable to subject"))
            }
            Term::Literal(_) => {
                Err(anyhow::anyhow!("Literals cannot be subjects in RDF"))
            }
        }
    }

    /// Convert rule Term to core Predicate
    fn term_to_predicate(&self, term: &Term) -> Result<Predicate> {
        match term {
            Term::Constant(value) => {
                Ok(Predicate::NamedNode(NamedNode::new(value)?))
            }
            Term::Variable(_) => {
                Err(anyhow::anyhow!("Cannot convert unbound variable to predicate"))
            }
            Term::Literal(_) => {
                Err(anyhow::anyhow!("Literals cannot be predicates in RDF"))
            }
        }
    }

    /// Convert rule Term to core Object
    fn term_to_object(&self, term: &Term) -> Result<Object> {
        match term {
            Term::Constant(value) => {
                if value.starts_with("_:") {
                    // Blank node
                    Ok(Object::BlankNode(oxirs_core::BlankNode::new(&value[2..])?))
                } else {
                    // Named node
                    Ok(Object::NamedNode(NamedNode::new(value)?))
                }
            }
            Term::Literal(value) => {
                Ok(Object::Literal(Literal::new(value)))
            }
            Term::Variable(_) => {
                Err(anyhow::anyhow!("Cannot convert unbound variable to object"))
            }
        }
    }
}

/// Statistics about the rule-store integration
#[derive(Debug, Clone)]
pub struct IntegrationStats {
    /// Number of quads in the store
    pub store_quad_count: usize,
    /// Number of facts in the rule engine
    pub rule_fact_count: usize,
    /// Number of rules in the engine
    pub rule_count: usize,
}

impl std::fmt::Display for IntegrationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Store: {} quads, Rules: {} facts/{} rules", 
               self.store_quad_count, self.rule_fact_count, self.rule_count)
    }
}

/// Convenience functions for creating common rules from RDF patterns
pub mod rule_builders {
    use super::*;

    /// Create an RDFS subClassOf transitivity rule
    pub fn rdfs_subclass_transitivity() -> Rule {
        Rule {
            name: "rdfs_subclass_transitivity".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("http://www.w3.org/2000/01/rdf-schema#subClassOf".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("http://www.w3.org/2000/01/rdf-schema#subClassOf".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("http://www.w3.org/2000/01/rdf-schema#subClassOf".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
        }
    }

    /// Create an RDFS type inheritance rule
    pub fn rdfs_type_inheritance() -> Rule {
        Rule {
            name: "rdfs_type_inheritance".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()),
                    object: Term::Variable("C1".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("C1".to_string()),
                    predicate: Term::Constant("http://www.w3.org/2000/01/rdf-schema#subClassOf".to_string()),
                    object: Term::Variable("C2".to_string()),
                },
            ],
            head: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()),
                    object: Term::Variable("C2".to_string()),
                },
            ],
        }
    }

    /// Create a domain inference rule
    pub fn rdfs_domain_inference() -> Rule {
        Rule {
            name: "rdfs_domain_inference".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("P".to_string()),
                    predicate: Term::Constant("http://www.w3.org/2000/01/rdf-schema#domain".to_string()),
                    object: Term::Variable("C".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Variable("P".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
            ],
            head: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()),
                    object: Term::Variable("C".to_string()),
                },
            ],
        }
    }

    /// Create a range inference rule
    pub fn rdfs_range_inference() -> Rule {
        Rule {
            name: "rdfs_range_inference".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("P".to_string()),
                    predicate: Term::Constant("http://www.w3.org/2000/01/rdf-schema#range".to_string()),
                    object: Term::Variable("C".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Variable("P".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
            ],
            head: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()),
                    object: Term::Variable("C".to_string()),
                },
            ],
        }
    }

    /// Create all standard RDFS rules
    pub fn all_rdfs_rules() -> Vec<Rule> {
        vec![
            rdfs_subclass_transitivity(),
            rdfs_type_inheritance(),
            rdfs_domain_inference(),
            rdfs_range_inference(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::{NamedNode, Literal, Triple};

    #[test]
    fn test_integration_basic_workflow() {
        let mut integration = RuleIntegration::new();
        
        // Add some test data to the store
        let subject = NamedNode::new("http://example.org/person").unwrap();
        let predicate = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
        let object = NamedNode::new("http://example.org/Human").unwrap();
        
        let triple = Triple::new(
            subject.clone(),
            predicate.clone(),
            Object::NamedNode(object)
        );
        
        integration.store.insert_triple(triple.clone()).unwrap();
        
        // Add a rule: Human -> Mortal
        let rule = Rule {
            name: "human_mortal".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()),
                    object: Term::Constant("http://example.org/Human".to_string()),
                },
            ],
            head: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()),
                    object: Term::Constant("http://example.org/Mortal".to_string()),
                },
            ],
        };
        
        integration.add_rule(rule);
        
        // Apply rules
        let derived_count = integration.apply_rules().unwrap();
        assert!(derived_count > 0);
        
        // Check that the mortal type was derived
        let mortal_type = NamedNode::new("http://example.org/Mortal").unwrap();
        let results = integration.store.query_triples(
            Some(&Subject::NamedNode(subject)),
            Some(&Predicate::NamedNode(predicate)),
            Some(&Object::NamedNode(mortal_type))
        ).unwrap();
        
        assert!(!results.is_empty());
    }

    #[test]
    fn test_conversion_functions() {
        let integration = RuleIntegration::new();
        
        // Test triple to rule atom conversion
        let subject = NamedNode::new("http://example.org/subject").unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = Literal::new("test");
        
        let triple = Triple::new(subject, predicate, object);
        let rule_atom = integration.triple_to_rule_atom(&triple);
        
        match rule_atom {
            RuleAtom::Triple { subject: s, predicate: p, object: o } => {
                assert!(matches!(s, Term::Constant(_)));
                assert!(matches!(p, Term::Constant(_)));
                assert!(matches!(o, Term::Literal(_)));
            }
            _ => panic!("Expected triple rule atom"),
        }
        
        // Test rule atom to triple conversion
        let converted_triple = integration.rule_atom_to_triple(&rule_atom).unwrap();
        assert_eq!(converted_triple.subject().as_str(), "http://example.org/subject");
        assert_eq!(converted_triple.predicate().as_str(), "http://example.org/predicate");
    }

    #[test]
    fn test_rule_builders() {
        let rules = rule_builders::all_rdfs_rules();
        assert_eq!(rules.len(), 4);
        
        // Check that all rules have proper names
        let rule_names: Vec<String> = rules.iter().map(|r| r.name.clone()).collect();
        assert!(rule_names.contains(&"rdfs_subclass_transitivity".to_string()));
        assert!(rule_names.contains(&"rdfs_type_inheritance".to_string()));
        assert!(rule_names.contains(&"rdfs_domain_inference".to_string()));
        assert!(rule_names.contains(&"rdfs_range_inference".to_string()));
    }

    #[test]
    fn test_query_with_rules() {
        let mut integration = RuleIntegration::new();
        
        // Add RDFS rules
        integration.add_rules(rule_builders::all_rdfs_rules());
        
        // Add some test ontology data
        let person = NamedNode::new("http://example.org/Person").unwrap();
        let student = NamedNode::new("http://example.org/Student").unwrap();
        let subclass_pred = NamedNode::new("http://www.w3.org/2000/01/rdf-schema#subClassOf").unwrap();
        let type_pred = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
        let alice = NamedNode::new("http://example.org/alice").unwrap();
        
        // Student subClassOf Person
        let subclass_triple = Triple::new(
            Subject::NamedNode(student.clone()),
            Predicate::NamedNode(subclass_pred),
            Object::NamedNode(person.clone())
        );
        
        // alice type Student
        let alice_type_triple = Triple::new(
            Subject::NamedNode(alice.clone()),
            Predicate::NamedNode(type_pred.clone()),
            Object::NamedNode(student)
        );
        
        integration.store.insert_triple(subclass_triple).unwrap();
        integration.store.insert_triple(alice_type_triple).unwrap();
        
        // Query for all types of alice (should include both Student and Person via inference)
        let results = integration.query_with_rules(
            Some(&Subject::NamedNode(alice)),
            Some(&Predicate::NamedNode(type_pred)),
            None
        ).unwrap();
        
        // Should find at least 2 results (Student and Person)
        assert!(results.len() >= 2);
        
        let has_person_type = results.iter().any(|triple| {
            if let Object::NamedNode(node) = triple.object() {
                node.as_str() == "http://example.org/Person"
            } else {
                false
            }
        });
        
        assert!(has_person_type, "Should infer that alice is a Person");
    }

    #[test]
    fn test_statistics() {
        let mut integration = RuleIntegration::new();
        
        // Add some data
        let triple = Triple::new(
            NamedNode::new("http://example.org/s").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            Literal::new("o")
        );
        integration.store.insert_triple(triple).unwrap();
        
        // Add a rule
        integration.add_rule(rule_builders::rdfs_type_inheritance());
        
        let stats = integration.get_integration_stats().unwrap();
        assert_eq!(stats.store_quad_count, 1);
        assert_eq!(stats.rule_count, 1);
    }
}