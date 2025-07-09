//! RDFS Reasoning Engine
//!
//! Implementation of RDFS (RDF Schema) reasoning based on the W3C RDFS specification.
//! Supports class hierarchy inference, property hierarchy inference, and domain/range inference.

use crate::{Rule, RuleAtom, RuleEngine, Term};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, trace};

/// RDFS vocabulary constants
pub mod vocabulary {
    pub const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    pub const RDFS_SUBCLASS_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";
    pub const RDFS_SUBPROPERTY_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subPropertyOf";
    pub const RDFS_DOMAIN: &str = "http://www.w3.org/2000/01/rdf-schema#domain";
    pub const RDFS_RANGE: &str = "http://www.w3.org/2000/01/rdf-schema#range";
    pub const RDFS_CLASS: &str = "http://www.w3.org/2000/01/rdf-schema#Class";
    pub const RDFS_RESOURCE: &str = "http://www.w3.org/2000/01/rdf-schema#Resource";
    pub const RDFS_LITERAL: &str = "http://www.w3.org/2000/01/rdf-schema#Literal";
    pub const RDFS_DATATYPE: &str = "http://www.w3.org/2000/01/rdf-schema#Datatype";
    pub const RDF_PROPERTY: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property";
}

/// RDFS inference context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfsContext {
    /// Class hierarchy (subclass relationships)
    pub class_hierarchy: HashMap<String, HashSet<String>>,
    /// Property hierarchy (subproperty relationships)
    pub property_hierarchy: HashMap<String, HashSet<String>>,
    /// Property domains
    pub property_domains: HashMap<String, HashSet<String>>,
    /// Property ranges
    pub property_ranges: HashMap<String, HashSet<String>>,
    /// Known classes
    pub classes: HashSet<String>,
    /// Known properties
    pub properties: HashSet<String>,
}

impl Default for RdfsContext {
    fn default() -> Self {
        let mut context = Self {
            class_hierarchy: HashMap::new(),
            property_hierarchy: HashMap::new(),
            property_domains: HashMap::new(),
            property_ranges: HashMap::new(),
            classes: HashSet::new(),
            properties: HashSet::new(),
        };

        // Add RDFS built-in classes and properties
        context.initialize_builtin_vocabulary();
        context
    }
}

impl RdfsContext {
    /// Initialize built-in RDFS vocabulary
    fn initialize_builtin_vocabulary(&mut self) {
        use vocabulary::*;

        // Built-in classes
        self.classes.insert(RDFS_CLASS.to_string());
        self.classes.insert(RDFS_RESOURCE.to_string());
        self.classes.insert(RDFS_LITERAL.to_string());
        self.classes.insert(RDFS_DATATYPE.to_string());

        // Built-in properties
        self.properties.insert(RDF_TYPE.to_string());
        self.properties.insert(RDFS_SUBCLASS_OF.to_string());
        self.properties.insert(RDFS_SUBPROPERTY_OF.to_string());
        self.properties.insert(RDFS_DOMAIN.to_string());
        self.properties.insert(RDFS_RANGE.to_string());

        // RDFS Class hierarchy
        self.add_subclass_relation(RDFS_CLASS, RDFS_RESOURCE);
        self.add_subclass_relation(RDFS_DATATYPE, RDFS_CLASS);

        // Property domains and ranges
        self.add_property_domain(RDFS_SUBCLASS_OF, RDFS_CLASS);
        self.add_property_range(RDFS_SUBCLASS_OF, RDFS_CLASS);
        self.add_property_domain(RDFS_SUBPROPERTY_OF, RDF_PROPERTY);
        self.add_property_range(RDFS_SUBPROPERTY_OF, RDF_PROPERTY);
        self.add_property_domain(RDFS_DOMAIN, RDF_PROPERTY);
        self.add_property_range(RDFS_DOMAIN, RDFS_CLASS);
        self.add_property_domain(RDFS_RANGE, RDF_PROPERTY);
        self.add_property_range(RDFS_RANGE, RDFS_CLASS);
    }

    /// Add a subclass relation
    pub fn add_subclass_relation(&mut self, subclass: &str, superclass: &str) {
        self.class_hierarchy
            .entry(subclass.to_string())
            .or_default()
            .insert(superclass.to_string());
    }

    /// Add a subproperty relation
    pub fn add_subproperty_relation(&mut self, subproperty: &str, superproperty: &str) {
        self.property_hierarchy
            .entry(subproperty.to_string())
            .or_default()
            .insert(superproperty.to_string());
    }

    /// Add a property domain
    pub fn add_property_domain(&mut self, property: &str, domain: &str) {
        self.property_domains
            .entry(property.to_string())
            .or_default()
            .insert(domain.to_string());
    }

    /// Add a property range
    pub fn add_property_range(&mut self, property: &str, range: &str) {
        self.property_ranges
            .entry(property.to_string())
            .or_default()
            .insert(range.to_string());
    }

    /// Get all superclasses of a class (transitive closure)
    pub fn get_superclasses(&self, class: &str) -> HashSet<String> {
        let mut superclasses = HashSet::new();
        let mut to_visit = vec![class.to_string()];
        let mut visited = HashSet::new();

        while let Some(current) = to_visit.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            if let Some(direct_superclasses) = self.class_hierarchy.get(&current) {
                for superclass in direct_superclasses {
                    superclasses.insert(superclass.clone());
                    to_visit.push(superclass.clone());
                }
            }
        }

        superclasses
    }

    /// Get all superproperties of a property (transitive closure)
    pub fn get_superproperties(&self, property: &str) -> HashSet<String> {
        let mut superproperties = HashSet::new();
        let mut to_visit = vec![property.to_string()];
        let mut visited = HashSet::new();

        while let Some(current) = to_visit.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            if let Some(direct_superproperties) = self.property_hierarchy.get(&current) {
                for superproperty in direct_superproperties {
                    superproperties.insert(superproperty.clone());
                    to_visit.push(superproperty.clone());
                }
            }
        }

        superproperties
    }

    /// Check if class A is a subclass of class B
    pub fn is_subclass_of(&self, subclass: &str, superclass: &str) -> bool {
        if subclass == superclass {
            return true;
        }
        self.get_superclasses(subclass).contains(superclass)
    }

    /// Check if property A is a subproperty of property B
    pub fn is_subproperty_of(&self, subproperty: &str, superproperty: &str) -> bool {
        if subproperty == superproperty {
            return true;
        }
        self.get_superproperties(subproperty)
            .contains(superproperty)
    }
}

/// RDFS reasoning engine
#[derive(Debug)]
pub struct RdfsReasoner {
    /// RDFS context
    pub context: RdfsContext,
    /// Rule engine for RDFS rules
    pub rule_engine: RuleEngine,
}

impl Default for RdfsReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl RdfsReasoner {
    /// Create a new RDFS reasoner
    pub fn new() -> Self {
        let mut reasoner = Self {
            context: RdfsContext::default(),
            rule_engine: RuleEngine::new(),
        };

        reasoner.initialize_rdfs_rules();
        reasoner
    }

    /// Initialize RDFS entailment rules
    fn initialize_rdfs_rules(&mut self) {
        use vocabulary::*;

        // RDFS Rule 2: Triple (?p rdfs:domain ?c) + (?x ?p ?y) => (?x rdf:type ?c)
        self.rule_engine.add_rule(Rule {
            name: "rdfs2".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("p".to_string()),
                    predicate: Term::Constant(RDFS_DOMAIN.to_string()),
                    object: Term::Variable("c".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("x".to_string()),
                    predicate: Term::Variable("p".to_string()),
                    object: Term::Variable("y".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("x".to_string()),
                predicate: Term::Constant(RDF_TYPE.to_string()),
                object: Term::Variable("c".to_string()),
            }],
        });

        // RDFS Rule 3: Triple (?p rdfs:range ?c) + (?x ?p ?y) => (?y rdf:type ?c)
        self.rule_engine.add_rule(Rule {
            name: "rdfs3".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("p".to_string()),
                    predicate: Term::Constant(RDFS_RANGE.to_string()),
                    object: Term::Variable("c".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("x".to_string()),
                    predicate: Term::Variable("p".to_string()),
                    object: Term::Variable("y".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("y".to_string()),
                predicate: Term::Constant(RDF_TYPE.to_string()),
                object: Term::Variable("c".to_string()),
            }],
        });

        // RDFS Rule 5: Triple (?p rdfs:subPropertyOf ?q) + (?q rdfs:subPropertyOf ?r) => (?p rdfs:subPropertyOf ?r)
        self.rule_engine.add_rule(Rule {
            name: "rdfs5".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("p".to_string()),
                    predicate: Term::Constant(RDFS_SUBPROPERTY_OF.to_string()),
                    object: Term::Variable("q".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("q".to_string()),
                    predicate: Term::Constant(RDFS_SUBPROPERTY_OF.to_string()),
                    object: Term::Variable("r".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("p".to_string()),
                predicate: Term::Constant(RDFS_SUBPROPERTY_OF.to_string()),
                object: Term::Variable("r".to_string()),
            }],
        });

        // RDFS Rule 7: Triple (?x ?p ?y) + (?p rdfs:subPropertyOf ?q) => (?x ?q ?y)
        self.rule_engine.add_rule(Rule {
            name: "rdfs7".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("x".to_string()),
                    predicate: Term::Variable("p".to_string()),
                    object: Term::Variable("y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("p".to_string()),
                    predicate: Term::Constant(RDFS_SUBPROPERTY_OF.to_string()),
                    object: Term::Variable("q".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("x".to_string()),
                predicate: Term::Variable("q".to_string()),
                object: Term::Variable("y".to_string()),
            }],
        });

        // RDFS Rule 9: Triple (?x rdf:type ?c) + (?c rdfs:subClassOf ?d) => (?x rdf:type ?d)
        self.rule_engine.add_rule(Rule {
            name: "rdfs9".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("x".to_string()),
                    predicate: Term::Constant(RDF_TYPE.to_string()),
                    object: Term::Variable("c".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("c".to_string()),
                    predicate: Term::Constant(RDFS_SUBCLASS_OF.to_string()),
                    object: Term::Variable("d".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("x".to_string()),
                predicate: Term::Constant(RDF_TYPE.to_string()),
                object: Term::Variable("d".to_string()),
            }],
        });

        // RDFS Rule 11: Triple (?c rdfs:subClassOf ?d) + (?d rdfs:subClassOf ?e) => (?c rdfs:subClassOf ?e)
        self.rule_engine.add_rule(Rule {
            name: "rdfs11".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("c".to_string()),
                    predicate: Term::Constant(RDFS_SUBCLASS_OF.to_string()),
                    object: Term::Variable("d".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("d".to_string()),
                    predicate: Term::Constant(RDFS_SUBCLASS_OF.to_string()),
                    object: Term::Variable("e".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("c".to_string()),
                predicate: Term::Constant(RDFS_SUBCLASS_OF.to_string()),
                object: Term::Variable("e".to_string()),
            }],
        });

        info!(
            "Initialized {} RDFS entailment rules",
            self.rule_engine.rules.len()
        );
    }

    /// Process a triple and update the RDFS context
    pub fn process_triple(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> Result<Vec<RuleAtom>> {
        use vocabulary::*;

        let mut new_facts = Vec::new();

        match predicate {
            RDFS_SUBCLASS_OF => {
                debug!("Processing subClassOf: {} -> {}", subject, object);
                self.context.add_subclass_relation(subject, object);
                self.context.classes.insert(subject.to_string());
                self.context.classes.insert(object.to_string());
            }
            RDFS_SUBPROPERTY_OF => {
                debug!("Processing subPropertyOf: {} -> {}", subject, object);
                self.context.add_subproperty_relation(subject, object);
                self.context.properties.insert(subject.to_string());
                self.context.properties.insert(object.to_string());
            }
            RDFS_DOMAIN => {
                debug!("Processing domain: {} -> {}", subject, object);
                self.context.add_property_domain(subject, object);
                self.context.properties.insert(subject.to_string());
                self.context.classes.insert(object.to_string());
            }
            RDFS_RANGE => {
                debug!("Processing range: {} -> {}", subject, object);
                self.context.add_property_range(subject, object);
                self.context.properties.insert(subject.to_string());
                self.context.classes.insert(object.to_string());
            }
            RDF_TYPE => {
                debug!("Processing type: {} -> {}", subject, object);
                if object == RDFS_CLASS {
                    self.context.classes.insert(subject.to_string());
                } else if object == RDF_PROPERTY {
                    self.context.properties.insert(subject.to_string());
                }
            }
            _ => {
                trace!(
                    "Processing regular triple: {} {} {}",
                    subject,
                    predicate,
                    object
                );
            }
        }

        // Apply RDFS inference rules
        let input_fact = RuleAtom::Triple {
            subject: Term::Constant(subject.to_string()),
            predicate: Term::Constant(predicate.to_string()),
            object: Term::Constant(object.to_string()),
        };

        new_facts.push(input_fact);

        // Apply domain/range inference
        if let Some(domains) = self.context.property_domains.get(predicate) {
            for domain in domains {
                new_facts.push(RuleAtom::Triple {
                    subject: Term::Constant(subject.to_string()),
                    predicate: Term::Constant(RDF_TYPE.to_string()),
                    object: Term::Constant(domain.clone()),
                });
            }
        }

        if let Some(ranges) = self.context.property_ranges.get(predicate) {
            for range in ranges {
                new_facts.push(RuleAtom::Triple {
                    subject: Term::Constant(object.to_string()),
                    predicate: Term::Constant(RDF_TYPE.to_string()),
                    object: Term::Constant(range.clone()),
                });
            }
        }

        // Apply subproperty inference
        let superproperties = self.context.get_superproperties(predicate);
        for superproperty in superproperties {
            new_facts.push(RuleAtom::Triple {
                subject: Term::Constant(subject.to_string()),
                predicate: Term::Constant(superproperty),
                object: Term::Constant(object.to_string()),
            });
        }

        // Apply subclass inference for rdf:type triples
        if predicate == RDF_TYPE {
            let superclasses = self.context.get_superclasses(object);
            for superclass in superclasses {
                new_facts.push(RuleAtom::Triple {
                    subject: Term::Constant(subject.to_string()),
                    predicate: Term::Constant(RDF_TYPE.to_string()),
                    object: Term::Constant(superclass),
                });
            }
        }

        Ok(new_facts)
    }

    /// Perform complete RDFS inference on a set of facts
    pub fn infer(&mut self, facts: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        let mut all_facts = facts.to_vec();
        let mut new_facts_added = true;
        let mut iteration = 0;

        while new_facts_added {
            new_facts_added = false;
            iteration += 1;
            debug!("RDFS inference iteration {}", iteration);

            let current_facts = all_facts.clone();
            for fact in &current_facts {
                if let RuleAtom::Triple {
                    subject,
                    predicate,
                    object,
                } = fact
                {
                    if let (Term::Constant(s), Term::Constant(p), Term::Constant(o)) =
                        (subject, predicate, object)
                    {
                        let inferred = self.process_triple(s, p, o)?;
                        for new_fact in inferred {
                            if !all_facts.contains(&new_fact) {
                                all_facts.push(new_fact);
                                new_facts_added = true;
                            }
                        }
                    }
                }
            }

            // Prevent infinite loops
            if iteration > 100 {
                return Err(anyhow::anyhow!(
                    "RDFS inference did not converge after 100 iterations"
                ));
            }
        }

        info!(
            "RDFS inference completed after {} iterations, {} facts total",
            iteration,
            all_facts.len()
        );
        Ok(all_facts)
    }

    /// Check if a triple is entailed by RDFS semantics
    pub fn entails(&self, subject: &str, predicate: &str, object: &str) -> bool {
        use vocabulary::*;

        match predicate {
            RDF_TYPE => {
                // Check if subject is of type object through class hierarchy
                if self.context.classes.contains(object) {
                    return self.context.is_subclass_of(object, object); // Always true for defined classes
                }
                false
            }
            RDFS_SUBCLASS_OF => self.context.is_subclass_of(subject, object),
            RDFS_SUBPROPERTY_OF => self.context.is_subproperty_of(subject, object),
            _ => {
                // Check if triple follows from subproperty relations
                let superproperties = self.context.get_superproperties(predicate);
                superproperties.contains(predicate)
            }
        }
    }

    /// Get materialized RDFS schema information
    pub fn get_schema_info(&self) -> RdfsSchemaInfo {
        RdfsSchemaInfo {
            classes: self.context.classes.clone(),
            properties: self.context.properties.clone(),
            class_hierarchy: self.context.class_hierarchy.clone(),
            property_hierarchy: self.context.property_hierarchy.clone(),
            property_domains: self.context.property_domains.clone(),
            property_ranges: self.context.property_ranges.clone(),
        }
    }
}

/// RDFS schema information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfsSchemaInfo {
    pub classes: HashSet<String>,
    pub properties: HashSet<String>,
    pub class_hierarchy: HashMap<String, HashSet<String>>,
    pub property_hierarchy: HashMap<String, HashSet<String>>,
    pub property_domains: HashMap<String, HashSet<String>>,
    pub property_ranges: HashMap<String, HashSet<String>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rdfs_context_initialization() {
        let context = RdfsContext::default();
        assert!(context
            .classes
            .contains("http://www.w3.org/2000/01/rdf-schema#Class"));
        assert!(context
            .properties
            .contains("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"));
    }

    #[test]
    fn test_subclass_inference() {
        let mut context = RdfsContext::default();
        context.add_subclass_relation("A", "B");
        context.add_subclass_relation("B", "C");

        assert!(context.is_subclass_of("A", "B"));
        assert!(context.is_subclass_of("A", "C"));
        assert!(context.is_subclass_of("B", "C"));
        assert!(!context.is_subclass_of("C", "A"));
    }

    #[test]
    fn test_rdfs_reasoner() {
        let mut reasoner = RdfsReasoner::new();

        let facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("Person".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/2000/01/rdf-schema#subClassOf".to_string(),
                ),
                object: Term::Constant("Agent".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("john".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                object: Term::Constant("Person".to_string()),
            },
        ];

        let inferred = reasoner.infer(&facts).unwrap();

        // Should infer that john is of type Agent
        let expected = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            ),
            object: Term::Constant("Agent".to_string()),
        };

        assert!(inferred.contains(&expected));
    }
}
