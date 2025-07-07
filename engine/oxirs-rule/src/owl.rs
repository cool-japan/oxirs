//! OWL Reasoning Engine
//!
//! Implementation of OWL RL (Rule Language) profile reasoning.
//! Supports class expressions, property characteristics, and consistency checking.

use crate::{Rule, RuleAtom, RuleEngine, Term};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, trace, warn};

/// OWL vocabulary constants
pub mod vocabulary {
    // OWL Classes
    pub const OWL_CLASS: &str = "http://www.w3.org/2002/07/owl#Class";
    pub const OWL_THING: &str = "http://www.w3.org/2002/07/owl#Thing";
    pub const OWL_NOTHING: &str = "http://www.w3.org/2002/07/owl#Nothing";
    pub const OWL_ONTOLOGY: &str = "http://www.w3.org/2002/07/owl#Ontology";

    // OWL Properties
    pub const OWL_OBJECT_PROPERTY: &str = "http://www.w3.org/2002/07/owl#ObjectProperty";
    pub const OWL_DATATYPE_PROPERTY: &str = "http://www.w3.org/2002/07/owl#DatatypeProperty";
    pub const OWL_ANNOTATION_PROPERTY: &str = "http://www.w3.org/2002/07/owl#AnnotationProperty";
    pub const OWL_FUNCTIONAL_PROPERTY: &str = "http://www.w3.org/2002/07/owl#FunctionalProperty";
    pub const OWL_INVERSE_FUNCTIONAL_PROPERTY: &str =
        "http://www.w3.org/2002/07/owl#InverseFunctionalProperty";
    pub const OWL_TRANSITIVE_PROPERTY: &str = "http://www.w3.org/2002/07/owl#TransitiveProperty";
    pub const OWL_SYMMETRIC_PROPERTY: &str = "http://www.w3.org/2002/07/owl#SymmetricProperty";
    pub const OWL_ASYMMETRIC_PROPERTY: &str = "http://www.w3.org/2002/07/owl#AsymmetricProperty";
    pub const OWL_REFLEXIVE_PROPERTY: &str = "http://www.w3.org/2002/07/owl#ReflexiveProperty";
    pub const OWL_IRREFLEXIVE_PROPERTY: &str = "http://www.w3.org/2002/07/owl#IrreflexiveProperty";

    // OWL Relations
    pub const OWL_EQUIVALENT_CLASS: &str = "http://www.w3.org/2002/07/owl#equivalentClass";
    pub const OWL_EQUIVALENT_PROPERTY: &str = "http://www.w3.org/2002/07/owl#equivalentProperty";
    pub const OWL_DISJOINT_WITH: &str = "http://www.w3.org/2002/07/owl#disjointWith";
    pub const OWL_INVERSE_OF: &str = "http://www.w3.org/2002/07/owl#inverseOf";
    pub const OWL_SAME_AS: &str = "http://www.w3.org/2002/07/owl#sameAs";
    pub const OWL_DIFFERENT_FROM: &str = "http://www.w3.org/2002/07/owl#differentFrom";

    // Class Expressions
    pub const OWL_INTERSECTION_OF: &str = "http://www.w3.org/2002/07/owl#intersectionOf";
    pub const OWL_UNION_OF: &str = "http://www.w3.org/2002/07/owl#unionOf";
    pub const OWL_COMPLEMENT_OF: &str = "http://www.w3.org/2002/07/owl#complementOf";
    pub const OWL_ONE_OF: &str = "http://www.w3.org/2002/07/owl#oneOf";

    // Property Restrictions
    pub const OWL_RESTRICTION: &str = "http://www.w3.org/2002/07/owl#Restriction";
    pub const OWL_ON_PROPERTY: &str = "http://www.w3.org/2002/07/owl#onProperty";
    pub const OWL_ALL_VALUES_FROM: &str = "http://www.w3.org/2002/07/owl#allValuesFrom";
    pub const OWL_SOME_VALUES_FROM: &str = "http://www.w3.org/2002/07/owl#someValuesFrom";
    pub const OWL_HAS_VALUE: &str = "http://www.w3.org/2002/07/owl#hasValue";
    pub const OWL_MIN_CARDINALITY: &str = "http://www.w3.org/2002/07/owl#minCardinality";
    pub const OWL_MAX_CARDINALITY: &str = "http://www.w3.org/2002/07/owl#maxCardinality";
    pub const OWL_CARDINALITY: &str = "http://www.w3.org/2002/07/owl#cardinality";

    // RDF/RDFS used in OWL
    pub const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    pub const RDFS_SUBCLASS_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";
    pub const RDFS_SUBPROPERTY_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subPropertyOf";
}

/// OWL class expression types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClassExpression {
    /// Named class
    Class(String),
    /// Intersection of classes
    Intersection(Vec<ClassExpression>),
    /// Union of classes
    Union(Vec<ClassExpression>),
    /// Complement of a class
    Complement(Box<ClassExpression>),
    /// Enumeration of individuals
    OneOf(Vec<String>),
    /// Property restriction
    Restriction {
        property: String,
        constraint: Box<RestrictionType>,
    },
}

/// Types of property restrictions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RestrictionType {
    /// Universal restriction (all values from)
    AllValuesFrom(Box<ClassExpression>),
    /// Existential restriction (some values from)
    SomeValuesFrom(Box<ClassExpression>),
    /// Value restriction (has value)
    HasValue(String),
    /// Cardinality restrictions
    MinCardinality(u32),
    MaxCardinality(u32),
    ExactCardinality(u32),
}

/// Property characteristics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct PropertyCharacteristics {
    pub is_functional: bool,
    pub is_inverse_functional: bool,
    pub is_transitive: bool,
    pub is_symmetric: bool,
    pub is_asymmetric: bool,
    pub is_reflexive: bool,
    pub is_irreflexive: bool,
}


/// OWL knowledge base context
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OwlContext {
    /// Class equivalences
    pub equivalent_classes: HashMap<String, HashSet<String>>,
    /// Property equivalences
    pub equivalent_properties: HashMap<String, HashSet<String>>,
    /// Class disjointness
    pub disjoint_classes: HashMap<String, HashSet<String>>,
    /// Property inverses
    pub inverse_properties: HashMap<String, String>,
    /// Individual equivalences (sameAs)
    pub same_individuals: HashMap<String, HashSet<String>>,
    /// Individual differences (differentFrom)
    pub different_individuals: HashMap<String, HashSet<String>>,
    /// Property characteristics
    pub property_characteristics: HashMap<String, PropertyCharacteristics>,
    /// Class expressions
    pub class_expressions: HashMap<String, ClassExpression>,
    /// Known inconsistencies
    pub inconsistencies: Vec<String>,
}


impl OwlContext {
    /// Add class equivalence
    pub fn add_equivalent_classes(&mut self, class1: &str, class2: &str) {
        self.equivalent_classes
            .entry(class1.to_string())
            .or_default()
            .insert(class2.to_string());
        self.equivalent_classes
            .entry(class2.to_string())
            .or_default()
            .insert(class1.to_string());
    }

    /// Add property equivalence
    pub fn add_equivalent_properties(&mut self, prop1: &str, prop2: &str) {
        self.equivalent_properties
            .entry(prop1.to_string())
            .or_default()
            .insert(prop2.to_string());
        self.equivalent_properties
            .entry(prop2.to_string())
            .or_default()
            .insert(prop1.to_string());
    }

    /// Add class disjointness
    pub fn add_disjoint_classes(&mut self, class1: &str, class2: &str) {
        self.disjoint_classes
            .entry(class1.to_string())
            .or_default()
            .insert(class2.to_string());
        self.disjoint_classes
            .entry(class2.to_string())
            .or_default()
            .insert(class1.to_string());
    }

    /// Add property inverse
    pub fn add_inverse_properties(&mut self, prop1: &str, prop2: &str) {
        self.inverse_properties
            .insert(prop1.to_string(), prop2.to_string());
        self.inverse_properties
            .insert(prop2.to_string(), prop1.to_string());
    }

    /// Add individual equivalence
    pub fn add_same_individuals(&mut self, ind1: &str, ind2: &str) {
        self.same_individuals
            .entry(ind1.to_string())
            .or_default()
            .insert(ind2.to_string());
        self.same_individuals
            .entry(ind2.to_string())
            .or_default()
            .insert(ind1.to_string());
    }

    /// Add individual difference
    pub fn add_different_individuals(&mut self, ind1: &str, ind2: &str) {
        self.different_individuals
            .entry(ind1.to_string())
            .or_default()
            .insert(ind2.to_string());
        self.different_individuals
            .entry(ind2.to_string())
            .or_default()
            .insert(ind1.to_string());
    }

    /// Set property characteristic
    pub fn set_property_characteristic(
        &mut self,
        property: &str,
        characteristic: &str,
        value: bool,
    ) {
        let chars = self
            .property_characteristics
            .entry(property.to_string())
            .or_default();

        match characteristic {
            vocabulary::OWL_FUNCTIONAL_PROPERTY => chars.is_functional = value,
            vocabulary::OWL_INVERSE_FUNCTIONAL_PROPERTY => chars.is_inverse_functional = value,
            vocabulary::OWL_TRANSITIVE_PROPERTY => chars.is_transitive = value,
            vocabulary::OWL_SYMMETRIC_PROPERTY => chars.is_symmetric = value,
            vocabulary::OWL_ASYMMETRIC_PROPERTY => chars.is_asymmetric = value,
            vocabulary::OWL_REFLEXIVE_PROPERTY => chars.is_reflexive = value,
            vocabulary::OWL_IRREFLEXIVE_PROPERTY => chars.is_irreflexive = value,
            _ => {}
        }
    }

    /// Check for inconsistencies
    pub fn check_consistency(&mut self) -> bool {
        self.inconsistencies.clear();

        // Check for same and different individuals
        for (ind1, same_set) in &self.same_individuals {
            if let Some(diff_set) = self.different_individuals.get(ind1) {
                for ind2 in same_set {
                    if diff_set.contains(ind2) {
                        self.inconsistencies.push(format!(
                            "Individual {ind1} is both same as and different from {ind2}"
                        ));
                    }
                }
            }
        }

        // Check for disjoint and equivalent classes
        for (class1, equiv_set) in &self.equivalent_classes {
            if let Some(disjoint_set) = self.disjoint_classes.get(class1) {
                for class2 in equiv_set {
                    if disjoint_set.contains(class2) {
                        self.inconsistencies.push(format!(
                            "Class {class1} is both equivalent to and disjoint with {class2}"
                        ));
                    }
                }
            }
        }

        self.inconsistencies.is_empty()
    }
}

/// OWL RL reasoner
#[derive(Debug)]
pub struct OwlReasoner {
    /// OWL context
    pub context: OwlContext,
    /// Rule engine for OWL RL rules
    pub rule_engine: RuleEngine,
}

impl Default for OwlReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl OwlReasoner {
    /// Create a new OWL reasoner
    pub fn new() -> Self {
        let mut reasoner = Self {
            context: OwlContext::default(),
            rule_engine: RuleEngine::new(),
        };

        reasoner.initialize_owl_rl_rules();
        reasoner
    }

    /// Initialize OWL RL entailment rules
    fn initialize_owl_rl_rules(&mut self) {
        // Equivalence rules
        self.add_equivalence_rules();

        // Property characteristic rules
        self.add_property_characteristic_rules();

        // Disjointness rules
        self.add_disjointness_rules();

        // Individual identity rules
        self.add_identity_rules();

        info!(
            "Initialized {} OWL RL entailment rules",
            self.rule_engine.rules.len()
        );
    }

    /// Add class and property equivalence rules
    fn add_equivalence_rules(&mut self) {
        use vocabulary::*;

        // Class equivalence symmetry: C1 equivalentClass C2 => C2 equivalentClass C1
        self.rule_engine.add_rule(Rule {
            name: "owl_equiv_class_sym".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("C1".to_string()),
                predicate: Term::Constant(OWL_EQUIVALENT_CLASS.to_string()),
                object: Term::Variable("C2".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("C2".to_string()),
                predicate: Term::Constant(OWL_EQUIVALENT_CLASS.to_string()),
                object: Term::Variable("C1".to_string()),
            }],
        });

        // Class equivalence transitivity
        self.rule_engine.add_rule(Rule {
            name: "owl_equiv_class_trans".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("C1".to_string()),
                    predicate: Term::Constant(OWL_EQUIVALENT_CLASS.to_string()),
                    object: Term::Variable("C2".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("C2".to_string()),
                    predicate: Term::Constant(OWL_EQUIVALENT_CLASS.to_string()),
                    object: Term::Variable("C3".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("C1".to_string()),
                predicate: Term::Constant(OWL_EQUIVALENT_CLASS.to_string()),
                object: Term::Variable("C3".to_string()),
            }],
        });

        // Equivalent classes have same instances
        self.rule_engine.add_rule(Rule {
            name: "owl_equiv_class_inst".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant(RDF_TYPE.to_string()),
                    object: Term::Variable("C1".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("C1".to_string()),
                    predicate: Term::Constant(OWL_EQUIVALENT_CLASS.to_string()),
                    object: Term::Variable("C2".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(RDF_TYPE.to_string()),
                object: Term::Variable("C2".to_string()),
            }],
        });

        // Property equivalence rules (similar structure)
        self.rule_engine.add_rule(Rule {
            name: "owl_equiv_prop_sym".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("P1".to_string()),
                predicate: Term::Constant(OWL_EQUIVALENT_PROPERTY.to_string()),
                object: Term::Variable("P2".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("P2".to_string()),
                predicate: Term::Constant(OWL_EQUIVALENT_PROPERTY.to_string()),
                object: Term::Variable("P1".to_string()),
            }],
        });
    }

    /// Add property characteristic rules
    fn add_property_characteristic_rules(&mut self) {
        use vocabulary::*;

        // Functional property rule
        self.rule_engine.add_rule(Rule {
            name: "owl_functional".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("P".to_string()),
                    predicate: Term::Constant(RDF_TYPE.to_string()),
                    object: Term::Constant(OWL_FUNCTIONAL_PROPERTY.to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Variable("P".to_string()),
                    object: Term::Variable("Y1".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Variable("P".to_string()),
                    object: Term::Variable("Y2".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Y1".to_string()),
                predicate: Term::Constant(OWL_SAME_AS.to_string()),
                object: Term::Variable("Y2".to_string()),
            }],
        });

        // Transitive property rule
        self.rule_engine.add_rule(Rule {
            name: "owl_transitive".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("P".to_string()),
                    predicate: Term::Constant(RDF_TYPE.to_string()),
                    object: Term::Constant(OWL_TRANSITIVE_PROPERTY.to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Variable("P".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Variable("P".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Variable("P".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        });

        // Symmetric property rule
        self.rule_engine.add_rule(Rule {
            name: "owl_symmetric".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("P".to_string()),
                    predicate: Term::Constant(RDF_TYPE.to_string()),
                    object: Term::Constant(OWL_SYMMETRIC_PROPERTY.to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Variable("P".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Y".to_string()),
                predicate: Term::Variable("P".to_string()),
                object: Term::Variable("X".to_string()),
            }],
        });

        // Inverse property rule
        self.rule_engine.add_rule(Rule {
            name: "owl_inverse".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("P1".to_string()),
                    predicate: Term::Constant(OWL_INVERSE_OF.to_string()),
                    object: Term::Variable("P2".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Variable("P1".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Y".to_string()),
                predicate: Term::Variable("P2".to_string()),
                object: Term::Variable("X".to_string()),
            }],
        });
    }

    /// Add disjointness rules
    fn add_disjointness_rules(&mut self) {
        use vocabulary::*;

        // Disjoint classes cannot have common instances
        // This would typically generate inconsistency warnings rather than new facts
        self.rule_engine.add_rule(Rule {
            name: "owl_disjoint_check".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("C1".to_string()),
                    predicate: Term::Constant(OWL_DISJOINT_WITH.to_string()),
                    object: Term::Variable("C2".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant(RDF_TYPE.to_string()),
                    object: Term::Variable("C1".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant(RDF_TYPE.to_string()),
                    object: Term::Variable("C2".to_string()),
                },
            ],
            head: vec![
                // This would typically signal an inconsistency
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("http://oxirs.org/inconsistent".to_string()),
                    object: Term::Constant("true".to_string()),
                },
            ],
        });
    }

    /// Add individual identity rules
    fn add_identity_rules(&mut self) {
        use vocabulary::*;

        // sameAs symmetry
        self.rule_engine.add_rule(Rule {
            name: "owl_same_sym".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(OWL_SAME_AS.to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Y".to_string()),
                predicate: Term::Constant(OWL_SAME_AS.to_string()),
                object: Term::Variable("X".to_string()),
            }],
        });

        // sameAs transitivity
        self.rule_engine.add_rule(Rule {
            name: "owl_same_trans".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant(OWL_SAME_AS.to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant(OWL_SAME_AS.to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(OWL_SAME_AS.to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        });

        // Same individuals have same properties
        self.rule_engine.add_rule(Rule {
            name: "owl_same_prop".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant(OWL_SAME_AS.to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Variable("P".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Y".to_string()),
                predicate: Term::Variable("P".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        });
    }

    /// Process a triple and update the OWL context
    pub fn process_triple(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> Result<Vec<RuleAtom>> {
        use vocabulary::*;

        let mut new_facts = Vec::new();

        match predicate {
            OWL_EQUIVALENT_CLASS => {
                debug!("Processing equivalentClass: {} ≡ {}", subject, object);
                self.context.add_equivalent_classes(subject, object);
            }
            OWL_EQUIVALENT_PROPERTY => {
                debug!("Processing equivalentProperty: {} ≡ {}", subject, object);
                self.context.add_equivalent_properties(subject, object);
            }
            OWL_DISJOINT_WITH => {
                debug!("Processing disjointWith: {} ⊥ {}", subject, object);
                self.context.add_disjoint_classes(subject, object);
            }
            OWL_INVERSE_OF => {
                debug!("Processing inverseOf: {} ⁻¹ {}", subject, object);
                self.context.add_inverse_properties(subject, object);
            }
            OWL_SAME_AS => {
                debug!("Processing sameAs: {} = {}", subject, object);
                self.context.add_same_individuals(subject, object);
            }
            OWL_DIFFERENT_FROM => {
                debug!("Processing differentFrom: {} ≠ {}", subject, object);
                self.context.add_different_individuals(subject, object);
            }
            RDF_TYPE => {
                // Handle property characteristics
                match object {
                    OWL_FUNCTIONAL_PROPERTY
                    | OWL_INVERSE_FUNCTIONAL_PROPERTY
                    | OWL_TRANSITIVE_PROPERTY
                    | OWL_SYMMETRIC_PROPERTY
                    | OWL_ASYMMETRIC_PROPERTY
                    | OWL_REFLEXIVE_PROPERTY
                    | OWL_IRREFLEXIVE_PROPERTY => {
                        debug!(
                            "Processing property characteristic: {} rdf:type {}",
                            subject, object
                        );
                        self.context
                            .set_property_characteristic(subject, object, true);
                    }
                    _ => {
                        trace!("Processing regular type: {} rdf:type {}", subject, object);
                    }
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

        // Apply basic OWL RL inference
        let input_fact = RuleAtom::Triple {
            subject: Term::Constant(subject.to_string()),
            predicate: Term::Constant(predicate.to_string()),
            object: Term::Constant(object.to_string()),
        };

        new_facts.push(input_fact);

        // Apply property characteristic inferences
        if let Some(characteristics) = self.context.property_characteristics.get(predicate) {
            if characteristics.is_symmetric {
                new_facts.push(RuleAtom::Triple {
                    subject: Term::Constant(object.to_string()),
                    predicate: Term::Constant(predicate.to_string()),
                    object: Term::Constant(subject.to_string()),
                });
            }
        }

        // Apply equivalence inferences
        if let Some(equiv_classes) = self.context.equivalent_classes.get(object) {
            for equiv_class in equiv_classes {
                new_facts.push(RuleAtom::Triple {
                    subject: Term::Constant(subject.to_string()),
                    predicate: Term::Constant(predicate.to_string()),
                    object: Term::Constant(equiv_class.clone()),
                });
            }
        }

        Ok(new_facts)
    }

    /// Perform complete OWL RL inference
    pub fn infer(&mut self, facts: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        let mut all_facts = facts.to_vec();
        let mut new_facts_added = true;
        let mut iteration = 0;

        while new_facts_added {
            new_facts_added = false;
            iteration += 1;
            debug!("OWL RL inference iteration {}", iteration);

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

            // Apply rule engine
            let rule_inferred = self.rule_engine.forward_chain(&all_facts)?;
            for new_fact in rule_inferred {
                if !all_facts.contains(&new_fact) {
                    all_facts.push(new_fact);
                    new_facts_added = true;
                }
            }

            // Prevent infinite loops
            if iteration > 100 {
                return Err(anyhow::anyhow!(
                    "OWL RL inference did not converge after 100 iterations"
                ));
            }
        }

        // Check consistency
        if !self.context.check_consistency() {
            warn!(
                "Inconsistencies detected: {:?}",
                self.context.inconsistencies
            );
        }

        info!(
            "OWL RL inference completed after {} iterations, {} facts total",
            iteration,
            all_facts.len()
        );
        Ok(all_facts)
    }

    /// Check if the knowledge base is consistent
    pub fn is_consistent(&mut self) -> bool {
        self.context.check_consistency()
    }

    /// Get detected inconsistencies
    pub fn get_inconsistencies(&self) -> &[String] {
        &self.context.inconsistencies
    }

    /// Get materialized OWL information
    pub fn get_owl_info(&self) -> OwlInfo {
        OwlInfo {
            equivalent_classes: self.context.equivalent_classes.clone(),
            equivalent_properties: self.context.equivalent_properties.clone(),
            disjoint_classes: self.context.disjoint_classes.clone(),
            inverse_properties: self.context.inverse_properties.clone(),
            same_individuals: self.context.same_individuals.clone(),
            different_individuals: self.context.different_individuals.clone(),
            property_characteristics: self.context.property_characteristics.clone(),
            inconsistencies: self.context.inconsistencies.clone(),
        }
    }
}

/// OWL knowledge base information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwlInfo {
    pub equivalent_classes: HashMap<String, HashSet<String>>,
    pub equivalent_properties: HashMap<String, HashSet<String>>,
    pub disjoint_classes: HashMap<String, HashSet<String>>,
    pub inverse_properties: HashMap<String, String>,
    pub same_individuals: HashMap<String, HashSet<String>>,
    pub different_individuals: HashMap<String, HashSet<String>>,
    pub property_characteristics: HashMap<String, PropertyCharacteristics>,
    pub inconsistencies: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_owl_context_equivalence() {
        let mut context = OwlContext::default();
        context.add_equivalent_classes("Person", "Human");

        assert!(context
            .equivalent_classes
            .get("Person")
            .unwrap()
            .contains("Human"));
        assert!(context
            .equivalent_classes
            .get("Human")
            .unwrap()
            .contains("Person"));
    }

    #[test]
    fn test_property_characteristics() {
        let mut context = OwlContext::default();
        context.set_property_characteristic("parentOf", vocabulary::OWL_TRANSITIVE_PROPERTY, true);

        let chars = context.property_characteristics.get("parentOf").unwrap();
        assert!(chars.is_transitive);
        assert!(!chars.is_symmetric);
    }

    #[test]
    fn test_consistency_checking() {
        let mut context = OwlContext::default();
        context.add_same_individuals("john", "johndoe");
        context.add_different_individuals("john", "johndoe");

        assert!(!context.check_consistency());
        assert!(!context.inconsistencies.is_empty());
    }

    #[test]
    fn test_owl_reasoner() {
        let mut reasoner = OwlReasoner::new();

        let facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("Person".to_string()),
                predicate: Term::Constant(vocabulary::OWL_EQUIVALENT_CLASS.to_string()),
                object: Term::Constant("Human".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("john".to_string()),
                predicate: Term::Constant(vocabulary::RDF_TYPE.to_string()),
                object: Term::Constant("Person".to_string()),
            },
        ];

        let inferred = reasoner.infer(&facts).unwrap();

        // Should infer that john is also of type Human
        let expected = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant(vocabulary::RDF_TYPE.to_string()),
            object: Term::Constant("Human".to_string()),
        };

        assert!(inferred.contains(&expected));
    }

    #[test]
    fn test_transitive_property() {
        let mut reasoner = OwlReasoner::new();

        let facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("ancestorOf".to_string()),
                predicate: Term::Constant(vocabulary::RDF_TYPE.to_string()),
                object: Term::Constant(vocabulary::OWL_TRANSITIVE_PROPERTY.to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("john".to_string()),
                predicate: Term::Constant("ancestorOf".to_string()),
                object: Term::Constant("mary".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("mary".to_string()),
                predicate: Term::Constant("ancestorOf".to_string()),
                object: Term::Constant("bob".to_string()),
            },
        ];

        let inferred = reasoner.infer(&facts).unwrap();

        // Should infer transitive relationship
        let expected = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("ancestorOf".to_string()),
            object: Term::Constant("bob".to_string()),
        };

        assert!(inferred.contains(&expected));
    }
}
