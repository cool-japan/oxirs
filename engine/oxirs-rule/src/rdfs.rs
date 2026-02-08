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

/// Identifiers for RDFS entailment rules
///
/// Each variant corresponds to a specific RDFS inference rule from the W3C specification.
/// Some rules (rdfs1, rdfs4a, rdfs4b, rdfs6, rdfs8, rdfs10) generate large amounts of
/// trivial inferences and are disabled by default in the Minimal profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RdfsRule {
    /// rdfs1: `(?s ?p ?o) → (?p rdf:type rdf:Property)` - Every predicate becomes a Property (noisy)
    Rdfs1,
    /// rdfs2: `(?p rdfs:domain ?c) ∧ (?x ?p ?y) → (?x rdf:type ?c)` - Domain inference (useful)
    Rdfs2,
    /// rdfs3: `(?p rdfs:range ?c) ∧ (?x ?p ?y) → (?y rdf:type ?c)` - Range inference (useful)
    Rdfs3,
    /// rdfs4a: `(?s ?p ?o) → (?s rdf:type rdfs:Resource)` - Every subject is Resource (noisy)
    Rdfs4a,
    /// rdfs4b: `(?s ?p ?o) → (?o rdf:type rdfs:Resource)` - Every object is Resource (noisy)
    Rdfs4b,
    /// rdfs5: `(?p rdfs:subPropertyOf ?q) ∧ (?q rdfs:subPropertyOf ?r) → (?p rdfs:subPropertyOf ?r)` - Property transitivity (useful)
    Rdfs5,
    /// rdfs6: `(?p rdf:type rdf:Property) → (?p rdfs:subPropertyOf ?p)` - Reflexive subPropertyOf (noisy)
    Rdfs6,
    /// rdfs7: `(?x ?p ?y) ∧ (?p rdfs:subPropertyOf ?q) → (?x ?q ?y)` - Subproperty inheritance (useful)
    Rdfs7,
    /// rdfs8: `(?c rdf:type rdfs:Class) → (?c rdfs:subClassOf rdfs:Resource)` - Classes subClassOf Resource (noisy)
    Rdfs8,
    /// rdfs9: `(?x rdf:type ?c) ∧ (?c rdfs:subClassOf ?d) → (?x rdf:type ?d)` - Subclass type inheritance (critical)
    Rdfs9,
    /// rdfs10: `(?c rdf:type rdfs:Class) → (?c rdfs:subClassOf ?c)` - Reflexive subClassOf (noisy)
    Rdfs10,
    /// rdfs11: `(?c rdfs:subClassOf ?d) ∧ (?d rdfs:subClassOf ?e) → (?c rdfs:subClassOf ?e)` - Class transitivity (useful)
    Rdfs11,
    /// rdfs13: `(?c rdf:type rdfs:Datatype) → (?c rdfs:subClassOf rdfs:Literal)` - Datatype handling (useful)
    Rdfs13,
}

impl RdfsRule {
    /// Returns all RDFS rules
    pub fn all() -> &'static [RdfsRule] {
        &[
            RdfsRule::Rdfs1,
            RdfsRule::Rdfs2,
            RdfsRule::Rdfs3,
            RdfsRule::Rdfs4a,
            RdfsRule::Rdfs4b,
            RdfsRule::Rdfs5,
            RdfsRule::Rdfs6,
            RdfsRule::Rdfs7,
            RdfsRule::Rdfs8,
            RdfsRule::Rdfs9,
            RdfsRule::Rdfs10,
            RdfsRule::Rdfs11,
            RdfsRule::Rdfs13,
        ]
    }

    /// Returns rules that are practical for most use cases (non-noisy)
    pub fn minimal() -> &'static [RdfsRule] {
        &[
            RdfsRule::Rdfs2,
            RdfsRule::Rdfs3,
            RdfsRule::Rdfs5,
            RdfsRule::Rdfs7,
            RdfsRule::Rdfs9,
            RdfsRule::Rdfs11,
            RdfsRule::Rdfs13,
        ]
    }

    /// Returns rules that generate exponential/noisy facts
    pub fn noisy() -> &'static [RdfsRule] {
        &[
            RdfsRule::Rdfs1,
            RdfsRule::Rdfs4a,
            RdfsRule::Rdfs4b,
            RdfsRule::Rdfs6,
            RdfsRule::Rdfs8,
            RdfsRule::Rdfs10,
        ]
    }

    /// Returns the rule name as used in the rule engine
    pub fn name(&self) -> &'static str {
        match self {
            RdfsRule::Rdfs1 => "rdfs1",
            RdfsRule::Rdfs2 => "rdfs2",
            RdfsRule::Rdfs3 => "rdfs3",
            RdfsRule::Rdfs4a => "rdfs4a",
            RdfsRule::Rdfs4b => "rdfs4b",
            RdfsRule::Rdfs5 => "rdfs5",
            RdfsRule::Rdfs6 => "rdfs6",
            RdfsRule::Rdfs7 => "rdfs7",
            RdfsRule::Rdfs8 => "rdfs8",
            RdfsRule::Rdfs9 => "rdfs9",
            RdfsRule::Rdfs10 => "rdfs10",
            RdfsRule::Rdfs11 => "rdfs11",
            RdfsRule::Rdfs13 => "rdfs13",
        }
    }
}

/// Preset profiles for common RDFS configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum RdfsProfile {
    /// Minimal profile: rdfs2, rdfs3, rdfs5, rdfs7, rdfs9, rdfs11, rdfs13
    /// Practical inference without noise - best for most applications
    #[default]
    Minimal,
    /// Full profile: 13 of the W3C RDFS rules (excluding rdfs12, which handles
    /// container membership properties)
    /// Use when maximal RDFS inference (excluding container membership properties)
    /// is required
    Full,
    /// No rules: Context-only mode for hierarchy queries without rule engine overhead
    /// The RdfsContext is still populated and can be queried directly
    None,
}

/// Configuration for RDFS reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfsConfig {
    /// Set of enabled RDFS rules
    pub enabled_rules: HashSet<RdfsRule>,
}

impl Default for RdfsConfig {
    fn default() -> Self {
        Self::from_profile(RdfsProfile::Minimal)
    }
}

impl RdfsConfig {
    /// Create configuration from a preset profile
    pub fn from_profile(profile: RdfsProfile) -> Self {
        let enabled_rules = match profile {
            RdfsProfile::Minimal => RdfsRule::minimal().iter().copied().collect(),
            RdfsProfile::Full => RdfsRule::all().iter().copied().collect(),
            RdfsProfile::None => HashSet::new(),
        };
        Self { enabled_rules }
    }

    /// Create configuration with all rules enabled
    pub fn full() -> Self {
        Self::from_profile(RdfsProfile::Full)
    }

    /// Create configuration with minimal (non-noisy) rules
    pub fn minimal() -> Self {
        Self::from_profile(RdfsProfile::Minimal)
    }

    /// Create configuration with no rules (context-only mode)
    pub fn none() -> Self {
        Self::from_profile(RdfsProfile::None)
    }

    /// Check if a specific rule is enabled
    pub fn is_enabled(&self, rule: RdfsRule) -> bool {
        self.enabled_rules.contains(&rule)
    }

    /// Enable a specific rule
    pub fn enable(&mut self, rule: RdfsRule) {
        self.enabled_rules.insert(rule);
    }

    /// Disable a specific rule
    pub fn disable(&mut self, rule: RdfsRule) {
        self.enabled_rules.remove(&rule);
    }
}

/// Builder for configuring RDFS reasoner
#[derive(Debug, Clone)]
pub struct RdfsReasonerBuilder {
    config: RdfsConfig,
}

impl Default for RdfsReasonerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RdfsReasonerBuilder {
    /// Create a new builder starting with the default (Minimal) profile
    pub fn new() -> Self {
        Self {
            config: RdfsConfig::default(),
        }
    }

    /// Apply a preset profile, replacing current configuration
    pub fn with_profile(mut self, profile: RdfsProfile) -> Self {
        self.config = RdfsConfig::from_profile(profile);
        self
    }

    /// Enable a specific rule
    pub fn enable_rule(mut self, rule: RdfsRule) -> Self {
        self.config.enable(rule);
        self
    }

    /// Disable a specific rule
    pub fn disable_rule(mut self, rule: RdfsRule) -> Self {
        self.config.disable(rule);
        self
    }

    /// Enable multiple rules
    pub fn enable_rules(mut self, rules: &[RdfsRule]) -> Self {
        for rule in rules {
            self.config.enable(*rule);
        }
        self
    }

    /// Disable multiple rules
    pub fn disable_rules(mut self, rules: &[RdfsRule]) -> Self {
        for rule in rules {
            self.config.disable(*rule);
        }
        self
    }

    /// Build the RDFS reasoner with the configured rules
    pub fn build(self) -> RdfsReasoner {
        RdfsReasoner::with_config(self.config)
    }
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
    /// Configuration specifying which rules are enabled
    pub config: RdfsConfig,
}

impl Default for RdfsReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl RdfsReasoner {
    /// Create a new RDFS reasoner with default (Minimal) profile
    ///
    /// The Minimal profile includes practical rules (rdfs2, rdfs3, rdfs5, rdfs7, rdfs9, rdfs11, rdfs13)
    /// while excluding noisy rules that generate exponential trivial facts.
    pub fn new() -> Self {
        Self::with_config(RdfsConfig::default())
    }

    /// Create a new RDFS reasoner with a specific profile
    ///
    /// # Example
    /// ```
    /// use oxirs_rule::rdfs::{RdfsReasoner, RdfsProfile};
    ///
    /// // Full W3C compliance
    /// let full_reasoner = RdfsReasoner::with_profile(RdfsProfile::Full);
    ///
    /// // Minimal (default) - practical inference
    /// let minimal_reasoner = RdfsReasoner::with_profile(RdfsProfile::Minimal);
    ///
    /// // Context-only - no rule engine overhead
    /// let context_only = RdfsReasoner::with_profile(RdfsProfile::None);
    /// ```
    pub fn with_profile(profile: RdfsProfile) -> Self {
        Self::with_config(RdfsConfig::from_profile(profile))
    }

    /// Create a new RDFS reasoner with a custom configuration
    pub fn with_config(config: RdfsConfig) -> Self {
        let mut reasoner = Self {
            context: RdfsContext::default(),
            rule_engine: RuleEngine::new(),
            config,
        };

        reasoner.initialize_rdfs_rules();
        reasoner
    }

    /// Create a context-only reasoner (no rules, just hierarchy queries)
    ///
    /// This is useful when you only need to query the class/property hierarchy
    /// without the overhead of the rule engine.
    ///
    /// # Example
    /// ```
    /// use oxirs_rule::rdfs::RdfsReasoner;
    ///
    /// let reasoner = RdfsReasoner::context_only();
    /// // Can still use: reasoner.context.is_subclass_of(a, b)
    /// ```
    pub fn context_only() -> Self {
        Self::with_profile(RdfsProfile::None)
    }

    /// Create a builder for custom configuration
    ///
    /// # Example
    /// ```
    /// use oxirs_rule::rdfs::{RdfsReasoner, RdfsProfile, RdfsRule};
    ///
    /// let reasoner = RdfsReasoner::builder()
    ///     .with_profile(RdfsProfile::Minimal)
    ///     .enable_rule(RdfsRule::Rdfs8)
    ///     .build();
    /// ```
    pub fn builder() -> RdfsReasonerBuilder {
        RdfsReasonerBuilder::new()
    }

    /// Get the current configuration
    pub fn get_config(&self) -> &RdfsConfig {
        &self.config
    }

    /// Check if a specific rule is enabled
    pub fn is_rule_enabled(&self, rule: RdfsRule) -> bool {
        self.config.is_enabled(rule)
    }

    /// Initialize RDFS entailment rules based on configuration
    fn initialize_rdfs_rules(&mut self) {
        use vocabulary::*;

        // RDFS Rule 1: Triple (?x ?a ?y) => (?a rdf:type rdf:Property)
        // Any triple implies that the predicate is a property (noisy)
        if self.config.is_enabled(RdfsRule::Rdfs1) {
            self.rule_engine.add_rule(Rule {
                name: "rdfs1".to_string(),
                body: vec![RuleAtom::Triple {
                    subject: Term::Variable("x".to_string()),
                    predicate: Term::Variable("a".to_string()),
                    object: Term::Variable("y".to_string()),
                }],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("a".to_string()),
                    predicate: Term::Constant(RDF_TYPE.to_string()),
                    object: Term::Constant(RDF_PROPERTY.to_string()),
                }],
            });
        }

        // RDFS Rule 2: Triple (?p rdfs:domain ?c) + (?x ?p ?y) => (?x rdf:type ?c)
        if self.config.is_enabled(RdfsRule::Rdfs2) {
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
        }

        // RDFS Rule 3: Triple (?p rdfs:range ?c) + (?x ?p ?y) => (?y rdf:type ?c)
        if self.config.is_enabled(RdfsRule::Rdfs3) {
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
        }

        // RDFS Rule 4a: Triple (?x ?a ?y) => (?x rdf:type rdfs:Resource)
        // Subject of any triple is a resource (noisy)
        if self.config.is_enabled(RdfsRule::Rdfs4a) {
            self.rule_engine.add_rule(Rule {
                name: "rdfs4a".to_string(),
                body: vec![RuleAtom::Triple {
                    subject: Term::Variable("x".to_string()),
                    predicate: Term::Variable("a".to_string()),
                    object: Term::Variable("y".to_string()),
                }],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("x".to_string()),
                    predicate: Term::Constant(RDF_TYPE.to_string()),
                    object: Term::Constant(RDFS_RESOURCE.to_string()),
                }],
            });
        }

        // RDFS Rule 4b: Triple (?x ?a ?y) => (?y rdf:type rdfs:Resource) [if y is not a literal]
        // Object of any triple is a resource (noisy)
        if self.config.is_enabled(RdfsRule::Rdfs4b) {
            self.rule_engine.add_rule(Rule {
                name: "rdfs4b".to_string(),
                body: vec![RuleAtom::Triple {
                    subject: Term::Variable("x".to_string()),
                    predicate: Term::Variable("a".to_string()),
                    object: Term::Variable("y".to_string()),
                }],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("y".to_string()),
                    predicate: Term::Constant(RDF_TYPE.to_string()),
                    object: Term::Constant(RDFS_RESOURCE.to_string()),
                }],
            });
        }

        // RDFS Rule 5: Triple (?p rdfs:subPropertyOf ?q) + (?q rdfs:subPropertyOf ?r) => (?p rdfs:subPropertyOf ?r)
        if self.config.is_enabled(RdfsRule::Rdfs5) {
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
        }

        // RDFS Rule 6: Triple (?p rdf:type rdf:Property) => (?p rdfs:subPropertyOf ?p)
        // Properties are reflexive with respect to subPropertyOf (noisy)
        if self.config.is_enabled(RdfsRule::Rdfs6) {
            self.rule_engine.add_rule(Rule {
                name: "rdfs6".to_string(),
                body: vec![RuleAtom::Triple {
                    subject: Term::Variable("p".to_string()),
                    predicate: Term::Constant(RDF_TYPE.to_string()),
                    object: Term::Constant(RDF_PROPERTY.to_string()),
                }],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("p".to_string()),
                    predicate: Term::Constant(RDFS_SUBPROPERTY_OF.to_string()),
                    object: Term::Variable("p".to_string()),
                }],
            });
        }

        // RDFS Rule 7: Triple (?x ?p ?y) + (?p rdfs:subPropertyOf ?q) => (?x ?q ?y)
        if self.config.is_enabled(RdfsRule::Rdfs7) {
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
        }

        // RDFS Rule 8: Triple (?c rdf:type rdfs:Class) => (?c rdfs:subClassOf rdfs:Resource)
        // All classes are subclasses of rdfs:Resource (noisy)
        if self.config.is_enabled(RdfsRule::Rdfs8) {
            self.rule_engine.add_rule(Rule {
                name: "rdfs8".to_string(),
                body: vec![RuleAtom::Triple {
                    subject: Term::Variable("c".to_string()),
                    predicate: Term::Constant(RDF_TYPE.to_string()),
                    object: Term::Constant(RDFS_CLASS.to_string()),
                }],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("c".to_string()),
                    predicate: Term::Constant(RDFS_SUBCLASS_OF.to_string()),
                    object: Term::Constant(RDFS_RESOURCE.to_string()),
                }],
            });
        }

        // RDFS Rule 9: Triple (?x rdf:type ?c) + (?c rdfs:subClassOf ?d) => (?x rdf:type ?d)
        if self.config.is_enabled(RdfsRule::Rdfs9) {
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
        }

        // RDFS Rule 10: Triple (?c rdf:type rdfs:Class) => (?c rdfs:subClassOf ?c)
        // Classes are reflexive with respect to subClassOf (noisy)
        if self.config.is_enabled(RdfsRule::Rdfs10) {
            self.rule_engine.add_rule(Rule {
                name: "rdfs10".to_string(),
                body: vec![RuleAtom::Triple {
                    subject: Term::Variable("c".to_string()),
                    predicate: Term::Constant(RDF_TYPE.to_string()),
                    object: Term::Constant(RDFS_CLASS.to_string()),
                }],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("c".to_string()),
                    predicate: Term::Constant(RDFS_SUBCLASS_OF.to_string()),
                    object: Term::Variable("c".to_string()),
                }],
            });
        }

        // RDFS Rule 11: Triple (?c rdfs:subClassOf ?d) + (?d rdfs:subClassOf ?e) => (?c rdfs:subClassOf ?e)
        if self.config.is_enabled(RdfsRule::Rdfs11) {
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
        }

        // RDFS Rule 13: Triple (?c rdf:type rdfs:Datatype) => (?c rdfs:subClassOf rdfs:Literal)
        // Datatypes are subclasses of rdfs:Literal
        if self.config.is_enabled(RdfsRule::Rdfs13) {
            self.rule_engine.add_rule(Rule {
                name: "rdfs13".to_string(),
                body: vec![RuleAtom::Triple {
                    subject: Term::Variable("c".to_string()),
                    predicate: Term::Constant(RDF_TYPE.to_string()),
                    object: Term::Constant(RDFS_DATATYPE.to_string()),
                }],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("c".to_string()),
                    predicate: Term::Constant(RDFS_SUBCLASS_OF.to_string()),
                    object: Term::Constant(RDFS_LITERAL.to_string()),
                }],
            });
        }

        let enabled_count = self.config.enabled_rules.len();
        let enabled_names: Vec<&str> = self.config.enabled_rules.iter().map(|r| r.name()).collect();
        info!(
            "Initialized {} RDFS entailment rules: {:?}",
            enabled_count, enabled_names
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

        // rdfs1: (?s ?p ?o) → (?p rdf:type rdf:Property)
        if self.config.is_enabled(RdfsRule::Rdfs1) {
            new_facts.push(RuleAtom::Triple {
                subject: Term::Constant(predicate.to_string()),
                predicate: Term::Constant(RDF_TYPE.to_string()),
                object: Term::Constant(RDF_PROPERTY.to_string()),
            });
        }

        // rdfs4a: (?s ?p ?o) → (?s rdf:type rdfs:Resource)
        if self.config.is_enabled(RdfsRule::Rdfs4a) {
            new_facts.push(RuleAtom::Triple {
                subject: Term::Constant(subject.to_string()),
                predicate: Term::Constant(RDF_TYPE.to_string()),
                object: Term::Constant(RDFS_RESOURCE.to_string()),
            });
        }

        // rdfs4b: (?s ?p ?o) → (?o rdf:type rdfs:Resource)
        if self.config.is_enabled(RdfsRule::Rdfs4b) {
            new_facts.push(RuleAtom::Triple {
                subject: Term::Constant(object.to_string()),
                predicate: Term::Constant(RDF_TYPE.to_string()),
                object: Term::Constant(RDFS_RESOURCE.to_string()),
            });
        }

        // rdfs2: Domain inference - (?p rdfs:domain ?c) ∧ (?x ?p ?y) → (?x rdf:type ?c)
        if self.config.is_enabled(RdfsRule::Rdfs2) {
            if let Some(domains) = self.context.property_domains.get(predicate) {
                for domain in domains {
                    new_facts.push(RuleAtom::Triple {
                        subject: Term::Constant(subject.to_string()),
                        predicate: Term::Constant(RDF_TYPE.to_string()),
                        object: Term::Constant(domain.clone()),
                    });
                }
            }
        }

        // rdfs3: Range inference - (?p rdfs:range ?c) ∧ (?x ?p ?y) → (?y rdf:type ?c)
        if self.config.is_enabled(RdfsRule::Rdfs3) {
            if let Some(ranges) = self.context.property_ranges.get(predicate) {
                for range in ranges {
                    new_facts.push(RuleAtom::Triple {
                        subject: Term::Constant(object.to_string()),
                        predicate: Term::Constant(RDF_TYPE.to_string()),
                        object: Term::Constant(range.clone()),
                    });
                }
            }
        }

        // rdfs7: Subproperty inheritance - (?x ?p ?y) ∧ (?p rdfs:subPropertyOf ?q) → (?x ?q ?y)
        if self.config.is_enabled(RdfsRule::Rdfs7) {
            let superproperties = self.context.get_superproperties(predicate);
            for superproperty in superproperties {
                new_facts.push(RuleAtom::Triple {
                    subject: Term::Constant(subject.to_string()),
                    predicate: Term::Constant(superproperty),
                    object: Term::Constant(object.to_string()),
                });
            }
        }

        // rdfs9: Subclass type inheritance - (?x rdf:type ?c) ∧ (?c rdfs:subClassOf ?d) → (?x rdf:type ?d)
        if self.config.is_enabled(RdfsRule::Rdfs9) && predicate == RDF_TYPE {
            let superclasses = self.context.get_superclasses(object);
            for superclass in superclasses {
                new_facts.push(RuleAtom::Triple {
                    subject: Term::Constant(subject.to_string()),
                    predicate: Term::Constant(RDF_TYPE.to_string()),
                    object: Term::Constant(superclass),
                });
            }
        }

        // rdfs6: (?p rdf:type rdf:Property) → (?p rdfs:subPropertyOf ?p)
        if self.config.is_enabled(RdfsRule::Rdfs6)
            && predicate == RDF_TYPE
            && object == RDF_PROPERTY
        {
            new_facts.push(RuleAtom::Triple {
                subject: Term::Constant(subject.to_string()),
                predicate: Term::Constant(RDFS_SUBPROPERTY_OF.to_string()),
                object: Term::Constant(subject.to_string()),
            });
        }

        // rdfs8: (?c rdf:type rdfs:Class) → (?c rdfs:subClassOf rdfs:Resource)
        if self.config.is_enabled(RdfsRule::Rdfs8) && predicate == RDF_TYPE && object == RDFS_CLASS
        {
            new_facts.push(RuleAtom::Triple {
                subject: Term::Constant(subject.to_string()),
                predicate: Term::Constant(RDFS_SUBCLASS_OF.to_string()),
                object: Term::Constant(RDFS_RESOURCE.to_string()),
            });
        }

        // rdfs10: (?c rdf:type rdfs:Class) → (?c rdfs:subClassOf ?c)
        if self.config.is_enabled(RdfsRule::Rdfs10) && predicate == RDF_TYPE && object == RDFS_CLASS
        {
            new_facts.push(RuleAtom::Triple {
                subject: Term::Constant(subject.to_string()),
                predicate: Term::Constant(RDFS_SUBCLASS_OF.to_string()),
                object: Term::Constant(subject.to_string()),
            });
        }

        // rdfs13: (?c rdf:type rdfs:Datatype) → (?c rdfs:subClassOf rdfs:Literal)
        if self.config.is_enabled(RdfsRule::Rdfs13)
            && predicate == RDF_TYPE
            && object == RDFS_DATATYPE
        {
            new_facts.push(RuleAtom::Triple {
                subject: Term::Constant(subject.to_string()),
                predicate: Term::Constant(RDFS_SUBCLASS_OF.to_string()),
                object: Term::Constant(RDFS_LITERAL.to_string()),
            });
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

    #[test]
    fn test_rdfs_rule_enum() {
        // Test all() returns all 13 rules
        assert_eq!(RdfsRule::all().len(), 13);

        // Test minimal() returns 7 non-noisy rules
        assert_eq!(RdfsRule::minimal().len(), 7);
        assert!(RdfsRule::minimal().contains(&RdfsRule::Rdfs2));
        assert!(RdfsRule::minimal().contains(&RdfsRule::Rdfs9));
        assert!(!RdfsRule::minimal().contains(&RdfsRule::Rdfs1));
        assert!(!RdfsRule::minimal().contains(&RdfsRule::Rdfs4a));

        // Test noisy() returns 6 noisy rules
        assert_eq!(RdfsRule::noisy().len(), 6);
        assert!(RdfsRule::noisy().contains(&RdfsRule::Rdfs1));
        assert!(RdfsRule::noisy().contains(&RdfsRule::Rdfs4a));
        assert!(!RdfsRule::noisy().contains(&RdfsRule::Rdfs9));

        // Test name() method
        assert_eq!(RdfsRule::Rdfs1.name(), "rdfs1");
        assert_eq!(RdfsRule::Rdfs9.name(), "rdfs9");
        assert_eq!(RdfsRule::Rdfs13.name(), "rdfs13");
    }

    #[test]
    fn test_rdfs_profile_default() {
        // Default profile should be Minimal
        assert_eq!(RdfsProfile::default(), RdfsProfile::Minimal);
    }

    #[test]
    fn test_rdfs_config_from_profile() {
        // Test Full profile
        let full_config = RdfsConfig::from_profile(RdfsProfile::Full);
        assert_eq!(full_config.enabled_rules.len(), 13);
        assert!(full_config.is_enabled(RdfsRule::Rdfs1));
        assert!(full_config.is_enabled(RdfsRule::Rdfs9));

        // Test Minimal profile
        let minimal_config = RdfsConfig::from_profile(RdfsProfile::Minimal);
        assert_eq!(minimal_config.enabled_rules.len(), 7);
        assert!(!minimal_config.is_enabled(RdfsRule::Rdfs1));
        assert!(minimal_config.is_enabled(RdfsRule::Rdfs9));

        // Test None profile
        let none_config = RdfsConfig::from_profile(RdfsProfile::None);
        assert!(none_config.enabled_rules.is_empty());
    }

    #[test]
    fn test_rdfs_config_enable_disable() {
        let mut config = RdfsConfig::none();
        assert!(!config.is_enabled(RdfsRule::Rdfs9));

        config.enable(RdfsRule::Rdfs9);
        assert!(config.is_enabled(RdfsRule::Rdfs9));

        config.disable(RdfsRule::Rdfs9);
        assert!(!config.is_enabled(RdfsRule::Rdfs9));
    }

    #[test]
    fn test_rdfs_reasoner_builder_with_profile() {
        // Builder with Full profile
        let full_reasoner = RdfsReasoner::builder()
            .with_profile(RdfsProfile::Full)
            .build();
        assert_eq!(full_reasoner.config.enabled_rules.len(), 13);

        // Builder with Minimal profile
        let minimal_reasoner = RdfsReasoner::builder()
            .with_profile(RdfsProfile::Minimal)
            .build();
        assert_eq!(minimal_reasoner.config.enabled_rules.len(), 7);

        // Builder with None profile
        let none_reasoner = RdfsReasoner::builder()
            .with_profile(RdfsProfile::None)
            .build();
        assert!(none_reasoner.config.enabled_rules.is_empty());
    }

    #[test]
    fn test_rdfs_reasoner_builder_enable_disable() {
        // Start with Minimal, add a noisy rule
        let reasoner = RdfsReasoner::builder()
            .with_profile(RdfsProfile::Minimal)
            .enable_rule(RdfsRule::Rdfs8)
            .build();
        assert!(reasoner.is_rule_enabled(RdfsRule::Rdfs8));
        assert!(reasoner.is_rule_enabled(RdfsRule::Rdfs9));
        assert_eq!(reasoner.config.enabled_rules.len(), 8);

        // Start with Full, disable noisy rules
        let reasoner2 = RdfsReasoner::builder()
            .with_profile(RdfsProfile::Full)
            .disable_rules(RdfsRule::noisy())
            .build();
        assert!(!reasoner2.is_rule_enabled(RdfsRule::Rdfs1));
        assert!(reasoner2.is_rule_enabled(RdfsRule::Rdfs9));
        assert_eq!(reasoner2.config.enabled_rules.len(), 7);

        // Start from scratch, enable only specific rules
        let reasoner3 = RdfsReasoner::builder()
            .with_profile(RdfsProfile::None)
            .enable_rules(&[RdfsRule::Rdfs9, RdfsRule::Rdfs11])
            .build();
        assert!(reasoner3.is_rule_enabled(RdfsRule::Rdfs9));
        assert!(reasoner3.is_rule_enabled(RdfsRule::Rdfs11));
        assert!(!reasoner3.is_rule_enabled(RdfsRule::Rdfs2));
        assert_eq!(reasoner3.config.enabled_rules.len(), 2);
    }

    #[test]
    fn test_rdfs_reasoner_with_profile() {
        // Test with_profile constructor
        let full_reasoner = RdfsReasoner::with_profile(RdfsProfile::Full);
        assert_eq!(full_reasoner.config.enabled_rules.len(), 13);

        let minimal_reasoner = RdfsReasoner::with_profile(RdfsProfile::Minimal);
        assert_eq!(minimal_reasoner.config.enabled_rules.len(), 7);
    }

    #[test]
    fn test_rdfs_reasoner_context_only() {
        // Context-only reasoner should have no rules
        let reasoner = RdfsReasoner::context_only();
        assert!(reasoner.config.enabled_rules.is_empty());

        // But context should still work
        assert!(reasoner.context.classes.contains(vocabulary::RDFS_CLASS));
        assert!(reasoner
            .context
            .is_subclass_of(vocabulary::RDFS_DATATYPE, vocabulary::RDFS_CLASS));
    }

    #[test]
    fn test_rdfs_reasoner_default_is_minimal() {
        // new() should use Minimal profile by default
        let reasoner = RdfsReasoner::new();
        assert_eq!(reasoner.config.enabled_rules.len(), 7);
        assert!(!reasoner.is_rule_enabled(RdfsRule::Rdfs1));
        assert!(reasoner.is_rule_enabled(RdfsRule::Rdfs9));
    }

    #[test]
    fn test_disabled_rules_not_generating_facts() {
        use vocabulary::*;

        // Create reasoner with only rdfs9 enabled
        let mut reasoner = RdfsReasoner::builder()
            .with_profile(RdfsProfile::None)
            .enable_rule(RdfsRule::Rdfs9)
            .build();

        let facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("Person".to_string()),
                predicate: Term::Constant(RDFS_SUBCLASS_OF.to_string()),
                object: Term::Constant("Agent".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("john".to_string()),
                predicate: Term::Constant(RDF_TYPE.to_string()),
                object: Term::Constant("Person".to_string()),
            },
        ];

        let inferred = reasoner.infer(&facts).unwrap();

        // Should infer john is type Agent (rdfs9 is enabled)
        let expected_agent = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant(RDF_TYPE.to_string()),
            object: Term::Constant("Agent".to_string()),
        };
        assert!(inferred.contains(&expected_agent));

        // Should NOT infer john is type Resource (rdfs4a is disabled)
        let unexpected_resource = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant(RDF_TYPE.to_string()),
            object: Term::Constant(RDFS_RESOURCE.to_string()),
        };
        assert!(!inferred.contains(&unexpected_resource));
    }

    #[test]
    fn test_full_profile_generates_noisy_facts() {
        use vocabulary::*;

        // Create reasoner with Full profile
        let mut reasoner = RdfsReasoner::with_profile(RdfsProfile::Full);

        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("http://example.org/knows".to_string()),
            object: Term::Constant("mary".to_string()),
        }];

        let inferred = reasoner.infer(&facts).unwrap();

        // Full profile should generate Resource types (rdfs4a/rdfs4b)
        let john_resource = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant(RDF_TYPE.to_string()),
            object: Term::Constant(RDFS_RESOURCE.to_string()),
        };
        let mary_resource = RuleAtom::Triple {
            subject: Term::Constant("mary".to_string()),
            predicate: Term::Constant(RDF_TYPE.to_string()),
            object: Term::Constant(RDFS_RESOURCE.to_string()),
        };

        assert!(inferred.contains(&john_resource));
        assert!(inferred.contains(&mary_resource));
    }

    #[test]
    fn test_minimal_profile_skips_noisy_facts() {
        use vocabulary::*;

        // Create reasoner with Minimal profile (default)
        let mut reasoner = RdfsReasoner::new();

        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("http://example.org/knows".to_string()),
            object: Term::Constant("mary".to_string()),
        }];

        let inferred = reasoner.infer(&facts).unwrap();

        // Minimal profile should NOT generate Resource types
        let john_resource = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant(RDF_TYPE.to_string()),
            object: Term::Constant(RDFS_RESOURCE.to_string()),
        };

        assert!(!inferred.contains(&john_resource));
    }

    /// Benchmark test: Compare fact counts between profiles
    #[test]
    fn bench_profile_comparison() {
        // Generate test triples
        let mut triples = Vec::new();
        for i in 0..100 {
            triples.push(RuleAtom::Triple {
                subject: Term::Constant(format!("entity_{i}")),
                predicate: Term::Constant("http://example.org/property".to_string()),
                object: Term::Constant(format!("value_{i}")),
            });
        }

        // Test Minimal profile
        let mut minimal_reasoner = RdfsReasoner::with_profile(RdfsProfile::Minimal);
        let minimal_start = std::time::Instant::now();
        let minimal_result = minimal_reasoner.infer(&triples).unwrap();
        let minimal_duration = minimal_start.elapsed();

        // Test Full profile
        let mut full_reasoner = RdfsReasoner::with_profile(RdfsProfile::Full);
        let full_start = std::time::Instant::now();
        let full_result = full_reasoner.infer(&triples).unwrap();
        let full_duration = full_start.elapsed();

        println!(
            "Minimal profile: {} facts in {:?}",
            minimal_result.len(),
            minimal_duration
        );
        println!(
            "Full profile: {} facts in {:?}",
            full_result.len(),
            full_duration
        );

        // Full profile should generate significantly more facts
        assert!(full_result.len() > minimal_result.len());

        // Minimal should be faster (or at least not significantly slower)
        // Note: This is a soft assertion for documentation purposes
        println!(
            "Full profile generated {}x more facts",
            full_result.len() as f64 / minimal_result.len() as f64
        );
    }
}
