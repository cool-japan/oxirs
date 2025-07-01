//! # Getting Started with OxiRS Rule Engine
//!
//! This module provides a comprehensive getting started guide for the OxiRS Rule Engine,
//! including basic concepts, rule authoring best practices, and step-by-step examples.

use anyhow::Result;
use std::collections::HashMap;

use crate::{Rule, RuleAtom, RuleEngine, Term};

/// # Quick Start Guide
///
/// This section provides the fastest way to get up and running with the OxiRS Rule Engine.
///
/// ## Basic Example
///
/// ```rust
/// use oxirs_rule::{RuleEngine, Rule, RuleAtom, Term};
///
/// // Create a new rule engine
/// let mut engine = RuleEngine::new();
///
/// // Define a simple rule: if X is a person, then X is mortal
/// let rule = Rule {
///     name: "mortality_rule".to_string(),
///     body: vec![
///         RuleAtom::Triple {
///             subject: Term::Variable("X".to_string()),
///             predicate: Term::Constant("type".to_string()),
///             object: Term::Constant("Person".to_string()),
///         }
///     ],
///     head: vec![
///         RuleAtom::Triple {
///             subject: Term::Variable("X".to_string()),
///             predicate: Term::Constant("type".to_string()),
///             object: Term::Constant("Mortal".to_string()),
///         }
///     ],
/// };
///
/// // Add the rule to the engine
/// engine.add_rule(rule);
///
/// // Add initial facts
/// let facts = vec![
///     RuleAtom::Triple {
///         subject: Term::Constant("socrates".to_string()),
///         predicate: Term::Constant("type".to_string()),
///         object: Term::Constant("Person".to_string()),
///     }
/// ];
///
/// // Run forward chaining to derive new facts
/// let derived_facts = engine.forward_chain(&facts).unwrap();
/// println!("Derived facts: {:?}", derived_facts);
/// ```

pub struct GettingStartedGuide;

impl GettingStartedGuide {
    /// Create a basic rule engine setup for learning
    pub fn create_basic_engine() -> RuleEngine {
        RuleEngine::new()
    }

    /// Demonstrate basic rule creation and execution
    pub fn basic_example() -> Result<Vec<RuleAtom>> {
        let mut engine = RuleEngine::new();

        // Rule: If X is a person, then X is mortal
        let mortality_rule = Rule {
            name: "mortality_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Person".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Mortal".to_string()),
            }],
        };

        engine.add_rule(mortality_rule);

        // Initial facts
        let facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("socrates".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Person".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("plato".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Person".to_string()),
            },
        ];

        engine.forward_chain(&facts)
    }

    /// Demonstrate backward chaining for goal-driven reasoning
    pub fn backward_chaining_example() -> Result<bool> {
        let mut engine = RuleEngine::new();

        // Rule: If X teaches Y, and Y is a student, then X is a teacher
        let teaching_rule = Rule {
            name: "teaching_rule".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("teaches".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("type".to_string()),
                    object: Term::Constant("Student".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Teacher".to_string()),
            }],
        };

        engine.add_rule(teaching_rule);

        // Add facts
        engine.add_fact(RuleAtom::Triple {
            subject: Term::Constant("aristotle".to_string()),
            predicate: Term::Constant("teaches".to_string()),
            object: Term::Constant("alexander".to_string()),
        });

        engine.add_fact(RuleAtom::Triple {
            subject: Term::Constant("alexander".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("Student".to_string()),
        });

        // Query: Is Aristotle a teacher?
        let goal = RuleAtom::Triple {
            subject: Term::Constant("aristotle".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("Teacher".to_string()),
        };

        engine.backward_chain(&goal)
    }

    /// Demonstrate family relationship reasoning
    pub fn family_relationships_example() -> Result<Vec<RuleAtom>> {
        let mut engine = RuleEngine::new();

        // Rule 1: If X is parent of Y, and Y is parent of Z, then X is grandparent of Z
        let grandparent_rule = Rule {
            name: "grandparent_rule".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("parent".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("parent".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("grandparent".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        };

        // Rule 2: If X is parent of Y, then Y is child of X
        let child_rule = Rule {
            name: "child_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Y".to_string()),
                predicate: Term::Constant("child".to_string()),
                object: Term::Variable("X".to_string()),
            }],
        };

        engine.add_rule(grandparent_rule);
        engine.add_rule(child_rule);

        // Family facts
        let facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("alice".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Constant("bob".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("bob".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Constant("charlie".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("bob".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Constant("diana".to_string()),
            },
        ];

        engine.forward_chain(&facts)
    }

    /// Demonstrate ontology reasoning with classes and properties
    pub fn ontology_reasoning_example() -> Result<Vec<RuleAtom>> {
        let mut engine = RuleEngine::new();

        // Rule 1: If X is subclass of Y, and Z is instance of X, then Z is instance of Y
        let subclass_rule = Rule {
            name: "subclass_inheritance".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("subClassOf".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Z".to_string()),
                    predicate: Term::Constant("type".to_string()),
                    object: Term::Variable("X".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Z".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        };

        // Rule 2: If P is subproperty of Q, and X P Y, then X Q Y
        let subproperty_rule = Rule {
            name: "subproperty_inheritance".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("P".to_string()),
                    predicate: Term::Constant("subPropertyOf".to_string()),
                    object: Term::Variable("Q".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Variable("P".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Variable("Q".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        };

        engine.add_rule(subclass_rule);
        engine.add_rule(subproperty_rule);

        // Ontology facts
        let facts = vec![
            // Class hierarchy
            RuleAtom::Triple {
                subject: Term::Constant("Dog".to_string()),
                predicate: Term::Constant("subClassOf".to_string()),
                object: Term::Constant("Mammal".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("Mammal".to_string()),
                predicate: Term::Constant("subClassOf".to_string()),
                object: Term::Constant("Animal".to_string()),
            },
            // Property hierarchy
            RuleAtom::Triple {
                subject: Term::Constant("owns".to_string()),
                predicate: Term::Constant("subPropertyOf".to_string()),
                object: Term::Constant("related".to_string()),
            },
            // Instance data
            RuleAtom::Triple {
                subject: Term::Constant("fido".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Dog".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("john".to_string()),
                predicate: Term::Constant("owns".to_string()),
                object: Term::Constant("fido".to_string()),
            },
        ];

        engine.forward_chain(&facts)
    }
}

/// # Rule Authoring Best Practices
///
/// This section provides comprehensive guidelines for writing effective rules.

pub struct RuleAuthoringBestPractices;

impl RuleAuthoringBestPractices {
    /// Guidelines for naming variables in rules
    pub fn variable_naming_guidelines() -> Vec<&'static str> {
        vec![
            "Use meaningful variable names (e.g., 'Person' instead of 'X')",
            "Use consistent naming conventions across rules",
            "Use uppercase for variables to distinguish from constants",
            "Use descriptive names for complex patterns",
            "Avoid single-letter variables except for simple cases",
            "Use domain-specific prefixes (e.g., 'pers_' for person variables)",
        ]
    }

    /// Guidelines for structuring rule bodies
    pub fn rule_structure_guidelines() -> Vec<&'static str> {
        vec![
            "Order atoms from most specific to least specific",
            "Place constant-heavy atoms first for efficient filtering",
            "Group related atoms together for readability",
            "Use built-in predicates judiciously for validation",
            "Keep rule bodies concise and focused",
            "Avoid overly complex rules - split into multiple simpler rules",
        ]
    }

    /// Performance optimization tips
    pub fn performance_tips() -> Vec<&'static str> {
        vec![
            "Place most selective atoms first in rule bodies",
            "Use constants instead of variables when possible",
            "Avoid rules that generate infinite loops",
            "Use indexing hints for frequently used patterns",
            "Consider rule ordering for optimization",
            "Profile rule execution to identify bottlenecks",
            "Use caching for expensive computations",
            "Minimize variable scope and reuse",
        ]
    }

    /// Common pitfalls and how to avoid them
    pub fn common_pitfalls() -> Vec<(&'static str, &'static str)> {
        vec![
            ("Infinite recursion", "Always ensure termination conditions"),
            (
                "Variable scope confusion",
                "Use clear naming and scope management",
            ),
            (
                "Performance bottlenecks",
                "Profile and optimize critical rules",
            ),
            ("Rule conflicts", "Test rule interactions thoroughly"),
            (
                "Over-generalization",
                "Be specific about rule applicability",
            ),
            ("Under-specification", "Include necessary constraints"),
            ("Complex debugging", "Use rule tracing and debugging tools"),
            ("Memory issues", "Monitor memory usage with large rule sets"),
        ]
    }

    /// Example of a well-structured rule with comments
    pub fn well_structured_rule_example() -> Rule {
        Rule {
            name: "employee_manager_hierarchy".to_string(),
            body: vec![
                // Most specific constraint first
                RuleAtom::Triple {
                    subject: Term::Variable("Employee".to_string()),
                    predicate: Term::Constant("type".to_string()),
                    object: Term::Constant("Person".to_string()),
                },
                // Relationship constraint
                RuleAtom::Triple {
                    subject: Term::Variable("Employee".to_string()),
                    predicate: Term::Constant("worksFor".to_string()),
                    object: Term::Variable("Manager".to_string()),
                },
                // Manager validation
                RuleAtom::Triple {
                    subject: Term::Variable("Manager".to_string()),
                    predicate: Term::Constant("type".to_string()),
                    object: Term::Constant("Manager".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Employee".to_string()),
                predicate: Term::Constant("hasManager".to_string()),
                object: Term::Variable("Manager".to_string()),
            }],
        }
    }

    /// Example of rule validation and testing
    pub fn validate_rule_example() -> Result<bool> {
        let rule = Self::well_structured_rule_example();

        // Validation checks
        if rule.name.is_empty() {
            return Ok(false);
        }

        if rule.body.is_empty() || rule.head.is_empty() {
            return Ok(false);
        }

        // Check for variable consistency
        let mut body_vars = std::collections::HashSet::new();
        let mut head_vars = std::collections::HashSet::new();

        for atom in &rule.body {
            Self::collect_variables(atom, &mut body_vars);
        }

        for atom in &rule.head {
            Self::collect_variables(atom, &mut head_vars);
        }

        // All head variables should appear in body (safety condition)
        for var in &head_vars {
            if !body_vars.contains(var) {
                println!("Warning: Variable {} in head but not in body", var);
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn collect_variables(atom: &RuleAtom, vars: &mut std::collections::HashSet<String>) {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                if let Term::Variable(var) = subject {
                    vars.insert(var.clone());
                }
                if let Term::Variable(var) = predicate {
                    vars.insert(var.clone());
                }
                if let Term::Variable(var) = object {
                    vars.insert(var.clone());
                }
            }
            _ => {} // Handle other atom types as needed
        }
    }
}

/// # Debugging and Troubleshooting Guide
///
/// This section provides guidance on debugging rule engines and troubleshooting common issues.

pub struct DebuggingGuide;

impl DebuggingGuide {
    /// Enable debugging for rule execution
    pub fn enable_debugging_example() -> Result<()> {
        use crate::debug::DebuggableRuleEngine;

        let mut debug_engine = DebuggableRuleEngine::new();

        // Enable debugging with step mode
        debug_engine.enable_debugging(true);

        // Add breakpoints
        debug_engine.add_breakpoint("specific_rule_name");

        // Execute with debugging
        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("test".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("entity".to_string()),
        }];

        let _result = debug_engine.debug_forward_chain(&facts)?;

        // Get debugging information
        let trace = debug_engine.get_trace();
        let metrics = debug_engine.get_metrics();
        let conflicts = debug_engine.get_conflicts();

        println!("Execution trace: {} entries", trace.len());
        println!("Performance metrics: {:?}", metrics);
        println!("Detected conflicts: {}", conflicts.len());

        // Generate debug report
        let report = debug_engine.generate_debug_report();
        println!("{}", report);

        Ok(())
    }

    /// Common debugging techniques
    pub fn debugging_techniques() -> Vec<(&'static str, &'static str)> {
        vec![
            (
                "Rule tracing",
                "Enable execution tracing to see rule firing order",
            ),
            (
                "Breakpoints",
                "Set breakpoints on specific rules for step debugging",
            ),
            (
                "Performance profiling",
                "Measure rule execution times and memory usage",
            ),
            (
                "Conflict detection",
                "Identify contradictory or redundant rules",
            ),
            ("Derivation paths", "Trace how specific facts were derived"),
            ("Cache analysis", "Monitor cache hit rates and performance"),
            ("Memory monitoring", "Track memory usage during execution"),
            (
                "Statistics collection",
                "Gather execution statistics for optimization",
            ),
        ]
    }

    /// Troubleshooting checklist
    pub fn troubleshooting_checklist() -> Vec<&'static str> {
        vec![
            "Check rule syntax and structure",
            "Verify variable consistency between head and body",
            "Ensure termination conditions for recursive rules",
            "Validate input facts format and types",
            "Check for infinite loops or cycles",
            "Monitor memory usage for large datasets",
            "Verify rule execution order and dependencies",
            "Test with simplified rule sets first",
            "Use debugging tools to trace execution",
            "Check for rule conflicts and contradictions",
        ]
    }
}

/// # Integration Examples
///
/// Examples of integrating the rule engine with other components.

pub struct IntegrationExamples;

impl IntegrationExamples {
    /// Integration with RDF data
    pub fn rdf_integration_example() -> Result<Vec<RuleAtom>> {
        let mut engine = RuleEngine::new();

        // Rule for RDFS subclass inference
        let rdfs_rule = Rule {
            name: "rdfs_subclass".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant(
                        "http://www.w3.org/2000/01/rdf-schema#subClassOf".to_string(),
                    ),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Z".to_string()),
                    predicate: Term::Constant(
                        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                    ),
                    object: Term::Variable("X".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Z".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                object: Term::Variable("Y".to_string()),
            }],
        };

        engine.add_rule(rdfs_rule);

        // RDF facts (using full URIs)
        let facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("http://example.org/Dog".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/2000/01/rdf-schema#subClassOf".to_string(),
                ),
                object: Term::Constant("http://example.org/Animal".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("http://example.org/fido".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                object: Term::Constant("http://example.org/Dog".to_string()),
            },
        ];

        engine.forward_chain(&facts)
    }

    /// Integration with caching system
    pub fn caching_integration_example() -> Result<()> {
        use crate::cache::RuleCache;

        let cache = RuleCache::new();
        let mut engine = RuleEngine::new();

        // Enable caching
        engine.set_cache(Some(cache));

        // Rules will now be automatically cached
        let rule = Rule {
            name: "cached_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("likes".to_string()),
                object: Term::Constant("coffee".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("drinks".to_string()),
                object: Term::Constant("beverage".to_string()),
            }],
        };

        engine.add_rule(rule);

        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant("likes".to_string()),
            object: Term::Constant("coffee".to_string()),
        }];

        // First execution - results will be cached
        let _result1 = engine.forward_chain(&facts)?;

        // Second execution - results will be retrieved from cache
        let _result2 = engine.forward_chain(&facts)?;

        // Get cache statistics
        if let Some(cache) = engine.get_cache() {
            let stats = cache.get_statistics();
            println!("Cache hit rate: {:.2}%", stats.rule_cache.hit_rate * 100.0);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_example() {
        let result = GettingStartedGuide::basic_example();
        assert!(result.is_ok());
        let facts = result.unwrap();
        assert!(!facts.is_empty());

        // Should derive that socrates and plato are mortal
        let mortal_facts: Vec<_> = facts
            .iter()
            .filter(|fact| match fact {
                RuleAtom::Triple {
                    predicate, object, ..
                } => {
                    predicate == &Term::Constant("type".to_string())
                        && object == &Term::Constant("Mortal".to_string())
                }
                _ => false,
            })
            .collect();

        assert!(mortal_facts.len() >= 2);
    }

    #[test]
    fn test_backward_chaining_example() {
        let result = GettingStartedGuide::backward_chaining_example();
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should prove that Aristotle is a teacher
    }

    #[test]
    fn test_family_relationships() {
        let result = GettingStartedGuide::family_relationships_example();
        assert!(result.is_ok());
        let facts = result.unwrap();

        // Should derive grandparent and child relationships
        let grandparent_facts: Vec<_> = facts
            .iter()
            .filter(|fact| match fact {
                RuleAtom::Triple { predicate, .. } => {
                    predicate == &Term::Constant("grandparent".to_string())
                }
                _ => false,
            })
            .collect();

        assert!(!grandparent_facts.is_empty());
    }

    #[test]
    fn test_rule_validation() {
        let result = RuleAuthoringBestPractices::validate_rule_example();
        assert!(result.is_ok());
        assert!(result.unwrap()); // Well-structured rule should validate
    }

    #[test]
    fn test_ontology_reasoning() {
        let result = GettingStartedGuide::ontology_reasoning_example();
        assert!(result.is_ok());
        let facts = result.unwrap();

        // Should derive class and property inheritance
        assert!(!facts.is_empty());
    }

    #[test]
    fn test_rdf_integration() {
        let result = IntegrationExamples::rdf_integration_example();
        assert!(result.is_ok());
        let facts = result.unwrap();

        // Should derive that fido is an Animal (via subclass inference)
        let animal_facts: Vec<_> = facts
            .iter()
            .filter(|fact| match fact {
                RuleAtom::Triple {
                    subject, object, ..
                } => {
                    subject == &Term::Constant("http://example.org/fido".to_string())
                        && object == &Term::Constant("http://example.org/Animal".to_string())
                }
                _ => false,
            })
            .collect();

        assert!(!animal_facts.is_empty());
    }
}
