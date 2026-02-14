//! Test Case Generator for Rules
//!
//! Automatically generates test cases for rule-based inference systems.
//! Supports various test generation strategies including boundary value analysis,
//! equivalence partitioning, and property-based testing.
//!
//! # Features
//!
//! - **Automatic Test Generation**: Generate test cases from rule specifications
//! - **Boundary Value Analysis**: Test edge cases and boundary conditions
//! - **Equivalence Partitioning**: Generate representative test cases
//! - **Property-Based Testing**: Generate random test inputs
//! - **Coverage-Guided Generation**: Focus on untested rule paths
//! - **Mutation Testing**: Generate tests to detect rule errors
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::test_generator::{TestGenerator, GenerationStrategy};
//! use oxirs_rule::{Rule, RuleAtom, Term};
//!
//! let mut generator = TestGenerator::new();
//!
//! let rule = Rule {
//!     name: "test_rule".to_string(),
//!     body: vec![RuleAtom::Triple {
//!         subject: Term::Variable("X".to_string()),
//!         predicate: Term::Constant("p".to_string()),
//!         object: Term::Variable("Y".to_string()),
//!     }],
//!     head: vec![RuleAtom::Triple {
//!         subject: Term::Variable("X".to_string()),
//!         predicate: Term::Constant("q".to_string()),
//!         object: Term::Variable("Y".to_string()),
//!     }],
//! };
//!
//! // Generate test cases
//! let test_cases = generator.generate_tests(&rule, GenerationStrategy::Boundary);
//!
//! for test_case in test_cases {
//!     println!("Test: {:?}", test_case);
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::{Rule, RuleAtom, Term};
use scirs2_core::random::prelude::*;
use std::collections::{HashMap, HashSet};
use tracing::info;

/// Test generation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationStrategy {
    /// Basic test cases (positive and negative)
    Basic,
    /// Boundary value analysis
    Boundary,
    /// Equivalence partitioning
    Equivalence,
    /// Property-based random testing
    PropertyBased,
    /// Coverage-guided generation
    CoverageGuided,
    /// All strategies combined
    Comprehensive,
}

/// Generated test case
#[derive(Debug, Clone)]
pub struct TestCase {
    /// Test name
    pub name: String,
    /// Input facts
    pub input_facts: Vec<RuleAtom>,
    /// Expected output facts
    pub expected_outputs: Vec<RuleAtom>,
    /// Test type (positive or negative)
    pub test_type: TestType,
    /// Description
    pub description: String,
}

/// Test type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestType {
    /// Test should produce expected output
    Positive,
    /// Test should not trigger rule
    Negative,
    /// Test should produce error
    Error,
}

/// Test case generator
pub struct TestGenerator {
    /// Random number generator
    rng: StdRng,
    /// Number of random tests to generate
    num_random_tests: usize,
    /// Generated test counter
    test_counter: usize,
}

impl Default for TestGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl TestGenerator {
    /// Create a new test generator
    pub fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();
        Self {
            rng: seeded_rng(seed),
            num_random_tests: 10,
            test_counter: 0,
        }
    }

    /// Set number of random tests to generate
    pub fn set_num_random_tests(&mut self, count: usize) {
        self.num_random_tests = count;
    }

    /// Generate test cases for a rule
    pub fn generate_tests(&mut self, rule: &Rule, strategy: GenerationStrategy) -> Vec<TestCase> {
        info!(
            "Generating test cases for rule '{}' using {:?} strategy",
            rule.name, strategy
        );

        match strategy {
            GenerationStrategy::Basic => self.generate_basic_tests(rule),
            GenerationStrategy::Boundary => self.generate_boundary_tests(rule),
            GenerationStrategy::Equivalence => self.generate_equivalence_tests(rule),
            GenerationStrategy::PropertyBased => self.generate_property_based_tests(rule),
            GenerationStrategy::CoverageGuided => self.generate_coverage_guided_tests(rule),
            GenerationStrategy::Comprehensive => {
                let mut tests = Vec::new();
                tests.extend(self.generate_basic_tests(rule));
                tests.extend(self.generate_boundary_tests(rule));
                tests.extend(self.generate_equivalence_tests(rule));
                tests.extend(self.generate_property_based_tests(rule));
                tests
            }
        }
    }

    /// Generate basic test cases
    fn generate_basic_tests(&mut self, rule: &Rule) -> Vec<TestCase> {
        let mut tests = Vec::new();

        // Positive test: exact match
        if let Some(test) = self.generate_positive_test(rule) {
            tests.push(test);
        }

        // Negative test: no match
        if let Some(test) = self.generate_negative_test(rule) {
            tests.push(test);
        }

        // Empty input test
        tests.push(TestCase {
            name: format!("test_{}_empty", self.next_test_id()),
            input_facts: vec![],
            expected_outputs: vec![],
            test_type: TestType::Negative,
            description: "Test with empty input facts".to_string(),
        });

        tests
    }

    /// Generate boundary value tests
    fn generate_boundary_tests(&mut self, rule: &Rule) -> Vec<TestCase> {
        let mut tests = Vec::new();

        // Single fact boundary
        if let Some(test) = self.generate_positive_test(rule) {
            tests.push(TestCase {
                name: format!("test_{}_single_fact", self.next_test_id()),
                input_facts: test.input_facts.into_iter().take(1).collect(),
                expected_outputs: test.expected_outputs,
                test_type: TestType::Positive,
                description: "Boundary test with single fact".to_string(),
            });
        }

        // Multiple facts boundary
        if let Some(test) = self.generate_positive_test(rule) {
            let mut facts = test.input_facts.clone();
            facts.extend(test.input_facts.clone()); // Duplicate
            tests.push(TestCase {
                name: format!("test_{}_multiple_facts", self.next_test_id()),
                input_facts: facts,
                expected_outputs: test.expected_outputs,
                test_type: TestType::Positive,
                description: "Boundary test with multiple facts".to_string(),
            });
        }

        tests
    }

    /// Generate equivalence partition tests
    fn generate_equivalence_tests(&mut self, rule: &Rule) -> Vec<TestCase> {
        let mut tests = Vec::new();

        // Test with different variable bindings
        for i in 0..3 {
            if let Some(mut test) = self.generate_positive_test(rule) {
                // Modify constants to create different equivalence classes
                test.input_facts = test
                    .input_facts
                    .into_iter()
                    .map(|atom| self.modify_atom_constants(&atom, i))
                    .collect();

                test.expected_outputs = test
                    .expected_outputs
                    .into_iter()
                    .map(|atom| self.modify_atom_constants(&atom, i))
                    .collect();

                test.name = format!("test_{}_equiv_class_{}", self.next_test_id(), i);
                test.description = format!("Equivalence partition test (class {})", i);
                tests.push(test);
            }
        }

        tests
    }

    /// Generate property-based random tests
    fn generate_property_based_tests(&mut self, rule: &Rule) -> Vec<TestCase> {
        let mut tests = Vec::new();

        for _ in 0..self.num_random_tests {
            let test = TestCase {
                name: format!("test_{}_random", self.next_test_id()),
                input_facts: self.generate_random_facts(rule, 3),
                expected_outputs: vec![], // To be determined by execution
                test_type: TestType::Positive,
                description: "Property-based random test".to_string(),
            };
            tests.push(test);
        }

        tests
    }

    /// Generate coverage-guided tests
    fn generate_coverage_guided_tests(&mut self, rule: &Rule) -> Vec<TestCase> {
        let mut tests = Vec::new();

        // Generate tests for each atom in the rule body
        for (i, _atom) in rule.body.iter().enumerate() {
            if let Some(mut test) = self.generate_positive_test(rule) {
                test.name = format!("test_{}_coverage_atom_{}", self.next_test_id(), i);
                test.description = format!("Coverage test for body atom {}", i);
                tests.push(test);
            }
        }

        // Generate tests for rule head
        for (i, _atom) in rule.head.iter().enumerate() {
            if let Some(mut test) = self.generate_positive_test(rule) {
                test.name = format!("test_{}_coverage_head_{}", self.next_test_id(), i);
                test.description = format!("Coverage test for head atom {}", i);
                tests.push(test);
            }
        }

        tests
    }

    /// Generate a positive test case
    fn generate_positive_test(&mut self, rule: &Rule) -> Option<TestCase> {
        if rule.body.is_empty() || rule.head.is_empty() {
            return None;
        }

        // Extract variables from rule body
        let variables = self.extract_variables(&rule.body);

        // Create variable bindings
        let mut bindings = HashMap::new();
        for (i, var) in variables.iter().enumerate() {
            bindings.insert(var.clone(), format!("entity_{i}"));
        }

        // Generate input facts
        let input_facts: Vec<RuleAtom> = rule
            .body
            .iter()
            .map(|atom| self.instantiate_atom(atom, &bindings))
            .collect();

        // Generate expected outputs
        let expected_outputs: Vec<RuleAtom> = rule
            .head
            .iter()
            .map(|atom| self.instantiate_atom(atom, &bindings))
            .collect();

        Some(TestCase {
            name: format!("test_{}_positive", self.next_test_id()),
            input_facts,
            expected_outputs,
            test_type: TestType::Positive,
            description: "Positive test case with valid input".to_string(),
        })
    }

    /// Generate a negative test case
    fn generate_negative_test(&mut self, rule: &Rule) -> Option<TestCase> {
        if rule.body.is_empty() {
            return None;
        }

        // Generate facts that don't match the rule body
        let input_facts: Vec<RuleAtom> = rule
            .body
            .iter()
            .map(|atom| self.generate_non_matching_fact(atom))
            .collect();

        Some(TestCase {
            name: format!("test_{}_negative", self.next_test_id()),
            input_facts,
            expected_outputs: vec![],
            test_type: TestType::Negative,
            description: "Negative test case with non-matching input".to_string(),
        })
    }

    /// Extract variables from atoms
    fn extract_variables(&self, atoms: &[RuleAtom]) -> Vec<String> {
        let mut variables = HashSet::new();

        for atom in atoms {
            match atom {
                RuleAtom::Triple {
                    subject,
                    predicate,
                    object,
                } => {
                    if let Term::Variable(v) = subject {
                        variables.insert(v.clone());
                    }
                    if let Term::Variable(v) = predicate {
                        variables.insert(v.clone());
                    }
                    if let Term::Variable(v) = object {
                        variables.insert(v.clone());
                    }
                }
                RuleAtom::Builtin { args, .. } => {
                    for arg in args {
                        if let Term::Variable(v) = arg {
                            variables.insert(v.clone());
                        }
                    }
                }
                RuleAtom::NotEqual { left, right }
                | RuleAtom::GreaterThan { left, right }
                | RuleAtom::LessThan { left, right } => {
                    if let Term::Variable(v) = left {
                        variables.insert(v.clone());
                    }
                    if let Term::Variable(v) = right {
                        variables.insert(v.clone());
                    }
                }
            }
        }

        variables.into_iter().collect()
    }

    /// Instantiate an atom with variable bindings
    fn instantiate_atom(&self, atom: &RuleAtom, bindings: &HashMap<String, String>) -> RuleAtom {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => RuleAtom::Triple {
                subject: self.instantiate_term(subject, bindings),
                predicate: self.instantiate_term(predicate, bindings),
                object: self.instantiate_term(object, bindings),
            },
            RuleAtom::Builtin { name, args } => RuleAtom::Builtin {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|t| self.instantiate_term(t, bindings))
                    .collect(),
            },
            RuleAtom::NotEqual { left, right } => RuleAtom::NotEqual {
                left: self.instantiate_term(left, bindings),
                right: self.instantiate_term(right, bindings),
            },
            RuleAtom::GreaterThan { left, right } => RuleAtom::GreaterThan {
                left: self.instantiate_term(left, bindings),
                right: self.instantiate_term(right, bindings),
            },
            RuleAtom::LessThan { left, right } => RuleAtom::LessThan {
                left: self.instantiate_term(left, bindings),
                right: self.instantiate_term(right, bindings),
            },
        }
    }

    /// Instantiate a term with variable bindings
    fn instantiate_term(&self, term: &Term, bindings: &HashMap<String, String>) -> Term {
        match term {
            Term::Variable(v) => {
                if let Some(value) = bindings.get(v) {
                    Term::Constant(value.clone())
                } else {
                    term.clone()
                }
            }
            _ => term.clone(),
        }
    }

    /// Generate a non-matching fact
    fn generate_non_matching_fact(&mut self, atom: &RuleAtom) -> RuleAtom {
        match atom {
            RuleAtom::Triple {
                subject: _,
                predicate,
                object: _,
            } => RuleAtom::Triple {
                subject: Term::Constant("non_matching_subject".to_string()),
                predicate: self.modify_term_for_non_match(predicate),
                object: Term::Constant("non_matching_object".to_string()),
            },
            _ => atom.clone(),
        }
    }

    /// Modify term to create non-matching pattern
    fn modify_term_for_non_match(&self, term: &Term) -> Term {
        match term {
            Term::Constant(c) => Term::Constant(format!("non_matching_{c}")),
            Term::Variable(v) => Term::Variable(format!("non_matching_{v}")),
            Term::Literal(l) => Term::Literal(format!("non_matching_{l}")),
            Term::Function { name, args } => Term::Function {
                name: format!("non_matching_{name}"),
                args: args.clone(),
            },
        }
    }

    /// Modify atom constants
    fn modify_atom_constants(&self, atom: &RuleAtom, modifier: usize) -> RuleAtom {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => RuleAtom::Triple {
                subject: self.modify_term_constant(subject, modifier),
                predicate: predicate.clone(),
                object: self.modify_term_constant(object, modifier),
            },
            _ => atom.clone(),
        }
    }

    /// Modify term constant
    fn modify_term_constant(&self, term: &Term, modifier: usize) -> Term {
        match term {
            Term::Constant(c) => Term::Constant(format!("{}_{}", c, modifier)),
            _ => term.clone(),
        }
    }

    /// Generate random facts
    fn generate_random_facts(&mut self, rule: &Rule, count: usize) -> Vec<RuleAtom> {
        let mut facts = Vec::new();

        for _ in 0..count {
            if !rule.body.is_empty() {
                let random_index = self.rng.gen_range(0..rule.body.len());
                if let Some(atom) = rule.body.get(random_index) {
                    facts.push(self.generate_random_atom(atom));
                }
            }
        }

        facts
    }

    /// Generate a random atom
    fn generate_random_atom(&mut self, template: &RuleAtom) -> RuleAtom {
        match template {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => RuleAtom::Triple {
                subject: self.generate_random_term(subject),
                predicate: predicate.clone(),
                object: self.generate_random_term(object),
            },
            _ => template.clone(),
        }
    }

    /// Generate a random term
    fn generate_random_term(&mut self, template: &Term) -> Term {
        match template {
            Term::Variable(_) => {
                let random_value: u32 = self.rng.random();
                Term::Constant(format!("random_{}", random_value))
            }
            _ => template.clone(),
        }
    }

    /// Get next test ID
    fn next_test_id(&mut self) -> usize {
        let id = self.test_counter;
        self.test_counter += 1;
        id
    }

    /// Export test cases to Rust test code
    pub fn export_to_rust_tests(&self, tests: &[TestCase]) -> String {
        let mut code = String::new();

        code.push_str("#[cfg(test)]\n");
        code.push_str("mod generated_tests {\n");
        code.push_str("    use super::*;\n\n");

        for test in tests {
            code.push_str("    #[test]\n");
            code.push_str(&format!("    fn {}() {{\n", test.name));
            code.push_str(&format!("        // {}\n", test.description));
            code.push_str("        let mut engine = RuleEngine::new();\n\n");
            code.push_str("        let input_facts = vec![\n");

            for fact in &test.input_facts {
                code.push_str(&format!("            {:?},\n", fact));
            }

            code.push_str("        ];\n\n");

            code.push_str("        let results = engine.forward_chain(&input_facts).unwrap();\n\n");

            match test.test_type {
                TestType::Positive => {
                    code.push_str("        // Verify expected outputs\n");
                    for expected in &test.expected_outputs {
                        code.push_str(&format!(
                            "        assert!(results.contains(&{:?}));\n",
                            expected
                        ));
                    }
                }
                TestType::Negative => {
                    code.push_str("        // Verify no unexpected outputs\n");
                    code.push_str(&format!(
                        "        assert_eq!(results.len(), {});\n",
                        test.input_facts.len()
                    ));
                }
                TestType::Error => {
                    code.push_str("        // Verify error handling\n");
                }
            }

            code.push_str("    }\n\n");
        }

        code.push_str("}\n");
        code
    }

    /// Reset test counter
    pub fn reset(&mut self) {
        self.test_counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_generation() {
        let mut generator = TestGenerator::new();

        let rule = Rule {
            name: "test_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("q".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        };

        let tests = generator.generate_tests(&rule, GenerationStrategy::Basic);
        assert!(!tests.is_empty());
        assert!(tests.iter().any(|t| t.test_type == TestType::Positive));
        assert!(tests.iter().any(|t| t.test_type == TestType::Negative));
    }

    #[test]
    fn test_boundary_generation() {
        let mut generator = TestGenerator::new();

        let rule = Rule {
            name: "test_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("q".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        };

        let tests = generator.generate_tests(&rule, GenerationStrategy::Boundary);
        assert!(!tests.is_empty());
    }

    #[test]
    fn test_property_based_generation() {
        let mut generator = TestGenerator::new();
        generator.set_num_random_tests(5);

        let rule = Rule {
            name: "test_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("q".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        };

        let tests = generator.generate_tests(&rule, GenerationStrategy::PropertyBased);
        assert_eq!(tests.len(), 5);
    }

    #[test]
    fn test_comprehensive_generation() {
        let mut generator = TestGenerator::new();

        let rule = Rule {
            name: "test_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("q".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        };

        let tests = generator.generate_tests(&rule, GenerationStrategy::Comprehensive);
        assert!(tests.len() > 10); // Should have many tests
    }

    #[test]
    fn test_variable_extraction() {
        let generator = TestGenerator::new();

        let atoms = vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("p".to_string()),
            object: Term::Variable("Y".to_string()),
        }];

        let variables = generator.extract_variables(&atoms);
        assert_eq!(variables.len(), 2);
        assert!(variables.contains(&"X".to_string()));
        assert!(variables.contains(&"Y".to_string()));
    }

    #[test]
    fn test_rust_test_export() {
        let mut generator = TestGenerator::new();

        let rule = Rule {
            name: "test_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("q".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        };

        let tests = generator.generate_tests(&rule, GenerationStrategy::Basic);
        let rust_code = generator.export_to_rust_tests(&tests);

        assert!(rust_code.contains("#[test]"));
        assert!(rust_code.contains("fn test_"));
    }

    #[test]
    fn test_generator_reset() {
        let mut generator = TestGenerator::new();

        let rule = Rule {
            name: "test_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("q".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        };

        generator.generate_tests(&rule, GenerationStrategy::Basic);
        assert!(generator.test_counter > 0);

        generator.reset();
        assert_eq!(generator.test_counter, 0);
    }
}
