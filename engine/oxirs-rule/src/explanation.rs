//! Explanation and Provenance Tracking for Rule-Based Reasoning
//!
//! Provides comprehensive explanation capabilities for understanding how facts were derived,
//! including provenance tracking, inference graphs, and why/how explanations.
//!
//! # Features
//!
//! - **Provenance Tracking**: Record the complete derivation history
//! - **Why Explanations**: Explain why a fact is true
//! - **How Explanations**: Show how a fact was derived
//! - **Inference Graphs**: Visualize the reasoning process
//! - **Justifications**: Provide minimal supporting evidence
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::explanation::{ExplanationEngine, ExplanationRequest, DerivationMethod};
//! use oxirs_rule::{RuleEngine, Rule, RuleAtom, Term};
//!
//! let mut engine = RuleEngine::new();
//! let mut explainer = ExplanationEngine::new();
//!
//! // Add rule
//! let rule = Rule {
//!     name: "mortal".to_string(),
//!     body: vec![RuleAtom::Triple {
//!         subject: Term::Variable("X".to_string()),
//!         predicate: Term::Constant("type".to_string()),
//!         object: Term::Constant("human".to_string()),
//!     }],
//!     head: vec![RuleAtom::Triple {
//!         subject: Term::Variable("X".to_string()),
//!         predicate: Term::Constant("type".to_string()),
//!         object: Term::Constant("mortal".to_string()),
//!     }],
//! };
//! engine.add_rule(rule.clone());
//!
//! // Run inference with tracking
//! let fact = RuleAtom::Triple {
//!     subject: Term::Constant("socrates".to_string()),
//!     predicate: Term::Constant("type".to_string()),
//!     object: Term::Constant("human".to_string()),
//! };
//!
//! // Record the asserted fact
//! let fact_id = explainer.record_assertion(fact.clone());
//!
//! let facts = vec![fact];
//! let results = engine.forward_chain(&facts).unwrap();
//!
//! // Record the derived fact
//! let target = RuleAtom::Triple {
//!     subject: Term::Constant("socrates".to_string()),
//!     predicate: Term::Constant("type".to_string()),
//!     object: Term::Constant("mortal".to_string()),
//! };
//!
//! explainer.record_derivation(
//!     target.clone(),
//!     Some(rule),
//!     vec![fact_id],
//!     DerivationMethod::ForwardChaining
//! );
//!
//! let explanation = explainer.explain_why(&target).unwrap();
//! println!("Explanation: {}", explanation);
//! ```

use crate::{Rule, RuleAtom};
use anyhow::Result;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use tracing::{debug, info};

/// Unique identifier for derivation steps
type DerivationId = usize;

/// Derivation step in the reasoning process
#[derive(Debug, Clone)]
pub struct DerivationStep {
    /// Unique ID for this derivation
    pub id: DerivationId,
    /// The derived fact
    pub fact: RuleAtom,
    /// Rule that was applied (if any)
    pub rule: Option<Rule>,
    /// Facts that were used to derive this fact
    pub premises: Vec<DerivationId>,
    /// Timestamp of derivation
    pub timestamp: u64,
    /// Derivation method
    pub method: DerivationMethod,
}

/// Method used for derivation
#[derive(Debug, Clone, PartialEq)]
pub enum DerivationMethod {
    /// Asserted directly (axiom)
    Asserted,
    /// Derived via forward chaining
    ForwardChaining,
    /// Derived via backward chaining
    BackwardChaining,
    /// Derived via RETE network
    Rete,
    /// Derived via RDFS reasoning
    Rdfs,
    /// Derived via OWL reasoning
    Owl,
    /// Derived via SWRL reasoning
    Swrl,
}

impl fmt::Display for DerivationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DerivationMethod::Asserted => write!(f, "asserted"),
            DerivationMethod::ForwardChaining => write!(f, "forward chaining"),
            DerivationMethod::BackwardChaining => write!(f, "backward chaining"),
            DerivationMethod::Rete => write!(f, "RETE"),
            DerivationMethod::Rdfs => write!(f, "RDFS reasoning"),
            DerivationMethod::Owl => write!(f, "OWL reasoning"),
            DerivationMethod::Swrl => write!(f, "SWRL reasoning"),
        }
    }
}

/// Explanation for a derived fact
#[derive(Debug, Clone)]
pub struct Explanation {
    /// The fact being explained
    pub target: RuleAtom,
    /// Derivation steps leading to this fact
    pub steps: Vec<DerivationStep>,
    /// Justification (minimal supporting evidence)
    pub justification: Justification,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
}

impl fmt::Display for Explanation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Explanation for: {:?}", self.target)?;
        writeln!(f, "Confidence: {:.2}%", self.confidence * 100.0)?;
        writeln!(f, "\nDerivation steps:")?;
        for (i, step) in self.steps.iter().enumerate() {
            writeln!(f, "  {}. {:?} via {}", i + 1, step.fact, step.method)?;
            if let Some(rule) = &step.rule {
                writeln!(f, "     Rule: {}", rule.name)?;
            }
        }
        writeln!(f, "\nJustification:")?;
        writeln!(f, "{}", self.justification)?;
        Ok(())
    }
}

/// Justification providing minimal supporting evidence
#[derive(Debug, Clone)]
pub struct Justification {
    /// Axioms (asserted facts) needed
    pub axioms: Vec<RuleAtom>,
    /// Rules needed
    pub rules: Vec<Rule>,
    /// Inference chain
    pub chain: Vec<String>,
}

impl fmt::Display for Justification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  Axioms:")?;
        for axiom in &self.axioms {
            writeln!(f, "    - {:?}", axiom)?;
        }
        writeln!(f, "  Rules:")?;
        for rule in &self.rules {
            writeln!(f, "    - {}", rule.name)?;
        }
        writeln!(f, "  Inference chain:")?;
        for (i, step) in self.chain.iter().enumerate() {
            writeln!(f, "    {}. {}", i + 1, step)?;
        }
        Ok(())
    }
}

/// Request for explanation
#[derive(Debug, Clone)]
pub struct ExplanationRequest {
    /// Fact to explain
    pub target: RuleAtom,
    /// Type of explanation requested
    pub explanation_type: ExplanationType,
    /// Maximum depth of explanation
    pub max_depth: usize,
    /// Include all derivation paths or just one
    pub include_all_paths: bool,
}

/// Type of explanation
#[derive(Debug, Clone, PartialEq)]
pub enum ExplanationType {
    /// Why is this fact true?
    Why,
    /// How was this fact derived?
    How,
    /// What are all the ways to derive this fact?
    AllPaths,
    /// What is the minimal justification?
    Minimal,
}

/// Explanation engine
pub struct ExplanationEngine {
    /// Derivation history
    derivations: HashMap<DerivationId, DerivationStep>,
    /// Index: fact -> derivation IDs
    fact_index: HashMap<RuleAtom, Vec<DerivationId>>,
    /// Next derivation ID
    next_id: DerivationId,
    /// Current timestamp
    timestamp: u64,
}

impl Default for ExplanationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ExplanationEngine {
    /// Create a new explanation engine
    pub fn new() -> Self {
        Self {
            derivations: HashMap::new(),
            fact_index: HashMap::new(),
            next_id: 0,
            timestamp: 0,
        }
    }

    /// Record an asserted fact
    pub fn record_assertion(&mut self, fact: RuleAtom) -> DerivationId {
        let id = self.next_id;
        self.next_id += 1;
        self.timestamp += 1;

        let step = DerivationStep {
            id,
            fact: fact.clone(),
            rule: None,
            premises: vec![],
            timestamp: self.timestamp,
            method: DerivationMethod::Asserted,
        };

        self.derivations.insert(id, step);
        self.fact_index.entry(fact).or_default().push(id);

        debug!("Recorded assertion: ID {}", id);
        id
    }

    /// Record a derived fact
    pub fn record_derivation(
        &mut self,
        fact: RuleAtom,
        rule: Option<Rule>,
        premises: Vec<DerivationId>,
        method: DerivationMethod,
    ) -> DerivationId {
        let id = self.next_id;
        self.next_id += 1;
        self.timestamp += 1;

        let step = DerivationStep {
            id,
            fact: fact.clone(),
            rule,
            premises,
            timestamp: self.timestamp,
            method: method.clone(),
        };

        debug!("Recorded derivation: ID {} via {:?}", id, method);

        self.derivations.insert(id, step);
        self.fact_index.entry(fact).or_default().push(id);

        id
    }

    /// Explain why a fact is true
    pub fn explain_why(&self, target: &RuleAtom) -> Result<Explanation> {
        info!("Generating 'why' explanation for: {:?}", target);

        let request = ExplanationRequest {
            target: target.clone(),
            explanation_type: ExplanationType::Why,
            max_depth: 100,
            include_all_paths: false,
        };

        self.generate_explanation(&request)
    }

    /// Explain how a fact was derived
    pub fn explain_how(&self, target: &RuleAtom) -> Result<Explanation> {
        info!("Generating 'how' explanation for: {:?}", target);

        let request = ExplanationRequest {
            target: target.clone(),
            explanation_type: ExplanationType::How,
            max_depth: 100,
            include_all_paths: false,
        };

        self.generate_explanation(&request)
    }

    /// Find all derivation paths
    pub fn explain_all_paths(&self, target: &RuleAtom) -> Result<Explanation> {
        info!("Finding all derivation paths for: {:?}", target);

        let request = ExplanationRequest {
            target: target.clone(),
            explanation_type: ExplanationType::AllPaths,
            max_depth: 100,
            include_all_paths: true,
        };

        self.generate_explanation(&request)
    }

    /// Find minimal justification
    pub fn explain_minimal(&self, target: &RuleAtom) -> Result<Explanation> {
        info!("Finding minimal justification for: {:?}", target);

        let request = ExplanationRequest {
            target: target.clone(),
            explanation_type: ExplanationType::Minimal,
            max_depth: 100,
            include_all_paths: false,
        };

        self.generate_explanation(&request)
    }

    /// Generate explanation based on request
    fn generate_explanation(&self, request: &ExplanationRequest) -> Result<Explanation> {
        // Find derivation IDs for the target fact
        let derivation_ids = self
            .fact_index
            .get(&request.target)
            .ok_or_else(|| anyhow::anyhow!("Fact not found in derivation history"))?;

        if derivation_ids.is_empty() {
            return Err(anyhow::anyhow!("No derivations found for fact"));
        }

        // Build explanation from derivation graph
        let steps = self.collect_derivation_steps(derivation_ids[0], request.max_depth)?;
        let justification = self.build_justification(&steps)?;

        Ok(Explanation {
            target: request.target.clone(),
            steps,
            justification,
            confidence: 1.0, // TODO: Implement confidence calculation
        })
    }

    /// Collect all derivation steps leading to a fact
    fn collect_derivation_steps(
        &self,
        target_id: DerivationId,
        max_depth: usize,
    ) -> Result<Vec<DerivationStep>> {
        let mut steps = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back((target_id, 0));

        while let Some((id, depth)) = queue.pop_front() {
            if depth > max_depth || !visited.insert(id) {
                continue;
            }

            if let Some(step) = self.derivations.get(&id) {
                steps.push(step.clone());

                // Add premises to queue
                for &premise_id in &step.premises {
                    queue.push_back((premise_id, depth + 1));
                }
            }
        }

        // Sort by timestamp for logical order
        steps.sort_by_key(|s| s.timestamp);

        Ok(steps)
    }

    /// Build justification from derivation steps
    fn build_justification(&self, steps: &[DerivationStep]) -> Result<Justification> {
        let mut axioms = Vec::new();
        let mut rules = Vec::new();
        let mut chain = Vec::new();

        for step in steps {
            match step.method {
                DerivationMethod::Asserted => {
                    axioms.push(step.fact.clone());
                    chain.push(format!("Axiom: {:?}", step.fact));
                }
                _ => {
                    if let Some(rule) = &step.rule {
                        if !rules.iter().any(|r: &Rule| r.name == rule.name) {
                            rules.push(rule.clone());
                        }
                        chain.push(format!(
                            "Apply rule '{}' to derive {:?}",
                            rule.name, step.fact
                        ));
                    } else {
                        chain.push(format!("Derive {:?} via {}", step.fact, step.method));
                    }
                }
            }
        }

        Ok(Justification {
            axioms,
            rules,
            chain,
        })
    }

    /// Get derivation graph as DOT format for visualization
    pub fn to_dot(&self, target: &RuleAtom) -> Result<String> {
        let mut dot = String::from("digraph Derivation {\n");
        dot.push_str("  rankdir=BT;\n");
        dot.push_str("  node [shape=box];\n\n");

        // Find derivations for target
        if let Some(derivation_ids) = self.fact_index.get(target) {
            let mut visited = HashSet::new();

            for &id in derivation_ids {
                self.add_to_dot(id, &mut dot, &mut visited);
            }
        }

        dot.push_str("}\n");
        Ok(dot)
    }

    /// Add derivation to DOT graph recursively
    fn add_to_dot(&self, id: DerivationId, dot: &mut String, visited: &mut HashSet<DerivationId>) {
        if !visited.insert(id) {
            return;
        }

        if let Some(step) = self.derivations.get(&id) {
            // Add node
            let label = format!("{:?}", step.fact);
            let color = match step.method {
                DerivationMethod::Asserted => "lightblue",
                _ => "lightgreen",
            };

            dot.push_str(&format!(
                "  n{} [label=\"{}\", fillcolor={}, style=filled];\n",
                id, label, color
            ));

            // Add edges from premises
            for &premise_id in &step.premises {
                dot.push_str(&format!("  n{} -> n{};\n", premise_id, id));
                self.add_to_dot(premise_id, dot, visited);
            }
        }
    }

    /// Get statistics
    pub fn get_stats(&self) -> ExplanationStats {
        let asserted_count = self
            .derivations
            .values()
            .filter(|s| s.method == DerivationMethod::Asserted)
            .count();

        let derived_count = self.derivations.len() - asserted_count;

        ExplanationStats {
            total_derivations: self.derivations.len(),
            asserted_facts: asserted_count,
            derived_facts: derived_count,
            unique_facts: self.fact_index.len(),
        }
    }

    /// Clear all derivation history
    pub fn clear(&mut self) {
        self.derivations.clear();
        self.fact_index.clear();
        self.next_id = 0;
        self.timestamp = 0;
    }
}

/// Statistics about explanations
#[derive(Debug, Clone)]
pub struct ExplanationStats {
    pub total_derivations: usize,
    pub asserted_facts: usize,
    pub derived_facts: usize,
    pub unique_facts: usize,
}

impl fmt::Display for ExplanationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Derivations: {}, Asserted: {}, Derived: {}, Unique: {}",
            self.total_derivations, self.asserted_facts, self.derived_facts, self.unique_facts
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Term;

    #[test]
    fn test_record_assertion() {
        let mut engine = ExplanationEngine::new();

        let fact = RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("human".to_string()),
        };

        let id = engine.record_assertion(fact.clone());
        assert_eq!(id, 0);

        let step = engine.derivations.get(&id).unwrap();
        assert_eq!(step.method, DerivationMethod::Asserted);
    }

    #[test]
    fn test_record_derivation() {
        let mut engine = ExplanationEngine::new();

        let premise = RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("human".to_string()),
        };

        let premise_id = engine.record_assertion(premise);

        let derived = RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("mortal".to_string()),
        };

        let rule = Rule {
            name: "mortality".to_string(),
            body: vec![],
            head: vec![],
        };

        let derived_id = engine.record_derivation(
            derived,
            Some(rule),
            vec![premise_id],
            DerivationMethod::ForwardChaining,
        );

        assert_eq!(derived_id, 1);
    }

    #[test]
    fn test_explain_why() {
        let mut engine = ExplanationEngine::new();

        let fact = RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("human".to_string()),
        };

        engine.record_assertion(fact.clone());

        let explanation = engine.explain_why(&fact).unwrap();
        assert_eq!(explanation.steps.len(), 1);
        assert_eq!(explanation.steps[0].method, DerivationMethod::Asserted);
    }

    #[test]
    fn test_justification() {
        let mut engine = ExplanationEngine::new();

        let premise = RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("human".to_string()),
        };

        let premise_id = engine.record_assertion(premise.clone());

        let derived = RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("mortal".to_string()),
        };

        let rule = Rule {
            name: "mortality".to_string(),
            body: vec![],
            head: vec![],
        };

        engine.record_derivation(
            derived.clone(),
            Some(rule),
            vec![premise_id],
            DerivationMethod::ForwardChaining,
        );

        let explanation = engine.explain_why(&derived).unwrap();
        assert!(!explanation.justification.axioms.is_empty());
        assert!(!explanation.justification.rules.is_empty());
    }

    #[test]
    fn test_stats() {
        let mut engine = ExplanationEngine::new();

        engine.record_assertion(RuleAtom::Triple {
            subject: Term::Constant("a".to_string()),
            predicate: Term::Constant("p".to_string()),
            object: Term::Constant("b".to_string()),
        });

        let stats = engine.get_stats();
        assert_eq!(stats.asserted_facts, 1);
        assert_eq!(stats.total_derivations, 1);
    }
}
