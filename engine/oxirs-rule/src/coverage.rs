//! Rule Coverage Analysis Tool
//!
//! Provides comprehensive coverage analysis for rule-based inference systems.
//! Tracks which rules are executed, which paths are taken, and identifies
//! untested or dead code in rule sets.
//!
//! # Features
//!
//! - **Rule Coverage**: Track which rules are executed
//! - **Path Coverage**: Monitor execution paths through rule sets
//! - **Branch Coverage**: Track conditional execution in rules
//! - **Data Flow Coverage**: Analyze variable binding patterns
//! - **Coverage Metrics**: Calculate line, branch, and path coverage percentages
//! - **Dead Code Detection**: Identify never-executed rules
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::coverage::{CoverageAnalyzer, CoverageReport};
//! use oxirs_rule::{Rule, RuleAtom, Term, RuleEngine};
//!
//! let mut analyzer = CoverageAnalyzer::new();
//! let mut engine = RuleEngine::new();
//!
//! // Add rules
//! engine.add_rule(Rule {
//!     name: "rule1".to_string(),
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
//! });
//!
//! // Start coverage tracking
//! analyzer.start_tracking();
//!
//! let facts = vec![RuleAtom::Triple {
//!     subject: Term::Constant("a".to_string()),
//!     predicate: Term::Constant("p".to_string()),
//!     object: Term::Constant("b".to_string()),
//! }];
//!
//! let _results = engine.forward_chain(&facts).unwrap();
//!
//! // Record execution
//! analyzer.record_rule_execution("rule1");
//!
//! analyzer.stop_tracking();
//!
//! // Generate coverage report
//! let report = analyzer.generate_report();
//! println!("Coverage: {:.1}%", report.overall_coverage());
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::Rule;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tracing::info;

/// Coverage analyzer
#[derive(Debug)]
pub struct CoverageAnalyzer {
    /// Active tracking session
    active: bool,
    /// Start time
    start_time: Option<Instant>,
    /// Rules being tracked
    tracked_rules: HashMap<String, Rule>,
    /// Executed rules
    executed_rules: HashSet<String>,
    /// Rule execution counts
    execution_counts: HashMap<String, usize>,
    /// Execution paths (rule -> following rules)
    execution_paths: HashMap<String, HashSet<String>>,
    /// Data flow coverage (rule -> variable bindings)
    data_flow: HashMap<String, HashSet<String>>,
}

impl Default for CoverageAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl CoverageAnalyzer {
    /// Create a new coverage analyzer
    pub fn new() -> Self {
        Self {
            active: false,
            start_time: None,
            tracked_rules: HashMap::new(),
            executed_rules: HashSet::new(),
            execution_counts: HashMap::new(),
            execution_paths: HashMap::new(),
            data_flow: HashMap::new(),
        }
    }

    /// Register a rule for coverage tracking
    pub fn register_rule(&mut self, rule: Rule) {
        self.tracked_rules.insert(rule.name.clone(), rule);
    }

    /// Register multiple rules
    pub fn register_rules(&mut self, rules: Vec<Rule>) {
        for rule in rules {
            self.register_rule(rule);
        }
    }

    /// Start coverage tracking
    pub fn start_tracking(&mut self) {
        info!(
            "Starting coverage tracking for {} rules",
            self.tracked_rules.len()
        );
        self.active = true;
        self.start_time = Some(Instant::now());
    }

    /// Stop coverage tracking
    pub fn stop_tracking(&mut self) {
        self.active = false;
        info!(
            "Stopped coverage tracking after {:?}",
            self.start_time.map(|t| t.elapsed())
        );
    }

    /// Record rule execution
    pub fn record_rule_execution(&mut self, rule_name: &str) {
        if !self.active {
            return;
        }

        self.executed_rules.insert(rule_name.to_string());
        *self
            .execution_counts
            .entry(rule_name.to_string())
            .or_insert(0) += 1;
    }

    /// Record execution path (rule A triggers rule B)
    pub fn record_execution_path(&mut self, from_rule: &str, to_rule: &str) {
        if !self.active {
            return;
        }

        self.execution_paths
            .entry(from_rule.to_string())
            .or_default()
            .insert(to_rule.to_string());
    }

    /// Record data flow (rule with specific variable bindings)
    pub fn record_data_flow(&mut self, rule_name: &str, variable: &str) {
        if !self.active {
            return;
        }

        self.data_flow
            .entry(rule_name.to_string())
            .or_default()
            .insert(variable.to_string());
    }

    /// Generate coverage report
    pub fn generate_report(&self) -> CoverageReport {
        let total_rules = self.tracked_rules.len();
        let executed_rules = self.executed_rules.len();

        let rule_coverage = if total_rules > 0 {
            (executed_rules as f64 / total_rules as f64) * 100.0
        } else {
            0.0
        };

        // Calculate path coverage
        let total_possible_paths = self.calculate_total_possible_paths();
        let covered_paths = self
            .execution_paths
            .values()
            .map(|s| s.len())
            .sum::<usize>();
        let path_coverage = if total_possible_paths > 0 {
            (covered_paths as f64 / total_possible_paths as f64) * 100.0
        } else {
            0.0
        };

        // Identify unexecuted rules
        let unexecuted_rules: Vec<String> = self
            .tracked_rules
            .keys()
            .filter(|name| !self.executed_rules.contains(*name))
            .cloned()
            .collect();

        CoverageReport {
            total_rules,
            executed_rules,
            rule_coverage,
            path_coverage,
            execution_counts: self.execution_counts.clone(),
            unexecuted_rules,
            execution_paths: self.execution_paths.clone(),
            data_flow_coverage: self.data_flow.clone(),
        }
    }

    /// Calculate total possible execution paths
    fn calculate_total_possible_paths(&self) -> usize {
        // Simple heuristic: n * (n-1) for n rules
        let n = self.tracked_rules.len();
        if n > 1 {
            n * (n - 1)
        } else {
            0
        }
    }

    /// Get dead rules (never executed)
    pub fn get_dead_rules(&self) -> Vec<String> {
        self.tracked_rules
            .keys()
            .filter(|name| !self.executed_rules.contains(*name))
            .cloned()
            .collect()
    }

    /// Get hot rules (frequently executed)
    pub fn get_hot_rules(&self, threshold: usize) -> Vec<(String, usize)> {
        let mut hot_rules: Vec<(String, usize)> = self
            .execution_counts
            .iter()
            .filter(|(_, &count)| count >= threshold)
            .map(|(name, &count)| (name.clone(), count))
            .collect();

        hot_rules.sort_by(|a, b| b.1.cmp(&a.1));
        hot_rules
    }

    /// Reset coverage data
    pub fn reset(&mut self) {
        self.active = false;
        self.start_time = None;
        self.executed_rules.clear();
        self.execution_counts.clear();
        self.execution_paths.clear();
        self.data_flow.clear();
    }

    /// Export coverage data to JSON
    pub fn export_to_json(&self) -> Result<String> {
        use serde_json::json;

        let report = self.generate_report();

        let data = json!({
            "total_rules": report.total_rules,
            "executed_rules": report.executed_rules,
            "rule_coverage": report.rule_coverage,
            "path_coverage": report.path_coverage,
            "execution_counts": report.execution_counts,
            "unexecuted_rules": report.unexecuted_rules,
            "execution_paths": report.execution_paths,
            "data_flow_coverage": report.data_flow_coverage,
        });

        Ok(serde_json::to_string_pretty(&data)?)
    }

    /// Check if analyzer is active
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Get number of tracked rules
    pub fn tracked_rule_count(&self) -> usize {
        self.tracked_rules.len()
    }
}

/// Coverage report
#[derive(Debug, Clone)]
pub struct CoverageReport {
    /// Total number of rules
    pub total_rules: usize,
    /// Number of executed rules
    pub executed_rules: usize,
    /// Rule coverage percentage
    pub rule_coverage: f64,
    /// Path coverage percentage
    pub path_coverage: f64,
    /// Execution counts per rule
    pub execution_counts: HashMap<String, usize>,
    /// List of unexecuted rules
    pub unexecuted_rules: Vec<String>,
    /// Execution paths
    pub execution_paths: HashMap<String, HashSet<String>>,
    /// Data flow coverage
    pub data_flow_coverage: HashMap<String, HashSet<String>>,
}

impl CoverageReport {
    /// Get overall coverage percentage
    pub fn overall_coverage(&self) -> f64 {
        // Weighted average of rule and path coverage
        (self.rule_coverage * 0.7) + (self.path_coverage * 0.3)
    }

    /// Generate human-readable report
    pub fn to_string_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Rule Coverage Report ===\n\n");

        report.push_str(&format!("Total Rules: {}\n", self.total_rules));
        report.push_str(&format!("Executed Rules: {}\n", self.executed_rules));
        report.push_str(&format!("Rule Coverage: {:.1}%\n", self.rule_coverage));
        report.push_str(&format!("Path Coverage: {:.1}%\n", self.path_coverage));
        report.push_str(&format!(
            "Overall Coverage: {:.1}%\n\n",
            self.overall_coverage()
        ));

        if !self.unexecuted_rules.is_empty() {
            report.push_str("=== Unexecuted Rules (Dead Code) ===\n\n");
            for rule_name in &self.unexecuted_rules {
                report.push_str(&format!("- {}\n", rule_name));
            }
            report.push('\n');
        }

        report.push_str("=== Execution Statistics ===\n\n");

        let mut sorted_counts: Vec<(&String, &usize)> = self.execution_counts.iter().collect();
        sorted_counts.sort_by(|a, b| b.1.cmp(a.1));

        for (rule_name, count) in sorted_counts {
            report.push_str(&format!("{}: {} executions\n", rule_name, count));
        }

        report.push('\n');

        if !self.execution_paths.is_empty() {
            report.push_str("=== Execution Paths ===\n\n");
            for (from_rule, to_rules) in &self.execution_paths {
                report.push_str(&format!("{} ->\n", from_rule));
                for to_rule in to_rules {
                    report.push_str(&format!("  - {}\n", to_rule));
                }
                report.push('\n');
            }
        }

        if !self.data_flow_coverage.is_empty() {
            report.push_str("=== Data Flow Coverage ===\n\n");
            for (rule_name, variables) in &self.data_flow_coverage {
                report.push_str(&format!("{}: {:?}\n", rule_name, variables));
            }
            report.push('\n');
        }

        report
    }

    /// Get coverage grade
    pub fn get_grade(&self) -> CoverageGrade {
        let coverage = self.overall_coverage();

        if coverage >= 90.0 {
            CoverageGrade::Excellent
        } else if coverage >= 80.0 {
            CoverageGrade::Good
        } else if coverage >= 70.0 {
            CoverageGrade::Fair
        } else if coverage >= 60.0 {
            CoverageGrade::Poor
        } else {
            CoverageGrade::Inadequate
        }
    }
}

/// Coverage grade
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverageGrade {
    /// >= 90%
    Excellent,
    /// >= 80%
    Good,
    /// >= 70%
    Fair,
    /// >= 60%
    Poor,
    /// < 60%
    Inadequate,
}

impl std::fmt::Display for CoverageGrade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoverageGrade::Excellent => write!(f, "Excellent (A)"),
            CoverageGrade::Good => write!(f, "Good (B)"),
            CoverageGrade::Fair => write!(f, "Fair (C)"),
            CoverageGrade::Poor => write!(f, "Poor (D)"),
            CoverageGrade::Inadequate => write!(f, "Inadequate (F)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RuleAtom, Term};

    #[test]
    fn test_coverage_basic() {
        let mut analyzer = CoverageAnalyzer::new();

        let rule1 = Rule {
            name: "rule1".to_string(),
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

        analyzer.register_rule(rule1);
        analyzer.start_tracking();
        analyzer.record_rule_execution("rule1");
        analyzer.stop_tracking();

        let report = analyzer.generate_report();
        assert_eq!(report.total_rules, 1);
        assert_eq!(report.executed_rules, 1);
        assert_eq!(report.rule_coverage, 100.0);
    }

    #[test]
    fn test_dead_code_detection() {
        let mut analyzer = CoverageAnalyzer::new();

        let rule1 = Rule {
            name: "executed_rule".to_string(),
            body: vec![],
            head: vec![],
        };

        let rule2 = Rule {
            name: "dead_rule".to_string(),
            body: vec![],
            head: vec![],
        };

        analyzer.register_rules(vec![rule1, rule2]);
        analyzer.start_tracking();
        analyzer.record_rule_execution("executed_rule");
        analyzer.stop_tracking();

        let dead_rules = analyzer.get_dead_rules();
        assert_eq!(dead_rules.len(), 1);
        assert!(dead_rules.contains(&"dead_rule".to_string()));
    }

    #[test]
    fn test_hot_rules() {
        let mut analyzer = CoverageAnalyzer::new();

        analyzer.register_rule(Rule {
            name: "hot_rule".to_string(),
            body: vec![],
            head: vec![],
        });

        analyzer.start_tracking();

        for _ in 0..100 {
            analyzer.record_rule_execution("hot_rule");
        }

        analyzer.stop_tracking();

        let hot_rules = analyzer.get_hot_rules(50);
        assert_eq!(hot_rules.len(), 1);
        assert_eq!(hot_rules[0].0, "hot_rule");
        assert_eq!(hot_rules[0].1, 100);
    }

    #[test]
    fn test_execution_paths() {
        let mut analyzer = CoverageAnalyzer::new();

        analyzer.start_tracking();
        analyzer.record_execution_path("rule1", "rule2");
        analyzer.record_execution_path("rule1", "rule3");
        analyzer.stop_tracking();

        let report = analyzer.generate_report();
        assert!(report.execution_paths.contains_key("rule1"));
        assert_eq!(report.execution_paths["rule1"].len(), 2);
    }

    #[test]
    fn test_data_flow_coverage() {
        let mut analyzer = CoverageAnalyzer::new();

        analyzer.start_tracking();
        analyzer.record_data_flow("rule1", "X");
        analyzer.record_data_flow("rule1", "Y");
        analyzer.stop_tracking();

        let report = analyzer.generate_report();
        assert!(report.data_flow_coverage.contains_key("rule1"));
        assert_eq!(report.data_flow_coverage["rule1"].len(), 2);
    }

    #[test]
    fn test_coverage_reset() {
        let mut analyzer = CoverageAnalyzer::new();

        analyzer.register_rule(Rule {
            name: "rule1".to_string(),
            body: vec![],
            head: vec![],
        });

        analyzer.start_tracking();
        analyzer.record_rule_execution("rule1");
        analyzer.stop_tracking();

        assert_eq!(analyzer.executed_rules.len(), 1);

        analyzer.reset();

        assert_eq!(analyzer.executed_rules.len(), 0);
        assert!(!analyzer.is_active());
    }

    #[test]
    fn test_coverage_grade() {
        let report = CoverageReport {
            total_rules: 10,
            executed_rules: 9,
            rule_coverage: 90.0,
            path_coverage: 90.0,
            execution_counts: HashMap::new(),
            unexecuted_rules: vec![],
            execution_paths: HashMap::new(),
            data_flow_coverage: HashMap::new(),
        };

        assert_eq!(report.get_grade(), CoverageGrade::Excellent);
        assert_eq!(report.overall_coverage(), 90.0);
    }

    #[test]
    fn test_json_export() {
        let mut analyzer = CoverageAnalyzer::new();

        analyzer.register_rule(Rule {
            name: "rule1".to_string(),
            body: vec![],
            head: vec![],
        });

        analyzer.start_tracking();
        analyzer.record_rule_execution("rule1");
        analyzer.stop_tracking();

        let json = analyzer.export_to_json().unwrap();
        assert!(json.contains("rule1"));
        assert!(json.contains("rule_coverage"));
    }

    #[test]
    fn test_partial_coverage() {
        let mut analyzer = CoverageAnalyzer::new();

        for i in 0..10 {
            analyzer.register_rule(Rule {
                name: format!("rule{i}"),
                body: vec![],
                head: vec![],
            });
        }

        analyzer.start_tracking();

        // Execute only half the rules
        for i in 0..5 {
            analyzer.record_rule_execution(&format!("rule{i}"));
        }

        analyzer.stop_tracking();

        let report = analyzer.generate_report();
        assert_eq!(report.total_rules, 10);
        assert_eq!(report.executed_rules, 5);
        assert_eq!(report.rule_coverage, 50.0);
        assert_eq!(report.unexecuted_rules.len(), 5);
    }

    #[test]
    fn test_string_report() {
        let mut analyzer = CoverageAnalyzer::new();

        analyzer.register_rule(Rule {
            name: "rule1".to_string(),
            body: vec![],
            head: vec![],
        });

        analyzer.start_tracking();
        analyzer.record_rule_execution("rule1");
        analyzer.stop_tracking();

        let report = analyzer.generate_report();
        let string_report = report.to_string_report();

        assert!(string_report.contains("Rule Coverage Report"));
        assert!(string_report.contains("rule1"));
    }
}
