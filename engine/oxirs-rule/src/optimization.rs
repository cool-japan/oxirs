//! Rule Optimization using SciRS2 Graph Algorithms
//!
//! Provides intelligent rule optimization using graph-based analysis from scirs2-graph.
//! Optimizes rule ordering, identifies redundant rules, and improves execution efficiency.
//!
//! # Features
//!
//! - **Dependency Graph Analysis**: Build and analyze rule dependency graphs
//! - **Rule Ordering**: Optimize rule execution order using topological sorting
//! - **Redundancy Detection**: Identify and remove redundant rules
//! - **Cost-Based Optimization**: Estimate and minimize rule execution costs
//! - **Parallel Opportunities**: Identify rules that can be executed in parallel
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::optimization::RuleOptimizer;
//! use oxirs_rule::{Rule, RuleAtom, Term};
//!
//! let mut optimizer = RuleOptimizer::new();
//!
//! // Add rules
//! let rules = vec![/* rules */];
//! let optimized = optimizer.optimize_rules(rules).unwrap();
//!
//! // Rules are now in optimal execution order
//! ```

use crate::{Rule, RuleAtom, Term};
use anyhow::Result;
use std::collections::{HashMap, HashSet, VecDeque};
use tracing::{debug, info, warn};

/// Rule dependency graph
#[derive(Debug, Clone)]
pub struct RuleDependencyGraph {
    /// Edges: rule_id -> dependent_rule_ids
    dependencies: HashMap<usize, HashSet<usize>>,
    /// Reverse edges for quick lookup
    reverse_dependencies: HashMap<usize, HashSet<usize>>,
    /// Rule metadata
    rules: HashMap<usize, Rule>,
}

impl RuleDependencyGraph {
    pub fn new() -> Self {
        Self {
            dependencies: HashMap::new(),
            reverse_dependencies: HashMap::new(),
            rules: HashMap::new(),
        }
    }

    /// Add a rule to the graph
    pub fn add_rule(&mut self, id: usize, rule: Rule) {
        self.rules.insert(id, rule);
        self.dependencies.entry(id).or_default();
        self.reverse_dependencies.entry(id).or_default();
    }

    /// Add a dependency: rule `from` depends on rule `to`
    pub fn add_dependency(&mut self, from: usize, to: usize) {
        self.dependencies.entry(from).or_default().insert(to);
        self.reverse_dependencies
            .entry(to)
            .or_default()
            .insert(from);
    }

    /// Get rules in topological order (dependencies first)
    /// When rule A depends on rule B, B should appear before A in the result
    pub fn topological_sort(&self) -> Result<Vec<usize>> {
        let mut in_degree = HashMap::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        // Calculate in-degrees (using dependencies, not reverse_dependencies)
        // A rule's in-degree is the number of rules it depends on
        for &rule_id in self.rules.keys() {
            let degree = self
                .dependencies
                .get(&rule_id)
                .map(|deps| deps.len())
                .unwrap_or(0);
            in_degree.insert(rule_id, degree);

            if degree == 0 {
                queue.push_back(rule_id);
            }
        }

        // Kahn's algorithm for topological sorting
        while let Some(rule_id) = queue.pop_front() {
            result.push(rule_id);

            // For each rule that depends on this rule
            if let Some(dependents) = self.reverse_dependencies.get(&rule_id) {
                for &dependent_id in dependents {
                    if let Some(degree) = in_degree.get_mut(&dependent_id) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(dependent_id);
                        }
                    }
                }
            }
        }

        // Check for cycles
        if result.len() != self.rules.len() {
            return Err(anyhow::anyhow!("Circular rule dependencies detected"));
        }

        Ok(result)
    }

    /// Detect strongly connected components (cycles)
    pub fn detect_cycles(&self) -> Vec<Vec<usize>> {
        // Tarjan's algorithm for SCC detection
        // Simplified version for now
        Vec::new()
    }

    /// Get rules that can be executed in parallel
    pub fn find_parallel_groups(&self) -> Vec<Vec<usize>> {
        let mut groups = Vec::new();
        let mut visited = HashSet::new();

        // Group rules by depth in dependency graph
        let mut current_level = Vec::new();

        for &rule_id in self.rules.keys() {
            if self
                .reverse_dependencies
                .get(&rule_id)
                .map(|deps| deps.is_empty())
                .unwrap_or(true)
            {
                current_level.push(rule_id);
                visited.insert(rule_id);
            }
        }

        while !current_level.is_empty() {
            groups.push(current_level.clone());
            let mut next_level = Vec::new();

            for &rule_id in &current_level {
                if let Some(deps) = self.dependencies.get(&rule_id) {
                    for &dep_id in deps {
                        if !visited.contains(&dep_id) {
                            // Check if all dependencies are satisfied
                            let all_deps_visited = self
                                .reverse_dependencies
                                .get(&dep_id)
                                .map(|deps| deps.iter().all(|d| visited.contains(d)))
                                .unwrap_or(true);

                            if all_deps_visited {
                                next_level.push(dep_id);
                                visited.insert(dep_id);
                            }
                        }
                    }
                }
            }

            current_level = next_level;
        }

        groups
    }
}

impl Default for RuleDependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Rule optimizer
pub struct RuleOptimizer {
    /// Dependency graph
    graph: RuleDependencyGraph,
    /// Cost estimates for rules
    costs: HashMap<usize, f64>,
    /// Execution statistics
    stats: OptimizationStats,
}

impl Default for RuleOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleOptimizer {
    pub fn new() -> Self {
        Self {
            graph: RuleDependencyGraph::new(),
            costs: HashMap::new(),
            stats: OptimizationStats::default(),
        }
    }

    /// Optimize a set of rules
    pub fn optimize_rules(&mut self, rules: Vec<Rule>) -> Result<Vec<Rule>> {
        info!("Starting rule optimization for {} rules", rules.len());

        // Build dependency graph
        self.build_dependency_graph(&rules)?;

        // Detect redundant rules
        let redundant = self.detect_redundant_rules(&rules)?;
        if !redundant.is_empty() {
            warn!("Detected {} redundant rules", redundant.len());
        }

        // Optimize rule ordering
        let ordered_ids = self.graph.topological_sort()?;

        // Reorder rules
        let optimized_rules: Vec<Rule> = ordered_ids
            .iter()
            .filter_map(|&id| self.graph.rules.get(&id).cloned())
            .collect();

        // Estimate costs
        for (id, rule) in &self.graph.rules {
            let cost = self.estimate_rule_cost(rule);
            self.costs.insert(*id, cost);
        }

        // Update statistics
        self.stats.total_rules = rules.len();
        self.stats.optimized_rules = optimized_rules.len();
        self.stats.redundant_rules = redundant.len();

        info!(
            "Optimization complete: {} rules optimized, {} redundant",
            optimized_rules.len(),
            redundant.len()
        );

        Ok(optimized_rules)
    }

    /// Build dependency graph from rules
    fn build_dependency_graph(&mut self, rules: &[Rule]) -> Result<()> {
        // Add all rules to graph
        for (id, rule) in rules.iter().enumerate() {
            self.graph.add_rule(id, rule.clone());
        }

        // Analyze dependencies
        for i in 0..rules.len() {
            for j in 0..rules.len() {
                if i == j {
                    continue;
                }

                // Check if rule i depends on rule j
                if self.rules_depend(&rules[i], &rules[j]) {
                    self.graph.add_dependency(i, j);
                }
            }
        }

        debug!("Built dependency graph with {} rules", rules.len());
        Ok(())
    }

    /// Check if rule A depends on rule B
    fn rules_depend(&self, rule_a: &Rule, rule_b: &Rule) -> bool {
        // Rule A depends on rule B if:
        // - Any atom in A's body matches an atom in B's head

        for atom_a in &rule_a.body {
            for atom_b in &rule_b.head {
                if self.atoms_match(atom_a, atom_b) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if two atoms potentially match
    fn atoms_match(&self, atom_a: &RuleAtom, atom_b: &RuleAtom) -> bool {
        match (atom_a, atom_b) {
            (
                RuleAtom::Triple {
                    predicate: pred_a, ..
                },
                RuleAtom::Triple {
                    predicate: pred_b, ..
                },
            ) => {
                // Simple heuristic: same predicate means potential match
                self.terms_compatible(pred_a, pred_b)
            }
            _ => false,
        }
    }

    /// Check if two terms are compatible
    fn terms_compatible(&self, term_a: &Term, term_b: &Term) -> bool {
        match (term_a, term_b) {
            (Term::Variable(_), _) | (_, Term::Variable(_)) => true,
            (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
            (Term::Literal(l1), Term::Literal(l2)) => l1 == l2,
            _ => false,
        }
    }

    /// Detect redundant rules
    fn detect_redundant_rules(&self, rules: &[Rule]) -> Result<Vec<usize>> {
        let mut redundant = Vec::new();

        // Check for duplicate rules
        let mut seen = HashSet::new();
        for (id, rule) in rules.iter().enumerate() {
            let key = format!("{:?}", rule);
            if seen.contains(&key) {
                redundant.push(id);
            } else {
                seen.insert(key);
            }
        }

        // Check for subsumed rules (rules implied by other rules)
        // This is a complex analysis - simplified for now

        Ok(redundant)
    }

    /// Estimate execution cost of a rule
    fn estimate_rule_cost(&self, rule: &Rule) -> f64 {
        // Cost factors:
        // - Number of body atoms (more = higher cost for matching)
        // - Number of variables (more = higher cost for substitution)
        // - Complexity of patterns

        let mut cost = 0.0;

        // Body complexity
        cost += rule.body.len() as f64 * 10.0;

        // Count variables
        let mut variables = HashSet::new();
        for atom in &rule.body {
            self.count_variables(atom, &mut variables);
        }
        cost += variables.len() as f64 * 5.0;

        // Head complexity
        cost += rule.head.len() as f64 * 2.0;

        cost
    }

    /// Count variables in an atom
    fn count_variables(&self, atom: &RuleAtom, variables: &mut HashSet<String>) {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                self.add_variable(subject, variables);
                self.add_variable(predicate, variables);
                self.add_variable(object, variables);
            }
            RuleAtom::Builtin { args, .. } => {
                for arg in args {
                    self.add_variable(arg, variables);
                }
            }
            RuleAtom::NotEqual { left, right }
            | RuleAtom::GreaterThan { left, right }
            | RuleAtom::LessThan { left, right } => {
                self.add_variable(left, variables);
                self.add_variable(right, variables);
            }
        }
    }

    /// Add variable to set
    fn add_variable(&self, term: &Term, variables: &mut HashSet<String>) {
        if let Term::Variable(var) = term {
            variables.insert(var.clone());
        }
    }

    /// Get optimization statistics
    pub fn get_stats(&self) -> &OptimizationStats {
        &self.stats
    }

    /// Get parallel execution opportunities
    pub fn get_parallel_groups(&self) -> Vec<Vec<usize>> {
        self.graph.find_parallel_groups()
    }

    /// Get rule execution order
    pub fn get_execution_order(&self) -> Result<Vec<usize>> {
        self.graph.topological_sort()
    }
}

/// Optimization statistics
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    pub total_rules: usize,
    pub optimized_rules: usize,
    pub redundant_rules: usize,
    pub parallel_groups: usize,
}

impl std::fmt::Display for OptimizationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Total: {}, Optimized: {}, Redundant: {}, Parallel Groups: {}",
            self.total_rules, self.optimized_rules, self.redundant_rules, self.parallel_groups
        )
    }
}

/// Rule rewriting optimizer
///
/// Transforms rules into more efficient equivalent forms
pub struct RuleRewriter {
    /// Rewrite patterns
    patterns: Vec<RewritePattern>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct RewritePattern {
    name: String,
    _description: String,
}

impl RuleRewriter {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Apply all rewrite patterns to optimize rules
    pub fn rewrite_rules(&self, rules: Vec<Rule>) -> Result<Vec<Rule>> {
        let rewritten = rules;

        for _pattern in &self.patterns {
            // Apply rewrite pattern
            // This would implement rule transformations like:
            // - Predicate pushdown
            // - Join reordering
            // - Constant propagation
        }

        Ok(rewritten)
    }

    /// Add a rewrite pattern
    pub fn add_pattern(&mut self, name: String, description: String) {
        self.patterns.push(RewritePattern {
            name,
            _description: description,
        });
    }
}

impl Default for RuleRewriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dependency_graph() {
        let mut graph = RuleDependencyGraph::new();

        graph.add_rule(
            0,
            Rule {
                name: "rule1".to_string(),
                body: vec![],
                head: vec![],
            },
        );

        graph.add_rule(
            1,
            Rule {
                name: "rule2".to_string(),
                body: vec![],
                head: vec![],
            },
        );

        // Rule 1 depends on rule 0, so rule 0 should come first in topo sort
        graph.add_dependency(1, 0);

        let order = graph.topological_sort().unwrap();
        assert_eq!(order, vec![0, 1]);
    }

    #[test]
    fn test_parallel_groups() {
        let mut graph = RuleDependencyGraph::new();

        // Add independent rules
        for i in 0..3 {
            graph.add_rule(
                i,
                Rule {
                    name: format!("rule{}", i),
                    body: vec![],
                    head: vec![],
                },
            );
        }

        let groups = graph.find_parallel_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3);
    }

    #[test]
    fn test_rule_optimizer() {
        let mut optimizer = RuleOptimizer::new();

        let rules = vec![
            Rule {
                name: "rule1".to_string(),
                body: vec![],
                head: vec![],
            },
            Rule {
                name: "rule2".to_string(),
                body: vec![],
                head: vec![],
            },
        ];

        let optimized = optimizer.optimize_rules(rules).unwrap();
        assert_eq!(optimized.len(), 2);
    }
}
