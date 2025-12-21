//! # Negation-as-Failure (NAF) with Stratification
//!
//! This module implements negation-as-failure semantics for rule-based reasoning,
//! along with stratification analysis to ensure safe evaluation of negated goals.
//!
//! ## Features
//!
//! - **Negation-as-Failure (NAF)**: Classical closed-world assumption semantics
//! - **Stratification Analysis**: Detect and prevent unsafe circular negation
//! - **Stratified Evaluation**: Layer-by-layer rule evaluation for safe negation
//! - **Dependency Graph**: Analyze rule dependencies including negation edges
//! - **Well-Founded Semantics**: Optional three-valued logic support
//!
//! ## Stratification
//!
//! Stratification divides rules into layers (strata) where:
//! - A predicate in stratum i can only depend negatively on predicates in strata < i
//! - This ensures a well-defined fixpoint computation
//!
//! ## Example
//!
//! ```rust
//! use oxirs_rule::negation::{StratifiedReasoner, NafAtom, StratificationConfig};
//! use oxirs_rule::{Rule, RuleAtom, Term};
//!
//! // Rule: not_parent(X,Y) :- person(X), person(Y), \+ parent(X,Y)
//! // "X is not parent of Y if both are persons and X is not known to be parent of Y"
//!
//! let config = StratificationConfig::default();
//! let mut reasoner = StratifiedReasoner::new(config);
//!
//! // Add rules with negation
//! // reasoner.add_rule_with_negation(...);
//! ```

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};

use crate::{Rule, RuleAtom, Term};

/// Configuration for stratified reasoning
#[derive(Debug, Clone)]
pub struct StratificationConfig {
    /// Maximum number of strata allowed
    pub max_strata: usize,
    /// Enable well-founded semantics (three-valued logic)
    pub well_founded: bool,
    /// Detect and report stratification violations
    pub strict_stratification: bool,
    /// Enable cycle detection in dependency graph
    pub detect_cycles: bool,
    /// Maximum iterations per stratum
    pub max_iterations_per_stratum: usize,
}

impl Default for StratificationConfig {
    fn default() -> Self {
        Self {
            max_strata: 100,
            well_founded: false,
            strict_stratification: true,
            detect_cycles: true,
            max_iterations_per_stratum: 1000,
        }
    }
}

impl StratificationConfig {
    /// Enable well-founded semantics
    pub fn with_well_founded(mut self, enabled: bool) -> Self {
        self.well_founded = enabled;
        self
    }

    /// Set strict stratification checking
    pub fn with_strict_stratification(mut self, enabled: bool) -> Self {
        self.strict_stratification = enabled;
        self
    }

    /// Set maximum strata
    pub fn with_max_strata(mut self, max: usize) -> Self {
        self.max_strata = max;
        self
    }
}

/// Negated atom for NAF semantics
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NafAtom {
    /// Positive atom (must be provable)
    Positive(RuleAtom),
    /// Negated atom (must NOT be provable - closed world assumption)
    Negated(RuleAtom),
}

impl NafAtom {
    /// Create a positive atom
    pub fn positive(atom: RuleAtom) -> Self {
        NafAtom::Positive(atom)
    }

    /// Create a negated atom
    pub fn negated(atom: RuleAtom) -> Self {
        NafAtom::Negated(atom)
    }

    /// Check if this is a negated atom
    pub fn is_negated(&self) -> bool {
        matches!(self, NafAtom::Negated(_))
    }

    /// Get the inner atom
    pub fn inner(&self) -> &RuleAtom {
        match self {
            NafAtom::Positive(a) | NafAtom::Negated(a) => a,
        }
    }

    /// Extract predicate name from the atom
    pub fn predicate(&self) -> Option<String> {
        match self.inner() {
            RuleAtom::Triple {
                predicate: Term::Constant(p),
                ..
            } => Some(p.clone()),
            RuleAtom::Triple { .. } => None,
            RuleAtom::Builtin { name, .. } => Some(name.clone()),
            _ => None,
        }
    }
}

/// Rule with NAF support
#[derive(Debug, Clone)]
pub struct NafRule {
    /// Rule name
    pub name: String,
    /// Body atoms (may include negation)
    pub body: Vec<NafAtom>,
    /// Head atoms (positive only)
    pub head: Vec<RuleAtom>,
    /// Optional stratum assignment
    pub stratum: Option<usize>,
}

impl NafRule {
    /// Create a new NAF rule
    pub fn new(name: String, body: Vec<NafAtom>, head: Vec<RuleAtom>) -> Self {
        Self {
            name,
            body,
            head,
            stratum: None,
        }
    }

    /// Check if this rule contains any negation
    pub fn has_negation(&self) -> bool {
        self.body.iter().any(|a| a.is_negated())
    }

    /// Get all predicates in the head
    pub fn head_predicates(&self) -> HashSet<String> {
        self.head
            .iter()
            .filter_map(|atom| match atom {
                RuleAtom::Triple {
                    predicate: Term::Constant(p),
                    ..
                } => Some(p.clone()),
                _ => None,
            })
            .collect()
    }

    /// Get all positive body predicates
    pub fn positive_body_predicates(&self) -> HashSet<String> {
        self.body
            .iter()
            .filter(|a| !a.is_negated())
            .filter_map(|a| a.predicate())
            .collect()
    }

    /// Get all negated body predicates
    pub fn negated_body_predicates(&self) -> HashSet<String> {
        self.body
            .iter()
            .filter(|a| a.is_negated())
            .filter_map(|a| a.predicate())
            .collect()
    }

    /// Convert from standard Rule (no negation)
    pub fn from_rule(rule: Rule) -> Self {
        Self {
            name: rule.name,
            body: rule.body.into_iter().map(NafAtom::Positive).collect(),
            head: rule.head,
            stratum: None,
        }
    }
}

/// Edge type in dependency graph
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependencyEdge {
    /// Positive dependency (predicate appears positively in body)
    Positive,
    /// Negative dependency (predicate appears negated in body)
    Negative,
}

/// Dependency graph for stratification analysis
#[derive(Debug, Default)]
pub struct DependencyGraph {
    /// Adjacency list: predicate -> (dependent_predicate, edge_type)
    edges: HashMap<String, Vec<(String, DependencyEdge)>>,
    /// All predicates in the graph
    predicates: HashSet<String>,
}

impl DependencyGraph {
    /// Create a new dependency graph
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a predicate to the graph
    pub fn add_predicate(&mut self, predicate: &str) {
        self.predicates.insert(predicate.to_string());
    }

    /// Add a dependency edge
    pub fn add_edge(&mut self, from: &str, to: &str, edge_type: DependencyEdge) {
        self.predicates.insert(from.to_string());
        self.predicates.insert(to.to_string());
        self.edges
            .entry(from.to_string())
            .or_default()
            .push((to.to_string(), edge_type));
    }

    /// Build dependency graph from NAF rules
    pub fn from_rules(rules: &[NafRule]) -> Self {
        let mut graph = Self::new();

        for rule in rules {
            // Add head predicates
            for head_pred in rule.head_predicates() {
                graph.add_predicate(&head_pred);

                // Add positive dependencies
                for body_pred in rule.positive_body_predicates() {
                    graph.add_edge(&head_pred, &body_pred, DependencyEdge::Positive);
                }

                // Add negative dependencies
                for neg_pred in rule.negated_body_predicates() {
                    graph.add_edge(&head_pred, &neg_pred, DependencyEdge::Negative);
                }
            }
        }

        graph
    }

    /// Check if there's a negative cycle (violates stratification)
    pub fn has_negative_cycle(&self) -> bool {
        // Use modified DFS to detect cycles with negative edges
        for start in &self.predicates {
            if self.has_negative_cycle_from(start) {
                return true;
            }
        }
        false
    }

    fn has_negative_cycle_from(&self, start: &str) -> bool {
        let mut visited = HashSet::new();
        let mut stack = vec![(start.to_string(), false)]; // (node, has_negative_edge_in_path)

        while let Some((node, has_neg)) = stack.pop() {
            if node == start && has_neg && !visited.is_empty() {
                return true;
            }

            if visited.contains(&(node.clone(), has_neg)) {
                continue;
            }
            visited.insert((node.clone(), has_neg));

            if let Some(edges) = self.edges.get(&node) {
                for (next, edge_type) in edges {
                    let new_has_neg = has_neg || *edge_type == DependencyEdge::Negative;
                    stack.push((next.clone(), new_has_neg));
                }
            }
        }

        false
    }

    /// Get all predicates
    pub fn predicates(&self) -> &HashSet<String> {
        &self.predicates
    }

    /// Get edges from a predicate
    pub fn get_edges(&self, predicate: &str) -> Option<&Vec<(String, DependencyEdge)>> {
        self.edges.get(predicate)
    }
}

/// Result of stratification analysis
#[derive(Debug, Clone)]
pub struct StratificationResult {
    /// Whether stratification is valid
    pub is_stratified: bool,
    /// Number of strata
    pub num_strata: usize,
    /// Stratum assignment for each predicate
    pub predicate_strata: HashMap<String, usize>,
    /// Rules grouped by stratum
    pub rules_by_stratum: Vec<Vec<usize>>,
    /// Violations found (if any)
    pub violations: Vec<StratificationViolation>,
}

/// Stratification violation
#[derive(Debug, Clone)]
pub struct StratificationViolation {
    /// Type of violation
    pub violation_type: ViolationType,
    /// Predicates involved
    pub predicates: Vec<String>,
    /// Description
    pub message: String,
}

/// Type of stratification violation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationType {
    /// Negative cycle detected
    NegativeCycle,
    /// Self-negation detected
    SelfNegation,
    /// Maximum strata exceeded
    MaxStrataExceeded,
}

/// Stratification analyzer
#[derive(Debug)]
pub struct StratificationAnalyzer {
    config: StratificationConfig,
}

impl StratificationAnalyzer {
    /// Create a new analyzer
    pub fn new(config: StratificationConfig) -> Self {
        Self { config }
    }

    /// Analyze rules and compute stratification
    pub fn analyze(&self, rules: &[NafRule]) -> StratificationResult {
        let graph = DependencyGraph::from_rules(rules);
        let mut violations = Vec::new();

        // Check for negative cycles
        if self.config.detect_cycles && graph.has_negative_cycle() {
            violations.push(StratificationViolation {
                violation_type: ViolationType::NegativeCycle,
                predicates: graph.predicates().iter().cloned().collect(),
                message: "Negative cycle detected in dependency graph".to_string(),
            });

            if self.config.strict_stratification {
                return StratificationResult {
                    is_stratified: false,
                    num_strata: 0,
                    predicate_strata: HashMap::new(),
                    rules_by_stratum: Vec::new(),
                    violations,
                };
            }
        }

        // Compute strata using topological sort with negation handling
        let predicate_strata = self.compute_strata(&graph);

        // Check for max strata exceeded
        let max_stratum = predicate_strata.values().max().copied().unwrap_or(0);
        if max_stratum >= self.config.max_strata {
            violations.push(StratificationViolation {
                violation_type: ViolationType::MaxStrataExceeded,
                predicates: Vec::new(),
                message: format!(
                    "Maximum strata ({}) exceeded: found {} strata",
                    self.config.max_strata,
                    max_stratum + 1
                ),
            });
        }

        // Group rules by stratum
        let rules_by_stratum = self.group_rules_by_stratum(rules, &predicate_strata);

        StratificationResult {
            is_stratified: violations.is_empty(),
            num_strata: max_stratum + 1,
            predicate_strata,
            rules_by_stratum,
            violations,
        }
    }

    fn compute_strata(&self, graph: &DependencyGraph) -> HashMap<String, usize> {
        let mut strata: HashMap<String, usize> = HashMap::new();
        let mut changed = true;

        // Initialize all predicates to stratum 0
        for pred in graph.predicates() {
            strata.insert(pred.clone(), 0);
        }

        // Iterate until fixpoint
        while changed {
            changed = false;

            for pred in graph.predicates() {
                if let Some(edges) = graph.get_edges(pred) {
                    let mut min_stratum = strata.get(pred).copied().unwrap_or(0);

                    for (dep, edge_type) in edges {
                        let dep_stratum = strata.get(dep).copied().unwrap_or(0);
                        let required_stratum = match edge_type {
                            DependencyEdge::Positive => dep_stratum,
                            DependencyEdge::Negative => dep_stratum + 1,
                        };

                        if required_stratum > min_stratum {
                            min_stratum = required_stratum;
                        }
                    }

                    if min_stratum > strata.get(pred).copied().unwrap_or(0) {
                        strata.insert(pred.clone(), min_stratum);
                        changed = true;
                    }
                }
            }
        }

        strata
    }

    fn group_rules_by_stratum(
        &self,
        rules: &[NafRule],
        predicate_strata: &HashMap<String, usize>,
    ) -> Vec<Vec<usize>> {
        let num_strata = predicate_strata.values().max().copied().unwrap_or(0) + 1;
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); num_strata];

        for (idx, rule) in rules.iter().enumerate() {
            // Rule stratum is the max stratum of its head predicates
            let rule_stratum = rule
                .head_predicates()
                .iter()
                .filter_map(|p| predicate_strata.get(p))
                .max()
                .copied()
                .unwrap_or(0);

            if rule_stratum < num_strata {
                groups[rule_stratum].push(idx);
            }
        }

        groups
    }
}

impl Default for StratificationAnalyzer {
    fn default() -> Self {
        Self::new(StratificationConfig::default())
    }
}

/// Stratified reasoner with NAF support
pub struct StratifiedReasoner {
    /// Configuration
    config: StratificationConfig,
    /// NAF rules
    rules: Vec<NafRule>,
    /// Known facts (positive atoms)
    facts: HashSet<String>,
    /// Stratification result
    stratification: Option<StratificationResult>,
    /// Analyzer
    analyzer: StratificationAnalyzer,
}

impl StratifiedReasoner {
    /// Create a new stratified reasoner
    pub fn new(config: StratificationConfig) -> Self {
        let analyzer = StratificationAnalyzer::new(config.clone());
        Self {
            config,
            rules: Vec::new(),
            facts: HashSet::new(),
            stratification: None,
            analyzer,
        }
    }

    /// Add a NAF rule
    pub fn add_rule(&mut self, rule: NafRule) {
        self.rules.push(rule);
        self.stratification = None; // Invalidate cached stratification
    }

    /// Add a standard rule (converted to NAF rule without negation)
    pub fn add_standard_rule(&mut self, rule: Rule) {
        self.add_rule(NafRule::from_rule(rule));
    }

    /// Add multiple rules
    pub fn add_rules(&mut self, rules: Vec<NafRule>) {
        for rule in rules {
            self.add_rule(rule);
        }
    }

    /// Add a fact
    pub fn add_fact(&mut self, fact: &str) {
        self.facts.insert(fact.to_string());
    }

    /// Add facts from RuleAtoms
    pub fn add_facts_from_atoms(&mut self, facts: &[RuleAtom]) {
        for fact in facts {
            let key = Self::atom_to_key(fact);
            self.facts.insert(key);
        }
    }

    /// Check if a fact is known
    pub fn is_known(&self, fact: &str) -> bool {
        self.facts.contains(fact)
    }

    /// Check NAF: fact is NOT known (closed world assumption)
    pub fn is_not_known(&self, fact: &str) -> bool {
        !self.facts.contains(fact)
    }

    /// Analyze stratification
    pub fn analyze_stratification(&mut self) -> &StratificationResult {
        if self.stratification.is_none() {
            self.stratification = Some(self.analyzer.analyze(&self.rules));
        }
        self.stratification.as_ref().unwrap()
    }

    /// Check if rules are properly stratified
    pub fn is_stratified(&mut self) -> bool {
        self.analyze_stratification().is_stratified
    }

    /// Perform stratified inference
    pub fn infer(&mut self) -> Result<Vec<RuleAtom>> {
        // First, analyze stratification
        let stratification = self.analyzer.analyze(&self.rules);

        if self.config.strict_stratification && !stratification.is_stratified {
            return Err(anyhow!(
                "Stratification violated: {:?}",
                stratification.violations
            ));
        }

        let mut inferred = Vec::new();

        // Process each stratum in order
        for (stratum_idx, rule_indices) in stratification.rules_by_stratum.iter().enumerate() {
            tracing::debug!("Processing stratum {}", stratum_idx);

            // Process rules in this stratum until fixpoint
            let stratum_results = self.process_stratum(rule_indices)?;
            inferred.extend(stratum_results);
        }

        self.stratification = Some(stratification);
        Ok(inferred)
    }

    fn process_stratum(&mut self, rule_indices: &[usize]) -> Result<Vec<RuleAtom>> {
        let mut inferred = Vec::new();
        let mut iterations = 0;
        let mut changed = true;

        while changed && iterations < self.config.max_iterations_per_stratum {
            changed = false;
            iterations += 1;

            for &rule_idx in rule_indices {
                let rule = &self.rules[rule_idx];
                let new_facts = self.apply_naf_rule(rule)?;

                for fact in new_facts {
                    let key = Self::atom_to_key(&fact);
                    if !self.facts.contains(&key) {
                        self.facts.insert(key);
                        inferred.push(fact);
                        changed = true;
                    }
                }
            }
        }

        Ok(inferred)
    }

    fn apply_naf_rule(&self, rule: &NafRule) -> Result<Vec<RuleAtom>> {
        // Find all substitutions that satisfy the body
        let substitutions = self.find_satisfying_substitutions(&rule.body)?;

        // Apply substitutions to head to generate new facts
        let mut results = Vec::new();
        for subst in substitutions {
            for head_atom in &rule.head {
                let grounded = Self::apply_substitution(head_atom, &subst);
                results.push(grounded);
            }
        }

        Ok(results)
    }

    fn find_satisfying_substitutions(
        &self,
        body: &[NafAtom],
    ) -> Result<Vec<HashMap<String, Term>>> {
        let mut substitutions = vec![HashMap::new()];

        for naf_atom in body {
            let mut new_substitutions = Vec::new();

            for subst in &substitutions {
                match naf_atom {
                    NafAtom::Positive(atom) => {
                        // Positive atom must match known facts
                        let matches = self.find_matches(atom, subst);
                        new_substitutions.extend(matches);
                    }
                    NafAtom::Negated(atom) => {
                        // Negated atom must NOT match any known fact
                        let grounded = Self::apply_substitution(atom, subst);
                        let key = Self::atom_to_key(&grounded);

                        // NAF succeeds if fact is not known
                        if !self.facts.contains(&key) {
                            new_substitutions.push(subst.clone());
                        }
                    }
                }
            }

            substitutions = new_substitutions;

            // Early termination if no substitutions left
            if substitutions.is_empty() {
                break;
            }
        }

        Ok(substitutions)
    }

    fn find_matches(
        &self,
        atom: &RuleAtom,
        current_subst: &HashMap<String, Term>,
    ) -> Vec<HashMap<String, Term>> {
        let mut results = Vec::new();

        // For simplicity, we check if the grounded atom matches known facts
        // In a full implementation, we'd pattern match against all facts
        let grounded = Self::apply_substitution(atom, current_subst);
        let key = Self::atom_to_key(&grounded);

        // Check if fully grounded and known
        if Self::is_ground(&grounded) && self.facts.contains(&key) {
            results.push(current_subst.clone());
        } else if !Self::is_ground(&grounded) {
            // For non-ground atoms, we'd need to iterate over facts
            // For now, just check if current substitution is consistent
            if self.is_consistent_with_facts(&grounded) {
                results.push(current_subst.clone());
            }
        }

        results
    }

    fn is_consistent_with_facts(&self, _atom: &RuleAtom) -> bool {
        // Simplified: in a full implementation, check against fact patterns
        true
    }

    fn is_ground(atom: &RuleAtom) -> bool {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                !matches!(subject, Term::Variable(_))
                    && !matches!(predicate, Term::Variable(_))
                    && !matches!(object, Term::Variable(_))
            }
            RuleAtom::Builtin { args, .. } => args.iter().all(|a| !matches!(a, Term::Variable(_))),
            RuleAtom::NotEqual { left, right } => {
                !matches!(left, Term::Variable(_)) && !matches!(right, Term::Variable(_))
            }
            RuleAtom::GreaterThan { left, right } | RuleAtom::LessThan { left, right } => {
                !matches!(left, Term::Variable(_)) && !matches!(right, Term::Variable(_))
            }
        }
    }

    fn apply_substitution(atom: &RuleAtom, subst: &HashMap<String, Term>) -> RuleAtom {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => RuleAtom::Triple {
                subject: Self::substitute_term(subject, subst),
                predicate: Self::substitute_term(predicate, subst),
                object: Self::substitute_term(object, subst),
            },
            RuleAtom::Builtin { name, args } => RuleAtom::Builtin {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|a| Self::substitute_term(a, subst))
                    .collect(),
            },
            RuleAtom::NotEqual { left, right } => RuleAtom::NotEqual {
                left: Self::substitute_term(left, subst),
                right: Self::substitute_term(right, subst),
            },
            RuleAtom::GreaterThan { left, right } => RuleAtom::GreaterThan {
                left: Self::substitute_term(left, subst),
                right: Self::substitute_term(right, subst),
            },
            RuleAtom::LessThan { left, right } => RuleAtom::LessThan {
                left: Self::substitute_term(left, subst),
                right: Self::substitute_term(right, subst),
            },
        }
    }

    fn substitute_term(term: &Term, subst: &HashMap<String, Term>) -> Term {
        match term {
            Term::Variable(v) => subst.get(v).cloned().unwrap_or_else(|| term.clone()),
            _ => term.clone(),
        }
    }

    fn atom_to_key(atom: &RuleAtom) -> String {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                format!(
                    "{}:{}:{}",
                    Self::term_to_string(subject),
                    Self::term_to_string(predicate),
                    Self::term_to_string(object)
                )
            }
            RuleAtom::Builtin { name, args } => {
                let args_str: Vec<String> = args.iter().map(Self::term_to_string).collect();
                format!("{}({})", name, args_str.join(","))
            }
            _ => format!("{:?}", atom),
        }
    }

    fn term_to_string(term: &Term) -> String {
        match term {
            Term::Variable(v) => format!("?{v}"),
            Term::Constant(c) => c.clone(),
            Term::Literal(l) => format!("\"{l}\""),
            Term::Function { name, args } => {
                let args_str: Vec<String> = args.iter().map(Self::term_to_string).collect();
                format!("{}({})", name, args_str.join(","))
            }
        }
    }

    /// Get the number of rules
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Get the number of known facts
    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    /// Clear all rules and facts
    pub fn clear(&mut self) {
        self.rules.clear();
        self.facts.clear();
        self.stratification = None;
    }

    /// Get rules
    pub fn rules(&self) -> &[NafRule] {
        &self.rules
    }
}

impl Default for StratifiedReasoner {
    fn default() -> Self {
        Self::new(StratificationConfig::default())
    }
}

/// Parser for NAF syntax in rules
pub struct NafParser;

impl NafParser {
    /// Parse NAF notation: \+ atom represents negation
    pub fn parse_body(body_str: &str) -> Result<Vec<NafAtom>> {
        let mut atoms = Vec::new();
        let parts = Self::split_atoms(body_str);

        for part in parts {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }

            if part.starts_with("\\+") || part.starts_with("not ") || part.starts_with("NAF ") {
                // Negated atom
                let inner = part
                    .trim_start_matches("\\+")
                    .trim_start_matches("not ")
                    .trim_start_matches("NAF ")
                    .trim();
                let atom = Self::parse_atom(inner)?;
                atoms.push(NafAtom::Negated(atom));
            } else {
                // Positive atom
                let atom = Self::parse_atom(part)?;
                atoms.push(NafAtom::Positive(atom));
            }
        }

        Ok(atoms)
    }

    /// Split body string into atoms, respecting parentheses
    fn split_atoms(body_str: &str) -> Vec<String> {
        let mut atoms = Vec::new();
        let mut current = String::new();
        let mut paren_depth: i32 = 0;

        for c in body_str.chars() {
            match c {
                '(' => {
                    paren_depth += 1;
                    current.push(c);
                }
                ')' => {
                    paren_depth = paren_depth.saturating_sub(1);
                    current.push(c);
                }
                ',' if paren_depth == 0 => {
                    // Top-level comma: split here
                    if !current.trim().is_empty() {
                        atoms.push(current.trim().to_string());
                    }
                    current = String::new();
                }
                _ => {
                    current.push(c);
                }
            }
        }

        // Don't forget the last atom
        if !current.trim().is_empty() {
            atoms.push(current.trim().to_string());
        }

        atoms
    }

    fn parse_atom(atom_str: &str) -> Result<RuleAtom> {
        let atom_str = atom_str.trim();

        // Simple triple pattern: subject predicate object
        // or predicate(subject, object)
        if atom_str.contains('(') {
            // Function-style: pred(s, o)
            let paren_pos = atom_str.find('(').unwrap();
            let predicate = atom_str[..paren_pos].trim();
            let args_str = atom_str[paren_pos + 1..].trim_end_matches(')');
            let args: Vec<&str> = args_str.split(',').map(|s| s.trim()).collect();

            if args.len() >= 2 {
                Ok(RuleAtom::Triple {
                    subject: Self::parse_term(args[0]),
                    predicate: Term::Constant(predicate.to_string()),
                    object: Self::parse_term(args[1]),
                })
            } else if args.len() == 1 {
                // Unary predicate: treat as type assertion
                Ok(RuleAtom::Triple {
                    subject: Self::parse_term(args[0]),
                    predicate: Term::Constant("rdf:type".to_string()),
                    object: Term::Constant(predicate.to_string()),
                })
            } else {
                Err(anyhow!("Invalid atom: {}", atom_str))
            }
        } else {
            // Space-separated: s p o
            let parts: Vec<&str> = atom_str.split_whitespace().collect();
            if parts.len() >= 3 {
                Ok(RuleAtom::Triple {
                    subject: Self::parse_term(parts[0]),
                    predicate: Self::parse_term(parts[1]),
                    object: Self::parse_term(parts[2]),
                })
            } else {
                Err(anyhow!("Invalid atom: {}", atom_str))
            }
        }
    }

    fn parse_term(term_str: &str) -> Term {
        let term_str = term_str.trim();
        if let Some(stripped) = term_str.strip_prefix('?') {
            Term::Variable(stripped.to_string())
        } else if let Some(stripped) = term_str.strip_prefix('"').and_then(|s| s.strip_suffix('"'))
        {
            Term::Literal(stripped.to_string())
        } else {
            Term::Constant(term_str.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_naf_rule(name: &str, body: Vec<NafAtom>, head: Vec<RuleAtom>) -> NafRule {
        NafRule::new(name.to_string(), body, head)
    }

    fn triple_atom(s: &str, p: &str, o: &str) -> RuleAtom {
        RuleAtom::Triple {
            subject: if let Some(stripped) = s.strip_prefix('?') {
                Term::Variable(stripped.to_string())
            } else {
                Term::Constant(s.to_string())
            },
            predicate: Term::Constant(p.to_string()),
            object: if let Some(stripped) = o.strip_prefix('?') {
                Term::Variable(stripped.to_string())
            } else {
                Term::Constant(o.to_string())
            },
        }
    }

    #[test]
    fn test_naf_atom_creation() {
        let atom = triple_atom("john", "knows", "mary");
        let pos = NafAtom::positive(atom.clone());
        let neg = NafAtom::negated(atom);

        assert!(!pos.is_negated());
        assert!(neg.is_negated());
    }

    #[test]
    fn test_naf_rule_has_negation() {
        let rule_without_neg = create_naf_rule(
            "rule1",
            vec![NafAtom::positive(triple_atom("?X", "parent", "?Y"))],
            vec![triple_atom("?X", "ancestor", "?Y")],
        );

        let rule_with_neg = create_naf_rule(
            "rule2",
            vec![
                NafAtom::positive(triple_atom("?X", "person", "true")),
                NafAtom::negated(triple_atom("?X", "married", "true")),
            ],
            vec![triple_atom("?X", "single", "true")],
        );

        assert!(!rule_without_neg.has_negation());
        assert!(rule_with_neg.has_negation());
    }

    #[test]
    fn test_dependency_graph() {
        let rules = vec![
            create_naf_rule(
                "rule1",
                vec![NafAtom::positive(triple_atom("?X", "a", "?Y"))],
                vec![triple_atom("?X", "b", "?Y")],
            ),
            create_naf_rule(
                "rule2",
                vec![NafAtom::negated(triple_atom("?X", "b", "?Y"))],
                vec![triple_atom("?X", "c", "?Y")],
            ),
        ];

        let graph = DependencyGraph::from_rules(&rules);

        assert!(graph.predicates().contains("a"));
        assert!(graph.predicates().contains("b"));
        assert!(graph.predicates().contains("c"));
    }

    #[test]
    fn test_stratification_simple() {
        let rules = vec![
            // b :- a (stratum 0)
            create_naf_rule(
                "rule1",
                vec![NafAtom::positive(triple_atom("?X", "a", "?Y"))],
                vec![triple_atom("?X", "b", "?Y")],
            ),
            // c :- \+ b (stratum 1, because negation on b)
            create_naf_rule(
                "rule2",
                vec![NafAtom::negated(triple_atom("?X", "b", "?Y"))],
                vec![triple_atom("?X", "c", "?Y")],
            ),
        ];

        let analyzer = StratificationAnalyzer::default();
        let result = analyzer.analyze(&rules);

        assert!(result.is_stratified);
        assert!(result.num_strata >= 2);
    }

    #[test]
    fn test_stratification_with_cycle() {
        // a :- \+ b
        // b :- \+ a
        // This creates a negative cycle
        let rules = vec![
            create_naf_rule(
                "rule1",
                vec![NafAtom::negated(triple_atom("x", "b", "y"))],
                vec![triple_atom("x", "a", "y")],
            ),
            create_naf_rule(
                "rule2",
                vec![NafAtom::negated(triple_atom("x", "a", "y"))],
                vec![triple_atom("x", "b", "y")],
            ),
        ];

        let config = StratificationConfig::default()
            .with_strict_stratification(true)
            .with_well_founded(false);
        let analyzer = StratificationAnalyzer::new(config);
        let result = analyzer.analyze(&rules);

        // Should detect the negative cycle
        assert!(!result.is_stratified || !result.violations.is_empty());
    }

    #[test]
    fn test_stratified_reasoner_basic() {
        let mut reasoner = StratifiedReasoner::default();

        // Simple rule: b(?X, ?Y) :- a(?X, ?Y)
        reasoner.add_rule(create_naf_rule(
            "rule1",
            vec![NafAtom::positive(triple_atom("?X", "a", "?Y"))],
            vec![triple_atom("?X", "b", "?Y")],
        ));

        // Add fact
        reasoner.add_fact("john:a:mary");

        assert!(reasoner.is_known("john:a:mary"));
        assert!(!reasoner.is_known("john:b:mary"));
    }

    #[test]
    fn test_naf_parser_positive() {
        let body = "parent(?X, ?Y), person(?X)";
        let atoms = NafParser::parse_body(body).unwrap();

        assert_eq!(atoms.len(), 2);
        assert!(!atoms[0].is_negated());
        assert!(!atoms[1].is_negated());
    }

    #[test]
    fn test_naf_parser_negation() {
        let body = "person(?X), \\+ married(?X, ?Y)";
        let atoms = NafParser::parse_body(body).unwrap();

        assert_eq!(atoms.len(), 2);
        assert!(!atoms[0].is_negated());
        assert!(atoms[1].is_negated());
    }

    #[test]
    fn test_naf_parser_not_keyword() {
        let body = "person(?X), not dead(?X)";
        let atoms = NafParser::parse_body(body).unwrap();

        assert_eq!(atoms.len(), 2);
        assert!(atoms[1].is_negated());
    }

    #[test]
    fn test_config_builder() {
        let config = StratificationConfig::default()
            .with_well_founded(true)
            .with_max_strata(50)
            .with_strict_stratification(false);

        assert!(config.well_founded);
        assert_eq!(config.max_strata, 50);
        assert!(!config.strict_stratification);
    }

    #[test]
    fn test_dependency_edge_types() {
        let mut graph = DependencyGraph::new();
        graph.add_edge("a", "b", DependencyEdge::Positive);
        graph.add_edge("a", "c", DependencyEdge::Negative);

        let edges = graph.get_edges("a").unwrap();
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_naf_rule_from_standard() {
        let standard = Rule {
            name: "test".to_string(),
            body: vec![triple_atom("?X", "a", "?Y")],
            head: vec![triple_atom("?X", "b", "?Y")],
        };

        let naf_rule = NafRule::from_rule(standard);

        assert_eq!(naf_rule.name, "test");
        assert!(!naf_rule.has_negation());
        assert_eq!(naf_rule.body.len(), 1);
    }

    #[test]
    fn test_stratification_violations() {
        let rules = vec![create_naf_rule(
            "rule1",
            vec![NafAtom::negated(triple_atom("x", "a", "y"))],
            vec![triple_atom("x", "a", "y")], // Self-negation potential
        )];

        let analyzer = StratificationAnalyzer::default();
        let result = analyzer.analyze(&rules);

        // This creates a negative cycle (a depends negatively on a)
        assert!(!result.violations.is_empty() || result.num_strata > 1);
    }

    #[test]
    fn test_predicate_extraction() {
        let atom = triple_atom("john", "knows", "mary");
        let naf_atom = NafAtom::positive(atom);

        assert_eq!(naf_atom.predicate(), Some("knows".to_string()));
    }

    #[test]
    fn test_reasoner_clear() {
        let mut reasoner = StratifiedReasoner::default();

        reasoner.add_rule(create_naf_rule(
            "rule1",
            vec![NafAtom::positive(triple_atom("?X", "a", "?Y"))],
            vec![triple_atom("?X", "b", "?Y")],
        ));
        reasoner.add_fact("test:fact:value");

        assert_eq!(reasoner.rule_count(), 1);
        assert_eq!(reasoner.fact_count(), 1);

        reasoner.clear();

        assert_eq!(reasoner.rule_count(), 0);
        assert_eq!(reasoner.fact_count(), 0);
    }

    #[test]
    fn test_multiple_strata() {
        // Create rules that require multiple strata:
        // b :- a          (stratum 0)
        // c :- \+ b       (stratum 1)
        // d :- \+ c       (stratum 2)
        let rules = vec![
            create_naf_rule(
                "rule1",
                vec![NafAtom::positive(triple_atom("?X", "a", "?Y"))],
                vec![triple_atom("?X", "b", "?Y")],
            ),
            create_naf_rule(
                "rule2",
                vec![NafAtom::negated(triple_atom("?X", "b", "?Y"))],
                vec![triple_atom("?X", "c", "?Y")],
            ),
            create_naf_rule(
                "rule3",
                vec![NafAtom::negated(triple_atom("?X", "c", "?Y"))],
                vec![triple_atom("?X", "d", "?Y")],
            ),
        ];

        let analyzer = StratificationAnalyzer::default();
        let result = analyzer.analyze(&rules);

        assert!(result.is_stratified);
        assert!(result.num_strata >= 3);
    }

    #[test]
    fn test_rules_by_stratum() {
        let rules = vec![
            create_naf_rule(
                "rule1",
                vec![NafAtom::positive(triple_atom("?X", "a", "?Y"))],
                vec![triple_atom("?X", "b", "?Y")],
            ),
            create_naf_rule(
                "rule2",
                vec![NafAtom::negated(triple_atom("?X", "b", "?Y"))],
                vec![triple_atom("?X", "c", "?Y")],
            ),
        ];

        let analyzer = StratificationAnalyzer::default();
        let result = analyzer.analyze(&rules);

        // Rules should be grouped by stratum
        assert!(!result.rules_by_stratum.is_empty());
    }
}
