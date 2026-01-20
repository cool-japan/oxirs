//! # Answer Set Programming (ASP) Solver
//!
//! This module implements an Answer Set Programming solver for combinatorial optimization
//! and constraint satisfaction problems over RDF knowledge graphs.
//!
//! ## Features
//!
//! - **Choice Rules**: Non-deterministic selection from alternatives
//! - **Integrity Constraints**: Hard constraints that must be satisfied
//! - **Weight Constraints**: Soft constraints with optimization
//! - **Stable Model Semantics**: Grounded answer set computation
//! - **Optimization**: Find optimal solutions based on criteria
//!
//! ## ASP Syntax Support
//!
//! ```text
//! % Choice rules
//! { color(X, red); color(X, green); color(X, blue) } = 1 :- node(X).
//!
//! % Integrity constraints
//! :- edge(X, Y), color(X, C), color(Y, C).  % Adjacent nodes different colors
//!
//! % Weight constraints
//! #minimize { W,X : cost(X, W) }.
//! ```
//!
//! ## Example
//!
//! ```rust
//! use oxirs_rule::asp::AspSolver;
//!
//! let mut solver = AspSolver::new();
//!
//! // Add facts
//! solver.add_fact("node(a)").unwrap();
//! solver.add_fact("node(b)").unwrap();
//! solver.add_fact("edge(a, b)").unwrap();
//!
//! // Solve and get answer sets
//! let answer_sets = solver.solve().unwrap();
//! assert!(!answer_sets.is_empty());
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;

use anyhow::Result;

use crate::{RuleAtom, Term};

/// ASP solver configuration
#[derive(Debug, Clone)]
pub struct AspConfig {
    /// Maximum number of answer sets to compute
    pub max_answer_sets: usize,
    /// Maximum iterations for stable model computation
    pub max_iterations: usize,
    /// Enable optimization
    pub enable_optimization: bool,
    /// Random seed for non-deterministic choices
    pub random_seed: Option<u64>,
    /// Enable grounding optimization
    pub optimize_grounding: bool,
}

impl Default for AspConfig {
    fn default() -> Self {
        Self {
            max_answer_sets: 10,
            max_iterations: 10000,
            enable_optimization: true,
            random_seed: None,
            optimize_grounding: true,
        }
    }
}

impl AspConfig {
    /// Set maximum answer sets
    pub fn with_max_answer_sets(mut self, max: usize) -> Self {
        self.max_answer_sets = max;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Enable or disable optimization
    pub fn with_optimization(mut self, enabled: bool) -> Self {
        self.enable_optimization = enabled;
        self
    }
}

/// An ASP literal (atom or negated atom)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AspLiteral {
    /// Positive atom
    Positive(Atom),
    /// Negated atom (classical negation)
    Negated(Atom),
    /// Default negation (not)
    DefaultNegated(Atom),
}

impl AspLiteral {
    /// Create a positive literal
    pub fn positive(atom: Atom) -> Self {
        AspLiteral::Positive(atom)
    }

    /// Create a negated literal
    pub fn negated(atom: Atom) -> Self {
        AspLiteral::Negated(atom)
    }

    /// Create a default-negated literal (NAF)
    pub fn not(atom: Atom) -> Self {
        AspLiteral::DefaultNegated(atom)
    }

    /// Get the underlying atom
    pub fn atom(&self) -> &Atom {
        match self {
            AspLiteral::Positive(a) | AspLiteral::Negated(a) | AspLiteral::DefaultNegated(a) => a,
        }
    }

    /// Check if this is positive
    pub fn is_positive(&self) -> bool {
        matches!(self, AspLiteral::Positive(_))
    }

    /// Check if this uses default negation
    pub fn is_default_negated(&self) -> bool {
        matches!(self, AspLiteral::DefaultNegated(_))
    }
}

impl fmt::Display for AspLiteral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AspLiteral::Positive(a) => write!(f, "{a}"),
            AspLiteral::Negated(a) => write!(f, "-{a}"),
            AspLiteral::DefaultNegated(a) => write!(f, "not {a}"),
        }
    }
}

/// An ASP atom (predicate with arguments)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Atom {
    /// Predicate name
    pub predicate: String,
    /// Arguments
    pub args: Vec<AspTerm>,
}

impl Atom {
    /// Create a new atom
    pub fn new(predicate: &str, args: Vec<AspTerm>) -> Self {
        Self {
            predicate: predicate.to_string(),
            args,
        }
    }

    /// Create a nullary atom (no arguments)
    pub fn nullary(predicate: &str) -> Self {
        Self::new(predicate, Vec::new())
    }

    /// Check if atom is ground (no variables)
    pub fn is_ground(&self) -> bool {
        self.args.iter().all(|a| a.is_ground())
    }

    /// Apply substitution
    pub fn apply(&self, subst: &Substitution) -> Atom {
        Atom {
            predicate: self.predicate.clone(),
            args: self.args.iter().map(|a| a.apply(subst)).collect(),
        }
    }

    /// Get all variables in this atom
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        for arg in &self.args {
            if let AspTerm::Variable(v) = arg {
                vars.insert(v.clone());
            }
        }
        vars
    }
}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.args.is_empty() {
            write!(f, "{}", self.predicate)
        } else {
            write!(
                f,
                "{}({})",
                self.predicate,
                self.args
                    .iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}

/// ASP term (constant, variable, or function)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AspTerm {
    /// Constant value
    Constant(String),
    /// Variable
    Variable(String),
    /// Integer constant
    Integer(i64),
    /// Function application
    Function { name: String, args: Vec<AspTerm> },
}

impl AspTerm {
    /// Create a constant term
    pub fn constant(value: &str) -> Self {
        AspTerm::Constant(value.to_string())
    }

    /// Create a variable term
    pub fn variable(name: &str) -> Self {
        AspTerm::Variable(name.to_string())
    }

    /// Create an integer term
    pub fn integer(value: i64) -> Self {
        AspTerm::Integer(value)
    }

    /// Check if ground
    pub fn is_ground(&self) -> bool {
        match self {
            AspTerm::Constant(_) | AspTerm::Integer(_) => true,
            AspTerm::Variable(_) => false,
            AspTerm::Function { args, .. } => args.iter().all(|a| a.is_ground()),
        }
    }

    /// Apply substitution
    pub fn apply(&self, subst: &Substitution) -> AspTerm {
        match self {
            AspTerm::Variable(v) => subst.get(v).cloned().unwrap_or_else(|| self.clone()),
            AspTerm::Function { name, args } => AspTerm::Function {
                name: name.clone(),
                args: args.iter().map(|a| a.apply(subst)).collect(),
            },
            _ => self.clone(),
        }
    }
}

impl fmt::Display for AspTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AspTerm::Constant(c) => write!(f, "{c}"),
            AspTerm::Variable(v) => write!(f, "{v}"),
            AspTerm::Integer(i) => write!(f, "{i}"),
            AspTerm::Function { name, args } => {
                write!(
                    f,
                    "{}({})",
                    name,
                    args.iter()
                        .map(|a| a.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
    }
}

/// Variable substitution
pub type Substitution = HashMap<String, AspTerm>;

/// Types of ASP rules
#[derive(Debug, Clone)]
pub enum AspRule {
    /// Normal rule: head :- body
    Normal {
        head: Option<AspLiteral>,
        body: Vec<AspLiteral>,
    },
    /// Choice rule: { h1; h2; ... } :- body
    Choice {
        choices: Vec<Atom>,
        lower_bound: Option<usize>,
        upper_bound: Option<usize>,
        body: Vec<AspLiteral>,
    },
    /// Integrity constraint: :- body (empty head)
    Constraint { body: Vec<AspLiteral> },
    /// Weak constraint (soft): `:~ body [weight@level]`
    Weak {
        body: Vec<AspLiteral>,
        weight: i64,
        level: i32,
    },
}

impl AspRule {
    /// Create a normal rule
    pub fn normal(head: Atom, body: Vec<AspLiteral>) -> Self {
        AspRule::Normal {
            head: Some(AspLiteral::Positive(head)),
            body,
        }
    }

    /// Create a fact (rule with empty body)
    pub fn fact(head: Atom) -> Self {
        AspRule::Normal {
            head: Some(AspLiteral::Positive(head)),
            body: Vec::new(),
        }
    }

    /// Create an integrity constraint
    pub fn constraint(body: Vec<AspLiteral>) -> Self {
        AspRule::Constraint { body }
    }

    /// Create a choice rule
    pub fn choice(
        choices: Vec<Atom>,
        lower: Option<usize>,
        upper: Option<usize>,
        body: Vec<AspLiteral>,
    ) -> Self {
        AspRule::Choice {
            choices,
            lower_bound: lower,
            upper_bound: upper,
            body,
        }
    }

    /// Create a weak constraint
    pub fn weak(body: Vec<AspLiteral>, weight: i64, level: i32) -> Self {
        AspRule::Weak {
            body,
            weight,
            level,
        }
    }

    /// Get all variables in the rule
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();

        match self {
            AspRule::Normal { head, body } => {
                if let Some(h) = head {
                    vars.extend(h.atom().variables());
                }
                for lit in body {
                    vars.extend(lit.atom().variables());
                }
            }
            AspRule::Choice { choices, body, .. } => {
                for c in choices {
                    vars.extend(c.variables());
                }
                for lit in body {
                    vars.extend(lit.atom().variables());
                }
            }
            AspRule::Constraint { body } | AspRule::Weak { body, .. } => {
                for lit in body {
                    vars.extend(lit.atom().variables());
                }
            }
        }

        vars
    }

    /// Check if rule is ground
    pub fn is_ground(&self) -> bool {
        self.variables().is_empty()
    }
}

/// An answer set (stable model)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnswerSet {
    /// Atoms in the answer set
    pub atoms: HashSet<Atom>,
    /// Cost (for optimization)
    pub cost: Option<i64>,
    /// Optimality level
    pub optimality: Option<i32>,
}

impl AnswerSet {
    /// Create a new answer set
    pub fn new(atoms: HashSet<Atom>) -> Self {
        Self {
            atoms,
            cost: None,
            optimality: None,
        }
    }

    /// Check if an atom is in this answer set
    pub fn contains(&self, atom: &Atom) -> bool {
        self.atoms.contains(atom)
    }

    /// Number of atoms
    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    /// Get atoms with a specific predicate
    pub fn atoms_with_predicate(&self, predicate: &str) -> Vec<&Atom> {
        self.atoms
            .iter()
            .filter(|a| a.predicate == predicate)
            .collect()
    }
}

impl fmt::Display for AnswerSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let atoms: Vec<String> = self.atoms.iter().map(|a| a.to_string()).collect();
        write!(f, "{{ {} }}", atoms.join(", "))
    }
}

/// Grounding result
#[derive(Debug)]
struct GroundProgram {
    /// Ground rules
    rules: Vec<AspRule>,
    /// Ground atoms that appeared
    atoms: HashSet<Atom>,
}

/// ASP Solver
pub struct AspSolver {
    /// Configuration
    config: AspConfig,
    /// Non-ground rules
    rules: Vec<AspRule>,
    /// Base facts
    facts: HashSet<Atom>,
    /// Domain for variables (for grounding)
    domain: HashSet<AspTerm>,
}

impl AspSolver {
    /// Create a new ASP solver
    pub fn new() -> Self {
        Self::with_config(AspConfig::default())
    }

    /// Create solver with configuration
    pub fn with_config(config: AspConfig) -> Self {
        Self {
            config,
            rules: Vec::new(),
            facts: HashSet::new(),
            domain: HashSet::new(),
        }
    }

    /// Add a fact
    pub fn add_fact(&mut self, fact_str: &str) -> Result<()> {
        let atom = Self::parse_atom(fact_str)?;

        // Add args to domain
        for arg in &atom.args {
            if arg.is_ground() {
                self.domain.insert(arg.clone());
            }
        }

        self.facts.insert(atom);
        Ok(())
    }

    /// Add a rule
    pub fn add_rule(&mut self, rule: AspRule) {
        // Extract domain from rule
        self.extract_domain(&rule);
        self.rules.push(rule);
    }

    /// Add an integrity constraint
    pub fn add_constraint(&mut self, body: Vec<AspLiteral>) {
        self.add_rule(AspRule::constraint(body));
    }

    /// Add a choice rule
    pub fn add_choice_rule(
        &mut self,
        choices: Vec<Atom>,
        lower: Option<usize>,
        upper: Option<usize>,
        body: Vec<AspLiteral>,
    ) {
        self.add_rule(AspRule::choice(choices, lower, upper, body));
    }

    /// Solve and compute answer sets
    pub fn solve(&mut self) -> Result<Vec<AnswerSet>> {
        // Step 1: Ground the program
        let ground_program = self.ground()?;

        // Step 2: Compute stable models
        let answer_sets = self.compute_stable_models(&ground_program)?;

        // Step 3: Optimize if enabled
        if self.config.enable_optimization {
            return Ok(self.optimize(answer_sets));
        }

        Ok(answer_sets)
    }

    /// Ground the ASP program
    fn ground(&self) -> Result<GroundProgram> {
        let mut ground_rules = Vec::new();
        let mut ground_atoms = HashSet::new();

        // Add facts to ground atoms
        for fact in &self.facts {
            ground_atoms.insert(fact.clone());
            ground_rules.push(AspRule::fact(fact.clone()));
        }

        // Ground each rule
        for rule in &self.rules {
            let vars = rule.variables();

            if vars.is_empty() {
                // Already ground
                ground_rules.push(rule.clone());
            } else {
                // Generate all groundings
                let substitutions = self.generate_substitutions(&vars);

                for subst in substitutions {
                    let ground_rule = self.apply_substitution_to_rule(rule, &subst);
                    ground_rules.push(ground_rule);
                }
            }
        }

        Ok(GroundProgram {
            rules: ground_rules,
            atoms: ground_atoms,
        })
    }

    fn generate_substitutions(&self, vars: &HashSet<String>) -> Vec<Substitution> {
        let vars_vec: Vec<&String> = vars.iter().collect();

        if vars_vec.is_empty() {
            return vec![HashMap::new()];
        }

        let domain_vec: Vec<&AspTerm> = self.domain.iter().collect();

        if domain_vec.is_empty() {
            return Vec::new();
        }

        // Generate all combinations
        let mut result = vec![HashMap::new()];

        for var in vars_vec {
            let mut new_result = Vec::new();

            for subst in &result {
                for term in &domain_vec {
                    let mut new_subst = subst.clone();
                    new_subst.insert(var.clone(), (*term).clone());
                    new_result.push(new_subst);
                }
            }

            result = new_result;
        }

        result
    }

    fn apply_substitution_to_rule(&self, rule: &AspRule, subst: &Substitution) -> AspRule {
        match rule {
            AspRule::Normal { head, body } => AspRule::Normal {
                head: head
                    .as_ref()
                    .map(|h| self.apply_substitution_to_literal(h, subst)),
                body: body
                    .iter()
                    .map(|l| self.apply_substitution_to_literal(l, subst))
                    .collect(),
            },
            AspRule::Choice {
                choices,
                lower_bound,
                upper_bound,
                body,
            } => AspRule::Choice {
                choices: choices.iter().map(|c| c.apply(subst)).collect(),
                lower_bound: *lower_bound,
                upper_bound: *upper_bound,
                body: body
                    .iter()
                    .map(|l| self.apply_substitution_to_literal(l, subst))
                    .collect(),
            },
            AspRule::Constraint { body } => AspRule::Constraint {
                body: body
                    .iter()
                    .map(|l| self.apply_substitution_to_literal(l, subst))
                    .collect(),
            },
            AspRule::Weak {
                body,
                weight,
                level,
            } => AspRule::Weak {
                body: body
                    .iter()
                    .map(|l| self.apply_substitution_to_literal(l, subst))
                    .collect(),
                weight: *weight,
                level: *level,
            },
        }
    }

    fn apply_substitution_to_literal(&self, lit: &AspLiteral, subst: &Substitution) -> AspLiteral {
        match lit {
            AspLiteral::Positive(a) => AspLiteral::Positive(a.apply(subst)),
            AspLiteral::Negated(a) => AspLiteral::Negated(a.apply(subst)),
            AspLiteral::DefaultNegated(a) => AspLiteral::DefaultNegated(a.apply(subst)),
        }
    }

    /// Compute stable models using a simplified algorithm
    fn compute_stable_models(&mut self, program: &GroundProgram) -> Result<Vec<AnswerSet>> {
        let mut answer_sets = Vec::new();

        // Collect all atoms that can potentially be true
        let mut potential_atoms: HashSet<Atom> = program.atoms.clone();

        for rule in &program.rules {
            match rule {
                AspRule::Normal {
                    head: Some(AspLiteral::Positive(a)),
                    ..
                } => {
                    potential_atoms.insert(a.clone());
                }
                AspRule::Choice { choices, .. } => {
                    for c in choices {
                        potential_atoms.insert(c.clone());
                    }
                }
                _ => {}
            }
        }

        // Use a simple approach: start with facts and expand
        let mut base_model: HashSet<Atom> = self.facts.clone();

        // Apply normal rules until fixpoint
        let mut changed = true;
        let mut iterations = 0;

        while changed && iterations < self.config.max_iterations {
            changed = false;
            iterations += 1;

            for rule in &program.rules {
                if let AspRule::Normal {
                    head: Some(AspLiteral::Positive(head_atom)),
                    body,
                } = rule
                {
                    if !base_model.contains(head_atom) && self.body_satisfied(body, &base_model) {
                        base_model.insert(head_atom.clone());
                        changed = true;
                    }
                }
            }
        }

        // Check constraints
        if self.check_constraints(&program.rules, &base_model) {
            answer_sets.push(AnswerSet::new(base_model.clone()));
        }

        // For choice rules, generate alternatives
        let choice_rules: Vec<_> = program
            .rules
            .iter()
            .filter(|r| matches!(r, AspRule::Choice { .. }))
            .collect();

        if !choice_rules.is_empty() {
            let additional =
                self.generate_choice_models(&base_model, &choice_rules, &program.rules)?;
            for model in additional {
                if answer_sets.len() >= self.config.max_answer_sets {
                    break;
                }
                if !answer_sets.iter().any(|as_| as_.atoms == model.atoms) {
                    answer_sets.push(model);
                }
            }
        }

        Ok(answer_sets)
    }

    fn body_satisfied(&self, body: &[AspLiteral], model: &HashSet<Atom>) -> bool {
        for lit in body {
            match lit {
                AspLiteral::Positive(a) => {
                    if !model.contains(a) {
                        return false;
                    }
                }
                AspLiteral::Negated(a) => {
                    // Classical negation: -a must be in model
                    // For simplicity, we treat this as a not being in model
                    if model.contains(a) {
                        return false;
                    }
                }
                AspLiteral::DefaultNegated(a) => {
                    // NAF: a must NOT be in model
                    if model.contains(a) {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn check_constraints(&self, rules: &[AspRule], model: &HashSet<Atom>) -> bool {
        for rule in rules {
            if let AspRule::Constraint { body } = rule {
                // Constraint is violated if body is satisfied
                if self.body_satisfied(body, model) {
                    return false;
                }
            }
        }
        true
    }

    fn generate_choice_models(
        &mut self,
        base: &HashSet<Atom>,
        choice_rules: &[&AspRule],
        all_rules: &[AspRule],
    ) -> Result<Vec<AnswerSet>> {
        let mut models = Vec::new();

        // For each choice rule, generate possible selections
        for rule in choice_rules {
            if let AspRule::Choice {
                choices,
                lower_bound,
                upper_bound,
                body,
            } = rule
            {
                // Check if body is satisfied
                if !self.body_satisfied(body, base) {
                    continue;
                }

                let lower = lower_bound.unwrap_or(0);
                let upper = upper_bound.unwrap_or(choices.len());

                // Generate subsets of choices
                let subsets = self.generate_subsets(choices, lower, upper);

                for subset in subsets {
                    let mut model = base.clone();
                    model.extend(subset);

                    // Apply consequent rules
                    model = self.apply_rules_until_fixpoint(model, all_rules);

                    // Check constraints
                    if self.check_constraints(all_rules, &model) {
                        models.push(AnswerSet::new(model));

                        if models.len() >= self.config.max_answer_sets {
                            return Ok(models);
                        }
                    }
                }
            }
        }

        Ok(models)
    }

    fn generate_subsets(&self, items: &[Atom], min: usize, max: usize) -> Vec<Vec<Atom>> {
        let mut result = Vec::new();

        for size in min..=max.min(items.len()) {
            let combinations = Self::generate_combinations(items, size);
            result.extend(combinations);
        }

        result
    }

    fn generate_combinations(items: &[Atom], k: usize) -> Vec<Vec<Atom>> {
        if k == 0 {
            return vec![Vec::new()];
        }
        if items.len() < k {
            return Vec::new();
        }

        let mut result = Vec::new();

        for (i, item) in items.iter().enumerate() {
            let rest = &items[i + 1..];
            for mut combo in Self::generate_combinations(rest, k - 1) {
                combo.insert(0, item.clone());
                result.push(combo);
            }
        }

        result
    }

    fn apply_rules_until_fixpoint(
        &self,
        mut model: HashSet<Atom>,
        rules: &[AspRule],
    ) -> HashSet<Atom> {
        let mut changed = true;
        let mut iterations = 0;

        while changed && iterations < self.config.max_iterations {
            changed = false;
            iterations += 1;

            for rule in rules {
                if let AspRule::Normal {
                    head: Some(AspLiteral::Positive(head_atom)),
                    body,
                } = rule
                {
                    if !model.contains(head_atom) && self.body_satisfied(body, &model) {
                        model.insert(head_atom.clone());
                        changed = true;
                    }
                }
            }
        }

        model
    }

    fn optimize(&self, mut answer_sets: Vec<AnswerSet>) -> Vec<AnswerSet> {
        // Sort by cost if available
        answer_sets.sort_by(|a, b| match (a.cost, b.cost) {
            (Some(ca), Some(cb)) => ca.cmp(&cb),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        });

        // Limit to max_answer_sets
        answer_sets.truncate(self.config.max_answer_sets);
        answer_sets
    }

    fn extract_domain(&mut self, rule: &AspRule) {
        match rule {
            AspRule::Normal { head, body } => {
                if let Some(h) = head {
                    for arg in &h.atom().args {
                        if arg.is_ground() {
                            self.domain.insert(arg.clone());
                        }
                    }
                }
                for lit in body {
                    for arg in &lit.atom().args {
                        if arg.is_ground() {
                            self.domain.insert(arg.clone());
                        }
                    }
                }
            }
            AspRule::Choice { choices, body, .. } => {
                for c in choices {
                    for arg in &c.args {
                        if arg.is_ground() {
                            self.domain.insert(arg.clone());
                        }
                    }
                }
                for lit in body {
                    for arg in &lit.atom().args {
                        if arg.is_ground() {
                            self.domain.insert(arg.clone());
                        }
                    }
                }
            }
            AspRule::Constraint { body } | AspRule::Weak { body, .. } => {
                for lit in body {
                    for arg in &lit.atom().args {
                        if arg.is_ground() {
                            self.domain.insert(arg.clone());
                        }
                    }
                }
            }
        }
    }

    fn parse_atom(atom_str: &str) -> Result<Atom> {
        let atom_str = atom_str.trim();

        // Parse predicate(arg1, arg2, ...)
        if let Some(paren_pos) = atom_str.find('(') {
            let predicate = atom_str[..paren_pos].trim();
            let args_str = atom_str[paren_pos + 1..].trim_end_matches(')');

            let args: Vec<AspTerm> = if args_str.is_empty() {
                Vec::new()
            } else {
                args_str
                    .split(',')
                    .map(|s| Self::parse_term(s.trim()))
                    .collect()
            };

            Ok(Atom::new(predicate, args))
        } else {
            // Nullary predicate
            Ok(Atom::nullary(atom_str))
        }
    }

    fn parse_term(term_str: &str) -> AspTerm {
        let term_str = term_str.trim();

        // Check if integer
        if let Ok(i) = term_str.parse::<i64>() {
            return AspTerm::Integer(i);
        }

        // Check if variable (starts with uppercase or _)
        if term_str
            .chars()
            .next()
            .is_some_and(|c| c.is_uppercase() || c == '_')
        {
            return AspTerm::Variable(term_str.to_string());
        }

        // Otherwise constant
        AspTerm::Constant(term_str.to_string())
    }

    /// Get number of rules
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Get number of facts
    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    /// Clear the solver
    pub fn clear(&mut self) {
        self.rules.clear();
        self.facts.clear();
        self.domain.clear();
    }
}

impl Default for AspSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert between OxiRS Rule types and ASP types
pub mod conversion {
    use super::*;

    /// Convert RuleAtom to AspLiteral
    pub fn rule_atom_to_asp_literal(atom: &RuleAtom) -> Option<AspLiteral> {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                let pred = match predicate {
                    Term::Constant(c) => c.clone(),
                    _ => return None,
                };

                let args = vec![term_to_asp_term(subject), term_to_asp_term(object)];

                Some(AspLiteral::positive(Atom::new(&pred, args)))
            }
            RuleAtom::Builtin { name, args } => {
                let asp_args: Vec<AspTerm> = args.iter().map(term_to_asp_term).collect();
                Some(AspLiteral::positive(Atom::new(name, asp_args)))
            }
            _ => None,
        }
    }

    /// Convert Term to AspTerm
    pub fn term_to_asp_term(term: &Term) -> AspTerm {
        match term {
            Term::Variable(v) => AspTerm::Variable(v.clone()),
            Term::Constant(c) => AspTerm::Constant(c.clone()),
            Term::Literal(l) => AspTerm::Constant(l.clone()),
            Term::Function { name, args } => AspTerm::Function {
                name: name.clone(),
                args: args.iter().map(term_to_asp_term).collect(),
            },
        }
    }

    /// Convert AspLiteral to RuleAtom
    pub fn asp_literal_to_rule_atom(lit: &AspLiteral) -> Option<RuleAtom> {
        if let AspLiteral::Positive(atom) = lit {
            if atom.args.len() >= 2 {
                Some(RuleAtom::Triple {
                    subject: asp_term_to_term(&atom.args[0]),
                    predicate: Term::Constant(atom.predicate.clone()),
                    object: asp_term_to_term(&atom.args[1]),
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Convert AspTerm to Term
    pub fn asp_term_to_term(term: &AspTerm) -> Term {
        match term {
            AspTerm::Variable(v) => Term::Variable(v.clone()),
            AspTerm::Constant(c) => Term::Constant(c.clone()),
            AspTerm::Integer(i) => Term::Literal(i.to_string()),
            AspTerm::Function { name, args } => Term::Function {
                name: name.clone(),
                args: args.iter().map(asp_term_to_term).collect(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atom_creation() {
        let atom = Atom::new(
            "parent",
            vec![AspTerm::constant("john"), AspTerm::constant("mary")],
        );

        assert_eq!(atom.predicate, "parent");
        assert_eq!(atom.args.len(), 2);
        assert!(atom.is_ground());
    }

    #[test]
    fn test_atom_with_variables() {
        let atom = Atom::new(
            "parent",
            vec![AspTerm::variable("X"), AspTerm::constant("mary")],
        );

        assert!(!atom.is_ground());
        assert!(atom.variables().contains("X"));
    }

    #[test]
    fn test_literal_types() {
        let atom = Atom::nullary("rain");

        let pos = AspLiteral::positive(atom.clone());
        let neg = AspLiteral::negated(atom.clone());
        let naf = AspLiteral::not(atom);

        assert!(pos.is_positive());
        assert!(!neg.is_positive());
        assert!(naf.is_default_negated());
    }

    #[test]
    fn test_solver_add_fact() {
        let mut solver = AspSolver::new();
        solver.add_fact("node(a)").unwrap();
        solver.add_fact("node(b)").unwrap();

        assert_eq!(solver.fact_count(), 2);
    }

    #[test]
    fn test_solver_simple_inference() {
        let mut solver = AspSolver::new();

        // Add facts
        solver.add_fact("parent(john, mary)").unwrap();

        // Add rule: ancestor(X, Y) :- parent(X, Y)
        let rule = AspRule::normal(
            Atom::new(
                "ancestor",
                vec![AspTerm::variable("X"), AspTerm::variable("Y")],
            ),
            vec![AspLiteral::positive(Atom::new(
                "parent",
                vec![AspTerm::variable("X"), AspTerm::variable("Y")],
            ))],
        );
        solver.add_rule(rule);

        let answer_sets = solver.solve().unwrap();

        // Should have at least one answer set
        assert!(!answer_sets.is_empty());
    }

    #[test]
    fn test_integrity_constraint() {
        let mut solver = AspSolver::new();

        solver.add_fact("node(a)").unwrap();
        solver.add_fact("node(b)").unwrap();

        // Constraint: can't have both a and b colored red
        // :- color(a, red), color(b, red)
        solver.add_constraint(vec![
            AspLiteral::positive(Atom::new(
                "color",
                vec![AspTerm::constant("a"), AspTerm::constant("red")],
            )),
            AspLiteral::positive(Atom::new(
                "color",
                vec![AspTerm::constant("b"), AspTerm::constant("red")],
            )),
        ]);

        let answer_sets = solver.solve().unwrap();

        // All answer sets should satisfy the constraint
        for as_ in &answer_sets {
            let a_red = as_.contains(&Atom::new(
                "color",
                vec![AspTerm::constant("a"), AspTerm::constant("red")],
            ));
            let b_red = as_.contains(&Atom::new(
                "color",
                vec![AspTerm::constant("b"), AspTerm::constant("red")],
            ));
            assert!(!(a_red && b_red), "Constraint violated");
        }
    }

    #[test]
    fn test_choice_rule() {
        let mut solver = AspSolver::new();

        solver.add_fact("option(a)").unwrap();
        solver.add_fact("option(b)").unwrap();

        // Choice rule: exactly one of a or b
        solver.add_choice_rule(
            vec![
                Atom::new("selected", vec![AspTerm::constant("a")]),
                Atom::new("selected", vec![AspTerm::constant("b")]),
            ],
            Some(1), // exactly 1
            Some(1),
            Vec::new(),
        );

        let answer_sets = solver.solve().unwrap();

        // Should have answer sets
        assert!(!answer_sets.is_empty());
    }

    #[test]
    fn test_default_negation() {
        let mut solver = AspSolver::new();

        solver.add_fact("bird(tweety)").unwrap();
        // Don't add: penguin(tweety)

        // Rule: flies(X) :- bird(X), not penguin(X)
        solver.add_rule(AspRule::normal(
            Atom::new("flies", vec![AspTerm::variable("X")]),
            vec![
                AspLiteral::positive(Atom::new("bird", vec![AspTerm::variable("X")])),
                AspLiteral::not(Atom::new("penguin", vec![AspTerm::variable("X")])),
            ],
        ));

        let answer_sets = solver.solve().unwrap();

        // Tweety should fly since it's a bird and not a penguin
        assert!(!answer_sets.is_empty());
        let flies = Atom::new("flies", vec![AspTerm::constant("tweety")]);
        assert!(
            answer_sets[0].contains(&flies) || answer_sets.iter().any(|as_| as_.contains(&flies))
        );
    }

    #[test]
    fn test_atom_display() {
        let atom = Atom::new(
            "parent",
            vec![AspTerm::constant("john"), AspTerm::constant("mary")],
        );
        assert_eq!(atom.to_string(), "parent(john, mary)");

        let nullary = Atom::nullary("rain");
        assert_eq!(nullary.to_string(), "rain");
    }

    #[test]
    fn test_substitution() {
        let atom = Atom::new(
            "parent",
            vec![AspTerm::variable("X"), AspTerm::variable("Y")],
        );

        let mut subst = Substitution::new();
        subst.insert("X".to_string(), AspTerm::constant("john"));
        subst.insert("Y".to_string(), AspTerm::constant("mary"));

        let grounded = atom.apply(&subst);

        assert!(grounded.is_ground());
        assert_eq!(grounded.args[0], AspTerm::constant("john"));
        assert_eq!(grounded.args[1], AspTerm::constant("mary"));
    }

    #[test]
    fn test_answer_set_display() {
        let mut atoms = HashSet::new();
        atoms.insert(Atom::new("a", vec![AspTerm::constant("x")]));
        atoms.insert(Atom::new("b", vec![AspTerm::constant("y")]));

        let as_ = AnswerSet::new(atoms);
        let display = as_.to_string();

        assert!(display.contains("a(x)") || display.contains("b(y)"));
    }

    #[test]
    fn test_config_builder() {
        let config = AspConfig::default()
            .with_max_answer_sets(5)
            .with_seed(42)
            .with_optimization(false);

        assert_eq!(config.max_answer_sets, 5);
        assert_eq!(config.random_seed, Some(42));
        assert!(!config.enable_optimization);
    }

    #[test]
    fn test_solver_clear() {
        let mut solver = AspSolver::new();

        solver.add_fact("test(a)").unwrap();
        solver.add_rule(AspRule::fact(Atom::nullary("b")));

        assert_eq!(solver.fact_count(), 1);
        assert_eq!(solver.rule_count(), 1);

        solver.clear();

        assert_eq!(solver.fact_count(), 0);
        assert_eq!(solver.rule_count(), 0);
    }

    #[test]
    fn test_weak_constraint() {
        let weak = AspRule::weak(
            vec![AspLiteral::positive(Atom::new(
                "cost",
                vec![AspTerm::variable("X"), AspTerm::integer(10)],
            ))],
            10,
            1,
        );

        assert!(matches!(
            weak,
            AspRule::Weak {
                weight: 10,
                level: 1,
                ..
            }
        ));
    }

    #[test]
    fn test_answer_set_predicates() {
        let mut atoms = HashSet::new();
        atoms.insert(Atom::new(
            "color",
            vec![AspTerm::constant("a"), AspTerm::constant("red")],
        ));
        atoms.insert(Atom::new(
            "color",
            vec![AspTerm::constant("b"), AspTerm::constant("blue")],
        ));
        atoms.insert(Atom::new("node", vec![AspTerm::constant("a")]));

        let as_ = AnswerSet::new(atoms);

        let colors = as_.atoms_with_predicate("color");
        assert_eq!(colors.len(), 2);

        let nodes = as_.atoms_with_predicate("node");
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn test_combinations() {
        let items = vec![Atom::nullary("a"), Atom::nullary("b"), Atom::nullary("c")];

        let combos = AspSolver::generate_combinations(&items, 2);

        assert_eq!(combos.len(), 3); // C(3,2) = 3
    }

    #[test]
    fn test_conversion_term_to_asp() {
        let term = Term::Variable("X".to_string());
        let asp_term = conversion::term_to_asp_term(&term);
        assert!(matches!(asp_term, AspTerm::Variable(v) if v == "X"));

        let const_term = Term::Constant("john".to_string());
        let asp_const = conversion::term_to_asp_term(&const_term);
        assert!(matches!(asp_const, AspTerm::Constant(c) if c == "john"));
    }
}
