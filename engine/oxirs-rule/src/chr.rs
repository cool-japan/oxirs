//! # Constraint Handling Rules (CHR) Engine
//!
//! Declarative constraint solving framework for logic programming with constraints.
//! CHR extends Datalog with constraint propagation and simplification.
//!
//! ## CHR Rule Types
//!
//! - **Simplification**: `H <=> G | B` - Replaces head with body when guard holds
//! - **Propagation**: `H ==> G | B` - Keeps head and adds body
//! - **Simpagation**: `H1 \ H2 <=> G | B` - Hybrid: keeps H1, removes H2
//!
//! ## Example
//!
//! ```rust
//! use oxirs_rule::chr::{ChrEngine, ChrRule, ChrTerm, Constraint};
//!
//! let mut engine = ChrEngine::new();
//!
//! // Simplification rule: leq(X, Y), leq(Y, X) <=> X = Y
//! engine.add_rule(ChrRule::simplification(
//!     "antisymmetry",
//!     vec![Constraint::binary("leq", "X", "Y"), Constraint::binary("leq", "Y", "X")],
//!     vec![],  // no guard
//!     vec![Constraint::eq("X", "Y")],
//! ));
//!
//! // Add constraints
//! engine.add_constraint(Constraint::new("leq", vec![ChrTerm::const_("a"), ChrTerm::const_("b")]));
//! engine.add_constraint(Constraint::new("leq", vec![ChrTerm::const_("b"), ChrTerm::const_("a")]));
//!
//! // Solve
//! let result = engine.solve()?;
//! // Should produce equality constraint
//! assert!(result.iter().any(|c| c.name == "="));
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! ## References
//!
//! - [CHR Home](https://dtai.cs.kuleuven.be/CHR/)
//! - T. Frühwirth, "Theory and Practice of Constraint Handling Rules"

use crate::Term;
use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet, VecDeque};

/// Constraint term (logic variable or constant)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChrTerm {
    /// Variable (starts with uppercase by convention)
    Var(String),
    /// Constant value
    Const(String),
    /// Integer constant
    Int(i64),
    /// Function application
    Func(String, Vec<ChrTerm>),
}

impl ChrTerm {
    /// Create a variable term
    pub fn var(name: &str) -> Self {
        Self::Var(name.to_string())
    }

    /// Create a constant term
    pub fn const_(name: &str) -> Self {
        Self::Const(name.to_string())
    }

    /// Create an integer term
    pub fn int(n: i64) -> Self {
        Self::Int(n)
    }

    /// Check if term is a variable
    pub fn is_var(&self) -> bool {
        matches!(self, Self::Var(_))
    }

    /// Get variable name if this is a variable
    pub fn var_name(&self) -> Option<&str> {
        match self {
            Self::Var(n) => Some(n),
            _ => None,
        }
    }

    /// Apply substitution to term
    pub fn apply_subst(&self, subst: &Substitution) -> Self {
        match self {
            Self::Var(v) => subst.get(v).cloned().unwrap_or_else(|| self.clone()),
            Self::Const(_) | Self::Int(_) => self.clone(),
            Self::Func(name, args) => Self::Func(
                name.clone(),
                args.iter().map(|a| a.apply_subst(subst)).collect(),
            ),
        }
    }

    /// Get all variables in term
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut HashSet<String>) {
        match self {
            Self::Var(v) => {
                vars.insert(v.clone());
            }
            Self::Const(_) | Self::Int(_) => {}
            Self::Func(_, args) => {
                for arg in args {
                    arg.collect_vars(vars);
                }
            }
        }
    }
}

impl std::fmt::Display for ChrTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Var(v) => write!(f, "{}", v),
            Self::Const(c) => write!(f, "{}", c),
            Self::Int(n) => write!(f, "{}", n),
            Self::Func(name, args) => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
        }
    }
}

/// Variable substitution mapping
pub type Substitution = HashMap<String, ChrTerm>;

/// A constraint in the constraint store
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Constraint {
    /// Constraint predicate name
    pub name: String,
    /// Arguments
    pub args: Vec<ChrTerm>,
    /// Unique constraint ID
    pub id: usize,
}

impl Constraint {
    /// Create a new constraint
    pub fn new(name: &str, args: Vec<ChrTerm>) -> Self {
        Self {
            name: name.to_string(),
            args,
            id: 0,
        }
    }

    /// Create a unary constraint
    pub fn unary(name: &str, arg: &str) -> Self {
        Self::new(name, vec![ChrTerm::var(arg)])
    }

    /// Create a binary constraint
    pub fn binary(name: &str, arg1: &str, arg2: &str) -> Self {
        Self::new(name, vec![ChrTerm::var(arg1), ChrTerm::var(arg2)])
    }

    /// Create a binary constraint with constants
    pub fn binary_const(name: &str, arg1: &str, arg2: &str) -> Self {
        Self::new(name, vec![ChrTerm::const_(arg1), ChrTerm::const_(arg2)])
    }

    /// Create an equality constraint
    pub fn eq(arg1: &str, arg2: &str) -> Self {
        Self::new("=", vec![ChrTerm::var(arg1), ChrTerm::var(arg2)])
    }

    /// Create an inequality constraint
    pub fn neq(arg1: &str, arg2: &str) -> Self {
        Self::new("\\=", vec![ChrTerm::var(arg1), ChrTerm::var(arg2)])
    }

    /// Create a less-than-or-equal constraint
    pub fn leq(arg1: &str, arg2: &str) -> Self {
        Self::new("leq", vec![ChrTerm::var(arg1), ChrTerm::var(arg2)])
    }

    /// Apply substitution to constraint
    pub fn apply_subst(&self, subst: &Substitution) -> Self {
        Self {
            name: self.name.clone(),
            args: self.args.iter().map(|a| a.apply_subst(subst)).collect(),
            id: self.id,
        }
    }

    /// Get all variables in constraint
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        for arg in &self.args {
            vars.extend(arg.variables());
        }
        vars
    }

    /// Check if constraint matches pattern (for rule matching)
    pub fn matches(&self, pattern: &Constraint, subst: &mut Substitution) -> bool {
        if self.name != pattern.name || self.args.len() != pattern.args.len() {
            return false;
        }

        for (self_arg, pattern_arg) in self.args.iter().zip(pattern.args.iter()) {
            if !Self::term_matches(self_arg, pattern_arg, subst) {
                return false;
            }
        }

        true
    }

    fn term_matches(term: &ChrTerm, pattern: &ChrTerm, subst: &mut Substitution) -> bool {
        match (term, pattern) {
            (_, ChrTerm::Var(v)) => {
                if let Some(bound) = subst.get(v) {
                    bound == term
                } else {
                    subst.insert(v.clone(), term.clone());
                    true
                }
            }
            (ChrTerm::Const(c1), ChrTerm::Const(c2)) => c1 == c2,
            (ChrTerm::Int(n1), ChrTerm::Int(n2)) => n1 == n2,
            (ChrTerm::Func(n1, args1), ChrTerm::Func(n2, args2)) => {
                if n1 != n2 || args1.len() != args2.len() {
                    return false;
                }
                for (a1, a2) in args1.iter().zip(args2.iter()) {
                    if !Self::term_matches(a1, a2, subst) {
                        return false;
                    }
                }
                true
            }
            _ => false,
        }
    }
}

impl std::fmt::Display for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.args.is_empty() {
            write!(f, "{}", self.name)
        } else {
            write!(f, "{}(", self.name)?;
            for (i, arg) in self.args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", arg)?;
            }
            write!(f, ")")
        }
    }
}

/// CHR rule type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChrRuleType {
    /// Simplification: H <=> G | B (removes head, adds body)
    Simplification,
    /// Propagation: H ==> G | B (keeps head, adds body)
    Propagation,
    /// Simpagation: H1 \ H2 <=> G | B (keeps H1, removes H2, adds body)
    Simpagation,
}

/// Guard condition for CHR rules
#[derive(Debug, Clone)]
pub enum Guard {
    /// True (always succeeds)
    True,
    /// Equality check
    Equal(ChrTerm, ChrTerm),
    /// Inequality check
    NotEqual(ChrTerm, ChrTerm),
    /// Less than
    LessThan(ChrTerm, ChrTerm),
    /// Less than or equal
    LessEq(ChrTerm, ChrTerm),
    /// Greater than
    GreaterThan(ChrTerm, ChrTerm),
    /// Greater than or equal
    GreaterEq(ChrTerm, ChrTerm),
    /// Conjunction of guards
    And(Vec<Guard>),
    /// Disjunction of guards
    Or(Vec<Guard>),
    /// Negation
    Not(Box<Guard>),
    /// Built-in predicate
    Builtin(String, Vec<ChrTerm>),
}

impl Guard {
    /// Evaluate guard with given substitution
    pub fn evaluate(&self, subst: &Substitution) -> bool {
        match self {
            Self::True => true,
            Self::Equal(t1, t2) => {
                let v1 = t1.apply_subst(subst);
                let v2 = t2.apply_subst(subst);
                v1 == v2
            }
            Self::NotEqual(t1, t2) => {
                let v1 = t1.apply_subst(subst);
                let v2 = t2.apply_subst(subst);
                v1 != v2
            }
            Self::LessThan(t1, t2) => Self::compare_terms(t1, t2, subst, |a, b| a < b),
            Self::LessEq(t1, t2) => Self::compare_terms(t1, t2, subst, |a, b| a <= b),
            Self::GreaterThan(t1, t2) => Self::compare_terms(t1, t2, subst, |a, b| a > b),
            Self::GreaterEq(t1, t2) => Self::compare_terms(t1, t2, subst, |a, b| a >= b),
            Self::And(guards) => guards.iter().all(|g| g.evaluate(subst)),
            Self::Or(guards) => guards.iter().any(|g| g.evaluate(subst)),
            Self::Not(inner) => !inner.evaluate(subst),
            Self::Builtin(name, _args) => {
                // Handle built-in predicates
                match name.as_str() {
                    "true" => true,
                    "false" => false,
                    _ => true, // Default to true for unknown builtins
                }
            }
        }
    }

    fn compare_terms<F>(t1: &ChrTerm, t2: &ChrTerm, subst: &Substitution, cmp: F) -> bool
    where
        F: Fn(i64, i64) -> bool,
    {
        let v1 = t1.apply_subst(subst);
        let v2 = t2.apply_subst(subst);

        match (&v1, &v2) {
            (ChrTerm::Int(n1), ChrTerm::Int(n2)) => cmp(*n1, *n2),
            _ => false,
        }
    }
}

/// CHR rule
#[derive(Debug, Clone)]
pub struct ChrRule {
    /// Rule name
    pub name: String,
    /// Rule type
    pub rule_type: ChrRuleType,
    /// Head constraints to keep (for simpagation)
    pub kept_head: Vec<Constraint>,
    /// Head constraints to remove
    pub removed_head: Vec<Constraint>,
    /// Guard conditions
    pub guard: Guard,
    /// Body constraints to add
    pub body: Vec<Constraint>,
    /// Priority (lower = higher priority)
    pub priority: i32,
}

impl ChrRule {
    /// Create a simplification rule: H <=> G | B
    pub fn simplification(
        name: &str,
        head: Vec<Constraint>,
        guards: Vec<Guard>,
        body: Vec<Constraint>,
    ) -> Self {
        Self {
            name: name.to_string(),
            rule_type: ChrRuleType::Simplification,
            kept_head: vec![],
            removed_head: head,
            guard: if guards.is_empty() {
                Guard::True
            } else {
                Guard::And(guards)
            },
            body,
            priority: 0,
        }
    }

    /// Create a propagation rule: H ==> G | B
    pub fn propagation(
        name: &str,
        head: Vec<Constraint>,
        guards: Vec<Guard>,
        body: Vec<Constraint>,
    ) -> Self {
        Self {
            name: name.to_string(),
            rule_type: ChrRuleType::Propagation,
            kept_head: head,
            removed_head: vec![],
            guard: if guards.is_empty() {
                Guard::True
            } else {
                Guard::And(guards)
            },
            body,
            priority: 0,
        }
    }

    /// Create a simpagation rule: H1 \ H2 <=> G | B
    pub fn simpagation(
        name: &str,
        kept: Vec<Constraint>,
        removed: Vec<Constraint>,
        guards: Vec<Guard>,
        body: Vec<Constraint>,
    ) -> Self {
        Self {
            name: name.to_string(),
            rule_type: ChrRuleType::Simpagation,
            kept_head: kept,
            removed_head: removed,
            guard: if guards.is_empty() {
                Guard::True
            } else {
                Guard::And(guards)
            },
            body,
            priority: 0,
        }
    }

    /// Get all head constraints
    pub fn all_head(&self) -> Vec<&Constraint> {
        self.kept_head
            .iter()
            .chain(self.removed_head.iter())
            .collect()
    }

    /// Get variables in rule
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        for c in &self.kept_head {
            vars.extend(c.variables());
        }
        for c in &self.removed_head {
            vars.extend(c.variables());
        }
        for c in &self.body {
            vars.extend(c.variables());
        }
        vars
    }
}

impl std::fmt::Display for ChrRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: ", self.name)?;

        // Head
        if !self.kept_head.is_empty() {
            for (i, c) in self.kept_head.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", c)?;
            }
            if !self.removed_head.is_empty() {
                write!(f, " \\ ")?;
            }
        }

        for (i, c) in self.removed_head.iter().enumerate() {
            if i > 0 || !self.kept_head.is_empty() {
                write!(f, ", ")?;
            }
            write!(f, "{}", c)?;
        }

        // Rule operator
        match self.rule_type {
            ChrRuleType::Simplification => write!(f, " <=> ")?,
            ChrRuleType::Propagation => write!(f, " ==> ")?,
            ChrRuleType::Simpagation => write!(f, " <=> ")?,
        }

        // Body
        if self.body.is_empty() {
            write!(f, "true")?;
        } else {
            for (i, c) in self.body.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", c)?;
            }
        }

        Ok(())
    }
}

/// Propagation history to prevent infinite propagation
#[derive(Debug, Default)]
struct PropagationHistory {
    /// Set of (rule_name, constraint_ids) that have been applied
    applied: HashSet<(String, Vec<usize>)>,
}

impl PropagationHistory {
    fn new() -> Self {
        Self::default()
    }

    fn has_fired(&self, rule_name: &str, constraint_ids: &[usize]) -> bool {
        let key = (rule_name.to_string(), constraint_ids.to_vec());
        self.applied.contains(&key)
    }

    fn record(&mut self, rule_name: &str, constraint_ids: &[usize]) {
        let key = (rule_name.to_string(), constraint_ids.to_vec());
        self.applied.insert(key);
    }

    fn clear(&mut self) {
        self.applied.clear();
    }
}

/// Constraint store
#[derive(Debug, Default)]
pub struct ConstraintStore {
    /// All constraints
    constraints: Vec<Constraint>,
    /// Constraints indexed by predicate name
    index: HashMap<String, HashSet<usize>>,
    /// Next constraint ID
    next_id: usize,
    /// Removed constraint IDs
    removed: HashSet<usize>,
}

impl ConstraintStore {
    /// Create a new empty store
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a constraint to the store
    pub fn add(&mut self, mut constraint: Constraint) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        constraint.id = id;

        self.index
            .entry(constraint.name.clone())
            .or_default()
            .insert(id);

        self.constraints.push(constraint);
        id
    }

    /// Remove a constraint by ID
    pub fn remove(&mut self, id: usize) {
        if let Some(constraint) = self.constraints.iter().find(|c| c.id == id) {
            if let Some(ids) = self.index.get_mut(&constraint.name) {
                ids.remove(&id);
            }
        }
        self.removed.insert(id);
    }

    /// Get constraint by ID
    pub fn get(&self, id: usize) -> Option<&Constraint> {
        if self.removed.contains(&id) {
            return None;
        }
        self.constraints.iter().find(|c| c.id == id)
    }

    /// Get all active constraints
    pub fn all(&self) -> Vec<&Constraint> {
        self.constraints
            .iter()
            .filter(|c| !self.removed.contains(&c.id))
            .collect()
    }

    /// Get constraints by predicate name
    pub fn by_name(&self, name: &str) -> Vec<&Constraint> {
        if let Some(ids) = self.index.get(name) {
            ids.iter()
                .filter(|id| !self.removed.contains(*id))
                .filter_map(|id| self.get(*id))
                .collect()
        } else {
            vec![]
        }
    }

    /// Check if store contains a constraint
    pub fn contains(&self, constraint: &Constraint) -> bool {
        self.all().iter().any(|c| {
            c.name == constraint.name
                && c.args.len() == constraint.args.len()
                && c.args
                    .iter()
                    .zip(constraint.args.iter())
                    .all(|(a, b)| a == b)
        })
    }

    /// Get number of active constraints
    pub fn len(&self) -> usize {
        self.constraints.len() - self.removed.len()
    }

    /// Check if store is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear the store
    pub fn clear(&mut self) {
        self.constraints.clear();
        self.index.clear();
        self.next_id = 0;
        self.removed.clear();
    }
}

/// CHR engine configuration
#[derive(Debug, Clone)]
pub struct ChrConfig {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Enable propagation history
    pub use_history: bool,
    /// Enable constraint simplification
    pub simplify: bool,
}

impl Default for ChrConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10000,
            use_history: true,
            simplify: true,
        }
    }
}

/// CHR engine statistics
#[derive(Debug, Default, Clone)]
pub struct ChrStats {
    /// Number of rule applications
    pub rule_applications: usize,
    /// Number of propagations
    pub propagations: usize,
    /// Number of simplifications
    pub simplifications: usize,
    /// Number of iterations
    pub iterations: usize,
    /// Number of constraints added
    pub constraints_added: usize,
    /// Number of constraints removed
    pub constraints_removed: usize,
}

/// CHR execution engine
#[derive(Debug)]
pub struct ChrEngine {
    /// Rules
    rules: Vec<ChrRule>,
    /// Constraint store
    store: ConstraintStore,
    /// Propagation history
    history: PropagationHistory,
    /// Configuration
    config: ChrConfig,
    /// Statistics
    stats: ChrStats,
    /// Work queue (newly added constraint IDs)
    work_queue: VecDeque<usize>,
}

impl Default for ChrEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ChrEngine {
    /// Create a new CHR engine
    pub fn new() -> Self {
        Self::with_config(ChrConfig::default())
    }

    /// Create engine with custom configuration
    pub fn with_config(config: ChrConfig) -> Self {
        Self {
            rules: Vec::new(),
            store: ConstraintStore::new(),
            history: PropagationHistory::new(),
            config,
            stats: ChrStats::default(),
            work_queue: VecDeque::new(),
        }
    }

    /// Add a rule to the engine
    pub fn add_rule(&mut self, rule: ChrRule) {
        self.rules.push(rule);
    }

    /// Add multiple rules
    pub fn add_rules(&mut self, rules: Vec<ChrRule>) {
        self.rules.extend(rules);
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: Constraint) {
        let id = self.store.add(constraint);
        self.work_queue.push_back(id);
        self.stats.constraints_added += 1;
    }

    /// Add multiple constraints
    pub fn add_constraints(&mut self, constraints: Vec<Constraint>) {
        for c in constraints {
            self.add_constraint(c);
        }
    }

    /// Get all active constraints
    pub fn constraints(&self) -> Vec<&Constraint> {
        self.store.all()
    }

    /// Get statistics
    pub fn stats(&self) -> &ChrStats {
        &self.stats
    }

    /// Clear all constraints
    pub fn clear(&mut self) {
        self.store.clear();
        self.history.clear();
        self.stats = ChrStats::default();
        self.work_queue.clear();
    }

    /// Solve constraints by applying rules until fixpoint
    pub fn solve(&mut self) -> Result<Vec<Constraint>> {
        self.stats.iterations = 0;

        while self.stats.iterations < self.config.max_iterations {
            self.stats.iterations += 1;

            let mut applied = false;

            // Try to apply rules
            for rule_idx in 0..self.rules.len() {
                if self.try_apply_rule(rule_idx)? {
                    applied = true;
                    break; // Restart from first rule
                }
            }

            if !applied {
                break;
            }
        }

        // Return final constraint store
        Ok(self.store.all().into_iter().cloned().collect())
    }

    /// Try to apply a specific rule
    fn try_apply_rule(&mut self, rule_idx: usize) -> Result<bool> {
        let rule = self.rules[rule_idx].clone();

        // Find matching constraints for the head
        let matches = self.find_matching_constraints(&rule)?;

        for (subst, matched_ids) in matches {
            // Check propagation history
            if rule.rule_type == ChrRuleType::Propagation
                && self.config.use_history
                && self.history.has_fired(&rule.name, &matched_ids)
            {
                continue;
            }

            // Check guard
            if !rule.guard.evaluate(&subst) {
                continue;
            }

            // Apply rule
            self.apply_rule(&rule, &subst, &matched_ids)?;
            return Ok(true);
        }

        Ok(false)
    }

    /// Find constraints matching rule head
    fn find_matching_constraints(&self, rule: &ChrRule) -> Result<Vec<(Substitution, Vec<usize>)>> {
        let all_head: Vec<&Constraint> = rule.all_head();

        if all_head.is_empty() {
            return Ok(vec![(Substitution::new(), vec![])]);
        }

        // Find matches for first head constraint
        let first = all_head[0];
        let candidates = self.store.by_name(&first.name);

        let mut results = Vec::new();

        for candidate in candidates {
            let mut subst = Substitution::new();
            if candidate.matches(first, &mut subst) {
                // Try to match remaining head constraints
                let remaining: Vec<&&Constraint> = all_head.iter().skip(1).collect();
                let matches = self.match_remaining(&remaining, &subst, &[candidate.id])?;

                for (s, ids) in matches {
                    results.push((s, ids));
                }
            }
        }

        Ok(results)
    }

    /// Match remaining head constraints
    fn match_remaining(
        &self,
        remaining: &[&&Constraint],
        subst: &Substitution,
        matched_ids: &[usize],
    ) -> Result<Vec<(Substitution, Vec<usize>)>> {
        if remaining.is_empty() {
            return Ok(vec![(subst.clone(), matched_ids.to_vec())]);
        }

        let first = remaining[0];
        let applied_first = first.apply_subst(subst);
        let candidates = self.store.by_name(&applied_first.name);

        let mut results = Vec::new();

        for candidate in candidates {
            // Skip already matched constraints
            if matched_ids.contains(&candidate.id) {
                continue;
            }

            let mut new_subst = subst.clone();
            if candidate.matches(&applied_first, &mut new_subst) {
                let mut new_ids = matched_ids.to_vec();
                new_ids.push(candidate.id);

                let rest: Vec<&&Constraint> = remaining.iter().skip(1).copied().collect();
                let sub_matches = self.match_remaining(&rest, &new_subst, &new_ids)?;

                results.extend(sub_matches);
            }
        }

        Ok(results)
    }

    /// Apply a rule with given substitution
    fn apply_rule(
        &mut self,
        rule: &ChrRule,
        subst: &Substitution,
        matched_ids: &[usize],
    ) -> Result<()> {
        self.stats.rule_applications += 1;

        // Record propagation
        if rule.rule_type == ChrRuleType::Propagation && self.config.use_history {
            self.history.record(&rule.name, matched_ids);
            self.stats.propagations += 1;
        }

        // Remove constraints (for simplification and simpagation)
        if rule.rule_type == ChrRuleType::Simplification
            || rule.rule_type == ChrRuleType::Simpagation
        {
            // Determine which constraints to remove
            let kept_count = rule.kept_head.len();
            for (i, id) in matched_ids.iter().enumerate() {
                // For simpagation, only remove constraints after kept_head
                if rule.rule_type == ChrRuleType::Simpagation && i < kept_count {
                    continue;
                }
                self.store.remove(*id);
                self.stats.constraints_removed += 1;
            }
            self.stats.simplifications += 1;
        }

        // Add body constraints
        for body_constraint in &rule.body {
            let new_constraint = body_constraint.apply_subst(subst);

            // Check for equality constraints and handle unification
            if new_constraint.name == "=" && new_constraint.args.len() == 2 {
                // This is an equality constraint - could trigger unification
                // For now, just add it to store
                self.add_constraint(new_constraint);
            } else {
                // Don't add duplicates
                if !self.store.contains(&new_constraint) {
                    self.add_constraint(new_constraint);
                }
            }
        }

        Ok(())
    }

    /// Convert from OxiRS Term
    pub fn term_from_oxirs(term: &Term) -> ChrTerm {
        match term {
            Term::Variable(v) => ChrTerm::Var(v.clone()),
            Term::Constant(c) => {
                // Try to parse as integer
                if let Ok(n) = c.parse::<i64>() {
                    ChrTerm::Int(n)
                } else {
                    ChrTerm::Const(c.clone())
                }
            }
            Term::Literal(l) => ChrTerm::Const(l.clone()),
            Term::Function { name, args } => ChrTerm::Func(
                name.clone(),
                args.iter().map(Self::term_from_oxirs).collect(),
            ),
        }
    }

    /// Convert to OxiRS Term
    pub fn term_to_oxirs(term: &ChrTerm) -> Term {
        match term {
            ChrTerm::Var(v) => Term::Variable(v.clone()),
            ChrTerm::Const(c) => Term::Constant(c.clone()),
            ChrTerm::Int(n) => Term::Constant(n.to_string()),
            ChrTerm::Func(name, args) => Term::Function {
                name: name.clone(),
                args: args.iter().map(Self::term_to_oxirs).collect(),
            },
        }
    }
}

// =============================================================================
// CHR Parser
// =============================================================================

/// Simple CHR rule parser
pub struct ChrParser;

impl ChrParser {
    /// Parse a CHR rule from string
    pub fn parse_rule(input: &str) -> Result<ChrRule> {
        let input = input.trim();

        // Find rule name (before colon)
        let (name, rest) = if let Some(colon_pos) = input.find(':') {
            let name = input[..colon_pos].trim();
            let rest = input[colon_pos + 1..].trim();
            (name.to_string(), rest.to_string())
        } else {
            ("rule".to_string(), input.to_string())
        };

        // Determine rule type and split
        let (rule_type, head_str, body_str) = if rest.contains("<=>") {
            let parts: Vec<&str> = rest.splitn(2, "<=>").collect();
            (
                ChrRuleType::Simplification,
                parts[0].trim(),
                parts.get(1).map(|s| s.trim()).unwrap_or("true"),
            )
        } else if rest.contains("==>") {
            let parts: Vec<&str> = rest.splitn(2, "==>").collect();
            (
                ChrRuleType::Propagation,
                parts[0].trim(),
                parts.get(1).map(|s| s.trim()).unwrap_or("true"),
            )
        } else {
            return Err(anyhow!("Invalid CHR rule syntax: missing <=> or ==>"));
        };

        // Check for simpagation (\ in head)
        let (kept_head, removed_head) = if head_str.contains('\\') {
            let parts: Vec<&str> = head_str.splitn(2, '\\').collect();
            let kept = Self::parse_constraints(parts[0].trim())?;
            let removed = Self::parse_constraints(parts.get(1).map(|s| s.trim()).unwrap_or(""))?;
            (kept, removed)
        } else {
            let constraints = Self::parse_constraints(head_str)?;
            match rule_type {
                ChrRuleType::Simplification => (vec![], constraints),
                ChrRuleType::Propagation => (constraints, vec![]),
                ChrRuleType::Simpagation => (vec![], constraints),
            }
        };

        // Parse guard and body
        let (guard, body_constraints) = if body_str.contains('|') {
            let parts: Vec<&str> = body_str.splitn(2, '|').collect();
            let guard = Self::parse_guard(parts[0].trim())?;
            let body = Self::parse_constraints(parts.get(1).map(|s| s.trim()).unwrap_or("true"))?;
            (guard, body)
        } else {
            (Guard::True, Self::parse_constraints(body_str)?)
        };

        Ok(ChrRule {
            name,
            rule_type: if !kept_head.is_empty() && !removed_head.is_empty() {
                ChrRuleType::Simpagation
            } else {
                rule_type
            },
            kept_head,
            removed_head,
            guard,
            body: body_constraints,
            priority: 0,
        })
    }

    /// Parse a list of constraints
    fn parse_constraints(input: &str) -> Result<Vec<Constraint>> {
        if input.is_empty() || input == "true" {
            return Ok(vec![]);
        }

        let mut constraints = Vec::new();
        let mut depth = 0;
        let mut start = 0;

        for (i, c) in input.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                ',' if depth == 0 => {
                    let part = input[start..i].trim();
                    if !part.is_empty() {
                        constraints.push(Self::parse_constraint(part)?);
                    }
                    start = i + 1;
                }
                _ => {}
            }
        }

        let last = input[start..].trim();
        if !last.is_empty() && last != "true" {
            constraints.push(Self::parse_constraint(last)?);
        }

        Ok(constraints)
    }

    /// Parse a single constraint
    fn parse_constraint(input: &str) -> Result<Constraint> {
        let input = input.trim();

        // Check for equality
        if input.contains('=') && !input.contains("\\=") {
            let parts: Vec<&str> = input.splitn(2, '=').collect();
            return Ok(Constraint::new(
                "=",
                vec![
                    Self::parse_term(parts[0].trim())?,
                    Self::parse_term(parts.get(1).map(|s| s.trim()).unwrap_or(""))?,
                ],
            ));
        }

        // Parse predicate(args) format
        if let Some(paren_pos) = input.find('(') {
            let name = input[..paren_pos].trim();
            let args_str = input[paren_pos + 1..].trim_end_matches(')');
            let args = Self::parse_args(args_str)?;
            Ok(Constraint::new(name, args))
        } else {
            // Nullary predicate
            Ok(Constraint::new(input, vec![]))
        }
    }

    /// Parse constraint arguments
    fn parse_args(input: &str) -> Result<Vec<ChrTerm>> {
        if input.is_empty() {
            return Ok(vec![]);
        }

        let mut args = Vec::new();
        let mut depth = 0;
        let mut start = 0;

        for (i, c) in input.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                ',' if depth == 0 => {
                    let part = input[start..i].trim();
                    if !part.is_empty() {
                        args.push(Self::parse_term(part)?);
                    }
                    start = i + 1;
                }
                _ => {}
            }
        }

        let last = input[start..].trim();
        if !last.is_empty() {
            args.push(Self::parse_term(last)?);
        }

        Ok(args)
    }

    /// Parse a single term
    fn parse_term(input: &str) -> Result<ChrTerm> {
        let input = input.trim();

        if input.is_empty() {
            return Err(anyhow!("Empty term"));
        }

        // Check for function
        if let Some(paren_pos) = input.find('(') {
            let name = input[..paren_pos].trim();
            let args_str = input[paren_pos + 1..].trim_end_matches(')');
            let args = Self::parse_args(args_str)?;
            return Ok(ChrTerm::Func(name.to_string(), args));
        }

        // Check for integer
        if let Ok(n) = input.parse::<i64>() {
            return Ok(ChrTerm::Int(n));
        }

        // Variable (uppercase first char) or constant
        if input
            .chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false)
        {
            Ok(ChrTerm::Var(input.to_string()))
        } else {
            Ok(ChrTerm::Const(input.to_string()))
        }
    }

    /// Parse a guard expression
    fn parse_guard(input: &str) -> Result<Guard> {
        let input = input.trim();

        if input.is_empty() || input == "true" {
            return Ok(Guard::True);
        }

        // Check for comparison operators
        if input.contains("\\=") {
            let parts: Vec<&str> = input.splitn(2, "\\=").collect();
            return Ok(Guard::NotEqual(
                Self::parse_term(parts[0].trim())?,
                Self::parse_term(parts.get(1).map(|s| s.trim()).unwrap_or(""))?,
            ));
        }

        if input.contains(">=") {
            let parts: Vec<&str> = input.splitn(2, ">=").collect();
            return Ok(Guard::GreaterEq(
                Self::parse_term(parts[0].trim())?,
                Self::parse_term(parts.get(1).map(|s| s.trim()).unwrap_or(""))?,
            ));
        }

        if input.contains("<=") {
            let parts: Vec<&str> = input.splitn(2, "<=").collect();
            return Ok(Guard::LessEq(
                Self::parse_term(parts[0].trim())?,
                Self::parse_term(parts.get(1).map(|s| s.trim()).unwrap_or(""))?,
            ));
        }

        if input.contains('>') {
            let parts: Vec<&str> = input.splitn(2, '>').collect();
            return Ok(Guard::GreaterThan(
                Self::parse_term(parts[0].trim())?,
                Self::parse_term(parts.get(1).map(|s| s.trim()).unwrap_or(""))?,
            ));
        }

        if input.contains('<') {
            let parts: Vec<&str> = input.splitn(2, '<').collect();
            return Ok(Guard::LessThan(
                Self::parse_term(parts[0].trim())?,
                Self::parse_term(parts.get(1).map(|s| s.trim()).unwrap_or(""))?,
            ));
        }

        if input.contains('=') {
            let parts: Vec<&str> = input.splitn(2, '=').collect();
            return Ok(Guard::Equal(
                Self::parse_term(parts[0].trim())?,
                Self::parse_term(parts.get(1).map(|s| s.trim()).unwrap_or(""))?,
            ));
        }

        // Default to builtin
        Ok(Guard::Builtin(input.to_string(), vec![]))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chr_term_creation() {
        let var = ChrTerm::var("X");
        assert!(var.is_var());
        assert_eq!(var.var_name(), Some("X"));

        let const_ = ChrTerm::const_("foo");
        assert!(!const_.is_var());

        let int = ChrTerm::int(42);
        assert!(matches!(int, ChrTerm::Int(42)));
    }

    #[test]
    fn test_chr_term_substitution() {
        let var = ChrTerm::var("X");
        let mut subst = Substitution::new();
        subst.insert("X".to_string(), ChrTerm::const_("value"));

        let result = var.apply_subst(&subst);
        assert_eq!(result, ChrTerm::const_("value"));
    }

    #[test]
    fn test_constraint_creation() {
        let c = Constraint::binary("leq", "X", "Y");
        assert_eq!(c.name, "leq");
        assert_eq!(c.args.len(), 2);
    }

    #[test]
    fn test_constraint_matching() {
        let pattern = Constraint::binary("leq", "X", "Y");
        let instance = Constraint::new("leq", vec![ChrTerm::const_("a"), ChrTerm::const_("b")]);

        let mut subst = Substitution::new();
        assert!(instance.matches(&pattern, &mut subst));
        assert_eq!(subst.get("X"), Some(&ChrTerm::const_("a")));
        assert_eq!(subst.get("Y"), Some(&ChrTerm::const_("b")));
    }

    #[test]
    fn test_chr_rule_simplification() {
        let rule = ChrRule::simplification(
            "reflexivity",
            vec![Constraint::binary("leq", "X", "X")],
            vec![],
            vec![], // true body
        );

        assert_eq!(rule.rule_type, ChrRuleType::Simplification);
        assert!(rule.kept_head.is_empty());
        assert_eq!(rule.removed_head.len(), 1);
    }

    #[test]
    fn test_chr_rule_propagation() {
        let rule = ChrRule::propagation(
            "transitivity",
            vec![
                Constraint::binary("leq", "X", "Y"),
                Constraint::binary("leq", "Y", "Z"),
            ],
            vec![Guard::NotEqual(ChrTerm::var("X"), ChrTerm::var("Z"))],
            vec![Constraint::binary("leq", "X", "Z")],
        );

        assert_eq!(rule.rule_type, ChrRuleType::Propagation);
        assert_eq!(rule.kept_head.len(), 2);
    }

    #[test]
    fn test_constraint_store() {
        let mut store = ConstraintStore::new();

        let id1 = store.add(Constraint::binary("leq", "a", "b"));
        let id2 = store.add(Constraint::binary("leq", "b", "c"));

        assert_eq!(store.len(), 2);
        assert!(store.get(id1).is_some());
        assert!(store.get(id2).is_some());

        store.remove(id1);
        assert_eq!(store.len(), 1);
        assert!(store.get(id1).is_none());
    }

    #[test]
    fn test_chr_engine_basic() {
        let mut engine = ChrEngine::new();

        // Reflexivity rule: leq(X, X) <=> true
        engine.add_rule(ChrRule::simplification(
            "reflexivity",
            vec![Constraint::binary("leq", "X", "X")],
            vec![],
            vec![],
        ));

        // Add reflexive constraint
        engine.add_constraint(Constraint::new(
            "leq",
            vec![ChrTerm::const_("a"), ChrTerm::const_("a")],
        ));

        let result = engine.solve().unwrap();
        // Should be simplified away
        assert!(result.is_empty());
    }

    #[test]
    fn test_chr_antisymmetry() {
        let mut engine = ChrEngine::new();

        // Antisymmetry: leq(X, Y), leq(Y, X) <=> X = Y
        engine.add_rule(ChrRule::simplification(
            "antisymmetry",
            vec![
                Constraint::binary("leq", "X", "Y"),
                Constraint::binary("leq", "Y", "X"),
            ],
            vec![],
            vec![Constraint::eq("X", "Y")],
        ));

        engine.add_constraint(Constraint::new(
            "leq",
            vec![ChrTerm::const_("a"), ChrTerm::const_("b")],
        ));
        engine.add_constraint(Constraint::new(
            "leq",
            vec![ChrTerm::const_("b"), ChrTerm::const_("a")],
        ));

        let result = engine.solve().unwrap();
        // Should produce equality constraint
        assert!(result.iter().any(|c| c.name == "="));
    }

    #[test]
    fn test_chr_propagation() {
        let mut engine = ChrEngine::new();

        // Transitivity propagation: leq(X, Y), leq(Y, Z) ==> leq(X, Z)
        engine.add_rule(ChrRule::propagation(
            "transitivity",
            vec![
                Constraint::binary("leq", "X", "Y"),
                Constraint::binary("leq", "Y", "Z"),
            ],
            vec![],
            vec![Constraint::binary("leq", "X", "Z")],
        ));

        engine.add_constraint(Constraint::new(
            "leq",
            vec![ChrTerm::const_("a"), ChrTerm::const_("b")],
        ));
        engine.add_constraint(Constraint::new(
            "leq",
            vec![ChrTerm::const_("b"), ChrTerm::const_("c")],
        ));

        let result = engine.solve().unwrap();

        // Should contain original constraints plus derived leq(a, c)
        assert!(result.iter().any(|c| {
            c.name == "leq"
                && c.args.len() == 2
                && c.args[0] == ChrTerm::const_("a")
                && c.args[1] == ChrTerm::const_("c")
        }));
    }

    #[test]
    fn test_chr_guard() {
        let guard = Guard::Equal(ChrTerm::var("X"), ChrTerm::const_("value"));

        let mut subst = Substitution::new();
        subst.insert("X".to_string(), ChrTerm::const_("value"));
        assert!(guard.evaluate(&subst));

        subst.insert("X".to_string(), ChrTerm::const_("other"));
        assert!(!guard.evaluate(&subst));
    }

    #[test]
    fn test_chr_parser_simplification() {
        let rule = ChrParser::parse_rule("reflexivity: leq(X, X) <=> true").unwrap();
        assert_eq!(rule.name, "reflexivity");
        assert_eq!(rule.rule_type, ChrRuleType::Simplification);
        assert_eq!(rule.removed_head.len(), 1);
    }

    #[test]
    fn test_chr_parser_propagation() {
        let rule = ChrParser::parse_rule("trans: leq(X, Y), leq(Y, Z) ==> leq(X, Z)").unwrap();
        assert_eq!(rule.name, "trans");
        assert_eq!(rule.rule_type, ChrRuleType::Propagation);
        assert_eq!(rule.kept_head.len(), 2);
        assert_eq!(rule.body.len(), 1);
    }

    #[test]
    fn test_chr_parser_with_guard() {
        let rule = ChrParser::parse_rule("idempotence: leq(X, Y), leq(X, Y) <=> true | leq(X, Y)")
            .unwrap();
        assert_eq!(rule.name, "idempotence");
    }

    #[test]
    fn test_chr_stats() {
        let mut engine = ChrEngine::new();

        engine.add_rule(ChrRule::simplification(
            "test",
            vec![Constraint::binary("test", "X", "X")],
            vec![],
            vec![],
        ));

        engine.add_constraint(Constraint::new(
            "test",
            vec![ChrTerm::const_("a"), ChrTerm::const_("a")],
        ));

        engine.solve().unwrap();

        assert!(engine.stats().rule_applications > 0);
    }

    #[test]
    fn test_chr_term_display() {
        let term = ChrTerm::Func(
            "f".to_string(),
            vec![ChrTerm::var("X"), ChrTerm::const_("a")],
        );
        let display = format!("{}", term);
        assert!(display.contains("f("));
        assert!(display.contains("X"));
        assert!(display.contains("a"));
    }

    #[test]
    fn test_chr_constraint_display() {
        let c = Constraint::binary("leq", "X", "Y");
        let display = format!("{}", c);
        assert!(display.contains("leq"));
        assert!(display.contains("X"));
        assert!(display.contains("Y"));
    }

    #[test]
    fn test_chr_rule_display() {
        let rule = ChrRule::simplification(
            "test",
            vec![Constraint::binary("p", "X", "Y")],
            vec![],
            vec![Constraint::binary("q", "X", "Y")],
        );
        let display = format!("{}", rule);
        assert!(display.contains("test"));
        assert!(display.contains("<=>"));
    }
}
