//! Variable binding optimization for query execution
//!
//! This module provides efficient variable binding management with
//! constraint propagation and early pruning of invalid bindings.

use crate::model::*;
use crate::query::algebra::{Expression, TermPattern};
use crate::OxirsError;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

/// Optimized binding set for variable management
#[derive(Debug, Clone)]
pub struct BindingSet {
    /// Variables in this binding set (using SmallVec for cache efficiency)
    pub variables: SmallVec<[Variable; 8]>,
    /// Current bindings
    pub bindings: Vec<TermBinding>,
    /// Constraints on variables
    pub constraints: Vec<Constraint>,
    /// Index for fast variable lookup
    var_index: HashMap<Variable, usize>,
}

/// A single variable binding
#[derive(Debug, Clone)]
pub struct TermBinding {
    /// The variable being bound
    pub variable: Variable,
    /// The term it's bound to
    pub term: Term,
    /// Binding metadata
    pub metadata: BindingMetadata,
}

/// Metadata about a binding
#[derive(Debug, Clone, Default)]
pub struct BindingMetadata {
    /// Source pattern that created this binding
    pub source_pattern_id: usize,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Whether this binding is fixed (cannot be changed)
    pub is_fixed: bool,
}

/// Constraint on variable bindings
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Type constraint (e.g., must be a literal)
    TypeConstraint {
        variable: Variable,
        allowed_types: HashSet<TermType>,
    },
    /// Value constraint (e.g., numeric range)
    ValueConstraint {
        variable: Variable,
        constraint: ValueConstraintType,
    },
    /// Relationship constraint between variables
    RelationshipConstraint {
        left: Variable,
        right: Variable,
        relation: RelationType,
    },
    /// Custom filter expression
    FilterConstraint {
        expression: Expression,
    },
}

/// Types of terms for type constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TermType {
    NamedNode,
    BlankNode,
    Literal,
    NumericLiteral,
    StringLiteral,
    BooleanLiteral,
    DateTimeLiteral,
}

/// Value constraint types
#[derive(Debug, Clone)]
pub enum ValueConstraintType {
    /// Numeric range
    NumericRange { min: f64, max: f64 },
    /// String pattern
    StringPattern(regex::Regex),
    /// One of a set of values
    OneOf(HashSet<Term>),
    /// Not one of a set of values
    NoneOf(HashSet<Term>),
}

/// Relationship types between variables
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationType {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

impl BindingSet {
    /// Create a new empty binding set
    pub fn new() -> Self {
        Self {
            variables: SmallVec::new(),
            bindings: Vec::new(),
            constraints: Vec::new(),
            var_index: HashMap::new(),
        }
    }

    /// Create a binding set with initial variables
    pub fn with_variables(vars: Vec<Variable>) -> Self {
        let mut var_index = HashMap::new();
        for (idx, var) in vars.iter().enumerate() {
            var_index.insert(var.clone(), idx);
        }

        Self {
            variables: vars.into(),
            bindings: Vec::new(),
            constraints: Vec::new(),
            var_index,
        }
    }

    /// Add a variable to the binding set
    pub fn add_variable(&mut self, var: Variable) -> usize {
        if let Some(&idx) = self.var_index.get(&var) {
            idx
        } else {
            let idx = self.variables.len();
            self.variables.push(var.clone());
            self.var_index.insert(var, idx);
            idx
        }
    }

    /// Bind a variable to a term
    pub fn bind(
        &mut self,
        var: Variable,
        term: Term,
        metadata: BindingMetadata,
    ) -> Result<(), OxirsError> {
        // Check if variable exists
        if !self.var_index.contains_key(&var) {
            self.add_variable(var.clone());
        }

        // Check constraints
        self.validate_binding(&var, &term)?;

        // Remove any existing binding for this variable
        self.bindings.retain(|b| b.variable != var);

        // Add new binding
        self.bindings.push(TermBinding {
            variable: var,
            term,
            metadata,
        });

        Ok(())
    }

    /// Get binding for a variable
    pub fn get(&self, var: &Variable) -> Option<&Term> {
        self.bindings
            .iter()
            .find(|b| &b.variable == var)
            .map(|b| &b.term)
    }

    /// Check if a variable is bound
    pub fn is_bound(&self, var: &Variable) -> bool {
        self.bindings.iter().any(|b| &b.variable == var)
    }

    /// Get all unbound variables
    pub fn unbound_variables(&self) -> Vec<&Variable> {
        self.variables
            .iter()
            .filter(|v| !self.is_bound(v))
            .collect()
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Validate a binding against constraints
    fn validate_binding(&self, var: &Variable, term: &Term) -> Result<(), OxirsError> {
        for constraint in &self.constraints {
            match constraint {
                Constraint::TypeConstraint {
                    variable,
                    allowed_types,
                } => {
                    if variable == var && !self.check_type_constraint(term, allowed_types) {
                        return Err(OxirsError::Query(format!(
                            "Type constraint violation for variable {}",
                            var
                        )));
                    }
                }
                Constraint::ValueConstraint {
                    variable,
                    constraint,
                } => {
                    if variable == var && !self.check_value_constraint(term, constraint) {
                        return Err(OxirsError::Query(format!(
                            "Value constraint violation for variable {}",
                            var
                        )));
                    }
                }
                _ => {} // Other constraints checked elsewhere
            }
        }
        Ok(())
    }

    /// Check type constraint
    fn check_type_constraint(&self, term: &Term, allowed_types: &HashSet<TermType>) -> bool {
        let term_type = match term {
            Term::NamedNode(_) => TermType::NamedNode,
            Term::BlankNode(_) => TermType::BlankNode,
            Term::Literal(lit) => {
                let datatype = lit.datatype();
                let datatype_str = datatype.as_str();
                if datatype_str == "http://www.w3.org/2001/XMLSchema#integer"
                    || datatype_str == "http://www.w3.org/2001/XMLSchema#decimal"
                    || datatype_str == "http://www.w3.org/2001/XMLSchema#double"
                {
                    TermType::NumericLiteral
                } else if datatype_str == "http://www.w3.org/2001/XMLSchema#boolean" {
                    TermType::BooleanLiteral
                } else if datatype_str == "http://www.w3.org/2001/XMLSchema#dateTime" {
                    TermType::DateTimeLiteral
                } else if datatype_str == "http://www.w3.org/2001/XMLSchema#string"
                    || datatype_str == "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString"
                {
                    TermType::StringLiteral
                } else {
                    TermType::Literal
                }
            }
            _ => return false, // Variables and QuotedTriples not supported
        };

        allowed_types.contains(&term_type) || allowed_types.contains(&TermType::Literal)
    }

    /// Check value constraint
    fn check_value_constraint(&self, term: &Term, constraint: &ValueConstraintType) -> bool {
        match constraint {
            ValueConstraintType::NumericRange { min, max } => {
                if let Term::Literal(lit) = term {
                    if let Ok(value) = lit.value().parse::<f64>() {
                        return value >= *min && value <= *max;
                    }
                }
                false
            }
            ValueConstraintType::StringPattern(regex) => {
                if let Term::Literal(lit) = term {
                    return regex.is_match(lit.value());
                }
                false
            }
            ValueConstraintType::OneOf(allowed) => allowed.contains(term),
            ValueConstraintType::NoneOf(forbidden) => !forbidden.contains(term),
        }
    }

    /// Clone bindings into a HashMap for easier use
    pub fn to_map(&self) -> HashMap<Variable, Term> {
        self.bindings
            .iter()
            .map(|b| (b.variable.clone(), b.term.clone()))
            .collect()
    }

    /// Merge another binding set into this one
    pub fn merge(&mut self, other: &BindingSet) -> Result<(), OxirsError> {
        // Add variables
        for var in &other.variables {
            self.add_variable(var.clone());
        }

        // Merge bindings
        for binding in &other.bindings {
            // Check for conflicts
            if let Some(existing) = self.get(&binding.variable) {
                if existing != &binding.term {
                    return Err(OxirsError::Query(format!(
                        "Conflicting bindings for variable {}",
                        binding.variable
                    )));
                }
            } else {
                self.bindings.push(binding.clone());
            }
        }

        // Add constraints
        self.constraints.extend(other.constraints.clone());

        Ok(())
    }

    /// Apply bindings to a term pattern
    pub fn apply_to_pattern(&self, pattern: &TermPattern) -> TermPattern {
        match pattern {
            TermPattern::Variable(var) => {
                if let Some(term) = self.get(var) {
                    match term {
                        Term::NamedNode(n) => TermPattern::NamedNode(n.clone()),
                        Term::BlankNode(b) => TermPattern::BlankNode(b.clone()),
                        Term::Literal(l) => TermPattern::Literal(l.clone()),
                        _ => pattern.clone(), // Keep as variable if can't convert
                    }
                } else {
                    pattern.clone()
                }
            }
            _ => pattern.clone(),
        }
    }
}

/// Binding optimizer that manages efficient variable binding
pub struct BindingOptimizer {
    /// Cache of binding sets for reuse
    binding_cache: HashMap<String, Arc<BindingSet>>,
    /// Statistics about binding operations
    stats: BindingStats,
}

/// Statistics about binding operations
#[derive(Debug, Default)]
struct BindingStats {
    /// Total bindings created
    bindings_created: usize,
    /// Cache hits
    cache_hits: usize,
    /// Cache misses
    cache_misses: usize,
    /// Constraint violations
    constraint_violations: usize,
}

impl BindingOptimizer {
    /// Create a new binding optimizer
    pub fn new() -> Self {
        Self {
            binding_cache: HashMap::new(),
            stats: BindingStats::default(),
        }
    }

    /// Optimize bindings for a set of variables
    pub fn optimize_bindings(
        &mut self,
        variables: Vec<Variable>,
        constraints: Vec<Constraint>,
    ) -> Arc<BindingSet> {
        // Create cache key
        let cache_key = self.create_cache_key(&variables, &constraints);

        // Check cache
        if let Some(cached) = self.binding_cache.get(&cache_key) {
            self.stats.cache_hits += 1;
            return Arc::clone(cached);
        }

        self.stats.cache_misses += 1;

        // Create new binding set
        let mut binding_set = BindingSet::with_variables(variables);
        for constraint in constraints {
            binding_set.add_constraint(constraint);
        }

        // Analyze constraints for optimization opportunities
        self.propagate_constraints(&mut binding_set);

        // Cache and return
        let arc_set = Arc::new(binding_set);
        self.binding_cache.insert(cache_key, Arc::clone(&arc_set));
        arc_set
    }

    /// Create cache key from variables and constraints
    fn create_cache_key(&self, variables: &[Variable], constraints: &[Constraint]) -> String {
        let mut key = String::new();
        for var in variables {
            key.push_str(&var.as_str());
            key.push(',');
        }
        key.push('|');
        // Simple constraint representation for caching
        key.push_str(&format!("{}", constraints.len()));
        key
    }

    /// Propagate constraints to find additional restrictions
    fn propagate_constraints(&self, binding_set: &mut BindingSet) {
        // Build constraint graph with indices instead of references
        let mut constraint_graph: HashMap<Variable, Vec<usize>> = HashMap::new();
        
        for (idx, constraint) in binding_set.constraints.iter().enumerate() {
            match constraint {
                Constraint::TypeConstraint { variable, .. }
                | Constraint::ValueConstraint { variable, .. } => {
                    constraint_graph
                        .entry(variable.clone())
                        .or_insert_with(Vec::new)
                        .push(idx);
                }
                Constraint::RelationshipConstraint { left, right, .. } => {
                    constraint_graph
                        .entry(left.clone())
                        .or_insert_with(Vec::new)
                        .push(idx);
                    constraint_graph
                        .entry(right.clone())
                        .or_insert_with(Vec::new)
                        .push(idx);
                }
                _ => {}
            }
        }
        
        // Propagate equality constraints
        self.propagate_equality_constraints(binding_set, constraint_graph);
    }

    /// Propagate equality constraints
    fn propagate_equality_constraints(
        &self,
        binding_set: &mut BindingSet,
        constraint_graph: HashMap<Variable, Vec<usize>>,
    ) {
        // Find equality relationships
        let mut equiv_classes: HashMap<Variable, Variable> = HashMap::new();

        for constraint in &binding_set.constraints {
            if let Constraint::RelationshipConstraint {
                left,
                right,
                relation: RelationType::Equal,
            } = constraint
            {
                // Union-find style merging
                let left_root = self.find_root(left, &equiv_classes);
                let right_root = self.find_root(right, &equiv_classes);
                if left_root != right_root {
                    equiv_classes.insert(left_root, right_root.clone());
                }
            }
        }

        // Apply transitivity of constraints within equivalence classes
        for (var, root) in &equiv_classes {
            if var != root {
                // Copy constraints from root to var
                if let Some(root_constraints) = constraint_graph.get(root) {
                    for &constraint in root_constraints {
                        // Would add derived constraints here
                    }
                }
            }
        }
    }

    /// Find root of equivalence class (union-find)
    fn find_root<'a>(
        &self,
        var: &'a Variable,
        equiv_classes: &'a HashMap<Variable, Variable>,
    ) -> Variable {
        let mut current = var.clone();
        while let Some(parent) = equiv_classes.get(&current) {
            if parent == &current {
                break;
            }
            current = parent.clone();
        }
        current
    }

    /// Get statistics
    pub fn stats(&self) -> String {
        format!(
            "Bindings created: {}, Cache hits: {}, Cache misses: {}, Violations: {}",
            self.stats.bindings_created,
            self.stats.cache_hits,
            self.stats.cache_misses,
            self.stats.constraint_violations
        )
    }
}

/// Binding iterator that yields valid combinations
pub struct BindingIterator {
    /// Base bindings to extend
    base_bindings: Vec<HashMap<Variable, Term>>,
    /// Variables to bind
    variables: Vec<Variable>,
    /// Possible values for each variable
    possible_values: HashMap<Variable, Vec<Term>>,
    /// Current position in iteration
    current_position: Vec<usize>,
    /// Constraints to check
    constraints: Vec<Constraint>,
}

impl BindingIterator {
    /// Create a new binding iterator
    pub fn new(
        base_bindings: Vec<HashMap<Variable, Term>>,
        variables: Vec<Variable>,
        possible_values: HashMap<Variable, Vec<Term>>,
        constraints: Vec<Constraint>,
    ) -> Self {
        let current_position = vec![0; variables.len()];
        Self {
            base_bindings,
            variables,
            possible_values,
            current_position,
            constraints,
        }
    }

    /// Get next valid binding combination
    pub fn next_valid(&mut self) -> Option<HashMap<Variable, Term>> {
        while let Some(binding) = self.next_combination() {
            if self.validate_binding(&binding) {
                return Some(binding);
            }
        }
        None
    }

    /// Get next combination (without validation)
    fn next_combination(&mut self) -> Option<HashMap<Variable, Term>> {
        if self.base_bindings.is_empty() {
            return None;
        }

        // Check if we've exhausted all combinations
        if self.current_position.iter().all(|&p| p == 0) && self.current_position.len() > 0 {
            // First iteration
        } else if self.current_position.is_empty() {
            return None;
        }

        // Build current combination
        let mut result = self.base_bindings[0].clone();
        for (i, var) in self.variables.iter().enumerate() {
            if let Some(values) = self.possible_values.get(var) {
                if self.current_position[i] < values.len() {
                    result.insert(var.clone(), values[self.current_position[i]].clone());
                }
            }
        }

        // Increment position
        self.increment_position();

        Some(result)
    }

    /// Increment position for next iteration
    fn increment_position(&mut self) {
        for i in (0..self.current_position.len()).rev() {
            if let Some(values) = self.possible_values.get(&self.variables[i]) {
                if self.current_position[i] + 1 < values.len() {
                    self.current_position[i] += 1;
                    return;
                } else {
                    self.current_position[i] = 0;
                }
            }
        }
        // If we get here, we've wrapped around completely
        self.current_position.clear();
    }

    /// Validate a binding against constraints
    fn validate_binding(&self, binding: &HashMap<Variable, Term>) -> bool {
        for constraint in &self.constraints {
            match constraint {
                Constraint::RelationshipConstraint {
                    left,
                    right,
                    relation,
                } => {
                    if let (Some(left_val), Some(right_val)) =
                        (binding.get(left), binding.get(right))
                    {
                        if !self.check_relation(left_val, right_val, *relation) {
                            return false;
                        }
                    }
                }
                _ => {} // Other constraints checked during binding
            }
        }
        true
    }

    /// Check relationship between values
    fn check_relation(&self, left: &Term, right: &Term, relation: RelationType) -> bool {
        match relation {
            RelationType::Equal => left == right,
            RelationType::NotEqual => left != right,
            _ => {
                // Numeric comparisons
                if let (Term::Literal(l_lit), Term::Literal(r_lit)) = (left, right) {
                    if let (Ok(l_val), Ok(r_val)) =
                        (l_lit.value().parse::<f64>(), r_lit.value().parse::<f64>())
                    {
                        match relation {
                            RelationType::LessThan => l_val < r_val,
                            RelationType::LessThanOrEqual => l_val <= r_val,
                            RelationType::GreaterThan => l_val > r_val,
                            RelationType::GreaterThanOrEqual => l_val >= r_val,
                            _ => false,
                        }
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
        }
    }
}

impl Default for BindingSet {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for BindingOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binding_set_basic() {
        let mut bindings = BindingSet::new();
        let var_x = Variable::new("x").unwrap();
        let var_y = Variable::new("y").unwrap();

        // Add variables
        bindings.add_variable(var_x.clone());
        bindings.add_variable(var_y.clone());

        // Bind x to a value
        let term = Term::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
        bindings
            .bind(var_x.clone(), term.clone(), BindingMetadata::default())
            .unwrap();

        // Check binding
        assert_eq!(bindings.get(&var_x), Some(&term));
        assert_eq!(bindings.get(&var_y), None);

        // Check unbound variables
        let unbound = bindings.unbound_variables();
        assert_eq!(unbound.len(), 1);
        assert_eq!(unbound[0], &var_y);
    }

    #[test]
    fn test_type_constraints() {
        let mut bindings = BindingSet::new();
        let var = Variable::new("x").unwrap();

        // Add type constraint
        bindings.add_constraint(Constraint::TypeConstraint {
            variable: var.clone(),
            allowed_types: vec![TermType::Literal, TermType::NumericLiteral]
                .into_iter()
                .collect(),
        });

        // Try to bind to named node (should fail)
        let named_node = Term::NamedNode(NamedNode::new("http://example.org/thing").unwrap());
        assert!(bindings
            .bind(var.clone(), named_node, BindingMetadata::default())
            .is_err());

        // Try to bind to literal (should succeed)
        let literal = Term::Literal(Literal::new("test"));
        assert!(bindings
            .bind(var.clone(), literal, BindingMetadata::default())
            .is_ok());
    }

    #[test]
    fn test_value_constraints() {
        let mut bindings = BindingSet::new();
        let var = Variable::new("age").unwrap();

        // Add numeric range constraint
        bindings.add_constraint(Constraint::ValueConstraint {
            variable: var.clone(),
            constraint: ValueConstraintType::NumericRange {
                min: 0.0,
                max: 150.0,
            },
        });

        // Valid age
        let valid_age = Term::Literal(Literal::new("25"));
        assert!(bindings
            .bind(var.clone(), valid_age, BindingMetadata::default())
            .is_ok());

        // Invalid age
        let invalid_age = Term::Literal(Literal::new("200"));
        assert!(bindings
            .bind(var.clone(), invalid_age, BindingMetadata::default())
            .is_err());
    }

    #[test]
    fn test_binding_merge() {
        let mut bindings1 = BindingSet::new();
        let mut bindings2 = BindingSet::new();

        let var_x = Variable::new("x").unwrap();
        let var_y = Variable::new("y").unwrap();

        let term_x = Term::NamedNode(NamedNode::new("http://example.org/x").unwrap());
        let term_y = Term::NamedNode(NamedNode::new("http://example.org/y").unwrap());

        bindings1
            .bind(var_x.clone(), term_x.clone(), BindingMetadata::default())
            .unwrap();
        bindings2
            .bind(var_y.clone(), term_y.clone(), BindingMetadata::default())
            .unwrap();

        // Merge
        bindings1.merge(&bindings2).unwrap();

        // Check both bindings exist
        assert_eq!(bindings1.get(&var_x), Some(&term_x));
        assert_eq!(bindings1.get(&var_y), Some(&term_y));
    }

    #[test]
    fn test_binding_optimizer() {
        let mut optimizer = BindingOptimizer::new();

        let vars = vec![Variable::new("x").unwrap(), Variable::new("y").unwrap()];
        let constraints = vec![];

        // First call should miss cache
        let _bindings1 = optimizer.optimize_bindings(vars.clone(), constraints.clone());

        // Second call should hit cache
        let _bindings2 = optimizer.optimize_bindings(vars, constraints);

        let stats = optimizer.stats();
        assert!(stats.contains("Cache hits: 1"));
    }
}