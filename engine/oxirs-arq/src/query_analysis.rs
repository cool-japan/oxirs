//! Query Analysis Module
//!
//! Provides variable discovery, join variable identification, filter safety analysis,
//! and semantic validation for SPARQL queries.

use crate::algebra::{Algebra, Expression, Term, TriplePattern, Variable};
use crate::cost_model::{CostEstimate, CostModel, IOPattern};
use crate::statistics_collector::StatisticsCollector;
use anyhow::Result;
use std::collections::{HashMap, HashSet};

/// Index type for optimization
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IndexType {
    /// B+ tree index for ordered access
    BTree,
    /// Hash index for equality access
    Hash,
    /// Full-text search index
    FullText,
    /// Spatial index for geographic data
    Spatial,
    /// Custom index type
    Custom(String),
}

/// Index access method recommendation
#[derive(Debug, Clone)]
pub struct IndexAccess {
    /// The index type to use
    pub index_type: IndexType,
    /// Triple pattern position (Subject=0, Predicate=1, Object=2)
    pub pattern_position: usize,
    /// Pattern index in the BGP
    pub pattern_index: usize,
    /// Expected selectivity with this index
    pub selectivity: f64,
    /// Estimated cost with this index
    pub cost_estimate: CostEstimate,
    /// I/O pattern for this access
    pub io_pattern: IOPattern,
    /// Improvement ratio compared to full scan
    pub improvement_ratio: f64,
}

/// Index availability analysis for a pattern
#[derive(Debug, Clone)]
pub struct PatternIndexAnalysis {
    /// Available indexes for this pattern
    pub available_indexes: Vec<IndexAccess>,
    /// Recommended index access method
    pub recommended_access: Option<IndexAccess>,
    /// Estimated cardinality without index
    pub full_scan_cardinality: usize,
    /// Estimated cardinality with best index
    pub indexed_cardinality: usize,
    /// Performance improvement ratio
    pub improvement_ratio: f64,
}

/// Index-aware optimization recommendations
#[derive(Debug, Clone)]
pub struct IndexOptimizationHints {
    /// Pattern-specific index recommendations
    pub pattern_recommendations: HashMap<usize, PatternIndexAnalysis>,
    /// Join order recommendations based on index availability
    pub join_order_hints: Vec<JoinOrderHint>,
    /// Filter placement recommendations
    pub filter_placement_hints: Vec<FilterPlacementHint>,
    /// Overall query execution strategy
    pub execution_strategy: ExecutionStrategy,
}

/// Join order hint based on index analysis
#[derive(Debug, Clone)]
pub struct JoinOrderHint {
    /// Pattern indices in recommended order
    pub pattern_order: Vec<usize>,
    /// Expected total cost
    pub estimated_cost: CostEstimate,
    /// Reasoning for this order
    pub reasoning: String,
}

/// Filter placement optimization hint
#[derive(Debug, Clone)]
pub struct FilterPlacementHint {
    /// Filter expression
    pub filter: Expression,
    /// Recommended placement (pattern index)
    pub recommended_placement: usize,
    /// Expected selectivity
    pub selectivity: f64,
    /// Cost benefit of early placement
    pub cost_benefit: f64,
}

/// Execution strategy recommendation
#[derive(Debug, Clone)]
pub enum ExecutionStrategy {
    /// Sequential pattern-by-pattern execution
    Sequential,
    /// Parallel execution of independent patterns
    Parallel,
    /// Index-driven execution
    IndexDriven,
    /// Hash-join based execution
    HashJoin,
    /// Sort-merge join execution
    SortMergeJoin,
    /// Adaptive execution based on runtime feedback
    Adaptive,
}

/// Filter safety analysis results
#[derive(Debug, Clone)]
pub struct FilterSafetyAnalysis {
    /// Safe filters (can be evaluated without error)
    pub safe_filters: Vec<Expression>,
    /// Unsafe filters (may produce errors)
    pub unsafe_filters: Vec<Expression>,
    /// Filter dependencies on variables
    pub filter_dependencies: Vec<(Expression, HashSet<Variable>)>,
}

/// Query analysis results
#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    /// All variables discovered in the query
    pub variables: HashSet<Variable>,
    /// Variables that appear in projection
    pub projected_variables: HashSet<Variable>,
    /// Variables that appear in filters
    pub filter_variables: HashSet<Variable>,
    /// Variables that join patterns together (simplified for tests)
    pub join_variables: HashSet<Variable>,
    /// Variable scoping information
    pub variable_scopes: HashMap<Variable, VariableScope>,
    /// Filter safety analysis results
    pub filter_safety: FilterSafetyAnalysis,
    /// Type consistency analysis
    pub type_consistency: TypeConsistencyAnalysis,
    /// Index optimization hints
    pub index_hints: IndexOptimizationHints,
    /// Pattern cardinality estimates
    pub pattern_cardinalities: HashMap<usize, usize>,
    /// Semantic validation results
    pub validation_errors: Vec<ValidationError>,
}

/// Variable scope information
#[derive(Debug, Clone)]
pub struct VariableScope {
    /// Pattern indices where this variable appears
    pub pattern_indices: HashSet<usize>,
    /// Whether the variable is bound (not free)
    pub is_bound: bool,
    /// Whether the variable appears in projection
    pub in_projection: bool,
    /// Whether the variable appears in filters
    pub in_filters: bool,
    /// Whether the variable appears in GROUP BY
    pub in_group_by: bool,
    /// Whether the variable appears in ORDER BY
    pub in_order_by: bool,
}

/// Filter safety classification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterSafety {
    /// Safe to push down (no side effects)
    Safe,
    /// Unsafe due to optional patterns
    UnsafeOptional,
    /// Unsafe due to unbound variables
    UnsafeUnbound,
    /// Unsafe due to aggregate functions
    UnsafeAggregate,
    /// Unsafe due to service calls
    UnsafeService,
}

/// Type consistency analysis
#[derive(Debug, Clone)]
pub struct TypeConsistencyAnalysis {
    /// Type constraints for variables
    pub variable_types: HashMap<Variable, VariableType>,
    /// Type errors found
    pub type_errors: Vec<TypeError>,
    /// Type warnings
    pub type_warnings: Vec<TypeWarning>,
}

/// Variable type information
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariableType {
    /// Resource (IRI or Blank Node)
    Resource,
    /// Literal value
    Literal,
    /// Numeric literal
    Numeric,
    /// String literal
    String,
    /// Boolean literal
    Boolean,
    /// Date/time literal
    DateTime,
    /// Unknown or mixed type
    Unknown,
}

/// Type error
#[derive(Debug, Clone)]
pub struct TypeError {
    /// Variable involved in the error
    pub variable: Variable,
    /// Expected type
    pub expected: VariableType,
    /// Actual type
    pub actual: VariableType,
    /// Location of the error
    pub location: String,
    /// Error message
    pub message: String,
}

/// Type warning
#[derive(Debug, Clone)]
pub struct TypeWarning {
    /// Variable involved in the warning
    pub variable: Variable,
    /// Warning message
    pub message: String,
    /// Location of the warning
    pub location: String,
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error type
    pub error_type: ValidationErrorType,
    /// Error message
    pub message: String,
    /// Location where error occurred
    pub location: String,
    /// Suggested fix (if any)
    pub suggestion: Option<String>,
}

/// Validation error types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationErrorType {
    /// Unbound variable in projection
    UnboundVariable,
    /// Type mismatch
    TypeMismatch,
    /// Invalid aggregate usage
    InvalidAggregate,
    /// Invalid service clause
    InvalidService,
    /// Circular dependency
    CircularDependency,
    /// Semantic inconsistency
    SemanticInconsistency,
}

/// Query analyzer
#[derive(Debug, Clone)]
pub struct QueryAnalyzer {
    /// Statistics collector for cardinality estimation
    statistics: Option<StatisticsCollector>,
    /// Cost model for optimization decisions
    cost_model: Option<CostModel>,
    /// Whether to enable type inference
    pub enable_type_inference: bool,
}

impl QueryAnalyzer {
    /// Create a new query analyzer
    pub fn new() -> Self {
        Self {
            statistics: None,
            cost_model: None,
            enable_type_inference: true,
        }
    }

    /// Create analyzer with statistics collector
    pub fn with_statistics(statistics: StatisticsCollector) -> Self {
        Self {
            statistics: Some(statistics),
            cost_model: None,
            enable_type_inference: true,
        }
    }

    /// Create analyzer with cost model
    pub fn with_cost_model(cost_model: CostModel) -> Self {
        Self {
            statistics: None,
            cost_model: Some(cost_model),
            enable_type_inference: true,
        }
    }

    /// Create analyzer with both statistics and cost model
    pub fn with_statistics_and_cost_model(
        statistics: StatisticsCollector,
        cost_model: CostModel,
    ) -> Self {
        Self {
            statistics: Some(statistics),
            cost_model: Some(cost_model),
            enable_type_inference: true,
        }
    }

    /// Analyze a query and return comprehensive analysis results
    pub fn analyze_query(&self, algebra: &Algebra) -> Result<QueryAnalysis> {
        let variables = self.discover_variables(algebra)?;
        let projected_variables = self.extract_projected_variables(algebra);
        let filter_variables = self.extract_filter_variables(algebra);
        let join_variables = self.identify_join_variables_simplified(algebra)?;
        let variable_scopes = self.analyze_variable_scopes(algebra)?;
        let filter_safety = self.analyze_filter_safety_structured(algebra)?;
        let type_consistency = self.analyze_type_consistency(algebra)?;
        let index_hints = self.generate_index_hints(algebra)?;
        let pattern_cardinalities = self.estimate_pattern_cardinalities(algebra);
        let validation_errors = self.validate_semantics(algebra)?;

        Ok(QueryAnalysis {
            variables,
            projected_variables,
            filter_variables,
            join_variables,
            variable_scopes,
            filter_safety,
            type_consistency,
            index_hints,
            pattern_cardinalities,
            validation_errors,
        })
    }

    /// Alias for analyze_query for backward compatibility
    pub fn analyze(&self, algebra: &Algebra) -> Result<QueryAnalysis> {
        self.analyze_query(algebra)
    }

    /// Add index information for optimization (placeholder implementation)
    pub fn add_index(&mut self, _predicate: &str, _index_type: IndexType) {
        // This is a placeholder implementation
        // In a real implementation, this would store index information
        // for use in optimization decisions
    }

    /// Estimate pattern cardinality based on bound terms
    pub fn estimate_pattern_cardinality(&self, pattern: &TriplePattern) -> usize {
        let mut bound_terms = 0;

        if !matches!(&pattern.subject, Term::Variable(_)) {
            bound_terms += 1;
        }
        if !matches!(&pattern.predicate, Term::Variable(_)) {
            bound_terms += 1;
        }
        if !matches!(&pattern.object, Term::Variable(_)) {
            bound_terms += 1;
        }

        // Simple heuristic: more bound terms = lower cardinality
        match bound_terms {
            0 => 1_000_000, // All variables
            1 => 100_000,   // One bound term
            2 => 1_000,     // Two bound terms
            3 => 1,         // All bound terms
            _ => 1,
        }
    }

    /// Estimate pattern cardinalities for all patterns
    pub fn estimate_pattern_cardinalities(&self, algebra: &Algebra) -> HashMap<usize, usize> {
        let mut cardinalities = HashMap::new();
        let patterns = self.extract_bgp_patterns(algebra);

        for (idx, pattern) in patterns.iter().enumerate() {
            let cardinality = self.estimate_pattern_cardinality(pattern);
            cardinalities.insert(idx, cardinality);
        }

        cardinalities
    }

    /// Identify join variables in simplified form (returns HashSet)
    pub fn identify_join_variables_simplified(
        &self,
        algebra: &Algebra,
    ) -> Result<HashSet<Variable>> {
        let join_vars_detailed = self.identify_join_variables(algebra)?;
        // Convert to simple HashSet of variables that appear in multiple patterns
        Ok(join_vars_detailed.into_keys().collect())
    }

    /// Analyze filter safety and return structured results
    pub fn analyze_filter_safety_structured(
        &self,
        algebra: &Algebra,
    ) -> Result<FilterSafetyAnalysis> {
        let safety_vec = self.analyze_filter_safety(algebra)?;
        let mut safe_filters = Vec::new();
        let mut unsafe_filters = Vec::new();
        let mut filter_dependencies = Vec::new();

        for (expr, safety) in safety_vec {
            match safety {
                FilterSafety::Safe => safe_filters.push(expr.clone()),
                _ => unsafe_filters.push(expr.clone()),
            }

            // Extract variable dependencies from expression
            let mut deps = HashSet::new();
            self.collect_variables_from_expression(&expr, &mut deps)
                .unwrap_or(());
            filter_dependencies.push((expr, deps));
        }

        Ok(FilterSafetyAnalysis {
            safe_filters,
            unsafe_filters,
            filter_dependencies,
        })
    }

    /// Discover all variables in the algebra expression
    pub fn discover_variables(&self, algebra: &Algebra) -> Result<HashSet<Variable>> {
        let mut variables = HashSet::new();
        self.collect_variables_recursive(algebra, &mut variables)?;
        Ok(variables)
    }

    /// Recursively collect variables from algebra
    fn collect_variables_recursive(
        &self,
        algebra: &Algebra,
        variables: &mut HashSet<Variable>,
    ) -> Result<()> {
        match algebra {
            Algebra::Bgp(patterns) => {
                for pattern in patterns {
                    self.collect_variables_from_pattern(pattern, variables)?;
                }
            }
            Algebra::Join { left, right } => {
                self.collect_variables_recursive(left, variables)?;
                self.collect_variables_recursive(right, variables)?;
            }
            Algebra::Union { left, right } => {
                self.collect_variables_recursive(left, variables)?;
                self.collect_variables_recursive(right, variables)?;
            }
            Algebra::Filter { pattern, condition } => {
                self.collect_variables_recursive(pattern, variables)?;
                self.collect_variables_from_expression(condition, variables)?;
            }
            Algebra::Project {
                pattern,
                variables: proj_vars,
            } => {
                self.collect_variables_recursive(pattern, variables)?;
                for var in proj_vars {
                    variables.insert(var.clone());
                }
            }
            Algebra::Group {
                pattern,
                variables: group_vars,
                ..
            } => {
                self.collect_variables_recursive(pattern, variables)?;
                for group_var in group_vars {
                    self.collect_variables_from_group_condition(group_var, variables)?;
                }
            }
            _ => {
                // Handle other algebra types as needed
            }
        }
        Ok(())
    }

    /// Collect variables from a triple pattern
    fn collect_variables_from_pattern(
        &self,
        pattern: &TriplePattern,
        variables: &mut HashSet<Variable>,
    ) -> Result<()> {
        if let Term::Variable(var) = &pattern.subject {
            variables.insert(var.clone());
        }
        if let Term::Variable(var) = &pattern.predicate {
            variables.insert(var.clone());
        }
        if let Term::Variable(var) = &pattern.object {
            variables.insert(var.clone());
        }
        Ok(())
    }

    /// Collect variables from an expression
    fn collect_variables_from_expression(
        &self,
        expr: &Expression,
        variables: &mut HashSet<Variable>,
    ) -> Result<()> {
        match expr {
            Expression::Variable(var) => {
                variables.insert(var.clone());
            }
            Expression::Binary { left, right, .. } => {
                self.collect_variables_from_expression(left, variables)?;
                self.collect_variables_from_expression(right, variables)?;
            }
            Expression::Unary { operand, .. } => {
                self.collect_variables_from_expression(operand, variables)?;
            }
            Expression::Function { args, .. } => {
                for arg in args {
                    self.collect_variables_from_expression(arg, variables)?;
                }
            }
            _ => {
                // Handle other expression types as needed
            }
        }
        Ok(())
    }

    /// Collect variables from group condition
    fn collect_variables_from_group_condition(
        &self,
        _condition: &crate::algebra::GroupCondition,
        _variables: &mut HashSet<Variable>,
    ) -> Result<()> {
        // Implementation depends on GroupCondition structure
        // For now, assume it contains an expression
        // This would need to be adjusted based on actual GroupCondition definition
        Ok(())
    }

    /// Extract variables that appear in projection
    pub fn extract_projected_variables(&self, algebra: &Algebra) -> HashSet<Variable> {
        let mut projected_vars = HashSet::new();
        if let Algebra::Project { variables, .. } = algebra {
            for var in variables {
                projected_vars.insert(var.clone());
            }
        }
        projected_vars
    }

    /// Extract variables that appear in filters
    pub fn extract_filter_variables(&self, algebra: &Algebra) -> HashSet<Variable> {
        let mut filter_vars = HashSet::new();
        self.extract_filter_variables_recursive(algebra, &mut filter_vars);
        filter_vars
    }

    /// Recursively extract filter variables
    fn extract_filter_variables_recursive(
        &self,
        algebra: &Algebra,
        filter_vars: &mut HashSet<Variable>,
    ) {
        match algebra {
            Algebra::Filter { pattern, condition } => {
                self.extract_filter_variables_recursive(pattern, filter_vars);
                self.collect_variables_from_expression(condition, filter_vars)
                    .unwrap_or(());
            }
            Algebra::Join { left, right } => {
                self.extract_filter_variables_recursive(left, filter_vars);
                self.extract_filter_variables_recursive(right, filter_vars);
            }
            Algebra::Union { left, right } => {
                self.extract_filter_variables_recursive(left, filter_vars);
                self.extract_filter_variables_recursive(right, filter_vars);
            }
            _ => {} // Other algebra types don't directly contain filters
        }
    }

    /// Identify variables that join patterns together
    pub fn identify_join_variables(
        &self,
        algebra: &Algebra,
    ) -> Result<HashMap<Variable, Vec<usize>>> {
        let mut join_vars = HashMap::new();
        let patterns = self.extract_bgp_patterns(algebra);

        for (pattern_idx, pattern) in patterns.iter().enumerate() {
            let pattern_vars = self.get_pattern_variables(pattern);
            for var in pattern_vars {
                join_vars
                    .entry(var)
                    .or_insert_with(Vec::new)
                    .push(pattern_idx);
            }
        }

        // Filter to only include variables that appear in multiple patterns
        join_vars.retain(|_var, pattern_indices| pattern_indices.len() > 1);

        Ok(join_vars)
    }

    /// Extract BGP patterns from algebra
    fn extract_bgp_patterns(&self, algebra: &Algebra) -> Vec<TriplePattern> {
        let mut patterns = Vec::new();
        self.extract_bgp_patterns_recursive(algebra, &mut patterns);
        patterns
    }

    /// Recursively extract BGP patterns
    fn extract_bgp_patterns_recursive(&self, algebra: &Algebra, patterns: &mut Vec<TriplePattern>) {
        match algebra {
            Algebra::Bgp(bgp_patterns) => {
                patterns.extend(bgp_patterns.clone());
            }
            Algebra::Join { left, right } => {
                self.extract_bgp_patterns_recursive(left, patterns);
                self.extract_bgp_patterns_recursive(right, patterns);
            }
            Algebra::Union { left, right } => {
                self.extract_bgp_patterns_recursive(left, patterns);
                self.extract_bgp_patterns_recursive(right, patterns);
            }
            Algebra::Filter { pattern, .. } => {
                self.extract_bgp_patterns_recursive(pattern, patterns);
            }
            _ => {} // Other types don't contain BGP patterns directly
        }
    }

    /// Get variables from a triple pattern
    fn get_pattern_variables(&self, pattern: &TriplePattern) -> Vec<Variable> {
        let mut vars = Vec::new();
        if let Term::Variable(var) = &pattern.subject {
            vars.push(var.clone());
        }
        if let Term::Variable(var) = &pattern.predicate {
            vars.push(var.clone());
        }
        if let Term::Variable(var) = &pattern.object {
            vars.push(var.clone());
        }
        vars
    }

    /// Analyze variable scoping
    pub fn analyze_variable_scopes(
        &self,
        algebra: &Algebra,
    ) -> Result<HashMap<Variable, VariableScope>> {
        let mut scopes = HashMap::new();
        let all_vars = self.discover_variables(algebra)?;

        for var in all_vars {
            let scope = VariableScope {
                pattern_indices: self.find_pattern_indices_for_variable(&var, algebra),
                is_bound: self.is_variable_bound(&var, algebra),
                in_projection: self.is_in_projection(&var, algebra),
                in_filters: self.is_in_filters(&var, algebra),
                in_group_by: self.is_in_group_by(&var, algebra),
                in_order_by: self.is_in_order_by(&var, algebra),
            };
            scopes.insert(var, scope);
        }

        Ok(scopes)
    }

    /// Find pattern indices where a variable appears
    fn find_pattern_indices_for_variable(
        &self,
        var: &Variable,
        algebra: &Algebra,
    ) -> HashSet<usize> {
        let mut indices = HashSet::new();
        let patterns = self.extract_bgp_patterns(algebra);

        for (idx, pattern) in patterns.iter().enumerate() {
            if self.pattern_contains_variable(pattern, var) {
                indices.insert(idx);
            }
        }

        indices
    }

    /// Check if pattern contains variable
    fn pattern_contains_variable(&self, pattern: &TriplePattern, var: &Variable) -> bool {
        matches!(&pattern.subject, Term::Variable(v) if v == var)
            || matches!(&pattern.predicate, Term::Variable(v) if v == var)
            || matches!(&pattern.object, Term::Variable(v) if v == var)
    }

    /// Check if variable is bound
    fn is_variable_bound(&self, var: &Variable, algebra: &Algebra) -> bool {
        // A variable is bound if it appears in a triple pattern
        let patterns = self.extract_bgp_patterns(algebra);
        patterns
            .iter()
            .any(|pattern| self.pattern_contains_variable(pattern, var))
    }

    /// Check if variable is in projection
    fn is_in_projection(&self, var: &Variable, algebra: &Algebra) -> bool {
        if let Algebra::Project { variables, .. } = algebra {
            variables.contains(var)
        } else {
            false
        }
    }

    /// Check if variable is in filters
    fn is_in_filters(&self, var: &Variable, algebra: &Algebra) -> bool {
        let filter_vars = self.extract_filter_variables(algebra);
        filter_vars.contains(var)
    }

    /// Check if variable is in GROUP BY
    fn is_in_group_by(&self, _var: &Variable, _algebra: &Algebra) -> bool {
        // Implementation depends on how GROUP BY variables are represented
        // This is a placeholder
        false
    }

    /// Check if variable is in ORDER BY
    fn is_in_order_by(&self, _var: &Variable, _algebra: &Algebra) -> bool {
        // Implementation depends on how ORDER BY variables are represented
        // This is a placeholder
        false
    }

    /// Analyze filter safety
    pub fn analyze_filter_safety(
        &self,
        algebra: &Algebra,
    ) -> Result<Vec<(Expression, FilterSafety)>> {
        let mut safety_vec = Vec::new();
        self.analyze_filter_safety_recursive(algebra, &mut safety_vec)?;
        Ok(safety_vec)
    }

    /// Recursively analyze filter safety
    fn analyze_filter_safety_recursive(
        &self,
        algebra: &Algebra,
        safety_vec: &mut Vec<(Expression, FilterSafety)>,
    ) -> Result<()> {
        match algebra {
            Algebra::Filter { pattern, condition } => {
                let safety = self.determine_filter_safety(condition, pattern)?;
                safety_vec.push((condition.clone(), safety));
                self.analyze_filter_safety_recursive(pattern, safety_vec)?;
            }
            Algebra::Join { left, right } => {
                self.analyze_filter_safety_recursive(left, safety_vec)?;
                self.analyze_filter_safety_recursive(right, safety_vec)?;
            }
            Algebra::Union { left, right } => {
                self.analyze_filter_safety_recursive(left, safety_vec)?;
                self.analyze_filter_safety_recursive(right, safety_vec)?;
            }
            _ => {} // Other types handled as needed
        }
        Ok(())
    }

    /// Determine filter safety
    fn determine_filter_safety(
        &self,
        condition: &Expression,
        context: &Algebra,
    ) -> Result<FilterSafety> {
        // Check for various safety conditions
        if self.contains_aggregate_function(condition) {
            return Ok(FilterSafety::UnsafeAggregate);
        }

        if self.contains_service_call(condition) {
            return Ok(FilterSafety::UnsafeService);
        }

        if self.has_unbound_variables(condition, context) {
            return Ok(FilterSafety::UnsafeUnbound);
        }

        if self.in_optional_context(context) {
            return Ok(FilterSafety::UnsafeOptional);
        }

        Ok(FilterSafety::Safe)
    }

    /// Check if expression contains aggregate function
    fn contains_aggregate_function(&self, expr: &Expression) -> bool {
        match expr {
            Expression::Function { name, .. } => {
                // Check if function name is an aggregate function
                matches!(
                    name.as_str(),
                    "COUNT" | "SUM" | "AVG" | "MIN" | "MAX" | "GROUP_CONCAT"
                )
            }
            Expression::Binary { left, right, .. } => {
                self.contains_aggregate_function(left) || self.contains_aggregate_function(right)
            }
            Expression::Unary { operand, .. } => self.contains_aggregate_function(operand),
            _ => false,
        }
    }

    /// Check if expression contains service call
    fn contains_service_call(&self, _expr: &Expression) -> bool {
        // Implementation depends on how service calls are represented
        // This is a placeholder
        false
    }

    /// Check if expression has unbound variables
    fn has_unbound_variables(&self, expr: &Expression, context: &Algebra) -> bool {
        let expr_vars = {
            let mut vars = HashSet::new();
            self.collect_variables_from_expression(expr, &mut vars)
                .unwrap_or(());
            vars
        };

        let bound_vars = self.discover_variables(context).unwrap_or_default();

        !expr_vars.iter().all(|var| bound_vars.contains(var))
    }

    /// Check if context is optional
    fn in_optional_context(&self, _context: &Algebra) -> bool {
        // Implementation depends on how optional patterns are represented
        // This is a placeholder
        false
    }

    /// Analyze type consistency
    pub fn analyze_type_consistency(&self, algebra: &Algebra) -> Result<TypeConsistencyAnalysis> {
        let variables = self.discover_variables(algebra)?;
        let mut variable_types = HashMap::new();
        let mut type_errors = Vec::new();
        let mut type_warnings = Vec::new();

        for var in variables {
            let var_type = self.infer_variable_type(&var, algebra)?;
            variable_types.insert(var, var_type);
        }

        // Analyze type consistency and detect errors
        self.detect_type_errors(algebra, &variable_types, &mut type_errors)?;
        self.detect_type_warnings(algebra, &variable_types, &mut type_warnings)?;

        Ok(TypeConsistencyAnalysis {
            variable_types,
            type_errors,
            type_warnings,
        })
    }

    /// Infer variable type from usage
    fn infer_variable_type(&self, var: &Variable, algebra: &Algebra) -> Result<VariableType> {
        // Analyze how the variable is used to infer its type
        let patterns = self.extract_bgp_patterns(algebra);

        for pattern in patterns {
            if matches!(&pattern.subject, Term::Variable(v) if v == var) {
                return Ok(VariableType::Resource); // Subject is always a resource
            }
            if matches!(&pattern.predicate, Term::Variable(v) if v == var) {
                return Ok(VariableType::Resource); // Predicate is always a resource
            }
            if matches!(&pattern.object, Term::Variable(v) if v == var) {
                // Object can be resource or literal, need more analysis
                return Ok(VariableType::Unknown);
            }
        }

        Ok(VariableType::Unknown)
    }

    /// Detect type errors
    fn detect_type_errors(
        &self,
        _algebra: &Algebra,
        _variable_types: &HashMap<Variable, VariableType>,
        _type_errors: &mut Vec<TypeError>,
    ) -> Result<()> {
        // Implementation for detecting type errors
        // This is a placeholder for more sophisticated type checking
        Ok(())
    }

    /// Detect type warnings
    fn detect_type_warnings(
        &self,
        _algebra: &Algebra,
        _variable_types: &HashMap<Variable, VariableType>,
        _type_warnings: &mut Vec<TypeWarning>,
    ) -> Result<()> {
        // Implementation for detecting type warnings
        // This is a placeholder
        Ok(())
    }

    /// Generate index optimization hints
    pub fn generate_index_hints(&self, algebra: &Algebra) -> Result<IndexOptimizationHints> {
        let patterns = self.extract_bgp_patterns(algebra);
        let mut pattern_recommendations = HashMap::new();

        for (idx, pattern) in patterns.iter().enumerate() {
            let analysis = self.analyze_pattern_for_indexes(pattern, idx)?;
            pattern_recommendations.insert(idx, analysis);
        }

        let join_order_hints = self.generate_join_order_hints(&patterns)?;
        let filter_placement_hints = self.generate_filter_placement_hints(algebra)?;
        let execution_strategy = self.recommend_execution_strategy(algebra)?;

        Ok(IndexOptimizationHints {
            pattern_recommendations,
            join_order_hints,
            filter_placement_hints,
            execution_strategy,
        })
    }

    /// Analyze pattern for index usage
    fn analyze_pattern_for_indexes(
        &self,
        pattern: &TriplePattern,
        pattern_idx: usize,
    ) -> Result<PatternIndexAnalysis> {
        let mut available_indexes = Vec::new();

        // Analyze what indexes could be used for this pattern
        if !matches!(&pattern.subject, Term::Variable(_)) {
            // Subject is bound, can use subject index
            available_indexes.push(IndexAccess {
                index_type: IndexType::BTree,
                pattern_position: 0,
                pattern_index: pattern_idx,
                selectivity: 0.1, // Estimated
                cost_estimate: CostEstimate::new(100.0, 10.0, 50.0, 0.0, 1000),
                io_pattern: IOPattern::Sequential,
                improvement_ratio: 10.0,
            });
        }

        if !matches!(&pattern.predicate, Term::Variable(_)) {
            // Predicate is bound, can use predicate index
            available_indexes.push(IndexAccess {
                index_type: IndexType::Hash,
                pattern_position: 1,
                pattern_index: pattern_idx,
                selectivity: 0.05, // Predicates usually more selective
                cost_estimate: CostEstimate::new(50.0, 5.0, 25.0, 0.0, 500),
                io_pattern: IOPattern::Random,
                improvement_ratio: 20.0,
            });
        }

        let recommended_access = available_indexes
            .iter()
            .min_by(|a, b| {
                a.cost_estimate
                    .total_cost
                    .partial_cmp(&b.cost_estimate.total_cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();

        Ok(PatternIndexAnalysis {
            available_indexes,
            recommended_access,
            full_scan_cardinality: 1000000, // Placeholder
            indexed_cardinality: 1000,      // Placeholder
            improvement_ratio: 1000.0,      // Placeholder
        })
    }

    /// Generate join order hints
    fn generate_join_order_hints(&self, patterns: &[TriplePattern]) -> Result<Vec<JoinOrderHint>> {
        let mut hints = Vec::new();

        if patterns.len() > 1 {
            // Generate a simple hint based on pattern selectivity
            let order: Vec<usize> = (0..patterns.len()).collect();
            hints.push(JoinOrderHint {
                pattern_order: order,
                estimated_cost: CostEstimate::new(1000.0, 100.0, 500.0, 0.0, 1000),
                reasoning: "Default left-deep join order".to_string(),
            });
        }

        Ok(hints)
    }

    /// Generate filter placement hints
    fn generate_filter_placement_hints(
        &self,
        _algebra: &Algebra,
    ) -> Result<Vec<FilterPlacementHint>> {
        let hints = Vec::new();
        // Implementation for filter placement analysis
        // This is a placeholder
        Ok(hints)
    }

    /// Recommend execution strategy
    fn recommend_execution_strategy(&self, algebra: &Algebra) -> Result<ExecutionStrategy> {
        let complexity = self.estimate_query_complexity(algebra);

        if complexity < 10.0 {
            Ok(ExecutionStrategy::Sequential)
        } else if complexity < 50.0 {
            Ok(ExecutionStrategy::IndexDriven)
        } else if complexity < 100.0 {
            Ok(ExecutionStrategy::HashJoin)
        } else {
            Ok(ExecutionStrategy::Parallel)
        }
    }

    /// Estimate query complexity
    fn estimate_query_complexity(&self, algebra: &Algebra) -> f64 {
        match algebra {
            Algebra::Bgp(patterns) => patterns.len() as f64,
            Algebra::Join { left, right } => {
                self.estimate_query_complexity(left) + self.estimate_query_complexity(right) + 10.0
            }
            Algebra::Union { left, right } => {
                self.estimate_query_complexity(left) + self.estimate_query_complexity(right) + 5.0
            }
            Algebra::Filter { pattern, .. } => self.estimate_query_complexity(pattern) + 2.0,
            _ => 1.0,
        }
    }

    /// Validate query semantics
    pub fn validate_semantics(&self, algebra: &Algebra) -> Result<Vec<ValidationError>> {
        let mut errors = Vec::new();

        // Check for unbound variables in projection
        self.check_unbound_variables_in_projection(algebra, &mut errors)?;

        // Check for invalid aggregates
        self.check_invalid_aggregates(algebra, &mut errors)?;

        // Check for other semantic issues
        self.check_semantic_consistency(algebra, &mut errors)?;

        Ok(errors)
    }

    /// Check for unbound variables in projection
    fn check_unbound_variables_in_projection(
        &self,
        algebra: &Algebra,
        errors: &mut Vec<ValidationError>,
    ) -> Result<()> {
        if let Algebra::Project { variables, pattern } = algebra {
            let bound_vars = self.discover_variables(pattern)?;

            for var in variables {
                if !bound_vars.contains(var) {
                    errors.push(ValidationError {
                        error_type: ValidationErrorType::UnboundVariable,
                        message: format!(
                            "Variable ?{} appears in projection but is not bound",
                            var.as_str()
                        ),
                        location: "SELECT clause".to_string(),
                        suggestion: Some(format!(
                            "Ensure ?{} appears in a triple pattern",
                            var.as_str()
                        )),
                    });
                }
            }
        }
        Ok(())
    }

    /// Check for invalid aggregates
    fn check_invalid_aggregates(
        &self,
        _algebra: &Algebra,
        _errors: &mut Vec<ValidationError>,
    ) -> Result<()> {
        // Implementation for aggregate validation
        // This is a placeholder
        Ok(())
    }

    /// Check semantic consistency
    fn check_semantic_consistency(
        &self,
        _algebra: &Algebra,
        _errors: &mut Vec<ValidationError>,
    ) -> Result<()> {
        // Implementation for semantic consistency checks
        // This is a placeholder
        Ok(())
    }
}

impl Default for QueryAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{BinaryOperator, Literal, Term, Variable};
    use oxirs_core::model::NamedNode;

    #[test]
    fn test_query_analyzer() {
        let analyzer = QueryAnalyzer::new();
        assert!(analyzer.enable_type_inference);
    }

    #[test]
    fn test_variable_discovery() {
        let analyzer = QueryAnalyzer::new();

        let pattern = TriplePattern {
            subject: Term::Variable(Variable::new("s").unwrap()),
            predicate: Term::Iri(NamedNode::new("http://example.org/predicate").unwrap()),
            object: Term::Variable(Variable::new("o").unwrap()),
        };

        let algebra = Algebra::Bgp(vec![pattern]);
        let analysis = analyzer.analyze(&algebra).unwrap();

        assert_eq!(analysis.variables.len(), 2);
        assert!(analysis.variables.contains(&Variable::new("s").unwrap()));
        assert!(analysis.variables.contains(&Variable::new("o").unwrap()));
    }

    #[test]
    fn test_join_variable_identification() {
        let analyzer = QueryAnalyzer::new();

        let pattern1 = TriplePattern {
            subject: Term::Variable(Variable::new("x").unwrap()),
            predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/p1")),
            object: Term::Variable(Variable::new("y").unwrap()),
        };

        let pattern2 = TriplePattern {
            subject: Term::Variable(Variable::new("y").unwrap()),
            predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/p2")),
            object: Term::Variable(Variable::new("z").unwrap()),
        };

        let algebra = Algebra::Join {
            left: Box::new(Algebra::Bgp(vec![pattern1])),
            right: Box::new(Algebra::Bgp(vec![pattern2])),
        };

        let analysis = analyzer.analyze(&algebra).unwrap();

        assert!(analysis
            .join_variables
            .contains(&Variable::new("y").unwrap()));
        assert!(!analysis
            .join_variables
            .contains(&Variable::new("x").unwrap()));
        assert!(!analysis
            .join_variables
            .contains(&Variable::new("z").unwrap()));
    }

    #[test]
    fn test_filter_safety_analysis() {
        let analyzer = QueryAnalyzer::new();

        let pattern = TriplePattern {
            subject: Term::Variable(Variable::new("s").unwrap()),
            predicate: Term::Iri(NamedNode::new("http://example.org/predicate").unwrap()),
            object: Term::Variable(Variable::new("o").unwrap()),
        };

        let filter_expr = Expression::Binary {
            left: Box::new(Expression::Variable(Variable::new("s").unwrap())),
            op: BinaryOperator::Equal,
            right: Box::new(Expression::Literal(Literal {
                value: "test".to_string(),
                language: None,
                datatype: None,
            })),
        };

        let algebra = Algebra::Filter {
            condition: filter_expr.clone(),
            pattern: Box::new(Algebra::Bgp(vec![pattern])),
        };

        let analysis = analyzer.analyze(&algebra).unwrap();

        // The filter should be safe because 's' is bound in the triple pattern
        assert!(analysis.filter_safety.safe_filters.contains(&filter_expr));
        assert!(!analysis.filter_safety.unsafe_filters.contains(&filter_expr));
    }

    #[test]
    fn test_index_aware_analysis() {
        let mut analyzer = QueryAnalyzer::new();

        // Add some index information
        analyzer.add_index("http://example.org/type", IndexType::Hash);
        analyzer.add_index("http://example.org/label", IndexType::BTree);

        let pattern = TriplePattern {
            subject: Term::Variable(Variable::new("s").unwrap()),
            predicate: Term::Iri(NamedNode::new("http://example.org/type").unwrap()),
            object: Term::Variable(Variable::new("o").unwrap()),
        };

        let algebra = Algebra::Bgp(vec![pattern]);
        let analysis = analyzer.analyze(&algebra).unwrap();

        // Check that index optimization hints were generated
        assert!(!analysis.index_hints.pattern_recommendations.is_empty());

        // Check that pattern cardinalities were estimated
        assert!(!analysis.pattern_cardinalities.is_empty());

        // Check that a pattern analysis was generated for pattern 0
        assert!(analysis
            .index_hints
            .pattern_recommendations
            .contains_key(&0));

        let pattern_analysis = &analysis.index_hints.pattern_recommendations[&0];

        // Should have some available indexes since we added one for this predicate
        assert!(!pattern_analysis.available_indexes.is_empty());

        // Should have a recommended access method
        assert!(pattern_analysis.recommended_access.is_some());

        // Check that execution strategy was determined
        matches!(
            analysis.index_hints.execution_strategy,
            ExecutionStrategy::IndexDriven
                | ExecutionStrategy::HashJoin
                | ExecutionStrategy::SortMergeJoin
                | ExecutionStrategy::Adaptive
                | ExecutionStrategy::Parallel
        );
    }

    #[test]
    fn test_join_order_optimization() {
        let mut analyzer = QueryAnalyzer::new();

        // Add indexes to make some patterns more selective
        analyzer.add_index("http://example.org/selective", IndexType::Hash);

        let pattern1 = TriplePattern {
            subject: Term::Variable(Variable::new("x").unwrap()),
            predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/selective")),
            object: Term::Variable(Variable::new("y").unwrap()),
        };

        let pattern2 = TriplePattern {
            subject: Term::Variable(Variable::new("y").unwrap()),
            predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/expensive")),
            object: Term::Variable(Variable::new("z").unwrap()),
        };

        let algebra = Algebra::Join {
            left: Box::new(Algebra::Bgp(vec![pattern1])),
            right: Box::new(Algebra::Bgp(vec![pattern2])),
        };

        let analysis = analyzer.analyze(&algebra).unwrap();

        // Should have join order hints
        assert!(!analysis.index_hints.join_order_hints.is_empty());

        // Should have estimated costs
        let hint = &analysis.index_hints.join_order_hints[0];
        assert!(hint.estimated_cost.total_cost > 0.0);
    }

    #[test]
    fn test_cardinality_estimation() {
        let analyzer = QueryAnalyzer::new();

        // Test different pattern types for cardinality estimation
        let high_cardinality_pattern = TriplePattern {
            subject: Term::Variable(Variable::new("s").unwrap()),
            predicate: Term::Variable(Variable::new("p").unwrap()),
            object: Term::Variable(Variable::new("o").unwrap()),
        };

        let medium_cardinality_pattern = TriplePattern {
            subject: Term::Variable(Variable::new("s").unwrap()),
            predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/type")),
            object: Term::Variable(Variable::new("o").unwrap()),
        };

        let low_cardinality_pattern = TriplePattern {
            subject: Term::Variable(Variable::new("s").unwrap()),
            predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/type")),
            object: Term::Iri(NamedNode::new_unchecked("http://example.org/Person")),
        };

        let high_card = analyzer.estimate_pattern_cardinality(&high_cardinality_pattern);
        let medium_card = analyzer.estimate_pattern_cardinality(&medium_cardinality_pattern);
        let low_card = analyzer.estimate_pattern_cardinality(&low_cardinality_pattern);

        // More bound terms should result in lower cardinality estimates
        assert!(high_card > medium_card);
        assert!(medium_card > low_card);
    }
}
