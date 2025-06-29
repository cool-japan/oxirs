//! Query Analysis Module
//!
//! Provides variable discovery, join variable identification, filter safety analysis,
//! and semantic validation for SPARQL queries.

use crate::algebra::{Algebra, Expression, Literal, Term, TriplePattern, Variable};
use crate::cost_model::{CostEstimate, CostModel, IOPattern};
use crate::statistics_collector::{Histogram, StatisticsCollector};
use anyhow::{anyhow, Result};
use oxirs_core::model::NamedNode;
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
    /// Use indexes with nested loop joins
    IndexNestedLoop,
    /// Use hash joins with table scans
    HashJoin,
    /// Use sort-merge joins
    SortMergeJoin,
    /// Use hybrid approach
    Hybrid,
    /// Use parallel execution
    Parallel,
}

/// Query analysis results
#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    /// All variables discovered in the query
    pub variables: HashSet<Variable>,
    /// Variables that are bound (appear in triple patterns)
    pub bound_variables: HashSet<Variable>,
    /// Variables that are projected (appear in SELECT)
    pub projected_variables: HashSet<Variable>,
    /// Join variables (variables shared between patterns)
    pub join_variables: HashSet<Variable>,
    /// Filter safety analysis
    pub filter_safety: FilterSafetyAnalysis,
    /// Variable scoping information
    pub variable_scopes: HashMap<Variable, VariableScope>,
    /// Type inference results
    pub type_inference: TypeInferenceResults,
    /// Semantic validation issues
    pub validation_issues: Vec<ValidationIssue>,
    /// Index-aware optimization hints
    pub index_optimization: IndexOptimizationHints,
    /// Pattern cardinality estimates
    pub pattern_cardinalities: HashMap<usize, usize>,
    /// Join selectivity estimates
    pub join_selectivities: HashMap<(usize, usize), f64>,
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

/// Variable scope information
#[derive(Debug, Clone)]
pub struct VariableScope {
    /// Where the variable is defined
    pub definition_location: ScopeLocation,
    /// Where the variable is used
    pub usage_locations: Vec<ScopeLocation>,
    /// Whether the variable is optional
    pub is_optional: bool,
    /// Whether the variable can be null
    pub can_be_null: bool,
}

/// Location in the query where a variable appears
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScopeLocation {
    TriplePattern(usize),
    Filter(usize),
    Projection,
    GroupBy,
    OrderBy,
    Having,
    Bind,
    Subquery(usize),
}

/// Type inference results
#[derive(Debug, Clone)]
pub struct TypeInferenceResults {
    /// Inferred types for variables
    pub variable_types: HashMap<Variable, InferredType>,
    /// Type constraints from the query
    pub type_constraints: Vec<TypeConstraint>,
    /// Type conflicts (should be resolved or reported as errors)
    pub type_conflicts: Vec<TypeConflict>,
}

/// Inferred RDF term type
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InferredType {
    IRI,
    Literal(Option<String>), // Optional datatype IRI
    BlankNode,
    Unknown,
    Conflict(Vec<InferredType>),
}

/// Type constraint from query structure
#[derive(Debug, Clone)]
pub struct TypeConstraint {
    pub variable: Variable,
    pub required_type: InferredType,
    pub source: ConstraintSource,
    pub confidence: f64,
}

/// Source of a type constraint
#[derive(Debug, Clone)]
pub enum ConstraintSource {
    TripleSubject,
    TriplePredicate,
    TripleObject,
    FunctionArgument { function: String, position: usize },
    Comparison { operator: String },
    Filter(Expression),
}

/// Type conflict between different inferences
#[derive(Debug, Clone)]
pub struct TypeConflict {
    pub variable: Variable,
    pub conflicting_types: Vec<InferredType>,
    pub sources: Vec<ConstraintSource>,
}

/// Semantic validation issue
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub severity: ValidationSeverity,
    pub message: String,
    pub location: Option<ScopeLocation>,
    pub variable: Option<Variable>,
}

/// Severity of validation issues
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationSeverity {
    Error,
    Warning,
    Info,
}

/// Query analyzer
pub struct QueryAnalyzer {
    /// Enable advanced type inference
    enable_type_inference: bool,
    /// Enable semantic validation
    enable_validation: bool,
    /// Enable index-aware optimization
    enable_index_analysis: bool,
    /// Cost model for optimization
    cost_model: Option<CostModel>,
    /// Statistics collector for cardinality estimation
    statistics: Option<StatisticsCollector>,
    /// Available index information
    available_indexes: HashMap<String, Vec<IndexType>>,
}

impl QueryAnalyzer {
    /// Create a new query analyzer
    pub fn new() -> Self {
        Self {
            enable_type_inference: true,
            enable_validation: true,
            enable_index_analysis: true,
            cost_model: None,
            statistics: None,
            available_indexes: HashMap::new(),
        }
    }

    /// Create analyzer with specific settings
    pub fn with_config(enable_type_inference: bool, enable_validation: bool) -> Self {
        Self {
            enable_type_inference,
            enable_validation,
            enable_index_analysis: true,
            cost_model: None,
            statistics: None,
            available_indexes: HashMap::new(),
        }
    }

    /// Create analyzer with cost model and statistics
    pub fn with_cost_model(
        enable_type_inference: bool,
        enable_validation: bool,
        cost_model: CostModel,
        statistics: StatisticsCollector,
    ) -> Self {
        Self {
            enable_type_inference,
            enable_validation,
            enable_index_analysis: true,
            cost_model: Some(cost_model),
            statistics: Some(statistics),
            available_indexes: HashMap::new(),
        }
    }

    /// Add available index information
    pub fn add_index(&mut self, predicate: &str, index_type: IndexType) {
        self.available_indexes
            .entry(predicate.to_string())
            .or_insert_with(Vec::new)
            .push(index_type);
    }

    /// Set available indexes
    pub fn set_available_indexes(&mut self, indexes: HashMap<String, Vec<IndexType>>) {
        self.available_indexes = indexes;
    }

    /// Enable or disable index analysis
    pub fn set_index_analysis(&mut self, enable: bool) {
        self.enable_index_analysis = enable;
    }

    /// Analyze a SPARQL query algebra
    pub fn analyze(&self, algebra: &Algebra) -> Result<QueryAnalysis> {
        let mut analysis = QueryAnalysis {
            variables: HashSet::new(),
            bound_variables: HashSet::new(),
            projected_variables: HashSet::new(),
            join_variables: HashSet::new(),
            filter_safety: FilterSafetyAnalysis {
                safe_filters: Vec::new(),
                unsafe_filters: Vec::new(),
                filter_dependencies: Vec::new(),
            },
            variable_scopes: HashMap::new(),
            type_inference: TypeInferenceResults {
                variable_types: HashMap::new(),
                type_constraints: Vec::new(),
                type_conflicts: Vec::new(),
            },
            validation_issues: Vec::new(),
            index_optimization: IndexOptimizationHints {
                pattern_recommendations: HashMap::new(),
                join_order_hints: Vec::new(),
                filter_placement_hints: Vec::new(),
                execution_strategy: ExecutionStrategy::HashJoin,
            },
            pattern_cardinalities: HashMap::new(),
            join_selectivities: HashMap::new(),
        };

        // Step 1: Discover all variables
        self.discover_variables(algebra, &mut analysis)?;

        // Step 2: Analyze variable scoping
        self.analyze_scoping(algebra, &mut analysis)?;

        // Step 3: Identify join variables
        self.identify_join_variables(algebra, &mut analysis)?;

        // Step 4: Analyze filter safety
        self.analyze_filter_safety(algebra, &mut analysis)?;

        // Step 5: Type inference (if enabled)
        if self.enable_type_inference {
            self.infer_types(algebra, &mut analysis)?;
        }

        // Step 6: Semantic validation (if enabled)
        if self.enable_validation {
            self.validate_semantics(algebra, &mut analysis)?;
        }

        // Step 7: Index-aware optimization analysis (if enabled)
        if self.enable_index_analysis {
            self.analyze_index_optimization(algebra, &mut analysis)?;
        }

        Ok(analysis)
    }

    /// Discover all variables in the query
    fn discover_variables(&self, algebra: &Algebra, analysis: &mut QueryAnalysis) -> Result<()> {
        self.discover_variables_recursive(algebra, analysis, 0)
    }

    fn discover_variables_recursive(
        &self,
        algebra: &Algebra,
        analysis: &mut QueryAnalysis,
        depth: usize,
    ) -> Result<()> {
        match algebra {
            Algebra::Bgp(patterns) => {
                for pattern in patterns {
                    self.analyze_triple_pattern(pattern, analysis, depth)?;
                }
            }
            Algebra::Filter { condition, pattern } => {
                self.analyze_expression(condition, analysis)?;
                self.discover_variables_recursive(pattern, analysis, depth + 1)?;
            }
            Algebra::Join { left, right } => {
                self.discover_variables_recursive(left, analysis, depth + 1)?;
                self.discover_variables_recursive(right, analysis, depth + 1)?;
            }
            Algebra::LeftJoin { left, right, .. } => {
                self.discover_variables_recursive(left, analysis, depth + 1)?;
                self.discover_variables_recursive(right, analysis, depth + 1)?;
            }
            Algebra::Union { left, right } => {
                self.discover_variables_recursive(left, analysis, depth + 1)?;
                self.discover_variables_recursive(right, analysis, depth + 1)?;
            }
            Algebra::Project { variables, pattern } => {
                for var in variables {
                    analysis.variables.insert(var.clone());
                    analysis.projected_variables.insert(var.clone());
                }
                self.discover_variables_recursive(pattern, analysis, depth + 1)?;
            }
            Algebra::Extend {
                variable,
                expr,
                pattern,
            } => {
                analysis.variables.insert(variable.clone());
                self.analyze_expression(&expr, analysis)?;
                self.discover_variables_recursive(pattern, analysis, depth + 1)?;
            }
            Algebra::Distinct { pattern } | Algebra::Reduced { pattern } => {
                self.discover_variables_recursive(pattern, analysis, depth + 1)?;
            }
            Algebra::OrderBy { pattern, .. } => {
                self.discover_variables_recursive(pattern, analysis, depth + 1)?;
            }
            Algebra::Slice { pattern, .. } => {
                self.discover_variables_recursive(pattern, analysis, depth + 1)?;
            }
            Algebra::Group {
                variables,
                aggregates,
                pattern,
                ..
            } => {
                for group_condition in variables {
                    // Extract variables from the group condition expression
                    self.analyze_expression(&group_condition.expr, analysis)?;
                    if let Some(alias) = &group_condition.alias {
                        analysis.variables.insert(alias.clone());
                    }
                }
                for (var, _) in aggregates {
                    analysis.variables.insert(var.clone());
                }
                self.discover_variables_recursive(pattern, analysis, depth + 1)?;
            }
            Algebra::PropertyPath {
                subject, object, ..
            } => {
                self.analyze_term(subject, analysis);
                self.analyze_term(object, analysis);
            }
            Algebra::Minus { left, right } => {
                self.discover_variables_recursive(left, analysis, depth + 1)?;
                self.discover_variables_recursive(right, analysis, depth + 1)?;
            }
            Algebra::Service { pattern, .. } => {
                self.discover_variables_recursive(pattern, analysis, depth + 1)?;
            }
            Algebra::Graph { pattern, graph } => {
                self.analyze_term(graph, analysis);
                self.discover_variables_recursive(pattern, analysis, depth + 1)?;
            }
            Algebra::Having { pattern, condition } => {
                self.analyze_expression(condition, analysis)?;
                self.discover_variables_recursive(pattern, analysis, depth + 1)?;
            }
            Algebra::Values { variables, .. } => {
                for var in variables {
                    analysis.variables.insert(var.clone());
                }
            }
            Algebra::Table | Algebra::Zero => {
                // No variables to analyze
            }
        }
        Ok(())
    }

    fn analyze_term(&self, term: &Term, analysis: &mut QueryAnalysis) {
        if let Term::Variable(var) = term {
            analysis.variables.insert(var.clone());
        }
    }

    fn analyze_triple_pattern(
        &self,
        pattern: &TriplePattern,
        analysis: &mut QueryAnalysis,
        pattern_index: usize,
    ) -> Result<()> {
        if let Term::Variable(var) = &pattern.subject {
            analysis.variables.insert(var.clone());
            analysis.bound_variables.insert(var.clone());
            self.record_variable_scope(var, ScopeLocation::TriplePattern(pattern_index), analysis);
        }

        if let Term::Variable(var) = &pattern.predicate {
            analysis.variables.insert(var.clone());
            analysis.bound_variables.insert(var.clone());
            self.record_variable_scope(var, ScopeLocation::TriplePattern(pattern_index), analysis);
        }

        if let Term::Variable(var) = &pattern.object {
            analysis.variables.insert(var.clone());
            analysis.bound_variables.insert(var.clone());
            self.record_variable_scope(var, ScopeLocation::TriplePattern(pattern_index), analysis);
        }

        Ok(())
    }

    fn analyze_expression(
        &self,
        expression: &Expression,
        analysis: &mut QueryAnalysis,
    ) -> Result<()> {
        match expression {
            Expression::Variable(var) => {
                analysis.variables.insert(var.clone());
            }
            Expression::Binary { left, right, .. } => {
                self.analyze_expression(left, analysis)?;
                self.analyze_expression(right, analysis)?;
            }
            Expression::Unary { expr: operand, .. } => {
                self.analyze_expression(operand, analysis)?;
            }
            Expression::Function { args, .. } => {
                for arg in args {
                    self.analyze_expression(arg, analysis)?;
                }
            }
            Expression::Conditional {
                condition,
                then_expr: if_true,
                else_expr: if_false,
            } => {
                self.analyze_expression(condition, analysis)?;
                self.analyze_expression(if_true, analysis)?;
                self.analyze_expression(if_false, analysis)?;
            }
            Expression::Bound(var) => {
                analysis.variables.insert(var.clone());
            }
            _ => {} // Literals, constants, etc.
        }
        Ok(())
    }

    fn record_variable_scope(
        &self,
        variable: &Variable,
        location: ScopeLocation,
        analysis: &mut QueryAnalysis,
    ) {
        let scope = analysis
            .variable_scopes
            .entry(variable.clone())
            .or_insert_with(|| VariableScope {
                definition_location: location.clone(),
                usage_locations: Vec::new(),
                is_optional: false,
                can_be_null: false,
            });
        scope.usage_locations.push(location);
    }

    /// Analyze variable scoping rules
    fn analyze_scoping(&self, algebra: &Algebra, analysis: &mut QueryAnalysis) -> Result<()> {
        // For now, simplified scoping analysis
        // In a full implementation, this would track variable visibility rules
        Ok(())
    }

    /// Identify variables that participate in joins
    fn identify_join_variables(
        &self,
        algebra: &Algebra,
        analysis: &mut QueryAnalysis,
    ) -> Result<()> {
        let mut pattern_variables: Vec<HashSet<Variable>> = Vec::new();
        self.collect_pattern_variables(algebra, &mut pattern_variables)?;

        // Find variables that appear in multiple patterns
        for i in 0..pattern_variables.len() {
            for j in (i + 1)..pattern_variables.len() {
                let intersection: HashSet<_> = pattern_variables[i]
                    .intersection(&pattern_variables[j])
                    .cloned()
                    .collect();
                analysis.join_variables.extend(intersection);
            }
        }

        Ok(())
    }

    fn collect_pattern_variables(
        &self,
        algebra: &Algebra,
        pattern_variables: &mut Vec<HashSet<Variable>>,
    ) -> Result<()> {
        match algebra {
            Algebra::Bgp(patterns) if patterns.len() == 1 => {
                let pattern = &patterns[0];
                let mut vars = HashSet::new();
                if let Term::Variable(var) = &pattern.subject {
                    vars.insert(var.clone());
                }
                if let Term::Variable(var) = &pattern.predicate {
                    vars.insert(var.clone());
                }
                if let Term::Variable(var) = &pattern.object {
                    vars.insert(var.clone());
                }
                pattern_variables.push(vars);
            }
            Algebra::Bgp(patterns) => {
                for pattern in patterns {
                    let mut vars = HashSet::new();
                    if let Term::Variable(var) = &pattern.subject {
                        vars.insert(var.clone());
                    }
                    if let Term::Variable(var) = &pattern.predicate {
                        vars.insert(var.clone());
                    }
                    if let Term::Variable(var) = &pattern.object {
                        vars.insert(var.clone());
                    }
                    pattern_variables.push(vars);
                }
            }
            Algebra::Join { left, right }
            | Algebra::LeftJoin { left, right, .. }
            | Algebra::Union { left, right } => {
                self.collect_pattern_variables(left, pattern_variables)?;
                self.collect_pattern_variables(right, pattern_variables)?;
            }
            Algebra::Filter { pattern, .. }
            | Algebra::Project { pattern, .. }
            | Algebra::Extend { pattern, .. }
            | Algebra::Distinct { pattern }
            | Algebra::Reduced { pattern }
            | Algebra::OrderBy { pattern, .. }
            | Algebra::Slice { pattern, .. }
            | Algebra::Group { pattern, .. } => {
                self.collect_pattern_variables(pattern, pattern_variables)?;
            }
            Algebra::PropertyPath {
                subject, object, ..
            } => {
                let mut vars = HashSet::new();
                if let Term::Variable(var) = subject {
                    vars.insert(var.clone());
                }
                if let Term::Variable(var) = object {
                    vars.insert(var.clone());
                }
                pattern_variables.push(vars);
            }
            Algebra::Minus { left, right } => {
                self.collect_pattern_variables(left, pattern_variables)?;
                self.collect_pattern_variables(right, pattern_variables)?;
            }
            Algebra::Service { pattern, .. } => {
                self.collect_pattern_variables(pattern, pattern_variables)?;
            }
            Algebra::Graph { pattern, .. } => {
                self.collect_pattern_variables(pattern, pattern_variables)?;
            }
            Algebra::Having { pattern, .. } => {
                self.collect_pattern_variables(pattern, pattern_variables)?;
            }
            Algebra::Values { .. } | Algebra::Table | Algebra::Zero => {
                // These don't contribute pattern variables
            }
        }
        Ok(())
    }

    /// Analyze filter safety
    fn analyze_filter_safety(&self, algebra: &Algebra, analysis: &mut QueryAnalysis) -> Result<()> {
        self.analyze_filter_safety_recursive(algebra, analysis)
    }

    fn analyze_filter_safety_recursive(
        &self,
        algebra: &Algebra,
        analysis: &mut QueryAnalysis,
    ) -> Result<()> {
        match algebra {
            Algebra::Filter { condition, pattern } => {
                // Analyze the filter expression for safety
                let mut dependencies = HashSet::new();
                self.collect_filter_dependencies(condition, &mut dependencies)?;

                // A filter is considered safe if all its dependencies are bound before evaluation
                let is_safe = dependencies
                    .iter()
                    .all(|var| analysis.bound_variables.contains(var));

                if is_safe {
                    analysis.filter_safety.safe_filters.push(condition.clone());
                } else {
                    analysis
                        .filter_safety
                        .unsafe_filters
                        .push(condition.clone());
                }

                analysis
                    .filter_safety
                    .filter_dependencies
                    .push((condition.clone(), dependencies));

                self.analyze_filter_safety_recursive(pattern, analysis)?;
            }
            Algebra::Join { left, right }
            | Algebra::LeftJoin { left, right, .. }
            | Algebra::Union { left, right } => {
                self.analyze_filter_safety_recursive(left, analysis)?;
                self.analyze_filter_safety_recursive(right, analysis)?;
            }
            Algebra::Project { pattern, .. }
            | Algebra::Extend { pattern, .. }
            | Algebra::Distinct { pattern }
            | Algebra::Reduced { pattern }
            | Algebra::OrderBy { pattern, .. }
            | Algebra::Slice { pattern, .. }
            | Algebra::Group { pattern, .. } => {
                self.analyze_filter_safety_recursive(pattern, analysis)?;
            }
            _ => {} // Base cases
        }
        Ok(())
    }

    fn collect_filter_dependencies(
        &self,
        expression: &Expression,
        dependencies: &mut HashSet<Variable>,
    ) -> Result<()> {
        match expression {
            Expression::Variable(var) => {
                dependencies.insert(var.clone());
            }
            Expression::Binary { left, right, .. } => {
                self.collect_filter_dependencies(left, dependencies)?;
                self.collect_filter_dependencies(right, dependencies)?;
            }
            Expression::Unary { expr, .. } => {
                self.collect_filter_dependencies(expr, dependencies)?;
            }
            Expression::Function { args, .. } => {
                for arg in args {
                    self.collect_filter_dependencies(arg, dependencies)?;
                }
            }
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.collect_filter_dependencies(condition, dependencies)?;
                self.collect_filter_dependencies(then_expr, dependencies)?;
                self.collect_filter_dependencies(else_expr, dependencies)?;
            }
            Expression::Bound(var) => {
                dependencies.insert(var.clone());
            }
            _ => {} // Literals, constants
        }
        Ok(())
    }

    /// Infer types for variables
    fn infer_types(&self, algebra: &Algebra, analysis: &mut QueryAnalysis) -> Result<()> {
        // Basic type inference based on position in triple patterns
        self.infer_types_from_patterns(algebra, analysis)?;

        // Resolve type conflicts
        self.resolve_type_conflicts(analysis)?;

        Ok(())
    }

    fn infer_types_from_patterns(
        &self,
        algebra: &Algebra,
        analysis: &mut QueryAnalysis,
    ) -> Result<()> {
        match algebra {
            Algebra::Bgp(patterns) => {
                for pattern in patterns {
                    self.infer_types_from_triple_pattern(pattern, analysis)?;
                }
            }
            _ => {
                // Recursively process other algebra types
                // Implementation would continue for all algebra types
            }
        }
        Ok(())
    }

    /// Infer types from a single triple pattern
    fn infer_types_from_triple_pattern(
        &self,
        pattern: &TriplePattern,
        analysis: &mut QueryAnalysis,
    ) -> Result<()> {
        // Subject is typically IRI or blank node
        if let Term::Variable(var) = &pattern.subject {
            let constraint = TypeConstraint {
                variable: var.clone(),
                required_type: InferredType::IRI,
                source: ConstraintSource::TripleSubject,
                confidence: 0.8,
            };
            analysis.type_inference.type_constraints.push(constraint);
        }

        // Predicate is always IRI
        if let Term::Variable(var) = &pattern.predicate {
            let constraint = TypeConstraint {
                variable: var.clone(),
                required_type: InferredType::IRI,
                source: ConstraintSource::TriplePredicate,
                confidence: 1.0,
            };
            analysis.type_inference.type_constraints.push(constraint);
        }

        // Object can be IRI, literal, or blank node
        if let Term::Variable(var) = &pattern.object {
            let constraint = TypeConstraint {
                variable: var.clone(),
                required_type: InferredType::Unknown,
                source: ConstraintSource::TripleObject,
                confidence: 0.3,
            };
            analysis.type_inference.type_constraints.push(constraint);
        }

        Ok(())
    }

    fn resolve_type_conflicts(&self, analysis: &mut QueryAnalysis) -> Result<()> {
        // Group constraints by variable
        let mut constraints_by_var: HashMap<Variable, Vec<&TypeConstraint>> = HashMap::new();

        for constraint in &analysis.type_inference.type_constraints {
            constraints_by_var
                .entry(constraint.variable.clone())
                .or_insert_with(Vec::new)
                .push(constraint);
        }

        // Resolve conflicts for each variable
        for (var, constraints) in constraints_by_var {
            if constraints.len() > 1 {
                // Check for conflicts
                let mut types: HashSet<_> = constraints.iter().map(|c| &c.required_type).collect();

                if types.len() > 1 {
                    // There's a conflict - record it
                    let conflict = TypeConflict {
                        variable: var.clone(),
                        conflicting_types: types.into_iter().cloned().collect(),
                        sources: constraints.iter().map(|c| c.source.clone()).collect(),
                    };
                    analysis.type_inference.type_conflicts.push(conflict);
                }
            }

            // Infer the most likely type based on confidence
            if let Some(best_constraint) = constraints
                .iter()
                .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            {
                analysis
                    .type_inference
                    .variable_types
                    .insert(var.clone(), best_constraint.required_type.clone());
            }
        }

        Ok(())
    }

    /// Validate query semantics
    fn validate_semantics(&self, algebra: &Algebra, analysis: &mut QueryAnalysis) -> Result<()> {
        // Check for unbound variables in projection
        for var in &analysis.projected_variables {
            if !analysis.bound_variables.contains(var) {
                analysis.validation_issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    message: format!("Variable {} is projected but not bound", var.name()),
                    location: Some(ScopeLocation::Projection),
                    variable: Some(var.clone()),
                });
            }
        }

        // Check for unused variables
        for var in &analysis.bound_variables {
            if !analysis.projected_variables.contains(var) && !analysis.join_variables.contains(var)
            {
                analysis.validation_issues.push(ValidationIssue {
                    severity: ValidationSeverity::Warning,
                    message: format!("Variable {} is bound but not used", var.name()),
                    location: None,
                    variable: Some(var.clone()),
                });
            }
        }

        // Check type conflicts
        for conflict in &analysis.type_inference.type_conflicts {
            analysis.validation_issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: format!(
                    "Type conflict for variable {}: {:?}",
                    conflict.variable.name(),
                    conflict.conflicting_types
                ),
                location: None,
                variable: Some(conflict.variable.clone()),
            });
        }

        Ok(())
    }

    /// Perform index-aware optimization analysis
    fn analyze_index_optimization(
        &self,
        algebra: &Algebra,
        analysis: &mut QueryAnalysis,
    ) -> Result<()> {
        // Collect all BGP patterns for analysis
        let mut bgp_patterns = Vec::new();
        self.collect_bgp_patterns(algebra, &mut bgp_patterns)?;

        // Analyze each pattern for index opportunities
        for (pattern_index, pattern) in bgp_patterns.iter().enumerate() {
            let pattern_analysis = self.analyze_pattern_indexes(pattern, pattern_index)?;
            analysis
                .index_optimization
                .pattern_recommendations
                .insert(pattern_index, pattern_analysis);

            // Estimate pattern cardinality
            let cardinality = self.estimate_pattern_cardinality(pattern);
            analysis
                .pattern_cardinalities
                .insert(pattern_index, cardinality);
        }

        // Analyze join order based on index availability
        self.analyze_join_order(&bgp_patterns, analysis)?;

        // Analyze filter placement opportunities
        self.analyze_filter_placement(algebra, analysis)?;

        // Determine overall execution strategy
        analysis.index_optimization.execution_strategy =
            self.determine_execution_strategy(&analysis);

        Ok(())
    }

    /// Collect all BGP patterns from the algebra
    fn collect_bgp_patterns(
        &self,
        algebra: &Algebra,
        patterns: &mut Vec<TriplePattern>,
    ) -> Result<()> {
        match algebra {
            Algebra::Bgp(bgp_patterns) => {
                patterns.extend(bgp_patterns.iter().cloned());
            }
            Algebra::Join { left, right }
            | Algebra::LeftJoin { left, right, .. }
            | Algebra::Union { left, right }
            | Algebra::Minus { left, right } => {
                self.collect_bgp_patterns(left, patterns)?;
                self.collect_bgp_patterns(right, patterns)?;
            }
            Algebra::Filter { pattern, .. }
            | Algebra::Project { pattern, .. }
            | Algebra::Extend { pattern, .. }
            | Algebra::Distinct { pattern }
            | Algebra::Reduced { pattern }
            | Algebra::OrderBy { pattern, .. }
            | Algebra::Slice { pattern, .. }
            | Algebra::Group { pattern, .. }
            | Algebra::Service { pattern, .. }
            | Algebra::Graph { pattern, .. }
            | Algebra::Having { pattern, .. } => {
                self.collect_bgp_patterns(pattern, patterns)?;
            }
            _ => {} // Terminal nodes
        }
        Ok(())
    }

    /// Analyze index opportunities for a single pattern
    fn analyze_pattern_indexes(
        &self,
        pattern: &TriplePattern,
        pattern_index: usize,
    ) -> Result<PatternIndexAnalysis> {
        let mut available_indexes = Vec::new();
        let full_scan_cardinality = self.estimate_pattern_cardinality(pattern);

        // Analyze subject position
        if let Term::Variable(_) = &pattern.subject {
            available_indexes.extend(self.analyze_position_indexes(pattern, 0, pattern_index));
        }

        // Analyze predicate position
        if let Term::Variable(_) = &pattern.predicate {
            available_indexes.extend(self.analyze_position_indexes(pattern, 1, pattern_index));
        } else if let Term::Iri(predicate_iri) = &pattern.predicate {
            // Check for indexes on this specific predicate
            let predicate_str = predicate_iri.as_str();
            if let Some(index_types) = self.available_indexes.get(predicate_str) {
                for index_type in index_types {
                    let selectivity = self.estimate_predicate_selectivity(predicate_str);
                    let cost =
                        self.estimate_index_cost(index_type, selectivity, full_scan_cardinality);

                    available_indexes.push(IndexAccess {
                        index_type: index_type.clone(),
                        pattern_position: 1,
                        pattern_index,
                        selectivity,
                        cost_estimate: cost,
                        io_pattern: self.get_io_pattern_for_index(index_type),
                        improvement_ratio: 1.0 / selectivity, // Better selectivity = higher improvement
                    });
                }
            }
        }

        // Analyze object position
        if let Term::Variable(_) = &pattern.object {
            available_indexes.extend(self.analyze_position_indexes(pattern, 2, pattern_index));
        }

        // Find the best index access method
        let recommended_access = available_indexes
            .iter()
            .min_by(|a, b| {
                a.cost_estimate
                    .total_cost
                    .partial_cmp(&b.cost_estimate.total_cost)
                    .unwrap()
            })
            .cloned();

        let indexed_cardinality = recommended_access
            .as_ref()
            .map(|access| (full_scan_cardinality as f64 * access.selectivity) as usize)
            .unwrap_or(full_scan_cardinality);

        let improvement_ratio = if indexed_cardinality > 0 {
            full_scan_cardinality as f64 / indexed_cardinality as f64
        } else {
            1.0
        };

        Ok(PatternIndexAnalysis {
            available_indexes,
            recommended_access,
            full_scan_cardinality,
            indexed_cardinality,
            improvement_ratio,
        })
    }

    /// Analyze index opportunities for a specific position in a pattern
    fn analyze_position_indexes(
        &self,
        pattern: &TriplePattern,
        position: usize,
        pattern_index: usize,
    ) -> Vec<IndexAccess> {
        let mut indexes = Vec::new();
        let cardinality = self.estimate_pattern_cardinality(pattern);

        // Analyze different index types for this position
        for index_type in &[IndexType::BTree, IndexType::Hash] {
            let selectivity = self.estimate_position_selectivity(pattern, position);
            let cost = self.estimate_index_cost(index_type, selectivity, cardinality);

            indexes.push(IndexAccess {
                index_type: index_type.clone(),
                pattern_position: position,
                pattern_index,
                selectivity,
                cost_estimate: cost,
                io_pattern: self.get_io_pattern_for_index(index_type),
                improvement_ratio: 1.0 / selectivity, // Better selectivity = higher improvement
            });
        }

        indexes
    }

    /// Analyze optimal join order based on index availability
    fn analyze_join_order(
        &self,
        patterns: &[TriplePattern],
        analysis: &mut QueryAnalysis,
    ) -> Result<()> {
        if patterns.len() <= 1 {
            return Ok(());
        }

        // Generate join order recommendations based on selectivity and index availability
        let mut pattern_costs: Vec<(usize, f64)> = Vec::new();

        for (i, pattern) in patterns.iter().enumerate() {
            let cardinality = analysis.pattern_cardinalities.get(&i).unwrap_or(&1000000);
            let has_good_index = analysis
                .index_optimization
                .pattern_recommendations
                .get(&i)
                .and_then(|p| p.recommended_access.as_ref())
                .map(|access| access.selectivity < 0.5) // Good selectivity indicates good improvement
                .unwrap_or(false);

            let effective_cost = if has_good_index {
                *cardinality as f64 / 10.0 // Bonus for indexed patterns
            } else {
                *cardinality as f64
            };

            pattern_costs.push((i, effective_cost));
        }

        // Sort by effective cost (lowest first)
        pattern_costs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let pattern_order: Vec<usize> = pattern_costs.into_iter().map(|(i, _)| i).collect();
        let total_cost = self.estimate_join_sequence_cost(&pattern_order, analysis);

        let hint = JoinOrderHint {
            pattern_order,
            estimated_cost: total_cost,
            reasoning: "Index-aware join ordering based on selectivity and index availability"
                .to_string(),
        };

        analysis.index_optimization.join_order_hints.push(hint);

        Ok(())
    }

    /// Analyze filter placement opportunities
    fn analyze_filter_placement(
        &self,
        algebra: &Algebra,
        analysis: &mut QueryAnalysis,
    ) -> Result<()> {
        // For each filter, recommend optimal placement
        for filter in &analysis.filter_safety.safe_filters {
            let selectivity = self.estimate_expression_selectivity(filter);

            // Find the earliest pattern where all filter dependencies are bound
            let mut dependencies = HashSet::new();
            let _ = self.collect_filter_dependencies(filter, &mut dependencies);

            if let Some(earliest_pattern) =
                self.find_earliest_binding_pattern(&dependencies, analysis)
            {
                let cost_benefit = selectivity * 1000.0; // Simplified cost benefit

                let hint = FilterPlacementHint {
                    filter: filter.clone(),
                    recommended_placement: earliest_pattern,
                    selectivity,
                    cost_benefit,
                };

                analysis
                    .index_optimization
                    .filter_placement_hints
                    .push(hint);
            }
        }

        Ok(())
    }

    /// Determine overall execution strategy
    fn determine_execution_strategy(&self, analysis: &QueryAnalysis) -> ExecutionStrategy {
        let has_good_indexes = analysis
            .index_optimization
            .pattern_recommendations
            .values()
            .any(|p| {
                p.recommended_access
                    .as_ref()
                    .map(|a| a.improvement_ratio > 5.0)
                    .unwrap_or(false)
            });

        let total_cardinality: usize = analysis.pattern_cardinalities.values().sum();
        let pattern_count = analysis.pattern_cardinalities.len();

        match (has_good_indexes, total_cardinality, pattern_count) {
            (true, _, _) => ExecutionStrategy::IndexNestedLoop,
            (false, cardinality, _) if cardinality > 1_000_000 => ExecutionStrategy::Parallel,
            (false, _, count) if count <= 2 => ExecutionStrategy::HashJoin,
            (false, _, _) => ExecutionStrategy::SortMergeJoin,
        }
    }

    /// Estimate pattern cardinality
    fn estimate_pattern_cardinality(&self, pattern: &TriplePattern) -> usize {
        if let Some(stats) = &self.statistics {
            // Use statistics if available
            self.estimate_pattern_cardinality_with_stats(pattern, stats)
        } else {
            // Default heuristic estimation
            match (&pattern.subject, &pattern.predicate, &pattern.object) {
                (Term::Variable(_), Term::Variable(_), Term::Variable(_)) => 1_000_000, // Very high
                (Term::Variable(_), Term::Variable(_), _) => 100_000,
                (Term::Variable(_), _, Term::Variable(_)) => 10_000,
                (_, Term::Variable(_), Term::Variable(_)) => 50_000,
                (Term::Variable(_), _, _) => 1_000,
                (_, Term::Variable(_), _) => 5_000,
                (_, _, Term::Variable(_)) => 2_000,
                _ => 1, // All bound
            }
        }
    }

    /// Estimate pattern cardinality using statistics
    fn estimate_pattern_cardinality_with_stats(
        &self,
        _pattern: &TriplePattern,
        _stats: &StatisticsCollector,
    ) -> usize {
        // This would use actual statistics from the collector
        // For now, return a placeholder
        10_000
    }

    /// Estimate predicate selectivity
    fn estimate_predicate_selectivity(&self, predicate: &str) -> f64 {
        // This would use statistics or heuristics
        // Common predicates might have known selectivities
        match predicate {
            p if p.contains("type") => 0.1,
            p if p.contains("label") => 0.5,
            _ => 0.3,
        }
    }

    /// Estimate position selectivity for a pattern
    fn estimate_position_selectivity(&self, pattern: &TriplePattern, position: usize) -> f64 {
        match position {
            0 => 0.1,  // Subject is typically selective
            1 => 0.05, // Predicate is very selective
            2 => 0.3,  // Object varies
            _ => 1.0,
        }
    }

    /// Estimate cost for using a specific index
    fn estimate_index_cost(
        &self,
        index_type: &IndexType,
        selectivity: f64,
        base_cardinality: usize,
    ) -> CostEstimate {
        let result_cardinality = (base_cardinality as f64 * selectivity) as usize;

        match index_type {
            IndexType::BTree => {
                // B-tree has log(n) access cost
                let access_cost = (base_cardinality as f64).log2() * 10.0;
                CostEstimate::new(
                    access_cost,
                    access_cost * 0.1,
                    100.0,
                    0.0,
                    result_cardinality,
                )
            }
            IndexType::Hash => {
                // Hash has O(1) access cost but only for equality
                CostEstimate::new(10.0, 5.0, 50.0, 0.0, result_cardinality)
            }
            IndexType::FullText => CostEstimate::new(50.0, 20.0, 200.0, 0.0, result_cardinality),
            IndexType::Spatial => CostEstimate::new(100.0, 50.0, 300.0, 0.0, result_cardinality),
            IndexType::Custom(_) => CostEstimate::new(75.0, 30.0, 150.0, 0.0, result_cardinality),
        }
    }

    /// Get I/O pattern for index type
    fn get_io_pattern_for_index(&self, index_type: &IndexType) -> IOPattern {
        match index_type {
            IndexType::BTree => IOPattern::IndexScan,
            IndexType::Hash => IOPattern::Random,
            IndexType::FullText => IOPattern::Sequential,
            IndexType::Spatial => IOPattern::Random,
            IndexType::Custom(_) => IOPattern::IndexScan,
        }
    }

    /// Estimate join sequence cost
    fn estimate_join_sequence_cost(
        &self,
        pattern_order: &[usize],
        analysis: &QueryAnalysis,
    ) -> CostEstimate {
        let mut total_cost = 0.0;
        let mut cumulative_cardinality = 1;

        for &pattern_idx in pattern_order {
            let pattern_cardinality = analysis
                .pattern_cardinalities
                .get(&pattern_idx)
                .unwrap_or(&1000);
            cumulative_cardinality = (cumulative_cardinality * pattern_cardinality).min(1_000_000);
            total_cost += cumulative_cardinality as f64;
        }

        CostEstimate::new(
            total_cost,
            total_cost * 0.1,
            total_cost * 0.05,
            0.0,
            cumulative_cardinality,
        )
    }

    /// Find earliest pattern that binds all dependencies
    fn find_earliest_binding_pattern(
        &self,
        dependencies: &HashSet<Variable>,
        analysis: &QueryAnalysis,
    ) -> Option<usize> {
        // This would analyze which pattern first binds all the required variables
        // For now, return pattern 0 as a placeholder
        if !dependencies.is_empty() {
            Some(0)
        } else {
            None
        }
    }

    /// Estimate expression selectivity (reuse existing method but make it more sophisticated)
    fn estimate_expression_selectivity(&self, expression: &Expression) -> f64 {
        match expression {
            Expression::Binary { op, .. } => {
                // Different operators have different selectivity characteristics
                match op.to_string().as_str() {
                    "=" => 0.1,
                    "!=" => 0.9,
                    "<" | ">" => 0.33,
                    "<=" | ">=" => 0.5,
                    _ => 0.5,
                }
            }
            Expression::Function { name, .. } => {
                // Extract local name from IRI (everything after last # or /)
                let local_name = name.split(&['#', '/'][..]).last().unwrap_or(name);
                match local_name {
                    "regex" => 0.3,
                    "contains" => 0.2,
                    "starts" | "ends" => 0.15,
                    _ => 0.5,
                }
            }
            _ => 0.5, // Default selectivity
        }
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
    use crate::algebra::{BinaryOperator, Term, Variable};
    use oxirs_core::model::NamedNode;

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
        assert!(!analysis
            .index_optimization
            .pattern_recommendations
            .is_empty());

        // Check that pattern cardinalities were estimated
        assert!(!analysis.pattern_cardinalities.is_empty());

        // Check that a pattern analysis was generated for pattern 0
        assert!(analysis
            .index_optimization
            .pattern_recommendations
            .contains_key(&0));

        let pattern_analysis = &analysis.index_optimization.pattern_recommendations[&0];

        // Should have some available indexes since we added one for this predicate
        assert!(!pattern_analysis.available_indexes.is_empty());

        // Should have a recommended access method
        assert!(pattern_analysis.recommended_access.is_some());

        // Check that execution strategy was determined
        matches!(
            analysis.index_optimization.execution_strategy,
            ExecutionStrategy::IndexNestedLoop
                | ExecutionStrategy::HashJoin
                | ExecutionStrategy::SortMergeJoin
                | ExecutionStrategy::Hybrid
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
        assert!(!analysis.index_optimization.join_order_hints.is_empty());

        // Should have estimated costs
        let hint = &analysis.index_optimization.join_order_hints[0];
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
