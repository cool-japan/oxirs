//! Comprehensive SPARQL Query Validation
//!
//! This module provides extensive validation for SPARQL queries, including:
//! - Syntax validation
//! - Semantic validation (variable bindings, type consistency)
//! - Best practices and performance warnings
//! - Security validation (injection risks, resource limits)
//! - SPARQL 1.1/1.2 compliance checking

use crate::algebra::{Aggregate, Algebra, Expression, Term, Variable};
use crate::query_analysis::{ValidationError, ValidationErrorType};
use anyhow::Result;
use std::collections::HashSet;

/// Comprehensive query validator
pub struct QueryValidator {
    /// Configuration
    config: ValidationConfig,
    /// Statistics for validation
    stats: ValidationStatistics,
}

/// Validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enable strict SPARQL compliance checking
    pub strict_compliance: bool,
    /// Enable performance warnings
    pub performance_warnings: bool,
    /// Enable security checks
    pub security_checks: bool,
    /// Maximum query complexity allowed
    pub max_complexity: usize,
    /// Maximum number of triple patterns
    pub max_triple_patterns: usize,
    /// Maximum recursion depth for property paths
    pub max_path_depth: usize,
    /// Warn about cartesian products
    pub warn_cartesian_products: bool,
    /// Check for type consistency
    pub check_type_consistency: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strict_compliance: true,
            performance_warnings: true,
            security_checks: true,
            max_complexity: 1000,
            max_triple_patterns: 100,
            max_path_depth: 10,
            warn_cartesian_products: true,
            check_type_consistency: true,
        }
    }
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// List of validation errors (must be fixed)
    pub errors: Vec<ValidationError>,
    /// List of warnings (should be reviewed)
    pub warnings: Vec<ValidationWarning>,
    /// Query complexity score
    pub complexity_score: usize,
    /// Validation passed
    pub is_valid: bool,
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning type
    pub warning_type: ValidationWarningType,
    /// Warning message
    pub message: String,
    /// Location where warning occurred
    pub location: String,
    /// Suggestion for improvement
    pub suggestion: Option<String>,
    /// Severity level (1-10)
    pub severity: u8,
}

/// Validation warning types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationWarningType {
    /// Performance concern
    Performance,
    /// Best practices violation
    BestPractice,
    /// Potential cartesian product
    CartesianProduct,
    /// Deprecated feature usage
    Deprecated,
    /// Type inconsistency (non-fatal)
    TypeInconsistency,
    /// Unbounded query
    UnboundedQuery,
    /// Complex filter expression
    ComplexFilter,
    /// Missing index hint
    MissingIndexHint,
}

/// Validation statistics
#[derive(Debug, Clone, Default)]
pub struct ValidationStatistics {
    /// Total queries validated
    pub total_validated: usize,
    /// Total errors found
    pub total_errors: usize,
    /// Total warnings found
    pub total_warnings: usize,
}

impl QueryValidator {
    /// Create a new query validator
    pub fn new() -> Self {
        Self::with_config(ValidationConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self {
            config,
            stats: ValidationStatistics::default(),
        }
    }

    /// Validate a SPARQL query algebra
    pub fn validate(&mut self, algebra: &Algebra) -> Result<ValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // 1. Validate variable bindings
        self.validate_variable_bindings(algebra, &mut errors)?;

        // 2. Validate aggregates
        self.validate_aggregates(algebra, &mut errors)?;

        // 3. Check for cartesian products
        if self.config.warn_cartesian_products {
            Self::check_cartesian_products(algebra, &mut warnings)?;
        }

        // 4. Validate filters
        Self::validate_filters(algebra, &mut errors, &mut warnings)?;

        // 5. Check query complexity
        let complexity_score = Self::calculate_complexity(algebra)?;
        if complexity_score > self.config.max_complexity {
            errors.push(ValidationError {
                error_type: ValidationErrorType::SemanticInconsistency,
                message: format!(
                    "Query complexity ({}) exceeds maximum allowed ({})",
                    complexity_score, self.config.max_complexity
                ),
                location: "Overall query".to_string(),
                suggestion: Some("Consider breaking the query into smaller parts".to_string()),
            });
        }

        // 6. Performance warnings
        if self.config.performance_warnings {
            self.check_performance_issues(algebra, &mut warnings)?;
        }

        // 7. Security checks
        if self.config.security_checks {
            self.check_security_issues(algebra, &mut warnings)?;
        }

        // 8. Type consistency checks
        if self.config.check_type_consistency {
            self.check_type_consistency(algebra, &mut warnings)?;
        }

        // Update statistics
        self.stats.total_validated += 1;
        self.stats.total_errors += errors.len();
        self.stats.total_warnings += warnings.len();

        let is_valid = errors.is_empty();

        Ok(ValidationResult {
            errors,
            warnings,
            complexity_score,
            is_valid,
        })
    }

    /// Validate variable bindings
    fn validate_variable_bindings(
        &self,
        algebra: &Algebra,
        errors: &mut Vec<ValidationError>,
    ) -> Result<()> {
        match algebra {
            Algebra::Project { variables, pattern } => {
                let bound_vars = Self::collect_bound_variables(pattern)?;

                for var in variables {
                    if !bound_vars.contains(var) {
                        errors.push(ValidationError {
                            error_type: ValidationErrorType::UnboundVariable,
                            message: format!(
                                "Variable ?{} in SELECT clause is not bound in the query pattern",
                                var.as_str()
                            ),
                            location: "SELECT clause".to_string(),
                            suggestion: Some(format!(
                                "Add a triple pattern that binds ?{}",
                                var.as_str()
                            )),
                        });
                    }
                }
            }
            Algebra::Group {
                pattern,
                variables,
                aggregates,
            } => {
                let bound_vars = Self::collect_bound_variables(pattern)?;

                // Validate GROUP BY variables
                for group_cond in variables {
                    // Validate the grouping expression
                    Self::validate_expression_variables(&group_cond.expr, &bound_vars, errors)?;
                }

                // Validate aggregate expressions
                for (_, agg) in aggregates {
                    self.validate_aggregate_expression(agg, &bound_vars, errors)?;
                }
            }
            Algebra::OrderBy { pattern, .. } => {
                self.validate_variable_bindings(pattern, errors)?;
            }
            Algebra::Join { left, right } | Algebra::LeftJoin { left, right, .. } => {
                self.validate_variable_bindings(left, errors)?;
                self.validate_variable_bindings(right, errors)?;
            }
            Algebra::Union { left, right } => {
                self.validate_variable_bindings(left, errors)?;
                self.validate_variable_bindings(right, errors)?;
            }
            Algebra::Filter { pattern, .. } => {
                self.validate_variable_bindings(pattern, errors)?;
            }
            _ => {}
        }

        Ok(())
    }

    /// Collect all bound variables in algebra
    fn collect_bound_variables(algebra: &Algebra) -> Result<HashSet<Variable>> {
        let mut vars = HashSet::new();

        match algebra {
            Algebra::Bgp(patterns) => {
                for pattern in patterns {
                    if let Term::Variable(v) = &pattern.subject {
                        vars.insert(v.clone());
                    }
                    if let Term::Variable(v) = &pattern.predicate {
                        vars.insert(v.clone());
                    }
                    if let Term::Variable(v) = &pattern.object {
                        vars.insert(v.clone());
                    }
                }
            }
            Algebra::Join { left, right } => {
                vars.extend(Self::collect_bound_variables(left)?);
                vars.extend(Self::collect_bound_variables(right)?);
            }
            Algebra::LeftJoin { left, right, .. } => {
                vars.extend(Self::collect_bound_variables(left)?);
                vars.extend(Self::collect_bound_variables(right)?);
            }
            Algebra::Union { left, right } => {
                vars.extend(Self::collect_bound_variables(left)?);
                vars.extend(Self::collect_bound_variables(right)?);
            }
            Algebra::Filter { pattern, .. } => {
                vars.extend(Self::collect_bound_variables(pattern)?);
            }
            Algebra::Project { pattern, .. } => {
                vars.extend(Self::collect_bound_variables(pattern)?);
            }
            Algebra::Group { pattern, .. } => {
                vars.extend(Self::collect_bound_variables(pattern)?);
            }
            _ => {}
        }

        Ok(vars)
    }

    /// Validate aggregates
    fn validate_aggregates(
        &self,
        algebra: &Algebra,
        errors: &mut Vec<ValidationError>,
    ) -> Result<()> {
        if let Algebra::Group {
            variables: by,
            aggregates,
            pattern,
        } = algebra
        {
            let bound_vars = Self::collect_bound_variables(pattern)?;

            for (result_var, agg) in aggregates {
                // Check if aggregate uses appropriate variables
                match agg {
                    Aggregate::Count { expr, .. } => {
                        if let Some(expr) = expr {
                            Self::validate_expression_variables(expr, &bound_vars, errors)?;
                        }
                    }
                    Aggregate::Sum { expr, .. }
                    | Aggregate::Avg { expr, .. }
                    | Aggregate::Min { expr, .. }
                    | Aggregate::Max { expr, .. } => {
                        Self::validate_expression_variables(expr, &bound_vars, errors)?;
                    }
                    Aggregate::GroupConcat { expr, .. } => {
                        Self::validate_expression_variables(expr, &bound_vars, errors)?;
                    }
                    Aggregate::Sample { expr, .. } => {
                        Self::validate_expression_variables(expr, &bound_vars, errors)?;
                    }
                }

                // Check if result variable conflicts with group by expression aliases
                for group_cond in by {
                    if let Some(alias) = &group_cond.alias {
                        if alias == result_var {
                            errors.push(ValidationError {
                                error_type: ValidationErrorType::InvalidAggregate,
                                message: format!(
                                    "Aggregate result variable ?{} conflicts with GROUP BY alias",
                                    result_var.as_str()
                                ),
                                location: "GROUP BY clause".to_string(),
                                suggestion: Some(
                                    "Use a different variable name for the aggregate result"
                                        .to_string(),
                                ),
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Validate aggregate expression
    fn validate_aggregate_expression(
        &self,
        agg: &Aggregate,
        bound_vars: &HashSet<Variable>,
        errors: &mut Vec<ValidationError>,
    ) -> Result<()> {
        match agg {
            Aggregate::Count { expr, .. } => {
                if let Some(expr) = expr {
                    Self::validate_expression_variables(expr, bound_vars, errors)?;
                }
            }
            Aggregate::Sum { expr, .. }
            | Aggregate::Avg { expr, .. }
            | Aggregate::Min { expr, .. }
            | Aggregate::Max { expr, .. } => {
                Self::validate_expression_variables(expr, bound_vars, errors)?;
            }
            Aggregate::GroupConcat { expr, .. } => {
                Self::validate_expression_variables(expr, bound_vars, errors)?;
            }
            Aggregate::Sample { expr, .. } => {
                Self::validate_expression_variables(expr, bound_vars, errors)?;
            }
        }
        Ok(())
    }

    /// Validate expression variables
    fn validate_expression_variables(
        expr: &Expression,
        bound_vars: &HashSet<Variable>,
        errors: &mut Vec<ValidationError>,
    ) -> Result<()> {
        match expr {
            Expression::Variable(v) => {
                if !bound_vars.contains(v) {
                    errors.push(ValidationError {
                        error_type: ValidationErrorType::UnboundVariable,
                        message: format!(
                            "Variable ?{} used in expression is not bound",
                            v.as_str()
                        ),
                        location: "Expression".to_string(),
                        suggestion: Some(format!("Bind ?{} in a triple pattern", v.as_str())),
                    });
                }
            }
            Expression::Binary { left, right, op: _ } => {
                Self::validate_expression_variables(left, bound_vars, errors)?;
                Self::validate_expression_variables(right, bound_vars, errors)?;
            }
            Expression::Unary { operand, op: _ } => {
                Self::validate_expression_variables(operand, bound_vars, errors)?;
            }
            Expression::Function { args, .. } => {
                for arg in args {
                    Self::validate_expression_variables(arg, bound_vars, errors)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Check for cartesian products
    fn check_cartesian_products(
        algebra: &Algebra,
        warnings: &mut Vec<ValidationWarning>,
    ) -> Result<()> {
        if let Algebra::Join { left, right } = algebra {
            let left_vars = Self::collect_bound_variables(left)?;
            let right_vars = Self::collect_bound_variables(right)?;

            // Check if joins share any variables
            let shared_vars: HashSet<_> = left_vars.intersection(&right_vars).collect();

            if shared_vars.is_empty() {
                warnings.push(ValidationWarning {
                    warning_type: ValidationWarningType::CartesianProduct,
                    message: "Detected potential cartesian product: join without shared variables".to_string(),
                    location: "JOIN".to_string(),
                    suggestion: Some("Ensure the join patterns share at least one variable to avoid expensive cartesian products".to_string()),
                    severity: 8,
                });
            }

            // Recursively check nested joins
            Self::check_cartesian_products(left, warnings)?;
            Self::check_cartesian_products(right, warnings)?;
        }

        Ok(())
    }

    /// Validate filters
    fn validate_filters(
        algebra: &Algebra,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationWarning>,
    ) -> Result<()> {
        if let Algebra::Filter { pattern, condition } = algebra {
            let bound_vars = Self::collect_bound_variables(pattern)?;

            // Validate filter expression uses bound variables
            Self::validate_expression_variables(condition, &bound_vars, errors)?;

            // Check filter complexity
            let complexity = Self::calculate_expression_complexity(condition);
            if complexity > 10 {
                warnings.push(ValidationWarning {
                    warning_type: ValidationWarningType::ComplexFilter,
                    message: format!("Filter expression has high complexity ({})", complexity),
                    location: "FILTER clause".to_string(),
                    suggestion: Some(
                        "Consider simplifying the filter or breaking it into multiple filters"
                            .to_string(),
                    ),
                    severity: 5,
                });
            }

            // Recursively validate nested patterns
            Self::validate_filters(pattern, errors, warnings)?;
        }

        Ok(())
    }

    /// Calculate query complexity
    pub fn calculate_complexity(algebra: &Algebra) -> Result<usize> {
        let mut complexity = 0;

        match algebra {
            Algebra::Bgp(patterns) => {
                complexity += patterns.len();
            }
            Algebra::Join { left, right } => {
                complexity += 2; // Join operation cost
                complexity += Self::calculate_complexity(left)?;
                complexity += Self::calculate_complexity(right)?;
            }
            Algebra::LeftJoin { left, right, .. } => {
                complexity += 3; // Optional join is more expensive
                complexity += Self::calculate_complexity(left)?;
                complexity += Self::calculate_complexity(right)?;
            }
            Algebra::Union { left, right } => {
                complexity += 2;
                complexity += Self::calculate_complexity(left)?;
                complexity += Self::calculate_complexity(right)?;
            }
            Algebra::Filter {
                pattern,
                condition: expr,
            } => {
                complexity += 1;
                complexity += Self::calculate_expression_complexity(expr);
                complexity += Self::calculate_complexity(pattern)?;
            }
            Algebra::Group {
                pattern,
                variables: by,
                aggregates,
            } => {
                complexity += 5; // Grouping is expensive
                complexity += by.len();
                complexity += aggregates.len() * 2;
                complexity += Self::calculate_complexity(pattern)?;
            }
            Algebra::OrderBy { pattern, .. } => {
                complexity += 3; // Sorting is expensive
                complexity += Self::calculate_complexity(pattern)?;
            }
            _ => {
                complexity += 1;
            }
        }

        Ok(complexity)
    }

    /// Calculate expression complexity
    fn calculate_expression_complexity(expr: &Expression) -> usize {
        match expr {
            Expression::Binary { left, right, .. } => {
                1 + Self::calculate_expression_complexity(left)
                    + Self::calculate_expression_complexity(right)
            }
            Expression::Unary { operand, .. } => 1 + Self::calculate_expression_complexity(operand),
            Expression::Function { args, .. } => {
                2 + args
                    .iter()
                    .map(Self::calculate_expression_complexity)
                    .sum::<usize>()
            }
            _ => 1,
        }
    }

    /// Check performance issues
    fn check_performance_issues(
        &self,
        algebra: &Algebra,
        warnings: &mut Vec<ValidationWarning>,
    ) -> Result<()> {
        // Check for unbounded queries
        if Self::is_unbounded(algebra)? {
            warnings.push(ValidationWarning {
                warning_type: ValidationWarningType::UnboundedQuery,
                message: "Query has no LIMIT clause and may return very large result sets"
                    .to_string(),
                location: "Overall query".to_string(),
                suggestion: Some(
                    "Add a LIMIT clause to prevent excessive memory usage".to_string(),
                ),
                severity: 6,
            });
        }

        // Check triple pattern count
        let pattern_count = Self::count_triple_patterns(algebra)?;
        if pattern_count > self.config.max_triple_patterns {
            warnings.push(ValidationWarning {
                warning_type: ValidationWarningType::Performance,
                message: format!(
                    "Query has {} triple patterns (max recommended: {})",
                    pattern_count, self.config.max_triple_patterns
                ),
                location: "WHERE clause".to_string(),
                suggestion: Some("Consider breaking the query into smaller subqueries".to_string()),
                severity: 7,
            });
        }

        Ok(())
    }

    /// Check if query is unbounded
    fn is_unbounded(algebra: &Algebra) -> Result<bool> {
        match algebra {
            Algebra::Slice { .. } => Ok(false),
            Algebra::Project { pattern, .. } => Self::is_unbounded(pattern),
            Algebra::OrderBy { pattern, .. } => Self::is_unbounded(pattern),
            Algebra::Group { pattern, .. } => Self::is_unbounded(pattern),
            Algebra::Filter { pattern, .. } => Self::is_unbounded(pattern),
            _ => Ok(true),
        }
    }

    /// Count triple patterns
    fn count_triple_patterns(algebra: &Algebra) -> Result<usize> {
        match algebra {
            Algebra::Bgp(patterns) => Ok(patterns.len()),
            Algebra::Join { left, right }
            | Algebra::LeftJoin { left, right, .. }
            | Algebra::Union { left, right } => {
                Ok(Self::count_triple_patterns(left)? + Self::count_triple_patterns(right)?)
            }
            Algebra::Filter { pattern, .. } => Self::count_triple_patterns(pattern),
            Algebra::Project { pattern, .. } => Self::count_triple_patterns(pattern),
            _ => Ok(0),
        }
    }

    /// Check security issues
    fn check_security_issues(
        &self,
        _algebra: &Algebra,
        _warnings: &mut Vec<ValidationWarning>,
    ) -> Result<()> {
        // Placeholder for security checks
        // Could include checks for:
        // - Suspicious patterns that might indicate injection
        // - Queries that access sensitive predicates
        // - Potential DoS patterns
        Ok(())
    }

    /// Check type consistency
    fn check_type_consistency(
        &self,
        _algebra: &Algebra,
        _warnings: &mut Vec<ValidationWarning>,
    ) -> Result<()> {
        // Placeholder for type consistency checks
        // Could include checks for:
        // - Numeric operations on non-numeric variables
        // - String operations on non-string variables
        // - Comparison of incompatible types
        Ok(())
    }

    /// Get validation statistics
    pub fn statistics(&self) -> &ValidationStatistics {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.stats = ValidationStatistics::default();
    }
}

impl Default for QueryValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Literal, Term, TriplePattern};
    use oxirs_core::model::NamedNode;

    fn create_term(s: &str) -> Term {
        Term::Iri(NamedNode::new(s).unwrap())
    }

    fn create_var(name: &str) -> Variable {
        Variable::new(name).unwrap()
    }

    #[test]
    fn test_validator_creation() {
        let validator = QueryValidator::new();
        assert!(validator.config.strict_compliance);
    }

    #[test]
    fn test_unbound_variable_detection() {
        let mut validator = QueryValidator::new();

        // Create a query with unbound variable in projection
        let pattern = Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(create_var("s")),
            predicate: create_term("http://example.org/p"),
            object: Term::Variable(create_var("o")),
        }]);

        let query = Algebra::Project {
            variables: vec![create_var("s"), create_var("unbound")],
            pattern: Box::new(pattern),
        };

        let result = validator.validate(&query).unwrap();
        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(
            result.errors[0].error_type,
            ValidationErrorType::UnboundVariable
        );
    }

    #[test]
    fn test_valid_query() {
        let mut validator = QueryValidator::new();

        let pattern = Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(create_var("s")),
            predicate: create_term("http://example.org/p"),
            object: Term::Variable(create_var("o")),
        }]);

        let query = Algebra::Project {
            variables: vec![create_var("s"), create_var("o")],
            pattern: Box::new(pattern),
        };

        let result = validator.validate(&query).unwrap();
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_cartesian_product_warning() {
        let mut validator = QueryValidator::new();

        // Create two BGPs with no shared variables (cartesian product)
        let left = Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(create_var("s1")),
            predicate: create_term("http://example.org/p1"),
            object: Term::Variable(create_var("o1")),
        }]);

        let right = Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(create_var("s2")),
            predicate: create_term("http://example.org/p2"),
            object: Term::Variable(create_var("o2")),
        }]);

        let query = Algebra::Join {
            left: Box::new(left),
            right: Box::new(right),
        };

        let result = validator.validate(&query).unwrap();
        assert!(result.is_valid); // No errors, just warnings
        assert!(!result.warnings.is_empty());
        assert_eq!(
            result.warnings[0].warning_type,
            ValidationWarningType::CartesianProduct
        );
    }

    #[test]
    fn test_complexity_calculation() {
        let _validator = QueryValidator::new();

        // Simple BGP
        let simple = Algebra::Bgp(vec![
            TriplePattern {
                subject: Term::Variable(create_var("s")),
                predicate: create_term("http://example.org/p"),
                object: Term::Variable(create_var("o")),
            },
            TriplePattern {
                subject: Term::Variable(create_var("o")),
                predicate: create_term("http://example.org/p2"),
                object: Term::Literal(Literal::string("test")),
            },
        ]);

        let complexity = QueryValidator::calculate_complexity(&simple).unwrap();
        assert_eq!(complexity, 2);
    }

    #[test]
    fn test_validation_statistics() {
        let mut validator = QueryValidator::new();

        let pattern = Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(create_var("s")),
            predicate: create_term("http://example.org/p"),
            object: Term::Variable(create_var("o")),
        }]);

        let query = Algebra::Project {
            variables: vec![create_var("s")],
            pattern: Box::new(pattern),
        };

        validator.validate(&query).unwrap();

        let stats = validator.statistics();
        assert_eq!(stats.total_validated, 1);
    }

    #[test]
    fn test_custom_config() {
        let config = ValidationConfig {
            max_complexity: 50,
            warn_cartesian_products: false,
            ..Default::default()
        };

        let validator = QueryValidator::with_config(config);
        assert_eq!(validator.config.max_complexity, 50);
        assert!(!validator.config.warn_cartesian_products);
    }

    #[test]
    fn test_aggregate_validation() {
        let mut validator = QueryValidator::new();

        let pattern = Algebra::Bgp(vec![TriplePattern {
            subject: Term::Variable(create_var("s")),
            predicate: create_term("http://example.org/p"),
            object: Term::Variable(create_var("o")),
        }]);

        let query = Algebra::Group {
            pattern: Box::new(pattern),
            variables: vec![],
            aggregates: vec![(
                create_var("count"),
                Aggregate::Count {
                    distinct: false,
                    expr: None,
                },
            )],
        };

        let result = validator.validate(&query).unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_performance_warnings() {
        let mut validator = QueryValidator::with_config(ValidationConfig {
            max_triple_patterns: 2,
            ..Default::default()
        });

        // Create a query with 3 triple patterns
        let pattern = Algebra::Bgp(vec![
            TriplePattern {
                subject: Term::Variable(create_var("s")),
                predicate: create_term("http://example.org/p1"),
                object: Term::Variable(create_var("o")),
            },
            TriplePattern {
                subject: Term::Variable(create_var("o")),
                predicate: create_term("http://example.org/p2"),
                object: Term::Variable(create_var("o2")),
            },
            TriplePattern {
                subject: Term::Variable(create_var("o2")),
                predicate: create_term("http://example.org/p3"),
                object: Term::Literal(Literal::string("test")),
            },
        ]);

        let result = validator.validate(&pattern).unwrap();
        assert!(result.is_valid);
        assert!(!result.warnings.is_empty());
    }
}
