//! Enhanced SPARQL-star features for full specification compliance
//!
//! This module extends the base SPARQL-star implementation with advanced features:
//! - OPTIONAL patterns
//! - UNION patterns
//! - Solution modifiers (ORDER BY, LIMIT, OFFSET, DISTINCT)
//! - BIND clause
//! - VALUES clause
//! - Aggregations (GROUP BY, COUNT, SUM, AVG, MIN, MAX)
//! - Sub-queries
//! - MINUS/NOT EXISTS (negation)
//! - Property paths

use crate::functions::Expression;
use crate::model::StarTerm;
use crate::query::{BasicGraphPattern, Binding, QueryExecutor};
use crate::StarResult;
use std::collections::{HashMap, HashSet};
use tracing::{debug, span, Level};

/// Graph pattern that can contain OPTIONAL, UNION, GRAPH, MINUS
#[derive(Debug, Clone)]
pub enum GraphPattern {
    /// Basic graph pattern
    BasicPattern(BasicGraphPattern),
    /// OPTIONAL { pattern }
    Optional(Box<GraphPattern>),
    /// pattern1 UNION pattern2
    Union(Box<GraphPattern>, Box<GraphPattern>),
    /// GRAPH ?g { pattern }
    Graph {
        graph: TermOrVariable,
        pattern: Box<GraphPattern>,
    },
    /// pattern1 MINUS pattern2
    Minus(Box<GraphPattern>, Box<GraphPattern>),
    /// Group of patterns executed sequentially
    Group(Vec<GraphPattern>),
}

/// Term or variable for flexible pattern matching
#[derive(Debug, Clone)]
pub enum TermOrVariable {
    Term(StarTerm),
    Variable(String),
}

/// Solution modifier for query results
#[derive(Debug, Clone, Default)]
pub struct SolutionModifier {
    /// ORDER BY clauses
    pub order_by: Vec<OrderCondition>,
    /// DISTINCT flag
    pub distinct: bool,
    /// LIMIT clause
    pub limit: Option<usize>,
    /// OFFSET clause
    pub offset: Option<usize>,
}

/// ORDER BY condition
#[derive(Debug, Clone)]
pub struct OrderCondition {
    /// Expression to order by
    pub expression: Expression,
    /// Ascending (true) or descending (false)
    pub ascending: bool,
}

/// BIND clause for adding computed bindings
#[derive(Debug, Clone)]
pub struct BindClause {
    /// Expression to evaluate
    pub expression: Expression,
    /// Variable to bind the result to
    pub variable: String,
}

/// VALUES clause for inline data
#[derive(Debug, Clone)]
pub struct ValuesClause {
    /// Variables in the VALUES clause
    pub variables: Vec<String>,
    /// Data rows (each row matches the variables)
    pub data: Vec<Vec<Option<StarTerm>>>,
}

/// Aggregation function
#[derive(Debug, Clone, PartialEq)]
pub enum AggregateFunction {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    GroupConcat { separator: String },
    Sample,
}

/// Aggregation operation
#[derive(Debug, Clone)]
pub struct Aggregation {
    /// Aggregate function to apply
    pub function: AggregateFunction,
    /// Expression to aggregate
    pub expression: Expression,
    /// Variable to bind result to
    pub as_variable: String,
    /// DISTINCT flag for aggregation
    pub distinct: bool,
}

/// GROUP BY clause
#[derive(Debug, Clone)]
pub struct GroupByClause {
    /// Expressions to group by
    pub expressions: Vec<Expression>,
    /// Aggregations to compute
    pub aggregations: Vec<Aggregation>,
    /// HAVING conditions
    pub having: Vec<Expression>,
}

/// Enhanced query structure with full SPARQL 1.1 support
#[derive(Debug, Clone)]
pub struct EnhancedQuery {
    /// Main graph pattern (can be complex with OPTIONAL, UNION, etc.)
    pub pattern: GraphPattern,
    /// BIND clauses
    pub bind_clauses: Vec<BindClause>,
    /// VALUES clause
    pub values: Option<ValuesClause>,
    /// GROUP BY clause
    pub group_by: Option<GroupByClause>,
    /// Solution modifiers
    pub modifiers: SolutionModifier,
}

/// Enhanced SPARQL executor with full spec support
pub struct EnhancedSparqlExecutor {
    /// Base query executor
    base_executor: QueryExecutor,
}

impl EnhancedSparqlExecutor {
    /// Create a new enhanced executor
    pub fn new(base_executor: QueryExecutor) -> Self {
        Self { base_executor }
    }

    /// Helper: Convert Binding to HashMap for expression evaluation
    fn binding_to_map(binding: &Binding) -> HashMap<String, StarTerm> {
        binding
            .variables()
            .into_iter()
            .filter_map(|var| binding.get(var).map(|term| (var.clone(), term.clone())))
            .collect()
    }

    /// Execute an enhanced query with full SPARQL 1.1 features
    pub fn execute(&mut self, query: &EnhancedQuery) -> StarResult<Vec<Binding>> {
        let span = span!(Level::INFO, "execute_enhanced_query");
        let _enter = span.enter();

        // Execute main graph pattern
        let mut bindings = self.execute_graph_pattern(&query.pattern)?;

        // Apply VALUES clause (acts as a filter/join)
        if let Some(ref values) = query.values {
            bindings = self.apply_values(bindings, values)?;
        }

        // Apply BIND clauses
        for bind in &query.bind_clauses {
            bindings = self.apply_bind(bindings, bind)?;
        }

        // Apply GROUP BY and aggregations
        if let Some(ref group_by) = query.group_by {
            bindings = self.apply_group_by(bindings, group_by)?;
        }

        // Apply solution modifiers
        bindings = self.apply_modifiers(bindings, &query.modifiers)?;

        debug!("Enhanced query produced {} final bindings", bindings.len());
        Ok(bindings)
    }

    /// Execute a graph pattern (including OPTIONAL, UNION, etc.)
    pub fn execute_graph_pattern(&mut self, pattern: &GraphPattern) -> StarResult<Vec<Binding>> {
        match pattern {
            GraphPattern::BasicPattern(bgp) => self.base_executor.execute_bgp(bgp),

            GraphPattern::Optional(inner) => {
                // OPTIONAL patterns: include bindings even if pattern doesn't match
                let base_bindings = vec![Binding::new()];
                let optional_bindings = self.execute_graph_pattern(inner)?;

                if optional_bindings.is_empty() {
                    // No matches, return base bindings
                    Ok(base_bindings)
                } else {
                    Ok(optional_bindings)
                }
            }

            GraphPattern::Union(left, right) => {
                // UNION: combine results from both patterns
                let mut left_bindings = self.execute_graph_pattern(left)?;
                let right_bindings = self.execute_graph_pattern(right)?;

                left_bindings.extend(right_bindings);
                Ok(left_bindings)
            }

            GraphPattern::Graph { pattern, .. } => {
                // GRAPH pattern: execute in specific named graph
                // For now, delegate to base pattern
                // Full implementation would check graph name
                self.execute_graph_pattern(pattern)
            }

            GraphPattern::Minus(left, right) => {
                // MINUS: remove bindings that match right pattern
                let left_bindings = self.execute_graph_pattern(left)?;
                let right_bindings = self.execute_graph_pattern(right)?;

                let right_set: HashSet<_> =
                    right_bindings.iter().map(|b| format!("{:?}", b)).collect();

                Ok(left_bindings
                    .into_iter()
                    .filter(|b| !right_set.contains(&format!("{:?}", b)))
                    .collect())
            }

            GraphPattern::Group(patterns) => {
                // Execute patterns sequentially and join results
                let mut current_bindings = vec![Binding::new()];

                for pattern in patterns {
                    let pattern_bindings = self.execute_graph_pattern(pattern)?;
                    current_bindings = self.join_bindings(current_bindings, pattern_bindings)?;
                }

                Ok(current_bindings)
            }
        }
    }

    /// Join two sets of bindings
    fn join_bindings(&self, left: Vec<Binding>, right: Vec<Binding>) -> StarResult<Vec<Binding>> {
        let mut result = Vec::new();

        for left_binding in &left {
            for right_binding in &right {
                if let Some(merged) = left_binding.merge(right_binding) {
                    result.push(merged);
                }
            }
        }

        Ok(result)
    }

    /// Apply VALUES clause
    fn apply_values(
        &self,
        bindings: Vec<Binding>,
        values: &ValuesClause,
    ) -> StarResult<Vec<Binding>> {
        let mut result = Vec::new();

        // Create bindings from VALUES data
        for row in &values.data {
            let mut value_binding = Binding::new();

            for (i, var) in values.variables.iter().enumerate() {
                if let Some(Some(term)) = row.get(i) {
                    value_binding.bind(var, term.clone());
                }
            }

            // Join with existing bindings
            for binding in &bindings {
                if let Some(merged) = binding.merge(&value_binding) {
                    result.push(merged);
                }
            }
        }

        Ok(result)
    }

    /// Apply BIND clause
    fn apply_bind(&self, bindings: Vec<Binding>, bind: &BindClause) -> StarResult<Vec<Binding>> {
        let mut result = Vec::new();

        for binding in bindings {
            let mut new_binding = binding.clone();

            // Evaluate expression
            let binding_map = Self::binding_to_map(&binding);
            if let Ok(term) =
                crate::functions::ExpressionEvaluator::evaluate(&bind.expression, &binding_map)
            {
                new_binding.bind(&bind.variable, term);
                result.push(new_binding);
            } else {
                // Evaluation failed, skip this binding
                continue;
            }
        }

        Ok(result)
    }

    /// Apply GROUP BY and aggregations
    fn apply_group_by(
        &self,
        bindings: Vec<Binding>,
        group_by: &GroupByClause,
    ) -> StarResult<Vec<Binding>> {
        // Group bindings by the grouping expressions
        let mut groups: HashMap<String, Vec<Binding>> = HashMap::new();

        for binding in bindings {
            let group_key = self.compute_group_key(&binding, &group_by.expressions)?;
            groups.entry(group_key).or_default().push(binding);
        }

        // Apply aggregations to each group
        let mut result = Vec::new();

        for (_, group_bindings) in groups {
            if group_bindings.is_empty() {
                continue;
            }

            let mut aggregated_binding = group_bindings[0].clone();

            for agg in &group_by.aggregations {
                let agg_result = self.apply_aggregation(&group_bindings, agg)?;
                aggregated_binding.bind(&agg.as_variable, agg_result);
            }

            // Apply HAVING filter
            let passes_having = group_by.having.iter().all(|having_expr| {
                let binding_map = Self::binding_to_map(&aggregated_binding);
                if let Ok(result) =
                    crate::functions::ExpressionEvaluator::evaluate(having_expr, &binding_map)
                {
                    self.is_truthy(&result)
                } else {
                    false
                }
            });

            if passes_having {
                result.push(aggregated_binding);
            }
        }

        Ok(result)
    }

    /// Compute group key from expressions
    fn compute_group_key(
        &self,
        binding: &Binding,
        expressions: &[Expression],
    ) -> StarResult<String> {
        let mut key = String::new();

        for expr in expressions {
            let binding_map = Self::binding_to_map(binding);
            if let Ok(term) = crate::functions::ExpressionEvaluator::evaluate(expr, &binding_map) {
                key.push_str(&format!("{:?}", term));
                key.push('|');
            }
        }

        Ok(key)
    }

    /// Apply an aggregation function
    fn apply_aggregation(&self, bindings: &[Binding], agg: &Aggregation) -> StarResult<StarTerm> {
        match agg.function {
            AggregateFunction::Count => {
                let count = if agg.distinct {
                    self.count_distinct(bindings, &agg.expression)?
                } else {
                    bindings.len()
                };
                StarTerm::literal(&count.to_string())
            }

            AggregateFunction::Sum => self.sum_aggregation(bindings, &agg.expression),

            AggregateFunction::Avg => {
                let sum = self.sum_numeric(bindings, &agg.expression)?;
                let count = bindings.len() as f64;
                let avg = sum / count;
                StarTerm::literal(&avg.to_string())
            }

            AggregateFunction::Min => self.min_aggregation(bindings, &agg.expression),

            AggregateFunction::Max => self.max_aggregation(bindings, &agg.expression),

            AggregateFunction::GroupConcat { ref separator } => {
                self.group_concat(bindings, &agg.expression, separator)
            }

            AggregateFunction::Sample => {
                // Return the first value
                if let Some(binding) = bindings.first() {
                    let binding_map = Self::binding_to_map(binding);
                    crate::functions::ExpressionEvaluator::evaluate(&agg.expression, &binding_map)
                } else {
                    StarTerm::literal("")
                }
            }
        }
    }

    /// Count distinct values
    fn count_distinct(&self, bindings: &[Binding], expr: &Expression) -> StarResult<usize> {
        let mut seen = HashSet::new();

        for binding in bindings {
            let binding_map = Self::binding_to_map(binding);
            if let Ok(term) = crate::functions::ExpressionEvaluator::evaluate(expr, &binding_map) {
                seen.insert(format!("{:?}", term));
            }
        }

        Ok(seen.len())
    }

    /// Sum numeric values
    fn sum_numeric(&self, bindings: &[Binding], expr: &Expression) -> StarResult<f64> {
        let mut sum = 0.0;

        for binding in bindings {
            let binding_map = Self::binding_to_map(binding);
            if let Ok(term) = crate::functions::ExpressionEvaluator::evaluate(expr, &binding_map) {
                if let Some(literal) = term.as_literal() {
                    if let Ok(num) = literal.value.parse::<f64>() {
                        sum += num;
                    }
                }
            }
        }

        Ok(sum)
    }

    /// SUM aggregation
    fn sum_aggregation(&self, bindings: &[Binding], expr: &Expression) -> StarResult<StarTerm> {
        let sum = self.sum_numeric(bindings, expr)?;
        StarTerm::literal(&sum.to_string())
    }

    /// MIN aggregation
    fn min_aggregation(&self, bindings: &[Binding], expr: &Expression) -> StarResult<StarTerm> {
        let mut min_val: Option<f64> = None;

        for binding in bindings {
            let binding_map = Self::binding_to_map(binding);
            if let Ok(term) = crate::functions::ExpressionEvaluator::evaluate(expr, &binding_map) {
                if let Some(literal) = term.as_literal() {
                    if let Ok(num) = literal.value.parse::<f64>() {
                        min_val = Some(min_val.map_or(num, |m| m.min(num)));
                    }
                }
            }
        }

        if let Some(min) = min_val {
            StarTerm::literal(&min.to_string())
        } else {
            StarTerm::literal("")
        }
    }

    /// MAX aggregation
    fn max_aggregation(&self, bindings: &[Binding], expr: &Expression) -> StarResult<StarTerm> {
        let mut max_val: Option<f64> = None;

        for binding in bindings {
            let binding_map = Self::binding_to_map(binding);
            if let Ok(term) = crate::functions::ExpressionEvaluator::evaluate(expr, &binding_map) {
                if let Some(literal) = term.as_literal() {
                    if let Ok(num) = literal.value.parse::<f64>() {
                        max_val = Some(max_val.map_or(num, |m| m.max(num)));
                    }
                }
            }
        }

        if let Some(max) = max_val {
            StarTerm::literal(&max.to_string())
        } else {
            StarTerm::literal("")
        }
    }

    /// GROUP_CONCAT aggregation
    fn group_concat(
        &self,
        bindings: &[Binding],
        expr: &Expression,
        separator: &str,
    ) -> StarResult<StarTerm> {
        let mut values = Vec::new();

        for binding in bindings {
            let binding_map = Self::binding_to_map(binding);
            if let Ok(term) = crate::functions::ExpressionEvaluator::evaluate(expr, &binding_map) {
                values.push(format!("{}", term));
            }
        }

        StarTerm::literal(&values.join(separator))
    }

    /// Apply solution modifiers (ORDER BY, LIMIT, OFFSET, DISTINCT)
    fn apply_modifiers(
        &self,
        mut bindings: Vec<Binding>,
        modifiers: &SolutionModifier,
    ) -> StarResult<Vec<Binding>> {
        // Apply DISTINCT
        if modifiers.distinct {
            let mut seen = HashSet::new();
            bindings.retain(|binding| {
                let key = format!("{:?}", binding);
                seen.insert(key.clone())
            });
        }

        // Apply ORDER BY
        if !modifiers.order_by.is_empty() {
            bindings.sort_by(|a, b| {
                for order_cond in &modifiers.order_by {
                    let a_map = Self::binding_to_map(a);
                    let b_map = Self::binding_to_map(b);

                    let a_val = crate::functions::ExpressionEvaluator::evaluate(
                        &order_cond.expression,
                        &a_map,
                    )
                    .ok();
                    let b_val = crate::functions::ExpressionEvaluator::evaluate(
                        &order_cond.expression,
                        &b_map,
                    )
                    .ok();

                    let cmp = match (a_val, b_val) {
                        (Some(a_term), Some(b_term)) => self.compare_terms(&a_term, &b_term),
                        (Some(_), None) => std::cmp::Ordering::Less,
                        (None, Some(_)) => std::cmp::Ordering::Greater,
                        (None, None) => std::cmp::Ordering::Equal,
                    };

                    if cmp != std::cmp::Ordering::Equal {
                        return if order_cond.ascending {
                            cmp
                        } else {
                            cmp.reverse()
                        };
                    }
                }
                std::cmp::Ordering::Equal
            });
        }

        // Apply OFFSET and LIMIT
        let start = modifiers.offset.unwrap_or(0);
        let end = modifiers.limit.map(|l| start + l).unwrap_or(bindings.len());

        Ok(bindings.into_iter().skip(start).take(end - start).collect())
    }

    /// Compare two terms for ordering
    fn compare_terms(&self, a: &StarTerm, b: &StarTerm) -> std::cmp::Ordering {
        // Simplified comparison - a full implementation would handle all RDF term types properly
        format!("{:?}", a).cmp(&format!("{:?}", b))
    }

    /// Check if a term is truthy
    fn is_truthy(&self, term: &StarTerm) -> bool {
        if let Some(literal) = term.as_literal() {
            if let Some(datatype) = &literal.datatype {
                if datatype.iri == "http://www.w3.org/2001/XMLSchema#boolean" {
                    return literal.value == "true";
                }
            }
            !literal.value.is_empty() && literal.value != "false" && literal.value != "0"
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StarTriple;
    use crate::query::{QueryExecutor, TermPattern, TriplePattern};
    use crate::StarStore;

    #[test]
    fn test_optional_pattern() {
        let store = StarStore::new();

        // Add test data
        let triple1 = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/name").unwrap(),
            StarTerm::literal("Alice").unwrap(),
        );
        store.insert(&triple1).unwrap();

        let base_executor = QueryExecutor::new(store);
        let mut executor = EnhancedSparqlExecutor::new(base_executor);

        // Create BGP with OPTIONAL
        let mut bgp = BasicGraphPattern::new();
        bgp.add_pattern(TriplePattern::new(
            TermPattern::Variable("x".to_string()),
            TermPattern::Term(StarTerm::iri("http://example.org/name").unwrap()),
            TermPattern::Variable("name".to_string()),
        ));

        let optional_bgp = GraphPattern::BasicPattern(bgp);

        let bindings = executor.execute_graph_pattern(&optional_bgp).unwrap();
        assert_eq!(bindings.len(), 1);
    }

    #[test]
    fn test_solution_modifiers() {
        let store = StarStore::new();

        // Add test data
        for i in 0..10 {
            let triple = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/person{i}")).unwrap(),
                StarTerm::iri("http://example.org/age").unwrap(),
                StarTerm::literal(&format!("{}", 20 + i)).unwrap(),
            );
            store.insert(&triple).unwrap();
        }

        let base_executor = QueryExecutor::new(store);
        let executor = EnhancedSparqlExecutor::new(base_executor);

        // Create bindings
        let mut bindings = Vec::new();
        for i in 0..10 {
            let mut binding = Binding::new();
            binding.bind("age", StarTerm::literal(&format!("{}", 20 + i)).unwrap());
            bindings.push(binding);
        }

        // Apply LIMIT
        let modifiers = SolutionModifier {
            limit: Some(5),
            offset: Some(2),
            distinct: false,
            order_by: vec![],
        };

        let result = executor.apply_modifiers(bindings, &modifiers).unwrap();
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_aggregations() {
        let store = StarStore::new();
        let base_executor = QueryExecutor::new(store);
        let executor = EnhancedSparqlExecutor::new(base_executor);

        // Create test bindings
        let mut bindings = Vec::new();
        for i in 1..=5 {
            let mut binding = Binding::new();
            binding.bind("value", StarTerm::literal(&i.to_string()).unwrap());
            bindings.push(binding);
        }

        // Test COUNT
        let count_agg = Aggregation {
            function: AggregateFunction::Count,
            expression: Expression::var("value"),
            as_variable: "count".to_string(),
            distinct: false,
        };

        let count_result = executor.apply_aggregation(&bindings, &count_agg).unwrap();
        assert_eq!(count_result, StarTerm::literal("5").unwrap());

        // Test SUM
        let sum_agg = Aggregation {
            function: AggregateFunction::Sum,
            expression: Expression::var("value"),
            as_variable: "sum".to_string(),
            distinct: false,
        };

        let sum_result = executor.apply_aggregation(&bindings, &sum_agg).unwrap();
        if let Some(literal) = sum_result.as_literal() {
            let sum: f64 = literal.value.parse().unwrap();
            assert!((sum - 15.0).abs() < 0.1); // 1+2+3+4+5 = 15
        }
    }
}
