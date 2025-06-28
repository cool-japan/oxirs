//! Query execution planning
//!
//! This module is responsible for converting SPARQL algebra into
//! optimized execution plans.

use crate::model::*;
use crate::query::algebra;
use crate::query::algebra::{Expression, OrderExpression, Query, QueryForm, SelectVariables};
use crate::OxirsError;

/// Convert algebra TriplePattern to model TriplePattern
/// This function bridges the gap between algebra and model pattern representations
pub fn convert_triple_pattern(pattern: &algebra::TriplePattern) -> TriplePattern {
    let subject = match &pattern.subject {
        algebra::TermPattern::NamedNode(nn) => Some(SubjectPattern::NamedNode(nn.clone())),
        algebra::TermPattern::BlankNode(bn) => Some(SubjectPattern::BlankNode(bn.clone())),
        algebra::TermPattern::Variable(v) => Some(SubjectPattern::Variable(v.clone())),
        algebra::TermPattern::Literal(_) => None, // Literals can't be subjects in RDF
    };

    let predicate = match &pattern.predicate {
        algebra::TermPattern::NamedNode(nn) => Some(PredicatePattern::NamedNode(nn.clone())),
        algebra::TermPattern::Variable(v) => Some(PredicatePattern::Variable(v.clone())),
        _ => None, // Only named nodes and variables can be predicates in RDF
    };

    let object = match &pattern.object {
        algebra::TermPattern::NamedNode(nn) => Some(ObjectPattern::NamedNode(nn.clone())),
        algebra::TermPattern::BlankNode(bn) => Some(ObjectPattern::BlankNode(bn.clone())),
        algebra::TermPattern::Literal(lit) => Some(ObjectPattern::Literal(lit.clone())),
        algebra::TermPattern::Variable(v) => Some(ObjectPattern::Variable(v.clone())),
    };

    TriplePattern::new(subject, predicate, object)
}

/// Convert model TriplePattern to algebra TriplePattern
/// This function provides the reverse conversion for compatibility
pub fn convert_to_algebra_pattern(
    pattern: &TriplePattern,
) -> Result<algebra::TriplePattern, OxirsError> {
    let subject = match pattern.subject() {
        Some(SubjectPattern::NamedNode(nn)) => algebra::TermPattern::NamedNode(nn.clone()),
        Some(SubjectPattern::BlankNode(bn)) => algebra::TermPattern::BlankNode(bn.clone()),
        Some(SubjectPattern::Variable(v)) => algebra::TermPattern::Variable(v.clone()),
        None => {
            return Err(OxirsError::Query(
                "Subject pattern is required in algebra representation".to_string(),
            ))
        }
    };

    let predicate = match pattern.predicate() {
        Some(PredicatePattern::NamedNode(nn)) => algebra::TermPattern::NamedNode(nn.clone()),
        Some(PredicatePattern::Variable(v)) => algebra::TermPattern::Variable(v.clone()),
        None => {
            return Err(OxirsError::Query(
                "Predicate pattern is required in algebra representation".to_string(),
            ))
        }
    };

    let object = match pattern.object() {
        Some(ObjectPattern::NamedNode(nn)) => algebra::TermPattern::NamedNode(nn.clone()),
        Some(ObjectPattern::BlankNode(bn)) => algebra::TermPattern::BlankNode(bn.clone()),
        Some(ObjectPattern::Literal(lit)) => algebra::TermPattern::Literal(lit.clone()),
        Some(ObjectPattern::Variable(v)) => algebra::TermPattern::Variable(v.clone()),
        None => {
            return Err(OxirsError::Query(
                "Object pattern is required in algebra representation".to_string(),
            ))
        }
    };

    Ok(algebra::TriplePattern {
        subject,
        predicate,
        object,
    })
}

/// A query execution plan
#[derive(Debug, Clone)]
pub enum ExecutionPlan {
    /// Scan all triples matching a pattern
    TripleScan {
        pattern: crate::model::pattern::TriplePattern,
    },
    /// Join two sub-plans
    HashJoin {
        left: Box<ExecutionPlan>,
        right: Box<ExecutionPlan>,
        join_vars: Vec<Variable>,
    },
    /// Filter results
    Filter {
        input: Box<ExecutionPlan>,
        condition: Expression,
    },
    /// Project specific variables
    Project {
        input: Box<ExecutionPlan>,
        vars: Vec<Variable>,
    },
    /// Sort results
    Sort {
        input: Box<ExecutionPlan>,
        order_by: Vec<OrderExpression>,
    },
    /// Limit results
    Limit {
        input: Box<ExecutionPlan>,
        limit: usize,
        offset: usize,
    },
    /// Union of two plans
    Union {
        left: Box<ExecutionPlan>,
        right: Box<ExecutionPlan>,
    },
    /// Distinct results
    Distinct { input: Box<ExecutionPlan> },
}

/// Query planner that converts algebra to execution plans
pub struct QueryPlanner;

impl QueryPlanner {
    /// Creates a new query planner
    pub fn new() -> Self {
        QueryPlanner
    }

    /// Plans a query for execution
    pub fn plan_query(&self, query: &Query) -> Result<ExecutionPlan, OxirsError> {
        match &query.form {
            QueryForm::Select {
                where_clause,
                variables,
                distinct,
                order_by,
                limit,
                offset,
                ..
            } => {
                let mut plan = self.plan_graph_pattern(where_clause)?;

                // Add projection if needed
                if let SelectVariables::Specific(vars) = variables {
                    plan = ExecutionPlan::Project {
                        input: Box::new(plan),
                        vars: vars.clone(),
                    };
                }

                // Add distinct if needed
                if *distinct {
                    plan = ExecutionPlan::Distinct {
                        input: Box::new(plan),
                    };
                }

                // Add ordering if needed
                if !order_by.is_empty() {
                    plan = ExecutionPlan::Sort {
                        input: Box::new(plan),
                        order_by: order_by.clone(),
                    };
                }

                // Add limit/offset if needed
                if let Some(limit_val) = limit {
                    plan = ExecutionPlan::Limit {
                        input: Box::new(plan),
                        limit: *limit_val,
                        offset: *offset,
                    };
                } else if *offset > 0 {
                    plan = ExecutionPlan::Limit {
                        input: Box::new(plan),
                        limit: usize::MAX,
                        offset: *offset,
                    };
                }

                Ok(plan)
            }
            _ => Err(OxirsError::Query(
                "Only SELECT queries are currently supported".to_string(),
            )),
        }
    }

    /// Plans a graph pattern
    fn plan_graph_pattern(
        &self,
        pattern: &algebra::GraphPattern,
    ) -> Result<ExecutionPlan, OxirsError> {
        match pattern {
            algebra::GraphPattern::Bgp(patterns) => {
                if patterns.is_empty() {
                    return Err(OxirsError::Query("Empty basic graph pattern".to_string()));
                }

                // Start with the first pattern
                let mut plan = ExecutionPlan::TripleScan {
                    pattern: convert_triple_pattern(&patterns[0]),
                };

                // Join with remaining patterns
                for pattern in &patterns[1..] {
                    let right_plan = ExecutionPlan::TripleScan {
                        pattern: convert_triple_pattern(pattern),
                    };

                    // Find join variables
                    let join_vars = self.find_join_variables(&plan, &right_plan);

                    plan = ExecutionPlan::HashJoin {
                        left: Box::new(plan),
                        right: Box::new(right_plan),
                        join_vars,
                    };
                }

                Ok(plan)
            }
            algebra::GraphPattern::Filter { expr, inner } => {
                let inner_plan = self.plan_graph_pattern(inner)?;
                Ok(ExecutionPlan::Filter {
                    input: Box::new(inner_plan),
                    condition: expr.clone(),
                })
            }
            algebra::GraphPattern::Union(left, right) => {
                let left_plan = self.plan_graph_pattern(left)?;
                let right_plan = self.plan_graph_pattern(right)?;
                Ok(ExecutionPlan::Union {
                    left: Box::new(left_plan),
                    right: Box::new(right_plan),
                })
            }
            _ => Err(OxirsError::Query(
                "Graph pattern not yet supported".to_string(),
            )),
        }
    }

    /// Find variables that appear in both plans (for joins)
    fn find_join_variables(&self, _left: &ExecutionPlan, _right: &ExecutionPlan) -> Vec<Variable> {
        // Placeholder - would analyze both plans to find common variables
        Vec::new()
    }
}

impl Default for QueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}
