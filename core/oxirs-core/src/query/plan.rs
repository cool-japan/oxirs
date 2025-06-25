//! Query execution planning
//! 
//! This module is responsible for converting SPARQL algebra into
//! optimized execution plans.

use crate::query::algebra::*;
use crate::model::*;
use crate::OxirsError;

/// A query execution plan
#[derive(Debug, Clone)]
pub enum ExecutionPlan {
    /// Scan all triples matching a pattern
    TripleScan {
        pattern: TriplePattern,
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
    Distinct {
        input: Box<ExecutionPlan>,
    },
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
            QueryForm::Select { where_clause, variables, distinct, order_by, limit, offset, .. } => {
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
            _ => Err(OxirsError::Query("Only SELECT queries are currently supported".to_string())),
        }
    }
    
    /// Plans a graph pattern
    fn plan_graph_pattern(&self, pattern: &GraphPattern) -> Result<ExecutionPlan, OxirsError> {
        match pattern {
            GraphPattern::Bgp(patterns) => {
                if patterns.is_empty() {
                    return Err(OxirsError::Query("Empty basic graph pattern".to_string()));
                }
                
                // Start with the first pattern
                let mut plan = ExecutionPlan::TripleScan {
                    pattern: patterns[0].clone(),
                };
                
                // Join with remaining patterns
                for pattern in &patterns[1..] {
                    let right_plan = ExecutionPlan::TripleScan {
                        pattern: pattern.clone(),
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
            GraphPattern::Filter { expr, inner } => {
                let inner_plan = self.plan_graph_pattern(inner)?;
                Ok(ExecutionPlan::Filter {
                    input: Box::new(inner_plan),
                    condition: expr.clone(),
                })
            }
            GraphPattern::Union(left, right) => {
                let left_plan = self.plan_graph_pattern(left)?;
                let right_plan = self.plan_graph_pattern(right)?;
                Ok(ExecutionPlan::Union {
                    left: Box::new(left_plan),
                    right: Box::new(right_plan),
                })
            }
            _ => Err(OxirsError::Query("Graph pattern not yet supported".to_string())),
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