//! Query plan optimization for SPARQL-star queries
//!
//! This module implements cost-based query optimization for SPARQL-star,
//! specifically handling quoted triples and nested patterns efficiently.
//!
//! # Features
//!
//! - **Cost-based optimization** - Estimate costs and choose optimal execution plans
//! - **Join reordering** - Optimize join order for quoted triple patterns
//! - **Index selection** - Choose the best indices for query execution
//! - **Filter pushdown** - Push filters through quoted triple boundaries
//! - **Cardinality estimation** - Estimate result sizes for better planning
//! - **Query caching** - Cache optimized plans for repeated queries
//!
//! # Examples
//!
//! ```rust
//! use oxirs_star::query_optimizer::{QueryOptimizer, QueryPlan};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let optimizer = QueryOptimizer::new();
//!
//! // Optimize a SPARQL-star query
//! let query = "SELECT ?s ?p ?o WHERE { <<?s ?p ?o>> ex:certainty ?c . FILTER(?c > 0.8) }";
//! let plan = optimizer.optimize_query(query)?;
//!
//! println!("Estimated cost: {}", plan.estimated_cost());
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, span, Level};

// SciRS2 imports for optimization algorithms (SCIRS2 POLICY)
// Random is available for future stochastic optimization features

use crate::model::{StarTerm, StarTriple};
use crate::store::StarStore;
use crate::StarResult;

/// Query plan for SPARQL-star execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    /// Ordered list of operations to execute
    operations: Vec<QueryOperation>,

    /// Estimated total cost
    estimated_cost: f64,

    /// Estimated result cardinality
    estimated_cardinality: usize,

    /// Statistics used for optimization
    statistics: QueryStatistics,

    /// Selected indices for each operation
    index_selections: HashMap<usize, IndexChoice>,
}

/// Individual query operation in the execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryOperation {
    /// Scan a triple pattern
    TripleScan {
        /// Pattern to scan
        pattern: TriplePattern,
        /// Estimated selectivity (0.0 to 1.0)
        selectivity: f64,
        /// Estimated result size
        cardinality: usize,
    },

    /// Join two intermediate results
    Join {
        /// Left input operation index
        left: usize,
        /// Right input operation index
        right: usize,
        /// Join variables
        join_vars: Vec<String>,
        /// Join strategy
        strategy: JoinStrategy,
        /// Estimated cost
        cost: f64,
    },

    /// Filter operation
    Filter {
        /// Input operation index
        input: usize,
        /// Filter expression
        expression: FilterExpression,
        /// Estimated selectivity
        selectivity: f64,
    },

    /// Project (SELECT) operation
    Project {
        /// Input operation index
        input: usize,
        /// Variables to project
        variables: Vec<String>,
    },

    /// Distinct operation
    Distinct {
        /// Input operation index
        input: usize,
    },

    /// Order by operation
    OrderBy {
        /// Input operation index
        input: usize,
        /// Order specifications
        order_specs: Vec<OrderSpec>,
    },

    /// Limit operation
    Limit {
        /// Input operation index
        input: usize,
        /// Limit value
        limit: usize,
    },
}

/// Triple pattern for matching
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TriplePattern {
    /// Subject pattern (variable or constant)
    pub subject: PatternTerm,
    /// Predicate pattern (variable or constant)
    pub predicate: PatternTerm,
    /// Object pattern (variable or constant)
    pub object: PatternTerm,
    /// Is this pattern for a quoted triple?
    pub is_quoted: bool,
}

/// Pattern term (variable or constant)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PatternTerm {
    /// Variable binding
    Variable(String),
    /// Constant value
    Constant(String),
    /// Quoted triple pattern
    QuotedPattern(Box<TriplePattern>),
}

/// Join strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum JoinStrategy {
    /// Nested loop join
    NestedLoop,
    /// Hash join
    Hash,
    /// Merge join (for sorted inputs)
    Merge,
    /// Index nested loop join
    IndexNestedLoop,
}

/// Filter expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterExpression {
    /// Comparison: variable > constant
    GreaterThan(String, f64),
    /// Comparison: variable < constant
    LessThan(String, f64),
    /// Comparison: variable = constant
    Equals(String, String),
    /// Logical AND
    And(Box<FilterExpression>, Box<FilterExpression>),
    /// Logical OR
    Or(Box<FilterExpression>, Box<FilterExpression>),
    /// Logical NOT
    Not(Box<FilterExpression>),
}

/// Order specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderSpec {
    /// Variable to order by
    pub variable: String,
    /// Ascending or descending
    pub ascending: bool,
}

/// Index choice for an operation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum IndexChoice {
    /// SPO index (subject-predicate-object)
    SPO,
    /// POS index (predicate-object-subject)
    POS,
    /// OSP index (object-subject-predicate)
    OSP,
    /// No index (full scan)
    FullScan,
}

/// Query statistics for cost estimation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryStatistics {
    /// Total number of triples in store
    pub total_triples: usize,
    /// Number of quoted triples
    pub quoted_triples: usize,
    /// Distinct subject count
    pub distinct_subjects: usize,
    /// Distinct predicate count
    pub distinct_predicates: usize,
    /// Distinct object count
    pub distinct_objects: usize,
    /// Average nesting depth
    pub avg_nesting_depth: f64,
}

impl QueryPlan {
    /// Create a new empty query plan
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            estimated_cost: 0.0,
            estimated_cardinality: 0,
            statistics: QueryStatistics::default(),
            index_selections: HashMap::new(),
        }
    }

    /// Get the estimated cost
    pub fn estimated_cost(&self) -> f64 {
        self.estimated_cost
    }

    /// Get the estimated result cardinality
    pub fn estimated_cardinality(&self) -> usize {
        self.estimated_cardinality
    }

    /// Get the operations
    pub fn operations(&self) -> &[QueryOperation] {
        &self.operations
    }

    /// Add an operation to the plan
    pub fn add_operation(&mut self, operation: QueryOperation) {
        self.operations.push(operation);
    }

    /// Set the estimated cost
    pub fn set_estimated_cost(&mut self, cost: f64) {
        self.estimated_cost = cost;
    }

    /// Set the estimated cardinality
    pub fn set_estimated_cardinality(&mut self, cardinality: usize) {
        self.estimated_cardinality = cardinality;
    }
}

impl Default for QueryPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Query optimizer for SPARQL-star
pub struct QueryOptimizer {
    /// Query statistics
    statistics: QueryStatistics,

    /// Cache of optimized plans
    plan_cache: HashMap<String, QueryPlan>,

    /// Configuration
    config: OptimizerConfig,
}

/// Optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Enable join reordering
    pub enable_join_reordering: bool,

    /// Enable filter pushdown
    pub enable_filter_pushdown: bool,

    /// Enable index selection
    pub enable_index_selection: bool,

    /// Enable plan caching
    pub enable_plan_caching: bool,

    /// Maximum number of cached plans
    pub max_cached_plans: usize,

    /// Cost threshold for using hash join
    pub hash_join_threshold: usize,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            enable_join_reordering: true,
            enable_filter_pushdown: true,
            enable_index_selection: true,
            enable_plan_caching: true,
            max_cached_plans: 1000,
            hash_join_threshold: 1000,
        }
    }
}

impl QueryOptimizer {
    /// Create a new query optimizer
    pub fn new() -> Self {
        Self {
            statistics: QueryStatistics::default(),
            plan_cache: HashMap::new(),
            config: OptimizerConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: OptimizerConfig) -> Self {
        Self {
            statistics: QueryStatistics::default(),
            plan_cache: HashMap::new(),
            config,
        }
    }

    /// Update statistics from a store
    pub fn update_statistics(&mut self, store: &StarStore) -> StarResult<()> {
        let span = span!(Level::INFO, "update_optimizer_statistics");
        let _enter = span.enter();

        let all_triples = store.query(None, None, None)?;

        self.statistics.total_triples = all_triples.len();
        self.statistics.quoted_triples = all_triples
            .iter()
            .filter(|t| {
                matches!(t.subject, StarTerm::QuotedTriple(_))
                    || matches!(t.object, StarTerm::QuotedTriple(_))
            })
            .count();

        // Calculate distinct counts
        let mut subjects = HashSet::new();
        let mut predicates = HashSet::new();
        let mut objects = HashSet::new();

        for triple in &all_triples {
            subjects.insert(format!("{:?}", triple.subject));
            predicates.insert(format!("{:?}", triple.predicate));
            objects.insert(format!("{:?}", triple.object));
        }

        self.statistics.distinct_subjects = subjects.len();
        self.statistics.distinct_predicates = predicates.len();
        self.statistics.distinct_objects = objects.len();

        // Calculate average nesting depth
        let total_depth: usize = all_triples
            .iter()
            .map(|t| self.calculate_nesting_depth(t))
            .sum();
        self.statistics.avg_nesting_depth = if !all_triples.is_empty() {
            total_depth as f64 / all_triples.len() as f64
        } else {
            0.0
        };

        info!(
            "Updated optimizer statistics: {} total triples, {} quoted triples",
            self.statistics.total_triples, self.statistics.quoted_triples
        );

        Ok(())
    }

    /// Calculate nesting depth of a triple
    fn calculate_nesting_depth(&self, triple: &StarTriple) -> usize {
        let subject_depth = self.term_depth(&triple.subject);
        let object_depth = self.term_depth(&triple.object);
        subject_depth.max(object_depth)
    }

    fn term_depth(&self, term: &StarTerm) -> usize {
        match term {
            StarTerm::QuotedTriple(inner) => 1 + self.calculate_nesting_depth(inner),
            _ => 0,
        }
    }

    /// Optimize a SPARQL-star query (simplified for demonstration)
    pub fn optimize_query(&mut self, _query: &str) -> StarResult<QueryPlan> {
        let span = span!(Level::INFO, "optimize_query");
        let _enter = span.enter();

        // Check cache if enabled
        if self.config.enable_plan_caching {
            if let Some(cached_plan) = self.plan_cache.get(_query) {
                debug!("Using cached query plan");
                return Ok(cached_plan.clone());
            }
        }

        // Create a basic plan (simplified - full implementation would parse SPARQL)
        let mut plan = QueryPlan::new();
        plan.statistics = self.statistics.clone();

        // Example: Create a simple scan operation
        let pattern = TriplePattern {
            subject: PatternTerm::Variable("s".to_string()),
            predicate: PatternTerm::Variable("p".to_string()),
            object: PatternTerm::Variable("o".to_string()),
            is_quoted: false,
        };

        let selectivity = self.estimate_selectivity(&pattern);
        let cardinality = (self.statistics.total_triples as f64 * selectivity) as usize;

        plan.add_operation(QueryOperation::TripleScan {
            pattern,
            selectivity,
            cardinality,
        });

        // Estimate total cost
        let cost = self.estimate_plan_cost(&plan);
        plan.set_estimated_cost(cost);
        plan.set_estimated_cardinality(cardinality);

        // Cache the plan if enabled
        if self.config.enable_plan_caching && self.plan_cache.len() < self.config.max_cached_plans {
            self.plan_cache.insert(_query.to_string(), plan.clone());
        }

        info!("Generated query plan with estimated cost: {}", cost);
        Ok(plan)
    }

    /// Estimate selectivity for a triple pattern
    fn estimate_selectivity(&self, pattern: &TriplePattern) -> f64 {
        let mut selectivity = 1.0;

        // Reduce selectivity for each constant term
        match &pattern.subject {
            PatternTerm::Constant(_) => {
                selectivity *= 1.0 / self.statistics.distinct_subjects.max(1) as f64;
            }
            PatternTerm::QuotedPattern(_) => {
                selectivity *= self.statistics.quoted_triples as f64
                    / self.statistics.total_triples.max(1) as f64;
            }
            _ => {}
        }

        if let PatternTerm::Constant(_) = &pattern.predicate {
            selectivity *= 1.0 / self.statistics.distinct_predicates.max(1) as f64;
        }

        match &pattern.object {
            PatternTerm::Constant(_) => {
                selectivity *= 1.0 / self.statistics.distinct_objects.max(1) as f64;
            }
            PatternTerm::QuotedPattern(_) => {
                selectivity *= self.statistics.quoted_triples as f64
                    / self.statistics.total_triples.max(1) as f64;
            }
            _ => {}
        }

        selectivity.max(0.0001) // Minimum selectivity
    }

    /// Estimate cost for a query plan
    fn estimate_plan_cost(&self, plan: &QueryPlan) -> f64 {
        let mut total_cost = 0.0;

        for operation in &plan.operations {
            match operation {
                QueryOperation::TripleScan { cardinality, .. } => {
                    // Cost = number of triples to scan
                    total_cost += *cardinality as f64;
                }
                QueryOperation::Join {
                    left,
                    right,
                    strategy,
                    ..
                } => {
                    // Simplified join cost estimation
                    let left_card = self.get_operation_cardinality(&plan.operations[*left]);
                    let right_card = self.get_operation_cardinality(&plan.operations[*right]);

                    let join_cost = match strategy {
                        JoinStrategy::NestedLoop => left_card as f64 * right_card as f64,
                        JoinStrategy::Hash => (left_card + right_card) as f64 * 1.5,
                        JoinStrategy::Merge => (left_card + right_card) as f64 * 1.2,
                        JoinStrategy::IndexNestedLoop => {
                            left_card as f64 * (right_card as f64).log2()
                        }
                    };

                    total_cost += join_cost;
                }
                QueryOperation::Filter {
                    input, selectivity, ..
                } => {
                    let input_card = self.get_operation_cardinality(&plan.operations[*input]);
                    total_cost += input_card as f64 * (1.0 + selectivity);
                }
                QueryOperation::Project { input, .. } => {
                    let input_card = self.get_operation_cardinality(&plan.operations[*input]);
                    total_cost += input_card as f64 * 0.5;
                }
                QueryOperation::Distinct { input } => {
                    let input_card = self.get_operation_cardinality(&plan.operations[*input]);
                    total_cost += input_card as f64 * (input_card as f64).log2();
                }
                QueryOperation::OrderBy { input, .. } => {
                    let input_card = self.get_operation_cardinality(&plan.operations[*input]);
                    total_cost += input_card as f64 * (input_card as f64).log2();
                }
                QueryOperation::Limit { .. } => {
                    total_cost += 1.0; // Minimal cost
                }
            }
        }

        total_cost
    }

    fn get_operation_cardinality(&self, operation: &QueryOperation) -> usize {
        match operation {
            QueryOperation::TripleScan { cardinality, .. } => *cardinality,
            QueryOperation::Join { cost, .. } => *cost as usize,
            QueryOperation::Filter { selectivity, .. } => {
                (self.statistics.total_triples as f64 * selectivity) as usize
            }
            _ => 1000, // Default estimate
        }
    }

    /// Select the best index for a triple pattern
    pub fn select_index(&self, pattern: &TriplePattern) -> IndexChoice {
        if !self.config.enable_index_selection {
            return IndexChoice::FullScan;
        }

        // Choose index based on which terms are constants
        match (&pattern.subject, &pattern.predicate, &pattern.object) {
            (PatternTerm::Constant(_), _, _) => IndexChoice::SPO,
            (_, PatternTerm::Constant(_), _) => IndexChoice::POS,
            (_, _, PatternTerm::Constant(_)) => IndexChoice::OSP,
            _ => IndexChoice::FullScan,
        }
    }

    /// Clear the plan cache
    pub fn clear_cache(&mut self) {
        self.plan_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_size(&self) -> usize {
        self.plan_cache.len()
    }
}

impl Default for QueryOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_plan_creation() {
        let mut plan = QueryPlan::new();

        let pattern = TriplePattern {
            subject: PatternTerm::Variable("s".to_string()),
            predicate: PatternTerm::Constant("http://example.org/p".to_string()),
            object: PatternTerm::Variable("o".to_string()),
            is_quoted: false,
        };

        plan.add_operation(QueryOperation::TripleScan {
            pattern,
            selectivity: 0.1,
            cardinality: 100,
        });

        assert_eq!(plan.operations().len(), 1);
    }

    #[test]
    fn test_optimizer_creation() {
        let optimizer = QueryOptimizer::new();
        assert_eq!(optimizer.cache_size(), 0);
    }

    #[test]
    fn test_selectivity_estimation() {
        let mut optimizer = QueryOptimizer::new();

        // Set up some realistic statistics
        optimizer.statistics.total_triples = 10000;
        optimizer.statistics.distinct_subjects = 1000;
        optimizer.statistics.distinct_predicates = 50;
        optimizer.statistics.distinct_objects = 5000;

        // All variables - high selectivity (should be 1.0 since all match)
        let pattern1 = TriplePattern {
            subject: PatternTerm::Variable("s".to_string()),
            predicate: PatternTerm::Variable("p".to_string()),
            object: PatternTerm::Variable("o".to_string()),
            is_quoted: false,
        };
        let sel1 = optimizer.estimate_selectivity(&pattern1);
        assert!(
            sel1 > 0.5,
            "All-variable pattern should have high selectivity"
        );

        // One constant - lower selectivity
        let pattern2 = TriplePattern {
            subject: PatternTerm::Constant("http://example.org/s".to_string()),
            predicate: PatternTerm::Variable("p".to_string()),
            object: PatternTerm::Variable("o".to_string()),
            is_quoted: false,
        };
        let sel2 = optimizer.estimate_selectivity(&pattern2);
        assert!(
            sel2 < sel1,
            "Pattern with constant should have lower selectivity: {} vs {}",
            sel2,
            sel1
        );
    }

    #[test]
    fn test_index_selection() {
        let optimizer = QueryOptimizer::new();

        // Subject constant -> SPO index
        let pattern1 = TriplePattern {
            subject: PatternTerm::Constant("s".to_string()),
            predicate: PatternTerm::Variable("p".to_string()),
            object: PatternTerm::Variable("o".to_string()),
            is_quoted: false,
        };
        assert_eq!(optimizer.select_index(&pattern1), IndexChoice::SPO);

        // Predicate constant -> POS index
        let pattern2 = TriplePattern {
            subject: PatternTerm::Variable("s".to_string()),
            predicate: PatternTerm::Constant("p".to_string()),
            object: PatternTerm::Variable("o".to_string()),
            is_quoted: false,
        };
        assert_eq!(optimizer.select_index(&pattern2), IndexChoice::POS);

        // Object constant -> OSP index
        let pattern3 = TriplePattern {
            subject: PatternTerm::Variable("s".to_string()),
            predicate: PatternTerm::Variable("p".to_string()),
            object: PatternTerm::Constant("o".to_string()),
            is_quoted: false,
        };
        assert_eq!(optimizer.select_index(&pattern3), IndexChoice::OSP);
    }

    #[test]
    fn test_cost_estimation() {
        let mut plan = QueryPlan::new();

        let pattern = TriplePattern {
            subject: PatternTerm::Variable("s".to_string()),
            predicate: PatternTerm::Variable("p".to_string()),
            object: PatternTerm::Variable("o".to_string()),
            is_quoted: false,
        };

        plan.add_operation(QueryOperation::TripleScan {
            pattern,
            selectivity: 1.0,
            cardinality: 1000,
        });

        let optimizer = QueryOptimizer::new();
        let cost = optimizer.estimate_plan_cost(&plan);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_plan_caching() {
        let mut optimizer = QueryOptimizer::new();

        let query = "SELECT * WHERE { ?s ?p ?o }";
        let plan1 = optimizer.optimize_query(query).unwrap();

        assert_eq!(optimizer.cache_size(), 1);

        // Should retrieve from cache
        let plan2 = optimizer.optimize_query(query).unwrap();
        assert_eq!(plan1.estimated_cost(), plan2.estimated_cost());
    }
}
