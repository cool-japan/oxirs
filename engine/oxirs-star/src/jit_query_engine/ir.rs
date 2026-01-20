//! Intermediate Representation (IR) for JIT-compiled SPARQL-star Queries
//!
//! This module provides an IR that bridges high-level SPARQL-star syntax
//! with low-level native code generation via scirs2_core::jit.
//!
//! The IR is designed to be:
//! - Optimizable (constant folding, dead code elimination)
//! - Analyzable (cost estimation, parallelization hints)
//! - Compilable (direct translation to LLVM IR or GPU kernels)

use serde::{Deserialize, Serialize};
use std::fmt;

/// IR operation type for SPARQL-star queries
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IrOp {
    /// Triple pattern match: (subject, predicate, object)
    TriplePattern {
        subject: IrTerm,
        predicate: IrTerm,
        object: IrTerm,
    },
    /// Quoted triple pattern: << s p o >> as subject/object
    QuotedTriplePattern {
        inner: Box<IrOp>,
        position: QuotePosition,
    },
    /// Filter expression
    Filter { condition: IrExpr },
    /// Join operation (inner, left, optional)
    Join {
        left: Box<IrOp>,
        right: Box<IrOp>,
        join_type: JoinType,
    },
    /// Union operation
    Union { left: Box<IrOp>, right: Box<IrOp> },
    /// Projection (SELECT variables)
    Project { vars: Vec<String>, child: Box<IrOp> },
    /// Distinct modifier
    Distinct { child: Box<IrOp> },
    /// Limit/Offset
    Slice {
        child: Box<IrOp>,
        limit: Option<usize>,
        offset: usize,
    },
    /// Order by
    Order {
        child: Box<IrOp>,
        conditions: Vec<OrderCondition>,
    },
    /// Index scan (optimized triple access)
    IndexScan {
        index_type: IndexType,
        keys: Vec<IrTerm>,
    },
    /// Sequential scan (fallback)
    SeqScan,
}

/// Position of quoted triple in pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuotePosition {
    Subject,
    Object,
}

/// Join type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JoinType {
    Inner,
    Left,
    Optional,
}

/// Index type for optimized access
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    SPO, // Subject-Predicate-Object
    POS, // Predicate-Object-Subject
    OSP, // Object-Subject-Predicate
}

/// IR term (variable, constant, quoted triple)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IrTerm {
    /// Variable (e.g., ?s, ?p, ?o)
    Variable(String),
    /// Constant IRI
    Iri(String),
    /// Literal with optional datatype/language
    Literal {
        value: String,
        datatype: Option<String>,
        language: Option<String>,
    },
    /// Blank node
    BlankNode(String),
    /// Quoted triple reference
    QuotedTriple(Box<(IrTerm, IrTerm, IrTerm)>),
    /// Wildcard (matches any)
    Any,
}

/// IR expression for filters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IrExpr {
    /// Variable reference
    Var(String),
    /// Constant value
    Const(IrValue),
    /// Binary operation
    BinOp {
        op: BinOp,
        left: Box<IrExpr>,
        right: Box<IrExpr>,
    },
    /// Unary operation
    UnaryOp { op: UnaryOp, operand: Box<IrExpr> },
    /// Function call (e.g., STR, LANG, DATATYPE)
    FunctionCall { name: String, args: Vec<IrExpr> },
    /// Exists/Not Exists
    Exists { pattern: Box<IrOp>, negated: bool },
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinOp {
    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    // Logical
    And,
    Or,
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    // String
    Concat,
    Regex,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Not,
    Neg,
    IsIri,
    IsBlank,
    IsLiteral,
}

/// IR value type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IrValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Null,
}

/// Order condition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderCondition {
    Asc(String),  // Variable name
    Desc(String), // Variable name
}

/// IR query plan
#[derive(Debug, Clone)]
pub struct IrQueryPlan {
    /// Root operation
    pub root: IrOp,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Parallelization hints
    pub parallel_hints: ParallelHints,
    /// Memory hints
    pub memory_hints: MemoryHints,
}

/// Parallelization hints for JIT compiler
#[derive(Debug, Clone, Default)]
pub struct ParallelHints {
    /// Can parallelize across triples
    pub can_parallelize_scans: bool,
    /// Join can be parallelized
    pub can_parallelize_joins: bool,
    /// Number of parallel tasks
    pub suggested_parallelism: Option<usize>,
}

/// Memory access hints for JIT compiler
#[derive(Debug, Clone, Default)]
pub struct MemoryHints {
    /// Expected result set size
    pub estimated_results: usize,
    /// Uses index (fast path)
    pub uses_index: bool,
    /// Sequential access pattern
    pub sequential_access: bool,
}

impl IrQueryPlan {
    /// Create a new IR query plan
    pub fn new(root: IrOp) -> Self {
        Self {
            root,
            estimated_cost: 0.0,
            parallel_hints: ParallelHints::default(),
            memory_hints: MemoryHints::default(),
        }
    }

    /// Estimate execution cost
    pub fn estimate_cost(&mut self) {
        self.estimated_cost = Self::cost_recursive(&self.root);
    }

    fn cost_recursive(op: &IrOp) -> f64 {
        match op {
            IrOp::TriplePattern { .. } => 10.0,
            IrOp::QuotedTriplePattern { .. } => 15.0,
            IrOp::Filter { .. } => 5.0,
            IrOp::Join { left, right, .. } => {
                50.0 + Self::cost_recursive(left) + Self::cost_recursive(right)
            }
            IrOp::Union { left, right } => Self::cost_recursive(left) + Self::cost_recursive(right),
            IrOp::Project { child, .. } => 2.0 + Self::cost_recursive(child),
            IrOp::Distinct { child } => 20.0 + Self::cost_recursive(child),
            IrOp::Slice { child, .. } => Self::cost_recursive(child),
            IrOp::Order { child, .. } => 30.0 + Self::cost_recursive(child),
            IrOp::IndexScan { .. } => 5.0,
            IrOp::SeqScan => 100.0,
        }
    }

    /// Analyze parallelization opportunities
    pub fn analyze_parallelism(&mut self) {
        self.parallel_hints = Self::analyze_parallel_recursive(&self.root);
    }

    fn analyze_parallel_recursive(op: &IrOp) -> ParallelHints {
        match op {
            IrOp::TriplePattern { .. } | IrOp::IndexScan { .. } => ParallelHints {
                can_parallelize_scans: true,
                can_parallelize_joins: false,
                suggested_parallelism: Some(num_cpus::get()),
            },
            IrOp::Join {
                left,
                right,
                join_type,
            } => {
                let left_hints = Self::analyze_parallel_recursive(left);
                let right_hints = Self::analyze_parallel_recursive(right);
                ParallelHints {
                    can_parallelize_scans: left_hints.can_parallelize_scans
                        && right_hints.can_parallelize_scans,
                    can_parallelize_joins: *join_type == JoinType::Inner,
                    suggested_parallelism: Some(num_cpus::get()),
                }
            }
            _ => ParallelHints::default(),
        }
    }
}

impl fmt::Display for IrOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrOp::TriplePattern {
                subject,
                predicate,
                object,
            } => write!(f, "({:?} {:?} {:?})", subject, predicate, object),
            IrOp::QuotedTriplePattern { inner, position } => {
                write!(f, "<<{:?}>> at {:?}", inner, position)
            }
            IrOp::Filter { condition } => write!(f, "FILTER({:?})", condition),
            IrOp::Join { join_type, .. } => write!(f, "JOIN({:?})", join_type),
            IrOp::Union { .. } => write!(f, "UNION"),
            IrOp::Project { vars, .. } => write!(f, "PROJECT({:?})", vars),
            IrOp::Distinct { .. } => write!(f, "DISTINCT"),
            IrOp::IndexScan { index_type, .. } => write!(f, "INDEX_SCAN({:?})", index_type),
            _ => write!(f, "{:?}", self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_term_creation() {
        let var = IrTerm::Variable("s".to_string());
        assert!(matches!(var, IrTerm::Variable(_)));

        let iri = IrTerm::Iri("http://example.org/foo".to_string());
        assert!(matches!(iri, IrTerm::Iri(_)));
    }

    #[test]
    fn test_triple_pattern_ir() {
        let pattern = IrOp::TriplePattern {
            subject: IrTerm::Variable("s".to_string()),
            predicate: IrTerm::Variable("p".to_string()),
            object: IrTerm::Variable("o".to_string()),
        };

        match pattern {
            IrOp::TriplePattern { .. } => {
                // Successfully created a TriplePattern
            }
            _ => panic!("Expected TriplePattern"),
        }
    }

    #[test]
    fn test_cost_estimation() {
        let pattern = IrOp::TriplePattern {
            subject: IrTerm::Variable("s".to_string()),
            predicate: IrTerm::Variable("p".to_string()),
            object: IrTerm::Variable("o".to_string()),
        };

        let mut plan = IrQueryPlan::new(pattern);
        plan.estimate_cost();

        assert_eq!(plan.estimated_cost, 10.0);
    }

    #[test]
    fn test_join_cost_estimation() {
        let left = IrOp::TriplePattern {
            subject: IrTerm::Variable("s".to_string()),
            predicate: IrTerm::Variable("p".to_string()),
            object: IrTerm::Variable("o".to_string()),
        };

        let right = IrOp::TriplePattern {
            subject: IrTerm::Variable("s".to_string()),
            predicate: IrTerm::Iri("http://ex.org/name".to_string()),
            object: IrTerm::Variable("name".to_string()),
        };

        let join = IrOp::Join {
            left: Box::new(left),
            right: Box::new(right),
            join_type: JoinType::Inner,
        };

        let mut plan = IrQueryPlan::new(join);
        plan.estimate_cost();

        assert_eq!(plan.estimated_cost, 70.0); // 50 + 10 + 10
    }

    #[test]
    fn test_parallelism_analysis() {
        let pattern = IrOp::TriplePattern {
            subject: IrTerm::Variable("s".to_string()),
            predicate: IrTerm::Variable("p".to_string()),
            object: IrTerm::Variable("o".to_string()),
        };

        let mut plan = IrQueryPlan::new(pattern);
        plan.analyze_parallelism();

        assert!(plan.parallel_hints.can_parallelize_scans);
        assert!(plan.parallel_hints.suggested_parallelism.is_some());
    }

    #[test]
    fn test_quoted_triple_ir() {
        let inner = IrOp::TriplePattern {
            subject: IrTerm::Variable("s".to_string()),
            predicate: IrTerm::Variable("p".to_string()),
            object: IrTerm::Variable("o".to_string()),
        };

        let quoted = IrOp::QuotedTriplePattern {
            inner: Box::new(inner),
            position: QuotePosition::Subject,
        };

        match quoted {
            IrOp::QuotedTriplePattern { position, .. } => {
                assert_eq!(position, QuotePosition::Subject);
            }
            _ => panic!("Expected QuotedTriplePattern"),
        }
    }

    #[test]
    fn test_ir_expression() {
        let expr = IrExpr::BinOp {
            op: BinOp::Eq,
            left: Box::new(IrExpr::Var("x".to_string())),
            right: Box::new(IrExpr::Const(IrValue::Int(42))),
        };

        match expr {
            IrExpr::BinOp { op, .. } => assert_eq!(op, BinOp::Eq),
            _ => panic!("Expected BinOp"),
        }
    }
}
