//! # SHACL Node Expressions (SHACL-AF `sh:values`) – module facade
//!
//! Implements SHACL Advanced Features node expression evaluation, enabling
//! declarative computation of RDF node values from graph patterns. Node
//! expressions are the foundation of SHACL rules (sh:values, sh:condition) and
//! allow complex value derivation without SPARQL.
//!
//! The implementation has been split into three sibling modules to keep each
//! source file below the workspace 2000-line refactor threshold:
//!
//! - [`node_expr_types`](super::node_expr_types) – AST, RDF term, property
//!   path, graph trait, in-memory graph, config / stats / errors, and the
//!   [`ExprBuilder`](super::node_expr_types::ExprBuilder) fluent API.
//! - [`node_expr_evaluator`](super::node_expr_evaluator) –
//!   [`NodeExprEvaluator`](super::node_expr_evaluator::NodeExprEvaluator) and
//!   evaluation control flow.
//! - [`node_expr_builtins`](super::node_expr_builtins) – built-in function
//!   implementations (sh:strlen, sh:concat, sh:sum) and registration helper.
//!
//! All public items remain available from this path for backward compatibility
//! through the re-exports below.
//!
//! ## Supported Expression Types
//!
//! - **Focus Node** (`sh:this`): The current focus node
//! - **Constant** (`sh:value`): A fixed IRI, literal, or blank node
//! - **Path** (`sh:path`): Follow a property path from the focus node
//! - **Filter Shape** (`sh:filterShape`): Nodes matching a given shape
//! - **Function** (`sh:function`): Apply a SHACL function to arguments
//! - **Intersection** (`sh:intersection`): Nodes common to all operands
//! - **Union** (`sh:union`): Nodes in any operand
//! - **Minus** (`sh:minus`): Nodes in the first but not the second operand
//! - **If-Then-Else** (`sh:if`): Conditional value selection
//! - **Count** (`sh:count`): Count matching nodes
//! - **Distinct** (`sh:distinct`): Deduplicate results
//! - **OrderBy** (`sh:orderBy`): Sort results
//! - **Limit/Offset** (`sh:limit`, `sh:offset`): Pagination
//!
//! ## References
//!
//! - <https://www.w3.org/TR/shacl-af/#node-expressions>

pub use crate::node_expr_builtins::*;
pub use crate::node_expr_evaluator::*;
pub use crate::node_expr_types::*;
