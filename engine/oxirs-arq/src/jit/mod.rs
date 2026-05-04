//! JIT compilation phases b, c, and d — Cranelift-based codegen for filter expressions,
//! hash-join key comparison, ORDER BY evaluation, PROJECT column extraction, DISTINCT
//! deduplication hashing, and HAVING clause predicates over aggregate results.
//!
//! # Phase b — filter expressions
//!
//! Compiles a supported subset of SPARQL filter expressions to native machine code,
//! giving 5–10× speedup on hot numeric-filter queries.
//!
//! ## Supported subset
//!
//! - Numeric literals (xsd:integer, xsd:decimal, xsd:double, xsd:float, xsd:int, xsd:long)
//! - Variables (`?x` → `f64` slot loaded from a caller-provided slice)
//! - Comparisons: `<`, `>`, `<=`, `>=`, `=`, `!=`
//! - Arithmetic: `+`, `-`, `*`, `/`
//! - Logical: `&&`, `||`, `!`
//! - Built-ins: `ABS()`, `CEIL()`, `FLOOR()`, `ROUND()`
//!
//! ## Fall-back to interpreter
//!
//! Everything else (string ops, REGEX, ISIRI, ISBLANK, ISLITERAL, mixed types, …) returns
//! `None` from `try_lower` and the caller must delegate to the interpreted evaluator.
//!
//! # Phase c — hash-join key comparison
//!
//! Compiles a multi-key join comparator: given two `&[f64]` rows, returns `true` if all
//! configured key columns match. Supports epsilon (`|a-b| < 1e-9`) and exact (bitcast to
//! `i64`) comparison modes per column.
//!
//! # Phase c — ORDER BY evaluation
//!
//! Compiles a multi-column lexicographic comparator returning `-1 / 0 / 1` with per-column
//! ascending/descending flags baked into the native function.
//!
//! # Phase d — PROJECT/DISTINCT/HAVING codegen
//!
//! Three new operator families completing full SPARQL operator JIT coverage:
//!
//! ## PROJECT column extraction
//!
//! [`ProjectCompiler`] compiles column selection and reordering for the SPARQL PROJECT
//! and GROUP BY operators.  The compiled function copies selected columns from a source
//! `f64` slice into a destination slice, performing a bounds check and returning `1` on
//! success or `0` on bounds error.
//!
//! ## DISTINCT deduplication hashing
//!
//! [`DistinctCompiler`] compiles an FNV-1a hash over selected key columns for DISTINCT
//! deduplication.  Each selected column's `f64` bit pattern is incorporated into the
//! hash via bitcast to `i64`, ensuring that distinct NaN payloads hash differently.
//!
//! ## HAVING clause predicates
//!
//! [`HavingCompiler`] is a thin typed wrapper over [`FilterCompiler`] for HAVING clause
//! compilation.  It validates that all variable references in the predicate expression
//! appear in the supplied aggregate-variable-to-index map before delegating to Cranelift
//! codegen.  The resulting [`CompiledFilter`] evaluates the predicate when called with
//! aggregate result tuples as `f64` slices.
//!
//! # Feature gate
//!
//! All items in this module require the `jit` Cargo feature.

#[cfg(feature = "jit")]
pub mod distinct_compiler;
#[cfg(feature = "jit")]
pub mod filter_compiler;
#[cfg(feature = "jit")]
pub mod having_compiler;
#[cfg(feature = "jit")]
pub mod jit_cache;
#[cfg(feature = "jit")]
pub mod join_compiler;
#[cfg(feature = "jit")]
pub mod order_compiler;
#[cfg(feature = "jit")]
pub mod project_compiler;

#[cfg(feature = "jit")]
pub use distinct_compiler::{
    CompiledDistinct, DistinctCompiler, DistinctCompilerError, DistinctKeySpec,
};
#[cfg(feature = "jit")]
pub use filter_compiler::{
    BinOp, BuiltinFunc, CompiledFilter, FilterCompiler, FilterCompilerError, FilterExpr,
    VarIndexMap,
};
#[cfg(feature = "jit")]
pub use having_compiler::{AggVarMap, HavingCompiler};
#[cfg(feature = "jit")]
pub use jit_cache::JitFilterCache;
#[cfg(feature = "jit")]
pub use join_compiler::{CompiledJoinKey, JoinCompiler, JoinCompilerError, JoinKeySpec};
#[cfg(feature = "jit")]
pub use order_compiler::{CompiledOrder, OrderCompiler, OrderCompilerError, OrderKeySpec};
#[cfg(feature = "jit")]
pub use project_compiler::{CompiledProject, ProjectCompiler, ProjectCompilerError, ProjectSpec};
