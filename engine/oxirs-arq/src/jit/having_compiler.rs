//! Thin semantic wrapper over [`FilterCompiler`] for SPARQL HAVING clause compilation
//! (JIT phase d).
//!
//! HAVING clauses apply the same numeric comparison logic as FILTER clauses, but operate
//! on aggregate result variables (e.g. `?sum`, `?count`) rather than per-solution binding
//! variables.  This module provides a typed wrapper so that HAVING compilation is
//! distinguishable from plain FILTER compilation in the query engine, while reusing all of
//! the Cranelift IR generation from `filter_compiler`.
//!
//! # Usage
//!
//! ```ignore
//! use oxirs_arq::jit::{AggVarMap, FilterExpr, BinOp, HavingCompiler};
//!
//! let mut compiler = HavingCompiler::new();
//! let mut agg_map = AggVarMap::new();
//! agg_map.insert("sum".to_string(), 0);
//!
//! // HAVING (?sum > 100)
//! let expr = FilterExpr::BinOp {
//!     op: BinOp::Gt,
//!     left: Box::new(FilterExpr::Variable("sum".to_string())),
//!     right: Box::new(FilterExpr::Literal(100.0)),
//! };
//!
//! let compiled = compiler.compile_having(expr, agg_map).unwrap();
//! // Feed aggregate result tuple as HashMap<String, f64>
//! ```

use std::collections::HashMap;

use super::filter_compiler::{CompiledFilter, FilterCompiler, FilterCompilerError, FilterExpr};

/// Maps aggregate variable name to its position in the aggregate result tuple.
///
/// For example, `{"sum" → 0, "count" → 1}` means the first `f64` in the aggregate
/// result slice is the `?sum` value and the second is `?count`.
pub type AggVarMap = HashMap<String, usize>;

/// Compiles HAVING clause expressions over aggregate results.
///
/// Semantically identical to [`FilterCompiler`] but typed to the aggregate context —
/// the variable→index map refers to positions in the GROUP BY output tuple rather than
/// per-binding slots.  The additional `validate_vars` call on `compile_having` ensures
/// that all variable references in the expression are present in the supplied
/// [`AggVarMap`] before compilation begins.
pub struct HavingCompiler {
    inner: FilterCompiler,
}

impl HavingCompiler {
    /// Create a new `HavingCompiler`.
    pub fn new() -> Self {
        HavingCompiler {
            inner: FilterCompiler,
        }
    }

    /// Compile a HAVING predicate over aggregate results.
    ///
    /// `expr` uses variable names that map to aggregate result slots via `agg_var_map`.
    /// Returns a [`CompiledFilter`] that evaluates the predicate when called with the
    /// aggregate result tuple as an `f64` slice.
    ///
    /// # Errors
    ///
    /// Returns [`FilterCompilerError::UnsupportedExpression`] if any variable in `expr`
    /// is not present in `agg_var_map`.
    ///
    /// Returns other [`FilterCompilerError`] variants on Cranelift codegen failures.
    ///
    /// Returns [`FilterCompilerError::UnsupportedExpression`] if the expression is not
    /// in the JIT-supported subset (this should not happen for well-formed HAVING
    /// predicates, but is handled defensively).
    pub fn compile_having(
        &mut self,
        expr: FilterExpr,
        agg_var_map: AggVarMap,
    ) -> Result<CompiledFilter, FilterCompilerError> {
        // Validate: all variable references in expr must appear in agg_var_map.
        validate_vars(&expr, &agg_var_map)?;

        // Delegate to FilterCompiler — it returns Result<Option<CompiledFilter>, _>.
        // The Option is None only for unsupported expressions; since we validated the
        // expression already, None is unexpected, but we handle it defensively.
        self.inner.compile(&expr, agg_var_map).and_then(|opt| {
            opt.ok_or_else(|| {
                FilterCompilerError::UnsupportedExpression(
                    "filter compiler returned None for a validated HAVING FilterExpr".to_string(),
                )
            })
        })
    }
}

impl Default for HavingCompiler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Variable validation
// ---------------------------------------------------------------------------

/// Recursively walk `expr` and verify that every [`FilterExpr::Variable`] node appears
/// in `map`.  Returns an error naming the first missing variable.
fn validate_vars(expr: &FilterExpr, map: &AggVarMap) -> Result<(), FilterCompilerError> {
    match expr {
        FilterExpr::Variable(name) => {
            if !map.contains_key(name.as_str()) {
                return Err(FilterCompilerError::UnsupportedExpression(format!(
                    "variable '{}' not in agg_var_map",
                    name
                )));
            }
        }
        FilterExpr::BinOp { left, right, .. } => {
            validate_vars(left, map)?;
            validate_vars(right, map)?;
        }
        FilterExpr::UnaryNot(inner) => {
            validate_vars(inner, map)?;
        }
        FilterExpr::Builtin { arg: inner, .. } => {
            validate_vars(inner, map)?;
        }
        FilterExpr::Literal(_) => {
            // No variables to check.
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests (inline)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::filter_compiler::BinOp;

    fn compiler() -> HavingCompiler {
        HavingCompiler::new()
    }

    fn agg_map_single(name: &str) -> AggVarMap {
        let mut m = AggVarMap::new();
        m.insert(name.to_string(), 0);
        m
    }

    #[test]
    fn test_sum_gt_threshold_pass() {
        let mut c = compiler();
        let map = agg_map_single("sum");
        let expr = FilterExpr::BinOp {
            op: BinOp::Gt,
            left: Box::new(FilterExpr::Variable("sum".to_string())),
            right: Box::new(FilterExpr::Literal(100.0)),
        };
        let cf = c.compile_having(expr, map).expect("compile ok");
        let mut binding = HashMap::new();
        binding.insert("sum".to_string(), 150.0f64);
        assert_eq!(cf.evaluate(&binding), Some(true));
    }

    #[test]
    fn test_unknown_var_returns_error() {
        let mut c = compiler();
        let map = agg_map_single("sum");
        let expr = FilterExpr::BinOp {
            op: BinOp::Gt,
            left: Box::new(FilterExpr::Variable("unknown".to_string())),
            right: Box::new(FilterExpr::Literal(0.0)),
        };
        let result = c.compile_having(expr, map);
        assert!(
            matches!(result, Err(FilterCompilerError::UnsupportedExpression(_))),
            "expected UnsupportedExpression, got: {:?}",
            result
        );
    }

    #[test]
    fn test_literal_only_having() {
        // HAVING (1 < 2) — no variables, always true
        let mut c = compiler();
        let expr = FilterExpr::BinOp {
            op: BinOp::Lt,
            left: Box::new(FilterExpr::Literal(1.0)),
            right: Box::new(FilterExpr::Literal(2.0)),
        };
        let cf = c
            .compile_having(expr, AggVarMap::new())
            .expect("compile ok");
        let empty: HashMap<String, f64> = HashMap::new();
        assert_eq!(cf.evaluate(&empty), Some(true));
    }
}
