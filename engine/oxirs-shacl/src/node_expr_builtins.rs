//! Built-in functions for SHACL node expressions.
//!
//! Each built-in receives evaluated argument lists (a `Vec` per argument
//! position, since node expressions are set-valued) and returns the set of
//! result terms. Registration is performed by
//! [`register_builtins`], invoked from the evaluator constructor.

use std::collections::HashMap;

use super::node_expr_evaluator::BuiltinFunction;
use super::node_expr_types::{NodeExprError, NodeTerm};

/// `sh:strlen(?value)` – length in characters of each input term's string form.
pub fn builtin_strlen(args: &[Vec<NodeTerm>]) -> Result<Vec<NodeTerm>, NodeExprError> {
    let mut results = Vec::new();
    for term in &args[0] {
        let len = term.as_str().len();
        results.push(NodeTerm::typed_literal(
            len.to_string(),
            "http://www.w3.org/2001/XMLSchema#integer",
        ));
    }
    Ok(results)
}

/// `sh:concat(?a, ?b)` – Cartesian-product string concatenation of two value sets.
pub fn builtin_concat(args: &[Vec<NodeTerm>]) -> Result<Vec<NodeTerm>, NodeExprError> {
    let mut results = Vec::new();
    for a in &args[0] {
        for b in &args[1] {
            let concatenated = format!("{}{}", a.as_str(), b.as_str());
            results.push(NodeTerm::literal(concatenated));
        }
    }
    Ok(results)
}

/// `sh:sum(?value)` – numeric sum of all literals in the argument set.
pub fn builtin_sum(args: &[Vec<NodeTerm>]) -> Result<Vec<NodeTerm>, NodeExprError> {
    let mut total = 0.0_f64;
    for term in &args[0] {
        let val = term.as_float().ok_or_else(|| {
            NodeExprError::TypeError(format!(
                "Cannot convert '{}' to number for sum",
                term.as_str()
            ))
        })?;
        total += val;
    }
    Ok(vec![NodeTerm::typed_literal(
        total.to_string(),
        "http://www.w3.org/2001/XMLSchema#double",
    )])
}

/// Register the standard SHACL-AF node expression built-in functions into a registry.
pub(crate) fn register_builtins(functions: &mut HashMap<String, BuiltinFunction>) {
    functions.insert(
        "sh:strlen".to_string(),
        BuiltinFunction::new(1, builtin_strlen),
    );
    functions.insert(
        "sh:concat".to_string(),
        BuiltinFunction::new(2, builtin_concat),
    );
    functions.insert("sh:sum".to_string(), BuiltinFunction::new(1, builtin_sum));
}
