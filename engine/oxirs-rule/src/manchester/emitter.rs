//! AST → Manchester Syntax text serializer.
//!
//! The primary entry point is [`emit`], which converts a [`ManchesterExpr`]
//! back into a Manchester Syntax string.
//!
//! ## Parenthesization rules
//!
//! The spec allows compound expressions (`and`, `or`) to appear without outer
//! parentheses only at the *top level* of the expression.  In any nested
//! position where ambiguity could arise, parentheses are added.  This ensures
//! that `parse(emit(expr))` always round-trips to the same AST.

use super::{ManchesterError, ManchesterExpr};

/// Serialize a [`ManchesterExpr`] to an OWL Manchester Syntax string.
///
/// The result is human-readable and suitable for storing in ontology files or
/// displaying in user interfaces.  Compound expressions (`and`, `or`) are
/// parenthesized when they appear in nested positions to avoid ambiguity.
///
/// # Errors
///
/// Returns [`ManchesterError::EmitError`] for structurally invalid expressions,
/// such as an empty `And`, `Or`, or `OneOf` operand list.
pub fn emit(expr: &ManchesterExpr) -> Result<String, ManchesterError> {
    emit_inner(expr, true)
}

/// Internal recursive emitter.
///
/// `top_level` controls whether `And`/`Or` nodes are wrapped in parentheses.
/// At the top level they are *not* parenthesized; in nested positions they are.
fn emit_inner(expr: &ManchesterExpr, top_level: bool) -> Result<String, ManchesterError> {
    match expr {
        // ── Atomic ────────────────────────────────────────────────────────────
        ManchesterExpr::Class(name) => {
            if name.is_empty() {
                return Err(ManchesterError::EmitError(
                    "Class name must not be empty".to_string(),
                ));
            }
            Ok(name.clone())
        }

        // ── Boolean connectives ───────────────────────────────────────────────
        ManchesterExpr::And(arms) => {
            if arms.is_empty() {
                return Err(ManchesterError::EmitError(
                    "`And` must have at least one arm".to_string(),
                ));
            }
            let parts: Result<Vec<String>, _> =
                arms.iter().map(|arm| emit_inner(arm, false)).collect();
            let joined = parts?.join(" and ");
            if top_level {
                Ok(joined)
            } else {
                Ok(format!("({joined})"))
            }
        }

        ManchesterExpr::Or(arms) => {
            if arms.is_empty() {
                return Err(ManchesterError::EmitError(
                    "`Or` must have at least one arm".to_string(),
                ));
            }
            let parts: Result<Vec<String>, _> =
                arms.iter().map(|arm| emit_inner(arm, false)).collect();
            let joined = parts?.join(" or ");
            if top_level {
                Ok(joined)
            } else {
                Ok(format!("({joined})"))
            }
        }

        ManchesterExpr::Not(inner) => {
            // `not` is always followed by a single primary; no parens needed
            // around the `not` itself (it binds tighter than `and`/`or`).
            let s = emit_inner(inner, false)?;
            Ok(format!("not {s}"))
        }

        // ── Property restrictions ─────────────────────────────────────────────
        ManchesterExpr::Some { property, filler } => {
            validate_property(property)?;
            let filler_str = emit_inner(filler, false)?;
            Ok(format!("{property} some {filler_str}"))
        }

        ManchesterExpr::Only { property, filler } => {
            validate_property(property)?;
            let filler_str = emit_inner(filler, false)?;
            Ok(format!("{property} only {filler_str}"))
        }

        ManchesterExpr::Min {
            property,
            cardinality,
            filler,
        } => {
            validate_property(property)?;
            let filler_str = emit_optional_filler(filler)?;
            Ok(format!("{property} min {cardinality}{filler_str}"))
        }

        ManchesterExpr::Max {
            property,
            cardinality,
            filler,
        } => {
            validate_property(property)?;
            let filler_str = emit_optional_filler(filler)?;
            Ok(format!("{property} max {cardinality}{filler_str}"))
        }

        ManchesterExpr::Exactly {
            property,
            cardinality,
            filler,
        } => {
            validate_property(property)?;
            let filler_str = emit_optional_filler(filler)?;
            Ok(format!("{property} exactly {cardinality}{filler_str}"))
        }

        // ── Nominal / value ────────────────────────────────────────────────────
        ManchesterExpr::HasValue {
            property,
            individual,
        } => {
            validate_property(property)?;
            if individual.is_empty() {
                return Err(ManchesterError::EmitError(
                    "HasValue individual must not be empty".to_string(),
                ));
            }
            Ok(format!("{property} value {individual}"))
        }

        ManchesterExpr::OneOf(individuals) => {
            if individuals.is_empty() {
                return Err(ManchesterError::EmitError(
                    "`OneOf` must contain at least one individual".to_string(),
                ));
            }
            let joined = individuals.join(" ");
            Ok(format!("{{{joined}}}"))
        }
    }
}

/// Emit an optional filler class expression.
///
/// Returns `""` when `filler` is `None`, or `" C"` (with a leading space) when
/// it is `Some(C)`, so callers can simply append the result to the base string.
fn emit_optional_filler(filler: &Option<Box<ManchesterExpr>>) -> Result<String, ManchesterError> {
    match filler {
        None => Ok(String::new()),
        Some(f) => {
            let s = emit_inner(f, false)?;
            Ok(format!(" {s}"))
        }
    }
}

/// Validate that a property name is non-empty.
fn validate_property(property: &str) -> Result<(), ManchesterError> {
    if property.is_empty() {
        return Err(ManchesterError::EmitError(
            "property name must not be empty".to_string(),
        ));
    }
    Ok(())
}
