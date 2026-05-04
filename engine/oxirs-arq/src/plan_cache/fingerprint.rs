//! Algebra-level fingerprinting for the JIT plan cache.
//!
//! This module computes a stable `u64` hash that identifies the *structural shape*
//! of a SPARQL algebra plan, independent of variable spelling.  Two plans that
//! differ only in variable naming (e.g. `?x ?y` vs `?a ?b`) produce the **same**
//! fingerprint when the structure is identical.
//!
//! ## Algorithm
//! 1. Render the plan to its `Debug` string (cheap, always available).
//! 2. Scan the rendered string for `name: "<varname>"` patterns emitted by the
//!    derived `Debug` impl for `Variable { name: "…" }`.  For each unique
//!    `varname` encountered (in left-to-right order), replace the quoted string
//!    with `"_v{index}"` — this is the variable normalisation step.
//! 3. Feed the normalised bytes into a [`seahash::SeaHasher`] and return the
//!    final `u64`.
//!
//! The Debug-string approach is correct for all `Algebra` variants without
//! requiring a hand-rolled 20+-variant structural walker.  The `Variable`
//! struct derives `Debug`, producing `Variable { name: "varname" }`, so we key
//! on the `name: "` token which is unique to variable name fields in this AST.

use seahash::SeaHasher;
use std::hash::Hasher;

use crate::algebra::Algebra;

/// Compute a stable, normalised structural fingerprint for a query plan.
///
/// Variable names are normalised so `?x ?y` and `?a ?b` produce the same hash
/// when the plan structure is identical.
///
/// ```rust
/// use oxirs_arq::algebra::{Algebra, Term, Variable, TriplePattern};
/// use oxirs_arq::plan_cache::compute_fingerprint;
/// use oxirs_core::model::NamedNode;
///
/// let pred = Term::Iri(NamedNode::new_unchecked("http://example.org/p"));
/// let x = Term::Variable(Variable::new("x").unwrap());
/// let y = Term::Variable(Variable::new("y").unwrap());
/// let a = Term::Variable(Variable::new("a").unwrap());
/// let b = Term::Variable(Variable::new("b").unwrap());
///
/// let plan_xy = Algebra::Bgp(vec![TriplePattern { subject: x, predicate: pred.clone(), object: y }]);
/// let plan_ab = Algebra::Bgp(vec![TriplePattern { subject: a, predicate: pred.clone(), object: b }]);
///
/// // Structurally identical → same fingerprint
/// assert_eq!(compute_fingerprint(&plan_xy), compute_fingerprint(&plan_ab));
/// ```
pub fn compute_fingerprint(plan: &Algebra) -> u64 {
    let raw = format!("{plan:?}");
    let normalised = normalise_variable_names(&raw);
    let mut hasher = SeaHasher::new();
    hasher.write(normalised.as_bytes());
    hasher.finish()
}

/// Normalise variable names in a Debug-rendered algebra string.
///
/// `Variable` derives `Debug`, producing `Variable { name: "varname" }`.
/// This function scans for the token sequence `name: "` (which uniquely
/// identifies the name field of a `Variable` struct in the algebra Debug
/// output) and replaces the quoted value with `"_v{index}"` in first-encounter
/// order.
///
/// Any other quoted strings (IRI values, literal values) are left unchanged
/// because they do not appear immediately after `name: `.
fn normalise_variable_names(raw: &str) -> String {
    use std::collections::HashMap;

    // The needle we scan for in the Debug output.
    // Variable derives Debug → "Variable { name: \"varname\" }"
    // We anchor on the full struct prefix "Variable { name: \"" rather than
    // just "name: \"" to avoid false matches against other struct fields named
    // `name` (e.g. `Expression::Function { name: "fn_name", ... }`), which
    // would cause structurally distinct plans to collide.
    const NEEDLE: &str = "Variable { name: \"";
    let needle_len = NEEDLE.len();

    let bytes = raw.as_bytes();
    let len = bytes.len();

    let mut out = String::with_capacity(len);
    let mut mapping: HashMap<String, usize> = HashMap::new();
    let mut next_idx: usize = 0;
    let mut i = 0_usize;

    while i < len {
        // Try to match the needle at position i.
        if i + needle_len <= len && &raw[i..i + needle_len] == NEEDLE {
            // Emit the needle as-is.
            out.push_str(NEEDLE);
            i += needle_len;

            // Collect the quoted variable name until the closing `"`.
            let mut var_name = String::new();
            let mut escaped = false;
            while i < len {
                let b = bytes[i];
                if escaped {
                    var_name.push(b as char);
                    escaped = false;
                    i += 1;
                } else if b == b'\\' {
                    escaped = true;
                    var_name.push(b as char);
                    i += 1;
                } else if b == b'"' {
                    // End of the quoted name.  Emit normalised replacement.
                    let idx = *mapping.entry(var_name.clone()).or_insert_with(|| {
                        let v = next_idx;
                        next_idx += 1;
                        v
                    });
                    out.push_str(&format!("_v{idx}\""));
                    i += 1; // consume closing `"`
                    break;
                } else {
                    var_name.push(b as char);
                    i += 1;
                }
            }
        } else {
            // Emit the current character unchanged.
            out.push(bytes[i] as char);
            i += 1;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Term, TriplePattern, Variable};
    use oxirs_core::model::NamedNode;

    fn pred() -> Term {
        Term::Iri(NamedNode::new_unchecked("http://example.org/p"))
    }

    fn var(name: &str) -> Term {
        Term::Variable(Variable::new(name).expect("valid var"))
    }

    fn bgp(s: Term, o: Term) -> Algebra {
        Algebra::Bgp(vec![TriplePattern {
            subject: s,
            predicate: pred(),
            object: o,
        }])
    }

    #[test]
    fn identical_variables_same_fingerprint() {
        let p1 = bgp(var("x"), var("y"));
        let p2 = bgp(var("a"), var("b"));
        assert_eq!(compute_fingerprint(&p1), compute_fingerprint(&p2));
    }

    #[test]
    fn structurally_different_plans_different_fingerprint() {
        let p1 = bgp(var("x"), var("y"));
        let p2 = Algebra::Bgp(vec![
            TriplePattern {
                subject: var("x"),
                predicate: pred(),
                object: var("y"),
            },
            TriplePattern {
                subject: var("y"),
                predicate: pred(),
                object: var("z"),
            },
        ]);
        assert_ne!(compute_fingerprint(&p1), compute_fingerprint(&p2));
    }

    #[test]
    fn same_plan_same_fingerprint_deterministic() {
        let p = bgp(var("x"), var("y"));
        let fp1 = compute_fingerprint(&p);
        let fp2 = compute_fingerprint(&p);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn join_plans_correct_normalisation() {
        let p1 = Algebra::Join {
            left: Box::new(bgp(var("a"), var("b"))),
            right: Box::new(bgp(var("c"), var("d"))),
        };
        let p2 = Algebra::Join {
            left: Box::new(bgp(var("x"), var("y"))),
            right: Box::new(bgp(var("z"), var("w"))),
        };
        assert_eq!(compute_fingerprint(&p1), compute_fingerprint(&p2));
    }

    #[test]
    fn empty_bgp_fingerprint_stable() {
        let p = Algebra::Bgp(vec![]);
        let fp = compute_fingerprint(&p);
        assert_eq!(fp, compute_fingerprint(&p));
    }

    /// Verify the raw Debug output contains the expected `name: "` token so
    /// the normaliser can match it.
    #[test]
    fn debug_output_contains_name_token() {
        let v = crate::algebra::Variable::new("myvar").expect("valid var");
        let dbg = format!("{v:?}");
        assert!(
            dbg.contains("name: \"myvar\""),
            "Expected 'name: \"myvar\"' in Variable debug output, got: {dbg}"
        );
    }
}
