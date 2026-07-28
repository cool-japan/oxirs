//! # QueryExecutor - apply_order_by_group Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Solution;

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Apply order by to solution
    pub(super) fn apply_order_by(
        &self,
        mut solution: Solution,
        conditions: &[crate::algebra::OrderCondition],
    ) -> Solution {
        if conditions.is_empty() {
            return solution;
        }
        solution.sort_by(|a, b| {
            for condition in conditions {
                let val_a = self.evaluate_order_expression(&condition.expr, a);
                let val_b = self.evaluate_order_expression(&condition.expr, b);
                let cmp = match (val_a, val_b) {
                    (
                        Some(crate::algebra::Term::Literal(lit_a)),
                        Some(crate::algebra::Term::Literal(lit_b)),
                    ) => compare_literals(&lit_a, &lit_b),
                    (
                        Some(crate::algebra::Term::Iri(iri_a)),
                        Some(crate::algebra::Term::Iri(iri_b)),
                    ) => iri_a.as_str().cmp(iri_b.as_str()),
                    (Some(a), Some(b)) => format!("{a}").cmp(&format!("{b}")),
                    // SPARQL 1.1 §15.1: unbound (or errored) keys sort lowest;
                    // `cmp.reverse()` below then yields unbound-last for DESC.
                    (Some(_), None) => std::cmp::Ordering::Greater,
                    (None, Some(_)) => std::cmp::Ordering::Less,
                    (None, None) => std::cmp::Ordering::Equal,
                };
                let result = if condition.ascending {
                    cmp
                } else {
                    cmp.reverse()
                };
                if result != std::cmp::Ordering::Equal {
                    return result;
                }
            }
            std::cmp::Ordering::Equal
        });
        solution
    }
    /// Evaluate an expression for ordering.
    ///
    /// Delegates to the general expression evaluator so expression keys
    /// (`ORDER BY STR(?s)`, arithmetic, …) are actually computed; an
    /// evaluation error makes the key unbound, which sorts lowest per
    /// SPARQL 1.1 §15.1 (same fail-soft contract as `apply_group_by`'s
    /// expression keys).
    pub(super) fn evaluate_order_expression(
        &self,
        expr: &crate::algebra::Expression,
        binding: &std::collections::HashMap<crate::algebra::Variable, crate::algebra::Term>,
    ) -> Option<crate::algebra::Term> {
        self.evaluate_expression(expr, binding).ok()
    }
}

/// Compare two literals with a single composite key applied uniformly:
/// recognized-numeric literals (partition 0, by value) sort before all other
/// literals (partition 1, by lexical form). A per-pair "both parse as f64?"
/// branch is not transitive (`5 < 10` numerically but `"10" < "3abc" < "5"`
/// lexically forms a cycle) and corrupts `sort_by`'s merge.
///
/// Partition 0 orders by the composite (f64 value, exact decimal value, byte
/// order): the exact-value middle step keeps integers beyond 2^53 — which
/// collide in f64 — numerically ordered, and each step is a total preorder on
/// every partition-0 lexical, so the whole key stays transitive.
pub(crate) fn compare_literals(
    lit_a: &crate::algebra::Literal,
    lit_b: &crate::algebra::Literal,
) -> std::cmp::Ordering {
    let (grp_a, num_a, str_a) = literal_sort_key(lit_a);
    let (grp_b, num_b, str_b) = literal_sort_key(lit_b);
    grp_a.cmp(&grp_b).then_with(|| {
        if grp_a == 0 {
            num_a
                .partial_cmp(&num_b)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| exact_numeric_cmp(str_a, str_b))
                .then_with(|| str_a.cmp(str_b))
        } else {
            str_a.cmp(str_b)
        }
    })
}

/// Exact numeric comparison of two XSD numeric lexical forms
/// (`[sign] digits [. digits] [eE [sign] digits]`), with no precision loss.
/// Inputs are known to parse as finite f64 (partition-0 gate), so a parse
/// failure here is unreachable; it degrades to `Equal` (byte order decides).
fn exact_numeric_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    match (decompose_numeric(a), decompose_numeric(b)) {
        (Some(ka), Some(kb)) => ka.cmp_value(&kb),
        _ => std::cmp::Ordering::Equal,
    }
}

/// A numeric lexical normalized to sign × 0.{digits} × 10^exp, with leading
/// and trailing zeros stripped from `digits` (empty digits == zero).
struct DecimalKey {
    negative: bool,
    digits: String,
    exp: i64,
}

impl DecimalKey {
    fn cmp_value(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        let zero_a = self.digits.is_empty();
        let zero_b = other.digits.is_empty();
        match (zero_a, zero_b) {
            (true, true) => return Ordering::Equal,
            (true, false) => {
                return if other.negative {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
            (false, true) => {
                return if self.negative {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
            (false, false) => {}
        }
        match (self.negative, other.negative) {
            (true, false) => return Ordering::Less,
            (false, true) => return Ordering::Greater,
            _ => {}
        }
        // Same sign, both non-zero: larger decimal exponent wins; at equal
        // exponent compare digit strings right-padded to a common length.
        let magnitude = self.exp.cmp(&other.exp).then_with(|| {
            let len = self.digits.len().max(other.digits.len());
            let pad = |d: &str| {
                let mut s = d.to_string();
                while s.len() < len {
                    s.push('0');
                }
                s
            };
            pad(&self.digits).cmp(&pad(&other.digits))
        });
        if self.negative {
            magnitude.reverse()
        } else {
            magnitude
        }
    }
}

fn decompose_numeric(s: &str) -> Option<DecimalKey> {
    let (negative, rest) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let (mantissa, exp_part) = match rest.split_once(['e', 'E']) {
        Some((m, e)) => (m, Some(e)),
        None => (rest, None),
    };
    let exp_shift: i64 = match exp_part {
        Some(e) => e.parse().ok()?,
        None => 0,
    };
    let (int_part, frac_part) = match mantissa.split_once('.') {
        Some((i, f)) => (i, f),
        None => (mantissa, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return None;
    }
    if !int_part.bytes().all(|c| c.is_ascii_digit())
        || !frac_part.bytes().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    // Normalize to 0.{digits} × 10^exp: strip leading zeros off the integer
    // part (adjusting exp), then trailing zeros off the fraction tail.
    let mut digits = String::with_capacity(int_part.len() + frac_part.len());
    digits.push_str(int_part);
    digits.push_str(frac_part);
    let stripped_lead = digits.len() - digits.trim_start_matches('0').len();
    let mut exp = int_part.len() as i64 - stripped_lead as i64 + exp_shift;
    let mut digits: String = digits.trim_start_matches('0').to_string();
    let trimmed_len = digits.trim_end_matches('0').len();
    digits.truncate(trimmed_len);
    if digits.is_empty() {
        exp = 0;
    }
    Some(DecimalKey {
        negative,
        digits,
        exp,
    })
}

fn literal_sort_key(lit: &crate::algebra::Literal) -> (u8, f64, &str) {
    let is_numeric_dt = matches!(
        lit.datatype.as_ref().map(|d| d.as_str()),
        Some("http://www.w3.org/2001/XMLSchema#integer")
            | Some("http://www.w3.org/2001/XMLSchema#decimal")
            | Some("http://www.w3.org/2001/XMLSchema#float")
            | Some("http://www.w3.org/2001/XMLSchema#double")
            | Some("http://www.w3.org/2001/XMLSchema#long")
            | Some("http://www.w3.org/2001/XMLSchema#int")
            | Some("http://www.w3.org/2001/XMLSchema#short")
            | Some("http://www.w3.org/2001/XMLSchema#byte")
            | Some("http://www.w3.org/2001/XMLSchema#nonNegativeInteger")
            | Some("http://www.w3.org/2001/XMLSchema#positiveInteger")
            | Some("http://www.w3.org/2001/XMLSchema#nonPositiveInteger")
            | Some("http://www.w3.org/2001/XMLSchema#negativeInteger")
            | Some("http://www.w3.org/2001/XMLSchema#unsignedLong")
            | Some("http://www.w3.org/2001/XMLSchema#unsignedInt")
            | Some("http://www.w3.org/2001/XMLSchema#unsignedShort")
            | Some("http://www.w3.org/2001/XMLSchema#unsignedByte")
    );
    if is_numeric_dt {
        if let Ok(n) = lit.value.parse::<f64>() {
            // "NaN"^^xsd:double parses but has no total order; keep it in the
            // lexical partition so the comparator stays transitive.
            if !n.is_nan() {
                return (0, n, lit.value.as_str());
            }
        }
    }
    (1, 0.0, lit.value.as_str())
}
