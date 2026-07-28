//! Shared N-hop SPARQL graph-pattern construction.
//!
//! Multiple retrieval code paths in this crate need a WHERE-clause fragment
//! that matches every node reachable from a starting term within `1..=hops`
//! forward steps along *any* predicate (an "N-hop neighbor expansion").
//!
//! The naive way to express "any predicate, repeated" in SPARQL 1.1 is a
//! property path such as `(:|!:){1,n}` or `!()`. Both are wrong for this
//! use case:
//!
//! - `:` / `!:` reference the *empty* prefix, which is never declared by
//!   these generated queries, so a spec-compliant parser rejects the query
//!   with an "undefined prefix" error before it ever runs.
//! - `!()` (negated property set with zero disjuncts) is valid per the
//!   SPARQL 1.1 grammar (`'!' '(' PathOneInPropertySet? ... ')'`) but this
//!   workspace's own parser (`oxirs-arq`) requires at least one disjunct
//!   inside `!( ... )`, so it also fails to parse.
//! - Property-path expressions are in any case not legal inside a
//!   `CONSTRUCT` template (`ConstructTriples` only allows `VarOrIri` / `a`
//!   as a predicate), so splicing a path expression into a CONSTRUCT
//!   template — as some earlier call sites in this crate did — produces a
//!   syntactically invalid query regardless of the path's own validity.
//!
//! [`build_forward_hop_pattern`] sidesteps all of this by expanding the
//! path into an explicit `UNION` of plain (path-free) basic graph pattern
//! chains, one per hop count from `1` to `hops`. Bare variable predicates
//! in ordinary triple patterns are universally supported, portable SPARQL
//! 1.1 — no property-path or CONSTRUCT-template restriction applies.

/// Build a SPARQL 1.1 graph-pattern fragment matching every node reachable
/// from `start_term` within `1..=hops` forward steps, binding the reached
/// node to `end_var`.
///
/// `start_term` may be a variable (`"?seed"`) or a fixed IRI term
/// (`"<http://example.org/e>"`) — it is spliced into the pattern verbatim.
/// `end_var` must be a SPARQL variable (e.g. `"?neighbor"`).
///
/// The returned fragment is a `GroupOrUnionGraphPattern`: one `{ ... }`
/// group per hop count, `UNION`-joined (with **no** extra enclosing brace
/// pair wrapped around the whole chain — a bare `{A} UNION {B} UNION {C}`
/// sequence is itself already a single, valid `GraphPatternNotTriples`
/// element per the SPARQL 1.1 grammar, safe to splice directly into a
/// caller's larger `{ ... }` block possibly followed by more triples; an
/// extra wrapping `{ ... }` around it is not only unnecessary but trips up
/// this workspace's own parser when the chain has exactly one branch,
/// producing a redundant `{ { ... } }` double-group it doesn't accept in
/// that position). Intermediate predicate/node variables are named
/// `?{predicate_prefix}1`, `?{predicate_prefix}2`, ... and
/// `?{node_prefix}1`, `?{node_prefix}2`, ... — choose prefixes that do not
/// collide with any variable name used elsewhere in the enclosing query
/// (the default callers in this crate use `"hp"` / `"hn"`, short for "hop
/// predicate" / "hop node").
///
/// `hops` is clamped to a minimum of `1` (an expansion of zero hops isn't a
/// meaningful "neighbor" query, so it is treated the same as one hop rather
/// than emitting a malformed empty pattern).
pub fn build_forward_hop_pattern(
    start_term: &str,
    end_var: &str,
    hops: usize,
    predicate_prefix: &str,
    node_prefix: &str,
) -> String {
    let hops = hops.max(1);

    let branches: Vec<String> = (1..=hops)
        .map(|hop| {
            let mut pattern = String::new();
            let mut prev = start_term.to_string();
            for step in 1..=hop {
                let next = if step == hop {
                    end_var.to_string()
                } else {
                    format!("?{node_prefix}{step}")
                };
                pattern.push_str(&format!("{prev} ?{predicate_prefix}{step} {next} . "));
                prev = next;
            }
            format!("{{ {} }}", pattern.trim_end())
        })
        .collect();

    branches.join(" UNION ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regression_single_hop_is_one_branch() {
        let pattern = build_forward_hop_pattern("?seed", "?neighbor", 1, "hp", "hn");
        // No extra enclosing brace pair around a single branch — see the
        // function doc for why the redundant `{ { ... } }` form is avoided.
        assert_eq!(pattern, "{ ?seed ?hp1 ?neighbor . }");
    }

    #[test]
    fn regression_zero_hops_clamped_to_one() {
        let zero = build_forward_hop_pattern("?seed", "?neighbor", 0, "hp", "hn");
        let one = build_forward_hop_pattern("?seed", "?neighbor", 1, "hp", "hn");
        assert_eq!(zero, one);
    }

    #[test]
    fn regression_multi_hop_has_one_branch_per_hop_count() {
        let pattern = build_forward_hop_pattern("?seed", "?neighbor", 3, "hp", "hn");
        // One branch each for 1-hop, 2-hop, 3-hop reachability.
        assert_eq!(pattern.matches("UNION").count(), 2);
        assert!(pattern.contains("?seed ?hp1 ?neighbor ."));
        assert!(pattern.contains("?seed ?hp1 ?hn1 . ?hn1 ?hp2 ?neighbor ."));
        assert!(pattern.contains("?seed ?hp1 ?hn1 . ?hn1 ?hp2 ?hn2 . ?hn2 ?hp3 ?neighbor ."));
    }

    #[test]
    fn regression_no_bogus_empty_prefix_tokens() {
        // The whole point of this helper: never emit the buggy `:`/`!:`
        // empty-prefix "any predicate" hack, nor an unsupported `!()`.
        let pattern = build_forward_hop_pattern("?seed", "?neighbor", 5, "hp", "hn");
        assert!(!pattern.contains(":|!:"));
        assert!(!pattern.contains("!("));
    }

    #[test]
    fn regression_supports_fixed_iri_start_term() {
        let pattern =
            build_forward_hop_pattern("<http://example.org/e>", "?hopnode", 2, "hp", "hn");
        assert!(pattern.contains("<http://example.org/e> ?hp1 ?hn1 ."));
        assert!(pattern.contains("?hn1 ?hp2 ?hopnode ."));
    }
}
