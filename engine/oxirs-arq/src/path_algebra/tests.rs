use std::collections::{HashMap, HashSet};

use super::ast::{NpsItem, PathDirection, PropertyPath};
use super::evaluator::PathAlgebraEvaluator;

/// Build an evaluator backed by an in-memory triple store.
///
/// `triples`: list of `(subject, predicate, object)` string tuples.
fn make_evaluator(
    triples: Vec<(&'static str, &'static str, &'static str)>,
) -> PathAlgebraEvaluator {
    let store: Vec<(String, String, String)> = triples
        .into_iter()
        .map(|(s, p, o)| (s.to_string(), p.to_string(), o.to_string()))
        .collect();

    let store_triple = store.clone();
    let store_fwd_pred = store.clone();
    let store_rev_pred = store.clone();
    let store_neighbours = store.clone();

    PathAlgebraEvaluator::new(
        move |s, p, o| {
            store_triple
                .iter()
                .any(|(ts, tp, to)| ts == s && tp == p && to == o)
        },
        move |node| {
            store_fwd_pred
                .iter()
                .filter(|(s, _, _)| s == node)
                .map(|(_, p, _)| p.clone())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect()
        },
        move |node| {
            store_rev_pred
                .iter()
                .filter(|(_, _, o)| o == node)
                .map(|(_, p, _)| p.clone())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect()
        },
        move |node, pred, dir| match dir {
            super::ast::PathDirection::Forward => store_neighbours
                .iter()
                .filter(|(s, p, _)| s == node && p == pred)
                .map(|(_, _, o)| o.clone())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect(),
            super::ast::PathDirection::Backward => store_neighbours
                .iter()
                .filter(|(_, p, o)| o == node && p == pred)
                .map(|(s, _, _)| s.clone())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect(),
        },
    )
}

fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort_unstable();
    v
}

// ── Link traversal ─────────────────────────────────────────────────────────

#[test]
fn test_link_forward_single_result() {
    let ev = make_evaluator(vec![("a", "knows", "b")]);
    let path = PropertyPath::Link("knows".into());
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(result, vec!["b"]);
}

#[test]
fn test_link_forward_no_result() {
    let ev = make_evaluator(vec![("a", "knows", "b")]);
    let path = PropertyPath::Link("hates".into());
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_link_forward_multiple_objects() {
    let ev = make_evaluator(vec![("a", "knows", "b"), ("a", "knows", "c")]);
    let path = PropertyPath::Link("knows".into());
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(sorted(result), vec!["b", "c"]);
}

#[test]
fn test_link_backward() {
    // inverse traversal: from "b" backward over "knows" should return "a"
    let ev = make_evaluator(vec![("a", "knows", "b")]);
    let path = PropertyPath::Link("knows".into());
    let result = ev.evaluate("b", &path, PathDirection::Backward).unwrap();
    assert_eq!(result, vec!["a"]);
}

// ── Inverse path ────────────────────────────────────────────────────────────

#[test]
fn test_inverse_path_from_object() {
    let ev = make_evaluator(vec![("a", "knows", "b")]);
    let path = PropertyPath::Inverse(Box::new(PropertyPath::Link("knows".into())));
    let result = ev.evaluate("b", &path, PathDirection::Forward).unwrap();
    assert_eq!(result, vec!["a"]);
}

#[test]
fn test_inverse_path_chain() {
    let ev = make_evaluator(vec![("a", "knows", "b"), ("b", "knows", "c")]);
    let path = PropertyPath::Inverse(Box::new(PropertyPath::Link("knows".into())));
    let result = ev.evaluate("c", &path, PathDirection::Forward).unwrap();
    assert_eq!(result, vec!["b"]);
}

// ── Sequence (a/b) ─────────────────────────────────────────────────────────

#[test]
fn test_sequence_two_steps() {
    let ev = make_evaluator(vec![("a", "p", "b"), ("b", "q", "c")]);
    let path = PropertyPath::Sequence(
        Box::new(PropertyPath::Link("p".into())),
        Box::new(PropertyPath::Link("q".into())),
    );
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(result, vec!["c"]);
}

#[test]
fn test_sequence_no_connection() {
    let ev = make_evaluator(vec![("a", "p", "b"), ("x", "q", "c")]);
    let path = PropertyPath::Sequence(
        Box::new(PropertyPath::Link("p".into())),
        Box::new(PropertyPath::Link("q".into())),
    );
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_sequence_fan_out() {
    let ev = make_evaluator(vec![
        ("a", "p", "b"),
        ("a", "p", "c"),
        ("b", "q", "d"),
        ("c", "q", "e"),
    ]);
    let path = PropertyPath::Sequence(
        Box::new(PropertyPath::Link("p".into())),
        Box::new(PropertyPath::Link("q".into())),
    );
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(sorted(result), vec!["d", "e"]);
}

// ── Alternative (a|b) ──────────────────────────────────────────────────────

#[test]
fn test_alternative_both_present() {
    let ev = make_evaluator(vec![("a", "p", "b"), ("a", "q", "c")]);
    let path = PropertyPath::Alternative(
        Box::new(PropertyPath::Link("p".into())),
        Box::new(PropertyPath::Link("q".into())),
    );
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(sorted(result), vec!["b", "c"]);
}

#[test]
fn test_alternative_only_left() {
    let ev = make_evaluator(vec![("a", "p", "b")]);
    let path = PropertyPath::Alternative(
        Box::new(PropertyPath::Link("p".into())),
        Box::new(PropertyPath::Link("q".into())),
    );
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(result, vec!["b"]);
}

#[test]
fn test_alternative_deduplicates() {
    let ev = make_evaluator(vec![("a", "p", "b"), ("a", "q", "b")]);
    let path = PropertyPath::Alternative(
        Box::new(PropertyPath::Link("p".into())),
        Box::new(PropertyPath::Link("q".into())),
    );
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(result, vec!["b"]);
}

// ── ZeroOrMore (a*) ────────────────────────────────────────────────────────

#[test]
fn test_zero_or_more_includes_start() {
    let ev = make_evaluator(vec![("a", "p", "b"), ("b", "p", "c")]);
    let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Link("p".into())));
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(sorted(result), vec!["a", "b", "c"]);
}

#[test]
fn test_zero_or_more_no_outgoing_edges() {
    let ev = make_evaluator(vec![]);
    let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Link("p".into())));
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(result, vec!["a"]);
}

#[test]
fn test_zero_or_more_cycle_detection() {
    let ev = make_evaluator(vec![("a", "p", "b"), ("b", "p", "c"), ("c", "p", "a")]);
    let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Link("p".into())));
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(sorted(result), vec!["a", "b", "c"]);
}

#[test]
fn test_zero_or_more_diamond() {
    let ev = make_evaluator(vec![
        ("a", "p", "b"),
        ("a", "p", "c"),
        ("b", "p", "d"),
        ("c", "p", "d"),
    ]);
    let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Link("p".into())));
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(sorted(result), vec!["a", "b", "c", "d"]);
}

// ── OneOrMore (a+) ─────────────────────────────────────────────────────────

#[test]
fn test_one_or_more_excludes_start() {
    let ev = make_evaluator(vec![("a", "p", "b"), ("b", "p", "c")]);
    let path = PropertyPath::OneOrMore(Box::new(PropertyPath::Link("p".into())));
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert!(!result.contains(&"a".to_string()));
    assert_eq!(sorted(result), vec!["b", "c"]);
}

#[test]
fn test_one_or_more_no_steps_empty() {
    let ev = make_evaluator(vec![]);
    let path = PropertyPath::OneOrMore(Box::new(PropertyPath::Link("p".into())));
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_one_or_more_cycle_detection() {
    let ev = make_evaluator(vec![("a", "p", "b"), ("b", "p", "a")]);
    let path = PropertyPath::OneOrMore(Box::new(PropertyPath::Link("p".into())));
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(sorted(result), vec!["b"]);
}

// ── Optional (a?) ──────────────────────────────────────────────────────────

#[test]
fn test_optional_with_match() {
    let ev = make_evaluator(vec![("a", "p", "b")]);
    let path = PropertyPath::Optional(Box::new(PropertyPath::Link("p".into())));
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(sorted(result), vec!["a", "b"]);
}

#[test]
fn test_optional_no_match_returns_start() {
    let ev = make_evaluator(vec![]);
    let path = PropertyPath::Optional(Box::new(PropertyPath::Link("p".into())));
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(result, vec!["a"]);
}

// ── NegatedPropertySet (!(a|b|^c)) ─────────────────────────────────────────

#[test]
fn test_nps_blocks_forward_predicate() {
    let ev = make_evaluator(vec![("a", "knows", "b"), ("a", "likes", "c")]);
    let nps = vec![NpsItem::Forward("knows".into())];
    let path = PropertyPath::NegatedPropertySet(nps);
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(result, vec!["c"]);
}

#[test]
fn test_nps_blocks_all_forward_predicates() {
    let ev = make_evaluator(vec![("a", "knows", "b"), ("a", "likes", "c")]);
    let nps = vec![
        NpsItem::Forward("knows".into()),
        NpsItem::Forward("likes".into()),
    ];
    let path = PropertyPath::NegatedPropertySet(nps);
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_nps_inverse_item_blocked_backward() {
    let ev = make_evaluator(vec![("x", "knows", "a"), ("y", "likes", "a")]);
    let nps = vec![NpsItem::Inverse("knows".into())];
    let path = PropertyPath::NegatedPropertySet(nps);
    let result = ev.evaluate("a", &path, PathDirection::Backward).unwrap();
    assert_eq!(result, vec!["y"]);
}

#[test]
fn test_nps_empty_blocks_nothing() {
    let ev = make_evaluator(vec![("a", "p", "b"), ("a", "q", "c")]);
    let path = PropertyPath::NegatedPropertySet(vec![]);
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(sorted(result), vec!["b", "c"]);
}

// ── Nested combinations ────────────────────────────────────────────────────

#[test]
fn test_nested_sequence_with_alternative() {
    // (p|q)/r
    let ev = make_evaluator(vec![
        ("a", "p", "b"),
        ("a", "q", "c"),
        ("b", "r", "d"),
        ("c", "r", "e"),
    ]);
    let path = PropertyPath::Sequence(
        Box::new(PropertyPath::Alternative(
            Box::new(PropertyPath::Link("p".into())),
            Box::new(PropertyPath::Link("q".into())),
        )),
        Box::new(PropertyPath::Link("r".into())),
    );
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(sorted(result), vec!["d", "e"]);
}

#[test]
fn test_nested_inverse_in_sequence() {
    // p/^q: first forward over p, then backward over q
    let ev = make_evaluator(vec![("a", "p", "b"), ("c", "q", "b")]);
    let path = PropertyPath::Sequence(
        Box::new(PropertyPath::Link("p".into())),
        Box::new(PropertyPath::Inverse(Box::new(PropertyPath::Link(
            "q".into(),
        )))),
    );
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(result, vec!["c"]);
}

#[test]
fn test_zero_or_more_with_sequence_inside() {
    // (p/q)* starting from a
    let ev = make_evaluator(vec![
        ("a", "p", "m"),
        ("m", "q", "b"),
        ("b", "p", "n"),
        ("n", "q", "c"),
    ]);
    let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Sequence(
        Box::new(PropertyPath::Link("p".into())),
        Box::new(PropertyPath::Link("q".into())),
    )));
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(sorted(result), vec!["a", "b", "c"]);
}

#[test]
fn test_deep_chain_one_or_more() {
    // Build a chain: a→b→c→d→e via predicate "next"
    let ev = make_evaluator(vec![
        ("a", "next", "b"),
        ("b", "next", "c"),
        ("c", "next", "d"),
        ("d", "next", "e"),
    ]);
    let path = PropertyPath::OneOrMore(Box::new(PropertyPath::Link("next".into())));
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(sorted(result), vec!["b", "c", "d", "e"]);
}

#[test]
fn test_alternative_in_zero_or_more() {
    // (p|q)* from a
    let ev = make_evaluator(vec![("a", "p", "b"), ("b", "q", "c"), ("c", "p", "d")]);
    let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Alternative(
        Box::new(PropertyPath::Link("p".into())),
        Box::new(PropertyPath::Link("q".into())),
    )));
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    assert_eq!(sorted(result), vec!["a", "b", "c", "d"]);
}

#[test]
fn test_optional_inside_sequence() {
    // p/q? — p then optional q
    let ev = make_evaluator(vec![("a", "p", "b"), ("a", "p", "c"), ("b", "q", "d")]);
    let path = PropertyPath::Sequence(
        Box::new(PropertyPath::Link("p".into())),
        Box::new(PropertyPath::Optional(Box::new(PropertyPath::Link(
            "q".into(),
        )))),
    );
    let result = ev.evaluate("a", &path, PathDirection::Forward).unwrap();
    // b→d and b itself (optional), c itself (no q from c)
    assert_eq!(sorted(result), vec!["b", "c", "d"]);
}

#[test]
fn test_multiple_inverse_in_alternative() {
    // (^p | ^q) from c — finds subjects that have p or q pointing to c
    let ev = make_evaluator(vec![("a", "p", "c"), ("b", "q", "c"), ("x", "r", "c")]);
    let path = PropertyPath::Alternative(
        Box::new(PropertyPath::Inverse(Box::new(PropertyPath::Link(
            "p".into(),
        )))),
        Box::new(PropertyPath::Inverse(Box::new(PropertyPath::Link(
            "q".into(),
        )))),
    );
    let result = ev.evaluate("c", &path, PathDirection::Forward).unwrap();
    assert_eq!(sorted(result), vec!["a", "b"]);
}

/// Verify the evaluator handles a large linear graph without stack overflow.
#[test]
fn test_large_linear_graph_no_stack_overflow() {
    let n = 500usize;
    let mut triples = Vec::with_capacity(n);
    for i in 0..n {
        triples.push((format!("n{}", i), "next".to_string(), format!("n{}", i + 1)));
    }
    let store: HashMap<(String, String), Vec<String>> = {
        let mut m: HashMap<(String, String), Vec<String>> = HashMap::new();
        for (s, p, o) in &triples {
            m.entry((s.clone(), p.clone())).or_default().push(o.clone());
        }
        m
    };
    let store_triple = store.clone();
    let store_fwd = store.clone();
    let store_rev = store.clone();
    let store_nb = store.clone();
    let ev = PathAlgebraEvaluator::new(
        move |s, p, o| {
            store_triple
                .get(&(s.to_string(), p.to_string()))
                .is_some_and(|v| v.contains(&o.to_string()))
        },
        move |node| {
            store_fwd
                .keys()
                .filter(|(s, _)| s == node)
                .map(|(_, p)| p.clone())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect()
        },
        move |node| {
            // In the large-graph test store, values are forward neighbours only.
            // Reverse lookup: check all (s, p) pairs whose value list contains node.
            store_rev
                .iter()
                .filter(|(_, v)| v.contains(&node.to_string()))
                .map(|((_, p), _)| p.clone())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect()
        },
        move |node, p, dir| match dir {
            super::ast::PathDirection::Forward => store_nb
                .get(&(node.to_string(), p.to_string()))
                .cloned()
                .unwrap_or_default(),
            super::ast::PathDirection::Backward => store_nb
                .iter()
                .filter(|((_, pred), v)| pred == p && v.contains(&node.to_string()))
                .map(|((s, _), _)| s.clone())
                .collect(),
        },
    );

    let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Link("next".into())));
    let result = ev.evaluate("n0", &path, PathDirection::Forward).unwrap();
    assert_eq!(result.len(), n + 1);
}
