//! # Rule Index Tests
//!
//! Comprehensive test suite for rule indexing: lookup, statistics, dependency graph,
//! priority ordering, wildcard matching, and incremental re-indexing.

use crate::rule_index_store::{wildcard_matches, PriorityIndex, RuleIndex, RuleIndexBuilder};
use crate::rule_index_types::IndexConfig;
use crate::{Rule, RuleAtom, Term};

fn create_test_rule(name: &str, pred: &str) -> Rule {
    Rule {
        name: name.to_string(),
        body: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant(pred.to_string()),
            object: Term::Variable("Y".to_string()),
        }],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant(format!("inferred_{pred}")),
            object: Term::Variable("Y".to_string()),
        }],
    }
}

#[test]
fn test_add_and_find_rule() {
    let index = RuleIndex::with_defaults();
    let rule = create_test_rule("rule1", "parent");

    let id = index.add_rule(rule);
    assert_eq!(id, 0);

    let found = index.find_rules_for_triple(None, "parent", None);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0], 0);
}

#[test]
fn test_multiple_rules_same_predicate() {
    let index = RuleIndex::with_defaults();

    index.add_rule(create_test_rule("rule1", "parent"));
    index.add_rule(create_test_rule("rule2", "parent"));
    index.add_rule(create_test_rule("rule3", "child"));

    let parent_rules = index.find_rules_for_triple(None, "parent", None);
    assert_eq!(parent_rules.len(), 2);

    let child_rules = index.find_rules_for_triple(None, "child", None);
    assert_eq!(child_rules.len(), 1);
}

#[test]
fn test_remove_rule() {
    let index = RuleIndex::with_defaults();

    let id1 = index.add_rule(create_test_rule("rule1", "parent"));
    let id2 = index.add_rule(create_test_rule("rule2", "parent"));

    assert_eq!(index.rule_count(), 2);

    let removed = index.remove_rule(id1);
    assert!(removed.is_some());
    assert_eq!(removed.expect("already checked is_some").name, "rule1");

    assert_eq!(index.rule_count(), 1);

    let found = index.find_rules_for_triple(None, "parent", None);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0], id2);
}

#[test]
fn test_statistics() {
    let config = IndexConfig::default().with_statistics(true);
    let index = RuleIndex::new(config);

    index.add_rule(create_test_rule("rule1", "parent"));
    index.add_rule(create_test_rule("rule2", "child"));

    // Perform lookups
    index.find_rules_for_triple(None, "parent", None);
    index.find_rules_for_triple(None, "child", None);
    index.find_rules_for_triple(None, "unknown", None);

    let stats = index.statistics_snapshot();
    assert_eq!(stats.total_lookups, 3);
    assert!(stats.predicate_hits >= 2); // At least 2 hits for known predicates
}

#[test]
fn test_clear() {
    let index = RuleIndex::with_defaults();

    index.add_rule(create_test_rule("rule1", "parent"));
    index.add_rule(create_test_rule("rule2", "child"));

    assert_eq!(index.rule_count(), 2);

    index.clear();

    assert_eq!(index.rule_count(), 0);
}

#[test]
fn test_builder() {
    let index = RuleIndexBuilder::new()
        .config(IndexConfig::default().with_combined_indexing(true))
        .add_rule(create_test_rule("rule1", "parent"))
        .add_rule(create_test_rule("rule2", "child"))
        .build();

    assert_eq!(index.rule_count(), 2);
}

#[test]
fn test_get_rule() {
    let index = RuleIndex::with_defaults();
    let rule = create_test_rule("rule1", "parent");

    let id = index.add_rule(rule.clone());

    let retrieved = index.get_rule(id);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.expect("already checked is_some").name, "rule1");

    let not_found = index.get_rule(999);
    assert!(not_found.is_none());
}

#[test]
fn test_get_all_rules() {
    let index = RuleIndex::with_defaults();

    index.add_rule(create_test_rule("rule1", "parent"));
    index.add_rule(create_test_rule("rule2", "child"));
    index.add_rule(create_test_rule("rule3", "sibling"));

    let all = index.get_all_rules();
    assert_eq!(all.len(), 3);
}

#[test]
fn test_memory_usage() {
    let index = RuleIndex::with_defaults();

    for i in 0..100 {
        index.add_rule(create_test_rule(
            &format!("rule{i}"),
            &format!("pred{}", i % 10),
        ));
    }

    let usage = index.memory_usage();
    assert!(usage > 0);
}

#[test]
fn test_combined_key_indexing() {
    let index = RuleIndex::new(IndexConfig::default().with_combined_indexing(true));

    // Rule with constant subject
    let rule_with_const = Rule {
        name: "const_subject".to_string(),
        body: vec![RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("knows".to_string()),
            object: Term::Variable("Y".to_string()),
        }],
        head: vec![RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("friend".to_string()),
            object: Term::Variable("Y".to_string()),
        }],
    };

    index.add_rule(rule_with_const);

    // Should find with specific subject
    let found = index.find_rules_for_triple(Some("john"), "knows", None);
    assert_eq!(found.len(), 1);
}

#[test]
fn test_first_arg_indexing() {
    let config = IndexConfig::default()
        .with_predicate_indexing(true)
        .with_first_arg_indexing(true);
    let index = RuleIndex::new(config);

    index.add_rule(create_test_rule("rule1", "parent"));

    // Lookup should work
    let found = index.find_rules_for_triple(None, "parent", None);
    assert!(!found.is_empty());
}

#[test]
fn test_statistics_reset() {
    let index = RuleIndex::with_defaults();
    index.add_rule(create_test_rule("rule1", "parent"));

    index.find_rules_for_triple(None, "parent", None);

    let stats_before = index.statistics_snapshot();
    assert!(stats_before.total_lookups > 0);

    index.reset_statistics();

    let stats_after = index.statistics_snapshot();
    assert_eq!(stats_after.total_lookups, 0);
}

#[test]
fn test_hit_rates() {
    let index = RuleIndex::with_defaults();
    index.add_rule(create_test_rule("rule1", "parent"));
    index.add_rule(create_test_rule("rule2", "child"));

    // Make some lookups
    for _ in 0..10 {
        index.find_rules_for_triple(None, "parent", None);
    }

    let stats = index.statistics();
    assert!(stats.predicate_hit_rate() > 0.0);
}

#[test]
fn test_builtin_indexing() {
    let index = RuleIndex::with_defaults();

    let rule = Rule {
        name: "builtin_rule".to_string(),
        body: vec![
            RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("hasAge".to_string()),
                object: Term::Variable("Age".to_string()),
            },
            RuleAtom::Builtin {
                name: "greaterThan".to_string(),
                args: vec![
                    Term::Variable("Age".to_string()),
                    Term::Constant("18".to_string()),
                ],
            },
        ],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("isAdult".to_string()),
            object: Term::Constant("true".to_string()),
        }],
    };

    index.add_rule(rule);

    let found = index.find_rules_for_triple(None, "hasAge", None);
    assert_eq!(found.len(), 1);
}

// ---------------------------------------------------------------------------
// Dependency graph tests
// ---------------------------------------------------------------------------

fn create_chain_rule(name: &str, body_pred: &str, head_pred: &str) -> Rule {
    Rule {
        name: name.to_string(),
        body: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant(body_pred.to_string()),
            object: Term::Variable("Y".to_string()),
        }],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant(head_pred.to_string()),
            object: Term::Variable("Y".to_string()),
        }],
    }
}

#[test]
fn test_dependency_graph_basic() {
    let index = RuleIndex::with_defaults();
    // r1: parent -> ancestor
    index.add_rule(create_chain_rule("r1", "parent", "ancestor"));
    // r2: ancestor -> reachable
    index.add_rule(create_chain_rule("r2", "ancestor", "reachable"));

    let graph = index.dependency_graph();
    assert_eq!(graph.edge_count(), 1);
    // r1 triggers r2
    let triggered = graph.triggered_by(0);
    assert_eq!(triggered.len(), 1);
    assert_eq!(triggered[0], 1);
}

#[test]
fn test_dependency_graph_triggers_of() {
    let index = RuleIndex::with_defaults();
    index.add_rule(create_chain_rule("r1", "parent", "ancestor"));
    index.add_rule(create_chain_rule("r2", "ancestor", "reachable"));

    let graph = index.dependency_graph();
    let triggers = graph.triggers_of(1);
    assert_eq!(triggers.len(), 1);
    assert_eq!(triggers[0], 0);
}

#[test]
fn test_dependency_graph_roots() {
    let index = RuleIndex::with_defaults();
    index.add_rule(create_chain_rule("r1", "parent", "ancestor"));
    index.add_rule(create_chain_rule("r2", "ancestor", "reachable"));
    index.add_rule(create_chain_rule("r3", "reachable", "connected"));

    let graph = index.dependency_graph();
    let roots = graph.roots();
    assert_eq!(roots.len(), 1);
    assert_eq!(roots[0], 0);
}

#[test]
fn test_dependency_graph_no_cycle() {
    let index = RuleIndex::with_defaults();
    index.add_rule(create_chain_rule("r1", "parent", "ancestor"));
    index.add_rule(create_chain_rule("r2", "ancestor", "reachable"));

    let graph = index.dependency_graph();
    assert!(!graph.has_cycle());
}

#[test]
fn test_dependency_graph_cycle() {
    let index = RuleIndex::with_defaults();
    index.add_rule(create_chain_rule("r1", "a", "b"));
    index.add_rule(create_chain_rule("r2", "b", "a"));

    let graph = index.dependency_graph();
    assert!(graph.has_cycle());
}

#[test]
fn test_dependency_graph_topological_order() {
    let index = RuleIndex::with_defaults();
    index.add_rule(create_chain_rule("r1", "parent", "ancestor"));
    index.add_rule(create_chain_rule("r2", "ancestor", "reachable"));
    index.add_rule(create_chain_rule("r3", "reachable", "connected"));

    let graph = index.dependency_graph();
    let topo = graph.topological_order();
    assert_eq!(topo.len(), 3);
    // r1 (0) should come before r2 (1), r2 before r3 (2)
    let pos_r1 = topo.iter().position(|&x| x == 0).expect("r1 in topo");
    let pos_r2 = topo.iter().position(|&x| x == 1).expect("r2 in topo");
    let pos_r3 = topo.iter().position(|&x| x == 2).expect("r3 in topo");
    assert!(pos_r1 < pos_r2);
    assert!(pos_r2 < pos_r3);
}

#[test]
fn test_dependency_graph_topological_cycle_empty() {
    let index = RuleIndex::with_defaults();
    index.add_rule(create_chain_rule("r1", "a", "b"));
    index.add_rule(create_chain_rule("r2", "b", "a"));

    let graph = index.dependency_graph();
    let topo = graph.topological_order();
    assert!(topo.is_empty()); // cycle detected
}

#[test]
fn test_dependency_graph_empty_index() {
    let index = RuleIndex::with_defaults();
    let graph = index.dependency_graph();
    assert_eq!(graph.edge_count(), 0);
    assert!(graph.roots().is_empty());
    assert!(!graph.has_cycle());
}

#[test]
fn test_dependency_graph_self_no_cycle() {
    // A rule that doesn't reference itself
    let index = RuleIndex::with_defaults();
    index.add_rule(create_test_rule("r1", "parent"));
    let graph = index.dependency_graph();
    assert_eq!(graph.edge_count(), 0);
}

#[test]
fn test_dependency_graph_edges() {
    let index = RuleIndex::with_defaults();
    index.add_rule(create_chain_rule("r1", "parent", "ancestor"));
    index.add_rule(create_chain_rule("r2", "ancestor", "reachable"));

    let graph = index.dependency_graph();
    let edges = graph.edges();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].from, 0);
    assert_eq!(edges[0].to, 1);
    assert_eq!(edges[0].predicate, "ancestor");
}

// ---------------------------------------------------------------------------
// Priority index tests
// ---------------------------------------------------------------------------

#[test]
fn test_priority_set_and_get() {
    let mut pi = PriorityIndex::new();
    pi.set_priority(0, 10);
    pi.set_priority(1, 5);
    assert_eq!(pi.get_priority(0), 10);
    assert_eq!(pi.get_priority(1), 5);
    assert_eq!(pi.get_priority(99), 0); // default
}

#[test]
fn test_priority_sort() {
    let mut pi = PriorityIndex::new();
    pi.set_priority(0, 1);
    pi.set_priority(1, 10);
    pi.set_priority(2, 5);

    let mut ids = vec![0, 1, 2];
    pi.sort_by_priority(&mut ids);
    assert_eq!(ids, vec![1, 2, 0]); // 10, 5, 1
}

#[test]
fn test_priority_ordered_rules() {
    let index = RuleIndex::with_defaults();
    index.add_rule(create_test_rule("low", "p1"));
    index.add_rule(create_test_rule("high", "p2"));
    index.add_rule(create_test_rule("mid", "p3"));

    let mut pi = PriorityIndex::new();
    pi.set_priority(0, 1);
    pi.set_priority(1, 100);
    pi.set_priority(2, 50);

    let ordered = pi.ordered_rules(&index);
    assert_eq!(ordered.len(), 3);
    assert_eq!(ordered[0].name, "high");
    assert_eq!(ordered[1].name, "mid");
    assert_eq!(ordered[2].name, "low");
}

#[test]
fn test_priority_remove() {
    let mut pi = PriorityIndex::new();
    pi.set_priority(0, 10);
    assert_eq!(pi.len(), 1);

    pi.remove_priority(0);
    assert_eq!(pi.len(), 0);
    assert_eq!(pi.get_priority(0), 0);
}

#[test]
fn test_priority_is_empty() {
    let pi = PriorityIndex::new();
    assert!(pi.is_empty());
}

#[test]
fn test_priority_clear() {
    let mut pi = PriorityIndex::new();
    pi.set_priority(0, 10);
    pi.set_priority(1, 20);
    pi.clear();
    assert!(pi.is_empty());
}

// ---------------------------------------------------------------------------
// Wildcard matching tests
// ---------------------------------------------------------------------------

#[test]
fn test_wildcard_star() {
    assert!(wildcard_matches("*", "anything"));
    assert!(wildcard_matches("*", ""));
}

#[test]
fn test_wildcard_exact() {
    assert!(wildcard_matches("parent", "parent"));
    assert!(!wildcard_matches("parent", "child"));
}

#[test]
fn test_wildcard_prefix() {
    assert!(wildcard_matches("par*", "parent"));
    assert!(!wildcard_matches("par*", "child"));
}

#[test]
fn test_wildcard_suffix() {
    assert!(wildcard_matches("*ent", "parent"));
    assert!(!wildcard_matches("*ent", "child"));
}

#[test]
fn test_wildcard_middle() {
    assert!(wildcard_matches("p*t", "parent"));
    assert!(wildcard_matches("p*t", "pet"));
    assert!(!wildcard_matches("p*t", "patch")); // doesn't end with t
}

#[test]
fn test_wildcard_find_rules() {
    let index = RuleIndex::with_defaults();
    index.add_rule(create_test_rule("r1", "parent"));
    index.add_rule(create_test_rule("r2", "child"));
    index.add_rule(create_test_rule("r3", "partner"));

    let found = index.find_rules_by_wildcard("par*");
    assert_eq!(found.len(), 2); // parent and partner
}

#[test]
fn test_wildcard_star_all() {
    let index = RuleIndex::with_defaults();
    index.add_rule(create_test_rule("r1", "parent"));
    index.add_rule(create_test_rule("r2", "child"));

    let found = index.find_rules_by_wildcard("*");
    assert_eq!(found.len(), 2);
}

// ---------------------------------------------------------------------------
// Incremental re-indexing tests
// ---------------------------------------------------------------------------

#[test]
fn test_reindex_rule() {
    let index = RuleIndex::with_defaults();
    let id = index.add_rule(create_test_rule("r1", "parent"));
    assert!(index.reindex_rule(id));
    let found = index.find_rules_for_triple(None, "parent", None);
    assert_eq!(found.len(), 1);
}

#[test]
fn test_reindex_invalid_id() {
    let index = RuleIndex::with_defaults();
    assert!(!index.reindex_rule(999));
}

#[test]
fn test_replace_rule() {
    let index = RuleIndex::with_defaults();
    let id = index.add_rule(create_test_rule("r1", "parent"));

    let old = index.replace_rule(id, create_test_rule("r1_new", "child"));
    assert!(old.is_some());
    assert_eq!(old.expect("old rule").name, "r1");

    // The stored rule should be updated
    let retrieved = index.get_rule(id);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.expect("rule exists").name, "r1_new");

    // New predicate should be indexed via the predicate index
    let found_child = index.find_rules_for_triple(None, "child", None);
    assert!(found_child.contains(&id));

    // Verify old predicate is removed from predicate index
    let found_by_pred = index.find_rules_by_predicate("parent");
    assert!(found_by_pred.is_empty());
}

#[test]
fn test_replace_rule_invalid_id() {
    let index = RuleIndex::with_defaults();
    let result = index.replace_rule(999, create_test_rule("r1", "parent"));
    assert!(result.is_none());
}

#[test]
fn test_predicate_density() {
    let index = RuleIndex::with_defaults();
    assert_eq!(index.predicate_density(), 0.0);

    index.add_rule(create_test_rule("r1", "parent"));
    index.add_rule(create_test_rule("r2", "parent"));
    index.add_rule(create_test_rule("r3", "child"));

    let density = index.predicate_density();
    // 3 rule entries across 2 predicates => density = 1.5
    assert!(density > 1.0);
}

// ---------------------------------------------------------------------------
// Multiple adds via add_rules
// ---------------------------------------------------------------------------

#[test]
fn test_add_rules_batch() {
    let index = RuleIndex::with_defaults();
    let ids = index.add_rules(vec![
        create_test_rule("r1", "parent"),
        create_test_rule("r2", "child"),
        create_test_rule("r3", "sibling"),
    ]);
    assert_eq!(ids.len(), 3);
    assert_eq!(index.rule_count(), 3);
}

// ---------------------------------------------------------------------------
// Remove non-existent rule
// ---------------------------------------------------------------------------

#[test]
fn test_remove_nonexistent_rule() {
    let index = RuleIndex::with_defaults();
    let result = index.remove_rule(999);
    assert!(result.is_none());
}
