//! Tests for SPARQL 1.1 UPDATE graph management operations.

#[cfg(test)]
mod tests {
    use crate::update_graph_management_ops::GraphManagementExecutor;
    use crate::update_graph_management_protocol::{
        GraphManagementParser, GraphManagementRequestHandler,
    };
    use crate::update_graph_management_types::{
        GraphManagementDataset, GraphManagementOp, GraphManagementResult, GraphManagementTarget,
        Triple,
    };

    // -----------------------------------------------------------------------
    // Helper builders
    // -----------------------------------------------------------------------

    fn triple(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(s, p, o)
    }

    fn ex(local: &str) -> String {
        format!("http://example.org/{local}")
    }

    fn g1() -> String {
        ex("g1")
    }
    fn g2() -> String {
        ex("g2")
    }
    fn g3() -> String {
        ex("g3")
    }

    fn t1() -> Triple {
        triple(&ex("s1"), &ex("p1"), &ex("o1"))
    }
    fn t2() -> Triple {
        triple(&ex("s2"), &ex("p2"), &ex("o2"))
    }
    fn t3() -> Triple {
        triple(&ex("s3"), &ex("p3"), &ex("o3"))
    }

    fn dataset_with_default_triples(triples: &[Triple]) -> GraphManagementDataset {
        let mut ds = GraphManagementDataset::new();
        for t in triples {
            ds.add_triple(None, t.clone());
        }
        ds
    }

    fn dataset_with_named_triples(iri: &str, triples: &[Triple]) -> GraphManagementDataset {
        let mut ds = GraphManagementDataset::new();
        for t in triples {
            ds.add_triple(Some(iri), t.clone());
        }
        ds
    }

    // -----------------------------------------------------------------------
    // Triple / dataset basics
    // -----------------------------------------------------------------------

    #[test]
    fn test_triple_new() {
        let t = triple("s", "p", "o");
        assert_eq!(t.subject, "s");
        assert_eq!(t.predicate, "p");
        assert_eq!(t.object, "o");
    }

    #[test]
    fn test_triple_equality() {
        let a = triple("s", "p", "o");
        let b = triple("s", "p", "o");
        let c = triple("x", "p", "o");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_dataset_new_is_empty() {
        let ds = GraphManagementDataset::new();
        assert_eq!(ds.triple_count(None), 0);
        assert!(ds.graph_names().is_empty());
    }

    #[test]
    fn test_dataset_add_triple_default_graph() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(None, t1());
        assert_eq!(ds.triple_count(None), 1);
    }

    #[test]
    fn test_dataset_add_triple_named_graph() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        assert_eq!(ds.triple_count(Some(&g1())), 1);
        assert!(ds.named_graph_exists(&g1()));
    }

    #[test]
    fn test_dataset_add_triple_creates_named_graph() {
        let mut ds = GraphManagementDataset::new();
        assert!(!ds.named_graph_exists(&g1()));
        ds.add_triple(Some(&g1()), t1());
        assert!(ds.named_graph_exists(&g1()));
    }

    #[test]
    fn test_dataset_get_graph_default_empty() {
        let ds = GraphManagementDataset::new();
        assert_eq!(ds.get_graph(None).len(), 0);
    }

    #[test]
    fn test_dataset_get_graph_nonexistent_named_returns_empty() {
        let ds = GraphManagementDataset::new();
        assert_eq!(ds.get_graph(Some("http://no-such-graph")).len(), 0);
    }

    #[test]
    fn test_dataset_graph_names() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g2()), t2());
        let mut names = ds.graph_names();
        names.sort();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_dataset_triple_count_multiple() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(None, t1());
        ds.add_triple(None, t2());
        ds.add_triple(Some(&g1()), t3());
        assert_eq!(ds.triple_count(None), 2);
        assert_eq!(ds.triple_count(Some(&g1())), 1);
    }

    // -----------------------------------------------------------------------
    // CLEAR tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_clear_default_removes_all_triples() {
        let mut ds = dataset_with_default_triples(&[t1(), t2(), t3()]);
        let op = GraphManagementOp::Clear {
            target: GraphManagementTarget::Default,
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(None), 0);
        assert_eq!(result.triples_affected, 3);
        assert!(result.graphs_affected.contains(&"DEFAULT".to_owned()));
    }

    #[test]
    fn test_clear_named_graph_empties_it() {
        let mut ds = dataset_with_named_triples(&g1(), &[t1(), t2()]);
        let op = GraphManagementOp::Clear {
            target: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 0);
        // The graph itself still exists (CLEAR does not drop)
        assert!(ds.named_graph_exists(&g1()));
    }

    #[test]
    fn test_clear_named_keeps_other_graphs_intact() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g2()), t2());
        let op = GraphManagementOp::Clear {
            target: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 0);
        assert_eq!(ds.triple_count(Some(&g2())), 1);
    }

    #[test]
    fn test_clear_named_nonexistent_without_silent_errors() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Clear {
            target: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_err());
    }

    #[test]
    fn test_clear_named_nonexistent_with_silent_succeeds() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Clear {
            target: GraphManagementTarget::Named(g1()),
            silent: true,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_ok());
    }

    #[test]
    fn test_clear_all_named_removes_all_named_graphs_content() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g2()), t2());
        ds.add_triple(None, t3()); // default should remain untouched
        let op = GraphManagementOp::Clear {
            target: GraphManagementTarget::AllNamed,
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 2);
        assert_eq!(ds.triple_count(Some(&g1())), 0);
        assert_eq!(ds.triple_count(Some(&g2())), 0);
        assert_eq!(ds.triple_count(None), 1); // default untouched
    }

    #[test]
    fn test_clear_all_removes_everything() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(None, t1());
        ds.add_triple(Some(&g1()), t2());
        ds.add_triple(Some(&g2()), t3());
        let op = GraphManagementOp::Clear {
            target: GraphManagementTarget::All,
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 3);
        assert_eq!(ds.triple_count(None), 0);
        assert_eq!(ds.triple_count(Some(&g1())), 0);
        assert_eq!(ds.triple_count(Some(&g2())), 0);
    }

    #[test]
    fn test_clear_all_on_empty_dataset_is_ok() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Clear {
            target: GraphManagementTarget::All,
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 0);
    }

    #[test]
    fn test_clear_returns_graphs_affected_list() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        let op = GraphManagementOp::Clear {
            target: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert!(result.graphs_affected.contains(&g1()));
    }

    // -----------------------------------------------------------------------
    // DROP tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_drop_named_graph_removes_it() {
        let mut ds = dataset_with_named_triples(&g1(), &[t1(), t2()]);
        let op = GraphManagementOp::Drop {
            target: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert!(!ds.named_graph_exists(&g1()));
    }

    #[test]
    fn test_drop_named_graph_reports_triples_affected() {
        let mut ds = dataset_with_named_triples(&g1(), &[t1(), t2(), t3()]);
        let op = GraphManagementOp::Drop {
            target: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 3);
    }

    #[test]
    fn test_drop_silent_on_nonexistent_graph_succeeds() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Drop {
            target: GraphManagementTarget::Named(g1()),
            silent: true,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_ok());
    }

    #[test]
    fn test_drop_nonsilent_on_nonexistent_graph_errors() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Drop {
            target: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_err());
    }

    #[test]
    fn test_drop_default_clears_default_graph() {
        let mut ds = dataset_with_default_triples(&[t1(), t2()]);
        let op = GraphManagementOp::Drop {
            target: GraphManagementTarget::Default,
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(None), 0);
    }

    #[test]
    fn test_drop_all_named_removes_all_named_graphs() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g2()), t2());
        ds.add_triple(None, t3());
        let op = GraphManagementOp::Drop {
            target: GraphManagementTarget::AllNamed,
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert!(ds.graph_names().is_empty());
        assert_eq!(ds.triple_count(None), 1); // default untouched
    }

    #[test]
    fn test_drop_all_removes_default_and_named() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(None, t1());
        ds.add_triple(Some(&g1()), t2());
        let op = GraphManagementOp::Drop {
            target: GraphManagementTarget::All,
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 2);
        assert_eq!(ds.triple_count(None), 0);
        assert!(ds.graph_names().is_empty());
    }

    #[test]
    fn test_drop_named_does_not_affect_other_named_graphs() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g2()), t2());
        let op = GraphManagementOp::Drop {
            target: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert!(!ds.named_graph_exists(&g1()));
        assert!(ds.named_graph_exists(&g2()));
    }

    // -----------------------------------------------------------------------
    // CREATE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_creates_empty_named_graph() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Create {
            graph: g1(),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert!(ds.named_graph_exists(&g1()));
        assert_eq!(ds.triple_count(Some(&g1())), 0);
    }

    #[test]
    fn test_create_reports_affected_graph() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Create {
            graph: g1(),
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert!(result.graphs_affected.contains(&g1()));
    }

    #[test]
    fn test_create_existing_graph_without_silent_errors() {
        let mut ds = GraphManagementDataset::new();
        ds.named_graphs.insert(g1(), vec![]);
        let op = GraphManagementOp::Create {
            graph: g1(),
            silent: false,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_err());
    }

    #[test]
    fn test_create_silent_on_existing_graph_succeeds() {
        let mut ds = GraphManagementDataset::new();
        ds.named_graphs.insert(g1(), vec![t1()]);
        let op = GraphManagementOp::Create {
            graph: g1(),
            silent: true,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        // Content must be preserved
        assert_eq!(ds.triple_count(Some(&g1())), 1);
        assert_eq!(result.triples_affected, 0);
    }

    #[test]
    fn test_create_multiple_graphs() {
        let mut ds = GraphManagementDataset::new();
        for g in [&g1(), &g2(), &g3()] {
            GraphManagementExecutor::execute(
                &GraphManagementOp::Create {
                    graph: g.clone(),
                    silent: false,
                },
                &mut ds,
            )
            .unwrap();
        }
        assert_eq!(ds.graph_names().len(), 3);
    }

    // -----------------------------------------------------------------------
    // COPY tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_copy_named_to_named_copies_triples() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g1()), t2());
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g2())), 2);
    }

    #[test]
    fn test_copy_clears_destination_first() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g2()), t3()); // pre-existing in dest
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        // Destination should contain only what was in source
        assert_eq!(ds.triple_count(Some(&g2())), 1);
        assert_eq!(ds.get_graph(Some(&g2()))[0], t1());
    }

    #[test]
    fn test_copy_leaves_source_unchanged() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g1()), t2());
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 2);
    }

    #[test]
    fn test_copy_from_default_to_named() {
        let mut ds = dataset_with_default_triples(&[t1(), t2()]);
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Default,
            destination: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 2);
        assert_eq!(ds.triple_count(None), 2); // source unchanged
    }

    #[test]
    fn test_copy_from_named_to_default() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(None, t3()); // pre-existing default should be cleared
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Default,
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(None), 1);
        assert_eq!(ds.get_graph(None)[0], t1());
    }

    #[test]
    fn test_copy_self_to_self_is_noop() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 0);
        assert_eq!(ds.triple_count(Some(&g1())), 1); // content preserved
    }

    #[test]
    fn test_copy_nonexistent_source_without_silent_errors() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_err());
    }

    #[test]
    fn test_copy_nonexistent_source_with_silent_succeeds() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: true,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_ok());
    }

    #[test]
    fn test_copy_reports_triples_affected() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g1()), t2());
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 2);
    }

    // -----------------------------------------------------------------------
    // MOVE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_move_named_to_named_copies_then_drops_source() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g1()), t2());
        let op = GraphManagementOp::Move {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g2())), 2);
        assert!(!ds.named_graph_exists(&g1())); // source dropped
    }

    #[test]
    fn test_move_clears_destination_before_writing() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g2()), t3()); // pre-existing
        let op = GraphManagementOp::Move {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g2())), 1);
        assert_eq!(ds.get_graph(Some(&g2()))[0], t1());
    }

    #[test]
    fn test_move_from_default_to_named() {
        let mut ds = dataset_with_default_triples(&[t1(), t2()]);
        let op = GraphManagementOp::Move {
            source: GraphManagementTarget::Default,
            destination: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 2);
        assert_eq!(ds.triple_count(None), 0); // default cleared
    }

    #[test]
    fn test_move_from_named_to_default() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        let op = GraphManagementOp::Move {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Default,
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(None), 1);
        assert!(!ds.named_graph_exists(&g1()));
    }

    #[test]
    fn test_move_self_to_self_is_noop() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        let op = GraphManagementOp::Move {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 0);
        // Content preserved
        assert_eq!(ds.triple_count(Some(&g1())), 1);
    }

    #[test]
    fn test_move_nonexistent_source_without_silent_errors() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Move {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_err());
    }

    #[test]
    fn test_move_nonexistent_source_with_silent_succeeds() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Move {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: true,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_ok());
    }

    #[test]
    fn test_move_reports_triples_affected() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g1()), t2());
        ds.add_triple(Some(&g1()), t3());
        let op = GraphManagementOp::Move {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 3);
    }

    // -----------------------------------------------------------------------
    // ADD tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_merges_triples_into_destination() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g2()), t2());
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g2())), 2); // t2 + t1
    }

    #[test]
    fn test_add_does_not_clear_destination() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g2()), t2()); // pre-existing
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        // Both original and new triples must be present
        let g2_triples = ds.get_graph(Some(&g2()));
        assert!(g2_triples.contains(&t1()));
        assert!(g2_triples.contains(&t2()));
    }

    #[test]
    fn test_add_leaves_source_unchanged() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 1);
    }

    #[test]
    fn test_add_from_default_to_named() {
        let mut ds = dataset_with_default_triples(&[t1(), t2()]);
        ds.add_triple(Some(&g1()), t3());
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Default,
            destination: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 3);
        assert_eq!(ds.triple_count(None), 2); // default unchanged
    }

    #[test]
    fn test_add_from_named_to_default() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(None, t2());
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Default,
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(None), 2);
        assert_eq!(ds.triple_count(Some(&g1())), 1); // unchanged
    }

    #[test]
    fn test_add_self_to_self_is_noop() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 0);
        assert_eq!(ds.triple_count(Some(&g1())), 1);
    }

    #[test]
    fn test_add_nonexistent_source_without_silent_errors() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_err());
    }

    #[test]
    fn test_add_nonexistent_source_with_silent_succeeds() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: true,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_ok());
    }

    #[test]
    fn test_add_reports_triples_affected() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g1()), t2());
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 2);
    }

    // -----------------------------------------------------------------------
    // LOAD tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_load_without_silent_returns_error() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Load {
            iri: "http://example.org/data.ttl".to_owned(),
            into_graph: None,
            silent: false,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_err());
    }

    #[test]
    fn test_load_with_silent_returns_ok() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Load {
            iri: "http://example.org/data.ttl".to_owned(),
            into_graph: None,
            silent: true,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_ok());
    }

    #[test]
    fn test_load_silent_into_named_graph_returns_ok() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Load {
            iri: "http://example.org/data.ttl".to_owned(),
            into_graph: Some(g1()),
            silent: true,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        // Silent load is a no-op; dataset unchanged
        assert_eq!(result.triples_affected, 0);
    }

    // -----------------------------------------------------------------------
    // Composed / interaction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_then_add_then_clear_lifecycle() {
        let mut ds = GraphManagementDataset::new();

        // 1. CREATE
        GraphManagementExecutor::execute(
            &GraphManagementOp::Create {
                graph: g1(),
                silent: false,
            },
            &mut ds,
        )
        .unwrap();
        assert!(ds.named_graph_exists(&g1()));

        // 2. ADD from default (empty) to g1 — no-op effectively
        ds.add_triple(None, t1());
        GraphManagementExecutor::execute(
            &GraphManagementOp::Add {
                source: GraphManagementTarget::Default,
                destination: GraphManagementTarget::Named(g1()),
                silent: false,
            },
            &mut ds,
        )
        .unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 1);

        // 3. CLEAR g1
        GraphManagementExecutor::execute(
            &GraphManagementOp::Clear {
                target: GraphManagementTarget::Named(g1()),
                silent: false,
            },
            &mut ds,
        )
        .unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 0);
        assert!(ds.named_graph_exists(&g1())); // graph still exists after CLEAR
    }

    #[test]
    fn test_move_then_drop_leaves_clean_dataset() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g1()), t2());

        // MOVE g1 -> g2
        GraphManagementExecutor::execute(
            &GraphManagementOp::Move {
                source: GraphManagementTarget::Named(g1()),
                destination: GraphManagementTarget::Named(g2()),
                silent: false,
            },
            &mut ds,
        )
        .unwrap();

        // DROP g2
        GraphManagementExecutor::execute(
            &GraphManagementOp::Drop {
                target: GraphManagementTarget::Named(g2()),
                silent: false,
            },
            &mut ds,
        )
        .unwrap();

        assert!(ds.graph_names().is_empty());
    }

    #[test]
    fn test_copy_then_add_accumulates_duplicates() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());

        // COPY g1 -> g2
        GraphManagementExecutor::execute(
            &GraphManagementOp::Copy {
                source: GraphManagementTarget::Named(g1()),
                destination: GraphManagementTarget::Named(g2()),
                silent: false,
            },
            &mut ds,
        )
        .unwrap();

        // ADD g1 -> g2 (g2 already has the triple; duplicates are allowed in a bag model)
        GraphManagementExecutor::execute(
            &GraphManagementOp::Add {
                source: GraphManagementTarget::Named(g1()),
                destination: GraphManagementTarget::Named(g2()),
                silent: false,
            },
            &mut ds,
        )
        .unwrap();

        assert_eq!(ds.triple_count(Some(&g2())), 2);
    }

    #[test]
    fn test_graph_management_result_default() {
        let r = GraphManagementResult::default();
        assert_eq!(r.triples_affected, 0);
        assert!(r.graphs_affected.is_empty());
    }

    #[test]
    fn test_target_label_default() {
        let label = GraphManagementExecutor::target_label(&GraphManagementTarget::Default);
        assert_eq!(label, "DEFAULT");
    }

    #[test]
    fn test_target_label_named() {
        let label = GraphManagementExecutor::target_label(&GraphManagementTarget::Named(g1()));
        assert_eq!(label, g1());
    }

    #[test]
    fn test_target_label_all() {
        let label = GraphManagementExecutor::target_label(&GraphManagementTarget::All);
        assert_eq!(label, "ALL");
    }

    #[test]
    fn test_target_label_all_named() {
        let label = GraphManagementExecutor::target_label(&GraphManagementTarget::AllNamed);
        assert_eq!(label, "NAMED");
    }

    #[test]
    fn test_graph_management_target_equality() {
        assert_eq!(
            GraphManagementTarget::Default,
            GraphManagementTarget::Default
        );
        assert_eq!(
            GraphManagementTarget::Named(g1()),
            GraphManagementTarget::Named(g1())
        );
        assert_ne!(
            GraphManagementTarget::Named(g1()),
            GraphManagementTarget::Named(g2())
        );
        assert_ne!(GraphManagementTarget::All, GraphManagementTarget::AllNamed);
    }

    #[test]
    fn test_add_to_nonexistent_destination_creates_it() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        // g2 does not exist yet
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g2())), 1);
    }

    #[test]
    fn test_copy_to_nonexistent_destination_creates_it() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert!(ds.named_graph_exists(&g2()));
        assert_eq!(ds.triple_count(Some(&g2())), 1);
    }

    #[test]
    fn test_move_to_nonexistent_destination_creates_it() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        let op = GraphManagementOp::Move {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert!(ds.named_graph_exists(&g2()));
        assert!(!ds.named_graph_exists(&g1()));
    }

    #[test]
    fn test_clear_named_graph_after_multiple_adds() {
        let mut ds = GraphManagementDataset::new();
        for t in [t1(), t2(), t3()] {
            ds.add_triple(Some(&g1()), t);
        }
        GraphManagementExecutor::execute(
            &GraphManagementOp::Clear {
                target: GraphManagementTarget::Named(g1()),
                silent: false,
            },
            &mut ds,
        )
        .unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 0);
    }

    #[test]
    fn test_drop_all_named_on_empty_dataset_ok() {
        let mut ds = GraphManagementDataset::new();
        let result = GraphManagementExecutor::execute(
            &GraphManagementOp::Drop {
                target: GraphManagementTarget::AllNamed,
                silent: false,
            },
            &mut ds,
        )
        .unwrap();
        assert_eq!(result.triples_affected, 0);
    }

    // -----------------------------------------------------------------------
    // Protocol / parser tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_clear_default() {
        let op = GraphManagementParser::parse("CLEAR DEFAULT").unwrap();
        assert_eq!(
            op,
            GraphManagementOp::Clear {
                target: GraphManagementTarget::Default,
                silent: false
            }
        );
    }

    #[test]
    fn test_parse_clear_silent_graph() {
        let op =
            GraphManagementParser::parse("CLEAR SILENT GRAPH <http://example.org/g1>").unwrap();
        assert_eq!(
            op,
            GraphManagementOp::Clear {
                target: GraphManagementTarget::Named("http://example.org/g1".to_string()),
                silent: true
            }
        );
    }

    #[test]
    fn test_parse_drop_all() {
        let op = GraphManagementParser::parse("DROP ALL").unwrap();
        assert_eq!(
            op,
            GraphManagementOp::Drop {
                target: GraphManagementTarget::All,
                silent: false
            }
        );
    }

    #[test]
    fn test_parse_create_graph() {
        let op = GraphManagementParser::parse("CREATE GRAPH <http://example.org/g1>").unwrap();
        assert_eq!(
            op,
            GraphManagementOp::Create {
                graph: "http://example.org/g1".to_string(),
                silent: false
            }
        );
    }

    #[test]
    fn test_parse_load_silent() {
        let op = GraphManagementParser::parse("LOAD SILENT <http://example.org/data.ttl>").unwrap();
        assert_eq!(
            op,
            GraphManagementOp::Load {
                iri: "http://example.org/data.ttl".to_string(),
                into_graph: None,
                silent: true
            }
        );
    }

    #[test]
    fn test_parse_copy() {
        let op =
            GraphManagementParser::parse("COPY <http://example.org/g1> TO <http://example.org/g2>")
                .unwrap();
        assert_eq!(
            op,
            GraphManagementOp::Copy {
                source: GraphManagementTarget::Named("http://example.org/g1".to_string()),
                destination: GraphManagementTarget::Named("http://example.org/g2".to_string()),
                silent: false
            }
        );
    }

    #[test]
    fn test_parse_move_default_to_named() {
        let op = GraphManagementParser::parse("MOVE DEFAULT TO <http://example.org/g1>").unwrap();
        assert_eq!(
            op,
            GraphManagementOp::Move {
                source: GraphManagementTarget::Default,
                destination: GraphManagementTarget::Named("http://example.org/g1".to_string()),
                silent: false
            }
        );
    }

    #[test]
    fn test_parse_add_with_graph_keyword() {
        let op = GraphManagementParser::parse(
            "ADD GRAPH <http://example.org/g1> TO GRAPH <http://example.org/g2>",
        )
        .unwrap();
        assert_eq!(
            op,
            GraphManagementOp::Add {
                source: GraphManagementTarget::Named("http://example.org/g1".to_string()),
                destination: GraphManagementTarget::Named("http://example.org/g2".to_string()),
                silent: false
            }
        );
    }

    #[test]
    fn test_parse_unknown_keyword_errors() {
        assert!(GraphManagementParser::parse("SELECT ?s WHERE { ?s ?p ?o }").is_err());
    }

    #[test]
    fn test_parse_empty_input_errors() {
        assert!(GraphManagementParser::parse("").is_err());
    }

    #[test]
    fn test_http_handler_success() {
        let mut ds = GraphManagementDataset::new();
        let resp = GraphManagementRequestHandler::handle("CLEAR DEFAULT", &mut ds);
        assert!(resp.is_success());
        assert_eq!(resp.status_code, 200);
    }

    #[test]
    fn test_http_handler_parse_error() {
        let mut ds = GraphManagementDataset::new();
        let resp = GraphManagementRequestHandler::handle("INVALID COMMAND", &mut ds);
        assert!(!resp.is_success());
        assert_eq!(resp.status_code, 400);
    }

    #[test]
    fn test_http_handler_runtime_error() {
        let mut ds = GraphManagementDataset::new();
        // Dropping a non-existent graph (not silent) should give 500
        let resp = GraphManagementRequestHandler::handle(
            "DROP GRAPH <http://example.org/no-such-graph>",
            &mut ds,
        );
        assert!(!resp.is_success());
        assert_eq!(resp.status_code, 500);
    }
}
