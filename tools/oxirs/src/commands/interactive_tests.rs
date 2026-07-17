//! Tests for the interactive REPL mode

#[cfg(test)]
mod tests {
    use crate::commands::interactive_session::{is_query_complete, validate_sparql_syntax};

    #[test]
    fn test_query_complete_simple() {
        assert!(is_query_complete("SELECT * WHERE { ?s ?p ?o }"));
        assert!(is_query_complete(
            "ASK { ?s a <http://example.org/Person> }"
        ));
        assert!(is_query_complete(
            "PREFIX ex: <http://example.org/> SELECT * WHERE { ?s ?p ?o }"
        ));
    }

    #[test]
    fn test_query_incomplete_braces() {
        assert!(!is_query_complete("SELECT * WHERE {"));
        assert!(!is_query_complete("SELECT * WHERE { ?s ?p ?o"));
        assert!(!is_query_complete("SELECT * WHERE { ?s { ?p ?o }"));
    }

    #[test]
    fn test_query_complete_nested_braces() {
        assert!(is_query_complete(
            "SELECT * WHERE { { ?s ?p ?o } UNION { ?a ?b ?c } }"
        ));
        assert!(is_query_complete(
            "SELECT * WHERE { GRAPH <g> { ?s ?p ?o } }"
        ));
    }

    #[test]
    fn test_query_incomplete_quotes() {
        assert!(!is_query_complete("SELECT * WHERE { ?s ?p \"unclosed"));
        assert!(!is_query_complete("SELECT * WHERE { ?s ?p 'unclosed"));
    }

    #[test]
    fn test_query_complete_quotes() {
        assert!(is_query_complete("SELECT * WHERE { ?s ?p \"value\" }"));
        assert!(is_query_complete("SELECT * WHERE { ?s ?p 'value' }"));
        assert!(is_query_complete(
            r#"SELECT * WHERE { ?s ?p "value with \"escaped\" quotes" }"#
        ));
    }

    #[test]
    fn test_query_complete_triple_quotes() {
        assert!(is_query_complete(
            r#"SELECT * WHERE { ?s ?p """triple quoted value""" }"#
        ));
        assert!(is_query_complete(
            r#"SELECT * WHERE { ?s ?p '''triple quoted value''' }"#
        ));
    }

    #[test]
    fn test_query_incomplete_triple_quotes() {
        assert!(!is_query_complete(r#"SELECT * WHERE { ?s ?p """unclosed"#));
        assert!(!is_query_complete(r#"SELECT * WHERE { ?s ?p '''unclosed"#));
    }

    #[test]
    fn test_query_complete_brackets() {
        assert!(is_query_complete("SELECT * WHERE { ?s [ ?p ?o ] }"));
        assert!(is_query_complete("SELECT * WHERE { [ ?p ?o ] ?p2 ?o2 }"));
    }

    #[test]
    fn test_query_incomplete_brackets() {
        assert!(!is_query_complete("SELECT * WHERE { ?s [ ?p ?o }"));
        assert!(!is_query_complete("SELECT * WHERE { [ ?p ?o ?p2 ?o2 }"));
    }

    #[test]
    fn test_query_complete_parentheses() {
        assert!(is_query_complete("SELECT * WHERE { FILTER (1 + 2) }"));
        assert!(is_query_complete(
            "SELECT * WHERE { BIND ((1 + 2) AS ?sum) }"
        ));
    }

    #[test]
    fn test_query_incomplete_parentheses() {
        assert!(!is_query_complete("SELECT * WHERE { FILTER (1 + 2 }"));
        assert!(!is_query_complete(
            "SELECT * WHERE { BIND ((1 + 2 AS ?sum) }"
        ));
    }

    #[test]
    fn test_query_continuation_backslash() {
        assert!(!is_query_complete("SELECT * WHERE { ?s ?p ?o } \\"));
        assert!(!is_query_complete("PREFIX ex: <http://example.org/> \\"));
    }

    #[test]
    fn test_query_empty() {
        assert!(!is_query_complete(""));
        assert!(!is_query_complete("   "));
        assert!(!is_query_complete("\n\n"));
    }

    #[test]
    fn test_query_complex_multiline() {
        let query = r#"SELECT ?name ?email WHERE {
            ?person foaf:name ?name .
            ?person foaf:mbox ?email
        }"#;
        assert!(is_query_complete(query));
    }

    #[test]
    fn test_query_with_comments() {
        let query = "SELECT * WHERE { # This is a comment\n ?s ?p ?o }";
        assert!(is_query_complete(query));
    }

    #[test]
    fn test_query_braces_in_strings() {
        assert!(is_query_complete(
            r#"SELECT * WHERE { ?s ?p "value with { braces }" }"#
        ));
        assert!(is_query_complete(
            r#"SELECT * WHERE { ?s ?p "value with ( parens )" }"#
        ));
        assert!(is_query_complete(
            r#"SELECT * WHERE { ?s ?p "value with [ brackets ]" }"#
        ));
    }

    #[test]
    fn test_syntax_validation_valid_query() {
        let query = "SELECT * WHERE { ?s ?p ?o }";
        let hints = validate_sparql_syntax(query);
        assert!(hints.is_empty());
    }

    #[test]
    fn test_syntax_validation_missing_where() {
        let query = "SELECT * { ?s ?p ?o }";
        let hints = validate_sparql_syntax(query);
        assert!(!hints.is_empty());
        assert!(hints.iter().any(|h| h.contains("WHERE")));
    }

    #[test]
    fn test_syntax_validation_missing_prefix() {
        let query = "SELECT * WHERE { ?s rdf:type ?o }";
        let hints = validate_sparql_syntax(query);
        assert!(hints.iter().any(|h| h.contains("PREFIX rdf:")));
    }

    #[test]
    fn test_syntax_validation_with_prefix() {
        let query = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\nSELECT * WHERE { ?s rdf:type ?o }";
        let hints = validate_sparql_syntax(query);
        assert!(hints.iter().all(|h| !h.contains("PREFIX rdf:")));
    }

    #[test]
    fn test_syntax_validation_multiple_prefixes() {
        let query = "SELECT * WHERE { ?s rdf:type ?o . ?s foaf:name ?name }";
        let hints = validate_sparql_syntax(query);
        assert!(hints.len() >= 2); // Should suggest both rdf and foaf prefixes
    }

    #[test]
    fn test_syntax_validation_ask_query() {
        let query = "ASK { ?s ?p ?o }";
        let hints = validate_sparql_syntax(query);
        assert!(hints.is_empty());
    }

    #[test]
    fn test_syntax_validation_filter_syntax() {
        let query = "SELECT * WHERE { ?s ?p ?o FILTER ?o > 10 }";
        let hints = validate_sparql_syntax(query);
        assert!(hints.iter().any(|h| h.contains("FILTER")));
    }
}

/// Non-TTY dispatch tests for the `:`-prefixed REPL extension commands.
///
/// Each test drives [`dispatch_colon_command`] directly (never the readline
/// loop) so parsing, routing, and the resulting [`ReplState`] mutations are
/// verified without a terminal. Every test builds its store and persistence in
/// a unique directory under [`std::env::temp_dir`] — never a hardcoded path.
#[cfg(test)]
mod repl_dispatch_tests {
    use crate::cli::ascii_diagram::DiagramTriple;
    use crate::cli::formatters::{Binding, QueryResults as FormatterQueryResults, RdfTerm};
    use crate::cli::repl_commands::{dispatch_colon_command, ColonOutcome, ReplState};
    use oxirs_core::rdf_store::RdfStore;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, RwLock};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    /// Create a unique, isolated temp directory for one test.
    fn unique_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut dir = std::env::temp_dir();
        dir.push(format!("oxirs_repl_{tag}_{nanos}_{n}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// Build a [`ReplState`] backed by a fresh empty store in a unique temp dir.
    ///
    /// Returns the state and its base directory (bookmarks persist to
    /// `base/bookmarks.json`).
    fn make_state(tag: &str) -> (ReplState, PathBuf) {
        let base = unique_dir(tag);
        let store_dir = base.join("store");
        std::fs::create_dir_all(&store_dir).expect("create store dir");
        let store = Arc::new(RwLock::new(RdfStore::open(&store_dir).expect("open store")));
        let state = ReplState::new(
            store,
            "default".to_string(),
            store_dir.to_string_lossy().to_string(),
            &base,
        )
        .expect("build repl state");
        (state, base)
    }

    /// A minimal SELECT-shaped result usable as the REPL's "last result".
    fn sample_results() -> FormatterQueryResults {
        FormatterQueryResults {
            variables: vec!["s".to_string(), "name".to_string()],
            bindings: vec![Binding {
                values: vec![
                    Some(RdfTerm::Uri {
                        value: "http://example.org/alice".to_string(),
                    }),
                    Some(RdfTerm::Literal {
                        value: "Alice".to_string(),
                        lang: None,
                        datatype: None,
                    }),
                ],
            }],
        }
    }

    #[test]
    fn dispatch_unknown_command_is_unknown() {
        let (mut state, _base) = make_state("unknown");
        let outcome = dispatch_colon_command(&mut state, &[], ":frobnicate now");
        assert!(matches!(outcome, ColonOutcome::Unknown));
    }

    #[test]
    fn dispatch_help_is_handled() {
        let (mut state, _base) = make_state("help");
        assert!(matches!(
            dispatch_colon_command(&mut state, &[], ":help"),
            ColonOutcome::Handled
        ));
    }

    #[test]
    fn bookmark_add_list_run_rm_roundtrip() {
        let (mut state, base) = make_state("bookmark");

        // add: parsed as name + query, saved and persisted.
        let out = dispatch_colon_command(
            &mut state,
            &[],
            ":bookmark add q1 SELECT * WHERE { ?s ?p ?o }",
        );
        assert!(matches!(out, ColonOutcome::Handled));
        assert_eq!(state.bookmarks.list().len(), 1);
        assert_eq!(state.bookmarks.list()[0].name, "q1");
        assert!(
            base.join("bookmarks.json").exists(),
            "bookmark should persist beside session state"
        );

        // list: pure read, handled.
        assert!(matches!(
            dispatch_colon_command(&mut state, &[], ":bookmark list"),
            ColonOutcome::Handled
        ));

        // run: injects the stored query back into the execution path.
        match dispatch_colon_command(&mut state, &[], ":bookmark run q1") {
            ColonOutcome::Inject(query) => assert!(query.contains("SELECT"), "query: {query}"),
            other => panic!("expected Inject, got {other:?}"),
        }

        // rm: removes the bookmark.
        assert!(matches!(
            dispatch_colon_command(&mut state, &[], ":bookmark rm q1"),
            ColonOutcome::Handled
        ));
        assert!(state.bookmarks.list().is_empty());
    }

    #[test]
    fn bookmark_run_missing_does_not_inject() {
        let (mut state, _base) = make_state("bm_missing");
        assert!(matches!(
            dispatch_colon_command(&mut state, &[], ":bookmark run nope"),
            ColonOutcome::Handled
        ));
    }

    #[test]
    fn bookmark_add_requires_name_and_query() {
        let (mut state, _base) = make_state("bm_usage");
        // Only a name, no query -> usage message, nothing saved.
        assert!(matches!(
            dispatch_colon_command(&mut state, &[], ":bookmark add onlyname"),
            ColonOutcome::Handled
        ));
        assert!(state.bookmarks.list().is_empty());
    }

    #[test]
    fn export_last_result_writes_csv() {
        let (mut state, base) = make_state("export_csv");
        state.last_result = Some(sample_results());
        let out_path = base.join("out.csv");
        let cmd = format!(":export csv {}", out_path.display());
        assert!(matches!(
            dispatch_colon_command(&mut state, &[], &cmd),
            ColonOutcome::Handled
        ));
        assert!(out_path.exists(), "export should create the file");
        let content = std::fs::read_to_string(&out_path).expect("read csv");
        assert!(content.contains("Alice"), "csv content: {content}");
    }

    #[test]
    fn export_without_result_creates_no_file() {
        let (mut state, base) = make_state("export_none");
        let out_path = base.join("none.csv");
        let cmd = format!(":export csv {}", out_path.display());
        assert!(matches!(
            dispatch_colon_command(&mut state, &[], &cmd),
            ColonOutcome::Handled
        ));
        assert!(!out_path.exists(), "no last result -> no file written");
    }

    #[test]
    fn export_unknown_format_creates_no_file() {
        let (mut state, base) = make_state("export_bad");
        state.last_result = Some(sample_results());
        let out_path = base.join("bad.out");
        let cmd = format!(":export toml {}", out_path.display());
        assert!(matches!(
            dispatch_colon_command(&mut state, &[], &cmd),
            ColonOutcome::Handled
        ));
        assert!(!out_path.exists(), "unknown format -> no file written");
    }

    #[test]
    fn diagram_from_last_graph_is_handled() {
        let (mut state, _base) = make_state("diagram_graph");
        state.last_graph = Some(vec![DiagramTriple {
            subject: "http://example.org/alice".to_string(),
            predicate: "http://xmlns.com/foaf/0.1/knows".to_string(),
            object: "http://example.org/bob".to_string(),
        }]);
        for style in ["", " tree", " graph", " compact", " list"] {
            let cmd = format!(":diagram{style}");
            assert!(
                matches!(
                    dispatch_colon_command(&mut state, &[], &cmd),
                    ColonOutcome::Handled
                ),
                "style {style:?} should be handled"
            );
        }
        // Unknown style is reported but still handled.
        assert!(matches!(
            dispatch_colon_command(&mut state, &[], ":diagram spiral"),
            ColonOutcome::Handled
        ));
    }

    #[test]
    fn diagram_falls_back_to_store_sample() {
        let (mut state, _base) = make_state("diagram_sample");
        // Populate the active store so the fallback sampler has real triples.
        {
            let mut guard = state.store.write().expect("store write");
            guard
                .insert_string_triple(
                    "http://example.org/alice",
                    "http://xmlns.com/foaf/0.1/name",
                    "Alice",
                )
                .expect("insert triple");
        }
        assert!(state.last_graph.is_none());
        // No last graph -> handler samples the dataset instead of failing.
        assert!(matches!(
            dispatch_colon_command(&mut state, &[], ":diagram"),
            ColonOutcome::Handled
        ));
    }

    #[test]
    fn dataset_add_use_list_switches_active() {
        let (mut state, base) = make_state("dataset");
        // A second real dataset directory (empty store is created on open).
        let ds2 = base.join("ds2");
        std::fs::create_dir_all(&ds2).expect("mk ds2");

        let add_cmd = format!(":dataset add ds2 {}", ds2.display());
        assert!(matches!(
            dispatch_colon_command(&mut state, &[], &add_cmd),
            ColonOutcome::Handled
        ));
        assert!(state.datasets.get("ds2").is_some());

        assert!(matches!(
            dispatch_colon_command(&mut state, &[], ":dataset list"),
            ColonOutcome::Handled
        ));

        // Switching opens the new store and reports StoreChanged so the loop
        // can refresh schema completion.
        assert!(matches!(
            dispatch_colon_command(&mut state, &[], ":dataset use ds2"),
            ColonOutcome::StoreChanged
        ));
        assert_eq!(state.active_dataset, "ds2");
    }

    #[test]
    fn dataset_use_unknown_leaves_active_unchanged() {
        let (mut state, _base) = make_state("dataset_missing");
        assert!(matches!(
            dispatch_colon_command(&mut state, &[], ":dataset use nope"),
            ColonOutcome::Handled
        ));
        assert_eq!(state.active_dataset, "default");
    }

    #[test]
    fn hsearch_matches_history() {
        let (mut state, _base) = make_state("hsearch");
        let history = vec![
            "SELECT * WHERE { ?s ?p ?o }".to_string(),
            "ASK { ?x a ?y }".to_string(),
        ];
        assert!(matches!(
            dispatch_colon_command(&mut state, &history, ":hsearch SELECT"),
            ColonOutcome::Handled
        ));
        // Empty term -> usage message, still handled.
        assert!(matches!(
            dispatch_colon_command(&mut state, &history, ":hsearch"),
            ColonOutcome::Handled
        ));
    }

    #[test]
    fn visual_without_stdin_is_cancelled_and_handled() {
        let (mut state, _base) = make_state("visual");
        // Under nextest the test process has no interactive stdin, so the
        // builder reads EOF, cancels, and the command is Handled (never Inject).
        assert!(matches!(
            dispatch_colon_command(&mut state, &[], ":visual"),
            ColonOutcome::Handled
        ));
    }
}
