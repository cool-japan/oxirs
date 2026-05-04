//! Integration tests for the JenaText full-text SPARQL integration.
//!
//! Requires the `text-search` Cargo feature.
//!
//! Test coverage:
//! - Index and search single document
//! - Correct subject IRI is returned
//! - No match returns empty result set
//! - max_results limit is respected
//! - Predicate filter narrows results
//! - num_docs tracks indexed documents
//! - Multiple subjects ranked by score
//! - Commit and reload consistency
//! - property function IRI is in Jena namespace
//! - Case-insensitive search (tantivy's default English analyzer)
//! - PropertyFunction build validation
//! - PropertyFunction execute with variable subject
//! - PropertyFunction execute with bound subject (filter mode)
//! - PropertyFunction execute with predicate filter argument
//! - PropertyFunction execute with maxResults argument

#[cfg(feature = "text-search")]
mod tests {
    use std::sync::Arc;

    use oxirs_arq::algebra::{Iri, Literal, Term as AlgebraTerm, Variable};
    use oxirs_arq::executor::InMemoryDataset;
    use oxirs_arq::property_functions::{
        PropFuncArg, PropertyFunction, PropertyFunctionContext, PropertyFunctionRegistry,
    };
    use oxirs_arq::text_search::{
        register_text_query, TextQueryPropertyFunction, TextSearchIndex, TEXT_NAMESPACE,
        TEXT_QUERY_IRI,
    };

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// Build and populate a test index with a handful of RDF triples.
    fn build_test_index() -> Arc<TextSearchIndex> {
        let index = TextSearchIndex::new_in_memory().expect("create index");
        index
            .index_triple(
                "http://example.org/doc1",
                "http://example.org/label",
                "semantic web technologies",
            )
            .expect("index triple 1");
        index
            .index_triple(
                "http://example.org/doc2",
                "http://example.org/label",
                "knowledge graphs and ontologies",
            )
            .expect("index triple 2");
        index
            .index_triple(
                "http://example.org/doc3",
                "http://example.org/title",
                "semantic search engines",
            )
            .expect("index triple 3");
        index
            .index_triple(
                "http://example.org/doc4",
                "http://example.org/label",
                "linked data and the web",
            )
            .expect("index triple 4");
        index.commit().expect("commit");
        Arc::new(index)
    }

    fn make_literal_arg(s: &str) -> PropFuncArg {
        PropFuncArg::list(vec![AlgebraTerm::Literal(Literal {
            value: s.to_string(),
            language: None,
            datatype: None,
        })])
    }

    fn make_context() -> PropertyFunctionContext {
        let dataset: Arc<dyn oxirs_arq::executor::Dataset> = Arc::new(InMemoryDataset::new());
        PropertyFunctionContext::new(dataset)
    }

    fn var_subject() -> PropFuncArg {
        PropFuncArg::node(AlgebraTerm::Variable(Variable::new("doc").expect("var")))
    }

    fn iri_subject(iri: &str) -> PropFuncArg {
        PropFuncArg::node(AlgebraTerm::Iri(Iri::new(iri).expect("iri")))
    }

    // -------------------------------------------------------------------------
    // TextSearchIndex unit tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_index_and_search_single_doc() {
        let index = TextSearchIndex::new_in_memory().expect("create index");
        index
            .index_triple(
                "http://example.org/s1",
                "http://example.org/p",
                "Rust programming language",
            )
            .expect("index");
        index.commit().expect("commit");

        let results = index.search("Rust", 10).expect("search");
        assert!(!results.is_empty(), "should find at least one result");
    }

    #[test]
    fn test_search_returns_correct_subject() {
        let index = build_test_index();
        let results = index.search("semantic", 10).expect("search");
        assert!(!results.is_empty());

        let subjects: Vec<&str> = results.iter().map(|r| r.subject_iri.as_str()).collect();
        assert!(
            subjects.contains(&"http://example.org/doc1")
                || subjects.contains(&"http://example.org/doc3"),
            "expected doc1 or doc3 but got {:?}",
            subjects
        );
    }

    #[test]
    fn test_search_no_match_returns_empty() {
        let index = build_test_index();
        let results = index.search("xyzzy_no_match_term", 10).expect("search");
        assert!(results.is_empty(), "expected no results for nonsense query");
    }

    #[test]
    fn test_search_max_results_respected() {
        let index = TextSearchIndex::new_in_memory().expect("create");
        for i in 0..20_u32 {
            index
                .index_triple(
                    &format!("http://example.org/s{i}"),
                    "http://example.org/p",
                    &format!("common keyword document number {i}"),
                )
                .expect("index");
        }
        index.commit().expect("commit");

        let results = index.search("common keyword", 5).expect("search");
        assert!(
            results.len() <= 5,
            "expected at most 5 results, got {}",
            results.len()
        );
    }

    #[test]
    fn test_search_predicate_filter() {
        let index = build_test_index();

        // doc1 and doc2 have label predicate with "semantic"/"knowledge" etc.
        // doc3 has title predicate. Only label should be returned when filtering by label.
        let label_pred = "http://example.org/label";
        let results = index
            .search_predicate("semantic", label_pred, 10)
            .expect("search_predicate");

        for r in &results {
            assert_eq!(
                r.predicate_iri, label_pred,
                "all results should have label predicate"
            );
        }
        // doc3 (title predicate) should NOT appear
        let subjects: Vec<&str> = results.iter().map(|r| r.subject_iri.as_str()).collect();
        assert!(
            !subjects.contains(&"http://example.org/doc3"),
            "doc3 has title predicate; should be excluded"
        );
    }

    #[test]
    fn test_num_docs_tracks_inserts() {
        let index = TextSearchIndex::new_in_memory().expect("create");
        assert_eq!(index.num_docs(), 0, "empty index has 0 docs");

        index
            .index_triple("http://example.org/s1", "http://p", "hello world")
            .expect("index");
        index.commit().expect("commit");
        assert_eq!(index.num_docs(), 1);

        index
            .index_triple("http://example.org/s2", "http://p", "goodbye world")
            .expect("index");
        index.commit().expect("commit");
        assert_eq!(index.num_docs(), 2);
    }

    #[test]
    fn test_search_multiple_subjects_ranked_by_score() {
        let index = TextSearchIndex::new_in_memory().expect("create");
        // Index several docs so we reliably get multiple results
        for i in 0..5_u32 {
            index
                .index_triple(
                    &format!("http://example.org/doc_{i}"),
                    "http://p",
                    &format!("semantic web document number {i} about linked data"),
                )
                .expect("index");
        }
        index.commit().expect("commit");

        let results = index.search("semantic web", 10).expect("search");
        assert!(
            results.len() >= 2,
            "expected at least 2 results from 5 docs, got {}",
            results.len()
        );
        // Results should be in non-increasing score order
        let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
        for w in scores.windows(2) {
            assert!(
                w[0] >= w[1],
                "results should be in non-increasing score order: {:?}",
                scores
            );
        }
    }

    #[test]
    fn test_commit_and_reload() {
        let index = TextSearchIndex::new_in_memory().expect("create");

        // Before commit: no results
        let before = index.search("hello", 10).expect("search before");
        assert!(before.is_empty(), "should be empty before commit");

        index
            .index_triple("http://example.org/s", "http://p", "hello tantivy world")
            .expect("index");
        // Still empty until commit
        let before_commit = index.search("hello", 10).expect("search before commit");
        assert!(
            before_commit.is_empty(),
            "should be empty before explicit commit"
        );

        index.commit().expect("commit");
        let after = index.search("hello", 10).expect("search after");
        assert!(!after.is_empty(), "should find results after commit");
    }

    #[test]
    fn test_search_case_insensitive() {
        let index = TextSearchIndex::new_in_memory().expect("create");
        index
            .index_triple(
                "http://example.org/s",
                "http://p",
                "Semantic Web Technologies",
            )
            .expect("index");
        index.commit().expect("commit");

        // Tantivy's default English tokenizer lowercases terms
        let results = index.search("semantic", 10).expect("search lower");
        assert!(
            !results.is_empty(),
            "lowercase query should match mixed-case literal"
        );

        let results_upper = index.search("SEMANTIC", 10).expect("search upper");
        assert!(
            !results_upper.is_empty(),
            "uppercase query should also match (tantivy lowercases query terms)"
        );
    }

    // -------------------------------------------------------------------------
    // PropertyFunction integration tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_property_fn_iri_is_jena_namespace() {
        assert_eq!(
            TextQueryPropertyFunction::iri(),
            "http://jena.apache.org/text#query"
        );
        assert!(
            TextQueryPropertyFunction::iri().starts_with(TEXT_NAMESPACE),
            "IRI should be in the Jena text namespace"
        );
        assert_eq!(TEXT_QUERY_IRI, "http://jena.apache.org/text#query");
    }

    #[test]
    fn test_property_fn_build_validates_predicate() {
        let index = build_test_index();
        let pf = TextQueryPropertyFunction::new(index);
        let ctx = make_context();

        let obj = make_literal_arg("semantic");
        let subject = var_subject();

        // Correct predicate: ok
        let ok = pf.build(&subject, TEXT_QUERY_IRI, &obj, &ctx);
        assert!(ok.is_ok(), "build with correct predicate should succeed");

        // Wrong predicate: error
        let err = pf.build(&subject, "http://example.org/wrong", &obj, &ctx);
        assert!(err.is_err(), "build with wrong predicate should fail");
    }

    #[test]
    fn test_property_fn_execute_variable_subject() {
        let index = build_test_index();
        let pf = TextQueryPropertyFunction::new(index);
        let ctx = make_context();

        let result = pf
            .execute(
                &var_subject(),
                TEXT_QUERY_IRI,
                &make_literal_arg("semantic"),
                &ctx,
            )
            .expect("execute");

        let solutions = result.into_solutions();
        assert!(!solutions.is_empty(), "should produce at least one binding");

        // Each solution should bind ?doc to an IRI
        let doc_var = Variable::new("doc").expect("var");
        for sol in &solutions {
            assert!(
                sol.contains_key(&doc_var),
                "solution should contain ?doc binding"
            );
            assert!(
                matches!(sol.get(&doc_var), Some(AlgebraTerm::Iri(_))),
                "?doc should be bound to an IRI"
            );
        }
    }

    #[test]
    fn test_property_fn_execute_bound_subject_filter_mode() {
        let index = build_test_index();
        let pf = TextQueryPropertyFunction::new(index);
        let ctx = make_context();

        // doc1 contains "semantic"
        let found = pf
            .execute(
                &iri_subject("http://example.org/doc1"),
                TEXT_QUERY_IRI,
                &make_literal_arg("semantic"),
                &ctx,
            )
            .expect("execute bound hit");
        assert!(
            !found.into_solutions().is_empty(),
            "doc1 should match 'semantic'"
        );

        // doc2 contains "knowledge" not "semantic" in label
        let not_found = pf
            .execute(
                &iri_subject("http://example.org/doc2"),
                TEXT_QUERY_IRI,
                &make_literal_arg("semantic"),
                &ctx,
            )
            .expect("execute bound miss");
        // doc2 does NOT have "semantic" in its literal — should return empty
        assert!(
            not_found.into_solutions().is_empty(),
            "doc2 should not match 'semantic'"
        );
    }

    #[test]
    fn test_property_fn_execute_with_max_results() {
        let index = TextSearchIndex::new_in_memory().expect("create");
        for i in 0..15_u32 {
            index
                .index_triple(
                    &format!("http://example.org/doc{i}"),
                    "http://p",
                    &format!("common search result number {i}"),
                )
                .expect("index");
        }
        index.commit().expect("commit");
        let index = Arc::new(index);

        let pf = TextQueryPropertyFunction::new(Arc::clone(&index));
        let ctx = make_context();

        // (queryString, maxResults=3)
        let obj = PropFuncArg::list(vec![
            AlgebraTerm::Literal(Literal {
                value: "common search".to_string(),
                language: None,
                datatype: None,
            }),
            AlgebraTerm::Literal(Literal {
                value: "3".to_string(),
                language: None,
                datatype: None,
            }),
        ]);

        let result = pf
            .execute(&var_subject(), TEXT_QUERY_IRI, &obj, &ctx)
            .expect("execute with maxResults");
        let solutions = result.into_solutions();
        assert!(
            solutions.len() <= 3,
            "should return at most 3 results, got {}",
            solutions.len()
        );
    }

    #[test]
    fn test_property_fn_execute_with_predicate_filter_arg() {
        let index = build_test_index();
        let pf = TextQueryPropertyFunction::new(index);
        let ctx = make_context();

        let label_pred = "http://example.org/label";
        let obj = PropFuncArg::list(vec![
            AlgebraTerm::Iri(Iri::new(label_pred).expect("iri")),
            AlgebraTerm::Literal(Literal {
                value: "semantic".to_string(),
                language: None,
                datatype: None,
            }),
        ]);

        let result = pf
            .execute(&var_subject(), TEXT_QUERY_IRI, &obj, &ctx)
            .expect("execute pred filter");
        let solutions = result.into_solutions();

        // doc3 (title predicate) should not appear in results
        let doc_var = Variable::new("doc").expect("var");
        let doc3_iri = Iri::new("http://example.org/doc3").expect("iri");
        for sol in &solutions {
            if let Some(AlgebraTerm::Iri(iri)) = sol.get(&doc_var) {
                assert_ne!(
                    iri, &doc3_iri,
                    "doc3 uses title predicate and should be excluded"
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Registry integration
    // -------------------------------------------------------------------------

    #[test]
    fn test_register_text_query_in_registry() {
        let index = build_test_index();
        let registry = PropertyFunctionRegistry::new();
        register_text_query(&registry, index).expect("register");

        assert!(
            registry.is_property_function(TEXT_QUERY_IRI),
            "text:query should be registered"
        );
        assert!(
            registry.get(TEXT_QUERY_IRI).is_some(),
            "registry.get should return the function"
        );
    }
}
