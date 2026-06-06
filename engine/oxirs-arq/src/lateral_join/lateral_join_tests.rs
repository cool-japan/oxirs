use super::*;

fn iri(s: &str) -> LateralValue {
    LateralValue::Iri(s.to_string())
}

fn lit(s: &str) -> LateralValue {
    LateralValue::Literal {
        value: s.to_string(),
        datatype: None,
        lang: None,
    }
}

fn typed_lit(s: &str, dt: &str) -> LateralValue {
    LateralValue::Literal {
        value: s.to_string(),
        datatype: Some(dt.to_string()),
        lang: None,
    }
}

fn make_row(bindings: &[(&str, LateralValue)]) -> SolutionMapping {
    bindings
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

fn simple_subquery() -> LateralSubquery {
    LateralSubquery {
        description: "SELECT (MAX(?score) AS ?maxScore) WHERE { ?person :score ?score }"
            .to_string(),
        correlated_vars: vec!["person".to_string()],
        projected_vars: vec!["maxScore".to_string()],
        has_aggregates: true,
        limit: None,
        order_by: vec![],
    }
}

fn simple_lateral() -> LateralJoin {
    LateralJoin {
        left_description: "?person a :Student".to_string(),
        subquery: simple_subquery(),
        strategy: LateralStrategy::NestedLoop,
        pushed_filters: vec![],
    }
}

// ── Executor: nested-loop tests ───────────────────────────────────────

#[test]
fn test_nested_loop_basic() {
    let lateral = simple_lateral();
    let left = vec![
        make_row(&[("person", iri("http://ex.org/alice"))]),
        make_row(&[("person", iri("http://ex.org/bob"))]),
    ];

    let mut exec = LateralJoinExecutor::with_defaults();
    let results = exec
        .execute(&lateral, &left, |correlated| {
            let person = correlated.get("person").cloned();
            match person {
                Some(LateralValue::Iri(p)) if p.contains("alice") => Ok(vec![make_row(&[(
                    "maxScore",
                    typed_lit("95", "xsd:integer"),
                )])]),
                Some(LateralValue::Iri(p)) if p.contains("bob") => Ok(vec![make_row(&[(
                    "maxScore",
                    typed_lit("87", "xsd:integer"),
                )])]),
                _ => Ok(vec![]),
            }
        })
        .expect("execution should succeed");

    assert_eq!(results.len(), 2);
    assert!(results[0].contains_key("person"));
    assert!(results[0].contains_key("maxScore"));
}

#[test]
fn test_nested_loop_empty_left() {
    let lateral = simple_lateral();
    let left: Vec<SolutionMapping> = vec![];
    let mut exec = LateralJoinExecutor::with_defaults();
    let results = exec
        .execute(&lateral, &left, |_| Ok(vec![make_row(&[("x", lit("1"))])]))
        .expect("execution should succeed");
    assert!(results.is_empty());
    assert_eq!(exec.stats().left_rows, 0);
}

#[test]
fn test_nested_loop_empty_subquery_result() {
    let lateral = simple_lateral();
    let left = vec![make_row(&[("person", iri("http://ex.org/charlie"))])];
    let mut exec = LateralJoinExecutor::with_defaults();
    let results = exec
        .execute(&lateral, &left, |_| Ok(vec![]))
        .expect("execution should succeed");
    assert!(results.is_empty());
    assert_eq!(exec.stats().subquery_evaluations, 1);
}

#[test]
fn test_nested_loop_multiple_subquery_results() {
    let lateral = LateralJoin {
        left_description: "?dept a :Department".to_string(),
        subquery: LateralSubquery {
            description: "SELECT ?emp WHERE { ?emp :worksIn ?dept }".to_string(),
            correlated_vars: vec!["dept".to_string()],
            projected_vars: vec!["emp".to_string()],
            has_aggregates: false,
            limit: None,
            order_by: vec![],
        },
        strategy: LateralStrategy::NestedLoop,
        pushed_filters: vec![],
    };

    let left = vec![make_row(&[("dept", iri("http://ex.org/engineering"))])];
    let mut exec = LateralJoinExecutor::with_defaults();
    let results = exec
        .execute(&lateral, &left, |_| {
            Ok(vec![
                make_row(&[("emp", iri("http://ex.org/alice"))]),
                make_row(&[("emp", iri("http://ex.org/bob"))]),
                make_row(&[("emp", iri("http://ex.org/charlie"))]),
            ])
        })
        .expect("execution should succeed");

    assert_eq!(results.len(), 3);
    for r in &results {
        assert!(r.contains_key("dept"));
        assert!(r.contains_key("emp"));
    }
}

#[test]
fn test_nested_loop_subquery_error_propagation() {
    let lateral = simple_lateral();
    let left = vec![make_row(&[("person", iri("http://ex.org/alice"))])];
    let mut exec = LateralJoinExecutor::with_defaults();
    let result = exec.execute(&lateral, &left, |_| {
        Err(LateralJoinError::SubqueryError(
            "timeout in remote endpoint".to_string(),
        ))
    });
    assert!(result.is_err());
}

// ── Executor: cached correlation tests ────────────────────────────────

#[test]
fn test_cached_correlation_hits() {
    let mut lateral = simple_lateral();
    lateral.strategy = LateralStrategy::CachedCorrelation;

    // Multiple rows with the same person — should hit cache
    let left = vec![
        make_row(&[("person", iri("http://ex.org/alice")), ("x", lit("1"))]),
        make_row(&[("person", iri("http://ex.org/alice")), ("x", lit("2"))]),
        make_row(&[("person", iri("http://ex.org/alice")), ("x", lit("3"))]),
    ];

    let mut exec = LateralJoinExecutor::with_defaults();
    let results = exec
        .execute(&lateral, &left, |_| {
            Ok(vec![make_row(&[("maxScore", lit("95"))])])
        })
        .expect("execution should succeed");

    assert_eq!(results.len(), 3);
    assert_eq!(exec.stats().cache_hits, 2);
    assert_eq!(exec.stats().cache_misses, 1);
    assert_eq!(exec.stats().subquery_evaluations, 1);
}

#[test]
fn test_cached_correlation_different_keys() {
    let mut lateral = simple_lateral();
    lateral.strategy = LateralStrategy::CachedCorrelation;

    let left = vec![
        make_row(&[("person", iri("http://ex.org/alice"))]),
        make_row(&[("person", iri("http://ex.org/bob"))]),
        make_row(&[("person", iri("http://ex.org/charlie"))]),
    ];

    let mut exec = LateralJoinExecutor::with_defaults();
    let _ = exec
        .execute(&lateral, &left, |_| {
            Ok(vec![make_row(&[("maxScore", lit("90"))])])
        })
        .expect("execution should succeed");

    assert_eq!(exec.stats().cache_hits, 0);
    assert_eq!(exec.stats().cache_misses, 3);
    assert_eq!(exec.stats().subquery_evaluations, 3);
}

#[test]
fn test_cache_hit_ratio() {
    let stats = LateralJoinStats {
        cache_hits: 75,
        cache_misses: 25,
        ..Default::default()
    };
    let ratio = stats.cache_hit_ratio();
    assert!((ratio - 75.0).abs() < 0.01);
}

#[test]
fn test_cache_hit_ratio_empty() {
    let stats = LateralJoinStats::default();
    assert_eq!(stats.cache_hit_ratio(), 0.0);
}

// ── Executor: batched tests ───────────────────────────────────────────

#[test]
fn test_batched_execution() {
    let mut lateral = simple_lateral();
    lateral.strategy = LateralStrategy::BatchedValues;

    let left = vec![
        make_row(&[("person", iri("http://ex.org/alice"))]),
        make_row(&[("person", iri("http://ex.org/bob"))]),
    ];

    let config = LateralJoinConfig {
        batch_size: 10,
        ..Default::default()
    };
    let mut exec = LateralJoinExecutor::new(config);
    let results = exec
        .execute(&lateral, &left, |_| {
            // Return results for both alice and bob
            Ok(vec![
                make_row(&[
                    ("person", iri("http://ex.org/alice")),
                    ("maxScore", lit("95")),
                ]),
                make_row(&[
                    ("person", iri("http://ex.org/bob")),
                    ("maxScore", lit("87")),
                ]),
            ])
        })
        .expect("execution should succeed");

    assert_eq!(results.len(), 2);
    assert_eq!(exec.stats().batches_submitted, 1);
}

#[test]
fn test_batched_multiple_batches() {
    let mut lateral = simple_lateral();
    lateral.strategy = LateralStrategy::BatchedValues;

    let left: Vec<_> = (0..5)
        .map(|i| make_row(&[("person", iri(&format!("http://ex.org/p{i}")))]))
        .collect();

    let config = LateralJoinConfig {
        batch_size: 2,
        ..Default::default()
    };
    let mut exec = LateralJoinExecutor::new(config);
    let _ = exec
        .execute(&lateral, &left, |bindings| {
            // Return the bindings back with a score
            let person = bindings.get("person").cloned();
            match person {
                Some(p) => Ok(vec![make_row(&[("person", p), ("maxScore", lit("90"))])]),
                None => Ok(vec![]),
            }
        })
        .expect("execution should succeed");

    assert_eq!(exec.stats().batches_submitted, 3); // ceil(5/2)
}

// ── Executor: pushed filters ──────────────────────────────────────────

#[test]
fn test_pushed_filter_equality() {
    let lateral = LateralJoin {
        left_description: "?person a :Student".to_string(),
        subquery: simple_subquery(),
        strategy: LateralStrategy::NestedLoop,
        pushed_filters: vec!["?person = <http://ex.org/alice>".to_string()],
    };

    let left = vec![
        make_row(&[("person", iri("http://ex.org/alice"))]),
        make_row(&[("person", iri("http://ex.org/bob"))]),
        make_row(&[("person", iri("http://ex.org/charlie"))]),
    ];

    let mut exec = LateralJoinExecutor::with_defaults();
    let results = exec
        .execute(&lateral, &left, |_| {
            Ok(vec![make_row(&[("maxScore", lit("95"))])])
        })
        .expect("execution should succeed");

    assert_eq!(results.len(), 1); // Only alice passes the filter
    assert_eq!(exec.stats().rows_filtered, 2);
}

// ── Executor: reset ───────────────────────────────────────────────────

#[test]
fn test_executor_reset() {
    let mut exec = LateralJoinExecutor::with_defaults();
    exec.stats = LateralJoinStats {
        left_rows: 100,
        result_rows: 200,
        subquery_evaluations: 50,
        cache_hits: 30,
        cache_misses: 20,
        ..Default::default()
    };
    exec.cache.insert("key".to_string(), vec![]);

    exec.reset();
    assert_eq!(exec.stats().left_rows, 0);
    assert_eq!(exec.stats().result_rows, 0);
    assert!(exec.cache.is_empty());
}

// ── Optimizer tests ───────────────────────────────────────────────────

#[test]
fn test_optimizer_chooses_cached_for_repeated_keys() {
    let optimizer = LateralOptimizer::new();
    let subquery = simple_subquery();
    let estimate = optimizer.choose_strategy(10000, 50, &subquery);
    // With 10000 left rows but only 50 distinct keys, caching should win
    assert!(
        estimate.strategy == LateralStrategy::CachedCorrelation
            || estimate.strategy == LateralStrategy::Decorrelate
    );
}

#[test]
fn test_optimizer_decorrelate_single_agg() {
    let optimizer = LateralOptimizer::new();
    let subquery = LateralSubquery {
        description: "SELECT (COUNT(*) AS ?cnt) WHERE { ?x :p ?y }".to_string(),
        correlated_vars: vec!["x".to_string()],
        projected_vars: vec!["cnt".to_string()],
        has_aggregates: true,
        limit: None,
        order_by: vec![],
    };
    let estimate = optimizer.choose_strategy(5000, 5000, &subquery);
    // With all distinct keys and single aggregate, decorrelation should be cheapest
    assert_eq!(estimate.strategy, LateralStrategy::Decorrelate);
}

#[test]
fn test_optimizer_cannot_decorrelate_multi_vars() {
    let optimizer = LateralOptimizer::new();
    let subquery = LateralSubquery {
        description: "test".to_string(),
        correlated_vars: vec!["x".to_string(), "y".to_string()],
        projected_vars: vec!["result".to_string()],
        has_aggregates: true,
        limit: None,
        order_by: vec![],
    };
    assert!(!optimizer.can_decorrelate(&subquery));
}

#[test]
fn test_optimizer_analyze_all_strategies() {
    let optimizer = LateralOptimizer::new();
    let subquery = simple_subquery();
    let estimates = optimizer.analyze(1000, 100, &subquery);
    assert!(estimates.len() >= 3);
    // Should be sorted by cost
    for w in estimates.windows(2) {
        assert!(w[0].estimated_cost <= w[1].estimated_cost);
    }
}

#[test]
fn test_optimizer_with_limit_subquery() {
    let optimizer = LateralOptimizer::new();
    let subquery = LateralSubquery {
        description: "SELECT ?x WHERE { ... } LIMIT 10".to_string(),
        correlated_vars: vec!["person".to_string()],
        projected_vars: vec!["x".to_string()],
        has_aggregates: false,
        limit: Some(10),
        order_by: vec![],
    };
    let estimate = optimizer.choose_strategy(100, 100, &subquery);
    // With LIMIT, cost should be lower
    assert!(estimate.estimated_cost > 0.0);
}

#[test]
fn test_optimizer_with_order_by() {
    let optimizer = LateralOptimizer::new();
    let subquery = LateralSubquery {
        description: "SELECT ?x WHERE { ... } ORDER BY ?x".to_string(),
        correlated_vars: vec!["person".to_string()],
        projected_vars: vec!["x".to_string()],
        has_aggregates: false,
        limit: None,
        order_by: vec![OrderSpec {
            variable: "x".to_string(),
            ascending: true,
        }],
    };
    let estimate = optimizer.choose_strategy(100, 100, &subquery);
    assert!(estimate.estimated_cost > 0.0);
}

// ── Validator tests ───────────────────────────────────────────────────

#[test]
fn test_validator_valid_lateral() {
    let subquery = simple_subquery();
    let left_vars = vec!["person".to_string(), "name".to_string()];
    let result = LateralValidator::validate(&subquery, &left_vars, 0, 4);
    assert!(result.is_valid);
    assert!(result.errors.is_empty());
    assert_eq!(result.detected_correlated_vars, vec!["person".to_string()]);
}

#[test]
fn test_validator_unbound_correlated_var() {
    let subquery = LateralSubquery {
        description: "test".to_string(),
        correlated_vars: vec!["missing_var".to_string()],
        projected_vars: vec!["result".to_string()],
        has_aggregates: false,
        limit: None,
        order_by: vec![],
    };
    let left_vars = vec!["person".to_string()];
    let result = LateralValidator::validate(&subquery, &left_vars, 0, 4);
    assert!(!result.is_valid);
    assert_eq!(result.errors.len(), 1);
    assert_eq!(
        result.errors[0].code,
        LateralErrorCode::UnboundCorrelatedVar
    );
}

#[test]
fn test_validator_excessive_nesting() {
    let subquery = simple_subquery();
    let left_vars = vec!["person".to_string()];
    let result = LateralValidator::validate(&subquery, &left_vars, 5, 4);
    assert!(!result.is_valid);
    assert!(result
        .errors
        .iter()
        .any(|e| e.code == LateralErrorCode::ExcessiveNesting));
}

#[test]
fn test_validator_no_correlation_warning() {
    let subquery = LateralSubquery {
        description: "test".to_string(),
        correlated_vars: vec![],
        projected_vars: vec!["x".to_string()],
        has_aggregates: false,
        limit: None,
        order_by: vec![],
    };
    let left_vars = vec!["person".to_string()];
    let result = LateralValidator::validate(&subquery, &left_vars, 0, 4);
    assert!(result.is_valid);
    assert!(!result.warnings.is_empty());
}

#[test]
fn test_validator_variable_conflict_warning() {
    let subquery = LateralSubquery {
        description: "test".to_string(),
        correlated_vars: vec!["person".to_string()],
        projected_vars: vec![
            "person".to_string(),
            "name".to_string(),
            "extra".to_string(),
        ],
        has_aggregates: false,
        limit: None,
        order_by: vec![],
    };
    let left_vars = vec!["person".to_string(), "name".to_string()];
    let result = LateralValidator::validate(&subquery, &left_vars, 0, 4);
    // "name" conflicts — it's projected by subquery but also in left_vars
    // "person" does not conflict because it is in correlated_vars
    assert!(result.warnings.iter().any(|w| w.contains("name")));
}

#[test]
fn test_validator_output_vars() {
    let subquery = LateralSubquery {
        description: "test".to_string(),
        correlated_vars: vec!["person".to_string()],
        projected_vars: vec!["maxScore".to_string()],
        has_aggregates: true,
        limit: None,
        order_by: vec![],
    };
    let left_vars = vec!["person".to_string()];
    let result = LateralValidator::validate(&subquery, &left_vars, 0, 4);
    assert!(result.output_vars.contains(&"person".to_string()));
    assert!(result.output_vars.contains(&"maxScore".to_string()));
}

// ── Parser tests ──────────────────────────────────────────────────────

#[test]
fn test_parser_detect_single_lateral() {
    let query = r#"
        SELECT ?person ?maxScore WHERE {
            ?person a :Student .
            LATERAL {
                SELECT (MAX(?score) AS ?maxScore)
                WHERE { ?person :score ?score }
            }
        }
    "#;
    let clauses = LateralParser::detect_lateral_clauses(query);
    assert_eq!(clauses.len(), 1);
    assert!(clauses[0].has_select);
    assert!(clauses[0].body.contains("MAX"));
}

#[test]
fn test_parser_detect_multiple_laterals() {
    let query = r#"
        SELECT * WHERE {
            ?x a :Foo .
            LATERAL { SELECT ?a WHERE { ?x :p ?a } }
            LATERAL { SELECT ?b WHERE { ?x :q ?b } }
        }
    "#;
    let clauses = LateralParser::detect_lateral_clauses(query);
    assert_eq!(clauses.len(), 2);
}

#[test]
fn test_parser_no_lateral() {
    let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
    let clauses = LateralParser::detect_lateral_clauses(query);
    assert!(clauses.is_empty());
}

#[test]
fn test_parser_lateral_not_keyword() {
    // "LATERALLY" should not match
    let query = "SELECT ?x WHERE { ?x :laterally ?y }";
    let clauses = LateralParser::detect_lateral_clauses(query);
    assert!(clauses.is_empty());
}

#[test]
fn test_extract_variables() {
    let fragment = "?person :hasExam ?exam . ?exam :score ?score";
    let vars = LateralParser::extract_variables(fragment);
    assert!(vars.contains(&"person".to_string()));
    assert!(vars.contains(&"exam".to_string()));
    assert!(vars.contains(&"score".to_string()));
}

#[test]
fn test_extract_variables_dollar_sign() {
    let fragment = "$person :hasExam $exam";
    let vars = LateralParser::extract_variables(fragment);
    assert!(vars.contains(&"person".to_string()));
    assert!(vars.contains(&"exam".to_string()));
}

#[test]
fn test_detect_aggregates() {
    assert!(LateralParser::detect_aggregates("SELECT (MAX(?x) AS ?m)"));
    assert!(LateralParser::detect_aggregates("SELECT (COUNT(*) AS ?c)"));
    assert!(LateralParser::detect_aggregates("SUM(?val)"));
    assert!(!LateralParser::detect_aggregates(
        "SELECT ?x WHERE { ?x :p ?y }"
    ));
}

#[test]
fn test_detect_order_by() {
    assert!(LateralParser::detect_order_by(
        "SELECT ?x WHERE { } ORDER BY ?x"
    ));
    assert!(!LateralParser::detect_order_by(
        "SELECT ?x WHERE { ?x :p ?y }"
    ));
}

#[test]
fn test_detect_limit() {
    assert!(LateralParser::detect_limit("SELECT ?x WHERE { } LIMIT 10"));
    assert!(!LateralParser::detect_limit("SELECT ?x WHERE { ?x :p ?y }"));
}

// ── LateralValue display tests ────────────────────────────────────────

#[test]
fn test_lateral_value_display_iri() {
    let v = iri("http://example.org/foo");
    assert_eq!(format!("{v}"), "<http://example.org/foo>");
}

#[test]
fn test_lateral_value_display_literal() {
    let v = lit("hello");
    assert_eq!(format!("{v}"), "\"hello\"");
}

#[test]
fn test_lateral_value_display_typed_literal() {
    let v = typed_lit("42", "xsd:integer");
    assert_eq!(format!("{v}"), "\"42\"^^<xsd:integer>");
}

#[test]
fn test_lateral_value_display_lang_literal() {
    let v = LateralValue::Literal {
        value: "hello".to_string(),
        datatype: None,
        lang: Some("en".to_string()),
    };
    assert_eq!(format!("{v}"), "\"hello\"@en");
}

#[test]
fn test_lateral_value_display_blank() {
    let v = LateralValue::BlankNode("b0".to_string());
    assert_eq!(format!("{v}"), "_:b0");
}

// ── Error display tests ───────────────────────────────────────────────

#[test]
fn test_error_display_subquery() {
    let e = LateralJoinError::SubqueryError("connection refused".to_string());
    assert!(format!("{e}").contains("connection refused"));
}

#[test]
fn test_error_display_timeout() {
    let e = LateralJoinError::Timeout {
        description: "remote service".to_string(),
        elapsed_ms: 5000,
    };
    assert!(format!("{e}").contains("5000ms"));
}

#[test]
fn test_error_display_incompatible() {
    let e = LateralJoinError::IncompatibleBindings {
        variable: "x".to_string(),
        left_value: "1".to_string(),
        right_value: "2".to_string(),
    };
    assert!(format!("{e}").contains("x"));
}

#[test]
fn test_error_display_nesting() {
    let e = LateralJoinError::NestingDepthExceeded { depth: 5, max: 4 };
    assert!(format!("{e}").contains("5"));
}

// ── Strategy display tests ────────────────────────────────────────────

#[test]
fn test_strategy_display() {
    assert_eq!(format!("{}", LateralStrategy::NestedLoop), "NestedLoop");
    assert_eq!(
        format!("{}", LateralStrategy::BatchedValues),
        "BatchedValues"
    );
    assert_eq!(format!("{}", LateralStrategy::Decorrelate), "Decorrelate");
    assert_eq!(
        format!("{}", LateralStrategy::CachedCorrelation),
        "CachedCorrelation"
    );
}

// ── Merge mapping tests ───────────────────────────────────────────────

#[test]
fn test_merge_disjoint_mappings() {
    let left = make_row(&[("a", lit("1"))]);
    let right = make_row(&[("b", lit("2"))]);
    let merged = LateralJoinExecutor::merge_mappings(&left, &right).expect("merge should work");
    assert_eq!(merged.len(), 2);
    assert!(merged.contains_key("a"));
    assert!(merged.contains_key("b"));
}

#[test]
fn test_merge_overlapping_mappings() {
    let left = make_row(&[("a", lit("1")), ("b", lit("old"))]);
    let right = make_row(&[("b", lit("new")), ("c", lit("3"))]);
    let merged = LateralJoinExecutor::merge_mappings(&left, &right).expect("merge should work");
    assert_eq!(merged.len(), 3);
    // Right overwrites left for LATERAL semantics
    assert_eq!(merged.get("b"), Some(&lit("new")));
}

// ── Avg subquery time ─────────────────────────────────────────────────

#[test]
fn test_avg_subquery_time() {
    let stats = LateralJoinStats {
        subquery_evaluations: 10,
        subquery_time_ms: 500,
        ..Default::default()
    };
    assert!((stats.avg_subquery_time_ms() - 50.0).abs() < 0.01);
}

#[test]
fn test_avg_subquery_time_zero() {
    let stats = LateralJoinStats::default();
    assert_eq!(stats.avg_subquery_time_ms(), 0.0);
}

// ── Config tests ──────────────────────────────────────────────────────

#[test]
fn test_default_config() {
    let config = LateralJoinConfig::default();
    assert_eq!(config.batch_size, 128);
    assert_eq!(config.cache_capacity, 4096);
    assert_eq!(config.max_nesting_depth, 4);
    assert!(config.auto_decorrelate);
}

#[test]
fn test_optimizer_config_default() {
    let config = LateralOptimizerConfig::default();
    assert_eq!(config.cache_threshold, 1000);
    assert_eq!(config.batch_threshold, 500);
}

// ── is_compatible tests ───────────────────────────────────────────────

#[test]
fn test_is_compatible_same_values() {
    let left = make_row(&[("x", iri("http://ex.org/a"))]);
    let right = make_row(&[("x", iri("http://ex.org/a")), ("y", lit("1"))]);
    assert!(LateralJoinExecutor::is_compatible(
        &left,
        &right,
        &["x".to_string()]
    ));
}

#[test]
fn test_is_compatible_different_values() {
    let left = make_row(&[("x", iri("http://ex.org/a"))]);
    let right = make_row(&[("x", iri("http://ex.org/b"))]);
    assert!(!LateralJoinExecutor::is_compatible(
        &left,
        &right,
        &["x".to_string()]
    ));
}

#[test]
fn test_is_compatible_unbound() {
    let left = make_row(&[("x", iri("http://ex.org/a"))]);
    let right = make_row(&[("y", lit("1"))]); // x not in right
    assert!(LateralJoinExecutor::is_compatible(
        &left,
        &right,
        &["x".to_string()]
    ));
}

// ── Parse equality filter ─────────────────────────────────────────────

#[test]
fn test_parse_equality_filter_iri() {
    let result = LateralJoinExecutor::parse_equality_filter("?x = <http://ex.org/a>");
    assert_eq!(
        result,
        Some(("x".to_string(), "<http://ex.org/a>".to_string()))
    );
}

#[test]
fn test_parse_equality_filter_literal() {
    let result = LateralJoinExecutor::parse_equality_filter("?name = \"Alice\"");
    assert_eq!(result, Some(("name".to_string(), "\"Alice\"".to_string())));
}

#[test]
fn test_parse_equality_filter_invalid() {
    let result = LateralJoinExecutor::parse_equality_filter("?x > 5");
    assert!(result.is_none());
}

// ── find_matching_brace ───────────────────────────────────────────────

#[test]
fn test_find_matching_brace_simple() {
    let s = "{ hello }";
    assert_eq!(LateralParser::find_matching_brace(s, 0), Some(8));
}

#[test]
fn test_find_matching_brace_nested() {
    let s = "{ { inner } outer }";
    assert_eq!(LateralParser::find_matching_brace(s, 0), Some(18));
}

#[test]
fn test_find_matching_brace_unmatched() {
    let s = "{ no closing";
    assert_eq!(LateralParser::find_matching_brace(s, 0), None);
}

#[test]
fn test_find_matching_brace_not_brace() {
    let s = "hello";
    assert_eq!(LateralParser::find_matching_brace(s, 0), None);
}
