//! Tests for the RML mapping module.

#[cfg(test)]
mod tests {
    use crate::mapping::{
        DataSource, MappingEngine, MappingError, MappingRule, MappingRuleBuilder, ObjectSpec,
        PredicateObjectMap, Row, Template,
    };

    // ── helpers ──────────────────────────────────────────────────────────

    fn engine() -> MappingEngine {
        MappingEngine::new()
    }

    fn lenient_engine() -> MappingEngine {
        MappingEngine::new_lenient()
    }

    fn xsd(local: &str) -> String {
        format!("http://www.w3.org/2001/XMLSchema#{local}")
    }

    fn ex(local: &str) -> String {
        format!("http://example.org/{local}")
    }

    fn foaf(local: &str) -> String {
        format!("http://xmlns.com/foaf/0.1/{local}")
    }

    // ── Template tests ───────────────────────────────────────────────────

    #[test]
    fn test_template_simple_substitution() {
        let tpl = Template::new("http://example.org/{id}");
        let mut row = Row::new();
        row.values.insert("id".to_string(), "42".to_string());
        let result = tpl.render(&row, 0).expect("should succeed");
        assert_eq!(result, "http://example.org/42");
    }

    #[test]
    fn test_template_multiple_placeholders() {
        let tpl = Template::new("http://example.org/{type}/{id}");
        let mut row = Row::new();
        row.values.insert("type".to_string(), "person".to_string());
        row.values.insert("id".to_string(), "7".to_string());
        let result = tpl.render(&row, 0).expect("should succeed");
        assert_eq!(result, "http://example.org/person/7");
    }

    #[test]
    fn test_template_percent_encoding() {
        let tpl = Template::new("http://example.org/{name}");
        let mut row = Row::new();
        row.values
            .insert("name".to_string(), "hello world".to_string());
        let result = tpl.render(&row, 0).expect("should succeed");
        assert_eq!(result, "http://example.org/hello%20world");
    }

    #[test]
    fn test_template_missing_column_error() {
        let tpl = Template::new("http://example.org/{missing}");
        let row = Row::new();
        let err = tpl.render(&row, 3).unwrap_err();
        match err {
            MappingError::UnresolvableTemplate {
                column, row_index, ..
            } => {
                assert_eq!(column, "missing");
                assert_eq!(row_index, 3);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn test_template_no_placeholders() {
        let tpl = Template::new("http://example.org/constant");
        let row = Row::new();
        let result = tpl.render(&row, 0).expect("should succeed");
        assert_eq!(result, "http://example.org/constant");
    }

    #[test]
    fn test_template_slash_encoded() {
        let tpl = Template::new("http://example.org/{path}");
        let mut row = Row::new();
        row.values.insert("path".to_string(), "a/b/c".to_string());
        let result = tpl.render(&row, 0).expect("should succeed");
        // '/' is not in RFC 3986 unreserved, should be encoded
        assert_eq!(result, "http://example.org/a%2Fb%2Fc");
    }

    // ── Row tests ────────────────────────────────────────────────────────

    #[test]
    fn test_row_get() {
        let mut row = Row::new();
        row.values.insert("key".to_string(), "value".to_string());
        assert_eq!(row.get("key"), Some("value"));
        assert_eq!(row.get("absent"), None);
    }

    #[test]
    fn test_row_contains() {
        let row = Row::from_pairs(vec![("x".to_string(), "1".to_string())]);
        assert!(row.contains("x"));
        assert!(!row.contains("y"));
    }

    #[test]
    fn test_row_display() {
        let row = Row::from_pairs(vec![("a".to_string(), "1".to_string())]);
        let s = format!("{row}");
        assert!(s.contains("a: 1"));
    }

    // ── CSV parsing tests ────────────────────────────────────────────────

    #[test]
    fn test_csv_basic_parse() {
        let csv = "id,name,age\n1,Alice,30\n2,Bob,25";
        let (headers, rows) = MappingEngine::parse_csv(csv, ',').expect("should succeed");
        assert_eq!(headers, vec!["id", "name", "age"]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("name"), Some("Alice"));
        assert_eq!(rows[1].get("age"), Some("25"));
    }

    #[test]
    fn test_csv_tab_delimiter() {
        let csv = "id\tvalue\n1\thello\n2\tworld";
        let (_headers, rows) = MappingEngine::parse_csv(csv, '\t').expect("should succeed");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("value"), Some("hello"));
    }

    #[test]
    fn test_csv_quoted_fields() {
        let csv = "id,desc\n1,\"hello, world\"\n2,simple";
        let (_headers, rows) = MappingEngine::parse_csv(csv, ',').expect("should succeed");
        assert_eq!(rows[0].get("desc"), Some("hello, world"));
        assert_eq!(rows[1].get("desc"), Some("simple"));
    }

    #[test]
    fn test_csv_escaped_quotes() {
        let csv = "id,text\n1,\"say \"\"hi\"\"\"\n";
        let (_headers, rows) = MappingEngine::parse_csv(csv, ',').expect("should succeed");
        assert_eq!(rows[0].get("text"), Some("say \"hi\""));
    }

    #[test]
    fn test_csv_crlf_endings() {
        let csv = "id,name\r\n1,Alice\r\n2,Bob\r\n";
        let (_headers, rows) = MappingEngine::parse_csv(csv, ',').expect("should succeed");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("name"), Some("Alice"));
    }

    #[test]
    fn test_csv_semicolon_delimiter() {
        let csv = "id;value\n1;alpha\n2;beta";
        let (_headers, rows) = MappingEngine::parse_csv(csv, ';').expect("should succeed");
        assert_eq!(rows[0].get("value"), Some("alpha"));
        assert_eq!(rows[1].get("value"), Some("beta"));
    }

    #[test]
    fn test_csv_empty_content_returns_empty() {
        let (headers, rows) = MappingEngine::parse_csv("", ',').expect("should succeed");
        assert!(headers.is_empty());
        assert!(rows.is_empty());
    }

    #[test]
    fn test_csv_field_count_mismatch_error() {
        let csv = "id,name\n1,Alice,extra\n";
        let err = MappingEngine::parse_csv(csv, ',').unwrap_err();
        assert!(matches!(err, MappingError::CsvParseError { .. }));
    }

    #[test]
    fn test_csv_trailing_empty_lines_skipped() {
        let csv = "id,name\n1,Alice\n\n\n";
        let (_headers, rows) = MappingEngine::parse_csv(csv, ',').expect("should succeed");
        assert_eq!(rows.len(), 1);
    }

    // ── JSON parsing tests ───────────────────────────────────────────────

    #[test]
    fn test_json_flat_objects() {
        let json = r#"[{"id":"1","name":"Alice"},{"id":"2","name":"Bob"}]"#;
        let rows = MappingEngine::parse_json(json, None).expect("should succeed");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("name"), Some("Alice"));
        assert_eq!(rows[1].get("id"), Some("2"));
    }

    #[test]
    fn test_json_nested_path() {
        let json = r#"{"data":{"people":[{"id":"1","name":"Alice"}]}}"#;
        let rows = MappingEngine::parse_json(json, Some("data.people")).expect("should succeed");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("name"), Some("Alice"));
    }

    #[test]
    fn test_json_numeric_values_coerced() {
        let json = r#"[{"id":1,"score":9.5,"active":true}]"#;
        let rows = MappingEngine::parse_json(json, None).expect("should succeed");
        assert_eq!(rows[0].get("id"), Some("1"));
        assert_eq!(rows[0].get("score"), Some("9.5"));
        assert_eq!(rows[0].get("active"), Some("true"));
    }

    #[test]
    fn test_json_null_value_becomes_empty() {
        let json = r#"[{"id":"1","name":null}]"#;
        let rows = MappingEngine::parse_json(json, None).expect("should succeed");
        assert_eq!(rows[0].get("name"), Some(""));
    }

    #[test]
    fn test_json_invalid_json_error() {
        let err = MappingEngine::parse_json("not json", None).unwrap_err();
        assert!(matches!(err, MappingError::JsonParseError { .. }));
    }

    #[test]
    fn test_json_path_no_match_error() {
        let json = r#"{"a":{}}"#;
        let err = MappingEngine::parse_json(json, Some("a.b.c")).unwrap_err();
        assert!(matches!(err, MappingError::JsonPathNoMatch { .. }));
    }

    #[test]
    fn test_json_root_not_array_error() {
        let json = r#"{"key":"value"}"#;
        let err = MappingEngine::parse_json(json, None).unwrap_err();
        assert!(matches!(err, MappingError::JsonPathNoMatch { .. }));
    }

    #[test]
    fn test_json_empty_array() {
        let json = r#"[]"#;
        let rows = MappingEngine::parse_json(json, None).expect("should succeed");
        assert!(rows.is_empty());
    }

    // ── Basic CSV mapping tests ──────────────────────────────────────────

    #[test]
    fn test_csv_mapping_single_predicate() {
        let csv = "id,name\n1,Alice";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let t = &triples[0];
        assert_eq!(t.subject().to_string(), format!("<{}>", ex("1")));
        assert_eq!(t.predicate().to_string(), format!("<{}>", foaf("name")));
        assert!(t.object().to_string().contains("Alice"));
    }

    #[test]
    fn test_csv_mapping_two_rows_two_predicates() {
        let csv = "id,name,age\n1,Alice,30\n2,Bob,25";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .map(foaf("age"), ObjectSpec::Column("age".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 4); // 2 rows × 2 predicates
    }

    #[test]
    fn test_csv_mapping_typed_integer() {
        let csv = "id,age\n1,42";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                foaf("age"),
                ObjectSpec::TypedColumn {
                    column: "age".to_string(),
                    datatype: xsd("integer"),
                },
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains("42"), "object should contain 42, got: {obj}");
        assert!(
            obj.contains("integer"),
            "object should contain xsd:integer, got: {obj}"
        );
    }

    #[test]
    fn test_csv_mapping_typed_date() {
        let csv = "id,dob\n1,1990-01-15";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                ex("dob"),
                ObjectSpec::TypedColumn {
                    column: "dob".to_string(),
                    datatype: xsd("date"),
                },
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains("1990-01-15"));
        assert!(obj.contains("date"));
    }

    #[test]
    fn test_csv_mapping_constant_object() {
        let csv = "id\n1\n2";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                ObjectSpec::Constant("Person".to_string()),
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 2);
        for t in &triples {
            assert!(t.object().to_string().contains("Person"));
        }
    }

    #[test]
    fn test_csv_mapping_constant_iri_object() {
        let csv = "id\n1";
        let person_class = ex("Person");
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                ObjectSpec::ConstantIri(person_class.clone()),
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains(&person_class));
    }

    #[test]
    fn test_csv_mapping_template_object() {
        let csv = "id,dept\n1,sales";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                ex("department"),
                ObjectSpec::Template(Template::new("http://example.org/dept/{dept}")),
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains("sales"), "got: {obj}");
    }

    #[test]
    fn test_csv_mapping_lang_fixed() {
        let csv = "id,label\n1,Hello";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                ex("label"),
                ObjectSpec::LangFixed {
                    column: "label".to_string(),
                    lang: "en".to_string(),
                },
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains("Hello"), "got: {obj}");
        assert!(obj.contains("en"), "got: {obj}");
    }

    #[test]
    fn test_csv_mapping_lang_column() {
        let csv = "id,label,lang\n1,Bonjour,fr";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                ex("label"),
                ObjectSpec::LangColumn {
                    column: "label".to_string(),
                    lang_column: "lang".to_string(),
                },
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains("Bonjour"), "got: {obj}");
        assert!(obj.contains("fr"), "got: {obj}");
    }

    // ── Named graph tests ────────────────────────────────────────────────

    #[test]
    fn test_named_graph_assignment() {
        let csv = "id\n1";
        let graph = "http://example.org/graph1";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(ex("type"), ObjectSpec::Constant("X".to_string()))
            .graph(graph)
            .build();
        assert_eq!(rule.graph_name.as_deref(), Some(graph));
        // Engine still produces triples (named graph metadata is on rule)
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
    }

    // ── JSON mapping tests ───────────────────────────────────────────────

    #[test]
    fn test_json_mapping_flat() {
        let json = r#"[{"id":"1","name":"Alice"},{"id":"2","name":"Bob"}]"#;
        let rule = MappingRuleBuilder::new("test")
            .json_source(json)
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_json_mapping_nested_path() {
        let json = r#"{"items":[{"id":"10","val":"x"},{"id":"20","val":"y"}]}"#;
        let rule = MappingRuleBuilder::new("test")
            .json_source_with_path(json, "items")
            .subject_template(ex("{id}"))
            .map(ex("val"), ObjectSpec::Column("val".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_json_mapping_typed_integer_column() {
        let json = r#"[{"id":"1","count":42}]"#;
        let rule = MappingRuleBuilder::new("test")
            .json_source(json)
            .subject_template(ex("{id}"))
            .map(
                ex("count"),
                ObjectSpec::TypedColumn {
                    column: "count".to_string(),
                    datatype: xsd("integer"),
                },
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains("42"));
        assert!(obj.contains("integer"));
    }

    #[test]
    fn test_json_mapping_multi_predicates() {
        let json = r#"[{"id":"1","name":"Alice","age":"30","city":"NYC"}]"#;
        let rule = MappingRuleBuilder::new("test")
            .json_source(json)
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .map(foaf("age"), ObjectSpec::Column("age".to_string()))
            .map(ex("city"), ObjectSpec::Column("city".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 3);
    }

    // ── Inline values tests ──────────────────────────────────────────────

    #[test]
    fn test_inline_values_mapping() {
        let rule = MappingRuleBuilder::new("test")
            .inline_source(
                vec!["id".to_string(), "name".to_string()],
                vec![
                    vec!["1".to_string(), "Alice".to_string()],
                    vec!["2".to_string(), "Bob".to_string()],
                ],
            )
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 2);
    }

    // ── Batch execution tests ────────────────────────────────────────────

    #[test]
    fn test_execute_all_multiple_rules() {
        let csv1 = "id,name\n1,Alice";
        let csv2 = "id,name\n100,Bob";
        let rule1 = MappingRuleBuilder::new("r1")
            .csv_source(csv1)
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .build();
        let rule2 = MappingRuleBuilder::new("r2")
            .csv_source(csv2)
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .build();
        let triples = engine()
            .execute_all(&[rule1, rule2])
            .expect("should succeed");
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_execute_all_empty_rules() {
        let triples = engine().execute_all(&[]).expect("should succeed");
        assert!(triples.is_empty());
    }

    // ── Error case tests ─────────────────────────────────────────────────

    #[test]
    fn test_missing_column_error() {
        let csv = "id\n1";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .build();
        let err = engine().execute(&rule).unwrap_err();
        assert!(matches!(err, MappingError::MissingColumn { column, .. } if column == "name"));
    }

    #[test]
    fn test_missing_subject_column_error() {
        let csv = "name\nAlice";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{missing_id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .build();
        let err = engine().execute(&rule).unwrap_err();
        assert!(matches!(err, MappingError::UnresolvableTemplate { .. }));
    }

    #[test]
    fn test_lenient_engine_skips_bad_rows() {
        let csv = "id,name\n1,Alice\n2,Bob";
        // Subject template referencing column that does not exist for a "ghost" row
        // We test this via a bad predicate-object map with a missing column
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            // "score" does not exist; lenient engine should skip those triples
            .map(ex("score"), ObjectSpec::Column("score".to_string()))
            .build();
        let triples = lenient_engine().execute(&rule).expect("should succeed");
        // Both rows fail on the score column; lenient skips them
        assert_eq!(triples.len(), 0);
    }

    #[test]
    fn test_invalid_predicate_iri_error() {
        let csv = "id\n1";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map("not a valid iri", ObjectSpec::Constant("x".to_string()))
            .build();
        let err = engine().execute(&rule).unwrap_err();
        assert!(matches!(err, MappingError::InvalidPredicateIri { .. }));
    }

    #[test]
    fn test_invalid_subject_iri_error() {
        let csv = "id\n1";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            // Template produces a string that may not be a valid absolute IRI
            .subject_template("not-an-iri/{id}")
            .map(foaf("name"), ObjectSpec::Constant("x".to_string()))
            .build();
        let err = engine().execute(&rule).unwrap_err();
        assert!(matches!(err, MappingError::InvalidIri { .. }));
    }

    // ── Builder pattern tests ────────────────────────────────────────────

    #[test]
    fn test_builder_chain() {
        let rule = MappingRuleBuilder::new("chain_test")
            .csv_source("id,x,y\n1,2,3")
            .subject_template(ex("{id}"))
            .map(ex("x"), ObjectSpec::Column("x".to_string()))
            .map(ex("y"), ObjectSpec::Column("y".to_string()))
            .graph("http://example.org/g1")
            .build();
        assert_eq!(rule.name, "chain_test");
        assert_eq!(rule.predicate_object_maps.len(), 2);
        assert_eq!(rule.graph_name.as_deref(), Some("http://example.org/g1"));
    }

    #[test]
    fn test_builder_csv_with_delimiter() {
        let rule = MappingRuleBuilder::new("pipe")
            .csv_source_with_delimiter("id|name\n1|Alice", '|')
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        assert!(triples[0].object().to_string().contains("Alice"));
    }

    #[test]
    fn test_builder_json_source_with_path() {
        let json = r#"{"list":[{"id":"5","v":"ok"}]}"#;
        let rule = MappingRuleBuilder::new("j")
            .json_source_with_path(json, "list")
            .subject_template(ex("{id}"))
            .map(ex("v"), ObjectSpec::Column("v".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        assert!(triples[0].object().to_string().contains("ok"));
    }

    // ── IRI generation tests ─────────────────────────────────────────────

    #[test]
    fn test_iri_from_column_value() {
        let csv = "id,related_id\n1,99";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                ex("related"),
                ObjectSpec::Template(Template::new("http://example.org/item/{related_id}")),
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains("99"), "got: {obj}");
    }

    #[test]
    fn test_iri_generation_with_special_chars() {
        let csv = "id\nhello world";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(ex("self"), ObjectSpec::ConstantIri(ex("x")))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        // Subject should have space encoded as %20
        let subj = triples[0].subject().to_string();
        assert!(subj.contains("%20"), "got: {subj}");
    }

    // ── Multiple rules interaction tests ─────────────────────────────────

    #[test]
    fn test_multiple_rules_different_sources() {
        let csv = "id,label\n1,CSV-item";
        let json = r#"[{"id":"2","label":"JSON-item"}]"#;
        let rule_csv = MappingRuleBuilder::new("r_csv")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(ex("label"), ObjectSpec::Column("label".to_string()))
            .build();
        let rule_json = MappingRuleBuilder::new("r_json")
            .json_source(json)
            .subject_template(ex("{id}"))
            .map(ex("label"), ObjectSpec::Column("label".to_string()))
            .build();
        let triples = engine()
            .execute_all(&[rule_csv, rule_json])
            .expect("should succeed");
        assert_eq!(triples.len(), 2);
    }

    // ── Typed literal tests ──────────────────────────────────────────────

    #[test]
    fn test_typed_literal_float() {
        let csv = "id,score\n1,3.14";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                ex("score"),
                ObjectSpec::TypedColumn {
                    column: "score".to_string(),
                    datatype: xsd("decimal"),
                },
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains("3.14"));
        assert!(obj.contains("decimal"));
    }

    #[test]
    fn test_typed_literal_boolean() {
        let csv = "id,active\n1,true";
        let rule = MappingRuleBuilder::new("test")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .map(
                ex("active"),
                ObjectSpec::TypedColumn {
                    column: "active".to_string(),
                    datatype: xsd("boolean"),
                },
            )
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 1);
        let obj = triples[0].object().to_string();
        assert!(obj.contains("true"));
        assert!(obj.contains("boolean"));
    }

    // ── Edge case tests ──────────────────────────────────────────────────

    #[test]
    fn test_empty_csv_produces_no_triples() {
        let rule = MappingRuleBuilder::new("empty")
            .csv_source("")
            .subject_template(ex("{id}"))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert!(triples.is_empty());
    }

    #[test]
    fn test_csv_only_header_produces_no_triples() {
        let rule = MappingRuleBuilder::new("header-only")
            .csv_source("id,name")
            .subject_template(ex("{id}"))
            .map(foaf("name"), ObjectSpec::Column("name".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert!(triples.is_empty());
    }

    #[test]
    fn test_json_array_empty_produces_no_triples() {
        let rule = MappingRuleBuilder::new("empty-json")
            .json_source("[]")
            .subject_template(ex("{id}"))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert!(triples.is_empty());
    }

    #[test]
    fn test_no_predicate_object_maps_produces_no_triples() {
        let csv = "id\n1\n2";
        let rule = MappingRuleBuilder::new("no-pom")
            .csv_source(csv)
            .subject_template(ex("{id}"))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert!(triples.is_empty());
    }

    #[test]
    fn test_percent_encode_unicode() {
        let tpl = Template::new("http://example.org/{name}");
        let mut row = Row::new();
        row.values
            .insert("name".to_string(), "こんにちは".to_string());
        let result = tpl.render(&row, 0).expect("should succeed");
        // Should be percent-encoded
        assert!(result.starts_with("http://example.org/%"));
        assert!(!result.contains("こんにちは"));
    }

    #[test]
    fn test_csv_mapping_pipe_delimiter_multi_row() {
        let csv = "id|label\n10|alpha\n20|beta\n30|gamma";
        let rule = MappingRuleBuilder::new("pipe-multi")
            .csv_source_with_delimiter(csv, '|')
            .subject_template(ex("{id}"))
            .map(ex("label"), ObjectSpec::Column("label".to_string()))
            .build();
        let triples = engine().execute(&rule).expect("should succeed");
        assert_eq!(triples.len(), 3);
    }

    #[test]
    fn test_mapping_engine_default() {
        let engine = MappingEngine::default();
        assert!(!engine.skip_errors);
    }

    #[test]
    fn test_mapping_rule_add_pom() {
        let mut rule = MappingRule::new(
            "r",
            DataSource::Csv {
                content: "id\n1".to_string(),
                delimiter: ',',
            },
            Template::new(ex("{id}")),
        );
        assert!(rule.predicate_object_maps.is_empty());
        rule.add_predicate_object_map(PredicateObjectMap::new(
            ex("p"),
            ObjectSpec::Constant("v".to_string()),
        ));
        assert_eq!(rule.predicate_object_maps.len(), 1);
    }

    #[test]
    fn test_predicate_object_map_construction() {
        let pom = PredicateObjectMap::new(
            "http://example.org/pred",
            ObjectSpec::Column("col".to_string()),
        );
        assert_eq!(pom.predicate, "http://example.org/pred");
    }

    #[test]
    fn test_row_from_pairs() {
        let row = Row::from_pairs(vec![
            ("a".to_string(), "1".to_string()),
            ("b".to_string(), "2".to_string()),
        ]);
        assert_eq!(row.get("a"), Some("1"));
        assert_eq!(row.get("b"), Some("2"));
    }

    #[test]
    fn test_json_deeply_nested_path() {
        let json = r#"{"a":{"b":{"c":[{"id":"1","name":"deep"}]}}}"#;
        let rows = MappingEngine::parse_json(json, Some("a.b.c")).expect("should succeed");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("name"), Some("deep"));
    }

    #[test]
    fn test_csv_quoted_field_with_newline() {
        let csv = "id,desc\n1,\"line1\nline2\"\n2,simple";
        let (_headers, rows) = MappingEngine::parse_csv(csv, ',').expect("should succeed");
        assert_eq!(rows.len(), 2);
        assert!(rows[0].get("desc").expect("should succeed").contains('\n'));
        assert_eq!(rows[1].get("desc"), Some("simple"));
    }

    #[test]
    fn test_template_display() {
        let tpl = Template::new("http://example.org/{id}");
        assert_eq!(tpl.to_string(), "http://example.org/{id}");
    }

    #[test]
    fn test_row_iter() {
        let row = Row::from_pairs(vec![
            ("x".to_string(), "1".to_string()),
            ("y".to_string(), "2".to_string()),
        ]);
        let count = row.iter().count();
        assert_eq!(count, 2);
    }
}
