//! Comprehensive tests for the SPARQL-Generate module.
//!
//! Covers AST construction, parsing, execution (single and multi-row),
//! error paths, and edge cases.

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::generate::{
        ast::{GenerateLiteral, GenerateQuery, TemplateClause},
        executor::{Bindings, GenerateExecutor},
        parser::parse,
        GenerateError,
    };

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn bindings(pairs: &[(&str, &str)]) -> Bindings {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    // ── Test 1: Parse a simple GENERATE query with one text + one var clause ──

    #[test]
    fn test_parse_simple_generate() {
        let src = r#"
            GENERATE {
                "Hello, " ?name
            }
            WHERE { ?s foaf:name ?name . }
        "#;

        let q = parse(src).expect("parse should succeed");
        // Template must have at least two clauses: the text and the var.
        assert!(!q.template.is_empty(), "template must not be empty");
        // At least one clause must reference the variable `name`.
        let has_var = q
            .template
            .iter()
            .any(|c| matches!(&c.expr, GenerateLiteral::Var(v) if v == "name"));
        assert!(has_var, "expected a Var(name) clause");
    }

    // ── Test 2: Parse a multi-clause template ────────────────────────────────

    #[test]
    fn test_parse_multiclause_template() {
        let src = r#"
            GENERATE {
                "id=" ?id
                "name=" ?label
                "type=" ?kind
            }
            WHERE { ?s ex:id ?id ; rdfs:label ?label ; rdf:type ?kind . }
        "#;

        let q = parse(src).expect("parse should succeed");
        // Each "prefix + var" pair is one clause.
        assert!(
            q.template.len() >= 3,
            "expected at least 3 clauses, got {}",
            q.template.len()
        );
    }

    // ── Test 3: Evaluate template for a single binding row ───────────────────

    #[test]
    fn test_evaluate_one_binding() {
        let clause = TemplateClause {
            prefix: Some("name=".to_string()),
            expr: GenerateLiteral::Var("name".to_string()),
            suffix: None,
        };
        let query = GenerateQuery::new(vec![clause], "?s foaf:name ?name .");
        let exec = GenerateExecutor::new(query);

        let row = bindings(&[("name", "Alice")]);
        let result = exec.evaluate_one(&row).expect("evaluation should succeed");

        assert_eq!(result.text, "name=Alice");
        assert_eq!(result.binding_count, 1);
    }

    // ── Test 4: Evaluate template over multiple rows ─────────────────────────

    #[test]
    fn test_evaluate_multiple_rows() {
        let clause = TemplateClause {
            prefix: Some("item=".to_string()),
            expr: GenerateLiteral::Var("item".to_string()),
            suffix: None,
        };
        let query = GenerateQuery::new(vec![clause], "?s ex:item ?item .");
        let exec = GenerateExecutor::new(query);

        let rows = vec![
            bindings(&[("item", "apple")]),
            bindings(&[("item", "banana")]),
            bindings(&[("item", "cherry")]),
        ];

        let results = exec.evaluate_all(&rows).expect("evaluation should succeed");

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].text, "item=apple");
        assert_eq!(results[1].text, "item=banana");
        assert_eq!(results[2].text, "item=cherry");
    }

    // ── Test 5: generate_text joins rows with separator ──────────────────────

    #[test]
    fn test_generate_text_joined() {
        let clause = TemplateClause::expr_only(GenerateLiteral::Var("v".to_string()));
        let query = GenerateQuery::new(vec![clause], "?s ?p ?v .");
        let exec = GenerateExecutor::new(query);

        let rows = vec![
            bindings(&[("v", "one")]),
            bindings(&[("v", "two")]),
            bindings(&[("v", "three")]),
        ];

        let text = exec
            .generate_text(&rows, ", ")
            .expect("generate_text should succeed");
        assert_eq!(text, "one, two, three");
    }

    // ── Test 6: UnboundVariable error when binding is missing ────────────────

    #[test]
    fn test_unbound_variable() {
        let clause = TemplateClause::expr_only(GenerateLiteral::Var("missing".to_string()));
        let query = GenerateQuery::new(vec![clause], "?s ?p ?missing .");
        let exec = GenerateExecutor::new(query);

        let row = bindings(&[("other", "value")]);
        let err = exec
            .evaluate_one(&row)
            .expect_err("should fail with UnboundVariable");

        assert!(
            matches!(err, GenerateError::UnboundVariable(ref v) if v == "missing"),
            "expected UnboundVariable(missing), got {err:?}"
        );
    }

    // ── Test 7: Concat literal evaluates correctly ───────────────────────────

    #[test]
    fn test_concat_literal() {
        let lit = GenerateLiteral::Concat(vec![
            GenerateLiteral::Text("x=".to_string()),
            GenerateLiteral::Var("x".to_string()),
        ]);

        let clause = TemplateClause::expr_only(lit);
        let query = GenerateQuery::new(vec![clause], "?s ?p ?x .");
        let exec = GenerateExecutor::new(query);

        let row = bindings(&[("x", "42")]);
        let result = exec.evaluate_one(&row).expect("evaluation should succeed");
        assert_eq!(result.text, "x=42");
    }

    // ── Test 8: Text-only literal ignores bindings ────────────────────────────

    #[test]
    fn test_literal_text_only() {
        let clause = TemplateClause::expr_only(GenerateLiteral::Text("hello".to_string()));
        let query = GenerateQuery::new(vec![clause], "");
        let exec = GenerateExecutor::new(query);

        // Even with completely unrelated bindings the output is always "hello".
        let row = bindings(&[("irrelevant", "data")]);
        let result = exec.evaluate_one(&row).expect("evaluation should succeed");
        assert_eq!(result.text, "hello");
        // No bindings were used.
        assert_eq!(result.binding_count, 0);
    }

    // ── Test 9: Empty template with one binding produces empty string ─────────

    #[test]
    fn test_empty_template() {
        let query = GenerateQuery::new(vec![], "?s ?p ?o .");
        let exec = GenerateExecutor::new(query);

        let row = bindings(&[("s", "sub"), ("p", "pred"), ("o", "obj")]);
        let result = exec.evaluate_one(&row).expect("evaluation should succeed");

        assert_eq!(result.text, "");
        assert_eq!(result.binding_count, 0);
    }

    // ── Test 10: Zero rows produces empty result set ──────────────────────────

    #[test]
    fn test_empty_rows() {
        let clause = TemplateClause::expr_only(GenerateLiteral::Var("x".to_string()));
        let query = GenerateQuery::new(vec![clause], "?s ?p ?x .");
        let exec = GenerateExecutor::new(query);

        let results = exec
            .evaluate_all(&[])
            .expect("evaluate_all on empty rows should succeed");
        assert!(results.is_empty(), "expected empty result set");

        let text = exec
            .generate_text(&[], " | ")
            .expect("generate_text on empty rows should succeed");
        assert_eq!(text, "");
    }

    // ── Test 11: TemplateClause prefix and suffix are preserved ──────────────

    #[test]
    fn test_prefix_and_suffix_on_clause() {
        let clause = TemplateClause {
            prefix: Some("[".to_string()),
            expr: GenerateLiteral::Var("val".to_string()),
            suffix: Some("]".to_string()),
        };
        let query = GenerateQuery::new(vec![clause], "?s ex:val ?val .");
        let exec = GenerateExecutor::new(query);

        let row = bindings(&[("val", "42")]);
        let result = exec.evaluate_one(&row).expect("evaluation should succeed");
        assert_eq!(result.text, "[42]");
    }

    // ── Test 12: WHERE body is correctly extracted by the parser ──────────────

    #[test]
    fn test_parse_where_body_extraction() {
        let src = r#"
            GENERATE { ?name }
            WHERE {
                ?s foaf:name ?name ;
                   rdf:type foaf:Person .
            }
        "#;

        let q = parse(src).expect("parse should succeed");
        let body = &q.where_body;

        // The body should contain the triple patterns (whitespace-trimmed).
        assert!(
            body.contains("foaf:name"),
            "expected 'foaf:name' in WHERE body, got: {body:?}"
        );
        assert!(
            body.contains("foaf:Person"),
            "expected 'foaf:Person' in WHERE body, got: {body:?}"
        );
        // The outer braces must NOT be part of the captured body.
        assert!(
            !body.starts_with('{'),
            "WHERE body must not start with '{{'"
        );
        assert!(!body.ends_with('}'), "WHERE body must not end with '}}'");
    }

    // ── Test 13: eval_literal public method works independently ───────────────

    #[test]
    fn test_eval_literal_public() {
        let query = GenerateQuery::new(vec![], "");
        let exec = GenerateExecutor::new(query);

        let lit = GenerateLiteral::Concat(vec![
            GenerateLiteral::Text("hello ".to_string()),
            GenerateLiteral::Var("who".to_string()),
        ]);

        let row = bindings(&[("who", "world")]);
        let out = exec
            .eval_literal(&lit, &row)
            .expect("eval_literal should succeed");
        assert_eq!(out, "hello world");
    }

    // ── Test 14: GenerateQuery::new sets sensible defaults ────────────────────

    #[test]
    fn test_generate_query_defaults() {
        let q = GenerateQuery::new(vec![], "BODY");
        assert!(q.prefix_decls.is_empty());
        assert!(q.template.is_empty());
        assert!(q.iterator.is_none());
        assert_eq!(q.where_body, "BODY");
        assert!(q.is_empty());
        assert_eq!(q.clause_count(), 0);
    }

    // ── Test 15: Multiple variable references in one row ─────────────────────

    #[test]
    fn test_multiple_vars_in_template() {
        let clauses = vec![
            TemplateClause {
                prefix: Some("first=".to_string()),
                expr: GenerateLiteral::Var("first".to_string()),
                suffix: Some(",".to_string()),
            },
            TemplateClause {
                prefix: Some("last=".to_string()),
                expr: GenerateLiteral::Var("last".to_string()),
                suffix: None,
            },
        ];
        let query = GenerateQuery::new(clauses, "?s foaf:firstName ?first ; foaf:lastName ?last .");
        let exec = GenerateExecutor::new(query);

        let row = bindings(&[("first", "John"), ("last", "Doe")]);
        let result = exec.evaluate_one(&row).expect("evaluation should succeed");
        assert_eq!(result.text, "first=John,last=Doe");
        assert_eq!(result.binding_count, 2);
    }
}
