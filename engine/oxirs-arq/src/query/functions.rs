//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::queryparser_type::QueryParser;
use super::types::{Query, UpdateRequest};

/// Convenience function to parse a SPARQL query
pub fn parse_query(query_str: &str) -> Result<Query> {
    let mut parser = QueryParser::new();
    parser.parse(query_str)
}
/// Convenience function to parse a SPARQL UPDATE request
pub fn parse_update(update_str: &str) -> Result<UpdateRequest> {
    let mut parser = QueryParser::new();
    parser.parse_update(update_str)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Algebra, Variable};
    use crate::query::types::{QueryType, Token};
    use crate::update::UpdateOperation;
    #[test]
    fn test_simple_select_query() {
        let query_str = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?person ?name WHERE {
                ?person foaf:name ?name .
            }
        "#;
        let query = parse_query(query_str).unwrap();
        assert_eq!(query.query_type, QueryType::Select);
        assert_eq!(
            query.select_variables,
            vec![
                Variable::new("person").unwrap(),
                Variable::new("name").unwrap()
            ]
        );
        assert!(!query.prefixes.is_empty());
    }
    #[test]
    fn test_construct_query() {
        let query_str = r#"
            CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }
        "#;
        let query = parse_query(query_str).unwrap();
        assert_eq!(query.query_type, QueryType::Construct);
        assert_eq!(query.construct_template.len(), 1);
    }
    #[test]
    fn test_ask_query() {
        let query_str = r#"
            ASK WHERE { ?s ?p ?o }
        "#;
        let query = parse_query(query_str).unwrap();
        assert_eq!(query.query_type, QueryType::Ask);
    }
    #[test]
    fn test_tokenization() {
        let mut parser = QueryParser::new();
        parser.tokenize("SELECT ?x WHERE { ?x ?y ?z }").unwrap();
        println!("Tokens: {:?}", parser.tokens);
        let mut parser2 = QueryParser::new();
        parser2.tokenize("foaf:name").unwrap();
        println!("Prefixed name tokens: {:?}", parser2.tokens);
        assert!(matches!(parser.tokens[0], Token::Select));
        assert!(matches!(parser.tokens[1], Token::Variable(_)));
        assert!(matches!(parser.tokens[2], Token::Where));
    }
    #[test]
    fn test_union_query() {
        let query_str = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?name WHERE {
                { ?person foaf:name ?name }
                UNION
                { ?person rdfs:label ?name }
            }
        "#;
        let mut parser = QueryParser::new();
        parser.tokenize(query_str).unwrap();
        println!("Union query tokens: {:?}", parser.tokens);
        let query = parse_query(query_str)
            .map_err(|e| {
                eprintln!("Parse error: {e}");
                e
            })
            .unwrap();
        assert_eq!(query.query_type, QueryType::Select);
        assert_eq!(query.select_variables, vec![Variable::new("name").unwrap()]);
        match &query.where_clause {
            Algebra::Union { left, right } => {
                if let Algebra::Bgp(patterns) = left.as_ref() {
                    assert_eq!(patterns.len(), 1);
                } else {
                    panic!("Expected BGP on left side of union");
                }
                if let Algebra::Bgp(patterns) = right.as_ref() {
                    assert_eq!(patterns.len(), 1);
                } else {
                    panic!("Expected BGP on right side of union");
                }
            }
            _ => panic!("Expected Union algebra"),
        }
    }
    #[test]
    fn test_update_parsing() {
        let update_str = r#"INSERT DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }"#;
        let update_request = parse_update(update_str).unwrap();
        assert_eq!(update_request.operations.len(), 1);
        match &update_request.operations[0] {
            UpdateOperation::InsertData { data } => {
                assert_eq!(data.len(), 1);
            }
            _ => panic!("Expected InsertData operation"),
        }
    }
    #[test]
    fn test_update_tokenization() {
        let mut parser = QueryParser::new();
        parser.tokenize("INSERT DATA").unwrap();
        println!("UPDATE tokens: {:?}", parser.tokens);
        assert!(matches!(parser.tokens[0], Token::Insert));
        assert!(matches!(parser.tokens[1], Token::Data));
    }
    // ── Regression tests for the arq-front hardening pass ──────────────────

    /// A FILTER inside an OPTIONAL group must become the LeftJoin's filter, not
    /// stay nested inside the `right` operand.
    #[test]
    fn regression_optional_filter_lifted_into_leftjoin() {
        let query = r#"
            PREFIX : <http://example.org/>
            SELECT * WHERE {
                ?x :p ?v .
                OPTIONAL { ?y :q ?w . FILTER(?v > 3) }
            }
        "#;
        let parsed = parse_query(query).unwrap();
        // Find the LeftJoin node anywhere in the tree (top-level shape is a
        // Join of the first BGP and the LeftJoin).
        fn find_left_join(a: &Algebra) -> Option<&Algebra> {
            match a {
                Algebra::LeftJoin { .. } => Some(a),
                Algebra::Join { left, right } => {
                    find_left_join(left).or_else(|| find_left_join(right))
                }
                _ => None,
            }
        }
        let lj = find_left_join(&parsed.where_clause).expect("expected a LeftJoin");
        match lj {
            Algebra::LeftJoin { right, filter, .. } => {
                assert!(filter.is_some(), "OPTIONAL filter must be lifted");
                // The right operand must NOT itself be a Filter anymore.
                assert!(
                    !matches!(right.as_ref(), Algebra::Filter { .. }),
                    "filter must not remain nested in right operand"
                );
            }
            _ => unreachable!(),
        }
    }

    /// Numeric literals in term position must be typed by lexical form.
    #[test]
    fn regression_integer_literal_datatype() {
        use crate::algebra::Term;
        let query = r#"SELECT * WHERE { ?s <http://example.org/p> 42 }"#;
        let parsed = parse_query(query).unwrap();
        let bgp = match &parsed.where_clause {
            Algebra::Bgp(p) => p,
            other => panic!("expected BGP, got {other:?}"),
        };
        match &bgp[0].object {
            Term::Literal(lit) => {
                let dt = lit.datatype.as_ref().expect("datatype set");
                assert_eq!(dt.as_str(), "http://www.w3.org/2001/XMLSchema#integer");
            }
            other => panic!("expected literal object, got {other:?}"),
        }

        let query_dbl = r#"SELECT * WHERE { ?s <http://example.org/p> 4e2 }"#;
        let parsed_dbl = parse_query(query_dbl).unwrap();
        if let Algebra::Bgp(p) = &parsed_dbl.where_clause {
            if let Term::Literal(lit) = &p[0].object {
                assert_eq!(
                    lit.datatype.as_ref().unwrap().as_str(),
                    "http://www.w3.org/2001/XMLSchema#double"
                );
            } else {
                panic!("expected literal");
            }
        }
    }

    /// The number scanner must not swallow a statement-terminating '.' or an
    /// arithmetic operator.
    #[test]
    fn regression_number_scanner_stops_at_terminator() {
        let query = r#"SELECT * WHERE { <http://example.org/s> <http://example.org/p> 30. ?x <http://example.org/q> 40 }"#;
        let parsed = parse_query(query).unwrap();
        let bgp = match &parsed.where_clause {
            Algebra::Bgp(p) => p,
            other => panic!("expected BGP, got {other:?}"),
        };
        assert_eq!(bgp.len(), 2, "the '.' must terminate the first triple");
        if let crate::algebra::Term::Literal(lit) = &bgp[0].object {
            assert_eq!(lit.value, "30", "trailing '.' must not be absorbed");
        } else {
            panic!("expected literal");
        }
    }

    /// SERVICE SILENT must parse and set the silent flag.
    #[test]
    fn regression_service_silent() {
        let query = r#"
            SELECT * WHERE {
                SERVICE SILENT <http://example.org/sparql> { ?s ?p ?o }
            }
        "#;
        let parsed = parse_query(query).unwrap();
        fn find_service(a: &Algebra) -> Option<bool> {
            match a {
                Algebra::Service { silent, .. } => Some(*silent),
                Algebra::Join { left, right } | Algebra::Union { left, right } => {
                    find_service(left).or_else(|| find_service(right))
                }
                _ => None,
            }
        }
        assert_eq!(find_service(&parsed.where_clause), Some(true));
    }

    /// VALUES ... UNDEF must leave the variable unbound (absent) for that row.
    #[test]
    fn regression_values_undef_unbound() {
        let query = r#"
            PREFIX : <http://example.org/>
            SELECT * WHERE {
                VALUES (?a ?b) { (:x UNDEF) (:y :z) }
            }
        "#;
        let parsed = parse_query(query).unwrap();
        fn find_values(a: &Algebra) -> Option<&Algebra> {
            match a {
                Algebra::Values { .. } => Some(a),
                Algebra::Join { left, right } => find_values(left).or_else(|| find_values(right)),
                _ => None,
            }
        }
        match find_values(&parsed.where_clause).expect("VALUES node") {
            Algebra::Values {
                variables,
                bindings,
            } => {
                assert_eq!(variables.len(), 2);
                // First row: ?a bound, ?b absent (UNDEF).
                let b0 = &bindings[0];
                assert!(b0.contains_key(&variables[0]));
                assert!(
                    !b0.contains_key(&variables[1]),
                    "UNDEF must omit the variable"
                );
                // Second row: both bound.
                let b1 = &bindings[1];
                assert!(b1.contains_key(&variables[0]));
                assert!(b1.contains_key(&variables[1]));
            }
            _ => unreachable!(),
        }
    }

    /// WITH <g> DELETE WHERE must scope the delete to <g>, not the default graph.
    #[test]
    fn regression_with_delete_where_scoped_to_graph() {
        let update = r#"WITH <http://example.org/g> DELETE WHERE { ?s ?p ?o }"#;
        let req = parse_update(update).unwrap();
        match &req.operations[0] {
            UpdateOperation::DeleteWhere { pattern } => match pattern.as_ref() {
                Algebra::Graph { graph, .. } => match graph {
                    crate::algebra::Term::Iri(n) => {
                        assert_eq!(n.as_str(), "http://example.org/g")
                    }
                    other => panic!("expected IRI graph, got {other:?}"),
                },
                other => panic!("DELETE WHERE pattern must be scoped by GRAPH, got {other:?}"),
            },
            other => panic!("expected DeleteWhere, got {other:?}"),
        }
    }

    /// GRAPH blocks inside INSERT DATA must parse and attach the graph to quads.
    #[test]
    fn regression_graph_block_in_insert_data() {
        let update = r#"INSERT DATA {
            <http://example.org/s> <http://example.org/p> <http://example.org/o> .
            GRAPH <http://example.org/g> { <http://example.org/a> <http://example.org/b> <http://example.org/c> }
        }"#;
        let req = parse_update(update).unwrap();
        match &req.operations[0] {
            UpdateOperation::InsertData { data } => {
                assert_eq!(data.len(), 2);
                assert!(data[0].graph.is_none(), "first quad is in default graph");
                match &data[1].graph {
                    Some(crate::update::GraphReference::Iri(iri)) => {
                        assert_eq!(iri, "http://example.org/g")
                    }
                    other => panic!("second quad must be in named graph, got {other:?}"),
                }
            }
            other => panic!("expected InsertData, got {other:?}"),
        }
    }

    /// Trailing garbage after a complete query is a syntax error.
    #[test]
    fn regression_trailing_tokens_rejected() {
        let query = r#"SELECT * WHERE { ?s ?p ?o } this is nonsense"#;
        assert!(
            parse_query(query).is_err(),
            "trailing tokens must be rejected"
        );
    }

    /// A non-integer LIMIT is a parse error, not a silent "return everything".
    #[test]
    fn regression_limit_non_integer_rejected() {
        assert!(parse_query(r#"SELECT * WHERE { ?s ?p ?o } LIMIT 2.5"#).is_err());
        // A valid integer LIMIT still works and OFFSET-then-LIMIT order too.
        let ok = parse_query(r#"SELECT * WHERE { ?s ?p ?o } LIMIT 5 OFFSET 2"#).unwrap();
        assert_eq!(ok.limit, Some(5));
        assert_eq!(ok.offset, Some(2));
    }

    /// CONSTRUCT with `WHERE` on its own line after the template's closing
    /// `}`. The tokenizer emits a `Token::Newline` per line break and nothing
    /// upstream filters it out of the stream; `parse_construct_query` used to
    /// check for `WHERE`/`{` without ever skipping that token first, so this
    /// extremely common layout ("template on one line, `WHERE` on the next")
    /// failed with "Expected LeftBrace, found Newline".
    #[test]
    fn regression_construct_where_on_new_line() {
        let query = "CONSTRUCT { ?s ?p ?o }\nWHERE { ?s ?p ?o }";
        let parsed = parse_query(query)
            .map_err(|e| format!("must parse WHERE on a new line: {e}"))
            .unwrap();
        assert_eq!(parsed.query_type, QueryType::Construct);
        assert_eq!(parsed.construct_template.len(), 1);
        assert!(matches!(parsed.where_clause, Algebra::Bgp(ref p) if p.len() == 1));
    }

    /// A CONSTRUCT template whose last triple has no trailing `.` before the
    /// closing `}` (itself on its own line) must still parse — exercising the
    /// same gap as `regression_construct_where_on_new_line` but at the
    /// template/`}` boundary instead of the `}`/`WHERE` boundary.
    #[test]
    fn regression_construct_trailing_newline_before_template_close() {
        let query = "CONSTRUCT {\n  ?s ?p ?o\n}\nWHERE { ?s ?p ?o }";
        let parsed = parse_query(query).unwrap();
        assert_eq!(parsed.construct_template.len(), 1);
    }

    /// Multiple blank lines between the template and `WHERE` are just a run
    /// of consecutive `Token::Newline`s and must be skipped in full, not just
    /// one at a time.
    #[test]
    fn regression_construct_blank_line_before_where() {
        let query = "CONSTRUCT { ?s ?p ?o }\n\n\nWHERE { ?s ?p ?o }";
        let parsed = parse_query(query).unwrap();
        assert_eq!(parsed.construct_template.len(), 1);
    }

    /// A CRLF line ending between the template and `WHERE`. The tokenizer
    /// drops `\r` as plain whitespace and emits `Newline` only for the `\n`,
    /// so this exercises the identical code path as the bare-`\n` case, but
    /// is worth locking in explicitly given CRLF is common in
    /// Windows-authored or copy-pasted queries.
    #[test]
    fn regression_construct_crlf_before_where() {
        let query = "CONSTRUCT { ?s ?p ?o }\r\nWHERE { ?s ?p ?o }";
        let parsed = parse_query(query).unwrap();
        assert_eq!(parsed.construct_template.len(), 1);
    }

    /// The `CONSTRUCT WHERE { ... }` shorthand (SPARQL 1.1 §16.2.4) with a
    /// newline both after `CONSTRUCT` and after `WHERE`. The shorthand form
    /// requires the `WHERE` keyword (dropping it would make `CONSTRUCT { … }`
    /// ambiguous with the explicit-template form), so this also exercises the
    /// `expect_token(Token::Where)` branch rather than only `match_token`.
    #[test]
    fn regression_construct_where_shorthand_newline() {
        let query = "CONSTRUCT\nWHERE\n{ ?s ?p ?o }";
        let parsed = parse_query(query).unwrap();
        // Shorthand: the template is the WHERE block's BGP itself.
        assert_eq!(parsed.construct_template.len(), 1);
        assert!(matches!(parsed.where_clause, Algebra::Bgp(ref p) if p.len() == 1));
    }

    /// Same-line CONSTRUCT (both the explicit-template and shorthand forms)
    /// must keep working after the newline-skipping fix above — this is the
    /// non-regression counterpart to the `_newline`/`_new_line` tests.
    #[test]
    fn regression_construct_same_line_forms_still_parse() {
        let explicit = parse_query("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }").unwrap();
        assert_eq!(explicit.construct_template.len(), 1);

        let shorthand = parse_query("CONSTRUCT WHERE { ?s ?p ?o }").unwrap();
        assert_eq!(shorthand.construct_template.len(), 1);

        // WHERE is optional in the explicit-template form.
        let no_where = parse_query("CONSTRUCT { ?s ?p ?o } { ?s ?p ?o }").unwrap();
        assert_eq!(no_where.construct_template.len(), 1);
    }

    /// `FROM`/`FROM NAMED` dataset clauses spread across lines between the
    /// CONSTRUCT template and `WHERE`. `parse_dataset_clause` is shared by
    /// every query form, so this also guards the fix made there.
    #[test]
    fn regression_construct_dataset_clause_multiline() {
        let query = "CONSTRUCT { ?s ?p ?o }\nFROM <http://example.org/g1>\nFROM NAMED <http://example.org/g2>\nWHERE { ?s ?p ?o }";
        let parsed = parse_query(query).unwrap();
        assert_eq!(parsed.dataset.default_graphs.len(), 1);
        assert_eq!(parsed.dataset.named_graphs.len(), 1);
    }

    /// Bonus coverage: `parse_dataset_clause` is shared with SELECT, so
    /// fixing it for CONSTRUCT incidentally fixes the identical
    /// `SELECT ?vars \n WHERE { … }` layout (an extremely common way to
    /// format a SPARQL query) — lock that in too.
    #[test]
    fn regression_select_where_on_new_line() {
        let query = "SELECT ?s ?p ?o\nWHERE { ?s ?p ?o }";
        let parsed = parse_query(query).unwrap();
        assert_eq!(
            parsed.select_variables,
            vec![
                Variable::new("s").unwrap(),
                Variable::new("p").unwrap(),
                Variable::new("o").unwrap()
            ]
        );
    }

    /// `ASK` with `WHERE { … }` on its own line — the same "head keyword on
    /// one line, `WHERE`/group on the next" layout the CONSTRUCT/SELECT fix
    /// covers. `parse_ask_query` previously never skipped `Token::Newline`
    /// itself (it only worked incidentally through `parse_dataset_clause`'s
    /// own internal skip), so this locks in the now-explicit skip.
    #[test]
    fn regression_ask_where_on_new_line() {
        let query = "ASK\nWHERE { ?s ?p ?o }";
        let parsed = parse_query(query)
            .map_err(|e| format!("must parse WHERE on a new line: {e}"))
            .unwrap();
        assert_eq!(parsed.query_type, QueryType::Ask);
    }

    /// Blank lines and a CRLF line ending between `ASK` and `WHERE`, plus the
    /// previously-undiscovered gap of a newline between the `WHERE` keyword
    /// itself and the group's opening `{` (`ASK WHERE\n{ … }`) — `\r` is
    /// dropped as plain whitespace by the tokenizer, so only the `\n` half of
    /// a CRLF ever surfaces as `Token::Newline`.
    #[test]
    fn regression_ask_blank_line_and_crlf() {
        let blank_lines = parse_query("ASK\n\n\nWHERE { ?s ?p ?o }")
            .map_err(|e| format!("must parse blank lines before WHERE: {e}"))
            .unwrap();
        assert_eq!(blank_lines.query_type, QueryType::Ask);

        let crlf = parse_query("ASK\r\nWHERE { ?s ?p ?o }")
            .map_err(|e| format!("must parse CRLF before WHERE: {e}"))
            .unwrap();
        assert_eq!(crlf.query_type, QueryType::Ask);

        let where_brace_newline = parse_query("ASK WHERE\n{ ?s ?p ?o }")
            .map_err(|e| format!("must parse a newline between WHERE and '{{': {e}"))
            .unwrap();
        assert_eq!(where_brace_newline.query_type, QueryType::Ask);
    }

    /// `DESCRIBE ?x` with `WHERE { … }` on its own line, exercising the same
    /// gap as `regression_ask_where_on_new_line` in the DESCRIBE target-list
    /// / dataset-clause / WHERE-keyword boundary, plus the WHERE/`{` newline
    /// boundary found alongside the ASK fix.
    #[test]
    fn regression_describe_var_where_on_new_line() {
        let query = "DESCRIBE ?x\nWHERE { ?x ?p ?o }";
        let parsed = parse_query(query)
            .map_err(|e| format!("must parse WHERE on a new line: {e}"))
            .unwrap();
        assert_eq!(parsed.query_type, QueryType::Describe);
        assert_eq!(parsed.describe_targets.len(), 1);

        let where_brace_newline = parse_query("DESCRIBE ?x WHERE\n{ ?x ?p ?o }")
            .map_err(|e| format!("must parse a newline between WHERE and '{{': {e}"))
            .unwrap();
        assert_eq!(where_brace_newline.query_type, QueryType::Describe);
    }

    /// `DESCRIBE <iri>` with no `WHERE` clause at all — SPARQL 1.1 §16.4
    /// makes the WHERE clause fully optional for DESCRIBE, so this must keep
    /// parsing to an empty (`Algebra::Zero`) where-clause after the
    /// newline-skipping hardening above, not regress into requiring one.
    #[test]
    fn regression_describe_iri_no_where_still_parses() {
        let parsed = parse_query("DESCRIBE <http://example.org/x>")
            .map_err(|e| format!("DESCRIBE with no WHERE must still parse: {e}"))
            .unwrap();
        assert_eq!(parsed.query_type, QueryType::Describe);
        assert_eq!(parsed.describe_targets.len(), 1);
        assert!(matches!(parsed.where_clause, Algebra::Zero));

        // A trailing newline after the IRI (no WHERE) must not confuse the
        // "end of query" check either.
        let with_trailing_newline = parse_query("DESCRIBE <http://example.org/x>\n")
            .map_err(|e| format!("trailing newline with no WHERE must still parse: {e}"))
            .unwrap();
        assert_eq!(with_trailing_newline.describe_targets.len(), 1);
    }

    /// Same-line ASK/DESCRIBE (both with and without `WHERE`) must keep
    /// working after the newline-skipping fixes above — the non-regression
    /// counterpart to the `_new_line`/`_newline` tests, mirroring
    /// `regression_construct_same_line_forms_still_parse`.
    #[test]
    fn regression_ask_describe_same_line_forms_still_parse() {
        let ask_with_where = parse_query("ASK WHERE { ?s ?p ?o }").unwrap();
        assert_eq!(ask_with_where.query_type, QueryType::Ask);

        // WHERE is optional for ASK.
        let ask_no_where = parse_query("ASK { ?s ?p ?o }").unwrap();
        assert_eq!(ask_no_where.query_type, QueryType::Ask);

        let describe_with_where = parse_query("DESCRIBE ?x WHERE { ?x ?p ?o }").unwrap();
        assert_eq!(describe_with_where.query_type, QueryType::Describe);

        // WHERE is optional for DESCRIBE.
        let describe_no_where = parse_query("DESCRIBE <http://example.org/x>").unwrap();
        assert_eq!(describe_no_where.query_type, QueryType::Describe);

        let describe_star = parse_query("DESCRIBE * WHERE { ?s ?p ?o }").unwrap();
        assert!(describe_star.describe_all);
    }

    /// The `WHERE`/`{` newline gap discovered while hardening ASK/DESCRIBE
    /// also affects SELECT (`parse_select_query` never skipped a newline
    /// between the optional `WHERE` keyword and the group's opening `{`) —
    /// lock in the fix there too since the sweep is meant to cover every
    /// query head consistently.
    #[test]
    fn regression_select_where_newline_before_brace() {
        let query = "SELECT ?s WHERE\n{ ?s ?p ?o }";
        let parsed = parse_query(query)
            .map_err(|e| format!("must parse a newline between WHERE and '{{': {e}"))
            .unwrap();
        assert_eq!(parsed.query_type, QueryType::Select);
    }

    // ── Regression tests: UPDATE newline robustness ────────────────────────
    //
    // The SELECT/CONSTRUCT/ASK/DESCRIBE query heads were hardened above to
    // tolerate a `Token::Newline` between a keyword and its `{`/`WHERE` (see
    // `regression_construct_where_on_new_line` and neighbours). The SPARQL
    // UPDATE parsers in `queryparser_parsing_2.rs` (`parse_insert_where`,
    // `parse_delete_where`, `parse_delete_insert_where`) plus their dispatch
    // in `queryparser_parsing_12.rs`/`_13.rs` (`parse_insert_operation`,
    // `parse_delete_operation`) and the `WITH`/prologue handling in
    // `parse_update_request` had zero newline-skip calls, so the extremely
    // common multi-line UPDATE layout used against the fuseki `/update`
    // endpoint failed to parse. These tests lock in the fix.

    /// `DELETE { … }` / `INSERT { … }` / `WHERE { … }` each on their own
    /// line — the canonical multi-line UPDATE layout.
    #[test]
    fn regression_update_delete_insert_where_multiline() {
        let update = "DELETE { ?s ?p ?o }\nINSERT { ?s ?p ?new }\nWHERE  { ?s ?p ?o }";
        let req = parse_update(update)
            .map_err(|e| format!("must parse a multi-line DELETE/INSERT/WHERE: {e}"))
            .unwrap();
        match &req.operations[0] {
            UpdateOperation::DeleteInsertWhere {
                delete_template,
                insert_template,
                ..
            } => {
                assert_eq!(delete_template.len(), 1);
                assert_eq!(insert_template.len(), 1);
            }
            other => panic!("expected DeleteInsertWhere, got {other:?}"),
        }
    }

    /// `INSERT DATA\n{ … }`: a newline between the `DATA` keyword and the
    /// block's opening `{`.
    #[test]
    fn regression_update_insert_data_newline_before_brace() {
        let update =
            "INSERT DATA\n{ <http://example.org/s> <http://example.org/p> <http://example.org/o> }";
        let req = parse_update(update)
            .map_err(|e| format!("must parse a newline between DATA and '{{': {e}"))
            .unwrap();
        match &req.operations[0] {
            UpdateOperation::InsertData { data } => assert_eq!(data.len(), 1),
            other => panic!("expected InsertData, got {other:?}"),
        }
    }

    /// `DELETE WHERE\n{ … }`: the `DELETE WHERE` shorthand with a newline
    /// before the block's opening `{`. Also exercises `parse_delete_operation`
    /// dispatching correctly to `parse_delete_where` when `DELETE` and
    /// `WHERE` are separated by a newline (`DELETE\nWHERE { … }`), since a
    /// missed skip there would previously misdispatch into
    /// `parse_delete_insert_where` instead.
    #[test]
    fn regression_update_delete_where_shorthand_newline() {
        let update = "DELETE WHERE\n{ ?s ?p ?o }";
        let req = parse_update(update)
            .map_err(|e| format!("must parse a newline between WHERE and '{{': {e}"))
            .unwrap();
        assert!(matches!(
            &req.operations[0],
            UpdateOperation::DeleteWhere { .. }
        ));

        let delete_newline_where = parse_update("DELETE\nWHERE { ?s ?p ?o }")
            .map_err(|e| format!("must parse a newline between DELETE and WHERE: {e}"))
            .unwrap();
        assert!(matches!(
            &delete_newline_where.operations[0],
            UpdateOperation::DeleteWhere { .. }
        ));
    }

    /// `WITH <g>` on its own line, followed by a multi-line
    /// DELETE/INSERT/WHERE. `USING` is not currently accepted by the parser
    /// at all (no `Token::Using` handling exists in any UPDATE parse
    /// function — a separate, pre-existing gap out of scope for this
    /// newline-robustness fix), so this test covers `WITH` only.
    #[test]
    fn regression_update_with_clause_multiline() {
        let update =
            "WITH <http://example.org/g>\nDELETE { ?s ?p ?o }\nINSERT { ?s ?p ?new }\nWHERE { ?s ?p ?o }";
        let req = parse_update(update)
            .map_err(|e| format!("must parse a multi-line WITH/DELETE/INSERT/WHERE: {e}"))
            .unwrap();
        match &req.operations[0] {
            UpdateOperation::DeleteInsertWhere { pattern, .. } => match pattern.as_ref() {
                Algebra::Graph { graph, .. } => match graph {
                    crate::algebra::Term::Iri(n) => {
                        assert_eq!(n.as_str(), "http://example.org/g")
                    }
                    other => panic!("expected IRI graph, got {other:?}"),
                },
                other => panic!("WHERE pattern must be scoped by GRAPH, got {other:?}"),
            },
            other => panic!("expected DeleteInsertWhere, got {other:?}"),
        }
    }

    /// Multiple `PREFIX` declarations in an UPDATE prologue, each on its own
    /// line: `parse_update_request`'s prologue loop previously had no
    /// `Token::Newline` arm, so it mistook the newline after the first
    /// `PREFIX` line for "prologue over" and broke out before consuming the
    /// second `PREFIX` — which then surfaced as "Expected UPDATE operation"
    /// once the operations loop saw a stray `Token::Prefix`.
    #[test]
    fn regression_update_prefix_prologue_multiline() {
        let update = "PREFIX a: <http://example.org/a#>\nPREFIX b: <http://example.org/b#>\nINSERT DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }";
        let req = parse_update(update)
            .map_err(|e| format!("must parse multiple PREFIX lines in an UPDATE prologue: {e}"))
            .unwrap();
        assert_eq!(req.prefixes.len(), 2);
        assert_eq!(req.operations.len(), 1);
    }

    /// Same-line UPDATE forms (INSERT DATA, DELETE DATA, DELETE WHERE
    /// shorthand, DELETE/INSERT/WHERE, and WITH) must keep working after the
    /// newline-skipping fixes above — the non-regression counterpart to the
    /// `_multiline`/`_newline` tests, mirroring
    /// `regression_construct_same_line_forms_still_parse`.
    #[test]
    fn regression_update_same_line_forms_still_parse() {
        let insert_data = parse_update(
            "INSERT DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }",
        )
        .unwrap();
        assert!(matches!(
            &insert_data.operations[0],
            UpdateOperation::InsertData { .. }
        ));

        let delete_data = parse_update(
            "DELETE DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }",
        )
        .unwrap();
        assert!(matches!(
            &delete_data.operations[0],
            UpdateOperation::DeleteData { .. }
        ));

        let delete_where = parse_update("DELETE WHERE { ?s ?p ?o }").unwrap();
        assert!(matches!(
            &delete_where.operations[0],
            UpdateOperation::DeleteWhere { .. }
        ));

        let delete_insert_where =
            parse_update("DELETE { ?s ?p ?o } INSERT { ?s ?p ?new } WHERE { ?s ?p ?o }").unwrap();
        assert!(matches!(
            &delete_insert_where.operations[0],
            UpdateOperation::DeleteInsertWhere { .. }
        ));

        let insert_where = parse_update("INSERT { ?s ?p ?new } WHERE { ?s ?p ?o }").unwrap();
        assert!(matches!(
            &insert_where.operations[0],
            UpdateOperation::InsertWhere { .. }
        ));

        let with_clause =
            parse_update("WITH <http://example.org/g> DELETE WHERE { ?s ?p ?o }").unwrap();
        assert!(matches!(
            &with_clause.operations[0],
            UpdateOperation::DeleteWhere { .. }
        ));
    }

    #[test]
    fn test_multiple_union_query() {
        let query_str = r#"
            PREFIX : <http://example.org/>
            SELECT ?x WHERE {
                { ?x a :ClassA }
                UNION
                { ?x a :ClassB }
                UNION
                { ?x a :ClassC }
            }
        "#;
        let query = parse_query(query_str).unwrap();
        match &query.where_clause {
            Algebra::Union { left: _, right } => match right.as_ref() {
                Algebra::Union { .. } => {}
                _ => panic!("Expected nested Union on right side"),
            },
            _ => panic!("Expected Union algebra"),
        }
    }
}
