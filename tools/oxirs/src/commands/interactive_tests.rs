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
