//! GQL (ISO/IEC 39075:2024) → SPARQL bridge.
//!
//! This module provides a parser for a meaningful subset of GQL MATCH queries
//! and a translator that emits equivalent SPARQL 1.1 SELECT queries.
//!
//! # Supported GQL subset
//!
//! ```text
//! GqlQuery     ::= MATCH GraphPattern (WHERE Predicate)? RETURN ReturnClause
//! GraphPattern ::= NodePattern (EdgePattern NodePattern)*
//! NodePattern  ::= '(' VarOpt LabelFilter? PropFilter? ')'
//! EdgePattern  ::= '-[' VarOpt LabelFilter? ']->'
//!                | '<-[' VarOpt LabelFilter? ']-'
//! VarOpt       ::= Ident?
//! LabelFilter  ::= ':' Ident
//! PropFilter   ::= '{' PropKV (',' PropKV)* '}'
//! PropKV       ::= Ident ':' Literal
//! ReturnClause ::= Ident (',' Ident)*
//! Predicate    ::= Ident '.' Ident '=' Literal
//! Literal      ::= '"' [^"]* '"' | NUMBER | 'true' | 'false'
//! ```
//!
//! # Quick example
//!
//! ```rust
//! use oxirs_arq::GqlToSparqlTranslator;
//!
//! let t = GqlToSparqlTranslator::new();
//! let sparql = t.translate(
//!     r#"MATCH (x:Person)-[e:knows]->(y:Person) RETURN x, y"#
//! ).unwrap();
//! assert!(sparql.starts_with("SELECT ?x ?y"));
//! ```

pub mod ast;
pub mod parser;
pub mod translator;

pub use ast::{
    EdgeDirection, EdgePattern, GqlLiteral, GqlPredicate, GqlQuery, NodePattern, PathSegment,
};
pub use translator::GqlToSparqlTranslator;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during GQL parsing or translation.
#[derive(Debug, thiserror::Error)]
pub enum GqlTranslateError {
    /// A syntax error encountered while parsing the GQL input.
    #[error("Parse error at position {pos}: {msg}")]
    ParseError {
        /// Byte offset in the original GQL string where the error was detected.
        pos: usize,
        /// Human-readable description of the error.
        msg: String,
    },

    /// An error that occurs during AST → SPARQL translation (e.g. invalid
    /// structure that the parser accepted but the translator cannot handle).
    #[error("Translation error: {msg}")]
    TranslateError {
        /// Human-readable description of the translation error.
        msg: String,
    },

    /// The RETURN clause is empty — the resulting SPARQL would be invalid.
    #[error("Empty RETURN clause")]
    EmptyReturn,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{ast::*, GqlToSparqlTranslator, GqlTranslateError};
    use crate::gql::parser::parse_gql;

    // Helper: create a default translator.
    fn trans() -> GqlToSparqlTranslator {
        GqlToSparqlTranslator::new()
    }

    // ── Test 1: Simple node pattern ─────────────────────────────────────────

    #[test]
    fn test_simple_node_pattern() {
        let sparql = trans()
            .translate("MATCH (x:Person) RETURN x")
            .expect("translation should succeed");

        assert!(
            sparql.starts_with("SELECT ?x WHERE"),
            "expected SELECT ?x WHERE, got: {sparql}"
        );
        assert!(
            sparql.contains("?x a <http://example.org/Person>"),
            "expected rdf:type triple, got: {sparql}"
        );
    }

    // ── Test 2: Forward edge pattern ────────────────────────────────────────

    #[test]
    fn test_forward_edge_pattern() {
        let sparql = trans()
            .translate("MATCH (x:Person)-[e:knows]->(y:Person) RETURN x, y")
            .expect("translation should succeed");

        assert!(
            sparql.starts_with("SELECT ?x ?y WHERE"),
            "expected SELECT ?x ?y WHERE, got: {sparql}"
        );
        assert!(
            sparql.contains("?x <http://example.org/knows> ?y"),
            "expected knows triple, got: {sparql}"
        );
        // Both nodes should have rdf:type triples.
        assert!(sparql.contains("?x a <http://example.org/Person>"));
        assert!(sparql.contains("?y a <http://example.org/Person>"));
    }

    // ── Test 3: Backward edge ───────────────────────────────────────────────

    #[test]
    fn test_backward_edge_pattern() {
        let sparql = trans()
            .translate("MATCH (x:Person)<-[e:knows]-(y:Person) RETURN x, y")
            .expect("translation should succeed");

        // Backward: y knows x → ?y <knows> ?x
        assert!(
            sparql.contains("?y <http://example.org/knows> ?x"),
            "expected reversed knows triple, got: {sparql}"
        );
    }

    // ── Test 4: Anonymous node `(_)` ────────────────────────────────────────

    #[test]
    fn test_anonymous_node() {
        let query = parse_gql("MATCH (_:Thing) RETURN x").expect("parse should succeed");

        // The anonymous `_` should be replaced by a fresh generated variable.
        match &query.match_pattern[0] {
            PathSegment::Node(n) => {
                let v = n.var.as_deref().unwrap_or("");
                assert!(
                    v.starts_with("_anon"),
                    "expected generated anon variable, got: {v}"
                );
            }
            PathSegment::Edge(_) => panic!("expected Node segment"),
        }
    }

    // ── Test 5: Property filter adds property triple ─────────────────────────

    #[test]
    fn test_property_filter() {
        let sparql = trans()
            .translate(r#"MATCH (x:Person {name: "Alice"}) RETURN x"#)
            .expect("translation should succeed");

        assert!(
            sparql.contains(r#"?x <http://example.org/name> "Alice""#),
            "expected name property triple, got: {sparql}"
        );
    }

    // ── Test 6: WHERE clause adds FILTER ────────────────────────────────────

    #[test]
    fn test_where_adds_filter() {
        let sparql = trans()
            .translate("MATCH (x:Person) WHERE x.age = 30 RETURN x")
            .expect("translation should succeed");

        assert!(
            sparql.contains("?x <http://example.org/age> ?x_age"),
            "expected auxiliary age triple, got: {sparql}"
        );
        assert!(
            sparql.contains("FILTER(?x_age = 30)"),
            "expected FILTER, got: {sparql}"
        );
    }

    // ── Test 7: Three-node chain ─────────────────────────────────────────────

    #[test]
    fn test_three_node_chain() {
        let sparql = trans()
            .translate(
                "MATCH (a:Person)-[e1:knows]->(b:Person)-[e2:likes]->(c:Thing) RETURN a, b, c",
            )
            .expect("translation should succeed");

        assert!(sparql.starts_with("SELECT ?a ?b ?c WHERE"));
        assert!(sparql.contains("?a <http://example.org/knows> ?b"));
        assert!(sparql.contains("?b <http://example.org/likes> ?c"));
    }

    // ── Test 8: Integer literal not quoted ──────────────────────────────────

    #[test]
    fn test_integer_literal_unquoted() {
        let sparql = trans()
            .translate("MATCH (x:Product {age: 42}) RETURN x")
            .expect("translation should succeed");

        // Integer must appear bare, not as `"42"`.
        assert!(
            sparql.contains("?x <http://example.org/age> 42"),
            "expected bare integer literal, got: {sparql}"
        );
        assert!(
            !sparql.contains("\"42\""),
            "integer must not be quoted, got: {sparql}"
        );
    }

    // ── Test 9: Boolean literal → XSD typed ─────────────────────────────────

    #[test]
    fn test_boolean_literal_xsd() {
        let lit = GqlToSparqlTranslator::new().literal_to_sparql(&GqlLiteral::Bool(true));
        assert_eq!(lit, "\"true\"^^xsd:boolean");

        let lit_false = GqlToSparqlTranslator::new().literal_to_sparql(&GqlLiteral::Bool(false));
        assert_eq!(lit_false, "\"false\"^^xsd:boolean");
    }

    // ── Test 10: Multiple return variables ───────────────────────────────────

    #[test]
    fn test_multiple_return_vars() {
        let sparql = trans()
            .translate("MATCH (x:A)-[e1:rel1]->(y:B)-[e2:rel2]->(z:C) RETURN x, y, z")
            .expect("translation should succeed");

        assert!(
            sparql.starts_with("SELECT ?x ?y ?z WHERE"),
            "expected ?x ?y ?z projection, got: {sparql}"
        );
    }

    // ── Test 11: Parse error on invalid token ────────────────────────────────

    #[test]
    fn test_parse_error_invalid_token() {
        // `@` is not a valid GQL character.
        let result = trans().translate("MATCH @invalid RETURN x");
        assert!(
            matches!(result, Err(GqlTranslateError::ParseError { .. })),
            "expected ParseError, got: {result:?}"
        );
    }

    // ── Test 12: Custom prefix used in translation ───────────────────────────

    #[test]
    fn test_custom_prefix() {
        let t = GqlToSparqlTranslator::with_prefix("http://kg.example.com/");
        let sparql = t
            .translate("MATCH (x:Person) RETURN x")
            .expect("translation should succeed");

        assert!(
            sparql.contains("?x a <http://kg.example.com/Person>"),
            "expected custom prefix in IRI, got: {sparql}"
        );
        assert!(
            !sparql.contains("http://example.org/"),
            "default prefix should NOT appear, got: {sparql}"
        );
    }

    // ── Bonus test 13: Empty return clause error ─────────────────────────────

    #[test]
    fn test_empty_return_error() {
        // Syntactically we cannot produce an empty RETURN list through normal
        // parsing (RETURN always requires at least one identifier), so we test
        // the error variant directly.
        let error = GqlTranslateError::EmptyReturn;
        let msg = error.to_string();
        assert_eq!(msg, "Empty RETURN clause");
    }

    // ── Bonus test 14: Node with no var and no label ─────────────────────────

    #[test]
    fn test_fully_anonymous_node() {
        let sparql = trans()
            .translate("MATCH (:Person)-[e:knows]->(y) RETURN y")
            .expect("translation should succeed");

        // y should appear in SELECT.
        assert!(sparql.contains("SELECT ?y WHERE"));
        // The left node has label Person but no variable binding.
        assert!(sparql.contains("<http://example.org/Person>"));
    }

    // ── Bonus test 15: Edge with no label uses edge var ──────────────────────

    #[test]
    fn test_edge_without_label() {
        // `-[rel]->` has a variable but no label.
        let sparql = trans()
            .translate("MATCH (x)-[rel]->(y) RETURN x, y")
            .expect("translation should succeed");

        // The predicate slot should be `?rel`.
        assert!(
            sparql.contains("?x ?rel ?y"),
            "expected edge var as predicate, got: {sparql}"
        );
    }

    // ── Bonus test 16: Float literal ─────────────────────────────────────────

    #[test]
    fn test_float_literal() {
        let sparql = trans()
            .translate("MATCH (x:Product {price: 9.99}) RETURN x")
            .expect("translation should succeed");

        assert!(
            sparql.contains("9.99"),
            "expected float literal in output, got: {sparql}"
        );
    }

    // ── Bonus test 17: Negative integer literal ──────────────────────────────

    #[test]
    fn test_negative_integer_in_where() {
        let sparql = trans()
            .translate("MATCH (x:Item) WHERE x.delta = -5 RETURN x")
            .expect("translation should succeed");

        assert!(
            sparql.contains("FILTER(?x_delta = -5)"),
            "expected negative integer in filter, got: {sparql}"
        );
    }

    // ── Bonus test 18: AST round-trip through parse_gql ─────────────────────

    #[test]
    fn test_ast_roundtrip() {
        let q = parse_gql("MATCH (a:Person)-[e:knows]->(b:Person) WHERE a.age = 25 RETURN a, b")
            .expect("parse should succeed");

        assert_eq!(q.return_vars, vec!["a", "b"]);
        assert_eq!(q.match_pattern.len(), 3);

        let pred = q.where_pred.as_ref().expect("should have WHERE predicate");
        assert_eq!(pred.var, "a");
        assert_eq!(pred.prop, "age");
        assert_eq!(pred.value, GqlLiteral::Int(25));
    }
}
