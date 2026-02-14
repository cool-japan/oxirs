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
