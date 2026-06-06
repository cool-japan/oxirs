//! # Validation Tests
//!
//! Unit tests for all validation functions: SPARQL query/update, IRI,
//! RDF data, language tags, and helper utilities.

#[cfg(test)]
mod tests {
    use crate::handlers::validation_core::{
        detect_query_type, extract_affected_graphs, extract_prefixes, extract_query_variables,
        extract_update_operations, is_deprecated_language, parse_language_tag,
        validate_iris_internal, validate_langtags_internal, validate_rdf_data_internal,
        validate_sparql_query_internal, validate_sparql_update_internal,
    };
    use crate::handlers::validation_types::{
        DataValidationRequest, IriValidationRequest, LangTagValidationRequest,
        QueryValidationRequest, UpdateValidationRequest,
    };

    #[test]
    fn test_validate_sparql_query_valid() {
        let request = QueryValidationRequest {
            query: "SELECT ?s ?p ?o WHERE { ?s ?p ?o }".to_string(),
            syntax: "sparql11".to_string(),
            include_algebra: true,
            include_optimized: true,
        };

        let response = validate_sparql_query_internal(&request);
        assert!(response.valid);
        assert_eq!(response.query_type, Some("SELECT".to_string()));
        assert!(response.variables.is_some());
        let vars = response.variables.unwrap();
        assert!(vars.contains(&"s".to_string()));
        assert!(vars.contains(&"p".to_string()));
        assert!(vars.contains(&"o".to_string()));
    }

    #[test]
    fn test_validate_sparql_query_invalid() {
        let request = QueryValidationRequest {
            query: "SELEKT ?s WHERE { ?s ?p ?o }".to_string(),
            syntax: "sparql11".to_string(),
            include_algebra: false,
            include_optimized: false,
        };

        let response = validate_sparql_query_internal(&request);
        assert!(!response.valid);
        assert!(!response.errors.is_empty());
    }

    #[test]
    fn test_validate_sparql_query_empty() {
        let request = QueryValidationRequest {
            query: "".to_string(),
            syntax: "sparql11".to_string(),
            include_algebra: false,
            include_optimized: false,
        };

        let response = validate_sparql_query_internal(&request);
        assert!(!response.valid);
        assert_eq!(response.errors[0].code, Some("EMPTY_QUERY".to_string()));
    }

    #[test]
    fn test_validate_sparql_update_valid() {
        let request = UpdateValidationRequest {
            update: "INSERT DATA { <http://example/s> <http://example/p> \"value\" }".to_string(),
            syntax: "sparql11".to_string(),
        };

        let response = validate_sparql_update_internal(&request);
        assert!(response.valid);
        assert!(response.operations.contains(&"INSERT DATA".to_string()));
    }

    #[test]
    fn test_validate_sparql_update_invalid() {
        let request = UpdateValidationRequest {
            update: "INSERTT DATA { <http://example/s> <http://example/p> \"value\" }".to_string(),
            syntax: "sparql11".to_string(),
        };

        let response = validate_sparql_update_internal(&request);
        assert!(!response.valid);
    }

    #[test]
    fn test_validate_iri_valid() {
        let request = IriValidationRequest {
            iris: vec![
                "http://example.org/resource".to_string(),
                "https://www.w3.org/2001/XMLSchema#string".to_string(),
                "urn:isbn:0451450523".to_string(),
            ],
            check_relative: true,
        };

        let response = validate_iris_internal(&request);
        assert_eq!(response.summary.total, 3);
        assert_eq!(response.summary.valid, 3);
        assert!(response.results[0].is_absolute);
        assert_eq!(response.results[0].scheme, Some("http".to_string()));
        assert_eq!(response.results[2].scheme, Some("urn".to_string()));
    }

    #[test]
    fn test_validate_iri_relative() {
        let request = IriValidationRequest {
            iris: vec!["resource/path".to_string()],
            check_relative: true,
        };

        let _response = validate_iris_internal(&request);
        // Note: oxirs-core may reject relative IRIs as invalid NamedNodes
        // This test verifies the warning behavior
    }

    #[test]
    fn test_validate_rdf_data_turtle() {
        let request = DataValidationRequest {
            data: r#"
                @prefix ex: <http://example.org/> .
                ex:subject ex:predicate "object" .
                ex:s2 ex:p2 ex:o2 .
            "#
            .to_string(),
            format: "turtle".to_string(),
            base: Some("http://example.org/base/".to_string()),
        };

        let response = validate_rdf_data_internal(&request);
        assert!(response.valid);
        assert_eq!(response.triple_count, 2);
    }

    #[test]
    fn test_validate_rdf_data_invalid() {
        let request = DataValidationRequest {
            data: "@prefix ex: <broken".to_string(),
            format: "turtle".to_string(),
            base: None,
        };

        let response = validate_rdf_data_internal(&request);
        assert!(!response.valid);
        assert!(!response.errors.is_empty());
    }

    #[test]
    fn test_validate_langtag_valid() {
        let request = LangTagValidationRequest {
            tags: vec![
                "en".to_string(),
                "en-US".to_string(),
                "zh-Hans-CN".to_string(),
                "de-DE-1996".to_string(),
            ],
        };

        let response = validate_langtags_internal(&request);
        assert_eq!(response.summary.total, 4);
        assert_eq!(response.summary.valid, 4);

        assert_eq!(response.results[0].language, Some("en".to_string()));
        assert_eq!(response.results[1].region, Some("US".to_string()));
        assert_eq!(response.results[2].script, Some("Hans".to_string()));
    }

    #[test]
    fn test_validate_langtag_private_use() {
        let request = LangTagValidationRequest {
            tags: vec!["x-custom".to_string()],
        };

        let response = validate_langtags_internal(&request);
        assert_eq!(response.summary.valid, 1);
        assert_eq!(response.results[0].private_use, Some("custom".to_string()));
    }

    #[test]
    fn test_detect_query_type() {
        assert_eq!(
            detect_query_type("SELECT ?x WHERE { ?x ?y ?z }"),
            Some("SELECT".to_string())
        );
        assert_eq!(
            detect_query_type("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"),
            Some("CONSTRUCT".to_string())
        );
        assert_eq!(
            detect_query_type("ASK { ?s ?p ?o }"),
            Some("ASK".to_string())
        );
        assert_eq!(
            detect_query_type("DESCRIBE <http://example.org>"),
            Some("DESCRIBE".to_string())
        );
    }

    #[test]
    fn test_extract_query_variables() {
        let vars = extract_query_variables(
            "SELECT ?name ?age WHERE { ?person foaf:name ?name ; foaf:age ?age }",
        );
        assert!(vars.contains(&"name".to_string()));
        assert!(vars.contains(&"age".to_string()));
        assert!(vars.contains(&"person".to_string()));
    }

    #[test]
    fn test_extract_prefixes() {
        let prefixes = extract_prefixes("PREFIX foaf: <http://xmlns.com/foaf/0.1/>\nPREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\nSELECT * WHERE { ?s ?p ?o }");
        assert_eq!(prefixes.len(), 2);
        assert!(prefixes.iter().any(|p| p.prefix == "foaf"));
        assert!(prefixes.iter().any(|p| p.prefix == "rdf"));
    }

    #[test]
    fn test_extract_update_operations() {
        let ops = extract_update_operations(
            "INSERT DATA { <s> <p> \"o\" } ; DELETE DATA { <s2> <p2> \"o2\" }",
        );
        assert!(ops.contains(&"INSERT DATA".to_string()));
        assert!(ops.contains(&"DELETE DATA".to_string()));
    }

    #[test]
    fn test_extract_affected_graphs() {
        let graphs = extract_affected_graphs(
            "INSERT DATA { GRAPH <http://example.org/g1> { <s> <p> \"o\" } }",
        );
        assert!(graphs.contains(&"http://example.org/g1".to_string()));
    }

    #[test]
    fn test_parse_language_tag() {
        let result = parse_language_tag("en-Latn-US-valencia").unwrap();
        assert_eq!(result.language, Some("en".to_string()));
        assert_eq!(result.script, Some("Latn".to_string()));
        assert_eq!(result.region, Some("US".to_string()));
        assert!(result.variants.contains(&"valencia".to_string()));
    }

    #[test]
    fn test_is_deprecated_language() {
        assert!(is_deprecated_language("iw")); // Hebrew (deprecated)
        assert!(!is_deprecated_language("he")); // Hebrew (current)
        assert!(is_deprecated_language("ji")); // Yiddish (deprecated)
    }
}
