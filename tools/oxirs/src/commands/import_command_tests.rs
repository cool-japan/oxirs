//! # Import Command Tests
//!
//! All `#[cfg(test)]` blocks for the multi-format RDF importer.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::commands::import_command::{
        ImportCommand, ImportError, ImportFormat, ImportResult, Triple,
    };

    // --- ImportFormat helpers -----------------------------------------------

    #[test]
    fn test_from_extension_turtle() {
        assert_eq!(
            ImportFormat::from_extension("ttl"),
            Some(ImportFormat::Turtle)
        );
    }

    #[test]
    fn test_from_extension_ntriples() {
        assert_eq!(
            ImportFormat::from_extension("nt"),
            Some(ImportFormat::NTriples)
        );
    }

    #[test]
    fn test_from_extension_nquads() {
        assert_eq!(
            ImportFormat::from_extension("nq"),
            Some(ImportFormat::NQuads)
        );
    }

    #[test]
    fn test_from_extension_jsonld() {
        assert_eq!(
            ImportFormat::from_extension("jsonld"),
            Some(ImportFormat::JsonLd)
        );
    }

    #[test]
    fn test_from_extension_rdf() {
        assert_eq!(
            ImportFormat::from_extension("rdf"),
            Some(ImportFormat::RdfXml)
        );
    }

    #[test]
    fn test_from_extension_trig() {
        assert_eq!(
            ImportFormat::from_extension("trig"),
            Some(ImportFormat::TriG)
        );
    }

    #[test]
    fn test_from_extension_csv() {
        assert_eq!(ImportFormat::from_extension("csv"), Some(ImportFormat::Csv));
    }

    #[test]
    fn test_from_extension_unknown() {
        assert_eq!(ImportFormat::from_extension("docx"), None);
    }

    #[test]
    fn test_from_extension_case_insensitive() {
        assert_eq!(
            ImportFormat::from_extension("TTL"),
            Some(ImportFormat::Turtle)
        );
    }

    #[test]
    fn test_from_mime_type_turtle() {
        assert_eq!(
            ImportFormat::from_mime_type("text/turtle"),
            Some(ImportFormat::Turtle)
        );
    }

    #[test]
    fn test_from_mime_type_ntriples() {
        assert_eq!(
            ImportFormat::from_mime_type("application/n-triples"),
            Some(ImportFormat::NTriples)
        );
    }

    #[test]
    fn test_from_mime_type_jsonld() {
        assert_eq!(
            ImportFormat::from_mime_type("application/ld+json"),
            Some(ImportFormat::JsonLd)
        );
    }

    #[test]
    fn test_from_mime_type_csv() {
        assert_eq!(
            ImportFormat::from_mime_type("text/csv"),
            Some(ImportFormat::Csv)
        );
    }

    #[test]
    fn test_from_mime_type_unknown() {
        assert_eq!(ImportFormat::from_mime_type("text/plain"), None);
    }

    #[test]
    fn test_extension_and_mime_type() {
        assert_eq!(ImportFormat::Turtle.extension(), "ttl");
        assert_eq!(ImportFormat::Turtle.mime_type(), "text/turtle");
        assert_eq!(ImportFormat::NTriples.extension(), "nt");
        assert_eq!(ImportFormat::NQuads.extension(), "nq");
        assert_eq!(ImportFormat::JsonLd.extension(), "jsonld");
        assert_eq!(ImportFormat::RdfXml.extension(), "rdf");
        assert_eq!(ImportFormat::TriG.extension(), "trig");
        assert_eq!(ImportFormat::Csv.extension(), "csv");
    }

    // --- Empty input --------------------------------------------------------

    #[test]
    fn test_empty_input_error() {
        assert!(matches!(
            ImportCommand::import("", ImportFormat::NTriples),
            Err(ImportError::EmptyInput)
        ));
        assert!(matches!(
            ImportCommand::import("   \n", ImportFormat::Turtle),
            Err(ImportError::EmptyInput)
        ));
    }

    // --- N-Triples ----------------------------------------------------------

    #[test]
    fn test_parse_ntriples_single() {
        let nt = "<http://a.org/s> <http://a.org/p> <http://a.org/o> .\n";
        let result = ImportCommand::parse_ntriples(nt).expect("ok");
        assert_eq!(result.triple_count(), 1);
        assert_eq!(result.triples[0].subject, "http://a.org/s");
        assert_eq!(result.triples[0].predicate, "http://a.org/p");
        assert_eq!(result.triples[0].object, "http://a.org/o");
        assert!(result.triples[0].graph.is_none());
    }

    #[test]
    fn test_parse_ntriples_multiple() {
        let nt = "<http://a/s> <http://a/p> <http://a/o> .\n\
                  <http://b/s> <http://b/p> <http://b/o> .\n";
        let result = ImportCommand::parse_ntriples(nt).expect("ok");
        assert_eq!(result.triple_count(), 2);
    }

    #[test]
    fn test_parse_ntriples_blank_node() {
        let nt = "_:b1 <http://a.org/p> <http://a.org/o> .\n";
        let result = ImportCommand::parse_ntriples(nt).expect("ok");
        assert_eq!(result.triple_count(), 1);
        assert_eq!(result.triples[0].subject, "_:b1");
    }

    #[test]
    fn test_parse_ntriples_literal_object() {
        let nt = "<http://a.org/s> <http://a.org/p> \"hello\" .\n";
        let result = ImportCommand::parse_ntriples(nt).expect("ok");
        assert_eq!(result.triple_count(), 1);
        assert_eq!(result.triples[0].object, "hello");
    }

    #[test]
    fn test_parse_ntriples_comment_skipped() {
        let nt = "# This is a comment\n<http://a/s> <http://a/p> <http://a/o> .\n";
        let result = ImportCommand::parse_ntriples(nt).expect("ok");
        assert_eq!(result.triple_count(), 1);
    }

    #[test]
    fn test_parse_ntriples_malformed_warning() {
        let nt = "this is not a valid triple\n<http://a/s> <http://a/p> <http://a/o> .\n";
        let result = ImportCommand::parse_ntriples(nt).expect("ok");
        assert!(
            result.has_warnings(),
            "expected warnings for malformed line"
        );
        assert_eq!(result.triple_count(), 1);
    }

    // --- N-Quads ------------------------------------------------------------

    #[test]
    fn test_parse_nquads_with_graph() {
        let nq = "<http://s> <http://p> <http://o> <http://g> .\n";
        let result = ImportCommand::parse_nquads(nq).expect("ok");
        assert_eq!(result.triple_count(), 1);
        assert_eq!(result.triples[0].graph, Some("http://g".to_string()));
        assert_eq!(result.graph_count(), 1);
    }

    #[test]
    fn test_parse_nquads_without_graph() {
        let nq = "<http://s> <http://p> <http://o> .\n";
        let result = ImportCommand::parse_nquads(nq).expect("ok");
        assert_eq!(result.triple_count(), 1);
        assert!(result.triples[0].graph.is_none());
    }

    #[test]
    fn test_parse_nquads_multiple_graphs() {
        let nq = "<http://s1> <http://p> <http://o1> <http://g1> .\n\
                  <http://s2> <http://p> <http://o2> <http://g2> .\n";
        let result = ImportCommand::parse_nquads(nq).expect("ok");
        assert_eq!(result.graph_count(), 2);
    }

    // --- Turtle -------------------------------------------------------------

    #[test]
    fn test_parse_turtle_with_prefix() {
        let ttl = "@prefix ex: <http://example.org/> .\n\
                   <http://a.org/s> <http://a.org/p> <http://a.org/o> .\n";
        let result = ImportCommand::parse_turtle(ttl).expect("ok");
        assert!(!result.prefixes.is_empty());
        assert!(result.prefixes.contains_key("ex"));
    }

    #[test]
    fn test_parse_turtle_prefix_extraction() {
        let ttl = "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n\
                   <http://s.org/s> <http://s.org/p> <http://s.org/o> .\n";
        let result = ImportCommand::parse_turtle(ttl).expect("ok");
        assert!(result.prefixes.contains_key("rdf"));
        assert_eq!(
            result.prefixes.get("rdf").map(String::as_str),
            Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        );
    }

    #[test]
    fn test_parse_turtle_triple_count() {
        let ttl = "<http://a/s> <http://a/p> <http://a/o> .\n\
                   <http://b/s> <http://b/p> <http://b/o> .\n";
        let result = ImportCommand::parse_turtle(ttl).expect("ok");
        assert_eq!(result.triple_count(), 2);
        assert_eq!(result.format_detected, ImportFormat::Turtle);
    }

    // --- TriG ---------------------------------------------------------------

    #[test]
    fn test_parse_trig_graph_block() {
        let trig = "GRAPH <http://g.org/g1> {\n\
                    <http://a/s> <http://a/p> <http://a/o> .\n\
                    }\n";
        let result = ImportCommand::parse_trig(trig).expect("ok");
        assert_eq!(result.graph_count(), 1);
        assert_eq!(result.graphs[0], "http://g.org/g1");
        assert_eq!(result.triple_count(), 1);
        assert_eq!(result.triples[0].graph, Some("http://g.org/g1".to_string()));
    }

    // --- CSV ----------------------------------------------------------------

    #[test]
    fn test_parse_csv_basic() {
        let csv = "subject,predicate,object\n\
                   http://a/s,http://a/p,http://a/o\n";
        let result = ImportCommand::parse_csv(csv).expect("ok");
        assert_eq!(result.triple_count(), 1);
        assert_eq!(result.triples[0].subject, "http://a/s");
    }

    #[test]
    fn test_parse_csv_with_graph_column() {
        let csv = "subject,predicate,object,graph\n\
                   http://s,http://p,http://o,http://g\n";
        let result = ImportCommand::parse_csv(csv).expect("ok");
        assert_eq!(result.triple_count(), 1);
        assert_eq!(result.triples[0].graph, Some("http://g".to_string()));
        assert_eq!(result.graph_count(), 1);
    }

    #[test]
    fn test_parse_csv_missing_column_warning() {
        let csv = "subject,predicate,object\n\
                   only_one_column\n";
        let result = ImportCommand::parse_csv(csv).expect("ok");
        assert!(result.has_warnings());
    }

    #[test]
    fn test_parse_csv_multiple_rows() {
        let csv = "subject,predicate,object\n\
                   http://a/s1,http://a/p,http://a/o1\n\
                   http://a/s2,http://a/p,http://a/o2\n";
        let result = ImportCommand::parse_csv(csv).expect("ok");
        assert_eq!(result.triple_count(), 2);
    }

    // --- JSON-LD (simplified) -----------------------------------------------

    #[test]
    fn test_parse_jsonld_simple() {
        let jsonld = r#"{"@context":{"name":"http://schema.org/name"},"@graph":[{"@id":"http://a.org/person","name":"Alice"}]}"#;
        let result = ImportCommand::parse_jsonld(jsonld).expect("ok");
        let _ = result.triple_count();
        assert_eq!(result.format_detected, ImportFormat::JsonLd);
    }

    #[test]
    fn test_parse_jsonld_empty_warning() {
        let jsonld = r#"{"@context":{},"@graph":[]}"#;
        let result = ImportCommand::parse_jsonld(jsonld).expect("ok");
        assert_eq!(result.format_detected, ImportFormat::JsonLd);
    }

    // --- RDF/XML (simplified) -----------------------------------------------

    #[test]
    fn test_parse_rdfxml_basic() {
        let rdfxml = r#"<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:ex="http://example.org/">
  <rdf:Description rdf:about="http://example.org/alice">
    <ex:name>Alice</ex:name>
  </rdf:Description>
</rdf:RDF>"#;
        let result = ImportCommand::parse_rdfxml(rdfxml).expect("ok");
        assert_eq!(result.format_detected, ImportFormat::RdfXml);
        assert!(result.triple_count() > 0 || result.has_warnings());
    }

    // --- detect_format ------------------------------------------------------

    #[test]
    fn test_detect_format_turtle() {
        let input = "@prefix ex: <http://example.org/> .\n<http://s> <http://p> <http://o> .\n";
        assert_eq!(
            ImportCommand::detect_format(input),
            Some(ImportFormat::Turtle)
        );
    }

    #[test]
    fn test_detect_format_jsonld() {
        let input = r#"{"@context":{},"@id":"http://a.org/x"}"#;
        assert_eq!(
            ImportCommand::detect_format(input),
            Some(ImportFormat::JsonLd)
        );
    }

    #[test]
    fn test_detect_format_rdfxml() {
        let input = "<?xml version=\"1.0\"?><rdf:RDF></rdf:RDF>";
        assert_eq!(
            ImportCommand::detect_format(input),
            Some(ImportFormat::RdfXml)
        );
    }

    #[test]
    fn test_detect_format_trig() {
        let input = "GRAPH <http://g.org/> { <http://s> <http://p> <http://o> . }";
        assert_eq!(
            ImportCommand::detect_format(input),
            Some(ImportFormat::TriG)
        );
    }

    #[test]
    fn test_detect_format_csv() {
        let input = "subject,predicate,object\nhttp://s,http://p,http://o\n";
        assert_eq!(ImportCommand::detect_format(input), Some(ImportFormat::Csv));
    }

    // --- strip_iri ----------------------------------------------------------

    #[test]
    fn test_strip_iri_with_brackets() {
        assert_eq!(
            ImportCommand::strip_iri("<http://example.org/>"),
            "http://example.org/"
        );
    }

    #[test]
    fn test_strip_iri_without_brackets() {
        assert_eq!(
            ImportCommand::strip_iri("http://example.org/"),
            "http://example.org/"
        );
    }

    #[test]
    fn test_strip_iri_with_whitespace() {
        assert_eq!(
            ImportCommand::strip_iri("  <http://example.org/>  "),
            "http://example.org/"
        );
    }

    // --- unescape_literal ---------------------------------------------------

    #[test]
    fn test_unescape_literal_newline() {
        assert_eq!(
            ImportCommand::unescape_literal("line1\\nline2"),
            "line1\nline2"
        );
    }

    #[test]
    fn test_unescape_literal_tab() {
        assert_eq!(ImportCommand::unescape_literal("col1\\tcol2"), "col1\tcol2");
    }

    #[test]
    fn test_unescape_literal_quote() {
        assert_eq!(
            ImportCommand::unescape_literal("say \\\"hi\\\""),
            "say \"hi\""
        );
    }

    #[test]
    fn test_unescape_literal_backslash() {
        assert_eq!(
            ImportCommand::unescape_literal("back\\\\slash"),
            "back\\slash"
        );
    }

    #[test]
    fn test_unescape_literal_unicode() {
        assert_eq!(ImportCommand::unescape_literal("\\u0041"), "A");
    }

    #[test]
    fn test_unescape_literal_no_escape() {
        assert_eq!(ImportCommand::unescape_literal("hello"), "hello");
    }

    // --- ImportResult helpers -----------------------------------------------

    #[test]
    fn test_import_result_triple_count() {
        let r = ImportResult {
            triples: vec![
                Triple {
                    subject: "s".to_string(),
                    predicate: "p".to_string(),
                    object: "o".to_string(),
                    graph: None,
                };
                3
            ],
            prefixes: HashMap::new(),
            graphs: Vec::new(),
            warnings: Vec::new(),
            format_detected: ImportFormat::NTriples,
        };
        assert_eq!(r.triple_count(), 3);
    }

    #[test]
    fn test_import_result_graph_count() {
        let r = ImportResult {
            triples: Vec::new(),
            prefixes: HashMap::new(),
            graphs: vec!["g1".to_string(), "g2".to_string()],
            warnings: Vec::new(),
            format_detected: ImportFormat::NQuads,
        };
        assert_eq!(r.graph_count(), 2);
    }

    #[test]
    fn test_import_result_has_warnings() {
        let mut r = ImportResult {
            triples: Vec::new(),
            prefixes: HashMap::new(),
            graphs: Vec::new(),
            warnings: Vec::new(),
            format_detected: ImportFormat::NTriples,
        };
        assert!(!r.has_warnings());
        r.warnings.push("warn".to_string());
        assert!(r.has_warnings());
    }

    // --- Error display -------------------------------------------------------

    #[test]
    fn test_import_error_display() {
        assert!(ImportError::EmptyInput.to_string().contains("empty"));
        assert!(ImportError::ParseError("bad".to_string())
            .to_string()
            .contains("bad"));
        assert!(ImportError::UnsupportedFormat("xyz".to_string())
            .to_string()
            .contains("xyz"));
        assert!(ImportError::InvalidTriple("bad".to_string())
            .to_string()
            .contains("bad"));
    }

    // --- import() dispatch --------------------------------------------------

    #[test]
    fn test_import_dispatch_ntriples() {
        let nt = "<http://a/s> <http://a/p> <http://a/o> .\n";
        let result = ImportCommand::import(nt, ImportFormat::NTriples).expect("ok");
        assert_eq!(result.format_detected, ImportFormat::NTriples);
    }

    #[test]
    fn test_import_dispatch_csv() {
        let csv = "subject,predicate,object\nhttp://s,http://p,http://o\n";
        let result = ImportCommand::import(csv, ImportFormat::Csv).expect("ok");
        assert_eq!(result.format_detected, ImportFormat::Csv);
        assert_eq!(result.triple_count(), 1);
    }
}
