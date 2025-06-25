//! Compatibility tests to ensure OxiRS implementations match expected RDF behavior

use oxirs_core::model::{
    BlankNode, Literal, NamedNode, Object, Predicate, Subject, Triple, Variable,
};

#[cfg(test)]
mod iri_tests {
    use super::*;

    #[test]
    fn test_iri_validation() {
        // Valid IRIs
        assert!(NamedNode::new("http://example.org/").is_ok());
        assert!(NamedNode::new("https://example.org/path").is_ok());
        assert!(NamedNode::new("http://example.org/path?query=1").is_ok());
        assert!(NamedNode::new("http://example.org/path#fragment").is_ok());
        assert!(NamedNode::new("urn:uuid:12345678-1234-5678-1234-567812345678").is_ok());
        assert!(NamedNode::new("file:///path/to/file").is_ok());
        assert!(NamedNode::new("ftp://ftp.example.org/file.txt").is_ok());

        // Invalid IRIs
        assert!(NamedNode::new("").is_err());
        assert!(NamedNode::new("not a valid iri").is_err());
        assert!(NamedNode::new("http://example.org/<invalid>").is_err());
        assert!(NamedNode::new("http://example.org/{invalid}").is_err());
    }

    #[test]
    fn test_iri_normalization() {
        // Scheme normalization
        let iri1 = NamedNode::new_normalized("HTTP://EXAMPLE.ORG/").unwrap();
        assert_eq!(iri1.as_str(), "http://example.org/");

        // Percent encoding normalization
        let iri2 = NamedNode::new_normalized("http://example.org/%2f").unwrap();
        assert_eq!(iri2.as_str(), "http://example.org/%2F");

        // Path normalization
        let iri3 = NamedNode::new_normalized("http://example.org/./path").unwrap();
        assert_eq!(iri3.as_str(), "http://example.org/path");
    }

    #[test]
    fn test_iri_display() {
        let iri = NamedNode::new("http://example.org/resource").unwrap();
        assert_eq!(format!("{}", iri), "<http://example.org/resource>");
    }
}

#[cfg(test)]
mod blank_node_tests {
    use super::*;

    #[test]
    fn test_blank_node_validation() {
        // Valid blank nodes
        assert!(BlankNode::new("node1").is_ok());
        assert!(BlankNode::new("_:node1").is_ok());
        assert!(BlankNode::new("node_1").is_ok());
        assert!(BlankNode::new("node-1.2").is_ok());

        // Invalid blank nodes
        assert!(BlankNode::new("").is_err());
        assert!(BlankNode::new("_:").is_err());
        assert!(BlankNode::new("123node").is_err()); // Can't start with digit
        assert!(BlankNode::new("node with space").is_err());
        assert!(BlankNode::new("node@invalid").is_err());
    }

    #[test]
    fn test_blank_node_unique_generation() {
        let bn1 = BlankNode::new_unique();
        let bn2 = BlankNode::new_unique();
        assert_ne!(bn1.id(), bn2.id());
        // Unique blank nodes are created with hex IDs that start with a-f for RDF/XML compatibility
        assert!(matches!(bn1.id().as_bytes().first(), Some(b'a'..=b'f')));
        assert!(matches!(bn2.id().as_bytes().first(), Some(b'a'..=b'f')));
    }

    #[test]
    fn test_blank_node_display() {
        let bn = BlankNode::new("test").unwrap();
        assert_eq!(format!("{}", bn), "_:test");
    }
}

#[cfg(test)]
mod literal_tests {
    use super::*;

    #[test]
    fn test_simple_literal() {
        let lit = Literal::new("hello world");
        assert_eq!(lit.value(), "hello world");
        assert_eq!(
            lit.datatype().as_str(),
            "http://www.w3.org/2001/XMLSchema#string"
        );
        assert!(lit.language().is_none());
        assert_eq!(format!("{}", lit), "\"hello world\"");
    }

    #[test]
    fn test_typed_literal() {
        let datatype = NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap();
        let lit = Literal::new_typed("42", datatype.clone());
        assert_eq!(lit.value(), "42");
        assert_eq!(
            lit.datatype().as_str(),
            "http://www.w3.org/2001/XMLSchema#integer"
        );
        assert_eq!(
            format!("{}", lit),
            "\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>"
        );
    }

    #[test]
    fn test_language_literal() {
        let lit = Literal::new_lang("hello", "en").unwrap();
        assert_eq!(lit.value(), "hello");
        assert_eq!(lit.language().unwrap(), "en");
        assert_eq!(format!("{}", lit), "\"hello\"@en");

        // Complex language tags
        let lit2 = Literal::new_lang("hello", "en-US").unwrap();
        assert_eq!(lit2.language().unwrap(), "en-US");

        // Invalid language tags
        assert!(Literal::new_lang("hello", "123").is_err());
        assert!(Literal::new_lang("hello", "").is_err());
    }

    #[test]
    fn test_xsd_validation() {
        // Valid XSD types
        let xsd_bool = NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap();
        let xsd_int = NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap();
        let xsd_decimal = NamedNode::new("http://www.w3.org/2001/XMLSchema#decimal").unwrap();
        let xsd_date = NamedNode::new("http://www.w3.org/2001/XMLSchema#date").unwrap();

        assert!(Literal::new_typed_validated("true", xsd_bool.clone()).is_ok());
        assert!(Literal::new_typed_validated("false", xsd_bool.clone()).is_ok());
        assert!(Literal::new_typed_validated("42", xsd_int.clone()).is_ok());
        assert!(Literal::new_typed_validated("-42", xsd_int.clone()).is_ok());
        assert!(Literal::new_typed_validated("3.14", xsd_decimal.clone()).is_ok());
        assert!(Literal::new_typed_validated("2024-01-01", xsd_date.clone()).is_ok());

        // Invalid XSD types
        assert!(Literal::new_typed_validated("not a boolean", xsd_bool).is_err());
        assert!(Literal::new_typed_validated("not a number", xsd_int).is_err());
        assert!(Literal::new_typed_validated("2024-13-01", xsd_date).is_err());
    }
}

#[cfg(test)]
mod variable_tests {
    use super::*;

    #[test]
    fn test_variable_validation() {
        // Valid variables
        assert!(Variable::new("x").is_ok());
        assert!(Variable::new("?x").is_ok());
        assert!(Variable::new("$x").is_ok());
        assert!(Variable::new("var123").is_ok());
        assert!(Variable::new("_var").is_ok());

        // Invalid variables
        assert!(Variable::new("").is_err());
        assert!(Variable::new("?").is_err());
        assert!(Variable::new("$").is_err());
        assert!(Variable::new("123var").is_err());
        assert!(Variable::new("var-name").is_err());

        // Reserved keywords
        assert!(Variable::new("select").is_err());
        assert!(Variable::new("where").is_err());
        assert!(Variable::new("SELECT").is_err()); // Case insensitive
    }

    #[test]
    fn test_variable_display() {
        let var = Variable::new("x").unwrap();
        assert_eq!(format!("{}", var), "?x");
        assert_eq!(var.name(), "x");
        assert_eq!(var.with_prefix(), "?x");
    }
}

#[cfg(test)]
mod triple_tests {
    use super::*;

    #[test]
    fn test_triple_creation() {
        let s = NamedNode::new("http://example.org/subject").unwrap();
        let p = NamedNode::new("http://example.org/predicate").unwrap();
        let o = Literal::new("object");

        let triple = Triple::new(s.clone(), p.clone(), o.clone());
        assert!(triple.is_ground());
        assert!(!triple.has_variables());

        match triple.subject() {
            Subject::NamedNode(n) => assert_eq!(n.as_str(), "http://example.org/subject"),
            _ => panic!("Expected NamedNode subject"),
        }
    }

    #[test]
    fn test_triple_with_variables() {
        let s = Variable::new("s").unwrap();
        let p = NamedNode::new("http://example.org/predicate").unwrap();
        let o = Variable::new("o").unwrap();

        let triple = Triple::new(s, p, o);
        assert!(!triple.is_ground());
        assert!(triple.has_variables());
    }

    #[test]
    fn test_triple_pattern_matching() {
        let s = NamedNode::new("http://example.org/s").unwrap();
        let p = NamedNode::new("http://example.org/p").unwrap();
        let o = Literal::new("o");

        let triple = Triple::new(s.clone(), p.clone(), o.clone());

        // Exact match
        assert!(triple.matches_pattern(
            Some(&Subject::NamedNode(s.clone())),
            Some(&Predicate::NamedNode(p.clone())),
            Some(&Object::Literal(o.clone()))
        ));

        // Wildcard matches
        assert!(triple.matches_pattern(None, None, None));
        assert!(triple.matches_pattern(Some(&Subject::NamedNode(s.clone())), None, None));

        // Non-match
        let other_s = NamedNode::new("http://example.org/other").unwrap();
        assert!(!triple.matches_pattern(Some(&Subject::NamedNode(other_s)), None, None));
    }

    #[test]
    fn test_triple_display() {
        let s = NamedNode::new("http://example.org/s").unwrap();
        let p = NamedNode::new("http://example.org/p").unwrap();
        let o = Literal::new("o");

        let triple = Triple::new(s, p, o);
        let display_str = format!("{}", triple);

        assert!(display_str.contains("<http://example.org/s>"));
        assert!(display_str.contains("<http://example.org/p>"));
        assert!(display_str.contains("\"o\""));
        assert!(display_str.ends_with(" ."));
    }
}

#[cfg(test)]
mod ordering_tests {
    use super::*;

    #[test]
    fn test_term_ordering() {
        // Create different term types
        let nn1 = NamedNode::new("http://example.org/a").unwrap();
        let nn2 = NamedNode::new("http://example.org/b").unwrap();
        let bn1 = BlankNode::new("b1").unwrap();
        let bn2 = BlankNode::new("b2").unwrap();
        let lit1 = Literal::new("a");
        let lit2 = Literal::new("b");

        // Named nodes should order alphabetically
        assert!(nn1 < nn2);

        // Blank nodes should order alphabetically
        assert!(bn1 < bn2);

        // Literals should order alphabetically
        assert!(lit1 < lit2);
    }

    #[test]
    fn test_triple_ordering() {
        let s1 = NamedNode::new("http://example.org/a").unwrap();
        let s2 = NamedNode::new("http://example.org/b").unwrap();
        let p = NamedNode::new("http://example.org/p").unwrap();
        let o = Literal::new("o");

        let triple1 = Triple::new(s1, p.clone(), o.clone());
        let triple2 = Triple::new(s2, p, o);

        assert!(triple1 < triple2);

        // Test that triples can be sorted
        let mut triples = vec![triple2.clone(), triple1.clone()];
        triples.sort();
        assert_eq!(triples, vec![triple1, triple2]);
    }
}

#[cfg(test)]
mod serialization_tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_named_node_serialization() {
        let nn = NamedNode::new("http://example.org/test").unwrap();
        let json = serde_json::to_string(&nn).unwrap();
        let deserialized: NamedNode = serde_json::from_str(&json).unwrap();
        assert_eq!(nn, deserialized);
    }

    #[test]
    fn test_blank_node_serialization() {
        let bn = BlankNode::new("test").unwrap();
        let json = serde_json::to_string(&bn).unwrap();
        let deserialized: BlankNode = serde_json::from_str(&json).unwrap();
        assert_eq!(bn, deserialized);
    }

    #[test]
    fn test_literal_serialization() {
        let lit = Literal::new_lang("hello", "en").unwrap();
        let json = serde_json::to_string(&lit).unwrap();
        let deserialized: Literal = serde_json::from_str(&json).unwrap();
        assert_eq!(lit, deserialized);
    }

    #[test]
    fn test_triple_serialization() {
        let s = NamedNode::new("http://example.org/s").unwrap();
        let p = NamedNode::new("http://example.org/p").unwrap();
        let o = Literal::new("o");

        let triple = Triple::new(s, p, o);
        let json = serde_json::to_string(&triple).unwrap();
        let deserialized: Triple = serde_json::from_str(&json).unwrap();
        assert_eq!(triple, deserialized);
    }
}
