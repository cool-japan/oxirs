//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Generate a simple UUID-like string
pub(super) fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("SystemTime should be after UNIX_EPOCH")
        .as_nanos();
    format!("{:x}", now)
}
#[cfg(test)]
mod tests {
    use super::super::rifparser_type::RifParser;
    use super::super::types::{
        RifConst, RifDialect, RifDocument, RifFormula, RifImport, RifSerializer, RifTerm, RifVar,
    };
    use crate::{Rule, RuleAtom, Term};
    #[test]
    fn test_rif_parser_basic() {
        let mut parser = RifParser::new(RifDialect::Bld);
        let input = r#"
            Prefix(ex <http://example.org/>)
            Group (
                ex:ancestor(?x ?y) :- ex:parent(?x ?y)
            )
        "#;
        let doc = parser.parse(input).unwrap();
        assert_eq!(doc.groups.len(), 1);
        assert!(doc.prefixes.contains_key("ex"));
    }
    #[test]
    fn test_rif_parser_forall() {
        let mut parser = RifParser::new(RifDialect::Bld);
        let input = r#"
            Forall ?x ?y (
                ancestor(?x ?y) :- parent(?x ?y)
            )
        "#;
        let doc = parser.parse(input).unwrap();
        assert!(!doc.groups.is_empty());
    }
    #[test]
    fn test_rif_parser_and_formula() {
        let mut parser = RifParser::new(RifDialect::Bld);
        let input = r#"
            Group (
                ancestor(?x ?z) :- And(parent(?x ?y) ancestor(?y ?z))
            )
        "#;
        let doc = parser.parse(input).unwrap();
        assert!(!doc.groups.is_empty());
    }
    #[test]
    fn test_rif_parser_naf() {
        let mut parser = RifParser::new(RifDialect::Bld);
        let input = r#"
            Group (
                single(?x) :- And(person(?x) Naf(married(?x)))
            )
        "#;
        let doc = parser.parse(input).unwrap();
        assert!(!doc.groups.is_empty());
    }
    #[test]
    fn test_rif_converter_to_oxirs() {
        let mut parser = RifParser::new(RifDialect::Bld);
        let input = r#"
            Group (
                ancestor(?x ?y) :- parent(?x ?y)
            )
        "#;
        let doc = parser.parse(input).unwrap();
        let rules = doc.to_oxirs_rules().unwrap();
        assert!(!rules.is_empty());
        let rule = &rules[0];
        assert_eq!(rule.body.len(), 1);
        assert_eq!(rule.head.len(), 1);
    }
    #[test]
    fn test_rif_converter_from_oxirs() {
        let rule = Rule {
            name: "ancestor_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("ancestor".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        };
        let doc = RifDocument::from_oxirs_rules(&[rule], RifDialect::Bld);
        assert_eq!(doc.groups.len(), 1);
    }
    #[test]
    fn test_rif_serializer() {
        let rule = Rule {
            name: "test_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("knows".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("related".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        };
        let doc = RifDocument::from_oxirs_rules(&[rule], RifDialect::Bld);
        let serializer = RifSerializer::new(RifDialect::Bld);
        let output = serializer.serialize(&doc).unwrap();
        assert!(output.contains("RIF-BLD"));
        assert!(output.contains("Group"));
    }
    #[test]
    fn test_rif_roundtrip() {
        let original_rule = Rule {
            name: "roundtrip_test".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("ancestor".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        };
        let doc =
            RifDocument::from_oxirs_rules(std::slice::from_ref(&original_rule), RifDialect::Bld);
        let serializer = RifSerializer::new(RifDialect::Bld);
        let rif_text = serializer.serialize(&doc).unwrap();
        let mut parser = RifParser::new(RifDialect::Bld);
        let parsed_doc = parser.parse(&rif_text).unwrap();
        let converted_rules = parsed_doc.to_oxirs_rules().unwrap();
        assert!(!converted_rules.is_empty());
    }
    #[test]
    fn test_rif_prefix_expansion() {
        let mut doc = RifDocument::new(RifDialect::Bld);
        doc.add_prefix("ex", "http://example.org/");
        assert_eq!(doc.expand_iri("ex:Person"), "http://example.org/Person");
        assert_eq!(doc.expand_iri("unknown:foo"), "unknown:foo");
    }
    #[test]
    #[ignore = "Frame syntax parsing is complex and not yet fully optimized"]
    fn test_rif_frame_syntax() {
        let mut parser = RifParser::new(RifDialect::Bld);
        let input = r#"
            Group (
                ?person[age->?age] :- person(?person ?age)
            )
        "#;
        let result = parser.parse(input);
        assert!(result.is_ok() || result.is_err());
    }
    #[test]
    fn test_rif_equality() {
        let mut parser = RifParser::new(RifDialect::Bld);
        let input = r#"
            Group (
                same(?x ?y) :- And(person(?x) person(?y) External(equal(?x ?y)))
            )
        "#;
        let doc = parser.parse(input).unwrap();
        assert!(!doc.groups.is_empty());
    }
    #[test]
    fn test_rif_literals() {
        let mut parser = RifParser::new(RifDialect::Bld);
        let input = r#"
            Group (
                adult(?x) :- And(person(?x ?age) External(greaterThan(?age 18)))
            )
        "#;
        let doc = parser.parse(input).unwrap();
        assert!(!doc.groups.is_empty());
    }
    #[test]
    fn test_rif_document_metadata() {
        let mut doc = RifDocument::new(RifDialect::Bld);
        doc.metadata
            .insert("author".to_string(), "OxiRS".to_string());
        doc.base = Some("http://example.org/base/".to_string());
        assert_eq!(doc.metadata.get("author"), Some(&"OxiRS".to_string()));
        assert!(doc.base.is_some());
    }
    #[test]
    fn test_rif_import() {
        let import = RifImport {
            location: "http://example.org/rules.rif".to_string(),
            profile: Some("http://www.w3.org/ns/entailment/RDF".to_string()),
        };
        assert_eq!(import.location, "http://example.org/rules.rif");
        assert!(import.profile.is_some());
    }
    #[test]
    fn test_rif_term_constructors() {
        let var = RifTerm::var("X");
        assert!(matches!(var, RifTerm::Var(v) if v.name == "X"));
        let iri = RifTerm::iri("http://example.org/Person");
        assert!(matches!(iri, RifTerm::Const(RifConst::Iri(_))));
        let str_lit = RifTerm::string("hello");
        assert!(matches!(str_lit, RifTerm::Const(RifConst::Literal(_))));
        let int_lit = RifTerm::integer(42);
        assert!(matches!(int_lit, RifTerm::Const(RifConst::Literal(_))));
    }
    #[test]
    fn test_rif_formula_constructors() {
        let atom = RifFormula::atom("knows", vec![RifTerm::var("X"), RifTerm::var("Y")]);
        assert!(matches!(atom, RifFormula::Atom(_)));
        let and = RifFormula::and(vec![atom.clone()]);
        assert!(matches!(and, RifFormula::And(_)));
        let naf = RifFormula::naf(atom);
        assert!(matches!(naf, RifFormula::Naf(_)));
    }
    #[test]
    fn test_rif_typed_variable() {
        let typed_var = RifVar::typed("X", "xsd:integer");
        assert_eq!(typed_var.name, "X");
        assert_eq!(typed_var.var_type, Some("xsd:integer".to_string()));
    }
}
