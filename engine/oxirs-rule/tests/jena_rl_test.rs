// Integration tests for the Jena Rule Language (JRL) parser and lowerer.
// Run with:  cargo nextest run -p oxirs-rule --test jena_rl_test

use oxirs_rule::jena_rl::lexer::{Lexer, Token};
use oxirs_rule::jena_rl::lowering::lower_rule_set;
use oxirs_rule::jena_rl::parser::{parse, JrlAtom, JrlLiteral, JrlTerm};
use oxirs_rule::{parse_and_lower, parse_jrl, RuleAtom, Term};

// Common prefix declaration for tests that use `ex:`.
const EX_PREFIX: &str = "@prefix ex: <http://example.org/> .\n";

// Helper: prepend the ex: prefix and call parse_and_lower.
fn lower_ex(body: &str) -> Vec<oxirs_rule::Rule> {
    let src = format!("{}{}", EX_PREFIX, body);
    parse_and_lower(&src).expect("parse_and_lower should succeed")
}

// ---------------------------------------------------------------------------
// Lexer tests
// ---------------------------------------------------------------------------

fn tokenize_ok(input: &str) -> Vec<Token> {
    Lexer::tokenize(input)
        .expect("tokenize should succeed")
        .into_iter()
        .map(|st| st.token)
        .collect()
}

#[test]
fn test_tokenize_variable() {
    let toks = tokenize_ok("?person ?_x ?var123");
    assert_eq!(toks[0], Token::Variable("person".to_string()));
    assert_eq!(toks[1], Token::Variable("_x".to_string()));
    assert_eq!(toks[2], Token::Variable("var123".to_string()));
}

#[test]
fn test_tokenize_iri() {
    let toks = tokenize_ok("<http://example.org/Foo>");
    assert_eq!(toks[0], Token::Iri("http://example.org/Foo".to_string()));
}

#[test]
fn test_tokenize_prefixed_name() {
    let toks = tokenize_ok("rdf:type rdfs:subClassOf ex:Person");
    assert_eq!(
        toks[0],
        Token::PrefixedName("rdf".to_string(), "type".to_string())
    );
    assert_eq!(
        toks[1],
        Token::PrefixedName("rdfs".to_string(), "subClassOf".to_string())
    );
    assert_eq!(
        toks[2],
        Token::PrefixedName("ex".to_string(), "Person".to_string())
    );
}

#[test]
fn test_tokenize_arrow() {
    let toks = tokenize_ok("->");
    assert_eq!(toks[0], Token::Arrow);
}

#[test]
fn test_tokenize_brackets() {
    let toks = tokenize_ok("[ ]");
    assert_eq!(toks[0], Token::LBracket);
    assert_eq!(toks[1], Token::RBracket);
}

#[test]
fn test_tokenize_comment_ignored() {
    // # comment line should produce zero tokens (just the variable after)
    let toks = tokenize_ok("# full line comment\n?x # trailing comment\n?y");
    assert_eq!(toks.len(), 2, "only two variable tokens expected");
    assert_eq!(toks[0], Token::Variable("x".to_string()));
    assert_eq!(toks[1], Token::Variable("y".to_string()));
}

// ---------------------------------------------------------------------------
// Parser tests
// ---------------------------------------------------------------------------

fn do_parse_ok(input: &str) -> oxirs_rule::jena_rl::parser::JrlRuleSet {
    let toks = Lexer::tokenize(input).expect("lex ok");
    parse(&toks).expect("parse ok")
}

#[test]
fn test_parse_simple_rule() {
    // Use full IRIs to avoid any prefix resolution issues in parser tests.
    let rs = do_parse_ok(
        "[(?a rdf:type <http://example.org/Person>) -> (?a rdf:type <http://example.org/Human>)]",
    );
    assert_eq!(rs.rules.len(), 1);
    assert_eq!(rs.rules[0].conditions.len(), 1);
    assert_eq!(rs.rules[0].consequences.len(), 1);
    assert!(!rs.rules[0].is_backward);
}

#[test]
fn test_parse_rule_with_name() {
    let rs = do_parse_ok(
        "[parentRule: (?x <http://example.org/parent> ?y) -> (?y <http://example.org/child> ?x)]",
    );
    assert_eq!(rs.rules[0].name, Some("parentRule".to_string()));
}

#[test]
fn test_parse_builtin_equal() {
    let rs = do_parse_ok(
        "[r: (?x rdf:value ?v) (equal ?v 42) -> (?x rdf:type <http://example.org/Answer>)]",
    );
    let has_equal = rs.rules[0]
        .conditions
        .iter()
        .any(|a| matches!(a, JrlAtom::Builtin { name, .. } if name == "equal"));
    assert!(has_equal, "body should contain `equal` builtin");
}

#[test]
fn test_parse_multiple_conditions() {
    let rs = do_parse_ok(
        "[chain: (?a <http://example.org/p> ?b) (?b <http://example.org/q> ?c) -> (?a <http://example.org/r> ?c)]",
    );
    assert_eq!(rs.rules[0].conditions.len(), 2);
    assert_eq!(rs.rules[0].consequences.len(), 1);
}

#[test]
fn test_parse_prefix_declaration() {
    let src = "@prefix ex: <http://example.org/> .\n[r: (?x ex:p ?y) -> (?x ex:q ?y)]";
    let rs = do_parse_ok(src);
    assert_eq!(
        rs.prefixes.get("ex"),
        Some(&"http://example.org/".to_string())
    );
    assert_eq!(rs.rules.len(), 1);
}

#[test]
fn test_parse_string_literal_in_triple() {
    let rs = do_parse_ok(
        r#"[r: (?x <http://example.org/name> "Bob") -> (?x rdf:type <http://example.org/Person>)]"#,
    );
    let cond = &rs.rules[0].conditions[0];
    match cond {
        JrlAtom::Triple { object, .. } => {
            assert_eq!(
                *object,
                JrlTerm::Literal(JrlLiteral::String("Bob".to_string()))
            );
        }
        _ => panic!("expected Triple atom, got {:?}", cond),
    }
}

#[test]
fn test_parse_default_rdf_prefix_available() {
    // `rdf`, `rdfs`, `xsd`, `owl` should be resolvable even without @prefix declaration
    let rs = do_parse_ok("[r: (?x rdf:type rdfs:Class) -> (?x rdf:type owl:Class)]");
    assert!(rs.prefixes.contains_key("rdf"));
    assert!(rs.prefixes.contains_key("rdfs"));
    assert!(rs.prefixes.contains_key("xsd"));
    assert!(rs.prefixes.contains_key("owl"));
}

// ---------------------------------------------------------------------------
// Integration tests: parse_and_lower
// ---------------------------------------------------------------------------

#[test]
fn test_lower_simple_triple_rule() {
    let rules = lower_ex("[r: (?a rdf:type ex:Person) -> (?a rdf:type ex:Human)]");
    assert_eq!(rules.len(), 1);
    assert_eq!(rules[0].name, "r");
    assert!(matches!(rules[0].body[0], RuleAtom::Triple { .. }));
    assert!(matches!(rules[0].head[0], RuleAtom::Triple { .. }));
}

#[test]
fn test_lower_with_builtin_sum() {
    let rules = lower_ex("[r: (?x ex:a ?a) (?x ex:b ?b) (sum ?a ?b ?c) -> (?x ex:total ?c)]");
    let has_sum = rules[0]
        .body
        .iter()
        .any(|a| matches!(a, RuleAtom::Builtin { name, .. } if name == "sum"));
    assert!(has_sum, "sum builtin should be lowered as generic Builtin");
}

#[test]
fn test_lower_rule_set_with_prefixes() {
    let src =
        "@prefix ex: <http://example.org/> .\n[myRule: (?x ex:knows ?y) -> (?y ex:knownBy ?x)]";
    let rules = parse_and_lower(src).expect("lower ok");
    assert_eq!(rules.len(), 1);
    assert_eq!(rules[0].name, "myRule");
    // predicate should expand to full IRI
    match &rules[0].body[0] {
        RuleAtom::Triple { predicate, .. } => {
            assert_eq!(
                predicate,
                &Term::Constant("http://example.org/knows".to_string())
            );
        }
        _ => panic!("expected Triple"),
    }
}

#[test]
fn test_parse_jrl_empty_returns_empty() {
    let rs = parse_jrl("").expect("empty input OK");
    assert!(rs.rules.is_empty());
    assert!(
        rs.prefixes.contains_key("rdf"),
        "default prefixes should be populated"
    );
}

#[test]
fn test_lower_not_equal_maps_to_typed_atom() {
    let rules = lower_ex("[r: (?x ex:age ?a) (notEqual ?a 0) -> (?x ex:alive true)]");
    let has_ne = rules[0]
        .body
        .iter()
        .any(|a| matches!(a, RuleAtom::NotEqual { .. }));
    assert!(has_ne, "notEqual should map to RuleAtom::NotEqual");
}

#[test]
fn test_lower_less_than_maps_to_typed_atom() {
    let rules = lower_ex("[r: (?x ex:score ?s) (lessThan ?s 50) -> (?x ex:failing true)]");
    let has_lt = rules[0]
        .body
        .iter()
        .any(|a| matches!(a, RuleAtom::LessThan { .. }));
    assert!(has_lt);
}

#[test]
fn test_lower_greater_than_maps_to_typed_atom() {
    let rules = lower_ex("[r: (?x ex:score ?s) (greaterThan ?s 90) -> (?x ex:distinction true)]");
    let has_gt = rules[0]
        .body
        .iter()
        .any(|a| matches!(a, RuleAtom::GreaterThan { .. }));
    assert!(has_gt);
}

#[test]
fn test_lower_comments_skipped_correctly() {
    let src = format!(
        "{}# This is a comment\n# Another comment\n[r: (?x rdf:type ex:Foo) -> (?x rdf:type ex:Bar)]",
        EX_PREFIX
    );
    let rules = parse_and_lower(&src).expect("lower ok");
    assert_eq!(rules.len(), 1);
}

#[test]
fn test_lower_multiple_rules_in_one_file() {
    let rules = lower_ex("[r1: (?a ex:p ?b) -> (?a ex:q ?b)]\n[r2: (?x ex:q ?y) -> (?x ex:r ?y)]");
    assert_eq!(rules.len(), 2);
    assert_eq!(rules[0].name, "r1");
    assert_eq!(rules[1].name, "r2");
}

#[test]
fn test_lower_variable_terms_stay_variable() {
    let rules = lower_ex("[r: (?s ex:p ?o) -> (?s ex:q ?o)]");
    match &rules[0].body[0] {
        RuleAtom::Triple {
            subject, object, ..
        } => {
            assert!(matches!(subject, Term::Variable(n) if n == "s"));
            assert!(matches!(object, Term::Variable(n) if n == "o"));
        }
        _ => panic!("expected Triple"),
    }
}

#[test]
fn test_lower_rdf_type_expands_correctly() {
    let rules = lower_ex("[r: (?x rdf:type ex:Foo) -> (?x rdf:type ex:Bar)]");
    match &rules[0].body[0] {
        RuleAtom::Triple { predicate, .. } => {
            assert_eq!(
                predicate,
                &Term::Constant("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string())
            );
        }
        _ => panic!("expected Triple"),
    }
}

#[test]
fn test_lower_string_literal_to_term_literal() {
    let rules = lower_ex(r#"[r: (?x ex:name "Alice") -> (?x rdf:type ex:Person)]"#);
    match &rules[0].body[0] {
        RuleAtom::Triple { object, .. } => {
            assert_eq!(*object, Term::Literal("Alice".to_string()));
        }
        _ => panic!("expected Triple"),
    }
}

#[test]
fn test_lower_backward_rule_conditions_become_body() {
    let src = format!(
        "{}[anc: (?x ex:ancestor ?z) <- (?x ex:parent ?z)]",
        EX_PREFIX
    );
    let toks = Lexer::tokenize(&src).expect("lex ok");
    let rs = parse(&toks).expect("parse ok");
    assert!(rs.rules[0].is_backward);

    let rules = lower_rule_set(&rs).expect("lower ok");
    // backward rules are still lowered: conditions=body, consequences=head
    assert_eq!(rules[0].body.len(), 1);
    assert_eq!(rules[0].head.len(), 1);
}

#[test]
fn test_lower_chain_rule_with_transitive_pattern() {
    let src = format!(
        "{}[trans: (?x ex:parent ?y) (?y ex:parent ?z) -> (?x ex:grandparent ?z)]",
        EX_PREFIX
    );
    let rules = parse_and_lower(&src).expect("lower ok");
    assert_eq!(rules[0].body.len(), 2);
    assert_eq!(rules[0].head.len(), 1);
    // Both conditions must be Triple atoms with variable subjects/objects
    for atom in &rules[0].body {
        assert!(matches!(atom, RuleAtom::Triple { .. }));
    }
}
