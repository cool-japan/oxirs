//! SPARQL 1.1 `BuiltInCall` parsing and evaluation tests.
//!
//! The oxirs-arq expression grammar previously had no `BuiltInCall` production:
//! a bare keyword such as `lang` / `isIRI` / `regex` was lexed as an empty-prefix
//! `PrefixedName` and lowered to `Expression::Function { name: ":lang", .. }`,
//! which the evaluator (keyed on bare names) rejected as an unknown function.
//! These tests lock in the fix:
//!
//! * Part (a) — the parser recognises built-in call names **case-insensitively**
//!   and lowers each to the canonical AST shape the evaluator expects (a bare
//!   lower-case `Function`, or a dedicated `Unary` / `Bound` / `Conditional`
//!   variant), validates argument counts, and never reclassifies a leading-colon
//!   user function such as `:lang` as a built-in.
//! * Part (b) — a query parsed by the real parser and executed by the real
//!   engine over an in-memory dataset returns the correct rows for
//!   `FILTER(lang(?l)="ja")`, `BIND(STR(?s) AS ?x)`, `regex`, `bound`, the
//!   `isIRI`/`isLiteral` predicates, `STRLEN`/`CONTAINS`, and `COALESCE`/`IF`.

use oxirs_arq::algebra::{
    Aggregate, BinaryOperator, Expression, GroupCondition, PropertyPath, Term, UnaryOperator,
};
use oxirs_arq::query::{ProjectionItem, QueryParser};
use oxirs_arq::{Algebra, Dataset, Literal, QueryExecutor, TriplePattern, Variable};
use oxirs_core::model::NamedNode;

// ---------------------------------------------------------------------------
// Part (a): parser-level BuiltInCall coverage
// ---------------------------------------------------------------------------

/// Parse `SELECT (<select_expr> AS ?y) WHERE { ?x ?p ?o }` and return the parsed
/// projection expression, so a built-in call can be inspected as an AST.
fn projected_expr(select_expr: &str) -> Expression {
    let query = format!("SELECT ({select_expr} AS ?y) WHERE {{ ?x ?p ?o }}");
    let mut parser = QueryParser::new();
    let parsed = parser
        .parse(&query)
        .unwrap_or_else(|e| panic!("query `{query}` should parse: {e}"));
    match parsed.projection_items.into_iter().next() {
        Some(ProjectionItem::Expression { expr, .. }) => expr,
        other => panic!("expected an expression projection, got {other:?}"),
    }
}

/// Parse a projection expression, expecting a parse failure.
fn expect_parse_error(select_expr: &str) {
    let query = format!("SELECT ({select_expr} AS ?y) WHERE {{ ?x ?p ?o }}");
    let mut parser = QueryParser::new();
    assert!(
        parser.parse(&query).is_err(),
        "query `{query}` must fail to parse (arity/shape error)"
    );
}

#[test]
fn builtin_lowers_to_bare_canonical_function_name() {
    // A function-routed built-in becomes an `Expression::Function` keyed by its
    // canonical lower-case name — matching the evaluator's function table, not
    // the old `:lang` empty-prefix spelling.
    match projected_expr("LANG(?x)") {
        Expression::Function { name, args } => {
            assert_eq!(name, "lang", "canonical built-in name must be bare `lang`");
            assert_eq!(args.len(), 1);
        }
        other => panic!("LANG(?x) must lower to a Function, got {other:?}"),
    }

    match projected_expr("REGEX(?x, \"foo\")") {
        Expression::Function { name, args } => {
            assert_eq!(name, "regex");
            assert_eq!(args.len(), 2);
        }
        other => panic!("REGEX must lower to a Function, got {other:?}"),
    }
}

#[test]
fn builtin_recognition_is_case_insensitive() {
    for spelling in ["LANG(?x)", "lang(?x)", "Lang(?x)", "lAnG(?x)"] {
        match projected_expr(spelling) {
            Expression::Function { name, .. } => {
                assert_eq!(name, "lang", "`{spelling}` must lower to bare `lang`")
            }
            other => panic!("`{spelling}` must lower to a Function, got {other:?}"),
        }
    }
}

#[test]
fn type_check_predicates_lower_to_unary_variants() {
    // The evaluator handles isIRI/isURI/isBLANK/isLITERAL/isNUMERIC as native
    // `Unary` operators, so the parser must route them there (not to a Function).
    assert!(matches!(
        projected_expr("isIRI(?x)"),
        Expression::Unary {
            op: UnaryOperator::IsIri,
            ..
        }
    ));
    assert!(matches!(
        projected_expr("isURI(?x)"),
        Expression::Unary {
            op: UnaryOperator::IsIri,
            ..
        }
    ));
    assert!(matches!(
        projected_expr("isBLANK(?x)"),
        Expression::Unary {
            op: UnaryOperator::IsBlank,
            ..
        }
    ));
    assert!(matches!(
        projected_expr("isLITERAL(?x)"),
        Expression::Unary {
            op: UnaryOperator::IsLiteral,
            ..
        }
    ));
    assert!(matches!(
        projected_expr("isNUMERIC(?x)"),
        Expression::Unary {
            op: UnaryOperator::IsNumeric,
            ..
        }
    ));
}

#[test]
fn bound_and_if_lower_to_dedicated_variants() {
    assert!(
        matches!(projected_expr("BOUND(?x)"), Expression::Bound(_)),
        "BOUND(?var) must lower to Expression::Bound"
    );
    assert!(
        matches!(
            projected_expr("IF(?a, ?b, ?c)"),
            Expression::Conditional { .. }
        ),
        "IF(a,b,c) must lower to Expression::Conditional"
    );
}

#[test]
fn bound_requires_a_variable_argument() {
    // BOUND takes a variable, not an arbitrary expression.
    expect_parse_error("BOUND(1 + 2)");
}

#[test]
fn builtin_arity_errors_are_parse_time() {
    // Wrong argument counts must fail at parse time (surfacing as a 4xx over
    // HTTP) rather than deep in execution.
    expect_parse_error("REGEX(?x)"); // needs 2-3
    expect_parse_error("IF(?a, ?b)"); // needs exactly 3
    expect_parse_error("CONTAINS(?x)"); // needs exactly 2
    expect_parse_error("LANGMATCHES(?x)"); // needs exactly 2
    expect_parse_error("STRLEN(?x, ?y)"); // needs exactly 1
    expect_parse_error("NOW(?x)"); // needs exactly 0
    expect_parse_error("REPLACE(?x, ?y)"); // needs 3-4
}

#[test]
fn variadic_builtins_accept_multiple_arguments() {
    match projected_expr("COALESCE(?a, ?b, ?c)") {
        Expression::Function { name, args } => {
            assert_eq!(name, "coalesce");
            assert_eq!(args.len(), 3);
        }
        other => panic!("COALESCE must lower to a Function, got {other:?}"),
    }
    match projected_expr("CONCAT(?a, ?b, ?c, ?d)") {
        Expression::Function { name, args } => {
            assert_eq!(name, "concat");
            assert_eq!(args.len(), 4);
        }
        other => panic!("CONCAT must lower to a Function, got {other:?}"),
    }
}

#[test]
fn leading_colon_user_function_is_not_a_builtin() {
    // `:lang(?x)` under `PREFIX : <…>` is a user-defined function call, NOT the
    // LANG built-in. The tokenizer must keep it a PrefixedName so it lowers to a
    // colon-qualified Function name, never the bare `lang` built-in.
    let query = "PREFIX : <http://ex/> SELECT (:lang(?x) AS ?y) WHERE { ?x ?p ?o }";
    let mut parser = QueryParser::new();
    let parsed = parser
        .parse(query)
        .expect("prefixed user function should parse");
    match parsed.projection_items.into_iter().next() {
        Some(ProjectionItem::Expression {
            expr: Expression::Function { name, .. },
            ..
        }) => assert_eq!(
            name, ":lang",
            "a leading-colon `:lang` must stay a user function, not the LANG built-in"
        ),
        other => panic!("expected a colon-qualified user Function, got {other:?}"),
    }
}

#[test]
fn aggregate_names_are_not_swallowed_by_builtin_path() {
    // COUNT/SUM/... are recognised on the aggregate path, not as built-in calls.
    let mut parser = QueryParser::new();
    let parsed = parser
        .parse("SELECT (COUNT(?x) AS ?n) WHERE { ?x ?p ?o }")
        .expect("aggregate SELECT should parse");
    assert!(
        matches!(
            parsed.projection_items.first(),
            Some(ProjectionItem::Aggregate { .. })
        ),
        "COUNT(?x) must parse as an aggregate projection, not a built-in call"
    );
}

// ---------------------------------------------------------------------------
// Part (b): end-to-end parse + evaluate over an in-memory dataset
// ---------------------------------------------------------------------------

/// Minimal in-memory dataset over the default graph, mirroring the mock used by
/// the existing arq compliance tests.
struct MemDataset {
    triples: Vec<(Term, Term, Term)>,
}

impl MemDataset {
    fn add(&mut self, s: Term, p: Term, o: Term) {
        self.triples.push((s, p, o));
    }
}

fn iri(s: &str) -> Term {
    Term::Iri(NamedNode::new_unchecked(s))
}

fn lang_lit(value: &str, lang: &str) -> Term {
    Term::Literal(Literal {
        value: value.to_string(),
        language: Some(lang.to_string()),
        datatype: None,
    })
}

fn typed_lit(value: &str, datatype: &str) -> Term {
    Term::Literal(Literal {
        value: value.to_string(),
        language: None,
        datatype: Some(NamedNode::new_unchecked(datatype)),
    })
}

const LABEL: &str = "http://ex/label";
const REF: &str = "http://ex/ref";
const AGE: &str = "http://ex/age";
const XSD_INT: &str = "http://www.w3.org/2001/XMLSchema#integer";

/// Build the shared multilingual fixture:
/// `c1`/`c2` each carry a `@ja` and an `@en` label, `c1` has an IRI-valued `ref`
/// and an `xsd:integer` `age`.
fn multilingual_dataset() -> MemDataset {
    let mut ds = MemDataset {
        triples: Vec::new(),
    };
    ds.add(iri("http://ex/c1"), iri(LABEL), lang_lit("犬", "ja"));
    ds.add(iri("http://ex/c1"), iri(LABEL), lang_lit("dog", "en"));
    ds.add(iri("http://ex/c2"), iri(LABEL), lang_lit("猫", "ja"));
    ds.add(iri("http://ex/c2"), iri(LABEL), lang_lit("cat", "en"));
    ds.add(iri("http://ex/c1"), iri(REF), iri("http://ex/other"));
    ds.add(iri("http://ex/c1"), iri(AGE), typed_lit("30", XSD_INT));
    ds
}

/// Match a pattern slot (subject/object) against a stored term.
fn slot_matches(slot: &Term, value: &Term) -> bool {
    matches!(slot, Term::Variable(_)) || slot == value
}

/// Match a pattern predicate against a stored predicate. The parser lowers a
/// plain IRI predicate to `Term::PropertyPath(PropertyPath::Iri(..))`, so unwrap
/// that single-IRI/variable path form before comparing to the stored IRI.
fn pred_matches(slot: &Term, stored: &Term) -> bool {
    match slot {
        Term::Variable(_) => true,
        Term::PropertyPath(PropertyPath::Variable(_)) => true,
        Term::PropertyPath(PropertyPath::Iri(n)) => *stored == Term::Iri(n.clone()),
        other => other == stored,
    }
}

impl Dataset for MemDataset {
    fn find_triples(&self, pattern: &TriplePattern) -> anyhow::Result<Vec<(Term, Term, Term)>> {
        Ok(self
            .triples
            .iter()
            .filter(|(s, p, o)| {
                slot_matches(&pattern.subject, s)
                    && pred_matches(&pattern.predicate, p)
                    && slot_matches(&pattern.object, o)
            })
            .cloned()
            .collect())
    }

    fn contains_triple(&self, s: &Term, p: &Term, o: &Term) -> anyhow::Result<bool> {
        Ok(self
            .triples
            .iter()
            .any(|(a, b, c)| a == s && b == p && c == o))
    }

    fn subjects(&self) -> anyhow::Result<Vec<Term>> {
        Ok(self.triples.iter().map(|(s, _, _)| s.clone()).collect())
    }

    fn predicates(&self) -> anyhow::Result<Vec<Term>> {
        Ok(self.triples.iter().map(|(_, p, _)| p.clone()).collect())
    }

    fn objects(&self) -> anyhow::Result<Vec<Term>> {
        Ok(self.triples.iter().map(|(_, _, o)| o.clone()).collect())
    }
}

/// Parse `query`, execute its WHERE clause against `dataset`, and return the raw
/// solution bindings.
fn run(query: &str, dataset: &MemDataset) -> Vec<std::collections::HashMap<Variable, Term>> {
    let mut parser = QueryParser::new();
    let parsed = parser
        .parse(query)
        .unwrap_or_else(|e| panic!("query `{query}` should parse: {e}"));
    let mut executor = QueryExecutor::new();
    let (solution, _stats) = executor
        .execute(&parsed.where_clause, dataset)
        .unwrap_or_else(|e| panic!("query `{query}` should execute: {e}"));
    solution
        .iter()
        .map(|binding| {
            binding
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        })
        .collect()
}

fn var(name: &str) -> Variable {
    Variable::new(name).expect("valid variable")
}

#[test]
fn eval_filter_lang_ja_returns_only_japanese_rows() {
    // The wik.jp core query: keep only the @ja labels.
    let ds = multilingual_dataset();
    let rows = run(
        "SELECT ?c ?l WHERE { ?c <http://ex/label> ?l FILTER(lang(?l) = \"ja\") }",
        &ds,
    );
    assert_eq!(
        rows.len(),
        2,
        "exactly the two @ja labels must survive: {rows:?}"
    );
    for row in &rows {
        match row.get(&var("l")) {
            Some(Term::Literal(lit)) => assert_eq!(
                lit.language.as_deref(),
                Some("ja"),
                "every surviving ?l must be @ja: {lit:?}"
            ),
            other => panic!("?l must bind a literal, got {other:?}"),
        }
    }
}

#[test]
fn eval_bind_str_binds_the_lexical_form() {
    let ds = multilingual_dataset();
    let rows = run(
        "SELECT ?c ?x WHERE { ?c <http://ex/ref> ?o BIND(STR(?c) AS ?x) }",
        &ds,
    );
    assert_eq!(rows.len(), 1, "one ref triple: {rows:?}");
    match rows[0].get(&var("x")) {
        Some(Term::Literal(lit)) => assert_eq!(lit.value, "http://ex/c1"),
        other => panic!("BIND(STR(?c)) must bind ?x to the IRI string, got {other:?}"),
    }
}

#[test]
fn eval_filter_regex_matches_substring() {
    let ds = multilingual_dataset();
    let rows = run(
        "SELECT ?l WHERE { ?c <http://ex/label> ?l FILTER(regex(?l, \"犬\")) }",
        &ds,
    );
    assert_eq!(rows.len(), 1, "only the 犬 label matches: {rows:?}");
}

#[test]
fn eval_filter_regex_case_insensitive_flag() {
    let ds = multilingual_dataset();
    let rows = run(
        "SELECT ?l WHERE { ?c <http://ex/label> ?l FILTER(regex(?l, \"DOG\", \"i\")) }",
        &ds,
    );
    assert_eq!(
        rows.len(),
        1,
        "case-insensitive regex must match `dog`: {rows:?}"
    );
}

#[test]
fn eval_filter_bound_and_unbound() {
    let ds = multilingual_dataset();
    let bound_rows = run(
        "SELECT ?l WHERE { ?c <http://ex/label> ?l FILTER(bound(?l)) }",
        &ds,
    );
    assert_eq!(
        bound_rows.len(),
        4,
        "all four labels are bound: {bound_rows:?}"
    );

    let unbound_rows = run(
        "SELECT ?l WHERE { ?c <http://ex/label> ?l FILTER(bound(?missing)) }",
        &ds,
    );
    assert!(
        unbound_rows.is_empty(),
        "bound(?missing) is false for every row: {unbound_rows:?}"
    );
}

#[test]
fn eval_filter_is_iri_and_is_literal() {
    let ds = multilingual_dataset();
    // Objects of ?c ?p ?o: 4 literals + 1 IRI (ref) + 1 typed literal (age) = 5 literals, 1 IRI.
    let iri_rows = run("SELECT ?o WHERE { ?c ?p ?o FILTER(isIRI(?o)) }", &ds);
    assert_eq!(
        iri_rows.len(),
        1,
        "only the ref object is an IRI: {iri_rows:?}"
    );

    let literal_rows = run("SELECT ?o WHERE { ?c ?p ?o FILTER(isLiteral(?o)) }", &ds);
    assert_eq!(
        literal_rows.len(),
        5,
        "the four labels plus the age are literals: {literal_rows:?}"
    );
}

#[test]
fn eval_strlen_and_contains() {
    let ds = multilingual_dataset();
    // CONTAINS filter over the label text.
    let contains_rows = run(
        "SELECT ?l WHERE { ?c <http://ex/label> ?l FILTER(CONTAINS(?l, \"猫\")) }",
        &ds,
    );
    assert_eq!(
        contains_rows.len(),
        1,
        "only 猫 contains 猫: {contains_rows:?}"
    );

    // STRLEN via BIND, then filter numerically (avoids the typed/untyped
    // equality subtlety and just proves STRLEN evaluates to a number).
    let strlen_rows = run(
        "SELECT ?n WHERE { ?c <http://ex/label> ?l BIND(STRLEN(?l) AS ?n) FILTER(?n > 0) }",
        &ds,
    );
    assert_eq!(
        strlen_rows.len(),
        4,
        "every label has positive length: {strlen_rows:?}"
    );
    for row in &strlen_rows {
        match row.get(&var("n")) {
            Some(Term::Literal(lit)) => {
                let len: i64 = lit.value.parse().expect("STRLEN yields an integer");
                assert!(len >= 1, "STRLEN must be >= 1, got {len}");
            }
            other => panic!("STRLEN must bind ?n to an integer literal, got {other:?}"),
        }
    }
}

#[test]
fn eval_coalesce_and_if() {
    let ds = multilingual_dataset();
    // COALESCE skips the unbound ?missing and returns ?l.
    let coalesce_rows = run(
        "SELECT ?x WHERE { ?c <http://ex/label> ?l BIND(COALESCE(?missing, ?l) AS ?x) FILTER(lang(?l)=\"ja\") }",
        &ds,
    );
    assert_eq!(coalesce_rows.len(), 2, "two @ja rows: {coalesce_rows:?}");
    for row in &coalesce_rows {
        assert!(
            row.get(&var("x")).is_some(),
            "COALESCE(?missing, ?l) must bind ?x: {row:?}"
        );
    }

    // IF(isLiteral(?l), "lit", "notlit") -> "lit" for every label row.
    let if_rows = run(
        "SELECT ?x WHERE { ?c <http://ex/label> ?l BIND(IF(isLiteral(?l), \"lit\", \"notlit\") AS ?x) }",
        &ds,
    );
    assert_eq!(if_rows.len(), 4);
    for row in &if_rows {
        match row.get(&var("x")) {
            Some(Term::Literal(lit)) => assert_eq!(lit.value, "lit"),
            other => panic!("IF must bind ?x to \"lit\", got {other:?}"),
        }
    }
}

#[test]
fn eval_langmatches() {
    let ds = multilingual_dataset();
    let rows = run(
        "SELECT ?l WHERE { ?c <http://ex/label> ?l FILTER(langMatches(lang(?l), \"ja\")) }",
        &ds,
    );
    assert_eq!(
        rows.len(),
        2,
        "langMatches(lang(?l), \"ja\") keeps the @ja rows: {rows:?}"
    );
}

#[test]
fn eval_unknown_function_is_a_loud_error() {
    // A parseable but unimplemented built-in must fail loud on the filter path
    // (an UnknownFunctionError), never silently drop rows.
    let ds = multilingual_dataset();
    let mut parser = QueryParser::new();
    let parsed = parser
        .parse(
            "SELECT ?l WHERE { ?c <http://ex/label> ?l FILTER(REPLACE(?l, \"a\", \"b\") = \"x\") }",
        )
        .expect("REPLACE parses (arity ok) even though the evaluator lacks it");
    let mut executor = QueryExecutor::new();
    assert!(
        executor.execute(&parsed.where_clause, &ds).is_err(),
        "an unimplemented built-in must fail loud, not return a silently-shrunk result"
    );
}

// ---------------------------------------------------------------------------
// Round 2: bare `a` (rdf:type), language/typed literals, GROUP BY (expr),
//          HAVING(COUNT(*))
// ---------------------------------------------------------------------------

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const XSD_INTEGER: &str = "http://www.w3.org/2001/XMLSchema#integer";

/// Find the first triple of the first BGP in an algebra tree.
fn first_bgp_triple(alg: &Algebra) -> Option<TriplePattern> {
    match alg {
        Algebra::Bgp(triples) => triples.first().cloned(),
        Algebra::Filter { pattern, .. } => first_bgp_triple(pattern),
        Algebra::Join { left, right } => first_bgp_triple(left).or_else(|| first_bgp_triple(right)),
        Algebra::Extend { pattern, .. } => first_bgp_triple(pattern),
        _ => None,
    }
}

#[test]
fn bare_a_parses_as_rdf_type_predicate() {
    let mut parser = QueryParser::new();
    let parsed = parser
        .parse("SELECT ?s WHERE { ?s a <http://ex/C> }")
        .expect("`?s a <C>` must parse");
    let triple = first_bgp_triple(&parsed.where_clause).expect("a BGP triple");
    let is_rdf_type = match &triple.predicate {
        Term::PropertyPath(PropertyPath::Iri(n)) => n.as_str() == RDF_TYPE,
        Term::Iri(n) => n.as_str() == RDF_TYPE,
        _ => false,
    };
    assert!(
        is_rdf_type,
        "`a` must lower to the rdf:type predicate, got {:?}",
        triple.predicate
    );
}

#[test]
fn a_lookalikes_are_not_rdf_type() {
    // `?a` is a variable, `:a` / `a:p` are prefixed names, `"a"` is a literal.
    // None of these may be reclassified as the rdf:type shorthand.
    let mut p = QueryParser::new();
    let q = p
        .parse("PREFIX : <http://ex/> SELECT ?x WHERE { ?x :a ?y }")
        .expect("`:a` predicate must parse");
    match first_bgp_triple(&q.where_clause).map(|t| t.predicate) {
        Some(Term::PropertyPath(PropertyPath::Iri(n))) | Some(Term::Iri(n)) => {
            assert_eq!(
                n.as_str(),
                "http://ex/a",
                "`:a` must resolve to the default-prefix IRI, not rdf:type"
            )
        }
        other => panic!("`:a` predicate must be an IRI, got {other:?}"),
    }

    let mut p2 = QueryParser::new();
    let q2 = p2
        .parse("PREFIX a: <http://ex/> SELECT ?x WHERE { ?x a:p ?y }")
        .expect("`a:p` (prefix `a`) must parse");
    match first_bgp_triple(&q2.where_clause).map(|t| t.predicate) {
        Some(Term::PropertyPath(PropertyPath::Iri(n))) | Some(Term::Iri(n)) => {
            assert_eq!(
                n.as_str(),
                "http://ex/p",
                "`a:p` must resolve via the `a` prefix"
            )
        }
        other => panic!("`a:p` predicate must be an IRI, got {other:?}"),
    }

    // `?a` variable and `"a"` literal must both parse cleanly.
    let mut p3 = QueryParser::new();
    assert!(p3.parse("SELECT ?a WHERE { ?a ?p ?o }").is_ok());
    let mut p4 = QueryParser::new();
    let q4 = p4
        .parse("SELECT ?s WHERE { ?s ?p \"a\" }")
        .expect("`\"a\"` literal object must parse");
    assert!(matches!(
        first_bgp_triple(&q4.where_clause).map(|t| t.object),
        Some(Term::Literal(l)) if l.value == "a"
    ));
}

#[test]
fn eval_bare_a_matches_rdf_type() {
    let mut ds = MemDataset {
        triples: Vec::new(),
    };
    ds.add(iri("http://ex/s1"), iri(RDF_TYPE), iri("http://ex/C"));
    ds.add(iri("http://ex/s2"), iri(RDF_TYPE), iri("http://ex/C"));
    ds.add(iri("http://ex/s3"), iri(RDF_TYPE), iri("http://ex/D"));
    let rows = run("SELECT ?s WHERE { ?s a <http://ex/C> }", &ds);
    assert_eq!(
        rows.len(),
        2,
        "`?s a <C>` must match exactly the two type-C subjects: {rows:?}"
    );
}

#[test]
fn lang_literal_parses_in_pattern_and_filter() {
    // Pattern object position.
    let mut p = QueryParser::new();
    let q = p
        .parse("SELECT ?s WHERE { ?s ?p \"foo\"@ja }")
        .expect("a language-tagged literal object must parse");
    match first_bgp_triple(&q.where_clause).map(|t| t.object) {
        Some(Term::Literal(l)) => {
            assert_eq!(l.value, "foo");
            assert_eq!(l.language.as_deref(), Some("ja"));
        }
        other => panic!("object must be a lang literal, got {other:?}"),
    }

    // Expression (FILTER) position: the RHS of `?x = "foo"@ja`.
    match projected_expr("?x = \"foo\"@ja") {
        Expression::Binary { right, .. } => match *right {
            Expression::Literal(l) => {
                assert_eq!(l.value, "foo");
                assert_eq!(l.language.as_deref(), Some("ja"));
            }
            other => panic!("RHS must be a lang literal, got {other:?}"),
        },
        other => panic!("`?x = \"foo\"@ja` must be a Binary comparison, got {other:?}"),
    }
}

#[test]
fn typed_literal_parses_with_iri_and_prefixed_datatype() {
    // Absolute-IRI datatype.
    let mut p = QueryParser::new();
    let q = p
        .parse("SELECT ?s WHERE { ?s ?p \"1\"^^<http://www.w3.org/2001/XMLSchema#integer> }")
        .expect("an IRI-typed literal must parse");
    match first_bgp_triple(&q.where_clause).map(|t| t.object) {
        Some(Term::Literal(l)) => {
            assert_eq!(l.value, "1");
            assert_eq!(l.datatype.as_ref().map(|d| d.as_str()), Some(XSD_INTEGER));
        }
        other => panic!("object must be a typed literal, got {other:?}"),
    }

    // Prefixed datatype resolves against the declared prefix.
    let mut p2 = QueryParser::new();
    let q2 = p2
        .parse("PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> SELECT ?s WHERE { ?s ?p \"1\"^^xsd:integer }")
        .expect("a prefixed-datatype literal must parse");
    match first_bgp_triple(&q2.where_clause).map(|t| t.object) {
        Some(Term::Literal(l)) => {
            assert_eq!(l.datatype.as_ref().map(|d| d.as_str()), Some(XSD_INTEGER))
        }
        other => panic!("prefixed-datatype object must be a typed literal, got {other:?}"),
    }
}

#[test]
fn eval_filter_lang_literal_exact_match() {
    let ds = multilingual_dataset();
    // Only 犬@ja matches: 猫@ja differs in value, dog/cat differ in tag.
    let rows = run(
        "SELECT ?c WHERE { ?c <http://ex/label> ?l FILTER(?l = \"犬\"@ja) }",
        &ds,
    );
    assert_eq!(
        rows.len(),
        1,
        "exact lang-literal equality must match only 犬@ja: {rows:?}"
    );
}

#[test]
fn eval_group_by_expression_key_splits_groups() {
    // Regression guard for the GROUP BY (expr) silent mis-count: before the fix,
    // a non-variable grouping key was ignored and every row collapsed into one
    // group. LANG(?l) must split the four labels into a ja group and an en group.
    let ds = multilingual_dataset();
    let group = Algebra::Group {
        pattern: Box::new(Algebra::Bgp(vec![TriplePattern::new(
            Term::Variable(var("c")),
            Term::Iri(NamedNode::new_unchecked(LABEL)),
            Term::Variable(var("l")),
        )])),
        variables: vec![GroupCondition {
            expr: Expression::Function {
                name: "lang".to_string(),
                args: vec![Expression::Variable(var("l"))],
            },
            alias: None,
        }],
        aggregates: vec![(
            var("n"),
            Aggregate::Count {
                distinct: false,
                expr: None,
            },
        )],
    };
    let mut executor = QueryExecutor::new();
    let (solution, _stats) = executor.execute(&group, &ds).expect("group exec");
    assert_eq!(
        solution.len(),
        2,
        "GROUP BY (LANG(?l)) must yield 2 groups (ja, en), not 1: {solution:?}"
    );
    let mut counts: Vec<i64> = solution
        .iter()
        .filter_map(|b| {
            b.get(&var("n")).and_then(|t| match t {
                Term::Literal(l) => l.value.parse().ok(),
                _ => None,
            })
        })
        .collect();
    counts.sort();
    assert_eq!(
        counts,
        vec![2, 2],
        "each language group has 2 labels: {solution:?}"
    );
}

#[test]
fn having_count_star_parses() {
    // `COUNT(*)` in a HAVING expression previously failed with "Expected primary
    // expression" (the star was not accepted in an expression's argument list).
    let mut parser = QueryParser::new();
    let parsed = parser
        .parse(
            "SELECT ?d (COUNT(?s) AS ?c) WHERE { ?s <http://ex/p> ?d } \
             GROUP BY ?d HAVING (COUNT(*) > 1)",
        )
        .expect("HAVING (COUNT(*) > 1) must parse");
    assert!(
        parsed.having.is_some(),
        "the HAVING condition must be captured"
    );
}

// ---------------------------------------------------------------------------
// Round 3: predicate-object lists (`;`), object lists (`,`),
//          GROUP BY (expr AS ?var)
// ---------------------------------------------------------------------------

/// Collect every triple pattern from the BGPs in an algebra tree, as sorted
/// debug strings, for set comparison of the `;`/`,` and `.` forms.
fn triple_set(alg: &Algebra) -> Vec<String> {
    fn go(a: &Algebra, out: &mut Vec<String>) {
        match a {
            Algebra::Bgp(triples) => out.extend(triples.iter().map(|t| format!("{t:?}"))),
            Algebra::Filter { pattern, .. } => go(pattern, out),
            Algebra::Join { left, right } => {
                go(left, out);
                go(right, out);
            }
            Algebra::LeftJoin { left, right, .. } => {
                go(left, out);
                go(right, out);
            }
            Algebra::Union { left, right } => {
                go(left, out);
                go(right, out);
            }
            Algebra::Extend { pattern, .. } => go(pattern, out),
            Algebra::Graph { pattern, .. } => go(pattern, out),
            _ => {}
        }
    }
    let mut out = Vec::new();
    go(alg, &mut out);
    out.sort();
    out
}

fn where_triples(query: &str) -> Vec<String> {
    let mut parser = QueryParser::new();
    let parsed = parser
        .parse(query)
        .unwrap_or_else(|e| panic!("query `{query}` should parse: {e}"));
    triple_set(&parsed.where_clause)
}

#[test]
fn predicate_object_list_expands_like_dot_form() {
    // `;` (predicate list) must expand to exactly the `.`-separated triples.
    let semicolon =
        where_triples("PREFIX : <http://ex/> SELECT * WHERE { ?c a :C ; :p ?l ; :q ?n }");
    let dot =
        where_triples("PREFIX : <http://ex/> SELECT * WHERE { ?c a :C . ?c :p ?l . ?c :q ?n }");
    assert_eq!(
        semicolon, dot,
        "predicate-object list `;` must yield the same triples as the `.` form"
    );
    assert_eq!(semicolon.len(), 3, "three triples expected: {semicolon:?}");
}

#[test]
fn object_list_expands_like_dot_form() {
    // `,` (object list) must expand to one triple per object.
    let comma = where_triples("PREFIX : <http://ex/> SELECT * WHERE { ?c a :C , :D , :E }");
    let dot = where_triples("PREFIX : <http://ex/> SELECT * WHERE { ?c a :C . ?c a :D . ?c a :E }");
    assert_eq!(comma, dot, "object list `,` must match the `.` form");
    assert_eq!(comma.len(), 3, "three triples expected: {comma:?}");
}

#[test]
fn combined_semicolon_and_comma_and_property_path() {
    // `;`, `,` and a property-path predicate together.
    let mixed =
        where_triples("PREFIX : <http://ex/> SELECT * WHERE { ?c a :C ; :p ?a , ?b ; :p1/:p2 ?d }");
    let dot = where_triples(
        "PREFIX : <http://ex/> SELECT * WHERE { \
         ?c a :C . ?c :p ?a . ?c :p ?b . ?c :p1/:p2 ?d }",
    );
    assert_eq!(mixed, dot, "combined `;`/`,`/path must match the `.` form");
    assert_eq!(mixed.len(), 4, "four triples expected: {mixed:?}");
}

#[test]
fn trailing_semicolon_is_tolerated() {
    // A trailing `;` (and `;;`) before `}` or `.` is valid SPARQL.
    let base = where_triples("PREFIX : <http://ex/> SELECT * WHERE { ?s :p ?o }");
    for q in [
        "PREFIX : <http://ex/> SELECT * WHERE { ?s :p ?o ; }",
        "PREFIX : <http://ex/> SELECT * WHERE { ?s :p ?o ; ; }",
        "PREFIX : <http://ex/> SELECT * WHERE { ?s :p ?o ; . }",
    ] {
        assert_eq!(
            where_triples(q),
            base,
            "trailing-semicolon form `{q}` must yield the single triple"
        );
    }
}

#[test]
fn semicolon_in_optional_and_construct() {
    // `;` must work inside an OPTIONAL block and a CONSTRUCT template.
    let mut p = QueryParser::new();
    let opt = p
        .parse("PREFIX : <http://ex/> SELECT * WHERE { ?c a :C OPTIONAL { ?c :p ?l ; :q ?n } }")
        .expect("`;` inside OPTIONAL must parse");
    assert_eq!(
        triple_set(&opt.where_clause).len(),
        3,
        "1 required + 2 optional triples"
    );

    let mut p2 = QueryParser::new();
    let con = p2
        .parse("PREFIX : <http://ex/> CONSTRUCT { ?c :p ?l ; :q ?l } WHERE { ?c :label ?l }")
        .expect("`;` inside a CONSTRUCT template must parse");
    assert_eq!(
        con.construct_template.len(),
        2,
        "the CONSTRUCT template must expand to two triples"
    );
}

#[test]
fn blank_node_and_collection_syntax_are_clean_parse_errors() {
    // Blank-node property lists `[ … ]` / `[]` and RDF collections `( … )` are
    // not yet implemented. They must surface as a clear parse error (a 4xx over
    // HTTP), never a silent wrong answer.
    for query in [
        "PREFIX : <http://ex/> SELECT * WHERE { [ :p ?o ] }",
        "PREFIX : <http://ex/> SELECT * WHERE { [] :p ?o }",
        "PREFIX : <http://ex/> SELECT * WHERE { ?s :p [ :q ?o ] }",
        "PREFIX : <http://ex/> SELECT * WHERE { ?s :p ( :a :b ) }",
    ] {
        let mut parser = QueryParser::new();
        assert!(
            parser.parse(query).is_err(),
            "unsupported blank-node/collection syntax must be a parse error: `{query}`"
        );
    }
}

#[test]
fn eval_property_path_sequence() {
    // A `/`-sequence property path (previously a parse failure: the lexer emits
    // `/` as `Token::Divide`, which the path-sequence parser now accepts) must
    // both parse and evaluate: c1 --broader--> c2 --label--> {neko, cat}.
    let broader = "http://www.w3.org/2004/02/skos/core#broader";
    let mut ds = MemDataset {
        triples: Vec::new(),
    };
    ds.add(iri("http://ex/c1"), iri(broader), iri("http://ex/c2"));
    ds.add(iri("http://ex/c2"), iri(LABEL), lang_lit("neko", "ja"));
    ds.add(iri("http://ex/c2"), iri(LABEL), lang_lit("cat", "en"));
    let rows = run(
        "SELECT ?bl WHERE { <http://ex/c1> \
         <http://www.w3.org/2004/02/skos/core#broader>/<http://ex/label> ?bl }",
        &ds,
    );
    assert_eq!(
        rows.len(),
        2,
        "the broader/label sequence must reach c2's two labels: {rows:?}"
    );
}

#[test]
fn eval_predicate_object_list_matches_dot_form() {
    // End-to-end: the `;` form returns the same rows as the `.` form.
    let mut ds = MemDataset {
        triples: Vec::new(),
    };
    ds.add(iri("http://ex/c1"), iri(RDF_TYPE), iri("http://ex/C"));
    ds.add(iri("http://ex/c1"), iri("http://ex/p"), iri("http://ex/o1"));
    ds.add(iri("http://ex/c2"), iri(RDF_TYPE), iri("http://ex/C"));
    ds.add(iri("http://ex/c2"), iri("http://ex/p"), iri("http://ex/o2"));
    let semi = run(
        "SELECT ?c ?o WHERE { ?c a <http://ex/C> ; <http://ex/p> ?o }",
        &ds,
    );
    let dot = run(
        "SELECT ?c ?o WHERE { ?c a <http://ex/C> . ?c <http://ex/p> ?o }",
        &ds,
    );
    assert_eq!(semi.len(), 2, "two subjects match: {semi:?}");
    assert_eq!(
        semi.len(),
        dot.len(),
        "`;` and `.` forms must return the same rows"
    );
}

#[test]
fn group_by_expr_as_var_parses_and_binds_alias() {
    // `GROUP BY (LANG(?l) AS ?g)` must parse (the alias lives inside the parens).
    let mut parser = QueryParser::new();
    let parsed = parser
        .parse(
            "SELECT ?g (COUNT(*) AS ?n) WHERE { ?c <http://ex/label> ?l } \
             GROUP BY (LANG(?l) AS ?g)",
        )
        .expect("GROUP BY (expr AS ?var) must parse");
    assert_eq!(parsed.group_by.len(), 1, "one grouping condition");
    assert_eq!(
        parsed.group_by[0].alias.as_ref().map(|v| v.as_str()),
        Some("g"),
        "the grouping alias ?g must be captured"
    );

    // The executor binds the alias to the evaluated key.
    let ds = multilingual_dataset();
    let group = Algebra::Group {
        pattern: Box::new(Algebra::Bgp(vec![TriplePattern::new(
            Term::Variable(var("c")),
            Term::Iri(NamedNode::new_unchecked(LABEL)),
            Term::Variable(var("l")),
        )])),
        variables: parsed.group_by.clone(),
        aggregates: vec![(
            var("n"),
            Aggregate::Count {
                distinct: false,
                expr: None,
            },
        )],
    };
    let mut executor = QueryExecutor::new();
    let (solution, _stats) = executor.execute(&group, &ds).expect("group exec");
    assert_eq!(solution.len(), 2, "two language groups: {solution:?}");
    let mut langs: Vec<String> = solution
        .iter()
        .filter_map(|b| match b.get(&var("g")) {
            Some(Term::Literal(l)) => Some(l.value.clone()),
            _ => None,
        })
        .collect();
    langs.sort();
    assert_eq!(
        langs,
        vec!["en".to_string(), "ja".to_string()],
        "the alias ?g must bind the language key: {solution:?}"
    );
}

// ---------------------------------------------------------------------------
// Round 4: MINUS / OPTIONAL composition, FILTER (NOT) EXISTS, unary `!`,
//          IN / NOT IN, subquery clean-error
// ---------------------------------------------------------------------------

const DEP: &str = "http://ex/dep";

/// Three concepts typed `ex:Concept`; only `c2` is flagged deprecated
/// (`ex:dep "yes"`).
fn concept_dataset() -> MemDataset {
    let mut ds = MemDataset {
        triples: Vec::new(),
    };
    for c in ["http://ex/c1", "http://ex/c2", "http://ex/c3"] {
        ds.add(iri(c), iri(RDF_TYPE), iri("http://ex/Concept"));
    }
    ds.add(
        iri("http://ex/c2"),
        iri(DEP),
        Term::Literal(Literal {
            value: "yes".to_string(),
            language: None,
            datatype: None,
        }),
    );
    ds
}

#[test]
fn eval_minus_shared_variable_subtracts() {
    // Regression guard for the MINUS silent no-op: `MINUS { ?c dep "yes" }`
    // must remove the deprecated concept (shared variable ?c).
    let ds = concept_dataset();
    let rows = run(
        "SELECT ?c WHERE { ?c a <http://ex/Concept> MINUS { ?c <http://ex/dep> \"yes\" } }",
        &ds,
    );
    assert_eq!(
        rows.len(),
        2,
        "MINUS must subtract the deprecated concept c2, leaving c1 and c3: {rows:?}"
    );
    assert!(
        rows.iter().all(|b| match b.get(&var("c")) {
            Some(Term::Iri(n)) => n.as_str() != "http://ex/c2",
            _ => true,
        }),
        "c2 must have been removed by MINUS: {rows:?}"
    );
}

#[test]
fn eval_minus_disjoint_variables_removes_nothing() {
    // SPARQL: a MINUS whose right pattern shares NO variable with the left
    // removes nothing.
    let ds = concept_dataset();
    let rows = run(
        "SELECT ?c WHERE { ?c a <http://ex/Concept> MINUS { ?x <http://ex/dep> \"yes\" } }",
        &ds,
    );
    assert_eq!(
        rows.len(),
        3,
        "a variable-disjoint MINUS must remove nothing: {rows:?}"
    );
}

#[test]
fn eval_optional_keeps_unmatched_left_rows() {
    // Regression guard: OPTIONAL was collapsing into an inner join. Every
    // concept must survive; only c2 gets ?d bound.
    let ds = concept_dataset();
    let rows = run(
        "SELECT ?c ?d WHERE { ?c a <http://ex/Concept> OPTIONAL { ?c <http://ex/dep> ?d } }",
        &ds,
    );
    assert_eq!(
        rows.len(),
        3,
        "OPTIONAL must keep all three concepts: {rows:?}"
    );
    let bound = rows.iter().filter(|b| b.get(&var("d")).is_some()).count();
    assert_eq!(bound, 1, "only c2 has a dep value bound: {rows:?}");
}

#[test]
fn filter_exists_and_not_exists_parse() {
    // `EXISTS { … }` lowers to Expression::Exists; `NOT EXISTS` is unary NOT
    // applied to it.
    assert!(matches!(
        projected_expr("EXISTS { ?a ?b ?c }"),
        Expression::Exists(_)
    ));
    assert!(matches!(
        projected_expr("NOT EXISTS { ?a ?b ?c }"),
        Expression::Unary {
            op: UnaryOperator::Not,
            ..
        }
    ));
}

#[test]
fn eval_filter_exists_and_not_exists() {
    let ds = concept_dataset();
    let exists = run(
        "SELECT ?c WHERE { ?c a <http://ex/Concept> FILTER EXISTS { ?c <http://ex/dep> \"yes\" } }",
        &ds,
    );
    assert_eq!(
        exists.len(),
        1,
        "FILTER EXISTS keeps only the deprecated c2: {exists:?}"
    );

    let not_exists = run(
        "SELECT ?c WHERE { ?c a <http://ex/Concept> FILTER NOT EXISTS { ?c <http://ex/dep> \"yes\" } }",
        &ds,
    );
    assert_eq!(
        not_exists.len(),
        2,
        "FILTER NOT EXISTS keeps the non-deprecated c1 and c3: {not_exists:?}"
    );
}

#[test]
fn unary_not_parses_and_evaluates() {
    // `!` lowers to a unary NOT.
    assert!(matches!(
        projected_expr("!BOUND(?x)"),
        Expression::Unary {
            op: UnaryOperator::Not,
            ..
        }
    ));

    // `!BOUND(?d)` over an OPTIONAL keeps the rows where ?d is unbound.
    let ds = concept_dataset();
    let rows = run(
        "SELECT ?c WHERE { ?c a <http://ex/Concept> OPTIONAL { ?c <http://ex/dep> ?d } FILTER(!BOUND(?d)) }",
        &ds,
    );
    assert_eq!(
        rows.len(),
        2,
        "!BOUND(?d) keeps the two non-deprecated concepts: {rows:?}"
    );

    // `!EXISTS { … }` is the same as NOT EXISTS.
    let not_exists = run(
        "SELECT ?c WHERE { ?c a <http://ex/Concept> FILTER(!EXISTS { ?c <http://ex/dep> \"yes\" }) }",
        &ds,
    );
    assert_eq!(
        not_exists.len(),
        2,
        "!EXISTS keeps c1 and c3: {not_exists:?}"
    );
}

#[test]
fn in_and_not_in_parse() {
    match projected_expr("?x IN (1, 2, 3)") {
        Expression::Binary {
            op: BinaryOperator::In,
            right,
            ..
        } => match *right {
            Expression::Function { name, args } => {
                assert_eq!(name, "list");
                assert_eq!(args.len(), 3);
            }
            other => panic!("IN right operand must be a list, got {other:?}"),
        },
        other => panic!("`?x IN (…)` must be a Binary IN, got {other:?}"),
    }
    assert!(matches!(
        projected_expr("?x NOT IN (1)"),
        Expression::Binary {
            op: BinaryOperator::NotIn,
            ..
        }
    ));
    // Empty list parses.
    assert!(matches!(
        projected_expr("?x IN ()"),
        Expression::Binary {
            op: BinaryOperator::In,
            ..
        }
    ));
}

#[test]
fn eval_in_and_not_in() {
    let ds = concept_dataset();
    let in_rows = run(
        "SELECT ?c WHERE { ?c a <http://ex/Concept> FILTER(?c IN (<http://ex/c1>, <http://ex/c3>)) }",
        &ds,
    );
    assert_eq!(in_rows.len(), 2, "IN keeps c1 and c3: {in_rows:?}");

    let not_in_rows = run(
        "SELECT ?c WHERE { ?c a <http://ex/Concept> FILTER(?c NOT IN (<http://ex/c2>)) }",
        &ds,
    );
    assert_eq!(
        not_in_rows.len(),
        2,
        "NOT IN (<c2>) keeps c1 and c3: {not_in_rows:?}"
    );

    let empty_rows = run(
        "SELECT ?c WHERE { ?c a <http://ex/Concept> FILTER(?c IN ()) }",
        &ds,
    );
    assert!(
        empty_rows.is_empty(),
        "IN () is always false: {empty_rows:?}"
    );
}

#[test]
fn subquery_is_a_clean_parse_error() {
    // A `{ SELECT … }` subquery is not yet implemented; it must be a clear parse
    // error (a 4xx over HTTP), never a silent wrong answer.
    let mut parser = QueryParser::new();
    assert!(
        parser
            .parse("SELECT ?c WHERE { { SELECT ?c WHERE { ?c a <http://ex/Concept> } } }")
            .is_err(),
        "an unsupported subquery must be a parse error, not a silent result"
    );
}

// ---------------------------------------------------------------------------
// Round 5-1: EXISTS / NOT EXISTS with an inner FILTER (and correlated form)
// ---------------------------------------------------------------------------

/// Concepts with mixed language labels: c1 has @ja + @en, c2 has @ja only,
/// c3 has @en only.
fn mixed_lang_dataset() -> MemDataset {
    let label = "http://ex/label";
    let mut ds = MemDataset {
        triples: Vec::new(),
    };
    ds.add(iri("http://ex/c1"), iri(label), lang_lit("inu", "ja"));
    ds.add(iri("http://ex/c1"), iri(label), lang_lit("dog", "en"));
    ds.add(iri("http://ex/c2"), iri(label), lang_lit("neko", "ja"));
    ds.add(iri("http://ex/c3"), iri(label), lang_lit("x", "en"));
    ds
}

#[test]
fn eval_not_exists_with_inner_filter() {
    // The reported P0: an inner FILTER inside NOT EXISTS was ignored (nothing
    // excluded). `NOT EXISTS { ?c label ?e FILTER(lang(?e)="en") }` must keep
    // only the concept with NO English label (c2).
    let ds = mixed_lang_dataset();
    let rows = run(
        "SELECT DISTINCT ?c WHERE { ?c <http://ex/label> ?any \
         FILTER NOT EXISTS { ?c <http://ex/label> ?e FILTER(lang(?e) = \"en\") } }",
        &ds,
    );
    let concepts: std::collections::HashSet<String> = rows
        .iter()
        .filter_map(|b| match b.get(&var("c")) {
            Some(Term::Iri(n)) => Some(n.as_str().to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(
        concepts,
        std::collections::HashSet::from(["http://ex/c2".to_string()]),
        "NOT EXISTS with an inner lang filter must keep only the en-less concept c2: {concepts:?}"
    );
}

#[test]
fn eval_exists_with_inner_filter() {
    // `EXISTS { ?c label ?e FILTER(lang(?e)="en") }` must match the concepts
    // that DO have an English label (c1, c3).
    let ds = mixed_lang_dataset();
    let rows = run(
        "SELECT DISTINCT ?c WHERE { ?c <http://ex/label> ?any \
         FILTER EXISTS { ?c <http://ex/label> ?e FILTER(lang(?e) = \"en\") } }",
        &ds,
    );
    let concepts: std::collections::HashSet<String> = rows
        .iter()
        .filter_map(|b| match b.get(&var("c")) {
            Some(Term::Iri(n)) => Some(n.as_str().to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(
        concepts,
        std::collections::HashSet::from(["http://ex/c1".to_string(), "http://ex/c3".to_string()]),
        "EXISTS with an inner lang filter must match the en-labelled concepts: {concepts:?}"
    );
}

#[test]
fn eval_correlated_exists_references_outer_vars() {
    // A correlated EXISTS whose inner FILTER references BOTH outer variables
    // (?c and ?l): keep (?c,?l) rows whose label ?l is also held by a DIFFERENT
    // subject.
    let mut ds = MemDataset {
        triples: Vec::new(),
    };
    let p = "http://ex/p";
    ds.add(iri("http://ex/c1"), iri(p), lang_lit("a", "en"));
    ds.add(iri("http://ex/c1"), iri(p), lang_lit("b", "en"));
    ds.add(iri("http://ex/c2"), iri(p), lang_lit("a", "en"));
    let rows = run(
        "SELECT ?c ?l WHERE { ?c <http://ex/p> ?l \
         FILTER EXISTS { ?other <http://ex/p> ?l FILTER(?other != ?c) } }",
        &ds,
    );
    // \"a\" is shared by c1 and c2; \"b\" is unique to c1.
    assert_eq!(
        rows.len(),
        2,
        "only the shared-label rows (c1,a) and (c2,a) survive the correlated EXISTS: {rows:?}"
    );
    for row in &rows {
        match row.get(&var("l")) {
            Some(Term::Literal(lit)) => assert_eq!(
                lit.value, "a",
                "the surviving label must be the shared \"a\": {rows:?}"
            ),
            other => panic!("?l must bind a literal, got {other:?}"),
        }
    }
}

#[test]
fn eval_not_exists_without_inner_filter_still_works() {
    // Non-regression: the filter-less NOT EXISTS form (the R4 case).
    let ds = concept_dataset();
    let rows = run(
        "SELECT ?c WHERE { ?c a <http://ex/Concept> \
         FILTER NOT EXISTS { ?c <http://ex/dep> \"yes\" } }",
        &ds,
    );
    assert_eq!(
        rows.len(),
        2,
        "filter-less NOT EXISTS keeps the two non-deprecated concepts: {rows:?}"
    );
}

// ---------------------------------------------------------------------------
// Round 5-2: high-cardinality joins must use a hash join, not a nested loop
// ---------------------------------------------------------------------------

/// Build a dataset of `n` concepts, each `rdf:type ex:Concept` with one
/// `ex:label` literal.
fn many_concepts_dataset(n: usize) -> MemDataset {
    let mut ds = MemDataset {
        triples: Vec::with_capacity(2 * n),
    };
    for i in 0..n {
        let c = format!("http://ex/c{i}");
        ds.add(iri(&c), iri(RDF_TYPE), iri("http://ex/Concept"));
        ds.add(
            iri(&c),
            iri("http://ex/label"),
            Term::Literal(Literal {
                value: format!("label {i}"),
                language: None,
                datatype: None,
            }),
        );
    }
    ds
}

/// Run `query` against an `n`-concept dataset on a worker thread and require it
/// to finish within `secs`; otherwise the join has regressed to a nested loop.
fn run_within(query: &'static str, n: usize, secs: u64, expected_rows: usize, label: &str) {
    use std::sync::mpsc;
    use std::time::{Duration, Instant};
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let ds = many_concepts_dataset(n);
        let start = Instant::now();
        let rows = run(query, &ds);
        let _ = tx.send((rows.len(), start.elapsed()));
    });
    match rx.recv_timeout(Duration::from_secs(secs)) {
        Ok((rows, elapsed)) => {
            eprintln!("[{label}] {n} concepts joined in {elapsed:?} -> {rows} rows");
            assert_eq!(rows, expected_rows, "[{label}] row count mismatch");
        }
        Err(_) => panic!(
            "[{label}] join over {n} concepts did not finish within {secs}s \
             — nested-loop regression"
        ),
    }
}

#[test]
fn perf_predicate_object_list_join_is_not_nested_loop() {
    run_within(
        "SELECT ?c ?l WHERE { ?c a <http://ex/Concept> ; <http://ex/label> ?l }",
        20_000,
        30,
        20_000,
        "bgp-join",
    );
}

#[test]
fn perf_optional_join_is_not_nested_loop() {
    run_within(
        "SELECT ?c ?l WHERE { ?c a <http://ex/Concept> OPTIONAL { ?c <http://ex/label> ?l } }",
        20_000,
        30,
        20_000,
        "optional-join",
    );
}
