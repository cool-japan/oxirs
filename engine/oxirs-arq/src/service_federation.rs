//! Real SPARQL 1.1 Protocol federation for `SERVICE` clauses.
//!
//! [`execute_service_clause`] issues an actual HTTP SPARQL query to the remote
//! endpoint named in a `SERVICE` clause, parses the SPARQL 1.1 JSON results,
//! and returns them as a [`Solution`]. It honours the `SILENT` modifier: on any
//! failure a `SILENT` service yields an empty solution, while a non-`SILENT`
//! service surfaces the error — it never substitutes local data for a remote
//! result.
//!
//! The HTTP request is driven on a dedicated worker thread with its own
//! single-threaded Tokio runtime so it is safe to call from both synchronous
//! and asynchronous contexts without risking a nested-runtime panic. The
//! networking path requires the `parallel` feature (which pulls in Tokio); when
//! that feature is disabled a non-`SILENT` `SERVICE` fails loudly instead of
//! fabricating a result.

use crate::algebra::{
    Algebra, BinaryOperator, Expression, Literal, PropertyPath, Solution, Term, UnaryOperator,
};
use anyhow::{anyhow, Result};
use std::time::Duration;

/// Default timeout for a federated `SERVICE` request.
pub const DEFAULT_SERVICE_TIMEOUT: Duration = Duration::from_secs(60);

/// Execute a SPARQL `SERVICE` clause against its remote endpoint.
///
/// On success the remote solution is returned verbatim. On failure the `silent`
/// flag decides the outcome: `SILENT` services return an empty solution,
/// non-`SILENT` services return the error.
pub fn execute_service_clause(
    endpoint: &Term,
    pattern: &Algebra,
    silent: bool,
) -> Result<Solution> {
    match execute_service_inner(endpoint, pattern) {
        Ok(solution) => Ok(solution),
        Err(err) => {
            if silent {
                tracing::warn!("SILENT SERVICE failed, returning empty solution: {err}");
                Ok(Vec::new())
            } else {
                Err(err)
            }
        }
    }
}

fn execute_service_inner(endpoint: &Term, pattern: &Algebra) -> Result<Solution> {
    let endpoint_url = endpoint_iri(endpoint)?;
    let sparql = algebra_to_select_query(pattern)?;
    http_post_sparql(&endpoint_url, &sparql, DEFAULT_SERVICE_TIMEOUT)
}

/// Extract the endpoint IRI string from a `SERVICE` endpoint term.
fn endpoint_iri(endpoint: &Term) -> Result<String> {
    match endpoint {
        Term::Iri(iri) => Ok(iri.as_str().to_string()),
        Term::Variable(v) => Err(anyhow!(
            "SERVICE endpoint variable ?{} is unbound; variable service endpoints are not supported",
            v.name()
        )),
        other => Err(anyhow!("SERVICE endpoint must be an IRI, got {other:?}")),
    }
}

// ---------------------------------------------------------------------------
// Algebra -> SPARQL query text serialization
// ---------------------------------------------------------------------------

/// Serialize a `SERVICE` group graph pattern into a `SELECT * WHERE { ... }`
/// query that can be sent to a remote endpoint.
///
/// Only group-graph-pattern shapes are serializable; query-level modifiers
/// (projection, DISTINCT, ORDER BY, LIMIT, GROUP BY, sub-SERVICE) that would
/// change result semantics if silently dropped return an error instead so the
/// caller never sends a query that means something different from the source.
pub fn algebra_to_select_query(pattern: &Algebra) -> Result<String> {
    let mut body = String::new();
    write_pattern(pattern, &mut body, 1)?;
    Ok(format!("SELECT * WHERE {{\n{body}}}\n"))
}

fn write_pattern(algebra: &Algebra, out: &mut String, indent: usize) -> Result<()> {
    let pad = "  ".repeat(indent);
    match algebra {
        Algebra::Bgp(patterns) => {
            for tp in patterns {
                out.push_str(&format!(
                    "{pad}{} {} {} .\n",
                    term_to_sparql(&tp.subject)?,
                    term_to_sparql(&tp.predicate)?,
                    term_to_sparql(&tp.object)?
                ));
            }
            Ok(())
        }
        Algebra::Join { left, right } => {
            write_pattern(left, out, indent)?;
            write_pattern(right, out, indent)
        }
        Algebra::Union { left, right } => {
            out.push_str(&format!("{pad}{{\n"));
            write_pattern(left, out, indent + 1)?;
            out.push_str(&format!("{pad}}} UNION {{\n"));
            write_pattern(right, out, indent + 1)?;
            out.push_str(&format!("{pad}}}\n"));
            Ok(())
        }
        Algebra::LeftJoin {
            left,
            right,
            filter,
        } => {
            write_pattern(left, out, indent)?;
            out.push_str(&format!("{pad}OPTIONAL {{\n"));
            write_pattern(right, out, indent + 1)?;
            if let Some(f) = filter {
                out.push_str(&format!("{pad}  FILTER({})\n", expr_to_sparql(f)?));
            }
            out.push_str(&format!("{pad}}}\n"));
            Ok(())
        }
        Algebra::Filter { pattern, condition } => {
            write_pattern(pattern, out, indent)?;
            out.push_str(&format!("{pad}FILTER({})\n", expr_to_sparql(condition)?));
            Ok(())
        }
        Algebra::Minus { left, right } => {
            write_pattern(left, out, indent)?;
            out.push_str(&format!("{pad}MINUS {{\n"));
            write_pattern(right, out, indent + 1)?;
            out.push_str(&format!("{pad}}}\n"));
            Ok(())
        }
        Algebra::Graph { graph, pattern } => {
            out.push_str(&format!("{pad}GRAPH {} {{\n", term_to_sparql(graph)?));
            write_pattern(pattern, out, indent + 1)?;
            out.push_str(&format!("{pad}}}\n"));
            Ok(())
        }
        Algebra::Extend {
            pattern,
            variable,
            expr,
        } => {
            write_pattern(pattern, out, indent)?;
            out.push_str(&format!(
                "{pad}BIND({} AS {})\n",
                expr_to_sparql(expr)?,
                variable.with_prefix()
            ));
            Ok(())
        }
        Algebra::PropertyPath {
            subject,
            path,
            object,
        } => {
            out.push_str(&format!(
                "{pad}{} {} {} .\n",
                term_to_sparql(subject)?,
                path_to_sparql(path)?,
                term_to_sparql(object)?
            ));
            Ok(())
        }
        Algebra::Values {
            variables,
            bindings,
        } => {
            out.push_str(&pad);
            out.push_str("VALUES (");
            for (i, var) in variables.iter().enumerate() {
                if i > 0 {
                    out.push(' ');
                }
                out.push_str(&var.with_prefix());
            }
            out.push_str(") {\n");
            for binding in bindings {
                out.push_str(&pad);
                out.push_str("  (");
                for (i, var) in variables.iter().enumerate() {
                    if i > 0 {
                        out.push(' ');
                    }
                    match binding.get(var) {
                        Some(term) => out.push_str(&term_to_sparql(term)?),
                        None => out.push_str("UNDEF"),
                    }
                }
                out.push_str(")\n");
            }
            out.push_str(&pad);
            out.push_str("}\n");
            Ok(())
        }
        Algebra::Table | Algebra::Zero | Algebra::Empty => Ok(()),
        other => Err(anyhow!(
            "cannot serialize algebra construct for SERVICE federation: {other:?}"
        )),
    }
}

fn term_to_sparql(term: &Term) -> Result<String> {
    match term {
        Term::Variable(v) => Ok(v.with_prefix()),
        Term::Iri(iri) => Ok(format!("<{}>", iri.as_str())),
        Term::Literal(lit) => Ok(literal_to_sparql(lit)),
        Term::BlankNode(id) => Ok(format!("_:{id}")),
        Term::QuotedTriple(qt) => Ok(format!(
            "<< {} {} {} >>",
            term_to_sparql(&qt.subject)?,
            term_to_sparql(&qt.predicate)?,
            term_to_sparql(&qt.object)?
        )),
        Term::PropertyPath(p) => path_to_sparql(p),
    }
}

fn literal_to_sparql(lit: &Literal) -> String {
    let escaped = escape_literal(&lit.value);
    if let Some(lang) = &lit.language {
        format!("\"{escaped}\"@{lang}")
    } else if let Some(dt) = &lit.datatype {
        let dt_str = dt.as_str();
        if dt_str == "http://www.w3.org/2001/XMLSchema#string" {
            format!("\"{escaped}\"")
        } else {
            format!("\"{escaped}\"^^<{dt_str}>")
        }
    } else {
        format!("\"{escaped}\"")
    }
}

fn escape_literal(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

fn path_to_sparql(path: &PropertyPath) -> Result<String> {
    match path {
        PropertyPath::Iri(iri) => Ok(format!("<{}>", iri.as_str())),
        PropertyPath::Variable(v) => Ok(v.with_prefix()),
        PropertyPath::Inverse(inner) => Ok(format!("^{}", path_to_sparql(inner)?)),
        PropertyPath::Sequence(a, b) => {
            Ok(format!("({} / {})", path_to_sparql(a)?, path_to_sparql(b)?))
        }
        PropertyPath::Alternative(a, b) => {
            Ok(format!("({} | {})", path_to_sparql(a)?, path_to_sparql(b)?))
        }
        PropertyPath::ZeroOrMore(inner) => Ok(format!("{}*", path_to_sparql(inner)?)),
        PropertyPath::OneOrMore(inner) => Ok(format!("{}+", path_to_sparql(inner)?)),
        PropertyPath::ZeroOrOne(inner) => Ok(format!("{}?", path_to_sparql(inner)?)),
        PropertyPath::NegatedPropertySet(paths) => {
            let mut parts = Vec::with_capacity(paths.len());
            for p in paths {
                parts.push(path_to_sparql(p)?);
            }
            Ok(format!("!({})", parts.join(" | ")))
        }
    }
}

fn expr_to_sparql(expr: &Expression) -> Result<String> {
    match expr {
        Expression::Variable(v) => Ok(v.with_prefix()),
        Expression::Literal(lit) => Ok(literal_to_sparql(lit)),
        Expression::Iri(iri) => Ok(format!("<{}>", iri.as_str())),
        Expression::Bound(v) => Ok(format!("bound({})", v.with_prefix())),
        Expression::Binary { op, left, right } => {
            let l = expr_to_sparql(left)?;
            let r = expr_to_sparql(right)?;
            match op {
                BinaryOperator::SameTerm => Ok(format!("sameTerm({l}, {r})")),
                BinaryOperator::In => Ok(format!("{l} IN ({r})")),
                BinaryOperator::NotIn => Ok(format!("{l} NOT IN ({r})")),
                _ => Ok(format!("({l} {} {r})", binary_op_symbol(op))),
            }
        }
        Expression::Unary { op, operand } => {
            let inner = expr_to_sparql(operand)?;
            match op {
                UnaryOperator::Not => Ok(format!("!({inner})")),
                UnaryOperator::Plus => Ok(format!("(+{inner})")),
                UnaryOperator::Minus => Ok(format!("(-{inner})")),
                UnaryOperator::IsIri => Ok(format!("isIRI({inner})")),
                UnaryOperator::IsBlank => Ok(format!("isBlank({inner})")),
                UnaryOperator::IsLiteral => Ok(format!("isLiteral({inner})")),
                UnaryOperator::IsNumeric => Ok(format!("isNumeric({inner})")),
            }
        }
        Expression::Conditional {
            condition,
            then_expr,
            else_expr,
        } => Ok(format!(
            "IF({}, {}, {})",
            expr_to_sparql(condition)?,
            expr_to_sparql(then_expr)?,
            expr_to_sparql(else_expr)?
        )),
        Expression::Function { name, args } => {
            let mut parts = Vec::with_capacity(args.len());
            for a in args {
                parts.push(expr_to_sparql(a)?);
            }
            Ok(format!("{name}({})", parts.join(", ")))
        }
        Expression::Exists(_) | Expression::NotExists(_) => Err(anyhow!(
            "EXISTS/NOT EXISTS cannot be serialized for SERVICE federation"
        )),
    }
}

fn binary_op_symbol(op: &BinaryOperator) -> &'static str {
    match op {
        BinaryOperator::Add => "+",
        BinaryOperator::Subtract => "-",
        BinaryOperator::Multiply => "*",
        BinaryOperator::Divide => "/",
        BinaryOperator::Equal => "=",
        BinaryOperator::NotEqual => "!=",
        BinaryOperator::Less => "<",
        BinaryOperator::LessEqual => "<=",
        BinaryOperator::Greater => ">",
        BinaryOperator::GreaterEqual => ">=",
        BinaryOperator::And => "&&",
        BinaryOperator::Or => "||",
        BinaryOperator::SameTerm => "sameTerm",
        BinaryOperator::In => "IN",
        BinaryOperator::NotIn => "NOT IN",
    }
}

// ---------------------------------------------------------------------------
// SPARQL 1.1 JSON results parsing
// ---------------------------------------------------------------------------

/// Parse a SPARQL 1.1 JSON results document into a [`Solution`].
pub fn parse_sparql_json_results(json: &serde_json::Value) -> Result<Solution> {
    use crate::algebra::{Binding, Variable};

    // ASK responses carry a top-level "boolean"; represent true as a single
    // empty binding and false as no bindings.
    if let Some(b) = json.get("boolean").and_then(|v| v.as_bool()) {
        return Ok(if b { vec![Binding::new()] } else { Vec::new() });
    }

    let bindings = json
        .get("results")
        .and_then(|r| r.get("bindings"))
        .and_then(|b| b.as_array())
        .ok_or_else(|| anyhow!("invalid SPARQL JSON results: missing results.bindings array"))?;

    let mut solutions = Vec::with_capacity(bindings.len());
    for row in bindings {
        let mut binding = Binding::new();
        if let Some(obj) = row.as_object() {
            for (var_name, value) in obj {
                let variable = Variable::new_unchecked(var_name.clone());
                binding.insert(variable, parse_json_term(value)?);
            }
        }
        solutions.push(binding);
    }
    Ok(solutions)
}

fn parse_json_term(value: &serde_json::Value) -> Result<Term> {
    use crate::algebra::Literal as AlgebraLiteral;
    use oxirs_core::model::NamedNode;

    let term_type = value
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("SPARQL JSON term missing 'type'"))?;
    let term_value = value
        .get("value")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("SPARQL JSON term missing 'value'"))?;

    match term_type {
        "uri" => Ok(Term::Iri(NamedNode::new_unchecked(term_value))),
        "literal" | "typed-literal" => {
            let datatype = value.get("datatype").and_then(|v| v.as_str());
            let language = value.get("xml:lang").and_then(|v| v.as_str());
            if let Some(lang) = language {
                Ok(Term::Literal(AlgebraLiteral {
                    value: term_value.to_string(),
                    language: Some(lang.to_string()),
                    datatype: None,
                }))
            } else if let Some(dt) = datatype {
                Ok(Term::Literal(AlgebraLiteral {
                    value: term_value.to_string(),
                    language: None,
                    datatype: Some(NamedNode::new_unchecked(dt)),
                }))
            } else {
                Ok(Term::Literal(AlgebraLiteral {
                    value: term_value.to_string(),
                    language: None,
                    datatype: None,
                }))
            }
        }
        "bnode" => Ok(Term::BlankNode(term_value.to_string())),
        other => Err(anyhow!("unknown SPARQL JSON term type: {other}")),
    }
}

// ---------------------------------------------------------------------------
// HTTP transport (requires a Tokio runtime, gated on the `parallel` feature)
// ---------------------------------------------------------------------------

/// POST a SPARQL query to `endpoint` and parse the JSON results.
///
/// The request runs on a dedicated worker thread with its own current-thread
/// Tokio runtime, so this is safe to call from any context (including from
/// inside another runtime) without a nested-runtime panic.
#[cfg(feature = "parallel")]
fn http_post_sparql(endpoint: &str, query: &str, timeout: Duration) -> Result<Solution> {
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| anyhow!("failed to build Tokio runtime for SERVICE: {e}"))?;
                rt.block_on(async {
                    let client = reqwest::Client::builder()
                        .timeout(timeout)
                        .build()
                        .map_err(|e| anyhow!("failed to build HTTP client: {e}"))?;
                    let response = client
                        .post(endpoint)
                        .header("Accept", "application/sparql-results+json")
                        .header("Content-Type", "application/sparql-query")
                        .body(query.to_string())
                        .send()
                        .await
                        .map_err(|e| anyhow!("SERVICE request to <{endpoint}> failed: {e}"))?;
                    let status = response.status();
                    if !status.is_success() {
                        let body = response.text().await.unwrap_or_default();
                        return Err(anyhow!(
                            "SERVICE endpoint <{endpoint}> returned HTTP {status}: {body}"
                        ));
                    }
                    let json: serde_json::Value = response.json().await.map_err(|e| {
                        anyhow!("SERVICE endpoint <{endpoint}> returned invalid JSON: {e}")
                    })?;
                    parse_sparql_json_results(&json)
                })
            })
            .join()
            .map_err(|_| anyhow!("SERVICE federation worker thread panicked"))?
    })
}

#[cfg(not(feature = "parallel"))]
fn http_post_sparql(endpoint: &str, _query: &str, _timeout: Duration) -> Result<Solution> {
    Err(anyhow!(
        "SERVICE federation to <{endpoint}> requires the 'parallel' feature (Tokio runtime); \
         rebuild with --features parallel to enable remote SPARQL calls"
    ))
}

/// Fetch a document over HTTP for SPARQL `LOAD`, returning the body text and
/// the reported `Content-Type` (if any). Requires the `parallel` feature.
#[cfg(feature = "parallel")]
pub fn http_get_document(url: &str, timeout: Duration) -> Result<(String, Option<String>)> {
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| anyhow!("failed to build Tokio runtime for LOAD: {e}"))?;
                rt.block_on(async {
                    let client = reqwest::Client::builder()
                        .timeout(timeout)
                        .build()
                        .map_err(|e| anyhow!("failed to build HTTP client: {e}"))?;
                    let response = client
                        .get(url)
                        .header(
                            "Accept",
                            "text/turtle, application/n-triples, application/rdf+xml;q=0.9, \
                             application/ld+json;q=0.8, */*;q=0.5",
                        )
                        .send()
                        .await
                        .map_err(|e| anyhow!("LOAD request to <{url}> failed: {e}"))?;
                    let status = response.status();
                    if !status.is_success() {
                        let body = response.text().await.unwrap_or_default();
                        return Err(anyhow!("LOAD <{url}> returned HTTP {status}: {body}"));
                    }
                    let content_type = response
                        .headers()
                        .get(reqwest::header::CONTENT_TYPE)
                        .and_then(|v| v.to_str().ok())
                        .map(|s| s.to_string());
                    let body = response
                        .text()
                        .await
                        .map_err(|e| anyhow!("LOAD <{url}> body read failed: {e}"))?;
                    Ok((body, content_type))
                })
            })
            .join()
            .map_err(|_| anyhow!("LOAD worker thread panicked"))?
    })
}

#[cfg(not(feature = "parallel"))]
pub fn http_get_document(url: &str, _timeout: Duration) -> Result<(String, Option<String>)> {
    Err(anyhow!(
        "LOAD <{url}> requires the 'parallel' feature (Tokio runtime); \
         rebuild with --features parallel to enable remote document fetching"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Literal, TriplePattern, Variable};
    use oxirs_core::model::NamedNode;

    fn var(name: &str) -> Term {
        Term::Variable(Variable::new_unchecked(name))
    }

    fn iri(s: &str) -> Term {
        Term::Iri(NamedNode::new_unchecked(s))
    }

    #[test]
    fn service_bgp_serializes_to_select() {
        let bgp = Algebra::Bgp(vec![TriplePattern {
            subject: var("s"),
            predicate: iri("http://example.org/p"),
            object: var("o"),
        }]);
        let q = algebra_to_select_query(&bgp).expect("serialize");
        assert!(q.contains("SELECT * WHERE"));
        assert!(q.contains("?s <http://example.org/p> ?o ."));
    }

    #[test]
    fn service_typed_literal_roundtrips_to_query() {
        let bgp = Algebra::Bgp(vec![TriplePattern {
            subject: var("s"),
            predicate: iri("http://example.org/age"),
            object: Term::Literal(Literal {
                value: "25".to_string(),
                language: None,
                datatype: Some(NamedNode::new_unchecked(
                    "http://www.w3.org/2001/XMLSchema#integer",
                )),
            }),
        }]);
        let q = algebra_to_select_query(&bgp).expect("serialize");
        assert!(q.contains("\"25\"^^<http://www.w3.org/2001/XMLSchema#integer>"));
    }

    #[test]
    fn service_parse_json_bindings() {
        let json = serde_json::json!({
            "head": {"vars": ["s", "o"]},
            "results": {"bindings": [
                {"s": {"type": "uri", "value": "http://example.org/a"},
                 "o": {"type": "literal", "value": "hi", "xml:lang": "en"}}
            ]}
        });
        let sol = parse_sparql_json_results(&json).expect("parse");
        assert_eq!(sol.len(), 1);
        let binding = &sol[0];
        let s = binding.get(&Variable::new_unchecked("s")).expect("s bound");
        assert!(matches!(s, Term::Iri(_)));
        let o = binding.get(&Variable::new_unchecked("o")).expect("o bound");
        match o {
            Term::Literal(lit) => {
                assert_eq!(lit.value, "hi");
                assert_eq!(lit.language.as_deref(), Some("en"));
            }
            _ => panic!("expected literal"),
        }
    }

    #[test]
    fn service_parse_ask_boolean() {
        let json = serde_json::json!({"head": {}, "boolean": true});
        let sol = parse_sparql_json_results(&json).expect("parse");
        assert_eq!(sol.len(), 1);

        let json_false = serde_json::json!({"head": {}, "boolean": false});
        let sol_false = parse_sparql_json_results(&json_false).expect("parse");
        assert!(sol_false.is_empty());
    }

    #[test]
    fn service_endpoint_must_be_iri() {
        assert!(endpoint_iri(&var("e")).is_err());
        assert_eq!(
            endpoint_iri(&iri("http://example.org/sparql")).expect("iri"),
            "http://example.org/sparql"
        );
    }
}
