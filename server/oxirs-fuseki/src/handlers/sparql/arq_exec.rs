//! SPARQL query execution wired to the real oxirs-arq engine.
//!
//! The legacy path executed queries through `oxirs_core::query::QueryEngine`, a
//! deliberately simplified engine whose parser only extracts WHERE triple
//! patterns — it silently drops `FILTER`, never parses `LIMIT`/`OFFSET`/`ORDER
//! BY`/aggregates, and on any failure the fuseki handler returned `200 OK` with
//! an empty result. That produced *silent wrong answers* (an existing triple's
//! `ASK` returning `false`, `FILTER` ignored, `LIMIT` returning nothing).
//!
//! This module runs SELECT/ASK queries through `oxirs-arq`'s real algebra
//! executor against the live store (via [`StoreRefDataset`], zero data copy),
//! so joins, `FILTER`, `OPTIONAL`, `UNION`, `BIND`, `DISTINCT`, `ORDER BY` and
//! `LIMIT`/`OFFSET` are actually evaluated. Anything that cannot be executed
//! correctly returns an HTTP error — never a silent empty result.
//!
//! Scope / honest limitations (returned as errors, not wrong data):
//! * `GRAPH` (named graphs) and `SERVICE` (federation) are rejected — the arq
//!   `Dataset` abstraction reads only the default graph, so honouring them
//!   would silently return default-graph data.
//! * Aggregate projections (`SELECT (COUNT(*) AS ?n)`) are handled by
//!   [`aggregate`](Self) below when present; other unsupported forms
//!   (subqueries, property paths the store can't resolve) surface as errors.
//! * CONSTRUCT/DESCRIBE are intentionally NOT routed here (see
//!   `core::execute_sparql_query`) — arq's parser cannot yet handle valid forms
//!   such as `DESCRIBE <iri>` (no WHERE) or `CONSTRUCT WHERE { … }` shorthand,
//!   so they keep the legacy store path to avoid rejecting valid queries.

use crate::error::{FusekiError, FusekiResult};
use crate::handlers::sparql::core::QueryResult;
use crate::store::Store;
use oxirs_arq::algebra::{Aggregate, Algebra, Expression, GroupCondition, Term as ArqTerm};
use oxirs_arq::executor::{ExecutionStrategy, QueryExecutor, StoreRefDataset};
use oxirs_arq::query::{parse_query, Query, QueryType};
use oxirs_core::model::Variable;
use std::collections::HashMap;

/// Execute a SELECT or ASK query against the store's default graph via oxirs-arq.
///
/// Returns a fuseki [`QueryResult`] on success. A parse failure yields an HTTP
/// 400 (`query_parsing`); an unsupported construct yields an explicit error; an
/// execution failure yields an HTTP 500 (`query_execution`). It never returns a
/// successful-but-empty result to paper over a failure.
pub fn execute_query(query_str: &str, store: &Store) -> FusekiResult<QueryResult> {
    // Aggregate projections (`SELECT (COUNT(*) AS ?n) …`) do not parse through
    // arq's parser at all (it breaks on `(`), so detect and route them first.
    if let Some(spec) = aggregate::detect(query_str) {
        return aggregate::execute(query_str, store, spec);
    }

    let parsed = parse_query(query_str)
        .map_err(|e| FusekiError::query_parsing(format!("SPARQL parse error: {e}")))?;

    match parsed.query_type {
        QueryType::Select | QueryType::Ask => {}
        QueryType::Construct | QueryType::Describe => {
            return Err(FusekiError::query_execution(
                "CONSTRUCT/DESCRIBE are not routed through the arq executor",
            ));
        }
    }

    reject_unsupported(&parsed.where_clause)?;
    let algebra = build_select_algebra(&parsed);

    let solution = run(store, &algebra)?;

    Ok(match parsed.query_type {
        QueryType::Ask => ask_result(!solution.is_empty()),
        _ => select_result(solution),
    })
}

/// Acquire the default-graph store under a read guard and execute `algebra`
/// synchronously via a `Serial`-strategy arq executor.
///
/// The read guard is held only for the duration of this synchronous call (no
/// `.await` occurs while it is held). `Serial` is forced because the adaptive/
/// parallel strategies do not reliably evaluate `Group` (aggregation).
fn run(store: &Store, algebra: &Algebra) -> FusekiResult<oxirs_arq::algebra::Solution> {
    let arc = store.get_dataset(None)?;
    let guard = arc
        .read()
        .map_err(|e| FusekiError::store(format!("failed to acquire store read lock: {e}")))?;
    let dataset = StoreRefDataset::new(&*guard);
    let mut executor = QueryExecutor::new();
    executor.set_strategy(ExecutionStrategy::Serial);
    let (solution, _stats) = executor
        .execute(algebra, &dataset)
        .map_err(|e| FusekiError::query_execution(format!("query execution failed: {e}")))?;
    Ok(solution)
}

/// Wrap a parsed SELECT query's WHERE algebra with its solution modifiers in
/// SPARQL evaluation order: WHERE → ORDER BY → PROJECT → DISTINCT → SLICE.
///
/// `ORDER BY` is applied before projection so it can reference variables that
/// are not in the SELECT list; `SLICE` (LIMIT/OFFSET) is outermost.
fn build_select_algebra(q: &Query) -> Algebra {
    let mut alg = q.where_clause.clone();
    if q.query_type == QueryType::Ask {
        // ASK ignores projection/order/slice — the boolean is just "any match".
        return alg;
    }
    if !q.order_by.is_empty() {
        alg = Algebra::OrderBy {
            pattern: Box::new(alg),
            conditions: q.order_by.clone(),
        };
    }
    if !q.select_variables.is_empty() {
        // Empty select_variables == `SELECT *` (project nothing / keep all vars).
        alg = Algebra::Project {
            pattern: Box::new(alg),
            variables: q.select_variables.clone(),
        };
    }
    if q.distinct {
        alg = Algebra::Distinct {
            pattern: Box::new(alg),
        };
    }
    if q.limit.is_some() || q.offset.is_some() {
        alg = Algebra::Slice {
            pattern: Box::new(alg),
            offset: q.offset,
            limit: q.limit,
        };
    }
    alg
}

/// Reject algebra nodes the arq `Dataset` abstraction cannot honour, so they
/// return an explicit error instead of silently reading the default graph.
fn reject_unsupported(alg: &Algebra) -> FusekiResult<()> {
    match alg {
        Algebra::Graph { .. } => Err(FusekiError::query_parsing(
            "Named-graph (GRAPH) queries are not supported by this endpoint",
        )),
        Algebra::Service { .. } => Err(FusekiError::query_parsing(
            "SPARQL SERVICE (federation) is not supported by this endpoint",
        )),
        Algebra::Join { left, right }
        | Algebra::LeftJoin { left, right, .. }
        | Algebra::Union { left, right }
        | Algebra::Minus { left, right } => {
            reject_unsupported(left)?;
            reject_unsupported(right)
        }
        Algebra::Filter { pattern, .. }
        | Algebra::Extend { pattern, .. }
        | Algebra::Project { pattern, .. }
        | Algebra::Distinct { pattern }
        | Algebra::Reduced { pattern }
        | Algebra::Slice { pattern, .. }
        | Algebra::OrderBy { pattern, .. }
        | Algebra::Group { pattern, .. }
        | Algebra::Having { pattern, .. } => reject_unsupported(pattern),
        _ => Ok(()),
    }
}

/// Build a fuseki `QueryResult` for an ASK boolean.
fn ask_result(value: bool) -> QueryResult {
    QueryResult {
        query_type: "ASK".to_string(),
        execution_time_ms: 0,
        result_count: Some(1),
        bindings: None,
        boolean: Some(value),
        construct_graph: None,
        describe_graph: None,
    }
}

/// Build a fuseki `QueryResult` for SELECT bindings.
fn select_result(solution: oxirs_arq::algebra::Solution) -> QueryResult {
    let bindings: Vec<HashMap<String, serde_json::Value>> =
        solution.iter().map(binding_to_json).collect();
    QueryResult {
        query_type: "SELECT".to_string(),
        execution_time_ms: 0,
        result_count: Some(bindings.len()),
        bindings: Some(bindings),
        boolean: None,
        construct_graph: None,
        describe_graph: None,
    }
}

/// Convert one arq binding (`Variable -> Term`) to the SPARQL Results JSON row
/// shape (`var name -> {type,value,...}`).
fn binding_to_json(binding: &HashMap<Variable, ArqTerm>) -> HashMap<String, serde_json::Value> {
    binding
        .iter()
        .map(|(var, term)| (var.name().to_string(), term_to_json(term)))
        .collect()
}

/// Convert an arq `Term` to a SPARQL Query Results JSON term object.
fn term_to_json(term: &ArqTerm) -> serde_json::Value {
    match term {
        ArqTerm::Iri(iri) => serde_json::json!({"type": "uri", "value": iri.as_str()}),
        ArqTerm::BlankNode(b) => serde_json::json!({"type": "bnode", "value": b}),
        ArqTerm::Literal(literal) => {
            let mut v = serde_json::json!({"type": "literal", "value": literal.value});
            if let Some(lang) = &literal.language {
                v["xml:lang"] = serde_json::Value::String(lang.clone());
            } else if let Some(dt) = &literal.datatype {
                if dt.as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                    v["datatype"] = serde_json::Value::String(dt.as_str().to_string());
                }
            }
            v
        }
        ArqTerm::Variable(var) => {
            serde_json::json!({"type": "literal", "value": format!("?{}", var.name())})
        }
        other => serde_json::json!({"type": "literal", "value": format!("{other:?}")}),
    }
}

/// Aggregate-projection handling.
///
/// arq's parser cannot parse `SELECT (COUNT(*) AS ?n) …` (it breaks on `(`), yet
/// the executor fully evaluates `Algebra::Group`. This submodule extracts the
/// aggregate projection from the raw query, re-parses a sanitized `SELECT *`
/// form to obtain the WHERE pattern + GROUP BY / modifiers from arq, then
/// assembles `Algebra::Group` by hand and executes it. Only the standard
/// aggregate functions over `*` or a single variable are supported; anything
/// else returns an explicit error rather than a wrong answer.
mod aggregate {
    use super::*;
    use oxirs_arq::query::parse_query;

    /// A parsed aggregate projection: the ordered projection (group vars and
    /// aggregate result vars) plus the aggregate specs.
    pub struct Spec {
        /// Projected non-aggregate variables (must equal the GROUP BY keys).
        pub group_vars: Vec<Variable>,
        /// `(result_var, aggregate)` pairs, in projection order.
        pub aggregates: Vec<(Variable, Aggregate)>,
        /// Full projection order (group var names then aggregate result names).
        pub projection: Vec<Variable>,
    }

    /// Detect an aggregate projection and parse it. Returns `None` when the
    /// SELECT projection contains no `(… AS ?v)` item (i.e. a plain query that
    /// arq's parser can handle directly).
    pub fn detect(query: &str) -> Option<Spec> {
        let upper = query.to_uppercase();
        if !upper.trim_start().starts_with("SELECT") {
            return None;
        }
        // The projection is the text between SELECT and the first top-level WHERE.
        let (proj_text, _rest) = split_select_projection(query)?;
        if !proj_text.contains('(') {
            return None;
        }
        parse_projection(proj_text)
    }

    /// Execute an aggregate query. `spec` is the parsed projection; the WHERE
    /// pattern, GROUP BY keys and modifiers are obtained by re-parsing a
    /// sanitized `SELECT *` form through arq.
    pub fn execute(query: &str, store: &Store, spec: Spec) -> FusekiResult<QueryResult> {
        let sanitized = sanitize_to_select_star(query).ok_or_else(|| {
            FusekiError::query_parsing("Could not parse aggregate SELECT projection")
        })?;
        let parsed = parse_query(&sanitized).map_err(|e| {
            FusekiError::query_parsing(format!("SPARQL parse error (aggregate query): {e}"))
        })?;
        super::reject_unsupported(&parsed.where_clause)?;

        // GROUP BY keys come from arq's parse; if the query has explicit
        // GROUP BY they must match the projected non-aggregate vars.
        let group_conditions: Vec<GroupCondition> = if parsed.group_by.is_empty() {
            spec.group_vars
                .iter()
                .map(|v| GroupCondition {
                    expr: Expression::Variable(v.clone()),
                    alias: None,
                })
                .collect()
        } else {
            parsed.group_by.clone()
        };

        let mut alg = Algebra::Group {
            pattern: Box::new(parsed.where_clause.clone()),
            variables: group_conditions,
            aggregates: spec.aggregates.clone(),
        };
        if !parsed.order_by.is_empty() {
            alg = Algebra::OrderBy {
                pattern: Box::new(alg),
                conditions: parsed.order_by.clone(),
            };
        }
        alg = Algebra::Project {
            pattern: Box::new(alg),
            variables: spec.projection.clone(),
        };
        if parsed.distinct {
            alg = Algebra::Distinct {
                pattern: Box::new(alg),
            };
        }
        if parsed.limit.is_some() || parsed.offset.is_some() {
            alg = Algebra::Slice {
                pattern: Box::new(alg),
                offset: parsed.offset,
                limit: parsed.limit,
            };
        }

        let solution = super::run(store, &alg)?;
        Ok(super::select_result(solution))
    }

    /// Return `(projection_text, rest_after_where)` splitting on the first
    /// top-level `WHERE` keyword (case-insensitive). `None` if no `WHERE`.
    fn split_select_projection(query: &str) -> Option<(&str, &str)> {
        let trimmed = query.trim_start();
        if trimmed.len() < 6 {
            return None;
        }
        // Skip the leading "SELECT" keyword (6 bytes, ASCII).
        let after_select = &trimmed[6..];
        let where_rel = find_keyword_ci(after_select, "WHERE")?;
        let proj = after_select[..where_rel].trim();
        let rest = &after_select[where_rel + 5..];
        Some((proj, rest))
    }

    /// Rewrite the SELECT projection to `*`, preserving a leading
    /// `DISTINCT`/`REDUCED`. Everything else (WHERE, GROUP BY, modifiers) is
    /// left intact so arq parses the pattern and modifiers faithfully.
    fn sanitize_to_select_star(query: &str) -> Option<String> {
        let trimmed = query.trim_start();
        if trimmed.len() < 6 {
            return None;
        }
        let after_select = &trimmed[6..];
        let where_rel = find_keyword_ci(after_select, "WHERE")?;
        let projection = &after_select[..where_rel];
        let proj_upper = projection.trim_start().to_uppercase();
        let modifier = if proj_upper.starts_with("DISTINCT") {
            "DISTINCT "
        } else if proj_upper.starts_with("REDUCED") {
            "REDUCED "
        } else {
            ""
        };
        Some(format!("SELECT {modifier}* {}", &after_select[where_rel..]))
    }

    /// Parse an aggregate projection into a [`Spec`]. Supported item forms:
    /// * a bare variable `?v` (a GROUP BY key), and
    /// * `(AGG([DISTINCT] (*|?v)) AS ?result)` where `AGG` is one of the
    ///   standard SPARQL aggregates.
    ///
    /// Returns `None` if any projection item is not one of those (caller then
    /// treats the query as unsupported → explicit error).
    fn parse_projection(proj_text: &str) -> Option<Spec> {
        let items = split_projection_items(proj_text);
        let mut group_vars = Vec::new();
        let mut aggregates = Vec::new();
        let mut projection = Vec::new();
        for item in items {
            let item = item.trim();
            if item.is_empty() || item == "*" {
                continue;
            }
            if let Some(var) = item.strip_prefix('?') {
                let v = Variable::new(var).ok()?;
                group_vars.push(v.clone());
                projection.push(v);
            } else if item.starts_with('(') {
                let (result_var, agg) = parse_aggregate_item(item)?;
                projection.push(result_var.clone());
                aggregates.push((result_var, agg));
            } else {
                return None;
            }
        }
        if aggregates.is_empty() {
            return None;
        }
        Some(Spec {
            group_vars,
            aggregates,
            projection,
        })
    }

    /// Parse a single `(AGG([DISTINCT] arg) AS ?result)` projection item.
    fn parse_aggregate_item(item: &str) -> Option<(Variable, Aggregate)> {
        let inner = item.strip_prefix('(')?.strip_suffix(')')?.trim();
        // Split on the top-level " AS " (case-insensitive).
        let as_pos = find_keyword_ci(inner, "AS")?;
        let (agg_text, alias_text) = (inner[..as_pos].trim(), inner[as_pos + 2..].trim());
        let result_var = Variable::new(alias_text.strip_prefix('?')?).ok()?;

        // agg_text = FUNC( [DISTINCT] arg )
        let open = agg_text.find('(')?;
        let func = agg_text[..open].trim().to_uppercase();
        let args = agg_text[open + 1..].strip_suffix(')')?.trim();
        let (distinct, arg) = match args.strip_prefix("DISTINCT ").or_else(|| {
            let up = args.to_uppercase();
            up.starts_with("DISTINCT ").then(|| &args[9..])
        }) {
            Some(rest) => (true, rest.trim()),
            None => (false, args),
        };

        let expr = if arg == "*" {
            None
        } else {
            // Only `*` or a single variable are supported as the aggregate arg;
            // anything else (`strip_prefix` returns None) makes this unsupported.
            let var = arg.strip_prefix('?')?;
            Some(Expression::Variable(Variable::new(var).ok()?))
        };

        let agg = match func.as_str() {
            "COUNT" => Aggregate::Count { distinct, expr },
            "SUM" => Aggregate::Sum {
                distinct,
                expr: expr?,
            },
            "AVG" => Aggregate::Avg {
                distinct,
                expr: expr?,
            },
            "MIN" => Aggregate::Min {
                distinct,
                expr: expr?,
            },
            "MAX" => Aggregate::Max {
                distinct,
                expr: expr?,
            },
            "SAMPLE" => Aggregate::Sample {
                distinct,
                expr: expr?,
            },
            "GROUP_CONCAT" => Aggregate::GroupConcat {
                distinct,
                expr: expr?,
                separator: None,
            },
            _ => return None,
        };
        Some((result_var, agg))
    }

    /// Split a SELECT projection into items, keeping `(… )` groups intact.
    fn split_projection_items(proj: &str) -> Vec<String> {
        let mut items = Vec::new();
        let mut depth = 0i32;
        let mut current = String::new();
        for ch in proj.chars() {
            match ch {
                '(' => {
                    depth += 1;
                    current.push(ch);
                }
                ')' => {
                    depth -= 1;
                    current.push(ch);
                }
                c if c.is_whitespace() && depth == 0 => {
                    if !current.trim().is_empty() {
                        items.push(current.trim().to_string());
                    }
                    current.clear();
                }
                _ => current.push(ch),
            }
        }
        if !current.trim().is_empty() {
            items.push(current.trim().to_string());
        }
        items
    }

    /// Find the byte offset of a whitespace-delimited keyword (case-insensitive)
    /// at top level (not inside `<…>` or quotes). Returns the offset of the
    /// keyword start, or `None`.
    fn find_keyword_ci(haystack: &str, keyword: &str) -> Option<usize> {
        let kw_len = keyword.len();
        let bytes = haystack.as_bytes();
        let mut in_iri = false;
        let mut in_dquote = false;
        let mut in_squote = false;
        // Iterate by char so `i` is always a valid char boundary (queries may
        // contain multi-byte characters, e.g. in a malformed projection).
        for (i, ch) in haystack.char_indices() {
            match ch {
                '<' if !in_dquote && !in_squote => in_iri = true,
                '>' if in_iri => in_iri = false,
                '"' if !in_squote && !in_iri => in_dquote = !in_dquote,
                '\'' if !in_dquote && !in_iri => in_squote = !in_squote,
                _ if !in_iri && !in_dquote && !in_squote && ch.is_ascii_alphabetic() => {
                    let end = i + kw_len;
                    if end <= bytes.len()
                        && haystack.is_char_boundary(end)
                        && haystack[i..end].eq_ignore_ascii_case(keyword)
                        && (i == 0 || !bytes[i - 1].is_ascii_alphanumeric())
                        && (end == bytes.len() || !bytes[end].is_ascii_alphanumeric())
                    {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
        None
    }
}
