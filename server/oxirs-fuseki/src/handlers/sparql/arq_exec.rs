//! SPARQL query execution wired to the real oxirs-arq engine.
//!
//! Every query form — SELECT, ASK, CONSTRUCT and DESCRIBE — is parsed by the
//! oxirs-arq parser and evaluated by its algebra executor against the live
//! store (via [`StoreRefDataset`], zero data copy). Joins, `FILTER`,
//! `OPTIONAL`, `UNION`, `MINUS`, `BIND`, aggregation (`GROUP BY` / `HAVING`),
//! `ORDER BY`, `DISTINCT` and `LIMIT`/`OFFSET` are all actually evaluated.
//!
//! The `FROM` / `FROM NAMED` dataset clause is honoured by wrapping the base
//! dataset in a [`with_dataset_clause`] view before execution (an empty clause
//! is a transparent passthrough), so named-graph (`GRAPH`) scoping and dataset
//! construction produce correct answers rather than default-graph data.
//!
//! There is no silent-empty fallback: a parse failure surfaces as HTTP 400, an
//! execution failure as HTTP 500, and a genuinely unexecutable construct as an
//! explicit typed error. `SERVICE` federation and `GRAPH` scoping are executed
//! for real by the engine — they are no longer rejected, and the dataset no
//! longer unions every named graph into a plain BGP.

use crate::error::{FusekiError, FusekiResult};
use crate::handlers::sparql::core::{serialize_triples_to_turtle, QueryResult};
use crate::store::Store;
use oxirs_arq::algebra::{
    Aggregate, Algebra, Expression, Literal as ArqLiteral, Solution, Term as ArqTerm,
    Triple as ArqTriple, Variable,
};
use oxirs_arq::executor::{with_dataset_clause, ExecutionStrategy, QueryExecutor, StoreRefDataset};
use oxirs_arq::query::{
    parse_query, DatasetClause, DescribeTarget, ProjectionItem, Query, QueryType,
};
use oxirs_arq::{describe, instantiate_construct};
use oxirs_core::model::{
    BlankNode, Literal as CoreLiteral, Object, Predicate, Subject, Triple as CoreTriple,
};
use std::collections::{HashMap, HashSet};

/// Parse a SPARQL query string and execute it against the store.
///
/// This is a convenience wrapper that parses once and dispatches on the parsed
/// query form. The fuseki handler ([`crate::handlers::sparql::core`]) parses
/// the query itself for routing and calls the form-specific entry points
/// ([`execute_select_or_ask`], [`execute_construct`], [`execute_describe`])
/// directly, so this wrapper is provided for standalone / test callers.
///
/// A parse failure yields HTTP 400 (`query_parsing`); an unsupported construct
/// yields an explicit typed error; an execution failure yields HTTP 500
/// (`query_execution`). It never returns a successful-but-empty result to paper
/// over a failure.
pub fn execute_query(query_str: &str, store: &Store) -> FusekiResult<QueryResult> {
    let parsed = parse_query(query_str)
        .map_err(|e| FusekiError::query_parsing(format!("SPARQL parse error: {e}")))?;
    dispatch(&parsed, store)
}

/// Dispatch a parsed query to the form-specific executor.
pub fn dispatch(query: &Query, store: &Store) -> FusekiResult<QueryResult> {
    match query.query_type {
        QueryType::Select | QueryType::Ask => execute_select_or_ask(query, store),
        QueryType::Construct => execute_construct(query, store),
        QueryType::Describe => execute_describe(query, store),
    }
}

/// Execute a parsed SELECT or ASK query.
///
/// SELECT builds the full solution-modifier stack natively from the parsed
/// query (grouping/aggregation, HAVING, projected expressions, ORDER BY,
/// projection, DISTINCT and slicing); ASK reduces to "any match".
pub fn execute_select_or_ask(query: &Query, store: &Store) -> FusekiResult<QueryResult> {
    let algebra = build_select_algebra(query)?;
    let solution = run(store, &query.dataset, &algebra)?;
    match query.query_type {
        QueryType::Ask => Ok(ask_result(!solution.is_empty())),
        _ => select_result(solution),
    }
}

/// Execute a parsed CONSTRUCT query.
///
/// Runs the WHERE pattern (with any ORDER BY / LIMIT / OFFSET applied to the
/// solution sequence) and instantiates the CONSTRUCT template per row. An empty
/// template — whether written explicitly (`CONSTRUCT {}`) or produced by an
/// empty `CONSTRUCT WHERE {}` shorthand — is a 400, never a silent empty graph.
pub fn execute_construct(query: &Query, store: &Store) -> FusekiResult<QueryResult> {
    if query.construct_template.is_empty() {
        return Err(FusekiError::query_parsing(
            "CONSTRUCT template is empty: there is nothing to construct",
        ));
    }
    let algebra = build_graph_where_algebra(query);
    let solution = run(store, &query.dataset, &algebra)?;
    // The oxirs-arq engine's `instantiate_construct` accepts path-encoded
    // (length-one `PropertyPath::Iri`/`Variable`) template predicates natively,
    // so the template is passed through unchanged — no caller-side normalization.
    let triples = instantiate_construct(&query.construct_template, &solution).map_err(|e| {
        FusekiError::query_execution(format!("CONSTRUCT instantiation failed: {e}"))
    })?;
    let graph = serialize_arq_graph(&triples);
    Ok(construct_result(graph, triples.len()))
}

/// Execute a parsed DESCRIBE query.
///
/// Resolves the described-node set from the explicit `DESCRIBE` targets (IRIs
/// and variables) plus, for `DESCRIBE *`, every variable in scope of the WHERE
/// solution. `DESCRIBE <iri>` with no WHERE describes the IRIs directly against
/// an empty solution (a plain CBD lookup); `DESCRIBE *` with no WHERE has
/// nothing in scope and is a 400. The resulting Concise Bounded Description is
/// serialized like CONSTRUCT.
pub fn execute_describe(query: &Query, store: &Store) -> FusekiResult<QueryResult> {
    // `Algebra::Zero` is the parser's default when no WHERE block is present;
    // any real WHERE parses to a BGP/Table/... instead.
    let has_where = !matches!(query.where_clause, Algebra::Zero);

    if query.describe_all && !has_where {
        return Err(FusekiError::query_parsing(
            "DESCRIBE * requires a WHERE clause: there is nothing in scope to describe",
        ));
    }

    // Split explicit targets into concrete IRI terms and variables.
    let mut targets: Vec<ArqTerm> = Vec::new();
    let mut target_vars: Vec<Variable> = Vec::new();
    for target in &query.describe_targets {
        match target {
            DescribeTarget::Iri(iri) => targets.push(ArqTerm::Iri(iri.clone())),
            DescribeTarget::Variable(var) => target_vars.push(var.clone()),
        }
    }

    // Hold the store guard for the whole synchronous describe: the executor and
    // the CBD lookup both read through the same dataset view.
    let arc = store.get_dataset(None)?;
    let guard = arc
        .read()
        .map_err(|e| FusekiError::store(format!("failed to acquire store read lock: {e}")))?;
    let base = StoreRefDataset::new(&*guard);
    let view = with_dataset_clause(&base, &query.dataset);

    let solution: Solution = if has_where {
        let algebra = build_graph_where_algebra(query);
        let mut executor = QueryExecutor::new();
        executor.set_strategy(ExecutionStrategy::Serial);
        let (solution, _stats) = executor
            .execute(&algebra, &view)
            .map_err(|e| FusekiError::query_execution(format!("DESCRIBE WHERE failed: {e}")))?;
        solution
    } else {
        Vec::new()
    };

    // DESCRIBE * describes every variable bound anywhere in the solution.
    if query.describe_all {
        let mut seen: HashSet<Variable> = HashSet::new();
        for row in &solution {
            for var in row.keys() {
                if seen.insert(var.clone()) {
                    target_vars.push(var.clone());
                }
            }
        }
    }

    let triples = describe(&targets, &target_vars, &solution, &view)
        .map_err(|e| FusekiError::query_execution(format!("DESCRIBE failed: {e}")))?;
    let graph = serialize_arq_graph(&triples);
    Ok(describe_result(graph, triples.len()))
}

/// Build the dataset view for `clause` over the store's default dataset and
/// execute `algebra` synchronously via a `Serial`-strategy arq executor.
///
/// The read guard is held only for the duration of this synchronous call (no
/// `.await` occurs while it is held). `Serial` is forced because the adaptive/
/// parallel strategies do not reliably evaluate `Group` (aggregation). An empty
/// `clause` makes the view a transparent passthrough, so wrapping is always
/// safe.
fn run(store: &Store, clause: &DatasetClause, algebra: &Algebra) -> FusekiResult<Solution> {
    let arc = store.get_dataset(None)?;
    let guard = arc
        .read()
        .map_err(|e| FusekiError::store(format!("failed to acquire store read lock: {e}")))?;
    let base = StoreRefDataset::new(&*guard);
    let view = with_dataset_clause(&base, clause);
    let mut executor = QueryExecutor::new();
    executor.set_strategy(ExecutionStrategy::Serial);
    let (solution, _stats) = executor
        .execute(algebra, &view)
        .map_err(|e| FusekiError::query_execution(format!("query execution failed: {e}")))?;
    Ok(solution)
}

/// Build the full SELECT solution-modifier algebra natively from the parsed
/// query.
///
/// Evaluation order (SPARQL 1.1 §18.2.4): WHERE → Group (grouping/aggregation)
/// → Having → Extend (projected `(expr AS ?v)`) → OrderBy → Project → Distinct
/// → Slice. Grouping is introduced when the projection has aggregate items or
/// the query has an explicit `GROUP BY`; an aggregate with no `GROUP BY` is the
/// implicit single group (`variables: []`), and a `GROUP BY` with no aggregate
/// is a plain grouping. ORDER BY is applied before projection so it may
/// reference non-projected variables and aggregate results.
///
/// ASK ignores projection/order/slice — the boolean is just "any match".
fn build_select_algebra(query: &Query) -> FusekiResult<Algebra> {
    let mut alg = query.where_clause.clone();
    if query.query_type == QueryType::Ask {
        return Ok(alg);
    }

    // Collect aggregate projections `(AGG(...) AS ?alias)` in projection order.
    let aggregates: Vec<(Variable, Aggregate)> = query
        .projection_items
        .iter()
        .filter_map(|item| match item {
            ProjectionItem::Aggregate { aggregate, alias } => {
                Some((alias.clone(), aggregate.clone()))
            }
            _ => None,
        })
        .collect();

    // HAVING is passed through to the engine verbatim. The oxirs-arq
    // `Algebra::Having` executor detects aggregate function calls inside the
    // condition (`HAVING (COUNT(?s) > 1)`), hoists them into per-group
    // aggregates evaluated alongside the declared ones, and rewrites the filter
    // — so no caller-side rewrite (and its arity validation) is required here.
    let having_condition = query.having.clone();

    let has_grouping = !aggregates.is_empty() || !query.group_by.is_empty();

    // In an aggregate query, every plain projected variable must be a grouping
    // key; projecting a non-grouped, non-aggregated variable is a SPARQL error
    // (fail loud rather than emit a silently-unbound column).
    if has_grouping {
        let grouped: HashSet<&Variable> = query
            .group_by
            .iter()
            .filter_map(|gc| match &gc.expr {
                Expression::Variable(v) => Some(v),
                _ => None,
            })
            .chain(query.group_by.iter().filter_map(|gc| gc.alias.as_ref()))
            .collect();
        for item in &query.projection_items {
            if let ProjectionItem::Variable(var) = item {
                if !grouped.contains(var) {
                    return Err(FusekiError::query_parsing(format!(
                        "SELECT variable ?{} must be a GROUP BY key or wrapped in an aggregate \
                         function",
                        var.name()
                    )));
                }
            }
        }
    }

    if has_grouping {
        alg = Algebra::Group {
            pattern: Box::new(alg),
            variables: query.group_by.clone(),
            aggregates,
        };
    }

    if let Some(condition) = having_condition {
        alg = Algebra::Having {
            pattern: Box::new(alg),
            condition,
        };
    }

    // Projected expressions become Extend nodes, in projection order, so a later
    // `(expr AS ?v)` can reference an alias bound by an earlier one.
    for item in &query.projection_items {
        if let ProjectionItem::Expression { expr, alias } = item {
            alg = Algebra::Extend {
                pattern: Box::new(alg),
                variable: alias.clone(),
                expr: expr.clone(),
            };
        }
    }

    if !query.order_by.is_empty() {
        alg = Algebra::OrderBy {
            pattern: Box::new(alg),
            conditions: query.order_by.clone(),
        };
    }

    // `select_variables` carries the ordered output columns (aliases included).
    // Empty == `SELECT *` (project nothing / keep every in-scope variable).
    if !query.select_variables.is_empty() {
        alg = Algebra::Project {
            pattern: Box::new(alg),
            variables: query.select_variables.clone(),
        };
    }

    if query.distinct {
        alg = Algebra::Distinct {
            pattern: Box::new(alg),
        };
    }

    if query.limit.is_some() || query.offset.is_some() {
        alg = Algebra::Slice {
            pattern: Box::new(alg),
            offset: query.offset,
            limit: query.limit,
        };
    }

    Ok(alg)
}

/// Build the WHERE algebra for a CONSTRUCT / DESCRIBE query, applying the
/// solution-sequence modifiers that carry over (ORDER BY then LIMIT/OFFSET).
///
/// CONSTRUCT / DESCRIBE have no SELECT projection: every WHERE variable stays in
/// scope so the template / describe step can reference it. LIMIT/OFFSET bound
/// the number of WHERE solutions (SPARQL 1.1 §16.2), not the emitted triples.
fn build_graph_where_algebra(query: &Query) -> Algebra {
    let mut alg = query.where_clause.clone();
    if !query.order_by.is_empty() {
        alg = Algebra::OrderBy {
            pattern: Box::new(alg),
            conditions: query.order_by.clone(),
        };
    }
    if query.limit.is_some() || query.offset.is_some() {
        alg = Algebra::Slice {
            pattern: Box::new(alg),
            offset: query.offset,
            limit: query.limit,
        };
    }
    alg
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
///
/// Fallible because a solution term that cannot be a legitimate SPARQL results
/// value (an `ArqTerm::PropertyPath`) is a 500 fail-loud rather than a fabricated
/// binding — see [`term_to_json`].
fn select_result(solution: Solution) -> FusekiResult<QueryResult> {
    let bindings: Vec<HashMap<String, serde_json::Value>> = solution
        .iter()
        .map(binding_to_json)
        .collect::<FusekiResult<_>>()?;
    Ok(QueryResult {
        query_type: "SELECT".to_string(),
        execution_time_ms: 0,
        result_count: Some(bindings.len()),
        bindings: Some(bindings),
        boolean: None,
        construct_graph: None,
        describe_graph: None,
    })
}

/// Build a fuseki `QueryResult` carrying a CONSTRUCT graph (serialized RDF).
fn construct_result(graph: String, triple_count: usize) -> QueryResult {
    QueryResult {
        query_type: "CONSTRUCT".to_string(),
        execution_time_ms: 0,
        result_count: Some(triple_count),
        bindings: None,
        boolean: None,
        construct_graph: Some(graph),
        describe_graph: None,
    }
}

/// Build a fuseki `QueryResult` carrying a DESCRIBE graph (serialized RDF).
fn describe_result(graph: String, triple_count: usize) -> QueryResult {
    QueryResult {
        query_type: "DESCRIBE".to_string(),
        execution_time_ms: 0,
        result_count: Some(triple_count),
        bindings: None,
        boolean: None,
        construct_graph: None,
        describe_graph: Some(graph),
    }
}

/// Convert one arq binding (`Variable -> Term`) to the SPARQL Results JSON row
/// shape (`var name -> {type,value,...}`).
fn binding_to_json(
    binding: &HashMap<Variable, ArqTerm>,
) -> FusekiResult<HashMap<String, serde_json::Value>> {
    binding
        .iter()
        .map(|(var, term)| Ok((var.name().to_string(), term_to_json(term)?)))
        .collect()
}

/// Convert an arq `Term` to a SPARQL Query Results JSON term object.
///
/// The match is exhaustive over [`ArqTerm`] — there is no catch-all arm that
/// would fabricate a plain literal from a Rust `Debug` string. A `QuotedTriple`
/// is serialized per the RDF-star SPARQL results convention
/// (`{"type":"triple","value":{"subject":…,"predicate":…,"object":…}}`),
/// recursing into `term_to_json` for the three positions. A `PropertyPath` can
/// never be a legitimate solution binding, so it is a 500 fail-loud error rather
/// than an invented value.
fn term_to_json(term: &ArqTerm) -> FusekiResult<serde_json::Value> {
    Ok(match term {
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
        ArqTerm::QuotedTriple(triple) => serde_json::json!({
            "type": "triple",
            "value": {
                "subject": term_to_json(&triple.subject)?,
                "predicate": term_to_json(&triple.predicate)?,
                "object": term_to_json(&triple.object)?,
            }
        }),
        ArqTerm::PropertyPath(path) => {
            return Err(FusekiError::query_execution(format!(
                "property path term cannot be a SPARQL solution binding: {path}"
            )));
        }
    })
}

/// Serialize a set of arq graph triples (CONSTRUCT / DESCRIBE output) to Turtle,
/// reusing the shared core serializer after converting to oxirs-core triples.
///
/// Triples that cannot form a well-formed RDF triple in the core model (e.g. an
/// RDF-star quoted-triple term the serializer does not render) are dropped, so
/// the serialized graph never contains an ill-formed statement.
fn serialize_arq_graph(triples: &[ArqTriple]) -> String {
    let core: Vec<CoreTriple> = triples.iter().filter_map(arq_triple_to_core).collect();
    serialize_triples_to_turtle(&core)
}

/// Convert an arq algebra `Triple` into an oxirs-core `Triple`, or `None` when a
/// term is not valid in its position (dropping the triple).
fn arq_triple_to_core(triple: &ArqTriple) -> Option<CoreTriple> {
    let subject = arq_term_to_subject(&triple.subject)?;
    let predicate = arq_term_to_predicate(&triple.predicate)?;
    let object = arq_term_to_object(&triple.object)?;
    Some(CoreTriple::new(subject, predicate, object))
}

/// Map an arq term to a core subject (IRI or blank node).
fn arq_term_to_subject(term: &ArqTerm) -> Option<Subject> {
    match term {
        ArqTerm::Iri(iri) => Some(Subject::NamedNode(iri.clone())),
        ArqTerm::BlankNode(id) => BlankNode::new(id).ok().map(Subject::BlankNode),
        _ => None,
    }
}

/// Map an arq term to a core predicate (IRI only).
fn arq_term_to_predicate(term: &ArqTerm) -> Option<Predicate> {
    match term {
        ArqTerm::Iri(iri) => Some(Predicate::NamedNode(iri.clone())),
        _ => None,
    }
}

/// Map an arq term to a core object (IRI, literal or blank node).
fn arq_term_to_object(term: &ArqTerm) -> Option<Object> {
    match term {
        ArqTerm::Iri(iri) => Some(Object::NamedNode(iri.clone())),
        ArqTerm::BlankNode(id) => BlankNode::new(id).ok().map(Object::BlankNode),
        ArqTerm::Literal(lit) => Some(Object::Literal(arq_literal_to_core(lit))),
        _ => None,
    }
}

/// Build a core `Literal` from an arq `Literal`, preserving datatype/language.
fn arq_literal_to_core(lit: &ArqLiteral) -> CoreLiteral {
    if let Some(lang) = &lit.language {
        CoreLiteral::new_language_tagged_literal(&lit.value, lang)
            .unwrap_or_else(|_| CoreLiteral::new(&lit.value))
    } else if let Some(dt) = &lit.datatype {
        CoreLiteral::new_typed(&lit.value, dt.clone())
    } else {
        CoreLiteral::new(&lit.value)
    }
}
