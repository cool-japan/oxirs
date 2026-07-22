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
use oxirs_arq::query_governor::{BudgetExceeded, ExecutionBudget};
use oxirs_arq::{describe, instantiate_construct};
use oxirs_core::model::{
    BlankNode, Literal as CoreLiteral, Object, Predicate, Subject, Triple as CoreTriple,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

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

/// Dispatch a parsed query to the form-specific executor with no resource
/// budget (unbounded execution). Kept for standalone / test callers; the fuseki
/// query handler uses [`dispatch_with_budget`] to enforce the query timeout.
pub fn dispatch(query: &Query, store: &Store) -> FusekiResult<QueryResult> {
    dispatch_with_budget(query, store, None)
}

/// Dispatch a parsed query, attaching an optional [`ExecutionBudget`] so the
/// oxirs-arq engine enforces the query's wall-time (and, if configured, row /
/// scan) limits *during* evaluation. `None` means unbounded — behaviour
/// identical to the historical [`dispatch`].
pub fn dispatch_with_budget(
    query: &Query,
    store: &Store,
    budget: Option<Arc<ExecutionBudget>>,
) -> FusekiResult<QueryResult> {
    match query.query_type {
        QueryType::Select | QueryType::Ask => execute_select_or_ask(query, store, budget),
        QueryType::Construct => execute_construct(query, store, budget),
        QueryType::Describe => execute_describe(query, store, budget),
    }
}

/// Map an oxirs-arq engine error to a fuseki HTTP error, promoting a resource
/// budget breach to the correct status.
///
/// A **wall-time** budget breach is a timeout, mapped to `408 Request Timeout`
/// (`TimeoutWithMessage`). 408 is chosen deliberately so a client sees the same
/// status whether the *cooperative* query budget aborts the work or the outer
/// `TimeoutLayer` safety net trips — both mean "your request exceeded the time
/// limit". A **row / scan** budget breach is a resource cap the query blew past,
/// mapped to `503 Service Unavailable`: the server refused to keep spending
/// resources on it, and blindly retrying the identical query will not help. Any
/// other engine error keeps its historical `500` (`query_execution`).
///
/// Preserving the typed [`BudgetExceeded`] end-to-end depends on the engine
/// wrapping it with `anyhow::Error::new` (not `anyhow!("{e}")`); see
/// `QueryExecutor::budget_check_time`.
fn map_engine_error(context: &str, err: anyhow::Error) -> FusekiError {
    if let Some(budget_err) = err.downcast_ref::<BudgetExceeded>() {
        return match budget_err {
            BudgetExceeded::TimeoutExceeded {
                elapsed_ms,
                limit_ms,
            } => FusekiError::TimeoutWithMessage(format!(
                "query exceeded its {limit_ms} ms execution-time budget (ran {elapsed_ms} ms) \
                 and was aborted by the server"
            )),
            BudgetExceeded::ResultRowsExceeded { .. }
            | BudgetExceeded::TriplesScannedExceeded { .. } => {
                FusekiError::service_unavailable(format!("{context}: {budget_err}"))
            }
        };
    }
    FusekiError::query_execution(format!("{context}: {err}"))
}

/// Execute a parsed SELECT or ASK query.
///
/// SELECT builds the full solution-modifier stack natively from the parsed
/// query (grouping/aggregation, HAVING, projected expressions, ORDER BY,
/// projection, DISTINCT and slicing); ASK reduces to "any match".
pub fn execute_select_or_ask(
    query: &Query,
    store: &Store,
    budget: Option<Arc<ExecutionBudget>>,
) -> FusekiResult<QueryResult> {
    let algebra = build_select_algebra(query)?;
    let solution = run(store, &query.dataset, &algebra, budget)?;
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
pub fn execute_construct(
    query: &Query,
    store: &Store,
    budget: Option<Arc<ExecutionBudget>>,
) -> FusekiResult<QueryResult> {
    if query.construct_template.is_empty() {
        return Err(FusekiError::query_parsing(
            "CONSTRUCT template is empty: there is nothing to construct",
        ));
    }
    let algebra = build_graph_where_algebra(query);
    let solution = run(store, &query.dataset, &algebra, budget)?;
    // The oxirs-arq engine's `instantiate_construct` accepts path-encoded
    // (length-one `PropertyPath::Iri`/`Variable`) template predicates natively,
    // so the template is passed through unchanged — no caller-side normalization.
    let triples = instantiate_construct(&query.construct_template, &solution).map_err(|e| {
        FusekiError::query_execution(format!("CONSTRUCT instantiation failed: {e}"))
    })?;
    let graph = serialize_arq_graph(&triples)?;
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
pub fn execute_describe(
    query: &Query,
    store: &Store,
    budget: Option<Arc<ExecutionBudget>>,
) -> FusekiResult<QueryResult> {
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
        if let Some(budget) = budget {
            executor = executor.with_budget(budget);
        }
        let (solution, _stats) = executor
            .execute(&algebra, &view)
            .map_err(|e| map_engine_error("DESCRIBE WHERE failed", e))?;
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
    let graph = serialize_arq_graph(&triples)?;
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
fn run(
    store: &Store,
    clause: &DatasetClause,
    algebra: &Algebra,
    budget: Option<Arc<ExecutionBudget>>,
) -> FusekiResult<Solution> {
    let arc = store.get_dataset(None)?;
    let guard = arc
        .read()
        .map_err(|e| FusekiError::store(format!("failed to acquire store read lock: {e}")))?;
    let base = StoreRefDataset::new(&*guard);
    let view = with_dataset_clause(&base, clause);
    let mut executor = QueryExecutor::new();
    executor.set_strategy(ExecutionStrategy::Serial);
    // Attach the wall-time budget so the engine's throttled `check_time` calls in
    // hash_join / execute_minus / apply_left_join / execute_serial abort a
    // runaway before it monopolises this (blocking) thread.
    if let Some(budget) = budget {
        executor = executor.with_budget(budget);
    }
    let (solution, _stats) = executor
        .execute(algebra, &view)
        .map_err(|e| map_engine_error("query execution failed", e))?;
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
/// RDF-star quoted-triple subjects/objects are represented in the output
/// (Turtle-star `<< s p o >>`), not dropped. A term that genuinely cannot
/// occupy its position in a well-formed RDF triple — an unbound variable
/// that survived instantiation, a bare property-path term, or a quoted
/// triple used as a predicate — is a construction bug, not a value to
/// silently omit: this returns an explicit error so the caller surfaces a
/// 500 instead of a silently-incomplete graph.
fn serialize_arq_graph(triples: &[ArqTriple]) -> FusekiResult<String> {
    let core: Vec<CoreTriple> = triples
        .iter()
        .map(arq_triple_to_core)
        .collect::<Result<_, String>>()
        .map_err(|e| {
            FusekiError::query_execution(format!("cannot serialize constructed triple: {e}"))
        })?;
    Ok(serialize_triples_to_turtle(&core))
}

/// Convert an arq algebra `Triple` into an oxirs-core `Triple`. Returns an
/// error describing the offending term/position when the triple cannot be
/// represented in the core RDF-star model.
fn arq_triple_to_core(triple: &ArqTriple) -> Result<CoreTriple, String> {
    let subject = arq_term_to_subject(&triple.subject)?;
    let predicate = arq_term_to_predicate(&triple.predicate)?;
    let object = arq_term_to_object(&triple.object)?;
    Ok(CoreTriple::new(subject, predicate, object))
}

/// Map an arq term to a core subject (IRI, blank node, or RDF-star quoted triple).
fn arq_term_to_subject(term: &ArqTerm) -> Result<Subject, String> {
    match term {
        ArqTerm::Iri(iri) => Ok(Subject::NamedNode(iri.clone())),
        ArqTerm::BlankNode(id) => BlankNode::new(id)
            .map(Subject::BlankNode)
            .map_err(|e| format!("invalid blank node id '{id}': {e}")),
        ArqTerm::QuotedTriple(inner) => {
            let core_inner = arq_triple_to_core(inner)?;
            Ok(Subject::QuotedTriple(Box::new(
                oxirs_core::model::star::QuotedTriple::new(core_inner),
            )))
        }
        ArqTerm::Variable(v) => Err(format!(
            "unbound variable ?{} in constructed triple subject position",
            v.name()
        )),
        ArqTerm::Literal(lit) => Err(format!(
            "literal '{}' cannot be used as a triple subject",
            lit.value
        )),
        ArqTerm::PropertyPath(path) => Err(format!(
            "property path term cannot be a constructed triple subject: {path}"
        )),
    }
}

/// Map an arq term to a core predicate (IRI only: RDF-star forbids a
/// quoted triple, literal, or variable in predicate position of a
/// constructed triple).
fn arq_term_to_predicate(term: &ArqTerm) -> Result<Predicate, String> {
    match term {
        ArqTerm::Iri(iri) => Ok(Predicate::NamedNode(iri.clone())),
        ArqTerm::Variable(v) => Err(format!(
            "unbound variable ?{} in constructed triple predicate position",
            v.name()
        )),
        other => Err(format!(
            "term cannot be used as a constructed triple predicate: {other}"
        )),
    }
}

/// Map an arq term to a core object (IRI, literal, blank node, or
/// RDF-star quoted triple).
fn arq_term_to_object(term: &ArqTerm) -> Result<Object, String> {
    match term {
        ArqTerm::Iri(iri) => Ok(Object::NamedNode(iri.clone())),
        ArqTerm::BlankNode(id) => BlankNode::new(id)
            .map(Object::BlankNode)
            .map_err(|e| format!("invalid blank node id '{id}': {e}")),
        ArqTerm::Literal(lit) => Ok(Object::Literal(arq_literal_to_core(lit))),
        ArqTerm::QuotedTriple(inner) => {
            let core_inner = arq_triple_to_core(inner)?;
            Ok(Object::QuotedTriple(Box::new(
                oxirs_core::model::star::QuotedTriple::new(core_inner),
            )))
        }
        ArqTerm::Variable(v) => Err(format!(
            "unbound variable ?{} in constructed triple object position",
            v.name()
        )),
        ArqTerm::PropertyPath(path) => Err(format!(
            "property path term cannot be a constructed triple object: {path}"
        )),
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
