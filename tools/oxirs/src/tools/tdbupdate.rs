//! TDB Update tool
//!
//! Execute SPARQL Update operations against a local TDB RDF store.
//! Supports INSERT DATA, DELETE DATA, and pattern-based insert/delete/modify.

use super::ToolResult;
use oxirs_arq::update_protocol::{PatternTerm, SparqlUpdate, SparqlUpdateParser, TriplePattern};
use oxirs_tdb::dictionary::Term as TdbTerm;
use oxirs_tdb::{TdbConfig, TdbStore};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// Variable bindings produced while matching a WHERE clause: variable name
/// (without the leading `?`) -> bound raw term string.
type Binding = HashMap<String, String>;

// ─── WHERE-pattern evaluation against TdbStore ─────────────────────────────────
//
// `TdbStore::insert`/`delete` (see the `InsertData`/`DeleteData` arms below)
// always encode every triple position as `Term::Iri(raw_string)`, regardless
// of whether the surface syntax looked like an IRI, literal, or blank node.
// WHERE-pattern matching below follows the exact same convention so it finds
// triples written through this tool, and instantiated INSERT/DELETE results
// are written back through the same `store.insert`/`store.delete` calls.

/// Resolve a single triple-pattern position to a concrete `TdbTerm` for
/// querying, or `None` when it is an unbound variable (a store-query
/// wildcard).
fn pattern_position_to_query_term(term: &PatternTerm, binding: &Binding) -> Option<TdbTerm> {
    match term {
        PatternTerm::Variable(var) => binding.get(var.as_str()).map(|v| TdbTerm::Iri(v.clone())),
        PatternTerm::Iri(iri) => Some(TdbTerm::Iri(iri.clone())),
        PatternTerm::Literal(lit) => Some(TdbTerm::Iri(lit.clone())),
        PatternTerm::BlankNode(bn) => Some(TdbTerm::Iri(bn.clone())),
    }
}

/// Extract the raw string value carried by a `TdbTerm`, regardless of variant.
fn term_raw_value(term: &TdbTerm) -> String {
    match term {
        TdbTerm::Iri(s) => s.clone(),
        TdbTerm::BlankNode(s) => s.clone(),
        TdbTerm::Literal { value, .. } => value.clone(),
    }
}

/// Extend `binding` with the variable this pattern position binds to
/// `matched`, checking consistency against any prior binding of the same
/// variable (a join). Concrete (non-variable) positions always succeed since
/// the store query already constrained them to match.
fn bind_matched_term(term: &PatternTerm, matched: &TdbTerm, binding: &mut Binding) -> bool {
    if let PatternTerm::Variable(var) = term {
        let value = term_raw_value(matched);
        match binding.get(var.as_str()) {
            Some(existing) => *existing == value,
            None => {
                binding.insert(var.clone(), value);
                true
            }
        }
    } else {
        true
    }
}

/// Resolve a triple-pattern position to its final string value given a
/// completed binding, or `None` if it is a variable left unbound.
fn resolve_pattern_term(term: &PatternTerm, binding: &Binding) -> Option<String> {
    match term {
        PatternTerm::Variable(var) => binding.get(var.as_str()).cloned(),
        PatternTerm::Iri(iri) => Some(iri.clone()),
        PatternTerm::Literal(lit) => Some(lit.clone()),
        PatternTerm::BlankNode(bn) => Some(bn.clone()),
    }
}

/// Instantiate a triple pattern (template or WHERE pattern) against a
/// completed binding into a concrete `(subject, predicate, object)` triple,
/// or `None` if any position is an unbound variable.
fn instantiate_pattern(
    pattern: &TriplePattern,
    binding: &Binding,
) -> Option<(String, String, String)> {
    Some((
        resolve_pattern_term(&pattern.s, binding)?,
        resolve_pattern_term(&pattern.p, binding)?,
        resolve_pattern_term(&pattern.o, binding)?,
    ))
}

/// Evaluate a WHERE clause (a conjunction of triple patterns) against the
/// real TDB store via `TdbStore::query_triples`, joining consecutive
/// patterns on shared variable names. Returns every consistent set of
/// variable bindings — the real SPARQL Update WHERE-pattern evaluation this
/// tool previously only logged a "best-effort" notice about.
fn match_patterns_against_store(
    store: &TdbStore,
    patterns: &[TriplePattern],
) -> anyhow::Result<Vec<Binding>> {
    let mut results: Vec<Binding> = vec![Binding::new()];

    for pattern in patterns {
        let mut next = Vec::new();
        for binding in &results {
            let s_term = pattern_position_to_query_term(&pattern.s, binding);
            let p_term = pattern_position_to_query_term(&pattern.p, binding);
            let o_term = pattern_position_to_query_term(&pattern.o, binding);

            let matches = store.query_triples(s_term.as_ref(), p_term.as_ref(), o_term.as_ref())?;

            for (s, p, o) in matches {
                let mut candidate = binding.clone();
                if bind_matched_term(&pattern.s, &s, &mut candidate)
                    && bind_matched_term(&pattern.p, &p, &mut candidate)
                    && bind_matched_term(&pattern.o, &o, &mut candidate)
                {
                    next.push(candidate);
                }
            }
        }
        results = next;
    }

    Ok(results)
}

// ─── Execute a single update op against the TDB store ─────────────────────────

fn apply_update(store: &mut TdbStore, update: &SparqlUpdate) -> anyhow::Result<(usize, usize)> {
    let mut inserted = 0usize;
    let mut deleted = 0usize;

    match update {
        SparqlUpdate::InsertData(triples) => {
            for t in triples {
                store.insert(&t.s, &t.p, &t.o)?;
                inserted += 1;
            }
        }
        SparqlUpdate::DeleteData(triples) => {
            for t in triples {
                // delete() takes bare string values; pass them through
                let removed = store.delete(&t.s, &t.p, &t.o)?;
                if removed {
                    deleted += 1;
                }
            }
        }
        SparqlUpdate::ClearGraph { .. } => {
            // Clear the default graph
            let count_before = store.count();
            store.clear()?;
            deleted = count_before;
        }
        // Other operations (named graphs, LOAD, COPY, MOVE, ADD) are logged but
        // not yet backed by the TDB API — return a descriptive notice.
        SparqlUpdate::CreateGraph { iri, .. } => {
            println!("Note: CREATE GRAPH <{iri}> is a no-op for the default graph store");
        }
        SparqlUpdate::DropGraph { iri, .. } => {
            let target = iri.as_deref().unwrap_or("<default>");
            println!("Note: DROP GRAPH {target} — clearing default graph");
            let count_before = store.count();
            store.clear()?;
            deleted = count_before;
        }
        SparqlUpdate::InsertWhere {
            template,
            where_clause,
        } => {
            let bindings = match_patterns_against_store(store, where_clause)?;
            let mut seen = HashSet::new();
            for binding in &bindings {
                for tp in template {
                    if let Some(triple) = instantiate_pattern(tp, binding) {
                        if seen.insert(triple.clone()) {
                            store.insert(&triple.0, &triple.1, &triple.2)?;
                            inserted += 1;
                        }
                    }
                }
            }
        }
        SparqlUpdate::DeleteWhere {
            template,
            where_clause,
        } => {
            // `DELETE WHERE { pattern }` (the common shorthand) parses with an
            // empty `template`, meaning "delete exactly what matched"; fall
            // back to `where_clause` as the delete template in that case.
            let delete_template: &[TriplePattern] = if template.is_empty() {
                where_clause
            } else {
                template
            };
            let bindings = match_patterns_against_store(store, where_clause)?;
            let mut seen = HashSet::new();
            for binding in &bindings {
                for tp in delete_template {
                    if let Some(triple) = instantiate_pattern(tp, binding) {
                        if seen.insert(triple.clone())
                            && store.delete(&triple.0, &triple.1, &triple.2)?
                        {
                            deleted += 1;
                        }
                    }
                }
            }
        }
        SparqlUpdate::Modify {
            delete,
            insert,
            where_clause,
        } => {
            let bindings = match_patterns_against_store(store, where_clause)?;
            let mut inserted_seen = HashSet::new();
            let mut deleted_seen = HashSet::new();
            for binding in &bindings {
                for tp in delete {
                    if let Some(triple) = instantiate_pattern(tp, binding) {
                        if deleted_seen.insert(triple.clone())
                            && store.delete(&triple.0, &triple.1, &triple.2)?
                        {
                            deleted += 1;
                        }
                    }
                }
                for tp in insert {
                    if let Some(triple) = instantiate_pattern(tp, binding) {
                        if inserted_seen.insert(triple.clone()) {
                            store.insert(&triple.0, &triple.1, &triple.2)?;
                            inserted += 1;
                        }
                    }
                }
            }
        }
        SparqlUpdate::CopyGraph { source, target, .. } => {
            println!("Note: COPY <{source}> TO <{target}> is advisory only in single-graph mode");
        }
        SparqlUpdate::MoveGraph { source, target, .. } => {
            println!("Note: MOVE <{source}> TO <{target}> is advisory only in single-graph mode");
        }
        SparqlUpdate::AddGraph { source, target, .. } => {
            println!("Note: ADD <{source}> TO <{target}> is advisory only in single-graph mode");
        }
        SparqlUpdate::Load { iri, .. } => {
            println!("Note: LOAD <{iri}> is not implemented in TDB Update tool");
        }
    }
    Ok((inserted, deleted))
}

// ─── Main entry point ─────────────────────────────────────────────────────────

/// Execute SPARQL Update operations against a local TDB store.
///
/// * `location`    — TDB store directory path
/// * `update`      — SPARQL Update string, or file path when `file` is true
/// * `file`        — when true, read the update from `update` as a file path
pub async fn run(location: PathBuf, update: String, file: bool) -> ToolResult {
    // Resolve update string
    let update_str = if file {
        std::fs::read_to_string(&update)
            .map_err(|e| format!("Cannot read update file '{}': {e}", update))?
    } else {
        update
    };

    // Validate and open TDB store
    if !location.exists() {
        return Err(format!("TDB location does not exist: {}", location.display()).into());
    }
    let config = TdbConfig::new(&location);
    let mut store = TdbStore::open_with_config(config)
        .map_err(|e| format!("Failed to open TDB store at '{}': {e}", location.display()))?;

    // Parse SPARQL Update
    let operations = SparqlUpdateParser::parse(&update_str)
        .map_err(|e| format!("SPARQL Update parse error: {e}"))?;

    if operations.is_empty() {
        println!("No update operations found in input.");
        return Ok(());
    }

    println!("Executing {} update operation(s)...", operations.len());

    let mut total_inserted = 0usize;
    let mut total_deleted = 0usize;

    for (idx, op) in operations.iter().enumerate() {
        let (ins, del) = apply_update(&mut store, op)
            .map_err(|e| format!("Update operation {} failed: {e}", idx + 1))?;
        total_inserted += ins;
        total_deleted += del;
    }

    println!("Update complete.");
    println!("  Triples inserted: {total_inserted}");
    println!("  Triples deleted:  {total_deleted}");
    println!("  Store size:       {} triples", store.count());

    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    /// Convert a bare IRI string (e.g., `<http://…>`) or literal string into a TdbTerm.
    fn str_to_tdb_term(s: &str) -> TdbTerm {
        let trimmed = s.trim();
        if let Some(inner) = trimmed.strip_prefix('<').and_then(|t| t.strip_suffix('>')) {
            TdbTerm::Iri(inner.to_string())
        } else if trimmed.starts_with('"') {
            // Minimal literal parsing: strip surrounding quotes
            let inner = trimmed.trim_matches('"');
            TdbTerm::literal(inner)
        } else if let Some(id) = trimmed.strip_prefix("_:") {
            TdbTerm::BlankNode(id.to_string())
        } else {
            // Treat bare tokens as IRIs (e.g. already-expanded prefixes)
            TdbTerm::Iri(trimmed.to_string())
        }
    }

    #[tokio::test]
    async fn test_missing_location_returns_error() {
        let loc = env::temp_dir().join("tdbupdate_no_such_dir_abc999");
        let res = run(
            loc,
            "INSERT DATA { <http://s> <http://p> <http://o> }".into(),
            false,
        )
        .await;
        assert!(res.is_err(), "should fail for non-existent location");
    }

    #[tokio::test]
    async fn test_insert_data_roundtrip() {
        let tmp = env::temp_dir().join("tdbupdate_insert_test");
        // Initialize the TDB store directory so the location exists
        let config = TdbConfig::new(&tmp);
        let _ = TdbStore::open_with_config(config);
        let res = run(
            tmp.clone(),
            "INSERT DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }"
                .into(),
            false,
        )
        .await;
        let _ = std::fs::remove_dir_all(&tmp);
        assert!(res.is_ok(), "insert should succeed: {:?}", res.err());
    }

    #[tokio::test]
    async fn test_empty_update_is_ok() {
        let tmp = env::temp_dir().join("tdbupdate_empty_test");
        // Create a valid store directory first so the open succeeds
        let config = TdbConfig::new(&tmp);
        let _ = TdbStore::open_with_config(config);
        let res = run(tmp.clone(), "# no ops\n".into(), false).await;
        let _ = std::fs::remove_dir_all(&tmp);
        assert!(res.is_ok(), "empty update should succeed: {:?}", res.err());
    }

    #[test]
    fn test_apply_update_insert_where_writes_matched_triples() {
        let tmp = env::temp_dir().join("tdbupdate_apply_insert_where_unit");
        let _ = std::fs::remove_dir_all(&tmp);
        let config = TdbConfig::new(&tmp);
        let mut store = TdbStore::open_with_config(config).expect("open store");
        store
            .insert(
                "http://example.org/x",
                "http://example.org/type",
                "http://example.org/Person",
            )
            .expect("seed insert");

        let ops = SparqlUpdateParser::parse(
            "INSERT { ?s <http://example.org/tag> <http://example.org/seen> } \
             WHERE { ?s <http://example.org/type> <http://example.org/Person> }",
        )
        .expect("parse update");
        assert_eq!(ops.len(), 1);

        let (inserted, deleted) = apply_update(&mut store, &ops[0]).expect("apply update");
        assert_eq!(
            (inserted, deleted),
            (1, 0),
            "must report the one real WHERE-matched insertion, not silently no-op"
        );
        assert!(store
            .contains(
                "http://example.org/x",
                "http://example.org/tag",
                "http://example.org/seen"
            )
            .expect("contains query"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_apply_update_delete_where_shorthand_removes_matched_triples() {
        let tmp = env::temp_dir().join("tdbupdate_apply_delete_where_unit");
        let _ = std::fs::remove_dir_all(&tmp);
        let config = TdbConfig::new(&tmp);
        let mut store = TdbStore::open_with_config(config).expect("open store");
        store
            .insert(
                "http://example.org/bob",
                "http://example.org/type",
                "http://example.org/Person",
            )
            .expect("seed insert");

        let ops = SparqlUpdateParser::parse(
            "DELETE WHERE { ?s <http://example.org/type> <http://example.org/Person> }",
        )
        .expect("parse update");
        assert_eq!(ops.len(), 1);

        let (inserted, deleted) = apply_update(&mut store, &ops[0]).expect("apply update");
        assert_eq!(
            (inserted, deleted),
            (0, 1),
            "DELETE WHERE shorthand must actually remove the matched triple"
        );
        assert!(!store
            .contains(
                "http://example.org/bob",
                "http://example.org/type",
                "http://example.org/Person"
            )
            .expect("contains query"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_apply_update_modify_deletes_and_inserts_via_join() {
        let tmp = env::temp_dir().join("tdbupdate_apply_modify_unit");
        let _ = std::fs::remove_dir_all(&tmp);
        let config = TdbConfig::new(&tmp);
        let mut store = TdbStore::open_with_config(config).expect("open store");
        store
            .insert(
                "http://example.org/carol",
                "http://example.org/status",
                "http://example.org/pending",
            )
            .expect("seed insert");

        let ops = SparqlUpdateParser::parse(
            "DELETE { ?s <http://example.org/status> <http://example.org/pending> } \
             INSERT { ?s <http://example.org/status> <http://example.org/done> } \
             WHERE { ?s <http://example.org/status> <http://example.org/pending> }",
        )
        .expect("parse update");
        assert_eq!(ops.len(), 1);

        let (inserted, deleted) = apply_update(&mut store, &ops[0]).expect("apply update");
        assert_eq!((inserted, deleted), (1, 1));
        assert!(!store
            .contains(
                "http://example.org/carol",
                "http://example.org/status",
                "http://example.org/pending"
            )
            .expect("contains query"));
        assert!(store
            .contains(
                "http://example.org/carol",
                "http://example.org/status",
                "http://example.org/done"
            )
            .expect("contains query"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_run_insert_where_end_to_end_persists_to_disk() {
        let tmp = env::temp_dir().join("tdbupdate_run_insert_where_e2e");
        let _ = std::fs::remove_dir_all(&tmp);
        let config = TdbConfig::new(&tmp);
        let _ = TdbStore::open_with_config(config);

        let update = "INSERT DATA { <http://example.org/alice> <http://example.org/type> <http://example.org/Person> } ; \
                       INSERT { ?s <http://example.org/isPerson> <http://example.org/true> } \
                       WHERE { ?s <http://example.org/type> <http://example.org/Person> }";

        let res = run(tmp.clone(), update.into(), false).await;
        assert!(
            res.is_ok(),
            "combined update should succeed: {:?}",
            res.err()
        );

        let reopened =
            TdbStore::open_with_config(TdbConfig::new(&tmp)).expect("reopen store after run()");
        assert!(
            reopened
                .contains(
                    "http://example.org/alice",
                    "http://example.org/isPerson",
                    "http://example.org/true"
                )
                .expect("contains query"),
            "the CLI `oxirs tdbupdate` entry point must persist real INSERT WHERE results, not just print a fake summary"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_str_to_tdb_term_iri() {
        let t = str_to_tdb_term("<http://example.org/test>");
        assert_eq!(t, TdbTerm::Iri("http://example.org/test".to_string()));
    }

    #[test]
    fn test_str_to_tdb_term_literal() {
        let t = str_to_tdb_term("\"hello\"");
        assert!(matches!(t, TdbTerm::Literal { .. }));
    }

    #[test]
    fn test_str_to_tdb_term_blank_node() {
        let t = str_to_tdb_term("_:b0");
        assert_eq!(t, TdbTerm::BlankNode("b0".to_string()));
    }
}
