//! TDB Update tool
//!
//! Execute SPARQL Update operations against a local TDB RDF store.
//! Supports INSERT DATA, DELETE DATA, and pattern-based insert/delete/modify.

use super::ToolResult;
use oxirs_arq::update_protocol::{
    ClearType, DropType, PatternTerm, SparqlUpdate, SparqlUpdateParser, TriplePattern,
};
use oxirs_tdb::dictionary::Term as TdbTerm;
use oxirs_tdb::TdbStore;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Variable bindings produced while matching a WHERE clause: variable name
/// (without the leading `?`) -> the concrete, correctly-typed `TdbTerm` it is
/// bound to (preserving IRI/literal/blank-node kind through joins and
/// template instantiation — see the module-level note below).
type Binding = HashMap<String, TdbTerm>;

// ─── Term construction from parser output ──────────────────────────────────────
//
// The standalone SPARQL Update parser (`oxirs_arq::update_protocol`) hands
// back raw lexical forms: `Triple.{s,p,o}` (from `INSERT DATA`/`DELETE DATA`)
// are bare strings where an IRI has already had its `<…>` stripped but a
// literal keeps its surrounding quotes and a blank node keeps its `_:`
// prefix; `PatternTerm::{Iri,Literal,BlankNode}` (from WHERE/template
// patterns) already distinguish the term kind, with `Literal` still quoted
// and `BlankNode` already stripped of its `_:` prefix. Earlier revisions of
// this tool collapsed every position to `TdbTerm::Iri(..)` regardless of
// kind, which round-tripped internally but persisted RDF literals and blank
// nodes as if they were IRIs — silently corrupting the data model for any
// other consumer of the same TDB store (e.g. `tdbquery`/`tdbdump`). The
// helpers below build the *real* `TdbTerm` kind instead.

/// Unescape a still-quoted literal token (e.g. `"hello\n"` or `'it''s'`-style
/// backslash escapes) into its lexical value, stripping the surrounding
/// quote characters.
fn unquote_literal(token: &str) -> String {
    let inner: &str = if token.len() >= 2 {
        let bytes = token.as_bytes();
        let first = bytes[0];
        let last = bytes[bytes.len() - 1];
        if (first == b'"' || first == b'\'') && first == last {
            &token[1..token.len() - 1]
        } else {
            token
        }
    } else {
        token
    };

    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('"') => out.push('"'),
                Some('\'') => out.push('\''),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Convert a concrete `Triple.{s,p,o}` raw string (from `INSERT DATA` /
/// `DELETE DATA`) into the correctly-typed `TdbTerm` it denotes.
fn triple_string_to_term(raw: &str) -> TdbTerm {
    if let Some(id) = raw.strip_prefix("_:") {
        TdbTerm::BlankNode(id.to_string())
    } else if raw.starts_with('"') || raw.starts_with('\'') {
        TdbTerm::literal(unquote_literal(raw))
    } else {
        TdbTerm::Iri(raw.to_string())
    }
}

// ─── WHERE-pattern evaluation against TdbStore ─────────────────────────────────

/// Resolve a single triple-pattern position to a concrete `TdbTerm` for
/// querying, or `None` when it is an unbound variable (a store-query
/// wildcard).
fn pattern_position_to_query_term(term: &PatternTerm, binding: &Binding) -> Option<TdbTerm> {
    match term {
        PatternTerm::Variable(var) => binding.get(var.as_str()).cloned(),
        PatternTerm::Iri(iri) => Some(TdbTerm::Iri(iri.clone())),
        PatternTerm::Literal(lit) => Some(TdbTerm::literal(unquote_literal(lit))),
        PatternTerm::BlankNode(bn) => Some(TdbTerm::BlankNode(bn.clone())),
    }
}

/// Extend `binding` with the variable this pattern position binds to
/// `matched`, checking consistency against any prior binding of the same
/// variable (a join). Concrete (non-variable) positions always succeed since
/// the store query already constrained them to match.
fn bind_matched_term(term: &PatternTerm, matched: &TdbTerm, binding: &mut Binding) -> bool {
    if let PatternTerm::Variable(var) = term {
        match binding.get(var.as_str()) {
            Some(existing) => existing == matched,
            None => {
                binding.insert(var.clone(), matched.clone());
                true
            }
        }
    } else {
        true
    }
}

/// Resolve a triple-pattern position to its final `TdbTerm` given a
/// completed binding, or `None` if it is a variable left unbound.
fn resolve_pattern_term(term: &PatternTerm, binding: &Binding) -> Option<TdbTerm> {
    match term {
        PatternTerm::Variable(var) => binding.get(var.as_str()).cloned(),
        PatternTerm::Iri(iri) => Some(TdbTerm::Iri(iri.clone())),
        PatternTerm::Literal(lit) => Some(TdbTerm::literal(unquote_literal(lit))),
        PatternTerm::BlankNode(bn) => Some(TdbTerm::BlankNode(bn.clone())),
    }
}

/// Instantiate a triple pattern (template or WHERE pattern) against a
/// completed binding into a concrete `(subject, predicate, object)` triple of
/// correctly-typed terms, or `None` if any position is an unbound variable.
fn instantiate_pattern(
    pattern: &TriplePattern,
    binding: &Binding,
) -> Option<(TdbTerm, TdbTerm, TdbTerm)> {
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

// ─── DELETE-side type limitation ────────────────────────────────────────────────
//
// `TdbStore::delete`/`contains` are string-only APIs that unconditionally
// encode every position as `Term::Iri` when looking the term up in the
// dictionary (this is a limitation of `oxirs_tdb::TdbStore`'s public surface,
// not of this tool). Since INSERT now correctly stores literals and blank
// nodes under their real `Term` variant, a delete lookup that force-encodes
// them as `Term::Iri` would silently miss the dictionary entry entirely. Per
// the fail-loud contract, detect that mismatch up front and return a clear,
// explicit error rather than letting a literal/blank-node DELETE silently
// match nothing (or bubble up an opaque "not found in dictionary" error).

/// Render a `TdbTerm` back to the raw string `TdbStore::delete` expects, or
/// `Err` with an explanatory message when the term's kind cannot be honored
/// by that string-only, IRI-only-matching API.
fn term_to_delete_raw(position: &str, term: &TdbTerm) -> anyhow::Result<String> {
    match term {
        TdbTerm::Iri(s) => Ok(s.clone()),
        TdbTerm::Literal { value, .. } => Err(anyhow::anyhow!(
            "DELETE of a literal {position} (\"{value}\") is not supported by `oxirs tdbupdate`: \
             the underlying TdbStore::delete API only matches IRI-typed dictionary entries"
        )),
        TdbTerm::BlankNode(id) => Err(anyhow::anyhow!(
            "DELETE of a blank-node {position} (_:{id}) is not supported by `oxirs tdbupdate`: \
             the underlying TdbStore::delete API only matches IRI-typed dictionary entries"
        )),
    }
}

/// Reject (up front) a concrete `INSERT DATA`/`DELETE DATA` triple position
/// whose raw lexical form is a literal or blank node, for the same reason as
/// [`term_to_delete_raw`].
fn reject_unsupported_delete_term(position: &str, raw: &str) -> anyhow::Result<()> {
    if raw.starts_with('"') || raw.starts_with('\'') {
        anyhow::bail!(
            "DELETE of a literal {position} (\"{raw}\") is not supported by `oxirs tdbupdate`: \
             the underlying TdbStore::delete API only matches IRI-typed dictionary entries"
        );
    }
    if raw.starts_with("_:") {
        anyhow::bail!(
            "DELETE of a blank-node {position} ({raw}) is not supported by `oxirs tdbupdate`: \
             the underlying TdbStore::delete API only matches IRI-typed dictionary entries"
        );
    }
    Ok(())
}

// ─── LOAD ────────────────────────────────────────────────────────────────────

/// Resolve a `LOAD` source IRI to a local filesystem path, or an explicit
/// error when it names a scheme this tool cannot fetch.
fn load_source_to_path(iri: &str) -> anyhow::Result<PathBuf> {
    if let Some(path) = iri.strip_prefix("file://") {
        Ok(PathBuf::from(path))
    } else if let Some(path) = iri.strip_prefix("file:") {
        Ok(PathBuf::from(path))
    } else if iri.starts_with("http://") || iri.starts_with("https://") {
        anyhow::bail!(
            "LOAD <{iri}>: fetching remote RDF sources over HTTP(S) is not supported by \
             `oxirs tdbupdate`; download the file locally and LOAD it via a file:// IRI or a \
             plain path instead"
        )
    } else if iri.contains("://") {
        anyhow::bail!("LOAD <{iri}>: unsupported URI scheme")
    } else {
        // A bare (relative or absolute) filesystem path, as commonly accepted
        // by SPARQL Update tools for local LOAD sources.
        Ok(PathBuf::from(iri))
    }
}

/// Actually perform `LOAD <iri>` into the default graph: read the local file,
/// parse it per its detected RDF format, and insert every resulting triple
/// with its real term kind. Returns the number of triples inserted.
fn load_iri_into_store(store: &mut TdbStore, iri: &str) -> anyhow::Result<usize> {
    let path = load_source_to_path(iri)?;
    if !path.exists() {
        anyhow::bail!("LOAD <{iri}>: file not found at '{}'", path.display());
    }
    let content = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("LOAD <{iri}>: failed to read '{}': {e}", path.display()))?;
    let format = super::utils::detect_rdf_format(&path);
    let (triples, errors) = super::tdbloader::parse_rdf_for_loading(&content, &format)
        .map_err(|e| anyhow::anyhow!("LOAD <{iri}>: {e}"))?;
    if errors > 0 {
        anyhow::bail!("LOAD <{iri}>: {errors} line(s) failed to parse as {format}");
    }

    let mut count = 0usize;
    for t in &triples {
        let s = super::tdbloader::parse_term(&t.subject)
            .map_err(|e| anyhow::anyhow!("LOAD <{iri}>: invalid subject '{}': {e}", t.subject))?;
        let p = super::tdbloader::parse_term(&t.predicate).map_err(|e| {
            anyhow::anyhow!("LOAD <{iri}>: invalid predicate '{}': {e}", t.predicate)
        })?;
        let o = super::tdbloader::parse_term(&t.object)
            .map_err(|e| anyhow::anyhow!("LOAD <{iri}>: invalid object '{}': {e}", t.object))?;
        store.insert_triple(&s, &p, &o)?;
        count += 1;
    }
    Ok(count)
}

// ─── Execute a single update op against the TDB store ─────────────────────────

fn apply_update(store: &mut TdbStore, update: &SparqlUpdate) -> anyhow::Result<(usize, usize)> {
    let mut inserted = 0usize;
    let mut deleted = 0usize;

    match update {
        SparqlUpdate::InsertData(triples) => {
            for t in triples {
                let s = triple_string_to_term(&t.s);
                let p = triple_string_to_term(&t.p);
                let o = triple_string_to_term(&t.o);
                store.insert_triple(&s, &p, &o)?;
                inserted += 1;
            }
        }
        SparqlUpdate::DeleteData(triples) => {
            for t in triples {
                reject_unsupported_delete_term("subject", &t.s)?;
                reject_unsupported_delete_term("predicate", &t.p)?;
                reject_unsupported_delete_term("object", &t.o)?;
                // delete() takes bare string values; pass them through
                let removed = store.delete(&t.s, &t.p, &t.o)?;
                if removed {
                    deleted += 1;
                }
            }
        }
        SparqlUpdate::ClearGraph {
            iri,
            clear_type,
            silent,
        } => {
            deleted = apply_graph_scope_clear(
                store,
                clear_type.clone(),
                iri.as_deref(),
                *silent,
                "CLEAR",
            )?;
        }
        SparqlUpdate::CreateGraph { iri, .. } => {
            anyhow::bail!(
                "CREATE GRAPH <{iri}>: named graphs are not supported by this \
                 single-default-graph TDB store"
            );
        }
        SparqlUpdate::DropGraph {
            iri,
            drop_type,
            silent,
        } => {
            let clear_type = match drop_type {
                DropType::Graph => ClearType::Graph,
                DropType::Default => ClearType::Default,
                DropType::Named => ClearType::Named,
                DropType::All => ClearType::All,
            };
            deleted = apply_graph_scope_clear(store, clear_type, iri.as_deref(), *silent, "DROP")?;
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
                            store.insert_triple(&triple.0, &triple.1, &triple.2)?;
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
                        if seen.insert(triple.clone()) {
                            let sraw = term_to_delete_raw("subject", &triple.0)?;
                            let praw = term_to_delete_raw("predicate", &triple.1)?;
                            let oraw = term_to_delete_raw("object", &triple.2)?;
                            if store.delete(&sraw, &praw, &oraw)? {
                                deleted += 1;
                            }
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
                        if deleted_seen.insert(triple.clone()) {
                            let sraw = term_to_delete_raw("subject", &triple.0)?;
                            let praw = term_to_delete_raw("predicate", &triple.1)?;
                            let oraw = term_to_delete_raw("object", &triple.2)?;
                            if store.delete(&sraw, &praw, &oraw)? {
                                deleted += 1;
                            }
                        }
                    }
                }
                for tp in insert {
                    if let Some(triple) = instantiate_pattern(tp, binding) {
                        if inserted_seen.insert(triple.clone()) {
                            store.insert_triple(&triple.0, &triple.1, &triple.2)?;
                            inserted += 1;
                        }
                    }
                }
            }
        }
        SparqlUpdate::CopyGraph { source, target, .. } => {
            anyhow::bail!(
                "COPY <{source}> TO <{target}>: named graphs are not supported by this \
                 single-default-graph TDB store"
            );
        }
        SparqlUpdate::MoveGraph { source, target, .. } => {
            anyhow::bail!(
                "MOVE <{source}> TO <{target}>: named graphs are not supported by this \
                 single-default-graph TDB store"
            );
        }
        SparqlUpdate::AddGraph { source, target, .. } => {
            anyhow::bail!(
                "ADD <{source}> TO <{target}>: named graphs are not supported by this \
                 single-default-graph TDB store"
            );
        }
        SparqlUpdate::Load { iri, into, silent } => {
            if let Some(target_graph) = into {
                anyhow::bail!(
                    "LOAD <{iri}> INTO GRAPH <{target_graph}>: named graphs are not supported by \
                     this single-default-graph TDB store"
                );
            }
            match load_iri_into_store(store, iri) {
                Ok(count) => inserted += count,
                Err(e) => {
                    if *silent {
                        eprintln!("Warning: LOAD <{iri}> SILENT suppressed error: {e}");
                    } else {
                        return Err(e);
                    }
                }
            }
        }
    }
    Ok((inserted, deleted))
}

/// Shared `CLEAR`/`DROP` graph-scope handling for this single-default-graph
/// store. `DEFAULT`/`ALL` really do clear the store's data; `NAMED` targets
/// the (always-empty) set of named graphs, which vacuously succeeds; and a
/// specific `GRAPH <iri>` target maps onto the SPARQL-spec-mandated "no such
/// graph" condition — which `SILENT` is precisely defined to suppress — since
/// no named graph has ever existed in this store.
fn apply_graph_scope_clear(
    store: &mut TdbStore,
    scope: ClearType,
    iri: Option<&str>,
    silent: bool,
    verb: &str,
) -> anyhow::Result<usize> {
    match scope {
        ClearType::Default | ClearType::All => {
            let count_before = store.count();
            store.clear()?;
            Ok(count_before)
        }
        ClearType::Named => Ok(0),
        ClearType::Graph => {
            if silent {
                Ok(0)
            } else {
                let target = iri.unwrap_or("<unknown>");
                anyhow::bail!(
                    "{verb} GRAPH <{target}>: no such named graph (this TDB store only has a \
                     default graph); use {verb} SILENT to ignore, or {verb} DEFAULT/ALL to \
                     clear the default graph"
                )
            }
        }
    }
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

    let config = resolve_tdb_config(&location)?;
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

/// Build the real `oxirs_tdb::TdbConfig` to open `location` with, honoring
/// the active oxirs CLI profile's `tools.tdb.{cache_size,file_mode,sync_mode}`
/// instead of silently discarding them. A missing/unavailable config
/// directory falls back to engine defaults (no config file is required to
/// use this tool); a config file that exists but fails to parse, or that
/// sets an unrecognized `sync_mode`/`file_mode`, is a fail-loud error.
fn resolve_tdb_config(location: &Path) -> Result<oxirs_tdb::TdbConfig, String> {
    let tdb_profile = match crate::config::ConfigManager::new() {
        Ok(mut manager) => match manager.load_profile("default") {
            Ok(cfg) => cfg.tools.tdb.clone(),
            Err(e) => {
                return Err(format!(
                    "Failed to load oxirs CLI configuration profile: {e}"
                ));
            }
        },
        Err(_) => Default::default(),
    };
    tdb_profile
        .to_oxirs_tdb_config(location)
        .map_err(|e| format!("Invalid TDB configuration: {e}"))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_tdb::TdbConfig;
    use std::env;

    /// Serializes `OXIRS_CONFIG_DIR` env mutation across the env-dependent
    /// regression tests below so they don't race each other. `tokio::sync::Mutex`
    /// (not `std::sync::Mutex`) so the guard can be held safely across `.await`.
    static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

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

    // ─── Regression: DROP/CLEAR GRAPH must not wipe the default graph for a
    // named-graph target (P1 finding #1) ──────────────────────────────────

    #[test]
    fn regression_clear_graph_specific_iri_does_not_wipe_default_graph() {
        let tmp = env::temp_dir().join("tdbupdate_clear_graph_specific_unit");
        let _ = std::fs::remove_dir_all(&tmp);
        let config = TdbConfig::new(&tmp);
        let mut store = TdbStore::open_with_config(config).expect("open store");
        store
            .insert(
                "http://example.org/s",
                "http://example.org/p",
                "http://example.org/o",
            )
            .expect("seed insert");

        let ops =
            SparqlUpdateParser::parse("CLEAR GRAPH <http://example.org/g>").expect("parse update");
        let result = apply_update(&mut store, &ops[0]);
        assert!(
            result.is_err(),
            "CLEAR GRAPH <iri> without SILENT on a nonexistent named graph must error, not wipe the default graph"
        );
        assert_eq!(
            store.count(),
            1,
            "the default graph's data must survive an errored CLEAR GRAPH <iri>"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn regression_clear_graph_silent_specific_iri_is_noop_not_wipe() {
        let tmp = env::temp_dir().join("tdbupdate_clear_graph_silent_unit");
        let _ = std::fs::remove_dir_all(&tmp);
        let config = TdbConfig::new(&tmp);
        let mut store = TdbStore::open_with_config(config).expect("open store");
        store
            .insert(
                "http://example.org/s",
                "http://example.org/p",
                "http://example.org/o",
            )
            .expect("seed insert");

        let ops = SparqlUpdateParser::parse("CLEAR SILENT GRAPH <http://example.org/g>")
            .expect("parse update");
        let (inserted, deleted) = apply_update(&mut store, &ops[0]).expect("silent clear succeeds");
        assert_eq!((inserted, deleted), (0, 0));
        assert_eq!(
            store.count(),
            1,
            "CLEAR SILENT GRAPH <iri> on a nonexistent named graph must be a true no-op"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn regression_clear_default_still_wipes_default_graph() {
        let tmp = env::temp_dir().join("tdbupdate_clear_default_unit");
        let _ = std::fs::remove_dir_all(&tmp);
        let config = TdbConfig::new(&tmp);
        let mut store = TdbStore::open_with_config(config).expect("open store");
        store
            .insert(
                "http://example.org/s",
                "http://example.org/p",
                "http://example.org/o",
            )
            .expect("seed insert");

        let ops = SparqlUpdateParser::parse("CLEAR DEFAULT").expect("parse update");
        let (_inserted, deleted) = apply_update(&mut store, &ops[0]).expect("clear default");
        assert_eq!(deleted, 1);
        assert_eq!(store.count(), 0);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn regression_drop_graph_named_is_vacuous_noop() {
        let tmp = env::temp_dir().join("tdbupdate_drop_named_unit");
        let _ = std::fs::remove_dir_all(&tmp);
        let config = TdbConfig::new(&tmp);
        let mut store = TdbStore::open_with_config(config).expect("open store");
        store
            .insert(
                "http://example.org/s",
                "http://example.org/p",
                "http://example.org/o",
            )
            .expect("seed insert");

        let ops = SparqlUpdateParser::parse("DROP NAMED").expect("parse update");
        let (_inserted, deleted) = apply_update(&mut store, &ops[0]).expect("drop named");
        assert_eq!(deleted, 0);
        assert_eq!(
            store.count(),
            1,
            "DROP NAMED must not touch the default graph (there are no named graphs to drop)"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ─── Regression: LOAD/COPY/MOVE/ADD/CREATE GRAPH must fail loud, not
    // silently succeed (P1 finding #2) ────────────────────────────────────

    #[test]
    fn regression_create_graph_fails_loud() {
        let tmp = env::temp_dir().join("tdbupdate_create_graph_unit");
        let _ = std::fs::remove_dir_all(&tmp);
        let config = TdbConfig::new(&tmp);
        let mut store = TdbStore::open_with_config(config).expect("open store");

        let ops =
            SparqlUpdateParser::parse("CREATE GRAPH <http://example.org/g>").expect("parse update");
        let result = apply_update(&mut store, &ops[0]);
        assert!(
            result.is_err(),
            "CREATE GRAPH must fail loud, not silently succeed"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn regression_copy_move_add_graph_fail_loud() {
        let tmp = env::temp_dir().join("tdbupdate_copy_move_add_unit");
        let _ = std::fs::remove_dir_all(&tmp);
        let config = TdbConfig::new(&tmp);
        let mut store = TdbStore::open_with_config(config).expect("open store");

        for stmt in [
            "COPY <http://example.org/a> TO <http://example.org/b>",
            "MOVE <http://example.org/a> TO <http://example.org/b>",
            "ADD <http://example.org/a> TO <http://example.org/b>",
        ] {
            let ops = SparqlUpdateParser::parse(stmt).expect("parse update");
            let result = apply_update(&mut store, &ops[0]);
            assert!(
                result.is_err(),
                "'{stmt}' must fail loud, not silently succeed"
            );
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn regression_load_missing_file_fails_loud() {
        let tmp = env::temp_dir().join("tdbupdate_load_missing_unit");
        let _ = std::fs::remove_dir_all(&tmp);
        let config = TdbConfig::new(&tmp);
        let mut store = TdbStore::open_with_config(config).expect("open store");

        let ops = SparqlUpdateParser::parse(
            "LOAD <file:///nonexistent/path/oxirs_regression_missing.ttl>",
        )
        .expect("parse update");
        let result = apply_update(&mut store, &ops[0]);
        assert!(result.is_err(), "LOAD of a missing file must fail loud");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn regression_load_into_graph_fails_loud() {
        let tmp = env::temp_dir().join("tdbupdate_load_into_graph_unit");
        let _ = std::fs::remove_dir_all(&tmp);
        let config = TdbConfig::new(&tmp);
        let mut store = TdbStore::open_with_config(config).expect("open store");

        let ops = SparqlUpdateParser::parse(
            "LOAD <http://example.org/data.ttl> INTO GRAPH <http://example.org/g>",
        )
        .expect("parse update");
        let result = apply_update(&mut store, &ops[0]);
        assert!(
            result.is_err(),
            "LOAD ... INTO GRAPH must fail loud (named graphs unsupported)"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn regression_load_real_file_inserts_real_triples() {
        let tmp = env::temp_dir().join("tdbupdate_load_real_unit");
        let _ = std::fs::remove_dir_all(&tmp);
        let config = TdbConfig::new(&tmp);
        let mut store = TdbStore::open_with_config(config).expect("open store");

        let data_dir = env::temp_dir().join("tdbupdate_load_real_data_unit");
        std::fs::create_dir_all(&data_dir).expect("create data dir");
        let data_file = data_dir.join("oxirs_regression_load.nt");
        std::fs::write(
            &data_file,
            "<http://example.org/s> <http://example.org/p> \"loaded value\" .\n",
        )
        .expect("write data file");

        let update_text = format!("LOAD <file://{}>", data_file.display());
        let ops = SparqlUpdateParser::parse(&update_text).expect("parse update");
        let (inserted, _deleted) = apply_update(&mut store, &ops[0]).expect("real LOAD succeeds");
        assert_eq!(
            inserted, 1,
            "LOAD must really parse and insert the file's triple"
        );
        // `tdbloader::parse_term` types an unsuffixed literal as plain
        // `xsd:string`; query by the real Literal term rather than the
        // string-only `contains()` API (which forces `Term::Iri` and would
        // never find a correctly-typed literal object — see the DELETE-side
        // type-limitation note above).
        let matches = store
            .query_triples(
                Some(&TdbTerm::Iri("http://example.org/s".to_string())),
                Some(&TdbTerm::Iri("http://example.org/p".to_string())),
                Some(&TdbTerm::literal_with_datatype(
                    "loaded value",
                    "http://www.w3.org/2001/XMLSchema#string",
                )),
            )
            .expect("query loaded literal");
        assert_eq!(
            matches.len(),
            1,
            "LOAD must insert the file's triple with its real literal type"
        );

        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::remove_dir_all(&data_dir);
    }

    // ─── Regression: literals/blank nodes must round-trip with their real
    // RDF type, not collapse to IRI (P2 finding #3) ────────────────────────

    #[test]
    fn regression_insert_data_literal_object_is_stored_as_literal() {
        let tmp = env::temp_dir().join("tdbupdate_literal_insert_unit");
        let _ = std::fs::remove_dir_all(&tmp);
        let config = TdbConfig::new(&tmp);
        let mut store = TdbStore::open_with_config(config).expect("open store");

        let ops = SparqlUpdateParser::parse(
            "INSERT DATA { <http://example.org/s> <http://example.org/p> \"hello\" }",
        )
        .expect("parse update");
        let (inserted, _deleted) = apply_update(&mut store, &ops[0]).expect("insert literal");
        assert_eq!(inserted, 1);

        // The object must be queryable as a real Literal term, not an IRI
        // whose lexical form happens to be the quoted string.
        let matches = store
            .query_triples(
                Some(&TdbTerm::Iri("http://example.org/s".to_string())),
                Some(&TdbTerm::Iri("http://example.org/p".to_string())),
                Some(&TdbTerm::literal("hello")),
            )
            .expect("query by literal term");
        assert_eq!(
            matches.len(),
            1,
            "object must round-trip as a real Literal term"
        );

        // And it must NOT also be queryable as an IRI with the same raw text.
        let iri_matches = store
            .query_triples(
                Some(&TdbTerm::Iri("http://example.org/s".to_string())),
                Some(&TdbTerm::Iri("http://example.org/p".to_string())),
                Some(&TdbTerm::Iri("hello".to_string())),
            )
            .expect("query by iri term");
        assert!(
            iri_matches.is_empty(),
            "a literal object must never be stored as if it were an IRI"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn regression_insert_where_literal_binding_preserves_type_through_join() {
        let tmp = env::temp_dir().join("tdbupdate_literal_join_unit");
        let _ = std::fs::remove_dir_all(&tmp);
        let config = TdbConfig::new(&tmp);
        let mut store = TdbStore::open_with_config(config).expect("open store");
        store
            .insert_triple(
                &TdbTerm::Iri("http://example.org/x".to_string()),
                &TdbTerm::Iri("http://example.org/name".to_string()),
                &TdbTerm::literal("Alice"),
            )
            .expect("seed literal triple");

        // WHERE must find the literal-typed triple (it would not, if the
        // pattern's literal position were force-encoded as an IRI), and the
        // template's own literal position must also match by real type.
        let ops = SparqlUpdateParser::parse(
            "INSERT { ?s <http://example.org/tagged> \"yes\" } \
             WHERE { ?s <http://example.org/name> \"Alice\" }",
        )
        .expect("parse update");
        let (inserted, _deleted) = apply_update(&mut store, &ops[0]).expect("apply update");
        assert_eq!(
            inserted, 1,
            "WHERE-clause literal matching must find the literal-typed seed triple"
        );
        assert!(
            store
                .query_triples(
                    Some(&TdbTerm::Iri("http://example.org/x".to_string())),
                    Some(&TdbTerm::Iri("http://example.org/tagged".to_string())),
                    Some(&TdbTerm::literal("yes")),
                )
                .expect("query")
                .len()
                == 1
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn regression_insert_data_blank_node_is_stored_as_blank_node() {
        let tmp = env::temp_dir().join("tdbupdate_blank_node_insert_unit");
        let _ = std::fs::remove_dir_all(&tmp);
        let config = TdbConfig::new(&tmp);
        let mut store = TdbStore::open_with_config(config).expect("open store");

        let ops = SparqlUpdateParser::parse(
            "INSERT DATA { _:b0 <http://example.org/p> <http://example.org/o> }",
        )
        .expect("parse update");
        let (inserted, _deleted) = apply_update(&mut store, &ops[0]).expect("insert blank node");
        assert_eq!(inserted, 1);

        let matches = store
            .query_triples(Some(&TdbTerm::BlankNode("b0".to_string())), None, None)
            .expect("query by blank node term");
        assert_eq!(
            matches.len(),
            1,
            "subject must round-trip as a real BlankNode term"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn regression_delete_data_literal_object_fails_loud_instead_of_silently_wrong() {
        let tmp = env::temp_dir().join("tdbupdate_delete_literal_unit");
        let _ = std::fs::remove_dir_all(&tmp);
        let config = TdbConfig::new(&tmp);
        let mut store = TdbStore::open_with_config(config).expect("open store");
        store
            .insert_triple(
                &TdbTerm::Iri("http://example.org/s".to_string()),
                &TdbTerm::Iri("http://example.org/p".to_string()),
                &TdbTerm::literal("hello"),
            )
            .expect("seed literal triple");

        let ops = SparqlUpdateParser::parse(
            "DELETE DATA { <http://example.org/s> <http://example.org/p> \"hello\" }",
        )
        .expect("parse update");
        let result = apply_update(&mut store, &ops[0]);
        assert!(
            result.is_err(),
            "DELETE of a literal-typed triple must fail loud (documented store limitation), \
             not silently report success/failure while leaving the data untouched"
        );
        // The seed triple must still be present: the failed DELETE must not
        // have corrupted or partially mutated the store.
        assert!(
            store
                .query_triples(
                    Some(&TdbTerm::Iri("http://example.org/s".to_string())),
                    Some(&TdbTerm::Iri("http://example.org/p".to_string())),
                    Some(&TdbTerm::literal("hello")),
                )
                .expect("query")
                .len()
                == 1
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn regression_delete_data_iri_only_triple_still_works() {
        let tmp = env::temp_dir().join("tdbupdate_delete_iri_unit");
        let _ = std::fs::remove_dir_all(&tmp);
        let config = TdbConfig::new(&tmp);
        let mut store = TdbStore::open_with_config(config).expect("open store");
        store
            .insert(
                "http://example.org/s",
                "http://example.org/p",
                "http://example.org/o",
            )
            .expect("seed insert");

        let ops = SparqlUpdateParser::parse(
            "DELETE DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }",
        )
        .expect("parse update");
        let (_inserted, deleted) =
            apply_update(&mut store, &ops[0]).expect("plain-IRI delete still works");
        assert_eq!(deleted, 1);
        assert_eq!(store.count(), 0);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ─── Regression: CLI profile's tools.tdb config must reach the store
    // (P2 config-flag-matrix finding) ───────────────────────────────────────

    #[test]
    fn regression_resolve_tdb_config_defaults_when_no_profile_file() {
        // No OXIRS_CONFIG_DIR override is set up for this test, so this
        // exercises the "no config file present" fallback path end to end.
        let tmp = env::temp_dir().join("tdbupdate_resolve_config_default_unit");
        let config = resolve_tdb_config(&tmp).expect("must not fail with no profile file");
        // Must be a real, usable config (matches the engine's own defaults).
        let baseline = TdbConfig::new(&tmp);
        assert_eq!(config.buffer_pool_size, baseline.buffer_pool_size);
    }

    #[tokio::test]
    async fn regression_cli_profile_cache_size_reaches_store_end_to_end() {
        let _guard = ENV_LOCK.lock().await;

        let config_dir = env::temp_dir().join("tdbupdate_cli_profile_config_dir_unit");
        let _ = std::fs::remove_dir_all(&config_dir);
        std::fs::create_dir_all(&config_dir).expect("create config dir");
        std::fs::write(
            config_dir.join("config.toml"),
            "[tools.tdb]\ncache_size = 4321\nsync_mode = \"sync\"\n",
        )
        .expect("write config.toml");

        std::env::set_var("OXIRS_CONFIG_DIR", &config_dir);

        let tmp = env::temp_dir().join("tdbupdate_cli_profile_store_unit");
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = TdbStore::open_with_config(TdbConfig::new(&tmp));

        let res = run(
            tmp.clone(),
            "INSERT DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }"
                .into(),
            false,
        )
        .await;

        std::env::remove_var("OXIRS_CONFIG_DIR");
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::remove_dir_all(&config_dir);

        assert!(
            res.is_ok(),
            "update with a valid tools.tdb profile config should succeed: {:?}",
            res.err()
        );
    }

    #[tokio::test]
    async fn regression_cli_profile_invalid_sync_mode_fails_loud() {
        let _guard = ENV_LOCK.lock().await;

        let config_dir = env::temp_dir().join("tdbupdate_cli_profile_bad_config_dir_unit");
        let _ = std::fs::remove_dir_all(&config_dir);
        std::fs::create_dir_all(&config_dir).expect("create config dir");
        std::fs::write(
            config_dir.join("config.toml"),
            "[tools.tdb]\nsync_mode = \"bogus-mode\"\n",
        )
        .expect("write config.toml");

        std::env::set_var("OXIRS_CONFIG_DIR", &config_dir);

        let tmp = env::temp_dir().join("tdbupdate_cli_profile_bad_store_unit");
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = TdbStore::open_with_config(TdbConfig::new(&tmp));

        let res = run(
            tmp.clone(),
            "INSERT DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }"
                .into(),
            false,
        )
        .await;

        std::env::remove_var("OXIRS_CONFIG_DIR");
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::remove_dir_all(&config_dir);

        assert!(
            res.is_err(),
            "an unrecognized tools.tdb.sync_mode must fail loud instead of being silently ignored"
        );
    }
}
