//! # Store - execute_sparql_update_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use oxirs_arq::{ClearType, SparqlUpdate, SparqlUpdateParser};
use std::collections::{HashMap, HashSet};

impl Store {
    /// Execute a SPARQL update operation.
    ///
    /// Returns: (operation_type, quads_inserted, quads_deleted, affected_graphs)
    ///
    /// # Dispatch strategy
    ///
    /// The operation type is decided from the **parsed SPARQL Update AST**
    /// (via `oxirs_arq::SparqlUpdateParser`), never from a substring search
    /// over the raw query text. This closes a critical data-loss bug where an
    /// `INSERT DATA` whose literal object merely *contained* the substring
    /// `CLEAR`/`DROP`/`ADD` (e.g. the words "nuclear", "unclear", "address")
    /// was misrouted to a destructive graph-management handler that silently
    /// wiped the default graph.
    ///
    /// A raw, unparseable update returns HTTP 400 (`FusekiError::bad_request`)
    /// and NEVER falls through to a destructive default. When the (deliberately
    /// small-subset) AST parser cannot handle an otherwise valid statement
    /// (e.g. `INSERT DATA { GRAPH <g> { … } }`, which the AST parser rejects),
    /// dispatch falls back to an **anchored leading-keyword** classifier that
    /// keys only on the operation keyword at the start of the statement — this
    /// too cannot be fooled by literal content buried later in the query.
    pub(super) fn execute_sparql_update(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        let statements = Self::split_top_level_statements(sparql);
        if statements.is_empty() {
            return Err(FusekiError::bad_request(
                "Empty SPARQL UPDATE request (no operations found)",
            ));
        }

        if statements.len() == 1 {
            return self.dispatch_one_statement(store, &statements[0]);
        }

        // Multiple `;`-separated operations: apply each in order, aggregating
        // the reported counts and affected-graph list.
        let mut total_inserted = 0;
        let mut total_deleted = 0;
        let mut affected: Vec<String> = Vec::new();
        for stmt in &statements {
            let (_label, inserted, deleted, graphs) = self.dispatch_one_statement(store, stmt)?;
            total_inserted += inserted;
            total_deleted += deleted;
            affected.extend(graphs);
        }
        affected.sort();
        affected.dedup();
        Ok(("MULTIPLE", total_inserted, total_deleted, affected))
    }

    /// Classify and execute a single top-level update statement.
    ///
    /// The parsed AST is authoritative for dispatch; only when the AST parser
    /// cannot handle the statement do we fall back to anchored leading-keyword
    /// classification (which is still immune to the substring-misroute bug).
    fn dispatch_one_statement(
        &self,
        store: &mut dyn CoreStore,
        stmt: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        match SparqlUpdateParser::parse_one(stmt) {
            Ok(update) => self.dispatch_parsed_update(store, &update, stmt),
            Err(_) => self.dispatch_by_leading_keyword(store, stmt),
        }
    }

    /// Dispatch on the parsed `SparqlUpdate` variant.
    ///
    /// The AST decides *which* handler runs; the concrete data / graph parsing
    /// is delegated to the existing per-operation handlers (which re-parse the
    /// raw statement with the RDF term parser that correctly preserves literal
    /// datatypes and language tags). CLEAR is handled directly from the parsed
    /// scope so it can never fall through to a destructive default.
    fn dispatch_parsed_update(
        &self,
        store: &mut dyn CoreStore,
        update: &SparqlUpdate,
        stmt: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        match update {
            SparqlUpdate::InsertData(_) => self.execute_insert_data_operation(store, stmt),
            SparqlUpdate::DeleteData(_) => self.execute_delete_data_operation(store, stmt),
            SparqlUpdate::InsertWhere { .. } => self.execute_insert_operation(store, stmt),
            SparqlUpdate::DeleteWhere { .. } => self.execute_delete_where_operation(store, stmt),
            SparqlUpdate::Modify { .. } => self.execute_delete_insert_operation(store, stmt),
            SparqlUpdate::CreateGraph { .. } => self.execute_create_operation(store, stmt),
            SparqlUpdate::DropGraph { .. } => self.execute_drop_operation(store, stmt),
            SparqlUpdate::ClearGraph {
                clear_type, iri, ..
            } => self.execute_clear_operation_ast(store, clear_type, iri.as_deref()),
            SparqlUpdate::CopyGraph { .. } => self.execute_copy_operation(store, stmt),
            SparqlUpdate::MoveGraph { .. } => self.execute_move_operation(store, stmt),
            SparqlUpdate::AddGraph { .. } => self.execute_add_operation(store, stmt),
            SparqlUpdate::Load { .. } => self.execute_load_operation(store, stmt),
        }
    }

    /// Anchored fallback classifier for statements the AST parser rejects
    /// (e.g. `INSERT DATA { GRAPH <g> { … } }`). Keys strictly on the leading
    /// operation keyword after stripping `PREFIX`/`BASE` declarations, so a
    /// literal containing `clear`/`drop` elsewhere in the statement can never
    /// select a destructive operation. An unrecognized leading keyword returns
    /// HTTP 400 rather than silently succeeding or destroying data.
    fn dispatch_by_leading_keyword(
        &self,
        store: &mut dyn CoreStore,
        stmt: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        let body = Self::strip_leading_decls(stmt);
        let tokens: Vec<String> = body
            .split_whitespace()
            .take(2)
            .map(|t| t.trim_end_matches('{').to_uppercase())
            .collect();
        let first = tokens.first().map(String::as_str).unwrap_or("");
        let second = tokens.get(1).map(String::as_str).unwrap_or("");

        match (first, second) {
            ("INSERT", "DATA") => self.execute_insert_data_operation(store, stmt),
            ("DELETE", "DATA") => self.execute_delete_data_operation(store, stmt),
            ("DELETE", "WHERE") => self.execute_delete_where_operation(store, stmt),
            ("DELETE", _) | ("WITH", _) => self.execute_delete_insert_operation(store, stmt),
            ("INSERT", _) => self.execute_insert_operation(store, stmt),
            ("CREATE", _) => self.execute_create_operation(store, stmt),
            ("DROP", _) => self.execute_drop_operation(store, stmt),
            ("CLEAR", _) => self.execute_clear_operation(store, stmt),
            ("COPY", _) => self.execute_copy_operation(store, stmt),
            ("MOVE", _) => self.execute_move_operation(store, stmt),
            ("ADD", _) => self.execute_add_operation(store, stmt),
            ("LOAD", _) => self.execute_load_operation(store, stmt),
            _ => Err(FusekiError::bad_request(format!(
                "Unrecognized or unsupported SPARQL UPDATE operation: {}",
                stmt.trim()
            ))),
        }
    }

    /// Split a SPARQL update string into top-level `;`-separated statements.
    ///
    /// Splits only on semicolons that appear at the top level — outside string
    /// literals (`"…"` / `'…'`), IRI references (`<…>`), and `{ … }` blocks
    /// (Turtle predicate-object lists inside a data block use `;` internally and
    /// must NOT trigger a split).
    fn split_top_level_statements(sparql: &str) -> Vec<String> {
        let mut statements = Vec::new();
        let mut current = String::new();
        let mut in_dquote = false;
        let mut in_squote = false;
        let mut in_iri = false;
        let mut brace_depth: i32 = 0;
        let mut escaped = false;

        for ch in sparql.chars() {
            if escaped {
                current.push(ch);
                escaped = false;
                continue;
            }
            match ch {
                '\\' if in_dquote || in_squote => {
                    current.push(ch);
                    escaped = true;
                }
                '"' if !in_squote && !in_iri => {
                    in_dquote = !in_dquote;
                    current.push(ch);
                }
                '\'' if !in_dquote && !in_iri => {
                    in_squote = !in_squote;
                    current.push(ch);
                }
                '<' if !in_dquote && !in_squote && !in_iri => {
                    in_iri = true;
                    current.push(ch);
                }
                '>' if in_iri => {
                    in_iri = false;
                    current.push(ch);
                }
                '{' if !in_dquote && !in_squote && !in_iri => {
                    brace_depth += 1;
                    current.push(ch);
                }
                '}' if !in_dquote && !in_squote && !in_iri => {
                    brace_depth -= 1;
                    current.push(ch);
                }
                ';' if brace_depth <= 0 && !in_dquote && !in_squote && !in_iri => {
                    let trimmed = current.trim();
                    if !trimmed.is_empty() {
                        statements.push(trimmed.to_string());
                    }
                    current.clear();
                }
                _ => current.push(ch),
            }
        }
        let trimmed = current.trim();
        if !trimmed.is_empty() {
            statements.push(trimmed.to_string());
        }
        statements
    }

    /// Strip leading `PREFIX name: <iri>` and `BASE <iri>` declarations,
    /// returning the remainder of the statement (used only by the anchored
    /// fallback classifier to find the real leading operation keyword).
    fn strip_leading_decls(stmt: &str) -> &str {
        let mut rest = stmt.trim_start();
        loop {
            let upper = rest.to_uppercase();
            if upper.starts_with("PREFIX") || upper.starts_with("BASE") {
                if let Some(gt) = rest.find('>') {
                    rest = rest[gt + 1..].trim_start();
                    continue;
                }
            }
            break;
        }
        rest
    }

    /// Execute a CLEAR operation from its parsed scope.
    ///
    /// Unlike the legacy substring path, this never falls through to a
    /// destructive default: an unspecified/graph scope without an IRI is an
    /// explicit error, and `CLEAR NAMED` clears only named graphs.
    fn execute_clear_operation_ast(
        &self,
        store: &mut dyn CoreStore,
        clear_type: &ClearType,
        iri: Option<&str>,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        match clear_type {
            ClearType::All => {
                let all_quads = store.find_quads(None, None, None, None).map_err(|e| {
                    FusekiError::update_execution(format!("Failed to query all quads: {e}"))
                })?;
                let mut deleted_count = 0;
                for quad in all_quads {
                    if store.remove_quad(&quad).map_err(|e| {
                        FusekiError::update_execution(format!("Failed to remove quad: {e}"))
                    })? {
                        deleted_count += 1;
                    }
                }
                info!("Cleared all graphs: {} quads removed", deleted_count);
                Ok(("CLEAR ALL", 0, deleted_count, vec!["*all*".to_string()]))
            }
            ClearType::Default => self.clear_default_graph(store),
            ClearType::Named => {
                let all_quads = store.find_quads(None, None, None, None).map_err(|e| {
                    FusekiError::update_execution(format!("Failed to query all quads: {e}"))
                })?;
                let named: Vec<Quad> = all_quads
                    .into_iter()
                    .filter(|quad| {
                        !matches!(
                            quad.graph_name(),
                            oxirs_core::model::GraphName::DefaultGraph
                        )
                    })
                    .collect();
                let mut deleted_count = 0;
                for quad in named {
                    if store.remove_quad(&quad).map_err(|e| {
                        FusekiError::update_execution(format!("Failed to remove quad: {e}"))
                    })? {
                        deleted_count += 1;
                    }
                }
                info!("Cleared all named graphs: {} quads removed", deleted_count);
                Ok(("CLEAR NAMED", 0, deleted_count, vec!["*named*".to_string()]))
            }
            ClearType::Graph => {
                let graph_iri = iri.ok_or_else(|| {
                    FusekiError::bad_request("CLEAR GRAPH requires a graph IRI".to_string())
                })?;
                self.clear_named_graph(store, graph_iri)
            }
        }
    }

    /// Execute CLEAR operation (anchored-keyword fallback path).
    ///
    /// Only reached when the AST parser rejected the statement AND the leading
    /// keyword is genuinely `CLEAR`, so this operates on a real CLEAR statement.
    /// It classifies on anchored scope keywords and returns an explicit error
    /// for a malformed CLEAR rather than defaulting to wiping the default graph.
    fn execute_clear_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        let sparql_upper = sparql.to_uppercase();
        if sparql_upper.contains("CLEAR ALL") {
            let all_quads = store.find_quads(None, None, None, None).map_err(|e| {
                FusekiError::update_execution(format!("Failed to query all quads: {e}"))
            })?;
            let mut deleted_count = 0;
            for quad in all_quads {
                if store.remove_quad(&quad).map_err(|e| {
                    FusekiError::update_execution(format!("Failed to remove quad: {e}"))
                })? {
                    deleted_count += 1;
                }
            }
            info!("Cleared all graphs: {} quads removed", deleted_count);
            return Ok(("CLEAR ALL", 0, deleted_count, vec!["*all*".to_string()]));
        }
        if sparql_upper.contains("CLEAR NAMED") {
            return self.execute_clear_operation_ast(store, &ClearType::Named, None);
        }
        if sparql_upper.contains("CLEAR DEFAULT") {
            return self.clear_default_graph(store);
        }
        if sparql_upper.contains("CLEAR GRAPH") {
            if let Some(graph_iri) = self.extract_graph_iri(sparql)? {
                return self.clear_named_graph(store, &graph_iri);
            }
            return Err(FusekiError::bad_request(
                "CLEAR GRAPH requires a valid <graphIRI>".to_string(),
            ));
        }
        Err(FusekiError::bad_request(
            "Invalid CLEAR syntax: expected CLEAR [SILENT] (GRAPH <IRI> | DEFAULT | NAMED | ALL)"
                .to_string(),
        ))
    }

    /// Execute INSERT DATA operation
    fn execute_insert_data_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        let data_block = self.extract_data_block(sparql, "INSERT DATA")?;
        let quads = self.parse_data_block(&data_block)?;
        let affected_graphs = self.extract_graph_names(&quads);
        let inserted_count = bulk_insert_quads(&*store, quads)
            .map_err(|e| FusekiError::update_execution(format!("Failed to insert quads: {e}")))?;
        Ok(("INSERT DATA", inserted_count, 0, affected_graphs))
    }
    /// Execute DELETE DATA operation
    fn execute_delete_data_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        let data_block = self.extract_data_block(sparql, "DELETE DATA")?;
        let quads = self.parse_data_block(&data_block)?;
        let affected_graphs = self.extract_graph_names(&quads);
        let mut deleted_count = 0;
        for quad in quads {
            if store
                .remove_quad(&quad)
                .map_err(|e| FusekiError::update_execution(format!("Failed to remove quad: {e}")))?
            {
                deleted_count += 1;
            }
        }
        Ok(("DELETE DATA", 0, deleted_count, affected_graphs))
    }
    /// Execute simple INSERT operation
    fn execute_insert_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        let data_block = self.extract_data_block(sparql, "INSERT")?;
        let quads = self.parse_data_block(&data_block)?;
        let affected_graphs = self.extract_graph_names(&quads);
        let inserted_count = bulk_insert_quads(&*store, quads)
            .map_err(|e| FusekiError::update_execution(format!("Failed to insert quads: {e}")))?;
        Ok(("INSERT", inserted_count, 0, affected_graphs))
    }
    /// Execute CREATE operation (SPARQL 1.1)
    /// Syntax: CREATE [SILENT] GRAPH <graphIRI>
    fn execute_create_operation(
        &self,
        _store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        info!("Executing CREATE operation: {}", sparql.trim());
        let sparql_upper = sparql.to_uppercase();
        let silent = sparql_upper.contains("SILENT");
        let graph_iri = self.extract_graph_iri_for_management(sparql, "CREATE")?;
        info!(
            "Graph '{}' marked for creation (graphs are created implicitly on data insertion)",
            graph_iri
        );
        if silent {
            Ok(("CREATE SILENT", 0, 0, vec![graph_iri]))
        } else {
            Ok(("CREATE", 0, 0, vec![graph_iri]))
        }
    }
    /// Execute DROP operation (SPARQL 1.1)
    /// Syntax: DROP [SILENT] (GRAPH <graphIRI> | DEFAULT | NAMED | ALL)
    fn execute_drop_operation(
        &self,
        store: &mut dyn CoreStore,
        sparql: &str,
    ) -> FusekiResult<(&'static str, usize, usize, Vec<String>)> {
        info!("Executing DROP operation: {}", sparql.trim());
        let sparql_upper = sparql.to_uppercase();
        let silent = sparql_upper.contains("SILENT");
        if sparql_upper.contains("DROP") && sparql_upper.contains("ALL") {
            let quad_count = store.len().map_err(|e| {
                FusekiError::update_execution(format!("Failed to get store size: {e}"))
            })?;
            store.clear_all().map_err(|e| {
                FusekiError::update_execution(format!("Failed to drop all graphs: {e}"))
            })?;
            info!("Dropped all graphs: {} quads removed", quad_count);
            return if silent {
                Ok(("DROP SILENT ALL", 0, quad_count, vec!["*all*".to_string()]))
            } else {
                Ok(("DROP ALL", 0, quad_count, vec!["*all*".to_string()]))
            };
        }
        if sparql_upper.contains("DROP") && sparql_upper.contains("DEFAULT") {
            return if silent {
                let result = self.clear_default_graph(store)?;
                Ok(("DROP SILENT DEFAULT", result.1, result.2, result.3))
            } else {
                let result = self.clear_default_graph(store)?;
                Ok(("DROP DEFAULT", result.1, result.2, result.3))
            };
        }
        if sparql_upper.contains("DROP") && sparql_upper.contains("NAMED") {
            let all_quads_raw = store.find_quads(None, None, None, None).map_err(|e| {
                FusekiError::update_execution(format!("Failed to query all quads: {e}"))
            })?;
            let all_quads: Vec<Quad> = all_quads_raw
                .into_iter()
                .filter(|quad| {
                    !matches!(
                        quad.graph_name(),
                        oxirs_core::model::GraphName::DefaultGraph
                    )
                })
                .collect();
            let deleted_count = all_quads.len();
            for quad in all_quads {
                store.remove_quad(&quad).map_err(|e| {
                    FusekiError::update_execution(format!("Failed to remove quad: {e}"))
                })?;
            }
            info!("Dropped all named graphs: {} quads removed", deleted_count);
            return if silent {
                Ok((
                    "DROP SILENT NAMED",
                    0,
                    deleted_count,
                    vec!["*named*".to_string()],
                ))
            } else {
                Ok(("DROP NAMED", 0, deleted_count, vec!["*named*".to_string()]))
            };
        }
        if let Ok(graph_iri) = self.extract_graph_iri_for_management(sparql, "DROP") {
            let result = self.clear_named_graph(store, &graph_iri)?;
            return if silent {
                Ok(("DROP SILENT GRAPH", result.1, result.2, result.3))
            } else {
                Ok(("DROP GRAPH", result.1, result.2, result.3))
            };
        }
        Err(FusekiError::update_execution(
            "Invalid DROP syntax: expected DROP [SILENT] (GRAPH <IRI> | DEFAULT | NAMED | ALL)"
                .to_string(),
        ))
    }
}

#[cfg(test)]
mod update_dispatch_tests {
    use crate::store::Store;

    /// INSERT DATA whose literal object contains the substring "clear" must
    /// insert the data and MUST NOT be misrouted to a destructive CLEAR that
    /// wipes the default graph. This is the core P0 regression.
    #[test]
    fn insert_data_with_clear_literal_does_not_wipe_default_graph() {
        let store = Store::new().expect("create store");
        store
            .update(
                "INSERT DATA { <http://example.org/s> <http://example.org/p> \"nuclear reactor\" . }",
            )
            .expect("insert with 'clear' literal should succeed");
        // A second row so we can prove nothing was wiped.
        store
            .update(
                "INSERT DATA { <http://example.org/s2> <http://example.org/p2> \"all clear now\" . }",
            )
            .expect("second insert should succeed");

        let count = store.count_triples("default");
        assert_eq!(
            count, 2,
            "both triples must survive; the 'clear' literal must not trigger a destructive clear"
        );
    }

    /// INSERT DATA with literals containing drop/create/add substrings must not
    /// be misrouted to graph-management operations.
    #[test]
    fn insert_data_with_drop_create_add_literals_inserts() {
        let store = Store::new().expect("create store");
        store
            .update(
                "INSERT DATA { <http://example.org/a> <http://example.org/p> \"drop table\" . }",
            )
            .expect("'drop' literal insert");
        store
            .update(
                "INSERT DATA { <http://example.org/b> <http://example.org/p> \"create index\" . }",
            )
            .expect("'create' literal insert");
        store
            .update("INSERT DATA { <http://example.org/c> <http://example.org/p> \"add more\" . }")
            .expect("'add' literal insert");

        let count = store.count_triples("default");
        assert_eq!(
            count, 3,
            "all three literal-bearing triples must be present"
        );
    }

    /// An unparseable / unrecognized update must return an error (HTTP 400),
    /// never silently succeed or fall through to a destructive default.
    #[test]
    fn unrecognized_update_returns_error() {
        let store = Store::new().expect("create store");
        let result = store.update("FROBNICATE <http://example.org/g>");
        assert!(
            result.is_err(),
            "unrecognized update keyword must be rejected, got: {result:?}"
        );
    }

    /// A genuine CLEAR DEFAULT still clears the default graph.
    #[test]
    fn genuine_clear_default_still_works() {
        let store = Store::new().expect("create store");
        store
            .update("INSERT DATA { <http://example.org/s> <http://example.org/p> \"v\" . }")
            .expect("seed insert");
        let result = store.update("CLEAR DEFAULT").expect("clear default");
        assert_eq!(result.stats.operation_type, "CLEAR DEFAULT");
        let count = store.count_triples("default");
        assert_eq!(count, 0, "CLEAR DEFAULT must empty the default graph");
    }

    /// Regression: `execute_insert_data_operation` was switched from a
    /// per-quad `insert_quad` loop to a single `bulk_insert_quads` call so
    /// large `INSERT DATA` payloads take one write lock and one fsync
    /// instead of one per quad. The reported `quads_inserted` count must
    /// keep matching the old per-quad-loop semantics: a quad already
    /// duplicated *within the same batch* (or already present in the store)
    /// is not counted twice.
    #[test]
    fn insert_data_bulk_path_dedups_like_the_old_per_quad_loop() {
        let store = Store::new().expect("create store");

        // Duplicate triple appears twice in one INSERT DATA batch. Note:
        // `parse_data_block` (via `parse_ntriples_document`) splits the block
        // into individual statements on top-level `.` terminators -- a `.`
        // that is not inside an `<IRI>`, a quoted literal, or a `#` comment --
        // so several `.`-separated triples may share one physical line just
        // as well as being on separate lines; either form works here. Any
        // statement that fails to parse aborts the whole block (no partial
        // apply), so putting the triples on separate lines below is purely
        // for readability, not a parser requirement.
        let result = store
            .update(
                "INSERT DATA {\n\
                   <http://example.org/s> <http://example.org/p> \"v\" .\n\
                   <http://example.org/s> <http://example.org/p> \"v\" .\n\
                   <http://example.org/s2> <http://example.org/p2> \"v2\" .\n\
                 }",
            )
            .expect("insert data with in-batch duplicate should succeed");
        assert_eq!(
            result.stats.quads_inserted, 2,
            "the in-batch duplicate must not be double-counted"
        );
        assert_eq!(store.count_triples("default"), 2);

        // Re-inserting the same data again must report zero *new* insertions,
        // even though the batch itself has two distinct quads.
        let result2 = store
            .update(
                "INSERT DATA {\n\
                   <http://example.org/s> <http://example.org/p> \"v\" .\n\
                   <http://example.org/s2> <http://example.org/p2> \"v2\" .\n\
                 }",
            )
            .expect("re-insert of already-present data should succeed");
        assert_eq!(
            result2.stats.quads_inserted, 0,
            "quads already present in the store must not be recounted as new"
        );
        assert_eq!(
            store.count_triples("default"),
            2,
            "store contents must be unchanged by the redundant insert"
        );
    }

    /// Regression: the plain `INSERT` fallback path (distinct from
    /// `INSERT DATA`) was likewise switched to `bulk_insert_quads`; it must
    /// keep reporting an accurate new-insert count for a batch containing an
    /// in-batch duplicate.
    #[test]
    fn plain_insert_bulk_path_dedups_like_the_old_per_quad_loop() {
        let store = Store::new().expect("create store");
        let result = store
            .update(
                "INSERT {\n\
                   <http://example.org/a> <http://example.org/p> \"x\" .\n\
                   <http://example.org/a> <http://example.org/p> \"x\" .\n\
                 }",
            )
            .expect("plain INSERT with in-batch duplicate should succeed");
        assert_eq!(
            result.stats.quads_inserted, 1,
            "the in-batch duplicate must not be double-counted"
        );
        assert_eq!(store.count_triples("default"), 1);
    }
}
