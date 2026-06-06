//! Execution engine for SPARQL CONSTRUCT queries.
//!
//! [`ConstructEngine`] evaluates the WHERE clause to obtain solution mappings,
//! instantiates the CONSTRUCT template per solution, and applies deduplication
//! and result limits.

use super::construct_parser::parse_construct_query;
use super::construct_types::{ConstructConfig, ConstructQuery, ConstructStats};
use super::evaluate_pattern;
use crate::error::WasmResult;
use crate::store::OxiRSStore;
use crate::Triple;
use std::collections::{HashMap, HashSet};

/// Engine for executing SPARQL CONSTRUCT queries.
pub struct ConstructEngine {
    pub(crate) config: ConstructConfig,
}

impl ConstructEngine {
    /// Create a new CONSTRUCT engine with default configuration.
    pub fn new() -> Self {
        Self {
            config: ConstructConfig::default(),
        }
    }

    /// Create a new CONSTRUCT engine with the given configuration.
    pub fn with_config(config: ConstructConfig) -> Self {
        Self { config }
    }

    /// Execute a CONSTRUCT query against the store.
    pub fn execute(
        &self,
        sparql: &str,
        store: &OxiRSStore,
    ) -> WasmResult<(Vec<Triple>, ConstructStats)> {
        let query = parse_construct_query(sparql)?;
        self.execute_parsed(&query, store)
    }

    /// Execute a pre-parsed CONSTRUCT query.
    pub fn execute_parsed(
        &self,
        query: &ConstructQuery,
        store: &OxiRSStore,
    ) -> WasmResult<(Vec<Triple>, ConstructStats)> {
        let mut stats = ConstructStats {
            template_triple_count: query.template.len(),
            ..Default::default()
        };

        // Evaluate WHERE clause to get solution mappings
        let mut solutions: Vec<HashMap<String, String>> = vec![HashMap::new()];
        for pattern in &query.where_patterns {
            solutions = evaluate_pattern(pattern, solutions, store)?;
        }

        // Apply OFFSET
        if let Some(offset) = query.offset {
            if offset >= solutions.len() {
                solutions.clear();
            } else {
                solutions = solutions.into_iter().skip(offset).collect();
            }
        }

        // Apply LIMIT
        if let Some(limit) = query.limit {
            solutions.truncate(limit);
        }

        stats.solution_count = solutions.len();

        // Instantiate template for each solution
        let mut blank_counter: u64 = 0;
        let mut all_triples: Vec<(String, String, String)> = Vec::new();

        for solution in &solutions {
            // Each solution gets its own blank node scope
            let mut blank_scope: HashMap<String, String> = HashMap::new();

            for template_triple in &query.template {
                let s_opt = template_triple.subject.instantiate(
                    solution,
                    &mut blank_scope,
                    &mut blank_counter,
                    &self.config.blank_node_prefix,
                );
                let p_opt = template_triple.predicate.instantiate(
                    solution,
                    &mut blank_scope,
                    &mut blank_counter,
                    &self.config.blank_node_prefix,
                );
                let o_opt = template_triple.object.instantiate(
                    solution,
                    &mut blank_scope,
                    &mut blank_counter,
                    &self.config.blank_node_prefix,
                );

                match (s_opt, p_opt, o_opt) {
                    (Some(s), Some(p), Some(o)) => {
                        all_triples.push((s, p, o));
                    }
                    _ => {
                        stats.skipped_unbound += 1;
                    }
                }
            }
        }

        stats.raw_triple_count = all_triples.len();
        stats.blank_nodes_generated = blank_counter;

        // Deduplicate if configured
        let result_triples = if self.config.deduplicate {
            let mut seen: HashSet<(String, String, String)> = HashSet::new();
            let mut deduped = Vec::new();
            for triple in all_triples {
                if seen.insert(triple.clone()) {
                    deduped.push(Triple::new(&triple.0, &triple.1, &triple.2));
                }
            }
            deduped
        } else {
            all_triples
                .into_iter()
                .map(|(s, p, o)| Triple::new(&s, &p, &o))
                .collect()
        };

        stats.deduped_triple_count = result_triples.len();

        // Apply max_triples limit
        let result_triples = if let Some(max) = self.config.max_triples {
            result_triples.into_iter().take(max).collect()
        } else {
            result_triples
        };

        Ok((result_triples, stats))
    }
}

impl Default for ConstructEngine {
    fn default() -> Self {
        Self::new()
    }
}
