//! # SPARQL Update Delta Support
//!
//! Delta computation and streaming for SPARQL Updates.
//!
//! This module provides sophisticated delta computation for SPARQL Updates,
//! converting update operations into fine-grained change events and RDF Patches.
//! It supports tracking changes at the triple level and provides efficient
//! streaming of update operations.

use crate::{PatchOperation, RdfPatch, StreamEvent};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, error, warn};

/// Delta computation for SPARQL Updates with advanced parsing
pub struct DeltaComputer {
    enable_optimization: bool,
    track_provenance: bool,
    current_context: Option<String>,
    operation_counter: u64,
}

impl DeltaComputer {
    pub fn new() -> Self {
        Self {
            enable_optimization: true,
            track_provenance: false,
            current_context: None,
            operation_counter: 0,
        }
    }

    pub fn with_optimization(mut self, enabled: bool) -> Self {
        self.enable_optimization = enabled;
        self
    }

    pub fn with_provenance(mut self, enabled: bool) -> Self {
        self.track_provenance = enabled;
        self
    }

    pub fn set_context(&mut self, context: Option<String>) {
        self.current_context = context;
    }

    /// Compute delta from SPARQL Update
    pub fn compute_delta(&mut self, update: &str) -> Result<Vec<StreamEvent>> {
        let mut events = Vec::new();

        // Parse the SPARQL update to identify operations
        let operations = self.parse_sparql_update(update)?;

        for operation in operations {
            let mut operation_events = self.process_update_operation(&operation)?;
            events.append(&mut operation_events);
        }

        if self.enable_optimization {
            events = self.optimize_events(events);
        }

        debug!("Computed {} delta events from SPARQL update", events.len());
        Ok(events)
    }

    /// Convert delta to RDF Patch
    pub fn delta_to_patch(&self, events: &[StreamEvent]) -> Result<RdfPatch> {
        let mut patch = RdfPatch::new();

        for event in events {
            let operation = match event {
                StreamEvent::TripleAdded {
                    subject,
                    predicate,
                    object,
                } => PatchOperation::Add {
                    subject: subject.clone(),
                    predicate: predicate.clone(),
                    object: object.clone(),
                },
                StreamEvent::TripleRemoved {
                    subject,
                    predicate,
                    object,
                } => PatchOperation::Delete {
                    subject: subject.clone(),
                    predicate: predicate.clone(),
                    object: object.clone(),
                },
                StreamEvent::GraphCleared { graph } => {
                    if let Some(graph_uri) = graph {
                        PatchOperation::DeleteGraph {
                            graph: graph_uri.clone(),
                        }
                    } else {
                        // Default graph clear - we'll represent this as a special operation
                        continue;
                    }
                }
                StreamEvent::SparqlUpdate { .. } => {
                    // Skip SPARQL update events in patch conversion
                    continue;
                }
            };

            patch.add_operation(operation);
        }

        debug!(
            "Converted {} events to RDF patch with {} operations",
            events.len(),
            patch.operations.len()
        );
        Ok(patch)
    }

    fn parse_sparql_update(&mut self, update: &str) -> Result<Vec<UpdateOperation>> {
        let mut operations = Vec::new();
        let normalized = self.normalize_sparql(update);

        // Split into individual operations (simplified)
        let statements = self.split_statements(&normalized);

        for statement in statements {
            if let Some(operation) = self.parse_statement(&statement)? {
                operations.push(operation);
            }
        }

        Ok(operations)
    }

    fn normalize_sparql(&self, update: &str) -> String {
        // Basic normalization - remove extra whitespace, normalize line endings
        let re = Regex::new(r"\s+").unwrap();
        re.replace_all(update.trim(), " ").to_string()
    }

    fn split_statements(&self, update: &str) -> Vec<String> {
        // Split on semicolons that aren't inside quotes or braces
        let mut statements = Vec::new();
        let mut current = String::new();
        let mut in_quotes = false;
        let mut brace_depth = 0;
        let mut chars = update.chars().peekable();

        while let Some(c) = chars.next() {
            match c {
                '"' => {
                    in_quotes = !in_quotes;
                    current.push(c);
                }
                '{' if !in_quotes => {
                    brace_depth += 1;
                    current.push(c);
                }
                '}' if !in_quotes => {
                    brace_depth -= 1;
                    current.push(c);
                }
                ';' if !in_quotes && brace_depth == 0 => {
                    if !current.trim().is_empty() {
                        statements.push(current.trim().to_string());
                        current.clear();
                    }
                }
                _ => {
                    current.push(c);
                }
            }
        }

        if !current.trim().is_empty() {
            statements.push(current.trim().to_string());
        }

        statements
    }

    fn parse_statement(&mut self, statement: &str) -> Result<Option<UpdateOperation>> {
        let upper = statement.to_uppercase();

        if upper.contains("INSERT DATA") {
            self.parse_insert_data(statement)
        } else if upper.contains("DELETE DATA") {
            self.parse_delete_data(statement)
        } else if upper.contains("INSERT") && upper.contains("WHERE") {
            self.parse_insert_where(statement)
        } else if upper.contains("DELETE") && upper.contains("WHERE") {
            self.parse_delete_where(statement)
        } else if upper.contains("DELETE") && upper.contains("INSERT") {
            self.parse_delete_insert(statement)
        } else if upper.contains("CLEAR") {
            self.parse_clear(statement)
        } else if upper.contains("DROP") {
            self.parse_drop(statement)
        } else if upper.contains("CREATE") {
            self.parse_create(statement)
        } else if upper.contains("LOAD") {
            self.parse_load(statement)
        } else {
            warn!("Unknown SPARQL update operation: {}", statement);
            Ok(None)
        }
    }

    fn parse_insert_data(&mut self, statement: &str) -> Result<Option<UpdateOperation>> {
        if let Some(data_block) = self.extract_data_block(statement, "INSERT DATA")? {
            let triples = self.parse_triples(&data_block)?;
            Ok(Some(UpdateOperation::InsertData { triples }))
        } else {
            Ok(None)
        }
    }

    fn parse_delete_data(&mut self, statement: &str) -> Result<Option<UpdateOperation>> {
        if let Some(data_block) = self.extract_data_block(statement, "DELETE DATA")? {
            let triples = self.parse_triples(&data_block)?;
            Ok(Some(UpdateOperation::DeleteData { triples }))
        } else {
            Ok(None)
        }
    }

    fn parse_insert_where(&mut self, statement: &str) -> Result<Option<UpdateOperation>> {
        // Simplified parsing for INSERT ... WHERE
        let insert_block = self.extract_data_block(statement, "INSERT")?;
        let where_block = self.extract_data_block(statement, "WHERE")?;

        if let (Some(insert), Some(where_clause)) = (insert_block, where_block) {
            let insert_triples = self.parse_triples(&insert)?;
            Ok(Some(UpdateOperation::InsertWhere {
                insert: insert_triples,
                where_clause,
            }))
        } else {
            Ok(None)
        }
    }

    fn parse_delete_where(&mut self, statement: &str) -> Result<Option<UpdateOperation>> {
        if let Some(where_block) = self.extract_data_block(statement, "WHERE")? {
            let delete_patterns = self.parse_triples(&where_block)?;
            Ok(Some(UpdateOperation::DeleteWhere {
                patterns: delete_patterns,
            }))
        } else {
            Ok(None)
        }
    }

    fn parse_delete_insert(&mut self, statement: &str) -> Result<Option<UpdateOperation>> {
        let delete_block = self.extract_data_block(statement, "DELETE")?;
        let insert_block = self.extract_data_block(statement, "INSERT")?;
        let where_block = self.extract_data_block(statement, "WHERE")?;

        let delete_triples = if let Some(delete) = delete_block {
            self.parse_triples(&delete)?
        } else {
            Vec::new()
        };

        let insert_triples = if let Some(insert) = insert_block {
            self.parse_triples(&insert)?
        } else {
            Vec::new()
        };

        Ok(Some(UpdateOperation::DeleteInsert {
            delete: delete_triples,
            insert: insert_triples,
            where_clause: where_block,
        }))
    }

    fn parse_clear(&mut self, statement: &str) -> Result<Option<UpdateOperation>> {
        let upper = statement.to_uppercase();
        if upper.contains("CLEAR ALL") {
            Ok(Some(UpdateOperation::ClearAll))
        } else if upper.contains("CLEAR DEFAULT") {
            Ok(Some(UpdateOperation::ClearDefault))
        } else {
            // Extract graph URI
            let graph = self.extract_graph_uri(statement)?;
            Ok(Some(UpdateOperation::ClearGraph { graph }))
        }
    }

    fn parse_drop(&mut self, statement: &str) -> Result<Option<UpdateOperation>> {
        let graph = self.extract_graph_uri(statement)?;
        Ok(Some(UpdateOperation::DropGraph { graph }))
    }

    fn parse_create(&mut self, statement: &str) -> Result<Option<UpdateOperation>> {
        let graph = self.extract_graph_uri(statement)?;
        Ok(Some(UpdateOperation::CreateGraph { graph }))
    }

    fn parse_load(&mut self, statement: &str) -> Result<Option<UpdateOperation>> {
        // Extract URL and optional graph
        let parts: Vec<&str> = statement.split_whitespace().collect();
        if parts.len() >= 2 {
            let url = parts[1].trim_matches('<').trim_matches('>').to_string();
            let graph = if parts.len() > 3 && parts[2].to_uppercase() == "INTO" {
                Some(parts[3].trim_matches('<').trim_matches('>').to_string())
            } else {
                None
            };
            Ok(Some(UpdateOperation::Load { url, graph }))
        } else {
            Err(anyhow!("Invalid LOAD statement: {}", statement))
        }
    }

    fn extract_data_block(&self, statement: &str, keyword: &str) -> Result<Option<String>> {
        let upper = statement.to_uppercase();
        let keyword_upper = keyword.to_uppercase();

        if let Some(start) = upper.find(&keyword_upper) {
            let after_keyword = &statement[start + keyword.len()..].trim();

            if let Some(open_brace) = after_keyword.find('{') {
                let from_brace = &after_keyword[open_brace + 1..];

                // Find matching closing brace
                let mut brace_count = 1;
                let mut end_pos = 0;

                for (i, c) in from_brace.char_indices() {
                    match c {
                        '{' => brace_count += 1,
                        '}' => {
                            brace_count -= 1;
                            if brace_count == 0 {
                                end_pos = i;
                                break;
                            }
                        }
                        _ => {}
                    }
                }

                if brace_count == 0 {
                    Ok(Some(from_brace[..end_pos].trim().to_string()))
                } else {
                    Err(anyhow!("Unmatched braces in {}", keyword))
                }
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    fn extract_graph_uri(&self, statement: &str) -> Result<Option<String>> {
        // Simple graph URI extraction - look for GRAPH <uri> pattern
        let re = Regex::new(r"GRAPH\s+<([^>]+)>").unwrap();
        if let Some(captures) = re.captures(&statement.to_uppercase()) {
            if let Some(uri) = captures.get(1) {
                return Ok(Some(uri.as_str().to_string()));
            }
        }

        // Look for bare URI after keyword
        let parts: Vec<&str> = statement.split_whitespace().collect();
        for part in parts {
            if part.starts_with('<') && part.ends_with('>') {
                return Ok(Some(part[1..part.len() - 1].to_string()));
            }
        }

        Ok(None)
    }

    fn parse_triples(&self, data: &str) -> Result<Vec<Triple>> {
        let mut triples = Vec::new();

        // Simple triple parsing - split by periods and parse each line
        for line in data.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Remove trailing period
            let line = line.trim_end_matches('.');

            // Split by whitespace to get subject, predicate, object
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                let subject = parts[0].to_string();
                let predicate = parts[1].to_string();
                let object = parts[2..].join(" "); // Handle multi-word objects

                triples.push(Triple {
                    subject,
                    predicate,
                    object,
                });
            }
        }

        Ok(triples)
    }

    fn process_update_operation(
        &mut self,
        operation: &UpdateOperation,
    ) -> Result<Vec<StreamEvent>> {
        let mut events = Vec::new();
        self.operation_counter += 1;

        match operation {
            UpdateOperation::InsertData { triples } => {
                for triple in triples {
                    events.push(StreamEvent::TripleAdded {
                        subject: triple.subject.clone(),
                        predicate: triple.predicate.clone(),
                        object: triple.object.clone(),
                    });
                }
            }
            UpdateOperation::DeleteData { triples } => {
                for triple in triples {
                    events.push(StreamEvent::TripleRemoved {
                        subject: triple.subject.clone(),
                        predicate: triple.predicate.clone(),
                        object: triple.object.clone(),
                    });
                }
            }
            UpdateOperation::InsertWhere { insert, .. } => {
                // For INSERT WHERE, we generate add events for the insert patterns
                // In a real implementation, we'd need to evaluate the WHERE clause
                for triple in insert {
                    events.push(StreamEvent::TripleAdded {
                        subject: triple.subject.clone(),
                        predicate: triple.predicate.clone(),
                        object: triple.object.clone(),
                    });
                }
            }
            UpdateOperation::DeleteWhere { patterns } => {
                // For DELETE WHERE, we generate remove events for matching patterns
                for pattern in patterns {
                    events.push(StreamEvent::TripleRemoved {
                        subject: pattern.subject.clone(),
                        predicate: pattern.predicate.clone(),
                        object: pattern.object.clone(),
                    });
                }
            }
            UpdateOperation::DeleteInsert { delete, insert, .. } => {
                // Process deletes first, then inserts
                for triple in delete {
                    events.push(StreamEvent::TripleRemoved {
                        subject: triple.subject.clone(),
                        predicate: triple.predicate.clone(),
                        object: triple.object.clone(),
                    });
                }
                for triple in insert {
                    events.push(StreamEvent::TripleAdded {
                        subject: triple.subject.clone(),
                        predicate: triple.predicate.clone(),
                        object: triple.object.clone(),
                    });
                }
            }
            UpdateOperation::ClearAll => {
                events.push(StreamEvent::GraphCleared { graph: None });
            }
            UpdateOperation::ClearDefault => {
                events.push(StreamEvent::GraphCleared { graph: None });
            }
            UpdateOperation::ClearGraph { graph } => {
                events.push(StreamEvent::GraphCleared {
                    graph: graph.clone(),
                });
            }
            UpdateOperation::DropGraph { graph } => {
                events.push(StreamEvent::GraphCleared {
                    graph: graph.clone(),
                });
            }
            UpdateOperation::CreateGraph { .. } => {
                // Graph creation doesn't generate data events
            }
            UpdateOperation::Load { .. } => {
                // LOAD operations would generate events based on loaded data
                // For now, we just record the operation
                events.push(StreamEvent::SparqlUpdate {
                    query: format!("Operation #{}: {:?}", self.operation_counter, operation),
                });
            }
        }

        Ok(events)
    }

    fn optimize_events(&self, events: Vec<StreamEvent>) -> Vec<StreamEvent> {
        // Remove redundant add/remove pairs
        let mut optimized = Vec::new();
        let mut seen_operations = HashSet::new();
        let original_count = events.len();

        for event in events {
            let event_key = match &event {
                StreamEvent::TripleAdded {
                    subject,
                    predicate,
                    object,
                }
                | StreamEvent::TripleRemoved {
                    subject,
                    predicate,
                    object,
                } => {
                    format!("{}|{}|{}", subject, predicate, object)
                }
                StreamEvent::GraphCleared { graph } => {
                    format!("graph_clear|{:?}", graph)
                }
                StreamEvent::SparqlUpdate { query } => {
                    format!("sparql|{}", query)
                }
            };

            if !seen_operations.contains(&event_key) {
                seen_operations.insert(event_key);
                optimized.push(event);
            }
        }

        debug!("Optimized {} events to {}", original_count, optimized.len());
        optimized
    }
}

impl Default for DeltaComputer {
    fn default() -> Self {
        Self::new()
    }
}

/// Parsed SPARQL Update operations
#[derive(Debug, Clone)]
enum UpdateOperation {
    InsertData {
        triples: Vec<Triple>,
    },
    DeleteData {
        triples: Vec<Triple>,
    },
    InsertWhere {
        insert: Vec<Triple>,
        where_clause: String,
    },
    DeleteWhere {
        patterns: Vec<Triple>,
    },
    DeleteInsert {
        delete: Vec<Triple>,
        insert: Vec<Triple>,
        where_clause: Option<String>,
    },
    ClearAll,
    ClearDefault,
    ClearGraph {
        graph: Option<String>,
    },
    DropGraph {
        graph: Option<String>,
    },
    CreateGraph {
        graph: Option<String>,
    },
    Load {
        url: String,
        graph: Option<String>,
    },
}

/// Simple triple representation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Triple {
    subject: String,
    predicate: String,
    object: String,
}

/// Delta stream processor with batching and buffering
pub struct DeltaProcessor {
    computer: DeltaComputer,
    buffer: Vec<StreamEvent>,
    batch_size: usize,
    max_buffer_age: chrono::Duration,
    last_flush: DateTime<Utc>,
    stats: ProcessorStats,
}

#[derive(Debug, Default)]
struct ProcessorStats {
    updates_processed: u64,
    events_generated: u64,
    batches_sent: u64,
    last_activity: Option<DateTime<Utc>>,
}

impl DeltaProcessor {
    pub fn new() -> Self {
        Self {
            computer: DeltaComputer::new(),
            buffer: Vec::new(),
            batch_size: 100,
            max_buffer_age: chrono::Duration::seconds(30),
            last_flush: Utc::now(),
            stats: ProcessorStats::default(),
        }
    }

    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    pub fn with_max_buffer_age(mut self, duration: chrono::Duration) -> Self {
        self.max_buffer_age = duration;
        self
    }

    /// Process SPARQL Update and generate stream events
    pub async fn process_update(&mut self, update: &str) -> Result<Vec<StreamEvent>> {
        let events = self.computer.compute_delta(update)?;

        self.stats.updates_processed += 1;
        self.stats.events_generated += events.len() as u64;
        self.stats.last_activity = Some(Utc::now());

        // Add to buffer
        for event in &events {
            self.buffer.push(event.clone());
        }

        // Check if we should flush
        let should_flush = self.buffer.len() >= self.batch_size
            || Utc::now() - self.last_flush > self.max_buffer_age;

        if should_flush {
            Ok(self.flush())
        } else {
            Ok(events)
        }
    }

    /// Force flush buffered events
    pub fn flush(&mut self) -> Vec<StreamEvent> {
        let events = self.buffer.clone();
        self.buffer.clear();
        self.last_flush = Utc::now();

        if !events.is_empty() {
            self.stats.batches_sent += 1;
            debug!("Flushed {} buffered events", events.len());
        }

        events
    }

    /// Get processor statistics
    pub fn get_stats(&self) -> &ProcessorStats {
        &self.stats
    }

    /// Convert multiple updates to a single patch
    pub async fn updates_to_patch(&mut self, updates: &[String]) -> Result<RdfPatch> {
        let mut all_events = Vec::new();

        for update in updates {
            let events = self.computer.compute_delta(update)?;
            all_events.extend(events);
        }

        self.computer.delta_to_patch(&all_events)
    }
}

impl Default for DeltaProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch processor for handling multiple updates efficiently
pub struct BatchDeltaProcessor {
    processors: Vec<DeltaProcessor>,
    current_processor: usize,
    round_robin: bool,
}

impl BatchDeltaProcessor {
    pub fn new(num_processors: usize) -> Self {
        let mut processors = Vec::new();
        for _ in 0..num_processors {
            processors.push(DeltaProcessor::new());
        }

        Self {
            processors,
            current_processor: 0,
            round_robin: true,
        }
    }

    pub async fn process_updates(&mut self, updates: &[String]) -> Result<Vec<StreamEvent>> {
        let mut all_events = Vec::new();

        for update in updates {
            let processor_idx = if self.round_robin {
                let idx = self.current_processor;
                self.current_processor = (self.current_processor + 1) % self.processors.len();
                idx
            } else {
                0 // Use first processor for sequential processing
            };

            let events = self.processors[processor_idx]
                .process_update(update)
                .await?;
            all_events.extend(events);
        }

        Ok(all_events)
    }

    pub fn flush_all(&mut self) -> Vec<StreamEvent> {
        let mut all_events = Vec::new();
        for processor in &mut self.processors {
            let events = processor.flush();
            all_events.extend(events);
        }
        all_events
    }
}
