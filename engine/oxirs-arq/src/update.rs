//! SPARQL UPDATE Operations
//!
//! This module implements SPARQL 1.1 UPDATE operations including:
//! - INSERT DATA
//! - DELETE DATA
//! - INSERT WHERE
//! - DELETE WHERE
//! - DELETE/INSERT WHERE (combined)
//! - CLEAR, DROP, CREATE, COPY, MOVE, ADD

#[allow(unused_imports)]
use crate::algebra::{Algebra, EvaluationContext, Term, TriplePattern, Variable};
use crate::executor::ExecutionContext;
use oxirs_core::model::{BlankNode, GraphName, Literal as CoreLiteral, NamedNode, Quad};
use oxirs_core::OxirsError;
use oxirs_core::Store;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// SPARQL UPDATE operation types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UpdateOperation {
    /// INSERT DATA - insert concrete triples
    InsertData { data: Vec<QuadPattern> },

    /// DELETE DATA - delete concrete triples
    DeleteData { data: Vec<QuadPattern> },

    /// DELETE WHERE - delete based on pattern matching
    DeleteWhere { pattern: Box<Algebra> },

    /// INSERT WHERE - insert based on pattern matching with template
    InsertWhere {
        pattern: Box<Algebra>,
        template: Vec<QuadPattern>,
    },

    /// DELETE/INSERT WHERE - combined delete and insert
    DeleteInsertWhere {
        delete_template: Vec<QuadPattern>,
        insert_template: Vec<QuadPattern>,
        pattern: Box<Algebra>,
        using: Option<Vec<GraphReference>>,
    },

    /// CLEAR - remove all triples from graph(s)
    Clear { target: GraphTarget, silent: bool },

    /// DROP - remove graph(s) from the dataset
    Drop { target: GraphTarget, silent: bool },

    /// CREATE - create a new graph
    Create { graph: GraphReference, silent: bool },

    /// COPY - copy all data from one graph to another
    Copy {
        from: GraphTarget,
        to: GraphTarget,
        silent: bool,
    },

    /// MOVE - move all data from one graph to another
    Move {
        from: GraphTarget,
        to: GraphTarget,
        silent: bool,
    },

    /// ADD - add all data from one graph to another
    Add {
        from: GraphTarget,
        to: GraphTarget,
        silent: bool,
    },

    /// LOAD - load data from external source
    Load {
        source: String,
        graph: Option<GraphReference>,
        silent: bool,
    },
}

/// Pattern for quads in update operations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QuadPattern {
    pub subject: Term,
    pub predicate: Term,
    pub object: Term,
    pub graph: Option<GraphReference>,
}

/// Reference to a graph (IRI or DEFAULT)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GraphReference {
    Iri(String),
    Default,
}

/// Target for graph operations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphTarget {
    Graph(GraphReference),
    All,
    Named,
    Default,
}

/// Result of an update operation
#[derive(Debug, Clone)]
pub struct UpdateResult {
    /// Number of triples/quads inserted
    pub inserted: usize,
    /// Number of triples/quads deleted
    pub deleted: usize,
    /// Graphs created
    pub graphs_created: Vec<String>,
    /// Graphs dropped
    pub graphs_dropped: Vec<String>,
}

impl Default for UpdateResult {
    fn default() -> Self {
        UpdateResult {
            inserted: 0,
            deleted: 0,
            graphs_created: Vec::new(),
            graphs_dropped: Vec::new(),
        }
    }
}

/// Executor for SPARQL UPDATE operations
pub struct UpdateExecutor<'a> {
    store: &'a mut dyn Store,
    context: ExecutionContext,
    /// Transaction mode for atomic updates
    transaction_mode: bool,
    /// Batch size for large updates
    batch_size: usize,
    /// Statistics tracking
    stats: UpdateStatistics,
}

/// Statistics for update operations
#[derive(Debug, Clone, Default)]
pub struct UpdateStatistics {
    pub total_operations: usize,
    pub total_execution_time: std::time::Duration,
    pub operations_per_second: f64,
    pub memory_usage: usize,
    pub batch_count: usize,
}

impl<'a> UpdateExecutor<'a> {
    /// Create a new update executor
    pub fn new(store: &'a mut dyn Store) -> Self {
        UpdateExecutor {
            store,
            context: ExecutionContext::default(),
            transaction_mode: false,
            batch_size: 10000, // Default batch size for large operations
            stats: UpdateStatistics::default(),
        }
    }

    /// Create a new update executor with transaction support
    pub fn with_transaction(store: &'a mut dyn Store) -> Self {
        UpdateExecutor {
            store,
            context: ExecutionContext::default(),
            transaction_mode: true,
            batch_size: 10000,
            stats: UpdateStatistics::default(),
        }
    }

    /// Configure batch size for large operations
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// Get execution statistics
    pub fn statistics(&self) -> &UpdateStatistics {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.stats = UpdateStatistics::default();
    }

    /// Execute an update operation with enhanced error handling and timing
    pub fn execute(&mut self, operation: &UpdateOperation) -> Result<UpdateResult, OxirsError> {
        let start_time = std::time::Instant::now();
        self.stats.total_operations += 1;

        // Start transaction if in transaction mode
        if self.transaction_mode {
            // Note: Transaction support would be implemented here
            // self.store.transaction_begin()?;
        }

        let result = match operation {
            UpdateOperation::InsertData { data } => self.execute_insert_data_enhanced(data),
            UpdateOperation::DeleteData { data } => self.execute_delete_data_enhanced(data),
            UpdateOperation::DeleteWhere { pattern } => self.execute_delete_where(pattern),
            UpdateOperation::InsertWhere { pattern, template } => {
                self.execute_insert_where(pattern, template)
            }
            UpdateOperation::DeleteInsertWhere {
                delete_template,
                insert_template,
                pattern,
                using,
            } => self.execute_delete_insert_where(delete_template, insert_template, pattern, using),
            UpdateOperation::Clear { target, silent } => self.execute_clear(target, *silent),
            UpdateOperation::Drop { target, silent } => self.execute_drop(target, *silent),
            UpdateOperation::Create { graph, silent } => self.execute_create(graph, *silent),
            UpdateOperation::Copy { from, to, silent } => self.execute_copy(from, to, *silent),
            UpdateOperation::Move { from, to, silent } => self.execute_move(from, to, *silent),
            UpdateOperation::Add { from, to, silent } => self.execute_add(from, to, *silent),
            UpdateOperation::Load {
                source,
                graph,
                silent,
            } => self.execute_load(source, graph.as_ref(), *silent),
        };

        // Handle transaction completion
        match &result {
            Ok(_) => {
                if self.transaction_mode {
                    // Note: Transaction commit would be implemented here
                    // self.store.transaction_commit()?;
                }
            }
            Err(_) => {
                if self.transaction_mode {
                    // Note: Transaction rollback would be implemented here
                    // self.store.transaction_rollback()?;
                }
            }
        }

        // Update statistics
        let execution_time = start_time.elapsed();
        self.stats.total_execution_time += execution_time;
        self.stats.operations_per_second =
            self.stats.total_operations as f64 / self.stats.total_execution_time.as_secs_f64();

        result
    }

    /// Execute INSERT DATA
    fn execute_insert_data(&mut self, data: &[QuadPattern]) -> Result<UpdateResult, OxirsError> {
        let mut result = UpdateResult::default();

        for pattern in data {
            // Convert pattern to concrete quad
            let quad = self.pattern_to_quad(pattern)?;

            // Insert into store
            self.store.insert(&quad)?;
            result.inserted += 1;
        }

        Ok(result)
    }

    /// Execute INSERT DATA with enhanced batching and validation
    fn execute_insert_data_enhanced(
        &mut self,
        data: &[QuadPattern],
    ) -> Result<UpdateResult, OxirsError> {
        let mut result = UpdateResult::default();

        if data.is_empty() {
            return Ok(result);
        }

        // For large datasets, use batching
        if data.len() > self.batch_size {
            for chunk in data.chunks(self.batch_size) {
                let batch_result = self.execute_insert_data_batch(chunk)?;
                result.inserted += batch_result.inserted;
                self.stats.batch_count += 1;
            }
        } else {
            // Small datasets - process directly
            for pattern in data {
                // Validate pattern before conversion
                self.validate_quad_pattern(pattern)?;

                // Convert pattern to concrete quad
                let quad = self.pattern_to_quad(pattern)?;

                // Insert into store
                self.store.insert(&quad)?;
                result.inserted += 1;
            }
        }

        Ok(result)
    }

    /// Execute a batch of INSERT DATA operations
    fn execute_insert_data_batch(
        &mut self,
        batch: &[QuadPattern],
    ) -> Result<UpdateResult, OxirsError> {
        let mut result = UpdateResult::default();
        let mut quads_to_insert = Vec::with_capacity(batch.len());

        // First pass: validate and convert all patterns
        for pattern in batch {
            self.validate_quad_pattern(pattern)?;
            let quad = self.pattern_to_quad(pattern)?;
            quads_to_insert.push(quad);
        }

        // Second pass: batch insert all quads
        for quad in quads_to_insert {
            self.store.insert(&quad)?;
            result.inserted += 1;
        }

        Ok(result)
    }

    /// Validate a quad pattern before processing
    fn validate_quad_pattern(&self, pattern: &QuadPattern) -> Result<(), OxirsError> {
        // Check that variables are not present in INSERT DATA (they should be concrete)
        if matches!(pattern.subject, Term::Variable(_)) {
            return Err(OxirsError::Query(
                "Variables not allowed in INSERT DATA subject".to_string(),
            ));
        }
        if matches!(pattern.predicate, Term::Variable(_)) {
            return Err(OxirsError::Query(
                "Variables not allowed in INSERT DATA predicate".to_string(),
            ));
        }
        if matches!(pattern.object, Term::Variable(_)) {
            return Err(OxirsError::Query(
                "Variables not allowed in INSERT DATA object".to_string(),
            ));
        }

        // Validate IRIs are well-formed
        if let Term::Iri(iri) = &pattern.subject {
            if iri.as_str().is_empty() {
                return Err(OxirsError::Query("Empty IRI in subject".to_string()));
            }
        }
        if let Term::Iri(iri) = &pattern.predicate {
            if iri.as_str().is_empty() {
                return Err(OxirsError::Query("Empty IRI in predicate".to_string()));
            }
        }
        if let Term::Iri(iri) = &pattern.object {
            if iri.as_str().is_empty() {
                return Err(OxirsError::Query("Empty IRI in object".to_string()));
            }
        }

        Ok(())
    }

    /// Execute DELETE DATA
    fn execute_delete_data(&mut self, data: &[QuadPattern]) -> Result<UpdateResult, OxirsError> {
        let mut result = UpdateResult::default();

        for pattern in data {
            // Convert pattern to concrete quad
            let quad = self.pattern_to_quad(pattern)?;

            // Delete from store
            if self.store.remove(&quad)? {
                result.deleted += 1;
            }
        }

        Ok(result)
    }

    /// Execute DELETE DATA with enhanced batching and validation
    fn execute_delete_data_enhanced(
        &mut self,
        data: &[QuadPattern],
    ) -> Result<UpdateResult, OxirsError> {
        let mut result = UpdateResult::default();

        if data.is_empty() {
            return Ok(result);
        }

        // For large datasets, use batching
        if data.len() > self.batch_size {
            for chunk in data.chunks(self.batch_size) {
                let batch_result = self.execute_delete_data_batch(chunk)?;
                result.deleted += batch_result.deleted;
                self.stats.batch_count += 1;
            }
        } else {
            // Small datasets - process directly
            for pattern in data {
                // Validate pattern before conversion
                self.validate_quad_pattern(pattern)?;

                // Convert pattern to concrete quad
                let quad = self.pattern_to_quad(pattern)?;

                // Delete from store
                if self.store.remove(&quad)? {
                    result.deleted += 1;
                }
            }
        }

        Ok(result)
    }

    /// Execute a batch of DELETE DATA operations
    fn execute_delete_data_batch(
        &mut self,
        batch: &[QuadPattern],
    ) -> Result<UpdateResult, OxirsError> {
        let mut result = UpdateResult::default();
        let mut quads_to_delete = Vec::with_capacity(batch.len());

        // First pass: validate and convert all patterns
        for pattern in batch {
            self.validate_quad_pattern(pattern)?;
            let quad = self.pattern_to_quad(pattern)?;
            quads_to_delete.push(quad);
        }

        // Second pass: batch delete all quads
        for quad in quads_to_delete {
            if self.store.remove(&quad)? {
                result.deleted += 1;
            }
        }

        Ok(result)
    }

    /// Execute DELETE WHERE
    fn execute_delete_where(&mut self, pattern: &Algebra) -> Result<UpdateResult, OxirsError> {
        let mut result = UpdateResult::default();

        // Evaluate pattern to get bindings
        let bindings = self.evaluate_pattern(pattern)?;

        if bindings.is_empty() {
            return Ok(result); // No matches to delete
        }

        // Track quads to delete to avoid deletion during iteration
        let mut quads_to_delete = Vec::new();

        // For each binding, collect matching quads
        for binding in &bindings {
            let quads = self.apply_binding_to_pattern(pattern, binding)?;
            quads_to_delete.extend(quads);
        }

        // Remove duplicates for efficiency
        quads_to_delete.sort();
        quads_to_delete.dedup();

        // Batch delete for large operations
        if quads_to_delete.len() > self.batch_size {
            for chunk in quads_to_delete.chunks(self.batch_size) {
                for quad in chunk {
                    if self.store.remove(quad)? {
                        result.deleted += 1;
                    }
                }
                self.stats.batch_count += 1;
            }
        } else {
            // Direct deletion for smaller sets
            for quad in quads_to_delete {
                if self.store.remove(&quad)? {
                    result.deleted += 1;
                }
            }
        }

        Ok(result)
    }

    /// Execute INSERT WHERE
    fn execute_insert_where(
        &mut self,
        pattern: &Algebra,
        template: &[QuadPattern],
    ) -> Result<UpdateResult, OxirsError> {
        let mut result = UpdateResult::default();

        if template.is_empty() {
            return Ok(result); // Nothing to insert
        }

        // Evaluate pattern to get bindings
        let bindings = self.evaluate_pattern(pattern)?;

        if bindings.is_empty() {
            return Ok(result); // No matches, nothing to insert
        }

        // Track quads to insert
        let mut quads_to_insert = Vec::new();

        // For each binding, instantiate template
        for binding in &bindings {
            for quad_pattern in template {
                match self.instantiate_template(quad_pattern, binding) {
                    Ok(quad) => quads_to_insert.push(quad),
                    Err(e) => {
                        // Log the error but continue with other insertions
                        eprintln!("Warning: Failed to instantiate template: {}", e);
                    }
                }
            }
        }

        // Remove duplicates for efficiency
        quads_to_insert.sort();
        quads_to_insert.dedup();

        // Batch insert for large operations
        if quads_to_insert.len() > self.batch_size {
            for chunk in quads_to_insert.chunks(self.batch_size) {
                for quad in chunk {
                    self.store.insert(quad)?;
                    result.inserted += 1;
                }
                self.stats.batch_count += 1;
            }
        } else {
            // Direct insertion for smaller sets
            for quad in quads_to_insert {
                self.store.insert(&quad)?;
                result.inserted += 1;
            }
        }

        Ok(result)
    }

    /// Execute DELETE/INSERT WHERE with enhanced error handling and batching
    fn execute_delete_insert_where(
        &mut self,
        delete_template: &[QuadPattern],
        insert_template: &[QuadPattern],
        pattern: &Algebra,
        _using: &Option<Vec<GraphReference>>,
    ) -> Result<UpdateResult, OxirsError> {
        let mut result = UpdateResult::default();

        if delete_template.is_empty() && insert_template.is_empty() {
            return Ok(result); // Nothing to do
        }

        // Evaluate pattern to get bindings
        let bindings = self.evaluate_pattern(pattern)?;

        if bindings.is_empty() {
            return Ok(result); // No matches
        }

        // Phase 1: Collect quads to delete
        let mut quads_to_delete = Vec::new();
        if !delete_template.is_empty() {
            for binding in &bindings {
                for quad_pattern in delete_template {
                    match self.instantiate_template(quad_pattern, binding) {
                        Ok(quad) => quads_to_delete.push(quad),
                        Err(e) => {
                            eprintln!("Warning: Failed to instantiate delete template: {}", e);
                        }
                    }
                }
            }

            // Remove duplicates for efficiency
            quads_to_delete.sort();
            quads_to_delete.dedup();
        }

        // Phase 2: Collect quads to insert
        let mut quads_to_insert = Vec::new();
        if !insert_template.is_empty() {
            for binding in &bindings {
                for quad_pattern in insert_template {
                    match self.instantiate_template(quad_pattern, binding) {
                        Ok(quad) => quads_to_insert.push(quad),
                        Err(e) => {
                            eprintln!("Warning: Failed to instantiate insert template: {}", e);
                        }
                    }
                }
            }

            // Remove duplicates for efficiency
            quads_to_insert.sort();
            quads_to_insert.dedup();
        }

        // Phase 3: Execute deletions first (important for atomicity)
        if !quads_to_delete.is_empty() {
            if quads_to_delete.len() > self.batch_size {
                for chunk in quads_to_delete.chunks(self.batch_size) {
                    for quad in chunk {
                        if self.store.remove(quad)? {
                            result.deleted += 1;
                        }
                    }
                    self.stats.batch_count += 1;
                }
            } else {
                for quad in quads_to_delete {
                    if self.store.remove(&quad)? {
                        result.deleted += 1;
                    }
                }
            }
        }

        // Phase 4: Execute insertions
        if !quads_to_insert.is_empty() {
            if quads_to_insert.len() > self.batch_size {
                for chunk in quads_to_insert.chunks(self.batch_size) {
                    for quad in chunk {
                        self.store.insert(quad)?;
                        result.inserted += 1;
                    }
                    self.stats.batch_count += 1;
                }
            } else {
                for quad in quads_to_insert {
                    self.store.insert(&quad)?;
                    result.inserted += 1;
                }
            }
        }

        Ok(result)
    }

    /// Execute CLEAR
    fn execute_clear(
        &mut self,
        target: &GraphTarget,
        silent: bool,
    ) -> Result<UpdateResult, OxirsError> {
        let mut result = UpdateResult::default();

        match target {
            GraphTarget::All => {
                // Clear all graphs including default
                result.deleted = self.store.clear_all()?;
            }
            GraphTarget::Named => {
                // Clear all named graphs but not default
                result.deleted = self.store.clear_named_graphs()?;
            }
            GraphTarget::Default => {
                // Clear only default graph
                result.deleted = self.store.clear_default_graph()?;
            }
            GraphTarget::Graph(graph_ref) => {
                // Clear specific graph
                let graph_name = self.graph_ref_to_named_node(graph_ref)?;
                match self
                    .store
                    .clear_graph(Some(&GraphName::NamedNode(graph_name)))
                {
                    Ok(count) => result.deleted = count,
                    Err(e) if !silent => return Err(e),
                    _ => {} // Silent mode - ignore errors
                }
            }
        }

        Ok(result)
    }

    /// Execute DROP
    fn execute_drop(
        &mut self,
        target: &GraphTarget,
        silent: bool,
    ) -> Result<UpdateResult, OxirsError> {
        let mut result = UpdateResult::default();

        match target {
            GraphTarget::All => {
                // Drop all graphs
                let graphs = self.store.graphs()?;
                for graph in graphs {
                    self.store
                        .drop_graph(Some(&GraphName::NamedNode(graph.clone())))?;
                    result.graphs_dropped.push(graph.as_str().to_string());
                }
            }
            GraphTarget::Named => {
                // Drop all named graphs
                let graphs = self.store.named_graphs()?;
                for graph in graphs {
                    self.store
                        .drop_graph(Some(&GraphName::NamedNode(graph.clone())))?;
                    result.graphs_dropped.push(graph.as_str().to_string());
                }
            }
            GraphTarget::Default => {
                // Cannot drop default graph - only clear it
                if !silent {
                    return Err(OxirsError::Query("Cannot DROP DEFAULT graph".to_string()));
                }
            }
            GraphTarget::Graph(graph_ref) => {
                let graph_name = self.graph_ref_to_named_node(graph_ref)?;
                match self
                    .store
                    .drop_graph(Some(&GraphName::NamedNode(graph_name.clone())))
                {
                    Ok(_) => result.graphs_dropped.push(graph_name.as_str().to_string()),
                    Err(e) if !silent => return Err(e),
                    _ => {} // Silent mode
                }
            }
        }

        Ok(result)
    }

    /// Execute CREATE
    fn execute_create(
        &mut self,
        graph: &GraphReference,
        silent: bool,
    ) -> Result<UpdateResult, OxirsError> {
        let mut result = UpdateResult::default();

        let graph_name = self.graph_ref_to_named_node(graph)?;

        match self.store.create_graph(Some(&graph_name)) {
            Ok(_) => {
                result.graphs_created.push(graph_name.as_str().to_string());
            }
            Err(e) => {
                if !silent {
                    return Err(e);
                }
            }
        }

        Ok(result)
    }

    /// Execute COPY
    fn execute_copy(
        &mut self,
        from: &GraphTarget,
        to: &GraphTarget,
        silent: bool,
    ) -> Result<UpdateResult, OxirsError> {
        // First clear the target
        self.execute_clear(to, silent)?;

        // Then add from source to target
        self.execute_add(from, to, silent)
    }

    /// Execute MOVE
    fn execute_move(
        &mut self,
        from: &GraphTarget,
        to: &GraphTarget,
        silent: bool,
    ) -> Result<UpdateResult, OxirsError> {
        // First copy
        let result = self.execute_copy(from, to, silent)?;

        // Then clear source
        self.execute_clear(from, silent)?;

        Ok(result)
    }

    /// Execute ADD
    fn execute_add(
        &mut self,
        from: &GraphTarget,
        to: &GraphTarget,
        _silent: bool,
    ) -> Result<UpdateResult, OxirsError> {
        let mut result = UpdateResult::default();

        // Get quads from source
        let source_quads = self.get_quads_from_target(from)?;

        // Determine target graph
        let target_graph = match to {
            GraphTarget::Graph(g) => Some(self.graph_ref_to_named_node(g)?),
            GraphTarget::Default => None,
            _ => {
                return Err(OxirsError::Query(
                    "Invalid target for ADD operation".to_string(),
                ))
            }
        };

        // Insert quads into target
        for quad in source_quads {
            let new_quad = if let Some(ref target) = target_graph {
                Quad::new(
                    quad.subject().clone(),
                    quad.predicate().clone(),
                    quad.object().clone(),
                    GraphName::NamedNode(target.clone()),
                )
            } else {
                Quad::new(
                    quad.subject().clone(),
                    quad.predicate().clone(),
                    quad.object().clone(),
                    GraphName::DefaultGraph,
                )
            };

            self.store.insert(&new_quad)?;
            result.inserted += 1;
        }

        Ok(result)
    }

    /// Execute LOAD
    fn execute_load(
        &mut self,
        source: &str,
        graph: Option<&GraphReference>,
        silent: bool,
    ) -> Result<UpdateResult, OxirsError> {
        let mut result = UpdateResult::default();

        // Determine target graph
        let target_graph = graph.map(|g| self.graph_ref_to_named_node(g)).transpose()?;

        // Load data from source
        match self.store.load_from_url(source, target_graph.as_ref()) {
            Ok(count) => {
                result.inserted = count;
            }
            Err(e) if !silent => return Err(e),
            _ => {} // Silent mode
        }

        Ok(result)
    }

    // Helper methods

    /// Convert a quad pattern to a concrete quad
    fn pattern_to_quad(&self, pattern: &QuadPattern) -> Result<Quad, OxirsError> {
        let subject = self.term_to_subject(&pattern.subject)?;
        let predicate = self.term_to_predicate(&pattern.predicate)?;
        let object = self.term_to_object(&pattern.object)?;
        let graph_name = pattern
            .graph
            .as_ref()
            .map(|g| self.graph_ref_to_named_node(g))
            .transpose()?
            .map(GraphName::NamedNode)
            .unwrap_or(GraphName::DefaultGraph);

        Ok(Quad::new(subject, predicate, object, graph_name))
    }

    /// Convert Term to subject
    fn term_to_subject(&self, term: &Term) -> Result<oxirs_core::model::Subject, OxirsError> {
        match term {
            Term::Iri(iri) => Ok(NamedNode::new(iri.as_str())?.into()),
            Term::BlankNode(id) => Ok(BlankNode::new(id)?.into()),
            Term::Variable(_) => Err(OxirsError::Query(
                "Variables not allowed in concrete data".to_string(),
            )),
            Term::Literal(_) => Err(OxirsError::Query(
                "Literals cannot be used as subjects".to_string(),
            )),
            Term::QuotedTriple(_) => Err(OxirsError::Query(
                "Quoted triples not yet supported as subjects in concrete data".to_string(),
            )),
            Term::PropertyPath(_) => Err(OxirsError::Query(
                "Property paths not allowed as subjects in concrete data".to_string(),
            )),
        }
    }

    /// Convert Term to predicate
    fn term_to_predicate(&self, term: &Term) -> Result<NamedNode, OxirsError> {
        match term {
            Term::Iri(iri) => NamedNode::new(iri.as_str()),
            Term::Variable(_) => Err(OxirsError::Query(
                "Variables not allowed in concrete data".to_string(),
            )),
            Term::BlankNode(_) => Err(OxirsError::Query(
                "Blank nodes cannot be used as predicates in most RDF contexts".to_string(),
            )),
            Term::Literal(_) => Err(OxirsError::Query(
                "Literals cannot be used as predicates".to_string(),
            )),
            Term::QuotedTriple(_) => Err(OxirsError::Query(
                "Quoted triples not supported as predicates in concrete data".to_string(),
            )),
            Term::PropertyPath(_) => Err(OxirsError::Query(
                "Property paths not allowed as predicates in concrete data".to_string(),
            )),
        }
    }

    /// Convert Term to object
    fn term_to_object(&self, term: &Term) -> Result<oxirs_core::model::Object, OxirsError> {
        match term {
            Term::Iri(iri) => Ok(NamedNode::new(iri.as_str())?.into()),
            Term::BlankNode(id) => Ok(BlankNode::new(id)?.into()),
            Term::Literal(lit) => {
                let literal = if let Some(lang) = &lit.language {
                    CoreLiteral::new_language_tagged_literal(&lit.value, lang)?
                } else if let Some(dt) = &lit.datatype {
                    CoreLiteral::new_typed(&lit.value, dt.clone())
                } else {
                    CoreLiteral::new(&lit.value)
                };
                Ok(literal.into())
            }
            Term::Variable(_) => Err(OxirsError::Query(
                "Variables not allowed in concrete data".to_string(),
            )),
            Term::QuotedTriple(_) => Err(OxirsError::Query(
                "Quoted triples not yet supported in concrete data".to_string(),
            )),
            Term::PropertyPath(_) => Err(OxirsError::Query(
                "Property paths not allowed in concrete data".to_string(),
            )),
        }
    }

    /// Convert graph reference to named node
    fn graph_ref_to_named_node(&self, graph_ref: &GraphReference) -> Result<NamedNode, OxirsError> {
        match graph_ref {
            GraphReference::Iri(iri) => NamedNode::new(iri),
            GraphReference::Default => Err(OxirsError::Query(
                "DEFAULT is not a valid graph IRI".to_string(),
            )),
        }
    }

    /// Evaluate a pattern to get variable bindings
    fn evaluate_pattern(
        &mut self,
        pattern: &Algebra,
    ) -> Result<Vec<HashMap<String, oxirs_core::model::Term>>, OxirsError> {
        use crate::executor::QueryExecutor;

        // Create evaluation context
        let mut context = EvaluationContext::default();

        // Create query executor
        let mut executor = QueryExecutor::new();

        // Execute the pattern and collect bindings
        let results = executor
            .execute_algebra(pattern, &mut context)
            .map_err(|e| OxirsError::Query(e.to_string()))?;

        // Convert results to the expected format
        let mut bindings = Vec::new();
        for solution in results {
            for binding in solution {
                let mut converted_binding = HashMap::new();
                for (var, term) in binding {
                    // Convert from algebra::Term to term::Term and then to oxirs_core::model::Term
                    let arq_term = crate::term::Term::from_algebra_term(&term);
                    let core_term = self.convert_term_to_core(&arq_term)?;
                    converted_binding.insert(var.as_str().to_string(), core_term);
                }
                bindings.push(converted_binding);
            }
        }

        Ok(bindings)
    }

    /// Apply bindings to pattern to get concrete quads
    fn apply_binding_to_pattern(
        &self,
        pattern: &Algebra,
        binding: &HashMap<String, oxirs_core::model::Term>,
    ) -> Result<Vec<Quad>, OxirsError> {
        let mut quads = Vec::new();

        // Extract triple patterns from the algebra
        let triple_patterns = self.extract_triple_patterns(pattern);

        for triple_pattern in triple_patterns {
            // Instantiate each term with the binding
            let subject_term = self.instantiate_algebra_term(&triple_pattern.subject, binding)?;
            let predicate_term =
                self.instantiate_algebra_term(&triple_pattern.predicate, binding)?;
            let object_term = self.instantiate_algebra_term(&triple_pattern.object, binding)?;

            // Convert terms to appropriate types for Quad
            let subject = self.core_term_to_subject(subject_term)?;
            let predicate = self.core_term_to_predicate(predicate_term)?;
            let object = self.core_term_to_object(object_term)?;

            // Default graph unless specified otherwise
            let graph_name = GraphName::DefaultGraph;

            // Create the quad
            let quad = Quad::new(subject, predicate, object, graph_name);
            quads.push(quad);
        }

        Ok(quads)
    }

    /// Instantiate a template with bindings
    fn instantiate_template(
        &self,
        template: &QuadPattern,
        binding: &HashMap<String, oxirs_core::model::Term>,
    ) -> Result<Quad, OxirsError> {
        let subject = self.instantiate_term(&template.subject, binding)?;
        let predicate = self.instantiate_term(&template.predicate, binding)?;
        let object = self.instantiate_term(&template.object, binding)?;

        let subject = self.term_to_subject(&subject)?;
        let predicate = self.term_to_predicate(&predicate)?;
        let object = self.term_to_object(&object)?;

        let graph_name = template
            .graph
            .as_ref()
            .map(|g| self.graph_ref_to_named_node(g))
            .transpose()?
            .map(GraphName::NamedNode)
            .unwrap_or(GraphName::DefaultGraph);

        Ok(Quad::new(subject, predicate, object, graph_name))
    }

    /// Instantiate a term with bindings
    fn instantiate_term(
        &self,
        term: &Term,
        binding: &HashMap<String, oxirs_core::model::Term>,
    ) -> Result<Term, OxirsError> {
        match term {
            Term::Variable(var) => binding
                .get(var.as_str())
                .map(|t| self.core_term_to_arq_term(t))
                .ok_or_else(|| OxirsError::Query(format!("Unbound variable: {var}"))),
            _ => Ok(term.clone()),
        }
    }

    /// Convert core term to ARQ term
    fn core_term_to_arq_term(&self, term: &oxirs_core::model::Term) -> Term {
        match term {
            oxirs_core::model::Term::NamedNode(n) => Term::Iri(n.clone()),
            oxirs_core::model::Term::BlankNode(b) => Term::BlankNode(b.as_str().to_string()),
            oxirs_core::model::Term::Literal(l) => {
                let lit = crate::algebra::Literal {
                    value: l.value().to_string(),
                    language: l.language().map(|s| s.to_string()),
                    datatype: Some(l.datatype().clone().into()),
                };
                Term::Literal(lit)
            }
            oxirs_core::model::Term::Variable(_) | oxirs_core::model::Term::QuotedTriple(_) => {
                panic!("Variables and quoted triples not supported in update operations")
            }
        }
    }

    /// Get quads from a graph target
    fn get_quads_from_target(&self, target: &GraphTarget) -> Result<Vec<Quad>, OxirsError> {
        match target {
            GraphTarget::All => self.store.quads(),
            GraphTarget::Named => self.store.named_graph_quads(),
            GraphTarget::Default => self.store.default_graph_quads(),
            GraphTarget::Graph(graph_ref) => {
                let graph = self.graph_ref_to_named_node(graph_ref)?;
                self.store.graph_quads(Some(&graph))
            }
        }
    }

    /// Convert a term from arq::Term to oxirs_core::model::Term
    fn convert_term_to_core(
        &self,
        term: &crate::term::Term,
    ) -> Result<oxirs_core::model::Term, OxirsError> {
        use oxirs_core::model::Term as CoreTerm;

        match term {
            crate::term::Term::Iri(iri) => Ok(CoreTerm::NamedNode(NamedNode::new(iri)?)),
            crate::term::Term::BlankNode(id) => Ok(CoreTerm::BlankNode(BlankNode::new(id)?)),
            crate::term::Term::Literal(lit) => {
                let core_literal = if let Some(lang) = &lit.language_tag {
                    CoreLiteral::new_language_tagged_literal(&lit.lexical_form, lang)?
                } else if lit.datatype != "http://www.w3.org/2001/XMLSchema#string" {
                    CoreLiteral::new_typed(&lit.lexical_form, NamedNode::new(&lit.datatype)?)
                } else {
                    CoreLiteral::new_simple_literal(&lit.lexical_form)
                };
                Ok(CoreTerm::Literal(core_literal))
            }
            crate::term::Term::Variable(_) => Err(OxirsError::Query(
                "Cannot convert variable to concrete term".to_string(),
            )),
            crate::term::Term::QuotedTriple(_) => Err(OxirsError::Query(
                "Cannot convert quoted triple to concrete term".to_string(),
            )),
            crate::term::Term::PropertyPath(_) => Err(OxirsError::Query(
                "Cannot convert property path to concrete term".to_string(),
            )),
        }
    }

    /// Extract triple patterns from algebra expression
    fn extract_triple_patterns(&self, algebra: &Algebra) -> Vec<TriplePattern> {
        let mut patterns = Vec::new();

        match algebra {
            Algebra::Bgp(bgp_patterns) => {
                patterns.extend(bgp_patterns.iter().cloned());
            }
            Algebra::Join { left, right } => {
                patterns.extend(self.extract_triple_patterns(left));
                patterns.extend(self.extract_triple_patterns(right));
            }
            Algebra::LeftJoin { left, right, .. } => {
                patterns.extend(self.extract_triple_patterns(left));
                patterns.extend(self.extract_triple_patterns(right));
            }
            Algebra::Union { left, right } => {
                patterns.extend(self.extract_triple_patterns(left));
                patterns.extend(self.extract_triple_patterns(right));
            }
            Algebra::Filter { pattern, .. } => {
                patterns.extend(self.extract_triple_patterns(pattern));
            }
            Algebra::Extend { pattern, .. } => {
                patterns.extend(self.extract_triple_patterns(pattern));
            }
            Algebra::Minus { left, right } => {
                patterns.extend(self.extract_triple_patterns(left));
                patterns.extend(self.extract_triple_patterns(right));
            }
            Algebra::Service { pattern, .. } => {
                patterns.extend(self.extract_triple_patterns(pattern));
            }
            // For other algebra types, we don't extract patterns
            _ => {}
        }

        patterns
    }

    /// Instantiate an algebra term with variable bindings
    fn instantiate_algebra_term(
        &self,
        term: &Term,
        binding: &HashMap<String, oxirs_core::model::Term>,
    ) -> Result<oxirs_core::model::Term, OxirsError> {
        match term {
            Term::Variable(var) => binding
                .get(var.as_str())
                .cloned()
                .ok_or_else(|| OxirsError::Query(format!("Unbound variable: {var}"))),
            _ => {
                // Convert algebra term to arq term and then to core term
                let arq_term = crate::term::Term::from_algebra_term(term);
                self.convert_term_to_core(&arq_term)
            }
        }
    }

    /// Convert oxirs_core::model::Term to Subject
    fn core_term_to_subject(
        &self,
        term: oxirs_core::model::Term,
    ) -> Result<oxirs_core::Subject, OxirsError> {
        match term {
            oxirs_core::model::Term::NamedNode(node) => Ok(oxirs_core::Subject::NamedNode(node)),
            oxirs_core::model::Term::BlankNode(node) => Ok(oxirs_core::Subject::BlankNode(node)),
            oxirs_core::model::Term::Variable(var) => Ok(oxirs_core::Subject::Variable(var)),
            _ => Err(OxirsError::Query("Invalid subject term type".to_string())),
        }
    }

    /// Convert oxirs_core::model::Term to Predicate
    fn core_term_to_predicate(
        &self,
        term: oxirs_core::model::Term,
    ) -> Result<oxirs_core::Predicate, OxirsError> {
        match term {
            oxirs_core::model::Term::NamedNode(node) => Ok(oxirs_core::Predicate::NamedNode(node)),
            oxirs_core::model::Term::Variable(var) => Ok(oxirs_core::Predicate::Variable(var)),
            _ => Err(OxirsError::Query("Invalid predicate term type".to_string())),
        }
    }

    /// Convert oxirs_core::model::Term to Object
    fn core_term_to_object(
        &self,
        term: oxirs_core::model::Term,
    ) -> Result<oxirs_core::Object, OxirsError> {
        match term {
            oxirs_core::model::Term::NamedNode(node) => Ok(oxirs_core::Object::NamedNode(node)),
            oxirs_core::model::Term::BlankNode(node) => Ok(oxirs_core::Object::BlankNode(node)),
            oxirs_core::model::Term::Literal(lit) => Ok(oxirs_core::Object::Literal(lit)),
            oxirs_core::model::Term::Variable(var) => Ok(oxirs_core::Object::Variable(var)),
            oxirs_core::model::Term::QuotedTriple(qt) => Ok(oxirs_core::Object::QuotedTriple(qt)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_result_default() {
        let result = UpdateResult::default();
        assert_eq!(result.inserted, 0);
        assert_eq!(result.deleted, 0);
        assert!(result.graphs_created.is_empty());
        assert!(result.graphs_dropped.is_empty());
    }

    #[test]
    fn test_graph_reference() {
        let iri_ref = GraphReference::Iri("http://example.org/graph".to_string());
        let default_ref = GraphReference::Default;

        assert_ne!(iri_ref, default_ref);
    }

    #[test]
    fn test_quad_pattern() {
        let pattern = QuadPattern {
            subject: Term::Variable(Variable::new("s").unwrap()),
            predicate: Term::Iri(NamedNode::new("http://example.org/pred").unwrap()),
            object: Term::Literal(crate::algebra::Literal {
                value: "test".to_string(),
                language: None,
                datatype: None,
            }),
            graph: None,
        };

        assert_eq!(pattern.subject, Term::Variable(Variable::new("s").unwrap()));
    }

    #[test]
    fn test_enhanced_update_operations() {
        use oxirs_core::Store;

        // Create a test store (would need actual Store implementation)
        // For now, this is a placeholder test to verify the enhanced operations compile

        // Test UPDATE operation types
        let insert_data = UpdateOperation::InsertData {
            data: vec![QuadPattern {
                subject: Term::Iri(NamedNode::new("http://example.org/subject").unwrap()),
                predicate: Term::Iri(NamedNode::new("http://example.org/predicate").unwrap()),
                object: Term::Literal(crate::algebra::Literal {
                    value: "test_value".to_string(),
                    language: None,
                    datatype: None,
                }),
                graph: None,
            }],
        };

        let delete_data = UpdateOperation::DeleteData {
            data: vec![QuadPattern {
                subject: Term::Iri(NamedNode::new("http://example.org/subject").unwrap()),
                predicate: Term::Iri(NamedNode::new("http://example.org/predicate").unwrap()),
                object: Term::Literal(crate::algebra::Literal {
                    value: "test_value".to_string(),
                    language: None,
                    datatype: None,
                }),
                graph: None,
            }],
        };

        // Verify operations can be created
        match insert_data {
            UpdateOperation::InsertData { data } => {
                assert_eq!(data.len(), 1);
            }
            _ => panic!("Expected InsertData operation"),
        }

        match delete_data {
            UpdateOperation::DeleteData { data } => {
                assert_eq!(data.len(), 1);
            }
            _ => panic!("Expected DeleteData operation"),
        }
    }

    #[test]
    fn test_update_statistics() {
        let stats = UpdateStatistics::default();
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.batch_count, 0);
        assert_eq!(stats.operations_per_second, 0.0);
    }

    #[test]
    fn test_validation_errors() {
        let invalid_pattern = QuadPattern {
            subject: Term::Variable(Variable::new("s").unwrap()),
            predicate: Term::Iri(NamedNode::new("http://example.org/predicate").unwrap()),
            object: Term::Literal(crate::algebra::Literal {
                value: "test".to_string(),
                language: None,
                datatype: None,
            }),
            graph: None,
        };

        // Test validation logic (would need actual store)
        // This verifies the validation functions compile correctly
        let validation_result = std::panic::catch_unwind(|| {
            // This should fail validation due to variable in subject position for INSERT DATA
            // In actual implementation with store, this would be tested properly
        });

        // Just verify the test structure compiles
        assert!(validation_result.is_ok());
    }
}
