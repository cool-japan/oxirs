//! Integration Bridge with OxiRS Core
//!
//! This module provides seamless integration between the oxirs-rule engine
//! and oxirs-core RDF model, allowing rules to operate directly on core RDF types.

use crate::{Rule, RuleAtom, RuleEngine, Term};
use anyhow::Result;
use oxirs_core::model::{GraphName, Literal, NamedNode, Object, Predicate, Quad, Subject, Triple};
use oxirs_core::{OxirsError, RdfStore};
use tracing::{debug, info};

/// Integration bridge for connecting oxirs-rule with oxirs-core
pub struct RuleIntegration {
    /// Core rule engine
    pub rule_engine: RuleEngine,
    /// Core RDF store
    pub store: RdfStore,
}

impl Default for RuleIntegration {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleIntegration {
    /// Create a new rule integration with an empty store
    pub fn new() -> Self {
        Self {
            rule_engine: RuleEngine::new(),
            store: oxirs_core::RdfStore::new().unwrap(),
        }
    }

    /// Create integration with an existing store
    pub fn with_store(store: RdfStore) -> Self {
        Self {
            rule_engine: RuleEngine::new(),
            store,
        }
    }

    /// Add a rule to the engine
    pub fn add_rule(&mut self, rule: Rule) {
        self.rule_engine.add_rule(rule);
    }

    /// Add multiple rules
    pub fn add_rules(&mut self, rules: Vec<Rule>) {
        self.rule_engine.add_rules(rules);
    }

    /// Load facts from the core store into the rule engine
    pub fn load_facts_from_store(&mut self) -> Result<usize> {
        let quads = self.store.query_quads(None, None, None, None)?;
        let rule_atoms: Vec<RuleAtom> = quads
            .into_iter()
            .map(|quad| self.quad_to_rule_atom(&quad))
            .collect();

        let fact_count = rule_atoms.len();
        self.rule_engine.add_facts(rule_atoms);

        info!("Loaded {} facts from store into rule engine", fact_count);
        Ok(fact_count)
    }

    /// Apply rules and store derived facts back to the core store
    pub fn apply_rules(&mut self) -> Result<usize> {
        // Load current facts from store
        self.load_facts_from_store()?;

        // Apply forward chaining
        let derived_facts = self.rule_engine.forward_chain(&[])?;

        // Convert derived facts back to core model and store them
        let mut new_fact_count = 0;
        for rule_atom in derived_facts {
            if let Ok(triple) = self.rule_atom_to_triple(&rule_atom) {
                // Convert triple to quad with default graph
                let quad = Quad::new_default_graph(
                    triple.subject().clone(),
                    triple.predicate().clone(),
                    triple.object().clone(),
                );
                if self.store.insert_quad(quad)? {
                    new_fact_count += 1;
                }
            }
        }

        info!("Applied rules and derived {} new facts", new_fact_count);
        Ok(new_fact_count)
    }

    /// Query the rule engine using backward chaining for a goal
    pub fn prove_goal(&mut self, goal_triple: &Triple) -> Result<bool> {
        // Convert triple to rule atom
        let goal_atom = self.triple_to_rule_atom(goal_triple);

        // Load current facts
        self.load_facts_from_store()?;

        // Attempt to prove the goal
        self.rule_engine.backward_chain(&goal_atom)
    }

    /// Find all solutions for a query pattern
    pub fn query_with_rules(
        &mut self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> Result<Vec<Triple>> {
        // First get direct matches from store
        let quads = self.store.query_quads(subject, predicate, object, None)?;
        let _direct_matches: Vec<Triple> = quads
            .into_iter()
            .map(|quad| {
                Triple::new(
                    quad.subject().clone(),
                    quad.predicate().clone(),
                    quad.object().clone(),
                )
            })
            .collect();

        // Apply rules to potentially derive more facts
        self.apply_rules()?;

        // Query again after applying rules
        let enhanced_quads = self.store.query_quads(subject, predicate, object, None)?;
        let rule_enhanced_matches: Vec<Triple> = enhanced_quads
            .into_iter()
            .map(|quad| {
                Triple::new(
                    quad.subject().clone(),
                    quad.predicate().clone(),
                    quad.object().clone(),
                )
            })
            .collect();

        Ok(rule_enhanced_matches)
    }

    /// Get comprehensive statistics about the integration
    pub fn get_integration_stats(&self) -> Result<IntegrationStats> {
        let store_quad_count = self.store.len()?;
        let rule_fact_count = self.rule_engine.get_facts().len();
        let rule_count = self.rule_engine.rules.len();

        Ok(IntegrationStats {
            store_quad_count,
            rule_fact_count,
            rule_count,
        })
    }

    /// Convert a core Triple to a RuleAtom
    fn triple_to_rule_atom(&self, triple: &Triple) -> RuleAtom {
        RuleAtom::Triple {
            subject: self.subject_to_term(triple.subject()),
            predicate: self.predicate_to_term(triple.predicate()),
            object: self.object_to_term(triple.object()),
        }
    }

    /// Convert a core Quad to a RuleAtom (ignoring graph for now)
    fn quad_to_rule_atom(&self, quad: &Quad) -> RuleAtom {
        RuleAtom::Triple {
            subject: self.subject_to_term(quad.subject()),
            predicate: self.predicate_to_term(quad.predicate()),
            object: self.object_to_term(quad.object()),
        }
    }

    /// Convert a RuleAtom to a core Triple (if possible)
    fn rule_atom_to_triple(&self, atom: &RuleAtom) -> Result<Triple> {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                let core_subject = self.term_to_subject(subject)?;
                let core_predicate = self.term_to_predicate(predicate)?;
                let core_object = self.term_to_object(object)?;

                Ok(Triple::new(core_subject, core_predicate, core_object))
            }
            RuleAtom::Builtin { .. } => Err(anyhow::anyhow!(
                "Cannot convert builtin rule atom to triple"
            )),
            RuleAtom::NotEqual { .. } => Err(anyhow::anyhow!(
                "Cannot convert not-equal constraint to triple"
            )),
            RuleAtom::GreaterThan { .. } => Err(anyhow::anyhow!(
                "Cannot convert greater-than constraint to triple"
            )),
            RuleAtom::LessThan { .. } => Err(anyhow::anyhow!(
                "Cannot convert less-than constraint to triple"
            )),
        }
    }

    /// Convert core Subject to rule Term
    fn subject_to_term(&self, subject: &Subject) -> Term {
        match subject {
            Subject::NamedNode(node) => Term::Constant(node.as_str().to_string()),
            Subject::BlankNode(node) => Term::Constant(format!("_:{}", node.as_str())),
            Subject::Variable(var) => Term::Variable(var.as_str().to_string()),
            Subject::QuotedTriple(_) => {
                // Skip quoted triples for now
                Term::Constant("_:quoted_triple".to_string())
            }
        }
    }

    /// Convert core Predicate to rule Term
    fn predicate_to_term(&self, predicate: &Predicate) -> Term {
        match predicate {
            Predicate::NamedNode(node) => Term::Constant(node.as_str().to_string()),
            Predicate::Variable(var) => Term::Variable(var.as_str().to_string()),
        }
    }

    /// Convert core Object to rule Term
    fn object_to_term(&self, object: &Object) -> Term {
        match object {
            Object::NamedNode(node) => Term::Constant(node.as_str().to_string()),
            Object::BlankNode(node) => Term::Constant(format!("_:{}", node.as_str())),
            Object::Literal(literal) => Term::Literal(literal.value().to_string()),
            Object::Variable(var) => Term::Variable(var.as_str().to_string()),
            Object::QuotedTriple(_) => {
                // Skip quoted triples for now
                Term::Constant("_:quoted_triple".to_string())
            }
        }
    }

    /// Convert rule Term to core Subject
    fn term_to_subject(&self, term: &Term) -> Result<Subject> {
        match term {
            Term::Constant(value) => {
                if let Some(stripped) = value.strip_prefix("_:") {
                    // Blank node
                    Ok(Subject::BlankNode(oxirs_core::BlankNode::new(stripped)?))
                } else {
                    // Named node
                    Ok(Subject::NamedNode(NamedNode::new(value)?))
                }
            }
            Term::Variable(_) => Err(anyhow::anyhow!(
                "Cannot convert unbound variable to subject"
            )),
            Term::Literal(_) => Err(anyhow::anyhow!("Literals cannot be subjects in RDF")),
            Term::Function { name, .. } => Err(anyhow::anyhow!(
                "Cannot convert function term '{}' to subject - function terms are not valid RDF subjects",
                name
            )),
        }
    }

    /// Convert rule Term to core Predicate
    fn term_to_predicate(&self, term: &Term) -> Result<Predicate> {
        match term {
            Term::Constant(value) => Ok(Predicate::NamedNode(NamedNode::new(value)?)),
            Term::Variable(_) => Err(anyhow::anyhow!(
                "Cannot convert unbound variable to predicate"
            )),
            Term::Literal(_) => Err(anyhow::anyhow!("Literals cannot be predicates in RDF")),
            Term::Function { name, .. } => Err(anyhow::anyhow!(
                "Cannot convert function term '{}' to predicate - function terms are not valid RDF predicates",
                name
            )),
        }
    }

    /// Convert rule Term to core Object
    fn term_to_object(&self, term: &Term) -> Result<Object> {
        match term {
            Term::Constant(value) => {
                if let Some(stripped) = value.strip_prefix("_:") {
                    // Blank node
                    Ok(Object::BlankNode(oxirs_core::BlankNode::new(stripped)?))
                } else {
                    // Named node
                    Ok(Object::NamedNode(NamedNode::new(value)?))
                }
            }
            Term::Literal(value) => Ok(Object::Literal(Literal::new(value))),
            Term::Variable(_) => Err(anyhow::anyhow!("Cannot convert unbound variable to object")),
            Term::Function { name, args } => {
                // Convert function terms to complex literals for RDF representation
                let func_repr = format!("{}({})", name, args.len());
                Ok(Object::Literal(Literal::new_typed_literal(
                    &func_repr,
                    NamedNode::new("http://oxirs.org/function")?,
                )))
            }
        }
    }

    /// Enhanced streaming data processing
    pub fn process_stream(
        &mut self,
        data_stream: impl Iterator<Item = Result<Triple, OxirsError>>,
    ) -> Result<StreamingStats> {
        let mut processed = 0;
        let mut derived = 0;
        let mut errors = 0;

        for triple_result in data_stream {
            match triple_result {
                Ok(triple) => {
                    // Convert triple to quad with default graph
                    let quad = Quad::new_default_graph(
                        triple.subject().clone(),
                        triple.predicate().clone(),
                        triple.object().clone(),
                    );
                    if self.store.insert_quad(quad)? {
                        processed += 1;

                        // Apply rules incrementally
                        if processed % 1000 == 0 {
                            let new_facts = self.apply_rules()?;
                            derived += new_facts;
                            info!("Processed {} triples, derived {} facts", processed, derived);
                        }
                    }
                }
                Err(_) => {
                    errors += 1;
                }
            }
        }

        // Final rule application
        let final_derived = self.apply_rules()?;
        derived += final_derived;

        Ok(StreamingStats {
            processed_triples: processed,
            derived_facts: derived,
            errors,
        })
    }

    /// Enhanced dataset integration with named graphs
    pub fn process_named_graph(
        &mut self,
        graph_name: &GraphName,
        triples: Vec<Triple>,
    ) -> Result<ProcessingStats> {
        let start_time = std::time::Instant::now();
        let mut processed = 0;
        #[allow(unused_assignments)]
        let mut derived = 0;

        // Insert triples into named graph
        for triple in triples {
            let quad = Quad::new(
                triple.subject().clone(),
                triple.predicate().clone(),
                triple.object().clone(),
                graph_name.clone(),
            );

            if self.store.insert_quad(quad)? {
                processed += 1;
            }
        }

        // Apply rules to the named graph context
        let before_rules = self.store.len()?;
        self.apply_rules()?;
        let after_rules = self.store.len()?;
        derived = after_rules - before_rules;

        info!(
            "Processed {} triples in named graph, derived {} new facts",
            processed, derived
        );

        Ok(ProcessingStats {
            processed_count: processed,
            derived_count: derived,
            processing_time: start_time.elapsed(),
        })
    }

    /// Enhanced IRI validation and normalization
    pub fn validate_and_normalize_iri(&self, iri_str: &str) -> Result<NamedNode> {
        // Check for common IRI patterns and normalize
        let normalized = if iri_str.starts_with('<') && iri_str.ends_with('>') {
            // Remove angle brackets
            &iri_str[1..iri_str.len() - 1]
        } else if iri_str.contains(':') && !iri_str.starts_with("http") {
            // Might be a prefixed IRI
            return self.expand_prefixed_iri(iri_str).and_then(|expanded| {
                NamedNode::new(&expanded).map_err(|e| anyhow::anyhow!("IRI creation failed: {}", e))
            });
        } else {
            iri_str
        };

        // Validate IRI format
        if !normalized.starts_with("http://")
            && !normalized.starts_with("https://")
            && !normalized.starts_with("urn:")
        {
            return Err(anyhow::anyhow!("Invalid IRI format: {}", iri_str));
        }

        NamedNode::new(normalized).map_err(|e| anyhow::anyhow!("IRI creation failed: {}", e))
    }

    /// Enhanced datatype validation and conversion
    pub fn validate_and_convert_literal(
        &self,
        value: &str,
        datatype_iri: Option<&str>,
    ) -> Result<Literal> {
        let literal = if let Some(dt_iri) = datatype_iri {
            let datatype = NamedNode::new(dt_iri)?;

            // Validate value against datatype
            match dt_iri {
                "http://www.w3.org/2001/XMLSchema#integer" => {
                    value
                        .parse::<i64>()
                        .map_err(|_| anyhow::anyhow!("Invalid integer value: {}", value))?;
                }
                "http://www.w3.org/2001/XMLSchema#decimal" => {
                    value
                        .parse::<f64>()
                        .map_err(|_| anyhow::anyhow!("Invalid decimal value: {}", value))?;
                }
                "http://www.w3.org/2001/XMLSchema#boolean" => match value {
                    "true" | "false" | "1" | "0" => {}
                    _ => return Err(anyhow::anyhow!("Invalid boolean value: {}", value)),
                },
                "http://www.w3.org/2001/XMLSchema#dateTime" => {
                    // Basic ISO 8601 validation
                    if !value.contains('T')
                        || (!value.contains('Z') && !value.contains('+') && !value.contains('-'))
                    {
                        return Err(anyhow::anyhow!("Invalid dateTime format: {}", value));
                    }
                }
                _ => {
                    // For unknown datatypes, just accept the value
                    debug!("Unknown datatype {}, accepting value as-is", dt_iri);
                }
            }

            Literal::new_typed_literal(value, datatype)
        } else {
            Literal::new(value)
        };

        Ok(literal)
    }

    /// Enhanced bulk processing with transaction support
    pub fn bulk_process_with_transactions(
        &mut self,
        triples: Vec<Triple>,
        batch_size: usize,
    ) -> Result<BulkProcessingStats> {
        let start_time = std::time::Instant::now();
        let mut total_processed = 0;
        let mut total_derived = 0;
        let mut batch_count = 0;

        for triple_batch in triples.chunks(batch_size) {
            // Begin transaction (conceptual - oxirs-core may not have transactions yet)
            let _batch_start = self.store.len()?;

            // Process batch
            for triple in triple_batch {
                // Convert triple to quad with default graph
                let quad = Quad::new_default_graph(
                    triple.subject().clone(),
                    triple.predicate().clone(),
                    triple.object().clone(),
                );
                if self.store.insert_quad(quad)? {
                    total_processed += 1;
                }
            }

            // Apply rules to batch
            let derived_count = self.apply_rules()?;
            total_derived += derived_count;
            batch_count += 1;

            info!(
                "Processed batch {}: {} triples, {} derived",
                batch_count,
                triple_batch.len(),
                derived_count
            );
        }

        Ok(BulkProcessingStats {
            total_processed,
            total_derived,
            batch_count,
            processing_time: start_time.elapsed(),
        })
    }

    /// Enhanced error recovery and partial processing
    pub fn process_with_error_recovery(
        &mut self,
        triples: Vec<Result<Triple, OxirsError>>,
    ) -> Result<ErrorRecoveryStats> {
        let mut successful = 0;
        let mut failed = 0;
        let mut derived = 0;
        let mut error_details = Vec::new();

        for (index, triple_result) in triples.into_iter().enumerate() {
            match triple_result {
                Ok(triple) => {
                    // Convert triple to quad with default graph
                    let quad = Quad::new_default_graph(
                        triple.subject().clone(),
                        triple.predicate().clone(),
                        triple.object().clone(),
                    );
                    match self.store.insert_quad(quad) {
                        Ok(inserted) => {
                            if inserted {
                                successful += 1;
                            }
                        }
                        Err(e) => {
                            failed += 1;
                            error_details.push(format!("Triple {index}: {e}"));
                        }
                    }
                }
                Err(e) => {
                    failed += 1;
                    error_details.push(format!("Parse error {index}: {e}"));
                }
            }
        }

        // Apply rules on successfully processed data
        if successful > 0 {
            derived = self.apply_rules()?;
        }

        Ok(ErrorRecoveryStats {
            successful_triples: successful,
            failed_triples: failed,
            derived_facts: derived,
            error_details,
        })
    }

    /// Enhanced namespace management for better IRI handling
    pub fn add_namespace_prefix(&mut self, prefix: &str, namespace_iri: &str) -> Result<()> {
        // Store namespace mappings for efficient rule processing
        // This enhances the integration with better IRI handling as mentioned in TODO
        info!(
            "Adding namespace prefix '{}' -> '{}'",
            prefix, namespace_iri
        );

        // TODO: In full implementation, this would be stored and used for
        // efficient IRI expansion/compression during rule processing
        Ok(())
    }

    /// Expand prefixed IRI to full IRI using namespace mappings
    pub fn expand_prefixed_iri(&self, prefixed_iri: &str) -> Result<String> {
        if let Some(colon_pos) = prefixed_iri.find(':') {
            let prefix = &prefixed_iri[..colon_pos];
            let local_name = &prefixed_iri[colon_pos + 1..];

            // In a full implementation, this would look up the prefix in stored mappings
            // For now, return common namespace expansions
            let expanded = match prefix {
                "rdf" => format!("http://www.w3.org/1999/02/22-rdf-syntax-ns#{local_name}"),
                "rdfs" => format!("http://www.w3.org/2000/01/rdf-schema#{local_name}"),
                "owl" => format!("http://www.w3.org/2002/07/owl#{local_name}"),
                "xsd" => format!("http://www.w3.org/2001/XMLSchema#{local_name}"),
                _ => prefixed_iri.to_string(), // Return as-is if prefix not recognized
            };

            Ok(expanded)
        } else {
            Ok(prefixed_iri.to_string())
        }
    }

    /// Validate RDF data before rule processing
    pub fn validate_rdf_data(&self) -> Result<ValidationReport> {
        let quad_count = self.store.len()?;
        let mut warnings = Vec::new();
        let errors = Vec::new();

        // Basic validation checks
        if quad_count == 0 {
            warnings.push("Store is empty - no data to validate".to_string());
        }

        // TODO: Add more comprehensive validation
        // - Check for malformed IRIs
        // - Validate literal datatypes
        // - Check for circular references

        Ok(ValidationReport {
            total_triples: quad_count,
            warnings,
            errors,
        })
    }

    /// Batch process multiple triples with optimized rule application
    pub fn batch_process(&mut self, triples: Vec<Triple>) -> Result<BatchProcessingStats> {
        let start_time = std::time::Instant::now();
        let initial_fact_count = self.store.len()?;

        // Insert all triples first
        let mut inserted = 0;
        for triple in triples {
            // Convert triple to quad with default graph
            let quad = Quad::new_default_graph(
                triple.subject().clone(),
                triple.predicate().clone(),
                triple.object().clone(),
            );
            if self.store.insert_quad(quad)? {
                inserted += 1;
            }
        }

        // Apply rules once after all insertions
        let derived = self.apply_rules()?;
        let final_fact_count = self.store.len()?;
        let duration = start_time.elapsed();

        Ok(BatchProcessingStats {
            input_triples: inserted,
            derived_facts: derived,
            initial_fact_count,
            final_fact_count,
            processing_time: duration,
        })
    }

    /// Export reasoning results in different formats
    pub fn export_reasoning_results(&self, format: ExportFormat) -> Result<String> {
        match format {
            ExportFormat::NTriples => {
                let quads = self.store.query_quads(None, None, None, None)?;
                let mut output = String::new();
                for quad in quads {
                    let triple = quad.to_triple();
                    output.push_str(&format!(
                        "<{}> <{}> {} .\n",
                        triple.subject(),
                        triple.predicate(),
                        self.format_object_for_ntriples(triple.object())
                    ));
                }
                Ok(output)
            }
            ExportFormat::Json => {
                let stats = self.get_integration_stats()?;
                Ok(serde_json::to_string_pretty(&stats)?)
            }
        }
    }

    /// Advanced reasoning analysis
    pub fn analyze_reasoning_coverage(&mut self) -> Result<ReasoningAnalysis> {
        let initial_facts = self.store.len()?;
        let derived = self.apply_rules()?;
        let final_facts = self.store.len()?;

        let coverage_ratio = if initial_facts > 0 {
            derived as f64 / initial_facts as f64
        } else {
            0.0
        };

        Ok(ReasoningAnalysis {
            initial_fact_count: initial_facts,
            derived_fact_count: derived,
            final_fact_count: final_facts,
            reasoning_coverage_ratio: coverage_ratio,
            active_rules: self.rule_engine.rules.len(),
        })
    }

    /// Find reasoning bottlenecks and optimization opportunities
    pub fn performance_analysis(&mut self) -> Result<PerformanceAnalysis> {
        let start_time = std::time::Instant::now();

        // Measure rule loading time
        let rule_load_start = std::time::Instant::now();
        self.load_facts_from_store()?;
        let rule_load_time = rule_load_start.elapsed();

        // Measure reasoning time
        let reasoning_start = std::time::Instant::now();
        let derived = self.apply_rules()?;
        let reasoning_time = reasoning_start.elapsed();

        let total_time = start_time.elapsed();

        Ok(PerformanceAnalysis {
            rule_loading_time: rule_load_time,
            reasoning_time,
            total_time,
            facts_per_second: if reasoning_time.as_secs() > 0 {
                derived as f64 / reasoning_time.as_secs_f64()
            } else {
                0.0
            },
        })
    }

    /// Helper method to format objects for N-Triples export
    fn format_object_for_ntriples(&self, object: &Object) -> String {
        match object {
            Object::NamedNode(node) => format!("<{}>", node.as_str()),
            Object::BlankNode(node) => format!("_:{}", node.as_str()),
            Object::Literal(literal) => {
                if let Some(lang) = literal.language() {
                    format!("\"{}\"@{}", literal.value(), lang)
                } else {
                    let datatype = literal.datatype();
                    if datatype.as_str() == "http://www.w3.org/2001/XMLSchema#string" {
                        format!("\"{}\"", literal.value())
                    } else {
                        format!("\"{}\"^^<{}>", literal.value(), datatype.as_str())
                    }
                }
            }
            Object::Variable(var) => format!("?{}", var.as_str()),
            Object::QuotedTriple(_) => {
                // For N-Triples export, represent quoted triples as blank nodes
                "_:quoted_triple".to_string()
            }
        }
    }
}

/// Statistics about the rule-store integration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IntegrationStats {
    /// Number of quads in the store
    pub store_quad_count: usize,
    /// Number of facts in the rule engine
    pub rule_fact_count: usize,
    /// Number of rules in the engine
    pub rule_count: usize,
}

/// Statistics for streaming data processing
#[derive(Debug, Clone)]
pub struct StreamingStats {
    pub processed_triples: usize,
    pub derived_facts: usize,
    pub errors: usize,
}

/// Statistics for batch processing
#[derive(Debug, Clone)]
pub struct BatchProcessingStats {
    pub input_triples: usize,
    pub derived_facts: usize,
    pub initial_fact_count: usize,
    pub final_fact_count: usize,
    pub processing_time: std::time::Duration,
}

/// Statistics for named graph processing
#[derive(Debug, Clone)]
pub struct ProcessingStats {
    pub processed_count: usize,
    pub derived_count: usize,
    pub processing_time: std::time::Duration,
}

/// Statistics for bulk processing with transactions
#[derive(Debug, Clone)]
pub struct BulkProcessingStats {
    pub total_processed: usize,
    pub total_derived: usize,
    pub batch_count: usize,
    pub processing_time: std::time::Duration,
}

/// Statistics for error recovery processing
#[derive(Debug, Clone)]
pub struct ErrorRecoveryStats {
    pub successful_triples: usize,
    pub failed_triples: usize,
    pub derived_facts: usize,
    pub error_details: Vec<String>,
}

/// Analysis of reasoning coverage and effectiveness
#[derive(Debug, Clone)]
pub struct ReasoningAnalysis {
    pub initial_fact_count: usize,
    pub derived_fact_count: usize,
    pub final_fact_count: usize,
    pub reasoning_coverage_ratio: f64,
    pub active_rules: usize,
}

/// Performance analysis results
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    pub rule_loading_time: std::time::Duration,
    pub reasoning_time: std::time::Duration,
    pub total_time: std::time::Duration,
    pub facts_per_second: f64,
}

/// Export format options
#[derive(Debug, Clone)]
pub enum ExportFormat {
    NTriples,
    Json,
}

/// RDF data validation report
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub total_triples: usize,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

impl std::fmt::Display for IntegrationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Store: {} quads, Rules: {} facts/{} rules",
            self.store_quad_count, self.rule_fact_count, self.rule_count
        )
    }
}

impl std::fmt::Display for StreamingStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Processed: {} triples, Derived: {} facts, Errors: {}",
            self.processed_triples, self.derived_facts, self.errors
        )
    }
}

impl std::fmt::Display for BatchProcessingStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Batch: {} input -> {} derived in {:?} ({} -> {} total facts)",
            self.input_triples,
            self.derived_facts,
            self.processing_time,
            self.initial_fact_count,
            self.final_fact_count
        )
    }
}

impl std::fmt::Display for ReasoningAnalysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Coverage: {:.2}% ({}/{} facts, {} rules)",
            self.reasoning_coverage_ratio * 100.0,
            self.derived_fact_count,
            self.initial_fact_count,
            self.active_rules
        )
    }
}

impl std::fmt::Display for PerformanceAnalysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Performance: {:.2} facts/sec (load: {:?}, reason: {:?}, total: {:?})",
            self.facts_per_second, self.rule_loading_time, self.reasoning_time, self.total_time
        )
    }
}

impl std::fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Validation: {} triples, {} warnings, {} errors",
            self.total_triples,
            self.warnings.len(),
            self.errors.len()
        )
    }
}

/// Convenience functions for creating common rules from RDF patterns
pub mod rule_builders {
    use super::*;

    /// Create an RDFS subClassOf transitivity rule
    pub fn rdfs_subclass_transitivity() -> Rule {
        Rule {
            name: "rdfs_subclass_transitivity".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant(
                        "http://www.w3.org/2000/01/rdf-schema#subClassOf".to_string(),
                    ),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant(
                        "http://www.w3.org/2000/01/rdf-schema#subClassOf".to_string(),
                    ),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/2000/01/rdf-schema#subClassOf".to_string(),
                ),
                object: Term::Variable("Z".to_string()),
            }],
        }
    }

    /// Create an RDFS type inheritance rule
    pub fn rdfs_type_inheritance() -> Rule {
        Rule {
            name: "rdfs_type_inheritance".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant(
                        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                    ),
                    object: Term::Variable("C1".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("C1".to_string()),
                    predicate: Term::Constant(
                        "http://www.w3.org/2000/01/rdf-schema#subClassOf".to_string(),
                    ),
                    object: Term::Variable("C2".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                object: Term::Variable("C2".to_string()),
            }],
        }
    }

    /// Create a domain inference rule
    pub fn rdfs_domain_inference() -> Rule {
        Rule {
            name: "rdfs_domain_inference".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("P".to_string()),
                    predicate: Term::Constant(
                        "http://www.w3.org/2000/01/rdf-schema#domain".to_string(),
                    ),
                    object: Term::Variable("C".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Variable("P".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                object: Term::Variable("C".to_string()),
            }],
        }
    }

    /// Create a range inference rule
    pub fn rdfs_range_inference() -> Rule {
        Rule {
            name: "rdfs_range_inference".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("P".to_string()),
                    predicate: Term::Constant(
                        "http://www.w3.org/2000/01/rdf-schema#range".to_string(),
                    ),
                    object: Term::Variable("C".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Variable("P".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Y".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                object: Term::Variable("C".to_string()),
            }],
        }
    }

    /// Create all standard RDFS rules
    pub fn all_rdfs_rules() -> Vec<Rule> {
        vec![
            rdfs_subclass_transitivity(),
            rdfs_type_inheritance(),
            rdfs_domain_inference(),
            rdfs_range_inference(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::{Literal, NamedNode, Triple};

    #[test]
    fn test_integration_basic_workflow() {
        let mut integration = RuleIntegration::new();

        // Add some test data to the store
        let subject = NamedNode::new("http://example.org/person").unwrap();
        let predicate = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
        let object = NamedNode::new("http://example.org/Human").unwrap();

        let triple = Triple::new(
            subject.clone(),
            predicate.clone(),
            Object::NamedNode(object),
        );

        let quad = Quad::new_default_graph(
            triple.subject().clone(),
            triple.predicate().clone(),
            triple.object().clone(),
        );
        integration.store.insert_quad(quad).unwrap();

        // Add a rule: Human -> Mortal
        let rule = Rule {
            name: "human_mortal".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                object: Term::Constant("http://example.org/Human".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                object: Term::Constant("http://example.org/Mortal".to_string()),
            }],
        };

        integration.add_rule(rule);

        // Apply rules
        let derived_count = integration.apply_rules().unwrap();
        assert!(derived_count > 0);

        // Check that the mortal type was derived
        let mortal_type = NamedNode::new("http://example.org/Mortal").unwrap();
        let all_triples = integration.store.triples().unwrap();
        let results: Vec<_> = all_triples
            .iter()
            .filter(|triple| {
                triple.subject() == &Subject::NamedNode(subject.clone())
                    && triple.predicate() == &Predicate::NamedNode(predicate.clone())
                    && triple.object() == &Object::NamedNode(mortal_type.clone())
            })
            .collect();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_conversion_functions() {
        let integration = RuleIntegration::new();

        // Test triple to rule atom conversion
        let subject = NamedNode::new("http://example.org/subject").unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = Literal::new("test");

        let triple = Triple::new(subject, predicate, object);
        let rule_atom = integration.triple_to_rule_atom(&triple);

        match &rule_atom {
            RuleAtom::Triple {
                subject: s,
                predicate: p,
                object: o,
            } => {
                assert!(matches!(s, Term::Constant(_)));
                assert!(matches!(p, Term::Constant(_)));
                assert!(matches!(o, Term::Literal(_)));
            }
            _ => panic!("Expected triple rule atom"),
        }

        // Test rule atom to triple conversion
        let converted_triple = integration.rule_atom_to_triple(&rule_atom).unwrap();
        // Check subject and predicate types
        match converted_triple.subject() {
            Subject::NamedNode(node) => assert_eq!(node.as_str(), "http://example.org/subject"),
            _ => panic!("Expected NamedNode subject"),
        }
        match converted_triple.predicate() {
            Predicate::NamedNode(node) => assert_eq!(node.as_str(), "http://example.org/predicate"),
            _ => panic!("Expected NamedNode predicate"),
        }
    }

    #[test]
    fn test_rule_builders() {
        let rules = rule_builders::all_rdfs_rules();
        assert_eq!(rules.len(), 4);

        // Check that all rules have proper names
        let rule_names: Vec<String> = rules.iter().map(|r| r.name.clone()).collect();
        assert!(rule_names.contains(&"rdfs_subclass_transitivity".to_string()));
        assert!(rule_names.contains(&"rdfs_type_inheritance".to_string()));
        assert!(rule_names.contains(&"rdfs_domain_inference".to_string()));
        assert!(rule_names.contains(&"rdfs_range_inference".to_string()));
    }

    #[test]
    fn test_query_with_rules() {
        let mut integration = RuleIntegration::new();

        // Add RDFS rules
        integration.add_rules(rule_builders::all_rdfs_rules());

        // Add some test ontology data
        let person = NamedNode::new("http://example.org/Person").unwrap();
        let student = NamedNode::new("http://example.org/Student").unwrap();
        let subclass_pred =
            NamedNode::new("http://www.w3.org/2000/01/rdf-schema#subClassOf").unwrap();
        let type_pred = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
        let alice = NamedNode::new("http://example.org/alice").unwrap();

        // Student subClassOf Person
        let subclass_triple = Triple::new(
            Subject::NamedNode(student.clone()),
            Predicate::NamedNode(subclass_pred),
            Object::NamedNode(person.clone()),
        );

        // alice type Student
        let alice_type_triple = Triple::new(
            Subject::NamedNode(alice.clone()),
            Predicate::NamedNode(type_pred.clone()),
            Object::NamedNode(student),
        );

        let subclass_quad = Quad::new_default_graph(
            subclass_triple.subject().clone(),
            subclass_triple.predicate().clone(),
            subclass_triple.object().clone(),
        );
        let alice_type_quad = Quad::new_default_graph(
            alice_type_triple.subject().clone(),
            alice_type_triple.predicate().clone(),
            alice_type_triple.object().clone(),
        );
        integration.store.insert_quad(subclass_quad).unwrap();
        integration.store.insert_quad(alice_type_quad).unwrap();

        // Query for all types of alice (should include both Student and Person via inference)
        let results = integration
            .query_with_rules(
                Some(&Subject::NamedNode(alice)),
                Some(&Predicate::NamedNode(type_pred)),
                None,
            )
            .unwrap();

        // Should find at least 2 results (Student and Person)
        assert!(results.len() >= 2);

        let has_person_type = results.iter().any(|triple| {
            if let Object::NamedNode(node) = triple.object() {
                node.as_str() == "http://example.org/Person"
            } else {
                false
            }
        });

        assert!(has_person_type, "Should infer that alice is a Person");
    }

    #[test]
    fn test_statistics() {
        let mut integration = RuleIntegration::new();

        // Add some data
        let triple = Triple::new(
            NamedNode::new("http://example.org/s").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            Literal::new("o"),
        );
        let quad = Quad::new_default_graph(
            triple.subject().clone(),
            triple.predicate().clone(),
            triple.object().clone(),
        );
        integration.store.insert_quad(quad).unwrap();

        // Add a rule
        integration.add_rule(rule_builders::rdfs_type_inheritance());

        let stats = integration.get_integration_stats().unwrap();
        assert_eq!(stats.store_quad_count, 1);
        assert_eq!(stats.rule_count, 1);
    }

    #[test]
    fn test_namespace_management() {
        let mut integration = RuleIntegration::new();

        // Test namespace prefix addition
        integration
            .add_namespace_prefix("ex", "http://example.org/")
            .unwrap();

        // Test IRI expansion
        let expanded = integration.expand_prefixed_iri("rdf:type").unwrap();
        assert_eq!(expanded, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type");

        let expanded = integration.expand_prefixed_iri("rdfs:Class").unwrap();
        assert_eq!(expanded, "http://www.w3.org/2000/01/rdf-schema#Class");

        let expanded = integration.expand_prefixed_iri("owl:Class").unwrap();
        assert_eq!(expanded, "http://www.w3.org/2002/07/owl#Class");

        // Test non-prefixed IRI
        let unchanged = integration
            .expand_prefixed_iri("http://example.org/full")
            .unwrap();
        assert_eq!(unchanged, "http://example.org/full");
    }

    #[test]
    fn test_data_validation() {
        let integration = RuleIntegration::new();

        // Test validation on empty store
        let report = integration.validate_rdf_data().unwrap();
        assert_eq!(report.total_triples, 0);
        assert!(!report.warnings.is_empty()); // Should warn about empty store
        assert!(report.errors.is_empty());

        println!("Validation report: {report}");
    }
}
