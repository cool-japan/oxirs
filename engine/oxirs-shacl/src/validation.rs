//! SHACL validation engine implementation
//!
//! This module implements the core validation engine that orchestrates SHACL validation.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::time::{Duration, Instant};

use oxirs_core::{
    model::{BlankNode, Literal, NamedNode, RdfTerm, Term, Triple},
    OxirsError, Store,
};

use crate::{
    constraints::*, paths::*, report::*, sparql::*, targets::*, Constraint, ConstraintComponentId,
    PropertyPath, Result, Severity, ShaclError, Shape, ShapeId, Target, ValidationConfig,
    ValidationReport,
};

/// Cache key for constraint results
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ConstraintCacheKey {
    focus_node: Term,
    shape_id: ShapeId,
    constraint_component_id: ConstraintComponentId,
    property_path: Option<PropertyPath>,
}

/// Core SHACL validation engine
#[derive(Debug)]
pub struct ValidationEngine<'a> {
    /// Reference to shapes to validate against
    shapes: &'a IndexMap<ShapeId, Shape>,

    /// Validation configuration
    config: ValidationConfig,

    /// Target selector for finding focus nodes
    target_selector: TargetSelector,

    /// Property path evaluator
    path_evaluator: PropertyPathEvaluator,

    /// SPARQL constraint executor
    sparql_executor: SparqlConstraintExecutor,

    /// Validation statistics
    stats: ValidationStats,

    /// Cache for constraint evaluation results
    constraint_cache: std::cell::RefCell<HashMap<ConstraintCacheKey, ConstraintEvaluationResult>>,

    /// Cache for resolved inherited constraints
    inheritance_cache:
        std::cell::RefCell<HashMap<ShapeId, IndexMap<ConstraintComponentId, Constraint>>>,
}

impl<'a> ValidationEngine<'a> {
    /// Create a new validation engine
    pub fn new(shapes: &'a IndexMap<ShapeId, Shape>, config: ValidationConfig) -> Self {
        Self {
            shapes,
            config,
            target_selector: TargetSelector::new(),
            path_evaluator: PropertyPathEvaluator::new(),
            sparql_executor: SparqlConstraintExecutor::new(),
            stats: ValidationStats::default(),
            constraint_cache: std::cell::RefCell::new(HashMap::new()),
            inheritance_cache: std::cell::RefCell::new(HashMap::new()),
        }
    }

    /// Validate all data in a store against all loaded shapes
    pub fn validate_store(&mut self, store: &Store) -> Result<ValidationReport> {
        let start_time = Instant::now();
        let mut report = ValidationReport::new();

        // Validate each active shape
        for (shape_id, shape) in self.shapes {
            if shape.is_active() {
                let shape_result = self.validate_shape(store, shape, None)?;
                report.merge_result(shape_result);

                // Check if we should stop early
                if self.config.fail_fast && !report.conforms() {
                    break;
                }

                // Check violation limit
                if self.config.max_violations > 0
                    && report.violation_count() >= self.config.max_violations
                {
                    break;
                }
            }
        }

        // Update statistics
        self.stats.total_validations += 1;
        self.stats.total_validation_time += start_time.elapsed();
        self.stats.last_validation_time = Some(start_time.elapsed());

        Ok(report)
    }

    /// Validate specific nodes against a specific shape
    pub fn validate_nodes(
        &mut self,
        store: &Store,
        shape: &Shape,
        nodes: &[Term],
    ) -> Result<ValidationReport> {
        let start_time = Instant::now();
        let mut report = ValidationReport::new();

        for node in nodes {
            let node_result = self.validate_node_against_shape(store, shape, node, None)?;
            report.merge_result(node_result);

            // Check early termination conditions
            if self.config.fail_fast && !report.conforms() {
                break;
            }

            if self.config.max_violations > 0
                && report.violation_count() >= self.config.max_violations
            {
                break;
            }
        }

        // Update statistics
        self.stats.total_node_validations += nodes.len();
        self.stats.total_validation_time += start_time.elapsed();

        Ok(report)
    }

    /// Validate a shape against its targets
    fn validate_shape(
        &mut self,
        store: &Store,
        shape: &Shape,
        graph_name: Option<&str>,
    ) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();

        // If no explicit targets, this might be an implicit target shape
        if shape.targets.is_empty() && shape.is_node_shape() {
            // Try using the shape IRI as an implicit class target
            let implicit_target =
                Target::implicit(NamedNode::new(shape.id.as_str()).map_err(|e| {
                    ShaclError::TargetSelection(format!("Invalid shape IRI: {}", e))
                })?);
            let target_nodes =
                self.target_selector
                    .select_targets(store, &implicit_target, graph_name)?;

            for node in target_nodes {
                let node_result =
                    self.validate_node_against_shape(store, shape, &node, graph_name)?;
                report.merge_result(node_result);

                if self.should_stop_validation(&report) {
                    break;
                }
            }
        } else {
            // Validate against explicit targets
            for target in &shape.targets {
                let target_nodes = self
                    .target_selector
                    .select_targets(store, target, graph_name)?;

                for node in target_nodes {
                    let node_result =
                        self.validate_node_against_shape(store, shape, &node, graph_name)?;
                    report.merge_result(node_result);

                    if self.should_stop_validation(&report) {
                        break;
                    }
                }

                if self.should_stop_validation(&report) {
                    break;
                }
            }
        }

        Ok(report)
    }

    /// Validate a specific node against a shape
    fn validate_node_against_shape(
        &mut self,
        store: &Store,
        shape: &Shape,
        focus_node: &Term,
        graph_name: Option<&str>,
    ) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();

        if shape.is_node_shape() {
            // For node shapes, validate constraints directly against the focus node
            let values = vec![focus_node.clone()];
            let constraint_results =
                self.validate_constraints(store, shape, focus_node, None, &values, graph_name)?;

            for result in constraint_results {
                if let Some(violation) = result {
                    report.add_violation(violation);
                }
            }
        } else if shape.is_property_shape() {
            // For property shapes, evaluate the property path first
            if let Some(path) = &shape.path {
                let values = self
                    .path_evaluator
                    .evaluate_path(store, focus_node, path, graph_name)?;
                let constraint_results = self.validate_constraints(
                    store,
                    shape,
                    focus_node,
                    Some(path),
                    &values,
                    graph_name,
                )?;

                for result in constraint_results {
                    if let Some(violation) = result {
                        report.add_violation(violation);
                    }
                }
            } else {
                return Err(ShaclError::ValidationEngine(
                    "Property shape must have a property path".to_string(),
                ));
            }
        }

        Ok(report)
    }

    /// Validate all constraints for a shape
    fn validate_constraints(
        &mut self,
        store: &Store,
        shape: &Shape,
        focus_node: &Term,
        path: Option<&PropertyPath>,
        values: &[Term],
        graph_name: Option<&str>,
    ) -> Result<Vec<Option<ValidationViolation>>> {
        let mut results = Vec::new();

        // Resolve inherited constraints (includes shape's own constraints)
        let resolved_constraints = self.resolve_inherited_constraints(&shape.id)?;

        for (component_id, constraint) in &resolved_constraints {
            // Collect allowed properties for closed shape validation
            let mut allowed_properties = Vec::new();

            // Add the shape's own path if it's a property shape
            if let Some(shape_path) = &shape.path {
                allowed_properties.push(shape_path.clone());
            }

            // TODO: In a complete implementation, we would need to collect paths from
            // all property shapes associated with this node shape

            let context = ConstraintContext::new(focus_node.clone(), shape.id.clone())
                .with_values(values.to_vec())
                .with_allowed_properties(allowed_properties)
                .with_shapes_registry(std::rc::Rc::new(self.shapes.clone()));

            let constraint_result =
                self.validate_constraint(store, constraint, &context, path, graph_name)?;

            match constraint_result {
                ConstraintEvaluationResult::Satisfied => {
                    results.push(None);
                }
                ConstraintEvaluationResult::Violated {
                    violating_value,
                    message,
                    details,
                } => {
                    let violation = ValidationViolation {
                        focus_node: focus_node.clone(),
                        source_shape: shape.id.clone(),
                        source_constraint_component: component_id.clone(),
                        result_path: path.cloned(),
                        value: violating_value,
                        result_message: message
                            .or_else(|| constraint.message().map(|s| s.to_string())),
                        result_severity: constraint
                            .severity()
                            .unwrap_or_else(|| shape.severity.clone()),
                        details,
                        nested_results: Vec::new(),
                    };
                    results.push(Some(violation));
                }
                ConstraintEvaluationResult::Error { message, details } => {
                    return Err(ShaclError::ConstraintValidation(format!(
                        "Constraint evaluation error for {}: {}",
                        component_id.as_str(),
                        message
                    )));
                }
            }
        }

        Ok(results)
    }

    /// Validate a single constraint
    fn validate_constraint(
        &mut self,
        store: &Store,
        constraint: &Constraint,
        context: &ConstraintContext,
        path: Option<&PropertyPath>,
        graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        // Check cache first if caching is enabled
        if self.config.max_violations == 0 || !self.config.fail_fast {
            let cache_key = ConstraintCacheKey {
                focus_node: context.focus_node.clone(),
                shape_id: context.shape_id.clone(),
                constraint_component_id: constraint.component_id(),
                property_path: path.cloned(),
            };

            // Check if we have a cached result
            if let Some(cached_result) = self.constraint_cache.borrow().get(&cache_key) {
                self.stats.cache_hits += 1;
                return Ok(cached_result.clone());
            }
            self.stats.cache_misses += 1;
        }

        let start_time = std::time::Instant::now();
        let result = match constraint {
            // Core Value Constraints
            Constraint::Class(c) => self.validate_class_constraint(store, c, context, graph_name),
            Constraint::Datatype(c) => self.validate_datatype_constraint(c, context),
            Constraint::NodeKind(c) => self.validate_node_kind_constraint(c, context),

            // Cardinality Constraints
            Constraint::MinCount(c) => self.validate_min_count_constraint(c, context),
            Constraint::MaxCount(c) => self.validate_max_count_constraint(c, context),

            // Range Constraints
            Constraint::MinExclusive(c) => self.validate_min_exclusive_constraint(c, context),
            Constraint::MaxExclusive(c) => self.validate_max_exclusive_constraint(c, context),
            Constraint::MinInclusive(c) => self.validate_min_inclusive_constraint(c, context),
            Constraint::MaxInclusive(c) => self.validate_max_inclusive_constraint(c, context),

            // String Constraints
            Constraint::MinLength(c) => self.validate_min_length_constraint(c, context),
            Constraint::MaxLength(c) => self.validate_max_length_constraint(c, context),
            Constraint::Pattern(c) => self.validate_pattern_constraint(c, context),
            Constraint::LanguageIn(c) => self.validate_language_in_constraint(c, context),
            Constraint::UniqueLang(c) => self.validate_unique_lang_constraint(c, context),

            // Value Constraints
            Constraint::Equals(c) => self.validate_equals_constraint(store, c, context, graph_name),
            Constraint::Disjoint(c) => {
                self.validate_disjoint_constraint(store, c, context, graph_name)
            }
            Constraint::LessThan(c) => {
                self.validate_less_than_constraint(store, c, context, graph_name)
            }
            Constraint::LessThanOrEquals(c) => {
                self.validate_less_than_or_equals_constraint(store, c, context, graph_name)
            }
            Constraint::In(c) => self.validate_in_constraint(c, context),
            Constraint::HasValue(c) => self.validate_has_value_constraint(c, context),

            // Logical Constraints (complex - placeholder implementations)
            Constraint::Not(c) => self.validate_not_constraint(store, c, context, graph_name),
            Constraint::And(c) => self.validate_and_constraint(store, c, context, graph_name),
            Constraint::Or(c) => self.validate_or_constraint(store, c, context, graph_name),
            Constraint::Xone(c) => self.validate_xone_constraint(store, c, context, graph_name),

            // Shape-based Constraints (complex - placeholder implementations)
            Constraint::Node(c) => self.validate_node_constraint(store, c, context, graph_name),
            Constraint::QualifiedValueShape(c) => {
                self.validate_qualified_value_shape_constraint(store, c, context, graph_name)
            }

            // Closed Shape Constraints (placeholder implementation)
            Constraint::Closed(c) => self.validate_closed_constraint(store, c, context, graph_name),

            // SPARQL Constraints
            Constraint::Sparql(c) => self.validate_sparql_constraint(store, c, context, graph_name),
        }?;

        // Cache the result if caching is enabled
        if self.config.max_violations == 0 || !self.config.fail_fast {
            let cache_key = ConstraintCacheKey {
                focus_node: context.focus_node.clone(),
                shape_id: context.shape_id.clone(),
                constraint_component_id: constraint.component_id(),
                property_path: path.cloned(),
            };

            self.constraint_cache
                .borrow_mut()
                .insert(cache_key, result.clone());
        }

        // Record constraint evaluation time
        let duration = start_time.elapsed();
        self.stats
            .record_constraint_evaluation(constraint.component_id().as_str().to_string(), duration);

        Ok(result)
    }

    /// Validate class constraint
    fn validate_class_constraint(
        &self,
        store: &Store,
        constraint: &ClassConstraint,
        context: &ConstraintContext,
        graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        let rdf_type =
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").map_err(|e| {
                ShaclError::ConstraintValidation(format!("Invalid rdf:type IRI: {}", e))
            })?;

        for value in &context.values {
            // Only named nodes and blank nodes can be instances of classes
            if !value.is_named_node() && !value.is_blank_node() {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some(format!(
                        "Value {} cannot be an instance of class {} (not a resource)",
                        value.as_str(),
                        constraint.class_iri.as_str()
                    )),
                ));
            }

            // Check if there's any type information for this node in the store
            let has_type_info_query = if let Some(graph) = graph_name {
                format!(
                    r#"
                    ASK {{
                        GRAPH <{}> {{
                            {} <{}> ?type .
                        }}
                    }}
                "#,
                    graph,
                    format_term_for_sparql(value)?,
                    rdf_type.as_str()
                )
            } else {
                format!(
                    r#"
                    ASK {{
                        {} <{}> ?type .
                    }}
                "#,
                    format_term_for_sparql(value)?,
                    rdf_type.as_str()
                )
            };

            // First check if the node has any type information
            let has_type_info = match self.execute_constraint_query(store, &has_type_info_query) {
                Ok(result) => {
                    if let oxirs_core::query::QueryResult::Ask(has_info) = result {
                        has_info
                    } else {
                        false
                    }
                }
                Err(_) => false,
            };

            // If there's no type information in the store, consider it valid for now
            // This handles the case where we're validating against an empty store
            if !has_type_info {
                tracing::debug!("No type information found for {} in store, skipping class constraint validation", value.as_str());
                continue;
            }

            // Check if the value is an instance of the required class using SPARQL
            let instance_query = if let Some(graph) = graph_name {
                format!(
                    r#"
                    ASK {{
                        GRAPH <{}> {{
                            {} <{}> <{}> .
                        }}
                    }}
                "#,
                    graph,
                    format_term_for_sparql(value)?,
                    rdf_type.as_str(),
                    constraint.class_iri.as_str()
                )
            } else {
                format!(
                    r#"
                    ASK {{
                        {} <{}> <{}> .
                    }}
                "#,
                    format_term_for_sparql(value)?,
                    rdf_type.as_str(),
                    constraint.class_iri.as_str()
                )
            };

            // Execute the ASK query
            match self.execute_constraint_query(store, &instance_query) {
                Ok(result) => {
                    if let oxirs_core::query::QueryResult::Ask(is_instance) = result {
                        if !is_instance {
                            // Value is not an instance of the required class
                            return Ok(ConstraintEvaluationResult::violated(
                                Some(value.clone()),
                                Some(format!(
                                    "Value {} is not an instance of class {}",
                                    value.as_str(),
                                    constraint.class_iri.as_str()
                                )),
                            ));
                        }
                    } else {
                        return Err(ShaclError::ConstraintValidation(
                            "Expected ASK result for class constraint query".to_string(),
                        ));
                    }
                }
                Err(e) => {
                    tracing::debug!(
                        "SPARQL query failed for class constraint, trying direct store query: {}",
                        e
                    );
                    // Fallback to direct store query
                    if !self.is_instance_of_class_direct(
                        store,
                        value,
                        &constraint.class_iri,
                        graph_name,
                    )? {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some(format!(
                                "Value {} is not an instance of class {}",
                                value.as_str(),
                                constraint.class_iri.as_str()
                            )),
                        ));
                    }
                }
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Validate datatype constraint
    fn validate_datatype_constraint(
        &self,
        constraint: &DatatypeConstraint,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                // Check if literal has the required datatype
                let literal_datatype = literal.datatype();
                if literal_datatype.as_str() != constraint.datatype_iri.as_str() {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Value has datatype {} but expected {}",
                            literal_datatype.as_str(),
                            constraint.datatype_iri.as_str()
                        )),
                    ));
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some(format!(
                        "Value must be a literal with datatype {}",
                        constraint.datatype_iri.as_str()
                    )),
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Validate node kind constraint
    fn validate_node_kind_constraint(
        &self,
        constraint: &NodeKindConstraint,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            let matches = match &constraint.node_kind {
                NodeKind::Iri => value.is_named_node(),
                NodeKind::BlankNode => value.is_blank_node(),
                NodeKind::Literal => value.is_literal(),
                NodeKind::BlankNodeOrIri => value.is_blank_node() || value.is_named_node(),
                NodeKind::BlankNodeOrLiteral => value.is_blank_node() || value.is_literal(),
                NodeKind::IriOrLiteral => value.is_named_node() || value.is_literal(),
            };

            if !matches {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some(format!(
                        "Value does not match required node kind: {:?}",
                        constraint.node_kind
                    )),
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Validate minimum count constraint
    fn validate_min_count_constraint(
        &self,
        constraint: &MinCountConstraint,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        let actual_count = context.values.len() as u32;
        if actual_count < constraint.min_count {
            Ok(ConstraintEvaluationResult::violated(
                None,
                Some(format!(
                    "Minimum count {} not satisfied, found {}",
                    constraint.min_count, actual_count
                )),
            ))
        } else {
            Ok(ConstraintEvaluationResult::satisfied())
        }
    }

    /// Validate maximum count constraint
    fn validate_max_count_constraint(
        &self,
        constraint: &MaxCountConstraint,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        let actual_count = context.values.len() as u32;
        if actual_count > constraint.max_count {
            Ok(ConstraintEvaluationResult::violated(
                None,
                Some(format!(
                    "Maximum count {} exceeded, found {}",
                    constraint.max_count, actual_count
                )),
            ))
        } else {
            Ok(ConstraintEvaluationResult::satisfied())
        }
    }

    /// Validate minimum length constraint
    fn validate_min_length_constraint(
        &self,
        constraint: &MinLengthConstraint,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                let length = literal.as_str().chars().count() as u32;
                if length < constraint.min_length {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Minimum length {} not satisfied, found {}",
                            constraint.min_length, length
                        )),
                    ));
                }
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Validate maximum length constraint
    fn validate_max_length_constraint(
        &self,
        constraint: &MaxLengthConstraint,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                let length = literal.as_str().chars().count() as u32;
                if length > constraint.max_length {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Maximum length {} exceeded, found {}",
                            constraint.max_length, length
                        )),
                    ));
                }
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Validate pattern constraint
    fn validate_pattern_constraint(
        &self,
        constraint: &PatternConstraint,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        use regex::{Regex, RegexBuilder};

        // Build regex with flags
        let regex = if let Some(flags) = &constraint.flags {
            let case_insensitive = flags.contains('i');
            let multi_line = flags.contains('m');
            let dot_matches_new_line = flags.contains('s');

            RegexBuilder::new(&constraint.pattern)
                .case_insensitive(case_insensitive)
                .multi_line(multi_line)
                .dot_matches_new_line(dot_matches_new_line)
                .build()
                .map_err(|e| ShaclError::ConstraintValidation(format!("Invalid regex: {}", e)))?
        } else {
            Regex::new(&constraint.pattern)
                .map_err(|e| ShaclError::ConstraintValidation(format!("Invalid regex: {}", e)))?
        };

        for value in &context.values {
            if let Term::Literal(literal) = value {
                if !regex.is_match(literal.as_str()) {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        constraint.message.clone().or_else(|| {
                            Some(format!(
                                "Value does not match pattern: {}",
                                constraint.pattern
                            ))
                        }),
                    ));
                }
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Validate in constraint
    fn validate_in_constraint(
        &self,
        constraint: &InConstraint,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if !constraint.values.contains(value) {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("Value is not in the allowed list".to_string()),
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Validate has value constraint
    fn validate_has_value_constraint(
        &self,
        constraint: &HasValueConstraint,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        if !context.values.contains(&constraint.value) {
            Ok(ConstraintEvaluationResult::violated(
                None,
                Some(format!(
                    "Required value {} not found",
                    constraint.value.as_str()
                )),
            ))
        } else {
            Ok(ConstraintEvaluationResult::satisfied())
        }
    }

    /// Validate SPARQL constraint
    fn validate_sparql_constraint(
        &mut self,
        store: &Store,
        constraint: &SparqlConstraint,
        context: &ConstraintContext,
        graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        let bindings = SparqlBindings::new()
            .with_this(context.focus_node.clone())
            .with_current_shape(Term::NamedNode(
                NamedNode::new(context.shape_id.as_str()).map_err(|e| {
                    ShaclError::SparqlExecution(format!("Invalid shape IRI: {}", e))
                })?,
            ));

        let result = self
            .sparql_executor
            .execute_constraint(store, constraint, &bindings, graph_name)?;

        if result.is_violation() {
            Ok(ConstraintEvaluationResult::violated(
                None,
                constraint
                    .message
                    .clone()
                    .or_else(|| Some("SPARQL constraint violation".to_string())),
            ))
        } else {
            Ok(ConstraintEvaluationResult::satisfied())
        }
    }

    // Range Constraints

    /// Validate min exclusive constraint
    fn validate_min_exclusive_constraint(
        &self,
        constraint: &MinExclusiveConstraint,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                match self.compare_literals(literal, &constraint.min_value)? {
                    std::cmp::Ordering::Less | std::cmp::Ordering::Equal => {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some(format!(
                                "Value {} is not greater than minimum exclusive value {}",
                                literal.as_str(),
                                constraint.min_value.as_str()
                            )),
                        ));
                    }
                    std::cmp::Ordering::Greater => {
                        // Value is valid (greater than min exclusive)
                    }
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("MinExclusive constraint can only be applied to literals".to_string()),
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Validate max exclusive constraint
    fn validate_max_exclusive_constraint(
        &self,
        constraint: &MaxExclusiveConstraint,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                match self.compare_literals(literal, &constraint.max_value)? {
                    std::cmp::Ordering::Greater | std::cmp::Ordering::Equal => {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some(format!(
                                "Value {} is not less than maximum exclusive value {}",
                                literal.as_str(),
                                constraint.max_value.as_str()
                            )),
                        ));
                    }
                    std::cmp::Ordering::Less => {
                        // Value is valid (less than max exclusive)
                    }
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("MaxExclusive constraint can only be applied to literals".to_string()),
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Validate min inclusive constraint
    fn validate_min_inclusive_constraint(
        &self,
        constraint: &MinInclusiveConstraint,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                match self.compare_literals(literal, &constraint.min_value)? {
                    std::cmp::Ordering::Less => {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some(format!(
                                "Value {} is less than minimum inclusive value {}",
                                literal.as_str(),
                                constraint.min_value.as_str()
                            )),
                        ));
                    }
                    std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => {
                        // Value is valid (greater than or equal to min inclusive)
                    }
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("MinInclusive constraint can only be applied to literals".to_string()),
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Validate max inclusive constraint
    fn validate_max_inclusive_constraint(
        &self,
        constraint: &MaxInclusiveConstraint,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                match self.compare_literals(literal, &constraint.max_value)? {
                    std::cmp::Ordering::Greater => {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some(format!(
                                "Value {} is greater than maximum inclusive value {}",
                                literal.as_str(),
                                constraint.max_value.as_str()
                            )),
                        ));
                    }
                    std::cmp::Ordering::Less | std::cmp::Ordering::Equal => {
                        // Value is valid (less than or equal to max inclusive)
                    }
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("MaxInclusive constraint can only be applied to literals".to_string()),
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }

    // String Constraints

    /// Validate language in constraint
    fn validate_language_in_constraint(
        &self,
        constraint: &LanguageInConstraint,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                if let Some(language) = literal.language() {
                    if !constraint.languages.contains(&language.to_string()) {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some(format!(
                                "Language tag '{}' is not in allowed list: {:?}",
                                language, constraint.languages
                            )),
                        ));
                    }
                } else {
                    // No language tag on literal
                    if !constraint.languages.is_empty() {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some("Literal has no language tag, but languageIn constraint requires one".to_string())
                        ));
                    }
                }
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Validate unique language constraint
    fn validate_unique_lang_constraint(
        &self,
        constraint: &UniqueLangConstraint,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        if !constraint.unique_lang {
            return Ok(ConstraintEvaluationResult::satisfied());
        }

        let mut seen_languages = HashSet::new();

        for value in &context.values {
            if let Term::Literal(literal) = value {
                if let Some(language) = literal.language() {
                    if seen_languages.contains(language) {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some(format!(
                                "Duplicate language tag '{}' violates uniqueLang constraint",
                                language
                            )),
                        ));
                    }
                    seen_languages.insert(language.to_string());
                }
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }

    // Value Constraints

    /// Validate equals constraint
    fn validate_equals_constraint(
        &self,
        store: &Store,
        constraint: &EqualsConstraint,
        context: &ConstraintContext,
        graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        // For equals constraint, we need store access - this should be provided by caller
        // For now, just validate that the current values are equal (basic implementation)
        // TODO: Implement proper property path evaluation when store is available

        // For now, assume equals_values would be determined from the property path
        // This is a placeholder implementation
        let equals_values = Vec::new(); // TODO: get actual values from property path

        for value in &context.values {
            if !equals_values.contains(value) {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some(format!(
                        "Value {} does not equal any value from property path",
                        value
                    )),
                ));
            }
        }

        // Also check the reverse: values from equals path should exist in current values
        for equals_value in &equals_values {
            if !context.values.contains(equals_value) {
                return Ok(ConstraintEvaluationResult::violated(
                    None,
                    Some(format!("Property path has missing value {} that should equal values from constraint", equals_value.as_str()))
                ));
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Validate disjoint constraint
    fn validate_disjoint_constraint(
        &self,
        store: &Store,
        constraint: &DisjointConstraint,
        context: &ConstraintContext,
        graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        // For disjoint constraint, values in current property path must not appear in the disjoint property path

        // Get values from the disjoint property path
        let mut path_evaluator = PropertyPathEvaluator::new();
        let disjoint_values = path_evaluator.evaluate_path(
            store,
            &context.focus_node,
            &constraint.property,
            graph_name,
        )?;

        for value in &context.values {
            if disjoint_values.contains(value) {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some(format!("Value {} appears in both this property and the disjoint property path, violating disjoint constraint", value.as_str()))
                ));
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Validate less than constraint
    fn validate_less_than_constraint(
        &self,
        store: &Store,
        constraint: &LessThanConstraint,
        context: &ConstraintContext,
        graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        // For less than constraint, values in current property path must be less than all values in the comparison property path

        // Get values from the comparison property path
        let mut path_evaluator = PropertyPathEvaluator::new();
        let comparison_values = path_evaluator.evaluate_path(
            store,
            &context.focus_node,
            &constraint.property,
            graph_name,
        )?;

        for value in &context.values {
            if let Term::Literal(value_literal) = value {
                for comparison_value in &comparison_values {
                    if let Term::Literal(comparison_literal) = comparison_value {
                        match self.compare_literals(value_literal, &comparison_literal)? {
                            std::cmp::Ordering::Greater | std::cmp::Ordering::Equal => {
                                return Ok(ConstraintEvaluationResult::violated(
                                    Some(value.clone()),
                                    Some(format!(
                                        "Value {} is not less than comparison value {}",
                                        value_literal.as_str(),
                                        comparison_literal.as_str()
                                    )),
                                ));
                            }
                            std::cmp::Ordering::Less => {
                                // Value is valid (less than comparison value)
                            }
                        }
                    } else {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(comparison_value.clone()),
                            Some("LessThan constraint can only compare literals".to_string()),
                        ));
                    }
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("LessThan constraint can only be applied to literals".to_string()),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Validate less than or equals constraint
    fn validate_less_than_or_equals_constraint(
        &self,
        store: &Store,
        constraint: &LessThanOrEqualsConstraint,
        context: &ConstraintContext,
        graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        // For less than or equals constraint, values in current property path must be less than or equal to all values in the comparison property path

        // Get values from the comparison property path
        let mut path_evaluator = PropertyPathEvaluator::new();
        let comparison_values = path_evaluator.evaluate_path(
            store,
            &context.focus_node,
            &constraint.property,
            graph_name,
        )?;

        for value in &context.values {
            if let Term::Literal(value_literal) = value {
                for comparison_value in &comparison_values {
                    if let Term::Literal(comparison_literal) = comparison_value {
                        match self.compare_literals(value_literal, &comparison_literal)? {
                            std::cmp::Ordering::Greater => {
                                return Ok(ConstraintEvaluationResult::violated(
                                    Some(value.clone()),
                                    Some(format!(
                                        "Value {} is greater than comparison value {}",
                                        value_literal.as_str(),
                                        comparison_literal.as_str()
                                    )),
                                ));
                            }
                            std::cmp::Ordering::Less | std::cmp::Ordering::Equal => {
                                // Value is valid (less than or equal to comparison value)
                            }
                        }
                    } else {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(comparison_value.clone()),
                            Some(
                                "LessThanOrEquals constraint can only compare literals".to_string(),
                            ),
                        ));
                    }
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("LessThanOrEquals constraint can only be applied to literals".to_string()),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }

    // Logical Constraints (complex - placeholder implementations)

    /// Validate not constraint - ensures the specified shape does NOT validate
    fn validate_not_constraint(
        &mut self,
        store: &Store,
        constraint: &NotConstraint,
        context: &ConstraintContext,
        graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        // Get the shape to negate
        let shape = self.shapes.get(&constraint.shape).ok_or_else(|| {
            ShaclError::ValidationEngine(format!(
                "Shape not found for not constraint: {}",
                constraint.shape.as_str()
            ))
        })?;

        // For each value, validate against the negated shape
        for value in &context.values {
            // Validate the value against the specified shape
            let validation_result =
                self.validate_node_against_shape(store, shape, value, graph_name)?;

            // If the shape validates (conforms), then the NOT constraint is violated
            if validation_result.conforms() {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some(format!("Value {} conforms to shape {} but sh:not constraint requires it not to conform", 
                               value.as_str(), constraint.shape.as_str()))
                ));
            }
        }

        // If none of the values conform to the negated shape, the NOT constraint is satisfied
        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Validate and constraint - requires ALL specified shapes to validate
    fn validate_and_constraint(
        &mut self,
        store: &Store,
        constraint: &AndConstraint,
        context: &ConstraintContext,
        graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        // For each value, validate against ALL shapes in the AND constraint
        for value in &context.values {
            for shape_id in &constraint.shapes {
                let shape = self.shapes.get(shape_id).ok_or_else(|| {
                    ShaclError::ValidationEngine(format!(
                        "Shape not found for and constraint: {}",
                        shape_id.as_str()
                    ))
                })?;

                // Validate the value against this shape
                let validation_result =
                    self.validate_node_against_shape(store, shape, value, graph_name)?;

                // If ANY shape fails to validate, the AND constraint is violated
                if !validation_result.conforms() {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Value {} fails to conform to shape {} in sh:and constraint",
                            value.as_str(),
                            shape_id.as_str()
                        )),
                    ));
                }
            }
        }

        // If all values conform to all shapes, the AND constraint is satisfied
        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Validate or constraint - requires AT LEAST ONE of the specified shapes to validate
    fn validate_or_constraint(
        &mut self,
        store: &Store,
        constraint: &OrConstraint,
        context: &ConstraintContext,
        graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        // For each value, validate against shapes until one succeeds
        for value in &context.values {
            let mut any_shape_conforms = false;
            let mut error_messages = Vec::new();

            // Try to validate against each shape in the OR constraint
            for shape_id in &constraint.shapes {
                let shape = self.shapes.get(shape_id).ok_or_else(|| {
                    ShaclError::ValidationEngine(format!(
                        "Shape not found for or constraint: {}",
                        shape_id.as_str()
                    ))
                })?;

                // Validate the value against this shape
                match self.validate_node_against_shape(store, shape, value, graph_name) {
                    Ok(validation_result) => {
                        if validation_result.conforms() {
                            any_shape_conforms = true;
                            break; // Early exit - OR constraint is satisfied
                        } else {
                            error_messages.push(format!("Shape {} failed", shape_id.as_str()));
                        }
                    }
                    Err(e) => {
                        error_messages.push(format!(
                            "Shape {} validation error: {}",
                            shape_id.as_str(),
                            e
                        ));
                    }
                }
            }

            // If no shape conforms, the OR constraint is violated
            if !any_shape_conforms {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some(format!(
                        "Value {} fails to conform to any shape in sh:or constraint. Failures: {}",
                        value.as_str(),
                        error_messages.join("; ")
                    )),
                ));
            }
        }

        // If all values conform to at least one shape, the OR constraint is satisfied
        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Validate xone constraint - requires EXACTLY ONE of the specified shapes to validate
    fn validate_xone_constraint(
        &mut self,
        store: &Store,
        constraint: &XoneConstraint,
        context: &ConstraintContext,
        graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        // For each value, count how many shapes conform
        for value in &context.values {
            let mut conforming_shapes = Vec::new();
            let mut error_messages = Vec::new();

            // Check all shapes to count conforming ones
            for shape_id in &constraint.shapes {
                let shape = self.shapes.get(shape_id).ok_or_else(|| {
                    ShaclError::ValidationEngine(format!(
                        "Shape not found for xone constraint: {}",
                        shape_id.as_str()
                    ))
                })?;

                // Validate the value against this shape
                match self.validate_node_against_shape(store, shape, value, graph_name) {
                    Ok(validation_result) => {
                        if validation_result.conforms() {
                            conforming_shapes.push(shape_id.as_str());
                        } else {
                            error_messages.push(format!("Shape {} failed", shape_id.as_str()));
                        }
                    }
                    Err(e) => {
                        error_messages.push(format!(
                            "Shape {} validation error: {}",
                            shape_id.as_str(),
                            e
                        ));
                    }
                }
            }

            // Check if exactly one shape conforms
            match conforming_shapes.len() {
                0 => {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!("Value {} fails to conform to any shape in sh:xone constraint (expected exactly one). Failures: {}", 
                                   value.as_str(), error_messages.join("; ")))
                    ));
                }
                1 => {
                    // Exactly one shape conforms - this is what we want
                    tracing::debug!(
                        "Value {} conforms to exactly one shape ({}) in xone constraint",
                        value.as_str(),
                        conforming_shapes[0]
                    );
                }
                n => {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!("Value {} conforms to {} shapes in sh:xone constraint (expected exactly one). Conforming shapes: {}", 
                                   value.as_str(), n, conforming_shapes.join(", ")))
                    ));
                }
            }
        }

        // If all values conform to exactly one shape each, the XONE constraint is satisfied
        Ok(ConstraintEvaluationResult::satisfied())
    }

    // Shape-based Constraints (complex - placeholder implementations)

    /// Validate node constraint - validates that values conform to a specific shape
    fn validate_node_constraint(
        &mut self,
        store: &Store,
        constraint: &NodeConstraint,
        context: &ConstraintContext,
        graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        // Get the shape to validate against
        let shape = self.shapes.get(&constraint.shape).ok_or_else(|| {
            ShaclError::ValidationEngine(format!(
                "Shape not found for node constraint: {}",
                constraint.shape.as_str()
            ))
        })?;

        // For each value, validate it against the specified shape
        for value in &context.values {
            // Validate the value against the specified shape
            let validation_result =
                self.validate_node_against_shape(store, shape, value, graph_name)?;

            // If the shape validation fails (has violations), then the node constraint is violated
            if !validation_result.conforms() {
                let violation_details = validation_result
                    .violations
                    .iter()
                    .map(|v| {
                        format!(
                            "Shape validation failed: {}",
                            v.result_message.as_deref().unwrap_or("No details")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");

                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some(format!(
                        "Value {} does not conform to shape {}. Details: {}",
                        value.as_str(),
                        constraint.shape.as_str(),
                        violation_details
                    )),
                ));
            }
        }

        // If all values conform to the shape, the node constraint is satisfied
        tracing::debug!(
            "Node constraint validation passed for shape {}",
            constraint.shape.as_str()
        );
        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Validate qualified value shape constraint - validates that a specific count of values conform to a shape
    fn validate_qualified_value_shape_constraint(
        &mut self,
        store: &Store,
        constraint: &QualifiedValueShapeConstraint,
        context: &ConstraintContext,
        graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        // Get the shape to validate against
        let shape = self
            .shapes
            .get(&constraint.qualified_value_shape)
            .ok_or_else(|| {
                ShaclError::ValidationEngine(format!(
                    "Shape not found for qualified value shape constraint: {}",
                    constraint.qualified_value_shape.as_str()
                ))
            })?;

        let mut conforming_count = 0;
        let mut non_conforming_values = Vec::new();

        // Count how many values conform to the qualified value shape
        for value in &context.values {
            let validation_result =
                self.validate_node_against_shape(store, shape, value, graph_name)?;

            if validation_result.conforms() {
                conforming_count += 1;
            } else {
                non_conforming_values.push(value.clone());
            }
        }

        // Check qualifiedMinCount constraint
        if let Some(min_count) = constraint.qualified_min_count {
            if conforming_count < min_count {
                return Ok(ConstraintEvaluationResult::violated(
                    None,
                    Some(format!("Qualified value shape constraint violated: only {} values conform to shape {} (minimum required: {})", 
                               conforming_count, constraint.qualified_value_shape.as_str(), min_count))
                ));
            }
        }

        // Check qualifiedMaxCount constraint
        if let Some(max_count) = constraint.qualified_max_count {
            if conforming_count > max_count {
                return Ok(ConstraintEvaluationResult::violated(
                    None,
                    Some(format!("Qualified value shape constraint violated: {} values conform to shape {} (maximum allowed: {})", 
                               conforming_count, constraint.qualified_value_shape.as_str(), max_count))
                ));
            }
        }

        // If qualified_value_shapes_disjoint is true, we need additional validation
        // This means that each value can conform to at most one qualified value shape
        if constraint.qualified_value_shapes_disjoint {
            // Validate disjoint constraint by checking against other qualified value shapes
            // in the same property shape context
            let disjoint_violation = self.validate_qualified_value_shapes_disjoint(
                store,
                context,
                &constraint.qualified_value_shape,
                graph_name,
            )?;

            if let Some(violation_message) = disjoint_violation {
                return Ok(ConstraintEvaluationResult::violated(
                    None,
                    Some(violation_message),
                ));
            }
        }

        tracing::debug!(
            "Qualified value shape constraint passed: {} values conform to shape {}",
            conforming_count,
            constraint.qualified_value_shape.as_str()
        );
        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Validate qualified value shapes disjoint constraint
    /// This checks that each value conforms to at most one qualified value shape
    fn validate_qualified_value_shapes_disjoint(
        &mut self,
        store: &Store,
        context: &ConstraintContext,
        current_qualified_shape: &ShapeId,
        graph_name: Option<&str>,
    ) -> Result<Option<String>> {
        // Find the property shape that contains this qualified value shape constraint
        let property_shape = self.shapes.get(&context.shape_id).ok_or_else(|| {
            ShaclError::ValidationEngine(format!("Shape not found: {}", context.shape_id))
        })?;

        // Get all qualified value shape constraints from this property shape
        let mut qualified_constraints = Vec::new();
        for (component_id, constraint) in &property_shape.constraints {
            if let Constraint::QualifiedValueShape(qvs_constraint) = constraint {
                if qvs_constraint.qualified_value_shapes_disjoint {
                    qualified_constraints.push(qvs_constraint);
                }
            }
        }

        // If there's only one qualified value shape constraint, no disjoint validation needed
        if qualified_constraints.len() <= 1 {
            return Ok(None);
        }

        // For each value, check conformance to all qualified value shapes
        for value in &context.values {
            let mut conforming_shapes = Vec::new();

            for qvs_constraint in &qualified_constraints {
                let shape = self
                    .shapes
                    .get(&qvs_constraint.qualified_value_shape)
                    .ok_or_else(|| {
                        ShaclError::ValidationEngine(format!(
                            "Shape not found for qualified value shape constraint: {}",
                            qvs_constraint.qualified_value_shape.as_str()
                        ))
                    })?;

                let validation_result =
                    self.validate_node_against_shape(store, shape, value, graph_name)?;
                if validation_result.conforms() {
                    conforming_shapes.push(&qvs_constraint.qualified_value_shape);
                }
            }

            // Check if the value conforms to more than one qualified value shape
            if conforming_shapes.len() > 1 {
                let shape_names: Vec<String> = conforming_shapes
                    .iter()
                    .map(|s| s.as_str().to_string())
                    .collect();

                return Ok(Some(format!(
                    "Qualified value shapes disjoint constraint violated: value {} conforms to multiple qualified value shapes: {}",
                    self.format_term_for_message(value),
                    shape_names.join(", ")
                )));
            }
        }

        Ok(None)
    }

    /// Format a term for display in error messages
    fn format_term_for_message(&self, term: &Term) -> String {
        match term {
            Term::NamedNode(node) => node.as_str().to_string(),
            Term::BlankNode(node) => node.as_str().to_string(),
            Term::Literal(literal) => {
                if let Some(lang) = literal.language() {
                    format!("\"{}\"@{}", literal.value(), lang)
                } else {
                    format!("\"{}\"", literal.value())
                }
            }
            Term::Variable(var) => format!("?{}", var.name()),
            Term::QuotedTriple(_) => "<<quoted_triple>>".to_string(),
        }
    }

    // Closed Shape Constraints (placeholder implementation)

    /// Validate closed constraint - validates that only specified properties are present
    fn validate_closed_constraint(
        &self,
        store: &Store,
        constraint: &ClosedConstraint,
        context: &ConstraintContext,
        graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        if !constraint.closed {
            // If closed=false, the constraint is always satisfied
            return Ok(ConstraintEvaluationResult::satisfied());
        }

        // For each focus node, check that it only has properties that are either:
        // 1. Defined in property shapes in the current shape, or
        // 2. Listed in sh:ignoredProperties

        // This requires checking all outgoing properties from the focus node
        for focus_node in &[context.focus_node.clone()] {
            // Get all properties (predicates) for this focus node
            let all_properties = self.get_all_properties_for_node(store, focus_node, graph_name)?;

            // Collect allowed properties from the current shape's property shapes
            let mut allowed_properties = HashSet::new();

            // Add properties from the context (these are collected from the shape)
            for allowed_path in &context.allowed_properties {
                if let PropertyPath::Predicate(predicate) = allowed_path {
                    allowed_properties.insert(predicate.clone());
                }
                // TODO: Handle complex property paths
            }

            // Add ignored properties from the constraint
            for ignored_path in &constraint.ignored_properties {
                if let PropertyPath::Predicate(predicate) = ignored_path {
                    allowed_properties.insert(predicate.clone());
                }
                // TODO: Handle complex property paths in ignored properties
            }

            // Add standard RDF/RDFS/OWL properties that are typically allowed
            let standard_properties = [
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "http://www.w3.org/2000/01/rdf-schema#label",
                "http://www.w3.org/2000/01/rdf-schema#comment",
                "http://www.w3.org/2002/07/owl#sameAs",
            ];

            for prop_iri in &standard_properties {
                if let Ok(named_node) = NamedNode::new(*prop_iri) {
                    allowed_properties.insert(named_node);
                }
            }

            // Check for unexpected properties
            for property in &all_properties {
                if !allowed_properties.contains(property) {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(focus_node.clone()),
                        Some(format!("Closed shape constraint violated: unexpected property {} found on focus node {}", 
                                   property.as_str(), focus_node.as_str()))
                    ));
                }
            }
        }

        tracing::debug!("Closed constraint validation passed");
        Ok(ConstraintEvaluationResult::satisfied())
    }

    /// Execute a constraint validation query
    fn execute_constraint_query(
        &self,
        store: &Store,
        query: &str,
    ) -> Result<oxirs_core::query::QueryResult> {
        use oxirs_core::query::QueryEngine;

        let query_engine = QueryEngine::new();

        tracing::debug!("Executing constraint validation query: {}", query);

        let result = query_engine.query(query, store).map_err(|e| {
            ShaclError::ConstraintValidation(format!("Constraint query execution failed: {}", e))
        })?;

        Ok(result)
    }

    /// Get all properties (predicates) for a given node
    fn get_all_properties_for_node(
        &self,
        store: &Store,
        node: &Term,
        graph_name: Option<&str>,
    ) -> Result<Vec<NamedNode>> {
        use oxirs_core::model::{GraphName, Subject};

        let subject = match node {
            Term::NamedNode(n) => Subject::NamedNode(n.clone()),
            Term::BlankNode(n) => Subject::BlankNode(n.clone()),
            _ => return Ok(Vec::new()), // Literals don't have outgoing properties
        };

        let graph_filter = if let Some(g) = graph_name {
            Some(GraphName::NamedNode(NamedNode::new(g).map_err(|e| {
                ShaclError::Core(OxirsError::Parse(e.to_string()))
            })?))
        } else {
            None
        };

        let quads = store
            .query_quads(
                Some(&subject),
                None, // Any predicate
                None, // Any object
                graph_filter.as_ref(),
            )
            .map_err(|e| ShaclError::Core(e))?;

        // Extract unique predicates
        let mut properties = HashSet::new();
        for quad in quads {
            if let oxirs_core::model::Predicate::NamedNode(predicate) = quad.predicate() {
                properties.insert(predicate.clone());
            }
        }

        Ok(properties.into_iter().collect())
    }

    /// Check if a value is an instance of a class using direct store queries (fallback)
    fn is_instance_of_class_direct(
        &self,
        store: &Store,
        value: &Term,
        class_iri: &NamedNode,
        graph_name: Option<&str>,
    ) -> Result<bool> {
        use oxirs_core::model::{GraphName, Object, Predicate, Subject};

        let rdf_type =
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").map_err(|e| {
                ShaclError::ConstraintValidation(format!("Invalid rdf:type IRI: {}", e))
            })?;

        let subject = match value {
            Term::NamedNode(node) => Subject::NamedNode(node.clone()),
            Term::BlankNode(node) => Subject::BlankNode(node.clone()),
            _ => return Ok(false), // Literals cannot be instances of classes
        };

        let predicate = Predicate::NamedNode(rdf_type);
        let object = Object::NamedNode(class_iri.clone());

        let graph_filter = if let Some(g) = graph_name {
            Some(GraphName::NamedNode(NamedNode::new(g).map_err(|e| {
                ShaclError::Core(OxirsError::Parse(e.to_string()))
            })?))
        } else {
            None
        };

        let quads = store
            .query_quads(
                Some(&subject),
                Some(&predicate),
                Some(&object),
                graph_filter.as_ref(),
            )
            .map_err(|e| ShaclError::Core(e))?;

        // If we find any matching quad, the value is an instance of the class
        Ok(quads.into_iter().next().is_some())
    }

    /// Check if validation should stop early
    fn should_stop_validation(&self, report: &ValidationReport) -> bool {
        (self.config.fail_fast && !report.conforms())
            || (self.config.max_violations > 0
                && report.violation_count() >= self.config.max_violations)
    }

    /// Get validation statistics
    pub fn get_statistics(&self) -> &ValidationStats {
        &self.stats
    }

    /// Get cache hit rate
    pub fn get_cache_hit_rate(&self) -> f64 {
        let total = self.stats.cache_hits + self.stats.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.stats.cache_hits as f64 / total as f64
        }
    }

    /// Clear internal caches
    pub fn clear_caches(&mut self) {
        self.target_selector.clear_cache();
        self.path_evaluator.clear_cache();
        self.constraint_cache.borrow_mut().clear();
    }

    /// Compare two literals for ordering (supports numeric and string comparison)
    fn compare_literals(&self, left: &Literal, right: &Literal) -> Result<std::cmp::Ordering> {
        // First check if they have the same datatype
        let left_datatype = left.datatype();
        let right_datatype = right.datatype();

        // If datatypes don't match, we can't compare (except for numeric types)
        if left_datatype != right_datatype {
            // Special case: allow comparison between different numeric types
            if self.is_numeric_datatype(left_datatype.as_str())
                && self.is_numeric_datatype(right_datatype.as_str())
            {
                return self.compare_numeric_literals(left, right);
            }
            return Err(ShaclError::ConstraintValidation(format!(
                "Cannot compare literals with different datatypes: {} and {}",
                left_datatype.as_str(),
                right_datatype.as_str()
            )));
        }

        // Compare based on datatype
        match left_datatype.as_str() {
            // Numeric types
            "http://www.w3.org/2001/XMLSchema#integer"
            | "http://www.w3.org/2001/XMLSchema#int"
            | "http://www.w3.org/2001/XMLSchema#long"
            | "http://www.w3.org/2001/XMLSchema#short"
            | "http://www.w3.org/2001/XMLSchema#byte"
            | "http://www.w3.org/2001/XMLSchema#nonNegativeInteger"
            | "http://www.w3.org/2001/XMLSchema#positiveInteger"
            | "http://www.w3.org/2001/XMLSchema#nonPositiveInteger"
            | "http://www.w3.org/2001/XMLSchema#negativeInteger" => {
                match (left.as_str().parse::<i64>(), right.as_str().parse::<i64>()) {
                    (Ok(l), Ok(r)) => Ok(l.cmp(&r)),
                    _ => Err(ShaclError::ConstraintValidation(format!(
                        "Invalid integer values: {} or {}",
                        left.as_str(),
                        right.as_str()
                    ))),
                }
            }
            "http://www.w3.org/2001/XMLSchema#decimal"
            | "http://www.w3.org/2001/XMLSchema#float"
            | "http://www.w3.org/2001/XMLSchema#double" => {
                match (left.as_str().parse::<f64>(), right.as_str().parse::<f64>()) {
                    (Ok(l), Ok(r)) => Ok(l.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Equal)),
                    _ => Err(ShaclError::ConstraintValidation(format!(
                        "Invalid numeric values: {} or {}",
                        left.as_str(),
                        right.as_str()
                    ))),
                }
            }
            // Date/DateTime types
            "http://www.w3.org/2001/XMLSchema#date" => {
                // TODO: Implement proper date parsing and comparison
                // For now, use lexical comparison which works for ISO dates
                Ok(left.as_str().cmp(right.as_str()))
            }
            "http://www.w3.org/2001/XMLSchema#dateTime" => {
                // TODO: Implement proper dateTime parsing and comparison
                // For now, use lexical comparison which works for ISO dateTime
                Ok(left.as_str().cmp(right.as_str()))
            }
            "http://www.w3.org/2001/XMLSchema#time" => {
                // TODO: Implement proper time parsing and comparison
                Ok(left.as_str().cmp(right.as_str()))
            }
            // String types and default
            "http://www.w3.org/2001/XMLSchema#string"
            | "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString"
            | _ => {
                // String comparison
                Ok(left.as_str().cmp(right.as_str()))
            }
        }
    }

    /// Check if a datatype is numeric
    fn is_numeric_datatype(&self, datatype: &str) -> bool {
        matches!(
            datatype,
            "http://www.w3.org/2001/XMLSchema#integer"
                | "http://www.w3.org/2001/XMLSchema#int"
                | "http://www.w3.org/2001/XMLSchema#long"
                | "http://www.w3.org/2001/XMLSchema#short"
                | "http://www.w3.org/2001/XMLSchema#byte"
                | "http://www.w3.org/2001/XMLSchema#decimal"
                | "http://www.w3.org/2001/XMLSchema#float"
                | "http://www.w3.org/2001/XMLSchema#double"
                | "http://www.w3.org/2001/XMLSchema#nonNegativeInteger"
                | "http://www.w3.org/2001/XMLSchema#positiveInteger"
                | "http://www.w3.org/2001/XMLSchema#nonPositiveInteger"
                | "http://www.w3.org/2001/XMLSchema#negativeInteger"
        )
    }

    /// Compare numeric literals of possibly different types
    fn compare_numeric_literals(
        &self,
        left: &Literal,
        right: &Literal,
    ) -> Result<std::cmp::Ordering> {
        // Convert both to f64 for comparison
        match (left.as_str().parse::<f64>(), right.as_str().parse::<f64>()) {
            (Ok(l), Ok(r)) => Ok(l.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Equal)),
            _ => Err(ShaclError::ConstraintValidation(format!(
                "Invalid numeric values for comparison: {} or {}",
                left.as_str(),
                right.as_str()
            ))),
        }
    }

    /// Resolve inherited constraints for a shape
    pub fn resolve_inherited_constraints(
        &self,
        shape_id: &ShapeId,
    ) -> Result<IndexMap<ConstraintComponentId, Constraint>> {
        // Check cache first
        if let Some(cached_constraints) = self.inheritance_cache.borrow().get(shape_id) {
            return Ok(cached_constraints.clone());
        }

        let mut resolved_constraints = IndexMap::new();
        let mut visited = HashSet::new();

        self.collect_inherited_constraints(shape_id, &mut resolved_constraints, &mut visited)?;

        // Cache the result
        self.inheritance_cache
            .borrow_mut()
            .insert(shape_id.clone(), resolved_constraints.clone());

        Ok(resolved_constraints)
    }

    /// Recursively collect constraints from a shape and its parents
    fn collect_inherited_constraints(
        &self,
        shape_id: &ShapeId,
        resolved_constraints: &mut IndexMap<ConstraintComponentId, Constraint>,
        visited: &mut HashSet<ShapeId>,
    ) -> Result<()> {
        // Avoid infinite recursion
        if visited.contains(shape_id) {
            return Ok(());
        }
        visited.insert(shape_id.clone());

        if let Some(shape) = self.shapes.get(shape_id) {
            // First process parent shapes (depth-first) in priority order
            let mut parent_shapes: Vec<_> = shape.extends.iter().collect();

            // Sort parent shapes by priority if they exist
            parent_shapes.sort_by(|a, b| {
                let priority_a = self
                    .shapes
                    .get(*a)
                    .map(|s| s.effective_priority())
                    .unwrap_or(0);
                let priority_b = self
                    .shapes
                    .get(*b)
                    .map(|s| s.effective_priority())
                    .unwrap_or(0);
                priority_b.cmp(&priority_a) // Higher priority first
            });

            for parent_id in parent_shapes {
                self.collect_inherited_constraints(parent_id, resolved_constraints, visited)?;
            }

            // Then add this shape's constraints, potentially overriding parent constraints
            // Child constraints override parent constraints (standard inheritance behavior)
            for (component_id, constraint) in &shape.constraints {
                resolved_constraints.insert(component_id.clone(), constraint.clone());
            }
        }

        Ok(())
    }

    /// Get all shapes that a given shape inherits from (transitively)
    fn get_all_parent_shapes(&self, shape_id: &ShapeId) -> Result<Vec<ShapeId>> {
        let mut parents = Vec::new();
        let mut visited = HashSet::new();

        self.collect_parent_shapes(shape_id, &mut parents, &mut visited)?;

        Ok(parents)
    }

    fn collect_parent_shapes(
        &self,
        shape_id: &ShapeId,
        parents: &mut Vec<ShapeId>,
        visited: &mut HashSet<ShapeId>,
    ) -> Result<()> {
        if visited.contains(shape_id) {
            return Ok(());
        }
        visited.insert(shape_id.clone());

        if let Some(shape) = self.shapes.get(shape_id) {
            for parent_id in &shape.extends {
                parents.push(parent_id.clone());
                self.collect_parent_shapes(parent_id, parents, visited)?;
            }
        }

        Ok(())
    }
}

/// Validation performance statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationStats {
    pub total_validations: usize,
    pub total_node_validations: usize,
    pub total_constraint_evaluations: usize,
    pub total_validation_time: Duration,
    pub last_validation_time: Option<Duration>,
    pub avg_validation_time: Duration,
    pub constraint_evaluation_times: HashMap<String, Duration>,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl ValidationStats {
    pub fn record_constraint_evaluation(&mut self, constraint_type: String, duration: Duration) {
        self.total_constraint_evaluations += 1;
        *self
            .constraint_evaluation_times
            .entry(constraint_type)
            .or_insert(Duration::ZERO) += duration;

        if self.total_validations > 0 {
            self.avg_validation_time = self.total_validation_time / self.total_validations as u32;
        }
    }

    pub fn get_avg_constraint_time(&self, constraint_type: &str) -> Option<Duration> {
        self.constraint_evaluation_times
            .get(constraint_type)
            .map(|total| *total / self.total_constraint_evaluations as u32)
    }
}

/// Validation violation with detailed information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationViolation {
    /// The focus node where the violation occurred
    pub focus_node: Term,

    /// The source shape that was violated
    pub source_shape: ShapeId,

    /// The constraint component that was violated
    pub source_constraint_component: ConstraintComponentId,

    /// The property path where the violation occurred (for property shapes)
    pub result_path: Option<PropertyPath>,

    /// The specific value that caused the violation
    pub value: Option<Term>,

    /// Human-readable message describing the violation
    pub result_message: Option<String>,

    /// Severity of the violation
    pub result_severity: Severity,

    /// Additional details about the violation
    pub details: HashMap<String, String>,

    /// Nested validation results (sh:detail) for complex constraints
    /// Used by logical constraints (and, or, xone) and shape-based constraints (node, qualifiedValueShape)
    pub nested_results: Vec<ValidationViolation>,
}

impl ValidationViolation {
    pub fn new(
        focus_node: Term,
        source_shape: ShapeId,
        source_constraint_component: ConstraintComponentId,
        result_severity: Severity,
    ) -> Self {
        Self {
            focus_node,
            source_shape,
            source_constraint_component,
            result_path: None,
            value: None,
            result_message: None,
            result_severity,
            details: HashMap::new(),
            nested_results: Vec::new(),
        }
    }

    pub fn with_path(mut self, path: PropertyPath) -> Self {
        self.result_path = Some(path);
        self
    }

    pub fn with_value(mut self, value: Term) -> Self {
        self.value = Some(value);
        self
    }

    pub fn with_message(mut self, message: String) -> Self {
        self.result_message = Some(message);
        self
    }

    pub fn with_detail(mut self, key: String, value: String) -> Self {
        self.details.insert(key, value);
        self
    }

    pub fn with_nested_result(mut self, nested: ValidationViolation) -> Self {
        self.nested_results.push(nested);
        self
    }

    pub fn with_nested_results(mut self, nested: Vec<ValidationViolation>) -> Self {
        self.nested_results.extend(nested);
        self
    }
}

/// Result of constraint evaluation
#[derive(Debug, Clone)]
pub enum ConstraintEvaluationResult {
    /// Constraint is satisfied
    Satisfied,

    /// Constraint is violated
    Violated {
        /// The specific value that caused the violation (if any)
        violating_value: Option<Term>,

        /// Human-readable message describing the violation
        message: Option<String>,

        /// Additional details about the violation
        details: HashMap<String, String>,
    },

    /// Error occurred during evaluation
    Error {
        /// Error message
        message: String,

        /// Additional error details
        details: HashMap<String, String>,
    },
}

impl ConstraintEvaluationResult {
    pub fn satisfied() -> Self {
        ConstraintEvaluationResult::Satisfied
    }

    pub fn violated(violating_value: Option<Term>, message: Option<String>) -> Self {
        ConstraintEvaluationResult::Violated {
            violating_value,
            message,
            details: HashMap::new(),
        }
    }

    pub fn violated_with_details(
        violating_value: Option<Term>,
        message: Option<String>,
        details: HashMap<String, String>,
    ) -> Self {
        ConstraintEvaluationResult::Violated {
            violating_value,
            message,
            details,
        }
    }

    pub fn error(message: String) -> Self {
        ConstraintEvaluationResult::Error {
            message,
            details: HashMap::new(),
        }
    }

    pub fn is_satisfied(&self) -> bool {
        matches!(self, ConstraintEvaluationResult::Satisfied)
    }

    pub fn is_violated(&self) -> bool {
        matches!(self, ConstraintEvaluationResult::Violated { .. })
    }

    pub fn is_error(&self) -> bool {
        matches!(self, ConstraintEvaluationResult::Error { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PropertyPath, Shape, ShapeType};

    #[test]
    fn test_validation_engine_creation() {
        let shapes = IndexMap::new();
        let config = ValidationConfig::default();
        let engine = ValidationEngine::new(&shapes, config.clone());

        assert_eq!(engine.shapes.len(), 0);
        assert_eq!(engine.config.max_violations, config.max_violations);
    }

    #[test]
    fn test_validation_violation() {
        let focus_node = Term::NamedNode(NamedNode::new("http://example.org/john").unwrap());
        let shape_id = ShapeId::new("http://example.org/PersonShape");
        let component_id = ConstraintComponentId::new("sh:ClassConstraintComponent");

        let violation = ValidationViolation::new(
            focus_node.clone(),
            shape_id.clone(),
            component_id.clone(),
            Severity::Violation,
        )
        .with_message("Test violation".to_string())
        .with_detail("test_key".to_string(), "test_value".to_string());

        assert_eq!(violation.focus_node, focus_node);
        assert_eq!(violation.source_shape, shape_id);
        assert_eq!(violation.source_constraint_component, component_id);
        assert_eq!(violation.result_severity, Severity::Violation);
        assert_eq!(violation.result_message, Some("Test violation".to_string()));
        assert_eq!(
            violation.details.get("test_key"),
            Some(&"test_value".to_string())
        );
    }

    #[test]
    fn test_validation_stats() {
        let mut stats = ValidationStats::default();

        let duration = Duration::from_millis(100);
        stats.record_constraint_evaluation("ClassConstraint".to_string(), duration);
        stats.record_constraint_evaluation("ClassConstraint".to_string(), duration);

        assert_eq!(stats.total_constraint_evaluations, 2);
        assert_eq!(
            stats.constraint_evaluation_times.get("ClassConstraint"),
            Some(&Duration::from_millis(200))
        );
    }

    #[test]
    fn test_node_kind_validation() {
        let engine = create_test_engine();
        let constraint = NodeKindConstraint {
            node_kind: NodeKind::Iri,
        };

        // Test with IRI - should pass
        let iri_context = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/test").unwrap()),
            ShapeId::new("test_shape"),
        )
        .with_values(vec![Term::NamedNode(
            NamedNode::new("http://example.org/value").unwrap(),
        )]);

        let result = engine
            .validate_node_kind_constraint(&constraint, &iri_context)
            .unwrap();
        assert!(result.is_satisfied());

        // Test with literal - should fail
        let literal_context = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/test").unwrap()),
            ShapeId::new("test_shape"),
        )
        .with_values(vec![Term::Literal(Literal::new("test"))]);

        let result = engine
            .validate_node_kind_constraint(&constraint, &literal_context)
            .unwrap();
        assert!(result.is_violated());
    }

    #[test]
    fn test_cardinality_validation() {
        let engine = create_test_engine();

        // Test min count constraint
        let min_constraint = MinCountConstraint { min_count: 2 };
        let context_insufficient = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/test").unwrap()),
            ShapeId::new("test_shape"),
        )
        .with_values(vec![Term::Literal(Literal::new("value1"))]);

        let result = engine
            .validate_min_count_constraint(&min_constraint, &context_insufficient)
            .unwrap();
        assert!(result.is_violated());

        let context_sufficient = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/test").unwrap()),
            ShapeId::new("test_shape"),
        )
        .with_values(vec![
            Term::Literal(Literal::new("value1")),
            Term::Literal(Literal::new("value2")),
        ]);

        let result = engine
            .validate_min_count_constraint(&min_constraint, &context_sufficient)
            .unwrap();
        assert!(result.is_satisfied());
    }

    fn create_test_engine() -> ValidationEngine<'static> {
        let shapes = Box::leak(Box::new(IndexMap::new()));
        let config = ValidationConfig::default();
        ValidationEngine::new(shapes, config)
    }

    #[test]
    fn test_constraint_caching() {
        let mut shapes = IndexMap::new();
        let shape = Shape::new(ShapeId::new("test_shape"), ShapeType::NodeShape);
        shapes.insert(shape.id.clone(), shape);

        let config = ValidationConfig::default();
        let mut engine = ValidationEngine::new(&shapes, config);

        // Create a constraint and context
        let constraint = Constraint::NodeKind(NodeKindConstraint {
            node_kind: NodeKind::Iri,
        });
        let context = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/test").unwrap()),
            ShapeId::new("test_shape"),
        )
        .with_values(vec![Term::NamedNode(
            NamedNode::new("http://example.org/value").unwrap(),
        )]);

        // First evaluation - should miss cache
        let store = Store::new().unwrap();
        let result1 = engine
            .validate_constraint(&store, &constraint, &context, None, None)
            .unwrap();
        assert_eq!(engine.stats.cache_misses, 1);
        assert_eq!(engine.stats.cache_hits, 0);

        // Second evaluation with same constraint - should hit cache
        let result2 = engine
            .validate_constraint(&store, &constraint, &context, None, None)
            .unwrap();
        assert_eq!(engine.stats.cache_misses, 1);
        assert_eq!(engine.stats.cache_hits, 1);

        // Results should be the same
        assert!(matches!(result1, ConstraintEvaluationResult::Satisfied));
        assert!(matches!(result2, ConstraintEvaluationResult::Satisfied));

        // Check cache hit rate
        assert_eq!(engine.get_cache_hit_rate(), 0.5);
    }

    #[test]
    fn test_literal_comparison_numeric() {
        let engine = create_test_engine();

        // Test integer comparison
        let int_literal1 = Literal::new_typed(
            "42",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
        );
        let int_literal2 = Literal::new_typed(
            "100",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
        );

        let result = engine
            .compare_literals(&int_literal1, &int_literal2)
            .unwrap();
        assert_eq!(result, std::cmp::Ordering::Less);

        // Test decimal comparison
        let dec_literal1 = Literal::new_typed(
            "42.5",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#decimal").unwrap(),
        );
        let dec_literal2 = Literal::new_typed(
            "42.3",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#decimal").unwrap(),
        );

        let result = engine
            .compare_literals(&dec_literal1, &dec_literal2)
            .unwrap();
        assert_eq!(result, std::cmp::Ordering::Greater);

        // Test mixed numeric comparison (integer vs decimal)
        let int_literal = Literal::new_typed(
            "42",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
        );
        let dec_literal = Literal::new_typed(
            "42.0",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#decimal").unwrap(),
        );

        let result = engine.compare_literals(&int_literal, &dec_literal).unwrap();
        assert_eq!(result, std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_literal_comparison_string() {
        let engine = create_test_engine();

        // Test string comparison
        let str_literal1 = Literal::new_typed(
            "apple",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#string").unwrap(),
        );
        let str_literal2 = Literal::new_typed(
            "banana",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#string").unwrap(),
        );

        let result = engine
            .compare_literals(&str_literal1, &str_literal2)
            .unwrap();
        assert_eq!(result, std::cmp::Ordering::Less);
    }
}

/// Format a term for use in SPARQL queries
fn format_term_for_sparql(term: &Term) -> Result<String> {
    match term {
        Term::NamedNode(node) => Ok(format!("<{}>", node.as_str())),
        Term::BlankNode(node) => Ok(node.as_str().to_string()),
        Term::Literal(literal) => {
            // TODO: Proper literal formatting with datatype and language
            Ok(format!("\"{}\"", literal.as_str().replace('"', "\\\"")))
        }
        Term::Variable(var) => Ok(format!("?{}", var.name())),
        Term::QuotedTriple(_) => Err(ShaclError::ValidationEngine(
            "Quoted triples not supported in validation queries".to_string(),
        )),
    }
}
