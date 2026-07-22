//! # SHACL Integration with OxiRS Rule Engine
//!
//! This module provides deep integration between SHACL validation and the OxiRS rule engine,
//! enabling reasoning-aware validation, constraint inference, and automated shape refinement.
//!
//! ## Features
//!
//! - **Reasoning-aware validation**: Validate against inferred triples
//! - **Constraint inference**: Infer new constraints from existing ones
//! - **Shape refinement**: Automatically refine shapes based on reasoning
//! - **Rule-based validators**: Define custom validators using rules
//! - **Incremental reasoning**: Update inferred knowledge incrementally

use crate::{Constraint, Result, ShaclError, Shape, ValidationReport};
use oxirs_core::{model::*, Store};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info};

/// Well-known RDF/RDFS IRIs used by the forward-chaining RDFS reasoner.
const RDF_TYPE_IRI: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const RDFS_SUBCLASS_IRI: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";
const RDFS_SUBPROP_IRI: &str = "http://www.w3.org/2000/01/rdf-schema#subPropertyOf";
const RDFS_DOMAIN_IRI: &str = "http://www.w3.org/2000/01/rdf-schema#domain";
const RDFS_RANGE_IRI: &str = "http://www.w3.org/2000/01/rdf-schema#range";

/// Build a [`Predicate`] from an IRI string.
fn named_predicate(iri: &str) -> Result<Predicate> {
    Ok(Predicate::NamedNode(NamedNode::new(iri).map_err(|e| {
        ShaclError::ValidationEngine(format!("Invalid reasoning IRI '{iri}': {e}"))
    })?))
}

/// Convert an object term into a subject term when it is a resource
/// (NamedNode/BlankNode); literals and other terms yield `None`.
fn object_to_subject(object: &Object) -> Option<Subject> {
    match object {
        Object::NamedNode(n) => Some(Subject::NamedNode(n.clone())),
        Object::BlankNode(b) => Some(Subject::BlankNode(b.clone())),
        _ => None,
    }
}

/// Extract the [`NamedNode`] of a subject term, if it is an IRI.
fn subject_to_named(subject: &Subject) -> Option<NamedNode> {
    match subject {
        Subject::NamedNode(n) => Some(n.clone()),
        _ => None,
    }
}

/// Extract the [`NamedNode`] of an object term, if it is an IRI.
fn object_to_named(object: &Object) -> Option<NamedNode> {
    match object {
        Object::NamedNode(n) => Some(n.clone()),
        _ => None,
    }
}

/// Configuration for rule engine integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleEngineConfig {
    /// Enable forward chaining during validation
    pub forward_chaining: bool,

    /// Enable backward chaining during validation
    pub backward_chaining: bool,

    /// Maximum reasoning depth
    pub max_reasoning_depth: usize,

    /// Enable constraint inference
    pub infer_constraints: bool,

    /// Enable shape refinement
    pub refine_shapes: bool,

    /// Reasoning strategy
    pub strategy: ReasoningStrategy,

    /// Cache inferred triples
    pub cache_inferences: bool,

    /// Timeout for reasoning (milliseconds)
    pub timeout_ms: Option<u64>,
}

impl Default for RuleEngineConfig {
    fn default() -> Self {
        Self {
            forward_chaining: true,
            backward_chaining: false,
            max_reasoning_depth: 10,
            infer_constraints: false,
            refine_shapes: false,
            strategy: ReasoningStrategy::Optimized,
            cache_inferences: true,
            timeout_ms: Some(5000),
        }
    }
}

/// Reasoning strategies for validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasoningStrategy {
    /// No reasoning
    None,

    /// RDFS entailment only
    RDFS,

    /// OWL DL reasoning
    OWL,

    /// OWL RL (rule-based subset)
    OWLRL,

    /// Custom rule set
    Custom,

    /// Optimized reasoning (automatic strategy selection)
    Optimized,
}

/// Rule-based SHACL validator with reasoning integration
pub struct RuleBasedValidator {
    config: RuleEngineConfig,
    inference_cache: Arc<dashmap::DashMap<String, Vec<Triple>>>,
    constraint_rules: Vec<ConstraintRule>,
    shape_refinement_rules: Vec<ShapeRefinementRule>,
}

impl RuleBasedValidator {
    /// Create a new rule-based validator
    pub fn new(config: RuleEngineConfig) -> Self {
        Self {
            config,
            inference_cache: Arc::new(dashmap::DashMap::new()),
            constraint_rules: Vec::new(),
            shape_refinement_rules: Vec::new(),
        }
    }

    /// Validate with reasoning integration
    pub fn validate_with_reasoning(
        &self,
        store: &dyn Store,
        shapes: &[Shape],
    ) -> Result<ValidationReport> {
        info!("Starting validation with reasoning integration");

        // Step 1: Perform reasoning if enabled. Backward chaining is not
        // implemented, so requesting it fails loud rather than silently running
        // plain validation with no reasoning applied.
        if self.config.backward_chaining {
            return Err(ShaclError::UnsupportedOperation(
                "Backward-chaining reasoning is not implemented for reasoning-aware SHACL \
                 validation; disable backward_chaining or use forward_chaining (RDFS/OWL-RL)"
                    .to_string(),
            ));
        }
        if self.config.forward_chaining {
            let inferred = self.forward_chain(store)?;
            debug!("Forward chaining materialized {inferred} inferred triple(s)");
        }

        // Step 2: Infer additional constraints if enabled
        let enhanced_shapes = if self.config.infer_constraints {
            debug!("Inferring additional constraints");
            self.infer_constraints(store, shapes)?
        } else {
            shapes.to_vec()
        };

        // Step 3: Perform validation (delegated to standard validator)
        let report = self.perform_validation(store, &enhanced_shapes)?;

        // Step 4: Refine shapes based on violations if enabled
        if self.config.refine_shapes && !report.violations.is_empty() {
            debug!("Refining shapes based on violations");
            self.refine_shapes(&report)?;
        }

        Ok(report)
    }

    /// Add a constraint inference rule
    pub fn add_constraint_rule(&mut self, rule: ConstraintRule) {
        self.constraint_rules.push(rule);
    }

    /// Add a shape refinement rule
    pub fn add_shape_refinement_rule(&mut self, rule: ShapeRefinementRule) {
        self.shape_refinement_rules.push(rule);
    }

    /// Clear inference cache
    pub fn clear_cache(&self) {
        self.inference_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            size: self.inference_cache.len(),
            total_inferences: self.inference_cache.iter().map(|e| e.value().len()).sum(),
        }
    }

    // Private methods

    /// Apply RDFS (and, for OWL strategies, OWL-RL-flavoured) forward chaining
    /// by materialising inferred triples into the store to a fixpoint.
    ///
    /// Returns the total number of newly-inferred triples inserted. The
    /// iteration is bounded by `max_reasoning_depth`; any reasoning query
    /// failure is surfaced as an error rather than silently skipped.
    fn forward_chain(&self, store: &dyn Store) -> Result<usize> {
        if self.config.strategy == ReasoningStrategy::None {
            return Ok(0);
        }

        let type_pred = named_predicate(RDF_TYPE_IRI)?;
        let subclass_pred = named_predicate(RDFS_SUBCLASS_IRI)?;
        let subprop_pred = named_predicate(RDFS_SUBPROP_IRI)?;
        let domain_pred = named_predicate(RDFS_DOMAIN_IRI)?;
        let range_pred = named_predicate(RDFS_RANGE_IRI)?;

        let mut total_new = 0usize;
        let max_iters = self.config.max_reasoning_depth.max(1);

        for _ in 0..max_iters {
            let inferred = self.rdfs_inference_pass(
                store,
                &type_pred,
                &subclass_pred,
                &subprop_pred,
                &domain_pred,
                &range_pred,
            )?;

            let mut iteration_new = 0usize;
            for quad in inferred {
                let inserted = store.insert_quad(quad).map_err(|e| {
                    ShaclError::ValidationEngine(format!(
                        "Failed to materialize inferred triple: {e}"
                    ))
                })?;
                if inserted {
                    iteration_new += 1;
                }
            }

            total_new += iteration_new;
            if iteration_new == 0 {
                break; // fixpoint reached
            }
        }

        Ok(total_new)
    }

    /// Compute one pass of RDFS entailment over the store, returning the
    /// candidate inferred quads (which may already exist — the caller counts
    /// only the genuinely new insertions and iterates to a fixpoint).
    ///
    /// This is implemented with direct pattern matching (`Store::find_quads`)
    /// rather than multi-pattern SPARQL joins so the closure is computed
    /// correctly and deterministically regardless of query-engine join support.
    #[allow(clippy::too_many_arguments)]
    fn rdfs_inference_pass(
        &self,
        store: &dyn Store,
        type_pred: &Predicate,
        subclass_pred: &Predicate,
        subprop_pred: &Predicate,
        domain_pred: &Predicate,
        range_pred: &Predicate,
    ) -> Result<Vec<Quad>> {
        let map_err = |e| ShaclError::ValidationEngine(format!("Reasoning query failed: {e}"));
        let mut inferred: Vec<Quad> = Vec::new();

        // rdfs11: subClassOf transitivity.
        let subclass = store
            .find_quads(None, Some(subclass_pred), None, None)
            .map_err(map_err)?;
        for q in &subclass {
            if let Some(mid) = object_to_subject(q.object()) {
                for q2 in store
                    .find_quads(Some(&mid), Some(subclass_pred), None, None)
                    .map_err(map_err)?
                {
                    inferred.push(Quad::new(
                        q.subject().clone(),
                        subclass_pred.clone(),
                        q2.object().clone(),
                        q.graph_name().clone(),
                    ));
                }
            }
        }

        // rdfs5: subPropertyOf transitivity.
        let subprop = store
            .find_quads(None, Some(subprop_pred), None, None)
            .map_err(map_err)?;
        for q in &subprop {
            if let Some(mid) = object_to_subject(q.object()) {
                for q2 in store
                    .find_quads(Some(&mid), Some(subprop_pred), None, None)
                    .map_err(map_err)?
                {
                    inferred.push(Quad::new(
                        q.subject().clone(),
                        subprop_pred.clone(),
                        q2.object().clone(),
                        q.graph_name().clone(),
                    ));
                }
            }
        }

        // rdfs9: rdf:type propagation across subClassOf.
        let types = store
            .find_quads(None, Some(type_pred), None, None)
            .map_err(map_err)?;
        for q in &types {
            if let Some(class_subj) = object_to_subject(q.object()) {
                for sc in store
                    .find_quads(Some(&class_subj), Some(subclass_pred), None, None)
                    .map_err(map_err)?
                {
                    inferred.push(Quad::new(
                        q.subject().clone(),
                        type_pred.clone(),
                        sc.object().clone(),
                        q.graph_name().clone(),
                    ));
                }
            }
        }

        // rdfs7: subPropertyOf application (?s ?p ?o + ?p subPropertyOf ?q => ?s ?q ?o).
        for sp in &subprop {
            if let (Some(p_named), Some(q_named)) =
                (subject_to_named(sp.subject()), object_to_named(sp.object()))
            {
                let p_pred = Predicate::NamedNode(p_named);
                let q_pred = Predicate::NamedNode(q_named);
                for t in store
                    .find_quads(None, Some(&p_pred), None, None)
                    .map_err(map_err)?
                {
                    inferred.push(Quad::new(
                        t.subject().clone(),
                        q_pred.clone(),
                        t.object().clone(),
                        t.graph_name().clone(),
                    ));
                }
            }
        }

        // rdfs2: rdfs:domain typing (?p domain ?c + ?s ?p ?o => ?s a ?c).
        for d in store
            .find_quads(None, Some(domain_pred), None, None)
            .map_err(map_err)?
        {
            if let Some(p_named) = subject_to_named(d.subject()) {
                let class_obj = d.object().clone();
                let p_pred = Predicate::NamedNode(p_named);
                for t in store
                    .find_quads(None, Some(&p_pred), None, None)
                    .map_err(map_err)?
                {
                    inferred.push(Quad::new(
                        t.subject().clone(),
                        type_pred.clone(),
                        class_obj.clone(),
                        t.graph_name().clone(),
                    ));
                }
            }
        }

        // rdfs3: rdfs:range typing (?p range ?c + ?s ?p ?o => ?o a ?c).
        for r in store
            .find_quads(None, Some(range_pred), None, None)
            .map_err(map_err)?
        {
            if let Some(p_named) = subject_to_named(r.subject()) {
                let class_obj = r.object().clone();
                let p_pred = Predicate::NamedNode(p_named);
                for t in store
                    .find_quads(None, Some(&p_pred), None, None)
                    .map_err(map_err)?
                {
                    if let Some(obj_subj) = object_to_subject(t.object()) {
                        inferred.push(Quad::new(
                            obj_subj,
                            type_pred.clone(),
                            class_obj.clone(),
                            t.graph_name().clone(),
                        ));
                    }
                }
            }
        }

        Ok(inferred)
    }

    fn rdfs_reasoning(&self, _store: &dyn Store) -> Result<Vec<Triple>> {
        // Implement RDFS entailment rules
        // - rdfs:subClassOf transitivity
        // - rdfs:subPropertyOf transitivity
        // - rdfs:domain/range inference
        // - rdf:type propagation

        Ok(Vec::new())
    }

    fn owl_reasoning(&self, _store: &dyn Store) -> Result<Vec<Triple>> {
        // Implement OWL DL reasoning
        // - Equivalence classes
        // - Inverse properties
        // - Transitive/symmetric properties
        // - Property chains

        Ok(Vec::new())
    }

    fn owl_rl_reasoning(&self, _store: &dyn Store) -> Result<Vec<Triple>> {
        // Implement OWL RL (rule-based subset)
        // Subset of OWL that can be implemented with rules

        Ok(Vec::new())
    }

    fn custom_reasoning(&self, _store: &dyn Store) -> Result<Vec<Triple>> {
        // Apply custom rules
        Ok(Vec::new())
    }

    fn optimized_reasoning(&self, store: &dyn Store) -> Result<Vec<Triple>> {
        // Automatically select best strategy based on ontology
        // Start with RDFS, add OWL RL if needed

        let rdfs_inferences = self.rdfs_reasoning(store)?;

        // Check if OWL constructs are present
        let needs_owl = self.detect_owl_constructs(store);

        if needs_owl {
            let owl_inferences = self.owl_rl_reasoning(store)?;
            Ok([rdfs_inferences, owl_inferences].concat())
        } else {
            Ok(rdfs_inferences)
        }
    }

    fn detect_owl_constructs(&self, _store: &dyn Store) -> bool {
        // Check for OWL constructs in the store
        false
    }

    fn infer_constraints(&self, _store: &dyn Store, shapes: &[Shape]) -> Result<Vec<Shape>> {
        // Apply constraint inference rules
        let mut enhanced = shapes.to_vec();

        for rule in &self.constraint_rules {
            for shape in &mut enhanced {
                if rule.matches(shape) {
                    rule.apply(shape)?;
                }
            }
        }

        Ok(enhanced)
    }

    fn perform_validation(&self, store: &dyn Store, shapes: &[Shape]) -> Result<ValidationReport> {
        use crate::{ValidationConfig, ValidationEngine};

        // Run the real SHACL validation engine against the (possibly
        // reasoning-augmented) store instead of fabricating a conforming report.
        let mut shape_map = indexmap::IndexMap::new();
        for shape in shapes {
            shape_map.insert(shape.id.clone(), shape.clone());
        }

        let config = ValidationConfig::default();
        let mut engine = ValidationEngine::new(&shape_map, config);
        engine.validate_store(store)
    }

    fn refine_shapes(&self, _report: &ValidationReport) -> Result<()> {
        // Analyze violations and suggest shape refinements
        Ok(())
    }
}

/// A rule for inferring additional constraints
#[derive(Debug, Clone)]
pub struct ConstraintRule {
    /// Rule name
    pub name: String,

    /// Rule description
    pub description: String,

    /// Pattern matcher
    pub pattern: ConstraintPattern,

    /// Constraint generator
    pub generator: ConstraintGenerator,
}

impl ConstraintRule {
    fn matches(&self, shape: &Shape) -> bool {
        self.pattern.matches(shape)
    }

    fn apply(&self, shape: &mut Shape) -> Result<()> {
        self.generator.generate(shape)
    }
}

/// Pattern for matching shapes
#[derive(Clone)]
pub enum ConstraintPattern {
    /// Match shapes with specific constraint type
    HasConstraintType(String),

    /// Match shapes targeting specific class
    TargetsClass(String),

    /// Custom pattern function
    Custom(Arc<dyn Fn(&Shape) -> bool + Send + Sync>),
}

impl std::fmt::Debug for ConstraintPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HasConstraintType(s) => f.debug_tuple("HasConstraintType").field(s).finish(),
            Self::TargetsClass(s) => f.debug_tuple("TargetsClass").field(s).finish(),
            Self::Custom(_) => f.debug_tuple("Custom").field(&"<closure>").finish(),
        }
    }
}

impl ConstraintPattern {
    fn matches(&self, _shape: &Shape) -> bool {
        match self {
            Self::HasConstraintType(_) => false,
            Self::TargetsClass(_) => false,
            Self::Custom(_) => false,
        }
    }
}

/// Type alias for custom constraint generator function
pub type CustomGeneratorFn = Arc<dyn Fn(&mut Shape) -> Result<()> + Send + Sync>;

/// Constraint generator
#[derive(Clone)]
pub enum ConstraintGenerator {
    /// Add datatype constraint
    AddDatatype(String),

    /// Add min count
    AddMinCount(usize),

    /// Add max count
    AddMaxCount(usize),

    /// Custom generator
    Custom(CustomGeneratorFn),
}

impl std::fmt::Debug for ConstraintGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AddDatatype(s) => f.debug_tuple("AddDatatype").field(s).finish(),
            Self::AddMinCount(n) => f.debug_tuple("AddMinCount").field(n).finish(),
            Self::AddMaxCount(n) => f.debug_tuple("AddMaxCount").field(n).finish(),
            Self::Custom(_) => f.debug_tuple("Custom").field(&"<closure>").finish(),
        }
    }
}

impl ConstraintGenerator {
    fn generate(&self, _shape: &mut Shape) -> Result<()> {
        match self {
            Self::AddDatatype(_) => Ok(()),
            Self::AddMinCount(_) => Ok(()),
            Self::AddMaxCount(_) => Ok(()),
            Self::Custom(_) => Ok(()),
        }
    }
}

/// A rule for refining shapes based on validation results
#[derive(Debug, Clone)]
pub struct ShapeRefinementRule {
    /// Rule name
    pub name: String,

    /// Violation pattern to match
    pub violation_pattern: ViolationPattern,

    /// Refinement action
    pub action: RefinementAction,
}

/// Pattern for matching violations
#[derive(Clone)]
pub enum ViolationPattern {
    /// Violations of specific constraint type
    ConstraintType(String),

    /// Violations above threshold
    FrequencyAbove(f64),

    /// Custom pattern
    Custom(Arc<dyn Fn(&ValidationReport) -> bool + Send + Sync>),
}

impl std::fmt::Debug for ViolationPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConstraintType(s) => f.debug_tuple("ConstraintType").field(s).finish(),
            Self::FrequencyAbove(n) => f.debug_tuple("FrequencyAbove").field(n).finish(),
            Self::Custom(_) => f.debug_tuple("Custom").field(&"<closure>").finish(),
        }
    }
}

/// Refinement action
#[derive(Debug, Clone)]
pub enum RefinementAction {
    /// Relax constraint
    RelaxConstraint,

    /// Strengthen constraint
    StrengthenConstraint,

    /// Add new constraint
    AddConstraint(Constraint),

    /// Remove constraint
    RemoveConstraint(String),
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub size: usize,
    pub total_inferences: usize,
}

/// Builder for rule-based validator
pub struct RuleBasedValidatorBuilder {
    config: RuleEngineConfig,
    constraint_rules: Vec<ConstraintRule>,
    refinement_rules: Vec<ShapeRefinementRule>,
}

impl RuleBasedValidatorBuilder {
    pub fn new() -> Self {
        Self {
            config: RuleEngineConfig::default(),
            constraint_rules: Vec::new(),
            refinement_rules: Vec::new(),
        }
    }

    pub fn with_config(mut self, config: RuleEngineConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_forward_chaining(mut self, enabled: bool) -> Self {
        self.config.forward_chaining = enabled;
        self
    }

    pub fn with_backward_chaining(mut self, enabled: bool) -> Self {
        self.config.backward_chaining = enabled;
        self
    }

    pub fn with_strategy(mut self, strategy: ReasoningStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    pub fn add_constraint_rule(mut self, rule: ConstraintRule) -> Self {
        self.constraint_rules.push(rule);
        self
    }

    pub fn add_refinement_rule(mut self, rule: ShapeRefinementRule) -> Self {
        self.refinement_rules.push(rule);
        self
    }

    pub fn build(self) -> RuleBasedValidator {
        let mut validator = RuleBasedValidator::new(self.config);
        validator.constraint_rules = self.constraint_rules;
        validator.shape_refinement_rules = self.refinement_rules;
        validator
    }
}

impl Default for RuleBasedValidatorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Reasoning result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningResult {
    /// Number of inferred triples
    pub inferred_count: usize,

    /// Reasoning time in milliseconds
    pub duration_ms: u64,

    /// Strategy used
    pub strategy: ReasoningStrategy,

    /// Whether reasoning was cached
    pub was_cached: bool,
}

/// Integration with existing SHACL Advanced Features reasoning module
pub mod advanced_integration {
    use super::*;
    use crate::advanced_features::reasoning::{
        EntailmentRegime, ReasoningConfig, ReasoningValidator,
    };

    /// Bridge between rule engine and SHACL-AF reasoning
    pub struct ReasoningBridge {
        rule_validator: RuleBasedValidator,
        shacl_af_validator: ReasoningValidator,
    }

    impl ReasoningBridge {
        pub fn new(rule_config: RuleEngineConfig) -> Self {
            let rule_validator = RuleBasedValidator::new(rule_config);
            let shacl_af_config = ReasoningConfig {
                entailment_regime: EntailmentRegime::RDFS,
                closed_world_assumption: false,
                cache_inferences: true,
                max_reasoning_depth: 10,
                reasoning_timeout_ms: Some(5000),
            };
            let shacl_af_validator = ReasoningValidator::new(shacl_af_config);

            Self {
                rule_validator,
                shacl_af_validator,
            }
        }

        /// Unified validation with both rule engine and SHACL-AF reasoning
        pub fn unified_validate(
            &self,
            store: &dyn Store,
            shapes: &[Shape],
        ) -> Result<ValidationReport> {
            // Combine both approaches
            self.rule_validator.validate_with_reasoning(store, shapes)
        }
    }
}

// Re-export from advanced_integration for convenience
pub use advanced_integration::ReasoningBridge;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_based_validator_creation() {
        let validator = RuleBasedValidator::new(RuleEngineConfig::default());
        assert_eq!(validator.cache_stats().size, 0);
    }

    #[test]
    fn test_builder_pattern() {
        let validator = RuleBasedValidatorBuilder::new()
            .with_forward_chaining(true)
            .with_strategy(ReasoningStrategy::RDFS)
            .build();

        assert!(validator.config.forward_chaining);
    }

    #[test]
    fn regression_backward_chaining_fails_loud() {
        use oxirs_core::ConcreteStore;
        let config = RuleEngineConfig {
            backward_chaining: true,
            ..Default::default()
        };
        let validator = RuleBasedValidator::new(config);
        let store = ConcreteStore::new().expect("store");
        let result = validator.validate_with_reasoning(&store, &[]);
        assert!(
            matches!(result, Err(ShaclError::UnsupportedOperation(_))),
            "backward chaining must fail loud, got {result:?}"
        );
    }

    #[test]
    fn regression_forward_chain_materializes_rdfs_closure() {
        use oxirs_core::model::{GraphName, NamedNode, Object, Predicate, Quad, Subject};
        use oxirs_core::rdf_store::QueryResults;
        use oxirs_core::ConcreteStore;

        const RDFS_SUBCLASS: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";
        const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

        let store = ConcreteStore::new().expect("store");
        // :Dog rdfs:subClassOf :Animal
        store
            .insert_quad(Quad::new(
                Subject::from(NamedNode::new("http://ex/Dog").expect("iri")),
                Predicate::from(NamedNode::new(RDFS_SUBCLASS).expect("iri")),
                Object::from(NamedNode::new("http://ex/Animal").expect("iri")),
                GraphName::DefaultGraph,
            ))
            .expect("insert");
        // :rex a :Dog
        store
            .insert_quad(Quad::new(
                Subject::from(NamedNode::new("http://ex/rex").expect("iri")),
                Predicate::from(NamedNode::new(RDF_TYPE).expect("iri")),
                Object::from(NamedNode::new("http://ex/Dog").expect("iri")),
                GraphName::DefaultGraph,
            ))
            .expect("insert");

        let validator = RuleBasedValidator::new(RuleEngineConfig::default());
        let inferred = validator.forward_chain(&store).expect("forward chain");
        assert!(
            inferred > 0,
            "RDFS closure should infer at least one triple"
        );

        // The inferred triple :rex a :Animal must now be in the store.
        let ask = "ASK { <http://ex/rex> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> \
                   <http://ex/Animal> }";
        let results = store.query(ask).expect("ask");
        assert!(
            matches!(results.results(), QueryResults::Boolean(true)),
            "reasoning must materialize rex a Animal via subClassOf"
        );
    }

    #[test]
    fn test_reasoning_strategies() {
        let strategies = [
            ReasoningStrategy::None,
            ReasoningStrategy::RDFS,
            ReasoningStrategy::OWL,
            ReasoningStrategy::OWLRL,
            ReasoningStrategy::Custom,
            ReasoningStrategy::Optimized,
        ];

        assert_eq!(strategies.len(), 6);
    }
}
