//! SHACL Validation Integration
//!
//! Provides integration between SHACL shape validation and rule-based reasoning.
//! Enables validation-driven reasoning and shape-aware rule application.
//!
//! # Features
//!
//! - **Validation Hooks**: Trigger rules before/after SHACL validation
//! - **Shape-Aware Reasoning**: Apply rules based on shape constraints
//! - **Constraint Repair**: Use rules to fix validation violations
//! - **Validation Caching**: Cache validation results for efficiency
//! - **Incremental Validation**: Validate only changed facts
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::shacl_integration::{ShaclRuleIntegration, ValidationMode};
//! use oxirs_rule::RuleEngine;
//!
//! let mut engine = RuleEngine::new();
//! let integration = ShaclRuleIntegration::new(engine);
//!
//! // Validate with rule-based reasoning
//! // let report = integration.validate_with_reasoning(&shapes, &data)?;
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::{Rule, RuleAtom, RuleEngine};
use anyhow::Result;
use scirs2_core::metrics::Timer;
use std::collections::HashMap;
use tracing::{debug, info, warn};

// Global metrics for validation performance
lazy_static::lazy_static! {
    static ref VALIDATION_DIRECT_TIMER: Timer = Timer::new("shacl_validation_direct".to_string());
    static ref VALIDATION_PRE_REASONING_TIMER: Timer = Timer::new("shacl_validation_pre_reasoning".to_string());
}

/// Validation execution mode
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationMode {
    /// Validate without reasoning
    Direct,
    /// Apply rules before validation
    PreReasoning,
    /// Apply rules after validation (repair)
    PostReasoning,
    /// Apply rules before and after validation
    Full,
}

/// Validation severity level
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational message
    Info,
    /// Warning - should be addressed
    Warning,
    /// Violation - must be fixed
    Violation,
}

/// Validation result for a single constraint
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the constraint is satisfied
    pub valid: bool,
    /// Severity level
    pub severity: Severity,
    /// Constraint that was checked
    pub constraint_type: String,
    /// Focus node that violated the constraint
    pub focus_node: Option<String>,
    /// Value that violated the constraint
    pub value: Option<String>,
    /// Message describing the violation
    pub message: String,
}

impl ValidationResult {
    /// Create a new validation result
    pub fn new(valid: bool, severity: Severity, constraint_type: String, message: String) -> Self {
        Self {
            valid,
            severity,
            constraint_type,
            focus_node: None,
            value: None,
            message,
        }
    }

    /// Set focus node
    pub fn with_focus_node(mut self, node: String) -> Self {
        self.focus_node = Some(node);
        self
    }

    /// Set value
    pub fn with_value(mut self, value: String) -> Self {
        self.value = Some(value);
        self
    }
}

/// Validation report
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Whether overall validation passed
    pub conforms: bool,
    /// Individual validation results
    pub results: Vec<ValidationResult>,
    /// Statistics
    pub stats: ValidationStats,
}

impl ValidationReport {
    /// Create a new validation report
    pub fn new(conforms: bool) -> Self {
        Self {
            conforms,
            results: Vec::new(),
            stats: ValidationStats::default(),
        }
    }

    /// Add a validation result
    pub fn add_result(&mut self, result: ValidationResult) {
        self.conforms = self.conforms && result.valid;
        self.results.push(result);
    }

    /// Get violation count
    pub fn violation_count(&self) -> usize {
        self.results
            .iter()
            .filter(|r| !r.valid && r.severity == Severity::Violation)
            .count()
    }

    /// Get warning count
    pub fn warning_count(&self) -> usize {
        self.results
            .iter()
            .filter(|r| !r.valid && r.severity == Severity::Warning)
            .count()
    }
}

/// Validation statistics
#[derive(Debug, Clone, Default)]
pub struct ValidationStats {
    /// Total shapes validated
    pub shapes_validated: usize,
    /// Total constraints checked
    pub constraints_checked: usize,
    /// Validation time in milliseconds
    pub validation_time_ms: u128,
    /// Rules applied during validation
    pub rules_applied: usize,
}

/// Shape constraint for validation
#[derive(Debug, Clone)]
pub struct ShapeConstraint {
    /// Constraint identifier
    pub id: String,
    /// Constraint type (e.g., "sh:minCount", "sh:pattern")
    pub constraint_type: String,
    /// Target predicate
    pub predicate: Option<String>,
    /// Expected value or pattern
    pub expected: Option<String>,
    /// Severity level
    pub severity: Severity,
}

impl ShapeConstraint {
    /// Create a new shape constraint
    pub fn new(id: String, constraint_type: String) -> Self {
        Self {
            id,
            constraint_type,
            predicate: None,
            expected: None,
            severity: Severity::Violation,
        }
    }
}

/// SHACL-Rule integration manager
pub struct ShaclRuleIntegration {
    /// Underlying rule engine
    engine: RuleEngine,
    /// Validation mode
    mode: ValidationMode,
    /// Shape-specific rules
    shape_rules: HashMap<String, Vec<String>>,
    /// Constraint repair rules
    repair_rules: HashMap<String, Vec<Rule>>,
    /// Validation cache
    validation_cache: HashMap<String, ValidationResult>,
    /// Statistics
    stats: IntegrationStats,
    /// Cached inferred facts (optimization)
    inferred_cache: Option<Vec<RuleAtom>>,
    /// Hash of data when cache was created
    data_hash: u64,
}

impl ShaclRuleIntegration {
    /// Create new integration
    pub fn new(engine: RuleEngine) -> Self {
        Self {
            engine,
            mode: ValidationMode::Full,
            shape_rules: HashMap::new(),
            repair_rules: HashMap::new(),
            validation_cache: HashMap::new(),
            stats: IntegrationStats::default(),
            inferred_cache: None,
            data_hash: 0,
        }
    }

    /// Set validation mode
    pub fn set_mode(&mut self, mode: ValidationMode) {
        info!("Setting validation mode to {:?}", mode);
        self.mode = mode;
    }

    /// Get current mode
    pub fn get_mode(&self) -> &ValidationMode {
        &self.mode
    }

    /// Register a shape-specific rule
    pub fn register_shape_rule(&mut self, shape_id: String, rule_name: String) {
        debug!("Registering rule '{}' for shape '{}'", rule_name, shape_id);
        self.shape_rules
            .entry(shape_id)
            .or_default()
            .push(rule_name);
    }

    /// Register a constraint repair rule
    pub fn register_repair_rule(&mut self, constraint_type: String, rule: Rule) {
        debug!(
            "Registering repair rule '{}' for constraint '{}'",
            rule.name, constraint_type
        );
        self.repair_rules
            .entry(constraint_type)
            .or_default()
            .push(rule);
    }

    /// Validate with reasoning (optimized with SIMD)
    pub fn validate_with_reasoning(
        &mut self,
        constraints: &[ShapeConstraint],
        data: &[RuleAtom],
    ) -> Result<ValidationReport> {
        let _validation_timer = VALIDATION_DIRECT_TIMER.start();
        self.stats.total_validations += 1;
        let start = std::time::Instant::now();

        let mut report = ValidationReport::new(true);

        // Apply pre-reasoning if needed
        let data_to_validate = match self.mode {
            ValidationMode::PreReasoning | ValidationMode::Full => {
                self.apply_pre_reasoning(data)?
            }
            _ => {
                // Use SIMD deduplication for direct mode with large datasets
                if data.len() > 100 {
                    use crate::simd_ops::SimdMatcher;
                    let matcher = SimdMatcher::new();
                    let mut deduped = data.to_vec();
                    matcher.batch_deduplicate(&mut deduped);
                    deduped
                } else {
                    data.to_vec()
                }
            }
        };

        // Perform validation
        for constraint in constraints {
            let result = self.validate_constraint(constraint, &data_to_validate)?;
            report.add_result(result.clone());

            // Cache result
            self.validation_cache.insert(constraint.id.clone(), result);
        }

        report.stats.validation_time_ms = start.elapsed().as_millis();
        report.stats.shapes_validated = 1;
        report.stats.constraints_checked = constraints.len();

        // Apply post-reasoning if needed (repair)
        if !report.conforms
            && (self.mode == ValidationMode::PostReasoning || self.mode == ValidationMode::Full)
        {
            self.apply_repairs(&mut report, data)?;
        }

        Ok(report)
    }

    /// Apply pre-validation reasoning (optimized with caching)
    fn apply_pre_reasoning(&mut self, data: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        let _timer = VALIDATION_PRE_REASONING_TIMER.start();
        debug!("Applying pre-validation reasoning");

        // Use cached inference for performance
        let inferred = self.get_inferred_facts(data)?;
        self.stats.pre_reasoning_applications += 1;

        Ok(inferred)
    }

    /// Validate a single constraint
    fn validate_constraint(
        &self,
        constraint: &ShapeConstraint,
        data: &[RuleAtom],
    ) -> Result<ValidationResult> {
        // Check cache first
        if let Some(cached) = self.validation_cache.get(&constraint.id) {
            debug!("Cache hit for constraint '{}'", constraint.id);
            return Ok(cached.clone());
        }

        // Simplified validation - in a full implementation, this would
        // check various SHACL constraint types
        let valid = self.check_constraint(constraint, data);

        let message = if valid {
            format!("Constraint '{}' satisfied", constraint.constraint_type)
        } else {
            format!("Constraint '{}' violated", constraint.constraint_type)
        };

        Ok(ValidationResult::new(
            valid,
            constraint.severity.clone(),
            constraint.constraint_type.clone(),
            message,
        ))
    }

    /// Check if constraint is satisfied
    fn check_constraint(&self, constraint: &ShapeConstraint, data: &[RuleAtom]) -> bool {
        // Simplified check - just verify some data exists
        // In a full implementation, this would check specific constraint types
        match constraint.constraint_type.as_str() {
            "sh:minCount" => {
                data.len()
                    >= constraint
                        .expected
                        .as_ref()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1)
            }
            "sh:maxCount" => {
                data.len()
                    <= constraint
                        .expected
                        .as_ref()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(100)
            }
            _ => !data.is_empty(),
        }
    }

    /// Apply constraint repairs
    fn apply_repairs(&mut self, report: &mut ValidationReport, data: &[RuleAtom]) -> Result<()> {
        debug!("Applying constraint repairs");

        let violations: Vec<_> = report
            .results
            .iter()
            .filter(|r| !r.valid && r.severity == Severity::Violation)
            .collect();

        for violation in violations {
            if let Some(repair_rules) = self.repair_rules.get(&violation.constraint_type) {
                for rule in repair_rules {
                    self.engine.add_rule(rule.clone());
                }

                // Re-apply reasoning
                let repaired = self.engine.forward_chain(data)?;

                // Re-validate
                warn!(
                    "Applied {} repair rules for constraint '{}'",
                    repair_rules.len(),
                    violation.constraint_type
                );

                self.stats.repairs_applied += repair_rules.len();
                self.stats.post_reasoning_applications += 1;

                // Update facts
                self.engine.add_facts(repaired);
            }
        }

        Ok(())
    }

    /// Clear validation cache
    pub fn clear_cache(&mut self) {
        debug!(
            "Clearing validation cache ({} entries)",
            self.validation_cache.len()
        );
        self.validation_cache.clear();
    }

    /// Get underlying engine (mutable)
    pub fn engine_mut(&mut self) -> &mut RuleEngine {
        &mut self.engine
    }

    /// Get underlying engine (immutable)
    pub fn engine(&self) -> &RuleEngine {
        &self.engine
    }

    /// Get integration statistics
    pub fn get_stats(&self) -> &IntegrationStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = IntegrationStats::default();
    }

    /// Compute hash of data for cache invalidation
    fn compute_data_hash(&self, data: &[RuleAtom]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.len().hash(&mut hasher);

        // Hash first and last few items for performance
        let sample_size = data.len().min(10);
        for atom in data.iter().take(sample_size) {
            format!("{:?}", atom).hash(&mut hasher);
        }
        if data.len() > sample_size {
            for atom in data.iter().skip(data.len() - sample_size) {
                format!("{:?}", atom).hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Invalidate inference cache
    pub fn invalidate_cache(&mut self) {
        self.inferred_cache = None;
        self.data_hash = 0;
        debug!("Inference cache invalidated");
    }

    /// Get or compute inferred facts with caching
    fn get_inferred_facts(&mut self, data: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        let current_hash = self.compute_data_hash(data);

        // Check if cache is valid
        if let Some(ref cached) = self.inferred_cache {
            if current_hash == self.data_hash {
                debug!("Using cached inferred facts ({} facts)", cached.len());
                self.stats.cache_hits += 1;
                return Ok(cached.clone());
            }
        }

        // Cache miss - compute inference
        debug!("Cache miss - inferring facts");
        self.stats.cache_misses += 1;

        self.engine.add_facts(data.to_vec());
        let inferred = self.engine.forward_chain(data)?;

        // Update cache
        self.inferred_cache = Some(inferred.clone());
        self.data_hash = current_hash;

        Ok(inferred)
    }
}

/// Integration statistics
#[derive(Debug, Clone, Default)]
pub struct IntegrationStats {
    /// Total validations performed
    pub total_validations: usize,
    /// Pre-reasoning applications
    pub pre_reasoning_applications: usize,
    /// Post-reasoning applications
    pub post_reasoning_applications: usize,
    /// Repairs applied
    pub repairs_applied: usize,
    /// Inference cache hits
    pub cache_hits: usize,
    /// Inference cache misses
    pub cache_misses: usize,
}

impl std::fmt::Display for IntegrationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Validations: {}, Pre-reasoning: {}, Post-reasoning: {}, Repairs: {}, Cache(hits/misses): {}/{}",
            self.total_validations,
            self.pre_reasoning_applications,
            self.post_reasoning_applications,
            self.repairs_applied,
            self.cache_hits,
            self.cache_misses
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Term;

    #[test]
    fn test_validation_result_creation() {
        let result = ValidationResult::new(
            true,
            Severity::Info,
            "sh:minCount".to_string(),
            "Min count satisfied".to_string(),
        );

        assert!(result.valid);
        assert_eq!(result.severity, Severity::Info);
    }

    #[test]
    fn test_validation_report() {
        let mut report = ValidationReport::new(true);

        report.add_result(ValidationResult::new(
            false,
            Severity::Violation,
            "sh:minCount".to_string(),
            "Min count not satisfied".to_string(),
        ));

        assert!(!report.conforms);
        assert_eq!(report.violation_count(), 1);
    }

    #[test]
    fn test_shacl_integration_creation() {
        let engine = RuleEngine::new();
        let integration = ShaclRuleIntegration::new(engine);

        assert_eq!(*integration.get_mode(), ValidationMode::Full);
    }

    #[test]
    fn test_validation_mode_setting() {
        let engine = RuleEngine::new();
        let mut integration = ShaclRuleIntegration::new(engine);

        integration.set_mode(ValidationMode::PreReasoning);
        assert_eq!(*integration.get_mode(), ValidationMode::PreReasoning);
    }

    #[test]
    fn test_shape_rule_registration() {
        let engine = RuleEngine::new();
        let mut integration = ShaclRuleIntegration::new(engine);

        integration.register_shape_rule("PersonShape".to_string(), "age_validation".to_string());

        assert_eq!(integration.shape_rules.len(), 1);
    }

    #[test]
    fn test_repair_rule_registration() {
        let engine = RuleEngine::new();
        let mut integration = ShaclRuleIntegration::new(engine);

        let repair_rule = Rule {
            name: "fix_mincount".to_string(),
            body: vec![],
            head: vec![],
        };

        integration.register_repair_rule("sh:minCount".to_string(), repair_rule);

        assert_eq!(integration.repair_rules.len(), 1);
    }

    #[test]
    fn test_validation_with_reasoning() {
        let mut engine = RuleEngine::new();
        engine.add_fact(RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("Person".to_string()),
        });

        let mut integration = ShaclRuleIntegration::new(engine);

        let constraint = ShapeConstraint::new("c1".to_string(), "sh:minCount".to_string());

        let data = vec![RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("age".to_string()),
            object: Term::Literal("30".to_string()),
        }];

        let report = integration
            .validate_with_reasoning(&[constraint], &data)
            .unwrap();

        assert_eq!(report.results.len(), 1);
    }

    #[test]
    fn test_cache_clearing() {
        let engine = RuleEngine::new();
        let mut integration = ShaclRuleIntegration::new(engine);

        integration.validation_cache.insert(
            "test".to_string(),
            ValidationResult::new(true, Severity::Info, "test".to_string(), "test".to_string()),
        );

        assert_eq!(integration.validation_cache.len(), 1);

        integration.clear_cache();
        assert_eq!(integration.validation_cache.len(), 0);
    }
}
