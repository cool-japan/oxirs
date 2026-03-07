//! # Defeasible Reasoning for SHACL
//!
//! This module implements defeasible logic for SHACL validation, allowing for:
//! - Default rules that can be overridden
//! - Prioritized constraint resolution
//! - Exception handling in validation
//! - Conflict resolution strategies
//!
//! ## Overview
//!
//! Defeasible logic extends classical logic with reasoning that can be defeated by
//! contrary evidence. This is particularly useful for:
//! - Policy validation with exceptions
//! - Context-dependent constraints
//! - Hierarchical validation rules
//! - Default reasoning with overrides

use crate::report::ValidationSummary;
use crate::{Result, ShaclError, ShapeId, ValidationReport, ValidationViolation};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info};

/// Priority level for defeasible rules
pub type Priority = i32;

/// Defeasible rule for SHACL validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefeasibleRule {
    /// Unique identifier
    pub id: String,

    /// Rule name
    pub name: String,

    /// Rule description
    pub description: Option<String>,

    /// Priority (higher value = higher priority)
    pub priority: Priority,

    /// Rule type
    pub rule_type: DefeasibleRuleType,

    /// Condition that activates this rule
    pub condition: RuleCondition,

    /// Action to perform when rule fires
    pub action: RuleAction,

    /// Rules that this rule overrides
    pub overrides: Vec<String>,

    /// Rules that override this rule
    pub overridden_by: Vec<String>,

    /// Whether this rule is enabled
    pub enabled: bool,
}

/// Type of defeasible rule
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DefeasibleRuleType {
    /// Strict rule - cannot be overridden
    Strict,

    /// Default rule - can be overridden by higher priority rules
    Default,

    /// Exception rule - overrides default rules
    Exception,

    /// Preference rule - used for conflict resolution
    Preference,
}

/// Condition for rule activation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleCondition {
    /// Always active
    Always,

    /// Active for specific shape
    ShapeId(ShapeId),

    /// Active for shape targeting specific class
    TargetClass(String),

    /// Active when property has specific value
    PropertyValue { property: String, value: String },

    /// Active in specific context
    Context(HashMap<String, String>),

    /// Logical AND of conditions
    And(Vec<RuleCondition>),

    /// Logical OR of conditions
    Or(Vec<RuleCondition>),

    /// Logical NOT of condition
    Not(Box<RuleCondition>),
}

/// Action to perform when rule fires
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleAction {
    /// Override severity of violation
    OverrideSeverity(crate::Severity),

    /// Suppress violation
    SuppressViolation,

    /// Add additional constraint
    AddConstraint(String),

    /// Relax constraint
    RelaxConstraint {
        constraint_id: String,
        relaxation: f64,
    },

    /// Custom action
    Custom(String),
}

/// Defeasible reasoning engine for SHACL
pub struct DefeasibleEngine {
    /// Configured rules
    rules: Vec<DefeasibleRule>,

    /// Rule dependency graph
    dependency_graph: HashMap<String, Vec<String>>,

    /// Conflict resolution strategy
    strategy: ConflictResolutionStrategy,

    /// Execution statistics
    stats: DefeasibleStats,
}

impl DefeasibleEngine {
    /// Create a new defeasible reasoning engine
    pub fn new(strategy: ConflictResolutionStrategy) -> Self {
        Self {
            rules: Vec::new(),
            dependency_graph: HashMap::new(),
            strategy,
            stats: DefeasibleStats::default(),
        }
    }

    /// Add a defeasible rule
    pub fn add_rule(&mut self, rule: DefeasibleRule) -> Result<()> {
        // Validate rule
        self.validate_rule(&rule)?;

        // Update dependency graph
        self.update_dependency_graph(&rule);

        // Check for cycles
        if self.has_cycles() {
            return Err(ShaclError::ValidationEngine(
                "Circular rule dependencies detected".to_string(),
            ));
        }

        self.rules.push(rule);
        Ok(())
    }

    /// Remove a rule
    pub fn remove_rule(&mut self, rule_id: &str) -> Result<()> {
        self.rules.retain(|r| r.id != rule_id);
        self.rebuild_dependency_graph();
        Ok(())
    }

    /// Get all rules
    pub fn rules(&self) -> &[DefeasibleRule] {
        &self.rules
    }

    /// Apply defeasible reasoning to validation report
    pub fn apply_reasoning(&mut self, report: &ValidationReport) -> Result<ValidationReport> {
        info!(
            "Applying defeasible reasoning to {} violations",
            report.violations.len()
        );

        let mut modified_violations = Vec::new();
        let mut suppressed_count = 0;

        for violation in &report.violations {
            match self.process_violation(violation)? {
                ViolationResolution::Keep(v) => modified_violations.push(v),
                ViolationResolution::Modify(v) => {
                    debug!("Modified violation: {:?}", v);
                    modified_violations.push(v);
                }
                ViolationResolution::Suppress => {
                    debug!("Suppressed violation");
                    suppressed_count += 1;
                }
            }
        }

        self.stats.violations_processed += report.violations.len();
        self.stats.violations_suppressed += suppressed_count;
        self.stats.violations_modified += modified_violations.len();

        Ok(ValidationReport {
            conforms: modified_violations.is_empty(),
            violations: modified_violations,
            metadata: report.metadata.clone(),
            summary: ValidationSummary::default(),
        })
    }

    /// Resolve conflicts between rules
    pub fn resolve_conflicts<'a>(
        &self,
        applicable_rules: &[&'a DefeasibleRule],
    ) -> Result<Vec<&'a DefeasibleRule>> {
        match self.strategy {
            ConflictResolutionStrategy::Priority => self.resolve_by_priority(applicable_rules),
            ConflictResolutionStrategy::Specificity => {
                self.resolve_by_specificity(applicable_rules)
            }
            ConflictResolutionStrategy::Recency => self.resolve_by_recency(applicable_rules),
            ConflictResolutionStrategy::Custom => self.resolve_custom(applicable_rules),
        }
    }

    /// Get execution statistics
    pub fn stats(&self) -> &DefeasibleStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = DefeasibleStats::default();
    }

    // Private methods

    fn validate_rule(&self, rule: &DefeasibleRule) -> Result<()> {
        if rule.id.is_empty() {
            return Err(ShaclError::Configuration(
                "Rule ID cannot be empty".to_string(),
            ));
        }

        if self.rules.iter().any(|r| r.id == rule.id) {
            return Err(ShaclError::Configuration(format!(
                "Rule with ID '{}' already exists",
                rule.id
            )));
        }

        Ok(())
    }

    fn update_dependency_graph(&mut self, rule: &DefeasibleRule) {
        self.dependency_graph
            .insert(rule.id.clone(), rule.overrides.clone());
    }

    fn rebuild_dependency_graph(&mut self) {
        self.dependency_graph.clear();
        let rules_data: Vec<(String, Vec<String>)> = self
            .rules
            .iter()
            .map(|r| (r.id.clone(), r.overrides.clone()))
            .collect();

        for (id, overrides) in rules_data {
            self.dependency_graph.insert(id, overrides);
        }
    }

    fn has_cycles(&self) -> bool {
        // Use DFS to detect cycles
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for rule_id in self.dependency_graph.keys() {
            if !visited.contains(rule_id)
                && self.has_cycle_util(rule_id, &mut visited, &mut rec_stack)
            {
                return true;
            }
        }

        false
    }

    fn has_cycle_util(
        &self,
        rule_id: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        visited.insert(rule_id.to_string());
        rec_stack.insert(rule_id.to_string());

        if let Some(dependencies) = self.dependency_graph.get(rule_id) {
            for dep_id in dependencies {
                if !visited.contains(dep_id) {
                    if self.has_cycle_util(dep_id, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(dep_id) {
                    return true;
                }
            }
        }

        rec_stack.remove(rule_id);
        false
    }

    fn process_violation(
        &mut self,
        violation: &ValidationViolation,
    ) -> Result<ViolationResolution> {
        // Find applicable rules and collect their data
        let applicable_rules: Vec<DefeasibleRule> = self
            .rules
            .iter()
            .filter(|r| r.enabled && self.rule_applies(r, violation))
            .cloned()
            .collect();

        if applicable_rules.is_empty() {
            return Ok(ViolationResolution::Keep(violation.clone()));
        }

        // Create references for conflict resolution
        let rule_refs: Vec<&DefeasibleRule> = applicable_rules.iter().collect();
        let resolved_rules = self.resolve_conflicts(&rule_refs)?;

        // Apply the highest priority rule
        if let Some(&rule) = resolved_rules.first() {
            self.apply_rule_action(rule, violation)
        } else {
            Ok(ViolationResolution::Keep(violation.clone()))
        }
    }

    #[allow(clippy::only_used_in_recursion)]
    fn rule_applies(&self, rule: &DefeasibleRule, _violation: &ValidationViolation) -> bool {
        match &rule.condition {
            RuleCondition::Always => true,
            RuleCondition::ShapeId(_) => false, // Would check violation.shape_id
            RuleCondition::TargetClass(_) => false,
            RuleCondition::PropertyValue { .. } => false,
            RuleCondition::Context(_) => false,
            RuleCondition::And(conditions) => conditions.iter().all(|c| {
                self.rule_applies(
                    &DefeasibleRule {
                        id: rule.id.clone(),
                        name: rule.name.clone(),
                        description: rule.description.clone(),
                        priority: rule.priority,
                        rule_type: rule.rule_type.clone(),
                        condition: c.clone(),
                        action: rule.action.clone(),
                        overrides: rule.overrides.clone(),
                        overridden_by: rule.overridden_by.clone(),
                        enabled: rule.enabled,
                    },
                    _violation,
                )
            }),
            RuleCondition::Or(conditions) => conditions.iter().any(|c| {
                self.rule_applies(
                    &DefeasibleRule {
                        id: rule.id.clone(),
                        name: rule.name.clone(),
                        description: rule.description.clone(),
                        priority: rule.priority,
                        rule_type: rule.rule_type.clone(),
                        condition: c.clone(),
                        action: rule.action.clone(),
                        overrides: rule.overrides.clone(),
                        overridden_by: rule.overridden_by.clone(),
                        enabled: rule.enabled,
                    },
                    _violation,
                )
            }),
            RuleCondition::Not(condition) => !self.rule_applies(
                &DefeasibleRule {
                    id: rule.id.clone(),
                    name: rule.name.clone(),
                    description: rule.description.clone(),
                    priority: rule.priority,
                    rule_type: rule.rule_type.clone(),
                    condition: (**condition).clone(),
                    action: rule.action.clone(),
                    overrides: rule.overrides.clone(),
                    overridden_by: rule.overridden_by.clone(),
                    enabled: rule.enabled,
                },
                _violation,
            ),
        }
    }

    fn apply_rule_action(
        &mut self,
        rule: &DefeasibleRule,
        violation: &ValidationViolation,
    ) -> Result<ViolationResolution> {
        self.stats.rules_applied += 1;

        match &rule.action {
            RuleAction::OverrideSeverity(severity) => {
                let mut modified = violation.clone();
                modified.result_severity = *severity;
                Ok(ViolationResolution::Modify(modified))
            }
            RuleAction::SuppressViolation => Ok(ViolationResolution::Suppress),
            RuleAction::AddConstraint(_) => Ok(ViolationResolution::Keep(violation.clone())),
            RuleAction::RelaxConstraint { .. } => Ok(ViolationResolution::Keep(violation.clone())),
            RuleAction::Custom(_) => Ok(ViolationResolution::Keep(violation.clone())),
        }
    }

    fn resolve_by_priority<'a>(
        &self,
        rules: &[&'a DefeasibleRule],
    ) -> Result<Vec<&'a DefeasibleRule>> {
        let mut sorted = rules.to_vec();
        sorted.sort_by_key(|r| std::cmp::Reverse(r.priority));

        // Return only the highest priority rules
        if let Some(first) = sorted.first() {
            let max_priority = first.priority;
            Ok(sorted
                .into_iter()
                .filter(|r| r.priority == max_priority)
                .collect())
        } else {
            Ok(Vec::new())
        }
    }

    fn resolve_by_specificity<'a>(
        &self,
        rules: &[&'a DefeasibleRule],
    ) -> Result<Vec<&'a DefeasibleRule>> {
        // More specific rules (with more detailed conditions) take precedence
        let mut sorted = rules.to_vec();
        sorted.sort_by_key(|r| std::cmp::Reverse(self.rule_specificity(r)));

        if let Some(first) = sorted.first() {
            let max_specificity = self.rule_specificity(first);
            Ok(sorted
                .into_iter()
                .filter(|r| self.rule_specificity(r) == max_specificity)
                .collect())
        } else {
            Ok(Vec::new())
        }
    }

    fn resolve_by_recency<'a>(
        &self,
        rules: &[&'a DefeasibleRule],
    ) -> Result<Vec<&'a DefeasibleRule>> {
        // Later rules override earlier ones
        if let Some(last) = rules.last() {
            Ok(vec![*last])
        } else {
            Ok(Vec::new())
        }
    }

    fn resolve_custom<'a>(&self, rules: &[&'a DefeasibleRule]) -> Result<Vec<&'a DefeasibleRule>> {
        // Custom resolution logic
        self.resolve_by_priority(rules)
    }

    fn rule_specificity(&self, rule: &DefeasibleRule) -> usize {
        // Calculate specificity based on condition complexity
        self.condition_specificity(&rule.condition)
    }

    #[allow(clippy::only_used_in_recursion)]
    fn condition_specificity(&self, condition: &RuleCondition) -> usize {
        match condition {
            RuleCondition::Always => 0,
            RuleCondition::ShapeId(_) => 1,
            RuleCondition::TargetClass(_) => 1,
            RuleCondition::PropertyValue { .. } => 2,
            RuleCondition::Context(ctx) => ctx.len(),
            RuleCondition::And(conditions) => conditions
                .iter()
                .map(|c| self.condition_specificity(c))
                .sum(),
            RuleCondition::Or(conditions) => conditions
                .iter()
                .map(|c| self.condition_specificity(c))
                .max()
                .unwrap_or(0),
            RuleCondition::Not(condition) => self.condition_specificity(condition),
        }
    }
}

/// Strategy for resolving conflicts between rules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolutionStrategy {
    /// Resolve by priority (higher priority wins)
    Priority,

    /// Resolve by specificity (more specific wins)
    Specificity,

    /// Resolve by recency (later rules win)
    Recency,

    /// Custom resolution strategy
    Custom,
}

/// Result of processing a violation with defeasible reasoning
#[derive(Debug, Clone)]
enum ViolationResolution {
    /// Keep the violation as-is
    Keep(ValidationViolation),

    /// Modify the violation
    Modify(ValidationViolation),

    /// Suppress the violation
    Suppress,
}

/// Statistics for defeasible reasoning
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DefeasibleStats {
    /// Number of violations processed
    pub violations_processed: usize,

    /// Number of violations suppressed
    pub violations_suppressed: usize,

    /// Number of violations modified
    pub violations_modified: usize,

    /// Number of rules applied
    pub rules_applied: usize,
}

/// Builder for defeasible engine
pub struct DefeasibleEngineBuilder {
    strategy: ConflictResolutionStrategy,
    rules: Vec<DefeasibleRule>,
}

impl DefeasibleEngineBuilder {
    pub fn new() -> Self {
        Self {
            strategy: ConflictResolutionStrategy::Priority,
            rules: Vec::new(),
        }
    }

    pub fn with_strategy(mut self, strategy: ConflictResolutionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn add_rule(mut self, rule: DefeasibleRule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn build(self) -> Result<DefeasibleEngine> {
        let mut engine = DefeasibleEngine::new(self.strategy);

        for rule in self.rules {
            engine.add_rule(rule)?;
        }

        Ok(engine)
    }
}

impl Default for DefeasibleEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defeasible_engine_creation() {
        let engine = DefeasibleEngine::new(ConflictResolutionStrategy::Priority);
        assert_eq!(engine.rules().len(), 0);
    }

    #[test]
    fn test_add_rule() {
        let mut engine = DefeasibleEngine::new(ConflictResolutionStrategy::Priority);

        let rule = DefeasibleRule {
            id: "rule1".to_string(),
            name: "Test Rule".to_string(),
            description: None,
            priority: 10,
            rule_type: DefeasibleRuleType::Default,
            condition: RuleCondition::Always,
            action: RuleAction::SuppressViolation,
            overrides: Vec::new(),
            overridden_by: Vec::new(),
            enabled: true,
        };

        engine.add_rule(rule).expect("operation should succeed");
        assert_eq!(engine.rules().len(), 1);
    }

    #[test]
    fn test_conflict_resolution_priority() {
        let engine = DefeasibleEngine::new(ConflictResolutionStrategy::Priority);

        let rule1 = DefeasibleRule {
            id: "rule1".to_string(),
            name: "Low Priority".to_string(),
            description: None,
            priority: 1,
            rule_type: DefeasibleRuleType::Default,
            condition: RuleCondition::Always,
            action: RuleAction::SuppressViolation,
            overrides: Vec::new(),
            overridden_by: Vec::new(),
            enabled: true,
        };

        let rule2 = DefeasibleRule {
            id: "rule2".to_string(),
            name: "High Priority".to_string(),
            description: None,
            priority: 10,
            rule_type: DefeasibleRuleType::Exception,
            condition: RuleCondition::Always,
            action: RuleAction::SuppressViolation,
            overrides: Vec::new(),
            overridden_by: Vec::new(),
            enabled: true,
        };

        let resolved = engine
            .resolve_by_priority(&[&rule1, &rule2])
            .expect("resolution should succeed");
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].id, "rule2");
    }

    #[test]
    fn test_builder_pattern() {
        let rule = DefeasibleRule {
            id: "rule1".to_string(),
            name: "Test Rule".to_string(),
            description: None,
            priority: 10,
            rule_type: DefeasibleRuleType::Default,
            condition: RuleCondition::Always,
            action: RuleAction::SuppressViolation,
            overrides: Vec::new(),
            overridden_by: Vec::new(),
            enabled: true,
        };

        let engine = DefeasibleEngineBuilder::new()
            .with_strategy(ConflictResolutionStrategy::Priority)
            .add_rule(rule)
            .build()
            .expect("operation should succeed");

        assert_eq!(engine.rules().len(), 1);
    }
}
