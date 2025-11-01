//! SHACL Advanced Features - Rules (SHACL-AF)
//!
//! Implementation of SHACL Rules for data transformation and inferencing.
//! Based on the W3C SHACL Advanced Features specification.
//!
//! Note: This is an alpha implementation with stub methods that will be
//! fully implemented in future releases.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use oxirs_core::{model::NamedNode, Store};

use crate::{PropertyPath, Result, ShaclError};

/// SHACL Rule types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuleType {
    /// Triple rule - generates triples based on SPARQL query
    TripleRule,
    /// SPARQL CONSTRUCT rule - uses CONSTRUCT query
    ConstructRule,
    /// SPARQLRule - generic SPARQL-based rule
    SparqlRule,
}

/// A SHACL Rule for data transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaclRule {
    /// Rule identifier
    pub id: String,

    /// Rule type
    pub rule_type: RuleType,

    /// Condition query (SELECT query that determines when to apply the rule)
    pub condition: Option<String>,

    /// Subject path or expression
    pub subject: Option<PropertyPath>,

    /// Predicate
    pub predicate: Option<NamedNode>,

    /// Object path or expression
    pub object: Option<PropertyPath>,

    /// CONSTRUCT query (for ConstructRule)
    pub construct: Option<String>,

    /// Generic SPARQL query (for SparqlRule)
    pub sparql: Option<String>,

    /// Priority for rule execution order
    pub order: Option<i32>,

    /// Whether this rule is deactivated
    pub deactivated: bool,

    /// Rule metadata
    pub metadata: RuleMetadata,
}

/// Metadata for SHACL Rules
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuleMetadata {
    /// Human-readable label
    pub label: Option<String>,

    /// Description of what this rule does
    pub description: Option<String>,

    /// Author
    pub author: Option<String>,

    /// Version
    pub version: Option<String>,

    /// Tags for categorization
    pub tags: Vec<String>,

    /// Custom properties
    pub custom: HashMap<String, String>,
}

impl ShaclRule {
    /// Create a new triple rule
    pub fn triple_rule(
        id: String,
        subject: PropertyPath,
        predicate: NamedNode,
        object: PropertyPath,
    ) -> Self {
        Self {
            id,
            rule_type: RuleType::TripleRule,
            condition: None,
            subject: Some(subject),
            predicate: Some(predicate),
            object: Some(object),
            construct: None,
            sparql: None,
            order: None,
            deactivated: false,
            metadata: RuleMetadata::default(),
        }
    }

    /// Create a new CONSTRUCT rule
    pub fn construct_rule(id: String, construct_query: String) -> Self {
        Self {
            id,
            rule_type: RuleType::ConstructRule,
            condition: None,
            subject: None,
            predicate: None,
            object: None,
            construct: Some(construct_query),
            sparql: None,
            order: None,
            deactivated: false,
            metadata: RuleMetadata::default(),
        }
    }

    /// Create a new generic SPARQL rule
    pub fn sparql_rule(id: String, sparql_query: String) -> Self {
        Self {
            id,
            rule_type: RuleType::SparqlRule,
            condition: None,
            subject: None,
            predicate: None,
            object: None,
            construct: None,
            sparql: Some(sparql_query),
            order: None,
            deactivated: false,
            metadata: RuleMetadata::default(),
        }
    }

    /// Set the condition query for this rule
    pub fn with_condition(mut self, condition: String) -> Self {
        self.condition = Some(condition);
        self
    }

    /// Set the execution order for this rule
    pub fn with_order(mut self, order: i32) -> Self {
        self.order = Some(order);
        self
    }

    /// Set metadata for this rule
    pub fn with_metadata(mut self, metadata: RuleMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Check if this rule is active
    pub fn is_active(&self) -> bool {
        !self.deactivated
    }

    /// Get the execution order (defaults to 0)
    pub fn effective_order(&self) -> i32 {
        self.order.unwrap_or(0)
    }
}

/// Result of rule execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleExecutionResult {
    /// Rule ID
    pub rule_id: String,

    /// Number of triples added
    pub triples_added: usize,

    /// Number of triples removed
    pub triples_removed: usize,

    /// Number of nodes affected
    pub nodes_affected: usize,

    /// Execution time in milliseconds
    pub execution_time_ms: u64,

    /// Any errors encountered
    pub errors: Vec<String>,

    /// Whether the rule execution was successful
    pub success: bool,
}

/// SHACL Rule Engine for executing transformation rules
pub struct RuleEngine {
    /// Registered rules
    rules: HashMap<String, ShaclRule>,

    /// Rule execution statistics
    stats: RuleEngineStats,

    /// Maximum number of iterations for recursive rule application
    max_iterations: usize,

    /// Cache of compiled SPARQL queries
    query_cache: HashMap<String, String>,
}

/// Statistics for rule engine execution
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuleEngineStats {
    /// Total number of rules executed
    pub rules_executed: usize,

    /// Total triples added across all rules
    pub total_triples_added: usize,

    /// Total triples removed across all rules
    pub total_triples_removed: usize,

    /// Total execution time in milliseconds
    pub total_execution_time_ms: u64,

    /// Number of rule execution errors
    pub errors: usize,
}

impl RuleEngine {
    /// Create a new rule engine
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            stats: RuleEngineStats::default(),
            max_iterations: 100,
            query_cache: HashMap::new(),
        }
    }

    /// Register a new rule
    pub fn register_rule(&mut self, rule: ShaclRule) -> Result<()> {
        if self.rules.contains_key(&rule.id) {
            return Err(ShaclError::Configuration(format!(
                "Rule {} already registered",
                rule.id
            )));
        }
        self.rules.insert(rule.id.clone(), rule);
        Ok(())
    }

    /// Unregister a rule
    pub fn unregister_rule(&mut self, rule_id: &str) -> Result<()> {
        self.rules
            .remove(rule_id)
            .ok_or_else(|| ShaclError::Configuration(format!("Rule {} not found", rule_id)))?;
        Ok(())
    }

    /// Get a rule by ID
    pub fn get_rule(&self, rule_id: &str) -> Option<&ShaclRule> {
        self.rules.get(rule_id)
    }

    /// Get all active rules
    pub fn active_rules(&self) -> Vec<&ShaclRule> {
        self.rules.values().filter(|r| r.is_active()).collect()
    }

    /// Execute all active rules on the given store
    pub fn execute_rules(&mut self, store: &dyn Store) -> Result<Vec<RuleExecutionResult>> {
        let mut results = Vec::new();

        // Collect and sort rules by execution order
        let mut active_rules: Vec<_> = self.active_rules().into_iter().cloned().collect();
        active_rules.sort_by_key(|r| r.effective_order());

        for rule in active_rules {
            let result = self.execute_rule(&rule, store)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Execute a single rule
    #[allow(unused_variables)]
    pub fn execute_rule(
        &mut self,
        rule: &ShaclRule,
        store: &dyn Store,
    ) -> Result<RuleExecutionResult> {
        let start = std::time::Instant::now();

        let mut result = RuleExecutionResult {
            rule_id: rule.id.clone(),
            triples_added: 0,
            triples_removed: 0,
            nodes_affected: 0,
            execution_time_ms: 0,
            errors: Vec::new(),
            success: false,
        };

        // Check condition if present
        if let Some(ref condition) = rule.condition {
            if !self.evaluate_condition(condition, store)? {
                result.success = true;
                result.execution_time_ms = start.elapsed().as_millis() as u64;
                return Ok(result);
            }
        }

        // Execute based on rule type
        match rule.rule_type {
            RuleType::TripleRule => {
                self.execute_triple_rule(rule, store, &mut result)?;
            }
            RuleType::ConstructRule => {
                self.execute_construct_rule(rule, store, &mut result)?;
            }
            RuleType::SparqlRule => {
                self.execute_sparql_rule(rule, store, &mut result)?;
            }
        }

        result.execution_time_ms = start.elapsed().as_millis() as u64;
        result.success = result.errors.is_empty();

        // Update statistics
        self.stats.rules_executed += 1;
        self.stats.total_triples_added += result.triples_added;
        self.stats.total_triples_removed += result.triples_removed;
        self.stats.total_execution_time_ms += result.execution_time_ms;
        if !result.success {
            self.stats.errors += 1;
        }

        Ok(result)
    }

    /// Evaluate a condition query (stub)
    #[allow(unused_variables)]
    fn evaluate_condition(&self, condition: &str, store: &dyn Store) -> Result<bool> {
        // TODO: Execute SPARQL ASK query
        Ok(true)
    }

    /// Execute a triple rule (stub)
    #[allow(unused_variables)]
    fn execute_triple_rule(
        &self,
        rule: &ShaclRule,
        store: &dyn Store,
        result: &mut RuleExecutionResult,
    ) -> Result<()> {
        // TODO: Implement triple rule execution
        result
            .errors
            .push("Triple rule execution not yet fully implemented".to_string());
        Ok(())
    }

    /// Execute a CONSTRUCT rule (stub)
    #[allow(unused_variables)]
    fn execute_construct_rule(
        &self,
        rule: &ShaclRule,
        store: &dyn Store,
        result: &mut RuleExecutionResult,
    ) -> Result<()> {
        if let Some(ref construct_query) = rule.construct {
            tracing::debug!("Executing CONSTRUCT query: {}", construct_query);
            result
                .errors
                .push("CONSTRUCT rule execution not yet fully implemented".to_string());
        }
        Ok(())
    }

    /// Execute a generic SPARQL rule (stub)
    #[allow(unused_variables)]
    fn execute_sparql_rule(
        &self,
        rule: &ShaclRule,
        store: &dyn Store,
        result: &mut RuleExecutionResult,
    ) -> Result<()> {
        if let Some(ref sparql_query) = rule.sparql {
            tracing::debug!("Executing SPARQL query: {}", sparql_query);
            result
                .errors
                .push("SPARQL rule execution not yet fully implemented".to_string());
        }
        Ok(())
    }

    /// Execute rules iteratively until fixpoint
    pub fn execute_until_fixpoint(
        &mut self,
        store: &dyn Store,
    ) -> Result<Vec<RuleExecutionResult>> {
        let mut all_results = Vec::new();
        let mut iteration = 0;

        loop {
            if iteration >= self.max_iterations {
                return Err(ShaclError::Configuration(format!(
                    "Maximum iterations ({}) reached during rule execution",
                    self.max_iterations
                )));
            }

            let results = self.execute_rules(store)?;

            // Check if any triples were added in this iteration
            let triples_added: usize = results.iter().map(|r| r.triples_added).sum();

            all_results.extend(results);

            if triples_added == 0 {
                // Fixpoint reached
                break;
            }

            iteration += 1;
        }

        Ok(all_results)
    }

    /// Get rule engine statistics
    pub fn stats(&self) -> &RuleEngineStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = RuleEngineStats::default();
    }

    /// Set maximum iterations for fixpoint calculation
    pub fn set_max_iterations(&mut self, max_iterations: usize) {
        self.max_iterations = max_iterations;
    }

    /// Clear the query cache
    pub fn clear_cache(&mut self) {
        self.query_cache.clear();
    }
}

impl Default for RuleEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triple_rule_creation() {
        let subject =
            PropertyPath::Predicate(NamedNode::new("http://example.org/hasParent").unwrap());
        let predicate = NamedNode::new("http://example.org/hasAncestor").unwrap();
        let object =
            PropertyPath::Predicate(NamedNode::new("http://example.org/hasParent").unwrap());

        let rule = ShaclRule::triple_rule("ancestorRule".to_string(), subject, predicate, object);

        assert_eq!(rule.id, "ancestorRule");
        assert_eq!(rule.rule_type, RuleType::TripleRule);
        assert!(rule.is_active());
    }

    #[test]
    fn test_construct_rule_creation() {
        let construct_query = r#"
            CONSTRUCT {
                ?person a :Adult .
            }
            WHERE {
                ?person :age ?age .
                FILTER (?age >= 18)
            }
        "#
        .to_string();

        let rule = ShaclRule::construct_rule("adultRule".to_string(), construct_query.clone());

        assert_eq!(rule.id, "adultRule");
        assert_eq!(rule.rule_type, RuleType::ConstructRule);
        assert_eq!(rule.construct, Some(construct_query));
    }

    #[test]
    fn test_rule_engine() {
        let mut engine = RuleEngine::new();

        let rule = ShaclRule::sparql_rule(
            "testRule".to_string(),
            "SELECT * WHERE { ?s ?p ?o }".to_string(),
        );

        assert!(engine.register_rule(rule).is_ok());
        assert_eq!(engine.active_rules().len(), 1);
    }

    #[test]
    fn test_rule_ordering() {
        let mut engine = RuleEngine::new();

        let rule1 = ShaclRule::sparql_rule("rule1".to_string(), "".to_string()).with_order(10);
        let rule2 = ShaclRule::sparql_rule("rule2".to_string(), "".to_string()).with_order(5);
        let rule3 = ShaclRule::sparql_rule("rule3".to_string(), "".to_string()).with_order(15);

        engine.register_rule(rule1).unwrap();
        engine.register_rule(rule2).unwrap();
        engine.register_rule(rule3).unwrap();

        let active_rules = engine.active_rules();
        assert_eq!(active_rules.len(), 3);

        // Verify they can be sorted by order
        let mut sorted_rules = active_rules;
        sorted_rules.sort_by_key(|r| r.effective_order());
        assert_eq!(sorted_rules[0].id, "rule2");
        assert_eq!(sorted_rules[1].id, "rule1");
        assert_eq!(sorted_rules[2].id, "rule3");
    }
}
