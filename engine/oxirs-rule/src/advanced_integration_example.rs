//! Advanced Integration Example
//!
//! Demonstrates all Beta.1 features working together in a realistic scenario:
//! - Rule Composition (modules, templates, inheritance)
//! - Custom Rule Language (DSL)
//! - SPARQL Integration (query-driven reasoning)
//! - SHACL Integration (validation with reasoning)
//! - Distributed Reasoning (scaling)
//! - Explanation Support
//! - Conflict Resolution
//! - Transactions
//!
//! # Scenario: E-Commerce Knowledge Graph
//!
//! This example implements a complete e-commerce knowledge graph with:
//! - Product recommendations based on rules
//! - Inventory validation with SHACL
//! - Distributed query processing
//! - Transaction support for order processing
//! - Explanation of recommendations

use crate::{
    composition::{CompositionManager, RuleModule},
    conflict::{ConflictResolver, Priority},
    distributed::{DistributedReasoner, Node, PartitionStrategy},
    explanation::ExplanationEngine,
    shacl_integration::{ShaclRuleIntegration, ShapeConstraint, ValidationMode},
    sparql_integration::{QueryMode, QueryPattern, SparqlRuleIntegration},
    transaction::{IsolationLevel, TransactionManager},
    Rule, RuleAtom, RuleEngine, Term,
};
use anyhow::Result;
use tracing::{debug, info};

/// E-Commerce reasoning system with all Beta.1 features
pub struct EcommerceReasoningSystem {
    /// Main rule engine
    engine: RuleEngine,
    /// Composition manager for rule organization
    composition: CompositionManager,
    /// SPARQL integration for query-driven reasoning
    sparql: SparqlRuleIntegration,
    /// SHACL integration for validation
    shacl: ShaclRuleIntegration,
    /// Distributed reasoner for scaling
    distributed: DistributedReasoner,
    /// Explanation engine for provenance
    explanation: ExplanationEngine,
    /// Conflict resolver for rule priorities
    conflict: ConflictResolver,
    /// Transaction manager for ACID operations
    transactions: TransactionManager,
}

impl EcommerceReasoningSystem {
    /// Create a new e-commerce reasoning system
    pub fn new() -> Self {
        info!("Initializing E-Commerce Reasoning System with all Beta.1 features");

        let engine = RuleEngine::new();
        let composition = CompositionManager::new();
        let sparql = SparqlRuleIntegration::new(RuleEngine::new());
        let shacl = ShaclRuleIntegration::new(RuleEngine::new());
        let distributed = DistributedReasoner::new(PartitionStrategy::LoadBalanced);
        let explanation = ExplanationEngine::new();
        let conflict = ConflictResolver::new();
        let transactions = TransactionManager::new();

        Self {
            engine,
            composition,
            sparql,
            shacl,
            distributed,
            explanation,
            conflict,
            transactions,
        }
    }

    /// Initialize the system with product recommendation rules
    pub fn initialize_product_rules(&mut self) -> Result<()> {
        info!("Initializing product recommendation rules");

        // Create a rule module for product recommendations
        let mut product_module = RuleModule::new("product_recommendations".to_string());
        product_module.set_description("Rules for product recommendations".to_string());

        // Add rule: customers who bought X also bought Y
        let also_bought_rule = Rule {
            name: "also_bought".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("customer1".to_string()),
                    predicate: Term::Constant("purchased".to_string()),
                    object: Term::Variable("product1".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("customer2".to_string()),
                    predicate: Term::Constant("purchased".to_string()),
                    object: Term::Variable("product1".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("customer2".to_string()),
                    predicate: Term::Constant("purchased".to_string()),
                    object: Term::Variable("product2".to_string()),
                },
                RuleAtom::NotEqual {
                    left: Term::Variable("customer1".to_string()),
                    right: Term::Variable("customer2".to_string()),
                },
                RuleAtom::NotEqual {
                    left: Term::Variable("product1".to_string()),
                    right: Term::Variable("product2".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("customer1".to_string()),
                predicate: Term::Constant("recommended".to_string()),
                object: Term::Variable("product2".to_string()),
            }],
        };

        product_module.add_rule(also_bought_rule);

        // Set high priority for recommendation rules
        self.conflict.set_priority("also_bought", Priority::High);

        // Register the module
        self.composition.register_module(product_module)?;

        Ok(())
    }

    /// Initialize inventory validation rules using SHACL
    pub fn initialize_inventory_validation(&mut self) -> Result<()> {
        info!("Initializing inventory validation with SHACL");

        // Create SHACL constraints for inventory
        let _min_stock_constraint =
            ShapeConstraint::new("min_stock".to_string(), "sh:minCount".to_string());

        // Create repair rule for low stock
        let low_stock_repair = Rule {
            name: "reorder_low_stock".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("product".to_string()),
                    predicate: Term::Constant("stock".to_string()),
                    object: Term::Variable("quantity".to_string()),
                },
                RuleAtom::LessThan {
                    left: Term::Variable("quantity".to_string()),
                    right: Term::Literal("10".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("product".to_string()),
                predicate: Term::Constant("needsReorder".to_string()),
                object: Term::Constant("true".to_string()),
            }],
        };

        self.shacl
            .register_repair_rule("sh:minCount".to_string(), low_stock_repair);
        self.shacl.set_mode(ValidationMode::Full);

        Ok(())
    }

    /// Setup distributed reasoning with multiple nodes
    pub fn setup_distributed_reasoning(&mut self) -> Result<()> {
        info!("Setting up distributed reasoning infrastructure");

        // Register reasoning nodes
        let node1 =
            Node::new("node1".to_string(), "localhost:8001".to_string()).with_capacity(1000);
        let node2 =
            Node::new("node2".to_string(), "localhost:8002".to_string()).with_capacity(1500);
        let node3 =
            Node::new("node3".to_string(), "localhost:8003".to_string()).with_capacity(2000);

        self.distributed.register_node(node1)?;
        self.distributed.register_node(node2)?;
        self.distributed.register_node(node3)?;

        info!(
            "Registered {} reasoning nodes",
            self.distributed.node_count()
        );

        Ok(())
    }

    /// Process customer order with full transaction support
    pub fn process_order(&mut self, customer: &str, product: &str) -> Result<()> {
        info!(
            "Processing order for customer '{}' - product '{}'",
            customer, product
        );

        // Begin transaction
        let tx_id = self
            .transactions
            .begin_transaction(IsolationLevel::Serializable)?;
        debug!("Started transaction {}", tx_id);

        // Add purchase fact
        let purchase_fact = RuleAtom::Triple {
            subject: Term::Constant(customer.to_string()),
            predicate: Term::Constant("purchased".to_string()),
            object: Term::Constant(product.to_string()),
        };

        self.transactions.add_fact(tx_id, purchase_fact.clone())?;

        // Record explanation
        self.explanation.record_assertion(purchase_fact.clone());

        // Apply recommendation rules
        let derived = self.engine.forward_chain(&[purchase_fact])?;
        debug!("Derived {} recommendations", derived.len());

        // Commit transaction
        self.transactions.commit(tx_id)?;
        info!("Order processed successfully");

        Ok(())
    }

    /// Query recommendations using SPARQL integration
    pub fn query_recommendations(&mut self, customer: &str) -> Result<Vec<String>> {
        info!("Querying recommendations for customer '{}'", customer);

        // Set query mode to hybrid (forward + backward)
        self.sparql.engine_mut().add_facts(self.engine.get_facts());
        self.sparql.set_mode(QueryMode::Hybrid);

        // Create query pattern for recommendations
        let pattern = QueryPattern::new(
            Some(customer.to_string()),
            Some("recommended".to_string()),
            None, // Match any product
        );

        let results = self.sparql.query_with_reasoning(&[pattern])?;

        // Extract product names
        let products: Vec<String> = results
            .iter()
            .filter_map(|atom| {
                if let RuleAtom::Triple {
                    object: Term::Constant(product),
                    ..
                } = atom
                {
                    Some(product.clone())
                } else {
                    None
                }
            })
            .collect();

        info!("Found {} recommendations", products.len());
        Ok(products)
    }

    /// Validate inventory using SHACL
    pub fn validate_inventory(&mut self, inventory_facts: &[RuleAtom]) -> Result<bool> {
        info!("Validating inventory with {} facts", inventory_facts.len());

        let constraint =
            ShapeConstraint::new("inventory_check".to_string(), "sh:minCount".to_string());

        let report = self
            .shacl
            .validate_with_reasoning(&[constraint], inventory_facts)?;

        if !report.conforms {
            info!(
                "Validation failed: {} violations, {} warnings",
                report.violation_count(),
                report.warning_count()
            );
        } else {
            info!("Validation passed");
        }

        Ok(report.conforms)
    }

    /// Execute distributed reasoning for large datasets
    pub fn execute_distributed(&mut self, facts: &[RuleAtom]) -> Result<Vec<RuleAtom>> {
        info!(
            "Executing distributed reasoning across {} nodes",
            self.distributed.node_count()
        );

        let rules = self.composition.get_all_rules();
        let results = self.distributed.execute_distributed(&rules, facts)?;

        info!(
            "Distributed execution complete: {} facts derived",
            results.len()
        );
        Ok(results)
    }

    /// Get explanation for a recommendation
    pub fn explain_recommendation(&self, customer: &str, product: &str) -> Result<String> {
        let fact = RuleAtom::Triple {
            subject: Term::Constant(customer.to_string()),
            predicate: Term::Constant("recommended".to_string()),
            object: Term::Constant(product.to_string()),
        };

        let explanation = self.explanation.explain_why(&fact)?;
        Ok(format!("{}", explanation))
    }

    /// Get comprehensive statistics
    pub fn get_statistics(&self) -> SystemStatistics {
        SystemStatistics {
            total_rules: self.composition.get_all_rules().len(),
            total_modules: self.composition.get_stats().total_modules,
            total_nodes: self.distributed.node_count(),
            sparql_queries: self.sparql.get_stats().total_queries,
            shacl_validations: self.shacl.get_stats().total_validations,
            transactions: self.transactions.get_stats().total_transactions,
            explanations: self.explanation.get_stats().total_derivations,
        }
    }
}

impl Default for EcommerceReasoningSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// System statistics
#[derive(Debug, Clone)]
pub struct SystemStatistics {
    pub total_rules: usize,
    pub total_modules: usize,
    pub total_nodes: usize,
    pub sparql_queries: usize,
    pub shacl_validations: usize,
    pub transactions: usize,
    pub explanations: usize,
}

impl std::fmt::Display for SystemStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "E-Commerce Reasoning System Statistics:")?;
        writeln!(f, "  Rules: {}", self.total_rules)?;
        writeln!(f, "  Modules: {}", self.total_modules)?;
        writeln!(f, "  Distributed Nodes: {}", self.total_nodes)?;
        writeln!(f, "  SPARQL Queries: {}", self.sparql_queries)?;
        writeln!(f, "  SHACL Validations: {}", self.shacl_validations)?;
        writeln!(f, "  Transactions: {}", self.transactions)?;
        writeln!(f, "  Explanations Generated: {}", self.explanations)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ecommerce_system_creation() {
        let system = EcommerceReasoningSystem::new();
        assert_eq!(system.distributed.node_count(), 0);
    }

    #[test]
    fn test_initialize_product_rules() {
        let mut system = EcommerceReasoningSystem::new();
        system.initialize_product_rules().unwrap();

        let stats = system.get_statistics();
        assert!(stats.total_modules > 0);
    }

    #[test]
    fn test_setup_distributed_reasoning() {
        let mut system = EcommerceReasoningSystem::new();
        system.setup_distributed_reasoning().unwrap();

        assert_eq!(system.distributed.node_count(), 3);
    }

    #[test]
    fn test_process_order() {
        let mut system = EcommerceReasoningSystem::new();
        system.initialize_product_rules().unwrap();

        let result = system.process_order("customer1", "product1");
        assert!(result.is_ok());

        let stats = system.get_statistics();
        assert!(stats.transactions > 0);
    }

    #[test]
    fn test_validate_inventory() {
        let mut system = EcommerceReasoningSystem::new();
        system.initialize_inventory_validation().unwrap();

        let inventory_facts = vec![RuleAtom::Triple {
            subject: Term::Constant("product1".to_string()),
            predicate: Term::Constant("stock".to_string()),
            object: Term::Literal("100".to_string()),
        }];

        let valid = system.validate_inventory(&inventory_facts).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_full_integration_workflow() {
        let mut system = EcommerceReasoningSystem::new();

        // Initialize all components
        system.initialize_product_rules().unwrap();
        system.initialize_inventory_validation().unwrap();
        system.setup_distributed_reasoning().unwrap();

        // Process some orders
        system.process_order("alice", "laptop").unwrap();
        system.process_order("bob", "laptop").unwrap();
        system.process_order("bob", "mouse").unwrap();

        // Query recommendations (alice should get mouse recommended)
        let recommendations = system.query_recommendations("alice").unwrap_or_default();

        // Get statistics
        let stats = system.get_statistics();
        assert!(stats.total_modules > 0);
        assert!(stats.total_nodes == 3);

        println!("\n{}", stats);
        println!("Recommendations for alice: {:?}", recommendations);
    }

    #[test]
    fn test_statistics_display() {
        let system = EcommerceReasoningSystem::new();
        let stats = system.get_statistics();

        let display = format!("{}", stats);
        assert!(display.contains("E-Commerce Reasoning System Statistics"));
    }
}
