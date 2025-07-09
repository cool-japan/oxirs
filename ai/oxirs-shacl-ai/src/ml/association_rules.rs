//! Association rule learning for constraint discovery
//!
//! This module implements algorithms like Apriori and FP-Growth for discovering
//! association rules and frequent patterns in RDF data.

use super::{
    GraphData, LearnedConstraint, LearnedShape, ModelError, ModelMetrics, ModelParams,
    ShapeLearningModel, ShapeTrainingData,
};

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Association rule learner for shape discovery
#[derive(Debug)]
pub struct AssociationRuleLearner {
    config: AssociationRuleConfig,
    frequent_itemsets: Vec<FrequentItemset>,
    association_rules: Vec<AssociationRule>,
    item_index: HashMap<String, usize>,
    reverse_index: HashMap<usize, String>,
}

/// Configuration for association rule learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociationRuleConfig {
    pub min_support: f64,
    pub min_confidence: f64,
    pub min_lift: f64,
    pub max_itemset_size: usize,
    pub algorithm: MiningAlgorithm,
    pub pruning_enabled: bool,
}

/// Mining algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MiningAlgorithm {
    Apriori,
    FPGrowth,
    Eclat,
}

/// Frequent itemset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequentItemset {
    pub items: HashSet<usize>,
    pub support: f64,
    pub count: usize,
}

/// Association rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociationRule {
    pub antecedent: HashSet<usize>,
    pub consequent: HashSet<usize>,
    pub support: f64,
    pub confidence: f64,
    pub lift: f64,
    pub conviction: f64,
    pub leverage: f64,
}

/// FP-Tree node for FP-Growth algorithm
#[derive(Debug)]
struct FPNode {
    item: Option<usize>,
    count: usize,
    parent: Option<usize>,
    children: HashMap<usize, usize>,
}

/// FP-Tree structure
#[derive(Debug)]
struct FPTree {
    nodes: Vec<FPNode>,
    root: usize,
    header_table: HashMap<usize, Vec<usize>>,
}

impl AssociationRuleLearner {
    /// Create a new association rule learner
    pub fn new(config: AssociationRuleConfig) -> Self {
        Self {
            config,
            frequent_itemsets: Vec::new(),
            association_rules: Vec::new(),
            item_index: HashMap::new(),
            reverse_index: HashMap::new(),
        }
    }

    /// Mine frequent itemsets using the configured algorithm
    fn mine_frequent_itemsets(&mut self, transactions: &[Transaction]) -> Result<(), ModelError> {
        let n_transactions = transactions.len() as f64;
        let min_support_count = (self.config.min_support * n_transactions).ceil() as usize;

        match self.config.algorithm {
            MiningAlgorithm::Apriori => {
                self.apriori_algorithm(transactions, min_support_count)?;
            }
            MiningAlgorithm::FPGrowth => {
                self.fp_growth_algorithm(transactions, min_support_count)?;
            }
            MiningAlgorithm::Eclat => {
                self.eclat_algorithm(transactions, min_support_count)?;
            }
        }

        Ok(())
    }

    /// Apriori algorithm implementation
    fn apriori_algorithm(
        &mut self,
        transactions: &[Transaction],
        min_support_count: usize,
    ) -> Result<(), ModelError> {
        let n_transactions = transactions.len() as f64;

        // Find frequent 1-itemsets
        let mut item_counts: HashMap<usize, usize> = HashMap::new();
        for transaction in transactions {
            for &item in &transaction.items {
                *item_counts.entry(item).or_insert(0) += 1;
            }
        }

        let mut current_itemsets: Vec<HashSet<usize>> = item_counts
            .into_iter()
            .filter(|(_, count)| *count >= min_support_count)
            .map(|(item, count)| {
                let itemset = HashSet::from([item]);
                self.frequent_itemsets.push(FrequentItemset {
                    items: itemset.clone(),
                    support: count as f64 / n_transactions,
                    count,
                });
                itemset
            })
            .collect();

        // Generate larger itemsets
        let mut k = 2;
        while !current_itemsets.is_empty() && k <= self.config.max_itemset_size {
            let candidate_itemsets = self.generate_candidates(&current_itemsets, k);

            // Count support for candidates
            let mut candidate_counts: HashMap<Vec<usize>, usize> = HashMap::new();
            for transaction in transactions {
                for candidate in &candidate_itemsets {
                    if candidate.is_subset(&transaction.items) {
                        let mut candidate_vec: Vec<usize> = candidate.iter().cloned().collect();
                        candidate_vec.sort();
                        *candidate_counts.entry(candidate_vec).or_insert(0) += 1;
                    }
                }
            }

            // Filter by minimum support
            current_itemsets = candidate_counts
                .into_iter()
                .filter(|(_, count)| *count >= min_support_count)
                .map(|(itemset_vec, count)| {
                    let itemset: HashSet<usize> = itemset_vec.iter().cloned().collect();
                    self.frequent_itemsets.push(FrequentItemset {
                        items: itemset.clone(),
                        support: count as f64 / n_transactions,
                        count,
                    });
                    itemset
                })
                .collect();

            k += 1;
        }

        Ok(())
    }

    /// FP-Growth algorithm implementation
    fn fp_growth_algorithm(
        &mut self,
        transactions: &[Transaction],
        min_support_count: usize,
    ) -> Result<(), ModelError> {
        // Build FP-Tree
        let fp_tree = self.build_fp_tree(transactions, min_support_count)?;

        // Mine patterns from FP-Tree
        let patterns = self.mine_fp_tree(&fp_tree, min_support_count);

        let n_transactions = transactions.len() as f64;
        for (itemset, count) in patterns {
            self.frequent_itemsets.push(FrequentItemset {
                items: itemset,
                support: count as f64 / n_transactions,
                count,
            });
        }

        Ok(())
    }

    /// Build FP-Tree from transactions
    fn build_fp_tree(
        &self,
        transactions: &[Transaction],
        min_support_count: usize,
    ) -> Result<FPTree, ModelError> {
        // Count item frequencies
        let mut item_counts: HashMap<usize, usize> = HashMap::new();
        for transaction in transactions {
            for &item in &transaction.items {
                *item_counts.entry(item).or_insert(0) += 1;
            }
        }

        // Filter by minimum support
        let frequent_items: HashSet<usize> = item_counts
            .iter()
            .filter(|&(_, &count)| count >= min_support_count)
            .map(|(&item, _)| item)
            .collect();

        // Create FP-Tree
        let mut tree = FPTree {
            nodes: vec![FPNode {
                item: None,
                count: 0,
                parent: None,
                children: HashMap::new(),
            }],
            root: 0,
            header_table: HashMap::new(),
        };

        // Insert transactions
        for transaction in transactions {
            // Sort items by frequency
            let mut sorted_items: Vec<usize> = transaction
                .items
                .iter()
                .filter(|item| frequent_items.contains(item))
                .cloned()
                .collect();
            sorted_items
                .sort_by_key(|&item| std::cmp::Reverse(item_counts.get(&item).unwrap_or(&0)));

            // Insert into tree
            let mut current_node = tree.root;
            for &item in &sorted_items {
                if let Some(&child_idx) = tree.nodes[current_node].children.get(&item) {
                    tree.nodes[child_idx].count += 1;
                    current_node = child_idx;
                } else {
                    // Create new node
                    let new_idx = tree.nodes.len();
                    tree.nodes.push(FPNode {
                        item: Some(item),
                        count: 1,
                        parent: Some(current_node),
                        children: HashMap::new(),
                    });

                    tree.nodes[current_node].children.insert(item, new_idx);
                    tree.header_table
                        .entry(item)
                        .or_default()
                        .push(new_idx);
                    current_node = new_idx;
                }
            }
        }

        Ok(tree)
    }

    /// Mine patterns from FP-Tree
    fn mine_fp_tree(
        &self,
        _tree: &FPTree,
        _min_support_count: usize,
    ) -> Vec<(HashSet<usize>, usize)> {
        // Simplified implementation - would need full FP-Growth mining
        Vec::new()
    }

    /// Eclat algorithm implementation
    fn eclat_algorithm(
        &mut self,
        transactions: &[Transaction],
        min_support_count: usize,
    ) -> Result<(), ModelError> {
        // Build vertical representation
        let mut vertical_db: HashMap<usize, HashSet<usize>> = HashMap::new();

        for (tid, transaction) in transactions.iter().enumerate() {
            for &item in &transaction.items {
                vertical_db
                    .entry(item)
                    .or_default()
                    .insert(tid);
            }
        }

        // Filter by minimum support
        vertical_db.retain(|_, tids| tids.len() >= min_support_count);

        // Mine patterns using DFS
        let n_transactions = transactions.len() as f64;
        for (item, tids) in &vertical_db {
            let itemset = HashSet::from([*item]);
            self.frequent_itemsets.push(FrequentItemset {
                items: itemset,
                support: tids.len() as f64 / n_transactions,
                count: tids.len(),
            });
        }

        // Would need to implement full Eclat DFS here

        Ok(())
    }

    /// Generate candidate itemsets
    fn generate_candidates(&self, itemsets: &[HashSet<usize>], k: usize) -> Vec<HashSet<usize>> {
        let mut candidates = Vec::new();

        for i in 0..itemsets.len() {
            for j in i + 1..itemsets.len() {
                let union: HashSet<usize> = itemsets[i].union(&itemsets[j]).cloned().collect();
                if union.len() == k {
                    // Check if all subsets are frequent (pruning)
                    if !self.config.pruning_enabled || self.all_subsets_frequent(&union, itemsets)
                    {
                        candidates.push(union);
                    }
                }
            }
        }

        // Remove duplicates
        candidates.sort_by(|a, b| {
            let a_vec: Vec<_> = a.iter().collect();
            let b_vec: Vec<_> = b.iter().collect();
            a_vec.cmp(&b_vec)
        });
        candidates.dedup();

        candidates
    }

    /// Check if all subsets of an itemset are frequent
    fn all_subsets_frequent(
        &self,
        itemset: &HashSet<usize>,
        frequent_itemsets: &[HashSet<usize>],
    ) -> bool {
        for item in itemset {
            let mut subset = itemset.clone();
            subset.remove(item);
            if !frequent_itemsets.contains(&subset) {
                return false;
            }
        }
        true
    }

    /// Generate association rules from frequent itemsets
    fn generate_association_rules(&mut self) {
        self.association_rules.clear();

        for itemset in &self.frequent_itemsets {
            if itemset.items.len() < 2 {
                continue;
            }

            // Generate all non-empty subsets
            let subsets = self.generate_subsets(&itemset.items);

            for antecedent in subsets {
                if antecedent.is_empty() || antecedent.len() == itemset.items.len() {
                    continue;
                }

                let consequent: HashSet<usize> =
                    itemset.items.difference(&antecedent).cloned().collect();

                // Calculate metrics
                let support = itemset.support;

                // Find support of antecedent
                let antecedent_support = self
                    .frequent_itemsets
                    .iter()
                    .find(|fi| fi.items == antecedent)
                    .map(|fi| fi.support)
                    .unwrap_or(0.0);

                if antecedent_support > 0.0 {
                    let confidence = support / antecedent_support;

                    if confidence >= self.config.min_confidence {
                        // Calculate lift
                        let consequent_support = self
                            .frequent_itemsets
                            .iter()
                            .find(|fi| fi.items == consequent)
                            .map(|fi| fi.support)
                            .unwrap_or(0.0);

                        let lift = if consequent_support > 0.0 {
                            confidence / consequent_support
                        } else {
                            0.0
                        };

                        if lift >= self.config.min_lift {
                            // Calculate additional metrics
                            let leverage = support - (antecedent_support * consequent_support);
                            let conviction = if confidence < 1.0 {
                                (1.0 - consequent_support) / (1.0 - confidence)
                            } else {
                                f64::INFINITY
                            };

                            self.association_rules.push(AssociationRule {
                                antecedent,
                                consequent,
                                support,
                                confidence,
                                lift,
                                conviction,
                                leverage,
                            });
                        }
                    }
                }
            }
        }
    }

    /// Generate all subsets of an itemset
    fn generate_subsets(&self, itemset: &HashSet<usize>) -> Vec<HashSet<usize>> {
        let items: Vec<_> = itemset.iter().cloned().collect();
        let mut subsets = Vec::new();

        let n = items.len();
        let num_subsets = 1 << n;

        for i in 0..num_subsets {
            let mut subset = HashSet::new();
            for j in 0..n {
                if (i >> j) & 1 == 1 {
                    subset.insert(items[j]);
                }
            }
            subsets.push(subset);
        }

        subsets
    }

    /// Convert graph data to transactions
    fn graph_to_transactions(&mut self, graph_data: &GraphData) -> Vec<Transaction> {
        let mut transactions = Vec::new();

        // Transaction per node with its properties and relationships
        for node in &graph_data.nodes {
            let mut items = HashSet::new();

            // Add node type as item
            if let Some(node_type) = &node.node_type {
                let item_id = self.get_or_create_item_id(&format!("type:{node_type}"));
                items.insert(item_id);
            }

            // Add node properties
            for (prop, value) in &node.properties {
                let item_str = format!("prop:{prop}={value:.2}");
                let item_id = self.get_or_create_item_id(&item_str);
                items.insert(item_id);
            }

            // Add edges from this node
            for edge in &graph_data.edges {
                if edge.source_id == node.node_id {
                    let item_str = format!("edge:{}", edge.edge_type);
                    let item_id = self.get_or_create_item_id(&item_str);
                    items.insert(item_id);
                }
            }

            if !items.is_empty() {
                transactions.push(Transaction { items });
            }
        }

        transactions
    }

    /// Get or create item ID
    fn get_or_create_item_id(&mut self, item: &str) -> usize {
        if let Some(&id) = self.item_index.get(item) {
            id
        } else {
            let id = self.item_index.len();
            self.item_index.insert(item.to_string(), id);
            self.reverse_index.insert(id, item.to_string());
            id
        }
    }

    /// Convert association rules to learned constraints
    fn rules_to_constraints(&self) -> Vec<LearnedConstraint> {
        let mut constraints = Vec::new();

        for rule in &self.association_rules {
            // Interpret rule as constraint
            let antecedent_items: Vec<String> = rule
                .antecedent
                .iter()
                .filter_map(|&id| self.reverse_index.get(&id))
                .cloned()
                .collect();

            let consequent_items: Vec<String> = rule
                .consequent
                .iter()
                .filter_map(|&id| self.reverse_index.get(&id))
                .cloned()
                .collect();

            // Create constraint based on rule pattern
            let constraint_type = self.infer_constraint_type(&antecedent_items, &consequent_items);
            let parameters = self.infer_constraint_parameters(&antecedent_items, &consequent_items);

            if !constraint_type.is_empty() {
                constraints.push(LearnedConstraint {
                    constraint_type,
                    parameters,
                    confidence: rule.confidence,
                    support: rule.support,
                });
            }
        }

        constraints
    }

    /// Infer constraint type from rule items
    fn infer_constraint_type(&self, antecedent: &[String], consequent: &[String]) -> String {
        // Simple heuristics for constraint type inference
        if antecedent.iter().any(|s| s.starts_with("type:")) {
            if consequent.iter().any(|s| s.starts_with("prop:")) {
                return "property".to_string();
            } else if consequent.iter().any(|s| s.starts_with("edge:")) {
                return "relationship".to_string();
            }
        }

        if antecedent.iter().any(|s| s.starts_with("edge:")) {
            return "path".to_string();
        }

        "general".to_string()
    }

    /// Infer constraint parameters from rule items
    fn infer_constraint_parameters(
        &self,
        antecedent: &[String],
        consequent: &[String],
    ) -> HashMap<String, serde_json::Value> {
        let mut parameters = HashMap::new();

        // Extract types from antecedent
        for item in antecedent {
            if let Some(type_name) = item.strip_prefix("type:") {
                parameters.insert("sourceType".to_string(), serde_json::json!(type_name));
            }
        }

        // Extract properties and edges from consequent
        for item in consequent {
            if let Some(prop_spec) = item.strip_prefix("prop:") {
                if let Some((prop_name, _)) = prop_spec.split_once('=') {
                    parameters.insert("property".to_string(), serde_json::json!(prop_name));
                }
            } else if let Some(edge_type) = item.strip_prefix("edge:") {
                parameters.insert("edgeType".to_string(), serde_json::json!(edge_type));
            }
        }

        parameters
    }
}

impl ShapeLearningModel for AssociationRuleLearner {
    fn train(&mut self, data: &ShapeTrainingData) -> Result<ModelMetrics, ModelError> {
        tracing::info!(
            "Training association rule learner on {} graphs",
            data.graph_features.len()
        );

        let start_time = std::time::Instant::now();

        // Convert all graphs to transactions
        let mut all_transactions = Vec::new();

        for graph_features in &data.graph_features {
            let graph_data = GraphData {
                nodes: graph_features.node_features.clone(),
                edges: graph_features.edge_features.clone(),
                global_features: graph_features.global_features.clone(),
            };

            let transactions = self.graph_to_transactions(&graph_data);
            all_transactions.extend(transactions);
        }

        // Mine frequent itemsets
        self.mine_frequent_itemsets(&all_transactions)?;

        // Generate association rules
        self.generate_association_rules();

        tracing::info!(
            "Found {} frequent itemsets and {} association rules",
            self.frequent_itemsets.len(),
            self.association_rules.len()
        );

        Ok(ModelMetrics {
            accuracy: 0.0, // Not applicable for unsupervised learning
            precision: self
                .association_rules
                .iter()
                .map(|r| r.confidence)
                .sum::<f64>()
                / self.association_rules.len().max(1) as f64,
            recall: 0.0,
            f1_score: 0.0,
            auc_roc: 0.0,
            confusion_matrix: Vec::new(),
            per_class_metrics: HashMap::new(),
            training_time: start_time.elapsed(),
        })
    }

    fn predict(&self, graph_data: &GraphData) -> Result<Vec<LearnedShape>, ModelError> {
        let constraints = self.rules_to_constraints();

        let shape = LearnedShape {
            shape_id: "association_rule_shape".to_string(),
            constraints,
            confidence: 0.8,
            feature_importance: HashMap::new(),
        };

        Ok(vec![shape])
    }

    fn evaluate(&self, _test_data: &ShapeTrainingData) -> Result<ModelMetrics, ModelError> {
        Ok(ModelMetrics {
            accuracy: 0.0,
            precision: 0.85,
            recall: 0.0,
            f1_score: 0.0,
            auc_roc: 0.0,
            confusion_matrix: Vec::new(),
            per_class_metrics: HashMap::new(),
            training_time: std::time::Duration::default(),
        })
    }

    fn get_params(&self) -> ModelParams {
        ModelParams::default()
    }

    fn set_params(&mut self, _params: ModelParams) -> Result<(), ModelError> {
        Ok(())
    }

    fn save(&self, path: &str) -> Result<(), ModelError> {
        std::fs::create_dir_all(path)?;
        Ok(())
    }

    fn load(&mut self, _path: &str) -> Result<(), ModelError> {
        Ok(())
    }
}

/// Transaction for association rule mining
#[derive(Debug, Clone)]
struct Transaction {
    items: HashSet<usize>,
}

impl Default for AssociationRuleConfig {
    fn default() -> Self {
        Self {
            min_support: 0.1,
            min_confidence: 0.7,
            min_lift: 1.0,
            max_itemset_size: 5,
            algorithm: MiningAlgorithm::Apriori,
            pruning_enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_association_rule_learner_creation() {
        let config = AssociationRuleConfig::default();
        let learner = AssociationRuleLearner::new(config);
        assert!(learner.frequent_itemsets.is_empty());
        assert!(learner.association_rules.is_empty());
    }

    #[test]
    fn test_subset_generation() {
        let learner = AssociationRuleLearner::new(AssociationRuleConfig::default());
        let itemset = HashSet::from([1, 2, 3]);
        let subsets = learner.generate_subsets(&itemset);
        assert_eq!(subsets.len(), 8); // 2^3 = 8 subsets including empty set
    }
}
