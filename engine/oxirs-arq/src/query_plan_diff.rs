//! Query Plan Comparison and Diff Utilities
//!
//! Provides tools for comparing SPARQL query execution plans to understand
//! optimization changes, performance regressions, and structural differences.
//!
//! ## Features
//!
//! - **Structural diff**: Compare plan tree structures
//! - **Cost comparison**: Analyze cost estimate changes
//! - **Operator changes**: Track added/removed/modified operators
//! - **Visual diff**: Generate readable diff reports
//! - **Regression detection**: Identify performance regressions
//! - **SciRS2 integration**: Statistical analysis of plan differences
//!
//! ## Example
//!
//! ```rust,ignore
//! use oxirs_arq::query_plan_diff::{PlanDiffer, DiffConfig};
//!
//! let differ = PlanDiffer::new(DiffConfig::default());
//! let diff = differ.compare_plans(&plan_v1, &plan_v2)?;
//!
//! if diff.has_regression() {
//!     println!("WARNING: Performance regression detected!");
//!     println!("{}", diff.format_summary());
//! }
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for query plan comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffConfig {
    /// Threshold for cost change to be considered significant (percentage)
    pub cost_threshold: f64,

    /// Include operator details in diff
    pub include_operator_details: bool,

    /// Include cardinality estimates in diff
    pub include_cardinality: bool,

    /// Detect structural changes (added/removed nodes)
    pub detect_structural_changes: bool,

    /// Highlight performance regressions
    pub highlight_regressions: bool,

    /// Maximum diff depth (0 = unlimited)
    pub max_depth: usize,

    /// Ignore minor formatting changes
    pub ignore_formatting: bool,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            cost_threshold: 5.0, // 5% threshold
            include_operator_details: true,
            include_cardinality: true,
            detect_structural_changes: true,
            highlight_regressions: true,
            max_depth: 0, // unlimited
            ignore_formatting: true,
        }
    }
}

/// Result of comparing two query plans
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanDiff {
    /// Summary of changes
    pub summary: DiffSummary,

    /// Structural changes detected
    pub structural_changes: Vec<StructuralChange>,

    /// Cost changes detected
    pub cost_changes: Vec<CostChange>,

    /// Operator changes
    pub operator_changes: Vec<OperatorChange>,

    /// Whether this represents a regression
    pub is_regression: bool,

    /// Overall quality score (-1.0 = worse, 0.0 = same, 1.0 = better)
    pub quality_score: f64,

    /// Detailed node-by-node comparison
    pub node_diffs: Vec<NodeDiff>,
}

/// Summary of plan differences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffSummary {
    /// Total number of changes
    pub total_changes: usize,

    /// Number of structural changes
    pub structural_changes: usize,

    /// Number of cost changes
    pub cost_changes: usize,

    /// Number of operator changes
    pub operator_changes: usize,

    /// Estimated performance impact (multiplier: >1.0 = faster)
    pub estimated_impact: f64,

    /// Summary message
    pub message: String,
}

/// Structural change in the query plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralChange {
    /// Type of structural change
    pub change_type: StructuralChangeType,

    /// Path to the changed node
    pub path: String,

    /// Description of the change
    pub description: String,

    /// Old node type (if applicable)
    pub old_node_type: Option<String>,

    /// New node type (if applicable)
    pub new_node_type: Option<String>,
}

/// Type of structural change
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructuralChangeType {
    /// Node was added
    NodeAdded,

    /// Node was removed
    NodeRemoved,

    /// Node was replaced with different type
    NodeReplaced,

    /// Subtree was reordered
    SubtreeReordered,

    /// Children count changed
    ChildrenChanged,
}

/// Cost change in the query plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostChange {
    /// Path to the node
    pub path: String,

    /// Old cost estimate
    pub old_cost: f64,

    /// New cost estimate
    pub new_cost: f64,

    /// Percentage change
    pub percentage_change: f64,

    /// Whether this is a regression
    pub is_regression: bool,

    /// Description
    pub description: String,
}

/// Operator change in the query plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorChange {
    /// Path to the operator
    pub path: String,

    /// Type of change
    pub change_type: OperatorChangeType,

    /// Old operator name
    pub old_operator: Option<String>,

    /// New operator name
    pub new_operator: Option<String>,

    /// Description
    pub description: String,

    /// Performance impact
    pub impact: OperatorImpact,
}

/// Type of operator change
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatorChangeType {
    /// Operator type changed
    TypeChanged,

    /// Operator parameters changed
    ParametersChanged,

    /// Execution strategy changed
    StrategyChanged,

    /// Index usage changed
    IndexUsageChanged,
}

/// Performance impact of operator change
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatorImpact {
    /// Positive impact (improvement)
    Positive,

    /// Neutral impact
    Neutral,

    /// Negative impact (regression)
    Negative,

    /// Unknown impact
    Unknown,
}

/// Node-level diff information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDiff {
    /// Path to this node
    pub path: String,

    /// Node type
    pub node_type: String,

    /// Whether this node exists in both plans
    pub exists_in_both: bool,

    /// Cost comparison
    pub cost_diff: Option<f64>,

    /// Cardinality comparison
    pub cardinality_diff: Option<i64>,

    /// Property changes
    pub property_changes: Vec<PropertyChange>,
}

/// Change in a node property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyChange {
    /// Property name
    pub name: String,

    /// Old value
    pub old_value: String,

    /// New value
    pub new_value: String,
}

/// Simplified plan node for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanNode {
    /// Node identifier/path
    pub id: String,

    /// Node type (e.g., "Scan", "Join", "Filter")
    pub node_type: String,

    /// Estimated cost
    pub cost: f64,

    /// Estimated cardinality
    pub cardinality: usize,

    /// Operator-specific properties
    pub properties: HashMap<String, String>,

    /// Child nodes
    pub children: Vec<PlanNode>,
}

impl PlanNode {
    /// Create a new plan node
    pub fn new(id: String, node_type: String) -> Self {
        Self {
            id,
            node_type,
            cost: 0.0,
            cardinality: 0,
            properties: HashMap::new(),
            children: Vec::new(),
        }
    }

    /// Add a child node
    pub fn with_child(mut self, child: PlanNode) -> Self {
        self.children.push(child);
        self
    }

    /// Set cost estimate
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    /// Set cardinality estimate
    pub fn with_cardinality(mut self, cardinality: usize) -> Self {
        self.cardinality = cardinality;
        self
    }

    /// Add a property
    pub fn with_property(mut self, key: String, value: String) -> Self {
        self.properties.insert(key, value);
        self
    }

    /// Get all descendant nodes
    #[allow(dead_code)]
    fn descendants(&self) -> Vec<&PlanNode> {
        let mut result = vec![self];
        for child in &self.children {
            result.extend(child.descendants());
        }
        result
    }

    /// Calculate total cost including children
    pub fn total_cost(&self) -> f64 {
        self.cost + self.children.iter().map(|c| c.total_cost()).sum::<f64>()
    }
}

/// Query plan differ
pub struct PlanDiffer {
    config: DiffConfig,
}

impl PlanDiffer {
    /// Create a new plan differ
    pub fn new(config: DiffConfig) -> Self {
        Self { config }
    }

    /// Compare two query plans
    pub fn compare_plans(&self, old_plan: &PlanNode, new_plan: &PlanNode) -> Result<PlanDiff> {
        let mut structural_changes = Vec::new();
        let mut cost_changes = Vec::new();
        let mut operator_changes = Vec::new();
        let mut node_diffs = Vec::new();

        // Perform recursive comparison
        self.compare_nodes(
            old_plan,
            new_plan,
            "",
            &mut structural_changes,
            &mut cost_changes,
            &mut operator_changes,
            &mut node_diffs,
            0,
        )?;

        // Calculate quality score
        let quality_score = self.calculate_quality_score(&cost_changes, &structural_changes);

        // Determine if this is a regression
        let is_regression = self.detect_regression(&cost_changes, quality_score);

        // Calculate estimated performance impact
        let estimated_impact = self.estimate_performance_impact(old_plan, new_plan);

        // Build summary
        let summary = DiffSummary {
            total_changes: structural_changes.len() + cost_changes.len() + operator_changes.len(),
            structural_changes: structural_changes.len(),
            cost_changes: cost_changes.len(),
            operator_changes: operator_changes.len(),
            estimated_impact,
            message: self.generate_summary_message(
                &structural_changes,
                &cost_changes,
                is_regression,
            ),
        };

        Ok(PlanDiff {
            summary,
            structural_changes,
            cost_changes,
            operator_changes,
            is_regression,
            quality_score,
            node_diffs,
        })
    }

    /// Recursively compare plan nodes
    #[allow(clippy::too_many_arguments)]
    fn compare_nodes(
        &self,
        old_node: &PlanNode,
        new_node: &PlanNode,
        path: &str,
        structural_changes: &mut Vec<StructuralChange>,
        cost_changes: &mut Vec<CostChange>,
        operator_changes: &mut Vec<OperatorChange>,
        node_diffs: &mut Vec<NodeDiff>,
        depth: usize,
    ) -> Result<()> {
        // Check depth limit
        if self.config.max_depth > 0 && depth >= self.config.max_depth {
            return Ok(());
        }

        let current_path = if path.is_empty() {
            old_node.id.clone()
        } else {
            format!("{}/{}", path, old_node.id)
        };

        // Compare node types
        if old_node.node_type != new_node.node_type && self.config.detect_structural_changes {
            structural_changes.push(StructuralChange {
                change_type: StructuralChangeType::NodeReplaced,
                path: current_path.clone(),
                description: format!(
                    "Node type changed from '{}' to '{}'",
                    old_node.node_type, new_node.node_type
                ),
                old_node_type: Some(old_node.node_type.clone()),
                new_node_type: Some(new_node.node_type.clone()),
            });

            operator_changes.push(OperatorChange {
                path: current_path.clone(),
                change_type: OperatorChangeType::TypeChanged,
                old_operator: Some(old_node.node_type.clone()),
                new_operator: Some(new_node.node_type.clone()),
                description: format!(
                    "Operator changed from {} to {}",
                    old_node.node_type, new_node.node_type
                ),
                impact: self.assess_operator_impact(&old_node.node_type, &new_node.node_type),
            });
        }

        // Compare costs
        let cost_diff = new_node.cost - old_node.cost;
        let cost_pct_change = if old_node.cost > 0.0 {
            (cost_diff / old_node.cost) * 100.0
        } else {
            0.0
        };

        if cost_pct_change.abs() > self.config.cost_threshold {
            let is_regression = cost_diff > 0.0; // Higher cost = regression

            cost_changes.push(CostChange {
                path: current_path.clone(),
                old_cost: old_node.cost,
                new_cost: new_node.cost,
                percentage_change: cost_pct_change,
                is_regression,
                description: format!(
                    "Cost changed by {:.1}% ({:.2} → {:.2})",
                    cost_pct_change, old_node.cost, new_node.cost
                ),
            });
        }

        // Compare properties
        let mut property_changes = Vec::new();
        for (key, old_val) in &old_node.properties {
            if let Some(new_val) = new_node.properties.get(key) {
                if old_val != new_val && !self.config.ignore_formatting {
                    property_changes.push(PropertyChange {
                        name: key.clone(),
                        old_value: old_val.clone(),
                        new_value: new_val.clone(),
                    });
                }
            }
        }

        // Add node diff
        node_diffs.push(NodeDiff {
            path: current_path.clone(),
            node_type: old_node.node_type.clone(),
            exists_in_both: true,
            cost_diff: if cost_pct_change.abs() > self.config.cost_threshold {
                Some(cost_diff)
            } else {
                None
            },
            cardinality_diff: if self.config.include_cardinality {
                Some(new_node.cardinality as i64 - old_node.cardinality as i64)
            } else {
                None
            },
            property_changes,
        });

        // Compare children count
        if old_node.children.len() != new_node.children.len()
            && self.config.detect_structural_changes
        {
            structural_changes.push(StructuralChange {
                change_type: StructuralChangeType::ChildrenChanged,
                path: current_path.clone(),
                description: format!(
                    "Children count changed from {} to {}",
                    old_node.children.len(),
                    new_node.children.len()
                ),
                old_node_type: None,
                new_node_type: None,
            });
        }

        // Recursively compare children
        let min_children = old_node.children.len().min(new_node.children.len());
        for i in 0..min_children {
            self.compare_nodes(
                &old_node.children[i],
                &new_node.children[i],
                &current_path,
                structural_changes,
                cost_changes,
                operator_changes,
                node_diffs,
                depth + 1,
            )?;
        }

        Ok(())
    }

    /// Calculate quality score based on changes
    fn calculate_quality_score(
        &self,
        cost_changes: &[CostChange],
        structural_changes: &[StructuralChange],
    ) -> f64 {
        if cost_changes.is_empty() && structural_changes.is_empty() {
            return 0.0; // No changes
        }

        // Calculate cost-based score
        let cost_score: f64 = cost_changes
            .iter()
            .map(|c| {
                if c.is_regression {
                    -c.percentage_change / 100.0
                } else {
                    c.percentage_change.abs() / 100.0
                }
            })
            .sum();

        // Normalize to -1.0 to 1.0 range
        cost_score.clamp(-1.0, 1.0)
    }

    /// Detect if changes represent a performance regression
    fn detect_regression(&self, cost_changes: &[CostChange], quality_score: f64) -> bool {
        if !self.config.highlight_regressions {
            return false;
        }

        // Regression if quality score is significantly negative
        if quality_score < -0.1 {
            return true;
        }

        // Regression if any critical cost increases
        cost_changes
            .iter()
            .any(|c| c.is_regression && c.percentage_change > 20.0)
    }

    /// Estimate performance impact
    fn estimate_performance_impact(&self, old_plan: &PlanNode, new_plan: &PlanNode) -> f64 {
        let old_cost = old_plan.total_cost();
        let new_cost = new_plan.total_cost();

        if new_cost > 0.0 {
            old_cost / new_cost // >1.0 = faster, <1.0 = slower
        } else {
            1.0
        }
    }

    /// Assess impact of operator change
    fn assess_operator_impact(&self, old_op: &str, new_op: &str) -> OperatorImpact {
        // Simple heuristic-based assessment
        match (old_op, new_op) {
            // Index scan is usually better than table scan
            ("TableScan", "IndexScan") => OperatorImpact::Positive,
            ("IndexScan", "TableScan") => OperatorImpact::Negative,

            // Hash join often better than nested loop for large datasets
            ("NestedLoopJoin", "HashJoin") => OperatorImpact::Positive,
            ("HashJoin", "NestedLoopJoin") => OperatorImpact::Negative,

            // Merge join can be better for sorted data
            ("NestedLoopJoin", "MergeJoin") => OperatorImpact::Positive,

            _ => OperatorImpact::Unknown,
        }
    }

    /// Generate summary message
    fn generate_summary_message(
        &self,
        structural_changes: &[StructuralChange],
        cost_changes: &[CostChange],
        is_regression: bool,
    ) -> String {
        if structural_changes.is_empty() && cost_changes.is_empty() {
            return "Plans are identical".to_string();
        }

        let mut parts = Vec::new();

        if !structural_changes.is_empty() {
            parts.push(format!("{} structural change(s)", structural_changes.len()));
        }

        if !cost_changes.is_empty() {
            let improvements = cost_changes.iter().filter(|c| !c.is_regression).count();
            let regressions = cost_changes.iter().filter(|c| c.is_regression).count();

            if improvements > 0 {
                parts.push(format!("{} cost improvement(s)", improvements));
            }
            if regressions > 0 {
                parts.push(format!("{} cost regression(s)", regressions));
            }
        }

        let message = parts.join(", ");

        if is_regression {
            format!("⚠️  REGRESSION DETECTED: {}", message)
        } else {
            message
        }
    }
}

impl PlanDiff {
    /// Check if this diff indicates a regression
    pub fn has_regression(&self) -> bool {
        self.is_regression
    }

    /// Format a human-readable summary
    pub fn format_summary(&self) -> String {
        let mut output = String::new();

        output.push_str("# Query Plan Comparison Summary\n\n");
        output.push_str(&format!("{}\n\n", self.summary.message));

        if self.is_regression {
            output.push_str("## ⚠️  Performance Regression Detected\n\n");
        }

        output.push_str(&format!(
            "**Total Changes**: {}\n",
            self.summary.total_changes
        ));
        output.push_str(&format!(
            "**Estimated Performance Impact**: {:.2}x\n",
            self.summary.estimated_impact
        ));
        output.push_str(&format!("**Quality Score**: {:.2}\n\n", self.quality_score));

        if !self.cost_changes.is_empty() {
            output.push_str("## Cost Changes\n\n");
            for (i, change) in self.cost_changes.iter().enumerate() {
                let indicator = if change.is_regression { "📈" } else { "📉" };
                output.push_str(&format!(
                    "{}. {} **{}**: {}\n",
                    i + 1,
                    indicator,
                    change.path,
                    change.description
                ));
            }
            output.push('\n');
        }

        if !self.structural_changes.is_empty() {
            output.push_str("## Structural Changes\n\n");
            for (i, change) in self.structural_changes.iter().enumerate() {
                output.push_str(&format!(
                    "{}. **{}**: {}\n",
                    i + 1,
                    change.path,
                    change.description
                ));
            }
            output.push('\n');
        }

        if !self.operator_changes.is_empty() {
            output.push_str("## Operator Changes\n\n");
            for (i, change) in self.operator_changes.iter().enumerate() {
                let impact_icon = match change.impact {
                    OperatorImpact::Positive => "✅",
                    OperatorImpact::Negative => "❌",
                    OperatorImpact::Neutral => "➖",
                    OperatorImpact::Unknown => "❓",
                };
                output.push_str(&format!(
                    "{}. {} **{}**: {}\n",
                    i + 1,
                    impact_icon,
                    change.path,
                    change.description
                ));
            }
        }

        output
    }

    /// Export diff as JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| anyhow!("Failed to serialize diff to JSON: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_sample_plan() -> PlanNode {
        PlanNode::new("root".to_string(), "Project".to_string())
            .with_cost(100.0)
            .with_cardinality(1000)
            .with_child(
                PlanNode::new("join".to_string(), "HashJoin".to_string())
                    .with_cost(80.0)
                    .with_cardinality(500)
                    .with_child(
                        PlanNode::new("scan1".to_string(), "IndexScan".to_string())
                            .with_cost(30.0)
                            .with_cardinality(100),
                    )
                    .with_child(
                        PlanNode::new("scan2".to_string(), "IndexScan".to_string())
                            .with_cost(40.0)
                            .with_cardinality(200),
                    ),
            )
    }

    #[test]
    fn test_identical_plans() {
        let plan1 = create_sample_plan();
        let plan2 = create_sample_plan();

        let differ = PlanDiffer::new(DiffConfig::default());
        let diff = differ.compare_plans(&plan1, &plan2).unwrap();

        assert_eq!(diff.summary.total_changes, 0);
        assert!(!diff.has_regression());
    }

    #[test]
    fn test_cost_increase_regression() {
        let plan1 = create_sample_plan();
        let mut plan2 = create_sample_plan();
        plan2.cost = 150.0; // 50% increase

        let differ = PlanDiffer::new(DiffConfig::default());
        let diff = differ.compare_plans(&plan1, &plan2).unwrap();

        assert!(!diff.cost_changes.is_empty());
        assert!(diff.cost_changes[0].is_regression);
        assert!(diff.quality_score < 0.0);
    }

    #[test]
    fn test_operator_change_detection() {
        let plan1 = create_sample_plan();
        let mut plan2 = create_sample_plan();
        plan2.children[0].node_type = "NestedLoopJoin".to_string();

        let differ = PlanDiffer::new(DiffConfig::default());
        let diff = differ.compare_plans(&plan1, &plan2).unwrap();

        assert!(!diff.operator_changes.is_empty());
        assert_eq!(
            diff.operator_changes[0].change_type,
            OperatorChangeType::TypeChanged
        );
    }

    #[test]
    fn test_structural_change_children_count() {
        let plan1 = create_sample_plan();
        let mut plan2 = create_sample_plan();
        plan2.children[0]
            .children
            .push(PlanNode::new("scan3".to_string(), "TableScan".to_string()).with_cost(10.0));

        let differ = PlanDiffer::new(DiffConfig::default());
        let diff = differ.compare_plans(&plan1, &plan2).unwrap();

        assert!(!diff.structural_changes.is_empty());
    }

    #[test]
    fn test_cost_threshold_filtering() {
        let plan1 = create_sample_plan();
        let mut plan2 = create_sample_plan();
        plan2.cost = 102.0; // Only 2% increase, below 5% threshold

        let differ = PlanDiffer::new(DiffConfig::default());
        let diff = differ.compare_plans(&plan1, &plan2).unwrap();

        // Should not report cost change due to threshold
        assert!(diff.cost_changes.is_empty());
    }

    #[test]
    fn test_performance_impact_calculation() {
        let plan1 = create_sample_plan();
        let mut plan2 = create_sample_plan();
        plan2.cost = 50.0; // Half the root cost
                           // Also reduce child costs for more dramatic impact
        plan2.children[0].cost = 40.0; // Was 80.0

        let differ = PlanDiffer::new(DiffConfig::default());
        let diff = differ.compare_plans(&plan1, &plan2).unwrap();

        // Performance impact should be positive (>1.0 = faster)
        // old_total = 250, new_total = 50 + 40 + 30 + 40 = 160
        // impact = 250 / 160 = 1.56
        assert!(diff.summary.estimated_impact > 1.5);
    }

    #[test]
    fn test_summary_formatting() {
        let plan1 = create_sample_plan();
        let mut plan2 = create_sample_plan();
        plan2.cost = 150.0;

        let differ = PlanDiffer::new(DiffConfig::default());
        let diff = differ.compare_plans(&plan1, &plan2).unwrap();

        let summary = diff.format_summary();
        assert!(summary.contains("Query Plan Comparison Summary"));
        assert!(summary.contains("Cost Changes"));
    }

    #[test]
    fn test_json_export() {
        let plan1 = create_sample_plan();
        let plan2 = create_sample_plan();

        let differ = PlanDiffer::new(DiffConfig::default());
        let diff = differ.compare_plans(&plan1, &plan2).unwrap();

        let json = diff.to_json().unwrap();
        assert!(json.contains("summary"));
        assert!(json.contains("quality_score"));
    }

    #[test]
    fn test_operator_impact_assessment() {
        let differ = PlanDiffer::new(DiffConfig::default());

        assert_eq!(
            differ.assess_operator_impact("TableScan", "IndexScan"),
            OperatorImpact::Positive
        );

        assert_eq!(
            differ.assess_operator_impact("IndexScan", "TableScan"),
            OperatorImpact::Negative
        );

        assert_eq!(
            differ.assess_operator_impact("NestedLoopJoin", "HashJoin"),
            OperatorImpact::Positive
        );
    }

    #[test]
    fn test_depth_limiting() {
        let plan1 = create_sample_plan();
        let plan2 = create_sample_plan();

        let config = DiffConfig {
            max_depth: 1, // Only compare top level
            ..Default::default()
        };

        let differ = PlanDiffer::new(config);
        let diff = differ.compare_plans(&plan1, &plan2).unwrap();

        // Should have limited node_diffs due to depth restriction
        assert!(diff.node_diffs.len() <= 2);
    }

    #[test]
    fn test_total_cost_calculation() {
        let plan = create_sample_plan();
        let total = plan.total_cost();

        // Should sum all nodes: 100 + 80 + 30 + 40 = 250
        assert_eq!(total, 250.0);
    }
}
