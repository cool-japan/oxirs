//! GraphQL query complexity analysis.
//!
//! Prevents expensive or abusive queries by calculating a complexity score
//! and comparing it against configurable limits.  Each field has a base cost;
//! list fields multiply the accumulated child cost by a configurable factor.

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Configuration types
// ────────────────────────────────────────────────────────────────────────────

/// Per-field complexity override for a specific type.
#[derive(Debug, Clone)]
pub struct FieldComplexity {
    pub field_name: String,
    /// Base cost charged for selecting this field.
    pub base_cost: usize,
    /// Multiplier applied to this field's child cost subtree.
    pub multiplier: usize,
}

impl FieldComplexity {
    pub fn new(field_name: impl Into<String>, base_cost: usize, multiplier: usize) -> Self {
        Self {
            field_name: field_name.into(),
            base_cost,
            multiplier,
        }
    }
}

/// Complexity rules for a GraphQL object type.
#[derive(Debug, Clone)]
pub struct ComplexityRule {
    pub type_name: String,
    /// Per-field overrides keyed by field name.
    pub fields: HashMap<String, FieldComplexity>,
}

impl ComplexityRule {
    pub fn new(type_name: impl Into<String>) -> Self {
        Self {
            type_name: type_name.into(),
            fields: HashMap::new(),
        }
    }

    pub fn with_field(mut self, fc: FieldComplexity) -> Self {
        self.fields.insert(fc.field_name.clone(), fc);
        self
    }
}

/// Global limits and defaults for complexity analysis.
#[derive(Debug, Clone)]
pub struct ComplexityConfig {
    /// Maximum allowed complexity score (inclusive).
    pub max_complexity: usize,
    /// Maximum allowed query depth (inclusive).
    pub max_depth: usize,
    /// Default cost assigned to every field without an explicit rule.
    pub default_cost: usize,
    /// Multiplier applied to child cost when the field returns a list.
    pub list_multiplier: usize,
}

impl ComplexityConfig {
    pub fn new(
        max_complexity: usize,
        max_depth: usize,
        default_cost: usize,
        list_multiplier: usize,
    ) -> Self {
        Self {
            max_complexity,
            max_depth,
            default_cost,
            list_multiplier,
        }
    }
}

impl Default for ComplexityConfig {
    fn default() -> Self {
        Self::new(1_000, 10, 1, 10)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Query tree
// ────────────────────────────────────────────────────────────────────────────

/// A node in a simplified GraphQL query tree.
#[derive(Debug, Clone)]
pub struct QueryNode {
    pub field: String,
    pub type_name: String,
    pub children: Vec<QueryNode>,
    /// `true` if this field returns a list of items (e.g. `[User!]!`).
    pub is_list: bool,
    /// Resolved argument values (e.g. `"first" -> "10"`).
    pub args: HashMap<String, String>,
}

impl QueryNode {
    /// Create a leaf node (no children, not a list).
    pub fn leaf(field: impl Into<String>, type_name: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            type_name: type_name.into(),
            children: vec![],
            is_list: false,
            args: HashMap::new(),
        }
    }

    /// Create a non-list node with children.
    pub fn with_children(
        field: impl Into<String>,
        type_name: impl Into<String>,
        children: Vec<QueryNode>,
    ) -> Self {
        Self {
            field: field.into(),
            type_name: type_name.into(),
            children,
            is_list: false,
            args: HashMap::new(),
        }
    }

    /// Create a list field node with children.
    pub fn list_field(
        field: impl Into<String>,
        type_name: impl Into<String>,
        children: Vec<QueryNode>,
    ) -> Self {
        Self {
            field: field.into(),
            type_name: type_name.into(),
            children,
            is_list: true,
            args: HashMap::new(),
        }
    }

    /// Depth of this node's subtree (leaf = 1).
    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            1
        } else {
            1 + self.children.iter().map(|c| c.depth()).max().unwrap_or(0)
        }
    }

    /// Total number of fields (nodes) in this subtree.
    pub fn field_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.field_count()).sum::<usize>()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Analysis result
// ────────────────────────────────────────────────────────────────────────────

/// Output of a complexity analysis pass.
#[derive(Debug, Clone)]
pub struct ComplexityResult {
    /// Computed total complexity score.
    pub complexity: usize,
    /// Maximum query depth observed.
    pub depth: usize,
    /// `true` if `complexity > max_complexity`.
    pub exceeds_limit: bool,
    /// `true` if `depth > max_depth`.
    pub depth_exceeded: bool,
    /// Per-field breakdown: `(path_string, cost)`.
    pub breakdown: Vec<(String, usize)>,
}

// ────────────────────────────────────────────────────────────────────────────
// Analyzer
// ────────────────────────────────────────────────────────────────────────────

/// Analyses GraphQL query trees for complexity and depth.
pub struct QueryComplexityAnalyzer {
    config: ComplexityConfig,
    rules: HashMap<String, ComplexityRule>,
}

impl QueryComplexityAnalyzer {
    pub fn new(config: ComplexityConfig) -> Self {
        Self {
            config,
            rules: HashMap::new(),
        }
    }

    /// Register per-type complexity rules.
    pub fn add_rule(&mut self, rule: ComplexityRule) {
        self.rules.insert(rule.type_name.clone(), rule);
    }

    /// Analyse the given query root and return a [`ComplexityResult`].
    pub fn analyze(&self, root: &QueryNode) -> ComplexityResult {
        let mut breakdown = Vec::new();
        let complexity = self.traverse(root, &mut breakdown);
        let depth = root.depth();
        ComplexityResult {
            complexity,
            depth,
            exceeds_limit: complexity > self.config.max_complexity,
            depth_exceeded: depth > self.config.max_depth,
            breakdown,
        }
    }

    /// Returns the cost of `node` (depth is not used in cost calculation but kept for API compatibility).
    pub fn node_cost(&self, node: &QueryNode, _depth: usize) -> usize {
        // Base cost: check per-type rule first, fall back to default
        let base = self
            .rules
            .get(&node.type_name)
            .and_then(|r| r.fields.get(&node.field))
            .map(|fc| fc.base_cost)
            .unwrap_or(self.config.default_cost);

        // Multiplier: rule-defined or list_multiplier if this is a list field
        let multiplier = self
            .rules
            .get(&node.type_name)
            .and_then(|r| r.fields.get(&node.field))
            .map(|fc| fc.multiplier)
            .unwrap_or(if node.is_list {
                self.config.list_multiplier
            } else {
                1
            });

        let child_cost: usize = node.children.iter().map(|c| self.node_cost(c, 0)).sum();

        base + multiplier * child_cost
    }

    fn traverse(&self, node: &QueryNode, breakdown: &mut Vec<(String, usize)>) -> usize {
        let base = self
            .rules
            .get(&node.type_name)
            .and_then(|r| r.fields.get(&node.field))
            .map(|fc| fc.base_cost)
            .unwrap_or(self.config.default_cost);

        let multiplier = self
            .rules
            .get(&node.type_name)
            .and_then(|r| r.fields.get(&node.field))
            .map(|fc| fc.multiplier)
            .unwrap_or(if node.is_list {
                self.config.list_multiplier
            } else {
                1
            });

        let child_cost: usize = node
            .children
            .iter()
            .map(|c| self.traverse(c, breakdown))
            .sum();

        let total = base + multiplier * child_cost;
        breakdown.push((node.field.clone(), total));
        total
    }

    /// Returns `true` if the result is within all configured limits.
    pub fn is_allowed(&self, result: &ComplexityResult) -> bool {
        !result.exceeds_limit && !result.depth_exceeded
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Simple per-field cost API (lighter-weight, v1.7.0)
// ────────────────────────────────────────────────────────────────────────────

/// A simple flat complexity rule: cost per field name with optional multiplier argument.
#[derive(Debug, Clone)]
pub struct SimpleCostRule {
    /// Field this rule applies to.
    pub field_name: String,
    /// Base cost for this field.
    pub cost: usize,
    /// If set, multiply `cost` by the value of this argument.
    pub multiplier_arg: Option<String>,
}

impl SimpleCostRule {
    /// Create a fixed-cost rule.
    pub fn fixed(field_name: impl Into<String>, cost: usize) -> Self {
        Self {
            field_name: field_name.into(),
            cost,
            multiplier_arg: None,
        }
    }

    /// Create a rule whose cost scales with a numeric argument.
    pub fn multiplied(
        field_name: impl Into<String>,
        cost: usize,
        multiplier_arg: impl Into<String>,
    ) -> Self {
        Self {
            field_name: field_name.into(),
            cost,
            multiplier_arg: Some(multiplier_arg.into()),
        }
    }
}

/// A flat (non-tree) representation of a GraphQL field selection.
#[derive(Debug, Clone)]
pub struct QueryField {
    /// Name of the selected field.
    pub name: String,
    /// Numeric arguments for multiplier evaluation.
    pub args: HashMap<String, usize>,
    /// Nested selections.
    pub children: Vec<QueryField>,
}

impl QueryField {
    /// Create a leaf field with no args and no children.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            args: HashMap::new(),
            children: Vec::new(),
        }
    }

    /// Add a numeric argument (builder).
    pub fn with_arg(mut self, key: impl Into<String>, val: usize) -> Self {
        self.args.insert(key.into(), val);
        self
    }

    /// Add a child field (builder).
    pub fn with_child(mut self, child: QueryField) -> Self {
        self.children.push(child);
        self
    }

    /// Maximum nesting depth (leaf = 1).
    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            1
        } else {
            1 + self.children.iter().map(|c| c.depth()).max().unwrap_or(0)
        }
    }
}

/// Lightweight complexity analyzer driven by [`SimpleCostRule`]s.
pub struct SimpleComplexityAnalyzer {
    rules: HashMap<String, SimpleCostRule>,
    default_cost: usize,
    limit: usize,
}

impl SimpleComplexityAnalyzer {
    /// Create with the given cost limit (default field cost = 1).
    pub fn new(limit: usize) -> Self {
        Self {
            rules: HashMap::new(),
            default_cost: 1,
            limit,
        }
    }

    /// Override the default cost for fields without an explicit rule.
    pub fn with_default_cost(mut self, cost: usize) -> Self {
        self.default_cost = cost;
        self
    }

    /// Register a field rule.
    pub fn add_rule(&mut self, rule: SimpleCostRule) {
        self.rules.insert(rule.field_name.clone(), rule);
    }

    /// Cost for a single field + its args.
    pub fn cost_for_field(&self, name: &str, args: &HashMap<String, usize>) -> usize {
        match self.rules.get(name) {
            Some(rule) => match &rule.multiplier_arg {
                Some(arg) => {
                    let m = args.get(arg).copied().unwrap_or(1).max(1);
                    rule.cost.saturating_mul(m)
                }
                None => rule.cost,
            },
            None => self.default_cost,
        }
    }

    /// Analyse the given list of top-level fields and return total cost + per-field breakdown.
    pub fn analyze_fields(&self, fields: &[QueryField]) -> (usize, Vec<(String, usize)>) {
        let mut breakdown = Vec::new();
        let total = self.sum_fields(fields, &mut breakdown);
        (total, breakdown)
    }

    /// Returns `true` when query cost is within the limit.
    pub fn is_within_limit(&self, fields: &[QueryField]) -> bool {
        let (total, _) = self.analyze_fields(fields);
        total <= self.limit
    }

    fn sum_fields(&self, fields: &[QueryField], out: &mut Vec<(String, usize)>) -> usize {
        let mut total = 0_usize;
        for f in fields {
            let cost = self.cost_for_field(&f.name, &f.args);
            out.push((f.name.clone(), cost));
            total = total.saturating_add(cost);
            total = total.saturating_add(self.sum_fields(&f.children, out));
        }
        total
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn default_analyzer() -> QueryComplexityAnalyzer {
        QueryComplexityAnalyzer::new(ComplexityConfig::new(100, 5, 1, 10))
    }

    // ── QueryNode helpers ─────────────────────────────────────────────────────

    #[test]
    fn test_leaf_depth_is_1() {
        let n = QueryNode::leaf("id", "String");
        assert_eq!(n.depth(), 1);
    }

    #[test]
    fn test_leaf_field_count_is_1() {
        let n = QueryNode::leaf("id", "String");
        assert_eq!(n.field_count(), 1);
    }

    #[test]
    fn test_with_children_depth() {
        let n = QueryNode::with_children(
            "user",
            "User",
            vec![
                QueryNode::leaf("id", "ID"),
                QueryNode::leaf("name", "String"),
            ],
        );
        assert_eq!(n.depth(), 2);
    }

    #[test]
    fn test_with_children_field_count() {
        let n = QueryNode::with_children(
            "user",
            "User",
            vec![
                QueryNode::leaf("id", "ID"),
                QueryNode::leaf("name", "String"),
            ],
        );
        assert_eq!(n.field_count(), 3);
    }

    #[test]
    fn test_list_field_is_list() {
        let n = QueryNode::list_field("users", "User", vec![QueryNode::leaf("id", "ID")]);
        assert!(n.is_list);
    }

    #[test]
    fn test_list_field_depth() {
        let n = QueryNode::list_field("users", "User", vec![QueryNode::leaf("id", "ID")]);
        assert_eq!(n.depth(), 2);
    }

    #[test]
    fn test_nested_depth() {
        let inner =
            QueryNode::with_children("posts", "Post", vec![QueryNode::leaf("title", "String")]);
        let outer = QueryNode::with_children("user", "User", vec![inner]);
        assert_eq!(outer.depth(), 3);
    }

    // ── single leaf cost ──────────────────────────────────────────────────────

    #[test]
    fn test_single_leaf_complexity() {
        let a = default_analyzer();
        let r = a.analyze(&QueryNode::leaf("id", "ID"));
        assert_eq!(r.complexity, 1); // default_cost = 1
    }

    #[test]
    fn test_two_leaves_cost_sum() {
        let a = default_analyzer();
        let n = QueryNode::with_children(
            "root",
            "Query",
            vec![
                QueryNode::leaf("a", "String"),
                QueryNode::leaf("b", "String"),
            ],
        );
        // root: 1 + 1*(1 + 1) = 3
        let r = a.analyze(&n);
        assert_eq!(r.complexity, 3);
    }

    // ── list multiplier ───────────────────────────────────────────────────────

    #[test]
    fn test_list_field_multiplier_applied() {
        let a = default_analyzer(); // list_multiplier=10
        let n = QueryNode::list_field("users", "User", vec![QueryNode::leaf("id", "ID")]);
        // list: base=1, child_cost=1, multiplier=10 → 1 + 10*1 = 11
        let r = a.analyze(&n);
        assert_eq!(r.complexity, 11);
    }

    #[test]
    fn test_non_list_no_extra_multiplier() {
        let a = default_analyzer();
        let n = QueryNode::with_children("user", "User", vec![QueryNode::leaf("id", "ID")]);
        // base=1 + 1*(1) = 2
        let r = a.analyze(&n);
        assert_eq!(r.complexity, 2);
    }

    #[test]
    fn test_list_field_two_children() {
        let a = default_analyzer();
        let n = QueryNode::list_field(
            "users",
            "User",
            vec![
                QueryNode::leaf("id", "ID"),
                QueryNode::leaf("name", "String"),
            ],
        );
        // base=1, child_cost=2, multiplier=10 → 1 + 20 = 21
        let r = a.analyze(&n);
        assert_eq!(r.complexity, 21);
    }

    // ── exceeds_limit / depth_exceeded ───────────────────────────────────────

    #[test]
    fn test_exceeds_limit_detected() {
        let a = QueryComplexityAnalyzer::new(ComplexityConfig::new(5, 10, 1, 10));
        let n = QueryNode::list_field(
            "users",
            "User",
            vec![
                QueryNode::leaf("id", "ID"),
                QueryNode::leaf("name", "String"),
                QueryNode::leaf("email", "String"),
            ],
        );
        // 1 + 10*3 = 31 > 5
        let r = a.analyze(&n);
        assert!(r.exceeds_limit);
    }

    #[test]
    fn test_does_not_exceed_limit() {
        let a = QueryComplexityAnalyzer::new(ComplexityConfig::new(100, 10, 1, 10));
        let r = a.analyze(&QueryNode::leaf("id", "ID"));
        assert!(!r.exceeds_limit);
    }

    #[test]
    fn test_depth_exceeded_detected() {
        let a = QueryComplexityAnalyzer::new(ComplexityConfig::new(10_000, 2, 1, 1));
        // depth 3 > max_depth 2
        let deep = QueryNode::with_children(
            "a",
            "A",
            vec![QueryNode::with_children(
                "b",
                "B",
                vec![QueryNode::leaf("c", "C")],
            )],
        );
        let r = a.analyze(&deep);
        assert!(r.depth_exceeded);
    }

    #[test]
    fn test_depth_not_exceeded() {
        let a = QueryComplexityAnalyzer::new(ComplexityConfig::new(10_000, 5, 1, 1));
        let n = QueryNode::with_children("a", "A", vec![QueryNode::leaf("b", "B")]);
        let r = a.analyze(&n);
        assert!(!r.depth_exceeded);
    }

    // ── custom rules ──────────────────────────────────────────────────────────

    #[test]
    fn test_custom_rule_overrides_default_cost() {
        let mut a = QueryComplexityAnalyzer::new(ComplexityConfig::new(10_000, 10, 1, 10));
        a.add_rule(
            ComplexityRule::new("Query").with_field(FieldComplexity::new("expensiveField", 50, 1)),
        );
        let n = QueryNode::leaf("expensiveField", "Query"); // note: field="expensiveField", type_name="Query"
                                                            // Wait - the rule is keyed on type_name; let's construct correctly:
        let n2 = QueryNode {
            field: "expensiveField".to_string(),
            type_name: "Query".to_string(),
            children: vec![],
            is_list: false,
            args: HashMap::new(),
        };
        let r = a.analyze(&n2);
        assert_eq!(r.complexity, 50); // base_cost=50 overrides default 1
        let _ = n; // suppress unused warning
    }

    #[test]
    fn test_custom_rule_multiplier_override() {
        let mut a = QueryComplexityAnalyzer::new(ComplexityConfig::new(10_000, 10, 1, 5));
        // Override: multiplier = 100 for this list field
        a.add_rule(
            ComplexityRule::new("Query").with_field(FieldComplexity::new("heavyList", 1, 100)),
        );
        let n = QueryNode {
            field: "heavyList".to_string(),
            type_name: "Query".to_string(),
            children: vec![QueryNode::leaf("id", "ID")],
            is_list: true, // is_list=true but multiplier overridden by rule
            args: HashMap::new(),
        };
        // base=1, child_cost=1, multiplier=100 → 1 + 100*1 = 101
        let r = a.analyze(&n);
        assert_eq!(r.complexity, 101);
    }

    // ── breakdown ─────────────────────────────────────────────────────────────

    #[test]
    fn test_breakdown_contains_all_fields() {
        let a = default_analyzer();
        let n = QueryNode::with_children(
            "root",
            "Query",
            vec![
                QueryNode::leaf("a", "String"),
                QueryNode::leaf("b", "String"),
            ],
        );
        let r = a.analyze(&n);
        let fields: Vec<&str> = r.breakdown.iter().map(|(f, _)| f.as_str()).collect();
        assert!(fields.contains(&"root"));
        assert!(fields.contains(&"a"));
        assert!(fields.contains(&"b"));
    }

    #[test]
    fn test_breakdown_not_empty_for_single_leaf() {
        let a = default_analyzer();
        let r = a.analyze(&QueryNode::leaf("id", "ID"));
        assert!(!r.breakdown.is_empty());
    }

    // ── is_allowed ────────────────────────────────────────────────────────────

    #[test]
    fn test_is_allowed_within_limits() {
        let a = default_analyzer();
        let r = a.analyze(&QueryNode::leaf("id", "ID"));
        assert!(a.is_allowed(&r));
    }

    #[test]
    fn test_is_allowed_complexity_exceeded() {
        let a = QueryComplexityAnalyzer::new(ComplexityConfig::new(0, 10, 1, 1));
        let r = a.analyze(&QueryNode::leaf("id", "ID")); // complexity=1 > max=0
        assert!(!a.is_allowed(&r));
    }

    #[test]
    fn test_is_allowed_depth_exceeded() {
        let a = QueryComplexityAnalyzer::new(ComplexityConfig::new(10_000, 1, 1, 1));
        let n = QueryNode::with_children("a", "A", vec![QueryNode::leaf("b", "B")]); // depth=2 > max=1
        let r = a.analyze(&n);
        assert!(!a.is_allowed(&r));
    }

    // ── deep nesting ──────────────────────────────────────────────────────────

    #[test]
    fn test_deep_nesting_cost_calculation() {
        let a = QueryComplexityAnalyzer::new(ComplexityConfig::new(100_000, 20, 1, 1));
        // depth 5: a→b→c→d→e(leaf)
        let n = QueryNode::with_children(
            "a",
            "A",
            vec![QueryNode::with_children(
                "b",
                "B",
                vec![QueryNode::with_children(
                    "c",
                    "C",
                    vec![QueryNode::with_children(
                        "d",
                        "D",
                        vec![QueryNode::leaf("e", "E")],
                    )],
                )],
            )],
        );
        let r = a.analyze(&n);
        // Each node: base=1, multiplier=1, child propagates
        // e=1, d=1+1=2, c=1+2=3, b=1+3=4, a=1+4=5
        assert_eq!(r.complexity, 5);
        assert_eq!(r.depth, 5);
    }

    #[test]
    fn test_zero_child_leaf_node() {
        let a = default_analyzer();
        let n = QueryNode::leaf("scalar", "String");
        let r = a.analyze(&n);
        assert_eq!(r.complexity, 1);
        assert_eq!(r.depth, 1);
        assert_eq!(n.field_count(), 1);
    }

    #[test]
    fn test_node_cost_matches_analyze() {
        let a = default_analyzer();
        let n = QueryNode::list_field("items", "Item", vec![QueryNode::leaf("id", "ID")]);
        let r = a.analyze(&n);
        let nc = a.node_cost(&n, 0);
        assert_eq!(r.complexity, nc);
    }

    #[test]
    fn test_multiple_rules_independent() {
        let mut a = QueryComplexityAnalyzer::new(ComplexityConfig::new(10_000, 10, 1, 10));
        a.add_rule(ComplexityRule::new("TypeA").with_field(FieldComplexity::new("fieldA", 5, 1)));
        a.add_rule(ComplexityRule::new("TypeB").with_field(FieldComplexity::new("fieldB", 20, 1)));
        let na = QueryNode {
            field: "fieldA".to_string(),
            type_name: "TypeA".to_string(),
            children: vec![],
            is_list: false,
            args: HashMap::new(),
        };
        let nb = QueryNode {
            field: "fieldB".to_string(),
            type_name: "TypeB".to_string(),
            children: vec![],
            is_list: false,
            args: HashMap::new(),
        };
        let ra = a.analyze(&na);
        let rb = a.analyze(&nb);
        assert_eq!(ra.complexity, 5);
        assert_eq!(rb.complexity, 20);
    }

    // ── SimpleComplexityAnalyzer / QueryField ─────────────────────────────────

    #[test]
    fn test_query_field_new_defaults() {
        let f = QueryField::new("users");
        assert_eq!(f.name, "users");
        assert!(f.args.is_empty());
        assert!(f.children.is_empty());
    }

    #[test]
    fn test_query_field_with_arg() {
        let f = QueryField::new("list").with_arg("first", 10);
        assert_eq!(f.args.get("first"), Some(&10));
    }

    #[test]
    fn test_query_field_with_child() {
        let f = QueryField::new("user").with_child(QueryField::new("id"));
        assert_eq!(f.children.len(), 1);
    }

    #[test]
    fn test_query_field_depth_leaf() {
        let f = QueryField::new("id");
        assert_eq!(f.depth(), 1);
    }

    #[test]
    fn test_query_field_depth_two_levels() {
        let f = QueryField::new("user").with_child(QueryField::new("id"));
        assert_eq!(f.depth(), 2);
    }

    #[test]
    fn test_query_field_depth_three_levels() {
        let f =
            QueryField::new("a").with_child(QueryField::new("b").with_child(QueryField::new("c")));
        assert_eq!(f.depth(), 3);
    }

    #[test]
    fn test_simple_analyzer_empty_fields_zero_cost() {
        let a = SimpleComplexityAnalyzer::new(100);
        assert!(a.is_within_limit(&[]));
        let (total, _) = a.analyze_fields(&[]);
        assert_eq!(total, 0);
    }

    #[test]
    fn test_simple_analyzer_default_cost_one() {
        let a = SimpleComplexityAnalyzer::new(100);
        let fields = vec![QueryField::new("a"), QueryField::new("b")];
        let (total, _) = a.analyze_fields(&fields);
        assert_eq!(total, 2);
    }

    #[test]
    fn test_simple_analyzer_custom_default_cost() {
        let a = SimpleComplexityAnalyzer::new(100).with_default_cost(5);
        let fields = vec![QueryField::new("a")];
        let (total, _) = a.analyze_fields(&fields);
        assert_eq!(total, 5);
    }

    #[test]
    fn test_simple_analyzer_fixed_rule() {
        let mut a = SimpleComplexityAnalyzer::new(100);
        a.add_rule(SimpleCostRule::fixed("expensive", 50));
        let fields = vec![QueryField::new("expensive")];
        let (total, _) = a.analyze_fields(&fields);
        assert_eq!(total, 50);
    }

    #[test]
    fn test_simple_analyzer_multiplied_rule() {
        let mut a = SimpleComplexityAnalyzer::new(10_000);
        a.add_rule(SimpleCostRule::multiplied("list", 10, "first"));
        let fields = vec![QueryField::new("list").with_arg("first", 5)];
        let (total, _) = a.analyze_fields(&fields);
        assert_eq!(total, 50); // 10 * 5
    }

    #[test]
    fn test_simple_analyzer_multiplied_missing_arg_defaults_one() {
        let mut a = SimpleComplexityAnalyzer::new(10_000);
        a.add_rule(SimpleCostRule::multiplied("list", 10, "first"));
        let fields = vec![QueryField::new("list")]; // no first arg
        let (total, _) = a.analyze_fields(&fields);
        assert_eq!(total, 10); // 10 * 1
    }

    #[test]
    fn test_simple_analyzer_within_limit_true() {
        let a = SimpleComplexityAnalyzer::new(100);
        let fields = vec![QueryField::new("x")];
        assert!(a.is_within_limit(&fields));
    }

    #[test]
    fn test_simple_analyzer_exceeds_limit() {
        let a = SimpleComplexityAnalyzer::new(0);
        let fields = vec![QueryField::new("x")];
        assert!(!a.is_within_limit(&fields));
    }

    #[test]
    fn test_simple_analyzer_breakdown_has_entries() {
        let a = SimpleComplexityAnalyzer::new(100);
        let fields = vec![QueryField::new("a"), QueryField::new("b")];
        let (_, breakdown) = a.analyze_fields(&fields);
        assert_eq!(breakdown.len(), 2);
    }

    #[test]
    fn test_simple_analyzer_children_included_in_total() {
        let a = SimpleComplexityAnalyzer::new(100);
        let f = QueryField::new("user")
            .with_child(QueryField::new("name"))
            .with_child(QueryField::new("email"));
        let (total, _) = a.analyze_fields(&[f]);
        assert_eq!(total, 3); // user + name + email
    }

    #[test]
    fn test_simple_cost_rule_fixed() {
        let rule = SimpleCostRule::fixed("field", 7);
        assert_eq!(rule.cost, 7);
        assert!(rule.multiplier_arg.is_none());
    }

    #[test]
    fn test_simple_cost_rule_multiplied() {
        let rule = SimpleCostRule::multiplied("field", 7, "count");
        assert_eq!(rule.multiplier_arg, Some("count".to_string()));
    }

    #[test]
    fn test_cost_for_field_unknown_uses_default() {
        let a = SimpleComplexityAnalyzer::new(100).with_default_cost(3);
        let cost = a.cost_for_field("unknown", &HashMap::new());
        assert_eq!(cost, 3);
    }

    #[test]
    fn test_cost_for_field_known_fixed() {
        let mut a = SimpleComplexityAnalyzer::new(100);
        a.add_rule(SimpleCostRule::fixed("known", 42));
        let cost = a.cost_for_field("known", &HashMap::new());
        assert_eq!(cost, 42);
    }

    #[test]
    fn test_query_field_multiple_children() {
        let f = QueryField::new("root")
            .with_child(QueryField::new("a"))
            .with_child(QueryField::new("b"))
            .with_child(QueryField::new("c"));
        assert_eq!(f.children.len(), 3);
    }
}
