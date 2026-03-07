//! Apollo-Federation-style query planner for distributed GraphQL sub-graphs.
//!
//! This module implements a lightweight but complete federation planner:
//! - `FederationQueryPlanner`: splits a GraphQL query across registered
//!   sub-graphs based on type ownership (`@key` directives) and produces a
//!   dependency-ordered `FederationPlan`.
//! - `EntityResolver`: handles `_entities` lookups for cross-graph references.
//!
//! The planner operates purely in-process and does **not** make network
//! calls — execution is the responsibility of the caller.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Sub-graph metadata
// ---------------------------------------------------------------------------

/// A `@key` directive on a federation type.
///
/// Corresponds to `@key(fields: "id")` in a sub-graph SDL.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FederationKey {
    /// The type that carries this key (e.g. `"User"`).
    pub type_name: String,
    /// The fields that form the key (e.g. `["id"]`).
    pub fields: Vec<String>,
}

/// A registered sub-graph in the federation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubGraph {
    /// Unique sub-graph name (e.g. `"accounts"`).
    pub name: String,
    /// URL of the sub-graph's GraphQL endpoint.
    pub url: String,
    /// GraphQL types owned by this sub-graph.
    pub types: Vec<String>,
    /// `@key` directives defined in this sub-graph.
    pub keys: Vec<FederationKey>,
}

impl SubGraph {
    /// Create a new sub-graph descriptor.
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            types: vec![],
            keys: vec![],
        }
    }

    /// Add an owned type.
    pub fn with_type(mut self, type_name: impl Into<String>) -> Self {
        self.types.push(type_name.into());
        self
    }

    /// Add a federation key.
    pub fn with_key(mut self, key: FederationKey) -> Self {
        self.keys.push(key);
        self
    }
}

// ---------------------------------------------------------------------------
// Plan types
// ---------------------------------------------------------------------------

/// A single step in a `FederationPlan`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationStep {
    /// The sub-graph this step targets.
    pub subgraph: String,
    /// The GraphQL query (or fragment) to send to the sub-graph.
    pub query: String,
    /// Indices of earlier steps whose results must be available before this
    /// step executes.
    pub depends_on: Vec<usize>,
    /// The top-level type this step resolves (informational).
    pub resolves_type: String,
}

/// An execution plan produced by `FederationQueryPlanner`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationPlan {
    /// Ordered list of execution steps.
    pub steps: Vec<FederationStep>,
    /// Estimated end-to-end latency in milliseconds.
    ///
    /// Computed as the sum of `base_latency_ms` for each step along the
    /// critical path.
    pub estimated_latency_ms: u64,
    /// Whether all steps are independent (no `depends_on` links).
    pub is_parallelizable: bool,
}

impl FederationPlan {
    /// Returns `true` if the plan has no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------

/// Configuration for the federation planner.
#[derive(Debug, Clone)]
pub struct FederationPlannerConfig {
    /// Assumed per-hop network latency in milliseconds used for cost
    /// estimation.
    pub base_latency_ms: u64,
}

impl Default for FederationPlannerConfig {
    fn default() -> Self {
        Self {
            base_latency_ms: 50,
        }
    }
}

/// Distributes a GraphQL query across federated sub-graphs.
///
/// The planner uses a two-phase approach:
/// 1. **Routing**: for each top-level selection in the query, find the
///    sub-graph that owns that type.
/// 2. **Dependency resolution**: if a type in sub-graph B references a key
///    field from sub-graph A, add a dependency edge so that B's step runs
///    after A's.
#[derive(Debug)]
pub struct FederationQueryPlanner {
    config: FederationPlannerConfig,
}

impl FederationQueryPlanner {
    /// Create a new planner with default configuration.
    pub fn new() -> Self {
        Self {
            config: FederationPlannerConfig::default(),
        }
    }

    /// Create a planner with custom configuration.
    pub fn with_config(config: FederationPlannerConfig) -> Self {
        Self { config }
    }

    /// Plan a GraphQL query across the given sub-graphs.
    ///
    /// The `query` string is a simplified GraphQL query; the planner extracts
    /// top-level field names and matches them to owning sub-graphs.
    ///
    /// Returns a `FederationPlan` with a step for each sub-graph that has at
    /// least one matching type.
    pub fn plan_query(&self, query: &str, subgraphs: &[SubGraph]) -> FederationPlan {
        // Build a type-to-subgraph index
        let type_owner: HashMap<&str, &SubGraph> = subgraphs
            .iter()
            .flat_map(|sg| sg.types.iter().map(move |t| (t.as_str(), sg)))
            .collect();

        // Extract type names referenced in the query
        let referenced_types = Self::extract_type_references(query);

        // Group referenced types by owning sub-graph
        let mut subgraph_fields: HashMap<&str, Vec<String>> = HashMap::new();
        for type_name in &referenced_types {
            if let Some(sg) = type_owner.get(type_name.as_str()) {
                subgraph_fields
                    .entry(sg.name.as_str())
                    .or_default()
                    .push(type_name.clone());
            }
        }

        // Detect cross-graph key dependencies
        // For each pair (A, B), if B references a type whose key is provided
        // by A, B depends on A.
        let key_owners: HashMap<&str, &str> = subgraphs
            .iter()
            .flat_map(|sg| {
                sg.keys
                    .iter()
                    .map(move |k| (k.type_name.as_str(), sg.name.as_str()))
            })
            .collect();

        // Build steps
        let mut steps: Vec<FederationStep> = Vec::new();
        // Deterministic ordering: sort sub-graph names
        let mut sg_names: Vec<&str> = subgraph_fields.keys().copied().collect();
        sg_names.sort_unstable();

        // Map sub-graph name → step index for dependency resolution
        let name_to_idx: HashMap<&str, usize> = sg_names
            .iter()
            .enumerate()
            .map(|(i, &name)| (name, i))
            .collect();

        for sg_name in &sg_names {
            let fields = &subgraph_fields[sg_name];
            let query_fragment = Self::build_query_fragment(fields);

            // Compute depends_on: if any type resolved by this sub-graph has a
            // key owned by another sub-graph that is also in the plan, add that
            // step as a dependency.
            let mut depends_on: Vec<usize> = fields
                .iter()
                .filter_map(|type_name| {
                    let key_owner_sg = key_owners.get(type_name.as_str())?;
                    if key_owner_sg != sg_name {
                        name_to_idx.get(key_owner_sg).copied()
                    } else {
                        None
                    }
                })
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            depends_on.sort_unstable();

            steps.push(FederationStep {
                subgraph: sg_name.to_string(),
                query: query_fragment,
                depends_on,
                resolves_type: fields.join(", "),
            });
        }

        // Compute estimated latency along the critical path
        let estimated_latency_ms = self.estimate_latency(&steps);
        let is_parallelizable = steps.iter().all(|s| s.depends_on.is_empty());

        FederationPlan {
            steps,
            estimated_latency_ms,
            is_parallelizable,
        }
    }

    /// Extract type names referenced in a query.
    ///
    /// A "type reference" here means a capitalised identifier that appears
    /// either as an inline fragment condition (`... on TypeName`) or as a
    /// field with a selection set.  Simple scalar field names are ignored.
    fn extract_type_references(query: &str) -> Vec<String> {
        let mut types = Vec::new();
        for line in query.lines() {
            let trimmed = line.trim();
            // Inline fragment: `... on TypeName {`
            if let Some(rest) = trimmed.strip_prefix("... on ") {
                let type_name = rest.trim_end_matches('{').trim();
                if type_name.starts_with(|c: char| c.is_uppercase()) {
                    types.push(type_name.to_string());
                }
                continue;
            }
            // Object field with selection: `TypeName {` or `fieldName: TypeName {`
            let candidate = trimmed
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_end_matches('{');
            if candidate.starts_with(|c: char| c.is_uppercase()) && trimmed.ends_with('{') {
                types.push(candidate.to_string());
            }
        }
        types
    }

    /// Build a minimal GraphQL selection set for the given type names.
    fn build_query_fragment(types: &[String]) -> String {
        let inner: Vec<String> = types
            .iter()
            .map(|t| format!("  {t} {{ __typename id }}"))
            .collect();
        format!("{{\n{}\n}}", inner.join("\n"))
    }

    /// Estimate end-to-end latency as the depth of the dependency chain
    /// multiplied by `base_latency_ms`.
    fn estimate_latency(&self, steps: &[FederationStep]) -> u64 {
        if steps.is_empty() {
            return 0;
        }

        // BFS/DP to find the longest dependency chain
        let mut depth = vec![0usize; steps.len()];
        for (i, step) in steps.iter().enumerate() {
            let max_dep_depth = step.depends_on.iter().map(|&d| depth[d]).max().unwrap_or(0);
            depth[i] = max_dep_depth + 1;
        }

        let max_depth = depth.into_iter().max().unwrap_or(1);
        (max_depth as u64) * self.config.base_latency_ms
    }
}

impl Default for FederationQueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Entity resolver
// ---------------------------------------------------------------------------

/// Resolves cross-graph entity references using the `_entities` query.
///
/// In Apollo Federation, a sub-graph exposes a special `_entities` root field
/// that accepts `representations` (an array of `{ __typename, ...key fields }`)
/// and returns the full entity.
#[derive(Debug, Default)]
pub struct EntityResolver {
    /// Mapping from type name to the sub-graph URL that owns it.
    type_to_url: HashMap<String, String>,
}

impl EntityResolver {
    /// Create a new entity resolver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the URL of the sub-graph that owns `type_name`.
    pub fn register_type(&mut self, type_name: impl Into<String>, url: impl Into<String>) {
        self.type_to_url.insert(type_name.into(), url.into());
    }

    /// Register all types from a slice of `SubGraph` descriptors.
    pub fn register_subgraphs(&mut self, subgraphs: &[SubGraph]) {
        for sg in subgraphs {
            for type_name in &sg.types {
                self.type_to_url.insert(type_name.clone(), sg.url.clone());
            }
        }
    }

    /// Resolve entity representations for `typename`.
    ///
    /// In a real implementation this would dispatch HTTP requests to the
    /// owning sub-graph.  Here we return the representations enriched with
    /// a synthetic `_resolved: true` flag so callers can confirm the call
    /// was routed correctly without needing a live network.
    pub fn resolve_entities(
        &self,
        typename: &str,
        representations: &[serde_json::Value],
    ) -> Vec<serde_json::Value> {
        representations
            .iter()
            .map(|rep| {
                let mut obj = match rep {
                    serde_json::Value::Object(m) => m.clone(),
                    _ => serde_json::Map::new(),
                };
                obj.insert(
                    "__typename".to_string(),
                    serde_json::Value::String(typename.to_string()),
                );
                obj.insert(
                    "_resolved".to_string(),
                    serde_json::Value::Bool(self.type_to_url.contains_key(typename)),
                );
                if let Some(url) = self.type_to_url.get(typename) {
                    obj.insert(
                        "_owning_subgraph".to_string(),
                        serde_json::Value::String(url.clone()),
                    );
                }
                serde_json::Value::Object(obj)
            })
            .collect()
    }

    /// Returns the URL of the sub-graph that owns `typename`, or `None`.
    pub fn owner_url(&self, typename: &str) -> Option<&str> {
        self.type_to_url.get(typename).map(|s| s.as_str())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn accounts_sg() -> SubGraph {
        SubGraph::new("accounts", "https://accounts.example.com/graphql")
            .with_type("User")
            .with_type("Account")
            .with_key(FederationKey {
                type_name: "User".to_string(),
                fields: vec!["id".to_string()],
            })
    }

    fn products_sg() -> SubGraph {
        SubGraph::new("products", "https://products.example.com/graphql")
            .with_type("Product")
            .with_type("Category")
            .with_key(FederationKey {
                type_name: "Product".to_string(),
                fields: vec!["sku".to_string()],
            })
    }

    fn reviews_sg() -> SubGraph {
        SubGraph::new("reviews", "https://reviews.example.com/graphql")
            .with_type("Review")
            .with_key(FederationKey {
                type_name: "Review".to_string(),
                fields: vec!["id".to_string()],
            })
    }

    // --- FederationKey ---

    #[test]
    fn test_federation_key_construction() {
        let key = FederationKey {
            type_name: "User".to_string(),
            fields: vec!["id".to_string()],
        };
        assert_eq!(key.type_name, "User");
        assert_eq!(key.fields, vec!["id"]);
    }

    // --- SubGraph ---

    #[test]
    fn test_subgraph_builder() {
        let sg = accounts_sg();
        assert_eq!(sg.name, "accounts");
        assert!(sg.types.contains(&"User".to_string()));
        assert!(!sg.keys.is_empty());
    }

    // --- FederationQueryPlanner ---

    #[test]
    fn test_plan_empty_query() {
        let planner = FederationQueryPlanner::new();
        let plan = planner.plan_query("{ }", &[accounts_sg(), products_sg()]);
        assert!(plan.steps.is_empty());
    }

    #[test]
    fn test_plan_single_subgraph() {
        let planner = FederationQueryPlanner::new();
        let query = "{\n  User {\n    id name\n  }\n}";
        let plan = planner.plan_query(query, &[accounts_sg(), products_sg()]);
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].subgraph, "accounts");
    }

    #[test]
    fn test_plan_multiple_subgraphs() {
        let planner = FederationQueryPlanner::new();
        let query = "{\n  User {\n    id\n  }\n  Product {\n    sku\n  }\n}";
        let subgraphs = vec![accounts_sg(), products_sg()];
        let plan = planner.plan_query(query, &subgraphs);
        assert_eq!(plan.steps.len(), 2);
        let sg_names: Vec<&str> = plan.steps.iter().map(|s| s.subgraph.as_str()).collect();
        assert!(sg_names.contains(&"accounts"));
        assert!(sg_names.contains(&"products"));
    }

    #[test]
    fn test_plan_parallelizable_when_no_deps() {
        let planner = FederationQueryPlanner::new();
        let query = "{\n  User {\n    id\n  }\n  Product {\n    sku\n  }\n}";
        let plan = planner.plan_query(query, &[accounts_sg(), products_sg()]);
        // Neither User nor Product has cross-graph key deps in this test
        assert!(plan.is_parallelizable);
    }

    #[test]
    fn test_plan_estimated_latency_nonzero() {
        let planner = FederationQueryPlanner::new();
        let query = "{\n  User {\n    id\n  }\n}";
        let plan = planner.plan_query(query, &[accounts_sg()]);
        assert!(plan.estimated_latency_ms > 0);
    }

    #[test]
    fn test_plan_step_contains_query_fragment() {
        let planner = FederationQueryPlanner::new();
        let query = "{\n  Product {\n    sku\n  }\n}";
        let plan = planner.plan_query(query, &[products_sg()]);
        assert!(!plan.steps.is_empty());
        assert!(!plan.steps[0].query.is_empty());
    }

    #[test]
    fn test_plan_is_empty() {
        let plan = FederationPlan {
            steps: vec![],
            estimated_latency_ms: 0,
            is_parallelizable: true,
        };
        assert!(plan.is_empty());
    }

    #[test]
    fn test_plan_unrecognised_type_ignored() {
        let planner = FederationQueryPlanner::new();
        let query = "{\n  Ghost {\n    id\n  }\n}";
        let plan = planner.plan_query(query, &[accounts_sg()]);
        assert!(plan.steps.is_empty());
    }

    #[test]
    fn test_plan_three_subgraphs() {
        let planner = FederationQueryPlanner::new();
        let query =
            "{\n  User {\n    id\n  }\n  Product {\n    sku\n  }\n  Review {\n    id\n  }\n}";
        let plan = planner.plan_query(query, &[accounts_sg(), products_sg(), reviews_sg()]);
        assert_eq!(plan.steps.len(), 3);
    }

    // --- EntityResolver ---

    #[test]
    fn test_entity_resolver_register_type() {
        let mut resolver = EntityResolver::new();
        resolver.register_type("User", "https://accounts.example.com/graphql");
        assert_eq!(
            resolver.owner_url("User"),
            Some("https://accounts.example.com/graphql")
        );
    }

    #[test]
    fn test_entity_resolver_register_subgraphs() {
        let mut resolver = EntityResolver::new();
        resolver.register_subgraphs(&[accounts_sg(), products_sg()]);
        assert!(resolver.owner_url("User").is_some());
        assert!(resolver.owner_url("Product").is_some());
    }

    #[test]
    fn test_resolve_entities_enriches_response() {
        let mut resolver = EntityResolver::new();
        resolver.register_type("User", "https://accounts.example.com/graphql");
        let reps = vec![serde_json::json!({"__typename": "User", "id": "1"})];
        let resolved = resolver.resolve_entities("User", &reps);
        assert_eq!(resolved.len(), 1);
        let obj = resolved[0].as_object().expect("object");
        assert_eq!(obj["__typename"], "User");
        assert_eq!(obj["_resolved"], true);
    }

    #[test]
    fn test_resolve_entities_unknown_type_not_resolved() {
        let resolver = EntityResolver::new();
        let reps = vec![serde_json::json!({"__typename": "Ghost", "id": "99"})];
        let resolved = resolver.resolve_entities("Ghost", &reps);
        let obj = resolved[0].as_object().expect("object");
        assert_eq!(obj["_resolved"], false);
    }

    #[test]
    fn test_resolve_entities_multiple_representations() {
        let mut resolver = EntityResolver::new();
        resolver.register_type("Product", "https://products.example.com/graphql");
        let reps = vec![
            serde_json::json!({"sku": "ABC"}),
            serde_json::json!({"sku": "DEF"}),
            serde_json::json!({"sku": "GHI"}),
        ];
        let resolved = resolver.resolve_entities("Product", &reps);
        assert_eq!(resolved.len(), 3);
        for r in &resolved {
            assert_eq!(r["__typename"], "Product");
            assert_eq!(r["_resolved"], true);
        }
    }

    #[test]
    fn test_owner_url_unregistered_type_returns_none() {
        let resolver = EntityResolver::new();
        assert!(resolver.owner_url("Unknown").is_none());
    }

    #[test]
    fn test_planner_config_latency() {
        let config = FederationPlannerConfig {
            base_latency_ms: 100,
        };
        let planner = FederationQueryPlanner::with_config(config);
        let query = "{\n  User {\n    id\n  }\n}";
        let plan = planner.plan_query(query, &[accounts_sg()]);
        // 1 step, depth 1 → 1 * 100 = 100ms
        assert_eq!(plan.estimated_latency_ms, 100);
    }

    #[test]
    fn test_step_resolves_type_field_populated() {
        let planner = FederationQueryPlanner::new();
        let query = "{\n  User {\n    id\n  }\n}";
        let plan = planner.plan_query(query, &[accounts_sg()]);
        assert!(!plan.steps[0].resolves_type.is_empty());
    }
}
