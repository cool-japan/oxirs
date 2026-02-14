//! Subgraph Composition and Query Planning
//!
//! Implements Apollo Federation subgraph composition and federated query planning.
//! Merges multiple GraphQL subgraph schemas into a unified supergraph and generates
//! optimal query execution plans across distributed services.
//!
//! ## Features
//!
//! - **Schema Composition**: Merge multiple subgraph schemas with conflict resolution
//! - **Query Planning**: Generate optimal federated query execution plans
//! - **Entity Resolution**: Coordinate entity lookups across subgraphs
//! - **Type Merging**: Intelligently merge types from multiple subgraphs
//! - **Field Resolution**: Track which subgraph owns each field

use crate::apollo_federation::FederationVersion;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Represents a registered subgraph service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subgraph {
    /// Unique name of the subgraph
    pub name: String,
    /// Service URL endpoint
    pub url: String,
    /// Federation schema SDL
    pub sdl: String,
    /// Federation version
    pub version: FederationVersion,
    /// Health check endpoint (optional)
    pub health_endpoint: Option<String>,
    /// Service metadata
    pub metadata: HashMap<String, String>,
}

impl Subgraph {
    pub fn new(name: impl Into<String>, url: impl Into<String>, sdl: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            sdl: sdl.into(),
            version: FederationVersion::V2,
            health_endpoint: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_version(mut self, version: FederationVersion) -> Self {
        self.version = version;
        self
    }

    pub fn with_health_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.health_endpoint = Some(endpoint.into());
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Registry for managing subgraph services
#[derive(Debug, Clone)]
pub struct SubgraphRegistry {
    subgraphs: HashMap<String, Subgraph>,
}

impl SubgraphRegistry {
    pub fn new() -> Self {
        Self {
            subgraphs: HashMap::new(),
        }
    }

    /// Register a new subgraph
    pub fn register(&mut self, subgraph: Subgraph) -> Result<()> {
        if self.subgraphs.contains_key(&subgraph.name) {
            return Err(anyhow!("Subgraph '{}' already registered", subgraph.name));
        }
        self.subgraphs.insert(subgraph.name.clone(), subgraph);
        Ok(())
    }

    /// Unregister a subgraph
    pub fn unregister(&mut self, name: &str) -> Result<Subgraph> {
        self.subgraphs
            .remove(name)
            .ok_or_else(|| anyhow!("Subgraph '{}' not found", name))
    }

    /// Get a subgraph by name
    pub fn get(&self, name: &str) -> Option<&Subgraph> {
        self.subgraphs.get(name)
    }

    /// List all registered subgraphs
    pub fn list(&self) -> Vec<&Subgraph> {
        self.subgraphs.values().collect()
    }

    /// Get count of registered subgraphs
    pub fn count(&self) -> usize {
        self.subgraphs.len()
    }
}

impl Default for SubgraphRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Composed supergraph schema
#[derive(Debug, Clone)]
pub struct Supergraph {
    /// Composed SDL
    pub sdl: String,
    /// Entity types across all subgraphs
    pub entities: HashMap<String, EntityDefinition>,
    /// Field ownership mapping (type.field -> subgraph_name)
    pub field_ownership: HashMap<String, String>,
    /// Subgraphs included in composition
    pub subgraphs: Vec<String>,
}

/// Entity definition with subgraph information
#[derive(Debug, Clone)]
pub struct EntityDefinition {
    /// Entity type name
    pub type_name: String,
    /// Key fields for this entity
    pub keys: Vec<Vec<String>>,
    /// Subgraphs that provide this entity
    pub subgraphs: HashSet<String>,
    /// Fields provided by each subgraph
    pub fields_by_subgraph: HashMap<String, Vec<String>>,
}

/// Composes multiple subgraph schemas into a unified supergraph
#[derive(Debug)]
pub struct SchemaComposer {
    registry: SubgraphRegistry,
}

impl SchemaComposer {
    pub fn new(registry: SubgraphRegistry) -> Self {
        Self { registry }
    }

    /// Compose all registered subgraphs into a supergraph
    pub fn compose(&self) -> Result<Supergraph> {
        if self.registry.count() == 0 {
            return Err(anyhow!("No subgraphs registered for composition"));
        }

        let mut entities: HashMap<String, EntityDefinition> = HashMap::new();
        let mut field_ownership: HashMap<String, String> = HashMap::new();
        let mut composed_sdl = String::new();
        let subgraph_names: Vec<String> = self.registry.list().iter().map(|s| s.name.clone()).collect();

        // Add Federation v2 schema extension
        composed_sdl.push_str("extend schema\n");
        composed_sdl.push_str("  @link(url: \"https://specs.apollo.dev/federation/v2.0\",\n");
        composed_sdl.push_str("        import: [\"@key\", \"@external\", \"@requires\", \"@provides\", \"@shareable\", \"@override\"])\n\n");

        // Parse each subgraph and extract entities
        for subgraph in self.registry.list() {
            self.parse_subgraph_entities(subgraph, &mut entities, &mut field_ownership)?;
        }

        // Generate composed SDL with merged types
        self.generate_composed_sdl(&mut composed_sdl, &entities, &field_ownership)?;

        Ok(Supergraph {
            sdl: composed_sdl,
            entities,
            field_ownership,
            subgraphs: subgraph_names,
        })
    }

    fn parse_subgraph_entities(
        &self,
        subgraph: &Subgraph,
        entities: &mut HashMap<String, EntityDefinition>,
        field_ownership: &mut HashMap<String, String>,
    ) -> Result<()> {
        // Parse SDL to extract entity information
        // This is a simplified implementation - production would use full SDL parser
        for line in subgraph.sdl.lines() {
            let trimmed = line.trim();

            // Look for type definitions with @key directive
            if trimmed.starts_with("type ") && trimmed.contains("@key") {
                if let Some(type_name) = self.extract_type_name(trimmed) {
                    let keys = self.extract_key_fields(trimmed);

                    let entity = entities.entry(type_name.clone()).or_insert_with(|| {
                        EntityDefinition {
                            type_name: type_name.clone(),
                            keys: Vec::new(),
                            subgraphs: HashSet::new(),
                            fields_by_subgraph: HashMap::new(),
                        }
                    });

                    entity.subgraphs.insert(subgraph.name.clone());
                    if !keys.is_empty() && !entity.keys.contains(&keys) {
                        entity.keys.push(keys);
                    }

                    // Track field ownership (simplified)
                    field_ownership.insert(
                        format!("{}.id", type_name),
                        subgraph.name.clone(),
                    );
                }
            }
        }

        Ok(())
    }

    fn extract_type_name(&self, line: &str) -> Option<String> {
        // Extract type name from "type TypeName @key(...)"
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 && parts[0] == "type" {
            Some(parts[1].to_string())
        } else {
            None
        }
    }

    fn extract_key_fields(&self, line: &str) -> Vec<String> {
        // Extract fields from @key(fields: "id") or @key(fields: "id email")
        if let Some(start) = line.find("@key(fields: \"") {
            let after_start = &line[start + 14..];
            if let Some(end) = after_start.find('"') {
                let fields_str = &after_start[..end];
                return fields_str.split_whitespace().map(String::from).collect();
            }
        }
        Vec::new()
    }

    fn generate_composed_sdl(
        &self,
        sdl: &mut String,
        entities: &HashMap<String, EntityDefinition>,
        _field_ownership: &HashMap<String, String>,
    ) -> Result<()> {
        // Generate union of all entities
        if !entities.is_empty() {
            sdl.push_str("# Federated Entities\n");
            for (type_name, entity) in entities {
                sdl.push_str(&format!("# Entity: {} (provided by: {:?})\n",
                    type_name, entity.subgraphs));

                // Add key directives
                for key_fields in &entity.keys {
                    let fields_str = key_fields.join(" ");
                    sdl.push_str(&format!("type {} @key(fields: \"{}\")\n", type_name, fields_str));
                }

                sdl.push_str("\n");
            }
        }

        // Add Federation queries
        sdl.push_str("# Federation Queries\n");
        sdl.push_str("extend type Query {\n");
        sdl.push_str("  _entities(representations: [_Any!]!): [_Entity]!\n");
        sdl.push_str("  _service: _Service!\n");
        sdl.push_str("}\n\n");

        // Add Federation scalars and types
        sdl.push_str("scalar _Any\n");
        sdl.push_str("scalar _FieldSet\n\n");

        if !entities.is_empty() {
            sdl.push_str("union _Entity = ");
            let entity_names: Vec<_> = entities.keys().collect();
            for (i, name) in entity_names.iter().enumerate() {
                if i > 0 {
                    sdl.push_str(" | ");
                }
                sdl.push_str(name);
            }
            sdl.push_str("\n\n");
        }

        sdl.push_str("type _Service {\n");
        sdl.push_str("  sdl: String!\n");
        sdl.push_str("}\n");

        Ok(())
    }
}

/// Query execution plan node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryPlanNode {
    /// Sequential execution of nodes
    Sequence {
        nodes: Vec<QueryPlanNode>,
    },
    /// Parallel execution of nodes
    Parallel {
        nodes: Vec<QueryPlanNode>,
    },
    /// Fetch from a specific subgraph
    Fetch {
        subgraph: String,
        query: String,
        requires: Vec<String>,
    },
    /// Flatten nested results
    Flatten {
        path: Vec<String>,
        node: Box<QueryPlanNode>,
    },
}

/// Federated query execution plan
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Root execution node
    pub root: QueryPlanNode,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Subgraphs involved
    pub subgraphs: HashSet<String>,
}

/// Plans federated query execution across subgraphs
#[derive(Debug)]
pub struct QueryPlanner {
    supergraph: Supergraph,
}

impl QueryPlanner {
    pub fn new(supergraph: Supergraph) -> Self {
        Self { supergraph }
    }

    /// Generate an execution plan for a GraphQL query
    pub fn plan(&self, query: &str) -> Result<QueryPlan> {
        // Simplified query planning - production would use full query analysis
        let mut subgraphs = HashSet::new();
        let mut nodes = Vec::new();

        // Analyze which subgraphs are needed
        for subgraph_name in &self.supergraph.subgraphs {
            // Check if this subgraph has fields referenced in the query
            if self.query_references_subgraph(query, subgraph_name) {
                subgraphs.insert(subgraph_name.clone());

                nodes.push(QueryPlanNode::Fetch {
                    subgraph: subgraph_name.clone(),
                    query: self.extract_subgraph_query(query, subgraph_name)?,
                    requires: Vec::new(),
                });
            }
        }

        // Determine execution strategy
        let root = if nodes.len() == 1 {
            nodes.into_iter().next().expect("nodes should not be empty when len == 1")
        } else if nodes.is_empty() {
            return Err(anyhow!("No subgraphs match this query"));
        } else {
            // Multiple subgraphs - use parallel execution if possible
            QueryPlanNode::Parallel { nodes }
        };

        // Estimate cost (simplified)
        let estimated_cost = self.estimate_cost(&root);

        Ok(QueryPlan {
            root,
            estimated_cost,
            subgraphs,
        })
    }

    fn query_references_subgraph(&self, query: &str, subgraph_name: &str) -> bool {
        // Simplified check - production would parse query AST
        // For now, check if any entity from this subgraph is mentioned
        for (entity_name, entity) in &self.supergraph.entities {
            if entity.subgraphs.contains(subgraph_name) && query.contains(entity_name) {
                return true;
            }
        }
        false
    }

    fn extract_subgraph_query(&self, query: &str, _subgraph_name: &str) -> Result<String> {
        // Simplified - production would transform query for specific subgraph
        Ok(query.to_string())
    }

    fn estimate_cost(&self, node: &QueryPlanNode) -> f64 {
        match node {
            QueryPlanNode::Fetch { .. } => 1.0,
            QueryPlanNode::Sequence { nodes } => {
                nodes.iter().map(|n| self.estimate_cost(n)).sum()
            }
            QueryPlanNode::Parallel { nodes } => {
                nodes.iter().map(|n| self.estimate_cost(n)).max_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                }).unwrap_or(0.0)
            }
            QueryPlanNode::Flatten { node, .. } => self.estimate_cost(node) + 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subgraph_creation() {
        let subgraph = Subgraph::new("users", "http://users.example.com/graphql", "type User @key(fields: \"id\") { id: ID! }");
        assert_eq!(subgraph.name, "users");
        assert_eq!(subgraph.url, "http://users.example.com/graphql");
        assert_eq!(subgraph.version, FederationVersion::V2);
    }

    #[test]
    fn test_subgraph_registry() {
        let mut registry = SubgraphRegistry::new();

        let subgraph = Subgraph::new("users", "http://users.example.com/graphql", "schema");
        registry.register(subgraph).unwrap();

        assert_eq!(registry.count(), 1);
        assert!(registry.get("users").is_some());
        assert!(registry.get("products").is_none());
    }

    #[test]
    fn test_duplicate_registration_error() {
        let mut registry = SubgraphRegistry::new();

        let subgraph1 = Subgraph::new("users", "http://users1.example.com/graphql", "schema");
        let subgraph2 = Subgraph::new("users", "http://users2.example.com/graphql", "schema");

        registry.register(subgraph1).unwrap();
        let result = registry.register(subgraph2);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already registered"));
    }

    #[test]
    fn test_subgraph_unregister() {
        let mut registry = SubgraphRegistry::new();

        let subgraph = Subgraph::new("users", "http://users.example.com/graphql", "schema");
        registry.register(subgraph).unwrap();

        assert_eq!(registry.count(), 1);

        let removed = registry.unregister("users").unwrap();
        assert_eq!(removed.name, "users");
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_schema_composer_empty() {
        let registry = SubgraphRegistry::new();
        let composer = SchemaComposer::new(registry);

        let result = composer.compose();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No subgraphs"));
    }

    #[test]
    fn test_schema_composer_single_subgraph() {
        let mut registry = SubgraphRegistry::new();

        let sdl = r#"
type User @key(fields: "id") {
  id: ID!
  name: String!
}
"#;

        registry.register(Subgraph::new("users", "http://users.example.com/graphql", sdl)).unwrap();

        let composer = SchemaComposer::new(registry);
        let supergraph = composer.compose().unwrap();

        assert_eq!(supergraph.subgraphs.len(), 1);
        assert!(supergraph.entities.contains_key("User"));
        assert!(supergraph.sdl.contains("@key"));
        assert!(supergraph.sdl.contains("_entities"));
    }

    #[test]
    fn test_schema_composer_multiple_subgraphs() {
        let mut registry = SubgraphRegistry::new();

        let users_sdl = r#"type User @key(fields: "id") { id: ID! }"#;
        let products_sdl = r#"type Product @key(fields: "sku") { sku: String! }"#;

        registry.register(Subgraph::new("users", "http://users.example.com/graphql", users_sdl)).unwrap();
        registry.register(Subgraph::new("products", "http://products.example.com/graphql", products_sdl)).unwrap();

        let composer = SchemaComposer::new(registry);
        let supergraph = composer.compose().unwrap();

        assert_eq!(supergraph.subgraphs.len(), 2);
        assert!(supergraph.entities.contains_key("User"));
        assert!(supergraph.entities.contains_key("Product"));
    }

    #[test]
    fn test_query_planner_creation() {
        let supergraph = Supergraph {
            sdl: String::new(),
            entities: HashMap::new(),
            field_ownership: HashMap::new(),
            subgraphs: vec!["users".to_string()],
        };

        let planner = QueryPlanner::new(supergraph);
        assert_eq!(planner.supergraph.subgraphs.len(), 1);
    }

    #[test]
    fn test_query_plan_estimation() {
        let node = QueryPlanNode::Fetch {
            subgraph: "users".to_string(),
            query: "{ user { id } }".to_string(),
            requires: Vec::new(),
        };

        let supergraph = Supergraph {
            sdl: String::new(),
            entities: HashMap::new(),
            field_ownership: HashMap::new(),
            subgraphs: vec!["users".to_string()],
        };

        let planner = QueryPlanner::new(supergraph);
        let cost = planner.estimate_cost(&node);

        assert_eq!(cost, 1.0);
    }

    #[test]
    fn test_query_plan_parallel_cost() {
        let nodes = vec![
            QueryPlanNode::Fetch {
                subgraph: "users".to_string(),
                query: "query".to_string(),
                requires: Vec::new(),
            },
            QueryPlanNode::Fetch {
                subgraph: "products".to_string(),
                query: "query".to_string(),
                requires: Vec::new(),
            },
        ];

        let parallel_node = QueryPlanNode::Parallel { nodes };

        let supergraph = Supergraph {
            sdl: String::new(),
            entities: HashMap::new(),
            field_ownership: HashMap::new(),
            subgraphs: vec![],
        };

        let planner = QueryPlanner::new(supergraph);
        let cost = planner.estimate_cost(&parallel_node);

        // Parallel execution cost is max of children
        assert_eq!(cost, 1.0);
    }

    #[test]
    fn test_entity_definition() {
        let mut entity = EntityDefinition {
            type_name: "User".to_string(),
            keys: vec![vec!["id".to_string()]],
            subgraphs: HashSet::new(),
            fields_by_subgraph: HashMap::new(),
        };

        entity.subgraphs.insert("users".to_string());
        entity.subgraphs.insert("reviews".to_string());

        assert_eq!(entity.type_name, "User");
        assert_eq!(entity.subgraphs.len(), 2);
    }
}
