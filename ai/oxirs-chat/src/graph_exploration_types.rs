//! Types for graph exploration: states, paths, configs, results
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for graph exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationConfig {
    /// Maximum depth for path discovery
    pub max_depth: usize,
    /// Maximum number of paths to explore
    pub max_paths: usize,
    /// Maximum number of neighbors to expand per entity
    pub max_neighbors: usize,
    /// Minimum relevance score for path inclusion
    pub min_relevance_score: f32,
    /// Enable schema-aware filtering
    pub schema_aware: bool,
    /// Preferred relationship types for exploration
    pub preferred_relationships: Vec<String>,
    /// Blacklisted relationship types to avoid
    pub blacklisted_relationships: Vec<String>,
}

impl Default for ExplorationConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            max_paths: 100,
            max_neighbors: 20,
            min_relevance_score: 0.1,
            schema_aware: true,
            preferred_relationships: vec![
                "http://www.w3.org/2000/01/rdf-schema#subClassOf".to_string(),
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                "http://xmlns.com/foaf/0.1/knows".to_string(),
                "http://purl.org/dc/elements/1.1/creator".to_string(),
            ],
            blacklisted_relationships: vec!["http://www.w3.org/2002/07/owl#sameAs".to_string()],
        }
    }
}

/// A path through the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPath {
    /// Entities in the path
    pub entities: Vec<String>,
    /// Relationships connecting the entities
    pub relationships: Vec<String>,
    /// Path length (number of hops)
    pub length: usize,
    /// Relevance score for this path
    pub relevance_score: f32,
    /// Explanation of why this path is relevant
    pub explanation: String,
    /// Metadata about the path
    pub metadata: HashMap<String, String>,
}

/// Information about an expanded entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpandedEntity {
    /// The entity URI
    pub entity: String,
    /// Direct neighbors
    pub neighbors: Vec<EntityNeighbor>,
    /// Entity types
    pub types: Vec<String>,
    /// Properties and their values
    pub properties: HashMap<String, Vec<String>>,
    /// Relevance score
    pub relevance_score: f32,
    /// Schema information
    pub schema_info: Option<SchemaInfo>,
}

/// Information about a neighboring entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityNeighbor {
    /// Neighbor entity URI
    pub entity: String,
    /// Relationship connecting to this neighbor
    pub relationship: String,
    /// Direction of relationship (outgoing/incoming)
    pub direction: RelationshipDirection,
    /// Strength/weight of this relationship
    pub strength: f32,
    /// Labels or human-readable names
    pub labels: Vec<String>,
}

/// Direction of a relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipDirection {
    Outgoing,
    Incoming,
    Bidirectional,
}

/// Schema information for an entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaInfo {
    /// Classes this entity belongs to
    pub classes: Vec<String>,
    /// Domain restrictions
    pub domain_restrictions: Vec<String>,
    /// Range restrictions
    pub range_restrictions: Vec<String>,
    /// Cardinality constraints
    pub cardinality_constraints: HashMap<String, (Option<u32>, Option<u32>)>,
    /// Functional properties
    pub functional_properties: Vec<String>,
    /// Equivalent classes
    pub equivalent_classes: Vec<String>,
    /// Disjoint classes
    pub disjoint_classes: Vec<String>,
    /// SHACL shapes
    pub shacl_shapes: Vec<ShaclShape>,
}

/// SHACL shape definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaclShape {
    pub shape_id: String,
    pub target_class: Option<String>,
    pub property_shapes: Vec<PropertyShape>,
    pub constraints: Vec<ShapeConstraint>,
}

/// SHACL property shape
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyShape {
    pub path: String,
    pub datatype: Option<String>,
    pub min_count: Option<u32>,
    pub max_count: Option<u32>,
    pub node_kind: Option<String>,
    pub pattern: Option<String>,
}

/// SHACL constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeConstraint {
    pub constraint_type: String,
    pub value: String,
    pub message: Option<String>,
}

/// Query guidance suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryGuidance {
    pub suggestion_type: GuidanceType,
    pub title: String,
    pub description: String,
    pub sparql_template: String,
    pub confidence: f32,
    pub schema_rationale: String,
}

/// Types of query guidance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GuidanceType {
    ValidPropertyPath,
    TypeConstraint,
    CardinalityAwareness,
    BestPractice,
    ConsistencyCheck,
    SchemaRecommendation,
}

/// Exploration suggestion for interactive navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationSuggestion {
    pub suggestion_type: SuggestionType,
    pub title: String,
    pub description: String,
    pub action: ExplorationAction,
    pub confidence: f32,
}

/// Types of exploration suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestionType {
    ExploreNeighbor,
    ExploreType,
    DiscoverPaths,
    FindSimilar,
    SchemaAnalysis,
}

/// Actions that can be taken from exploration suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplorationAction {
    NavigateToEntity(String),
    FindSimilarEntities(String),
    DiscoverConnections(String),
    AnalyzeSchema(String),
    ExecuteQuery(String),
}

/// Related entity found through similarity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedEntity {
    pub entity: String,
    pub similarity_score: f32,
    pub relationship_type: String,
    pub explanation: String,
}

/// Schema validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub suggestions: Vec<String>,
}

/// Class hierarchy analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassHierarchyAnalysis {
    pub class: String,
    pub superclasses: Vec<String>,
    pub subclasses: Vec<String>,
    pub equivalent_classes: Vec<String>,
    pub disjoint_classes: Vec<String>,
    pub depth_from_root: u32,
    pub instance_count: u32,
}

/// SHACL shape validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeValidationResult {
    pub is_valid: bool,
    pub violations: Vec<ShapeViolation>,
    pub satisfied_shapes: Vec<String>,
}

/// SHACL shape violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeViolation {
    pub shape_id: String,
    pub property_path: String,
    pub violation_type: String,
    pub message: String,
    pub severity: ViolationSeverity,
}

/// Severity levels for SHACL violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Info,
    Warning,
    Violation,
}

/// Graph exploration results aggregator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationResults {
    pub paths: Vec<GraphPath>,
    pub expanded_entities: Vec<ExpandedEntity>,
    pub schema_suggestions: Vec<String>,
    pub exploration_metadata: HashMap<String, String>,
}

impl ExplorationResults {
    pub fn new() -> Self {
        Self {
            paths: Vec::new(),
            expanded_entities: Vec::new(),
            schema_suggestions: Vec::new(),
            exploration_metadata: HashMap::new(),
        }
    }

    /// Add exploration metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.exploration_metadata.insert(key, value);
    }

    /// Get a summary of the exploration results
    pub fn get_summary(&self) -> String {
        format!(
            "Exploration Results: {} paths found, {} entities expanded, {} schema suggestions",
            self.paths.len(),
            self.expanded_entities.len(),
            self.schema_suggestions.len()
        )
    }

    /// Convert to JSON for API responses
    pub fn to_json(&self) -> anyhow::Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| anyhow::anyhow!("JSON serialization failed: {}", e))
    }
}

impl Default for ExplorationResults {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal state for path discovery
#[derive(Debug, Clone)]
pub(crate) struct PathState {
    pub current_entity: String,
    pub path_entities: Vec<String>,
    pub path_relationships: Vec<String>,
    pub depth: usize,
    pub score: f32,
}

/// Ordered wrapper for PathState to use in priority queue
#[derive(Debug, Clone)]
pub(crate) struct PathStateOrdered {
    pub state: PathState,
}

impl PartialEq for PathStateOrdered {
    fn eq(&self, other: &Self) -> bool {
        self.state.score == other.state.score
    }
}

impl Eq for PathStateOrdered {}

impl PartialOrd for PathStateOrdered {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PathStateOrdered {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.state
            .score
            .partial_cmp(&other.state.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}
