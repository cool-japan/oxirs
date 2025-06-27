//! Semantic-aware sharding for distributed RDF storage
//!
//! This module implements intelligent sharding that keeps semantically related
//! RDF data together to minimize cross-shard queries and improve performance.

use crate::model::{Triple, NamedNode, Subject, Predicate, Object};
use crate::store::IndexedGraph;
use anyhow::{anyhow, Result};
use dashmap::DashMap;
use parking_lot::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Sharding configuration
#[derive(Debug, Clone)]
pub struct ShardingConfig {
    /// Number of shards
    pub shard_count: usize,
    
    /// Replication factor
    pub replication_factor: usize,
    
    /// Enable semantic-aware partitioning
    pub semantic_partitioning: bool,
    
    /// Maximum shard size (number of triples)
    pub max_shard_size: usize,
    
    /// Enable dynamic rebalancing
    pub enable_rebalancing: bool,
    
    /// Rebalancing threshold (imbalance ratio)
    pub rebalancing_threshold: f64,
}

impl Default for ShardingConfig {
    fn default() -> Self {
        Self {
            shard_count: 16,
            replication_factor: 3,
            semantic_partitioning: true,
            max_shard_size: 10_000_000,
            enable_rebalancing: true,
            rebalancing_threshold: 0.2,
        }
    }
}

/// Shard identifier
pub type ShardId = u32;

/// Node identifier
pub type NodeId = u64;

/// Sharding strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardingStrategy {
    /// Hash-based sharding (simple but no semantic awareness)
    Hash,
    
    /// Subject-based sharding (keeps all triples with same subject together)
    Subject,
    
    /// Predicate-based sharding (groups by predicate type)
    Predicate,
    
    /// Graph-based sharding (for named graphs)
    Graph,
    
    /// Semantic sharding (intelligent grouping based on relationships)
    Semantic(SemanticStrategy),
    
    /// Hybrid strategy combining multiple approaches
    Hybrid {
        primary: Box<ShardingStrategy>,
        secondary: Box<ShardingStrategy>,
    },
}

/// Semantic sharding strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticStrategy {
    /// Entity types to keep together
    pub entity_groups: HashMap<String, Vec<String>>,
    
    /// Predicates that indicate strong relationships
    pub relationship_predicates: Vec<String>,
    
    /// Namespace-based grouping
    pub namespace_groups: HashMap<String, ShardId>,
    
    /// Class hierarchy for grouping
    pub class_hierarchy: HashMap<String, String>,
}

impl Default for SemanticStrategy {
    fn default() -> Self {
        let mut entity_groups = HashMap::new();
        entity_groups.insert(
            "Person".to_string(),
            vec!["name".to_string(), "email".to_string(), "address".to_string()],
        );
        entity_groups.insert(
            "Organization".to_string(),
            vec!["name".to_string(), "location".to_string(), "employees".to_string()],
        );
        
        let relationship_predicates = vec![
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#subClassOf".to_string(),
            "http://www.w3.org/2002/07/owl#sameAs".to_string(),
        ];
        
        Self {
            entity_groups,
            relationship_predicates,
            namespace_groups: HashMap::new(),
            class_hierarchy: HashMap::new(),
        }
    }
}

/// Shard manager for distributed RDF storage
pub struct ShardManager {
    /// Configuration
    config: ShardingConfig,
    
    /// Sharding strategy
    strategy: ShardingStrategy,
    
    /// Shard metadata
    shard_metadata: Arc<RwLock<HashMap<ShardId, ShardMetadata>>>,
    
    /// Shard assignments (which nodes host which shards)
    shard_assignments: Arc<RwLock<HashMap<ShardId, Vec<NodeId>>>>,
    
    /// Entity to shard mapping for semantic sharding
    entity_shard_map: Arc<DashMap<String, ShardId>>,
    
    /// Statistics for each shard
    shard_stats: Arc<DashMap<ShardId, ShardStatistics>>,
    
    /// Pending migrations
    pending_migrations: Arc<Mutex<VecDeque<Migration>>>,
    
    /// Local shards (shards hosted on this node)
    local_shards: Arc<DashMap<ShardId, IndexedGraph>>,
}

/// Shard metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMetadata {
    pub id: ShardId,
    pub version: u64,
    pub triple_count: usize,
    pub size_bytes: usize,
    pub created_at: std::time::SystemTime,
    pub last_modified: std::time::SystemTime,
    pub primary_node: NodeId,
    pub replica_nodes: Vec<NodeId>,
}

/// Shard statistics
#[derive(Debug, Clone, Default)]
pub struct ShardStatistics {
    pub read_count: u64,
    pub write_count: u64,
    pub query_latency_ms: f64,
    pub hot_entities: Vec<String>,
    pub access_pattern: AccessPattern,
}

/// Access pattern for a shard
#[derive(Debug, Clone, Default)]
pub struct AccessPattern {
    pub read_heavy: bool,
    pub write_heavy: bool,
    pub temporal_locality: f64,
    pub spatial_locality: f64,
}

/// Migration operation
#[derive(Debug, Clone)]
pub struct Migration {
    pub shard_id: ShardId,
    pub from_node: NodeId,
    pub to_node: NodeId,
    pub triples: Vec<Triple>,
    pub reason: MigrationReason,
}

/// Reason for migration
#[derive(Debug, Clone)]
pub enum MigrationReason {
    /// Load balancing
    LoadBalance,
    /// Node failure
    NodeFailure,
    /// Manual rebalancing
    Manual,
    /// Semantic regrouping
    SemanticOptimization,
}

impl ShardManager {
    /// Create a new shard manager
    pub fn new(config: ShardingConfig, strategy: ShardingStrategy) -> Self {
        let mut shard_metadata = HashMap::new();
        let mut shard_assignments = HashMap::new();
        
        // Initialize shards
        for shard_id in 0..config.shard_count {
            let metadata = ShardMetadata {
                id: shard_id as ShardId,
                version: 0,
                triple_count: 0,
                size_bytes: 0,
                created_at: std::time::SystemTime::now(),
                last_modified: std::time::SystemTime::now(),
                primary_node: 0, // Will be assigned later
                replica_nodes: vec![],
            };
            shard_metadata.insert(shard_id as ShardId, metadata);
            shard_assignments.insert(shard_id as ShardId, vec![]);
        }
        
        Self {
            config,
            strategy,
            shard_metadata: Arc::new(RwLock::new(shard_metadata)),
            shard_assignments: Arc::new(RwLock::new(shard_assignments)),
            entity_shard_map: Arc::new(DashMap::new()),
            shard_stats: Arc::new(DashMap::new()),
            pending_migrations: Arc::new(Mutex::new(VecDeque::new())),
            local_shards: Arc::new(DashMap::new()),
        }
    }
    
    /// Determine which shard a triple belongs to
    pub fn get_shard_for_triple(&self, triple: &Triple) -> ShardId {
        match &self.strategy {
            ShardingStrategy::Hash => self.hash_shard(triple),
            ShardingStrategy::Subject => self.subject_shard(triple),
            ShardingStrategy::Predicate => self.predicate_shard(triple),
            ShardingStrategy::Graph => self.graph_shard(triple),
            ShardingStrategy::Semantic(strategy) => self.semantic_shard(triple, strategy),
            ShardingStrategy::Hybrid { primary, secondary } => {
                // Try primary strategy first, fallback to secondary
                let primary_shard = self.get_shard_with_strategy(triple, primary);
                if self.is_shard_overloaded(primary_shard) {
                    self.get_shard_with_strategy(triple, secondary)
                } else {
                    primary_shard
                }
            }
        }
    }
    
    /// Get shard using specific strategy
    fn get_shard_with_strategy(&self, triple: &Triple, strategy: &ShardingStrategy) -> ShardId {
        match strategy {
            ShardingStrategy::Hash => self.hash_shard(triple),
            ShardingStrategy::Subject => self.subject_shard(triple),
            ShardingStrategy::Predicate => self.predicate_shard(triple),
            ShardingStrategy::Graph => self.graph_shard(triple),
            ShardingStrategy::Semantic(s) => self.semantic_shard(triple, s),
            ShardingStrategy::Hybrid { primary, .. } => {
                self.get_shard_with_strategy(triple, primary)
            }
        }
    }
    
    /// Hash-based sharding
    fn hash_shard(&self, triple: &Triple) -> ShardId {
        let mut hasher = DefaultHasher::new();
        triple.subject().to_string().hash(&mut hasher);
        triple.predicate().to_string().hash(&mut hasher);
        triple.object().to_string().hash(&mut hasher);
        (hasher.finish() % self.config.shard_count as u64) as ShardId
    }
    
    /// Subject-based sharding
    fn subject_shard(&self, triple: &Triple) -> ShardId {
        let mut hasher = DefaultHasher::new();
        triple.subject().to_string().hash(&mut hasher);
        (hasher.finish() % self.config.shard_count as u64) as ShardId
    }
    
    /// Predicate-based sharding
    fn predicate_shard(&self, triple: &Triple) -> ShardId {
        let mut hasher = DefaultHasher::new();
        triple.predicate().to_string().hash(&mut hasher);
        (hasher.finish() % self.config.shard_count as u64) as ShardId
    }
    
    /// Graph-based sharding (simplified for default graph)
    fn graph_shard(&self, _triple: &Triple) -> ShardId {
        // In a real implementation, would use quad's graph component
        0
    }
    
    /// Semantic-aware sharding
    fn semantic_shard(&self, triple: &Triple, strategy: &SemanticStrategy) -> ShardId {
        let subject_str = triple.subject().to_string();
        
        // Check if we already have a mapping for this entity
        if let Some(shard) = self.entity_shard_map.get(&subject_str) {
            return *shard;
        }
        
        // Check namespace-based grouping
        for (namespace, shard_id) in &strategy.namespace_groups {
            if subject_str.starts_with(namespace) {
                self.entity_shard_map.insert(subject_str.clone(), *shard_id);
                return *shard_id;
            }
        }
        
        // Check if this is a relationship predicate
        let predicate_str = triple.predicate().to_string();
        if strategy.relationship_predicates.contains(&predicate_str) {
            // Keep related entities on the same shard
            if let Some(object_shard) = self.get_object_shard(triple.object()) {
                self.entity_shard_map.insert(subject_str.clone(), object_shard);
                return object_shard;
            }
        }
        
        // Check entity groups
        for (entity_type, properties) in &strategy.entity_groups {
            if predicate_str.contains(entity_type) || properties.iter().any(|p| predicate_str.contains(p)) {
                // Group entities of the same type together
                let mut hasher = DefaultHasher::new();
                entity_type.hash(&mut hasher);
                let shard = (hasher.finish() % self.config.shard_count as u64) as ShardId;
                self.entity_shard_map.insert(subject_str.clone(), shard);
                return shard;
            }
        }
        
        // Fallback to hash-based sharding
        let shard = self.hash_shard(triple);
        self.entity_shard_map.insert(subject_str, shard);
        shard
    }
    
    /// Get shard for an object if it's an entity
    fn get_object_shard(&self, object: &Object) -> Option<ShardId> {
        match object {
            Object::NamedNode(node) => {
                let node_str = node.as_str().to_string();
                self.entity_shard_map.get(&node_str).map(|s| *s)
            }
            Object::BlankNode(node) => {
                let node_str = node.as_str().to_string();
                self.entity_shard_map.get(&node_str).map(|s| *s)
            }
            _ => None,
        }
    }
    
    /// Check if a shard is overloaded
    fn is_shard_overloaded(&self, shard_id: ShardId) -> bool {
        if let Some(metadata) = self.shard_metadata.read().get(&shard_id) {
            metadata.triple_count > self.config.max_shard_size
        } else {
            false
        }
    }
    
    /// Insert a triple into the appropriate shard
    pub fn insert_triple(&self, triple: Triple) -> Result<()> {
        let shard_id = self.get_shard_for_triple(&triple);
        
        // Update local shard if we have it
        if let Some(mut shard) = self.local_shards.get_mut(&shard_id) {
            shard.insert(&triple);
            
            // Update statistics
            if let Some(mut stats) = self.shard_stats.get_mut(&shard_id) {
                stats.write_count += 1;
            }
            
            // Update metadata
            self.update_shard_metadata(shard_id, 1, 0);
        } else {
            // Forward to remote shard
            // In a real implementation, would send to remote node
            return Err(anyhow!("Shard {} not available locally", shard_id));
        }
        
        Ok(())
    }
    
    /// Query triples from shards
    pub fn query_triples(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> Result<Vec<Triple>> {
        let mut results = Vec::new();
        
        // Determine which shards to query
        let shards_to_query = self.get_shards_for_query(subject, predicate, object);
        
        // Query each relevant shard
        for shard_id in shards_to_query {
            if let Some(shard) = self.local_shards.get(&shard_id) {
                let shard_results = shard.match_pattern(subject, predicate, object);
                results.extend(shard_results);
                
                // Update statistics
                if let Some(mut stats) = self.shard_stats.get_mut(&shard_id) {
                    stats.read_count += 1;
                }
            }
        }
        
        Ok(results)
    }
    
    /// Determine which shards to query based on pattern
    fn get_shards_for_query(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        _object: Option<&Object>,
    ) -> Vec<ShardId> {
        // If we have a specific subject, we can route to specific shard
        if let Some(subj) = subject {
            let triple = Triple::new(
                subj.clone(),
                predicate.cloned().unwrap_or_else(|| Predicate::NamedNode(
                    NamedNode::new("http://example.org/dummy").unwrap()
                )),
                Object::NamedNode(NamedNode::new("http://example.org/dummy").unwrap()),
            );
            vec![self.get_shard_for_triple(&triple)]
        } else if let Some(pred) = predicate {
            // For predicate queries, might need to query multiple shards
            match &self.strategy {
                ShardingStrategy::Predicate => {
                    // Can route to specific shard
                    let mut hasher = DefaultHasher::new();
                    pred.to_string().hash(&mut hasher);
                    vec![(hasher.finish() % self.config.shard_count as u64) as ShardId]
                }
                _ => {
                    // Need to query all shards
                    (0..self.config.shard_count).map(|i| i as ShardId).collect()
                }
            }
        } else {
            // Full scan - query all shards
            (0..self.config.shard_count).map(|i| i as ShardId).collect()
        }
    }
    
    /// Update shard metadata
    fn update_shard_metadata(&self, shard_id: ShardId, triple_delta: i64, size_delta: i64) {
        let mut metadata = self.shard_metadata.write();
        if let Some(shard_meta) = metadata.get_mut(&shard_id) {
            if triple_delta > 0 {
                shard_meta.triple_count += triple_delta as usize;
            } else {
                shard_meta.triple_count = shard_meta.triple_count.saturating_sub((-triple_delta) as usize);
            }
            
            if size_delta > 0 {
                shard_meta.size_bytes += size_delta as usize;
            } else {
                shard_meta.size_bytes = shard_meta.size_bytes.saturating_sub((-size_delta) as usize);
            }
            
            shard_meta.last_modified = std::time::SystemTime::now();
            shard_meta.version += 1;
        }
    }
    
    /// Check if rebalancing is needed
    pub fn needs_rebalancing(&self) -> bool {
        if !self.config.enable_rebalancing {
            return false;
        }
        
        let metadata = self.shard_metadata.read();
        if metadata.is_empty() {
            return false;
        }
        
        let sizes: Vec<usize> = metadata.values().map(|m| m.triple_count).collect();
        let avg_size = sizes.iter().sum::<usize>() / sizes.len();
        let max_size = sizes.iter().max().copied().unwrap_or(0);
        let min_size = sizes.iter().min().copied().unwrap_or(0);
        
        // Check imbalance ratio
        if avg_size > 0 {
            let imbalance = (max_size as f64 - min_size as f64) / avg_size as f64;
            imbalance > self.config.rebalancing_threshold
        } else {
            false
        }
    }
    
    /// Plan rebalancing operations
    pub fn plan_rebalancing(&self) -> Vec<Migration> {
        let mut migrations = Vec::new();
        
        let metadata = self.shard_metadata.read();
        let mut shard_sizes: Vec<(ShardId, usize)> = metadata
            .iter()
            .map(|(id, meta)| (*id, meta.triple_count))
            .collect();
        
        // Sort by size
        shard_sizes.sort_by_key(|&(_, size)| size);
        
        let avg_size = shard_sizes.iter().map(|(_, size)| size).sum::<usize>() / shard_sizes.len();
        
        // Find overloaded and underloaded shards
        let overloaded: Vec<_> = shard_sizes
            .iter()
            .filter(|(_, size)| *size > avg_size + (avg_size as f64 * self.config.rebalancing_threshold) as usize)
            .collect();
        
        let underloaded: Vec<_> = shard_sizes
            .iter()
            .filter(|(_, size)| *size < avg_size - (avg_size as f64 * self.config.rebalancing_threshold) as usize)
            .collect();
        
        // Plan migrations from overloaded to underloaded
        for (over_shard, over_size) in overloaded {
            for (under_shard, under_size) in &underloaded {
                let to_move = (*over_size - avg_size).min(avg_size - *under_size);
                if to_move > 0 {
                    // In a real implementation, would select specific triples to move
                    let migration = Migration {
                        shard_id: *over_shard,
                        from_node: 0, // Would get from shard assignments
                        to_node: 0,   // Would get from shard assignments
                        triples: vec![], // Would select actual triples
                        reason: MigrationReason::LoadBalance,
                    };
                    migrations.push(migration);
                }
            }
        }
        
        migrations
    }
    
    /// Execute a migration
    pub async fn execute_migration(&self, migration: &Migration) -> Result<()> {
        // In a real implementation, would:
        // 1. Lock affected shards
        // 2. Copy triples to destination
        // 3. Verify successful copy
        // 4. Remove from source
        // 5. Update metadata and mappings
        // 6. Unlock shards
        
        // For now, just update metadata
        self.update_shard_metadata(migration.shard_id, -(migration.triples.len() as i64), 0);
        
        Ok(())
    }
    
    /// Get shard statistics
    pub fn get_shard_statistics(&self) -> HashMap<ShardId, ShardStatistics> {
        self.shard_stats
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect()
    }
    
    /// Get load distribution across shards
    pub fn get_load_distribution(&self) -> HashMap<ShardId, f64> {
        let total_ops: u64 = self.shard_stats
            .iter()
            .map(|entry| entry.value().read_count + entry.value().write_count)
            .sum();
        
        if total_ops == 0 {
            return HashMap::new();
        }
        
        self.shard_stats
            .iter()
            .map(|entry| {
                let shard_ops = entry.value().read_count + entry.value().write_count;
                (*entry.key(), shard_ops as f64 / total_ops as f64)
            })
            .collect()
    }
}

/// Shard router for query optimization
pub struct ShardRouter {
    manager: Arc<ShardManager>,
}

impl ShardRouter {
    /// Create a new shard router
    pub fn new(manager: Arc<ShardManager>) -> Self {
        Self { manager }
    }
    
    /// Route a SPARQL query to appropriate shards
    pub fn route_query(&self, query: &str) -> Result<Vec<ShardId>> {
        // In a real implementation, would parse SPARQL and analyze patterns
        // For now, return all shards for complex queries
        Ok((0..self.manager.config.shard_count).map(|i| i as ShardId).collect())
    }
    
    /// Optimize query plan for distributed execution
    pub fn optimize_distributed_query(&self, query: &str) -> Result<DistributedQueryPlan> {
        // Simplified implementation
        let shards = self.route_query(query)?;
        
        Ok(DistributedQueryPlan {
            query: query.to_string(),
            shard_operations: shards
                .into_iter()
                .map(|shard| ShardOperation {
                    shard_id: shard,
                    operation: query.to_string(),
                    estimated_cost: 1.0,
                })
                .collect(),
            merge_strategy: MergeStrategy::Union,
        })
    }
}

/// Distributed query plan
#[derive(Debug, Clone)]
pub struct DistributedQueryPlan {
    pub query: String,
    pub shard_operations: Vec<ShardOperation>,
    pub merge_strategy: MergeStrategy,
}

/// Operation on a specific shard
#[derive(Debug, Clone)]
pub struct ShardOperation {
    pub shard_id: ShardId,
    pub operation: String,
    pub estimated_cost: f64,
}

/// Strategy for merging results from multiple shards
#[derive(Debug, Clone)]
pub enum MergeStrategy {
    /// Simple union of results
    Union,
    /// Intersection of results
    Intersection,
    /// Join operation
    Join { join_key: String },
    /// Aggregation
    Aggregate { group_by: Vec<String> },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode, Triple};
    
    #[test]
    fn test_hash_sharding() {
        let config = ShardingConfig::default();
        let manager = ShardManager::new(config, ShardingStrategy::Hash);
        
        let triple1 = Triple::new(
            NamedNode::new("http://example.org/s1").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            Literal::new("value1"),
        );
        
        let triple2 = Triple::new(
            NamedNode::new("http://example.org/s2").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            Literal::new("value2"),
        );
        
        let shard1 = manager.get_shard_for_triple(&triple1);
        let shard2 = manager.get_shard_for_triple(&triple2);
        
        // Different subjects should likely go to different shards
        // (not guaranteed with hash, but likely with enough shards)
        assert!(shard1 < 16);
        assert!(shard2 < 16);
    }
    
    #[test]
    fn test_subject_sharding() {
        let config = ShardingConfig::default();
        let manager = ShardManager::new(config, ShardingStrategy::Subject);
        
        let subject = NamedNode::new("http://example.org/entity1").unwrap();
        
        let triple1 = Triple::new(
            subject.clone(),
            NamedNode::new("http://example.org/p1").unwrap(),
            Literal::new("value1"),
        );
        
        let triple2 = Triple::new(
            subject.clone(),
            NamedNode::new("http://example.org/p2").unwrap(),
            Literal::new("value2"),
        );
        
        let shard1 = manager.get_shard_for_triple(&triple1);
        let shard2 = manager.get_shard_for_triple(&triple2);
        
        // Same subject should go to same shard
        assert_eq!(shard1, shard2);
    }
    
    #[test]
    fn test_semantic_sharding() {
        let config = ShardingConfig::default();
        let strategy = SemanticStrategy::default();
        let manager = ShardManager::new(config, ShardingStrategy::Semantic(strategy));
        
        let person = NamedNode::new("http://example.org/person1").unwrap();
        
        let triple1 = Triple::new(
            person.clone(),
            NamedNode::new("http://example.org/name").unwrap(),
            Literal::new("John"),
        );
        
        let triple2 = Triple::new(
            person.clone(),
            NamedNode::new("http://example.org/email").unwrap(),
            Literal::new("john@example.org"),
        );
        
        let shard1 = manager.get_shard_for_triple(&triple1);
        let shard2 = manager.get_shard_for_triple(&triple2);
        
        // Same entity should stay on same shard
        assert_eq!(shard1, shard2);
    }
    
    #[test]
    fn test_rebalancing_detection() {
        let config = ShardingConfig {
            shard_count: 4,
            rebalancing_threshold: 0.2,
            ..Default::default()
        };
        let manager = ShardManager::new(config, ShardingStrategy::Hash);
        
        // Initially balanced
        assert!(!manager.needs_rebalancing());
        
        // Simulate imbalance
        manager.update_shard_metadata(0, 1000, 0);
        manager.update_shard_metadata(1, 100, 0);
        
        // Should need rebalancing now
        assert!(manager.needs_rebalancing());
    }
    
    #[test]
    fn test_query_routing() {
        let config = ShardingConfig::default();
        let manager = Arc::new(ShardManager::new(config, ShardingStrategy::Subject));
        let router = ShardRouter::new(manager);
        
        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        let shards = router.route_query(query).unwrap();
        
        // Full scan should query all shards
        assert_eq!(shards.len(), 16);
    }
}