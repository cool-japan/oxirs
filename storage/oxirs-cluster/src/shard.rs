//! Sharding with semantic-aware partitioning
//!
//! This module implements intelligent sharding that partitions RDF data
//! based on semantic relationships rather than simple hash-based distribution.

use crate::{ClusterError, Result};
use oxirs_core::model::{NamedNode, Triple};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Shard identifier
pub type ShardId = u32;

/// Sharding strategy for partitioning data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardingStrategy {
    /// Hash-based sharding (default)
    Hash {
        num_shards: u32,
    },
    /// Subject-based sharding (all triples with same subject go to same shard)
    Subject {
        num_shards: u32,
    },
    /// Predicate-based sharding (group by predicate types)
    Predicate {
        predicate_groups: HashMap<String, ShardId>,
    },
    /// Namespace-based sharding (group by IRI namespace)
    Namespace {
        namespace_mapping: HashMap<String, ShardId>,
    },
    /// Graph-based sharding (each named graph to specific shard)
    Graph {
        graph_mapping: HashMap<String, ShardId>,
    },
    /// Semantic clustering (group related concepts together)
    Semantic {
        concept_clusters: Vec<ConceptCluster>,
        similarity_threshold: f64,
    },
    /// Hybrid strategy combining multiple approaches
    Hybrid {
        primary: Box<ShardingStrategy>,
        secondary: Box<ShardingStrategy>,
    },
}

/// Concept cluster for semantic sharding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptCluster {
    /// Cluster identifier
    pub cluster_id: ShardId,
    /// Core concepts in this cluster
    pub core_concepts: HashSet<String>,
    /// Related predicates
    pub predicates: HashSet<String>,
    /// Namespace patterns
    pub namespace_patterns: Vec<String>,
    /// Weight for routing decisions
    pub weight: f64,
}

/// Shard metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMetadata {
    /// Shard identifier
    pub shard_id: ShardId,
    /// Node IDs responsible for this shard
    pub node_ids: Vec<u64>,
    /// Primary node for this shard
    pub primary_node: u64,
    /// Number of triples in this shard
    pub triple_count: usize,
    /// Size in bytes
    pub size_bytes: u64,
    /// Shard state
    pub state: ShardState,
    /// Last updated timestamp
    pub last_updated: u64,
}

/// Shard state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardState {
    /// Shard is active and serving requests
    Active,
    /// Shard is being migrated
    Migrating,
    /// Shard is being split
    Splitting,
    /// Shard is being merged
    Merging,
    /// Shard is offline
    Offline,
}

/// Shard router for determining shard placement
pub struct ShardRouter {
    /// Sharding strategy
    strategy: ShardingStrategy,
    /// Shard metadata
    shards: Arc<RwLock<HashMap<ShardId, ShardMetadata>>>,
    /// Concept similarity calculator
    similarity_calc: Option<Arc<dyn ConceptSimilarity>>,
    /// Cache for routing decisions
    routing_cache: Arc<RwLock<HashMap<String, ShardId>>>,
}

/// Trait for calculating concept similarity
pub trait ConceptSimilarity: Send + Sync {
    /// Calculate similarity between two concepts
    fn similarity(&self, concept1: &str, concept2: &str) -> f64;
    
    /// Find most similar concept cluster
    fn find_cluster(&self, concept: &str, clusters: &[ConceptCluster]) -> Option<ShardId>;
}

impl ShardRouter {
    /// Create a new shard router
    pub fn new(strategy: ShardingStrategy) -> Self {
        Self {
            strategy,
            shards: Arc::new(RwLock::new(HashMap::new())),
            similarity_calc: None,
            routing_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Set concept similarity calculator
    pub fn with_similarity_calculator(mut self, calc: Arc<dyn ConceptSimilarity>) -> Self {
        self.similarity_calc = Some(calc);
        self
    }
    
    /// Initialize shards
    pub async fn init_shards(&self, num_shards: u32, nodes_per_shard: usize) -> Result<()> {
        let mut shards = self.shards.write().await;
        
        for shard_id in 0..num_shards {
            let metadata = ShardMetadata {
                shard_id,
                node_ids: Vec::new(), // Will be assigned by shard manager
                primary_node: 0,
                triple_count: 0,
                size_bytes: 0,
                state: ShardState::Active,
                last_updated: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            };
            shards.insert(shard_id, metadata);
        }
        
        info!("Initialized {} shards", num_shards);
        Ok(())
    }
    
    /// Route a triple to appropriate shard
    pub async fn route_triple(&self, triple: &Triple) -> Result<ShardId> {
        // Check cache first
        let cache_key = format!("{:?}", triple);
        if let Some(&shard_id) = self.routing_cache.read().await.get(&cache_key) {
            return Ok(shard_id);
        }
        
        let shard_id = match &self.strategy {
            ShardingStrategy::Hash { num_shards } => {
                self.hash_route(&triple.subject().to_string(), *num_shards)
            }
            
            ShardingStrategy::Subject { num_shards } => {
                self.hash_route(&triple.subject().to_string(), *num_shards)
            }
            
            ShardingStrategy::Predicate { predicate_groups } => {
                let predicate_str = triple.predicate().to_string();
                predicate_groups.get(&predicate_str)
                    .copied()
                    .unwrap_or_else(|| self.hash_route(&predicate_str, predicate_groups.len() as u32))
            }
            
            ShardingStrategy::Namespace { namespace_mapping } => {
                self.route_by_namespace(&triple.subject().to_string(), namespace_mapping)?
            }
            
            ShardingStrategy::Graph { graph_mapping } => {
                // For now, default to subject-based routing
                // In a full implementation, this would check the graph context
                self.hash_route(&triple.subject().to_string(), graph_mapping.len() as u32)
            }
            
            ShardingStrategy::Semantic { concept_clusters, similarity_threshold } => {
                self.semantic_route(triple, concept_clusters, *similarity_threshold)?
            }
            
            ShardingStrategy::Hybrid { primary, secondary } => {
                // Try primary strategy first, fall back to secondary
                match self.route_with_strategy(triple, primary) {
                    Ok(shard) => shard,
                    Err(_) => self.route_with_strategy(triple, secondary)?,
                }
            }
        };
        
        // Cache the routing decision
        self.routing_cache.write().await.insert(cache_key, shard_id);
        
        Ok(shard_id)
    }
    
    /// Route using a specific strategy
    fn route_with_strategy(&self, triple: &Triple, strategy: &ShardingStrategy) -> Result<ShardId> {
        // Direct routing logic without recursion
        match strategy {
            ShardingStrategy::Hash { num_shards } => {
                Ok(self.hash_route(&triple.subject().to_string(), *num_shards))
            }
            
            ShardingStrategy::Subject { num_shards } => {
                Ok(self.hash_route(&triple.subject().to_string(), *num_shards))
            }
            
            ShardingStrategy::Predicate { predicate_groups } => {
                let predicate_str = triple.predicate().to_string();
                Ok(predicate_groups.get(&predicate_str)
                    .copied()
                    .unwrap_or_else(|| self.hash_route(&predicate_str, predicate_groups.len() as u32)))
            }
            
            ShardingStrategy::Namespace { namespace_mapping } => {
                self.route_by_namespace(&triple.subject().to_string(), namespace_mapping)
            }
            
            ShardingStrategy::Graph { graph_mapping } => {
                Ok(self.hash_route(&triple.subject().to_string(), graph_mapping.len() as u32))
            }
            
            ShardingStrategy::Semantic { concept_clusters, similarity_threshold } => {
                self.semantic_route(triple, concept_clusters, *similarity_threshold)
            }
            
            ShardingStrategy::Hybrid { .. } => {
                // For hybrid, just use hash routing to avoid recursion
                Ok(self.hash_route(&triple.subject().to_string(), 10))
            }
        }
    }
    
    /// Hash-based routing
    fn hash_route(&self, key: &str, num_shards: u32) -> ShardId {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() % num_shards as u64) as ShardId
    }
    
    /// Route by namespace
    fn route_by_namespace(&self, iri: &str, namespace_mapping: &HashMap<String, ShardId>) -> Result<ShardId> {
        // Extract namespace from IRI
        let namespace = if let Some(pos) = iri.rfind('#') {
            &iri[..=pos]
        } else if let Some(pos) = iri.rfind('/') {
            &iri[..=pos]
        } else {
            iri
        };
        
        Ok(namespace_mapping.get(namespace)
            .copied()
            .unwrap_or_else(|| self.hash_route(namespace, namespace_mapping.len() as u32)))
    }
    
    /// Semantic routing based on concept similarity
    fn semantic_route(&self, triple: &Triple, clusters: &[ConceptCluster], threshold: f64) -> Result<ShardId> {
        if let Some(similarity_calc) = &self.similarity_calc {
            // Extract concept from subject
            let concept = triple.subject().to_string();
            
            // Find best matching cluster
            if let Some(cluster_id) = similarity_calc.find_cluster(&concept, clusters) {
                return Ok(cluster_id);
            }
        }
        
        // Fall back to hash-based routing
        Ok(self.hash_route(&triple.subject().to_string(), clusters.len() as u32))
    }
    
    /// Get shard for a query pattern
    pub async fn route_query_pattern(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Result<Vec<ShardId>> {
        match &self.strategy {
            ShardingStrategy::Subject { num_shards } => {
                if let Some(subj) = subject {
                    // Query targets specific subject, route to its shard
                    Ok(vec![self.hash_route(subj, *num_shards)])
                } else {
                    // Query spans all subjects, check all shards
                    Ok((0..*num_shards).collect())
                }
            }
            
            ShardingStrategy::Predicate { predicate_groups } => {
                if let Some(pred) = predicate {
                    // Route to shard handling this predicate
                    if let Some(&shard_id) = predicate_groups.get(pred) {
                        Ok(vec![shard_id])
                    } else {
                        Ok(vec![self.hash_route(pred, predicate_groups.len() as u32)])
                    }
                } else {
                    // Query spans all predicates
                    let mut shard_ids: Vec<ShardId> = predicate_groups.values().copied().collect();
                    shard_ids.sort_unstable();
                    shard_ids.dedup();
                    Ok(shard_ids)
                }
            }
            
            _ => {
                // For other strategies, we need to check all shards
                let shards = self.shards.read().await;
                Ok(shards.keys().copied().collect())
            }
        }
    }
    
    /// Get shard metadata
    pub async fn get_shard_metadata(&self, shard_id: ShardId) -> Option<ShardMetadata> {
        self.shards.read().await.get(&shard_id).cloned()
    }
    
    /// Update shard metadata
    pub async fn update_shard_metadata(&self, metadata: ShardMetadata) -> Result<()> {
        self.shards.write().await.insert(metadata.shard_id, metadata);
        Ok(())
    }
    
    /// Get sharding statistics
    pub async fn get_statistics(&self) -> ShardingStatistics {
        let shards = self.shards.read().await;
        
        let total_triples: usize = shards.values().map(|s| s.triple_count).sum();
        let total_size: u64 = shards.values().map(|s| s.size_bytes).sum();
        let active_shards = shards.values().filter(|s| s.state == ShardState::Active).count();
        
        let mut distribution = Vec::new();
        for shard in shards.values() {
            distribution.push(ShardDistribution {
                shard_id: shard.shard_id,
                triple_count: shard.triple_count,
                size_bytes: shard.size_bytes,
                load_factor: if total_triples > 0 {
                    shard.triple_count as f64 / total_triples as f64
                } else {
                    0.0
                },
            });
        }
        
        ShardingStatistics {
            total_shards: shards.len(),
            active_shards,
            total_triples,
            total_size,
            distribution,
        }
    }
}

/// Sharding statistics
#[derive(Debug, Clone)]
pub struct ShardingStatistics {
    /// Total number of shards
    pub total_shards: usize,
    /// Number of active shards
    pub active_shards: usize,
    /// Total number of triples
    pub total_triples: usize,
    /// Total size in bytes
    pub total_size: u64,
    /// Distribution across shards
    pub distribution: Vec<ShardDistribution>,
}

/// Shard distribution information
#[derive(Debug, Clone)]
pub struct ShardDistribution {
    /// Shard identifier
    pub shard_id: ShardId,
    /// Number of triples
    pub triple_count: usize,
    /// Size in bytes
    pub size_bytes: u64,
    /// Load factor (0.0 to 1.0)
    pub load_factor: f64,
}

/// Default concept similarity calculator using simple string matching
pub struct DefaultConceptSimilarity;

impl ConceptSimilarity for DefaultConceptSimilarity {
    fn similarity(&self, concept1: &str, concept2: &str) -> f64 {
        // Simple implementation: check common prefix
        let common_prefix_len = concept1.chars()
            .zip(concept2.chars())
            .take_while(|(a, b)| a == b)
            .count();
        
        let max_len = concept1.len().max(concept2.len());
        if max_len > 0 {
            common_prefix_len as f64 / max_len as f64
        } else {
            0.0
        }
    }
    
    fn find_cluster(&self, concept: &str, clusters: &[ConceptCluster]) -> Option<ShardId> {
        let mut best_cluster = None;
        let mut best_score = 0.0;
        
        for cluster in clusters {
            // Check if concept matches any core concepts
            for core_concept in &cluster.core_concepts {
                let similarity = self.similarity(concept, core_concept);
                let weighted_score = similarity * cluster.weight;
                
                if weighted_score > best_score {
                    best_score = weighted_score;
                    best_cluster = Some(cluster.cluster_id);
                }
            }
            
            // Check namespace patterns
            for pattern in &cluster.namespace_patterns {
                if concept.starts_with(pattern) {
                    return Some(cluster.cluster_id);
                }
            }
        }
        
        best_cluster
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::{NamedNode, Triple};
    
    #[tokio::test]
    async fn test_hash_sharding() {
        let strategy = ShardingStrategy::Hash { num_shards: 4 };
        let router = ShardRouter::new(strategy);
        router.init_shards(4, 3).await.unwrap();
        
        let triple = Triple::new(
            NamedNode::new("http://example.org/subject1").unwrap(),
            NamedNode::new("http://example.org/predicate1").unwrap(),
            NamedNode::new("http://example.org/object1").unwrap(),
        );
        
        let shard_id = router.route_triple(&triple).await.unwrap();
        assert!(shard_id < 4);
        
        // Same triple should route to same shard
        let shard_id2 = router.route_triple(&triple).await.unwrap();
        assert_eq!(shard_id, shard_id2);
    }
    
    #[tokio::test]
    async fn test_namespace_sharding() {
        let mut namespace_mapping = HashMap::new();
        namespace_mapping.insert("http://example.org/".to_string(), 0);
        namespace_mapping.insert("http://schema.org/".to_string(), 1);
        
        let strategy = ShardingStrategy::Namespace { namespace_mapping };
        let router = ShardRouter::new(strategy);
        
        let triple1 = Triple::new(
            NamedNode::new("http://example.org/subject1").unwrap(),
            NamedNode::new("http://example.org/predicate1").unwrap(),
            NamedNode::new("http://example.org/object1").unwrap(),
        );
        
        let triple2 = Triple::new(
            NamedNode::new("http://schema.org/Person").unwrap(),
            NamedNode::new("http://schema.org/name").unwrap(),
            NamedNode::new("http://example.org/john").unwrap(),
        );
        
        assert_eq!(router.route_triple(&triple1).await.unwrap(), 0);
        assert_eq!(router.route_triple(&triple2).await.unwrap(), 1);
    }
    
    #[tokio::test]
    async fn test_semantic_sharding() {
        let clusters = vec![
            ConceptCluster {
                cluster_id: 0,
                core_concepts: vec!["http://schema.org/Person".to_string()].into_iter().collect(),
                predicates: vec!["http://schema.org/name".to_string()].into_iter().collect(),
                namespace_patterns: vec!["http://schema.org/".to_string()],
                weight: 1.0,
            },
            ConceptCluster {
                cluster_id: 1,
                core_concepts: vec!["http://example.org/Document".to_string()].into_iter().collect(),
                predicates: vec!["http://example.org/title".to_string()].into_iter().collect(),
                namespace_patterns: vec!["http://example.org/".to_string()],
                weight: 1.0,
            },
        ];
        
        let strategy = ShardingStrategy::Semantic {
            concept_clusters: clusters,
            similarity_threshold: 0.5,
        };
        
        let router = ShardRouter::new(strategy)
            .with_similarity_calculator(Arc::new(DefaultConceptSimilarity));
        
        let triple = Triple::new(
            NamedNode::new("http://schema.org/Person/123").unwrap(),
            NamedNode::new("http://schema.org/name").unwrap(),
            NamedNode::new("John Doe").unwrap(),
        );
        
        let shard_id = router.route_triple(&triple).await.unwrap();
        assert_eq!(shard_id, 0); // Should route to schema.org cluster
    }
    
    #[test]
    fn test_concept_similarity() {
        let calc = DefaultConceptSimilarity;
        
        assert_eq!(calc.similarity("http://example.org/Person", "http://example.org/Person"), 1.0);
        assert!(calc.similarity("http://example.org/Person", "http://example.org/Place") > 0.5);
        assert!(calc.similarity("http://example.org/Person", "http://schema.org/Person") < 0.5);
    }
}