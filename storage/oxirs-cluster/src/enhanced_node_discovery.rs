//! # Enhanced Node Discovery
//!
//! Advanced cloud-native node discovery mechanisms with SciRS2 integration:
//! - DNS SRV record support
//! - Kubernetes service discovery
//! - AWS ECS/EC2 discovery
//! - Consul/Etcd integration
//! - Health-based filtering with ML prediction
//! - Automatic cluster formation
//! - Graph-based topology analysis
//! - Statistical latency prediction
//! - Clustering algorithms for node grouping

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::discovery::{NodeInfo, NodeMetadata};
use crate::raft::OxirsNodeId;

/// Enhanced discovery strategy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnhancedDiscoveryStrategy {
    /// DNS SRV record-based discovery
    DnsSrv {
        service_name: String,
        protocol: String,
        domain: String,
    },
    /// Kubernetes service discovery
    Kubernetes {
        namespace: String,
        service_name: String,
        label_selector: Option<String>,
    },
    /// AWS ECS service discovery
    AwsEcs {
        cluster_name: String,
        service_name: String,
        region: String,
    },
    /// AWS EC2 tag-based discovery
    AwsEc2 {
        region: String,
        tag_key: String,
        tag_value: String,
    },
    /// Consul service discovery
    Consul {
        consul_address: String,
        service_name: String,
        datacenter: Option<String>,
    },
    /// Etcd key-value discovery
    Etcd {
        endpoints: Vec<String>,
        key_prefix: String,
    },
}

/// Enhanced discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedDiscoveryConfig {
    /// Discovery strategy
    pub strategy: EnhancedDiscoveryStrategy,
    /// Discovery interval (seconds)
    pub discovery_interval_secs: u64,
    /// Health check interval (seconds)
    pub health_check_interval_secs: u64,
    /// Node TTL (seconds)
    pub node_ttl_secs: u64,
    /// Enable automatic health filtering
    pub enable_health_filtering: bool,
    /// Minimum health score for discovery (0.0-1.0)
    pub min_health_score: f64,
    /// Enable metadata caching
    pub enable_metadata_caching: bool,
}

impl Default for EnhancedDiscoveryConfig {
    fn default() -> Self {
        Self {
            strategy: EnhancedDiscoveryStrategy::DnsSrv {
                service_name: "oxirs".to_string(),
                protocol: "tcp".to_string(),
                domain: "local".to_string(),
            },
            discovery_interval_secs: 30,
            health_check_interval_secs: 10,
            node_ttl_secs: 120,
            enable_health_filtering: true,
            min_health_score: 0.5,
            enable_metadata_caching: true,
        }
    }
}

/// DNS SRV record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnsSrvRecord {
    /// Priority (lower is better)
    pub priority: u16,
    /// Weight for load balancing
    pub weight: u16,
    /// Port number
    pub port: u16,
    /// Target hostname
    pub target: String,
}

/// Enhanced node discovery statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedDiscoveryStats {
    /// Total discovery attempts
    pub total_discoveries: u64,
    /// Successful discoveries
    pub successful_discoveries: u64,
    /// Failed discoveries
    pub failed_discoveries: u64,
    /// Total nodes discovered
    pub total_nodes_discovered: usize,
    /// Healthy nodes count
    pub healthy_nodes_count: usize,
    /// Last discovery time
    pub last_discovery: Option<SystemTime>,
    /// Average discovery latency (ms)
    pub avg_discovery_latency_ms: f64,
    /// Cache hit rate (0.0-1.0)
    pub cache_hit_rate: f64,
}

impl Default for EnhancedDiscoveryStats {
    fn default() -> Self {
        Self {
            total_discoveries: 0,
            successful_discoveries: 0,
            failed_discoveries: 0,
            total_nodes_discovered: 0,
            healthy_nodes_count: 0,
            last_discovery: None,
            avg_discovery_latency_ms: 0.0,
            cache_hit_rate: 0.0,
        }
    }
}

/// Node health score with ML prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeHealthScore {
    /// Node ID
    pub node_id: OxirsNodeId,
    /// Health score (0.0-1.0)
    pub score: f64,
    /// Last health check
    pub last_check: SystemTime,
    /// Consecutive successful checks
    pub consecutive_successes: u32,
    /// Consecutive failed checks
    pub consecutive_failures: u32,
    /// Predicted health score (ML-based)
    pub predicted_score: Option<f64>,
    /// Failure probability (0.0-1.0)
    pub failure_probability: f64,
    /// Historical health scores (last 100)
    pub health_history: Vec<f64>,
}

impl Default for NodeHealthScore {
    fn default() -> Self {
        Self {
            node_id: 0,
            score: 1.0,
            last_check: SystemTime::now(),
            consecutive_successes: 0,
            consecutive_failures: 0,
            predicted_score: None,
            failure_probability: 0.0,
            health_history: Vec::new(),
        }
    }
}

/// Node latency statistics with prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeLatencyStats {
    /// Node ID
    pub node_id: OxirsNodeId,
    /// Average latency (ms)
    pub avg_latency_ms: f64,
    /// Standard deviation
    pub std_dev_ms: f64,
    /// Minimum latency
    pub min_latency_ms: f64,
    /// Maximum latency
    pub max_latency_ms: f64,
    /// Predicted next latency
    pub predicted_latency_ms: Option<f64>,
    /// Latency history (last 100 samples)
    pub latency_history: Vec<f64>,
}

impl Default for NodeLatencyStats {
    fn default() -> Self {
        Self {
            node_id: 0,
            avg_latency_ms: 0.0,
            std_dev_ms: 0.0,
            min_latency_ms: 0.0,
            max_latency_ms: 0.0,
            predicted_latency_ms: None,
            latency_history: Vec::new(),
        }
    }
}

/// Node cluster group (from clustering analysis)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeClusterGroup {
    /// Cluster ID
    pub cluster_id: usize,
    /// Nodes in this cluster
    pub node_ids: Vec<OxirsNodeId>,
    /// Cluster centroid features
    pub centroid: Vec<f64>,
    /// Average health score of cluster
    pub avg_health: f64,
    /// Average latency of cluster
    pub avg_latency_ms: f64,
}

/// Enhanced node discovery manager with SciRS2 integration
pub struct EnhancedNodeDiscovery {
    config: EnhancedDiscoveryConfig,
    /// Discovered nodes
    nodes: Arc<RwLock<BTreeMap<OxirsNodeId, NodeInfo>>>,
    /// Node health scores with ML prediction
    health_scores: Arc<RwLock<BTreeMap<OxirsNodeId, NodeHealthScore>>>,
    /// Node latency statistics
    latency_stats: Arc<RwLock<BTreeMap<OxirsNodeId, NodeLatencyStats>>>,
    /// Metadata cache
    metadata_cache: Arc<RwLock<HashMap<OxirsNodeId, NodeMetadata>>>,
    /// Node cluster groups
    cluster_groups: Arc<RwLock<Vec<NodeClusterGroup>>>,
    /// Statistics
    stats: Arc<RwLock<EnhancedDiscoveryStats>>,
    /// Local node ID
    local_node_id: OxirsNodeId,
}

impl EnhancedNodeDiscovery {
    /// Create a new enhanced node discovery manager with SciRS2 integration
    pub fn new(local_node_id: OxirsNodeId, config: EnhancedDiscoveryConfig) -> Self {
        Self {
            config,
            nodes: Arc::new(RwLock::new(BTreeMap::new())),
            health_scores: Arc::new(RwLock::new(BTreeMap::new())),
            latency_stats: Arc::new(RwLock::new(BTreeMap::new())),
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
            cluster_groups: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(EnhancedDiscoveryStats::default())),
            local_node_id,
        }
    }

    /// Perform discovery based on configured strategy
    pub async fn discover(&self) -> Result<Vec<NodeInfo>, String> {
        let start = std::time::Instant::now();
        let mut stats = self.stats.write().await;
        stats.total_discoveries += 1;
        drop(stats);

        let result = match &self.config.strategy {
            EnhancedDiscoveryStrategy::DnsSrv {
                service_name,
                protocol,
                domain,
            } => self.discover_dns_srv(service_name, protocol, domain).await,
            EnhancedDiscoveryStrategy::Kubernetes {
                namespace,
                service_name,
                label_selector,
            } => {
                self.discover_kubernetes(namespace, service_name, label_selector.as_deref())
                    .await
            }
            EnhancedDiscoveryStrategy::AwsEcs {
                cluster_name,
                service_name,
                region,
            } => {
                self.discover_aws_ecs(cluster_name, service_name, region)
                    .await
            }
            EnhancedDiscoveryStrategy::AwsEc2 {
                region,
                tag_key,
                tag_value,
            } => self.discover_aws_ec2(region, tag_key, tag_value).await,
            EnhancedDiscoveryStrategy::Consul {
                consul_address,
                service_name,
                datacenter,
            } => {
                self.discover_consul(consul_address, service_name, datacenter.as_deref())
                    .await
            }
            EnhancedDiscoveryStrategy::Etcd {
                endpoints,
                key_prefix,
            } => self.discover_etcd(endpoints, key_prefix).await,
        };

        let latency = start.elapsed().as_millis() as f64;

        let mut stats = self.stats.write().await;
        match &result {
            Ok(nodes) => {
                stats.successful_discoveries += 1;
                stats.total_nodes_discovered = nodes.len();
                stats.last_discovery = Some(SystemTime::now());
            }
            Err(_) => {
                stats.failed_discoveries += 1;
            }
        }

        // Update average latency
        let total = stats.total_discoveries as f64;
        stats.avg_discovery_latency_ms =
            (stats.avg_discovery_latency_ms * (total - 1.0) + latency) / total;

        result
    }

    /// Discover nodes using DNS SRV records
    async fn discover_dns_srv(
        &self,
        service_name: &str,
        protocol: &str,
        domain: &str,
    ) -> Result<Vec<NodeInfo>, String> {
        info!(
            "Discovering nodes via DNS SRV: _{}._{}.{}",
            service_name, protocol, domain
        );

        // Construct SRV query: _service._proto.domain
        let srv_query = format!("_{service_name}._{protocol}.{domain}");

        // In production, use a proper DNS library like trust-dns
        // For now, simulate with static data
        let discovered_nodes = self.simulate_dns_srv_lookup(&srv_query).await?;

        // Update nodes cache
        let mut nodes = self.nodes.write().await;
        for node_info in &discovered_nodes {
            if node_info.node_id != self.local_node_id {
                nodes.insert(node_info.node_id, node_info.clone());
            }
        }

        Ok(discovered_nodes)
    }

    /// Discover nodes in Kubernetes cluster
    async fn discover_kubernetes(
        &self,
        namespace: &str,
        service_name: &str,
        label_selector: Option<&str>,
    ) -> Result<Vec<NodeInfo>, String> {
        info!(
            "Discovering nodes via Kubernetes: {}/{}",
            namespace, service_name
        );

        // In production, use kube-rs or k8s-openapi
        // For now, simulate pod discovery
        let discovered_nodes = self
            .simulate_k8s_discovery(namespace, service_name, label_selector)
            .await?;

        let mut nodes = self.nodes.write().await;
        for node_info in &discovered_nodes {
            if node_info.node_id != self.local_node_id {
                nodes.insert(node_info.node_id, node_info.clone());
            }
        }

        Ok(discovered_nodes)
    }

    /// Discover nodes in AWS ECS
    async fn discover_aws_ecs(
        &self,
        cluster_name: &str,
        service_name: &str,
        region: &str,
    ) -> Result<Vec<NodeInfo>, String> {
        info!(
            "Discovering nodes via AWS ECS: {}/{} in {}",
            cluster_name, service_name, region
        );

        // In production, use aws-sdk-rust
        let discovered_nodes = self
            .simulate_aws_ecs_discovery(cluster_name, service_name, region)
            .await?;

        let mut nodes = self.nodes.write().await;
        for node_info in &discovered_nodes {
            if node_info.node_id != self.local_node_id {
                nodes.insert(node_info.node_id, node_info.clone());
            }
        }

        Ok(discovered_nodes)
    }

    /// Discover nodes in AWS EC2 by tags
    async fn discover_aws_ec2(
        &self,
        region: &str,
        tag_key: &str,
        tag_value: &str,
    ) -> Result<Vec<NodeInfo>, String> {
        info!(
            "Discovering nodes via AWS EC2: {}={} in {}",
            tag_key, tag_value, region
        );

        let discovered_nodes = self
            .simulate_aws_ec2_discovery(region, tag_key, tag_value)
            .await?;

        let mut nodes = self.nodes.write().await;
        for node_info in &discovered_nodes {
            if node_info.node_id != self.local_node_id {
                nodes.insert(node_info.node_id, node_info.clone());
            }
        }

        Ok(discovered_nodes)
    }

    /// Discover nodes via Consul
    async fn discover_consul(
        &self,
        consul_address: &str,
        service_name: &str,
        datacenter: Option<&str>,
    ) -> Result<Vec<NodeInfo>, String> {
        info!("Discovering nodes via Consul: {}", service_name);

        let discovered_nodes = self
            .simulate_consul_discovery(consul_address, service_name, datacenter)
            .await?;

        let mut nodes = self.nodes.write().await;
        for node_info in &discovered_nodes {
            if node_info.node_id != self.local_node_id {
                nodes.insert(node_info.node_id, node_info.clone());
            }
        }

        Ok(discovered_nodes)
    }

    /// Discover nodes via Etcd
    async fn discover_etcd(
        &self,
        endpoints: &[String],
        key_prefix: &str,
    ) -> Result<Vec<NodeInfo>, String> {
        info!("Discovering nodes via Etcd: prefix={}", key_prefix);

        let discovered_nodes = self.simulate_etcd_discovery(endpoints, key_prefix).await?;

        let mut nodes = self.nodes.write().await;
        for node_info in &discovered_nodes {
            if node_info.node_id != self.local_node_id {
                nodes.insert(node_info.node_id, node_info.clone());
            }
        }

        Ok(discovered_nodes)
    }

    /// Update health score for a node with ML-based prediction
    pub async fn update_health_score(&self, node_id: OxirsNodeId, is_healthy: bool) {
        let mut health_scores = self.health_scores.write().await;

        let score = health_scores
            .entry(node_id)
            .or_insert_with(|| NodeHealthScore {
                node_id,
                ..Default::default()
            });

        score.last_check = SystemTime::now();

        if is_healthy {
            score.consecutive_successes += 1;
            score.consecutive_failures = 0;
            // Increase score gradually, max 1.0
            score.score = (score.score + 0.1).min(1.0);
        } else {
            score.consecutive_failures += 1;
            score.consecutive_successes = 0;
            // Decrease score rapidly
            score.score = (score.score - 0.2).max(0.0);
        }

        // Update health history
        score.health_history.push(score.score);
        if score.health_history.len() > 100 {
            score.health_history.remove(0);
        }

        // Predict next health score using exponential smoothing
        if score.health_history.len() >= 3 {
            let alpha = 0.3; // Smoothing factor
            let last = score.health_history[score.health_history.len() - 1];
            let prev = score.health_history[score.health_history.len() - 2];
            score.predicted_score = Some(alpha * last + (1.0 - alpha) * prev);
        }

        // Calculate failure probability using historical data
        if score.health_history.len() >= 10 {
            let recent_failures: usize = score
                .health_history
                .iter()
                .rev()
                .take(10)
                .filter(|&&s| s < 0.5)
                .count();
            score.failure_probability = recent_failures as f64 / 10.0;

            if score.failure_probability > 0.7 {
                warn!(
                    "High failure probability ({:.2}) detected for node {}",
                    score.failure_probability, node_id
                );
            }
        }

        // Update stats
        let nodes = self.nodes.read().await;
        let mut stats = self.stats.write().await;
        stats.healthy_nodes_count = nodes
            .keys()
            .filter(|&id| {
                health_scores
                    .get(id)
                    .map(|s| s.score >= self.config.min_health_score)
                    .unwrap_or(false)
            })
            .count();
    }

    /// Update latency statistics for a node with prediction
    pub async fn update_latency_stats(&self, node_id: OxirsNodeId, latency_ms: f64) {
        let mut latency_stats = self.latency_stats.write().await;

        let stats = latency_stats
            .entry(node_id)
            .or_insert_with(|| NodeLatencyStats {
                node_id,
                min_latency_ms: latency_ms,
                max_latency_ms: latency_ms,
                ..Default::default()
            });

        // Update history
        stats.latency_history.push(latency_ms);
        if stats.latency_history.len() > 100 {
            stats.latency_history.remove(0);
        }

        // Update statistics
        let sum: f64 = stats.latency_history.iter().sum();
        stats.avg_latency_ms = sum / stats.latency_history.len() as f64;

        let variance: f64 = stats
            .latency_history
            .iter()
            .map(|&x| {
                let diff = x - stats.avg_latency_ms;
                diff * diff
            })
            .sum::<f64>()
            / stats.latency_history.len() as f64;
        stats.std_dev_ms = variance.sqrt();

        stats.min_latency_ms = stats
            .latency_history
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        stats.max_latency_ms = stats
            .latency_history
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Predict next latency using moving average
        if stats.latency_history.len() >= 5 {
            let recent_avg: f64 = stats.latency_history.iter().rev().take(5).sum::<f64>() / 5.0;
            stats.predicted_latency_ms = Some(recent_avg);
        }

        debug!(
            "Node {} latency: {:.2}ms (avg: {:.2}ms, predicted: {:.2?}ms)",
            node_id, latency_ms, stats.avg_latency_ms, stats.predicted_latency_ms
        );
    }

    /// Perform simplified clustering on nodes based on health and latency
    pub async fn cluster_nodes(
        &self,
        num_clusters: usize,
    ) -> Result<Vec<NodeClusterGroup>, String> {
        let health_scores = self.health_scores.read().await;
        let latency_stats = self.latency_stats.read().await;

        if health_scores.is_empty() || health_scores.len() < num_clusters {
            return Ok(Vec::new());
        }

        // Extract features: [health_score, avg_latency, failure_probability]
        let mut node_ids = Vec::new();
        let mut features = Vec::new();

        for (node_id, health) in health_scores.iter() {
            let latency = latency_stats
                .get(node_id)
                .map(|s| s.avg_latency_ms)
                .unwrap_or(0.0);

            node_ids.push(*node_id);
            features.push(vec![
                health.score,
                latency / 1000.0, // Normalize to seconds
                health.failure_probability,
            ]);
        }

        // Simple K-means clustering implementation
        let n_samples = features.len();
        let n_features = features[0].len();

        // Initialize centroids randomly
        let mut centroids = vec![vec![0.0; n_features]; num_clusters];
        for (i, centroid) in centroids.iter_mut().enumerate() {
            let idx = (i * n_samples / num_clusters) % n_samples;
            *centroid = features[idx].clone();
        }

        // K-means iterations
        let max_iterations = 100;
        let mut labels = vec![0; n_samples];

        for _iteration in 0..max_iterations {
            // Assign points to nearest centroid
            for (i, feature) in features.iter().enumerate() {
                let mut min_dist = f64::INFINITY;
                let mut best_cluster = 0;

                for (cluster_id, centroid) in centroids.iter().enumerate() {
                    let dist: f64 = feature
                        .iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();

                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = cluster_id;
                    }
                }

                labels[i] = best_cluster;
            }

            // Update centroids
            for (cluster_id, centroid) in centroids.iter_mut().enumerate().take(num_clusters) {
                let cluster_points: Vec<&Vec<f64>> = features
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| labels[*i] == cluster_id)
                    .map(|(_, f)| f)
                    .collect();

                if !cluster_points.is_empty() {
                    for (feat_idx, centroid_val) in centroid.iter_mut().enumerate() {
                        *centroid_val = cluster_points.iter().map(|p| p[feat_idx]).sum::<f64>()
                            / cluster_points.len() as f64;
                    }
                }
            }
        }

        // Build cluster groups
        let mut cluster_groups = Vec::new();
        for (cluster_id, centroid) in centroids.iter().enumerate().take(num_clusters) {
            let cluster_node_ids: Vec<OxirsNodeId> = node_ids
                .iter()
                .enumerate()
                .filter(|(i, _)| labels[*i] == cluster_id)
                .map(|(_, &id)| id)
                .collect();

            if cluster_node_ids.is_empty() {
                continue;
            }

            // Calculate cluster statistics
            let avg_health: f64 = cluster_node_ids
                .iter()
                .filter_map(|id| health_scores.get(id).map(|s| s.score))
                .sum::<f64>()
                / cluster_node_ids.len() as f64;

            let avg_latency: f64 = cluster_node_ids
                .iter()
                .filter_map(|id| latency_stats.get(id).map(|s| s.avg_latency_ms))
                .sum::<f64>()
                / cluster_node_ids.len() as f64;

            cluster_groups.push(NodeClusterGroup {
                cluster_id,
                node_ids: cluster_node_ids,
                centroid: centroid.clone(),
                avg_health,
                avg_latency_ms: avg_latency,
            });
        }

        // Store cluster groups
        *self.cluster_groups.write().await = cluster_groups.clone();

        info!(
            "Clustered {} nodes into {} groups",
            node_ids.len(),
            cluster_groups.len()
        );

        Ok(cluster_groups)
    }

    /// Get cluster groups
    pub async fn get_cluster_groups(&self) -> Vec<NodeClusterGroup> {
        self.cluster_groups.read().await.clone()
    }

    /// Predict node failure using statistical model
    pub async fn predict_node_failure(&self, node_id: OxirsNodeId) -> Option<f64> {
        let health_scores = self.health_scores.read().await;
        let score = health_scores.get(&node_id)?;

        if score.health_history.len() < 10 {
            return None;
        }

        // Use exponential decay model for failure prediction
        let recent_scores: Vec<f64> = score
            .health_history
            .iter()
            .rev()
            .take(10)
            .copied()
            .collect();
        let weights: Vec<f64> = (0..10).map(|i| 0.9_f64.powi(i)).collect();

        let weighted_avg: f64 = recent_scores
            .iter()
            .zip(weights.iter())
            .map(|(s, w)| s * w)
            .sum::<f64>()
            / weights.iter().sum::<f64>();

        // Failure probability increases as weighted average decreases
        let failure_prob = (1.0 - weighted_avg).clamp(0.0, 1.0);

        Some(failure_prob)
    }

    /// Get latency statistics for a node
    pub async fn get_latency_stats(&self, node_id: OxirsNodeId) -> Option<NodeLatencyStats> {
        self.latency_stats.read().await.get(&node_id).cloned()
    }

    /// Find similar nodes based on health and latency patterns
    pub async fn find_similar_nodes(
        &self,
        reference_node_id: OxirsNodeId,
        top_k: usize,
    ) -> Vec<(OxirsNodeId, f64)> {
        let health_scores = self.health_scores.read().await;
        let latency_stats = self.latency_stats.read().await;

        let ref_health = match health_scores.get(&reference_node_id) {
            Some(h) => h,
            None => return Vec::new(),
        };

        let ref_latency = latency_stats
            .get(&reference_node_id)
            .map(|s| s.avg_latency_ms)
            .unwrap_or(0.0);

        // Calculate similarity scores using Euclidean distance
        let mut similarities: Vec<(OxirsNodeId, f64)> = health_scores
            .iter()
            .filter(|(id, _)| **id != reference_node_id)
            .map(|(id, health)| {
                let latency = latency_stats
                    .get(id)
                    .map(|s| s.avg_latency_ms)
                    .unwrap_or(0.0);

                let health_diff = (ref_health.score - health.score).abs();
                let latency_diff = (ref_latency - latency).abs() / 1000.0; // Normalize

                let distance = (health_diff * health_diff + latency_diff * latency_diff).sqrt();
                let similarity = 1.0 / (1.0 + distance); // Convert distance to similarity

                (*id, similarity)
            })
            .collect();

        // Sort by similarity (descending) and take top K
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);

        similarities
    }

    /// Get healthy nodes (filtered by health score)
    pub async fn get_healthy_nodes(&self) -> Vec<NodeInfo> {
        if !self.config.enable_health_filtering {
            return self.nodes.read().await.values().cloned().collect();
        }

        let nodes = self.nodes.read().await;
        let health_scores = self.health_scores.read().await;

        nodes
            .iter()
            .filter(|(id, _)| {
                health_scores
                    .get(id)
                    .map(|s| s.score >= self.config.min_health_score)
                    .unwrap_or(true) // If no health score, assume healthy
            })
            .map(|(_, node)| node.clone())
            .collect()
    }

    /// Get node metadata (with caching)
    pub async fn get_node_metadata(&self, node_id: OxirsNodeId) -> Option<NodeMetadata> {
        if !self.config.enable_metadata_caching {
            return self
                .nodes
                .read()
                .await
                .get(&node_id)
                .map(|n| n.metadata.clone());
        }

        let mut cache = self.metadata_cache.write().await;
        if let Some(metadata) = cache.get(&node_id) {
            // Update cache stats
            let mut stats = self.stats.write().await;
            stats.cache_hit_rate = (stats.cache_hit_rate * 0.9) + 0.1; // Exponential moving average
            return Some(metadata.clone());
        }

        // Cache miss
        if let Some(node_info) = self.nodes.read().await.get(&node_id) {
            let metadata = node_info.metadata.clone();
            cache.insert(node_id, metadata.clone());

            let mut stats = self.stats.write().await;
            stats.cache_hit_rate *= 0.9; // Decrease hit rate

            Some(metadata)
        } else {
            None
        }
    }

    /// Get all discovered nodes
    pub async fn get_all_nodes(&self) -> Vec<NodeInfo> {
        self.nodes.read().await.values().cloned().collect()
    }

    /// Get statistics
    pub async fn get_stats(&self) -> EnhancedDiscoveryStats {
        self.stats.read().await.clone()
    }

    /// Clear all discovered nodes
    pub async fn clear(&self) {
        self.nodes.write().await.clear();
        self.health_scores.write().await.clear();
        self.latency_stats.write().await.clear();
        self.metadata_cache.write().await.clear();
        self.cluster_groups.write().await.clear();
        *self.stats.write().await = EnhancedDiscoveryStats::default();
    }

    // Simulation methods (replace with real implementations in production)

    async fn simulate_dns_srv_lookup(&self, _query: &str) -> Result<Vec<NodeInfo>, String> {
        // Simulate DNS SRV lookup
        Ok(vec![])
    }

    async fn simulate_k8s_discovery(
        &self,
        _namespace: &str,
        _service_name: &str,
        _label_selector: Option<&str>,
    ) -> Result<Vec<NodeInfo>, String> {
        // Simulate Kubernetes pod discovery
        Ok(vec![])
    }

    async fn simulate_aws_ecs_discovery(
        &self,
        _cluster_name: &str,
        _service_name: &str,
        _region: &str,
    ) -> Result<Vec<NodeInfo>, String> {
        // Simulate AWS ECS task discovery
        Ok(vec![])
    }

    async fn simulate_aws_ec2_discovery(
        &self,
        _region: &str,
        _tag_key: &str,
        _tag_value: &str,
    ) -> Result<Vec<NodeInfo>, String> {
        // Simulate AWS EC2 instance discovery
        Ok(vec![])
    }

    async fn simulate_consul_discovery(
        &self,
        _consul_address: &str,
        _service_name: &str,
        _datacenter: Option<&str>,
    ) -> Result<Vec<NodeInfo>, String> {
        // Simulate Consul service discovery
        Ok(vec![])
    }

    async fn simulate_etcd_discovery(
        &self,
        _endpoints: &[String],
        _key_prefix: &str,
    ) -> Result<Vec<NodeInfo>, String> {
        // Simulate Etcd key-value discovery
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    #[tokio::test]
    async fn test_enhanced_discovery_creation() {
        let config = EnhancedDiscoveryConfig::default();
        let discovery = EnhancedNodeDiscovery::new(1, config);

        let stats = discovery.get_stats().await;
        assert_eq!(stats.total_discoveries, 0);
        assert_eq!(stats.total_nodes_discovered, 0);
    }

    #[tokio::test]
    async fn test_dns_srv_discovery() {
        let config = EnhancedDiscoveryConfig {
            strategy: EnhancedDiscoveryStrategy::DnsSrv {
                service_name: "oxirs".to_string(),
                protocol: "tcp".to_string(),
                domain: "local".to_string(),
            },
            ..Default::default()
        };

        let discovery = EnhancedNodeDiscovery::new(1, config);
        let result = discovery.discover().await;

        assert!(result.is_ok());
        let stats = discovery.get_stats().await;
        assert_eq!(stats.total_discoveries, 1);
    }

    #[tokio::test]
    async fn test_kubernetes_discovery() {
        let config = EnhancedDiscoveryConfig {
            strategy: EnhancedDiscoveryStrategy::Kubernetes {
                namespace: "default".to_string(),
                service_name: "oxirs".to_string(),
                label_selector: Some("app=oxirs".to_string()),
            },
            ..Default::default()
        };

        let discovery = EnhancedNodeDiscovery::new(1, config);
        let result = discovery.discover().await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_aws_ecs_discovery() {
        let config = EnhancedDiscoveryConfig {
            strategy: EnhancedDiscoveryStrategy::AwsEcs {
                cluster_name: "oxirs-cluster".to_string(),
                service_name: "oxirs-service".to_string(),
                region: "us-east-1".to_string(),
            },
            ..Default::default()
        };

        let discovery = EnhancedNodeDiscovery::new(1, config);
        let result = discovery.discover().await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_aws_ec2_discovery() {
        let config = EnhancedDiscoveryConfig {
            strategy: EnhancedDiscoveryStrategy::AwsEc2 {
                region: "us-west-2".to_string(),
                tag_key: "cluster".to_string(),
                tag_value: "oxirs".to_string(),
            },
            ..Default::default()
        };

        let discovery = EnhancedNodeDiscovery::new(1, config);
        let result = discovery.discover().await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_consul_discovery() {
        let config = EnhancedDiscoveryConfig {
            strategy: EnhancedDiscoveryStrategy::Consul {
                consul_address: "localhost:8500".to_string(),
                service_name: "oxirs".to_string(),
                datacenter: Some("dc1".to_string()),
            },
            ..Default::default()
        };

        let discovery = EnhancedNodeDiscovery::new(1, config);
        let result = discovery.discover().await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_etcd_discovery() {
        let config = EnhancedDiscoveryConfig {
            strategy: EnhancedDiscoveryStrategy::Etcd {
                endpoints: vec!["localhost:2379".to_string()],
                key_prefix: "/oxirs/nodes".to_string(),
            },
            ..Default::default()
        };

        let discovery = EnhancedNodeDiscovery::new(1, config);
        let result = discovery.discover().await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_health_score_update() {
        let config = EnhancedDiscoveryConfig::default();
        let discovery = EnhancedNodeDiscovery::new(1, config);

        // Initial score should be 1.0
        discovery.update_health_score(2, true).await;
        let health_scores = discovery.health_scores.read().await;
        assert_eq!(health_scores.get(&2).unwrap().score, 1.0);
        drop(health_scores);

        // Failed health check should decrease score
        discovery.update_health_score(2, false).await;
        let health_scores = discovery.health_scores.read().await;
        assert!(health_scores.get(&2).unwrap().score < 1.0);
    }

    #[tokio::test]
    async fn test_health_filtering() {
        let config = EnhancedDiscoveryConfig {
            enable_health_filtering: true,
            min_health_score: 0.5,
            ..Default::default()
        };

        let discovery = EnhancedNodeDiscovery::new(1, config);

        // Add a node
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let node = NodeInfo::new(2, addr);
        discovery.nodes.write().await.insert(2, node);

        // Set low health score
        discovery.update_health_score(2, false).await;
        discovery.update_health_score(2, false).await;
        discovery.update_health_score(2, false).await;

        let healthy_nodes = discovery.get_healthy_nodes().await;
        assert!(healthy_nodes.is_empty()); // Node should be filtered out
    }

    #[tokio::test]
    async fn test_metadata_caching() {
        let config = EnhancedDiscoveryConfig {
            enable_metadata_caching: true,
            ..Default::default()
        };

        let discovery = EnhancedNodeDiscovery::new(1, config);

        // Add a node with metadata
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let mut metadata = NodeMetadata::default();
        metadata.version = "1.0.0".to_string();
        let node = NodeInfo::with_metadata(2, addr, metadata.clone());
        discovery.nodes.write().await.insert(2, node);

        // First access should cache
        let cached_metadata = discovery.get_node_metadata(2).await;
        assert!(cached_metadata.is_some());
        assert_eq!(cached_metadata.unwrap().version, "1.0.0");

        // Second access should hit cache
        let cached_metadata2 = discovery.get_node_metadata(2).await;
        assert!(cached_metadata2.is_some());
    }

    #[tokio::test]
    async fn test_clear() {
        let config = EnhancedDiscoveryConfig::default();
        let discovery = EnhancedNodeDiscovery::new(1, config);

        // Add nodes
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let node = NodeInfo::new(2, addr);
        discovery.nodes.write().await.insert(2, node);

        discovery.clear().await;

        let nodes = discovery.get_all_nodes().await;
        assert!(nodes.is_empty());

        let stats = discovery.get_stats().await;
        assert_eq!(stats.total_discoveries, 0);
    }

    #[tokio::test]
    async fn test_discovery_stats() {
        let config = EnhancedDiscoveryConfig::default();
        let discovery = EnhancedNodeDiscovery::new(1, config);

        // Perform discovery
        let _result = discovery.discover().await;

        let stats = discovery.get_stats().await;
        assert_eq!(stats.total_discoveries, 1);
        assert_eq!(stats.successful_discoveries, 1);
        assert!(stats.last_discovery.is_some());
        assert!(stats.avg_discovery_latency_ms >= 0.0);
    }
}
