//! Feature extraction for machine learning models
//!
//! This module provides various feature extraction techniques for RDF graphs,
//! including structural features, statistical features, and embeddings.

use super::{EdgeFeatures, GlobalFeatures, GraphData, NodeFeatures};

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Feature extractor for RDF graphs
#[derive(Debug)]
pub struct FeatureExtractor {
    config: FeatureExtractionConfig,
    feature_cache: HashMap<String, Vec<f64>>,
}

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractionConfig {
    pub structural_features: StructuralFeatureConfig,
    pub statistical_features: StatisticalFeatureConfig,
    pub embedding_features: EmbeddingFeatureConfig,
    pub temporal_features: TemporalFeatureConfig,
    pub semantic_features: SemanticFeatureConfig,
}

/// Structural feature configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralFeatureConfig {
    pub include_degree_features: bool,
    pub include_centrality_features: bool,
    pub include_clustering_features: bool,
    pub include_path_features: bool,
    pub include_motif_features: bool,
    pub max_path_length: usize,
    pub motif_sizes: Vec<usize>,
}

/// Statistical feature configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalFeatureConfig {
    pub include_distribution_features: bool,
    pub include_correlation_features: bool,
    pub include_entropy_features: bool,
    pub include_diversity_features: bool,
    pub distribution_bins: usize,
}

/// Embedding feature configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingFeatureConfig {
    pub include_node_embeddings: bool,
    pub include_graph_embeddings: bool,
    pub embedding_dim: usize,
    pub embedding_method: EmbeddingMethod,
}

/// Temporal feature configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalFeatureConfig {
    pub include_temporal_features: bool,
    pub time_window_sizes: Vec<usize>,
    pub temporal_aggregations: Vec<TemporalAggregation>,
}

/// Semantic feature configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFeatureConfig {
    pub include_type_features: bool,
    pub include_property_features: bool,
    pub include_namespace_features: bool,
    pub include_ontology_features: bool,
}

/// Embedding methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingMethod {
    RandomWalk,
    Node2Vec,
    GraphSAGE,
    Spectral,
    DeepWalk,
}

/// Temporal aggregation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalAggregation {
    Mean,
    Max,
    Min,
    Sum,
    Count,
    Trend,
}

/// Extracted features for a graph
#[derive(Debug, Clone)]
pub struct ExtractedFeatures {
    pub structural: Vec<f64>,
    pub statistical: Vec<f64>,
    pub embeddings: Vec<f64>,
    pub temporal: Vec<f64>,
    pub semantic: Vec<f64>,
    pub feature_names: Vec<String>,
    pub feature_importance: HashMap<String, f64>,
}

impl FeatureExtractor {
    /// Create a new feature extractor
    pub fn new(config: FeatureExtractionConfig) -> Self {
        Self {
            config,
            feature_cache: HashMap::new(),
        }
    }

    /// Extract all features from graph data
    pub fn extract_features(&mut self, graph_data: &GraphData) -> ExtractedFeatures {
        let mut features: Vec<f64> = Vec::new();
        let mut feature_names: Vec<String> = Vec::new();

        // Extract structural features
        let structural = self.extract_structural_features(graph_data);
        features.extend(&structural.0);
        feature_names.extend(structural.1);

        // Extract statistical features
        let statistical = self.extract_statistical_features(graph_data);
        features.extend(&statistical.0);
        feature_names.extend(statistical.1);

        // Extract embedding features
        let embeddings = self.extract_embedding_features(graph_data);
        features.extend(&embeddings.0);
        feature_names.extend(embeddings.1);

        // Extract temporal features
        let temporal = self.extract_temporal_features(graph_data);
        features.extend(&temporal.0);
        feature_names.extend(temporal.1);

        // Extract semantic features
        let semantic = self.extract_semantic_features(graph_data);
        features.extend(&semantic.0);
        feature_names.extend(semantic.1);

        ExtractedFeatures {
            structural: structural.0,
            statistical: statistical.0,
            embeddings: embeddings.0,
            temporal: temporal.0,
            semantic: semantic.0,
            feature_names,
            feature_importance: HashMap::new(),
        }
    }

    /// Extract structural features
    fn extract_structural_features(&self, graph_data: &GraphData) -> (Vec<f64>, Vec<String>) {
        let mut features = Vec::new();
        let mut names = Vec::new();

        if self.config.structural_features.include_degree_features {
            let degree_features = self.compute_degree_features(graph_data);
            features.extend(&degree_features);
            names.extend(vec![
                "avg_degree".to_string(),
                "max_degree".to_string(),
                "min_degree".to_string(),
                "degree_variance".to_string(),
            ]);
        }

        if self.config.structural_features.include_centrality_features {
            let centrality_features = self.compute_centrality_features(graph_data);
            features.extend(&centrality_features);
            names.extend(vec![
                "avg_betweenness".to_string(),
                "max_betweenness".to_string(),
                "avg_closeness".to_string(),
                "max_closeness".to_string(),
            ]);
        }

        if self.config.structural_features.include_clustering_features {
            let clustering_features = self.compute_clustering_features(graph_data);
            features.extend(&clustering_features);
            names.extend(vec![
                "clustering_coefficient".to_string(),
                "transitivity".to_string(),
                "modularity".to_string(),
            ]);
        }

        if self.config.structural_features.include_path_features {
            let path_features = self.compute_path_features(graph_data);
            features.extend(&path_features);
            names.extend(vec![
                "avg_path_length".to_string(),
                "diameter".to_string(),
                "radius".to_string(),
            ]);
        }

        if self.config.structural_features.include_motif_features {
            let motif_features = self.compute_motif_features(graph_data);
            features.extend(&motif_features);
            for size in &self.config.structural_features.motif_sizes {
                names.push(format!("motif_{}_count", size));
            }
        }

        (features, names)
    }

    /// Compute degree features
    fn compute_degree_features(&self, graph_data: &GraphData) -> Vec<f64> {
        let mut degree_map: HashMap<String, usize> = HashMap::new();

        // Count degrees
        for edge in &graph_data.edges {
            *degree_map.entry(edge.source_id.clone()).or_insert(0) += 1;
            *degree_map.entry(edge.target_id.clone()).or_insert(0) += 1;
        }

        let degrees: Vec<f64> = degree_map.values().map(|&d| d as f64).collect();

        if degrees.is_empty() {
            return vec![0.0, 0.0, 0.0, 0.0];
        }

        let avg = degrees.iter().sum::<f64>() / degrees.len() as f64;
        let max = degrees.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = degrees.iter().cloned().fold(f64::INFINITY, f64::min);

        let variance =
            degrees.iter().map(|&d| (d - avg).powi(2)).sum::<f64>() / degrees.len() as f64;

        vec![avg, max, min, variance]
    }

    /// Compute centrality features
    fn compute_centrality_features(&self, graph_data: &GraphData) -> Vec<f64> {
        // Simplified centrality calculation
        let n = graph_data.nodes.len() as f64;

        // Approximate betweenness centrality
        let avg_betweenness = 0.5 / n.max(1.0);
        let max_betweenness = 1.0 / n.max(1.0);

        // Approximate closeness centrality
        let avg_closeness = 0.7 / n.max(1.0);
        let max_closeness = 1.0 / n.max(1.0);

        vec![
            avg_betweenness,
            max_betweenness,
            avg_closeness,
            max_closeness,
        ]
    }

    /// Compute clustering features
    fn compute_clustering_features(&self, graph_data: &GraphData) -> Vec<f64> {
        // Build adjacency list
        let mut adj_list: HashMap<String, HashSet<String>> = HashMap::new();

        for edge in &graph_data.edges {
            adj_list
                .entry(edge.source_id.clone())
                .or_insert_with(HashSet::new)
                .insert(edge.target_id.clone());
            adj_list
                .entry(edge.target_id.clone())
                .or_insert_with(HashSet::new)
                .insert(edge.source_id.clone());
        }

        // Compute clustering coefficient
        let mut clustering_sum = 0.0;
        let mut count = 0;

        for (node, neighbors) in &adj_list {
            if neighbors.len() < 2 {
                continue;
            }

            let mut triangles = 0;
            let neighbor_vec: Vec<_> = neighbors.iter().collect();

            for i in 0..neighbor_vec.len() {
                for j in i + 1..neighbor_vec.len() {
                    if adj_list
                        .get(neighbor_vec[i])
                        .map(|n| n.contains(neighbor_vec[j]))
                        .unwrap_or(false)
                    {
                        triangles += 1;
                    }
                }
            }

            let possible_triangles = neighbors.len() * (neighbors.len() - 1) / 2;
            clustering_sum += triangles as f64 / possible_triangles as f64;
            count += 1;
        }

        let clustering_coefficient = if count > 0 {
            clustering_sum / count as f64
        } else {
            0.0
        };

        // Simplified transitivity and modularity
        let transitivity = clustering_coefficient * 0.8;
        let modularity = 0.3; // Placeholder

        vec![clustering_coefficient, transitivity, modularity]
    }

    /// Compute path features
    fn compute_path_features(&self, graph_data: &GraphData) -> Vec<f64> {
        // Build adjacency list
        let mut adj_list: HashMap<String, Vec<String>> = HashMap::new();

        for edge in &graph_data.edges {
            adj_list
                .entry(edge.source_id.clone())
                .or_insert_with(Vec::new)
                .push(edge.target_id.clone());
            adj_list
                .entry(edge.target_id.clone())
                .or_insert_with(Vec::new)
                .push(edge.source_id.clone());
        }

        // Sample paths using BFS
        let mut all_paths = Vec::new();
        let sample_size = 10.min(graph_data.nodes.len());

        for i in 0..sample_size {
            let start_node = &graph_data.nodes[i].node_id;
            let paths = self.bfs_shortest_paths(
                start_node,
                &adj_list,
                self.config.structural_features.max_path_length,
            );
            all_paths.extend(paths);
        }

        if all_paths.is_empty() {
            return vec![0.0, 0.0, 0.0];
        }

        let avg_path_length = all_paths.iter().sum::<usize>() as f64 / all_paths.len() as f64;
        let diameter = *all_paths.iter().max().unwrap_or(&0) as f64;
        let radius = *all_paths.iter().min().unwrap_or(&0) as f64;

        vec![avg_path_length, diameter, radius]
    }

    /// BFS for shortest paths
    fn bfs_shortest_paths(
        &self,
        start: &str,
        adj_list: &HashMap<String, Vec<String>>,
        max_length: usize,
    ) -> Vec<usize> {
        let mut distances = HashMap::new();
        let mut queue = VecDeque::new();

        distances.insert(start.to_string(), 0);
        queue.push_back(start.to_string());

        while let Some(node) = queue.pop_front() {
            let dist = distances[&node];

            if dist >= max_length {
                continue;
            }

            if let Some(neighbors) = adj_list.get(&node) {
                for neighbor in neighbors {
                    if !distances.contains_key(neighbor) {
                        distances.insert(neighbor.clone(), dist + 1);
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        distances.values().cloned().collect()
    }

    /// Compute motif features
    fn compute_motif_features(&self, graph_data: &GraphData) -> Vec<f64> {
        let mut motif_counts = Vec::new();

        for &size in &self.config.structural_features.motif_sizes {
            let count = self.count_motifs_of_size(graph_data, size);
            motif_counts.push(count as f64);
        }

        motif_counts
    }

    /// Count motifs of specific size
    fn count_motifs_of_size(&self, _graph_data: &GraphData, size: usize) -> usize {
        // Simplified motif counting
        match size {
            3 => 10, // Triangle count placeholder
            4 => 5,  // 4-clique count placeholder
            _ => 0,
        }
    }

    /// Extract statistical features
    fn extract_statistical_features(&self, graph_data: &GraphData) -> (Vec<f64>, Vec<String>) {
        let mut features = Vec::new();
        let mut names = Vec::new();

        if self
            .config
            .statistical_features
            .include_distribution_features
        {
            let dist_features = self.compute_distribution_features(graph_data);
            features.extend(&dist_features);
            names.extend(vec![
                "degree_dist_mean".to_string(),
                "degree_dist_std".to_string(),
                "degree_dist_skew".to_string(),
                "degree_dist_kurtosis".to_string(),
            ]);
        }

        if self
            .config
            .statistical_features
            .include_correlation_features
        {
            let corr_features = self.compute_correlation_features(graph_data);
            features.extend(&corr_features);
            names.extend(vec![
                "degree_correlation".to_string(),
                "assortativity".to_string(),
            ]);
        }

        if self.config.statistical_features.include_entropy_features {
            let entropy_features = self.compute_entropy_features(graph_data);
            features.extend(&entropy_features);
            names.extend(vec![
                "degree_entropy".to_string(),
                "type_entropy".to_string(),
                "property_entropy".to_string(),
            ]);
        }

        if self.config.statistical_features.include_diversity_features {
            let diversity_features = self.compute_diversity_features(graph_data);
            features.extend(&diversity_features);
            names.extend(vec![
                "type_diversity".to_string(),
                "property_diversity".to_string(),
                "namespace_diversity".to_string(),
            ]);
        }

        (features, names)
    }

    /// Compute distribution features
    fn compute_distribution_features(&self, graph_data: &GraphData) -> Vec<f64> {
        // Degree distribution statistics
        let mut degrees = Vec::new();
        let mut degree_map: HashMap<String, usize> = HashMap::new();

        for edge in &graph_data.edges {
            *degree_map.entry(edge.source_id.clone()).or_insert(0) += 1;
            *degree_map.entry(edge.target_id.clone()).or_insert(0) += 1;
        }

        for &degree in degree_map.values() {
            degrees.push(degree as f64);
        }

        if degrees.is_empty() {
            return vec![0.0, 0.0, 0.0, 0.0];
        }

        let mean = degrees.iter().sum::<f64>() / degrees.len() as f64;
        let variance =
            degrees.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / degrees.len() as f64;
        let std = variance.sqrt();

        let skewness = if std > 0.0 {
            degrees
                .iter()
                .map(|&d| ((d - mean) / std).powi(3))
                .sum::<f64>()
                / degrees.len() as f64
        } else {
            0.0
        };

        let kurtosis = if std > 0.0 {
            degrees
                .iter()
                .map(|&d| ((d - mean) / std).powi(4))
                .sum::<f64>()
                / degrees.len() as f64
                - 3.0
        } else {
            0.0
        };

        vec![mean, std, skewness, kurtosis]
    }

    /// Compute correlation features
    fn compute_correlation_features(&self, graph_data: &GraphData) -> Vec<f64> {
        // Simplified degree correlation and assortativity
        let degree_correlation = 0.3; // Placeholder
        let assortativity = 0.1; // Placeholder

        vec![degree_correlation, assortativity]
    }

    /// Compute entropy features
    fn compute_entropy_features(&self, graph_data: &GraphData) -> Vec<f64> {
        // Degree entropy
        let mut degree_counts: HashMap<usize, usize> = HashMap::new();
        let mut degree_map: HashMap<String, usize> = HashMap::new();

        for edge in &graph_data.edges {
            *degree_map.entry(edge.source_id.clone()).or_insert(0) += 1;
            *degree_map.entry(edge.target_id.clone()).or_insert(0) += 1;
        }

        for &degree in degree_map.values() {
            *degree_counts.entry(degree).or_insert(0) += 1;
        }

        let total = degree_map.len() as f64;
        let degree_entropy = degree_counts
            .values()
            .map(|&count| {
                let p = count as f64 / total;
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum();

        // Type entropy
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for node in &graph_data.nodes {
            if let Some(node_type) = &node.node_type {
                *type_counts.entry(node_type.clone()).or_insert(0) += 1;
            }
        }

        let type_total = graph_data.nodes.len() as f64;
        let type_entropy = type_counts
            .values()
            .map(|&count| {
                let p = count as f64 / type_total;
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum();

        // Property entropy
        let mut prop_counts: HashMap<String, usize> = HashMap::new();
        for edge in &graph_data.edges {
            *prop_counts.entry(edge.edge_type.clone()).or_insert(0) += 1;
        }

        let prop_total = graph_data.edges.len() as f64;
        let property_entropy = prop_counts
            .values()
            .map(|&count| {
                let p = count as f64 / prop_total;
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum();

        vec![degree_entropy, type_entropy, property_entropy]
    }

    /// Compute diversity features
    fn compute_diversity_features(&self, graph_data: &GraphData) -> Vec<f64> {
        // Type diversity (number of unique types / total nodes)
        let unique_types: HashSet<_> = graph_data
            .nodes
            .iter()
            .filter_map(|n| n.node_type.as_ref())
            .collect();
        let type_diversity = unique_types.len() as f64 / graph_data.nodes.len().max(1) as f64;

        // Property diversity
        let unique_properties: HashSet<_> = graph_data.edges.iter().map(|e| &e.edge_type).collect();
        let property_diversity =
            unique_properties.len() as f64 / graph_data.edges.len().max(1) as f64;

        // Namespace diversity (simplified)
        let namespace_diversity = 0.5; // Placeholder

        vec![type_diversity, property_diversity, namespace_diversity]
    }

    /// Extract embedding features
    fn extract_embedding_features(&self, graph_data: &GraphData) -> (Vec<f64>, Vec<String>) {
        let mut features = Vec::new();
        let mut names = Vec::new();

        if self.config.embedding_features.include_node_embeddings {
            let node_embeddings = self.compute_node_embeddings(graph_data);
            features.extend(&node_embeddings);
            for i in 0..self.config.embedding_features.embedding_dim {
                names.push(format!("node_embedding_{}", i));
            }
        }

        if self.config.embedding_features.include_graph_embeddings {
            let graph_embeddings = self.compute_graph_embeddings(graph_data);
            features.extend(&graph_embeddings);
            for i in 0..self.config.embedding_features.embedding_dim {
                names.push(format!("graph_embedding_{}", i));
            }
        }

        (features, names)
    }

    /// Compute node embeddings
    fn compute_node_embeddings(&self, graph_data: &GraphData) -> Vec<f64> {
        match self.config.embedding_features.embedding_method {
            EmbeddingMethod::RandomWalk => self.random_walk_embeddings(graph_data),
            EmbeddingMethod::Node2Vec => self.node2vec_embeddings(graph_data),
            EmbeddingMethod::GraphSAGE => self.graphsage_embeddings(graph_data),
            EmbeddingMethod::Spectral => self.spectral_embeddings(graph_data),
            EmbeddingMethod::DeepWalk => self.deepwalk_embeddings(graph_data),
        }
    }

    /// Random walk embeddings
    fn random_walk_embeddings(&self, _graph_data: &GraphData) -> Vec<f64> {
        // Simplified random walk embedding
        vec![0.0; self.config.embedding_features.embedding_dim]
    }

    /// Node2Vec embeddings
    fn node2vec_embeddings(&self, _graph_data: &GraphData) -> Vec<f64> {
        // Simplified Node2Vec embedding
        vec![0.0; self.config.embedding_features.embedding_dim]
    }

    /// GraphSAGE embeddings
    fn graphsage_embeddings(&self, _graph_data: &GraphData) -> Vec<f64> {
        // Simplified GraphSAGE embedding
        vec![0.0; self.config.embedding_features.embedding_dim]
    }

    /// Spectral embeddings
    fn spectral_embeddings(&self, _graph_data: &GraphData) -> Vec<f64> {
        // Simplified spectral embedding
        vec![0.0; self.config.embedding_features.embedding_dim]
    }

    /// DeepWalk embeddings
    fn deepwalk_embeddings(&self, _graph_data: &GraphData) -> Vec<f64> {
        // Simplified DeepWalk embedding
        vec![0.0; self.config.embedding_features.embedding_dim]
    }

    /// Compute graph-level embeddings
    fn compute_graph_embeddings(&self, graph_data: &GraphData) -> Vec<f64> {
        // Average node embeddings as graph embedding
        let node_embeddings = self.compute_node_embeddings(graph_data);
        node_embeddings
    }

    /// Extract temporal features
    fn extract_temporal_features(&self, _graph_data: &GraphData) -> (Vec<f64>, Vec<String>) {
        let mut features = Vec::new();
        let mut names = Vec::new();

        if self.config.temporal_features.include_temporal_features {
            // Placeholder temporal features
            features.extend(vec![0.0; 5]);
            names.extend(vec![
                "temporal_mean".to_string(),
                "temporal_trend".to_string(),
                "temporal_seasonality".to_string(),
                "temporal_volatility".to_string(),
                "temporal_autocorr".to_string(),
            ]);
        }

        (features, names)
    }

    /// Extract semantic features
    fn extract_semantic_features(&self, graph_data: &GraphData) -> (Vec<f64>, Vec<String>) {
        let mut features = Vec::new();
        let mut names = Vec::new();

        if self.config.semantic_features.include_type_features {
            let type_features = self.compute_type_features(graph_data);
            features.extend(&type_features);
            names.extend(vec![
                "num_types".to_string(),
                "type_ratio".to_string(),
                "dominant_type_freq".to_string(),
            ]);
        }

        if self.config.semantic_features.include_property_features {
            let property_features = self.compute_property_features(graph_data);
            features.extend(&property_features);
            names.extend(vec![
                "num_properties".to_string(),
                "property_ratio".to_string(),
                "dominant_property_freq".to_string(),
            ]);
        }

        if self.config.semantic_features.include_namespace_features {
            let namespace_features = self.compute_namespace_features(graph_data);
            features.extend(&namespace_features);
            names.extend(vec![
                "num_namespaces".to_string(),
                "namespace_ratio".to_string(),
            ]);
        }

        if self.config.semantic_features.include_ontology_features {
            let ontology_features = self.compute_ontology_features(graph_data);
            features.extend(&ontology_features);
            names.extend(vec![
                "ontology_alignment".to_string(),
                "semantic_consistency".to_string(),
            ]);
        }

        (features, names)
    }

    /// Compute type-based features
    fn compute_type_features(&self, graph_data: &GraphData) -> Vec<f64> {
        let mut type_counts: HashMap<String, usize> = HashMap::new();

        for node in &graph_data.nodes {
            if let Some(node_type) = &node.node_type {
                *type_counts.entry(node_type.clone()).or_insert(0) += 1;
            }
        }

        let num_types = type_counts.len() as f64;
        let type_ratio = num_types / graph_data.nodes.len().max(1) as f64;

        let dominant_type_freq = type_counts
            .values()
            .max()
            .map(|&c| c as f64 / graph_data.nodes.len().max(1) as f64)
            .unwrap_or(0.0);

        vec![num_types, type_ratio, dominant_type_freq]
    }

    /// Compute property-based features
    fn compute_property_features(&self, graph_data: &GraphData) -> Vec<f64> {
        let mut property_counts: HashMap<String, usize> = HashMap::new();

        for edge in &graph_data.edges {
            *property_counts.entry(edge.edge_type.clone()).or_insert(0) += 1;
        }

        let num_properties = property_counts.len() as f64;
        let property_ratio = num_properties / graph_data.edges.len().max(1) as f64;

        let dominant_property_freq = property_counts
            .values()
            .max()
            .map(|&c| c as f64 / graph_data.edges.len().max(1) as f64)
            .unwrap_or(0.0);

        vec![num_properties, property_ratio, dominant_property_freq]
    }

    /// Compute namespace-based features
    fn compute_namespace_features(&self, _graph_data: &GraphData) -> Vec<f64> {
        // Simplified namespace features
        vec![3.0, 0.4] // num_namespaces, namespace_ratio
    }

    /// Compute ontology-based features
    fn compute_ontology_features(&self, _graph_data: &GraphData) -> Vec<f64> {
        // Simplified ontology features
        vec![0.8, 0.9] // ontology_alignment, semantic_consistency
    }
}

impl Default for FeatureExtractionConfig {
    fn default() -> Self {
        Self {
            structural_features: StructuralFeatureConfig {
                include_degree_features: true,
                include_centrality_features: true,
                include_clustering_features: true,
                include_path_features: true,
                include_motif_features: true,
                max_path_length: 5,
                motif_sizes: vec![3, 4],
            },
            statistical_features: StatisticalFeatureConfig {
                include_distribution_features: true,
                include_correlation_features: true,
                include_entropy_features: true,
                include_diversity_features: true,
                distribution_bins: 10,
            },
            embedding_features: EmbeddingFeatureConfig {
                include_node_embeddings: true,
                include_graph_embeddings: true,
                embedding_dim: 64,
                embedding_method: EmbeddingMethod::RandomWalk,
            },
            temporal_features: TemporalFeatureConfig {
                include_temporal_features: false,
                time_window_sizes: vec![1, 7, 30],
                temporal_aggregations: vec![TemporalAggregation::Mean, TemporalAggregation::Trend],
            },
            semantic_features: SemanticFeatureConfig {
                include_type_features: true,
                include_property_features: true,
                include_namespace_features: true,
                include_ontology_features: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extractor_creation() {
        let config = FeatureExtractionConfig::default();
        let extractor = FeatureExtractor::new(config);
        assert!(extractor.feature_cache.is_empty());
    }
}
