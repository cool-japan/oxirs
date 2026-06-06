//! GNN layer implementations: message passing, aggregation, readout functions.

use std::collections::HashMap;

use scirs2_core::ndarray_ext::{Array2, Array3, Axis};
use scirs2_core::random::{Random, Rng, RngExt};

use super::{GraphData, ModelError};
use crate::ml::gnn_types::{
    ActivationFunction, AggregationFunction, GNNArchitecture, GNNConfig, ScoringFunction,
};

/// GNN layer types (internal enum)
#[derive(Debug)]
pub enum GNNLayer {
    Gcn(GCNLayerState),
    Gat(GATLayerState),
    Gin(GINLayerState),
    GraphSAGE(GraphSAGELayerState),
    Mpnn(MPNNLayerState),
    GraphCompletion(GraphCompletionLayerState),
    EntityCompletion(EntityCompletionLayerState),
    RelationCompletion(RelationCompletionLayerState),
    GraphTransformer(GraphTransformerLayerState),
    HierarchicalGraphTransformer(HierarchicalGraphTransformerLayerState),
}

#[derive(Debug)]
pub struct GCNLayerState {
    pub weight: Array2<f64>,
    pub bias: Option<Array2<f64>>,
    pub input_dim: usize,
    pub output_dim: usize,
}

#[derive(Debug)]
pub struct GATLayerState {
    pub weight: Array2<f64>,
    pub attention_weight: Array2<f64>,
    pub bias: Option<Array2<f64>>,
    pub num_heads: usize,
    pub input_dim: usize,
    pub output_dim: usize,
}

#[derive(Debug)]
pub struct GINLayerState {
    pub mlp_layers: Vec<Array2<f64>>,
    pub epsilon: f64,
    pub input_dim: usize,
    pub output_dim: usize,
}

#[derive(Debug)]
pub struct GraphSAGELayerState {
    pub weight_self: Array2<f64>,
    pub weight_neighbor: Array2<f64>,
    pub bias: Option<Array2<f64>>,
    pub input_dim: usize,
    pub output_dim: usize,
}

#[derive(Debug)]
pub struct MPNNLayerState {
    pub message_net: Vec<Array2<f64>>,
    pub update_net: Vec<Array2<f64>>,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
}

#[derive(Debug)]
pub struct GraphCompletionLayerState {
    pub entity_encoder: Array2<f64>,
    pub relation_encoder: Array2<f64>,
    pub scoring_function: ScoringFunction,
    pub input_dim: usize,
    pub output_dim: usize,
}

#[derive(Debug)]
pub struct EntityCompletionLayerState {
    pub entity_embedding: Array2<f64>,
    pub context_encoder: Array2<f64>,
    pub completion_head: Array2<f64>,
    pub input_dim: usize,
    pub output_dim: usize,
}

#[derive(Debug)]
pub struct RelationCompletionLayerState {
    pub relation_embedding: Array2<f64>,
    pub pattern_encoder: Array2<f64>,
    pub completion_head: Array2<f64>,
    pub input_dim: usize,
    pub output_dim: usize,
}

#[derive(Debug, Clone)]
pub struct GraphTransformerLayerState {
    pub attention_weights: Array2<f64>,
    pub feed_forward: Array2<f64>,
}

#[derive(Debug, Clone)]
pub struct HierarchicalGraphTransformerLayerState {
    pub hierarchical_attention: Array3<f64>,
    pub level_encoders: Vec<Array2<f64>>,
}

/// Output layer for shape prediction
#[derive(Debug)]
pub struct OutputLayer {
    pub shape_classifier: Array2<f64>,
    pub constraint_heads: HashMap<String, Array2<f64>>,
    pub output_dim: usize,
}

/// Graph embedding result
#[derive(Debug)]
pub struct GraphEmbedding {
    pub node_embeddings: Array2<f64>,
    pub graph_embedding: Array2<f64>,
    pub attention_weights: Option<Array3<f64>>,
}

/// Gradients for a single layer
#[derive(Debug, Clone)]
pub struct LayerGradients {
    pub weight_gradients: Array2<f64>,
    pub bias_gradients: Option<Array2<f64>>,
}

// ─── Layer initializers ──────────────────────────────────────────────────────

pub fn init_gcn_layer(input_dim: usize, output_dim: usize) -> GCNLayerState {
    let mut rng = Random::default();
    let scale = (2.0 / input_dim as f64).sqrt();
    let weight =
        Array2::from_shape_fn((input_dim, output_dim), |_| rng.random_range(-scale..scale));
    let bias = Some(Array2::zeros((1, output_dim)));
    GCNLayerState {
        weight,
        bias,
        input_dim,
        output_dim,
    }
}

pub fn init_gat_layer(input_dim: usize, output_dim: usize, num_heads: usize) -> GATLayerState {
    let mut rng = Random::default();
    let scale = (2.0 / input_dim as f64).sqrt();
    let weight = Array2::from_shape_fn((input_dim, output_dim * num_heads), |_| {
        rng.random_range(-scale..scale)
    });
    let attention_weight = Array2::from_shape_fn((2 * output_dim, num_heads), |_| {
        rng.random_range(-scale..scale)
    });
    let bias = Some(Array2::zeros((1, output_dim * num_heads)));
    GATLayerState {
        weight,
        attention_weight,
        bias,
        num_heads,
        input_dim,
        output_dim,
    }
}

pub fn init_gin_layer(input_dim: usize, output_dim: usize) -> GINLayerState {
    let mut rng = Random::default();
    let hidden_dim = (input_dim + output_dim) / 2;
    let mlp_layers = vec![
        Array2::from_shape_fn((input_dim, hidden_dim), |_| rng.random_range(-0.1..0.1)),
        Array2::from_shape_fn((hidden_dim, output_dim), |_| rng.random_range(-0.1..0.1)),
    ];
    GINLayerState {
        mlp_layers,
        epsilon: 0.0,
        input_dim,
        output_dim,
    }
}

pub fn init_graphsage_layer(input_dim: usize, output_dim: usize) -> GraphSAGELayerState {
    let mut rng = Random::default();
    let scale = (2.0 / input_dim as f64).sqrt();
    let weight_self =
        Array2::from_shape_fn((input_dim, output_dim), |_| rng.random_range(-scale..scale));
    let weight_neighbor =
        Array2::from_shape_fn((input_dim, output_dim), |_| rng.random_range(-scale..scale));
    let bias = Some(Array2::zeros((1, output_dim)));
    GraphSAGELayerState {
        weight_self,
        weight_neighbor,
        bias,
        input_dim,
        output_dim,
    }
}

pub fn init_mpnn_layer(input_dim: usize, output_dim: usize) -> MPNNLayerState {
    let mut rng = Random::default();
    let hidden_dim = (input_dim + output_dim) / 2;
    let message_net = vec![
        Array2::from_shape_fn((input_dim * 2, hidden_dim), |_| rng.random_range(-0.1..0.1)),
        Array2::from_shape_fn((hidden_dim, input_dim), |_| rng.random_range(-0.1..0.1)),
    ];
    let update_net = vec![
        Array2::from_shape_fn((input_dim * 2, hidden_dim), |_| rng.random_range(-0.1..0.1)),
        Array2::from_shape_fn((hidden_dim, output_dim), |_| rng.random_range(-0.1..0.1)),
    ];
    MPNNLayerState {
        message_net,
        update_net,
        input_dim,
        hidden_dim,
        output_dim,
    }
}

pub fn init_graph_completion_layer(
    input_dim: usize,
    output_dim: usize,
) -> GraphCompletionLayerState {
    let mut rng = Random::default();
    let scale = (2.0 / input_dim as f64).sqrt();
    let entity_encoder =
        Array2::from_shape_fn((input_dim, output_dim), |_| rng.random_range(-scale..scale));
    let relation_encoder =
        Array2::from_shape_fn((input_dim, output_dim), |_| rng.random_range(-scale..scale));
    GraphCompletionLayerState {
        entity_encoder,
        relation_encoder,
        scoring_function: ScoringFunction::TransE,
        input_dim,
        output_dim,
    }
}

pub fn init_entity_completion_layer(
    input_dim: usize,
    output_dim: usize,
) -> EntityCompletionLayerState {
    let mut rng = Random::default();
    let scale = (2.0 / input_dim as f64).sqrt();
    let entity_embedding =
        Array2::from_shape_fn((input_dim, output_dim), |_| rng.random_range(-scale..scale));
    let context_encoder =
        Array2::from_shape_fn((input_dim, output_dim), |_| rng.random_range(-scale..scale));
    let completion_head = Array2::from_shape_fn((output_dim, output_dim), |_| {
        rng.random_range(-scale..scale)
    });
    EntityCompletionLayerState {
        entity_embedding,
        context_encoder,
        completion_head,
        input_dim,
        output_dim,
    }
}

pub fn init_relation_completion_layer(
    input_dim: usize,
    output_dim: usize,
) -> RelationCompletionLayerState {
    let mut rng = Random::default();
    let scale = (2.0 / input_dim as f64).sqrt();
    let relation_embedding =
        Array2::from_shape_fn((input_dim, output_dim), |_| rng.random_range(-scale..scale));
    let pattern_encoder =
        Array2::from_shape_fn((input_dim, output_dim), |_| rng.random_range(-scale..scale));
    let completion_head = Array2::from_shape_fn((output_dim, output_dim), |_| {
        rng.random_range(-scale..scale)
    });
    RelationCompletionLayerState {
        relation_embedding,
        pattern_encoder,
        completion_head,
        input_dim,
        output_dim,
    }
}

pub fn init_graph_transformer_layer(
    input_dim: usize,
    output_dim: usize,
) -> GraphTransformerLayerState {
    GraphTransformerLayerState {
        attention_weights: Array2::from_elem((input_dim, output_dim), 0.1),
        feed_forward: Array2::from_elem((output_dim, output_dim), 0.1),
    }
}

pub fn init_hierarchical_graph_transformer_layer(
    input_dim: usize,
    output_dim: usize,
) -> HierarchicalGraphTransformerLayerState {
    HierarchicalGraphTransformerLayerState {
        hierarchical_attention: Array3::from_elem((2, input_dim, output_dim), 0.1),
        level_encoders: vec![Array2::from_elem((input_dim, output_dim), 0.1); 3],
    }
}

pub fn initialize_layers(config: &GNNConfig) -> Vec<GNNLayer> {
    let mut layers = Vec::new();
    let mut current_dim = config.hidden_dim;

    for i in 0..config.num_layers {
        let output_dim = if i == config.num_layers - 1 {
            config.output_dim
        } else {
            config.hidden_dim
        };

        let layer = match config.architecture {
            GNNArchitecture::GCN => GNNLayer::Gcn(init_gcn_layer(current_dim, output_dim)),
            GNNArchitecture::GAT => GNNLayer::Gat(init_gat_layer(
                current_dim,
                output_dim,
                config.attention_heads,
            )),
            GNNArchitecture::GIN => GNNLayer::Gin(init_gin_layer(current_dim, output_dim)),
            GNNArchitecture::GraphSAGE => {
                GNNLayer::GraphSAGE(init_graphsage_layer(current_dim, output_dim))
            }
            GNNArchitecture::MPNN => GNNLayer::Mpnn(init_mpnn_layer(current_dim, output_dim)),
            GNNArchitecture::GraphCompletion => {
                GNNLayer::GraphCompletion(init_graph_completion_layer(current_dim, output_dim))
            }
            GNNArchitecture::EntityCompletion => {
                GNNLayer::EntityCompletion(init_entity_completion_layer(current_dim, output_dim))
            }
            GNNArchitecture::RelationCompletion => GNNLayer::RelationCompletion(
                init_relation_completion_layer(current_dim, output_dim),
            ),
            GNNArchitecture::GraphTransformer => {
                GNNLayer::GraphTransformer(init_graph_transformer_layer(current_dim, output_dim))
            }
            GNNArchitecture::HierarchicalGraphTransformer => {
                GNNLayer::HierarchicalGraphTransformer(init_hierarchical_graph_transformer_layer(
                    current_dim,
                    output_dim,
                ))
            }
        };

        layers.push(layer);
        current_dim = output_dim;
    }

    layers
}

pub fn initialize_output_layer(config: &GNNConfig) -> OutputLayer {
    let mut rng = Random::default();
    let num_shape_types = 10;
    let shape_classifier = Array2::from_shape_fn((config.output_dim, num_shape_types), |_| {
        rng.random_range(-0.1..0.1)
    });

    let mut constraint_heads = HashMap::new();
    let constraint_types = vec![
        "minCount",
        "maxCount",
        "datatype",
        "pattern",
        "minLength",
        "maxLength",
        "minInclusive",
        "maxInclusive",
        "class",
        "nodeKind",
    ];

    for constraint_type in constraint_types {
        let head_dim = match constraint_type {
            "datatype" | "class" | "nodeKind" => 50,
            _ => 10,
        };
        let head = Array2::from_shape_fn((config.output_dim, head_dim), |_| {
            rng.random_range(-0.1..0.1)
        });
        constraint_heads.insert(constraint_type.to_string(), head);
    }

    OutputLayer {
        shape_classifier,
        constraint_heads,
        output_dim: config.output_dim,
    }
}

// ─── Forward pass helpers ────────────────────────────────────────────────────

pub fn apply_activation(features: &Array2<f64>, activation: &ActivationFunction) -> Array2<f64> {
    match activation {
        ActivationFunction::ReLU => features.mapv(|x| x.max(0.0)),
        ActivationFunction::LeakyReLU(alpha) => {
            features.mapv(|x| if x > 0.0 { x } else { alpha * x })
        }
        ActivationFunction::ELU(alpha) => {
            features.mapv(|x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
        }
        ActivationFunction::Tanh => features.mapv(|x| x.tanh()),
        ActivationFunction::Sigmoid => features.mapv(|x| 1.0 / (1.0 + (-x).exp())),
        ActivationFunction::GELU => {
            features.mapv(|x| 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh()))
        }
    }
}

pub fn batch_normalize(features: &Array2<f64>) -> Array2<f64> {
    let mean = features
        .mean_axis(Axis(0))
        .expect("features array should have valid axis");
    let var = features.var_axis(Axis(0), 0.0);
    let std = var.mapv(|v| (v + 1e-5).sqrt());
    let mut normalized = features.clone();
    for i in 0..features.nrows() {
        for j in 0..features.ncols() {
            normalized[[i, j]] = (features[[i, j]] - mean[j]) / std[j];
        }
    }
    normalized
}

pub fn global_pool(features: &Array2<f64>, aggregation: &AggregationFunction) -> Array2<f64> {
    match aggregation {
        AggregationFunction::Sum => features.sum_axis(Axis(0)).insert_axis(Axis(0)),
        AggregationFunction::Mean => features
            .mean_axis(Axis(0))
            .expect("mean_axis should succeed")
            .insert_axis(Axis(0)),
        AggregationFunction::Max => features
            .fold_axis(Axis(0), f64::NEG_INFINITY, |&a, &b| a.max(b))
            .insert_axis(Axis(0)),
        AggregationFunction::Min => features
            .fold_axis(Axis(0), f64::INFINITY, |&a, &b| a.min(b))
            .insert_axis(Axis(0)),
        AggregationFunction::Attention => features
            .mean_axis(Axis(0))
            .expect("mean_axis should succeed")
            .insert_axis(Axis(0)),
    }
}

pub fn gcn_forward(
    layer: &GCNLayerState,
    features: &Array2<f64>,
    adj_matrix: &Array2<f64>,
) -> Result<Array2<f64>, ModelError> {
    let ah = adj_matrix.dot(features);
    let mut output = ah.dot(&layer.weight);
    if let Some(bias) = &layer.bias {
        output += bias;
    }
    Ok(output)
}

pub fn gat_forward(
    layer: &GATLayerState,
    features: &Array2<f64>,
    _adj_matrix: &Array2<f64>,
) -> Result<Array2<f64>, ModelError> {
    let transformed = features.dot(&layer.weight);
    if let Some(bias) = &layer.bias {
        Ok(transformed + bias)
    } else {
        Ok(transformed)
    }
}

pub fn gin_forward(
    layer: &GINLayerState,
    features: &Array2<f64>,
    adj_matrix: &Array2<f64>,
    activation: &ActivationFunction,
) -> Result<Array2<f64>, ModelError> {
    let neighbor_sum = adj_matrix.dot(features);
    let self_features = features * (1.0 + layer.epsilon);
    let combined = self_features + neighbor_sum;
    let mut output = combined;
    for mlp_layer in &layer.mlp_layers {
        output = output.dot(mlp_layer);
        output = apply_activation(&output, &ActivationFunction::ReLU);
    }
    Ok(output)
}

pub fn graphsage_forward(
    layer: &GraphSAGELayerState,
    features: &Array2<f64>,
    adj_matrix: &Array2<f64>,
) -> Result<Array2<f64>, ModelError> {
    let neighbor_agg = adj_matrix.dot(features);
    let self_transformed = features.dot(&layer.weight_self);
    let neighbor_transformed = neighbor_agg.dot(&layer.weight_neighbor);
    let mut output = self_transformed + neighbor_transformed;
    if let Some(bias) = &layer.bias {
        output += bias;
    }
    let norms = output.mapv(|x| x * x).sum_axis(Axis(1)).mapv(|x| x.sqrt());
    for i in 0..output.nrows() {
        if norms[i] > 0.0 {
            output.row_mut(i).mapv_inplace(|x| x / norms[i]);
        }
    }
    Ok(output)
}

pub fn mpnn_forward(
    layer: &MPNNLayerState,
    features: &Array2<f64>,
    adj_matrix: &Array2<f64>,
    activation: &ActivationFunction,
) -> Result<Array2<f64>, ModelError> {
    let messages = adj_matrix.dot(features);
    let mut combined = Array2::zeros((features.nrows(), layer.input_dim * 2));
    for i in 0..features.nrows() {
        for j in 0..layer.input_dim.min(features.ncols()) {
            combined[[i, j]] = features[[i, j]];
        }
    }
    for i in 0..messages.nrows().min(combined.nrows()) {
        for j in 0..layer.input_dim.min(messages.ncols()) {
            combined[[i, layer.input_dim + j]] = messages[[i, j]];
        }
    }
    let mut output = combined;
    for update_layer in &layer.update_net {
        output = output.dot(update_layer);
        output = apply_activation(&output, &ActivationFunction::ReLU);
    }
    Ok(output)
}

pub fn graph_completion_forward(
    layer: &GraphCompletionLayerState,
    features: &Array2<f64>,
    _adj_matrix: &Array2<f64>,
    graph_data: &GraphData,
) -> Result<Array2<f64>, ModelError> {
    let entity_embeddings = features.dot(&layer.entity_encoder);
    let num_relations = graph_data.edges.len().max(1);
    let mut relation_features = Array2::zeros((num_relations, layer.input_dim));
    for (i, edge) in graph_data.edges.iter().enumerate() {
        if i < num_relations {
            for (j, feature_val) in edge.properties.values().take(layer.input_dim).enumerate() {
                relation_features[[i, j]] = *feature_val;
            }
        }
    }
    let relation_embeddings = relation_features.dot(&layer.relation_encoder);
    let completion_scores = match layer.scoring_function {
        ScoringFunction::TransE => transe_scoring(&entity_embeddings, &relation_embeddings),
        ScoringFunction::DistMult => distmult_scoring(&entity_embeddings, &relation_embeddings),
        ScoringFunction::ComplEx => distmult_scoring(&entity_embeddings, &relation_embeddings),
        ScoringFunction::RotatE => transe_scoring(&entity_embeddings, &relation_embeddings),
        ScoringFunction::ConvE => distmult_scoring(&entity_embeddings, &relation_embeddings),
    };
    Ok(completion_scores)
}

pub fn entity_completion_forward(
    layer: &EntityCompletionLayerState,
    features: &Array2<f64>,
    _adj_matrix: &Array2<f64>,
    _graph_data: &GraphData,
) -> Result<Array2<f64>, ModelError> {
    let entity_embeddings = features.dot(&layer.entity_embedding);
    let context_embeddings = features.dot(&layer.context_encoder);
    let combined = entity_embeddings + context_embeddings;
    Ok(combined.dot(&layer.completion_head))
}

pub fn relation_completion_forward(
    layer: &RelationCompletionLayerState,
    features: &Array2<f64>,
    _adj_matrix: &Array2<f64>,
    graph_data: &GraphData,
) -> Result<Array2<f64>, ModelError> {
    let pattern_embeddings = features.dot(&layer.pattern_encoder);
    let num_edges = graph_data.edges.len().max(1);
    let mut relation_features = Array2::zeros((num_edges, layer.input_dim));
    for (i, edge) in graph_data.edges.iter().enumerate() {
        if i < num_edges {
            for (j, feature_val) in edge.properties.values().take(layer.input_dim).enumerate() {
                relation_features[[i, j]] = *feature_val;
            }
        }
    }
    let relation_embeddings = relation_features.dot(&layer.relation_embedding);
    let (pattern_rows, pattern_cols) = pattern_embeddings.dim();
    let (relation_rows, _) = relation_embeddings.dim();
    let combined = if pattern_rows >= relation_rows {
        let mut combined = pattern_embeddings.clone();
        for i in 0..relation_rows.min(pattern_rows) {
            for j in 0..pattern_cols.min(layer.output_dim) {
                combined[[i, j]] += relation_embeddings[[i, j]];
            }
        }
        combined
    } else {
        let mut combined = Array2::zeros((relation_rows, pattern_cols));
        for i in 0..pattern_rows.min(relation_rows) {
            for j in 0..pattern_cols {
                combined[[i, j]] = pattern_embeddings[[i, j]] + relation_embeddings[[i, j]];
            }
        }
        combined
    };
    Ok(combined.dot(&layer.completion_head))
}

pub fn graph_transformer_forward(
    layer: &GraphTransformerLayerState,
    input: &Array2<f64>,
    _adj_matrix: &Array2<f64>,
) -> Result<Array2<f64>, ModelError> {
    let attended = input.dot(&layer.attention_weights);
    Ok(attended.dot(&layer.feed_forward))
}

pub fn hierarchical_graph_transformer_forward(
    layer: &HierarchicalGraphTransformerLayerState,
    input: &Array2<f64>,
    _adj_matrix: &Array2<f64>,
) -> Result<Array2<f64>, ModelError> {
    let level0 = input.dot(&layer.level_encoders[0]);
    let level1 = level0.dot(&layer.level_encoders[1]);
    Ok(level1.dot(&layer.level_encoders[2]))
}

pub fn prepare_graph_data(
    graph_data: &GraphData,
    hidden_dim: usize,
) -> Result<(Array2<f64>, Array2<f64>), ModelError> {
    let num_nodes = graph_data.nodes.len();
    let mut adj_matrix = Array2::zeros((num_nodes, num_nodes));
    for edge in &graph_data.edges {
        let source_idx = graph_data
            .nodes
            .iter()
            .position(|n| n.node_id == edge.source_id)
            .ok_or_else(|| ModelError::PredictionError("Source node not found".to_string()))?;
        let target_idx = graph_data
            .nodes
            .iter()
            .position(|n| n.node_id == edge.target_id)
            .ok_or_else(|| ModelError::PredictionError("Target node not found".to_string()))?;
        adj_matrix[[source_idx, target_idx]] = 1.0;
        adj_matrix[[target_idx, source_idx]] = 1.0;
    }
    for i in 0..num_nodes {
        adj_matrix[[i, i]] = 1.0;
    }
    let degree = adj_matrix.sum_axis(Axis(1));
    let degree_inv_sqrt = degree.mapv(|d: f64| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 });
    for i in 0..num_nodes {
        for j in 0..num_nodes {
            adj_matrix[[i, j]] *= degree_inv_sqrt[i] * degree_inv_sqrt[j];
        }
    }
    let feature_dim = graph_data
        .nodes
        .first()
        .and_then(|n| n.embedding.as_ref())
        .map(|e| e.len())
        .unwrap_or(hidden_dim);
    let mut node_features = Array2::zeros((num_nodes, feature_dim));
    for (i, node) in graph_data.nodes.iter().enumerate() {
        if let Some(embedding) = &node.embedding {
            for (j, &val) in embedding.iter().enumerate() {
                if j < feature_dim {
                    node_features[[i, j]] = val;
                }
            }
        } else {
            let mut rng = Random::default();
            for j in 0..feature_dim {
                node_features[[i, j]] = rng.random_range(-0.1..0.1);
            }
        }
    }
    Ok((adj_matrix, node_features))
}

// ─── Scoring helpers ──────────────────────────────────────────────────────────

pub fn transe_scoring(
    entity_embeddings: &Array2<f64>,
    relation_embeddings: &Array2<f64>,
) -> Array2<f64> {
    let num_entities = entity_embeddings.nrows();
    let num_relations = relation_embeddings.nrows();
    let mut scores = Array2::zeros((num_entities, num_relations));
    for i in 0..num_entities {
        for j in 0..num_relations.min(num_entities) {
            let h = entity_embeddings.row(i);
            let r = relation_embeddings.row(j);
            let t = entity_embeddings.row(j);
            let diff = &h + &r - t;
            let norm = diff.mapv(|x| x * x).sum().sqrt();
            scores[[i, j]] = -norm;
        }
    }
    scores
}

pub fn distmult_scoring(
    entity_embeddings: &Array2<f64>,
    relation_embeddings: &Array2<f64>,
) -> Array2<f64> {
    let num_entities = entity_embeddings.nrows();
    let num_relations = relation_embeddings.nrows();
    let mut scores = Array2::zeros((num_entities, num_relations));
    for i in 0..num_entities {
        for j in 0..num_relations.min(num_entities) {
            let h = entity_embeddings.row(i);
            let r = relation_embeddings.row(j);
            let t = entity_embeddings.row(j);
            let score = h
                .iter()
                .zip(r.iter())
                .zip(t.iter())
                .map(|((h_val, r_val), t_val)| h_val * r_val * t_val)
                .sum::<f64>();
            scores[[i, j]] = score;
        }
    }
    scores
}
