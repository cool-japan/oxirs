//! Tests for GNN.

#[cfg(test)]
mod tests {
    use crate::ml::gnn::{GNNArchitecture, GNNConfig, GraphNeuralNetwork};

    #[test]
    fn test_gnn_creation() {
        let config = GNNConfig::default();
        let gnn = GraphNeuralNetwork::new(config);
        assert_eq!(gnn.layers.len(), 3);
    }

    #[test]
    fn test_gnn_architectures() {
        let architectures = vec![
            GNNArchitecture::GCN,
            GNNArchitecture::GAT,
            GNNArchitecture::GIN,
            GNNArchitecture::GraphSAGE,
            GNNArchitecture::MPNN,
        ];

        for arch in architectures {
            let config = GNNConfig {
                architecture: arch,
                ..Default::default()
            };
            let gnn = GraphNeuralNetwork::new(config);
            assert_eq!(gnn.layers.len(), 3);
        }
    }
}
