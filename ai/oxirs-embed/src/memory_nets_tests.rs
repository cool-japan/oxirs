//! Tests for memory-augmented network components.

#[cfg(test)]
mod tests {
    use scirs2_core::ndarray_ext::{Array1, Array2};

    use crate::memory_nets_controller::{
        ControllerNetwork, DNCConfig, DifferentiableNeuralComputer, ReadHead, WriteHead,
    };
    use crate::memory_nets_ops::{
        EpisodicConfig, EpisodicMemory, MemoryAugmentedNetwork, MemoryConfig, MemoryNetworks,
        MemoryNetworksConfig, SparseAccessMemory, SparseConfig,
    };

    #[tokio::test]
    async fn test_memory_augmented_network_creation() {
        let config = MemoryConfig::default();
        let network = MemoryAugmentedNetwork::new(config);
        assert!(network.is_ok());
    }

    #[tokio::test]
    async fn test_dnc_forward_pass() {
        let config = DNCConfig::default();
        let mut dnc = DifferentiableNeuralComputer::new(config);
        let input = Array1::zeros(64);

        let result = dnc.forward(&input);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_memory_networks_store_and_query() {
        let config = MemoryNetworksConfig::default();
        let mut memory_net = MemoryNetworks::new(config);

        let embedding = Array1::ones(128);
        let result = memory_net.store_memory("test content".to_string(), embedding.clone());
        assert!(result.is_ok());

        let query_result = memory_net.query(&embedding);
        assert!(query_result.is_ok());
    }

    #[tokio::test]
    async fn test_episodic_memory() {
        let config = EpisodicConfig::default();
        let mut episodic = EpisodicMemory::new(config);

        episodic.start_episode("test".to_string());

        let state = Array1::ones(128);
        let result = episodic.add_state(state, 1.0);
        assert!(result.is_ok());

        let end_result = episodic.end_episode(true);
        assert!(end_result.is_ok());
    }

    #[tokio::test]
    async fn test_sparse_memory() {
        let config = SparseConfig::default();
        let mut sparse = SparseAccessMemory::new(config);

        let value = Array1::ones(512);
        let store_result = sparse.store(123, value.clone());
        assert!(store_result.is_ok());

        let retrieved = sparse.retrieve(123);
        assert!(retrieved.is_some());

        let similar = sparse.find_similar(&value, 1);
        assert_eq!(similar.len(), 1);
    }

    #[test]
    fn test_controller_network() {
        let mut controller = ControllerNetwork::new(100, 256, 128);
        let input = Array1::zeros(100);

        let output = controller.forward(&input);
        assert_eq!(output.len(), 128);
    }

    #[test]
    fn test_read_head_weighting() {
        let read_head = ReadHead::new(64);
        let memory = Array2::zeros((128, 64));
        let link_matrix = Array2::zeros((128, 128));
        let prev_weighting = Array1::zeros(128);

        let weighting = read_head.generate_weighting(&memory, &link_matrix, &prev_weighting);
        assert_eq!(weighting.len(), 128);

        let sum = weighting.sum();
        assert!((sum - 1.0).abs() < 1e-6 || sum == 0.0);
    }

    #[test]
    fn test_write_head_operations() {
        let write_head = WriteHead::new(64);
        let memory = Array2::zeros((128, 64));
        let usage_vector = Array1::zeros(128);

        let weighting = write_head.generate_weighting(&memory, &usage_vector);
        assert_eq!(weighting.len(), 128);
    }
}
