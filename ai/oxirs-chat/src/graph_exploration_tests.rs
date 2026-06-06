//! Tests for graph exploration module
#[cfg(test)]
mod tests {
    use crate::graph_exploration_types::{ExplorationConfig, ExplorationResults, GraphPath};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_exploration_config() {
        let config = ExplorationConfig::default();
        assert_eq!(config.max_depth, 5);
        assert_eq!(config.max_paths, 100);
        assert!(config.schema_aware);
    }

    #[tokio::test]
    async fn test_graph_path_creation() {
        let path = GraphPath {
            entities: vec!["entity1".to_string(), "entity2".to_string()],
            relationships: vec!["relationship1".to_string()],
            length: 1,
            relevance_score: 0.8,
            explanation: "Test path".to_string(),
            metadata: HashMap::new(),
        };

        assert_eq!(path.length, 1);
        assert_eq!(path.relevance_score, 0.8);
    }

    #[tokio::test]
    async fn test_exploration_results() {
        let mut results = ExplorationResults::new();
        results.add_metadata("test_key".to_string(), "test_value".to_string());

        let summary = results.get_summary();
        assert!(summary.contains("0 paths found"));

        assert!(results.to_json().is_ok());
    }
}
