// Integration tests module

mod concurrent_tests;
mod memory_pool_tests;
// mod batch_execution_tests;
// mod streaming_results_tests;
// mod dataset_management_tests;

// Common test utilities
pub mod common {
    use std::time::Duration;
    use tokio::time::sleep;

    /// Wait for a condition to be true with timeout
    pub async fn wait_for_condition<F>(mut condition: F, timeout: Duration) -> bool
    where
        F: FnMut() -> bool,
    {
        let start = std::time::Instant::now();
        while start.elapsed() < timeout {
            if condition() {
                return true;
            }
            sleep(Duration::from_millis(10)).await;
        }
        false
    }

    /// Create test data for SPARQL queries
    pub fn create_test_dataset() -> Vec<(String, String, String)> {
        vec![
            ("http://example.org/subject1".to_string(),
             "http://example.org/predicate1".to_string(),
             "http://example.org/object1".to_string()),
            ("http://example.org/subject2".to_string(),
             "http://example.org/predicate1".to_string(),
             "http://example.org/object2".to_string()),
        ]
    }
}
