#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestEvent {
        id: u64,
        value: String,
    }

    #[tokio::test]
    async fn test_deadlock_repro() {
        // Configure to trigger retention policy immediately
        let config = VersioningConfig {
            max_versions: 1,
            ..Default::default()
        };
        let versioning = StreamVersioning::<TestEvent>::new(config);

        // Create version 1
        versioning
            .create_version(vec![], "v1")
            .await
            .unwrap();

        // Create version 2 - this should trigger retention policy and deadlock
        // because create_version holds locks and calls apply_retention_policy which wants locks
        versioning
            .create_version(vec![], "v2")
            .await
            .unwrap();
    }
}
