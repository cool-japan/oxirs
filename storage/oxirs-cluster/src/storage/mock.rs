//! Mock storage backend for testing
//!
//! Provides an in-memory implementation of StorageBackend for integration tests

use crate::shard::ShardId;
use crate::storage::persistent::StorageBackend;
use anyhow::Result;
use async_trait::async_trait;
use oxirs_core::model::Triple;
use oxirs_core::RdfTerm;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Mock storage backend using in-memory HashMap
#[derive(Debug, Default, Clone)]
pub struct MockStorageBackend {
    shards: Arc<RwLock<HashMap<ShardId, Vec<Triple>>>>,
}

impl MockStorageBackend {
    /// Create a new mock storage backend
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the number of shards
    pub async fn shard_count(&self) -> usize {
        self.shards.read().await.len()
    }

    /// Clear all data
    pub async fn clear(&self) {
        self.shards.write().await.clear();
    }
}

#[async_trait]
impl StorageBackend for MockStorageBackend {
    async fn create_shard(&self, shard_id: ShardId) -> Result<()> {
        self.shards.write().await.insert(shard_id, Vec::new());
        Ok(())
    }

    async fn delete_shard(&self, shard_id: ShardId) -> Result<()> {
        self.shards.write().await.remove(&shard_id);
        Ok(())
    }

    async fn insert_triple_to_shard(&self, shard_id: ShardId, triple: Triple) -> Result<()> {
        let mut shards = self.shards.write().await;
        if let Some(shard) = shards.get_mut(&shard_id) {
            shard.push(triple);
        } else {
            // Create shard if it doesn't exist
            shards.insert(shard_id, vec![triple]);
        }
        Ok(())
    }

    async fn delete_triple_from_shard(&self, shard_id: ShardId, triple: &Triple) -> Result<()> {
        let mut shards = self.shards.write().await;
        if let Some(shard) = shards.get_mut(&shard_id) {
            shard.retain(|t| t != triple);
        }
        Ok(())
    }

    async fn query_shard(
        &self,
        shard_id: ShardId,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Result<Vec<Triple>> {
        let shards = self.shards.read().await;
        if let Some(shard) = shards.get(&shard_id) {
            let results: Vec<Triple> = shard
                .iter()
                .filter(|triple| {
                    // Extract IRI from NamedNode without angle brackets for comparison
                    let subject_match = subject.map_or(true, |s| {
                        if let oxirs_core::model::Subject::NamedNode(named_node) = triple.subject()
                        {
                            named_node.as_str() == s
                        } else {
                            triple.subject().to_string() == s
                        }
                    });
                    let predicate_match =
                        predicate.map_or(true, |p| triple.predicate().as_str() == p);
                    let object_match = object.map_or(true, |o| {
                        if let oxirs_core::Object::NamedNode(named_node) = triple.object() {
                            named_node.as_str() == o
                        } else {
                            triple.object().to_string() == o
                        }
                    });

                    subject_match && predicate_match && object_match
                })
                .cloned()
                .collect();

            Ok(results)
        } else {
            Ok(Vec::new())
        }
    }

    async fn get_shard_size(&self, shard_id: ShardId) -> Result<u64> {
        let shards = self.shards.read().await;
        if let Some(shard) = shards.get(&shard_id) {
            // Estimate size as 100 bytes per triple
            Ok((shard.len() * 100) as u64)
        } else {
            Ok(0)
        }
    }

    async fn get_shard_triple_count(&self, shard_id: ShardId) -> Result<usize> {
        let shards = self.shards.read().await;
        Ok(shards.get(&shard_id).map_or(0, |s| s.len()))
    }

    async fn export_shard(&self, shard_id: ShardId) -> Result<Vec<Triple>> {
        let shards = self.shards.read().await;
        Ok(shards.get(&shard_id).cloned().unwrap_or_default())
    }

    async fn import_shard(&self, shard_id: ShardId, triples: Vec<Triple>) -> Result<()> {
        self.shards.write().await.insert(shard_id, triples);
        Ok(())
    }

    async fn get_shard_triples(&self, shard_id: ShardId) -> Result<Vec<Triple>> {
        let shards = self.shards.read().await;
        Ok(shards.get(&shard_id).cloned().unwrap_or_default())
    }

    async fn insert_triples_to_shard(&self, shard_id: ShardId, triples: Vec<Triple>) -> Result<()> {
        let mut shards = self.shards.write().await;
        if let Some(shard) = shards.get_mut(&shard_id) {
            shard.extend(triples);
        } else {
            shards.insert(shard_id, triples);
        }
        Ok(())
    }

    async fn mark_shard_for_deletion(&self, shard_id: ShardId) -> Result<()> {
        // In the mock implementation, we can just remove the shard immediately
        self.shards.write().await.remove(&shard_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::{Literal, NamedNode, Subject};

    #[tokio::test]
    async fn test_mock_backend_basic_operations() {
        let backend = MockStorageBackend::new();
        let shard_id = 1;

        // Create shard
        backend.create_shard(shard_id).await.unwrap();
        assert_eq!(backend.shard_count().await, 1);

        // Insert triple
        let triple = Triple::new(
            Subject::NamedNode(NamedNode::new("http://example.org/subject").unwrap()),
            NamedNode::new("http://example.org/predicate").unwrap(),
            oxirs_core::Object::Literal(Literal::new("object")),
        );
        backend
            .insert_triple_to_shard(shard_id, triple.clone())
            .await
            .unwrap();

        // Query
        let results = backend
            .query_shard(shard_id, None, None, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);

        // Delete triple
        backend
            .delete_triple_from_shard(shard_id, &triple)
            .await
            .unwrap();
        let results = backend
            .query_shard(shard_id, None, None, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 0);

        // Delete shard
        backend.delete_shard(shard_id).await.unwrap();
        assert_eq!(backend.shard_count().await, 0);
    }
}
