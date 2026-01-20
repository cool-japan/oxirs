//! Request deduplication for GraphQL queries
//!
//! This module provides a mechanism to coalesce identical concurrent requests
//! into a single execution, sharing the result among all callers.

use crate::execution::ExecutionResult;
use anyhow::Result;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use tokio::sync::{watch, RwLock};

/// Request deduplicator for coalescing identical concurrent requests
pub struct RequestDeduplicator {
    /// Map of in-flight requests
    /// Key is the unique request identifier (e.g., hash of query + variables)
    /// Value is a watch channel receiver that will receive the result
    #[allow(clippy::type_complexity)]
    inflight:
        Arc<RwLock<HashMap<String, watch::Receiver<Option<Result<ExecutionResult, String>>>>>>,
}

impl RequestDeduplicator {
    /// Create a new request deduplicator
    pub fn new() -> Self {
        Self {
            inflight: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Deduplicate a request
    ///
    /// If a request with the same key is already in flight, this method will wait for its result.
    /// Otherwise, it will execute the provided future and broadcast the result to all waiters.
    pub async fn deduplicate<F, Fut>(&self, key: String, execute: F) -> Result<ExecutionResult>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<ExecutionResult>>,
    {
        // Check if request is already in flight
        {
            let inflight = self.inflight.read().await;
            if let Some(rx) = inflight.get(&key) {
                let mut rx = rx.clone();
                drop(inflight);

                // Wait for the result
                if rx.changed().await.is_ok() {
                    let result = rx.borrow();
                    if let Some(res) = result.as_ref() {
                        return match res {
                            Ok(ok_res) => Ok(ok_res.clone()),
                            Err(err_msg) => Err(anyhow::anyhow!("{}", err_msg)),
                        };
                    }
                }
                return Err(anyhow::anyhow!(
                    "Request cancelled or failed to receive result"
                ));
            }
        }

        // Not in flight, create a new channel
        let (tx, rx) = watch::channel(None);

        {
            let mut inflight = self.inflight.write().await;
            // Double check to avoid race condition
            if let Some(rx) = inflight.get(&key) {
                let mut rx = rx.clone();
                drop(inflight);
                if rx.changed().await.is_ok() {
                    let result = rx.borrow();
                    if let Some(res) = result.as_ref() {
                        return match res {
                            Ok(ok_res) => Ok(ok_res.clone()),
                            Err(err_msg) => Err(anyhow::anyhow!("{}", err_msg)),
                        };
                    }
                }
                return Err(anyhow::anyhow!(
                    "Request cancelled or failed to receive result"
                ));
            }
            inflight.insert(key.clone(), rx);
        }

        // Execute the request
        let result = execute().await;

        // Broadcast result and remove from inflight map
        {
            let mut inflight = self.inflight.write().await;
            inflight.remove(&key);
        }

        // Send result to waiters
        let send_result = match &result {
            Ok(res) => Ok(res.clone()),
            Err(e) => Err(e.to_string()),
        };
        let _ = tx.send(Some(send_result));

        result
    }
}

impl Default for RequestDeduplicator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[tokio::test]
    async fn test_deduplication() {
        let deduplicator = Arc::new(RequestDeduplicator::new());
        let counter = Arc::new(AtomicUsize::new(0));
        let key = "test_key".to_string();

        let mut handles = vec![];

        for _ in 0..10 {
            let deduplicator = deduplicator.clone();
            let counter = counter.clone();
            let key = key.clone();

            handles.push(tokio::spawn(async move {
                deduplicator
                    .deduplicate(key, || async move {
                        // Simulate some work
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        counter.fetch_add(1, Ordering::SeqCst);
                        Ok(ExecutionResult::new())
                    })
                    .await
            }));
        }

        for handle in handles {
            let _ = handle.await.unwrap();
        }

        // Counter should be 1 because all requests were coalesced
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_different_keys() {
        let deduplicator = Arc::new(RequestDeduplicator::new());
        let counter = Arc::new(AtomicUsize::new(0));

        let h1 = {
            let deduplicator = deduplicator.clone();
            let counter = counter.clone();
            tokio::spawn(async move {
                deduplicator
                    .deduplicate("key1".to_string(), || async move {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        counter.fetch_add(1, Ordering::SeqCst);
                        Ok(ExecutionResult::new())
                    })
                    .await
            })
        };

        let h2 = {
            let deduplicator = deduplicator.clone();
            let counter = counter.clone();
            tokio::spawn(async move {
                deduplicator
                    .deduplicate("key2".to_string(), || async move {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        counter.fetch_add(1, Ordering::SeqCst);
                        Ok(ExecutionResult::new())
                    })
                    .await
            })
        };

        let _ = tokio::join!(h1, h2);

        // Counter should be 2 because keys are different
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }
}
