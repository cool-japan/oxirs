//! Graph-Level Authorization for RDF Named Graphs
//!
//! This module implements fine-grained authorization at the RDF named graph level,
//! extending the ReBAC model to support hierarchical permissions:
//!
//! - Dataset-level permissions (inherited by all graphs in the dataset)
//! - Graph-level permissions (specific to individual named graphs)
//! - Query result filtering (only return authorized graphs)
//!
//! ## Hierarchical Permission Model
//!
//! ```text
//! Instance
//!   └─ Dataset (user:alice is owner)
//!       ├─ Graph A (inherits: alice can read/write)
//!       ├─ Graph B (inherits: alice can read/write)
//!       └─ Graph C (explicit: user:bob can read only)
//! ```
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use oxirs_fuseki::auth::graph_auth::{GraphAuthorizationManager, GraphPermission};
//!
//! let manager = GraphAuthorizationManager::new(rebac_evaluator);
//!
//! // Check if user can read a specific graph
//! let can_read = manager.check_graph_permission(
//!     "user:alice",
//!     "dataset:public",
//!     Some("http://example.org/graph1"),
//!     GraphPermission::Read
//! ).await?;
//!
//! // Filter graphs based on user permissions
//! let allowed_graphs = manager.filter_authorized_graphs(
//!     "user:alice",
//!     "dataset:public",
//!     &all_graphs,
//!     GraphPermission::Read
//! ).await?;
//! ```

use super::rebac::{CheckRequest, CheckResponse, RebacEvaluator};
use crate::error::{FusekiError, FusekiResult};
use async_trait::async_trait;
use std::collections::HashSet;
use std::sync::Arc;
use tracing::{debug, instrument};

/// Graph-level permissions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphPermission {
    /// Read access to graph triples
    Read,
    /// Write access to graph (INSERT/DELETE)
    Write,
    /// Ability to drop/clear the graph
    Delete,
    /// Full control (creator/owner)
    Manage,
}

impl GraphPermission {
    /// Convert to ReBAC relation string
    pub fn to_relation(&self) -> &'static str {
        match self {
            GraphPermission::Read => "can_read",
            GraphPermission::Write => "can_write",
            GraphPermission::Delete => "can_delete",
            GraphPermission::Manage => "owner",
        }
    }

    /// Check if this permission implies another (e.g., Manage implies all others)
    pub fn implies(&self, other: &GraphPermission) -> bool {
        match self {
            GraphPermission::Manage => true, // Manage implies all permissions
            GraphPermission::Write => matches!(other, GraphPermission::Read), // Write implies Read
            GraphPermission::Delete => false,
            GraphPermission::Read => matches!(other, GraphPermission::Read),
        }
    }
}

/// Graph authorization check request
#[derive(Debug, Clone)]
pub struct GraphAuthRequest {
    /// User/subject requesting access
    pub subject: String,
    /// Dataset containing the graph
    pub dataset: String,
    /// Optional named graph URI (None = default graph)
    pub graph_uri: Option<String>,
    /// Permission being requested
    pub permission: GraphPermission,
}

impl GraphAuthRequest {
    pub fn new(
        subject: impl Into<String>,
        dataset: impl Into<String>,
        graph_uri: Option<String>,
        permission: GraphPermission,
    ) -> Self {
        Self {
            subject: subject.into(),
            dataset: dataset.into(),
            graph_uri,
            permission,
        }
    }
}

/// Graph authorization response
#[derive(Debug, Clone)]
pub struct GraphAuthResponse {
    /// Whether access is allowed
    pub allowed: bool,
    /// Reason for the decision
    pub reason: Option<String>,
    /// Which level granted access (dataset or graph)
    pub granted_by: Option<GrantLevel>,
}

/// Authorization grant level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrantLevel {
    /// Permission granted at dataset level (inherited)
    Dataset,
    /// Permission granted at graph level (explicit)
    Graph,
}

/// Manager for graph-level authorization with hierarchical permissions
pub struct GraphAuthorizationManager {
    rebac: Arc<dyn RebacEvaluator>,
    cache_enabled: bool,
}

impl GraphAuthorizationManager {
    /// Create a new graph authorization manager
    pub fn new(rebac: Arc<dyn RebacEvaluator>) -> Self {
        Self {
            rebac,
            cache_enabled: true,
        }
    }

    /// Enable or disable caching
    pub fn with_cache(mut self, enabled: bool) -> Self {
        self.cache_enabled = enabled;
        self
    }

    /// Check if a user has permission to access a specific graph
    ///
    /// This method implements hierarchical permission checking:
    /// 1. Check explicit graph-level permission (exact + implied)
    /// 2. Check dataset-level permission (exact + implied)
    /// 3. Apply permission implication rules (e.g., owner implies read/write)
    #[instrument(skip(self))]
    pub async fn check_graph_permission(
        &self,
        request: &GraphAuthRequest,
    ) -> FusekiResult<GraphAuthResponse> {
        let relation = request.permission.to_relation();
        let dataset_resource = format!("dataset:{}", request.dataset);

        // Get all permissions that could grant the requested permission
        let mut permissions_to_check = vec![request.permission];

        // Add higher permissions that imply the requested one
        for higher_perm in [
            GraphPermission::Manage,
            GraphPermission::Write,
            GraphPermission::Delete,
        ] {
            if higher_perm.implies(&request.permission) && higher_perm != request.permission {
                permissions_to_check.push(higher_perm);
            }
        }

        // Strategy 1: Check explicit graph-level permission
        if let Some(graph_uri) = &request.graph_uri {
            let graph_resource = format!("graph:{}", graph_uri);

            for perm in &permissions_to_check {
                let perm_relation = perm.to_relation();
                let graph_check =
                    CheckRequest::new(&request.subject, perm_relation, &graph_resource);

                if let Ok(response) = self.rebac.check(&graph_check).await {
                    if response.allowed {
                        debug!(
                            "Graph permission granted: {} {} {} (explicit, via {})",
                            request.subject, relation, graph_uri, perm_relation
                        );
                        return Ok(GraphAuthResponse {
                            allowed: true,
                            reason: Some(format!("Explicit graph permission: {}", perm_relation)),
                            granted_by: Some(GrantLevel::Graph),
                        });
                    }
                }
            }
        }

        // Strategy 2: Check dataset-level permission (inheritance)
        for perm in &permissions_to_check {
            let perm_relation = perm.to_relation();
            let dataset_check =
                CheckRequest::new(&request.subject, perm_relation, &dataset_resource);

            if let Ok(response) = self.rebac.check(&dataset_check).await {
                if response.allowed {
                    debug!(
                        "Graph permission granted: {} {} (inherited from dataset, via {})",
                        request.subject, relation, perm_relation
                    );
                    return Ok(GraphAuthResponse {
                        allowed: true,
                        reason: Some(format!(
                            "Inherited from dataset permission: {}",
                            perm_relation
                        )),
                        granted_by: Some(GrantLevel::Dataset),
                    });
                }
            }
        }

        // No permission found
        debug!(
            "Graph permission denied: {} {} {} (no matching permission)",
            request.subject,
            relation,
            request.graph_uri.as_deref().unwrap_or("default")
        );
        Ok(GraphAuthResponse {
            allowed: false,
            reason: Some("No matching permission found".to_string()),
            granted_by: None,
        })
    }

    /// Filter a list of graph URIs to only those the user can access
    ///
    /// This is used for query result filtering to ensure users only see
    /// graphs they have permission to read.
    #[instrument(skip(self, graph_uris))]
    pub async fn filter_authorized_graphs(
        &self,
        subject: &str,
        dataset: &str,
        graph_uris: &[String],
        permission: GraphPermission,
    ) -> FusekiResult<Vec<String>> {
        let mut authorized = Vec::new();

        // Optimization: Check if user has dataset-level permission
        // If so, all graphs are authorized (no need to check individually)
        let dataset_request =
            GraphAuthRequest::new(subject.to_string(), dataset.to_string(), None, permission);

        if let Ok(response) = self.check_graph_permission(&dataset_request).await {
            if response.allowed && response.granted_by == Some(GrantLevel::Dataset) {
                // User has dataset-level permission, return all graphs
                debug!(
                    "User {} has dataset-level {} permission, returning all {} graphs",
                    subject,
                    permission.to_relation(),
                    graph_uris.len()
                );
                return Ok(graph_uris.to_vec());
            }
        }

        // Check each graph individually
        for graph_uri in graph_uris {
            let request = GraphAuthRequest::new(
                subject.to_string(),
                dataset.to_string(),
                Some(graph_uri.clone()),
                permission,
            );

            if let Ok(response) = self.check_graph_permission(&request).await {
                if response.allowed {
                    authorized.push(graph_uri.clone());
                }
            }
        }

        debug!(
            "Filtered {} graphs to {} authorized for user {}",
            graph_uris.len(),
            authorized.len(),
            subject
        );

        Ok(authorized)
    }

    /// Batch check permissions for multiple graphs
    ///
    /// More efficient than calling check_graph_permission multiple times
    /// when you need to check many graphs at once.
    #[instrument(skip(self, requests))]
    pub async fn batch_check_graphs(
        &self,
        requests: &[GraphAuthRequest],
    ) -> FusekiResult<Vec<GraphAuthResponse>> {
        let mut responses = Vec::with_capacity(requests.len());

        // TODO: Optimize with actual batch checking at RebacEvaluator level
        for request in requests {
            let response = self.check_graph_permission(request).await?;
            responses.push(response);
        }

        Ok(responses)
    }

    /// Get all graphs a user has access to in a dataset
    ///
    /// This queries the ReBAC system to find all graph-level permissions
    /// for the given user and dataset.
    #[instrument(skip(self))]
    pub async fn get_authorized_graphs(
        &self,
        subject: &str,
        dataset: &str,
        permission: GraphPermission,
    ) -> FusekiResult<Vec<String>> {
        let relation = permission.to_relation();

        // Get all tuples for this subject
        let tuples = self
            .rebac
            .list_subject_tuples(subject)
            .await
            .map_err(|e| FusekiError::internal(format!("Failed to list tuples: {}", e)))?;

        let mut graphs = HashSet::new();

        // Check for dataset-level permission (grants access to all graphs)
        let dataset_resource = format!("dataset:{}", dataset);
        if tuples.iter().any(|t| {
            t.object == dataset_resource
                && (t.relation == relation || t.relation == GraphPermission::Manage.to_relation())
        }) {
            // User has dataset-level permission
            // Return special marker indicating "all graphs"
            return Ok(vec!["*".to_string()]);
        }

        // Collect explicit graph-level permissions
        for tuple in tuples {
            if tuple.relation == relation && tuple.object.starts_with("graph:") {
                // Extract graph URI from "graph:http://example.org/g1"
                if let Some(graph_uri) = tuple.object.strip_prefix("graph:") {
                    graphs.insert(graph_uri.to_string());
                }
            }
        }

        Ok(graphs.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::rebac::{InMemoryRebacManager, RelationshipTuple};

    #[tokio::test]
    async fn test_explicit_graph_permission() {
        let rebac = Arc::new(InMemoryRebacManager::new());

        // Add explicit graph permission
        rebac
            .add_tuple(RelationshipTuple::new(
                "user:alice",
                "can_read",
                "graph:http://example.org/g1",
            ))
            .await
            .unwrap();

        let manager = GraphAuthorizationManager::new(rebac);

        let request = GraphAuthRequest::new(
            "user:alice",
            "dataset:public",
            Some("http://example.org/g1".to_string()),
            GraphPermission::Read,
        );

        let response = manager.check_graph_permission(&request).await.unwrap();
        assert!(response.allowed);
        assert_eq!(response.granted_by, Some(GrantLevel::Graph));
    }

    #[tokio::test]
    async fn test_dataset_level_inheritance() {
        let rebac = Arc::new(InMemoryRebacManager::new());

        // Add dataset-level permission (should inherit to all graphs)
        rebac
            .add_tuple(RelationshipTuple::new(
                "user:alice",
                "can_read",
                "dataset:public",
            ))
            .await
            .unwrap();

        let manager = GraphAuthorizationManager::new(rebac);

        // Check access to a graph in this dataset
        let request = GraphAuthRequest::new(
            "user:alice",
            "public", // Dataset name without prefix
            Some("http://example.org/any-graph".to_string()),
            GraphPermission::Read,
        );

        let response = manager.check_graph_permission(&request).await.unwrap();
        assert!(response.allowed);
        assert_eq!(response.granted_by, Some(GrantLevel::Dataset));
    }

    #[tokio::test]
    async fn test_permission_implication() {
        let rebac = Arc::new(InMemoryRebacManager::new());

        // Add owner permission (should imply read/write)
        rebac
            .add_tuple(RelationshipTuple::new(
                "user:alice",
                "owner",
                "dataset:public",
            ))
            .await
            .unwrap();

        let manager = GraphAuthorizationManager::new(rebac);

        // Check read permission (should be implied by owner)
        let request = GraphAuthRequest::new(
            "user:alice",
            "public", // Dataset name without prefix
            Some("http://example.org/g1".to_string()),
            GraphPermission::Read,
        );

        let response = manager.check_graph_permission(&request).await.unwrap();
        assert!(response.allowed);
    }

    #[tokio::test]
    async fn test_filter_authorized_graphs() {
        let rebac = Arc::new(InMemoryRebacManager::new());

        // Alice can read graph1 and graph2
        rebac
            .add_tuple(RelationshipTuple::new(
                "user:alice",
                "can_read",
                "graph:http://example.org/g1",
            ))
            .await
            .unwrap();
        rebac
            .add_tuple(RelationshipTuple::new(
                "user:alice",
                "can_read",
                "graph:http://example.org/g2",
            ))
            .await
            .unwrap();

        let manager = GraphAuthorizationManager::new(rebac);

        let all_graphs = vec![
            "http://example.org/g1".to_string(),
            "http://example.org/g2".to_string(),
            "http://example.org/g3".to_string(),
        ];

        let authorized = manager
            .filter_authorized_graphs("user:alice", "public", &all_graphs, GraphPermission::Read)
            .await
            .unwrap();

        assert_eq!(authorized.len(), 2);
        assert!(authorized.contains(&"http://example.org/g1".to_string()));
        assert!(authorized.contains(&"http://example.org/g2".to_string()));
    }

    #[tokio::test]
    async fn test_filter_with_dataset_permission() {
        let rebac = Arc::new(InMemoryRebacManager::new());

        // Alice has dataset-level read (should see all graphs)
        rebac
            .add_tuple(RelationshipTuple::new(
                "user:alice",
                "can_read",
                "dataset:public",
            ))
            .await
            .unwrap();

        let manager = GraphAuthorizationManager::new(rebac);

        let all_graphs = vec![
            "http://example.org/g1".to_string(),
            "http://example.org/g2".to_string(),
            "http://example.org/g3".to_string(),
        ];

        let authorized = manager
            .filter_authorized_graphs("user:alice", "public", &all_graphs, GraphPermission::Read)
            .await
            .unwrap();

        assert_eq!(authorized.len(), 3); // All graphs authorized
    }

    #[tokio::test]
    async fn test_no_permission() {
        let rebac = Arc::new(InMemoryRebacManager::new());
        let manager = GraphAuthorizationManager::new(rebac);

        let request = GraphAuthRequest::new(
            "user:bob",
            "public", // Dataset name without prefix
            Some("http://example.org/g1".to_string()),
            GraphPermission::Read,
        );

        let response = manager.check_graph_permission(&request).await.unwrap();
        assert!(!response.allowed);
    }
}
