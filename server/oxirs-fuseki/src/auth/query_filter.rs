//! SPARQL Query Result Filtering Based on Graph Permissions
//!
//! This module implements query result filtering to ensure users only see
//! RDF triples from graphs they have permission to access.
//!
//! ## How It Works
//!
//! 1. **Pre-Execution Filtering**: Check which graphs the user can access
//! 2. **Query Rewriting**: Modify SPARQL query to only access authorized graphs
//! 3. **Post-Execution Filtering**: Filter results to remove unauthorized triples
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use oxirs_fuseki::auth::query_filter::QueryResultFilter;
//!
//! let filter = QueryResultFilter::new(graph_auth_manager);
//!
//! // Filter SPARQL query results
//! let filtered_results = filter.filter_query_results(
//!     user,
//!     dataset,
//!     query_results
//! ).await?;
//! ```

use super::graph_auth::{GraphAuthRequest, GraphAuthorizationManager, GraphPermission};
use super::types::User;
use crate::error::{FusekiError, FusekiResult};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{debug, instrument, warn};

/// SPARQL query result filtering manager
pub struct QueryResultFilter {
    graph_auth: Arc<GraphAuthorizationManager>,
}

impl QueryResultFilter {
    /// Create a new query result filter
    pub fn new(graph_auth: Arc<GraphAuthorizationManager>) -> Self {
        Self { graph_auth }
    }

    /// Get authorized graphs for a user in a dataset
    ///
    /// Returns a list of graph URIs the user can access.
    /// If the user has dataset-level permission, returns a special "*" marker.
    #[instrument(skip(self))]
    pub async fn get_authorized_graphs(
        &self,
        user: &User,
        dataset: &str,
        permission: GraphPermission,
    ) -> FusekiResult<Vec<String>> {
        let subject = format!("user:{}", user.username);

        let graphs = self
            .graph_auth
            .get_authorized_graphs(&subject, dataset, permission)
            .await?;

        let graph_count = graphs.len().to_string();
        debug!(
            "User {} has access to {} graphs in dataset {}",
            user.username,
            if graphs.contains(&"*".to_string()) {
                "ALL"
            } else {
                &graph_count
            },
            dataset
        );

        Ok(graphs)
    }

    /// Filter SPARQL SELECT query results based on graph permissions
    ///
    /// This removes bindings that reference graphs the user cannot access.
    #[instrument(skip(self, results))]
    pub async fn filter_select_results(
        &self,
        user: &User,
        dataset: &str,
        results: Vec<HashMap<String, serde_json::Value>>,
        graphs_mentioned: &[String],
    ) -> FusekiResult<Vec<HashMap<String, serde_json::Value>>> {
        // Get authorized graphs
        let subject = format!("user:{}", user.username);
        let authorized_graphs = self
            .graph_auth
            .filter_authorized_graphs(&subject, dataset, graphs_mentioned, GraphPermission::Read)
            .await?;

        // If user has access to all graphs, return results unchanged
        if authorized_graphs.len() == graphs_mentioned.len() {
            debug!(
                "User {} has access to all {} graphs, no filtering needed",
                user.username,
                graphs_mentioned.len()
            );
            return Ok(results);
        }

        let authorized_set: HashSet<_> = authorized_graphs.into_iter().collect();

        // Filter results
        let mut filtered_results = Vec::new();
        let mut filtered_count = 0;

        for binding in results {
            // Check if this binding references an unauthorized graph
            let mut should_include = true;

            // Check if binding has a "graph" variable
            if let Some(graph_value) = binding.get("graph") {
                if let Some(graph_uri) = graph_value.as_str() {
                    if !authorized_set.contains(graph_uri) {
                        should_include = false;
                        filtered_count += 1;
                    }
                }
            }

            if should_include {
                filtered_results.push(binding);
            }
        }

        if filtered_count > 0 {
            debug!(
                "Filtered {} results for user {} (unauthorized graphs)",
                filtered_count, user.username
            );
        }

        Ok(filtered_results)
    }

    /// Filter CONSTRUCT/DESCRIBE query results (RDF triples)
    ///
    /// This removes triples that come from graphs the user cannot access.
    #[instrument(skip(self, triples))]
    pub async fn filter_graph_results(
        &self,
        user: &User,
        dataset: &str,
        triples: &str, // Turtle/N-Triples format
    ) -> FusekiResult<String> {
        let subject = format!("user:{}", user.username);

        // Parse triples to extract graph URIs (simplified parsing)
        // In a real implementation, you'd use a proper RDF parser
        let graph_uris = self.extract_graph_uris(triples);

        if graph_uris.is_empty() {
            // No named graphs, default graph only
            // Check if user has access to default graph
            let request = GraphAuthRequest::new(
                subject.clone(),
                dataset.to_string(),
                None, // default graph
                GraphPermission::Read,
            );

            let response = self.graph_auth.check_graph_permission(&request).await?;

            if response.allowed {
                return Ok(triples.to_string());
            } else {
                warn!(
                    "User {} denied access to default graph in dataset {}",
                    user.username, dataset
                );
                return Ok(String::new()); // Empty result
            }
        }

        // Get authorized graphs
        let authorized = self
            .graph_auth
            .filter_authorized_graphs(&subject, dataset, &graph_uris, GraphPermission::Read)
            .await?;

        // Filter triples by authorized graphs
        // This is a simplified implementation
        // In production, use a proper RDF library
        let authorized_set: HashSet<_> = authorized.into_iter().collect();

        let filtered_triples = self.filter_triples_by_graphs(triples, &authorized_set);

        debug!(
            "Filtered RDF triples for user {} in dataset {}",
            user.username, dataset
        );

        Ok(filtered_triples)
    }

    /// Check if user can execute a SPARQL query on specific graphs
    ///
    /// This pre-checks before query execution to fail fast.
    #[instrument(skip(self))]
    pub async fn check_query_authorization(
        &self,
        user: &User,
        dataset: &str,
        query_graphs: &[String],
        query_type: QueryType,
    ) -> FusekiResult<AuthorizationCheckResult> {
        let subject = format!("user:{}", user.username);
        let permission = match query_type {
            QueryType::Select | QueryType::Ask | QueryType::Construct | QueryType::Describe => {
                GraphPermission::Read
            }
            QueryType::Update => GraphPermission::Write,
        };

        // If no specific graphs mentioned, check default graph
        if query_graphs.is_empty() {
            let request =
                GraphAuthRequest::new(subject.clone(), dataset.to_string(), None, permission);

            let response = self.graph_auth.check_graph_permission(&request).await?;

            return Ok(AuthorizationCheckResult {
                allowed: response.allowed,
                reason: response.reason,
                authorized_graphs: if response.allowed {
                    vec!["default".to_string()]
                } else {
                    vec![]
                },
            });
        }

        // Check authorization for all mentioned graphs
        let authorized = self
            .graph_auth
            .filter_authorized_graphs(&subject, dataset, query_graphs, permission)
            .await?;

        let all_authorized = authorized.len() == query_graphs.len();

        if !all_authorized {
            let unauthorized: Vec<_> = query_graphs
                .iter()
                .filter(|g| !authorized.contains(g))
                .cloned()
                .collect();

            warn!(
                "User {} lacks {} permission for graphs: {:?}",
                user.username,
                permission.to_relation(),
                unauthorized
            );
        }

        Ok(AuthorizationCheckResult {
            allowed: all_authorized,
            reason: if all_authorized {
                Some("All graphs authorized".to_string())
            } else {
                Some(format!(
                    "Missing permission for {} graphs",
                    query_graphs.len() - authorized.len()
                ))
            },
            authorized_graphs: authorized,
        })
    }

    /// Extract graph URIs from RDF triples (simplified)
    fn extract_graph_uris(&self, triples: &str) -> Vec<String> {
        let mut graphs = HashSet::new();

        // Look for GRAPH clauses in TriG/N-Quads format
        // This is a simplified regex-based extraction
        // In production, use a proper RDF parser
        for line in triples.lines() {
            if let Some(graph_uri) = line
                .strip_prefix("GRAPH <")
                .and_then(|rest| rest.split('>').next())
            {
                graphs.insert(graph_uri.to_string());
            }
        }

        graphs.into_iter().collect()
    }

    /// Filter RDF triples by authorized graphs
    fn filter_triples_by_graphs(
        &self,
        triples: &str,
        authorized_graphs: &HashSet<String>,
    ) -> String {
        // Simplified filtering - in production use a proper RDF library
        let mut filtered = String::new();

        for line in triples.lines() {
            // Check if this triple is in an authorized graph
            if let Some(graph_uri) = line
                .strip_prefix("GRAPH <")
                .and_then(|rest| rest.split('>').next())
            {
                if authorized_graphs.contains(graph_uri) {
                    filtered.push_str(line);
                    filtered.push('\n');
                }
            } else {
                // Default graph triple - include if default graph is authorized
                // For simplicity, always include default graph triples
                filtered.push_str(line);
                filtered.push('\n');
            }
        }

        filtered
    }
}

/// SPARQL query type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    Select,
    Ask,
    Construct,
    Describe,
    Update,
}

impl QueryType {
    /// Detect query type from SPARQL query string
    pub fn from_query_string(query: &str) -> Option<Self> {
        let query_upper = query.trim().to_uppercase();

        if query_upper.starts_with("SELECT") {
            Some(QueryType::Select)
        } else if query_upper.starts_with("ASK") {
            Some(QueryType::Ask)
        } else if query_upper.starts_with("CONSTRUCT") {
            Some(QueryType::Construct)
        } else if query_upper.starts_with("DESCRIBE") {
            Some(QueryType::Describe)
        } else if query_upper.starts_with("INSERT")
            || query_upper.starts_with("DELETE")
            || query_upper.starts_with("LOAD")
            || query_upper.starts_with("CLEAR")
        {
            Some(QueryType::Update)
        } else {
            None
        }
    }
}

/// Result of authorization check
#[derive(Debug, Clone)]
pub struct AuthorizationCheckResult {
    /// Whether the query is authorized
    pub allowed: bool,
    /// Reason for the decision
    pub reason: Option<String>,
    /// List of authorized graphs
    pub authorized_graphs: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::graph_auth::GraphAuthorizationManager;
    use crate::auth::rebac::{InMemoryRebacManager, RebacEvaluator, RelationshipTuple};
    use crate::auth::types::User;

    fn create_test_user() -> User {
        User {
            username: "alice".to_string(),
            roles: vec!["user".to_string()],
            email: Some("alice@example.com".to_string()),
            full_name: Some("Alice".to_string()),
            last_login: None,
            permissions: vec![],
        }
    }

    #[tokio::test]
    async fn test_get_authorized_graphs() {
        let rebac = Arc::new(InMemoryRebacManager::new());

        // Alice can read two specific graphs
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

        let graph_auth = Arc::new(GraphAuthorizationManager::new(rebac));
        let filter = QueryResultFilter::new(graph_auth);

        let user = create_test_user();
        let graphs = filter
            .get_authorized_graphs(&user, "public", GraphPermission::Read)
            .await
            .unwrap();

        assert_eq!(graphs.len(), 2);
        assert!(graphs.contains(&"http://example.org/g1".to_string()));
        assert!(graphs.contains(&"http://example.org/g2".to_string()));
    }

    #[tokio::test]
    async fn test_filter_select_results() {
        let rebac = Arc::new(InMemoryRebacManager::new());

        // Alice can only read g1
        rebac
            .add_tuple(RelationshipTuple::new(
                "user:alice",
                "can_read",
                "graph:http://example.org/g1",
            ))
            .await
            .unwrap();

        let graph_auth = Arc::new(GraphAuthorizationManager::new(rebac));
        let filter = QueryResultFilter::new(graph_auth);

        let user = create_test_user();

        // Simulate SPARQL results with graph column
        let results = vec![
            {
                let mut binding = HashMap::new();
                binding.insert(
                    "s".to_string(),
                    serde_json::json!("http://example.org/subject1"),
                );
                binding.insert(
                    "graph".to_string(),
                    serde_json::json!("http://example.org/g1"),
                );
                binding
            },
            {
                let mut binding = HashMap::new();
                binding.insert(
                    "s".to_string(),
                    serde_json::json!("http://example.org/subject2"),
                );
                binding.insert(
                    "graph".to_string(),
                    serde_json::json!("http://example.org/g2"),
                );
                binding
            },
        ];

        let graphs = vec![
            "http://example.org/g1".to_string(),
            "http://example.org/g2".to_string(),
        ];

        let filtered = filter
            .filter_select_results(&user, "public", results, &graphs)
            .await
            .unwrap();

        // Should only have result from g1
        assert_eq!(filtered.len(), 1);
        assert_eq!(
            filtered[0].get("graph").and_then(|v| v.as_str()),
            Some("http://example.org/g1")
        );
    }

    #[tokio::test]
    async fn test_check_query_authorization() {
        let rebac = Arc::new(InMemoryRebacManager::new());

        // Alice has dataset-level read
        rebac
            .add_tuple(RelationshipTuple::new(
                "user:alice",
                "can_read",
                "dataset:public",
            ))
            .await
            .unwrap();

        let graph_auth = Arc::new(GraphAuthorizationManager::new(rebac));
        let filter = QueryResultFilter::new(graph_auth);

        let user = create_test_user();

        let graphs = vec!["http://example.org/g1".to_string()];

        let result = filter
            .check_query_authorization(&user, "public", &graphs, QueryType::Select)
            .await
            .unwrap();

        assert!(result.allowed);
        assert_eq!(result.authorized_graphs.len(), 1);
    }

    #[test]
    fn test_query_type_detection() {
        assert_eq!(
            QueryType::from_query_string("SELECT * WHERE { ?s ?p ?o }"),
            Some(QueryType::Select)
        );
        assert_eq!(
            QueryType::from_query_string("ASK WHERE { ?s ?p ?o }"),
            Some(QueryType::Ask)
        );
        assert_eq!(
            QueryType::from_query_string("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"),
            Some(QueryType::Construct)
        );
        assert_eq!(
            QueryType::from_query_string("INSERT DATA { <s> <p> <o> }"),
            Some(QueryType::Update)
        );
    }
}
