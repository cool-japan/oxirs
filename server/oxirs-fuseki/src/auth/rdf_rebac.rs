//! RDF-Native ReBAC Implementation
//!
//! This module implements ReBAC using RDF triples stored in a named graph,
//! enabling SPARQL-based policy queries and inference.
//!
//! ## RDF Schema
//!
//! Relationships are stored as RDF triples in the authorization graph:
//!
//! ```turtle
//! @prefix auth: <http://oxirs.org/auth#> .
//! @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
//!
//! # Named graph for authorization data
//! GRAPH <urn:oxirs:auth:relationships> {
//!
//!   # Subject-Relation-Object triples
//!   <user:alice> auth:owner <dataset:public> .
//!   <user:bob> auth:canRead <graph:http://example.org/g1> .
//!
//!   # Organization membership (for transitive permissions)
//!   <user:alice> auth:memberOf <organization:engineering> .
//!   <organization:engineering> auth:canAccess <dataset:internal> .
//!
//!   # Conditional relationships using reification
//!   _:r1 a auth:Relationship ;
//!        auth:subject <user:charlie> ;
//!        auth:relation auth:canRead ;
//!        auth:object <dataset:temporary> ;
//!        auth:notBefore "2025-01-01T00:00:00Z"^^xsd:dateTime ;
//!        auth:notAfter "2025-12-31T23:59:59Z"^^xsd:dateTime .
//! }
//! ```
//!
//! ## SPARQL Inference Rules
//!
//! Permission implication rules can be expressed as SPARQL CONSTRUCT queries:
//!
//! ```sparql
//! # Rule: owner implies canRead, canWrite, canDelete
//! CONSTRUCT {
//!   ?subject auth:canRead ?object .
//!   ?subject auth:canWrite ?object .
//!   ?subject auth:canDelete ?object .
//! }
//! WHERE {
//!   ?subject auth:owner ?object .
//! }
//! ```

use super::rebac::{
    CheckRequest, CheckResponse, RebacError, RebacEvaluator, RelationshipCondition,
    RelationshipTuple, Result,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, instrument, warn};

/// Authorization vocabulary namespace
const AUTH_NS: &str = "http://oxirs.org/auth#";

/// Named graph URI for storing authorization relationships
const AUTH_GRAPH: &str = "urn:oxirs:auth:relationships";

/// RDF-based ReBAC manager using SPARQL for policy queries
pub struct RdfRebacManager {
    /// Reference to RDF store (would be actual store in production)
    store: Arc<RwLock<MockRdfStore>>,
    /// Enable inference rules
    inference_enabled: bool,
}

impl RdfRebacManager {
    /// Create a new RDF-based ReBAC manager with mock store (for testing)
    pub fn new(store: Arc<RwLock<MockRdfStore>>) -> Self {
        Self {
            store,
            inference_enabled: true,
        }
    }

    /// Create a new RDF-based ReBAC manager with production OxiRS store
    pub fn with_store(store: crate::store::Store) -> RdfRebacManagerProduction {
        RdfRebacManagerProduction {
            store: Arc::new(OxiRdfStore::new(store)),
            inference_enabled: true,
        }
    }

    /// Enable or disable inference
    pub fn with_inference(mut self, enabled: bool) -> Self {
        self.inference_enabled = enabled;
        self
    }

    /// Generate SPARQL ASK query to check relationship
    fn generate_ask_query(request: &CheckRequest) -> String {
        let subject = Self::uri_escape(&request.subject);
        let relation = Self::relation_to_property(&request.relation);
        let object = Self::uri_escape(&request.object);

        format!(
            r#"
PREFIX auth: <{}>

ASK {{
  GRAPH <{}> {{
    <{}> {} <{}>
  }}
}}
"#,
            AUTH_NS, AUTH_GRAPH, subject, relation, object
        )
    }

    /// Generate SPARQL query to check with inference
    fn generate_ask_query_with_inference(request: &CheckRequest) -> String {
        let subject = Self::uri_escape(&request.subject);
        let relation = Self::relation_to_property(&request.relation);
        let object = Self::uri_escape(&request.object);

        // Include inference rules for permission implication
        format!(
            r#"
PREFIX auth: <{}>

ASK {{
  GRAPH <{}> {{
    {{
      # Direct relationship
      <{}> {} <{}>
    }}
    UNION
    {{
      # Implied by owner
      <{}> auth:owner <{}> .
      FILTER ({} IN (auth:canRead, auth:canWrite, auth:canDelete))
    }}
    UNION
    {{
      # Implied by canWrite (implies canRead)
      <{}> auth:canWrite <{}> .
      FILTER ({} = auth:canRead)
    }}
  }}
}}
"#,
            AUTH_NS,
            AUTH_GRAPH,
            subject,
            relation,
            object,
            subject,
            object,
            relation,
            subject,
            object,
            relation
        )
    }

    /// Generate SPARQL INSERT query to add relationship
    fn generate_insert_query(tuple: &RelationshipTuple) -> String {
        let subject = Self::uri_escape(&tuple.subject);
        let relation = Self::relation_to_property(&tuple.relation);
        let object = Self::uri_escape(&tuple.object);

        if tuple.condition.is_none() {
            // Simple triple insertion
            format!(
                r#"
PREFIX auth: <{}>

INSERT DATA {{
  GRAPH <{}> {{
    <{}> {} <{}>
  }}
}}
"#,
                AUTH_NS, AUTH_GRAPH, subject, relation, object
            )
        } else {
            // Reified relationship with condition
            Self::generate_reified_insert(tuple)
        }
    }

    /// Generate reified relationship insertion (for conditional relationships)
    fn generate_reified_insert(tuple: &RelationshipTuple) -> String {
        let subject = Self::uri_escape(&tuple.subject);
        let relation = Self::relation_to_property(&tuple.relation);
        let object = Self::uri_escape(&tuple.object);

        let mut triples = vec![
            format!("    [] a auth:Relationship ;"),
            format!("       auth:subject <{}> ;", subject),
            format!("       auth:relation {} ;", relation),
            format!("       auth:object <{}> ", object),
        ];

        if let Some(condition) = &tuple.condition {
            match condition {
                RelationshipCondition::TimeWindow {
                    not_before,
                    not_after,
                } => {
                    if let Some(nb) = not_before {
                        triples.push(format!(
                            "       ; auth:notBefore \"{}\"^^xsd:dateTime",
                            nb.to_rfc3339()
                        ));
                    }
                    if let Some(na) = not_after {
                        triples.push(format!(
                            "       ; auth:notAfter \"{}\"^^xsd:dateTime",
                            na.to_rfc3339()
                        ));
                    }
                }
                RelationshipCondition::IpAddress { allowed_ips } => {
                    for ip in allowed_ips {
                        triples.push(format!("       ; auth:allowedIp \"{}\"", ip));
                    }
                }
                RelationshipCondition::Attribute { key, value } => {
                    triples.push(format!(
                        "       ; auth:attribute [ auth:key \"{}\" ; auth:value \"{}\" ]",
                        key, value
                    ));
                }
            }
        }

        triples.push("    .".to_string());

        format!(
            r#"
PREFIX auth: <{}>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

INSERT DATA {{
  GRAPH <{}> {{
{}
  }}
}}
"#,
            AUTH_NS,
            AUTH_GRAPH,
            triples.join("\n")
        )
    }

    /// Generate SPARQL DELETE query to remove relationship
    fn generate_delete_query(tuple: &RelationshipTuple) -> String {
        let subject = Self::uri_escape(&tuple.subject);
        let relation = Self::relation_to_property(&tuple.relation);
        let object = Self::uri_escape(&tuple.object);

        format!(
            r#"
PREFIX auth: <{}>

DELETE WHERE {{
  GRAPH <{}> {{
    <{}> {} <{}>
  }}
}}
"#,
            AUTH_NS, AUTH_GRAPH, subject, relation, object
        )
    }

    /// Generate SPARQL query to list subject's relationships
    fn generate_list_subject_query(subject: &str) -> String {
        let subject_uri = Self::uri_escape(subject);

        format!(
            r#"
PREFIX auth: <{}>

SELECT ?relation ?object
WHERE {{
  GRAPH <{}> {{
    <{}> ?relation ?object .
    FILTER (STRSTARTS(STR(?relation), STR(auth:)))
  }}
}}
"#,
            AUTH_NS, AUTH_GRAPH, subject_uri
        )
    }

    /// Generate SPARQL query to list object's relationships
    fn generate_list_object_query(object: &str) -> String {
        let object_uri = Self::uri_escape(object);

        format!(
            r#"
PREFIX auth: <{}>

SELECT ?subject ?relation
WHERE {{
  GRAPH <{}> {{
    ?subject ?relation <{}> .
    FILTER (STRSTARTS(STR(?relation), STR(auth:)))
  }}
}}
"#,
            AUTH_NS, AUTH_GRAPH, object_uri
        )
    }

    /// Convert relation string to RDF property URI
    fn relation_to_property(relation: &str) -> String {
        match relation {
            "can_read" => "auth:canRead".to_string(),
            "can_write" => "auth:canWrite".to_string(),
            "can_delete" => "auth:canDelete".to_string(),
            "owner" => "auth:owner".to_string(),
            "member" => "auth:memberOf".to_string(),
            "can_access" => "auth:canAccess".to_string(),
            "can_manage" => "auth:canManage".to_string(),
            other => format!("auth:{}", other.replace('_', "")),
        }
    }

    /// Escape URI for SPARQL
    fn uri_escape(s: &str) -> String {
        // Simple URI escaping (in production, use proper URI encoding)
        s.to_string()
    }
}

#[async_trait]
impl RebacEvaluator for RdfRebacManager {
    #[instrument(skip(self))]
    async fn check(&self, request: &CheckRequest) -> Result<CheckResponse> {
        debug!(
            "RDF ReBAC check: {} --{}-> {}",
            request.subject, request.relation, request.object
        );

        let query = if self.inference_enabled {
            Self::generate_ask_query_with_inference(request)
        } else {
            Self::generate_ask_query(request)
        };

        debug!("SPARQL ASK query:\n{}", query);

        // Execute SPARQL ASK query
        let store = self.store.read().await;
        let allowed = store.execute_ask(&query).await?;

        Ok(CheckResponse {
            allowed,
            reason: if allowed {
                Some("Relationship exists in RDF store".to_string())
            } else {
                Some("No matching relationship found".to_string())
            },
        })
    }

    #[instrument(skip(self))]
    async fn add_tuple(&self, tuple: RelationshipTuple) -> Result<()> {
        debug!(
            "Adding RDF tuple: {} --{}-> {}",
            tuple.subject, tuple.relation, tuple.object
        );

        let query = Self::generate_insert_query(&tuple);
        debug!("SPARQL INSERT query:\n{}", query);

        let mut store = self.store.write().await;
        store.execute_update(&query).await?;

        Ok(())
    }

    #[instrument(skip(self))]
    async fn remove_tuple(&self, tuple: &RelationshipTuple) -> Result<()> {
        debug!(
            "Removing RDF tuple: {} --{}-> {}",
            tuple.subject, tuple.relation, tuple.object
        );

        let query = Self::generate_delete_query(tuple);
        debug!("SPARQL DELETE query:\n{}", query);

        let mut store = self.store.write().await;
        store.execute_update(&query).await?;

        Ok(())
    }

    async fn list_subject_tuples(&self, subject: &str) -> Result<Vec<RelationshipTuple>> {
        let query = Self::generate_list_subject_query(subject);

        let store = self.store.read().await;
        let results = store.execute_select(&query).await?;

        let tuples = results
            .into_iter()
            .map(|(relation, object)| RelationshipTuple::new(subject.to_string(), relation, object))
            .collect();

        Ok(tuples)
    }

    async fn list_object_tuples(&self, object: &str) -> Result<Vec<RelationshipTuple>> {
        let query = Self::generate_list_object_query(object);

        let store = self.store.read().await;
        let results = store.execute_select(&query).await?;

        let tuples = results
            .into_iter()
            .map(|(subject, relation)| {
                RelationshipTuple::new(subject, relation, object.to_string())
            })
            .collect();

        Ok(tuples)
    }

    async fn batch_check(&self, requests: &[CheckRequest]) -> Result<Vec<CheckResponse>> {
        // TODO: Optimize with SPARQL UNION query
        let mut responses = Vec::with_capacity(requests.len());

        for request in requests {
            let response = self.check(request).await?;
            responses.push(response);
        }

        Ok(responses)
    }
}

/// Production RDF store using actual OxiRS Store
pub struct OxiRdfStore {
    /// Reference to OxiRS Store
    store: crate::store::Store,
}

impl OxiRdfStore {
    /// Create a new RDF store adapter
    pub fn new(store: crate::store::Store) -> Self {
        Self { store }
    }

    /// Execute SPARQL ASK query
    async fn execute_ask(&self, query: &str) -> Result<bool> {
        use crate::error::FusekiError;
        use oxirs_core::query::QueryResult as CoreQueryResult;

        let result = self
            .store
            .query(query)
            .map_err(|e| RebacError::Internal(format!("SPARQL ASK query failed: {}", e)))?;

        match result.inner {
            CoreQueryResult::Ask(boolean) => Ok(boolean),
            _ => Err(RebacError::Internal(
                "Expected ASK query result".to_string(),
            )),
        }
    }

    /// Execute SPARQL UPDATE query
    async fn execute_update(&self, query: &str) -> Result<()> {
        self.store
            .update(query)
            .map_err(|e| RebacError::Internal(format!("SPARQL UPDATE failed: {}", e)))?;
        Ok(())
    }

    /// Execute SPARQL SELECT query
    async fn execute_select(&self, query: &str) -> Result<Vec<(String, String)>> {
        use oxirs_core::query::QueryResult as CoreQueryResult;

        let result = self
            .store
            .query(query)
            .map_err(|e| RebacError::Internal(format!("SPARQL SELECT query failed: {}", e)))?;

        match result.inner {
            CoreQueryResult::Select { bindings, .. } => {
                let mut results = Vec::new();
                for binding in bindings {
                    // Extract first two variables from bindings
                    let mut values: Vec<String> =
                        binding.values().map(|term| term.to_string()).collect();
                    if values.len() >= 2 {
                        results.push((values.remove(0), values.remove(0)));
                    }
                }
                Ok(results)
            }
            _ => Err(RebacError::Internal(
                "Expected SELECT query result".to_string(),
            )),
        }
    }
}

/// Production RDF-based ReBAC manager using actual OxiRS Store
pub struct RdfRebacManagerProduction {
    /// Reference to production RDF store
    store: Arc<OxiRdfStore>,
    /// Enable inference rules
    inference_enabled: bool,
}

impl RdfRebacManagerProduction {
    /// Enable or disable inference
    pub fn with_inference(mut self, enabled: bool) -> Self {
        self.inference_enabled = enabled;
        self
    }
}

#[async_trait]
impl RebacEvaluator for RdfRebacManagerProduction {
    async fn check(&self, request: &CheckRequest) -> Result<CheckResponse> {
        let query = if self.inference_enabled {
            RdfRebacManager::generate_ask_query_with_inference(request)
        } else {
            RdfRebacManager::generate_ask_query(request)
        };

        let allowed = self.store.execute_ask(&query).await?;

        Ok(CheckResponse {
            allowed,
            reason: if allowed {
                Some("Relationship exists in RDF store".to_string())
            } else {
                Some("No matching relationship found".to_string())
            },
        })
    }

    async fn add_tuple(&self, tuple: RelationshipTuple) -> Result<()> {
        let query = RdfRebacManager::generate_insert_query(&tuple);
        self.store.execute_update(&query).await?;
        Ok(())
    }

    async fn remove_tuple(&self, tuple: &RelationshipTuple) -> Result<()> {
        let query = RdfRebacManager::generate_delete_query(tuple);
        self.store.execute_update(&query).await?;
        Ok(())
    }

    async fn list_subject_tuples(&self, subject: &str) -> Result<Vec<RelationshipTuple>> {
        let query = RdfRebacManager::generate_list_subject_query(subject);
        let results = self.store.execute_select(&query).await?;

        let tuples = results
            .into_iter()
            .map(|(relation, object)| RelationshipTuple::new(subject.to_string(), relation, object))
            .collect();

        Ok(tuples)
    }

    async fn list_object_tuples(&self, object: &str) -> Result<Vec<RelationshipTuple>> {
        let query = RdfRebacManager::generate_list_object_query(object);
        let results = self.store.execute_select(&query).await?;

        let tuples = results
            .into_iter()
            .map(|(subject, relation)| {
                RelationshipTuple::new(subject, relation, object.to_string())
            })
            .collect();

        Ok(tuples)
    }

    async fn batch_check(&self, requests: &[CheckRequest]) -> Result<Vec<CheckResponse>> {
        let mut responses = Vec::with_capacity(requests.len());

        for request in requests {
            let response = self.check(request).await?;
            responses.push(response);
        }

        Ok(responses)
    }
}

/// Mock RDF store for testing
/// In production, use OxiRdfStore
pub struct MockRdfStore {
    /// In-memory storage for testing
    triples: Vec<(String, String, String)>,
}

impl MockRdfStore {
    pub fn new() -> Self {
        Self {
            triples: Vec::new(),
        }
    }

    async fn execute_ask(&self, _query: &str) -> Result<bool> {
        // Mock implementation - in production, use OxiRdfStore
        Ok(false)
    }

    async fn execute_update(&mut self, _query: &str) -> Result<()> {
        // Mock implementation - in production, use OxiRdfStore
        Ok(())
    }

    async fn execute_select(&self, _query: &str) -> Result<Vec<(String, String)>> {
        // Mock implementation - in production, use OxiRdfStore
        Ok(vec![])
    }
}

impl Default for MockRdfStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_ask_query() {
        let request = CheckRequest::new("user:alice", "can_read", "dataset:public");
        let query = RdfRebacManager::generate_ask_query(&request);

        assert!(query.contains("ASK"));
        assert!(query.contains("<user:alice>"));
        assert!(query.contains("auth:canRead"));
        assert!(query.contains("<dataset:public>"));
    }

    #[test]
    fn test_generate_insert_query() {
        let tuple = RelationshipTuple::new("user:alice", "owner", "dataset:public");
        let query = RdfRebacManager::generate_insert_query(&tuple);

        assert!(query.contains("INSERT DATA"));
        assert!(query.contains("<user:alice>"));
        assert!(query.contains("auth:owner"));
        assert!(query.contains("<dataset:public>"));
    }

    #[test]
    fn test_generate_delete_query() {
        let tuple = RelationshipTuple::new("user:alice", "can_read", "dataset:public");
        let query = RdfRebacManager::generate_delete_query(&tuple);

        assert!(query.contains("DELETE WHERE"));
        assert!(query.contains("<user:alice>"));
        assert!(query.contains("auth:canRead"));
        assert!(query.contains("<dataset:public>"));
    }

    #[test]
    fn test_relation_to_property() {
        assert_eq!(
            RdfRebacManager::relation_to_property("can_read"),
            "auth:canRead"
        );
        assert_eq!(RdfRebacManager::relation_to_property("owner"), "auth:owner");
        assert_eq!(
            RdfRebacManager::relation_to_property("member"),
            "auth:memberOf"
        );
    }

    #[test]
    fn test_generate_reified_insert_with_time_condition() {
        let condition = RelationshipCondition::TimeWindow {
            not_before: Some(
                DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
                    .unwrap()
                    .into(),
            ),
            not_after: Some(
                DateTime::parse_from_rfc3339("2025-12-31T23:59:59Z")
                    .unwrap()
                    .into(),
            ),
        };

        let tuple = RelationshipTuple::with_condition(
            "user:alice",
            "can_read",
            "dataset:temporary",
            condition,
        );

        let query = RdfRebacManager::generate_reified_insert(&tuple);

        assert!(query.contains("auth:Relationship"));
        assert!(query.contains("auth:subject"));
        assert!(query.contains("auth:notBefore"));
        assert!(query.contains("auth:notAfter"));
        assert!(query.contains("xsd:dateTime"));
    }
}
