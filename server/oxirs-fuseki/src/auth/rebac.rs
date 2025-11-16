//! Relationship-Based Access Control (ReBAC) implementation
//!
//! This module provides fine-grained authorization based on relationships between
//! subjects (users, organizations) and resources (datasets, graphs, triples).
//!
//! The ReBAC model maps naturally to RDF triples:
//! - Subject (user:alice) → Predicate (can_read) → Object (dataset:public)
//! - Relationship tuples can be stored as RDF triples for SPARQL-based inference

use crate::auth::types::{Permission, User};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

/// ReBAC errors
#[derive(Debug, Error)]
pub enum RebacError {
    #[error("Relationship not found: {0}")]
    RelationshipNotFound(String),

    #[error("Invalid relationship tuple: {0}")]
    InvalidTuple(String),

    #[error("Permission denied: {subject} cannot {relation} {object}")]
    PermissionDenied {
        subject: String,
        relation: String,
        object: String,
    },

    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, RebacError>;

/// Relationship tuple representing a connection between subject and object
/// This maps to RDF triples: (subject, predicate, object)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct RelationshipTuple {
    /// Subject (e.g., "user:alice", "organization:engineering")
    pub subject: String,

    /// Relation/predicate (e.g., "owner", "can_read", "member")
    pub relation: String,

    /// Object/resource (e.g., "dataset:public", "graph:http://example.org/g1")
    pub object: String,

    /// Optional condition (e.g., time window, IP address)
    pub condition: Option<RelationshipCondition>,
}

/// Conditions that can be attached to relationships
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RelationshipCondition {
    /// Time-based condition
    TimeWindow {
        not_before: Option<chrono::DateTime<chrono::Utc>>,
        not_after: Option<chrono::DateTime<chrono::Utc>>,
    },

    /// IP address condition
    IpAddress { allowed_ips: Vec<String> },

    /// Custom attribute-based condition
    Attribute { key: String, value: String },
}

impl RelationshipTuple {
    /// Create a new relationship tuple
    pub fn new(
        subject: impl Into<String>,
        relation: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            relation: relation.into(),
            object: object.into(),
            condition: None,
        }
    }

    /// Create a tuple with a condition
    pub fn with_condition(
        subject: impl Into<String>,
        relation: impl Into<String>,
        object: impl Into<String>,
        condition: RelationshipCondition,
    ) -> Self {
        Self {
            subject: subject.into(),
            relation: relation.into(),
            object: object.into(),
            condition: Some(condition),
        }
    }

    /// Check if this tuple's condition is satisfied
    pub fn is_condition_satisfied(&self) -> bool {
        match &self.condition {
            None => true,
            Some(RelationshipCondition::TimeWindow {
                not_before,
                not_after,
            }) => {
                let now = chrono::Utc::now();
                let after_start = not_before.map_or(true, |start| now >= start);
                let before_end = not_after.map_or(true, |end| now <= end);
                after_start && before_end
            }
            Some(RelationshipCondition::IpAddress { allowed_ips }) => {
                // TODO: Get actual client IP from request context
                // For now, always allow
                !allowed_ips.is_empty()
            }
            Some(RelationshipCondition::Attribute { key, value }) => {
                // TODO: Check attributes from request context
                // For now, always allow
                !key.is_empty() && !value.is_empty()
            }
        }
    }
}

/// Authorization check request
#[derive(Debug, Clone)]
pub struct CheckRequest {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

impl CheckRequest {
    pub fn new(
        subject: impl Into<String>,
        relation: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            relation: relation.into(),
            object: object.into(),
        }
    }
}

/// Authorization check response
#[derive(Debug, Clone)]
pub struct CheckResponse {
    pub allowed: bool,
    pub reason: Option<String>,
}

impl CheckResponse {
    pub fn allow() -> Self {
        Self {
            allowed: true,
            reason: None,
        }
    }

    pub fn deny(reason: impl Into<String>) -> Self {
        Self {
            allowed: false,
            reason: Some(reason.into()),
        }
    }
}

/// ReBAC evaluator trait - defines the interface for authorization checks
#[async_trait]
pub trait RebacEvaluator: Send + Sync {
    /// Check if a subject has a specific relation to an object
    async fn check(&self, request: &CheckRequest) -> Result<CheckResponse>;

    /// Add a relationship tuple
    async fn add_tuple(&self, tuple: RelationshipTuple) -> Result<()>;

    /// Remove a relationship tuple
    async fn remove_tuple(&self, tuple: &RelationshipTuple) -> Result<()>;

    /// List all tuples for a subject
    async fn list_subject_tuples(&self, subject: &str) -> Result<Vec<RelationshipTuple>>;

    /// List all tuples for an object
    async fn list_object_tuples(&self, object: &str) -> Result<Vec<RelationshipTuple>>;

    /// Batch check multiple requests
    async fn batch_check(&self, requests: &[CheckRequest]) -> Result<Vec<CheckResponse>> {
        let mut results = Vec::with_capacity(requests.len());
        for request in requests {
            results.push(self.check(request).await?);
        }
        Ok(results)
    }
}

/// In-memory ReBAC manager
pub struct InMemoryRebacManager {
    /// Relationship tuples indexed by subject
    tuples_by_subject: Arc<RwLock<HashMap<String, Vec<RelationshipTuple>>>>,

    /// Relationship tuples indexed by object
    tuples_by_object: Arc<RwLock<HashMap<String, Vec<RelationshipTuple>>>>,

    /// Relationship graph for path-based checks
    graph: Arc<RwLock<RelationshipGraph>>,
}

/// Relationship graph for traversal-based authorization
#[derive(Debug, Default)]
struct RelationshipGraph {
    /// Edges in the graph: (subject, relation) -> Vec<object>
    edges: HashMap<(String, String), Vec<String>>,
}

impl RelationshipGraph {
    fn add_edge(&mut self, subject: String, relation: String, object: String) {
        self.edges
            .entry((subject, relation))
            .or_insert_with(Vec::new)
            .push(object);
    }

    fn remove_edge(&mut self, subject: &str, relation: &str, object: &str) {
        if let Some(objects) = self
            .edges
            .get_mut(&(subject.to_string(), relation.to_string()))
        {
            objects.retain(|o| o != object);
        }
    }

    /// Check if there's a path from subject to object via the given relation
    fn has_path(&self, subject: &str, relation: &str, object: &str) -> bool {
        // Direct check
        if let Some(objects) = self.edges.get(&(subject.to_string(), relation.to_string())) {
            if objects.contains(&object.to_string()) {
                return true;
            }
        }

        // TODO: Implement transitive relationship traversal
        // For now, only direct relationships are checked
        false
    }
}

impl InMemoryRebacManager {
    /// Create a new in-memory ReBAC manager
    pub fn new() -> Self {
        Self {
            tuples_by_subject: Arc::new(RwLock::new(HashMap::new())),
            tuples_by_object: Arc::new(RwLock::new(HashMap::new())),
            graph: Arc::new(RwLock::new(RelationshipGraph::default())),
        }
    }

    /// Initialize with predefined tuples (for testing/demo)
    pub async fn with_tuples(tuples: Vec<RelationshipTuple>) -> Result<Self> {
        let manager = Self::new();
        for tuple in tuples {
            manager.add_tuple(tuple).await?;
        }
        Ok(manager)
    }
}

impl Default for InMemoryRebacManager {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl RebacEvaluator for InMemoryRebacManager {
    async fn check(&self, request: &CheckRequest) -> Result<CheckResponse> {
        // Check if relationship exists in graph
        let graph = self.graph.read().await;
        let has_relation = graph.has_path(&request.subject, &request.relation, &request.object);

        if !has_relation {
            return Ok(CheckResponse::deny(format!(
                "{} does not have {} on {}",
                request.subject, request.relation, request.object
            )));
        }

        // Check conditions
        let tuples_by_subject = self.tuples_by_subject.read().await;
        if let Some(tuples) = tuples_by_subject.get(&request.subject) {
            for tuple in tuples {
                if tuple.relation == request.relation
                    && tuple.object == request.object
                    && !tuple.is_condition_satisfied()
                {
                    return Ok(CheckResponse::deny("Condition not satisfied"));
                }
            }
        }

        Ok(CheckResponse::allow())
    }

    async fn add_tuple(&self, tuple: RelationshipTuple) -> Result<()> {
        // Add to subject index (check for duplicates)
        {
            let mut tuples_by_subject = self.tuples_by_subject.write().await;
            let subject_tuples = tuples_by_subject
                .entry(tuple.subject.clone())
                .or_insert_with(Vec::new);

            // Only add if not already present
            if !subject_tuples.contains(&tuple) {
                subject_tuples.push(tuple.clone());
            }
        }

        // Add to object index (check for duplicates)
        {
            let mut tuples_by_object = self.tuples_by_object.write().await;
            let object_tuples = tuples_by_object
                .entry(tuple.object.clone())
                .or_insert_with(Vec::new);

            // Only add if not already present
            if !object_tuples.contains(&tuple) {
                object_tuples.push(tuple.clone());
            }
        }

        // Add to graph
        {
            let mut graph = self.graph.write().await;
            graph.add_edge(
                tuple.subject.clone(),
                tuple.relation.clone(),
                tuple.object.clone(),
            );
        }

        Ok(())
    }

    async fn remove_tuple(&self, tuple: &RelationshipTuple) -> Result<()> {
        // Remove from subject index
        {
            let mut tuples_by_subject = self.tuples_by_subject.write().await;
            if let Some(tuples) = tuples_by_subject.get_mut(&tuple.subject) {
                tuples.retain(|t| t != tuple);
            }
        }

        // Remove from object index
        {
            let mut tuples_by_object = self.tuples_by_object.write().await;
            if let Some(tuples) = tuples_by_object.get_mut(&tuple.object) {
                tuples.retain(|t| t != tuple);
            }
        }

        // Remove from graph
        {
            let mut graph = self.graph.write().await;
            graph.remove_edge(&tuple.subject, &tuple.relation, &tuple.object);
        }

        Ok(())
    }

    async fn list_subject_tuples(&self, subject: &str) -> Result<Vec<RelationshipTuple>> {
        let tuples_by_subject = self.tuples_by_subject.read().await;
        Ok(tuples_by_subject.get(subject).cloned().unwrap_or_default())
    }

    async fn list_object_tuples(&self, object: &str) -> Result<Vec<RelationshipTuple>> {
        let tuples_by_object = self.tuples_by_object.read().await;
        Ok(tuples_by_object.get(object).cloned().unwrap_or_default())
    }
}

/// Implement RebacEvaluator for `Arc<T>` to allow tests to use Arc-wrapped managers
#[async_trait]
impl<T: RebacEvaluator> RebacEvaluator for Arc<T> {
    async fn check(&self, request: &CheckRequest) -> Result<CheckResponse> {
        (**self).check(request).await
    }

    async fn add_tuple(&self, tuple: RelationshipTuple) -> Result<()> {
        (**self).add_tuple(tuple).await
    }

    async fn remove_tuple(&self, tuple: &RelationshipTuple) -> Result<()> {
        (**self).remove_tuple(tuple).await
    }

    async fn list_subject_tuples(&self, subject: &str) -> Result<Vec<RelationshipTuple>> {
        (**self).list_subject_tuples(subject).await
    }

    async fn list_object_tuples(&self, object: &str) -> Result<Vec<RelationshipTuple>> {
        (**self).list_object_tuples(object).await
    }
}

/// Utility functions for ReBAC integration
pub mod util {
    use super::*;

    /// Convert Permission to ReBAC relation string
    pub fn permission_to_relation(permission: &Permission) -> String {
        match permission {
            Permission::Read => "can_read".to_string(),
            Permission::Write => "can_write".to_string(),
            Permission::Admin => "can_admin".to_string(),
            Permission::GlobalAdmin => "global_admin".to_string(),
            Permission::GlobalRead => "global_read".to_string(),
            Permission::GlobalWrite => "global_write".to_string(),
            Permission::DatasetRead(ds) => format!("can_read_dataset:{}", ds),
            Permission::DatasetWrite(ds) => format!("can_write_dataset:{}", ds),
            Permission::DatasetCreate => "can_create_dataset".to_string(),
            Permission::DatasetDelete => "can_delete_dataset".to_string(),
            Permission::DatasetManage => "can_manage_dataset".to_string(),
            _ => format!("{:?}", permission).to_lowercase(),
        }
    }

    /// Create subject identifier from user
    pub fn user_to_subject(user: &User) -> String {
        format!("user:{}", user.username)
    }

    /// Create object identifier for dataset
    pub fn dataset_to_object(dataset: &str) -> String {
        format!("dataset:{}", dataset)
    }

    /// Create object identifier for graph
    pub fn graph_to_object(graph: &str) -> String {
        format!("graph:{}", graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_relationship() {
        let manager = InMemoryRebacManager::new();

        // Add relationship: alice can read dataset:public
        let tuple = RelationshipTuple::new("user:alice", "can_read", "dataset:public");
        manager.add_tuple(tuple).await.unwrap();

        // Check if alice can read dataset:public
        let request = CheckRequest::new("user:alice", "can_read", "dataset:public");
        let response = manager.check(&request).await.unwrap();
        assert!(response.allowed);

        // Check if alice can write dataset:public (should fail)
        let request = CheckRequest::new("user:alice", "can_write", "dataset:public");
        let response = manager.check(&request).await.unwrap();
        assert!(!response.allowed);
    }

    #[tokio::test]
    async fn test_list_tuples() {
        let manager = InMemoryRebacManager::new();

        // Add multiple relationships for alice
        manager
            .add_tuple(RelationshipTuple::new(
                "user:alice",
                "can_read",
                "dataset:public",
            ))
            .await
            .unwrap();
        manager
            .add_tuple(RelationshipTuple::new(
                "user:alice",
                "can_write",
                "dataset:private",
            ))
            .await
            .unwrap();

        // List all tuples for alice
        let tuples = manager.list_subject_tuples("user:alice").await.unwrap();
        assert_eq!(tuples.len(), 2);
    }

    #[tokio::test]
    async fn test_time_based_condition() {
        let manager = InMemoryRebacManager::new();

        // Add relationship with time window (already expired)
        let tuple = RelationshipTuple::with_condition(
            "user:alice",
            "can_read",
            "dataset:temporary",
            RelationshipCondition::TimeWindow {
                not_before: Some(chrono::Utc::now() - chrono::Duration::hours(2)),
                not_after: Some(chrono::Utc::now() - chrono::Duration::hours(1)),
            },
        );
        manager.add_tuple(tuple).await.unwrap();

        // Check should fail due to expired time window
        let request = CheckRequest::new("user:alice", "can_read", "dataset:temporary");
        let response = manager.check(&request).await.unwrap();
        assert!(!response.allowed);
    }

    #[tokio::test]
    async fn test_remove_tuple() {
        let manager = InMemoryRebacManager::new();

        // Add relationship
        let tuple = RelationshipTuple::new("user:alice", "can_read", "dataset:public");
        manager.add_tuple(tuple.clone()).await.unwrap();

        // Verify it exists
        let request = CheckRequest::new("user:alice", "can_read", "dataset:public");
        let response = manager.check(&request).await.unwrap();
        assert!(response.allowed);

        // Remove relationship
        manager.remove_tuple(&tuple).await.unwrap();

        // Verify it's gone
        let response = manager.check(&request).await.unwrap();
        assert!(!response.allowed);
    }
}
