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

    /// Object/resource (e.g., "dataset:public", "graph:`http://example.org/g1`")
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
        self.is_condition_satisfied_with_context(None)
    }

    /// Check if this tuple's condition is satisfied with request context
    pub fn is_condition_satisfied_with_context(&self, context: Option<&RequestContext>) -> bool {
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
                // Check actual client IP from request context
                if let Some(ctx) = context {
                    if let Some(client_ip) = &ctx.client_ip {
                        // Check if client IP matches any allowed IPs
                        // Support CIDR notation and exact matches
                        return allowed_ips
                            .iter()
                            .any(|allowed| Self::ip_matches(client_ip, allowed));
                    }
                }
                // If no context or no client IP, deny access
                false
            }
            Some(RelationshipCondition::Attribute { key, value }) => {
                // Check attributes from request context
                if let Some(ctx) = context {
                    if let Some(attr_value) = ctx.attributes.get(key) {
                        return attr_value == value;
                    }
                }
                // If no context or attribute not found, deny access
                false
            }
        }
    }

    /// Check if an IP address matches an allowed pattern
    /// Supports exact matches and basic CIDR notation
    fn ip_matches(client_ip: &str, allowed_pattern: &str) -> bool {
        // Exact match
        if client_ip == allowed_pattern {
            return true;
        }

        // Basic CIDR support (e.g., "192.168.1.0/24")
        if allowed_pattern.contains('/') {
            // Simple prefix matching for demonstration
            // In production, use a proper CIDR library like `ipnetwork`
            let prefix = allowed_pattern.split('/').next().unwrap_or("");
            let prefix_parts: Vec<&str> = prefix.split('.').collect();
            let client_parts: Vec<&str> = client_ip.split('.').collect();

            if prefix_parts.len() >= 3 && client_parts.len() >= 3 {
                // Match first 3 octets for /24
                return prefix_parts[0..3] == client_parts[0..3];
            }
        }

        false
    }
}

/// Authorization check request context
#[derive(Debug, Clone, Default)]
pub struct RequestContext {
    /// Client IP address (for IP-based conditions)
    pub client_ip: Option<String>,
    /// Custom attributes (for attribute-based conditions)
    pub attributes: HashMap<String, String>,
}

impl RequestContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_client_ip(mut self, ip: impl Into<String>) -> Self {
        self.client_ip = Some(ip.into());
        self
    }

    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}

/// Authorization check request
#[derive(Debug, Clone)]
pub struct CheckRequest {
    pub subject: String,
    pub relation: String,
    pub object: String,
    /// Request context for condition evaluation
    pub context: Option<RequestContext>,
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
            context: None,
        }
    }

    pub fn with_context(mut self, context: RequestContext) -> Self {
        self.context = Some(context);
        self
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
    /// Uses breadth-first search (BFS) for transitive relationship traversal
    fn has_path(&self, subject: &str, relation: &str, object: &str) -> bool {
        use std::collections::{HashSet, VecDeque};

        // Direct check first (optimization)
        if let Some(objects) = self.edges.get(&(subject.to_string(), relation.to_string())) {
            if objects.contains(&object.to_string()) {
                return true;
            }
        }

        // Transitive relationship traversal using BFS
        // Example: user:alice → member → org:engineering → owner → dataset:data
        // This allows checking if alice can access dataset:data through organizational hierarchy

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Start from the subject
        queue.push_back(subject.to_string());
        visited.insert(subject.to_string());

        while let Some(current) = queue.pop_front() {
            // Check all outgoing edges with the given relation
            if let Some(targets) = self.edges.get(&(current.clone(), relation.to_string())) {
                for target in targets {
                    // Found the target object
                    if target == object {
                        return true;
                    }

                    // Add to queue if not visited (avoid cycles)
                    if !visited.contains(target) {
                        visited.insert(target.clone());
                        queue.push_back(target.clone());
                    }
                }
            }

            // Also check for hierarchical relations (e.g., member → owner)
            // This allows permission inheritance through organizational structures
            if let Some(inherited_targets) = self.get_inherited_objects(&current, relation) {
                for target in inherited_targets {
                    if target == object {
                        return true;
                    }

                    if !visited.contains(&target) {
                        visited.insert(target.clone());
                        queue.push_back(target);
                    }
                }
            }
        }

        false
    }

    /// Get inherited objects through permission hierarchy
    /// For example: "member" may inherit "owner" permissions
    fn get_inherited_objects(&self, subject: &str, relation: &str) -> Option<Vec<String>> {
        // Define permission hierarchy
        let hierarchy = match relation {
            "owner" => vec!["owner"],
            "editor" => vec!["editor", "owner"], // editors inherit owner permissions
            "viewer" => vec!["viewer", "editor", "owner"], // viewers inherit all
            "member" => vec!["member", "owner"], // members inherit owner permissions
            _ => vec![relation],                 // Default: no inheritance
        };

        let mut inherited = Vec::new();
        for inherited_relation in hierarchy {
            if let Some(objects) = self
                .edges
                .get(&(subject.to_string(), inherited_relation.to_string()))
            {
                inherited.extend(objects.clone());
            }
        }

        if inherited.is_empty() {
            None
        } else {
            Some(inherited)
        }
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

        // Check conditions with request context
        let tuples_by_subject = self.tuples_by_subject.read().await;
        if let Some(tuples) = tuples_by_subject.get(&request.subject) {
            for tuple in tuples {
                if tuple.relation == request.relation
                    && tuple.object == request.object
                    && !tuple.is_condition_satisfied_with_context(request.context.as_ref())
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

    /// Optimized batch check implementation
    /// Acquires locks once and processes all requests in a single pass
    async fn batch_check(&self, requests: &[CheckRequest]) -> Result<Vec<CheckResponse>> {
        // Acquire read locks once for entire batch
        let graph = self.graph.read().await;
        let tuples_by_subject = self.tuples_by_subject.read().await;

        let mut results = Vec::with_capacity(requests.len());

        for request in requests {
            // Check if relationship exists in graph (uses transitive traversal)
            let has_relation = graph.has_path(&request.subject, &request.relation, &request.object);

            if !has_relation {
                results.push(CheckResponse::deny(format!(
                    "{} does not have {} on {}",
                    request.subject, request.relation, request.object
                )));
                continue;
            }

            // Check conditions for this specific relationship with context
            let mut condition_satisfied = true;
            if let Some(tuples) = tuples_by_subject.get(&request.subject) {
                for tuple in tuples {
                    if tuple.relation == request.relation
                        && tuple.object == request.object
                        && !tuple.is_condition_satisfied_with_context(request.context.as_ref())
                    {
                        condition_satisfied = false;
                        break;
                    }
                }
            }

            if condition_satisfied {
                results.push(CheckResponse::allow());
            } else {
                results.push(CheckResponse::deny("Condition not satisfied"));
            }
        }

        Ok(results)
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

    async fn batch_check(&self, requests: &[CheckRequest]) -> Result<Vec<CheckResponse>> {
        (**self).batch_check(requests).await
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
