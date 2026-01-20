//! Unified Policy Engine
//!
//! Combines traditional RBAC (Role-Based Access Control) with ReBAC (Relationship-Based Access Control)
//! to provide both backward compatibility and fine-grained authorization capabilities.
//!
//! Policy Evaluation Order:
//! 1. Check RBAC permissions (for backward compatibility with existing roles)
//! 2. If RBAC denies, check ReBAC relationships (new fine-grained control)
//! 3. Return combined result

use crate::auth::permissions::PermissionChecker;
use crate::auth::rebac::{CheckRequest, CheckResponse, RebacError, RebacEvaluator};
use crate::auth::types::{Permission, User};
use async_trait::async_trait;
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Policy engine errors
#[derive(Debug, Error)]
pub enum PolicyEngineError {
    #[error("ReBAC error: {0}")]
    Rebac(#[from] RebacError),

    #[error("Authorization denied: {0}")]
    Denied(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),
}

pub type Result<T> = std::result::Result<T, PolicyEngineError>;

/// Policy evaluation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyMode {
    /// Use only RBAC (traditional, for backward compatibility)
    RbacOnly,

    /// Use only ReBAC (new relationship-based)
    RebacOnly,

    /// Use both: RBAC first, then ReBAC if RBAC denies
    Combined,

    /// Require both RBAC and ReBAC to allow (most restrictive)
    Both,
}

/// Authorization context
#[derive(Debug, Clone)]
pub struct AuthorizationContext {
    /// The user making the request
    pub user: User,

    /// The requested action/permission
    pub action: String,

    /// The target resource
    pub resource: String,

    /// Additional context (IP address, time, etc.)
    pub metadata: std::collections::HashMap<String, String>,
}

impl AuthorizationContext {
    pub fn new(user: User, action: impl Into<String>, resource: impl Into<String>) -> Self {
        Self {
            user,
            action: action.into(),
            resource: resource.into(),
            metadata: std::collections::HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Unified policy engine combining RBAC and ReBAC
pub struct UnifiedPolicyEngine {
    /// ReBAC evaluator
    rebac: Arc<dyn RebacEvaluator>,

    /// Policy evaluation mode
    mode: PolicyMode,

    /// Whether to enable audit logging
    audit_enabled: bool,
}

impl UnifiedPolicyEngine {
    /// Create a new unified policy engine
    pub fn new(rebac: Arc<dyn RebacEvaluator>) -> Self {
        Self {
            rebac,
            mode: PolicyMode::Combined,
            audit_enabled: true,
        }
    }

    /// Set policy evaluation mode
    pub fn with_mode(mut self, mode: PolicyMode) -> Self {
        self.mode = mode;
        self
    }

    /// Enable or disable audit logging
    pub fn with_audit(mut self, enabled: bool) -> Self {
        self.audit_enabled = enabled;
        self
    }

    /// Check authorization using the unified policy
    pub async fn authorize(&self, context: &AuthorizationContext) -> Result<CheckResponse> {
        let start = std::time::Instant::now();

        let result = match self.mode {
            PolicyMode::RbacOnly => self.check_rbac_only(context).await,
            PolicyMode::RebacOnly => self.check_rebac_only(context).await,
            PolicyMode::Combined => self.check_combined(context).await,
            PolicyMode::Both => self.check_both(context).await,
        };

        let elapsed = start.elapsed();

        if self.audit_enabled {
            self.audit_check(context, &result, elapsed);
        }

        result
    }

    /// Check using RBAC only
    async fn check_rbac_only(&self, context: &AuthorizationContext) -> Result<CheckResponse> {
        debug!(
            "RBAC check: user={}, action={}, resource={}",
            context.user.username, context.action, context.resource
        );

        // Parse action into Permission
        if let Some(permission) = self.action_to_permission(&context.action, &context.resource) {
            if PermissionChecker::has_permission(&context.user, &permission) {
                return Ok(CheckResponse::allow());
            }

            // Also check for general permission if specific permission not found
            // e.g., if DatasetWrite not found, check for Write
            match permission {
                Permission::DatasetRead(_)
                    if PermissionChecker::has_permission(&context.user, &Permission::Read) =>
                {
                    return Ok(CheckResponse::allow());
                }
                Permission::DatasetWrite(_)
                    if PermissionChecker::has_permission(&context.user, &Permission::Write) =>
                {
                    return Ok(CheckResponse::allow());
                }
                Permission::DatasetManage
                    if PermissionChecker::has_permission(&context.user, &Permission::Admin) =>
                {
                    return Ok(CheckResponse::allow());
                }
                _ => {}
            }
        }

        Ok(CheckResponse::deny("RBAC: Permission denied"))
    }

    /// Check using ReBAC only
    async fn check_rebac_only(&self, context: &AuthorizationContext) -> Result<CheckResponse> {
        debug!(
            "ReBAC check: user={}, action={}, resource={}",
            context.user.username, context.action, context.resource
        );

        let subject = format!("user:{}", context.user.username);
        let request = CheckRequest::new(subject, &context.action, &context.resource);

        Ok(self.rebac.check(&request).await?)
    }

    /// Check using combined approach (RBAC first, then ReBAC)
    async fn check_combined(&self, context: &AuthorizationContext) -> Result<CheckResponse> {
        debug!(
            "Combined check: user={}, action={}, resource={}",
            context.user.username, context.action, context.resource
        );

        // Try RBAC first (backward compatibility)
        let rbac_result = self.check_rbac_only(context).await?;
        if rbac_result.allowed {
            debug!("Authorized via RBAC");
            return Ok(rbac_result);
        }

        // Fall back to ReBAC (fine-grained control)
        let rebac_result = self.check_rebac_only(context).await?;
        if rebac_result.allowed {
            debug!("Authorized via ReBAC");
            return Ok(rebac_result);
        }

        debug!("Denied by both RBAC and ReBAC");
        Ok(CheckResponse::deny("Both RBAC and ReBAC denied"))
    }

    /// Check requiring both RBAC and ReBAC to allow
    async fn check_both(&self, context: &AuthorizationContext) -> Result<CheckResponse> {
        debug!(
            "Both check: user={}, action={}, resource={}",
            context.user.username, context.action, context.resource
        );

        // Check RBAC
        let rbac_result = self.check_rbac_only(context).await?;
        if !rbac_result.allowed {
            debug!("Denied by RBAC");
            return Ok(rbac_result);
        }

        // Check ReBAC
        let rebac_result = self.check_rebac_only(context).await?;
        if !rebac_result.allowed {
            debug!("Denied by ReBAC");
            return Ok(rebac_result);
        }

        debug!("Authorized by both RBAC and ReBAC");
        Ok(CheckResponse::allow())
    }

    /// Convert action string to Permission enum
    fn action_to_permission(&self, action: &str, resource: &str) -> Option<Permission> {
        match action {
            "can_read" => {
                if resource.starts_with("dataset:") {
                    let dataset = resource.strip_prefix("dataset:")?;
                    Some(Permission::DatasetRead(dataset.to_string()))
                } else {
                    Some(Permission::Read)
                }
            }
            "can_write" => {
                if resource.starts_with("dataset:") {
                    let dataset = resource.strip_prefix("dataset:")?;
                    Some(Permission::DatasetWrite(dataset.to_string()))
                } else {
                    Some(Permission::Write)
                }
            }
            "can_admin" => Some(Permission::Admin),
            "global_admin" => Some(Permission::GlobalAdmin),
            "can_create_dataset" => Some(Permission::DatasetCreate),
            "can_delete_dataset" => Some(Permission::DatasetDelete),
            "can_manage_dataset" => Some(Permission::DatasetManage),
            "can_execute_query" => Some(Permission::QueryExecute),
            "can_execute_update" => Some(Permission::UpdateExecute),
            _ => None,
        }
    }

    /// Audit an authorization check
    fn audit_check(
        &self,
        context: &AuthorizationContext,
        result: &Result<CheckResponse>,
        elapsed: std::time::Duration,
    ) {
        match result {
            Ok(response) if response.allowed => {
                info!(
                    user = %context.user.username,
                    action = %context.action,
                    resource = %context.resource,
                    allowed = true,
                    duration_us = elapsed.as_micros(),
                    "Authorization check"
                );
            }
            Ok(response) => {
                warn!(
                    user = %context.user.username,
                    action = %context.action,
                    resource = %context.resource,
                    allowed = false,
                    reason = ?response.reason,
                    duration_us = elapsed.as_micros(),
                    "Authorization denied"
                );
            }
            Err(e) => {
                warn!(
                    user = %context.user.username,
                    action = %context.action,
                    resource = %context.resource,
                    error = %e,
                    duration_us = elapsed.as_micros(),
                    "Authorization error"
                );
            }
        }
    }

    /// Batch authorization check
    pub async fn batch_authorize(
        &self,
        contexts: &[AuthorizationContext],
    ) -> Result<Vec<CheckResponse>> {
        let mut results = Vec::with_capacity(contexts.len());

        for context in contexts {
            results.push(self.authorize(context).await?);
        }

        Ok(results)
    }
}

/// Helper functions for creating authorization contexts
pub mod helpers {
    use super::*;

    /// Create context for dataset read
    pub fn dataset_read(user: User, dataset: &str) -> AuthorizationContext {
        AuthorizationContext::new(user, "can_read", format!("dataset:{}", dataset))
    }

    /// Create context for dataset write
    pub fn dataset_write(user: User, dataset: &str) -> AuthorizationContext {
        AuthorizationContext::new(user, "can_write", format!("dataset:{}", dataset))
    }

    /// Create context for graph read
    pub fn graph_read(user: User, graph: &str) -> AuthorizationContext {
        AuthorizationContext::new(user, "can_read", format!("graph:{}", graph))
    }

    /// Create context for graph write
    pub fn graph_write(user: User, graph: &str) -> AuthorizationContext {
        AuthorizationContext::new(user, "can_write", format!("graph:{}", graph))
    }

    /// Create context for SPARQL query execution
    pub fn sparql_query(user: User, dataset: &str) -> AuthorizationContext {
        AuthorizationContext::new(user, "can_execute_query", format!("dataset:{}", dataset))
    }

    /// Create context for SPARQL update execution
    pub fn sparql_update(user: User, dataset: &str) -> AuthorizationContext {
        AuthorizationContext::new(user, "can_execute_update", format!("dataset:{}", dataset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::rebac::{InMemoryRebacManager, RelationshipTuple};
    use crate::auth::types::Permission;

    fn create_test_user(username: &str, roles: Vec<String>) -> User {
        User {
            username: username.to_string(),
            roles,
            email: Some(format!("{}@example.com", username)),
            full_name: Some(username.to_string()),
            last_login: None,
            permissions: vec![],
        }
    }

    #[tokio::test]
    async fn test_rbac_only_mode() {
        let rebac = Arc::new(InMemoryRebacManager::new());
        let engine = UnifiedPolicyEngine::new(rebac).with_mode(PolicyMode::RbacOnly);

        // Create user with 'writer' role
        let mut user = create_test_user("alice", vec!["writer".to_string()]);
        user.permissions.push(Permission::Read);
        user.permissions.push(Permission::Write);

        // Should succeed (user has Write permission)
        let context = AuthorizationContext::new(user.clone(), "can_write", "dataset:public");
        let result = engine.authorize(&context).await.unwrap();
        assert!(result.allowed);
    }

    #[tokio::test]
    async fn test_rebac_only_mode() {
        let rebac = Arc::new(InMemoryRebacManager::new());

        // Add relationship tuple
        rebac
            .add_tuple(RelationshipTuple::new(
                "user:alice",
                "can_read",
                "dataset:public",
            ))
            .await
            .unwrap();

        let engine = UnifiedPolicyEngine::new(rebac).with_mode(PolicyMode::RebacOnly);

        let user = create_test_user("alice", vec![]);
        let context = AuthorizationContext::new(user, "can_read", "dataset:public");

        let result = engine.authorize(&context).await.unwrap();
        assert!(result.allowed);
    }

    #[tokio::test]
    async fn test_combined_mode_rbac_allows() {
        let rebac = Arc::new(InMemoryRebacManager::new());
        let engine = UnifiedPolicyEngine::new(rebac).with_mode(PolicyMode::Combined);

        // User has RBAC permission
        let mut user = create_test_user("alice", vec!["reader".to_string()]);
        user.permissions.push(Permission::Read);

        let context = AuthorizationContext::new(user, "can_read", "any_resource");
        let result = engine.authorize(&context).await.unwrap();
        assert!(result.allowed);
    }

    #[tokio::test]
    async fn test_combined_mode_rebac_allows() {
        let rebac = Arc::new(InMemoryRebacManager::new());

        // Add ReBAC relationship
        rebac
            .add_tuple(RelationshipTuple::new(
                "user:bob",
                "can_write",
                "dataset:private",
            ))
            .await
            .unwrap();

        let engine = UnifiedPolicyEngine::new(rebac).with_mode(PolicyMode::Combined);

        // User has no RBAC permissions, but has ReBAC relationship
        let user = create_test_user("bob", vec![]);
        let context = AuthorizationContext::new(user, "can_write", "dataset:private");

        let result = engine.authorize(&context).await.unwrap();
        assert!(result.allowed);
    }

    #[tokio::test]
    async fn test_both_mode_requires_both() {
        let rebac = Arc::new(InMemoryRebacManager::new());

        // Add ReBAC relationship
        rebac
            .add_tuple(RelationshipTuple::new(
                "user:charlie",
                "can_read",
                "dataset:secure",
            ))
            .await
            .unwrap();

        let engine = UnifiedPolicyEngine::new(rebac).with_mode(PolicyMode::Both);

        // User has ReBAC relationship but no RBAC permission
        let user = create_test_user("charlie", vec![]);
        let context = AuthorizationContext::new(user, "can_read", "dataset:secure");

        let result = engine.authorize(&context).await.unwrap();
        // Should be denied because RBAC check fails
        assert!(!result.allowed);
    }

    #[tokio::test]
    async fn test_helpers() {
        let rebac = Arc::new(InMemoryRebacManager::new());
        let engine = UnifiedPolicyEngine::new(rebac);

        let user = create_test_user("alice", vec![]);

        // Test helper functions
        let context = helpers::dataset_read(user.clone(), "public");
        assert_eq!(context.action, "can_read");
        assert_eq!(context.resource, "dataset:public");

        let context = helpers::sparql_query(user, "test");
        assert_eq!(context.action, "can_execute_query");
        assert_eq!(context.resource, "dataset:test");
    }
}
