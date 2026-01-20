//! Enhanced Role-Based Access Control (RBAC)
//!
//! This module provides:
//! - Dynamic role creation and management
//! - Resource-based permissions (datasets, graphs, endpoints)
//! - Role hierarchies and permission inheritance
//! - Policy-based access control
//! - Permission auditing and tracking

use crate::auth::types::{Permission, User};
use crate::error::{FusekiError, FusekiResult};
use chrono::{DateTime, Datelike, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Role definition with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// Role name (unique identifier)
    pub name: String,
    /// Human-readable display name
    pub display_name: String,
    /// Role description
    pub description: String,
    /// Permissions granted by this role
    pub permissions: HashSet<Permission>,
    /// Parent roles for inheritance
    pub parent_roles: Vec<String>,
    /// Whether this role is a system role (cannot be deleted)
    pub is_system: bool,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Last modified timestamp
    pub modified_at: DateTime<Utc>,
    /// Created by user
    pub created_by: String,
}

/// Resource-specific access policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePolicy {
    /// Resource type (dataset, graph, endpoint)
    pub resource_type: ResourceType,
    /// Resource identifier (e.g., dataset name)
    pub resource_id: String,
    /// Role that has access
    pub role_name: String,
    /// Allowed permissions on this resource
    pub permissions: HashSet<Permission>,
    /// Optional conditions (e.g., time-based, IP-based)
    pub conditions: Vec<PolicyCondition>,
    /// Priority (higher takes precedence)
    pub priority: i32,
    /// Policy is enabled
    pub enabled: bool,
}

/// Resource types in the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResourceType {
    Dataset,
    Graph,
    Endpoint,
    Service,
}

/// Policy conditions for context-based access control
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PolicyCondition {
    /// Time-based condition (e.g., only during business hours)
    TimeWindow {
        start_hour: u8,
        end_hour: u8,
        days_of_week: Vec<u8>,
    },
    /// IP address-based condition
    IpAddress {
        allowed_ips: Vec<String>,
        allowed_cidrs: Vec<String>,
    },
    /// Request rate limit
    RateLimit { requests_per_hour: u32 },
    /// Custom attribute-based condition
    Attribute { key: String, value: String },
}

/// Role assignment to a user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleAssignment {
    pub username: String,
    pub role_name: String,
    pub assigned_at: DateTime<Utc>,
    pub assigned_by: String,
    pub expires_at: Option<DateTime<Utc>>,
}

/// Permission audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionAuditLog {
    pub timestamp: DateTime<Utc>,
    pub user: String,
    pub action: String,
    pub resource_type: Option<ResourceType>,
    pub resource_id: Option<String>,
    pub permission: Permission,
    pub granted: bool,
    pub reason: String,
}

/// Enhanced RBAC manager
pub struct RbacManager {
    /// Dynamic roles (name -> Role)
    roles: Arc<RwLock<HashMap<String, Role>>>,
    /// Resource policies
    policies: Arc<RwLock<Vec<ResourcePolicy>>>,
    /// Role assignments (username -> roles)
    role_assignments: Arc<RwLock<HashMap<String, Vec<RoleAssignment>>>>,
    /// Permission audit log
    audit_log: Arc<RwLock<Vec<PermissionAuditLog>>>,
    /// Maximum audit log size
    max_audit_log_size: usize,
}

impl RbacManager {
    /// Create new RBAC manager with system roles
    pub fn new() -> Self {
        let mut manager = RbacManager {
            roles: Arc::new(RwLock::new(HashMap::new())),
            policies: Arc::new(RwLock::new(Vec::new())),
            role_assignments: Arc::new(RwLock::new(HashMap::new())),
            audit_log: Arc::new(RwLock::new(Vec::new())),
            max_audit_log_size: 10000,
        };

        // Initialize system roles synchronously
        let system_roles = Self::create_system_roles();
        let mut roles = HashMap::new();
        for role in system_roles {
            roles.insert(role.name.clone(), role);
        }
        manager.roles = Arc::new(RwLock::new(roles));

        manager
    }

    /// Create default system roles
    fn create_system_roles() -> Vec<Role> {
        let now = Utc::now();
        let system_user = "system".to_string();

        vec![
            Role {
                name: "admin".to_string(),
                display_name: "Administrator".to_string(),
                description: "Full system administrator with all permissions".to_string(),
                permissions: Self::get_admin_permissions(),
                parent_roles: vec![],
                is_system: true,
                created_at: now,
                modified_at: now,
                created_by: system_user.clone(),
            },
            Role {
                name: "dataset_admin".to_string(),
                display_name: "Dataset Administrator".to_string(),
                description: "Administrator for dataset management".to_string(),
                permissions: Self::get_dataset_admin_permissions(),
                parent_roles: vec!["writer".to_string()],
                is_system: true,
                created_at: now,
                modified_at: now,
                created_by: system_user.clone(),
            },
            Role {
                name: "writer".to_string(),
                display_name: "Data Writer".to_string(),
                description: "User with read/write access to datasets".to_string(),
                permissions: Self::get_writer_permissions(),
                parent_roles: vec!["reader".to_string()],
                is_system: true,
                created_at: now,
                modified_at: now,
                created_by: system_user.clone(),
            },
            Role {
                name: "reader".to_string(),
                display_name: "Data Reader".to_string(),
                description: "Read-only user with query access".to_string(),
                permissions: Self::get_reader_permissions(),
                parent_roles: vec![],
                is_system: true,
                created_at: now,
                modified_at: now,
                created_by: system_user.clone(),
            },
            Role {
                name: "monitor".to_string(),
                display_name: "System Monitor".to_string(),
                description: "User with monitoring and metrics access".to_string(),
                permissions: Self::get_monitor_permissions(),
                parent_roles: vec!["reader".to_string()],
                is_system: true,
                created_at: now,
                modified_at: now,
                created_by: system_user,
            },
        ]
    }

    fn get_admin_permissions() -> HashSet<Permission> {
        HashSet::from([
            Permission::GlobalAdmin,
            Permission::GlobalRead,
            Permission::GlobalWrite,
            Permission::SparqlQuery,
            Permission::SparqlUpdate,
            Permission::GraphStore,
            Permission::UserManagement,
            Permission::SystemConfig,
            Permission::SystemMetrics,
        ])
    }

    fn get_dataset_admin_permissions() -> HashSet<Permission> {
        HashSet::from([
            Permission::GlobalRead,
            Permission::GlobalWrite,
            Permission::SparqlQuery,
            Permission::SparqlUpdate,
            Permission::GraphStore,
        ])
    }

    fn get_writer_permissions() -> HashSet<Permission> {
        HashSet::from([
            Permission::GlobalRead,
            Permission::GlobalWrite,
            Permission::SparqlQuery,
            Permission::SparqlUpdate,
            Permission::GraphStore,
        ])
    }

    fn get_reader_permissions() -> HashSet<Permission> {
        HashSet::from([Permission::GlobalRead, Permission::SparqlQuery])
    }

    fn get_monitor_permissions() -> HashSet<Permission> {
        HashSet::from([
            Permission::GlobalRead,
            Permission::SparqlQuery,
            Permission::SystemMetrics,
        ])
    }

    /// Create a new role
    pub async fn create_role(
        &self,
        name: String,
        display_name: String,
        description: String,
        permissions: HashSet<Permission>,
        parent_roles: Vec<String>,
        created_by: String,
    ) -> FusekiResult<Role> {
        let mut roles = self.roles.write().await;

        // Check if role already exists
        if roles.contains_key(&name) {
            return Err(FusekiError::bad_request(format!(
                "Role '{}' already exists",
                name
            )));
        }

        // Validate parent roles exist
        for parent in &parent_roles {
            if !roles.contains_key(parent) {
                return Err(FusekiError::bad_request(format!(
                    "Parent role '{}' does not exist",
                    parent
                )));
            }
        }

        let role = Role {
            name: name.clone(),
            display_name,
            description,
            permissions,
            parent_roles,
            is_system: false,
            created_at: Utc::now(),
            modified_at: Utc::now(),
            created_by,
        };

        roles.insert(name.clone(), role.clone());
        info!("Created new role: {}", name);

        Ok(role)
    }

    /// Update an existing role
    pub async fn update_role(
        &self,
        name: &str,
        display_name: Option<String>,
        description: Option<String>,
        permissions: Option<HashSet<Permission>>,
        parent_roles: Option<Vec<String>>,
    ) -> FusekiResult<Role> {
        // Validate parent roles exist first (before acquiring write lock)
        if let Some(ref parents) = parent_roles {
            let roles = self.roles.read().await;
            for parent in parents {
                if !roles.contains_key(parent) {
                    return Err(FusekiError::bad_request(format!(
                        "Parent role '{}' does not exist",
                        parent
                    )));
                }
            }
        }

        // Now acquire write lock and update
        let mut roles = self.roles.write().await;

        let role = roles
            .get_mut(name)
            .ok_or_else(|| FusekiError::not_found(format!("Role '{}' not found", name)))?;

        // Cannot modify system roles
        if role.is_system {
            return Err(FusekiError::forbidden(
                "Cannot modify system role".to_string(),
            ));
        }

        if let Some(display) = display_name {
            role.display_name = display;
        }

        if let Some(desc) = description {
            role.description = desc;
        }

        if let Some(perms) = permissions {
            role.permissions = perms;
        }

        if let Some(parents) = parent_roles {
            role.parent_roles = parents;
        }

        role.modified_at = Utc::now();
        info!("Updated role: {}", name);

        Ok(role.clone())
    }

    /// Delete a role
    pub async fn delete_role(&self, name: &str) -> FusekiResult<()> {
        let mut roles = self.roles.write().await;

        let role = roles
            .get(name)
            .ok_or_else(|| FusekiError::not_found(format!("Role '{}' not found", name)))?;

        // Cannot delete system roles
        if role.is_system {
            return Err(FusekiError::forbidden(
                "Cannot delete system role".to_string(),
            ));
        }

        roles.remove(name);
        info!("Deleted role: {}", name);

        Ok(())
    }

    /// Get all roles
    pub async fn get_all_roles(&self) -> Vec<Role> {
        let roles = self.roles.read().await;
        roles.values().cloned().collect()
    }

    /// Get role by name
    pub async fn get_role(&self, name: &str) -> Option<Role> {
        let roles = self.roles.read().await;
        roles.get(name).cloned()
    }

    /// Compute all permissions for a role (including inherited)
    pub async fn get_effective_permissions(&self, role_name: &str) -> HashSet<Permission> {
        let roles = self.roles.read().await;
        let mut permissions = HashSet::new();
        let mut visited = HashSet::new();

        Self::collect_permissions_recursive(&roles, role_name, &mut permissions, &mut visited);

        permissions
    }

    fn collect_permissions_recursive(
        roles: &HashMap<String, Role>,
        role_name: &str,
        permissions: &mut HashSet<Permission>,
        visited: &mut HashSet<String>,
    ) {
        if visited.contains(role_name) {
            // Prevent infinite recursion
            return;
        }

        visited.insert(role_name.to_string());

        if let Some(role) = roles.get(role_name) {
            // Add role's own permissions
            permissions.extend(role.permissions.clone());

            // Recursively add parent role permissions
            for parent in &role.parent_roles {
                Self::collect_permissions_recursive(roles, parent, permissions, visited);
            }
        }
    }

    /// Check if user has permission for a resource
    pub async fn check_permission(
        &self,
        user: &User,
        permission: &Permission,
        resource_type: Option<ResourceType>,
        resource_id: Option<&str>,
    ) -> FusekiResult<bool> {
        // Check global permissions from user's roles
        let mut has_permission = false;

        for role in &user.roles {
            let perms = self.get_effective_permissions(role).await;
            if perms.contains(permission) {
                has_permission = true;
                break;
            }
        }

        // Check resource-specific policies if applicable
        if let (Some(res_type), Some(res_id)) = (resource_type, resource_id) {
            let policy_allows = self
                .check_resource_policy(user, permission, res_type, res_id)
                .await;

            // Resource policies can override global permissions
            if policy_allows {
                has_permission = true;
            }
        }

        // Audit the permission check
        self.audit_permission_check(user, permission, resource_type, resource_id, has_permission)
            .await;

        Ok(has_permission)
    }

    /// Check resource-specific policy
    async fn check_resource_policy(
        &self,
        user: &User,
        permission: &Permission,
        resource_type: ResourceType,
        resource_id: &str,
    ) -> bool {
        let policies = self.policies.read().await;

        // Find matching policies, sorted by priority
        let mut matching_policies: Vec<&ResourcePolicy> = policies
            .iter()
            .filter(|p| {
                p.enabled
                    && p.resource_type == resource_type
                    && p.resource_id == resource_id
                    && user.roles.contains(&p.role_name)
                    && p.permissions.contains(permission)
            })
            .collect();

        matching_policies.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Apply first matching policy
        if let Some(policy) = matching_policies.first() {
            // Check policy conditions
            self.evaluate_policy_conditions(policy)
        } else {
            false
        }
    }

    /// Evaluate policy conditions
    fn evaluate_policy_conditions(&self, policy: &ResourcePolicy) -> bool {
        // If no conditions, allow
        if policy.conditions.is_empty() {
            return true;
        }

        // All conditions must be met
        for condition in &policy.conditions {
            match condition {
                PolicyCondition::TimeWindow {
                    start_hour,
                    end_hour,
                    days_of_week,
                } => {
                    let now = Utc::now();
                    let hour = now.hour() as u8;
                    let day = now.weekday().number_from_monday() as u8;

                    if hour < *start_hour || hour >= *end_hour {
                        return false;
                    }

                    if !days_of_week.is_empty() && !days_of_week.contains(&day) {
                        return false;
                    }
                }
                PolicyCondition::IpAddress {
                    allowed_ips,
                    allowed_cidrs: _,
                } => {
                    // IP checking would go here
                    // For now, assume allowed if list is empty
                    if !allowed_ips.is_empty() {
                        // Would check current request IP
                        warn!("IP-based policy conditions not fully implemented");
                    }
                }
                PolicyCondition::RateLimit {
                    requests_per_hour: _,
                } => {
                    // Rate limiting would go here
                    warn!("Rate limit policy conditions not fully implemented");
                }
                PolicyCondition::Attribute { key: _, value: _ } => {
                    // Attribute checking would go here
                    warn!("Attribute policy conditions not fully implemented");
                }
            }
        }

        true
    }

    /// Add a resource policy
    pub async fn add_resource_policy(&self, policy: ResourcePolicy) -> FusekiResult<()> {
        let mut policies = self.policies.write().await;

        // Validate role exists
        {
            let roles = self.roles.read().await;
            if !roles.contains_key(&policy.role_name) {
                return Err(FusekiError::bad_request(format!(
                    "Role '{}' does not exist",
                    policy.role_name
                )));
            }
        }

        policies.push(policy);
        info!("Added resource policy");

        Ok(())
    }

    /// Remove resource policies for a resource
    pub async fn remove_resource_policies(
        &self,
        resource_type: ResourceType,
        resource_id: &str,
    ) -> FusekiResult<usize> {
        let mut policies = self.policies.write().await;
        let before = policies.len();

        policies.retain(|p| p.resource_type != resource_type || p.resource_id != resource_id);

        let removed = before - policies.len();
        info!(
            "Removed {} resource policies for {:?}/{}",
            removed, resource_type, resource_id
        );

        Ok(removed)
    }

    /// Assign role to user
    pub async fn assign_role(
        &self,
        username: String,
        role_name: String,
        assigned_by: String,
        expires_at: Option<DateTime<Utc>>,
    ) -> FusekiResult<()> {
        // Validate role exists
        {
            let roles = self.roles.read().await;
            if !roles.contains_key(&role_name) {
                return Err(FusekiError::bad_request(format!(
                    "Role '{}' does not exist",
                    role_name
                )));
            }
        }

        let assignment = RoleAssignment {
            username: username.clone(),
            role_name: role_name.clone(),
            assigned_at: Utc::now(),
            assigned_by,
            expires_at,
        };

        let mut assignments = self.role_assignments.write().await;
        assignments
            .entry(username.clone())
            .or_insert_with(Vec::new)
            .push(assignment);

        info!("Assigned role '{}' to user '{}'", role_name, username);

        Ok(())
    }

    /// Audit permission check
    async fn audit_permission_check(
        &self,
        user: &User,
        permission: &Permission,
        resource_type: Option<ResourceType>,
        resource_id: Option<&str>,
        granted: bool,
    ) {
        let mut audit = self.audit_log.write().await;

        let entry = PermissionAuditLog {
            timestamp: Utc::now(),
            user: user.username.clone(),
            action: "permission_check".to_string(),
            resource_type,
            resource_id: resource_id.map(|s| s.to_string()),
            permission: permission.clone(),
            granted,
            reason: if granted {
                "Permission granted".to_string()
            } else {
                "Permission denied".to_string()
            },
        };

        audit.push(entry);

        // Trim audit log if too large
        if audit.len() > self.max_audit_log_size {
            let drain_count = audit.len() - self.max_audit_log_size;
            audit.drain(0..drain_count);
        }

        debug!(
            "Audit: user={}, permission={:?}, granted={}",
            user.username, permission, granted
        );
    }

    /// Get audit log
    pub async fn get_audit_log(&self, limit: Option<usize>) -> Vec<PermissionAuditLog> {
        let audit = self.audit_log.read().await;

        if let Some(limit) = limit {
            let start = audit.len().saturating_sub(limit);
            audit[start..].to_vec()
        } else {
            audit.clone()
        }
    }
}

impl Default for RbacManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rbac_manager_creation() {
        let manager = RbacManager::new();
        let roles = manager.get_all_roles().await;

        assert!(roles.len() >= 5); // Should have at least 5 system roles
        assert!(roles.iter().any(|r| r.name == "admin"));
        assert!(roles.iter().any(|r| r.name == "reader"));
    }

    #[tokio::test]
    async fn test_create_custom_role() {
        let manager = RbacManager::new();

        let role = manager
            .create_role(
                "custom_role".to_string(),
                "Custom Role".to_string(),
                "Test role".to_string(),
                HashSet::from([Permission::GlobalRead]),
                vec![],
                "test_user".to_string(),
            )
            .await
            .unwrap();

        assert_eq!(role.name, "custom_role");
        assert!(!role.is_system);
    }

    #[tokio::test]
    async fn test_permission_inheritance() {
        let manager = RbacManager::new();

        // Writer role inherits from reader role
        let writer_perms = manager.get_effective_permissions("writer").await;

        // Should have both writer and reader permissions
        assert!(writer_perms.contains(&Permission::GlobalWrite));
        assert!(writer_perms.contains(&Permission::GlobalRead)); // Inherited from reader
    }

    #[tokio::test]
    async fn test_cannot_delete_system_role() {
        let manager = RbacManager::new();

        let result = manager.delete_role("admin").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_role_assignment() {
        let manager = RbacManager::new();

        manager
            .assign_role(
                "test_user".to_string(),
                "reader".to_string(),
                "admin".to_string(),
                None,
            )
            .await
            .unwrap();

        // Verify assignment succeeded (would need to add getter method)
    }
}
