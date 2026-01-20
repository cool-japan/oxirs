//! Access control and RBAC for multi-tenancy

use crate::multi_tenancy::types::{MultiTenancyError, MultiTenancyResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// Permission for an operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    /// Read vectors
    Read,
    /// Write/insert vectors
    Write,
    /// Delete vectors
    Delete,
    /// Build indices
    BuildIndex,
    /// Administer tenant
    Admin,
    /// View metrics and analytics
    ViewMetrics,
    /// Manage billing
    ManageBilling,
    /// Custom permission
    Custom(u32),
}

impl Permission {
    /// Check if this permission includes another
    pub fn includes(&self, other: &Permission) -> bool {
        match self {
            Self::Admin => true, // Admin has all permissions
            _ => self == other,
        }
    }

    /// Get permission name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Delete => "delete",
            Self::BuildIndex => "build_index",
            Self::Admin => "admin",
            Self::ViewMetrics => "view_metrics",
            Self::ManageBilling => "manage_billing",
            Self::Custom(_) => "custom",
        }
    }
}

/// Role with a set of permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// Role name
    pub name: String,
    /// Permissions granted by this role
    pub permissions: HashSet<Permission>,
    /// Role description
    pub description: Option<String>,
}

impl Role {
    /// Create new role
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            permissions: HashSet::new(),
            description: None,
        }
    }

    /// Create read-only role
    pub fn readonly() -> Self {
        let mut role = Self::new("readonly");
        role.permissions.insert(Permission::Read);
        role.permissions.insert(Permission::ViewMetrics);
        role.description = Some("Read-only access to vectors and metrics".to_string());
        role
    }

    /// Create read-write role
    pub fn readwrite() -> Self {
        let mut role = Self::new("readwrite");
        role.permissions.insert(Permission::Read);
        role.permissions.insert(Permission::Write);
        role.permissions.insert(Permission::ViewMetrics);
        role.description = Some("Read and write access to vectors".to_string());
        role
    }

    /// Create admin role
    pub fn admin() -> Self {
        let mut role = Self::new("admin");
        role.permissions.insert(Permission::Admin);
        role.description = Some("Full administrative access".to_string());
        role
    }

    /// Add permission to role
    pub fn add_permission(&mut self, permission: Permission) {
        self.permissions.insert(permission);
    }

    /// Check if role has permission
    pub fn has_permission(&self, permission: Permission) -> bool {
        self.permissions.iter().any(|p| p.includes(&permission))
    }
}

/// Access policy for a tenant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPolicy {
    /// Tenant ID
    pub tenant_id: String,
    /// User/API key to role mapping
    pub user_roles: HashMap<String, Vec<String>>,
    /// Available roles
    pub roles: HashMap<String, Role>,
    /// IP whitelist (empty = no restriction)
    pub ip_whitelist: Vec<String>,
    /// IP blacklist
    pub ip_blacklist: Vec<String>,
}

impl AccessPolicy {
    /// Create new access policy
    pub fn new(tenant_id: impl Into<String>) -> Self {
        let mut policy = Self {
            tenant_id: tenant_id.into(),
            user_roles: HashMap::new(),
            roles: HashMap::new(),
            ip_whitelist: Vec::new(),
            ip_blacklist: Vec::new(),
        };

        // Add default roles
        policy.add_role(Role::readonly());
        policy.add_role(Role::readwrite());
        policy.add_role(Role::admin());

        policy
    }

    /// Add a role definition
    pub fn add_role(&mut self, role: Role) {
        self.roles.insert(role.name.clone(), role);
    }

    /// Assign role to user
    pub fn assign_role(&mut self, user_id: impl Into<String>, role_name: impl Into<String>) {
        self.user_roles
            .entry(user_id.into())
            .or_default()
            .push(role_name.into());
    }

    /// Check if user has permission
    pub fn has_permission(&self, user_id: &str, permission: Permission) -> bool {
        if let Some(role_names) = self.user_roles.get(user_id) {
            for role_name in role_names {
                if let Some(role) = self.roles.get(role_name) {
                    if role.has_permission(permission) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Check if IP is allowed
    pub fn is_ip_allowed(&self, ip: &str) -> bool {
        // Check blacklist first
        if self.ip_blacklist.contains(&ip.to_string()) {
            return false;
        }

        // If whitelist is empty, allow all (except blacklisted)
        if self.ip_whitelist.is_empty() {
            return true;
        }

        // Check whitelist
        self.ip_whitelist.contains(&ip.to_string())
    }
}

/// Access control manager
pub struct AccessControl {
    /// Policies by tenant
    policies: Arc<RwLock<HashMap<String, AccessPolicy>>>,
}

impl AccessControl {
    /// Create new access control manager
    pub fn new() -> Self {
        Self {
            policies: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set policy for tenant
    pub fn set_policy(&self, policy: AccessPolicy) -> MultiTenancyResult<()> {
        self.policies
            .write()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?
            .insert(policy.tenant_id.clone(), policy);
        Ok(())
    }

    /// Get policy for tenant
    pub fn get_policy(&self, tenant_id: &str) -> MultiTenancyResult<AccessPolicy> {
        self.policies
            .read()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?
            .get(tenant_id)
            .cloned()
            .ok_or_else(|| MultiTenancyError::TenantNotFound {
                tenant_id: tenant_id.to_string(),
            })
    }

    /// Check if user has permission
    pub fn check_permission(
        &self,
        tenant_id: &str,
        user_id: &str,
        permission: Permission,
    ) -> MultiTenancyResult<bool> {
        let policy = self.get_policy(tenant_id)?;
        Ok(policy.has_permission(user_id, permission))
    }

    /// Authorize operation
    pub fn authorize(
        &self,
        tenant_id: &str,
        user_id: &str,
        permission: Permission,
        client_ip: Option<&str>,
    ) -> MultiTenancyResult<()> {
        let policy = self.get_policy(tenant_id)?;

        // Check IP restrictions
        if let Some(ip) = client_ip {
            if !policy.is_ip_allowed(ip) {
                return Err(MultiTenancyError::AccessDenied {
                    tenant_id: tenant_id.to_string(),
                    reason: format!("IP {} not allowed", ip),
                });
            }
        }

        // Check permissions
        if !policy.has_permission(user_id, permission) {
            return Err(MultiTenancyError::AccessDenied {
                tenant_id: tenant_id.to_string(),
                reason: format!("User {} lacks permission {:?}", user_id, permission),
            });
        }

        Ok(())
    }

    /// Create default policy for tenant
    pub fn create_default_policy(&self, tenant_id: impl Into<String>) -> MultiTenancyResult<()> {
        let policy = AccessPolicy::new(tenant_id);
        self.set_policy(policy)
    }
}

impl Default for AccessControl {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permissions() {
        assert!(Permission::Admin.includes(&Permission::Read));
        assert!(Permission::Admin.includes(&Permission::Write));
        assert!(!Permission::Read.includes(&Permission::Write));
        assert!(Permission::Read.includes(&Permission::Read));
    }

    #[test]
    fn test_role_creation() {
        let role = Role::readonly();
        assert!(role.has_permission(Permission::Read));
        assert!(!role.has_permission(Permission::Write));
        assert!(!role.has_permission(Permission::Delete));

        let role = Role::readwrite();
        assert!(role.has_permission(Permission::Read));
        assert!(role.has_permission(Permission::Write));
        assert!(!role.has_permission(Permission::Delete));

        let role = Role::admin();
        assert!(role.has_permission(Permission::Read));
        assert!(role.has_permission(Permission::Write));
        assert!(role.has_permission(Permission::Delete));
        assert!(role.has_permission(Permission::Admin));
    }

    #[test]
    fn test_access_policy() {
        let mut policy = AccessPolicy::new("tenant1");

        // Assign roles
        policy.assign_role("user1", "readonly");
        policy.assign_role("user2", "readwrite");
        policy.assign_role("user3", "admin");

        // Check permissions
        assert!(policy.has_permission("user1", Permission::Read));
        assert!(!policy.has_permission("user1", Permission::Write));

        assert!(policy.has_permission("user2", Permission::Read));
        assert!(policy.has_permission("user2", Permission::Write));
        assert!(!policy.has_permission("user2", Permission::Delete));

        assert!(policy.has_permission("user3", Permission::Read));
        assert!(policy.has_permission("user3", Permission::Write));
        assert!(policy.has_permission("user3", Permission::Delete));
        assert!(policy.has_permission("user3", Permission::Admin));
    }

    #[test]
    fn test_ip_restrictions() {
        let mut policy = AccessPolicy::new("tenant1");

        // No restrictions by default
        assert!(policy.is_ip_allowed("192.168.1.1"));
        assert!(policy.is_ip_allowed("10.0.0.1"));

        // Add to blacklist
        policy.ip_blacklist.push("192.168.1.100".to_string());
        assert!(!policy.is_ip_allowed("192.168.1.100"));
        assert!(policy.is_ip_allowed("192.168.1.1"));

        // Add whitelist (restricts to only those IPs)
        policy.ip_whitelist.push("192.168.1.1".to_string());
        policy.ip_whitelist.push("192.168.1.2".to_string());
        assert!(policy.is_ip_allowed("192.168.1.1"));
        assert!(policy.is_ip_allowed("192.168.1.2"));
        assert!(!policy.is_ip_allowed("10.0.0.1"));
        assert!(!policy.is_ip_allowed("192.168.1.100")); // Blacklist takes precedence
    }

    #[test]
    fn test_access_control_manager() {
        let ac = AccessControl::new();

        // Create default policy
        ac.create_default_policy("tenant1").unwrap();

        // Get policy and modify
        let mut policy = ac.get_policy("tenant1").unwrap();
        policy.assign_role("user1", "readonly");
        policy.assign_role("user2", "admin");
        ac.set_policy(policy).unwrap();

        // Check permissions
        assert!(ac
            .check_permission("tenant1", "user1", Permission::Read)
            .unwrap());
        assert!(!ac
            .check_permission("tenant1", "user1", Permission::Write)
            .unwrap());

        assert!(ac
            .check_permission("tenant1", "user2", Permission::Admin)
            .unwrap());

        // Authorize operations
        assert!(ac
            .authorize("tenant1", "user1", Permission::Read, None)
            .is_ok());
        assert!(ac
            .authorize("tenant1", "user1", Permission::Write, None)
            .is_err());
        assert!(ac
            .authorize("tenant1", "user2", Permission::Write, None)
            .is_ok());
    }

    #[test]
    fn test_authorize_with_ip() {
        let ac = AccessControl::new();
        let mut policy = AccessPolicy::new("tenant1");
        policy.assign_role("user1", "readonly");
        policy.ip_whitelist.push("192.168.1.1".to_string());
        ac.set_policy(policy).unwrap();

        // Should succeed with allowed IP
        assert!(ac
            .authorize("tenant1", "user1", Permission::Read, Some("192.168.1.1"))
            .is_ok());

        // Should fail with disallowed IP
        assert!(ac
            .authorize("tenant1", "user1", Permission::Read, Some("10.0.0.1"))
            .is_err());
    }

    #[test]
    fn test_custom_roles() {
        let mut role = Role::new("custom");
        role.add_permission(Permission::Read);
        role.add_permission(Permission::ViewMetrics);
        role.add_permission(Permission::Custom(100));

        assert!(role.has_permission(Permission::Read));
        assert!(role.has_permission(Permission::ViewMetrics));
        assert!(role.has_permission(Permission::Custom(100)));
        assert!(!role.has_permission(Permission::Write));
    }
}
