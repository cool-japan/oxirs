//! Permission system and role-based access control

use crate::auth::types::{Permission, User};
use crate::config::UserConfig;
use std::collections::HashSet;

/// Permission checker for validating user access
pub struct PermissionChecker;

impl PermissionChecker {
    /// Check if user has specific permission
    pub fn has_permission(user: &User, permission: &Permission) -> bool {
        // Check direct permissions
        if user.permissions.contains(permission) {
            return true;
        }

        // Check role-based permissions
        Self::check_role_permissions(user, permission)
    }

    /// Check role-based permissions
    fn check_role_permissions(user: &User, permission: &Permission) -> bool {
        for role in &user.roles {
            if let Some(role_permissions) = Self::get_role_permissions(role) {
                if role_permissions.contains(permission) {
                    return true;
                }
            }
        }
        false
    }

    /// Get permissions for a specific role
    fn get_role_permissions(role: &str) -> Option<HashSet<Permission>> {
        let mut permissions = HashSet::new();

        match role {
            "admin" => {
                permissions.extend(vec![
                    Permission::Read,
                    Permission::Write,
                    Permission::Admin,
                    Permission::DatasetCreate,
                    Permission::DatasetDelete,
                    Permission::DatasetManage,
                    Permission::UserManage,
                    Permission::SystemConfig,
                    Permission::QueryExecute,
                    Permission::UpdateExecute,
                    Permission::GraphStore,
                    Permission::Upload,
                    Permission::Download,
                    Permission::Backup,
                    Permission::Restore,
                    Permission::Monitor,
                    Permission::Audit,
                    Permission::ServiceManage,
                    Permission::ClusterManage,
                    Permission::FederationManage,
                ]);
            }
            "user" => {
                permissions.extend(vec![
                    Permission::Read,
                    Permission::QueryExecute,
                    Permission::Download,
                ]);
            }
            "writer" => {
                permissions.extend(vec![
                    Permission::Read,
                    Permission::Write,
                    Permission::QueryExecute,
                    Permission::UpdateExecute,
                    Permission::GraphStore,
                    Permission::Upload,
                    Permission::Download,
                ]);
            }
            "monitor" => {
                permissions.extend(vec![
                    Permission::Read,
                    Permission::Monitor,
                    Permission::QueryExecute,
                ]);
            }
            "dataset_admin" => {
                permissions.extend(vec![
                    Permission::Read,
                    Permission::Write,
                    Permission::DatasetCreate,
                    Permission::DatasetDelete,
                    Permission::DatasetManage,
                    Permission::QueryExecute,
                    Permission::UpdateExecute,
                    Permission::GraphStore,
                    Permission::Upload,
                    Permission::Download,
                    Permission::Backup,
                    Permission::Restore,
                ]);
            }
            _ => return None,
        }

        Some(permissions)
    }

    /// Compute user permissions based on roles and explicit permissions
    pub fn compute_user_permissions(user_config: &UserConfig) -> Vec<Permission> {
        let mut permissions = HashSet::new();

        // Add explicit permissions
        permissions.extend(user_config.permissions.iter().cloned());

        // Add role-based permissions
        for role in &user_config.roles {
            if let Some(role_permissions) = Self::get_role_permissions(role) {
                permissions.extend(role_permissions);
            }
        }

        permissions.into_iter().collect()
    }

    /// Check if user can access a specific dataset
    pub fn can_access_dataset(
        user: &User,
        dataset: &str,
        required_permission: &Permission,
    ) -> bool {
        // Check for dataset-specific permissions
        match required_permission {
            Permission::Read => {
                Self::has_permission(user, &Permission::Read)
                    || Self::has_permission(user, &Permission::Admin)
            }
            Permission::Write => {
                Self::has_permission(user, &Permission::Write)
                    || Self::has_permission(user, &Permission::Admin)
            }
            Permission::Admin => Self::has_permission(user, &Permission::Admin),
            _ => Self::has_permission(user, required_permission),
        }
    }

    /// Get all available permissions for display
    pub fn get_all_permissions() -> Vec<Permission> {
        vec![
            Permission::Read,
            Permission::Write,
            Permission::Admin,
            Permission::DatasetCreate,
            Permission::DatasetDelete,
            Permission::DatasetManage,
            Permission::UserManage,
            Permission::SystemConfig,
            Permission::QueryExecute,
            Permission::UpdateExecute,
            Permission::GraphStore,
            Permission::Upload,
            Permission::Download,
            Permission::Backup,
            Permission::Restore,
            Permission::Monitor,
            Permission::Audit,
            Permission::ServiceManage,
            Permission::ClusterManage,
            Permission::FederationManage,
        ]
    }

    /// Get all available roles
    pub fn get_all_roles() -> Vec<&'static str> {
        vec!["admin", "user", "writer", "monitor", "dataset_admin"]
    }

    /// Validate role name
    pub fn is_valid_role(role: &str) -> bool {
        Self::get_all_roles().contains(&role)
    }

    /// Get role description
    pub fn get_role_description(role: &str) -> Option<&'static str> {
        match role {
            "admin" => Some("Full system administrator with all permissions"),
            "user" => Some("Read-only user with query access"),
            "writer" => Some("User with read/write access to datasets"),
            "monitor" => Some("User with monitoring and read access"),
            "dataset_admin" => Some("Administrator for dataset management"),
            _ => None,
        }
    }
}
