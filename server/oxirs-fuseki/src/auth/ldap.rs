//! LDAP authentication integration for OxiRS Fuseki
//!
//! This module provides LDAP/Active Directory authentication support,
//! enabling enterprise directory integration.

use crate::{
    auth::{AuthResult, Permission, User},
    config::LdapConfig,
    error::{FusekiError, FusekiResult},
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// LDAP authentication service
#[derive(Clone)]
pub struct LdapService {
    config: Arc<LdapConfig>,
    connection_pool: Arc<RwLock<LdapConnectionPool>>,
    user_cache: Arc<RwLock<HashMap<String, CachedUser>>>,
}

/// LDAP connection pool for managing connections
#[derive(Debug)]
pub struct LdapConnectionPool {
    connections: Vec<LdapConnection>,
    max_connections: usize,
    active_connections: usize,
}

/// Individual LDAP connection
#[derive(Debug, Clone)]
pub struct LdapConnection {
    pub id: String,
    pub server_url: String,
    pub is_connected: bool,
    pub last_used: DateTime<Utc>,
    pub use_tls: bool,
}

/// Cached LDAP user information
#[derive(Debug, Clone)]
pub struct CachedUser {
    pub user: User,
    pub cached_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub ldap_dn: String,
}

/// LDAP user attributes from directory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LdapUserAttributes {
    pub dn: String,
    pub cn: Option<String>,
    pub sn: Option<String>,
    pub given_name: Option<String>,
    pub mail: Option<String>,
    pub uid: Option<String>,
    pub sam_account_name: Option<String>,
    pub display_name: Option<String>,
    pub department: Option<String>,
    pub title: Option<String>,
    pub telephone_number: Option<String>,
    pub member_of: Vec<String>,
    pub object_class: Vec<String>,
}

/// LDAP search result
#[derive(Debug, Clone)]
pub struct LdapSearchResult {
    pub dn: String,
    pub attributes: HashMap<String, Vec<String>>,
}

/// LDAP group information
#[derive(Debug, Clone)]
pub struct LdapGroup {
    pub dn: String,
    pub cn: String,
    pub description: Option<String>,
    pub members: Vec<String>,
}

/// LDAP authentication request
#[derive(Debug, Deserialize)]
pub struct LdapAuthRequest {
    pub username: String,
    pub password: String,
    pub domain: Option<String>,
}

/// LDAP search parameters
#[derive(Debug, Clone)]
pub struct LdapSearchParams {
    pub base_dn: String,
    pub scope: LdapScope,
    pub filter: String,
    pub attributes: Vec<String>,
    pub size_limit: Option<u32>,
    pub time_limit: Option<u32>,
}

/// LDAP search scope
#[derive(Debug, Clone)]
pub enum LdapScope {
    Base,
    OneLevel,
    Subtree,
}

impl LdapService {
    /// Create new LDAP service
    pub async fn new(config: LdapConfig) -> Result<Self, FusekiError> {
        let connection_pool = LdapConnectionPool::new(10); // Default max 10 connections

        let service = LdapService {
            config: Arc::new(config),
            connection_pool: Arc::new(RwLock::new(connection_pool)),
            user_cache: Arc::new(RwLock::new(HashMap::new())),
        };

        Ok(service)
    }

    /// Authenticate user against LDAP directory
    pub async fn authenticate_ldap_user(
        &self,
        username: &str,
        password: &str,
    ) -> FusekiResult<AuthResult> {
        debug!("Attempting LDAP authentication for user: {}", username);

        // Check cache first
        if let Some(cached_user) = self.get_cached_user(username).await {
            if self
                .verify_cached_user_password(&cached_user, password)
                .await?
            {
                info!(
                    "LDAP authentication successful from cache for user: {}",
                    username
                );
                return Ok(AuthResult::Authenticated(cached_user.user));
            }
        }

        // Perform LDAP bind authentication
        let user_dn = self.find_user_dn(username).await?;

        if self.bind_user(&user_dn, password).await? {
            // Authentication successful, get user attributes
            let user_attributes = self.get_user_attributes(&user_dn).await?;
            let user = self.map_ldap_user_to_internal(user_attributes).await?;

            // Cache the user
            self.cache_user(username, user.clone(), user_dn).await;

            info!("LDAP authentication successful for user: {}", username);
            Ok(AuthResult::Authenticated(user))
        } else {
            warn!("LDAP authentication failed for user: {}", username);
            Ok(AuthResult::Unauthenticated)
        }
    }

    /// Find user DN in LDAP directory
    async fn find_user_dn(&self, username: &str) -> FusekiResult<String> {
        debug!("Searching for user DN: {}", username);

        // Construct search filter
        let filter = self.config.user_filter.replace("{username}", username);

        let search_params = LdapSearchParams {
            base_dn: self.config.user_base_dn.clone(),
            scope: LdapScope::Subtree,
            filter,
            attributes: vec!["dn".to_string()],
            size_limit: Some(1),
            time_limit: Some(30),
        };

        let search_results = self.search(&search_params).await?;

        if search_results.is_empty() {
            return Err(FusekiError::authentication(format!(
                "User not found: {}",
                username
            )));
        }

        Ok(search_results[0].dn.clone())
    }

    /// Bind user to LDAP (authenticate)
    async fn bind_user(&self, user_dn: &str, password: &str) -> FusekiResult<bool> {
        debug!("Binding user: {}", user_dn);

        // This is a simplified implementation
        // In a real implementation, this would use an LDAP client library
        // like ldap3 or rust-ldap to perform the actual bind operation

        // Simulate LDAP bind
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Simple validation - in production this would be a real LDAP bind
        if password.is_empty() || user_dn.is_empty() {
            return Ok(false);
        }

        // Mock successful bind for demonstration
        Ok(true)
    }

    /// Get user attributes from LDAP
    async fn get_user_attributes(&self, user_dn: &str) -> FusekiResult<LdapUserAttributes> {
        debug!("Getting user attributes for: {}", user_dn);

        let search_params = LdapSearchParams {
            base_dn: user_dn.to_string(),
            scope: LdapScope::Base,
            filter: "(objectClass=*)".to_string(),
            attributes: vec![
                "cn".to_string(),
                "sn".to_string(),
                "givenName".to_string(),
                "mail".to_string(),
                "uid".to_string(),
                "sAMAccountName".to_string(),
                "displayName".to_string(),
                "department".to_string(),
                "title".to_string(),
                "telephoneNumber".to_string(),
                "memberOf".to_string(),
                "objectClass".to_string(),
            ],
            size_limit: Some(1),
            time_limit: Some(30),
        };

        let search_results = self.search(&search_params).await?;

        if search_results.is_empty() {
            return Err(FusekiError::authentication(
                "Failed to retrieve user attributes",
            ));
        }

        let result = &search_results[0];
        let attrs = &result.attributes;

        let user_attributes = LdapUserAttributes {
            dn: result.dn.clone(),
            cn: get_first_attribute(attrs, "cn"),
            sn: get_first_attribute(attrs, "sn"),
            given_name: get_first_attribute(attrs, "givenName"),
            mail: get_first_attribute(attrs, "mail"),
            uid: get_first_attribute(attrs, "uid"),
            sam_account_name: get_first_attribute(attrs, "sAMAccountName"),
            display_name: get_first_attribute(attrs, "displayName"),
            department: get_first_attribute(attrs, "department"),
            title: get_first_attribute(attrs, "title"),
            telephone_number: get_first_attribute(attrs, "telephoneNumber"),
            member_of: attrs.get("memberOf").cloned().unwrap_or_default(),
            object_class: attrs.get("objectClass").cloned().unwrap_or_default(),
        };

        Ok(user_attributes)
    }

    /// Search LDAP directory
    async fn search(&self, params: &LdapSearchParams) -> FusekiResult<Vec<LdapSearchResult>> {
        debug!(
            "LDAP search: base={}, filter={}",
            params.base_dn, params.filter
        );

        // This is a simplified mock implementation
        // In a real implementation, this would use an LDAP client library

        // Simulate search delay
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Mock search results
        let mock_result = LdapSearchResult {
            dn: format!("cn=testuser,{}", self.config.user_base_dn),
            attributes: {
                let mut attrs = HashMap::new();
                attrs.insert("cn".to_string(), vec!["testuser".to_string()]);
                attrs.insert("mail".to_string(), vec!["testuser@example.com".to_string()]);
                attrs.insert("displayName".to_string(), vec!["Test User".to_string()]);
                attrs.insert(
                    "memberOf".to_string(),
                    vec![
                        format!("cn=users,{}", self.config.group_base_dn),
                        format!("cn=developers,{}", self.config.group_base_dn),
                    ],
                );
                attrs
            },
        };

        Ok(vec![mock_result])
    }

    /// Map LDAP user to internal user structure
    async fn map_ldap_user_to_internal(&self, ldap_user: LdapUserAttributes) -> FusekiResult<User> {
        // Extract username (prefer sAMAccountName for AD, uid for OpenLDAP)
        let username = ldap_user
            .sam_account_name
            .or(ldap_user.uid)
            .or(ldap_user.cn)
            .ok_or_else(|| FusekiError::authentication("No suitable username attribute found"))?;

        // Extract display name
        let full_name =
            ldap_user
                .display_name
                .or_else(|| match (&ldap_user.given_name, &ldap_user.sn) {
                    (Some(given), Some(surname)) => Some(format!("{} {}", given, surname)),
                    (Some(given), None) => Some(given.clone()),
                    (None, Some(surname)) => Some(surname.clone()),
                    _ => ldap_user.cn.clone(),
                });

        // Map LDAP groups to roles
        let roles = self.map_ldap_groups_to_roles(&ldap_user.member_of).await;

        // Compute permissions based on roles
        let permissions = self.compute_permissions_for_roles(&roles).await;

        let user = User {
            username,
            roles,
            email: ldap_user.mail,
            full_name,
            last_login: Some(Utc::now()),
            permissions,
        };

        Ok(user)
    }

    /// Map LDAP groups to internal roles
    async fn map_ldap_groups_to_roles(&self, member_of: &[String]) -> Vec<String> {
        let mut roles = Vec::new();

        for group_dn in member_of {
            // Extract group CN from DN
            if let Some(cn) = extract_cn_from_dn(group_dn) {
                let role = self.map_ldap_group_to_role(&cn);
                if !role.is_empty() && !roles.contains(&role) {
                    roles.push(role);
                }
            }
        }

        // Default role if none found
        if roles.is_empty() {
            roles.push("user".to_string());
        }

        roles
    }

    /// Map LDAP group to internal role
    fn map_ldap_group_to_role(&self, group_cn: &str) -> String {
        // Configurable mapping - in production this would be configurable
        match group_cn.to_lowercase().as_str() {
            "domain admins" | "administrators" | "fuseki-admins" => "admin".to_string(),
            "developers" | "fuseki-writers" => "writer".to_string(),
            "users" | "fuseki-readers" => "reader".to_string(),
            "everyone" | "domain users" => "user".to_string(),
            _ => {
                // Check for custom dataset roles
                if group_cn.starts_with("fuseki-") {
                    group_cn.to_lowercase()
                } else {
                    "user".to_string()
                }
            }
        }
    }

    /// Compute permissions for roles
    async fn compute_permissions_for_roles(&self, roles: &[String]) -> Vec<Permission> {
        let mut permissions = Vec::new();

        for role in roles {
            match role.as_str() {
                "admin" => {
                    permissions.extend(vec![
                        Permission::GlobalAdmin,
                        Permission::GlobalRead,
                        Permission::GlobalWrite,
                        Permission::SparqlQuery,
                        Permission::SparqlUpdate,
                        Permission::GraphStore,
                        Permission::UserManagement,
                        Permission::SystemConfig,
                        Permission::SystemMetrics,
                    ]);
                }
                "writer" => {
                    permissions.extend(vec![
                        Permission::GlobalRead,
                        Permission::GlobalWrite,
                        Permission::SparqlQuery,
                        Permission::SparqlUpdate,
                        Permission::GraphStore,
                    ]);
                }
                "reader" => {
                    permissions.extend(vec![Permission::GlobalRead, Permission::SparqlQuery]);
                }
                "user" => {
                    permissions.extend(vec![Permission::GlobalRead, Permission::SparqlQuery]);
                }
                _ => {}
            }
        }

        // Remove duplicates
        permissions.sort();
        permissions.dedup();

        permissions
    }

    /// Cache user information
    async fn cache_user(&self, username: &str, user: User, ldap_dn: String) {
        let cached_user = CachedUser {
            user,
            cached_at: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::minutes(30), // 30 min cache
            ldap_dn,
        };

        let mut cache = self.user_cache.write().await;
        cache.insert(username.to_string(), cached_user);
    }

    /// Get cached user
    async fn get_cached_user(&self, username: &str) -> Option<CachedUser> {
        let cache = self.user_cache.read().await;

        if let Some(cached_user) = cache.get(username) {
            if Utc::now() < cached_user.expires_at {
                return Some(cached_user.clone());
            }
        }

        None
    }

    /// Verify cached user password (simplified)
    async fn verify_cached_user_password(
        &self,
        cached_user: &CachedUser,
        password: &str,
    ) -> FusekiResult<bool> {
        // In a real implementation, this could either:
        // 1. Re-bind to LDAP to verify password (recommended)
        // 2. Use cached password hash (less secure)
        // 3. Skip password verification for cached users (least secure)

        // For this implementation, we'll re-bind to LDAP
        self.bind_user(&cached_user.ldap_dn, password).await
    }

    /// Cleanup expired cache entries
    pub async fn cleanup_expired_cache(&self) {
        let mut cache = self.user_cache.write().await;
        let now = Utc::now();

        cache.retain(|_, cached_user| cached_user.expires_at > now);
    }

    /// Test LDAP connection
    pub async fn test_connection(&self) -> FusekiResult<bool> {
        debug!("Testing LDAP connection to: {}", self.config.server);

        // Attempt to bind with service account
        let bind_result = self
            .bind_user(&self.config.bind_dn, &self.config.bind_password)
            .await?;

        if bind_result {
            info!("LDAP connection test successful");
            Ok(true)
        } else {
            warn!("LDAP connection test failed");
            Ok(false)
        }
    }

    /// Get user groups from LDAP
    pub async fn get_user_groups(&self, username: &str) -> FusekiResult<Vec<LdapGroup>> {
        let user_dn = self.find_user_dn(username).await?;
        let user_attributes = self.get_user_attributes(&user_dn).await?;

        let mut groups = Vec::new();

        for group_dn in &user_attributes.member_of {
            if let Ok(group) = self.get_group_info(group_dn).await {
                groups.push(group);
            }
        }

        Ok(groups)
    }

    /// Get group information
    async fn get_group_info(&self, group_dn: &str) -> FusekiResult<LdapGroup> {
        let search_params = LdapSearchParams {
            base_dn: group_dn.to_string(),
            scope: LdapScope::Base,
            filter: "(objectClass=*)".to_string(),
            attributes: vec![
                "cn".to_string(),
                "description".to_string(),
                "member".to_string(),
            ],
            size_limit: Some(1),
            time_limit: Some(30),
        };

        let search_results = self.search(&search_params).await?;

        if search_results.is_empty() {
            return Err(FusekiError::authentication("Group not found"));
        }

        let result = &search_results[0];
        let attrs = &result.attributes;

        let group = LdapGroup {
            dn: result.dn.clone(),
            cn: get_first_attribute(attrs, "cn").unwrap_or_default(),
            description: get_first_attribute(attrs, "description"),
            members: attrs.get("member").cloned().unwrap_or_default(),
        };

        Ok(group)
    }
}

impl LdapConnectionPool {
    /// Create new connection pool
    pub fn new(max_connections: usize) -> Self {
        LdapConnectionPool {
            connections: Vec::new(),
            max_connections,
            active_connections: 0,
        }
    }

    /// Get available connection
    pub async fn get_connection(&mut self) -> Option<LdapConnection> {
        // Find available connection
        for connection in &mut self.connections {
            if !connection.is_connected
                && Utc::now() - connection.last_used < chrono::Duration::minutes(5)
            {
                connection.is_connected = true;
                connection.last_used = Utc::now();
                return Some(connection.clone());
            }
        }

        // Create new connection if under limit
        if self.active_connections < self.max_connections {
            let connection = LdapConnection {
                id: uuid::Uuid::new_v4().to_string(),
                server_url: "ldap://localhost:389".to_string(), // Would use actual config
                is_connected: true,
                last_used: Utc::now(),
                use_tls: false,
            };

            self.connections.push(connection.clone());
            self.active_connections += 1;

            Some(connection)
        } else {
            None
        }
    }

    /// Return connection to pool
    pub async fn return_connection(&mut self, mut connection: LdapConnection) {
        connection.is_connected = false;
        connection.last_used = Utc::now();

        // Update connection in pool
        for pool_conn in &mut self.connections {
            if pool_conn.id == connection.id {
                *pool_conn = connection;
                break;
            }
        }
    }
}

/// Extract CN from LDAP DN
fn extract_cn_from_dn(dn: &str) -> Option<String> {
    for component in dn.split(',') {
        let component = component.trim();
        if component.starts_with("cn=") || component.starts_with("CN=") {
            return Some(component[3..].to_string());
        }
    }
    None
}

/// Get first attribute value from LDAP attributes
fn get_first_attribute(attrs: &HashMap<String, Vec<String>>, attr_name: &str) -> Option<String> {
    attrs
        .get(attr_name)
        .and_then(|values| values.first())
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LdapConfig;

    fn create_test_ldap_config() -> LdapConfig {
        LdapConfig {
            server: "ldap://localhost:389".to_string(),
            bind_dn: "cn=admin,dc=example,dc=com".to_string(),
            bind_password: "admin_password".to_string(),
            user_base_dn: "ou=users,dc=example,dc=com".to_string(),
            user_filter: "(&(objectClass=person)(uid={username}))".to_string(),
            group_base_dn: "ou=groups,dc=example,dc=com".to_string(),
            group_filter: "(&(objectClass=groupOfNames)(member={userdn}))".to_string(),
            use_tls: false,
        }
    }

    #[tokio::test]
    async fn test_ldap_service_creation() {
        let config = create_test_ldap_config();
        let service = LdapService::new(config);

        assert_eq!(service.config.server, "ldap://localhost:389");
        assert_eq!(service.config.user_base_dn, "ou=users,dc=example,dc=com");
    }

    #[test]
    fn test_cn_extraction() {
        assert_eq!(
            extract_cn_from_dn("cn=users,ou=groups,dc=example,dc=com"),
            Some("users".to_string())
        );

        assert_eq!(
            extract_cn_from_dn("CN=Administrators,OU=Groups,DC=example,DC=com"),
            Some("Administrators".to_string())
        );

        assert_eq!(extract_cn_from_dn("ou=users,dc=example,dc=com"), None);
    }

    #[tokio::test]
    async fn test_group_mapping() {
        let config = create_test_ldap_config();
        let service = LdapService::new(config);

        let groups = vec![
            "cn=administrators,ou=groups,dc=example,dc=com".to_string(),
            "cn=users,ou=groups,dc=example,dc=com".to_string(),
        ];

        let roles = service.map_ldap_groups_to_roles(&groups).await;
        assert!(roles.contains(&"admin".to_string()));
        assert!(roles.contains(&"reader".to_string()));
    }

    #[tokio::test]
    async fn test_user_caching() {
        let config = create_test_ldap_config();
        let service = LdapService::new(config);

        let user = User {
            username: "testuser".to_string(),
            roles: vec!["user".to_string()],
            email: Some("test@example.com".to_string()),
            full_name: Some("Test User".to_string()),
            last_login: Some(Utc::now()),
            permissions: vec![Permission::GlobalRead],
        };

        service
            .cache_user(
                "testuser",
                user.clone(),
                "cn=testuser,ou=users,dc=example,dc=com".to_string(),
            )
            .await;

        let cached = service.get_cached_user("testuser").await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().user.username, "testuser");
    }

    #[tokio::test]
    async fn test_connection_pool() {
        let mut pool = LdapConnectionPool::new(2);

        let conn1 = pool.get_connection().await;
        assert!(conn1.is_some());

        let conn2 = pool.get_connection().await;
        assert!(conn2.is_some());

        let conn3 = pool.get_connection().await;
        assert!(conn3.is_none()); // Pool limit reached

        // Return connection
        if let Some(conn) = conn1 {
            pool.return_connection(conn).await;
        }

        let conn4 = pool.get_connection().await;
        assert!(conn4.is_some()); // Should get returned connection
    }
}
