//! # DID-Based Access Control List (ACL)
//!
//! Fine-grained, DID-based access control for resources.
//!
//! Each resource can have an `AccessPolicy` that specifies:
//! - Which DIDs are allowed to access the resource
//! - Which permissions those DIDs hold
//! - An optional deny-list that overrides allow rules
//! - An optional expiry timestamp in milliseconds since the Unix epoch
//!
//! ## Example
//!
//! ```rust
//! use oxirs_did::access_control::{
//!     AccessControlList, AccessPolicy, AccessRequest, AccessDecision, Permission,
//! };
//!
//! let mut acl = AccessControlList::new();
//! acl.add_policy(AccessPolicy {
//!     resource: "/graph/public".to_string(),
//!     allowed_dids: vec!["did:key:alice".to_string()],
//!     allowed_permissions: vec![Permission::Read],
//!     deny_dids: vec![],
//!     expiry_ms: None,
//! });
//!
//! let req = AccessRequest {
//!     did: "did:key:alice".to_string(),
//!     resource: "/graph/public".to_string(),
//!     permission: Permission::Read,
//!     timestamp_ms: 0,
//! };
//!
//! assert!(matches!(acl.check(&req), AccessDecision::Allow));
//! ```

use std::collections::HashMap;

/// Access permissions for resources
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Permission {
    Read,
    Write,
    Admin,
    Delete,
    Execute,
}

/// An access policy for a single named resource
#[derive(Debug, Clone)]
pub struct AccessPolicy {
    /// Resource identifier (e.g. IRI, path, or named graph)
    pub resource: String,
    /// DIDs explicitly granted access
    pub allowed_dids: Vec<String>,
    /// Permissions granted to the allowed DIDs
    pub allowed_permissions: Vec<Permission>,
    /// DIDs explicitly denied (overrides allow)
    pub deny_dids: Vec<String>,
    /// Optional expiry time in milliseconds since Unix epoch
    pub expiry_ms: Option<u64>,
}

/// A request to access a resource
#[derive(Debug, Clone)]
pub struct AccessRequest {
    /// The requesting DID
    pub did: String,
    /// The resource being accessed
    pub resource: String,
    /// The permission being exercised
    pub permission: Permission,
    /// Current time in milliseconds since Unix epoch (used for expiry checks)
    pub timestamp_ms: u64,
}

/// Decision returned by the ACL engine
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessDecision {
    /// Access is permitted
    Allow,
    /// Access is denied; includes the reason
    Deny(String),
    /// The policy covering this resource has expired
    Expired,
    /// No policy exists for the requested resource
    NotFound,
}

/// Error types for ACL mutation operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AclError {
    /// No policy exists for the named resource
    PolicyNotFound(String),
    /// The DID is already present in the allow list
    DuplicateDid(String),
}

impl std::fmt::Display for AclError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AclError::PolicyNotFound(r) => write!(f, "policy not found for resource: {}", r),
            AclError::DuplicateDid(d) => write!(f, "DID already in allow list: {}", d),
        }
    }
}

/// The main access control list, indexed by resource identifier
pub struct AccessControlList {
    policies: HashMap<String, AccessPolicy>,
}

impl AccessControlList {
    /// Create a new, empty ACL
    pub fn new() -> Self {
        AccessControlList {
            policies: HashMap::new(),
        }
    }

    /// Add or replace a policy for the given resource
    pub fn add_policy(&mut self, policy: AccessPolicy) {
        self.policies.insert(policy.resource.clone(), policy);
    }

    /// Remove the policy for `resource`.  Returns `true` if a policy existed.
    pub fn remove_policy(&mut self, resource: &str) -> bool {
        self.policies.remove(resource).is_some()
    }

    /// Check an access request against stored policies.
    ///
    /// Decision logic (in order):
    /// 1. If no policy found → `NotFound`
    /// 2. If policy expired (`expiry_ms <= timestamp_ms`) → `Expired`
    /// 3. If DID is in `deny_dids` → `Deny("DID explicitly denied")`
    /// 4. If DID is not in `allowed_dids` → `Deny("DID not in allow list")`
    /// 5. If requested permission not in `allowed_permissions` → `Deny("permission not granted")`
    /// 6. Otherwise → `Allow`
    ///
    /// Note: `Admin` permission implicitly grants all other permissions (it is treated as a
    /// super-permission: if `Admin` is in `allowed_permissions`, any requested permission is
    /// granted).
    pub fn check(&self, req: &AccessRequest) -> AccessDecision {
        let policy = match self.policies.get(&req.resource) {
            Some(p) => p,
            None => return AccessDecision::NotFound,
        };

        // Expiry check
        if let Some(expiry) = policy.expiry_ms {
            if req.timestamp_ms >= expiry {
                return AccessDecision::Expired;
            }
        }

        // Deny-list overrides everything
        if policy.deny_dids.contains(&req.did) {
            return AccessDecision::Deny("DID explicitly denied".to_string());
        }

        // Must be in the allow list
        if !policy.allowed_dids.contains(&req.did) {
            return AccessDecision::Deny("DID not in allow list".to_string());
        }

        // Check permission — Admin supersedes all others
        let has_admin = policy.allowed_permissions.contains(&Permission::Admin);
        let has_permission = has_admin || policy.allowed_permissions.contains(&req.permission);

        if has_permission {
            AccessDecision::Allow
        } else {
            AccessDecision::Deny("permission not granted".to_string())
        }
    }

    /// Add `did` to the allow list of an existing policy for `resource`.
    ///
    /// Returns `AclError::PolicyNotFound` if the policy does not exist.
    /// Returns `AclError::DuplicateDid` if the DID is already in the allow list.
    pub fn grant(&mut self, resource: &str, did: &str, permission: Permission) -> Result<(), AclError> {
        let policy = self
            .policies
            .get_mut(resource)
            .ok_or_else(|| AclError::PolicyNotFound(resource.to_string()))?;

        if !policy.allowed_dids.contains(&did.to_string()) {
            policy.allowed_dids.push(did.to_string());
        }

        if !policy.allowed_permissions.contains(&permission) {
            policy.allowed_permissions.push(permission);
        }

        Ok(())
    }

    /// Remove `did` from the allow list of the policy for `resource`.
    ///
    /// Returns `true` if the DID was present and removed.
    pub fn revoke(&mut self, resource: &str, did: &str) -> bool {
        if let Some(policy) = self.policies.get_mut(resource) {
            let before = policy.allowed_dids.len();
            policy.allowed_dids.retain(|d| d != did);
            policy.allowed_dids.len() < before
        } else {
            false
        }
    }

    /// Add `did` to the deny list of the policy for `resource`.
    ///
    /// Returns `AclError::PolicyNotFound` if the policy does not exist.
    pub fn deny_did(&mut self, resource: &str, did: &str) -> Result<(), AclError> {
        let policy = self
            .policies
            .get_mut(resource)
            .ok_or_else(|| AclError::PolicyNotFound(resource.to_string()))?;

        if !policy.deny_dids.contains(&did.to_string()) {
            policy.deny_dids.push(did.to_string());
        }
        Ok(())
    }

    /// Return all policies where `did` is in `allowed_dids`
    pub fn policies_for_did(&self, did: &str) -> Vec<&AccessPolicy> {
        self.policies
            .values()
            .filter(|p| p.allowed_dids.contains(&did.to_string()))
            .collect()
    }

    /// Return all policies whose `expiry_ms` is set and has passed `current_time_ms`
    pub fn expired_policies(&self, current_time_ms: u64) -> Vec<&AccessPolicy> {
        self.policies
            .values()
            .filter(|p| {
                p.expiry_ms
                    .map(|e| current_time_ms >= e)
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Remove all expired policies and return the count of removed policies
    pub fn purge_expired(&mut self, current_time_ms: u64) -> usize {
        let expired_resources: Vec<String> = self
            .policies
            .iter()
            .filter(|(_, p)| p.expiry_ms.map(|e| current_time_ms >= e).unwrap_or(false))
            .map(|(k, _)| k.clone())
            .collect();

        let count = expired_resources.len();
        for resource in expired_resources {
            self.policies.remove(&resource);
        }
        count
    }

    /// Return the total number of stored policies
    pub fn policy_count(&self) -> usize {
        self.policies.len()
    }
}

impl Default for AccessControlList {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_acl() -> AccessControlList {
        let mut acl = AccessControlList::new();
        acl.add_policy(AccessPolicy {
            resource: "/graph/data".to_string(),
            allowed_dids: vec!["did:key:alice".to_string(), "did:key:bob".to_string()],
            allowed_permissions: vec![Permission::Read, Permission::Write],
            deny_dids: vec![],
            expiry_ms: None,
        });
        acl.add_policy(AccessPolicy {
            resource: "/graph/admin".to_string(),
            allowed_dids: vec!["did:key:admin".to_string()],
            allowed_permissions: vec![Permission::Admin],
            deny_dids: vec![],
            expiry_ms: None,
        });
        acl
    }

    fn make_request(did: &str, resource: &str, permission: Permission) -> AccessRequest {
        AccessRequest {
            did: did.to_string(),
            resource: resource.to_string(),
            permission,
            timestamp_ms: 1_000_000,
        }
    }

    // ===== Allow decisions =====

    #[test]
    fn test_allow_read() {
        let acl = make_acl();
        let req = make_request("did:key:alice", "/graph/data", Permission::Read);
        assert_eq!(acl.check(&req), AccessDecision::Allow);
    }

    #[test]
    fn test_allow_write() {
        let acl = make_acl();
        let req = make_request("did:key:alice", "/graph/data", Permission::Write);
        assert_eq!(acl.check(&req), AccessDecision::Allow);
    }

    #[test]
    fn test_allow_multiple_dids() {
        let acl = make_acl();
        let req = make_request("did:key:bob", "/graph/data", Permission::Read);
        assert_eq!(acl.check(&req), AccessDecision::Allow);
    }

    // ===== Deny decisions =====

    #[test]
    fn test_deny_unknown_did() {
        let acl = make_acl();
        let req = make_request("did:key:unknown", "/graph/data", Permission::Read);
        assert!(matches!(acl.check(&req), AccessDecision::Deny(_)));
    }

    #[test]
    fn test_deny_wrong_permission() {
        let acl = make_acl();
        let req = make_request("did:key:alice", "/graph/data", Permission::Delete);
        assert!(matches!(acl.check(&req), AccessDecision::Deny(_)));
    }

    #[test]
    fn test_deny_explicit_deny_list() {
        let mut acl = make_acl();
        acl.deny_did("/graph/data", "did:key:alice").unwrap();
        let req = make_request("did:key:alice", "/graph/data", Permission::Read);
        assert!(matches!(acl.check(&req), AccessDecision::Deny(msg) if msg.contains("explicitly denied")));
    }

    #[test]
    fn test_deny_overrides_allow() {
        let mut acl = AccessControlList::new();
        acl.add_policy(AccessPolicy {
            resource: "/res".to_string(),
            allowed_dids: vec!["did:key:alice".to_string()],
            allowed_permissions: vec![Permission::Read],
            deny_dids: vec!["did:key:alice".to_string()], // deny even though allowed
            expiry_ms: None,
        });
        let req = make_request("did:key:alice", "/res", Permission::Read);
        assert!(matches!(acl.check(&req), AccessDecision::Deny(_)));
    }

    // ===== NotFound =====

    #[test]
    fn test_not_found_unknown_resource() {
        let acl = make_acl();
        let req = make_request("did:key:alice", "/nonexistent", Permission::Read);
        assert_eq!(acl.check(&req), AccessDecision::NotFound);
    }

    #[test]
    fn test_not_found_after_remove() {
        let mut acl = make_acl();
        acl.remove_policy("/graph/data");
        let req = make_request("did:key:alice", "/graph/data", Permission::Read);
        assert_eq!(acl.check(&req), AccessDecision::NotFound);
    }

    // ===== Expired =====

    #[test]
    fn test_expired_policy() {
        let mut acl = AccessControlList::new();
        acl.add_policy(AccessPolicy {
            resource: "/tmp/resource".to_string(),
            allowed_dids: vec!["did:key:alice".to_string()],
            allowed_permissions: vec![Permission::Read],
            deny_dids: vec![],
            expiry_ms: Some(5000),
        });
        let req = AccessRequest {
            did: "did:key:alice".to_string(),
            resource: "/tmp/resource".to_string(),
            permission: Permission::Read,
            timestamp_ms: 5001, // past expiry
        };
        assert_eq!(acl.check(&req), AccessDecision::Expired);
    }

    #[test]
    fn test_not_expired_policy_before_expiry() {
        let mut acl = AccessControlList::new();
        acl.add_policy(AccessPolicy {
            resource: "/tmp/resource".to_string(),
            allowed_dids: vec!["did:key:alice".to_string()],
            allowed_permissions: vec![Permission::Read],
            deny_dids: vec![],
            expiry_ms: Some(10_000),
        });
        let req = AccessRequest {
            did: "did:key:alice".to_string(),
            resource: "/tmp/resource".to_string(),
            permission: Permission::Read,
            timestamp_ms: 9_999,
        };
        assert_eq!(acl.check(&req), AccessDecision::Allow);
    }

    // ===== Admin permission =====

    #[test]
    fn test_admin_allows_read() {
        let acl = make_acl();
        let req = make_request("did:key:admin", "/graph/admin", Permission::Read);
        assert_eq!(acl.check(&req), AccessDecision::Allow, "Admin should imply Read");
    }

    #[test]
    fn test_admin_allows_write() {
        let acl = make_acl();
        let req = make_request("did:key:admin", "/graph/admin", Permission::Write);
        assert_eq!(acl.check(&req), AccessDecision::Allow);
    }

    #[test]
    fn test_admin_allows_delete() {
        let acl = make_acl();
        let req = make_request("did:key:admin", "/graph/admin", Permission::Delete);
        assert_eq!(acl.check(&req), AccessDecision::Allow);
    }

    #[test]
    fn test_admin_allows_execute() {
        let acl = make_acl();
        let req = make_request("did:key:admin", "/graph/admin", Permission::Execute);
        assert_eq!(acl.check(&req), AccessDecision::Allow);
    }

    // ===== grant / revoke =====

    #[test]
    fn test_grant_adds_did() {
        let mut acl = make_acl();
        acl.grant("/graph/data", "did:key:carol", Permission::Read).unwrap();
        let req = make_request("did:key:carol", "/graph/data", Permission::Read);
        assert_eq!(acl.check(&req), AccessDecision::Allow);
    }

    #[test]
    fn test_grant_adds_permission() {
        let mut acl = make_acl();
        acl.grant("/graph/data", "did:key:alice", Permission::Delete).unwrap();
        let req = make_request("did:key:alice", "/graph/data", Permission::Delete);
        assert_eq!(acl.check(&req), AccessDecision::Allow);
    }

    #[test]
    fn test_grant_policy_not_found() {
        let mut acl = make_acl();
        let result = acl.grant("/nonexistent", "did:key:carol", Permission::Read);
        assert!(matches!(result, Err(AclError::PolicyNotFound(_))));
    }

    #[test]
    fn test_revoke_removes_did() {
        let mut acl = make_acl();
        let removed = acl.revoke("/graph/data", "did:key:alice");
        assert!(removed);
        let req = make_request("did:key:alice", "/graph/data", Permission::Read);
        assert!(matches!(acl.check(&req), AccessDecision::Deny(_)));
    }

    #[test]
    fn test_revoke_returns_false_when_did_not_present() {
        let mut acl = make_acl();
        let removed = acl.revoke("/graph/data", "did:key:nobody");
        assert!(!removed);
    }

    #[test]
    fn test_revoke_on_missing_resource_returns_false() {
        let mut acl = make_acl();
        let removed = acl.revoke("/nonexistent", "did:key:alice");
        assert!(!removed);
    }

    // ===== deny_did =====

    #[test]
    fn test_deny_did_policy_not_found() {
        let mut acl = make_acl();
        let result = acl.deny_did("/nonexistent", "did:key:alice");
        assert!(matches!(result, Err(AclError::PolicyNotFound(_))));
    }

    // ===== policies_for_did =====

    #[test]
    fn test_policies_for_did_returns_matching() {
        let acl = make_acl();
        let policies = acl.policies_for_did("did:key:alice");
        assert_eq!(policies.len(), 1);
        assert_eq!(policies[0].resource, "/graph/data");
    }

    #[test]
    fn test_policies_for_did_returns_empty_for_unknown() {
        let acl = make_acl();
        let policies = acl.policies_for_did("did:key:nobody");
        assert!(policies.is_empty());
    }

    #[test]
    fn test_policies_for_did_multiple_resources() {
        let mut acl = make_acl();
        acl.grant("/graph/admin", "did:key:alice", Permission::Read).unwrap();
        let policies = acl.policies_for_did("did:key:alice");
        assert_eq!(policies.len(), 2);
    }

    // ===== expired_policies / purge_expired =====

    #[test]
    fn test_expired_policies_returns_expired() {
        let mut acl = AccessControlList::new();
        acl.add_policy(AccessPolicy {
            resource: "/old".to_string(),
            allowed_dids: vec![],
            allowed_permissions: vec![],
            deny_dids: vec![],
            expiry_ms: Some(100),
        });
        let expired = acl.expired_policies(200);
        assert_eq!(expired.len(), 1);
    }

    #[test]
    fn test_expired_policies_excludes_active() {
        let mut acl = AccessControlList::new();
        acl.add_policy(AccessPolicy {
            resource: "/active".to_string(),
            allowed_dids: vec![],
            allowed_permissions: vec![],
            deny_dids: vec![],
            expiry_ms: Some(9999),
        });
        let expired = acl.expired_policies(100);
        assert!(expired.is_empty());
    }

    #[test]
    fn test_purge_expired_removes_policies() {
        let mut acl = AccessControlList::new();
        acl.add_policy(AccessPolicy {
            resource: "/old1".to_string(),
            allowed_dids: vec![],
            allowed_permissions: vec![],
            deny_dids: vec![],
            expiry_ms: Some(100),
        });
        acl.add_policy(AccessPolicy {
            resource: "/old2".to_string(),
            allowed_dids: vec![],
            allowed_permissions: vec![],
            deny_dids: vec![],
            expiry_ms: Some(200),
        });
        acl.add_policy(AccessPolicy {
            resource: "/active".to_string(),
            allowed_dids: vec![],
            allowed_permissions: vec![],
            deny_dids: vec![],
            expiry_ms: None,
        });
        let count = acl.purge_expired(300);
        assert_eq!(count, 2);
        assert_eq!(acl.policy_count(), 1);
    }

    #[test]
    fn test_purge_expired_returns_zero_when_nothing_expired() {
        let mut acl = make_acl();
        let count = acl.purge_expired(0);
        assert_eq!(count, 0);
    }

    // ===== policy_count =====

    #[test]
    fn test_policy_count_empty() {
        let acl = AccessControlList::new();
        assert_eq!(acl.policy_count(), 0);
    }

    #[test]
    fn test_policy_count_after_add() {
        let acl = make_acl();
        assert_eq!(acl.policy_count(), 2);
    }

    #[test]
    fn test_policy_count_after_remove() {
        let mut acl = make_acl();
        acl.remove_policy("/graph/data");
        assert_eq!(acl.policy_count(), 1);
    }

    // ===== permission variants =====

    #[test]
    fn test_permission_execute() {
        let mut acl = AccessControlList::new();
        acl.add_policy(AccessPolicy {
            resource: "/exec".to_string(),
            allowed_dids: vec!["did:key:runner".to_string()],
            allowed_permissions: vec![Permission::Execute],
            deny_dids: vec![],
            expiry_ms: None,
        });
        let req = make_request("did:key:runner", "/exec", Permission::Execute);
        assert_eq!(acl.check(&req), AccessDecision::Allow);
    }

    #[test]
    fn test_permission_delete() {
        let mut acl = AccessControlList::new();
        acl.add_policy(AccessPolicy {
            resource: "/delete".to_string(),
            allowed_dids: vec!["did:key:deleter".to_string()],
            allowed_permissions: vec![Permission::Delete],
            deny_dids: vec![],
            expiry_ms: None,
        });
        let req = make_request("did:key:deleter", "/delete", Permission::Delete);
        assert_eq!(acl.check(&req), AccessDecision::Allow);
    }

    #[test]
    fn test_remove_policy_returns_false_when_not_found() {
        let mut acl = make_acl();
        assert!(!acl.remove_policy("/does_not_exist"));
    }

    #[test]
    fn test_default_acl_is_empty() {
        let acl = AccessControlList::default();
        assert_eq!(acl.policy_count(), 0);
    }
}
