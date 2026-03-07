//! Graph-Level Access Control Lists (ACL)
//!
//! This module implements per-named-graph ACL entries that sit **on top of** the
//! existing dataset-level RBAC model provided by [`crate::auth::rbac`].
//!
//! # Permission model
//!
//! Authorization is a strict AND of two layers:
//!
//! ```text
//! ALLOW  iff  dataset_rbac_allows(principal, action)
//!             AND  graph_acl_allows(graph_iri, principal, graph_permission)
//! ```
//!
//! The graph ACL layer is additive: if **no** ACL entry exists for a graph the
//! default decision is `Deny`, preserving the principle of least-privilege.
//! A principal whose dataset-level role includes `GlobalAdmin` or who holds
//! the graph-level `Admin` permission bypasses all graph-specific checks.
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_fuseki::auth::graph_acl::{GraphAclStore, GraphPermission};
//!
//! let store = GraphAclStore::new();
//! store.grant("http://example.org/g1", "alice", GraphPermission::Read).await?;
//!
//! let can_read = store
//!     .check("http://example.org/g1", "alice", GraphPermission::Read)
//!     .await;
//! assert!(can_read);
//! ```

use crate::auth::types::{Permission, User};
use crate::error::{FusekiError, FusekiResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

// ──────────────────────────────────────────────────────────────────────────────
// Core permission enum
// ──────────────────────────────────────────────────────────────────────────────

/// Fine-grained permission that can be attached to a named-graph ACL entry.
///
/// Permissions are **additive**: `Admin` implies both `Write` and `Read`;
/// `Write` implies `Read`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum GraphPermission {
    /// Allows SPARQL SELECT / CONSTRUCT / DESCRIBE / ASK against the graph.
    Read,
    /// Allows INSERT DATA / DELETE DATA / LOAD / CLEAR for the graph.
    /// Implies `Read`.
    Write,
    /// Full control: can grant/revoke ACL entries and perform destructive ops.
    /// Implies `Write` and `Read`.
    Admin,
}

impl GraphPermission {
    /// Returns `true` when `self` logically implies `other`.
    ///
    /// ```text
    /// Admin  → Write → Read
    /// ```
    pub fn implies(self, other: GraphPermission) -> bool {
        match self {
            GraphPermission::Admin => true,
            GraphPermission::Write => {
                matches!(other, GraphPermission::Read | GraphPermission::Write)
            }
            GraphPermission::Read => matches!(other, GraphPermission::Read),
        }
    }

    /// Returns a human-readable label.
    pub fn as_str(self) -> &'static str {
        match self {
            GraphPermission::Read => "read",
            GraphPermission::Write => "write",
            GraphPermission::Admin => "admin",
        }
    }
}

impl std::fmt::Display for GraphPermission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Principal
// ──────────────────────────────────────────────────────────────────────────────

/// A security principal that can appear in a graph ACL entry.
///
/// Two principal kinds are supported:
/// - `User(username)` — a specific authenticated user
/// - `Role(role_name)` — any user who holds the named RBAC role
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Principal {
    User { name: String },
    Role { name: String },
}

impl Principal {
    /// Construct a user principal.
    pub fn user(name: impl Into<String>) -> Self {
        Principal::User { name: name.into() }
    }

    /// Construct a role principal.
    pub fn role(name: impl Into<String>) -> Self {
        Principal::Role { name: name.into() }
    }

    /// Returns `true` when this principal applies to `user`.
    pub fn matches_user(&self, user: &User) -> bool {
        match self {
            Principal::User { name } => &user.username == name,
            Principal::Role { name } => user.roles.contains(name),
        }
    }
}

impl std::fmt::Display for Principal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Principal::User { name } => write!(f, "user:{name}"),
            Principal::Role { name } => write!(f, "role:{name}"),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ACL entry (the persistent record)
// ──────────────────────────────────────────────────────────────────────────────

/// A single ACL entry that binds a principal to a permission on a named graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAclEntry {
    /// The named graph IRI (e.g. `"http://example.org/graph/private"`).
    pub graph_iri: String,
    /// Who holds this permission.
    pub principal: Principal,
    /// The permission that is granted.
    pub permission: GraphPermission,
    /// When this entry was created.
    pub granted_at: DateTime<Utc>,
    /// Optional: who created this entry (for audit trails).
    pub granted_by: Option<String>,
}

impl GraphAclEntry {
    fn new(
        graph_iri: impl Into<String>,
        principal: Principal,
        permission: GraphPermission,
        granted_by: Option<String>,
    ) -> Self {
        Self {
            graph_iri: graph_iri.into(),
            principal,
            permission,
            granted_at: Utc::now(),
            granted_by,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Management request / response types
// ──────────────────────────────────────────────────────────────────────────────

/// Request to grant a principal access to a named graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrantGraphAccess {
    pub graph_iri: String,
    pub principal: Principal,
    pub permission: GraphPermission,
    /// Username of the person performing the grant (for auditing).
    pub granted_by: Option<String>,
}

/// Request to revoke a principal's access to a named graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevokeGraphAccess {
    pub graph_iri: String,
    pub principal: Principal,
    /// If `None`, **all** permissions for this principal on the graph are revoked.
    pub permission: Option<GraphPermission>,
}

/// Summary of the access-check result returned by [`GraphAclPolicy`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AclDecision {
    /// Access is allowed. Contains the reason for diagnostic purposes.
    Allow { reason: AllowReason },
    /// Access is denied.
    Deny { reason: DenyReason },
}

impl AclDecision {
    /// Returns `true` iff this decision is `Allow`.
    pub fn is_allow(&self) -> bool {
        matches!(self, AclDecision::Allow { .. })
    }
}

/// Why access was allowed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllowReason {
    /// Principal holds a dataset-level admin permission (bypasses graph ACL).
    DatasetAdmin,
    /// Principal has an explicit graph-level ACL entry (or implied by higher entry).
    ExplicitAcl,
}

/// Why access was denied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenyReason {
    /// Dataset-level RBAC already denied the request — graph ACL was not consulted.
    DatasetRbacDenied,
    /// Dataset-level RBAC passed but there is no matching graph ACL entry.
    NoAclEntry,
}

// ──────────────────────────────────────────────────────────────────────────────
// GraphAcl — the per-graph permission set
// ──────────────────────────────────────────────────────────────────────────────

/// In-memory representation of all ACL entries for a single named graph.
///
/// Stored as `HashMap<Principal, HashSet<GraphPermission>>` for O(1) lookups.
#[derive(Debug, Default, Clone)]
struct GraphAcl {
    entries: HashMap<Principal, HashSet<GraphPermission>>,
}

impl GraphAcl {
    fn grant(&mut self, principal: Principal, permission: GraphPermission) {
        self.entries
            .entry(principal)
            .or_insert_with(HashSet::new)
            .insert(permission);
    }

    /// Revoke a specific permission (or all permissions if `permission` is `None`).
    fn revoke(&mut self, principal: &Principal, permission: Option<GraphPermission>) -> bool {
        match self.entries.get_mut(principal) {
            None => false,
            Some(perms) => {
                if let Some(p) = permission {
                    let removed = perms.remove(&p);
                    if perms.is_empty() {
                        self.entries.remove(principal);
                    }
                    removed
                } else {
                    self.entries.remove(principal).is_some()
                }
            }
        }
    }

    /// Check whether `principal` has (or is implied to have) `permission`.
    ///
    /// A principal satisfies the check when they hold **any** permission that
    /// `implies` the requested one.
    fn check(&self, principal: &Principal, permission: GraphPermission) -> bool {
        match self.entries.get(principal) {
            None => false,
            Some(perms) => perms.iter().any(|p| p.implies(permission)),
        }
    }

    /// Check whether `user` (matched against user and role principals) has `permission`.
    fn check_user(&self, user: &User, permission: GraphPermission) -> bool {
        self.entries.iter().any(|(principal, perms)| {
            principal.matches_user(user) && perms.iter().any(|p| p.implies(permission))
        })
    }

    /// Enumerate all `(Principal, GraphPermission)` pairs in this ACL.
    fn list(&self) -> Vec<(Principal, GraphPermission)> {
        let mut out = Vec::new();
        for (principal, perms) in &self.entries {
            for perm in perms {
                out.push((principal.clone(), *perm));
            }
        }
        out.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.to_string().cmp(&b.0.to_string())));
        out
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GraphAclStore — the top-level CRUD service
// ──────────────────────────────────────────────────────────────────────────────

/// Thread-safe in-memory store for graph-level ACL entries.
///
/// This is the primary API surface used by the server to manage graph ACLs.
/// All mutating operations are async-safe behind `tokio::sync::RwLock`.
///
/// # Clonable handle
///
/// `GraphAclStore` is `Clone` and wraps its state in an `Arc`, so callers can
/// cheaply share it across Axum handler closures.
#[derive(Debug, Clone)]
pub struct GraphAclStore {
    inner: Arc<RwLock<HashMap<String, GraphAcl>>>,
}

impl GraphAclStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    // ── Mutations ─────────────────────────────────────────────────────────────

    /// Grant `principal` the given `permission` on `graph_iri`.
    ///
    /// Idempotent: granting an already-held permission is a no-op.
    pub async fn grant(&self, req: &GrantGraphAccess) -> FusekiResult<GraphAclEntry> {
        let mut store = self.inner.write().await;
        let acl = store
            .entry(req.graph_iri.clone())
            .or_insert_with(GraphAcl::default);

        acl.grant(req.principal.clone(), req.permission);

        let entry = GraphAclEntry::new(
            &req.graph_iri,
            req.principal.clone(),
            req.permission,
            req.granted_by.clone(),
        );

        info!(
            "GraphACL grant: {} {} on <{}>",
            req.principal, req.permission, req.graph_iri
        );

        Ok(entry)
    }

    /// Revoke access from `principal` on `graph_iri`.
    ///
    /// Returns `Ok(true)` if an entry was removed, `Ok(false)` if nothing
    /// matched (idempotent — not an error).
    pub async fn revoke(&self, req: &RevokeGraphAccess) -> FusekiResult<bool> {
        let mut store = self.inner.write().await;

        match store.get_mut(&req.graph_iri) {
            None => {
                debug!("GraphACL revoke: no ACL for <{}> (noop)", req.graph_iri);
                Ok(false)
            }
            Some(acl) => {
                let removed = acl.revoke(&req.principal, req.permission);
                if removed {
                    info!(
                        "GraphACL revoke: {} {:?} on <{}>",
                        req.principal, req.permission, req.graph_iri
                    );
                    // Clean up empty ACL maps
                    if acl.entries.is_empty() {
                        store.remove(&req.graph_iri);
                    }
                }
                Ok(removed)
            }
        }
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Check whether a `Principal` directly holds (or is implied to hold)
    /// `permission` on `graph_iri`.
    ///
    /// This is the low-level check that does **not** consider dataset-level RBAC.
    /// Use [`GraphAclPolicy::check`] for the combined check.
    pub async fn check(
        &self,
        graph_iri: &str,
        principal: &Principal,
        permission: GraphPermission,
    ) -> bool {
        let store = self.inner.read().await;
        match store.get(graph_iri) {
            None => false,
            Some(acl) => acl.check(principal, permission),
        }
    }

    /// Check whether `user` (matching on username and roles) has `permission`
    /// on `graph_iri`.
    pub async fn check_user(
        &self,
        graph_iri: &str,
        user: &User,
        permission: GraphPermission,
    ) -> bool {
        let store = self.inner.read().await;
        match store.get(graph_iri) {
            None => false,
            Some(acl) => acl.check_user(user, permission),
        }
    }

    /// List all ACL entries for a named graph.
    pub async fn list_graph(&self, graph_iri: &str) -> Vec<GraphAclEntry> {
        let store = self.inner.read().await;
        match store.get(graph_iri) {
            None => vec![],
            Some(acl) => acl
                .list()
                .into_iter()
                .map(|(principal, permission)| {
                    GraphAclEntry::new(graph_iri, principal, permission, None)
                })
                .collect(),
        }
    }

    /// List all ACL entries across all graphs.
    pub async fn list_all(&self) -> Vec<GraphAclEntry> {
        let store = self.inner.read().await;
        let mut out = Vec::new();
        for (graph_iri, acl) in store.iter() {
            for (principal, permission) in acl.list() {
                out.push(GraphAclEntry::new(graph_iri, principal, permission, None));
            }
        }
        out.sort_by(|a, b| {
            a.graph_iri
                .cmp(&b.graph_iri)
                .then(a.permission.cmp(&b.permission))
        });
        out
    }

    /// List all graphs for which `principal` has any explicit ACL entry.
    pub async fn graphs_for_principal(
        &self,
        principal: &Principal,
    ) -> Vec<(String, GraphPermission)> {
        let store = self.inner.read().await;
        let mut out = Vec::new();
        for (graph_iri, acl) in store.iter() {
            if let Some(perms) = acl.entries.get(principal) {
                for perm in perms {
                    out.push((graph_iri.clone(), *perm));
                }
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        out
    }

    /// Remove **all** ACL entries for a named graph (e.g., when the graph is
    /// deleted from the dataset).
    pub async fn purge_graph(&self, graph_iri: &str) -> FusekiResult<usize> {
        let mut store = self.inner.write().await;
        match store.remove(graph_iri) {
            None => Ok(0),
            Some(acl) => {
                let count = acl.entries.values().map(|s| s.len()).sum();
                info!(
                    "GraphACL purge: removed {} entries for <{}>",
                    count, graph_iri
                );
                Ok(count)
            }
        }
    }

    /// Return the total number of ACL entries currently stored.
    pub async fn entry_count(&self) -> usize {
        let store = self.inner.read().await;
        store
            .values()
            .map(|acl| acl.entries.values().map(|s| s.len()).sum::<usize>())
            .sum()
    }
}

impl Default for GraphAclStore {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GraphAclPolicy — AND-combines dataset RBAC + graph ACL
// ──────────────────────────────────────────────────────────────────────────────

/// Policy evaluator that AND-combines the dataset-level RBAC decision with the
/// graph-level ACL decision.
///
/// The evaluation algorithm:
///
/// 1. If the user holds `Permission::GlobalAdmin` or `Permission::DatasetAdmin(dataset_id)`,
///    return `Allow { reason: DatasetAdmin }` immediately — admins bypass graph ACLs.
/// 2. Check dataset-level RBAC via the supplied `dataset_rbac_check` callback.
///    If denied → `Deny { reason: DatasetRbacDenied }`.
/// 3. Look up the graph ACL in the store.
///    If no matching entry exists → `Deny { reason: NoAclEntry }`.
/// 4. Otherwise → `Allow { reason: ExplicitAcl }`.
#[derive(Clone)]
pub struct GraphAclPolicy {
    store: GraphAclStore,
}

impl GraphAclPolicy {
    /// Create a policy backed by the given ACL store.
    pub fn new(store: GraphAclStore) -> Self {
        Self { store }
    }

    /// Perform the combined access-control check.
    ///
    /// # Parameters
    /// - `user` — The authenticated user.
    /// - `dataset_id` — Identifier of the dataset that contains the graph
    ///   (used for dataset-admin bypass check).
    /// - `graph_iri` — The named graph being accessed.
    /// - `permission` — The graph-level permission being requested.
    pub async fn check(
        &self,
        user: &User,
        dataset_id: &str,
        graph_iri: &str,
        permission: GraphPermission,
    ) -> AclDecision {
        // ── Step 1: Admin bypass ─────────────────────────────────────────────
        let is_global_admin = user.permissions.contains(&Permission::GlobalAdmin);
        let is_dataset_admin = user
            .permissions
            .contains(&Permission::DatasetAdmin(dataset_id.to_string()));

        if is_global_admin || is_dataset_admin {
            debug!(
                "GraphAclPolicy: admin bypass for {} on <{}>",
                user.username, graph_iri
            );
            return AclDecision::Allow {
                reason: AllowReason::DatasetAdmin,
            };
        }

        // ── Step 2: Dataset-level RBAC gate ──────────────────────────────────
        // At dataset level we require at least the read permission for any
        // graph-level check; write/admin require the corresponding dataset perm.
        let dataset_ok = self.dataset_rbac_allows(user, dataset_id, permission);
        if !dataset_ok {
            debug!(
                "GraphAclPolicy: dataset RBAC denied {} {:?} in dataset '{}'",
                user.username, permission, dataset_id
            );
            return AclDecision::Deny {
                reason: DenyReason::DatasetRbacDenied,
            };
        }

        // ── Step 3: Graph ACL check ───────────────────────────────────────────
        let acl_allows = self.store.check_user(graph_iri, user, permission).await;
        if acl_allows {
            debug!(
                "GraphAclPolicy: ACL allowed {} {:?} on <{}>",
                user.username, permission, graph_iri
            );
            AclDecision::Allow {
                reason: AllowReason::ExplicitAcl,
            }
        } else {
            debug!(
                "GraphAclPolicy: no ACL entry for {} {:?} on <{}>",
                user.username, permission, graph_iri
            );
            AclDecision::Deny {
                reason: DenyReason::NoAclEntry,
            }
        }
    }

    /// Minimal dataset-level RBAC check embedded directly in the policy.
    ///
    /// This uses the `Permission` enum already on the `User` struct so it works
    /// without an async call to the full `RbacManager`.  The rules are:
    ///
    /// | Permission requested | Dataset-level requirements                              |
    /// |----------------------|--------------------------------------------------------|
    /// | `Read`               | `GlobalRead`, `DatasetRead(id)`, `GlobalWrite`, `GlobalAdmin` |
    /// | `Write`              | `GlobalWrite`, `DatasetWrite(id)`, `GlobalAdmin`       |
    /// | `Admin`              | `GlobalAdmin`, `DatasetAdmin(id)`                      |
    fn dataset_rbac_allows(
        &self,
        user: &User,
        dataset_id: &str,
        permission: GraphPermission,
    ) -> bool {
        user.permissions.iter().any(|p| match permission {
            GraphPermission::Read => {
                matches!(
                    p,
                    Permission::GlobalRead
                        | Permission::GlobalWrite
                        | Permission::GlobalAdmin
                        | Permission::SparqlQuery
                ) || matches!(p, Permission::DatasetRead(id) if id == dataset_id)
                    || matches!(p, Permission::DatasetWrite(id) if id == dataset_id)
                    || matches!(p, Permission::DatasetAdmin(id) if id == dataset_id)
            }

            GraphPermission::Write => {
                matches!(
                    p,
                    Permission::GlobalWrite | Permission::GlobalAdmin | Permission::SparqlUpdate
                ) || matches!(p, Permission::DatasetWrite(id) if id == dataset_id)
                    || matches!(p, Permission::DatasetAdmin(id) if id == dataset_id)
            }

            GraphPermission::Admin => {
                matches!(p, Permission::GlobalAdmin)
                    || matches!(p, Permission::DatasetAdmin(id) if id == dataset_id)
            }
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_user(username: &str, roles: &[&str], permissions: &[Permission]) -> User {
        User {
            username: username.to_string(),
            roles: roles.iter().map(|s| s.to_string()).collect(),
            email: None,
            full_name: None,
            last_login: Some(Utc::now()),
            permissions: permissions.to_vec(),
        }
    }

    fn grant_req(
        graph_iri: &str,
        principal: Principal,
        permission: GraphPermission,
    ) -> GrantGraphAccess {
        GrantGraphAccess {
            graph_iri: graph_iri.to_string(),
            principal,
            permission,
            granted_by: Some("admin".to_string()),
        }
    }

    fn revoke_req(
        graph_iri: &str,
        principal: Principal,
        permission: Option<GraphPermission>,
    ) -> RevokeGraphAccess {
        RevokeGraphAccess {
            graph_iri: graph_iri.to_string(),
            principal,
            permission,
        }
    }

    const G1: &str = "http://example.org/graph/g1";
    const G2: &str = "http://example.org/graph/g2";
    const G3: &str = "http://example.org/graph/g3";

    // ── 1. Grant read and verify ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_grant_read_allows_read() {
        let store = GraphAclStore::new();
        let alice = Principal::user("alice");
        store
            .grant(&grant_req(G1, alice.clone(), GraphPermission::Read))
            .await
            .unwrap();

        assert!(store.check(G1, &alice, GraphPermission::Read).await);
    }

    // ── 2. Grant read does not allow write ────────────────────────────────────

    #[tokio::test]
    async fn test_grant_read_denies_write() {
        let store = GraphAclStore::new();
        let alice = Principal::user("alice");
        store
            .grant(&grant_req(G1, alice.clone(), GraphPermission::Read))
            .await
            .unwrap();

        assert!(!store.check(G1, &alice, GraphPermission::Write).await);
    }

    // ── 3. Grant write implies read ───────────────────────────────────────────

    #[tokio::test]
    async fn test_grant_write_implies_read() {
        let store = GraphAclStore::new();
        let bob = Principal::user("bob");
        store
            .grant(&grant_req(G1, bob.clone(), GraphPermission::Write))
            .await
            .unwrap();

        assert!(store.check(G1, &bob, GraphPermission::Read).await);
        assert!(store.check(G1, &bob, GraphPermission::Write).await);
        assert!(!store.check(G1, &bob, GraphPermission::Admin).await);
    }

    // ── 4. Grant admin implies write and read ─────────────────────────────────

    #[tokio::test]
    async fn test_grant_admin_implies_all() {
        let store = GraphAclStore::new();
        let carol = Principal::user("carol");
        store
            .grant(&grant_req(G1, carol.clone(), GraphPermission::Admin))
            .await
            .unwrap();

        assert!(store.check(G1, &carol, GraphPermission::Read).await);
        assert!(store.check(G1, &carol, GraphPermission::Write).await);
        assert!(store.check(G1, &carol, GraphPermission::Admin).await);
    }

    // ── 5. Revoke specific permission ─────────────────────────────────────────

    #[tokio::test]
    async fn test_revoke_permission() {
        let store = GraphAclStore::new();
        let alice = Principal::user("alice");
        store
            .grant(&grant_req(G1, alice.clone(), GraphPermission::Write))
            .await
            .unwrap();
        let removed = store
            .revoke(&revoke_req(G1, alice.clone(), Some(GraphPermission::Write)))
            .await
            .unwrap();
        assert!(removed);
        assert!(!store.check(G1, &alice, GraphPermission::Write).await);
    }

    // ── 6. Revoke all permissions for a principal ─────────────────────────────

    #[tokio::test]
    async fn test_revoke_all_permissions() {
        let store = GraphAclStore::new();
        let alice = Principal::user("alice");
        store
            .grant(&grant_req(G1, alice.clone(), GraphPermission::Read))
            .await
            .unwrap();
        store
            .grant(&grant_req(G1, alice.clone(), GraphPermission::Write))
            .await
            .unwrap();

        let removed = store
            .revoke(&revoke_req(G1, alice.clone(), None))
            .await
            .unwrap();
        assert!(removed);
        assert!(!store.check(G1, &alice, GraphPermission::Read).await);
        assert!(!store.check(G1, &alice, GraphPermission::Write).await);
        assert_eq!(store.entry_count().await, 0);
    }

    // ── 7. Revoke is idempotent on missing entry ──────────────────────────────

    #[tokio::test]
    async fn test_revoke_missing_entry_is_ok() {
        let store = GraphAclStore::new();
        let alice = Principal::user("alice");
        let result = store
            .revoke(&revoke_req(G1, alice.clone(), Some(GraphPermission::Read)))
            .await;
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    // ── 8. No ACL entry denies access ─────────────────────────────────────────

    #[tokio::test]
    async fn test_no_entry_denies_access() {
        let store = GraphAclStore::new();
        let alice = Principal::user("alice");
        assert!(!store.check(G1, &alice, GraphPermission::Read).await);
    }

    // ── 9. Multiple principals, multiple graphs ───────────────────────────────

    #[tokio::test]
    async fn test_multiple_principals_multiple_graphs() {
        let store = GraphAclStore::new();
        let alice = Principal::user("alice");
        let bob = Principal::user("bob");
        store
            .grant(&grant_req(G1, alice.clone(), GraphPermission::Read))
            .await
            .unwrap();
        store
            .grant(&grant_req(G2, bob.clone(), GraphPermission::Write))
            .await
            .unwrap();

        assert!(store.check(G1, &alice, GraphPermission::Read).await);
        assert!(!store.check(G1, &bob, GraphPermission::Read).await);
        assert!(store.check(G2, &bob, GraphPermission::Read).await);
        assert!(!store.check(G2, &alice, GraphPermission::Write).await);
    }

    // ── 10. Role-based principal matching ─────────────────────────────────────

    #[tokio::test]
    async fn test_role_principal_matches_user_with_role() {
        let store = GraphAclStore::new();
        let editor_role = Principal::role("editor");
        store
            .grant(&grant_req(G1, editor_role, GraphPermission::Write))
            .await
            .unwrap();

        let alice = make_user("alice", &["editor"], &[Permission::GlobalRead]);
        assert!(store.check_user(G1, &alice, GraphPermission::Write).await);
    }

    // ── 11. Role principal does not match user without role ───────────────────

    #[tokio::test]
    async fn test_role_principal_no_match_without_role() {
        let store = GraphAclStore::new();
        let editor_role = Principal::role("editor");
        store
            .grant(&grant_req(G1, editor_role, GraphPermission::Write))
            .await
            .unwrap();

        let bob = make_user("bob", &["reader"], &[Permission::GlobalRead]);
        assert!(!store.check_user(G1, &bob, GraphPermission::Write).await);
    }

    // ── 12. Grant returns correct entry ──────────────────────────────────────

    #[tokio::test]
    async fn test_grant_returns_entry() {
        let store = GraphAclStore::new();
        let alice = Principal::user("alice");
        let entry = store
            .grant(&grant_req(G1, alice.clone(), GraphPermission::Read))
            .await
            .unwrap();

        assert_eq!(entry.graph_iri, G1);
        assert_eq!(entry.principal, alice);
        assert_eq!(entry.permission, GraphPermission::Read);
        assert_eq!(entry.granted_by.as_deref(), Some("admin"));
    }

    // ── 13. List graph entries ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_list_graph_entries() {
        let store = GraphAclStore::new();
        store
            .grant(&grant_req(
                G1,
                Principal::user("alice"),
                GraphPermission::Read,
            ))
            .await
            .unwrap();
        store
            .grant(&grant_req(
                G1,
                Principal::user("bob"),
                GraphPermission::Write,
            ))
            .await
            .unwrap();
        store
            .grant(&grant_req(
                G2,
                Principal::user("carol"),
                GraphPermission::Admin,
            ))
            .await
            .unwrap();

        let g1_entries = store.list_graph(G1).await;
        assert_eq!(g1_entries.len(), 2);
        assert!(g1_entries.iter().all(|e| e.graph_iri == G1));

        let g2_entries = store.list_graph(G2).await;
        assert_eq!(g2_entries.len(), 1);
    }

    // ── 14. List all entries ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_list_all_entries() {
        let store = GraphAclStore::new();
        store
            .grant(&grant_req(
                G1,
                Principal::user("alice"),
                GraphPermission::Read,
            ))
            .await
            .unwrap();
        store
            .grant(&grant_req(
                G2,
                Principal::user("bob"),
                GraphPermission::Write,
            ))
            .await
            .unwrap();
        store
            .grant(&grant_req(
                G3,
                Principal::role("admin_role"),
                GraphPermission::Admin,
            ))
            .await
            .unwrap();

        let all = store.list_all().await;
        assert_eq!(all.len(), 3);
    }

    // ── 15. Purge graph ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_purge_graph() {
        let store = GraphAclStore::new();
        store
            .grant(&grant_req(
                G1,
                Principal::user("alice"),
                GraphPermission::Read,
            ))
            .await
            .unwrap();
        store
            .grant(&grant_req(
                G1,
                Principal::user("bob"),
                GraphPermission::Write,
            ))
            .await
            .unwrap();
        store
            .grant(&grant_req(
                G2,
                Principal::user("carol"),
                GraphPermission::Read,
            ))
            .await
            .unwrap();

        let removed = store.purge_graph(G1).await.unwrap();
        assert_eq!(removed, 2);
        assert_eq!(store.entry_count().await, 1); // G2 entry remains
        assert!(
            !store
                .check(G1, &Principal::user("alice"), GraphPermission::Read)
                .await
        );
    }

    // ── 16. Graphs for principal ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_graphs_for_principal() {
        let store = GraphAclStore::new();
        let alice = Principal::user("alice");
        store
            .grant(&grant_req(G1, alice.clone(), GraphPermission::Read))
            .await
            .unwrap();
        store
            .grant(&grant_req(G2, alice.clone(), GraphPermission::Write))
            .await
            .unwrap();
        store
            .grant(&grant_req(
                G3,
                Principal::user("bob"),
                GraphPermission::Read,
            ))
            .await
            .unwrap();

        let alice_graphs = store.graphs_for_principal(&alice).await;
        assert_eq!(alice_graphs.len(), 2);
        assert!(alice_graphs.iter().any(|(g, _)| g == G1));
        assert!(alice_graphs.iter().any(|(g, _)| g == G2));
    }

    // ── 17. Policy: dataset admin bypass ──────────────────────────────────────

    #[tokio::test]
    async fn test_policy_dataset_admin_bypasses_acl() {
        let store = GraphAclStore::new();
        // No ACL entry for G1 at all — admin should still get through
        let policy = GraphAclPolicy::new(store);

        let admin = make_user("admin", &["admin"], &[Permission::GlobalAdmin]);
        let decision = policy
            .check(&admin, "ds1", G1, GraphPermission::Write)
            .await;
        assert!(decision.is_allow());
        assert_eq!(
            decision,
            AclDecision::Allow {
                reason: AllowReason::DatasetAdmin
            }
        );
    }

    // ── 18. Policy: dataset-scoped admin bypass ───────────────────────────────

    #[tokio::test]
    async fn test_policy_dataset_scoped_admin_bypass() {
        let store = GraphAclStore::new();
        let policy = GraphAclPolicy::new(store);

        let ds_admin = make_user(
            "ds_admin",
            &["dataset_admin"],
            &[Permission::DatasetAdmin("ds1".to_string())],
        );
        let decision = policy
            .check(&ds_admin, "ds1", G1, GraphPermission::Admin)
            .await;
        assert!(decision.is_allow());
        assert_eq!(
            decision,
            AclDecision::Allow {
                reason: AllowReason::DatasetAdmin
            }
        );
    }

    // ── 19. Policy: dataset RBAC denied ───────────────────────────────────────

    #[tokio::test]
    async fn test_policy_dataset_rbac_denied() {
        let store = GraphAclStore::new();
        // Grant graph-level ACL but user has NO dataset-level permission
        store
            .grant(&grant_req(
                G1,
                Principal::user("alice"),
                GraphPermission::Write,
            ))
            .await
            .unwrap();
        let policy = GraphAclPolicy::new(store);

        let alice = make_user("alice", &[], &[]); // no permissions at all
        let decision = policy
            .check(&alice, "ds1", G1, GraphPermission::Write)
            .await;
        assert!(!decision.is_allow());
        assert_eq!(
            decision,
            AclDecision::Deny {
                reason: DenyReason::DatasetRbacDenied
            }
        );
    }

    // ── 20. Policy: no graph ACL entry ────────────────────────────────────────

    #[tokio::test]
    async fn test_policy_no_acl_entry_denied() {
        let store = GraphAclStore::new();
        // No graph ACL entry, but user has dataset read permission
        let policy = GraphAclPolicy::new(store);

        let alice = make_user("alice", &["reader"], &[Permission::GlobalRead]);
        let decision = policy.check(&alice, "ds1", G1, GraphPermission::Read).await;
        assert!(!decision.is_allow());
        assert_eq!(
            decision,
            AclDecision::Deny {
                reason: DenyReason::NoAclEntry
            }
        );
    }

    // ── 21. Policy: dataset read + graph ACL allows ───────────────────────────

    #[tokio::test]
    async fn test_policy_combined_allows() {
        let store = GraphAclStore::new();
        store
            .grant(&grant_req(
                G1,
                Principal::user("alice"),
                GraphPermission::Read,
            ))
            .await
            .unwrap();
        let policy = GraphAclPolicy::new(store);

        let alice = make_user("alice", &["reader"], &[Permission::GlobalRead]);
        let decision = policy.check(&alice, "ds1", G1, GraphPermission::Read).await;
        assert!(decision.is_allow());
        assert_eq!(
            decision,
            AclDecision::Allow {
                reason: AllowReason::ExplicitAcl
            }
        );
    }

    // ── 22. Policy: write blocked despite graph ACL when dataset denies write ──

    #[tokio::test]
    async fn test_policy_dataset_denies_write_despite_graph_acl() {
        let store = GraphAclStore::new();
        // Graph ACL grants write, but user only has GlobalRead (no write)
        store
            .grant(&grant_req(
                G1,
                Principal::user("alice"),
                GraphPermission::Write,
            ))
            .await
            .unwrap();
        let policy = GraphAclPolicy::new(store);

        let alice = make_user("alice", &["reader"], &[Permission::GlobalRead]);
        let decision = policy
            .check(&alice, "ds1", G1, GraphPermission::Write)
            .await;
        // Dataset RBAC doesn't permit write → deny
        assert!(!decision.is_allow());
        assert_eq!(
            decision,
            AclDecision::Deny {
                reason: DenyReason::DatasetRbacDenied
            }
        );
    }

    // ── 23. Policy: role-based principal in combined check ────────────────────

    #[tokio::test]
    async fn test_policy_role_based_principal_combined() {
        let store = GraphAclStore::new();
        store
            .grant(&grant_req(
                G1,
                Principal::role("writer"),
                GraphPermission::Write,
            ))
            .await
            .unwrap();
        let policy = GraphAclPolicy::new(store);

        let alice = make_user("alice", &["writer"], &[Permission::GlobalWrite]);
        let decision = policy
            .check(&alice, "ds1", G1, GraphPermission::Write)
            .await;
        assert!(decision.is_allow());
    }

    // ── 24. GraphPermission ordering and display ──────────────────────────────

    #[test]
    fn test_graph_permission_display() {
        assert_eq!(GraphPermission::Read.as_str(), "read");
        assert_eq!(GraphPermission::Write.as_str(), "write");
        assert_eq!(GraphPermission::Admin.as_str(), "admin");
        assert_eq!(format!("{}", GraphPermission::Read), "read");
    }

    // ── 25. Entry count stays accurate across grant/revoke ────────────────────

    #[tokio::test]
    async fn test_entry_count_accurate() {
        let store = GraphAclStore::new();
        assert_eq!(store.entry_count().await, 0);
        store
            .grant(&grant_req(
                G1,
                Principal::user("alice"),
                GraphPermission::Read,
            ))
            .await
            .unwrap();
        assert_eq!(store.entry_count().await, 1);
        store
            .grant(&grant_req(
                G1,
                Principal::user("bob"),
                GraphPermission::Write,
            ))
            .await
            .unwrap();
        assert_eq!(store.entry_count().await, 2);
        store
            .revoke(&revoke_req(
                G1,
                Principal::user("alice"),
                Some(GraphPermission::Read),
            ))
            .await
            .unwrap();
        assert_eq!(store.entry_count().await, 1);
    }

    // ── 26. Idempotent grant ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_idempotent_grant() {
        let store = GraphAclStore::new();
        let alice = Principal::user("alice");
        store
            .grant(&grant_req(G1, alice.clone(), GraphPermission::Read))
            .await
            .unwrap();
        store
            .grant(&grant_req(G1, alice.clone(), GraphPermission::Read))
            .await
            .unwrap();
        assert_eq!(store.entry_count().await, 1); // still just one entry
    }
}
