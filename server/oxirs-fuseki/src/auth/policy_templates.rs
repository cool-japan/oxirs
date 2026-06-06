//! Enterprise RBAC policy template system.
//!
//! This module provides a [`PolicyTemplateRegistry`] that ships three built-in
//! role templates:
//!
//! | Template   | Purpose                                              |
//! |------------|------------------------------------------------------|
//! | `dba`      | Database Administrator — full CRUD + admin control   |
//! | `readonly` | Read-only analyst — SELECT/ASK/CONSTRUCT only        |
//! | `auditor`  | Auditor — read access plus audit-log read            |
//!
//! Additional templates can be registered at runtime via [`PolicyTemplateRegistry::register`].

use super::types::{AuthError, Permission, User};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// GraphScope
// ──────────────────────────────────────────────────────────────────────────────

/// Describes which named graphs a policy template applies to.
///
/// `AllNamedGraphs` means the template grants access to every named graph in the
/// dataset — the typical choice for role templates applied at the account level.
/// `SpecificGraph` restricts the scope to a single well-known graph IRI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphScope {
    /// Access applies to every named graph in the dataset.
    AllNamedGraphs,
    /// Access applies only to the named graph identified by this IRI.
    SpecificGraph { iri: String },
}

impl GraphScope {
    /// Construct a `SpecificGraph` scope from an IRI string.
    pub fn specific(iri: impl Into<String>) -> Self {
        GraphScope::SpecificGraph { iri: iri.into() }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PolicyTemplate
// ──────────────────────────────────────────────────────────────────────────────

/// A named collection of permissions and graph scopes that can be applied to a
/// [`User`] in bulk.
///
/// Templates are identified by a lower-case `name` string (e.g. `"dba"`).
/// Applying a template extends the user's existing permissions — it does **not**
/// replace them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyTemplate {
    /// Unique identifier for the template (lower-case, snake_case).
    pub name: String,
    /// Human-readable explanation of what this template grants.
    pub description: String,
    /// The permissions that will be added to a user when this template is applied.
    pub permissions: Vec<Permission>,
    /// Which named graphs the template's permissions apply to.
    pub graph_scopes: Vec<GraphScope>,
}

impl PolicyTemplate {
    /// Create a new template with the given name, description, permissions, and
    /// graph scopes.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        permissions: Vec<Permission>,
        graph_scopes: Vec<GraphScope>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            permissions,
            graph_scopes,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Built-in template constructors
// ──────────────────────────────────────────────────────────────────────────────

/// Build the **DBA** template: full CRUD + admin/index/backup across all graphs.
fn dba_template() -> PolicyTemplate {
    PolicyTemplate::new(
        "dba",
        "Database Administrator — full CRUD, indexing, backup, and admin control",
        vec![
            Permission::Read,
            Permission::Write,
            Permission::Admin,
            Permission::GlobalAdmin,
            Permission::GlobalRead,
            Permission::GlobalWrite,
            Permission::DatasetCreate,
            Permission::DatasetDelete,
            Permission::DatasetManage,
            Permission::UserManage,
            Permission::SystemConfig,
            Permission::QueryExecute,
            Permission::UpdateExecute,
            Permission::SparqlQuery,
            Permission::SparqlUpdate,
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
        ],
        vec![GraphScope::AllNamedGraphs],
    )
}

/// Build the **ReadOnly** template: SELECT/ASK/CONSTRUCT/DESCRIBE; no writes,
/// no admin, no schema modification.
fn readonly_template() -> PolicyTemplate {
    PolicyTemplate::new(
        "readonly",
        "Read-only analyst — SPARQL SELECT/ASK/CONSTRUCT/DESCRIBE only; no write or admin access",
        vec![
            Permission::Read,
            Permission::GlobalRead,
            Permission::QueryExecute,
            Permission::SparqlQuery,
            Permission::Download,
        ],
        vec![GraphScope::AllNamedGraphs],
    )
}

/// Build the **Auditor** template: read access to data plus audit-log read.
fn auditor_template() -> PolicyTemplate {
    PolicyTemplate::new(
        "auditor",
        "Auditor — read access to all named graphs plus audit-log inspection",
        vec![
            Permission::Read,
            Permission::GlobalRead,
            Permission::QueryExecute,
            Permission::SparqlQuery,
            Permission::Download,
            Permission::Audit,
            Permission::ReadAudit,
        ],
        vec![GraphScope::AllNamedGraphs],
    )
}

// ──────────────────────────────────────────────────────────────────────────────
// PolicyTemplateRegistry
// ──────────────────────────────────────────────────────────────────────────────

/// Thread-local, owned registry of [`PolicyTemplate`] entries.
///
/// The registry is pre-populated with the three built-in templates (`dba`,
/// `readonly`, `auditor`) by calling [`PolicyTemplateRegistry::with_defaults`].
/// Custom templates can be added via [`PolicyTemplateRegistry::register`].
///
/// # Mutability
///
/// The registry is not `Clone` by design — it is expected to be wrapped in an
/// `Arc<RwLock<_>>` by the caller if it needs to be shared across threads.
pub struct PolicyTemplateRegistry {
    templates: HashMap<String, PolicyTemplate>,
}

impl PolicyTemplateRegistry {
    /// Create a registry that contains **only** the three default templates
    /// (`dba`, `readonly`, `auditor`).
    pub fn with_defaults() -> Self {
        let mut templates = HashMap::with_capacity(3);
        for tmpl in [dba_template(), readonly_template(), auditor_template()] {
            templates.insert(tmpl.name.clone(), tmpl);
        }
        Self { templates }
    }

    /// Register a custom [`PolicyTemplate`].
    ///
    /// # Errors
    ///
    /// Returns [`AuthError::DuplicateTemplate`] when a template with the same
    /// name is already present in the registry. Use [`PolicyTemplateRegistry::get`]
    /// to check for existence before calling this method if upsert semantics are
    /// needed.
    pub fn register(&mut self, template: PolicyTemplate) -> Result<(), AuthError> {
        if self.templates.contains_key(&template.name) {
            return Err(AuthError::DuplicateTemplate(template.name.clone()));
        }
        self.templates.insert(template.name.clone(), template);
        Ok(())
    }

    /// Look up a template by name.
    ///
    /// Returns `None` when no template with the given `name` exists.
    pub fn get(&self, name: &str) -> Option<&PolicyTemplate> {
        self.templates.get(name)
    }

    /// Return a sorted list of all registered templates.
    ///
    /// The order is deterministic (sorted by template name) so that callers
    /// can display or iterate predictably.
    pub fn list(&self) -> Vec<&PolicyTemplate> {
        let mut items: Vec<&PolicyTemplate> = self.templates.values().collect();
        items.sort_by(|a, b| a.name.cmp(&b.name));
        items
    }

    /// Apply a named template to a [`User`], extending their permission set.
    ///
    /// Permissions from the template that the user already holds are **not**
    /// duplicated — the method deduplicates before extending.
    ///
    /// # Errors
    ///
    /// Returns [`AuthError::UnknownTemplate`] when `template_name` does not
    /// resolve to a registered template.
    pub fn apply_to_user(&self, user: &mut User, template_name: &str) -> Result<(), AuthError> {
        let template = self
            .templates
            .get(template_name)
            .ok_or_else(|| AuthError::UnknownTemplate(template_name.to_string()))?;

        for permission in &template.permissions {
            if !user.permissions.contains(permission) {
                user.permissions.push(permission.clone());
            }
        }
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::types::{AuthError, Permission, User};
    use chrono::Utc;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_empty_user(username: &str) -> User {
        User {
            username: username.to_string(),
            roles: vec![],
            email: None,
            full_name: None,
            last_login: Some(Utc::now()),
            permissions: vec![],
        }
    }

    fn make_user_with_permissions(username: &str, permissions: Vec<Permission>) -> User {
        User {
            username: username.to_string(),
            roles: vec![],
            email: None,
            full_name: None,
            last_login: Some(Utc::now()),
            permissions,
        }
    }

    // ── 1. with_defaults returns exactly 3 templates ──────────────────────────

    #[test]
    fn test_with_defaults_has_three_templates() {
        let registry = PolicyTemplateRegistry::with_defaults();
        assert_eq!(registry.list().len(), 3);
    }

    // ── 2. get("dba") returns Some ────────────────────────────────────────────

    #[test]
    fn test_get_dba_returns_some() {
        let registry = PolicyTemplateRegistry::with_defaults();
        assert!(registry.get("dba").is_some());
    }

    // ── 3. get("readonly") returns Some ──────────────────────────────────────

    #[test]
    fn test_get_readonly_returns_some() {
        let registry = PolicyTemplateRegistry::with_defaults();
        assert!(registry.get("readonly").is_some());
    }

    // ── 4. get("auditor") returns Some ───────────────────────────────────────

    #[test]
    fn test_get_auditor_returns_some() {
        let registry = PolicyTemplateRegistry::with_defaults();
        assert!(registry.get("auditor").is_some());
    }

    // ── 5. get("nonexistent") returns None ───────────────────────────────────

    #[test]
    fn test_get_nonexistent_returns_none() {
        let registry = PolicyTemplateRegistry::with_defaults();
        assert!(registry.get("nonexistent").is_none());
    }

    // ── 6. register duplicate name returns Err(DuplicateTemplate) ────────────

    #[test]
    fn test_register_duplicate_returns_err() {
        let mut registry = PolicyTemplateRegistry::with_defaults();
        let duplicate = PolicyTemplate::new(
            "dba",
            "Another DBA template",
            vec![Permission::Read],
            vec![GraphScope::AllNamedGraphs],
        );
        let result = registry.register(duplicate);
        assert!(
            matches!(result, Err(AuthError::DuplicateTemplate(ref name)) if name == "dba"),
            "Expected DuplicateTemplate(\"dba\"), got {:?}",
            result
        );
    }

    // ── 7. register with new unique name returns Ok ───────────────────────────

    #[test]
    fn test_register_unique_name_returns_ok() {
        let mut registry = PolicyTemplateRegistry::with_defaults();
        let custom = PolicyTemplate::new(
            "custom_role",
            "A custom policy template",
            vec![Permission::Read, Permission::Monitor],
            vec![GraphScope::specific("http://example.org/g1")],
        );
        assert!(registry.register(custom).is_ok());
        assert!(registry.get("custom_role").is_some());
    }

    // ── 8. list returns all 3 default templates ───────────────────────────────

    #[test]
    fn test_list_returns_all_three_defaults() {
        let registry = PolicyTemplateRegistry::with_defaults();
        let names: Vec<&str> = registry.list().iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"dba"));
        assert!(names.contains(&"readonly"));
        assert!(names.contains(&"auditor"));
    }

    // ── 9. DBA template includes Write and Admin ──────────────────────────────

    #[test]
    fn test_dba_has_write_and_admin() {
        let registry = PolicyTemplateRegistry::with_defaults();
        let dba = registry.get("dba").expect("dba template must exist");
        assert!(
            dba.permissions.contains(&Permission::Write),
            "DBA template must contain Permission::Write"
        );
        assert!(
            dba.permissions.contains(&Permission::Admin),
            "DBA template must contain Permission::Admin"
        );
    }

    // ── 10. ReadOnly template does NOT include Write ──────────────────────────

    #[test]
    fn test_readonly_does_not_include_write() {
        let registry = PolicyTemplateRegistry::with_defaults();
        let readonly = registry
            .get("readonly")
            .expect("readonly template must exist");
        assert!(
            !readonly.permissions.contains(&Permission::Write),
            "ReadOnly template must NOT contain Permission::Write"
        );
        assert!(
            !readonly.permissions.contains(&Permission::Admin),
            "ReadOnly template must NOT contain Permission::Admin"
        );
    }

    // ── 11. Auditor template includes ReadAudit ───────────────────────────────

    #[test]
    fn test_auditor_includes_read_audit() {
        let registry = PolicyTemplateRegistry::with_defaults();
        let auditor = registry
            .get("auditor")
            .expect("auditor template must exist");
        assert!(
            auditor.permissions.contains(&Permission::ReadAudit),
            "Auditor template must contain Permission::ReadAudit"
        );
        assert!(
            auditor.permissions.contains(&Permission::Audit),
            "Auditor template must contain Permission::Audit"
        );
    }

    // ── 12. apply_to_user with valid template mutates user permissions ────────

    #[test]
    fn test_apply_to_user_grants_permissions() {
        let registry = PolicyTemplateRegistry::with_defaults();
        let mut user = make_empty_user("alice");
        registry
            .apply_to_user(&mut user, "readonly")
            .expect("apply_to_user with 'readonly' must succeed");

        assert!(
            user.permissions.contains(&Permission::Read),
            "User must have Read after applying readonly template"
        );
        assert!(
            user.permissions.contains(&Permission::QueryExecute),
            "User must have QueryExecute after applying readonly template"
        );
        assert!(
            !user.permissions.contains(&Permission::Write),
            "User must NOT have Write after applying readonly template"
        );
    }

    // ── 13. apply_to_user with unknown template returns Err(UnknownTemplate) ──

    #[test]
    fn test_apply_to_user_unknown_template_returns_err() {
        let registry = PolicyTemplateRegistry::with_defaults();
        let mut user = make_empty_user("bob");
        let result = registry.apply_to_user(&mut user, "does_not_exist");
        assert!(
            matches!(result, Err(AuthError::UnknownTemplate(ref name)) if name == "does_not_exist"),
            "Expected UnknownTemplate(\"does_not_exist\"), got {:?}",
            result
        );
    }

    // ── 14. serialize / deserialize round-trip of PolicyTemplate ─────────────

    #[test]
    fn test_policy_template_serde_roundtrip() {
        let original = PolicyTemplate::new(
            "test_template",
            "Round-trip serialization test",
            vec![Permission::Read, Permission::Write, Permission::ReadAudit],
            vec![
                GraphScope::AllNamedGraphs,
                GraphScope::specific("http://example.org/graph/test"),
            ],
        );

        let json = serde_json::to_string(&original).expect("serialization must succeed");
        let restored: PolicyTemplate =
            serde_json::from_str(&json).expect("deserialization must succeed");

        assert_eq!(restored.name, original.name);
        assert_eq!(restored.description, original.description);
        assert_eq!(restored.permissions.len(), original.permissions.len());
        assert_eq!(restored.graph_scopes.len(), original.graph_scopes.len());
        assert!(restored.permissions.contains(&Permission::ReadAudit));
        assert!(restored.graph_scopes.contains(&GraphScope::AllNamedGraphs));
    }

    // ── 15. apply_to_user deduplicates existing permissions ──────────────────

    #[test]
    fn test_apply_to_user_no_duplicate_permissions() {
        let registry = PolicyTemplateRegistry::with_defaults();
        // User already has Read permission
        let mut user = make_user_with_permissions("carol", vec![Permission::Read]);
        registry
            .apply_to_user(&mut user, "readonly")
            .expect("apply_to_user must succeed");

        // Read should appear exactly once
        let read_count = user
            .permissions
            .iter()
            .filter(|p| **p == Permission::Read)
            .count();
        assert_eq!(read_count, 1, "Read permission must not be duplicated");
    }

    // ── 16. DBA template applies all admin-class permissions ─────────────────

    #[test]
    fn test_apply_dba_template_grants_full_access() {
        let registry = PolicyTemplateRegistry::with_defaults();
        let mut user = make_empty_user("dba_user");
        registry
            .apply_to_user(&mut user, "dba")
            .expect("apply_to_user with 'dba' must succeed");

        let required = [
            Permission::Read,
            Permission::Write,
            Permission::Admin,
            Permission::Backup,
            Permission::Restore,
            Permission::UserManage,
            Permission::SystemConfig,
        ];
        for perm in &required {
            assert!(
                user.permissions.contains(perm),
                "DBA user must have {:?}",
                perm
            );
        }
    }

    // ── 17. list is sorted deterministically ─────────────────────────────────

    #[test]
    fn test_list_is_sorted_by_name() {
        let registry = PolicyTemplateRegistry::with_defaults();
        let names: Vec<&str> = registry.list().iter().map(|t| t.name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(
            names, sorted,
            "list() must return templates in sorted order"
        );
    }

    // ── 18. GraphScope SpecificGraph stores IRI correctly ────────────────────

    #[test]
    fn test_graph_scope_specific_stores_iri() {
        let scope = GraphScope::specific("http://example.org/my-graph");
        match &scope {
            GraphScope::SpecificGraph { iri } => {
                assert_eq!(iri, "http://example.org/my-graph");
            }
            GraphScope::AllNamedGraphs => panic!("Expected SpecificGraph variant"),
        }
    }
}
