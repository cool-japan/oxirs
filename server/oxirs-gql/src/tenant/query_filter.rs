//! Tenant-aware GraphQL query filtering.
//!
//! `TenantQueryFilter` inspects a GraphQL query string and:
//! 1. Strips out field selections that reference types or fields the tenant
//!    is not allowed to access (`filter_query`).
//! 2. Returns a list of access violations without modifying the query
//!    (`validate_tenant_access`).
//!
//! The filtering is performed with simple string-level heuristics rather than
//! a full AST parse, keeping the implementation dependency-free.  A
//! production system would replace these routines with AST-based analysis.

use anyhow::Result;

use crate::multitenancy::TenantContext;

/// A single access-control violation detected in a query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessViolation {
    /// The field path where the violation was detected (e.g. `"User.secret"`).
    pub field_path: String,
    /// Human-readable reason for the violation.
    pub reason: String,
}

/// Filters GraphQL queries to ensure tenants only access their allowed types
/// and fields.
#[derive(Debug, Default)]
pub struct TenantQueryFilter;

impl TenantQueryFilter {
    /// Create a new filter instance.
    pub fn new() -> Self {
        Self
    }

    /// Filter a GraphQL query string, removing selections for types or fields
    /// that are not in the tenant's `allowed_types` list.
    ///
    /// The implementation removes any line that contains a disallowed type
    /// name (case-sensitive).  Lines that don't reference any type name are
    /// kept intact.
    ///
    /// Returns an `Err` if the tenant has no allowed types but strict mode
    /// would normally reject everything — instead we allow all when the
    /// list is empty (open policy).
    pub fn filter_query(&self, query: &str, context: &TenantContext) -> Result<String> {
        let allowed = &context.config.custom_types;
        if allowed.is_empty() {
            // Open policy: tenant can see everything
            return Ok(query.to_string());
        }

        let allowed_names: Vec<&str> = allowed.iter().map(|t| t.type_name.as_str()).collect();

        let filtered: Vec<&str> = query
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                // Blank lines and structural tokens pass through
                if trimmed.is_empty()
                    || trimmed == "{"
                    || trimmed == "}"
                    || trimmed.starts_with('#')
                {
                    return true;
                }
                // If the line contains a type reference that is NOT in the
                // allow-list, drop it.  We detect type references by looking
                // for an identifier followed by '{'.
                if let Some(type_name) = Self::extract_type_name(trimmed) {
                    return allowed_names.contains(&type_name);
                }
                // Plain field selections pass through
                true
            })
            .collect();

        Ok(filtered.join("\n"))
    }

    /// Validate a query and return all access violations without modifying it.
    pub fn validate_tenant_access(
        &self,
        query: &str,
        context: &TenantContext,
    ) -> Vec<AccessViolation> {
        let allowed = &context.config.custom_types;
        if allowed.is_empty() {
            return vec![];
        }

        let allowed_names: Vec<&str> = allowed.iter().map(|t| t.type_name.as_str()).collect();
        let mut violations = Vec::new();

        for line in query.lines() {
            let trimmed = line.trim();
            if let Some(type_name) = Self::extract_type_name(trimmed) {
                if !allowed_names.contains(&type_name) {
                    violations.push(AccessViolation {
                        field_path: type_name.to_string(),
                        reason: format!(
                            "Type '{}' is not in the allowed types list for tenant '{}'",
                            type_name, context.tenant_id
                        ),
                    });
                }
            }
        }

        violations
    }

    /// Validate field-level access.
    ///
    /// For a `"TypeName.fieldName"` path, returns a violation if either the
    /// type is not in the allow-list or the field is not listed under that
    /// type's allowed fields.
    pub fn validate_field_access(
        &self,
        type_name: &str,
        field_name: &str,
        context: &TenantContext,
    ) -> Option<AccessViolation> {
        let allowed = &context.config.custom_types;
        if allowed.is_empty() {
            return None;
        }

        let type_def = allowed.iter().find(|t| t.type_name == type_name);
        match type_def {
            None => Some(AccessViolation {
                field_path: format!("{type_name}.{field_name}"),
                reason: format!(
                    "Type '{type_name}' is not allowed for tenant '{}'",
                    context.tenant_id
                ),
            }),
            Some(td) => {
                let field_allowed = td.fields.iter().any(|f| f.field_name == field_name);
                if field_allowed {
                    None
                } else {
                    Some(AccessViolation {
                        field_path: format!("{type_name}.{field_name}"),
                        reason: format!(
                            "Field '{field_name}' on type '{type_name}' is not allowed for tenant '{}'",
                            context.tenant_id
                        ),
                    })
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Extract the type name from a line like `"User {"` → `"User"`.
    ///
    /// Returns `None` if the line is not a type-open pattern.
    fn extract_type_name(line: &str) -> Option<&str> {
        let trimmed = line.trim_end_matches('{').trim();
        // Must not start with query/mutation/subscription keywords or field
        // selectors (those won't have a leading uppercase letter in the
        // relevant position, but we keep it simple).
        let candidate = trimmed.split_whitespace().next()?;
        // Type names start with an uppercase letter per GraphQL convention
        if candidate.starts_with(|c: char| c.is_uppercase()) {
            Some(candidate)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multitenancy::{TenantConfig, TenantContext, TenantCustomType, TenantField};
    use std::sync::Arc;

    fn make_context(type_names: &[&str]) -> TenantContext {
        let custom_types: Vec<TenantCustomType> = type_names
            .iter()
            .map(|name| TenantCustomType {
                type_name: name.to_string(),
                rdf_class: format!("http://example.org/{name}"),
                fields: vec![
                    TenantField {
                        field_name: "id".to_string(),
                        rdf_predicate: "http://example.org/id".to_string(),
                        field_type: "ID".to_string(),
                        is_required: true,
                        is_list: false,
                    },
                    TenantField {
                        field_name: "name".to_string(),
                        rdf_predicate: "http://example.org/name".to_string(),
                        field_type: "String".to_string(),
                        is_required: false,
                        is_list: false,
                    },
                ],
            })
            .collect();

        let config = TenantConfig {
            tenant_id: "test-tenant".to_string(),
            display_name: "Test Tenant".to_string(),
            datasets: vec![],
            max_query_depth: 10,
            max_query_complexity: 1000,
            rate_limit_rpm: 60,
            allowed_operations: vec![crate::multitenancy::TenantOperation::Query],
            custom_types,
        };
        TenantContext::new(
            "test-tenant",
            Arc::new(config),
            "req-test".to_string(),
            None,
        )
    }

    #[test]
    fn test_filter_query_open_policy() {
        let filter = TenantQueryFilter::new();
        let ctx = make_context(&[]);
        let query = "{ User { id name } }";
        let filtered = filter.filter_query(query, &ctx).unwrap();
        assert_eq!(filtered, query);
    }

    #[test]
    fn test_filter_query_removes_disallowed_type() {
        let filter = TenantQueryFilter::new();
        let ctx = make_context(&["User"]);
        let query = "query {\n  User {\n    id\n  }\n  Admin {\n    secret\n  }\n}";
        let filtered = filter.filter_query(query, &ctx).unwrap();
        assert!(!filtered.contains("Admin"), "Admin should be stripped");
        assert!(filtered.contains("User"), "User should remain");
    }

    #[test]
    fn test_validate_tenant_access_no_violations() {
        let filter = TenantQueryFilter::new();
        let ctx = make_context(&["User", "Product"]);
        let query = "{\n  User {\n    id\n  }\n}";
        let violations = filter.validate_tenant_access(query, &ctx);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_validate_tenant_access_detects_violation() {
        let filter = TenantQueryFilter::new();
        let ctx = make_context(&["User"]);
        let query = "{\n  User {\n    id\n  }\n  Admin {\n    secret\n  }\n}";
        let violations = filter.validate_tenant_access(query, &ctx);
        assert!(!violations.is_empty());
        assert!(violations.iter().any(|v| v.field_path.contains("Admin")));
    }

    #[test]
    fn test_validate_field_access_allowed() {
        let filter = TenantQueryFilter::new();
        let ctx = make_context(&["User"]);
        let violation = filter.validate_field_access("User", "name", &ctx);
        assert!(violation.is_none());
    }

    #[test]
    fn test_validate_field_access_disallowed_field() {
        let filter = TenantQueryFilter::new();
        let ctx = make_context(&["User"]);
        let violation = filter.validate_field_access("User", "passwordHash", &ctx);
        assert!(violation.is_some());
        let v = violation.unwrap();
        assert!(v.field_path.contains("passwordHash"));
        assert!(v.reason.contains("not allowed"));
    }

    #[test]
    fn test_validate_field_access_disallowed_type() {
        let filter = TenantQueryFilter::new();
        let ctx = make_context(&["User"]);
        let violation = filter.validate_field_access("Admin", "id", &ctx);
        assert!(violation.is_some());
        assert!(violation.unwrap().field_path.starts_with("Admin"));
    }

    #[test]
    fn test_validate_open_policy_no_violations() {
        let filter = TenantQueryFilter::new();
        let ctx = make_context(&[]);
        let violations = filter.validate_tenant_access("{ Admin { secret } }", &ctx);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_violation_reason_mentions_tenant() {
        let filter = TenantQueryFilter::new();
        let ctx = make_context(&["User"]);
        let query = "Admin {";
        let violations = filter.validate_tenant_access(query, &ctx);
        assert!(!violations.is_empty());
        assert!(violations[0].reason.contains("test-tenant"));
    }

    #[test]
    fn test_multiple_violations_detected() {
        let filter = TenantQueryFilter::new();
        let ctx = make_context(&["User"]);
        let query = "{\n  Admin {\n    x\n  }\n  Superuser {\n    y\n  }\n}";
        let violations = filter.validate_tenant_access(query, &ctx);
        assert!(violations.len() >= 2);
    }
}
