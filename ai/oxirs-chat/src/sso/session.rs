//! SSO session bridge — integrates SSO user identity with OxiRS chat session attributes.
//!
//! [`SsoSessionBridge`] converts [`SsoUserInfo`] into the flat key-value attribute map
//! expected by the session store, and provides convenience methods for group/role checks.
//!
//! # Example
//!
//! ```rust
//! use oxirs_chat::sso::oidc::SsoUserInfo;
//! use oxirs_chat::sso::session::SsoSessionBridge;
//! use std::collections::HashMap;
//!
//! let user = SsoUserInfo {
//!     subject: "user-42".to_string(),
//!     email: Some("alice@example.com".to_string()),
//!     name: Some("Alice".to_string()),
//!     groups: vec!["admins".to_string()],
//!     raw_claims: HashMap::new(),
//! };
//! let bridge = SsoSessionBridge::new(&user);
//! assert!(bridge.has_group("admins"));
//! let attrs = bridge.to_session_attributes();
//! assert_eq!(attrs.get("sso_subject").map(|s| s.as_str()), Some("user-42"));
//! ```

use std::collections::HashMap;

use super::oidc::SsoUserInfo;

// ── SsoSessionBridge ───────────────────────────────────────────────────────

/// Bridges SSO user identity to OxiRS chat session attributes.
///
/// Holds a reference to an [`SsoUserInfo`] so it can produce a flat
/// `HashMap<String, String>` for injection into the session store, as well as
/// answer group-membership queries efficiently.
pub struct SsoSessionBridge<'a> {
    user_info: &'a SsoUserInfo,
}

impl<'a> SsoSessionBridge<'a> {
    /// Create a new bridge backed by the given user info.
    pub fn new(user_info: &'a SsoUserInfo) -> Self {
        Self { user_info }
    }

    /// Convert the SSO user identity into a flat session-attribute map.
    ///
    /// The following keys are populated when the corresponding value is present:
    ///
    /// | Key | Source |
    /// |---|---|
    /// | `sso_subject` | `SsoUserInfo::subject` |
    /// | `sso_email` | `SsoUserInfo::email` |
    /// | `sso_name` | `SsoUserInfo::name` |
    /// | `sso_groups` | `SsoUserInfo::groups` joined with `,` |
    pub fn to_session_attributes(&self) -> HashMap<String, String> {
        let mut attrs = HashMap::new();

        attrs.insert("sso_subject".to_string(), self.user_info.subject.clone());

        if let Some(email) = &self.user_info.email {
            attrs.insert("sso_email".to_string(), email.clone());
        }

        if let Some(name) = &self.user_info.name {
            attrs.insert("sso_name".to_string(), name.clone());
        }

        if !self.user_info.groups.is_empty() {
            attrs.insert("sso_groups".to_string(), self.user_info.groups.join(","));
        }

        attrs
    }

    /// Return `true` if the user belongs to `group`.
    ///
    /// Comparison is case-sensitive.
    pub fn has_group(&self, group: &str) -> bool {
        self.user_info.groups.iter().any(|g| g == group)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_user(groups: Vec<&str>) -> SsoUserInfo {
        SsoUserInfo {
            subject: "sub-999".to_string(),
            email: Some("carol@example.com".to_string()),
            name: Some("Carol".to_string()),
            groups: groups.into_iter().map(|s| s.to_string()).collect(),
            raw_claims: HashMap::new(),
        }
    }

    #[test]
    fn test_session_bridge_to_attributes() {
        let user = make_user(vec!["admins", "users"]);
        let bridge = SsoSessionBridge::new(&user);
        let attrs = bridge.to_session_attributes();

        assert_eq!(
            attrs.get("sso_subject").map(|s| s.as_str()),
            Some("sub-999")
        );
        assert_eq!(
            attrs.get("sso_email").map(|s| s.as_str()),
            Some("carol@example.com")
        );
        assert_eq!(attrs.get("sso_name").map(|s| s.as_str()), Some("Carol"));

        let groups_str = attrs.get("sso_groups").expect("sso_groups must be present");
        assert!(groups_str.contains("admins"), "missing admins");
        assert!(groups_str.contains("users"), "missing users");
    }

    #[test]
    fn test_session_bridge_has_group() {
        let user = make_user(vec!["engineers", "rdf-users"]);
        let bridge = SsoSessionBridge::new(&user);

        assert!(bridge.has_group("engineers"), "should be in engineers");
        assert!(bridge.has_group("rdf-users"), "should be in rdf-users");
        assert!(!bridge.has_group("admins"), "should NOT be in admins");
        assert!(
            !bridge.has_group("Engineers"),
            "comparison is case-sensitive"
        );
    }

    #[test]
    fn test_session_bridge_no_email() {
        let user = SsoUserInfo {
            subject: "anon-sub".to_string(),
            email: None,
            name: None,
            groups: vec![],
            raw_claims: HashMap::new(),
        };
        let bridge = SsoSessionBridge::new(&user);
        let attrs = bridge.to_session_attributes();
        assert!(attrs.contains_key("sso_subject"));
        assert!(!attrs.contains_key("sso_email"), "email should be absent");
        assert!(!attrs.contains_key("sso_name"), "name should be absent");
        assert!(!attrs.contains_key("sso_groups"), "groups should be absent");
    }
}
