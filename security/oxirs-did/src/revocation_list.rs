//! # DID Credential Revocation List
//!
//! An in-memory credential revocation list keyed by credential ID.
//! Provides revoke/unrevoke/check operations and a simplified JSON export.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A record of a single revocation event.
#[derive(Debug, Clone)]
pub struct RevocationEntry {
    /// ID of the revoked credential.
    pub credential_id: String,
    /// Unix timestamp (milliseconds) when the credential was revoked.
    pub revoked_at: u64,
    /// Optional human-readable reason for revocation.
    pub reason: Option<String>,
}

/// An in-memory revocation list associated with a DID issuer.
pub struct RevocationList {
    list_id: String,
    issuer_did: String,
    entries: HashMap<String, RevocationEntry>,
    created_at: u64,
    updated_at: u64,
}

/// The result of checking whether a credential has been revoked.
#[derive(Debug, Clone)]
pub struct RevocationCheck {
    /// The credential that was checked.
    pub credential_id: String,
    /// Whether the credential is currently revoked.
    pub revoked: bool,
    /// When it was revoked (milliseconds since epoch), if applicable.
    pub revoked_at: Option<u64>,
    /// Reason for revocation, if provided.
    pub reason: Option<String>,
}

// ---------------------------------------------------------------------------
// RevocationList implementation
// ---------------------------------------------------------------------------

impl RevocationList {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Create a new, empty revocation list.
    ///
    /// * `list_id`    – A unique identifier for this list.
    /// * `issuer_did` – The DID of the issuer that controls this list.
    /// * `now_ms`     – Current Unix time in milliseconds.
    pub fn new(
        list_id: impl Into<String>,
        issuer_did: impl Into<String>,
        now_ms: u64,
    ) -> Self {
        Self {
            list_id: list_id.into(),
            issuer_did: issuer_did.into(),
            entries: HashMap::new(),
            created_at: now_ms,
            updated_at: now_ms,
        }
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// The unique identifier for this revocation list.
    pub fn list_id(&self) -> &str {
        &self.list_id
    }

    /// The issuer DID that controls this list.
    pub fn issuer_did(&self) -> &str {
        &self.issuer_did
    }

    /// Unix timestamp (ms) when this list was created.
    pub fn created_at(&self) -> u64 {
        self.created_at
    }

    /// Unix timestamp (ms) of the most recent modification.
    pub fn updated_at(&self) -> u64 {
        self.updated_at
    }

    // ------------------------------------------------------------------
    // Mutation
    // ------------------------------------------------------------------

    /// Revoke the credential identified by `credential_id`.
    ///
    /// If the credential is already revoked, the entry is updated with the
    /// new timestamp and reason.
    pub fn revoke(
        &mut self,
        credential_id: impl Into<String>,
        reason: Option<String>,
        now_ms: u64,
    ) {
        let id: String = credential_id.into();
        self.entries.insert(
            id.clone(),
            RevocationEntry {
                credential_id: id,
                revoked_at: now_ms,
                reason,
            },
        );
        self.updated_at = now_ms;
    }

    /// Remove the revocation for `credential_id`.
    ///
    /// Returns `true` if the credential was revoked and is now un-revoked,
    /// or `false` if it was not in the list.
    pub fn unrevoke(&mut self, credential_id: &str) -> bool {
        let removed = self.entries.remove(credential_id).is_some();
        if removed {
            self.updated_at = self.updated_at.saturating_add(1);
        }
        removed
    }

    // ------------------------------------------------------------------
    // Query
    // ------------------------------------------------------------------

    /// Check whether `credential_id` is revoked and return detailed status.
    pub fn check(&self, credential_id: &str) -> RevocationCheck {
        match self.entries.get(credential_id) {
            Some(entry) => RevocationCheck {
                credential_id: credential_id.to_string(),
                revoked: true,
                revoked_at: Some(entry.revoked_at),
                reason: entry.reason.clone(),
            },
            None => RevocationCheck {
                credential_id: credential_id.to_string(),
                revoked: false,
                revoked_at: None,
                reason: None,
            },
        }
    }

    /// Returns `true` if `credential_id` is currently revoked.
    pub fn is_revoked(&self, credential_id: &str) -> bool {
        self.entries.contains_key(credential_id)
    }

    /// Total number of revoked credentials.
    pub fn revoked_count(&self) -> usize {
        self.entries.len()
    }

    /// All revocation entries, in arbitrary order.
    pub fn all_revoked(&self) -> Vec<&RevocationEntry> {
        self.entries.values().collect()
    }

    /// All revocation entries where `revoked_at >= since_ms`.
    pub fn revoked_since(&self, since_ms: u64) -> Vec<&RevocationEntry> {
        self.entries
            .values()
            .filter(|e| e.revoked_at >= since_ms)
            .collect()
    }

    // ------------------------------------------------------------------
    // Serialisation
    // ------------------------------------------------------------------

    /// Produce a simplified JSON representation of the revocation list.
    pub fn to_json(&self) -> String {
        let entries_json: Vec<String> = self
            .entries
            .values()
            .map(|e| {
                let reason_part = match &e.reason {
                    Some(r) => format!(r#","reason":"{}""#, escape_json_string(r)),
                    None => String::new(),
                };
                format!(
                    r#"{{"credentialId":"{}","revokedAt":{}{}}}"#,
                    escape_json_string(&e.credential_id),
                    e.revoked_at,
                    reason_part
                )
            })
            .collect();

        format!(
            r#"{{"listId":"{}","issuerDid":"{}","createdAt":{},"updatedAt":{},"entries":[{}]}}"#,
            escape_json_string(&self.list_id),
            escape_json_string(&self.issuer_did),
            self.created_at,
            self.updated_at,
            entries_json.join(",")
        )
    }
}

// ---------------------------------------------------------------------------
// Helper: minimal JSON string escaping (no external dep required)
// ---------------------------------------------------------------------------

fn escape_json_string(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            '"' => vec!['\\', '"'],
            '\\' => vec!['\\', '\\'],
            '\n' => vec!['\\', 'n'],
            '\r' => vec!['\\', 'r'],
            '\t' => vec!['\\', 't'],
            other => vec![other],
        })
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_list() -> RevocationList {
        RevocationList::new("list-001", "did:example:issuer", 1_000_000)
    }

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------
    #[test]
    fn test_new_list_id() {
        let rl = make_list();
        assert_eq!(rl.list_id(), "list-001");
    }

    #[test]
    fn test_new_issuer_did() {
        let rl = make_list();
        assert_eq!(rl.issuer_did(), "did:example:issuer");
    }

    #[test]
    fn test_new_created_at() {
        let rl = make_list();
        assert_eq!(rl.created_at(), 1_000_000);
    }

    #[test]
    fn test_new_updated_at_equals_created_at() {
        let rl = make_list();
        assert_eq!(rl.updated_at(), rl.created_at());
    }

    #[test]
    fn test_new_empty_count() {
        let rl = make_list();
        assert_eq!(rl.revoked_count(), 0);
    }

    // -------------------------------------------------------------------------
    // revoke
    // -------------------------------------------------------------------------
    #[test]
    fn test_revoke_marks_credential() {
        let mut rl = make_list();
        rl.revoke("cred-001", None, 2_000_000);
        assert!(rl.is_revoked("cred-001"));
    }

    #[test]
    fn test_revoke_increments_count() {
        let mut rl = make_list();
        rl.revoke("cred-001", None, 2_000_000);
        assert_eq!(rl.revoked_count(), 1);
    }

    #[test]
    fn test_revoke_updates_updated_at() {
        let mut rl = make_list();
        rl.revoke("cred-001", None, 2_000_000);
        assert_eq!(rl.updated_at(), 2_000_000);
    }

    #[test]
    fn test_revoke_with_reason() {
        let mut rl = make_list();
        rl.revoke("cred-002", Some("compromised".to_string()), 2_000_000);
        let check = rl.check("cred-002");
        assert_eq!(check.reason, Some("compromised".to_string()));
    }

    #[test]
    fn test_revoke_twice_updates_entry() {
        let mut rl = make_list();
        rl.revoke("cred-001", None, 1_500_000);
        rl.revoke("cred-001", Some("key rotation".to_string()), 2_000_000);
        let check = rl.check("cred-001");
        assert_eq!(check.revoked_at, Some(2_000_000));
        assert_eq!(check.reason.as_deref(), Some("key rotation"));
    }

    #[test]
    fn test_revoke_multiple_credentials() {
        let mut rl = make_list();
        rl.revoke("cred-001", None, 1_000_001);
        rl.revoke("cred-002", None, 1_000_002);
        rl.revoke("cred-003", None, 1_000_003);
        assert_eq!(rl.revoked_count(), 3);
    }

    // -------------------------------------------------------------------------
    // unrevoke
    // -------------------------------------------------------------------------
    #[test]
    fn test_unrevoke_existing() {
        let mut rl = make_list();
        rl.revoke("cred-001", None, 1_500_000);
        let result = rl.unrevoke("cred-001");
        assert!(result);
        assert!(!rl.is_revoked("cred-001"));
    }

    #[test]
    fn test_unrevoke_nonexistent() {
        let mut rl = make_list();
        let result = rl.unrevoke("cred-999");
        assert!(!result);
    }

    #[test]
    fn test_unrevoke_decrements_count() {
        let mut rl = make_list();
        rl.revoke("cred-001", None, 1_500_000);
        rl.unrevoke("cred-001");
        assert_eq!(rl.revoked_count(), 0);
    }

    // -------------------------------------------------------------------------
    // check
    // -------------------------------------------------------------------------
    #[test]
    fn test_check_revoked() {
        let mut rl = make_list();
        rl.revoke("cred-001", Some("expired".into()), 1_500_000);
        let c = rl.check("cred-001");
        assert!(c.revoked);
        assert_eq!(c.revoked_at, Some(1_500_000));
        assert_eq!(c.reason.as_deref(), Some("expired"));
    }

    #[test]
    fn test_check_not_revoked() {
        let rl = make_list();
        let c = rl.check("cred-missing");
        assert!(!c.revoked);
        assert!(c.revoked_at.is_none());
        assert!(c.reason.is_none());
    }

    #[test]
    fn test_check_credential_id_preserved() {
        let rl = make_list();
        let c = rl.check("my-cred");
        assert_eq!(c.credential_id, "my-cred");
    }

    // -------------------------------------------------------------------------
    // is_revoked
    // -------------------------------------------------------------------------
    #[test]
    fn test_is_revoked_true() {
        let mut rl = make_list();
        rl.revoke("x", None, 1);
        assert!(rl.is_revoked("x"));
    }

    #[test]
    fn test_is_revoked_false() {
        let rl = make_list();
        assert!(!rl.is_revoked("x"));
    }

    // -------------------------------------------------------------------------
    // all_revoked
    // -------------------------------------------------------------------------
    #[test]
    fn test_all_revoked_empty() {
        let rl = make_list();
        assert!(rl.all_revoked().is_empty());
    }

    #[test]
    fn test_all_revoked_count() {
        let mut rl = make_list();
        rl.revoke("a", None, 1);
        rl.revoke("b", None, 2);
        assert_eq!(rl.all_revoked().len(), 2);
    }

    #[test]
    fn test_all_revoked_contains_ids() {
        let mut rl = make_list();
        rl.revoke("cred-x", None, 1);
        let ids: Vec<&str> = rl.all_revoked().iter().map(|e| e.credential_id.as_str()).collect();
        assert!(ids.contains(&"cred-x"));
    }

    // -------------------------------------------------------------------------
    // revoked_since
    // -------------------------------------------------------------------------
    #[test]
    fn test_revoked_since_basic() {
        let mut rl = make_list();
        rl.revoke("old", None, 500_000);
        rl.revoke("new", None, 2_000_000);
        let recent = rl.revoked_since(1_000_000);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].credential_id, "new");
    }

    #[test]
    fn test_revoked_since_all() {
        let mut rl = make_list();
        rl.revoke("a", None, 1_000_001);
        rl.revoke("b", None, 1_000_002);
        let all = rl.revoked_since(0);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_revoked_since_none() {
        let mut rl = make_list();
        rl.revoke("a", None, 500);
        let none = rl.revoked_since(1_000_000);
        assert!(none.is_empty());
    }

    #[test]
    fn test_revoked_since_inclusive() {
        let mut rl = make_list();
        rl.revoke("exact", None, 1_000_000);
        let result = rl.revoked_since(1_000_000);
        assert_eq!(result.len(), 1);
    }

    // -------------------------------------------------------------------------
    // to_json
    // -------------------------------------------------------------------------
    #[test]
    fn test_to_json_contains_list_id() {
        let rl = make_list();
        let json = rl.to_json();
        assert!(json.contains("list-001"), "json={json}");
    }

    #[test]
    fn test_to_json_contains_issuer_did() {
        let rl = make_list();
        let json = rl.to_json();
        assert!(json.contains("did:example:issuer"), "json={json}");
    }

    #[test]
    fn test_to_json_empty_entries() {
        let rl = make_list();
        let json = rl.to_json();
        assert!(json.contains("\"entries\":[]"), "json={json}");
    }

    #[test]
    fn test_to_json_with_entry() {
        let mut rl = make_list();
        rl.revoke("vc-123", Some("stolen".into()), 1_500_000);
        let json = rl.to_json();
        assert!(json.contains("vc-123"), "json={json}");
        assert!(json.contains("stolen"), "json={json}");
    }

    #[test]
    fn test_to_json_timestamps() {
        let rl = make_list();
        let json = rl.to_json();
        assert!(json.contains("1000000"), "json={json}");
    }

    #[test]
    fn test_to_json_no_unwrap_panics() {
        let mut rl = RevocationList::new("l", "d", 0);
        for i in 0..10 {
            rl.revoke(format!("cred-{i}"), Some(format!("reason-{i}")), i as u64 * 1000);
        }
        let json = rl.to_json();
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
    }

    // -------------------------------------------------------------------------
    // escape_json_string helper
    // -------------------------------------------------------------------------
    #[test]
    fn test_escape_json_quotes() {
        let escaped = escape_json_string(r#"He said "hello""#);
        assert!(escaped.contains(r#"\""#));
    }

    #[test]
    fn test_escape_json_backslash() {
        let escaped = escape_json_string(r"path\to\file");
        assert!(escaped.contains(r"\\"));
    }

    #[test]
    fn test_escape_json_newline() {
        let escaped = escape_json_string("line1\nline2");
        assert!(escaped.contains(r"\n"));
    }

    #[test]
    fn test_revocation_entry_fields() {
        let e = RevocationEntry {
            credential_id: "c".into(),
            revoked_at: 99,
            reason: Some("r".into()),
        };
        assert_eq!(e.credential_id, "c");
        assert_eq!(e.revoked_at, 99);
        assert_eq!(e.reason.as_deref(), Some("r"));
    }

    #[test]
    fn test_revocation_check_fields() {
        let c = RevocationCheck {
            credential_id: "x".into(),
            revoked: true,
            revoked_at: Some(42),
            reason: None,
        };
        assert_eq!(c.credential_id, "x");
        assert!(c.revoked);
        assert_eq!(c.revoked_at, Some(42));
    }
}
