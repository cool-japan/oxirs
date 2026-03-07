//! # Trust Chain
//!
//! DID trust chain validation — verifies a chain of DIDs where each
//! certifies the next.  A chain is ordered from leaf → root: the first
//! element is the leaf (end-entity), and the last element must be a root
//! (no issuer, `TrustLevel::Root`).
//!
//! ## Example
//!
//! ```rust
//! use oxirs_did::trust_chain::{
//!     TrustChain, TrustChainBuilder, ChainLink, TrustLevel,
//! };
//!
//! let mut builder = TrustChainBuilder::new();
//! builder
//!     .add(ChainLink {
//!         did: "did:example:leaf".to_string(),
//!         issuer_did: Some("did:example:root".to_string()),
//!         trust_level: TrustLevel::Leaf,
//!         is_revoked: false,
//!         issued_at: 0,
//!         expires_at: None,
//!     })
//!     .add(ChainLink {
//!         did: "did:example:root".to_string(),
//!         issuer_did: None,
//!         trust_level: TrustLevel::Root,
//!         is_revoked: false,
//!         issued_at: 0,
//!         expires_at: None,
//!     });
//!
//! let chain = builder.build();
//! assert!(chain.validate(1000).is_ok());
//! ```

use std::collections::HashSet;

// ─── Trust level ──────────────────────────────────────────────────────────────

/// The trust level of a DID in the chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustLevel {
    /// A self-signed root DID (no issuer).
    Root,
    /// An intermediate CA-like DID (has an issuer and certifies others).
    Intermediate,
    /// An end-entity DID (has an issuer, does not certify others).
    Leaf,
}

// ─── Chain link ───────────────────────────────────────────────────────────────

/// A single link in a DID trust chain.
#[derive(Debug, Clone)]
pub struct ChainLink {
    /// The DID this link represents.
    pub did: String,
    /// The DID of the issuer that certified this link (`None` for roots).
    pub issuer_did: Option<String>,
    /// The trust level of this link.
    pub trust_level: TrustLevel,
    /// Whether this link has been revoked.
    pub is_revoked: bool,
    /// Milliseconds since Unix epoch when the link was issued.
    pub issued_at: u64,
    /// Optional expiry time in milliseconds since Unix epoch.
    pub expires_at: Option<u64>,
}

impl ChainLink {
    /// Returns `true` when `now_ms` is after `expires_at`.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        self.expires_at.is_some_and(|exp| now_ms > exp)
    }
}

// ─── Errors ───────────────────────────────────────────────────────────────────

/// Errors that can occur during trust chain construction or validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustChainError {
    /// The chain has a gap: a link's issuer does not match the next link's DID.
    BrokenChain(String),
    /// A link in the chain has been revoked.
    RevokedLink(String),
    /// A link in the chain has expired.
    ExpiredLink(String),
    /// The chain contains no root link.
    NoRootFound,
    /// The chain contains a cycle (the contained list shows the cycle path).
    CyclicChain(Vec<String>),
    /// A referenced DID is not present in the chain.
    UnknownDid(String),
}

impl std::fmt::Display for TrustChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrustChainError::BrokenChain(msg) => write!(f, "Broken chain: {msg}"),
            TrustChainError::RevokedLink(did) => write!(f, "Revoked link: {did}"),
            TrustChainError::ExpiredLink(did) => write!(f, "Expired link: {did}"),
            TrustChainError::NoRootFound => write!(f, "No root found in chain"),
            TrustChainError::CyclicChain(path) => write!(f, "Cyclic chain: {:?}", path),
            TrustChainError::UnknownDid(did) => write!(f, "Unknown DID: {did}"),
        }
    }
}

impl std::error::Error for TrustChainError {}

// ─── Trust chain ──────────────────────────────────────────────────────────────

/// An ordered list of [`ChainLink`]s from leaf (index 0) → root (last index).
#[derive(Debug, Clone, Default)]
pub struct TrustChain {
    links: Vec<ChainLink>,
}

impl TrustChain {
    /// Create an empty trust chain.
    pub fn new() -> Self {
        Self { links: Vec::new() }
    }

    /// Append a link to the chain.
    ///
    /// Links are expected to be added in leaf-to-root order; the method does
    /// not enforce ordering at insertion time — call `validate` to check the
    /// full chain.
    pub fn add_link(&mut self, link: ChainLink) -> Result<(), TrustChainError> {
        // Reject duplicate DIDs immediately.
        if self.links.iter().any(|l| l.did == link.did) {
            return Err(TrustChainError::CyclicChain(vec![link.did.clone()]));
        }
        self.links.push(link);
        Ok(())
    }

    /// Validate the entire chain at `now_ms` (milliseconds since Unix epoch).
    ///
    /// Checks performed (in order):
    /// 1. No revoked links.
    /// 2. No expired links.
    /// 3. Each link's `issuer_did` matches the next link's `did`.
    /// 4. The last link must be a root (`issuer_did` is `None` and
    ///    `trust_level == TrustLevel::Root`).
    /// 5. No cycles (detected via a visited set).
    pub fn validate(&self, now_ms: u64) -> Result<(), TrustChainError> {
        if self.links.is_empty() {
            return Err(TrustChainError::NoRootFound);
        }

        // Cycle detection via a set of seen DIDs.
        let mut seen: HashSet<&str> = HashSet::new();

        for link in &self.links {
            // Revocation check.
            if link.is_revoked {
                return Err(TrustChainError::RevokedLink(link.did.clone()));
            }

            // Expiry check.
            if link.is_expired(now_ms) {
                return Err(TrustChainError::ExpiredLink(link.did.clone()));
            }

            // Cycle check.
            if !seen.insert(link.did.as_str()) {
                return Err(TrustChainError::CyclicChain(vec![link.did.clone()]));
            }
        }

        // Chain continuity check: link[i].issuer_did == link[i+1].did
        for i in 0..self.links.len().saturating_sub(1) {
            let current = &self.links[i];
            let next = &self.links[i + 1];

            match &current.issuer_did {
                None => {
                    // A non-terminal link with no issuer — there should only be
                    // one root, and it must be the last link.
                    return Err(TrustChainError::BrokenChain(format!(
                        "Link '{}' has no issuer but is not the root (last) element",
                        current.did
                    )));
                }
                Some(issuer) => {
                    if issuer != &next.did {
                        return Err(TrustChainError::BrokenChain(format!(
                            "Link '{}' claims issuer '{}' but next link is '{}'",
                            current.did, issuer, next.did
                        )));
                    }
                }
            }
        }

        // Root check: the last link must have no issuer and be TrustLevel::Root.
        let root = self
            .links
            .last()
            .expect("non-empty chain has a last element");
        if root.trust_level != TrustLevel::Root {
            return Err(TrustChainError::NoRootFound);
        }
        if root.issuer_did.is_some() {
            return Err(TrustChainError::BrokenChain(format!(
                "Root link '{}' must not have an issuer",
                root.did
            )));
        }

        Ok(())
    }

    /// Return the number of links in the chain.
    pub fn chain_length(&self) -> usize {
        self.links.len()
    }

    /// Return the root link (last element), if any.
    pub fn root(&self) -> Option<&ChainLink> {
        self.links.last()
    }

    /// Return the leaf link (first element), if any.
    pub fn leaf(&self) -> Option<&ChainLink> {
        self.links.first()
    }

    /// Return `true` if the chain contains a link for the given DID.
    pub fn contains(&self, did: &str) -> bool {
        self.links.iter().any(|l| l.did == did)
    }

    /// Iterate over all links (leaf → root order).
    pub fn iter(&self) -> impl Iterator<Item = &ChainLink> {
        self.links.iter()
    }
}

// ─── Builder ──────────────────────────────────────────────────────────────────

/// Fluent builder for [`TrustChain`].
#[derive(Debug, Default)]
pub struct TrustChainBuilder {
    links: Vec<ChainLink>,
}

impl TrustChainBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self { links: Vec::new() }
    }

    /// Append a [`ChainLink`] and return `&mut Self` for chaining.
    pub fn add(&mut self, link: ChainLink) -> &mut Self {
        self.links.push(link);
        self
    }

    /// Consume the builder and produce a [`TrustChain`].
    ///
    /// Note: this does *not* validate the chain; call
    /// [`TrustChain::validate`] after building.
    pub fn build(self) -> TrustChain {
        TrustChain { links: self.links }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Convenience constructor for a root [`ChainLink`].
pub fn root_link(did: &str, issued_at: u64) -> ChainLink {
    ChainLink {
        did: did.to_string(),
        issuer_did: None,
        trust_level: TrustLevel::Root,
        is_revoked: false,
        issued_at,
        expires_at: None,
    }
}

/// Convenience constructor for an intermediate [`ChainLink`].
pub fn intermediate_link(did: &str, issuer_did: &str, issued_at: u64) -> ChainLink {
    ChainLink {
        did: did.to_string(),
        issuer_did: Some(issuer_did.to_string()),
        trust_level: TrustLevel::Intermediate,
        is_revoked: false,
        issued_at,
        expires_at: None,
    }
}

/// Convenience constructor for a leaf [`ChainLink`].
pub fn leaf_link(did: &str, issuer_did: &str, issued_at: u64) -> ChainLink {
    ChainLink {
        did: did.to_string(),
        issuer_did: Some(issuer_did.to_string()),
        trust_level: TrustLevel::Leaf,
        is_revoked: false,
        issued_at,
        expires_at: None,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_root(did: &str) -> ChainLink {
        root_link(did, 0)
    }

    fn make_intermediate(did: &str, issuer: &str) -> ChainLink {
        intermediate_link(did, issuer, 0)
    }

    fn make_leaf(did: &str, issuer: &str) -> ChainLink {
        leaf_link(did, issuer, 0)
    }

    fn simple_two_link_chain() -> TrustChain {
        let mut chain = TrustChain::new();
        chain
            .add_link(make_leaf("did:ex:leaf", "did:ex:root"))
            .expect("add leaf");
        chain.add_link(make_root("did:ex:root")).expect("add root");
        chain
    }

    fn three_link_chain() -> TrustChain {
        let mut chain = TrustChain::new();
        chain
            .add_link(make_leaf("did:ex:leaf", "did:ex:mid"))
            .expect("add leaf");
        chain
            .add_link(make_intermediate("did:ex:mid", "did:ex:root"))
            .expect("add mid");
        chain.add_link(make_root("did:ex:root")).expect("add root");
        chain
    }

    // ── TrustLevel ───────────────────────────────────────────────────────────

    #[test]
    fn test_trust_level_debug() {
        assert_eq!(format!("{:?}", TrustLevel::Root), "Root");
        assert_eq!(format!("{:?}", TrustLevel::Intermediate), "Intermediate");
        assert_eq!(format!("{:?}", TrustLevel::Leaf), "Leaf");
    }

    #[test]
    fn test_trust_level_equality() {
        assert_eq!(TrustLevel::Root, TrustLevel::Root);
        assert_ne!(TrustLevel::Root, TrustLevel::Leaf);
        assert_ne!(TrustLevel::Intermediate, TrustLevel::Leaf);
    }

    // ── ChainLink ────────────────────────────────────────────────────────────

    #[test]
    fn test_chain_link_not_expired_when_no_expiry() {
        let link = make_root("did:ex:root");
        assert!(!link.is_expired(u64::MAX));
    }

    #[test]
    fn test_chain_link_not_expired_before_expiry() {
        let mut link = make_root("did:ex:root");
        link.expires_at = Some(1_000);
        assert!(!link.is_expired(999));
        assert!(!link.is_expired(1_000));
    }

    #[test]
    fn test_chain_link_expired_after_expiry() {
        let mut link = make_root("did:ex:root");
        link.expires_at = Some(1_000);
        assert!(link.is_expired(1_001));
    }

    #[test]
    fn test_chain_link_fields_accessible() {
        let link = ChainLink {
            did: "did:ex:test".to_string(),
            issuer_did: Some("did:ex:issuer".to_string()),
            trust_level: TrustLevel::Intermediate,
            is_revoked: true,
            issued_at: 42,
            expires_at: Some(100),
        };
        assert_eq!(link.did, "did:ex:test");
        assert_eq!(link.issuer_did.as_deref(), Some("did:ex:issuer"));
        assert_eq!(link.trust_level, TrustLevel::Intermediate);
        assert!(link.is_revoked);
        assert_eq!(link.issued_at, 42);
        assert_eq!(link.expires_at, Some(100));
    }

    // ── TrustChain construction ───────────────────────────────────────────────

    #[test]
    fn test_new_chain_is_empty() {
        let chain = TrustChain::new();
        assert_eq!(chain.chain_length(), 0);
        assert!(chain.root().is_none());
        assert!(chain.leaf().is_none());
    }

    #[test]
    fn test_add_link_single() {
        let mut chain = TrustChain::new();
        chain.add_link(make_root("did:ex:root")).expect("add root");
        assert_eq!(chain.chain_length(), 1);
    }

    #[test]
    fn test_add_link_duplicate_returns_error() {
        let mut chain = TrustChain::new();
        chain.add_link(make_root("did:ex:root")).expect("first add");
        let err = chain
            .add_link(make_root("did:ex:root"))
            .expect_err("duplicate DID");
        assert!(matches!(err, TrustChainError::CyclicChain(_)));
    }

    // ── TrustChain::root / leaf ───────────────────────────────────────────────

    #[test]
    fn test_root_returns_last_link() {
        let chain = simple_two_link_chain();
        assert_eq!(chain.root().map(|l| l.did.as_str()), Some("did:ex:root"));
    }

    #[test]
    fn test_leaf_returns_first_link() {
        let chain = simple_two_link_chain();
        assert_eq!(chain.leaf().map(|l| l.did.as_str()), Some("did:ex:leaf"));
    }

    #[test]
    fn test_root_and_leaf_same_for_single_link() {
        let mut chain = TrustChain::new();
        chain.add_link(make_root("did:ex:only")).expect("add");
        assert_eq!(
            chain.root().map(|l| l.did.as_str()),
            chain.leaf().map(|l| l.did.as_str())
        );
    }

    // ── TrustChain::contains ──────────────────────────────────────────────────

    #[test]
    fn test_contains_existing_did() {
        let chain = simple_two_link_chain();
        assert!(chain.contains("did:ex:leaf"));
        assert!(chain.contains("did:ex:root"));
    }

    #[test]
    fn test_contains_missing_did() {
        let chain = simple_two_link_chain();
        assert!(!chain.contains("did:ex:unknown"));
    }

    // ── TrustChain::validate — happy path ─────────────────────────────────────

    #[test]
    fn test_validate_single_root_link() {
        let mut chain = TrustChain::new();
        chain.add_link(make_root("did:ex:root")).expect("add");
        assert!(chain.validate(0).is_ok());
    }

    #[test]
    fn test_validate_two_link_chain() {
        let chain = simple_two_link_chain();
        assert!(chain.validate(0).is_ok());
    }

    #[test]
    fn test_validate_three_link_chain() {
        let chain = three_link_chain();
        assert!(chain.validate(0).is_ok());
    }

    #[test]
    fn test_validate_with_expiry_not_yet_expired() {
        let mut chain = TrustChain::new();
        let mut leaf = make_leaf("did:ex:leaf", "did:ex:root");
        leaf.expires_at = Some(2_000);
        chain.add_link(leaf).expect("add leaf");
        chain.add_link(make_root("did:ex:root")).expect("add root");
        assert!(chain.validate(1_000).is_ok());
    }

    // ── TrustChain::validate — empty chain ────────────────────────────────────

    #[test]
    fn test_validate_empty_chain_no_root() {
        let chain = TrustChain::new();
        let err = chain.validate(0).expect_err("empty chain should fail");
        assert_eq!(err, TrustChainError::NoRootFound);
    }

    // ── TrustChain::validate — revoked link ───────────────────────────────────

    #[test]
    fn test_validate_revoked_leaf() {
        let mut chain = TrustChain::new();
        let mut leaf = make_leaf("did:ex:leaf", "did:ex:root");
        leaf.is_revoked = true;
        chain.add_link(leaf).expect("add leaf");
        chain.add_link(make_root("did:ex:root")).expect("add root");
        let err = chain.validate(0).expect_err("revoked leaf should fail");
        assert!(matches!(err, TrustChainError::RevokedLink(ref d) if d == "did:ex:leaf"));
    }

    #[test]
    fn test_validate_revoked_root() {
        let mut chain = TrustChain::new();
        chain
            .add_link(make_leaf("did:ex:leaf", "did:ex:root"))
            .expect("add leaf");
        let mut root = make_root("did:ex:root");
        root.is_revoked = true;
        chain.add_link(root).expect("add root");
        let err = chain.validate(0).expect_err("revoked root should fail");
        assert!(matches!(err, TrustChainError::RevokedLink(ref d) if d == "did:ex:root"));
    }

    #[test]
    fn test_validate_revoked_intermediate() {
        let mut chain = TrustChain::new();
        chain
            .add_link(make_leaf("did:ex:leaf", "did:ex:mid"))
            .expect("add leaf");
        let mut mid = make_intermediate("did:ex:mid", "did:ex:root");
        mid.is_revoked = true;
        chain.add_link(mid).expect("add mid");
        chain.add_link(make_root("did:ex:root")).expect("add root");
        let err = chain
            .validate(0)
            .expect_err("revoked intermediate should fail");
        assert!(matches!(err, TrustChainError::RevokedLink(ref d) if d == "did:ex:mid"));
    }

    // ── TrustChain::validate — expired link ───────────────────────────────────

    #[test]
    fn test_validate_expired_leaf() {
        let mut chain = TrustChain::new();
        let mut leaf = make_leaf("did:ex:leaf", "did:ex:root");
        leaf.expires_at = Some(500);
        chain.add_link(leaf).expect("add leaf");
        chain.add_link(make_root("did:ex:root")).expect("add root");
        let err = chain.validate(1_000).expect_err("expired leaf should fail");
        assert!(matches!(err, TrustChainError::ExpiredLink(ref d) if d == "did:ex:leaf"));
    }

    #[test]
    fn test_validate_expired_root() {
        let mut chain = TrustChain::new();
        chain
            .add_link(make_leaf("did:ex:leaf", "did:ex:root"))
            .expect("add leaf");
        let mut root = make_root("did:ex:root");
        root.expires_at = Some(100);
        chain.add_link(root).expect("add root");
        let err = chain.validate(200).expect_err("expired root should fail");
        assert!(matches!(err, TrustChainError::ExpiredLink(ref d) if d == "did:ex:root"));
    }

    #[test]
    fn test_validate_expired_intermediate() {
        let mut chain = TrustChain::new();
        chain
            .add_link(make_leaf("did:ex:leaf", "did:ex:mid"))
            .expect("add leaf");
        let mut mid = make_intermediate("did:ex:mid", "did:ex:root");
        mid.expires_at = Some(50);
        chain.add_link(mid).expect("add mid");
        chain.add_link(make_root("did:ex:root")).expect("add root");
        let err = chain
            .validate(100)
            .expect_err("expired intermediate should fail");
        assert!(matches!(err, TrustChainError::ExpiredLink(ref d) if d == "did:ex:mid"));
    }

    // ── TrustChain::validate — broken chain ───────────────────────────────────

    #[test]
    fn test_validate_broken_chain_wrong_issuer() {
        let mut chain = TrustChain::new();
        // leaf claims issuer = "did:ex:wrong", but next link is "did:ex:root"
        let mut leaf = make_leaf("did:ex:leaf", "did:ex:wrong");
        leaf.issuer_did = Some("did:ex:wrong".to_string());
        chain.add_link(leaf).expect("add leaf");
        chain.add_link(make_root("did:ex:root")).expect("add root");
        let err = chain.validate(0).expect_err("broken chain should fail");
        assert!(matches!(err, TrustChainError::BrokenChain(_)));
    }

    #[test]
    fn test_validate_broken_chain_missing_intermediate() {
        // leaf → root, skipping required intermediate
        let mut chain = TrustChain::new();
        chain
            .add_link(make_leaf("did:ex:leaf", "did:ex:mid"))
            .expect("add leaf");
        // root is added but leaf points to "did:ex:mid", not "did:ex:root"
        chain.add_link(make_root("did:ex:root")).expect("add root");
        let err = chain.validate(0).expect_err("gap should fail");
        assert!(matches!(err, TrustChainError::BrokenChain(_)));
    }

    // ── TrustChain::validate — no root ────────────────────────────────────────

    #[test]
    fn test_validate_no_root_only_intermediate() {
        let mut chain = TrustChain::new();
        chain
            .add_link(make_leaf("did:ex:leaf", "did:ex:mid"))
            .expect("add leaf");
        chain
            .add_link(make_intermediate("did:ex:mid", "did:ex:root"))
            .expect("add mid");
        // intentionally do not add root
        // last link is Intermediate, not Root → NoRootFound
        let err = chain.validate(0).expect_err("no root should fail");
        // Could be BrokenChain (mid has issuer but is last) or NoRootFound
        assert!(
            matches!(err, TrustChainError::NoRootFound)
                || matches!(err, TrustChainError::BrokenChain(_))
        );
    }

    #[test]
    fn test_validate_last_link_not_root_trust_level() {
        let mut chain = TrustChain::new();
        // Add a leaf as the only link — trust_level is Leaf, not Root
        chain
            .add_link(make_leaf("did:ex:leaf", "did:ex:root"))
            .expect("add leaf");
        let err = chain.validate(0).expect_err("leaf-only should fail");
        // last link is not Root trust level → NoRootFound
        assert!(matches!(err, TrustChainError::NoRootFound));
    }

    // ── TrustChain::validate — non-root link has no issuer (not last) ─────────

    #[test]
    fn test_validate_root_like_link_in_middle() {
        let mut chain = TrustChain::new();
        // put a root-style link (no issuer) in the middle
        let orphan = root_link("did:ex:orphan", 0);
        chain.add_link(orphan).expect("add orphan");
        chain.add_link(make_root("did:ex:root")).expect("add root");
        // orphan has no issuer but is not the last link → BrokenChain
        let err = chain.validate(0).expect_err("orphan in middle should fail");
        assert!(matches!(err, TrustChainError::BrokenChain(_)));
    }

    // ── TrustChain::validate — root with issuer ───────────────────────────────

    #[test]
    fn test_validate_root_with_issuer_fails() {
        let mut chain = TrustChain::new();
        chain
            .add_link(make_leaf("did:ex:leaf", "did:ex:root"))
            .expect("add leaf");
        // Root that accidentally has an issuer
        let mut root = make_root("did:ex:root");
        root.issuer_did = Some("did:ex:extra".to_string());
        chain.add_link(root).expect("add root");
        let err = chain.validate(0).expect_err("root with issuer should fail");
        assert!(matches!(err, TrustChainError::BrokenChain(_)));
    }

    // ── Cycle detection ───────────────────────────────────────────────────────

    #[test]
    fn test_add_link_detects_duplicate_did() {
        let mut chain = TrustChain::new();
        chain
            .add_link(make_leaf("did:ex:a", "did:ex:b"))
            .expect("first");
        let err = chain
            .add_link(make_leaf("did:ex:a", "did:ex:c"))
            .expect_err("cycle");
        assert!(matches!(err, TrustChainError::CyclicChain(_)));
    }

    #[test]
    fn test_add_link_three_unique_dids_no_error() {
        let mut chain = TrustChain::new();
        chain
            .add_link(make_leaf("did:ex:leaf", "did:ex:mid"))
            .expect("leaf");
        chain
            .add_link(make_intermediate("did:ex:mid", "did:ex:root"))
            .expect("mid");
        chain.add_link(make_root("did:ex:root")).expect("root");
        assert_eq!(chain.chain_length(), 3);
    }

    // ── TrustChainBuilder ─────────────────────────────────────────────────────

    #[test]
    fn test_builder_empty_produces_empty_chain() {
        let chain = TrustChainBuilder::new().build();
        assert_eq!(chain.chain_length(), 0);
    }

    #[test]
    fn test_builder_single_root() {
        let mut builder = TrustChainBuilder::new();
        builder.add(make_root("did:ex:root"));
        let chain = builder.build();
        assert_eq!(chain.chain_length(), 1);
        assert!(chain.validate(0).is_ok());
    }

    #[test]
    fn test_builder_two_link_chain() {
        let mut builder = TrustChainBuilder::new();
        builder
            .add(make_leaf("did:ex:leaf", "did:ex:root"))
            .add(make_root("did:ex:root"));
        let chain = builder.build();
        assert!(chain.validate(0).is_ok());
    }

    #[test]
    fn test_builder_three_link_chain() {
        let mut builder = TrustChainBuilder::new();
        builder
            .add(make_leaf("did:ex:leaf", "did:ex:mid"))
            .add(make_intermediate("did:ex:mid", "did:ex:root"))
            .add(make_root("did:ex:root"));
        let chain = builder.build();
        assert!(chain.validate(0).is_ok());
    }

    #[test]
    fn test_builder_returns_mutable_ref_for_chaining() {
        let mut b = TrustChainBuilder::new();
        // add() returns &mut Self so it can be chained; we then call build() on b.
        b.add(make_root("did:ex:root"));
        let chain = b.build();
        assert_eq!(chain.chain_length(), 1);
    }

    // ── Convenience constructors ──────────────────────────────────────────────

    #[test]
    fn test_root_link_helper() {
        let link = root_link("did:ex:root", 12345);
        assert_eq!(link.did, "did:ex:root");
        assert!(link.issuer_did.is_none());
        assert_eq!(link.trust_level, TrustLevel::Root);
        assert!(!link.is_revoked);
        assert_eq!(link.issued_at, 12345);
        assert!(link.expires_at.is_none());
    }

    #[test]
    fn test_intermediate_link_helper() {
        let link = intermediate_link("did:ex:mid", "did:ex:root", 0);
        assert_eq!(link.trust_level, TrustLevel::Intermediate);
        assert_eq!(link.issuer_did.as_deref(), Some("did:ex:root"));
    }

    #[test]
    fn test_leaf_link_helper() {
        let link = leaf_link("did:ex:leaf", "did:ex:mid", 0);
        assert_eq!(link.trust_level, TrustLevel::Leaf);
        assert_eq!(link.issuer_did.as_deref(), Some("did:ex:mid"));
    }

    // ── iter ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_iter_order_leaf_to_root() {
        let chain = three_link_chain();
        let dids: Vec<&str> = chain.iter().map(|l| l.did.as_str()).collect();
        assert_eq!(dids, ["did:ex:leaf", "did:ex:mid", "did:ex:root"]);
    }

    // ── Error display ─────────────────────────────────────────────────────────

    #[test]
    fn test_error_display_broken_chain() {
        let e = TrustChainError::BrokenChain("test".to_string());
        assert!(e.to_string().contains("Broken chain"));
    }

    #[test]
    fn test_error_display_revoked_link() {
        let e = TrustChainError::RevokedLink("did:ex:x".to_string());
        assert!(e.to_string().contains("Revoked link"));
    }

    #[test]
    fn test_error_display_expired_link() {
        let e = TrustChainError::ExpiredLink("did:ex:x".to_string());
        assert!(e.to_string().contains("Expired link"));
    }

    #[test]
    fn test_error_display_no_root() {
        let e = TrustChainError::NoRootFound;
        assert!(e.to_string().contains("No root"));
    }

    #[test]
    fn test_error_display_cyclic_chain() {
        let e = TrustChainError::CyclicChain(vec!["did:ex:a".to_string()]);
        assert!(e.to_string().contains("Cyclic chain"));
    }

    #[test]
    fn test_error_display_unknown_did() {
        let e = TrustChainError::UnknownDid("did:ex:ghost".to_string());
        assert!(e.to_string().contains("Unknown DID"));
    }

    // ── Long chain (4 links) ──────────────────────────────────────────────────

    #[test]
    fn test_validate_four_link_chain() {
        let mut chain = TrustChain::new();
        chain
            .add_link(make_leaf("did:ex:leaf", "did:ex:mid1"))
            .expect("leaf");
        chain
            .add_link(make_intermediate("did:ex:mid1", "did:ex:mid2"))
            .expect("mid1");
        chain
            .add_link(make_intermediate("did:ex:mid2", "did:ex:root"))
            .expect("mid2");
        chain.add_link(make_root("did:ex:root")).expect("root");
        assert_eq!(chain.chain_length(), 4);
        assert!(chain.validate(0).is_ok());
    }

    #[test]
    fn test_validate_four_link_chain_one_expired() {
        let mut chain = TrustChain::new();
        chain
            .add_link(make_leaf("did:ex:leaf", "did:ex:mid1"))
            .expect("leaf");
        let mut mid1 = make_intermediate("did:ex:mid1", "did:ex:mid2");
        mid1.expires_at = Some(10);
        chain.add_link(mid1).expect("mid1");
        chain
            .add_link(make_intermediate("did:ex:mid2", "did:ex:root"))
            .expect("mid2");
        chain.add_link(make_root("did:ex:root")).expect("root");
        let err = chain.validate(100).expect_err("mid1 expired");
        assert!(matches!(err, TrustChainError::ExpiredLink(ref d) if d == "did:ex:mid1"));
    }

    // ── Chain length ──────────────────────────────────────────────────────────

    #[test]
    fn test_chain_length_zero() {
        assert_eq!(TrustChain::new().chain_length(), 0);
    }

    #[test]
    fn test_chain_length_two() {
        assert_eq!(simple_two_link_chain().chain_length(), 2);
    }

    #[test]
    fn test_chain_length_three() {
        assert_eq!(three_link_chain().chain_length(), 3);
    }

    // ── Clone ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_chain_link_clone() {
        let link = make_root("did:ex:r");
        let cloned = link.clone();
        assert_eq!(link.did, cloned.did);
        assert_eq!(link.trust_level, cloned.trust_level);
    }

    #[test]
    fn test_trust_chain_clone() {
        let chain = simple_two_link_chain();
        let cloned = chain.clone();
        assert_eq!(chain.chain_length(), cloned.chain_length());
    }
}
