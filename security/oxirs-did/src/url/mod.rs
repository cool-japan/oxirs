//! DID URL Dereferencing — W3C DID Core §7.2
//!
//! A DID URL extends a DID with optional path, query, and fragment components
//! following the same syntax rules as generic URIs.
//!
//! Format: `did:method:id[/path][?query][#fragment]`
//!
//! Dereferencing maps a DID URL to a resource within (or referenced by)
//! the DID Document identified by the base DID.

use crate::{DidDocument, DidError, DidResult};
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// DidUrl — parsed representation
// ─────────────────────────────────────────────────────────────────────────────

/// A parsed DID URL with optional path, query, and fragment components.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DidUrl {
    /// The base DID (e.g. `did:example:123`)
    pub did: String,
    /// Path component after the DID, without leading `/`
    pub path: Option<String>,
    /// Raw query string (without leading `?`)
    pub query: Option<String>,
    /// Fragment identifier (without leading `#`)
    pub fragment: Option<String>,
}

impl DidUrl {
    /// Parse a DID URL string.
    ///
    /// Handles all of:
    /// - `did:example:123`
    /// - `did:example:123#key-1`
    /// - `did:example:123/path`
    /// - `did:example:123/path?query=foo`
    /// - `did:example:123/path?query=foo#fragment`
    pub fn parse(url: &str) -> DidResult<Self> {
        if !url.starts_with("did:") {
            return Err(DidError::InvalidFormat(
                "DID URL must start with 'did:'".to_string(),
            ));
        }

        // Work left to right splitting off fragment, then query, then path.
        // DID method-specific IDs may contain colons but NOT `?`, `#`, or `/`
        // (those are reserved for URL structure).

        // 1. Split fragment
        let (before_fragment, fragment) = if let Some(pos) = url.find('#') {
            let frag = url[pos + 1..].to_string();
            if frag.is_empty() {
                return Err(DidError::InvalidFormat(
                    "Fragment identifier cannot be empty".to_string(),
                ));
            }
            (&url[..pos], Some(frag))
        } else {
            (url, None)
        };

        // 2. Split query
        let (before_query, query) = if let Some(pos) = before_fragment.find('?') {
            let q = before_fragment[pos + 1..].to_string();
            if q.is_empty() {
                return Err(DidError::InvalidFormat(
                    "Query string cannot be empty".to_string(),
                ));
            }
            (&before_fragment[..pos], Some(q))
        } else {
            (before_fragment, None)
        };

        // 3. Split path — everything after the third `:` that is followed by `/`
        //    We need to find where the method-specific-id ends and the path begins.
        //    The DID portion is `did:<method>:<method-specific-id>`, where
        //    method-specific-id continues until the first unencoded `/`.
        //
        //    Strategy: scan for the first '/' that appears after at least two ':'
        let (did_part, path) = extract_did_and_path(before_query)?;

        // Basic DID validation: must be did:<method>:<id>
        validate_did_structure(did_part)?;

        Ok(Self {
            did: did_part.to_string(),
            path,
            query,
            fragment,
        })
    }

    /// Reconstruct the full DID URL string.
    pub fn as_url_string(&self) -> String {
        let mut out = self.did.clone();
        if let Some(ref p) = self.path {
            out.push('/');
            out.push_str(p);
        }
        if let Some(ref q) = self.query {
            out.push('?');
            out.push_str(q);
        }
        if let Some(ref f) = self.fragment {
            out.push('#');
            out.push_str(f);
        }
        out
    }

    /// Return the base DID (without path/query/fragment).
    pub fn base_did(&self) -> &str {
        &self.did
    }

    /// Return the fragment component, if any.
    pub fn fragment_id(&self) -> Option<&str> {
        self.fragment.as_deref()
    }

    /// Return the query component, if any.
    pub fn query_string(&self) -> Option<&str> {
        self.query.as_deref()
    }

    /// Return the path component, if any.
    pub fn path_segment(&self) -> Option<&str> {
        self.path.as_deref()
    }

    /// Parse query into key=value pairs.
    pub fn query_params(&self) -> Vec<(String, String)> {
        match &self.query {
            None => vec![],
            Some(q) => q
                .split('&')
                .filter_map(|pair| {
                    let mut it = pair.splitn(2, '=');
                    let key = it.next()?.to_string();
                    let val = it.next().unwrap_or("").to_string();
                    Some((key, val))
                })
                .collect(),
        }
    }

    /// True when the URL has no path, query, or fragment (bare DID URL).
    pub fn is_bare_did(&self) -> bool {
        self.path.is_none() && self.query.is_none() && self.fragment.is_none()
    }
}

impl fmt::Display for DidUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_url_string())
    }
}

/// Extract the DID portion and optional path from a string that has already
/// had fragment and query stripped.
fn extract_did_and_path(s: &str) -> DidResult<(&str, Option<String>)> {
    // Count colons to find where the method-specific-id starts.
    // A minimal DID is `did:method:id` — colons 0 and 1 delimit "did" and method.
    // The method-specific-id runs until the first unencoded '/'.
    let colon_count = s.chars().filter(|&c| c == ':').count();
    if colon_count < 2 {
        return Err(DidError::InvalidFormat(
            "DID URL must contain at least two colons (did:method:id)".to_string(),
        ));
    }

    if let Some(slash_pos) = s.find('/') {
        let did = &s[..slash_pos];
        let path = s[slash_pos + 1..].to_string();
        if path.is_empty() {
            return Err(DidError::InvalidFormat(
                "Path segment cannot be empty after '/'".to_string(),
            ));
        }
        Ok((did, Some(path)))
    } else {
        Ok((s, None))
    }
}

fn validate_did_structure(did: &str) -> DidResult<()> {
    let parts: Vec<&str> = did.splitn(3, ':').collect();
    if parts.len() < 3 || parts[0] != "did" || parts[1].is_empty() || parts[2].is_empty() {
        return Err(DidError::InvalidFormat(format!(
            "Invalid DID structure: '{}'",
            did
        )));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Dereferenced resource types
// ─────────────────────────────────────────────────────────────────────────────

/// The result of dereferencing a DID URL
#[derive(Debug, Clone)]
pub enum DereferencedResource {
    /// The full DID Document (bare DID URL with no path/fragment)
    Document(Box<DidDocument>),
    /// A specific verification method (addressed by fragment matching vm id)
    VerificationMethod(serde_json::Value),
    /// A service endpoint entry (addressed by fragment matching service id)
    Service(serde_json::Value),
    /// Arbitrary content bytes (addressed by path)
    ContentStream(Vec<u8>),
}

// ─────────────────────────────────────────────────────────────────────────────
// DidDereferencer
// ─────────────────────────────────────────────────────────────────────────────

/// Resolves DID URLs against a DID Document, returning the addressed resource.
pub struct DidDereferencer;

impl DidDereferencer {
    /// Dereference `url` against `doc`.
    ///
    /// Resolution priority:
    /// 1. If the URL has a **fragment**: match against verification method ids,
    ///    service ids, or return the raw fragment as a content stream.
    /// 2. If the URL has a **path**: return the path bytes as a ContentStream
    ///    (a real implementation would fetch external resources).
    /// 3. Otherwise: return the full DID Document.
    pub fn dereference(doc: &DidDocument, url: &DidUrl) -> DidResult<DereferencedResource> {
        // Ensure the URL's base DID matches the document
        if url.did != doc.id.as_str() {
            return Err(DidError::ResolutionFailed(format!(
                "DID URL base '{}' does not match document id '{}'",
                url.did,
                doc.id.as_str()
            )));
        }

        match (&url.fragment, &url.path) {
            // Fragment takes priority
            (Some(fragment), _) => Self::dereference_fragment(doc, fragment),
            // Path comes next
            (None, Some(path)) => Ok(DereferencedResource::ContentStream(
                path.as_bytes().to_vec(),
            )),
            // Bare DID URL → return the whole document
            (None, None) => Ok(DereferencedResource::Document(Box::new(doc.clone()))),
        }
    }

    fn dereference_fragment(doc: &DidDocument, fragment: &str) -> DidResult<DereferencedResource> {
        let full_id = format!("{}#{}", doc.id.as_str(), fragment);

        // 1. Search verification methods
        if let Some(vm) = doc
            .verification_method
            .iter()
            .find(|vm| vm.id == full_id || vm.id.ends_with(&format!("#{}", fragment)))
        {
            let value = serde_json::to_value(vm)
                .map_err(|e| DidError::SerializationError(e.to_string()))?;
            return Ok(DereferencedResource::VerificationMethod(value));
        }

        // 2. Search service endpoints
        if let Some(svc) = doc
            .service
            .iter()
            .find(|s| s.id == full_id || s.id.ends_with(&format!("#{}", fragment)))
        {
            let value = serde_json::to_value(svc)
                .map_err(|e| DidError::SerializationError(e.to_string()))?;
            return Ok(DereferencedResource::Service(value));
        }

        // 3. Not found
        Err(DidError::ResolutionFailed(format!(
            "Fragment '{}' not found in DID Document '{}'",
            fragment,
            doc.id.as_str()
        )))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::did::document::DidDocument;
    use crate::{Did, Service, VerificationMethod};

    fn sample_doc() -> DidDocument {
        let did = Did::new("did:example:123").unwrap();
        let mut doc = DidDocument::new(did);

        let vm =
            VerificationMethod::ed25519("did:example:123#key-1", "did:example:123", &[1u8; 32]);
        doc.verification_method.push(vm);

        doc.service.push(Service {
            id: "did:example:123#linked-domain".to_string(),
            service_type: "LinkedDomains".to_string(),
            service_endpoint: "https://example.com".to_string(),
        });

        doc
    }

    // ── DidUrl::parse ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_bare_did() {
        let url = DidUrl::parse("did:example:123").unwrap();
        assert_eq!(url.did, "did:example:123");
        assert!(url.path.is_none());
        assert!(url.query.is_none());
        assert!(url.fragment.is_none());
        assert!(url.is_bare_did());
    }

    #[test]
    fn test_parse_with_fragment() {
        let url = DidUrl::parse("did:example:123#key-1").unwrap();
        assert_eq!(url.did, "did:example:123");
        assert_eq!(url.fragment_id(), Some("key-1"));
        assert!(url.path.is_none());
    }

    #[test]
    fn test_parse_with_path() {
        let url = DidUrl::parse("did:example:123/path/to/resource").unwrap();
        assert_eq!(url.did, "did:example:123");
        assert_eq!(url.path_segment(), Some("path/to/resource"));
    }

    #[test]
    fn test_parse_with_query() {
        let url = DidUrl::parse("did:example:123?versionId=1").unwrap();
        assert_eq!(url.did, "did:example:123");
        assert_eq!(url.query_string(), Some("versionId=1"));
    }

    #[test]
    fn test_parse_full_url() {
        let url = DidUrl::parse("did:example:123/path?query=foo#frag").unwrap();
        assert_eq!(url.did, "did:example:123");
        assert_eq!(url.path_segment(), Some("path"));
        assert_eq!(url.query_string(), Some("query=foo"));
        assert_eq!(url.fragment_id(), Some("frag"));
    }

    #[test]
    fn test_parse_fragment_only() {
        let url = DidUrl::parse("did:example:123#linked-domain").unwrap();
        assert_eq!(url.fragment_id(), Some("linked-domain"));
        assert!(url.query.is_none());
    }

    #[test]
    fn test_parse_invalid_no_method() {
        assert!(DidUrl::parse("not-a-did").is_err());
    }

    #[test]
    fn test_parse_invalid_missing_id() {
        assert!(DidUrl::parse("did:example").is_err());
    }

    #[test]
    fn test_parse_did_key_url() {
        let did_key = "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK";
        let url = DidUrl::parse(&format!("{}#key-0", did_key)).unwrap();
        assert_eq!(url.did, did_key);
        assert_eq!(url.fragment_id(), Some("key-0"));
    }

    #[test]
    fn test_parse_ethr_with_fragment() {
        let url = DidUrl::parse("did:ethr:0xf3beac30c498d9e26865f34fcaa57dbb935b0d74#controller")
            .unwrap();
        assert_eq!(
            url.did,
            "did:ethr:0xf3beac30c498d9e26865f34fcaa57dbb935b0d74"
        );
        assert_eq!(url.fragment_id(), Some("controller"));
    }

    // ── DidUrl::to_string ─────────────────────────────────────────────────

    #[test]
    fn test_to_string_bare() {
        let url = DidUrl::parse("did:example:456").unwrap();
        assert_eq!(url.to_string(), "did:example:456");
    }

    #[test]
    fn test_to_string_with_all_components() {
        let url = DidUrl::parse("did:example:456/p?q=1#frag").unwrap();
        assert_eq!(url.to_string(), "did:example:456/p?q=1#frag");
    }

    #[test]
    fn test_to_string_roundtrip() {
        let original = "did:example:789#authentication";
        let url = DidUrl::parse(original).unwrap();
        assert_eq!(url.to_string(), original);
    }

    // ── query_params ──────────────────────────────────────────────────────

    #[test]
    fn test_query_params_single() {
        let url = DidUrl::parse("did:example:123?versionId=42").unwrap();
        let params = url.query_params();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0], ("versionId".to_string(), "42".to_string()));
    }

    #[test]
    fn test_query_params_multiple() {
        let url = DidUrl::parse("did:example:123?a=1&b=2&c=3").unwrap();
        let params = url.query_params();
        assert_eq!(params.len(), 3);
    }

    #[test]
    fn test_query_params_empty_when_no_query() {
        let url = DidUrl::parse("did:example:123").unwrap();
        assert!(url.query_params().is_empty());
    }

    // ── DidDereferencer ───────────────────────────────────────────────────

    #[test]
    fn test_dereference_bare_did_returns_document() {
        let doc = sample_doc();
        let url = DidUrl::parse("did:example:123").unwrap();
        let resource = DidDereferencer::dereference(&doc, &url).unwrap();

        assert!(matches!(resource, DereferencedResource::Document(ref _d)));
    }

    #[test]
    fn test_dereference_verification_method_by_fragment() {
        let doc = sample_doc();
        let url = DidUrl::parse("did:example:123#key-1").unwrap();
        let resource = DidDereferencer::dereference(&doc, &url).unwrap();

        match resource {
            DereferencedResource::VerificationMethod(vm) => {
                assert_eq!(vm["id"].as_str().unwrap(), "did:example:123#key-1");
                assert_eq!(vm["type"].as_str().unwrap(), "Ed25519VerificationKey2020");
            }
            other => panic!("Expected VerificationMethod, got {:?}", other),
        }
    }

    #[test]
    fn test_dereference_service_by_fragment() {
        let doc = sample_doc();
        let url = DidUrl::parse("did:example:123#linked-domain").unwrap();
        let resource = DidDereferencer::dereference(&doc, &url).unwrap();

        match resource {
            DereferencedResource::Service(svc) => {
                assert_eq!(svc["id"].as_str().unwrap(), "did:example:123#linked-domain");
                assert_eq!(svc["type"].as_str().unwrap(), "LinkedDomains");
            }
            other => panic!("Expected Service, got {:?}", other),
        }
    }

    #[test]
    fn test_dereference_path_returns_content_stream() {
        let doc = sample_doc();
        let url = DidUrl::parse("did:example:123/credentials/1").unwrap();
        let resource = DidDereferencer::dereference(&doc, &url).unwrap();

        match resource {
            DereferencedResource::ContentStream(bytes) => {
                assert_eq!(bytes, b"credentials/1");
            }
            other => panic!("Expected ContentStream, got {:?}", other),
        }
    }

    #[test]
    fn test_dereference_unknown_fragment_error() {
        let doc = sample_doc();
        let url = DidUrl::parse("did:example:123#nonexistent").unwrap();
        assert!(DidDereferencer::dereference(&doc, &url).is_err());
    }

    #[test]
    fn test_dereference_wrong_did_error() {
        let doc = sample_doc();
        let url = DidUrl::parse("did:example:OTHER#key-1").unwrap();
        assert!(DidDereferencer::dereference(&doc, &url).is_err());
    }

    #[test]
    fn test_dereference_document_has_correct_id() {
        let doc = sample_doc();
        let url = DidUrl::parse("did:example:123").unwrap();

        match DidDereferencer::dereference(&doc, &url).unwrap() {
            DereferencedResource::Document(boxed_doc) => {
                assert_eq!(boxed_doc.id.as_str(), "did:example:123");
            }
            _ => panic!("Expected Document"),
        }
    }
}
