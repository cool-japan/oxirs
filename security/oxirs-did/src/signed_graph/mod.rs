//! Signed RDF Graphs module

pub mod canonicalization;
pub mod signature;

use crate::{Did, DidResult, Proof, ProofPurpose};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A signed RDF graph
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SignedGraph {
    /// Graph URI (named graph identifier)
    pub graph_uri: String,

    /// RDF triples in the graph
    pub triples: Vec<RdfTriple>,

    /// Issuer DID
    pub issuer: Did,

    /// Issuance timestamp
    pub issued_at: DateTime<Utc>,

    /// Graph signature
    pub proof: Proof,

    /// Optional expiration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<DateTime<Utc>>,
}

/// RDF Triple representation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RdfTriple {
    pub subject: RdfTerm,
    pub predicate: String,
    pub object: RdfTerm,
}

/// RDF Term (subject or object)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum RdfTerm {
    /// IRI reference
    Iri(String),
    /// Blank node
    BlankNode(String),
    /// Literal value
    Literal {
        value: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        datatype: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        language: Option<String>,
    },
}

impl RdfTriple {
    pub fn new(subject: RdfTerm, predicate: &str, object: RdfTerm) -> Self {
        Self {
            subject,
            predicate: predicate.to_string(),
            object,
        }
    }

    /// Create triple with IRI subject and object
    pub fn iri(subject: &str, predicate: &str, object: &str) -> Self {
        Self {
            subject: RdfTerm::Iri(subject.to_string()),
            predicate: predicate.to_string(),
            object: RdfTerm::Iri(object.to_string()),
        }
    }

    /// Create triple with IRI subject and literal object
    pub fn literal(subject: &str, predicate: &str, value: &str, datatype: Option<&str>) -> Self {
        Self {
            subject: RdfTerm::Iri(subject.to_string()),
            predicate: predicate.to_string(),
            object: RdfTerm::Literal {
                value: value.to_string(),
                datatype: datatype.map(String::from),
                language: None,
            },
        }
    }
}

impl SignedGraph {
    /// Create a new signed graph (without signing yet)
    pub fn new(graph_uri: &str, triples: Vec<RdfTriple>, issuer: Did) -> Self {
        Self {
            graph_uri: graph_uri.to_string(),
            triples,
            issuer: issuer.clone(),
            issued_at: Utc::now(),
            proof: Proof::ed25519(
                &issuer.key_id("key-1"),
                ProofPurpose::AssertionMethod,
                &[], // Empty signature - to be filled
            ),
            expires_at: None,
        }
    }

    /// Sign the graph with Ed25519
    pub fn sign(mut self, signer: &crate::proof::ed25519::Ed25519Signer) -> DidResult<Self> {
        // Canonicalize the graph
        let canonical = self.canonicalize()?;

        // Hash the canonical form
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(canonical.as_bytes());

        // Sign the hash
        let signature = signer.sign(&hash);

        // Update proof
        self.proof = Proof::ed25519(
            &self.issuer.key_id("key-1"),
            ProofPurpose::AssertionMethod,
            &signature,
        );

        Ok(self)
    }

    /// Verify the graph signature
    pub async fn verify(
        &self,
        resolver: &crate::DidResolver,
    ) -> DidResult<crate::VerificationResult> {
        // Resolve issuer DID
        let did_doc = resolver.resolve(&self.issuer).await?;

        // Get verification key
        let vm = did_doc.get_assertion_method().ok_or_else(|| {
            crate::DidError::VerificationFailed("No assertion method in DID Document".to_string())
        })?;

        let public_key = vm.get_public_key_bytes()?;

        // Canonicalize
        let canonical = self.canonicalize()?;

        // Hash
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(canonical.as_bytes());

        // Verify signature
        let signature = self.proof.get_signature_bytes()?;
        let verifier = crate::proof::ed25519::Ed25519Verifier::from_bytes(&public_key)?;
        let valid = verifier.verify(&hash, &signature)?;

        if valid {
            Ok(crate::VerificationResult::success(self.issuer.as_str())
                .with_check("signature", true, None)
                .with_check("expiration", !self.is_expired(), None))
        } else {
            Ok(crate::VerificationResult::failure("Invalid signature"))
        }
    }

    /// Canonicalize the graph using RDFC-1.0 algorithm
    fn canonicalize(&self) -> DidResult<String> {
        canonicalization::canonicalize_graph(&self.triples)
    }

    /// Check if the graph signature is expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires) = self.expires_at {
            Utc::now() > expires
        } else {
            false
        }
    }

    /// Set expiration
    pub fn with_expiration(mut self, expires: DateTime<Utc>) -> Self {
        self.expires_at = Some(expires);
        self
    }

    /// Get the graph hash
    pub fn hash(&self) -> DidResult<String> {
        let canonical = self.canonicalize()?;
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(canonical.as_bytes());
        Ok(hex::encode(hash))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::ed25519::Ed25519Signer;

    #[test]
    fn test_rdf_triple() {
        let triple = RdfTriple::iri(
            "http://example.org/subject",
            "http://example.org/predicate",
            "http://example.org/object",
        );

        assert!(matches!(triple.subject, RdfTerm::Iri(_)));
        assert!(matches!(triple.object, RdfTerm::Iri(_)));
    }

    #[test]
    fn test_signed_graph() {
        let signer = Ed25519Signer::generate();
        let public_key = signer.public_key_bytes();
        let issuer = Did::new_key_ed25519(&public_key).unwrap();

        let triples = vec![RdfTriple::iri(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        )];

        let graph = SignedGraph::new("http://example.org/graph", triples, issuer);
        let signed = graph.sign(&signer).unwrap();

        assert!(signed.proof.proof_value.is_some());
    }
}
