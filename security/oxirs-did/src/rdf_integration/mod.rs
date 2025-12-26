//! RDF integration for DID/VC

use crate::signed_graph::{RdfTerm, RdfTriple};
use crate::{Did, DidResult, VerifiableCredential};

/// Convert a Verifiable Credential to RDF triples
pub fn vc_to_rdf(vc: &VerifiableCredential) -> DidResult<Vec<RdfTriple>> {
    let mut triples = Vec::new();

    // Credential URI
    let vc_uri = vc
        .id
        .clone()
        .unwrap_or_else(|| format!("urn:uuid:{}", uuid::Uuid::new_v4()));

    // Type triples
    for vc_type in &vc.credential_type {
        triples.push(RdfTriple::new(
            RdfTerm::Iri(vc_uri.clone()),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            RdfTerm::Iri(format!("https://www.w3.org/2018/credentials#{}", vc_type)),
        ));
    }

    // Issuer
    triples.push(RdfTriple::new(
        RdfTerm::Iri(vc_uri.clone()),
        "https://www.w3.org/2018/credentials#issuer",
        RdfTerm::Iri(vc.issuer.did().as_str().to_string()),
    ));

    // Issuance date
    if let Some(issued) = vc.issuance_date {
        triples.push(RdfTriple::new(
            RdfTerm::Iri(vc_uri.clone()),
            "https://www.w3.org/2018/credentials#issuanceDate",
            RdfTerm::Literal {
                value: issued.to_rfc3339(),
                datatype: Some("http://www.w3.org/2001/XMLSchema#dateTime".to_string()),
                language: None,
            },
        ));
    }

    // Expiration date
    if let Some(expires) = vc.expiration_date {
        triples.push(RdfTriple::new(
            RdfTerm::Iri(vc_uri.clone()),
            "https://www.w3.org/2018/credentials#expirationDate",
            RdfTerm::Literal {
                value: expires.to_rfc3339(),
                datatype: Some("http://www.w3.org/2001/XMLSchema#dateTime".to_string()),
                language: None,
            },
        ));
    }

    // Credential subject
    for subject in vc.credential_subject.subjects() {
        let subject_uri = subject
            .id
            .clone()
            .unwrap_or_else(|| format!("_:subject_{}", uuid::Uuid::new_v4()));

        triples.push(RdfTriple::new(
            RdfTerm::Iri(vc_uri.clone()),
            "https://www.w3.org/2018/credentials#credentialSubject",
            RdfTerm::Iri(subject_uri.clone()),
        ));

        // Claims
        for (key, value) in &subject.claims {
            let predicate = format!("https://schema.org/{}", key);
            let object = json_value_to_rdf_term(value);

            triples.push(RdfTriple::new(
                RdfTerm::Iri(subject_uri.clone()),
                &predicate,
                object,
            ));
        }
    }

    Ok(triples)
}

/// Convert JSON value to RDF term
fn json_value_to_rdf_term(value: &serde_json::Value) -> RdfTerm {
    match value {
        serde_json::Value::String(s) => RdfTerm::Literal {
            value: s.clone(),
            datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
            language: None,
        },
        serde_json::Value::Number(n) => {
            if n.is_i64() {
                RdfTerm::Literal {
                    value: n.to_string(),
                    datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
                    language: None,
                }
            } else {
                RdfTerm::Literal {
                    value: n.to_string(),
                    datatype: Some("http://www.w3.org/2001/XMLSchema#double".to_string()),
                    language: None,
                }
            }
        }
        serde_json::Value::Bool(b) => RdfTerm::Literal {
            value: b.to_string(),
            datatype: Some("http://www.w3.org/2001/XMLSchema#boolean".to_string()),
            language: None,
        },
        _ => RdfTerm::Literal {
            value: value.to_string(),
            datatype: None,
            language: None,
        },
    }
}

/// SPARQL extension functions for DID/VC
pub mod sparql_functions {
    /// Generate SPARQL query to find credentials by issuer
    pub fn credentials_by_issuer(issuer_did: &str) -> String {
        format!(
            r#"
PREFIX cred: <https://www.w3.org/2018/credentials#>

SELECT ?credential ?subject ?issuanceDate WHERE {{
    ?credential cred:issuer <{}> ;
                cred:credentialSubject ?subject ;
                cred:issuanceDate ?issuanceDate .
}}
ORDER BY DESC(?issuanceDate)
"#,
            issuer_did
        )
    }

    /// Generate SPARQL query to find credentials by subject
    pub fn credentials_by_subject(subject_did: &str) -> String {
        format!(
            r#"
PREFIX cred: <https://www.w3.org/2018/credentials#>

SELECT ?credential ?issuer ?type WHERE {{
    ?credential cred:credentialSubject <{}> ;
                cred:issuer ?issuer ;
                a ?type .
}}
"#,
            subject_did
        )
    }

    /// Generate SPARQL query to verify credential is not expired
    pub fn valid_credentials() -> String {
        r#"
PREFIX cred: <https://www.w3.org/2018/credentials#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?credential ?issuer ?subject WHERE {
    ?credential cred:issuer ?issuer ;
                cred:credentialSubject ?subject .
    OPTIONAL { ?credential cred:expirationDate ?expires }
    FILTER(!BOUND(?expires) || ?expires > NOW())
}
"#
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CredentialSubject;

    #[test]
    fn test_vc_to_rdf() {
        let issuer = Did::new_key_ed25519(&[0u8; 32]).unwrap();
        let subject = CredentialSubject::new(Some("did:example:holder"))
            .with_claim("name", "Alice")
            .with_claim("age", 30);

        let vc = VerifiableCredential::new(issuer, subject, vec!["PersonCredential".to_string()]);

        let triples = vc_to_rdf(&vc).unwrap();

        assert!(!triples.is_empty());
        assert!(triples.iter().any(|t| t.predicate.contains("issuer")));
        assert!(triples
            .iter()
            .any(|t| t.predicate.contains("credentialSubject")));
    }

    #[test]
    fn test_sparql_queries() {
        let query = sparql_functions::credentials_by_issuer("did:key:z123");
        assert!(query.contains("did:key:z123"));
        assert!(query.contains("SELECT"));

        let query = sparql_functions::valid_credentials();
        assert!(query.contains("NOW()"));
    }
}
