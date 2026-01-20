//! Graph signature utilities

use super::{RdfTriple, SignedGraph};
use crate::proof::ed25519::Ed25519Signer;
use crate::{Did, DidResolver, DidResult, Proof, ProofPurpose, VerificationResult};

/// Sign a set of RDF triples
pub async fn sign_triples(
    graph_uri: &str,
    triples: Vec<RdfTriple>,
    issuer: &Did,
    signer: &Ed25519Signer,
) -> DidResult<SignedGraph> {
    let graph = SignedGraph::new(graph_uri, triples, issuer.clone());
    graph.sign(signer)
}

/// Verify a signed graph
pub async fn verify_graph(
    graph: &SignedGraph,
    resolver: &DidResolver,
) -> DidResult<VerificationResult> {
    graph.verify(resolver).await
}

/// Batch sign multiple graphs
pub async fn batch_sign(
    graphs: Vec<(&str, Vec<RdfTriple>)>,
    issuer: &Did,
    signer: &Ed25519Signer,
) -> DidResult<Vec<SignedGraph>> {
    let mut results = Vec::with_capacity(graphs.len());

    for (uri, triples) in graphs {
        let signed = sign_triples(uri, triples, issuer, signer).await?;
        results.push(signed);
    }

    Ok(results)
}

/// Batch verify multiple graphs
pub async fn batch_verify(
    graphs: &[SignedGraph],
    resolver: &DidResolver,
) -> DidResult<Vec<VerificationResult>> {
    let mut results = Vec::with_capacity(graphs.len());

    for graph in graphs {
        let result = verify_graph(graph, resolver).await?;
        results.push(result);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Keystore;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_sign_and_verify() {
        let keystore = Arc::new(Keystore::new());
        let resolver = DidResolver::new();

        // Generate key
        let issuer = keystore.generate_ed25519().await.unwrap();
        let signer = keystore.get_signer(&issuer).await.unwrap();

        let triples = vec![RdfTriple::iri(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        )];

        let signed = sign_triples("http://example.org/graph", triples, &issuer, &signer)
            .await
            .unwrap();

        let result = verify_graph(&signed, &resolver).await.unwrap();
        assert!(result.valid);
    }
}
