# oxirs-did

**W3C DID and Verifiable Credentials implementation with Signed RDF Graphs**

[![Crates.io](https://img.shields.io/crates/v/oxirs-did.svg)](https://crates.io/crates/oxirs-did)
[![docs.rs](https://docs.rs/oxirs-did/badge.svg)](https://docs.rs/oxirs-did)

Full implementation of W3C Decentralized Identifiers (DID) and Verifiable Credentials (VC) specifications, with support for cryptographically signed RDF graphs.

## Features

- **DID Core 1.0**: W3C Recommendation compliant
- **VC Data Model 2.0**: Verifiable Credentials with Ed25519 proofs
- **did:key**: Deterministic DIDs from public keys (no network)
- **did:web**: HTTPS-based DID resolution (optional)
- **RDFC-1.0**: RDF Dataset Canonicalization for graph signing
- **Ed25519Signature2020**: Cryptographic proof suite

## Standards Compliance

- ✅ [W3C DID Core 1.0](https://www.w3.org/TR/did-core/)
- ✅ [W3C VC Data Model 2.0](https://www.w3.org/TR/vc-data-model-2.0/)
- ✅ [Ed25519Signature2020](https://w3c-ccg.github.io/lds-ed25519-2020/)
- ✅ [RDFC-1.0](https://www.w3.org/TR/rdf-canon/)
- ✅ [Multicodec](https://github.com/multiformats/multicodec)
- ✅ [Multibase](https://github.com/multiformats/multibase)

## Quick Start

### DID Creation and Resolution

```rust
use oxirs_did::{Did, DidResolver};
use oxirs_did::proof::ed25519::Ed25519Signer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate Ed25519 keypair
    let signer = Ed25519Signer::generate();
    let public_key = signer.public_key_bytes();

    // Create did:key from public key
    let did = Did::new_key_ed25519(&public_key)?;
    println!("DID: {}", did);
    // Output: did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK

    // Resolve DID to DID Document
    let resolver = DidResolver::new();
    let doc = resolver.resolve(&did).await?;

    println!("DID Document: {}", serde_json::to_string_pretty(&doc)?);

    Ok(())
}
```

### Issue Verifiable Credential

```rust
use oxirs_did::{Did, VerifiableCredential, VcIssuer};
use oxirs_did::proof::ed25519::Ed25519Signer;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Issuer identity
    let issuer_signer = Ed25519Signer::generate();
    let issuer_did = Did::new_key_ed25519(&issuer_signer.public_key_bytes())?;

    // Create credential
    let mut vc = VerifiableCredential::new(
        issuer_did.clone(),
        json!({
            "id": "did:key:z6Mk...",
            "email": "alice@example.com",
            "role": "Researcher"
        }),
    )
    .with_type("EmailCredential")
    .with_expiration_days(365);

    // Issue (sign) the credential
    let issuer = VcIssuer::new(issuer_signer);
    issuer.issue(&mut vc, None).await?;

    println!("{}", serde_json::to_string_pretty(&vc)?);

    Ok(())
}
```

### Verify Credential

```rust
use oxirs_did::{DidResolver, VcVerifier, VerifiableCredential};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let resolver = DidResolver::new();
    let verifier = VcVerifier::new(resolver);

    // Parse VC from JSON
    let vc: VerifiableCredential = serde_json::from_str(vc_json)?;

    // Verify
    let result = verifier.verify(&vc).await?;

    if result.valid {
        println!("✓ Credential is VALID");
        println!("  Issued by: {}", result.issuer.unwrap());
    } else {
        println!("✗ Credential is INVALID");
        for error in &result.errors {
            println!("  - {}", error);
        }
    }

    Ok(())
}
```

### Sign RDF Graph

```rust
use oxirs_did::signed_graph::{SignedGraph, RdfTriple};
use oxirs_did::proof::ed25519::Ed25519Signer;
use oxirs_did::{Did, DidResolver};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create signer
    let signer = Ed25519Signer::generate();
    let did = Did::new_key_ed25519(&signer.public_key_bytes())?;

    // Create RDF graph
    let triples = vec![
        RdfTriple::iri(
            "http://data.example/entity-1",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://schema.org/Dataset"
        ),
    ];

    // Sign graph
    let graph = SignedGraph::new(
        "http://data.example/research-2025",
        triples,
        did.clone()
    );
    let signed = graph.sign(&signer)?;

    println!("Graph hash: {}", signed.hash()?);

    // Verify later
    let resolver = DidResolver::new();
    let valid = signed.verify(&resolver).await?;
    println!("Verification: {}", if valid.valid { "✓" } else { "✗" });

    Ok(())
}
```

## DID Methods

### did:key (Default)

No network required, deterministic from public key:

```
did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK
         └─ Base58btc(0xed01 + Ed25519_public_key)
```

### did:web (Optional, feature = "did-web")

HTTPS-based resolution:

```
did:web:example.com
→ https://example.com/.well-known/did.json

did:web:example.com:users:alice
→ https://example.com/users/alice/did.json

did:web:example.com%3A8080
→ https://example.com:8080/.well-known/did.json
```

## Feature Flags

```toml
[dependencies]
oxirs-did = { version = "0.1", features = ["did-web", "signed-graphs"] }
```

Available features:
- `did-key` (default) - did:key method
- `did-web` - did:web method (requires reqwest)
- `did-ebsi` - European Blockchain Services Infrastructure
- `vc-data-model-2` (default) - W3C VC 2.0
- `signed-graphs` - RDF graph signing/verification
- `key-management` - Key storage

## Use Cases

- **Trustworthy AI**: Sign training datasets for provenance tracking
- **IoT Identity**: Decentralized identity for edge devices
- **Supply Chain**: Verifiable product certifications
- **Research Data**: Signed datasets for reproducibility
- **Federated Systems**: Self-sovereign identity across systems

## Dependencies

- `ed25519-dalek` - Ed25519 signatures
- `multibase`, `bs58` - Multiformat encoding
- `sha2` - Cryptographic hashing
- `scirs2-core` - Secure random number generation

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
