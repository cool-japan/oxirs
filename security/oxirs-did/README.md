# oxirs-did

**W3C DID and Verifiable Credentials implementation with Signed RDF Graphs**

[![Crates.io](https://img.shields.io/crates/v/oxirs-did.svg)](https://crates.io/crates/oxirs-did)
[![docs.rs](https://docs.rs/oxirs-did/badge.svg)](https://docs.rs/oxirs-did)

Full implementation of W3C Decentralized Identifiers (DID) and Verifiable Credentials (VC) specifications, with support for cryptographically signed RDF graphs.

## Features

- **DID Core 1.0**: W3C Recommendation compliant
- **VC Data Model 2.0**: Verifiable Credentials with Ed25519 proofs
- **DID Methods**: `did:key` (default, no network), `did:web` (feature `did-web`), `did:ethr` (default, ERC-1056), `did:ion` (default, Sidetree/ION), `did:pkh` (CAIP-10 blockchain accounts)
- **RDFC-1.0**: RDF Dataset Canonicalization for graph signing
- **Proof Suites**: Ed25519Signature2020, JWS (`JsonWebSignature2020`), BBS+ signatures (feature `bbs-plus`, default)
- **Key Management**: Key rotation and lifecycle tracking, ECDH key agreement
- **Revocation**: W3C `StatusList2021` and `RevocationList2020`
- **Trust & Presentation**: Trust chain verification, Verifiable Presentation builder/presenter
- **Selective Disclosure**: ZKP-based Pedersen commitments (feature `zkp`, default)

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
use oxirs_did::{CredentialIssuer, CredentialSubject, DidResolver, Keystore};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Issuer identity — the keystore generates and holds the signing key
    let keystore = Arc::new(Keystore::new());
    let issuer_did = keystore.generate_ed25519().await?;

    // Create credential subject
    let subject = CredentialSubject::new(Some("did:key:z6Mk..."))
        .with_claim("email", "alice@example.com")
        .with_claim("role", "Researcher");

    // Issue (sign) the credential
    let resolver = Arc::new(DidResolver::new());
    let issuer = CredentialIssuer::new(keystore, resolver);
    let vc = issuer
        .issue(&issuer_did, subject, vec!["EmailCredential".to_string()])
        .await?;

    println!("{}", serde_json::to_string_pretty(&vc)?);

    Ok(())
}
```

### Verify Credential

```rust
use oxirs_did::{CredentialVerifier, DidResolver, VerifiableCredential};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let resolver = Arc::new(DidResolver::new());
    let verifier = CredentialVerifier::new(resolver);

    // Parse a VC received as JSON (e.g. from an API or file)
    let vc: VerifiableCredential = serde_json::from_str(vc_json)?;

    // Verify
    let result = verifier.verify(&vc).await?;

    if result.valid {
        println!("✓ Credential is VALID");
        println!("  Issued by: {}", result.issuer.unwrap_or_default());
    } else {
        println!("✗ Credential is INVALID: {}", result.error.unwrap_or_default());
    }

    for check in &result.checks {
        println!("  [{}] {}", if check.passed { "✓" } else { "✗" }, check.name);
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

### Additional Methods (Default)

- **did:ethr** — `did:ethr:[network:]<ethereum-address>`, based on the ERC-1056 Ethereum DID Registry contract
- **did:ion** — `did:ion:<unique-suffix>`, Sidetree-anchored DIDs on the ION network (Bitcoin-anchored PKI)
- **did:pkh** — `did:pkh:<CAIP-2 chain namespace>:<account address>`, wraps an existing blockchain account (Ethereum, Solana, Bitcoin, ...) as a DID with no separate registration step

## Feature Flags

```toml
[dependencies]
oxirs-did = { version = "0.3.2", features = ["did-web", "signed-graphs"] }
```

Available features:
- `did-key` (default) - did:key method
- `did-web` - did:web method (requires reqwest)
- `did-ebsi` - European Blockchain Services Infrastructure (requires reqwest)
- `did-ethr` (default) - did:ethr method (ERC-1056)
- `did-ion` (default) - did:ion method (Sidetree/ION)
- `vc-data-model-2` (default) - W3C VC 2.0
- `signed-graphs` - RDF graph signing/verification
- `key-management` - Key storage
- `bbs-plus` (default) - BBS+ signatures for selective disclosure
- `zkp` (default) - ZKP-based selective disclosure (Pedersen commitments)
- `zkp-ristretto` - Hardened Pedersen commitments over the Ristretto255 group
- `keygen` - RSA key-pair generation helpers
- `fips` - FIPS 140-2 cryptographic boundary marker (see `docs/policies/fips-boundary.md`)

## Use Cases

- **Trustworthy AI**: Sign training datasets for provenance tracking
- **IoT Identity**: Decentralized identity for edge devices
- **Supply Chain**: Verifiable product certifications
- **Research Data**: Signed datasets for reproducibility
- **Federated Systems**: Self-sovereign identity across systems

## Dependencies

- `ed25519-dalek` - Ed25519 signatures
- `p256`, `bls12_381_plus`, `rsa` - Additional signature suites (ECDSA P-256, BLS12-381/BBS+, RSA)
- `sha2`, `sha3`, `hmac` - Cryptographic hashing
- `bs58` - Multiformat (multibase) encoding
- `scirs2-core` - Secure random number generation

## License

Licensed under the Apache License, Version 2.0.
