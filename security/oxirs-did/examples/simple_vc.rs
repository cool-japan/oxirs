//! Simple Verifiable Credential Example
//!
//! Demonstrates basic DID and VC functionality:
//! - Creating a DID
//! - Creating and signing a Verifiable Credential
//! - Verifying a credential

use oxirs_did::{
    CredentialIssuer, CredentialSubject, CredentialVerifier, Did, DidResolver, Keystore,
    VerifiableCredential,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Simple Verifiable Credential Example ===\n");

    // Step 1: Create issuer identity
    println!("Step 1: Creating issuer identity...");
    let keystore = Arc::new(Keystore::new());
    let issuer_did = keystore.generate_ed25519().await?;
    println!("  Issuer DID: {}\n", issuer_did);

    // Step 2: Create credential subject
    println!("Step 2: Creating credential subject...");
    let subject = CredentialSubject::new(Some("did:key:z6Mk..."))
        .with_claim("email", "alice@example.com")
        .with_claim("name", "Alice Smith")
        .with_claim("role", "Researcher");

    println!("  Subject ID: {}", subject.id.as_ref().unwrap());
    println!("  Claims: {:?}\n", subject.claims.keys());

    // Step 3: Create and issue credential
    println!("Step 3: Issuing credential...");
    let types = vec!["EmailCredential".to_string()];

    let resolver = Arc::new(DidResolver::new());
    let issuer = CredentialIssuer::new(keystore.clone(), resolver.clone());
    let signed_vc = issuer.issue(&issuer_did, subject, types).await?;

    println!("  Credential ID: {}", signed_vc.id.as_ref().unwrap());
    println!("  Types: {:?}", signed_vc.credential_type);
    println!(
        "  Proof: {}\n",
        if signed_vc.proof.is_some() {
            "✓ Present"
        } else {
            "✗ Missing"
        }
    );

    // Step 4: Verify credential
    println!("Step 4: Verifying credential...");
    let verifier = CredentialVerifier::new(resolver);

    let result = verifier.verify(&signed_vc).await?;

    println!(
        "  Verification: {}",
        if result.valid {
            "✓ VALID"
        } else {
            "✗ INVALID"
        }
    );
    println!("  Issuer: {}", result.issuer.as_ref().unwrap());

    for check in &result.checks {
        println!(
            "    {} {}: {}",
            if check.passed { "✓" } else { "✗" },
            check.name,
            check.details.as_deref().unwrap_or("OK")
        );
    }

    // Step 5: Display credential as JSON
    println!("\nStep 5: Credential JSON:");
    println!("{}", serde_json::to_string_pretty(&signed_vc)?);

    println!("\n✓ Example complete!");
    println!("\nThis demonstrates:");
    println!("  • Decentralized identity (DID)");
    println!("  • Cryptographic proofs (Ed25519)");
    println!("  • Verifiable claims (W3C VC 2.0)");
    println!("  • Self-contained verification (no central authority)");

    Ok(())
}
