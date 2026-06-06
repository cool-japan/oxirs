# OxiRS DID - TODO

*Version: 0.3.1 | Last Updated: June 6, 2026*

## Status: Production Ready

OxiRS DID v0.3.1 provides W3C Decentralized Identifiers (DID) and Verifiable Credentials (VC) support for secure, decentralized identity management in semantic web applications.

### Features
- ✅ W3C DID Core specification compliance
- ✅ `did:key` method implementation
- ✅ `did:web` method implementation
- ✅ Ed25519 cryptographic suite (Ed25519Signature2020)
- ✅ Verifiable Credentials issuance and verification
- ✅ JSON-LD Signatures with RDFC-1.0 canonicalization
- ✅ DID Document CRUD operations
- ✅ Proof creation and verification
- ✅ Additional DID methods (`did:ethr`, `did:ion`, `did:pkh`)
- ✅ JWS signature support (JsonWebSignature2020)
- ✅ Key rotation mechanism
- ✅ Revocation lists (W3C Status List 2021)
- ✅ Credential store (W3C VC storage)
- ✅ Access control for DID operations
- ✅ Trust chain verification
- ✅ Presentation builder and VC presenter
- ✅ VC verifier, presentation request
- ✅ Key derivation and key manager
- ✅ 1043 tests passing

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ DID Core, did:key/web, Ed25519, VC issuance/verification, 8 core features

### v0.2.3 - Current Release (March 16, 2026)
- ✅ Additional DID methods (did:ethr, did:ion, did:pkh)
- ✅ JWS signature support (JsonWebSignature2020)
- ✅ Key rotation mechanism
- ✅ Revocation lists
- ✅ Credential store, trust chain, presentation builder
- ✅ Key derivation, key manager, access control
- ✅ 1043 tests passing

### v0.3.0 - Planned (Q2 2026)
- [x] W3C test suite compliance (planned 2026-04-17)
  - **Goal:** Build automated W3C DID test suite runner using official test vectors covering all MUST/SHOULD assertions; add JWT-VC and SD-JWT credential format support for W3C alignment
  - **Design:** Embed official W3C DID Core 1.0 and VC Data Model 2.0 test vectors as static JSON/CBOR; automated assertion runner validates all MUST requirements; compliance matrix report; JWT-VC encoding/decoding; SD-JWT (Selective Disclosure JWT) implementation per IETF draft
  - **Files:** tests/w3c_compliance.rs (new), src/vc/jwt_vc.rs (new), src/vc/sd_jwt.rs (new)
  - **Tests:** All W3C DID Core MUST assertions; VC Data Model 2.0 test vectors; JWT-VC round-trip; SD-JWT disclosure correctness
  - **Risk:** W3C test suite coverage may reveal undiscovered compliance gaps; treat as bug-discovery phase
- [x] Long-term support guarantees (policy: docs/policies/lts.md) (completed 2026-05-17 via RFC-001)
- [x] Enterprise security features — FIPS 140-2 feature gate (`fips = []` in Cargo.toml, RFC-003 boundary doc) completed 2026-05-17; full decomposition in `docs/policies/enterprise.md`
- [x] Hardware Security Module (HSM) support (planned 2026-04-17)
  - **Goal:** Replace mock KMS backends with real PKCS#11, AWS KMS, GCP Cloud KMS, and Azure Key Vault implementations behind feature gates; add cryptographic operation audit logging
  - **Design:** PKCS#11 via pkcs11 crate (pure Rust FFI); Pkcs11Signer implementing DIDSigner trait; supports YubiHSM, Thales, SoftHSM2; AwsKmsSigner via aws-sdk-kms; GcpKmsSigner via google-cloud-kms; AzureKvSigner via azure_security_keyvault; all behind feature = ["hsm", "aws-kms", "gcp-kms", "azure-kms"]; AuditLog appends every sign/verify with timestamp, key_id, operation type
  - **Files:** src/kms/pkcs11.rs (new), src/kms/aws.rs, src/kms/gcp.rs, src/kms/azure.rs, src/kms/audit.rs (new), Cargo.toml
  - **Prerequisites:** pkcs11 crate (feature-gated), aws-sdk-kms (feature-gated), google-cloud-kms (feature-gated), azure_security_keyvault (feature-gated)
  - **Tests:** SoftHSM2 PKCS#11 integration test (feature-gated); audit log completeness; mock provider still used for default unit tests
  - **Risk:** PKCS#11 C FFI requires SoftHSM2 on CI; keep all HSM tests behind feature flags
- [x] ZKP-based selective disclosure (planned 2026-04-17)
  - **Goal:** Harden ZKP selective disclosure by replacing hash-based Pedersen commitments with proper prime-order group commitments over curve25519; verify cryptographic soundness (binding + hiding properties)
  - **Design:** Replace SHA-256(domain || G || m || H || r) with proper Pedersen commitment over Ristretto255 group using curve25519-dalek; Fiat-Shamir transcript via merlin crate; ensure commitment binding and hiding; BBS+ selective disclosure soundness property tests; document migration path from old commitment scheme
  - **Files:** src/zkp/pedersen.rs, src/zkp/selective_disclosure.rs
  - **Prerequisites:** curve25519-dalek (latest), merlin (latest)
  - **Tests:** Pedersen binding soundness (adversary cannot open commitment to different value); hiding property (indistinguishability); selective disclosure round-trip; BBS+ unlinkability test
  - **Risk:** Breaking change to commitment scheme; provide migration utility and document in changelog

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS DID v0.2.3 - Decentralized identity for semantic web*
