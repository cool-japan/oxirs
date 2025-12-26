# oxirs-did TODO

## High Priority

- [ ] **Additional DID Methods**:
  - [ ] `did:ethr` - Ethereum-based DIDs
  - [ ] `did:ion` - Bitcoin-anchored DIDs (Microsoft/Sidetree)
  - [ ] `did:pkh` - Public key hash (blockchain agnostic)

- [ ] **JWS Signature Support**: JsonWebSignature2020
  - Parse and verify JWS format
  - Compact JWS serialization
  - Integration with existing Ed25519 suite

- [ ] **Key Rotation**:
  - Key rotation mechanism in DID Document
  - Previous key tracking for historical verification
  - Revocation lists

- [ ] **Hardware Security Module (HSM)**:
  - PKCS#11 interface
  - YubiKey integration
  - Cloud KMS support (AWS, Azure, GCP)

## Medium Priority

- [ ] **Additional Signature Suites**:
  - [ ] ES256K (secp256k1)
  - [ ] RS256 (RSA)
  - [ ] BBS+ (selective disclosure)

- [ ] **DID Resolution Caching**:
  - Persistent cache (SQLite/RocksDB)
  - TTL with HTTP cache headers
  - Cache invalidation strategies

- [ ] **Verifiable Presentation Protocols**:
  - DIDComm for presentation exchange
  - OpenID Connect for Verifiable Presentations (OIDC4VP)
  - CHAPI (Credential Handler API) support

- [ ] **Advanced VC Features**:
  - Credential schemas with JSON Schema
  - Credential refresh
  - Credential suspension/revocation
  - Status lists

## Low Priority

- [ ] **DID Method Registry**:
  - Plugin system for custom methods
  - Dynamic method loading
  - Method capability discovery

- [ ] **Batch Operations**:
  - Batch VC issuance
  - Batch verification
  - Parallel signature verification

- [ ] **Interoperability Testing**:
  - W3C test suite compliance
  - Cross-implementation testing
  - DIF Universal Resolver integration

## Security Enhancements

- [ ] **Key Encryption**:
  - Encrypt stored keys (AES-256-GCM)
  - Key derivation (PBKDF2, Argon2)
  - Secure key wiping on drop

- [ ] **ZKP Integration**:
  - Zero-knowledge proofs for selective disclosure
  - BBS+ signatures
  - Anonymous credentials

- [ ] **Audit Logging**:
  - All signature operations logged
  - Tamper-evident logs
  - Integration with W3C PROV-O

## RDF Integration

- [ ] **DID as RDF**:
  - Serialize DID Document to RDF
  - Store DIDs in SPARQL endpoint
  - Query DID Documents with SPARQL

- [ ] **VC in Knowledge Graphs**:
  - Automatic VC → RDF conversion
  - SPARQL queries over credentials
  - Trust graph construction from VCs

- [ ] **Integration with oxirs-shacl**:
  - SHACL shapes for DID Documents
  - VC validation with SHACL
  - Automated shape generation

## Standards Tracking

- [ ] Monitor W3C DID updates
- [ ] Track VC 2.0 errata
- [ ] Follow RDFC-1.0 updates
- [ ] Watch for new signature suites

## Testing & Quality

- [ ] Fuzzing for parsers (did:key, did:web)
- [ ] Property-based testing for canonicalization
- [ ] Security audit (external)
- [ ] Timing attack resistance verification

## Documentation

- [ ] Tutorial: Implementing SSI with OxiRS
- [ ] Guide: Signing knowledge graphs
- [ ] Comparison: DID methods (key vs web vs ion)
- [ ] Security best practices

## Performance

- [ ] Benchmark against other Rust DID libraries
- [ ] Optimize canonicalization (parallel sorting)
- [ ] SIMD for Ed25519 batch verification
- [ ] GPU acceleration for bulk operations

## Dependencies to Consider

- `zeroize` - Secure key wiping
- `secrecy` - Secret type wrappers
- `subtle` - Constant-time operations
