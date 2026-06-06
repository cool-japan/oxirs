# FIPS 140-2 Cryptographic Boundary — RFC-003

**Status:** Feature-gate available (v0.3.1+)
**Effective:** 2026-05-17

## Scope

The `fips` feature flag enables FIPS 140-2 validated cryptographic operations
in OxiRS via the `ring` crate's FIPS-validated module.

## Crates in FIPS Boundary

| Crate | Feature flag | Scope |
|---|---|---|
| `oxirs-did` | `fips` | DID key generation, signature verification, VC signing |
| `oxirs-fuseki` | `fips` | TLS termination, JWT signing, OAuth token validation |

## Crates Out of Boundary

All other OxiRS crates are not in the FIPS cryptographic boundary. They may
use non-FIPS primitives for non-security purposes (hashing for indexing, etc.).

## Validated Algorithms (when fips feature enabled)

- AES-128-GCM, AES-256-GCM (symmetric encryption)
- ECDSA P-256, P-384 (signature generation/verification)
- RSA-2048, RSA-4096 with SHA-256/384/512 (signature)
- HMAC-SHA-256, HMAC-SHA-384, HMAC-SHA-512 (authentication)
- SHA-256, SHA-384, SHA-512 (hashing)

## Building with FIPS

```bash
cargo build -p oxirs-fuseki --features fips
cargo build -p oxirs-did --features fips
```

The `fips` feature is **not** included in the `default` feature set and must
be explicitly enabled. It is mutually exclusive with some `ring` configurations
— consult the `ring` crate documentation for your platform's requirements.

## Audit Evidence

Enable the `fips` feature and run `cargo test` to generate compliance evidence.
All cryptographic operations in the FIPS boundary use the validated module
when this flag is active.

## References

- NIST FIPS 140-2: https://csrc.nist.gov/publications/detail/fips/140/2/final
- ring FIPS support: https://github.com/briansmith/ring
- RFC-002 (Enterprise features): `docs/policies/enterprise.md`
