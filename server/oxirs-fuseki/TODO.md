# OxiRS Fuseki - TODO

*Version: 0.3.1 | Last Updated: June 6, 2026*

## Current Status

OxiRS Fuseki v0.3.1 is production-ready, providing a complete SPARQL 1.1/1.2 HTTP server with Apache Fuseki compatibility and modern enhancements.

### Production Features
- ✅ **SPARQL 1.1/1.2 Protocol** - Full W3C compliance with query and update support
- ✅ **Graph Store Protocol** - Direct graph access with multiple RDF formats
- ✅ **Validation Services** - Fuseki-compatible validation endpoints
- ✅ **Authentication & Authorization** - OAuth2/OIDC, JWT, RBAC, ReBAC
- ✅ **GraphQL API** - Modern query interface with interactive playground
- ✅ **REST API v2** - OpenAPI 3.0 with Swagger UI
- ✅ **WebSocket Support** - Real-time subscriptions and streaming
- ✅ **Performance Optimization** - Concurrency, memory pooling, request batching, result streaming
- ✅ **Production Operations** - Load balancing, edge caching, DDoS protection, security audit
- ✅ **Deployment Automation** - Docker, Kubernetes, Kubernetes Operator, Terraform, Ansible
- ✅ **Admin UI** - Modern web-based dashboard
- ✅ **Graph-Level RBAC** - Fine-grained graph access control
- ✅ **OAuth2 Refresh Token Rotation** - Replay detection and secure rotation
- ✅ **Query Logger** - Structured query audit logging
- ✅ **2144 tests passing** with zero warnings

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ Full SPARQL 1.1/1.2 protocol, authentication, GraphQL, WebSocket, 812 tests

### v0.2.3 - Current Release (March 16, 2026)
- ✅ Query result caching with intelligent invalidation
- ✅ Advanced connection pooling strategies
- ✅ Distributed query execution optimization
- ✅ Enhanced monitoring and alerting
- ✅ Full-text search integration
- ✅ Advanced federation capabilities
- ✅ Multi-region clustering improvements
- ✅ Enhanced AI integration
- ✅ Graph-level RBAC
- ✅ OAuth2 refresh token rotation with replay detection
- ✅ 2144 tests passing

### v0.3.0 - Planned (Q2 2026)
- [x] API stability harness (planned 2026-05-01)
  - **Goal:** Mirror round 17's `core/oxirs-core/src/api_surface.rs` pattern for the public surface of `oxirs-fuseki`. Concrete engineering interpretation of "API stability" — a parsed-AST baseline enforced by the test suite, blocking unintentional breaking changes to the server crate.
  - **Design:** New module `server/oxirs-fuseki/src/api_surface.rs` — copy the public types, parser, and diff machinery from `core/oxirs-core/src/api_surface.rs`. Data types (`TypeSig`, `FnSig`, `TraitSig`, `ApiSurface`, `SurfaceDiff`) are identical; only the source path differs. New bin `src/bin/api_snapshot.rs`. Initial baseline `api_baseline.json`. New test `tests/api_stability.rs`. Future cleanup: extract to shared crate once a third consumer appears.
  - **Files:** `server/oxirs-fuseki/src/api_surface.rs`, `src/bin/api_snapshot.rs`, `api_baseline.json`, `tests/api_stability.rs`, `Cargo.toml` (bin entry, syn/quote workspace deps), `src/lib.rs` (re-export `pub mod api_surface`).
  - **Prerequisites:** `syn`, `quote`, `serde_json` — all in workspace deps (added in round 17).
  - **Tests:** parse a fixture lib with known signatures → verify surface; diff a known modified surface → verify breaking flag; additive change is allowed; baseline round-trip JSON.
  - **Risk:** false positives on `pub use` re-exports. Mitigation: ignore re-exports that resolve to oxirs-core types; track only local declaration shape.
- [x] Enterprise SAML 2.0 SP (completed 2026-05-02)
  - Implemented `SamlResponseParser` with `quick-xml` 0.39 event-based parsing
  - XML parsing covers: `samlp:Response` → `saml:Assertion` → `saml:AttributeStatement`, `saml:NameID`, `saml:Conditions` (NotBefore/NotOnOrAfter/AudienceRestriction), `saml:AuthnStatement` (SessionNotOnOrAfter/SessionIndex)
  - `AuthnRequest::to_xml()` rebuilt with proper XML escaping via `write_xml_attr` + `xml_escape`
  - RSA-SHA256 signature verification via `ring` (avoids digest version conflict with `sha2 0.11`)
  - IdP certificate: PEM X.509, PEM PKCS#8, raw base64-DER all supported
  - `saml = ["dep:quick-xml", "dep:rsa"]` feature declared in Cargo.toml
  - 18 new integration tests in `tests/saml_test.rs`; 14 unit tests in `src/auth/saml.rs` — all passing
  - [x] RBAC policy templates (DBA, ReadOnly, Auditor) — `src/auth/policy_templates.rs` (completed 2026-05-17)
  - [x] FIPS 140-2 feature gate — `fips = []` in Cargo.toml, boundary doc RFC-003 (completed 2026-05-17)
  - [~] LDAP HA (high-availability LDAP failover/replica) — not yet implemented
  - [~] Cluster auth (cross-node authentication tokens) — not yet implemented
  - [~] Audit log export endpoint (HTTP API to stream `JsonLineAuditLogger` output) — not yet implemented
- [x] Complete Jena/Fuseki parity verification (completed 2026-04-30)
  - **Goal:** Verify wire-level / behavioral parity with Apache Jena Fuseki HTTP server over a defined request matrix and fix any gaps surfaced.
  - **Design:** Build parity matrix in `tests/jena_fuseki_parity.rs`. Endpoints: `/dataset/sparql` (query), `/dataset/update` (update), `/dataset/data` (graph store protocol), `/dataset/upload` (file upload), `$/datasets` (admin), `$/server`, `$/stats`, `$/ping`. Request matrix: 30+ representative requests covering protocol + error paths. For each: Content-Type, Accept negotiation, status codes, response shape (JSON Results, XML Results, Turtle, N-Triples, etc.). Reference outputs: pinned reference responses under `tests/fixtures/jena-fuseki-ref/` captured ahead of time from a Jena Fuseki snapshot (vendored). Gaps closed in `src/{handlers,protocol,content_negotiation}.rs`.
  - **Files:** `tests/jena_fuseki_parity.rs` (new), `tests/fixtures/jena-fuseki-ref/` (vendored), `src/{handlers,protocol,content_negotiation}.rs` (gap closure)
  - **Tests:** unit per-request shape against fixtures; integration full HTTP parity matrix (pass_rate >= 0.95 for spec-required; impl-detail divergences surface as warnings)
  - **Risk:** Jena Fuseki has Jena-specific behaviors not in any spec. Mitigation: classify each parity assertion as "spec-required" or "implementation-detail"; the gate covers spec-required only.
- [x] Comprehensive performance benchmarks (completed 2026-04-29)

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS Fuseki v0.3.1 - Production-ready SPARQL server*

## Proposed follow-ups

- API stability harness pattern: once a third consumer of the `api_surface` machinery appears, extract to a shared `oxirs-api-surface` crate.
