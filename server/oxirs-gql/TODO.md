# OxiRS GraphQL - TODO

*Version: 0.3.1 | Last Updated: June 6, 2026*

## Current Status

OxiRS GraphQL v0.3.1 is production-ready, providing a modern GraphQL interface for RDF data with automatic schema generation and AI-powered capabilities.

### Production Features
- ✅ **GraphQL Server** - Full GraphQL specification compliance
- ✅ **Automatic Schema Generation** - From RDFS/OWL with hot-reload
- ✅ **GraphQL ⇄ SPARQL Translation** - Bidirectional query translation
- ✅ **Subscriptions** - WebSocket-based real-time updates
- ✅ **Federation Support** - Apollo Federation v2 compatibility
- ✅ **Advanced Features** - File upload, APQ, persistent queries, rate limiting
- ✅ **Custom Directives** - @auth, @hasRole, @cacheControl, @constraint
- ✅ **Server-Sent Events** - HTTP-based real-time subscriptions
- ✅ **GraphQL Playground** - Advanced IDE with code generation
- ✅ **Code Generation** - Client code for TypeScript, Rust, Python, Go, Java, C#, Swift
- ✅ **Live Queries** - Automatic re-execution on data changes
- ✅ **Schema Documentation** - Markdown, HTML, JSON, OpenAPI output
- ✅ **Production Features** - CORS, JWT, OpenTelemetry, connection pooling
- ✅ **Enum Resolver** - Dynamic enum resolution for RDF types
- ✅ **Field Resolver Cache** - TTL+LRU caching for field resolution
- ✅ **Batch Resolver** - Optimized DataLoader-style batching
- ✅ **2081 tests passing** with zero warnings

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ Full GraphQL specification, automatic schema, subscriptions, federation, 1213 tests

### v0.2.3 - Current Release (March 16, 2026)
- ✅ GraphQL caching improvements (field resolver cache TTL+LRU)
- ✅ Advanced federation capabilities
- ✅ Enhanced subscription features (ChangeTracker, SubscriptionManager, Broadcaster)
- ✅ Performance optimization (DataLoader batching, parallel field resolution)
- ✅ Multi-tenant support
- ✅ Advanced security features
- ✅ Enhanced monitoring (OpenTelemetry, custom metrics)
- ✅ Production deployment templates
- ✅ Enum resolver, endpoint router, argument coercer
- ✅ 2081 tests passing

### v0.3.0 - Planned (Q2 2026)
- [x] Long-term support guarantees (policy: docs/policies/lts.md) (completed 2026-05-17 via RFC-001)
- [x] Complete GraphQL specification compliance (completed 2026-04-28)
  - **Implemented:** All 25 GraphQL June 2018 spec validation rules in `src/validation_spec.rs` (SpecValidator)
  - **Parser fix:** Fragment spread `...Name` parsing (peek_keyword word-boundary bug fixed)
  - **Files:** src/validation.rs (+SpecRule enum), src/validation_spec.rs (new, 880 lines), tests/spec_conformance.rs (new, 46 tests)
  - **Results:** 2141 tests passing, zero clippy warnings, zero fmt issues
- [x] Enterprise support (policy: docs/policies/enterprise.md, decomposed items listed therein) (completed 2026-05-17 via RFC-002)
- [x] Comprehensive benchmarks (completed 2026-04-29)

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS GraphQL v0.2.3 - Modern GraphQL interface for RDF*
