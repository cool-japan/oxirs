# OxiRS GraphQL - TODO

*Last Updated: October 12, 2025*

## ✅ Current Status: v0.1.0-alpha.3 Released

**oxirs-gql** provides a GraphQL interface for RDF data with automatic schema generation.

### Alpha.3 Release Status (October 12, 2025)
- **118 tests passing** with zero warnings (unit + integration)
- **GraphQL server** synchronized with persisted datasets & CLI configs
- **Schema generation** with hot-reload and prefix-aware mapping
- **GraphQL ⇄ SPARQL translation** covering vector/federation resolvers
- **Subscription bridge** to streaming SPARQL updates (experimental)
- **Released on crates.io**: `oxirs-gql = "0.1.0-alpha.3"`

## 🎯 Post-Alpha Development Roadmap

### Beta Release Targets (v0.1.0-beta.1 - December 2025)

#### Schema Generation
- [ ] Advanced schema generation from RDFS/OWL
- [ ] Custom type mappings
- [ ] Schema caching and hot-reload
- [ ] Schema stitching support

#### Query Translation
- [ ] Improved GraphQL to SPARQL translation
- [ ] Complex query support
- [ ] Pagination and filtering
- [ ] Aggregation queries

#### Features
- [ ] GraphQL subscriptions (WebSocket)
- [ ] DataLoader for batching
- [ ] Query complexity analysis
- [ ] Response caching

#### Developer Experience
- [ ] GraphiQL integration
- [ ] Schema introspection improvements
- [ ] Better error messages
- [ ] Query debugging tools

### v0.2.0 Targets (Q1 2026)
- [ ] Federation support
- [ ] Custom directives
- [ ] File upload support
- [ ] Rate limiting