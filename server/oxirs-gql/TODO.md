# OxiRS GraphQL - TODO

*Last Updated: September 30, 2025*

## ✅ Current Status: v0.1.0-alpha.1 Released

**oxirs-gql** provides a GraphQL interface for RDF data with automatic schema generation.

### Alpha Release Status
- **118 tests passing**
- **Zero compilation errors/warnings**
- **Basic GraphQL server**
- **Schema generation from RDF (basic)**
- **GraphQL to SPARQL translation (basic)**
- **Released on crates.io**: `oxirs-gql = "0.1.0-alpha.1"`

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