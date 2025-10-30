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
- **Released on crates.io**: `oxirs-gql = "0.1.0-beta.1"`

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

## 🎯 v0.1.0 Complete Feature Roadmap

### v0.1.0 Final Release Targets (Q4 2025) - ALL FEATURES

#### Federation Support (Target: v0.1.0)
- [ ] Apollo Federation v2 compatibility
- [ ] Federated entity resolution
- [ ] Subgraph composition
- [ ] Gateway implementation
- [ ] Service discovery and health checks
- [ ] Distributed tracing across subgraphs
- [ ] Schema stitching
- [ ] Cross-graph query optimization

#### Custom Directives (Target: v0.1.0)
- [ ] User-defined directives framework
- [ ] Authorization directives (@auth, @hasRole)
- [ ] Caching directives (@cacheControl)
- [ ] Deprecation directives (@deprecated)
- [ ] Rate limiting directives (@rateLimit)
- [ ] Transformation directives (@uppercase, @lowercase)
- [ ] Validation directives (@constraint, @pattern)
- [ ] Cost analysis directives (@cost)

#### File Upload Support (Target: v0.1.0)
- [ ] Multipart file upload handling
- [ ] Streaming uploads for large files
- [ ] Multiple file uploads
- [ ] Upload progress tracking
- [ ] File type validation
- [ ] Size limit enforcement
- [ ] Virus scanning integration
- [ ] Cloud storage upload (S3, GCS, Azure)

#### Rate Limiting (Target: v0.1.0)
- [ ] Query complexity analysis
- [ ] Depth limiting
- [ ] Breadth limiting
- [ ] Token bucket rate limiting
- [ ] Per-user/API key limits
- [ ] Sliding window counters
- [ ] Distributed rate limiting
- [ ] Adaptive rate limiting based on load

#### Query Optimization (Target: v0.1.0)
- [ ] Query cost estimation
- [ ] DataLoader for N+1 prevention
- [ ] Batch query optimization
- [ ] Persistent query documents
- [ ] Automatic persisted queries (APQ)
- [ ] Query allow/deny lists
- [ ] Query complexity scoring
- [ ] Field-level caching

#### Real-time Features (Target: v0.1.0)
- [ ] GraphQL subscriptions over WebSocket
- [ ] Server-sent events (SSE) support
- [ ] Live queries with automatic updates
- [ ] Subscription filtering
- [ ] Connection management
- [ ] Reconnection strategies
- [ ] Backpressure handling
- [ ] Subscription authentication

#### Developer Experience (Target: v0.1.0)
- [ ] GraphiQL IDE integration
- [ ] GraphQL Playground
- [ ] Schema documentation generator
- [ ] API explorer with examples
- [ ] Query builder UI
- [ ] Performance profiler
- [ ] Error tracking and reporting
- [ ] Code generation for clients

#### Production Features (Target: v0.1.0)
- [ ] Horizontal scaling support
- [ ] Connection pooling
- [ ] Health check endpoints
- [ ] Metrics and monitoring
- [ ] Distributed tracing (OpenTelemetry)
- [ ] Request logging
- [ ] CORS configuration
- [ ] JWT authentication