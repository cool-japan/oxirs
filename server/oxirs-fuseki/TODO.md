# OxiRS Fuseki - TODO

*Last Updated: September 30, 2025*

## ✅ Current Status: v0.1.0-alpha.1 Released

**oxirs-fuseki** provides a SPARQL 1.1/1.2 HTTP server with Apache Fuseki compatibility.

### Alpha Release Status
- **349 tests passing**
- **Zero compilation errors/warnings**
- **Basic SPARQL 1.1/1.2 support**
- **Multi-dataset support (basic)**
- **Authentication system (in progress)**
- **Released on crates.io**: `oxirs-fuseki = "0.1.0-alpha.1"`

## 🎯 Post-Alpha Development Roadmap

### Beta Release Targets (v0.1.0-beta.1 - December 2025)

#### Authentication & Security
- [ ] Complete OAuth2/OIDC implementation
- [ ] Role-based access control (RBAC)
- [ ] TLS/SSL support
- [ ] Security hardening and audit

#### Performance
- [ ] Connection pooling optimization
- [ ] Query result caching
- [ ] Concurrent request handling improvements
- [ ] Memory usage optimization

#### Features
- [ ] Complete SPARQL Update support
- [ ] WebSocket subscriptions
- [ ] Admin UI enhancements
- [ ] Dataset management API

#### Stability
- [ ] Production error handling
- [ ] Comprehensive logging
- [ ] Health checks and monitoring
- [ ] Graceful shutdown and restart

### v0.2.0 Targets (Q1 2026)
- [ ] Full Fuseki feature parity
- [ ] Advanced federation support
- [ ] Real-time update notifications
- [ ] Performance profiling tools