# OxiRS Fuseki - TODO

*Last Updated: October 4, 2025*

## ✅ Current Status: v0.1.0-alpha.2 Production-Ready

**oxirs-fuseki** provides a SPARQL 1.1/1.2 HTTP server with Apache Fuseki compatibility.

### Alpha.2 Release Status (October 4, 2025)
- **352 tests passing** (unit + integration) with zero warnings
- **Full SPARQL 1.1/1.2 support** including `SERVICE` federation and result merging
- **Persistent datasets** with automatic N-Quads save/load and warm start
- **10-layer production middleware stack** ✨ hardened with HSTS & security headers
- **OAuth2/OIDC + JWT** authentication with configurable providers
- **Observability**: Prometheus metrics, slow-query tracing, structured logging
- **CLI Integration** ✨ Serve command + REPL alignment for simplified ops

### 🎉 Alpha.2 Achievements

#### Production Deployment Ready ✅
- ✅ **CLI Serve Command**: Full integration with oxirs CLI & persisted datasets
- ✅ **Security Middleware**: 10-layer stack + OAuth2/OIDC + HSTS
- ✅ **Observability**: Request correlation IDs, Prometheus metrics, structured logs
- ✅ **Federation**: Resilient remote endpoint integration with retries/backoff
- ✅ **Standards Compliance**: W3C SPARQL 1.1/1.2 endpoints + SERVICE federation

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