# OxiRS Stream - TODO

*Last Updated: October 4, 2025*

## ✅ Current Status: v0.1.0-alpha.2 Released (Experimental)

**oxirs-stream** provides real-time RDF data streaming (experimental feature).

### Alpha.2 Release Status (October 4, 2025)
- **Comprehensive test suite** with streaming + persistence scenarios
- **Kafka & NATS integrations** featuring checkpointed offsets and retries
- **Stream processing** aligned with CLI persistence and SPARQL federation
- **Metrics & monitoring** via Prometheus/SciRS2 instrumentation
- **CLI integration** for batch ingest/export to live streams
- **Released on crates.io**: `oxirs-stream = "0.1.0-alpha.2"` (experimental)

## 🎯 Post-Alpha Development Roadmap

### Beta Release Targets (v0.1.0-beta.1 - December 2025)

#### Stream Processing
- [ ] Advanced stream operators
- [ ] Windowing functions
- [ ] Aggregations
- [ ] Pattern matching

#### Integration
- [ ] Additional message brokers
- [ ] SPARQL stream extensions
- [ ] GraphQL subscriptions
- [ ] Storage integration

#### Performance
- [ ] Throughput optimization
- [ ] Latency reduction
- [ ] Memory usage
- [ ] Backpressure handling

#### Reliability
- [ ] Error handling
- [ ] Retry mechanisms
- [ ] Dead letter queues
- [ ] Monitoring and metrics

### v0.2.0 Targets (Q1 2026)
- [ ] Complex event processing
- [ ] Stream analytics
- [ ] Multi-stream joins
- [ ] Production hardening