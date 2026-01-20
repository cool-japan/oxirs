# OxiRS CAN Bus - TODO

*Version: 0.1.0 | Last Updated: 2026-01-06*

## Status: Production Ready

OxiRS CAN Bus v0.1.0 provides automotive CAN bus integration with J1939 protocol support, DBC file parsing, and RDF mapping for vehicle telematics and industrial automation.

### Features
- ✅ SocketCAN integration (Linux)
- ✅ CAN frame parsing (standard and extended IDs)
- ✅ CAN FD support (64-byte payloads)
- ✅ J1939 protocol implementation (PGN extraction, multi-packet reassembly)
- ✅ Common J1939 PGNs (engine data, vehicle speed, temperature)
- ✅ DBC file parser (messages, signals, value tables)
- ✅ Signal decoding (little/big endian, scaling, offsets)
- ✅ RDF triple generation from CAN frames
- ✅ W3C PROV-O timestamp tracking
- ✅ SAMM aspect model generation from DBC files
- ✅ 101 tests passing

## Future Roadmap

### v0.2.0 - Production Testing (Q1 2026 - Expanded)
- [ ] OBD-II testing with real vehicles
- [ ] J1939 testing with heavy vehicles
- [ ] Performance validation (10,000 messages/sec)
- [ ] Comprehensive API documentation
- [ ] UDS (Unified Diagnostic Services, ISO 14229)
- [ ] CANopen support (DS-301 profiles)
- [ ] Recording and playback (.asc, .blf formats)
- [ ] Bus analysis and monitoring tools

### v1.0.0 - LTS Release (Q2 2026)
- [ ] Long-term support guarantees
- [ ] Enterprise integration features
- [ ] GUI tools and web interface
- [ ] Automotive digital twin support

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

**Note**: This crate is Linux-specific. For macOS/Windows development, use virtual CAN (vcan) for testing.

---

*OxiRS CAN Bus v0.1.0 - Automotive telematics for semantic web*
