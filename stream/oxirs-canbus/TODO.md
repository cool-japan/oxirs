# OxiRS CAN Bus - TODO

*Version: 0.2.2 | Last Updated: 2026-03-16*

## Status: Production Ready

OxiRS CAN Bus v0.2.2 provides automotive CAN bus integration with J1939 protocol support, DBC file parsing, and RDF mapping for vehicle telematics and industrial automation.

### Features
- ✅ SocketCAN integration (Linux)
- ✅ CAN frame parsing (standard and extended IDs)
- ✅ CAN FD support (64-byte payloads)
- ✅ J1939 protocol implementation (PGN extraction, multi-packet reassembly)
- ✅ Common J1939 PGNs (engine data, vehicle speed, temperature)
- ✅ DBC file parser (messages, signals, value tables)
- ✅ Signal decoding (little/big endian, scaling, offsets) Intel/Motorola DBC
- ✅ RDF triple generation from CAN frames
- ✅ W3C PROV-O timestamp tracking
- ✅ SAMM aspect model generation from DBC files
- ✅ UDS (Unified Diagnostic Services, ISO 14229)
- ✅ CANopen support (DS-301 profiles)
- ✅ OBD-II decoder
- ✅ Protocol analyzer
- ✅ PGN decoder
- ✅ Frame aggregator, frame validator, signal monitor
- ✅ CAN scheduler, gateway bridge, recording extensions
- ✅ 1125 tests passing

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ SocketCAN, J1939, DBC parser, RDF generation, SAMM, 101 tests

### v0.2.2 - Current Release (March 16, 2026)
- ✅ OBD-II decoder
- ✅ UDS (Unified Diagnostic Services, ISO 14229)
- ✅ CANopen support (DS-301 profiles)
- ✅ PGN decoder, protocol analyzer, bit timing
- ✅ Frame validator, frame aggregator, signal monitor
- ✅ Gateway bridge, recording extensions
- ✅ 1125 tests passing

### v0.3.0 - Planned (Q2 2026)
- [ ] Long-term support guarantees
- [ ] Enterprise integration features
- [ ] GUI tools and web interface
- [ ] Automotive digital twin support

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

**Note**: This crate is Linux-specific. For macOS/Windows development, use virtual CAN (vcan) for testing.

---

*OxiRS CAN Bus v0.2.2 - Automotive telematics for semantic web*
