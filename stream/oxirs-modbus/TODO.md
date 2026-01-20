# OxiRS Modbus - TODO

*Version: 0.1.0 | Last Updated: 2026-01-06*

## Status: Production Ready

OxiRS Modbus v0.1.0 provides industrial Modbus protocol support with RDF mapping for factory automation, energy management, and IoT integration.

### Features
- ✅ Modbus TCP client implementation
- ✅ Modbus RTU client implementation (serial)
- ✅ Transaction ID management and CRC16 calculation
- ✅ Function codes: Read Holding Registers (0x03), Read Input Registers (0x04), Write Single Register (0x06)
- ✅ Register mapping engine with TOML configuration
- ✅ Data type support (INT16, UINT16, INT32, UINT32, FLOAT32, BIT)
- ✅ Linear scaling and enum value mapping
- ✅ RDF triple generation with XSD datatypes
- ✅ W3C PROV-O timestamp tracking
- ✅ QUDT unit support
- ✅ Polling scheduler with interval configuration
- ✅ Connection pooling and health monitoring
- ✅ 40 tests passing

## Future Roadmap

### v0.2.0 - Extended Protocol Support (Q1 2026 - Expanded)
- [ ] Additional function codes (Read Coils 0x01, Read Discrete Inputs 0x02)
- [ ] Write Multiple Registers (0x10) and Write Multiple Coils (0x0F)
- [ ] SAMM aspect model integration
- [ ] Real-world PLC testing
- [ ] Prometheus metrics integration
- [ ] Modbus ASCII support (legacy devices)
- [ ] Modbus over TLS (security extensions)
- [ ] Register auto-discovery

### v1.0.0 - LTS Release (Q2 2026)
- [ ] Long-term support guarantees
- [ ] Enterprise integration features
- [ ] GUI register browser
- [ ] OPC UA translation support

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS Modbus v0.1.0 - Industrial IoT for semantic web*
