# OxiRS Modbus - TODO

*Version: 0.2.3 | Last Updated: 2026-03-16*

## Status: Production Ready

OxiRS Modbus v0.2.3 provides industrial Modbus protocol support with RDF mapping for factory automation, energy management, and IoT integration.

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
- ✅ Additional function codes: Read Coils (0x01), Read Discrete Inputs (0x02), Write Multiple Coils (0x0F)
- ✅ Coil register map (FC01/02/05/15)
- ✅ SAMM aspect model integration
- ✅ Prometheus metrics integration
- ✅ Modbus ASCII support (legacy devices)
- ✅ Modbus over TLS (security extensions)
- ✅ Register watcher for change detection
- ✅ Register encoder and validator
- ✅ 1095 tests passing

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ Modbus TCP/RTU, register mapping, RDF generation, 40 tests

### v0.2.3 - Current Release (March 16, 2026)
- ✅ Additional function codes (Read Coils 0x01, Read Discrete Inputs 0x02)
- ✅ Write Multiple Registers (0x10) and Write Multiple Coils (0x0F)
- ✅ SAMM aspect model integration
- ✅ Prometheus metrics integration
- ✅ Modbus ASCII support (legacy devices)
- ✅ Modbus over TLS (security extensions)
- ✅ Holding register bank, coil register map, signal decoder
- ✅ Diagnostic monitor, register watcher, register encoder
- ✅ 1095 tests passing

### v0.3.0 - Planned (Q2 2026)
- [ ] Long-term support guarantees
- [ ] Enterprise integration features
- [ ] GUI register browser
- [ ] OPC UA translation support
- [ ] Register auto-discovery

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS Modbus v0.2.3 - Industrial IoT for semantic web*
