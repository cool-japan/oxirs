# oxirs-modbus Development TODO

**Status**: Phase D - Month 1-4 (Under Development)
**Target**: v0.3.0 (Q2 2026)
**Priority**: CRITICAL (60% of factories use Modbus)

---

## Month 1: Protocol Foundation (Weeks 1-4)

### High Priority - Week 1-2: Modbus TCP Client ✅ COMPLETE

- [x] **TCP client implementation** (`protocol/tcp.rs`)
  - [x] TcpStream connection with timeout
  - [x] Transaction ID management (atomic counter)
  - [x] ADU (Application Data Unit) frame building
  - [x] Response parsing and validation
  - [x] Function code 0x03 (Read Holding Registers)
  - [x] Function code 0x06 (Write Single Register)
  - [x] Error response handling
  - [x] Unit tests (2 tests)

### High Priority - Week 3-4: Modbus RTU Client ✅ COMPLETE

- [x] **RTU client implementation** (`protocol/rtu.rs`)
  - [x] Serial port integration (tokio-serial)
  - [x] CRC16 calculation (`protocol/crc.rs`)
  - [x] RTU frame building (no transaction ID)
  - [x] 3.5-character gap detection
  - [x] Baud rate configuration (9600-115200)
  - [x] Function code 0x03, 0x04, 0x06
  - [x] Unit tests (2 tests)

### Medium Priority - Week 3-4: Additional Function Codes

- [ ] **Read operations** (`protocol/functions.rs`)
  - [ ] 0x01: Read Coils (digital outputs)
  - [ ] 0x02: Read Discrete Inputs (digital inputs)
  - [ ] 0x04: Read Input Registers (read-only analog)
- [ ] **Write operations**
  - [ ] 0x0F: Write Multiple Coils
  - [ ] 0x10: Write Multiple Registers
- [ ] Exception response handling (codes 0x01-0x0B)

---

## Month 2: RDF Integration (Weeks 5-8)

### High Priority - Week 5-6: Register Mapping ✅ COMPLETE

- [x] **Mapping engine** (`mapping/register_map.rs`)
  - [x] TOML configuration parser
  - [x] RegisterMapping struct (address, data_type, predicate, unit, scaling)
  - [x] ModbusDataType enum (INT16, UINT16, INT32, UINT32, FLOAT32, BIT)
  - [x] LinearScaling (multiplier, offset)
  - [x] Enum value mapping (u16 → string labels)
  - [x] to_rdf_literal() conversion
  - [x] Unit tests (12 tests)

- [x] **Data type conversions** (`mapping/data_types.rs`)
  - [x] INT16/UINT16 → xsd:short/xsd:unsignedShort
  - [x] INT32/UINT32 → xsd:int/xsd:unsignedInt (big-endian)
  - [x] FLOAT32 → xsd:float (IEEE 754, two registers)
  - [x] BIT extraction (0-15) → xsd:boolean
  - [x] Unit tests (11 tests)

### High Priority - Week 7: RDF Triple Generation ✅ COMPLETE

- [x] **Triple generator** (`rdf/triple_generator.rs`)
  - [x] ModbusTripleGenerator struct
  - [x] generate_triples() method
  - [x] Subject IRI construction (base_iri + device_id)
  - [x] Predicate IRI from mapping
  - [x] Object literal with datatype
  - [x] W3C PROV-O timestamp triples (prov:generatedAtTime)
  - [x] QUDT unit triples (optional)
  - [x] Unit tests (5 tests)

- [x] **Graph updater** (`rdf/graph_updater.rs`)
  - [x] SPARQL UPDATE execution
  - [x] Named graph support
  - [x] Batch INSERT DATA
  - [x] Error handling and retry logic

### Medium Priority - Week 8: Polling System ✅ PARTIAL

- [x] **Polling scheduler** (`polling/scheduler.rs`)
  - [x] Cron-like interval configuration
  - [x] Tokio interval timers
  - [x] Multi-device polling orchestration
  - [x] Error recovery and reconnection
  - [x] Unit tests (4 tests)

- [ ] **Change detector** (`polling/change_detector.rs`)
  - [x] Previous value tracking (implemented in triple_generator.rs)
  - [x] Deadband threshold configuration
  - [x] detect_changes() method (via deadband in RegisterMapping)
  - [x] Only emit RDF updates on significant changes
  - [ ] Standalone module (currently integrated)

- [x] **Batch reader** (`polling/batch_reader.rs`)
  - [x] Optimize sparse register reads (batch_reads() in RegisterMap)
  - [x] create_batches() algorithm (max 125 registers per request)
  - [x] Gap detection in address ranges
  - [ ] Parallel batch execution
  - [x] Unit tests (integrated)

---

## Month 3: SAMM Integration (Weeks 9-12)

### High Priority - Week 9-10: SAMM Aspect Model Mapping

- [ ] **SAMM integration** (`mapping/samm_integration.rs`)
  - [ ] ModbusDeviceAspect model generation
  - [ ] Register → SAMM Property mapping
  - [ ] Data type → SAMM Characteristic mapping
  - [ ] Unit → SAMM Unit mapping (QUDT)
  - [ ] Auto-generate .ttl files from register maps

### Medium Priority - Week 11-12: Real-World Testing

- [ ] **PLC testing**
  - [ ] Test with Schneider Modicon M221 PLC
  - [ ] Test with Siemens S7-1200 (if available)
  - [ ] Verify register read accuracy
  - [ ] Performance testing (1,000 devices × 1Hz)

- [ ] **Energy meter testing**
  - [ ] Test with Eastron SDM630 energy meter
  - [ ] Verify FLOAT32 parsing
  - [ ] Test unit conversion (W → kW)

---

## Month 4: Production Hardening (Weeks 13-16)

### High Priority - Week 13-14: Performance Optimization ✅ PARTIAL

- [x] **Connection pooling** (`client/connection_pool.rs`)
  - [x] Pool size configuration
  - [x] Connection health monitoring (integrated)
  - [x] Automatic reconnection on failure
  - [x] Unit tests (4 tests)

- [ ] **Health monitoring** (`client/health_monitor.rs`)
  - [x] Connection status tracking (integrated in pool)
  - [x] Error rate monitoring (integrated in pool)
  - [ ] Standalone health monitor module
  - [ ] Prometheus metrics

### High Priority - Week 15-16: Documentation

- [ ] **API documentation**
  - [ ] 95%+ rustdoc coverage
  - [ ] Code examples for all public APIs
  - [ ] Integration examples

- [ ] **User guides**
  - [ ] Quick start guide
  - [ ] Configuration reference
  - [ ] Troubleshooting guide
  - [ ] Device compatibility matrix

---

## Low Priority - Future Enhancements (v0.3.1+)

### Protocol Extensions

- [ ] Modbus ASCII support (legacy devices)
- [ ] Modbus over UDP (port 502)
- [ ] Modbus security extensions (TLS encryption)

### Advanced Features

- [ ] Register auto-discovery (scan device capabilities)
- [ ] Historical data playback (replay from logs)
- [ ] Modbus gateway mode (OxiRS as Modbus slave/server)
- [ ] Advanced data types (INT64, DOUBLE, STRING)

### Integration

- [ ] oxirs-vec integration for anomaly detection
- [ ] oxirs-tsdb integration for time-series compression
- [ ] GUI register browser (web interface)
- [ ] OPC UA → Modbus translation

---

## Test Coverage Goals

- **Unit tests**: 200+ (target)
- **Integration tests**: 50+ (target)
- **Benchmarks**: 5+ performance benchmarks
- **Coverage**: 95%+ line coverage

---

## Performance Validation

- [ ] Benchmark: 1,000 devices × 1Hz polling
- [ ] Benchmark: Read 100 registers in <10ms
- [ ] Load test: 7-day continuous operation
- [ ] Memory profiling: <10MB per device

---

## Documentation Requirements

- [ ] README.md with quick start
- [ ] API docs (rustdoc) - 95%+ coverage
- [x] 5+ working examples (5 created: simple_tcp, simple_rtu, mock_server_demo, register_mapping, rdf_integration)
- [ ] Industrial use case guide
- [ ] Device compatibility matrix

---

**Implementation Plan**: See `/tmp/oxirs_enhancement_modbus.md` for complete details.

**Next Milestone**: Week 4 - Modbus TCP/RTU protocol layer complete.
