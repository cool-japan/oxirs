# oxirs-canbus Development TODO

**Status**: Phase D - Month 3-5 (Under Development)
**Target**: v0.3.0 (Q2 2026)
**Priority**: HIGH (Automotive differentiation)

---

## Month 3: Socketcan + J1939 (Weeks 9-12)

### High Priority - Week 9-10: Socketcan Integration ✅ COMPLETE

- [x] **CAN interface** (`protocol/socketcan_client.rs`)
  - [x] CanSocket wrapper around socketcan crate (Linux only)
  - [x] Interface connection (can0, vcan0, etc.)
  - [x] Frame reception (recv_frame())
  - [x] Frame transmission (send_frame())
  - [x] CAN ID filtering (standard 11-bit, extended 29-bit)
  - [x] Error frame handling
  - [x] Unit tests (0 tests - Linux-only, requires hardware)

- [x] **CAN frame parsing** (`protocol/frame.rs`)
  - [x] CanFrame struct (id, data, dlc)
  - [x] Standard vs. Extended ID detection
  - [x] Remote Transmission Request (RTR) handling
  - [x] CAN FD support (64-byte payloads)
  - [x] Unit tests (25 tests)

### High Priority - Week 11-12: J1939 Protocol ✅ COMPLETE

- [x] **J1939 implementation** (`protocol/j1939.rs`)
  - [x] Parameter Group Number (PGN) extraction from 29-bit CAN ID
  - [x] Source Address extraction
  - [x] Priority field extraction
  - [x] Multi-packet message reassembly (TP.CM, TP.DT)
  - [x] J1939 address claiming
  - [x] Unit tests (17 tests)

- [x] **Common PGNs** (`protocol/j1939_pgns.rs`)
  - [x] PGN 61444: Electronic Engine Controller 1 (engine speed, torque)
  - [x] PGN 65265: Cruise Control/Vehicle Speed
  - [x] PGN 65262: Engine Temperature
  - [x] PGN 65263: Engine Fluid Level/Pressure
  - [x] PGN decoder registry with signal extraction (8 decoders)

---

## Month 4: DBC Parser + RDF Mapping (Weeks 13-16)

### High Priority - Week 13-14: DBC File Parser ✅ COMPLETE

- [x] **DBC parser** (`dbc/parser.rs`)
  - [x] Lexer (tokenize BO_, SG_, VAL_, CM_ directives)
  - [x] Parser (message and signal definitions)
  - [x] Message definitions (CAN ID, DLC, sender node)
  - [x] Signal definitions (start bit, length, byte order, scale, offset, min, max, unit)
  - [x] Value tables (enum mappings: 0="Off", 1="On")
  - [x] Comments and attributes
  - [x] Unit tests (17 tests)

- [x] **Signal decoding** (`dbc/signal.rs`)
  - [x] Extract signal bits from CAN frame data bytes
  - [x] Little-endian vs. big-endian (Intel vs. Motorola byte order)
  - [x] Signed vs. unsigned integer conversion
  - [x] Scaling and offset application: physical_value = raw_value * scale + offset
  - [x] Enum value lookup from value tables
  - [x] Unit tests (27 tests)

### High Priority - Week 15-16: RDF Integration ✅ COMPLETE

- [x] **RDF mapper** (`rdf/mapper.rs`)
  - [x] CanRdfMapper struct (holds DBC and RDF config)
  - [x] map_frame() - CAN frame → Vec<Triple>
  - [x] Subject IRI generation (vehicle ID + message name)
  - [x] Predicate IRI from signal name
  - [x] Object literal with XSD datatype (xsd:float, xsd:int, xsd:string)
  - [x] W3C PROV-O timestamp tracking (prov:generatedAtTime)
  - [x] Unit tests (10 tests)

- [x] **SAMM integration** (`rdf/samm_integration.rs`)
  - [x] Auto-generate SAMM Aspect Models from DBC files
  - [x] DBC Message → SAMM Aspect mapping
  - [x] DBC Signal → SAMM Property mapping
  - [x] Value table → SAMM Enumeration characteristic
  - [x] Export to .ttl format with proper namespaces
  - [x] Unit tests (5 tests)

---

## Month 5: Production Testing (Weeks 17-20)

### High Priority - Week 17-18: Real-World Testing

- [ ] **OBD-II testing**
  - [ ] Test with ELM327 OBD-II Bluetooth adapter
  - [ ] Decode Mode 01 PIDs (vehicle speed, RPM, coolant temp, MAF)
  - [ ] Verify physical value calculations
  - [ ] Test with multiple vehicle makes (Toyota, Honda, Ford)

- [ ] **J1939 testing**
  - [ ] Test with J1939 simulator or real heavy vehicle
  - [ ] Decode engine data (PGN 61444: rpm, torque)
  - [ ] Test multi-packet message reassembly (>8 bytes)
  - [ ] Verify Source Address handling

- [ ] **Performance testing**
  - [ ] 10,000 CAN messages/sec throughput
  - [ ] <1ms RDF conversion latency
  - [ ] 8 CAN interfaces simultaneously
  - [ ] Memory profiling (<50MB for 100K frames/sec)

### High Priority - Week 19-20: Documentation & Release

- [ ] **API documentation**
  - [ ] 95%+ rustdoc coverage
  - [ ] Module-level documentation with examples
  - [ ] DBC format specification guide
  - [ ] J1939 PGN reference table

- [x] **Examples** (6 created)
  - [x] can_frame_demo.rs - Basic CAN frame reception
  - [x] dbc_parsing.rs - Parse and print DBC file
  - [x] j1939_engine.rs - Decode heavy vehicle engine data
  - [x] obd2_diagnostics.rs - Automotive OBD-II diagnostics
  - [x] rdf_integration.rs - End-to-end CAN → RDF pipeline
  - [x] samm_export.rs - SAMM aspect model export

- [ ] **Guides**
  - [ ] Quick start guide
  - [ ] DBC file format tutorial
  - [ ] J1939 protocol explanation
  - [ ] Troubleshooting guide

---

## Low Priority - Future Enhancements (v0.3.1+)

### Protocol Extensions

- [ ] **UDS (Unified Diagnostic Services, ISO 14229)**
  - [ ] Diagnostic session control (0x10)
  - [ ] ECU reset (0x11)
  - [ ] Security access (0x27)
  - [ ] Read data by identifier (0x22)
  - [ ] Write data by identifier (0x2E)
  - [ ] Read/clear DTCs (0x19, 0x14)

- [ ] **CANopen support**
  - [ ] CANopen device profiles (DS-301)
  - [ ] SDO (Service Data Objects) for configuration
  - [ ] PDO (Process Data Objects) for real-time data
  - [ ] NMT (Network Management)
  - [ ] Emergency messages

### Advanced Features

- [ ] **Recording and playback**
  - [ ] .asc file format (Vector CANalyzer)
  - [ ] .blf file format (Vector Binary Log)
  - [ ] .log file format (PCAN-View)
  - [ ] Replay with original timing

- [ ] **Bus analysis**
  - [ ] Real-time CAN bus load monitoring
  - [ ] Message frequency analysis
  - [ ] Error frame detection and logging
  - [ ] Automatic DBC generation from traffic sniffing

- [ ] **Gateway mode**
  - [ ] Bridge multiple CAN interfaces
  - [ ] CAN → Ethernet gateway
  - [ ] Message filtering and routing
  - [ ] Protocol translation (CAN ↔ Ethernet)

### Integration

- [ ] **oxirs-vec integration**
  - [ ] CAN message pattern clustering
  - [ ] Anomaly detection in vehicle behavior

- [ ] **oxirs-tsdb integration**
  - [ ] Time-series storage for high-frequency signals
  - [ ] Compression for repetitive CAN data

- [ ] **GUI tools**
  - [ ] Web-based CAN bus analyzer
  - [ ] DBC editor interface
  - [ ] Live signal viewer

- [ ] **Automotive digital twins**
  - [ ] Vehicle state synchronization
  - [ ] Predictive maintenance
  - [ ] Fleet management integration

---

## Test Coverage Goals

- **Unit tests**: 200+ (target)
- **Integration tests**: 30+ (target)
- **Benchmarks**: 5+ performance benchmarks
- **Coverage**: 95%+ line coverage

---

## Performance Validation

- [ ] Benchmark: 10,000 CAN messages/sec ingestion
- [ ] Benchmark: <1ms RDF conversion per frame
- [ ] Load test: 8 CAN interfaces @ 1,000 msg/sec each
- [ ] Memory profiling: <50MB for 100K frames/sec
- [ ] Reliability test: 7-day continuous operation

---

## Documentation Requirements

- [ ] README.md with architecture diagram
- [ ] API docs (rustdoc) - 95%+ coverage
- [x] 5+ working examples (6 created: can_frame_demo, dbc_parsing, j1939_engine, obd2_diagnostics, rdf_integration, samm_export)
- [ ] DBC format specification
- [ ] J1939 PGN reference guide (top 20 PGNs)
- [ ] OBD-II PID reference (Mode 01 PIDs)

---

## Hardware Setup (Optional)

### For Testing

- [ ] **PEAK PCAN-USB** adapter (~$200) - Industry standard
- [ ] **OBD-II Bluetooth adapter** (~$20-50) - Consumer testing
- [ ] **Virtual CAN** setup (vcan0) - Development without hardware
- [ ] **Sample DBC files**:
  - [ ] Download from CSS Electronics: https://github.com/CSS-Electronics/can-database-example
  - [ ] OBD-II DBC: https://github.com/commaai/opendbc
  - [ ] J1939 DBC: SAE J1939 Digital Annex

### For Production

- [ ] Real vehicle for OBD-II testing (any car with CAN)
- [ ] Heavy vehicle for J1939 testing (truck, bus, tractor)
- [ ] CAN bus simulator (optional, ~$500)

---

## Standards Documentation

- [ ] ISO 11898-1:2015 - CAN data link layer
- [ ] SAE J1939 - Recommended Practice for Control and Communications Network
- [ ] ISO 15765 - Diagnostic communication over CAN (UDS)
- [ ] Vector CANdb++ DBC file format specification

---

**Implementation Plan**: See `/tmp/oxirs_enhancement_summary.md` for overview.

**Next Milestone**: Week 12 - Socketcan + J1939 complete.

**Note**: This crate is Linux-specific. For macOS/Windows development, use virtual CAN (vcan) for testing.
