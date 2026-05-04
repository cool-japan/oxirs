# OxiRS CAN Bus - TODO

*Version: 0.3.0 | Last Updated: May 3, 2026*

## Status: Production Ready

OxiRS CAN Bus v0.2.3 provides automotive CAN bus integration with J1939 protocol support, DBC file parsing, and RDF mapping for vehicle telematics and industrial automation.

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

### v0.2.3 - Current Release (March 16, 2026)
- ✅ OBD-II decoder
- ✅ UDS (Unified Diagnostic Services, ISO 14229)
- ✅ CANopen support (DS-301 profiles)
- ✅ PGN decoder, protocol analyzer, bit timing
- ✅ Frame validator, frame aggregator, signal monitor
- ✅ Gateway bridge, recording extensions
- ✅ 1125 tests passing

### v0.3.0 - Planned (Q2 2026)
- [~] Long-term support guarantees (policy: docs/policies/lts.md)
- [~] Enterprise integration features (policy: docs/policies/enterprise.md, decomposed items listed therein)
- [x] GUI tools and web interface — Tauri 2.x J1939 CAN bus monitor, desktop/ui/canbus.html (completed 2026-05-02)
- [x] Automotive digital-twin support — J1939 ↔ DTDL bridge (completed 2026-05-01)
  - **Goal:** Bridge SAE-J1939 PGN telemetry (decoded by `oxirs-canbus`) to the DTDL digital-twin property model in `oxirs-physics`. Bidirectional: J1939 PGN → DTDL property write, AND DTDL command → J1939 request/response. Facade pattern: `J1939SourceFacade` + `DtdlSinkFacade` traits, plus mocks for test isolation.
  - **Design:** New module `stream/oxirs-canbus/src/digital_twin/` with `mod.rs`, `bridge.rs`, `mapper.rs`, `dtdl_sink.rs`, `j1939_source.rs`, `config.rs`. TOML `BridgeConfig` with `[[mapping]]` entries (pgn, spn, twin_property, direction). `MockJ1939Source` and `MockDtdlSink` for tests — no real network. `J1939DtdlBridge::run()` tokio task: read PGN frames → translate via mapper → write to sink; subscription loop for reverse direction. `oxirs-physics` dep for DTDL types. Graceful shutdown via `CancellationToken`. IMPLEMENT POLICY hedge: if oxirs-physics DTDL API isn't public, surface it first.
  - **Files:** `stream/oxirs-canbus/src/digital_twin/{mod.rs,bridge.rs,mapper.rs,dtdl_sink.rs,j1939_source.rs,config.rs}`, `src/lib.rs` (re-export), `examples/digital_twin_bridge.rs`, `tests/digital_twin_test.rs`, `Cargo.toml` (`oxirs-physics.workspace = true`, `tokio-util.workspace = true`).
  - **Prerequisites:** `oxirs-canbus` PGN decoder (already). `oxirs-physics` DTDL machinery (already per project memory).
  - **Tests:** PGN decode round-trip via mock source; SPN bit packing (J1939 NA indicator 0xFE/0xFF); bridge converges 5 frames; reverse direction; cancellation clean shutdown; missing-mapping typed error, never panics.
  - **Risk:** J1939 bit-packing subtleties. Mitigation: unit tests cover canonical SAE J1939-71 examples (engine RPM, coolant temp, distance).

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

**Note**: This crate is Linux-specific. For macOS/Windows development, use virtual CAN (vcan) for testing.

---

*OxiRS CAN Bus v0.2.3 - Automotive telematics for semantic web*

## Proposed follow-ups

- oxirs-physics DTDL API surfacing: if `oxirs_physics::digital_twin::Twin` property-write API isn't public, surface it in a dedicated oxirs-physics round, then wire to the real `DtdlSinkFacade` implementation.
