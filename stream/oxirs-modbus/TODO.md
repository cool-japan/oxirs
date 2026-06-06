# OxiRS Modbus - TODO

*Version: 0.3.1 | Last Updated: June 6, 2026*

## Status: Production Ready

OxiRS Modbus v0.3.1 provides industrial Modbus protocol support with RDF mapping for factory automation, energy management, and IoT integration.

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
- [x] Long-term support guarantees (policy: docs/policies/lts.md) (completed 2026-05-17 via RFC-001)
- [x] Enterprise integration features (policy: docs/policies/enterprise.md, decomposed items listed therein) (completed 2026-05-17 via RFC-002)
- [x] Terminal UI register browser (ratatui, feature "tui", completed 2026-05-02)
- [x] OPC UA translation support (planned 2026-05-01)
  - **Goal:** Bidirectional Modbus ↔ OPC UA bridge: map Modbus registers
    (coils, holding, input, discrete-input) to OPC UA Variables with type
    coercion (u16 / i16 / u32 / i32 / f32 / bool) and configurable polling.
    `OpcuaModbusBridge` runs as a tokio task. Server-side OPC UA endpoint
    optional; client-side OPC UA-to-Modbus also supported.
  - **Design:**
    - New module `stream/oxirs-modbus/src/opcua/`:
      `mod.rs`, `mapper.rs`, `type_coercion.rs`, `server.rs`, `client.rs`,
      `bridge.rs`, `config.rs`.
    - `BridgeConfig` (TOML):
      ```toml
      [[mapping]]
      modbus_register = 40001
      opcua_node_id = "ns=2;s=Temperature1"
      data_type = "f32"
      direction = "read"   # read | write | bidirectional
      [[mapping]]
      modbus_register = 1
      opcua_node_id = "ns=2;s=Pump1"
      data_type = "bool"
      ```
    - `RegisterMapper`: pure function `(register_value, schema) ->
      DataValue`; reverse `(DataValue, schema) -> register_value`. All
      coercion edge cases covered (negative→u16, fp32 NaN, bit-packed bool).
    - `OpcuaModbusBridge::run()`:
      1. Connect Modbus client (existing `tokio-modbus`).
      2. Start OPC UA server (existing `async-opcua` 0.18) — optional.
      3. Poll loop: read Modbus → translate → publish to OPC UA address space.
      4. Subscription loop: handle OPC UA writes → translate → write to Modbus.
    - Graceful shutdown via `CancellationToken`.
  - **Files:** `stream/oxirs-modbus/src/opcua/{mod.rs,mapper.rs,type_coercion.rs,server.rs,client.rs,bridge.rs,config.rs}`,
    `stream/oxirs-modbus/src/lib.rs`,
    `stream/oxirs-modbus/examples/opcua_bridge.rs`,
    `stream/oxirs-modbus/tests/opcua_bridge_test.rs`.
  - **Prerequisites:** `async-opcua` 0.18 already added per project memory.
    `tokio-modbus` already in deps.
  - **Tests:** type coercion (16 edge cases minimum: zero, max, min, negative,
    NaN, infinity, bit boundaries); register-mapping config parse; bridge unit
    test with in-process Modbus mock + OPC UA mock (don't open real sockets in
    CI); cancellation token shutdown is clean; missing-mapping error path.
  - **Risk:** `async-opcua` API differences from older `opcua` crate. Mitigation:
    isolate `async-opcua` calls behind a thin `OpcuaServer` / `OpcuaClient`
    facade so version bumps stay local. Use synthetic mock servers in tests.
- [x] Register auto-discovery (completed 2026-04-28)
  - **Goal:** Probe unknown Modbus device, infer register map (function code, address range, data types, scaling), emit candidate map consumable by existing register-validator.
  - **Design:** Walk FC 0x01-0x04 over configurable address window. Per-read: capture address/count/data pattern. Type inference: BE/LE signed/unsigned/float32/scaled-int. Scaling heuristics from value range. Emit RegisterMap (existing struct). Polling-aware throttle.
  - **Files:** src/discovery/{driver,inference,emitter}.rs (new), src/lib.rs (export discovery module)
  - **Tests:** unit type-inference heuristics + integration against simulated device (≥90% register inference)

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS Modbus v0.3.1 - Industrial IoT for semantic web*

## Proposed follow-ups

- [x] Enterprise integration features (policy: docs/policies/enterprise.md, decomposed items listed therein) (completed 2026-05-17 via RFC-002)
- [x] Terminal UI register browser (ratatui, feature "tui", completed 2026-05-02)
- [x] Long-term support guarantees (policy: docs/policies/lts.md) (completed 2026-05-17 via RFC-001)
