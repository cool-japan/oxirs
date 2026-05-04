//! Integration tests for the Modbus ↔ OPC UA bridge.
//!
//! All tests use in-process mocks — no real Modbus TCP connections or OPC UA
//! sockets are opened.

use oxirs_modbus::opcua::{
    client::{MockModbusClient, ModbusClientFacade},
    config::{BridgeConfig, DataTypeSpec, Direction, RegisterMapping},
    mapper::RegisterMapper,
    registers_to_value,
    server::MockOpcuaServer,
    value_to_registers, BridgeError, CoercionError, DataValue, OpcuaModbusBridge,
};
use tokio_util::sync::CancellationToken;

// ════════════════════════════════════════════════════════════════════════════
// Type coercion — 16 edge-case tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_u16_zero() {
    let v = registers_to_value(&[0u16], &DataTypeSpec::U16).expect("ok");
    assert_eq!(v, DataValue::U16(0));
}

#[test]
fn test_u16_max() {
    let v = registers_to_value(&[u16::MAX], &DataTypeSpec::U16).expect("ok");
    assert_eq!(v, DataValue::U16(65535));
}

#[test]
fn test_i16_zero() {
    let v = registers_to_value(&[0u16], &DataTypeSpec::I16).expect("ok");
    assert_eq!(v, DataValue::I16(0));
}

#[test]
fn test_i16_min() {
    // 0x8000 as u16 → i16::MIN (-32768)
    let v = registers_to_value(&[0x8000u16], &DataTypeSpec::I16).expect("ok");
    assert_eq!(v, DataValue::I16(i16::MIN));
}

#[test]
fn test_i16_max() {
    let v = registers_to_value(&[0x7FFFu16], &DataTypeSpec::I16).expect("ok");
    assert_eq!(v, DataValue::I16(i16::MAX));
}

#[test]
fn test_u32_zero() {
    let v = registers_to_value(&[0u16, 0u16], &DataTypeSpec::U32).expect("ok");
    assert_eq!(v, DataValue::U32(0));
}

#[test]
fn test_u32_max() {
    let v = registers_to_value(&[0xFFFFu16, 0xFFFFu16], &DataTypeSpec::U32).expect("ok");
    assert_eq!(v, DataValue::U32(u32::MAX));
}

#[test]
fn test_i32_min() {
    // 0x80000000 → i32::MIN
    let high = 0x8000u16;
    let low = 0x0000u16;
    let v = registers_to_value(&[high, low], &DataTypeSpec::I32).expect("ok");
    assert_eq!(v, DataValue::I32(i32::MIN));
}

#[test]
fn test_f32_normal() {
    let original = std::f32::consts::PI;
    let bits = original.to_bits();
    let high = (bits >> 16) as u16;
    let low = (bits & 0xFFFF) as u16;
    let v = registers_to_value(&[high, low], &DataTypeSpec::F32).expect("ok");
    if let DataValue::F32(f) = v {
        assert!((f - original).abs() < f32::EPSILON * 10.0);
    } else {
        panic!("expected F32 variant");
    }
}

#[test]
fn test_f32_nan_to_u16_is_error() {
    let nan = DataValue::F32(f32::NAN);
    let err = value_to_registers(&nan, &DataTypeSpec::U16).unwrap_err();
    assert!(
        matches!(err, CoercionError::NanNotRepresentable(_)),
        "expected NanNotRepresentable, got: {:?}",
        err
    );
}

#[test]
fn test_f32_infinity_to_i16_is_error() {
    let inf = DataValue::F32(f32::INFINITY);
    let err = value_to_registers(&inf, &DataTypeSpec::I16).unwrap_err();
    assert!(
        matches!(err, CoercionError::InfinityNotRepresentable(_)),
        "expected InfinityNotRepresentable, got: {:?}",
        err
    );
}

#[test]
fn test_bool_false() {
    let v = registers_to_value(&[0u16], &DataTypeSpec::Bool).expect("ok");
    assert_eq!(v, DataValue::Bool(false));
}

#[test]
fn test_bool_true_nonzero() {
    // Any nonzero value → true
    for &raw in &[1u16, 2, 100, 0xFFFF] {
        let v = registers_to_value(&[raw], &DataTypeSpec::Bool).expect("ok");
        assert_eq!(
            v,
            DataValue::Bool(true),
            "register value {} should be true",
            raw
        );
    }
}

#[test]
fn test_bool_to_register() {
    let regs = value_to_registers(&DataValue::Bool(true), &DataTypeSpec::Bool).expect("ok");
    assert_eq!(regs, vec![1u16]);

    let regs = value_to_registers(&DataValue::Bool(false), &DataTypeSpec::Bool).expect("ok");
    assert_eq!(regs, vec![0u16]);
}

#[test]
fn test_negative_i16_to_u16_is_out_of_range() {
    let neg = DataValue::I16(-1);
    let err = value_to_registers(&neg, &DataTypeSpec::U16).unwrap_err();
    assert!(
        matches!(err, CoercionError::OutOfRange { value: -1, .. }),
        "expected OutOfRange(-1), got: {:?}",
        err
    );
}

#[test]
fn test_bit_boundary_i16() {
    // 0x8000 → -32768
    let v = registers_to_value(&[0x8000u16], &DataTypeSpec::I16).expect("ok");
    assert_eq!(v, DataValue::I16(-32768));
}

// ════════════════════════════════════════════════════════════════════════════
// F32 round-trip (encode then decode)
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_f32_round_trip() {
    let values = [0.0_f32, -1.5, 1234.5678, f32::MAX, f32::MIN_POSITIVE];
    for &original in &values {
        let encoded = value_to_registers(&DataValue::F32(original), &DataTypeSpec::F32)
            .unwrap_or_else(|e| panic!("encode {} failed: {}", original, e));
        let decoded = registers_to_value(&encoded, &DataTypeSpec::F32)
            .unwrap_or_else(|e| panic!("decode {:?} failed: {}", encoded, e));
        if let DataValue::F32(f) = decoded {
            assert_eq!(
                f.to_bits(),
                original.to_bits(),
                "round-trip mismatch for {}",
                original
            );
        } else {
            panic!("expected F32 after round-trip");
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Config parse test
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_bridge_config_parse() {
    let toml_src = r#"
        poll_interval_ms = 250
        modbus_host = "10.0.0.1"
        modbus_port = 502

        [[mappings]]
        modbus_register = 40001
        opcua_node_id = "ns=2;s=Temperature1"
        data_type = "f32"
        direction = "read"

        [[mappings]]
        modbus_register = 1
        opcua_node_id = "ns=2;s=Pump1"
        data_type = "bool"
        direction = "bidirectional"
    "#;

    let config: BridgeConfig = toml::from_str(toml_src).expect("parse ok");
    assert_eq!(config.poll_interval_ms, 250);
    assert_eq!(config.modbus_host, "10.0.0.1");
    assert_eq!(config.modbus_port, 502);
    assert!(config.opcua_endpoint.is_none());
    assert_eq!(config.mappings.len(), 2);

    assert_eq!(config.mappings[0].modbus_register, 40001);
    assert_eq!(config.mappings[0].opcua_node_id, "ns=2;s=Temperature1");
    assert_eq!(config.mappings[0].data_type, DataTypeSpec::F32);
    assert_eq!(config.mappings[0].direction, Direction::Read);

    assert_eq!(config.mappings[1].modbus_register, 1);
    assert_eq!(config.mappings[1].opcua_node_id, "ns=2;s=Pump1");
    assert_eq!(config.mappings[1].data_type, DataTypeSpec::Bool);
    assert_eq!(config.mappings[1].direction, Direction::Bidirectional);
}

// ════════════════════════════════════════════════════════════════════════════
// Mapper tests
// ════════════════════════════════════════════════════════════════════════════

fn two_mapping_config() -> BridgeConfig {
    BridgeConfig {
        poll_interval_ms: 1000,
        modbus_host: "127.0.0.1".to_owned(),
        modbus_port: 502,
        opcua_endpoint: None,
        mappings: vec![
            RegisterMapping {
                modbus_register: 100,
                opcua_node_id: "ns=2;s=Speed".to_owned(),
                data_type: DataTypeSpec::U16,
                direction: Direction::Read,
            },
            RegisterMapping {
                modbus_register: 200,
                opcua_node_id: "ns=2;s=Valve".to_owned(),
                data_type: DataTypeSpec::Bool,
                direction: Direction::Bidirectional,
            },
        ],
    }
}

#[test]
fn test_mapper_find_by_register() {
    let mapper = RegisterMapper::new(two_mapping_config());
    let m = mapper.find_mapping(100).expect("should find register 100");
    assert_eq!(m.opcua_node_id, "ns=2;s=Speed");
    assert_eq!(m.data_type, DataTypeSpec::U16);
}

#[test]
fn test_mapper_missing() {
    let mapper = RegisterMapper::new(two_mapping_config());
    assert!(mapper.find_mapping(9999).is_none());
}

#[test]
fn test_mapper_find_by_node() {
    let mapper = RegisterMapper::new(two_mapping_config());
    let m = mapper
        .find_mapping_by_node("ns=2;s=Valve")
        .expect("should find node");
    assert_eq!(m.modbus_register, 200);
}

#[test]
fn test_mapper_readable_writable() {
    let mapper = RegisterMapper::new(two_mapping_config());
    // Speed(Read) + Valve(Bidirectional) = 2 readable
    assert_eq!(mapper.all_readable().len(), 2);
    // Only Valve(Bidirectional) is writable
    assert_eq!(mapper.all_writable().len(), 1);
}

// ════════════════════════════════════════════════════════════════════════════
// Bridge integration tests
// ════════════════════════════════════════════════════════════════════════════

fn bridge_config_with_register(
    register: u16,
    spec: DataTypeSpec,
    direction: Direction,
) -> BridgeConfig {
    BridgeConfig {
        poll_interval_ms: 100,
        modbus_host: "127.0.0.1".to_owned(),
        modbus_port: 502,
        opcua_endpoint: None,
        mappings: vec![RegisterMapping {
            modbus_register: register,
            opcua_node_id: format!("ns=2;s=Reg{}", register),
            data_type: spec,
            direction,
        }],
    }
}

#[tokio::test]
async fn test_bridge_poll_publishes() {
    let config = bridge_config_with_register(100, DataTypeSpec::U16, Direction::Read);
    let opcua = MockOpcuaServer::new();
    let modbus = MockModbusClient::new();

    // Pre-populate register 100 with value 42.
    modbus.set_register(100, vec![42u16]);

    let cancel = CancellationToken::new();
    let bridge = OpcuaModbusBridge::new(config, opcua, modbus, cancel);

    // Run a single poll cycle.
    bridge.poll_once().await;

    let published = bridge.opcua_server.get_published();
    assert_eq!(
        published.len(),
        1,
        "should have published exactly one value"
    );
    assert_eq!(published[0].0, "ns=2;s=Reg100");
    assert_eq!(published[0].1, DataValue::U16(42));
}

#[tokio::test]
async fn test_bridge_poll_publishes_f32() {
    let config = bridge_config_with_register(200, DataTypeSpec::F32, Direction::Read);
    let opcua = MockOpcuaServer::new();
    let modbus = MockModbusClient::new();

    // Encode pi as two big-endian words.
    let bits = std::f32::consts::PI.to_bits();
    let high = (bits >> 16) as u16;
    let low = (bits & 0xFFFF) as u16;
    modbus.set_register(200, vec![high, low]);

    let cancel = CancellationToken::new();
    let bridge = OpcuaModbusBridge::new(config, opcua, modbus, cancel);
    bridge.poll_once().await;

    let published = bridge.opcua_server.get_published();
    assert_eq!(published.len(), 1);
    if let DataValue::F32(f) = published[0].1 {
        assert!((f - std::f32::consts::PI).abs() < 1e-5_f32);
    } else {
        panic!("expected F32 in published value");
    }
}

#[tokio::test]
async fn test_bridge_write_forwards() {
    // Bidirectional: OPC UA write → Modbus register.
    let config = bridge_config_with_register(300, DataTypeSpec::U16, Direction::Bidirectional);
    let opcua = MockOpcuaServer::new();
    let modbus = MockModbusClient::new();

    let cancel = CancellationToken::new();
    let bridge = OpcuaModbusBridge::new(config, opcua, modbus, cancel);

    // Simulate an OPC UA write of U16(99) to register 300.
    bridge
        .forward_write_once("ns=2;s=Reg300", DataValue::U16(99))
        .await
        .expect("forward write ok");

    // Verify the register was written.
    let regs = bridge
        .modbus_client
        .read_holding_registers(300, 1)
        .await
        .expect("read ok");
    assert_eq!(regs, vec![99u16]);
}

#[tokio::test]
async fn test_bridge_write_read_direction_ignored() {
    // Write events to a Read-only mapping should be silently ignored.
    let config = bridge_config_with_register(400, DataTypeSpec::U16, Direction::Read);
    let opcua = MockOpcuaServer::new();
    let modbus = MockModbusClient::new();

    let cancel = CancellationToken::new();
    let bridge = OpcuaModbusBridge::new(config, opcua, modbus, cancel);

    // Should succeed (no error) but the register should NOT be written.
    bridge
        .forward_write_once("ns=2;s=Reg400", DataValue::U16(55))
        .await
        .expect("forward ok (ignored)");

    // Register 400 was never set, so reading it should fail.
    assert!(
        bridge
            .modbus_client
            .read_holding_registers(400, 1)
            .await
            .is_err(),
        "register 400 should not have been written"
    );
}

#[tokio::test]
async fn test_bridge_cancellation() {
    let config = BridgeConfig {
        poll_interval_ms: 60_000, // very long interval so we definitely cancel first
        modbus_host: "127.0.0.1".to_owned(),
        modbus_port: 502,
        opcua_endpoint: None,
        mappings: vec![],
    };
    let opcua = MockOpcuaServer::new();
    let modbus = MockModbusClient::new();

    let cancel = CancellationToken::new();
    let bridge = OpcuaModbusBridge::new(config, opcua, modbus, cancel.clone());

    // Run the bridge in a background task.
    let handle = tokio::spawn(async move { bridge.run().await });

    // Give the bridge a moment to enter its select! loop.
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    // Signal cancellation.
    cancel.cancel();

    // The bridge should shut down cleanly (Ok(())) within a short timeout.
    let result = tokio::time::timeout(std::time::Duration::from_millis(500), handle)
        .await
        .expect("bridge should shut down within timeout")
        .expect("task should not panic");

    assert!(
        result.is_ok(),
        "bridge should shut down without error: {:?}",
        result
    );
}

#[tokio::test]
async fn test_bridge_missing_modbus_register_does_not_crash() {
    // If a Modbus read fails, the bridge should log and continue, not crash.
    let config = bridge_config_with_register(999, DataTypeSpec::U16, Direction::Read);
    let opcua = MockOpcuaServer::new();
    let modbus = MockModbusClient::new();
    // Register 999 is NOT pre-populated → read_holding_registers will return an error.

    let cancel = CancellationToken::new();
    let bridge = OpcuaModbusBridge::new(config, opcua, modbus, cancel);

    // poll_once must not panic or return an error.
    bridge.poll_once().await;

    // Nothing should have been published.
    assert_eq!(bridge.opcua_server.get_published().len(), 0);
}

#[tokio::test]
async fn test_bridge_forward_write_unknown_node_ignored() {
    let config = two_mapping_config();
    let opcua = MockOpcuaServer::new();
    let modbus = MockModbusClient::new();

    let cancel = CancellationToken::new();
    let bridge = OpcuaModbusBridge::new(config, opcua, modbus, cancel);

    // Write to a node that doesn't exist in the config.
    let result = bridge
        .forward_write_once("ns=99;s=Unknown", DataValue::U16(0))
        .await;
    assert!(result.is_ok(), "unknown node should be silently ignored");
}
