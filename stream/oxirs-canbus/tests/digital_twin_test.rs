//! Integration tests for the J1939 ↔ DTDL bridge.
//!
//! These tests exercise the bridge end-to-end with mock source and sink
//! implementations.  No real network or hardware is required.

use oxirs_canbus::digital_twin::client::J1939Frame;
use oxirs_canbus::digital_twin::mapper::{extract_spn, spn_to_twin_value, SpnValue};
use oxirs_canbus::digital_twin::{
    BridgeConfig, J1939DtdlBridge, MappingDirection, MockDtdlSink, MockJ1939Source, RegisterMapping,
};
use oxirs_physics::digital_twin::twin_value::TwinValue;
use tokio_util::sync::CancellationToken;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_config() -> BridgeConfig {
    BridgeConfig {
        poll_interval_ms: 10,
        mappings: vec![RegisterMapping {
            pgn: 65262,
            spn: 0,
            twin_property: "engine.coolant_temp_c".to_string(),
            direction: MappingDirection::Read,
        }],
    }
}

fn multi_pgn_config() -> BridgeConfig {
    BridgeConfig {
        poll_interval_ms: 10,
        mappings: vec![
            RegisterMapping {
                pgn: 65262,
                spn: 0,
                twin_property: "engine.coolant_temp_c".to_string(),
                direction: MappingDirection::Read,
            },
            RegisterMapping {
                pgn: 65265,
                spn: 0,
                twin_property: "vehicle.speed_kmh".to_string(),
                direction: MappingDirection::Read,
            },
            RegisterMapping {
                pgn: 61444,
                spn: 0,
                twin_property: "engine.rpm".to_string(),
                direction: MappingDirection::Bidirectional,
            },
        ],
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bridge integration tests
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_bridge_processes_frames() {
    let frames = vec![J1939Frame::new(65262, 0, [75, 0, 0, 0, 0, 0, 0, 0])];
    let source = MockJ1939Source::new(frames);
    let sink = MockDtdlSink::new();
    let cancel = CancellationToken::new();
    let mut bridge = J1939DtdlBridge::new(make_config(), source, sink.clone(), cancel);
    bridge.run().await.expect("bridge should complete");
    let val = sink
        .get("engine.coolant_temp_c")
        .expect("property should be set");
    assert_eq!(val, TwinValue::Integer(75));
}

#[tokio::test]
async fn test_bridge_cancels_cleanly() {
    let source = MockJ1939Source::new(vec![]);
    let sink = MockDtdlSink::new();
    let cancel = CancellationToken::new();
    cancel.cancel();
    let mut bridge = J1939DtdlBridge::new(make_config(), source, sink, cancel);
    bridge
        .run()
        .await
        .expect("cancelled bridge should complete");
}

#[test]
fn test_na_indicator_not_written() {
    let data = [0xFE, 0, 0, 0, 0, 0, 0, 0];
    let spn_val = extract_spn(&data, 110);
    assert_eq!(spn_val, SpnValue::NotAvailable);
    assert!(spn_to_twin_value(spn_val).is_none());
}

#[test]
fn test_na_indicator_0xff_not_written() {
    let data = [0xFF, 0, 0, 0, 0, 0, 0, 0];
    let spn_val = extract_spn(&data, 110);
    assert_eq!(spn_val, SpnValue::NotAvailable);
    assert!(spn_to_twin_value(spn_val).is_none());
}

#[test]
fn test_config_deserialization() {
    let toml_str = r#"
poll_interval_ms = 100
[[mapping]]
pgn = 65262
spn = 110
twin_property = "engine.coolant_temp_c"
direction = "read"
"#;
    let cfg: BridgeConfig = toml::from_str(toml_str).expect("should deserialize");
    assert_eq!(cfg.mappings.len(), 1);
    assert_eq!(cfg.mappings[0].pgn, 65262);
    assert_eq!(cfg.mappings[0].spn, 110);
    assert_eq!(cfg.mappings[0].twin_property, "engine.coolant_temp_c");
    assert_eq!(cfg.mappings[0].direction, MappingDirection::Read);
}

#[tokio::test]
async fn test_bridge_multiple_pgns() {
    let frames = vec![
        J1939Frame::new(65262, 0, [90, 0, 0, 0, 0, 0, 0, 0]),
        J1939Frame::new(65265, 0, [55, 0, 0, 0, 0, 0, 0, 0]),
        J1939Frame::new(61444, 0, [120, 0, 0, 0, 0, 0, 0, 0]),
    ];
    let source = MockJ1939Source::new(frames);
    let sink = MockDtdlSink::new();
    let cancel = CancellationToken::new();
    let mut bridge = J1939DtdlBridge::new(multi_pgn_config(), source, sink.clone(), cancel);
    bridge.run().await.expect("bridge should complete");

    assert_eq!(
        sink.get("engine.coolant_temp_c"),
        Some(TwinValue::Integer(90))
    );
    assert_eq!(sink.get("vehicle.speed_kmh"), Some(TwinValue::Integer(55)));
    assert_eq!(sink.get("engine.rpm"), Some(TwinValue::Integer(120)));
}

#[tokio::test]
async fn test_bridge_na_frame_skips_write() {
    let frames = vec![J1939Frame::new(65262, 0, [0xFE, 0, 0, 0, 0, 0, 0, 0])];
    let source = MockJ1939Source::new(frames);
    let sink = MockDtdlSink::new();
    let cancel = CancellationToken::new();
    let mut bridge = J1939DtdlBridge::new(make_config(), source, sink.clone(), cancel);
    bridge.run().await.expect("bridge should complete");
    assert!(
        sink.get("engine.coolant_temp_c").is_none(),
        "NA indicator must not write to the twin"
    );
}

#[tokio::test]
async fn test_bridge_unmapped_pgn_ignored() {
    // PGN 99999 is not in the config
    let frames = vec![J1939Frame::new(99999, 0, [42, 0, 0, 0, 0, 0, 0, 0])];
    let source = MockJ1939Source::new(frames);
    let sink = MockDtdlSink::new();
    let cancel = CancellationToken::new();
    let mut bridge = J1939DtdlBridge::new(make_config(), source, sink.clone(), cancel);
    bridge.run().await.expect("bridge should complete");
    assert!(sink.is_empty(), "unmapped PGN should not write anything");
}

#[tokio::test]
async fn test_bridge_converges_five_frames() {
    // Simulate 5 consecutive ET1 frames arriving; the last value should win.
    let frames: Vec<J1939Frame> = (60u8..=64u8)
        .map(|v| J1939Frame::new(65262, 0, [v, 0, 0, 0, 0, 0, 0, 0]))
        .collect();
    assert_eq!(frames.len(), 5);

    let source = MockJ1939Source::new(frames);
    let sink = MockDtdlSink::new();
    let cancel = CancellationToken::new();
    let mut bridge = J1939DtdlBridge::new(make_config(), source, sink.clone(), cancel);
    bridge.run().await.expect("bridge should complete");

    assert_eq!(
        sink.get("engine.coolant_temp_c"),
        Some(TwinValue::Integer(64)),
        "last frame value should be written"
    );
}

#[tokio::test]
async fn test_bridge_never_panics_on_boundary_values() {
    // Boundary: 0x00 and 0xFD (max valid, just below NA indicators)
    let frames = vec![
        J1939Frame::new(65262, 0, [0x00, 0, 0, 0, 0, 0, 0, 0]),
        J1939Frame::new(65262, 0, [0xFD, 0, 0, 0, 0, 0, 0, 0]),
    ];
    let source = MockJ1939Source::new(frames);
    let sink = MockDtdlSink::new();
    let cancel = CancellationToken::new();
    let mut bridge = J1939DtdlBridge::new(make_config(), source, sink.clone(), cancel);
    bridge.run().await.expect("bridge should not panic");
    // 0xFD = 253 — valid, should be written
    assert_eq!(
        sink.get("engine.coolant_temp_c"),
        Some(TwinValue::Integer(253))
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// SPN value unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_extract_spn_standard_byte() {
    let data = [100u8, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(extract_spn(&data, 84), SpnValue::Integer(100));
}

#[test]
fn test_extract_spn_zero_byte() {
    let data = [0u8; 8];
    assert_eq!(extract_spn(&data, 0), SpnValue::Integer(0));
}

#[test]
fn test_extract_spn_max_valid_byte() {
    let data = [0xFDu8, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(extract_spn(&data, 110), SpnValue::Integer(253));
}

#[test]
fn test_twin_value_integer_round_trip() {
    let spn = SpnValue::Integer(42);
    assert_eq!(spn_to_twin_value(spn), Some(TwinValue::Integer(42)));
}

#[test]
fn test_twin_value_float_round_trip() {
    let spn = SpnValue::Float(98.6);
    assert_eq!(spn_to_twin_value(spn), Some(TwinValue::Float(98.6)));
}

#[test]
fn test_twin_value_boolean_round_trip() {
    let spn = SpnValue::Boolean(true);
    assert_eq!(spn_to_twin_value(spn), Some(TwinValue::Boolean(true)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Config serialization / deserialization
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_config_roundtrip_json() {
    let cfg = BridgeConfig {
        poll_interval_ms: 50,
        mappings: vec![RegisterMapping {
            pgn: 61444,
            spn: 190,
            twin_property: "engine.rpm".to_string(),
            direction: MappingDirection::Bidirectional,
        }],
    };
    let json = serde_json::to_string(&cfg).expect("serialize");
    let cfg2: BridgeConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(cfg2.poll_interval_ms, 50);
    assert_eq!(cfg2.mappings[0].pgn, 61444);
    assert_eq!(cfg2.mappings[0].direction, MappingDirection::Bidirectional);
}

#[test]
fn test_config_mapping_direction_write() {
    let toml_str = r#"
poll_interval_ms = 10
[[mapping]]
pgn = 61444
spn = 190
twin_property = "engine.rpm"
direction = "write"
"#;
    let cfg: BridgeConfig = toml::from_str(toml_str).expect("parse");
    assert_eq!(cfg.mappings[0].direction, MappingDirection::Write);
}

#[test]
fn test_config_empty_mappings() {
    let toml_str = "poll_interval_ms = 100\n";
    let cfg: BridgeConfig = toml::from_str(toml_str).expect("parse empty mappings");
    assert!(cfg.mappings.is_empty());
}
