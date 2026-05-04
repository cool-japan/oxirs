//! Example: J1939 → DTDL bridge with mock components.
//!
//! Demonstrates how to wire a [`MockJ1939Source`] and a [`MockDtdlSink`] together
//! via the [`J1939DtdlBridge`] and verify that SAE J1939 ET1 coolant temperature
//! data flows through to the DTDL property store.
//!
//! Run with:
//! ```bash
//! cargo run --example digital_twin_bridge
//! ```

use oxirs_canbus::digital_twin::client::J1939Frame;
use oxirs_canbus::digital_twin::{
    BridgeConfig, J1939DtdlBridge, MappingDirection, MockDtdlSink, MockJ1939Source, RegisterMapping,
};
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Configure the bridge ────────────────────────────────────────────────
    // Map J1939 ET1 (PGN 65262, which contains SPN 110 — Engine Coolant
    // Temperature) to the DTDL property "engine.coolant_temp_c".
    let config = BridgeConfig {
        poll_interval_ms: 100,
        mappings: vec![
            RegisterMapping {
                pgn: 65262, // ET1 — Engine Temperatures
                spn: 0,     // Simplified: treat whole PGN as one SPN
                twin_property: "engine.coolant_temp_c".to_string(),
                direction: MappingDirection::Read,
            },
            RegisterMapping {
                pgn: 65265, // CCVS — Cruise Control / Vehicle Speed
                spn: 0,
                twin_property: "vehicle.speed_kmh".to_string(),
                direction: MappingDirection::Read,
            },
            RegisterMapping {
                pgn: 61444, // EEC1 — Electronic Engine Controller 1
                spn: 0,
                twin_property: "engine.rpm".to_string(),
                direction: MappingDirection::Bidirectional,
            },
        ],
    };

    // ── Build mock J1939 source ─────────────────────────────────────────────
    // Each frame carries one byte of telemetry in data[0]; bytes 1-7 are zero.
    let frames = vec![
        // Coolant temp: 75 °C (raw byte 75 for this example)
        J1939Frame::new(65262, 0x00, [75, 0, 0, 0, 0, 0, 0, 0]),
        // Vehicle speed: 80 km/h
        J1939Frame::new(65265, 0x00, [80, 0, 0, 0, 0, 0, 0, 0]),
        // Engine RPM: 120 (scaled; raw byte in our simplified model)
        J1939Frame::new(61444, 0x00, [120, 0, 0, 0, 0, 0, 0, 0]),
        // NA indicator — should be silently dropped
        J1939Frame::new(65262, 0x00, [0xFE, 0, 0, 0, 0, 0, 0, 0]),
        // Unmapped PGN — should be silently ignored
        J1939Frame::new(99999, 0x00, [42, 0, 0, 0, 0, 0, 0, 0]),
    ];
    let source = MockJ1939Source::new(frames);

    // ── Build mock DTDL sink ────────────────────────────────────────────────
    let sink = MockDtdlSink::new();

    // ── Run the bridge ──────────────────────────────────────────────────────
    // The bridge processes all frames from the mock source and then stops
    // when the source is exhausted.
    let cancel = CancellationToken::new();
    let mut bridge = J1939DtdlBridge::new(config, source, sink.clone(), cancel);
    bridge.run().await?;

    // ── Print results ───────────────────────────────────────────────────────
    println!("Bridge finished. Twin properties written:");

    if let Some(temp) = sink.get("engine.coolant_temp_c") {
        println!("  engine.coolant_temp_c = {temp}");
    }
    if let Some(speed) = sink.get("vehicle.speed_kmh") {
        println!("  vehicle.speed_kmh     = {speed}");
    }
    if let Some(rpm) = sink.get("engine.rpm") {
        println!("  engine.rpm            = {rpm}");
    }

    println!("Total properties in sink: {}", sink.len());
    println!("Bridge ran successfully");

    Ok(())
}
