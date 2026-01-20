//! CAN frame parsing demonstration
//!
//! This example shows CAN frame creation, signal extraction,
//! and J1939 PGN decoding.

use oxirs_canbus::{CanFrame, CanId, CanbusResult};

fn main() -> CanbusResult<()> {
    println!("=== OxiRS CANbus Frame Demo ===\n");

    // Demo 1: Standard CAN ID
    demo_standard_can()?;

    // Demo 2: Extended CAN ID
    demo_extended_can()?;

    // Demo 3: J1939 PGN extraction
    demo_j1939_pgn()?;

    // Demo 4: Signal extraction
    demo_signal_extraction()?;

    println!("\n✅ CAN frame demo completed!");

    Ok(())
}

fn demo_standard_can() -> CanbusResult<()> {
    println!("Demo 1: Standard CAN Frame (11-bit ID)");
    println!("---------------------------------------");

    let id = CanId::standard(0x123)?;
    let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let frame = CanFrame::new(id, data)?;

    println!("  CAN ID: 0x{:03X} (11-bit standard)", id.as_raw());
    println!("  Data: {:02X?}", frame.data);
    println!("  DLC: {} bytes", frame.dlc());
    println!("  ✓ Standard CAN frame created\n");

    Ok(())
}

fn demo_extended_can() -> CanbusResult<()> {
    println!("Demo 2: Extended CAN Frame (29-bit ID)");
    println!("---------------------------------------");

    let id = CanId::extended(0x18FEF100)?;
    let data = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
    let frame = CanFrame::new(id, data)?;

    println!("  CAN ID: 0x{:08X} (29-bit extended)", id.as_raw());
    println!("  Data: {:02X?}", frame.data);
    println!("  DLC: {} bytes", frame.dlc());
    println!("  ✓ Extended CAN frame created\n");

    Ok(())
}

fn demo_j1939_pgn() -> CanbusResult<()> {
    println!("Demo 3: J1939 PGN Extraction");
    println!("----------------------------");

    // PGN 61444 (0xF004): Electronic Engine Controller 1
    // Example CAN ID: 0x0CF00400
    // - Priority: 3
    // - PGN: 61444 (engine speed, torque)
    // - Source Address: 0 (engine ECU)

    let id = CanId::extended(0x0CF00400)?;

    let pgn = id.extract_j1939_pgn().unwrap();
    let priority = id.extract_j1939_priority().unwrap();
    let source_addr = id.extract_j1939_source_address().unwrap();

    println!("  CAN ID: 0x{:08X}", id.as_raw());
    println!(
        "  PGN: {} (0x{:04X}) - Electronic Engine Controller 1",
        pgn, pgn
    );
    println!("  Priority: {}", priority);
    println!("  Source Address: {} (Engine ECU)", source_addr);
    println!("  ✓ J1939 frame decoded\n");

    Ok(())
}

fn demo_signal_extraction() -> CanbusResult<()> {
    println!("Demo 4: Signal Extraction from CAN Frame");
    println!("-----------------------------------------");

    let id = CanId::standard(0x100)?;

    // Simulate CAN frame with engine data:
    // Bytes 0-1: Engine speed (little-endian, 0.125 rpm per bit)
    // Bytes 2-3: Engine torque (little-endian, 1 Nm per bit)
    let data = vec![
        0xE8, 0x03, // 1000 in little-endian = 1000 * 0.125 = 125 rpm
        0x96, 0x00, // 150 in little-endian = 150 Nm
    ];

    let frame = CanFrame::new(id, data)?;

    // Extract engine speed (bytes 0-1, little-endian)
    let speed_raw = frame.extract_value_le(0, 2).unwrap();
    let speed_rpm = speed_raw as f64 * 0.125;

    // Extract engine torque (bytes 2-3, little-endian)
    let torque_raw = frame.extract_value_le(2, 2).unwrap();
    let torque_nm = torque_raw as f64;

    println!("  CAN ID: 0x{:03X}", id.as_raw());
    println!("  Raw data: {:02X?}", frame.data);
    println!("  Engine Speed: {} rpm (raw: {})", speed_rpm, speed_raw);
    println!("  Engine Torque: {} Nm (raw: {})", torque_nm, torque_raw);
    println!("  ✓ Signals extracted and decoded\n");

    Ok(())
}
