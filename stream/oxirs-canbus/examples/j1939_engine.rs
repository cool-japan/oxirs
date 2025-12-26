//! J1939 Heavy Vehicle Engine Data Decoder
//!
//! Demonstrates decoding J1939 Parameter Group Numbers (PGNs) from
//! heavy vehicle CAN bus data (trucks, buses, agricultural equipment).
//!
//! # J1939 CAN ID Structure (29-bit Extended)
//!
//! ```text
//! | Priority | Reserved | Data Page | PDU Format | PDU Specific | Source Addr |
//! |  3 bits  |  1 bit   |   1 bit   |   8 bits   |    8 bits    |   8 bits    |
//! ```
//!
//! # Usage
//!
//! ```bash
//! cargo run --example j1939_engine
//! ```

use oxirs_canbus::{
    CanFrame, CanId, CanbusResult, J1939Message, J1939Processor, PgnRegistry, PGN_AMB, PGN_CCVS,
    PGN_EEC1, PGN_EEC2, PGN_EFLP1, PGN_ET1, PGN_LFE, PGN_VEP1,
};

fn main() -> CanbusResult<()> {
    println!("=== OxiRS J1939 Heavy Vehicle Engine Data ===\n");

    // Create J1939 processor for multi-packet handling
    let mut processor = J1939Processor::new();

    // Create PGN registry with standard decoders
    let registry = PgnRegistry::with_standard_decoders();

    // List supported PGNs
    println!("Supported J1939 PGNs:");
    println!("---------------------");
    println!(
        "  PGN {} (0x{:04X}): Electronic Engine Controller 1 (EEC1)",
        PGN_EEC1, PGN_EEC1
    );
    println!(
        "  PGN {} (0x{:04X}): Electronic Engine Controller 2 (EEC2)",
        PGN_EEC2, PGN_EEC2
    );
    println!(
        "  PGN {} (0x{:04X}): Cruise Control/Vehicle Speed (CCVS)",
        PGN_CCVS, PGN_CCVS
    );
    println!(
        "  PGN {} (0x{:04X}): Engine Temperature 1 (ET1)",
        PGN_ET1, PGN_ET1
    );
    println!(
        "  PGN {} (0x{:04X}): Engine Fluid Level/Pressure 1 (EFL/P1)",
        PGN_EFLP1, PGN_EFLP1
    );
    println!("  PGN {} (0x{:04X}): Fuel Economy (LFE)", PGN_LFE, PGN_LFE);
    println!(
        "  PGN {} (0x{:04X}): Ambient Conditions (AMB)",
        PGN_AMB, PGN_AMB
    );
    println!(
        "  PGN {} (0x{:04X}): Vehicle Electrical Power 1 (VEP1)",
        PGN_VEP1, PGN_VEP1
    );
    println!();

    // Simulate J1939 frames from a heavy vehicle
    let frames = create_simulated_j1939_frames()?;

    println!("Processing J1939 Frames:");
    println!("------------------------\n");

    for (name, frame) in frames {
        println!("--- {} ---", name);
        println!("CAN ID: 0x{:08X}", frame.id.as_raw());

        // Process through J1939 layer
        if let Some(message) = processor.process(&frame) {
            decode_j1939_message(&message, &registry);
        } else {
            // For single-packet messages, create J1939Message directly
            if let Some(message) = J1939Message::from_frame(&frame) {
                decode_j1939_message(&message, &registry);
            }
        }
        println!();
    }

    // Demonstrate PGN breakdown
    demo_pgn_breakdown()?;

    println!("=== J1939 Demo Complete ===");

    Ok(())
}

fn create_simulated_j1939_frames() -> CanbusResult<Vec<(&'static str, CanFrame)>> {
    let mut frames = Vec::new();

    // EEC1 (PGN 61444 = 0xF004): Engine Speed, Torque
    // CAN ID: 0x0CF00400 (Priority=3, PGN=61444, SA=0)
    // Engine Speed @ offset 3-4: 2000 rpm = 16000 (0x3E80)
    // Engine Torque @ offset 2: 75% = 200 (125+75)
    let eec1_data = vec![
        0x00, // Torque Mode
        0x00, // Driver's Demand
        0xC8, // Actual Torque (200 = 75%)
        0x80, 0x3E, // Engine Speed (16000 = 2000 rpm)
        0x00, // Source Address
        0xFF, 0xFF,
    ];
    let eec1 = CanFrame::new(CanId::extended(0x0CF00400)?, eec1_data)?;
    frames.push(("EEC1 - Electronic Engine Controller 1", eec1));

    // EEC2 (PGN 61443 = 0xF003): Accelerator Position
    // CAN ID: 0x0CF00300 (Priority=3, PGN=61443, SA=0)
    let eec2_data = vec![
        0x66, // Accelerator Pedal Position 1 (40%)
        0x99, // Accelerator Pedal Position 2 (60%)
        0xFF, // Road Speed Limit Status
        0x33, // Actual Engine Percent Torque High Resolution
        0xFF, 0xFF, 0xFF, 0xFF,
    ];
    let eec2 = CanFrame::new(CanId::extended(0x0CF00300)?, eec2_data)?;
    frames.push(("EEC2 - Electronic Engine Controller 2", eec2));

    // CCVS (PGN 65265 = 0xFEF1): Vehicle Speed
    // CAN ID: 0x18FEF100 (Priority=6, PGN=65265, SA=0)
    let ccvs_data = vec![
        0xFF, // Two Speed Axle Switch
        0xFF, // Parking Brake Switch
        0x20, 0x4E, // Wheel-Based Vehicle Speed (85 km/h = 21760)
        0xFF, // Cruise Control States
        0xFF, // Service Brake Switch
        0xFF, // Cruise Control Switches
        0xFF,
    ];
    let ccvs = CanFrame::new(CanId::extended(0x18FEF100)?, ccvs_data)?;
    frames.push(("CCVS - Cruise Control/Vehicle Speed", ccvs));

    // ET1 (PGN 65262 = 0xFEEE): Engine Temperature
    // CAN ID: 0x18FEEE00 (Priority=6, PGN=65262, SA=0)
    let et1_data = vec![
        0x82, // Engine Coolant Temperature (90°C = 130)
        0x5A, // Fuel Temperature (50°C = 90)
        0x64, 0x00, // Engine Oil Temperature (60°C = 100)
        0xFF, 0xFF, // Turbo Oil Temperature
        0xFF, // Engine Intercooler Temperature
        0xFF,
    ];
    let et1 = CanFrame::new(CanId::extended(0x18FEEE00)?, et1_data)?;
    frames.push(("ET1 - Engine Temperature 1", et1));

    // EFLP1 (PGN 65263 = 0xFEEF): Fluid Levels/Pressures
    // CAN ID: 0x18FEEF00 (Priority=6, PGN=65263, SA=0)
    let eflp1_data = vec![
        0x50, // Fuel Delivery Pressure (32 kPa = 80)
        0xFF, // Extended Crankcase Blow-by Pressure
        0x8C, // Engine Oil Level (70% = 140)
        0xA0, // Engine Oil Pressure (80 kPa = 160)
        0xFF, 0xFF, // Crankcase Pressure
        0x64, // Coolant Level (50% = 100)
        0xFF,
    ];
    let eflp1 = CanFrame::new(CanId::extended(0x18FEEF00)?, eflp1_data)?;
    frames.push(("EFLP1 - Engine Fluid Level/Pressure 1", eflp1));

    // LFE (PGN 65266 = 0xFEF2): Fuel Economy
    // CAN ID: 0x18FEF200 (Priority=6, PGN=65266, SA=0)
    let lfe_data = vec![
        0x60, 0x00, // Engine Fuel Rate (4.8 L/h = 96)
        0xFA, 0x00, // Instantaneous Fuel Economy (2.5 km/L = 250)
        0xFF, 0xFF, // Average Fuel Economy
        0xFF, // Throttle Position
        0xFF,
    ];
    let lfe = CanFrame::new(CanId::extended(0x18FEF200)?, lfe_data)?;
    frames.push(("LFE - Fuel Economy", lfe));

    // AMB (PGN 65269 = 0xFEF5): Ambient Conditions
    // CAN ID: 0x18FEF500 (Priority=6, PGN=65269, SA=0)
    let amb_data = vec![
        0x50, // Barometric Pressure (80 kPa = 80)
        0x00, 0x00, // Cab Interior Temperature
        0x96, 0x00, // Ambient Air Temperature (25°C)
        0xFF, // Air Inlet Temperature
        0xFF, 0xFF,
    ];
    let amb = CanFrame::new(CanId::extended(0x18FEF500)?, amb_data)?;
    frames.push(("AMB - Ambient Conditions", amb));

    // VEP1 (PGN 65271 = 0xFEF7): Vehicle Electrical Power
    // CAN ID: 0x18FEF700 (Priority=6, PGN=65271, SA=0)
    let vep1_data = vec![
        0xFF, // Net Battery Current
        0xFF, // Alternator Current
        0xFF, 0xFF, // Alternator Potential
        0x58, 0x02, // Electrical Potential (Battery) (28.4V = 568)
        0xFF, 0xFF,
    ];
    let vep1 = CanFrame::new(CanId::extended(0x18FEF700)?, vep1_data)?;
    frames.push(("VEP1 - Vehicle Electrical Power 1", vep1));

    Ok(frames)
}

fn decode_j1939_message(message: &J1939Message, registry: &PgnRegistry) {
    let pgn = message.header.pgn.value();
    println!("  PGN: {} (0x{:04X})", pgn, pgn);
    println!("  Priority: {}", message.header.priority);
    println!("  Source: {}", message.header.source_address);
    println!("  Data: {:02X?}", message.data);

    if let Some(decoded) = registry.decode(message) {
        println!("  Decoded Signals:");
        for signal in &decoded.signals {
            let status = if !signal.valid { " [INVALID]" } else { "" };
            println!(
                "    {}: {:.2} {}{}",
                signal.name, signal.value, signal.unit, status
            );
        }
    } else {
        println!("  (No decoder registered for this PGN)");
    }
}

fn demo_pgn_breakdown() -> CanbusResult<()> {
    println!("\nJ1939 CAN ID Structure Breakdown:");
    println!("---------------------------------");

    // Example: EEC1 CAN ID = 0x0CF00400
    let can_id = CanId::extended(0x0CF00400)?;

    let priority = can_id.extract_j1939_priority().unwrap();
    let pgn = can_id.extract_j1939_pgn().unwrap();
    let source = can_id.extract_j1939_source_address().unwrap();

    println!("  CAN ID: 0x{:08X}", can_id.as_raw());
    println!("  Binary: {:029b}", can_id.as_raw());
    println!();
    println!("  Priority:      {} (bits 26-28)", priority);
    println!("  Reserved:      0 (bit 25)");
    println!("  Data Page:     0 (bit 24)");
    println!(
        "  PDU Format:    0x{:02X} ({}) (bits 16-23)",
        (pgn >> 8) & 0xFF,
        (pgn >> 8) & 0xFF
    );
    println!(
        "  PDU Specific:  0x{:02X} ({}) (bits 8-15)",
        pgn & 0xFF,
        pgn & 0xFF
    );
    println!("  Source Addr:   0x{:02X} ({}) (bits 0-7)", source, source);
    println!();
    println!("  Resolved PGN:  {} (0x{:04X})", pgn, pgn);
    println!("  PGN Name:      Electronic Engine Controller 1 (EEC1)");
    println!();

    // PDU format explanation
    println!("PDU Format Types:");
    println!("  PDU1 (PF < 240): Point-to-point, PGN includes destination");
    println!("  PDU2 (PF >= 240): Broadcast, PGN is PF + PS << 8");
    println!();

    // Common PGN ranges
    println!("Common PGN Ranges:");
    println!("  0-8191:     Proprietary A");
    println!("  8192-60159: Standard J1939 Parameter Groups");
    println!("  61440-65279: Proprietary B");
    println!("  65280-65535: Address Management");
    println!();

    Ok(())
}
