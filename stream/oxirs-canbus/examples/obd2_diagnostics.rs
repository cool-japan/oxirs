//! OBD-II Diagnostics Example
//!
//! Demonstrates decoding OBD-II (On-Board Diagnostics) data from
//! passenger vehicles using CAN bus.
//!
//! # OBD-II Protocol Overview
//!
//! - **Request CAN ID**: 0x7DF (broadcast) or 0x7E0-0x7E7 (ECU-specific)
//! - **Response CAN ID**: 0x7E8-0x7EF (ECU responses)
//! - **Format**: Mode + PID + Data
//!
//! # Common OBD-II Modes
//!
//! - Mode 01: Current data (real-time sensor values)
//! - Mode 02: Freeze frame data
//! - Mode 03: Diagnostic Trouble Codes (DTCs)
//! - Mode 04: Clear DTCs
//! - Mode 09: Vehicle Information (VIN, calibration IDs)
//!
//! # Usage
//!
//! ```bash
//! cargo run --example obd2_diagnostics
//! ```

use oxirs_canbus::{CanFrame, CanId, CanbusResult};
use std::collections::HashMap;

/// OBD-II PID definitions for Mode 01
#[derive(Debug, Clone)]
struct Obd2Pid {
    #[allow(dead_code)]
    code: u8,
    name: &'static str,
    unit: &'static str,
    decode: fn(&[u8]) -> f64,
}

fn main() -> CanbusResult<()> {
    println!("=== OxiRS OBD-II Diagnostics Example ===\n");

    // Display OBD-II protocol overview
    print_obd2_overview();

    // Create PID decoder registry
    let pids = create_pid_registry();

    // Simulate OBD-II responses
    let responses = create_simulated_responses()?;

    println!("Decoding OBD-II Responses:");
    println!("--------------------------\n");

    for (pid_name, frame) in responses {
        decode_obd2_response(&frame, &pids, pid_name)?;
    }

    // Demonstrate DTC decoding
    demo_dtc_decoding()?;

    // Demonstrate VIN decoding
    demo_vin_decoding()?;

    println!("\n=== OBD-II Demo Complete ===");

    Ok(())
}

fn print_obd2_overview() {
    println!("OBD-II Protocol Overview:");
    println!("-------------------------");
    println!("  Request CAN IDs:");
    println!("    0x7DF: Broadcast request to all ECUs");
    println!("    0x7E0-0x7E7: Direct request to specific ECU");
    println!();
    println!("  Response CAN IDs:");
    println!("    0x7E8: Engine Control Module (ECM)");
    println!("    0x7E9: Transmission Control Module (TCM)");
    println!("    0x7EA-0x7EF: Other ECUs");
    println!();
    println!("  Message Format (ISO 15765-4 CAN):");
    println!("    Byte 0: Number of data bytes");
    println!("    Byte 1: Mode (+ 0x40 for response)");
    println!("    Byte 2: PID");
    println!("    Bytes 3-7: Data");
    println!();
    println!("Common Mode 01 PIDs:");
    println!("--------------------");
    println!("  PID 0x04: Calculated Engine Load (%)");
    println!("  PID 0x05: Engine Coolant Temperature (degC)");
    println!("  PID 0x0C: Engine RPM");
    println!("  PID 0x0D: Vehicle Speed (km/h)");
    println!("  PID 0x0F: Intake Air Temperature (degC)");
    println!("  PID 0x10: MAF Air Flow Rate (g/s)");
    println!("  PID 0x11: Throttle Position (%)");
    println!("  PID 0x2F: Fuel Tank Level (%)");
    println!("  PID 0x5C: Engine Oil Temperature (degC)");
    println!();
}

fn create_pid_registry() -> HashMap<u8, Obd2Pid> {
    let mut pids = HashMap::new();

    // PID 0x04: Calculated Engine Load
    pids.insert(
        0x04,
        Obd2Pid {
            code: 0x04,
            name: "Calculated Engine Load",
            unit: "%",
            decode: |data| data[0] as f64 * 100.0 / 255.0,
        },
    );

    // PID 0x05: Engine Coolant Temperature
    pids.insert(
        0x05,
        Obd2Pid {
            code: 0x05,
            name: "Engine Coolant Temperature",
            unit: "degC",
            decode: |data| data[0] as f64 - 40.0,
        },
    );

    // PID 0x0C: Engine RPM
    pids.insert(
        0x0C,
        Obd2Pid {
            code: 0x0C,
            name: "Engine RPM",
            unit: "rpm",
            decode: |data| ((data[0] as u16 * 256 + data[1] as u16) as f64) / 4.0,
        },
    );

    // PID 0x0D: Vehicle Speed
    pids.insert(
        0x0D,
        Obd2Pid {
            code: 0x0D,
            name: "Vehicle Speed",
            unit: "km/h",
            decode: |data| data[0] as f64,
        },
    );

    // PID 0x0F: Intake Air Temperature
    pids.insert(
        0x0F,
        Obd2Pid {
            code: 0x0F,
            name: "Intake Air Temperature",
            unit: "degC",
            decode: |data| data[0] as f64 - 40.0,
        },
    );

    // PID 0x10: MAF Air Flow Rate
    pids.insert(
        0x10,
        Obd2Pid {
            code: 0x10,
            name: "MAF Air Flow Rate",
            unit: "g/s",
            decode: |data| ((data[0] as u16 * 256 + data[1] as u16) as f64) / 100.0,
        },
    );

    // PID 0x11: Throttle Position
    pids.insert(
        0x11,
        Obd2Pid {
            code: 0x11,
            name: "Throttle Position",
            unit: "%",
            decode: |data| data[0] as f64 * 100.0 / 255.0,
        },
    );

    // PID 0x2F: Fuel Tank Level
    pids.insert(
        0x2F,
        Obd2Pid {
            code: 0x2F,
            name: "Fuel Tank Level",
            unit: "%",
            decode: |data| data[0] as f64 * 100.0 / 255.0,
        },
    );

    // PID 0x5C: Engine Oil Temperature
    pids.insert(
        0x5C,
        Obd2Pid {
            code: 0x5C,
            name: "Engine Oil Temperature",
            unit: "degC",
            decode: |data| data[0] as f64 - 40.0,
        },
    );

    pids
}

fn create_simulated_responses() -> CanbusResult<Vec<(&'static str, CanFrame)>> {
    let mut responses = Vec::new();

    // Response from ECM (0x7E8)
    // Format: [length, mode+0x40, PID, data...]

    // Engine Load: 75%
    let engine_load = CanFrame::new(
        CanId::standard(0x7E8)?,
        vec![0x03, 0x41, 0x04, 0xBF, 0x00, 0x00, 0x00, 0x00], // 0xBF = 191 -> 75%
    )?;
    responses.push(("Engine Load", engine_load));

    // Coolant Temperature: 90 degC
    let coolant_temp = CanFrame::new(
        CanId::standard(0x7E8)?,
        vec![0x03, 0x41, 0x05, 0x82, 0x00, 0x00, 0x00, 0x00], // 0x82 = 130 -> 90 degC
    )?;
    responses.push(("Coolant Temperature", coolant_temp));

    // Engine RPM: 2500 rpm
    let engine_rpm = CanFrame::new(
        CanId::standard(0x7E8)?,
        vec![0x04, 0x41, 0x0C, 0x27, 0x10, 0x00, 0x00, 0x00], // 0x2710 = 10000 -> 2500 rpm
    )?;
    responses.push(("Engine RPM", engine_rpm));

    // Vehicle Speed: 85 km/h
    let vehicle_speed = CanFrame::new(
        CanId::standard(0x7E8)?,
        vec![0x03, 0x41, 0x0D, 0x55, 0x00, 0x00, 0x00, 0x00], // 0x55 = 85 km/h
    )?;
    responses.push(("Vehicle Speed", vehicle_speed));

    // Intake Air Temperature: 25 degC
    let intake_temp = CanFrame::new(
        CanId::standard(0x7E8)?,
        vec![0x03, 0x41, 0x0F, 0x41, 0x00, 0x00, 0x00, 0x00], // 0x41 = 65 -> 25 degC
    )?;
    responses.push(("Intake Air Temperature", intake_temp));

    // MAF Air Flow Rate: 15.5 g/s
    let maf_rate = CanFrame::new(
        CanId::standard(0x7E8)?,
        vec![0x04, 0x41, 0x10, 0x06, 0x0E, 0x00, 0x00, 0x00], // 0x060E = 1550 -> 15.5 g/s
    )?;
    responses.push(("MAF Air Flow Rate", maf_rate));

    // Throttle Position: 30%
    let throttle = CanFrame::new(
        CanId::standard(0x7E8)?,
        vec![0x03, 0x41, 0x11, 0x4D, 0x00, 0x00, 0x00, 0x00], // 0x4D = 77 -> 30%
    )?;
    responses.push(("Throttle Position", throttle));

    // Fuel Tank Level: 65%
    let fuel_level = CanFrame::new(
        CanId::standard(0x7E8)?,
        vec![0x03, 0x41, 0x2F, 0xA6, 0x00, 0x00, 0x00, 0x00], // 0xA6 = 166 -> 65%
    )?;
    responses.push(("Fuel Tank Level", fuel_level));

    Ok(responses)
}

fn decode_obd2_response(
    frame: &CanFrame,
    pids: &HashMap<u8, Obd2Pid>,
    expected_name: &str,
) -> CanbusResult<()> {
    println!("--- {} ---", expected_name);
    println!("  CAN ID: 0x{:03X}", frame.id.as_raw());
    println!("  Raw Data: {:02X?}", frame.data);

    // Parse OBD-II response
    let length = frame.data[0] as usize;
    let mode = frame.data[1];
    let pid = frame.data[2];

    // Verify mode is a response (0x41 for Mode 01 response)
    if mode != 0x41 {
        println!("  Error: Unexpected mode 0x{:02X}", mode);
        return Ok(());
    }

    // Get PID definition
    if let Some(pid_def) = pids.get(&pid) {
        let data = &frame.data[3..length + 1];
        let value = (pid_def.decode)(data);

        println!("  Mode: 0x{:02X} (Current Data Response)", mode);
        println!("  PID: 0x{:02X} ({})", pid, pid_def.name);
        println!("  Value: {:.2} {}", value, pid_def.unit);
    } else {
        println!("  Mode: 0x{:02X}", mode);
        println!("  PID: 0x{:02X} (Unknown)", pid);
    }

    println!();
    Ok(())
}

fn demo_dtc_decoding() -> CanbusResult<()> {
    println!("Diagnostic Trouble Code (DTC) Decoding:");
    println!("---------------------------------------");

    // DTC format: 2 bytes per code
    // Byte 1: First nibble = type, Second nibble = first digit
    // Byte 2: Remaining digits
    //
    // Type codes:
    //   0x00-0x03: P0xxx (Powertrain)
    //   0x40-0x43: P1xxx (Manufacturer-specific powertrain)
    //   0x80-0x83: P2xxx (Powertrain)
    //   0xC0-0xC3: P3xxx (Powertrain)
    //   0x04-0x07: C0xxx (Chassis)
    //   etc.

    // Simulate DTC response (Mode 03)
    let dtc_data = vec![
        0x06, 0x43, // 6 bytes, Mode 03 response
        0x01, 0x33, // P0133 - O2 Sensor Circuit Slow Response
        0x03, 0x00, // P0300 - Random/Multiple Cylinder Misfire Detected
        0x04, 0x20, // P0420 - Catalyst System Efficiency Below Threshold
        0x00, 0x00,
    ];

    let dtc_frame = CanFrame::new(CanId::standard(0x7E8)?, dtc_data)?;

    println!("  CAN ID: 0x{:03X}", dtc_frame.id.as_raw());
    println!("  Raw Data: {:02X?}", dtc_frame.data);
    println!("  Number of DTCs: 3");
    println!();
    println!("  Decoded DTCs:");

    // Decode P0133
    println!("    P0133 - O2 Sensor Circuit Slow Response (Bank 1, Sensor 1)");
    println!("      Description: Oxygen sensor response time is too slow");
    println!("      Common causes: Failed O2 sensor, exhaust leak, wiring issue");

    // Decode P0300
    println!("    P0300 - Random/Multiple Cylinder Misfire Detected");
    println!("      Description: Engine misfires occurring in multiple cylinders");
    println!("      Common causes: Spark plugs, ignition coils, fuel injectors");

    // Decode P0420
    println!("    P0420 - Catalyst System Efficiency Below Threshold (Bank 1)");
    println!("      Description: Catalytic converter efficiency is degraded");
    println!("      Common causes: Failed catalytic converter, exhaust leak");

    println!();
    Ok(())
}

fn demo_vin_decoding() -> CanbusResult<()> {
    println!("Vehicle Identification Number (VIN) Decoding:");
    println!("---------------------------------------------");
    println!("  Mode: 09 (Request Vehicle Information)");
    println!("  PID: 02 (Vehicle Identification Number)");
    println!();

    // VIN is 17 characters, sent via multi-frame ISO-TP
    // For simplicity, we show the decoded result
    let vin = "1HGCM82633A123456";

    println!("  VIN: {}", vin);
    println!();
    println!("  VIN Breakdown:");
    println!(
        "    World Manufacturer ID (WMI): {} = Honda Motor Company",
        &vin[0..3]
    );
    println!(
        "    Vehicle Descriptor (VDS): {} = Accord EX Sedan",
        &vin[3..9]
    );
    println!("    Vehicle Identifier (VIS): {}", &vin[9..17]);
    println!("      Model Year: {} = 2003", &vin[9..10]);
    println!("      Assembly Plant: {} = Marysville, Ohio", &vin[10..11]);
    println!("      Sequential Number: {}", &vin[11..17]);

    println!();
    Ok(())
}
