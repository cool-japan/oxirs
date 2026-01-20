//! Simple Modbus RTU client example
//!
//! This example demonstrates basic Modbus RTU communication
//! over a serial port (RS-232/RS-485).
//!
//! # Hardware Setup
//!
//! Connect a Modbus RTU device (PLC, sensor, etc.) to a serial port.
//! Common settings: 9600 baud, 8N1 (8 data bits, no parity, 1 stop bit)
//!
//! # Virtual Serial Port Testing (Linux)
//!
//! For testing without hardware, use socat to create virtual serial ports:
//! ```bash
//! # Create a virtual serial port pair
//! socat -d -d pty,raw,echo=0 pty,raw,echo=0
//! # Note the created ports (e.g., /dev/pts/3 and /dev/pts/4)
//! # Use one for the client, one for a simulator
//! ```
//!
//! # Usage
//!
//! ```bash
//! # With a real serial device:
//! cargo run --example simple_rtu --features rtu -- /dev/ttyUSB0
//!
//! # With a different baud rate:
//! cargo run --example simple_rtu --features rtu -- /dev/ttyUSB0 19200
//!
//! # On Windows:
//! cargo run --example simple_rtu --features rtu -- COM3
//! ```

#[cfg(feature = "rtu")]
use oxirs_modbus::{protocol::rtu::ModbusRtuClient, ModbusResult};
use std::env;

#[cfg(feature = "rtu")]
#[tokio::main]
async fn main() -> ModbusResult<()> {
    let args: Vec<String> = env::args().collect();

    // Parse command-line arguments
    let port = args.get(1).map(|s| s.as_str()).unwrap_or("/dev/ttyUSB0");
    let baud_rate: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(9600);
    let unit_id: u8 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    println!("Modbus RTU Client Example");
    println!("========================");
    println!("Port:      {}", port);
    println!("Baud rate: {}", baud_rate);
    println!("Unit ID:   {}", unit_id);
    println!();

    // Open serial connection
    println!("Opening serial port...");
    let mut client = ModbusRtuClient::open(port, baud_rate, unit_id)?;

    println!("Serial port opened successfully");

    // Read holding registers (function code 0x03)
    println!("\nReading holding registers 0-9 (10 registers)...");
    match client.read_holding_registers(0, 10).await {
        Ok(registers) => {
            println!("Read {} registers:", registers.len());
            for (i, value) in registers.iter().enumerate() {
                println!("  Register {}: {} (0x{:04X})", i, value, value);
            }
        }
        Err(e) => {
            println!("Failed to read holding registers: {:?}", e);
        }
    }

    // Read input registers (function code 0x04)
    println!("\nReading input registers 0-4 (5 registers)...");
    match client.read_input_registers(0, 5).await {
        Ok(input_regs) => {
            println!("Read {} input registers:", input_regs.len());
            for (i, value) in input_regs.iter().enumerate() {
                let description = match i {
                    0 => "Temperature",
                    1 => "Humidity",
                    2 => "Pressure",
                    3 => "Voltage",
                    4 => "Current",
                    _ => "Unknown",
                };
                println!("  Input Register {} ({}): {}", i, description, value);
            }
        }
        Err(e) => {
            println!("Failed to read input registers: {:?}", e);
        }
    }

    // Read coils (function code 0x01)
    println!("\nReading coils 0-7 (8 coils)...");
    match client.read_coils(0, 8).await {
        Ok(coils) => {
            println!("Read {} coils:", coils.len());
            for (i, state) in coils.iter().enumerate() {
                let state_str = if *state { "ON " } else { "OFF" };
                println!("  Coil {}: {}", i, state_str);
            }
        }
        Err(e) => {
            println!("Failed to read coils: {:?}", e);
        }
    }

    // Read discrete inputs (function code 0x02)
    println!("\nReading discrete inputs 0-7 (8 inputs)...");
    match client.read_discrete_inputs(0, 8).await {
        Ok(inputs) => {
            println!("Read {} discrete inputs:", inputs.len());
            for (i, state) in inputs.iter().enumerate() {
                let state_str = if *state { "ON " } else { "OFF" };
                println!("  Input {}: {}", i, state_str);
            }
        }
        Err(e) => {
            println!("Failed to read discrete inputs: {:?}", e);
        }
    }

    // Write single register (function code 0x06)
    println!("\nWriting value 1234 to register 10...");
    match client.write_single_register(10, 1234).await {
        Ok(()) => {
            println!("Write successful");
        }
        Err(e) => {
            println!("Failed to write register: {:?}", e);
        }
    }

    println!("\nExample completed!");

    Ok(())
}

#[cfg(not(feature = "rtu"))]
fn main() {
    eprintln!("This example requires the 'rtu' feature.");
    eprintln!("Run with: cargo run --example simple_rtu --features rtu");
}
