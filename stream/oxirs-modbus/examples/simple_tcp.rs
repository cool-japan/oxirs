//! Simple Modbus TCP client example
//!
//! This example demonstrates basic Modbus TCP communication
//! with a PLC or sensor.
//!
//! # Usage
//!
//! First, start a Modbus TCP server (simulator):
//! ```bash
//! python stream/oxirs-modbus/test-data/test_modbus_server.py
//! ```
//!
//! Then run this example:
//! ```bash
//! cargo run --example simple_tcp
//! ```

use oxirs_modbus::{ModbusResult, ModbusTcpClient};

#[tokio::main]
async fn main() -> ModbusResult<()> {
    // Connect to Modbus TCP server
    println!("Connecting to Modbus TCP server at 127.0.0.1:502...");

    let mut client = ModbusTcpClient::connect("127.0.0.1:502", 1).await?;

    println!("✓ Connected successfully");

    // Read holding registers (function code 0x03)
    println!("\nReading registers 0-9 (10 registers)...");
    let registers = client.read_holding_registers(0, 10).await?;

    println!("✓ Read {} registers:", registers.len());
    for (i, value) in registers.iter().enumerate() {
        println!("  Register {}: {}", i, value);
    }

    // Read input registers (function code 0x04)
    println!("\nReading input registers 0-4 (5 registers)...");
    let input_regs = client.read_input_registers(0, 5).await?;

    println!("✓ Read {} input registers:", input_regs.len());
    for (i, value) in input_regs.iter().enumerate() {
        println!("  Input Register {}: {}", i, value);
    }

    // Write single register (function code 0x06)
    println!("\nWriting value 100 to register 0...");
    client.write_single_register(0, 100).await?;
    println!("✓ Write successful");

    // Read back to verify
    println!("\nReading register 0 to verify write...");
    let verify = client.read_holding_registers(0, 1).await?;
    println!("✓ Register 0 = {} (should be 100)", verify[0]);

    println!("\n✅ Example completed successfully!");

    Ok(())
}
