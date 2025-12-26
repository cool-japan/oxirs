//! Mock Server Demo
//!
//! This example demonstrates how to use the MockModbusServer for testing
//! Modbus TCP clients without real hardware. Useful for CI/CD pipelines
//! and local development.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example mock_server_demo --features testing
//! ```

#[cfg(not(feature = "testing"))]
fn main() {
    eprintln!("This example requires the 'testing' feature.");
    eprintln!("Run with: cargo run --example mock_server_demo --features testing");
}

#[cfg(feature = "testing")]
use oxirs_modbus::protocol::ModbusTcpClient;
#[cfg(feature = "testing")]
use oxirs_modbus::testing::{MockModbusServer, MockServerData};

#[cfg(feature = "testing")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Mock Modbus Server Demo");
    println!("=======================\n");

    // Example 1: Start server with default test data
    println!("Example 1: Default test data");
    println!("----------------------------");

    let server = MockModbusServer::start().await?;
    println!("Mock server started at: {}", server.address());

    let mut client = ModbusTcpClient::connect(server.address(), 1).await?;

    // Read holding registers
    let registers = client.read_holding_registers(0, 5).await?;
    println!("Holding registers 0-4: {:?}", registers);
    // Should be: [100, 200, 300, 400, 500] (incrementing pattern)

    // Read input registers (simulated sensor data)
    let inputs = client.read_input_registers(0, 5).await?;
    println!("Input registers 0-4: {:?}", inputs);
    // Temperature=225, Humidity=501, Pressure=1013, Voltage=3300, Current=150

    // Interpret sensor values
    println!("\nSensor interpretations:");
    println!("  Temperature: {:.1} C", inputs[0] as f64 / 10.0);
    println!("  Humidity:    {:.1} %", inputs[1] as f64 / 10.0);
    println!("  Pressure:    {} hPa", inputs[2]);
    println!("  Voltage:     {:.1} V", inputs[3] as f64 / 10.0);
    println!("  Current:     {:.1} A", inputs[4] as f64 / 10.0);

    // Read coils
    let coils = client.read_coils(0, 8).await?;
    println!("\nCoils 0-7: {:?}", coils);

    // Read discrete inputs
    let discrete = client.read_discrete_inputs(0, 8).await?;
    println!("Discrete inputs 0-7: {:?}", discrete);

    server.shutdown().await;
    println!("\nServer shutdown.\n");

    // Example 2: Custom test data
    println!("Example 2: Custom test data");
    println!("---------------------------");

    let mut custom_data = MockServerData::new();

    // Custom holding registers (simulated motor controller)
    custom_data.holding_registers.insert(0, 1500); // RPM setpoint
    custom_data.holding_registers.insert(1, 1480); // Actual RPM
    custom_data.holding_registers.insert(2, 50); // Load percentage
    custom_data.holding_registers.insert(3, 75); // Temperature
    custom_data.holding_registers.insert(4, 0); // Status (0=running)

    // Custom input registers (motor feedback)
    custom_data.input_registers.insert(0, 1482); // Measured RPM
    custom_data.input_registers.insert(1, 350); // Motor current (35.0A)
    custom_data.input_registers.insert(2, 4800); // Voltage (480.0V)

    // Custom coils (motor control bits)
    custom_data.coils.insert(0, true); // Enable
    custom_data.coils.insert(1, false); // Direction (0=forward)
    custom_data.coils.insert(2, false); // Brake
    custom_data.coils.insert(3, true); // Ready

    let server = MockModbusServer::start_with_data(custom_data).await?;
    println!("Mock server started at: {}", server.address());

    let mut client = ModbusTcpClient::connect(server.address(), 1).await?;

    // Read motor controller data
    let motor_regs = client.read_holding_registers(0, 5).await?;
    println!("\nMotor Controller Status:");
    println!("  RPM Setpoint:  {} RPM", motor_regs[0]);
    println!("  Actual RPM:    {} RPM", motor_regs[1]);
    println!("  Load:          {}%", motor_regs[2]);
    println!("  Temperature:   {} C", motor_regs[3]);
    println!(
        "  Status:        {}",
        if motor_regs[4] == 0 {
            "Running"
        } else {
            "Stopped"
        }
    );

    // Read motor feedback
    let feedback = client.read_input_registers(0, 3).await?;
    println!("\nMotor Feedback:");
    println!("  Measured RPM:  {} RPM", feedback[0]);
    println!("  Motor Current: {:.1} A", feedback[1] as f64 / 10.0);
    println!("  Voltage:       {:.1} V", feedback[2] as f64 / 10.0);

    // Read control coils
    let coils = client.read_coils(0, 4).await?;
    println!("\nControl Bits:");
    println!("  Enable:    {}", if coils[0] { "ON" } else { "OFF" });
    println!(
        "  Direction: {}",
        if coils[1] { "Reverse" } else { "Forward" }
    );
    println!("  Brake:     {}", if coils[2] { "ON" } else { "OFF" });
    println!("  Ready:     {}", if coils[3] { "YES" } else { "NO" });

    server.shutdown().await;
    println!("\nServer shutdown.\n");

    println!("Demo completed!");

    Ok(())
}
