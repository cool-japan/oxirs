//! Integration tests for Modbus TCP client with mock server
//!
//! These tests verify the client correctly communicates with the server
//! for all supported function codes.
//!
//! Run with: `cargo test --features testing`

#![cfg(feature = "testing")]

use oxirs_modbus::protocol::ModbusTcpClient;
use oxirs_modbus::testing::{MockModbusServer, MockServerData};
use std::time::Duration;

#[tokio::test]
async fn test_read_holding_registers() {
    let server = MockModbusServer::start().await.unwrap();
    let mut client = ModbusTcpClient::connect(server.address(), 1).await.unwrap();

    // Read registers 0-9 (test data has incrementing pattern: 100, 200, 300...)
    let registers = client.read_holding_registers(0, 10).await.unwrap();

    assert_eq!(registers.len(), 10);
    assert_eq!(registers[0], 100);
    assert_eq!(registers[1], 200);
    assert_eq!(registers[2], 300);
    assert_eq!(registers[9], 1000);

    server.shutdown().await;
}

#[tokio::test]
async fn test_read_input_registers() {
    let server = MockModbusServer::start().await.unwrap();
    let mut client = ModbusTcpClient::connect(server.address(), 1).await.unwrap();

    // Read input registers (sensor simulation data)
    let registers = client.read_input_registers(0, 5).await.unwrap();

    assert_eq!(registers.len(), 5);
    assert_eq!(registers[0], 225); // Temperature: 22.5°C
    assert_eq!(registers[1], 501); // Humidity: 50.1%
    assert_eq!(registers[2], 1013); // Pressure: 1013 hPa
    assert_eq!(registers[3], 3300); // Voltage: 330.0V
    assert_eq!(registers[4], 150); // Current: 15.0A

    server.shutdown().await;
}

#[tokio::test]
async fn test_read_coils() {
    let server = MockModbusServer::start().await.unwrap();
    let mut client = ModbusTcpClient::connect(server.address(), 1).await.unwrap();

    // Read coils (alternating pattern)
    let coils = client.read_coils(0, 8).await.unwrap();

    assert_eq!(coils.len(), 8);
    // Alternating pattern: 0=true, 1=false, 2=true, etc.
    assert!(coils[0]);
    assert!(!coils[1]);
    assert!(coils[2]);
    assert!(!coils[3]);

    server.shutdown().await;
}

#[tokio::test]
async fn test_read_discrete_inputs() {
    let server = MockModbusServer::start().await.unwrap();
    let mut client = ModbusTcpClient::connect(server.address(), 1).await.unwrap();

    // Read discrete inputs (first 4 are true, rest are false)
    let inputs = client.read_discrete_inputs(0, 8).await.unwrap();

    assert_eq!(inputs.len(), 8);
    // First 4 are ON, rest are OFF
    assert!(inputs[0]);
    assert!(inputs[1]);
    assert!(inputs[2]);
    assert!(inputs[3]);
    assert!(!inputs[4]);
    assert!(!inputs[5]);

    server.shutdown().await;
}

#[tokio::test]
async fn test_write_single_register() {
    let server = MockModbusServer::start().await.unwrap();
    let mut client = ModbusTcpClient::connect(server.address(), 1).await.unwrap();

    // Write register (server echoes request for confirmation)
    client.write_single_register(10, 12345).await.unwrap();

    // Note: The mock server echoes the write but doesn't store it persistently
    // In a real implementation, we'd verify the write took effect

    server.shutdown().await;
}

#[tokio::test]
async fn test_custom_server_data() {
    // Create custom data
    let mut data = MockServerData::new();
    data.holding_registers.insert(0, 42);
    data.holding_registers.insert(1, 1337);

    let server = MockModbusServer::start_with_data(data).await.unwrap();
    let mut client = ModbusTcpClient::connect(server.address(), 1).await.unwrap();

    let registers = client.read_holding_registers(0, 2).await.unwrap();

    assert_eq!(registers[0], 42);
    assert_eq!(registers[1], 1337);

    server.shutdown().await;
}

#[tokio::test]
async fn test_multiple_reads() {
    let server = MockModbusServer::start().await.unwrap();
    let mut client = ModbusTcpClient::connect(server.address(), 1).await.unwrap();

    // Multiple consecutive reads
    for _ in 0..10 {
        let registers = client.read_holding_registers(0, 5).await.unwrap();
        assert_eq!(registers.len(), 5);
    }

    server.shutdown().await;
}

#[tokio::test]
async fn test_read_single_register() {
    let server = MockModbusServer::start().await.unwrap();
    let mut client = ModbusTcpClient::connect(server.address(), 1).await.unwrap();

    let registers = client.read_holding_registers(0, 1).await.unwrap();

    assert_eq!(registers.len(), 1);
    assert_eq!(registers[0], 100);

    server.shutdown().await;
}

#[tokio::test]
async fn test_read_max_registers() {
    let server = MockModbusServer::start().await.unwrap();
    let mut client = ModbusTcpClient::connect(server.address(), 1).await.unwrap();

    // Read 100 registers (test data has 100 holding registers populated)
    let registers = client.read_holding_registers(0, 100).await.unwrap();

    assert_eq!(registers.len(), 100);
    // Verify pattern
    assert_eq!(registers[0], 100);
    assert_eq!(registers[99], 10000); // (99 + 1) * 100

    server.shutdown().await;
}

#[tokio::test]
async fn test_read_single_coil() {
    let server = MockModbusServer::start().await.unwrap();
    let mut client = ModbusTcpClient::connect(server.address(), 1).await.unwrap();

    let coils = client.read_coils(0, 1).await.unwrap();

    assert_eq!(coils.len(), 1);
    assert!(coils[0]); // First coil is ON

    server.shutdown().await;
}

#[tokio::test]
async fn test_timeout_configuration() {
    let server = MockModbusServer::start().await.unwrap();
    let mut client = ModbusTcpClient::connect(server.address(), 1).await.unwrap();

    // Set a longer timeout
    client.set_timeout(Duration::from_secs(10));

    let registers = client.read_holding_registers(0, 5).await.unwrap();
    assert_eq!(registers.len(), 5);

    server.shutdown().await;
}

#[tokio::test]
async fn test_sensor_data_interpretation() {
    let server = MockModbusServer::start().await.unwrap();
    let mut client = ModbusTcpClient::connect(server.address(), 1).await.unwrap();

    let registers = client.read_input_registers(0, 5).await.unwrap();

    // Interpret sensor values (scaled integers)
    let temperature_celsius = registers[0] as f64 / 10.0;
    let humidity_percent = registers[1] as f64 / 10.0;
    let pressure_hpa = registers[2] as f64;
    let voltage_volts = registers[3] as f64 / 10.0;
    let current_amps = registers[4] as f64 / 10.0;

    assert!((temperature_celsius - 22.5).abs() < 0.01);
    assert!((humidity_percent - 50.1).abs() < 0.01);
    assert!((pressure_hpa - 1013.0).abs() < 0.01);
    assert!((voltage_volts - 330.0).abs() < 0.01);
    assert!((current_amps - 15.0).abs() < 0.01);

    server.shutdown().await;
}

#[tokio::test]
async fn test_write_multiple_registers() {
    let server = MockModbusServer::start().await.unwrap();
    let mut client = ModbusTcpClient::connect(server.address(), 1).await.unwrap();

    // Write multiple registers
    let values = vec![1000, 2000, 3000, 4000, 5000];
    client.write_multiple_registers(100, &values).await.unwrap();

    // Verify the write by reading back
    let read_back = client.read_holding_registers(100, 5).await.unwrap();

    assert_eq!(read_back.len(), 5);
    assert_eq!(read_back[0], 1000);
    assert_eq!(read_back[1], 2000);
    assert_eq!(read_back[2], 3000);
    assert_eq!(read_back[3], 4000);
    assert_eq!(read_back[4], 5000);

    server.shutdown().await;
}

#[tokio::test]
async fn test_write_multiple_coils() {
    let server = MockModbusServer::start().await.unwrap();
    let mut client = ModbusTcpClient::connect(server.address(), 1).await.unwrap();

    // Write multiple coils
    let coils = vec![true, false, true, true, false, false, true, true];
    client.write_multiple_coils(100, &coils).await.unwrap();

    // Verify the write by reading back
    let read_back = client.read_coils(100, 8).await.unwrap();

    assert_eq!(read_back.len(), 8);
    assert!(read_back[0]); // true
    assert!(!read_back[1]); // false
    assert!(read_back[2]); // true
    assert!(read_back[3]); // true
    assert!(!read_back[4]); // false
    assert!(!read_back[5]); // false
    assert!(read_back[6]); // true
    assert!(read_back[7]); // true

    server.shutdown().await;
}

#[tokio::test]
async fn test_write_single_register_verify() {
    let server = MockModbusServer::start().await.unwrap();
    let mut client = ModbusTcpClient::connect(server.address(), 1).await.unwrap();

    // Write a single register
    client.write_single_register(50, 12345).await.unwrap();

    // Verify by reading back
    let read_back = client.read_holding_registers(50, 1).await.unwrap();

    assert_eq!(read_back.len(), 1);
    assert_eq!(read_back[0], 12345);

    server.shutdown().await;
}

#[tokio::test]
async fn test_write_multiple_registers_batch() {
    let server = MockModbusServer::start().await.unwrap();
    let mut client = ModbusTcpClient::connect(server.address(), 1).await.unwrap();

    // Write a larger batch (50 registers)
    let values: Vec<u16> = (0..50).map(|i| i * 100).collect();
    client.write_multiple_registers(200, &values).await.unwrap();

    // Verify by reading back
    let read_back = client.read_holding_registers(200, 50).await.unwrap();

    assert_eq!(read_back.len(), 50);
    for (i, &val) in read_back.iter().enumerate() {
        assert_eq!(val, (i * 100) as u16);
    }

    server.shutdown().await;
}
