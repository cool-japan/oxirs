//! Modbus protocol CLI commands
//!
//! Provides CLI commands for:
//! - Monitoring Modbus TCP devices (real network I/O via `oxirs_modbus`)
//! - Reading and writing holding registers over TCP
//! - Running a real mock Modbus TCP server for testing
//!
//! Modbus RTU (serial) and CAN/register-to-RDF generation are not yet wired
//! into the CLI (RTU would require enabling `oxirs-modbus`'s `rtu` feature,
//! which pulls in the `tokio-serial` -> `serialport` C/udev dependency chain
//! that the COOLJAPAN Pure-Rust default-build policy keeps out of the
//! default build); those commands fail loudly with an explicit error rather
//! than silently reporting fake success.

use crate::cli::CliContext;
use crate::cli_actions::ModbusAction;
use anyhow::{Context, Result};
use colored::Colorize;
use oxirs_modbus::protocol::ModbusTcpClient;
use oxirs_modbus::testing::MockModbusServer;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::Duration;

/// Execute Modbus command
pub async fn execute(action: ModbusAction, _ctx: &CliContext) -> Result<()> {
    match action {
        ModbusAction::MonitorTcp {
            address,
            unit_id,
            start,
            count,
            interval,
            format,
            output,
        } => monitor_tcp_command(address, unit_id, start, count, interval, format, output).await,
        ModbusAction::MonitorRtu { .. } => {
            anyhow::bail!(
                "`oxirs modbus monitor-rtu` is not yet supported: RTU (serial) support \
                 requires enabling oxirs-modbus's `rtu` feature, which pulls in a \
                 C/udev serial-port dependency chain kept out of the default Pure-Rust \
                 build. Use `monitor-tcp` against a Modbus TCP gateway instead."
            )
        }
        ModbusAction::Read {
            device,
            register_type,
            address,
            count,
            datatype,
        } => read_command(device, register_type, address, count, datatype).await,
        ModbusAction::Write {
            device,
            address,
            value,
            datatype,
        } => write_command(device, address, value, datatype).await,
        ModbusAction::ToRdf { .. } => {
            anyhow::bail!(
                "`oxirs modbus to-rdf` is not yet wired into the CLI: register-map \
                 configuration parsing and RDF/PROV-O triple generation from live \
                 readings have not been implemented in this pass."
            )
        }
        ModbusAction::MockServer { port, config } => mock_server_command(port, config).await,
    }
}

/// Connect a real `ModbusTcpClient` to `device`, reporting a clear error if
/// the address cannot be reached.
async fn connect_tcp(device: &str, unit_id: u8) -> Result<ModbusTcpClient> {
    ModbusTcpClient::connect(device, unit_id)
        .await
        .with_context(|| format!("Failed to connect to Modbus TCP device at {device}"))
}

/// Read `count` registers of `register_type` starting at `address` and
/// return them as raw `u16` values.
async fn read_registers(
    client: &mut ModbusTcpClient,
    register_type: &str,
    address: u16,
    count: u16,
) -> Result<Vec<u16>> {
    match register_type.to_lowercase().as_str() {
        "holding" => client
            .read_holding_registers(address, count)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read holding registers: {e}")),
        "input" => client
            .read_input_registers(address, count)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read input registers: {e}")),
        "coil" => {
            let bits = client
                .read_coils(address, count)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to read coils: {e}"))?;
            Ok(bits.into_iter().map(u16::from).collect())
        }
        "discrete" => {
            let bits = client
                .read_discrete_inputs(address, count)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to read discrete inputs: {e}"))?;
            Ok(bits.into_iter().map(u16::from).collect())
        }
        other => anyhow::bail!(
            "Unsupported register type '{other}'; supported: holding, input, coil, discrete"
        ),
    }
}

/// Interpret raw register(s) according to `datatype`. Returns a display string.
fn interpret_registers(registers: &[u16], datatype: Option<&str>) -> String {
    let dt = datatype.map(str::to_lowercase);
    match dt.as_deref() {
        Some("int16") => format!("{}", registers.first().copied().unwrap_or(0) as i16),
        Some("uint16") | None => registers
            .iter()
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        Some(name @ ("int32" | "uint32" | "float32")) if registers.len() < 2 => {
            format!("(need 2 registers for '{name}', got {})", registers.len())
        }
        Some("int32") => {
            let raw = ((registers[0] as u32) << 16) | registers[1] as u32;
            format!("{}", raw as i32)
        }
        Some("uint32") => {
            let raw = ((registers[0] as u32) << 16) | registers[1] as u32;
            format!("{raw}")
        }
        Some("float32") => {
            let raw = ((registers[0] as u32) << 16) | registers[1] as u32;
            format!("{}", f32::from_bits(raw))
        }
        Some("bit") => format!("{}", registers.first().copied().unwrap_or(0) != 0),
        Some(other) => format!("(unsupported datatype '{other}', raw: {registers:?})"),
    }
}

async fn monitor_tcp_command(
    address: String,
    unit_id: u8,
    start: u16,
    count: u16,
    interval: u64,
    format: String,
    output: Option<PathBuf>,
) -> Result<()> {
    println!("{}", "Monitor Modbus TCP device".bright_cyan().bold());
    println!("Address: {}", address.yellow());
    println!("Unit ID: {}", unit_id);
    println!("Registers: {} - {}", start, start + count.max(1) - 1);
    println!("Interval: {} ms", interval);
    if let Some(out) = &output {
        println!("Output: {} ({})", out.display(), format);
    }

    let mut client = connect_tcp(&address, unit_id).await?;
    println!("\n{} Connected", "✓".green());
    println!("{} Press Ctrl+C to stop\n", "Info:".cyan());

    let mut out_file = match &output {
        Some(path) => Some(
            std::fs::File::create(path)
                .with_context(|| format!("Failed to create output file {}", path.display()))?,
        ),
        None => None,
    };

    let mut ticker = tokio::time::interval(Duration::from_millis(interval.max(1)));
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                println!("\n{} Monitoring stopped", "Info:".cyan());
                break;
            }
            _ = ticker.tick() => {
                let now = chrono::Local::now();
                match client.read_holding_registers(start, count).await {
                    Ok(registers) => {
                        for (i, value) in registers.iter().enumerate() {
                            let line = format!(
                                "[{}] Register {}: {}",
                                now.format("%H:%M:%S%.3f"),
                                start + i as u16,
                                value
                            );
                            println!("{line}");
                            if let Some(f) = out_file.as_mut() {
                                writeln!(f, "{line}").with_context(|| "Failed to write output file")?;
                            }
                        }
                    }
                    Err(e) => {
                        anyhow::bail!("Failed to read registers from {address}: {e}");
                    }
                }
            }
        }
    }

    Ok(())
}

async fn read_command(
    device: String,
    register_type: String,
    address: u16,
    count: u16,
    datatype: Option<String>,
) -> Result<()> {
    println!("{}", "Read Modbus registers".bright_cyan().bold());
    println!("Device: {}", device.yellow());
    println!("Type: {}", register_type);
    println!("Address: {}", address);
    println!("Count: {}", count);
    if let Some(dt) = &datatype {
        println!("Data type: {}", dt);
    }

    let mut client = connect_tcp(&device, 1).await?;
    let registers = read_registers(&mut client, &register_type, address, count).await?;

    println!("\n{} Raw registers: {:?}", "✓".green(), registers);
    println!(
        "Interpreted value: {}",
        interpret_registers(&registers, datatype.as_deref())
    );

    Ok(())
}

async fn write_command(
    device: String,
    address: u16,
    value: String,
    datatype: String,
) -> Result<()> {
    println!("{}", "Write Modbus register".bright_cyan().bold());
    println!("Device: {}", device.yellow());
    println!("Address: {}", address);
    println!("Value: {}", value);
    println!("Data type: {}", datatype);

    let raw: u16 = match datatype.to_lowercase().as_str() {
        "uint16" | "int16" | "bit" => value
            .parse::<i32>()
            .with_context(|| format!("Invalid integer value '{value}'"))?
            as u16,
        other => anyhow::bail!(
            "`oxirs modbus write` currently only supports single-register datatypes \
             (uint16, int16, bit); '{other}' requires a multi-register write which is \
             not yet wired into the CLI."
        ),
    };

    let mut client = connect_tcp(&device, 1).await?;
    client
        .write_single_register(address, raw)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to write register: {e}"))?;

    println!("\n{} Register written", "✓".green());

    Ok(())
}

async fn mock_server_command(port: u16, config: Option<PathBuf>) -> Result<()> {
    println!("{}", "Start Modbus mock server".bright_cyan().bold());
    println!("Requested port: {}", port.to_string().yellow());

    if let Some(cfg) = &config {
        println!(
            "{} --config is not yet supported for the mock server; using built-in test data ({})",
            "Warning:".yellow(),
            cfg.display()
        );
    }

    // The underlying `MockModbusServer` always binds an OS-assigned ephemeral
    // port (`127.0.0.1:0`) rather than a caller-chosen one; report the real
    // bound address truthfully instead of pretending `--port` was honored.
    let server = MockModbusServer::start()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to start mock Modbus server: {e}"))?;

    println!(
        "\n{} Mock server running on {} (real listener, holding/input registers pre-populated)",
        "✓".green(),
        server.address()
    );
    if server.address().split(':').next_back() != Some(&port.to_string()) {
        println!(
            "{} the underlying mock server binds an OS-assigned ephemeral port; \
             requested port {port} could not be honored.",
            "Note:".yellow()
        );
    }
    println!("{} Press Ctrl+C to stop", "Info:".cyan());

    tokio::signal::ctrl_c()
        .await
        .context("Failed to listen for Ctrl+C")?;

    println!("\n{} Shutting down mock server", "Info:".cyan());
    server.shutdown().await;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: `oxirs modbus read` (and by extension `monitor-tcp` /
    /// `write`) must perform a real TCP round trip against a Modbus server
    /// instead of printing "TODO" and returning `Ok(())` with no I/O at all.
    #[tokio::test]
    async fn test_read_command_round_trips_against_real_mock_server() {
        let server = MockModbusServer::start()
            .await
            .expect("start mock modbus server");

        let mut client = connect_tcp(server.address(), 1)
            .await
            .expect("connect to mock server");

        // MockServerData::with_test_data() seeds holding_registers[0] = 100.
        let registers = read_registers(&mut client, "holding", 0, 1)
            .await
            .expect("read holding register 0");

        assert_eq!(registers, vec![100]);

        server.shutdown().await;
    }

    #[tokio::test]
    async fn test_write_then_read_round_trips_a_real_value() {
        let server = MockModbusServer::start()
            .await
            .expect("start mock modbus server");

        let mut writer = connect_tcp(server.address(), 1)
            .await
            .expect("connect writer");
        writer
            .write_single_register(50, 4242)
            .await
            .expect("write register 50");

        let mut reader = connect_tcp(server.address(), 1)
            .await
            .expect("connect reader");
        let registers = read_registers(&mut reader, "holding", 50, 1)
            .await
            .expect("read register 50 back");

        assert_eq!(registers, vec![4242]);

        server.shutdown().await;
    }

    #[test]
    fn test_interpret_registers_float32() {
        let value = 22.5_f32;
        let bits = value.to_bits();
        let hi = (bits >> 16) as u16;
        let lo = (bits & 0xFFFF) as u16;
        let display = interpret_registers(&[hi, lo], Some("float32"));
        let parsed: f32 = display.parse().expect("should parse as float");
        assert!((parsed - 22.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_interpret_registers_uint16() {
        assert_eq!(interpret_registers(&[42], Some("uint16")), "42");
    }
}
