//! Register Mapping Example
//!
//! Demonstrates the register-to-RDF mapping functionality:
//! - Creating register maps
//! - Data type conversions (INT16, FLOAT32, etc.)
//! - Scaling configurations
//! - Deadband change detection
//! - Batch read optimization
//!
//! # Usage
//!
//! ```bash
//! cargo run --example register_mapping
//! ```

use oxirs_modbus::mapping::{
    decode_registers, LinearScaling, ModbusDataType, ModbusValue, RegisterMap, RegisterMapping,
    RegisterType,
};
use oxirs_modbus::ModbusResult;

fn main() -> ModbusResult<()> {
    println!("=== OxiRS Modbus Register Mapping Example ===\n");

    // Demo 1: Basic register mapping
    demo_basic_mapping()?;

    // Demo 2: Data type conversions
    demo_data_types()?;

    // Demo 3: Scaling and deadband
    demo_scaling_deadband()?;

    // Demo 4: Batch read optimization
    demo_batch_optimization()?;

    // Demo 5: TOML configuration
    demo_toml_config()?;

    println!("\n=== Register Mapping Example Complete ===");
    Ok(())
}

fn demo_basic_mapping() -> ModbusResult<()> {
    println!("Demo 1: Basic Register Mapping");
    println!("------------------------------");

    // Create a register map for a PLC
    let mut map = RegisterMap::new("plc001", "http://factory.example.com/device");

    // Add temperature sensor at register 0
    map.add_register(
        RegisterMapping::new(
            0,
            ModbusDataType::Float32,
            "http://factory.example.com/property/temperature",
        )
        .with_name("Temperature Sensor 1")
        .with_unit("CEL"),
    );

    // Add pressure sensor at register 2 (FLOAT32 uses 2 registers)
    map.add_register(
        RegisterMapping::new(
            2,
            ModbusDataType::Float32,
            "http://factory.example.com/property/pressure",
        )
        .with_name("Pressure Sensor 1")
        .with_unit("BAR"),
    );

    // Add motor speed at register 4 (INT16, single register)
    map.add_register(
        RegisterMapping::new(
            4,
            ModbusDataType::Int16,
            "http://factory.example.com/property/motorSpeed",
        )
        .with_name("Motor Speed")
        .with_unit("RPM"),
    );

    // Add pump status at register 5 (UINT16 for enum)
    map.add_register(
        RegisterMapping::new(
            5,
            ModbusDataType::Uint16,
            "http://factory.example.com/property/pumpStatus",
        )
        .with_name("Pump Status"),
    );

    println!("Created register map for device: {}", map.device_id);
    println!("Base IRI: {}", map.base_iri);
    println!("Number of registers mapped: {}", map.registers.len());
    println!();

    for reg in &map.registers {
        println!(
            "  Register {}: {} ({:?})",
            reg.address,
            reg.name.as_deref().unwrap_or("unnamed"),
            reg.data_type
        );
        if let Some(unit) = &reg.unit {
            println!("    Unit: {}", unit);
        }
    }

    println!();
    Ok(())
}

fn demo_data_types() -> ModbusResult<()> {
    println!("Demo 2: Data Type Conversions");
    println!("-----------------------------");

    // Demonstrate how different data types are decoded

    // INT16: Signed 16-bit (one register)
    let int16_regs = vec![0xFF9C]; // -100 in two's complement
    let int16_value = decode_registers(&int16_regs, ModbusDataType::Int16)?;
    println!("INT16: registers {:04X?} = {:?}", int16_regs, int16_value);

    // UINT16: Unsigned 16-bit (one register)
    let uint16_regs = vec![0x03E8]; // 1000
    let uint16_value = decode_registers(&uint16_regs, ModbusDataType::Uint16)?;
    println!(
        "UINT16: registers {:04X?} = {:?}",
        uint16_regs, uint16_value
    );

    // INT32: Signed 32-bit (two registers)
    let int32_regs = vec![0x0001, 0x86A0]; // 100000
    let int32_value = decode_registers(&int32_regs, ModbusDataType::Int32)?;
    println!("INT32: registers {:04X?} = {:?}", int32_regs, int32_value);

    // FLOAT32: IEEE 754 (two registers) - 22.5
    let float32_regs = vec![0x41B4, 0x0000]; // 22.5
    let float32_value = decode_registers(&float32_regs, ModbusDataType::Float32)?;
    println!(
        "FLOAT32: registers {:04X?} = {:?}",
        float32_regs, float32_value
    );

    // Demonstrate ModbusValue variants
    println!();
    println!("ModbusValue variants:");

    match int16_value {
        ModbusValue::Int(v) => println!("  Int variant: {}", v),
        _ => println!("  Unexpected type"),
    }

    match uint16_value {
        ModbusValue::Uint(v) => println!("  Uint variant: {}", v),
        _ => println!("  Unexpected type"),
    }

    match float32_value {
        ModbusValue::Float(v) => println!("  Float variant: {:.2}", v),
        _ => println!("  Unexpected type"),
    }

    println!();
    Ok(())
}

fn demo_scaling_deadband() -> ModbusResult<()> {
    println!("Demo 3: Scaling and Deadband");
    println!("----------------------------");

    // Create register with scaling
    let scaling = LinearScaling::new(0.1, -40.0);
    println!("Linear Scaling: value = raw * {} + {}", 0.1, -40.0);
    println!("  Raw 0 -> {} degC", scaling.apply(0.0));
    println!("  Raw 250 -> {} degC", scaling.apply(250.0));
    println!("  Raw 500 -> {} degC", scaling.apply(500.0));
    println!("  Raw 1000 -> {} degC", scaling.apply(1000.0));

    println!();

    // Create a register map with deadband
    let mut map = RegisterMap::new("sensor001", "http://example.org/device");

    map.add_register(
        RegisterMapping::new(
            0,
            ModbusDataType::Uint16,
            "http://example.org/property/temperature",
        )
        .with_name("Temperature")
        .with_unit("CEL")
        .with_scaling(0.1, -40.0)
        .with_deadband(0.5), // Only report changes > 0.5 degrees
    );

    println!("Deadband Example:");
    println!("  Deadband threshold: 0.5 degrees");
    println!();

    // Simulate readings
    let readings: [f64; 6] = [22.0, 22.3, 22.1, 22.8, 22.7, 23.5];
    let mut last_reported: Option<f64> = None;

    println!(
        "  {:>10} | {:>10} | {:>12} | {:>10}",
        "Reading", "Scaled", "Change", "Report?"
    );
    println!("  {:-<10}-+-{:-<10}-+-{:-<12}-+-{:-<10}", "", "", "", "");

    for (i, &raw) in readings.iter().enumerate() {
        let scaled = raw; // Already scaled for demo
        let change = last_reported.map(|last| (scaled - last).abs());
        let should_report = change.map(|c| c > 0.5).unwrap_or(true);

        let change_str = change
            .map(|c| format!("{:.2}", c))
            .unwrap_or_else(|| "-".to_string());
        let report_str = if should_report { "YES" } else { "no" };

        println!(
            "  {:>10.1} | {:>10.1} | {:>12} | {:>10}",
            raw, scaled, change_str, report_str
        );

        if should_report {
            last_reported = Some(scaled);
        }

        if i == 0 {
            last_reported = Some(scaled); // First reading always establishes baseline
        }
    }

    println!();
    println!("Result: Only 3 updates instead of 6 (50% reduction)");

    println!();
    Ok(())
}

fn demo_batch_optimization() -> ModbusResult<()> {
    println!("Demo 4: Batch Read Optimization");
    println!("-------------------------------");

    // Create a register map with scattered registers
    let mut map = RegisterMap::new("plc001", "http://factory.example.com/device");

    // Cluster 1: Registers 0-5
    for i in 0..6 {
        map.add_register(
            RegisterMapping::new(
                i,
                ModbusDataType::Uint16,
                format!("http://factory.example.com/property/sensor{}", i),
            )
            .with_name(format!("Sensor {}", i)),
        );
    }

    // Gap: Registers 6-49 not used

    // Cluster 2: Registers 50-60
    for i in 50..61 {
        map.add_register(
            RegisterMapping::new(
                i,
                ModbusDataType::Uint16,
                format!("http://factory.example.com/property/sensor{}", i),
            )
            .with_name(format!("Sensor {}", i)),
        );
    }

    // Gap: Registers 61-99 not used

    // Single register: Register 100
    map.add_register(
        RegisterMapping::new(
            100,
            ModbusDataType::Float32,
            "http://factory.example.com/property/temperature",
        )
        .with_name("Temperature"),
    );

    println!("Register layout:");
    println!("  Cluster 1: Registers 0-5 (6 registers)");
    println!("  Gap: Registers 6-49 (unused)");
    println!("  Cluster 2: Registers 50-60 (11 registers)");
    println!("  Gap: Registers 61-99 (unused)");
    println!("  Single: Register 100 (2 registers for FLOAT32)");
    println!();

    // Get optimized batches (max 125 registers per request)
    let batches = map.batch_reads(RegisterType::Holding, 125);

    println!("Optimized batch reads (max 125 registers per request):");
    for (i, batch) in batches.iter().enumerate() {
        println!(
            "  Batch {}: Start={}, Count={} registers",
            i + 1,
            batch.0,
            batch.1
        );
    }

    println!();
    println!("Optimization Results:");
    println!("  Without batching: {} individual requests", 6 + 11 + 1);
    println!("  With batching: {} requests", batches.len());
    println!(
        "  Reduction: {}%",
        ((18 - batches.len()) as f32 / 18.0 * 100.0) as i32
    );

    println!();
    Ok(())
}

fn demo_toml_config() -> ModbusResult<()> {
    println!("Demo 5: TOML Configuration");
    println!("--------------------------");

    // Example TOML content
    let toml_content = r#"
device_id = "energy_meter_001"
base_iri = "http://factory.example.com/device"
polling_interval_ms = 1000

[[registers]]
address = 0
data_type = "FLOAT32"
register_type = "Input"
predicate = "http://factory.example.com/property/voltage_L1"
name = "Voltage L1"
unit = "V"
description = "Phase 1 voltage"

[[registers]]
address = 2
data_type = "FLOAT32"
register_type = "Input"
predicate = "http://factory.example.com/property/voltage_L2"
name = "Voltage L2"
unit = "V"

[[registers]]
address = 4
data_type = "FLOAT32"
register_type = "Input"
predicate = "http://factory.example.com/property/voltage_L3"
name = "Voltage L3"
unit = "V"

[[registers]]
address = 10
data_type = "FLOAT32"
register_type = "Input"
predicate = "http://factory.example.com/property/current_L1"
name = "Current L1"
unit = "A"

[registers.scaling]
multiplier = 0.001
offset = 0.0

[[registers]]
address = 20
data_type = "FLOAT32"
register_type = "Input"
predicate = "http://factory.example.com/property/power_total"
name = "Total Power"
unit = "kW"
deadband = 0.1
"#;

    println!("Example TOML configuration:");
    println!("{}", toml_content);

    // Parse the TOML
    let map: RegisterMap = toml::from_str(toml_content).expect("Failed to parse TOML");

    println!("Parsed configuration:");
    println!("  Device ID: {}", map.device_id);
    println!("  Base IRI: {}", map.base_iri);
    println!("  Polling interval: {}ms", map.polling_interval_ms);
    println!("  Registers: {}", map.registers.len());

    println!();
    println!("Mapped Registers:");
    for reg in &map.registers {
        println!(
            "  Address {}: {} ({:?})",
            reg.address,
            reg.name.as_deref().unwrap_or("unnamed"),
            reg.data_type
        );
    }

    println!();
    Ok(())
}
