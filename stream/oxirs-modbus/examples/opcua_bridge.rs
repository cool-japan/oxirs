//! Demonstrates the OPC UA bridge configuration and type coercion without
//! opening any real network sockets.

use oxirs_modbus::opcua::{
    registers_to_value, value_to_registers, BridgeConfig, DataTypeSpec, DataValue, Direction,
    RegisterMapper, RegisterMapping,
};

fn main() -> anyhow::Result<()> {
    // ── 1. Parse a BridgeConfig from a TOML string ───────────────────────────
    let toml_src = r#"
        poll_interval_ms = 500
        modbus_host = "192.168.1.10"
        modbus_port = 502
        opcua_endpoint = "opc.tcp://localhost:4840"

        [[mappings]]
        modbus_register = 100
        opcua_node_id = "ns=2;s=Temperature"
        data_type = "f32"
        direction = "read"

        [[mappings]]
        modbus_register = 200
        opcua_node_id = "ns=2;s=PumpEnable"
        data_type = "bool"
        direction = "bidirectional"

        [[mappings]]
        modbus_register = 300
        opcua_node_id = "ns=2;s=Setpoint"
        data_type = "u16"
        direction = "write"
    "#;

    let config: BridgeConfig = toml::from_str(toml_src)?;
    println!("Bridge configuration:");
    println!("  poll interval : {} ms", config.poll_interval_ms);
    println!(
        "  modbus host   : {}:{}",
        config.modbus_host, config.modbus_port
    );
    println!(
        "  OPC UA endpoint: {}",
        config.opcua_endpoint.as_deref().unwrap_or("<none>")
    );
    println!("  {} mapping(s):", config.mappings.len());
    for m in &config.mappings {
        println!(
            "    register {:>5} ↔ {} ({:?}, {:?})",
            m.modbus_register, m.opcua_node_id, m.data_type, m.direction
        );
    }

    // ── 2. Demonstrate the RegisterMapper ────────────────────────────────────
    let mapper = RegisterMapper::new(config.clone());
    println!("\nReadable mappings: {}", mapper.all_readable().len());
    println!("Writable mappings: {}", mapper.all_writable().len());

    if let Some(m) = mapper.find_mapping(100) {
        println!("Found mapping for register 100: {}", m.opcua_node_id);
    }

    // ── 3. Type coercion examples ─────────────────────────────────────────────
    println!("\nType coercion examples:");

    // U16 round-trip
    let u16_val = DataValue::U16(1234);
    let u16_regs = value_to_registers(&u16_val, &DataTypeSpec::U16)?;
    let u16_back = registers_to_value(&u16_regs, &DataTypeSpec::U16)?;
    println!(
        "  U16 round-trip: {:?} → {:?} → {:?}",
        u16_val, u16_regs, u16_back
    );

    // F32 round-trip (two registers, big-endian word order)
    let temp = 23.75_f32; // not an approx_constant, safe to use literally
    let f32_val = DataValue::F32(temp);
    let f32_regs = value_to_registers(&f32_val, &DataTypeSpec::F32)?;
    let f32_back = registers_to_value(&f32_regs, &DataTypeSpec::F32)?;
    println!(
        "  F32 round-trip: {} → {:?} → {:?}",
        temp, f32_regs, f32_back
    );

    // Bool round-trip
    let bool_val = DataValue::Bool(true);
    let bool_regs = value_to_registers(&bool_val, &DataTypeSpec::Bool)?;
    let bool_back = registers_to_value(&bool_regs, &DataTypeSpec::Bool)?;
    println!(
        "  Bool round-trip: {:?} → {:?} → {:?}",
        bool_val, bool_regs, bool_back
    );

    // I16 minimum value
    let i16_min = DataValue::I16(i16::MIN);
    let i16_regs = value_to_registers(&i16_min, &DataTypeSpec::I16)?;
    let i16_back = registers_to_value(&i16_regs, &DataTypeSpec::I16)?;
    println!(
        "  I16 min round-trip: {:?} → {:?} → {:?}",
        i16_min, i16_regs, i16_back
    );

    // NaN detection
    let nan_val = DataValue::F32(f32::NAN);
    match value_to_registers(&nan_val, &DataTypeSpec::U16) {
        Err(e) => println!("  NaN→U16 correctly rejected: {}", e),
        Ok(r) => println!("  UNEXPECTED: NaN→U16 produced {:?}", r),
    }

    // Infinity detection
    let inf_val = DataValue::F32(f32::INFINITY);
    match value_to_registers(&inf_val, &DataTypeSpec::I16) {
        Err(e) => println!("  Inf→I16 correctly rejected: {}", e),
        Ok(r) => println!("  UNEXPECTED: Inf→I16 produced {:?}", r),
    }

    // Negative-to-U16 error
    let neg_val = DataValue::I16(-1);
    match value_to_registers(&neg_val, &DataTypeSpec::U16) {
        Err(e) => println!("  Negative(-1)→U16 correctly rejected: {}", e),
        Ok(r) => println!("  UNEXPECTED: Negative→U16 produced {:?}", r),
    }

    println!("\nAll coercion examples completed successfully.");
    Ok(())
}
