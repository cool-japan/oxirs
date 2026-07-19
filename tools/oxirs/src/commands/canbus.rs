//! CANbus/J1939 protocol CLI commands
//!
//! Provides CLI commands for:
//! - Parsing DBC files (real `oxirs_canbus::parse_dbc_file`)
//! - Decoding CAN frames against a DBC database (real `SignalDecoder`)
//! - Generating SAMM Aspect Models from DBC (real `DbcSammGenerator`)
//!
//! Live CAN interface access (`monitor`, `send`, `to-rdf`, `replay`) is not
//! yet wired into the CLI: the underlying `oxirs_canbus::CanbusClient` is
//! Linux-only (SocketCAN), and this pass focuses on the file/computation
//! based commands that are portable and verifiable in CI. Those commands
//! fail loudly with an explicit error instead of silently reporting fake
//! success.

use crate::cli::CliContext;
use crate::cli_actions::CanbusAction;
use anyhow::{Context, Result};
use colored::Colorize;
use oxirs_canbus::{
    parse_dbc_file, DbcDatabase, DbcMessage, DbcSammGenerator, SammConfig, SignalDecoder,
};
use prettytable::{row, Table};
use std::path::PathBuf;

/// Execute CANbus command
pub async fn execute(action: CanbusAction, _ctx: &CliContext) -> Result<()> {
    match action {
        CanbusAction::Monitor { .. } => {
            anyhow::bail!(
                "`oxirs canbus monitor` is not yet wired into the CLI: live SocketCAN \
                 interface access is Linux-only and has not been integrated in this \
                 pass. Use `parse-dbc` / `decode` against captured frames instead."
            )
        }
        CanbusAction::ParseDbc {
            file,
            format,
            detailed,
        } => parse_dbc_command(file, format, detailed).await,
        CanbusAction::Decode {
            id,
            data,
            dbc,
            format,
        } => decode_command(id, data, dbc, format).await,
        CanbusAction::Send { .. } => {
            anyhow::bail!(
                "`oxirs canbus send` is not yet wired into the CLI: live SocketCAN \
                 interface access is Linux-only and has not been integrated in this pass."
            )
        }
        CanbusAction::ToSamm {
            dbc,
            output,
            namespace,
            per_message,
        } => to_samm_command(dbc, output, namespace, per_message).await,
        CanbusAction::ToRdf { .. } => {
            anyhow::bail!(
                "`oxirs canbus to-rdf` is not yet wired into the CLI: live-capture \
                 RDF/PROV-O triple generation from a CAN interface has not been \
                 implemented in this pass."
            )
        }
        CanbusAction::Replay { .. } => {
            anyhow::bail!(
                "`oxirs canbus replay` is not yet wired into the CLI: live SocketCAN \
                 interface access is Linux-only and has not been integrated in this pass."
            )
        }
    }
}

async fn parse_dbc_command(file: PathBuf, _format: String, detailed: bool) -> Result<()> {
    println!("{}", "Parse DBC file".bright_cyan().bold());
    println!("File: {}", file.display().to_string().yellow());

    let database: DbcDatabase = parse_dbc_file(&file)
        .map_err(|e| anyhow::anyhow!("Failed to parse DBC file {}: {e}", file.display()))?;

    println!(
        "\n{} Parsed {} message(s), {} node(s)",
        "✓".green(),
        database.messages.len(),
        database.nodes.len()
    );

    let mut table = Table::new();
    table.add_row(row!["Message ID", "Name", "DLC", "Signals"]);
    for message in &database.messages {
        table.add_row(row![
            format!("0x{:08X}", message.id),
            message.name.clone(),
            message.dlc.to_string(),
            message.signals.len().to_string()
        ]);
    }
    table.printstd();

    if detailed {
        println!("\n{}", "Signal Details:".bright_cyan());
        for message in &database.messages {
            for signal in &message.signals {
                println!(
                    "  {} :: {} - {} bit(s) at offset {}, scale: {}, unit: {}",
                    message.name,
                    signal.name,
                    signal.bit_length,
                    signal.start_bit,
                    signal.factor,
                    if signal.unit.is_empty() {
                        "-"
                    } else {
                        signal.unit.as_str()
                    }
                );
            }
        }
    }

    Ok(())
}

async fn decode_command(id: String, data: String, dbc: PathBuf, _format: String) -> Result<()> {
    println!("{}", "Decode CAN frame".bright_cyan().bold());
    println!("CAN ID: {}", id.yellow());
    println!("Data: {}", data);
    println!("DBC: {}", dbc.display());

    let can_id = parse_can_id(&id)?;
    let can_data = parse_hex_data(&data)?;

    println!(
        "\nParsed: 0x{:03X} [{}] {}",
        can_id,
        can_data.len(),
        can_data
            .iter()
            .map(|b| format!("{:02X}", b))
            .collect::<Vec<_>>()
            .join(" ")
    );

    let database = parse_dbc_file(&dbc)
        .map_err(|e| anyhow::anyhow!("Failed to parse DBC file {}: {e}", dbc.display()))?;
    let decoder = SignalDecoder::new(&database);
    let decoded = decoder
        .decode_message(can_id, &can_data)
        .map_err(|e| anyhow::anyhow!("Failed to decode CAN ID 0x{can_id:03X}: {e}"))?;

    if decoded.is_empty() {
        anyhow::bail!(
            "No signals decoded for CAN ID 0x{can_id:03X}: the DBC file has no message \
             definition for this ID, or every signal failed to extract from the given data."
        );
    }

    let mut names: Vec<&String> = decoded.keys().collect();
    names.sort();

    let mut table = Table::new();
    table.add_row(row!["Signal", "Value", "Unit", "Description"]);
    for name in names {
        let value = &decoded[name];
        table.add_row(row![
            value.name.clone(),
            format!("{:.3}", value.physical_value),
            value.unit.clone(),
            value.description.clone().unwrap_or_default()
        ]);
    }
    table.printstd();

    Ok(())
}

async fn to_samm_command(
    dbc: PathBuf,
    output: PathBuf,
    namespace: String,
    per_message: bool,
) -> Result<()> {
    println!("{}", "Generate SAMM from DBC".bright_cyan().bold());
    println!("DBC: {}", dbc.display().to_string().yellow());
    println!("Output: {}", output.display());
    println!("Namespace: {}", namespace);

    let database = parse_dbc_file(&dbc)
        .map_err(|e| anyhow::anyhow!("Failed to parse DBC file {}: {e}", dbc.display()))?;

    if database.messages.is_empty() {
        anyhow::bail!(
            "DBC file {} defines no messages; nothing to generate.",
            dbc.display()
        );
    }

    let config = SammConfig::new("1.0.0", namespace);
    let generator = DbcSammGenerator::new(config);

    std::fs::create_dir_all(&output)
        .with_context(|| format!("Failed to create output directory {}", output.display()))?;

    let mut generated: Vec<(String, usize)> = Vec::new();
    if per_message {
        println!("Mode: {}", "Separate Aspect per message".yellow());
        for message in &database.messages {
            let ttl = generator.generate_for_message(message);
            let file_name = format!("{}.ttl", sanitize_filename(&message.name));
            let path = output.join(&file_name);
            std::fs::write(&path, ttl)
                .with_context(|| format!("Failed to write {}", path.display()))?;
            generated.push((file_name, message.signals.len()));
        }
    } else {
        let ttl = generator.generate_from_database(&database);
        let path = output.join("Aspects.ttl");
        std::fs::write(&path, ttl)
            .with_context(|| format!("Failed to write {}", path.display()))?;
        let total_signals: usize = database.messages.iter().map(signal_count).sum();
        generated.push(("Aspects.ttl".to_string(), total_signals));
    }

    println!("\n{} SAMM generation complete", "✓".green());
    for (file_name, property_count) in &generated {
        println!(
            "  ✓ Generated {} ({} properties)",
            file_name, property_count
        );
    }

    Ok(())
}

fn signal_count(message: &DbcMessage) -> usize {
    message.signals.len()
}

fn sanitize_filename(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "Message".to_string()
    } else {
        sanitized
    }
}

fn parse_can_id(id_str: &str) -> Result<u32> {
    if id_str.starts_with("0x") || id_str.starts_with("0X") {
        u32::from_str_radix(&id_str[2..], 16).context("Invalid hex CAN ID")
    } else {
        id_str.parse::<u32>().context("Invalid CAN ID")
    }
}

fn parse_hex_data(data_str: &str) -> Result<Vec<u8>> {
    let clean = data_str
        .replace(" ", "")
        .replace("0x", "")
        .replace("0X", "");
    if clean.len() % 2 != 0 {
        anyhow::bail!("Hex data must have even number of characters");
    }

    (0..clean.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&clean[i..i + 2], 16).context("Invalid hex byte"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dbc_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "oxirs-canbus-cmd-test-{tag}-{}.dbc",
            std::process::id() as u64 * 1_299_709 + line!() as u64
        ))
    }

    const SAMPLE_DBC: &str = r#"
VERSION ""

NS_ :
    NS_DESC_
    CM_
    BA_DEF_
    BA_

BS_:

BU_: ECU

BO_ 100 EngineData: 8 ECU
 SG_ EngineSpeed : 24|16@1+ (0.125,0) [0|8000] "rpm" ECU
 SG_ EngineTemp : 8|8@1+ (1,-40) [-40|215] "degC" ECU
"#;

    /// Regression test: `oxirs canbus parse-dbc` must parse the real DBC
    /// file instead of printing hardcoded fake message rows.
    #[tokio::test]
    async fn test_parse_dbc_command_parses_real_file() {
        let path = temp_dbc_path("parse");
        std::fs::write(&path, SAMPLE_DBC).expect("write sample dbc");

        let database = parse_dbc_file(&path).expect("parse real dbc file");
        assert_eq!(database.messages.len(), 1);
        assert_eq!(database.messages[0].name, "EngineData");
        assert_eq!(database.messages[0].signals.len(), 2);

        std::fs::remove_file(&path).ok();
    }

    /// Regression test: `oxirs canbus decode` must decode the frame against
    /// the real DBC-defined signals instead of printing hardcoded fake
    /// "EngineSpeed 1850.0 rpm" values.
    #[tokio::test]
    async fn test_decode_command_decodes_real_signal_values() {
        let path = temp_dbc_path("decode");
        std::fs::write(&path, SAMPLE_DBC).expect("write sample dbc");

        let database = parse_dbc_file(&path).expect("parse real dbc file");
        let decoder = SignalDecoder::new(&database);

        // EngineSpeed: 16 bits at byte offset 24 (byte 3), little-endian,
        // scale 0.125 -> raw 800 => 100.0 rpm. Byte layout (8 bytes):
        // [_, _, _, lo, hi, _, _, _] with EngineTemp raw=140 at byte 1.
        let mut data = [0u8; 8];
        data[1] = 140; // EngineTemp raw -> physical = 140 - 40 = 100
        data[3] = 0x20; // EngineSpeed low byte (raw 800 = 0x0320)
        data[4] = 0x03; // EngineSpeed high byte

        let decoded = decoder
            .decode_message(100, &data)
            .expect("decode real frame");

        assert_eq!(decoded.len(), 2);
        let speed = &decoded["EngineSpeed"];
        assert!((speed.physical_value - 100.0).abs() < 0.01);
        assert_eq!(speed.unit, "rpm");

        let temp = &decoded["EngineTemp"];
        assert!((temp.physical_value - 100.0).abs() < 0.01);

        std::fs::remove_file(&path).ok();
    }

    /// Regression test: `oxirs canbus to-samm` must write real generated
    /// Turtle content derived from the DBC signals, not a fabricated
    /// "Generated Movement.ttl (5 properties)" success message with no file.
    #[tokio::test]
    async fn test_to_samm_command_writes_real_turtle_output() {
        let dbc_path = temp_dbc_path("samm");
        std::fs::write(&dbc_path, SAMPLE_DBC).expect("write sample dbc");

        let out_dir = std::env::temp_dir().join(format!(
            "oxirs-canbus-samm-out-{}",
            std::process::id() as u64 * 7_247 + line!() as u64
        ));

        to_samm_command(
            dbc_path.clone(),
            out_dir.clone(),
            "urn:samm:org.example.can".to_string(),
            false,
        )
        .await
        .expect("to_samm_command should succeed");

        let generated = std::fs::read_to_string(out_dir.join("Aspects.ttl"))
            .expect("Aspects.ttl should have been written");
        assert!(generated.contains("EngineData") || generated.contains("EngineSpeed"));

        std::fs::remove_file(&dbc_path).ok();
        std::fs::remove_dir_all(&out_dir).ok();
    }

    #[test]
    fn test_parse_can_id_hex_and_decimal() {
        assert_eq!(parse_can_id("0x0CF00400").unwrap(), 0x0CF00400);
        assert_eq!(parse_can_id("100").unwrap(), 100);
        assert!(parse_can_id("not-a-number").is_err());
    }

    #[test]
    fn test_parse_hex_data() {
        assert_eq!(
            parse_hex_data("DEADBEEF").unwrap(),
            vec![0xDE, 0xAD, 0xBE, 0xEF]
        );
        assert!(parse_hex_data("ABC").is_err()); // odd length
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("Engine Data!"), "Engine_Data_");
        assert_eq!(sanitize_filename(""), "Message");
    }
}
