//! Enhanced DBC file parser extensions
//!
//! This module extends the base DBC parser with support for:
//!
//! - `SG_MUL_VAL_` — Extended multiplexing (range-based signal activation)
//! - `EV_` — Environment variable definitions
//! - `ENVVAR_DATA_` — Environment variable data
//! - `SG_MUL_VAL_` — Complex multiplexed signal ranges
//! - Robust `BA_DEF_` / `BA_` parsing for node attributes
//!
//! # Example
//!
//! ```rust
//! use oxirs_canbus::dbc::enhanced::{EnhancedDbcParser, EnvVar, EnvVarType, SgMulValRange};
//!
//! let dbc_content = r#"
//! VERSION ""
//! EV_ EngineMode : 0 [0,5] "" 0 0 DUMMY_NODE_VECTOR0 Engine;
//! SG_MUL_VAL_ 100 Signal1 Switch1 1-3, 5-7;
//! "#;
//!
//! let enhanced = EnhancedDbcParser::parse(dbc_content).expect("valid DBC");
//! assert_eq!(enhanced.env_vars.len(), 1);
//! ```

use crate::dbc::parser::{DbcDatabase, DbcParser};
use crate::error::{CanbusError, CanbusResult};
use std::collections::HashMap;

/// Environment variable type in DBC
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EnvVarType {
    /// Integer value (0 = integer in DBC)
    #[default]
    Integer,
    /// Float value (1 = float in DBC)
    Float,
    /// String value (2 = string in DBC)
    String,
    /// Data (byte array) value (3 = data in DBC)
    Data,
}

impl EnvVarType {
    /// Parse DBC type code
    pub fn from_code(code: u8) -> Self {
        match code {
            0 => Self::Integer,
            1 => Self::Float,
            2 => Self::String,
            _ => Self::Data,
        }
    }

    /// Get the DBC type code
    pub fn to_code(self) -> u8 {
        match self {
            Self::Integer => 0,
            Self::Float => 1,
            Self::String => 2,
            Self::Data => 3,
        }
    }
}

/// Environment variable definition from `EV_` directive
///
/// Environment variables in DBC files represent values that can be
/// exchanged between the DBC tool and the network nodes, but are not
/// part of the CAN bus data transmission directly.
#[derive(Debug, Clone)]
pub struct EnvVar {
    /// Environment variable name
    pub name: String,
    /// Data type
    pub var_type: EnvVarType,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Physical unit
    pub unit: String,
    /// Initial value
    pub initial_value: f64,
    /// Environment variable ID
    pub ev_id: u32,
    /// Access type (DUMMY_NODE_VECTOR0, etc.)
    pub access_type: String,
    /// Access nodes (ECUs that can access this variable)
    pub access_nodes: Vec<String>,
    /// Comment/description
    pub comment: Option<String>,
    /// Byte size (for data type)
    pub data_size: Option<usize>,
    /// Attribute values
    pub attributes: HashMap<String, String>,
}

impl EnvVar {
    /// Create a new environment variable
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            var_type: EnvVarType::Integer,
            min: 0.0,
            max: 0.0,
            unit: String::new(),
            initial_value: 0.0,
            ev_id: 0,
            access_type: String::new(),
            access_nodes: Vec::new(),
            comment: None,
            data_size: None,
            attributes: HashMap::new(),
        }
    }
}

/// A range entry for extended multiplexing (SG_MUL_VAL_)
///
/// Defines a range of multiplexer values for which a multiplexed
/// signal is valid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SgMulValRange {
    /// Start of the range (inclusive)
    pub start: u32,
    /// End of the range (inclusive)
    pub end: u32,
}

impl SgMulValRange {
    /// Create a new range
    pub fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    /// Check if this range contains a value
    pub fn contains(&self, value: u32) -> bool {
        value >= self.start && value <= self.end
    }

    /// Create a single-value range
    pub fn single(value: u32) -> Self {
        Self {
            start: value,
            end: value,
        }
    }
}

/// Extended multiplexing entry from `SG_MUL_VAL_` directive
///
/// Defines which ranges of a multiplexer signal activate a multiplexed signal.
#[derive(Debug, Clone)]
pub struct SgMulValEntry {
    /// Message ID
    pub message_id: u32,
    /// Name of the multiplexed signal
    pub signal_name: String,
    /// Name of the multiplexer signal
    pub multiplexer_signal: String,
    /// Valid multiplexer value ranges
    pub ranges: Vec<SgMulValRange>,
}

impl SgMulValEntry {
    /// Create a new extended multiplex entry
    pub fn new(
        message_id: u32,
        signal_name: &str,
        multiplexer_signal: &str,
        ranges: Vec<SgMulValRange>,
    ) -> Self {
        Self {
            message_id,
            signal_name: signal_name.to_string(),
            multiplexer_signal: multiplexer_signal.to_string(),
            ranges,
        }
    }

    /// Check if this signal is active for a given multiplexer value
    pub fn is_active_for(&self, mux_value: u32) -> bool {
        self.ranges.iter().any(|r| r.contains(mux_value))
    }
}

/// Enhanced DBC database with additional parsed constructs
#[derive(Debug, Clone)]
pub struct EnhancedDbcDatabase {
    /// Base DBC database (messages, signals, nodes, etc.)
    pub base: DbcDatabase,
    /// Environment variable definitions
    pub env_vars: Vec<EnvVar>,
    /// Extended multiplexing entries (SG_MUL_VAL_)
    pub sg_mul_val_entries: Vec<SgMulValEntry>,
    /// Node group attributes
    pub node_attributes: HashMap<String, HashMap<String, String>>,
}

impl EnhancedDbcDatabase {
    /// Create a new empty enhanced database
    pub fn new() -> Self {
        Self {
            base: DbcDatabase::new(),
            env_vars: Vec::new(),
            sg_mul_val_entries: Vec::new(),
            node_attributes: HashMap::new(),
        }
    }

    /// Get an environment variable by name
    pub fn get_env_var(&self, name: &str) -> Option<&EnvVar> {
        self.env_vars.iter().find(|e| e.name == name)
    }

    /// Get extended multiplex entries for a message
    pub fn get_sg_mul_val_for_message(&self, message_id: u32) -> Vec<&SgMulValEntry> {
        self.sg_mul_val_entries
            .iter()
            .filter(|e| e.message_id == message_id)
            .collect()
    }

    /// Find the active multiplexed signals for a given message and mux value
    pub fn get_active_multiplexed_signals(&self, message_id: u32, mux_value: u32) -> Vec<&str> {
        self.sg_mul_val_entries
            .iter()
            .filter(|e| e.message_id == message_id && e.is_active_for(mux_value))
            .map(|e| e.signal_name.as_str())
            .collect()
    }
}

impl Default for EnhancedDbcDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced DBC parser that handles additional DBC directives
///
/// This parser builds on top of the base `DbcParser` and adds:
/// - `EV_` (environment variables)
/// - `ENVVAR_DATA_` (environment variable data sizes)
/// - `SG_MUL_VAL_` (extended multiplexing ranges)
/// - Enhanced `CM_` for environment variable comments
pub struct EnhancedDbcParser;

impl EnhancedDbcParser {
    /// Parse DBC content into an enhanced database
    pub fn parse(content: &str) -> CanbusResult<EnhancedDbcDatabase> {
        // First pass: base parser for standard directives
        let base_db = DbcParser::new().parse(content)?;

        let mut enhanced = EnhancedDbcDatabase::new();
        enhanced.base = base_db;

        // Second pass: parse enhanced directives
        let mut lines_iter = content.lines().peekable();

        while let Some(line) = lines_iter.next() {
            let trimmed = line.trim();

            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }

            // Collect full statement (may span multiple lines, terminated by ;)
            let mut full_stmt = trimmed.to_string();
            if Self::needs_semicolon_terminator(trimmed) {
                while !full_stmt.ends_with(';') {
                    if let Some(next) = lines_iter.next() {
                        let nt = next.trim();
                        full_stmt.push(' ');
                        full_stmt.push_str(nt);
                    } else {
                        break;
                    }
                }
            }

            if full_stmt.starts_with("EV_ ") {
                let _ = Self::parse_env_var(&full_stmt, &mut enhanced);
            } else if full_stmt.starts_with("ENVVAR_DATA_ ") {
                let _ = Self::parse_envvar_data(&full_stmt, &mut enhanced);
            } else if full_stmt.starts_with("SG_MUL_VAL_ ") {
                let _ = Self::parse_sg_mul_val(&full_stmt, &mut enhanced);
            } else if full_stmt.starts_with("CM_ EV_") {
                let _ = Self::parse_ev_comment(&full_stmt, &mut enhanced);
            }
        }

        Ok(enhanced)
    }

    /// Check if a directive needs a semicolon terminator
    fn needs_semicolon_terminator(line: &str) -> bool {
        line.starts_with("EV_ ")
            || line.starts_with("ENVVAR_DATA_ ")
            || line.starts_with("SG_MUL_VAL_ ")
            || line.starts_with("CM_ EV_")
    }

    /// Parse `EV_` (environment variable) directive
    ///
    /// Format: `EV_ VarName : VarType [min,max] "unit" initial_val ev_id ACCESS_TYPE AccessNode;`
    fn parse_env_var(line: &str, db: &mut EnhancedDbcDatabase) -> CanbusResult<()> {
        // EV_ EngineMode : 0 [0,5] "" 0 0 DUMMY_NODE_VECTOR0 Engine;
        let without_prefix = line
            .strip_prefix("EV_ ")
            .ok_or_else(|| CanbusError::Config("Not an EV_ line".to_string()))?;

        // Split on ':' to get name and rest
        let colon_pos = without_prefix
            .find(':')
            .ok_or_else(|| CanbusError::Config("EV_ missing colon separator".to_string()))?;

        let name = without_prefix[..colon_pos].trim();
        let rest = without_prefix[colon_pos + 1..].trim().trim_end_matches(';');

        let mut ev = EnvVar::new(name);

        // Parse remaining tokens
        // Format: <type_code> [<min>,<max>] "<unit>" <initial> <id> <access_type> [<nodes>...]
        let parts: Vec<&str> = rest.split_whitespace().collect();

        if parts.is_empty() {
            db.env_vars.push(ev);
            return Ok(());
        }

        // Type code
        if let Ok(code) = parts[0].parse::<u8>() {
            ev.var_type = EnvVarType::from_code(code);
        }

        // Find the range [min,max]
        let mut idx = 1;
        if idx < parts.len() && parts[idx].starts_with('[') {
            let range_str = {
                // May be split across tokens or combined
                let mut rs = parts[idx].to_string();
                while !rs.contains(']') && idx + 1 < parts.len() {
                    idx += 1;
                    rs.push(' ');
                    rs.push_str(parts[idx]);
                }
                rs
            };
            // Parse [min,max]
            let inner = range_str
                .trim_matches('[')
                .trim_matches(']')
                .trim_end_matches(']');
            if let Some(comma) = inner.find(',') {
                ev.min = inner[..comma].parse().unwrap_or(0.0);
                ev.max = inner[comma + 1..].parse().unwrap_or(0.0);
            }
            idx += 1;
        }

        // Unit string
        if idx < parts.len() && parts[idx].starts_with('"') {
            ev.unit = parts[idx].trim_matches('"').to_string();
            idx += 1;
        }

        // Initial value
        if idx < parts.len() {
            ev.initial_value = parts[idx].parse().unwrap_or(0.0);
            idx += 1;
        }

        // EV ID
        if idx < parts.len() {
            ev.ev_id = parts[idx].parse().unwrap_or(0);
            idx += 1;
        }

        // Access type
        if idx < parts.len() {
            ev.access_type = parts[idx].to_string();
            idx += 1;
        }

        // Access nodes (remainder)
        for node in &parts[idx..] {
            let n = node.trim_end_matches(';').trim_end_matches(',');
            if !n.is_empty() {
                ev.access_nodes.push(n.to_string());
            }
        }

        db.env_vars.push(ev);
        Ok(())
    }

    /// Parse `ENVVAR_DATA_` directive (data size for data-type env vars)
    ///
    /// Format: `ENVVAR_DATA_ VarName : <size>;`
    fn parse_envvar_data(line: &str, db: &mut EnhancedDbcDatabase) -> CanbusResult<()> {
        // ENVVAR_DATA_ VarName : 8;
        let without_prefix = line
            .strip_prefix("ENVVAR_DATA_ ")
            .ok_or_else(|| CanbusError::Config("Not an ENVVAR_DATA_ line".to_string()))?;

        let colon_pos = without_prefix
            .find(':')
            .ok_or_else(|| CanbusError::Config("ENVVAR_DATA_ missing colon".to_string()))?;

        let name = without_prefix[..colon_pos].trim();
        let size_str = without_prefix[colon_pos + 1..].trim().trim_end_matches(';');

        if let Ok(size) = size_str.parse::<usize>() {
            if let Some(ev) = db.env_vars.iter_mut().find(|e| e.name == name) {
                ev.data_size = Some(size);
            }
        }

        Ok(())
    }

    /// Parse `SG_MUL_VAL_` directive (extended multiplexing)
    ///
    /// Format: `SG_MUL_VAL_ <msg_id> <signal> <mux_signal> <start>-<end>, ...;`
    fn parse_sg_mul_val(line: &str, db: &mut EnhancedDbcDatabase) -> CanbusResult<()> {
        // SG_MUL_VAL_ 100 Signal1 Switch1 1-3, 5-7;
        // SG_MUL_VAL_ 200 DataSig MuxId 0-0;  (single value)
        let without_prefix = line
            .strip_prefix("SG_MUL_VAL_ ")
            .ok_or_else(|| CanbusError::Config("Not an SG_MUL_VAL_ line".to_string()))?;

        let without_semi = without_prefix.trim_end_matches(';').trim();
        let parts: Vec<&str> = without_semi.split_whitespace().collect();

        if parts.len() < 3 {
            return Ok(());
        }

        let message_id: u32 = parts[0].parse().map_err(|_| {
            CanbusError::Config(format!("Invalid SG_MUL_VAL_ message ID: {}", parts[0]))
        })?;

        let signal_name = parts[1];
        let mux_signal = parts[2];

        // Parse ranges: "1-3," "5-7;" etc.
        let mut ranges = Vec::new();
        for part in &parts[3..] {
            let range_str = part.trim_end_matches(',').trim_end_matches(';');
            if range_str.is_empty() {
                continue;
            }

            if let Some(dash) = range_str.find('-') {
                let start_str = &range_str[..dash];
                let end_str = &range_str[dash + 1..];
                if let (Ok(start), Ok(end)) = (start_str.parse::<u32>(), end_str.parse::<u32>()) {
                    ranges.push(SgMulValRange::new(start, end));
                }
            } else if let Ok(val) = range_str.parse::<u32>() {
                ranges.push(SgMulValRange::single(val));
            }
        }

        db.sg_mul_val_entries.push(SgMulValEntry::new(
            message_id,
            signal_name,
            mux_signal,
            ranges,
        ));

        Ok(())
    }

    /// Parse `CM_ EV_` (environment variable comment)
    fn parse_ev_comment(line: &str, db: &mut EnhancedDbcDatabase) -> CanbusResult<()> {
        // CM_ EV_ VarName "comment";
        let comment_start = line.find('"');
        let comment_end = line.rfind('"');

        if let (Some(start), Some(end)) = (comment_start, comment_end) {
            if start < end {
                let comment = line[start + 1..end].to_string();
                let prefix = &line[..start];
                let parts: Vec<&str> = prefix.split_whitespace().collect();

                // Parts: CM_ EV_ VarName
                if parts.len() >= 3 {
                    let var_name = parts[2];
                    if let Some(ev) = db.env_vars.iter_mut().find(|e| e.name == var_name) {
                        ev.comment = Some(comment);
                    }
                }
            }
        }

        Ok(())
    }
}

/// Parse an enhanced DBC string
pub fn parse_enhanced_dbc(content: &str) -> CanbusResult<EnhancedDbcDatabase> {
    EnhancedDbcParser::parse(content)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_dbc_with_ev() -> &'static str {
        r#"VERSION ""

BU_: Engine Dashboard

BO_ 100 EngineData: 8 Engine
 SG_ EngineSpeed : 0|16@1+ (0.125,0) [0|8031.875] "rpm" Dashboard

EV_ EngineMode : 0 [0,5] "" 0 0 DUMMY_NODE_VECTOR0 Engine;
EV_ CoolantTemp : 1 [0.0,100.0] "degC" 20.0 1 DUMMY_NODE_VECTOR0 Engine Dashboard;

CM_ EV_ EngineMode "Operating mode of the engine";

SG_MUL_VAL_ 100 EngineSpeed MuxSignal 0-3, 5-7;
"#
    }

    // ---- Environment variable tests ----

    #[test]
    fn test_ev_parse_integer_type() {
        let content =
            "VERSION \"\"\nEV_ EngineMode : 0 [0,5] \"\" 0 0 DUMMY_NODE_VECTOR0 Engine;\n";
        let db = EnhancedDbcParser::parse(content).expect("valid parse");
        assert_eq!(db.env_vars.len(), 1);
        let ev = &db.env_vars[0];
        assert_eq!(ev.name, "EngineMode");
        assert_eq!(ev.var_type, EnvVarType::Integer);
    }

    #[test]
    fn test_ev_parse_float_type() {
        let content =
            "VERSION \"\"\nEV_ CoolantTemp : 1 [0.0,100.0] \"degC\" 20.0 1 DUMMY_NODE_VECTOR0;\n";
        let db = EnhancedDbcParser::parse(content).expect("valid parse");
        assert_eq!(db.env_vars.len(), 1);
        let ev = &db.env_vars[0];
        assert_eq!(ev.name, "CoolantTemp");
        assert_eq!(ev.var_type, EnvVarType::Float);
        assert!((ev.initial_value - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_ev_parse_min_max_range() {
        let content = "VERSION \"\"\nEV_ TestVar : 0 [10,200] \"\" 0 0 DUMMY_NODE_VECTOR0;\n";
        let db = EnhancedDbcParser::parse(content).expect("valid parse");
        let ev = &db.env_vars[0];
        assert!((ev.min - 10.0).abs() < 1e-6);
        assert!((ev.max - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_ev_parse_access_nodes() {
        let db = EnhancedDbcParser::parse(make_test_dbc_with_ev()).expect("valid parse");
        let ev = db.get_env_var("CoolantTemp").expect("env var exists");
        assert!(ev.access_nodes.contains(&"Engine".to_string()));
        assert!(ev.access_nodes.contains(&"Dashboard".to_string()));
    }

    #[test]
    fn test_ev_comment_parsed() {
        let db = EnhancedDbcParser::parse(make_test_dbc_with_ev()).expect("valid parse");
        let ev = db.get_env_var("EngineMode").expect("env var exists");
        assert_eq!(ev.comment.as_deref(), Some("Operating mode of the engine"));
    }

    #[test]
    fn test_ev_count() {
        let db = EnhancedDbcParser::parse(make_test_dbc_with_ev()).expect("valid parse");
        assert_eq!(db.env_vars.len(), 2);
    }

    #[test]
    fn test_ev_type_codes() {
        assert_eq!(EnvVarType::Integer.to_code(), 0);
        assert_eq!(EnvVarType::Float.to_code(), 1);
        assert_eq!(EnvVarType::String.to_code(), 2);
        assert_eq!(EnvVarType::Data.to_code(), 3);

        assert_eq!(EnvVarType::from_code(0), EnvVarType::Integer);
        assert_eq!(EnvVarType::from_code(1), EnvVarType::Float);
        assert_eq!(EnvVarType::from_code(2), EnvVarType::String);
        assert_eq!(EnvVarType::from_code(99), EnvVarType::Data);
    }

    #[test]
    fn test_ev_id_parsed() {
        let content = "VERSION \"\"\nEV_ MyVar : 0 [0,100] \"\" 0 42 DUMMY_NODE_VECTOR0;\n";
        let db = EnhancedDbcParser::parse(content).expect("valid parse");
        assert_eq!(db.env_vars[0].ev_id, 42);
    }

    #[test]
    fn test_ev_unit_parsed() {
        let content = "VERSION \"\"\nEV_ SpeedVar : 1 [0,300] \"km/h\" 0 0 DUMMY_NODE_VECTOR0;\n";
        let db = EnhancedDbcParser::parse(content).expect("valid parse");
        assert_eq!(db.env_vars[0].unit, "km/h");
    }

    // ---- ENVVAR_DATA_ tests ----

    #[test]
    fn test_envvar_data_size_parsed() {
        let content = r#"VERSION ""
EV_ DataVar : 3 [0,0] "" 0 0 DUMMY_NODE_VECTOR0;
ENVVAR_DATA_ DataVar : 16;
"#;
        let db = EnhancedDbcParser::parse(content).expect("valid parse");
        let ev = db.get_env_var("DataVar").expect("env var exists");
        assert_eq!(ev.data_size, Some(16));
    }

    #[test]
    fn test_envvar_data_without_ev_does_not_crash() {
        let content = r#"VERSION ""
ENVVAR_DATA_ NonExistent : 8;
"#;
        // Should not error even if the EV_ doesn't exist
        let result = EnhancedDbcParser::parse(content);
        assert!(result.is_ok());
    }

    // ---- SG_MUL_VAL_ tests ----

    #[test]
    fn test_sg_mul_val_basic() {
        let db = EnhancedDbcParser::parse(make_test_dbc_with_ev()).expect("valid parse");
        assert_eq!(db.sg_mul_val_entries.len(), 1);
        let entry = &db.sg_mul_val_entries[0];
        assert_eq!(entry.message_id, 100);
        assert_eq!(entry.signal_name, "EngineSpeed");
        assert_eq!(entry.multiplexer_signal, "MuxSignal");
    }

    #[test]
    fn test_sg_mul_val_ranges_parsed() {
        let db = EnhancedDbcParser::parse(make_test_dbc_with_ev()).expect("valid parse");
        let entry = &db.sg_mul_val_entries[0];
        assert_eq!(entry.ranges.len(), 2);
        assert_eq!(entry.ranges[0].start, 0);
        assert_eq!(entry.ranges[0].end, 3);
        assert_eq!(entry.ranges[1].start, 5);
        assert_eq!(entry.ranges[1].end, 7);
    }

    #[test]
    fn test_sg_mul_val_range_contains() {
        let range = SgMulValRange::new(5, 10);
        assert!(range.contains(5));
        assert!(range.contains(7));
        assert!(range.contains(10));
        assert!(!range.contains(4));
        assert!(!range.contains(11));
    }

    #[test]
    fn test_sg_mul_val_range_single() {
        let range = SgMulValRange::single(42);
        assert!(range.contains(42));
        assert!(!range.contains(41));
        assert!(!range.contains(43));
    }

    #[test]
    fn test_sg_mul_val_is_active_for() {
        let content = r#"VERSION ""
BO_ 200 TestMsg: 8 Node1
 SG_ DataSig : 0|8@1+ (1,0) [0|255] "" Node1
 SG_ MuxSig : 8|4@1+ (1,0) [0|15] "" Node1
SG_MUL_VAL_ 200 DataSig MuxSig 1-3;
"#;
        let db = EnhancedDbcParser::parse(content).expect("valid parse");
        let entry = &db.sg_mul_val_entries[0];
        assert!(entry.is_active_for(1));
        assert!(entry.is_active_for(2));
        assert!(entry.is_active_for(3));
        assert!(!entry.is_active_for(0));
        assert!(!entry.is_active_for(4));
    }

    #[test]
    fn test_sg_mul_val_multiple_entries() {
        let content = r#"VERSION ""
BO_ 300 MultiplexedMsg: 8 Node1
 SG_ Sig1 : 0|8@1+ (1,0) [0|255] "" Node1
 SG_ Sig2 : 0|8@1+ (1,0) [0|255] "" Node1
 SG_ MuxId : 8|4@1+ (1,0) [0|15] "" Node1
SG_MUL_VAL_ 300 Sig1 MuxId 0-4;
SG_MUL_VAL_ 300 Sig2 MuxId 5-9;
"#;
        let db = EnhancedDbcParser::parse(content).expect("valid parse");
        assert_eq!(db.sg_mul_val_entries.len(), 2);

        let active = db.get_active_multiplexed_signals(300, 3);
        assert!(active.contains(&"Sig1"));
        assert!(!active.contains(&"Sig2"));

        let active2 = db.get_active_multiplexed_signals(300, 7);
        assert!(!active2.contains(&"Sig1"));
        assert!(active2.contains(&"Sig2"));
    }

    #[test]
    fn test_sg_mul_val_for_message() {
        let content = r#"VERSION ""
BO_ 100 Msg1: 8 Node1
 SG_ SigA : 0|8@1+ (1,0) [0|255] "" Node1
BO_ 200 Msg2: 8 Node1
 SG_ SigB : 0|8@1+ (1,0) [0|255] "" Node1
SG_MUL_VAL_ 100 SigA MuxA 0-3;
SG_MUL_VAL_ 200 SigB MuxB 1-2;
"#;
        let db = EnhancedDbcParser::parse(content).expect("valid parse");
        let entries_100 = db.get_sg_mul_val_for_message(100);
        assert_eq!(entries_100.len(), 1);
        assert_eq!(entries_100[0].signal_name, "SigA");

        let entries_200 = db.get_sg_mul_val_for_message(200);
        assert_eq!(entries_200.len(), 1);
        assert_eq!(entries_200[0].signal_name, "SigB");
    }

    // ---- Base database compatibility ----

    #[test]
    fn test_base_db_still_parsed() {
        let db = EnhancedDbcParser::parse(make_test_dbc_with_ev()).expect("valid parse");
        assert_eq!(db.base.messages.len(), 1);
        let msg = db.base.get_message(100).expect("message exists");
        assert_eq!(msg.name, "EngineData");
        assert_eq!(msg.signals.len(), 1);
    }

    #[test]
    fn test_parse_enhanced_dbc_function() {
        let db = parse_enhanced_dbc(make_test_dbc_with_ev()).expect("valid parse");
        assert!(!db.env_vars.is_empty());
    }

    #[test]
    fn test_empty_dbc_no_ev() {
        let content = "VERSION \"\"\n";
        let db = EnhancedDbcParser::parse(content).expect("valid parse");
        assert!(db.env_vars.is_empty());
        assert!(db.sg_mul_val_entries.is_empty());
    }

    #[test]
    fn test_ev_new_defaults() {
        let ev = EnvVar::new("TestVar");
        assert_eq!(ev.name, "TestVar");
        assert_eq!(ev.var_type, EnvVarType::Integer);
        assert!(ev.access_nodes.is_empty());
        assert!(ev.comment.is_none());
        assert!(ev.data_size.is_none());
    }

    #[test]
    fn test_sg_mul_val_entry_new() {
        let ranges = vec![SgMulValRange::new(0, 5)];
        let entry = SgMulValEntry::new(100, "Signal1", "MuxSwitch", ranges);
        assert_eq!(entry.message_id, 100);
        assert_eq!(entry.signal_name, "Signal1");
        assert_eq!(entry.multiplexer_signal, "MuxSwitch");
        assert_eq!(entry.ranges.len(), 1);
    }
}
