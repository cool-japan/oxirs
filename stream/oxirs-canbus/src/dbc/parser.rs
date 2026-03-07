//! DBC (Vector CANdb++) file parser
//!
//! Parses DBC files which define CAN message and signal definitions.
//! The DBC format is the industry standard for CAN database files.
//!
//! # Supported Directives
//!
//! - `VERSION` - Database version string
//! - `NS_` - New symbols section
//! - `BS_` - Bus speed section
//! - `BU_` - Node/ECU definitions
//! - `BO_` - Message definitions
//! - `SG_` - Signal definitions (within messages)
//! - `CM_` - Comments for messages/signals
//! - `BA_DEF_` - Attribute definitions
//! - `BA_` - Attribute values
//! - `VAL_` - Value descriptions (enumerations)
//! - `VAL_TABLE_` - Standalone value tables

use crate::error::{CanbusError, CanbusResult};
use std::collections::HashMap;

/// Byte order for signal extraction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ByteOrder {
    /// Little-endian (Intel byte order) - LSB first
    #[default]
    LittleEndian,
    /// Big-endian (Motorola byte order) - MSB first
    BigEndian,
}

/// Value type for signals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValueType {
    /// Unsigned integer
    #[default]
    Unsigned,
    /// Signed integer
    Signed,
}

/// Multiplexer type for signals
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum MultiplexerType {
    /// Not multiplexed
    #[default]
    None,
    /// Multiplexer signal (switch)
    Multiplexer,
    /// Multiplexed signal with value
    Multiplexed(u32),
}

/// Signal definition from DBC file
#[derive(Debug, Clone)]
pub struct DbcSignal {
    /// Signal name
    pub name: String,
    /// Start bit position
    pub start_bit: u32,
    /// Signal length in bits
    pub bit_length: u32,
    /// Byte order (Intel/Motorola)
    pub byte_order: ByteOrder,
    /// Value type (signed/unsigned)
    pub value_type: ValueType,
    /// Scale factor for physical value calculation
    pub factor: f64,
    /// Offset for physical value calculation
    pub offset: f64,
    /// Minimum physical value
    pub min: f64,
    /// Maximum physical value
    pub max: f64,
    /// Unit string
    pub unit: String,
    /// Receiving nodes
    pub receivers: Vec<String>,
    /// Multiplexer information
    pub multiplexer: MultiplexerType,
    /// Value descriptions (enum mappings)
    pub value_descriptions: HashMap<i64, String>,
    /// Comment/description
    pub comment: Option<String>,
}

impl DbcSignal {
    /// Create a new signal with default values
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            start_bit: 0,
            bit_length: 1,
            byte_order: ByteOrder::default(),
            value_type: ValueType::default(),
            factor: 1.0,
            offset: 0.0,
            min: 0.0,
            max: 0.0,
            unit: String::new(),
            receivers: Vec::new(),
            multiplexer: MultiplexerType::default(),
            value_descriptions: HashMap::new(),
            comment: None,
        }
    }

    /// Calculate physical value from raw integer
    pub fn to_physical(&self, raw: i64) -> f64 {
        raw as f64 * self.factor + self.offset
    }

    /// Calculate raw value from physical value
    pub fn to_raw(&self, physical: f64) -> i64 {
        ((physical - self.offset) / self.factor).round() as i64
    }

    /// Get value description for raw value (enum lookup)
    pub fn get_value_description(&self, raw: i64) -> Option<&str> {
        self.value_descriptions.get(&raw).map(|s| s.as_str())
    }
}

/// Message definition from DBC file
#[derive(Debug, Clone)]
pub struct DbcMessage {
    /// CAN message ID (without extended flag)
    pub id: u32,
    /// Whether this is an extended (29-bit) CAN ID
    pub is_extended: bool,
    /// Message name
    pub name: String,
    /// Data Length Code (bytes)
    pub dlc: u8,
    /// Transmitting node name
    pub transmitter: String,
    /// Signals contained in this message
    pub signals: Vec<DbcSignal>,
    /// Comment/description
    pub comment: Option<String>,
    /// Custom attributes
    pub attributes: HashMap<String, AttributeValue>,
}

impl DbcMessage {
    /// Create a new message
    pub fn new(id: u32, name: &str, dlc: u8) -> Self {
        Self {
            id,
            is_extended: id > 0x7FF, // Heuristic: ID > 11 bits is extended
            name: name.to_string(),
            dlc,
            transmitter: String::new(),
            signals: Vec::new(),
            comment: None,
            attributes: HashMap::new(),
        }
    }

    /// Get signal by name
    pub fn get_signal(&self, name: &str) -> Option<&DbcSignal> {
        self.signals.iter().find(|s| s.name == name)
    }

    /// Get mutable signal by name
    pub fn get_signal_mut(&mut self, name: &str) -> Option<&mut DbcSignal> {
        self.signals.iter_mut().find(|s| s.name == name)
    }
}

/// Node (ECU) definition
#[derive(Debug, Clone)]
pub struct DbcNode {
    /// Node name
    pub name: String,
    /// Comment/description
    pub comment: Option<String>,
    /// Custom attributes
    pub attributes: HashMap<String, AttributeValue>,
}

impl DbcNode {
    /// Create a new node
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            comment: None,
            attributes: HashMap::new(),
        }
    }
}

/// Attribute value types
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    /// Integer value
    Int(i64),
    /// Floating point value
    Float(f64),
    /// String value
    String(String),
    /// Enumeration value
    Enum(String),
}

/// Attribute definition
#[derive(Debug, Clone)]
pub struct AttributeDefinition {
    /// Attribute name
    pub name: String,
    /// Object type (message, signal, node, or database)
    pub object_type: AttributeObjectType,
    /// Value type and constraints
    pub value_type: AttributeValueType,
    /// Default value
    pub default_value: Option<AttributeValue>,
}

/// Attribute object type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeObjectType {
    /// Database-level attribute
    Database,
    /// Node attribute
    Node,
    /// Message attribute
    Message,
    /// Signal attribute
    Signal,
}

/// Attribute value type with constraints
#[derive(Debug, Clone)]
pub enum AttributeValueType {
    /// Integer with min/max constraints
    Int {
        /// Minimum allowed value
        min: i64,
        /// Maximum allowed value
        max: i64,
    },
    /// Float with min/max constraints
    Float {
        /// Minimum allowed value
        min: f64,
        /// Maximum allowed value
        max: f64,
    },
    /// String value (no constraints)
    String,
    /// Enumeration with allowed values
    Enum {
        /// List of allowed enumeration values
        values: Vec<String>,
    },
}

/// Complete DBC database
#[derive(Debug, Clone)]
pub struct DbcDatabase {
    /// Version string
    pub version: String,
    /// Nodes/ECUs
    pub nodes: Vec<DbcNode>,
    /// Messages
    pub messages: Vec<DbcMessage>,
    /// Attribute definitions
    pub attribute_definitions: Vec<AttributeDefinition>,
    /// Value tables (standalone enumerations)
    pub value_tables: HashMap<String, HashMap<i64, String>>,
    /// Database-level attributes
    pub attributes: HashMap<String, AttributeValue>,
}

impl DbcDatabase {
    /// Create an empty database
    pub fn new() -> Self {
        Self {
            version: String::new(),
            nodes: Vec::new(),
            messages: Vec::new(),
            attribute_definitions: Vec::new(),
            value_tables: HashMap::new(),
            attributes: HashMap::new(),
        }
    }

    /// Get message by ID
    pub fn get_message(&self, id: u32) -> Option<&DbcMessage> {
        self.messages.iter().find(|m| m.id == id)
    }

    /// Get message by name
    pub fn get_message_by_name(&self, name: &str) -> Option<&DbcMessage> {
        self.messages.iter().find(|m| m.name == name)
    }

    /// Get node by name
    pub fn get_node(&self, name: &str) -> Option<&DbcNode> {
        self.nodes.iter().find(|n| n.name == name)
    }

    /// Get all signal names across all messages
    pub fn all_signals(&self) -> impl Iterator<Item = (&DbcMessage, &DbcSignal)> {
        self.messages
            .iter()
            .flat_map(|m| m.signals.iter().map(move |s| (m, s)))
    }
}

impl Default for DbcDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// DBC Parser
pub struct DbcParser {
    /// Current line number for error reporting
    line_number: usize,
    /// Current database being built
    database: DbcDatabase,
}

impl DbcParser {
    /// Create a new parser
    pub fn new() -> Self {
        Self {
            line_number: 0,
            database: DbcDatabase::new(),
        }
    }

    /// Parse a DBC file from string content
    pub fn parse(&mut self, content: &str) -> CanbusResult<DbcDatabase> {
        self.database = DbcDatabase::new();
        self.line_number = 0;

        let mut lines = content.lines().peekable();

        while let Some(line) = lines.next() {
            self.line_number += 1;
            let trimmed = line.trim();

            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }

            // Check if this directive needs multiline handling
            let needs_semicolon = Self::directive_needs_semicolon(trimmed);

            let mut full_line = trimmed.to_string();

            if needs_semicolon {
                // Collect continuation lines until we find a semicolon
                while !full_line.ends_with(';') {
                    if let Some(next) = lines.next() {
                        self.line_number += 1;
                        let next_trimmed = next.trim();
                        full_line.push(' ');
                        full_line.push_str(next_trimmed);
                    } else {
                        break;
                    }
                }
            }

            self.parse_line(&full_line)?;
        }

        Ok(std::mem::take(&mut self.database))
    }

    /// Check if a directive requires a semicolon terminator
    fn directive_needs_semicolon(line: &str) -> bool {
        // Directives that can span multiple lines and end with semicolon
        // Note: BO_ and SG_ do NOT end with semicolons in DBC format
        line.starts_with("CM_ ")
            || line.starts_with("BA_DEF_ ")
            || line.starts_with("BA_DEF_DEF_ ")
            || line.starts_with("BA_ ")
            || line.starts_with("VAL_TABLE_ ")
            || line.starts_with("VAL_ ")
    }

    /// Parse a single line/directive
    fn parse_line(&mut self, line: &str) -> CanbusResult<()> {
        // Trim the line to handle indented signals
        let trimmed = line.trim();

        if trimmed.starts_with("VERSION") {
            self.parse_version(trimmed)
        } else if trimmed.starts_with("NS_") {
            // New symbols - ignored
            Ok(())
        } else if trimmed.starts_with("BS_") {
            // Bus speed - ignored
            Ok(())
        } else if trimmed.starts_with("BU_") {
            self.parse_nodes(trimmed)
        } else if trimmed.starts_with("BO_ ") {
            self.parse_message(trimmed)
        } else if trimmed.starts_with("SG_ ") {
            self.parse_standalone_signal(trimmed)
        } else if trimmed.starts_with("CM_ ") {
            self.parse_comment(line) // Keep original for semicolon-terminated lines
        } else if trimmed.starts_with("BA_DEF_ ") {
            self.parse_attribute_definition(line)
        } else if trimmed.starts_with("BA_DEF_DEF_ ") {
            self.parse_attribute_default(line)
        } else if trimmed.starts_with("BA_ ") {
            self.parse_attribute_value(line)
        } else if trimmed.starts_with("VAL_TABLE_ ") {
            self.parse_value_table(line)
        } else if trimmed.starts_with("VAL_ ") {
            self.parse_value_descriptions(line)
        } else {
            // Unknown directive - skip
            Ok(())
        }
    }

    /// Parse VERSION directive
    fn parse_version(&mut self, line: &str) -> CanbusResult<()> {
        // VERSION "1.0"
        if let Some(start) = line.find('"') {
            if let Some(end) = line[start + 1..].find('"') {
                self.database.version = line[start + 1..start + 1 + end].to_string();
            }
        }
        Ok(())
    }

    /// Parse BU_ (nodes) directive
    fn parse_nodes(&mut self, line: &str) -> CanbusResult<()> {
        // BU_: Node1 Node2 Node3
        let content = line.strip_prefix("BU_:").unwrap_or("").trim();
        for name in content.split_whitespace() {
            if !name.is_empty() {
                self.database.nodes.push(DbcNode::new(name));
            }
        }
        Ok(())
    }

    /// Parse BO_ (message) directive
    fn parse_message(&mut self, line: &str) -> CanbusResult<()> {
        // BO_ 2024 EngineData: 8 Vector__XXX
        //  SG_ EngineSpeed : 0|16@1+ (0.125,0) [0|8031.875] "rpm" Vector__XXX
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(self.error("Invalid message format"));
        }

        let id = parts[1]
            .parse::<u32>()
            .map_err(|_| self.error("Invalid message ID"))?;

        let name = parts[2].trim_end_matches(':');
        let dlc = parts[3]
            .parse::<u8>()
            .map_err(|_| self.error("Invalid DLC"))?;

        let mut message = DbcMessage::new(id, name, dlc);

        if parts.len() > 4 {
            message.transmitter = parts[4].to_string();
        }

        // Parse embedded signals
        let signal_start = line.find("SG_");
        if let Some(pos) = signal_start {
            let signal_part = &line[pos..];
            if let Ok(signal) = self.parse_signal_definition(signal_part) {
                message.signals.push(signal);
            }
        }

        self.database.messages.push(message);
        Ok(())
    }

    /// Parse SG_ (signal) that appears on its own line
    fn parse_standalone_signal(&mut self, line: &str) -> CanbusResult<()> {
        if let Ok(signal) = self.parse_signal_definition(line) {
            // Add to the last message
            if let Some(msg) = self.database.messages.last_mut() {
                msg.signals.push(signal);
            }
        }
        Ok(())
    }

    /// Parse signal definition
    fn parse_signal_definition(&self, line: &str) -> CanbusResult<DbcSignal> {
        // SG_ EngineSpeed : 0|16@1+ (0.125,0) [0|8031.875] "rpm" Vector__XXX
        // SG_ SignalName M : 0|8@1+ (1,0) [0|0] "" Vector__XXX  (multiplexer)
        // SG_ SignalName m5 : 0|8@1+ (1,0) [0|0] "" Vector__XXX  (multiplexed value 5)

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 7 {
            return Err(self.error("Invalid signal format"));
        }

        let name = parts[1];
        let mut signal = DbcSignal::new(name);

        // Check for multiplexer indicator
        let mut bit_spec_idx = 3;
        if parts[2] == "M" || parts[2] == "m" {
            signal.multiplexer = MultiplexerType::Multiplexer;
            bit_spec_idx = 4;
        } else if parts[2].starts_with('m') && parts[2].len() > 1 {
            if let Ok(val) = parts[2][1..].parse::<u32>() {
                signal.multiplexer = MultiplexerType::Multiplexed(val);
            }
            bit_spec_idx = 4;
        } else if parts[2] != ":" {
            // Skip if format doesn't match
            bit_spec_idx = 3;
        }

        // Parse bit specification: start|length@byte_order+/-
        // e.g., "0|16@1+"
        let bit_spec = parts
            .get(bit_spec_idx)
            .ok_or_else(|| self.error("Missing bit spec"))?;
        self.parse_bit_spec(bit_spec, &mut signal)?;

        // Parse factor and offset: (factor,offset)
        for part in &parts[bit_spec_idx + 1..] {
            if part.starts_with('(') && part.contains(',') {
                self.parse_factor_offset(part, &mut signal)?;
            } else if part.starts_with('[') && part.contains('|') {
                self.parse_min_max(part, &mut signal)?;
            } else if part.starts_with('"') {
                signal.unit = part.trim_matches('"').to_string();
            } else if !part.starts_with('[') && !part.starts_with('(') && !part.starts_with('"') {
                // Receiver nodes
                let receiver = part.trim_matches(',');
                if !receiver.is_empty() && receiver != "Vector__XXX" {
                    signal.receivers.push(receiver.to_string());
                }
            }
        }

        Ok(signal)
    }

    /// Parse bit specification (start|length@byte_order_sign)
    fn parse_bit_spec(&self, spec: &str, signal: &mut DbcSignal) -> CanbusResult<()> {
        // Format: start|length@byte_order+/-
        // e.g., "0|16@1+" or "7|8@0-"

        let at_pos = spec
            .find('@')
            .ok_or_else(|| self.error("Invalid bit spec: missing @"))?;
        let pipe_pos = spec
            .find('|')
            .ok_or_else(|| self.error("Invalid bit spec: missing |"))?;

        signal.start_bit = spec[..pipe_pos]
            .parse()
            .map_err(|_| self.error("Invalid start bit"))?;

        signal.bit_length = spec[pipe_pos + 1..at_pos]
            .parse()
            .map_err(|_| self.error("Invalid bit length"))?;

        let order_sign = &spec[at_pos + 1..];
        signal.byte_order = if order_sign.starts_with('1') {
            ByteOrder::LittleEndian
        } else {
            ByteOrder::BigEndian
        };

        signal.value_type = if order_sign.ends_with('-') {
            ValueType::Signed
        } else {
            ValueType::Unsigned
        };

        Ok(())
    }

    /// Parse factor and offset: (factor,offset)
    fn parse_factor_offset(&self, spec: &str, signal: &mut DbcSignal) -> CanbusResult<()> {
        let inner = spec.trim_matches(|c| c == '(' || c == ')');
        let parts: Vec<&str> = inner.split(',').collect();
        if parts.len() == 2 {
            signal.factor = parts[0].parse().unwrap_or(1.0);
            signal.offset = parts[1].parse().unwrap_or(0.0);
        }
        Ok(())
    }

    /// Parse min and max: [min|max]
    fn parse_min_max(&self, spec: &str, signal: &mut DbcSignal) -> CanbusResult<()> {
        let inner = spec.trim_matches(|c| c == '[' || c == ']');
        let parts: Vec<&str> = inner.split('|').collect();
        if parts.len() == 2 {
            signal.min = parts[0].parse().unwrap_or(0.0);
            signal.max = parts[1].parse().unwrap_or(0.0);
        }
        Ok(())
    }

    /// Parse CM_ (comment) directive
    fn parse_comment(&mut self, line: &str) -> CanbusResult<()> {
        // CM_ "Database comment";
        // CM_ SG_ 2024 EngineSpeed "Engine speed in RPM";
        // CM_ BO_ 2024 "Engine data message";
        // CM_ BU_ ECU1 "Engine control unit";

        let comment_start = line.find('"');
        let comment_end = line.rfind('"');

        if let (Some(start), Some(end)) = (comment_start, comment_end) {
            if start < end {
                let comment = line[start + 1..end].to_string();

                if line.starts_with("CM_ SG_") {
                    // Signal comment
                    let parts: Vec<&str> = line[7..start].split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(msg_id) = parts[0].parse::<u32>() {
                            let sig_name = parts[1];
                            if let Some(msg) =
                                self.database.messages.iter_mut().find(|m| m.id == msg_id)
                            {
                                if let Some(sig) =
                                    msg.signals.iter_mut().find(|s| s.name == sig_name)
                                {
                                    sig.comment = Some(comment);
                                }
                            }
                        }
                    }
                } else if line.starts_with("CM_ BO_") {
                    // Message comment
                    let parts: Vec<&str> = line[7..start].split_whitespace().collect();
                    if !parts.is_empty() {
                        if let Ok(msg_id) = parts[0].parse::<u32>() {
                            if let Some(msg) =
                                self.database.messages.iter_mut().find(|m| m.id == msg_id)
                            {
                                msg.comment = Some(comment);
                            }
                        }
                    }
                } else if line.starts_with("CM_ BU_") {
                    // Node comment
                    let parts: Vec<&str> = line[7..start].split_whitespace().collect();
                    if !parts.is_empty() {
                        let node_name = parts[0];
                        if let Some(node) =
                            self.database.nodes.iter_mut().find(|n| n.name == node_name)
                        {
                            node.comment = Some(comment);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Parse BA_DEF_ (attribute definition)
    fn parse_attribute_definition(&mut self, line: &str) -> CanbusResult<()> {
        // BA_DEF_ BO_ "AttributeName" INT 0 100;
        // BA_DEF_ SG_ "AttributeName" FLOAT 0.0 100.0;
        // BA_DEF_ "DatabaseAttr" STRING;
        // BA_DEF_ BO_ "GenMsgCycleTime" INT 0 10000;

        let object_type = if line.contains(" BO_ ") {
            AttributeObjectType::Message
        } else if line.contains(" SG_ ") {
            AttributeObjectType::Signal
        } else if line.contains(" BU_ ") {
            AttributeObjectType::Node
        } else {
            AttributeObjectType::Database
        };

        // Extract attribute name
        let name_start = line.find('"');
        let name_end = line[name_start.unwrap_or(0) + 1..].find('"');

        if let (Some(start), Some(end)) = (name_start, name_end) {
            let name = line[start + 1..start + 1 + end].to_string();
            let remaining = &line[start + 2 + end..];

            let value_type = self.parse_attribute_value_type(remaining)?;

            self.database
                .attribute_definitions
                .push(AttributeDefinition {
                    name,
                    object_type,
                    value_type,
                    default_value: None,
                });
        }

        Ok(())
    }

    /// Parse attribute value type
    fn parse_attribute_value_type(&self, spec: &str) -> CanbusResult<AttributeValueType> {
        let parts: Vec<&str> = spec.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(AttributeValueType::String);
        }

        match parts[0] {
            "INT" => {
                let min = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
                let max = parts
                    .get(2)
                    .and_then(|s| s.trim_end_matches(';').parse().ok())
                    .unwrap_or(0);
                Ok(AttributeValueType::Int { min, max })
            }
            "FLOAT" | "HEX" => {
                let min = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let max = parts
                    .get(2)
                    .and_then(|s| s.trim_end_matches(';').parse().ok())
                    .unwrap_or(0.0);
                Ok(AttributeValueType::Float { min, max })
            }
            "STRING" => Ok(AttributeValueType::String),
            "ENUM" => {
                let values: Vec<String> = parts[1..]
                    .iter()
                    .map(|s| {
                        s.trim_matches(|c| c == '"' || c == ',' || c == ';')
                            .to_string()
                    })
                    .filter(|s| !s.is_empty())
                    .collect();
                Ok(AttributeValueType::Enum { values })
            }
            _ => Ok(AttributeValueType::String),
        }
    }

    /// Parse BA_DEF_DEF_ (attribute default value)
    fn parse_attribute_default(&mut self, line: &str) -> CanbusResult<()> {
        // BA_DEF_DEF_ "AttributeName" 100;
        let name_start = line.find('"');
        let name_end = line[name_start.unwrap_or(0) + 1..].find('"');

        if let (Some(start), Some(end)) = (name_start, name_end) {
            let name = &line[start + 1..start + 1 + end];
            let remaining = line[start + 2 + end..].trim().trim_end_matches(';');

            // First, find the value type without mutating
            let value_type = self
                .database
                .attribute_definitions
                .iter()
                .find(|d| d.name == name)
                .map(|d| d.value_type.clone());

            // Now parse and update
            if let Some(vt) = value_type {
                let default_value = Self::parse_attribute_literal_static(remaining, &vt);
                if let Some(def) = self
                    .database
                    .attribute_definitions
                    .iter_mut()
                    .find(|d| d.name == name)
                {
                    def.default_value = Some(default_value);
                }
            }
        }

        Ok(())
    }

    /// Parse attribute literal value (static version for borrow checker)
    fn parse_attribute_literal_static(
        value: &str,
        value_type: &AttributeValueType,
    ) -> AttributeValue {
        let trimmed = value.trim().trim_matches('"');
        match value_type {
            AttributeValueType::Int { .. } => AttributeValue::Int(trimmed.parse().unwrap_or(0)),
            AttributeValueType::Float { .. } => {
                AttributeValue::Float(trimmed.parse().unwrap_or(0.0))
            }
            AttributeValueType::Enum { .. } => AttributeValue::Enum(trimmed.to_string()),
            AttributeValueType::String => AttributeValue::String(trimmed.to_string()),
        }
    }

    /// Parse BA_ (attribute value)
    fn parse_attribute_value(&mut self, line: &str) -> CanbusResult<()> {
        // BA_ "AttributeName" BO_ 2024 100;
        // BA_ "AttributeName" SG_ 2024 SignalName 50;

        let name_start = line.find('"');
        let name_end = line[name_start.unwrap_or(0) + 1..].find('"');

        if let (Some(start), Some(end)) = (name_start, name_end) {
            let attr_name = line[start + 1..start + 1 + end].to_string();
            let remaining = &line[start + 2 + end..];
            let parts: Vec<&str> = remaining.split_whitespace().collect();

            // Find attribute definition for type info
            let value_type = self
                .database
                .attribute_definitions
                .iter()
                .find(|d| d.name == attr_name)
                .map(|d| d.value_type.clone())
                .unwrap_or(AttributeValueType::String);

            if parts.len() >= 2 && parts[0] == "BO_" {
                // Message attribute
                if let Ok(msg_id) = parts[1].parse::<u32>() {
                    if let Some(value_str) = parts.get(2) {
                        let value = Self::parse_attribute_literal_static(
                            value_str.trim_end_matches(';'),
                            &value_type,
                        );
                        if let Some(msg) =
                            self.database.messages.iter_mut().find(|m| m.id == msg_id)
                        {
                            msg.attributes.insert(attr_name, value);
                        }
                    }
                }
            } else if parts.len() >= 3 && parts[0] == "SG_" {
                // Signal attribute - note: DbcSignal doesn't have attributes by default
                // This is parsed but not stored currently
                let _msg_id = parts[1].parse::<u32>().ok();
                let _sig_name = parts[2];
            }
        }

        Ok(())
    }

    /// Parse VAL_TABLE_ (standalone value table)
    fn parse_value_table(&mut self, line: &str) -> CanbusResult<()> {
        // VAL_TABLE_ TableName 0 "Value0" 1 "Value1" 2 "Value2";
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            return Ok(());
        }

        let name = parts[1].to_string();
        let mut values = HashMap::new();

        let mut i = 2;
        while i + 1 < parts.len() {
            if let Ok(key) = parts[i].parse::<i64>() {
                let value = parts[i + 1].trim_matches(|c| c == '"' || c == ';');
                values.insert(key, value.to_string());
                i += 2;
            } else {
                break;
            }
        }

        self.database.value_tables.insert(name, values);
        Ok(())
    }

    /// Parse VAL_ (value descriptions for signal)
    fn parse_value_descriptions(&mut self, line: &str) -> CanbusResult<()> {
        // VAL_ 2024 SignalName 0 "Off" 1 "On" 2 "Error";
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Ok(());
        }

        let msg_id: u32 = match parts[1].parse() {
            Ok(id) => id,
            Err(_) => return Ok(()),
        };

        let sig_name = parts[2];
        let mut values = HashMap::new();

        let mut i = 3;
        while i + 1 < parts.len() {
            if let Ok(key) = parts[i].parse::<i64>() {
                let value = parts[i + 1].trim_matches(|c| c == '"' || c == ';');
                values.insert(key, value.to_string());
                i += 2;
            } else {
                i += 1;
            }
        }

        // Apply to signal
        if let Some(msg) = self.database.messages.iter_mut().find(|m| m.id == msg_id) {
            if let Some(sig) = msg.signals.iter_mut().find(|s| s.name == sig_name) {
                sig.value_descriptions = values;
            }
        }

        Ok(())
    }

    /// Create error with line number
    fn error(&self, message: &str) -> CanbusError {
        CanbusError::DbcParseError {
            line: self.line_number,
            message: message.to_string(),
        }
    }
}

impl Default for DbcParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a DBC file from string content
pub fn parse_dbc(content: &str) -> CanbusResult<DbcDatabase> {
    DbcParser::new().parse(content)
}

/// Parse a DBC file from path
pub fn parse_dbc_file(path: impl AsRef<std::path::Path>) -> CanbusResult<DbcDatabase> {
    let content = std::fs::read_to_string(path.as_ref()).map_err(CanbusError::Io)?;
    parse_dbc(&content)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_DBC: &str = r#"
VERSION ""

NS_ :
    NS_DESC_
    CM_
    BA_DEF_
    BA_

BS_:

BU_: Engine Dashboard Transmission

BO_ 2024 EngineData: 8 Engine
 SG_ EngineSpeed : 0|16@1+ (0.125,0) [0|8031.875] "rpm" Dashboard
 SG_ EngineTemp : 16|8@1+ (1,-40) [-40|215] "degC" Dashboard
 SG_ ThrottlePos : 24|8@1+ (0.392157,0) [0|100] "%" Dashboard

BO_ 2028 VehicleSpeed: 4 Transmission
 SG_ Speed : 0|16@1+ (0.01,0) [0|655.35] "km/h" Dashboard

CM_ BO_ 2024 "Engine data message containing RPM, temperature and throttle";
CM_ SG_ 2024 EngineSpeed "Engine rotational speed in RPM";
CM_ SG_ 2024 EngineTemp "Engine coolant temperature";
CM_ BU_ Engine "Engine control unit";

BA_DEF_ BO_ "GenMsgCycleTime" INT 0 10000;
BA_DEF_DEF_ "GenMsgCycleTime" 100;
BA_ "GenMsgCycleTime" BO_ 2024 50;

VAL_ 2024 ThrottlePos 0 "Closed" 100 "WOT";
"#;

    #[test]
    fn test_parse_dbc() {
        let db = parse_dbc(TEST_DBC).expect("DBC parsing should succeed");

        assert_eq!(db.nodes.len(), 3);
        assert_eq!(db.messages.len(), 2);
    }

    #[test]
    fn test_parse_message() {
        let db = parse_dbc(TEST_DBC).expect("DBC parsing should succeed");

        let engine_msg = db.get_message(2024).expect("message should exist");
        assert_eq!(engine_msg.name, "EngineData");
        assert_eq!(engine_msg.dlc, 8);
        assert_eq!(engine_msg.transmitter, "Engine");
        assert_eq!(engine_msg.signals.len(), 3);
    }

    #[test]
    fn test_parse_signal() {
        let db = parse_dbc(TEST_DBC).expect("DBC parsing should succeed");

        let engine_msg = db.get_message(2024).expect("message should exist");
        let speed_sig = engine_msg
            .get_signal("EngineSpeed")
            .expect("signal should exist");

        assert_eq!(speed_sig.start_bit, 0);
        assert_eq!(speed_sig.bit_length, 16);
        assert_eq!(speed_sig.factor, 0.125);
        assert_eq!(speed_sig.offset, 0.0);
        assert_eq!(speed_sig.unit, "rpm");
        assert_eq!(speed_sig.byte_order, ByteOrder::LittleEndian);
        assert_eq!(speed_sig.value_type, ValueType::Unsigned);
    }

    #[test]
    fn test_parse_signal_with_offset() {
        let db = parse_dbc(TEST_DBC).expect("DBC parsing should succeed");

        let engine_msg = db.get_message(2024).expect("message should exist");
        let temp_sig = engine_msg
            .get_signal("EngineTemp")
            .expect("signal should exist");

        assert_eq!(temp_sig.start_bit, 16);
        assert_eq!(temp_sig.bit_length, 8);
        assert_eq!(temp_sig.factor, 1.0);
        assert_eq!(temp_sig.offset, -40.0);
        assert_eq!(temp_sig.min, -40.0);
        assert_eq!(temp_sig.max, 215.0);
    }

    #[test]
    fn test_physical_value_conversion() {
        let db = parse_dbc(TEST_DBC).expect("DBC parsing should succeed");

        let engine_msg = db.get_message(2024).expect("message should exist");

        // Test engine speed: raw 16000 -> physical 2000 rpm
        let speed_sig = engine_msg
            .get_signal("EngineSpeed")
            .expect("signal should exist");
        assert!((speed_sig.to_physical(16000) - 2000.0).abs() < 0.001);

        // Test engine temp: raw 125 -> physical 85°C
        let temp_sig = engine_msg
            .get_signal("EngineTemp")
            .expect("signal should exist");
        assert!((temp_sig.to_physical(125) - 85.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_comments() {
        let db = parse_dbc(TEST_DBC).expect("DBC parsing should succeed");

        let engine_msg = db.get_message(2024).expect("message should exist");
        assert!(engine_msg.comment.is_some());
        assert!(engine_msg
            .comment
            .as_ref()
            .expect("reference should be available")
            .contains("Engine data message"));

        let speed_sig = engine_msg
            .get_signal("EngineSpeed")
            .expect("signal should exist");
        assert!(speed_sig.comment.is_some());
        assert!(speed_sig
            .comment
            .as_ref()
            .expect("reference should be available")
            .contains("rotational speed"));
    }

    #[test]
    fn test_parse_value_descriptions() {
        let db = parse_dbc(TEST_DBC).expect("DBC parsing should succeed");

        let engine_msg = db.get_message(2024).expect("message should exist");
        let throttle_sig = engine_msg
            .get_signal("ThrottlePos")
            .expect("signal should exist");

        assert_eq!(throttle_sig.get_value_description(0), Some("Closed"));
        assert_eq!(throttle_sig.get_value_description(100), Some("WOT"));
    }

    #[test]
    fn test_parse_attributes() {
        let db = parse_dbc(TEST_DBC).expect("DBC parsing should succeed");

        assert_eq!(db.attribute_definitions.len(), 1);
        let attr_def = &db.attribute_definitions[0];
        assert_eq!(attr_def.name, "GenMsgCycleTime");

        let engine_msg = db.get_message(2024).expect("message should exist");
        assert!(engine_msg.attributes.contains_key("GenMsgCycleTime"));
    }

    #[test]
    fn test_parse_nodes() {
        let db = parse_dbc(TEST_DBC).expect("DBC parsing should succeed");

        assert!(db.get_node("Engine").is_some());
        assert!(db.get_node("Dashboard").is_some());
        assert!(db.get_node("Transmission").is_some());

        let engine_node = db.get_node("Engine").expect("operation should succeed");
        assert!(engine_node.comment.is_some());
    }

    #[test]
    fn test_vehicle_speed_message() {
        let db = parse_dbc(TEST_DBC).expect("DBC parsing should succeed");

        let speed_msg = db.get_message(2028).expect("message should exist");
        assert_eq!(speed_msg.name, "VehicleSpeed");
        assert_eq!(speed_msg.dlc, 4);

        let speed_sig = speed_msg.get_signal("Speed").expect("signal should exist");
        assert_eq!(speed_sig.factor, 0.01);

        // Test: raw 10000 -> physical 100 km/h
        assert!((speed_sig.to_physical(10000) - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_byte_order_big_endian() {
        let dbc = r#"
BO_ 100 TestMsg: 8 ECU
 SG_ BigEndianSig : 7|16@0+ (1,0) [0|65535] "" Vector__XXX
"#;
        let db = parse_dbc(dbc).expect("DBC parsing should succeed");

        let msg = db.get_message(100).expect("message should exist");
        let sig = msg.get_signal("BigEndianSig").expect("signal should exist");
        assert_eq!(sig.byte_order, ByteOrder::BigEndian);
    }

    #[test]
    fn test_signed_signal() {
        let dbc = r#"
BO_ 100 TestMsg: 8 ECU
 SG_ SignedSig : 0|16@1- (1,0) [-32768|32767] "" Vector__XXX
"#;
        let db = parse_dbc(dbc).expect("DBC parsing should succeed");

        let msg = db.get_message(100).expect("message should exist");
        let sig = msg.get_signal("SignedSig").expect("signal should exist");
        assert_eq!(sig.value_type, ValueType::Signed);
    }

    #[test]
    fn test_all_signals() {
        let db = parse_dbc(TEST_DBC).expect("DBC parsing should succeed");

        let all_sigs: Vec<_> = db.all_signals().collect();
        assert_eq!(all_sigs.len(), 4); // 3 in EngineData + 1 in VehicleSpeed
    }

    #[test]
    fn test_empty_dbc() {
        let dbc = "VERSION \"\"";
        let db = parse_dbc(dbc).expect("DBC parsing should succeed");
        assert!(db.messages.is_empty());
    }
}
