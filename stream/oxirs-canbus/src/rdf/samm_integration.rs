//! SAMM (Semantic Aspect Meta Model) integration for CAN/DBC
//!
//! Provides automatic generation of SAMM Aspect Models from DBC files.
//! This enables semantic interoperability with Digital Twin platforms.
//!
//! # Overview
//!
//! The SAMM integration maps:
//! - DbcDatabase → Multiple SAMM Aspects (one per message)
//! - DbcMessage → SAMM Aspect
//! - DbcSignal → SAMM Property
//! - Signal unit → QUDT Unit
//!
//! # Example
//!
//! ```no_run
//! use oxirs_canbus::dbc::parse_dbc;
//! use oxirs_canbus::rdf::{DbcSammGenerator, SammConfig};
//!
//! let dbc_content = r#"
//! BO_ 2024 EngineData: 8 Engine
//!  SG_ EngineSpeed : 0|16@1+ (0.125,0) [0|8031.875] "rpm" Dashboard
//!  SG_ EngineTemp : 16|8@1+ (1,-40) [-40|215] "degC" Dashboard
//! "#;
//!
//! let db = parse_dbc(dbc_content).unwrap();
//! let config = SammConfig::new("1.0.0", "http://automotive.example.com/");
//! let generator = DbcSammGenerator::new(config);
//!
//! let ttl = generator.generate_from_database(&db);
//! std::fs::write("EngineData.ttl", ttl).unwrap();
//! ```

use crate::dbc::{ByteOrder, DbcDatabase, DbcMessage, DbcSignal, MultiplexerType, ValueType};
use crate::error::CanbusResult;
use std::collections::HashSet;
use std::fmt::Write;

/// SAMM meta-model namespace prefix (version 2.1.0)
pub const SAMM_PREFIX: &str = "urn:samm:org.eclipse.esmf.samm:meta-model:2.1.0#";
/// SAMM characteristic namespace prefix (version 2.1.0)
pub const SAMM_C_PREFIX: &str = "urn:samm:org.eclipse.esmf.samm:characteristic:2.1.0#";
/// SAMM entity namespace prefix (version 2.1.0)
pub const SAMM_E_PREFIX: &str = "urn:samm:org.eclipse.esmf.samm:entity:2.1.0#";
/// SAMM unit namespace prefix (version 2.1.0)
pub const SAMM_U_PREFIX: &str = "urn:samm:org.eclipse.esmf.samm:unit:2.1.0#";
/// XML Schema datatype namespace prefix
pub const XSD_PREFIX: &str = "http://www.w3.org/2001/XMLSchema#";
/// RDF Schema namespace prefix
pub const RDFS_PREFIX: &str = "http://www.w3.org/2000/01/rdf-schema#";

/// SAMM generation configuration
#[derive(Debug, Clone)]
pub struct SammConfig {
    /// Version of the generated aspect model
    pub version: String,
    /// Base namespace for generated IRIs
    pub namespace: String,
    /// Whether to include detailed comments
    pub include_comments: bool,
    /// Whether to generate constraints from min/max values
    pub generate_constraints: bool,
    /// Whether to generate enumerations from value descriptions
    pub generate_enumerations: bool,
}

impl Default for SammConfig {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            namespace: "http://automotive.oxirs.org/canbus/".to_string(),
            include_comments: true,
            generate_constraints: true,
            generate_enumerations: true,
        }
    }
}

impl SammConfig {
    /// Create a new SAMM configuration
    pub fn new(version: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            namespace: namespace.into(),
            ..Default::default()
        }
    }
}

/// SAMM Aspect Model generator for DBC files
///
/// Generates Turtle format SAMM models from DBC databases.
#[derive(Debug, Clone, Default)]
pub struct DbcSammGenerator {
    /// Configuration
    config: SammConfig,
}

impl DbcSammGenerator {
    /// Create a new DBC SAMM generator
    pub fn new(config: SammConfig) -> Self {
        Self { config }
    }

    /// Generate SAMM Aspect Models for all messages in a DBC database
    ///
    /// Returns a single Turtle file containing all aspects.
    pub fn generate_from_database(&self, database: &DbcDatabase) -> String {
        let mut output = String::with_capacity(16384);

        // Generate prefixes
        self.write_prefixes(&mut output);

        // Generate an Aspect for each message
        for message in &database.messages {
            self.write_message_aspect(&mut output, message);
        }

        output
    }

    /// Generate SAMM Aspect Model for a single message
    pub fn generate_for_message(&self, message: &DbcMessage) -> String {
        let mut output = String::with_capacity(4096);

        self.write_prefixes(&mut output);
        self.write_message_aspect(&mut output, message);

        output
    }

    /// Write Turtle prefixes
    fn write_prefixes(&self, output: &mut String) {
        writeln!(output, "@prefix : <{}> .", self.config.namespace).unwrap();
        writeln!(output, "@prefix samm: <{}> .", SAMM_PREFIX).unwrap();
        writeln!(output, "@prefix samm-c: <{}> .", SAMM_C_PREFIX).unwrap();
        writeln!(output, "@prefix samm-e: <{}> .", SAMM_E_PREFIX).unwrap();
        writeln!(output, "@prefix unit: <{}> .", SAMM_U_PREFIX).unwrap();
        writeln!(output, "@prefix xsd: <{}> .", XSD_PREFIX).unwrap();
        writeln!(output, "@prefix rdfs: <{}> .", RDFS_PREFIX).unwrap();
        writeln!(output).unwrap();
    }

    /// Write a complete Aspect for a CAN message
    fn write_message_aspect(&self, output: &mut String, message: &DbcMessage) {
        let aspect_name = self.sanitize_name(&message.name);

        if self.config.include_comments {
            writeln!(output, "# ========================================").unwrap();
            writeln!(output, "# Aspect Model for CAN Message: {}", message.name).unwrap();
            writeln!(
                output,
                "# CAN ID: {:#X} ({} bytes)",
                message.id, message.dlc
            )
            .unwrap();
            if !message.transmitter.is_empty() {
                writeln!(output, "# Transmitter: {}", message.transmitter).unwrap();
            }
            writeln!(output, "# Signals: {}", message.signals.len()).unwrap();
            writeln!(output, "# ========================================").unwrap();
            writeln!(output).unwrap();
        }

        // Write Aspect definition
        self.write_aspect_definition(output, message, &aspect_name);

        // Write Properties
        self.write_properties(output, message, &aspect_name);

        // Write Characteristics
        self.write_characteristics(output, message, &aspect_name);

        // Write Enumerations (if enabled and signal has value descriptions)
        if self.config.generate_enumerations {
            self.write_enumerations(output, message, &aspect_name);
        }

        // Write Constraints (if enabled)
        if self.config.generate_constraints {
            self.write_constraints(output, message, &aspect_name);
        }

        writeln!(output).unwrap();
    }

    /// Write the Aspect definition
    fn write_aspect_definition(
        &self,
        output: &mut String,
        message: &DbcMessage,
        aspect_name: &str,
    ) {
        writeln!(output, ":{}Aspect a samm:Aspect ;", aspect_name).unwrap();
        writeln!(output, "    samm:preferredName \"{}\"@en ;", message.name).unwrap();

        // Description
        let default_desc = format!(
            "CAN message {} (ID: {:#X}) with {} signals",
            message.name,
            message.id,
            message.signals.len()
        );
        let desc = message.comment.as_deref().unwrap_or(&default_desc);
        writeln!(
            output,
            "    samm:description \"{}\"@en ;",
            escape_string(desc)
        )
        .unwrap();

        // Properties list
        if !message.signals.is_empty() {
            writeln!(output, "    samm:properties (").unwrap();
            for signal in &message.signals {
                let prop_name = self.sanitize_name(&signal.name);
                writeln!(output, "        :{}Property", prop_name).unwrap();
            }
            writeln!(output, "    ) .").unwrap();
        } else {
            // Close with empty properties
            output.truncate(output.len() - 2);
            writeln!(output, " .").unwrap();
        }

        writeln!(output).unwrap();
    }

    /// Write Property definitions
    fn write_properties(&self, output: &mut String, message: &DbcMessage, aspect_name: &str) {
        if self.config.include_comments && !message.signals.is_empty() {
            writeln!(output, "# Properties for {}", message.name).unwrap();
            writeln!(output).unwrap();
        }

        for signal in &message.signals {
            self.write_property(output, signal, aspect_name);
        }
    }

    /// Write a single Property definition
    fn write_property(&self, output: &mut String, signal: &DbcSignal, _aspect_name: &str) {
        let prop_name = self.sanitize_name(&signal.name);
        let char_name = self.characteristic_name(signal);

        writeln!(output, ":{}Property a samm:Property ;", prop_name).unwrap();
        writeln!(output, "    samm:preferredName \"{}\"@en ;", signal.name).unwrap();

        // Description
        if let Some(ref comment) = signal.comment {
            writeln!(
                output,
                "    samm:description \"{}\"@en ;",
                escape_string(comment)
            )
            .unwrap();
        } else if self.config.include_comments {
            let desc = format!(
                "Signal at bit {} ({} bits, {} byte order)",
                signal.start_bit,
                signal.bit_length,
                if signal.byte_order == ByteOrder::LittleEndian {
                    "Intel"
                } else {
                    "Motorola"
                }
            );
            writeln!(output, "    samm:description \"{}\"@en ;", desc).unwrap();
        }

        // Characteristic reference
        writeln!(
            output,
            "    samm:characteristic :{}Characteristic .",
            char_name
        )
        .unwrap();
        writeln!(output).unwrap();
    }

    /// Write Characteristic definitions
    fn write_characteristics(&self, output: &mut String, message: &DbcMessage, aspect_name: &str) {
        if self.config.include_comments && !message.signals.is_empty() {
            writeln!(output, "# Characteristics for {}", message.name).unwrap();
            writeln!(output).unwrap();
        }

        // Track unique characteristics to avoid duplicates
        let mut written: HashSet<String> = HashSet::new();

        for signal in &message.signals {
            let char_name = self.characteristic_name(signal);
            if written.contains(&char_name) {
                continue;
            }
            written.insert(char_name.clone());

            self.write_characteristic(output, signal, aspect_name);
        }
    }

    /// Write a single Characteristic definition
    fn write_characteristic(&self, output: &mut String, signal: &DbcSignal, _aspect_name: &str) {
        let char_name = self.characteristic_name(signal);
        let (samm_type, xsd_type) = self.signal_type_mapping(signal);

        // Check if this is an enumeration
        let is_enum = !signal.value_descriptions.is_empty() && self.config.generate_enumerations;

        if is_enum {
            let enum_name = self.sanitize_name(&signal.name);
            writeln!(
                output,
                ":{}Characteristic a samm-c:Enumeration ;",
                char_name
            )
            .unwrap();
            writeln!(output, "    samm:dataType :{}Enumeration .", enum_name).unwrap();
        } else {
            writeln!(output, ":{}Characteristic a {} ;", char_name, samm_type).unwrap();
            write!(output, "    samm:dataType {} ", xsd_type).unwrap();

            // Add unit if present
            if !signal.unit.is_empty() {
                let samm_unit = self.map_unit(&signal.unit);
                writeln!(output, ";").unwrap();
                writeln!(output, "    samm-c:unit unit:{} .", samm_unit).unwrap();
            } else {
                writeln!(output, ".").unwrap();
            }
        }

        writeln!(output).unwrap();
    }

    /// Write Enumeration definitions for signals with value descriptions
    fn write_enumerations(&self, output: &mut String, message: &DbcMessage, _aspect_name: &str) {
        let signals_with_enums: Vec<_> = message
            .signals
            .iter()
            .filter(|s| !s.value_descriptions.is_empty())
            .collect();

        if signals_with_enums.is_empty() {
            return;
        }

        if self.config.include_comments {
            writeln!(output, "# Enumerations for {}", message.name).unwrap();
            writeln!(output).unwrap();
        }

        for signal in signals_with_enums {
            let enum_name = self.sanitize_name(&signal.name);

            // Write enumeration entity
            writeln!(output, ":{}Enumeration a rdfs:Datatype ;", enum_name).unwrap();
            writeln!(output, "    samm:preferredName \"{}\"@en ;", signal.name).unwrap();

            // Write values
            let mut values: Vec<_> = signal.value_descriptions.iter().collect();
            values.sort_by_key(|(k, _)| *k);

            writeln!(output, "    owl:oneOf (").unwrap();
            for (raw, desc) in &values {
                writeln!(output, "        \"{}\" # raw value: {}", desc, raw).unwrap();
            }
            writeln!(output, "    ) .").unwrap();
            writeln!(output).unwrap();
        }
    }

    /// Write Constraint definitions for signals with min/max values
    fn write_constraints(&self, output: &mut String, message: &DbcMessage, _aspect_name: &str) {
        let signals_with_constraints: Vec<_> = message
            .signals
            .iter()
            .filter(|s| s.min != 0.0 || s.max != 0.0)
            .filter(|s| s.value_descriptions.is_empty()) // Skip enums
            .collect();

        if signals_with_constraints.is_empty() {
            return;
        }

        if self.config.include_comments {
            writeln!(output, "# Constraints for {}", message.name).unwrap();
            writeln!(output).unwrap();
        }

        for signal in signals_with_constraints {
            let constraint_name = self.sanitize_name(&signal.name);

            writeln!(
                output,
                ":{}RangeConstraint a samm-c:RangeConstraint ;",
                constraint_name
            )
            .unwrap();

            if signal.min != 0.0 {
                writeln!(
                    output,
                    "    samm-c:minValue \"{}\"^^xsd:decimal ;",
                    signal.min
                )
                .unwrap();
            }

            if signal.max != 0.0 {
                write!(
                    output,
                    "    samm-c:maxValue \"{}\"^^xsd:decimal ",
                    signal.max
                )
                .unwrap();
            } else {
                // Remove trailing " ;\n"
                output.truncate(output.len() - 2);
                write!(output, " ").unwrap();
            }

            writeln!(output, ".").unwrap();
            writeln!(output).unwrap();
        }
    }

    /// Generate characteristic name from signal
    fn characteristic_name(&self, signal: &DbcSignal) -> String {
        let base = self.sanitize_name(&signal.name);

        // Add unit suffix if present
        if !signal.unit.is_empty() {
            let unit = self.sanitize_name(&signal.unit);
            format!("{}{}", base, unit)
        } else {
            base
        }
    }

    /// Sanitize a name for use in SAMM identifiers
    fn sanitize_name(&self, name: &str) -> String {
        let mut result = String::with_capacity(name.len());
        let mut capitalize_next = true;

        for c in name.chars() {
            if c.is_alphanumeric() {
                if capitalize_next {
                    result.extend(c.to_uppercase());
                    capitalize_next = false;
                } else {
                    result.push(c);
                }
            } else {
                capitalize_next = true;
            }
        }

        if result.is_empty() {
            return "Unknown".to_string();
        }

        result
    }

    /// Map signal to SAMM characteristic type and XSD datatype
    fn signal_type_mapping(&self, signal: &DbcSignal) -> (&'static str, &'static str) {
        // Check if it's a measurement (has unit)
        let has_unit = !signal.unit.is_empty();

        match signal.value_type {
            ValueType::Unsigned => {
                if signal.bit_length == 1 {
                    ("samm-c:Boolean", "xsd:boolean")
                } else if signal.bit_length <= 16 {
                    if has_unit {
                        ("samm-c:Measurement", "xsd:unsignedShort")
                    } else {
                        ("samm:Characteristic", "xsd:unsignedShort")
                    }
                } else if signal.bit_length <= 32 {
                    if has_unit {
                        ("samm-c:Measurement", "xsd:unsignedInt")
                    } else {
                        ("samm:Characteristic", "xsd:unsignedInt")
                    }
                } else if has_unit {
                    ("samm-c:Measurement", "xsd:unsignedLong")
                } else {
                    ("samm:Characteristic", "xsd:unsignedLong")
                }
            }
            ValueType::Signed => {
                if signal.bit_length <= 16 {
                    if has_unit {
                        ("samm-c:Measurement", "xsd:short")
                    } else {
                        ("samm:Characteristic", "xsd:short")
                    }
                } else if signal.bit_length <= 32 {
                    if has_unit {
                        ("samm-c:Measurement", "xsd:int")
                    } else {
                        ("samm:Characteristic", "xsd:int")
                    }
                } else if has_unit {
                    ("samm-c:Measurement", "xsd:long")
                } else {
                    ("samm:Characteristic", "xsd:long")
                }
            }
        }
    }

    /// Map unit string to SAMM/QUDT unit
    fn map_unit(&self, unit: &str) -> String {
        let unit_lower = unit.to_lowercase();
        match unit_lower.as_str() {
            // Automotive common units
            "rpm" | "r/min" | "rev/min" => "revolutionPerMinute".to_string(),
            "km/h" | "kph" => "kilometrePerHour".to_string(),
            "mph" | "mi/h" => "milePerHour".to_string(),
            "m/s" | "mps" => "metrePerSecond".to_string(),
            // Temperature
            "degc" | "°c" | "deg c" | "celsius" => "degreeCelsius".to_string(),
            "degf" | "°f" | "deg f" | "fahrenheit" => "degreeFahrenheit".to_string(),
            "k" | "kelvin" => "kelvin".to_string(),
            // Percentage
            "%" | "percent" | "percentage" => "percent".to_string(),
            // Pressure
            "bar" => "bar".to_string(),
            "psi" => "poundForcePerSquareInch".to_string(),
            "kpa" | "kpascal" => "kilopascal".to_string(),
            "pa" | "pascal" => "pascal".to_string(),
            "mbar" => "millibar".to_string(),
            // Electrical
            "v" | "volt" | "volts" => "volt".to_string(),
            "a" | "amp" | "ampere" | "amps" => "ampere".to_string(),
            "w" | "watt" | "watts" => "watt".to_string(),
            "kw" | "kilowatt" => "kilowatt".to_string(),
            "hz" | "hertz" => "hertz".to_string(),
            // Torque
            "nm" | "n·m" | "newton-meter" | "newtonmeter" => "newtonMetre".to_string(),
            "lb-ft" | "lbft" => "poundForceFoot".to_string(),
            // Fuel economy
            "l/100km" => "litrePerHundredKilometre".to_string(),
            "mpg" => "milePerGallon".to_string(),
            "l/h" | "l/hr" => "litrePerHour".to_string(),
            // Mass
            "kg" | "kilogram" => "kilogram".to_string(),
            "g" | "gram" => "gram".to_string(),
            "lb" | "pound" => "pound".to_string(),
            // Length
            "m" | "meter" | "metre" => "metre".to_string(),
            "cm" | "centimeter" => "centimetre".to_string(),
            "mm" | "millimeter" => "millimetre".to_string(),
            "km" | "kilometer" => "kilometre".to_string(),
            "in" | "inch" => "inch".to_string(),
            "ft" | "foot" => "foot".to_string(),
            "mi" | "mile" => "mile".to_string(),
            // Time
            "s" | "sec" | "second" => "second".to_string(),
            "ms" | "millisecond" => "millisecond".to_string(),
            "min" | "minute" => "minute".to_string(),
            "h" | "hr" | "hour" => "hour".to_string(),
            // Volume
            "l" | "liter" | "litre" => "litre".to_string(),
            "gal" | "gallon" => "gallon".to_string(),
            // Default: use sanitized name
            _ => self.sanitize_name(unit),
        }
    }

    /// Save generated SAMM model to file
    pub fn save_to_file(
        &self,
        database: &DbcDatabase,
        path: impl AsRef<std::path::Path>,
    ) -> CanbusResult<()> {
        let ttl = self.generate_from_database(database);
        std::fs::write(path, ttl).map_err(|e| {
            crate::error::CanbusError::Config(format!("Failed to write SAMM model: {}", e))
        })
    }
}

/// Escape special characters in strings for Turtle format
fn escape_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// SAMM validation result
#[derive(Debug, Clone)]
pub struct SammValidationResult {
    /// Whether validation passed
    pub valid: bool,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Validation errors
    pub errors: Vec<String>,
}

impl SammValidationResult {
    /// Check if result has any issues
    pub fn has_issues(&self) -> bool {
        !self.valid || !self.warnings.is_empty()
    }
}

/// Validate a DBC database for SAMM compatibility
pub fn validate_for_samm(database: &DbcDatabase) -> SammValidationResult {
    let mut result = SammValidationResult {
        valid: true,
        warnings: Vec::new(),
        errors: Vec::new(),
    };

    // Check for empty database
    if database.messages.is_empty() {
        result
            .errors
            .push("DBC database has no messages".to_string());
        result.valid = false;
        return result;
    }

    // Check each message
    for message in &database.messages {
        // Check for empty signals
        if message.signals.is_empty() {
            result.warnings.push(format!(
                "Message '{}' (ID: {:#X}) has no signals",
                message.name, message.id
            ));
        }

        // Check each signal
        for signal in &message.signals {
            // Check for missing unit on measurement-like signals
            if signal.unit.is_empty() && (signal.factor != 1.0 || signal.offset != 0.0) {
                result.warnings.push(format!(
                    "Signal '{}' in message '{}' has scaling but no unit",
                    signal.name, message.name
                ));
            }

            // Check for multiplexed signals (may need special handling)
            if matches!(signal.multiplexer, MultiplexerType::Multiplexed(_)) {
                result.warnings.push(format!(
                    "Signal '{}' is multiplexed - SAMM model may need manual adjustment",
                    signal.name
                ));
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dbc::parse_dbc;

    const TEST_DBC: &str = r#"
VERSION ""

BU_: Engine Dashboard

BO_ 2024 EngineData: 8 Engine
 SG_ EngineSpeed : 0|16@1+ (0.125,0) [0|8031.875] "rpm" Dashboard
 SG_ EngineTemp : 16|8@1+ (1,-40) [-40|215] "degC" Dashboard
 SG_ ThrottlePos : 24|8@1+ (0.392157,0) [0|100] "%" Dashboard

BO_ 2028 VehicleSpeed: 4 Transmission
 SG_ Speed : 0|16@1+ (0.01,0) [0|655.35] "km/h" Dashboard

CM_ BO_ 2024 "Engine data message containing RPM, temperature and throttle";
CM_ SG_ 2024 EngineSpeed "Engine rotational speed in RPM";

VAL_ 2024 ThrottlePos 0 "Closed" 100 "WOT";
"#;

    #[test]
    fn test_samm_generator_creation() {
        let config = SammConfig::new("1.0.0", "http://test.org/");
        let gen = DbcSammGenerator::new(config.clone());
        assert_eq!(gen.config.version, "1.0.0");
        assert_eq!(gen.config.namespace, "http://test.org/");
    }

    #[test]
    fn test_generate_from_database() {
        let db = parse_dbc(TEST_DBC).unwrap();
        let config = SammConfig::new("1.0.0", "http://automotive.example.com/");
        let gen = DbcSammGenerator::new(config);

        let ttl = gen.generate_from_database(&db);

        // Check prefixes
        assert!(ttl.contains("@prefix samm:"));
        assert!(ttl.contains("@prefix samm-c:"));
        assert!(ttl.contains("@prefix unit:"));

        // Check aspects
        assert!(ttl.contains(":EngineDataAspect a samm:Aspect"));
        assert!(ttl.contains(":VehicleSpeedAspect a samm:Aspect"));

        // Check properties
        assert!(ttl.contains(":EngineSpeedProperty a samm:Property"));
        assert!(ttl.contains(":EngineTempProperty a samm:Property"));
        assert!(ttl.contains(":SpeedProperty a samm:Property"));

        // Check units
        assert!(ttl.contains("unit:revolutionPerMinute"));
        assert!(ttl.contains("unit:degreeCelsius"));
        assert!(ttl.contains("unit:kilometrePerHour"));
    }

    #[test]
    fn test_generate_for_single_message() {
        let db = parse_dbc(TEST_DBC).unwrap();
        let config = SammConfig::new("1.0.0", "http://test.org/");
        let gen = DbcSammGenerator::new(config);

        let message = db.get_message(2024).unwrap();
        let ttl = gen.generate_for_message(message);

        // Should have EngineData but not VehicleSpeed
        assert!(ttl.contains(":EngineDataAspect"));
        assert!(!ttl.contains(":VehicleSpeedAspect"));
    }

    #[test]
    fn test_sanitize_name() {
        let gen = DbcSammGenerator::default();

        assert_eq!(gen.sanitize_name("EngineSpeed"), "EngineSpeed");
        assert_eq!(gen.sanitize_name("engine_speed"), "EngineSpeed");
        assert_eq!(gen.sanitize_name("engine-temp"), "EngineTemp");
        assert_eq!(gen.sanitize_name("PGN_61444"), "PGN61444");
    }

    #[test]
    fn test_unit_mapping() {
        let gen = DbcSammGenerator::default();

        assert_eq!(gen.map_unit("rpm"), "revolutionPerMinute");
        assert_eq!(gen.map_unit("km/h"), "kilometrePerHour");
        assert_eq!(gen.map_unit("degC"), "degreeCelsius");
        assert_eq!(gen.map_unit("%"), "percent");
        assert_eq!(gen.map_unit("Nm"), "newtonMetre");
        assert_eq!(gen.map_unit("bar"), "bar");
    }

    #[test]
    fn test_signal_type_mapping() {
        let gen = DbcSammGenerator::default();

        // Unsigned 16-bit with unit -> Measurement
        let mut sig = DbcSignal::new("test");
        sig.bit_length = 16;
        sig.value_type = ValueType::Unsigned;
        sig.unit = "rpm".to_string();
        let (samm, xsd) = gen.signal_type_mapping(&sig);
        assert_eq!(samm, "samm-c:Measurement");
        assert_eq!(xsd, "xsd:unsignedShort");

        // Boolean (1-bit)
        sig.bit_length = 1;
        sig.unit.clear();
        let (samm, xsd) = gen.signal_type_mapping(&sig);
        assert_eq!(samm, "samm-c:Boolean");
        assert_eq!(xsd, "xsd:boolean");

        // Signed 32-bit
        sig.bit_length = 32;
        sig.value_type = ValueType::Signed;
        let (samm, xsd) = gen.signal_type_mapping(&sig);
        assert_eq!(samm, "samm:Characteristic");
        assert_eq!(xsd, "xsd:int");
    }

    #[test]
    fn test_validate_for_samm() {
        let db = parse_dbc(TEST_DBC).unwrap();
        let result = validate_for_samm(&db);

        assert!(result.valid);
    }

    #[test]
    fn test_validate_empty_database() {
        let db = DbcDatabase::default();
        let result = validate_for_samm(&db);

        assert!(!result.valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_enumeration_generation() {
        let db = parse_dbc(TEST_DBC).unwrap();
        let config = SammConfig::new("1.0.0", "http://test.org/");
        let gen = DbcSammGenerator::new(config);

        let ttl = gen.generate_from_database(&db);

        // ThrottlePos has value descriptions, should generate enumeration
        assert!(ttl.contains(":ThrottlePosEnumeration"));
        assert!(ttl.contains("samm-c:Enumeration"));
    }

    #[test]
    fn test_constraints_generation() {
        let db = parse_dbc(TEST_DBC).unwrap();
        let config = SammConfig::new("1.0.0", "http://test.org/");
        let gen = DbcSammGenerator::new(config);

        let ttl = gen.generate_from_database(&db);

        // EngineSpeed has min/max, should generate constraint
        assert!(ttl.contains("samm-c:RangeConstraint"));
        assert!(ttl.contains("samm-c:maxValue"));
    }

    #[test]
    fn test_escape_string() {
        assert_eq!(escape_string("hello"), "hello");
        assert_eq!(escape_string("hello\"world"), "hello\\\"world");
        assert_eq!(escape_string("line1\nline2"), "line1\\nline2");
    }

    #[test]
    fn test_config_default() {
        let config = SammConfig::default();
        assert!(config.include_comments);
        assert!(config.generate_constraints);
        assert!(config.generate_enumerations);
    }
}
