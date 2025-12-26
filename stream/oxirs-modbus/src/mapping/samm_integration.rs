//! SAMM (Semantic Aspect Meta Model) integration for Modbus
//!
//! Provides automatic generation of SAMM Aspect Models from Modbus register maps.
//! This enables semantic interoperability with Digital Twin platforms.
//!
//! # Overview
//!
//! The SAMM integration maps:
//! - RegisterMap → SAMM Aspect
//! - RegisterMapping → SAMM Property
//! - ModbusDataType → SAMM Characteristic
//! - Unit → QUDT Unit
//!
//! # Example
//!
//! ```no_run
//! use oxirs_modbus::mapping::{RegisterMap, SammGenerator};
//!
//! let register_map = RegisterMap::from_toml("modbus_map.toml").unwrap();
//! let generator = SammGenerator::new("1.0.0", "http://example.org/");
//!
//! let ttl = generator.generate_aspect_model(&register_map);
//! std::fs::write("ModbusDevice.ttl", ttl).unwrap();
//! ```

use super::{ModbusDataType, RegisterMap, RegisterMapping, RegisterType};
use crate::error::ModbusResult;
use std::collections::HashSet;
use std::fmt::Write;

/// SAMM namespace prefixes
pub const SAMM_PREFIX: &str = "urn:samm:org.eclipse.esmf.samm:meta-model:2.1.0#";
pub const SAMM_C_PREFIX: &str = "urn:samm:org.eclipse.esmf.samm:characteristic:2.1.0#";
pub const SAMM_E_PREFIX: &str = "urn:samm:org.eclipse.esmf.samm:entity:2.1.0#";
pub const SAMM_U_PREFIX: &str = "urn:samm:org.eclipse.esmf.samm:unit:2.1.0#";
pub const XSD_PREFIX: &str = "http://www.w3.org/2001/XMLSchema#";
pub const RDFS_PREFIX: &str = "http://www.w3.org/2000/01/rdf-schema#";

/// SAMM Aspect Model generator
///
/// Generates Turtle format SAMM models from Modbus register maps.
#[derive(Debug, Clone)]
pub struct SammGenerator {
    /// Version of the generated aspect model
    version: String,
    /// Base namespace for generated IRIs
    namespace: String,
    /// Whether to include detailed comments
    include_comments: bool,
    /// Whether to generate operations for writable registers
    generate_operations: bool,
}

impl Default for SammGenerator {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            namespace: "http://example.org/modbus/".to_string(),
            include_comments: true,
            generate_operations: true,
        }
    }
}

impl SammGenerator {
    /// Create a new SAMM generator
    pub fn new(version: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            namespace: namespace.into(),
            ..Default::default()
        }
    }

    /// Set whether to include detailed comments
    pub fn with_comments(mut self, include: bool) -> Self {
        self.include_comments = include;
        self
    }

    /// Set whether to generate operations for writable registers
    pub fn with_operations(mut self, generate: bool) -> Self {
        self.generate_operations = generate;
        self
    }

    /// Generate a complete SAMM Aspect Model from a register map
    pub fn generate_aspect_model(&self, register_map: &RegisterMap) -> String {
        let mut output = String::with_capacity(8192);

        // Generate prefixes
        self.write_prefixes(&mut output, &register_map.device_id);

        // Generate Aspect definition
        self.write_aspect(&mut output, register_map);

        // Generate Properties
        self.write_properties(&mut output, &register_map.registers);

        // Generate Characteristics
        self.write_characteristics(&mut output, &register_map.registers);

        // Generate Operations (if enabled)
        if self.generate_operations {
            self.write_operations(&mut output, &register_map.registers);
        }

        output
    }

    /// Write Turtle prefixes
    fn write_prefixes(&self, output: &mut String, device_id: &str) {
        let aspect_ns = self.aspect_namespace(device_id);

        writeln!(output, "@prefix : <{}> .", aspect_ns).unwrap();
        writeln!(output, "@prefix samm: <{}> .", SAMM_PREFIX).unwrap();
        writeln!(output, "@prefix samm-c: <{}> .", SAMM_C_PREFIX).unwrap();
        writeln!(output, "@prefix samm-e: <{}> .", SAMM_E_PREFIX).unwrap();
        writeln!(output, "@prefix unit: <{}> .", SAMM_U_PREFIX).unwrap();
        writeln!(output, "@prefix xsd: <{}> .", XSD_PREFIX).unwrap();
        writeln!(output, "@prefix rdfs: <{}> .", RDFS_PREFIX).unwrap();
        writeln!(output).unwrap();
    }

    /// Write the main Aspect definition
    fn write_aspect(&self, output: &mut String, register_map: &RegisterMap) {
        let aspect_name = self.sanitize_name(&register_map.device_id);

        if self.include_comments {
            writeln!(
                output,
                "# Aspect Model for Modbus device: {}",
                register_map.device_id
            )
            .unwrap();
            writeln!(
                output,
                "# Auto-generated from Modbus register map with {} registers",
                register_map.registers.len()
            )
            .unwrap();
            writeln!(output).unwrap();
        }

        // Aspect definition
        writeln!(output, ":{}Aspect a samm:Aspect ;", aspect_name).unwrap();
        writeln!(output, "    samm:preferredName \"{}\"@en ;", aspect_name).unwrap();

        // Description
        writeln!(
            output,
            "    samm:description \"Modbus device aspect model for {} with {} register mappings.\"@en ;",
            register_map.device_id,
            register_map.registers.len()
        )
        .unwrap();

        // Properties list
        let properties: Vec<String> = register_map
            .registers
            .iter()
            .map(|r| format!(":{}Property", self.property_name(r)))
            .collect();

        if !properties.is_empty() {
            writeln!(output, "    samm:properties (").unwrap();
            for prop in &properties {
                writeln!(output, "        {}", prop).unwrap();
            }
            writeln!(output, "    ) ;").unwrap();
        }

        // Operations (writable registers)
        if self.generate_operations {
            let writable: Vec<&RegisterMapping> = register_map
                .registers
                .iter()
                .filter(|r| matches!(r.register_type, RegisterType::Holding | RegisterType::Coil))
                .collect();

            if !writable.is_empty() {
                let operations: Vec<String> = writable
                    .iter()
                    .map(|r| format!(":write{}Operation", self.property_name(r)))
                    .collect();

                writeln!(output, "    samm:operations (").unwrap();
                for op in &operations {
                    writeln!(output, "        {}", op).unwrap();
                }
                writeln!(output, "    ) .").unwrap();
            } else {
                // No operations, close with period
                output.truncate(output.len() - 2); // Remove trailing " ;\n"
                writeln!(output, " .").unwrap();
            }
        } else {
            // No operations, close with period
            output.truncate(output.len() - 2); // Remove trailing " ;\n"
            writeln!(output, " .").unwrap();
        }

        writeln!(output).unwrap();
    }

    /// Write property definitions
    fn write_properties(&self, output: &mut String, registers: &[RegisterMapping]) {
        if self.include_comments {
            writeln!(output, "# Properties").unwrap();
            writeln!(output).unwrap();
        }

        for mapping in registers {
            self.write_property(output, mapping);
        }
    }

    /// Write a single property
    fn write_property(&self, output: &mut String, mapping: &RegisterMapping) {
        let prop_name = self.property_name(mapping);
        let char_name = self.characteristic_name(mapping);

        writeln!(output, ":{}Property a samm:Property ;", prop_name).unwrap();

        // Preferred name
        let display_name = mapping.name.as_deref().unwrap_or(&prop_name);
        writeln!(output, "    samm:preferredName \"{}\"@en ;", display_name).unwrap();

        // Description
        if self.include_comments {
            let reg_type_str = format!("{:?}", mapping.register_type).to_lowercase();
            writeln!(
                output,
                "    samm:description \"Modbus {} register at address {} (type: {:?})\"@en ;",
                reg_type_str, mapping.address, mapping.data_type
            )
            .unwrap();
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

    /// Write characteristic definitions
    fn write_characteristics(&self, output: &mut String, registers: &[RegisterMapping]) {
        if self.include_comments {
            writeln!(output, "# Characteristics").unwrap();
            writeln!(output).unwrap();
        }

        // Track unique characteristics to avoid duplicates
        let mut written: HashSet<String> = HashSet::new();

        for mapping in registers {
            let char_name = self.characteristic_name(mapping);
            if written.contains(&char_name) {
                continue;
            }
            written.insert(char_name.clone());

            self.write_characteristic(output, mapping);
        }
    }

    /// Write a single characteristic
    fn write_characteristic(&self, output: &mut String, mapping: &RegisterMapping) {
        let char_name = self.characteristic_name(mapping);
        let (samm_type, xsd_type) = self.data_type_mapping(mapping.data_type);

        writeln!(output, ":{}Characteristic a {} ;", char_name, samm_type).unwrap();
        writeln!(output, "    samm:dataType {} ", xsd_type).unwrap();

        // Add unit if present
        if let Some(ref unit) = mapping.unit {
            let samm_unit = self.map_unit(unit);
            // Remove the period we just wrote
            output.truncate(output.len() - 2); // Remove " \n"
            writeln!(output, ";").unwrap();
            writeln!(output, "    samm-c:unit unit:{} .", samm_unit).unwrap();
        } else {
            // Complete with period
            output.truncate(output.len() - 2);
            writeln!(output, ".").unwrap();
        }

        writeln!(output).unwrap();
    }

    /// Write operation definitions for writable registers
    fn write_operations(&self, output: &mut String, registers: &[RegisterMapping]) {
        let writable: Vec<&RegisterMapping> = registers
            .iter()
            .filter(|r| matches!(r.register_type, RegisterType::Holding | RegisterType::Coil))
            .collect();

        if writable.is_empty() {
            return;
        }

        if self.include_comments {
            writeln!(output, "# Operations").unwrap();
            writeln!(output).unwrap();
        }

        for mapping in writable {
            self.write_operation(output, mapping);
        }
    }

    /// Write a single operation
    fn write_operation(&self, output: &mut String, mapping: &RegisterMapping) {
        let prop_name = self.property_name(mapping);

        writeln!(output, ":write{}Operation a samm:Operation ;", prop_name).unwrap();
        let display_name = mapping.name.as_deref().unwrap_or(&prop_name);
        writeln!(
            output,
            "    samm:preferredName \"Write {}\"@en ;",
            display_name
        )
        .unwrap();
        let reg_type_str = format!("{:?}", mapping.register_type).to_lowercase();
        writeln!(
            output,
            "    samm:description \"Write value to Modbus {} register at address {}\"@en ;",
            reg_type_str, mapping.address
        )
        .unwrap();

        // Input property
        writeln!(output, "    samm:input ( :{}Property ) .", prop_name).unwrap();
        writeln!(output).unwrap();
    }

    /// Get namespace for the aspect
    fn aspect_namespace(&self, device_id: &str) -> String {
        format!(
            "{}{}:{}#",
            self.namespace,
            self.sanitize_name(device_id),
            self.version
        )
    }

    /// Generate property name from mapping
    fn property_name(&self, mapping: &RegisterMapping) -> String {
        mapping
            .name
            .as_ref()
            .map(|n| self.sanitize_name(n))
            .unwrap_or_else(|| format!("{:?}Addr{}", mapping.register_type, mapping.address))
    }

    /// Generate characteristic name from mapping
    fn characteristic_name(&self, mapping: &RegisterMapping) -> String {
        // Use data type + optional unit for characteristic name
        let type_name = format!("{:?}", mapping.data_type);
        if let Some(ref unit) = mapping.unit {
            format!("{}{}", type_name, self.sanitize_name(unit))
        } else {
            type_name
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

        // Ensure starts with uppercase letter
        if result.is_empty() {
            return "Unknown".to_string();
        }

        result
    }

    /// Map ModbusDataType to SAMM characteristic type and XSD datatype
    fn data_type_mapping(&self, data_type: ModbusDataType) -> (&'static str, &'static str) {
        match data_type {
            // Boolean types -> samm-c:Boolean
            ModbusDataType::Bit(_) => ("samm-c:Boolean", "xsd:boolean"),
            // Integer types -> samm:Characteristic
            ModbusDataType::Int16 => ("samm:Characteristic", "xsd:short"),
            ModbusDataType::Uint16 => ("samm:Characteristic", "xsd:unsignedShort"),
            ModbusDataType::Int32 => ("samm:Characteristic", "xsd:int"),
            ModbusDataType::Uint32 => ("samm:Characteristic", "xsd:unsignedInt"),
            // Float types -> samm-c:Measurement
            ModbusDataType::Float32 => ("samm-c:Measurement", "xsd:float"),
            ModbusDataType::Float64 => ("samm-c:Measurement", "xsd:double"),
            // String type
            ModbusDataType::String(_) => ("samm:Characteristic", "xsd:string"),
        }
    }

    /// Map unit string to SAMM/QUDT unit
    fn map_unit(&self, unit: &str) -> String {
        // Common unit mappings
        let unit_upper = unit.to_uppercase();
        match unit_upper.as_str() {
            // Temperature
            "CEL" | "CELSIUS" | "°C" => "degreeCelsius".to_string(),
            "FAH" | "FAHRENHEIT" | "°F" => "degreeFahrenheit".to_string(),
            "K" | "KELVIN" => "kelvin".to_string(),
            // Pressure
            "PA" | "PASCAL" => "pascal".to_string(),
            "BAR" => "bar".to_string(),
            "PSI" => "poundForcePerSquareInch".to_string(),
            "HPA" => "hectopascal".to_string(),
            // Electrical
            "V" | "VOLT" | "VOLTS" => "volt".to_string(),
            "A" | "AMP" | "AMPERE" | "AMPS" => "ampere".to_string(),
            "W" | "WATT" | "WATTS" => "watt".to_string(),
            "KW" | "KILOWATT" => "kilowatt".to_string(),
            "KWH" | "KILOWATTHOUR" => "kilowattHour".to_string(),
            "OHM" | "OHMS" => "ohm".to_string(),
            "HZ" | "HERTZ" => "hertz".to_string(),
            // Flow/Speed
            "M/S" | "MPS" => "metrePerSecond".to_string(),
            "RPM" => "revolutionPerMinute".to_string(),
            "L/MIN" | "LPM" => "litrePerMinute".to_string(),
            "M3/H" => "cubicMetrePerHour".to_string(),
            // Mass/Weight
            "KG" | "KILOGRAM" => "kilogram".to_string(),
            "G" | "GRAM" => "gram".to_string(),
            "TON" | "TONNE" => "tonne".to_string(),
            // Length
            "M" | "METER" | "METRE" => "metre".to_string(),
            "CM" | "CENTIMETER" => "centimetre".to_string(),
            "MM" | "MILLIMETER" => "millimetre".to_string(),
            // Percentage/Ratio
            "%" | "PERCENT" | "PERCENTAGE" => "percent".to_string(),
            // Time
            "S" | "SEC" | "SECOND" => "second".to_string(),
            "MS" | "MILLISECOND" => "millisecond".to_string(),
            "MIN" | "MINUTE" => "minute".to_string(),
            "H" | "HR" | "HOUR" => "hour".to_string(),
            // Default: use as-is (might be a valid QUDT unit)
            _ => self.sanitize_name(unit),
        }
    }

    /// Generate aspect model and save to file
    pub fn save_to_file(
        &self,
        register_map: &RegisterMap,
        path: impl AsRef<std::path::Path>,
    ) -> ModbusResult<()> {
        let ttl = self.generate_aspect_model(register_map);
        std::fs::write(path, ttl).map_err(|e| {
            crate::error::ModbusError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to write SAMM model: {}", e),
            ))
        })
    }
}

/// Generate SAMM-compliant property IRI from predicate
pub fn to_samm_property_iri(predicate: &str) -> String {
    // If it's already a full IRI, extract the local name
    if let Some(pos) = predicate.rfind('#') {
        predicate[pos + 1..].to_string()
    } else if let Some(pos) = predicate.rfind('/') {
        predicate[pos + 1..].to_string()
    } else {
        predicate.to_string()
    }
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

/// Validate a register map for SAMM compatibility
pub fn validate_for_samm(register_map: &RegisterMap) -> SammValidationResult {
    let mut result = SammValidationResult {
        valid: true,
        warnings: Vec::new(),
        errors: Vec::new(),
    };

    // Check for empty register map
    if register_map.registers.is_empty() {
        result
            .errors
            .push("Register map has no registers".to_string());
        result.valid = false;
    }

    // Check each register
    for mapping in &register_map.registers {
        // Check for missing name
        if mapping.name.is_none() {
            result.warnings.push(format!(
                "Register at address {} has no name (will use address-based name)",
                mapping.address
            ));
        }

        // Check for missing unit on measurement types
        if matches!(
            mapping.data_type,
            ModbusDataType::Float32 | ModbusDataType::Float64
        ) && mapping.unit.is_none()
        {
            result.warnings.push(format!(
                "Float register '{}' at address {} has no unit",
                mapping.name.as_deref().unwrap_or("unnamed"),
                mapping.address
            ));
        }

        // Check for invalid predicate IRIs
        if !mapping.predicate.starts_with("http://") && !mapping.predicate.starts_with("urn:") {
            result.warnings.push(format!(
                "Register at address {} has non-standard predicate IRI: {}",
                mapping.address, mapping.predicate
            ));
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_samm_generator_creation() {
        let gen = SammGenerator::new("1.0.0", "http://test.org/");
        assert_eq!(gen.version, "1.0.0");
        assert_eq!(gen.namespace, "http://test.org/");
    }

    #[test]
    fn test_sanitize_name() {
        let gen = SammGenerator::default();
        assert_eq!(gen.sanitize_name("temperature"), "Temperature");
        assert_eq!(gen.sanitize_name("motor_speed"), "MotorSpeed");
        assert_eq!(gen.sanitize_name("pump-1-flow"), "Pump1Flow");
        assert_eq!(gen.sanitize_name("PLC001"), "PLC001");
    }

    #[test]
    fn test_data_type_mapping() {
        let gen = SammGenerator::default();

        let (char_type, xsd) = gen.data_type_mapping(ModbusDataType::Float32);
        assert_eq!(char_type, "samm-c:Measurement");
        assert_eq!(xsd, "xsd:float");

        let (char_type, xsd) = gen.data_type_mapping(ModbusDataType::Uint16);
        assert_eq!(char_type, "samm:Characteristic");
        assert_eq!(xsd, "xsd:unsignedShort");

        let (char_type, xsd) = gen.data_type_mapping(ModbusDataType::Bit(0));
        assert_eq!(char_type, "samm-c:Boolean");
        assert_eq!(xsd, "xsd:boolean");
    }

    #[test]
    fn test_unit_mapping() {
        let gen = SammGenerator::default();

        assert_eq!(gen.map_unit("CEL"), "degreeCelsius");
        assert_eq!(gen.map_unit("°C"), "degreeCelsius");
        assert_eq!(gen.map_unit("V"), "volt");
        assert_eq!(gen.map_unit("kW"), "kilowatt");
        assert_eq!(gen.map_unit("%"), "percent");
        assert_eq!(gen.map_unit("RPM"), "revolutionPerMinute");
    }

    #[test]
    fn test_generate_aspect_model() {
        use super::super::RegisterMapping;

        let mut register_map = RegisterMap::new("plc001", "http://factory.example.com/device");
        register_map.add_register(
            RegisterMapping::new(
                0,
                ModbusDataType::Float32,
                "http://factory.example.com/property/temperature",
            )
            .with_name("Temperature")
            .with_unit("CEL"),
        );
        register_map.add_register(
            RegisterMapping::new(
                2,
                ModbusDataType::Uint16,
                "http://factory.example.com/property/status",
            )
            .with_name("Status"),
        );

        let gen = SammGenerator::new("1.0.0", "http://factory.example.com/samm/");
        let ttl = gen.generate_aspect_model(&register_map);

        // Verify content
        assert!(ttl.contains("@prefix samm:"));
        assert!(ttl.contains(":Plc001Aspect a samm:Aspect"));
        assert!(ttl.contains(":TemperatureProperty a samm:Property"));
        assert!(ttl.contains(":StatusProperty a samm:Property"));
        assert!(ttl.contains("samm-c:Measurement"));
        assert!(ttl.contains("unit:degreeCelsius"));
    }

    #[test]
    fn test_validate_for_samm() {
        let mut register_map = RegisterMap::new("test", "http://test.org/");

        // Empty map should fail
        let result = validate_for_samm(&register_map);
        assert!(!result.valid);
        assert!(!result.errors.is_empty());

        // Add a register without name
        register_map.add_register(RegisterMapping::new(
            0,
            ModbusDataType::Float32,
            "http://test.org/temp",
        ));

        let result = validate_for_samm(&register_map);
        assert!(result.valid);
        assert!(!result.warnings.is_empty()); // Warning for no name and no unit
    }

    #[test]
    fn test_to_samm_property_iri() {
        assert_eq!(
            to_samm_property_iri("http://example.org/property#temperature"),
            "temperature"
        );
        assert_eq!(
            to_samm_property_iri("http://example.org/property/temperature"),
            "temperature"
        );
        assert_eq!(to_samm_property_iri("temperature"), "temperature");
    }

    #[test]
    fn test_operations_generation() {
        use super::super::RegisterMapping;

        let mut register_map = RegisterMap::new("motor", "http://example.org/");

        // Holding register (writable)
        let mut holding =
            RegisterMapping::new(0, ModbusDataType::Uint16, "http://example.org/setpoint");
        holding = holding.with_name("Setpoint");
        holding.register_type = RegisterType::Holding;
        register_map.add_register(holding);

        // Input register (read-only)
        let mut input =
            RegisterMapping::new(100, ModbusDataType::Uint16, "http://example.org/actual");
        input = input.with_name("Actual");
        input.register_type = RegisterType::Input;
        register_map.add_register(input);

        let gen = SammGenerator::new("1.0.0", "http://example.org/samm/").with_operations(true);
        let ttl = gen.generate_aspect_model(&register_map);

        // Should have operation for holding register
        assert!(ttl.contains(":writeSetpointOperation a samm:Operation"));
        // Should NOT have operation for input register
        assert!(!ttl.contains(":writeActualOperation"));
    }
}
