//! RDF triple generation from Modbus register data
//!
//! Converts Modbus register values to RDF triples with proper typing
//! and W3C PROV-O provenance tracking.

use crate::error::ModbusResult;
use crate::mapping::{decode_registers, ModbusValue, RegisterMap, RegisterMapping, RegisterType};
use chrono::{DateTime, Utc};
use oxirs_core::model::{Literal, NamedNode, Triple};
use std::collections::HashMap;

/// Well-known namespace IRIs
pub mod ns {
    /// W3C PROV-O ontology
    pub const PROV: &str = "http://www.w3.org/ns/prov#";
    /// QUDT units ontology
    pub const QUDT: &str = "http://qudt.org/schema/qudt/";
    /// QUDT units vocabulary
    pub const QUDT_UNIT: &str = "http://qudt.org/vocab/unit/";
    /// XSD datatypes
    pub const XSD: &str = "http://www.w3.org/2001/XMLSchema#";
    /// RDF namespace
    pub const RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
    /// RDFS namespace
    pub const RDFS: &str = "http://www.w3.org/2000/01/rdf-schema#";
}

/// Generated triple with optional provenance
#[derive(Debug, Clone)]
pub struct GeneratedTriple {
    /// The main data triple
    pub triple: Triple,
    /// Timestamp of generation
    pub timestamp: DateTime<Utc>,
    /// Optional unit IRI
    pub unit: Option<String>,
}

impl GeneratedTriple {
    /// Generate provenance triples (prov:generatedAtTime)
    pub fn provenance_triples(&self) -> Vec<Triple> {
        use oxirs_core::model::Subject;

        let mut triples = Vec::new();

        // Extract subject as NamedNode
        let subject_node = match self.triple.subject() {
            Subject::NamedNode(n) => Some(n.clone()),
            _ => None,
        };

        // Add timestamp triple
        if let Some(ref subject) = subject_node {
            let prov_time = format!("{}generatedAtTime", ns::PROV);
            if let Ok(pred) = NamedNode::new(&prov_time) {
                let lit = Literal::new_typed(
                    self.timestamp.to_rfc3339(),
                    NamedNode::new(format!("{}dateTime", ns::XSD))
                        .expect("XSD dateTime IRI should be valid"),
                );
                triples.push(Triple::new(subject.clone(), pred, lit));
            }
        }

        // Add unit triple if present
        if let Some(ref unit) = self.unit {
            if let Some(ref subject) = subject_node {
                let qudt_unit = format!("{}unit", ns::QUDT);
                if let (Ok(pred), Ok(unit_node)) =
                    (NamedNode::new(&qudt_unit), NamedNode::new(unit))
                {
                    triples.push(Triple::new(subject.clone(), pred, unit_node));
                }
            }
        }

        triples
    }
}

/// Modbus to RDF triple generator
pub struct ModbusTripleGenerator {
    /// Register map configuration
    register_map: RegisterMap,
    /// Previous values for change detection
    previous_values: HashMap<(u16, RegisterType), f64>,
}

impl ModbusTripleGenerator {
    /// Create a new triple generator
    pub fn new(register_map: RegisterMap) -> Self {
        Self {
            register_map,
            previous_values: HashMap::new(),
        }
    }

    /// Get the register map
    pub fn register_map(&self) -> &RegisterMap {
        &self.register_map
    }

    /// Generate triples from register values
    ///
    /// # Arguments
    ///
    /// * `values` - Map of register address to raw u16 values
    /// * `register_type` - Type of registers (holding, input, etc.)
    /// * `timestamp` - Timestamp of the reading
    ///
    /// # Returns
    ///
    /// Vector of generated triples
    pub fn generate_triples(
        &mut self,
        values: &HashMap<u16, Vec<u16>>,
        register_type: RegisterType,
        timestamp: DateTime<Utc>,
    ) -> ModbusResult<Vec<GeneratedTriple>> {
        let mut triples = Vec::new();
        let subject_iri = self.register_map.subject_iri();

        for mapping in self.register_map.registers.iter() {
            if mapping.register_type != register_type {
                continue;
            }

            // Get register values for this mapping
            if let Some(regs) = values.get(&mapping.address) {
                // Decode value
                let raw_value = decode_registers(regs, mapping.data_type)?;

                // Apply scaling
                let scaled_value = if let Some(raw_f64) = raw_value.as_f64() {
                    let scaling = mapping.get_scaling();
                    let scaled = scaling.apply(raw_f64);

                    // Check deadband
                    let key = (mapping.address, mapping.register_type);
                    if let Some(&prev) = self.previous_values.get(&key) {
                        if let Some(deadband) = mapping.deadband {
                            if (scaled - prev).abs() < deadband {
                                continue; // Skip - within deadband
                            }
                        }
                    }
                    self.previous_values.insert(key, scaled);

                    ModbusValue::Float(scaled)
                } else {
                    raw_value
                };

                // Handle enum mapping
                let final_value = if let Some(ref enum_map) = mapping.enum_values {
                    if let ModbusValue::Uint(v) = scaled_value {
                        if let Some(label) = enum_map.get_label(v as u16) {
                            ModbusValue::String(label.to_string())
                        } else {
                            scaled_value
                        }
                    } else {
                        scaled_value
                    }
                } else {
                    scaled_value
                };

                // Create triple
                if let Some(triple) = self.create_triple(&subject_iri, mapping, &final_value)? {
                    triples.push(GeneratedTriple {
                        triple,
                        timestamp,
                        unit: mapping.unit.clone(),
                    });
                }
            }
        }

        Ok(triples)
    }

    /// Generate triples from flat register array
    ///
    /// # Arguments
    ///
    /// * `start_address` - Starting register address
    /// * `values` - Contiguous register values
    /// * `register_type` - Type of registers
    /// * `timestamp` - Timestamp of the reading
    pub fn generate_from_array(
        &mut self,
        start_address: u16,
        values: &[u16],
        register_type: RegisterType,
        timestamp: DateTime<Utc>,
    ) -> ModbusResult<Vec<GeneratedTriple>> {
        // Convert to HashMap
        let mut map = HashMap::new();
        for (i, &val) in values.iter().enumerate() {
            let addr = start_address + i as u16;
            // Group consecutive registers for multi-register types
            map.insert(addr, vec![val]);
        }

        // For multi-register types (FLOAT32, INT32, etc.), we need to
        // combine consecutive registers
        for mapping in self.register_map.registers.iter() {
            if mapping.register_type != register_type {
                continue;
            }
            let count = mapping.data_type.register_count();
            if count > 1 {
                let addr = mapping.address;
                let mut regs = Vec::with_capacity(count);
                for i in 0..count {
                    if let Some(v) = map.get(&(addr + i as u16)) {
                        regs.push(v[0]);
                    }
                }
                if regs.len() == count {
                    map.insert(addr, regs);
                }
            }
        }

        self.generate_triples(&map, register_type, timestamp)
    }

    /// Create a single triple from a mapping and value
    fn create_triple(
        &self,
        subject_iri: &str,
        mapping: &RegisterMapping,
        value: &ModbusValue,
    ) -> ModbusResult<Option<Triple>> {
        // Parse subject and predicate IRIs
        let subject = match NamedNode::new(subject_iri) {
            Ok(n) => n,
            Err(_) => return Ok(None),
        };

        let predicate = match NamedNode::new(&mapping.predicate) {
            Ok(n) => n,
            Err(_) => return Ok(None),
        };

        // Create typed literal
        let literal = match value {
            ModbusValue::Int(v) => {
                let datatype = NamedNode::new(mapping.data_type.xsd_datatype())
                    .expect("XSD datatype IRI should be valid");
                Literal::new_typed(v.to_string(), datatype)
            }
            ModbusValue::Uint(v) => {
                let datatype = NamedNode::new(mapping.data_type.xsd_datatype())
                    .expect("XSD datatype IRI should be valid");
                Literal::new_typed(v.to_string(), datatype)
            }
            ModbusValue::Float(v) => {
                let datatype = NamedNode::new(format!("{}float", ns::XSD))
                    .expect("XSD float IRI should be valid");
                Literal::new_typed(format!("{}", v), datatype)
            }
            ModbusValue::Bool(v) => {
                let datatype = NamedNode::new(format!("{}boolean", ns::XSD))
                    .expect("XSD boolean IRI should be valid");
                Literal::new_typed(if *v { "true" } else { "false" }, datatype)
            }
            ModbusValue::String(s) => {
                let datatype = NamedNode::new(format!("{}string", ns::XSD))
                    .expect("XSD string IRI should be valid");
                Literal::new_typed(s.clone(), datatype)
            }
        };

        Ok(Some(Triple::new(subject, predicate, literal)))
    }

    /// Reset change detection state
    pub fn reset_change_detection(&mut self) {
        self.previous_values.clear();
    }
}

/// Builder for common unit IRIs
pub struct QudtUnit;

impl QudtUnit {
    /// Celsius
    pub fn celsius() -> String {
        format!("{}DEG_C", ns::QUDT_UNIT)
    }

    /// Fahrenheit
    pub fn fahrenheit() -> String {
        format!("{}DEG_F", ns::QUDT_UNIT)
    }

    /// Kelvin
    pub fn kelvin() -> String {
        format!("{}K", ns::QUDT_UNIT)
    }

    /// Percent
    pub fn percent() -> String {
        format!("{}PERCENT", ns::QUDT_UNIT)
    }

    /// Pascal
    pub fn pascal() -> String {
        format!("{}PA", ns::QUDT_UNIT)
    }

    /// Bar
    pub fn bar() -> String {
        format!("{}BAR", ns::QUDT_UNIT)
    }

    /// Volt
    pub fn volt() -> String {
        format!("{}V", ns::QUDT_UNIT)
    }

    /// Ampere
    pub fn ampere() -> String {
        format!("{}A", ns::QUDT_UNIT)
    }

    /// Watt
    pub fn watt() -> String {
        format!("{}W", ns::QUDT_UNIT)
    }

    /// Kilowatt
    pub fn kilowatt() -> String {
        format!("{}KiloW", ns::QUDT_UNIT)
    }

    /// Hertz
    pub fn hertz() -> String {
        format!("{}HZ", ns::QUDT_UNIT)
    }

    /// Revolution per minute
    pub fn rpm() -> String {
        format!("{}REV-PER-MIN", ns::QUDT_UNIT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::{ModbusDataType, RegisterMap, RegisterMapping};

    fn create_test_map() -> RegisterMap {
        let mut map = RegisterMap::new("plc001", "http://factory.example.com/device");

        map.add_register(
            RegisterMapping::new(
                0,
                ModbusDataType::Float32,
                "http://factory.example.com/property/temperature",
            )
            .with_name("Temperature")
            .with_unit(QudtUnit::celsius())
            .with_scaling(0.1, 0.0),
        );

        map.add_register(
            RegisterMapping::new(
                2,
                ModbusDataType::Uint16,
                "http://factory.example.com/property/status",
            )
            .with_name("Status"),
        );

        map
    }

    #[test]
    fn test_generate_triples() {
        let map = create_test_map();
        let mut generator = ModbusTripleGenerator::new(map);

        // IEEE 754: 22.5 = 0x41B40000
        let mut values = HashMap::new();
        values.insert(0u16, vec![0x41B4, 0x0000]); // FLOAT32 = 22.5
        values.insert(2u16, vec![1]); // Status = 1

        let timestamp = Utc::now();
        let triples = generator
            .generate_triples(&values, RegisterType::Holding, timestamp)
            .unwrap();

        assert_eq!(triples.len(), 2);

        // Check temperature triple
        let temp_triple = &triples[0];
        let pred_contains_temp = match temp_triple.triple.predicate() {
            oxirs_core::model::Predicate::NamedNode(n) => n.as_str().contains("temperature"),
            _ => false,
        };
        assert!(pred_contains_temp);
    }

    #[test]
    fn test_deadband_filtering() {
        let mut map = RegisterMap::new("plc001", "http://factory.example.com/device");
        map.add_register(
            RegisterMapping::new(
                0,
                ModbusDataType::Uint16,
                "http://factory.example.com/property/value",
            )
            .with_deadband(10.0),
        );

        let mut generator = ModbusTripleGenerator::new(map);
        let timestamp = Utc::now();

        // First reading: 100
        let mut values = HashMap::new();
        values.insert(0u16, vec![100]);
        let triples1 = generator
            .generate_triples(&values, RegisterType::Holding, timestamp)
            .unwrap();
        assert_eq!(triples1.len(), 1); // First reading always generates

        // Second reading: 105 (within deadband)
        values.insert(0u16, vec![105]);
        let triples2 = generator
            .generate_triples(&values, RegisterType::Holding, timestamp)
            .unwrap();
        assert_eq!(triples2.len(), 0); // Filtered by deadband

        // Third reading: 115 (outside deadband)
        values.insert(0u16, vec![115]);
        let triples3 = generator
            .generate_triples(&values, RegisterType::Holding, timestamp)
            .unwrap();
        assert_eq!(triples3.len(), 1); // Generates new triple
    }

    #[test]
    fn test_provenance_triples() {
        let map = create_test_map();
        let mut generator = ModbusTripleGenerator::new(map);

        let mut values = HashMap::new();
        values.insert(0u16, vec![0x41B4, 0x0000]);

        let timestamp = Utc::now();
        let triples = generator
            .generate_triples(&values, RegisterType::Holding, timestamp)
            .unwrap();

        let prov_triples = triples[0].provenance_triples();

        // Should have timestamp and unit triples
        assert!(!prov_triples.is_empty());

        // Check timestamp triple
        let has_timestamp = prov_triples.iter().any(|t| match t.predicate() {
            oxirs_core::model::Predicate::NamedNode(n) => n.as_str().contains("generatedAtTime"),
            _ => false,
        });
        assert!(has_timestamp);
    }

    #[test]
    fn test_qudt_units() {
        assert!(QudtUnit::celsius().contains("DEG_C"));
        assert!(QudtUnit::volt().contains("/V"));
        assert!(QudtUnit::watt().contains("/W"));
    }

    #[test]
    fn test_generate_from_array() {
        let mut map = RegisterMap::new("plc001", "http://factory.example.com/device");
        map.add_register(RegisterMapping::new(
            0,
            ModbusDataType::Uint16,
            "http://factory.example.com/property/reg0",
        ));
        map.add_register(RegisterMapping::new(
            1,
            ModbusDataType::Uint16,
            "http://factory.example.com/property/reg1",
        ));

        let mut generator = ModbusTripleGenerator::new(map);
        let timestamp = Utc::now();

        let values = vec![100, 200, 300];
        let triples = generator
            .generate_from_array(0, &values, RegisterType::Holding, timestamp)
            .unwrap();

        assert_eq!(triples.len(), 2); // Only mapped registers
    }
}
