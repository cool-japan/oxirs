//! SAMM (Semantic Aspect Meta Model) Aspect integration for Modbus devices.
//!
//! Provides structured semantic mapping between raw Modbus register values and
//! SAMM-modelled device aspects. Supports:
//! - `ModbusAspect` — device metadata + register mappings
//! - `ModbusSammMapper` — converts register arrays to SAMM-structured JSON
//! - `ModbusRdfMapper` — generates SAREF4ENER-based RDF triples from readings
//!
//! # Example
//!
//! ```
//! use oxirs_modbus::samm::{
//!     DeviceInfo, RegisterMapping, RegisterDataType,
//!     ModbusSammMapper, ModbusRdfMapper,
//! };
//!
//! let device = DeviceInfo {
//!     manufacturer: "Siemens".to_string(),
//!     model: "S7-1200".to_string(),
//!     firmware: "4.5".to_string(),
//!     unit_id: 1,
//! };
//!
//! let mappings = vec![
//!     RegisterMapping {
//!         address: 0,
//!         name: "temperature".to_string(),
//!         data_type: RegisterDataType::Float32,
//!         scale: 0.1,
//!         unit: "CEL".to_string(),
//!     },
//! ];
//!
//! let values = vec![0x4248u16, 0x0000]; // 50.0 °C in IEEE 754
//! let mapper = ModbusSammMapper::new();
//! let json = mapper.map_holding_registers(&values, &mappings);
//! ```

use serde_json::{json, Value};

/// Information identifying a Modbus device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Device manufacturer name.
    pub manufacturer: String,
    /// Device model identifier.
    pub model: String,
    /// Firmware version string.
    pub firmware: String,
    /// Modbus unit identifier (1–247).
    pub unit_id: u8,
}

/// Semantic data type of a Modbus register mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterDataType {
    /// Unsigned 16-bit integer (1 register).
    UInt16,
    /// Signed 16-bit integer (1 register).
    Int16,
    /// Unsigned 32-bit integer (2 registers, big-endian).
    UInt32,
    /// Signed 32-bit integer (2 registers, big-endian).
    Int32,
    /// IEEE 754 single-precision float (2 registers, big-endian).
    Float32,
    /// IEEE 754 double-precision float (4 registers, big-endian).
    Float64,
    /// Single boolean flag extracted from a register bit 0.
    Bool,
}

impl RegisterDataType {
    /// How many 16-bit registers this type occupies.
    pub fn register_count(self) -> usize {
        match self {
            Self::UInt16 | Self::Int16 | Self::Bool => 1,
            Self::UInt32 | Self::Int32 | Self::Float32 => 2,
            Self::Float64 => 4,
        }
    }

    /// XSD datatype IRI.
    pub fn xsd_type(self) -> &'static str {
        match self {
            Self::UInt16 => "xsd:unsignedShort",
            Self::Int16 => "xsd:short",
            Self::UInt32 => "xsd:unsignedInt",
            Self::Int32 => "xsd:int",
            Self::Float32 | Self::Float64 => "xsd:float",
            Self::Bool => "xsd:boolean",
        }
    }

    /// SAMM characteristic name.
    pub fn samm_characteristic(self) -> &'static str {
        match self {
            Self::UInt16 | Self::UInt32 => "samm-c:Measurement",
            Self::Int16 | Self::Int32 => "samm-c:Measurement",
            Self::Float32 | Self::Float64 => "samm-c:Measurement",
            Self::Bool => "samm-c:SingleEntity",
        }
    }
}

/// Describes the mapping of a single Modbus register (or register group) to a
/// named semantic property.
#[derive(Debug, Clone, PartialEq)]
pub struct RegisterMapping {
    /// Zero-based register address in the Modbus address space.
    pub address: u16,
    /// Semantic property name (snake_case preferred).
    pub name: String,
    /// Data type used for decoding the register value(s).
    pub data_type: RegisterDataType,
    /// Multiplication scale applied to the raw decoded value.
    pub scale: f64,
    /// Physical unit string (QUDT or plain label, e.g. `"CEL"`, `"Hz"`).
    pub unit: String,
}

impl RegisterMapping {
    /// Decode the raw register value(s) at `address` from `values`, applying scale.
    ///
    /// Returns `None` if there are insufficient registers at the given offset.
    pub fn decode_value(&self, values: &[u16], base_address: u16) -> Option<f64> {
        let offset = self.address.checked_sub(base_address)? as usize;
        let needed = self.data_type.register_count();
        if offset + needed > values.len() {
            return None;
        }
        let raw = decode_raw(&values[offset..offset + needed], self.data_type)?;
        Some(raw * self.scale)
    }
}

/// Decode a raw numeric value from a register slice according to data type.
fn decode_raw(regs: &[u16], dtype: RegisterDataType) -> Option<f64> {
    match dtype {
        RegisterDataType::UInt16 => Some(f64::from(regs[0])),
        RegisterDataType::Int16 => Some(f64::from(regs[0] as i16)),
        RegisterDataType::UInt32 => {
            if regs.len() < 2 {
                return None;
            }
            let v = (u32::from(regs[0]) << 16) | u32::from(regs[1]);
            Some(f64::from(v))
        }
        RegisterDataType::Int32 => {
            if regs.len() < 2 {
                return None;
            }
            let v = ((regs[0] as u32) << 16) | u32::from(regs[1]);
            Some(f64::from(v as i32))
        }
        RegisterDataType::Float32 => {
            if regs.len() < 2 {
                return None;
            }
            let bits = (u32::from(regs[0]) << 16) | u32::from(regs[1]);
            Some(f64::from(f32::from_bits(bits)))
        }
        RegisterDataType::Float64 => {
            if regs.len() < 4 {
                return None;
            }
            let hi = (u64::from(regs[0]) << 48) | (u64::from(regs[1]) << 32);
            let lo = (u64::from(regs[2]) << 16) | u64::from(regs[3]);
            Some(f64::from_bits(hi | lo))
        }
        RegisterDataType::Bool => Some(if regs[0] != 0 { 1.0 } else { 0.0 }),
    }
}

// ── ModbusSammMapper ──────────────────────────────────────────────────────

/// Maps arrays of raw Modbus register values to SAMM-structured JSON objects.
#[derive(Debug, Default, Clone)]
pub struct ModbusSammMapper {
    /// Base address used when computing offsets into the register array.
    ///
    /// Defaults to 0 (i.e., the values slice starts at address 0).
    pub base_address: u16,
}

impl ModbusSammMapper {
    /// Create a mapper with `base_address` 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a mapper with a custom base address.
    pub fn with_base(base_address: u16) -> Self {
        Self { base_address }
    }

    /// Map a slice of raw holding-register values to a SAMM JSON properties object.
    ///
    /// Each mapping that can be decoded is added as a key/value entry. Missing
    /// or out-of-range addresses are silently skipped.
    pub fn map_holding_registers(&self, values: &[u16], mappings: &[RegisterMapping]) -> Value {
        self.build_properties(values, mappings, "HoldingRegister")
    }

    /// Map a slice of raw input-register values to a SAMM JSON properties object.
    pub fn map_input_registers(&self, values: &[u16], mappings: &[RegisterMapping]) -> Value {
        self.build_properties(values, mappings, "InputRegister")
    }

    fn build_properties(
        &self,
        values: &[u16],
        mappings: &[RegisterMapping],
        register_type: &str,
    ) -> Value {
        let mut props = serde_json::Map::new();
        for m in mappings {
            if let Some(v) = m.decode_value(values, self.base_address) {
                let entry = json!({
                    "value": v,
                    "unit": m.unit,
                    "dataType": format!("{:?}", m.data_type),
                    "registerType": register_type,
                    "address": m.address,
                    "scale": m.scale,
                });
                props.insert(m.name.clone(), entry);
            }
        }
        Value::Object(props)
    }

    /// Wrap a properties object in a full SAMM Aspect JSON envelope.
    pub fn to_samm_json(&self, device: &DeviceInfo, properties: &Value) -> Value {
        json!({
            "@context": {
                "samm": "urn:samm:org.eclipse.esmf.samm:meta-model:2.1.0#",
                "saref": "https://saref.etsi.org/core/",
                "xsd": "http://www.w3.org/2001/XMLSchema#"
            },
            "@type": "samm:Aspect",
            "aspectName": format!("ModbusDevice_{}", device.model.replace(' ', "_")),
            "device": {
                "manufacturer": device.manufacturer,
                "model": device.model,
                "firmware": device.firmware,
                "unitId": device.unit_id
            },
            "properties": properties
        })
    }
}

// ── RDF triple representation ─────────────────────────────────────────────

/// A minimal RDF triple used by `ModbusRdfMapper`.
#[derive(Debug, Clone, PartialEq)]
pub struct Triple {
    /// Subject IRI.
    pub subject: String,
    /// Predicate IRI.
    pub predicate: String,
    /// Object: either an IRI or a literal with optional datatype.
    pub object: TripleObject,
}

/// Object component of a `Triple`.
#[derive(Debug, Clone, PartialEq)]
pub enum TripleObject {
    /// Named node (IRI).
    Iri(String),
    /// Literal value with a datatype IRI.
    Literal {
        /// Lexical form.
        value: String,
        /// Datatype IRI (XSD or other).
        datatype: String,
    },
}

impl TripleObject {
    /// Create a plain IRI object.
    pub fn iri(s: impl Into<String>) -> Self {
        Self::Iri(s.into())
    }

    /// Create an XSD double literal.
    pub fn xsd_double(v: f64) -> Self {
        Self::Literal {
            value: v.to_string(),
            datatype: "http://www.w3.org/2001/XMLSchema#double".to_string(),
        }
    }

    /// Create an XSD string literal.
    pub fn xsd_string(s: impl Into<String>) -> Self {
        Self::Literal {
            value: s.into(),
            datatype: "http://www.w3.org/2001/XMLSchema#string".to_string(),
        }
    }
}

// ── ModbusRdfMapper ───────────────────────────────────────────────────────

/// Generates SAREF4ENER-based RDF triples from Modbus readings.
///
/// Uses the SAREF ontology predicates:
/// - `saref:Measurement`
/// - `saref:hasValue`
/// - `saref:isMeasuredIn`
#[derive(Debug, Clone)]
pub struct ModbusRdfMapper {
    /// Base IRI for generated device resources.
    pub base_iri: String,
}

impl ModbusRdfMapper {
    // Ontology namespaces
    const SAREF: &'static str = "https://saref.etsi.org/core/";
    const RDF_TYPE: &'static str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    const RDFS_LABEL: &'static str = "http://www.w3.org/2000/01/rdf-schema#label";
    const XSD: &'static str = "http://www.w3.org/2001/XMLSchema#";

    /// Create a new mapper with the given base IRI.
    pub fn new(base_iri: impl Into<String>) -> Self {
        Self {
            base_iri: base_iri.into(),
        }
    }

    /// Generate RDF triples for a set of register readings.
    ///
    /// For each `(mapping, value)` pair, emits:
    /// - A `saref:Measurement` instance
    /// - `saref:hasValue` → numeric literal
    /// - `saref:isMeasuredIn` → unit IRI
    /// - `rdfs:label` → property name
    /// - Device provenance: `saref:isPropertyOf` → device IRI
    pub fn to_rdf_triples(
        &self,
        device: &DeviceInfo,
        readings: &[(RegisterMapping, f64)],
    ) -> Vec<Triple> {
        let device_iri = format!(
            "{}device/{}/{}",
            self.base_iri,
            urlify(&device.manufacturer),
            urlify(&device.model)
        );
        let unit_id_iri = format!("{}unit/{}", device_iri, device.unit_id);

        let mut triples = Vec::with_capacity(readings.len() * 6 + 4);

        // Device triples
        triples.push(Triple {
            subject: device_iri.clone(),
            predicate: Self::RDF_TYPE.to_string(),
            object: TripleObject::iri(format!("{}Device", Self::SAREF)),
        });
        triples.push(Triple {
            subject: device_iri.clone(),
            predicate: format!("{}hasManufacturer", Self::SAREF),
            object: TripleObject::xsd_string(&device.manufacturer),
        });
        triples.push(Triple {
            subject: device_iri.clone(),
            predicate: format!("{}hasModel", Self::SAREF),
            object: TripleObject::xsd_string(&device.model),
        });
        triples.push(Triple {
            subject: unit_id_iri.clone(),
            predicate: format!("{}{}unitId", Self::XSD, ""),
            object: TripleObject::Literal {
                value: device.unit_id.to_string(),
                datatype: format!("{}unsignedByte", Self::XSD),
            },
        });

        // Measurement triples
        for (idx, (mapping, value)) in readings.iter().enumerate() {
            let measurement_iri = format!(
                "{}measurement/{}/{}",
                self.base_iri,
                urlify(&device.model),
                idx
            );

            triples.push(Triple {
                subject: measurement_iri.clone(),
                predicate: Self::RDF_TYPE.to_string(),
                object: TripleObject::iri(format!("{}Measurement", Self::SAREF)),
            });
            triples.push(Triple {
                subject: measurement_iri.clone(),
                predicate: Self::RDFS_LABEL.to_string(),
                object: TripleObject::xsd_string(&mapping.name),
            });
            triples.push(Triple {
                subject: measurement_iri.clone(),
                predicate: format!("{}hasValue", Self::SAREF),
                object: TripleObject::xsd_double(*value),
            });
            triples.push(Triple {
                subject: measurement_iri.clone(),
                predicate: format!("{}isMeasuredIn", Self::SAREF),
                object: TripleObject::iri(format!(
                    "http://qudt.org/vocab/unit/{}",
                    urlify(&mapping.unit)
                )),
            });
            triples.push(Triple {
                subject: measurement_iri.clone(),
                predicate: format!("{}isPropertyOf", Self::SAREF),
                object: TripleObject::iri(device_iri.clone()),
            });
            triples.push(Triple {
                subject: measurement_iri.clone(),
                predicate: format!("{}relatesTo", Self::SAREF),
                object: TripleObject::Literal {
                    value: mapping.address.to_string(),
                    datatype: format!("{}unsignedShort", Self::XSD),
                },
            });
        }

        triples
    }

    /// Serialize triples to N-Triples format for inspection / testing.
    pub fn to_ntriples(triples: &[Triple]) -> String {
        let mut out = String::new();
        for t in triples {
            let subj = format!("<{}>", t.subject);
            let pred = format!("<{}>", t.predicate);
            let obj = match &t.object {
                TripleObject::Iri(iri) => format!("<{}>", iri),
                TripleObject::Literal { value, datatype } => {
                    format!("\"{}\"^^<{}>", value.replace('"', "\\\""), datatype)
                }
            };
            out.push_str(&format!("{subj} {pred} {obj} .\n"));
        }
        out
    }
}

/// Convert a string to a URL-safe slug.
fn urlify(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// ── tests ──────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn make_device() -> DeviceInfo {
        DeviceInfo {
            manufacturer: "TestCorp".to_string(),
            model: "TC-100".to_string(),
            firmware: "2.3.1".to_string(),
            unit_id: 5,
        }
    }

    fn temp_mapping() -> RegisterMapping {
        RegisterMapping {
            address: 0,
            name: "temperature".to_string(),
            data_type: RegisterDataType::Float32,
            scale: 0.1,
            unit: "CEL".to_string(),
        }
    }

    // ── RegisterDataType ─────────────────────────────────────────────────

    #[test]
    fn test_register_count_uint16() {
        assert_eq!(RegisterDataType::UInt16.register_count(), 1);
    }

    #[test]
    fn test_register_count_float32() {
        assert_eq!(RegisterDataType::Float32.register_count(), 2);
    }

    #[test]
    fn test_register_count_float64() {
        assert_eq!(RegisterDataType::Float64.register_count(), 4);
    }

    #[test]
    fn test_register_count_bool() {
        assert_eq!(RegisterDataType::Bool.register_count(), 1);
    }

    #[test]
    fn test_xsd_type_labels() {
        assert_eq!(RegisterDataType::UInt16.xsd_type(), "xsd:unsignedShort");
        assert_eq!(RegisterDataType::Int16.xsd_type(), "xsd:short");
        assert_eq!(RegisterDataType::Bool.xsd_type(), "xsd:boolean");
    }

    // ── RegisterMapping decode ────────────────────────────────────────────

    #[test]
    fn test_decode_uint16() {
        let m = RegisterMapping {
            address: 0,
            name: "count".to_string(),
            data_type: RegisterDataType::UInt16,
            scale: 1.0,
            unit: "".to_string(),
        };
        // value 42
        assert_eq!(m.decode_value(&[42], 0), Some(42.0));
    }

    #[test]
    fn test_decode_int16_negative() {
        let m = RegisterMapping {
            address: 0,
            name: "temp".to_string(),
            data_type: RegisterDataType::Int16,
            scale: 1.0,
            unit: "CEL".to_string(),
        };
        // -10 in two's complement u16
        let raw = (-10i16) as u16;
        assert_eq!(m.decode_value(&[raw], 0), Some(-10.0));
    }

    #[test]
    fn test_decode_float32() {
        // 50.0f32 → bits 0x42480000 → regs [0x4248, 0x0000]
        let f: f32 = 50.0;
        let bits = f.to_bits();
        let hi = (bits >> 16) as u16;
        let lo = (bits & 0xFFFF) as u16;
        let m = RegisterMapping {
            address: 0,
            name: "temperature".to_string(),
            data_type: RegisterDataType::Float32,
            scale: 1.0,
            unit: "CEL".to_string(),
        };
        let result = m.decode_value(&[hi, lo], 0).expect("decoded");
        assert!((result - 50.0).abs() < 1e-4, "got {result}");
    }

    #[test]
    fn test_decode_with_scale() {
        let m = RegisterMapping {
            address: 0,
            name: "voltage".to_string(),
            data_type: RegisterDataType::UInt16,
            scale: 0.1,
            unit: "V".to_string(),
        };
        // raw 2300 → 230.0 V
        assert!((m.decode_value(&[2300], 0).unwrap() - 230.0).abs() < 1e-9);
    }

    #[test]
    fn test_decode_with_address_offset() {
        let m = RegisterMapping {
            address: 10,
            name: "freq".to_string(),
            data_type: RegisterDataType::UInt16,
            scale: 0.01,
            unit: "Hz".to_string(),
        };
        let values = vec![0u16; 11]; // 11 registers starting at address 0
        let mut vals = values.clone();
        vals[10] = 5000; // address 10 offset = 10
        assert!((m.decode_value(&vals, 0).unwrap() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_decode_returns_none_when_insufficient_registers() {
        let m = RegisterMapping {
            address: 0,
            name: "x".to_string(),
            data_type: RegisterDataType::Float32,
            scale: 1.0,
            unit: "".to_string(),
        };
        // Only 1 register, Float32 needs 2
        assert_eq!(m.decode_value(&[0x1234], 0), None);
    }

    // ── ModbusSammMapper ─────────────────────────────────────────────────

    #[test]
    fn test_map_holding_registers_basic() {
        let mapper = ModbusSammMapper::new();
        let mappings = vec![RegisterMapping {
            address: 0,
            name: "speed".to_string(),
            data_type: RegisterDataType::UInt16,
            scale: 1.0,
            unit: "RPM".to_string(),
        }];
        let result = mapper.map_holding_registers(&[1500], &mappings);
        let speed = &result["speed"];
        assert!((speed["value"].as_f64().unwrap() - 1500.0).abs() < 1e-9);
        assert_eq!(speed["unit"].as_str().unwrap(), "RPM");
    }

    #[test]
    fn test_map_input_registers_type_label() {
        let mapper = ModbusSammMapper::new();
        let mappings = vec![RegisterMapping {
            address: 0,
            name: "pressure".to_string(),
            data_type: RegisterDataType::UInt16,
            scale: 1.0,
            unit: "PA".to_string(),
        }];
        let result = mapper.map_input_registers(&[1000], &mappings);
        assert_eq!(
            result["pressure"]["registerType"].as_str().unwrap(),
            "InputRegister"
        );
    }

    #[test]
    fn test_map_skips_out_of_range_addresses() {
        let mapper = ModbusSammMapper::new();
        let mappings = vec![RegisterMapping {
            address: 100, // out of range of values slice
            name: "phantom".to_string(),
            data_type: RegisterDataType::UInt16,
            scale: 1.0,
            unit: "".to_string(),
        }];
        let result = mapper.map_holding_registers(&[0u16; 5], &mappings);
        // "phantom" not present (address out of range)
        assert!(result.get("phantom").is_none());
    }

    #[test]
    fn test_to_samm_json_structure() {
        let mapper = ModbusSammMapper::new();
        let device = make_device();
        let props = mapper.map_holding_registers(
            &[42],
            &[RegisterMapping {
                address: 0,
                name: "x".to_string(),
                data_type: RegisterDataType::UInt16,
                scale: 1.0,
                unit: "".to_string(),
            }],
        );
        let samm_json = mapper.to_samm_json(&device, &props);
        assert_eq!(samm_json["@type"].as_str().unwrap(), "samm:Aspect");
        assert_eq!(
            samm_json["device"]["manufacturer"].as_str().unwrap(),
            "TestCorp"
        );
        assert_eq!(samm_json["device"]["unitId"].as_u64().unwrap(), 5);
    }

    #[test]
    fn test_to_samm_json_contains_properties() {
        let mapper = ModbusSammMapper::new();
        let device = make_device();
        let mappings = vec![temp_mapping()];
        // 50.0f32 float registers
        let f: f32 = 50.0;
        let bits = f.to_bits();
        let regs = vec![(bits >> 16) as u16, (bits & 0xFFFF) as u16];
        let props = mapper.map_holding_registers(&regs, &mappings);
        let samm_json = mapper.to_samm_json(&device, &props);
        // temperature should be present in properties
        assert!(samm_json["properties"]["temperature"].is_object());
    }

    // ── ModbusRdfMapper ──────────────────────────────────────────────────

    #[test]
    fn test_to_rdf_triples_count() {
        let mapper = ModbusRdfMapper::new("http://example.org/");
        let device = make_device();
        let readings = vec![(temp_mapping(), 25.0)];
        let triples = mapper.to_rdf_triples(&device, &readings);
        // 4 device triples + 6 measurement triples
        assert_eq!(triples.len(), 10);
    }

    #[test]
    fn test_to_rdf_triples_device_type() {
        let mapper = ModbusRdfMapper::new("http://example.org/");
        let device = make_device();
        let triples = mapper.to_rdf_triples(&device, &[]);
        let type_triple = triples
            .iter()
            .find(|t| t.predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
        assert!(type_triple.is_some());
        if let TripleObject::Iri(iri) = &type_triple.unwrap().object {
            assert!(iri.contains("saref.etsi.org"), "got {iri}");
        } else {
            panic!("Expected IRI object");
        }
    }

    #[test]
    fn test_to_rdf_triples_measurement_has_value() {
        let mapper = ModbusRdfMapper::new("http://example.org/");
        let device = make_device();
        let readings = vec![(temp_mapping(), 42.5)];
        let triples = mapper.to_rdf_triples(&device, &readings);
        let has_value = triples.iter().find(|t| t.predicate.contains("hasValue"));
        assert!(has_value.is_some(), "hasValue triple missing");
        if let TripleObject::Literal { value, .. } = &has_value.unwrap().object {
            assert!(value.contains("42.5"), "got {value}");
        } else {
            panic!("Expected literal");
        }
    }

    #[test]
    fn test_to_rdf_triples_unit_iri() {
        let mapper = ModbusRdfMapper::new("http://example.org/");
        let device = make_device();
        let readings = vec![(temp_mapping(), 25.0)];
        let triples = mapper.to_rdf_triples(&device, &readings);
        let unit_triple = triples
            .iter()
            .find(|t| t.predicate.contains("isMeasuredIn"));
        assert!(unit_triple.is_some());
        if let TripleObject::Iri(iri) = &unit_triple.unwrap().object {
            assert!(iri.contains("qudt.org"), "got {iri}");
            assert!(iri.contains("CEL"), "got {iri}");
        } else {
            panic!("Expected IRI");
        }
    }

    #[test]
    fn test_to_ntriples_format() {
        let triples = vec![Triple {
            subject: "http://ex.org/s".to_string(),
            predicate: "http://ex.org/p".to_string(),
            object: TripleObject::xsd_double(42.0),
        }];
        let nt = ModbusRdfMapper::to_ntriples(&triples);
        assert!(nt.contains("<http://ex.org/s>"));
        assert!(nt.contains("<http://ex.org/p>"));
        assert!(nt.ends_with(".\n"));
    }

    #[test]
    fn test_multiple_readings() {
        let mapper = ModbusRdfMapper::new("http://factory.org/");
        let device = make_device();
        let m2 = RegisterMapping {
            address: 2,
            name: "pressure".to_string(),
            data_type: RegisterDataType::UInt16,
            scale: 0.01,
            unit: "BAR".to_string(),
        };
        let readings = vec![(temp_mapping(), 20.0), (m2, 1.013)];
        let triples = mapper.to_rdf_triples(&device, &readings);
        // 4 device + 2*6 measurement
        assert_eq!(triples.len(), 16);
    }

    #[test]
    fn test_ntriples_escapes_quotes() {
        let triples = vec![Triple {
            subject: "http://ex.org/s".to_string(),
            predicate: "http://ex.org/p".to_string(),
            object: TripleObject::xsd_string("hello \"world\""),
        }];
        let nt = ModbusRdfMapper::to_ntriples(&triples);
        assert!(nt.contains("\\\""), "escaping missing in: {nt}");
    }

    #[test]
    fn test_urlify_replaces_spaces() {
        assert_eq!(urlify("Hello World"), "Hello_World");
    }

    #[test]
    fn test_bool_decode_true() {
        let m = RegisterMapping {
            address: 0,
            name: "flag".to_string(),
            data_type: RegisterDataType::Bool,
            scale: 1.0,
            unit: "".to_string(),
        };
        assert_eq!(m.decode_value(&[1], 0), Some(1.0));
        assert_eq!(m.decode_value(&[0], 0), Some(0.0));
    }
}
