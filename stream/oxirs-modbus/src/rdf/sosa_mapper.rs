//! SOSA/SSN ontology mapper for Modbus register readings
//!
//! Converts Modbus register readings to RDF triples following the W3C
//! Semantic Sensor Network (SSN) ontology and its SOSA (Sensor, Observation,
//! Sample, Actuator) core:
//!
//! - SOSA: <http://www.w3.org/ns/sosa/>
//! - SSN:  <http://www.w3.org/ns/ssn/>
//! - QUDT: <http://qudt.org/schema/qudt/>
//!
//! Each Modbus register reading becomes an `sosa:Observation` with:
//!
//! ```turtle
//! <obs_42> a sosa:Observation ;
//!     sosa:madeBySensor <device/sensor_01> ;
//!     sosa:observedProperty <property/temperature> ;
//!     sosa:hasFeatureOfInterest <device/sensor_01> ;
//!     sosa:hasSimpleResult "22.5"^^xsd:double ;
//!     sosa:resultTime "2026-02-23T04:15:00Z"^^xsd:dateTime ;
//!     qudt:hasUnit qudt:DEG_C .
//! ```

use crate::codec::ModbusTypedValue;
use crate::registry::device_registry::{ModbusDevice, RegisterDefinition};
use std::sync::atomic::{AtomicU64, Ordering};

/// Well-known ontology namespace constants used by this mapper.
pub mod ns {
    /// SOSA ontology namespace
    pub const SOSA: &str = "http://www.w3.org/ns/sosa/";
    /// SSN ontology namespace
    pub const SSN: &str = "http://www.w3.org/ns/ssn/";
    /// QUDT schema namespace
    pub const QUDT: &str = "http://qudt.org/schema/qudt/";
    /// QUDT unit vocabulary namespace
    pub const QUDT_UNIT: &str = "http://qudt.org/vocab/unit/";
    /// XSD datatype namespace
    pub const XSD: &str = "http://www.w3.org/2001/XMLSchema#";
    /// RDF namespace
    pub const RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
    /// RDFS namespace
    pub const RDFS: &str = "http://www.w3.org/2000/01/rdf-schema#";
    /// W3C PROV-O namespace
    pub const PROV: &str = "http://www.w3.org/ns/prov#";
    /// OWL namespace
    pub const OWL: &str = "http://www.w3.org/2002/07/owl#";
}

/// A single RDF triple expressed as three IRI/literal strings.
///
/// `(subject, predicate, object)` — the object is either an IRI enclosed in
/// `<…>` or a typed literal `"value"^^<type>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdfTriple {
    /// Subject IRI (without angle brackets)
    pub subject: String,
    /// Predicate IRI (without angle brackets)
    pub predicate: String,
    /// Object: either an IRI string or a Turtle-formatted literal
    pub object: RdfObject,
}

impl RdfTriple {
    /// Format as a Turtle statement (without trailing `.`).
    pub fn to_turtle(&self) -> String {
        format!(
            "<{}> <{}> {}",
            self.subject,
            self.predicate,
            self.object.to_turtle()
        )
    }
}

/// An RDF object — either a named node (IRI) or a typed literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RdfObject {
    /// Named node (IRI)
    Iri(String),
    /// Typed literal with XSD datatype
    TypedLiteral { value: String, datatype: String },
    /// Plain string literal
    StringLiteral(String),
}

impl RdfObject {
    /// Format as a Turtle term.
    pub fn to_turtle(&self) -> String {
        match self {
            RdfObject::Iri(iri) => format!("<{}>", iri),
            RdfObject::TypedLiteral { value, datatype } => {
                format!("\"{}\"^^<{}>", escape_turtle_string(value), datatype)
            }
            RdfObject::StringLiteral(s) => {
                format!("\"{}\"", escape_turtle_string(s))
            }
        }
    }
}

/// Escape special characters in a Turtle string literal.
fn escape_turtle_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Maps Modbus register readings to SOSA/SSN RDF observations.
///
/// Each call to [`Self::reading_to_triples`] produces a complete set of triples
/// modelling one SOSA Observation, plus optional unit annotation triples.
///
/// # Thread Safety
///
/// The mapper is `Send + Sync`; the observation counter uses an atomic integer.
///
/// # Example
///
/// ```rust
/// use oxirs_modbus::rdf::sosa_mapper::{ModbusToSosaMapper, MapperConfig};
/// use oxirs_modbus::codec::ModbusTypedValue;
/// use oxirs_modbus::registry::device_registry::{
///     DeviceRegistry, ModbusDevice, DeviceType, RegisterMap, RegisterDefinition,
///     RegisterType as DeviceRegisterType,
/// };
/// use oxirs_modbus::mapping::ModbusDataType;
///
/// let config = MapperConfig {
///     base_iri: "http://factory.example.com/".to_string(),
///     emit_ssn_system: true,
///     emit_qudt_units: true,
///     emit_prov_o: false,
/// };
/// let mapper = ModbusToSosaMapper::new(config);
///
/// let reg = RegisterDefinition::new(
///     "temperature",
///     30001,
///     DeviceRegisterType::InputRegister,
///     ModbusDataType::Int16,
/// )
/// .with_unit("°C")
/// .with_rdf_predicate("http://example.com/property/temperature");
///
/// let device = ModbusDevice::new("sensor_01", 1, DeviceType::TemperatureSensor)
///     .with_tcp("192.168.1.10", 502);
///
/// let value = ModbusTypedValue::F64(22.5);
/// let triples = mapper.reading_to_triples(&device, &reg, &value, 22.5, 1740274500);
/// assert!(!triples.is_empty());
/// ```
pub struct ModbusToSosaMapper {
    config: MapperConfig,
    observation_counter: AtomicU64,
}

/// Configuration for [`ModbusToSosaMapper`].
#[derive(Debug, Clone)]
pub struct MapperConfig {
    /// Base IRI for generated observations and sensor IRIs.
    /// Should end with `/` (e.g. `"http://factory.example.com/"`).
    pub base_iri: String,
    /// Emit `ssn:System` and `ssn:hasSubSystem` triples for device grouping.
    pub emit_ssn_system: bool,
    /// Emit QUDT unit triples (`qudt:hasUnit`).
    pub emit_qudt_units: bool,
    /// Emit W3C PROV-O provenance triples (`prov:generatedAtTime`).
    pub emit_prov_o: bool,
}

impl Default for MapperConfig {
    fn default() -> Self {
        Self {
            base_iri: "http://oxirs.example.com/modbus/".to_string(),
            emit_ssn_system: true,
            emit_qudt_units: true,
            emit_prov_o: true,
        }
    }
}

impl ModbusToSosaMapper {
    /// Create a new mapper with explicit configuration.
    pub fn new(config: MapperConfig) -> Self {
        Self {
            config,
            observation_counter: AtomicU64::new(1),
        }
    }

    /// Generate SOSA/SSN RDF triples for a single Modbus register reading.
    ///
    /// # Arguments
    ///
    /// * `device` - Device descriptor from the registry.
    /// * `register` - Register definition (signal metadata).
    /// * `raw_value` - The decoded, un-scaled value.
    /// * `scaled_value` - The physical value after applying scale factor and offset.
    /// * `unix_timestamp` - POSIX timestamp (seconds since epoch) of the reading.
    ///
    /// # Returns
    ///
    /// A `Vec` of [`RdfTriple`]s representing the complete SOSA Observation.
    pub fn reading_to_triples(
        &self,
        device: &ModbusDevice,
        register: &RegisterDefinition,
        raw_value: &ModbusTypedValue,
        scaled_value: f64,
        unix_timestamp: u64,
    ) -> Vec<RdfTriple> {
        let mut triples = Vec::with_capacity(16);

        let obs_iri = self.observation_iri();
        let sensor_iri = self.sensor_iri(&device.device_id, &register.name);
        let property_iri = register
            .rdf_predicate
            .as_deref()
            .map(|s| s.to_string())
            .unwrap_or_else(|| self.property_iri(&device.device_id, &register.name));
        let feature_iri = self.feature_iri(&device.device_id);
        let timestamp_str = unix_to_iso8601(unix_timestamp);

        // ── rdf:type ──────────────────────────────────────────────────────
        triples.push(self.type_triple(&obs_iri, &format!("{}Observation", ns::SOSA)));

        // ── sosa:madeBySensor ─────────────────────────────────────────────
        triples.push(self.iri_triple(&obs_iri, &format!("{}madeBySensor", ns::SOSA), &sensor_iri));

        // ── sosa:observedProperty ─────────────────────────────────────────
        triples.push(self.iri_triple(
            &obs_iri,
            &format!("{}observedProperty", ns::SOSA),
            &property_iri,
        ));

        // ── sosa:hasFeatureOfInterest ─────────────────────────────────────
        triples.push(self.iri_triple(
            &obs_iri,
            &format!("{}hasFeatureOfInterest", ns::SOSA),
            &feature_iri,
        ));

        // ── sosa:hasSimpleResult ──────────────────────────────────────────
        triples.push(RdfTriple {
            subject: obs_iri.clone(),
            predicate: format!("{}hasSimpleResult", ns::SOSA),
            object: typed_literal_for_value(raw_value, scaled_value),
        });

        // ── sosa:resultTime ───────────────────────────────────────────────
        triples.push(RdfTriple {
            subject: obs_iri.clone(),
            predicate: format!("{}resultTime", ns::SOSA),
            object: RdfObject::TypedLiteral {
                value: timestamp_str.clone(),
                datatype: format!("{}dateTime", ns::XSD),
            },
        });

        // ── qudt:hasUnit ──────────────────────────────────────────────────
        if self.config.emit_qudt_units {
            if let Some(unit) = &register.unit {
                let unit_iri = unit_symbol_to_qudt_iri(unit);
                triples.push(self.iri_triple(&obs_iri, &format!("{}hasUnit", ns::QUDT), &unit_iri));
            }
        }

        // ── W3C PROV-O ────────────────────────────────────────────────────
        if self.config.emit_prov_o {
            triples.push(RdfTriple {
                subject: obs_iri.clone(),
                predicate: format!("{}generatedAtTime", ns::PROV),
                object: RdfObject::TypedLiteral {
                    value: timestamp_str,
                    datatype: format!("{}dateTime", ns::XSD),
                },
            });
        }

        // ── sensor type annotation ────────────────────────────────────────
        triples.push(self.type_triple(&sensor_iri, &format!("{}Sensor", ns::SOSA)));
        triples.push(self.iri_triple(&sensor_iri, &format!("{}observes", ns::SOSA), &property_iri));

        // ── SSN system (optional) ─────────────────────────────────────────
        if self.config.emit_ssn_system {
            let device_iri = self.device_iri(&device.device_id);
            triples.push(self.type_triple(&device_iri, &format!("{}System", ns::SSN)));
            triples.push(self.iri_triple(
                &device_iri,
                &format!("{}hasSubSystem", ns::SSN),
                &sensor_iri,
            ));
        }

        triples
    }

    /// Generate RDF triples describing a device (its type, location, etc.)
    /// without any observation. Use this for device catalogue graphs.
    pub fn device_description_triples(&self, device: &ModbusDevice) -> Vec<RdfTriple> {
        let mut triples = Vec::new();
        let device_iri = self.device_iri(&device.device_id);

        triples.push(self.type_triple(&device_iri, &format!("{}System", ns::SSN)));

        // Add a label if there's a meaningful device type string
        triples.push(RdfTriple {
            subject: device_iri.clone(),
            predicate: format!("{}label", ns::RDFS),
            object: RdfObject::StringLiteral(format!(
                "{} ({})",
                device.device_id,
                device.device_type.label()
            )),
        });

        // Unit ID annotation as a custom data property
        triples.push(RdfTriple {
            subject: device_iri.clone(),
            predicate: format!("{}unitId", self.config.base_iri),
            object: RdfObject::TypedLiteral {
                value: device.unit_id.to_string(),
                datatype: format!("{}integer", ns::XSD),
            },
        });

        // IP address when present
        if let Some(ip) = &device.ip_address {
            triples.push(RdfTriple {
                subject: device_iri.clone(),
                predicate: format!("{}ipAddress", self.config.base_iri),
                object: RdfObject::StringLiteral(ip.clone()),
            });
        }

        triples
    }

    // ── IRI generators ────────────────────────────────────────────────────

    /// Generate a unique observation IRI using a monotonic counter.
    pub fn observation_iri(&self) -> String {
        let n = self.observation_counter.fetch_add(1, Ordering::Relaxed);
        format!("{}observation/{}", self.base(), n)
    }

    /// IRI for a sensor (device + signal combination).
    pub fn sensor_iri(&self, device_id: &str, signal_name: &str) -> String {
        format!(
            "{}sensor/{}/{}",
            self.base(),
            url_encode(device_id),
            url_encode(signal_name)
        )
    }

    /// IRI for the device as a whole (feature of interest / SSN system).
    pub fn device_iri(&self, device_id: &str) -> String {
        format!("{}device/{}", self.base(), url_encode(device_id))
    }

    /// IRI for the feature of interest (same as device IRI by default).
    pub fn feature_iri(&self, device_id: &str) -> String {
        self.device_iri(device_id)
    }

    /// IRI for an observed property.
    pub fn property_iri(&self, device_id: &str, signal_name: &str) -> String {
        format!(
            "{}property/{}/{}",
            self.base(),
            url_encode(device_id),
            url_encode(signal_name)
        )
    }

    // ── private helpers ───────────────────────────────────────────────────

    fn base(&self) -> &str {
        self.config.base_iri.trim_end_matches('/')
    }

    fn type_triple(&self, subject: &str, type_iri: &str) -> RdfTriple {
        RdfTriple {
            subject: subject.to_string(),
            predicate: format!("{}type", ns::RDF),
            object: RdfObject::Iri(type_iri.to_string()),
        }
    }

    fn iri_triple(&self, subject: &str, predicate: &str, object_iri: &str) -> RdfTriple {
        RdfTriple {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: RdfObject::Iri(object_iri.to_string()),
        }
    }
}

// ── private free functions ─────────────────────────────────────────────────────

/// Convert a Unix timestamp (seconds) to an ISO 8601 / XSD dateTime string.
///
/// Uses integer arithmetic only (no external time crate) for zero-dependency operation.
fn unix_to_iso8601(secs: u64) -> String {
    // Days from epoch to year boundaries (non-leap approximation via Gregorian proleptic)
    // We use the de-facto Rata Die / Gregorian algorithm for date decomposition.
    let secs_per_day: u64 = 86_400;
    let days = secs / secs_per_day;
    let time_of_day = secs % secs_per_day;

    let (year, month, day) = days_to_ymd(days);

    let hh = time_of_day / 3600;
    let mm = (time_of_day % 3600) / 60;
    let ss = time_of_day % 60;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hh, mm, ss
    )
}

/// Gregorian calendar conversion: days since 1970-01-01 → (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Shift epoch to 1 Mar 2000 for easier leap year handling
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Map a physical unit symbol to a QUDT unit IRI.
///
/// Falls back to a literal-appended IRI for unknown symbols.
fn unit_symbol_to_qudt_iri(unit: &str) -> String {
    let qudt_id = match unit {
        "°C" | "degC" | "Celsius" | "CEL" => "DEG_C",
        "°F" | "degF" | "Fahrenheit" => "DEG_F",
        "K" | "Kelvin" => "K",
        "%" | "pct" | "percent" => "PERCENT",
        "Pa" | "pascal" => "PA",
        "kPa" => "KiloPA",
        "MPa" => "MegaPA",
        "bar" | "BAR" => "BAR",
        "mbar" => "MilliBAR",
        "psi" | "PSI" => "PSI",
        "V" | "volt" | "Volt" => "V",
        "mV" => "MilliV",
        "kV" => "KiloV",
        "A" | "amp" | "Amp" | "Ampere" => "A",
        "mA" => "MilliA",
        "W" | "watt" | "Watt" => "W",
        "kW" => "KiloW",
        "MW" => "MegaW",
        "kWh" | "KWh" => "KiloW-HR",
        "VA" => "V-A",
        "kVA" => "KiloV-A",
        "VAR" => "V-A-Reactive",
        "Hz" | "hz" | "hertz" => "HZ",
        "rpm" | "RPM" => "REV-PER-MIN",
        "m" | "meter" => "M",
        "mm" => "MilliM",
        "cm" => "CentiM",
        "km" => "KiloM",
        "m3" | "m³" => "M3",
        "L" | "liter" => "L",
        "m3/h" | "m³/h" => "M3-PER-HR",
        "L/min" => "L-PER-MIN",
        "kg" => "KG",
        "g" => "GM",
        "t" | "tonne" => "TON_Metric",
        "s" | "sec" | "second" => "SEC",
        "min" | "minute" => "MIN",
        "h" | "hour" => "HR",
        "Ω" | "ohm" | "Ohm" => "OHM",
        "kΩ" | "kOhm" => "KiloOHM",
        _ => {
            // Unknown unit: construct a local IRI
            return format!("{}unit/{}", ns::QUDT_UNIT, url_encode(unit));
        }
    };
    format!("{}{}", ns::QUDT_UNIT, qudt_id)
}

/// Produce a typed literal object for the `sosa:hasSimpleResult` triple.
///
/// Uses `xsd:boolean` for booleans, `xsd:double` for numeric types, and
/// `xsd:string` for text values.
fn typed_literal_for_value(raw_value: &ModbusTypedValue, scaled_value: f64) -> RdfObject {
    use crate::codec::ModbusTypedValue as V;
    match raw_value {
        V::Bool(b) => RdfObject::TypedLiteral {
            value: if *b { "true" } else { "false" }.to_string(),
            datatype: format!("{}boolean", ns::XSD),
        },
        V::Str(s) => RdfObject::TypedLiteral {
            value: s.clone(),
            datatype: format!("{}string", ns::XSD),
        },
        _ => RdfObject::TypedLiteral {
            value: format_double(scaled_value),
            datatype: format!("{}double", ns::XSD),
        },
    }
}

/// Format an f64 avoiding unnecessary trailing zeros while always producing
/// a valid XSD double lexical form.
fn format_double(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{:.1}", v)
    } else {
        // Full precision representation
        format!("{}", v)
    }
}

/// Minimal percent-encoding for path segments (replaces spaces and `/`).
fn url_encode(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            ' ' => "%20".to_string(),
            '/' => "%2F".to_string(),
            '#' => "%23".to_string(),
            '?' => "%3F".to_string(),
            '&' => "%26".to_string(),
            '+' => "%2B".to_string(),
            '=' => "%3D".to_string(),
            _ => c.to_string(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::ModbusDataType;
    use crate::registry::device_registry::{
        DeviceType, ModbusDevice, RegisterDefinition, RegisterType as DevRegType,
    };

    fn make_device() -> ModbusDevice {
        ModbusDevice::new("sensor_01", 1, DeviceType::TemperatureSensor)
            .with_tcp("192.168.1.10", 502)
    }

    fn make_register(name: &str) -> RegisterDefinition {
        RegisterDefinition::new(
            name,
            30001,
            DevRegType::InputRegister,
            ModbusDataType::Int16,
        )
        .with_unit("°C")
        .with_rdf_predicate("http://example.com/property/temperature")
    }

    fn make_mapper() -> ModbusToSosaMapper {
        ModbusToSosaMapper::new(MapperConfig {
            base_iri: "http://factory.example.com/".to_string(),
            emit_ssn_system: true,
            emit_qudt_units: true,
            emit_prov_o: true,
        })
    }

    #[test]
    fn test_observation_produces_triples() {
        let mapper = make_mapper();
        let device = make_device();
        let register = make_register("temperature");
        let value = ModbusTypedValue::F64(22.5);

        let triples = mapper.reading_to_triples(&device, &register, &value, 22.5, 1_740_274_500);

        assert!(!triples.is_empty(), "should produce at least one triple");

        // Must include rdf:type sosa:Observation
        let has_obs_type = triples.iter().any(|t| {
            t.predicate.contains("type")
                && matches!(&t.object, RdfObject::Iri(iri) if iri.contains("Observation"))
        });
        assert!(has_obs_type, "missing rdf:type sosa:Observation");
    }

    #[test]
    fn test_sensor_iri_generation() {
        let mapper = make_mapper();
        let iri = mapper.sensor_iri("sensor_01", "temperature");
        assert!(iri.contains("sensor_01"));
        assert!(iri.contains("temperature"));
    }

    #[test]
    fn test_device_iri_generation() {
        let mapper = make_mapper();
        let iri = mapper.device_iri("meter_a");
        assert!(iri.starts_with("http://factory.example.com"));
        assert!(iri.contains("meter_a"));
    }

    #[test]
    fn test_made_by_sensor_triple_present() {
        let mapper = make_mapper();
        let device = make_device();
        let register = make_register("temperature");
        let value = ModbusTypedValue::I16(225);

        let triples = mapper.reading_to_triples(&device, &register, &value, 22.5, 0);
        let has_sensor = triples.iter().any(|t| t.predicate.contains("madeBySensor"));
        assert!(has_sensor, "missing sosa:madeBySensor");
    }

    #[test]
    fn test_qudt_unit_triple_present() {
        let mapper = make_mapper();
        let device = make_device();
        let register = make_register("temperature"); // has unit °C

        let triples =
            mapper.reading_to_triples(&device, &register, &ModbusTypedValue::F64(22.5), 22.5, 0);
        let has_unit = triples
            .iter()
            .any(|t| t.predicate.contains("hasUnit") || t.predicate.contains("unit"));
        assert!(has_unit, "missing QUDT unit triple");
    }

    #[test]
    fn test_qudt_unit_mapping() {
        assert!(unit_symbol_to_qudt_iri("°C").contains("DEG_C"));
        assert!(unit_symbol_to_qudt_iri("kW").contains("KiloW"));
        assert!(unit_symbol_to_qudt_iri("bar").contains("BAR"));
        assert!(unit_symbol_to_qudt_iri("rpm").contains("REV-PER-MIN"));
        assert!(unit_symbol_to_qudt_iri("A").contains("/A"));
        // Unknown unit falls back to a local IRI
        let unknown = unit_symbol_to_qudt_iri("widgets/s");
        assert!(unknown.contains("unit/"));
    }

    #[test]
    fn test_prov_o_triple_present() {
        let mapper = make_mapper();
        let device = make_device();
        let register = make_register("temperature");
        let value = ModbusTypedValue::F64(25.0);

        let triples = mapper.reading_to_triples(&device, &register, &value, 25.0, 0);
        let has_prov = triples
            .iter()
            .any(|t| t.predicate.contains("generatedAtTime"));
        assert!(has_prov, "missing prov:generatedAtTime");
    }

    #[test]
    fn test_ssn_system_triples_present() {
        let mapper = make_mapper();
        let device = make_device();
        let register = make_register("temperature");
        let value = ModbusTypedValue::F64(25.0);

        let triples = mapper.reading_to_triples(&device, &register, &value, 25.0, 0);
        let has_system = triples
            .iter()
            .any(|t| matches!(&t.object, RdfObject::Iri(iri) if iri.contains("System")));
        assert!(has_system, "missing ssn:System triple");
    }

    #[test]
    fn test_observation_counter_increments() {
        let mapper = make_mapper();
        let iri1 = mapper.observation_iri();
        let iri2 = mapper.observation_iri();
        assert_ne!(iri1, iri2, "consecutive observation IRIs must differ");
    }

    #[test]
    fn test_bool_value_typed_literal() {
        let mapper = make_mapper();
        let device = make_device();
        let mut reg = make_register("alarm");
        reg.unit = None; // no unit for boolean
        let value = ModbusTypedValue::Bool(true);

        let triples = mapper.reading_to_triples(&device, &reg, &value, 1.0, 0);
        let result_triple = triples
            .iter()
            .find(|t| t.predicate.contains("hasSimpleResult"));
        assert!(result_triple.is_some());
        if let Some(t) = result_triple {
            match &t.object {
                RdfObject::TypedLiteral { value, datatype } => {
                    assert_eq!(value, "true");
                    assert!(datatype.contains("boolean"));
                }
                _ => panic!("expected TypedLiteral for bool"),
            }
        }
    }

    #[test]
    fn test_device_description_triples() {
        let mapper = make_mapper();
        let device = make_device();
        let triples = mapper.device_description_triples(&device);
        assert!(!triples.is_empty());

        let has_label = triples.iter().any(|t| t.predicate.contains("label"));
        assert!(has_label, "missing rdfs:label triple");
    }

    #[test]
    fn test_rdf_triple_to_turtle() {
        let triple = RdfTriple {
            subject: "http://example.com/obs/1".to_string(),
            predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            object: RdfObject::Iri("http://www.w3.org/ns/sosa/Observation".to_string()),
        };
        let turtle = triple.to_turtle();
        assert!(turtle.contains("<http://example.com/obs/1>"));
        assert!(turtle.contains("<http://www.w3.org/ns/sosa/Observation>"));
    }

    #[test]
    fn test_turtle_literal_escaping() {
        let obj = RdfObject::StringLiteral("hello \"world\"\nnewline".to_string());
        let turtle = obj.to_turtle();
        assert!(turtle.contains(r#"\""#));
        assert!(turtle.contains(r"\n"));
    }

    #[test]
    fn test_unix_to_iso8601_epoch() {
        // 1970-01-01T00:00:00Z
        let ts = unix_to_iso8601(0);
        assert_eq!(ts, "1970-01-01T00:00:00Z");
    }

    #[test]
    fn test_unix_to_iso8601_known_date() {
        // 2026-02-23T04:15:00Z = 1740273300 sec (approx)
        // Use a verifiable value: 2000-01-01T00:00:00Z = 946684800
        let ts = unix_to_iso8601(946_684_800);
        assert_eq!(ts, "2000-01-01T00:00:00Z");
    }

    #[test]
    fn test_no_unit_triple_when_disabled() {
        let mapper = ModbusToSosaMapper::new(MapperConfig {
            base_iri: "http://test.com/".to_string(),
            emit_ssn_system: false,
            emit_qudt_units: false,
            emit_prov_o: false,
        });
        let device = make_device();
        let register = make_register("temperature");
        let value = ModbusTypedValue::F64(20.0);

        let triples = mapper.reading_to_triples(&device, &register, &value, 20.0, 0);
        let has_unit = triples.iter().any(|t| t.predicate.contains("hasUnit"));
        assert!(!has_unit, "unit triple should be absent when disabled");

        let has_system = triples
            .iter()
            .any(|t| matches!(&t.object, RdfObject::Iri(i) if i.contains("System")));
        assert!(
            !has_system,
            "SSN system triples should be absent when disabled"
        );
    }

    #[test]
    fn test_url_encode() {
        assert_eq!(url_encode("hello world"), "hello%20world");
        assert_eq!(url_encode("a/b"), "a%2Fb");
        assert_eq!(url_encode("normal"), "normal");
    }

    #[test]
    fn test_format_double() {
        assert_eq!(format_double(1.0), "1.0");
        assert_eq!(format_double(22.5), "22.5");
    }
}
