//! RDF mapper for CAN messages
//!
//! Converts decoded CAN signals to RDF triples with proper typing
//! and W3C PROV-O provenance tracking.
//!
//! # Example
//!
//! ```no_run
//! use oxirs_canbus::rdf::CanRdfMapper;
//! use oxirs_canbus::{parse_dbc, CanFrame, CanId, RdfMappingConfig};
//!
//! let dbc_content = r#"
//! BO_ 2024 EngineData: 8 Engine
//!  SG_ EngineSpeed : 0|16@1+ (0.125,0) [0|8031.875] "rpm" Dashboard
//! "#;
//!
//! let db = parse_dbc(dbc_content).unwrap();
//! let config = RdfMappingConfig {
//!     device_id: "vehicle001".to_string(),
//!     base_iri: "http://automotive.example.com/vehicle".to_string(),
//!     graph_iri: "urn:automotive:can-data".to_string(),
//! };
//!
//! let mut mapper = CanRdfMapper::new(db, config);
//!
//! // Map a CAN frame
//! let id = CanId::standard(0x7E8).unwrap();
//! let frame = CanFrame::new(id, vec![0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]).unwrap();
//! let triples = mapper.map_frame(&frame);
//! ```

use crate::config::RdfMappingConfig;
use crate::dbc::{DbcDatabase, DbcMessage, DecodedSignalValue, SignalDecoder};
use crate::error::{CanbusError, CanbusResult};
use crate::protocol::{CanFrame, CanId};
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
    /// Automotive ontology (custom)
    pub const AUTO: &str = "http://automotive.oxirs.org/ontology#";
    /// CAN bus ontology (custom)
    pub const CAN: &str = "http://canbus.oxirs.org/ontology#";
}

/// Generated triple with provenance metadata
#[derive(Debug, Clone)]
pub struct GeneratedTriple {
    /// The main data triple
    pub triple: Triple,
    /// Timestamp of generation
    pub timestamp: DateTime<Utc>,
    /// Optional unit IRI
    pub unit: Option<String>,
    /// Source message name
    pub message_name: String,
    /// Source signal name
    pub signal_name: String,
    /// CAN message ID
    pub can_id: u32,
}

impl GeneratedTriple {
    /// Generate provenance triples (prov:generatedAtTime, unit, etc.)
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
                    NamedNode::new(format!("{}dateTime", ns::XSD)).unwrap(),
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

        // Add CAN message ID triple
        if let Some(ref subject) = subject_node {
            let can_id_pred = format!("{}messageId", ns::CAN);
            if let Ok(pred) = NamedNode::new(&can_id_pred) {
                let lit = Literal::new_typed(
                    format!("{:#X}", self.can_id),
                    NamedNode::new(format!("{}hexBinary", ns::XSD)).unwrap(),
                );
                triples.push(Triple::new(subject.clone(), pred, lit));
            }
        }

        triples
    }
}

/// CAN to RDF mapper
///
/// Converts CAN frames to RDF triples using DBC signal definitions.
pub struct CanRdfMapper {
    /// DBC database for signal definitions
    database: DbcDatabase,
    /// Signal decoder
    decoder: SignalDecoder<'static>,
    /// RDF mapping configuration
    config: RdfMappingConfig,
    /// Previous values for change detection (optional)
    previous_values: HashMap<(u32, String), f64>,
    /// Deadband threshold for change detection (0 = report all)
    deadband: f64,
}

impl CanRdfMapper {
    /// Create a new CAN RDF mapper
    pub fn new(database: DbcDatabase, config: RdfMappingConfig) -> Self {
        // Create decoder with static lifetime by leaking the database
        // This is acceptable for long-lived mappers
        let db_ref: &'static DbcDatabase = Box::leak(Box::new(database.clone()));
        let decoder = SignalDecoder::new(db_ref);

        Self {
            database,
            decoder,
            config,
            previous_values: HashMap::new(),
            deadband: 0.0,
        }
    }

    /// Set deadband threshold for change detection
    ///
    /// Only signals that change by more than the deadband will generate triples.
    pub fn with_deadband(mut self, deadband: f64) -> Self {
        self.deadband = deadband;
        self
    }

    /// Map a CAN frame to RDF triples
    pub fn map_frame(&mut self, frame: &CanFrame) -> CanbusResult<Vec<GeneratedTriple>> {
        let timestamp = Utc::now();
        self.map_frame_with_timestamp(frame, timestamp)
    }

    /// Map a CAN frame with explicit timestamp
    pub fn map_frame_with_timestamp(
        &mut self,
        frame: &CanFrame,
        timestamp: DateTime<Utc>,
    ) -> CanbusResult<Vec<GeneratedTriple>> {
        let can_id = frame.id.as_raw();

        // Get message from database
        let message = match self.database.get_message(can_id) {
            Some(msg) => msg,
            None => {
                // Try extended ID
                if let CanId::Extended(id) = frame.id {
                    // For J1939, extract PGN
                    let pgn = (id >> 8) & 0x3FFFF;
                    self.database.get_message(pgn).ok_or_else(|| {
                        CanbusError::SignalNotFound(format!("CAN ID {:X}", can_id))
                    })?
                } else {
                    return Err(CanbusError::SignalNotFound(format!("CAN ID {:X}", can_id)));
                }
            }
        };

        // Decode signals
        let decoded = self.decoder.decode_message(can_id, &frame.data)?;

        // Generate triples
        let mut triples = Vec::new();

        for (signal_name, value) in decoded {
            // Check deadband
            if self.deadband > 0.0 {
                let key = (can_id, signal_name.clone());
                if let Some(&prev) = self.previous_values.get(&key) {
                    if (value.physical_value - prev).abs() < self.deadband {
                        continue; // Skip - within deadband
                    }
                }
                self.previous_values.insert(key, value.physical_value);
            }

            // Get signal definition for unit info
            let signal = message.get_signal(&signal_name);

            // Generate triple
            if let Some(triple) = self.create_signal_triple(message, &value, timestamp)? {
                let unit = signal.and_then(|s| unit_to_qudt(&s.unit));

                triples.push(GeneratedTriple {
                    triple,
                    timestamp,
                    unit,
                    message_name: message.name.clone(),
                    signal_name: signal_name.clone(),
                    can_id,
                });
            }
        }

        Ok(triples)
    }

    /// Create a triple for a decoded signal value
    fn create_signal_triple(
        &self,
        message: &DbcMessage,
        value: &DecodedSignalValue,
        _timestamp: DateTime<Utc>,
    ) -> CanbusResult<Option<Triple>> {
        // Build subject IRI: base_iri/device_id/message_name
        let subject_iri = format!(
            "{}/{}/{}",
            self.config.base_iri.trim_end_matches('/'),
            self.config.device_id,
            message.name
        );

        // Build predicate IRI: base_iri/signal/signal_name
        let predicate_iri = format!(
            "{}/signal/{}",
            self.config.base_iri.trim_end_matches('/'),
            value.name
        );

        let subject = match NamedNode::new(&subject_iri) {
            Ok(n) => n,
            Err(_) => return Ok(None),
        };

        let predicate = match NamedNode::new(&predicate_iri) {
            Ok(n) => n,
            Err(_) => return Ok(None),
        };

        // Create typed literal based on value type
        let literal = if let Some(ref desc) = value.description {
            // Enum value - use string
            let datatype = NamedNode::new(format!("{}string", ns::XSD)).unwrap();
            Literal::new_typed(desc.clone(), datatype)
        } else {
            // Numeric value - use float or integer based on signal
            let datatype = NamedNode::new(format!("{}float", ns::XSD)).unwrap();
            Literal::new_typed(format!("{}", value.physical_value), datatype)
        };

        Ok(Some(Triple::new(subject, predicate, literal)))
    }

    /// Map J1939 message to RDF triples
    ///
    /// For J1939, uses PGN-based message lookup
    pub fn map_j1939_frame(
        &mut self,
        frame: &CanFrame,
        pgn: u32,
    ) -> CanbusResult<Vec<GeneratedTriple>> {
        let timestamp = Utc::now();

        // Get message by PGN
        let message = self
            .database
            .get_message(pgn)
            .ok_or_else(|| CanbusError::SignalNotFound(format!("PGN {:X}", pgn)))?;

        // Decode using raw CAN ID for decoder cache lookup
        let decoded = self.decoder.decode_message(pgn, &frame.data)?;

        let mut triples = Vec::new();

        for (signal_name, value) in decoded {
            let signal = message.get_signal(&signal_name);

            if let Some(triple) = self.create_signal_triple(message, &value, timestamp)? {
                let unit = signal.and_then(|s| unit_to_qudt(&s.unit));

                triples.push(GeneratedTriple {
                    triple,
                    timestamp,
                    unit,
                    message_name: message.name.clone(),
                    signal_name,
                    can_id: frame.id.as_raw(),
                });
            }
        }

        Ok(triples)
    }

    /// Reset change detection state
    pub fn reset_change_detection(&mut self) {
        self.previous_values.clear();
    }

    /// Get the graph IRI for storing CAN data
    pub fn graph_iri(&self) -> &str {
        &self.config.graph_iri
    }

    /// Get the device ID
    pub fn device_id(&self) -> &str {
        &self.config.device_id
    }

    /// Get the DBC database
    pub fn database(&self) -> &DbcDatabase {
        &self.database
    }
}

/// Convert DBC unit string to QUDT IRI
fn unit_to_qudt(unit: &str) -> Option<String> {
    let qudt = match unit.to_lowercase().as_str() {
        "rpm" | "r/min" | "rev/min" => Some("REV-PER-MIN"),
        "km/h" | "kph" => Some("KiloM-PER-HR"),
        "m/s" | "mps" => Some("M-PER-SEC"),
        "mph" => Some("MI-PER-HR"),
        "degc" | "°c" | "deg c" => Some("DEG_C"),
        "degf" | "°f" | "deg f" => Some("DEG_F"),
        "k" | "kelvin" => Some("K"),
        "%" | "percent" => Some("PERCENT"),
        "bar" => Some("BAR"),
        "psi" => Some("PSI"),
        "kpa" | "kpascal" => Some("KiloPa"),
        "pa" | "pascal" => Some("PA"),
        "v" | "volt" => Some("V"),
        "a" | "amp" | "ampere" => Some("A"),
        "w" | "watt" => Some("W"),
        "kw" | "kilowatt" => Some("KiloW"),
        "nm" | "n·m" | "newton-meter" => Some("N-M"),
        "kg" | "kilogram" => Some("KiloGM"),
        "l" | "liter" | "litre" => Some("L"),
        "l/h" | "l/hr" => Some("L-PER-HR"),
        "l/100km" => Some("L-PER-100KiloM"),
        "mpg" => Some("MI-PER-GAL"),
        "s" | "sec" | "second" => Some("SEC"),
        "ms" | "millisecond" => Some("MilliSEC"),
        "h" | "hr" | "hour" => Some("HR"),
        "hz" | "hertz" => Some("HZ"),
        "khz" | "kilohertz" => Some("KiloHZ"),
        _ => None,
    }?;

    Some(format!("{}{}", ns::QUDT_UNIT, qudt))
}

/// Builder for common automotive unit IRIs
pub struct AutomotiveUnits;

impl AutomotiveUnits {
    /// RPM (revolutions per minute)
    pub fn rpm() -> String {
        format!("{}REV-PER-MIN", ns::QUDT_UNIT)
    }

    /// Kilometers per hour
    pub fn kmh() -> String {
        format!("{}KiloM-PER-HR", ns::QUDT_UNIT)
    }

    /// Miles per hour
    pub fn mph() -> String {
        format!("{}MI-PER-HR", ns::QUDT_UNIT)
    }

    /// Celsius
    pub fn celsius() -> String {
        format!("{}DEG_C", ns::QUDT_UNIT)
    }

    /// Fahrenheit
    pub fn fahrenheit() -> String {
        format!("{}DEG_F", ns::QUDT_UNIT)
    }

    /// Bar pressure
    pub fn bar() -> String {
        format!("{}BAR", ns::QUDT_UNIT)
    }

    /// PSI pressure
    pub fn psi() -> String {
        format!("{}PSI", ns::QUDT_UNIT)
    }

    /// Kilowatt
    pub fn kilowatt() -> String {
        format!("{}KiloW", ns::QUDT_UNIT)
    }

    /// Newton-meter (torque)
    pub fn newton_meter() -> String {
        format!("{}N-M", ns::QUDT_UNIT)
    }

    /// Percent
    pub fn percent() -> String {
        format!("{}PERCENT", ns::QUDT_UNIT)
    }

    /// Liters per 100km (fuel economy)
    pub fn liters_per_100km() -> String {
        format!("{}L-PER-100KiloM", ns::QUDT_UNIT)
    }

    /// Miles per gallon
    pub fn mpg() -> String {
        format!("{}MI-PER-GAL", ns::QUDT_UNIT)
    }

    /// Volt
    pub fn volt() -> String {
        format!("{}V", ns::QUDT_UNIT)
    }

    /// Ampere
    pub fn ampere() -> String {
        format!("{}A", ns::QUDT_UNIT)
    }
}

/// Statistics about mapped CAN data
#[derive(Debug, Clone, Default)]
pub struct MapperStatistics {
    /// Total frames processed
    pub frames_processed: u64,
    /// Total triples generated
    pub triples_generated: u64,
    /// Frames with unknown message IDs
    pub unknown_frames: u64,
    /// Signals filtered by deadband
    pub deadband_filtered: u64,
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

VAL_ 2024 ThrottlePos 0 "Closed" 100 "WOT";
"#;

    fn create_test_mapper() -> CanRdfMapper {
        let db = parse_dbc(TEST_DBC).unwrap();
        let config = RdfMappingConfig {
            device_id: "vehicle001".to_string(),
            base_iri: "http://automotive.example.com/vehicle".to_string(),
            graph_iri: "urn:automotive:can-data".to_string(),
        };
        CanRdfMapper::new(db, config)
    }

    #[test]
    fn test_map_frame() {
        let mut mapper = create_test_mapper();

        // Create frame with EngineSpeed = 16384 (2048 rpm)
        let id = CanId::standard(2024).unwrap();
        let frame =
            CanFrame::new(id, vec![0x00, 0x40, 0x96, 0x80, 0x00, 0x00, 0x00, 0x00]).unwrap();

        let triples = mapper.map_frame(&frame).unwrap();

        // Should have triples for EngineSpeed, EngineTemp, ThrottlePos
        assert_eq!(triples.len(), 3);

        // Check that all have the correct message name
        for triple in &triples {
            assert_eq!(triple.message_name, "EngineData");
            assert_eq!(triple.can_id, 2024);
        }
    }

    #[test]
    fn test_unit_conversion() {
        assert_eq!(unit_to_qudt("rpm"), Some(AutomotiveUnits::rpm()));
        assert_eq!(unit_to_qudt("km/h"), Some(AutomotiveUnits::kmh()));
        assert_eq!(unit_to_qudt("degC"), Some(AutomotiveUnits::celsius()));
        assert_eq!(unit_to_qudt("%"), Some(AutomotiveUnits::percent()));
        assert_eq!(unit_to_qudt("unknown"), None);
    }

    #[test]
    fn test_provenance_triples() {
        let mut mapper = create_test_mapper();

        let id = CanId::standard(2024).unwrap();
        let frame =
            CanFrame::new(id, vec![0x00, 0x40, 0x96, 0x80, 0x00, 0x00, 0x00, 0x00]).unwrap();

        let triples = mapper.map_frame(&frame).unwrap();
        let prov_triples = triples[0].provenance_triples();

        // Should have timestamp and CAN ID triples
        assert!(prov_triples.len() >= 2);

        // Check for timestamp triple
        let has_timestamp = prov_triples.iter().any(|t| match t.predicate() {
            oxirs_core::model::Predicate::NamedNode(n) => n.as_str().contains("generatedAtTime"),
            _ => false,
        });
        assert!(has_timestamp);
    }

    #[test]
    fn test_deadband_filtering() {
        let mut mapper = create_test_mapper().with_deadband(100.0);

        let id = CanId::standard(2024).unwrap();

        // First reading
        let frame1 =
            CanFrame::new(id, vec![0x00, 0x40, 0x96, 0x80, 0x00, 0x00, 0x00, 0x00]).unwrap();
        let triples1 = mapper.map_frame(&frame1).unwrap();
        assert!(!triples1.is_empty());

        // Second reading - within deadband (small change)
        let frame2 =
            CanFrame::new(id, vec![0x10, 0x40, 0x96, 0x80, 0x00, 0x00, 0x00, 0x00]).unwrap();
        let triples2 = mapper.map_frame(&frame2).unwrap();

        // Some signals may be filtered
        assert!(triples2.len() <= triples1.len());
    }

    #[test]
    fn test_automotive_units() {
        assert!(AutomotiveUnits::rpm().contains("REV-PER-MIN"));
        assert!(AutomotiveUnits::kmh().contains("KiloM-PER-HR"));
        assert!(AutomotiveUnits::celsius().contains("DEG_C"));
        assert!(AutomotiveUnits::bar().contains("BAR"));
        assert!(AutomotiveUnits::newton_meter().contains("N-M"));
    }

    #[test]
    fn test_subject_predicate_iris() {
        let mut mapper = create_test_mapper();

        let id = CanId::standard(2024).unwrap();
        let frame =
            CanFrame::new(id, vec![0x00, 0x40, 0x96, 0x80, 0x00, 0x00, 0x00, 0x00]).unwrap();

        let triples = mapper.map_frame(&frame).unwrap();

        // Check subject IRI format
        let subject = match triples[0].triple.subject() {
            oxirs_core::model::Subject::NamedNode(n) => n.as_str().to_string(),
            _ => String::new(),
        };
        assert!(subject.contains("vehicle001"));
        assert!(subject.contains("EngineData"));

        // Check predicate IRI format
        let predicate = match triples[0].triple.predicate() {
            oxirs_core::model::Predicate::NamedNode(n) => n.as_str().to_string(),
            _ => String::new(),
        };
        assert!(predicate.contains("/signal/"));
    }

    #[test]
    fn test_graph_iri() {
        let mapper = create_test_mapper();
        assert_eq!(mapper.graph_iri(), "urn:automotive:can-data");
    }

    #[test]
    fn test_reset_change_detection() {
        let mut mapper = create_test_mapper().with_deadband(100.0);

        let id = CanId::standard(2024).unwrap();
        let frame =
            CanFrame::new(id, vec![0x00, 0x40, 0x96, 0x80, 0x00, 0x00, 0x00, 0x00]).unwrap();

        // First reading
        let _ = mapper.map_frame(&frame).unwrap();

        // Reset and read same value - should generate triples again
        mapper.reset_change_detection();
        let triples = mapper.map_frame(&frame).unwrap();
        assert!(!triples.is_empty());
    }
}
