//! Advanced RDF mapping for CAN bus / J1939 data
//!
//! Maps decoded J1939 PGNs and Diagnostic Trouble Codes (DTCs) to RDF triples
//! using automotive ontologies:
//!
//! - **VSSo** (Vehicle Signal Specification ontology) – signals and observations
//! - **SOSA/SSN** (W3C Sensor, Observation, Sample, and Actuator) – observations
//! - **PROV-O** (W3C Provenance Ontology) – data lineage
//! - **QUDT** (Quantities, Units, Dimensions) – units of measure
//! - Custom `J1939:` namespace for DTCs and diagnostic events
//!
//! # Ontology Namespaces
//!
//! | Prefix | IRI |
//! |--------|-----|
//! | `vsso:` | `https://github.com/w3c/vsso/blob/gh-pages/spec/core/vsso.ttl#` |
//! | `sosa:` | `http://www.w3.org/ns/sosa/` |
//! | `ssn:`  | `http://www.w3.org/ns/ssn/` |
//! | `prov:` | `http://www.w3.org/ns/prov#` |
//! | `qudt:` | `http://qudt.org/schema/qudt/` |
//! | `j1939:`| `http://j1939.oxirs.org/ontology#` |
//!
//! # Example
//!
//! ```rust
//! use oxirs_canbus::rdf::can_to_rdf::CanToRdfMapper;
//! use oxirs_canbus::j1939::diagnostics::DiagnosticTroubleCode;
//!
//! let mapper = CanToRdfMapper::new(
//!     "http://example.com/vehicle/",
//!     "urn:vehicle:truck001",
//! );
//!
//! // Map raw PGN data to triples
//! let triples = mapper.pgn_to_triples(61444, &[0x00, 0x7D, 0x7D, 0x80, 0x3E, 0x00, 0x00, 0x7D], 1706000000);
//! assert!(!triples.is_empty());
//!
//! // Map a DTC to triples
//! let dtc = DiagnosticTroubleCode::new(190, 3);
//! let dtc_triples = mapper.dtc_to_triples(&dtc, 0x00, 1706000000);
//! assert!(!dtc_triples.is_empty());
//! ```

use crate::j1939::diagnostics::{known_spn_description, DiagnosticTroubleCode, Dm1Message};
use crate::protocol::j1939::{J1939Header, J1939Message, Pgn};
use crate::protocol::j1939_pgns::PgnRegistry;
#[cfg(test)]
use crate::protocol::j1939_pgns::{PGN_CCVS, PGN_EEC1};

// ============================================================================
// Namespace constants
// ============================================================================

/// VSSo (Vehicle Signal Specification ontology)
pub const NS_VSSO: &str = "https://github.com/w3c/vsso/blob/gh-pages/spec/core/vsso.ttl#";
/// SOSA – Sensor, Observation, Sample, Actuator
pub const NS_SOSA: &str = "http://www.w3.org/ns/sosa/";
/// SSN – Semantic Sensor Network
pub const NS_SSN: &str = "http://www.w3.org/ns/ssn/";
/// W3C PROV-O
pub const NS_PROV: &str = "http://www.w3.org/ns/prov#";
/// QUDT units
pub const NS_QUDT: &str = "http://qudt.org/schema/qudt/";
/// QUDT unit vocabulary
pub const NS_QUDT_UNIT: &str = "http://qudt.org/vocab/unit/";
/// XSD datatypes
pub const NS_XSD: &str = "http://www.w3.org/2001/XMLSchema#";
/// RDF syntax
pub const NS_RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
/// J1939 ontology (OxiRS custom)
pub const NS_J1939: &str = "http://j1939.oxirs.org/ontology#";

// ============================================================================
// RdfTriple – a simple lightweight triple representation
// ============================================================================

/// A lightweight RDF triple consisting of subject, predicate, and object strings.
///
/// For integration with `oxirs-core` model types, convert using the IRI strings
/// directly (all IRIs are angle-bracket enclosed when serialised to Turtle).
#[derive(Debug, Clone, PartialEq)]
pub struct RdfTriple {
    /// Subject IRI (no angle brackets)
    pub subject: String,
    /// Predicate IRI (no angle brackets)
    pub predicate: String,
    /// Object – either an IRI or a typed/plain literal
    pub object: RdfObject,
}

/// The object component of an RDF triple
#[derive(Debug, Clone, PartialEq)]
pub enum RdfObject {
    /// A named node (IRI)
    Iri(String),
    /// A plain string literal
    Literal(String),
    /// A typed literal (value, datatype IRI)
    TypedLiteral(String, String),
    /// A language-tagged string literal
    LangLiteral(String, String),
}

impl RdfObject {
    /// Serialize to Turtle/N-Triples object syntax.
    pub fn to_turtle(&self) -> String {
        match self {
            RdfObject::Iri(iri) => format!("<{}>", iri),
            RdfObject::Literal(s) => format!("\"{}\"", s.replace('"', "\\\"")),
            RdfObject::TypedLiteral(val, dt) => {
                format!("\"{}\"^^<{}>", val.replace('"', "\\\""), dt)
            }
            RdfObject::LangLiteral(val, lang) => {
                format!("\"{}\"@{}", val.replace('"', "\\\""), lang)
            }
        }
    }
}

impl RdfTriple {
    /// Create a triple with an IRI object.
    pub fn iri(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: RdfObject::Iri(object.into()),
        }
    }

    /// Create a triple with a typed literal object.
    pub fn typed_literal(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        value: impl Into<String>,
        datatype: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: RdfObject::TypedLiteral(value.into(), datatype.into()),
        }
    }

    /// Create a triple with a plain string literal object.
    pub fn literal(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: RdfObject::Literal(value.into()),
        }
    }

    /// Serialize to Turtle syntax (without trailing `.`).
    pub fn to_turtle(&self) -> String {
        format!(
            "<{}> <{}> {}",
            self.subject,
            self.predicate,
            self.object.to_turtle()
        )
    }
}

// ============================================================================
// CanToRdfMapper
// ============================================================================

/// Maps CAN bus / J1939 data to RDF triples using automotive ontologies.
///
/// Create one mapper per vehicle / device and reuse it for all incoming
/// messages. The mapper is stateless with respect to message values
/// (it does not track previous values).
pub struct CanToRdfMapper {
    /// Base IRI for vehicle-specific resources
    base_iri: String,
    /// IRI that identifies the vehicle/device
    vehicle_iri: String,
    /// PGN registry for decoding J1939 messages
    pgn_registry: PgnRegistry,
}

impl CanToRdfMapper {
    /// Create a new mapper.
    ///
    /// # Arguments
    ///
    /// * `base_iri`    – Namespace for generated resources, e.g. `"http://example.com/vehicle/"`
    /// * `vehicle_iri` – IRI of the vehicle subject, e.g. `"urn:vehicle:truck001"`
    pub fn new(base_iri: &str, vehicle_iri: &str) -> Self {
        Self {
            base_iri: base_iri.trim_end_matches('/').to_string(),
            vehicle_iri: vehicle_iri.to_string(),
            pgn_registry: PgnRegistry::with_standard_decoders(),
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Map a decoded J1939 PGN to RDF triples.
    ///
    /// Produces SOSA `Observation` instances for each decoded signal, linked
    /// to the vehicle via `sosa:hasFeatureOfInterest`.
    ///
    /// Returns an empty `Vec` when the PGN is not recognised or decoding fails.
    pub fn pgn_to_triples(&self, pgn: u32, data: &[u8], timestamp_secs: u64) -> Vec<RdfTriple> {
        // Build a synthetic J1939Message for the registry decoder
        let msg = self.build_message(pgn, data);

        let decoded = match self.pgn_registry.decode(&msg) {
            Some(d) => d,
            None => return Vec::new(),
        };

        let mut triples = Vec::new();
        let ts_xsd = format!("{}", timestamp_secs);

        for signal in &decoded.signals {
            if !signal.valid {
                continue;
            }

            // Observation IRI: <base>/obs/<pgn>/<signal>/<timestamp>
            let obs_iri = format!(
                "{}/obs/{}/{}/{}",
                self.base_iri,
                pgn,
                sanitize_iri_fragment(signal.name),
                timestamp_secs
            );

            // rdf:type sosa:Observation
            triples.push(RdfTriple::iri(
                &obs_iri,
                format!("{}type", NS_RDF),
                format!("{}Observation", NS_SOSA),
            ));

            // sosa:hasFeatureOfInterest -> vehicle
            triples.push(RdfTriple::iri(
                &obs_iri,
                format!("{}hasFeatureOfInterest", NS_SOSA),
                &self.vehicle_iri,
            ));

            // sosa:observedProperty -> VSSo or J1939 property
            let prop_iri = self.signal_property_iri(pgn, signal.name);
            triples.push(RdfTriple::iri(
                &obs_iri,
                format!("{}observedProperty", NS_SOSA),
                &prop_iri,
            ));

            // sosa:hasSimpleResult -> xsd:double
            triples.push(RdfTriple::typed_literal(
                &obs_iri,
                format!("{}hasSimpleResult", NS_SOSA),
                format!("{:.6}", signal.value),
                format!("{}double", NS_XSD),
            ));

            // qudt:unit -> unit IRI
            let unit_iri = unit_to_qudt_iri(signal.unit);
            triples.push(RdfTriple::iri(
                &obs_iri,
                format!("{}unit", NS_QUDT),
                unit_iri,
            ));

            // prov:generatedAtTime -> xsd:dateTime (approximated from unix seconds)
            triples.push(RdfTriple::typed_literal(
                &obs_iri,
                format!("{}generatedAtTime", NS_PROV),
                &ts_xsd,
                format!("{}integer", NS_XSD),
            ));

            // j1939:pgn -> xsd:unsignedInt
            triples.push(RdfTriple::typed_literal(
                &obs_iri,
                format!("{}pgn", NS_J1939),
                pgn.to_string(),
                format!("{}unsignedInt", NS_XSD),
            ));

            // ssn:isPropertyOf -> vehicle
            triples.push(RdfTriple::iri(
                &prop_iri,
                format!("{}isPropertyOf", NS_SSN),
                &self.vehicle_iri,
            ));
        }

        triples
    }

    /// Map a Diagnostic Trouble Code to RDF triples.
    ///
    /// Generates a `j1939:DiagnosticTroubleCode` resource with all relevant
    /// properties (SPN, FMI, occurrence count, FMI description, SPN description).
    pub fn dtc_to_triples(
        &self,
        dtc: &DiagnosticTroubleCode,
        source_addr: u8,
        timestamp_secs: u64,
    ) -> Vec<RdfTriple> {
        let dtc_iri = format!(
            "{}/dtc/{}/{}/{}",
            self.base_iri, source_addr, dtc.spn, dtc.fmi
        );

        let mut triples = Vec::new();

        // rdf:type j1939:DiagnosticTroubleCode
        triples.push(RdfTriple::iri(
            &dtc_iri,
            format!("{}type", NS_RDF),
            format!("{}DiagnosticTroubleCode", NS_J1939),
        ));

        // j1939:spn
        triples.push(RdfTriple::typed_literal(
            &dtc_iri,
            format!("{}spn", NS_J1939),
            dtc.spn.to_string(),
            format!("{}unsignedInt", NS_XSD),
        ));

        // j1939:fmi
        triples.push(RdfTriple::typed_literal(
            &dtc_iri,
            format!("{}fmi", NS_J1939),
            dtc.fmi.to_string(),
            format!("{}unsignedShort", NS_XSD),
        ));

        // j1939:occurrenceCount
        triples.push(RdfTriple::typed_literal(
            &dtc_iri,
            format!("{}occurrenceCount", NS_J1939),
            dtc.occurrence_count.to_string(),
            format!("{}unsignedShort", NS_XSD),
        ));

        // j1939:conversionMethod
        triples.push(RdfTriple::typed_literal(
            &dtc_iri,
            format!("{}conversionMethod", NS_J1939),
            dtc.cm.to_string(),
            format!("{}boolean", NS_XSD),
        ));

        // j1939:fmiDescription
        triples.push(RdfTriple::literal(
            &dtc_iri,
            format!("{}fmiDescription", NS_J1939),
            dtc.fmi_description(),
        ));

        // j1939:spnDescription (when known)
        if let Some(desc) = known_spn_description(dtc.spn) {
            triples.push(RdfTriple::literal(
                &dtc_iri,
                format!("{}spnDescription", NS_J1939),
                desc,
            ));
        }

        // j1939:sourceAddress
        triples.push(RdfTriple::typed_literal(
            &dtc_iri,
            format!("{}sourceAddress", NS_J1939),
            source_addr.to_string(),
            format!("{}unsignedShort", NS_XSD),
        ));

        // j1939:isActive
        triples.push(RdfTriple::typed_literal(
            &dtc_iri,
            format!("{}isActive", NS_J1939),
            dtc.is_active().to_string(),
            format!("{}boolean", NS_XSD),
        ));

        // j1939:shortLabel
        triples.push(RdfTriple::literal(
            &dtc_iri,
            format!("{}shortLabel", NS_J1939),
            dtc.short_label(),
        ));

        // prov:generatedAtTime
        triples.push(RdfTriple::typed_literal(
            &dtc_iri,
            format!("{}generatedAtTime", NS_PROV),
            timestamp_secs.to_string(),
            format!("{}integer", NS_XSD),
        ));

        // j1939:reportedByVehicle -> vehicle IRI
        triples.push(RdfTriple::iri(
            &dtc_iri,
            format!("{}reportedByVehicle", NS_J1939),
            &self.vehicle_iri,
        ));

        triples
    }

    /// Map a complete DM1 message (with all its DTCs) to RDF triples.
    ///
    /// Generates lamp status triples for the vehicle plus individual DTC triples.
    pub fn dm1_to_triples(
        &self,
        dm1: &Dm1Message,
        source_addr: u8,
        timestamp_secs: u64,
    ) -> Vec<RdfTriple> {
        let mut triples = Vec::new();

        // DM1 event IRI
        let dm1_iri = format!("{}/dm1/{}/{}", self.base_iri, source_addr, timestamp_secs);

        // rdf:type j1939:DM1Message
        triples.push(RdfTriple::iri(
            &dm1_iri,
            format!("{}type", NS_RDF),
            format!("{}DM1Message", NS_J1939),
        ));

        // j1939:milLamp
        triples.push(RdfTriple::literal(
            &dm1_iri,
            format!("{}milLamp", NS_J1939),
            dm1.mil_lamp.to_string(),
        ));

        // j1939:rslLamp
        triples.push(RdfTriple::literal(
            &dm1_iri,
            format!("{}rslLamp", NS_J1939),
            dm1.rsl_lamp.to_string(),
        ));

        // j1939:awlLamp
        triples.push(RdfTriple::literal(
            &dm1_iri,
            format!("{}awlLamp", NS_J1939),
            dm1.awl_lamp.to_string(),
        ));

        // j1939:plLamp
        triples.push(RdfTriple::literal(
            &dm1_iri,
            format!("{}plLamp", NS_J1939),
            dm1.pl_lamp.to_string(),
        ));

        // j1939:hasActiveFaults
        triples.push(RdfTriple::typed_literal(
            &dm1_iri,
            format!("{}hasActiveFaults", NS_J1939),
            dm1.has_active_faults().to_string(),
            format!("{}boolean", NS_XSD),
        ));

        // j1939:faultCount
        triples.push(RdfTriple::typed_literal(
            &dm1_iri,
            format!("{}faultCount", NS_J1939),
            dm1.fault_count().to_string(),
            format!("{}unsignedInt", NS_XSD),
        ));

        // prov:generatedAtTime
        triples.push(RdfTriple::typed_literal(
            &dm1_iri,
            format!("{}generatedAtTime", NS_PROV),
            timestamp_secs.to_string(),
            format!("{}integer", NS_XSD),
        ));

        // j1939:reportedByVehicle
        triples.push(RdfTriple::iri(
            &dm1_iri,
            format!("{}reportedByVehicle", NS_J1939),
            &self.vehicle_iri,
        ));

        // For each DTC, generate DTC triples and link them
        for dtc in &dm1.dtcs {
            let dtc_iri = format!(
                "{}/dtc/{}/{}/{}",
                self.base_iri, source_addr, dtc.spn, dtc.fmi
            );

            // j1939:containsDtc -> dtc IRI
            triples.push(RdfTriple::iri(
                &dm1_iri,
                format!("{}containsDtc", NS_J1939),
                &dtc_iri,
            ));

            // Add full DTC triples
            let dtc_triples = self.dtc_to_triples(dtc, source_addr, timestamp_secs);
            triples.extend(dtc_triples);
        }

        triples
    }

    /// Generate VSSo-compatible IRI for a named signal.
    ///
    /// If the signal maps to a well-known VSSo concept the VSSo IRI is
    /// returned; otherwise a J1939-namespaced IRI is returned.
    pub fn vsso_iri(&self, signal_name: &str) -> String {
        match vsso_concept(signal_name) {
            Some(concept) => format!("{}{}", NS_VSSO, concept),
            None => format!("{}{}", NS_J1939, sanitize_iri_fragment(signal_name)),
        }
    }

    /// Return the vehicle IRI used by this mapper.
    pub fn vehicle_iri(&self) -> &str {
        &self.vehicle_iri
    }

    /// Return the base IRI used by this mapper.
    pub fn base_iri(&self) -> &str {
        &self.base_iri
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Build a minimal J1939Message for feeding to the PGN registry decoder.
    fn build_message(&self, pgn: u32, data: &[u8]) -> J1939Message {
        use chrono::Utc;
        J1939Message {
            header: J1939Header {
                priority: 6,
                pgn: Pgn::new(pgn),
                source_address: 0x00,
                destination_address: None,
            },
            data: data.to_vec(),
            timestamp: Utc::now(),
            is_multipacket: false,
        }
    }

    /// Return an appropriate property IRI for a signal, preferring VSSo when possible.
    fn signal_property_iri(&self, pgn: u32, signal_name: &str) -> String {
        // Try VSSo mapping first
        if let Some(concept) = vsso_concept(signal_name) {
            return format!("{}{}", NS_VSSO, concept);
        }

        // Fall back to J1939 namespace with PGN context
        format!("{}{}/{}", NS_J1939, pgn, sanitize_iri_fragment(signal_name))
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Replace characters not legal in IRI path segments with underscores.
fn sanitize_iri_fragment(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Map a J1939 signal name to its VSSo concept, if known.
fn vsso_concept(signal_name: &str) -> Option<&'static str> {
    match signal_name {
        "EngineSpeed" | "ActualEnginePercentTorque" => Some("EngineSpeed"),
        "WheelBasedVehicleSpeed" | "CruiseControlSetSpeed" => Some("VehicleSpeed"),
        "AcceleratorPedalPosition1" | "AcceleratorPedalPosition2" => {
            Some("AcceleratorPedalPosition")
        }
        "EngineCoolantTemperature" => Some("EngineCoolantTemperature"),
        "EngineOilTemperature" => Some("EngineOilTemperature"),
        "EngineOilPressure" => Some("EngineOilPressure"),
        "EngineIntakeManifoldPressure" | "EngineAirIntakeTemperature" => {
            Some("IntakeManifoldPressure")
        }
        "FuelRate" => Some("FuelRate"),
        "InstantaneousFuelEconomy" => Some("InstantaneousFuelConsumption"),
        "AverageInstantaneousFuelEconomy" => Some("AverageFuelConsumption"),
        "BarometricPressure" => Some("AmbientAirPressure"),
        "AmbientAirTemperature" => Some("AmbientAirTemperature"),
        "ElectricalPotential" => Some("BatteryVoltage"),
        _ => None,
    }
}

/// Map a unit string from PGN decoders to a QUDT unit IRI.
fn unit_to_qudt_iri(unit: &str) -> String {
    let suffix = match unit {
        "rpm" | "RPM" => "REV-PER-MIN",
        "km/h" => "KiloM-PER-HR",
        "°C" | "deg C" | "C" => "DEG_C",
        "kPa" => "KiloPASCAL",
        "Pa" => "PA",
        "%" => "PERCENT",
        "L/h" => "L-PER-HR",
        "km/L" => "KiloM-PER-L",
        "V" => "V",
        "A" => "A",
        "Nm" | "N·m" => "N-M",
        "W" | "kW" => "W",
        "kg" => "KiloGM",
        "L" => "L",
        "m/s" => "M-PER-SEC",
        "bar" | "Bar" => "BAR",
        "h" => "HR",
        "s" | "sec" => "SEC",
        "Hz" => "HZ",
        _ => "UNITLESS",
    };
    format!("{}{}", NS_QUDT_UNIT, suffix)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::j1939::diagnostics::{DiagnosticTroubleCode, Dm1Message, LampStatus};

    fn make_mapper() -> CanToRdfMapper {
        CanToRdfMapper::new("http://example.com/vehicle", "urn:vehicle:truck001")
    }

    #[test]
    fn test_mapper_creation() {
        let m = make_mapper();
        assert_eq!(m.vehicle_iri(), "urn:vehicle:truck001");
        assert_eq!(m.base_iri(), "http://example.com/vehicle");
    }

    #[test]
    fn test_pgn_to_triples_eec1() {
        let m = make_mapper();
        // EEC1 data: engine speed 2000 rpm (raw = 2000/0.125 = 16000 = 0x3E80)
        let data = vec![0x00, 0x80, 0x3E, 0x80, 0x3E, 0x00, 0x00, 0x7D];
        let triples = m.pgn_to_triples(PGN_EEC1, &data, 1706000000);
        // Should produce multiple triples for signals
        assert!(!triples.is_empty());
        // Check for SOSA Observation type
        let has_observation = triples
            .iter()
            .any(|t| matches!(&t.object, RdfObject::Iri(iri) if iri.contains("Observation")));
        assert!(has_observation, "Expected sosa:Observation type triple");
    }

    #[test]
    fn test_pgn_to_triples_unknown_pgn() {
        let m = make_mapper();
        let triples = m.pgn_to_triples(99999, &[0u8; 8], 1706000000);
        assert!(triples.is_empty());
    }

    #[test]
    fn test_dtc_to_triples() {
        let m = make_mapper();
        let dtc = DiagnosticTroubleCode::new(190, 3); // Engine Speed, Voltage Above Normal
        let triples = m.dtc_to_triples(&dtc, 0x00, 1706000000);

        assert!(!triples.is_empty());

        // Must have rdf:type triple
        let has_type = triples.iter().any(|t| {
            t.predicate.contains("type")
                && matches!(&t.object, RdfObject::Iri(i) if i.contains("DiagnosticTroubleCode"))
        });
        assert!(has_type, "Missing rdf:type triple");

        // Must have SPN triple
        let has_spn = triples.iter().any(|t| t.predicate.contains("spn"));
        assert!(has_spn, "Missing spn triple");

        // Must have FMI description triple
        let has_fmi_desc = triples
            .iter()
            .any(|t| t.predicate.contains("fmiDescription"));
        assert!(has_fmi_desc, "Missing fmiDescription triple");

        // Engine Speed SPN should have description
        let has_spn_desc = triples
            .iter()
            .any(|t| t.predicate.contains("spnDescription"));
        assert!(has_spn_desc, "Missing spnDescription for known SPN 190");
    }

    #[test]
    fn test_dtc_to_triples_unknown_spn() {
        let m = make_mapper();
        let dtc = DiagnosticTroubleCode::new(99999, 11); // Unknown SPN
        let triples = m.dtc_to_triples(&dtc, 0x01, 1706000000);
        // Should still produce triples but without spnDescription
        let has_spn_desc = triples
            .iter()
            .any(|t| t.predicate.contains("spnDescription"));
        assert!(
            !has_spn_desc,
            "Should not have spnDescription for unknown SPN"
        );
    }

    #[test]
    fn test_dm1_to_triples() {
        let m = make_mapper();
        let dm1 = Dm1Message {
            mil_lamp: LampStatus::On,
            rsl_lamp: LampStatus::Off,
            awl_lamp: LampStatus::Off,
            pl_lamp: LampStatus::Off,
            dtcs: vec![
                DiagnosticTroubleCode::new(190, 3),
                DiagnosticTroubleCode::new(100, 4),
            ],
        };

        let triples = m.dm1_to_triples(&dm1, 0x00, 1706000000);
        assert!(!triples.is_empty());

        // DM1 message type triple
        let has_dm1_type = triples
            .iter()
            .any(|t| matches!(&t.object, RdfObject::Iri(i) if i.contains("DM1Message")));
        assert!(has_dm1_type, "Missing DM1Message type triple");

        // DTC link triples
        let has_dtc_link = triples.iter().any(|t| t.predicate.contains("containsDtc"));
        assert!(has_dtc_link, "Missing containsDtc link");

        // Lamp status triples
        let has_mil = triples.iter().any(|t| t.predicate.contains("milLamp"));
        assert!(has_mil, "Missing milLamp triple");
    }

    #[test]
    fn test_vsso_iri() {
        let m = make_mapper();
        assert!(m.vsso_iri("EngineSpeed").contains("vsso.ttl#EngineSpeed"));
        assert!(m
            .vsso_iri("EngineCoolantTemperature")
            .contains("EngineCoolantTemperature"));
        // Unknown signal falls back to J1939 namespace
        let unknown = m.vsso_iri("SomeMysterySignal");
        assert!(
            unknown.contains("j1939.oxirs.org"),
            "Unknown signal should use J1939 namespace"
        );
    }

    #[test]
    fn test_rdf_triple_turtle_serialization() {
        let t = RdfTriple::typed_literal(
            "http://example.com/obs/1",
            "http://www.w3.org/ns/sosa/hasSimpleResult",
            "2000.5",
            "http://www.w3.org/2001/XMLSchema#double",
        );
        let turtle = t.to_turtle();
        assert!(turtle.contains("<http://example.com/obs/1>"));
        assert!(turtle.contains("2000.5"));
        assert!(turtle.contains("XMLSchema#double"));
    }

    #[test]
    fn test_unit_to_qudt_mapping() {
        assert!(unit_to_qudt_iri("rpm").contains("REV-PER-MIN"));
        assert!(unit_to_qudt_iri("kPa").contains("KiloPASCAL"));
        assert!(unit_to_qudt_iri("%").contains("PERCENT"));
        assert!(unit_to_qudt_iri("V").contains("/V"));
        assert!(unit_to_qudt_iri("UnknownUnit").contains("UNITLESS"));
    }

    #[test]
    fn test_sanitize_iri_fragment() {
        assert_eq!(sanitize_iri_fragment("EngineSpeed"), "EngineSpeed");
        assert_eq!(sanitize_iri_fragment("Engine Speed"), "Engine_Speed");
        assert_eq!(sanitize_iri_fragment("Signal/Name"), "Signal_Name");
    }

    #[test]
    fn test_pgn_to_triples_ccvs() {
        let m = make_mapper();
        // CCVS data: vehicle speed ~80 km/h
        let data = vec![0xFF, 0x40, 0x1F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let triples = m.pgn_to_triples(PGN_CCVS, &data, 1706000000);
        // CCVS may produce triples if decoder returns valid signals
        // We just verify the call doesn't panic
        let _ = triples;
    }

    #[test]
    fn test_pgn_to_triples_has_vehicle_link() {
        let m = make_mapper();
        let data = vec![0x00, 0x7D, 0x7D, 0x80, 0x3E, 0x00, 0x00, 0x7D];
        let triples = m.pgn_to_triples(PGN_EEC1, &data, 1706000000);

        // At least one triple should reference the vehicle IRI
        let has_vehicle_ref = triples
            .iter()
            .any(|t| matches!(&t.object, RdfObject::Iri(iri) if iri.contains("truck001")));
        assert!(has_vehicle_ref, "Expected vehicle IRI in triples");
    }
}
