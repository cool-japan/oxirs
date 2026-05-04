//! DTDL v3 → RDF Mapping
//!
//! Converts a parsed [`DtdlInterface`] into a set of [`RdfTriple`]s using:
//! - `rdf:type` for element classification
//! - `rdfs:label` for display names
//! - `http://oxirs.io/physics/dtdl#` namespace for DTDL-specific predicates
//! - `http://qudt.org/schema/qudt/unit` for unit annotations where known
//! - `http://www.w3.org/2001/XMLSchema#*` for schema type references
//!
//! # Design
//!
//! The mapper is intentionally *additive*: it produces triples and does not
//! remove or mutate any existing graph state.  The resulting triple set can
//! be serialised to Turtle, N-Triples, or JSON-LD by the caller.

use super::types::{DtdlContent, DtdlInterface};

// ─────────────────────────────────────────────────────────────────────────────
// Namespace constants
// ─────────────────────────────────────────────────────────────────────────────

/// OxiRS DTDL vocabulary namespace.
pub const OXPHY_NS: &str = "http://oxirs.io/physics/dtdl#";

/// W3C RDF type predicate.
pub const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

/// RDFS label predicate.
pub const RDFS_LABEL: &str = "http://www.w3.org/2000/01/rdf-schema#label";

/// QUDT unit predicate.
pub const QUDT_UNIT: &str = "http://qudt.org/schema/qudt/unit";

// ─────────────────────────────────────────────────────────────────────────────
// RdfTriple
// ─────────────────────────────────────────────────────────────────────────────

/// A simple RDF triple represented as three strings (subject, predicate, object).
///
/// IRI subjects/objects are bare IRI strings; literal objects are enclosed in
/// double-quotes (`"value"`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdfTriple {
    /// Subject IRI.
    pub subject: String,
    /// Predicate IRI.
    pub predicate: String,
    /// Object — either an IRI or a quoted literal string.
    pub object: String,
}

impl RdfTriple {
    /// Construct a new triple.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }

    /// Returns `true` if the object looks like a quoted literal.
    pub fn object_is_literal(&self) -> bool {
        self.object.starts_with('"')
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QUDT unit mapping
// ─────────────────────────────────────────────────────────────────────────────

/// Map a DTDL unit name to a QUDT unit URI.
///
/// Returns `None` for unit names not present in the mapping table; the caller
/// then falls back to storing the raw unit string as a literal annotation.
pub fn dtdl_unit_to_qudt(unit: &str) -> Option<&'static str> {
    match unit {
        "Celsius" | "degreeCelsius" | "celsius" => Some("http://qudt.org/vocab/unit/DEG_C"),
        "Fahrenheit" | "degreeFahrenheit" | "fahrenheit" => {
            Some("http://qudt.org/vocab/unit/DEG_F")
        }
        "Kelvin" | "kelvin" => Some("http://qudt.org/vocab/unit/K"),
        "metre" | "meter" | "Meter" | "Metre" => Some("http://qudt.org/vocab/unit/M"),
        "centimetre" | "centimeter" => Some("http://qudt.org/vocab/unit/CentiM"),
        "millimetre" | "millimeter" => Some("http://qudt.org/vocab/unit/MilliM"),
        "kilogram" | "kilogramme" | "Kilogram" => Some("http://qudt.org/vocab/unit/KiloGM"),
        "gram" | "Gram" => Some("http://qudt.org/vocab/unit/GM"),
        "second" | "Second" => Some("http://qudt.org/vocab/unit/SEC"),
        "minute" | "Minute" => Some("http://qudt.org/vocab/unit/MIN"),
        "hour" | "Hour" => Some("http://qudt.org/vocab/unit/HR"),
        "pascal" | "Pascal" => Some("http://qudt.org/vocab/unit/PA"),
        "kilopascal" | "Kilopascal" => Some("http://qudt.org/vocab/unit/KiloPA"),
        "watt" | "Watt" => Some("http://qudt.org/vocab/unit/W"),
        "kilowatt" | "Kilowatt" => Some("http://qudt.org/vocab/unit/KiloW"),
        "kilowattHour" | "KilowattHour" | "kilowatt-hour" => {
            Some("http://qudt.org/vocab/unit/KiloW-HR")
        }
        "joule" | "Joule" => Some("http://qudt.org/vocab/unit/J"),
        "ampere" | "Ampere" | "amp" => Some("http://qudt.org/vocab/unit/A"),
        "volt" | "Volt" => Some("http://qudt.org/vocab/unit/V"),
        "ohm" | "Ohm" => Some("http://qudt.org/vocab/unit/OHM"),
        "hertz" | "Hertz" => Some("http://qudt.org/vocab/unit/HZ"),
        "newton" | "Newton" => Some("http://qudt.org/vocab/unit/N"),
        "radian" | "Radian" => Some("http://qudt.org/vocab/unit/RAD"),
        "degree" | "Degree" | "degreeOfArc" => Some("http://qudt.org/vocab/unit/DEG"),
        "metersPerSecond" | "metrePerSecond" | "mPerSecond" => {
            Some("http://qudt.org/vocab/unit/M-PER-SEC")
        }
        "kilometrePerHour" | "kilometerPerHour" => Some("http://qudt.org/vocab/unit/KiloM-PER-HR"),
        "percent" | "Percent" | "percentage" => Some("http://qudt.org/vocab/unit/PERCENT"),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main mapping function
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a parsed [`DtdlInterface`] to a set of [`RdfTriple`]s.
///
/// The DTMI of the interface becomes the subject IRI.  Content elements
/// are assigned sub-IRIs of the form `<interface-dtmi>#<name>` unless they
/// carry their own `@id`.
pub fn interface_to_rdf(iface: &DtdlInterface) -> Vec<RdfTriple> {
    let mut triples: Vec<RdfTriple> = Vec::new();
    let iface_iri = iface.id.0.clone();

    // Interface rdf:type
    triples.push(RdfTriple::new(
        iface_iri.clone(),
        RDF_TYPE,
        format!("{OXPHY_NS}DtdlInterface"),
    ));

    // Display name → rdfs:label
    if let Some(dn) = &iface.display_name {
        let label = extract_display_name(dn);
        if !label.is_empty() {
            triples.push(RdfTriple::new(
                iface_iri.clone(),
                RDFS_LABEL,
                format!("\"{label}\""),
            ));
        }
    }

    // Description
    if let Some(desc) = &iface.description {
        triples.push(RdfTriple::new(
            iface_iri.clone(),
            format!("{OXPHY_NS}description"),
            format!("\"{desc}\""),
        ));
    }

    // Comment
    if let Some(comment) = &iface.comment {
        triples.push(RdfTriple::new(
            iface_iri.clone(),
            format!("{OXPHY_NS}comment"),
            format!("\"{comment}\""),
        ));
    }

    // extends
    if let Some(ext) = &iface.extends {
        for base_iri in extract_extends_iris(ext) {
            triples.push(RdfTriple::new(
                iface_iri.clone(),
                format!("{OXPHY_NS}extends"),
                base_iri,
            ));
        }
    }

    // Content elements
    for content in iface.contents.as_deref().unwrap_or(&[]) {
        let content_triples = map_content(&iface_iri, content);
        triples.extend(content_triples);
    }

    triples
}

// ─────────────────────────────────────────────────────────────────────────────
// Content element mappers
// ─────────────────────────────────────────────────────────────────────────────

fn map_content(iface_iri: &str, content: &DtdlContent) -> Vec<RdfTriple> {
    match content {
        DtdlContent::Telemetry(t) => {
            let iri =
                t.id.as_ref()
                    .map(|d| d.0.clone())
                    .unwrap_or_else(|| format!("{iface_iri}#{}", t.name));

            let mut v = vec![
                RdfTriple::new(iri.clone(), RDF_TYPE, format!("{OXPHY_NS}DtdlTelemetry")),
                RdfTriple::new(
                    iface_iri.to_owned(),
                    format!("{OXPHY_NS}hasTelemetry"),
                    iri.clone(),
                ),
                RdfTriple::new(
                    iri.clone(),
                    format!("{OXPHY_NS}name"),
                    format!("\"{}\"", t.name),
                ),
                RdfTriple::new(
                    iri.clone(),
                    format!("{OXPHY_NS}schema"),
                    t.schema.to_xsd_uri().to_owned(),
                ),
            ];

            if let Some(unit) = &t.unit {
                push_unit_triple(&mut v, &iri, unit);
            }

            // Semantic type annotations (second+ elements in the @type array)
            if let serde_json::Value::Array(arr) = &t.element_type {
                for extra in arr.iter().skip(1) {
                    if let Some(sem_type) = extra.as_str() {
                        v.push(RdfTriple::new(
                            iri.clone(),
                            format!("{OXPHY_NS}semanticType"),
                            format!("{OXPHY_NS}{sem_type}"),
                        ));
                    }
                }
            }

            if let Some(desc) = &t.description {
                v.push(RdfTriple::new(
                    iri.clone(),
                    format!("{OXPHY_NS}description"),
                    format!("\"{desc}\""),
                ));
            }

            if let Some(comment) = &t.comment {
                v.push(RdfTriple::new(
                    iri.clone(),
                    format!("{OXPHY_NS}comment"),
                    format!("\"{comment}\""),
                ));
            }

            v
        }

        DtdlContent::Property(p) => {
            let iri =
                p.id.as_ref()
                    .map(|d| d.0.clone())
                    .unwrap_or_else(|| format!("{iface_iri}#{}", p.name));

            let mut v = vec![
                RdfTriple::new(iri.clone(), RDF_TYPE, format!("{OXPHY_NS}DtdlProperty")),
                RdfTriple::new(
                    iface_iri.to_owned(),
                    format!("{OXPHY_NS}hasProperty"),
                    iri.clone(),
                ),
                RdfTriple::new(
                    iri.clone(),
                    format!("{OXPHY_NS}name"),
                    format!("\"{}\"", p.name),
                ),
                RdfTriple::new(
                    iri.clone(),
                    format!("{OXPHY_NS}schema"),
                    p.schema.to_xsd_uri().to_owned(),
                ),
            ];

            if let Some(writable) = p.writable {
                v.push(RdfTriple::new(
                    iri.clone(),
                    format!("{OXPHY_NS}writable"),
                    format!("\"{writable}\""),
                ));
            }

            if let Some(unit) = &p.unit {
                push_unit_triple(&mut v, &iri, unit);
            }

            v
        }

        DtdlContent::Command(c) => {
            let iri = format!("{iface_iri}#{}", c.name);
            vec![
                RdfTriple::new(iri.clone(), RDF_TYPE, format!("{OXPHY_NS}DtdlCommand")),
                RdfTriple::new(
                    iface_iri.to_owned(),
                    format!("{OXPHY_NS}hasCommand"),
                    iri.clone(),
                ),
                RdfTriple::new(iri, format!("{OXPHY_NS}name"), format!("\"{}\"", c.name)),
            ]
        }

        DtdlContent::Component(comp) => {
            let iri = format!("{iface_iri}#{}", comp.name);
            vec![
                RdfTriple::new(iri.clone(), RDF_TYPE, format!("{OXPHY_NS}DtdlComponent")),
                RdfTriple::new(
                    iface_iri.to_owned(),
                    format!("{OXPHY_NS}hasComponent"),
                    iri.clone(),
                ),
                RdfTriple::new(
                    iri.clone(),
                    format!("{OXPHY_NS}name"),
                    format!("\"{}\"", comp.name),
                ),
                RdfTriple::new(iri, format!("{OXPHY_NS}schema"), comp.schema.0.clone()),
            ]
        }

        DtdlContent::Relationship(rel) => {
            let iri = format!("{iface_iri}#{}", rel.name);
            let mut v = vec![
                RdfTriple::new(iri.clone(), RDF_TYPE, format!("{OXPHY_NS}DtdlRelationship")),
                RdfTriple::new(
                    iface_iri.to_owned(),
                    format!("{OXPHY_NS}hasRelationship"),
                    iri.clone(),
                ),
                RdfTriple::new(
                    iri.clone(),
                    format!("{OXPHY_NS}name"),
                    format!("\"{}\"", rel.name),
                ),
            ];

            if let Some(target) = &rel.target {
                v.push(RdfTriple::new(
                    iri.clone(),
                    format!("{OXPHY_NS}target"),
                    target.0.clone(),
                ));
            }

            if let Some(desc) = &rel.description {
                v.push(RdfTriple::new(
                    iri,
                    format!("{OXPHY_NS}description"),
                    format!("\"{desc}\""),
                ));
            }

            v
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Push a unit triple for the given element IRI.
///
/// If the unit maps to a known QUDT URI, the object is an IRI; otherwise the
/// raw unit string is stored as a literal under `oxphy:unit`.
fn push_unit_triple(triples: &mut Vec<RdfTriple>, iri: &str, unit: &str) {
    match dtdl_unit_to_qudt(unit) {
        Some(qudt_uri) => {
            triples.push(RdfTriple::new(iri, QUDT_UNIT, qudt_uri));
        }
        None => {
            triples.push(RdfTriple::new(
                iri,
                format!("{OXPHY_NS}unit"),
                format!("\"{unit}\""),
            ));
        }
    }
}

/// Extract a display name string from a DTDL `displayName` value.
///
/// Handles:
/// - Plain string: `"Thermostat"`
/// - Language map object: `{ "en": "Thermostat", "de": "Thermostat" }`
fn extract_display_name(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Object(map) => {
            // Prefer English; fall back to any entry
            map.get("en")
                .or_else(|| map.values().next())
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned()
        }
        _ => String::new(),
    }
}

/// Extract a list of base interface IRIs from the DTDL `extends` value.
///
/// Handles both a single DTMI string and an array of DTMI strings.
fn extract_extends_iris(value: &serde_json::Value) -> Vec<String> {
    match value {
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str())
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

/// Reverse lookup: given an element IRI, return the content element type.
///
/// This is a lightweight helper for the reverse mapper; it inspects the
/// triples for a `rdf:type` predicate with one of the known `oxphy:Dtdl*`
/// objects.
pub fn rdf_type_to_content_kind(triples: &[RdfTriple], element_iri: &str) -> Option<&'static str> {
    for triple in triples {
        if triple.subject == element_iri && triple.predicate == RDF_TYPE {
            let kind = match triple.object.as_str() {
                s if s == format!("{OXPHY_NS}DtdlTelemetry") => "Telemetry",
                s if s == format!("{OXPHY_NS}DtdlProperty") => "Property",
                s if s == format!("{OXPHY_NS}DtdlCommand") => "Command",
                s if s == format!("{OXPHY_NS}DtdlComponent") => "Component",
                s if s == format!("{OXPHY_NS}DtdlRelationship") => "Relationship",
                _ => continue,
            };
            return Some(kind);
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtdl::parser::parse_dtdl_interface;

    #[test]
    fn empty_interface_produces_type_triple() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:Empty;1"
        }"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        let triples = interface_to_rdf(&iface);
        assert!(triples
            .iter()
            .any(|t| t.subject == "dtmi:test:Empty;1" && t.predicate == RDF_TYPE));
    }

    #[test]
    fn display_name_becomes_label() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:Named;1",
            "displayName": "My Device"
        }"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        let triples = interface_to_rdf(&iface);
        let label = triples
            .iter()
            .find(|t| t.predicate == RDFS_LABEL)
            .expect("label triple");
        assert_eq!(label.object, "\"My Device\"");
    }

    #[test]
    fn language_map_display_name() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:Multi;1",
            "displayName": { "en": "Sensor", "de": "Sensor DE" }
        }"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        let triples = interface_to_rdf(&iface);
        let label = triples
            .iter()
            .find(|t| t.predicate == RDFS_LABEL)
            .expect("label triple");
        assert_eq!(label.object, "\"Sensor\"");
    }

    #[test]
    fn telemetry_mapped_with_qudt_celsius() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:Thermo;1",
            "contents": [
                { "@type": ["Telemetry","Temperature"], "name": "temp", "schema": "double", "unit": "Celsius" }
            ]
        }"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        let triples = interface_to_rdf(&iface);

        // Has hasTelemetry link
        assert!(triples.iter().any(|t| t.predicate.contains("hasTelemetry")));

        // Has QUDT Celsius unit
        assert!(triples.iter().any(|t| t.object.contains("DEG_C")));
    }

    #[test]
    fn property_with_writable_flag() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:Prop;1",
            "contents": [
                { "@type": "Property", "name": "target", "schema": "double", "writable": true }
            ]
        }"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        let triples = interface_to_rdf(&iface);
        assert!(triples.iter().any(|t| t.predicate.contains("writable")));
        assert!(triples.iter().any(|t| t.predicate.contains("hasProperty")));
    }

    #[test]
    fn command_mapped() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:Cmd;1",
            "contents": [{ "@type": "Command", "name": "reboot" }]
        }"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        let triples = interface_to_rdf(&iface);
        assert!(triples.iter().any(|t| t.predicate.contains("hasCommand")));
    }

    #[test]
    fn relationship_with_target() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:Building;1",
            "contents": [
                { "@type": "Relationship", "name": "contains", "target": "dtmi:test:Room;1" }
            ]
        }"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        let triples = interface_to_rdf(&iface);
        assert!(triples.iter().any(|t| t.predicate.contains("target")));
        assert!(triples.iter().any(|t| t.object == "dtmi:test:Room;1"));
    }

    #[test]
    fn unknown_unit_stored_as_literal() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:U;1",
            "contents": [
                { "@type": "Telemetry", "name": "x", "schema": "double", "unit": "furlongPerFortnight" }
            ]
        }"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        let triples = interface_to_rdf(&iface);
        let unit_triple = triples
            .iter()
            .find(|t| t.predicate.ends_with("unit"))
            .expect("unit triple");
        assert!(unit_triple.object_is_literal());
        assert!(unit_triple.object.contains("furlongPerFortnight"));
    }

    #[test]
    fn rdf_type_to_content_kind_telemetry() {
        let triples = vec![RdfTriple::new(
            "dtmi:x;1#temp",
            RDF_TYPE,
            format!("{OXPHY_NS}DtdlTelemetry"),
        )];
        assert_eq!(
            rdf_type_to_content_kind(&triples, "dtmi:x;1#temp"),
            Some("Telemetry")
        );
    }

    #[test]
    fn dtdl_unit_to_qudt_coverage() {
        assert!(dtdl_unit_to_qudt("Celsius").is_some());
        assert!(dtdl_unit_to_qudt("watt").is_some());
        assert!(dtdl_unit_to_qudt("kilowattHour").is_some());
        assert!(dtdl_unit_to_qudt("unknownUnit").is_none());
    }

    #[test]
    fn extends_single_iri_mapped() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:Child;1",
            "extends": "dtmi:test:Base;1"
        }"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        let triples = interface_to_rdf(&iface);
        assert!(triples.iter().any(|t| t.predicate.contains("extends")));
        assert!(triples.iter().any(|t| t.object == "dtmi:test:Base;1"));
    }

    #[test]
    fn semantic_type_annotation_triple() {
        let json = r#"{
            "@type": "Interface",
            "@id": "dtmi:test:ST;1",
            "contents": [
                { "@type": ["Telemetry","Humidity"], "name": "hum", "schema": "double" }
            ]
        }"#;
        let iface = parse_dtdl_interface(json).expect("parse");
        let triples = interface_to_rdf(&iface);
        assert!(triples
            .iter()
            .any(|t| t.predicate.contains("semanticType") && t.object.contains("Humidity")));
    }
}
