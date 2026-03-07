//! RDF/SPARQL Extraction for Physics Models
//!
//! Extracts physics data from SPARQL query result rows and converts them
//! to first-class physics types (`FemMaterial`, `FemNode`, `NodalLoad`).
//! Also provides a `PhysicsQueryBuilder` that generates ready-to-use
//! SPARQL SELECT strings for common physics queries.
//!
//! # Example
//!
//! ```rust
//! use oxirs_physics::rdf_extraction::{
//!     SparqlBinding, SparqlRow, SparqlValue, PhysicsRdfExtractor, PhysicsQueryBuilder,
//! };
//!
//! // Build a row that represents a material
//! let row = SparqlRow(vec![
//!     SparqlBinding { variable: "youngsModulus".to_string(),
//!                     value: SparqlValue::Literal("200e9".to_string(), None) },
//!     SparqlBinding { variable: "poissonsRatio".to_string(),
//!                     value: SparqlValue::Literal("0.3".to_string(), None) },
//!     SparqlBinding { variable: "thermalConductivity".to_string(),
//!                     value: SparqlValue::Literal("50.0".to_string(), None) },
//!     SparqlBinding { variable: "density".to_string(),
//!                     value: SparqlValue::Literal("7850.0".to_string(), None) },
//! ]);
//! let mats = PhysicsRdfExtractor::extract_materials(&[row]);
//! assert_eq!(mats.len(), 1);
//!
//! // Build a SPARQL query string
//! let q = PhysicsQueryBuilder::material_query("http://example.org/mat#Steel");
//! assert!(q.contains("SELECT"));
//! ```

use crate::fem::{FemMaterial, NodalLoad};
use crate::samm::fem_bridge::{SammAspect, SammDataType, SammProperty};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────
// SPARQL result row types
// ─────────────────────────────────────────────

/// A single RDF term value returned from a SPARQL query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SparqlValue {
    /// IRI resource.
    Iri(String),
    /// Literal with optional datatype IRI.
    Literal(String, Option<String>),
    /// Blank node identifier.
    BlankNode(String),
}

impl SparqlValue {
    /// Extract the string representation (lexical form for literals, IRI for IRIs, id for blanks).
    pub fn as_str(&self) -> &str {
        match self {
            SparqlValue::Iri(s) => s,
            SparqlValue::Literal(s, _) => s,
            SparqlValue::BlankNode(s) => s,
        }
    }

    /// Try to parse as f64.
    pub fn as_f64(&self) -> Option<f64> {
        self.as_str().parse::<f64>().ok()
    }

    /// Try to parse as i64.
    pub fn as_i64(&self) -> Option<i64> {
        self.as_str().parse::<i64>().ok()
    }
}

/// A named binding from a SPARQL result (one variable → one value).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlBinding {
    /// Variable name (without leading `?`).
    pub variable: String,
    /// Bound value.
    pub value: SparqlValue,
}

/// One row of a SPARQL SELECT result (ordered list of bindings).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlRow(pub Vec<SparqlBinding>);

impl SparqlRow {
    /// Look up the value bound to `variable`, returning `None` when absent.
    pub fn get(&self, variable: &str) -> Option<&SparqlValue> {
        self.0
            .iter()
            .find(|b| b.variable == variable)
            .map(|b| &b.value)
    }

    /// Look up a variable and try to parse it as `f64`.
    pub fn get_f64(&self, variable: &str) -> Option<f64> {
        self.get(variable)?.as_f64()
    }

    /// Look up a variable and try to parse it as `i64`.
    pub fn get_i64(&self, variable: &str) -> Option<i64> {
        self.get(variable)?.as_i64()
    }
}

// ─────────────────────────────────────────────
// PhysicsRdfExtractor
// ─────────────────────────────────────────────

/// Extracts physics domain objects from SPARQL result rows.
pub struct PhysicsRdfExtractor;

impl PhysicsRdfExtractor {
    /// Extract `FemMaterial` values from SPARQL rows.
    ///
    /// Each row that contains at least `youngsModulus` **or**
    /// `thermalConductivity` yields one `FemMaterial`.  Missing fields fall
    /// back to sensible defaults.
    ///
    /// Expected variables: `youngsModulus`, `poissonsRatio`,
    /// `thermalConductivity`, `density`.
    pub fn extract_materials(rows: &[SparqlRow]) -> Vec<FemMaterial> {
        rows.iter()
            .filter_map(|row| {
                let has_data = row.get("youngsModulus").is_some()
                    || row.get("thermalConductivity").is_some()
                    || row.get("density").is_some();
                if !has_data {
                    return None;
                }
                Some(FemMaterial {
                    youngs_modulus: row.get_f64("youngsModulus").unwrap_or(200e9),
                    poissons_ratio: row.get_f64("poissonsRatio").unwrap_or(0.3),
                    thermal_conductivity: row.get_f64("thermalConductivity").unwrap_or(50.0),
                    density: row.get_f64("density").unwrap_or(7850.0),
                })
            })
            .collect()
    }

    /// Extract node (x, y) coordinate pairs from SPARQL rows.
    ///
    /// Expected variables: `x`, `y`.
    pub fn extract_nodes(rows: &[SparqlRow]) -> Vec<(f64, f64)> {
        rows.iter()
            .filter_map(|row| {
                let x = row.get_f64("x")?;
                let y = row.get_f64("y")?;
                Some((x, y))
            })
            .collect()
    }

    /// Extract `NodalLoad` values from SPARQL rows.
    ///
    /// Expected variables: `nodeId` (integer), `fx`, `fy`.
    pub fn extract_loads(rows: &[SparqlRow]) -> Vec<NodalLoad> {
        rows.iter()
            .filter_map(|row| {
                let node_id = row.get_i64("nodeId")? as usize;
                let fx = row.get_f64("fx").unwrap_or(0.0);
                let fy = row.get_f64("fy").unwrap_or(0.0);
                Some(NodalLoad { node_id, fx, fy })
            })
            .collect()
    }

    /// Convert a slice of `(subject, predicate, object)` RDF triples to a
    /// `SammAspect` using property-name heuristics.
    ///
    /// The `urn` of the resulting aspect is taken from the most common subject
    /// (or `"urn:unknown"` when the triple slice is empty).  The aspect name
    /// is derived from the last fragment/path segment of the URN.
    pub fn rdf_triples_to_samm(triples: &[(String, String, String)]) -> SammAspect {
        if triples.is_empty() {
            return SammAspect {
                urn: "urn:unknown".to_string(),
                name: "Unknown".to_string(),
                properties: vec![],
            };
        }

        // Determine the most frequent subject as the aspect IRI.
        let mut subject_counts: HashMap<&str, usize> = HashMap::new();
        for (s, _, _) in triples {
            *subject_counts.entry(s.as_str()).or_insert(0) += 1;
        }
        let aspect_iri = subject_counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(s, _)| s)
            .unwrap_or("urn:unknown");

        // Derive a human-readable name from the IRI.
        let name = iri_local_name(aspect_iri);

        // Build properties from predicate/object pairs.
        let properties: Vec<SammProperty> = triples
            .iter()
            .filter(|(s, _, _)| s.as_str() == aspect_iri)
            .map(|(_, p, o)| {
                let prop_name = iri_local_name(p.as_str());
                let (data_type, value) = infer_samm_property(o.as_str());
                SammProperty {
                    name: prop_name.to_string(),
                    data_type,
                    unit: None,
                    value: Some(value),
                }
            })
            .collect();

        SammAspect {
            urn: aspect_iri.to_string(),
            name: name.to_string(),
            properties,
        }
    }

    /// Extract time-series data from SPARQL rows.
    ///
    /// Expected variables: `timestamp` (integer Unix epoch seconds), `value` (f64).
    pub fn sparql_result_to_time_series(rows: &[SparqlRow]) -> Vec<(i64, f64)> {
        rows.iter()
            .filter_map(|row| {
                let ts = row.get_i64("timestamp")?;
                let val = row.get_f64("value")?;
                Some((ts, val))
            })
            .collect()
    }
}

// ─────────────────────────────────────────────
// PhysicsQueryBuilder
// ─────────────────────────────────────────────

/// Generates SPARQL SELECT queries for common physics data retrieval patterns.
pub struct PhysicsQueryBuilder;

impl PhysicsQueryBuilder {
    /// SPARQL SELECT query to retrieve material properties for a given IRI.
    ///
    /// Returns columns: `youngsModulus`, `poissonsRatio`,
    /// `thermalConductivity`, `density`.
    pub fn material_query(material_iri: &str) -> String {
        format!(
            r#"PREFIX phys: <http://oxirs.org/physics#>
PREFIX qudt: <http://qudt.org/schema/qudt/>
PREFIX unit: <http://qudt.org/vocab/unit/>

SELECT ?youngsModulus ?poissonsRatio ?thermalConductivity ?density
WHERE {{
  <{iri}> phys:youngsModulus ?youngsModulus ;
           phys:poissonsRatio ?poissonsRatio .
  OPTIONAL {{ <{iri}> phys:thermalConductivity ?thermalConductivity }}
  OPTIONAL {{ <{iri}> phys:density ?density }}
}}"#,
            iri = material_iri
        )
    }

    /// SPARQL SELECT query to retrieve node coordinates and element connectivity
    /// for a structural mesh.
    pub fn structure_query(structure_iri: &str) -> String {
        format!(
            r#"PREFIX phys: <http://oxirs.org/physics#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

SELECT ?nodeId ?x ?y ?fx ?fy
WHERE {{
  <{iri}> phys:hasNode ?node .
  ?node phys:nodeId ?nodeId ;
        phys:x ?x ;
        phys:y ?y .
  OPTIONAL {{ ?node phys:loadFx ?fx }}
  OPTIONAL {{ ?node phys:loadFy ?fy }}
}}
ORDER BY ?nodeId"#,
            iri = structure_iri
        )
    }

    /// SPARQL SELECT query to retrieve stored simulation results for a simulation IRI.
    pub fn simulation_result_query(sim_iri: &str) -> String {
        format!(
            r#"PREFIX phys: <http://oxirs.org/physics#>
PREFIX sosa: <http://www.w3.org/ns/sosa/>
PREFIX qudt: <http://qudt.org/schema/qudt/>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

SELECT ?property ?value ?unit ?timestamp
WHERE {{
  <{iri}> sosa:hasResult ?obs .
  ?obs phys:property ?property ;
       qudt:numericValue ?value .
  OPTIONAL {{ ?obs qudt:unit ?unit }}
  OPTIONAL {{ ?obs phys:timestamp ?timestamp }}
}}
ORDER BY ?property"#,
            iri = sim_iri
        )
    }
}

// ─────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────

/// Extract the local name (fragment or last path segment) from an IRI.
fn iri_local_name(iri: &str) -> &str {
    // Prefer fragment identifier
    if let Some(pos) = iri.rfind('#') {
        return &iri[pos + 1..];
    }
    // Fall back to last path segment
    if let Some(pos) = iri.rfind('/') {
        return &iri[pos + 1..];
    }
    // Last resort: last ':' segment (e.g. for URNs)
    if let Some(pos) = iri.rfind(':') {
        return &iri[pos + 1..];
    }
    iri
}

/// Guess the SAMM data type and produce a JSON value from a literal string.
fn infer_samm_property(literal: &str) -> (SammDataType, serde_json::Value) {
    // Try integer first (strict: no decimal point)
    if !literal.contains('.') {
        if let Ok(i) = literal.parse::<i64>() {
            return (SammDataType::Integer, serde_json::Value::Number(i.into()));
        }
    }
    // Try float
    if let Ok(f) = literal.parse::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(f) {
            return (SammDataType::Float, serde_json::Value::Number(n));
        }
    }
    // Boolean
    if literal.eq_ignore_ascii_case("true") {
        return (SammDataType::Boolean, serde_json::Value::Bool(true));
    }
    if literal.eq_ignore_ascii_case("false") {
        return (SammDataType::Boolean, serde_json::Value::Bool(false));
    }
    // Default: string
    (
        SammDataType::String,
        serde_json::Value::String(literal.to_string()),
    )
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Helpers ----

    fn lit(var: &str, val: &str) -> SparqlBinding {
        SparqlBinding {
            variable: var.to_string(),
            value: SparqlValue::Literal(val.to_string(), None),
        }
    }

    fn material_row(e: &str, nu: &str, k: &str, rho: &str) -> SparqlRow {
        SparqlRow(vec![
            lit("youngsModulus", e),
            lit("poissonsRatio", nu),
            lit("thermalConductivity", k),
            lit("density", rho),
        ])
    }

    // ────────────────────────────────────────
    // SparqlValue
    // ────────────────────────────────────────

    #[test]
    fn test_sparql_value_as_str_literal() {
        let v = SparqlValue::Literal("42.0".to_string(), None);
        assert_eq!(v.as_str(), "42.0");
    }

    #[test]
    fn test_sparql_value_as_str_iri() {
        let v = SparqlValue::Iri("http://example.org/foo".to_string());
        assert_eq!(v.as_str(), "http://example.org/foo");
    }

    #[test]
    fn test_sparql_value_as_f64_success() {
        let v = SparqlValue::Literal("3.125".to_string(), None);
        assert!((v.as_f64().expect("should parse") - 3.125).abs() < 1e-10);
    }

    #[test]
    fn test_sparql_value_as_f64_fail() {
        let v = SparqlValue::Literal("not-a-number".to_string(), None);
        assert!(v.as_f64().is_none());
    }

    #[test]
    fn test_sparql_value_as_i64_success() {
        let v = SparqlValue::Literal("42".to_string(), None);
        assert_eq!(v.as_i64().expect("should parse"), 42);
    }

    // ────────────────────────────────────────
    // SparqlRow
    // ────────────────────────────────────────

    #[test]
    fn test_sparql_row_get_present() {
        let row = SparqlRow(vec![lit("x", "1.0"), lit("y", "2.0")]);
        assert!(row.get("x").is_some());
        assert!(row.get("z").is_none());
    }

    #[test]
    fn test_sparql_row_get_f64() {
        let row = SparqlRow(vec![lit("x", "3.5")]);
        assert!((row.get_f64("x").expect("should parse") - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_sparql_row_get_i64() {
        let row = SparqlRow(vec![lit("nodeId", "7")]);
        assert_eq!(row.get_i64("nodeId").expect("should parse"), 7);
    }

    // ────────────────────────────────────────
    // extract_materials
    // ────────────────────────────────────────

    #[test]
    fn test_extract_materials_single_row() {
        let rows = vec![material_row("200000000000.0", "0.3", "50.0", "7850.0")];
        let mats = PhysicsRdfExtractor::extract_materials(&rows);
        assert_eq!(mats.len(), 1);
        assert!((mats[0].youngs_modulus - 200e9).abs() / 200e9 < 0.01);
        assert!((mats[0].poissons_ratio - 0.3).abs() < 1e-6);
        assert!((mats[0].thermal_conductivity - 50.0).abs() < 1e-6);
        assert!((mats[0].density - 7850.0).abs() < 1.0);
    }

    #[test]
    fn test_extract_materials_defaults_for_missing_fields() {
        // Only youngsModulus present — others should default
        let rows = vec![SparqlRow(vec![lit("youngsModulus", "70000000000.0")])];
        let mats = PhysicsRdfExtractor::extract_materials(&rows);
        assert_eq!(mats.len(), 1);
        assert!((mats[0].youngs_modulus - 70e9).abs() / 70e9 < 0.01);
        // Default poisson's ratio
        assert!((mats[0].poissons_ratio - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_extract_materials_empty_rows() {
        let rows: Vec<SparqlRow> = vec![];
        let mats = PhysicsRdfExtractor::extract_materials(&rows);
        assert!(mats.is_empty());
    }

    #[test]
    fn test_extract_materials_row_without_material_data_skipped() {
        // Row only has timestamp and value — not material data
        let rows = vec![SparqlRow(vec![
            lit("timestamp", "1000"),
            lit("value", "42.0"),
        ])];
        let mats = PhysicsRdfExtractor::extract_materials(&rows);
        assert!(mats.is_empty());
    }

    #[test]
    fn test_extract_materials_multiple_rows() {
        let rows = vec![
            material_row("200000000000.0", "0.3", "50.0", "7850.0"),
            material_row("70000000000.0", "0.33", "205.0", "2700.0"),
        ];
        let mats = PhysicsRdfExtractor::extract_materials(&rows);
        assert_eq!(mats.len(), 2);
    }

    // ────────────────────────────────────────
    // extract_nodes
    // ────────────────────────────────────────

    #[test]
    fn test_extract_nodes_basic() {
        let rows = vec![
            SparqlRow(vec![lit("x", "0.0"), lit("y", "0.0")]),
            SparqlRow(vec![lit("x", "1.0"), lit("y", "0.0")]),
            SparqlRow(vec![lit("x", "0.5"), lit("y", "1.0")]),
        ];
        let nodes = PhysicsRdfExtractor::extract_nodes(&rows);
        assert_eq!(nodes.len(), 3);
        assert!((nodes[1].0 - 1.0).abs() < 1e-10);
        assert!((nodes[2].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_nodes_missing_y_skipped() {
        let rows = vec![
            SparqlRow(vec![lit("x", "1.0")]), // missing y
            SparqlRow(vec![lit("x", "2.0"), lit("y", "3.0")]),
        ];
        let nodes = PhysicsRdfExtractor::extract_nodes(&rows);
        assert_eq!(nodes.len(), 1);
        assert!((nodes[0].0 - 2.0).abs() < 1e-10);
    }

    // ────────────────────────────────────────
    // extract_loads
    // ────────────────────────────────────────

    #[test]
    fn test_extract_loads_basic() {
        let rows = vec![
            SparqlRow(vec![
                lit("nodeId", "0"),
                lit("fx", "1000.0"),
                lit("fy", "0.0"),
            ]),
            SparqlRow(vec![
                lit("nodeId", "1"),
                lit("fx", "0.0"),
                lit("fy", "-500.0"),
            ]),
        ];
        let loads = PhysicsRdfExtractor::extract_loads(&rows);
        assert_eq!(loads.len(), 2);
        assert_eq!(loads[0].node_id, 0);
        assert!((loads[0].fx - 1000.0).abs() < 1e-10);
        assert!((loads[1].fy + 500.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_loads_defaults_fx_fy_to_zero() {
        let rows = vec![SparqlRow(vec![lit("nodeId", "3")])]; // no fx / fy
        let loads = PhysicsRdfExtractor::extract_loads(&rows);
        assert_eq!(loads.len(), 1);
        assert!((loads[0].fx).abs() < 1e-10);
        assert!((loads[0].fy).abs() < 1e-10);
    }

    #[test]
    fn test_extract_loads_missing_node_id_skipped() {
        let rows = vec![SparqlRow(vec![lit("fx", "500.0"), lit("fy", "0.0")])];
        let loads = PhysicsRdfExtractor::extract_loads(&rows);
        assert!(loads.is_empty());
    }

    // ────────────────────────────────────────
    // rdf_triples_to_samm
    // ────────────────────────────────────────

    #[test]
    fn test_rdf_triples_to_samm_empty() {
        let samm = PhysicsRdfExtractor::rdf_triples_to_samm(&[]);
        assert_eq!(samm.urn, "urn:unknown");
    }

    #[test]
    fn test_rdf_triples_to_samm_basic() {
        let triples = vec![
            (
                "http://example.org/mat#Steel".to_string(),
                "http://oxirs.org/physics#youngsModulus".to_string(),
                "200000000000".to_string(),
            ),
            (
                "http://example.org/mat#Steel".to_string(),
                "http://oxirs.org/physics#density".to_string(),
                "7850".to_string(),
            ),
        ];
        let samm = PhysicsRdfExtractor::rdf_triples_to_samm(&triples);
        assert_eq!(samm.urn, "http://example.org/mat#Steel");
        assert_eq!(samm.name, "Steel");
        assert_eq!(samm.properties.len(), 2);
    }

    #[test]
    fn test_rdf_triples_to_samm_string_property() {
        let triples = vec![(
            "urn:example:model".to_string(),
            "http://schema.org/name".to_string(),
            "My Model".to_string(),
        )];
        let samm = PhysicsRdfExtractor::rdf_triples_to_samm(&triples);
        assert_eq!(samm.properties[0].data_type, SammDataType::String);
    }

    // ────────────────────────────────────────
    // sparql_result_to_time_series
    // ────────────────────────────────────────

    #[test]
    fn test_time_series_extraction() {
        let rows = vec![
            SparqlRow(vec![lit("timestamp", "1000"), lit("value", "23.5")]),
            SparqlRow(vec![lit("timestamp", "2000"), lit("value", "24.1")]),
            SparqlRow(vec![lit("timestamp", "3000"), lit("value", "25.0")]),
        ];
        let series = PhysicsRdfExtractor::sparql_result_to_time_series(&rows);
        assert_eq!(series.len(), 3);
        assert_eq!(series[0].0, 1000);
        assert!((series[1].1 - 24.1).abs() < 1e-10);
    }

    #[test]
    fn test_time_series_missing_value_skipped() {
        let rows = vec![
            SparqlRow(vec![lit("timestamp", "1000")]), // missing value
            SparqlRow(vec![lit("timestamp", "2000"), lit("value", "99.0")]),
        ];
        let series = PhysicsRdfExtractor::sparql_result_to_time_series(&rows);
        assert_eq!(series.len(), 1);
        assert_eq!(series[0].0, 2000);
    }

    // ────────────────────────────────────────
    // PhysicsQueryBuilder
    // ────────────────────────────────────────

    #[test]
    fn test_material_query_contains_select() {
        let q = PhysicsQueryBuilder::material_query("http://example.org/mat#Steel");
        assert!(q.contains("SELECT"), "Query must start with SELECT");
        assert!(q.contains("youngsModulus"));
        assert!(q.contains("poissonsRatio"));
    }

    #[test]
    fn test_material_query_contains_iri() {
        let iri = "http://example.org/mat#Aluminium";
        let q = PhysicsQueryBuilder::material_query(iri);
        assert!(q.contains(iri));
    }

    #[test]
    fn test_structure_query_contains_select() {
        let q = PhysicsQueryBuilder::structure_query("http://example.org/struct#Bridge");
        assert!(q.contains("SELECT"));
        assert!(q.contains("nodeId"));
        assert!(q.contains("?x"));
        assert!(q.contains("?y"));
    }

    #[test]
    fn test_simulation_result_query_contains_select() {
        let q = PhysicsQueryBuilder::simulation_result_query("urn:sim:001");
        assert!(q.contains("SELECT"));
        assert!(q.contains("value"));
        assert!(q.contains("property"));
    }

    #[test]
    fn test_iri_local_name_fragment() {
        assert_eq!(iri_local_name("http://example.org/ns#Steel"), "Steel");
    }

    #[test]
    fn test_iri_local_name_path() {
        assert_eq!(iri_local_name("http://example.org/ns/Steel"), "Steel");
    }

    #[test]
    fn test_infer_samm_integer() {
        let (dt, _) = infer_samm_property("42");
        assert_eq!(dt, SammDataType::Integer);
    }

    #[test]
    fn test_infer_samm_float() {
        let (dt, _) = infer_samm_property("3.14");
        assert_eq!(dt, SammDataType::Float);
    }

    #[test]
    fn test_infer_samm_boolean() {
        let (dt, _) = infer_samm_property("true");
        assert_eq!(dt, SammDataType::Boolean);
    }

    #[test]
    fn test_infer_samm_string() {
        let (dt, _) = infer_samm_property("hello world");
        assert_eq!(dt, SammDataType::String);
    }
}
