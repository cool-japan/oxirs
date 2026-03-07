//! Physics-RDF Roundtrip Conversion
//!
//! Provides three core types:
//!
//! - [`PhysicsToRdf`]: converts physics simulation results to RDF triples
//!   using SOSA/SSN and QUDT ontologies.
//! - [`RdfToPhysics`]: parses a slice of [`Triple`] values (e.g. produced by a
//!   previous `PhysicsToRdf::convert` call) and extracts boundary conditions
//!   and material properties, enabling a full roundtrip without an embedded
//!   store dependency.
//! - [`SparqlPhysicsQuery`]: executes SPARQL-like queries over an in-memory
//!   triple index built from physics result triples.
//!
//! # Ontology namespaces used
//!
//! | Prefix | Namespace                                       |
//! |--------|-------------------------------------------------|
//! | sosa   | <http://www.w3.org/ns/sosa/>                    |
//! | ssn    | <http://www.w3.org/ns/ssn/>                     |
//! | qudt   | <http://qudt.org/schema/qudt/>                  |
//! | unit   | <http://qudt.org/vocab/unit/>                   |
//! | ex     | <http://oxirs.org/example/physics#>             |
//! | phys   | <http://oxirs.org/physics#>                     |
//! | prov   | <http://www.w3.org/ns/prov#>                    |
//! | xsd    | <http://www.w3.org/2001/XMLSchema#>             |

use crate::error::{PhysicsError, PhysicsResult};
use crate::simulation::result_injection::SimulationResult;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

// ──────────────────────────────────────────────────────────────────────────────
// Namespace constants
// ──────────────────────────────────────────────────────────────────────────────

pub const NS_SOSA: &str = "http://www.w3.org/ns/sosa/";
pub const NS_SSN: &str = "http://www.w3.org/ns/ssn/";
pub const NS_QUDT: &str = "http://qudt.org/schema/qudt/";
pub const NS_UNIT: &str = "http://qudt.org/vocab/unit/";
pub const NS_EX: &str = "http://oxirs.org/example/physics#";
pub const NS_PHYS: &str = "http://oxirs.org/physics#";
pub const NS_PROV: &str = "http://www.w3.org/ns/prov#";
pub const NS_XSD: &str = "http://www.w3.org/2001/XMLSchema#";
pub const NS_RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
pub const NS_RDFS: &str = "http://www.w3.org/2000/01/rdf-schema#";

// ──────────────────────────────────────────────────────────────────────────────
// Unit mapping: quantity name → QUDT unit suffix
// ──────────────────────────────────────────────────────────────────────────────

fn qudt_unit_for(property: &str) -> &'static str {
    match property {
        "temperature" => "DEG_C",
        "temperature_k" => "K",
        "pressure" => "PA",
        "velocity" | "velocity_x" | "velocity_y" | "velocity_z" => "M-PER-SEC",
        "mass" => "KiloGM",
        "energy" | "kinetic_energy" | "potential_energy" | "total_energy" => "J",
        "power" => "W",
        "force" | "force_x" | "force_y" | "force_z" => "N",
        "density" => "KiloGM-PER-M3",
        "viscosity" => "PA-SEC",
        "thermal_conductivity" => "W-PER-M-K",
        "specific_heat" => "J-PER-KiloGM-K",
        "length" | "position_x" | "position_y" | "position_z" => "M",
        "time" => "SEC",
        "frequency" => "HZ",
        "voltage" => "V",
        "current" => "A",
        "resistance" => "OHM",
        "entropy" => "J-PER-K",
        _ => "UNITLESS",
    }
}

/// Turtle preamble with all prefixes we use.
fn turtle_preamble() -> String {
    format!(
        "@prefix rdf:  <{NS_RDF}> .\n\
         @prefix rdfs: <{NS_RDFS}> .\n\
         @prefix xsd:  <{NS_XSD}> .\n\
         @prefix sosa: <{NS_SOSA}> .\n\
         @prefix ssn:  <{NS_SSN}> .\n\
         @prefix qudt: <{NS_QUDT}> .\n\
         @prefix unit: <{NS_UNIT}> .\n\
         @prefix ex:   <{NS_EX}> .\n\
         @prefix phys: <{NS_PHYS}> .\n\
         @prefix prov: <{NS_PROV}> .\n\n"
    )
}

// ──────────────────────────────────────────────────────────────────────────────
// RDF triple representation (lightweight, no store dependency)
// ──────────────────────────────────────────────────────────────────────────────

/// A minimal N-Triple-style triple (subject, predicate, object as strings).
#[derive(Debug, Clone, PartialEq)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl Triple {
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

    /// Format as a Turtle triple statement (without trailing `.`).
    pub fn to_turtle_statement(&self) -> String {
        format!("{} {} {}", self.subject, self.predicate, self.object)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Parsed boundary condition (physics-level, no oxirs-core dependency here)
// ──────────────────────────────────────────────────────────────────────────────

/// A boundary condition extracted from RDF.
#[derive(Debug, Clone, PartialEq)]
pub struct RdfBoundaryCondition {
    /// Subject IRI of the boundary condition node.
    pub iri: String,
    /// Condition type (e.g. `"inlet"`, `"wall"`, `"outlet"`).
    pub condition_type: String,
    /// Name of the physical property.
    pub property: String,
    /// Numeric value in SI units.
    pub value: f64,
    /// QUDT unit suffix.
    pub unit: String,
}

/// A material property extracted from RDF.
#[derive(Debug, Clone, PartialEq)]
pub struct RdfMaterialProperty {
    /// Subject IRI of the material node.
    pub iri: String,
    /// Material or property name.
    pub name: String,
    /// Numeric value.
    pub value: f64,
    /// QUDT unit suffix.
    pub unit: String,
    /// Optional description.
    pub description: Option<String>,
}

// ──────────────────────────────────────────────────────────────────────────────
// PhysicsToRdf
// ──────────────────────────────────────────────────────────────────────────────

/// Converts physics simulation results into RDF triples following SOSA/SSN
/// and QUDT ontologies.
///
/// # SOSA modelling
///
/// Each scalar quantity in a [`SimulationResult`] is mapped to a
/// `sosa:Observation`:
///
/// ```turtle
/// ex:obs_<run>_<prop>_<t>  a sosa:Observation ;
///     sosa:observedProperty  phys:<prop> ;
///     sosa:hasSimpleResult   "<value>"^^xsd:double ;
///     sosa:resultTime        "<ts>"^^xsd:dateTime ;
///     sosa:hasFeatureOfInterest ex:dt_<entity> .
/// ```
///
/// The digital twin entity is represented as:
///
/// ```turtle
/// ex:dt_<entity>  a ex:DigitalTwin ;
///     ex:hasState  ex:state_<run>_<t> .
/// ```
pub struct PhysicsToRdf {
    /// Base IRI for generated entities.
    pub base_iri: String,
    /// Include provenance triples (W3C PROV).
    pub include_provenance: bool,
    /// Include digital twin state triples.
    pub include_digital_twin: bool,
    /// Include QUDT unit annotations.
    pub include_units: bool,
}

impl Default for PhysicsToRdf {
    fn default() -> Self {
        Self {
            base_iri: NS_EX.to_string(),
            include_provenance: true,
            include_digital_twin: true,
            include_units: true,
        }
    }
}

impl PhysicsToRdf {
    /// Create a converter with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert a simulation result to a list of RDF triples.
    pub fn convert(&self, result: &SimulationResult) -> Vec<Triple> {
        let mut triples = Vec::new();

        let run_id = sanitize_iri_fragment(&result.simulation_run_id);
        let entity_frag = sanitize_iri_fragment(&result.entity_iri);
        let ts_lit = format!(
            "\"{}\"^^<{}dateTime>",
            result.timestamp.format("%Y-%m-%dT%H:%M:%SZ"),
            NS_XSD
        );

        // ── Digital twin entity ──────────────────────────────────────────────
        if self.include_digital_twin {
            let dt_iri = format!("<{}dt_{}>", self.base_iri, entity_frag);
            triples.push(Triple::new(
                dt_iri.clone(),
                format!("<{}type>", NS_RDF),
                format!("<{}DigitalTwin>", NS_EX),
            ));
            triples.push(Triple::new(
                dt_iri.clone(),
                format!("<{}label>", NS_RDFS),
                format!("\"Digital Twin of {}\"", result.entity_iri),
            ));
        }

        // ── Observations from state trajectory ───────────────────────────────
        for (t_idx, sv) in result.state_trajectory.iter().enumerate() {
            let state_iri = format!("<{}state_{}_t{}>", self.base_iri, run_id, t_idx);

            // Digital twin → state link
            if self.include_digital_twin {
                let dt_iri = format!("<{}dt_{}>", self.base_iri, entity_frag);
                triples.push(Triple::new(
                    dt_iri,
                    format!("<{}hasState>", NS_EX),
                    state_iri.clone(),
                ));
                triples.push(Triple::new(
                    state_iri.clone(),
                    format!("<{}type>", NS_RDF),
                    format!("<{}SimulationState>", NS_EX),
                ));
                triples.push(Triple::new(
                    state_iri.clone(),
                    format!("<{}simulationRunId>", NS_EX),
                    format!("\"{}\"^^<{}string>", result.simulation_run_id, NS_XSD),
                ));
                triples.push(Triple::new(
                    state_iri.clone(),
                    format!("<{}timestamp>", NS_EX),
                    ts_lit.clone(),
                ));
                triples.push(Triple::new(
                    state_iri.clone(),
                    format!("<{}simTime>", NS_PHYS),
                    format!("\"{}\"^^<{}double>", sv.time, NS_XSD),
                ));
            }

            // One SOSA Observation per measured quantity
            for (prop, &val) in &sv.state {
                let prop_frag = sanitize_iri_fragment(prop);
                let obs_iri = format!("<{}obs_{}_{}_{}>", self.base_iri, run_id, prop_frag, t_idx);
                let prop_iri = format!("<{}{}>", NS_PHYS, prop_frag);
                let feature_iri = format!("<{}dt_{}>", self.base_iri, entity_frag);

                // Observation type
                triples.push(Triple::new(
                    obs_iri.clone(),
                    format!("<{}type>", NS_RDF),
                    format!("<{}Observation>", NS_SOSA),
                ));
                // Observed property
                triples.push(Triple::new(
                    obs_iri.clone(),
                    format!("<{}observedProperty>", NS_SOSA),
                    prop_iri.clone(),
                ));
                // Simple result
                triples.push(Triple::new(
                    obs_iri.clone(),
                    format!("<{}hasSimpleResult>", NS_SOSA),
                    format!("\"{}\"^^<{}double>", val, NS_XSD),
                ));
                // Result time
                triples.push(Triple::new(
                    obs_iri.clone(),
                    format!("<{}resultTime>", NS_SOSA),
                    ts_lit.clone(),
                ));
                // Feature of interest
                triples.push(Triple::new(
                    obs_iri.clone(),
                    format!("<{}hasFeatureOfInterest>", NS_SOSA),
                    feature_iri,
                ));

                // Observable property type declaration
                triples.push(Triple::new(
                    prop_iri.clone(),
                    format!("<{}type>", NS_RDF),
                    format!("<{}ObservableProperty>", NS_SOSA),
                ));
                triples.push(Triple::new(
                    prop_iri.clone(),
                    format!("<{}label>", NS_RDFS),
                    format!("\"{}\"", prop),
                ));

                // QUDT unit annotation
                if self.include_units {
                    let qudt_unit = qudt_unit_for(prop);
                    triples.push(Triple::new(
                        obs_iri.clone(),
                        format!("<{}unit>", NS_QUDT),
                        format!("<{}{}>", NS_UNIT, qudt_unit),
                    ));
                    triples.push(Triple::new(
                        obs_iri.clone(),
                        format!("<{}numericValue>", NS_QUDT),
                        format!("\"{}\"^^<{}double>", val, NS_XSD),
                    ));
                }

                // Link observation to state
                if self.include_digital_twin {
                    triples.push(Triple::new(
                        state_iri.clone(),
                        format!("<{}hasObservation>", NS_SSN),
                        obs_iri,
                    ));
                }
            }
        }

        // ── Derived quantities ────────────────────────────────────────────────
        for (prop, &val) in &result.derived_quantities {
            let prop_frag = sanitize_iri_fragment(prop);
            let obs_iri = format!("<{}derived_{}_{}>", self.base_iri, run_id, prop_frag);
            triples.push(Triple::new(
                obs_iri.clone(),
                format!("<{}type>", NS_RDF),
                format!("<{}Observation>", NS_SOSA),
            ));
            triples.push(Triple::new(
                obs_iri.clone(),
                format!("<{}observedProperty>", NS_SOSA),
                format!("<{}{}>", NS_PHYS, prop_frag),
            ));
            triples.push(Triple::new(
                obs_iri.clone(),
                format!("<{}hasSimpleResult>", NS_SOSA),
                format!("\"{}\"^^<{}double>", val, NS_XSD),
            ));
        }

        // ── Provenance ────────────────────────────────────────────────────────
        if self.include_provenance {
            let activity_iri = format!("<{}activity_{}>", self.base_iri, run_id);
            triples.push(Triple::new(
                activity_iri.clone(),
                format!("<{}type>", NS_RDF),
                format!("<{}Activity>", NS_PROV),
            ));
            triples.push(Triple::new(
                activity_iri.clone(),
                format!("<{}startedAtTime>", NS_PROV),
                ts_lit.clone(),
            ));
            triples.push(Triple::new(
                activity_iri.clone(),
                format!("<{}wasAssociatedWith>", NS_PROV),
                format!(
                    "<{}software/{}>",
                    NS_PHYS,
                    sanitize_iri_fragment(&result.provenance.software)
                ),
            ));
            triples.push(Triple::new(
                activity_iri.clone(),
                format!("<{}converged>", NS_PHYS),
                format!(
                    "\"{}\"^^<{}boolean>",
                    result.convergence_info.converged, NS_XSD
                ),
            ));
        }

        triples
    }

    /// Serialise to Turtle format (uses prefix preamble + one triple per line).
    pub fn to_turtle(&self, result: &SimulationResult) -> String {
        let triples = self.convert(result);
        let mut out = turtle_preamble();
        for t in &triples {
            let _ = writeln!(out, "{} .\n", t.to_turtle_statement());
        }
        out
    }

    /// Return triples grouped by subject IRI.
    pub fn to_subject_map(&self, result: &SimulationResult) -> HashMap<String, Vec<Triple>> {
        let mut map: HashMap<String, Vec<Triple>> = HashMap::new();
        for t in self.convert(result) {
            map.entry(t.subject.clone()).or_default().push(t);
        }
        map
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// RdfToPhysics
// ──────────────────────────────────────────────────────────────────────────────

/// Parses a slice of [`Triple`] values to extract physics simulation
/// parameters (boundary conditions, material properties).
///
/// In a production context this would drive `oxirs-core`'s Turtle parser and
/// SPARQL engine.  The current implementation works directly from
/// `Vec<Triple>` to enable zero-copy roundtrips.
///
/// # Supported triple patterns
///
/// **Boundary conditions**:
/// ```text
/// <ex:bc_inlet>  rdf:type     <phys:BoundaryCondition> ;
///                phys:bcType      "inlet" ;
///                phys:bcProperty  "velocity" ;
///                phys:bcValue     "1.5"^^xsd:double ;
///                phys:bcUnit      "M-PER-SEC" .
/// ```
///
/// **Material properties**:
/// ```text
/// <ex:material_steel>  rdf:type    <phys:Material> ;
///                      rdfs:label  "Steel" ;
///                      phys:value  "50.2"^^xsd:double ;
///                      phys:unit   "W-PER-M-K" .
/// ```
pub struct RdfToPhysics {
    /// Physics namespace to look for.
    pub phys_ns: String,
    /// Ignore absence of boundary conditions (best-effort extraction).
    pub lenient: bool,
}

impl Default for RdfToPhysics {
    fn default() -> Self {
        Self {
            phys_ns: NS_PHYS.to_string(),
            lenient: true,
        }
    }
}

impl RdfToPhysics {
    /// Create a parser with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse boundary conditions from a triple list.
    pub fn extract_boundary_conditions(
        &self,
        triples: &[Triple],
    ) -> PhysicsResult<Vec<RdfBoundaryCondition>> {
        let by_subject = group_by_subject(triples);
        let mut bcs = Vec::new();

        let bc_type_iri = format!("<{}BoundaryCondition>", NS_EX);
        let bc_type_iri2 = format!("<{}BoundaryCondition>", self.phys_ns);

        for (subj, props) in &by_subject {
            let is_bc = props
                .iter()
                .any(|t| is_rdf_type(t) && (t.object == bc_type_iri || t.object == bc_type_iri2));
            if !is_bc {
                continue;
            }

            let condition_type = find_object_str(props, &format!("<{}bcType>", self.phys_ns))
                .or_else(|| find_object_str(props, &format!("<{}conditionType>", self.phys_ns)))
                .unwrap_or_else(|| "unspecified".to_string());

            let property = find_object_str(props, &format!("<{}bcProperty>", self.phys_ns))
                .or_else(|| find_object_str(props, &format!("<{}observedProperty>", NS_SOSA)))
                .unwrap_or_else(|| "unknown".to_string());

            let value =
                find_object_double(props, &format!("<{}bcValue>", self.phys_ns)).unwrap_or(0.0);

            let unit = find_object_str(props, &format!("<{}bcUnit>", self.phys_ns))
                .unwrap_or_else(|| "UNITLESS".to_string());

            bcs.push(RdfBoundaryCondition {
                iri: subj.clone(),
                condition_type,
                property,
                value,
                unit,
            });
        }

        if bcs.is_empty() && !self.lenient {
            return Err(PhysicsError::ParameterExtraction(
                "no boundary conditions found in triples".to_string(),
            ));
        }
        Ok(bcs)
    }

    /// Extract material properties from the triple list.
    pub fn extract_material_properties(
        &self,
        triples: &[Triple],
    ) -> PhysicsResult<Vec<RdfMaterialProperty>> {
        let by_subject = group_by_subject(triples);
        let mut mats = Vec::new();

        let mat_type_iri = format!("<{}Material>", NS_EX);
        let mat_type_iri2 = format!("<{}Material>", self.phys_ns);

        for (subj, props) in &by_subject {
            let is_mat = props
                .iter()
                .any(|t| is_rdf_type(t) && (t.object == mat_type_iri || t.object == mat_type_iri2));
            if !is_mat {
                continue;
            }

            let name =
                find_object_str(props, &format!("<{}label>", NS_RDFS)).unwrap_or_else(|| {
                    // Fallback: derive from subject IRI
                    let stripped = subj
                        .strip_prefix('<')
                        .and_then(|s| s.strip_suffix('>'))
                        .unwrap_or(subj.as_str());
                    stripped
                        .rsplit(['/', '#'])
                        .next()
                        .unwrap_or("unknown")
                        .to_string()
                });

            let value =
                find_object_double(props, &format!("<{}value>", self.phys_ns)).unwrap_or(0.0);

            let unit = find_object_str(props, &format!("<{}unit>", self.phys_ns))
                .unwrap_or_else(|| "UNITLESS".to_string());

            let description = find_object_str(props, &format!("<{}description>", NS_RDFS));

            mats.push(RdfMaterialProperty {
                iri: subj.clone(),
                name,
                value,
                unit,
                description,
            });
        }
        Ok(mats)
    }

    /// Extract observations (property IRI → value) from a triple list.
    ///
    /// Scans for `sosa:Observation` triples and returns `(property, value)` pairs.
    pub fn extract_observations(&self, triples: &[Triple]) -> Vec<(String, f64)> {
        let by_subject = group_by_subject(triples);
        let obs_type = format!("<{}Observation>", NS_SOSA);
        let rdf_type_pred = format!("<{}type>", NS_RDF);
        let observed_prop_pred = format!("<{}observedProperty>", NS_SOSA);
        let simple_result_pred = format!("<{}hasSimpleResult>", NS_SOSA);

        let mut results = Vec::new();

        for props in by_subject.values() {
            let is_obs = props
                .iter()
                .any(|t| t.predicate == rdf_type_pred && t.object == obs_type);
            if !is_obs {
                continue;
            }
            let prop = find_object_str(props, &observed_prop_pred).unwrap_or_default();
            let value = find_object_double(props, &simple_result_pred);
            if let Some(v) = value {
                results.push((prop, v));
            }
        }
        results
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SparqlPhysicsQuery
// ──────────────────────────────────────────────────────────────────────────────

/// Executes SPARQL-like queries over an in-memory triple index built from
/// physics simulation result triples.
pub struct SparqlPhysicsQuery {
    /// Indexed triple store: subject → list of (predicate, object).
    store: HashMap<String, Vec<(String, String)>>,
    /// All triples (for full-scan queries).
    triples: Vec<Triple>,
}

impl SparqlPhysicsQuery {
    /// Build the query index from a slice of triples.
    pub fn new(triples: &[Triple]) -> Self {
        let mut store: HashMap<String, Vec<(String, String)>> = HashMap::new();
        for t in triples {
            store
                .entry(t.subject.clone())
                .or_default()
                .push((t.predicate.clone(), t.object.clone()));
        }
        Self {
            store,
            triples: triples.to_vec(),
        }
    }

    /// Build from a simulation result using [`PhysicsToRdf`].
    pub fn from_result(result: &SimulationResult) -> Self {
        let converter = PhysicsToRdf::new();
        let triples = converter.convert(result);
        Self::new(&triples)
    }

    /// Return the maximum `sosa:hasSimpleResult` for "temperature" observations.
    pub fn get_max_temperature(&self) -> Option<f64> {
        self.get_max_for_property("temperature")
    }

    /// Return the maximum `sosa:hasSimpleResult` for observations of the named property.
    pub fn get_max_for_property(&self, property: &str) -> Option<f64> {
        let prop_iri = format!("<{}{}>", NS_PHYS, sanitize_iri_fragment(property));
        let observed_prop_pred = format!("<{}observedProperty>", NS_SOSA);
        let simple_result_pred = format!("<{}hasSimpleResult>", NS_SOSA);

        let obs_subjects: Vec<&String> = self
            .triples
            .iter()
            .filter(|t| t.predicate == observed_prop_pred && t.object == prop_iri)
            .map(|t| &t.subject)
            .collect();

        let mut max_val: Option<f64> = None;
        for subj in obs_subjects {
            if let Some(preds) = self.store.get(subj) {
                for (pred, obj) in preds {
                    if pred == &simple_result_pred {
                        if let Some(v) = extract_double_literal(obj) {
                            max_val = Some(match max_val {
                                None => v,
                                Some(cur) => cur.max(v),
                            });
                        }
                    }
                }
            }
        }
        max_val
    }

    /// Return all observations within a simulation time range `[t_start, t_end]`.
    ///
    /// Returns a list of `(property_iri, value)` pairs.
    pub fn get_observations_in_range(&self, t_start: f64, t_end: f64) -> Vec<(String, f64)> {
        let sim_time_pred = format!("<{}simTime>", NS_PHYS);
        let has_obs_pred = format!("<{}hasObservation>", NS_SSN);
        let simple_result_pred = format!("<{}hasSimpleResult>", NS_SOSA);
        let observed_prop_pred = format!("<{}observedProperty>", NS_SOSA);

        // Find state IRIs whose simTime is in [t_start, t_end]
        let in_range_states: Vec<String> = self
            .triples
            .iter()
            .filter(|t| t.predicate == sim_time_pred)
            .filter_map(|t| {
                extract_double_literal(&t.object).and_then(|sim_t| {
                    if sim_t >= t_start && sim_t <= t_end {
                        Some(t.subject.clone())
                    } else {
                        None
                    }
                })
            })
            .collect();

        let mut results = Vec::new();

        for state_subj in &in_range_states {
            if let Some(preds) = self.store.get(state_subj) {
                let obs_iris: Vec<String> = preds
                    .iter()
                    .filter(|(p, _)| p == &has_obs_pred)
                    .map(|(_, o)| o.clone())
                    .collect();

                for obs_iri in obs_iris {
                    if let Some(obs_preds) = self.store.get(&obs_iri) {
                        let prop = obs_preds
                            .iter()
                            .find(|(p, _)| p == &observed_prop_pred)
                            .map(|(_, o)| o.clone())
                            .unwrap_or_default();
                        let value = obs_preds
                            .iter()
                            .find(|(p, _)| p == &simple_result_pred)
                            .and_then(|(_, o)| extract_double_literal(o));

                        if let Some(v) = value {
                            results.push((prop, v));
                        }
                    }
                }
            }
        }
        results
    }

    /// Return the minimum `sosa:hasSimpleResult` for the named property.
    pub fn get_min_for_property(&self, property: &str) -> Option<f64> {
        let prop_iri = format!("<{}{}>", NS_PHYS, sanitize_iri_fragment(property));
        let observed_prop_pred = format!("<{}observedProperty>", NS_SOSA);
        let simple_result_pred = format!("<{}hasSimpleResult>", NS_SOSA);

        let obs_subjects: Vec<&String> = self
            .triples
            .iter()
            .filter(|t| t.predicate == observed_prop_pred && t.object == prop_iri)
            .map(|t| &t.subject)
            .collect();

        let mut min_val: Option<f64> = None;
        for subj in obs_subjects {
            if let Some(preds) = self.store.get(subj) {
                for (pred, obj) in preds {
                    if pred == &simple_result_pred {
                        if let Some(v) = extract_double_literal(obj) {
                            min_val = Some(match min_val {
                                None => v,
                                Some(cur) => cur.min(v),
                            });
                        }
                    }
                }
            }
        }
        min_val
    }

    /// Return the mean `sosa:hasSimpleResult` for the named property.
    pub fn get_mean_for_property(&self, property: &str) -> Option<f64> {
        let prop_iri = format!("<{}{}>", NS_PHYS, sanitize_iri_fragment(property));
        let observed_prop_pred = format!("<{}observedProperty>", NS_SOSA);
        let simple_result_pred = format!("<{}hasSimpleResult>", NS_SOSA);

        let obs_subjects: Vec<&String> = self
            .triples
            .iter()
            .filter(|t| t.predicate == observed_prop_pred && t.object == prop_iri)
            .map(|t| &t.subject)
            .collect();

        let mut sum = 0.0_f64;
        let mut count = 0usize;
        for subj in obs_subjects {
            if let Some(preds) = self.store.get(subj) {
                for (pred, obj) in preds {
                    if pred == &simple_result_pred {
                        if let Some(v) = extract_double_literal(obj) {
                            sum += v;
                            count += 1;
                        }
                    }
                }
            }
        }
        if count == 0 {
            None
        } else {
            Some(sum / count as f64)
        }
    }

    /// Count total `sosa:Observation` triples in the store.
    pub fn count_observations(&self) -> usize {
        let obs_type = format!("<{}Observation>", NS_SOSA);
        let rdf_type = format!("<{}type>", NS_RDF);
        self.triples
            .iter()
            .filter(|t| t.predicate == rdf_type && t.object == obs_type)
            .count()
    }

    /// Return all distinct `sosa:observedProperty` IRIs.
    pub fn list_observed_properties(&self) -> Vec<String> {
        let observed_prop_pred = format!("<{}observedProperty>", NS_SOSA);
        let mut seen = std::collections::HashSet::new();
        self.triples
            .iter()
            .filter(|t| t.predicate == observed_prop_pred)
            .map(|t| t.object.trim().to_string())
            .filter(|o| seen.insert(o.clone()))
            .collect()
    }

    /// Return all `(property_iri, value)` pairs for a given property.
    pub fn all_values_for_property(&self, property: &str) -> Vec<f64> {
        let prop_iri = format!("<{}{}>", NS_PHYS, sanitize_iri_fragment(property));
        let observed_prop_pred = format!("<{}observedProperty>", NS_SOSA);
        let simple_result_pred = format!("<{}hasSimpleResult>", NS_SOSA);

        let obs_subjects: Vec<&String> = self
            .triples
            .iter()
            .filter(|t| t.predicate == observed_prop_pred && t.object == prop_iri)
            .map(|t| &t.subject)
            .collect();

        let mut values = Vec::new();
        for subj in obs_subjects {
            if let Some(preds) = self.store.get(subj) {
                for (pred, obj) in preds {
                    if pred == &simple_result_pred {
                        if let Some(v) = extract_double_literal(obj) {
                            values.push(v);
                        }
                    }
                }
            }
        }
        values
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper functions
// ──────────────────────────────────────────────────────────────────────────────

/// Replace characters that are not safe in IRI path segments.
pub(crate) fn sanitize_iri_fragment(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => c,
            _ => '_',
        })
        .collect()
}

/// Group triples by subject.
fn group_by_subject(triples: &[Triple]) -> HashMap<String, Vec<Triple>> {
    let mut map: HashMap<String, Vec<Triple>> = HashMap::new();
    for t in triples {
        map.entry(t.subject.clone()).or_default().push(t.clone());
    }
    map
}

/// True if the triple has `rdf:type` as predicate.
fn is_rdf_type(t: &Triple) -> bool {
    t.predicate == format!("<{}type>", NS_RDF)
        || t.predicate == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
        || t.predicate == "a"
}

/// Find the string value of the first triple with matching predicate.
fn find_object_str(props: &[Triple], predicate: &str) -> Option<String> {
    props
        .iter()
        .find(|t| t.predicate == predicate)
        .map(|t| strip_literal_quotes(&t.object))
}

/// Find the double value of the first triple with matching predicate.
fn find_object_double(props: &[Triple], predicate: &str) -> Option<f64> {
    props
        .iter()
        .find(|t| t.predicate == predicate)
        .and_then(|t| extract_double_literal(&t.object))
}

/// Strip Turtle literal quotes and datatype suffix: `"1.5"^^xsd:double` → `1.5`.
fn strip_literal_quotes(s: &str) -> String {
    let trimmed = s.trim();
    let without_dt = if let Some(pos) = trimmed.rfind("^^") {
        &trimmed[..pos]
    } else {
        trimmed
    };
    without_dt.trim_matches('"').to_string()
}

/// Parse a double from a typed Turtle literal: `"1.5"^^xsd:double`.
pub(crate) fn extract_double_literal(s: &str) -> Option<f64> {
    strip_literal_quotes(s).trim().parse::<f64>().ok()
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::result_injection::{
        ConvergenceInfo, SimulationProvenance, SimulationResult, StateVector,
    };
    use chrono::Utc;
    use std::collections::HashMap;

    // ── Test helpers ──────────────────────────────────────────────────────────

    fn make_state(time: f64, temperature: f64, pressure: f64) -> StateVector {
        let mut state = HashMap::new();
        state.insert("temperature".to_string(), temperature);
        state.insert("pressure".to_string(), pressure);
        StateVector { time, state }
    }

    fn make_result() -> SimulationResult {
        let trajectory = vec![
            make_state(0.0, 300.0, 101325.0),
            make_state(1.0, 350.0, 101325.0),
            make_state(2.0, 400.0, 102000.0),
        ];
        let mut derived = HashMap::new();
        derived.insert("max_temperature".to_string(), 400.0);
        derived.insert("pressure_drop".to_string(), 675.0);

        SimulationResult {
            entity_iri: "urn:example:reactor:1".to_string(),
            simulation_run_id: "run-abc-123".to_string(),
            timestamp: Utc::now(),
            state_trajectory: trajectory,
            derived_quantities: derived,
            convergence_info: ConvergenceInfo {
                converged: true,
                iterations: 42,
                final_residual: 1e-8,
            },
            provenance: SimulationProvenance {
                software: "OxiRS-Physics".to_string(),
                version: "0.3.0".to_string(),
                parameters_hash: "abc123".to_string(),
                executed_at: Utc::now(),
                execution_time_ms: 1500,
            },
        }
    }

    fn make_bc_triples() -> Vec<Triple> {
        let rdf_type = format!("<{}type>", NS_RDF);
        let bc_iri = "<http://oxirs.org/example/physics#bc_inlet>";
        vec![
            Triple::new(bc_iri, rdf_type, format!("<{}BoundaryCondition>", NS_EX)),
            Triple::new(bc_iri, format!("<{}bcType>", NS_PHYS), "\"inlet\""),
            Triple::new(bc_iri, format!("<{}bcProperty>", NS_PHYS), "\"velocity\""),
            Triple::new(
                bc_iri,
                format!("<{}bcValue>", NS_PHYS),
                format!("\"1.5\"^^<{}double>", NS_XSD),
            ),
            Triple::new(bc_iri, format!("<{}bcUnit>", NS_PHYS), "\"M-PER-SEC\""),
        ]
    }

    fn make_material_triples() -> Vec<Triple> {
        let rdf_type = format!("<{}type>", NS_RDF);
        let mat_iri = "<http://oxirs.org/example/physics#material_steel>";
        vec![
            Triple::new(mat_iri, rdf_type, format!("<{}Material>", NS_EX)),
            Triple::new(mat_iri, format!("<{}label>", NS_RDFS), "\"Steel\""),
            Triple::new(
                mat_iri,
                format!("<{}value>", NS_PHYS),
                format!("\"50.2\"^^<{}double>", NS_XSD),
            ),
            Triple::new(mat_iri, format!("<{}unit>", NS_PHYS), "\"W-PER-M-K\""),
        ]
    }

    // ── PhysicsToRdf tests ────────────────────────────────────────────────────

    #[test]
    fn test_convert_produces_triples() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let triples = converter.convert(&result);
        assert!(!triples.is_empty(), "expected non-empty triples");
    }

    #[test]
    fn test_convert_contains_sosa_observations() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let triples = converter.convert(&result);
        let obs_type = format!("<{}Observation>", NS_SOSA);
        let rdf_type_pred = format!("<{}type>", NS_RDF);
        let obs_count = triples
            .iter()
            .filter(|t| t.predicate == rdf_type_pred && t.object == obs_type)
            .count();
        // 3 timesteps × 2 properties + 2 derived = ≥8 observations
        assert!(
            obs_count >= 8,
            "expected ≥8 SOSA observations, got {obs_count}"
        );
    }

    #[test]
    fn test_convert_contains_digital_twin() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let triples = converter.convert(&result);
        let dt_type = format!("<{}DigitalTwin>", NS_EX);
        let rdf_type_pred = format!("<{}type>", NS_RDF);
        let dt_count = triples
            .iter()
            .filter(|t| t.predicate == rdf_type_pred && t.object == dt_type)
            .count();
        assert_eq!(dt_count, 1, "expected exactly one DigitalTwin type triple");
    }

    #[test]
    fn test_convert_contains_qudt_units() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let triples = converter.convert(&result);
        let qudt_unit_pred = format!("<{}unit>", NS_QUDT);
        let unit_count = triples
            .iter()
            .filter(|t| t.predicate == qudt_unit_pred)
            .count();
        assert!(unit_count > 0, "expected QUDT unit triples");
    }

    #[test]
    fn test_to_turtle_contains_prefixes() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let turtle = converter.to_turtle(&result);
        assert!(turtle.contains("@prefix sosa:"), "missing sosa prefix");
        assert!(turtle.contains("@prefix qudt:"), "missing qudt prefix");
        assert!(turtle.contains("@prefix prov:"), "missing prov prefix");
    }

    #[test]
    fn test_to_subject_map_groups_correctly() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let map = converter.to_subject_map(&result);
        assert!(!map.is_empty(), "expected non-empty subject map");
        for v in map.values() {
            assert!(
                !v.is_empty(),
                "subject group should have at least one triple"
            );
        }
    }

    #[test]
    fn test_convert_no_digital_twin() {
        let converter = PhysicsToRdf {
            include_digital_twin: false,
            ..PhysicsToRdf::new()
        };
        let result = make_result();
        let triples = converter.convert(&result);
        let dt_type = format!("<{}DigitalTwin>", NS_EX);
        let rdf_type_pred = format!("<{}type>", NS_RDF);
        let dt_count = triples
            .iter()
            .filter(|t| t.predicate == rdf_type_pred && t.object == dt_type)
            .count();
        assert_eq!(dt_count, 0);
    }

    #[test]
    fn test_convert_no_provenance() {
        let converter = PhysicsToRdf {
            include_provenance: false,
            ..PhysicsToRdf::new()
        };
        let result = make_result();
        let triples = converter.convert(&result);
        let activity_type = format!("<{}Activity>", NS_PROV);
        let rdf_type_pred = format!("<{}type>", NS_RDF);
        let prov_count = triples
            .iter()
            .filter(|t| t.predicate == rdf_type_pred && t.object == activity_type)
            .count();
        assert_eq!(prov_count, 0);
    }

    #[test]
    fn test_convert_contains_observed_property() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let triples = converter.convert(&result);
        let obs_prop_pred = format!("<{}observedProperty>", NS_SOSA);
        assert!(triples.iter().any(|t| t.predicate == obs_prop_pred));
    }

    #[test]
    fn test_convert_has_simple_result_values() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let triples = converter.convert(&result);
        let sr_pred = format!("<{}hasSimpleResult>", NS_SOSA);
        let values: Vec<f64> = triples
            .iter()
            .filter(|t| t.predicate == sr_pred)
            .filter_map(|t| extract_double_literal(&t.object))
            .collect();
        assert!(!values.is_empty(), "expected hasSimpleResult values");
        assert!(values.contains(&300.0), "expected temperature 300.0");
    }

    #[test]
    fn test_convert_contains_prov_activity() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let triples = converter.convert(&result);
        let activity_type = format!("<{}Activity>", NS_PROV);
        let rdf_type_pred = format!("<{}type>", NS_RDF);
        assert!(
            triples
                .iter()
                .any(|t| t.predicate == rdf_type_pred && t.object == activity_type),
            "expected prov:Activity triple"
        );
    }

    // ── RdfToPhysics tests ────────────────────────────────────────────────────

    #[test]
    fn test_extract_boundary_conditions_basic() {
        let parser = RdfToPhysics::new();
        let triples = make_bc_triples();
        let bcs = parser.extract_boundary_conditions(&triples).unwrap();
        assert_eq!(bcs.len(), 1);
        let bc = &bcs[0];
        assert_eq!(bc.condition_type, "inlet");
        assert_eq!(bc.property, "velocity");
        assert!((bc.value - 1.5).abs() < 1e-10);
        assert_eq!(bc.unit, "M-PER-SEC");
    }

    #[test]
    fn test_extract_material_properties() {
        let parser = RdfToPhysics::new();
        let triples = make_material_triples();
        let mats = parser.extract_material_properties(&triples).unwrap();
        assert_eq!(mats.len(), 1);
        let mat = &mats[0];
        assert_eq!(mat.name, "Steel");
        assert!((mat.value - 50.2).abs() < 1e-10);
        assert_eq!(mat.unit, "W-PER-M-K");
    }

    #[test]
    fn test_extract_no_bcs_lenient() {
        let parser = RdfToPhysics {
            lenient: true,
            ..RdfToPhysics::default()
        };
        let triples = vec![Triple::new("<ex:foo>", "<ex:bar>", "<ex:baz>")];
        let bcs = parser.extract_boundary_conditions(&triples).unwrap();
        assert!(bcs.is_empty());
    }

    #[test]
    fn test_extract_no_bcs_strict() {
        let parser = RdfToPhysics {
            lenient: false,
            ..RdfToPhysics::default()
        };
        let triples = vec![Triple::new("<ex:foo>", "<ex:bar>", "<ex:baz>")];
        assert!(parser.extract_boundary_conditions(&triples).is_err());
    }

    #[test]
    fn test_extract_observations_from_converted() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let triples = converter.convert(&result);
        let parser = RdfToPhysics::new();
        let obs = parser.extract_observations(&triples);
        assert!(
            !obs.is_empty(),
            "expected observations from converted result"
        );
    }

    // ── Roundtrip test ────────────────────────────────────────────────────────

    #[test]
    fn test_roundtrip_observations_queryable() {
        let converter = PhysicsToRdf::new();
        let result = make_result();
        let triples = converter.convert(&result);
        let query = SparqlPhysicsQuery::new(&triples);
        let obs_count = query.count_observations();
        // 3 timesteps × 2 props + 2 derived = 8
        assert!(obs_count >= 8, "expected ≥8 observations, got {obs_count}");
    }

    // ── SparqlPhysicsQuery tests ──────────────────────────────────────────────

    #[test]
    fn test_get_max_temperature() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let max_temp = query.get_max_temperature();
        assert!(max_temp.is_some());
        let max = max_temp.unwrap();
        assert!((max - 400.0).abs() < 1e-6, "expected 400.0, got {max}");
    }

    #[test]
    fn test_get_min_temperature() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let min_temp = query.get_min_for_property("temperature");
        assert!(min_temp.is_some());
        let min = min_temp.unwrap();
        assert!((min - 300.0).abs() < 1e-6, "expected 300.0, got {min}");
    }

    #[test]
    fn test_get_mean_for_property() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let mean = query.get_mean_for_property("temperature");
        assert!(mean.is_some());
        // Mean of 300, 350, 400 = 350
        let m = mean.unwrap();
        assert!((m - 350.0).abs() < 1e-6, "expected 350.0, got {m}");
    }

    #[test]
    fn test_get_observations_in_range() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let obs = query.get_observations_in_range(0.5, 1.5);
        // simTime=1.0 is in [0.5, 1.5]
        assert!(!obs.is_empty(), "expected observations in range");
    }

    #[test]
    fn test_get_observations_out_of_range() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let obs = query.get_observations_in_range(10.0, 20.0);
        assert!(
            obs.is_empty(),
            "expected no observations outside time range"
        );
    }

    #[test]
    fn test_list_observed_properties() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let props = query.list_observed_properties();
        assert!(!props.is_empty());
    }

    #[test]
    fn test_count_observations() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        assert!(query.count_observations() >= 8);
    }

    #[test]
    fn test_from_result_constructor() {
        let result = make_result();
        let query = SparqlPhysicsQuery::from_result(&result);
        assert!(query.count_observations() >= 8);
    }

    #[test]
    fn test_max_for_unknown_property_returns_none() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        assert!(query.get_max_for_property("nonexistent_xyz").is_none());
    }

    #[test]
    fn test_all_values_for_property() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let temps = query.all_values_for_property("temperature");
        assert_eq!(temps.len(), 3, "expected 3 temperature observations");
    }

    // ── Utility helper tests ──────────────────────────────────────────────────

    #[test]
    fn test_sanitize_iri_fragment() {
        assert_eq!(sanitize_iri_fragment("run-abc-123"), "run-abc-123");
        assert_eq!(sanitize_iri_fragment("urn:foo:bar"), "urn_foo_bar");
        assert_eq!(sanitize_iri_fragment("a b c"), "a_b_c");
    }

    #[test]
    fn test_extract_double_literal() {
        assert_eq!(extract_double_literal("\"1.5\"^^xsd:double"), Some(1.5));
        assert_eq!(
            extract_double_literal("\"300.0\"^^<http://www.w3.org/2001/XMLSchema#double>"),
            Some(300.0)
        );
        assert_eq!(extract_double_literal("\"notanumber\""), None);
    }

    #[test]
    fn test_strip_literal_quotes() {
        assert_eq!(strip_literal_quotes("\"hello\""), "hello");
        assert_eq!(strip_literal_quotes("\"1.5\"^^xsd:double"), "1.5");
    }

    #[test]
    fn test_triple_statement() {
        let t = Triple::new("<ex:s>", "<ex:p>", "<ex:o>");
        let stmt = t.to_turtle_statement();
        assert!(stmt.contains("<ex:s>"));
        assert!(stmt.contains("<ex:p>"));
        assert!(stmt.contains("<ex:o>"));
    }

    #[test]
    fn test_qudt_unit_for_known() {
        assert_eq!(qudt_unit_for("temperature"), "DEG_C");
        assert_eq!(qudt_unit_for("pressure"), "PA");
        assert_eq!(qudt_unit_for("mass"), "KiloGM");
    }

    #[test]
    fn test_qudt_unit_for_unknown() {
        assert_eq!(qudt_unit_for("some_made_up_quantity"), "UNITLESS");
    }

    // ── Additional PhysicsToRdf tests ─────────────────────────────────────────

    #[test]
    fn test_physics_to_rdf_no_provenance() {
        let conv = PhysicsToRdf {
            include_provenance: false,
            include_digital_twin: true,
            include_units: true,
            base_iri: NS_EX.to_string(),
        };
        let result = make_result();
        let triples = conv.convert(&result);
        let has_prov = triples.iter().any(|t| t.predicate.contains("prov#"));
        assert!(!has_prov, "expected no provenance triples");
    }

    #[test]
    fn test_physics_to_rdf_no_digital_twin() {
        let conv = PhysicsToRdf {
            include_provenance: true,
            include_digital_twin: false,
            include_units: true,
            base_iri: NS_EX.to_string(),
        };
        let result = make_result();
        let triples = conv.convert(&result);
        let has_dt = triples.iter().any(|t| t.object.contains("DigitalTwin"));
        assert!(!has_dt, "expected no DigitalTwin triples when disabled");
    }

    #[test]
    fn test_physics_to_rdf_no_units() {
        let conv = PhysicsToRdf {
            include_provenance: true,
            include_digital_twin: true,
            include_units: false,
            base_iri: NS_EX.to_string(),
        };
        let result = make_result();
        let triples = conv.convert(&result);
        let has_unit = triples
            .iter()
            .any(|t| t.predicate.contains("qudt.org/schema/qudt/unit"));
        assert!(!has_unit, "expected no qudt:unit triples when disabled");
    }

    #[test]
    fn test_to_turtle_contains_prefix() {
        let conv = PhysicsToRdf::new();
        let result = make_result();
        let turtle = conv.to_turtle(&result);
        assert!(
            turtle.contains("@prefix sosa:"),
            "turtle must have sosa prefix"
        );
        assert!(
            turtle.contains("@prefix qudt:"),
            "turtle must have qudt prefix"
        );
        assert!(
            turtle.contains("@prefix prov:"),
            "turtle must have prov prefix"
        );
    }

    #[test]
    fn test_to_subject_map_has_digital_twin() {
        let conv = PhysicsToRdf::new();
        let result = make_result();
        let map = conv.to_subject_map(&result);
        let has_dt_key = map.keys().any(|k| k.contains("dt_"));
        assert!(has_dt_key, "subject map must contain a dt_ key");
    }

    #[test]
    fn test_roundtrip_observation_count() {
        let conv = PhysicsToRdf::new();
        let result = make_result();
        let triples = conv.convert(&result);
        let obs_count = triples
            .iter()
            .filter(|t| t.object.contains("Observation>"))
            .count();
        // 3 time steps × 2 properties + 2 derived = 8 observations
        assert!(
            obs_count >= 8,
            "expected at least 8 observations, got {obs_count}"
        );
    }

    #[test]
    fn test_rdf_to_physics_extract_bc_strict_empty_error() {
        let parser = RdfToPhysics {
            phys_ns: NS_PHYS.to_string(),
            lenient: false,
        };
        let triples: Vec<Triple> = vec![];
        let result = parser.extract_boundary_conditions(&triples);
        assert!(result.is_err(), "strict mode must error on empty BC list");
    }

    #[test]
    fn test_rdf_to_physics_bc_type_and_value() {
        let parser = RdfToPhysics::new();
        let triples = make_bc_triples();
        let bcs = parser.extract_boundary_conditions(&triples).unwrap();
        assert_eq!(bcs.len(), 1);
        assert_eq!(bcs[0].condition_type, "inlet");
        assert_eq!(bcs[0].property, "velocity");
        assert!((bcs[0].value - 1.5).abs() < 1e-10);
        assert_eq!(bcs[0].unit, "M-PER-SEC");
    }

    #[test]
    fn test_rdf_to_physics_material_property_extraction() {
        let parser = RdfToPhysics::new();
        let triples = make_material_triples();
        let mats = parser.extract_material_properties(&triples).unwrap();
        assert!(!mats.is_empty(), "expected at least one material property");
        assert_eq!(mats[0].unit, "W-PER-M-K");
        assert!((mats[0].value - 50.2).abs() < 1e-6);
    }

    #[test]
    fn test_physics_to_rdf_roundtrip_extract_observations() {
        let conv = PhysicsToRdf::new();
        let result = make_result();
        let triples = conv.convert(&result);

        let parser = RdfToPhysics::new();
        let obs = parser.extract_observations(&triples);
        assert!(
            !obs.is_empty(),
            "should extract at least one observation from roundtrip"
        );
    }

    #[test]
    fn test_sparql_query_get_max_temperature() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let max = query.get_max_temperature();
        assert!(max.is_some());
        let v = max.unwrap();
        assert!((v - 400.0).abs() < 1e-6, "expected max temp 400.0, got {v}");
    }

    #[test]
    fn test_sparql_query_get_min_for_property() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let min = query.get_min_for_property("temperature");
        assert!(min.is_some());
        let v = min.unwrap();
        assert!((v - 300.0).abs() < 1e-6, "expected min temp 300.0, got {v}");
    }

    #[test]
    fn test_sparql_query_mean_for_property() {
        let query = SparqlPhysicsQuery::from_result(&make_result());
        let mean = query.get_mean_for_property("pressure");
        assert!(mean.is_some());
        // pressure values: 101325, 101325, 102000 → mean ≈ 101550
        let m = mean.unwrap();
        assert!(m > 101000.0 && m < 103000.0, "unexpected mean pressure {m}");
    }

    #[test]
    fn test_triple_eq() {
        let t1 = Triple::new("<ex:s>", "<ex:p>", "<ex:o>");
        let t2 = Triple::new("<ex:s>", "<ex:p>", "<ex:o>");
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_triple_clone() {
        let t = Triple::new("<ex:s>", "<ex:p>", "<ex:o>");
        let c = t.clone();
        assert_eq!(t, c);
    }

    #[test]
    fn test_namespace_constants_non_empty() {
        assert!(!NS_SOSA.is_empty());
        assert!(!NS_SSN.is_empty());
        assert!(!NS_QUDT.is_empty());
        assert!(!NS_UNIT.is_empty());
        assert!(!NS_EX.is_empty());
        assert!(!NS_PHYS.is_empty());
        assert!(!NS_PROV.is_empty());
        assert!(!NS_XSD.is_empty());
        assert!(!NS_RDF.is_empty());
        assert!(!NS_RDFS.is_empty());
    }

    #[test]
    fn test_rdf_bc_iri_preserved() {
        let parser = RdfToPhysics::new();
        let triples = make_bc_triples();
        let bcs = parser.extract_boundary_conditions(&triples).unwrap();
        assert!(!bcs[0].iri.is_empty(), "BC IRI should be non-empty");
        assert!(
            bcs[0].iri.contains("bc_inlet"),
            "IRI should contain bc_inlet"
        );
    }

    #[test]
    fn test_physics_to_rdf_empty_trajectory() {
        let conv = PhysicsToRdf::new();
        let result = SimulationResult {
            entity_iri: "urn:example:empty:1".to_string(),
            simulation_run_id: "run-empty".to_string(),
            timestamp: chrono::Utc::now(),
            state_trajectory: vec![],
            derived_quantities: HashMap::new(),
            convergence_info: ConvergenceInfo {
                converged: true,
                iterations: 0,
                final_residual: 0.0,
            },
            provenance: SimulationProvenance {
                software: "OxiRS".to_string(),
                version: "0.2.0".to_string(),
                parameters_hash: "0".to_string(),
                executed_at: chrono::Utc::now(),
                execution_time_ms: 0,
            },
        };
        let triples = conv.convert(&result);
        // Digital twin triple should still be generated
        let has_dt = triples.iter().any(|t| t.object.contains("DigitalTwin"));
        assert!(
            has_dt,
            "should produce DigitalTwin triple even with empty trajectory"
        );
    }

    #[test]
    fn test_physics_to_rdf_default_base_iri_is_ex() {
        let conv = PhysicsToRdf::default();
        assert_eq!(conv.base_iri, NS_EX);
    }

    #[test]
    fn test_to_turtle_newlines_per_triple() {
        let conv = PhysicsToRdf::new();
        let result = make_result();
        let turtle = conv.to_turtle(&result);
        // Each triple ends with " .\n"
        let triple_count = turtle.matches(" .").count();
        assert!(
            triple_count >= 10,
            "expected at least 10 triples in turtle, got {triple_count}"
        );
    }

    #[test]
    fn test_qudt_unit_velocity() {
        assert_eq!(qudt_unit_for("velocity"), "M-PER-SEC");
        assert_eq!(qudt_unit_for("velocity_x"), "M-PER-SEC");
        assert_eq!(qudt_unit_for("velocity_y"), "M-PER-SEC");
    }

    #[test]
    fn test_qudt_unit_various() {
        assert_eq!(qudt_unit_for("density"), "KiloGM-PER-M3");
        assert_eq!(qudt_unit_for("voltage"), "V");
        assert_eq!(qudt_unit_for("entropy"), "J-PER-K");
        assert_eq!(qudt_unit_for("frequency"), "HZ");
    }

    #[test]
    fn test_rdf_material_empty_ok() {
        let parser = RdfToPhysics::new();
        let triples: Vec<Triple> = vec![];
        let mats = parser.extract_material_properties(&triples).unwrap();
        assert!(mats.is_empty(), "empty triples -> empty material list");
    }
}
