//! Mapping logic: physics objects → RDF triples, ontology bindings, and
//! reverse parsing from RDF triples back to physics parameters.

use crate::error::{PhysicsError, PhysicsResult};
use crate::rdf::physics_rdf_types::{
    PhysicsToRdfConfig, RdfBoundaryCondition, RdfMaterialProperty, Triple, NS_EX, NS_PHYS, NS_PROV,
    NS_QUDT, NS_RDF, NS_RDFS, NS_SOSA, NS_SSN, NS_UNIT, NS_XSD,
};
use crate::simulation::result_injection::SimulationResult;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

// ──────────────────────────────────────────────────────────────────────────────
// Unit mapping: quantity name → QUDT unit suffix
// ──────────────────────────────────────────────────────────────────────────────

pub(crate) fn qudt_unit_for(property: &str) -> &'static str {
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
pub(crate) fn turtle_preamble() -> String {
    use crate::rdf::physics_rdf_types::{NS_RDF as RDF, NS_RDFS as RDFS};
    format!(
        "@prefix rdf:  <{RDF}> .\n\
         @prefix rdfs: <{RDFS}> .\n\
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

    /// Build from a `PhysicsToRdfConfig`.
    pub fn from_config(cfg: PhysicsToRdfConfig) -> Self {
        Self {
            base_iri: cfg.base_iri,
            include_provenance: cfg.include_provenance,
            include_digital_twin: cfg.include_digital_twin,
            include_units: cfg.include_units,
        }
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
// Helper functions (shared with physics_rdf_serializer)
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
pub(crate) fn group_by_subject(triples: &[Triple]) -> HashMap<String, Vec<Triple>> {
    let mut map: HashMap<String, Vec<Triple>> = HashMap::new();
    for t in triples {
        map.entry(t.subject.clone()).or_default().push(t.clone());
    }
    map
}

/// True if the triple has `rdf:type` as predicate.
pub(crate) fn is_rdf_type(t: &Triple) -> bool {
    t.predicate == format!("<{}type>", NS_RDF)
        || t.predicate == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
        || t.predicate == "a"
}

/// Find the string value of the first triple with matching predicate.
pub(crate) fn find_object_str(props: &[Triple], predicate: &str) -> Option<String> {
    props
        .iter()
        .find(|t| t.predicate == predicate)
        .map(|t| strip_literal_quotes(&t.object))
}

/// Find the double value of the first triple with matching predicate.
pub(crate) fn find_object_double(props: &[Triple], predicate: &str) -> Option<f64> {
    props
        .iter()
        .find(|t| t.predicate == predicate)
        .and_then(|t| extract_double_literal(&t.object))
}

/// Strip Turtle literal quotes and datatype suffix: `"1.5"^^xsd:double` → `1.5`.
pub(crate) fn strip_literal_quotes(s: &str) -> String {
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
