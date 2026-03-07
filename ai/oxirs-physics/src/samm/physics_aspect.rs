//! SAMM Aspect Model and AAS Bridge for Physics Simulations
//!
//! Provides three core types:
//!
//! - [`PhysicsAspect`]: A typed SAMM Aspect Model representing a physics
//!   simulation run, with properties `simulationId`, `modelType`,
//!   `physicalDomain`, `parameters`, and `results`.
//!
//! - [`SammPhysicsMapper`]: Converts between [`PhysicsAspect`] and JSON
//!   representations compatible with SAMM tooling and the OxiRS physics
//!   simulation engine.
//!
//! - [`PhysicsAasSubmodel`]: Asset Administration Shell (AAS) submodel
//!   representation of physics simulation data, serialisable to the IEC 63278
//!   JSON format.
//!
//! # SAMM characteristics modelled
//!
//! - `SimulationStatus`: Pending | Running | Completed | Failed
//! - `PhysicalDomain`: Thermal | Fluid | Structural | Electromagnetic | Multiphysics

use crate::error::{PhysicsError, PhysicsResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Enumerations (SAMM Characteristic: Enumeration)
// ──────────────────────────────────────────────────────────────────────────────

/// SAMM `SimulationStatus` characteristic enumeration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum SimulationStatus {
    /// Simulation has been created but not yet started.
    Pending,
    /// Simulation is currently executing.
    Running,
    /// Simulation finished successfully.
    Completed,
    /// Simulation terminated with an error.
    Failed,
}

impl SimulationStatus {
    /// Parse from a case-insensitive string.
    pub fn parse(s: &str) -> PhysicsResult<Self> {
        match s.to_lowercase().as_str() {
            "pending" => Ok(Self::Pending),
            "running" => Ok(Self::Running),
            "completed" | "done" | "success" => Ok(Self::Completed),
            "failed" | "error" => Ok(Self::Failed),
            other => Err(PhysicsError::SammParsing(format!(
                "unknown SimulationStatus: {other}"
            ))),
        }
    }

    /// Return the canonical SAMM string value.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "Pending",
            Self::Running => "Running",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
        }
    }

    /// True when the simulation has reached a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed)
    }
}

impl std::fmt::Display for SimulationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// SAMM `PhysicalDomain` characteristic enumeration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum PhysicalDomain {
    /// Heat transfer, thermodynamics.
    Thermal,
    /// Computational fluid dynamics (CFD).
    Fluid,
    /// Structural mechanics, FEM.
    Structural,
    /// Maxwell equations, EM fields.
    Electromagnetic,
    /// Coupled multi-physics simulation.
    Multiphysics,
}

impl PhysicalDomain {
    /// Parse from a case-insensitive string.
    pub fn parse(s: &str) -> PhysicsResult<Self> {
        match s.to_lowercase().as_str() {
            "thermal" | "heat" => Ok(Self::Thermal),
            "fluid" | "cfd" => Ok(Self::Fluid),
            "structural" | "fem" | "mechanical" => Ok(Self::Structural),
            "electromagnetic" | "em" | "electro" => Ok(Self::Electromagnetic),
            "multiphysics" | "multi" | "coupled" => Ok(Self::Multiphysics),
            other => Err(PhysicsError::SammParsing(format!(
                "unknown PhysicalDomain: {other}"
            ))),
        }
    }

    /// Return the canonical SAMM string value.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Thermal => "Thermal",
            Self::Fluid => "Fluid",
            Self::Structural => "Structural",
            Self::Electromagnetic => "Electromagnetic",
            Self::Multiphysics => "Multiphysics",
        }
    }
}

impl std::fmt::Display for PhysicalDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PhysicsAspect – SAMM Aspect Model for Physics Simulation
// ──────────────────────────────────────────────────────────────────────────────

/// A single scalar simulation parameter with unit annotation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationParameter {
    /// Parameter name (e.g. `"inletVelocity"`).
    pub name: String,
    /// Numeric value in SI units.
    pub value: f64,
    /// QUDT unit suffix (e.g. `"M-PER-SEC"`).
    pub unit: String,
    /// Optional description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl SimulationParameter {
    /// Create a new simulation parameter.
    pub fn new(name: impl Into<String>, value: f64, unit: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value,
            unit: unit.into(),
            description: None,
        }
    }

    /// Create a parameter with a description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

/// A single scalar simulation result with unit annotation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationResultValue {
    /// Result name (e.g. `"maxTemperature"`).
    pub name: String,
    /// Numeric value.
    pub value: f64,
    /// QUDT unit suffix.
    pub unit: String,
    /// Source time step index (if applicable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_step: Option<usize>,
}

impl SimulationResultValue {
    /// Create a new result value.
    pub fn new(name: impl Into<String>, value: f64, unit: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value,
            unit: unit.into(),
            time_step: None,
        }
    }
}

/// SAMM Aspect Model for a physics simulation run.
///
/// Maps directly to a SAMM Aspect TTL with:
/// - `samm:Aspect` as class
/// - Properties: `simulationId`, `modelType`, `physicalDomain`,
///   `parameters`, `results`
/// - Characteristics: `SimulationStatus` (Enumeration), `PhysicalDomain`
///   (Enumeration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsAspect {
    /// Unique identifier for this simulation run (SAMM property `simulationId`).
    pub simulation_id: String,

    /// Human-readable model name (SAMM property `modelType`, e.g. `"ThermalFEM"`).
    pub model_type: String,

    /// Physical domain of the simulation (SAMM property `physicalDomain`).
    pub physical_domain: PhysicalDomain,

    /// Current simulation status (SAMM Characteristic `SimulationStatus`).
    pub status: SimulationStatus,

    /// Input parameters (SAMM property `parameters`, Characteristic `ParameterList`).
    pub parameters: Vec<SimulationParameter>,

    /// Output results (SAMM property `results`, Characteristic `ResultList`).
    pub results: Vec<SimulationResultValue>,

    /// Asset IRI this aspect is attached to (e.g. `"urn:example:motor:42"`).
    pub asset_iri: String,

    /// Optional convergence flag from the simulation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub converged: Option<bool>,

    /// Optional execution time in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_time_ms: Option<u64>,

    /// Arbitrary metadata key-value pairs.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl PhysicsAspect {
    /// Create a minimal physics aspect.
    pub fn new(
        simulation_id: impl Into<String>,
        model_type: impl Into<String>,
        physical_domain: PhysicalDomain,
        asset_iri: impl Into<String>,
    ) -> Self {
        Self {
            simulation_id: simulation_id.into(),
            model_type: model_type.into(),
            physical_domain,
            status: SimulationStatus::Pending,
            parameters: Vec::new(),
            results: Vec::new(),
            asset_iri: asset_iri.into(),
            converged: None,
            execution_time_ms: None,
            metadata: HashMap::new(),
        }
    }

    /// Add a simulation parameter.
    pub fn with_parameter(mut self, param: SimulationParameter) -> Self {
        self.parameters.push(param);
        self
    }

    /// Add a simulation result value.
    pub fn with_result(mut self, result: SimulationResultValue) -> Self {
        self.results.push(result);
        self
    }

    /// Transition to `Running` status.
    pub fn start(&mut self) -> PhysicsResult<()> {
        if self.status != SimulationStatus::Pending {
            return Err(PhysicsError::Simulation(format!(
                "cannot start simulation in status {}",
                self.status
            )));
        }
        self.status = SimulationStatus::Running;
        Ok(())
    }

    /// Transition to `Completed` status with convergence information.
    pub fn complete(&mut self, converged: bool, execution_time_ms: u64) -> PhysicsResult<()> {
        if self.status != SimulationStatus::Running {
            return Err(PhysicsError::Simulation(format!(
                "cannot complete simulation in status {}",
                self.status
            )));
        }
        self.status = SimulationStatus::Completed;
        self.converged = Some(converged);
        self.execution_time_ms = Some(execution_time_ms);
        Ok(())
    }

    /// Transition to `Failed` status.
    pub fn fail(&mut self, reason: impl Into<String>) -> PhysicsResult<()> {
        self.status = SimulationStatus::Failed;
        self.metadata
            .insert("failure_reason".to_string(), reason.into());
        Ok(())
    }

    /// Look up a parameter by name.
    pub fn parameter(&self, name: &str) -> Option<&SimulationParameter> {
        self.parameters.iter().find(|p| p.name == name)
    }

    /// Look up a result value by name.
    pub fn result_value(&self, name: &str) -> Option<&SimulationResultValue> {
        self.results.iter().find(|r| r.name == name)
    }

    /// Validate the aspect is ready to run.
    pub fn validate(&self) -> PhysicsResult<()> {
        if self.simulation_id.is_empty() {
            return Err(PhysicsError::SammParsing(
                "simulationId must not be empty".to_string(),
            ));
        }
        if self.model_type.is_empty() {
            return Err(PhysicsError::SammParsing(
                "modelType must not be empty".to_string(),
            ));
        }
        if self.asset_iri.is_empty() {
            return Err(PhysicsError::SammParsing(
                "assetIri must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SammPhysicsMapper
// ──────────────────────────────────────────────────────────────────────────────

/// Converts between [`PhysicsAspect`] and JSON representations.
///
/// The JSON output follows the SAMM Digital Twin JSON payload format used by
/// Eclipse Tractus-X and the AAS-3.0 serialisation.
pub struct SammPhysicsMapper;

impl SammPhysicsMapper {
    /// Serialise a [`PhysicsAspect`] to a SAMM-compatible JSON value.
    ///
    /// The output structure is:
    /// ```json
    /// {
    ///   "simulationId": "...",
    ///   "modelType": "...",
    ///   "physicalDomain": "Thermal",
    ///   "status": "Completed",
    ///   "assetIri": "...",
    ///   "parameters": [...],
    ///   "results": [...]
    /// }
    /// ```
    pub fn to_samm_json(aspect: &PhysicsAspect) -> PhysicsResult<serde_json::Value> {
        serde_json::to_value(aspect)
            .map_err(|e| PhysicsError::SammParsing(format!("JSON serialisation failed: {e}")))
    }

    /// Deserialise a [`PhysicsAspect`] from a SAMM-compatible JSON value.
    pub fn from_samm_json(json: &serde_json::Value) -> PhysicsResult<PhysicsAspect> {
        serde_json::from_value(json.clone())
            .map_err(|e| PhysicsError::SammParsing(format!("JSON deserialisation failed: {e}")))
    }

    /// Deserialise from a JSON string.
    pub fn from_samm_json_str(s: &str) -> PhysicsResult<PhysicsAspect> {
        serde_json::from_str(s)
            .map_err(|e| PhysicsError::SammParsing(format!("JSON parse failed: {e}")))
    }

    /// Build a [`PhysicsAspect`] from flat maps of parameters and results.
    ///
    /// Useful when bridging from the `SammAspectModel` parsed by
    /// `SammAspectParser`.
    pub fn from_flat_maps(
        simulation_id: impl Into<String>,
        model_type: impl Into<String>,
        domain: PhysicalDomain,
        asset_iri: impl Into<String>,
        params: &HashMap<String, (f64, String)>,
        results: &HashMap<String, (f64, String)>,
    ) -> PhysicsAspect {
        let parameters = params
            .iter()
            .map(|(k, (v, u))| SimulationParameter::new(k.clone(), *v, u.clone()))
            .collect();

        let result_values = results
            .iter()
            .map(|(k, (v, u))| SimulationResultValue::new(k.clone(), *v, u.clone()))
            .collect();

        PhysicsAspect {
            simulation_id: simulation_id.into(),
            model_type: model_type.into(),
            physical_domain: domain,
            status: SimulationStatus::Completed,
            parameters,
            results: result_values,
            asset_iri: asset_iri.into(),
            converged: Some(true),
            execution_time_ms: None,
            metadata: HashMap::new(),
        }
    }

    /// Extract all parameter values as a flat `name → value` map.
    pub fn parameters_as_map(aspect: &PhysicsAspect) -> HashMap<String, f64> {
        aspect
            .parameters
            .iter()
            .map(|p| (p.name.clone(), p.value))
            .collect()
    }

    /// Extract all result values as a flat `name → value` map.
    pub fn results_as_map(aspect: &PhysicsAspect) -> HashMap<String, f64> {
        aspect
            .results
            .iter()
            .map(|r| (r.name.clone(), r.value))
            .collect()
    }

    /// Generate a minimal SAMM TTL snippet for this aspect (non-validating,
    /// for display / documentation purposes).
    pub fn to_samm_ttl_snippet(aspect: &PhysicsAspect) -> String {
        let mut out = String::new();
        out.push_str("@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.0.0#> .\n");
        out.push_str("@prefix phys: <http://oxirs.org/physics#> .\n");
        out.push_str("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\n");
        out.push_str(&format!(
            "phys:{} a samm:Aspect ;\n",
            sanitize_id(&aspect.simulation_id)
        ));
        out.push_str(&format!("    rdfs:label \"{}\" ;\n", aspect.model_type));
        out.push_str(&format!(
            "    phys:physicalDomain phys:{} ;\n",
            aspect.physical_domain.as_str()
        ));
        out.push_str(&format!(
            "    phys:status phys:{} .\n",
            aspect.status.as_str()
        ));
        out
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PhysicsAasSubmodel – AAS Submodel representation
// ──────────────────────────────────────────────────────────────────────────────

/// Severity of a submodel element.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AasElementKind {
    /// Submodel (top-level).
    Submodel,
    /// Submodel element collection.
    SubmodelElementCollection,
    /// Single property.
    Property,
}

/// A single AAS submodel element (property or collection).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AasElement {
    pub id_short: String,
    pub model_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<AasElement>,
}

impl AasElement {
    fn property(id_short: impl Into<String>, value: serde_json::Value, value_type: &str) -> Self {
        Self {
            id_short: id_short.into(),
            model_type: "Property".to_string(),
            value: Some(value),
            value_type: Some(value_type.to_string()),
            description: None,
            children: Vec::new(),
        }
    }

    fn collection(id_short: impl Into<String>, children: Vec<AasElement>) -> Self {
        Self {
            id_short: id_short.into(),
            model_type: "SubmodelElementCollection".to_string(),
            value: None,
            value_type: None,
            description: None,
            children,
        }
    }
}

/// Asset Administration Shell (AAS) submodel for physics simulation data.
///
/// Follows the AAS metamodel v3.0 JSON serialisation format (IEC 63278).
///
/// The submodel structure is:
/// ```json
/// {
///   "modelType": "Submodel",
///   "idShort": "PhysicsSimulation",
///   "id": "urn:...",
///   "submodelElements": [
///     { "modelType": "Property", "idShort": "SimulationId", "value": "..." },
///     { "modelType": "Property", "idShort": "ModelType", "value": "..." },
///     { "modelType": "Property", "idShort": "PhysicalDomain", "value": "..." },
///     { "modelType": "Property", "idShort": "Status", "value": "..." },
///     { "modelType": "SubmodelElementCollection", "idShort": "Parameters", ... },
///     { "modelType": "SubmodelElementCollection", "idShort": "Results", ... }
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsAasSubmodel {
    /// AAS submodel identifier (IRI).
    pub id: String,
    /// Short human-readable identifier.
    pub id_short: String,
    /// Semantic ID linking to a SAMM aspect urn.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_id: Option<String>,
    /// Top-level AAS submodel elements.
    pub submodel_elements: Vec<AasElement>,
}

impl PhysicsAasSubmodel {
    /// Build an AAS submodel from a [`PhysicsAspect`].
    pub fn from_aspect(aspect: &PhysicsAspect) -> Self {
        let mut elements: Vec<AasElement> = vec![
            AasElement::property(
                "SimulationId",
                serde_json::Value::String(aspect.simulation_id.clone()),
                "xs:string",
            ),
            AasElement::property(
                "ModelType",
                serde_json::Value::String(aspect.model_type.clone()),
                "xs:string",
            ),
            AasElement::property(
                "PhysicalDomain",
                serde_json::Value::String(aspect.physical_domain.as_str().to_string()),
                "xs:string",
            ),
            AasElement::property(
                "Status",
                serde_json::Value::String(aspect.status.as_str().to_string()),
                "xs:string",
            ),
            AasElement::property(
                "AssetIri",
                serde_json::Value::String(aspect.asset_iri.clone()),
                "xs:anyURI",
            ),
        ];

        // Converged
        if let Some(conv) = aspect.converged {
            elements.push(AasElement::property(
                "Converged",
                serde_json::Value::Bool(conv),
                "xs:boolean",
            ));
        }

        // Execution time
        if let Some(et) = aspect.execution_time_ms {
            elements.push(AasElement::property(
                "ExecutionTimeMs",
                serde_json::json!(et),
                "xs:long",
            ));
        }

        // Parameters collection
        let param_children: Vec<AasElement> = aspect
            .parameters
            .iter()
            .map(|p| {
                AasElement::collection(
                    sanitize_id(&p.name),
                    vec![
                        AasElement::property("Name", serde_json::json!(p.name), "xs:string"),
                        AasElement::property("Value", serde_json::json!(p.value), "xs:double"),
                        AasElement::property("Unit", serde_json::json!(p.unit), "xs:string"),
                    ],
                )
            })
            .collect();
        elements.push(AasElement::collection("Parameters", param_children));

        // Results collection
        let result_children: Vec<AasElement> = aspect
            .results
            .iter()
            .map(|r| {
                AasElement::collection(
                    sanitize_id(&r.name),
                    vec![
                        AasElement::property("Name", serde_json::json!(r.name), "xs:string"),
                        AasElement::property("Value", serde_json::json!(r.value), "xs:double"),
                        AasElement::property("Unit", serde_json::json!(r.unit), "xs:string"),
                    ],
                )
            })
            .collect();
        elements.push(AasElement::collection("Results", result_children));

        // Metadata
        if !aspect.metadata.is_empty() {
            let meta_children: Vec<AasElement> = aspect
                .metadata
                .iter()
                .map(|(k, v)| {
                    AasElement::property(
                        sanitize_id(k),
                        serde_json::Value::String(v.clone()),
                        "xs:string",
                    )
                })
                .collect();
            elements.push(AasElement::collection("Metadata", meta_children));
        }

        Self {
            id: format!(
                "urn:oxirs:physics:submodel:{}",
                sanitize_id(&aspect.simulation_id)
            ),
            id_short: "PhysicsSimulation".to_string(),
            semantic_id: Some("urn:samm:org.eclipse.esmf.samm:meta-model:2.0.0#Aspect".to_string()),
            submodel_elements: elements,
        }
    }

    /// Serialise to AAS JSON format.
    pub fn to_aas_json(&self) -> PhysicsResult<serde_json::Value> {
        serde_json::to_value(self)
            .map_err(|e| PhysicsError::SammParsing(format!("AAS JSON serialisation failed: {e}")))
    }

    /// Deserialise from AAS JSON format.
    pub fn from_aas_json(json: &serde_json::Value) -> PhysicsResult<Self> {
        serde_json::from_value(json.clone())
            .map_err(|e| PhysicsError::SammParsing(format!("AAS JSON deserialisation failed: {e}")))
    }

    /// Deserialise from an AAS JSON string.
    pub fn from_aas_json_str(s: &str) -> PhysicsResult<Self> {
        serde_json::from_str(s)
            .map_err(|e| PhysicsError::SammParsing(format!("AAS JSON parse failed: {e}")))
    }

    /// Find a top-level element by `idShort`.
    pub fn find_element(&self, id_short: &str) -> Option<&AasElement> {
        self.submodel_elements
            .iter()
            .find(|e| e.id_short == id_short)
    }

    /// Extract a string property value by idShort.
    pub fn get_string_property(&self, id_short: &str) -> Option<String> {
        self.find_element(id_short)
            .and_then(|e| e.value.as_ref())
            .and_then(|v| v.as_str().map(str::to_string))
    }

    /// Reconstruct a [`PhysicsAspect`] from this AAS submodel.
    pub fn to_aspect(&self) -> PhysicsResult<PhysicsAspect> {
        let simulation_id = self
            .get_string_property("SimulationId")
            .ok_or_else(|| PhysicsError::SammParsing("missing SimulationId".to_string()))?;

        let model_type = self
            .get_string_property("ModelType")
            .ok_or_else(|| PhysicsError::SammParsing("missing ModelType".to_string()))?;

        let domain_str = self
            .get_string_property("PhysicalDomain")
            .ok_or_else(|| PhysicsError::SammParsing("missing PhysicalDomain".to_string()))?;
        let physical_domain = PhysicalDomain::parse(&domain_str)?;

        let status_str = self
            .get_string_property("Status")
            .unwrap_or_else(|| "Pending".to_string());
        let status = SimulationStatus::parse(&status_str)?;

        let asset_iri = self.get_string_property("AssetIri").unwrap_or_default();

        let converged = self
            .find_element("Converged")
            .and_then(|e| e.value.as_ref())
            .and_then(|v| v.as_bool());

        let execution_time_ms = self
            .find_element("ExecutionTimeMs")
            .and_then(|e| e.value.as_ref())
            .and_then(|v| v.as_u64());

        // Reconstruct parameters from the collection
        let parameters = self
            .find_element("Parameters")
            .map(|coll| {
                coll.children
                    .iter()
                    .filter_map(|child| {
                        let name = child
                            .children
                            .iter()
                            .find(|c| c.id_short == "Name")
                            .and_then(|c| c.value.as_ref())
                            .and_then(|v| v.as_str().map(str::to_string))?;
                        let value = child
                            .children
                            .iter()
                            .find(|c| c.id_short == "Value")
                            .and_then(|c| c.value.as_ref())
                            .and_then(|v| v.as_f64())?;
                        let unit = child
                            .children
                            .iter()
                            .find(|c| c.id_short == "Unit")
                            .and_then(|c| c.value.as_ref())
                            .and_then(|v| v.as_str().map(str::to_string))
                            .unwrap_or_default();
                        Some(SimulationParameter {
                            name,
                            value,
                            unit,
                            description: None,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Reconstruct results from the collection
        let results = self
            .find_element("Results")
            .map(|coll| {
                coll.children
                    .iter()
                    .filter_map(|child| {
                        let name = child
                            .children
                            .iter()
                            .find(|c| c.id_short == "Name")
                            .and_then(|c| c.value.as_ref())
                            .and_then(|v| v.as_str().map(str::to_string))?;
                        let value = child
                            .children
                            .iter()
                            .find(|c| c.id_short == "Value")
                            .and_then(|c| c.value.as_ref())
                            .and_then(|v| v.as_f64())?;
                        let unit = child
                            .children
                            .iter()
                            .find(|c| c.id_short == "Unit")
                            .and_then(|c| c.value.as_ref())
                            .and_then(|v| v.as_str().map(str::to_string))
                            .unwrap_or_default();
                        Some(SimulationResultValue {
                            name,
                            value,
                            unit,
                            time_step: None,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(PhysicsAspect {
            simulation_id,
            model_type,
            physical_domain,
            status,
            parameters,
            results,
            asset_iri,
            converged,
            execution_time_ms,
            metadata: HashMap::new(),
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper
// ──────────────────────────────────────────────────────────────────────────────

/// Replace non-alphanumeric characters with underscores for safe AAS idShort.
fn sanitize_id(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SimulationStatus ──────────────────────────────────────────────────────

    #[test]
    fn test_simulation_status_from_str_valid() {
        assert_eq!(
            SimulationStatus::parse("pending").unwrap(),
            SimulationStatus::Pending
        );
        assert_eq!(
            SimulationStatus::parse("Running").unwrap(),
            SimulationStatus::Running
        );
        assert_eq!(
            SimulationStatus::parse("Completed").unwrap(),
            SimulationStatus::Completed
        );
        assert_eq!(
            SimulationStatus::parse("done").unwrap(),
            SimulationStatus::Completed
        );
        assert_eq!(
            SimulationStatus::parse("failed").unwrap(),
            SimulationStatus::Failed
        );
        assert_eq!(
            SimulationStatus::parse("error").unwrap(),
            SimulationStatus::Failed
        );
    }

    #[test]
    fn test_simulation_status_from_str_invalid() {
        assert!(SimulationStatus::parse("unknown_state").is_err());
    }

    #[test]
    fn test_simulation_status_is_terminal() {
        assert!(!SimulationStatus::Pending.is_terminal());
        assert!(!SimulationStatus::Running.is_terminal());
        assert!(SimulationStatus::Completed.is_terminal());
        assert!(SimulationStatus::Failed.is_terminal());
    }

    #[test]
    fn test_simulation_status_display() {
        assert_eq!(format!("{}", SimulationStatus::Completed), "Completed");
        assert_eq!(format!("{}", SimulationStatus::Failed), "Failed");
    }

    // ── PhysicalDomain ────────────────────────────────────────────────────────

    #[test]
    fn test_physical_domain_from_str_valid() {
        assert_eq!(
            PhysicalDomain::parse("thermal").unwrap(),
            PhysicalDomain::Thermal
        );
        assert_eq!(
            PhysicalDomain::parse("heat").unwrap(),
            PhysicalDomain::Thermal
        );
        assert_eq!(PhysicalDomain::parse("cfd").unwrap(), PhysicalDomain::Fluid);
        assert_eq!(
            PhysicalDomain::parse("fem").unwrap(),
            PhysicalDomain::Structural
        );
        assert_eq!(
            PhysicalDomain::parse("em").unwrap(),
            PhysicalDomain::Electromagnetic
        );
        assert_eq!(
            PhysicalDomain::parse("coupled").unwrap(),
            PhysicalDomain::Multiphysics
        );
    }

    #[test]
    fn test_physical_domain_from_str_invalid() {
        assert!(PhysicalDomain::parse("quantum_gravity").is_err());
    }

    #[test]
    fn test_physical_domain_display() {
        assert_eq!(format!("{}", PhysicalDomain::Thermal), "Thermal");
        assert_eq!(
            format!("{}", PhysicalDomain::Electromagnetic),
            "Electromagnetic"
        );
    }

    // ── PhysicsAspect ─────────────────────────────────────────────────────────

    fn make_completed_aspect() -> PhysicsAspect {
        PhysicsAspect::new(
            "sim-001",
            "ThermalFEM",
            PhysicalDomain::Thermal,
            "urn:example:motor:42",
        )
        .with_parameter(SimulationParameter::new("inletTemp", 300.0, "K"))
        .with_parameter(SimulationParameter::new("heatFlux", 1000.0, "W-PER-M2"))
        .with_result(SimulationResultValue::new("maxTemperature", 450.0, "K"))
        .with_result(SimulationResultValue::new("avgTemperature", 375.0, "K"))
    }

    #[test]
    fn test_aspect_new_defaults_pending() {
        let aspect = PhysicsAspect::new("s1", "M1", PhysicalDomain::Fluid, "urn:ex:1");
        assert_eq!(aspect.status, SimulationStatus::Pending);
        assert!(aspect.parameters.is_empty());
        assert!(aspect.results.is_empty());
    }

    #[test]
    fn test_aspect_validate_ok() {
        let aspect = make_completed_aspect();
        assert!(aspect.validate().is_ok());
    }

    #[test]
    fn test_aspect_validate_empty_id_err() {
        let aspect = PhysicsAspect::new("", "M1", PhysicalDomain::Thermal, "urn:ex:1");
        assert!(aspect.validate().is_err());
    }

    #[test]
    fn test_aspect_start_and_complete() {
        let mut aspect = PhysicsAspect::new("s1", "M1", PhysicalDomain::Thermal, "urn:ex:1");
        assert!(aspect.start().is_ok());
        assert_eq!(aspect.status, SimulationStatus::Running);
        assert!(aspect.complete(true, 1200).is_ok());
        assert_eq!(aspect.status, SimulationStatus::Completed);
        assert_eq!(aspect.converged, Some(true));
        assert_eq!(aspect.execution_time_ms, Some(1200));
    }

    #[test]
    fn test_aspect_start_when_not_pending_fails() {
        let mut aspect = PhysicsAspect::new("s1", "M1", PhysicalDomain::Thermal, "urn:ex:1");
        aspect.start().unwrap();
        // Cannot start again while Running
        assert!(aspect.start().is_err());
    }

    #[test]
    fn test_aspect_fail() {
        let mut aspect = PhysicsAspect::new("s1", "M1", PhysicalDomain::Thermal, "urn:ex:1");
        aspect.start().unwrap();
        assert!(aspect.fail("solver diverged").is_ok());
        assert_eq!(aspect.status, SimulationStatus::Failed);
        assert_eq!(
            aspect.metadata.get("failure_reason").map(String::as_str),
            Some("solver diverged")
        );
    }

    #[test]
    fn test_aspect_parameter_lookup() {
        let aspect = make_completed_aspect();
        let param = aspect.parameter("inletTemp");
        assert!(param.is_some());
        assert!((param.unwrap().value - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_aspect_result_lookup() {
        let aspect = make_completed_aspect();
        let res = aspect.result_value("maxTemperature");
        assert!(res.is_some());
        assert!((res.unwrap().value - 450.0).abs() < 1e-10);
    }

    // ── SammPhysicsMapper ─────────────────────────────────────────────────────

    #[test]
    fn test_to_samm_json_roundtrip() {
        let aspect = make_completed_aspect();
        let json = SammPhysicsMapper::to_samm_json(&aspect).expect("serialise");
        let restored = SammPhysicsMapper::from_samm_json(&json).expect("deserialise");
        assert_eq!(restored.simulation_id, aspect.simulation_id);
        assert_eq!(restored.model_type, aspect.model_type);
        assert_eq!(restored.physical_domain, aspect.physical_domain);
        assert_eq!(restored.parameters.len(), aspect.parameters.len());
        assert_eq!(restored.results.len(), aspect.results.len());
    }

    #[test]
    fn test_from_samm_json_str_roundtrip() {
        let aspect = make_completed_aspect();
        let json_str = serde_json::to_string(&aspect).expect("to_string");
        let restored = SammPhysicsMapper::from_samm_json_str(&json_str).expect("from_str");
        assert_eq!(restored.simulation_id, aspect.simulation_id);
    }

    #[test]
    fn test_from_flat_maps() {
        let mut params = HashMap::new();
        params.insert("temperature".to_string(), (300.0, "K".to_string()));
        let mut results = HashMap::new();
        results.insert("maxTemp".to_string(), (500.0, "K".to_string()));

        let aspect = SammPhysicsMapper::from_flat_maps(
            "sim-002",
            "CFD",
            PhysicalDomain::Fluid,
            "urn:ex:pump",
            &params,
            &results,
        );
        assert_eq!(aspect.simulation_id, "sim-002");
        assert_eq!(aspect.parameters.len(), 1);
        assert_eq!(aspect.results.len(), 1);
    }

    #[test]
    fn test_parameters_as_map() {
        let aspect = make_completed_aspect();
        let map = SammPhysicsMapper::parameters_as_map(&aspect);
        assert!((map["inletTemp"] - 300.0).abs() < 1e-10);
        assert!((map["heatFlux"] - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_results_as_map() {
        let aspect = make_completed_aspect();
        let map = SammPhysicsMapper::results_as_map(&aspect);
        assert!((map["maxTemperature"] - 450.0).abs() < 1e-10);
        assert!((map["avgTemperature"] - 375.0).abs() < 1e-10);
    }

    #[test]
    fn test_to_samm_ttl_snippet_contains_aspect() {
        let aspect = make_completed_aspect();
        let ttl = SammPhysicsMapper::to_samm_ttl_snippet(&aspect);
        assert!(
            ttl.contains("samm:Aspect"),
            "expected samm:Aspect in TTL snippet"
        );
        assert!(
            ttl.contains("Thermal"),
            "expected PhysicalDomain in TTL snippet"
        );
    }

    // ── PhysicsAasSubmodel ────────────────────────────────────────────────────

    #[test]
    fn test_aas_submodel_from_aspect_structure() {
        let aspect = make_completed_aspect();
        let submodel = PhysicsAasSubmodel::from_aspect(&aspect);
        assert_eq!(submodel.id_short, "PhysicsSimulation");
        assert!(submodel.id.contains("sim-001"));
        assert!(submodel.semantic_id.is_some());
    }

    #[test]
    fn test_aas_submodel_json_roundtrip() {
        let aspect = make_completed_aspect();
        let submodel = PhysicsAasSubmodel::from_aspect(&aspect);
        let json = submodel.to_aas_json().expect("to_aas_json");
        let restored = PhysicsAasSubmodel::from_aas_json(&json).expect("from_aas_json");
        assert_eq!(restored.id, submodel.id);
        assert_eq!(restored.id_short, submodel.id_short);
        assert_eq!(
            restored.submodel_elements.len(),
            submodel.submodel_elements.len()
        );
    }

    #[test]
    fn test_aas_submodel_to_aspect_roundtrip() {
        let mut original = make_completed_aspect();
        original.status = SimulationStatus::Completed;
        original.converged = Some(true);
        original.execution_time_ms = Some(5000);

        let submodel = PhysicsAasSubmodel::from_aspect(&original);
        let restored = submodel.to_aspect().expect("to_aspect");

        assert_eq!(restored.simulation_id, original.simulation_id);
        assert_eq!(restored.model_type, original.model_type);
        assert_eq!(restored.physical_domain, original.physical_domain);
        assert_eq!(restored.parameters.len(), original.parameters.len());
        assert_eq!(restored.results.len(), original.results.len());
        assert_eq!(restored.converged, Some(true));
    }

    #[test]
    fn test_aas_submodel_get_string_property() {
        let aspect = make_completed_aspect();
        let submodel = PhysicsAasSubmodel::from_aspect(&aspect);
        assert_eq!(
            submodel.get_string_property("SimulationId"),
            Some("sim-001".to_string())
        );
        assert_eq!(
            submodel.get_string_property("ModelType"),
            Some("ThermalFEM".to_string())
        );
        assert_eq!(
            submodel.get_string_property("PhysicalDomain"),
            Some("Thermal".to_string())
        );
    }

    #[test]
    fn test_aas_submodel_parameters_collection() {
        let aspect = make_completed_aspect();
        let submodel = PhysicsAasSubmodel::from_aspect(&aspect);
        let params_elem = submodel.find_element("Parameters");
        assert!(params_elem.is_some(), "expected Parameters collection");
        let coll = params_elem.unwrap();
        assert_eq!(coll.children.len(), 2, "expected 2 parameter children");
    }

    #[test]
    fn test_aas_submodel_results_collection() {
        let aspect = make_completed_aspect();
        let submodel = PhysicsAasSubmodel::from_aspect(&aspect);
        let results_elem = submodel.find_element("Results");
        assert!(results_elem.is_some(), "expected Results collection");
        let coll = results_elem.unwrap();
        assert_eq!(coll.children.len(), 2, "expected 2 result children");
    }

    #[test]
    fn test_aas_submodel_from_json_str_roundtrip() {
        let aspect = make_completed_aspect();
        let submodel = PhysicsAasSubmodel::from_aspect(&aspect);
        let json_str = serde_json::to_string(&submodel).expect("to_string");
        let restored = PhysicsAasSubmodel::from_aas_json_str(&json_str).expect("from_str");
        assert_eq!(restored.id, submodel.id);
    }

    #[test]
    fn test_sanitize_id() {
        assert_eq!(sanitize_id("inletTemp"), "inletTemp");
        assert_eq!(sanitize_id("max temperature"), "max_temperature");
        assert_eq!(sanitize_id("value:123"), "value_123");
    }

    #[test]
    fn test_simulation_parameter_with_description() {
        let p =
            SimulationParameter::new("vel", 2.5, "M-PER-SEC").with_description("inlet velocity");
        assert_eq!(p.description, Some("inlet velocity".to_string()));
    }

    // ── Additional PhysicsAspect tests ───────────────────────────────────────

    #[test]
    fn test_physics_aspect_with_parameter_builder() {
        let aspect = PhysicsAspect::new(
            "sim-100",
            "ThermalFEM",
            PhysicalDomain::Thermal,
            "urn:example:asset:100",
        )
        .with_parameter(SimulationParameter::new("temp", 300.0, "K"))
        .with_parameter(SimulationParameter::new("pressure", 101325.0, "PA"));

        assert_eq!(aspect.parameters.len(), 2);
        assert_eq!(aspect.parameter("temp").unwrap().value, 300.0);
        assert_eq!(aspect.parameter("pressure").unwrap().unit, "PA");
    }

    #[test]
    fn test_physics_aspect_with_result_builder() {
        let aspect = PhysicsAspect::new(
            "sim-200",
            "FluidDynamics",
            PhysicalDomain::Fluid,
            "urn:example:asset:200",
        )
        .with_result(SimulationResultValue::new("max_velocity", 5.0, "M-PER-SEC"))
        .with_result(SimulationResultValue::new("pressure_drop", 500.0, "PA"));

        assert_eq!(aspect.results.len(), 2);
        assert_eq!(aspect.result_value("max_velocity").unwrap().value, 5.0);
    }

    #[test]
    fn test_physics_aspect_state_transitions() {
        let mut aspect = PhysicsAspect::new(
            "sim-300",
            "StructuralFEM",
            PhysicalDomain::Structural,
            "urn:example:asset:300",
        );

        assert!(aspect.start().is_ok());
        assert_eq!(aspect.status, SimulationStatus::Running);

        assert!(aspect.complete(true, 2000).is_ok());
        assert_eq!(aspect.status, SimulationStatus::Completed);
        assert_eq!(aspect.converged, Some(true));
        assert_eq!(aspect.execution_time_ms, Some(2000));
    }

    #[test]
    fn test_physics_aspect_fail_transition() {
        let mut aspect = PhysicsAspect::new(
            "sim-400",
            "EMSolver",
            PhysicalDomain::Electromagnetic,
            "urn:example:asset:400",
        );
        aspect.start().unwrap();

        assert!(aspect.fail("diverged at step 42").is_ok());
        assert_eq!(aspect.status, SimulationStatus::Failed);
        let reason = aspect
            .metadata
            .get("failure_reason")
            .map(|s| s.as_str())
            .unwrap_or("");
        assert!(
            reason.contains("diverged"),
            "expected failure reason to contain 'diverged', got: {reason}"
        );
    }

    #[test]
    fn test_physics_aspect_validate_ok() {
        let mut aspect = PhysicsAspect::new(
            "sim-500",
            "ThermalFEM",
            PhysicalDomain::Thermal,
            "urn:example:asset:500",
        );
        aspect.start().unwrap();
        aspect.complete(true, 0).unwrap();
        assert!(
            aspect.validate().is_ok(),
            "completed aspect with non-empty ID must validate"
        );
    }

    #[test]
    fn test_physics_aspect_validate_empty_id_fails() {
        let mut aspect = PhysicsAspect::new(
            "valid-id",
            "ThermalFEM",
            PhysicalDomain::Thermal,
            "urn:example:asset:valid",
        );
        aspect.simulation_id = String::new();
        assert!(
            aspect.validate().is_err(),
            "empty simulation_id must fail validation"
        );
    }

    #[test]
    fn test_samm_mapper_to_from_json_roundtrip() {
        let mut aspect = PhysicsAspect::new(
            "sim-rt-001",
            "ThermalFEM",
            PhysicalDomain::Thermal,
            "urn:example:asset:rt-001",
        );
        aspect.start().unwrap();
        aspect.complete(true, 500).unwrap();
        let aspect = aspect
            .with_parameter(SimulationParameter::new("inletTemp", 400.0, "K"))
            .with_result(SimulationResultValue::new("maxTemp", 450.0, "K"));

        let json = SammPhysicsMapper::to_samm_json(&aspect).expect("to_samm_json");
        let restored = SammPhysicsMapper::from_samm_json(&json).expect("from_samm_json");

        assert_eq!(restored.simulation_id, aspect.simulation_id);
        assert_eq!(restored.physical_domain, aspect.physical_domain);
        assert_eq!(restored.parameters.len(), aspect.parameters.len());
        assert_eq!(restored.results.len(), aspect.results.len());
    }

    #[test]
    fn test_samm_mapper_from_samm_json_str() {
        let mut aspect = PhysicsAspect::new(
            "sim-str-001",
            "FluidDynamics",
            PhysicalDomain::Fluid,
            "urn:example:asset:str-001",
        );
        aspect.start().unwrap();
        let json = SammPhysicsMapper::to_samm_json(&aspect).expect("to_samm_json");
        let s = serde_json::to_string(&json).expect("to_string");
        let restored = SammPhysicsMapper::from_samm_json_str(&s).expect("from_samm_json_str");
        assert_eq!(restored.simulation_id, aspect.simulation_id);
    }

    #[test]
    fn test_samm_mapper_parameters_as_map() {
        let aspect = PhysicsAspect::new(
            "sim-map-001",
            "StructuralFEM",
            PhysicalDomain::Structural,
            "urn:example:asset:map-001",
        )
        .with_parameter(SimulationParameter::new("young_modulus", 200e9, "PA"))
        .with_parameter(SimulationParameter::new("poisson_ratio", 0.3, "UNITLESS"));

        let map = SammPhysicsMapper::parameters_as_map(&aspect);
        assert_eq!(map.len(), 2);
        assert!((map["young_modulus"] - 200e9).abs() < 1.0);
        assert!((map["poisson_ratio"] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_samm_mapper_results_as_map() {
        let aspect = PhysicsAspect::new(
            "sim-resmap",
            "FluidDynamics",
            PhysicalDomain::Fluid,
            "urn:example:asset:resmap",
        )
        .with_result(SimulationResultValue::new("drag_coeff", 0.35, "UNITLESS"))
        .with_result(SimulationResultValue::new("lift_coeff", 1.2, "UNITLESS"));

        let map = SammPhysicsMapper::results_as_map(&aspect);
        assert!((map["drag_coeff"] - 0.35).abs() < 1e-10);
        assert!((map["lift_coeff"] - 1.2).abs() < 1e-10);
    }

    #[test]
    fn test_samm_ttl_snippet_contains_aspect() {
        let mut aspect = PhysicsAspect::new(
            "sim-ttl-001",
            "ThermalFEM",
            PhysicalDomain::Thermal,
            "urn:example:asset:ttl-001",
        );
        aspect.start().unwrap();
        aspect.complete(true, 0).unwrap();
        let ttl = SammPhysicsMapper::to_samm_ttl_snippet(&aspect);
        assert!(
            ttl.contains("samm:Aspect"),
            "TTL must contain samm:Aspect declaration"
        );
        assert!(
            ttl.contains("sim-ttl-001"),
            "TTL must contain simulation ID"
        );
    }

    #[test]
    fn test_samm_mapper_from_flat_maps() {
        let mut params = HashMap::new();
        params.insert("temperature".to_string(), (350.0, "K".to_string()));
        params.insert("pressure".to_string(), (101325.0, "PA".to_string()));

        let mut results = HashMap::new();
        results.insert("max_stress".to_string(), (1e6, "PA".to_string()));

        let aspect = SammPhysicsMapper::from_flat_maps(
            "sim-flat-001",
            "StructuralFEM",
            PhysicalDomain::Structural,
            "urn:example:asset:flat-001",
            &params,
            &results,
        );

        assert_eq!(aspect.simulation_id, "sim-flat-001");
        assert_eq!(aspect.physical_domain, PhysicalDomain::Structural);
        assert_eq!(aspect.parameters.len(), 2);
        assert_eq!(aspect.results.len(), 1);
    }

    #[test]
    fn test_physical_domain_all_parse() {
        let domains = [
            ("thermal", PhysicalDomain::Thermal),
            ("fluid", PhysicalDomain::Fluid),
            ("structural", PhysicalDomain::Structural),
            ("electromagnetic", PhysicalDomain::Electromagnetic),
            ("multiphysics", PhysicalDomain::Multiphysics),
        ];
        for (s, expected) in &domains {
            let parsed = PhysicalDomain::parse(s).expect(s);
            assert_eq!(parsed, *expected, "failed to parse domain: {s}");
        }
    }

    #[test]
    fn test_simulation_parameter_unit_field() {
        let p = SimulationParameter::new("viscosity", 1e-3, "PA-SEC");
        assert_eq!(p.unit, "PA-SEC");
        assert!((p.value - 1e-3).abs() < 1e-15);
    }

    #[test]
    fn test_simulation_result_value_unit_field() {
        let r = SimulationResultValue::new("peak_temperature", 800.0, "K");
        assert_eq!(r.unit, "K");
        assert!((r.value - 800.0).abs() < 1e-10);
    }

    #[test]
    fn test_aas_submodel_find_element_missing() {
        use super::*;
        let aspect = PhysicsAspect::new(
            "sim-find-001",
            "ThermalFEM",
            PhysicalDomain::Thermal,
            "urn:example:asset:find-001",
        );
        let submodel = PhysicsAasSubmodel::from_aspect(&aspect);
        assert!(submodel.find_element("NonExistent").is_none());
    }

    #[test]
    fn test_aas_submodel_semantic_id_is_samm_iri() {
        let mut aspect = PhysicsAspect::new(
            "sim-semid-001",
            "FluidDynamics",
            PhysicalDomain::Fluid,
            "urn:example:asset:semid-001",
        );
        aspect.start().unwrap();
        aspect.complete(true, 0).unwrap();
        let submodel = PhysicsAasSubmodel::from_aspect(&aspect);
        let sem_id = submodel.semantic_id.as_deref().unwrap_or("");
        assert!(
            sem_id.contains("samm") || sem_id.contains("esmf"),
            "semantic_id should be a SAMM IRI, got: {sem_id}"
        );
    }
}
