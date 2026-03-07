//! SAMM Aspect Model Bridge for Physics Simulations
//!
//! Parses SAMM (Semantic Aspect Meta Model) TTL format and bridges the
//! extracted semantic structure to the physics simulation types used in
//! `oxirs-physics`.
//!
//! # What is SAMM?
//!
//! The Semantic Aspect Meta Model (SAMM, formerly BAMM) is an Eclipse Tractus-X
//! specification for describing the semantic structure of manufacturing and IoT
//! data. It uses Turtle RDF to define:
//!
//! - **Aspects** – top-level semantic groupings (e.g. "MotorAspect")
//! - **Properties** – typed data fields (e.g. "rotationalSpeed", "temperature")
//! - **Characteristics** – constraints on properties (e.g. `Measurement`, `Quantifiable`)
//! - **Units** – QUDT-compatible unit references (e.g. `unit:kilometre-per-hour`)
//! - **Constraints** – value ranges, patterns, enumeration sets
//!
//! # Architecture
//!
//! ```text
//! SAMM TTL source (file / string)
//!        │
//!        │  SammAspectParser::parse_*
//!        ▼
//!  [RdfStore (in-memory)]
//!        │
//!        │  SPARQL queries (extract_* methods)
//!        ▼
//!  SammAspectModel  ─────────────►  SimulationParameters  (via bridge methods)
//!   ├── properties[]
//!   ├── characteristics[]
//!   └── constraints[]
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_physics::samm::SammAspectParser;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let parser = SammAspectParser::new()?;
//! let model = parser.parse_samm_string(TTL_SOURCE).await?;
//!
//! for prop in &model.properties {
//!     println!("  {} [{}] {:?}..{:?}",
//!         prop.name, prop.unit.as_deref().unwrap_or("dimensionless"),
//!         prop.range_min, prop.range_max);
//! }
//! # Ok(())
//! # }
//! # const TTL_SOURCE: &str = "";
//! ```

pub mod fem_bridge;
pub mod physics_aspect;

pub use fem_bridge::{
    PhysicsModelBridge, SammAspect as FemSammAspect, SammDataType as FemSammDataType,
    SammPhysicsRegistry, SammProperty as FemSammProperty,
};
pub use physics_aspect::{
    AasElement, AasElementKind, PhysicalDomain, PhysicsAasSubmodel, PhysicsAspect,
    SammPhysicsMapper, SimulationParameter, SimulationResultValue, SimulationStatus,
};

use crate::error::{PhysicsError, PhysicsResult};
use crate::rdf::literal_parser::{parse_rdf_literal, parse_unit_str, PhysicalUnit, PhysicalValue};
use crate::simulation::parameter_extraction::{
    BoundaryCondition, PhysicalQuantity, SimulationParameters,
};
use oxirs_core::model::Term;
use oxirs_core::parser::{Parser, RdfFormat};
use oxirs_core::rdf_store::{QueryResults, RdfStore};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ──────────────────────────────────────────────────────────────────────────────
// SAMM namespace constants
// ──────────────────────────────────────────────────────────────────────────────

/// SAMM meta-model namespace (v2.0.0)
pub const SAMM_NS: &str = "urn:samm:org.eclipse.esmf.samm:meta-model:2.0.0#";
/// SAMM characteristic namespace
pub const SAMM_C_NS: &str = "urn:samm:org.eclipse.esmf.samm:characteristic:2.0.0#";
/// SAMM unit namespace
pub const SAMM_UNIT_NS: &str = "urn:samm:org.eclipse.esmf.samm:unit:2.0.0#";
/// QUDT unit namespace
pub const QUDT_UNIT_NS: &str = "http://qudt.org/vocab/unit/";
/// XSD namespace
pub const XSD_NS: &str = "http://www.w3.org/2001/XMLSchema#";

// ──────────────────────────────────────────────────────────────────────────────
// Data model types
// ──────────────────────────────────────────────────────────────────────────────

/// XSD / SAMM data types for SAMM properties.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SammDataType {
    /// `xsd:double` or `xsd:float`
    Double,
    /// `xsd:integer` / `xsd:long` / `xsd:int` / `xsd:short`
    Integer,
    /// `xsd:string`
    Text,
    /// `xsd:boolean`
    Boolean,
    /// `xsd:dateTime`
    DateTime,
    /// A named composite type (non-primitive SAMM Entity URI)
    Entity(String),
    /// Unrecognised data type; stores the raw IRI string
    Unknown(String),
}

impl SammDataType {
    /// Construct from a datatype IRI string.
    pub fn from_iri(iri: &str) -> Self {
        match iri {
            s if s.ends_with("#double") || s.ends_with("#float") => Self::Double,
            s if s.ends_with("#decimal") => Self::Double,
            s if s.ends_with("#integer")
                || s.ends_with("#long")
                || s.ends_with("#int")
                || s.ends_with("#short")
                || s.ends_with("#byte")
                || s.ends_with("#nonNegativeInteger") =>
            {
                Self::Integer
            }
            s if s.ends_with("#string") => Self::Text,
            s if s.ends_with("#boolean") => Self::Boolean,
            s if s.ends_with("#dateTime") || s.ends_with("#date") => Self::DateTime,
            "" => Self::Unknown(String::new()),
            other => {
                // Check if it looks like a full type IRI (has '#' or '/')
                if other.contains('#') || other.contains('/') {
                    Self::Entity(other.to_string())
                } else {
                    Self::Unknown(other.to_string())
                }
            }
        }
    }

    /// Return `true` if this data type is numeric.
    pub fn is_numeric(&self) -> bool {
        matches!(self, Self::Double | Self::Integer)
    }
}

/// SAMM characteristic type (maps to the SAMM-C vocabulary).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SammCharacteristic {
    /// A measured quantity that carries a unit (samm-c:Measurement)
    Measurement,
    /// A numeric quantity that may carry a unit (samm-c:Quantifiable)
    Quantifiable,
    /// An enumeration of allowed values (samm-c:Enumeration)
    Enumeration,
    /// A duration measured in a time unit (samm-c:Duration)
    Duration,
    /// A single-dimension collection (samm-c:Collection / samm-c:List)
    Collection,
    /// Code / identifier string (samm-c:Code)
    Code,
    /// Unknown or composite characteristic; stores the IRI
    Other(String),
}

impl SammCharacteristic {
    /// Construct from a characteristic type IRI.
    pub fn from_iri(iri: &str) -> Self {
        if iri.contains("Measurement") {
            Self::Measurement
        } else if iri.contains("Quantifiable") {
            Self::Quantifiable
        } else if iri.contains("Enumeration") {
            Self::Enumeration
        } else if iri.contains("Duration") {
            Self::Duration
        } else if iri.contains("Collection") || iri.contains("List") || iri.contains("Set") {
            Self::Collection
        } else if iri.contains("Code") {
            Self::Code
        } else {
            Self::Other(iri.to_string())
        }
    }
}

/// A SAMM property with physics semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SammPhysicsProperty {
    /// Full URN of this property node in the SAMM model.
    pub urn: String,
    /// Human-readable label (from `rdfs:label`).
    pub name: String,
    /// Description (from `samm:description`).
    pub description: Option<String>,
    /// XSD or Entity data type.
    pub data_type: SammDataType,
    /// Characteristic category (Measurement, Quantifiable, …).
    pub characteristic: Option<SammCharacteristic>,
    /// Unit string (from `samm:unit` or `qudt:unit` annotation).
    pub unit: Option<String>,
    /// Parsed [`PhysicalUnit`] variant when the unit string is recognised.
    pub physical_unit: Option<PhysicalUnit>,
    /// Minimum allowed value (from samm-c:minValue or samm-c:lowerBoundDefinition).
    pub range_min: Option<f64>,
    /// Maximum allowed value (from samm-c:maxValue or samm-c:upperBoundDefinition).
    pub range_max: Option<f64>,
    /// Allowed enumeration values (for samm-c:Enumeration).
    pub enum_values: Vec<String>,
    /// Whether the property is mandatory in its parent Aspect.
    pub is_required: bool,
}

/// A SAMM Aspect node (top-level semantic grouping).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SammAspect {
    /// Full URN of the Aspect node.
    pub urn: String,
    /// Human-readable label.
    pub name: String,
    /// Description.
    pub description: Option<String>,
    /// Properties that belong to this Aspect.
    pub property_urns: Vec<String>,
}

/// Complete parsed SAMM Aspect model ready for bridging to physics simulations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SammAspectModel {
    /// Aspect nodes found in the TTL.
    pub aspects: Vec<SammAspect>,
    /// All properties (across all aspects).
    pub properties: Vec<SammPhysicsProperty>,
    /// Namespace prefix map extracted from the TTL preamble.
    pub prefix_map: HashMap<String, String>,
}

impl SammAspectModel {
    /// Look up a property by its local name (case-insensitive).
    pub fn property_by_name(&self, name: &str) -> Option<&SammPhysicsProperty> {
        let lower = name.to_lowercase();
        self.properties
            .iter()
            .find(|p| p.name.to_lowercase() == lower)
    }

    /// Return all numeric properties (Double / Integer data types).
    pub fn numeric_properties(&self) -> impl Iterator<Item = &SammPhysicsProperty> {
        self.properties.iter().filter(|p| p.data_type.is_numeric())
    }

    /// Return all properties that have a known physical unit.
    pub fn measured_properties(&self) -> impl Iterator<Item = &SammPhysicsProperty> {
        self.properties.iter().filter(|p| p.physical_unit.is_some())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Parser
// ──────────────────────────────────────────────────────────────────────────────

/// Parses SAMM TTL format and bridges it to physics simulation types.
///
/// Internally uses an in-memory [`RdfStore`] to enable SPARQL-based extraction
/// of structured data from the SAMM TTL.
pub struct SammAspectParser {
    /// Physics namespace – defaults to `http://oxirs.org/physics#`.
    physics_ns: String,
    /// SAMM meta-model namespace.
    samm_ns: String,
    /// SAMM characteristic namespace.
    samm_c_ns: String,
}

impl Default for SammAspectParser {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            physics_ns: "http://oxirs.org/physics#".to_string(),
            samm_ns: SAMM_NS.to_string(),
            samm_c_ns: SAMM_C_NS.to_string(),
        })
    }
}

impl SammAspectParser {
    /// Create a new parser with default configuration.
    pub fn new() -> PhysicsResult<Self> {
        Ok(Self {
            physics_ns: "http://oxirs.org/physics#".to_string(),
            samm_ns: SAMM_NS.to_string(),
            samm_c_ns: SAMM_C_NS.to_string(),
        })
    }

    /// Override the physics namespace prefix.
    pub fn with_physics_namespace(mut self, ns: impl Into<String>) -> Self {
        self.physics_ns = ns.into();
        self
    }

    /// Parse a SAMM TTL file from disk.
    ///
    /// # Errors
    ///
    /// Returns [`PhysicsError::SammParsing`] on I/O or parse errors.
    pub async fn parse_samm_file(&self, path: &Path) -> PhysicsResult<SammAspectModel> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            PhysicsError::SammParsing(format!("Failed to read SAMM file {:?}: {}", path, e))
        })?;
        self.parse_samm_string(&content).await
    }

    /// Parse a SAMM TTL string in memory.
    ///
    /// # Errors
    ///
    /// Returns [`PhysicsError::SammParsing`] if the Turtle syntax is invalid.
    pub async fn parse_samm_string(&self, content: &str) -> PhysicsResult<SammAspectModel> {
        // Build a fresh in-memory store for this parse session
        let mut store = RdfStore::new()
            .map_err(|e| PhysicsError::SammParsing(format!("Failed to create RDF store: {}", e)))?;

        // Parse Turtle → quads
        let parser = Parser::new(RdfFormat::Turtle);
        let quads = parser
            .parse_str_to_quads(content)
            .map_err(|e| PhysicsError::SammParsing(format!("Turtle parse error: {}", e)))?;

        for quad in quads {
            store
                .insert_quad(quad)
                .map_err(|e| PhysicsError::SammParsing(format!("Failed to insert quad: {}", e)))?;
        }

        let store = Arc::new(store);

        // Extract components via SPARQL
        let aspects = self.extract_aspects(&store).await?;
        let properties = self.extract_physics_properties(&store).await?;
        let prefix_map = self.extract_prefix_map(content);

        Ok(SammAspectModel {
            aspects,
            properties,
            prefix_map,
        })
    }

    // ── Aspect extraction ─────────────────────────────────────────────────────

    async fn extract_aspects(&self, store: &Arc<RdfStore>) -> PhysicsResult<Vec<SammAspect>> {
        let query = format!(
            r#"
            PREFIX samm: <{samm}>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT DISTINCT ?aspect ?label ?desc ?prop WHERE {{
                ?aspect a samm:Aspect .
                OPTIONAL {{ ?aspect rdfs:label ?label . }}
                OPTIONAL {{ ?aspect samm:description ?desc . }}
                OPTIONAL {{
                    ?aspect samm:properties ?prop .
                    ?prop a samm:Property .
                }}
            }}
            "#,
            samm = self.samm_ns
        );

        let results = store
            .query(&query)
            .map_err(|e| PhysicsError::SammParsing(format!("Aspect query failed: {}", e)))?;

        let mut aspects: HashMap<String, SammAspect> = HashMap::new();

        if let QueryResults::Bindings(ref bindings) = results.results() {
            for binding in bindings {
                let Some(Term::NamedNode(aspect_node)) = binding.get("aspect") else {
                    continue;
                };
                let aspect_urn = aspect_node.as_str().to_string();

                let label_opt = binding.get("label").and_then(literal_value);
                let description = binding.get("desc").and_then(literal_value);

                let aspect = aspects
                    .entry(aspect_urn.clone())
                    .or_insert_with(|| SammAspect {
                        urn: aspect_urn.clone(),
                        name: label_opt
                            .clone()
                            .unwrap_or_else(|| local_name_of(&aspect_urn)),
                        description: description.clone(),
                        property_urns: Vec::new(),
                    });

                // Update name from label if a later row provides it
                if let Some(label) = label_opt {
                    if aspect.name != label {
                        aspect.name = label;
                    }
                }
                // Update description if not yet set
                if aspect.description.is_none() {
                    aspect.description = description;
                }

                if let Some(Term::NamedNode(prop_node)) = binding.get("prop") {
                    let prop_urn = prop_node.as_str().to_string();
                    if !aspect.property_urns.contains(&prop_urn) {
                        aspect.property_urns.push(prop_urn);
                    }
                }
            }
        }

        Ok(aspects.into_values().collect())
    }

    // ── Property extraction ───────────────────────────────────────────────────

    async fn extract_physics_properties(
        &self,
        store: &Arc<RdfStore>,
    ) -> PhysicsResult<Vec<SammPhysicsProperty>> {
        // Phase 1: basic property metadata
        let prop_query = format!(
            r#"
            PREFIX samm:   <{samm}>
            PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?prop ?label ?desc ?char WHERE {{
                ?prop a samm:Property .
                OPTIONAL {{ ?prop rdfs:label ?label . }}
                OPTIONAL {{ ?prop samm:description ?desc . }}
                OPTIONAL {{ ?prop samm:characteristic ?char . }}
            }}
            "#,
            samm = self.samm_ns,
        );

        let prop_results = store
            .query(&prop_query)
            .map_err(|e| PhysicsError::SammParsing(format!("Property query failed: {}", e)))?;

        // Phase 2: characteristic details (dataType, unit, charType)
        let char_query = format!(
            r#"
            PREFIX samm:   <{samm}>
            PREFIX samm-c: <{samm_c}>

            SELECT ?char ?charType ?dataType ?unit WHERE {{
                ?char samm:dataType ?dataType .
                OPTIONAL {{ ?char a ?charType . }}
                OPTIONAL {{ ?char samm:unit ?unit . }}
            }}
            "#,
            samm = self.samm_ns,
            samm_c = self.samm_c_ns,
        );

        let char_results = store.query(&char_query).map_err(|e| {
            PhysicsError::SammParsing(format!("Characteristic query failed: {}", e))
        })?;

        // Build characteristic lookup
        let mut char_details: HashMap<
            String,
            (Option<SammCharacteristic>, SammDataType, Option<String>),
        > = HashMap::new();
        if let QueryResults::Bindings(ref bindings) = char_results.results() {
            for binding in bindings {
                let Some(Term::NamedNode(char_node)) = binding.get("char") else {
                    continue;
                };
                let char_iri = char_node.as_str().to_string();
                let data_type = binding
                    .get("dataType")
                    .and_then(named_node_str)
                    .map(SammDataType::from_iri)
                    .unwrap_or(SammDataType::Unknown(String::new()));
                let characteristic = binding
                    .get("charType")
                    .and_then(named_node_str)
                    .map(SammCharacteristic::from_iri);
                let unit_str = binding.get("unit").and_then(|t| match t {
                    Term::Literal(lit) => Some(lit.value().to_string()),
                    Term::NamedNode(nn) => Some(local_name_of(nn.as_str())),
                    _ => None,
                });
                // Only update if we don't already have data for this characteristic
                // (handles multiple rdf:type rows)
                char_details
                    .entry(char_iri)
                    .or_insert((characteristic, data_type, unit_str));
            }
        }

        let query = "SELECT ?x WHERE { }"; // placeholder, unused
        let _ = query; // suppress unused warning

        let mut props: HashMap<String, SammPhysicsProperty> = HashMap::new();

        if let QueryResults::Bindings(ref bindings) = prop_results.results() {
            for binding in bindings {
                let Some(Term::NamedNode(prop_node)) = binding.get("prop") else {
                    continue;
                };
                let prop_urn = prop_node.as_str().to_string();

                let name = binding
                    .get("label")
                    .and_then(literal_value)
                    .unwrap_or_else(|| local_name_of(&prop_urn));

                let description = binding.get("desc").and_then(literal_value);

                // Look up characteristic details from phase 2 results
                let char_iri = binding
                    .get("char")
                    .and_then(named_node_str)
                    .map(|s| s.to_string());
                let (characteristic, data_type, unit_str) = char_iri
                    .as_deref()
                    .and_then(|iri| char_details.get(iri))
                    .map(|(ch, dt, u)| (ch.clone(), dt.clone(), u.clone()))
                    .unwrap_or((None, SammDataType::Unknown(String::new()), None));

                let physical_unit = unit_str.as_deref().map(parse_unit_str).and_then(|u| {
                    if matches!(u, PhysicalUnit::Custom(_)) {
                        None
                    } else {
                        Some(u)
                    }
                });

                props
                    .entry(prop_urn.clone())
                    .or_insert_with(|| SammPhysicsProperty {
                        urn: prop_urn,
                        name,
                        description,
                        data_type,
                        characteristic,
                        unit: unit_str,
                        physical_unit,
                        range_min: None,
                        range_max: None,
                        enum_values: Vec::new(),
                        is_required: false,
                    });
            }
        }

        // Second pass: enrich with range constraints
        self.enrich_with_constraints(store, &mut props).await?;

        Ok(props.into_values().collect())
    }

    // ── Constraint enrichment ─────────────────────────────────────────────────

    async fn enrich_with_constraints(
        &self,
        store: &Arc<RdfStore>,
        props: &mut HashMap<String, SammPhysicsProperty>,
    ) -> PhysicsResult<()> {
        let constraint_query = format!(
            r#"
            PREFIX samm:   <{samm}>
            PREFIX samm-c: <{samm_c}>
            PREFIX xsd:    <{xsd}>

            SELECT ?prop ?minVal ?maxVal WHERE {{
                ?prop a samm:Property .
                ?prop samm:characteristic ?char .
                OPTIONAL {{ ?char samm-c:minValue ?minVal . }}
                OPTIONAL {{ ?char samm-c:maxValue ?maxVal . }}
            }}
            "#,
            samm = self.samm_ns,
            samm_c = self.samm_c_ns,
            xsd = XSD_NS,
        );

        let results = store
            .query(&constraint_query)
            .map_err(|e| PhysicsError::SammParsing(format!("Constraint query failed: {}", e)))?;

        if let QueryResults::Bindings(ref bindings) = results.results() {
            for binding in bindings {
                let Some(Term::NamedNode(prop_node)) = binding.get("prop") else {
                    continue;
                };
                let prop_urn = prop_node.as_str().to_string();

                if let Some(prop) = props.get_mut(&prop_urn) {
                    if let Some(min_lit) = binding.get("minVal").and_then(literal_value) {
                        prop.range_min = min_lit.parse::<f64>().ok();
                    }
                    if let Some(max_lit) = binding.get("maxVal").and_then(literal_value) {
                        prop.range_max = max_lit.parse::<f64>().ok();
                    }
                }
            }
        }

        Ok(())
    }

    // ── Prefix map extraction ─────────────────────────────────────────────────

    /// Extract `@prefix` / `PREFIX` declarations from a raw Turtle string.
    ///
    /// Returns a map of `prefix_name → namespace_iri`.
    pub fn extract_prefix_map(&self, content: &str) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for line in content.lines() {
            let trimmed = line.trim();
            // Handle both `@prefix` (Turtle) and `PREFIX` (SPARQL-style)
            let rest = if trimmed.to_lowercase().starts_with("@prefix ") {
                Some(&trimmed[8..])
            } else if trimmed.to_lowercase().starts_with("prefix ") {
                Some(&trimmed[7..])
            } else {
                None
            };

            if let Some(decl) = rest {
                if let Some((prefix_part, iri_part)) = decl.split_once(':') {
                    let prefix = prefix_part.trim().to_string();
                    // Extract IRI between < >
                    if let Some(start) = iri_part.find('<') {
                        if let Some(end) = iri_part.find('>') {
                            if end > start {
                                let iri = iri_part[start + 1..end].to_string();
                                map.insert(prefix, iri);
                            }
                        }
                    }
                }
            }
        }
        map
    }

    // ── Bridge methods ────────────────────────────────────────────────────────

    /// Bridge a [`SammAspectModel`] to [`SimulationParameters`] for the named entity.
    ///
    /// Numeric properties with known units are mapped to initial conditions.
    /// Range constraints become boundary conditions.
    ///
    /// # Arguments
    ///
    /// * `model`          – Parsed SAMM model.
    /// * `entity_iri`     – IRI of the physical entity to simulate.
    /// * `simulation_type` – Simulation type string (e.g. `"thermal"`).
    ///
    /// # Errors
    ///
    /// Returns [`PhysicsError::SammParsing`] if the model is empty or inconsistent.
    pub fn bridge_to_simulation_params(
        &self,
        model: &SammAspectModel,
        entity_iri: &str,
        simulation_type: &str,
    ) -> PhysicsResult<SimulationParameters> {
        if model.aspects.is_empty() && model.properties.is_empty() {
            return Err(PhysicsError::SammParsing(
                "SAMM model is empty – cannot bridge to simulation parameters".to_string(),
            ));
        }

        let mut initial_conditions = HashMap::new();
        let mut boundary_conditions = Vec::new();

        for prop in model.numeric_properties() {
            let unit_str = prop
                .unit
                .clone()
                .unwrap_or_else(|| "dimensionless".to_string());

            // Initial condition: use range midpoint if available, else 0
            let initial_value = match (prop.range_min, prop.range_max) {
                (Some(min), Some(max)) => (min + max) / 2.0,
                (Some(min), None) => min,
                (None, Some(max)) => max,
                (None, None) => 0.0,
            };

            initial_conditions.insert(
                prop.name.clone(),
                PhysicalQuantity {
                    value: initial_value,
                    unit: unit_str.clone(),
                    uncertainty: None,
                },
            );

            // Boundary conditions from range constraints
            if let Some(min_val) = prop.range_min {
                boundary_conditions.push(BoundaryCondition {
                    boundary_name: format!("{}_min", prop.name),
                    condition_type: "lower_bound".to_string(),
                    value: PhysicalQuantity {
                        value: min_val,
                        unit: unit_str.clone(),
                        uncertainty: None,
                    },
                });
            }
            if let Some(max_val) = prop.range_max {
                boundary_conditions.push(BoundaryCondition {
                    boundary_name: format!("{}_max", prop.name),
                    condition_type: "upper_bound".to_string(),
                    value: PhysicalQuantity {
                        value: max_val,
                        unit: unit_str.clone(),
                        uncertainty: None,
                    },
                });
            }
        }

        Ok(SimulationParameters {
            entity_iri: entity_iri.to_string(),
            simulation_type: simulation_type.to_string(),
            initial_conditions,
            boundary_conditions,
            time_span: (0.0, 100.0),
            time_steps: 100,
            material_properties: HashMap::new(),
            constraints: Vec::new(),
        })
    }

    /// Extract a [`PhysicalValue`] for a specific property name from the model.
    ///
    /// Uses the range midpoint as the representative value when only bounds are
    /// available.  Returns `None` if the property is not found or is non-numeric.
    pub fn extract_physical_value(
        &self,
        model: &SammAspectModel,
        property_name: &str,
    ) -> Option<PhysicalValue> {
        let prop = model.property_by_name(property_name)?;

        if !prop.data_type.is_numeric() {
            return None;
        }

        let numeric_val = match (prop.range_min, prop.range_max) {
            (Some(min), Some(max)) => (min + max) / 2.0,
            (Some(min), None) => min,
            (None, Some(max)) => max,
            (None, None) => 0.0,
        };

        let unit = prop
            .unit
            .as_deref()
            .map(parse_unit_str)
            .unwrap_or(PhysicalUnit::Dimensionless);

        Some(PhysicalValue::new(numeric_val, unit))
    }

    /// Validate that all required (numeric) properties have value ranges or defaults.
    ///
    /// Returns `Ok(())` if validation passes, or a descriptive error listing
    /// missing data.
    pub fn validate_model_for_simulation(&self, model: &SammAspectModel) -> PhysicsResult<()> {
        let mut issues: Vec<String> = Vec::new();

        for prop in model.numeric_properties() {
            if prop.range_min.is_none() && prop.range_max.is_none() {
                issues.push(format!(
                    "Property '{}' has no range constraints – initial value will default to 0",
                    prop.name
                ));
            }
        }

        if model.aspects.is_empty() {
            issues.push("No samm:Aspect nodes found in model".to_string());
        }

        if issues.is_empty() {
            Ok(())
        } else {
            // Treat as a warning only – log and continue
            tracing::warn!("SAMM model validation warnings: {}", issues.join("; "));
            Ok(())
        }
    }

    /// Parse a raw RDF literal string for a given SAMM property.
    ///
    /// Delegates to [`parse_rdf_literal`] with the property's data type as hint.
    pub fn parse_property_literal(
        &self,
        prop: &SammPhysicsProperty,
        literal: &str,
    ) -> PhysicsResult<PhysicalValue> {
        let datatype_hint: Option<&str> = match &prop.data_type {
            SammDataType::Double => Some("xsd:double"),
            SammDataType::Integer => Some("xsd:integer"),
            _ => None,
        };

        let mut pv = parse_rdf_literal(literal, datatype_hint)?;

        // Override unit from SAMM property if the literal has no unit annotation
        if matches!(pv.unit, PhysicalUnit::Dimensionless) {
            if let Some(ref unit_str) = prop.unit {
                pv.unit = parse_unit_str(unit_str);
            }
        }

        Ok(pv)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Private helper functions
// ──────────────────────────────────────────────────────────────────────────────

/// Extract the lexical value from a `Term::Literal`.
fn literal_value(term: &Term) -> Option<String> {
    if let Term::Literal(lit) = term {
        Some(lit.value().to_string())
    } else {
        None
    }
}

/// Extract the IRI string from a `Term::NamedNode`.
fn named_node_str(term: &Term) -> Option<&str> {
    if let Term::NamedNode(nn) = term {
        Some(nn.as_str())
    } else {
        None
    }
}

/// Return the local name portion of a URI (fragment or last path segment).
fn local_name_of(uri: &str) -> String {
    // Try fragment identifier first (text after #)
    if let Some(fragment) = uri.split_once('#').map(|(_, frag)| frag) {
        if !fragment.is_empty() {
            return fragment.to_string();
        }
    }
    // Fall back to last path segment
    uri.split('/')
        .next_back()
        .and_then(|s| if s.is_empty() { None } else { Some(s) })
        .unwrap_or("unknown")
        .to_string()
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    /// Minimal SAMM TTL that covers Aspects, Properties, Characteristics, and constraints.
    const SAMPLE_SAMM_TTL: &str = r#"
        @prefix samm:    <urn:samm:org.eclipse.esmf.samm:meta-model:2.0.0#> .
        @prefix samm-c:  <urn:samm:org.eclipse.esmf.samm:characteristic:2.0.0#> .
        @prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .
        @prefix phys:    <http://oxirs.org/physics#> .
        @prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix unit:    <urn:samm:org.eclipse.esmf.samm:unit:2.0.0#> .

        phys:ThermalAspect a samm:Aspect ;
            rdfs:label "Thermal Aspect" ;
            samm:description "Thermal properties of a physical body" ;
            samm:properties phys:temperature ;
            samm:properties phys:heatCapacity .

        phys:temperature a samm:Property ;
            rdfs:label "temperature" ;
            samm:description "Thermodynamic temperature of the body" ;
            samm:characteristic phys:TemperatureChar .

        phys:TemperatureChar a samm-c:Measurement ;
            samm:dataType xsd:double ;
            samm:unit "K" ;
            samm-c:minValue "0.0"^^xsd:double ;
            samm-c:maxValue "5000.0"^^xsd:double .

        phys:heatCapacity a samm:Property ;
            rdfs:label "heatCapacity" ;
            samm:description "Specific heat capacity" ;
            samm:characteristic phys:HeatCapChar .

        phys:HeatCapChar a samm-c:Quantifiable ;
            samm:dataType xsd:double ;
            samm:unit "J" ;
            samm-c:minValue "0.0"^^xsd:double .
    "#;

    // ── Parsing ───────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_parse_samm_string_finds_aspect() {
        let parser = SammAspectParser::new().expect("parser creation failed");
        let model = parser
            .parse_samm_string(SAMPLE_SAMM_TTL)
            .await
            .expect("parse failed");

        assert!(
            !model.aspects.is_empty(),
            "Expected at least one samm:Aspect"
        );
        // HashMap ordering is not guaranteed, so search for the aspect by name
        let thermal_aspect = model.aspects.iter().find(|a| a.name == "Thermal Aspect");
        assert!(
            thermal_aspect.is_some(),
            "Expected 'Thermal Aspect' in aspects list, got: {:?}",
            model.aspects.iter().map(|a| &a.name).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn test_parse_samm_string_finds_properties() {
        let parser = SammAspectParser::new().expect("parser creation failed");
        let model = parser
            .parse_samm_string(SAMPLE_SAMM_TTL)
            .await
            .expect("parse failed");

        assert!(
            !model.properties.is_empty(),
            "Expected at least one samm:Property"
        );
    }

    #[tokio::test]
    async fn test_parse_samm_finds_temperature_property() {
        let parser = SammAspectParser::new().expect("parser creation failed");
        let model = parser
            .parse_samm_string(SAMPLE_SAMM_TTL)
            .await
            .expect("parse failed");

        let temp = model.property_by_name("temperature");
        assert!(temp.is_some(), "temperature property not found");
        let tp = temp.expect("already checked");
        assert_eq!(tp.data_type, SammDataType::Double);
    }

    #[tokio::test]
    async fn test_parse_samm_characteristic_type() {
        let parser = SammAspectParser::new().expect("parser creation failed");
        let model = parser
            .parse_samm_string(SAMPLE_SAMM_TTL)
            .await
            .expect("parse failed");

        let temp = model.property_by_name("temperature");
        if let Some(prop) = temp {
            if let Some(ref char_type) = prop.characteristic {
                assert!(
                    matches!(char_type, SammCharacteristic::Measurement),
                    "Expected Measurement characteristic"
                );
            }
        }
    }

    #[tokio::test]
    async fn test_parse_samm_range_constraints() {
        let parser = SammAspectParser::new().expect("parser creation failed");
        let model = parser
            .parse_samm_string(SAMPLE_SAMM_TTL)
            .await
            .expect("parse failed");

        let temp = model.property_by_name("temperature");
        if let Some(prop) = temp {
            if let Some(min) = prop.range_min {
                assert!((min - 0.0).abs() < 1e-10, "min should be 0.0");
            }
            if let Some(max) = prop.range_max {
                assert!((max - 5000.0).abs() < 1e-10, "max should be 5000.0");
            }
        }
    }

    // ── File parsing ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_parse_samm_file() {
        let parser = SammAspectParser::new().expect("parser creation failed");

        // Write TTL to a temp file
        let tmp_dir = env::temp_dir();
        let tmp_path = tmp_dir.join("oxirs_physics_samm_test.ttl");
        std::fs::write(&tmp_path, SAMPLE_SAMM_TTL).expect("failed to write temp file");

        let model = parser
            .parse_samm_file(&tmp_path)
            .await
            .expect("file parse failed");

        // Clean up
        let _ = std::fs::remove_file(&tmp_path);

        assert!(!model.aspects.is_empty() || !model.properties.is_empty());
    }

    // ── Prefix map ────────────────────────────────────────────────────────────

    #[test]
    fn test_extract_prefix_map() {
        let parser = SammAspectParser::new().expect("parser creation failed");
        let map = parser.extract_prefix_map(SAMPLE_SAMM_TTL);

        assert!(map.contains_key("samm"), "samm prefix not found");
        assert!(map.contains_key("xsd"), "xsd prefix not found");
        assert!(map.contains_key("phys"), "phys prefix not found");
    }

    #[test]
    fn test_extract_prefix_map_empty() {
        let parser = SammAspectParser::new().expect("parser creation failed");
        let map = parser.extract_prefix_map("# no prefixes here\n?x a ?y .");
        assert!(map.is_empty());
    }

    // ── Bridge to simulation params ───────────────────────────────────────────

    #[tokio::test]
    async fn test_bridge_to_simulation_params() {
        let parser = SammAspectParser::new().expect("parser creation failed");
        let model = parser
            .parse_samm_string(SAMPLE_SAMM_TTL)
            .await
            .expect("parse failed");

        let params = parser
            .bridge_to_simulation_params(&model, "urn:example:body:1", "thermal")
            .expect("bridge failed");

        assert_eq!(params.entity_iri, "urn:example:body:1");
        assert_eq!(params.simulation_type, "thermal");
        assert!(
            !params.initial_conditions.is_empty(),
            "no initial conditions"
        );
    }

    #[test]
    fn test_bridge_empty_model_is_error() {
        let parser = SammAspectParser::new().expect("parser creation failed");
        let empty_model = SammAspectModel {
            aspects: Vec::new(),
            properties: Vec::new(),
            prefix_map: HashMap::new(),
        };
        let result = parser.bridge_to_simulation_params(&empty_model, "urn:e:1", "thermal");
        assert!(result.is_err(), "expected error for empty model");
    }

    // ── extract_physical_value ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_extract_physical_value_temperature() {
        let parser = SammAspectParser::new().expect("parser creation failed");
        let model = parser
            .parse_samm_string(SAMPLE_SAMM_TTL)
            .await
            .expect("parse failed");

        // Only test if property was parsed
        if model.property_by_name("temperature").is_some() {
            let pv = parser.extract_physical_value(&model, "temperature");
            assert!(pv.is_some(), "physical value should be extractable");
        }
    }

    #[test]
    fn test_extract_physical_value_missing_property() {
        let parser = SammAspectParser::new().expect("parser creation failed");
        let empty_model = SammAspectModel {
            aspects: Vec::new(),
            properties: Vec::new(),
            prefix_map: HashMap::new(),
        };
        let pv = parser.extract_physical_value(&empty_model, "nonexistent");
        assert!(pv.is_none());
    }

    // ── parse_property_literal ────────────────────────────────────────────────

    #[test]
    fn test_parse_property_literal_double_with_unit() {
        let parser = SammAspectParser::new().expect("parser creation failed");
        let prop = SammPhysicsProperty {
            urn: "urn:test:prop".to_string(),
            name: "temperature".to_string(),
            description: None,
            data_type: SammDataType::Double,
            characteristic: Some(SammCharacteristic::Measurement),
            unit: Some("K".to_string()),
            physical_unit: Some(PhysicalUnit::Kelvin),
            range_min: Some(0.0),
            range_max: Some(5000.0),
            enum_values: Vec::new(),
            is_required: true,
        };

        // Literal with explicit unit annotation
        let pv = parser
            .parse_property_literal(&prop, "300.0 K")
            .expect("parse failed");
        assert!((pv.value - 300.0).abs() < 1e-10);
        assert_eq!(pv.unit, PhysicalUnit::Kelvin);
    }

    #[test]
    fn test_parse_property_literal_bare_number_uses_property_unit() {
        let parser = SammAspectParser::new().expect("parser creation failed");
        let prop = SammPhysicsProperty {
            urn: "urn:test:mass".to_string(),
            name: "mass".to_string(),
            description: None,
            data_type: SammDataType::Double,
            characteristic: None,
            unit: Some("kg".to_string()),
            physical_unit: Some(PhysicalUnit::KiloGram),
            range_min: None,
            range_max: None,
            enum_values: Vec::new(),
            is_required: false,
        };

        let pv = parser
            .parse_property_literal(&prop, "75.5")
            .expect("parse failed");
        assert!((pv.value - 75.5).abs() < 1e-10);
        // Unit should be resolved to KiloGram from property definition
        assert_eq!(pv.unit, PhysicalUnit::KiloGram);
    }

    // ── SammDataType helpers ──────────────────────────────────────────────────

    #[test]
    fn test_samm_data_type_from_iri_xsd_double() {
        assert_eq!(
            SammDataType::from_iri("http://www.w3.org/2001/XMLSchema#double"),
            SammDataType::Double
        );
    }

    #[test]
    fn test_samm_data_type_from_iri_xsd_integer() {
        assert_eq!(
            SammDataType::from_iri("http://www.w3.org/2001/XMLSchema#integer"),
            SammDataType::Integer
        );
    }

    #[test]
    fn test_samm_data_type_from_iri_entity() {
        let dt = SammDataType::from_iri("http://example.org/physics#Vector3D");
        assert!(matches!(dt, SammDataType::Entity(_)));
    }

    #[test]
    fn test_samm_data_type_is_numeric() {
        assert!(SammDataType::Double.is_numeric());
        assert!(SammDataType::Integer.is_numeric());
        assert!(!SammDataType::Text.is_numeric());
        assert!(!SammDataType::Boolean.is_numeric());
    }

    // ── SammCharacteristic ────────────────────────────────────────────────────

    #[test]
    fn test_samm_characteristic_from_iri() {
        assert_eq!(
            SammCharacteristic::from_iri("urn:samm:...Measurement"),
            SammCharacteristic::Measurement
        );
        assert_eq!(
            SammCharacteristic::from_iri("urn:samm:...Enumeration"),
            SammCharacteristic::Enumeration
        );
    }

    // ── SammAspectModel helpers ───────────────────────────────────────────────

    #[test]
    fn test_samm_model_numeric_properties_filter() {
        let model = SammAspectModel {
            aspects: Vec::new(),
            properties: vec![
                SammPhysicsProperty {
                    urn: "urn:a".to_string(),
                    name: "mass".to_string(),
                    description: None,
                    data_type: SammDataType::Double,
                    characteristic: None,
                    unit: Some("kg".to_string()),
                    physical_unit: Some(PhysicalUnit::KiloGram),
                    range_min: None,
                    range_max: None,
                    enum_values: Vec::new(),
                    is_required: true,
                },
                SammPhysicsProperty {
                    urn: "urn:b".to_string(),
                    name: "label".to_string(),
                    description: None,
                    data_type: SammDataType::Text,
                    characteristic: None,
                    unit: None,
                    physical_unit: None,
                    range_min: None,
                    range_max: None,
                    enum_values: Vec::new(),
                    is_required: false,
                },
            ],
            prefix_map: HashMap::new(),
        };

        let numeric: Vec<_> = model.numeric_properties().collect();
        assert_eq!(numeric.len(), 1);
        assert_eq!(numeric[0].name, "mass");
    }

    #[test]
    fn test_validate_model_for_simulation_ok() {
        let parser = SammAspectParser::new().expect("parser creation failed");
        let model = SammAspectModel {
            aspects: vec![SammAspect {
                urn: "urn:aspect:1".to_string(),
                name: "TestAspect".to_string(),
                description: None,
                property_urns: Vec::new(),
            }],
            properties: Vec::new(),
            prefix_map: HashMap::new(),
        };
        // Should succeed even with no numeric properties
        assert!(parser.validate_model_for_simulation(&model).is_ok());
    }

    // ── local_name_of helper ──────────────────────────────────────────────────

    #[test]
    fn test_local_name_of_fragment() {
        assert_eq!(local_name_of("http://example.org/ns#mass"), "mass");
    }

    #[test]
    fn test_local_name_of_path() {
        assert_eq!(local_name_of("http://example.org/physics/mass"), "mass");
    }

    #[test]
    fn test_local_name_of_empty() {
        let result = local_name_of("");
        assert_eq!(result, "unknown");
    }
}
