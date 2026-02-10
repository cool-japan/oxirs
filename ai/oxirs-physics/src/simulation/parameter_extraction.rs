//! Parameter Extraction from RDF and SAMM
//!
//! Extracts physical simulation parameters from RDF graphs using SPARQL queries.
//! Supports unit conversion, default values, and SAMM Aspect Model integration.

use crate::error::{PhysicsError, PhysicsResult};
use oxirs_core::model::{NamedNode, Term};
use oxirs_core::rdf_store::{QueryResults, RdfStore};
use scirs2_core::units::UnitRegistry;
use scirs2_core::validation::check_finite;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Extracts simulation parameters from RDF graphs and SAMM Aspect Models
pub struct ParameterExtractor {
    /// RDF store for querying
    store: Option<Arc<RdfStore>>,
    /// Unit conversion registry
    unit_registry: UnitRegistry,
    /// Configuration
    config: ExtractionConfig,
}

/// Extraction configuration
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Use fallback defaults for missing properties
    pub use_defaults: bool,
    /// Physics namespace prefix
    pub physics_prefix: String,
    /// Validate extracted values
    pub validate: bool,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            use_defaults: true,
            physics_prefix: "http://oxirs.org/physics#".to_string(),
            validate: true,
        }
    }
}

impl ParameterExtractor {
    /// Create a new parameter extractor without store (for testing)
    pub fn new() -> Self {
        Self {
            store: None,
            unit_registry: UnitRegistry::new(),
            config: ExtractionConfig::default(),
        }
    }

    /// Create a parameter extractor with RDF store
    pub fn with_store(store: Arc<RdfStore>) -> Self {
        Self {
            store: Some(store),
            unit_registry: UnitRegistry::new(),
            config: ExtractionConfig::default(),
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: ExtractionConfig) -> Self {
        self.config = config;
        self
    }

    /// Extract simulation parameters from RDF
    pub async fn extract(
        &self,
        entity_iri: &str,
        simulation_type: &str,
    ) -> PhysicsResult<SimulationParameters> {
        if let Some(ref store) = self.store {
            self.extract_from_rdf(store, entity_iri, simulation_type)
                .await
        } else {
            // Fallback to mock for testing
            self.extract_mock_parameters(entity_iri, simulation_type)
                .await
        }
    }

    /// Extract entity from RDF with all properties
    pub async fn extract_entity(
        &self,
        store: &RdfStore,
        entity_uri: &str,
    ) -> PhysicsResult<PhysicalEntity> {
        let entity_node = NamedNode::new(entity_uri)
            .map_err(|e| PhysicsError::ParameterExtraction(format!("Invalid IRI: {}", e)))?;

        // Query all properties for this entity
        let query = format!(
            r#"
            PREFIX phys: <{prefix}>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

            SELECT ?property ?value ?unit WHERE {{
                <{entity}> ?property ?value .
                OPTIONAL {{ ?value phys:unit ?unit }}
            }}
            "#,
            prefix = self.config.physics_prefix,
            entity = entity_uri
        );

        let results = store
            .query(&query)
            .map_err(|e| PhysicsError::RdfQuery(format!("Query failed: {}", e)))?;

        let mut properties = HashMap::new();

        // Parse query results
        if let QueryResults::Bindings(ref bindings) = results.results() {
            for binding in bindings {
                if let (Some(prop_term), Some(val_term)) = (binding.get("property"), binding.get("value"))
                {
                if let Term::NamedNode(prop_node) = prop_term {
                    let prop_name = prop_node.as_str().split('#').last().or_else(|| prop_node.as_str().split('/').last()).unwrap_or("unknown");

                    let unit = binding
                        .get("unit")
                        .and_then(|t| {
                            if let Term::Literal(lit) = t {
                                Some(lit.value().to_string())
                            } else {
                                None
                            }
                        })
                        .unwrap_or_else(|| "dimensionless".to_string());

                    if let Some(physical_value) = self.parse_value(val_term, &unit)? {
                        properties.insert(prop_name.to_string(), physical_value);
                    }
                }
            }
        }
        }

        // Extract relationships
        let relationships = self.extract_relationships(store, &entity_node).await?;

        Ok(PhysicalEntity {
            uri: entity_node,
            properties,
            relationships,
        })
    }

    /// Extract relationships for an entity
    async fn extract_relationships(
        &self,
        store: &RdfStore,
        entity: &NamedNode,
    ) -> PhysicsResult<Vec<EntityRelationship>> {
        let query = format!(
            r#"
            PREFIX phys: <{prefix}>

            SELECT ?predicate ?target WHERE {{
                <{entity}> ?predicate ?target .
            }}
            "#,
            prefix = self.config.physics_prefix,
            entity = entity.as_str()
        );

        let results = store
            .query(&query)
            .map_err(|e| PhysicsError::RdfQuery(format!("Relationship query failed: {}", e)))?;

        let mut relationships = Vec::new();

        if let QueryResults::Bindings(ref bindings) = results.results() {
            for binding in bindings {
                if let (Some(Term::NamedNode(pred)), Some(Term::NamedNode(target))) =
                    (binding.get("predicate"), binding.get("target"))
                {
                    relationships.push(EntityRelationship {
                        predicate: pred.clone(),
                        target: target.clone(),
                        relationship_type: self.infer_relationship_type(pred.as_str()),
                    });
                }
            }
        }

        Ok(relationships)
    }

    /// Infer relationship type from predicate
    fn infer_relationship_type(&self, predicate: &str) -> RelationType {
        let pred_lower = predicate.to_lowercase();
        if pred_lower.contains("connect") {
            RelationType::Connection
        } else if pred_lower.contains("constrain") {
            RelationType::Constraint
        } else if pred_lower.contains("force") || pred_lower.contains("load") {
            RelationType::Force
        } else {
            RelationType::Other
        }
    }

    /// Parse a term into a physical value
    fn parse_value(&self, term: &Term, unit: &str) -> PhysicsResult<Option<PhysicalValue>> {
        match term {
            Term::Literal(lit) => {
                let value_str = lit.value();

                // Try to parse as f64
                if let Ok(value) = value_str.parse::<f64>() {
                    if self.config.validate {
                        check_finite(value, "value").map_err(|e| {
                            PhysicsError::ParameterExtraction(format!(
                                "Invalid value (not finite): {}",
                                e
                            ))
                        })?;
                    }

                    let datatype = NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                        .map_err(|e| PhysicsError::ParameterExtraction(format!("Invalid datatype IRI: {}", e)))?;

                    Ok(Some(PhysicalValue {
                        value,
                        unit: unit.to_string(),
                        datatype,
                    }))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    /// Query a single property
    pub async fn query_property(
        &self,
        store: &RdfStore,
        entity: &NamedNode,
        property: &str,
    ) -> PhysicsResult<Option<PhysicalValue>> {
        let query = format!(
            r#"
            PREFIX phys: <{prefix}>

            SELECT ?value ?unit WHERE {{
                <{entity}> phys:{property} ?valueNode .
                ?valueNode phys:value ?value .
                OPTIONAL {{ ?valueNode phys:unit ?unit }}
            }}
            "#,
            prefix = self.config.physics_prefix,
            entity = entity.as_str(),
            property = property
        );

        let results = store
            .query(&query)
            .map_err(|e| PhysicsError::RdfQuery(format!("Property query failed: {}", e)))?;

        if let QueryResults::Bindings(ref bindings) = results.results() {
            for binding in bindings {
                if let Some(value_term) = binding.get("value") {
                let unit = binding
                    .get("unit")
                    .and_then(|t| {
                        if let Term::Literal(lit) = t {
                            Some(lit.value().to_string())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| "dimensionless".to_string());

                    return self.parse_value(value_term, &unit);
                }
            }
        }

        Ok(None)
    }

    /// Convert unit
    pub fn convert_unit(&self, value: f64, from: &str, to: &str) -> PhysicsResult<f64> {
        self.unit_registry
            .convert(value, from, to)
            .map_err(|e| PhysicsError::UnitConversion(format!("Conversion failed: {}", e)))
    }

    /// Get fallback default value
    pub fn fallback_to_default(&self, property: &str) -> PhysicalValue {
        match property {
            "mass" => PhysicalValue {
                value: 1.0,
                unit: "kg".to_string(),
                datatype: NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .ok()
                    .unwrap_or_else(|| panic!("Invalid datatype IRI")),
            },
            "temperature" => PhysicalValue {
                value: 293.15,
                unit: "K".to_string(),
                datatype: NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .ok()
                    .unwrap_or_else(|| panic!("Invalid datatype IRI")),
            },
            _ => PhysicalValue {
                value: 0.0,
                unit: "dimensionless".to_string(),
                datatype: NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .ok()
                    .unwrap_or_else(|| panic!("Invalid datatype IRI")),
            },
        }
    }

    /// Extract from RDF store
    async fn extract_from_rdf(
        &self,
        store: &RdfStore,
        entity_iri: &str,
        simulation_type: &str,
    ) -> PhysicsResult<SimulationParameters> {
        let entity = self.extract_entity(store, entity_iri).await?;

        // Convert physical entity to simulation parameters
        let mut initial_conditions = HashMap::new();
        let mut material_properties = HashMap::new();

        for (key, physical_value) in entity.properties.iter() {
            // Determine if this is an initial condition or material property
            if key.contains("initial") || key.contains("position") || key.contains("velocity") {
                initial_conditions.insert(
                    key.clone(),
                    PhysicalQuantity {
                        value: physical_value.value,
                        unit: physical_value.unit.clone(),
                        uncertainty: None,
                    },
                );
            } else if key.contains("modulus")
                || key.contains("density")
                || key.contains("conductivity")
                || key.contains("capacity")
            {
                material_properties.insert(
                    key.clone(),
                    MaterialProperty {
                        name: key.clone(),
                        value: physical_value.value,
                        unit: physical_value.unit.clone(),
                    },
                );
            }
        }

        // Add defaults if needed
        if self.config.use_defaults {
            if initial_conditions.is_empty() {
                match simulation_type {
                    "thermal" => {
                        initial_conditions.insert(
                            "temperature".to_string(),
                            PhysicalQuantity {
                                value: 293.15,
                                unit: "K".to_string(),
                                uncertainty: Some(0.1),
                            },
                        );
                    }
                    "mechanical" => {
                        initial_conditions.insert(
                            "displacement".to_string(),
                            PhysicalQuantity {
                                value: 0.0,
                                unit: "m".to_string(),
                                uncertainty: Some(1e-6),
                            },
                        );
                    }
                    _ => {}
                }
            }
        }

        Ok(SimulationParameters {
            entity_iri: entity_iri.to_string(),
            simulation_type: simulation_type.to_string(),
            initial_conditions,
            boundary_conditions: Vec::new(),
            time_span: (0.0, 100.0),
            time_steps: 100,
            material_properties,
            constraints: Vec::new(),
        })
    }

    /// Mock parameter extraction for testing and development
    async fn extract_mock_parameters(
        &self,
        entity_iri: &str,
        simulation_type: &str,
    ) -> PhysicsResult<SimulationParameters> {
        // Generate reasonable default parameters based on simulation type
        let (initial_conditions, material_properties) = match simulation_type {
            "thermal" => {
                let mut ic = HashMap::new();
                ic.insert(
                    "temperature".to_string(),
                    PhysicalQuantity {
                        value: 293.15, // 20°C in Kelvin
                        unit: "K".to_string(),
                        uncertainty: Some(0.1),
                    },
                );

                let mut mp = HashMap::new();
                mp.insert(
                    "thermal_conductivity".to_string(),
                    MaterialProperty {
                        name: "Thermal Conductivity".to_string(),
                        value: 1.0,
                        unit: "W/(m*K)".to_string(),
                    },
                );
                mp.insert(
                    "specific_heat".to_string(),
                    MaterialProperty {
                        name: "Specific Heat Capacity".to_string(),
                        value: 4186.0,
                        unit: "J/(kg*K)".to_string(),
                    },
                );
                mp.insert(
                    "density".to_string(),
                    MaterialProperty {
                        name: "Density".to_string(),
                        value: 1000.0,
                        unit: "kg/m^3".to_string(),
                    },
                );

                (ic, mp)
            }
            "mechanical" => {
                let mut ic = HashMap::new();
                ic.insert(
                    "displacement".to_string(),
                    PhysicalQuantity {
                        value: 0.0,
                        unit: "m".to_string(),
                        uncertainty: Some(1e-6),
                    },
                );

                let mut mp = HashMap::new();
                mp.insert(
                    "youngs_modulus".to_string(),
                    MaterialProperty {
                        name: "Young's Modulus".to_string(),
                        value: 200e9, // Steel: 200 GPa
                        unit: "Pa".to_string(),
                    },
                );
                mp.insert(
                    "poisson_ratio".to_string(),
                    MaterialProperty {
                        name: "Poisson's Ratio".to_string(),
                        value: 0.3,
                        unit: "dimensionless".to_string(),
                    },
                );

                (ic, mp)
            }
            _ => (HashMap::new(), HashMap::new()),
        };

        Ok(SimulationParameters {
            entity_iri: entity_iri.to_string(),
            simulation_type: simulation_type.to_string(),
            initial_conditions,
            boundary_conditions: Vec::new(),
            time_span: (0.0, 100.0),
            time_steps: 100,
            material_properties,
            constraints: Vec::new(),
        })
    }
}

impl Default for ParameterExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Physical entity extracted from RDF
#[derive(Debug, Clone)]
pub struct PhysicalEntity {
    pub uri: NamedNode,
    pub properties: HashMap<String, PhysicalValue>,
    pub relationships: Vec<EntityRelationship>,
}

/// Physical value with unit
#[derive(Debug, Clone)]
pub struct PhysicalValue {
    pub value: f64,
    pub unit: String,
    pub datatype: NamedNode,
}

/// Entity relationship
#[derive(Debug, Clone)]
pub struct EntityRelationship {
    pub predicate: NamedNode,
    pub target: NamedNode,
    pub relationship_type: RelationType,
}

/// Relationship type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelationType {
    Connection,
    Constraint,
    Force,
    Other,
}

/// Simulation Parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationParameters {
    /// Entity IRI being simulated
    pub entity_iri: String,

    /// Simulation type (e.g., "thermal", "fluid", "structural")
    pub simulation_type: String,

    /// Initial conditions
    pub initial_conditions: HashMap<String, PhysicalQuantity>,

    /// Boundary conditions
    pub boundary_conditions: Vec<BoundaryCondition>,

    /// Time span (start, end)
    pub time_span: (f64, f64),

    /// Number of time steps
    pub time_steps: usize,

    /// Material properties
    pub material_properties: HashMap<String, MaterialProperty>,

    /// Physics constraints
    pub constraints: Vec<String>,
}

/// Physical quantity with value and unit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalQuantity {
    pub value: f64,
    pub unit: String,
    pub uncertainty: Option<f64>,
}

/// Boundary condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryCondition {
    pub boundary_name: String,
    pub condition_type: String,
    pub value: PhysicalQuantity,
}

/// Material property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialProperty {
    pub name: String,
    pub value: f64,
    pub unit: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parameter_extractor_thermal() {
        let extractor = ParameterExtractor::new();

        let params = extractor
            .extract("urn:example:battery:001", "thermal")
            .await
            .expect("Failed to extract parameters");

        assert_eq!(params.entity_iri, "urn:example:battery:001");
        assert_eq!(params.simulation_type, "thermal");
        assert_eq!(params.time_steps, 100);
        assert_eq!(params.time_span, (0.0, 100.0));

        // Check thermal initial conditions
        let temp = params.initial_conditions.get("temperature").expect("Missing temperature");
        assert_eq!(temp.value, 293.15); // 20°C
        assert_eq!(temp.unit, "K");

        // Check material properties
        assert!(params
            .material_properties
            .contains_key("thermal_conductivity"));
        assert!(params.material_properties.contains_key("specific_heat"));
        assert!(params.material_properties.contains_key("density"));
    }

    #[tokio::test]
    async fn test_parameter_extractor_mechanical() {
        let extractor = ParameterExtractor::new();

        let params = extractor
            .extract("urn:example:beam:001", "mechanical")
            .await
            .expect("Failed to extract parameters");

        assert_eq!(params.simulation_type, "mechanical");

        // Check mechanical initial conditions
        let disp = params.initial_conditions.get("displacement").expect("Missing displacement");
        assert_eq!(disp.value, 0.0);
        assert_eq!(disp.unit, "m");

        // Check mechanical properties
        let youngs = params.material_properties.get("youngs_modulus").expect("Missing Young's modulus");
        assert_eq!(youngs.value, 200e9); // Steel
        assert_eq!(youngs.unit, "Pa");

        let poisson = params.material_properties.get("poisson_ratio").expect("Missing Poisson's ratio");
        assert_eq!(poisson.value, 0.3);
        assert_eq!(poisson.unit, "dimensionless");
    }

    #[tokio::test]
    async fn test_unit_conversion() {
        let extractor = ParameterExtractor::new();

        // g → kg
        let kg_value = extractor.convert_unit(1000.0, "g", "kg").expect("Failed to convert g to kg");
        assert!((kg_value - 1.0).abs() < 1e-10);

        // cm → m
        let m_value = extractor.convert_unit(100.0, "cm", "m").expect("Failed to convert cm to m");
        assert!((m_value - 1.0).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_fallback_defaults() {
        let extractor = ParameterExtractor::new();

        let mass_default = extractor.fallback_to_default("mass");
        assert_eq!(mass_default.value, 1.0);
        assert_eq!(mass_default.unit, "kg");

        let temp_default = extractor.fallback_to_default("temperature");
        assert_eq!(temp_default.value, 293.15);
        assert_eq!(temp_default.unit, "K");
    }

    #[test]
    fn test_physical_quantity_serialization() {
        let quantity = PhysicalQuantity {
            value: 300.0,
            unit: "K".to_string(),
            uncertainty: Some(0.5),
        };

        let json = serde_json::to_string(&quantity).expect("Failed to serialize");
        let deserialized: PhysicalQuantity = serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(deserialized.value, 300.0);
        assert_eq!(deserialized.unit, "K");
        assert_eq!(deserialized.uncertainty, Some(0.5));
    }

    #[test]
    fn test_simulation_parameters_serialization() {
        let mut ic = HashMap::new();
        ic.insert(
            "temperature".to_string(),
            PhysicalQuantity {
                value: 293.15,
                unit: "K".to_string(),
                uncertainty: None,
            },
        );

        let params = SimulationParameters {
            entity_iri: "urn:test".to_string(),
            simulation_type: "thermal".to_string(),
            initial_conditions: ic,
            boundary_conditions: Vec::new(),
            time_span: (0.0, 100.0),
            time_steps: 50,
            material_properties: HashMap::new(),
            constraints: Vec::new(),
        };

        let json = serde_json::to_string(&params).expect("Failed to serialize");
        let deserialized: SimulationParameters = serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(deserialized.entity_iri, "urn:test");
        assert_eq!(deserialized.simulation_type, "thermal");
        assert_eq!(deserialized.time_steps, 50);
    }

    #[tokio::test]
    async fn test_parameter_extractor_unknown_type() {
        let extractor = ParameterExtractor::new();

        let params = extractor
            .extract("urn:example:entity", "unknown_type")
            .await
            .expect("Failed to extract parameters");

        // Should return empty parameters for unknown types
        assert!(params.initial_conditions.is_empty());
        assert!(params.material_properties.is_empty());
    }

    #[test]
    fn test_extraction_config() {
        let config = ExtractionConfig {
            use_defaults: false,
            physics_prefix: "http://example.org/phys#".to_string(),
            validate: true,
        };

        assert!(!config.use_defaults);
        assert_eq!(config.physics_prefix, "http://example.org/phys#");
        assert!(config.validate);
    }

    #[test]
    fn test_relationship_type_inference() {
        let extractor = ParameterExtractor::new();

        assert_eq!(
            extractor.infer_relationship_type("http://example.org/connects"),
            RelationType::Connection
        );
        assert_eq!(
            extractor.infer_relationship_type("http://example.org/constrains"),
            RelationType::Constraint
        );
        assert_eq!(
            extractor.infer_relationship_type("http://example.org/applies_force"),
            RelationType::Force
        );
        assert_eq!(
            extractor.infer_relationship_type("http://example.org/other_relation"),
            RelationType::Other
        );
    }
}
