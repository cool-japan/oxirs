//! Simplified SAMM / FEM Bridge Types
//!
//! Provides lightweight data model types (`SammAspect`, `SammProperty`,
//! `SammDataType`) that connect SAMM (Semantic Aspect Meta Model) aspects
//! to FEM physics models without depending on the full SAMM TTL parser.
//!
//! These types are used by:
//! - `oxirs_physics::rdf_extraction` — to convert RDF triples → SAMM aspects
//! - `oxirs_physics::samm` — extended bridge methods on `SammAspect`
//!
//! # Design
//!
//! The types are deliberately minimal: only the fields that FEM solvers
//! actually consume are present.  The full SAMM TTL parser (`SammAspectParser`)
//! in `samm/mod.rs` remains the canonical, feature-complete parser for
//! production use.

use crate::fem::{BoundaryCondition, DofType, FemMaterial, FemMesh, FemSolution, ThermalSolution};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────
// Data Model
// ─────────────────────────────────────────────

/// Simplified SAMM data type used by the FEM bridge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SammDataType {
    /// Floating-point (f64).
    Float,
    /// Integer (i64).
    Integer,
    /// Text string.
    String,
    /// Boolean flag.
    Boolean,
    /// ISO 8601 duration.
    Duration,
}

/// A single typed property in a simplified SAMM aspect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SammProperty {
    /// Property name (camelCase, matching SAMM convention).
    pub name: std::string::String,
    /// XSD-compatible data type.
    pub data_type: SammDataType,
    /// Optional QUDT / SAMM unit string (e.g. `"unit:pascal"`).
    pub unit: Option<std::string::String>,
    /// Optional runtime value as JSON.
    pub value: Option<serde_json::Value>,
}

impl SammProperty {
    /// Try to extract a numeric value as `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match &self.value {
            Some(serde_json::Value::Number(n)) => n.as_f64(),
            Some(serde_json::Value::String(s)) => s.parse::<f64>().ok(),
            _ => None,
        }
    }

    /// Try to extract a boolean value.
    pub fn as_bool(&self) -> Option<bool> {
        match &self.value {
            Some(serde_json::Value::Bool(b)) => Some(*b),
            _ => None,
        }
    }
}

/// Simplified SAMM aspect: a named grouping of typed properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SammAspect {
    /// Full URN of this aspect (e.g. `urn:samm:com.example:1.0.0#SteelProperties`).
    pub urn: std::string::String,
    /// Human-readable aspect name.
    pub name: std::string::String,
    /// Properties belonging to this aspect.
    pub properties: Vec<SammProperty>,
}

impl SammAspect {
    /// Create a new empty aspect.
    pub fn new(urn: impl Into<std::string::String>, name: impl Into<std::string::String>) -> Self {
        Self {
            urn: urn.into(),
            name: name.into(),
            properties: vec![],
        }
    }

    /// Look up a property by exact name.
    pub fn property(&self, name: &str) -> Option<&SammProperty> {
        self.properties.iter().find(|p| p.name == name)
    }

    /// Look up a property by case-insensitive name.
    pub fn property_ci(&self, name: &str) -> Option<&SammProperty> {
        let lower = name.to_lowercase();
        self.properties
            .iter()
            .find(|p| p.name.to_lowercase() == lower)
    }

    /// Return all numeric properties as (name, f64) pairs.
    pub fn numeric_values(&self) -> HashMap<std::string::String, f64> {
        self.properties
            .iter()
            .filter_map(|p| p.as_f64().map(|v| (p.name.clone(), v)))
            .collect()
    }
}

// ─────────────────────────────────────────────
// PhysicsModelBridge
// ─────────────────────────────────────────────

/// Bridge: converts between SAMM aspects and FEM physics types.
pub struct PhysicsModelBridge;

impl PhysicsModelBridge {
    /// Convert a SAMM aspect to a `FemMaterial`.
    ///
    /// Recognised property names (case-insensitive):
    /// - `youngsModulus` / `youngs_modulus` → E (Pa)
    /// - `poissonsRatio` / `poissons_ratio` → ν
    /// - `thermalConductivity` / `thermal_conductivity` → k (W / m·K)
    /// - `density` → ρ (kg / m³)
    pub fn samm_to_fem_material(aspect: &SammAspect) -> Option<FemMaterial> {
        let find = |name: &str| -> Option<f64> {
            aspect
                .property_ci(name)
                .and_then(|p| p.as_f64())
                .or_else(|| {
                    aspect
                        .property_ci(&name.replace('_', ""))
                        .and_then(|p| p.as_f64())
                })
        };

        let youngs = find("youngsModulus").or_else(|| find("youngs_modulus"));
        let poisson = find("poissonsRatio").or_else(|| find("poissons_ratio"));
        let cond = find("thermalConductivity").or_else(|| find("thermal_conductivity"));
        let density = find("density");

        // We need at least one field to consider this a material aspect.
        if youngs.is_none() && poisson.is_none() && cond.is_none() && density.is_none() {
            return None;
        }

        Some(FemMaterial {
            youngs_modulus: youngs.unwrap_or(200e9),
            poissons_ratio: poisson.unwrap_or(0.3),
            thermal_conductivity: cond.unwrap_or(50.0),
            density: density.unwrap_or(7850.0),
        })
    }

    /// Extract boundary conditions from a SAMM aspect.
    ///
    /// Recognised property patterns:
    /// - `displacementBC` → `DofType::Displacement`
    /// - `temperatureBC`  → `DofType::Temperature`
    /// - `pressureBC`     → `DofType::Pressure`
    pub fn samm_to_boundary_conditions(aspect: &SammAspect) -> Vec<BoundaryCondition> {
        let mut bcs = Vec::new();

        if let Some(v) = aspect
            .property_ci("displacementBC")
            .and_then(|p| p.as_f64())
        {
            bcs.push(BoundaryCondition {
                dof: DofType::Displacement,
                value: v,
            });
        }
        if let Some(v) = aspect.property_ci("temperatureBC").and_then(|p| p.as_f64()) {
            bcs.push(BoundaryCondition {
                dof: DofType::Temperature,
                value: v,
            });
        }
        if let Some(v) = aspect.property_ci("pressureBC").and_then(|p| p.as_f64()) {
            bcs.push(BoundaryCondition {
                dof: DofType::Pressure,
                value: v,
            });
        }
        bcs
    }

    /// Convert a `FemSolution` to a SAMM aspect for result exchange.
    pub fn fem_solution_to_samm(solution: &FemSolution, base_urn: &str) -> SammAspect {
        let mut aspect = SammAspect::new(format!("{base_urn}#FemSolution"), "FemSolution");

        aspect.properties.push(SammProperty {
            name: "maxDisplacement".to_string(),
            data_type: SammDataType::Float,
            unit: Some("unit:metre".to_string()),
            value: serde_json::Number::from_f64(solution.max_displacement)
                .map(serde_json::Value::Number),
        });

        aspect.properties.push(SammProperty {
            name: "converged".to_string(),
            data_type: SammDataType::Boolean,
            unit: None,
            value: Some(serde_json::Value::Bool(solution.converged)),
        });

        aspect.properties.push(SammProperty {
            name: "nodeCount".to_string(),
            data_type: SammDataType::Integer,
            unit: None,
            value: Some(serde_json::Value::Number(
                (solution.displacements.len() as i64).into(),
            )),
        });

        aspect.properties.push(SammProperty {
            name: "elementCount".to_string(),
            data_type: SammDataType::Integer,
            unit: None,
            value: Some(serde_json::Value::Number(
                (solution.von_mises_stress.len() as i64).into(),
            )),
        });

        if let Some(max_vm) = solution.von_mises_stress.iter().cloned().reduce(f64::max) {
            aspect.properties.push(SammProperty {
                name: "maxVonMisesStress".to_string(),
                data_type: SammDataType::Float,
                unit: Some("unit:pascal".to_string()),
                value: serde_json::Number::from_f64(max_vm).map(serde_json::Value::Number),
            });
        }

        aspect
    }

    /// Convert a `ThermalSolution` to a SAMM aspect for result exchange.
    pub fn thermal_solution_to_samm(solution: &ThermalSolution, base_urn: &str) -> SammAspect {
        let mut aspect = SammAspect::new(format!("{base_urn}#ThermalSolution"), "ThermalSolution");

        aspect.properties.push(SammProperty {
            name: "maxTemperature".to_string(),
            data_type: SammDataType::Float,
            unit: Some("unit:kelvin".to_string()),
            value: serde_json::Number::from_f64(solution.max_temperature)
                .map(serde_json::Value::Number),
        });

        aspect.properties.push(SammProperty {
            name: "converged".to_string(),
            data_type: SammDataType::Boolean,
            unit: None,
            value: Some(serde_json::Value::Bool(solution.converged)),
        });

        aspect.properties.push(SammProperty {
            name: "nodeCount".to_string(),
            data_type: SammDataType::Integer,
            unit: None,
            value: Some(serde_json::Value::Number(
                (solution.temperatures.len() as i64).into(),
            )),
        });

        aspect
    }

    /// Extract all numeric SAMM properties as digital twin state variables.
    pub fn aspect_to_digital_twin_state(aspect: &SammAspect) -> HashMap<std::string::String, f64> {
        aspect.numeric_values()
    }
}

// ─────────────────────────────────────────────
// SammPhysicsRegistry
// ─────────────────────────────────────────────

/// Registry that associates SAMM aspects with FEM meshes and can trigger
/// static structural simulations on demand.
pub struct SammPhysicsRegistry {
    aspects: HashMap<std::string::String, SammAspect>,
    models: HashMap<std::string::String, FemMesh>,
}

impl Default for SammPhysicsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SammPhysicsRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            aspects: HashMap::new(),
            models: HashMap::new(),
        }
    }

    /// Register a SAMM aspect (keyed by its URN).
    pub fn register_aspect(&mut self, aspect: SammAspect) {
        self.aspects.insert(aspect.urn.clone(), aspect);
    }

    /// Register a FEM mesh under an arbitrary URN key.
    pub fn register_model(&mut self, urn: &str, mesh: FemMesh) {
        self.models.insert(urn.to_string(), mesh);
    }

    /// Run a static FEM solve for the mesh registered under `urn`.
    ///
    /// Returns `None` if no mesh is registered for that URN.
    /// Loads are extracted from the SAMM aspect registered under the same URN
    /// (if any).
    pub fn simulate(&self, urn: &str) -> Option<FemSolution> {
        let mesh = self.models.get(urn)?;
        use crate::fem::{FemSolver, NodalLoad};

        // Optionally derive a point load from the aspect
        let loads: Vec<NodalLoad> = self
            .aspects
            .get(urn)
            .and_then(|asp| {
                let fx = asp
                    .property_ci("loadFx")
                    .and_then(|p| p.as_f64())
                    .unwrap_or(0.0);
                let fy = asp
                    .property_ci("loadFy")
                    .and_then(|p| p.as_f64())
                    .unwrap_or(0.0);
                if fx == 0.0 && fy == 0.0 {
                    None
                } else {
                    // Apply to the last node (free end convention)
                    let last_node = mesh.node_count().saturating_sub(1);
                    Some(vec![NodalLoad {
                        node_id: last_node,
                        fx,
                        fy,
                    }])
                }
            })
            .unwrap_or_default();

        let solver = FemSolver::new();
        Some(solver.solve_static(mesh, &loads))
    }

    /// List all registered aspect URNs.
    pub fn list_aspects(&self) -> Vec<&str> {
        self.aspects.keys().map(|s| s.as_str()).collect()
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fem::{ElementType, FemMesh, FemSolution, ThermalSolution};

    fn steel_aspect() -> SammAspect {
        SammAspect {
            urn: "urn:samm:com.example:1.0.0#Steel".to_string(),
            name: "Steel".to_string(),
            properties: vec![
                SammProperty {
                    name: "youngsModulus".to_string(),
                    data_type: SammDataType::Float,
                    unit: Some("unit:pascal".to_string()),
                    value: serde_json::Number::from_f64(200e9).map(serde_json::Value::Number),
                },
                SammProperty {
                    name: "poissonsRatio".to_string(),
                    data_type: SammDataType::Float,
                    unit: None,
                    value: serde_json::Number::from_f64(0.3).map(serde_json::Value::Number),
                },
                SammProperty {
                    name: "thermalConductivity".to_string(),
                    data_type: SammDataType::Float,
                    unit: Some("unit:watt-per-metre-kelvin".to_string()),
                    value: serde_json::Number::from_f64(50.0).map(serde_json::Value::Number),
                },
                SammProperty {
                    name: "density".to_string(),
                    data_type: SammDataType::Float,
                    unit: Some("unit:kilogram-per-cubic-metre".to_string()),
                    value: serde_json::Number::from_f64(7850.0).map(serde_json::Value::Number),
                },
            ],
        }
    }

    fn make_simple_mesh() -> FemMesh {
        let mut mesh = FemMesh::new();
        let n0 = mesh.add_node(0.0, 0.0);
        let n1 = mesh.add_node(1.0, 0.0);
        use crate::fem::DofType;
        mesh.set_boundary_condition(n0, DofType::Displacement, 0.0);
        mesh.add_element(
            vec![n0, n1],
            crate::fem::FemMaterial::default(),
            ElementType::Bar1D,
        );
        mesh
    }

    // ────────────────────────────────────────
    // SammAspect helpers
    // ────────────────────────────────────────

    #[test]
    fn test_samm_aspect_new() {
        let asp = SammAspect::new("urn:test", "TestAspect");
        assert_eq!(asp.urn, "urn:test");
        assert_eq!(asp.name, "TestAspect");
        assert!(asp.properties.is_empty());
    }

    #[test]
    fn test_samm_aspect_property_lookup() {
        let asp = steel_aspect();
        assert!(asp.property("youngsModulus").is_some());
        assert!(asp.property("nonexistent").is_none());
    }

    #[test]
    fn test_samm_aspect_property_ci_lookup() {
        let asp = steel_aspect();
        assert!(asp.property_ci("YOUNGSMODULUS").is_some());
        assert!(asp.property_ci("YOUNGSmodulus").is_some());
    }

    #[test]
    fn test_samm_aspect_numeric_values() {
        let asp = steel_aspect();
        let vals = asp.numeric_values();
        assert!(vals.contains_key("youngsModulus"));
        assert!((vals["youngsModulus"] - 200e9).abs() / 200e9 < 0.01);
    }

    #[test]
    fn test_samm_property_as_f64() {
        let p = SammProperty {
            name: "x".to_string(),
            data_type: SammDataType::Float,
            unit: None,
            value: serde_json::Number::from_f64(3.125).map(serde_json::Value::Number),
        };
        assert!((p.as_f64().expect("should parse") - 3.125).abs() < 1e-10);
    }

    #[test]
    fn test_samm_property_as_bool() {
        let p = SammProperty {
            name: "flag".to_string(),
            data_type: SammDataType::Boolean,
            unit: None,
            value: Some(serde_json::Value::Bool(true)),
        };
        assert_eq!(p.as_bool(), Some(true));
    }

    // ────────────────────────────────────────
    // PhysicsModelBridge
    // ────────────────────────────────────────

    #[test]
    fn test_samm_to_fem_material_full() {
        let asp = steel_aspect();
        let mat = PhysicsModelBridge::samm_to_fem_material(&asp).expect("Should produce material");
        assert!((mat.youngs_modulus - 200e9).abs() / 200e9 < 0.01);
        assert!((mat.poissons_ratio - 0.3).abs() < 1e-6);
        assert!((mat.thermal_conductivity - 50.0).abs() < 1e-6);
        assert!((mat.density - 7850.0).abs() < 1.0);
    }

    #[test]
    fn test_samm_to_fem_material_defaults_for_missing() {
        // Only youngsModulus set
        let asp = SammAspect {
            urn: "urn:test".to_string(),
            name: "Test".to_string(),
            properties: vec![SammProperty {
                name: "youngsModulus".to_string(),
                data_type: SammDataType::Float,
                unit: None,
                value: serde_json::Number::from_f64(70e9).map(serde_json::Value::Number),
            }],
        };
        let mat = PhysicsModelBridge::samm_to_fem_material(&asp).expect("should produce material");
        assert!((mat.youngs_modulus - 70e9).abs() / 70e9 < 0.01);
        assert!((mat.poissons_ratio - 0.3).abs() < 1e-6); // default
    }

    #[test]
    fn test_samm_to_fem_material_none_when_empty() {
        let asp = SammAspect::new("urn:test", "Empty");
        assert!(PhysicsModelBridge::samm_to_fem_material(&asp).is_none());
    }

    #[test]
    fn test_samm_to_boundary_conditions_displacement() {
        let asp = SammAspect {
            urn: "urn:test".to_string(),
            name: "BC".to_string(),
            properties: vec![SammProperty {
                name: "displacementBC".to_string(),
                data_type: SammDataType::Float,
                unit: None,
                value: serde_json::Number::from_f64(0.0).map(serde_json::Value::Number),
            }],
        };
        let bcs = PhysicsModelBridge::samm_to_boundary_conditions(&asp);
        assert_eq!(bcs.len(), 1);
        assert_eq!(bcs[0].dof, DofType::Displacement);
    }

    #[test]
    fn test_samm_to_boundary_conditions_temperature() {
        let asp = SammAspect {
            urn: "urn:test".to_string(),
            name: "BC".to_string(),
            properties: vec![SammProperty {
                name: "temperatureBC".to_string(),
                data_type: SammDataType::Float,
                unit: None,
                value: serde_json::Number::from_f64(300.0).map(serde_json::Value::Number),
            }],
        };
        let bcs = PhysicsModelBridge::samm_to_boundary_conditions(&asp);
        assert_eq!(bcs.len(), 1);
        assert_eq!(bcs[0].dof, DofType::Temperature);
        assert!((bcs[0].value - 300.0).abs() < 1e-6);
    }

    #[test]
    fn test_fem_solution_to_samm() {
        let sol = FemSolution {
            displacements: vec![(0.0, 0.0), (1e-4, 0.0)],
            von_mises_stress: vec![1_000_000.0],
            max_displacement: 1e-4,
            converged: true,
        };
        let samm = PhysicsModelBridge::fem_solution_to_samm(&sol, "urn:test");
        assert!(samm.urn.contains("FemSolution"));
        assert!(samm.property("converged").is_some());
        assert_eq!(
            samm.property("converged").and_then(|p| p.as_bool()),
            Some(true)
        );
        assert!(samm.property("maxDisplacement").is_some());
        assert!(samm.property("maxVonMisesStress").is_some());
    }

    #[test]
    fn test_thermal_solution_to_samm() {
        let sol = ThermalSolution {
            temperatures: vec![300.0, 400.0],
            heat_flux: vec![(100.0, 0.0)],
            max_temperature: 400.0,
            converged: true,
        };
        let samm = PhysicsModelBridge::thermal_solution_to_samm(&sol, "urn:test");
        assert!(samm.urn.contains("ThermalSolution"));
        assert!(samm.property("maxTemperature").is_some());
        let max_t = samm
            .property("maxTemperature")
            .and_then(|p| p.as_f64())
            .expect("should have value");
        assert!((max_t - 400.0).abs() < 1.0);
    }

    #[test]
    fn test_aspect_to_digital_twin_state() {
        let asp = steel_aspect();
        let state = PhysicsModelBridge::aspect_to_digital_twin_state(&asp);
        assert!(state.contains_key("youngsModulus"));
        assert!(state.contains_key("density"));
    }

    // ────────────────────────────────────────
    // SammPhysicsRegistry
    // ────────────────────────────────────────

    #[test]
    fn test_registry_register_and_list_aspects() {
        let mut reg = SammPhysicsRegistry::new();
        reg.register_aspect(SammAspect::new("urn:asp1", "Asp1"));
        reg.register_aspect(SammAspect::new("urn:asp2", "Asp2"));
        let keys = reg.list_aspects();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"urn:asp1"));
        assert!(keys.contains(&"urn:asp2"));
    }

    #[test]
    fn test_registry_simulate_missing_urn_returns_none() {
        let reg = SammPhysicsRegistry::new();
        assert!(reg.simulate("urn:missing").is_none());
    }

    #[test]
    fn test_registry_simulate_registered_mesh() {
        let mut reg = SammPhysicsRegistry::new();
        reg.register_model("urn:model1", make_simple_mesh());
        let sol = reg.simulate("urn:model1").expect("should produce solution");
        assert!(sol.converged);
    }

    #[test]
    fn test_registry_simulate_with_aspect_loads() {
        let mut reg = SammPhysicsRegistry::new();
        reg.register_model("urn:model2", make_simple_mesh());
        let asp = SammAspect {
            urn: "urn:model2".to_string(),
            name: "LoadSpec".to_string(),
            properties: vec![SammProperty {
                name: "loadFx".to_string(),
                data_type: SammDataType::Float,
                unit: Some("unit:newton".to_string()),
                value: serde_json::Number::from_f64(10_000.0).map(serde_json::Value::Number),
            }],
        };
        reg.register_aspect(asp);
        let sol = reg.simulate("urn:model2").expect("solution");
        assert!(sol.converged);
        assert!(sol.max_displacement > 0.0);
    }

    #[test]
    fn test_registry_default_is_empty() {
        let reg = SammPhysicsRegistry::default();
        assert!(reg.list_aspects().is_empty());
    }
}
