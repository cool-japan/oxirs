//! # Material Property Database
//!
//! Provides a database of common engineering materials with property lookup,
//! temperature-dependent interpolation, and unit conversion. Used by FEM and
//! simulation modules to obtain material properties for structural, thermal,
//! and fluid analyses.
//!
//! ## Features
//!
//! - **Built-in materials**: Common metals, polymers, ceramics, and composites
//! - **Property lookup**: Density, Young's modulus, Poisson's ratio, thermal conductivity, etc.
//! - **Temperature interpolation**: Linear interpolation for temperature-dependent properties
//! - **Unit conversion**: Convert between SI and common engineering unit systems
//! - **Custom materials**: Add user-defined materials with arbitrary properties

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────
// Material property types
// ─────────────────────────────────────────────

/// Categories of materials.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MaterialCategory {
    Metal,
    Polymer,
    Ceramic,
    Composite,
    Fluid,
    Custom,
}

/// Standard material property identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropertyId {
    /// Density in kg/m^3.
    Density,
    /// Young's modulus (elastic modulus) in Pa.
    YoungsModulus,
    /// Poisson's ratio (dimensionless).
    PoissonsRatio,
    /// Yield strength in Pa.
    YieldStrength,
    /// Ultimate tensile strength in Pa.
    UltimateTensileStrength,
    /// Thermal conductivity in W/(m*K).
    ThermalConductivity,
    /// Specific heat capacity in J/(kg*K).
    SpecificHeat,
    /// Coefficient of thermal expansion in 1/K.
    ThermalExpansion,
    /// Melting point in K.
    MeltingPoint,
    /// Electrical resistivity in Ohm*m.
    ElectricalResistivity,
    /// Hardness (Brinell, HB).
    Hardness,
    /// Dynamic viscosity in Pa*s (for fluids).
    DynamicViscosity,
}

impl PropertyId {
    /// SI unit string for this property.
    pub fn unit(&self) -> &'static str {
        match self {
            PropertyId::Density => "kg/m^3",
            PropertyId::YoungsModulus => "Pa",
            PropertyId::PoissonsRatio => "-",
            PropertyId::YieldStrength => "Pa",
            PropertyId::UltimateTensileStrength => "Pa",
            PropertyId::ThermalConductivity => "W/(m*K)",
            PropertyId::SpecificHeat => "J/(kg*K)",
            PropertyId::ThermalExpansion => "1/K",
            PropertyId::MeltingPoint => "K",
            PropertyId::ElectricalResistivity => "Ohm*m",
            PropertyId::Hardness => "HB",
            PropertyId::DynamicViscosity => "Pa*s",
        }
    }
}

/// A property value that may be temperature-dependent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertyValue {
    /// Constant value (temperature-independent).
    Constant(f64),
    /// Temperature-dependent: sorted list of (temperature_K, value) pairs.
    TemperatureDependent(Vec<(f64, f64)>),
}

impl PropertyValue {
    /// Get the value at a given temperature (K).
    /// For constant values, temperature is ignored.
    pub fn at_temperature(&self, temperature_k: f64) -> f64 {
        match self {
            PropertyValue::Constant(v) => *v,
            PropertyValue::TemperatureDependent(points) => {
                if points.is_empty() {
                    return 0.0;
                }
                if points.len() == 1 {
                    return points[0].1;
                }
                // Clamp to range
                if temperature_k <= points[0].0 {
                    return points[0].1;
                }
                if temperature_k >= points[points.len() - 1].0 {
                    return points[points.len() - 1].1;
                }
                // Linear interpolation
                for window in points.windows(2) {
                    let (t0, v0) = window[0];
                    let (t1, v1) = window[1];
                    if temperature_k >= t0 && temperature_k <= t1 {
                        let frac = if (t1 - t0).abs() < f64::EPSILON {
                            0.0
                        } else {
                            (temperature_k - t0) / (t1 - t0)
                        };
                        return v0 + frac * (v1 - v0);
                    }
                }
                points[points.len() - 1].1
            }
        }
    }

    /// Get the constant value (if constant).
    pub fn constant_value(&self) -> Option<f64> {
        match self {
            PropertyValue::Constant(v) => Some(*v),
            PropertyValue::TemperatureDependent(_) => None,
        }
    }
}

/// A material definition with its properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Material {
    /// Unique material name.
    pub name: String,
    /// Material category.
    pub category: MaterialCategory,
    /// Optional description.
    pub description: Option<String>,
    /// Material properties.
    pub properties: HashMap<PropertyId, PropertyValue>,
}

impl Material {
    /// Create a new material.
    pub fn new(name: impl Into<String>, category: MaterialCategory) -> Self {
        Self {
            name: name.into(),
            category,
            description: None,
            properties: HashMap::new(),
        }
    }

    /// Set a constant property.
    pub fn with_property(mut self, id: PropertyId, value: f64) -> Self {
        self.properties.insert(id, PropertyValue::Constant(value));
        self
    }

    /// Set a temperature-dependent property.
    pub fn with_temp_property(mut self, id: PropertyId, points: Vec<(f64, f64)>) -> Self {
        self.properties
            .insert(id, PropertyValue::TemperatureDependent(points));
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Get a property value at a given temperature.
    pub fn get_property(&self, id: PropertyId, temperature_k: f64) -> Option<f64> {
        self.properties
            .get(&id)
            .map(|v| v.at_temperature(temperature_k))
    }

    /// Get a property at room temperature (293.15 K / 20 C).
    pub fn get_property_rt(&self, id: PropertyId) -> Option<f64> {
        self.get_property(id, 293.15)
    }

    /// Check if this material has a given property.
    pub fn has_property(&self, id: PropertyId) -> bool {
        self.properties.contains_key(&id)
    }
}

// ─────────────────────────────────────────────
// Unit conversion
// ─────────────────────────────────────────────

/// Temperature unit for conversions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemperatureUnit {
    Kelvin,
    Celsius,
    Fahrenheit,
}

/// Convert temperature between units.
pub fn convert_temperature(value: f64, from: TemperatureUnit, to: TemperatureUnit) -> f64 {
    // Convert to Kelvin first
    let kelvin = match from {
        TemperatureUnit::Kelvin => value,
        TemperatureUnit::Celsius => value + 273.15,
        TemperatureUnit::Fahrenheit => (value - 32.0) * 5.0 / 9.0 + 273.15,
    };
    // Convert from Kelvin to target
    match to {
        TemperatureUnit::Kelvin => kelvin,
        TemperatureUnit::Celsius => kelvin - 273.15,
        TemperatureUnit::Fahrenheit => (kelvin - 273.15) * 9.0 / 5.0 + 32.0,
    }
}

/// Pressure unit for conversions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureUnit {
    Pascal,
    MegaPascal,
    GigaPascal,
    Psi,
    Bar,
}

/// Convert pressure between units.
pub fn convert_pressure(value: f64, from: PressureUnit, to: PressureUnit) -> f64 {
    let pa = match from {
        PressureUnit::Pascal => value,
        PressureUnit::MegaPascal => value * 1e6,
        PressureUnit::GigaPascal => value * 1e9,
        PressureUnit::Psi => value * 6894.757,
        PressureUnit::Bar => value * 1e5,
    };
    match to {
        PressureUnit::Pascal => pa,
        PressureUnit::MegaPascal => pa / 1e6,
        PressureUnit::GigaPascal => pa / 1e9,
        PressureUnit::Psi => pa / 6894.757,
        PressureUnit::Bar => pa / 1e5,
    }
}

// ─────────────────────────────────────────────
// MaterialDatabase
// ─────────────────────────────────────────────

/// A database of materials with built-in common engineering materials.
pub struct MaterialDatabase {
    materials: HashMap<String, Material>,
}

impl MaterialDatabase {
    /// Create a new empty database.
    pub fn new() -> Self {
        Self {
            materials: HashMap::new(),
        }
    }

    /// Create a database pre-loaded with common engineering materials.
    pub fn with_builtins() -> Self {
        let mut db = Self::new();
        db.load_builtins();
        db
    }

    /// Load built-in materials.
    fn load_builtins(&mut self) {
        // Steel (AISI 1020)
        self.add_material(
            Material::new("Steel_AISI_1020", MaterialCategory::Metal)
                .with_description("Low carbon steel, AISI 1020")
                .with_property(PropertyId::Density, 7870.0)
                .with_property(PropertyId::YoungsModulus, 200e9)
                .with_property(PropertyId::PoissonsRatio, 0.29)
                .with_property(PropertyId::YieldStrength, 350e6)
                .with_property(PropertyId::UltimateTensileStrength, 420e6)
                .with_property(PropertyId::ThermalConductivity, 51.9)
                .with_property(PropertyId::SpecificHeat, 486.0)
                .with_property(PropertyId::ThermalExpansion, 11.7e-6)
                .with_property(PropertyId::MeltingPoint, 1793.0)
                .with_property(PropertyId::Hardness, 111.0),
        );

        // Aluminum 6061-T6
        self.add_material(
            Material::new("Aluminum_6061_T6", MaterialCategory::Metal)
                .with_description("Aluminum alloy 6061-T6")
                .with_property(PropertyId::Density, 2700.0)
                .with_property(PropertyId::YoungsModulus, 68.9e9)
                .with_property(PropertyId::PoissonsRatio, 0.33)
                .with_property(PropertyId::YieldStrength, 276e6)
                .with_property(PropertyId::UltimateTensileStrength, 310e6)
                .with_property(PropertyId::ThermalConductivity, 167.0)
                .with_property(PropertyId::SpecificHeat, 896.0)
                .with_property(PropertyId::ThermalExpansion, 23.6e-6)
                .with_property(PropertyId::MeltingPoint, 855.0 + 273.15),
        );

        // Titanium Ti-6Al-4V
        self.add_material(
            Material::new("Titanium_Ti6Al4V", MaterialCategory::Metal)
                .with_description("Titanium alloy Ti-6Al-4V (Grade 5)")
                .with_property(PropertyId::Density, 4430.0)
                .with_property(PropertyId::YoungsModulus, 113.8e9)
                .with_property(PropertyId::PoissonsRatio, 0.342)
                .with_property(PropertyId::YieldStrength, 880e6)
                .with_property(PropertyId::UltimateTensileStrength, 950e6)
                .with_property(PropertyId::ThermalConductivity, 6.7)
                .with_property(PropertyId::SpecificHeat, 526.3)
                .with_property(PropertyId::ThermalExpansion, 8.6e-6)
                .with_property(PropertyId::MeltingPoint, 1933.0),
        );

        // Copper (pure)
        self.add_material(
            Material::new("Copper_Pure", MaterialCategory::Metal)
                .with_description("Pure copper (OFHC)")
                .with_property(PropertyId::Density, 8960.0)
                .with_property(PropertyId::YoungsModulus, 117e9)
                .with_property(PropertyId::PoissonsRatio, 0.34)
                .with_property(PropertyId::YieldStrength, 70e6)
                .with_property(PropertyId::ThermalConductivity, 401.0)
                .with_property(PropertyId::SpecificHeat, 385.0)
                .with_property(PropertyId::ThermalExpansion, 16.5e-6)
                .with_property(PropertyId::MeltingPoint, 1357.77)
                .with_property(PropertyId::ElectricalResistivity, 1.678e-8),
        );

        // Stainless Steel 316L
        self.add_material(
            Material::new("Stainless_Steel_316L", MaterialCategory::Metal)
                .with_description("Austenitic stainless steel 316L")
                .with_property(PropertyId::Density, 8000.0)
                .with_property(PropertyId::YoungsModulus, 193e9)
                .with_property(PropertyId::PoissonsRatio, 0.27)
                .with_property(PropertyId::YieldStrength, 170e6)
                .with_property(PropertyId::UltimateTensileStrength, 485e6)
                .with_property(PropertyId::ThermalConductivity, 16.3)
                .with_property(PropertyId::SpecificHeat, 500.0)
                .with_property(PropertyId::ThermalExpansion, 16.0e-6)
                .with_property(PropertyId::MeltingPoint, 1673.0),
        );

        // CFRP (Carbon Fiber Reinforced Polymer)
        self.add_material(
            Material::new("CFRP", MaterialCategory::Composite)
                .with_description("Carbon fiber reinforced polymer (unidirectional)")
                .with_property(PropertyId::Density, 1600.0)
                .with_property(PropertyId::YoungsModulus, 181e9)
                .with_property(PropertyId::PoissonsRatio, 0.28)
                .with_property(PropertyId::UltimateTensileStrength, 1500e6)
                .with_property(PropertyId::ThermalConductivity, 7.0)
                .with_property(PropertyId::SpecificHeat, 1130.0)
                .with_property(PropertyId::ThermalExpansion, -0.2e-6),
        );

        // Water (at 20C)
        self.add_material(
            Material::new("Water", MaterialCategory::Fluid)
                .with_description("Pure water at atmospheric pressure")
                .with_property(PropertyId::Density, 998.2)
                .with_property(PropertyId::ThermalConductivity, 0.598)
                .with_property(PropertyId::SpecificHeat, 4182.0)
                .with_property(PropertyId::DynamicViscosity, 1.002e-3)
                .with_temp_property(
                    PropertyId::Density,
                    vec![
                        (273.15, 999.8),
                        (293.15, 998.2),
                        (323.15, 988.1),
                        (353.15, 971.8),
                        (373.15, 958.4),
                    ],
                ),
        );

        // Alumina (Al2O3)
        self.add_material(
            Material::new("Alumina_Al2O3", MaterialCategory::Ceramic)
                .with_description("Aluminum oxide ceramic (99.5%)")
                .with_property(PropertyId::Density, 3950.0)
                .with_property(PropertyId::YoungsModulus, 370e9)
                .with_property(PropertyId::PoissonsRatio, 0.22)
                .with_property(PropertyId::ThermalConductivity, 35.0)
                .with_property(PropertyId::SpecificHeat, 880.0)
                .with_property(PropertyId::MeltingPoint, 2345.0)
                .with_property(PropertyId::Hardness, 1440.0),
        );
    }

    /// Add a material to the database.
    pub fn add_material(&mut self, material: Material) {
        self.materials.insert(material.name.clone(), material);
    }

    /// Get a material by name.
    pub fn get_material(&self, name: &str) -> Option<&Material> {
        self.materials.get(name)
    }

    /// Remove a material.
    pub fn remove_material(&mut self, name: &str) -> bool {
        self.materials.remove(name).is_some()
    }

    /// Number of materials in the database.
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// List all material names.
    pub fn material_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.materials.keys().cloned().collect();
        names.sort();
        names
    }

    /// Find materials by category.
    pub fn find_by_category(&self, category: MaterialCategory) -> Vec<&Material> {
        self.materials
            .values()
            .filter(|m| m.category == category)
            .collect()
    }

    /// Find materials that have a given property within a range.
    pub fn find_by_property_range(
        &self,
        property: PropertyId,
        min_value: f64,
        max_value: f64,
    ) -> Vec<&Material> {
        self.materials
            .values()
            .filter(|m| {
                if let Some(val) = m.get_property_rt(property) {
                    val >= min_value && val <= max_value
                } else {
                    false
                }
            })
            .collect()
    }

    /// Compare a property across all materials.
    pub fn compare_property(&self, property: PropertyId) -> Vec<(String, f64)> {
        let mut result: Vec<(String, f64)> = self
            .materials
            .values()
            .filter_map(|m| m.get_property_rt(property).map(|v| (m.name.clone(), v)))
            .collect();
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }
}

impl Default for MaterialDatabase {
    fn default() -> Self {
        Self::with_builtins()
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> MaterialDatabase {
        MaterialDatabase::with_builtins()
    }

    // ═══ Database construction tests ═════════════════════

    #[test]
    fn test_empty_database() {
        let db = MaterialDatabase::new();
        assert_eq!(db.material_count(), 0);
    }

    #[test]
    fn test_builtins_loaded() {
        let db = db();
        assert!(db.material_count() >= 8);
    }

    #[test]
    fn test_default_has_builtins() {
        let db = MaterialDatabase::default();
        assert!(db.material_count() >= 8);
    }

    // ═══ Material lookup tests ═══════════════════════════

    #[test]
    fn test_get_steel() {
        let db = db();
        let steel = db.get_material("Steel_AISI_1020");
        assert!(steel.is_some());
        let steel = steel.expect("steel should exist");
        assert_eq!(steel.category, MaterialCategory::Metal);
    }

    #[test]
    fn test_get_nonexistent() {
        let db = db();
        assert!(db.get_material("Unobtanium").is_none());
    }

    #[test]
    fn test_material_names_sorted() {
        let db = db();
        let names = db.material_names();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    // ═══ Property lookup tests ═══════════════════════════

    #[test]
    fn test_density_steel() {
        let db = db();
        let steel = db.get_material("Steel_AISI_1020").expect("steel");
        let density = steel.get_property_rt(PropertyId::Density);
        assert!(density.is_some());
        assert!((density.expect("density") - 7870.0).abs() < 1.0);
    }

    #[test]
    fn test_youngs_modulus_aluminum() {
        let db = db();
        let al = db.get_material("Aluminum_6061_T6").expect("aluminum");
        let e = al.get_property_rt(PropertyId::YoungsModulus);
        assert!(e.is_some());
        assert!((e.expect("E") - 68.9e9).abs() < 1e8);
    }

    #[test]
    fn test_poissons_ratio() {
        let db = db();
        let ti = db.get_material("Titanium_Ti6Al4V").expect("titanium");
        let nu = ti.get_property_rt(PropertyId::PoissonsRatio);
        assert!(nu.is_some());
        let nu_val = nu.expect("nu");
        assert!(nu_val > 0.0 && nu_val < 0.5);
    }

    #[test]
    fn test_missing_property() {
        let db = db();
        let steel = db.get_material("Steel_AISI_1020").expect("steel");
        // Steel doesn't have DynamicViscosity
        assert!(steel
            .get_property_rt(PropertyId::DynamicViscosity)
            .is_none());
    }

    #[test]
    fn test_has_property() {
        let db = db();
        let copper = db.get_material("Copper_Pure").expect("copper");
        assert!(copper.has_property(PropertyId::ElectricalResistivity));
        assert!(!copper.has_property(PropertyId::DynamicViscosity));
    }

    // ═══ Temperature interpolation tests ═════════════════

    #[test]
    fn test_constant_value_temperature_independent() {
        let pv = PropertyValue::Constant(100.0);
        assert!((pv.at_temperature(200.0) - 100.0).abs() < 1e-10);
        assert!((pv.at_temperature(500.0) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolation_exact_point() {
        let pv = PropertyValue::TemperatureDependent(vec![
            (273.15, 999.8),
            (293.15, 998.2),
            (373.15, 958.4),
        ]);
        assert!((pv.at_temperature(293.15) - 998.2).abs() < 0.1);
    }

    #[test]
    fn test_interpolation_midpoint() {
        let pv = PropertyValue::TemperatureDependent(vec![(200.0, 100.0), (400.0, 200.0)]);
        assert!((pv.at_temperature(300.0) - 150.0).abs() < 0.1);
    }

    #[test]
    fn test_interpolation_below_range() {
        let pv = PropertyValue::TemperatureDependent(vec![(300.0, 100.0), (400.0, 200.0)]);
        assert!((pv.at_temperature(200.0) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolation_above_range() {
        let pv = PropertyValue::TemperatureDependent(vec![(300.0, 100.0), (400.0, 200.0)]);
        assert!((pv.at_temperature(500.0) - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolation_empty() {
        let pv = PropertyValue::TemperatureDependent(vec![]);
        assert!(pv.at_temperature(300.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolation_single_point() {
        let pv = PropertyValue::TemperatureDependent(vec![(300.0, 42.0)]);
        assert!((pv.at_temperature(300.0) - 42.0).abs() < 1e-10);
    }

    // ═══ Unit conversion tests ═══════════════════════════

    #[test]
    fn test_celsius_to_kelvin() {
        let k = convert_temperature(0.0, TemperatureUnit::Celsius, TemperatureUnit::Kelvin);
        assert!((k - 273.15).abs() < 0.01);
    }

    #[test]
    fn test_kelvin_to_celsius() {
        let c = convert_temperature(373.15, TemperatureUnit::Kelvin, TemperatureUnit::Celsius);
        assert!((c - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_fahrenheit_to_celsius() {
        let c = convert_temperature(212.0, TemperatureUnit::Fahrenheit, TemperatureUnit::Celsius);
        assert!((c - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_celsius_to_fahrenheit() {
        let f = convert_temperature(100.0, TemperatureUnit::Celsius, TemperatureUnit::Fahrenheit);
        assert!((f - 212.0).abs() < 0.1);
    }

    #[test]
    fn test_pressure_mpa_to_pa() {
        let pa = convert_pressure(1.0, PressureUnit::MegaPascal, PressureUnit::Pascal);
        assert!((pa - 1e6).abs() < 1.0);
    }

    #[test]
    fn test_pressure_gpa_to_mpa() {
        let mpa = convert_pressure(1.0, PressureUnit::GigaPascal, PressureUnit::MegaPascal);
        assert!((mpa - 1000.0).abs() < 0.1);
    }

    #[test]
    fn test_pressure_psi_to_bar() {
        let bar = convert_pressure(14.5038, PressureUnit::Psi, PressureUnit::Bar);
        assert!((bar - 1.0).abs() < 0.01);
    }

    // ═══ Category filter tests ═══════════════════════════

    #[test]
    fn test_find_by_category_metal() {
        let db = db();
        let metals = db.find_by_category(MaterialCategory::Metal);
        assert!(metals.len() >= 4);
        assert!(metals.iter().all(|m| m.category == MaterialCategory::Metal));
    }

    #[test]
    fn test_find_by_category_composite() {
        let db = db();
        let composites = db.find_by_category(MaterialCategory::Composite);
        assert!(!composites.is_empty());
    }

    #[test]
    fn test_find_by_category_empty() {
        let db = db();
        let customs = db.find_by_category(MaterialCategory::Custom);
        assert!(customs.is_empty()); // no custom materials in builtins
    }

    // ═══ Property range search tests ═════════════════════

    #[test]
    fn test_find_by_density_range() {
        let db = db();
        let light = db.find_by_property_range(PropertyId::Density, 0.0, 3000.0);
        assert!(!light.is_empty());
        // Aluminum should be included (density ~2700)
        assert!(light.iter().any(|m| m.name.contains("Aluminum")));
    }

    #[test]
    fn test_find_by_density_range_narrow() {
        let db = db();
        let results = db.find_by_property_range(PropertyId::Density, 4400.0, 4500.0);
        // Titanium ~4430
        assert!(results.iter().any(|m| m.name.contains("Titanium")));
    }

    // ═══ Property comparison tests ═══════════════════════

    #[test]
    fn test_compare_density() {
        let db = db();
        let comparison = db.compare_property(PropertyId::Density);
        assert!(!comparison.is_empty());
        // Should be sorted by value
        for window in comparison.windows(2) {
            assert!(window[0].1 <= window[1].1);
        }
    }

    // ═══ Custom material tests ═══════════════════════════

    #[test]
    fn test_add_custom_material() {
        let mut db = MaterialDatabase::new();
        let material = Material::new("Unobtanium", MaterialCategory::Custom)
            .with_property(PropertyId::Density, 1.0)
            .with_description("Fictional material");
        db.add_material(material);
        assert!(db.get_material("Unobtanium").is_some());
    }

    #[test]
    fn test_remove_material() {
        let mut db = db();
        let initial = db.material_count();
        assert!(db.remove_material("Copper_Pure"));
        assert_eq!(db.material_count(), initial - 1);
        assert!(!db.remove_material("Nonexistent"));
    }

    // ═══ PropertyId unit tests ═══════════════════════════

    #[test]
    fn test_property_id_units() {
        assert_eq!(PropertyId::Density.unit(), "kg/m^3");
        assert_eq!(PropertyId::YoungsModulus.unit(), "Pa");
        assert_eq!(PropertyId::PoissonsRatio.unit(), "-");
        assert_eq!(PropertyId::ThermalConductivity.unit(), "W/(m*K)");
    }

    // ═══ PropertyValue constant_value test ═══════════════

    #[test]
    fn test_constant_value() {
        let pv = PropertyValue::Constant(42.0);
        assert_eq!(pv.constant_value(), Some(42.0));
    }

    #[test]
    fn test_temp_dependent_no_constant() {
        let pv = PropertyValue::TemperatureDependent(vec![(300.0, 42.0)]);
        assert_eq!(pv.constant_value(), None);
    }

    // ═══ Material builder tests ══════════════════════════

    #[test]
    fn test_material_builder() {
        let mat = Material::new("Test", MaterialCategory::Metal)
            .with_property(PropertyId::Density, 8000.0)
            .with_property(PropertyId::YoungsModulus, 200e9)
            .with_description("Test material");
        assert_eq!(mat.name, "Test");
        assert_eq!(mat.properties.len(), 2);
        assert!(mat.description.is_some());
    }
}
