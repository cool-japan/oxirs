//! Dimensional Analysis and Unit Checking
//!
//! Validates dimensional consistency and unit compatibility using fundamental SI dimensions

use crate::error::{PhysicsError, PhysicsResult};
use std::collections::HashMap;

/// Fundamental SI dimensions (L, M, T, I, Θ, N, J)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dimensions {
    /// Length (m)
    pub length: i8,
    /// Mass (kg)
    pub mass: i8,
    /// Time (s)
    pub time: i8,
    /// Electric current (A)
    pub current: i8,
    /// Temperature (K)
    pub temperature: i8,
    /// Amount of substance (mol)
    pub amount: i8,
    /// Luminous intensity (cd)
    pub luminosity: i8,
}

impl Dimensions {
    /// Dimensionless quantity
    pub const fn dimensionless() -> Self {
        Self {
            length: 0,
            mass: 0,
            time: 0,
            current: 0,
            temperature: 0,
            amount: 0,
            luminosity: 0,
        }
    }

    /// Check if dimensions are equal
    pub fn equals(&self, other: &Self) -> bool {
        self.length == other.length
            && self.mass == other.mass
            && self.time == other.time
            && self.current == other.current
            && self.temperature == other.temperature
            && self.amount == other.amount
            && self.luminosity == other.luminosity
    }

    /// Check if dimensionless
    pub fn is_dimensionless(&self) -> bool {
        self.equals(&Self::dimensionless())
    }
}

/// Dimensional Analyzer
pub struct DimensionalAnalyzer {
    /// Known unit dimensions
    unit_dimensions: HashMap<String, Dimensions>,
}

impl DimensionalAnalyzer {
    pub fn new() -> Self {
        let mut analyzer = Self {
            unit_dimensions: HashMap::new(),
        };
        analyzer.initialize_si_units();
        analyzer
    }

    /// Initialize common SI units and derived units
    fn initialize_si_units(&mut self) {
        // Base SI units
        self.register_unit(
            "m",
            Dimensions {
                length: 1,
                ..Dimensions::dimensionless()
            },
        );
        self.register_unit(
            "kg",
            Dimensions {
                mass: 1,
                ..Dimensions::dimensionless()
            },
        );
        self.register_unit(
            "s",
            Dimensions {
                time: 1,
                ..Dimensions::dimensionless()
            },
        );
        self.register_unit(
            "A",
            Dimensions {
                current: 1,
                ..Dimensions::dimensionless()
            },
        );
        self.register_unit(
            "K",
            Dimensions {
                temperature: 1,
                ..Dimensions::dimensionless()
            },
        );
        self.register_unit(
            "mol",
            Dimensions {
                amount: 1,
                ..Dimensions::dimensionless()
            },
        );
        self.register_unit(
            "cd",
            Dimensions {
                luminosity: 1,
                ..Dimensions::dimensionless()
            },
        );

        // Derived units - Mechanical
        self.register_unit(
            "N",
            Dimensions {
                length: 1,
                mass: 1,
                time: -2,
                ..Dimensions::dimensionless()
            },
        ); // Newton
        self.register_unit(
            "Pa",
            Dimensions {
                length: -1,
                mass: 1,
                time: -2,
                ..Dimensions::dimensionless()
            },
        ); // Pascal
        self.register_unit(
            "J",
            Dimensions {
                length: 2,
                mass: 1,
                time: -2,
                ..Dimensions::dimensionless()
            },
        ); // Joule
        self.register_unit(
            "W",
            Dimensions {
                length: 2,
                mass: 1,
                time: -3,
                ..Dimensions::dimensionless()
            },
        ); // Watt

        // Derived units - Electrical
        self.register_unit(
            "V",
            Dimensions {
                length: 2,
                mass: 1,
                time: -3,
                current: -1,
                ..Dimensions::dimensionless()
            },
        ); // Volt
        self.register_unit(
            "Ω",
            Dimensions {
                length: 2,
                mass: 1,
                time: -3,
                current: -2,
                ..Dimensions::dimensionless()
            },
        ); // Ohm
        self.register_unit(
            "C",
            Dimensions {
                time: 1,
                current: 1,
                ..Dimensions::dimensionless()
            },
        ); // Coulomb

        // Thermal units
        self.register_unit(
            "°C",
            Dimensions {
                temperature: 1,
                ..Dimensions::dimensionless()
            },
        );
        self.register_unit(
            "W/(m*K)",
            Dimensions {
                length: 1,
                mass: 1,
                time: -3,
                temperature: -1,
                ..Dimensions::dimensionless()
            },
        ); // Thermal conductivity
        self.register_unit(
            "J/(kg*K)",
            Dimensions {
                length: 2,
                time: -2,
                temperature: -1,
                ..Dimensions::dimensionless()
            },
        ); // Specific heat
        self.register_unit(
            "kg/m^3",
            Dimensions {
                length: -3,
                mass: 1,
                ..Dimensions::dimensionless()
            },
        ); // Density

        // Dimensionless
        self.register_unit("dimensionless", Dimensions::dimensionless());
        self.register_unit("1", Dimensions::dimensionless());
    }

    /// Register a unit with its dimensions
    pub fn register_unit(&mut self, unit: &str, dimensions: Dimensions) {
        self.unit_dimensions.insert(unit.to_string(), dimensions);
    }

    /// Get dimensions for a unit
    pub fn get_dimensions(&self, unit: &str) -> Option<&Dimensions> {
        self.unit_dimensions.get(unit)
    }

    /// Check dimensional consistency between two quantities
    pub fn check_consistency(&self, unit1: &str, unit2: &str) -> PhysicsResult<bool> {
        let dim1 = self
            .get_dimensions(unit1)
            .ok_or_else(|| PhysicsError::UnitConversion(format!("Unknown unit: {}", unit1)))?;

        let dim2 = self
            .get_dimensions(unit2)
            .ok_or_else(|| PhysicsError::UnitConversion(format!("Unknown unit: {}", unit2)))?;

        Ok(dim1.equals(dim2))
    }

    /// Check if a unit is dimensionless
    pub fn is_dimensionless(&self, unit: &str) -> PhysicsResult<bool> {
        let dim = self
            .get_dimensions(unit)
            .ok_or_else(|| PhysicsError::UnitConversion(format!("Unknown unit: {}", unit)))?;

        Ok(dim.is_dimensionless())
    }
}

impl Default for DimensionalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Unit Checker - validates unit compatibility and conversions
pub struct UnitChecker {
    analyzer: DimensionalAnalyzer,
}

impl UnitChecker {
    pub fn new() -> Self {
        Self {
            analyzer: DimensionalAnalyzer::new(),
        }
    }

    /// Validate unit compatibility (same dimensions)
    pub fn check_compatibility(&self, unit1: &str, unit2: &str) -> PhysicsResult<bool> {
        self.analyzer.check_consistency(unit1, unit2)
    }

    /// Validate that a unit exists
    pub fn validate_unit(&self, unit: &str) -> PhysicsResult<()> {
        if self.analyzer.get_dimensions(unit).is_none() {
            return Err(PhysicsError::UnitConversion(format!(
                "Unknown or unsupported unit: {}",
                unit
            )));
        }
        Ok(())
    }

    /// Get canonical form of a unit (if available)
    pub fn canonical_form(&self, unit: &str) -> PhysicsResult<String> {
        // Validate unit exists
        self.validate_unit(unit)?;

        // For now, return the unit as-is
        // Future: Convert to canonical SI form (e.g., N -> kg·m/s²)
        Ok(unit.to_string())
    }

    /// Check if conversion is possible between units
    pub fn can_convert(&self, from: &str, to: &str) -> PhysicsResult<bool> {
        self.check_compatibility(from, to)
    }
}

impl Default for UnitChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensional_consistency() {
        let analyzer = DimensionalAnalyzer::new();

        // Same units should be consistent
        assert!(analyzer.check_consistency("m", "m").unwrap());
        assert!(analyzer.check_consistency("kg", "kg").unwrap());

        // Different units with same dimensions
        // Note: m and km have different prefixes but same dimensions
        // This would require prefix handling (future work)

        // Different dimensions should not be consistent
        assert!(!analyzer.check_consistency("m", "kg").unwrap());
        assert!(!analyzer.check_consistency("J", "Pa").unwrap());
    }

    #[test]
    fn test_derived_units() {
        let analyzer = DimensionalAnalyzer::new();

        // Force (N) = mass * length / time^2
        let newton = analyzer.get_dimensions("N").unwrap();
        assert_eq!(newton.mass, 1);
        assert_eq!(newton.length, 1);
        assert_eq!(newton.time, -2);

        // Energy (J) = mass * length^2 / time^2
        let joule = analyzer.get_dimensions("J").unwrap();
        assert_eq!(joule.mass, 1);
        assert_eq!(joule.length, 2);
        assert_eq!(joule.time, -2);

        // Power (W) = energy / time
        let watt = analyzer.get_dimensions("W").unwrap();
        assert_eq!(watt.mass, 1);
        assert_eq!(watt.length, 2);
        assert_eq!(watt.time, -3);
    }

    #[test]
    fn test_dimensionless() {
        let analyzer = DimensionalAnalyzer::new();

        assert!(analyzer.is_dimensionless("dimensionless").unwrap());
        assert!(analyzer.is_dimensionless("1").unwrap());
        assert!(!analyzer.is_dimensionless("m").unwrap());
    }

    #[test]
    fn test_unit_checker() {
        let checker = UnitChecker::new();

        // Valid units
        assert!(checker.validate_unit("m").is_ok());
        assert!(checker.validate_unit("kg").is_ok());
        assert!(checker.validate_unit("N").is_ok());

        // Invalid unit
        assert!(checker.validate_unit("invalid_unit").is_err());

        // Compatibility
        assert!(checker.check_compatibility("m", "m").unwrap());
        assert!(!checker.check_compatibility("m", "kg").unwrap());
    }

    #[test]
    fn test_thermal_units() {
        let analyzer = DimensionalAnalyzer::new();

        // Thermal conductivity: W/(m·K) = kg·m/s³·K
        let k = analyzer.get_dimensions("W/(m*K)").unwrap();
        assert_eq!(k.length, 1);
        assert_eq!(k.mass, 1);
        assert_eq!(k.time, -3);
        assert_eq!(k.temperature, -1);

        // Specific heat: J/(kg·K) = m²/(s²·K)
        let cp = analyzer.get_dimensions("J/(kg*K)").unwrap();
        assert_eq!(cp.length, 2);
        assert_eq!(cp.mass, 0);
        assert_eq!(cp.time, -2);
        assert_eq!(cp.temperature, -1);
    }
}
