//! SAMM/BAMM Unit Catalog with SI and derived units.
//!
//! Provides a catalog of physical units with conversion support via the SI unit system.
//! Conversion formula: `si_value = (value * from.si_factor) + from.si_offset`
//!                     `result  = (si_value - to.si_offset) / to.si_factor`

use std::collections::HashMap;
use std::fmt;

/// Broad category for a unit.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UnitCategory {
    /// SI base unit of length (metre)
    Length,
    /// SI base unit of mass (kilogram)
    Mass,
    /// SI base unit of time (second)
    Time,
    /// SI base unit of thermodynamic temperature (kelvin)
    Temperature,
    /// SI base unit of electric current (ampere)
    ElectricCurrent,
    /// SI base unit of amount of substance (mole)
    Amount,
    /// SI base unit of luminous intensity (candela)
    LuminousIntensity,
    /// Derived unit (combination of base units)
    Derived,
    /// Dimensionless unit (ratio, percentage, etc.)
    Dimensionless,
}

impl fmt::Display for UnitCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            UnitCategory::Length => "Length",
            UnitCategory::Mass => "Mass",
            UnitCategory::Time => "Time",
            UnitCategory::Temperature => "Temperature",
            UnitCategory::ElectricCurrent => "ElectricCurrent",
            UnitCategory::Amount => "Amount",
            UnitCategory::LuminousIntensity => "LuminousIntensity",
            UnitCategory::Derived => "Derived",
            UnitCategory::Dimensionless => "Dimensionless",
        };
        write!(f, "{s}")
    }
}

/// A physical unit definition.
#[derive(Debug, Clone)]
pub struct Unit {
    /// Full name, e.g., "metre"
    pub name: String,
    /// Abbreviated symbol, e.g., "m"
    pub symbol: String,
    /// SAMM URN, e.g., "urn:samm:org.eclipse.esmf.samm:unit:metre"
    pub urn: String,
    /// Dimensional category
    pub category: UnitCategory,
    /// Factor to convert `symbol` → SI base unit: `si_value = value * si_factor + si_offset`
    pub si_factor: f64,
    /// Offset for affine conversions (e.g., temperature): default 0.0
    pub si_offset: f64,
    /// Human-readable description
    pub description: String,
}

impl Unit {
    fn new(
        name: &str,
        symbol: &str,
        category: UnitCategory,
        si_factor: f64,
        si_offset: f64,
        description: &str,
    ) -> Self {
        let urn = format!(
            "urn:samm:org.eclipse.esmf.samm:unit:{}",
            name.replace(' ', "_")
        );
        Self {
            name: name.to_string(),
            symbol: symbol.to_string(),
            urn,
            category,
            si_factor,
            si_offset,
            description: description.to_string(),
        }
    }
}

/// Error type for unit catalog operations.
#[derive(Debug, Clone, PartialEq)]
pub enum UnitError {
    /// The requested unit symbol was not found.
    UnitNotFound(String),
    /// The two units belong to different categories and cannot be converted.
    IncompatibleUnits {
        /// Source unit symbol
        from: String,
        /// Target unit symbol
        to: String,
    },
}

impl fmt::Display for UnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnitError::UnitNotFound(sym) => write!(f, "Unit not found: '{sym}'"),
            UnitError::IncompatibleUnits { from, to } => {
                write!(
                    f,
                    "Cannot convert '{from}' to '{to}': incompatible categories"
                )
            }
        }
    }
}

impl std::error::Error for UnitError {}

/// A catalog of physical units with SI-based conversion.
pub struct UnitCatalog {
    /// Units indexed by symbol
    by_symbol: HashMap<String, Unit>,
    /// Units indexed by URN
    by_urn: HashMap<String, String>, // URN → symbol
}

impl UnitCatalog {
    /// Create a new catalog pre-populated with 30+ standard units.
    pub fn new() -> Self {
        let mut catalog = Self {
            by_symbol: HashMap::new(),
            by_urn: HashMap::new(),
        };
        catalog.populate();
        catalog
    }

    fn insert(&mut self, unit: Unit) {
        let urn = unit.urn.clone();
        let symbol = unit.symbol.clone();
        self.by_urn.insert(urn, symbol.clone());
        self.by_symbol.insert(symbol, unit);
    }

    fn populate(&mut self) {
        // --- Length (SI base: metre) ---
        self.insert(Unit::new(
            "metre",
            "m",
            UnitCategory::Length,
            1.0,
            0.0,
            "SI base unit of length",
        ));
        self.insert(Unit::new(
            "kilometre",
            "km",
            UnitCategory::Length,
            1000.0,
            0.0,
            "1000 metres",
        ));
        self.insert(Unit::new(
            "centimetre",
            "cm",
            UnitCategory::Length,
            0.01,
            0.0,
            "0.01 metres",
        ));
        self.insert(Unit::new(
            "millimetre",
            "mm",
            UnitCategory::Length,
            0.001,
            0.0,
            "0.001 metres",
        ));
        self.insert(Unit::new(
            "mile",
            "mi",
            UnitCategory::Length,
            1609.344,
            0.0,
            "International mile",
        ));
        self.insert(Unit::new(
            "foot",
            "ft",
            UnitCategory::Length,
            0.3048,
            0.0,
            "International foot",
        ));
        self.insert(Unit::new(
            "inch",
            "in",
            UnitCategory::Length,
            0.0254,
            0.0,
            "International inch",
        ));
        self.insert(Unit::new(
            "yard",
            "yd",
            UnitCategory::Length,
            0.9144,
            0.0,
            "International yard",
        ));

        // --- Mass (SI base: kilogram) ---
        self.insert(Unit::new(
            "kilogram",
            "kg",
            UnitCategory::Mass,
            1.0,
            0.0,
            "SI base unit of mass",
        ));
        self.insert(Unit::new(
            "gram",
            "g",
            UnitCategory::Mass,
            0.001,
            0.0,
            "0.001 kilograms",
        ));
        self.insert(Unit::new(
            "tonne",
            "t",
            UnitCategory::Mass,
            1000.0,
            0.0,
            "Metric tonne",
        ));
        self.insert(Unit::new(
            "pound",
            "lb",
            UnitCategory::Mass,
            0.453_592,
            0.0,
            "Avoirdupois pound",
        ));
        self.insert(Unit::new(
            "ounce",
            "oz",
            UnitCategory::Mass,
            0.028_349_5,
            0.0,
            "Avoirdupois ounce",
        ));

        // --- Time (SI base: second) ---
        self.insert(Unit::new(
            "second",
            "s",
            UnitCategory::Time,
            1.0,
            0.0,
            "SI base unit of time",
        ));
        self.insert(Unit::new(
            "minute",
            "min",
            UnitCategory::Time,
            60.0,
            0.0,
            "60 seconds",
        ));
        self.insert(Unit::new(
            "hour",
            "h",
            UnitCategory::Time,
            3600.0,
            0.0,
            "3600 seconds",
        ));
        self.insert(Unit::new(
            "day",
            "d",
            UnitCategory::Time,
            86400.0,
            0.0,
            "86400 seconds",
        ));
        self.insert(Unit::new(
            "millisecond",
            "ms",
            UnitCategory::Time,
            0.001,
            0.0,
            "0.001 seconds",
        ));

        // --- Temperature (SI base: kelvin) ---
        self.insert(Unit::new(
            "kelvin",
            "K",
            UnitCategory::Temperature,
            1.0,
            0.0,
            "SI base unit of temperature",
        ));
        // Celsius: K = °C + 273.15  →  si_factor=1.0, si_offset=273.15
        self.insert(Unit::new(
            "celsius",
            "°C",
            UnitCategory::Temperature,
            1.0,
            273.15,
            "Degrees Celsius",
        ));
        // Fahrenheit: K = (°F + 459.67) * 5/9  →  si_factor=5/9, si_offset = 459.67*5/9 = 255.3722...
        self.insert(Unit::new(
            "fahrenheit",
            "°F",
            UnitCategory::Temperature,
            5.0 / 9.0,
            255.372_222_222_222,
            "Degrees Fahrenheit",
        ));

        // --- Electric Current (SI base: ampere) ---
        self.insert(Unit::new(
            "ampere",
            "A",
            UnitCategory::ElectricCurrent,
            1.0,
            0.0,
            "SI base unit of electric current",
        ));
        self.insert(Unit::new(
            "milliampere",
            "mA",
            UnitCategory::ElectricCurrent,
            0.001,
            0.0,
            "0.001 amperes",
        ));

        // --- Derived SI units ---
        self.insert(Unit::new(
            "newton",
            "N",
            UnitCategory::Derived,
            1.0,
            0.0,
            "SI unit of force: kg·m/s²",
        ));
        self.insert(Unit::new(
            "joule",
            "J",
            UnitCategory::Derived,
            1.0,
            0.0,
            "SI unit of energy: kg·m²/s²",
        ));
        self.insert(Unit::new(
            "watt",
            "W",
            UnitCategory::Derived,
            1.0,
            0.0,
            "SI unit of power: J/s",
        ));
        self.insert(Unit::new(
            "pascal",
            "Pa",
            UnitCategory::Derived,
            1.0,
            0.0,
            "SI unit of pressure: N/m²",
        ));
        self.insert(Unit::new(
            "hertz",
            "Hz",
            UnitCategory::Derived,
            1.0,
            0.0,
            "SI unit of frequency: 1/s",
        ));
        self.insert(Unit::new(
            "volt",
            "V",
            UnitCategory::Derived,
            1.0,
            0.0,
            "SI unit of voltage: W/A",
        ));
        self.insert(Unit::new(
            "ohm",
            "Ω",
            UnitCategory::Derived,
            1.0,
            0.0,
            "SI unit of resistance: V/A",
        ));

        // --- Dimensionless ---
        self.insert(Unit::new(
            "percent",
            "%",
            UnitCategory::Dimensionless,
            0.01,
            0.0,
            "Parts per hundred",
        ));
        self.insert(Unit::new(
            "radian",
            "rad",
            UnitCategory::Dimensionless,
            1.0,
            0.0,
            "SI unit of plane angle",
        ));
    }

    /// Look up a unit by its symbol.
    pub fn get_by_symbol(&self, sym: &str) -> Option<&Unit> {
        self.by_symbol.get(sym)
    }

    /// Look up a unit by its SAMM URN.
    pub fn get_by_urn(&self, urn: &str) -> Option<&Unit> {
        let symbol = self.by_urn.get(urn)?;
        self.by_symbol.get(symbol)
    }

    /// Return all units of a given category, sorted by name.
    pub fn by_category(&self, cat: &UnitCategory) -> Vec<&Unit> {
        let mut units: Vec<&Unit> = self
            .by_symbol
            .values()
            .filter(|u| &u.category == cat)
            .collect();
        units.sort_by(|a, b| a.name.cmp(&b.name));
        units
    }

    /// Convert `value` from unit `from_symbol` to unit `to_symbol`.
    ///
    /// Conversion via SI:
    ///   `si = value * from.si_factor + from.si_offset`
    ///   `result = (si - to.si_offset) / to.si_factor`
    pub fn convert(
        &self,
        value: f64,
        from_symbol: &str,
        to_symbol: &str,
    ) -> Result<f64, UnitError> {
        let from = self
            .get_by_symbol(from_symbol)
            .ok_or_else(|| UnitError::UnitNotFound(from_symbol.to_string()))?;
        let to = self
            .get_by_symbol(to_symbol)
            .ok_or_else(|| UnitError::UnitNotFound(to_symbol.to_string()))?;

        if from.category != to.category {
            return Err(UnitError::IncompatibleUnits {
                from: from_symbol.to_string(),
                to: to_symbol.to_string(),
            });
        }

        let si_value = value * from.si_factor + from.si_offset;
        let result = (si_value - to.si_offset) / to.si_factor;
        Ok(result)
    }

    /// Return true if the two units can be converted (same category).
    pub fn is_convertible(&self, from: &str, to: &str) -> bool {
        match (self.get_by_symbol(from), self.get_by_symbol(to)) {
            (Some(a), Some(b)) => a.category == b.category,
            _ => false,
        }
    }

    /// Return a sorted list of all known unit symbols.
    pub fn all_symbols(&self) -> Vec<String> {
        let mut syms: Vec<String> = self.by_symbol.keys().cloned().collect();
        syms.sort();
        syms
    }
}

impl Default for UnitCatalog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog() -> UnitCatalog {
        UnitCatalog::new()
    }

    // ------ catalog completeness ------

    #[test]
    fn test_catalog_has_at_least_30_units() {
        let c = catalog();
        assert!(c.all_symbols().len() >= 30, "Expected ≥30 units");
    }

    #[test]
    fn test_all_symbols_sorted() {
        let c = catalog();
        let syms = c.all_symbols();
        let mut sorted = syms.clone();
        sorted.sort();
        assert_eq!(syms, sorted);
    }

    // ------ get_by_symbol ------

    #[test]
    fn test_get_metre() {
        let c = catalog();
        let u = c.get_by_symbol("m").expect("metre should exist");
        assert_eq!(u.name, "metre");
        assert_eq!(u.category, UnitCategory::Length);
        assert!((u.si_factor - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_get_kilometre() {
        let c = catalog();
        let u = c.get_by_symbol("km").expect("kilometre should exist");
        assert!((u.si_factor - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_get_missing_symbol() {
        let c = catalog();
        assert!(c.get_by_symbol("XYZ").is_none());
    }

    // ------ get_by_urn ------

    #[test]
    fn test_get_by_urn() {
        let c = catalog();
        let u = c.get_by_symbol("m").expect("metre");
        let by_urn = c.get_by_urn(&u.urn).expect("lookup by urn");
        assert_eq!(by_urn.symbol, "m");
    }

    #[test]
    fn test_get_by_urn_missing() {
        let c = catalog();
        assert!(c.get_by_urn("urn:samm:unknown").is_none());
    }

    // ------ by_category ------

    #[test]
    fn test_by_category_length() {
        let c = catalog();
        let units = c.by_category(&UnitCategory::Length);
        assert!(units.len() >= 8);
        // sorted by name
        let names: Vec<&str> = units.iter().map(|u| u.name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    #[test]
    fn test_by_category_temperature() {
        let c = catalog();
        let units = c.by_category(&UnitCategory::Temperature);
        let syms: Vec<&str> = units.iter().map(|u| u.symbol.as_str()).collect();
        assert!(syms.contains(&"K"));
        assert!(syms.contains(&"°C"));
        assert!(syms.contains(&"°F"));
    }

    #[test]
    fn test_by_category_derived() {
        let c = catalog();
        let units = c.by_category(&UnitCategory::Derived);
        assert!(units.len() >= 7);
    }

    // ------ convert: length ------

    #[test]
    fn test_convert_metre_to_kilometre() {
        let c = catalog();
        let r = c.convert(1000.0, "m", "km").expect("convert");
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_convert_kilometre_to_metre() {
        let c = catalog();
        let r = c.convert(1.0, "km", "m").expect("convert");
        assert!((r - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn test_convert_mile_to_kilometre() {
        let c = catalog();
        let r = c.convert(1.0, "mi", "km").expect("convert");
        assert!((r - 1.609344).abs() < 1e-6);
    }

    #[test]
    fn test_convert_foot_to_metre() {
        let c = catalog();
        let r = c.convert(1.0, "ft", "m").expect("convert");
        assert!((r - 0.3048).abs() < 1e-9);
    }

    #[test]
    fn test_convert_inch_to_centimetre() {
        let c = catalog();
        let r = c.convert(1.0, "in", "cm").expect("convert");
        assert!((r - 2.54).abs() < 1e-9);
    }

    // ------ convert: mass ------

    #[test]
    fn test_convert_kg_to_gram() {
        let c = catalog();
        let r = c.convert(1.0, "kg", "g").expect("convert");
        assert!((r - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn test_convert_pound_to_kg() {
        let c = catalog();
        let r = c.convert(1.0, "lb", "kg").expect("convert");
        assert!((r - 0.453592).abs() < 1e-5);
    }

    // ------ convert: time ------

    #[test]
    fn test_convert_hour_to_second() {
        let c = catalog();
        let r = c.convert(1.0, "h", "s").expect("convert");
        assert!((r - 3600.0).abs() < 1e-9);
    }

    #[test]
    fn test_convert_minute_to_hour() {
        let c = catalog();
        let r = c.convert(60.0, "min", "h").expect("convert");
        assert!((r - 1.0).abs() < 1e-9);
    }

    // ------ convert: temperature ------

    #[test]
    fn test_convert_celsius_to_kelvin() {
        let c = catalog();
        let r = c.convert(0.0, "°C", "K").expect("convert");
        assert!((r - 273.15).abs() < 1e-6);
    }

    #[test]
    fn test_convert_kelvin_to_celsius() {
        let c = catalog();
        let r = c.convert(273.15, "K", "°C").expect("convert");
        assert!(r.abs() < 1e-6);
    }

    #[test]
    fn test_convert_fahrenheit_to_celsius() {
        let c = catalog();
        let r = c.convert(32.0, "°F", "°C").expect("convert");
        assert!(r.abs() < 1e-4);
    }

    #[test]
    fn test_convert_celsius_to_fahrenheit() {
        let c = catalog();
        let r = c.convert(100.0, "°C", "°F").expect("convert");
        assert!((r - 212.0).abs() < 1e-4);
    }

    // ------ convert: errors ------

    #[test]
    fn test_convert_unit_not_found() {
        let c = catalog();
        let e = c.convert(1.0, "XYZ", "m").expect_err("should fail");
        assert!(matches!(e, UnitError::UnitNotFound(_)));
    }

    #[test]
    fn test_convert_incompatible_units() {
        let c = catalog();
        let e = c.convert(1.0, "m", "kg").expect_err("should fail");
        assert!(matches!(e, UnitError::IncompatibleUnits { .. }));
    }

    // ------ is_convertible ------

    #[test]
    fn test_is_convertible_same_category() {
        let c = catalog();
        assert!(c.is_convertible("m", "km"));
        assert!(c.is_convertible("K", "°C"));
    }

    #[test]
    fn test_is_convertible_different_category() {
        let c = catalog();
        assert!(!c.is_convertible("m", "kg"));
    }

    #[test]
    fn test_is_convertible_unknown_unit() {
        let c = catalog();
        assert!(!c.is_convertible("XYZ", "m"));
    }

    // ------ UnitError Display ------

    #[test]
    fn test_unit_not_found_display() {
        let e = UnitError::UnitNotFound("XYZ".to_string());
        assert!(e.to_string().contains("XYZ"));
    }

    #[test]
    fn test_incompatible_units_display() {
        let e = UnitError::IncompatibleUnits {
            from: "m".to_string(),
            to: "kg".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("m"));
        assert!(s.contains("kg"));
    }

    // ------ URN format ------

    #[test]
    fn test_urn_format() {
        let c = catalog();
        let u = c.get_by_symbol("m").expect("metre");
        assert!(u.urn.starts_with("urn:samm:org.eclipse.esmf.samm:unit:"));
    }
}
