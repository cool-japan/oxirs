/// SAMM physical unit conversion engine.
///
/// Converts measurements between physical units using a linear model:
/// `si_value = measurement_value * si_factor + si_offset`.
/// Inverse: `measurement_value = (si_value - si_offset) / si_factor`.
use std::collections::HashMap;

/// A physical unit with an affine mapping to SI.
#[derive(Debug, Clone)]
pub struct Unit {
    /// IRI identifier for the unit (e.g. `unit:degreeCelsius`).
    pub iri: String,
    /// Short symbol (e.g. "°C", "m", "kg").
    pub symbol: String,
    /// Quantity kind (e.g. "temperature", "length", "mass").
    pub quantity_kind: String,
    /// Multiplier to convert this unit to SI: `si = value * si_factor + si_offset`.
    pub si_factor: f64,
    /// Offset term for affine conversions (non-zero for Celsius/Fahrenheit).
    pub si_offset: f64,
}

/// Errors from unit conversion operations.
#[derive(Debug)]
pub enum ConversionError {
    /// The unit IRI is not registered.
    UnknownUnit(String),
    /// The two units have incompatible quantity kinds.
    IncompatibleUnits(String, String),
    /// Division by zero during inverse conversion.
    DivisionByZero,
}

impl std::fmt::Display for ConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownUnit(u) => write!(f, "Unknown unit: {u}"),
            Self::IncompatibleUnits(a, b) => {
                write!(f, "Incompatible units: {a} and {b}")
            }
            Self::DivisionByZero => write!(f, "Division by zero in unit conversion"),
        }
    }
}

impl std::error::Error for ConversionError {}

/// A successful conversion result with provenance information.
#[derive(Debug, Clone)]
pub struct ConversionResult {
    /// The converted value in the target unit.
    pub value: f64,
    /// Source unit IRI.
    pub from_unit: String,
    /// Target unit IRI.
    pub to_unit: String,
    /// Human-readable conversion formula applied.
    pub formula: String,
}

/// Engine for converting between registered physical units.
#[derive(Debug, Default)]
pub struct UnitConverter {
    units: HashMap<String, Unit>,
}

impl UnitConverter {
    /// Create an empty converter.
    pub fn new() -> Self {
        Self {
            units: HashMap::new(),
        }
    }

    /// Create a converter pre-loaded with SI and common units.
    pub fn with_defaults() -> Self {
        let mut uc = Self::new();
        // --- Length ---
        uc.register(Unit {
            iri: "unit:metre".to_string(),
            symbol: "m".to_string(),
            quantity_kind: "length".to_string(),
            si_factor: 1.0,
            si_offset: 0.0,
        });
        uc.register(Unit {
            iri: "unit:kilometre".to_string(),
            symbol: "km".to_string(),
            quantity_kind: "length".to_string(),
            si_factor: 1000.0,
            si_offset: 0.0,
        });
        uc.register(Unit {
            iri: "unit:mile".to_string(),
            symbol: "mi".to_string(),
            quantity_kind: "length".to_string(),
            si_factor: 1609.344,
            si_offset: 0.0,
        });
        uc.register(Unit {
            iri: "unit:centimetre".to_string(),
            symbol: "cm".to_string(),
            quantity_kind: "length".to_string(),
            si_factor: 0.01,
            si_offset: 0.0,
        });
        uc.register(Unit {
            iri: "unit:inch".to_string(),
            symbol: "in".to_string(),
            quantity_kind: "length".to_string(),
            si_factor: 0.0254,
            si_offset: 0.0,
        });
        uc.register(Unit {
            iri: "unit:foot".to_string(),
            symbol: "ft".to_string(),
            quantity_kind: "length".to_string(),
            si_factor: 0.3048,
            si_offset: 0.0,
        });
        // --- Mass ---
        uc.register(Unit {
            iri: "unit:kilogram".to_string(),
            symbol: "kg".to_string(),
            quantity_kind: "mass".to_string(),
            si_factor: 1.0,
            si_offset: 0.0,
        });
        uc.register(Unit {
            iri: "unit:gram".to_string(),
            symbol: "g".to_string(),
            quantity_kind: "mass".to_string(),
            si_factor: 0.001,
            si_offset: 0.0,
        });
        uc.register(Unit {
            iri: "unit:pound".to_string(),
            symbol: "lb".to_string(),
            quantity_kind: "mass".to_string(),
            si_factor: 0.453592,
            si_offset: 0.0,
        });
        // --- Time ---
        uc.register(Unit {
            iri: "unit:second".to_string(),
            symbol: "s".to_string(),
            quantity_kind: "time".to_string(),
            si_factor: 1.0,
            si_offset: 0.0,
        });
        uc.register(Unit {
            iri: "unit:minute".to_string(),
            symbol: "min".to_string(),
            quantity_kind: "time".to_string(),
            si_factor: 60.0,
            si_offset: 0.0,
        });
        uc.register(Unit {
            iri: "unit:hour".to_string(),
            symbol: "h".to_string(),
            quantity_kind: "time".to_string(),
            si_factor: 3600.0,
            si_offset: 0.0,
        });
        // --- Temperature ---
        uc.register(Unit {
            iri: "unit:kelvin".to_string(),
            symbol: "K".to_string(),
            quantity_kind: "temperature".to_string(),
            si_factor: 1.0,
            si_offset: 0.0,
        });
        uc.register(Unit {
            iri: "unit:degreeCelsius".to_string(),
            symbol: "°C".to_string(),
            quantity_kind: "temperature".to_string(),
            si_factor: 1.0,
            si_offset: 273.15,
        });
        uc.register(Unit {
            iri: "unit:degreeFahrenheit".to_string(),
            symbol: "°F".to_string(),
            quantity_kind: "temperature".to_string(),
            si_factor: 5.0 / 9.0,
            si_offset: 273.15 - (32.0 * 5.0 / 9.0),
        });
        // --- Volume ---
        uc.register(Unit {
            iri: "unit:cubicMetre".to_string(),
            symbol: "m³".to_string(),
            quantity_kind: "volume".to_string(),
            si_factor: 1.0,
            si_offset: 0.0,
        });
        uc.register(Unit {
            iri: "unit:litre".to_string(),
            symbol: "L".to_string(),
            quantity_kind: "volume".to_string(),
            si_factor: 0.001,
            si_offset: 0.0,
        });
        uc.register(Unit {
            iri: "unit:gallon".to_string(),
            symbol: "gal".to_string(),
            quantity_kind: "volume".to_string(),
            si_factor: 0.003785411784,
            si_offset: 0.0,
        });
        uc
    }

    /// Register a unit in the converter.
    pub fn register(&mut self, unit: Unit) {
        self.units.insert(unit.iri.clone(), unit);
    }

    /// Convert a value from one unit to another.
    pub fn convert(
        &self,
        value: f64,
        from_iri: &str,
        to_iri: &str,
    ) -> Result<ConversionResult, ConversionError> {
        let from = self
            .units
            .get(from_iri)
            .ok_or_else(|| ConversionError::UnknownUnit(from_iri.to_string()))?;
        let to = self
            .units
            .get(to_iri)
            .ok_or_else(|| ConversionError::UnknownUnit(to_iri.to_string()))?;

        if from.quantity_kind != to.quantity_kind {
            return Err(ConversionError::IncompatibleUnits(
                from.quantity_kind.clone(),
                to.quantity_kind.clone(),
            ));
        }

        // Convert to SI then to target
        let si_value = value * from.si_factor + from.si_offset;
        let result_value = self.convert_from_si_unit(si_value, to)?;

        let formula = format!("{value} {} = {} {}", from.symbol, result_value, to.symbol);

        Ok(ConversionResult {
            value: result_value,
            from_unit: from_iri.to_string(),
            to_unit: to_iri.to_string(),
            formula,
        })
    }

    /// Convert a value to its SI equivalent.
    pub fn to_si(&self, value: f64, unit_iri: &str) -> Result<f64, ConversionError> {
        let unit = self
            .units
            .get(unit_iri)
            .ok_or_else(|| ConversionError::UnknownUnit(unit_iri.to_string()))?;
        Ok(value * unit.si_factor + unit.si_offset)
    }

    /// Convert a SI value to the specified unit.
    pub fn from_si(&self, si_value: f64, unit_iri: &str) -> Result<f64, ConversionError> {
        let unit = self
            .units
            .get(unit_iri)
            .ok_or_else(|| ConversionError::UnknownUnit(unit_iri.to_string()))?;
        self.convert_from_si_unit(si_value, unit)
    }

    fn convert_from_si_unit(&self, si_value: f64, unit: &Unit) -> Result<f64, ConversionError> {
        if unit.si_factor == 0.0 {
            return Err(ConversionError::DivisionByZero);
        }
        Ok((si_value - unit.si_offset) / unit.si_factor)
    }

    /// Return all units belonging to a given quantity kind.
    pub fn units_for_quantity(&self, quantity_kind: &str) -> Vec<&Unit> {
        self.units
            .values()
            .filter(|u| u.quantity_kind == quantity_kind)
            .collect()
    }

    /// Return the total number of registered units.
    pub fn unit_count(&self) -> usize {
        self.units.len()
    }

    /// Retrieve a unit by IRI.
    pub fn get_unit(&self, iri: &str) -> Option<&Unit> {
        self.units.get(iri)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    // --- metres / kilometres ---

    #[test]
    fn test_metres_to_kilometres() {
        let uc = UnitConverter::with_defaults();
        let r = uc.convert(1000.0, "unit:metre", "unit:kilometre").unwrap();
        assert!(approx_eq(r.value, 1.0));
    }

    #[test]
    fn test_kilometres_to_metres() {
        let uc = UnitConverter::with_defaults();
        let r = uc.convert(1.0, "unit:kilometre", "unit:metre").unwrap();
        assert!(approx_eq(r.value, 1000.0));
    }

    #[test]
    fn test_same_unit_identity() {
        let uc = UnitConverter::with_defaults();
        let r = uc.convert(42.0, "unit:metre", "unit:metre").unwrap();
        assert!(approx_eq(r.value, 42.0));
    }

    // --- temperature (offset conversions) ---

    #[test]
    fn test_celsius_to_fahrenheit() {
        let uc = UnitConverter::with_defaults();
        let r = uc
            .convert(0.0, "unit:degreeCelsius", "unit:degreeFahrenheit")
            .unwrap();
        // 0°C = 32°F
        assert!((r.value - 32.0).abs() < 0.001);
    }

    #[test]
    fn test_celsius_100_to_fahrenheit() {
        let uc = UnitConverter::with_defaults();
        let r = uc
            .convert(100.0, "unit:degreeCelsius", "unit:degreeFahrenheit")
            .unwrap();
        // 100°C = 212°F
        assert!((r.value - 212.0).abs() < 0.001);
    }

    #[test]
    fn test_fahrenheit_to_celsius() {
        let uc = UnitConverter::with_defaults();
        let r = uc
            .convert(32.0, "unit:degreeFahrenheit", "unit:degreeCelsius")
            .unwrap();
        // 32°F = 0°C
        assert!(r.value.abs() < 0.001);
    }

    #[test]
    fn test_celsius_to_kelvin() {
        let uc = UnitConverter::with_defaults();
        let r = uc
            .convert(0.0, "unit:degreeCelsius", "unit:kelvin")
            .unwrap();
        assert!((r.value - 273.15).abs() < 0.001);
    }

    // --- distance ---

    #[test]
    fn test_miles_to_metres() {
        let uc = UnitConverter::with_defaults();
        let r = uc.convert(1.0, "unit:mile", "unit:metre").unwrap();
        assert!((r.value - 1609.344).abs() < 0.01);
    }

    #[test]
    fn test_inches_to_centimetres() {
        let uc = UnitConverter::with_defaults();
        let r = uc.convert(1.0, "unit:inch", "unit:centimetre").unwrap();
        assert!((r.value - 2.54).abs() < 0.001);
    }

    // --- volume ---

    #[test]
    fn test_litres_to_gallons() {
        let uc = UnitConverter::with_defaults();
        let r = uc.convert(1.0, "unit:litre", "unit:gallon").unwrap();
        // 1 L ≈ 0.264172 gal
        assert!((r.value - 0.264172).abs() < 0.0001);
    }

    #[test]
    fn test_gallons_to_litres() {
        let uc = UnitConverter::with_defaults();
        let r = uc.convert(1.0, "unit:gallon", "unit:litre").unwrap();
        assert!((r.value - 3.785411784).abs() < 0.0001);
    }

    // --- error cases ---

    #[test]
    fn test_unknown_from_unit() {
        let uc = UnitConverter::with_defaults();
        let err = uc.convert(1.0, "unit:unknown", "unit:metre");
        assert!(matches!(err, Err(ConversionError::UnknownUnit(_))));
    }

    #[test]
    fn test_unknown_to_unit() {
        let uc = UnitConverter::with_defaults();
        let err = uc.convert(1.0, "unit:metre", "unit:unknown");
        assert!(matches!(err, Err(ConversionError::UnknownUnit(_))));
    }

    #[test]
    fn test_incompatible_units() {
        let uc = UnitConverter::with_defaults();
        let err = uc.convert(1.0, "unit:metre", "unit:kilogram");
        assert!(matches!(err, Err(ConversionError::IncompatibleUnits(_, _))));
    }

    // --- to_si / from_si ---

    #[test]
    fn test_to_si_metre() {
        let uc = UnitConverter::with_defaults();
        let si = uc.to_si(5.0, "unit:metre").unwrap();
        assert!(approx_eq(si, 5.0));
    }

    #[test]
    fn test_to_si_kilometre() {
        let uc = UnitConverter::with_defaults();
        let si = uc.to_si(3.0, "unit:kilometre").unwrap();
        assert!(approx_eq(si, 3000.0));
    }

    #[test]
    fn test_from_si_metre_roundtrip() {
        let uc = UnitConverter::with_defaults();
        let si = uc.to_si(7.5, "unit:kilometre").unwrap();
        let back = uc.from_si(si, "unit:kilometre").unwrap();
        assert!((back - 7.5).abs() < EPSILON);
    }

    #[test]
    fn test_celsius_to_si_round_trip() {
        let uc = UnitConverter::with_defaults();
        let si = uc.to_si(25.0, "unit:degreeCelsius").unwrap();
        let back = uc.from_si(si, "unit:degreeCelsius").unwrap();
        assert!((back - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_to_si_unknown() {
        let uc = UnitConverter::with_defaults();
        assert!(uc.to_si(1.0, "unit:xyz").is_err());
    }

    #[test]
    fn test_from_si_unknown() {
        let uc = UnitConverter::with_defaults();
        assert!(uc.from_si(1.0, "unit:xyz").is_err());
    }

    // --- units_for_quantity ---

    #[test]
    fn test_units_for_length() {
        let uc = UnitConverter::with_defaults();
        let units = uc.units_for_quantity("length");
        assert!(!units.is_empty());
        assert!(units.iter().any(|u| u.iri == "unit:metre"));
        assert!(units.iter().any(|u| u.iri == "unit:kilometre"));
    }

    #[test]
    fn test_units_for_temperature() {
        let uc = UnitConverter::with_defaults();
        let units = uc.units_for_quantity("temperature");
        assert!(units.iter().any(|u| u.iri == "unit:degreeCelsius"));
        assert!(units.iter().any(|u| u.iri == "unit:kelvin"));
    }

    #[test]
    fn test_units_for_unknown_quantity() {
        let uc = UnitConverter::with_defaults();
        let units = uc.units_for_quantity("luminosity");
        assert!(units.is_empty());
    }

    // --- unit_count / get_unit ---

    #[test]
    fn test_unit_count_with_defaults() {
        let uc = UnitConverter::with_defaults();
        assert!(uc.unit_count() >= 10);
    }

    #[test]
    fn test_get_unit_present() {
        let uc = UnitConverter::with_defaults();
        let u = uc.get_unit("unit:metre");
        assert!(u.is_some());
        assert_eq!(u.unwrap().symbol, "m");
    }

    #[test]
    fn test_get_unit_absent() {
        let uc = UnitConverter::new();
        assert!(uc.get_unit("unit:metre").is_none());
    }

    // --- custom unit registration ---

    #[test]
    fn test_register_custom_unit() {
        let mut uc = UnitConverter::new();
        uc.register(Unit {
            iri: "unit:parsec".to_string(),
            symbol: "pc".to_string(),
            quantity_kind: "length".to_string(),
            si_factor: 3.085677581e16,
            si_offset: 0.0,
        });
        assert_eq!(uc.unit_count(), 1);
        assert!(uc.get_unit("unit:parsec").is_some());
    }

    // --- formula string ---

    #[test]
    fn test_conversion_result_formula_populated() {
        let uc = UnitConverter::with_defaults();
        let r = uc.convert(1000.0, "unit:metre", "unit:kilometre").unwrap();
        assert!(!r.formula.is_empty());
        assert!(r.formula.contains("km"));
    }

    // --- error display ---

    #[test]
    fn test_unknown_unit_error_display() {
        let err = ConversionError::UnknownUnit("unit:xyz".to_string());
        assert!(format!("{err}").contains("unit:xyz"));
    }

    #[test]
    fn test_incompatible_units_display() {
        let err = ConversionError::IncompatibleUnits("length".to_string(), "mass".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("length") && msg.contains("mass"));
    }

    #[test]
    fn test_division_by_zero_display() {
        let err = ConversionError::DivisionByZero;
        assert!(format!("{err}").contains("zero"));
    }

    // --- mass conversions ---

    #[test]
    fn test_grams_to_kilograms() {
        let uc = UnitConverter::with_defaults();
        let r = uc.convert(1000.0, "unit:gram", "unit:kilogram").unwrap();
        assert!((r.value - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_pounds_to_kilograms() {
        let uc = UnitConverter::with_defaults();
        let r = uc.convert(1.0, "unit:pound", "unit:kilogram").unwrap();
        assert!((r.value - 0.453592).abs() < 0.0001);
    }
}
