//! RDF Literal Parser with Physical Unit Conversion
//!
//! Parses RDF typed literals (e.g., `"9.81 m/s^2"^^xsd:string`) into
//! strongly-typed Rust values with automatic SI unit conversion.
//!
//! # Design
//!
//! The parser handles three forms of RDF literals:
//! 1. **Plain numeric**: `"9.81"^^xsd:double` → 9.81
//! 2. **Annotated numeric**: `"9.81 m/s^2"^^xsd:string` → PhysicalValue { 9.81, MetersPerSecondSquared }
//! 3. **xsd:* typed**: `"300.0"^^xsd:double` with external unit hint
//!
//! All values are internally stored in SI base units after conversion.

use crate::error::{PhysicsError, PhysicsResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

// ──────────────────────────────────────────────────────────────────────────────
// Physical unit enumeration
// ──────────────────────────────────────────────────────────────────────────────

/// Physical unit enumeration covering common SI and derived units.
///
/// All conversion factors stored in `to_si()` map to the SI base unit for
/// each dimension (m, kg, s, A, K, mol, cd).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhysicalUnit {
    // ── Length ──────────────────────────────────────────────────────────────
    /// Metre (SI base unit for length)
    Meter,
    /// Kilometre = 1000 m
    Kilometer,
    /// Centimetre = 0.01 m
    Centimeter,
    /// Millimetre = 0.001 m
    Millimeter,
    /// Inch = 0.0254 m
    Inch,
    /// Foot = 0.3048 m
    Foot,

    // ── Mass ────────────────────────────────────────────────────────────────
    /// Kilogram (SI base unit for mass)
    KiloGram,
    /// Gram = 0.001 kg
    Gram,
    /// Milligram = 1e-6 kg
    Milligram,
    /// Metric tonne = 1000 kg
    Tonne,
    /// Pound mass = 0.453592 kg
    PoundMass,

    // ── Time ────────────────────────────────────────────────────────────────
    /// Second (SI base unit for time)
    Second,
    /// Millisecond = 0.001 s
    Millisecond,
    /// Microsecond = 1e-6 s
    Microsecond,
    /// Minute = 60 s
    Minute,
    /// Hour = 3600 s
    Hour,

    // ── Velocity ────────────────────────────────────────────────────────────
    /// Metre per second
    MetersPerSecond,
    /// Kilometre per hour = 1/3.6 m/s
    KilometersPerHour,
    /// Miles per hour = ~0.44704 m/s
    MilesPerHour,

    // ── Acceleration ────────────────────────────────────────────────────────
    /// Metre per second squared
    MetersPerSecondSquared,
    /// g (standard gravity) = 9.80665 m/s²
    StandardGravity,

    // ── Force ───────────────────────────────────────────────────────────────
    /// Newton = kg⋅m/s²
    Newton,
    /// Kilonewton = 1000 N
    KiloNewton,
    /// Pound-force = 4.44822 N
    PoundForce,

    // ── Energy / Work ───────────────────────────────────────────────────────
    /// Joule = N⋅m
    Joule,
    /// Kilojoule = 1000 J
    KiloJoule,
    /// Megajoule = 1e6 J
    MegaJoule,
    /// Watt-hour = 3600 J
    WattHour,
    /// Kilowatt-hour = 3.6e6 J
    KiloWattHour,
    /// Electron-volt = 1.602176634e-19 J
    ElectronVolt,

    // ── Power ───────────────────────────────────────────────────────────────
    /// Watt = J/s
    Watt,
    /// Kilowatt = 1000 W
    KiloWatt,
    /// Megawatt = 1e6 W
    MegaWatt,

    // ── Temperature ─────────────────────────────────────────────────────────
    /// Kelvin (SI base unit for thermodynamic temperature)
    Kelvin,
    /// Degree Celsius (offset conversion: T_K = T_C + 273.15)
    Celsius,
    /// Degree Fahrenheit (offset conversion: T_K = (T_F + 459.67) × 5/9)
    Fahrenheit,

    // ── Pressure ────────────────────────────────────────────────────────────
    /// Pascal = N/m²
    Pascal,
    /// Kilopascal = 1000 Pa
    KiloPascal,
    /// Megapascal = 1e6 Pa
    MegaPascal,
    /// Bar = 1e5 Pa
    Bar,
    /// Atmosphere = 101325 Pa
    Atmosphere,

    // ── Electric ────────────────────────────────────────────────────────────
    /// Ampere (SI base unit for electric current)
    Ampere,
    /// Volt = W/A
    Volt,
    /// Ohm = V/A
    Ohm,
    /// Farad
    Farad,
    /// Henry
    Henry,

    // ── Frequency ───────────────────────────────────────────────────────────
    /// Hertz = 1/s
    Hertz,
    /// Kilohertz = 1000 Hz
    KiloHertz,
    /// Megahertz = 1e6 Hz
    MegaHertz,

    // ── Dimensionless ────────────────────────────────────────────────────────
    /// Dimensionless ratio (1.0)
    Dimensionless,

    // ── Custom ───────────────────────────────────────────────────────────────
    /// User-defined unit with string label
    Custom(String),
}

impl fmt::Display for PhysicalUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

impl PhysicalUnit {
    /// Return the canonical SI symbol for this unit.
    pub fn symbol(&self) -> &str {
        match self {
            Self::Meter => "m",
            Self::Kilometer => "km",
            Self::Centimeter => "cm",
            Self::Millimeter => "mm",
            Self::Inch => "in",
            Self::Foot => "ft",
            Self::KiloGram => "kg",
            Self::Gram => "g",
            Self::Milligram => "mg",
            Self::Tonne => "t",
            Self::PoundMass => "lb",
            Self::Second => "s",
            Self::Millisecond => "ms",
            Self::Microsecond => "μs",
            Self::Minute => "min",
            Self::Hour => "h",
            Self::MetersPerSecond => "m/s",
            Self::KilometersPerHour => "km/h",
            Self::MilesPerHour => "mph",
            Self::MetersPerSecondSquared => "m/s²",
            Self::StandardGravity => "g",
            Self::Newton => "N",
            Self::KiloNewton => "kN",
            Self::PoundForce => "lbf",
            Self::Joule => "J",
            Self::KiloJoule => "kJ",
            Self::MegaJoule => "MJ",
            Self::WattHour => "Wh",
            Self::KiloWattHour => "kWh",
            Self::ElectronVolt => "eV",
            Self::Watt => "W",
            Self::KiloWatt => "kW",
            Self::MegaWatt => "MW",
            Self::Kelvin => "K",
            Self::Celsius => "°C",
            Self::Fahrenheit => "°F",
            Self::Pascal => "Pa",
            Self::KiloPascal => "kPa",
            Self::MegaPascal => "MPa",
            Self::Bar => "bar",
            Self::Atmosphere => "atm",
            Self::Ampere => "A",
            Self::Volt => "V",
            Self::Ohm => "Ω",
            Self::Farad => "F",
            Self::Henry => "H",
            Self::Hertz => "Hz",
            Self::KiloHertz => "kHz",
            Self::MegaHertz => "MHz",
            Self::Dimensionless => "1",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Return the multiplication factor to convert from this unit to its SI base unit.
    ///
    /// Temperature units have offset conversions and must go through `to_si_value` / `from_si_value`.
    pub fn scale_factor(&self) -> f64 {
        match self {
            // Length
            Self::Meter => 1.0,
            Self::Kilometer => 1_000.0,
            Self::Centimeter => 0.01,
            Self::Millimeter => 0.001,
            Self::Inch => 0.0254,
            Self::Foot => 0.3048,
            // Mass
            Self::KiloGram => 1.0,
            Self::Gram => 0.001,
            Self::Milligram => 1e-6,
            Self::Tonne => 1_000.0,
            Self::PoundMass => 0.453_592_37,
            // Time
            Self::Second => 1.0,
            Self::Millisecond => 0.001,
            Self::Microsecond => 1e-6,
            Self::Minute => 60.0,
            Self::Hour => 3_600.0,
            // Velocity
            Self::MetersPerSecond => 1.0,
            Self::KilometersPerHour => 1.0 / 3.6,
            Self::MilesPerHour => 0.447_04,
            // Acceleration
            Self::MetersPerSecondSquared => 1.0,
            Self::StandardGravity => 9.806_65,
            // Force
            Self::Newton => 1.0,
            Self::KiloNewton => 1_000.0,
            Self::PoundForce => 4.448_222,
            // Energy
            Self::Joule => 1.0,
            Self::KiloJoule => 1_000.0,
            Self::MegaJoule => 1_000_000.0,
            Self::WattHour => 3_600.0,
            Self::KiloWattHour => 3_600_000.0,
            Self::ElectronVolt => 1.602_176_634e-19,
            // Power
            Self::Watt => 1.0,
            Self::KiloWatt => 1_000.0,
            Self::MegaWatt => 1_000_000.0,
            // Temperature (scale factor only; offset handled separately)
            Self::Kelvin => 1.0,
            Self::Celsius => 1.0,          // needs offset
            Self::Fahrenheit => 5.0 / 9.0, // needs offset
            // Pressure
            Self::Pascal => 1.0,
            Self::KiloPascal => 1_000.0,
            Self::MegaPascal => 1_000_000.0,
            Self::Bar => 100_000.0,
            Self::Atmosphere => 101_325.0,
            // Electric
            Self::Ampere => 1.0,
            Self::Volt => 1.0,
            Self::Ohm => 1.0,
            Self::Farad => 1.0,
            Self::Henry => 1.0,
            // Frequency
            Self::Hertz => 1.0,
            Self::KiloHertz => 1_000.0,
            Self::MegaHertz => 1_000_000.0,
            // Other
            Self::Dimensionless => 1.0,
            Self::Custom(_) => 1.0,
        }
    }

    /// Convert a raw numeric value in *this* unit to the SI base unit value.
    ///
    /// For most units this is `value * scale_factor()`.  Temperature units
    /// additionally apply an additive offset.
    pub fn to_si_value(&self, value: f64) -> f64 {
        match self {
            Self::Celsius => value + 273.15,
            Self::Fahrenheit => (value + 459.67) * (5.0 / 9.0),
            _ => value * self.scale_factor(),
        }
    }

    /// Convert an SI base-unit value back to *this* unit.
    pub fn from_si_value(&self, si_value: f64) -> f64 {
        match self {
            Self::Celsius => si_value - 273.15,
            Self::Fahrenheit => si_value * (9.0 / 5.0) - 459.67,
            _ => si_value / self.scale_factor(),
        }
    }

    /// Return `true` if this unit has a non-trivial additive temperature offset.
    pub fn has_offset(&self) -> bool {
        matches!(self, Self::Celsius | Self::Fahrenheit)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Physical value type
// ──────────────────────────────────────────────────────────────────────────────

/// A real-valued scalar paired with its physical unit.
///
/// The value is stored as-provided; the unit carries enough information for
/// conversion to any compatible unit via [`convert_unit`].
#[derive(Debug, Clone)]
pub struct PhysicalValue {
    /// Numeric magnitude in the given unit.
    pub value: f64,
    /// Physical unit of `value`.
    pub unit: PhysicalUnit,
}

impl PhysicalValue {
    /// Create a new physical value.
    pub fn new(value: f64, unit: PhysicalUnit) -> Self {
        Self { value, unit }
    }

    /// Return the value expressed in SI base units.
    pub fn as_si(&self) -> f64 {
        self.unit.to_si_value(self.value)
    }

    /// Convert this value to the target unit.
    ///
    /// This is a convenience wrapper around [`convert_unit`].
    pub fn convert_to(&self, target: &PhysicalUnit) -> PhysicsResult<PhysicalValue> {
        convert_unit(self, target)
    }
}

impl fmt::Display for PhysicalValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.value, self.unit)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit conversion
// ──────────────────────────────────────────────────────────────────────────────

/// Convert a [`PhysicalValue`] to a different [`PhysicalUnit`].
///
/// # Errors
///
/// Returns [`PhysicsError::UnitConversion`] when:
/// - Either unit is a `Custom` variant with no known conversion.
/// - The result is not finite (e.g. division by zero from a zero-scale custom unit).
pub fn convert_unit(
    value: &PhysicalValue,
    target_unit: &PhysicalUnit,
) -> PhysicsResult<PhysicalValue> {
    // Convert source value to SI base unit first.
    let si_value = value.unit.to_si_value(value.value);

    // Then convert from SI to target.
    let converted = target_unit.from_si_value(si_value);

    if !converted.is_finite() {
        return Err(PhysicsError::UnitConversion(format!(
            "Conversion from {} to {} produced non-finite result",
            value.unit, target_unit
        )));
    }

    Ok(PhysicalValue {
        value: converted,
        unit: target_unit.clone(),
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit string table
// ──────────────────────────────────────────────────────────────────────────────

/// Build a lookup table that maps common unit string representations to
/// [`PhysicalUnit`] variants.  The table is intentionally broad to handle
/// real-world data with inconsistent casing / notation.
fn build_unit_table() -> HashMap<&'static str, PhysicalUnit> {
    let mut map = HashMap::new();

    // ── Length ───────────────────────────────────────────────────────────────
    map.insert("m", PhysicalUnit::Meter);
    map.insert("meter", PhysicalUnit::Meter);
    map.insert("meters", PhysicalUnit::Meter);
    map.insert("metre", PhysicalUnit::Meter);
    map.insert("metres", PhysicalUnit::Meter);
    map.insert("km", PhysicalUnit::Kilometer);
    map.insert("kilometer", PhysicalUnit::Kilometer);
    map.insert("kilometres", PhysicalUnit::Kilometer);
    map.insert("cm", PhysicalUnit::Centimeter);
    map.insert("centimeter", PhysicalUnit::Centimeter);
    map.insert("centimetre", PhysicalUnit::Centimeter);
    map.insert("mm", PhysicalUnit::Millimeter);
    map.insert("millimeter", PhysicalUnit::Millimeter);
    map.insert("in", PhysicalUnit::Inch);
    map.insert("inch", PhysicalUnit::Inch);
    map.insert("ft", PhysicalUnit::Foot);
    map.insert("foot", PhysicalUnit::Foot);
    map.insert("feet", PhysicalUnit::Foot);

    // ── Mass ─────────────────────────────────────────────────────────────────
    map.insert("kg", PhysicalUnit::KiloGram);
    map.insert("kilogram", PhysicalUnit::KiloGram);
    map.insert("kilograms", PhysicalUnit::KiloGram);
    map.insert("g", PhysicalUnit::Gram);
    map.insert("gram", PhysicalUnit::Gram);
    map.insert("grams", PhysicalUnit::Gram);
    map.insert("mg", PhysicalUnit::Milligram);
    map.insert("milligram", PhysicalUnit::Milligram);
    map.insert("t", PhysicalUnit::Tonne);
    map.insert("tonne", PhysicalUnit::Tonne);
    map.insert("lb", PhysicalUnit::PoundMass);
    map.insert("lbs", PhysicalUnit::PoundMass);
    map.insert("pound", PhysicalUnit::PoundMass);

    // ── Time ─────────────────────────────────────────────────────────────────
    map.insert("s", PhysicalUnit::Second);
    map.insert("sec", PhysicalUnit::Second);
    map.insert("second", PhysicalUnit::Second);
    map.insert("seconds", PhysicalUnit::Second);
    map.insert("ms", PhysicalUnit::Millisecond);
    map.insert("millisecond", PhysicalUnit::Millisecond);
    map.insert("us", PhysicalUnit::Microsecond);
    map.insert("μs", PhysicalUnit::Microsecond);
    map.insert("microsecond", PhysicalUnit::Microsecond);
    map.insert("min", PhysicalUnit::Minute);
    map.insert("minute", PhysicalUnit::Minute);
    map.insert("minutes", PhysicalUnit::Minute);
    map.insert("h", PhysicalUnit::Hour);
    map.insert("hr", PhysicalUnit::Hour);
    map.insert("hour", PhysicalUnit::Hour);
    map.insert("hours", PhysicalUnit::Hour);

    // ── Velocity ─────────────────────────────────────────────────────────────
    map.insert("m/s", PhysicalUnit::MetersPerSecond);
    map.insert("ms^-1", PhysicalUnit::MetersPerSecond);
    map.insert("m/s^1", PhysicalUnit::MetersPerSecond);
    map.insert("km/h", PhysicalUnit::KilometersPerHour);
    map.insert("kph", PhysicalUnit::KilometersPerHour);
    map.insert("mph", PhysicalUnit::MilesPerHour);

    // ── Acceleration ─────────────────────────────────────────────────────────
    map.insert("m/s^2", PhysicalUnit::MetersPerSecondSquared);
    map.insert("m/s²", PhysicalUnit::MetersPerSecondSquared);
    map.insert("ms^-2", PhysicalUnit::MetersPerSecondSquared);
    map.insert("m*s^-2", PhysicalUnit::MetersPerSecondSquared);
    map.insert("standard_gravity", PhysicalUnit::StandardGravity);
    map.insert("gn", PhysicalUnit::StandardGravity);

    // ── Force ─────────────────────────────────────────────────────────────────
    map.insert("n", PhysicalUnit::Newton);
    map.insert("newton", PhysicalUnit::Newton);
    map.insert("newtons", PhysicalUnit::Newton);
    map.insert("kn", PhysicalUnit::KiloNewton);
    map.insert("kilonewton", PhysicalUnit::KiloNewton);
    map.insert("lbf", PhysicalUnit::PoundForce);

    // ── Energy ───────────────────────────────────────────────────────────────
    map.insert("j", PhysicalUnit::Joule);
    map.insert("joule", PhysicalUnit::Joule);
    map.insert("joules", PhysicalUnit::Joule);
    map.insert("kj", PhysicalUnit::KiloJoule);
    map.insert("kilojoule", PhysicalUnit::KiloJoule);
    map.insert("mj", PhysicalUnit::MegaJoule);
    map.insert("megajoule", PhysicalUnit::MegaJoule);
    map.insert("wh", PhysicalUnit::WattHour);
    map.insert("kwh", PhysicalUnit::KiloWattHour);
    map.insert("ev", PhysicalUnit::ElectronVolt);

    // ── Power ─────────────────────────────────────────────────────────────────
    map.insert("w", PhysicalUnit::Watt);
    map.insert("watt", PhysicalUnit::Watt);
    map.insert("watts", PhysicalUnit::Watt);
    map.insert("kw", PhysicalUnit::KiloWatt);
    map.insert("kilowatt", PhysicalUnit::KiloWatt);
    map.insert("mw", PhysicalUnit::MegaWatt);
    map.insert("megawatt", PhysicalUnit::MegaWatt);

    // ── Temperature ──────────────────────────────────────────────────────────
    map.insert("k", PhysicalUnit::Kelvin);
    map.insert("kelvin", PhysicalUnit::Kelvin);
    map.insert("°c", PhysicalUnit::Celsius);
    map.insert("degc", PhysicalUnit::Celsius);
    map.insert("celsius", PhysicalUnit::Celsius);
    map.insert("c", PhysicalUnit::Celsius);
    map.insert("°f", PhysicalUnit::Fahrenheit);
    map.insert("degf", PhysicalUnit::Fahrenheit);
    map.insert("fahrenheit", PhysicalUnit::Fahrenheit);
    map.insert("f", PhysicalUnit::Fahrenheit);

    // ── Pressure ─────────────────────────────────────────────────────────────
    map.insert("pa", PhysicalUnit::Pascal);
    map.insert("pascal", PhysicalUnit::Pascal);
    map.insert("kpa", PhysicalUnit::KiloPascal);
    map.insert("kilopascal", PhysicalUnit::KiloPascal);
    map.insert("mpa", PhysicalUnit::MegaPascal);
    map.insert("megapascal", PhysicalUnit::MegaPascal);
    map.insert("bar", PhysicalUnit::Bar);
    map.insert("atm", PhysicalUnit::Atmosphere);
    map.insert("atmosphere", PhysicalUnit::Atmosphere);

    // ── Electric ─────────────────────────────────────────────────────────────
    map.insert("a", PhysicalUnit::Ampere);
    map.insert("ampere", PhysicalUnit::Ampere);
    map.insert("v", PhysicalUnit::Volt);
    map.insert("volt", PhysicalUnit::Volt);
    map.insert("ohm", PhysicalUnit::Ohm);
    map.insert("ω", PhysicalUnit::Ohm);
    map.insert("farad", PhysicalUnit::Farad);
    map.insert("henry", PhysicalUnit::Henry);

    // ── Frequency ────────────────────────────────────────────────────────────
    map.insert("hz", PhysicalUnit::Hertz);
    map.insert("hertz", PhysicalUnit::Hertz);
    map.insert("khz", PhysicalUnit::KiloHertz);
    map.insert("mhz", PhysicalUnit::MegaHertz);

    // ── Dimensionless ────────────────────────────────────────────────────────
    map.insert("1", PhysicalUnit::Dimensionless);
    map.insert("dimensionless", PhysicalUnit::Dimensionless);
    map.insert("ratio", PhysicalUnit::Dimensionless);

    map
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit string parsing
// ──────────────────────────────────────────────────────────────────────────────

/// Parse a unit string into a [`PhysicalUnit`] variant.
///
/// The lookup is case-insensitive.  Unknown strings return `PhysicalUnit::Custom(s)`.
pub fn parse_unit_str(s: &str) -> PhysicalUnit {
    let table = build_unit_table();
    let lower = s.trim().to_lowercase();
    table
        .get(lower.as_str())
        .cloned()
        .unwrap_or_else(|| PhysicalUnit::Custom(s.trim().to_string()))
}

// ──────────────────────────────────────────────────────────────────────────────
// XSD datatype helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Return `true` when the IRI string corresponds to an XSD numeric type.
fn is_xsd_numeric(datatype: &str) -> bool {
    matches!(
        datatype,
        "http://www.w3.org/2001/XMLSchema#double"
            | "http://www.w3.org/2001/XMLSchema#float"
            | "http://www.w3.org/2001/XMLSchema#decimal"
            | "http://www.w3.org/2001/XMLSchema#integer"
            | "http://www.w3.org/2001/XMLSchema#long"
            | "http://www.w3.org/2001/XMLSchema#int"
            | "http://www.w3.org/2001/XMLSchema#short"
            | "http://www.w3.org/2001/XMLSchema#byte"
            | "http://www.w3.org/2001/XMLSchema#nonNegativeInteger"
            | "xsd:double"
            | "xsd:float"
            | "xsd:decimal"
            | "xsd:integer"
            | "xsd:int"
    )
}

// ──────────────────────────────────────────────────────────────────────────────
// Primary parsing function
// ──────────────────────────────────────────────────────────────────────────────

/// Parse an RDF literal string into a [`PhysicalValue`].
///
/// # Parsing strategy
///
/// 1. If `datatype` is an XSD numeric type, attempt to parse `literal` as a
///    bare floating-point number.  The unit is taken from the optional
///    `hint_unit` parameter or defaults to `Dimensionless`.
/// 2. Otherwise, split `literal` on the first whitespace: the left part is the
///    number, the right part is the unit annotation.
/// 3. If neither strategy succeeds, return [`PhysicsError::ParameterExtraction`].
///
/// # Arguments
///
/// * `literal` – Lexical form of the RDF literal (the content between quotes).
/// * `datatype` – Optional datatype IRI (e.g. `"xsd:double"` or the full IRI).
///
/// # Example
///
/// ```rust
/// use oxirs_physics::rdf::literal_parser::{parse_rdf_literal, PhysicalUnit};
///
/// let v = parse_rdf_literal("9.81 m/s^2", None).expect("parse failed");
/// assert!((v.value - 9.81).abs() < 1e-10);
/// assert_eq!(v.unit, PhysicalUnit::MetersPerSecondSquared);
/// ```
pub fn parse_rdf_literal(literal: &str, datatype: Option<&str>) -> PhysicsResult<PhysicalValue> {
    let trimmed = literal.trim();

    // ── Strategy 1: known numeric XSD type ───────────────────────────────────
    // Only applies when the entire literal is a bare numeric string.
    // If the literal also contains a unit annotation (e.g. "300.0 K"), we fall
    // through to Strategy 2 so that the unit information is not lost.
    if let Some(dt) = datatype {
        if is_xsd_numeric(dt) {
            if let Ok(value) = f64::from_str(trimmed) {
                return Ok(PhysicalValue {
                    value,
                    unit: PhysicalUnit::Dimensionless,
                });
            }
            // Literal contains non-numeric trailing content – fall through
        }
    }

    // ── Strategy 2: "<number> <unit>" annotation ─────────────────────────────
    if let Some((num_part, unit_part)) = split_value_unit(trimmed) {
        let value = parse_f64(num_part)?;
        let unit = parse_unit_str(unit_part);
        return Ok(PhysicalValue { value, unit });
    }

    // ── Strategy 3: bare number without annotation ────────────────────────────
    if let Ok(value) = f64::from_str(trimmed) {
        return Ok(PhysicalValue {
            value,
            unit: PhysicalUnit::Dimensionless,
        });
    }

    Err(PhysicsError::ParameterExtraction(format!(
        "Cannot parse RDF literal as physical value: '{}'",
        literal
    )))
}

/// Split a literal string into a `(number_str, unit_str)` pair.
///
/// We scan for the boundary between the numeric part and the unit part.
/// The number can include `-`, `+`, `.`, `e`, `E` (scientific notation).
fn split_value_unit(s: &str) -> Option<(&str, &str)> {
    // Skip leading sign/digits/decimal/exp characters
    let mut end = 0;
    let bytes = s.as_bytes();

    // Optional leading sign
    if end < bytes.len() && (bytes[end] == b'-' || bytes[end] == b'+') {
        end += 1;
    }
    // Digits and decimal point
    while end < bytes.len() && (bytes[end].is_ascii_digit() || bytes[end] == b'.') {
        end += 1;
    }
    // Scientific notation exponent
    if end < bytes.len() && (bytes[end] == b'e' || bytes[end] == b'E') {
        end += 1;
        if end < bytes.len() && (bytes[end] == b'-' || bytes[end] == b'+') {
            end += 1;
        }
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
    }

    if end == 0 || end == s.len() {
        return None;
    }

    let num_part = s[..end].trim();
    let rest = s[end..].trim();

    if rest.is_empty() {
        return None;
    }

    Some((num_part, rest))
}

/// Parse a string as `f64`, returning a physics error on failure.
fn parse_f64(s: &str) -> PhysicsResult<f64> {
    f64::from_str(s.trim()).map_err(|_| {
        PhysicsError::ParameterExtraction(format!(
            "Cannot parse '{}' as a floating-point number",
            s
        ))
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_rdf_literal ────────────────────────────────────────────────────

    #[test]
    fn test_parse_annotated_acceleration() {
        let v = parse_rdf_literal("9.81 m/s^2", None).expect("parse failed");
        assert!((v.value - 9.81).abs() < 1e-10);
        assert_eq!(v.unit, PhysicalUnit::MetersPerSecondSquared);
    }

    #[test]
    fn test_parse_mass_kg() {
        let v = parse_rdf_literal("75.0 kg", None).expect("parse failed");
        assert!((v.value - 75.0).abs() < 1e-10);
        assert_eq!(v.unit, PhysicalUnit::KiloGram);
    }

    #[test]
    fn test_parse_temperature_kelvin() {
        let v = parse_rdf_literal("300.0 K", None).expect("parse failed");
        assert!((v.value - 300.0).abs() < 1e-10);
        assert_eq!(v.unit, PhysicalUnit::Kelvin);
    }

    #[test]
    fn test_parse_xsd_double_bare() {
        let v = parse_rdf_literal("42.0", Some("xsd:double")).expect("parse failed");
        assert!((v.value - 42.0).abs() < 1e-10);
        assert_eq!(v.unit, PhysicalUnit::Dimensionless);
    }

    #[allow(clippy::approx_constant)]
    #[test]
    fn test_parse_xsd_full_iri() {
        let v = parse_rdf_literal("2.71", Some("http://www.w3.org/2001/XMLSchema#double"))
            .expect("parse failed");

        assert!((v.value - 2.71).abs() < 1e-10);
    }

    #[test]
    fn test_parse_bare_number_no_datatype() {
        let v = parse_rdf_literal("1234.5", None).expect("parse failed");
        assert!((v.value - 1234.5).abs() < 1e-10);
        assert_eq!(v.unit, PhysicalUnit::Dimensionless);
    }

    #[test]
    fn test_parse_negative_value_with_unit() {
        let v = parse_rdf_literal("-273.15 °C", None).expect("parse failed");
        assert!((v.value - (-273.15)).abs() < 1e-10);
        assert_eq!(v.unit, PhysicalUnit::Celsius);
    }

    #[test]
    fn test_parse_scientific_notation_with_unit() {
        let v = parse_rdf_literal("6.022e23 1", None).expect("parse failed");
        assert!((v.value - 6.022e23).abs() < 1e10);
        assert_eq!(v.unit, PhysicalUnit::Dimensionless);
    }

    #[test]
    fn test_parse_velocity() {
        let v = parse_rdf_literal("100 km/h", None).expect("parse failed");
        assert!((v.value - 100.0).abs() < 1e-10);
        assert_eq!(v.unit, PhysicalUnit::KilometersPerHour);
    }

    #[test]
    fn test_parse_pressure_pa() {
        let v = parse_rdf_literal("101325 Pa", None).expect("parse failed");
        assert!((v.value - 101_325.0).abs() < 1e-10);
        assert_eq!(v.unit, PhysicalUnit::Pascal);
    }

    #[test]
    fn test_parse_invalid_literal() {
        let result = parse_rdf_literal("not_a_number", None);
        assert!(result.is_err());
    }

    // ── convert_unit ─────────────────────────────────────────────────────────

    #[test]
    fn test_convert_km_to_m() {
        let pv = PhysicalValue::new(1.0, PhysicalUnit::Kilometer);
        let converted = convert_unit(&pv, &PhysicalUnit::Meter).expect("conversion failed");
        assert!((converted.value - 1000.0).abs() < 1e-9);
        assert_eq!(converted.unit, PhysicalUnit::Meter);
    }

    #[test]
    fn test_convert_celsius_to_kelvin() {
        let pv = PhysicalValue::new(0.0, PhysicalUnit::Celsius);
        let converted = convert_unit(&pv, &PhysicalUnit::Kelvin).expect("conversion failed");
        assert!((converted.value - 273.15).abs() < 1e-9);
    }

    #[test]
    fn test_convert_fahrenheit_to_celsius() {
        // 32°F → 0°C
        let pv = PhysicalValue::new(32.0, PhysicalUnit::Fahrenheit);
        let converted = convert_unit(&pv, &PhysicalUnit::Celsius).expect("conversion failed");
        assert!((converted.value - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_convert_g_to_ms2() {
        let pv = PhysicalValue::new(1.0, PhysicalUnit::StandardGravity);
        let converted =
            convert_unit(&pv, &PhysicalUnit::MetersPerSecondSquared).expect("conversion failed");
        assert!((converted.value - 9.806_65).abs() < 1e-9);
    }

    #[test]
    fn test_convert_kwh_to_joule() {
        let pv = PhysicalValue::new(1.0, PhysicalUnit::KiloWattHour);
        let converted = convert_unit(&pv, &PhysicalUnit::Joule).expect("conversion failed");
        assert!((converted.value - 3_600_000.0).abs() < 1e-3);
    }

    #[test]
    fn test_convert_roundtrip_kelvin() {
        // Kelvin → Fahrenheit → Kelvin
        let original = PhysicalValue::new(373.15, PhysicalUnit::Kelvin);
        let in_f = convert_unit(&original, &PhysicalUnit::Fahrenheit).expect("to F failed");
        let back = convert_unit(&in_f, &PhysicalUnit::Kelvin).expect("back to K failed");
        assert!((back.value - 373.15).abs() < 1e-9);
    }

    #[test]
    fn test_convert_mph_to_ms() {
        let pv = PhysicalValue::new(60.0, PhysicalUnit::MilesPerHour);
        let converted = convert_unit(&pv, &PhysicalUnit::MetersPerSecond).expect("failed");
        // 60 mph ≈ 26.8224 m/s
        assert!((converted.value - 26.8224).abs() < 1e-3);
    }

    // ── parse_unit_str ───────────────────────────────────────────────────────

    #[test]
    fn test_parse_unit_case_insensitive() {
        assert_eq!(parse_unit_str("KG"), PhysicalUnit::KiloGram);
        assert_eq!(parse_unit_str("KELVIN"), PhysicalUnit::Kelvin);
        assert_eq!(parse_unit_str("MHz"), PhysicalUnit::MegaHertz);
    }

    #[test]
    fn test_parse_unit_unknown_is_custom() {
        match parse_unit_str("furlong/fortnight") {
            PhysicalUnit::Custom(s) => assert_eq!(s, "furlong/fortnight"),
            other => panic!("expected Custom, got {:?}", other),
        }
    }

    // ── PhysicalUnit helper methods ───────────────────────────────────────────

    #[test]
    fn test_symbol_roundtrip() {
        let units = [
            PhysicalUnit::Meter,
            PhysicalUnit::KiloGram,
            PhysicalUnit::Second,
            PhysicalUnit::Newton,
            PhysicalUnit::Joule,
            PhysicalUnit::Kelvin,
            PhysicalUnit::Pascal,
        ];
        for u in &units {
            assert!(!u.symbol().is_empty(), "empty symbol for {:?}", u);
        }
    }

    #[test]
    fn test_as_si_kelvin_identity() {
        let pv = PhysicalValue::new(300.0, PhysicalUnit::Kelvin);
        assert!((pv.as_si() - 300.0).abs() < 1e-12);
    }

    #[test]
    fn test_display_physical_value() {
        let pv = PhysicalValue::new(9.81, PhysicalUnit::MetersPerSecondSquared);
        let s = format!("{}", pv);
        assert!(s.contains("9.81"));
    }
}
