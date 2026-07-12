//! Simple DTDL property value model for J1939 ↔ DTDL bridge integration.
//!
//! This module provides a lightweight, typed value container (`TwinValue`) and a
//! minimal digital twin property store (`Twin`) intended for use by protocol bridge
//! implementations such as `oxirs-canbus`'s J1939 ↔ DTDL bridge.
//!
//! Unlike the fuller [`DigitalTwin`](crate::digital_twin::DigitalTwin) (which tracks sensor/actuator state,
//! sync reports, and DTDL v2 interface definitions), `Twin` here is intentionally
//! minimal: it holds a flat map of named properties and a monotonically increasing
//! version counter that advances on every write.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// TwinValue
// ─────────────────────────────────────────────────────────────────────────────

/// A typed property value in a DTDL digital twin.
///
/// Covers the four primitive DTDL schemas most relevant to automotive telemetry:
/// - `double`/`float` → [`TwinValue::Float`]
/// - `integer`/`long`  → [`TwinValue::Integer`]
/// - `boolean`         → [`TwinValue::Boolean`]
/// - `string`          → [`TwinValue::Text`]
#[derive(Debug, Clone, PartialEq)]
pub enum TwinValue {
    /// A 64-bit floating-point value (DTDL `double` / `float`).
    Float(f64),
    /// A 64-bit signed integer value (DTDL `integer` / `long`).
    Integer(i64),
    /// A boolean value (DTDL `boolean`).
    Boolean(bool),
    /// A UTF-8 string value (DTDL `string`).
    Text(String),
}

impl TwinValue {
    /// Return the contained float, or `None` if this is a different variant.
    pub fn as_float(&self) -> Option<f64> {
        if let Self::Float(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Return the contained integer, or `None` if this is a different variant.
    pub fn as_integer(&self) -> Option<i64> {
        if let Self::Integer(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Return the contained boolean, or `None` if this is a different variant.
    pub fn as_boolean(&self) -> Option<bool> {
        if let Self::Boolean(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Return a reference to the contained string, or `None` if this is a different variant.
    pub fn as_text(&self) -> Option<&str> {
        if let Self::Text(v) = self {
            Some(v.as_str())
        } else {
            None
        }
    }
}

impl std::fmt::Display for TwinValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Float(v) => write!(f, "{v}"),
            Self::Integer(v) => write!(f, "{v}"),
            Self::Boolean(v) => write!(f, "{v}"),
            Self::Text(v) => write!(f, "{v}"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Twin
// ─────────────────────────────────────────────────────────────────────────────

/// A Digital Twin instance with named property read/write access.
///
/// This is a minimal, flat property store. Each call to [`Twin::set_property`]
/// increments the internal version counter, allowing callers to detect any
/// write without tracking individual property changes.
///
/// # Example
///
/// ```
/// use oxirs_physics::digital_twin::twin_value::{Twin, TwinValue};
///
/// let mut twin = Twin::new();
/// twin.set_property("engine.coolant_temp_c", TwinValue::Integer(75));
///
/// let val = twin.get_property("engine.coolant_temp_c").expect("property should exist");
/// assert_eq!(*val, TwinValue::Integer(75));
/// assert_eq!(twin.version(), 1);
/// ```
#[derive(Debug, Default)]
pub struct Twin {
    properties: HashMap<String, TwinValue>,
    version: u64,
}

impl Twin {
    /// Create an empty twin with version 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Write (or overwrite) a named property.
    ///
    /// Increments the version counter regardless of whether the value changed.
    pub fn set_property(&mut self, name: &str, value: TwinValue) {
        self.properties.insert(name.to_string(), value);
        self.version += 1;
    }

    /// Read a named property.
    ///
    /// Returns `None` when the property has never been written.
    pub fn get_property(&self, name: &str) -> Option<&TwinValue> {
        self.properties.get(name)
    }

    /// Remove a property, returning the previous value if it existed.
    pub fn remove_property(&mut self, name: &str) -> Option<TwinValue> {
        let removed = self.properties.remove(name);
        if removed.is_some() {
            self.version += 1;
        }
        removed
    }

    /// Monotonically increasing write counter.
    ///
    /// Starts at 0 and increments by 1 on every [`set_property`](Self::set_property)
    /// or [`remove_property`](Self::remove_property) that actually removes an entry.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Number of currently stored properties.
    pub fn len(&self) -> usize {
        self.properties.len()
    }

    /// Returns `true` when no properties have been set.
    pub fn is_empty(&self) -> bool {
        self.properties.is_empty()
    }

    /// Iterate over all (name, value) pairs in unspecified order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TwinValue)> {
        self.properties.iter().map(|(k, v)| (k.as_str(), v))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twin_default_is_empty() {
        let twin = Twin::new();
        assert!(twin.is_empty());
        assert_eq!(twin.len(), 0);
        assert_eq!(twin.version(), 0);
    }

    #[test]
    fn twin_set_and_get_float() {
        let mut twin = Twin::new();
        twin.set_property("temperature", TwinValue::Float(98.6));
        assert_eq!(
            twin.get_property("temperature"),
            Some(&TwinValue::Float(98.6))
        );
        assert_eq!(twin.version(), 1);
    }

    #[test]
    fn twin_set_and_get_integer() {
        let mut twin = Twin::new();
        twin.set_property("rpm", TwinValue::Integer(1500));
        assert_eq!(twin.get_property("rpm"), Some(&TwinValue::Integer(1500)));
    }

    #[test]
    fn twin_set_and_get_boolean() {
        let mut twin = Twin::new();
        twin.set_property("mil_on", TwinValue::Boolean(true));
        assert_eq!(twin.get_property("mil_on"), Some(&TwinValue::Boolean(true)));
    }

    #[test]
    fn twin_set_and_get_text() {
        let mut twin = Twin::new();
        twin.set_property("vin", TwinValue::Text("1HGCM82633A123456".to_string()));
        assert_eq!(
            twin.get_property("vin"),
            Some(&TwinValue::Text("1HGCM82633A123456".to_string()))
        );
    }

    #[test]
    fn twin_overwrite_increments_version_each_time() {
        let mut twin = Twin::new();
        twin.set_property("x", TwinValue::Integer(1));
        twin.set_property("x", TwinValue::Integer(2));
        assert_eq!(twin.version(), 2);
        assert_eq!(twin.get_property("x"), Some(&TwinValue::Integer(2)));
    }

    #[test]
    fn twin_missing_property_is_none() {
        let twin = Twin::new();
        assert!(twin.get_property("nonexistent").is_none());
    }

    #[test]
    fn twin_remove_decrements_len_and_increments_version() {
        let mut twin = Twin::new();
        twin.set_property("a", TwinValue::Integer(1));
        twin.set_property("b", TwinValue::Integer(2));
        assert_eq!(twin.len(), 2);
        let removed = twin.remove_property("a");
        assert_eq!(removed, Some(TwinValue::Integer(1)));
        assert_eq!(twin.len(), 1);
        assert_eq!(twin.version(), 3); // 2 sets + 1 remove
    }

    #[test]
    fn twin_remove_absent_property_does_not_increment_version() {
        let mut twin = Twin::new();
        twin.set_property("a", TwinValue::Integer(1));
        let removed = twin.remove_property("nosuchprop");
        assert!(removed.is_none());
        assert_eq!(twin.version(), 1); // only the initial set
    }

    #[test]
    fn twin_iter_covers_all_properties() {
        let mut twin = Twin::new();
        twin.set_property("p1", TwinValue::Integer(1));
        twin.set_property("p2", TwinValue::Boolean(false));
        let pairs: Vec<_> = twin.iter().collect();
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn twin_value_accessors() {
        assert_eq!(TwinValue::Float(1.5).as_float(), Some(1.5));
        assert_eq!(TwinValue::Integer(7).as_integer(), Some(7));
        assert_eq!(TwinValue::Boolean(true).as_boolean(), Some(true));
        assert_eq!(TwinValue::Text("hi".to_string()).as_text(), Some("hi"));

        // Cross-variant: returns None
        assert_eq!(TwinValue::Integer(7).as_float(), None);
        assert_eq!(TwinValue::Float(1.0).as_integer(), None);
        assert_eq!(TwinValue::Boolean(false).as_text(), None);
    }

    #[test]
    fn twin_value_display() {
        // Use 1.5 (not a recognized mathematical constant) to avoid the
        // clippy::approx_constant lint.
        assert_eq!(TwinValue::Float(1.5).to_string(), "1.5");
        assert_eq!(TwinValue::Integer(42).to_string(), "42");
        assert_eq!(TwinValue::Boolean(true).to_string(), "true");
        assert_eq!(TwinValue::Text("hello".to_string()).to_string(), "hello");
    }
}
