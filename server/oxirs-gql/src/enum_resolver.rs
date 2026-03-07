//! GraphQL enum type resolution and coercion.
//!
//! Provides type-safe enum value definitions, case-insensitive coercion,
//! deprecation tracking, and a registry for schema-wide enum management.

use std::collections::HashMap;

/// A single enum value definition within a GraphQL enum type.
#[derive(Debug, Clone, PartialEq)]
pub struct EnumValue {
    /// The canonical name of this enum value (e.g. `"ACTIVE"`).
    pub name: String,
    /// Optional human-readable description of this value.
    pub description: Option<String>,
    /// Whether this value has been deprecated and should not be used in new queries.
    pub is_deprecated: bool,
}

impl EnumValue {
    /// Create a new non-deprecated enum value with no description.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            is_deprecated: false,
        }
    }

    /// Create a new enum value with a description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Mark this value as deprecated.
    pub fn deprecated(mut self) -> Self {
        self.is_deprecated = true;
        self
    }
}

/// An enum type definition containing its name and ordered set of values.
///
/// # Examples
///
/// ```rust
/// use oxirs_gql::enum_resolver::EnumType;
///
/// let status = EnumType::new("Status")
///     .add_value("ACTIVE")
///     .add_value_with_desc("INACTIVE", "Not currently in use")
///     .deprecate("INACTIVE");
///
/// assert!(status.is_valid("ACTIVE"));
/// assert!(!status.is_valid("UNKNOWN"));
/// ```
#[derive(Debug, Clone)]
pub struct EnumType {
    /// The canonical GraphQL name of this enum type.
    pub name: String,
    /// Ordered list of values belonging to this enum.
    pub values: Vec<EnumValue>,
}

impl EnumType {
    /// Create a new empty enum type with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            values: Vec::new(),
        }
    }

    /// Add a value with no description (builder-style).
    pub fn add_value(mut self, name: &str) -> Self {
        self.values.push(EnumValue::new(name));
        self
    }

    /// Add a value with a description (builder-style).
    pub fn add_value_with_desc(mut self, name: &str, desc: &str) -> Self {
        self.values
            .push(EnumValue::new(name).with_description(desc));
        self
    }

    /// Mark an existing value as deprecated (builder-style).
    ///
    /// If no value with the given name exists, this is a no-op.
    pub fn deprecate(mut self, name: &str) -> Self {
        for v in &mut self.values {
            if v.name == name {
                v.is_deprecated = true;
                break;
            }
        }
        self
    }

    /// Return `true` if a value with the given exact name (case-sensitive) exists.
    pub fn is_valid(&self, value: &str) -> bool {
        self.values.iter().any(|v| v.name == value)
    }

    /// Perform a case-insensitive lookup of a value by name.
    ///
    /// Returns a reference to the matching `EnumValue` (including deprecated ones)
    /// or `None` if no value matches.
    pub fn coerce(&self, value: &str) -> Option<&EnumValue> {
        let lower = value.to_ascii_lowercase();
        self.values
            .iter()
            .find(|v| v.name.to_ascii_lowercase() == lower)
    }

    /// Return all non-deprecated values.
    pub fn active_values(&self) -> Vec<&EnumValue> {
        self.values.iter().filter(|v| !v.is_deprecated).collect()
    }
}

/// Registry of all enum types declared in a GraphQL schema.
///
/// # Examples
///
/// ```rust
/// use oxirs_gql::enum_resolver::{EnumRegistry, EnumType};
///
/// let mut registry = EnumRegistry::new();
/// registry.register(EnumType::new("Status").add_value("ACTIVE").add_value("INACTIVE"));
///
/// assert!(registry.validate("Status", "ACTIVE").is_ok());
/// assert!(registry.validate("Status", "UNKNOWN").is_err());
/// ```
#[derive(Debug, Default)]
pub struct EnumRegistry {
    types: HashMap<String, EnumType>,
}

impl EnumRegistry {
    /// Create a new empty enum registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an enum type. Overwrites any previously registered type with the same name.
    pub fn register(&mut self, enum_type: EnumType) {
        self.types.insert(enum_type.name.clone(), enum_type);
    }

    /// Look up an enum type by name, returning `None` if not found.
    pub fn get(&self, name: &str) -> Option<&EnumType> {
        self.types.get(name)
    }

    /// Validate that `value` is a valid (case-sensitive) member of the enum `type_name`.
    ///
    /// Returns `Err(String)` with a descriptive message when:
    /// - the enum type is not registered, or
    /// - the value does not exist in the enum.
    pub fn validate(&self, type_name: &str, value: &str) -> Result<(), String> {
        match self.types.get(type_name) {
            None => Err(format!("Unknown enum type: '{}'", type_name)),
            Some(enum_type) => {
                if enum_type.is_valid(value) {
                    Ok(())
                } else {
                    Err(format!(
                        "'{}' is not a valid value for enum '{}'",
                        value, type_name
                    ))
                }
            }
        }
    }

    /// Return the names of all registered enum types in an unspecified order.
    pub fn list_types(&self) -> Vec<&str> {
        self.types.keys().map(String::as_str).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- EnumValue ---

    #[test]
    fn test_enum_value_new_defaults() {
        let v = EnumValue::new("ACTIVE");
        assert_eq!(v.name, "ACTIVE");
        assert!(v.description.is_none());
        assert!(!v.is_deprecated);
    }

    #[test]
    fn test_enum_value_with_description() {
        let v = EnumValue::new("ACTIVE").with_description("Currently active");
        assert_eq!(v.description.as_deref(), Some("Currently active"));
    }

    #[test]
    fn test_enum_value_deprecated() {
        let v = EnumValue::new("OLD").deprecated();
        assert!(v.is_deprecated);
    }

    #[test]
    fn test_enum_value_clone() {
        let v1 = EnumValue::new("X").with_description("desc").deprecated();
        let v2 = v1.clone();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_enum_value_equality() {
        let v1 = EnumValue::new("A");
        let v2 = EnumValue::new("A");
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_enum_value_inequality() {
        let v1 = EnumValue::new("A");
        let v2 = EnumValue::new("B");
        assert_ne!(v1, v2);
    }

    // --- EnumType: construction ---

    #[test]
    fn test_enum_type_new_empty() {
        let e = EnumType::new("MyEnum");
        assert_eq!(e.name, "MyEnum");
        assert!(e.values.is_empty());
    }

    #[test]
    fn test_add_value_single() {
        let e = EnumType::new("E").add_value("A");
        assert_eq!(e.values.len(), 1);
        assert_eq!(e.values[0].name, "A");
    }

    #[test]
    fn test_add_value_multiple() {
        let e = EnumType::new("E")
            .add_value("A")
            .add_value("B")
            .add_value("C");
        assert_eq!(e.values.len(), 3);
    }

    #[test]
    fn test_add_value_with_desc() {
        let e = EnumType::new("E").add_value_with_desc("X", "An X value");
        assert_eq!(e.values[0].description.as_deref(), Some("An X value"));
    }

    #[test]
    fn test_add_value_no_desc_by_default() {
        let e = EnumType::new("E").add_value("Y");
        assert!(e.values[0].description.is_none());
    }

    // --- EnumType: deprecate ---

    #[test]
    fn test_deprecate_existing_value() {
        let e = EnumType::new("E").add_value("OLD").deprecate("OLD");
        assert!(e.values[0].is_deprecated);
    }

    #[test]
    fn test_deprecate_nonexistent_is_noop() {
        let e = EnumType::new("E").add_value("A").deprecate("MISSING");
        assert!(!e.values[0].is_deprecated);
        assert_eq!(e.values.len(), 1);
    }

    #[test]
    fn test_deprecate_only_target_value() {
        let e = EnumType::new("E")
            .add_value("A")
            .add_value("OLD")
            .add_value("B")
            .deprecate("OLD");
        assert!(!e.values[0].is_deprecated); // A
        assert!(e.values[1].is_deprecated); // OLD
        assert!(!e.values[2].is_deprecated); // B
    }

    // --- EnumType: is_valid ---

    #[test]
    fn test_is_valid_existing() {
        let e = EnumType::new("E").add_value("ACTIVE");
        assert!(e.is_valid("ACTIVE"));
    }

    #[test]
    fn test_is_valid_missing() {
        let e = EnumType::new("E").add_value("ACTIVE");
        assert!(!e.is_valid("INACTIVE"));
    }

    #[test]
    fn test_is_valid_case_sensitive() {
        let e = EnumType::new("E").add_value("ACTIVE");
        assert!(!e.is_valid("active"));
        assert!(!e.is_valid("Active"));
    }

    #[test]
    fn test_is_valid_deprecated_still_valid() {
        let e = EnumType::new("E").add_value("OLD").deprecate("OLD");
        // is_valid does not exclude deprecated values
        assert!(e.is_valid("OLD"));
    }

    // --- EnumType: coerce ---

    #[test]
    fn test_coerce_exact_match() {
        let e = EnumType::new("E").add_value("ACTIVE");
        let v = e.coerce("ACTIVE");
        assert!(v.is_some());
        assert_eq!(v.unwrap().name, "ACTIVE");
    }

    #[test]
    fn test_coerce_lowercase_input() {
        let e = EnumType::new("E").add_value("ACTIVE");
        let v = e.coerce("active");
        assert!(v.is_some());
        assert_eq!(v.unwrap().name, "ACTIVE");
    }

    #[test]
    fn test_coerce_mixed_case_input() {
        let e = EnumType::new("E").add_value("ACTIVE");
        let v = e.coerce("AcTiVe");
        assert!(v.is_some());
    }

    #[test]
    fn test_coerce_missing_value() {
        let e = EnumType::new("E").add_value("ACTIVE");
        assert!(e.coerce("UNKNOWN").is_none());
    }

    #[test]
    fn test_coerce_deprecated_value_found() {
        let e = EnumType::new("E").add_value("OLD").deprecate("OLD");
        let v = e.coerce("old");
        assert!(v.is_some());
        assert!(v.unwrap().is_deprecated);
    }

    // --- EnumType: active_values ---

    #[test]
    fn test_active_values_all_active() {
        let e = EnumType::new("E")
            .add_value("A")
            .add_value("B")
            .add_value("C");
        assert_eq!(e.active_values().len(), 3);
    }

    #[test]
    fn test_active_values_excludes_deprecated() {
        let e = EnumType::new("E")
            .add_value("A")
            .add_value("OLD")
            .add_value("B")
            .deprecate("OLD");
        let active = e.active_values();
        assert_eq!(active.len(), 2);
        assert!(active.iter().all(|v| v.name != "OLD"));
    }

    #[test]
    fn test_active_values_empty_when_all_deprecated() {
        let e = EnumType::new("E")
            .add_value("A")
            .add_value("B")
            .deprecate("A")
            .deprecate("B");
        assert!(e.active_values().is_empty());
    }

    // --- EnumRegistry ---

    #[test]
    fn test_registry_new_empty() {
        let r = EnumRegistry::new();
        assert!(r.list_types().is_empty());
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut r = EnumRegistry::new();
        r.register(EnumType::new("Status").add_value("ACTIVE"));
        let e = r.get("Status");
        assert!(e.is_some());
        assert_eq!(e.unwrap().name, "Status");
    }

    #[test]
    fn test_registry_get_unknown() {
        let r = EnumRegistry::new();
        assert!(r.get("NonExistent").is_none());
    }

    #[test]
    fn test_registry_register_overwrites() {
        let mut r = EnumRegistry::new();
        r.register(EnumType::new("E").add_value("A"));
        r.register(EnumType::new("E").add_value("B"));
        let e = r.get("E").unwrap();
        assert_eq!(e.values.len(), 1);
        assert_eq!(e.values[0].name, "B");
    }

    #[test]
    fn test_registry_validate_ok() {
        let mut r = EnumRegistry::new();
        r.register(EnumType::new("Status").add_value("ACTIVE"));
        assert!(r.validate("Status", "ACTIVE").is_ok());
    }

    #[test]
    fn test_registry_validate_invalid_value() {
        let mut r = EnumRegistry::new();
        r.register(EnumType::new("Status").add_value("ACTIVE"));
        let result = r.validate("Status", "UNKNOWN");
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("UNKNOWN"));
        assert!(msg.contains("Status"));
    }

    #[test]
    fn test_registry_validate_unknown_type() {
        let r = EnumRegistry::new();
        let result = r.validate("NonExistent", "VALUE");
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("NonExistent"));
    }

    #[test]
    fn test_registry_list_types_all_registered() {
        let mut r = EnumRegistry::new();
        r.register(EnumType::new("A").add_value("X"));
        r.register(EnumType::new("B").add_value("Y"));
        r.register(EnumType::new("C").add_value("Z"));
        let mut types = r.list_types();
        types.sort();
        assert_eq!(types, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_registry_list_types_empty() {
        let r = EnumRegistry::new();
        assert!(r.list_types().is_empty());
    }

    #[test]
    fn test_registry_multiple_enums_independent() {
        let mut r = EnumRegistry::new();
        r.register(
            EnumType::new("Status")
                .add_value("ACTIVE")
                .add_value("INACTIVE"),
        );
        r.register(EnumType::new("Role").add_value("ADMIN").add_value("USER"));
        assert!(r.validate("Status", "ACTIVE").is_ok());
        assert!(r.validate("Role", "ADMIN").is_ok());
        assert!(r.validate("Status", "ADMIN").is_err());
        assert!(r.validate("Role", "ACTIVE").is_err());
    }

    // --- Debug format ---

    #[test]
    fn test_enum_type_debug() {
        let e = EnumType::new("E").add_value("A");
        let dbg = format!("{:?}", e);
        assert!(dbg.contains("EnumType"));
    }

    #[test]
    fn test_registry_debug() {
        let r = EnumRegistry::new();
        let dbg = format!("{:?}", r);
        assert!(dbg.contains("EnumRegistry"));
    }

    // --- Edge cases ---

    #[test]
    fn test_enum_type_preserves_value_order() {
        let e = EnumType::new("E")
            .add_value("FIRST")
            .add_value("SECOND")
            .add_value("THIRD");
        assert_eq!(e.values[0].name, "FIRST");
        assert_eq!(e.values[1].name, "SECOND");
        assert_eq!(e.values[2].name, "THIRD");
    }

    #[test]
    fn test_coerce_empty_enum() {
        let e = EnumType::new("Empty");
        assert!(e.coerce("anything").is_none());
    }

    #[test]
    fn test_is_valid_empty_string() {
        let e = EnumType::new("E").add_value("A");
        assert!(!e.is_valid(""));
    }

    #[test]
    fn test_validate_empty_string_value() {
        let mut r = EnumRegistry::new();
        r.register(EnumType::new("E").add_value("A"));
        assert!(r.validate("E", "").is_err());
    }

    #[test]
    fn test_clone_enum_type() {
        let e1 = EnumType::new("E").add_value("A").add_value("B");
        let mut e2 = e1.clone();
        e2.values.push(EnumValue::new("C"));
        assert_eq!(e1.values.len(), 2);
        assert_eq!(e2.values.len(), 3);
    }
}
