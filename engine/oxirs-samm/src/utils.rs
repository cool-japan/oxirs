//! Utility functions for common SAMM operations
//!
//! This module provides helper functions that simplify common tasks when working
//! with SAMM models, including URN manipulation, naming conventions, and data type handling.

use crate::error::{Result, SammError};
use crate::metamodel::{Aspect, Characteristic, ModelElement, Property};

/// URN manipulation utilities
pub mod urn {
    use super::*;

    /// Extract the namespace from a SAMM URN
    ///
    /// Uses SIMD-accelerated extraction for optimal performance.
    ///
    /// # Example
    /// ```
    /// use oxirs_samm::utils::urn::extract_namespace;
    ///
    /// let namespace = extract_namespace("urn:samm:org.eclipse.examples:1.0.0#Movement").unwrap();
    /// assert_eq!(namespace, "org.eclipse.examples");
    /// ```
    pub fn extract_namespace(urn: &str) -> Result<String> {
        validate_urn(urn)?;

        // Use SIMD-accelerated extraction when possible
        if let Some(namespace) = crate::simd_ops::extract_namespace_fast(urn) {
            return Ok(namespace.to_string());
        }

        // Fallback to manual parsing
        let without_prefix = urn
            .strip_prefix("urn:samm:")
            .ok_or_else(|| SammError::InvalidUrn(format!("Invalid URN format: {}", urn)))?;

        let parts: Vec<&str> = without_prefix.split('#').collect();
        let namespace_version = parts
            .first()
            .ok_or_else(|| SammError::InvalidUrn("Missing namespace".to_string()))?;

        let nv_parts: Vec<&str> = namespace_version.split(':').collect();
        Ok(nv_parts
            .first()
            .ok_or_else(|| SammError::InvalidUrn("Missing namespace".to_string()))?
            .to_string())
    }

    /// Extract the version from a SAMM URN
    ///
    /// Uses SIMD-accelerated extraction for optimal performance.
    ///
    /// # Example
    /// ```
    /// use oxirs_samm::utils::urn::extract_version;
    ///
    /// let version = extract_version("urn:samm:org.eclipse.examples:1.0.0#Movement").unwrap();
    /// assert_eq!(version, "1.0.0");
    /// ```
    pub fn extract_version(urn: &str) -> Result<String> {
        validate_urn(urn)?;

        // Use SIMD-accelerated extraction when possible
        if let Some(version) = crate::simd_ops::extract_version_fast(urn) {
            return Ok(version.to_string());
        }

        // Fallback to manual parsing
        let without_prefix = urn
            .strip_prefix("urn:samm:")
            .ok_or_else(|| SammError::InvalidUrn(format!("Invalid URN format: {}", urn)))?;

        let parts: Vec<&str> = without_prefix.split('#').collect();
        let namespace_version = parts
            .first()
            .ok_or_else(|| SammError::InvalidUrn("Missing version".to_string()))?;

        let nv_parts: Vec<&str> = namespace_version.split(':').collect();
        Ok(nv_parts
            .get(1)
            .ok_or_else(|| SammError::InvalidUrn("Missing version".to_string()))?
            .to_string())
    }

    /// Extract the element name from a SAMM URN
    ///
    /// Uses SIMD-accelerated extraction for optimal performance.
    ///
    /// # Example
    /// ```
    /// use oxirs_samm::utils::urn::extract_element;
    ///
    /// let element = extract_element("urn:samm:org.eclipse.examples:1.0.0#Movement").unwrap();
    /// assert_eq!(element, "Movement");
    /// ```
    pub fn extract_element(urn: &str) -> Result<String> {
        validate_urn(urn)?;

        // Use SIMD-accelerated extraction when possible
        if let Some(element) = crate::simd_ops::extract_element_fast(urn) {
            return Ok(element.to_string());
        }

        // Fallback to manual parsing
        let parts: Vec<&str> = urn.split('#').collect();
        Ok(parts
            .get(1)
            .ok_or_else(|| SammError::InvalidUrn("Missing element name".to_string()))?
            .to_string())
    }

    /// Build a SAMM URN from its components
    ///
    /// # Example
    /// ```
    /// use oxirs_samm::utils::urn::build_urn;
    ///
    /// let urn = build_urn("org.eclipse.examples", "1.0.0", "Movement");
    /// assert_eq!(urn, "urn:samm:org.eclipse.examples:1.0.0#Movement");
    /// ```
    pub fn build_urn(namespace: &str, version: &str, element: &str) -> String {
        format!("urn:samm:{}:{}#{}", namespace, version, element)
    }

    /// Validate a SAMM URN format
    ///
    /// Returns `Ok(())` if the URN is valid, otherwise returns an error with details.
    pub fn validate_urn(urn: &str) -> Result<()> {
        if !urn.starts_with("urn:samm:") {
            return Err(SammError::InvalidUrn(format!(
                "URN must start with 'urn:samm:', got: {}",
                urn
            )));
        }

        let parts: Vec<&str> = urn.split('#').collect();
        if parts.len() != 2 {
            return Err(SammError::InvalidUrn(format!(
                "URN must contain exactly one '#' separator, got: {}",
                urn
            )));
        }

        let namespace_version = parts[0].strip_prefix("urn:samm:").unwrap();
        let nv_parts: Vec<&str> = namespace_version.split(':').collect();
        if nv_parts.len() != 2 {
            return Err(SammError::InvalidUrn(format!(
                "URN must contain namespace and version separated by ':', got: {}",
                urn
            )));
        }

        if parts[1].is_empty() {
            return Err(SammError::InvalidUrn(
                "Element name cannot be empty".to_string(),
            ));
        }

        Ok(())
    }

    /// Check if two URNs refer to the same element
    pub fn is_same_element(urn1: &str, urn2: &str) -> bool {
        urn1 == urn2
    }

    /// Check if two URNs are in the same namespace
    pub fn is_same_namespace(urn1: &str, urn2: &str) -> Result<bool> {
        let ns1 = extract_namespace(urn1)?;
        let ns2 = extract_namespace(urn2)?;
        Ok(ns1 == ns2)
    }
}

/// Naming convention utilities
pub mod naming {
    /// Convert PascalCase to camelCase
    ///
    /// # Example
    /// ```
    /// use oxirs_samm::utils::naming::to_camel_case;
    ///
    /// assert_eq!(to_camel_case("Movement"), "movement");
    /// assert_eq!(to_camel_case("IsMoving"), "isMoving");
    /// ```
    pub fn to_camel_case(s: &str) -> String {
        if s.is_empty() {
            return String::new();
        }

        let mut chars = s.chars();
        let first = chars.next().unwrap().to_lowercase().to_string();
        first + chars.as_str()
    }

    /// Convert camelCase to PascalCase
    ///
    /// # Example
    /// ```
    /// use oxirs_samm::utils::naming::to_pascal_case;
    ///
    /// assert_eq!(to_pascal_case("movement"), "Movement");
    /// assert_eq!(to_pascal_case("isMoving"), "IsMoving");
    /// ```
    pub fn to_pascal_case(s: &str) -> String {
        if s.is_empty() {
            return String::new();
        }

        let mut chars = s.chars();
        let first = chars.next().unwrap().to_uppercase().to_string();
        first + chars.as_str()
    }

    /// Convert to snake_case
    ///
    /// # Example
    /// ```
    /// use oxirs_samm::utils::naming::to_snake_case;
    ///
    /// assert_eq!(to_snake_case("Movement"), "movement");
    /// assert_eq!(to_snake_case("IsMoving"), "is_moving");
    /// assert_eq!(to_snake_case("HTTPResponse"), "http_response");
    /// ```
    pub fn to_snake_case(s: &str) -> String {
        let mut result = String::new();
        let mut chars = s.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch.is_uppercase() {
                if !result.is_empty() && chars.peek().is_some_and(|c| c.is_lowercase()) {
                    result.push('_');
                }
                result.push(ch.to_lowercase().next().unwrap());
            } else {
                result.push(ch);
            }
        }

        result
    }

    /// Validate that a name follows SAMM naming conventions
    ///
    /// Properties should be camelCase, Characteristics and Entities should be PascalCase
    pub fn is_valid_property_name(name: &str) -> bool {
        if name.is_empty() {
            return false;
        }

        let first_char = name.chars().next().unwrap();
        if !first_char.is_lowercase() && !first_char.is_ascii_digit() {
            return false;
        }

        name.chars().all(|c| c.is_alphanumeric())
    }

    /// Validate that a name follows PascalCase convention
    pub fn is_valid_characteristic_name(name: &str) -> bool {
        if name.is_empty() {
            return false;
        }

        let first_char = name.chars().next().unwrap();
        if !first_char.is_uppercase() {
            return false;
        }

        name.chars().all(|c| c.is_alphanumeric())
    }
}

/// Model inspection utilities
pub mod inspection {
    use super::*;

    /// Get all property names from an Aspect
    ///
    /// # Example
    /// ```no_run
    /// use oxirs_samm::utils::inspection::get_property_names;
    /// # use oxirs_samm::metamodel::Aspect;
    /// # fn example(aspect: &Aspect) {
    /// let names = get_property_names(aspect);
    /// println!("Properties: {:?}", names);
    /// # }
    /// ```
    pub fn get_property_names(aspect: &Aspect) -> Vec<String> {
        aspect.properties().iter().map(|p| p.name()).collect()
    }

    /// Find a property by name in an Aspect
    pub fn find_property<'a>(aspect: &'a Aspect, name: &str) -> Option<&'a Property> {
        aspect.properties().iter().find(|p| p.name() == name)
    }

    /// Check if an Aspect has a specific property
    pub fn has_property(aspect: &Aspect, name: &str) -> bool {
        find_property(aspect, name).is_some()
    }

    /// Get all required properties (non-optional) from an Aspect
    pub fn get_required_properties(aspect: &Aspect) -> Vec<&Property> {
        aspect.properties().iter().filter(|p| !p.optional).collect()
    }

    /// Get all optional properties from an Aspect
    pub fn get_optional_properties(aspect: &Aspect) -> Vec<&Property> {
        aspect.properties().iter().filter(|p| p.optional).collect()
    }

    /// Count total properties in an Aspect
    pub fn count_properties(aspect: &Aspect) -> usize {
        aspect.properties().len()
    }

    /// Get the data type of a property's characteristic
    pub fn get_property_data_type(property: &Property) -> Option<String> {
        property
            .characteristic
            .as_ref()
            .and_then(|c| c.data_type.clone())
    }

    /// Check if a characteristic is a collection type
    pub fn is_collection_characteristic(characteristic: &Characteristic) -> bool {
        matches!(
            characteristic.kind(),
            crate::metamodel::CharacteristicKind::Collection { .. }
                | crate::metamodel::CharacteristicKind::List { .. }
                | crate::metamodel::CharacteristicKind::Set { .. }
                | crate::metamodel::CharacteristicKind::SortedSet { .. }
        )
    }
}

/// Data type utilities
pub mod datatypes {
    /// Check if an XSD type is numeric
    pub fn is_numeric_type(xsd_type: &str) -> bool {
        matches!(
            xsd_type,
            "xsd:int"
                | "xsd:integer"
                | "xsd:long"
                | "xsd:short"
                | "xsd:byte"
                | "xsd:float"
                | "xsd:double"
                | "xsd:decimal"
                | "xsd:positiveInteger"
                | "xsd:negativeInteger"
                | "xsd:nonNegativeInteger"
                | "xsd:nonPositiveInteger"
                | "xsd:unsignedLong"
                | "xsd:unsignedInt"
                | "xsd:unsignedShort"
                | "xsd:unsignedByte"
        )
    }

    /// Check if an XSD type is a string type
    pub fn is_string_type(xsd_type: &str) -> bool {
        matches!(
            xsd_type,
            "xsd:string" | "xsd:token" | "xsd:normalizedString"
        )
    }

    /// Check if an XSD type is a date/time type
    pub fn is_datetime_type(xsd_type: &str) -> bool {
        matches!(
            xsd_type,
            "xsd:date" | "xsd:dateTime" | "xsd:time" | "xsd:gYear" | "xsd:gYearMonth"
        )
    }

    /// Check if an XSD type is a boolean
    pub fn is_boolean_type(xsd_type: &str) -> bool {
        xsd_type == "xsd:boolean"
    }

    /// Get the Rust equivalent type for an XSD type
    pub fn xsd_to_rust_type(xsd_type: &str) -> &'static str {
        match xsd_type {
            "xsd:int" | "xsd:integer" => "i32",
            "xsd:long" => "i64",
            "xsd:short" => "i16",
            "xsd:byte" => "i8",
            "xsd:unsignedInt" => "u32",
            "xsd:unsignedLong" => "u64",
            "xsd:unsignedShort" => "u16",
            "xsd:unsignedByte" => "u8",
            "xsd:float" => "f32",
            "xsd:double" | "xsd:decimal" => "f64",
            "xsd:boolean" => "bool",
            "xsd:string" | "xsd:token" | "xsd:normalizedString" => "String",
            "xsd:dateTime" | "xsd:date" | "xsd:time" => "String", // Would use chrono in real code
            _ => "String",                                        // Default fallback
        }
    }
}

/// Model statistics utilities
pub mod statistics {
    use super::*;

    /// Model statistics
    #[derive(Debug, Clone)]
    pub struct ModelStatistics {
        /// Total number of properties
        pub total_properties: usize,
        /// Number of required properties
        pub required_properties: usize,
        /// Number of optional properties
        pub optional_properties: usize,
        /// Number of operations
        pub total_operations: usize,
        /// Number of events
        pub total_events: usize,
        /// Properties with characteristics
        pub properties_with_characteristics: usize,
        /// Properties with example values
        pub properties_with_examples: usize,
    }

    /// Calculate comprehensive statistics for an aspect model
    ///
    /// # Example
    /// ```
    /// use oxirs_samm::metamodel::{Aspect, Property};
    /// use oxirs_samm::utils::statistics::calculate_statistics;
    ///
    /// let mut aspect = Aspect::new("urn:samm:test:1.0.0#Test".to_string());
    /// aspect.add_property(Property::new("urn:samm:test:1.0.0#prop1".to_string()));
    ///
    /// let stats = calculate_statistics(&aspect);
    /// assert_eq!(stats.total_properties, 1);
    /// assert_eq!(stats.required_properties, 1);
    /// ```
    pub fn calculate_statistics(aspect: &Aspect) -> ModelStatistics {
        let total_properties = aspect.properties.len();
        let required_properties = aspect.properties.iter().filter(|p| !p.optional).count();
        let optional_properties = aspect.properties.iter().filter(|p| p.optional).count();
        let properties_with_characteristics = aspect
            .properties
            .iter()
            .filter(|p| p.characteristic.is_some())
            .count();
        let properties_with_examples = aspect
            .properties
            .iter()
            .filter(|p| !p.example_values.is_empty())
            .count();

        ModelStatistics {
            total_properties,
            required_properties,
            optional_properties,
            total_operations: aspect.operations.len(),
            total_events: aspect.events.len(),
            properties_with_characteristics,
            properties_with_examples,
        }
    }

    /// Get the ratio of required to total properties
    pub fn required_ratio(aspect: &Aspect) -> f64 {
        let stats = calculate_statistics(aspect);
        if stats.total_properties == 0 {
            0.0
        } else {
            stats.required_properties as f64 / stats.total_properties as f64
        }
    }

    /// Get the ratio of optional to total properties
    pub fn optional_ratio(aspect: &Aspect) -> f64 {
        let stats = calculate_statistics(aspect);
        if stats.total_properties == 0 {
            0.0
        } else {
            stats.optional_properties as f64 / stats.total_properties as f64
        }
    }
}

/// Serialization utilities for quick export
pub mod serialization {
    use super::*;
    use serde_json;

    /// Serialize an aspect to a JSON string
    ///
    /// # Example
    /// ```
    /// use oxirs_samm::metamodel::Aspect;
    /// use oxirs_samm::utils::serialization::to_json_string;
    ///
    /// let aspect = Aspect::new("urn:samm:test:1.0.0#Test".to_string());
    ///
    /// let json = to_json_string(&aspect).unwrap();
    /// assert!(json.contains("Test"));
    /// ```
    pub fn to_json_string(aspect: &Aspect) -> Result<String> {
        serde_json::to_string(aspect)
            .map_err(|e| SammError::Other(format!("JSON serialization failed: {}", e)))
    }

    /// Serialize an aspect to pretty-printed JSON
    pub fn to_json_pretty(aspect: &Aspect) -> Result<String> {
        serde_json::to_string_pretty(aspect)
            .map_err(|e| SammError::Other(format!("JSON serialization failed: {}", e)))
    }

    /// Deserialize an aspect from a JSON string
    pub fn from_json_string(json: &str) -> Result<Aspect> {
        serde_json::from_str(json)
            .map_err(|e| SammError::Other(format!("JSON deserialization failed: {}", e)))
    }
}

/// Batch operation utilities for bulk modifications
pub mod batch {
    use super::*;

    /// Make all properties in an aspect optional
    ///
    /// # Example
    /// ```
    /// use oxirs_samm::metamodel::{Aspect, Property};
    /// use oxirs_samm::utils::batch::make_all_optional;
    ///
    /// let mut aspect = Aspect::new("urn:samm:test:1.0.0#Test".to_string());
    /// aspect.add_property(Property::new("urn:samm:test:1.0.0#prop1".to_string()));
    ///
    /// make_all_optional(&mut aspect);
    /// assert!(aspect.properties.iter().all(|p| p.optional));
    /// ```
    pub fn make_all_optional(aspect: &mut Aspect) {
        for property in &mut aspect.properties {
            property.optional = true;
        }
    }

    /// Make all properties in an aspect required
    pub fn make_all_required(aspect: &mut Aspect) {
        for property in &mut aspect.properties {
            property.optional = false;
        }
    }

    /// Set example values for all properties that don't have one
    pub fn set_example_values(aspect: &mut Aspect, default_value: &str) {
        for property in &mut aspect.properties {
            if property.example_values.is_empty() {
                property.example_values = vec![default_value.to_string()];
            }
        }
    }

    /// Remove all example values from properties
    pub fn clear_example_values(aspect: &mut Aspect) {
        for property in &mut aspect.properties {
            property.example_values.clear();
        }
    }

    /// Apply a transformation function to all properties
    pub fn apply_to_all_properties<F>(aspect: &mut Aspect, mut f: F)
    where
        F: FnMut(&mut Property),
    {
        for property in &mut aspect.properties {
            f(property);
        }
    }

    /// Filter properties by a predicate and return a new aspect
    pub fn filter_properties<F>(aspect: &Aspect, predicate: F) -> Aspect
    where
        F: Fn(&Property) -> bool,
    {
        let filtered_properties: Vec<Property> = aspect
            .properties
            .iter()
            .filter(|p| predicate(p))
            .cloned()
            .collect();

        Aspect {
            metadata: aspect.metadata.clone(),
            properties: filtered_properties,
            operations: aspect.operations.clone(),
            events: aspect.events.clone(),
        }
    }
}

/// Model cloning and merging utilities
pub mod merging {
    use super::*;
    use std::collections::HashSet;

    /// Merge two aspects, combining their properties
    ///
    /// Properties from the second aspect are added to the first.
    /// Duplicate properties (same URN) are kept from the first aspect.
    pub fn merge_aspects(first: &Aspect, second: &Aspect) -> Aspect {
        let mut merged = first.clone();

        // Track existing property URNs
        let existing_urns: HashSet<String> = merged
            .properties
            .iter()
            .map(|p| p.urn().to_owned())
            .collect();

        // Add properties from second that don't exist in first
        for property in &second.properties {
            if !existing_urns.contains(property.urn()) {
                merged.properties.push(property.clone());
            }
        }

        // Merge operations (same logic)
        let existing_op_urns: HashSet<String> = merged
            .operations
            .iter()
            .map(|o| o.urn().to_owned())
            .collect();

        for operation in &second.operations {
            if !existing_op_urns.contains(operation.urn()) {
                merged.operations.push(operation.clone());
            }
        }

        // Merge events
        let existing_event_urns: HashSet<String> =
            merged.events.iter().map(|e| e.urn().to_owned()).collect();

        for event in &second.events {
            if !existing_event_urns.contains(event.urn()) {
                merged.events.push(event.clone());
            }
        }

        merged
    }

    /// Deep clone an aspect
    pub fn deep_clone(aspect: &Aspect) -> Aspect {
        aspect.clone()
    }
}

/// Diff utilities for quick model comparison
pub mod diff {
    use super::*;

    /// Quick diff summary between two aspects
    #[derive(Debug, Clone)]
    pub struct QuickDiff {
        /// Number of properties added
        pub properties_added: usize,
        /// Number of properties removed
        pub properties_removed: usize,
        /// Number of properties modified
        pub properties_modified: usize,
        /// Number of operations added
        pub operations_added: usize,
        /// Number of operations removed
        pub operations_removed: usize,
    }

    /// Generate a quick diff summary between two aspects
    ///
    /// This is a lightweight alternative to full ModelComparison for quick checks.
    pub fn quick_diff(old: &Aspect, new: &Aspect) -> QuickDiff {
        use std::collections::HashSet;

        let old_prop_urns: HashSet<_> = old.properties.iter().map(|p| p.urn()).collect();
        let new_prop_urns: HashSet<_> = new.properties.iter().map(|p| p.urn()).collect();

        let properties_added = new_prop_urns.difference(&old_prop_urns).count();
        let properties_removed = old_prop_urns.difference(&new_prop_urns).count();

        // Count modified properties (intersection but with different content)
        let common_urns: Vec<_> = old_prop_urns.intersection(&new_prop_urns).collect();
        let mut properties_modified = 0;

        for urn in common_urns {
            let old_prop = old.properties.iter().find(|p| p.urn() == *urn).unwrap();
            let new_prop = new.properties.iter().find(|p| p.urn() == *urn).unwrap();

            if old_prop.optional != new_prop.optional
                || old_prop.example_values != new_prop.example_values
            {
                properties_modified += 1;
            }
        }

        let old_op_urns: HashSet<_> = old.operations.iter().map(|o| o.urn()).collect();
        let new_op_urns: HashSet<_> = new.operations.iter().map(|o| o.urn()).collect();

        let operations_added = new_op_urns.difference(&old_op_urns).count();
        let operations_removed = old_op_urns.difference(&new_op_urns).count();

        QuickDiff {
            properties_added,
            properties_removed,
            properties_modified,
            operations_added,
            operations_removed,
        }
    }

    /// Check if two aspects are identical
    pub fn are_identical(first: &Aspect, second: &Aspect) -> bool {
        let diff = quick_diff(first, second);
        diff.properties_added == 0
            && diff.properties_removed == 0
            && diff.properties_modified == 0
            && diff.operations_added == 0
            && diff.operations_removed == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_namespace() {
        let urn = "urn:samm:org.eclipse.examples:1.0.0#Movement";
        assert_eq!(urn::extract_namespace(urn).unwrap(), "org.eclipse.examples");
    }

    #[test]
    fn test_extract_version() {
        let urn = "urn:samm:org.eclipse.examples:1.0.0#Movement";
        assert_eq!(urn::extract_version(urn).unwrap(), "1.0.0");
    }

    #[test]
    fn test_extract_element() {
        let urn = "urn:samm:org.eclipse.examples:1.0.0#Movement";
        assert_eq!(urn::extract_element(urn).unwrap(), "Movement");
    }

    #[test]
    fn test_build_urn() {
        let urn = urn::build_urn("org.eclipse.examples", "1.0.0", "Movement");
        assert_eq!(urn, "urn:samm:org.eclipse.examples:1.0.0#Movement");
    }

    #[test]
    fn test_validate_urn_valid() {
        assert!(urn::validate_urn("urn:samm:org.eclipse:1.0.0#Test").is_ok());
    }

    #[test]
    fn test_validate_urn_invalid_prefix() {
        assert!(urn::validate_urn("urn:other:org.eclipse:1.0.0#Test").is_err());
    }

    #[test]
    fn test_validate_urn_missing_hash() {
        assert!(urn::validate_urn("urn:samm:org.eclipse:1.0.0").is_err());
    }

    #[test]
    fn test_is_same_namespace() {
        let urn1 = "urn:samm:org.eclipse:1.0.0#Test1";
        let urn2 = "urn:samm:org.eclipse:2.0.0#Test2";
        assert!(urn::is_same_namespace(urn1, urn2).unwrap());
    }

    #[test]
    fn test_to_camel_case() {
        assert_eq!(naming::to_camel_case("Movement"), "movement");
        assert_eq!(naming::to_camel_case("IsMoving"), "isMoving");
    }

    #[test]
    fn test_to_pascal_case() {
        assert_eq!(naming::to_pascal_case("movement"), "Movement");
        assert_eq!(naming::to_pascal_case("isMoving"), "IsMoving");
    }

    #[test]
    fn test_to_snake_case() {
        assert_eq!(naming::to_snake_case("Movement"), "movement");
        assert_eq!(naming::to_snake_case("IsMoving"), "is_moving");
        assert_eq!(naming::to_snake_case("HTTPResponse"), "http_response");
    }

    #[test]
    fn test_is_valid_property_name() {
        assert!(naming::is_valid_property_name("movement"));
        assert!(naming::is_valid_property_name("isMoving"));
        assert!(!naming::is_valid_property_name("Movement"));
        assert!(!naming::is_valid_property_name(""));
        assert!(!naming::is_valid_property_name("is-moving"));
    }

    #[test]
    fn test_is_valid_characteristic_name() {
        assert!(naming::is_valid_characteristic_name("Movement"));
        assert!(naming::is_valid_characteristic_name("IsMoving"));
        assert!(!naming::is_valid_characteristic_name("movement"));
        assert!(!naming::is_valid_characteristic_name(""));
    }

    #[test]
    fn test_is_numeric_type() {
        assert!(datatypes::is_numeric_type("xsd:int"));
        assert!(datatypes::is_numeric_type("xsd:float"));
        assert!(datatypes::is_numeric_type("xsd:double"));
        assert!(!datatypes::is_numeric_type("xsd:string"));
    }

    #[test]
    fn test_is_string_type() {
        assert!(datatypes::is_string_type("xsd:string"));
        assert!(datatypes::is_string_type("xsd:token"));
        assert!(!datatypes::is_string_type("xsd:int"));
    }

    #[test]
    fn test_is_datetime_type() {
        assert!(datatypes::is_datetime_type("xsd:date"));
        assert!(datatypes::is_datetime_type("xsd:dateTime"));
        assert!(!datatypes::is_datetime_type("xsd:string"));
    }

    #[test]
    fn test_is_boolean_type() {
        assert!(datatypes::is_boolean_type("xsd:boolean"));
        assert!(!datatypes::is_boolean_type("xsd:string"));
    }

    #[test]
    fn test_xsd_to_rust_type() {
        assert_eq!(datatypes::xsd_to_rust_type("xsd:int"), "i32");
        assert_eq!(datatypes::xsd_to_rust_type("xsd:long"), "i64");
        assert_eq!(datatypes::xsd_to_rust_type("xsd:double"), "f64");
        assert_eq!(datatypes::xsd_to_rust_type("xsd:boolean"), "bool");
        assert_eq!(datatypes::xsd_to_rust_type("xsd:string"), "String");
    }

    #[test]
    fn test_model_statistics() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#Test".to_string());
        aspect.add_property(Property::new("urn:samm:test:1.0.0#prop1".to_string()));
        aspect.add_property(Property::new("urn:samm:test:1.0.0#prop2".to_string()).as_optional());

        let stats = statistics::calculate_statistics(&aspect);
        assert_eq!(stats.total_properties, 2);
        assert_eq!(stats.required_properties, 1);
        assert_eq!(stats.optional_properties, 1);
    }

    #[test]
    fn test_serialization_to_json() {
        let aspect = Aspect::new("urn:samm:test:1.0.0#Test".to_string());

        let json = serialization::to_json_string(&aspect).unwrap();
        assert!(json.contains("Test"));
        assert!(json.contains("urn:samm:test:1.0.0"));
    }

    #[test]
    fn test_batch_operations() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#Test".to_string());
        aspect.add_property(Property::new("urn:samm:test:1.0.0#prop1".to_string()));
        aspect.add_property(Property::new("urn:samm:test:1.0.0#prop2".to_string()));

        batch::make_all_optional(&mut aspect);
        assert!(aspect.properties.iter().all(|p| p.optional));
    }
}
