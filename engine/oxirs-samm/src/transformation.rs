//! Model Transformation Utilities
//!
//! This module provides powerful utilities for transforming SAMM Aspect Models.
//! Useful for model refactoring, namespace changes, property renaming, and model evolution.
//!
//! # Features
//!
//! - **Property renaming** - Bulk rename properties with pattern matching
//! - **Namespace transformation** - Change model namespaces
//! - **Version updates** - Update model versions
//! - **Property transformations** - Convert optional/required, change characteristics
//! - **Metadata updates** - Bulk update names and descriptions
//! - **Rule-based transformations** - Apply custom transformation rules
//!
//! # Examples
//!
//! ```rust
//! use oxirs_samm::transformation::{ModelTransformation, TransformationRule};
//! use oxirs_samm::metamodel::Aspect;
//!
//! # fn example(mut aspect: Aspect) {
//! // Create transformation builder
//! let mut transformation = ModelTransformation::new(&mut aspect);
//!
//! // Rename namespace
//! transformation.change_namespace("urn:samm:org.example:1.0.0", "urn:samm:org.example:2.0.0");
//!
//! // Rename a property
//! transformation.rename_property("oldName", "newName");
//!
//! // Make all properties optional
//! transformation.make_all_properties_optional();
//!
//! // Apply transformations
//! let result = transformation.apply();
//! println!("Applied {} transformations", result.transformations_applied);
//! # }
//! ```

use crate::metamodel::{Aspect, ModelElement, Property};
use std::collections::HashMap;

/// Transformation builder for SAMM Aspect Models
///
/// Provides a fluent API for applying transformations to models.
pub struct ModelTransformation<'a> {
    aspect: &'a mut Aspect,
    rules: Vec<TransformationRule>,
}

/// A transformation rule to apply to the model
#[derive(Debug, Clone, PartialEq)]
pub enum TransformationRule {
    /// Rename a property (old_name, new_name)
    RenameProperty {
        /// Old property name
        old_name: String,
        /// New property name
        new_name: String,
    },
    /// Change namespace (old_namespace, new_namespace)
    ChangeNamespace {
        /// Old namespace
        old_namespace: String,
        /// New namespace
        new_namespace: String,
    },
    /// Make property optional (property_name)
    MakePropertyOptional {
        /// Property name
        property_name: String,
    },
    /// Make property required (property_name)
    MakePropertyRequired {
        /// Property name
        property_name: String,
    },
    /// Update preferred name (language, new_name)
    UpdatePreferredName {
        /// Language code
        language: String,
        /// New preferred name
        new_name: String,
    },
    /// Update description (language, new_description)
    UpdateDescription {
        /// Language code
        language: String,
        /// New description
        new_description: String,
    },
    /// Replace URN pattern (pattern, replacement)
    ReplaceUrnPattern {
        /// Pattern to match
        pattern: String,
        /// Replacement string
        replacement: String,
    },
}

/// Result of applying transformations
#[derive(Debug, Clone, PartialEq)]
pub struct TransformationResult {
    /// Number of transformations applied
    pub transformations_applied: usize,
    /// Transformations that succeeded
    pub successful_transformations: Vec<String>,
    /// Transformations that failed
    pub failed_transformations: Vec<String>,
    /// Warnings generated during transformation
    pub warnings: Vec<String>,
}

impl<'a> ModelTransformation<'a> {
    /// Creates a new transformation builder for the given aspect
    ///
    /// # Arguments
    ///
    /// * `aspect` - Mutable reference to the aspect to transform
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_samm::transformation::ModelTransformation;
    /// # use oxirs_samm::metamodel::Aspect;
    /// # fn example(mut aspect: Aspect) {
    /// let mut transformation = ModelTransformation::new(&mut aspect);
    /// # }
    /// ```
    pub fn new(aspect: &'a mut Aspect) -> Self {
        Self {
            aspect,
            rules: Vec::new(),
        }
    }

    /// Renames a property in the model
    ///
    /// # Arguments
    ///
    /// * `old_name` - Current name of the property
    /// * `new_name` - New name for the property
    pub fn rename_property(
        &mut self,
        old_name: impl Into<String>,
        new_name: impl Into<String>,
    ) -> &mut Self {
        self.rules.push(TransformationRule::RenameProperty {
            old_name: old_name.into(),
            new_name: new_name.into(),
        });
        self
    }

    /// Changes the namespace of all model elements
    ///
    /// # Arguments
    ///
    /// * `old_namespace` - Current namespace
    /// * `new_namespace` - New namespace
    pub fn change_namespace(
        &mut self,
        old_namespace: impl Into<String>,
        new_namespace: impl Into<String>,
    ) -> &mut Self {
        self.rules.push(TransformationRule::ChangeNamespace {
            old_namespace: old_namespace.into(),
            new_namespace: new_namespace.into(),
        });
        self
    }

    /// Makes a specific property optional
    ///
    /// # Arguments
    ///
    /// * `property_name` - Name of the property to make optional
    pub fn make_property_optional(&mut self, property_name: impl Into<String>) -> &mut Self {
        self.rules.push(TransformationRule::MakePropertyOptional {
            property_name: property_name.into(),
        });
        self
    }

    /// Makes a specific property required
    ///
    /// # Arguments
    ///
    /// * `property_name` - Name of the property to make required
    pub fn make_property_required(&mut self, property_name: impl Into<String>) -> &mut Self {
        self.rules.push(TransformationRule::MakePropertyRequired {
            property_name: property_name.into(),
        });
        self
    }

    /// Updates the preferred name for a language
    ///
    /// # Arguments
    ///
    /// * `language` - Language code (e.g., "en", "de")
    /// * `new_name` - New preferred name
    pub fn update_preferred_name(
        &mut self,
        language: impl Into<String>,
        new_name: impl Into<String>,
    ) -> &mut Self {
        self.rules.push(TransformationRule::UpdatePreferredName {
            language: language.into(),
            new_name: new_name.into(),
        });
        self
    }

    /// Updates the description for a language
    ///
    /// # Arguments
    ///
    /// * `language` - Language code (e.g., "en", "de")
    /// * `new_description` - New description
    pub fn update_description(
        &mut self,
        language: impl Into<String>,
        new_description: impl Into<String>,
    ) -> &mut Self {
        self.rules.push(TransformationRule::UpdateDescription {
            language: language.into(),
            new_description: new_description.into(),
        });
        self
    }

    /// Replaces a pattern in all URNs
    ///
    /// # Arguments
    ///
    /// * `pattern` - Pattern to find in URNs
    /// * `replacement` - Replacement string
    pub fn replace_urn_pattern(
        &mut self,
        pattern: impl Into<String>,
        replacement: impl Into<String>,
    ) -> &mut Self {
        self.rules.push(TransformationRule::ReplaceUrnPattern {
            pattern: pattern.into(),
            replacement: replacement.into(),
        });
        self
    }

    /// Makes all properties in the model optional
    pub fn make_all_properties_optional(&mut self) -> &mut Self {
        let names: Vec<_> = self
            .aspect
            .properties()
            .iter()
            .map(|p| p.name().to_string())
            .collect();
        for name in names {
            self.make_property_optional(name);
        }
        self
    }

    /// Makes all properties in the model required
    pub fn make_all_properties_required(&mut self) -> &mut Self {
        let names: Vec<_> = self
            .aspect
            .properties()
            .iter()
            .map(|p| p.name().to_string())
            .collect();
        for name in names {
            self.make_property_required(name);
        }
        self
    }

    /// Applies all transformation rules to the model
    ///
    /// # Returns
    ///
    /// A `TransformationResult` containing information about applied transformations
    pub fn apply(mut self) -> TransformationResult {
        let mut result = TransformationResult {
            transformations_applied: 0,
            successful_transformations: Vec::new(),
            failed_transformations: Vec::new(),
            warnings: Vec::new(),
        };

        let rules = self.rules.clone(); // Clone rules to avoid borrow checker issues
        for rule in &rules {
            match self.apply_rule(rule) {
                Ok(description) => {
                    result.transformations_applied += 1;
                    result.successful_transformations.push(description);
                }
                Err(error) => {
                    result.failed_transformations.push(error);
                }
            }
        }

        result
    }

    /// Applies a single transformation rule
    fn apply_rule(&mut self, rule: &TransformationRule) -> Result<String, String> {
        match rule {
            TransformationRule::RenameProperty { old_name, new_name } => {
                self.apply_rename_property(old_name, new_name)
            }
            TransformationRule::ChangeNamespace {
                old_namespace,
                new_namespace,
            } => self.apply_change_namespace(old_namespace, new_namespace),
            TransformationRule::MakePropertyOptional { property_name } => {
                self.apply_make_property_optional(property_name)
            }
            TransformationRule::MakePropertyRequired { property_name } => {
                self.apply_make_property_required(property_name)
            }
            TransformationRule::UpdatePreferredName { language, new_name } => {
                self.apply_update_preferred_name(language, new_name)
            }
            TransformationRule::UpdateDescription {
                language,
                new_description,
            } => self.apply_update_description(language, new_description),
            TransformationRule::ReplaceUrnPattern {
                pattern,
                replacement,
            } => self.apply_replace_urn_pattern(pattern, replacement),
        }
    }

    /// Applies property rename transformation
    fn apply_rename_property(&mut self, old_name: &str, new_name: &str) -> Result<String, String> {
        // Find property by name
        let property = self
            .aspect
            .properties
            .iter_mut()
            .find(|p| p.name() == old_name)
            .ok_or_else(|| format!("Property '{}' not found", old_name))?;

        // Update URN with new name
        let old_urn = property.urn().to_string();
        let new_urn = if let Some((namespace, _)) = old_urn.rsplit_once('#') {
            format!("{}#{}", namespace, new_name)
        } else {
            return Err(format!("Invalid URN format: {}", old_urn));
        };

        property.metadata.urn = new_urn;

        Ok(format!("Renamed property '{}' to '{}'", old_name, new_name))
    }

    /// Applies namespace change transformation
    fn apply_change_namespace(
        &mut self,
        old_namespace: &str,
        new_namespace: &str,
    ) -> Result<String, String> {
        // Update aspect URN
        let aspect_urn = self.aspect.urn().to_string();
        if let Some(element_name) = aspect_urn
            .strip_prefix(old_namespace)
            .and_then(|s| s.strip_prefix('#'))
        {
            self.aspect.metadata.urn = format!("{}#{}", new_namespace, element_name);
        }

        // Update all property URNs
        let mut count = 0;
        for property in &mut self.aspect.properties {
            let prop_urn = property.urn().to_string();
            if let Some(element_name) = prop_urn
                .strip_prefix(old_namespace)
                .and_then(|s| s.strip_prefix('#'))
            {
                property.metadata.urn = format!("{}#{}", new_namespace, element_name);
                count += 1;
            }
        }

        Ok(format!(
            "Changed namespace from '{}' to '{}' ({} elements updated)",
            old_namespace,
            new_namespace,
            count + 1
        ))
    }

    /// Applies make property optional transformation
    fn apply_make_property_optional(&mut self, property_name: &str) -> Result<String, String> {
        let property = self
            .aspect
            .properties
            .iter_mut()
            .find(|p| p.name() == property_name)
            .ok_or_else(|| format!("Property '{}' not found", property_name))?;

        if property.optional {
            return Err(format!("Property '{}' is already optional", property_name));
        }

        property.optional = true;
        Ok(format!("Made property '{}' optional", property_name))
    }

    /// Applies make property required transformation
    fn apply_make_property_required(&mut self, property_name: &str) -> Result<String, String> {
        let property = self
            .aspect
            .properties
            .iter_mut()
            .find(|p| p.name() == property_name)
            .ok_or_else(|| format!("Property '{}' not found", property_name))?;

        if !property.optional {
            return Err(format!("Property '{}' is already required", property_name));
        }

        property.optional = false;
        Ok(format!("Made property '{}' required", property_name))
    }

    /// Applies preferred name update transformation
    fn apply_update_preferred_name(
        &mut self,
        language: &str,
        new_name: &str,
    ) -> Result<String, String> {
        self.aspect
            .metadata
            .preferred_names
            .insert(language.to_string(), new_name.to_string());
        Ok(format!(
            "Updated preferred name for language '{}' to '{}'",
            language, new_name
        ))
    }

    /// Applies description update transformation
    fn apply_update_description(
        &mut self,
        language: &str,
        new_description: &str,
    ) -> Result<String, String> {
        self.aspect
            .metadata
            .descriptions
            .insert(language.to_string(), new_description.to_string());
        Ok(format!("Updated description for language '{}'", language))
    }

    /// Applies URN pattern replacement transformation
    fn apply_replace_urn_pattern(
        &mut self,
        pattern: &str,
        replacement: &str,
    ) -> Result<String, String> {
        let mut count = 0;

        // Update aspect URN
        let aspect_urn = self.aspect.urn().to_string();
        if aspect_urn.contains(pattern) {
            self.aspect.metadata.urn = aspect_urn.replace(pattern, replacement);
            count += 1;
        }

        // Update property URNs
        for property in &mut self.aspect.properties {
            let prop_urn = property.urn().to_string();
            if prop_urn.contains(pattern) {
                property.metadata.urn = prop_urn.replace(pattern, replacement);
                count += 1;
            }
        }

        if count == 0 {
            return Err(format!("Pattern '{}' not found in any URNs", pattern));
        }

        Ok(format!(
            "Replaced '{}' with '{}' in {} URNs",
            pattern, replacement, count
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rename_property() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        aspect.add_property(Property::new("urn:samm:test:1.0.0#oldName".to_string()));

        let mut transformation = ModelTransformation::new(&mut aspect);
        transformation.rename_property("oldName", "newName");
        let result = transformation.apply();

        assert_eq!(result.transformations_applied, 1);
        assert_eq!(aspect.properties()[0].name(), "newName");
    }

    #[test]
    fn test_change_namespace() {
        let mut aspect = Aspect::new("urn:samm:org.old:1.0.0#TestAspect".to_string());
        aspect.add_property(Property::new("urn:samm:org.old:1.0.0#prop1".to_string()));

        let mut transformation = ModelTransformation::new(&mut aspect);
        transformation.change_namespace("urn:samm:org.old:1.0.0", "urn:samm:org.new:2.0.0");
        let result = transformation.apply();

        assert_eq!(result.transformations_applied, 1);
        assert!(aspect.urn().contains("org.new:2.0.0"));
        assert!(aspect.properties()[0].urn().contains("org.new:2.0.0"));
    }

    #[test]
    fn test_make_property_optional() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        aspect.add_property(Property::new("urn:samm:test:1.0.0#prop1".to_string()));

        let mut transformation = ModelTransformation::new(&mut aspect);
        transformation.make_property_optional("prop1");
        let result = transformation.apply();

        assert_eq!(result.transformations_applied, 1);
        assert!(aspect.properties()[0].optional);
    }

    #[test]
    fn test_make_property_required() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        let mut prop = Property::new("urn:samm:test:1.0.0#prop1".to_string());
        prop.optional = true;
        aspect.add_property(prop);

        let mut transformation = ModelTransformation::new(&mut aspect);
        transformation.make_property_required("prop1");
        let result = transformation.apply();

        assert_eq!(result.transformations_applied, 1);
        assert!(!aspect.properties()[0].optional);
    }

    #[test]
    fn test_make_all_properties_optional() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        aspect.add_property(Property::new("urn:samm:test:1.0.0#prop1".to_string()));
        aspect.add_property(Property::new("urn:samm:test:1.0.0#prop2".to_string()));

        let mut transformation = ModelTransformation::new(&mut aspect);
        transformation.make_all_properties_optional();
        let result = transformation.apply();

        assert_eq!(result.transformations_applied, 2);
        assert!(aspect.properties().iter().all(|p| p.optional));
    }

    #[test]
    fn test_update_preferred_name() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        let mut transformation = ModelTransformation::new(&mut aspect);
        transformation.update_preferred_name("en", "Test Aspect");
        let result = transformation.apply();

        assert_eq!(result.transformations_applied, 1);
        assert_eq!(
            aspect.metadata.preferred_names.get("en"),
            Some(&"Test Aspect".to_string())
        );
    }

    #[test]
    fn test_replace_urn_pattern() {
        let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
        aspect.add_property(Property::new(
            "urn:samm:org.example:1.0.0#prop1".to_string(),
        ));

        let mut transformation = ModelTransformation::new(&mut aspect);
        transformation.replace_urn_pattern("org.example", "com.company");
        let result = transformation.apply();

        assert_eq!(result.transformations_applied, 1);
        assert!(aspect.urn().contains("com.company"));
        assert!(aspect.properties()[0].urn().contains("com.company"));
    }

    #[test]
    fn test_failed_transformation() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        let mut transformation = ModelTransformation::new(&mut aspect);
        transformation.rename_property("nonexistent", "newName");
        let result = transformation.apply();

        assert_eq!(result.transformations_applied, 0);
        assert_eq!(result.failed_transformations.len(), 1);
    }

    #[test]
    fn test_multiple_transformations() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());
        aspect.add_property(Property::new("urn:samm:test:1.0.0#prop1".to_string()));

        let mut transformation = ModelTransformation::new(&mut aspect);
        transformation
            .rename_property("prop1", "newProp")
            .make_property_optional("newProp")
            .update_preferred_name("en", "Test");
        let result = transformation.apply();

        assert_eq!(result.transformations_applied, 3);
        assert_eq!(result.successful_transformations.len(), 3);
    }
}
