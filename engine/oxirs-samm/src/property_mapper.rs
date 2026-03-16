//! # SAMM Property Mapper
//!
//! Provides bidirectional property mapping between SAMM aspect models and target schemas.
//! Supports a rich set of transform rules including renaming, constants, concatenation, and splitting.

use std::collections::HashMap;
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during property mapping.
#[derive(Debug, Clone, PartialEq)]
pub enum MapError {
    /// The requested mapping spec was not found.
    SpecNotFound(String),
    /// A required source property was not present in the input.
    RequiredMissing(String),
    /// A transform rule failed to produce a value.
    TransformFailed(String),
}

impl fmt::Display for MapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MapError::SpecNotFound(name) => write!(f, "Mapping spec not found: {name}"),
            MapError::RequiredMissing(prop) => {
                write!(f, "Required source property missing: {prop}")
            }
            MapError::TransformFailed(msg) => write!(f, "Transform failed: {msg}"),
        }
    }
}

impl std::error::Error for MapError {}

// ─────────────────────────────────────────────────────────────────────────────
// MappingDirection
// ─────────────────────────────────────────────────────────────────────────────

/// The direction in which a property mapping applies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MappingDirection {
    /// Only the forward (source → target) direction.
    Forward,
    /// Only the reverse (target → source) direction.
    Reverse,
    /// Both forward and reverse directions.
    Bidirectional,
}

// ─────────────────────────────────────────────────────────────────────────────
// TransformRule
// ─────────────────────────────────────────────────────────────────────────────

/// Transformation rule applied to the value(s) of a property during mapping.
#[derive(Debug, Clone, PartialEq)]
pub enum TransformRule {
    /// Pass the source value through unchanged.
    Direct,
    /// Emit a renamed copy: the value is unchanged but the key is the target name (handled by the
    /// field `target` in `PropertyMapping`; this variant keeps the rule orthogonal).
    Rename(String),
    /// Always emit this constant string regardless of the source value (or absence thereof).
    Constant(String),
    /// Concatenate the values of multiple named source properties with an optional separator.
    Concatenate(Vec<String>),
    /// Split the source value by `sep` and return the element at `index`.
    Split {
        /// The separator string to split on.
        sep: String,
        /// The zero-based index of the element to return after splitting.
        index: usize,
    },
    /// If the source property is absent, use this default value.
    Default(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// PropertyMapping
// ─────────────────────────────────────────────────────────────────────────────

/// A single property-to-property mapping within a `MappingSpec`.
#[derive(Debug, Clone)]
pub struct PropertyMapping {
    /// Name of the source property.
    pub source: String,
    /// Name of the target property.
    pub target: String,
    /// Direction(s) this mapping applies.
    pub direction: MappingDirection,
    /// How to transform the value.
    pub transform: TransformRule,
    /// If `true` and the source property is absent, `map_forward` returns an error.
    pub required: bool,
}

impl PropertyMapping {
    /// Construct a new `PropertyMapping`.
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        direction: MappingDirection,
        transform: TransformRule,
        required: bool,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            direction,
            transform,
            required,
        }
    }

    /// Convenience: direct mapping, bidirectional, not required.
    pub fn direct(source: impl Into<String>, target: impl Into<String>) -> Self {
        let s = source.into();
        let t = target.into();
        Self::new(
            s,
            t,
            MappingDirection::Bidirectional,
            TransformRule::Direct,
            false,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MappingSpec
// ─────────────────────────────────────────────────────────────────────────────

/// A complete specification describing how one model's properties map to another.
#[derive(Debug, Clone)]
pub struct MappingSpec {
    /// Unique name identifying this spec.
    pub name: String,
    /// Source model identifier (e.g. SAMM aspect URN).
    pub source_model: String,
    /// Target model identifier (e.g. target schema name).
    pub target_model: String,
    /// The individual property mappings.
    pub mappings: Vec<PropertyMapping>,
}

impl MappingSpec {
    /// Create a new `MappingSpec`.
    pub fn new(
        name: impl Into<String>,
        source_model: impl Into<String>,
        target_model: impl Into<String>,
        mappings: Vec<PropertyMapping>,
    ) -> Self {
        Self {
            name: name.into(),
            source_model: source_model.into(),
            target_model: target_model.into(),
            mappings,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MappedValue
// ─────────────────────────────────────────────────────────────────────────────

/// A single resolved mapping result.
#[derive(Debug, Clone, PartialEq)]
pub struct MappedValue {
    /// Original source property name.
    pub source_property: String,
    /// Resulting target property name.
    pub target_property: String,
    /// Transformed value.
    pub value: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// PropertyMapper
// ─────────────────────────────────────────────────────────────────────────────

/// Manages a registry of `MappingSpec`s and applies them to property maps.
#[derive(Debug, Default)]
pub struct PropertyMapper {
    specs: HashMap<String, MappingSpec>,
}

impl PropertyMapper {
    /// Create a new, empty `PropertyMapper`.
    pub fn new() -> Self {
        Self {
            specs: HashMap::new(),
        }
    }

    /// Register a `MappingSpec`. Replaces any existing spec with the same name.
    pub fn add_spec(&mut self, spec: MappingSpec) {
        self.specs.insert(spec.name.clone(), spec);
    }

    /// Return the names of all registered specs.
    pub fn list_specs(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.specs.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// Apply forward (source → target) mappings using the named spec.
    ///
    /// # Errors
    /// - `MapError::SpecNotFound` if the spec does not exist.
    /// - `MapError::RequiredMissing` if a required source property is absent.
    /// - `MapError::TransformFailed` if a transform cannot be applied.
    pub fn map_forward(
        &self,
        spec_name: &str,
        source: &HashMap<String, String>,
    ) -> Result<HashMap<String, String>, MapError> {
        let spec = self
            .specs
            .get(spec_name)
            .ok_or_else(|| MapError::SpecNotFound(spec_name.to_owned()))?;

        let mut result = HashMap::new();

        for mapping in &spec.mappings {
            // Only apply forward or bidirectional mappings.
            if mapping.direction == MappingDirection::Reverse {
                continue;
            }

            match self.apply_transform_forward(mapping, source)? {
                Some(value) => {
                    result.insert(mapping.target.clone(), value);
                }
                None => {
                    // Required check
                    if mapping.required {
                        return Err(MapError::RequiredMissing(mapping.source.clone()));
                    }
                }
            }
        }

        Ok(result)
    }

    /// Apply reverse (target → source) mappings using the named spec.
    ///
    /// Only mappings with direction `Reverse` or `Bidirectional` are applied.
    /// For reverse direction, `target` becomes the lookup key and `source` the output key.
    ///
    /// # Errors
    /// Same variants as `map_forward`.
    pub fn map_reverse(
        &self,
        spec_name: &str,
        target_map: &HashMap<String, String>,
    ) -> Result<HashMap<String, String>, MapError> {
        let spec = self
            .specs
            .get(spec_name)
            .ok_or_else(|| MapError::SpecNotFound(spec_name.to_owned()))?;

        let mut result = HashMap::new();

        for mapping in &spec.mappings {
            if mapping.direction == MappingDirection::Forward {
                continue;
            }

            match self.apply_transform_reverse(mapping, target_map)? {
                Some(value) => {
                    result.insert(mapping.source.clone(), value);
                }
                None => {
                    if mapping.required {
                        return Err(MapError::RequiredMissing(mapping.target.clone()));
                    }
                }
            }
        }

        Ok(result)
    }

    /// Return a list of validation warnings/errors for the named spec.
    ///
    /// Does NOT return a `Result` — warnings are returned as strings so callers
    /// can choose how to handle them.
    pub fn validate_spec(&self, spec_name: &str) -> Vec<String> {
        let mut issues = Vec::new();

        let spec = match self.specs.get(spec_name) {
            Some(s) => s,
            None => {
                issues.push(format!("Spec '{spec_name}' not found"));
                return issues;
            }
        };

        if spec.mappings.is_empty() {
            issues.push(format!("Spec '{spec_name}' has no mappings defined"));
        }

        let mut target_names: HashMap<&str, usize> = HashMap::new();
        for (i, m) in spec.mappings.iter().enumerate() {
            // Duplicate target detection
            if let Some(prev) = target_names.insert(m.target.as_str(), i) {
                issues.push(format!(
                    "Mapping {i}: target property '{}' is also used by mapping {prev}",
                    m.target
                ));
            }

            // Source must not be empty.
            if m.source.is_empty() {
                issues.push(format!("Mapping {i}: source property name is empty"));
            }

            // Target must not be empty.
            if m.target.is_empty() {
                issues.push(format!("Mapping {i}: target property name is empty"));
            }

            // Validate transform-specific invariants.
            match &m.transform {
                TransformRule::Concatenate(parts) if parts.is_empty() => {
                    issues.push(format!(
                        "Mapping {i}: Concatenate transform has no source parts"
                    ));
                }
                TransformRule::Split { sep, .. } if sep.is_empty() => {
                    issues.push(format!(
                        "Mapping {i}: Split transform has an empty separator"
                    ));
                }
                _ => {}
            }
        }

        issues
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Apply the transform rule in the forward direction.
    fn apply_transform_forward(
        &self,
        mapping: &PropertyMapping,
        source: &HashMap<String, String>,
    ) -> Result<Option<String>, MapError> {
        match &mapping.transform {
            TransformRule::Direct => Ok(source.get(&mapping.source).cloned()),

            TransformRule::Rename(_new_name) => {
                // `Rename` carries the new *key* as context; the value still comes from `source`.
                // The actual re-keying is performed by inserting under `mapping.target`.
                Ok(source.get(&mapping.source).cloned())
            }

            TransformRule::Constant(c) => Ok(Some(c.clone())),

            TransformRule::Concatenate(parts) => {
                // Collect values; if any part is missing we produce an empty string for it.
                let joined: String = parts
                    .iter()
                    .map(|p| source.get(p).map(|s| s.as_str()).unwrap_or(""))
                    .collect::<Vec<_>>()
                    .join("");
                Ok(Some(joined))
            }

            TransformRule::Split { sep, index } => {
                let raw = match source.get(&mapping.source) {
                    Some(v) => v,
                    None => return Ok(None),
                };
                let parts: Vec<&str> = raw.split(sep.as_str()).collect();
                match parts.get(*index) {
                    Some(part) => Ok(Some(part.to_string())),
                    None => Err(MapError::TransformFailed(format!(
                        "Split index {index} out of range for value '{}' (split by '{sep}')",
                        raw
                    ))),
                }
            }

            TransformRule::Default(default_val) => {
                let value = source
                    .get(&mapping.source)
                    .cloned()
                    .unwrap_or_else(|| default_val.clone());
                Ok(Some(value))
            }
        }
    }

    /// Apply the transform rule in the reverse direction.
    ///
    /// In reverse, the roles of `source` and `target` in the mapping struct are swapped:
    /// we look up `mapping.target` in `target_map` and write to `mapping.source`.
    fn apply_transform_reverse(
        &self,
        mapping: &PropertyMapping,
        target_map: &HashMap<String, String>,
    ) -> Result<Option<String>, MapError> {
        match &mapping.transform {
            TransformRule::Direct => Ok(target_map.get(&mapping.target).cloned()),

            TransformRule::Rename(_) => Ok(target_map.get(&mapping.target).cloned()),

            // Constants have no reverse: there is no source value to reconstruct.
            TransformRule::Constant(_) => Ok(None),

            // Concatenate / Split are not invertible; we do a best-effort pass-through.
            TransformRule::Concatenate(_) => Ok(target_map.get(&mapping.target).cloned()),

            TransformRule::Split { .. } => Ok(target_map.get(&mapping.target).cloned()),

            TransformRule::Default(default_val) => {
                let value = target_map
                    .get(&mapping.target)
                    .cloned()
                    .unwrap_or_else(|| default_val.clone());
                Ok(Some(value))
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_source(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn make_spec(mappings: Vec<PropertyMapping>) -> MappingSpec {
        MappingSpec::new("test_spec", "aspect:Source", "schema:Target", mappings)
    }

    // ── Basic forward mapping ────────────────────────────────────────────────

    #[test]
    fn test_forward_direct() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::direct(
            "name",
            "displayName",
        )]));
        let src = make_source(&[("name", "Alice")]);
        let result = mapper
            .map_forward("test_spec", &src)
            .expect("should succeed");
        assert_eq!(result.get("displayName").map(|s| s.as_str()), Some("Alice"));
    }

    #[test]
    fn test_forward_missing_optional() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "optional",
            "opt",
            MappingDirection::Forward,
            TransformRule::Direct,
            false,
        )]));
        let src = make_source(&[]);
        let result = mapper
            .map_forward("test_spec", &src)
            .expect("should succeed");
        assert!(!result.contains_key("opt"));
    }

    #[test]
    fn test_forward_required_missing_returns_error() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "id",
            "identifier",
            MappingDirection::Forward,
            TransformRule::Direct,
            true,
        )]));
        let src = make_source(&[]);
        let err = mapper.map_forward("test_spec", &src).unwrap_err();
        assert!(matches!(err, MapError::RequiredMissing(_)));
    }

    #[test]
    fn test_spec_not_found() {
        let mapper = PropertyMapper::new();
        let src = make_source(&[]);
        let err = mapper.map_forward("nonexistent", &src).unwrap_err();
        assert!(matches!(err, MapError::SpecNotFound(_)));
    }

    // ── Transform: Rename ────────────────────────────────────────────────────

    #[test]
    fn test_transform_rename_forward() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "legacyField",
            "newField",
            MappingDirection::Forward,
            TransformRule::Rename("newField".into()),
            false,
        )]));
        let src = make_source(&[("legacyField", "42")]);
        let result = mapper
            .map_forward("test_spec", &src)
            .expect("should succeed");
        assert_eq!(result.get("newField").map(|s| s.as_str()), Some("42"));
        assert!(!result.contains_key("legacyField"));
    }

    #[test]
    fn test_transform_rename_missing_source() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "missing",
            "newField",
            MappingDirection::Forward,
            TransformRule::Rename("newField".into()),
            false,
        )]));
        let src = make_source(&[]);
        let result = mapper
            .map_forward("test_spec", &src)
            .expect("should succeed");
        assert!(result.is_empty());
    }

    // ── Transform: Constant ──────────────────────────────────────────────────

    #[test]
    fn test_transform_constant_always_emitted() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "ignored",
            "version",
            MappingDirection::Forward,
            TransformRule::Constant("2.3.0".into()),
            false,
        )]));
        let src = make_source(&[]);
        let result = mapper
            .map_forward("test_spec", &src)
            .expect("should succeed");
        assert_eq!(result.get("version").map(|s| s.as_str()), Some("2.3.0"));
    }

    #[test]
    fn test_transform_constant_overrides_source() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "kind",
            "type",
            MappingDirection::Forward,
            TransformRule::Constant("fixed".into()),
            false,
        )]));
        let src = make_source(&[("kind", "variable")]);
        let result = mapper
            .map_forward("test_spec", &src)
            .expect("should succeed");
        assert_eq!(result.get("type").map(|s| s.as_str()), Some("fixed"));
    }

    // ── Transform: Concatenate ───────────────────────────────────────────────

    #[test]
    fn test_transform_concatenate_all_present() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "first",
            "fullName",
            MappingDirection::Forward,
            TransformRule::Concatenate(vec!["first".into(), "last".into()]),
            false,
        )]));
        let src = make_source(&[("first", "John"), ("last", "Doe")]);
        let result = mapper
            .map_forward("test_spec", &src)
            .expect("should succeed");
        assert_eq!(result.get("fullName").map(|s| s.as_str()), Some("JohnDoe"));
    }

    #[test]
    fn test_transform_concatenate_missing_part_uses_empty() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "a",
            "combined",
            MappingDirection::Forward,
            TransformRule::Concatenate(vec!["a".into(), "b".into()]),
            false,
        )]));
        let src = make_source(&[("a", "hello")]);
        let result = mapper
            .map_forward("test_spec", &src)
            .expect("should succeed");
        assert_eq!(result.get("combined").map(|s| s.as_str()), Some("hello"));
    }

    #[test]
    fn test_transform_concatenate_three_parts() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "_",
            "id",
            MappingDirection::Forward,
            TransformRule::Concatenate(vec!["ns".into(), "colon".into(), "local".into()]),
            false,
        )]));
        let src = make_source(&[("ns", "ex"), ("colon", ":"), ("local", "thing")]);
        let result = mapper
            .map_forward("test_spec", &src)
            .expect("should succeed");
        assert_eq!(result.get("id").map(|s| s.as_str()), Some("ex:thing"));
    }

    // ── Transform: Split ─────────────────────────────────────────────────────

    #[test]
    fn test_transform_split_index_zero() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "path",
            "head",
            MappingDirection::Forward,
            TransformRule::Split {
                sep: "/".into(),
                index: 0,
            },
            false,
        )]));
        let src = make_source(&[("path", "a/b/c")]);
        let result = mapper
            .map_forward("test_spec", &src)
            .expect("should succeed");
        assert_eq!(result.get("head").map(|s| s.as_str()), Some("a"));
    }

    #[test]
    fn test_transform_split_last_index() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "csv",
            "last_col",
            MappingDirection::Forward,
            TransformRule::Split {
                sep: ",".into(),
                index: 2,
            },
            false,
        )]));
        let src = make_source(&[("csv", "x,y,z")]);
        let result = mapper
            .map_forward("test_spec", &src)
            .expect("should succeed");
        assert_eq!(result.get("last_col").map(|s| s.as_str()), Some("z"));
    }

    #[test]
    fn test_transform_split_out_of_range_error() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "v",
            "part",
            MappingDirection::Forward,
            TransformRule::Split {
                sep: ".".into(),
                index: 99,
            },
            false,
        )]));
        let src = make_source(&[("v", "a.b")]);
        let err = mapper.map_forward("test_spec", &src).unwrap_err();
        assert!(matches!(err, MapError::TransformFailed(_)));
    }

    #[test]
    fn test_transform_split_source_missing() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "missing",
            "out",
            MappingDirection::Forward,
            TransformRule::Split {
                sep: ".".into(),
                index: 0,
            },
            false,
        )]));
        let src = make_source(&[]);
        let result = mapper
            .map_forward("test_spec", &src)
            .expect("should succeed");
        assert!(result.is_empty());
    }

    // ── Transform: Default ───────────────────────────────────────────────────

    #[test]
    fn test_transform_default_when_absent() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "color",
            "colour",
            MappingDirection::Forward,
            TransformRule::Default("red".into()),
            false,
        )]));
        let src = make_source(&[]);
        let result = mapper
            .map_forward("test_spec", &src)
            .expect("should succeed");
        assert_eq!(result.get("colour").map(|s| s.as_str()), Some("red"));
    }

    #[test]
    fn test_transform_default_when_present_uses_source() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "color",
            "colour",
            MappingDirection::Forward,
            TransformRule::Default("red".into()),
            false,
        )]));
        let src = make_source(&[("color", "blue")]);
        let result = mapper
            .map_forward("test_spec", &src)
            .expect("should succeed");
        assert_eq!(result.get("colour").map(|s| s.as_str()), Some("blue"));
    }

    // ── Reverse mapping ──────────────────────────────────────────────────────

    #[test]
    fn test_reverse_only_applied_for_reverse_direction() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![
            PropertyMapping::new(
                "source_prop",
                "target_prop",
                MappingDirection::Forward,
                TransformRule::Direct,
                false,
            ),
            PropertyMapping::new(
                "src2",
                "tgt2",
                MappingDirection::Reverse,
                TransformRule::Direct,
                false,
            ),
        ]));
        let tgt = make_source(&[("target_prop", "v1"), ("tgt2", "v2")]);
        let result = mapper
            .map_reverse("test_spec", &tgt)
            .expect("should succeed");
        // Forward mapping is excluded from reverse
        assert!(!result.contains_key("source_prop"));
        // Reverse mapping applied
        assert_eq!(result.get("src2").map(|s| s.as_str()), Some("v2"));
    }

    #[test]
    fn test_reverse_bidirectional() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "foo",
            "bar",
            MappingDirection::Bidirectional,
            TransformRule::Direct,
            false,
        )]));
        let tgt = make_source(&[("bar", "hello")]);
        let result = mapper
            .map_reverse("test_spec", &tgt)
            .expect("should succeed");
        assert_eq!(result.get("foo").map(|s| s.as_str()), Some("hello"));
    }

    #[test]
    fn test_reverse_constant_not_propagated() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "src",
            "const_field",
            MappingDirection::Bidirectional,
            TransformRule::Constant("fixed".into()),
            false,
        )]));
        let tgt = make_source(&[("const_field", "fixed")]);
        let result = mapper
            .map_reverse("test_spec", &tgt)
            .expect("should succeed");
        // Constants are not invertible
        assert!(!result.contains_key("src"));
    }

    #[test]
    fn test_reverse_default_transform() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "src",
            "tgt",
            MappingDirection::Bidirectional,
            TransformRule::Default("fallback".into()),
            false,
        )]));
        let tgt = make_source(&[]);
        let result = mapper
            .map_reverse("test_spec", &tgt)
            .expect("should succeed");
        assert_eq!(result.get("src").map(|s| s.as_str()), Some("fallback"));
    }

    // ── Bidirectional round-trip ──────────────────────────────────────────────

    #[test]
    fn test_bidirectional_round_trip() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "age",
            "years",
            MappingDirection::Bidirectional,
            TransformRule::Direct,
            false,
        )]));
        let src = make_source(&[("age", "30")]);
        let fwd = mapper
            .map_forward("test_spec", &src)
            .expect("should succeed");
        let rev = mapper
            .map_reverse("test_spec", &fwd)
            .expect("should succeed");
        assert_eq!(rev.get("age"), src.get("age"));
    }

    // ── Multiple mappings ─────────────────────────────────────────────────────

    #[test]
    fn test_multiple_mappings_forward() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![
            PropertyMapping::direct("a", "x"),
            PropertyMapping::direct("b", "y"),
            PropertyMapping::direct("c", "z"),
        ]));
        let src = make_source(&[("a", "1"), ("b", "2"), ("c", "3")]);
        let result = mapper
            .map_forward("test_spec", &src)
            .expect("should succeed");
        assert_eq!(result.len(), 3);
        assert_eq!(result.get("x").map(|s| s.as_str()), Some("1"));
        assert_eq!(result.get("y").map(|s| s.as_str()), Some("2"));
        assert_eq!(result.get("z").map(|s| s.as_str()), Some("3"));
    }

    // ── list_specs ───────────────────────────────────────────────────────────

    #[test]
    fn test_list_specs_empty() {
        let mapper = PropertyMapper::new();
        assert!(mapper.list_specs().is_empty());
    }

    #[test]
    fn test_list_specs_sorted() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(MappingSpec::new("zz", "a", "b", vec![]));
        mapper.add_spec(MappingSpec::new("aa", "a", "b", vec![]));
        mapper.add_spec(MappingSpec::new("mm", "a", "b", vec![]));
        let names = mapper.list_specs();
        assert_eq!(names, vec!["aa", "mm", "zz"]);
    }

    #[test]
    fn test_add_spec_replaces_existing() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(MappingSpec::new("s", "old_src", "old_tgt", vec![]));
        mapper.add_spec(MappingSpec::new("s", "new_src", "new_tgt", vec![]));
        assert_eq!(mapper.list_specs().len(), 1);
        assert_eq!(mapper.specs["s"].source_model, "new_src");
    }

    // ── validate_spec ────────────────────────────────────────────────────────

    #[test]
    fn test_validate_spec_not_found() {
        let mapper = PropertyMapper::new();
        let issues = mapper.validate_spec("ghost");
        assert!(!issues.is_empty());
        assert!(issues[0].contains("not found"));
    }

    #[test]
    fn test_validate_spec_empty_mappings_warning() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(MappingSpec::new("empty", "s", "t", vec![]));
        let issues = mapper.validate_spec("empty");
        assert!(issues.iter().any(|i| i.contains("no mappings")));
    }

    #[test]
    fn test_validate_spec_empty_source_name() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "",
            "tgt",
            MappingDirection::Forward,
            TransformRule::Direct,
            false,
        )]));
        let issues = mapper.validate_spec("test_spec");
        assert!(issues
            .iter()
            .any(|i| i.contains("source property name is empty")));
    }

    #[test]
    fn test_validate_spec_empty_target_name() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "src",
            "",
            MappingDirection::Forward,
            TransformRule::Direct,
            false,
        )]));
        let issues = mapper.validate_spec("test_spec");
        assert!(issues
            .iter()
            .any(|i| i.contains("target property name is empty")));
    }

    #[test]
    fn test_validate_spec_duplicate_target() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![
            PropertyMapping::direct("a", "out"),
            PropertyMapping::direct("b", "out"),
        ]));
        let issues = mapper.validate_spec("test_spec");
        assert!(issues.iter().any(|i| i.contains("out")));
    }

    #[test]
    fn test_validate_spec_concatenate_empty_parts() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "x",
            "y",
            MappingDirection::Forward,
            TransformRule::Concatenate(vec![]),
            false,
        )]));
        let issues = mapper.validate_spec("test_spec");
        assert!(issues.iter().any(|i| i.contains("no source parts")));
    }

    #[test]
    fn test_validate_spec_split_empty_separator() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "x",
            "y",
            MappingDirection::Forward,
            TransformRule::Split {
                sep: "".into(),
                index: 0,
            },
            false,
        )]));
        let issues = mapper.validate_spec("test_spec");
        assert!(issues.iter().any(|i| i.contains("empty separator")));
    }

    #[test]
    fn test_validate_spec_clean_returns_empty() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::direct("a", "b")]));
        let issues = mapper.validate_spec("test_spec");
        assert!(issues.is_empty(), "Expected no issues, got: {issues:?}");
    }

    // ── Edge cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_forward_empty_source_map() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::direct("x", "y")]));
        let result = mapper
            .map_forward("test_spec", &HashMap::new())
            .expect("should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_forward_empty_mappings() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(MappingSpec::new("empty", "s", "t", vec![]));
        let src = make_source(&[("a", "1")]);
        let result = mapper.map_forward("empty", &src).expect("should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_reverse_spec_not_found() {
        let mapper = PropertyMapper::new();
        let err = mapper.map_reverse("ghost", &HashMap::new()).unwrap_err();
        assert!(matches!(err, MapError::SpecNotFound(_)));
    }

    #[test]
    fn test_map_error_display_spec_not_found() {
        let e = MapError::SpecNotFound("x".into());
        assert!(e.to_string().contains("x"));
    }

    #[test]
    fn test_map_error_display_required_missing() {
        let e = MapError::RequiredMissing("prop".into());
        assert!(e.to_string().contains("prop"));
    }

    #[test]
    fn test_map_error_display_transform_failed() {
        let e = MapError::TransformFailed("reason".into());
        assert!(e.to_string().contains("reason"));
    }

    #[test]
    fn test_concatenate_single_part() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "sole",
            "out",
            MappingDirection::Forward,
            TransformRule::Concatenate(vec!["sole".into()]),
            false,
        )]));
        let src = make_source(&[("sole", "hello")]);
        let result = mapper
            .map_forward("test_spec", &src)
            .expect("should succeed");
        assert_eq!(result.get("out").map(|s| s.as_str()), Some("hello"));
    }

    #[test]
    fn test_split_by_space() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "words",
            "second",
            MappingDirection::Forward,
            TransformRule::Split {
                sep: " ".into(),
                index: 1,
            },
            false,
        )]));
        let src = make_source(&[("words", "hello world")]);
        let result = mapper
            .map_forward("test_spec", &src)
            .expect("should succeed");
        assert_eq!(result.get("second").map(|s| s.as_str()), Some("world"));
    }

    #[test]
    fn test_mapped_value_fields() {
        let mv = MappedValue {
            source_property: "src".into(),
            target_property: "tgt".into(),
            value: "val".into(),
        };
        assert_eq!(mv.source_property, "src");
        assert_eq!(mv.target_property, "tgt");
        assert_eq!(mv.value, "val");
    }

    #[test]
    fn test_property_mapping_direct_helper() {
        let pm = PropertyMapping::direct("a", "b");
        assert_eq!(pm.source, "a");
        assert_eq!(pm.target, "b");
        assert_eq!(pm.direction, MappingDirection::Bidirectional);
        assert_eq!(pm.transform, TransformRule::Direct);
        assert!(!pm.required);
    }

    #[test]
    fn test_mapping_spec_fields() {
        let spec = MappingSpec::new("s", "src_model", "tgt_model", vec![]);
        assert_eq!(spec.name, "s");
        assert_eq!(spec.source_model, "src_model");
        assert_eq!(spec.target_model, "tgt_model");
        assert!(spec.mappings.is_empty());
    }

    #[test]
    fn test_forward_skips_reverse_only_mappings() {
        let mut mapper = PropertyMapper::new();
        mapper.add_spec(make_spec(vec![PropertyMapping::new(
            "a",
            "b",
            MappingDirection::Reverse,
            TransformRule::Direct,
            false,
        )]));
        let src = make_source(&[("a", "1")]);
        let result = mapper
            .map_forward("test_spec", &src)
            .expect("should succeed");
        assert!(result.is_empty());
    }
}
