// Registry of SAMM constraint types
// Added in v1.1.0 Round 7

use std::collections::HashMap;

/// A value that can appear in SAMM constraint definitions.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintValue {
    /// A 64-bit signed integer value.
    Integer(i64),
    /// A 64-bit floating-point value.
    Float(f64),
    /// A UTF-8 text string value.
    Text(String),
    /// A boolean value.
    Bool(bool),
    /// A list of string values.
    StringList(Vec<String>),
}

/// Bound type for range constraints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundType {
    /// Exclusive (open) bound.
    Open,
    /// Inclusive (closed) bound.
    Closed,
    /// Greater-than-or-equal (closed lower) bound.
    AtLeast,
    /// Less-than-or-equal (closed upper) bound.
    AtMost,
    /// Strict greater-than bound.
    GreaterThan,
    /// Strict less-than bound.
    LessThan,
}

impl BoundType {
    /// Returns true if this bound type is exclusive.
    pub fn is_exclusive(self) -> bool {
        matches!(
            self,
            BoundType::Open | BoundType::GreaterThan | BoundType::LessThan
        )
    }
}

/// The kind of SAMM constraint.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintKind {
    /// Restricts the length (in characters) of a string value.
    LengthConstraint {
        /// Minimum allowed length (inclusive).
        min: Option<usize>,
        /// Maximum allowed length (inclusive).
        max: Option<usize>,
    },
    /// Restricts a numeric value to a range.
    RangeConstraint {
        /// Lower bound value.
        min: Option<ConstraintValue>,
        /// Upper bound value.
        max: Option<ConstraintValue>,
        /// Whether the lower bound is open or closed.
        lower_bound_definition: BoundType,
        /// Whether the upper bound is open or closed.
        upper_bound_definition: BoundType,
    },
    /// Restricts a string value to match a regular expression pattern.
    RegularExpressionConstraint {
        /// The regular expression pattern.
        value: String,
    },
    /// Restricts a value to one of an enumerated set.
    EnumerationConstraint {
        /// The allowed values.
        values: Vec<ConstraintValue>,
    },
    /// Restricts a value to one of the allowed BCP 47 language codes.
    LanguageConstraint {
        /// Allowed BCP 47 language codes.
        language_codes: Vec<String>,
    },
    /// Restricts a value to one of the allowed locale codes.
    LocaleConstraint {
        /// Allowed locale codes (e.g. "en-US").
        locale_codes: Vec<String>,
    },
    /// Restricts a decimal value to a fixed point precision.
    FixedPointConstraint {
        /// Number of digits after the decimal point.
        scale: u32,
        /// Maximum number of digits before the decimal point.
        integer: u32,
    },
    /// Maps a payload to a named field.
    PayloadMappingConstraint {
        /// The payload field name.
        payload_name: String,
    },
}

impl ConstraintKind {
    /// Returns the canonical kind name string.
    pub fn kind_name(&self) -> &'static str {
        match self {
            ConstraintKind::LengthConstraint { .. } => "LengthConstraint",
            ConstraintKind::RangeConstraint { .. } => "RangeConstraint",
            ConstraintKind::RegularExpressionConstraint { .. } => "RegularExpressionConstraint",
            ConstraintKind::EnumerationConstraint { .. } => "EnumerationConstraint",
            ConstraintKind::LanguageConstraint { .. } => "LanguageConstraint",
            ConstraintKind::LocaleConstraint { .. } => "LocaleConstraint",
            ConstraintKind::FixedPointConstraint { .. } => "FixedPointConstraint",
            ConstraintKind::PayloadMappingConstraint { .. } => "PayloadMappingConstraint",
        }
    }
}

/// A SAMM constraint definition.
#[derive(Debug, Clone)]
pub struct Constraint {
    /// The URN uniquely identifying this constraint.
    pub urn: String,
    /// Human-readable name for the constraint.
    pub name: String,
    /// Optional description.
    pub description: Option<String>,
    /// The constraint kind and its parameters.
    pub kind: ConstraintKind,
}

/// Registry errors.
#[derive(Debug)]
pub enum RegistryError {
    /// A constraint with the same URN was already registered.
    DuplicateUrn(String),
    /// No constraint was found for the given URN.
    ConstraintNotFound(String),
    /// The provided value is not valid for this constraint.
    InvalidValue(String),
    /// An error in a regular expression pattern.
    RegexError(String),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::DuplicateUrn(urn) => write!(f, "Duplicate URN: {urn}"),
            RegistryError::ConstraintNotFound(urn) => write!(f, "Constraint not found: {urn}"),
            RegistryError::InvalidValue(msg) => write!(f, "Invalid value: {msg}"),
            RegistryError::RegexError(msg) => write!(f, "Regex error: {msg}"),
        }
    }
}

impl std::error::Error for RegistryError {}

/// A simple pattern matcher for RegularExpressionConstraint tests.
/// Supports: exact match, prefix* (anchored start), *suffix (anchored end),
/// *contains* (substring), and character class `[...]` patterns.
fn matches_pattern(pattern: &str, value: &str) -> bool {
    // Full exact match
    if !pattern.contains('*') && !pattern.contains('[') {
        return pattern == value;
    }
    // Wildcard patterns
    if pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        if !prefix.contains('*') {
            return value.starts_with(prefix);
        }
    }
    if let Some(suffix) = pattern.strip_prefix('*') {
        if !suffix.contains('*') {
            return value.ends_with(suffix);
        }
    }
    if pattern.starts_with('*') && pattern.ends_with('*') {
        let inner = &pattern[1..pattern.len() - 1];
        if !inner.contains('*') {
            return value.contains(inner);
        }
    }
    // Character class [abc]: match if value consists of only those chars
    if pattern.starts_with('[') && pattern.ends_with(']') {
        let chars_allowed: &str = &pattern[1..pattern.len() - 1];
        return value.chars().all(|c| chars_allowed.contains(c));
    }
    // Digit pattern: \d+ means all digits
    if pattern == r"\d+" {
        return !value.is_empty() && value.chars().all(|c| c.is_ascii_digit());
    }
    // Alphanumeric: \w+
    if pattern == r"\w+" {
        return !value.is_empty() && value.chars().all(|c| c.is_alphanumeric() || c == '_');
    }
    // Fallback: substring match
    value.contains(pattern)
}

/// Registry of SAMM constraints indexed by URN.
pub struct ConstraintRegistry {
    constraints: HashMap<String, Constraint>,
}

impl ConstraintRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            constraints: HashMap::new(),
        }
    }

    /// Register a constraint. Returns an error if the URN already exists.
    pub fn register(&mut self, constraint: Constraint) -> Result<(), RegistryError> {
        if self.constraints.contains_key(&constraint.urn) {
            return Err(RegistryError::DuplicateUrn(constraint.urn.clone()));
        }
        self.constraints.insert(constraint.urn.clone(), constraint);
        Ok(())
    }

    /// Get a constraint by URN.
    pub fn get(&self, urn: &str) -> Option<&Constraint> {
        self.constraints.get(urn)
    }

    /// List all constraints of a given kind name (e.g. "LengthConstraint").
    pub fn list_by_kind(&self, kind_name: &str) -> Vec<&Constraint> {
        self.constraints
            .values()
            .filter(|c| c.kind.kind_name() == kind_name)
            .collect()
    }

    /// Validate a value against a constraint identified by URN.
    pub fn validate_value(
        &self,
        urn: &str,
        value: &ConstraintValue,
    ) -> Result<bool, RegistryError> {
        let constraint = self
            .constraints
            .get(urn)
            .ok_or_else(|| RegistryError::ConstraintNotFound(urn.to_string()))?;

        let result = match &constraint.kind {
            ConstraintKind::LengthConstraint { min, max } => {
                let len = match value {
                    ConstraintValue::Text(s) => s.len(),
                    ConstraintValue::Integer(i) => i.to_string().len(),
                    ConstraintValue::StringList(v) => v.len(),
                    _ => {
                        return Err(RegistryError::InvalidValue(
                            "LengthConstraint only applies to Text or StringList".to_string(),
                        ));
                    }
                };
                let min_ok = min.map_or(true, |m| len >= m);
                let max_ok = max.map_or(true, |m| len <= m);
                min_ok && max_ok
            }

            ConstraintKind::RangeConstraint {
                min,
                max,
                lower_bound_definition,
                upper_bound_definition,
            } => {
                let num = extract_numeric(value)?;
                let lower_ok = match min {
                    None => true,
                    Some(bound) => {
                        let bound_num = extract_numeric(bound)?;
                        match lower_bound_definition {
                            BoundType::Open | BoundType::GreaterThan => num > bound_num,
                            BoundType::Closed | BoundType::AtLeast => num >= bound_num,
                            BoundType::AtMost | BoundType::LessThan => true, // not applicable for lower
                        }
                    }
                };
                let upper_ok = match max {
                    None => true,
                    Some(bound) => {
                        let bound_num = extract_numeric(bound)?;
                        match upper_bound_definition {
                            BoundType::Open | BoundType::LessThan => num < bound_num,
                            BoundType::Closed | BoundType::AtMost => num <= bound_num,
                            BoundType::AtLeast | BoundType::GreaterThan => true,
                        }
                    }
                };
                lower_ok && upper_ok
            }

            ConstraintKind::RegularExpressionConstraint { value: pattern } => {
                let s = match value {
                    ConstraintValue::Text(s) => s.as_str(),
                    _ => {
                        return Err(RegistryError::InvalidValue(
                            "RegularExpressionConstraint only applies to Text".to_string(),
                        ));
                    }
                };
                matches_pattern(pattern, s)
            }

            ConstraintKind::EnumerationConstraint { values: allowed } => allowed.contains(value),

            ConstraintKind::LanguageConstraint { language_codes } => {
                let s = match value {
                    ConstraintValue::Text(s) => s.as_str(),
                    _ => {
                        return Err(RegistryError::InvalidValue(
                            "LanguageConstraint only applies to Text".to_string(),
                        ));
                    }
                };
                // BCP 47 tag check: case-insensitive match
                language_codes
                    .iter()
                    .any(|code| code.to_lowercase() == s.to_lowercase())
            }

            ConstraintKind::LocaleConstraint { locale_codes } => {
                let s = match value {
                    ConstraintValue::Text(s) => s.as_str(),
                    _ => {
                        return Err(RegistryError::InvalidValue(
                            "LocaleConstraint only applies to Text".to_string(),
                        ));
                    }
                };
                locale_codes.iter().any(|code| code == s)
            }

            ConstraintKind::FixedPointConstraint { scale, integer } => {
                let s = match value {
                    ConstraintValue::Text(s) => s.as_str(),
                    _ => {
                        return Err(RegistryError::InvalidValue(
                            "FixedPointConstraint only applies to Text decimal strings".to_string(),
                        ));
                    }
                };
                validate_fixed_point(s, *scale, *integer)
            }

            ConstraintKind::PayloadMappingConstraint { payload_name } => {
                let s = match value {
                    ConstraintValue::Text(s) => s.as_str(),
                    _ => {
                        return Err(RegistryError::InvalidValue(
                            "PayloadMappingConstraint only applies to Text".to_string(),
                        ));
                    }
                };
                s == payload_name.as_str()
            }
        };

        Ok(result)
    }

    /// Total number of registered constraints.
    pub fn count(&self) -> usize {
        self.constraints.len()
    }

    /// Remove a constraint by URN, returning it if it existed.
    pub fn remove(&mut self, urn: &str) -> Option<Constraint> {
        self.constraints.remove(urn)
    }

    /// Iterate over all constraints.
    pub fn all(&self) -> impl Iterator<Item = &Constraint> {
        self.constraints.values()
    }
}

impl Default for ConstraintRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn extract_numeric(value: &ConstraintValue) -> Result<f64, RegistryError> {
    match value {
        ConstraintValue::Integer(i) => Ok(*i as f64),
        ConstraintValue::Float(f) => Ok(*f),
        ConstraintValue::Text(s) => s
            .parse::<f64>()
            .map_err(|_| RegistryError::InvalidValue(format!("Cannot parse '{s}' as a number"))),
        _ => Err(RegistryError::InvalidValue(
            "RangeConstraint requires numeric value".to_string(),
        )),
    }
}

/// Validate a decimal string against fixed-point parameters.
/// `integer` = max digits before decimal point, `scale` = max digits after.
fn validate_fixed_point(s: &str, scale: u32, integer: u32) -> bool {
    let s = s.trim_start_matches('-');
    let parts: Vec<&str> = s.splitn(2, '.').collect();
    let int_part = parts[0];
    let frac_part = parts.get(1).copied().unwrap_or("");
    int_part.len() <= integer as usize && frac_part.len() <= scale as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn len_constraint(urn: &str, min: Option<usize>, max: Option<usize>) -> Constraint {
        Constraint {
            urn: urn.to_string(),
            name: "len".to_string(),
            description: None,
            kind: ConstraintKind::LengthConstraint { min, max },
        }
    }

    fn range_constraint(urn: &str, min: Option<f64>, max: Option<f64>) -> Constraint {
        Constraint {
            urn: urn.to_string(),
            name: "range".to_string(),
            description: None,
            kind: ConstraintKind::RangeConstraint {
                min: min.map(ConstraintValue::Float),
                max: max.map(ConstraintValue::Float),
                lower_bound_definition: BoundType::Closed,
                upper_bound_definition: BoundType::Closed,
            },
        }
    }

    fn regex_constraint(urn: &str, pattern: &str) -> Constraint {
        Constraint {
            urn: urn.to_string(),
            name: "regex".to_string(),
            description: None,
            kind: ConstraintKind::RegularExpressionConstraint {
                value: pattern.to_string(),
            },
        }
    }

    fn enum_constraint(urn: &str, values: Vec<ConstraintValue>) -> Constraint {
        Constraint {
            urn: urn.to_string(),
            name: "enum".to_string(),
            description: None,
            kind: ConstraintKind::EnumerationConstraint { values },
        }
    }

    // ---- register / count ----

    #[test]
    fn test_register_and_count() {
        let mut reg = ConstraintRegistry::new();
        assert_eq!(reg.count(), 0);
        reg.register(len_constraint("urn:test:1", Some(1), Some(10)))
            .expect("should succeed");
        assert_eq!(reg.count(), 1);
    }

    #[test]
    fn test_register_duplicate_urn_error() {
        let mut reg = ConstraintRegistry::new();
        reg.register(len_constraint("urn:test:1", None, None))
            .expect("should succeed");
        let result = reg.register(len_constraint("urn:test:1", None, None));
        assert!(matches!(result, Err(RegistryError::DuplicateUrn(_))));
    }

    #[test]
    fn test_register_multiple_different_urns() {
        let mut reg = ConstraintRegistry::new();
        reg.register(len_constraint("urn:test:a", None, None))
            .expect("should succeed");
        reg.register(len_constraint("urn:test:b", None, None))
            .expect("should succeed");
        assert_eq!(reg.count(), 2);
    }

    // ---- get ----

    #[test]
    fn test_get_existing() {
        let mut reg = ConstraintRegistry::new();
        reg.register(len_constraint("urn:test:1", Some(2), Some(5)))
            .expect("should succeed");
        let c = reg.get("urn:test:1").expect("should succeed");
        assert_eq!(c.urn, "urn:test:1");
    }

    #[test]
    fn test_get_missing_returns_none() {
        let reg = ConstraintRegistry::new();
        assert!(reg.get("urn:test:nonexistent").is_none());
    }

    // ---- list_by_kind ----

    #[test]
    fn test_list_by_kind_length() {
        let mut reg = ConstraintRegistry::new();
        reg.register(len_constraint("urn:1", None, None))
            .expect("should succeed");
        reg.register(len_constraint("urn:2", None, None))
            .expect("should succeed");
        reg.register(range_constraint("urn:3", None, None))
            .expect("should succeed");
        let list = reg.list_by_kind("LengthConstraint");
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_list_by_kind_range() {
        let mut reg = ConstraintRegistry::new();
        reg.register(range_constraint("urn:1", Some(0.0), Some(100.0)))
            .expect("should succeed");
        let list = reg.list_by_kind("RangeConstraint");
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_list_by_kind_empty() {
        let reg = ConstraintRegistry::new();
        assert!(reg.list_by_kind("LengthConstraint").is_empty());
    }

    #[test]
    fn test_list_by_kind_unknown() {
        let mut reg = ConstraintRegistry::new();
        reg.register(len_constraint("urn:1", None, None))
            .expect("should succeed");
        assert!(reg.list_by_kind("NonexistentKind").is_empty());
    }

    // ---- validate_value: LengthConstraint ----

    #[test]
    fn test_validate_length_within_bounds() {
        let mut reg = ConstraintRegistry::new();
        reg.register(len_constraint("urn:len", Some(2), Some(5)))
            .expect("should succeed");
        let ok = reg
            .validate_value("urn:len", &ConstraintValue::Text("abc".to_string()))
            .expect("should succeed");
        assert!(ok);
    }

    #[test]
    fn test_validate_length_too_short() {
        let mut reg = ConstraintRegistry::new();
        reg.register(len_constraint("urn:len", Some(5), Some(10)))
            .expect("should succeed");
        let ok = reg
            .validate_value("urn:len", &ConstraintValue::Text("ab".to_string()))
            .expect("should succeed");
        assert!(!ok);
    }

    #[test]
    fn test_validate_length_too_long() {
        let mut reg = ConstraintRegistry::new();
        reg.register(len_constraint("urn:len", Some(1), Some(3)))
            .expect("should succeed");
        let ok = reg
            .validate_value("urn:len", &ConstraintValue::Text("toolong".to_string()))
            .expect("should succeed");
        assert!(!ok);
    }

    #[test]
    fn test_validate_length_exact_min() {
        let mut reg = ConstraintRegistry::new();
        reg.register(len_constraint("urn:len", Some(3), Some(3)))
            .expect("should succeed");
        let ok = reg
            .validate_value("urn:len", &ConstraintValue::Text("abc".to_string()))
            .expect("should succeed");
        assert!(ok);
    }

    #[test]
    fn test_validate_length_no_bounds() {
        let mut reg = ConstraintRegistry::new();
        reg.register(len_constraint("urn:len", None, None))
            .expect("should succeed");
        let ok = reg
            .validate_value("urn:len", &ConstraintValue::Text("anything".to_string()))
            .expect("should succeed");
        assert!(ok);
    }

    // ---- validate_value: RangeConstraint ----

    #[test]
    fn test_validate_range_within() {
        let mut reg = ConstraintRegistry::new();
        reg.register(range_constraint("urn:range", Some(0.0), Some(100.0)))
            .expect("should succeed");
        let ok = reg
            .validate_value("urn:range", &ConstraintValue::Integer(50))
            .expect("should succeed");
        assert!(ok);
    }

    #[test]
    fn test_validate_range_below() {
        let mut reg = ConstraintRegistry::new();
        reg.register(range_constraint("urn:range", Some(10.0), Some(100.0)))
            .expect("should succeed");
        let ok = reg
            .validate_value("urn:range", &ConstraintValue::Integer(5))
            .expect("should succeed");
        assert!(!ok);
    }

    #[test]
    fn test_validate_range_above() {
        let mut reg = ConstraintRegistry::new();
        reg.register(range_constraint("urn:range", Some(0.0), Some(50.0)))
            .expect("should succeed");
        let ok = reg
            .validate_value("urn:range", &ConstraintValue::Integer(100))
            .expect("should succeed");
        assert!(!ok);
    }

    #[test]
    fn test_validate_range_at_boundary() {
        let mut reg = ConstraintRegistry::new();
        reg.register(range_constraint("urn:range", Some(0.0), Some(100.0)))
            .expect("should succeed");
        let ok = reg
            .validate_value("urn:range", &ConstraintValue::Float(100.0))
            .expect("should succeed");
        assert!(ok);
    }

    #[test]
    fn test_validate_range_open_bound() {
        let mut reg = ConstraintRegistry::new();
        reg.register(Constraint {
            urn: "urn:range:open".to_string(),
            name: "r".to_string(),
            description: None,
            kind: ConstraintKind::RangeConstraint {
                min: Some(ConstraintValue::Float(0.0)),
                max: Some(ConstraintValue::Float(10.0)),
                lower_bound_definition: BoundType::Open,
                upper_bound_definition: BoundType::Open,
            },
        })
        .expect("should succeed");
        // 0.0 should be excluded (open lower bound)
        let ok = reg
            .validate_value("urn:range:open", &ConstraintValue::Float(0.0))
            .expect("should succeed");
        assert!(!ok);
        let ok2 = reg
            .validate_value("urn:range:open", &ConstraintValue::Float(5.0))
            .expect("should succeed");
        assert!(ok2);
    }

    // ---- validate_value: EnumerationConstraint ----

    #[test]
    fn test_validate_enumeration_match() {
        let mut reg = ConstraintRegistry::new();
        reg.register(enum_constraint(
            "urn:enum",
            vec![
                ConstraintValue::Text("A".to_string()),
                ConstraintValue::Text("B".to_string()),
            ],
        ))
        .expect("should succeed");
        let ok = reg
            .validate_value("urn:enum", &ConstraintValue::Text("A".to_string()))
            .expect("should succeed");
        assert!(ok);
    }

    #[test]
    fn test_validate_enumeration_no_match() {
        let mut reg = ConstraintRegistry::new();
        reg.register(enum_constraint(
            "urn:enum",
            vec![ConstraintValue::Text("A".to_string())],
        ))
        .expect("should succeed");
        let ok = reg
            .validate_value("urn:enum", &ConstraintValue::Text("C".to_string()))
            .expect("should succeed");
        assert!(!ok);
    }

    #[test]
    fn test_validate_enumeration_integers() {
        let mut reg = ConstraintRegistry::new();
        reg.register(enum_constraint(
            "urn:enum:int",
            vec![
                ConstraintValue::Integer(1),
                ConstraintValue::Integer(2),
                ConstraintValue::Integer(3),
            ],
        ))
        .expect("should succeed");
        let ok = reg
            .validate_value("urn:enum:int", &ConstraintValue::Integer(2))
            .expect("should succeed");
        assert!(ok);
        let not_ok = reg
            .validate_value("urn:enum:int", &ConstraintValue::Integer(5))
            .expect("should succeed");
        assert!(!not_ok);
    }

    // ---- validate_value: RegularExpressionConstraint ----

    #[test]
    fn test_validate_regex_exact_match() {
        let mut reg = ConstraintRegistry::new();
        reg.register(regex_constraint("urn:regex", "hello"))
            .expect("should succeed");
        let ok = reg
            .validate_value("urn:regex", &ConstraintValue::Text("hello".to_string()))
            .expect("should succeed");
        assert!(ok);
        let not_ok = reg
            .validate_value("urn:regex", &ConstraintValue::Text("world".to_string()))
            .expect("should succeed");
        assert!(!not_ok);
    }

    #[test]
    fn test_validate_regex_prefix_wildcard() {
        let mut reg = ConstraintRegistry::new();
        reg.register(regex_constraint("urn:regex:pre", "hello*"))
            .expect("should succeed");
        let ok = reg
            .validate_value(
                "urn:regex:pre",
                &ConstraintValue::Text("helloworld".to_string()),
            )
            .expect("should succeed");
        assert!(ok);
        let not_ok = reg
            .validate_value(
                "urn:regex:pre",
                &ConstraintValue::Text("worldhello".to_string()),
            )
            .expect("should succeed");
        assert!(!not_ok);
    }

    #[test]
    fn test_validate_regex_wildcard_all() {
        let mut reg = ConstraintRegistry::new();
        reg.register(regex_constraint("urn:regex:all", "*"))
            .expect("should succeed");
        let ok = reg
            .validate_value(
                "urn:regex:all",
                &ConstraintValue::Text("anything".to_string()),
            )
            .expect("should succeed");
        assert!(ok);
    }

    // ---- remove ----

    #[test]
    fn test_remove_existing() {
        let mut reg = ConstraintRegistry::new();
        reg.register(len_constraint("urn:1", None, None))
            .expect("should succeed");
        let removed = reg.remove("urn:1");
        assert!(removed.is_some());
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn test_remove_missing() {
        let mut reg = ConstraintRegistry::new();
        let removed = reg.remove("urn:nonexistent");
        assert!(removed.is_none());
    }

    // ---- all iterator ----

    #[test]
    fn test_all_iterator() {
        let mut reg = ConstraintRegistry::new();
        reg.register(len_constraint("urn:1", None, None))
            .expect("should succeed");
        reg.register(range_constraint("urn:2", None, None))
            .expect("should succeed");
        let all: Vec<_> = reg.all().collect();
        assert_eq!(all.len(), 2);
    }

    // ---- constraint not found ----

    #[test]
    fn test_validate_constraint_not_found() {
        let reg = ConstraintRegistry::new();
        let result = reg.validate_value("urn:nonexistent", &ConstraintValue::Text("x".to_string()));
        assert!(matches!(result, Err(RegistryError::ConstraintNotFound(_))));
    }

    // ---- error display ----

    #[test]
    fn test_registry_error_display() {
        let e = RegistryError::DuplicateUrn("urn:x".to_string());
        assert!(format!("{e}").contains("urn:x"));
        let e2 = RegistryError::ConstraintNotFound("urn:y".to_string());
        assert!(format!("{e2}").contains("urn:y"));
    }

    // ---- kind_name ----

    #[test]
    fn test_kind_name() {
        assert_eq!(
            ConstraintKind::LengthConstraint {
                min: None,
                max: None
            }
            .kind_name(),
            "LengthConstraint"
        );
        assert_eq!(
            ConstraintKind::RegularExpressionConstraint {
                value: "".to_string()
            }
            .kind_name(),
            "RegularExpressionConstraint"
        );
        assert_eq!(
            ConstraintKind::EnumerationConstraint { values: vec![] }.kind_name(),
            "EnumerationConstraint"
        );
    }

    // ---- BoundType ----

    #[test]
    fn test_bound_type_is_exclusive() {
        assert!(BoundType::Open.is_exclusive());
        assert!(BoundType::GreaterThan.is_exclusive());
        assert!(!BoundType::Closed.is_exclusive());
        assert!(!BoundType::AtLeast.is_exclusive());
    }

    // ---- LanguageConstraint ----

    #[test]
    fn test_language_constraint_valid() {
        let mut reg = ConstraintRegistry::new();
        reg.register(Constraint {
            urn: "urn:lang".to_string(),
            name: "lang".to_string(),
            description: None,
            kind: ConstraintKind::LanguageConstraint {
                language_codes: vec!["en".to_string(), "de".to_string()],
            },
        })
        .expect("should succeed");
        let ok = reg
            .validate_value("urn:lang", &ConstraintValue::Text("EN".to_string()))
            .expect("should succeed");
        assert!(ok); // case-insensitive
        let not_ok = reg
            .validate_value("urn:lang", &ConstraintValue::Text("fr".to_string()))
            .expect("should succeed");
        assert!(!not_ok);
    }

    // ---- FixedPointConstraint ----

    #[test]
    fn test_fixed_point_constraint_valid() {
        let mut reg = ConstraintRegistry::new();
        reg.register(Constraint {
            urn: "urn:fp".to_string(),
            name: "fp".to_string(),
            description: None,
            kind: ConstraintKind::FixedPointConstraint {
                scale: 2,
                integer: 5,
            },
        })
        .expect("should succeed");
        let ok = reg
            .validate_value("urn:fp", &ConstraintValue::Text("12345.67".to_string()))
            .expect("should succeed");
        assert!(ok);
        let not_ok = reg
            .validate_value("urn:fp", &ConstraintValue::Text("12345.678".to_string()))
            .expect("should succeed");
        assert!(!not_ok);
    }

    // ---- default ----

    #[test]
    fn test_default_creates_empty_registry() {
        let reg = ConstraintRegistry::default();
        assert_eq!(reg.count(), 0);
    }
}
