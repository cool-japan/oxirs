//! GraphQL argument type coercion and validation.
//!
//! Implements the coercion rules from the GraphQL specification:
//! - Int→Float widening
//! - String→ID coercion
//! - Null rejection for NonNull types
//! - Enum value validation
//! - Recursive List and InputObject coercion

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// The five built-in GraphQL scalar types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalarType {
    Int,
    Float,
    String,
    Boolean,
    ID,
}

impl ScalarType {
    /// Return the canonical GraphQL type name.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Int => "Int",
            Self::Float => "Float",
            Self::String => "String",
            Self::Boolean => "Boolean",
            Self::ID => "ID",
        }
    }
}

impl std::fmt::Display for ScalarType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.type_name())
    }
}

/// A GraphQL type descriptor.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphqlType {
    /// A scalar leaf type.
    Scalar(ScalarType),
    /// The inner type cannot be null.
    NonNull(Box<GraphqlType>),
    /// A homogeneous list.
    List(Box<GraphqlType>),
    /// A named enum type.
    Enum(String),
    /// A named input object type.
    InputObject(String),
}

impl GraphqlType {
    /// Human-readable name for error messages.
    pub fn display_name(&self) -> String {
        match self {
            Self::Scalar(s) => s.to_string(),
            Self::NonNull(inner) => format!("{}!", inner.display_name()),
            Self::List(inner) => format!("[{}]", inner.display_name()),
            Self::Enum(n) => n.clone(),
            Self::InputObject(n) => n.clone(),
        }
    }
}

/// A runtime GraphQL argument value.
#[derive(Debug, Clone, PartialEq)]
pub enum ArgumentValue {
    Int(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Null,
    List(Vec<ArgumentValue>),
    Object(HashMap<String, ArgumentValue>),
}

impl ArgumentValue {
    /// Descriptive type name used in error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Int(_) => "Int",
            Self::Float(_) => "Float",
            Self::String(_) => "String",
            Self::Boolean(_) => "Boolean",
            Self::Null => "Null",
            Self::List(_) => "List",
            Self::Object(_) => "Object",
        }
    }
}

/// Errors produced during coercion.
#[derive(Debug, Clone, PartialEq)]
pub enum CoercionError {
    /// The value's type does not match the target type.
    TypeMismatch { expected: String, got: String },
    /// A `null` value was provided for a `NonNull` type.
    NullOnNonNull(String),
    /// The string value is not a member of the named enum.
    InvalidEnumValue { value: String, enum_type: String },
    /// An integer value overflowed the target range (i32 for GraphQL Int).
    IntegerOverflow(String),
    /// A string could not be parsed into the required scalar type.
    InvalidFormat(String),
}

impl std::fmt::Display for CoercionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeMismatch { expected, got } => {
                write!(f, "Type mismatch: expected {expected}, got {got}")
            }
            Self::NullOnNonNull(t) => write!(f, "Null value on NonNull type '{t}'"),
            Self::InvalidEnumValue { value, enum_type } => {
                write!(f, "'{value}' is not a valid value for enum '{enum_type}'")
            }
            Self::IntegerOverflow(s) => write!(f, "Integer overflow: {s}"),
            Self::InvalidFormat(s) => write!(f, "Invalid format: {s}"),
        }
    }
}

impl std::error::Error for CoercionError {}

// ────────────────────────────────────────────────────────────────────────────
// ArgumentCoercer
// ────────────────────────────────────────────────────────────────────────────

/// Coerces and validates GraphQL argument values against a target type.
#[derive(Debug, Default)]
pub struct ArgumentCoercer {
    /// Registered enum definitions: name → allowed value strings.
    enums: HashMap<String, Vec<String>>,
}

impl ArgumentCoercer {
    /// Create a new, empty coercer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an enum type with its allowed values.
    pub fn register_enum(&mut self, name: &str, values: Vec<String>) {
        self.enums.insert(name.to_string(), values);
    }

    /// Coerce `value` to `target_type`, following GraphQL coercion rules.
    pub fn coerce(
        &self,
        value: &ArgumentValue,
        target_type: &GraphqlType,
    ) -> Result<ArgumentValue, CoercionError> {
        match target_type {
            // ── NonNull ───────────────────────────────────────────────────────
            GraphqlType::NonNull(inner) => {
                if *value == ArgumentValue::Null {
                    return Err(CoercionError::NullOnNonNull(inner.display_name()));
                }
                self.coerce(value, inner)
            }

            // ── Null passthrough for nullable types ───────────────────────────
            _ if *value == ArgumentValue::Null => Ok(ArgumentValue::Null),

            // ── Scalar ────────────────────────────────────────────────────────
            GraphqlType::Scalar(scalar) => self.coerce_scalar(value, scalar),

            // ── List ──────────────────────────────────────────────────────────
            GraphqlType::List(item_type) => match value {
                ArgumentValue::List(items) => {
                    let coerced = items
                        .iter()
                        .map(|item| self.coerce(item, item_type))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(ArgumentValue::List(coerced))
                }
                // Spec: a non-list value is coerced into a single-element list.
                other => {
                    let coerced = self.coerce(other, item_type)?;
                    Ok(ArgumentValue::List(vec![coerced]))
                }
            },

            // ── Enum ──────────────────────────────────────────────────────────
            GraphqlType::Enum(enum_name) => {
                let s = match value {
                    ArgumentValue::String(s) => s.clone(),
                    other => {
                        return Err(CoercionError::TypeMismatch {
                            expected: enum_name.clone(),
                            got: other.type_name().to_string(),
                        });
                    }
                };
                self.validate_enum(&s, enum_name)?;
                Ok(ArgumentValue::String(s))
            }

            // ── InputObject ───────────────────────────────────────────────────
            GraphqlType::InputObject(type_name) => match value {
                ArgumentValue::Object(_) => Ok(value.clone()),
                other => Err(CoercionError::TypeMismatch {
                    expected: type_name.clone(),
                    got: other.type_name().to_string(),
                }),
            },
        }
    }

    /// Coerce a string (e.g. from a query variable or URL parameter) to
    /// `target_type`.
    pub fn coerce_string(
        &self,
        s: &str,
        target_type: &GraphqlType,
    ) -> Result<ArgumentValue, CoercionError> {
        match target_type {
            GraphqlType::NonNull(inner) => {
                if s.is_empty() {
                    return Err(CoercionError::NullOnNonNull(inner.display_name()));
                }
                self.coerce_string(s, inner)
            }
            GraphqlType::Scalar(scalar) => self.parse_string_as_scalar(s, scalar),
            GraphqlType::Enum(enum_name) => {
                self.validate_enum(s, enum_name)?;
                Ok(ArgumentValue::String(s.to_string()))
            }
            GraphqlType::List(item_type) => {
                // Try to parse as a single item wrapped in a list.
                let item = self.coerce_string(s, item_type)?;
                Ok(ArgumentValue::List(vec![item]))
            }
            GraphqlType::InputObject(n) => Err(CoercionError::InvalidFormat(format!(
                "Cannot coerce string to InputObject '{n}'"
            ))),
        }
    }

    /// Return `true` if `value` can be coerced to `target_type` without error.
    pub fn is_coercible(&self, value: &ArgumentValue, target_type: &GraphqlType) -> bool {
        self.coerce(value, target_type).is_ok()
    }

    // ── private helpers ───────────────────────────────────────────────────────

    fn coerce_scalar(
        &self,
        value: &ArgumentValue,
        scalar: &ScalarType,
    ) -> Result<ArgumentValue, CoercionError> {
        match scalar {
            ScalarType::Int => match value {
                ArgumentValue::Int(n) => {
                    // GraphQL Int is a 32-bit signed integer.
                    if *n < i64::from(i32::MIN) || *n > i64::from(i32::MAX) {
                        return Err(CoercionError::IntegerOverflow(n.to_string()));
                    }
                    Ok(ArgumentValue::Int(*n))
                }
                other => Err(CoercionError::TypeMismatch {
                    expected: "Int".to_string(),
                    got: other.type_name().to_string(),
                }),
            },
            ScalarType::Float => match value {
                // Int→Float widening is allowed by the spec.
                ArgumentValue::Int(n) => Ok(ArgumentValue::Float(*n as f64)),
                ArgumentValue::Float(f) => Ok(ArgumentValue::Float(*f)),
                other => Err(CoercionError::TypeMismatch {
                    expected: "Float".to_string(),
                    got: other.type_name().to_string(),
                }),
            },
            ScalarType::String => match value {
                ArgumentValue::String(s) => Ok(ArgumentValue::String(s.clone())),
                other => Err(CoercionError::TypeMismatch {
                    expected: "String".to_string(),
                    got: other.type_name().to_string(),
                }),
            },
            ScalarType::Boolean => match value {
                ArgumentValue::Boolean(b) => Ok(ArgumentValue::Boolean(*b)),
                other => Err(CoercionError::TypeMismatch {
                    expected: "Boolean".to_string(),
                    got: other.type_name().to_string(),
                }),
            },
            ScalarType::ID => match value {
                // String→ID and Int→ID are both allowed.
                ArgumentValue::String(s) => Ok(ArgumentValue::String(s.clone())),
                ArgumentValue::Int(n) => Ok(ArgumentValue::String(n.to_string())),
                other => Err(CoercionError::TypeMismatch {
                    expected: "ID".to_string(),
                    got: other.type_name().to_string(),
                }),
            },
        }
    }

    fn parse_string_as_scalar(
        &self,
        s: &str,
        scalar: &ScalarType,
    ) -> Result<ArgumentValue, CoercionError> {
        match scalar {
            ScalarType::Int => {
                let n: i64 = s.parse().map_err(|_| {
                    CoercionError::InvalidFormat(format!("Cannot parse '{s}' as Int"))
                })?;
                if n < i64::from(i32::MIN) || n > i64::from(i32::MAX) {
                    return Err(CoercionError::IntegerOverflow(s.to_string()));
                }
                Ok(ArgumentValue::Int(n))
            }
            ScalarType::Float => {
                let f: f64 = s.parse().map_err(|_| {
                    CoercionError::InvalidFormat(format!("Cannot parse '{s}' as Float"))
                })?;
                Ok(ArgumentValue::Float(f))
            }
            ScalarType::String | ScalarType::ID => Ok(ArgumentValue::String(s.to_string())),
            ScalarType::Boolean => match s {
                "true" | "1" => Ok(ArgumentValue::Boolean(true)),
                "false" | "0" => Ok(ArgumentValue::Boolean(false)),
                _ => Err(CoercionError::InvalidFormat(format!(
                    "Cannot parse '{s}' as Boolean"
                ))),
            },
        }
    }

    fn validate_enum(&self, value: &str, enum_name: &str) -> Result<(), CoercionError> {
        if let Some(allowed) = self.enums.get(enum_name) {
            if !allowed.contains(&value.to_string()) {
                return Err(CoercionError::InvalidEnumValue {
                    value: value.to_string(),
                    enum_type: enum_name.to_string(),
                });
            }
        }
        // If the enum isn't registered we accept any value (lenient mode).
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn coercer_with_status_enum() -> ArgumentCoercer {
        let mut c = ArgumentCoercer::new();
        c.register_enum(
            "Status",
            vec![
                "ACTIVE".to_string(),
                "INACTIVE".to_string(),
                "PENDING".to_string(),
            ],
        );
        c
    }

    // ── ScalarType ────────────────────────────────────────────────────────────

    #[test]
    fn test_scalar_type_name_int() {
        assert_eq!(ScalarType::Int.type_name(), "Int");
    }

    #[test]
    fn test_scalar_type_name_float() {
        assert_eq!(ScalarType::Float.type_name(), "Float");
    }

    #[test]
    fn test_scalar_type_display() {
        assert_eq!(ScalarType::Boolean.to_string(), "Boolean");
        assert_eq!(ScalarType::ID.to_string(), "ID");
    }

    // ── GraphqlType::display_name ─────────────────────────────────────────────

    #[test]
    fn test_graphql_type_display_scalar() {
        let t = GraphqlType::Scalar(ScalarType::String);
        assert_eq!(t.display_name(), "String");
    }

    #[test]
    fn test_graphql_type_display_nonnull() {
        let t = GraphqlType::NonNull(Box::new(GraphqlType::Scalar(ScalarType::Int)));
        assert_eq!(t.display_name(), "Int!");
    }

    #[test]
    fn test_graphql_type_display_list() {
        let t = GraphqlType::List(Box::new(GraphqlType::Scalar(ScalarType::String)));
        assert_eq!(t.display_name(), "[String]");
    }

    #[test]
    fn test_graphql_type_display_enum() {
        let t = GraphqlType::Enum("Status".to_string());
        assert_eq!(t.display_name(), "Status");
    }

    // ── Int coercion ──────────────────────────────────────────────────────────

    #[test]
    fn test_coerce_int_to_int() {
        let c = ArgumentCoercer::new();
        let v = c
            .coerce(
                &ArgumentValue::Int(42),
                &GraphqlType::Scalar(ScalarType::Int),
            )
            .expect("coerce int");
        assert_eq!(v, ArgumentValue::Int(42));
    }

    #[test]
    fn test_coerce_int_overflow_i32() {
        let c = ArgumentCoercer::new();
        let big = i64::from(i32::MAX) + 1;
        let err = c
            .coerce(
                &ArgumentValue::Int(big),
                &GraphqlType::Scalar(ScalarType::Int),
            )
            .expect_err("overflow");
        assert!(matches!(err, CoercionError::IntegerOverflow(_)));
    }

    #[test]
    fn test_coerce_int_to_float_widening() {
        let c = ArgumentCoercer::new();
        let v = c
            .coerce(
                &ArgumentValue::Int(10),
                &GraphqlType::Scalar(ScalarType::Float),
            )
            .expect("int to float");
        assert_eq!(v, ArgumentValue::Float(10.0));
    }

    #[test]
    fn test_coerce_float_to_int_fails() {
        let c = ArgumentCoercer::new();
        let err = c
            .coerce(
                &ArgumentValue::Float(2.71),
                &GraphqlType::Scalar(ScalarType::Int),
            )
            .expect_err("float to int");
        assert!(matches!(err, CoercionError::TypeMismatch { .. }));
    }

    // ── Float coercion ────────────────────────────────────────────────────────

    #[test]
    fn test_coerce_float_to_float() {
        let c = ArgumentCoercer::new();
        let v = c
            .coerce(
                &ArgumentValue::Float(2.71),
                &GraphqlType::Scalar(ScalarType::Float),
            )
            .expect("float");
        assert_eq!(v, ArgumentValue::Float(2.71));
    }

    // ── String coercion ───────────────────────────────────────────────────────

    #[test]
    fn test_coerce_string_to_string() {
        let c = ArgumentCoercer::new();
        let v = c
            .coerce(
                &ArgumentValue::String("hello".to_string()),
                &GraphqlType::Scalar(ScalarType::String),
            )
            .expect("string");
        assert_eq!(v, ArgumentValue::String("hello".to_string()));
    }

    #[test]
    fn test_coerce_int_to_string_fails() {
        let c = ArgumentCoercer::new();
        let err = c
            .coerce(
                &ArgumentValue::Int(1),
                &GraphqlType::Scalar(ScalarType::String),
            )
            .expect_err("int to string");
        assert!(matches!(err, CoercionError::TypeMismatch { .. }));
    }

    // ── Boolean coercion ──────────────────────────────────────────────────────

    #[test]
    fn test_coerce_bool_true() {
        let c = ArgumentCoercer::new();
        let v = c
            .coerce(
                &ArgumentValue::Boolean(true),
                &GraphqlType::Scalar(ScalarType::Boolean),
            )
            .expect("bool");
        assert_eq!(v, ArgumentValue::Boolean(true));
    }

    #[test]
    fn test_coerce_string_to_bool_fails() {
        let c = ArgumentCoercer::new();
        let err = c
            .coerce(
                &ArgumentValue::String("true".to_string()),
                &GraphqlType::Scalar(ScalarType::Boolean),
            )
            .expect_err("string to bool");
        assert!(matches!(err, CoercionError::TypeMismatch { .. }));
    }

    // ── ID coercion ───────────────────────────────────────────────────────────

    #[test]
    fn test_coerce_string_to_id() {
        let c = ArgumentCoercer::new();
        let v = c
            .coerce(
                &ArgumentValue::String("abc-123".to_string()),
                &GraphqlType::Scalar(ScalarType::ID),
            )
            .expect("string to ID");
        assert_eq!(v, ArgumentValue::String("abc-123".to_string()));
    }

    #[test]
    fn test_coerce_int_to_id() {
        let c = ArgumentCoercer::new();
        let v = c
            .coerce(
                &ArgumentValue::Int(999),
                &GraphqlType::Scalar(ScalarType::ID),
            )
            .expect("int to ID");
        assert_eq!(v, ArgumentValue::String("999".to_string()));
    }

    #[test]
    fn test_coerce_bool_to_id_fails() {
        let c = ArgumentCoercer::new();
        let err = c
            .coerce(
                &ArgumentValue::Boolean(false),
                &GraphqlType::Scalar(ScalarType::ID),
            )
            .expect_err("bool to ID");
        assert!(matches!(err, CoercionError::TypeMismatch { .. }));
    }

    // ── NonNull ───────────────────────────────────────────────────────────────

    #[test]
    fn test_nonnull_rejects_null() {
        let c = ArgumentCoercer::new();
        let err = c
            .coerce(
                &ArgumentValue::Null,
                &GraphqlType::NonNull(Box::new(GraphqlType::Scalar(ScalarType::String))),
            )
            .expect_err("null on nonnull");
        assert!(matches!(err, CoercionError::NullOnNonNull(_)));
    }

    #[test]
    fn test_nonnull_accepts_value() {
        let c = ArgumentCoercer::new();
        let v = c
            .coerce(
                &ArgumentValue::Int(1),
                &GraphqlType::NonNull(Box::new(GraphqlType::Scalar(ScalarType::Int))),
            )
            .expect("nonnull ok");
        assert_eq!(v, ArgumentValue::Int(1));
    }

    // ── Null on nullable ──────────────────────────────────────────────────────

    #[test]
    fn test_null_on_nullable_type_ok() {
        let c = ArgumentCoercer::new();
        let v = c
            .coerce(
                &ArgumentValue::Null,
                &GraphqlType::Scalar(ScalarType::String),
            )
            .expect("null ok on nullable");
        assert_eq!(v, ArgumentValue::Null);
    }

    // ── List ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_coerce_list_of_ints() {
        let c = ArgumentCoercer::new();
        let v = c
            .coerce(
                &ArgumentValue::List(vec![
                    ArgumentValue::Int(1),
                    ArgumentValue::Int(2),
                    ArgumentValue::Int(3),
                ]),
                &GraphqlType::List(Box::new(GraphqlType::Scalar(ScalarType::Int))),
            )
            .expect("list of ints");
        assert_eq!(
            v,
            ArgumentValue::List(vec![
                ArgumentValue::Int(1),
                ArgumentValue::Int(2),
                ArgumentValue::Int(3),
            ])
        );
    }

    #[test]
    fn test_coerce_single_value_into_list() {
        let c = ArgumentCoercer::new();
        let v = c
            .coerce(
                &ArgumentValue::Int(7),
                &GraphqlType::List(Box::new(GraphqlType::Scalar(ScalarType::Float))),
            )
            .expect("single to list");
        assert_eq!(v, ArgumentValue::List(vec![ArgumentValue::Float(7.0)]));
    }

    #[test]
    fn test_coerce_list_item_error_propagates() {
        let c = ArgumentCoercer::new();
        let err = c
            .coerce(
                &ArgumentValue::List(vec![
                    ArgumentValue::Int(1),
                    ArgumentValue::String("bad".to_string()),
                ]),
                &GraphqlType::List(Box::new(GraphqlType::Scalar(ScalarType::Int))),
            )
            .expect_err("bad list item");
        assert!(matches!(err, CoercionError::TypeMismatch { .. }));
    }

    // ── Enum ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_coerce_valid_enum_value() {
        let c = coercer_with_status_enum();
        let v = c
            .coerce(
                &ArgumentValue::String("ACTIVE".to_string()),
                &GraphqlType::Enum("Status".to_string()),
            )
            .expect("valid enum");
        assert_eq!(v, ArgumentValue::String("ACTIVE".to_string()));
    }

    #[test]
    fn test_coerce_invalid_enum_value() {
        let c = coercer_with_status_enum();
        let err = c
            .coerce(
                &ArgumentValue::String("DELETED".to_string()),
                &GraphqlType::Enum("Status".to_string()),
            )
            .expect_err("invalid enum");
        assert!(matches!(err, CoercionError::InvalidEnumValue { .. }));
    }

    #[test]
    fn test_coerce_int_to_enum_fails() {
        let c = coercer_with_status_enum();
        let err = c
            .coerce(
                &ArgumentValue::Int(0),
                &GraphqlType::Enum("Status".to_string()),
            )
            .expect_err("int to enum");
        assert!(matches!(err, CoercionError::TypeMismatch { .. }));
    }

    #[test]
    fn test_coerce_unregistered_enum_accepts_any_string() {
        let c = ArgumentCoercer::new();
        let v = c
            .coerce(
                &ArgumentValue::String("ANYTHING".to_string()),
                &GraphqlType::Enum("UnknownEnum".to_string()),
            )
            .expect("unknown enum lenient");
        assert_eq!(v, ArgumentValue::String("ANYTHING".to_string()));
    }

    // ── InputObject ───────────────────────────────────────────────────────────

    #[test]
    fn test_coerce_input_object() {
        let c = ArgumentCoercer::new();
        let mut obj = HashMap::new();
        obj.insert(
            "name".to_string(),
            ArgumentValue::String("Alice".to_string()),
        );
        let v = c
            .coerce(
                &ArgumentValue::Object(obj.clone()),
                &GraphqlType::InputObject("PersonInput".to_string()),
            )
            .expect("input object");
        assert_eq!(v, ArgumentValue::Object(obj));
    }

    #[test]
    fn test_coerce_non_object_to_input_object_fails() {
        let c = ArgumentCoercer::new();
        let err = c
            .coerce(
                &ArgumentValue::String("wrong".to_string()),
                &GraphqlType::InputObject("PersonInput".to_string()),
            )
            .expect_err("wrong input object");
        assert!(matches!(err, CoercionError::TypeMismatch { .. }));
    }

    // ── coerce_string ─────────────────────────────────────────────────────────

    #[test]
    fn test_coerce_string_as_int() {
        let c = ArgumentCoercer::new();
        let v = c
            .coerce_string("42", &GraphqlType::Scalar(ScalarType::Int))
            .expect("str to int");
        assert_eq!(v, ArgumentValue::Int(42));
    }

    #[test]
    fn test_coerce_string_as_float() {
        let c = ArgumentCoercer::new();
        let v = c
            .coerce_string("2.71", &GraphqlType::Scalar(ScalarType::Float))
            .expect("str to float");
        assert_eq!(v, ArgumentValue::Float(2.71));
    }

    #[test]
    fn test_coerce_string_as_bool_true() {
        let c = ArgumentCoercer::new();
        assert_eq!(
            c.coerce_string("true", &GraphqlType::Scalar(ScalarType::Boolean))
                .expect("true"),
            ArgumentValue::Boolean(true)
        );
    }

    #[test]
    fn test_coerce_string_as_bool_false() {
        let c = ArgumentCoercer::new();
        assert_eq!(
            c.coerce_string("0", &GraphqlType::Scalar(ScalarType::Boolean))
                .expect("0"),
            ArgumentValue::Boolean(false)
        );
    }

    #[test]
    fn test_coerce_string_invalid_bool() {
        let c = ArgumentCoercer::new();
        let err = c
            .coerce_string("yes", &GraphqlType::Scalar(ScalarType::Boolean))
            .expect_err("invalid bool");
        assert!(matches!(err, CoercionError::InvalidFormat(_)));
    }

    #[test]
    fn test_coerce_string_as_id() {
        let c = ArgumentCoercer::new();
        let v = c
            .coerce_string("user-1", &GraphqlType::Scalar(ScalarType::ID))
            .expect("str to id");
        assert_eq!(v, ArgumentValue::String("user-1".to_string()));
    }

    #[test]
    fn test_coerce_string_overflow() {
        let c = ArgumentCoercer::new();
        let big = (i64::from(i32::MAX) + 2).to_string();
        let err = c
            .coerce_string(&big, &GraphqlType::Scalar(ScalarType::Int))
            .expect_err("overflow");
        assert!(matches!(err, CoercionError::IntegerOverflow(_)));
    }

    #[test]
    fn test_coerce_string_nonnull_empty_fails() {
        let c = ArgumentCoercer::new();
        let err = c
            .coerce_string(
                "",
                &GraphqlType::NonNull(Box::new(GraphqlType::Scalar(ScalarType::String))),
            )
            .expect_err("empty nonnull");
        assert!(matches!(err, CoercionError::NullOnNonNull(_)));
    }

    #[test]
    fn test_coerce_string_enum() {
        let c = coercer_with_status_enum();
        let v = c
            .coerce_string("PENDING", &GraphqlType::Enum("Status".to_string()))
            .expect("enum via string");
        assert_eq!(v, ArgumentValue::String("PENDING".to_string()));
    }

    #[test]
    fn test_coerce_string_into_list() {
        let c = ArgumentCoercer::new();
        let v = c
            .coerce_string(
                "99",
                &GraphqlType::List(Box::new(GraphqlType::Scalar(ScalarType::Int))),
            )
            .expect("str to list");
        assert_eq!(v, ArgumentValue::List(vec![ArgumentValue::Int(99)]));
    }

    // ── is_coercible ──────────────────────────────────────────────────────────

    #[test]
    fn test_is_coercible_int_to_float() {
        let c = ArgumentCoercer::new();
        assert!(c.is_coercible(
            &ArgumentValue::Int(5),
            &GraphqlType::Scalar(ScalarType::Float)
        ));
    }

    #[test]
    fn test_is_coercible_string_to_id() {
        let c = ArgumentCoercer::new();
        assert!(c.is_coercible(
            &ArgumentValue::String("x".to_string()),
            &GraphqlType::Scalar(ScalarType::ID)
        ));
    }

    #[test]
    fn test_is_coercible_null_on_nonnull_false() {
        let c = ArgumentCoercer::new();
        assert!(!c.is_coercible(
            &ArgumentValue::Null,
            &GraphqlType::NonNull(Box::new(GraphqlType::Scalar(ScalarType::Int)))
        ));
    }

    #[test]
    fn test_is_coercible_float_to_int_false() {
        let c = ArgumentCoercer::new();
        assert!(!c.is_coercible(
            &ArgumentValue::Float(1.5),
            &GraphqlType::Scalar(ScalarType::Int)
        ));
    }

    // ── CoercionError display ─────────────────────────────────────────────────

    #[test]
    fn test_error_display_type_mismatch() {
        let e = CoercionError::TypeMismatch {
            expected: "Int".to_string(),
            got: "String".to_string(),
        };
        assert!(e.to_string().contains("Int"));
        assert!(e.to_string().contains("String"));
    }

    #[test]
    fn test_error_display_null_on_nonnull() {
        let e = CoercionError::NullOnNonNull("String!".to_string());
        assert!(e.to_string().contains("String!"));
    }

    #[test]
    fn test_error_display_invalid_enum() {
        let e = CoercionError::InvalidEnumValue {
            value: "BAD".to_string(),
            enum_type: "Status".to_string(),
        };
        assert!(e.to_string().contains("BAD"));
        assert!(e.to_string().contains("Status"));
    }

    #[test]
    fn test_argument_value_type_name_null() {
        assert_eq!(ArgumentValue::Null.type_name(), "Null");
    }

    #[test]
    fn test_argument_value_type_name_object() {
        assert_eq!(ArgumentValue::Object(HashMap::new()).type_name(), "Object");
    }

    // ── edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_coerce_negative_int() {
        let c = ArgumentCoercer::new();
        let v = c
            .coerce(
                &ArgumentValue::Int(-100),
                &GraphqlType::Scalar(ScalarType::Int),
            )
            .expect("negative int");
        assert_eq!(v, ArgumentValue::Int(-100));
    }

    #[test]
    fn test_coerce_zero_int() {
        let c = ArgumentCoercer::new();
        let v = c
            .coerce(
                &ArgumentValue::Int(0),
                &GraphqlType::Scalar(ScalarType::Int),
            )
            .expect("zero");
        assert_eq!(v, ArgumentValue::Int(0));
    }

    #[test]
    fn test_coerce_empty_list() {
        let c = ArgumentCoercer::new();
        let v = c
            .coerce(
                &ArgumentValue::List(vec![]),
                &GraphqlType::List(Box::new(GraphqlType::Scalar(ScalarType::Int))),
            )
            .expect("empty list");
        assert_eq!(v, ArgumentValue::List(vec![]));
    }

    #[test]
    fn test_coerce_nonnull_list_null_fails() {
        let c = ArgumentCoercer::new();
        let err = c
            .coerce(
                &ArgumentValue::Null,
                &GraphqlType::NonNull(Box::new(GraphqlType::List(Box::new(GraphqlType::Scalar(
                    ScalarType::Int,
                ))))),
            )
            .expect_err("null on nonnull list");
        assert!(matches!(err, CoercionError::NullOnNonNull(_)));
    }
}
