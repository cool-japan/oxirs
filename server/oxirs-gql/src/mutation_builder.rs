//! GraphQL mutation builder with argument validation.
//!
//! Provides a fluent builder API for constructing GraphQL mutation documents,
//! with validation of required arguments and detection of duplicate field names.

use std::collections::HashMap;

/// A value that can be passed as a GraphQL argument.
#[derive(Debug, Clone, PartialEq)]
pub enum ArgValue {
    /// String scalar.
    String(String),
    /// Integer scalar.
    Int(i64),
    /// Float scalar.
    Float(f64),
    /// Boolean scalar.
    Bool(bool),
    /// Null value.
    Null,
    /// List of values.
    List(Vec<ArgValue>),
    /// Input object.
    Object(HashMap<String, ArgValue>),
}

impl ArgValue {
    /// Render the value as a GraphQL literal string.
    pub fn to_graphql(&self) -> String {
        match self {
            ArgValue::String(s) => format!("\"{}\"", s.replace('"', "\\\"")),
            ArgValue::Int(n) => n.to_string(),
            ArgValue::Float(f) => format!("{f}"),
            ArgValue::Bool(b) => b.to_string(),
            ArgValue::Null => "null".to_string(),
            ArgValue::List(items) => {
                let inner: Vec<String> = items.iter().map(|v| v.to_graphql()).collect();
                format!("[{}]", inner.join(", "))
            }
            ArgValue::Object(fields) => {
                let mut pairs: Vec<String> = fields
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v.to_graphql()))
                    .collect();
                pairs.sort(); // deterministic output
                format!("{{{}}}", pairs.join(", "))
            }
        }
    }
}

/// A single argument for a mutation field.
#[derive(Debug, Clone)]
pub struct MutationArg {
    /// Argument name.
    pub name: String,
    /// Argument value.
    pub value: ArgValue,
    /// Whether this argument must be provided.
    pub required: bool,
}

/// A field (selection) within a mutation.
#[derive(Debug, Clone)]
pub struct MutationField {
    /// Field name.
    pub name: String,
    /// Optional alias.
    pub alias: Option<String>,
    /// Arguments passed to the field.
    pub args: Vec<MutationArg>,
    /// Sub-field selections (leaf field names).
    pub selection: Vec<String>,
}

/// A complete mutation document.
#[derive(Debug, Clone)]
pub struct Mutation {
    /// Mutation operation name.
    pub name: String,
    /// Top-level fields of the mutation.
    pub fields: Vec<MutationField>,
    /// Variables map for parameterized mutations.
    pub variables: HashMap<String, ArgValue>,
}

/// Errors that can occur while building a mutation.
#[derive(Debug, Clone, PartialEq)]
pub enum MutationError {
    /// A required argument was not supplied.
    MissingRequiredArg(String),
    /// The field name is not valid.
    InvalidField(String),
    /// The same field name appears more than once.
    DuplicateField(String),
}

impl std::fmt::Display for MutationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MutationError::MissingRequiredArg(a) => {
                write!(f, "Missing required argument: {a}")
            }
            MutationError::InvalidField(field) => write!(f, "Invalid field name: {field}"),
            MutationError::DuplicateField(field) => {
                write!(f, "Duplicate field name: {field}")
            }
        }
    }
}

impl std::error::Error for MutationError {}

// --- Internal builder helpers ---

#[derive(Default)]
struct FieldBuilder {
    name: Option<String>,
    alias: Option<String>,
    args: Vec<MutationArg>,
    selection: Vec<String>,
}

/// Fluent builder for a single mutation field.
impl FieldBuilder {
    fn new(name: String) -> Self {
        Self {
            name: Some(name),
            ..Default::default()
        }
    }
}

/// Fluent builder for a complete GraphQL mutation.
pub struct MutationBuilder {
    name: String,
    fields: Vec<MutationField>,
    current_field: Option<FieldBuilder>,
    variables: HashMap<String, ArgValue>,
}

impl MutationBuilder {
    /// Create a new builder with the given operation name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fields: Vec::new(),
            current_field: None,
            variables: HashMap::new(),
        }
    }

    /// Begin a new field.  Calling `field()` while another field is in progress
    /// automatically finalises the previous one.
    pub fn field(mut self, name: impl Into<String>) -> Self {
        self = self.flush_field();
        self.current_field = Some(FieldBuilder::new(name.into()));
        self
    }

    /// Set an alias for the current field.
    pub fn alias(mut self, alias: impl Into<String>) -> Self {
        if let Some(ref mut fb) = self.current_field {
            fb.alias = Some(alias.into());
        }
        self
    }

    /// Add an argument to the current field.
    pub fn arg(mut self, name: impl Into<String>, value: ArgValue) -> Self {
        if let Some(ref mut fb) = self.current_field {
            fb.args.push(MutationArg {
                name: name.into(),
                value,
                required: false,
            });
        }
        self
    }

    /// Add a *required* argument to the current field.
    pub fn required_arg(mut self, name: impl Into<String>, value: ArgValue) -> Self {
        if let Some(ref mut fb) = self.current_field {
            fb.args.push(MutationArg {
                name: name.into(),
                value,
                required: true,
            });
        }
        self
    }

    /// Add a sub-field selection to the current field.
    pub fn select(mut self, field: impl Into<String>) -> Self {
        if let Some(ref mut fb) = self.current_field {
            fb.selection.push(field.into());
        }
        self
    }

    /// Add a variable to the mutation.
    pub fn variable(mut self, name: impl Into<String>, value: ArgValue) -> Self {
        self.variables.insert(name.into(), value);
        self
    }

    /// Build the mutation, validating required arguments and duplicate fields.
    pub fn build(mut self) -> Result<Mutation, MutationError> {
        // Flush the last field in progress
        self = self.flush_field();

        // Validate field names
        for f in &self.fields {
            if f.name.is_empty() {
                return Err(MutationError::InvalidField(f.name.clone()));
            }
            // Check for required args with Null value
            for arg in &f.args {
                if arg.required && matches!(arg.value, ArgValue::Null) {
                    return Err(MutationError::MissingRequiredArg(format!(
                        "{}.{}",
                        f.name, arg.name
                    )));
                }
            }
        }

        // Check for duplicate field names (ignoring aliases)
        let mut seen: Vec<String> = Vec::new();
        for f in &self.fields {
            // Use alias if present for uniqueness, else field name
            let key = f.alias.clone().unwrap_or_else(|| f.name.clone());
            if seen.contains(&key) {
                return Err(MutationError::DuplicateField(key));
            }
            seen.push(key);
        }

        Ok(Mutation {
            name: self.name,
            fields: self.fields,
            variables: self.variables,
        })
    }

    /// Render a `Mutation` as a GraphQL mutation string.
    pub fn to_graphql_string(mutation: &Mutation) -> String {
        let mut out = format!("mutation {} {{\n", mutation.name);
        for field in &mutation.fields {
            // Alias prefix
            let name_part = if let Some(alias) = &field.alias {
                format!("{}: {}", alias, field.name)
            } else {
                field.name.clone()
            };

            // Arguments
            let args_part = if field.args.is_empty() {
                String::new()
            } else {
                let pairs: Vec<String> = field
                    .args
                    .iter()
                    .map(|a| format!("{}: {}", a.name, a.value.to_graphql()))
                    .collect();
                format!("({})", pairs.join(", "))
            };

            // Selection set
            let selection_part = if field.selection.is_empty() {
                String::new()
            } else {
                format!(" {{\n    {}\n  }}", field.selection.join("\n    "))
            };

            out.push_str(&format!("  {}{}{}{}\n", name_part, args_part, selection_part, ""));
        }
        out.push('}');
        out
    }

    // --- private helpers ---

    fn flush_field(mut self) -> Self {
        if let Some(fb) = self.current_field.take() {
            if let Some(name) = fb.name {
                self.fields.push(MutationField {
                    name,
                    alias: fb.alias,
                    args: fb.args,
                    selection: fb.selection,
                });
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ArgValue::to_graphql ---
    #[test]
    fn test_arg_value_string() {
        assert_eq!(ArgValue::String("hello".to_string()).to_graphql(), "\"hello\"");
    }

    #[test]
    fn test_arg_value_string_escapes_quotes() {
        let v = ArgValue::String("say \"hi\"".to_string());
        assert_eq!(v.to_graphql(), "\"say \\\"hi\\\"\"");
    }

    #[test]
    fn test_arg_value_int() {
        assert_eq!(ArgValue::Int(42).to_graphql(), "42");
    }

    #[test]
    fn test_arg_value_negative_int() {
        assert_eq!(ArgValue::Int(-5).to_graphql(), "-5");
    }

    #[test]
    fn test_arg_value_float() {
        assert_eq!(ArgValue::Float(3.14).to_graphql(), "3.14");
    }

    #[test]
    fn test_arg_value_bool_true() {
        assert_eq!(ArgValue::Bool(true).to_graphql(), "true");
    }

    #[test]
    fn test_arg_value_bool_false() {
        assert_eq!(ArgValue::Bool(false).to_graphql(), "false");
    }

    #[test]
    fn test_arg_value_null() {
        assert_eq!(ArgValue::Null.to_graphql(), "null");
    }

    #[test]
    fn test_arg_value_list() {
        let v = ArgValue::List(vec![ArgValue::Int(1), ArgValue::Int(2)]);
        assert_eq!(v.to_graphql(), "[1, 2]");
    }

    #[test]
    fn test_arg_value_object() {
        let mut m = HashMap::new();
        m.insert("id".to_string(), ArgValue::Int(1));
        let v = ArgValue::Object(m);
        assert!(v.to_graphql().contains("id: 1"));
    }

    // --- MutationBuilder basics ---
    #[test]
    fn test_build_empty() {
        let m = MutationBuilder::new("NoOp").build();
        assert!(m.is_ok());
        assert_eq!(m.unwrap().name, "NoOp");
    }

    #[test]
    fn test_build_single_field() {
        let m = MutationBuilder::new("M")
            .field("createUser")
            .build()
            .unwrap();
        assert_eq!(m.fields.len(), 1);
        assert_eq!(m.fields[0].name, "createUser");
    }

    #[test]
    fn test_build_multiple_fields() {
        let m = MutationBuilder::new("M")
            .field("createUser")
            .field("updateUser")
            .build()
            .unwrap();
        assert_eq!(m.fields.len(), 2);
    }

    #[test]
    fn test_build_with_alias() {
        let m = MutationBuilder::new("M")
            .field("createUser")
            .alias("newUser")
            .build()
            .unwrap();
        assert_eq!(m.fields[0].alias, Some("newUser".to_string()));
    }

    #[test]
    fn test_build_with_args() {
        let m = MutationBuilder::new("M")
            .field("createUser")
            .arg("name", ArgValue::String("Alice".to_string()))
            .arg("age", ArgValue::Int(30))
            .build()
            .unwrap();
        assert_eq!(m.fields[0].args.len(), 2);
    }

    #[test]
    fn test_build_with_selection() {
        let m = MutationBuilder::new("M")
            .field("createUser")
            .select("id")
            .select("name")
            .build()
            .unwrap();
        assert_eq!(m.fields[0].selection, vec!["id", "name"]);
    }

    // --- Duplicate field detection ---
    #[test]
    fn test_duplicate_field_error() {
        let result = MutationBuilder::new("M")
            .field("createUser")
            .field("createUser")
            .build();
        assert!(matches!(result, Err(MutationError::DuplicateField(_))));
    }

    #[test]
    fn test_duplicate_alias_error() {
        let result = MutationBuilder::new("M")
            .field("createUser")
            .alias("op")
            .field("updateUser")
            .alias("op")
            .build();
        assert!(matches!(result, Err(MutationError::DuplicateField(_))));
    }

    #[test]
    fn test_same_field_different_alias_ok() {
        let result = MutationBuilder::new("M")
            .field("createUser")
            .alias("a")
            .field("createUser")
            .alias("b")
            .build();
        assert!(result.is_ok());
    }

    // --- Required arg validation ---
    #[test]
    fn test_required_arg_null_is_error() {
        let result = MutationBuilder::new("M")
            .field("createUser")
            .required_arg("name", ArgValue::Null)
            .build();
        assert!(matches!(result, Err(MutationError::MissingRequiredArg(_))));
    }

    #[test]
    fn test_required_arg_provided_ok() {
        let result = MutationBuilder::new("M")
            .field("createUser")
            .required_arg("name", ArgValue::String("Alice".to_string()))
            .build();
        assert!(result.is_ok());
    }

    // --- to_graphql_string ---
    #[test]
    fn test_to_graphql_string_contains_mutation_name() {
        let m = MutationBuilder::new("CreateUser")
            .field("createUser")
            .build()
            .unwrap();
        let s = MutationBuilder::to_graphql_string(&m);
        assert!(s.contains("mutation CreateUser"));
    }

    #[test]
    fn test_to_graphql_string_contains_field() {
        let m = MutationBuilder::new("M")
            .field("createUser")
            .build()
            .unwrap();
        let s = MutationBuilder::to_graphql_string(&m);
        assert!(s.contains("createUser"));
    }

    #[test]
    fn test_to_graphql_string_contains_args() {
        let m = MutationBuilder::new("M")
            .field("createUser")
            .arg("name", ArgValue::String("Alice".to_string()))
            .build()
            .unwrap();
        let s = MutationBuilder::to_graphql_string(&m);
        assert!(s.contains("name:"));
        assert!(s.contains("Alice"));
    }

    #[test]
    fn test_to_graphql_string_contains_selection() {
        let m = MutationBuilder::new("M")
            .field("createUser")
            .select("id")
            .select("email")
            .build()
            .unwrap();
        let s = MutationBuilder::to_graphql_string(&m);
        assert!(s.contains("id"));
        assert!(s.contains("email"));
    }

    #[test]
    fn test_to_graphql_string_contains_alias() {
        let m = MutationBuilder::new("M")
            .field("createUser")
            .alias("newUser")
            .build()
            .unwrap();
        let s = MutationBuilder::to_graphql_string(&m);
        assert!(s.contains("newUser: createUser"));
    }

    // --- variables ---
    #[test]
    fn test_variable_stored() {
        let m = MutationBuilder::new("M")
            .variable("userId", ArgValue::Int(1))
            .build()
            .unwrap();
        assert_eq!(m.variables.get("userId"), Some(&ArgValue::Int(1)));
    }

    // --- MutationError display ---
    #[test]
    fn test_error_display_missing() {
        let e = MutationError::MissingRequiredArg("field.arg".to_string());
        let s = e.to_string();
        assert!(s.contains("field.arg"));
    }

    #[test]
    fn test_error_display_duplicate() {
        let e = MutationError::DuplicateField("createUser".to_string());
        assert!(e.to_string().contains("createUser"));
    }

    #[test]
    fn test_error_display_invalid() {
        let e = MutationError::InvalidField("".to_string());
        assert!(e.to_string().contains("Invalid field"));
    }

    // --- complex scenario ---
    #[test]
    fn test_complex_mutation() {
        let mut input_obj = HashMap::new();
        input_obj.insert("email".to_string(), ArgValue::String("a@b.com".to_string()));
        input_obj.insert("age".to_string(), ArgValue::Int(25));

        let m = MutationBuilder::new("RegisterUser")
            .field("register")
            .arg("input", ArgValue::Object(input_obj))
            .select("id")
            .select("token")
            .field("sendWelcomeEmail")
            .arg("to", ArgValue::String("a@b.com".to_string()))
            .select("success")
            .build()
            .unwrap();

        assert_eq!(m.fields.len(), 2);
        let s = MutationBuilder::to_graphql_string(&m);
        assert!(s.contains("register"));
        assert!(s.contains("sendWelcomeEmail"));
    }

    #[test]
    fn test_no_fields_is_valid() {
        let m = MutationBuilder::new("Noop").build().unwrap();
        assert_eq!(m.fields.len(), 0);
        let s = MutationBuilder::to_graphql_string(&m);
        assert!(s.starts_with("mutation Noop"));
    }
}
