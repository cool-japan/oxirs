//! GraphQL directive registry.
//!
//! Provides a central registry for GraphQL directive definitions, including
//! built-in directives (`@skip`, `@include`, `@deprecated`, `@specifiedBy`)
//! and support for custom application directives.

use std::collections::HashMap;

/// The location contexts where a directive may be applied.
#[derive(Debug, Clone, PartialEq)]
pub enum DirectiveLocation {
    Field,
    FragmentDefinition,
    FragmentSpread,
    InlineFragment,
    Query,
    Mutation,
    Subscription,
    Schema,
    Scalar,
    Object,
    FieldDefinition,
    ArgumentDefinition,
    Interface,
    Union,
    Enum,
    EnumValue,
    InputObject,
    InputFieldDefinition,
}

impl DirectiveLocation {
    /// Return the canonical SDL name for this location.
    pub fn sdl_name(&self) -> &'static str {
        match self {
            DirectiveLocation::Field => "FIELD",
            DirectiveLocation::FragmentDefinition => "FRAGMENT_DEFINITION",
            DirectiveLocation::FragmentSpread => "FRAGMENT_SPREAD",
            DirectiveLocation::InlineFragment => "INLINE_FRAGMENT",
            DirectiveLocation::Query => "QUERY",
            DirectiveLocation::Mutation => "MUTATION",
            DirectiveLocation::Subscription => "SUBSCRIPTION",
            DirectiveLocation::Schema => "SCHEMA",
            DirectiveLocation::Scalar => "SCALAR",
            DirectiveLocation::Object => "OBJECT",
            DirectiveLocation::FieldDefinition => "FIELD_DEFINITION",
            DirectiveLocation::ArgumentDefinition => "ARGUMENT_DEFINITION",
            DirectiveLocation::Interface => "INTERFACE",
            DirectiveLocation::Union => "UNION",
            DirectiveLocation::Enum => "ENUM",
            DirectiveLocation::EnumValue => "ENUM_VALUE",
            DirectiveLocation::InputObject => "INPUT_OBJECT",
            DirectiveLocation::InputFieldDefinition => "INPUT_FIELD_DEFINITION",
        }
    }
}

/// A single argument in a directive definition.
#[derive(Debug, Clone)]
pub struct DirectiveArg {
    pub name: String,
    pub ty: String,
    pub default_value: Option<String>,
}

impl DirectiveArg {
    pub fn new(name: impl Into<String>, ty: impl Into<String>) -> Self {
        DirectiveArg {
            name: name.into(),
            ty: ty.into(),
            default_value: None,
        }
    }

    pub fn with_default(mut self, default: impl Into<String>) -> Self {
        self.default_value = Some(default.into());
        self
    }
}

/// A complete definition of a GraphQL directive.
#[derive(Debug, Clone)]
pub struct DirectiveDef {
    pub name: String,
    pub description: Option<String>,
    pub locations: Vec<DirectiveLocation>,
    pub args: Vec<DirectiveArg>,
    pub is_repeatable: bool,
}

impl DirectiveDef {
    pub fn new(name: impl Into<String>) -> Self {
        DirectiveDef {
            name: name.into(),
            description: None,
            locations: Vec::new(),
            args: Vec::new(),
            is_repeatable: false,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn with_location(mut self, loc: DirectiveLocation) -> Self {
        self.locations.push(loc);
        self
    }

    pub fn with_arg(mut self, arg: DirectiveArg) -> Self {
        self.args.push(arg);
        self
    }

    pub fn repeatable(mut self) -> Self {
        self.is_repeatable = true;
        self
    }
}

/// Errors from the directive registry.
#[derive(Debug, PartialEq)]
pub enum RegistryError {
    AlreadyRegistered(String),
    NotFound(String),
    InvalidLocation { directive: String, location: String },
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::AlreadyRegistered(n) => write!(f, "Directive already registered: @{n}"),
            RegistryError::NotFound(n) => write!(f, "Directive not found: @{n}"),
            RegistryError::InvalidLocation {
                directive,
                location,
            } => {
                write!(
                    f,
                    "Directive @{directive} is not valid at location {location}"
                )
            }
        }
    }
}

impl std::error::Error for RegistryError {}

/// A registry of GraphQL directive definitions.
pub struct DirectiveRegistry {
    directives: HashMap<String, DirectiveDef>,
}

impl DirectiveRegistry {
    /// Create a new registry pre-populated with GraphQL built-in directives.
    pub fn new() -> Self {
        let mut registry = DirectiveRegistry {
            directives: HashMap::new(),
        };
        registry.register_builtins();
        registry
    }

    fn register_builtins(&mut self) {
        // @skip(if: Boolean!)
        let skip = DirectiveDef::new("skip")
            .with_description("Directs the executor to skip this field or fragment when the `if` argument is true.")
            .with_location(DirectiveLocation::Field)
            .with_location(DirectiveLocation::FragmentSpread)
            .with_location(DirectiveLocation::InlineFragment)
            .with_arg(DirectiveArg::new("if", "Boolean!"));
        self.directives.insert("skip".to_string(), skip);

        // @include(if: Boolean!)
        let include = DirectiveDef::new("include")
            .with_description("Directs the executor to include this field or fragment only when the `if` argument is true.")
            .with_location(DirectiveLocation::Field)
            .with_location(DirectiveLocation::FragmentSpread)
            .with_location(DirectiveLocation::InlineFragment)
            .with_arg(DirectiveArg::new("if", "Boolean!"));
        self.directives.insert("include".to_string(), include);

        // @deprecated(reason: String)
        let deprecated = DirectiveDef::new("deprecated")
            .with_description("Marks the field, argument, input field or enum value as deprecated.")
            .with_location(DirectiveLocation::FieldDefinition)
            .with_location(DirectiveLocation::ArgumentDefinition)
            .with_location(DirectiveLocation::InputFieldDefinition)
            .with_location(DirectiveLocation::EnumValue)
            .with_arg(DirectiveArg::new("reason", "String").with_default("No longer supported"));
        self.directives.insert("deprecated".to_string(), deprecated);

        // @specifiedBy(url: String!)
        let specified_by = DirectiveDef::new("specifiedBy")
            .with_description("Exposes a URL that specifies the behaviour of this scalar.")
            .with_location(DirectiveLocation::Scalar)
            .with_arg(DirectiveArg::new("url", "String!"));
        self.directives
            .insert("specifiedBy".to_string(), specified_by);
    }

    /// Register a custom directive. Returns error if already registered.
    pub fn register(&mut self, def: DirectiveDef) -> Result<(), RegistryError> {
        if self.directives.contains_key(&def.name) {
            return Err(RegistryError::AlreadyRegistered(def.name.clone()));
        }
        self.directives.insert(def.name.clone(), def);
        Ok(())
    }

    /// Look up a directive by name.
    pub fn get(&self, name: &str) -> Option<&DirectiveDef> {
        self.directives.get(name)
    }

    /// Return all directives sorted by name.
    pub fn all(&self) -> Vec<&DirectiveDef> {
        let mut v: Vec<&DirectiveDef> = self.directives.values().collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }

    /// Check whether a named directive may appear at the given location.
    pub fn is_valid_location(&self, name: &str, loc: &DirectiveLocation) -> bool {
        self.directives
            .get(name)
            .map(|d| d.locations.contains(loc))
            .unwrap_or(false)
    }

    /// Validate that a directive is known and valid at the specified location,
    /// returning an error if not.
    pub fn validate_usage(&self, name: &str, loc: &DirectiveLocation) -> Result<(), RegistryError> {
        let def = self
            .directives
            .get(name)
            .ok_or_else(|| RegistryError::NotFound(name.to_string()))?;
        if def.locations.contains(loc) {
            Ok(())
        } else {
            Err(RegistryError::InvalidLocation {
                directive: name.to_string(),
                location: loc.sdl_name().to_string(),
            })
        }
    }

    /// Render the registry (excluding built-ins) as SDL text.
    /// Built-ins are skipped because they are implicit in every schema.
    pub fn to_sdl(&self) -> String {
        let builtins = ["skip", "include", "deprecated", "specifiedBy"];
        let mut lines: Vec<String> = self
            .directives
            .values()
            .filter(|d| !builtins.contains(&d.name.as_str()))
            .map(render_directive_sdl)
            .collect();
        lines.sort();
        lines.join("\n")
    }

    /// Render the complete registry including built-ins.
    pub fn to_sdl_full(&self) -> String {
        let mut lines: Vec<String> = self.directives.values().map(render_directive_sdl).collect();
        lines.sort();
        lines.join("\n")
    }

    /// Count of registered directives.
    pub fn len(&self) -> usize {
        self.directives.len()
    }

    /// Whether the registry is empty (no directives at all).
    pub fn is_empty(&self) -> bool {
        self.directives.is_empty()
    }
}

impl Default for DirectiveRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn render_directive_sdl(d: &DirectiveDef) -> String {
    let mut s = String::new();
    if let Some(ref desc) = d.description {
        s.push_str(&format!("\"\"\" {desc} \"\"\"\n"));
    }
    s.push_str("directive @");
    s.push_str(&d.name);
    if !d.args.is_empty() {
        s.push('(');
        let args: Vec<String> = d
            .args
            .iter()
            .map(|a| {
                if let Some(ref default) = a.default_value {
                    format!("{}: {} = {}", a.name, a.ty, default)
                } else {
                    format!("{}: {}", a.name, a.ty)
                }
            })
            .collect();
        s.push_str(&args.join(", "));
        s.push(')');
    }
    if d.is_repeatable {
        s.push_str(" repeatable");
    }
    if !d.locations.is_empty() {
        s.push_str(" on ");
        let locs: Vec<&str> = d.locations.iter().map(|l| l.sdl_name()).collect();
        s.push_str(&locs.join(" | "));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_has_builtins() {
        let reg = DirectiveRegistry::new();
        assert!(reg.get("skip").is_some());
        assert!(reg.get("include").is_some());
        assert!(reg.get("deprecated").is_some());
        assert!(reg.get("specifiedBy").is_some());
    }

    #[test]
    fn test_new_has_four_builtins() {
        let reg = DirectiveRegistry::new();
        assert_eq!(reg.len(), 4);
    }

    #[test]
    fn test_skip_locations() {
        let reg = DirectiveRegistry::new();
        let skip = reg.get("skip").expect("should succeed");
        assert!(skip.locations.contains(&DirectiveLocation::Field));
        assert!(skip.locations.contains(&DirectiveLocation::FragmentSpread));
        assert!(skip.locations.contains(&DirectiveLocation::InlineFragment));
    }

    #[test]
    fn test_include_has_if_arg() {
        let reg = DirectiveRegistry::new();
        let include = reg.get("include").expect("should succeed");
        assert_eq!(include.args.len(), 1);
        assert_eq!(include.args[0].name, "if");
        assert_eq!(include.args[0].ty, "Boolean!");
    }

    #[test]
    fn test_deprecated_has_reason_with_default() {
        let reg = DirectiveRegistry::new();
        let dep = reg.get("deprecated").expect("should succeed");
        assert_eq!(dep.args[0].name, "reason");
        assert!(dep.args[0].default_value.is_some());
    }

    #[test]
    fn test_specified_by_url_arg() {
        let reg = DirectiveRegistry::new();
        let sb = reg.get("specifiedBy").expect("should succeed");
        assert_eq!(sb.args[0].name, "url");
        assert_eq!(sb.args[0].ty, "String!");
    }

    #[test]
    fn test_register_custom() {
        let mut reg = DirectiveRegistry::new();
        let d = DirectiveDef::new("auth")
            .with_location(DirectiveLocation::FieldDefinition)
            .with_arg(DirectiveArg::new("role", "String!"));
        reg.register(d).expect("should succeed");
        assert!(reg.get("auth").is_some());
    }

    #[test]
    fn test_register_duplicate_error() {
        let mut reg = DirectiveRegistry::new();
        let d1 = DirectiveDef::new("auth").with_location(DirectiveLocation::Field);
        let d2 = DirectiveDef::new("auth").with_location(DirectiveLocation::Object);
        reg.register(d1).expect("should succeed");
        let err = reg.register(d2).unwrap_err();
        assert_eq!(err, RegistryError::AlreadyRegistered("auth".to_string()));
    }

    #[test]
    fn test_get_unknown_returns_none() {
        let reg = DirectiveRegistry::new();
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn test_all_sorted_by_name() {
        let reg = DirectiveRegistry::new();
        let all = reg.all();
        let names: Vec<&str> = all.iter().map(|d| d.name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    #[test]
    fn test_is_valid_location_true() {
        let reg = DirectiveRegistry::new();
        assert!(reg.is_valid_location("skip", &DirectiveLocation::Field));
    }

    #[test]
    fn test_is_valid_location_false() {
        let reg = DirectiveRegistry::new();
        assert!(!reg.is_valid_location("skip", &DirectiveLocation::Object));
    }

    #[test]
    fn test_is_valid_location_unknown_directive() {
        let reg = DirectiveRegistry::new();
        assert!(!reg.is_valid_location("unknown", &DirectiveLocation::Field));
    }

    #[test]
    fn test_validate_usage_ok() {
        let reg = DirectiveRegistry::new();
        assert!(reg
            .validate_usage("include", &DirectiveLocation::Field)
            .is_ok());
    }

    #[test]
    fn test_validate_usage_wrong_location() {
        let reg = DirectiveRegistry::new();
        let err = reg
            .validate_usage("include", &DirectiveLocation::Schema)
            .unwrap_err();
        assert!(matches!(err, RegistryError::InvalidLocation { .. }));
    }

    #[test]
    fn test_validate_usage_not_found() {
        let reg = DirectiveRegistry::new();
        let err = reg
            .validate_usage("ghost", &DirectiveLocation::Field)
            .unwrap_err();
        assert_eq!(err, RegistryError::NotFound("ghost".to_string()));
    }

    #[test]
    fn test_to_sdl_excludes_builtins() {
        let mut reg = DirectiveRegistry::new();
        let d = DirectiveDef::new("auth").with_location(DirectiveLocation::FieldDefinition);
        reg.register(d).expect("should succeed");
        let sdl = reg.to_sdl();
        assert!(sdl.contains("@auth"));
        assert!(!sdl.contains("@skip"));
    }

    #[test]
    fn test_to_sdl_full_includes_builtins() {
        let reg = DirectiveRegistry::new();
        let sdl = reg.to_sdl_full();
        assert!(sdl.contains("@skip"));
        assert!(sdl.contains("@include"));
    }

    #[test]
    fn test_sdl_with_args() {
        let mut reg = DirectiveRegistry::new();
        let d = DirectiveDef::new("rateLimit")
            .with_location(DirectiveLocation::FieldDefinition)
            .with_arg(DirectiveArg::new("max", "Int!"))
            .with_arg(DirectiveArg::new("window", "String!"));
        reg.register(d).expect("should succeed");
        let sdl = reg.to_sdl();
        assert!(sdl.contains("max: Int!"));
        assert!(sdl.contains("window: String!"));
    }

    #[test]
    fn test_sdl_repeatable() {
        let mut reg = DirectiveRegistry::new();
        let d = DirectiveDef::new("tag")
            .with_location(DirectiveLocation::Field)
            .repeatable();
        reg.register(d).expect("should succeed");
        let sdl = reg.to_sdl();
        assert!(sdl.contains("repeatable"));
    }

    #[test]
    fn test_sdl_default_value() {
        let mut reg = DirectiveRegistry::new();
        let d = DirectiveDef::new("cache")
            .with_location(DirectiveLocation::FieldDefinition)
            .with_arg(DirectiveArg::new("maxAge", "Int").with_default("60"));
        reg.register(d).expect("should succeed");
        let sdl = reg.to_sdl();
        assert!(sdl.contains("= 60"));
    }

    #[test]
    fn test_sdl_location_names() {
        assert_eq!(DirectiveLocation::Field.sdl_name(), "FIELD");
        assert_eq!(DirectiveLocation::Query.sdl_name(), "QUERY");
        assert_eq!(
            DirectiveLocation::FieldDefinition.sdl_name(),
            "FIELD_DEFINITION"
        );
        assert_eq!(DirectiveLocation::InputObject.sdl_name(), "INPUT_OBJECT");
    }

    #[test]
    fn test_error_display_already_registered() {
        let e = RegistryError::AlreadyRegistered("auth".to_string());
        assert!(e.to_string().contains("already registered"));
    }

    #[test]
    fn test_error_display_not_found() {
        let e = RegistryError::NotFound("ghost".to_string());
        assert!(e.to_string().contains("not found"));
    }

    #[test]
    fn test_error_display_invalid_location() {
        let e = RegistryError::InvalidLocation {
            directive: "skip".to_string(),
            location: "SCHEMA".to_string(),
        };
        assert!(e.to_string().contains("not valid"));
    }

    #[test]
    fn test_is_empty_after_creation() {
        let reg = DirectiveRegistry::new();
        assert!(!reg.is_empty()); // has builtins
    }

    #[test]
    fn test_default_trait() {
        let reg = DirectiveRegistry::default();
        assert_eq!(reg.len(), 4);
    }

    #[test]
    fn test_deprecated_scalar_location() {
        let reg = DirectiveRegistry::new();
        // @deprecated should NOT be valid on Scalar
        assert!(!reg.is_valid_location("deprecated", &DirectiveLocation::Scalar));
    }

    #[test]
    fn test_specified_by_scalar_location() {
        let reg = DirectiveRegistry::new();
        assert!(reg.is_valid_location("specifiedBy", &DirectiveLocation::Scalar));
    }

    #[test]
    fn test_directive_def_builder() {
        let d = DirectiveDef::new("myDir")
            .with_description("A custom directive")
            .with_location(DirectiveLocation::Object)
            .with_arg(DirectiveArg::new("x", "String"))
            .repeatable();
        assert_eq!(d.name, "myDir");
        assert!(d.description.is_some());
        assert_eq!(d.locations.len(), 1);
        assert_eq!(d.args.len(), 1);
        assert!(d.is_repeatable);
    }

    #[test]
    fn test_all_locations_coverage() {
        // Ensure all locations have a non-empty SDL name
        let locs = [
            DirectiveLocation::Field,
            DirectiveLocation::FragmentDefinition,
            DirectiveLocation::FragmentSpread,
            DirectiveLocation::InlineFragment,
            DirectiveLocation::Query,
            DirectiveLocation::Mutation,
            DirectiveLocation::Subscription,
            DirectiveLocation::Schema,
            DirectiveLocation::Scalar,
            DirectiveLocation::Object,
            DirectiveLocation::FieldDefinition,
            DirectiveLocation::ArgumentDefinition,
            DirectiveLocation::Interface,
            DirectiveLocation::Union,
            DirectiveLocation::Enum,
            DirectiveLocation::EnumValue,
            DirectiveLocation::InputObject,
            DirectiveLocation::InputFieldDefinition,
        ];
        for loc in &locs {
            assert!(!loc.sdl_name().is_empty());
        }
    }
}
