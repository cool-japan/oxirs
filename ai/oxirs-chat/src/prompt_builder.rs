//! Prompt template builder with variable substitution and validation.
//!
//! Templates use `{{variable_name}}` as the substitution syntax.  Required
//! variables must be supplied at render time; optional variables use an empty
//! string if omitted.

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Error type
// ──────────────────────────────────────────────────────────────────────────────

/// Errors produced by prompt rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptError {
    /// A required variable was not supplied.
    MissingVariable(String),
    /// The requested template was not found in the builder.
    TemplateNotFound(String),
    /// An internal rendering error (e.g. malformed template syntax).
    RenderError(String),
}

impl std::fmt::Display for PromptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PromptError::MissingVariable(v) => write!(f, "missing required variable: {v}"),
            PromptError::TemplateNotFound(n) => write!(f, "template not found: {n}"),
            PromptError::RenderError(msg) => write!(f, "render error: {msg}"),
        }
    }
}

impl std::error::Error for PromptError {}

// ──────────────────────────────────────────────────────────────────────────────
// PromptTemplate
// ──────────────────────────────────────────────────────────────────────────────

/// A reusable prompt template with named variable placeholders.
///
/// Placeholders use the `{{variable_name}}` syntax.  Required variables must
/// be present in the variable map supplied to `render`; optional variables
/// default to an empty string when absent.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptTemplate {
    name: String,
    template: String,
    required_vars: Vec<String>,
    optional_vars: Vec<String>,
}

impl PromptTemplate {
    /// Create a new template.
    pub fn new(name: impl Into<String>, template: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            template: template.into(),
            required_vars: Vec::new(),
            optional_vars: Vec::new(),
        }
    }

    /// Declare a variable as required (builder pattern).
    pub fn required(mut self, var: impl Into<String>) -> Self {
        self.required_vars.push(var.into());
        self
    }

    /// Declare a variable as optional (builder pattern).
    pub fn optional(mut self, var: impl Into<String>) -> Self {
        self.optional_vars.push(var.into());
        self
    }

    /// Return the template name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the raw template string.
    pub fn raw(&self) -> &str {
        &self.template
    }

    /// Return all variable names (required and optional) mentioned in the
    /// declared lists.
    pub fn variables(&self) -> Vec<&str> {
        let mut vars: Vec<&str> = self
            .required_vars
            .iter()
            .chain(self.optional_vars.iter())
            .map(String::as_str)
            .collect();
        vars.sort_unstable();
        vars.dedup();
        vars
    }

    /// Return a list of required variables that are missing from `vars`.
    pub fn validate(&self, vars: &HashMap<String, String>) -> Vec<String> {
        self.required_vars
            .iter()
            .filter(|v| !vars.contains_key(v.as_str()))
            .cloned()
            .collect()
    }

    /// Render the template by substituting `{{key}}` placeholders.
    ///
    /// Returns `Err(PromptError::MissingVariable)` if any required variable is
    /// absent from `vars`.  Optional variables that are absent are replaced
    /// with an empty string.
    pub fn render(&self, vars: &HashMap<String, String>) -> Result<String, PromptError> {
        // Check required variables.
        let missing = self.validate(vars);
        if let Some(first) = missing.into_iter().next() {
            return Err(PromptError::MissingVariable(first));
        }

        let mut result = self.template.clone();
        // Substitute all `{{key}}` occurrences.
        // We do a single linear scan with a simple state machine.
        result = Self::substitute(&result, vars);
        Ok(result)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn substitute(template: &str, vars: &HashMap<String, String>) -> String {
        let mut output = String::with_capacity(template.len());
        let mut chars = template.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '{' && chars.peek() == Some(&'{') {
                chars.next(); // consume second '{'
                              // Collect the key until "}}"
                let mut key = String::new();
                let mut closed = false;
                while let Some(k) = chars.next() {
                    if k == '}' && chars.peek() == Some(&'}') {
                        chars.next(); // consume second '}'
                        closed = true;
                        break;
                    }
                    key.push(k);
                }
                if closed {
                    let key = key.trim().to_owned();
                    let value = vars.get(&key).map(String::as_str).unwrap_or("");
                    output.push_str(value);
                } else {
                    // Unclosed placeholder — emit as-is.
                    output.push_str("{{");
                    output.push_str(&key);
                }
            } else {
                output.push(c);
            }
        }

        output
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PromptBuilder
// ──────────────────────────────────────────────────────────────────────────────

/// A registry of named prompt templates with optional global variables.
///
/// Global variables are merged with local variables at render time; local
/// variables take precedence over globals with the same key.
#[derive(Debug, Default, Clone)]
pub struct PromptBuilder {
    templates: HashMap<String, PromptTemplate>,
    global_vars: HashMap<String, String>,
}

impl PromptBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            global_vars: HashMap::new(),
        }
    }

    /// Register a template.  If a template with the same name already exists
    /// it is replaced.
    pub fn add_template(&mut self, template: PromptTemplate) {
        self.templates.insert(template.name.clone(), template);
    }

    /// Set a global variable available to all templates.
    pub fn set_global(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.global_vars.insert(key.into(), value.into());
    }

    /// Render a template by name with the supplied local variables.
    ///
    /// Local variables shadow global variables with the same key.
    pub fn build(
        &self,
        template_name: &str,
        local_vars: HashMap<String, String>,
    ) -> Result<String, PromptError> {
        let template = self
            .templates
            .get(template_name)
            .ok_or_else(|| PromptError::TemplateNotFound(template_name.to_owned()))?;

        // Merge: global vars first, then local vars override.
        let mut merged = self.global_vars.clone();
        merged.extend(local_vars);

        template.render(&merged)
    }

    /// Return the number of registered templates.
    pub fn template_count(&self) -> usize {
        self.templates.len()
    }

    /// Return the names of all registered templates, sorted.
    pub fn list_templates(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.templates.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn vars(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    // ── PromptError ───────────────────────────────────────────────────────────

    #[test]
    fn test_prompt_error_display_missing() {
        let e = PromptError::MissingVariable("x".into());
        assert!(e.to_string().contains("x"));
    }

    #[test]
    fn test_prompt_error_display_not_found() {
        let e = PromptError::TemplateNotFound("tmpl".into());
        assert!(e.to_string().contains("tmpl"));
    }

    #[test]
    fn test_prompt_error_display_render() {
        let e = PromptError::RenderError("oops".into());
        assert!(e.to_string().contains("oops"));
    }

    #[test]
    fn test_prompt_error_equality() {
        assert_eq!(
            PromptError::MissingVariable("a".into()),
            PromptError::MissingVariable("a".into())
        );
        assert_ne!(
            PromptError::MissingVariable("a".into()),
            PromptError::MissingVariable("b".into())
        );
    }

    // ── PromptTemplate construction ───────────────────────────────────────────

    #[test]
    fn test_template_new() {
        let t = PromptTemplate::new("greet", "Hello, {{name}}!");
        assert_eq!(t.name(), "greet");
        assert_eq!(t.raw(), "Hello, {{name}}!");
    }

    #[test]
    fn test_template_required() {
        let t = PromptTemplate::new("t", "{{a}} {{b}}")
            .required("a")
            .required("b");
        assert_eq!(t.required_vars, vec!["a", "b"]);
    }

    #[test]
    fn test_template_optional() {
        let t = PromptTemplate::new("t", "{{a}}{{b}}").optional("b");
        assert_eq!(t.optional_vars, vec!["b"]);
    }

    // ── variables() ───────────────────────────────────────────────────────────

    #[test]
    fn test_variables_combined() {
        let t = PromptTemplate::new("t", "{{a}} {{b}} {{c}}")
            .required("a")
            .optional("b")
            .optional("c");
        let v = t.variables();
        assert!(v.contains(&"a"));
        assert!(v.contains(&"b"));
        assert!(v.contains(&"c"));
    }

    #[test]
    fn test_variables_deduplicated() {
        let t = PromptTemplate::new("t", "{{a}}")
            .required("a")
            .optional("a");
        let v = t.variables();
        assert_eq!(v.iter().filter(|&&x| x == "a").count(), 1);
    }

    // ── validate() ────────────────────────────────────────────────────────────

    #[test]
    fn test_validate_no_missing() {
        let t = PromptTemplate::new("t", "{{a}}").required("a");
        let missing = t.validate(&vars(&[("a", "value")]));
        assert!(missing.is_empty());
    }

    #[test]
    fn test_validate_missing_required() {
        let t = PromptTemplate::new("t", "{{a}} {{b}}")
            .required("a")
            .required("b");
        let missing = t.validate(&vars(&[("a", "hello")]));
        assert!(missing.contains(&"b".to_string()));
    }

    #[test]
    fn test_validate_optional_not_missing() {
        let t = PromptTemplate::new("t", "{{a}}").optional("a");
        // Optional vars not in `vars` should NOT appear in missing
        let missing = t.validate(&HashMap::new());
        assert!(missing.is_empty());
    }

    // ── render() ──────────────────────────────────────────────────────────────

    #[test]
    fn test_render_simple_substitution() {
        let t = PromptTemplate::new("greet", "Hello, {{name}}!").required("name");
        let result = t.render(&vars(&[("name", "World")])).unwrap();
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_render_multiple_vars() {
        let t = PromptTemplate::new("t", "{{a}} and {{b}}")
            .required("a")
            .required("b");
        let result = t.render(&vars(&[("a", "foo"), ("b", "bar")])).unwrap();
        assert_eq!(result, "foo and bar");
    }

    #[test]
    fn test_render_repeated_var() {
        let t = PromptTemplate::new("t", "{{x}} {{x}} {{x}}").required("x");
        let result = t.render(&vars(&[("x", "go")])).unwrap();
        assert_eq!(result, "go go go");
    }

    #[test]
    fn test_render_optional_missing_is_empty_string() {
        let t = PromptTemplate::new("t", "start {{opt}} end").optional("opt");
        let result = t.render(&HashMap::new()).unwrap();
        assert_eq!(result, "start  end");
    }

    #[test]
    fn test_render_missing_required_returns_error() {
        let t = PromptTemplate::new("t", "{{req}}").required("req");
        let err = t.render(&HashMap::new()).unwrap_err();
        assert!(matches!(err, PromptError::MissingVariable(_)));
    }

    #[test]
    fn test_render_no_placeholders() {
        let t = PromptTemplate::new("t", "Hello, World!");
        let result = t.render(&HashMap::new()).unwrap();
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_render_whitespace_in_placeholder() {
        let t = PromptTemplate::new("t", "{{ name }}").optional("name");
        let result = t.render(&vars(&[("name", "Alice")])).unwrap();
        assert_eq!(result, "Alice");
    }

    #[test]
    fn test_render_empty_template() {
        let t = PromptTemplate::new("t", "");
        let result = t.render(&HashMap::new()).unwrap();
        assert_eq!(result, "");
    }

    // ── PromptBuilder ─────────────────────────────────────────────────────────

    #[test]
    fn test_builder_new_empty() {
        let b = PromptBuilder::new();
        assert_eq!(b.template_count(), 0);
        assert!(b.list_templates().is_empty());
    }

    #[test]
    fn test_builder_add_template() {
        let mut b = PromptBuilder::new();
        b.add_template(PromptTemplate::new("t1", "hello"));
        assert_eq!(b.template_count(), 1);
    }

    #[test]
    fn test_builder_list_templates_sorted() {
        let mut b = PromptBuilder::new();
        b.add_template(PromptTemplate::new("c", "c"));
        b.add_template(PromptTemplate::new("a", "a"));
        b.add_template(PromptTemplate::new("b", "b"));
        assert_eq!(b.list_templates(), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_builder_build_basic() {
        let mut b = PromptBuilder::new();
        b.add_template(PromptTemplate::new("hi", "Hi {{name}}!").required("name"));
        let result = b.build("hi", vars(&[("name", "Alice")])).unwrap();
        assert_eq!(result, "Hi Alice!");
    }

    #[test]
    fn test_builder_build_not_found() {
        let b = PromptBuilder::new();
        let err = b.build("missing", HashMap::new()).unwrap_err();
        assert!(matches!(err, PromptError::TemplateNotFound(_)));
    }

    #[test]
    fn test_builder_global_vars() {
        let mut b = PromptBuilder::new();
        b.set_global("lang", "Rust");
        b.add_template(PromptTemplate::new("prog", "I love {{lang}}!").optional("lang"));
        let result = b.build("prog", HashMap::new()).unwrap();
        assert_eq!(result, "I love Rust!");
    }

    #[test]
    fn test_builder_local_overrides_global() {
        let mut b = PromptBuilder::new();
        b.set_global("lang", "Rust");
        b.add_template(PromptTemplate::new("prog", "Language: {{lang}}").optional("lang"));
        let result = b.build("prog", vars(&[("lang", "Python")])).unwrap();
        assert_eq!(result, "Language: Python");
    }

    #[test]
    fn test_builder_replace_template() {
        let mut b = PromptBuilder::new();
        b.add_template(PromptTemplate::new("t", "version 1"));
        b.add_template(PromptTemplate::new("t", "version 2"));
        assert_eq!(b.template_count(), 1);
        let result = b.build("t", HashMap::new()).unwrap();
        assert_eq!(result, "version 2");
    }

    #[test]
    fn test_builder_multiple_templates() {
        let mut b = PromptBuilder::new();
        b.add_template(PromptTemplate::new("a", "{{x}}").required("x"));
        b.add_template(PromptTemplate::new("b", "{{y}}").required("y"));

        assert_eq!(b.build("a", vars(&[("x", "1")])).unwrap(), "1");
        assert_eq!(b.build("b", vars(&[("y", "2")])).unwrap(), "2");
    }

    #[test]
    fn test_builder_global_plus_local_mix() {
        let mut b = PromptBuilder::new();
        b.set_global("system", "OxiRS");
        b.add_template(
            PromptTemplate::new("intro", "{{system}} welcomes {{user}}")
                .optional("system")
                .required("user"),
        );
        let result = b.build("intro", vars(&[("user", "Bob")])).unwrap();
        assert_eq!(result, "OxiRS welcomes Bob");
    }

    #[test]
    fn test_builder_missing_required_error() {
        let mut b = PromptBuilder::new();
        b.add_template(PromptTemplate::new("t", "{{req}}").required("req"));
        let err = b.build("t", HashMap::new()).unwrap_err();
        assert!(matches!(err, PromptError::MissingVariable(_)));
    }

    #[test]
    fn test_builder_build_multiline_template() {
        let tmpl = "Line 1: {{a}}\nLine 2: {{b}}\nLine 3: {{a}}";
        let mut b = PromptBuilder::new();
        b.add_template(
            PromptTemplate::new("multi", tmpl)
                .required("a")
                .required("b"),
        );
        let result = b
            .build("multi", vars(&[("a", "hello"), ("b", "world")]))
            .unwrap();
        assert_eq!(result, "Line 1: hello\nLine 2: world\nLine 3: hello");
    }

    #[test]
    fn test_template_clone() {
        let t = PromptTemplate::new("t", "{{x}}").required("x");
        let t2 = t.clone();
        assert_eq!(t, t2);
    }

    #[test]
    fn test_builder_default() {
        let b = PromptBuilder::default();
        assert_eq!(b.template_count(), 0);
    }

    #[test]
    fn test_builder_set_multiple_globals() {
        let mut b = PromptBuilder::new();
        b.set_global("a", "1");
        b.set_global("b", "2");
        b.set_global("a", "3"); // override
        b.add_template(
            PromptTemplate::new("t", "{{a}} {{b}}")
                .optional("a")
                .optional("b"),
        );
        let result = b.build("t", HashMap::new()).unwrap();
        assert_eq!(result, "3 2");
    }
}
