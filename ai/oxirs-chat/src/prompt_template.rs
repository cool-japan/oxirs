//! Prompt template engine with variable substitution and conditional blocks.
//!
//! Supports `{{var_name}}` placeholder replacement, `{{#if var}}…{{/if}}`
//! conditional blocks, built-in templates, and template chaining.

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// A declared variable in a prompt template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateVar {
    /// Variable name used in `{{name}}` placeholders.
    pub name: String,
    /// Human-readable description for tooling and error messages.
    pub description: String,
    /// If `true`, rendering fails when this variable is absent.
    pub required: bool,
    /// Default value used when the variable is absent and not required.
    pub default: Option<String>,
}

impl TemplateVar {
    /// Create a required variable with no default.
    pub fn required(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required: true,
            default: None,
        }
    }

    /// Create an optional variable with a default.
    pub fn optional(
        name: impl Into<String>,
        description: impl Into<String>,
        default: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required: false,
            default: Some(default.into()),
        }
    }
}

/// A prompt template.
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    /// Unique template identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Template text. Use `{{var}}` for variables and `{{#if var}}…{{/if}}` for conditionals.
    pub template: String,
    /// Declared variables.
    pub vars: Vec<TemplateVar>,
    /// Optional system prompt prepended before the user prompt.
    pub system_prompt: Option<String>,
}

impl PromptTemplate {
    /// Create a template with the given ID, name, and body.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        template: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            template: template.into(),
            vars: Vec::new(),
            system_prompt: None,
        }
    }

    /// Builder: add a variable declaration.
    pub fn with_var(mut self, var: TemplateVar) -> Self {
        self.vars.push(var);
        self
    }

    /// Builder: set the system prompt.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system_prompt = Some(system.into());
        self
    }
}

/// Errors from the template engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateError {
    /// No template with this ID is registered.
    NotFound(String),
    /// A required variable was not supplied.
    MissingRequired(String),
    /// A rendering error occurred (e.g. malformed template).
    RenderError(String),
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemplateError::NotFound(id) => write!(f, "template not found: {id}"),
            TemplateError::MissingRequired(var) => write!(f, "missing required variable: {var}"),
            TemplateError::RenderError(msg) => write!(f, "render error: {msg}"),
        }
    }
}

impl std::error::Error for TemplateError {}

// ──────────────────────────────────────────────────────────────────────────────
// TemplateEngine
// ──────────────────────────────────────────────────────────────────────────────

/// Registry and renderer for prompt templates.
pub struct TemplateEngine {
    templates: HashMap<String, PromptTemplate>,
}

impl TemplateEngine {
    /// Create an empty engine with no templates registered.
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    /// Create an engine pre-loaded with built-in templates.
    pub fn with_builtins() -> Self {
        let mut engine = Self::new();
        engine.register_builtins();
        engine
    }

    /// Register a template. Overwrites any existing template with the same ID.
    pub fn register(&mut self, template: PromptTemplate) {
        self.templates.insert(template.id.clone(), template);
    }

    /// Retrieve a registered template by ID.
    pub fn get(&self, id: &str) -> Option<&PromptTemplate> {
        self.templates.get(id)
    }

    /// List all registered template IDs.
    pub fn list_ids(&self) -> Vec<&str> {
        self.templates.keys().map(String::as_str).collect()
    }

    /// Render a template with the given variables.
    ///
    /// Returns the rendered string (system prompt separated by `\n---\n` when
    /// a system prompt is defined).
    pub fn render(
        &self,
        template_id: &str,
        vars: &HashMap<String, String>,
    ) -> Result<String, TemplateError> {
        let template = self
            .templates
            .get(template_id)
            .ok_or_else(|| TemplateError::NotFound(template_id.to_string()))?;

        // Check required variables
        for var_def in &template.vars {
            if var_def.required && !vars.contains_key(&var_def.name) {
                return Err(TemplateError::MissingRequired(var_def.name.clone()));
            }
        }

        // Build effective variable map (apply defaults for missing optional vars)
        let mut effective: HashMap<&str, String> = HashMap::new();
        for var_def in &template.vars {
            if let Some(val) = vars.get(&var_def.name) {
                effective.insert(var_def.name.as_str(), val.clone());
            } else if let Some(ref default) = var_def.default {
                effective.insert(var_def.name.as_str(), default.clone());
            }
        }
        // Also include any extra vars not declared (pass-through)
        for (k, v) in vars {
            effective.entry(k.as_str()).or_insert_with(|| v.clone());
        }

        let body = render_template(&template.template, &effective)?;

        Ok(match &template.system_prompt {
            Some(sys) => {
                let sys_rendered = render_template(sys, &effective)?;
                format!("{sys_rendered}\n---\n{body}")
            }
            None => body,
        })
    }

    /// Validate a template call: returns the list of missing required variable names.
    pub fn validate(&self, template_id: &str, vars: &HashMap<String, String>) -> Vec<String> {
        let Some(template) = self.templates.get(template_id) else {
            return vec![format!("template not found: {template_id}")];
        };
        template
            .vars
            .iter()
            .filter(|v| v.required && !vars.contains_key(&v.name))
            .map(|v| v.name.clone())
            .collect()
    }

    /// Chain templates: each template's rendered output is injected into the
    /// next template as the variable `"input"`.
    pub fn chain(
        &self,
        template_ids: &[&str],
        vars: &HashMap<String, String>,
    ) -> Result<String, TemplateError> {
        if template_ids.is_empty() {
            return Err(TemplateError::RenderError(
                "no templates in chain".to_string(),
            ));
        }
        let mut current_vars = vars.clone();
        let mut output = String::new();

        for &id in template_ids {
            output = self.render(id, &current_vars)?;
            current_vars.insert("input".to_string(), output.clone());
        }
        Ok(output)
    }

    // ── Built-in templates ────────────────────────────────────────────────────

    fn register_builtins(&mut self) {
        // sparql_query
        self.register(
            PromptTemplate::new(
                "sparql_query",
                "SPARQL Query Generator",
                "Generate a SPARQL query to answer: {{question}}\n\
                 Graph: {{#if graph}}<{{graph}}>{{/if}}\n\
                 Query:",
            )
            .with_var(TemplateVar::required(
                "question",
                "Natural language question",
            ))
            .with_var(TemplateVar::optional("graph", "Named graph IRI", ""))
            .with_system("You are a SPARQL expert. Output only valid SPARQL."),
        );

        // rag_answer
        self.register(
            PromptTemplate::new(
                "rag_answer",
                "RAG Answer Generator",
                "Context:\n{{context}}\n\nQuestion: {{question}}\nAnswer:",
            )
            .with_var(TemplateVar::required(
                "context",
                "Retrieved context passages",
            ))
            .with_var(TemplateVar::required("question", "User question"))
            .with_system("Answer based only on the provided context."),
        );

        // summarize
        self.register(
            PromptTemplate::new(
                "summarize",
                "Text Summarizer",
                "Summarize the following text{{#if style}} in a {{style}} style{{/if}}:\n\n{{input}}",
            )
            .with_var(TemplateVar::required("input", "Text to summarize"))
            .with_var(TemplateVar::optional("style", "Summary style", "")),
        );

        // classify
        self.register(
            PromptTemplate::new(
                "classify",
                "Text Classifier",
                "Classify the following text into one of [{{categories}}]:\n\n{{input}}\n\nCategory:",
            )
            .with_var(TemplateVar::required("input", "Text to classify"))
            .with_var(TemplateVar::required("categories", "Comma-separated category list")),
        );
    }
}

impl Default for TemplateEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Core rendering
// ──────────────────────────────────────────────────────────────────────────────

/// Render a template string by substituting `{{var}}` placeholders and
/// evaluating `{{#if var}}…{{/if}}` conditional blocks.
fn render_template(template: &str, vars: &HashMap<&str, String>) -> Result<String, TemplateError> {
    // First pass: process {{#if var}}…{{/if}} blocks
    let after_conditionals = process_conditionals(template, vars)?;
    // Second pass: replace {{var}} placeholders
    substitute_vars(&after_conditionals, vars)
}

/// Process all `{{#if var}}…{{/if}}` blocks.
fn process_conditionals(
    template: &str,
    vars: &HashMap<&str, String>,
) -> Result<String, TemplateError> {
    let mut result = template.to_string();

    while let Some(if_start) = result.find("{{#if ") {
        // Extract variable name
        let rest = &result[if_start + 6..];
        let var_end = rest
            .find("}}")
            .ok_or_else(|| TemplateError::RenderError("unclosed {{#if}}".to_string()))?;
        let var_name = rest[..var_end].trim();

        // Find the matching {{/if}}
        let block_start = if_start + 6 + var_end + 2;
        let block = &result[block_start..];
        let end_pos = block
            .find("{{/if}}")
            .ok_or_else(|| TemplateError::RenderError("missing {{/if}}".to_string()))?;

        let inner = &block[..end_pos];
        let after = &block[end_pos + 7..]; // skip {{/if}}

        // Evaluate condition: true when var is set and non-empty
        let condition = vars.get(var_name).map(|v| !v.is_empty()).unwrap_or(false);

        let replacement = if condition { inner } else { "" };

        result = format!("{}{}{}", &result[..if_start], replacement, after);
    }
    Ok(result)
}

/// Replace `{{var_name}}` placeholders with their values.
fn substitute_vars(template: &str, vars: &HashMap<&str, String>) -> Result<String, TemplateError> {
    let mut result = String::with_capacity(template.len());
    let mut remaining = template;

    while let Some(start) = remaining.find("{{") {
        result.push_str(&remaining[..start]);
        remaining = &remaining[start + 2..];

        let end = remaining
            .find("}}")
            .ok_or_else(|| TemplateError::RenderError("unclosed '{{'".to_string()))?;

        let var_name = remaining[..end].trim();
        remaining = &remaining[end + 2..];

        // Look up and substitute
        if let Some(val) = vars.get(var_name) {
            result.push_str(val);
        }
        // Silently drop unknown/missing optional variables
    }
    result.push_str(remaining);
    Ok(result)
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

    // ── TemplateVar ───────────────────────────────────────────────────────────

    #[test]
    fn test_template_var_required() {
        let v = TemplateVar::required("name", "The name");
        assert!(v.required);
        assert!(v.default.is_none());
        assert_eq!(v.name, "name");
    }

    #[test]
    fn test_template_var_optional() {
        let v = TemplateVar::optional("style", "Output style", "formal");
        assert!(!v.required);
        assert_eq!(v.default, Some("formal".to_string()));
    }

    // ── TemplateError ─────────────────────────────────────────────────────────

    #[test]
    fn test_error_display_not_found() {
        let e = TemplateError::NotFound("my_tpl".to_string());
        assert!(format!("{e}").contains("my_tpl"));
    }

    #[test]
    fn test_error_display_missing_required() {
        let e = TemplateError::MissingRequired("context".to_string());
        assert!(format!("{e}").contains("context"));
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(TemplateError::NotFound("x".into()));
        assert!(!e.to_string().is_empty());
    }

    // ── TemplateEngine::register / get / list ─────────────────────────────────

    #[test]
    fn test_engine_register_and_get() {
        let mut engine = TemplateEngine::new();
        engine.register(PromptTemplate::new("t1", "T1", "hello {{name}}"));
        assert!(engine.get("t1").is_some());
    }

    #[test]
    fn test_engine_list_ids() {
        let mut engine = TemplateEngine::new();
        engine.register(PromptTemplate::new("a", "A", ""));
        engine.register(PromptTemplate::new("b", "B", ""));
        let ids = engine.list_ids();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }

    #[test]
    fn test_engine_get_not_registered() {
        let engine = TemplateEngine::new();
        assert!(engine.get("missing").is_none());
    }

    #[test]
    fn test_engine_overwrite() {
        let mut engine = TemplateEngine::new();
        engine.register(PromptTemplate::new("t", "Old", "old"));
        engine.register(PromptTemplate::new("t", "New", "new"));
        assert_eq!(engine.get("t").expect("should succeed").name, "New");
    }

    // ── render ────────────────────────────────────────────────────────────────

    #[test]
    fn test_render_simple_substitution() {
        let mut engine = TemplateEngine::new();
        engine.register(PromptTemplate::new("t", "T", "Hello, {{name}}!"));
        let result = engine
            .render("t", &vars(&[("name", "Alice")]))
            .expect("should succeed");
        assert_eq!(result, "Hello, Alice!");
    }

    #[test]
    fn test_render_not_found() {
        let engine = TemplateEngine::new();
        let err = engine.render("missing", &vars(&[]));
        assert_eq!(err, Err(TemplateError::NotFound("missing".to_string())));
    }

    #[test]
    fn test_render_missing_required() {
        let mut engine = TemplateEngine::new();
        engine.register(
            PromptTemplate::new("t", "T", "{{x}}").with_var(TemplateVar::required("x", "required")),
        );
        let err = engine.render("t", &vars(&[]));
        assert_eq!(err, Err(TemplateError::MissingRequired("x".to_string())));
    }

    #[test]
    fn test_render_uses_default() {
        let mut engine = TemplateEngine::new();
        engine.register(
            PromptTemplate::new("t", "T", "Style: {{style}}")
                .with_var(TemplateVar::optional("style", "desc", "formal")),
        );
        let result = engine.render("t", &vars(&[])).expect("should succeed");
        assert_eq!(result, "Style: formal");
    }

    #[test]
    fn test_render_overrides_default() {
        let mut engine = TemplateEngine::new();
        engine.register(
            PromptTemplate::new("t", "T", "{{style}}")
                .with_var(TemplateVar::optional("style", "desc", "formal")),
        );
        let result = engine
            .render("t", &vars(&[("style", "casual")]))
            .expect("should succeed");
        assert_eq!(result, "casual");
    }

    #[test]
    fn test_render_with_system_prompt() {
        let mut engine = TemplateEngine::new();
        engine.register(
            PromptTemplate::new("t", "T", "Body: {{q}}")
                .with_var(TemplateVar::required("q", "q"))
                .with_system("System: be helpful"),
        );
        let result = engine
            .render("t", &vars(&[("q", "hello")]))
            .expect("should succeed");
        assert!(result.contains("System: be helpful"));
        assert!(result.contains("Body: hello"));
        assert!(result.contains("---"));
    }

    // ── Conditional blocks ────────────────────────────────────────────────────

    #[test]
    fn test_conditional_block_true() {
        let mut engine = TemplateEngine::new();
        engine.register(PromptTemplate::new(
            "t",
            "T",
            "A{{#if x}} and {{x}}{{/if}}!",
        ));
        let result = engine
            .render("t", &vars(&[("x", "B")]))
            .expect("should succeed");
        assert_eq!(result, "A and B!");
    }

    #[test]
    fn test_conditional_block_false_when_absent() {
        let mut engine = TemplateEngine::new();
        engine.register(PromptTemplate::new(
            "t",
            "T",
            "A{{#if x}} and {{x}}{{/if}}!",
        ));
        let result = engine.render("t", &vars(&[])).expect("should succeed");
        assert_eq!(result, "A!");
    }

    #[test]
    fn test_conditional_block_false_when_empty() {
        let mut engine = TemplateEngine::new();
        engine.register(PromptTemplate::new("t", "T", "A{{#if x}}[{{x}}]{{/if}}B"));
        let result = engine
            .render("t", &vars(&[("x", "")]))
            .expect("should succeed");
        assert_eq!(result, "AB");
    }

    #[test]
    fn test_multiple_conditional_blocks() {
        let mut engine = TemplateEngine::new();
        engine.register(PromptTemplate::new(
            "t",
            "T",
            "{{#if a}}A{{/if}}{{#if b}}B{{/if}}",
        ));
        let result = engine
            .render("t", &vars(&[("a", "yes"), ("b", "")]))
            .expect("should succeed");
        assert_eq!(result, "A");
    }

    // ── validate ──────────────────────────────────────────────────────────────

    #[test]
    fn test_validate_all_present() {
        let mut engine = TemplateEngine::new();
        engine.register(
            PromptTemplate::new("t", "T", "{{a}} {{b}}")
                .with_var(TemplateVar::required("a", ""))
                .with_var(TemplateVar::required("b", "")),
        );
        let missing = engine.validate("t", &vars(&[("a", "1"), ("b", "2")]));
        assert!(missing.is_empty());
    }

    #[test]
    fn test_validate_missing_required() {
        let mut engine = TemplateEngine::new();
        engine.register(
            PromptTemplate::new("t", "T", "{{a}}").with_var(TemplateVar::required("a", "")),
        );
        let missing = engine.validate("t", &vars(&[]));
        assert_eq!(missing, vec!["a"]);
    }

    #[test]
    fn test_validate_not_found() {
        let engine = TemplateEngine::new();
        let missing = engine.validate("ghost", &vars(&[]));
        assert!(!missing.is_empty());
    }

    // ── chain ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_chain_two_templates() {
        let mut engine = TemplateEngine::new();
        engine.register(PromptTemplate::new("t1", "T1", "Hello {{name}}"));
        engine.register(
            PromptTemplate::new("t2", "T2", "Summary: {{input}}")
                .with_var(TemplateVar::required("input", "chained input")),
        );
        let v = vars(&[("name", "Alice")]);
        let result = engine.chain(&["t1", "t2"], &v).expect("should succeed");
        assert_eq!(result, "Summary: Hello Alice");
    }

    #[test]
    fn test_chain_empty_returns_error() {
        let engine = TemplateEngine::new();
        let err = engine.chain(&[], &vars(&[]));
        assert!(err.is_err());
    }

    #[test]
    fn test_chain_single_template() {
        let mut engine = TemplateEngine::new();
        engine.register(PromptTemplate::new("t1", "T1", "Value: {{v}}"));
        let result = engine
            .chain(&["t1"], &vars(&[("v", "42")]))
            .expect("should succeed");
        assert_eq!(result, "Value: 42");
    }

    #[test]
    fn test_chain_propagates_extra_vars() {
        let mut engine = TemplateEngine::new();
        engine.register(PromptTemplate::new("t1", "T1", "{{a}}"));
        engine.register(
            PromptTemplate::new("t2", "T2", "{{input}} + {{b}}")
                .with_var(TemplateVar::required("input", "")),
        );
        let v = vars(&[("a", "X"), ("b", "Y")]);
        let result = engine.chain(&["t1", "t2"], &v).expect("should succeed");
        assert_eq!(result, "X + Y");
    }

    // ── Built-in templates ────────────────────────────────────────────────────

    #[test]
    fn test_builtin_sparql_query() {
        let engine = TemplateEngine::with_builtins();
        let v = vars(&[("question", "Who is Alice?")]);
        let result = engine.render("sparql_query", &v).expect("should succeed");
        assert!(result.contains("Who is Alice?"));
    }

    #[test]
    fn test_builtin_rag_answer() {
        let engine = TemplateEngine::with_builtins();
        let v = vars(&[
            ("context", "Alice is a developer."),
            ("question", "Who is Alice?"),
        ]);
        let result = engine.render("rag_answer", &v).expect("should succeed");
        assert!(result.contains("Alice is a developer."));
        assert!(result.contains("Who is Alice?"));
    }

    #[test]
    fn test_builtin_summarize() {
        let engine = TemplateEngine::with_builtins();
        let v = vars(&[("input", "Long text here.")]);
        let result = engine.render("summarize", &v).expect("should succeed");
        assert!(result.contains("Long text here."));
    }

    #[test]
    fn test_builtin_classify() {
        let engine = TemplateEngine::with_builtins();
        let v = vars(&[
            ("input", "I love Rust!"),
            ("categories", "positive,negative"),
        ]);
        let result = engine.render("classify", &v).expect("should succeed");
        assert!(result.contains("positive,negative"));
        assert!(result.contains("I love Rust!"));
    }

    #[test]
    fn test_builtin_sparql_with_graph() {
        let engine = TemplateEngine::with_builtins();
        let v = vars(&[
            ("question", "Count triples"),
            ("graph", "http://example.org/graph"),
        ]);
        let result = engine.render("sparql_query", &v).expect("should succeed");
        assert!(result.contains("http://example.org/graph"));
    }

    #[test]
    fn test_builtin_sparql_without_graph_no_angle_brackets() {
        let engine = TemplateEngine::with_builtins();
        let v = vars(&[("question", "Count triples")]);
        let result = engine.render("sparql_query", &v).expect("should succeed");
        // The {{#if graph}} block should be suppressed
        assert!(!result.contains("<>"));
    }

    #[test]
    fn test_four_builtins_registered() {
        let engine = TemplateEngine::with_builtins();
        assert!(engine.get("sparql_query").is_some());
        assert!(engine.get("rag_answer").is_some());
        assert!(engine.get("summarize").is_some());
        assert!(engine.get("classify").is_some());
    }

    // ── engine default ────────────────────────────────────────────────────────

    #[test]
    fn test_engine_default_empty() {
        let engine = TemplateEngine::default();
        assert!(engine.list_ids().is_empty());
    }

    // ── Multi-variable rendering ───────────────────────────────────────────────

    #[test]
    fn test_render_multiple_vars() {
        let mut engine = TemplateEngine::new();
        engine.register(PromptTemplate::new("t", "T", "{{a}}-{{b}}-{{c}}"));
        let result = engine
            .render("t", &vars(&[("a", "1"), ("b", "2"), ("c", "3")]))
            .expect("should succeed");
        assert_eq!(result, "1-2-3");
    }

    #[test]
    fn test_render_repeated_var() {
        let mut engine = TemplateEngine::new();
        engine.register(PromptTemplate::new("t", "T", "{{x}} and {{x}}"));
        let result = engine
            .render("t", &vars(&[("x", "hello")]))
            .expect("should succeed");
        assert_eq!(result, "hello and hello");
    }

    #[test]
    fn test_render_unknown_var_silently_dropped() {
        let mut engine = TemplateEngine::new();
        engine.register(PromptTemplate::new("t", "T", "A{{z}}B"));
        let result = engine.render("t", &vars(&[])).expect("should succeed");
        assert_eq!(result, "AB");
    }

    #[test]
    fn test_validate_optional_not_required() {
        let mut engine = TemplateEngine::new();
        engine.register(
            PromptTemplate::new("t", "T", "{{opt}}")
                .with_var(TemplateVar::optional("opt", "", "default")),
        );
        let missing = engine.validate("t", &vars(&[]));
        assert!(missing.is_empty());
    }
}
