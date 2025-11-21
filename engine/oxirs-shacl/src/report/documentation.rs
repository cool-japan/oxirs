//! Shape Documentation Generator
//!
//! This module generates comprehensive documentation for SHACL shapes in various formats
//! including Markdown, HTML, and JSON Schema-like documentation.
//!
//! # Features
//!
//! - Automatic documentation from shape definitions
//! - Multiple output formats (Markdown, HTML, reStructuredText)
//! - Cross-reference generation
//! - Constraint descriptions with examples
//! - Property path documentation
//! - Inheritance/extension visualization

use crate::{Constraint, PropertyPath, Shape, ShapeId, ShapeType, Target};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Documentation generator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationConfig {
    /// Output format
    pub format: DocumentationFormat,
    /// Include constraint details
    pub include_constraint_details: bool,
    /// Include property paths
    pub include_property_paths: bool,
    /// Include examples
    pub include_examples: bool,
    /// Include cross-references
    pub include_cross_references: bool,
    /// Include table of contents
    pub include_toc: bool,
    /// Maximum depth for nested shapes
    pub max_nesting_depth: usize,
    /// Custom CSS for HTML output
    pub custom_css: Option<String>,
    /// Title for the documentation
    pub title: String,
    /// Description for the documentation
    pub description: Option<String>,
}

impl Default for DocumentationConfig {
    fn default() -> Self {
        Self {
            format: DocumentationFormat::Markdown,
            include_constraint_details: true,
            include_property_paths: true,
            include_examples: true,
            include_cross_references: true,
            include_toc: true,
            max_nesting_depth: 3,
            custom_css: None,
            title: "SHACL Shapes Documentation".to_string(),
            description: None,
        }
    }
}

/// Documentation output formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DocumentationFormat {
    /// Markdown format
    Markdown,
    /// HTML format
    Html,
    /// reStructuredText format
    ReStructuredText,
    /// AsciiDoc format
    AsciiDoc,
}

/// Shape documentation generator
pub struct ShapeDocumentationGenerator {
    config: DocumentationConfig,
}

impl ShapeDocumentationGenerator {
    /// Create a new documentation generator
    pub fn new(config: DocumentationConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(DocumentationConfig::default())
    }

    /// Generate documentation for a collection of shapes
    pub fn generate(&self, shapes: &IndexMap<ShapeId, Shape>) -> String {
        match self.config.format {
            DocumentationFormat::Markdown => self.generate_markdown(shapes),
            DocumentationFormat::Html => self.generate_html(shapes),
            DocumentationFormat::ReStructuredText => self.generate_rst(shapes),
            DocumentationFormat::AsciiDoc => self.generate_asciidoc(shapes),
        }
    }

    /// Generate Markdown documentation
    fn generate_markdown(&self, shapes: &IndexMap<ShapeId, Shape>) -> String {
        let mut output = String::new();

        // Title
        output.push_str(&format!("# {}\n\n", self.config.title));

        // Description
        if let Some(desc) = &self.config.description {
            output.push_str(&format!("{}\n\n", desc));
        }

        // Table of Contents
        if self.config.include_toc {
            output.push_str("## Table of Contents\n\n");
            for shape in shapes.values() {
                let anchor = self.make_anchor(shape.id.as_str());
                output.push_str(&format!(
                    "- [{}](#{})\n",
                    shape.label.as_deref().unwrap_or(shape.id.as_str()),
                    anchor
                ));
            }
            output.push_str("\n---\n\n");
        }

        // Cross-references
        let cross_refs = if self.config.include_cross_references {
            self.build_cross_references(shapes)
        } else {
            HashMap::new()
        };

        // Generate documentation for each shape
        for shape in shapes.values() {
            output.push_str(&self.generate_shape_markdown(shape, &cross_refs));
            output.push_str("\n---\n\n");
        }

        output
    }

    /// Generate Markdown for a single shape
    fn generate_shape_markdown(
        &self,
        shape: &Shape,
        cross_refs: &HashMap<ShapeId, Vec<ShapeId>>,
    ) -> String {
        let mut output = String::new();

        // Shape header
        let title = shape.label.as_deref().unwrap_or(shape.id.as_str());
        output.push_str(&format!("## {}\n\n", title));

        // Shape ID
        output.push_str(&format!("**ID:** `{}`\n\n", shape.id.as_str()));

        // Shape type
        let shape_type = match shape.shape_type {
            ShapeType::NodeShape => "Node Shape",
            ShapeType::PropertyShape => "Property Shape",
        };
        output.push_str(&format!("**Type:** {}\n\n", shape_type));

        // Description
        if let Some(desc) = &shape.description {
            output.push_str(&format!("{}\n\n", desc));
        }

        // Targets
        if !shape.targets.is_empty() {
            output.push_str("### Targets\n\n");
            for target in &shape.targets {
                output.push_str(&format!("- {}\n", self.format_target(target)));
            }
            output.push('\n');
        }

        // Property path (for property shapes)
        if self.config.include_property_paths {
            if let Some(path) = &shape.path {
                output.push_str("### Property Path\n\n");
                output.push_str(&format!("`{}`\n\n", self.format_property_path(path)));
            }
        }

        // Constraints
        if !shape.constraints.is_empty() && self.config.include_constraint_details {
            output.push_str("### Constraints\n\n");
            output.push_str("| Constraint | Value | Description |\n");
            output.push_str("|------------|-------|-------------|\n");

            for (id, constraint) in &shape.constraints {
                let (value, desc) = self.describe_constraint(constraint);
                output.push_str(&format!("| `{}` | {} | {} |\n", id.as_str(), value, desc));
            }
            output.push('\n');
        }

        // Inheritance
        if !shape.extends.is_empty() {
            output.push_str("### Extends\n\n");
            for parent in &shape.extends {
                output.push_str(&format!("- `{}`\n", parent.as_str()));
            }
            output.push('\n');
        }

        // Cross-references
        if self.config.include_cross_references {
            if let Some(refs) = cross_refs.get(&shape.id) {
                if !refs.is_empty() {
                    output.push_str("### Referenced By\n\n");
                    for ref_id in refs {
                        output.push_str(&format!("- `{}`\n", ref_id.as_str()));
                    }
                    output.push('\n');
                }
            }
        }

        // Examples
        if self.config.include_examples {
            output.push_str("### Example\n\n");
            output.push_str("```turtle\n");
            output.push_str(&self.generate_example_turtle(shape));
            output.push_str("```\n\n");
        }

        output
    }

    /// Generate HTML documentation
    fn generate_html(&self, shapes: &IndexMap<ShapeId, Shape>) -> String {
        let mut output = String::new();

        // HTML header
        output.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
        output.push_str("  <meta charset=\"UTF-8\">\n");
        output.push_str(
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
        );
        output.push_str(&format!("  <title>{}</title>\n", self.config.title));

        // CSS
        if let Some(css) = &self.config.custom_css {
            output.push_str(&format!("  <style>{}</style>\n", css));
        } else {
            output.push_str("  <style>\n");
            output.push_str("    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 2rem; }\n");
            output.push_str("    h1 { color: #1a365d; border-bottom: 2px solid #2b6cb0; padding-bottom: 0.5rem; }\n");
            output.push_str("    h2 { color: #2c5282; margin-top: 2rem; }\n");
            output.push_str("    h3 { color: #4a5568; }\n");
            output.push_str("    code { background: #edf2f7; padding: 0.2rem 0.4rem; border-radius: 0.25rem; }\n");
            output.push_str("    pre { background: #2d3748; color: #e2e8f0; padding: 1rem; border-radius: 0.5rem; overflow-x: auto; }\n");
            output.push_str(
                "    table { border-collapse: collapse; width: 100%; margin: 1rem 0; }\n",
            );
            output.push_str(
                "    th, td { border: 1px solid #e2e8f0; padding: 0.75rem; text-align: left; }\n",
            );
            output.push_str("    th { background: #edf2f7; }\n");
            output.push_str("    .shape-card { border: 1px solid #e2e8f0; border-radius: 0.5rem; padding: 1.5rem; margin: 1rem 0; }\n");
            output.push_str(
                "    .toc { background: #f7fafc; padding: 1rem; border-radius: 0.5rem; }\n",
            );
            output.push_str("    .toc ul { list-style: none; padding-left: 1rem; }\n");
            output.push_str("    a { color: #2b6cb0; text-decoration: none; }\n");
            output.push_str("    a:hover { text-decoration: underline; }\n");
            output.push_str("  </style>\n");
        }

        output.push_str("</head>\n<body>\n");

        // Title
        output.push_str(&format!("<h1>{}</h1>\n", self.config.title));

        // Description
        if let Some(desc) = &self.config.description {
            output.push_str(&format!("<p>{}</p>\n", desc));
        }

        // Table of Contents
        if self.config.include_toc {
            output.push_str("<div class=\"toc\">\n<h2>Table of Contents</h2>\n<ul>\n");
            for shape in shapes.values() {
                let anchor = self.make_anchor(shape.id.as_str());
                let title = shape.label.as_deref().unwrap_or(shape.id.as_str());
                output.push_str(&format!(
                    "  <li><a href=\"#{}\">{}</a></li>\n",
                    anchor, title
                ));
            }
            output.push_str("</ul>\n</div>\n");
        }

        // Generate documentation for each shape
        let cross_refs = if self.config.include_cross_references {
            self.build_cross_references(shapes)
        } else {
            HashMap::new()
        };

        for shape in shapes.values() {
            output.push_str(&self.generate_shape_html(shape, &cross_refs));
        }

        output.push_str("</body>\n</html>");
        output
    }

    /// Generate HTML for a single shape
    fn generate_shape_html(
        &self,
        shape: &Shape,
        _cross_refs: &HashMap<ShapeId, Vec<ShapeId>>,
    ) -> String {
        let mut output = String::new();

        let anchor = self.make_anchor(shape.id.as_str());
        let title = shape.label.as_deref().unwrap_or(shape.id.as_str());

        output.push_str(&format!("<div class=\"shape-card\" id=\"{}\">\n", anchor));
        output.push_str(&format!("<h2>{}</h2>\n", title));

        // Shape ID and type
        output.push_str(&format!(
            "<p><strong>ID:</strong> <code>{}</code></p>\n",
            shape.id.as_str()
        ));
        let shape_type = match shape.shape_type {
            ShapeType::NodeShape => "Node Shape",
            ShapeType::PropertyShape => "Property Shape",
        };
        output.push_str(&format!("<p><strong>Type:</strong> {}</p>\n", shape_type));

        // Description
        if let Some(desc) = &shape.description {
            output.push_str(&format!("<p>{}</p>\n", desc));
        }

        // Constraints
        if !shape.constraints.is_empty() && self.config.include_constraint_details {
            output.push_str("<h3>Constraints</h3>\n<table>\n");
            output.push_str("<tr><th>Constraint</th><th>Value</th><th>Description</th></tr>\n");

            for (id, constraint) in &shape.constraints {
                let (value, desc) = self.describe_constraint(constraint);
                output.push_str(&format!(
                    "<tr><td><code>{}</code></td><td>{}</td><td>{}</td></tr>\n",
                    id.as_str(),
                    value,
                    desc
                ));
            }
            output.push_str("</table>\n");
        }

        // Example
        if self.config.include_examples {
            output.push_str("<h3>Example</h3>\n<pre>");
            output.push_str(&self.generate_example_turtle(shape));
            output.push_str("</pre>\n");
        }

        output.push_str("</div>\n");
        output
    }

    /// Generate reStructuredText documentation
    fn generate_rst(&self, shapes: &IndexMap<ShapeId, Shape>) -> String {
        let mut output = String::new();

        // Title
        let title_underline = "=".repeat(self.config.title.len());
        output.push_str(&format!("{}\n{}\n\n", self.config.title, title_underline));

        // Description
        if let Some(desc) = &self.config.description {
            output.push_str(&format!("{}\n\n", desc));
        }

        // Table of Contents
        if self.config.include_toc {
            output.push_str(".. contents:: Table of Contents\n   :depth: 2\n\n");
        }

        // Generate documentation for each shape
        for shape in shapes.values() {
            output.push_str(&self.generate_shape_rst(shape));
            output.push('\n');
        }

        output
    }

    /// Generate reStructuredText for a single shape
    fn generate_shape_rst(&self, shape: &Shape) -> String {
        let mut output = String::new();

        let title = shape.label.as_deref().unwrap_or(shape.id.as_str());
        let underline = "-".repeat(title.len());
        output.push_str(&format!("{}\n{}\n\n", title, underline));

        // Shape ID
        output.push_str(&format!("**ID:** ``{}``\n\n", shape.id.as_str()));

        // Description
        if let Some(desc) = &shape.description {
            output.push_str(&format!("{}\n\n", desc));
        }

        // Constraints
        if !shape.constraints.is_empty() && self.config.include_constraint_details {
            output.push_str("Constraints\n~~~~~~~~~~~\n\n");
            output.push_str(".. list-table::\n");
            output.push_str("   :header-rows: 1\n\n");
            output.push_str("   * - Constraint\n     - Value\n     - Description\n");

            for (id, constraint) in &shape.constraints {
                let (value, desc) = self.describe_constraint(constraint);
                output.push_str(&format!(
                    "   * - ``{}``\n     - {}\n     - {}\n",
                    id.as_str(),
                    value,
                    desc
                ));
            }
            output.push('\n');
        }

        // Example
        if self.config.include_examples {
            output.push_str("Example\n~~~~~~~\n\n");
            output.push_str(".. code-block:: turtle\n\n");
            for line in self.generate_example_turtle(shape).lines() {
                output.push_str(&format!("   {}\n", line));
            }
            output.push('\n');
        }

        output
    }

    /// Generate AsciiDoc documentation
    fn generate_asciidoc(&self, shapes: &IndexMap<ShapeId, Shape>) -> String {
        let mut output = String::new();

        // Title
        output.push_str(&format!("= {}\n\n", self.config.title));

        // Description
        if let Some(desc) = &self.config.description {
            output.push_str(&format!("{}\n\n", desc));
        }

        // Table of Contents
        if self.config.include_toc {
            output.push_str(":toc:\n:toclevels: 2\n\n");
        }

        // Generate documentation for each shape
        for shape in shapes.values() {
            output.push_str(&self.generate_shape_asciidoc(shape));
            output.push('\n');
        }

        output
    }

    /// Generate AsciiDoc for a single shape
    fn generate_shape_asciidoc(&self, shape: &Shape) -> String {
        let mut output = String::new();

        let title = shape.label.as_deref().unwrap_or(shape.id.as_str());
        output.push_str(&format!("== {}\n\n", title));

        // Shape ID
        output.push_str(&format!("*ID:* `{}`\n\n", shape.id.as_str()));

        // Description
        if let Some(desc) = &shape.description {
            output.push_str(&format!("{}\n\n", desc));
        }

        // Constraints
        if !shape.constraints.is_empty() && self.config.include_constraint_details {
            output.push_str("=== Constraints\n\n");
            output.push_str("[cols=\"1,1,2\", options=\"header\"]\n|===\n");
            output.push_str("|Constraint |Value |Description\n\n");

            for (id, constraint) in &shape.constraints {
                let (value, desc) = self.describe_constraint(constraint);
                output.push_str(&format!("|`{}`\n|{}\n|{}\n\n", id.as_str(), value, desc));
            }
            output.push_str("|===\n\n");
        }

        // Example
        if self.config.include_examples {
            output.push_str("=== Example\n\n");
            output.push_str("[source,turtle]\n----\n");
            output.push_str(&self.generate_example_turtle(shape));
            output.push_str("----\n\n");
        }

        output
    }

    /// Build cross-reference map
    fn build_cross_references(
        &self,
        shapes: &IndexMap<ShapeId, Shape>,
    ) -> HashMap<ShapeId, Vec<ShapeId>> {
        let mut refs: HashMap<ShapeId, Vec<ShapeId>> = HashMap::new();

        for shape in shapes.values() {
            // Check constraints for shape references
            for constraint in shape.constraints.values() {
                let referenced_shapes = self.get_referenced_shapes(constraint);
                for ref_id in referenced_shapes {
                    refs.entry(ref_id).or_default().push(shape.id.clone());
                }
            }

            // Check extends
            for parent in &shape.extends {
                refs.entry(parent.clone())
                    .or_default()
                    .push(shape.id.clone());
            }
        }

        refs
    }

    /// Get shapes referenced by a constraint
    fn get_referenced_shapes(&self, constraint: &Constraint) -> Vec<ShapeId> {
        match constraint {
            Constraint::Node(c) => vec![c.shape.clone()],
            Constraint::And(c) => c.shapes.clone(),
            Constraint::Or(c) => c.shapes.clone(),
            Constraint::Not(c) => vec![c.shape.clone()],
            Constraint::Xone(c) => c.shapes.clone(),
            Constraint::QualifiedValueShape(c) => vec![c.shape.clone()],
            _ => Vec::new(),
        }
    }

    /// Format a target for documentation
    #[allow(clippy::only_used_in_recursion)]
    fn format_target(&self, target: &Target) -> String {
        match target {
            Target::Class(class) => format!("Class: `{}`", class),
            Target::Node(node) => format!("Node: `{:?}`", node),
            Target::SubjectsOf(prop) => format!("Subjects of: `{}`", prop),
            Target::ObjectsOf(prop) => format!("Objects of: `{}`", prop),
            Target::Sparql(sparql) => format!("SPARQL: `{}`", sparql.query),
            Target::Implicit(class) => format!("Implicit class: `{}`", class),
            Target::Union(union) => format!("Union of {} targets", union.targets.len()),
            Target::Intersection(intersection) => {
                format!("Intersection of {} targets", intersection.targets.len())
            }
            Target::Difference(diff) => format!(
                "Difference ({} - {})",
                self.format_target(&diff.primary_target),
                self.format_target(&diff.exclusion_target)
            ),
            Target::Conditional(cond) => {
                format!("Conditional: {}", self.format_target(&cond.base_target))
            }
            Target::Hierarchical(hier) => format!("Hierarchical: {} levels", hier.max_depth),
            Target::PathBased(path) => format!("Path-based: {:?}", path.path),
        }
    }

    /// Format a property path
    #[allow(clippy::only_used_in_recursion)]
    fn format_property_path(&self, path: &PropertyPath) -> String {
        match path {
            PropertyPath::Predicate(p) => p.to_string(),
            PropertyPath::Sequence(paths) => {
                let parts: Vec<_> = paths.iter().map(|p| self.format_property_path(p)).collect();
                parts.join(" / ")
            }
            PropertyPath::Alternative(paths) => {
                let parts: Vec<_> = paths.iter().map(|p| self.format_property_path(p)).collect();
                format!("({})", parts.join(" | "))
            }
            PropertyPath::Inverse(p) => format!("^{}", self.format_property_path(p)),
            PropertyPath::ZeroOrMore(p) => format!("{}*", self.format_property_path(p)),
            PropertyPath::OneOrMore(p) => format!("{}+", self.format_property_path(p)),
            PropertyPath::ZeroOrOne(p) => format!("{}?", self.format_property_path(p)),
        }
    }

    /// Describe a constraint
    fn describe_constraint(&self, constraint: &Constraint) -> (String, String) {
        match constraint {
            Constraint::MinCount(c) => (
                c.min_count.to_string(),
                "Minimum number of values".to_string(),
            ),
            Constraint::MaxCount(c) => (
                c.max_count.to_string(),
                "Maximum number of values".to_string(),
            ),
            Constraint::Datatype(c) => (
                format!("`{}`", c.datatype_iri),
                "Required datatype".to_string(),
            ),
            Constraint::NodeKind(c) => (
                format!("{:?}", c.node_kind),
                "Required node kind".to_string(),
            ),
            Constraint::Class(c) => (
                format!("`{}`", c.class_iri),
                "Required RDF class".to_string(),
            ),
            Constraint::MinLength(c) => (
                c.min_length.to_string(),
                "Minimum string length".to_string(),
            ),
            Constraint::MaxLength(c) => (
                c.max_length.to_string(),
                "Maximum string length".to_string(),
            ),
            Constraint::Pattern(c) => (
                format!("`{}`", c.pattern),
                format!(
                    "Regex pattern{}",
                    c.flags
                        .as_ref()
                        .map(|f| format!(" (flags: {})", f))
                        .unwrap_or_default()
                ),
            ),
            Constraint::MinInclusive(c) => (
                c.min_value.to_string(),
                "Minimum value (inclusive)".to_string(),
            ),
            Constraint::MaxInclusive(c) => (
                c.max_value.to_string(),
                "Maximum value (inclusive)".to_string(),
            ),
            Constraint::MinExclusive(c) => (
                c.min_value.to_string(),
                "Minimum value (exclusive)".to_string(),
            ),
            Constraint::MaxExclusive(c) => (
                c.max_value.to_string(),
                "Maximum value (exclusive)".to_string(),
            ),
            Constraint::In(c) => (
                format!("{} values", c.values.len()),
                "Allowed values".to_string(),
            ),
            Constraint::LanguageIn(c) => (c.languages.join(", "), "Allowed languages".to_string()),
            Constraint::UniqueLang(_) => {
                ("true".to_string(), "Unique language per value".to_string())
            }
            Constraint::HasValue(c) => (format!("`{}`", c.value), "Required value".to_string()),
            Constraint::Closed(c) => (
                format!("{} allowed properties", c.allowed_properties.len()),
                "Closed shape".to_string(),
            ),
            Constraint::And(c) => (
                format!("{} shapes", c.shapes.len()),
                "All shapes must match".to_string(),
            ),
            Constraint::Or(c) => (
                format!("{} shapes", c.shapes.len()),
                "At least one shape must match".to_string(),
            ),
            Constraint::Not(c) => (
                format!("`{}`", c.shape.as_str()),
                "Shape must not match".to_string(),
            ),
            Constraint::Xone(c) => (
                format!("{} shapes", c.shapes.len()),
                "Exactly one shape must match".to_string(),
            ),
            Constraint::Node(c) => (
                format!("`{}`", c.shape.as_str()),
                "Referenced node shape".to_string(),
            ),
            Constraint::QualifiedValueShape(c) => {
                let count_str = match (c.qualified_min_count, c.qualified_max_count) {
                    (Some(min), Some(max)) => format!("{}-{}", min, max),
                    (Some(min), None) => format!("min {}", min),
                    (None, Some(max)) => format!("max {}", max),
                    (None, None) => "unspecified".to_string(),
                };
                (count_str, "Qualified value shape constraint".to_string())
            }
            Constraint::Equals(c) => (format!("`{}`", c.property), "Values must equal".to_string()),
            Constraint::Disjoint(c) => (
                format!("`{}`", c.property),
                "Values must be disjoint".to_string(),
            ),
            Constraint::LessThan(c) => (
                format!("`{}`", c.property),
                "Values must be less than".to_string(),
            ),
            Constraint::LessThanOrEquals(c) => (
                format!("`{}`", c.property),
                "Values must be less than or equal".to_string(),
            ),
            Constraint::Sparql(c) => (
                format!("SPARQL: {}", c.query.chars().take(50).collect::<String>()),
                "SPARQL-based constraint".to_string(),
            ),
        }
    }

    /// Generate example Turtle syntax for a shape
    fn generate_example_turtle(&self, shape: &Shape) -> String {
        let mut output = String::new();

        output.push_str(&format!("{}\n", shape.id.as_str()));
        output.push_str("    a sh:NodeShape ;\n");

        // Label
        if let Some(label) = &shape.label {
            output.push_str(&format!("    rdfs:label \"{}\" ;\n", label));
        }

        // Targets
        for target in &shape.targets {
            match target {
                Target::Class(class) => {
                    output.push_str(&format!("    sh:targetClass {} ;\n", class));
                }
                Target::Node(node) => {
                    output.push_str(&format!("    sh:targetNode {:?} ;\n", node));
                }
                Target::SubjectsOf(prop) => {
                    output.push_str(&format!("    sh:targetSubjectsOf {} ;\n", prop));
                }
                Target::ObjectsOf(prop) => {
                    output.push_str(&format!("    sh:targetObjectsOf {} ;\n", prop));
                }
                _ => {}
            }
        }

        // Constraints (simplified)
        for (id, constraint) in shape.constraints.iter().take(3) {
            let constraint_name = id.as_str().split(':').next_back().unwrap_or(id.as_str());
            match constraint {
                Constraint::MinCount(c) => {
                    output.push_str(&format!("    sh:minCount {} ;\n", c.min_count));
                }
                Constraint::MaxCount(c) => {
                    output.push_str(&format!("    sh:maxCount {} ;\n", c.max_count));
                }
                Constraint::Datatype(c) => {
                    output.push_str(&format!("    sh:datatype {} ;\n", c.datatype_iri));
                }
                _ => {
                    output.push_str(&format!("    # {} constraint\n", constraint_name));
                }
            }
        }

        if shape.constraints.len() > 3 {
            output.push_str(&format!(
                "    # ... {} more constraints\n",
                shape.constraints.len() - 3
            ));
        }

        output.push_str("    .\n");
        output
    }

    /// Make URL-safe anchor from shape ID
    fn make_anchor(&self, id: &str) -> String {
        id.chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    c.to_ascii_lowercase()
                } else {
                    '-'
                }
            })
            .collect()
    }
}

impl Default for ShapeDocumentationGenerator {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Builder for documentation generator
pub struct DocumentationBuilder {
    config: DocumentationConfig,
}

impl DocumentationBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: DocumentationConfig::default(),
        }
    }

    /// Set output format
    pub fn format(mut self, format: DocumentationFormat) -> Self {
        self.config.format = format;
        self
    }

    /// Set title
    pub fn title(mut self, title: &str) -> Self {
        self.config.title = title.to_string();
        self
    }

    /// Set description
    pub fn description(mut self, description: &str) -> Self {
        self.config.description = Some(description.to_string());
        self
    }

    /// Include/exclude constraint details
    pub fn include_constraints(mut self, include: bool) -> Self {
        self.config.include_constraint_details = include;
        self
    }

    /// Include/exclude examples
    pub fn include_examples(mut self, include: bool) -> Self {
        self.config.include_examples = include;
        self
    }

    /// Include/exclude table of contents
    pub fn include_toc(mut self, include: bool) -> Self {
        self.config.include_toc = include;
        self
    }

    /// Set custom CSS for HTML output
    pub fn custom_css(mut self, css: &str) -> Self {
        self.config.custom_css = Some(css.to_string());
        self
    }

    /// Build the documentation generator
    pub fn build(self) -> ShapeDocumentationGenerator {
        ShapeDocumentationGenerator::new(self.config)
    }
}

impl Default for DocumentationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::MinCountConstraint;

    fn create_test_shape() -> Shape {
        let mut shape = Shape::new(ShapeId::new("ex:PersonShape"), ShapeType::NodeShape);
        shape.label = Some("Person Shape".to_string());
        shape.description = Some("Validates person entities".to_string());
        shape.add_constraint(
            crate::ConstraintComponentId::new("sh:minCount"),
            Constraint::MinCount(MinCountConstraint { min_count: 1 }),
        );
        shape
    }

    #[test]
    fn test_generator_creation() {
        let generator = ShapeDocumentationGenerator::default_config();
        assert_eq!(generator.config.format, DocumentationFormat::Markdown);
    }

    #[test]
    fn test_markdown_generation() {
        let generator = ShapeDocumentationGenerator::default_config();
        let mut shapes = IndexMap::new();
        let shape = create_test_shape();
        shapes.insert(shape.id.clone(), shape);

        let output = generator.generate(&shapes);

        assert!(output.contains("# SHACL Shapes Documentation"));
        assert!(output.contains("Person Shape"));
        assert!(output.contains("sh:minCount"));
    }

    #[test]
    fn test_html_generation() {
        let generator = ShapeDocumentationGenerator::new(DocumentationConfig {
            format: DocumentationFormat::Html,
            ..Default::default()
        });

        let mut shapes = IndexMap::new();
        let shape = create_test_shape();
        shapes.insert(shape.id.clone(), shape);

        let output = generator.generate(&shapes);

        assert!(output.contains("<!DOCTYPE html>"));
        assert!(output.contains("<h1>"));
        assert!(output.contains("Person Shape"));
    }

    #[test]
    fn test_builder_pattern() {
        let generator = DocumentationBuilder::new()
            .format(DocumentationFormat::Html)
            .title("My Shapes")
            .description("Custom shapes for my project")
            .include_toc(false)
            .build();

        assert_eq!(generator.config.format, DocumentationFormat::Html);
        assert_eq!(generator.config.title, "My Shapes");
        assert!(!generator.config.include_toc);
    }

    #[test]
    fn test_constraint_description() {
        let generator = ShapeDocumentationGenerator::default_config();

        let constraint = Constraint::MinCount(MinCountConstraint { min_count: 1 });
        let (value, desc) = generator.describe_constraint(&constraint);

        assert_eq!(value, "1");
        assert!(desc.contains("Minimum"));
    }

    #[test]
    fn test_rst_generation() {
        let generator = ShapeDocumentationGenerator::new(DocumentationConfig {
            format: DocumentationFormat::ReStructuredText,
            ..Default::default()
        });

        let mut shapes = IndexMap::new();
        let shape = create_test_shape();
        shapes.insert(shape.id.clone(), shape);

        let output = generator.generate(&shapes);

        assert!(output.contains("="));
        assert!(output.contains(".. list-table::"));
    }

    #[test]
    fn test_asciidoc_generation() {
        let generator = ShapeDocumentationGenerator::new(DocumentationConfig {
            format: DocumentationFormat::AsciiDoc,
            ..Default::default()
        });

        let mut shapes = IndexMap::new();
        let shape = create_test_shape();
        shapes.insert(shape.id.clone(), shape);

        let output = generator.generate(&shapes);

        assert!(output.contains("= "));
        assert!(output.contains("[cols="));
    }

    #[test]
    fn test_anchor_generation() {
        let generator = ShapeDocumentationGenerator::default_config();

        let anchor = generator.make_anchor("ex:PersonShape");
        assert_eq!(anchor, "ex-personshape");
    }
}
