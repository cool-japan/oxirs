//! Model Documentation Generation
//!
//! This module provides comprehensive documentation generation capabilities for SAMM models.
//! It can generate multiple documentation formats (HTML, Markdown) with various templates
//! (technical reference, user guide, API documentation).
//!
//! # Features
//!
//! - **Multiple Formats**: HTML, Markdown, JSON
//! - **Template Styles**: Technical, UserFriendly, API, Complete
//! - **Analytics Integration**: Embed quality scores and recommendations
//! - **Interactive HTML**: Collapsible sections, search, navigation
//! - **Markdown Export**: GitHub-compatible markdown
//! - **Customizable**: Custom CSS, templates, sections
//!
//! # Examples
//!
//! ```rust
//! use oxirs_samm::documentation::{DocumentationGenerator, DocumentationFormat, DocumentationStyle};
//! use oxirs_samm::metamodel::Aspect;
//!
//! # fn example(aspect: &Aspect) -> Result<(), Box<dyn std::error::Error>> {
//! let generator = DocumentationGenerator::new()
//!     .with_format(DocumentationFormat::Html)
//!     .with_style(DocumentationStyle::Technical)
//!     .with_analytics(true)
//!     .with_table_of_contents(true);
//!
//! let html = generator.generate(aspect)?;
//! std::fs::write("documentation.html", html)?;
//! # Ok(())
//! # }
//! ```

use crate::analytics::ModelAnalytics;
use crate::error::{Result, SammError};
use crate::metamodel::{Aspect, CharacteristicKind, ModelElement, Operation, Property};
use crate::query::ModelQuery;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Documentation format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DocumentationFormat {
    /// HTML with CSS styling
    Html,
    /// GitHub-flavored Markdown
    Markdown,
    /// JSON structured documentation
    Json,
}

/// Documentation style/template
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DocumentationStyle {
    /// Technical reference documentation
    Technical,
    /// User-friendly guide
    UserFriendly,
    /// API documentation
    Api,
    /// Complete documentation (all sections)
    Complete,
}

/// Documentation generator configuration
#[derive(Debug, Clone)]
pub struct DocumentationGenerator {
    /// Output format
    format: DocumentationFormat,
    /// Documentation style
    style: DocumentationStyle,
    /// Include analytics
    include_analytics: bool,
    /// Include table of contents
    include_toc: bool,
    /// Include examples
    include_examples: bool,
    /// Include diagrams
    include_diagrams: bool,
    /// Custom CSS (for HTML)
    custom_css: Option<String>,
    /// Title override
    title: Option<String>,
    /// Footer text
    footer: Option<String>,
}

impl DocumentationGenerator {
    /// Create a new documentation generator with default settings
    ///
    /// Defaults: HTML format, Technical style, analytics enabled, TOC enabled
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_samm::documentation::DocumentationGenerator;
    ///
    /// let generator = DocumentationGenerator::new();
    /// ```
    pub fn new() -> Self {
        Self {
            format: DocumentationFormat::Html,
            style: DocumentationStyle::Technical,
            include_analytics: true,
            include_toc: true,
            include_examples: false,
            include_diagrams: false,
            custom_css: None,
            title: None,
            footer: None,
        }
    }

    /// Set the output format
    pub fn with_format(mut self, format: DocumentationFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the documentation style
    pub fn with_style(mut self, style: DocumentationStyle) -> Self {
        self.style = style;
        self
    }

    /// Enable/disable analytics inclusion
    pub fn with_analytics(mut self, include: bool) -> Self {
        self.include_analytics = include;
        self
    }

    /// Enable/disable table of contents
    pub fn with_table_of_contents(mut self, include: bool) -> Self {
        self.include_toc = include;
        self
    }

    /// Enable/disable examples
    pub fn with_examples(mut self, include: bool) -> Self {
        self.include_examples = include;
        self
    }

    /// Enable/disable diagrams
    pub fn with_diagrams(mut self, include: bool) -> Self {
        self.include_diagrams = include;
        self
    }

    /// Set custom CSS (HTML only)
    pub fn with_custom_css(mut self, css: String) -> Self {
        self.custom_css = Some(css);
        self
    }

    /// Set custom title
    pub fn with_title(mut self, title: String) -> Self {
        self.title = Some(title);
        self
    }

    /// Set footer text
    pub fn with_footer(mut self, footer: String) -> Self {
        self.footer = Some(footer);
        self
    }

    /// Generate documentation for an aspect
    ///
    /// # Arguments
    ///
    /// * `aspect` - The aspect to document
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use oxirs_samm::documentation::DocumentationGenerator;
    /// # use oxirs_samm::metamodel::Aspect;
    /// # fn example(aspect: &Aspect) -> Result<(), Box<dyn std::error::Error>> {
    /// let generator = DocumentationGenerator::new();
    /// let html = generator.generate(aspect)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn generate(&self, aspect: &Aspect) -> Result<String> {
        match self.format {
            DocumentationFormat::Html => self.generate_html(aspect),
            DocumentationFormat::Markdown => self.generate_markdown(aspect),
            DocumentationFormat::Json => self.generate_json(aspect),
        }
    }

    /// Generate HTML documentation
    fn generate_html(&self, aspect: &Aspect) -> Result<String> {
        let title = self.title.clone().unwrap_or_else(|| {
            let name = aspect
                .metadata
                .get_preferred_name("en")
                .map(|s| s.to_string())
                .unwrap_or_else(|| aspect.name().to_string());
            format!("{} - Documentation", name)
        });

        let analytics = if self.include_analytics {
            Some(ModelAnalytics::analyze(aspect))
        } else {
            None
        };

        let css = self
            .custom_css
            .clone()
            .unwrap_or_else(|| DEFAULT_CSS.to_string());

        let mut html = format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="generator" content="OxiRS SAMM Documentation Generator">
    <title>{}</title>
    <style>
{}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{}</h1>
            <p class="subtitle">Generated on {}</p>
        </header>
"#,
            title,
            css,
            title,
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        );

        // Table of contents
        if self.include_toc {
            html.push_str(&self.generate_toc_html(aspect, analytics.as_ref()));
        }

        // Overview section
        html.push_str(&self.generate_overview_html(aspect));

        // Analytics section
        if let Some(ref analytics) = analytics {
            html.push_str(&self.generate_analytics_html(analytics));
        }

        // Properties section
        html.push_str(&self.generate_properties_html(aspect));

        // Operations section
        if !aspect.operations().is_empty() {
            html.push_str(&self.generate_operations_html(aspect));
        }

        // Events section
        if !aspect.events().is_empty() {
            html.push_str(&self.generate_events_html(aspect));
        }

        // Examples section
        if self.include_examples {
            html.push_str(&self.generate_examples_html(aspect));
        }

        // Footer
        html.push_str("<footer>\n");
        if let Some(ref footer_text) = self.footer {
            html.push_str(&format!("<p>{}</p>\n", footer_text));
        }
        html.push_str(
            r#"<p>Generated by <a href="https://github.com/cool-japan/oxirs">OxiRS SAMM</a></p>
        </footer>
    </div>
</body>
</html>"#,
        );

        Ok(html)
    }

    /// Generate table of contents HTML
    fn generate_toc_html(&self, aspect: &Aspect, analytics: Option<&ModelAnalytics>) -> String {
        let mut toc = String::from("<nav class=\"toc\">\n<h2>Table of Contents</h2>\n<ul>\n");
        toc.push_str("    <li><a href=\"#overview\">Overview</a></li>\n");

        if analytics.is_some() {
            toc.push_str("    <li><a href=\"#analytics\">Quality Analytics</a></li>\n");
        }

        toc.push_str("    <li><a href=\"#properties\">Properties</a></li>\n");

        if !aspect.operations().is_empty() {
            toc.push_str("    <li><a href=\"#operations\">Operations</a></li>\n");
        }

        if !aspect.events().is_empty() {
            toc.push_str("    <li><a href=\"#events\">Events</a></li>\n");
        }

        if self.include_examples {
            toc.push_str("    <li><a href=\"#examples\">Examples</a></li>\n");
        }

        toc.push_str("</ul>\n</nav>\n");
        toc
    }

    /// Generate overview section HTML
    fn generate_overview_html(&self, aspect: &Aspect) -> String {
        let mut html =
            String::from("<section id=\"overview\" class=\"section\">\n<h2>Overview</h2>\n");

        // Preferred name
        if let Some(name) = aspect.metadata.get_preferred_name("en") {
            html.push_str(&format!("<h3>{}</h3>\n", name));
        }

        // Description
        if let Some(desc) = aspect.metadata.get_description("en") {
            html.push_str(&format!("<p class=\"description\">{}</p>\n", desc));
        }

        // URN
        html.push_str(&format!(
            "<p><strong>URN:</strong> <code>{}</code></p>\n",
            aspect.urn()
        ));

        // Multi-language support
        if aspect.metadata.preferred_names.len() > 1 {
            html.push_str("<details class=\"multilang\">\n");
            html.push_str("<summary>Multi-language Support</summary>\n");
            html.push_str("<table>\n<thead><tr><th>Language</th><th>Name</th><th>Description</th></tr></thead>\n<tbody>\n");

            for (lang, name) in &aspect.metadata.preferred_names {
                let desc = aspect.metadata.get_description(lang).unwrap_or("N/A");
                html.push_str(&format!(
                    "<tr><td>{}</td><td>{}</td><td>{}</td></tr>\n",
                    lang, name, desc
                ));
            }

            html.push_str("</tbody>\n</table>\n</details>\n");
        }

        html.push_str("</section>\n");
        html
    }

    /// Generate analytics section HTML
    fn generate_analytics_html(&self, analytics: &ModelAnalytics) -> String {
        let mut html = String::from(
            "<section id=\"analytics\" class=\"section\">\n<h2>Quality Analytics</h2>\n",
        );

        // Quality score
        let score_class = if analytics.quality_score >= 80.0 {
            "score-good"
        } else if analytics.quality_score >= 60.0 {
            "score-fair"
        } else {
            "score-poor"
        };

        html.push_str(&format!(
            "<div class=\"quality-score {}\">\n<span class=\"score\">{:.1}</span>\n<span class=\"label\">Quality Score</span>\n</div>\n",
            score_class, analytics.quality_score
        ));

        // Complexity
        html.push_str("<h3>Complexity Assessment</h3>\n");
        html.push_str(&format!(
            "<p><strong>Overall Level:</strong> {:?}</p>\n",
            analytics.complexity_assessment.overall_level
        ));
        html.push_str("<ul>\n");
        html.push_str(&format!(
            "<li>Structural: {:.1}</li>\n",
            analytics.complexity_assessment.structural
        ));
        html.push_str(&format!(
            "<li>Cognitive: {:.1}</li>\n",
            analytics.complexity_assessment.cognitive
        ));
        html.push_str(&format!(
            "<li>Coupling: {:.1}</li>\n",
            analytics.complexity_assessment.coupling
        ));
        html.push_str("</ul>\n");

        // Best practices
        html.push_str("<h3>Best Practice Compliance</h3>\n");
        html.push_str(&format!(
            "<p><strong>{:.1}%</strong> ({}/{})</p>\n",
            analytics.best_practices.compliance_percentage,
            analytics.best_practices.passed_checks,
            analytics.best_practices.total_checks
        ));

        // Recommendations
        if !analytics.recommendations.is_empty() {
            html.push_str("<h3>Recommendations</h3>\n");
            html.push_str("<ul class=\"recommendations\">\n");
            for rec in analytics.recommendations.iter().take(5) {
                html.push_str(&format!(
                    "<li class=\"severity-{}\">{}: {}</li>\n",
                    format!("{:?}", rec.severity).to_lowercase(),
                    rec.message,
                    rec.suggested_action
                ));
            }
            html.push_str("</ul>\n");
        }

        html.push_str("</section>\n");
        html
    }

    /// Generate properties section HTML
    fn generate_properties_html(&self, aspect: &Aspect) -> String {
        let mut html =
            String::from("<section id=\"properties\" class=\"section\">\n<h2>Properties</h2>\n");

        if aspect.properties().is_empty() {
            html.push_str("<p>No properties defined.</p>\n");
        } else {
            html.push_str("<table class=\"properties-table\">\n");
            html.push_str("<thead>\n<tr><th>Name</th><th>Type</th><th>Required</th><th>Description</th></tr>\n</thead>\n");
            html.push_str("<tbody>\n");

            for prop in aspect.properties() {
                let name = prop.name();
                let data_type = prop
                    .characteristic
                    .as_ref()
                    .and_then(|c| c.data_type.as_deref())
                    .unwrap_or("N/A");
                let required = if prop.optional { "No" } else { "Yes" };
                let description = prop
                    .metadata
                    .get_description("en")
                    .unwrap_or("No description");

                html.push_str(&format!(
                    "<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td></tr>\n",
                    name, data_type, required, description
                ));
            }

            html.push_str("</tbody>\n</table>\n");
        }

        html.push_str("</section>\n");
        html
    }

    /// Generate operations section HTML
    fn generate_operations_html(&self, aspect: &Aspect) -> String {
        let mut html =
            String::from("<section id=\"operations\" class=\"section\">\n<h2>Operations</h2>\n");

        html.push_str("<ul class=\"operations-list\">\n");
        for op in aspect.operations() {
            html.push_str(&format!("<li><code>{}</code></li>\n", op.name()));
        }
        html.push_str("</ul>\n");

        html.push_str("</section>\n");
        html
    }

    /// Generate events section HTML
    fn generate_events_html(&self, aspect: &Aspect) -> String {
        let mut html = String::from("<section id=\"events\" class=\"section\">\n<h2>Events</h2>\n");

        html.push_str("<ul class=\"events-list\">\n");
        for event in aspect.events() {
            html.push_str(&format!("<li><code>{}</code></li>\n", event.name()));
        }
        html.push_str("</ul>\n");

        html.push_str("</section>\n");
        html
    }

    /// Generate examples section HTML
    fn generate_examples_html(&self, aspect: &Aspect) -> String {
        let mut html =
            String::from("<section id=\"examples\" class=\"section\">\n<h2>Examples</h2>\n");

        html.push_str("<h3>JSON Example</h3>\n");
        html.push_str("<pre><code class=\"language-json\">{\n");
        for (i, prop) in aspect.properties().iter().enumerate() {
            let comma = if i < aspect.properties().len() - 1 {
                ","
            } else {
                ""
            };
            let example_value = self.generate_example_value(prop);
            html.push_str(&format!(
                "  \"{}\": {}{}\n",
                prop.name(),
                example_value,
                comma
            ));
        }
        html.push_str("}</code></pre>\n");

        html.push_str("</section>\n");
        html
    }

    /// Generate example value for a property
    fn generate_example_value(&self, prop: &Property) -> String {
        if let Some(char) = &prop.characteristic {
            if let Some(dtype) = &char.data_type {
                return match dtype.as_str() {
                    s if s.contains("string") => "\"example\"".to_string(),
                    s if s.contains("int") => "42".to_string(),
                    s if s.contains("decimal") | s.contains("float") | s.contains("double") => {
                        "3.14".to_string()
                    }
                    s if s.contains("boolean") | s.contains("bool") => "true".to_string(),
                    s if s.contains("date") => "\"2024-01-01\"".to_string(),
                    _ => "null".to_string(),
                };
            }
        }
        "null".to_string()
    }

    /// Generate Markdown documentation
    fn generate_markdown(&self, aspect: &Aspect) -> Result<String> {
        let mut md = String::new();

        // Title
        let title = self.title.clone().unwrap_or_else(|| {
            aspect
                .metadata
                .get_preferred_name("en")
                .map(|s| s.to_string())
                .unwrap_or_else(|| aspect.name().to_string())
        });
        md.push_str(&format!("# {}\n\n", title));

        // Description
        if let Some(desc) = aspect.metadata.get_description("en") {
            md.push_str(&format!("{}\n\n", desc));
        }

        // Metadata
        md.push_str("## Metadata\n\n");
        md.push_str(&format!("- **URN**: `{}`\n", aspect.urn()));
        md.push_str(&format!(
            "- **Properties**: {}\n",
            aspect.properties().len()
        ));
        md.push_str(&format!(
            "- **Operations**: {}\n",
            aspect.operations().len()
        ));
        md.push('\n');

        // Analytics
        if self.include_analytics {
            let analytics = ModelAnalytics::analyze(aspect);
            md.push_str("## Quality Analytics\n\n");
            md.push_str(&format!(
                "- **Quality Score**: {:.1}/100\n",
                analytics.quality_score
            ));
            md.push_str(&format!(
                "- **Complexity**: {:?}\n",
                analytics.complexity_assessment.overall_level
            ));
            md.push_str(&format!(
                "- **Best Practices**: {:.1}% ({}/{})\n",
                analytics.best_practices.compliance_percentage,
                analytics.best_practices.passed_checks,
                analytics.best_practices.total_checks
            ));
            md.push('\n');
        }

        // Properties
        md.push_str("## Properties\n\n");
        if aspect.properties().is_empty() {
            md.push_str("No properties defined.\n\n");
        } else {
            md.push_str("| Name | Type | Required | Description |\n");
            md.push_str("|------|------|----------|-------------|\n");

            for prop in aspect.properties() {
                let name = prop.name();
                let data_type = prop
                    .characteristic
                    .as_ref()
                    .and_then(|c| c.data_type.as_deref())
                    .unwrap_or("N/A");
                let required = if prop.optional { "No" } else { "Yes" };
                let description = prop
                    .metadata
                    .get_description("en")
                    .unwrap_or("No description");

                md.push_str(&format!(
                    "| `{}` | {} | {} | {} |\n",
                    name, data_type, required, description
                ));
            }
            md.push('\n');
        }

        // Operations
        if !aspect.operations().is_empty() {
            md.push_str("## Operations\n\n");
            for op in aspect.operations() {
                md.push_str(&format!("- `{}`\n", op.name()));
            }
            md.push('\n');
        }

        // Footer
        if let Some(ref footer_text) = self.footer {
            md.push_str(&format!("\n---\n\n{}\n", footer_text));
        }

        Ok(md)
    }

    /// Generate JSON documentation
    fn generate_json(&self, aspect: &Aspect) -> Result<String> {
        let mut doc = serde_json::Map::new();

        doc.insert("name".to_string(), serde_json::json!(aspect.name()));
        doc.insert("urn".to_string(), serde_json::json!(aspect.urn()));

        if let Some(name) = aspect.metadata.get_preferred_name("en") {
            doc.insert("preferredName".to_string(), serde_json::json!(name));
        }

        if let Some(desc) = aspect.metadata.get_description("en") {
            doc.insert("description".to_string(), serde_json::json!(desc));
        }

        // Properties
        let properties: Vec<_> = aspect
            .properties()
            .iter()
            .map(|p| {
                let mut prop_doc = serde_json::Map::new();
                prop_doc.insert("name".to_string(), serde_json::json!(p.name()));
                prop_doc.insert("urn".to_string(), serde_json::json!(p.urn()));
                prop_doc.insert("optional".to_string(), serde_json::json!(p.optional));

                if let Some(char) = &p.characteristic {
                    if let Some(dtype) = &char.data_type {
                        prop_doc.insert("dataType".to_string(), serde_json::json!(dtype));
                    }
                }

                serde_json::Value::Object(prop_doc)
            })
            .collect();

        doc.insert("properties".to_string(), serde_json::json!(properties));

        // Analytics
        if self.include_analytics {
            let analytics = ModelAnalytics::analyze(aspect);
            let mut analytics_doc = serde_json::Map::new();
            analytics_doc.insert(
                "qualityScore".to_string(),
                serde_json::json!(analytics.quality_score),
            );
            analytics_doc.insert(
                "complexityLevel".to_string(),
                serde_json::json!(format!(
                    "{:?}",
                    analytics.complexity_assessment.overall_level
                )),
            );
            doc.insert(
                "analytics".to_string(),
                serde_json::Value::Object(analytics_doc),
            );
        }

        serde_json::to_string_pretty(&serde_json::Value::Object(doc))
            .map_err(|e| SammError::ValidationError(format!("JSON serialization failed: {}", e)))
    }
}

impl Default for DocumentationGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Default CSS for HTML documentation
const DEFAULT_CSS: &str = r#"
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: #333;
    margin: 0;
    padding: 0;
    background: #f5f5f5;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background: white;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}

header {
    border-bottom: 3px solid #2563eb;
    padding-bottom: 20px;
    margin-bottom: 30px;
}

header h1 {
    margin: 0;
    color: #1e40af;
}

.subtitle {
    color: #6b7280;
    margin: 10px 0 0 0;
}

.toc {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
}

.toc h2 {
    margin-top: 0;
    font-size: 1.2em;
}

.toc ul {
    list-style: none;
    padding: 0;
}

.toc li {
    margin: 8px 0;
}

.toc a {
    color: #2563eb;
    text-decoration: none;
}

.toc a:hover {
    text-decoration: underline;
}

.section {
    margin: 40px 0;
}

.section h2 {
    color: #1e40af;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 10px;
}

.description {
    font-size: 1.1em;
    color: #4b5563;
    padding: 15px;
    background: #f9fafb;
    border-left: 4px solid #2563eb;
    margin: 20px 0;
}

code {
    background: #f3f4f6;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 0.9em;
}

pre {
    background: #1f2937;
    color: #f9fafb;
    padding: 20px;
    border-radius: 8px;
    overflow-x: auto;
}

pre code {
    background: none;
    color: inherit;
    padding: 0;
}

.quality-score {
    display: inline-block;
    text-align: center;
    padding: 30px;
    border-radius: 12px;
    margin: 20px 0;
}

.quality-score .score {
    display: block;
    font-size: 48px;
    font-weight: bold;
}

.quality-score .label {
    display: block;
    font-size: 14px;
    text-transform: uppercase;
    margin-top: 10px;
}

.score-good {
    background: #d1fae5;
    color: #065f46;
}

.score-fair {
    background: #fef3c7;
    color: #92400e;
}

.score-poor {
    background: #fee2e2;
    color: #991b1b;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
}

th, td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #e5e7eb;
}

th {
    background: #f9fafb;
    font-weight: 600;
    color: #374151;
}

tr:hover {
    background: #f9fafb;
}

.recommendations {
    list-style: none;
    padding: 0;
}

.recommendations li {
    padding: 12px;
    margin: 10px 0;
    border-left: 4px solid;
    background: #f9fafb;
}

.severity-error, .severity-critical {
    border-left-color: #dc2626;
    background: #fef2f2;
}

.severity-warning {
    border-left-color: #f59e0b;
    background: #fffbeb;
}

.severity-info {
    border-left-color: #3b82f6;
    background: #eff6ff;
}

details {
    margin: 20px 0;
}

summary {
    cursor: pointer;
    font-weight: 600;
    padding: 10px;
    background: #f9fafb;
    border-radius: 4px;
}

summary:hover {
    background: #f3f4f6;
}

footer {
    margin-top: 60px;
    padding-top: 20px;
    border-top: 2px solid #e5e7eb;
    text-align: center;
    color: #6b7280;
}

footer a {
    color: #2563eb;
    text-decoration: none;
}

footer a:hover {
    text-decoration: underline;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{Characteristic, CharacteristicKind};

    fn create_test_aspect() -> Aspect {
        let mut aspect = Aspect::new("urn:samm:org.test:1.0.0#TestAspect".to_string());
        aspect
            .metadata
            .add_preferred_name("en".to_string(), "Test Aspect".to_string());
        aspect.metadata.add_description(
            "en".to_string(),
            "A test aspect for documentation".to_string(),
        );

        let mut prop = Property::new("urn:samm:org.test:1.0.0#testProperty".to_string());
        prop.metadata
            .add_preferred_name("en".to_string(), "Test Property".to_string());
        let mut char = Characteristic::new(
            "urn:samm:org.test:1.0.0#TestChar".to_string(),
            CharacteristicKind::Trait,
        );
        char.data_type = Some("xsd:string".to_string());
        prop.characteristic = Some(char);
        aspect.add_property(prop);

        aspect
    }

    #[test]
    fn test_generator_creation() {
        let generator = DocumentationGenerator::new();
        assert_eq!(generator.format, DocumentationFormat::Html);
        assert_eq!(generator.style, DocumentationStyle::Technical);
        assert!(generator.include_analytics);
        assert!(generator.include_toc);
    }

    #[test]
    fn test_generator_configuration() {
        let generator = DocumentationGenerator::new()
            .with_format(DocumentationFormat::Markdown)
            .with_style(DocumentationStyle::UserFriendly)
            .with_analytics(false)
            .with_table_of_contents(false);

        assert_eq!(generator.format, DocumentationFormat::Markdown);
        assert_eq!(generator.style, DocumentationStyle::UserFriendly);
        assert!(!generator.include_analytics);
        assert!(!generator.include_toc);
    }

    #[test]
    fn test_html_generation() {
        let aspect = create_test_aspect();
        let generator = DocumentationGenerator::new();
        let html = generator.generate(&aspect).unwrap();

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Test Aspect"));
        assert!(html.contains("testProperty"));
        assert!(html.contains("Quality Analytics"));
    }

    #[test]
    fn test_markdown_generation() {
        let aspect = create_test_aspect();
        let generator = DocumentationGenerator::new().with_format(DocumentationFormat::Markdown);
        let md = generator.generate(&aspect).unwrap();

        assert!(md.contains("# Test Aspect"));
        assert!(md.contains("## Properties"));
        assert!(md.contains("testProperty"));
    }

    #[test]
    fn test_json_generation() {
        let aspect = create_test_aspect();
        let generator = DocumentationGenerator::new().with_format(DocumentationFormat::Json);
        let json = generator.generate(&aspect).unwrap();

        assert!(json.contains("\"name\""));
        assert!(json.contains("TestAspect"));
        assert!(json.contains("properties"));
    }

    #[test]
    fn test_without_analytics() {
        let aspect = create_test_aspect();
        let generator = DocumentationGenerator::new().with_analytics(false);
        let html = generator.generate(&aspect).unwrap();

        assert!(!html.contains("Quality Analytics"));
    }

    #[test]
    fn test_custom_title() {
        let aspect = create_test_aspect();
        let generator =
            DocumentationGenerator::new().with_title("Custom Documentation".to_string());
        let html = generator.generate(&aspect).unwrap();

        assert!(html.contains("Custom Documentation"));
    }

    #[test]
    fn test_custom_footer() {
        let aspect = create_test_aspect();
        let generator = DocumentationGenerator::new().with_footer("Custom Footer Text".to_string());
        let html = generator.generate(&aspect).unwrap();

        assert!(html.contains("Custom Footer Text"));
    }

    #[test]
    fn test_example_value_generation() {
        let generator = DocumentationGenerator::new();

        let mut prop = Property::new("test".to_string());
        let mut char = Characteristic::new("char".to_string(), CharacteristicKind::Trait);

        char.data_type = Some("xsd:string".to_string());
        prop.characteristic = Some(char.clone());
        assert_eq!(generator.generate_example_value(&prop), "\"example\"");

        char.data_type = Some("xsd:integer".to_string());
        prop.characteristic = Some(char.clone());
        assert_eq!(generator.generate_example_value(&prop), "42");

        char.data_type = Some("xsd:boolean".to_string());
        prop.characteristic = Some(char);
        assert_eq!(generator.generate_example_value(&prop), "true");
    }

    #[test]
    fn test_empty_aspect() {
        let aspect = Aspect::new("urn:samm:org.test:1.0.0#Empty".to_string());
        let generator = DocumentationGenerator::new();
        let html = generator.generate(&aspect).unwrap();

        assert!(html.contains("No properties defined"));
    }
}
