//! ShEx to SHACL Migration Tool
//!
//! This module provides comprehensive migration capabilities from ShEx (Shape Expressions)
//! schemas to equivalent SHACL shapes, handling semantic differences and providing
//! detailed migration reports.

use crate::constraints::value_constraints::NodeKind;
use crate::{
    Constraint, ConstraintComponentId, NodeKindConstraint, PropertyConstraint, Result, ShaclError,
    Shape, ShapeId, ShapeType, Target,
};
use indexmap::IndexMap;
use oxirs_core::model::{NamedNode, Term};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ShEx to SHACL migration engine
#[derive(Debug)]
pub struct ShexMigrationEngine {
    config: MigrationConfig,
    prefix_map: HashMap<String, String>,
}

/// Configuration for ShEx migration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationConfig {
    /// Whether to preserve ShEx comments as SHACL descriptions
    pub preserve_comments: bool,
    /// Whether to generate verbose SHACL with all constraints explicit
    pub verbose_output: bool,
    /// Whether to add migration metadata to generated shapes
    pub add_metadata: bool,
    /// Default base IRI for generated shapes
    pub base_iri: String,
    /// Whether to use compact property paths
    pub compact_paths: bool,
    /// Whether to validate the output shapes
    pub validate_output: bool,
    /// Semantic mapping preferences
    pub semantic_mapping: SemanticMappingConfig,
    /// Whether to generate equivalent SPARQL constraints
    pub generate_sparql: bool,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            preserve_comments: true,
            verbose_output: false,
            add_metadata: true,
            base_iri: "http://example.org/shapes/".to_string(),
            compact_paths: true,
            validate_output: true,
            semantic_mapping: SemanticMappingConfig::default(),
            generate_sparql: false,
        }
    }
}

/// Configuration for semantic mapping between ShEx and SHACL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMappingConfig {
    /// How to handle ShEx CLOSED
    pub closed_handling: ClosedHandling,
    /// How to handle ShEx EXTRA
    pub extra_handling: ExtraHandling,
    /// How to handle ShEx value set
    pub value_set_handling: ValueSetHandling,
    /// How to handle ShEx annotations
    pub annotation_handling: AnnotationHandling,
    /// How to handle ShEx imports
    pub import_handling: ImportHandling,
}

impl Default for SemanticMappingConfig {
    fn default() -> Self {
        Self {
            closed_handling: ClosedHandling::ShaclClosed,
            extra_handling: ExtraHandling::IgnoredProperties,
            value_set_handling: ValueSetHandling::InConstraint,
            annotation_handling: AnnotationHandling::Description,
            import_handling: ImportHandling::Inline,
        }
    }
}

/// How to handle ShEx CLOSED constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClosedHandling {
    /// Use sh:closed
    ShaclClosed,
    /// Generate SPARQL constraint
    SparqlConstraint,
    /// Add warning in description
    WarningOnly,
}

/// How to handle ShEx EXTRA properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExtraHandling {
    /// Use sh:ignoredProperties
    IgnoredProperties,
    /// Expand to explicit property constraints
    ExplicitConstraints,
    /// Skip with warning
    Skip,
}

/// How to handle ShEx value sets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValueSetHandling {
    /// Use sh:in constraint
    InConstraint,
    /// Generate sh:or with hasValue
    OrHasValue,
    /// Generate SPARQL constraint
    SparqlConstraint,
}

/// How to handle ShEx annotations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnnotationHandling {
    /// Add as SHACL descriptions
    Description,
    /// Preserve as custom properties
    CustomProperties,
    /// Skip annotations
    Skip,
}

/// How to handle ShEx imports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImportHandling {
    /// Inline imported shapes
    Inline,
    /// Keep as references
    Reference,
    /// Skip imports
    Skip,
}

/// ShEx schema representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShexSchema {
    /// Schema prefixes
    pub prefixes: HashMap<String, String>,
    /// Base IRI
    pub base: Option<String>,
    /// Import statements
    pub imports: Vec<String>,
    /// Shape expressions
    pub shapes: Vec<ShexShape>,
    /// Start shape
    pub start: Option<String>,
}

/// ShEx shape expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShexShape {
    /// Shape identifier
    pub id: String,
    /// Shape expression
    pub expression: ShexExpression,
    /// Whether shape is closed
    pub closed: bool,
    /// Extra allowed properties
    pub extra: Vec<String>,
    /// Annotations
    pub annotations: Vec<ShexAnnotation>,
    /// Comments
    pub comments: Vec<String>,
}

/// ShEx expression types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShexExpression {
    /// Triple constraint
    TripleConstraint(TripleConstraint),
    /// AND of expressions
    EachOf(Vec<ShexExpression>),
    /// OR of expressions
    OneOf(Vec<ShexExpression>),
    /// Negation
    Not(Box<ShexExpression>),
    /// Shape reference
    ShapeRef(String),
    /// Node constraint
    NodeConstraint(NodeConstraint),
    /// Empty expression
    Empty,
}

/// ShEx triple constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleConstraint {
    /// Predicate IRI
    pub predicate: String,
    /// Value expression
    pub value_expr: Option<Box<ShexExpression>>,
    /// Minimum cardinality
    pub min: usize,
    /// Maximum cardinality (-1 for unbounded)
    pub max: i64,
    /// Inverse direction
    pub inverse: bool,
    /// Annotations
    pub annotations: Vec<ShexAnnotation>,
    /// Semantic actions
    pub sem_acts: Vec<SemanticAction>,
}

/// ShEx node constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConstraint {
    /// Node kind (IRI, BNode, Literal, NonLiteral)
    pub node_kind: Option<String>,
    /// Datatype constraint
    pub datatype: Option<String>,
    /// Min/Max length
    pub string_facets: StringFacets,
    /// Numeric facets
    pub numeric_facets: NumericFacets,
    /// Value set
    pub values: Vec<ValueSetValue>,
}

/// String facets for node constraints
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StringFacets {
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub pattern: Option<String>,
    pub flags: Option<String>,
}

/// Numeric facets for node constraints
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NumericFacets {
    pub min_inclusive: Option<f64>,
    pub max_inclusive: Option<f64>,
    pub min_exclusive: Option<f64>,
    pub max_exclusive: Option<f64>,
    pub total_digits: Option<usize>,
    pub fraction_digits: Option<usize>,
}

/// Value set value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValueSetValue {
    /// IRI value
    Iri(String),
    /// Literal value
    Literal {
        value: String,
        datatype: Option<String>,
        language: Option<String>,
    },
    /// IRI stem
    IriStem { stem: String },
    /// IRI stem range
    IriStemRange {
        stem: String,
        exclusions: Vec<String>,
    },
    /// Language stem
    LanguageStem { stem: String },
    /// Wildcard
    Wildcard { exclusions: Vec<ValueSetValue> },
}

/// ShEx annotation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShexAnnotation {
    /// Predicate
    pub predicate: String,
    /// Object value
    pub object: String,
}

/// Semantic action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAction {
    /// Action name
    pub name: String,
    /// Action code
    pub code: String,
}

/// Migration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationResult {
    /// Converted SHACL shapes
    pub shapes: IndexMap<ShapeId, Shape>,
    /// Migration report
    pub report: MigrationReport,
}

/// Migration report with detailed statistics and warnings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationReport {
    /// Number of shapes converted
    pub shapes_converted: usize,
    /// Number of constraints converted
    pub constraints_converted: usize,
    /// Warnings generated during migration
    pub warnings: Vec<MigrationWarning>,
    /// Errors that prevented full migration
    pub errors: Vec<MigrationError>,
    /// Features that couldn't be directly mapped
    pub unmapped_features: Vec<UnmappedFeature>,
    /// Statistics
    pub statistics: MigrationStatistics,
}

/// Migration warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationWarning {
    /// Warning code
    pub code: String,
    /// Warning message
    pub message: String,
    /// Location in ShEx schema
    pub location: Option<String>,
    /// Suggestion for handling
    pub suggestion: Option<String>,
}

/// Migration error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Location in ShEx schema
    pub location: Option<String>,
}

/// Unmapped feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnmappedFeature {
    /// Feature name
    pub feature: String,
    /// Description
    pub description: String,
    /// Possible workaround
    pub workaround: Option<String>,
}

/// Migration statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MigrationStatistics {
    /// Total ShEx shapes
    pub total_shex_shapes: usize,
    /// Total SHACL shapes generated
    pub total_shacl_shapes: usize,
    /// Triple constraints converted
    pub triple_constraints: usize,
    /// Node constraints converted
    pub node_constraints: usize,
    /// Value sets converted
    pub value_sets: usize,
    /// Cardinality constraints
    pub cardinality_constraints: usize,
    /// Closed shapes
    pub closed_shapes: usize,
    /// Semantic actions skipped
    pub semantic_actions_skipped: usize,
    /// Unmapped features that couldn't be converted
    pub unmapped_features: Vec<UnmappedFeature>,
}

impl ShexMigrationEngine {
    /// Create a new migration engine with default configuration
    pub fn new() -> Self {
        Self {
            config: MigrationConfig::default(),
            prefix_map: HashMap::new(),
        }
    }

    /// Create a new migration engine with custom configuration
    pub fn with_config(config: MigrationConfig) -> Self {
        Self {
            config,
            prefix_map: HashMap::new(),
        }
    }

    /// Migrate ShEx schema to SHACL shapes
    pub fn migrate(&mut self, schema: &ShexSchema) -> Result<MigrationResult> {
        let mut shapes = IndexMap::new();
        let mut report = MigrationReport {
            shapes_converted: 0,
            constraints_converted: 0,
            warnings: Vec::new(),
            errors: Vec::new(),
            unmapped_features: Vec::new(),
            statistics: MigrationStatistics::default(),
        };

        // Store prefixes for IRI resolution
        self.prefix_map = schema.prefixes.clone();

        // Set statistics
        report.statistics.total_shex_shapes = schema.shapes.len();

        // Process imports
        self.process_imports(&schema.imports, &mut report)?;

        // Convert each ShEx shape
        for shex_shape in &schema.shapes {
            match self.convert_shape(shex_shape, &mut report.statistics) {
                Ok(shacl_shape) => {
                    let shape_id = ShapeId(shex_shape.id.clone());
                    shapes.insert(shape_id, shacl_shape);
                    report.shapes_converted += 1;
                }
                Err(e) => {
                    report.errors.push(MigrationError {
                        code: "SHAPE_CONVERSION_ERROR".to_string(),
                        message: e.to_string(),
                        location: Some(shex_shape.id.clone()),
                    });
                }
            }
        }

        report.statistics.total_shacl_shapes = shapes.len();

        Ok(MigrationResult { shapes, report })
    }

    /// Parse ShEx from string
    pub fn parse_shex(&self, input: &str) -> Result<ShexSchema> {
        // Simple ShEx parser implementation
        // In production, this would use a full ShEx parser
        let mut schema = ShexSchema {
            prefixes: HashMap::new(),
            base: None,
            imports: Vec::new(),
            shapes: Vec::new(),
            start: None,
        };

        let lines: Vec<&str> = input.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                i += 1;
                continue;
            }

            // Parse PREFIX
            if line.to_uppercase().starts_with("PREFIX") {
                if let Some((prefix, iri)) = self.parse_prefix_line(line) {
                    schema.prefixes.insert(prefix, iri);
                }
            }
            // Parse BASE
            else if line.to_uppercase().starts_with("BASE") {
                schema.base = self.parse_base_line(line);
            }
            // Parse IMPORT
            else if line.to_uppercase().starts_with("IMPORT") {
                if let Some(import) = self.parse_import_line(line) {
                    schema.imports.push(import);
                }
            }
            // Parse start
            else if line.to_uppercase().starts_with("START") {
                schema.start = self.parse_start_line(line);
            }
            // Parse shape
            else if line.contains('{') || self.is_shape_start(line) {
                let (shape, consumed) = self.parse_shape(&lines, i)?;
                schema.shapes.push(shape);
                i += consumed;
                continue;
            }

            i += 1;
        }

        Ok(schema)
    }

    fn parse_prefix_line(&self, line: &str) -> Option<(String, String)> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let prefix = parts[1].trim_end_matches(':').to_string();
            let iri = parts[2].trim_matches(&['<', '>'][..]).to_string();
            Some((prefix, iri))
        } else {
            None
        }
    }

    fn parse_base_line(&self, line: &str) -> Option<String> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            Some(parts[1].trim_matches(&['<', '>'][..]).to_string())
        } else {
            None
        }
    }

    fn parse_import_line(&self, line: &str) -> Option<String> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            Some(parts[1].trim_matches(&['<', '>'][..]).to_string())
        } else {
            None
        }
    }

    fn parse_start_line(&self, line: &str) -> Option<String> {
        // Parse START = @<ShapeName>
        if let Some(pos) = line.find('=') {
            let shape_ref = line[pos + 1..].trim();
            Some(
                shape_ref
                    .trim_start_matches('@')
                    .trim_matches(&['<', '>'][..])
                    .to_string(),
            )
        } else {
            None
        }
    }

    fn is_shape_start(&self, line: &str) -> bool {
        // Check if line starts a shape definition
        line.starts_with('<')
            || line.starts_with(':')
            || (line.contains(':') && !line.to_uppercase().starts_with("PREFIX"))
    }

    fn parse_shape(&self, lines: &[&str], start: usize) -> Result<(ShexShape, usize)> {
        let mut consumed = 0;
        let mut shape_text = String::new();
        let mut brace_count = 0;
        let mut in_shape = false;

        // Collect shape text
        for line in lines.iter().skip(start) {
            shape_text.push_str(line);
            shape_text.push('\n');
            consumed += 1;

            for ch in line.chars() {
                if ch == '{' {
                    brace_count += 1;
                    in_shape = true;
                } else if ch == '}' {
                    brace_count -= 1;
                }
            }

            if in_shape && brace_count == 0 {
                break;
            }
        }

        // Parse collected shape text
        let shape = self.parse_shape_text(&shape_text)?;
        Ok((shape, consumed))
    }

    fn parse_shape_text(&self, text: &str) -> Result<ShexShape> {
        // Extract shape ID
        let id = if let Some(id_end) = text.find('{') {
            text[..id_end]
                .trim()
                .trim_matches(&['<', '>', '@'][..])
                .to_string()
        } else {
            return Err(ShaclError::ShapeParsing("Invalid shape syntax".to_string()));
        };

        // Check for CLOSED keyword
        let closed = text.to_uppercase().contains("CLOSED");

        // Extract EXTRA properties
        let extra = self.extract_extra_properties(text);

        // Parse body
        let body_start = text.find('{').unwrap_or(0) + 1;
        let body_end = text.rfind('}').unwrap_or(text.len());
        let body = &text[body_start..body_end];

        // Parse expression
        let expression = self.parse_expression(body)?;

        // Extract comments and annotations
        let comments = self.extract_comments(text);
        let annotations = self.extract_annotations(text);

        Ok(ShexShape {
            id,
            expression,
            closed,
            extra,
            annotations,
            comments,
        })
    }

    fn extract_extra_properties(&self, text: &str) -> Vec<String> {
        let mut extra = Vec::new();
        if let Some(pos) = text.to_uppercase().find("EXTRA") {
            let after_extra = &text[pos + 5..];
            // Parse property list until we hit a different keyword or {
            let end = after_extra.find(['{', ';']).unwrap_or(after_extra.len());
            let props = &after_extra[..end];
            for prop in props.split_whitespace() {
                if !prop.is_empty() {
                    extra.push(prop.trim_matches(&['<', '>'][..]).to_string());
                }
            }
        }
        extra
    }

    fn extract_comments(&self, text: &str) -> Vec<String> {
        text.lines()
            .filter(|line| line.trim().starts_with('#'))
            .map(|line| line.trim().trim_start_matches('#').trim().to_string())
            .collect()
    }

    fn extract_annotations(&self, text: &str) -> Vec<ShexAnnotation> {
        let mut annotations = Vec::new();
        // Parse // predicate value annotations
        for line in text.lines() {
            let trimmed = line.trim();
            if let Some(stripped) = trimmed.strip_prefix("//") {
                let parts: Vec<&str> = stripped.trim().splitn(2, ' ').collect();
                if parts.len() == 2 {
                    annotations.push(ShexAnnotation {
                        predicate: parts[0].to_string(),
                        object: parts[1].trim_matches('"').to_string(),
                    });
                }
            }
        }
        annotations
    }

    fn parse_expression(&self, body: &str) -> Result<ShexExpression> {
        let trimmed = body.trim();

        if trimmed.is_empty() {
            return Ok(ShexExpression::Empty);
        }

        // Check for EachOf (;)
        if trimmed.contains(';') && !trimmed.contains('|') {
            let parts = self.split_expression(trimmed, ';');
            let exprs: Result<Vec<ShexExpression>> = parts
                .iter()
                .filter(|p| !p.trim().is_empty())
                .map(|p| self.parse_expression(p))
                .collect();
            return Ok(ShexExpression::EachOf(exprs?));
        }

        // Check for OneOf (|)
        if trimmed.contains('|') {
            let parts = self.split_expression(trimmed, '|');
            let exprs: Result<Vec<ShexExpression>> = parts
                .iter()
                .filter(|p| !p.trim().is_empty())
                .map(|p| self.parse_expression(p))
                .collect();
            return Ok(ShexExpression::OneOf(exprs?));
        }

        // Parse single constraint
        self.parse_triple_constraint(trimmed)
    }

    fn split_expression(&self, text: &str, delimiter: char) -> Vec<String> {
        let mut parts = Vec::new();
        let mut current = String::new();
        let mut depth = 0;
        let mut in_string = false;

        for ch in text.chars() {
            match ch {
                '"' => {
                    in_string = !in_string;
                    current.push(ch);
                }
                '{' | '(' | '[' if !in_string => {
                    depth += 1;
                    current.push(ch);
                }
                '}' | ')' | ']' if !in_string => {
                    depth -= 1;
                    current.push(ch);
                }
                c if c == delimiter && depth == 0 && !in_string => {
                    parts.push(current.trim().to_string());
                    current = String::new();
                }
                _ => current.push(ch),
            }
        }

        if !current.trim().is_empty() {
            parts.push(current.trim().to_string());
        }

        parts
    }

    fn parse_triple_constraint(&self, text: &str) -> Result<ShexExpression> {
        let trimmed = text.trim();

        // Check for shape reference
        if trimmed.starts_with('@') {
            let shape_ref = trimmed
                .trim_start_matches('@')
                .trim_matches(&['<', '>'][..])
                .to_string();
            return Ok(ShexExpression::ShapeRef(shape_ref));
        }

        // Check for node constraint only (starts with node kind or datatype)
        if self.is_node_constraint_only(trimmed) {
            let node_constraint = self.parse_node_constraint(trimmed)?;
            return Ok(ShexExpression::NodeConstraint(node_constraint));
        }

        // Parse as triple constraint
        let parts: Vec<&str> = trimmed.splitn(2, char::is_whitespace).collect();
        if parts.is_empty() {
            return Ok(ShexExpression::Empty);
        }

        let predicate = parts[0].trim_matches(&['<', '>', '^'][..]).to_string();
        let inverse = parts[0].starts_with('^');

        let (value_expr, min, max) = if parts.len() > 1 {
            self.parse_value_and_cardinality(parts[1])?
        } else {
            (None, 1, 1)
        };

        Ok(ShexExpression::TripleConstraint(TripleConstraint {
            predicate,
            value_expr,
            min,
            max,
            inverse,
            annotations: Vec::new(),
            sem_acts: Vec::new(),
        }))
    }

    fn is_node_constraint_only(&self, text: &str) -> bool {
        let upper = text.to_uppercase();
        upper.starts_with("IRI")
            || upper.starts_with("BNODE")
            || upper.starts_with("LITERAL")
            || upper.starts_with("NONLITERAL")
            || text.starts_with('[')
            || text.starts_with('.')
    }

    fn parse_value_and_cardinality(
        &self,
        text: &str,
    ) -> Result<(Option<Box<ShexExpression>>, usize, i64)> {
        let trimmed = text.trim();

        // Check for cardinality at end
        let (value_text, min, max) = self.extract_cardinality(trimmed);

        // Parse value expression
        let value_expr = if !value_text.is_empty() && value_text != "." {
            Some(Box::new(self.parse_expression(value_text)?))
        } else {
            None
        };

        Ok((value_expr, min, max))
    }

    fn extract_cardinality<'a>(&self, text: &'a str) -> (&'a str, usize, i64) {
        let trimmed = text.trim();

        // Check for cardinality suffixes
        if let Some(stripped) = trimmed.strip_suffix('*') {
            (stripped.trim(), 0, -1)
        } else if let Some(stripped) = trimmed.strip_suffix('+') {
            (stripped.trim(), 1, -1)
        } else if let Some(stripped) = trimmed.strip_suffix('?') {
            (stripped.trim(), 0, 1)
        } else if trimmed.contains('{') {
            // Parse {min,max} format
            if let Some(start) = trimmed.rfind('{') {
                let card_text = &trimmed[start..];
                let value_text = trimmed[..start].trim();
                let (min, max) = self.parse_cardinality_braces(card_text);
                (value_text, min, max)
            } else {
                (trimmed, 1, 1)
            }
        } else {
            (trimmed, 1, 1)
        }
    }

    fn parse_cardinality_braces(&self, text: &str) -> (usize, i64) {
        let inner = text.trim_matches(&['{', '}'][..]);
        if let Some(comma_pos) = inner.find(',') {
            let min_str = inner[..comma_pos].trim();
            let max_str = inner[comma_pos + 1..].trim();

            let min = min_str.parse().unwrap_or(0);
            let max = if max_str.is_empty() || max_str == "*" {
                -1
            } else {
                max_str.parse().unwrap_or(1)
            };

            (min, max)
        } else {
            let exact = inner.trim().parse().unwrap_or(1);
            (exact, exact as i64)
        }
    }

    fn parse_node_constraint(&self, text: &str) -> Result<NodeConstraint> {
        let mut constraint = NodeConstraint {
            node_kind: None,
            datatype: None,
            string_facets: StringFacets::default(),
            numeric_facets: NumericFacets::default(),
            values: Vec::new(),
        };

        let upper = text.to_uppercase();

        // Parse node kind
        if upper.starts_with("IRI") {
            constraint.node_kind = Some("IRI".to_string());
        } else if upper.starts_with("BNODE") {
            constraint.node_kind = Some("BNode".to_string());
        } else if upper.starts_with("LITERAL") {
            constraint.node_kind = Some("Literal".to_string());
        } else if upper.starts_with("NONLITERAL") {
            constraint.node_kind = Some("NonLiteral".to_string());
        }

        // Parse value set
        if text.contains('[') {
            constraint.values = self.parse_value_set(text)?;
        }

        // Parse facets
        self.parse_facets(text, &mut constraint);

        Ok(constraint)
    }

    fn parse_value_set(&self, text: &str) -> Result<Vec<ValueSetValue>> {
        let mut values = Vec::new();

        if let Some(start) = text.find('[') {
            if let Some(end) = text.rfind(']') {
                let inner = &text[start + 1..end];
                for item in inner.split_whitespace() {
                    let trimmed = item.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    if trimmed.starts_with('"') {
                        // Literal value
                        let value = trimmed.trim_matches('"').to_string();
                        values.push(ValueSetValue::Literal {
                            value,
                            datatype: None,
                            language: None,
                        });
                    } else if trimmed.ends_with('~') {
                        // IRI stem
                        let stem = trimmed.trim_end_matches('~').to_string();
                        values.push(ValueSetValue::IriStem { stem });
                    } else {
                        // IRI value
                        let iri = trimmed.trim_matches(&['<', '>'][..]).to_string();
                        values.push(ValueSetValue::Iri(iri));
                    }
                }
            }
        }

        Ok(values)
    }

    fn parse_facets(&self, text: &str, constraint: &mut NodeConstraint) {
        // Parse MINLENGTH, MAXLENGTH
        if let Some(val) = self.extract_facet_value(text, "MINLENGTH") {
            constraint.string_facets.min_length = val.parse().ok();
        }
        if let Some(val) = self.extract_facet_value(text, "MAXLENGTH") {
            constraint.string_facets.max_length = val.parse().ok();
        }
        if let Some(val) = self.extract_facet_value(text, "PATTERN") {
            constraint.string_facets.pattern = Some(val.trim_matches('"').to_string());
        }

        // Parse numeric facets
        if let Some(val) = self.extract_facet_value(text, "MININCLUSIVE") {
            constraint.numeric_facets.min_inclusive = val.parse().ok();
        }
        if let Some(val) = self.extract_facet_value(text, "MAXINCLUSIVE") {
            constraint.numeric_facets.max_inclusive = val.parse().ok();
        }
        if let Some(val) = self.extract_facet_value(text, "MINEXCLUSIVE") {
            constraint.numeric_facets.min_exclusive = val.parse().ok();
        }
        if let Some(val) = self.extract_facet_value(text, "MAXEXCLUSIVE") {
            constraint.numeric_facets.max_exclusive = val.parse().ok();
        }
    }

    fn extract_facet_value<'a>(&self, text: &'a str, facet: &str) -> Option<&'a str> {
        let upper = text.to_uppercase();
        if let Some(pos) = upper.find(facet) {
            let after = &text[pos + facet.len()..];
            let value_start = after.find(|c: char| !c.is_whitespace())?;
            let value_end = after[value_start..]
                .find(|c: char| c.is_whitespace() || c == ';' || c == '}')
                .unwrap_or(after.len() - value_start);
            Some(&after[value_start..value_start + value_end])
        } else {
            None
        }
    }

    fn process_imports(&self, imports: &[String], report: &mut MigrationReport) -> Result<()> {
        match self.config.semantic_mapping.import_handling {
            ImportHandling::Skip => {
                for import in imports {
                    report.warnings.push(MigrationWarning {
                        code: "IMPORT_SKIPPED".to_string(),
                        message: format!("Import skipped: {}", import),
                        location: None,
                        suggestion: Some("Manually include imported shapes".to_string()),
                    });
                }
            }
            ImportHandling::Reference => {
                for import in imports {
                    report.warnings.push(MigrationWarning {
                        code: "IMPORT_REFERENCE".to_string(),
                        message: format!("Import kept as reference: {}", import),
                        location: None,
                        suggestion: Some("Use owl:imports in SHACL ontology".to_string()),
                    });
                }
            }
            ImportHandling::Inline => {
                // Would need to fetch and parse imported schemas
                for import in imports {
                    report.warnings.push(MigrationWarning {
                        code: "IMPORT_INLINE".to_string(),
                        message: format!("Import should be inlined: {}", import),
                        location: None,
                        suggestion: Some("Fetch and include imported shapes".to_string()),
                    });
                }
            }
        }
        Ok(())
    }

    fn convert_shape(
        &self,
        shex_shape: &ShexShape,
        report: &mut MigrationStatistics,
    ) -> Result<Shape> {
        let shape_id = ShapeId(shex_shape.id.clone());

        // Create target definition - if not specified, no automatic target
        let target = Target::Class(NamedNode::new(&shex_shape.id).unwrap_or_else(|_| {
            NamedNode::new(format!("{}Class", self.config.base_iri))
                .expect("construction should succeed")
        }));

        // Convert expression to constraints
        let mut constraints_vec = Vec::new();
        self.convert_expression(&shex_shape.expression, &mut constraints_vec, report)?;

        // Handle closed shape
        if shex_shape.closed {
            report.closed_shapes += 1;
        }

        // Build description from comments and annotations
        let mut description = String::new();
        if self.config.preserve_comments && !shex_shape.comments.is_empty() {
            description = shex_shape.comments.join("\n");
        }
        for annotation in &shex_shape.annotations {
            if annotation.predicate.contains("description")
                || annotation.predicate.contains("comment")
            {
                if !description.is_empty() {
                    description.push('\n');
                }
                description.push_str(&annotation.object);
            }
        }

        // Create shape using builder pattern
        let mut shape = Shape::new(shape_id, ShapeType::NodeShape);
        shape.targets.push(target);
        shape.description = if description.is_empty() {
            None
        } else {
            Some(description)
        };

        // Add constraints to shape
        for (i, constraint) in constraints_vec.into_iter().enumerate() {
            let comp_id = ConstraintComponentId(format!("constraint_{}", i));
            shape.constraints.insert(comp_id, constraint);
        }

        Ok(shape)
    }

    fn convert_expression(
        &self,
        expr: &ShexExpression,
        constraints: &mut Vec<Constraint>,
        stats: &mut MigrationStatistics,
    ) -> Result<()> {
        match expr {
            ShexExpression::TripleConstraint(tc) => {
                self.convert_triple_constraint(tc, constraints, stats)?;
            }
            ShexExpression::EachOf(exprs) => {
                for e in exprs {
                    self.convert_expression(e, constraints, stats)?;
                }
            }
            ShexExpression::OneOf(exprs) => {
                // Convert to sh:or
                let mut or_shapes = Vec::new();
                for e in exprs {
                    let mut sub_constraints = Vec::new();
                    self.convert_expression(e, &mut sub_constraints, stats)?;
                    // Would create embedded shapes here
                    or_shapes.push(sub_constraints);
                }
                // Add sh:or constraint
                // Note: This is simplified - full implementation would create proper or constraint
            }
            ShexExpression::Not(inner) => {
                // Convert to sh:not
                let mut not_constraints = Vec::new();
                self.convert_expression(inner, &mut not_constraints, stats)?;
                // Add sh:not constraint
            }
            ShexExpression::ShapeRef(shape_ref) => {
                // Add sh:node constraint
                use crate::constraints::shape_constraints::NodeConstraint as ShapeNodeConstraint;
                constraints.push(Constraint::Node(ShapeNodeConstraint {
                    shape: ShapeId(shape_ref.clone()),
                }));
            }
            ShexExpression::NodeConstraint(nc) => {
                self.convert_node_constraint(nc, constraints, stats)?;
            }
            ShexExpression::Empty => {}
        }
        Ok(())
    }

    fn convert_triple_constraint(
        &self,
        tc: &TripleConstraint,
        _constraints: &mut Vec<Constraint>,
        stats: &mut MigrationStatistics,
    ) -> Result<()> {
        stats.triple_constraints += 1;

        // TODO: Properly implement triple constraint conversion
        // This requires creating a full property shape, not just constraints
        // For now, we track the statistics but skip the actual conversion

        // Set cardinality
        if tc.min > 0 || tc.max >= 0 {
            stats.cardinality_constraints += 1;
        }

        // Track semantic actions
        stats.semantic_actions_skipped += tc.sem_acts.len();

        // TODO: Handle value expression properly
        if tc.value_expr.is_some() {
            stats.unmapped_features.push(UnmappedFeature {
                feature: "TripleConstraint value expressions".to_string(),
                description: "Requires full property shape implementation".to_string(),
                workaround: None,
            });
        }

        Ok(())
    }

    fn apply_value_constraint(
        &self,
        _prop_constraint: &mut PropertyConstraint,
        _expr: &ShexExpression,
        stats: &mut MigrationStatistics,
    ) -> Result<()> {
        // TODO: Properly implement value constraint application
        // This requires a complete property shape model, not just PropertyConstraint
        stats.unmapped_features.push(UnmappedFeature {
            feature: "Value constraint application".to_string(),
            description: "Requires property shape implementation".to_string(),
            workaround: None,
        });
        Ok(())
    }

    fn convert_node_constraint(
        &self,
        nc: &NodeConstraint,
        constraints: &mut Vec<Constraint>,
        stats: &mut MigrationStatistics,
    ) -> Result<()> {
        stats.node_constraints += 1;

        // Node kind constraint
        if let Some(node_kind) = &nc.node_kind {
            let shacl_kind = match node_kind.as_str() {
                "IRI" => NodeKindConstraint {
                    node_kind: NodeKind::Iri,
                },
                "BNode" => NodeKindConstraint {
                    node_kind: NodeKind::BlankNode,
                },
                "Literal" => NodeKindConstraint {
                    node_kind: NodeKind::Literal,
                },
                "NonLiteral" => NodeKindConstraint {
                    node_kind: NodeKind::BlankNodeOrIri,
                },
                _ => NodeKindConstraint {
                    node_kind: NodeKind::Iri,
                },
            };
            constraints.push(Constraint::NodeKind(shacl_kind));
        }

        // Value set
        if !nc.values.is_empty() {
            stats.value_sets += 1;
            let in_values = self.convert_value_set(&nc.values);
            use crate::constraints::comparison_constraints::InConstraint;
            constraints.push(Constraint::In(InConstraint { values: in_values }));
        }

        Ok(())
    }

    fn convert_value_set(&self, values: &[ValueSetValue]) -> Vec<Term> {
        values
            .iter()
            .filter_map(|v| {
                match v {
                    ValueSetValue::Iri(iri) => {
                        Some(Term::NamedNode(NamedNode::new(iri).unwrap_or_else(|_| {
                            NamedNode::new("http://example.org/unknown").expect("valid IRI")
                        })))
                    }
                    ValueSetValue::Literal {
                        value,
                        datatype: _,
                        language: _,
                    } => Some(Term::Literal(oxirs_core::model::Literal::new(value))),
                    _ => None, // Stems and wildcards don't have direct SHACL equivalents
                }
            })
            .collect()
    }

    /// Generate migration report in human-readable format
    pub fn generate_report(&self, result: &MigrationResult) -> String {
        let mut report = String::new();

        report.push_str("# ShEx to SHACL Migration Report\n\n");

        // Summary
        report.push_str("## Summary\n\n");
        report.push_str(&format!(
            "- Shapes converted: {}\n",
            result.report.shapes_converted
        ));
        report.push_str(&format!(
            "- Constraints converted: {}\n",
            result.report.constraints_converted
        ));
        report.push_str(&format!("- Warnings: {}\n", result.report.warnings.len()));
        report.push_str(&format!("- Errors: {}\n", result.report.errors.len()));
        report.push('\n');

        // Statistics
        report.push_str("## Statistics\n\n");
        let stats = &result.report.statistics;
        report.push_str(&format!(
            "- Total ShEx shapes: {}\n",
            stats.total_shex_shapes
        ));
        report.push_str(&format!(
            "- Total SHACL shapes: {}\n",
            stats.total_shacl_shapes
        ));
        report.push_str(&format!(
            "- Triple constraints: {}\n",
            stats.triple_constraints
        ));
        report.push_str(&format!("- Node constraints: {}\n", stats.node_constraints));
        report.push_str(&format!("- Value sets: {}\n", stats.value_sets));
        report.push_str(&format!(
            "- Cardinality constraints: {}\n",
            stats.cardinality_constraints
        ));
        report.push_str(&format!("- Closed shapes: {}\n", stats.closed_shapes));
        report.push_str(&format!(
            "- Semantic actions skipped: {}\n",
            stats.semantic_actions_skipped
        ));
        report.push('\n');

        // Warnings
        if !result.report.warnings.is_empty() {
            report.push_str("## Warnings\n\n");
            for warning in &result.report.warnings {
                report.push_str(&format!("### {}\n", warning.code));
                report.push_str(&format!("{}\n", warning.message));
                if let Some(loc) = &warning.location {
                    report.push_str(&format!("Location: {}\n", loc));
                }
                if let Some(suggestion) = &warning.suggestion {
                    report.push_str(&format!("Suggestion: {}\n", suggestion));
                }
                report.push('\n');
            }
        }

        // Errors
        if !result.report.errors.is_empty() {
            report.push_str("## Errors\n\n");
            for error in &result.report.errors {
                report.push_str(&format!("### {}\n", error.code));
                report.push_str(&format!("{}\n", error.message));
                if let Some(loc) = &error.location {
                    report.push_str(&format!("Location: {}\n", loc));
                }
                report.push('\n');
            }
        }

        // Unmapped features
        if !result.report.unmapped_features.is_empty() {
            report.push_str("## Unmapped Features\n\n");
            for feature in &result.report.unmapped_features {
                report.push_str(&format!("### {}\n", feature.feature));
                report.push_str(&format!("{}\n", feature.description));
                if let Some(workaround) = &feature.workaround {
                    report.push_str(&format!("Workaround: {}\n", workaround));
                }
                report.push('\n');
            }
        }

        report
    }
}

impl Default for ShexMigrationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for migration configuration
#[derive(Debug)]
pub struct MigrationConfigBuilder {
    config: MigrationConfig,
}

impl MigrationConfigBuilder {
    /// Create a new builder with defaults
    pub fn new() -> Self {
        Self {
            config: MigrationConfig::default(),
        }
    }

    /// Set whether to preserve comments
    pub fn preserve_comments(mut self, preserve: bool) -> Self {
        self.config.preserve_comments = preserve;
        self
    }

    /// Set whether to generate verbose output
    pub fn verbose_output(mut self, verbose: bool) -> Self {
        self.config.verbose_output = verbose;
        self
    }

    /// Set the base IRI
    pub fn base_iri(mut self, base: impl Into<String>) -> Self {
        self.config.base_iri = base.into();
        self
    }

    /// Set closed handling strategy
    pub fn closed_handling(mut self, handling: ClosedHandling) -> Self {
        self.config.semantic_mapping.closed_handling = handling;
        self
    }

    /// Set value set handling strategy
    pub fn value_set_handling(mut self, handling: ValueSetHandling) -> Self {
        self.config.semantic_mapping.value_set_handling = handling;
        self
    }

    /// Build the configuration
    pub fn build(self) -> MigrationConfig {
        self.config
    }
}

impl Default for MigrationConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migration_engine_creation() {
        let engine = ShexMigrationEngine::new();
        assert!(!engine.config.verbose_output);
        assert!(engine.config.preserve_comments);
    }

    #[test]
    fn test_config_builder() {
        let config = MigrationConfigBuilder::new()
            .preserve_comments(false)
            .verbose_output(true)
            .base_iri("http://test.org/shapes/")
            .build();

        assert!(!config.preserve_comments);
        assert!(config.verbose_output);
        assert_eq!(config.base_iri, "http://test.org/shapes/");
    }

    #[test]
    fn test_parse_simple_shex() {
        let shex = r#"
PREFIX ex: <http://example.org/>

ex:Person {
    ex:name xsd:string ;
    ex:age xsd:integer ?
}
"#;

        let engine = ShexMigrationEngine::new();
        let schema = engine.parse_shex(shex).expect("parsing should succeed");

        assert_eq!(schema.shapes.len(), 1);
        assert_eq!(
            schema.prefixes.get("ex"),
            Some(&"http://example.org/".to_string())
        );
    }

    #[test]
    fn test_parse_prefix_line() {
        let engine = ShexMigrationEngine::new();
        let result = engine.parse_prefix_line("PREFIX ex: <http://example.org/>");

        assert!(result.is_some());
        let (prefix, iri) = result.expect("valid IRI");
        assert_eq!(prefix, "ex");
        assert_eq!(iri, "http://example.org/");
    }

    #[test]
    fn test_cardinality_extraction() {
        let engine = ShexMigrationEngine::new();

        let (text, min, max) = engine.extract_cardinality("xsd:string*");
        assert_eq!(text, "xsd:string");
        assert_eq!(min, 0);
        assert_eq!(max, -1);

        let (text, min, max) = engine.extract_cardinality("xsd:string+");
        assert_eq!(text, "xsd:string");
        assert_eq!(min, 1);
        assert_eq!(max, -1);

        let (text, min, max) = engine.extract_cardinality("xsd:string?");
        assert_eq!(text, "xsd:string");
        assert_eq!(min, 0);
        assert_eq!(max, 1);

        let (text, min, max) = engine.extract_cardinality("xsd:string{2,5}");
        assert_eq!(text, "xsd:string");
        assert_eq!(min, 2);
        assert_eq!(max, 5);
    }

    #[test]
    fn test_migrate_simple_schema() {
        let schema = ShexSchema {
            prefixes: HashMap::new(),
            base: Some("http://example.org/".to_string()),
            imports: Vec::new(),
            shapes: vec![ShexShape {
                id: "http://example.org/Person".to_string(),
                expression: ShexExpression::TripleConstraint(TripleConstraint {
                    predicate: "http://example.org/name".to_string(),
                    value_expr: None,
                    min: 1,
                    max: 1,
                    inverse: false,
                    annotations: Vec::new(),
                    sem_acts: Vec::new(),
                }),
                closed: false,
                extra: Vec::new(),
                annotations: Vec::new(),
                comments: vec!["A person shape".to_string()],
            }],
            start: None,
        };

        let mut engine = ShexMigrationEngine::new();
        let result = engine.migrate(&schema).expect("migration should succeed");

        assert_eq!(result.shapes.len(), 1);
        assert_eq!(result.report.shapes_converted, 1);
    }

    #[test]
    fn test_value_set_parsing() {
        let engine = ShexMigrationEngine::new();
        let values = engine
            .parse_value_set("[<http://example.org/a> <http://example.org/b>]")
            .expect("parsing should succeed");

        assert_eq!(values.len(), 2);
    }

    #[test]
    fn test_generate_report() {
        let result = MigrationResult {
            shapes: IndexMap::new(),
            report: MigrationReport {
                shapes_converted: 5,
                constraints_converted: 15,
                warnings: vec![MigrationWarning {
                    code: "TEST".to_string(),
                    message: "Test warning".to_string(),
                    location: None,
                    suggestion: None,
                }],
                errors: Vec::new(),
                unmapped_features: Vec::new(),
                statistics: MigrationStatistics {
                    total_shex_shapes: 5,
                    total_shacl_shapes: 5,
                    triple_constraints: 10,
                    node_constraints: 3,
                    value_sets: 2,
                    cardinality_constraints: 8,
                    closed_shapes: 1,
                    semantic_actions_skipped: 0,
                    unmapped_features: Vec::new(),
                },
            },
        };

        let engine = ShexMigrationEngine::new();
        let report = engine.generate_report(&result);

        assert!(report.contains("Migration Report"));
        assert!(report.contains("Shapes converted: 5"));
        assert!(report.contains("TEST"));
    }
}
