//! # JSON Schema to SHACL Converter
//!
//! Converts JSON Schema Draft-07 definitions to SHACL shapes for RDF validation.
//!
//! ## Supported Mappings
//!
//! | JSON Schema | SHACL |
//! |-------------|-------|
//! | `type: string` | `sh:datatype xsd:string` |
//! | `type: integer` | `sh:datatype xsd:integer` |
//! | `type: number` | `sh:datatype xsd:decimal` |
//! | `type: boolean` | `sh:datatype xsd:boolean` |
//! | `required: [...]` | `sh:minCount 1` |
//! | `minLength` / `maxLength` | `sh:minLength` / `sh:maxLength` |
//! | `pattern` | `sh:pattern` |
//! | `minimum` / `maximum` | `sh:minInclusive` / `sh:maxInclusive` |
//! | `allOf` | `sh:and` |
//! | `oneOf` | `sh:xone` |
//! | `$ref` | `sh:node` |

use anyhow::{anyhow, Result};
use serde_json::Value;

// ── Public types ──────────────────────────────────────────────────────────────

/// Main converter: JSON Schema → SHACL shapes.
#[derive(Debug, Clone)]
pub struct JsonSchemaToShacl {
    /// Base IRI for generated shapes
    pub base_iri: String,
    /// Prefix used for property paths
    pub property_prefix: String,
}

/// A SHACL NodeShape.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaclShape {
    /// Shape IRI
    pub iri: String,
    /// `sh:targetClass`
    pub target_class: Option<String>,
    /// Property constraints
    pub properties: Vec<ShaclPropertyConstraint>,
    /// Logical operators (sh:and, sh:or, sh:not, sh:xone)
    pub logical: Vec<ShaclLogical>,
}

impl ShaclShape {
    pub fn new(iri: impl Into<String>) -> Self {
        Self {
            iri: iri.into(),
            target_class: None,
            properties: Vec::new(),
            logical: Vec::new(),
        }
    }

    pub fn with_target_class(mut self, class: impl Into<String>) -> Self {
        self.target_class = Some(class.into());
        self
    }
}

/// A single `sh:property` constraint block.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaclPropertyConstraint {
    /// `sh:path`
    pub path: String,
    /// `sh:minCount`
    pub min_count: Option<usize>,
    /// `sh:maxCount`
    pub max_count: Option<usize>,
    /// `sh:datatype`
    pub datatype: Option<String>,
    /// `sh:nodeKind`
    pub node_kind: Option<NodeKind>,
    /// `sh:minInclusive`
    pub min_inclusive: Option<f64>,
    /// `sh:maxInclusive`
    pub max_inclusive: Option<f64>,
    /// `sh:minLength`
    pub min_length: Option<usize>,
    /// `sh:maxLength`
    pub max_length: Option<usize>,
    /// `sh:pattern`
    pub pattern: Option<String>,
    /// `sh:node` — reference to another shape
    pub node: Option<String>,
}

impl ShaclPropertyConstraint {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            min_count: None,
            max_count: None,
            datatype: None,
            node_kind: None,
            min_inclusive: None,
            max_inclusive: None,
            min_length: None,
            max_length: None,
            pattern: None,
            node: None,
        }
    }
}

/// SHACL node kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Iri,
    Literal,
    BlankNode,
}

impl NodeKind {
    pub fn as_shacl_str(self) -> &'static str {
        match self {
            NodeKind::Iri => "sh:IRI",
            NodeKind::Literal => "sh:Literal",
            NodeKind::BlankNode => "sh:BlankNode",
        }
    }
}

/// SHACL logical operators.
#[derive(Debug, Clone, PartialEq)]
pub enum ShaclLogical {
    And(Vec<ShaclShape>),
    Or(Vec<ShaclShape>),
    Not(Box<ShaclShape>),
    Xone(Vec<ShaclShape>),
}

// ── Implementation ────────────────────────────────────────────────────────────

impl JsonSchemaToShacl {
    /// Create a new converter with the given base IRI.
    pub fn new(base_iri: impl Into<String>) -> Self {
        Self {
            base_iri: base_iri.into(),
            property_prefix: String::new(),
        }
    }

    /// Create with an explicit property prefix.
    pub fn with_property_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.property_prefix = prefix.into();
        self
    }

    /// Convert a JSON Schema object to a SHACL shape.
    pub fn convert(&self, schema: &Value) -> Result<ShaclShape> {
        if !schema.is_object() {
            return Err(anyhow!("JSON Schema must be a JSON object"));
        }

        let title = schema["title"].as_str().unwrap_or("Shape");
        let shape_iri = format!("{}{}Shape", self.base_iri, title);
        let mut shape = ShaclShape::new(shape_iri);

        // sh:targetClass from title
        if let Some(t) = schema["title"].as_str() {
            shape.target_class = Some(format!("{}{}", self.base_iri, t));
        }

        // Required fields
        let required: Vec<String> = schema["required"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        // Properties
        if let Some(props) = schema["properties"].as_object() {
            shape.properties = self.convert_properties_map(props, &required)?;
        }

        // allOf
        if let Some(all_of) = schema["allOf"].as_array() {
            let sub_shapes: Result<Vec<ShaclShape>> =
                all_of.iter().map(|s| self.convert(s)).collect();
            shape.logical.push(ShaclLogical::And(sub_shapes?));
        }

        // anyOf
        if let Some(any_of) = schema["anyOf"].as_array() {
            let sub_shapes: Result<Vec<ShaclShape>> =
                any_of.iter().map(|s| self.convert(s)).collect();
            shape.logical.push(ShaclLogical::Or(sub_shapes?));
        }

        // oneOf
        if let Some(one_of) = schema["oneOf"].as_array() {
            let sub_shapes: Result<Vec<ShaclShape>> =
                one_of.iter().map(|s| self.convert(s)).collect();
            shape.logical.push(ShaclLogical::Xone(sub_shapes?));
        }

        // not
        if let Some(not_schema) = schema.get("not") {
            let inner = self.convert(not_schema)?;
            shape.logical.push(ShaclLogical::Not(Box::new(inner)));
        }

        Ok(shape)
    }

    /// Convert a single property definition to a `ShaclPropertyConstraint`.
    pub fn convert_property(
        &self,
        name: &str,
        prop_schema: &Value,
        required: &[String],
    ) -> Result<ShaclPropertyConstraint> {
        let path = if self.property_prefix.is_empty() {
            format!("{}:{}", self.base_iri, name)
        } else {
            format!("{}:{}", self.property_prefix, name)
        };
        let mut constraint = ShaclPropertyConstraint::new(path);

        // min/max count from required
        if required.contains(&name.to_string()) {
            constraint.min_count = Some(1);
        } else {
            constraint.min_count = Some(0);
        }

        // JSON type → SHACL datatype
        if let Some(json_type) = prop_schema["type"].as_str() {
            if let Some(dt) = Self::type_to_datatype(json_type) {
                constraint.datatype = Some(dt.to_string());
            }
            match json_type {
                "string" => constraint.node_kind = Some(NodeKind::Literal),
                "integer" | "number" | "boolean" => constraint.node_kind = Some(NodeKind::Literal),
                "object" => constraint.node_kind = Some(NodeKind::Iri),
                _ => {}
            }
        }

        // String constraints
        if let Some(min_len) = prop_schema["minLength"].as_u64() {
            constraint.min_length = Some(min_len as usize);
        }
        if let Some(max_len) = prop_schema["maxLength"].as_u64() {
            constraint.max_length = Some(max_len as usize);
        }
        if let Some(pattern) = prop_schema["pattern"].as_str() {
            constraint.pattern = Some(pattern.to_string());
        }

        // Numeric constraints
        if let Some(min) = prop_schema["minimum"].as_f64() {
            constraint.min_inclusive = Some(min);
        }
        if let Some(max) = prop_schema["maximum"].as_f64() {
            constraint.max_inclusive = Some(max);
        }

        // $ref → sh:node
        if let Some(ref_val) = prop_schema["$ref"].as_str() {
            let ref_name = ref_val
                .trim_start_matches("#/definitions/")
                .trim_start_matches("#/$defs/");
            constraint.node = Some(format!("{}{}Shape", self.base_iri, ref_name));
        }

        // enum
        if prop_schema["enum"].is_array() {
            // Treat enums as string literals if not already typed
            if constraint.datatype.is_none() {
                constraint.datatype = Some("xsd:string".to_string());
            }
        }

        Ok(constraint)
    }

    /// Convert the "properties" object of a schema.
    pub fn convert_properties(
        &self,
        properties: &Value,
        required: &[String],
    ) -> Result<Vec<ShaclPropertyConstraint>> {
        let obj = properties
            .as_object()
            .ok_or_else(|| anyhow!("'properties' must be an object"))?;
        self.convert_properties_map(obj, required)
    }

    fn convert_properties_map(
        &self,
        props: &serde_json::Map<String, Value>,
        required: &[String],
    ) -> Result<Vec<ShaclPropertyConstraint>> {
        props
            .iter()
            .map(|(name, prop_schema)| self.convert_property(name, prop_schema, required))
            .collect()
    }

    /// Map a JSON Schema type string to an XSD datatype.
    pub fn type_to_datatype(json_type: &str) -> Option<&'static str> {
        match json_type {
            "string" => Some("xsd:string"),
            "integer" => Some("xsd:integer"),
            "number" => Some("xsd:decimal"),
            "boolean" => Some("xsd:boolean"),
            _ => None,
        }
    }

    /// Serialize a SHACL shape to Turtle format.
    pub fn to_turtle(&self, shape: &ShaclShape) -> String {
        let mut out = String::new();

        // Prefixes
        out.push_str("@prefix sh: <http://www.w3.org/ns/shacl#> .\n");
        out.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n");
        out.push_str(&format!("@prefix ex: <{}> .\n\n", self.base_iri));

        self.write_shape_turtle(&mut out, shape, 0);
        out
    }

    fn write_shape_turtle(&self, out: &mut String, shape: &ShaclShape, _depth: usize) {
        out.push_str(&format!("<{}>\n", shape.iri));
        out.push_str("    a sh:NodeShape ;\n");

        if let Some(ref class) = shape.target_class {
            out.push_str(&format!("    sh:targetClass <{}> ;\n", class));
        }

        for prop in &shape.properties {
            out.push_str("    sh:property [\n");
            out.push_str(&format!("        sh:path <{}> ;\n", prop.path));
            if let Some(mc) = prop.min_count {
                out.push_str(&format!("        sh:minCount {} ;\n", mc));
            }
            if let Some(mc) = prop.max_count {
                out.push_str(&format!("        sh:maxCount {} ;\n", mc));
            }
            if let Some(ref dt) = prop.datatype {
                out.push_str(&format!("        sh:datatype {} ;\n", dt));
            }
            if let Some(nk) = prop.node_kind {
                out.push_str(&format!("        sh:nodeKind {} ;\n", nk.as_shacl_str()));
            }
            if let Some(min) = prop.min_inclusive {
                out.push_str(&format!("        sh:minInclusive {} ;\n", min));
            }
            if let Some(max) = prop.max_inclusive {
                out.push_str(&format!("        sh:maxInclusive {} ;\n", max));
            }
            if let Some(ml) = prop.min_length {
                out.push_str(&format!("        sh:minLength {} ;\n", ml));
            }
            if let Some(ml) = prop.max_length {
                out.push_str(&format!("        sh:maxLength {} ;\n", ml));
            }
            if let Some(ref pat) = prop.pattern {
                out.push_str(&format!("        sh:pattern \"{}\" ;\n", pat));
            }
            if let Some(ref node) = prop.node {
                out.push_str(&format!("        sh:node <{}> ;\n", node));
            }
            out.push_str("    ] ;\n");
        }

        for logical in &shape.logical {
            match logical {
                ShaclLogical::And(shapes) => {
                    out.push_str("    sh:and (\n");
                    for s in shapes {
                        out.push_str(&format!("        <{}>\n", s.iri));
                    }
                    out.push_str("    ) ;\n");
                }
                ShaclLogical::Or(shapes) => {
                    out.push_str("    sh:or (\n");
                    for s in shapes {
                        out.push_str(&format!("        <{}>\n", s.iri));
                    }
                    out.push_str("    ) ;\n");
                }
                ShaclLogical::Not(inner) => {
                    out.push_str(&format!("    sh:not <{}> ;\n", inner.iri));
                }
                ShaclLogical::Xone(shapes) => {
                    out.push_str("    sh:xone (\n");
                    for s in shapes {
                        out.push_str(&format!("        <{}>\n", s.iri));
                    }
                    out.push_str("    ) ;\n");
                }
            }
        }

        out.push_str(".\n");
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn converter() -> JsonSchemaToShacl {
        JsonSchemaToShacl::new("http://example.org/")
    }

    // ── Type mapping ─────────────────────────────────────────────────────────

    #[test]
    fn test_type_string_maps_to_xsd_string() {
        assert_eq!(
            JsonSchemaToShacl::type_to_datatype("string"),
            Some("xsd:string")
        );
    }

    #[test]
    fn test_type_integer_maps_to_xsd_integer() {
        assert_eq!(
            JsonSchemaToShacl::type_to_datatype("integer"),
            Some("xsd:integer")
        );
    }

    #[test]
    fn test_type_number_maps_to_xsd_decimal() {
        assert_eq!(
            JsonSchemaToShacl::type_to_datatype("number"),
            Some("xsd:decimal")
        );
    }

    #[test]
    fn test_type_boolean_maps_to_xsd_boolean() {
        assert_eq!(
            JsonSchemaToShacl::type_to_datatype("boolean"),
            Some("xsd:boolean")
        );
    }

    #[test]
    fn test_type_unknown_returns_none() {
        assert_eq!(JsonSchemaToShacl::type_to_datatype("array"), None);
        assert_eq!(JsonSchemaToShacl::type_to_datatype("object"), None);
        assert_eq!(JsonSchemaToShacl::type_to_datatype("null"), None);
    }

    // ── Required / optional fields ───────────────────────────────────────────

    #[test]
    fn test_required_field_min_count_one() {
        let schema = json!({
            "title": "Person",
            "properties": { "name": { "type": "string" } },
            "required": ["name"]
        });
        let shape = converter().convert(&schema).expect("should succeed");
        let prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("name"))
            .expect("should succeed");
        assert_eq!(prop.min_count, Some(1));
    }

    #[test]
    fn test_optional_field_min_count_zero() {
        let schema = json!({
            "title": "Person",
            "properties": { "nickname": { "type": "string" } }
        });
        let shape = converter().convert(&schema).expect("should succeed");
        let prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("nickname"))
            .expect("should succeed");
        assert_eq!(prop.min_count, Some(0));
    }

    #[test]
    fn test_required_multiple_fields() {
        let schema = json!({
            "title": "Employee",
            "properties": {
                "id": { "type": "integer" },
                "name": { "type": "string" },
                "salary": { "type": "number" }
            },
            "required": ["id", "name"]
        });
        let shape = converter().convert(&schema).expect("should succeed");
        let id_prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("id"))
            .expect("should succeed");
        let name_prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("name"))
            .expect("should succeed");
        let salary_prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("salary"))
            .expect("should succeed");
        assert_eq!(id_prop.min_count, Some(1));
        assert_eq!(name_prop.min_count, Some(1));
        assert_eq!(salary_prop.min_count, Some(0));
    }

    // ── String constraints ───────────────────────────────────────────────────

    #[test]
    fn test_string_min_length() {
        let schema = json!({
            "title": "T",
            "properties": { "pw": { "type": "string", "minLength": 8 } }
        });
        let shape = converter().convert(&schema).expect("should succeed");
        let prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("pw"))
            .expect("should succeed");
        assert_eq!(prop.min_length, Some(8));
    }

    #[test]
    fn test_string_max_length() {
        let schema = json!({
            "title": "T",
            "properties": { "bio": { "type": "string", "maxLength": 500 } }
        });
        let shape = converter().convert(&schema).expect("should succeed");
        let prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("bio"))
            .expect("should succeed");
        assert_eq!(prop.max_length, Some(500));
    }

    #[test]
    fn test_string_min_and_max_length() {
        let schema = json!({
            "title": "T",
            "properties": { "username": { "type": "string", "minLength": 3, "maxLength": 30 } }
        });
        let shape = converter().convert(&schema).expect("should succeed");
        let prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("username"))
            .expect("should succeed");
        assert_eq!(prop.min_length, Some(3));
        assert_eq!(prop.max_length, Some(30));
    }

    #[test]
    fn test_string_pattern() {
        let schema = json!({
            "title": "T",
            "properties": { "email": { "type": "string", "pattern": "^[^@]+@[^@]+$" } }
        });
        let shape = converter().convert(&schema).expect("should succeed");
        let prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("email"))
            .expect("should succeed");
        assert_eq!(prop.pattern.as_deref(), Some("^[^@]+@[^@]+$"));
    }

    #[test]
    fn test_string_pattern_url() {
        let schema = json!({
            "title": "T",
            "properties": { "url": { "type": "string", "pattern": "^https?://" } }
        });
        let shape = converter().convert(&schema).expect("should succeed");
        let prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("url"))
            .expect("should succeed");
        assert!(prop.pattern.is_some());
    }

    // ── Numeric constraints ──────────────────────────────────────────────────

    #[test]
    fn test_number_minimum() {
        let schema = json!({
            "title": "T",
            "properties": { "age": { "type": "integer", "minimum": 0 } }
        });
        let shape = converter().convert(&schema).expect("should succeed");
        let prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("age"))
            .expect("should succeed");
        assert_eq!(prop.min_inclusive, Some(0.0));
    }

    #[test]
    fn test_number_maximum() {
        let schema = json!({
            "title": "T",
            "properties": { "age": { "type": "integer", "maximum": 150 } }
        });
        let shape = converter().convert(&schema).expect("should succeed");
        let prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("age"))
            .expect("should succeed");
        assert_eq!(prop.max_inclusive, Some(150.0));
    }

    #[test]
    fn test_number_min_max_range() {
        let schema = json!({
            "title": "T",
            "properties": { "score": { "type": "number", "minimum": 0.0, "maximum": 100.0 } }
        });
        let shape = converter().convert(&schema).expect("should succeed");
        let prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("score"))
            .expect("should succeed");
        assert_eq!(prop.min_inclusive, Some(0.0));
        assert_eq!(prop.max_inclusive, Some(100.0));
    }

    #[test]
    fn test_integer_type_datatype() {
        let schema = json!({
            "title": "T",
            "properties": { "count": { "type": "integer" } }
        });
        let shape = converter().convert(&schema).expect("should succeed");
        let prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("count"))
            .expect("should succeed");
        assert_eq!(prop.datatype.as_deref(), Some("xsd:integer"));
    }

    #[test]
    fn test_boolean_type_datatype() {
        let schema = json!({
            "title": "T",
            "properties": { "active": { "type": "boolean" } }
        });
        let shape = converter().convert(&schema).expect("should succeed");
        let prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("active"))
            .expect("should succeed");
        assert_eq!(prop.datatype.as_deref(), Some("xsd:boolean"));
    }

    // ── allOf / oneOf / $ref ─────────────────────────────────────────────────

    #[test]
    fn test_allof_creates_sh_and() {
        let schema = json!({
            "title": "T",
            "allOf": [
                { "title": "A", "properties": {} },
                { "title": "B", "properties": {} }
            ]
        });
        let shape = converter().convert(&schema).expect("should succeed");
        assert!(shape
            .logical
            .iter()
            .any(|l| matches!(l, ShaclLogical::And(_))));
    }

    #[test]
    fn test_allof_count() {
        let schema = json!({
            "title": "T",
            "allOf": [
                { "title": "A", "properties": {} },
                { "title": "B", "properties": {} }
            ]
        });
        let shape = converter().convert(&schema).expect("should succeed");
        if let Some(ShaclLogical::And(shapes)) = shape
            .logical
            .iter()
            .find(|l| matches!(l, ShaclLogical::And(_)))
        {
            assert_eq!(shapes.len(), 2);
        } else {
            panic!("expected And");
        }
    }

    #[test]
    fn test_oneof_creates_sh_xone() {
        let schema = json!({
            "title": "T",
            "oneOf": [
                { "title": "X", "properties": {} },
                { "title": "Y", "properties": {} }
            ]
        });
        let shape = converter().convert(&schema).expect("should succeed");
        assert!(shape
            .logical
            .iter()
            .any(|l| matches!(l, ShaclLogical::Xone(_))));
    }

    #[test]
    fn test_oneof_count() {
        let schema = json!({
            "title": "T",
            "oneOf": [
                { "title": "X", "properties": {} },
                { "title": "Y", "properties": {} },
                { "title": "Z", "properties": {} }
            ]
        });
        let shape = converter().convert(&schema).expect("should succeed");
        if let Some(ShaclLogical::Xone(shapes)) = shape
            .logical
            .iter()
            .find(|l| matches!(l, ShaclLogical::Xone(_)))
        {
            assert_eq!(shapes.len(), 3);
        } else {
            panic!("expected Xone");
        }
    }

    #[test]
    fn test_anyof_creates_sh_or() {
        let schema = json!({
            "title": "T",
            "anyOf": [
                { "title": "P", "properties": {} },
                { "title": "Q", "properties": {} }
            ]
        });
        let shape = converter().convert(&schema).expect("should succeed");
        assert!(shape
            .logical
            .iter()
            .any(|l| matches!(l, ShaclLogical::Or(_))));
    }

    #[test]
    fn test_ref_creates_sh_node() {
        let schema = json!({
            "title": "Invoice",
            "properties": {
                "customer": { "$ref": "#/definitions/Customer" }
            }
        });
        let shape = converter().convert(&schema).expect("should succeed");
        let prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("customer"))
            .expect("should succeed");
        assert!(prop.node.is_some(), "should have sh:node for $ref");
        assert!(prop
            .node
            .as_deref()
            .expect("should succeed")
            .contains("Customer"));
    }

    #[test]
    fn test_ref_defs_path() {
        let schema = json!({
            "title": "Order",
            "properties": {
                "item": { "$ref": "#/$defs/Item" }
            }
        });
        let shape = converter().convert(&schema).expect("should succeed");
        let prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("item"))
            .expect("should succeed");
        assert!(prop
            .node
            .as_deref()
            .expect("should succeed")
            .contains("Item"));
    }

    // ── Turtle serialization ─────────────────────────────────────────────────

    #[test]
    fn test_to_turtle_contains_prefix_declarations() {
        let schema = json!({ "title": "T", "properties": {} });
        let conv = converter();
        let shape = conv.convert(&schema).expect("should succeed");
        let turtle = conv.to_turtle(&shape);
        assert!(turtle.contains("@prefix sh:"));
        assert!(turtle.contains("@prefix xsd:"));
    }

    #[test]
    fn test_to_turtle_contains_node_shape() {
        let schema = json!({ "title": "Person", "properties": {} });
        let conv = converter();
        let shape = conv.convert(&schema).expect("should succeed");
        let turtle = conv.to_turtle(&shape);
        assert!(turtle.contains("sh:NodeShape"));
    }

    #[test]
    fn test_to_turtle_contains_target_class() {
        let schema = json!({ "title": "Employee", "properties": {} });
        let conv = converter();
        let shape = conv.convert(&schema).expect("should succeed");
        let turtle = conv.to_turtle(&shape);
        assert!(turtle.contains("sh:targetClass"));
        assert!(turtle.contains("Employee"));
    }

    #[test]
    fn test_to_turtle_contains_property_path() {
        let schema = json!({
            "title": "T",
            "properties": { "name": { "type": "string" } }
        });
        let conv = converter();
        let shape = conv.convert(&schema).expect("should succeed");
        let turtle = conv.to_turtle(&shape);
        assert!(turtle.contains("sh:path"));
        assert!(turtle.contains("name"));
    }

    #[test]
    fn test_to_turtle_contains_datatype() {
        let schema = json!({
            "title": "T",
            "properties": { "age": { "type": "integer" } }
        });
        let conv = converter();
        let shape = conv.convert(&schema).expect("should succeed");
        let turtle = conv.to_turtle(&shape);
        assert!(turtle.contains("sh:datatype"));
        assert!(turtle.contains("xsd:integer"));
    }

    #[test]
    fn test_to_turtle_contains_min_count() {
        let schema = json!({
            "title": "T",
            "properties": { "name": { "type": "string" } },
            "required": ["name"]
        });
        let conv = converter();
        let shape = conv.convert(&schema).expect("should succeed");
        let turtle = conv.to_turtle(&shape);
        assert!(turtle.contains("sh:minCount 1"));
    }

    #[test]
    fn test_to_turtle_contains_pattern() {
        let schema = json!({
            "title": "T",
            "properties": { "code": { "type": "string", "pattern": "^[A-Z]+$" } }
        });
        let conv = converter();
        let shape = conv.convert(&schema).expect("should succeed");
        let turtle = conv.to_turtle(&shape);
        assert!(turtle.contains("sh:pattern"));
        assert!(turtle.contains("^[A-Z]+$"));
    }

    #[test]
    fn test_to_turtle_contains_min_inclusive() {
        let schema = json!({
            "title": "T",
            "properties": { "score": { "type": "number", "minimum": 0 } }
        });
        let conv = converter();
        let shape = conv.convert(&schema).expect("should succeed");
        let turtle = conv.to_turtle(&shape);
        assert!(turtle.contains("sh:minInclusive"));
    }

    #[test]
    fn test_to_turtle_contains_max_inclusive() {
        let schema = json!({
            "title": "T",
            "properties": { "rating": { "type": "number", "maximum": 5 } }
        });
        let conv = converter();
        let shape = conv.convert(&schema).expect("should succeed");
        let turtle = conv.to_turtle(&shape);
        assert!(turtle.contains("sh:maxInclusive"));
    }

    #[test]
    fn test_to_turtle_allof_sh_and() {
        let schema = json!({
            "title": "T",
            "allOf": [
                { "title": "A", "properties": {} },
                { "title": "B", "properties": {} }
            ]
        });
        let conv = converter();
        let shape = conv.convert(&schema).expect("should succeed");
        let turtle = conv.to_turtle(&shape);
        assert!(turtle.contains("sh:and"));
    }

    #[test]
    fn test_to_turtle_oneof_sh_xone() {
        let schema = json!({
            "title": "T",
            "oneOf": [
                { "title": "X", "properties": {} },
                { "title": "Y", "properties": {} }
            ]
        });
        let conv = converter();
        let shape = conv.convert(&schema).expect("should succeed");
        let turtle = conv.to_turtle(&shape);
        assert!(turtle.contains("sh:xone"));
    }

    #[test]
    fn test_to_turtle_not_sh_not() {
        let schema = json!({
            "title": "T",
            "properties": {},
            "not": { "title": "Excluded", "properties": {} }
        });
        let conv = converter();
        let shape = conv.convert(&schema).expect("should succeed");
        let turtle = conv.to_turtle(&shape);
        assert!(turtle.contains("sh:not"));
    }

    #[test]
    fn test_to_turtle_min_length() {
        let schema = json!({
            "title": "T",
            "properties": { "pw": { "type": "string", "minLength": 8 } }
        });
        let conv = converter();
        let shape = conv.convert(&schema).expect("should succeed");
        let turtle = conv.to_turtle(&shape);
        assert!(turtle.contains("sh:minLength 8"));
    }

    #[test]
    fn test_to_turtle_max_length() {
        let schema = json!({
            "title": "T",
            "properties": { "bio": { "type": "string", "maxLength": 200 } }
        });
        let conv = converter();
        let shape = conv.convert(&schema).expect("should succeed");
        let turtle = conv.to_turtle(&shape);
        assert!(turtle.contains("sh:maxLength 200"));
    }

    // ── convert_properties helper ────────────────────────────────────────────

    #[test]
    fn test_convert_properties_empty() {
        let props = json!({});
        let constraints = converter()
            .convert_properties(&props, &[])
            .expect("should succeed");
        assert!(constraints.is_empty());
    }

    #[test]
    fn test_convert_properties_two_props() {
        let props = json!({ "a": { "type": "string" }, "b": { "type": "integer" } });
        let constraints = converter()
            .convert_properties(&props, &[])
            .expect("should succeed");
        assert_eq!(constraints.len(), 2);
    }

    #[test]
    fn test_convert_property_no_type_has_no_datatype() {
        let prop_schema = json!({});
        let c = converter()
            .convert_property("foo", &prop_schema, &[])
            .expect("should succeed");
        assert!(c.datatype.is_none());
    }

    // ── NodeKind mapping ─────────────────────────────────────────────────────

    #[test]
    fn test_string_type_literal_node_kind() {
        let schema = json!({
            "title": "T",
            "properties": { "label": { "type": "string" } }
        });
        let shape = converter().convert(&schema).expect("should succeed");
        let prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("label"))
            .expect("should succeed");
        assert_eq!(prop.node_kind, Some(NodeKind::Literal));
    }

    #[test]
    fn test_object_type_iri_node_kind() {
        let schema = json!({
            "title": "T",
            "properties": { "link": { "type": "object" } }
        });
        let shape = converter().convert(&schema).expect("should succeed");
        let prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("link"))
            .expect("should succeed");
        assert_eq!(prop.node_kind, Some(NodeKind::Iri));
    }

    #[test]
    fn test_node_kind_shacl_str() {
        assert_eq!(NodeKind::Iri.as_shacl_str(), "sh:IRI");
        assert_eq!(NodeKind::Literal.as_shacl_str(), "sh:Literal");
        assert_eq!(NodeKind::BlankNode.as_shacl_str(), "sh:BlankNode");
    }

    // ── Shape IRI generation ─────────────────────────────────────────────────

    #[test]
    fn test_shape_iri_contains_title() {
        let schema = json!({ "title": "Product", "properties": {} });
        let shape = converter().convert(&schema).expect("should succeed");
        assert!(shape.iri.contains("Product"));
    }

    #[test]
    fn test_shape_iri_contains_shape_suffix() {
        let schema = json!({ "title": "Product", "properties": {} });
        let shape = converter().convert(&schema).expect("should succeed");
        assert!(shape.iri.contains("Shape"));
    }

    #[test]
    fn test_shape_target_class_set() {
        let schema = json!({ "title": "Vehicle", "properties": {} });
        let shape = converter().convert(&schema).expect("should succeed");
        assert!(shape.target_class.is_some());
        assert!(shape
            .target_class
            .as_deref()
            .expect("should succeed")
            .contains("Vehicle"));
    }

    // ── Error handling ───────────────────────────────────────────────────────

    #[test]
    fn test_convert_non_object_fails() {
        let schema = json!([1, 2, 3]);
        assert!(converter().convert(&schema).is_err());
    }

    #[test]
    fn test_convert_string_fails() {
        let schema = json!("not an object");
        assert!(converter().convert(&schema).is_err());
    }

    // ── ShaclShape builder ───────────────────────────────────────────────────

    #[test]
    fn test_shacl_shape_new() {
        let s = ShaclShape::new("http://example.org/MyShape");
        assert_eq!(s.iri, "http://example.org/MyShape");
        assert!(s.target_class.is_none());
        assert!(s.properties.is_empty());
        assert!(s.logical.is_empty());
    }

    #[test]
    fn test_shacl_shape_with_target_class() {
        let s = ShaclShape::new("http://example.org/Shape")
            .with_target_class("http://example.org/Person");
        assert_eq!(s.target_class.as_deref(), Some("http://example.org/Person"));
    }

    // ── ShaclPropertyConstraint builder ─────────────────────────────────────

    #[test]
    fn test_property_constraint_new() {
        let c = ShaclPropertyConstraint::new("http://example.org/name");
        assert_eq!(c.path, "http://example.org/name");
        assert!(c.min_count.is_none());
        assert!(c.datatype.is_none());
    }
}

// ---------------------------------------------------------------------------
// Extended schema_import / JSON-Schema-to-SHACL tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod extended_schema_import_tests {
    use super::*;
    use serde_json::json;

    fn converter() -> JsonSchemaToShacl {
        JsonSchemaToShacl::new("http://example.org/")
    }

    // ---- to_turtle output format ----------------------------------------

    #[test]
    fn test_to_turtle_contains_sh_prefix() {
        let schema = json!({ "title": "Item", "properties": {} });
        let shape = converter().convert(&schema).expect("convert");
        let ttl = converter().to_turtle(&shape);
        assert!(ttl.contains("@prefix sh: <http://www.w3.org/ns/shacl#>"));
    }

    #[test]
    fn test_to_turtle_contains_xsd_prefix() {
        let schema = json!({ "title": "Item", "properties": {} });
        let shape = converter().convert(&schema).expect("convert");
        let ttl = converter().to_turtle(&shape);
        assert!(ttl.contains("@prefix xsd: <http://www.w3.org/2001/XMLSchema#>"));
    }

    #[test]
    fn test_to_turtle_contains_base_iri_prefix() {
        let schema = json!({ "title": "Item", "properties": {} });
        let shape = converter().convert(&schema).expect("convert");
        let ttl = converter().to_turtle(&shape);
        assert!(ttl.contains("@prefix ex: <http://example.org/>"));
    }

    #[test]
    fn test_to_turtle_declares_a_sh_node_shape() {
        let schema = json!({ "title": "Thing", "properties": {} });
        let shape = converter().convert(&schema).expect("convert");
        let ttl = converter().to_turtle(&shape);
        assert!(ttl.contains("a sh:NodeShape"));
    }

    #[test]
    fn test_to_turtle_target_class_when_title_present() {
        let schema = json!({ "title": "Vehicle", "properties": {} });
        let shape = converter().convert(&schema).expect("convert");
        let ttl = converter().to_turtle(&shape);
        assert!(ttl.contains("sh:targetClass"));
        assert!(ttl.contains("Vehicle"));
    }

    #[test]
    fn test_to_turtle_string_property_has_datatype() {
        let schema = json!({
            "title": "Doc",
            "properties": { "label": { "type": "string" } }
        });
        let shape = converter().convert(&schema).expect("convert");
        let ttl = converter().to_turtle(&shape);
        assert!(ttl.contains("sh:datatype xsd:string"));
    }

    #[test]
    fn test_to_turtle_integer_property_has_datatype() {
        let schema = json!({
            "title": "Doc",
            "properties": { "count": { "type": "integer" } }
        });
        let shape = converter().convert(&schema).expect("convert");
        let ttl = converter().to_turtle(&shape);
        assert!(ttl.contains("sh:datatype xsd:integer"));
    }

    #[test]
    fn test_to_turtle_required_field_min_count() {
        let schema = json!({
            "title": "Doc",
            "properties": { "name": { "type": "string" } },
            "required": ["name"]
        });
        let shape = converter().convert(&schema).expect("convert");
        let ttl = converter().to_turtle(&shape);
        assert!(ttl.contains("sh:minCount 1"));
    }

    #[test]
    fn test_to_turtle_pattern_included() {
        let schema = json!({
            "title": "Doc",
            "properties": { "code": { "type": "string", "pattern": "^[A-Z]{3}$" } }
        });
        let shape = converter().convert(&schema).expect("convert");
        let ttl = converter().to_turtle(&shape);
        assert!(ttl.contains("sh:pattern"));
        assert!(ttl.contains("^[A-Z]{3}$"));
    }

    #[test]
    fn test_to_turtle_ends_with_dot() {
        let schema = json!({ "title": "Foo", "properties": {} });
        let shape = converter().convert(&schema).expect("convert");
        let ttl = converter().to_turtle(&shape);
        assert!(ttl.trim_end().ends_with('.'));
    }

    // ---- JsonSchemaToShacl builder -------------------------------------

    #[test]
    fn test_custom_base_iri() {
        let c = JsonSchemaToShacl::new("http://myontology.example/");
        let schema = json!({ "title": "Foo", "properties": {} });
        let shape = c.convert(&schema).expect("convert");
        assert!(shape.iri.contains("Foo"));
    }

    #[test]
    fn test_with_property_prefix_does_not_panic() {
        let c = JsonSchemaToShacl::new("http://example.org/")
            .with_property_prefix("http://props.example.org/");
        let schema = json!({
            "title": "Thing",
            "properties": { "weight": { "type": "number" } }
        });
        let result = c.convert(&schema);
        assert!(result.is_ok());
    }

    // ---- ShaclPropertyConstraint builder --------------------------------

    #[test]
    fn test_property_constraint_min_max_count_defaults_none() {
        let p = ShaclPropertyConstraint::new("http://ex/p");
        assert!(p.min_count.is_none());
        assert!(p.max_count.is_none());
    }

    #[test]
    fn test_property_constraint_min_max_inclusive() {
        let schema = json!({
            "title": "Doc",
            "properties": { "score": { "type": "number", "minimum": 0.0, "maximum": 100.0 } }
        });
        let shape = converter().convert(&schema).expect("convert");
        let prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("score"))
            .expect("score prop");
        assert_eq!(prop.min_inclusive, Some(0.0));
        assert_eq!(prop.max_inclusive, Some(100.0));
    }

    #[test]
    fn test_property_constraint_min_max_length() {
        let schema = json!({
            "title": "Doc",
            "properties": { "code": { "type": "string", "minLength": 3, "maxLength": 10 } }
        });
        let shape = converter().convert(&schema).expect("convert");
        let prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("code"))
            .expect("code prop");
        assert_eq!(prop.min_length, Some(3));
        assert_eq!(prop.max_length, Some(10));
    }

    // ---- Multiple properties -------------------------------------------

    #[test]
    fn test_multiple_properties_all_present() {
        let schema = json!({
            "title": "Person",
            "properties": {
                "name": { "type": "string" },
                "age":  { "type": "integer" },
                "email": { "type": "string" }
            }
        });
        let shape = converter().convert(&schema).expect("convert");
        assert_eq!(shape.properties.len(), 3);
    }

    #[test]
    fn test_empty_properties_gives_no_constraints() {
        let schema = json!({ "title": "Empty", "properties": {} });
        let shape = converter().convert(&schema).expect("convert");
        assert!(shape.properties.is_empty());
    }

    // ---- Logical operators ---------------------------------------------

    #[test]
    fn test_all_of_generates_sh_and() {
        let schema = json!({
            "title": "Multi",
            "allOf": [
                { "title": "A", "properties": {} },
                { "title": "B", "properties": {} }
            ]
        });
        let shape = converter().convert(&schema).expect("convert");
        let has_and = shape
            .logical
            .iter()
            .any(|l| matches!(l, ShaclLogical::And(_)));
        assert!(has_and, "sh:and logical constraint expected");
    }

    #[test]
    fn test_one_of_generates_sh_xone() {
        let schema = json!({
            "title": "Choice",
            "oneOf": [
                { "title": "X", "properties": {} },
                { "title": "Y", "properties": {} }
            ]
        });
        let shape = converter().convert(&schema).expect("convert");
        let has_xone = shape
            .logical
            .iter()
            .any(|l| matches!(l, ShaclLogical::Xone(_)));
        assert!(has_xone, "sh:xone logical constraint expected");
    }

    // ---- $ref handling ------------------------------------------------

    #[test]
    fn test_ref_generates_sh_node() {
        let schema = json!({
            "title": "Container",
            "properties": {
                "child": { "$ref": "#/definitions/Child" }
            }
        });
        let shape = converter().convert(&schema).expect("convert");
        let prop = shape
            .properties
            .iter()
            .find(|p| p.path.contains("child"))
            .expect("child prop");
        assert!(prop.node.is_some(), "sh:node reference expected");
    }

    // ---- NodeKind as_shacl_str (only existing variants) ---------------

    #[test]
    fn test_node_kind_iri_shacl_str() {
        assert_eq!(NodeKind::Iri.as_shacl_str(), "sh:IRI");
    }

    #[test]
    fn test_node_kind_literal_shacl_str() {
        assert_eq!(NodeKind::Literal.as_shacl_str(), "sh:Literal");
    }

    #[test]
    fn test_node_kind_blank_node_shacl_str() {
        assert_eq!(NodeKind::BlankNode.as_shacl_str(), "sh:BlankNode");
    }

    // ---- ShaclShape logical operations ---------------------------------

    #[test]
    fn test_shacl_shape_logical_default_empty() {
        let s = ShaclShape::new("http://ex/S");
        assert!(s.logical.is_empty());
    }

    #[test]
    fn test_shacl_shape_properties_default_empty() {
        let s = ShaclShape::new("http://ex/S");
        assert!(s.properties.is_empty());
    }

    // ---- Type mapping completeness ------------------------------------

    #[test]
    fn test_type_number_maps_to_xsd_decimal() {
        assert_eq!(
            JsonSchemaToShacl::type_to_datatype("number"),
            Some("xsd:decimal")
        );
    }

    #[test]
    fn test_type_null_returns_none() {
        assert_eq!(JsonSchemaToShacl::type_to_datatype("null"), None);
    }

    #[test]
    fn test_type_array_returns_none() {
        assert_eq!(JsonSchemaToShacl::type_to_datatype("array"), None);
    }

    // ---- to_turtle with min/max inclusive values ----------------------

    #[test]
    fn test_to_turtle_min_inclusive_present() {
        let schema = json!({
            "title": "T",
            "properties": { "age": { "type": "number", "minimum": 0.0 } }
        });
        let shape = converter().convert(&schema).expect("convert");
        let ttl = converter().to_turtle(&shape);
        assert!(ttl.contains("sh:minInclusive 0"));
    }

    #[test]
    fn test_to_turtle_max_inclusive_present() {
        let schema = json!({
            "title": "T",
            "properties": { "score": { "type": "number", "maximum": 100.0 } }
        });
        let shape = converter().convert(&schema).expect("convert");
        let ttl = converter().to_turtle(&shape);
        assert!(ttl.contains("sh:maxInclusive 100"));
    }

    #[test]
    fn test_to_turtle_min_length_present() {
        let schema = json!({
            "title": "T",
            "properties": { "code": { "type": "string", "minLength": 5 } }
        });
        let shape = converter().convert(&schema).expect("convert");
        let ttl = converter().to_turtle(&shape);
        assert!(ttl.contains("sh:minLength 5"));
    }

    #[test]
    fn test_to_turtle_max_length_present() {
        let schema = json!({
            "title": "T",
            "properties": { "tag": { "type": "string", "maxLength": 20 } }
        });
        let shape = converter().convert(&schema).expect("convert");
        let ttl = converter().to_turtle(&shape);
        assert!(ttl.contains("sh:maxLength 20"));
    }
}
