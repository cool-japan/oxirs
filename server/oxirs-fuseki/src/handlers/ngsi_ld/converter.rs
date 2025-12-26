//! NGSI-LD to RDF Conversion
//!
//! Bidirectional conversion between NGSI-LD entities and RDF triples.
//! This module enables OxiRS to store NGSI-LD data as native RDF graphs.

use super::types::{
    GeoProperty, GeoPropertyType, NgsiAttribute, NgsiContext, NgsiEntity, NgsiError, NgsiProperty,
    NgsiRelationship, NgsiType, PropertyType, RelationshipType,
};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// NGSI-LD namespace prefixes
pub const NGSI_LD_PREFIX: &str = "https://uri.etsi.org/ngsi-ld/";
pub const XSD_PREFIX: &str = "http://www.w3.org/2001/XMLSchema#";
pub const RDF_PREFIX: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
pub const GEOJSON_PREFIX: &str = "https://purl.org/geojson/vocab#";

/// RDF Triple representation
#[derive(Debug, Clone, PartialEq)]
pub struct RdfTriple {
    pub subject: String,
    pub predicate: String,
    pub object: RdfObject,
}

/// RDF Object (URI or Literal)
#[derive(Debug, Clone, PartialEq)]
pub enum RdfObject {
    /// URI reference
    Uri(String),
    /// Literal with optional datatype and language tag
    Literal {
        value: String,
        datatype: Option<String>,
        language: Option<String>,
    },
}

impl RdfObject {
    /// Create a URI object
    pub fn uri(uri: impl Into<String>) -> Self {
        RdfObject::Uri(uri.into())
    }

    /// Create a string literal
    pub fn string(value: impl Into<String>) -> Self {
        RdfObject::Literal {
            value: value.into(),
            datatype: Some(format!("{}string", XSD_PREFIX)),
            language: None,
        }
    }

    /// Create an integer literal
    pub fn integer(value: i64) -> Self {
        RdfObject::Literal {
            value: value.to_string(),
            datatype: Some(format!("{}integer", XSD_PREFIX)),
            language: None,
        }
    }

    /// Create a double literal
    pub fn double(value: f64) -> Self {
        RdfObject::Literal {
            value: value.to_string(),
            datatype: Some(format!("{}double", XSD_PREFIX)),
            language: None,
        }
    }

    /// Create a boolean literal
    pub fn boolean(value: bool) -> Self {
        RdfObject::Literal {
            value: value.to_string(),
            datatype: Some(format!("{}boolean", XSD_PREFIX)),
            language: None,
        }
    }

    /// Create a dateTime literal
    pub fn datetime(value: DateTime<Utc>) -> Self {
        RdfObject::Literal {
            value: value.to_rfc3339(),
            datatype: Some(format!("{}dateTime", XSD_PREFIX)),
            language: None,
        }
    }

    /// Create a language-tagged literal
    pub fn lang_string(value: impl Into<String>, lang: impl Into<String>) -> Self {
        RdfObject::Literal {
            value: value.into(),
            datatype: None,
            language: Some(lang.into()),
        }
    }
}

/// Trait for converting NGSI-LD to RDF
pub trait NgsiToRdf {
    /// Convert NGSI-LD entity to RDF triples
    fn to_rdf(&self) -> Result<Vec<RdfTriple>, NgsiError>;
}

/// Trait for converting RDF to NGSI-LD
pub trait RdfToNgsi {
    /// Convert RDF triples to NGSI-LD entity
    fn from_rdf(subject: &str, triples: &[RdfTriple]) -> Result<Self, NgsiError>
    where
        Self: Sized;
}

/// NGSI-LD to RDF converter
#[derive(Default)]
pub struct NgsiRdfConverter {
    /// Context for URI expansion
    context: NgsiContext,
    /// Graph name for storing entities
    graph_name: Option<String>,
}

impl NgsiRdfConverter {
    /// Create a new converter
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the context for URI expansion
    pub fn with_context(mut self, context: NgsiContext) -> Self {
        self.context = context;
        self
    }

    /// Set the graph name for storing triples
    pub fn with_graph(mut self, graph: impl Into<String>) -> Self {
        self.graph_name = Some(graph.into());
        self
    }

    /// Get the graph name for an entity
    pub fn entity_graph_name(&self, entity_id: &str) -> String {
        self.graph_name
            .clone()
            .unwrap_or_else(|| format!("urn:ngsi-ld:graph:{}", entity_id))
    }

    /// Convert NGSI-LD entity to RDF triples
    pub fn entity_to_rdf(&self, entity: &NgsiEntity) -> Result<Vec<RdfTriple>, NgsiError> {
        let mut triples = Vec::new();
        let subject = &entity.id;

        // Add rdf:type triple(s)
        for type_uri in entity.entity_type.all() {
            triples.push(RdfTriple {
                subject: subject.clone(),
                predicate: format!("{}type", RDF_PREFIX),
                object: RdfObject::uri(self.expand_type(type_uri)),
            });
        }

        // Add createdAt if present
        if let Some(created_at) = entity.created_at {
            triples.push(RdfTriple {
                subject: subject.clone(),
                predicate: format!("{}createdAt", NGSI_LD_PREFIX),
                object: RdfObject::datetime(created_at),
            });
        }

        // Add modifiedAt if present
        if let Some(modified_at) = entity.modified_at {
            triples.push(RdfTriple {
                subject: subject.clone(),
                predicate: format!("{}modifiedAt", NGSI_LD_PREFIX),
                object: RdfObject::datetime(modified_at),
            });
        }

        // Add location if present
        if let Some(ref location) = entity.location {
            self.geo_property_to_rdf(
                subject,
                &format!("{}location", NGSI_LD_PREFIX),
                location,
                &mut triples,
            )?;
        }

        // Add properties
        for (name, attr) in &entity.properties {
            let predicate = self.expand_predicate(name);
            match attr {
                NgsiAttribute::Property(prop) => {
                    self.property_to_rdf(subject, &predicate, prop, &mut triples)?;
                }
                NgsiAttribute::Relationship(rel) => {
                    self.relationship_to_rdf(subject, &predicate, rel, &mut triples)?;
                }
                NgsiAttribute::GeoProperty(geo) => {
                    self.geo_property_to_rdf(subject, &predicate, geo, &mut triples)?;
                }
            }
        }

        Ok(triples)
    }

    /// Convert property to RDF triples
    fn property_to_rdf(
        &self,
        subject: &str,
        predicate: &str,
        prop: &NgsiProperty,
        triples: &mut Vec<RdfTriple>,
    ) -> Result<(), NgsiError> {
        // Generate blank node or property URI
        let prop_node = format!("{}#property_{}", subject, self.hash_string(predicate));

        // Link subject to property node
        triples.push(RdfTriple {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: RdfObject::uri(prop_node.clone()),
        });

        // Add property type
        triples.push(RdfTriple {
            subject: prop_node.clone(),
            predicate: format!("{}type", RDF_PREFIX),
            object: RdfObject::uri(format!("{}Property", NGSI_LD_PREFIX)),
        });

        // Add hasValue
        let value_object = self.json_to_rdf_object(&prop.value)?;
        triples.push(RdfTriple {
            subject: prop_node.clone(),
            predicate: format!("{}hasValue", NGSI_LD_PREFIX),
            object: value_object,
        });

        // Add observedAt if present
        if let Some(observed_at) = prop.observed_at {
            triples.push(RdfTriple {
                subject: prop_node.clone(),
                predicate: format!("{}observedAt", NGSI_LD_PREFIX),
                object: RdfObject::datetime(observed_at),
            });
        }

        // Add unitCode if present
        if let Some(ref unit) = prop.unit_code {
            triples.push(RdfTriple {
                subject: prop_node.clone(),
                predicate: format!("{}unitCode", NGSI_LD_PREFIX),
                object: RdfObject::string(unit),
            });
        }

        // Add datasetId if present
        if let Some(ref dataset_id) = prop.dataset_id {
            triples.push(RdfTriple {
                subject: prop_node.clone(),
                predicate: format!("{}datasetId", NGSI_LD_PREFIX),
                object: RdfObject::uri(dataset_id),
            });
        }

        Ok(())
    }

    /// Convert relationship to RDF triples
    fn relationship_to_rdf(
        &self,
        subject: &str,
        predicate: &str,
        rel: &NgsiRelationship,
        triples: &mut Vec<RdfTriple>,
    ) -> Result<(), NgsiError> {
        let rel_node = format!("{}#relationship_{}", subject, self.hash_string(predicate));

        // Link subject to relationship node
        triples.push(RdfTriple {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: RdfObject::uri(rel_node.clone()),
        });

        // Add relationship type
        triples.push(RdfTriple {
            subject: rel_node.clone(),
            predicate: format!("{}type", RDF_PREFIX),
            object: RdfObject::uri(format!("{}Relationship", NGSI_LD_PREFIX)),
        });

        // Add hasObject
        triples.push(RdfTriple {
            subject: rel_node.clone(),
            predicate: format!("{}hasObject", NGSI_LD_PREFIX),
            object: RdfObject::uri(&rel.object),
        });

        // Add observedAt if present
        if let Some(observed_at) = rel.observed_at {
            triples.push(RdfTriple {
                subject: rel_node.clone(),
                predicate: format!("{}observedAt", NGSI_LD_PREFIX),
                object: RdfObject::datetime(observed_at),
            });
        }

        Ok(())
    }

    /// Convert geo-property to RDF triples
    fn geo_property_to_rdf(
        &self,
        subject: &str,
        predicate: &str,
        geo: &GeoProperty,
        triples: &mut Vec<RdfTriple>,
    ) -> Result<(), NgsiError> {
        let geo_node = format!("{}#geo_{}", subject, self.hash_string(predicate));

        // Link subject to geo node
        triples.push(RdfTriple {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: RdfObject::uri(geo_node.clone()),
        });

        // Add geo-property type
        triples.push(RdfTriple {
            subject: geo_node.clone(),
            predicate: format!("{}type", RDF_PREFIX),
            object: RdfObject::uri(format!("{}GeoProperty", NGSI_LD_PREFIX)),
        });

        // Serialize GeoJSON value
        let geojson = serde_json::to_string(&geo.value).map_err(|e| {
            NgsiError::InternalError(format!("GeoJSON serialization failed: {}", e))
        })?;

        triples.push(RdfTriple {
            subject: geo_node.clone(),
            predicate: format!("{}hasValue", NGSI_LD_PREFIX),
            object: RdfObject::Literal {
                value: geojson,
                datatype: Some(format!("{}GeoJSON", GEOJSON_PREFIX)),
                language: None,
            },
        });

        Ok(())
    }

    /// Convert JSON value to RDF object
    fn json_to_rdf_object(&self, value: &serde_json::Value) -> Result<RdfObject, NgsiError> {
        match value {
            serde_json::Value::Null => Ok(RdfObject::string("")),
            serde_json::Value::Bool(b) => Ok(RdfObject::boolean(*b)),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(RdfObject::integer(i))
                } else if let Some(f) = n.as_f64() {
                    Ok(RdfObject::double(f))
                } else {
                    Ok(RdfObject::string(n.to_string()))
                }
            }
            serde_json::Value::String(s) => {
                // Check if it looks like a URI
                if s.starts_with("http://") || s.starts_with("https://") || s.starts_with("urn:") {
                    Ok(RdfObject::uri(s))
                } else {
                    Ok(RdfObject::string(s))
                }
            }
            serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                // Serialize complex values as JSON strings
                let json = serde_json::to_string(value).map_err(|e| {
                    NgsiError::InternalError(format!("JSON serialization failed: {}", e))
                })?;
                Ok(RdfObject::Literal {
                    value: json,
                    datatype: Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#JSON".to_string()),
                    language: None,
                })
            }
        }
    }

    /// Expand a type URI using context
    fn expand_type(&self, type_name: &str) -> String {
        if type_name.contains("://") || type_name.starts_with("urn:") {
            type_name.to_string()
        } else {
            format!("{}{}", NGSI_LD_PREFIX, type_name)
        }
    }

    /// Expand a predicate using context
    fn expand_predicate(&self, name: &str) -> String {
        if name.contains("://") || name.starts_with("urn:") {
            name.to_string()
        } else {
            format!("{}{}", NGSI_LD_PREFIX, name)
        }
    }

    /// Simple hash for generating unique node IDs
    fn hash_string(&self, s: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }

    /// Convert RDF triples to NGSI-LD entity
    pub fn rdf_to_entity(
        &self,
        entity_id: &str,
        triples: &[RdfTriple],
    ) -> Result<NgsiEntity, NgsiError> {
        let mut entity_type: Option<String> = None;
        let mut created_at: Option<DateTime<Utc>> = None;
        let mut modified_at: Option<DateTime<Utc>> = None;
        let mut properties: HashMap<String, NgsiAttribute> = HashMap::new();
        let mut location: Option<GeoProperty> = None;

        // First pass: extract entity-level properties
        for triple in triples.iter().filter(|t| t.subject == entity_id) {
            let pred = &triple.predicate;

            if pred == &format!("{}type", RDF_PREFIX) {
                if let RdfObject::Uri(ref type_uri) = triple.object {
                    // Skip NGSI-LD meta types
                    if !type_uri.contains("Property")
                        && !type_uri.contains("Relationship")
                        && !type_uri.contains("GeoProperty")
                    {
                        entity_type = Some(self.compact_uri(type_uri));
                    }
                }
            } else if pred == &format!("{}createdAt", NGSI_LD_PREFIX) {
                if let RdfObject::Literal { ref value, .. } = triple.object {
                    created_at = DateTime::parse_from_rfc3339(value)
                        .ok()
                        .map(|dt| dt.with_timezone(&Utc));
                }
            } else if pred == &format!("{}modifiedAt", NGSI_LD_PREFIX) {
                if let RdfObject::Literal { ref value, .. } = triple.object {
                    modified_at = DateTime::parse_from_rfc3339(value)
                        .ok()
                        .map(|dt| dt.with_timezone(&Utc));
                }
            } else if pred == &format!("{}location", NGSI_LD_PREFIX) {
                // Handle location separately
                if let RdfObject::Uri(ref node) = triple.object {
                    location = self.extract_geo_property(node, triples)?;
                }
            } else if !pred.starts_with(NGSI_LD_PREFIX) || pred.contains('#') {
                // Extract other properties
                if let RdfObject::Uri(ref node) = triple.object {
                    let name = self.compact_uri(pred);
                    if let Some(attr) = self.extract_attribute(node, triples)? {
                        properties.insert(name, attr);
                    }
                }
            }
        }

        let entity_type = entity_type
            .ok_or_else(|| NgsiError::InvalidRequest("Entity type not found".to_string()))?;

        Ok(NgsiEntity {
            id: entity_id.to_string(),
            entity_type: NgsiType::Single(entity_type),
            context: Some(NgsiContext::default()),
            scope: None,
            location,
            observation_space: None,
            operation_space: None,
            created_at,
            modified_at,
            properties,
        })
    }

    /// Extract an attribute from RDF triples
    fn extract_attribute(
        &self,
        node: &str,
        triples: &[RdfTriple],
    ) -> Result<Option<NgsiAttribute>, NgsiError> {
        let node_triples: Vec<_> = triples.iter().filter(|t| t.subject == node).collect();

        // Check node type
        let type_triple = node_triples
            .iter()
            .find(|t| t.predicate == format!("{}type", RDF_PREFIX));

        if let Some(type_t) = type_triple {
            if let RdfObject::Uri(ref type_uri) = type_t.object {
                if type_uri.contains("Property") && !type_uri.contains("GeoProperty") {
                    return Ok(Some(NgsiAttribute::Property(
                        self.extract_property(node, &node_triples)?,
                    )));
                } else if type_uri.contains("Relationship") {
                    return Ok(Some(NgsiAttribute::Relationship(
                        self.extract_relationship(node, &node_triples)?,
                    )));
                } else if type_uri.contains("GeoProperty") {
                    if let Some(geo) = self.extract_geo_property(node, triples)? {
                        return Ok(Some(NgsiAttribute::GeoProperty(geo)));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Extract property from triples
    fn extract_property(
        &self,
        _node: &str,
        triples: &[&RdfTriple],
    ) -> Result<NgsiProperty, NgsiError> {
        let mut value = serde_json::Value::Null;
        let mut observed_at = None;
        let mut unit_code = None;
        let mut dataset_id = None;

        for triple in triples {
            if triple.predicate == format!("{}hasValue", NGSI_LD_PREFIX) {
                value = self.rdf_object_to_json(&triple.object)?;
            } else if triple.predicate == format!("{}observedAt", NGSI_LD_PREFIX) {
                if let RdfObject::Literal { ref value, .. } = triple.object {
                    observed_at = DateTime::parse_from_rfc3339(value)
                        .ok()
                        .map(|dt| dt.with_timezone(&Utc));
                }
            } else if triple.predicate == format!("{}unitCode", NGSI_LD_PREFIX) {
                if let RdfObject::Literal { ref value, .. } = triple.object {
                    unit_code = Some(value.clone());
                }
            } else if triple.predicate == format!("{}datasetId", NGSI_LD_PREFIX) {
                if let RdfObject::Uri(ref uri) = triple.object {
                    dataset_id = Some(uri.clone());
                }
            }
        }

        Ok(NgsiProperty {
            property_type: PropertyType::Property,
            value,
            observed_at,
            unit_code,
            dataset_id,
            instance_id: None,
            created_at: None,
            modified_at: None,
            sub_properties: HashMap::new(),
        })
    }

    /// Extract relationship from triples
    fn extract_relationship(
        &self,
        _node: &str,
        triples: &[&RdfTriple],
    ) -> Result<NgsiRelationship, NgsiError> {
        let mut object = String::new();
        let mut observed_at = None;
        let mut dataset_id = None;

        for triple in triples {
            if triple.predicate == format!("{}hasObject", NGSI_LD_PREFIX) {
                if let RdfObject::Uri(ref uri) = triple.object {
                    object = uri.clone();
                }
            } else if triple.predicate == format!("{}observedAt", NGSI_LD_PREFIX) {
                if let RdfObject::Literal { ref value, .. } = triple.object {
                    observed_at = DateTime::parse_from_rfc3339(value)
                        .ok()
                        .map(|dt| dt.with_timezone(&Utc));
                }
            } else if triple.predicate == format!("{}datasetId", NGSI_LD_PREFIX) {
                if let RdfObject::Uri(ref uri) = triple.object {
                    dataset_id = Some(uri.clone());
                }
            }
        }

        if object.is_empty() {
            return Err(NgsiError::InvalidRequest(
                "Relationship missing object".to_string(),
            ));
        }

        Ok(NgsiRelationship {
            rel_type: RelationshipType::Relationship,
            object,
            observed_at,
            dataset_id,
            instance_id: None,
            created_at: None,
            modified_at: None,
            sub_properties: HashMap::new(),
        })
    }

    /// Extract geo-property from triples
    fn extract_geo_property(
        &self,
        node: &str,
        triples: &[RdfTriple],
    ) -> Result<Option<GeoProperty>, NgsiError> {
        let node_triples: Vec<_> = triples.iter().filter(|t| t.subject == node).collect();

        for triple in &node_triples {
            if triple.predicate == format!("{}hasValue", NGSI_LD_PREFIX) {
                if let RdfObject::Literal { ref value, .. } = triple.object {
                    let geo_value: super::types::GeoJsonValue = serde_json::from_str(value)
                        .map_err(|e| {
                            NgsiError::InvalidRequest(format!("Invalid GeoJSON: {}", e))
                        })?;

                    return Ok(Some(GeoProperty {
                        geo_type: GeoPropertyType::GeoProperty,
                        value: geo_value,
                        observed_at: None,
                        dataset_id: None,
                        instance_id: None,
                    }));
                }
            }
        }

        Ok(None)
    }

    /// Convert RDF object to JSON value
    fn rdf_object_to_json(&self, obj: &RdfObject) -> Result<serde_json::Value, NgsiError> {
        match obj {
            RdfObject::Uri(uri) => Ok(serde_json::Value::String(uri.clone())),
            RdfObject::Literal {
                value, datatype, ..
            } => {
                if let Some(dt) = datatype {
                    if dt.contains("integer") || dt.contains("int") || dt.contains("long") {
                        if let Ok(i) = value.parse::<i64>() {
                            return Ok(serde_json::Value::Number(i.into()));
                        }
                    } else if dt.contains("double")
                        || dt.contains("float")
                        || dt.contains("decimal")
                    {
                        if let Ok(f) = value.parse::<f64>() {
                            if let Some(n) = serde_json::Number::from_f64(f) {
                                return Ok(serde_json::Value::Number(n));
                            }
                        }
                    } else if dt.contains("boolean") {
                        return Ok(serde_json::Value::Bool(value == "true"));
                    } else if dt.contains("JSON") {
                        return serde_json::from_str(value).map_err(|e| {
                            NgsiError::InvalidRequest(format!("Invalid JSON literal: {}", e))
                        });
                    }
                }
                Ok(serde_json::Value::String(value.clone()))
            }
        }
    }

    /// Compact a URI using default prefixes
    fn compact_uri(&self, uri: &str) -> String {
        uri.strip_prefix(NGSI_LD_PREFIX)
            .map(|s| s.to_string())
            .unwrap_or_else(|| uri.to_string())
    }
}

impl NgsiToRdf for NgsiEntity {
    fn to_rdf(&self) -> Result<Vec<RdfTriple>, NgsiError> {
        NgsiRdfConverter::new().entity_to_rdf(self)
    }
}

impl RdfToNgsi for NgsiEntity {
    fn from_rdf(subject: &str, triples: &[RdfTriple]) -> Result<Self, NgsiError> {
        NgsiRdfConverter::new().rdf_to_entity(subject, triples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_to_rdf() {
        let entity = NgsiEntity::new("urn:ngsi-ld:Vehicle:A123", "Vehicle").with_property(
            "speed",
            NgsiProperty::new(serde_json::json!(80.5)).with_unit("KMH"),
        );

        let converter = NgsiRdfConverter::new();
        let triples = converter.entity_to_rdf(&entity).unwrap();

        assert!(!triples.is_empty());

        // Check type triple
        let type_triple = triples
            .iter()
            .find(|t| t.predicate.contains("type"))
            .unwrap();
        assert_eq!(type_triple.subject, "urn:ngsi-ld:Vehicle:A123");

        // Check property triple
        let speed_triple = triples.iter().find(|t| t.predicate.contains("speed"));
        assert!(speed_triple.is_some());
    }

    #[test]
    fn test_rdf_object_types() {
        assert!(matches!(
            RdfObject::uri("http://example.org"),
            RdfObject::Uri(_)
        ));
        assert!(matches!(
            RdfObject::string("test"),
            RdfObject::Literal { .. }
        ));
        assert!(matches!(RdfObject::integer(42), RdfObject::Literal { .. }));
        assert!(matches!(RdfObject::double(1.5), RdfObject::Literal { .. }));
        assert!(matches!(
            RdfObject::boolean(true),
            RdfObject::Literal { .. }
        ));
    }

    #[test]
    fn test_json_to_rdf_conversion() {
        let converter = NgsiRdfConverter::new();

        let obj = converter
            .json_to_rdf_object(&serde_json::json!(42))
            .unwrap();
        assert!(matches!(obj, RdfObject::Literal { value, .. } if value == "42"));

        let obj = converter
            .json_to_rdf_object(&serde_json::json!("hello"))
            .unwrap();
        assert!(matches!(obj, RdfObject::Literal { value, .. } if value == "hello"));

        let obj = converter
            .json_to_rdf_object(&serde_json::json!(true))
            .unwrap();
        assert!(matches!(obj, RdfObject::Literal { value, .. } if value == "true"));
    }

    #[test]
    fn test_roundtrip_conversion() {
        let original = NgsiEntity::new("urn:ngsi-ld:Sensor:S001", "TemperatureSensor")
            .with_property(
                "temperature",
                NgsiProperty::new(serde_json::json!(25.5)).with_unit("CEL"),
            )
            .with_relationship(
                "controlledBy",
                NgsiRelationship::new("urn:ngsi-ld:Device:D001"),
            );

        let converter = NgsiRdfConverter::new();

        // Convert to RDF
        let triples = converter.entity_to_rdf(&original).unwrap();
        assert!(!triples.is_empty());

        // Convert back to NGSI-LD
        let restored = converter
            .rdf_to_entity("urn:ngsi-ld:Sensor:S001", &triples)
            .unwrap();

        assert_eq!(restored.id, original.id);
        assert_eq!(
            restored.entity_type.primary(),
            original.entity_type.primary()
        );
    }
}
