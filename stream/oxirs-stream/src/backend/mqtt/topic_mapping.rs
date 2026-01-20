//! MQTT Topic to RDF Mapping
//!
//! Maps MQTT topic patterns to RDF triples using configurable rules

use super::types::{TopicRdfMapping, TransformOperation, ValueTransformation};
use crate::error::{StreamError, StreamResult};
use crate::event::{EventMetadata, StreamEvent};
use chrono::Utc;
use std::collections::HashMap;

/// Topic pattern matcher
pub struct TopicMapper {
    /// Cache for compiled regex patterns
    pattern_cache: std::sync::Mutex<HashMap<String, regex::Regex>>,
}

impl TopicMapper {
    /// Create a new topic mapper
    pub fn new() -> Self {
        Self {
            pattern_cache: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Check if topic matches pattern (with MQTT wildcards)
    pub fn matches(&self, topic: &str, pattern: &str) -> bool {
        // MQTT wildcards:
        // + matches single level
        // # matches multiple levels (must be last)

        let topic_parts: Vec<&str> = topic.split('/').collect();
        let pattern_parts: Vec<&str> = pattern.split('/').collect();

        self.matches_parts(&topic_parts, &pattern_parts)
    }

    /// Match topic parts against pattern parts
    fn matches_parts(&self, topic: &[&str], pattern: &[&str]) -> bool {
        let mut ti = 0;
        let mut pi = 0;

        while pi < pattern.len() && ti < topic.len() {
            match pattern[pi] {
                "#" => {
                    // Multi-level wildcard - matches all remaining levels
                    return pi == pattern.len() - 1; // Must be last
                }
                "+" => {
                    // Single-level wildcard - skip one level
                    ti += 1;
                    pi += 1;
                }
                part if part == topic[ti] => {
                    // Exact match
                    ti += 1;
                    pi += 1;
                }
                _ => {
                    // No match
                    return false;
                }
            }
        }

        // Both must be consumed
        ti == topic.len() && pi == pattern.len()
    }

    /// Convert parsed payload to StreamEvents using RDF mapping
    pub fn to_stream_events(
        &self,
        topic: &str,
        parsed: &HashMap<String, serde_json::Value>,
        mapping: &TopicRdfMapping,
    ) -> StreamResult<Vec<StreamEvent>> {
        let mut events = Vec::new();

        // Extract topic segments for placeholder replacement
        let topic_segments: Vec<&str> = topic.split('/').collect();

        // Build subject URI from pattern
        let subject = self.apply_pattern(&mapping.subject_pattern, &topic_segments, parsed)?;

        // Build graph URI if specified
        let graph = mapping
            .graph_pattern
            .as_ref()
            .map(|pattern| self.apply_pattern(pattern, &topic_segments, parsed))
            .transpose()?;

        // Create rdf:type triple if specified
        if let Some(ref type_uri) = mapping.type_uri {
            let metadata = EventMetadata {
                event_id: uuid::Uuid::new_v4().to_string(),
                timestamp: Utc::now(),
                source: format!("mqtt:{}", topic),
                user: None,
                context: graph.clone(),
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            };

            events.push(StreamEvent::TripleAdded {
                subject: subject.clone(),
                predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                object: type_uri.clone(),
                graph: graph.clone(),
                metadata,
            });
        }

        // Create triple for each mapped predicate
        for (field, predicate) in &mapping.predicate_map {
            if let Some(value) = parsed.get(field) {
                // Apply transformations if any
                let transformed_value =
                    self.apply_transformations(value.clone(), field, &mapping.transformations)?;

                let object = self.json_to_rdf_literal(&transformed_value)?;

                let metadata = EventMetadata {
                    event_id: uuid::Uuid::new_v4().to_string(),
                    timestamp: Utc::now(),
                    source: format!("mqtt:{}", topic),
                    user: None,
                    context: graph.clone(),
                    caused_by: None,
                    version: "1.0".to_string(),
                    properties: HashMap::new(),
                    checksum: None,
                };

                events.push(StreamEvent::TripleAdded {
                    subject: subject.clone(),
                    predicate: predicate.clone(),
                    object,
                    graph: graph.clone(),
                    metadata,
                });
            }
        }

        Ok(events)
    }

    /// Apply pattern with placeholders
    fn apply_pattern(
        &self,
        pattern: &str,
        topic_segments: &[&str],
        payload: &HashMap<String, serde_json::Value>,
    ) -> StreamResult<String> {
        let mut result = pattern.to_string();

        // Replace {topic.N} placeholders
        for (i, segment) in topic_segments.iter().enumerate() {
            result = result.replace(&format!("{{topic.{}}}", i), segment);
        }

        // Replace {payload.field} placeholders
        for (field, value) in payload {
            if let Some(s) = value.as_str() {
                result = result.replace(&format!("{{payload.{}}}", field), s);
            } else if let Some(n) = value.as_i64() {
                result = result.replace(&format!("{{payload.{}}}", field), &n.to_string());
            } else if let Some(f) = value.as_f64() {
                result = result.replace(&format!("{{payload.{}}}", field), &f.to_string());
            }
        }

        Ok(result)
    }

    /// Apply value transformations
    fn apply_transformations(
        &self,
        mut value: serde_json::Value,
        field: &str,
        transformations: &[ValueTransformation],
    ) -> StreamResult<serde_json::Value> {
        for transform in transformations {
            if transform.field == field {
                value = match &transform.operation {
                    TransformOperation::Scale { factor } => {
                        if let Some(n) = value.as_f64() {
                            serde_json::json!(n * factor)
                        } else {
                            value
                        }
                    }
                    TransformOperation::Offset { value: offset } => {
                        if let Some(n) = value.as_f64() {
                            serde_json::json!(n + offset)
                        } else {
                            value
                        }
                    }
                    TransformOperation::LookupTable { table } => {
                        if let Some(s) = value.as_str() {
                            table.get(s).map(|v| serde_json::json!(v)).unwrap_or(value)
                        } else {
                            value
                        }
                    }
                    _ => value, // Other transformations not implemented yet
                };
            }
        }

        Ok(value)
    }

    /// Convert JSON value to RDF literal string
    fn json_to_rdf_literal(&self, value: &serde_json::Value) -> StreamResult<String> {
        match value {
            serde_json::Value::String(s) => Ok(format!("\"{}\"", s)),
            serde_json::Value::Number(n) => {
                if n.is_i64() {
                    Ok(format!(
                        "\"{}\"^^<http://www.w3.org/2001/XMLSchema#integer>",
                        n
                    ))
                } else if n.is_f64() {
                    Ok(format!(
                        "\"{}\"^^<http://www.w3.org/2001/XMLSchema#double>",
                        n
                    ))
                } else {
                    Ok(format!("\"{}\"", n))
                }
            }
            serde_json::Value::Bool(b) => Ok(format!(
                "\"{}\"^^<http://www.w3.org/2001/XMLSchema#boolean>",
                b
            )),
            serde_json::Value::Null => Ok("\"\"".to_string()),
            _ => {
                // Serialize complex types as JSON
                let json_str = serde_json::to_string(value).map_err(|e| {
                    StreamError::Serialization(format!("JSON serialization failed: {}", e))
                })?;
                Ok(format!(
                    "\"{}\"^^<http://www.w3.org/1999/02/22-rdf-syntax-ns#JSON>",
                    json_str.replace('"', "\\\"")
                ))
            }
        }
    }
}

impl Default for TopicMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Topic match result
pub struct TopicMatch {
    /// Matched topic pattern
    pub pattern: String,
    /// Extracted segments
    pub segments: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_matching() {
        let mapper = TopicMapper::new();

        // Exact match
        assert!(mapper.matches("factory/sensor1/temperature", "factory/sensor1/temperature"));

        // Single-level wildcard (+)
        assert!(mapper.matches("factory/sensor1/temperature", "factory/+/temperature"));
        assert!(!mapper.matches(
            "factory/floor1/sensor1/temperature",
            "factory/+/temperature"
        ));

        // Multi-level wildcard (#)
        assert!(mapper.matches("factory/sensor1/temperature", "factory/#"));
        assert!(mapper.matches("factory/floor1/sensor1/temperature", "factory/#"));
        assert!(!mapper.matches("plant/sensor1/temperature", "factory/#"));

        // Combined
        assert!(mapper.matches(
            "factory/floor1/sensor1/temperature",
            "factory/+/+/temperature"
        ));
        assert!(mapper.matches("factory/floor1/sensor1/temperature", "factory/+/#"));
    }

    #[test]
    fn test_pattern_application() {
        let mapper = TopicMapper::new();
        let topic_segments = vec!["factory", "sensor1", "temperature"];
        let mut payload = HashMap::new();
        payload.insert("device_id".to_string(), serde_json::json!("DEV001"));

        let result = mapper
            .apply_pattern(
                "urn:factory:{topic.0}:device:{payload.device_id}",
                &topic_segments,
                &payload,
            )
            .unwrap();

        assert_eq!(result, "urn:factory:factory:device:DEV001");
    }

    #[test]
    fn test_json_to_rdf_literal() {
        let mapper = TopicMapper::new();

        // String
        assert_eq!(
            mapper
                .json_to_rdf_literal(&serde_json::json!("hello"))
                .unwrap(),
            "\"hello\""
        );

        // Integer
        assert!(mapper
            .json_to_rdf_literal(&serde_json::json!(42))
            .unwrap()
            .contains("integer"));

        // Float
        assert!(mapper
            .json_to_rdf_literal(&serde_json::json!(1.5))
            .unwrap()
            .contains("double"));

        // Boolean
        assert!(mapper
            .json_to_rdf_literal(&serde_json::json!(true))
            .unwrap()
            .contains("boolean"));
    }

    #[test]
    fn test_value_transformations() {
        let mapper = TopicMapper::new();

        // Scale transformation
        let transform = ValueTransformation {
            field: "temperature".to_string(),
            operation: TransformOperation::Scale { factor: 1.8 },
        };

        let result = mapper
            .apply_transformations(serde_json::json!(100.0), "temperature", &[transform])
            .unwrap();

        assert_eq!(result.as_f64().unwrap(), 180.0);

        // Offset transformation
        let transform = ValueTransformation {
            field: "temperature".to_string(),
            operation: TransformOperation::Offset { value: 32.0 },
        };

        let result = mapper
            .apply_transformations(serde_json::json!(0.0), "temperature", &[transform])
            .unwrap();

        assert_eq!(result.as_f64().unwrap(), 32.0);
    }
}
