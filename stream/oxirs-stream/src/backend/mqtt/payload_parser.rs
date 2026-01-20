//! MQTT Payload Parser
//!
//! Parses various payload formats (JSON, Sparkplug B, Protobuf, CSV, etc.)

use super::types::PayloadFormat;
use crate::error::{StreamError, StreamResult};
use std::collections::HashMap;

/// Payload parser for various formats
pub struct PayloadParser;

impl PayloadParser {
    /// Create a new payload parser
    pub fn new() -> Self {
        Self
    }

    /// Parse payload based on format
    pub fn parse(
        &self,
        payload: &[u8],
        format: &PayloadFormat,
    ) -> StreamResult<HashMap<String, serde_json::Value>> {
        match format {
            PayloadFormat::Json {
                schema: _,
                root_path,
            } => self.parse_json(payload, root_path.as_deref()),
            PayloadFormat::SparkplugB { namespace } => self.parse_sparkplug(payload, namespace),
            PayloadFormat::PlainText { datatype } => self.parse_plain_text(payload, datatype),
            PayloadFormat::Csv {
                delimiter,
                headers,
                skip_header,
            } => self.parse_csv(payload, *delimiter, headers, *skip_header),
            PayloadFormat::Raw => self.parse_raw(payload),
            _ => Err(StreamError::NotSupported(
                "Format not yet implemented".to_string(),
            )),
        }
    }

    /// Parse JSON payload
    fn parse_json(
        &self,
        payload: &[u8],
        root_path: Option<&str>,
    ) -> StreamResult<HashMap<String, serde_json::Value>> {
        let value: serde_json::Value = serde_json::from_slice(payload)
            .map_err(|e| StreamError::Deserialization(format!("JSON parse error: {}", e)))?;

        // Extract root if specified
        let value = if let Some(path) = root_path {
            self.extract_json_path(&value, path)?
        } else {
            value
        };

        // Convert to flat map
        if let serde_json::Value::Object(obj) = value {
            Ok(obj.into_iter().collect())
        } else {
            Err(StreamError::Deserialization(
                "JSON root is not an object".to_string(),
            ))
        }
    }

    /// Parse Sparkplug B payload (Eclipse Industry 4.0 standard)
    fn parse_sparkplug(
        &self,
        payload: &[u8],
        namespace: &str,
    ) -> StreamResult<HashMap<String, serde_json::Value>> {
        #[cfg(feature = "sparkplug")]
        {
            self.parse_sparkplug_impl(payload, namespace)
        }

        #[cfg(not(feature = "sparkplug"))]
        {
            Err(StreamError::NotSupported(
                "Sparkplug B support not enabled. Enable 'sparkplug' feature".to_string(),
            ))
        }
    }

    #[cfg(feature = "sparkplug")]
    fn parse_sparkplug_impl(
        &self,
        payload: &[u8],
        _namespace: &str,
    ) -> StreamResult<HashMap<String, serde_json::Value>> {
        use super::sparkplug_b;
        use prost::Message;

        // Decode Sparkplug B payload
        let sparkplug_payload = sparkplug_b::Payload::decode(payload).map_err(|e| {
            StreamError::Deserialization(format!("Sparkplug B decode error: {}", e))
        })?;

        let mut result = HashMap::new();

        // Add timestamp
        if let Some(timestamp) = sparkplug_payload.timestamp {
            result.insert("timestamp".to_string(), serde_json::json!(timestamp));
        }

        // Add sequence number
        if let Some(seq) = sparkplug_payload.seq {
            result.insert("seq".to_string(), serde_json::json!(seq));
        }

        // Parse metrics
        for metric in sparkplug_payload.metrics {
            if let Some(ref name) = metric.name {
                let value = self.extract_sparkplug_metric_value(&metric);
                result.insert(name.clone(), value);
            }
        }

        Ok(result)
    }

    #[cfg(feature = "sparkplug")]
    fn extract_sparkplug_metric_value(
        &self,
        metric: &super::sparkplug_b::Metric,
    ) -> serde_json::Value {
        use super::sparkplug_b::metric::Value;

        // Extract value based on datatype
        if let Some(value) = &metric.value {
            match value {
                Value::IntValue(v) => serde_json::json!(v),
                Value::LongValue(v) => serde_json::json!(v),
                Value::FloatValue(v) => serde_json::json!(v),
                Value::DoubleValue(v) => serde_json::json!(v),
                Value::BooleanValue(v) => serde_json::json!(v),
                Value::StringValue(v) => serde_json::json!(v),
                _ => serde_json::Value::Null,
            }
        } else {
            serde_json::Value::Null
        }
    }

    /// Parse plain text payload
    fn parse_plain_text(
        &self,
        payload: &[u8],
        _datatype: &str,
    ) -> StreamResult<HashMap<String, serde_json::Value>> {
        let text = String::from_utf8(payload.to_vec())
            .map_err(|e| StreamError::Deserialization(format!("UTF-8 decode error: {}", e)))?;

        let mut result = HashMap::new();
        result.insert("value".to_string(), serde_json::json!(text));

        Ok(result)
    }

    /// Parse CSV payload
    fn parse_csv(
        &self,
        payload: &[u8],
        delimiter: char,
        headers: &[String],
        skip_header: bool,
    ) -> StreamResult<HashMap<String, serde_json::Value>> {
        let text = String::from_utf8(payload.to_vec())
            .map_err(|e| StreamError::Deserialization(format!("UTF-8 decode error: {}", e)))?;

        let lines: Vec<&str> = text.lines().collect();

        if lines.is_empty() {
            return Ok(HashMap::new());
        }

        let start_line = if skip_header { 1 } else { 0 };

        if start_line >= lines.len() {
            return Ok(HashMap::new());
        }

        let values: Vec<&str> = lines[start_line].split(delimiter).collect();

        let mut result = HashMap::new();
        for (i, value) in values.iter().enumerate() {
            let key = headers
                .get(i)
                .map(|s: &String| s.as_str())
                .unwrap_or("field");
            let trimmed: &str = value.trim();
            result.insert(key.to_string(), serde_json::json!(trimmed));
        }

        Ok(result)
    }

    /// Parse raw bytes (base64 encoded)
    fn parse_raw(&self, payload: &[u8]) -> StreamResult<HashMap<String, serde_json::Value>> {
        use base64::{engine::general_purpose, Engine as _};

        let encoded = general_purpose::STANDARD.encode(payload);

        let mut result = HashMap::new();
        result.insert("data".to_string(), serde_json::json!(encoded));
        result.insert("size".to_string(), serde_json::json!(payload.len()));

        Ok(result)
    }

    /// Extract value from JSON path (simple dot notation)
    fn extract_json_path(
        &self,
        value: &serde_json::Value,
        path: &str,
    ) -> StreamResult<serde_json::Value> {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = value;

        for part in parts {
            current = current.get(part).ok_or_else(|| {
                StreamError::Deserialization(format!("Path {} not found in JSON", path))
            })?;
        }

        Ok(current.clone())
    }
}

impl Default for PayloadParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Parsed payload with metadata
pub struct ParsedPayload {
    /// Parsed fields
    pub fields: HashMap<String, serde_json::Value>,
    /// Original size in bytes
    pub size: usize,
    /// Parsing timestamp
    pub parsed_at: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json() {
        let parser = PayloadParser::new();
        let payload = br#"{"temperature": 25.5, "humidity": 60}"#;

        let result = parser
            .parse(
                payload,
                &PayloadFormat::Json {
                    schema: None,
                    root_path: None,
                },
            )
            .unwrap();

        assert_eq!(result.get("temperature").unwrap().as_f64().unwrap(), 25.5);
        assert_eq!(result.get("humidity").unwrap().as_i64().unwrap(), 60);
    }

    #[test]
    fn test_parse_plain_text() {
        let parser = PayloadParser::new();
        let payload = b"Hello, MQTT!";

        let result = parser
            .parse(
                payload,
                &PayloadFormat::PlainText {
                    datatype: "xsd:string".to_string(),
                },
            )
            .unwrap();

        assert_eq!(
            result.get("value").unwrap().as_str().unwrap(),
            "Hello, MQTT!"
        );
    }

    #[test]
    fn test_parse_csv() {
        let parser = PayloadParser::new();
        let payload = b"sensor1,25.5,60";

        let result = parser
            .parse(
                payload,
                &PayloadFormat::Csv {
                    delimiter: ',',
                    headers: vec![
                        "device".to_string(),
                        "temp".to_string(),
                        "humidity".to_string(),
                    ],
                    skip_header: false,
                },
            )
            .unwrap();

        assert_eq!(result.get("device").unwrap().as_str().unwrap(), "sensor1");
        assert_eq!(result.get("temp").unwrap().as_str().unwrap(), "25.5");
    }

    #[test]
    fn test_extract_json_path() {
        let parser = PayloadParser::new();
        let json = serde_json::json!({
            "device": {
                "sensor": {
                    "temperature": 25.5
                }
            }
        });

        let result = parser
            .extract_json_path(&json, "device.sensor.temperature")
            .unwrap();
        assert_eq!(result.as_f64().unwrap(), 25.5);
    }
}
