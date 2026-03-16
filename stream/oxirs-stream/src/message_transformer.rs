//! # Message Transformer
//!
//! Message format transformation pipeline for schema evolution and format conversion.
//!
//! Provides a flexible pipeline system to map fields between different message formats
//! (JSON, Avro, Protobuf, CSV, Raw) while applying per-field string transformations.

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// Supported message serialisation formats.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MessageFormat {
    Json,
    Avro,
    Protobuf,
    Csv,
    Raw,
}

/// Elementary string transformation applied to a single field value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformFn {
    /// Convert value to uppercase.
    ToUpper,
    /// Convert value to lowercase.
    ToLower,
    /// Trim leading and trailing whitespace.
    Trim,
    /// Prepend a fixed string.
    Prefix(String),
    /// Append a fixed string.
    Suffix(String),
    /// Replace all occurrences of `from` with `to`.
    Replace { from: String, to: String },
    /// Pass value through unchanged.
    Identity,
}

impl TransformFn {
    /// Apply this transformation to `value` and return the result.
    pub fn apply(&self, value: &str) -> String {
        match self {
            TransformFn::ToUpper => value.to_uppercase(),
            TransformFn::ToLower => value.to_lowercase(),
            TransformFn::Trim => value.trim().to_string(),
            TransformFn::Prefix(prefix) => format!("{}{}", prefix, value),
            TransformFn::Suffix(suffix) => format!("{}{}", value, suffix),
            TransformFn::Replace { from, to } => value.replace(from.as_str(), to.as_str()),
            TransformFn::Identity => value.to_string(),
        }
    }
}

/// Mapping from one field to another with an optional transformation.
#[derive(Debug, Clone)]
pub struct FieldMapping {
    /// Name of the field in the source payload.
    pub source_field: String,
    /// Name of the field in the target payload.
    pub target_field: String,
    /// Optional transformation applied to the value.
    pub transform: Option<TransformFn>,
}

impl FieldMapping {
    /// Create a new field mapping without a transformation.
    pub fn new(source_field: impl Into<String>, target_field: impl Into<String>) -> Self {
        Self {
            source_field: source_field.into(),
            target_field: target_field.into(),
            transform: None,
        }
    }

    /// Create a new field mapping with a transformation.
    pub fn with_transform(
        source_field: impl Into<String>,
        target_field: impl Into<String>,
        transform: TransformFn,
    ) -> Self {
        Self {
            source_field: source_field.into(),
            target_field: target_field.into(),
            transform: Some(transform),
        }
    }
}

/// A named transformation pipeline: ordered field mappings plus format metadata.
#[derive(Debug, Clone)]
pub struct TransformPipeline {
    /// Ordered list of field mappings to apply.
    pub mappings: Vec<FieldMapping>,
    /// Expected format of incoming payloads.
    pub source_format: MessageFormat,
    /// Format of outgoing payloads.
    pub target_format: MessageFormat,
}

impl TransformPipeline {
    /// Create a new pipeline.
    pub fn new(
        source_format: MessageFormat,
        target_format: MessageFormat,
        mappings: Vec<FieldMapping>,
    ) -> Self {
        Self {
            mappings,
            source_format,
            target_format,
        }
    }
}

/// A keyed collection of fields with an associated format tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessagePayload {
    /// Field name → value pairs.
    pub fields: HashMap<String, String>,
    /// Format tag for this payload.
    pub format: MessageFormat,
}

impl MessagePayload {
    /// Create a new payload with the given fields and format.
    pub fn new(fields: HashMap<String, String>, format: MessageFormat) -> Self {
        Self { fields, format }
    }
}

/// Errors that can occur during transformation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformError {
    /// No pipeline registered under the given name.
    PipelineNotFound(String),
    /// A required source field was absent from the payload.
    FieldNotFound(String),
    /// The transformation step itself produced an error (reserved for future use).
    TransformFailed(String),
}

impl std::fmt::Display for TransformError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformError::PipelineNotFound(name) => {
                write!(f, "pipeline not found: {}", name)
            }
            TransformError::FieldNotFound(field) => {
                write!(f, "source field not found: {}", field)
            }
            TransformError::TransformFailed(msg) => {
                write!(f, "transform failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for TransformError {}

// ────────────────────────────────────────────────────────────────────────────
// MessageTransformer
// ────────────────────────────────────────────────────────────────────────────

/// Registry of named transformation pipelines.
///
/// # Example
/// ```rust
/// use oxirs_stream::message_transformer::*;
/// use std::collections::HashMap;
///
/// let mut transformer = MessageTransformer::new();
/// let pipeline = TransformPipeline::new(
///     MessageFormat::Json,
///     MessageFormat::Avro,
///     vec![FieldMapping::with_transform("name", "NAME", TransformFn::ToUpper)],
/// );
/// transformer.add_pipeline("upper", pipeline);
///
/// let mut fields = HashMap::new();
/// fields.insert("name".to_string(), "alice".to_string());
/// let payload = MessagePayload::new(fields, MessageFormat::Json);
/// let result = transformer.transform("upper", payload)?;
/// assert_eq!(result.fields["NAME"], "ALICE");
/// # Ok::<(), TransformError>(())
/// ```
#[derive(Debug, Default)]
pub struct MessageTransformer {
    pipelines: HashMap<String, TransformPipeline>,
}

impl MessageTransformer {
    /// Create an empty transformer.
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
        }
    }

    /// Register a named pipeline. Overwrites any existing pipeline with the same name.
    pub fn add_pipeline(&mut self, name: &str, pipeline: TransformPipeline) {
        self.pipelines.insert(name.to_string(), pipeline);
    }

    /// Apply the named pipeline to `payload`.
    ///
    /// Each [`FieldMapping`] in the pipeline is applied in order:
    /// 1. The value of `source_field` is read from the payload.
    /// 2. The optional [`TransformFn`] is applied.
    /// 3. The result is stored under `target_field` in the output payload.
    ///
    /// Fields present in the payload but not mentioned by any mapping are dropped.
    pub fn transform(
        &self,
        pipeline_name: &str,
        payload: MessagePayload,
    ) -> Result<MessagePayload, TransformError> {
        let pipeline = self
            .pipelines
            .get(pipeline_name)
            .ok_or_else(|| TransformError::PipelineNotFound(pipeline_name.to_string()))?;

        let mut output_fields: HashMap<String, String> = HashMap::new();

        for mapping in &pipeline.mappings {
            let value = payload
                .fields
                .get(&mapping.source_field)
                .ok_or_else(|| TransformError::FieldNotFound(mapping.source_field.clone()))?;

            let transformed = match &mapping.transform {
                Some(tf) => tf.apply(value),
                None => value.clone(),
            };

            output_fields.insert(mapping.target_field.clone(), transformed);
        }

        Ok(MessagePayload {
            fields: output_fields,
            format: pipeline.target_format.clone(),
        })
    }

    /// Apply a sequence of pipelines in order.
    ///
    /// The output of each pipeline is fed as the input to the next. The
    /// format of the intermediate payloads is updated by each pipeline.
    pub fn chain_pipelines(
        &self,
        names: &[&str],
        payload: MessagePayload,
    ) -> Result<MessagePayload, TransformError> {
        let mut current = payload;
        for name in names {
            current = self.transform(name, current)?;
        }
        Ok(current)
    }

    /// Return the names of all registered pipelines (order unspecified).
    pub fn list_pipelines(&self) -> Vec<&str> {
        self.pipelines.keys().map(|k| k.as_str()).collect()
    }

    /// Return the number of registered pipelines.
    pub fn pipeline_count(&self) -> usize {
        self.pipelines.len()
    }

    /// Check whether a pipeline with the given name exists.
    pub fn has_pipeline(&self, name: &str) -> bool {
        self.pipelines.contains_key(name)
    }

    /// Remove a pipeline by name. Returns `true` if it was present.
    pub fn remove_pipeline(&mut self, name: &str) -> bool {
        self.pipelines.remove(name).is_some()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_payload(fields: &[(&str, &str)], format: MessageFormat) -> MessagePayload {
        let map = fields
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        MessagePayload::new(map, format)
    }

    fn simple_transformer(
        name: &str,
        src_field: &str,
        tgt_field: &str,
        tf: TransformFn,
        src_fmt: MessageFormat,
        tgt_fmt: MessageFormat,
    ) -> MessageTransformer {
        let mut t = MessageTransformer::new();
        let mapping = FieldMapping::with_transform(src_field, tgt_field, tf);
        let pipeline = TransformPipeline::new(src_fmt, tgt_fmt, vec![mapping]);
        t.add_pipeline(name, pipeline);
        t
    }

    // ── TransformFn::ToUpper ──────────────────────────────────────────────

    #[test]
    fn test_to_upper_basic() {
        let tf = TransformFn::ToUpper;
        assert_eq!(tf.apply("hello"), "HELLO");
    }

    #[test]
    fn test_to_upper_already_upper() {
        assert_eq!(TransformFn::ToUpper.apply("WORLD"), "WORLD");
    }

    #[test]
    fn test_to_upper_mixed() {
        assert_eq!(TransformFn::ToUpper.apply("HeLLo WoRLd"), "HELLO WORLD");
    }

    #[test]
    fn test_to_upper_empty() {
        assert_eq!(TransformFn::ToUpper.apply(""), "");
    }

    // ── TransformFn::ToLower ──────────────────────────────────────────────

    #[test]
    fn test_to_lower_basic() {
        assert_eq!(TransformFn::ToLower.apply("HELLO"), "hello");
    }

    #[test]
    fn test_to_lower_already_lower() {
        assert_eq!(TransformFn::ToLower.apply("world"), "world");
    }

    #[test]
    fn test_to_lower_mixed() {
        assert_eq!(TransformFn::ToLower.apply("HeLLo"), "hello");
    }

    // ── TransformFn::Trim ────────────────────────────────────────────────

    #[test]
    fn test_trim_leading() {
        assert_eq!(TransformFn::Trim.apply("  hello"), "hello");
    }

    #[test]
    fn test_trim_trailing() {
        assert_eq!(TransformFn::Trim.apply("hello  "), "hello");
    }

    #[test]
    fn test_trim_both() {
        assert_eq!(TransformFn::Trim.apply("  hello  "), "hello");
    }

    #[test]
    fn test_trim_no_whitespace() {
        assert_eq!(TransformFn::Trim.apply("hello"), "hello");
    }

    #[test]
    fn test_trim_only_whitespace() {
        assert_eq!(TransformFn::Trim.apply("   "), "");
    }

    // ── TransformFn::Prefix ───────────────────────────────────────────────

    #[test]
    fn test_prefix_basic() {
        assert_eq!(
            TransformFn::Prefix("pre_".to_string()).apply("value"),
            "pre_value"
        );
    }

    #[test]
    fn test_prefix_empty_value() {
        assert_eq!(TransformFn::Prefix("pre_".to_string()).apply(""), "pre_");
    }

    #[test]
    fn test_prefix_empty_prefix() {
        assert_eq!(TransformFn::Prefix(String::new()).apply("value"), "value");
    }

    // ── TransformFn::Suffix ───────────────────────────────────────────────

    #[test]
    fn test_suffix_basic() {
        assert_eq!(
            TransformFn::Suffix("_suf".to_string()).apply("value"),
            "value_suf"
        );
    }

    #[test]
    fn test_suffix_empty_value() {
        assert_eq!(TransformFn::Suffix("_end".to_string()).apply(""), "_end");
    }

    #[test]
    fn test_suffix_empty_suffix() {
        assert_eq!(TransformFn::Suffix(String::new()).apply("value"), "value");
    }

    // ── TransformFn::Replace ──────────────────────────────────────────────

    #[test]
    fn test_replace_basic() {
        let tf = TransformFn::Replace {
            from: "foo".to_string(),
            to: "bar".to_string(),
        };
        assert_eq!(tf.apply("foo baz foo"), "bar baz bar");
    }

    #[test]
    fn test_replace_no_match() {
        let tf = TransformFn::Replace {
            from: "x".to_string(),
            to: "y".to_string(),
        };
        assert_eq!(tf.apply("hello"), "hello");
    }

    #[test]
    fn test_replace_empty_from() {
        // Replacing empty string inserts `to` between every character.
        let tf = TransformFn::Replace {
            from: String::new(),
            to: "-".to_string(),
        };
        // Standard Rust behaviour: replaces at every position.
        let result = tf.apply("ab");
        assert!(result.contains('-'));
    }

    #[test]
    fn test_replace_to_empty() {
        let tf = TransformFn::Replace {
            from: "o".to_string(),
            to: String::new(),
        };
        assert_eq!(tf.apply("foobar"), "fbar");
    }

    // ── TransformFn::Identity ─────────────────────────────────────────────

    #[test]
    fn test_identity_passthrough() {
        assert_eq!(TransformFn::Identity.apply("unchanged"), "unchanged");
    }

    #[test]
    fn test_identity_empty() {
        assert_eq!(TransformFn::Identity.apply(""), "");
    }

    // ── MessageTransformer::add_pipeline / has_pipeline / list_pipelines ──

    #[test]
    fn test_add_and_has_pipeline() {
        let mut t = MessageTransformer::new();
        assert!(!t.has_pipeline("p1"));
        let pipeline = TransformPipeline::new(MessageFormat::Json, MessageFormat::Csv, vec![]);
        t.add_pipeline("p1", pipeline);
        assert!(t.has_pipeline("p1"));
    }

    #[test]
    fn test_list_pipelines_empty() {
        let t = MessageTransformer::new();
        assert!(t.list_pipelines().is_empty());
    }

    #[test]
    fn test_list_pipelines_multiple() {
        let mut t = MessageTransformer::new();
        for name in ["a", "b", "c"] {
            let p = TransformPipeline::new(MessageFormat::Raw, MessageFormat::Raw, vec![]);
            t.add_pipeline(name, p);
        }
        let mut names = t.list_pipelines();
        names.sort();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_pipeline_count() {
        let mut t = MessageTransformer::new();
        assert_eq!(t.pipeline_count(), 0);
        t.add_pipeline(
            "x",
            TransformPipeline::new(MessageFormat::Json, MessageFormat::Json, vec![]),
        );
        assert_eq!(t.pipeline_count(), 1);
    }

    #[test]
    fn test_remove_pipeline() {
        let mut t = MessageTransformer::new();
        t.add_pipeline(
            "rm",
            TransformPipeline::new(MessageFormat::Json, MessageFormat::Json, vec![]),
        );
        assert!(t.remove_pipeline("rm"));
        assert!(!t.has_pipeline("rm"));
        assert!(!t.remove_pipeline("rm")); // second remove returns false
    }

    // ── MessageTransformer::transform ────────────────────────────────────

    #[test]
    fn test_transform_basic_field_rename() {
        let mut t = MessageTransformer::new();
        let mapping = FieldMapping::new("src", "dst");
        let pipeline =
            TransformPipeline::new(MessageFormat::Json, MessageFormat::Avro, vec![mapping]);
        t.add_pipeline("rename", pipeline);

        let payload = make_payload(&[("src", "value")], MessageFormat::Json);
        let result = t.transform("rename", payload).expect("should succeed");
        assert_eq!(result.fields.get("dst"), Some(&"value".to_string()));
        assert!(!result.fields.contains_key("src"));
        assert_eq!(result.format, MessageFormat::Avro);
    }

    #[test]
    fn test_transform_to_upper() {
        let t = simple_transformer(
            "p",
            "name",
            "NAME",
            TransformFn::ToUpper,
            MessageFormat::Json,
            MessageFormat::Json,
        );
        let payload = make_payload(&[("name", "alice")], MessageFormat::Json);
        let result = t.transform("p", payload).expect("ok");
        assert_eq!(result.fields["NAME"], "ALICE");
    }

    #[test]
    fn test_transform_prefix() {
        let t = simple_transformer(
            "p",
            "id",
            "id",
            TransformFn::Prefix("usr_".to_string()),
            MessageFormat::Raw,
            MessageFormat::Raw,
        );
        let payload = make_payload(&[("id", "42")], MessageFormat::Raw);
        let result = t.transform("p", payload).expect("ok");
        assert_eq!(result.fields["id"], "usr_42");
    }

    #[test]
    fn test_transform_suffix() {
        let t = simple_transformer(
            "p",
            "tag",
            "tag",
            TransformFn::Suffix("_v2".to_string()),
            MessageFormat::Csv,
            MessageFormat::Csv,
        );
        let payload = make_payload(&[("tag", "sensor")], MessageFormat::Csv);
        let result = t.transform("p", payload).expect("ok");
        assert_eq!(result.fields["tag"], "sensor_v2");
    }

    #[test]
    fn test_transform_replace() {
        let t = simple_transformer(
            "p",
            "path",
            "path",
            TransformFn::Replace {
                from: "/".to_string(),
                to: ".".to_string(),
            },
            MessageFormat::Json,
            MessageFormat::Json,
        );
        let payload = make_payload(&[("path", "a/b/c")], MessageFormat::Json);
        let result = t.transform("p", payload).expect("ok");
        assert_eq!(result.fields["path"], "a.b.c");
    }

    #[test]
    fn test_transform_pipeline_not_found() {
        let t = MessageTransformer::new();
        let payload = make_payload(&[], MessageFormat::Json);
        let err = t.transform("missing", payload).unwrap_err();
        assert_eq!(err, TransformError::PipelineNotFound("missing".to_string()));
    }

    #[test]
    fn test_transform_field_not_found() {
        let t = simple_transformer(
            "p",
            "nonexistent",
            "out",
            TransformFn::Identity,
            MessageFormat::Json,
            MessageFormat::Json,
        );
        let payload = make_payload(&[("other", "val")], MessageFormat::Json);
        let err = t.transform("p", payload).unwrap_err();
        assert_eq!(
            err,
            TransformError::FieldNotFound("nonexistent".to_string())
        );
    }

    #[test]
    fn test_transform_empty_pipeline_empty_output() {
        let mut t = MessageTransformer::new();
        let pipeline = TransformPipeline::new(MessageFormat::Json, MessageFormat::Avro, vec![]);
        t.add_pipeline("empty", pipeline);

        let payload = make_payload(&[("a", "1"), ("b", "2")], MessageFormat::Json);
        let result = t.transform("empty", payload).expect("ok");
        assert!(result.fields.is_empty());
        assert_eq!(result.format, MessageFormat::Avro);
    }

    #[test]
    fn test_transform_multiple_mappings() {
        let mut t = MessageTransformer::new();
        let mappings = vec![
            FieldMapping::with_transform("first", "FIRST", TransformFn::ToUpper),
            FieldMapping::with_transform("last", "LAST", TransformFn::ToLower),
            FieldMapping::new("email", "email_address"),
        ];
        let pipeline = TransformPipeline::new(MessageFormat::Json, MessageFormat::Json, mappings);
        t.add_pipeline("multi", pipeline);

        let payload = make_payload(
            &[("first", "Alice"), ("last", "SMITH"), ("email", "a@b.com")],
            MessageFormat::Json,
        );
        let result = t.transform("multi", payload).expect("ok");
        assert_eq!(result.fields["FIRST"], "ALICE");
        assert_eq!(result.fields["LAST"], "smith");
        assert_eq!(result.fields["email_address"], "a@b.com");
    }

    // ── chain_pipelines ───────────────────────────────────────────────────

    #[test]
    fn test_chain_single_pipeline() {
        let t = simple_transformer(
            "p",
            "v",
            "v",
            TransformFn::Trim,
            MessageFormat::Raw,
            MessageFormat::Raw,
        );
        let payload = make_payload(&[("v", "  hello  ")], MessageFormat::Raw);
        let result = t.chain_pipelines(&["p"], payload).expect("ok");
        assert_eq!(result.fields["v"], "hello");
    }

    #[test]
    fn test_chain_two_pipelines() {
        let mut t = MessageTransformer::new();

        // Pipeline 1: rename src→mid and uppercase
        let p1 = TransformPipeline::new(
            MessageFormat::Json,
            MessageFormat::Json,
            vec![FieldMapping::with_transform(
                "src",
                "mid",
                TransformFn::ToUpper,
            )],
        );
        // Pipeline 2: rename mid→dst and add prefix
        let p2 = TransformPipeline::new(
            MessageFormat::Json,
            MessageFormat::Avro,
            vec![FieldMapping::with_transform(
                "mid",
                "dst",
                TransformFn::Prefix(">>".to_string()),
            )],
        );
        t.add_pipeline("p1", p1);
        t.add_pipeline("p2", p2);

        let payload = make_payload(&[("src", "hello")], MessageFormat::Json);
        let result = t.chain_pipelines(&["p1", "p2"], payload).expect("ok");
        assert_eq!(result.fields["dst"], ">>HELLO");
        assert_eq!(result.format, MessageFormat::Avro);
    }

    #[test]
    fn test_chain_three_pipelines() {
        let mut t = MessageTransformer::new();

        let p1 = TransformPipeline::new(
            MessageFormat::Raw,
            MessageFormat::Raw,
            vec![FieldMapping::with_transform("a", "b", TransformFn::Trim)],
        );
        let p2 = TransformPipeline::new(
            MessageFormat::Raw,
            MessageFormat::Raw,
            vec![FieldMapping::with_transform("b", "c", TransformFn::ToUpper)],
        );
        let p3 = TransformPipeline::new(
            MessageFormat::Raw,
            MessageFormat::Csv,
            vec![FieldMapping::with_transform(
                "c",
                "d",
                TransformFn::Suffix("!".to_string()),
            )],
        );
        t.add_pipeline("p1", p1);
        t.add_pipeline("p2", p2);
        t.add_pipeline("p3", p3);

        let payload = make_payload(&[("a", "  hi  ")], MessageFormat::Raw);
        let result = t.chain_pipelines(&["p1", "p2", "p3"], payload).expect("ok");
        assert_eq!(result.fields["d"], "HI!");
        assert_eq!(result.format, MessageFormat::Csv);
    }

    #[test]
    fn test_chain_empty_names() {
        let t = MessageTransformer::new();
        let payload = make_payload(&[("x", "y")], MessageFormat::Json);
        let result = t.chain_pipelines(&[], payload.clone()).expect("ok");
        assert_eq!(result, payload);
    }

    #[test]
    fn test_chain_missing_pipeline_error() {
        let t = MessageTransformer::new();
        let payload = make_payload(&[("x", "y")], MessageFormat::Json);
        let err = t.chain_pipelines(&["missing"], payload).unwrap_err();
        assert!(matches!(err, TransformError::PipelineNotFound(_)));
    }

    // ── Error Display ─────────────────────────────────────────────────────

    #[test]
    fn test_error_display_pipeline_not_found() {
        let e = TransformError::PipelineNotFound("p".to_string());
        assert!(e.to_string().contains("pipeline not found"));
        assert!(e.to_string().contains("p"));
    }

    #[test]
    fn test_error_display_field_not_found() {
        let e = TransformError::FieldNotFound("f".to_string());
        assert!(e.to_string().contains("source field not found"));
        assert!(e.to_string().contains("f"));
    }

    #[test]
    fn test_error_display_transform_failed() {
        let e = TransformError::TransformFailed("boom".to_string());
        assert!(e.to_string().contains("transform failed"));
        assert!(e.to_string().contains("boom"));
    }

    // ── MessageFormat ─────────────────────────────────────────────────────

    #[test]
    fn test_message_format_equality() {
        assert_eq!(MessageFormat::Json, MessageFormat::Json);
        assert_ne!(MessageFormat::Json, MessageFormat::Avro);
    }

    #[test]
    fn test_all_message_formats_exist() {
        let formats = [
            MessageFormat::Json,
            MessageFormat::Avro,
            MessageFormat::Protobuf,
            MessageFormat::Csv,
            MessageFormat::Raw,
        ];
        assert_eq!(formats.len(), 5);
    }

    // ── FieldMapping constructors ─────────────────────────────────────────

    #[test]
    fn test_field_mapping_new_no_transform() {
        let m = FieldMapping::new("src", "dst");
        assert_eq!(m.source_field, "src");
        assert_eq!(m.target_field, "dst");
        assert!(m.transform.is_none());
    }

    #[test]
    fn test_field_mapping_with_transform() {
        let m = FieldMapping::with_transform("s", "t", TransformFn::ToUpper);
        assert!(m.transform.is_some());
    }

    // ── Overwrite existing pipeline ───────────────────────────────────────

    #[test]
    fn test_add_pipeline_overwrites() {
        let mut t = MessageTransformer::new();
        let p1 = TransformPipeline::new(
            MessageFormat::Json,
            MessageFormat::Json,
            vec![FieldMapping::new("a", "b")],
        );
        let p2 = TransformPipeline::new(
            MessageFormat::Json,
            MessageFormat::Avro,
            vec![FieldMapping::new("a", "c")],
        );
        t.add_pipeline("p", p1);
        t.add_pipeline("p", p2);
        // After overwrite, mapping goes to "c"
        let payload = make_payload(&[("a", "v")], MessageFormat::Json);
        let result = t.transform("p", payload).expect("ok");
        assert!(result.fields.contains_key("c"));
        assert!(!result.fields.contains_key("b"));
    }

    // ── Trim + ToLower combined via chain ─────────────────────────────────

    #[test]
    fn test_chain_trim_then_lower() {
        let mut t = MessageTransformer::new();
        let p1 = TransformPipeline::new(
            MessageFormat::Raw,
            MessageFormat::Raw,
            vec![FieldMapping::with_transform("x", "x", TransformFn::Trim)],
        );
        let p2 = TransformPipeline::new(
            MessageFormat::Raw,
            MessageFormat::Raw,
            vec![FieldMapping::with_transform("x", "x", TransformFn::ToLower)],
        );
        t.add_pipeline("trim", p1);
        t.add_pipeline("lower", p2);

        let payload = make_payload(&[("x", "  HELLO  ")], MessageFormat::Raw);
        let result = t.chain_pipelines(&["trim", "lower"], payload).expect("ok");
        assert_eq!(result.fields["x"], "hello");
    }

    // ── Idempotency of Identity transform ─────────────────────────────────

    #[test]
    fn test_identity_in_pipeline() {
        let mut t = MessageTransformer::new();
        let p = TransformPipeline::new(
            MessageFormat::Json,
            MessageFormat::Json,
            vec![FieldMapping::with_transform(
                "k",
                "k",
                TransformFn::Identity,
            )],
        );
        t.add_pipeline("id", p);
        let payload = make_payload(&[("k", "unchanged_value_123")], MessageFormat::Json);
        let result = t.transform("id", payload).expect("ok");
        assert_eq!(result.fields["k"], "unchanged_value_123");
    }
}
