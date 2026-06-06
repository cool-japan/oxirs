//! Main entry point for JSON-LD 1.1 Framing.
//!
//! Implements the [W3C JSON-LD 1.1 Framing](https://www.w3.org/TR/json-ld11-framing/)
//! specification, which allows reshaping a JSON-LD document into a tree structure
//! described by a "frame".
//!
//! # Overview
//!
//! Given an expanded JSON-LD document and a frame object, framing:
//!
//! 1. Builds a flat subject map from the expanded input.
//! 2. Matches subjects against the frame using type, `@id`, and property constraints.
//! 3. Embeds matched subjects according to the configured `EmbedPolicy`, applying
//!    cycle detection via `FramingState`.
//! 4. Optionally prunes properties (`@explicit`) and injects defaults (`@omitDefault`).
//! 5. Returns the framed document as `{"@graph": [...], "@context": {}}`.

use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

use super::framing_embed::{apply_defaults, apply_explicit, embed_subject, EmbedDecision};
use super::framing_match::matches_frame;

// ──────────────────────────────────────────────────────────────────────────────
// Error type
// ──────────────────────────────────────────────────────────────────────────────

/// Errors that may occur during JSON-LD 1.1 Framing.
#[derive(Debug, Error)]
pub enum FramingError {
    #[error("Invalid frame: {0}")]
    InvalidFrame(String),
    #[error("Expansion required before framing: {0}")]
    ExpansionRequired(String),
    #[error("Invalid @embed value: {0}")]
    InvalidEmbedValue(String),
    #[error("Cyclic embed detected for subject: {0}")]
    CyclicEmbed(String),
}

// ──────────────────────────────────────────────────────────────────────────────
// EmbedPolicy
// ──────────────────────────────────────────────────────────────────────────────

/// Controls how matched subjects are embedded in the framed output.
///
/// Corresponds to the JSON-LD 1.1 Framing `@embed` keyword values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbedPolicy {
    /// Embed only the first occurrence of a subject; subsequent references become `@id` links.
    First,
    /// Embed the last occurrence; earlier references become `@id` links.
    Last,
    /// Always embed, but use `@id` links to break detected cycles.
    Always,
    /// Never embed; all subject references are `@id` links.
    Never,
    /// Embed using `@link` references to support lossless round-trips.
    Link,
}

impl Default for EmbedPolicy {
    /// The W3C spec default is `@once`, which is equivalent to `First`.
    fn default() -> Self {
        Self::First
    }
}

impl EmbedPolicy {
    /// Parse from a JSON-LD `@embed` string value.
    pub fn parse_embed_value(s: &str) -> Result<Self, FramingError> {
        match s {
            "@first" | "@once" => Ok(Self::First),
            "@last" => Ok(Self::Last),
            "@always" => Ok(Self::Always),
            "@never" => Ok(Self::Never),
            "@link" => Ok(Self::Link),
            other => Err(FramingError::InvalidEmbedValue(other.to_string())),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// FramingOptions
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for a JSON-LD framing operation.
///
/// These correspond to the processing options defined in the W3C Framing spec
/// (§4.2 *Framing Flags*).
#[derive(Debug, Clone)]
pub struct FramingOptions {
    /// Embedding policy (default: `First`).
    pub embed: EmbedPolicy,
    /// When `true`, only frame-specified properties are included in the output.
    pub explicit: bool,
    /// When `true`, missing properties with `@default` values are omitted.
    pub omit_default: bool,
    /// When `true`, a subject must match **all** frame patterns (logical AND).
    /// When `false` (default), matching **any** frame property pattern is sufficient.
    pub require_all: bool,
    /// When `true`, prune `@none` values from language maps.
    pub pruned_none: bool,
}

impl Default for FramingOptions {
    fn default() -> Self {
        Self {
            embed: EmbedPolicy::First,
            explicit: false,
            omit_default: false,
            require_all: false,
            pruned_none: false,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// FramingState
// ──────────────────────────────────────────────────────────────────────────────

/// Mutable state shared across a single framing operation.
///
/// Tracks which subject IRIs have already been fully embedded to prevent
/// infinite recursion, and maintains `@link` references for the `Link` embed
/// policy.
#[derive(Debug, Default)]
pub struct FramingState {
    /// Subject IRIs that have already been embedded in the current output tree.
    pub embedded: HashSet<String>,
    /// For the `Link` embed policy: maps subject IRI → embedded node object.
    pub link: HashMap<String, Value>,
}

impl FramingState {
    /// Create a fresh framing state.
    pub fn new() -> Self {
        Self {
            embedded: HashSet::new(),
            link: HashMap::new(),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// JsonLdFramer
// ──────────────────────────────────────────────────────────────────────────────

/// Main struct for performing JSON-LD 1.1 Framing.
///
/// # Example
///
/// ```rust
/// use oxirs_core::jsonld::framing::{JsonLdFramer, FramingOptions, EmbedPolicy};
///
/// let options = FramingOptions::default();
/// let framer = JsonLdFramer::new(options);
///
/// let input = serde_json::json!([
///     {"@id": "http://example.org/alice", "@type": ["http://example.org/Person"],
///      "http://example.org/name": [{"@value": "Alice"}]}
/// ]);
/// let frame = serde_json::json!({"@type": "http://example.org/Person"});
///
/// let result = framer.frame(&input, &frame).expect("framing should succeed");
/// assert!(result["@graph"].is_array());
/// ```
#[derive(Debug, Clone)]
pub struct JsonLdFramer {
    options: FramingOptions,
}

impl JsonLdFramer {
    /// Create a new framer with the given options.
    pub fn new(options: FramingOptions) -> Self {
        Self { options }
    }

    /// Perform JSON-LD 1.1 Framing.
    ///
    /// Both `input` and `frame` must already be in **expanded** form (i.e. an
    /// array of subject nodes).  The returned document has the shape:
    ///
    /// ```json
    /// {
    ///   "@graph": [...],
    ///   "@context": {}
    /// }
    /// ```
    ///
    /// The caller is responsible for expansion and compaction if needed.
    pub fn frame(&self, input: &Value, frame: &Value) -> Result<Value, FramingError> {
        // ── Step 1: Build subject map from the expanded input ──────────────────
        let subjects = self.build_subject_map(input)?;

        // ── Step 2: Resolve frame options overrides from the frame object ──────
        let options = self.resolve_frame_options(frame);

        // ── Step 3: Frame subjects ─────────────────────────────────────────────
        let mut state = FramingState::new();
        let framed = self.frame_subjects(&mut state, &subjects, frame, &options)?;

        // ── Step 4: Build the output document ─────────────────────────────────
        Ok(json!({
            "@context": {},
            "@graph": framed
        }))
    }

    /// Build a flat subject map (`@id` → node object) from an expanded
    /// JSON-LD input.  Handles both array-form `[{...}, ...]` and single-
    /// object form `{...}`.
    fn build_subject_map(&self, input: &Value) -> Result<HashMap<String, Value>, FramingError> {
        let mut map: HashMap<String, Value> = HashMap::new();

        let nodes = match input {
            Value::Array(arr) => arr.as_slice(),
            Value::Object(_) => std::slice::from_ref(input),
            Value::Null => return Ok(map),
            _ => {
                return Err(FramingError::ExpansionRequired(
                    "Input must be an expanded JSON-LD array or object".to_string(),
                ))
            }
        };

        for node in nodes {
            self.collect_subjects(node, &mut map);
        }

        Ok(map)
    }

    /// Recursively extract all subject nodes from a node object, populating
    /// `map`.  Nested graph/value objects are traversed but only nodes with
    /// an `@id` are indexed.
    fn collect_subjects(&self, node: &Value, map: &mut HashMap<String, Value>) {
        let obj = match node {
            Value::Object(o) => o,
            Value::Array(arr) => {
                for item in arr {
                    self.collect_subjects(item, map);
                }
                return;
            }
            _ => return,
        };

        // Index if this node has an @id
        if let Some(Value::String(id)) = obj.get("@id") {
            map.entry(id.clone()).or_insert_with(|| node.clone());
        }

        // Recurse into property values and @graph
        for (key, val) in obj {
            if key == "@graph" {
                self.collect_subjects(val, map);
            } else if key.starts_with('@') {
                // skip other keywords
            } else {
                // property values are arrays of nodes/values
                if let Value::Array(vals) = val {
                    for v in vals {
                        if let Value::Object(vo) = v {
                            if !vo.contains_key("@value") {
                                self.collect_subjects(v, map);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Resolve per-operation options from `@embed`, `@explicit`, etc. flags
    /// embedded in the frame object, overlaying the constructor-level options.
    fn resolve_frame_options(&self, frame: &Value) -> FramingOptions {
        let mut opts = self.options.clone();

        if let Value::Object(fobj) = frame {
            if let Some(Value::String(embed_val)) = fobj.get("@embed") {
                if let Ok(policy) = EmbedPolicy::parse_embed_value(embed_val.as_str()) {
                    opts.embed = policy;
                }
            }
            if let Some(Value::Bool(explicit)) = fobj.get("@explicit") {
                opts.explicit = *explicit;
            }
            if let Some(Value::Bool(omit)) = fobj.get("@omitDefault") {
                opts.omit_default = *omit;
            }
            if let Some(Value::Bool(req_all)) = fobj.get("@requireAll") {
                opts.require_all = *req_all;
            }
        }

        opts
    }

    /// Match all subjects against the frame and return the embedded output nodes.
    ///
    /// Implements *§4.5 Framing algorithm* from the W3C spec:
    /// for each subject in the map, test `matches_frame`; if it matches, call
    /// `embed_subject`.
    pub fn frame_subjects(
        &self,
        state: &mut FramingState,
        subjects: &HashMap<String, Value>,
        frame: &Value,
        options: &FramingOptions,
    ) -> Result<Vec<Value>, FramingError> {
        let mut output: Vec<Value> = Vec::new();

        // Collect and sort subject IDs for deterministic ordering
        let mut ids: Vec<&String> = subjects.keys().collect();
        ids.sort();

        for id in ids {
            let subject = subjects.get(id).expect("key from map");

            if !matches_frame(subject, frame, options) {
                continue;
            }

            let embed_decision =
                super::framing_embed::apply_embed_policy(state, id, options.embed.clone());

            match embed_decision {
                EmbedDecision::Skip => {
                    // Emit a bare @id reference
                    output.push(json!({"@id": id}));
                }
                EmbedDecision::Link => {
                    // Reuse the previously computed node via @link
                    if let Some(linked) = state.link.get(id) {
                        output.push(linked.clone());
                    } else {
                        output.push(json!({"@id": id}));
                    }
                }
                EmbedDecision::Full => {
                    state.embedded.insert(id.clone());
                    let mut embedded = embed_subject(state, subject, frame, subjects, options)?;

                    // Apply @explicit pruning
                    embedded = apply_explicit(&embedded, frame, options);

                    // Inject @default values for missing properties
                    apply_defaults(&mut embedded, frame, options);

                    if options.embed == EmbedPolicy::Link {
                        state.link.insert(id.clone(), embedded.clone());
                    }

                    output.push(embedded);
                }
            }
        }

        Ok(output)
    }
}
