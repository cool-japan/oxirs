//! JSON-LD 1.1 writer with configurable output options.
//!
//! Exports: [`Triple`], [`Quad`], [`WriterObject`], [`JsonLdWriter`].

use serde_json::{Map, Value};
use std::collections::HashMap;

use super::jsonld_context::{JsonLdContext, JsonLdError, JsonLdResult};

// ─────────────────────────────────────────────────────────────────────────────
// Writer data types
// ─────────────────────────────────────────────────────────────────────────────

/// A lightweight triple/quad holder for the writer.
#[derive(Debug, Clone)]
pub struct Triple {
    /// Subject IRI or blank node string.
    pub subject: String,
    /// Predicate IRI string.
    pub predicate: String,
    /// Object term.
    pub object: WriterObject,
}

/// A lightweight quad holder for the writer.
#[derive(Debug, Clone)]
pub struct Quad {
    /// Subject IRI or blank node string.
    pub subject: String,
    /// Predicate IRI string.
    pub predicate: String,
    /// Object term.
    pub object: WriterObject,
    /// Named graph IRI or blank node string.
    pub graph: Option<String>,
}

/// Object in a triple or quad for the writer.
#[derive(Debug, Clone)]
pub enum WriterObject {
    /// An IRI node.
    Iri(String),
    /// A blank node.
    BlankNode(String),
    /// A plain literal.
    Literal(String),
    /// A typed literal.
    TypedLiteral(String, String),
    /// A language-tagged literal.
    LangLiteral(String, String),
}

impl WriterObject {
    pub(crate) fn to_json_ld_value(&self, ctx: Option<&JsonLdContext>) -> Value {
        match self {
            Self::Iri(iri) => {
                let compacted = ctx
                    .map(|c| c.compact_iri(iri))
                    .unwrap_or_else(|| iri.clone());
                serde_json::json!({ "@id": compacted })
            }
            Self::BlankNode(id) => serde_json::json!({ "@id": id }),
            Self::Literal(s) => serde_json::json!({ "@value": s }),
            Self::TypedLiteral(s, dt) => serde_json::json!({ "@value": s, "@type": dt }),
            Self::LangLiteral(s, lang) => serde_json::json!({ "@value": s, "@language": lang }),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JsonLdWriter
// ─────────────────────────────────────────────────────────────────────────────

/// JSON-LD 1.1 writer with configurable output options.
pub struct JsonLdWriter {
    /// Optional JSON-LD context to embed in the output.
    pub context: Option<Value>,
    /// Whether to compact IRIs using the context.
    pub compact: bool,
    /// Whether to pretty-print the JSON output.
    pub pretty: bool,
}

impl Default for JsonLdWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonLdWriter {
    /// Create a new writer with defaults (expanded, not pretty-printed).
    pub fn new() -> Self {
        Self {
            context: None,
            compact: false,
            pretty: false,
        }
    }

    /// Attach a JSON-LD context to the writer output.
    pub fn with_context(mut self, context: Value) -> Self {
        self.context = Some(context);
        self
    }

    /// Enable compaction of IRIs using the attached context.
    pub fn compact_mode(mut self) -> Self {
        self.compact = true;
        self
    }

    /// Enable pretty-printing (indented JSON).
    pub fn pretty_print(mut self) -> Self {
        self.pretty = true;
        self
    }

    /// Write a collection of triples as a JSON-LD document string.
    pub fn write_triples(&self, triples: &[Triple]) -> JsonLdResult<String> {
        let ctx_parsed = self
            .context
            .as_ref()
            .map(JsonLdContext::parse)
            .transpose()?;
        let ctx_ref = if self.compact {
            ctx_parsed.as_ref()
        } else {
            None
        };

        // Group by subject
        let mut subject_map: HashMap<String, Map<String, Value>> = HashMap::new();
        for triple in triples {
            let subj_compacted = ctx_ref
                .map(|c| c.compact_iri(&triple.subject))
                .unwrap_or_else(|| triple.subject.clone());

            let entry = subject_map
                .entry(triple.subject.clone())
                .or_insert_with(|| {
                    let mut m = Map::new();
                    m.insert("@id".into(), Value::String(subj_compacted.clone()));
                    m
                });

            let pred_key = ctx_ref
                .map(|c| c.compact_iri(&triple.predicate))
                .unwrap_or_else(|| triple.predicate.clone());

            let obj_value = triple.object.to_json_ld_value(ctx_ref);
            let values = entry
                .entry(pred_key)
                .or_insert_with(|| Value::Array(vec![]));
            if let Value::Array(arr) = values {
                arr.push(obj_value);
            }
        }

        let graph: Vec<Value> = subject_map.into_values().map(Value::Object).collect();
        let mut doc = Map::new();
        if let Some(ctx) = &self.context {
            doc.insert("@context".into(), ctx.clone());
        }
        doc.insert("@graph".into(), Value::Array(graph));

        self.serialize_json(&Value::Object(doc))
    }

    /// Write a collection of quads as a JSON-LD document string.
    pub fn write_quads(&self, quads: &[Quad]) -> JsonLdResult<String> {
        let ctx_parsed = self
            .context
            .as_ref()
            .map(JsonLdContext::parse)
            .transpose()?;
        let ctx_ref = if self.compact {
            ctx_parsed.as_ref()
        } else {
            None
        };

        // Group quads by graph, then by subject
        let mut graph_map: HashMap<String, HashMap<String, Map<String, Value>>> = HashMap::new();

        for quad in quads {
            let graph_key = quad.graph.clone().unwrap_or_else(|| "@default".into());
            let subj_compacted = ctx_ref
                .map(|c| c.compact_iri(&quad.subject))
                .unwrap_or_else(|| quad.subject.clone());

            let graph_entry = graph_map.entry(graph_key.clone()).or_default();
            let entry = graph_entry.entry(quad.subject.clone()).or_insert_with(|| {
                let mut m = Map::new();
                m.insert("@id".into(), Value::String(subj_compacted.clone()));
                m
            });

            let pred_key = ctx_ref
                .map(|c| c.compact_iri(&quad.predicate))
                .unwrap_or_else(|| quad.predicate.clone());

            let obj_value = quad.object.to_json_ld_value(ctx_ref);
            let values = entry
                .entry(pred_key)
                .or_insert_with(|| Value::Array(vec![]));
            if let Value::Array(arr) = values {
                arr.push(obj_value);
            }
        }

        // Build document
        let default_nodes = graph_map
            .remove("@default")
            .map(|m| m.into_values().map(Value::Object).collect::<Vec<_>>())
            .unwrap_or_default();

        let mut named_graphs: Vec<Value> = graph_map
            .into_iter()
            .map(|(graph_id, node_map)| {
                let nodes: Vec<Value> = node_map.into_values().map(Value::Object).collect();
                let compact_graph_id = ctx_ref
                    .map(|c| c.compact_iri(&graph_id))
                    .unwrap_or(graph_id);
                serde_json::json!({
                    "@id": compact_graph_id,
                    "@graph": nodes
                })
            })
            .collect();

        let mut all_nodes = default_nodes;
        all_nodes.append(&mut named_graphs);

        let mut doc = Map::new();
        if let Some(ctx) = &self.context {
            doc.insert("@context".into(), ctx.clone());
        }
        doc.insert("@graph".into(), Value::Array(all_nodes));

        self.serialize_json(&Value::Object(doc))
    }

    fn serialize_json(&self, value: &Value) -> JsonLdResult<String> {
        if self.pretty {
            serde_json::to_string_pretty(value).map_err(JsonLdError::Json)
        } else {
            serde_json::to_string(value).map_err(JsonLdError::Json)
        }
    }
}
