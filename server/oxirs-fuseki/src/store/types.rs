//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

/// Query execution statistics
#[derive(Debug, Clone, Serialize)]
pub struct QueryStats {
    pub execution_time: Duration,
    pub result_count: usize,
    pub query_type: String,
    pub success: bool,
    pub error_message: Option<String>,
}
/// RDF serialization formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RdfSerializationFormat {
    Turtle,
    NTriples,
    RdfXml,
    JsonLd,
    NQuads,
}
/// Represents a change in the store for WebSocket notifications
#[derive(Debug, Clone, Serialize)]
pub struct StoreChange {
    pub id: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub operation_type: String,
    pub affected_graphs: Vec<String>,
    pub triple_count: usize,
    pub dataset_name: Option<String>,
}
/// Parameters for change detection
#[derive(Debug, Clone)]
pub struct ChangeDetectionParams {
    pub since: chrono::DateTime<chrono::Utc>,
    pub graphs: Option<Vec<String>>,
    pub operation_types: Option<Vec<String>>,
    pub limit: Option<usize>,
}
/// Store metadata and statistics
#[derive(Debug, Clone, Default)]
pub(super) struct StoreMetadata {
    pub(super) created_at: Option<Instant>,
    pub(super) last_modified: Option<Instant>,
    pub(super) total_queries: u64,
    pub(super) total_updates: u64,
    pub(super) query_cache_hits: u64,
    pub(super) query_cache_misses: u64,
    pub(super) change_log: Vec<StoreChange>,
    pub(super) last_change_id: u64,
}
/// Query result wrapper with statistics
#[derive(Debug)]
pub struct QueryResult {
    pub inner: CoreQueryResult,
    pub stats: QueryStats,
}
impl QueryResult {
    /// Convert to JSON string based on result type
    pub fn to_json(&self) -> FusekiResult<String> {
        match &self.inner {
            CoreQueryResult::Select {
                variables,
                bindings,
            } => {
                let json_result = serde_json::json!(
                    { "head" : { "vars" : variables }, "results" : { "bindings" :
                    bindings.iter().map(| binding | { binding.iter().map(| (k, v) | { (k
                    .clone(), serde_json::json!({ "type" : match v { Term::NamedNode(_)
                    => "uri", Term::BlankNode(_) => "bnode", Term::Literal(_) =>
                    "literal", Term::Variable(_) => "variable", Term::QuotedTriple(_) =>
                    "quotedTriple", }, "value" : v.to_string() })) }).collect::<
                    serde_json::Map < String, serde_json::Value >> () }).collect::< Vec <
                    _ >> () } }
                );
                Ok(json_result.to_string())
            }
            CoreQueryResult::Ask(result) => {
                let json_result = serde_json::json!({ "head" : {}, "boolean" : result });
                Ok(json_result.to_string())
            }
            CoreQueryResult::Construct(_) => Err(FusekiError::unsupported_media_type(
                "CONSTRUCT queries should use RDF format, not JSON",
            )),
        }
    }
    /// Convert to XML string
    pub fn to_xml(&self) -> FusekiResult<String> {
        match &self.inner {
            CoreQueryResult::Select {
                variables,
                bindings,
            } => {
                let mut xml = String::from(
                    "<?xml version=\"1.0\"?>\n<sparql xmlns=\"http://www.w3.org/2005/sparql-results#\">\n",
                );
                xml.push_str("  <head>\n");
                for var in variables {
                    xml.push_str(&format!("    <variable name=\"{var}\"/>\n"));
                }
                xml.push_str("  </head>\n  <results>\n");
                for binding in bindings {
                    xml.push_str("    <result>\n");
                    for (var, term) in binding {
                        xml.push_str(&format!("      <binding name=\"{var}\">\n"));
                        match term {
                            Term::NamedNode(node) => {
                                xml.push_str(&format!("        <uri>{}</uri>\n", node.as_str()));
                            }
                            Term::BlankNode(node) => {
                                xml.push_str(&format!(
                                    "        <bnode>{}</bnode>\n",
                                    node.as_str()
                                ));
                            }
                            Term::Literal(literal) => {
                                xml.push_str(&format!(
                                    "        <literal>{}</literal>\n",
                                    literal.value()
                                ));
                            }
                            Term::Variable(variable) => {
                                xml.push_str(&format!(
                                    "        <variable>{}</variable>\n",
                                    variable.as_str()
                                ));
                            }
                            Term::QuotedTriple(triple) => {
                                xml.push_str(
                                    &format!(
                                        "        <quotedTriple>&lt;&lt;{} {} {}&gt;&gt;</quotedTriple>\n",
                                        triple.subject(), triple.predicate(), triple.object()
                                    ),
                                );
                            }
                        }
                        xml.push_str("      </binding>\n");
                    }
                    xml.push_str("    </result>\n");
                }
                xml.push_str("  </results>\n</sparql>");
                Ok(xml)
            }
            CoreQueryResult::Ask(result) => {
                let xml = format!(
                    "<?xml version=\"1.0\"?>\n<sparql xmlns=\"http://www.w3.org/2005/sparql-results#\">\n  <head/>\n  <boolean>{result}</boolean>\n</sparql>"
                );
                Ok(xml)
            }
            _ => Err(FusekiError::unsupported_media_type(
                "XML format only supported for SELECT and ASK queries",
            )),
        }
    }
    /// Convert to CSV string
    pub fn to_csv(&self) -> FusekiResult<String> {
        match &self.inner {
            CoreQueryResult::Select {
                variables,
                bindings,
            } => {
                let mut csv = variables.join(",");
                csv.push('\n');
                for binding in bindings {
                    let values: Vec<String> = variables
                        .iter()
                        .map(|var| {
                            binding
                                .get(var)
                                .map(|term| term.to_string())
                                .unwrap_or_default()
                        })
                        .collect();
                    csv.push_str(&values.join(","));
                    csv.push('\n');
                }
                Ok(csv)
            }
            _ => Err(FusekiError::unsupported_media_type(
                "CSV format only supported for SELECT queries",
            )),
        }
    }
    /// Convert to TSV string
    pub fn to_tsv(&self) -> FusekiResult<String> {
        match &self.inner {
            CoreQueryResult::Select {
                variables,
                bindings,
            } => {
                let mut tsv = variables.join("\t");
                tsv.push('\n');
                for binding in bindings {
                    let values: Vec<String> = variables
                        .iter()
                        .map(|var| {
                            binding
                                .get(var)
                                .map(|term| term.to_string())
                                .unwrap_or_default()
                        })
                        .collect();
                    tsv.push_str(&values.join("\t"));
                    tsv.push('\n');
                }
                Ok(tsv)
            }
            _ => Err(FusekiError::unsupported_media_type(
                "TSV format only supported for SELECT queries",
            )),
        }
    }
    /// Convert to RDF string (for CONSTRUCT/DESCRIBE)
    pub fn to_rdf(&self, format: RdfSerializationFormat) -> FusekiResult<String> {
        match &self.inner {
            CoreQueryResult::Construct(triples) => {
                let core_format = match format {
                    RdfSerializationFormat::Turtle => CoreRdfFormat::Turtle,
                    RdfSerializationFormat::NTriples => CoreRdfFormat::NTriples,
                    RdfSerializationFormat::RdfXml => CoreRdfFormat::RdfXml,
                    RdfSerializationFormat::JsonLd => {
                        return Err(FusekiError::unsupported_media_type(
                            "JSON-LD not supported yet",
                        ));
                    }
                    RdfSerializationFormat::NQuads => CoreRdfFormat::NQuads,
                };
                let serializer = Serializer::new(core_format);
                let graph = oxirs_core::model::graph::Graph::from_iter(triples.clone());
                serializer.serialize_graph(&graph).map_err(|e| {
                    FusekiError::parse(format!("Failed to serialize CONSTRUCT result: {e}"))
                })
            }
            _ => Err(FusekiError::unsupported_media_type(
                "RDF format only supported for CONSTRUCT and DESCRIBE queries",
            )),
        }
    }
    /// Get result in the specified format
    pub fn format_as(&self, format: ResultFormat) -> FusekiResult<String> {
        match format {
            ResultFormat::Json => self.to_json(),
            ResultFormat::Xml => self.to_xml(),
            ResultFormat::Csv => self.to_csv(),
            ResultFormat::Tsv => self.to_tsv(),
        }
    }
}
/// Update execution statistics
#[derive(Debug, Clone, Serialize)]
pub struct UpdateStats {
    pub execution_time: Duration,
    pub quads_inserted: usize,
    pub quads_deleted: usize,
    pub operation_type: String,
    pub success: bool,
    pub error_message: Option<String>,
}
/// Update result wrapper with statistics
#[derive(Debug)]
pub struct UpdateResult {
    pub stats: UpdateStats,
}
/// Store statistics
#[derive(Debug, Serialize)]
pub struct StoreStats {
    pub triple_count: usize,
    pub dataset_count: usize,
    pub total_queries: u64,
    pub total_updates: u64,
    pub cache_hit_ratio: f64,
    pub uptime_seconds: u64,
    pub change_log_size: usize,
    pub latest_change_id: u64,
}
/// SPARQL query result formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResultFormat {
    Json,
    Xml,
    Csv,
    Tsv,
}
