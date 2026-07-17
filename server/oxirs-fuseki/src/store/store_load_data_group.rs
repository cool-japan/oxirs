//! # Store - load_data_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

/// Insert a batch of quads through a single store handle.
///
/// This is the shared batch-ingest seam for every Fuseki write path (GSP
/// PUT/POST add, `/upload`, `load_data`/`import_data`, SPARQL `LOAD`). Callers
/// accumulate parsed data into a `Vec<Quad>` and hand it here in one call
/// instead of scattering per-quad `insert_quad` loops across the handlers.
///
/// It returns the number of quads that were newly inserted (duplicates already
/// present are not counted, matching the `bool` contract of
/// [`oxirs_core::Store::insert_quad`]). This now delegates to the native
/// [`oxirs_core::Store::bulk_insert_quads`] trait method, so durable backends
/// take a single write lock and perform a single append + one `fsync` for the
/// whole batch instead of a per-quad fsync loop. Keeping every Fuseki write
/// path routed through this one seam means the single-fsync contract lives in
/// one place.
pub fn bulk_insert_quads(
    store: &dyn CoreStore,
    quads: Vec<Quad>,
) -> Result<usize, oxirs_core::OxirsError> {
    store.bulk_insert_quads(quads)
}

impl Store {
    /// Load RDF data from a string into a dataset
    pub fn load_data(
        &self,
        data: &str,
        format: RdfSerializationFormat,
        dataset_name: Option<&str>,
    ) -> FusekiResult<usize> {
        let store = self.get_dataset(dataset_name)?;
        let store_guard = store
            .write()
            .map_err(|e| FusekiError::store(format!("Failed to acquire store write lock: {e}")))?;
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
        let parser = Parser::new(core_format);
        let quads = parser
            .parse_str_to_quads(data)
            .map_err(|e| FusekiError::parse(format!("Failed to parse RDF data: {e}")))?;
        // Accumulate the default-graph quads and insert them through the single
        // batched ingest path rather than a per-triple loop.
        let default_quads: Vec<Quad> = quads.into_iter().filter(|q| q.is_default_graph()).collect();
        let inserted_count = bulk_insert_quads(&*store_guard, default_quads)
            .map_err(|e| FusekiError::store(format!("Failed to insert triple: {e}")))?;
        info!("Loaded {} triples into dataset", inserted_count);
        Ok(inserted_count)
    }
    /// Export RDF data from a dataset as a string
    pub fn export_data(
        &self,
        format: RdfSerializationFormat,
        dataset_name: Option<&str>,
    ) -> FusekiResult<String> {
        let store = self.get_dataset(dataset_name)?;
        let store_guard = store
            .read()
            .map_err(|e| FusekiError::store(format!("Failed to acquire store read lock: {e}")))?;
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
        let triples = store_guard
            .triples()
            .map_err(|e| FusekiError::store(format!("Failed to query triples: {e}")))?;
        let graph = Graph::from_triples(triples);
        let serialized = serializer
            .serialize_graph(&graph)
            .map_err(|e| FusekiError::parse(format!("Failed to serialize data: {e}")))?;
        Ok(serialized)
    }
    /// Import RDF data into a dataset from a string
    pub async fn import_data(
        &self,
        data: &str,
        format: RdfSerializationFormat,
        dataset_name: Option<&str>,
    ) -> FusekiResult<usize> {
        let store = self.get_dataset(dataset_name)?;
        let store_guard = store
            .write()
            .map_err(|e| FusekiError::store(format!("Failed to acquire store write lock: {e}")))?;
        let core_format = match format {
            RdfSerializationFormat::Turtle => CoreRdfFormat::Turtle,
            RdfSerializationFormat::NTriples => CoreRdfFormat::NTriples,
            RdfSerializationFormat::RdfXml => CoreRdfFormat::RdfXml,
            RdfSerializationFormat::JsonLd => {
                return Err(FusekiError::unsupported_media_type(
                    "JSON-LD import not supported yet",
                ));
            }
            RdfSerializationFormat::NQuads => CoreRdfFormat::NQuads,
        };
        let parser = Parser::new(core_format);
        let quads = parser
            .parse_str_to_quads(data)
            .map_err(|e| FusekiError::parse(format!("Failed to parse RDF data: {e}")))?;
        // Route the whole payload through the single batched ingest path
        // (preserving each quad's original graph) instead of a per-quad loop.
        let inserted_count = bulk_insert_quads(&*store_guard, quads)
            .map_err(|e| FusekiError::store(format!("Failed to insert quad: {e}")))?;
        let mut metadata = self.metadata.write().map_err(|e| {
            FusekiError::store(format!("Failed to acquire metadata write lock: {e}"))
        })?;
        metadata.last_modified = Some(Instant::now());
        metadata.total_updates += 1;
        let change_id = metadata.last_change_id + 1;
        metadata.last_change_id = change_id;
        metadata.change_log.push(StoreChange {
            id: change_id,
            timestamp: chrono::Utc::now(),
            operation_type: "import".to_string(),
            affected_graphs: vec![dataset_name.unwrap_or("default").to_string()],
            triple_count: inserted_count,
            dataset_name: dataset_name.map(|s| s.to_string()),
        });
        let change_log_len = metadata.change_log.len();
        if change_log_len > 1000 {
            metadata.change_log.drain(0..change_log_len - 1000);
        }
        info!(
            "Imported {} triples into dataset '{}'",
            inserted_count,
            dataset_name.unwrap_or("default")
        );
        Ok(inserted_count)
    }
}

#[cfg(test)]
mod bulk_insert_tests {
    use super::bulk_insert_quads;
    use oxirs_core::model::{GraphName, Literal, NamedNode, Quad};
    use oxirs_core::rdf_store::ConcreteStore;

    fn quad(s: &str, o: &str, g: GraphName) -> Quad {
        Quad::new(
            NamedNode::new(s).expect("subject IRI"),
            NamedNode::new("http://example.org/p").expect("predicate IRI"),
            Literal::new_simple_literal(o),
            g,
        )
    }

    /// The batched ingest path inserts every quad and reports the number of
    /// newly-inserted quads (duplicates are not double-counted).
    #[test]
    fn bulk_insert_counts_new_quads_only() {
        let store = ConcreteStore::new().expect("create store");
        let quads = vec![
            quad("http://example.org/a", "1", GraphName::DefaultGraph),
            quad("http://example.org/b", "2", GraphName::DefaultGraph),
            // duplicate of the first quad — must not be counted again
            quad("http://example.org/a", "1", GraphName::DefaultGraph),
        ];
        let inserted = bulk_insert_quads(&store, quads).expect("bulk insert");
        assert_eq!(inserted, 2, "only two distinct quads should be inserted");
        let all = oxirs_core::Store::find_quads(&store, None, None, None, None).expect("find all");
        assert_eq!(all.len(), 2);
    }

    /// Named-graph membership is preserved through the batch helper.
    #[test]
    fn bulk_insert_preserves_named_graphs() {
        let store = ConcreteStore::new().expect("create store");
        let g = GraphName::NamedNode(NamedNode::new("http://example.org/g").expect("graph IRI"));
        let quads = vec![
            quad("http://example.org/a", "1", GraphName::DefaultGraph),
            quad("http://example.org/b", "2", g.clone()),
        ];
        let inserted = bulk_insert_quads(&store, quads).expect("bulk insert");
        assert_eq!(inserted, 2);
        let in_named =
            oxirs_core::Store::find_quads(&store, None, None, None, Some(&g)).expect("find named");
        assert_eq!(in_named.len(), 1, "named-graph quad must land in its graph");
    }
}
