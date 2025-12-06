//! # Store - load_data_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

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
        let graph = Graph::from_iter(
            quads
                .into_iter()
                .filter(|q| q.is_default_graph())
                .map(|q| q.to_triple()),
        );
        let mut inserted_count = 0;
        for triple in graph.iter() {
            if store_guard
                .insert_triple(triple.clone())
                .map_err(|e| FusekiError::store(format!("Failed to insert triple: {e}")))?
            {
                inserted_count += 1;
            }
        }
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
        let mut inserted_count = 0;
        for quad in quads {
            store_guard
                .insert_quad(quad)
                .map_err(|e| FusekiError::store(format!("Failed to insert quad: {e}")))?;
            inserted_count += 1;
        }
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
