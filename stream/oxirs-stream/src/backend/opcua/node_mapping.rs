//! OPC UA Node to RDF Mapping

use super::types::{NodeSubscription, OpcUaDataChange};
use crate::error::StreamResult;
use crate::event::{EventMetadata, StreamEvent};
use std::collections::HashMap;

/// Maps OPC UA nodes to RDF triples
pub struct NodeMapper;

impl NodeMapper {
    /// Create a new node mapper
    pub fn new() -> Self {
        Self
    }

    /// Convert OPC UA data change to RDF triple event
    pub fn to_stream_event(
        &self,
        change: &OpcUaDataChange,
        mapping: &NodeSubscription,
        source_endpoint: &str,
    ) -> StreamResult<StreamEvent> {
        let metadata = EventMetadata {
            event_id: uuid::Uuid::new_v4().to_string(),
            timestamp: change.source_timestamp.unwrap_or(change.server_timestamp),
            source: format!("opcua:{}", source_endpoint),
            user: None,
            context: mapping.rdf_graph.clone(),
            caused_by: None,
            version: "1.0".to_string(),
            properties: self.build_properties(change, mapping),
            checksum: None,
        };

        let object = change.value.to_rdf_literal();

        Ok(StreamEvent::TripleAdded {
            subject: mapping.rdf_subject.clone(),
            predicate: mapping.rdf_predicate.clone(),
            object,
            graph: mapping.rdf_graph.clone(),
            metadata,
        })
    }

    /// Build event properties from OPC UA data
    fn build_properties(
        &self,
        change: &OpcUaDataChange,
        mapping: &NodeSubscription,
    ) -> HashMap<String, String> {
        let mut props = HashMap::new();

        props.insert("nodeId".to_string(), change.node_id.clone());
        props.insert("statusCode".to_string(), change.status_code.to_string());
        props.insert(
            "datatype".to_string(),
            change.value.xsd_datatype().to_string(),
        );

        if let Some(ref unit) = mapping.unit_uri {
            props.insert("unit".to_string(), unit.clone());
        }

        if let Some(ref samm_prop) = mapping.samm_property {
            props.insert("sammProperty".to_string(), samm_prop.clone());
        }

        if let Some(ts) = change.source_timestamp {
            props.insert("sourceTimestamp".to_string(), ts.to_rfc3339());
        }

        props.insert(
            "serverTimestamp".to_string(),
            change.server_timestamp.to_rfc3339(),
        );

        props
    }
}

impl Default for NodeMapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_mapper_creation() {
        let _mapper = NodeMapper::new();
        // Just verify it constructs
    }
}
