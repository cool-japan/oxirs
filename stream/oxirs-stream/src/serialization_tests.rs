//! Unit tests for the serialization subsystem.

#[cfg(test)]
mod tests {
    use crate::serialization_encoder::{EventSerializer, FormatConverter};
    use crate::serialization_types::SerializationFormat;
    use crate::{CompressionType, EventMetadata, StreamEvent};

    #[tokio::test]
    async fn test_json_serialization() {
        let event = StreamEvent::Heartbeat {
            timestamp: chrono::Utc::now(),
            source: "test".to_string(),
            metadata: crate::event::EventMetadata::default(),
        };

        let serializer = EventSerializer::new(SerializationFormat::Json);
        let serialized = serializer.serialize(&event).await.unwrap();
        let deserialized = serializer.deserialize(&serialized).await.unwrap();

        match deserialized {
            StreamEvent::Heartbeat { source, .. } => {
                assert_eq!(source, "test");
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[tokio::test]
    async fn test_format_detection() {
        let json_data = b"{\"test\": \"data\"}";
        assert_eq!(
            SerializationFormat::detect(json_data),
            Some(SerializationFormat::Json)
        );

        let magic_data = b"PB03some_data";
        assert_eq!(
            SerializationFormat::detect(magic_data),
            Some(SerializationFormat::Protobuf)
        );
    }

    #[tokio::test]
    async fn test_compression() {
        let event = StreamEvent::Heartbeat {
            timestamp: chrono::Utc::now(),
            source: "test".to_string(),
            metadata: crate::event::EventMetadata::default(),
        };

        let serializer =
            EventSerializer::new(SerializationFormat::Json).with_compression(CompressionType::Gzip);

        let serialized = serializer.serialize(&event).await.unwrap();
        let deserialized = serializer.deserialize(&serialized).await.unwrap();

        match deserialized {
            StreamEvent::Heartbeat { source, .. } => {
                assert_eq!(source, "test");
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[tokio::test]
    async fn test_messagepack_serialization() {
        let metadata = EventMetadata::default();
        let event = StreamEvent::TripleAdded {
            subject: "http://example.org/subject".to_string(),
            predicate: "http://example.org/predicate".to_string(),
            object: "http://example.org/object".to_string(),
            graph: None,
            metadata,
        };

        let serializer = EventSerializer::new(SerializationFormat::MessagePack);
        let serialized = serializer.serialize(&event).await.unwrap();
        let deserialized = serializer.deserialize(&serialized).await.unwrap();

        match deserialized {
            StreamEvent::TripleAdded {
                subject,
                predicate,
                object,
                ..
            } => {
                assert_eq!(subject, "http://example.org/subject");
                assert_eq!(predicate, "http://example.org/predicate");
                assert_eq!(object, "http://example.org/object");
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[tokio::test]
    async fn test_format_conversion() {
        let event = StreamEvent::Heartbeat {
            timestamp: chrono::Utc::now(),
            source: "test".to_string(),
            metadata: crate::event::EventMetadata::default(),
        };

        // Serialize to JSON
        let json_serializer = EventSerializer::new(SerializationFormat::Json);
        let json_data = json_serializer.serialize(&event).await.unwrap();

        // Convert to MessagePack
        let converter =
            FormatConverter::new(SerializationFormat::Json, SerializationFormat::MessagePack);
        let msgpack_data = converter.convert(&json_data).await.unwrap();

        // Verify by deserializing
        let msgpack_serializer = EventSerializer::new(SerializationFormat::MessagePack);
        let deserialized = msgpack_serializer.deserialize(&msgpack_data).await.unwrap();

        match deserialized {
            StreamEvent::Heartbeat { source, .. } => {
                assert_eq!(source, "test");
            }
            _ => panic!("Wrong event type"),
        }
    }
}
