//! Patch compression and optimization

use super::{PatchParser, PatchSerializer};
use crate::{PatchOperation, RdfPatch};
use anyhow::Result;
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use std::collections::HashMap;
use std::io::{Read, Write};
use tracing::info;

pub struct PatchCompressor {
    compression_level: u32,
    enable_dictionary: bool,
    prefix_compression: bool,
}

impl PatchCompressor {
    pub fn new() -> Self {
        Self {
            compression_level: 6,
            enable_dictionary: true,
            prefix_compression: true,
        }
    }

    pub fn with_compression_level(mut self, level: u32) -> Self {
        self.compression_level = level.min(9);
        self
    }

    pub fn with_dictionary_compression(mut self, enabled: bool) -> Self {
        self.enable_dictionary = enabled;
        self
    }

    pub fn with_prefix_compression(mut self, enabled: bool) -> Self {
        self.prefix_compression = enabled;
        self
    }

    /// Compress patch using gzip compression
    pub fn compress_patch(&self, patch: &RdfPatch) -> Result<Vec<u8>> {
        // Serialize patch to string
        let serializer = PatchSerializer::new().with_pretty_print(false);
        let patch_str = serializer.serialize(patch)?;
        let original_len = patch_str.len();

        // Apply dictionary compression if enabled
        let optimized_str = if self.enable_dictionary {
            self.apply_dictionary_compression(&patch_str)?
        } else {
            patch_str
        };

        // Apply gzip compression
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(self.compression_level));
        encoder.write_all(optimized_str.as_bytes())?;
        let compressed = encoder.finish()?;

        info!(
            "Compressed patch: {} -> {} bytes ({:.1}% reduction)",
            original_len,
            compressed.len(),
            (1.0 - compressed.len() as f64 / original_len as f64) * 100.0
        );

        Ok(compressed)
    }

    /// Decompress patch from compressed bytes
    pub fn decompress_patch(&self, compressed_data: &[u8]) -> Result<RdfPatch> {
        // Decompress gzip
        let mut decoder = GzDecoder::new(compressed_data);
        let mut decompressed = String::new();
        decoder.read_to_string(&mut decompressed)?;

        // Apply dictionary decompression if needed
        let patch_str = if self.enable_dictionary {
            self.apply_dictionary_decompression(&decompressed)?
        } else {
            decompressed
        };

        // Parse patch
        let mut parser = PatchParser::new();
        parser.parse(&patch_str)
    }

    fn apply_dictionary_compression(&self, patch_str: &str) -> Result<String> {
        // Build frequency dictionary of common terms
        let mut word_freq = HashMap::new();
        for word in patch_str.split_whitespace() {
            *word_freq.entry(word.to_string()).or_insert(0) += 1;
        }

        // Create dictionary of most frequent terms
        let mut freq_words: Vec<_> = word_freq.into_iter().collect();
        freq_words.sort_by(|a, b| b.1.cmp(&a.1));

        let mut dictionary = HashMap::new();
        let mut compressed = patch_str.to_string();

        // Replace most frequent words with short codes
        for (i, (word, freq)) in freq_words.iter().take(256).enumerate() {
            if word.len() > 3 && *freq > 2 {
                let code = format!("#{i:02x}");
                dictionary.insert(code.clone(), word.clone());
                compressed = compressed.replace(word, &code);
            }
        }

        // Prepend dictionary to compressed string
        let mut dict_header = String::new();
        for (code, word) in dictionary {
            dict_header.push_str(&format!("{code}={word}\n"));
        }
        dict_header.push_str("---\n");
        dict_header.push_str(&compressed);

        Ok(dict_header)
    }

    fn apply_dictionary_decompression(&self, compressed_str: &str) -> Result<String> {
        if let Some(separator_pos) = compressed_str.find("---\n") {
            let (dict_part, content_part) = compressed_str.split_at(separator_pos);
            let content = &content_part[4..]; // Skip "---\n"

            let mut dictionary = HashMap::new();
            for line in dict_part.lines() {
                if let Some(eq_pos) = line.find('=') {
                    let code = &line[..eq_pos];
                    let word = &line[eq_pos + 1..];
                    dictionary.insert(code, word);
                }
            }

            let mut decompressed = content.to_string();
            for (code, word) in dictionary {
                decompressed = decompressed.replace(code, word);
            }

            Ok(decompressed)
        } else {
            Ok(compressed_str.to_string())
        }
    }

    /// Compress using prefix compression for common namespaces
    pub fn compress_with_prefixes(&self, patch: &RdfPatch) -> Result<RdfPatch> {
        let mut compressed = patch.clone();
        compressed.id = format!("{}-prefix-compressed", patch.id);

        if !self.prefix_compression {
            return Ok(compressed);
        }

        // Build frequency map of URI prefixes
        let mut prefix_freq = HashMap::new();
        for operation in &patch.operations {
            self.collect_uris_from_operation(operation, &mut prefix_freq);
        }

        // Find common prefixes
        let mut common_prefixes = HashMap::new();
        for (uri, freq) in prefix_freq {
            if freq > 2 {
                if let Some(prefix) = self.extract_namespace_prefix(&uri) {
                    if prefix.len() > 10 {
                        let short_prefix = format!("ns{}", common_prefixes.len());
                        common_prefixes.insert(prefix, short_prefix);
                    }
                }
            }
        }

        // Add prefix declarations to patch
        for (namespace, prefix) in &common_prefixes {
            compressed.add_operation(PatchOperation::AddPrefix {
                prefix: prefix.clone(),
                namespace: namespace.clone(),
            });
        }

        // Replace URIs with prefixed forms
        for operation in &mut compressed.operations {
            self.apply_prefix_compression_to_operation(operation, &common_prefixes);
        }

        info!(
            "Applied prefix compression: {} prefixes defined",
            common_prefixes.len()
        );
        Ok(compressed)
    }

    fn collect_uris_from_operation(
        &self,
        operation: &PatchOperation,
        prefix_freq: &mut HashMap<String, usize>,
    ) {
        match operation {
            PatchOperation::Add {
                subject,
                predicate,
                object,
            } => {
                *prefix_freq.entry(subject.clone()).or_insert(0) += 1;
                *prefix_freq.entry(predicate.clone()).or_insert(0) += 1;
                *prefix_freq.entry(object.clone()).or_insert(0) += 1;
            }
            PatchOperation::Delete {
                subject,
                predicate,
                object,
            } => {
                *prefix_freq.entry(subject.clone()).or_insert(0) += 1;
                *prefix_freq.entry(predicate.clone()).or_insert(0) += 1;
                *prefix_freq.entry(object.clone()).or_insert(0) += 1;
            }
            PatchOperation::AddGraph { graph } | PatchOperation::DeleteGraph { graph } => {
                *prefix_freq.entry(graph.clone()).or_insert(0) += 1;
            }
            _ => {}
        }
    }

    fn extract_namespace_prefix(&self, uri: &str) -> Option<String> {
        // Extract namespace part of URI (everything up to last # or /)
        if let Some(pos) = uri.rfind('#') {
            Some(uri[..pos + 1].to_string())
        } else {
            uri.rfind('/').map(|pos| uri[..pos + 1].to_string())
        }
    }

    fn apply_prefix_compression_to_operation(
        &self,
        operation: &mut PatchOperation,
        prefixes: &HashMap<String, String>,
    ) {
        match operation {
            PatchOperation::Add {
                subject,
                predicate,
                object,
            } => {
                *subject = self.compress_uri_with_prefixes(subject, prefixes);
                *predicate = self.compress_uri_with_prefixes(predicate, prefixes);
                *object = self.compress_uri_with_prefixes(object, prefixes);
            }
            PatchOperation::Delete {
                subject,
                predicate,
                object,
            } => {
                *subject = self.compress_uri_with_prefixes(subject, prefixes);
                *predicate = self.compress_uri_with_prefixes(predicate, prefixes);
                *object = self.compress_uri_with_prefixes(object, prefixes);
            }
            PatchOperation::AddGraph { graph } | PatchOperation::DeleteGraph { graph } => {
                *graph = self.compress_uri_with_prefixes(graph, prefixes);
            }
            _ => {}
        }
    }

    fn compress_uri_with_prefixes(&self, uri: &str, prefixes: &HashMap<String, String>) -> String {
        for (namespace, prefix) in prefixes {
            if uri.starts_with(namespace) {
                let local_name = &uri[namespace.len()..];
                return format!("{prefix}:{local_name}");
            }
        }
        uri.to_string()
    }
}

impl Default for PatchCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::context::{apply_patch_with_context, PatchContext};
    use crate::patch::result::{
        create_reverse_patch, create_transactional_patch, optimize_patch, validate_patch,
    };

    #[test]
    fn test_patch_serialization() {
        let mut patch = RdfPatch::new();
        patch.add_operation(PatchOperation::Header {
            key: "creator".to_string(),
            value: "test-suite".to_string(),
        });
        patch.add_operation(PatchOperation::TransactionBegin {
            transaction_id: Some("tx-123".to_string()),
        });
        patch.add_operation(PatchOperation::AddPrefix {
            prefix: "ex".to_string(),
            namespace: "http://example.org/".to_string(),
        });
        patch.add_operation(PatchOperation::Add {
            subject: "http://example.org/subject".to_string(),
            predicate: "http://example.org/predicate".to_string(),
            object: "\"Object literal\"".to_string(),
        });
        patch.add_operation(PatchOperation::Delete {
            subject: "http://example.org/subject2".to_string(),
            predicate: "http://example.org/predicate2".to_string(),
            object: "http://example.org/object2".to_string(),
        });
        patch.add_operation(PatchOperation::TransactionCommit);

        let serializer = PatchSerializer::new();
        let result = serializer.serialize(&patch);

        assert!(result.is_ok());
        let serialized = result.unwrap();
        assert!(serialized.contains("H creator test-suite"));
        assert!(serialized.contains("TX tx-123"));
        assert!(serialized.contains("PA ex:"));
        assert!(serialized.contains("A "));
        assert!(serialized.contains("D "));
        assert!(serialized.contains("TC"));
        assert!(serialized.contains("@prefix"));
    }

    #[test]
    fn test_patch_parsing() {
        let patch_content = r#"
@prefix ex: <http://example.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

H creator test-parser .
TX tx-456 .
PA ex2: <http://example2.org/> .
A ex:subject ex:predicate "Object literal" .
D ex:subject2 ex:predicate2 ex:object2 .
GA ex:graph1 .
GD ex:graph2 .
PD old: .
TC .
"#;

        let mut parser = PatchParser::new();
        let result = parser.parse(patch_content);

        assert!(result.is_ok());
        let patch = result.unwrap();
        assert_eq!(patch.operations.len(), 9);

        // Check header was captured
        assert_eq!(
            patch.headers.get("creator"),
            Some(&"test-parser".to_string())
        );

        // Check transaction ID was captured
        assert_eq!(patch.transaction_id, Some("tx-456".to_string()));

        // Check prefix was captured
        assert_eq!(
            patch.prefixes.get("ex2"),
            Some(&"http://example2.org/".to_string())
        );

        match &patch.operations[0] {
            PatchOperation::Header { key, value } => {
                assert_eq!(key, "creator");
                assert_eq!(value, "test-parser");
            }
            _ => panic!("Expected Header operation"),
        }

        match &patch.operations[1] {
            PatchOperation::TransactionBegin { transaction_id } => {
                assert_eq!(transaction_id, &Some("tx-456".to_string()));
            }
            _ => panic!("Expected TransactionBegin operation"),
        }
    }

    #[test]
    fn test_patch_round_trip() {
        let mut original_patch = RdfPatch::new();
        original_patch.add_operation(PatchOperation::Add {
            subject: "http://example.org/subject".to_string(),
            predicate: "http://example.org/predicate".to_string(),
            object: "\"Test object\"".to_string(),
        });

        // Serialize to string
        let serialized = original_patch.to_rdf_patch_format().unwrap();

        // Parse back from string
        let parsed_patch = RdfPatch::from_rdf_patch_format(&serialized).unwrap();

        // Check that we get the same operations
        assert_eq!(
            original_patch.operations.len(),
            parsed_patch.operations.len()
        );

        match (&original_patch.operations[0], &parsed_patch.operations[0]) {
            (
                PatchOperation::Add {
                    subject: s1,
                    predicate: p1,
                    object: o1,
                },
                PatchOperation::Add {
                    subject: s2,
                    predicate: p2,
                    object: o2,
                },
            ) => {
                assert_eq!(s1, s2);
                assert_eq!(p1, p2);
                assert_eq!(o1, o2);
            }
            _ => panic!("Operations don't match"),
        }
    }

    #[test]
    fn test_reverse_patch() {
        let mut patch = RdfPatch::new();
        patch.add_operation(PatchOperation::TransactionBegin {
            transaction_id: Some("tx-789".to_string()),
        });
        patch.add_operation(PatchOperation::Add {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "http://example.org/o".to_string(),
        });
        patch.add_operation(PatchOperation::AddGraph {
            graph: "http://example.org/graph".to_string(),
        });
        patch.add_operation(PatchOperation::TransactionCommit);
        patch.transaction_id = Some("tx-789".to_string());

        let reverse = create_reverse_patch(&patch).unwrap();

        // Should have TX, two reversed operations, and TC
        assert!(reverse.operations.len() >= 4);

        // First should be transaction begin (reversing the commit)
        match &reverse.operations[0] {
            PatchOperation::TransactionBegin { .. } => {}
            _ => panic!("Expected TransactionBegin operation"),
        }

        // Find the reversed operations
        let has_delete_graph = reverse.operations.iter().any(|op| {
            matches!(op, PatchOperation::DeleteGraph { graph } if graph == "http://example.org/graph")
        });
        let has_delete_triple = reverse.operations.iter().any(|op| {
            matches!(op, PatchOperation::Delete { subject, .. } if subject == "http://example.org/s")
        });

        assert!(has_delete_graph);
        assert!(has_delete_triple);

        // Last should be transaction commit
        match reverse.operations.last() {
            Some(PatchOperation::TransactionCommit) => {}
            _ => panic!("Expected TransactionCommit as last operation"),
        }
    }

    #[test]
    fn test_patch_optimization() {
        let mut patch = RdfPatch::new();
        let operation = PatchOperation::Add {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "http://example.org/o".to_string(),
        };

        // Add the same operation twice
        patch.add_operation(operation.clone());
        patch.add_operation(operation);

        let optimized = optimize_patch(&patch).unwrap();

        // Should remove duplicate
        assert_eq!(optimized.operations.len(), 1);
    }

    #[test]
    fn test_transactional_patch() {
        let operations = vec![
            PatchOperation::Add {
                subject: "s1".to_string(),
                predicate: "p1".to_string(),
                object: "o1".to_string(),
            },
            PatchOperation::Delete {
                subject: "s2".to_string(),
                predicate: "p2".to_string(),
                object: "o2".to_string(),
            },
        ];

        let patch = create_transactional_patch(operations);

        // Should have TX + 2 operations + TC = 4 total
        assert_eq!(patch.operations.len(), 4);

        // First should be transaction begin
        assert!(matches!(
            &patch.operations[0],
            PatchOperation::TransactionBegin { .. }
        ));

        // Last should be transaction commit
        assert!(matches!(
            &patch.operations[3],
            PatchOperation::TransactionCommit
        ));

        // Should have transaction ID set
        assert!(patch.transaction_id.is_some());
    }

    #[test]
    fn test_patch_validation() {
        let mut patch = RdfPatch::new();
        patch.add_operation(PatchOperation::Delete {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "http://example.org/o".to_string(),
        });

        let warnings = validate_patch(&patch).unwrap();

        // Should warn about deleting without prior addition
        assert!(!warnings.is_empty());
        assert!(warnings[0].contains("deleted without prior addition"));
    }

    #[test]
    fn test_patch_application() {
        let mut patch = RdfPatch::new();
        patch.add_operation(PatchOperation::Add {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "http://example.org/o".to_string(),
        });

        let context = PatchContext {
            strict_mode: false,
            validate_operations: true,
            dry_run: true,
        };

        let result = apply_patch_with_context(&patch, &context).unwrap();

        assert_eq!(result.total_operations, 1);
        assert_eq!(result.operations_applied, 1);
        assert!(result.is_success());
        assert_eq!(result.success_rate(), 1.0);
    }
}
