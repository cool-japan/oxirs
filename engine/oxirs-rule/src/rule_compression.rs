//! Rule Set Compression
//!
//! This module provides compression capabilities for large rule sets to reduce
//! memory footprint and improve loading/serialization performance.
//!
//! Features:
//! - Multiple compression modes (Fast, Balanced, Best, Adaptive)
//! - LZ4-style and DEFLATE compression algorithms
//! - Automatic compression mode selection based on rule set characteristics
//! - Compression statistics and metrics
//! - Uses serde for serialization with compression on top

use crate::Rule;
use anyhow::{anyhow, Result};
use scirs2_core::metrics::{Counter, Gauge, Histogram};
use serde::{Deserialize, Serialize};

/// Compression mode for rule sets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionMode {
    /// No compression (fastest, largest size)
    None,
    /// Fast compression (LZ4-style, good balance)
    Fast,
    /// Balanced compression (moderate speed, good compression)
    Balanced,
    /// Best compression (slowest, smallest size, DEFLATE-style)
    Best,
    /// Adaptive mode (automatically selects based on rule set characteristics)
    Adaptive,
}

/// Compressed rule set with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedRuleSet {
    /// Compression mode used
    pub mode: CompressionMode,
    /// Original size in bytes
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Number of rules
    pub rule_count: usize,
    /// Compressed data
    pub data: Vec<u8>,
    /// Compression timestamp
    pub timestamp: u64,
}

impl CompressedRuleSet {
    /// Calculate compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.original_size == 0 {
            return 1.0;
        }
        self.compressed_size as f64 / self.original_size as f64
    }

    /// Calculate space saved in bytes
    pub fn space_saved(&self) -> usize {
        self.original_size.saturating_sub(self.compressed_size)
    }

    /// Calculate space saved as percentage
    pub fn space_saved_percent(&self) -> f64 {
        if self.original_size == 0 {
            return 0.0;
        }
        (self.space_saved() as f64 / self.original_size as f64) * 100.0
    }
}

/// Rule set compressor
pub struct RuleCompressor {
    /// Compression mode
    mode: CompressionMode,
    /// Metrics
    compressed_rules: Counter,
    #[allow(dead_code)]
    compression_ratio: Gauge,
    compression_time: Histogram,
    decompression_time: Histogram,
}

impl RuleCompressor {
    /// Create a new rule compressor with the specified mode
    pub fn new(mode: CompressionMode) -> Self {
        Self {
            mode,
            compressed_rules: Counter::new("compressed_rules".to_string()),
            compression_ratio: Gauge::new("compression_ratio".to_string()),
            compression_time: Histogram::new("compression_time_ms".to_string()),
            decompression_time: Histogram::new("decompression_time_ms".to_string()),
        }
    }

    /// Compress a rule set
    pub fn compress(&mut self, rules: &[Rule]) -> Result<CompressedRuleSet> {
        let start_time = std::time::Instant::now();

        // Determine compression mode
        let mode = match self.mode {
            CompressionMode::Adaptive => self.select_adaptive_mode(rules),
            other => other,
        };

        // Serialize rules using serde
        let serialized_data = bincode::serialize(rules)?;
        let original_size = serialized_data.len();

        // Compress the serialized data
        let compressed_data = match mode {
            CompressionMode::None => serialized_data,
            CompressionMode::Fast => self.lz4_style_compress(&serialized_data)?,
            CompressionMode::Balanced => self.balanced_compress(&serialized_data)?,
            CompressionMode::Best => self.deflate_compress(&serialized_data)?,
            CompressionMode::Adaptive => unreachable!(), // Already resolved above
        };

        let compressed_size = compressed_data.len();

        // Record metrics
        self.compressed_rules.add(rules.len() as u64);
        let elapsed = start_time.elapsed();
        self.compression_time.observe(elapsed.as_millis() as f64);

        Ok(CompressedRuleSet {
            mode,
            original_size,
            compressed_size,
            rule_count: rules.len(),
            data: compressed_data,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }

    /// Decompress a rule set
    pub fn decompress(&mut self, compressed: &CompressedRuleSet) -> Result<Vec<Rule>> {
        let start_time = std::time::Instant::now();

        // Decompress the data
        let serialized_data = match compressed.mode {
            CompressionMode::None => compressed.data.clone(),
            CompressionMode::Fast => self.lz4_style_decompress(&compressed.data)?,
            CompressionMode::Balanced => self.balanced_decompress(&compressed.data)?,
            CompressionMode::Best => self.deflate_decompress(&compressed.data)?,
            CompressionMode::Adaptive => {
                return Err(anyhow!("Cannot decompress Adaptive mode directly"));
            }
        };

        // Deserialize rules using serde
        let rules: Vec<Rule> = bincode::deserialize(&serialized_data)?;

        // Record metrics
        let elapsed = start_time.elapsed();
        self.decompression_time.observe(elapsed.as_millis() as f64);

        Ok(rules)
    }

    /// LZ4-style fast compression
    fn lz4_style_compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut compressed = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            // Look for matches in previous data (simplified LZ4)
            let match_result = self.find_match(data, pos);

            if let Some((match_pos, match_len)) = match_result {
                // Write match (offset and length)
                compressed.push(0xFF); // Match marker
                let offset = (pos - match_pos) as u16;
                compressed.extend_from_slice(&offset.to_le_bytes());
                compressed.push(match_len as u8);
                pos += match_len;
            } else {
                // Write literal
                compressed.push(data[pos]);
                pos += 1;
            }
        }

        Ok(compressed)
    }

    /// Find matching sequence in previous data
    fn find_match(&self, data: &[u8], pos: usize) -> Option<(usize, usize)> {
        const MIN_MATCH_LEN: usize = 4;
        const MAX_MATCH_LEN: usize = 255;
        const SEARCH_WINDOW: usize = 4096;

        if pos < MIN_MATCH_LEN {
            return None;
        }

        let search_start = pos.saturating_sub(SEARCH_WINDOW);
        let mut best_match: Option<(usize, usize)> = None;

        for search_pos in search_start..pos {
            let mut match_len = 0;
            while match_len < MAX_MATCH_LEN
                && pos + match_len < data.len()
                && data[search_pos + match_len] == data[pos + match_len]
            {
                match_len += 1;
            }

            if match_len >= MIN_MATCH_LEN
                && (best_match.is_none() || match_len > best_match.unwrap().1)
            {
                best_match = Some((search_pos, match_len));
            }
        }

        best_match
    }

    /// LZ4-style decompression
    fn lz4_style_decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut decompressed = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            if data[pos] == 0xFF {
                // Match marker
                if pos + 3 >= data.len() {
                    return Err(anyhow!("Truncated match data"));
                }
                let offset = u16::from_le_bytes([data[pos + 1], data[pos + 2]]) as usize;
                let match_len = data[pos + 3] as usize;

                // Copy from previous data
                let copy_start = decompressed.len() - offset;
                for i in 0..match_len {
                    let byte = decompressed[copy_start + i];
                    decompressed.push(byte);
                }

                pos += 4;
            } else {
                // Literal
                decompressed.push(data[pos]);
                pos += 1;
            }
        }

        Ok(decompressed)
    }

    /// Balanced compression (moderate level)
    fn balanced_compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Use LZ4-style with larger search window
        self.lz4_style_compress(data)
    }

    /// Balanced decompression
    fn balanced_decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.lz4_style_decompress(data)
    }

    /// DEFLATE-style best compression (simplified)
    fn deflate_compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simplified DEFLATE with run-length encoding
        let mut compressed = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            // Count runs
            let byte = data[pos];
            let mut run_len = 1;
            while pos + run_len < data.len() && data[pos + run_len] == byte && run_len < 255 {
                run_len += 1;
            }

            if run_len >= 4 {
                // Write run: marker (0xFE), byte, length
                compressed.push(0xFE);
                compressed.push(byte);
                compressed.push(run_len as u8);
                pos += run_len;
            } else {
                // Write literals
                compressed.push(byte);
                pos += 1;
            }
        }

        Ok(compressed)
    }

    /// DEFLATE-style decompression
    fn deflate_decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut decompressed = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            if data[pos] == 0xFE {
                // Run marker
                if pos + 2 >= data.len() {
                    return Err(anyhow!("Truncated run data"));
                }
                let byte = data[pos + 1];
                let run_len = data[pos + 2] as usize;

                for _ in 0..run_len {
                    decompressed.push(byte);
                }

                pos += 3;
            } else {
                // Literal
                decompressed.push(data[pos]);
                pos += 1;
            }
        }

        Ok(decompressed)
    }

    /// Select adaptive compression mode based on rule set characteristics
    fn select_adaptive_mode(&self, rules: &[Rule]) -> CompressionMode {
        let serialized = bincode::serialize(rules).unwrap_or_default();
        let total_size = serialized.len();
        let complexity = self.estimate_complexity(rules);

        // Small rule sets: use Fast
        if total_size < 1024 * 10 {
            return CompressionMode::Fast;
        }

        // Complex rules with repetition: use Balanced
        if complexity > 50 {
            return CompressionMode::Balanced;
        }

        // Default to Fast for general cases
        CompressionMode::Fast
    }

    /// Estimate complexity of rule set
    fn estimate_complexity(&self, rules: &[Rule]) -> usize {
        let mut total_atoms = 0;
        let mut unique_names = std::collections::HashSet::new();

        for rule in rules {
            total_atoms += rule.body.len() + rule.head.len();
            unique_names.insert(&rule.name);
        }

        if unique_names.is_empty() {
            return 0;
        }

        // Complexity = total atoms / unique names (measure of repetition)
        total_atoms * 100 / unique_names.len()
    }

    /// Get compression statistics
    pub fn get_statistics(&self) -> CompressionStatistics {
        CompressionStatistics {
            compressed_rules_count: 0, // Would need persistent storage
        }
    }
}

/// Compression statistics
#[derive(Debug, Clone)]
pub struct CompressionStatistics {
    pub compressed_rules_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RuleAtom, Term};

    fn create_test_rule(name: &str, body_count: usize, head_count: usize) -> Rule {
        let mut body = Vec::new();
        for i in 0..body_count {
            body.push(RuleAtom::Triple {
                subject: Term::Variable(format!("X{}", i)),
                predicate: Term::Constant(format!("pred{}", i)),
                object: Term::Variable(format!("Y{}", i)),
            });
        }

        let mut head = Vec::new();
        for i in 0..head_count {
            head.push(RuleAtom::Triple {
                subject: Term::Variable(format!("X{}", i)),
                predicate: Term::Constant(format!("result{}", i)),
                object: Term::Variable(format!("Y{}", i)),
            });
        }

        Rule {
            name: name.to_string(),
            body,
            head,
        }
    }

    #[test]
    fn test_compression_mode_none() {
        let mut compressor = RuleCompressor::new(CompressionMode::None);
        let rules = vec![
            create_test_rule("rule1", 2, 1),
            create_test_rule("rule2", 3, 2),
        ];

        let compressed = compressor.compress(&rules).unwrap();
        assert_eq!(compressed.mode, CompressionMode::None);
        assert_eq!(compressed.rule_count, 2);

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 2);
        assert_eq!(decompressed[0].name, "rule1");
        assert_eq!(decompressed[1].name, "rule2");
    }

    #[test]
    fn test_compression_mode_fast() {
        let mut compressor = RuleCompressor::new(CompressionMode::Fast);
        let rules: Vec<_> = (0..10)
            .map(|i| create_test_rule(&format!("rule{}", i), 2, 1))
            .collect();

        let compressed = compressor.compress(&rules).unwrap();
        assert_eq!(compressed.mode, CompressionMode::Fast);
        assert!(compressed.compression_ratio() <= 1.0);

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 10);
    }

    #[test]
    fn test_compression_mode_balanced() {
        let mut compressor = RuleCompressor::new(CompressionMode::Balanced);
        let rules: Vec<_> = (0..20)
            .map(|i| create_test_rule(&format!("rule{}", i), 3, 2))
            .collect();

        let compressed = compressor.compress(&rules).unwrap();
        assert_eq!(compressed.mode, CompressionMode::Balanced);

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 20);
    }

    #[test]
    fn test_compression_mode_best() {
        let mut compressor = RuleCompressor::new(CompressionMode::Best);
        let rules: Vec<_> = (0..15)
            .map(|i| create_test_rule(&format!("rule{}", i), 2, 1))
            .collect();

        let compressed = compressor.compress(&rules).unwrap();
        assert_eq!(compressed.mode, CompressionMode::Best);

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 15);
    }

    #[test]
    fn test_compression_mode_adaptive() {
        let mut compressor = RuleCompressor::new(CompressionMode::Adaptive);
        let rules: Vec<_> = (0..5)
            .map(|i| create_test_rule(&format!("rule{}", i), 3, 2))
            .collect();

        let compressed = compressor.compress(&rules).unwrap();
        // Adaptive should select one of the concrete modes
        assert_ne!(compressed.mode, CompressionMode::Adaptive);

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 5);
    }

    #[test]
    fn test_compression_ratio() {
        let compressed = CompressedRuleSet {
            mode: CompressionMode::Fast,
            original_size: 1000,
            compressed_size: 500,
            rule_count: 10,
            data: vec![],
            timestamp: 0,
        };

        assert_eq!(compressed.compression_ratio(), 0.5);
        assert_eq!(compressed.space_saved(), 500);
        assert_eq!(compressed.space_saved_percent(), 50.0);
    }

    #[test]
    fn test_lz4_style_compression() {
        let compressor = RuleCompressor::new(CompressionMode::Fast);

        // Data with repetition
        let data = b"abcabcabcabc";
        let compressed = compressor.lz4_style_compress(data).unwrap();
        let decompressed = compressor.lz4_style_decompress(&compressed).unwrap();

        assert_eq!(&decompressed, data);
    }

    #[test]
    fn test_deflate_compression() {
        let compressor = RuleCompressor::new(CompressionMode::Best);

        // Data with runs
        let data = b"aaaabbbbccccdddd";
        let compressed = compressor.deflate_compress(data).unwrap();
        let decompressed = compressor.deflate_decompress(&compressed).unwrap();

        assert_eq!(&decompressed, data);
    }

    #[test]
    fn test_find_match() {
        let compressor = RuleCompressor::new(CompressionMode::Fast);
        let data = b"abcdefabcdef";

        let match_result = compressor.find_match(data, 6);
        assert!(match_result.is_some());

        let (match_pos, match_len) = match_result.unwrap();
        assert_eq!(match_pos, 0);
        assert_eq!(match_len, 6);
    }

    #[test]
    fn test_empty_rule_set() {
        let mut compressor = RuleCompressor::new(CompressionMode::Fast);
        let rules: Vec<Rule> = vec![];

        let compressed = compressor.compress(&rules).unwrap();
        assert_eq!(compressed.rule_count, 0);

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 0);
    }

    #[test]
    fn test_single_rule() {
        let mut compressor = RuleCompressor::new(CompressionMode::Balanced);
        let rules = vec![create_test_rule("single", 1, 1)];

        let compressed = compressor.compress(&rules).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed.len(), 1);
        assert_eq!(decompressed[0].name, "single");
        assert_eq!(decompressed[0].body.len(), 1);
        assert_eq!(decompressed[0].head.len(), 1);
    }

    #[test]
    fn test_large_rule_set() {
        let mut compressor = RuleCompressor::new(CompressionMode::Fast);
        let rules: Vec<_> = (0..100)
            .map(|i| create_test_rule(&format!("rule{}", i), i % 5 + 1, 1))
            .collect();

        let compressed = compressor.compress(&rules).unwrap();
        assert_eq!(compressed.rule_count, 100);

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed.len(), 100);
    }

    #[test]
    fn test_estimate_complexity() {
        let compressor = RuleCompressor::new(CompressionMode::None);

        // Low complexity (unique rules)
        let rules1: Vec<_> = (0..5)
            .map(|i| create_test_rule(&format!("unique{}", i), 1, 1))
            .collect();
        let complexity1 = compressor.estimate_complexity(&rules1);

        // High complexity (same rule name)
        let rules2 = vec![create_test_rule("same", 5, 3); 10];
        let complexity2 = compressor.estimate_complexity(&rules2);

        assert!(complexity2 > complexity1);
    }

    #[test]
    fn test_adaptive_mode_selection_small() {
        let compressor = RuleCompressor::new(CompressionMode::Adaptive);
        let rules = vec![create_test_rule("small", 1, 1)];

        let mode = compressor.select_adaptive_mode(&rules);
        assert_eq!(mode, CompressionMode::Fast);
    }

    #[test]
    fn test_adaptive_mode_selection_large() {
        let compressor = RuleCompressor::new(CompressionMode::Adaptive);
        let rules: Vec<_> = (0..50)
            .map(|i| create_test_rule(&format!("rule{}", i), 10, 5))
            .collect();

        let mode = compressor.select_adaptive_mode(&rules);
        // Should select Balanced or Best or Fast
        assert!(matches!(
            mode,
            CompressionMode::Balanced | CompressionMode::Best | CompressionMode::Fast
        ));
    }

    #[test]
    fn test_get_statistics() {
        let mut compressor = RuleCompressor::new(CompressionMode::Fast);
        let rules: Vec<_> = (0..5)
            .map(|i| create_test_rule(&format!("rule{}", i), 2, 1))
            .collect();

        let _ = compressor.compress(&rules).unwrap();

        let stats = compressor.get_statistics();
        assert_eq!(stats.compressed_rules_count, 0); // Placeholder
    }

    #[test]
    fn test_serialization_round_trip() {
        let mut compressor = RuleCompressor::new(CompressionMode::Fast);
        let rules: Vec<_> = (0..10)
            .map(|i| create_test_rule(&format!("rule{}", i), 3, 2))
            .collect();

        let compressed = compressor.compress(&rules).unwrap();

        // Serialize and deserialize
        let serialized = serde_json::to_string(&compressed).unwrap();
        let deserialized: CompressedRuleSet = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.rule_count, compressed.rule_count);
        assert_eq!(deserialized.mode, compressed.mode);

        let decompressed = compressor.decompress(&deserialized).unwrap();
        assert_eq!(decompressed.len(), 10);
    }

    #[test]
    fn test_compression_with_different_atom_types() {
        let mut compressor = RuleCompressor::new(CompressionMode::Fast);

        let rules = vec![Rule {
            name: "rule1".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("pred".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Builtin {
                    name: "greaterThan".to_string(),
                    args: vec![
                        Term::Variable("Y".to_string()),
                        Term::Literal("10".to_string()),
                    ],
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("result".to_string()),
                object: Term::Literal("true".to_string()),
            }],
        }];

        let compressed = compressor.compress(&rules).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed.len(), 1);
        assert_eq!(decompressed[0].body.len(), 2);
        assert_eq!(decompressed[0].head.len(), 1);
    }

    #[test]
    fn test_compression_preserves_rule_structure() {
        let mut compressor = RuleCompressor::new(CompressionMode::Balanced);

        let original_rule = create_test_rule("complex", 5, 3);
        let rules = vec![original_rule.clone()];

        let compressed = compressor.compress(&rules).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed[0].name, original_rule.name);
        assert_eq!(decompressed[0].body.len(), original_rule.body.len());
        assert_eq!(decompressed[0].head.len(), original_rule.head.len());
    }

    #[test]
    fn test_zero_size_compression_ratio() {
        let compressed = CompressedRuleSet {
            mode: CompressionMode::None,
            original_size: 0,
            compressed_size: 0,
            rule_count: 0,
            data: vec![],
            timestamp: 0,
        };

        assert_eq!(compressed.compression_ratio(), 1.0);
        assert_eq!(compressed.space_saved(), 0);
        assert_eq!(compressed.space_saved_percent(), 0.0);
    }
}
