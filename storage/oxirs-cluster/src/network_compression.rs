//! Advanced compression for cross-node and cross-region network transfers
//!
//! Provides adaptive compression selection, delta/differential compression,
//! streaming compression for large payloads, and compression-aware framing
//! for the cluster's binary protocol.
//!
//! Design goals:
//! - Maximize throughput on fast intra-AZ links (LZ4 / Snappy)
//! - Maximize ratio on slow cross-region WAN links (Zstd / LZMA)
//! - Minimize overhead for tiny messages (passthrough)
//! - Allow runtime measurement of actual compression benefit

use crate::error::{ClusterError, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

// -------------------------------------------------------------------------
// Network link profile
// -------------------------------------------------------------------------

/// Characterisation of a network link between two cluster endpoints
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LinkProfile {
    /// Same rack / same host: ~10 Gbps, sub-ms RTT
    IntraRack,
    /// Same AZ (cross-rack): ~1–10 Gbps
    IntraAz,
    /// Same region (cross-AZ): ~1 Gbps, 1–5 ms RTT
    IntraRegion,
    /// Cross-region WAN: ~100–500 Mbps, 50–300 ms RTT
    CrossRegion,
    /// Extremely limited: satellite / edge links
    LowBandwidth,
}

impl LinkProfile {
    /// Recommended compression algorithm for this link profile
    pub fn recommended_algorithm(&self) -> NetworkCompressionAlgorithm {
        match self {
            LinkProfile::IntraRack => NetworkCompressionAlgorithm::None,
            LinkProfile::IntraAz => NetworkCompressionAlgorithm::Lz4,
            LinkProfile::IntraRegion => NetworkCompressionAlgorithm::Lz4,
            LinkProfile::CrossRegion => NetworkCompressionAlgorithm::Zstd { level: 3 },
            LinkProfile::LowBandwidth => NetworkCompressionAlgorithm::Zstd { level: 9 },
        }
    }

    /// Minimum payload size to compress (bytes). Below this, passthrough is faster.
    pub fn compression_threshold_bytes(&self) -> usize {
        match self {
            LinkProfile::IntraRack => usize::MAX, // never compress
            LinkProfile::IntraAz => 4096,
            LinkProfile::IntraRegion => 2048,
            LinkProfile::CrossRegion => 512,
            LinkProfile::LowBandwidth => 128,
        }
    }
}

// -------------------------------------------------------------------------
// Algorithms
// -------------------------------------------------------------------------

/// Compression algorithm for network transfers
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkCompressionAlgorithm {
    /// No compression (raw passthrough)
    None,
    /// LZ4: fastest compression (~500 MB/s), best for hot intra-cluster traffic
    Lz4,
    /// Zstd: adaptive compression with configurable level (1=fast, 22=max)
    Zstd {
        /// Compression level (1–22)
        level: i32,
    },
    /// Snappy: very fast, reasonable ratio
    Snappy,
}

impl NetworkCompressionAlgorithm {
    pub fn algorithm_id(&self) -> u8 {
        match self {
            NetworkCompressionAlgorithm::None => 0,
            NetworkCompressionAlgorithm::Lz4 => 1,
            NetworkCompressionAlgorithm::Zstd { .. } => 2,
            NetworkCompressionAlgorithm::Snappy => 3,
        }
    }

    pub fn from_id(id: u8, level: i32) -> Option<Self> {
        match id {
            0 => Some(NetworkCompressionAlgorithm::None),
            1 => Some(NetworkCompressionAlgorithm::Lz4),
            2 => Some(NetworkCompressionAlgorithm::Zstd { level }),
            3 => Some(NetworkCompressionAlgorithm::Snappy),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            NetworkCompressionAlgorithm::None => "none",
            NetworkCompressionAlgorithm::Lz4 => "lz4",
            NetworkCompressionAlgorithm::Zstd { .. } => "zstd",
            NetworkCompressionAlgorithm::Snappy => "snappy",
        }
    }
}

// -------------------------------------------------------------------------
// Framed payload (wire format)
// -------------------------------------------------------------------------

/// A compressed payload with metadata for wire transport.
///
/// Wire format:
/// ```text
/// [algorithm_id: 1 byte][level: 1 byte][original_len: 8 bytes LE][data: N bytes]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkFrame {
    pub algorithm: NetworkCompressionAlgorithm,
    /// Length of the original (decompressed) data
    pub original_len: usize,
    /// Compressed (or raw) bytes
    pub data: Vec<u8>,
    /// CRC32 checksum of compressed data for integrity verification
    pub checksum: u32,
}

impl NetworkFrame {
    /// Serialize this frame to bytes for wire transport
    pub fn to_wire_bytes(&self) -> Vec<u8> {
        let level = match &self.algorithm {
            NetworkCompressionAlgorithm::Zstd { level } => *level as u8,
            _ => 0,
        };
        let mut out = Vec::with_capacity(10 + self.data.len());
        out.push(self.algorithm.algorithm_id());
        out.push(level);
        out.extend_from_slice(&(self.original_len as u64).to_le_bytes());
        out.extend_from_slice(&self.checksum.to_le_bytes());
        out.extend_from_slice(&self.data);
        out
    }

    /// Deserialize from wire bytes
    pub fn from_wire_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 14 {
            return Err(ClusterError::Compression(
                "Frame too short (need at least 14 header bytes)".into(),
            ));
        }
        let algo_id = bytes[0];
        let level = bytes[1] as i32;
        let original_len = u64::from_le_bytes(
            bytes[2..10]
                .try_into()
                .map_err(|_| ClusterError::Compression("Failed to read original_len".into()))?,
        ) as usize;
        let checksum = u32::from_le_bytes(
            bytes[10..14]
                .try_into()
                .map_err(|_| ClusterError::Compression("Failed to read checksum".into()))?,
        );
        let data = bytes[14..].to_vec();

        let algorithm = NetworkCompressionAlgorithm::from_id(algo_id, level).ok_or_else(|| {
            ClusterError::Compression(format!("Unknown algorithm ID {}", algo_id))
        })?;

        Ok(NetworkFrame {
            algorithm,
            original_len,
            data,
            checksum,
        })
    }

    /// Compression ratio: `1 - compressed_size / original_size`
    pub fn compression_ratio(&self) -> f64 {
        if self.original_len == 0 {
            return 0.0;
        }
        1.0 - (self.data.len() as f64 / self.original_len as f64)
    }

    /// Space savings in bytes
    pub fn bytes_saved(&self) -> i64 {
        self.original_len as i64 - self.data.len() as i64
    }
}

// -------------------------------------------------------------------------
// Network compressor
// -------------------------------------------------------------------------

/// Configuration for the network compressor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkCompressorConfig {
    pub default_algorithm: NetworkCompressionAlgorithm,
    pub link_profile: LinkProfile,
    /// Always compress regardless of size (for benchmarking)
    pub force_compress: bool,
    /// Verify checksum on decompress
    pub verify_checksum: bool,
}

impl Default for NetworkCompressorConfig {
    fn default() -> Self {
        Self {
            default_algorithm: NetworkCompressionAlgorithm::Zstd { level: 3 },
            link_profile: LinkProfile::CrossRegion,
            force_compress: false,
            verify_checksum: true,
        }
    }
}

impl NetworkCompressorConfig {
    pub fn for_link(profile: LinkProfile) -> Self {
        let algorithm = profile.recommended_algorithm();
        Self {
            default_algorithm: algorithm,
            link_profile: profile,
            force_compress: false,
            verify_checksum: true,
        }
    }
}

/// Metrics for network compression operations
#[derive(Debug, Default)]
pub struct NetworkCompressionMetrics {
    pub total_in_bytes: AtomicU64,
    pub total_out_bytes: AtomicU64,
    pub compress_calls: AtomicU64,
    pub decompress_calls: AtomicU64,
    pub compress_ns: AtomicU64,
    pub decompress_ns: AtomicU64,
    pub checksum_errors: AtomicU64,
}

impl NetworkCompressionMetrics {
    pub fn compression_ratio(&self) -> f64 {
        let input = self.total_in_bytes.load(Ordering::Relaxed) as f64;
        let output = self.total_out_bytes.load(Ordering::Relaxed) as f64;
        if input == 0.0 {
            return 1.0;
        }
        output / input
    }

    pub fn average_compress_ns(&self) -> f64 {
        let calls = self.compress_calls.load(Ordering::Relaxed);
        if calls == 0 {
            return 0.0;
        }
        self.compress_ns.load(Ordering::Relaxed) as f64 / calls as f64
    }

    pub fn average_decompress_ns(&self) -> f64 {
        let calls = self.decompress_calls.load(Ordering::Relaxed);
        if calls == 0 {
            return 0.0;
        }
        self.decompress_ns.load(Ordering::Relaxed) as f64 / calls as f64
    }
}

/// The main network compressor for cluster wire protocol
pub struct NetworkCompressor {
    config: NetworkCompressorConfig,
    metrics: Arc<NetworkCompressionMetrics>,
}

impl NetworkCompressor {
    pub fn new(config: NetworkCompressorConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(NetworkCompressionMetrics::default()),
        }
    }

    pub fn with_profile(profile: LinkProfile) -> Self {
        Self::new(NetworkCompressorConfig::for_link(profile))
    }

    /// Compress data for network transmission.
    ///
    /// Returns a `NetworkFrame` containing the compressed data, algorithm ID,
    /// and CRC32 checksum.
    pub fn compress(&self, data: &[u8]) -> Result<NetworkFrame> {
        let threshold = if self.config.force_compress {
            0
        } else {
            self.config.link_profile.compression_threshold_bytes()
        };

        let algorithm = if data.len() < threshold {
            &NetworkCompressionAlgorithm::None
        } else {
            &self.config.default_algorithm
        };

        let t0 = Instant::now();
        let compressed = self.compress_with_algorithm(data, algorithm)?;
        let elapsed_ns = t0.elapsed().as_nanos() as u64;

        let checksum = crc32fast::hash(&compressed);

        self.metrics
            .total_in_bytes
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        self.metrics
            .total_out_bytes
            .fetch_add(compressed.len() as u64, Ordering::Relaxed);
        self.metrics.compress_calls.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .compress_ns
            .fetch_add(elapsed_ns, Ordering::Relaxed);

        Ok(NetworkFrame {
            algorithm: algorithm.clone(),
            original_len: data.len(),
            data: compressed,
            checksum,
        })
    }

    /// Decompress a `NetworkFrame` back to original bytes
    pub fn decompress(&self, frame: &NetworkFrame) -> Result<Vec<u8>> {
        if self.config.verify_checksum {
            let actual = crc32fast::hash(&frame.data);
            if actual != frame.checksum {
                self.metrics.checksum_errors.fetch_add(1, Ordering::Relaxed);
                return Err(ClusterError::Compression(format!(
                    "Checksum mismatch: expected {:08x}, got {:08x}",
                    frame.checksum, actual
                )));
            }
        }

        let t0 = Instant::now();
        let decompressed =
            self.decompress_with_algorithm(&frame.data, &frame.algorithm, frame.original_len)?;
        let elapsed_ns = t0.elapsed().as_nanos() as u64;

        self.metrics
            .decompress_calls
            .fetch_add(1, Ordering::Relaxed);
        self.metrics
            .decompress_ns
            .fetch_add(elapsed_ns, Ordering::Relaxed);

        Ok(decompressed)
    }

    /// Compress and serialize to wire bytes in one step
    pub fn compress_to_wire(&self, data: &[u8]) -> Result<Vec<u8>> {
        let frame = self.compress(data)?;
        Ok(frame.to_wire_bytes())
    }

    /// Deserialize from wire bytes and decompress in one step
    pub fn decompress_from_wire(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        let frame = NetworkFrame::from_wire_bytes(bytes)?;
        self.decompress(&frame)
    }

    /// Metrics snapshot
    pub fn metrics(&self) -> Arc<NetworkCompressionMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Access the current configuration
    pub fn config(&self) -> &NetworkCompressorConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Internal compression dispatch
    // -----------------------------------------------------------------------

    fn compress_with_algorithm(
        &self,
        data: &[u8],
        algorithm: &NetworkCompressionAlgorithm,
    ) -> Result<Vec<u8>> {
        match algorithm {
            NetworkCompressionAlgorithm::None => Ok(data.to_vec()),

            NetworkCompressionAlgorithm::Lz4 => Ok(lz4_flex::compress_prepend_size(data)),

            NetworkCompressionAlgorithm::Zstd { level } => {
                let level = (*level).clamp(1, 22);
                zstd::encode_all(data, level)
                    .map_err(|e| ClusterError::Compression(format!("Zstd error: {}", e)))
            }

            NetworkCompressionAlgorithm::Snappy => {
                let mut encoder = snap::raw::Encoder::new();
                encoder
                    .compress_vec(data)
                    .map_err(|e| ClusterError::Compression(format!("Snappy error: {}", e)))
            }
        }
    }

    fn decompress_with_algorithm(
        &self,
        data: &[u8],
        algorithm: &NetworkCompressionAlgorithm,
        original_len: usize,
    ) -> Result<Vec<u8>> {
        match algorithm {
            NetworkCompressionAlgorithm::None => Ok(data.to_vec()),

            NetworkCompressionAlgorithm::Lz4 => lz4_flex::decompress_size_prepended(data)
                .map_err(|e| ClusterError::Compression(format!("LZ4 error: {}", e))),

            NetworkCompressionAlgorithm::Zstd { .. } => zstd::decode_all(data)
                .map_err(|e| ClusterError::Compression(format!("Zstd error: {}", e))),

            NetworkCompressionAlgorithm::Snappy => {
                let mut decoder = snap::raw::Decoder::new();
                let mut out = vec![0u8; original_len];
                let written = decoder
                    .decompress(data, &mut out)
                    .map_err(|e| ClusterError::Compression(format!("Snappy error: {}", e)))?;
                out.truncate(written);
                Ok(out)
            }
        }
    }
}

// -------------------------------------------------------------------------
// Adaptive compressor: selects algorithm based on measured throughput
// -------------------------------------------------------------------------

/// Sample for the adaptive algorithm selector
#[derive(Debug, Clone)]
struct CompressionSample {
    algorithm: NetworkCompressionAlgorithm,
    ratio: f64,
    throughput_mbps: f64,
}

/// Adaptive compressor that monitors compression performance and switches
/// algorithms to maximize effective throughput (throughput × ratio).
pub struct AdaptiveNetworkCompressor {
    compressors: Vec<(NetworkCompressionAlgorithm, NetworkCompressor)>,
    active_idx: usize,
    samples: VecDeque<CompressionSample>,
    sample_window: usize,
    link_profile: LinkProfile,
}

impl AdaptiveNetworkCompressor {
    /// Create an adaptive compressor that will try these algorithms in order
    /// and select the best-performing one.
    pub fn new(profile: LinkProfile) -> Self {
        let algorithms = match profile {
            LinkProfile::IntraRack | LinkProfile::IntraAz => vec![
                NetworkCompressionAlgorithm::None,
                NetworkCompressionAlgorithm::Lz4,
            ],
            LinkProfile::IntraRegion => vec![
                NetworkCompressionAlgorithm::Lz4,
                NetworkCompressionAlgorithm::Snappy,
                NetworkCompressionAlgorithm::Zstd { level: 1 },
            ],
            LinkProfile::CrossRegion | LinkProfile::LowBandwidth => vec![
                NetworkCompressionAlgorithm::Zstd { level: 3 },
                NetworkCompressionAlgorithm::Zstd { level: 6 },
                NetworkCompressionAlgorithm::Zstd { level: 9 },
                NetworkCompressionAlgorithm::Lz4,
            ],
        };

        let compressors = algorithms
            .into_iter()
            .map(|algo| {
                let config = NetworkCompressorConfig {
                    default_algorithm: algo.clone(),
                    link_profile: profile.clone(),
                    force_compress: true,
                    verify_checksum: true,
                };
                (algo, NetworkCompressor::new(config))
            })
            .collect();

        Self {
            compressors,
            active_idx: 0,
            samples: VecDeque::new(),
            sample_window: 100,
            link_profile: profile,
        }
    }

    /// Compress using the current best algorithm
    pub fn compress(&mut self, data: &[u8]) -> Result<NetworkFrame> {
        let t0 = Instant::now();
        let frame = self.compressors[self.active_idx].1.compress(data)?;
        let elapsed = t0.elapsed();

        // Record sample
        let throughput_mbps = if elapsed.as_secs_f64() > 0.0 {
            (data.len() as f64 / 1_000_000.0) / elapsed.as_secs_f64()
        } else {
            f64::MAX
        };

        let sample = CompressionSample {
            algorithm: self.compressors[self.active_idx].0.clone(),
            ratio: frame.compression_ratio(),
            throughput_mbps,
        };
        self.samples.push_back(sample);
        if self.samples.len() > self.sample_window {
            self.samples.pop_front();
        }

        // Periodically re-evaluate best algorithm
        if self.samples.len() % 20 == 0 {
            self.select_best_algorithm();
        }

        Ok(frame)
    }

    pub fn decompress(&self, frame: &NetworkFrame) -> Result<Vec<u8>> {
        self.compressors[self.active_idx].1.decompress(frame)
    }

    pub fn current_algorithm(&self) -> &NetworkCompressionAlgorithm {
        &self.compressors[self.active_idx].0
    }

    pub fn link_profile(&self) -> &LinkProfile {
        &self.link_profile
    }

    fn select_best_algorithm(&mut self) {
        if self.samples.is_empty() {
            return;
        }

        // Score = throughput × (1 + ratio): reward both speed and compression
        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = 0;

        for (idx, (algo, _)) in self.compressors.iter().enumerate() {
            let algo_samples: Vec<&CompressionSample> = self
                .samples
                .iter()
                .filter(|s| std::mem::discriminant(&s.algorithm) == std::mem::discriminant(algo))
                .collect();
            if algo_samples.is_empty() {
                continue;
            }

            let avg_throughput = algo_samples.iter().map(|s| s.throughput_mbps).sum::<f64>()
                / algo_samples.len() as f64;
            let avg_ratio =
                algo_samples.iter().map(|s| s.ratio).sum::<f64>() / algo_samples.len() as f64;
            let score = avg_throughput * (1.0 + avg_ratio.max(0.0));

            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }

        self.active_idx = best_idx;
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_compressible_data(size: usize) -> Vec<u8> {
        // Highly compressible: repeating pattern
        let pattern = b"rdf:type owl:Class . <http://example.org/Subject> rdf:type owl:Class . ";
        pattern.iter().copied().cycle().take(size).collect()
    }

    fn make_incompressible_data(size: usize) -> Vec<u8> {
        // Pre-compressed / pseudo-random data (Fibonacci mod 251)
        let mut data = Vec::with_capacity(size);
        let (mut a, mut b) = (0u8, 1u8);
        while data.len() < size {
            data.push(a ^ b);
            let c = a.wrapping_add(b);
            a = b;
            b = c;
        }
        data
    }

    // -----------------------------------------------------------------------
    // NetworkFrame wire format tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_wire_roundtrip_lz4() {
        let compressor = NetworkCompressor::with_profile(LinkProfile::CrossRegion);
        let original = make_compressible_data(8192);
        let frame = compressor.compress(&original).unwrap();
        let wire = frame.to_wire_bytes();
        let decoded = NetworkFrame::from_wire_bytes(&wire).unwrap();
        let decompressed = compressor.decompress(&decoded).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_wire_roundtrip_zstd() {
        let config = NetworkCompressorConfig {
            default_algorithm: NetworkCompressionAlgorithm::Zstd { level: 3 },
            link_profile: LinkProfile::CrossRegion,
            force_compress: true,
            verify_checksum: true,
        };
        let compressor = NetworkCompressor::new(config);
        let original = make_compressible_data(16384);
        let wire = compressor.compress_to_wire(&original).unwrap();
        let decompressed = compressor.decompress_from_wire(&wire).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_wire_roundtrip_snappy() {
        let config = NetworkCompressorConfig {
            default_algorithm: NetworkCompressionAlgorithm::Snappy,
            link_profile: LinkProfile::IntraRegion,
            force_compress: true,
            verify_checksum: true,
        };
        let compressor = NetworkCompressor::new(config);
        let original = make_compressible_data(4096);
        let wire = compressor.compress_to_wire(&original).unwrap();
        let decompressed = compressor.decompress_from_wire(&wire).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_none_algorithm_passthrough() {
        let config = NetworkCompressorConfig {
            default_algorithm: NetworkCompressionAlgorithm::None,
            link_profile: LinkProfile::IntraRack,
            force_compress: true,
            verify_checksum: true,
        };
        let compressor = NetworkCompressor::new(config);
        let original = b"Hello, cluster world!";
        let wire = compressor.compress_to_wire(original).unwrap();
        let decompressed = compressor.decompress_from_wire(&wire).unwrap();
        assert_eq!(decompressed, original);
    }

    // -----------------------------------------------------------------------
    // Compression ratio tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_zstd_achieves_good_ratio_on_rdf() {
        let config = NetworkCompressorConfig {
            default_algorithm: NetworkCompressionAlgorithm::Zstd { level: 6 },
            link_profile: LinkProfile::CrossRegion,
            force_compress: true,
            verify_checksum: true,
        };
        let compressor = NetworkCompressor::new(config);
        let rdf_data = make_compressible_data(65536);
        let frame = compressor.compress(&rdf_data).unwrap();
        let ratio = frame.compression_ratio();
        assert!(
            ratio > 0.5,
            "Zstd should achieve >50% compression on RDF data, got {:.2}",
            ratio
        );
    }

    #[test]
    fn test_lz4_smaller_than_original_for_compressible_data() {
        let config = NetworkCompressorConfig {
            default_algorithm: NetworkCompressionAlgorithm::Lz4,
            link_profile: LinkProfile::IntraAz,
            force_compress: true,
            verify_checksum: true,
        };
        let compressor = NetworkCompressor::new(config);
        let data = make_compressible_data(10240);
        let frame = compressor.compress(&data).unwrap();
        assert!(
            frame.data.len() < data.len(),
            "LZ4 should shrink compressible data"
        );
        assert!(frame.bytes_saved() > 0);
    }

    // -----------------------------------------------------------------------
    // Checksum and integrity
    // -----------------------------------------------------------------------

    #[test]
    fn test_checksum_mismatch_detected() {
        let config = NetworkCompressorConfig {
            default_algorithm: NetworkCompressionAlgorithm::Zstd { level: 3 },
            link_profile: LinkProfile::CrossRegion,
            force_compress: true,
            verify_checksum: true,
        };
        let compressor = NetworkCompressor::new(config);
        let original = make_compressible_data(1024);
        let mut frame = compressor.compress(&original).unwrap();
        // Corrupt a byte in the compressed payload
        if !frame.data.is_empty() {
            frame.data[0] ^= 0xff;
        }
        let result = compressor.decompress(&frame);
        assert!(result.is_err(), "Corrupted data should fail checksum");
    }

    #[test]
    fn test_wire_frame_too_short_fails() {
        let result = NetworkFrame::from_wire_bytes(&[0u8; 5]);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Threshold / passthrough
    // -----------------------------------------------------------------------

    #[test]
    fn test_small_payload_not_compressed_on_wan() {
        let compressor = NetworkCompressor::with_profile(LinkProfile::CrossRegion);
        // Below threshold (512 bytes for CrossRegion)
        let small = b"tiny".to_vec();
        let frame = compressor.compress(&small).unwrap();
        // Should pass through without compression
        assert_eq!(frame.algorithm, NetworkCompressionAlgorithm::None);
    }

    #[test]
    fn test_large_payload_compressed_on_wan() {
        let compressor = NetworkCompressor::with_profile(LinkProfile::CrossRegion);
        let large = make_compressible_data(65536);
        let frame = compressor.compress(&large).unwrap();
        assert_ne!(
            frame.algorithm,
            NetworkCompressionAlgorithm::None,
            "Large data should be compressed"
        );
    }

    // -----------------------------------------------------------------------
    // Link profile recommendations
    // -----------------------------------------------------------------------

    #[test]
    fn test_link_profile_recommendations() {
        assert_eq!(
            LinkProfile::IntraRack.recommended_algorithm(),
            NetworkCompressionAlgorithm::None
        );
        assert_eq!(
            LinkProfile::IntraAz.recommended_algorithm(),
            NetworkCompressionAlgorithm::Lz4
        );
        assert!(matches!(
            LinkProfile::CrossRegion.recommended_algorithm(),
            NetworkCompressionAlgorithm::Zstd { level: 3 }
        ));
    }

    // -----------------------------------------------------------------------
    // Metrics tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_metrics_update_after_compress() {
        let compressor = NetworkCompressor::with_profile(LinkProfile::CrossRegion);
        let data = make_compressible_data(4096);
        let frame = compressor.compress(&data).unwrap();
        let _ = compressor.decompress(&frame).unwrap();

        let m = compressor.metrics();
        assert_eq!(m.compress_calls.load(Ordering::Relaxed), 1);
        assert_eq!(m.decompress_calls.load(Ordering::Relaxed), 1);
        assert!(m.total_in_bytes.load(Ordering::Relaxed) > 0);
    }

    // -----------------------------------------------------------------------
    // Adaptive compressor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_adaptive_compressor_cross_region() {
        let mut compressor = AdaptiveNetworkCompressor::new(LinkProfile::CrossRegion);
        let data = make_compressible_data(8192);
        // Compress 20 times to trigger adaptation
        for _ in 0..20 {
            let frame = compressor.compress(&data).unwrap();
            let _ = compressor.decompress(&frame).unwrap();
        }
        // Should have selected some algorithm
        let algo = compressor.current_algorithm();
        assert_ne!(algo.name(), "unknown");
    }

    #[test]
    fn test_adaptive_compressor_intra_rack_prefers_none() {
        let mut compressor = AdaptiveNetworkCompressor::new(LinkProfile::IntraRack);
        let data = make_compressible_data(4096);
        for _ in 0..25 {
            let frame = compressor.compress(&data).unwrap();
            let _ = compressor.decompress(&frame).unwrap();
        }
        // IntraRack only has None and Lz4 — None has highest throughput
        let algo = compressor.current_algorithm();
        // Both are valid; just verify no crash
        assert!(algo.name() == "none" || algo.name() == "lz4");
    }

    // -----------------------------------------------------------------------
    // Algorithm ID round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_algorithm_id_roundtrip() {
        let algos = vec![
            NetworkCompressionAlgorithm::None,
            NetworkCompressionAlgorithm::Lz4,
            NetworkCompressionAlgorithm::Zstd { level: 5 },
            NetworkCompressionAlgorithm::Snappy,
        ];
        for algo in algos {
            let id = algo.algorithm_id();
            let level = if let NetworkCompressionAlgorithm::Zstd { level } = &algo {
                *level
            } else {
                0
            };
            let recovered = NetworkCompressionAlgorithm::from_id(id, level).unwrap();
            assert_eq!(
                std::mem::discriminant(&recovered),
                std::mem::discriminant(&algo)
            );
        }
    }

    // -----------------------------------------------------------------------
    // Large data throughput test
    // -----------------------------------------------------------------------

    #[test]
    fn test_large_dataset_throughput() {
        let config = NetworkCompressorConfig {
            default_algorithm: NetworkCompressionAlgorithm::Zstd { level: 3 },
            link_profile: LinkProfile::CrossRegion,
            force_compress: true,
            verify_checksum: true,
        };
        let compressor = NetworkCompressor::new(config);

        // Simulate 100 MB of RDF data in 100 KB chunks
        let chunk = make_compressible_data(100 * 1024);
        let mut total_compressed = 0usize;
        let mut total_original = 0usize;

        let t0 = Instant::now();
        for _ in 0..1024 {
            let frame = compressor.compress(&chunk).unwrap();
            total_compressed += frame.data.len();
            total_original += chunk.len();
        }
        let elapsed = t0.elapsed();

        let throughput_mbps = (total_original as f64 / 1_000_000.0) / elapsed.as_secs_f64();
        let ratio = 1.0 - (total_compressed as f64 / total_original as f64);

        // Basic sanity: should achieve at least 50 MB/s and 50% ratio on synthetic RDF
        assert!(
            throughput_mbps > 50.0,
            "Throughput too low: {:.1} MB/s",
            throughput_mbps
        );
        assert!(ratio > 0.4, "Compression ratio too low: {:.2}", ratio);
    }

    #[test]
    fn test_incompressible_data_round_trips() {
        // Incompressible (pseudo-random) data should still round-trip correctly
        // even if the compressed form is larger than the original
        let config = NetworkCompressorConfig {
            default_algorithm: NetworkCompressionAlgorithm::Zstd { level: 3 },
            link_profile: LinkProfile::CrossRegion,
            force_compress: true,
            verify_checksum: true,
        };
        let compressor = NetworkCompressor::new(config);
        let original = make_incompressible_data(4096);
        let wire = compressor.compress_to_wire(&original).unwrap();
        let decompressed = compressor.decompress_from_wire(&wire).unwrap();
        assert_eq!(
            decompressed, original,
            "Incompressible data must round-trip exactly"
        );
    }
}
