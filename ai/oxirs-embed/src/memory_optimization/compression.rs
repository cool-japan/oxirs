//! Compression for cached embeddings and model data

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

/// Compression algorithm
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// Zstandard (fast, good compression)
    Zstd,
    /// LZ4 (very fast, moderate compression)
    Lz4,
    /// Gzip (slower, better compression)
    Gzip,
    /// No compression
    None,
}

impl Default for CompressionAlgorithm {
    fn default() -> Self {
        Self::Zstd
    }
}

/// Compressor for data
pub struct Compressor {
    algorithm: CompressionAlgorithm,
}

impl Compressor {
    pub fn new(algorithm: CompressionAlgorithm) -> Self {
        Self { algorithm }
    }

    /// Compress data
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.algorithm {
            CompressionAlgorithm::Zstd => self.compress_zstd(data),
            CompressionAlgorithm::Lz4 => self.compress_lz4(data),
            CompressionAlgorithm::Gzip => self.compress_gzip(data),
            CompressionAlgorithm::None => Ok(data.to_vec()),
        }
    }

    /// Decompress data
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.algorithm {
            CompressionAlgorithm::Zstd => self.decompress_zstd(data),
            CompressionAlgorithm::Lz4 => self.decompress_lz4(data),
            CompressionAlgorithm::Gzip => self.decompress_gzip(data),
            CompressionAlgorithm::None => Ok(data.to_vec()),
        }
    }

    fn compress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::encode_all(data, 3).map_err(|e| anyhow!("Zstd compression failed: {}", e))
    }

    fn decompress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::decode_all(data).map_err(|e| anyhow!("Zstd decompression failed: {}", e))
    }

    fn compress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(lz4_flex::compress_prepend_size(data))
    }

    fn decompress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        lz4_flex::decompress_size_prepended(data)
            .map_err(|e| anyhow!("LZ4 decompression failed: {}", e))
    }

    fn compress_gzip(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(data)
            .map_err(|e| anyhow!("Gzip compression failed: {}", e))?;
        encoder
            .finish()
            .map_err(|e| anyhow!("Gzip compression failed: {}", e))
    }

    fn decompress_gzip(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::GzDecoder;

        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .map_err(|e| anyhow!("Gzip decompression failed: {}", e))?;
        Ok(decompressed)
    }

    /// Calculate compression ratio
    pub fn compression_ratio(&self, original: &[u8], compressed: &[u8]) -> f64 {
        if compressed.is_empty() {
            return 0.0;
        }
        original.len() as f64 / compressed.len() as f64
    }
}

/// Compressed embedding storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedEmbedding {
    pub compressed_data: Vec<u8>,
    pub original_size: usize,
    pub algorithm: CompressionAlgorithm,
}

impl CompressedEmbedding {
    /// Compress embeddings (f32 array)
    pub fn from_embeddings(embeddings: &[f32], algorithm: CompressionAlgorithm) -> Result<Self> {
        let compressor = Compressor::new(algorithm);

        // Convert f32 to bytes
        let bytes: Vec<u8> = embeddings
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let compressed_data = compressor.compress(&bytes)?;

        Ok(Self {
            compressed_data,
            original_size: bytes.len(),
            algorithm,
        })
    }

    /// Decompress to embeddings
    pub fn to_embeddings(&self) -> Result<Vec<f32>> {
        let compressor = Compressor::new(self.algorithm);
        let bytes = compressor.decompress(&self.compressed_data)?;

        // Convert bytes back to f32
        let embeddings: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(embeddings)
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        self.original_size as f64 / self.compressed_data.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zstd_compression() {
        let compressor = Compressor::new(CompressionAlgorithm::Zstd);
        let data = vec![42u8; 1000];

        let compressed = compressor.compress(&data).unwrap();
        assert!(compressed.len() < data.len());

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_lz4_compression() {
        let compressor = Compressor::new(CompressionAlgorithm::Lz4);
        let data = vec![42u8; 1000];

        let compressed = compressor.compress(&data).unwrap();
        assert!(compressed.len() < data.len());

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_gzip_compression() {
        let compressor = Compressor::new(CompressionAlgorithm::Gzip);
        let data = vec![42u8; 1000];

        let compressed = compressor.compress(&data).unwrap();
        assert!(compressed.len() < data.len());

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_compressed_embedding() {
        let embeddings = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let compressed = CompressedEmbedding::from_embeddings(&embeddings, CompressionAlgorithm::Zstd).unwrap();

        let decompressed = compressed.to_embeddings().unwrap();
        assert_eq!(decompressed.len(), embeddings.len());

        for (a, b) in embeddings.iter().zip(decompressed.iter()) {
            assert!((a - b).abs() < 0.001);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let compressor = Compressor::new(CompressionAlgorithm::Zstd);
        let data = vec![42u8; 1000];

        let compressed = compressor.compress(&data).unwrap();
        let ratio = compressor.compression_ratio(&data, &compressed);

        assert!(ratio > 1.0); // Should have some compression
    }

    #[test]
    fn test_no_compression() {
        let compressor = Compressor::new(CompressionAlgorithm::None);
        let data = vec![1, 2, 3, 4, 5];

        let compressed = compressor.compress(&data).unwrap();
        assert_eq!(compressed, data);

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }
}
