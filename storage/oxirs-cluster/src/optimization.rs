//! Storage and Serialization Optimization Module
//!
//! Provides binary encoding, compression, and performance optimizations
//! for distributed storage operations.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncWrite, AsyncWriteExt};
use zstd;

/// Binary serialization format with compression
#[derive(Debug, Clone)]
pub enum SerializationFormat {
    /// JSON format (fallback)
    Json,
    /// MessagePack binary format
    MessagePack,
    /// CBOR binary format
    Cbor,
    /// Bincode format (fastest)
    Bincode,
}

/// Compression algorithms
#[derive(Debug, Clone)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// LZ4 (fast compression/decompression)
    Lz4,
    /// Zstd (good compression ratio)
    Zstd,
    /// Deflate (standard compression)
    Deflate,
}

/// Serialization configuration
#[derive(Debug, Clone)]
pub struct SerializationConfig {
    pub format: SerializationFormat,
    pub compression: CompressionAlgorithm,
    pub compression_level: i32,
    pub enable_checksums: bool,
}

impl Default for SerializationConfig {
    fn default() -> Self {
        Self {
            format: SerializationFormat::Bincode,
            compression: CompressionAlgorithm::Lz4,
            compression_level: 6,
            enable_checksums: true,
        }
    }
}

/// Binary serializer with compression and integrity checking
pub struct BinarySerializer {
    config: SerializationConfig,
}

impl BinarySerializer {
    pub fn new(config: SerializationConfig) -> Self {
        Self { config }
    }

    /// Serialize data with compression and checksums
    pub fn serialize<T: Serialize>(&self, data: &T) -> Result<Vec<u8>> {
        // First serialize to binary format
        let mut binary_data = match self.config.format {
            SerializationFormat::Json => serde_json::to_vec(data)?,
            SerializationFormat::MessagePack => rmp_serde::to_vec(data)?,
            SerializationFormat::Cbor => serde_cbor::to_vec(data)?,
            SerializationFormat::Bincode => {
                oxicode::serde::encode_to_vec(data, oxicode::config::standard())?
            }
        };

        // Apply compression if enabled
        binary_data = match self.config.compression {
            CompressionAlgorithm::None => binary_data,
            CompressionAlgorithm::Lz4 => self.compress_lz4(&binary_data)?,
            CompressionAlgorithm::Zstd => self.compress_zstd(&binary_data)?,
            CompressionAlgorithm::Deflate => self.compress_deflate(&binary_data)?,
        };

        // Add checksum if enabled
        if self.config.enable_checksums {
            let checksum = crc32fast::hash(&binary_data);
            let mut result = Vec::with_capacity(binary_data.len() + 8);
            result.extend_from_slice(&checksum.to_le_bytes());
            result.extend_from_slice(&(binary_data.len() as u32).to_le_bytes());
            result.extend_from_slice(&binary_data);
            Ok(result)
        } else {
            Ok(binary_data)
        }
    }

    /// Deserialize data with decompression and checksum validation
    pub fn deserialize<T: for<'de> Deserialize<'de>>(&self, data: &[u8]) -> Result<T> {
        let (binary_data, _expected_checksum) = if self.config.enable_checksums {
            if data.len() < 8 {
                return Err(anyhow::anyhow!("Data too short for checksum"));
            }

            let checksum = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            let length = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

            if data.len() < 8 + length {
                return Err(anyhow::anyhow!("Data length mismatch"));
            }

            let binary_data = &data[8..8 + length];
            let computed_checksum = crc32fast::hash(binary_data);

            if checksum != computed_checksum {
                return Err(anyhow::anyhow!("Checksum validation failed"));
            }

            (binary_data, Some(checksum))
        } else {
            (data, None)
        };

        // Decompress if needed
        let decompressed_data = match self.config.compression {
            CompressionAlgorithm::None => binary_data.to_vec(),
            CompressionAlgorithm::Lz4 => self.decompress_lz4(binary_data)?,
            CompressionAlgorithm::Zstd => self.decompress_zstd(binary_data)?,
            CompressionAlgorithm::Deflate => self.decompress_deflate(binary_data)?,
        };

        // Deserialize from binary format
        let result = match self.config.format {
            SerializationFormat::Json => serde_json::from_slice(&decompressed_data)?,
            SerializationFormat::MessagePack => rmp_serde::from_slice(&decompressed_data)?,
            SerializationFormat::Cbor => serde_cbor::from_slice(&decompressed_data)?,
            SerializationFormat::Bincode => {
                oxicode::serde::decode_from_slice(&decompressed_data, oxicode::config::standard())
                    .map(|(v, _)| v)?
            }
        };

        Ok(result)
    }

    /// Compress data using LZ4
    fn compress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        use lz4_flex::compress_prepend_size;
        Ok(compress_prepend_size(data))
    }

    /// Decompress LZ4 data
    fn decompress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        use lz4_flex::decompress_size_prepended;
        decompress_size_prepended(data)
            .map_err(|e| anyhow::anyhow!("LZ4 decompression failed: {}", e))
    }

    /// Compress data using Zstd
    fn compress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::encode_all(data, self.config.compression_level)
            .map_err(|e| anyhow::anyhow!("Zstd compression failed: {}", e))
    }

    /// Decompress Zstd data
    fn decompress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::decode_all(data).map_err(|e| anyhow::anyhow!("Zstd decompression failed: {}", e))
    }

    /// Compress data using Deflate
    fn compress_deflate(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::write::ZlibEncoder;
        use flate2::Compression;
        use std::io::Write;

        let mut encoder = ZlibEncoder::new(
            Vec::new(),
            Compression::new(self.config.compression_level as u32),
        );
        encoder.write_all(data)?;
        encoder
            .finish()
            .map_err(|e| anyhow::anyhow!("Deflate compression failed: {}", e))
    }

    /// Decompress Deflate data
    fn decompress_deflate(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::ZlibDecoder;
        use std::io::Read;

        let mut decoder = ZlibDecoder::new(data);
        let mut result = Vec::new();
        decoder
            .read_to_end(&mut result)
            .map_err(|e| anyhow::anyhow!("Deflate decompression failed: {}", e))?;
        Ok(result)
    }
}

/// Atomic file writer with transaction semantics
pub struct AtomicFileWriter {
    temp_path: PathBuf,
    final_path: PathBuf,
    file: Option<File>,
}

impl AtomicFileWriter {
    /// Create a new atomic file writer
    pub async fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let final_path = path.as_ref().to_path_buf();
        let temp_path = final_path.with_extension("tmp");

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&temp_path)
            .await?;

        Ok(Self {
            temp_path,
            final_path,
            file: Some(file),
        })
    }

    /// Write data to the temporary file
    pub async fn write(&mut self, data: &[u8]) -> Result<()> {
        if let Some(ref mut file) = self.file {
            file.write_all(data).await?;
            file.sync_all().await?;
        }
        Ok(())
    }

    /// Commit the write by atomically moving the temp file to the final location
    pub async fn commit(mut self) -> Result<()> {
        if let Some(file) = self.file.take() {
            file.sync_all().await?;
            drop(file);
            tokio::fs::rename(&self.temp_path, &self.final_path).await?;
        }
        Ok(())
    }

    /// Abort the write by removing the temporary file
    pub async fn abort(self) -> Result<()> {
        if self.temp_path.exists() {
            tokio::fs::remove_file(&self.temp_path).await?;
        }
        Ok(())
    }
}

impl Drop for AtomicFileWriter {
    fn drop(&mut self) {
        if self.temp_path.exists() {
            let _ = std::fs::remove_file(&self.temp_path);
        }
    }
}

impl AsyncWrite for AtomicFileWriter {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        if let Some(ref mut file) = self.file {
            let file_pin = std::pin::Pin::new(file);
            file_pin.poll_write(cx, buf)
        } else {
            std::task::Poll::Ready(Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "File not open",
            )))
        }
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        if let Some(ref mut file) = self.file {
            let file_pin = std::pin::Pin::new(file);
            file_pin.poll_flush(cx)
        } else {
            std::task::Poll::Ready(Ok(()))
        }
    }

    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        if let Some(ref mut file) = self.file {
            let file_pin = std::pin::Pin::new(file);
            file_pin.poll_shutdown(cx)
        } else {
            std::task::Poll::Ready(Ok(()))
        }
    }
}

/// Corruption detector using checksums and integrity verification
pub struct CorruptionDetector {
    enable_deep_scan: bool,
}

impl CorruptionDetector {
    pub fn new(enable_deep_scan: bool) -> Self {
        Self { enable_deep_scan }
    }

    /// Verify file integrity using checksums
    pub async fn verify_file_integrity<P: AsRef<Path>>(&self, path: P) -> Result<bool> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(false);
        }

        let metadata = tokio::fs::metadata(path).await?;
        if metadata.len() == 0 {
            return Ok(true); // Empty file is considered valid
        }

        // Basic integrity check - read the entire file
        let data = tokio::fs::read(path).await?;

        // Verify file isn't truncated
        if (data.len() as u64) != metadata.len() {
            return Ok(false);
        }

        if self.enable_deep_scan {
            // Perform deep integrity check (checksum verification)
            self.verify_content_integrity(&data).await
        } else {
            Ok(true)
        }
    }

    /// Verify content integrity using checksums
    async fn verify_content_integrity(&self, data: &[u8]) -> Result<bool> {
        // For now, just verify the data is valid serialized format
        // In a full implementation, this would verify embedded checksums
        if data.len() >= 8 {
            // Check if it looks like checksummed data
            let checksum = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            let length = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

            if data.len() >= 8 + length {
                let content = &data[8..8 + length];
                let computed = crc32fast::hash(content);
                return Ok(checksum == computed);
            }
        }

        Ok(true) // Assume valid if no checksum format detected
    }

    /// Repair corrupted files by attempting to recover valid data
    pub async fn attempt_repair<P: AsRef<Path>>(&self, path: P) -> Result<bool> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(false);
        }

        // Try to read and validate the file
        let data = tokio::fs::read(path).await?;

        // If the file appears to have partial checksum data, try to extract valid parts
        if data.len() >= 8 {
            let length = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

            // If the length seems reasonable, try to extract that portion
            if length > 0 && data.len() >= 8 + length {
                let valid_data = &data[0..8 + length];

                // Write the potentially valid data back
                let backup_path = path.with_extension("backup");
                tokio::fs::rename(path, &backup_path).await?;
                tokio::fs::write(path, valid_data).await?;

                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Validate file using comprehensive checks
    pub async fn validate_file<P: AsRef<Path>>(&self, path: P) -> Result<bool> {
        self.verify_file_integrity(path).await
    }
}

/// Schema evolution support for backward compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl SchemaVersion {
    pub const CURRENT: SchemaVersion = SchemaVersion {
        major: 1,
        minor: 0,
        patch: 0,
    };

    pub fn is_compatible(&self, other: &SchemaVersion) -> bool {
        self.major == other.major && self.minor >= other.minor
    }
}

/// Versioned data wrapper for schema evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedData<T> {
    pub version: SchemaVersion,
    pub data: T,
}

impl<T> VersionedData<T> {
    pub fn new(data: T) -> Self {
        Self {
            version: SchemaVersion::CURRENT,
            data,
        }
    }

    pub fn validate_compatibility(&self) -> Result<()> {
        if !SchemaVersion::CURRENT.is_compatible(&self.version) {
            return Err(anyhow::anyhow!(
                "Incompatible schema version: {:?}, current: {:?}",
                self.version,
                SchemaVersion::CURRENT
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_binary_serialization() {
        let config = SerializationConfig::default();
        let serializer = BinarySerializer::new(config);

        let data = vec!["hello".to_string(), "world".to_string()];
        let serialized = serializer.serialize(&data).unwrap();
        let deserialized: Vec<String> = serializer.deserialize(&serialized).unwrap();

        assert_eq!(data, deserialized);
    }

    #[tokio::test]
    async fn test_atomic_file_writer() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.dat");

        let mut writer = AtomicFileWriter::new(&file_path).await.unwrap();
        writer.write_all(b"test data").await.unwrap();
        writer.commit().await.unwrap();

        let content = tokio::fs::read(&file_path).await.unwrap();
        assert_eq!(content, b"test data");
    }

    #[tokio::test]
    async fn test_corruption_detection() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.dat");

        let config = SerializationConfig::default();
        let serializer = BinarySerializer::new(config);
        let detector = CorruptionDetector::new(true);

        let data = vec!["test".to_string()];
        let serialized = serializer.serialize(&data).unwrap();
        tokio::fs::write(&file_path, &serialized).await.unwrap();

        assert!(detector.validate_file(&file_path).await.unwrap());

        // Corrupt the file - flip a bit in the middle of the data
        let mut corrupted = serialized.clone();
        let corrupt_idx = corrupted.len() / 2; // Use middle of actual data
        corrupted[corrupt_idx] = !corrupted[corrupt_idx]; // Flip a bit
        tokio::fs::write(&file_path, &corrupted).await.unwrap();

        assert!(!detector.validate_file(&file_path).await.unwrap());
    }

    #[test]
    fn test_schema_version_compatibility() {
        let v1_0_0 = SchemaVersion {
            major: 1,
            minor: 0,
            patch: 0,
        };
        let v1_1_0 = SchemaVersion {
            major: 1,
            minor: 1,
            patch: 0,
        };
        let v2_0_0 = SchemaVersion {
            major: 2,
            minor: 0,
            patch: 0,
        };

        assert!(v1_1_0.is_compatible(&v1_0_0));
        assert!(!v1_0_0.is_compatible(&v1_1_0));
        assert!(!v2_0_0.is_compatible(&v1_0_0));
    }
}
