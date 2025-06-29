//! Storage and Serialization Optimization Module
//!
//! Provides binary encoding, compression, and performance optimizations
//! for distributed storage operations.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
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
            SerializationFormat::Bincode => bincode::serialize(data)?,
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
        let mut data = data;

        // Verify checksum if enabled
        if self.config.enable_checksums {
            if data.len() < 8 {
                return Err(anyhow::anyhow!("Data too short for checksum"));
            }

            let stored_checksum = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            let data_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

            if data.len() < 8 + data_len {
                return Err(anyhow::anyhow!("Data length mismatch"));
            }

            let actual_data = &data[8..8 + data_len];
            let computed_checksum = crc32fast::hash(actual_data);

            if stored_checksum != computed_checksum {
                return Err(anyhow::anyhow!("Checksum verification failed"));
            }

            data = actual_data;
        }

        // Decompress data
        let decompressed_data = match self.config.compression {
            CompressionAlgorithm::None => data.to_vec(),
            CompressionAlgorithm::Lz4 => self.decompress_lz4(data)?,
            CompressionAlgorithm::Zstd => self.decompress_zstd(data)?,
            CompressionAlgorithm::Deflate => self.decompress_deflate(data)?,
        };

        // Deserialize from binary format
        let result = match self.config.format {
            SerializationFormat::Json => serde_json::from_slice(&decompressed_data)?,
            SerializationFormat::MessagePack => rmp_serde::from_slice(&decompressed_data)?,
            SerializationFormat::Cbor => serde_cbor::from_slice(&decompressed_data)?,
            SerializationFormat::Bincode => bincode::deserialize(&decompressed_data)?,
        };

        Ok(result)
    }

    fn compress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(lz4_flex::compress_prepend_size(data))
    }

    fn decompress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(lz4_flex::decompress_size_prepended(data)?)
    }

    fn compress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(zstd::bulk::compress(data, self.config.compression_level)?)
    }

    fn decompress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(zstd::bulk::decompress(data, data.len() * 4)?)
    }

    fn compress_deflate(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;

        let mut encoder = DeflateEncoder::new(
            Vec::new(),
            Compression::new(self.config.compression_level as u32),
        );
        encoder.write_all(data)?;
        Ok(encoder.finish()?)
    }

    fn decompress_deflate(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::DeflateDecoder;

        let mut decoder = DeflateDecoder::new(data);
        let mut result = Vec::new();
        decoder.read_to_end(&mut result)?;
        Ok(result)
    }
}

/// Atomic file operations for crash safety
pub struct AtomicFileWriter {
    temp_path: std::path::PathBuf,
    final_path: std::path::PathBuf,
    file: Option<File>,
}

impl AtomicFileWriter {
    /// Create a new atomic file writer
    pub async fn new(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let final_path = path.as_ref().to_path_buf();
        let temp_path = final_path.with_extension("tmp");

        let file = OpenOptions::new()
            .write(true)
            .create(true)
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
    pub async fn write_all(&mut self, data: &[u8]) -> Result<()> {
        if let Some(ref mut file) = self.file {
            file.write_all(data).await?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("File already committed or aborted"))
        }
    }

    /// Commit the write by atomically moving temp file to final location
    pub async fn commit(mut self) -> Result<()> {
        if let Some(mut file) = self.file.take() {
            file.sync_all().await?;
            drop(file);
            tokio::fs::rename(&self.temp_path, &self.final_path).await?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("File already committed or aborted"))
        }
    }

    /// Abort the write by deleting the temporary file
    pub async fn abort(mut self) -> Result<()> {
        self.file = None;
        if self.temp_path.exists() {
            tokio::fs::remove_file(&self.temp_path).await?;
        }
        Ok(())
    }
}

impl Drop for AtomicFileWriter {
    fn drop(&mut self) {
        if self.file.is_some() && self.temp_path.exists() {
            let _ = std::fs::remove_file(&self.temp_path);
        }
    }
}

/// Corruption detection and recovery utilities
pub struct CorruptionDetector {
    enable_validation: bool,
}

impl CorruptionDetector {
    pub fn new(enable_validation: bool) -> Self {
        Self { enable_validation }
    }

    /// Validate file integrity using checksums
    pub async fn validate_file(&self, path: &std::path::Path) -> Result<bool> {
        if !self.enable_validation {
            return Ok(true);
        }

        let data = tokio::fs::read(path).await?;

        // Check if file has expected format with checksum
        if data.len() < 8 {
            return Ok(false);
        }

        let stored_checksum = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let data_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

        if data.len() < 8 + data_len {
            return Ok(false);
        }

        let actual_data = &data[8..8 + data_len];
        let computed_checksum = crc32fast::hash(actual_data);

        Ok(stored_checksum == computed_checksum)
    }

    /// Attempt to recover corrupted file from backup
    pub async fn recover_from_backup(&self, path: &std::path::Path) -> Result<bool> {
        let backup_path = path.with_extension("backup");

        if backup_path.exists() {
            if self.validate_file(&backup_path).await? {
                tokio::fs::copy(&backup_path, path).await?;
                tracing::info!("Recovered {} from backup", path.display());
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Create backup of file before modification
    pub async fn create_backup(&self, path: &std::path::Path) -> Result<()> {
        if path.exists() {
            let backup_path = path.with_extension("backup");
            tokio::fs::copy(path, &backup_path).await?;
            tracing::debug!("Created backup: {}", backup_path.display());
        }
        Ok(())
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

        // Corrupt the file
        let mut corrupted = serialized;
        corrupted[20] = !corrupted[20]; // Flip a bit
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
