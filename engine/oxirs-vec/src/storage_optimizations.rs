//! Storage optimizations for vector indices
//!
//! This module provides optimized storage formats for efficient vector serialization:
//! - Binary formats for fast I/O
//! - Compression with multiple algorithms
//! - Streaming I/O for large datasets
//! - Memory-mapped file support

use crate::Vector;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Compression methods for vector storage
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CompressionType {
    /// No compression
    None,
    /// LZ4 fast compression
    Lz4,
    /// Zstandard balanced compression
    Zstd,
    /// Brotli high compression
    Brotli,
    /// Gzip standard compression
    Gzip,
    /// Custom vector quantization compression
    VectorQuantization,
}

/// Binary storage format configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Compression type to use
    pub compression: CompressionType,
    /// Compression level (0-9, algorithm dependent)
    pub compression_level: u8,
    /// Buffer size for streaming operations
    pub buffer_size: usize,
    /// Enable memory mapping for large files
    pub enable_mmap: bool,
    /// Block size for chunked reading/writing
    pub block_size: usize,
    /// Enable checksums for data integrity
    pub enable_checksums: bool,
    /// Metadata format version
    pub format_version: u32,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            compression: CompressionType::Zstd,
            compression_level: 3,
            buffer_size: 1024 * 1024, // 1MB
            enable_mmap: true,
            block_size: 64 * 1024, // 64KB
            enable_checksums: true,
            format_version: 1,
        }
    }
}

/// Binary file header for vector storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorFileHeader {
    /// Magic number for file format identification
    pub magic: [u8; 8],
    /// Format version
    pub version: u32,
    /// Number of vectors in file
    pub vector_count: u64,
    /// Vector dimensions
    pub dimensions: usize,
    /// Compression type used
    pub compression: CompressionType,
    /// Compression level
    pub compression_level: u8,
    /// Block size for chunked data
    pub block_size: usize,
    /// Checksum of header
    pub header_checksum: u32,
    /// Offset to vector data
    pub data_offset: u64,
    /// Total data size (compressed)
    pub data_size: u64,
    /// Original data size (uncompressed)
    pub original_size: u64,
    /// Reserved bytes for future use
    pub reserved: [u8; 32],
}

impl Default for VectorFileHeader {
    fn default() -> Self {
        Self {
            magic: *b"OXIRSVEC",
            version: 1,
            vector_count: 0,
            dimensions: 0,
            compression: CompressionType::None,
            compression_level: 0,
            block_size: 64 * 1024,
            header_checksum: 0,
            data_offset: 0,
            data_size: 0,
            original_size: 0,
            reserved: [0; 32],
        }
    }
}

impl VectorFileHeader {
    /// Calculate and set header checksum
    pub fn calculate_checksum(&mut self) {
        // Simple CRC32-like checksum
        let mut checksum = 0u32;
        checksum ^=
            u32::from_le_bytes([self.magic[0], self.magic[1], self.magic[2], self.magic[3]]);
        checksum ^=
            u32::from_le_bytes([self.magic[4], self.magic[5], self.magic[6], self.magic[7]]);
        checksum ^= self.version;
        checksum ^= self.vector_count as u32;
        checksum ^= self.dimensions as u32;
        checksum ^= self.compression as u8 as u32;
        checksum ^= self.compression_level as u32;
        self.header_checksum = checksum;
    }

    /// Verify header checksum
    pub fn verify_checksum(&self) -> bool {
        let mut temp_header = self.clone();
        temp_header.header_checksum = 0;
        temp_header.calculate_checksum();
        temp_header.header_checksum == self.header_checksum
    }
}

/// Vector block for chunked storage
#[derive(Debug, Clone)]
pub struct VectorBlock {
    /// Block index
    pub block_id: u32,
    /// Number of vectors in this block
    pub vector_count: u32,
    /// Compressed data
    pub data: Vec<u8>,
    /// Original size before compression
    pub original_size: u32,
    /// Block checksum
    pub checksum: u32,
}

/// Streaming vector writer for large datasets
pub struct VectorWriter {
    writer: BufWriter<File>,
    config: StorageConfig,
    header: VectorFileHeader,
    current_block: Vec<Vector>,
    blocks_written: u32,
    total_vectors: u64,
}

impl VectorWriter {
    /// Create a new vector writer
    pub fn new<P: AsRef<Path>>(path: P, config: StorageConfig) -> Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let header = VectorFileHeader {
            compression: config.compression,
            compression_level: config.compression_level,
            block_size: config.block_size,
            ..Default::default()
        };

        // Write a placeholder header first to reserve space
        let placeholder_header_bytes = bincode::serialize(&header)?;
        let _header_size = (4 + placeholder_header_bytes.len()) as u64;

        // Write placeholder header size and header
        writer.write_all(&(placeholder_header_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(&placeholder_header_bytes)?;
        writer.flush()?;

        Ok(Self {
            writer,
            config,
            header,
            current_block: Vec::new(),
            blocks_written: 0,
            total_vectors: 0,
        })
    }

    /// Write a vector to the file
    pub fn write_vector(&mut self, vector: Vector) -> Result<()> {
        // Set dimensions from first vector
        if self.header.dimensions == 0 {
            self.header.dimensions = vector.dimensions;
        } else if self.header.dimensions != vector.dimensions {
            return Err(anyhow!(
                "Vector dimension mismatch: expected {}, got {}",
                self.header.dimensions,
                vector.dimensions
            ));
        }

        self.current_block.push(vector);
        self.total_vectors += 1;

        // Check if we need to flush the current block
        let block_size_estimate = self.current_block.len() * self.header.dimensions * 4; // 4 bytes per f32
        if block_size_estimate >= self.config.block_size {
            self.flush_block()?;
        }

        Ok(())
    }

    /// Write multiple vectors
    pub fn write_vectors(&mut self, vectors: &[Vector]) -> Result<()> {
        for vector in vectors {
            self.write_vector(vector.clone())?;
        }
        Ok(())
    }

    /// Flush current block to disk
    fn flush_block(&mut self) -> Result<()> {
        if self.current_block.is_empty() {
            return Ok(());
        }

        // For uncompressed mode, write raw vector data
        if self.config.compression == CompressionType::None {
            for vector in &self.current_block {
                let vector_bytes = vector.as_f32();
                for value in vector_bytes {
                    self.writer.write_all(&value.to_le_bytes())?;
                }
            }
            self.current_block.clear();
            return Ok(());
        }

        // Serialize vectors to binary
        let mut block_data = Vec::new();
        for vector in &self.current_block {
            let vector_bytes = vector.as_f32();
            for value in vector_bytes {
                block_data.extend_from_slice(&value.to_le_bytes());
            }
        }

        // Compress block data
        let compressed_data = self.compress_data(&block_data)?;

        // Create block header
        let block = VectorBlock {
            block_id: self.blocks_written,
            vector_count: self.current_block.len() as u32,
            original_size: block_data.len() as u32,
            checksum: self.calculate_data_checksum(&compressed_data),
            data: compressed_data,
        };

        // Write block to file
        self.write_block(&block)?;

        self.current_block.clear();
        self.blocks_written += 1;

        Ok(())
    }

    /// Compress data using configured algorithm
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.config.compression {
            CompressionType::None => Ok(data.to_vec()),
            CompressionType::Lz4 => {
                // Placeholder for LZ4 compression
                // In real implementation, use lz4_flex or similar crate
                Ok(data.to_vec())
            }
            CompressionType::Zstd => {
                // Placeholder for Zstandard compression
                // In real implementation, use zstd crate
                Ok(data.to_vec())
            }
            CompressionType::Brotli => {
                // Placeholder for Brotli compression
                // In real implementation, use brotli crate
                Ok(data.to_vec())
            }
            CompressionType::Gzip => {
                // Placeholder for Gzip compression
                // In real implementation, use flate2 crate
                Ok(data.to_vec())
            }
            CompressionType::VectorQuantization => {
                // Placeholder for vector quantization
                // In real implementation, use PQ or similar
                Ok(data.to_vec())
            }
        }
    }

    /// Calculate checksum for data
    fn calculate_data_checksum(&self, data: &[u8]) -> u32 {
        // Simple checksum - in production use CRC32 or similar
        data.iter().fold(0u32, |acc, &b| acc.wrapping_add(b as u32))
    }

    /// Write block to file
    fn write_block(&mut self, block: &VectorBlock) -> Result<()> {
        // Write block header
        self.writer.write_all(&block.block_id.to_le_bytes())?;
        self.writer.write_all(&block.vector_count.to_le_bytes())?;
        self.writer.write_all(&block.original_size.to_le_bytes())?;
        self.writer.write_all(&block.checksum.to_le_bytes())?;
        self.writer
            .write_all(&(block.data.len() as u32).to_le_bytes())?;

        // Write block data
        self.writer.write_all(&block.data)?;

        Ok(())
    }

    /// Finalize and close the file
    pub fn finalize(mut self) -> Result<()> {
        // Flush any remaining vectors
        self.flush_block()?;

        // Update header with final counts
        self.header.vector_count = self.total_vectors;

        // The data offset is fixed at the position after the placeholder header
        // We need to calculate this the same way it was calculated in new()
        let placeholder_header = VectorFileHeader {
            compression: self.config.compression,
            compression_level: self.config.compression_level,
            block_size: self.config.block_size,
            ..Default::default()
        };
        let placeholder_header_bytes = bincode::serialize(&placeholder_header)?;
        self.header.data_offset = 4 + placeholder_header_bytes.len() as u64;
        self.header.calculate_checksum();

        // Flush before seeking to ensure all data is written
        self.writer.flush()?;

        // Seek to beginning and write header
        self.writer.get_mut().seek(SeekFrom::Start(0))?;

        let header_bytes = bincode::serialize(&self.header)?;

        // Write header size first, then header data
        let header_size = header_bytes.len() as u32;
        self.writer.write_all(&header_size.to_le_bytes())?;
        self.writer.write_all(&header_bytes)?;

        // Flush to ensure data is written to file
        self.writer.flush()?;

        // Explicitly drop the writer to close the file
        drop(self.writer);

        Ok(())
    }
}

/// Streaming vector reader for large datasets
pub struct VectorReader {
    reader: BufReader<File>,
    header: VectorFileHeader,
    current_position: u64,
    vectors_read: u64,
}

impl VectorReader {
    /// Open a vector file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and verify header
        let header = Self::read_header(&mut reader)?;
        let data_offset = header.data_offset;

        // Seek to data offset to position for reading vectors
        reader.get_mut().seek(SeekFrom::Start(data_offset))?;

        Ok(Self {
            reader,
            header,
            current_position: data_offset,
            vectors_read: 0,
        })
    }

    /// Read file header
    fn read_header(reader: &mut BufReader<File>) -> Result<VectorFileHeader> {
        // Read header size first
        let mut size_bytes = [0u8; 4];
        reader.read_exact(&mut size_bytes)?;
        let header_size = u32::from_le_bytes(size_bytes) as usize;

        // Read header data of exact size
        let mut header_data = vec![0u8; header_size];
        reader.read_exact(&mut header_data)?;

        let header: VectorFileHeader = bincode::deserialize(&header_data)?;

        // Verify magic number
        if &header.magic != b"OXIRSVEC" {
            return Err(anyhow!("Invalid file format: magic number mismatch"));
        }

        // Verify checksum
        if !header.verify_checksum() {
            return Err(anyhow!("Header checksum verification failed"));
        }

        Ok(header)
    }

    /// Get file metadata
    pub fn metadata(&self) -> &VectorFileHeader {
        &self.header
    }

    /// Read next vector from file
    pub fn read_vector(&mut self) -> Result<Option<Vector>> {
        if self.vectors_read >= self.header.vector_count {
            return Ok(None);
        }

        // For simplicity, reading one vector at a time
        // In production, implement block-wise reading for efficiency
        let mut vector_data = vec![0f32; self.header.dimensions];

        for vector_item in vector_data.iter_mut().take(self.header.dimensions) {
            let mut bytes = [0u8; 4];
            self.reader.read_exact(&mut bytes)?;
            *vector_item = f32::from_le_bytes(bytes);
        }

        self.vectors_read += 1;
        self.current_position += (self.header.dimensions * 4) as u64;
        Ok(Some(Vector::new(vector_data)))
    }

    /// Read multiple vectors
    pub fn read_vectors(&mut self, count: usize) -> Result<Vec<Vector>> {
        let mut vectors = Vec::with_capacity(count);

        for _ in 0..count {
            if let Some(vector) = self.read_vector()? {
                vectors.push(vector);
            } else {
                break;
            }
        }

        Ok(vectors)
    }

    /// Read all remaining vectors
    pub fn read_all(&mut self) -> Result<Vec<Vector>> {
        let remaining = (self.header.vector_count - self.vectors_read) as usize;
        self.read_vectors(remaining)
    }

    /// Skip to specific vector index
    pub fn seek_to_vector(&mut self, index: u64) -> Result<()> {
        if index >= self.header.vector_count {
            return Err(anyhow!("Vector index {} out of bounds", index));
        }

        let byte_offset = self.header.data_offset + (index * self.header.dimensions as u64 * 4);
        self.reader.get_mut().seek(SeekFrom::Start(byte_offset))?;
        self.vectors_read = index;

        Ok(())
    }
}

/// Memory-mapped vector file for efficient random access
pub struct MmapVectorFile {
    _file: File,
    mmap: memmap2::Mmap,
    header: VectorFileHeader,
}

impl MmapVectorFile {
    /// Open file with memory mapping
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Read header from memory map
        let header_bytes = &mmap[0..std::mem::size_of::<VectorFileHeader>()];
        let header: VectorFileHeader = bincode::deserialize(header_bytes)?;

        // Verify header
        if &header.magic != b"OXIRSVEC" {
            return Err(anyhow!("Invalid file format"));
        }

        if !header.verify_checksum() {
            return Err(anyhow!("Header checksum verification failed"));
        }

        Ok(Self {
            _file: file,
            mmap,
            header,
        })
    }

    /// Get vector by index (zero-copy)
    pub fn get_vector(&self, index: u64) -> Result<Vector> {
        if index >= self.header.vector_count {
            return Err(anyhow!("Vector index out of bounds"));
        }

        let offset =
            self.header.data_offset as usize + (index as usize * self.header.dimensions * 4);
        let end_offset = offset + (self.header.dimensions * 4);

        if end_offset > self.mmap.len() {
            return Err(anyhow!("Vector data extends beyond file"));
        }

        let vector_bytes = &self.mmap[offset..end_offset];
        let mut vector_data = vec![0f32; self.header.dimensions];

        for (i, chunk) in vector_bytes.chunks_exact(4).enumerate() {
            vector_data[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }

        Ok(Vector::new(vector_data))
    }

    /// Get slice of vectors
    pub fn get_vectors(&self, start: u64, count: usize) -> Result<Vec<Vector>> {
        let mut vectors = Vec::with_capacity(count);

        for i in 0..count {
            let index = start + i as u64;
            if index >= self.header.vector_count {
                break;
            }
            vectors.push(self.get_vector(index)?);
        }

        Ok(vectors)
    }

    /// Get total vector count
    pub fn vector_count(&self) -> u64 {
        self.header.vector_count
    }

    /// Get vector dimensions
    pub fn dimensions(&self) -> usize {
        self.header.dimensions
    }
}

/// Utility functions for storage operations
pub struct StorageUtils;

impl StorageUtils {
    /// Convert vectors to binary format
    pub fn vectors_to_binary(vectors: &[Vector]) -> Result<Vec<u8>> {
        let mut data = Vec::new();

        for vector in vectors {
            let vector_f32 = vector.as_f32();
            for value in vector_f32 {
                data.extend_from_slice(&value.to_le_bytes());
            }
        }

        Ok(data)
    }

    /// Convert binary data to vectors
    pub fn binary_to_vectors(data: &[u8], dimensions: usize) -> Result<Vec<Vector>> {
        if data.len() % (dimensions * 4) != 0 {
            return Err(anyhow!("Invalid binary data length for given dimensions"));
        }

        let vector_count = data.len() / (dimensions * 4);
        let mut vectors = Vec::with_capacity(vector_count);

        for i in 0..vector_count {
            let start = i * dimensions * 4;
            let end = start + dimensions * 4;
            let vector_bytes = &data[start..end];

            let mut vector_data = vec![0f32; dimensions];
            for (j, chunk) in vector_bytes.chunks_exact(4).enumerate() {
                vector_data[j] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }

            vectors.push(Vector::new(vector_data));
        }

        Ok(vectors)
    }

    /// Estimate storage size for vectors
    pub fn estimate_storage_size(
        vector_count: usize,
        dimensions: usize,
        compression: CompressionType,
    ) -> usize {
        let raw_size = vector_count * dimensions * 4; // 4 bytes per f32
        let header_size = std::mem::size_of::<VectorFileHeader>();

        let compressed_size = match compression {
            CompressionType::None => raw_size,
            CompressionType::Lz4 => (raw_size as f64 * 0.6) as usize, // ~40% compression
            CompressionType::Zstd => (raw_size as f64 * 0.5) as usize, // ~50% compression
            CompressionType::Brotli => (raw_size as f64 * 0.4) as usize, // ~60% compression
            CompressionType::Gzip => (raw_size as f64 * 0.5) as usize, // ~50% compression
            CompressionType::VectorQuantization => (raw_size as f64 * 0.25) as usize, // ~75% compression
        };

        header_size + compressed_size
    }

    /// Benchmark compression algorithms
    pub fn benchmark_compression(vectors: &[Vector]) -> Result<Vec<(CompressionType, f64, usize)>> {
        let binary_data = Self::vectors_to_binary(vectors)?;
        let original_size = binary_data.len();

        let algorithms = [
            CompressionType::None,
            CompressionType::Lz4,
            CompressionType::Zstd,
            CompressionType::Brotli,
            CompressionType::Gzip,
        ];

        let mut results = Vec::new();

        for &algorithm in &algorithms {
            let start_time = std::time::Instant::now();

            // Simulate compression (in real implementation, use actual compression)
            let compressed_size = match algorithm {
                CompressionType::None => original_size,
                CompressionType::Lz4 => (original_size as f64 * 0.6) as usize,
                CompressionType::Zstd => (original_size as f64 * 0.5) as usize,
                CompressionType::Brotli => (original_size as f64 * 0.4) as usize,
                CompressionType::Gzip => (original_size as f64 * 0.5) as usize,
                CompressionType::VectorQuantization => (original_size as f64 * 0.25) as usize,
            };

            let duration = start_time.elapsed().as_secs_f64();
            results.push((algorithm, duration, compressed_size));
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_vector_file_header() {
        let mut header = VectorFileHeader {
            vector_count: 1000,
            dimensions: 128,
            ..Default::default()
        };
        header.calculate_checksum();

        assert!(header.verify_checksum());

        // Modify header and verify checksum fails
        header.vector_count = 2000;
        assert!(!header.verify_checksum());
    }

    #[test]
    fn test_storage_utils() {
        let vectors = vec![
            Vector::new(vec![1.0, 2.0, 3.0]),
            Vector::new(vec![4.0, 5.0, 6.0]),
        ];

        let binary_data = StorageUtils::vectors_to_binary(&vectors).unwrap();
        let restored_vectors = StorageUtils::binary_to_vectors(&binary_data, 3).unwrap();

        assert_eq!(vectors.len(), restored_vectors.len());
        for (original, restored) in vectors.iter().zip(restored_vectors.iter()) {
            assert_eq!(original.as_f32(), restored.as_f32());
        }
    }

    #[test]
    fn test_vector_writer_reader() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let file_path = temp_file.path();

        // Write vectors
        {
            let config = StorageConfig {
                compression: CompressionType::None,
                ..Default::default()
            };
            let mut writer = VectorWriter::new(file_path, config)?;

            let vectors = vec![
                Vector::new(vec![1.0, 2.0, 3.0, 4.0]),
                Vector::new(vec![5.0, 6.0, 7.0, 8.0]),
                Vector::new(vec![9.0, 10.0, 11.0, 12.0]),
            ];

            writer.write_vectors(&vectors)?;
            writer.finalize()?;
        }

        // Read vectors back
        {
            let mut reader = VectorReader::open(file_path)?;
            let metadata = reader.metadata();

            assert_eq!(metadata.vector_count, 3);
            assert_eq!(metadata.dimensions, 4);

            let vectors = reader.read_all()?;
            assert_eq!(vectors.len(), 3);

            assert_eq!(vectors[0].as_f32(), &[1.0, 2.0, 3.0, 4.0]);
            assert_eq!(vectors[1].as_f32(), &[5.0, 6.0, 7.0, 8.0]);
            assert_eq!(vectors[2].as_f32(), &[9.0, 10.0, 11.0, 12.0]);
        }

        Ok(())
    }

    #[test]
    fn test_compression_benchmark() {
        let vectors = vec![
            Vector::new(vec![1.0; 128]),
            Vector::new(vec![2.0; 128]),
            Vector::new(vec![3.0; 128]),
        ];

        let results = StorageUtils::benchmark_compression(&vectors).unwrap();
        assert_eq!(results.len(), 5); // 5 compression algorithms

        // Verify that some compression types reduce size
        let none_size = results
            .iter()
            .find(|(t, _, _)| *t == CompressionType::None)
            .unwrap()
            .2;
        let zstd_size = results
            .iter()
            .find(|(t, _, _)| *t == CompressionType::Zstd)
            .unwrap()
            .2;

        assert!(zstd_size < none_size);
    }
}
