//! FAISS Compatibility Layer
//!
//! This module provides compatibility with Facebook AI Similarity Search (FAISS)
//! library, enabling import/export of vector indexes to/from FAISS format.
//! This allows seamless integration with the broader ML ecosystem.

use crate::{
    hnsw::{HnswConfig, HnswIndex},
    ivf::{IvfConfig, IvfIndex},
    similarity::SimilarityMetric,
    Vector, VectorIndex,
};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// FAISS index types that we support
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaissIndexType {
    /// Flat (brute force) index
    IndexFlatL2,
    IndexFlatIP,
    /// IVF with flat quantizer
    IndexIVFFlat,
    /// IVF with product quantization
    IndexIVFPQ,
    /// Hierarchical NSW (HNSW)
    IndexHNSWFlat,
    /// LSH (Locality Sensitive Hashing)
    IndexLSH,
    /// PCA + flat index
    IndexPCAFlat,
}

/// FAISS index metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaissIndexMetadata {
    pub index_type: FaissIndexType,
    pub dimension: usize,
    pub num_vectors: usize,
    pub metric_type: FaissMetricType,
    pub parameters: HashMap<String, FaissParameter>,
    pub version: String,
    pub created_at: String,
}

/// FAISS metric types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaissMetricType {
    /// L2 (Euclidean) distance
    L2,
    /// Inner product (dot product)
    InnerProduct,
    /// Cosine similarity (normalized inner product)
    Cosine,
}

/// FAISS parameter values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FaissParameter {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
}

/// FAISS compatibility layer for vector indexes
pub struct FaissCompatibility {
    supported_formats: Vec<FaissIndexType>,
    conversion_cache: HashMap<String, ConversionResult>,
}

/// Result of index conversion
#[derive(Debug, Clone)]
pub struct ConversionResult {
    pub success: bool,
    pub metadata: FaissIndexMetadata,
    pub performance_metrics: ConversionMetrics,
    pub warnings: Vec<String>,
}

/// Conversion performance metrics
#[derive(Debug, Clone, Default)]
pub struct ConversionMetrics {
    pub conversion_time: std::time::Duration,
    pub memory_used: usize,
    pub vectors_processed: usize,
    pub accuracy_preserved: f32, // 0.0 to 1.0
}

/// FAISS export configuration
#[derive(Debug, Clone)]
pub struct FaissExportConfig {
    pub target_format: FaissIndexType,
    pub compression_level: CompressionLevel,
    pub preserve_accuracy: bool,
    pub include_metadata: bool,
    pub chunk_size: usize,
}

/// FAISS import configuration
#[derive(Debug, Clone)]
pub struct FaissImportConfig {
    pub validate_format: bool,
    pub preserve_performance: bool,
    pub rebuild_if_incompatible: bool,
    pub batch_size: usize,
}

/// Compression levels for export
#[derive(Debug, Clone, Copy)]
pub enum CompressionLevel {
    None,
    Low,
    Medium,
    High,
    Maximum,
}

impl Default for FaissExportConfig {
    fn default() -> Self {
        Self {
            target_format: FaissIndexType::IndexHNSWFlat,
            compression_level: CompressionLevel::Medium,
            preserve_accuracy: true,
            include_metadata: true,
            chunk_size: 10000,
        }
    }
}

impl Default for FaissImportConfig {
    fn default() -> Self {
        Self {
            validate_format: true,
            preserve_performance: true,
            rebuild_if_incompatible: false,
            batch_size: 5000,
        }
    }
}

impl FaissCompatibility {
    /// Create a new FAISS compatibility layer
    pub fn new() -> Self {
        Self {
            supported_formats: vec![
                FaissIndexType::IndexFlatL2,
                FaissIndexType::IndexFlatIP,
                FaissIndexType::IndexIVFFlat,
                FaissIndexType::IndexIVFPQ,
                FaissIndexType::IndexHNSWFlat,
                FaissIndexType::IndexLSH,
            ],
            conversion_cache: HashMap::new(),
        }
    }

    /// Export an oxirs-vec index to FAISS format
    pub fn export_to_faiss<T: VectorIndex + 'static>(
        &mut self,
        index: &T,
        output_path: &Path,
        config: &FaissExportConfig,
    ) -> Result<ConversionResult> {
        let start_time = std::time::Instant::now();
        let mut warnings = Vec::new();

        // Detect the appropriate FAISS format for the index
        let detected_format = self.detect_optimal_faiss_format(index)?;
        let target_format = if detected_format != config.target_format {
            warnings.push(format!(
                "Requested format {:?} differs from optimal format {:?}",
                config.target_format, detected_format
            ));
            config.target_format
        } else {
            detected_format
        };

        // Create metadata
        let metadata = FaissIndexMetadata {
            index_type: target_format,
            dimension: self.get_index_dimension(index)?,
            num_vectors: self.get_index_size(index)?,
            metric_type: self.convert_similarity_metric(self.get_index_metric(index)?),
            parameters: self.extract_index_parameters(index, target_format)?,
            version: "oxirs-vec-1.0".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
        };

        // Export the index data
        self.write_faiss_index(index, output_path, &metadata, config)?;

        let conversion_time = start_time.elapsed();
        let performance_metrics = ConversionMetrics {
            conversion_time,
            memory_used: self.estimate_memory_usage(&metadata),
            vectors_processed: metadata.num_vectors,
            accuracy_preserved: self.estimate_accuracy_preservation(target_format, config),
        };

        let result = ConversionResult {
            success: true,
            metadata,
            performance_metrics,
            warnings,
        };

        // Cache the result
        let cache_key = format!("{:?}-{}", target_format, output_path.display());
        self.conversion_cache.insert(cache_key, result.clone());

        Ok(result)
    }

    /// Import a FAISS index to oxirs-vec format
    pub fn import_from_faiss(
        &mut self,
        input_path: &Path,
        config: &FaissImportConfig,
    ) -> Result<Box<dyn VectorIndex>> {
        // Read and validate FAISS metadata
        let metadata = self.read_faiss_metadata(input_path)?;

        if config.validate_format && !self.is_format_supported(&metadata.index_type) {
            return Err(anyhow!(
                "Unsupported FAISS format: {:?}",
                metadata.index_type
            ));
        }

        // Create appropriate oxirs-vec index based on FAISS type
        match metadata.index_type {
            FaissIndexType::IndexHNSWFlat => self.import_hnsw_index(input_path, &metadata, config),
            FaissIndexType::IndexIVFFlat | FaissIndexType::IndexIVFPQ => {
                self.import_ivf_index(input_path, &metadata, config)
            }
            FaissIndexType::IndexFlatL2 | FaissIndexType::IndexFlatIP => {
                self.import_flat_index(input_path, &metadata, config)
            }
            _ => Err(anyhow!(
                "Import not yet implemented for {:?}",
                metadata.index_type
            )),
        }
    }

    /// Convert HNSW index to FAISS format
    fn export_hnsw_to_faiss(
        &self,
        index: &HnswIndex,
        output_path: &Path,
        metadata: &FaissIndexMetadata,
        config: &FaissExportConfig,
    ) -> Result<()> {
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        // Write FAISS header
        self.write_faiss_header(&mut writer, metadata)?;

        // Write HNSW-specific data
        self.write_hnsw_data(&mut writer, index, config)?;

        // Write vectors in chunks
        self.write_vectors_chunked(&mut writer, index, config.chunk_size)?;

        writer.flush()?;
        Ok(())
    }

    /// Convert IVF index to FAISS format
    fn export_ivf_to_faiss(
        &self,
        index: &IvfIndex,
        output_path: &Path,
        metadata: &FaissIndexMetadata,
        config: &FaissExportConfig,
    ) -> Result<()> {
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        // Write FAISS header
        self.write_faiss_header(&mut writer, metadata)?;

        // Write IVF-specific data
        self.write_ivf_data(&mut writer, index, config)?;

        // Write centroids and inverted lists
        self.write_ivf_structure(&mut writer, index)?;

        writer.flush()?;
        Ok(())
    }

    /// Import HNSW index from FAISS format
    fn import_hnsw_index(
        &self,
        input_path: &Path,
        metadata: &FaissIndexMetadata,
        config: &FaissImportConfig,
    ) -> Result<Box<dyn VectorIndex>> {
        let file = File::open(input_path)?;
        let mut reader = BufReader::new(file);

        // Skip FAISS header
        self.skip_faiss_header(&mut reader)?;

        // Read HNSW configuration
        let hnsw_config = self.read_hnsw_config(&mut reader, metadata)?;

        // Create new HNSW index
        let mut index = HnswIndex::new(hnsw_config)?;

        // Read and import vectors in batches
        self.import_vectors_batched(&mut reader, &mut index, metadata, config.batch_size)?;

        Ok(Box::new(index))
    }

    /// Import IVF index from FAISS format
    fn import_ivf_index(
        &self,
        input_path: &Path,
        metadata: &FaissIndexMetadata,
        config: &FaissImportConfig,
    ) -> Result<Box<dyn VectorIndex>> {
        let file = File::open(input_path)?;
        let mut reader = BufReader::new(file);

        // Skip FAISS header
        self.skip_faiss_header(&mut reader)?;

        // Read IVF configuration
        let ivf_config = self.read_ivf_config(&mut reader, metadata)?;

        // Create new IVF index
        let mut index = IvfIndex::new(ivf_config)?;

        // Read centroids and structure
        self.read_ivf_structure(&mut reader, &mut index)?;

        // Import vectors. NOTE: `read_ivf_structure` (centroids) is still a
        // placeholder, so a freshly-constructed `IvfIndex` here is untrained;
        // `IvfIndex::insert` will return a clear "must be trained" error
        // rather than silently accepting vectors into an untrained index.
        self.import_vectors_batched(&mut reader, &mut index, metadata, config.batch_size)?;

        Ok(Box::new(index))
    }

    /// Import flat index from FAISS format
    fn import_flat_index(
        &self,
        input_path: &Path,
        metadata: &FaissIndexMetadata,
        _config: &FaissImportConfig,
    ) -> Result<Box<dyn VectorIndex>> {
        let file = File::open(input_path)?;
        let mut reader = BufReader::new(file);

        // Skip FAISS header
        self.skip_faiss_header(&mut reader)?;

        // Create a simple in-memory index for flat FAISS indexes
        let mut vectors = Vec::new();
        let mut uris = Vec::new();

        // Read all vectors
        for i in 0..metadata.num_vectors {
            let vector = self.read_vector(&mut reader, metadata.dimension)?;
            vectors.push(vector);
            uris.push(format!("faiss_vector_{i}"));
        }

        // Create a simple flat index implementation
        Ok(Box::new(SimpleVectorIndex::new(vectors, uris)))
    }

    /// Detect optimal FAISS format for an index
    fn detect_optimal_faiss_format<T: VectorIndex>(&self, index: &T) -> Result<FaissIndexType> {
        let size = self.get_index_size(index)?;
        let dimension = self.get_index_dimension(index)?;

        // Use heuristics to determine best format
        if size < 10000 {
            // Small datasets - use flat index
            Ok(FaissIndexType::IndexFlatL2)
        } else if dimension > 1000 {
            // High-dimensional - use IVF with PQ
            Ok(FaissIndexType::IndexIVFPQ)
        } else if size > 100000 {
            // Large dataset - use HNSW
            Ok(FaissIndexType::IndexHNSWFlat)
        } else {
            // Medium dataset - use IVF flat
            Ok(FaissIndexType::IndexIVFFlat)
        }
    }

    /// Check if a FAISS format is supported
    fn is_format_supported(&self, format: &FaissIndexType) -> bool {
        self.supported_formats.contains(format)
    }

    /// Convert similarity metric to FAISS metric type
    fn convert_similarity_metric(&self, metric: SimilarityMetric) -> FaissMetricType {
        match metric {
            SimilarityMetric::Cosine => FaissMetricType::Cosine,
            SimilarityMetric::Euclidean => FaissMetricType::L2,
            SimilarityMetric::DotProduct => FaissMetricType::InnerProduct,
            SimilarityMetric::Manhattan => FaissMetricType::L2, // Approximate with L2
            // All other metrics approximate with L2 for FAISS compatibility
            _ => FaissMetricType::L2,
        }
    }

    /// Write FAISS header to file
    fn write_faiss_header(
        &self,
        writer: &mut BufWriter<File>,
        metadata: &FaissIndexMetadata,
    ) -> Result<()> {
        // FAISS magic number
        writer.write_all(b"FAISS")?;

        // Version
        writer.write_all(&1u32.to_le_bytes())?;

        // Index type identifier
        let type_id = self.faiss_type_to_id(metadata.index_type);
        writer.write_all(&type_id.to_le_bytes())?;

        // Dimension
        writer.write_all(&(metadata.dimension as u32).to_le_bytes())?;

        // Number of vectors
        writer.write_all(&(metadata.num_vectors as u64).to_le_bytes())?;

        // Metric type
        let metric_id = self.faiss_metric_to_id(metadata.metric_type);
        writer.write_all(&metric_id.to_le_bytes())?;

        Ok(())
    }

    /// Skip FAISS header when reading
    fn skip_faiss_header(&self, reader: &mut BufReader<File>) -> Result<()> {
        let mut magic = [0u8; 5];
        reader.read_exact(&mut magic)?;

        if &magic != b"FAISS" {
            return Err(anyhow!("Invalid FAISS file format"));
        }

        // Skip version, type, dimension, count, metric
        let mut buffer = [0u8; 21]; // 4 + 4 + 4 + 8 + 1
        reader.read_exact(&mut buffer)?;

        Ok(())
    }

    /// Write HNSW-specific data
    fn write_hnsw_data(
        &self,
        writer: &mut BufWriter<File>,
        index: &HnswIndex,
        _config: &FaissExportConfig,
    ) -> Result<()> {
        // Write HNSW parameters
        let config = index.config();
        writer.write_all(&(config.m as u32).to_le_bytes())?;
        writer.write_all(&(config.m_l0 as u32).to_le_bytes())?;
        writer.write_all(&(config.ef as u32).to_le_bytes())?;
        writer.write_all(&config.ml.to_le_bytes())?;

        Ok(())
    }

    /// Write IVF-specific data
    fn write_ivf_data(
        &self,
        writer: &mut BufWriter<File>,
        index: &IvfIndex,
        _config: &FaissExportConfig,
    ) -> Result<()> {
        // Write IVF parameters
        let config = index.config();
        writer.write_all(&(config.n_clusters as u32).to_le_bytes())?;
        writer.write_all(&(config.n_probes as u32).to_le_bytes())?;

        Ok(())
    }

    /// Write vectors in chunks for memory efficiency.
    ///
    /// Vectors are enumerated once via [`VectorIndex::iter_vectors`] (real
    /// data, not a placeholder) and written out `chunk_size` at a time.
    fn write_vectors_chunked<T: VectorIndex>(
        &self,
        writer: &mut BufWriter<File>,
        index: &T,
        chunk_size: usize,
    ) -> Result<()> {
        let vectors = index.iter_vectors();
        let chunk_size = chunk_size.max(1);

        for chunk in vectors.chunks(chunk_size) {
            for (_uri, vector) in chunk {
                self.write_vector(writer, vector)?;
            }
        }

        Ok(())
    }

    /// Write a single vector to file
    fn write_vector(&self, writer: &mut BufWriter<File>, vector: &Vector) -> Result<()> {
        let data = vector.as_f32();
        for &value in &data {
            writer.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    }

    /// Read a single vector from file
    fn read_vector(&self, reader: &mut BufReader<File>, dimension: usize) -> Result<Vector> {
        let mut data = vec![0.0f32; dimension];
        for value in &mut data {
            let mut bytes = [0u8; 4];
            reader.read_exact(&mut bytes)?;
            *value = f32::from_le_bytes(bytes);
        }

        Ok(Vector::new(data))
    }

    /// Utility methods for index introspection.
    ///
    /// Derived from the real vector dimension of the first enumerable
    /// vector (via [`VectorIndex::iter_vectors`]); falls back to the common
    /// 768-dim transformer embedding size only when the index is empty (or
    /// not enumerable) and there is genuinely nothing to introspect.
    fn get_index_dimension<T: VectorIndex>(&self, index: &T) -> Result<usize> {
        Ok(index
            .iter_vectors()
            .first()
            .map(|(_, v)| v.dimensions)
            .unwrap_or(768))
    }

    /// Real vector count via [`VectorIndex::iter_vectors`] (0 for index
    /// types that don't support enumeration — see
    /// [`VectorIndex::supports_enumeration`]).
    fn get_index_size<T: VectorIndex>(&self, index: &T) -> Result<usize> {
        Ok(index.iter_vectors().len())
    }

    /// `VectorIndex` does not expose its configured similarity metric
    /// generically (it isn't part of the trait), so this is a conservative,
    /// clearly-documented default rather than a fabricated exact value.
    fn get_index_metric<T: VectorIndex>(&self, _index: &T) -> Result<SimilarityMetric> {
        Ok(SimilarityMetric::Cosine) // Documented conservative default
    }

    /// Helper methods for format conversion
    fn faiss_type_to_id(&self, faiss_type: FaissIndexType) -> u32 {
        match faiss_type {
            FaissIndexType::IndexFlatL2 => 0,
            FaissIndexType::IndexFlatIP => 1,
            FaissIndexType::IndexIVFFlat => 2,
            FaissIndexType::IndexIVFPQ => 3,
            FaissIndexType::IndexHNSWFlat => 4,
            FaissIndexType::IndexLSH => 5,
            FaissIndexType::IndexPCAFlat => 6,
        }
    }

    fn faiss_metric_to_id(&self, metric: FaissMetricType) -> u8 {
        match metric {
            FaissMetricType::L2 => 0,
            FaissMetricType::InnerProduct => 1,
            FaissMetricType::Cosine => 2,
        }
    }

    fn extract_index_parameters<T: VectorIndex>(
        &self,
        _index: &T,
        _format: FaissIndexType,
    ) -> Result<HashMap<String, FaissParameter>> {
        // Extract relevant parameters based on index type
        let mut params = HashMap::new();
        params.insert(
            "created_by".to_string(),
            FaissParameter::String("oxirs-vec".to_string()),
        );
        Ok(params)
    }

    fn estimate_memory_usage(&self, metadata: &FaissIndexMetadata) -> usize {
        // Rough estimate based on vectors and dimension
        metadata.num_vectors * metadata.dimension * 4 // 4 bytes per float
    }

    fn estimate_accuracy_preservation(
        &self,
        _format: FaissIndexType,
        _config: &FaissExportConfig,
    ) -> f32 {
        // Conservative estimate
        0.95 // 95% accuracy preservation
    }

    // Additional helper methods for reading configurations
    /// Read the HNSW-specific parameter block written by
    /// [`Self::write_hnsw_data`] (`m`, `m_l0`, `ef` as little-endian `u32`,
    /// then `ml` as little-endian `f64`; 20 bytes total).
    ///
    /// This *must* consume exactly the bytes `write_hnsw_data` wrote, even
    /// though only a subset of `HnswConfig`'s fields round-trip through the
    /// FAISS file: leaving them unread (the previous placeholder returned
    /// `HnswConfig::default()` without touching `reader` at all) shifts the
    /// reader's cursor 20 bytes short of where the vector payload actually
    /// starts, so every subsequent `read_vector` call reinterprets trailing
    /// HNSW-parameter/vector bytes at the wrong offset — silently producing
    /// garbage floats (e.g. subnormal/huge values from misaligned byte
    /// patterns) instead of the original vectors.
    fn read_hnsw_config(
        &self,
        reader: &mut BufReader<File>,
        _metadata: &FaissIndexMetadata,
    ) -> Result<HnswConfig> {
        let mut m_bytes = [0u8; 4];
        reader.read_exact(&mut m_bytes)?;
        let m = u32::from_le_bytes(m_bytes) as usize;

        let mut m_l0_bytes = [0u8; 4];
        reader.read_exact(&mut m_l0_bytes)?;
        let m_l0 = u32::from_le_bytes(m_l0_bytes) as usize;

        let mut ef_bytes = [0u8; 4];
        reader.read_exact(&mut ef_bytes)?;
        let ef = u32::from_le_bytes(ef_bytes) as usize;

        let mut ml_bytes = [0u8; 8];
        reader.read_exact(&mut ml_bytes)?;
        let ml = f64::from_le_bytes(ml_bytes);

        Ok(HnswConfig {
            m,
            m_l0,
            ef,
            ml,
            ..HnswConfig::default()
        })
    }

    /// Read the IVF-specific parameter block written by
    /// [`Self::write_ivf_data`] (`n_clusters`, `n_probes` as little-endian
    /// `u32`; 8 bytes total). See [`Self::read_hnsw_config`] for why every
    /// written byte must be consumed here regardless of how much of it
    /// [`IvfConfig`] actually retains.
    fn read_ivf_config(
        &self,
        reader: &mut BufReader<File>,
        _metadata: &FaissIndexMetadata,
    ) -> Result<IvfConfig> {
        let mut n_clusters_bytes = [0u8; 4];
        reader.read_exact(&mut n_clusters_bytes)?;
        let n_clusters = u32::from_le_bytes(n_clusters_bytes) as usize;

        let mut n_probes_bytes = [0u8; 4];
        reader.read_exact(&mut n_probes_bytes)?;
        let n_probes = u32::from_le_bytes(n_probes_bytes) as usize;

        Ok(IvfConfig {
            n_clusters,
            n_probes,
            ..IvfConfig::default()
        })
    }

    fn write_ivf_structure(&self, _writer: &mut BufWriter<File>, _index: &IvfIndex) -> Result<()> {
        // Write IVF centroids and inverted lists
        Ok(()) // Placeholder
    }

    fn read_ivf_structure(
        &self,
        _reader: &mut BufReader<File>,
        _index: &mut IvfIndex,
    ) -> Result<()> {
        // Read IVF centroids and structure
        Ok(()) // Placeholder
    }

    /// Import `metadata.num_vectors` vectors of `metadata.dimension` from
    /// `reader` in batches of `batch_size`, inserting each into `index`
    /// under a deterministic `faiss_vector_<i>` ID (matching
    /// [`Self::import_flat_index`]'s naming scheme, since FAISS files carry
    /// no per-vector ID/label of their own).
    fn import_vectors_batched<T: VectorIndex>(
        &self,
        reader: &mut BufReader<File>,
        index: &mut T,
        metadata: &FaissIndexMetadata,
        batch_size: usize,
    ) -> Result<()> {
        let batch_size = batch_size.max(1);
        let mut imported = 0usize;

        while imported < metadata.num_vectors {
            let batch_end = (imported + batch_size).min(metadata.num_vectors);
            for i in imported..batch_end {
                let vector = self.read_vector(reader, metadata.dimension)?;
                index.insert(format!("faiss_vector_{i}"), vector)?;
            }
            imported = batch_end;
        }

        Ok(())
    }

    /// Read and parse the real FAISS header written by
    /// [`Self::write_faiss_header`] (magic, version, type id, dimension,
    /// vector count, metric id) — no longer a fabricated placeholder.
    fn read_faiss_metadata(&self, input_path: &Path) -> Result<FaissIndexMetadata> {
        let file = File::open(input_path)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 5];
        reader.read_exact(&mut magic)?;
        if &magic != b"FAISS" {
            return Err(anyhow!("Invalid FAISS file format: bad magic bytes"));
        }

        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let _version = u32::from_le_bytes(version_bytes);

        let mut type_bytes = [0u8; 4];
        reader.read_exact(&mut type_bytes)?;
        let index_type = self.faiss_id_to_type(u32::from_le_bytes(type_bytes))?;

        let mut dim_bytes = [0u8; 4];
        reader.read_exact(&mut dim_bytes)?;
        let dimension = u32::from_le_bytes(dim_bytes) as usize;

        let mut count_bytes = [0u8; 8];
        reader.read_exact(&mut count_bytes)?;
        let num_vectors = u64::from_le_bytes(count_bytes) as usize;

        let mut metric_byte = [0u8; 1];
        reader.read_exact(&mut metric_byte)?;
        let metric_type = self.faiss_id_to_metric(metric_byte[0])?;

        Ok(FaissIndexMetadata {
            index_type,
            dimension,
            num_vectors,
            metric_type,
            parameters: HashMap::new(),
            version: "1.0".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
        })
    }

    fn write_faiss_index<T: VectorIndex + 'static>(
        &self,
        index: &T,
        output_path: &Path,
        metadata: &FaissIndexMetadata,
        config: &FaissExportConfig,
    ) -> Result<()> {
        match metadata.index_type {
            FaissIndexType::IndexHNSWFlat => {
                // Real downcast via `Any` (VectorIndex: 'static is required
                // for this, see the trait bound above).
                if let Some(hnsw_index) = self.try_cast_to_hnsw(index) {
                    self.export_hnsw_to_faiss(hnsw_index, output_path, metadata, config)
                } else {
                    Err(anyhow!("Index is not an HNSW index"))
                }
            }
            FaissIndexType::IndexIVFFlat | FaissIndexType::IndexIVFPQ => {
                if let Some(ivf_index) = self.try_cast_to_ivf(index) {
                    self.export_ivf_to_faiss(ivf_index, output_path, metadata, config)
                } else {
                    Err(anyhow!("Index is not an IVF index"))
                }
            }
            _ => Err(anyhow!(
                "Export format not yet implemented: {:?}",
                metadata.index_type
            )),
        }
    }

    // Real `Any`-based downcasting: FAISS export needs concrete access to
    // `HnswIndex`/`IvfIndex` internals (graph structure / centroids) that
    // aren't part of the generic `VectorIndex` trait, so we downcast the
    // generic `&T` back to the concrete type when it genuinely is one,
    // rather than always returning `None`.
    fn try_cast_to_hnsw<'a, T: VectorIndex + 'static>(
        &self,
        index: &'a T,
    ) -> Option<&'a HnswIndex> {
        (index as &dyn std::any::Any).downcast_ref::<HnswIndex>()
    }

    fn try_cast_to_ivf<'a, T: VectorIndex + 'static>(&self, index: &'a T) -> Option<&'a IvfIndex> {
        (index as &dyn std::any::Any).downcast_ref::<IvfIndex>()
    }

    /// Inverse of [`Self::faiss_type_to_id`].
    fn faiss_id_to_type(&self, id: u32) -> Result<FaissIndexType> {
        match id {
            0 => Ok(FaissIndexType::IndexFlatL2),
            1 => Ok(FaissIndexType::IndexFlatIP),
            2 => Ok(FaissIndexType::IndexIVFFlat),
            3 => Ok(FaissIndexType::IndexIVFPQ),
            4 => Ok(FaissIndexType::IndexHNSWFlat),
            5 => Ok(FaissIndexType::IndexLSH),
            6 => Ok(FaissIndexType::IndexPCAFlat),
            other => Err(anyhow!("Unknown FAISS index type id: {}", other)),
        }
    }

    /// Inverse of [`Self::faiss_metric_to_id`].
    fn faiss_id_to_metric(&self, id: u8) -> Result<FaissMetricType> {
        match id {
            0 => Ok(FaissMetricType::L2),
            1 => Ok(FaissMetricType::InnerProduct),
            2 => Ok(FaissMetricType::Cosine),
            other => Err(anyhow!("Unknown FAISS metric type id: {}", other)),
        }
    }
}

impl Default for FaissCompatibility {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple in-memory vector index for flat FAISS imports
pub struct SimpleVectorIndex {
    vectors: Vec<Vector>,
    uris: Vec<String>,
}

impl SimpleVectorIndex {
    pub fn new(vectors: Vec<Vector>, uris: Vec<String>) -> Self {
        Self { vectors, uris }
    }
}

impl VectorIndex for SimpleVectorIndex {
    fn insert(&mut self, uri: String, vector: Vector) -> Result<()> {
        self.uris.push(uri);
        self.vectors.push(vector);
        Ok(())
    }

    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        let mut results = Vec::new();

        for (i, vector) in self.vectors.iter().enumerate() {
            let similarity = self.compute_similarity(query, vector);
            results.push((self.uris[i].clone(), similarity));
        }

        // Sort by similarity (descending) and take top k
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        Ok(results)
    }

    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>> {
        let mut results = Vec::new();

        for (i, vector) in self.vectors.iter().enumerate() {
            let similarity = self.compute_similarity(query, vector);
            if similarity >= threshold {
                results.push((self.uris[i].clone(), similarity));
            }
        }

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    fn get_vector(&self, uri: &str) -> Option<&Vector> {
        self.uris
            .iter()
            .position(|u| u == uri)
            .map(|i| &self.vectors[i])
    }

    fn iter_vectors(&self) -> Vec<(String, Vector)> {
        self.uris
            .iter()
            .cloned()
            .zip(self.vectors.iter().cloned())
            .collect()
    }

    fn supports_enumeration(&self) -> bool {
        true
    }
}

impl SimpleVectorIndex {
    fn compute_similarity(&self, v1: &Vector, v2: &Vector) -> f32 {
        // Simple cosine similarity implementation
        let v1_data = v1.as_f32();
        let v2_data = v2.as_f32();

        if v1_data.len() != v2_data.len() {
            return 0.0;
        }

        let dot_product: f32 = v1_data.iter().zip(v2_data.iter()).map(|(a, b)| a * b).sum();
        let magnitude1: f32 = v1_data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude2: f32 = v2_data.iter().map(|x| x * x).sum::<f32>().sqrt();

        if magnitude1 == 0.0 || magnitude2 == 0.0 {
            0.0
        } else {
            dot_product / (magnitude1 * magnitude2)
        }
    }
}

/// Utility functions for FAISS compatibility
pub mod utils {
    use super::*;

    /// Convert oxirs-vec similarity metric to FAISS metric
    pub fn convert_metric(metric: SimilarityMetric) -> FaissMetricType {
        match metric {
            SimilarityMetric::Cosine => FaissMetricType::Cosine,
            SimilarityMetric::Euclidean => FaissMetricType::L2,
            SimilarityMetric::DotProduct => FaissMetricType::InnerProduct,
            SimilarityMetric::Manhattan => FaissMetricType::L2,
            // All other metrics approximate with L2 for FAISS compatibility
            _ => FaissMetricType::L2,
        }
    }

    /// Get recommended FAISS format for given constraints
    pub fn recommend_faiss_format(
        num_vectors: usize,
        dimension: usize,
        memory_constraint: Option<usize>,
        accuracy_requirement: f32,
    ) -> FaissIndexType {
        if num_vectors < 1000 || accuracy_requirement > 0.99 {
            FaissIndexType::IndexFlatL2
        } else if dimension > 1000 || memory_constraint.is_some_and(|mem| mem < 1024 * 1024 * 1024)
        {
            FaissIndexType::IndexIVFPQ
        } else if num_vectors > 100000 {
            FaissIndexType::IndexHNSWFlat
        } else {
            FaissIndexType::IndexIVFFlat
        }
    }

    /// Estimate memory requirements for FAISS format
    pub fn estimate_memory_requirement(
        format: FaissIndexType,
        num_vectors: usize,
        dimension: usize,
    ) -> usize {
        let base_memory = num_vectors * dimension * 4; // 4 bytes per float

        match format {
            FaissIndexType::IndexFlatL2 | FaissIndexType::IndexFlatIP => base_memory,
            FaissIndexType::IndexIVFFlat => base_memory + (num_vectors / 100) * dimension * 4, // Centroids
            FaissIndexType::IndexIVFPQ => base_memory / 8 + (num_vectors / 100) * dimension * 4, // PQ compression
            FaissIndexType::IndexHNSWFlat => base_memory * 2, // Graph structure overhead
            FaissIndexType::IndexLSH => base_memory / 2,      // Hash table compression
            FaissIndexType::IndexPCAFlat => base_memory, // Assuming no dimension reduction for estimate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_faiss_compatibility_creation() {
        let faiss_compat = FaissCompatibility::new();
        assert!(!faiss_compat.supported_formats.is_empty());
    }

    #[test]
    fn test_metric_conversion() {
        let faiss_compat = FaissCompatibility::new();

        assert_eq!(
            faiss_compat.convert_similarity_metric(SimilarityMetric::Cosine),
            FaissMetricType::Cosine
        );
        assert_eq!(
            faiss_compat.convert_similarity_metric(SimilarityMetric::Euclidean),
            FaissMetricType::L2
        );
        assert_eq!(
            faiss_compat.convert_similarity_metric(SimilarityMetric::DotProduct),
            FaissMetricType::InnerProduct
        );
    }

    #[test]
    fn test_simple_vector_index() -> Result<()> {
        let vectors = vec![
            Vector::new(vec![1.0, 0.0, 0.0]),
            Vector::new(vec![0.0, 1.0, 0.0]),
            Vector::new(vec![0.0, 0.0, 1.0]),
        ];
        let uris = vec!["v1".to_string(), "v2".to_string(), "v3".to_string()];

        let index = SimpleVectorIndex::new(vectors, uris);

        let query = Vector::new(vec![1.0, 0.0, 0.0]);
        let results = index.search_knn(&query, 2)?;

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "v1"); // Should be most similar to itself
        Ok(())
    }

    #[test]
    fn test_format_recommendation() {
        use crate::faiss_compatibility::utils::recommend_faiss_format;

        // Small dataset
        assert_eq!(
            recommend_faiss_format(100, 128, None, 0.9),
            FaissIndexType::IndexFlatL2
        );

        // Large dataset
        assert_eq!(
            recommend_faiss_format(1000000, 128, None, 0.8),
            FaissIndexType::IndexHNSWFlat
        );

        // High dimensional with memory constraint
        assert_eq!(
            recommend_faiss_format(50000, 2048, Some(512 * 1024 * 1024), 0.8),
            FaissIndexType::IndexIVFPQ
        );
    }

    /// Regression test for the P1 finding: `try_cast_to_hnsw`/`try_cast_to_ivf`
    /// used to always return `None`, so HNSW export always failed with
    /// "Index is not an HNSW index" even for real HNSW indexes. They must
    /// now perform a real `Any`-based downcast.
    #[test]
    fn test_try_cast_to_hnsw_succeeds_for_real_hnsw_index() -> Result<()> {
        use crate::hnsw::{HnswConfig, HnswIndex};

        let faiss_compat = FaissCompatibility::new();
        let index = HnswIndex::new(HnswConfig::default())?;

        assert!(
            faiss_compat.try_cast_to_hnsw(&index).is_some(),
            "downcasting a real HnswIndex must succeed, not return None"
        );

        // A different concrete type must not be misidentified as HNSW.
        let flat = SimpleVectorIndex::new(Vec::new(), Vec::new());
        assert!(faiss_compat.try_cast_to_ivf(&index).is_none());
        assert!(faiss_compat.try_cast_to_hnsw(&flat).is_none());
        Ok(())
    }

    /// Regression test for the P0/P1 findings: `read_faiss_metadata` used to
    /// always report `num_vectors: 0` regardless of file contents, and
    /// `import_vectors_batched` was an empty no-op. A full
    /// export -> import round trip through a real `HnswIndex` must now
    /// recover the original vectors.
    #[test]
    fn test_faiss_export_import_round_trip_hnsw() -> Result<()> {
        use crate::hnsw::{HnswConfig, HnswIndex};

        let mut faiss_compat = FaissCompatibility::new();
        let mut index = HnswIndex::new(HnswConfig::default())?;
        index.insert("doc_a".to_string(), Vector::new(vec![1.0, 0.0, 0.0, 0.0]))?;
        index.insert("doc_b".to_string(), Vector::new(vec![0.0, 1.0, 0.0, 0.0]))?;
        index.insert("doc_c".to_string(), Vector::new(vec![0.0, 0.0, 1.0, 0.0]))?;

        let dir = std::env::temp_dir().join(format!(
            "oxirs_vec_faiss_roundtrip_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir)?;
        let output_path = dir.join("index.faiss");

        let export_config = FaissExportConfig::default(); // target_format: IndexHNSWFlat
        let export_result = faiss_compat.export_to_faiss(&index, &output_path, &export_config)?;
        assert!(export_result.success);
        assert_eq!(export_result.metadata.num_vectors, 3);
        assert_eq!(export_result.metadata.dimension, 4);

        let import_config = FaissImportConfig::default();
        let imported = faiss_compat.import_from_faiss(&output_path, &import_config)?;

        // Real vectors must have round-tripped, not a fabricated empty index.
        // FAISS files carry no per-vector ID, and `HnswIndex::iter_vectors`
        // is not required to preserve insertion order (it walks a HashMap
        // internally), so compare the *set* of recovered vector values
        // rather than relying on ID or position.
        let recovered = imported.iter_vectors();
        assert_eq!(recovered.len(), 3);
        let mut recovered_values: Vec<Vec<f32>> =
            recovered.iter().map(|(_, v)| v.as_f32().to_vec()).collect();
        recovered_values.sort_by(|a, b| a.partial_cmp(b).expect("no NaNs in test vectors"));

        let mut expected_values = vec![
            vec![1.0f32, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        expected_values.sort_by(|a, b| a.partial_cmp(b).expect("no NaNs in test vectors"));

        assert_eq!(recovered_values, expected_values);

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }
}
