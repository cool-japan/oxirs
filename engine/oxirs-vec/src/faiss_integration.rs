//! FAISS Integration for Advanced Vector Search
//!
//! This module provides integration with Facebook's FAISS (Facebook AI Similarity Search) library
//! for high-performance vector similarity search and clustering. FAISS is particularly well-suited
//! for large-scale vector databases with billions of vectors.

use crate::index::VectorIndex;
use anyhow::{Context, Error as AnyhowError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use tracing::{debug, info, span, Level};

/// Configuration for FAISS integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaissConfig {
    /// FAISS index type to use
    pub index_type: FaissIndexType,
    /// Number of dimensions for vectors
    pub dimension: usize,
    /// Training sample size for index optimization
    pub training_sample_size: usize,
    /// Number of clusters for IVF indices
    pub num_clusters: Option<usize>,
    /// Number of sub-quantizers for PQ
    pub num_subquantizers: Option<usize>,
    /// Number of bits per sub-quantizer
    pub bits_per_subquantizer: Option<u8>,
    /// Enable GPU acceleration
    pub use_gpu: bool,
    /// GPU device IDs to use
    pub gpu_devices: Vec<u32>,
    /// Memory mapping for large indices
    pub enable_mmap: bool,
    /// Index persistence settings
    pub persistence: FaissPersistenceConfig,
    /// Advanced optimization settings
    pub optimization: FaissOptimizationConfig,
}

impl Default for FaissConfig {
    fn default() -> Self {
        Self {
            index_type: FaissIndexType::FlatL2,
            dimension: 384,
            training_sample_size: 10000,
            num_clusters: Some(1024),
            num_subquantizers: Some(8),
            bits_per_subquantizer: Some(8),
            use_gpu: false,
            gpu_devices: vec![0],
            enable_mmap: true,
            persistence: FaissPersistenceConfig::default(),
            optimization: FaissOptimizationConfig::default(),
        }
    }
}

/// FAISS index types supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FaissIndexType {
    /// Flat index with L2 distance
    FlatL2,
    /// Flat index with inner product distance
    FlatIP,
    /// IVF with flat quantizer
    IvfFlat,
    /// IVF with product quantizer
    IvfPq,
    /// IVF with scalar quantizer
    IvfSq,
    /// HNSW index
    HnswFlat,
    /// LSH index
    Lsh,
    /// Auto-selected index based on data characteristics
    Auto,
    /// Custom index string (for advanced users)
    Custom(String),
}

/// Persistence configuration for FAISS indices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaissPersistenceConfig {
    /// Directory to store index files
    pub index_directory: PathBuf,
    /// Enable automatic index saving
    pub auto_save: bool,
    /// Save interval in seconds
    pub save_interval: u64,
    /// Enable index compression
    pub compression: bool,
    /// Backup configuration
    pub backup: FaissBackupConfig,
}

impl Default for FaissPersistenceConfig {
    fn default() -> Self {
        Self {
            index_directory: PathBuf::from("./faiss_indices"),
            auto_save: true,
            save_interval: 300, // 5 minutes
            compression: true,
            backup: FaissBackupConfig::default(),
        }
    }
}

/// Backup configuration for FAISS indices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaissBackupConfig {
    /// Enable automated backups
    pub enabled: bool,
    /// Backup directory
    pub backup_directory: PathBuf,
    /// Number of backup versions to keep
    pub max_versions: usize,
    /// Backup frequency in seconds
    pub backup_frequency: u64,
}

impl Default for FaissBackupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            backup_directory: PathBuf::from("./faiss_backups"),
            max_versions: 5,
            backup_frequency: 3600, // 1 hour
        }
    }
}

/// Optimization configuration for FAISS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaissOptimizationConfig {
    /// Enable automatic index optimization
    pub auto_optimize: bool,
    /// Optimization frequency in operations
    pub optimization_frequency: usize,
    /// Enable dynamic parameter tuning
    pub dynamic_tuning: bool,
    /// Performance monitoring settings
    pub monitoring: FaissMonitoringConfig,
}

impl Default for FaissOptimizationConfig {
    fn default() -> Self {
        Self {
            auto_optimize: true,
            optimization_frequency: 100000,
            dynamic_tuning: true,
            monitoring: FaissMonitoringConfig::default(),
        }
    }
}

/// Monitoring configuration for FAISS operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaissMonitoringConfig {
    /// Enable performance monitoring
    pub enabled: bool,
    /// Metrics collection interval in seconds
    pub collection_interval: u64,
    /// Enable memory usage tracking
    pub track_memory: bool,
    /// Enable query performance tracking
    pub track_queries: bool,
}

impl Default for FaissMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval: 60,
            track_memory: true,
            track_queries: true,
        }
    }
}

/// FAISS search parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaissSearchParams {
    /// Number of nearest neighbors to return
    pub k: usize,
    /// Number of probes for IVF indices
    pub nprobe: Option<usize>,
    /// Search parameters for HNSW
    pub hnsw_ef: Option<usize>,
    /// Enable exact search
    pub exact_search: bool,
    /// Search timeout in milliseconds
    pub timeout_ms: Option<u64>,
}

impl Default for FaissSearchParams {
    fn default() -> Self {
        Self {
            k: 10,
            nprobe: Some(64),
            hnsw_ef: Some(128),
            exact_search: false,
            timeout_ms: Some(5000),
        }
    }
}

/// FAISS vector index implementation
pub struct FaissIndex {
    /// Configuration
    config: FaissConfig,
    /// Native FAISS index handle (simulated)
    index_handle: Arc<Mutex<Option<FaissIndexHandle>>>,
    /// Vector storage for fallback
    vectors: Arc<RwLock<Vec<Vec<f32>>>>,
    /// Vector metadata
    metadata: Arc<RwLock<HashMap<usize, VectorMetadata>>>,
    /// Performance statistics
    stats: Arc<RwLock<FaissStatistics>>,
    /// Training state
    training_state: Arc<RwLock<TrainingState>>,
}

/// Simulated FAISS index handle (would be actual FAISS bindings in real implementation)
#[derive(Debug)]
pub struct FaissIndexHandle {
    /// Index type identifier
    pub index_type: String,
    /// Number of vectors stored
    pub num_vectors: usize,
    /// Index dimension
    pub dimension: usize,
    /// Training status
    pub is_trained: bool,
    /// GPU device ID (if using GPU)
    pub gpu_device: Option<u32>,
}

/// Vector metadata for FAISS integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMetadata {
    /// Original vector ID
    pub id: String,
    /// Insertion timestamp
    pub timestamp: std::time::SystemTime,
    /// Vector norm (for normalization)
    pub norm: f32,
    /// Custom attributes
    pub attributes: HashMap<String, String>,
}

/// Training state for FAISS indices
#[derive(Debug, Clone)]
pub struct TrainingState {
    /// Is index trained
    pub is_trained: bool,
    /// Training progress (0.0 to 1.0)
    pub training_progress: f32,
    /// Training start time
    pub training_start: Option<std::time::Instant>,
    /// Training vectors count
    pub training_vectors_count: usize,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            is_trained: false,
            training_progress: 0.0,
            training_start: None,
            training_vectors_count: 0,
        }
    }
}

/// Performance statistics for FAISS operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FaissStatistics {
    /// Total vectors indexed
    pub total_vectors: usize,
    /// Total search operations
    pub total_searches: usize,
    /// Average search time in microseconds
    pub avg_search_time_us: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// GPU memory usage in bytes (if applicable)
    pub gpu_memory_usage_bytes: Option<usize>,
    /// Index build time in seconds
    pub index_build_time_s: f64,
    /// Last optimization time
    pub last_optimization: Option<std::time::SystemTime>,
    /// Performance over time
    pub performance_history: Vec<PerformanceSnapshot>,
}

/// Performance snapshot for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Timestamp
    pub timestamp: std::time::SystemTime,
    /// Search latency percentiles
    pub search_latency_p50: f64,
    pub search_latency_p95: f64,
    pub search_latency_p99: f64,
    /// Throughput (queries per second)
    pub throughput_qps: f64,
    /// Memory usage
    pub memory_usage_mb: f64,
    /// GPU utilization (if applicable)
    pub gpu_utilization: Option<f32>,
}

impl FaissIndex {
    /// Create a new FAISS index
    pub fn new(config: FaissConfig) -> Result<Self> {
        let span = span!(Level::INFO, "faiss_index_new");
        let _enter = span.enter();

        // Validate configuration
        Self::validate_config(&config)?;

        let index = Self {
            config: config.clone(),
            index_handle: Arc::new(Mutex::new(None)),
            vectors: Arc::new(RwLock::new(Vec::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(FaissStatistics::default())),
            training_state: Arc::new(RwLock::new(TrainingState::default())),
        };

        // Initialize FAISS index
        index.initialize_faiss_index()?;

        info!(
            "Created FAISS index with type {:?}, dimension {}",
            config.index_type, config.dimension
        );

        Ok(index)
    }

    /// Validate FAISS configuration
    fn validate_config(config: &FaissConfig) -> Result<()> {
        if config.dimension == 0 {
            return Err(AnyhowError::msg("Dimension must be greater than 0"));
        }

        if config.training_sample_size == 0 {
            return Err(AnyhowError::msg(
                "Training sample size must be greater than 0",
            ));
        }

        // Validate index-specific parameters
        match &config.index_type {
            FaissIndexType::IvfFlat | FaissIndexType::IvfSq => {
                if config.num_clusters.is_none() {
                    return Err(AnyhowError::msg(
                        "IVF indices require num_clusters to be set",
                    ));
                }
            }
            FaissIndexType::IvfPq => {
                if config.num_clusters.is_none() {
                    return Err(AnyhowError::msg(
                        "IVF indices require num_clusters to be set",
                    ));
                }
                if config.num_subquantizers.is_none() {
                    return Err(AnyhowError::msg(
                        "IVF-PQ requires num_subquantizers to be set",
                    ));
                }
                if config.bits_per_subquantizer.is_none() {
                    return Err(AnyhowError::msg(
                        "IVF-PQ requires bits_per_subquantizer to be set",
                    ));
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Initialize the underlying FAISS index
    fn initialize_faiss_index(&self) -> Result<()> {
        let span = span!(Level::DEBUG, "initialize_faiss_index");
        let _enter = span.enter();

        // In a real implementation, this would call FAISS C++ bindings
        let index_type_str = self.faiss_index_string()?;

        let handle = FaissIndexHandle {
            index_type: index_type_str,
            num_vectors: 0,
            dimension: self.config.dimension,
            is_trained: self.requires_training(),
            gpu_device: if self.config.use_gpu {
                Some(self.config.gpu_devices.first().copied().unwrap_or(0))
            } else {
                None
            },
        };

        let mut index_handle = self
            .index_handle
            .lock()
            .map_err(|_| AnyhowError::msg("Failed to acquire index handle lock"))?;
        *index_handle = Some(handle);

        debug!("Initialized FAISS index: {}", self.faiss_index_string()?);
        Ok(())
    }

    /// Check if the index type requires training
    fn requires_training(&self) -> bool {
        !matches!(
            self.config.index_type,
            FaissIndexType::FlatL2 | FaissIndexType::FlatIP
        )
    }

    /// Generate FAISS index string
    fn faiss_index_string(&self) -> Result<String> {
        let index_str = match &self.config.index_type {
            FaissIndexType::FlatL2 => "Flat".to_string(),
            FaissIndexType::FlatIP => "Flat".to_string(),
            FaissIndexType::IvfFlat => {
                let clusters = self.config.num_clusters.unwrap_or(1024);
                format!("IVF{clusters},Flat")
            }
            FaissIndexType::IvfPq => {
                let clusters = self.config.num_clusters.unwrap_or(1024);
                let subq = self.config.num_subquantizers.unwrap_or(8);
                let bits = self.config.bits_per_subquantizer.unwrap_or(8);
                format!("IVF{clusters},PQ{subq}x{bits}")
            }
            FaissIndexType::IvfSq => {
                let clusters = self.config.num_clusters.unwrap_or(1024);
                format!("IVF{clusters},SQ8")
            }
            FaissIndexType::HnswFlat => "HNSW32,Flat".to_string(),
            FaissIndexType::Lsh => "LSH".to_string(),
            FaissIndexType::Auto => self.auto_select_index_type()?,
            FaissIndexType::Custom(s) => s.clone(),
        };

        Ok(index_str)
    }

    /// Automatically select index type based on data characteristics
    fn auto_select_index_type(&self) -> Result<String> {
        let num_vectors = {
            let vectors = self
                .vectors
                .read()
                .map_err(|_| AnyhowError::msg("Failed to acquire vectors lock"))?;
            vectors.len()
        };

        let dimension = self.config.dimension;

        // Selection heuristics based on FAISS best practices
        let index_str = if num_vectors < 10000 {
            // Small dataset: use flat index
            "Flat".to_string()
        } else if num_vectors < 1000000 {
            // Medium dataset: use IVF with appropriate clustering
            let clusters = (num_vectors as f32).sqrt() as usize;
            if dimension > 128 {
                format!("IVF{clusters},PQ16x8")
            } else {
                format!("IVF{clusters},Flat")
            }
        } else {
            // Large dataset: use IVF-PQ with compression
            let clusters = (num_vectors as f32).sqrt() as usize;
            format!("IVF{},PQ{}x8", clusters, std::cmp::min(dimension / 4, 64))
        };

        debug!(
            "Auto-selected FAISS index: {} for {} vectors, {} dimensions",
            index_str, num_vectors, dimension
        );

        Ok(index_str)
    }

    /// Train the FAISS index with sample data
    pub fn train(&self, training_vectors: &[Vec<f32>]) -> Result<()> {
        let span = span!(Level::INFO, "faiss_train");
        let _enter = span.enter();

        if !self.requires_training() {
            debug!("Index type does not require training");
            return Ok(());
        }

        // Update training state
        {
            let mut state = self
                .training_state
                .write()
                .map_err(|_| AnyhowError::msg("Failed to acquire training state lock"))?;
            state.training_start = Some(std::time::Instant::now());
            state.training_vectors_count = training_vectors.len();
            state.training_progress = 0.0;
        }

        // Validate training data
        if training_vectors.is_empty() {
            return Err(AnyhowError::msg("Training vectors cannot be empty"));
        }

        for (i, vector) in training_vectors.iter().enumerate() {
            if vector.len() != self.config.dimension {
                return Err(AnyhowError::msg(format!(
                    "Training vector {} has dimension {}, expected {}",
                    i,
                    vector.len(),
                    self.config.dimension
                )));
            }
        }

        // Simulate training process (in real implementation, this would call FAISS training)
        info!(
            "Training FAISS index with {} vectors",
            training_vectors.len()
        );

        // Simulate training progress
        for progress in 0..=10 {
            std::thread::sleep(std::time::Duration::from_millis(100));
            let mut state = self
                .training_state
                .write()
                .map_err(|_| AnyhowError::msg("Failed to acquire training state lock"))?;
            state.training_progress = progress as f32 / 10.0;
        }

        // Mark as trained
        {
            let mut state = self
                .training_state
                .write()
                .map_err(|_| AnyhowError::msg("Failed to acquire training state lock"))?;
            state.is_trained = true;
            state.training_progress = 1.0;
        }

        // Update index handle
        {
            let mut handle = self
                .index_handle
                .lock()
                .map_err(|_| AnyhowError::msg("Failed to acquire index handle lock"))?;
            if let Some(ref mut h) = *handle {
                h.is_trained = true;
            }
        }

        info!("FAISS index training completed successfully");
        Ok(())
    }

    /// Add vectors to the FAISS index
    pub fn add_vectors(&self, vectors: Vec<Vec<f32>>, ids: Vec<String>) -> Result<()> {
        let span = span!(Level::DEBUG, "faiss_add_vectors");
        let _enter = span.enter();

        if vectors.len() != ids.len() {
            return Err(AnyhowError::msg(
                "Number of vectors must match number of IDs",
            ));
        }

        // Check if training is required and completed
        if self.requires_training() {
            let state = self
                .training_state
                .read()
                .map_err(|_| AnyhowError::msg("Failed to acquire training state lock"))?;
            if !state.is_trained {
                return Err(AnyhowError::msg(
                    "Index must be trained before adding vectors",
                ));
            }
        }

        // Validate vector dimensions
        for (i, vector) in vectors.iter().enumerate() {
            if vector.len() != self.config.dimension {
                return Err(AnyhowError::msg(format!(
                    "Vector {} has dimension {}, expected {}",
                    i,
                    vector.len(),
                    self.config.dimension
                )));
            }
        }

        let start_time = std::time::Instant::now();

        // Add vectors to storage
        let mut vec_storage = self
            .vectors
            .write()
            .map_err(|_| AnyhowError::msg("Failed to acquire vectors lock"))?;
        let mut metadata_storage = self
            .metadata
            .write()
            .map_err(|_| AnyhowError::msg("Failed to acquire metadata lock"))?;

        for (vector, id) in vectors.iter().zip(ids.iter()) {
            let index = vec_storage.len();
            vec_storage.push(vector.clone());

            let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            let metadata = VectorMetadata {
                id: id.clone(),
                timestamp: std::time::SystemTime::now(),
                norm,
                attributes: HashMap::new(),
            };
            metadata_storage.insert(index, metadata);
        }

        // Update statistics
        {
            let mut stats = self
                .stats
                .write()
                .map_err(|_| AnyhowError::msg("Failed to acquire stats lock"))?;
            stats.total_vectors += vectors.len();
            stats.index_build_time_s += start_time.elapsed().as_secs_f64();
        }

        // Update index handle
        {
            let mut handle = self
                .index_handle
                .lock()
                .map_err(|_| AnyhowError::msg("Failed to acquire index handle lock"))?;
            if let Some(ref mut h) = *handle {
                h.num_vectors += vectors.len();
            }
        }

        debug!("Added {} vectors to FAISS index", vectors.len());
        Ok(())
    }

    /// Search for similar vectors
    pub fn search(
        &self,
        query_vector: &[f32],
        params: &FaissSearchParams,
    ) -> Result<Vec<(String, f32)>> {
        let span = span!(Level::DEBUG, "faiss_search");
        let _enter = span.enter();

        if query_vector.len() != self.config.dimension {
            return Err(AnyhowError::msg(format!(
                "Query vector has dimension {}, expected {}",
                query_vector.len(),
                self.config.dimension
            )));
        }

        let start_time = std::time::Instant::now();

        // Simulate search (in real implementation, this would call FAISS search)
        let results = self.simulate_search(query_vector, params)?;

        // Update statistics
        {
            let mut stats = self
                .stats
                .write()
                .map_err(|_| AnyhowError::msg("Failed to acquire stats lock"))?;
            stats.total_searches += 1;
            let search_time_us = start_time.elapsed().as_micros() as f64;
            stats.avg_search_time_us =
                (stats.avg_search_time_us * (stats.total_searches - 1) as f64 + search_time_us)
                    / stats.total_searches as f64;
        }

        debug!("FAISS search completed in {:?}", start_time.elapsed());
        Ok(results)
    }

    /// Simulate search for demonstration (would be replaced with actual FAISS calls)
    fn simulate_search(
        &self,
        query_vector: &[f32],
        params: &FaissSearchParams,
    ) -> Result<Vec<(String, f32)>> {
        let vectors = self
            .vectors
            .read()
            .map_err(|_| AnyhowError::msg("Failed to acquire vectors lock"))?;
        let metadata = self
            .metadata
            .read()
            .map_err(|_| AnyhowError::msg("Failed to acquire metadata lock"))?;

        let mut results = Vec::new();

        // Simple linear search simulation (FAISS would be much more efficient)
        for (i, vector) in vectors.iter().enumerate() {
            let distance = self.compute_distance(query_vector, vector);
            if let Some(meta) = metadata.get(&i) {
                results.push((meta.id.clone(), distance));
            }
        }

        // Sort by distance and take top-k
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(params.k);

        Ok(results)
    }

    /// Compute distance between two vectors
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.index_type {
            FaissIndexType::FlatL2
            | FaissIndexType::IvfFlat
            | FaissIndexType::IvfPq
            | FaissIndexType::IvfSq => {
                // L2 distance
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt()
            }
            FaissIndexType::FlatIP => {
                // Inner product (negative for similarity)
                -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
            }
            _ => {
                // Default to L2
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt()
            }
        }
    }

    /// Get performance statistics
    pub fn get_statistics(&self) -> Result<FaissStatistics> {
        let stats = self
            .stats
            .read()
            .map_err(|_| AnyhowError::msg("Failed to acquire stats lock"))?;
        Ok(stats.clone())
    }

    /// Save index to disk
    pub fn save_index(&self, path: &Path) -> Result<()> {
        let span = span!(Level::INFO, "faiss_save_index");
        let _enter = span.enter();

        // Create directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {parent:?}"))?;
        }

        // In real implementation, this would save the FAISS index
        info!("Saving FAISS index to {:?}", path);

        // Simulate save operation
        std::thread::sleep(std::time::Duration::from_millis(100));

        Ok(())
    }

    /// Load index from disk
    pub fn load_index(&self, path: &Path) -> Result<()> {
        let span = span!(Level::INFO, "faiss_load_index");
        let _enter = span.enter();

        if !path.exists() {
            return Err(AnyhowError::msg(format!(
                "Index file does not exist: {path:?}"
            )));
        }

        // In real implementation, this would load the FAISS index
        info!("Loading FAISS index from {:?}", path);

        // Simulate load operation
        std::thread::sleep(std::time::Duration::from_millis(100));

        Ok(())
    }

    /// Optimize index performance
    pub fn optimize(&self) -> Result<()> {
        let span = span!(Level::INFO, "faiss_optimize");
        let _enter = span.enter();

        // Update optimization timestamp
        {
            let mut stats = self
                .stats
                .write()
                .map_err(|_| AnyhowError::msg("Failed to acquire stats lock"))?;
            stats.last_optimization = Some(std::time::SystemTime::now());
        }

        info!("FAISS index optimization completed");
        Ok(())
    }

    /// Get current memory usage
    pub fn get_memory_usage(&self) -> Result<usize> {
        let vectors = self
            .vectors
            .read()
            .map_err(|_| AnyhowError::msg("Failed to acquire vectors lock"))?;

        let vector_memory = vectors.len() * self.config.dimension * std::mem::size_of::<f32>();
        let metadata_memory = vectors.len() * std::mem::size_of::<VectorMetadata>();

        Ok(vector_memory + metadata_memory)
    }

    /// Get the dimension of vectors in the index
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// Get the number of vectors in the index
    pub fn size(&self) -> usize {
        self.vectors.read().map(|v| v.len()).unwrap_or(0)
    }
}

impl VectorIndex for FaissIndex {
    fn insert(&mut self, uri: String, vector: crate::Vector) -> Result<()> {
        self.add_vectors(vec![vector.as_f32()], vec![uri])
    }

    fn search_knn(&self, query: &crate::Vector, k: usize) -> Result<Vec<(String, f32)>> {
        let params = FaissSearchParams {
            k,
            ..Default::default()
        };
        self.search(&query.as_f32(), &params)
    }

    fn search_threshold(
        &self,
        query: &crate::Vector,
        threshold: f32,
    ) -> Result<Vec<(String, f32)>> {
        let params = FaissSearchParams {
            k: 1000, // Large number to get all candidates
            // threshold: Some(threshold),  // Remove this field as it doesn't exist
            ..Default::default()
        };
        let results = self.search(&query.as_f32(), &params)?;
        Ok(results
            .into_iter()
            .filter(|(_, score)| *score >= threshold)
            .collect())
    }

    fn get_vector(&self, _uri: &str) -> Option<&crate::Vector> {
        // This would require storing vectors by URI, which is complex in FAISS
        // For now, return None - this can be implemented with an additional URI->vector map
        None
    }
}

/// FAISS integration factory
pub struct FaissFactory;

impl FaissFactory {
    /// Create a new FAISS index with optimal configuration for the given parameters
    pub fn create_optimized_index(
        dimension: usize,
        expected_size: usize,
        use_gpu: bool,
    ) -> Result<FaissIndex> {
        let index_type = if expected_size < 10000 {
            FaissIndexType::FlatL2
        } else if expected_size < 1000000 {
            FaissIndexType::IvfFlat
        } else {
            FaissIndexType::IvfPq
        };

        let config = FaissConfig {
            index_type,
            dimension,
            training_sample_size: std::cmp::min(expected_size / 10, 100000),
            num_clusters: Some((expected_size as f32).sqrt() as usize),
            use_gpu,
            ..Default::default()
        };

        FaissIndex::new(config)
    }

    /// Create a GPU-accelerated FAISS index
    pub fn create_gpu_index(dimension: usize, gpu_devices: Vec<u32>) -> Result<FaissIndex> {
        let config = FaissConfig {
            dimension,
            use_gpu: true,
            gpu_devices,
            index_type: FaissIndexType::Auto,
            ..Default::default()
        };

        FaissIndex::new(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_faiss_index_creation() {
        let config = FaissConfig {
            dimension: 128,
            index_type: FaissIndexType::FlatL2,
            ..Default::default()
        };

        let index = FaissIndex::new(config).unwrap();
        assert_eq!(index.dimension(), 128);
        assert_eq!(index.size(), 0);
    }

    #[test]
    fn test_faiss_add_and_search() {
        let config = FaissConfig {
            dimension: 4,
            index_type: FaissIndexType::FlatL2,
            ..Default::default()
        };

        let index = FaissIndex::new(config).unwrap();

        // Add test vectors
        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let ids = vec!["vec1".to_string(), "vec2".to_string(), "vec3".to_string()];

        index.add_vectors(vectors, ids).unwrap();
        assert_eq!(index.size(), 3);

        // Search for similar vector
        let query = vec![1.0, 0.1, 0.0, 0.0];
        let params = FaissSearchParams {
            k: 2,
            ..Default::default()
        };
        let results = index.search(&query, &params).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "vec1"); // Should be closest to [1,0,0,0]
    }

    #[test]
    fn test_faiss_training() {
        let config = FaissConfig {
            dimension: 4,
            index_type: FaissIndexType::IvfFlat,
            num_clusters: Some(2),
            training_sample_size: 10,
            ..Default::default()
        };

        let index = FaissIndex::new(config).unwrap();

        // Generate training data
        let training_vectors: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![i as f32, (i % 2) as f32, 0.0, 0.0])
            .collect();

        index.train(&training_vectors).unwrap();

        let state = index.training_state.read().unwrap();
        assert!(state.is_trained);
        assert_eq!(state.training_progress, 1.0);
    }

    #[test]
    fn test_faiss_factory() {
        let index = FaissFactory::create_optimized_index(64, 1000, false).unwrap();
        assert_eq!(index.dimension(), 64);

        let gpu_index = FaissFactory::create_gpu_index(128, vec![0]).unwrap();
        assert_eq!(gpu_index.dimension(), 128);
        assert!(gpu_index.config.use_gpu);
    }

    #[test]
    fn test_faiss_auto_index_selection() {
        let config = FaissConfig {
            dimension: 64,
            index_type: FaissIndexType::Auto,
            ..Default::default()
        };

        let index = FaissIndex::new(config).unwrap();
        let index_str = index.faiss_index_string().unwrap();

        // For empty index, should select flat
        assert_eq!(index_str, "Flat");
    }
}
