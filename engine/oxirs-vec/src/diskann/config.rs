//! Configuration for DiskANN index

use serde::{Deserialize, Serialize};

/// Pruning strategy for graph construction
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, oxicode::Encode, oxicode::Decode,
)]
pub enum PruningStrategy {
    /// Standard alpha pruning
    Alpha,
    /// Robust pruning with diversification
    Robust,
    /// Hybrid approach
    Hybrid,
}

/// Search mode for queries
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, oxicode::Encode, oxicode::Decode,
)]
pub enum SearchMode {
    /// Load full index into memory (fastest)
    InMemory,
    /// Stream vectors from disk (memory efficient)
    Streaming,
    /// Hybrid with caching
    Cached,
}

/// Configuration for DiskANN index
#[derive(Debug, Clone, Serialize, Deserialize, oxicode::Encode, oxicode::Decode)]
pub struct DiskAnnConfig {
    /// Vector dimensionality
    pub dimension: usize,

    /// Maximum degree of graph nodes (R parameter)
    pub max_degree: usize,

    /// Beam width for graph construction (L parameter)
    pub build_beam_width: usize,

    /// Beam width for search (L_search parameter)
    pub search_beam_width: usize,

    /// Alpha parameter for pruning (typically 1.2)
    pub alpha: f32,

    /// Pruning strategy
    pub pruning_strategy: PruningStrategy,

    /// Search mode
    pub search_mode: SearchMode,

    /// Maximum number of vectors to keep in memory
    pub max_vectors_in_memory: Option<usize>,

    /// Use PQ compression for disk storage
    pub use_pq_compression: bool,

    /// Number of PQ sub-vectors (if compression enabled)
    pub pq_subvectors: Option<usize>,

    /// Number of bits per PQ code
    pub pq_bits: Option<u8>,

    /// Enable incremental updates
    pub enable_incremental_updates: bool,

    /// Number of entry points for search
    pub num_entry_points: usize,

    /// Buffer size for disk I/O (bytes)
    pub io_buffer_size: usize,
}

impl DiskAnnConfig {
    /// Create default configuration
    pub fn default_config(dimension: usize) -> Self {
        Self {
            dimension,
            max_degree: 64,
            build_beam_width: 100,
            search_beam_width: 75,
            alpha: 1.2,
            pruning_strategy: PruningStrategy::Robust,
            search_mode: SearchMode::Cached,
            max_vectors_in_memory: Some(100_000),
            use_pq_compression: false,
            pq_subvectors: None,
            pq_bits: None,
            enable_incremental_updates: false,
            num_entry_points: 1,
            io_buffer_size: 1 << 20, // 1MB
        }
    }

    /// Create configuration optimized for memory
    pub fn memory_optimized(dimension: usize) -> Self {
        Self {
            dimension,
            max_degree: 32,
            build_beam_width: 75,
            search_beam_width: 50,
            alpha: 1.2,
            pruning_strategy: PruningStrategy::Robust,
            search_mode: SearchMode::Streaming,
            max_vectors_in_memory: Some(10_000),
            use_pq_compression: true,
            pq_subvectors: Some(dimension / 16),
            pq_bits: Some(8),
            enable_incremental_updates: false,
            num_entry_points: 1,
            io_buffer_size: 512 * 1024, // 512KB
        }
    }

    /// Create configuration optimized for speed
    pub fn speed_optimized(dimension: usize) -> Self {
        Self {
            dimension,
            max_degree: 96,
            build_beam_width: 150,
            search_beam_width: 100,
            alpha: 1.2,
            pruning_strategy: PruningStrategy::Alpha,
            search_mode: SearchMode::InMemory,
            max_vectors_in_memory: Some(1_000_000),
            use_pq_compression: false,
            pq_subvectors: None,
            pq_bits: None,
            enable_incremental_updates: true,
            num_entry_points: 4,
            io_buffer_size: 4 << 20, // 4MB
        }
    }

    /// Create configuration for billion-scale datasets
    pub fn billion_scale(dimension: usize) -> Self {
        Self {
            dimension,
            max_degree: 64,
            build_beam_width: 100,
            search_beam_width: 64,
            alpha: 1.2,
            pruning_strategy: PruningStrategy::Robust,
            search_mode: SearchMode::Streaming,
            max_vectors_in_memory: Some(50_000),
            use_pq_compression: true,
            pq_subvectors: Some(dimension / 8),
            pq_bits: Some(8),
            enable_incremental_updates: false,
            num_entry_points: 8,
            io_buffer_size: 2 << 20, // 2MB
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.dimension == 0 {
            return Err("Dimension must be greater than 0".to_string());
        }

        if self.max_degree == 0 {
            return Err("Max degree must be greater than 0".to_string());
        }

        if self.build_beam_width == 0 {
            return Err("Build beam width must be greater than 0".to_string());
        }

        if self.search_beam_width == 0 {
            return Err("Search beam width must be greater than 0".to_string());
        }

        if self.alpha <= 0.0 {
            return Err("Alpha must be positive".to_string());
        }

        if self.use_pq_compression {
            if self.pq_subvectors.is_none() {
                return Err(
                    "PQ subvectors must be specified when compression is enabled".to_string(),
                );
            }
            if self.pq_bits.is_none() {
                return Err("PQ bits must be specified when compression is enabled".to_string());
            }

            let pq_subvectors = self
                .pq_subvectors
                .expect("pq_subvectors validated as Some above");
            if self.dimension % pq_subvectors != 0 {
                return Err(format!(
                    "Dimension {} must be divisible by PQ subvectors {}",
                    self.dimension, pq_subvectors
                ));
            }

            let pq_bits = self.pq_bits.expect("pq_bits validated as Some above");
            if pq_bits == 0 || pq_bits > 16 {
                return Err("PQ bits must be between 1 and 16".to_string());
            }
        }

        if self.num_entry_points == 0 {
            return Err("Number of entry points must be greater than 0".to_string());
        }

        if self.io_buffer_size == 0 {
            return Err("IO buffer size must be greater than 0".to_string());
        }

        Ok(())
    }

    /// Get memory estimate for index (in bytes)
    pub fn estimate_memory_usage(&self, num_vectors: usize) -> usize {
        // Graph structure: node_id (4 bytes) + neighbors (max_degree * 4 bytes)
        let graph_memory = num_vectors * (4 + self.max_degree * 4);

        // Vector data
        let vector_memory = if self.use_pq_compression {
            let pq_subvectors = self.pq_subvectors.unwrap_or(self.dimension / 8);
            let pq_bits = self.pq_bits.unwrap_or(8);
            let bytes_per_code = (pq_bits as usize + 7) / 8;
            num_vectors * pq_subvectors * bytes_per_code
        } else {
            num_vectors * self.dimension * 4 // f32
        };

        // In-memory vectors
        let inmem_vectors = self
            .max_vectors_in_memory
            .unwrap_or(num_vectors)
            .min(num_vectors);
        let inmem_memory = inmem_vectors * self.dimension * 4;

        graph_memory + vector_memory + inmem_memory + self.io_buffer_size
    }
}

impl Default for DiskAnnConfig {
    fn default() -> Self {
        Self::default_config(128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DiskAnnConfig::default_config(128);
        assert_eq!(config.dimension, 128);
        assert_eq!(config.max_degree, 64);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_memory_optimized() {
        let config = DiskAnnConfig::memory_optimized(256);
        assert_eq!(config.dimension, 256);
        assert!(config.use_pq_compression);
        assert_eq!(config.search_mode, SearchMode::Streaming);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_speed_optimized() {
        let config = DiskAnnConfig::speed_optimized(512);
        assert_eq!(config.dimension, 512);
        assert!(!config.use_pq_compression);
        assert_eq!(config.search_mode, SearchMode::InMemory);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_billion_scale() {
        let config = DiskAnnConfig::billion_scale(768);
        assert_eq!(config.dimension, 768);
        assert!(config.use_pq_compression);
        assert_eq!(config.search_mode, SearchMode::Streaming);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation() {
        let mut config = DiskAnnConfig::default_config(128);
        assert!(config.validate().is_ok());

        config.dimension = 0;
        assert!(config.validate().is_err());

        config = DiskAnnConfig::default_config(128);
        config.max_degree = 0;
        assert!(config.validate().is_err());

        config = DiskAnnConfig::default_config(128);
        config.use_pq_compression = true;
        assert!(config.validate().is_err()); // Missing PQ parameters
    }

    #[test]
    fn test_memory_estimation() {
        let config = DiskAnnConfig::default_config(128);
        let memory = config.estimate_memory_usage(1_000_000);
        assert!(memory > 0);

        let pq_config = DiskAnnConfig::memory_optimized(128);
        let pq_memory = pq_config.estimate_memory_usage(1_000_000);
        assert!(pq_memory < memory); // PQ should use less memory
    }

    #[test]
    fn test_pq_validation() {
        let mut config = DiskAnnConfig::default_config(128);
        config.use_pq_compression = true;
        config.pq_subvectors = Some(16);
        config.pq_bits = Some(8);
        assert!(config.validate().is_ok());

        config.pq_subvectors = Some(15); // 128 not divisible by 15
        assert!(config.validate().is_err());

        config.pq_subvectors = Some(16);
        config.pq_bits = Some(0);
        assert!(config.validate().is_err());

        config.pq_bits = Some(20); // > 16
        assert!(config.validate().is_err());
    }
}
