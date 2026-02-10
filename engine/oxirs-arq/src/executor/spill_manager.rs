//! Disk Spilling Manager for Memory-Efficient Query Execution
//!
//! This module provides disk spilling capabilities when memory pressure is high,
//! enabling execution of queries that would otherwise exceed available memory.

use crate::algebra::{Binding, Solution, Term, Variable};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{remove_file, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// JSON-serializable wrapper for Solution
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableSolution(Vec<SerializableBinding>);

/// JSON-serializable wrapper for Binding
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableBinding(Vec<(String, Term)>);

impl From<&Solution> for SerializableSolution {
    fn from(solution: &Solution) -> Self {
        SerializableSolution(
            solution
                .iter()
                .map(|binding| SerializableBinding::from(binding))
                .collect(),
        )
    }
}

impl From<SerializableSolution> for Solution {
    fn from(ser: SerializableSolution) -> Self {
        ser.0
            .into_iter()
            .map(|ser_binding| {
                let mut binding = Binding::new();
                for (var_name, term) in ser_binding.0 {
                    if let Ok(var) = Variable::new(&var_name) {
                        binding.insert(var, term);
                    }
                }
                binding
            })
            .collect()
    }
}

impl From<&Binding> for SerializableBinding {
    fn from(binding: &Binding) -> Self {
        SerializableBinding(
            binding
                .iter()
                .map(|(var, term)| (var.to_string(), term.clone()))
                .collect(),
        )
    }
}

/// Spill manager for disk-based overflow handling
pub struct SpillManager {
    spill_dir: PathBuf,
    config: SpillConfig,
    current_memory_bytes: Arc<AtomicU64>,
    active_spills: Arc<RwLock<HashMap<SpillId, SpillFile>>>,
}

/// Configuration for spilling operations
#[derive(Debug, Clone)]
pub struct SpillConfig {
    pub spill_threshold_percent: f64, // 0.8 (80% of memory)
    pub max_memory_mb: usize,         // 2048 MB
    pub spill_dir: PathBuf,           // /tmp/oxirs_spill
    pub compression: bool,            // true
}

impl Default for SpillConfig {
    fn default() -> Self {
        Self {
            spill_threshold_percent: 0.8,
            max_memory_mb: 2048,
            spill_dir: std::env::temp_dir().join("oxirs_spill"),
            compression: true,
        }
    }
}

/// Metadata for a spilled file
#[derive(Debug, Clone)]
pub struct SpillFile {
    pub id: SpillId,
    pub path: PathBuf,
    pub size: usize,
    pub num_rows: usize,
    pub compressed: bool,
}

/// Unique identifier for a spill file
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct SpillId(u64);

impl SpillId {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        SpillId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl SpillManager {
    /// Create a new spill manager
    pub fn new(config: SpillConfig) -> Result<Self> {
        // Ensure spill directory exists
        std::fs::create_dir_all(&config.spill_dir)?;

        Ok(Self {
            spill_dir: config.spill_dir.clone(),
            config,
            current_memory_bytes: Arc::new(AtomicU64::new(0)),
            active_spills: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Check if should spill to disk
    pub fn should_spill(&self) -> bool {
        let current_memory_mb = self.current_memory_usage_mb();
        let threshold_mb =
            (self.config.max_memory_mb as f64 * self.config.spill_threshold_percent) as usize;

        current_memory_mb > threshold_mb
    }

    /// Get current memory usage in MB
    pub fn current_memory_usage_mb(&self) -> usize {
        (self.current_memory_bytes.load(Ordering::Relaxed) / (1024 * 1024)) as usize
    }

    /// Spill solution data to disk
    pub fn spill(&mut self, data: &Solution) -> Result<SpillId> {
        if data.is_empty() {
            return Err(anyhow!("Cannot spill empty solution"));
        }

        let spill_id = SpillId::new();
        let path = self.spill_dir.join(format!("spill_{}.bin", spill_id.0));

        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);

        if self.config.compression {
            // Compress before writing
            let compressed = self.compress_solution(data)?;
            writer.write_all(&compressed)?;
            writer.flush()?;

            let size = compressed.len();
            let spill = SpillFile {
                id: spill_id,
                path: path.clone(),
                size,
                num_rows: data.len(),
                compressed: true,
            };

            self.active_spills
                .write()
                .map_err(|_| anyhow!("Failed to acquire write lock"))?
                .insert(spill_id, spill);

            tracing::info!(
                "Spilled {} rows to {} (compressed, {} bytes)",
                data.len(),
                path.display(),
                size
            );
        } else {
            // Write directly without compression using serde_json
            let serializable = SerializableSolution::from(data);
            serde_json::to_writer(&mut writer, &serializable)?;
            writer.flush()?;

            let size = std::fs::metadata(&path)?.len() as usize;
            let spill = SpillFile {
                id: spill_id,
                path: path.clone(),
                size,
                num_rows: data.len(),
                compressed: false,
            };

            self.active_spills
                .write()
                .map_err(|_| anyhow!("Failed to acquire write lock"))?
                .insert(spill_id, spill);

            tracing::info!(
                "Spilled {} rows to {} ({} bytes)",
                data.len(),
                path.display(),
                size
            );
        }

        Ok(spill_id)
    }

    /// Spill individual bindings to disk
    pub fn spill_bindings(&mut self, data: &[Binding]) -> Result<SpillId> {
        let solution: Solution = data.to_vec();
        self.spill(&solution)
    }

    /// Read spilled data back from disk
    pub fn read_spill(&self, spill_id: SpillId) -> Result<Solution> {
        let spills = self
            .active_spills
            .read()
            .map_err(|_| anyhow!("Failed to acquire read lock"))?;
        let spill = spills
            .get(&spill_id)
            .ok_or_else(|| anyhow!("Spill {} not found", spill_id.0))?;

        let file = File::open(&spill.path)?;
        let mut reader = BufReader::new(file);

        let data = if spill.compressed {
            let mut compressed = Vec::new();
            reader.read_to_end(&mut compressed)?;
            self.decompress_solution(&compressed)?
        } else {
            let serializable: SerializableSolution = serde_json::from_reader(&mut reader)?;
            Solution::from(serializable)
        };

        tracing::debug!("Read {} rows from spill {}", data.len(), spill_id.0);
        Ok(data)
    }

    /// Read spill as iterator for streaming
    pub fn read_spill_streaming(&self, spill_id: SpillId) -> Result<SpillIterator> {
        let spills = self
            .active_spills
            .read()
            .map_err(|_| anyhow!("Failed to acquire read lock"))?;
        let spill = spills
            .get(&spill_id)
            .ok_or_else(|| anyhow!("Spill {} not found", spill_id.0))?
            .clone();

        SpillIterator::new(spill)
    }

    /// Clean up spill file
    pub fn cleanup(&mut self, spill_id: SpillId) -> Result<()> {
        let mut spills = self
            .active_spills
            .write()
            .map_err(|_| anyhow!("Failed to acquire write lock"))?;
        if let Some(spill) = spills.remove(&spill_id) {
            if spill.path.exists() {
                remove_file(&spill.path)?;
            }
            tracing::debug!("Cleaned up spill {}", spill_id.0);
        }
        Ok(())
    }

    /// Clean up all spills
    pub fn cleanup_all(&mut self) -> Result<()> {
        let mut spills = self
            .active_spills
            .write()
            .map_err(|_| anyhow!("Failed to acquire write lock"))?;

        for (_, spill) in spills.drain() {
            if spill.path.exists() {
                if let Err(e) = remove_file(&spill.path) {
                    tracing::warn!("Failed to remove spill file {}: {}", spill.path.display(), e);
                }
            }
        }

        tracing::info!("Cleaned up all spill files");
        Ok(())
    }

    /// Get statistics about active spills
    pub fn statistics(&self) -> SpillStatistics {
        let spills = self
            .active_spills
            .read()
            .expect("Failed to acquire read lock");

        let total_size = spills.values().map(|s| s.size).sum();
        let total_rows = spills.values().map(|s| s.num_rows).sum();

        SpillStatistics {
            num_spills: spills.len(),
            total_size_bytes: total_size,
            total_rows,
            average_compression_ratio: self.calculate_average_compression_ratio(&spills),
        }
    }

    fn calculate_average_compression_ratio(&self, spills: &HashMap<SpillId, SpillFile>) -> f64 {
        let compressed_spills: Vec<_> = spills.values().filter(|s| s.compressed).collect();

        if compressed_spills.is_empty() {
            return 1.0;
        }

        // Estimate: assume 80% compression for compressed files
        0.2
    }

    /// Compress solution data
    fn compress_solution(&self, data: &Solution) -> Result<Vec<u8>> {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let serializable = SerializableSolution::from(data);
        let json_data = serde_json::to_vec(&serializable)?;
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&json_data)?;
        let compressed = encoder.finish()?;
        Ok(compressed)
    }

    /// Decompress solution data
    fn decompress_solution(&self, compressed: &[u8]) -> Result<Solution> {
        use flate2::read::GzDecoder;

        let decoder = GzDecoder::new(compressed);
        let mut reader = BufReader::new(decoder);
        let serializable: SerializableSolution = serde_json::from_reader(&mut reader)?;
        Ok(Solution::from(serializable))
    }

    /// Get number of active spills
    pub fn num_active_spills(&self) -> usize {
        self.active_spills
            .read()
            .expect("Failed to acquire read lock")
            .len()
    }
}

impl Drop for SpillManager {
    fn drop(&mut self) {
        // Clean up all spill files on drop
        if let Err(e) = self.cleanup_all() {
            tracing::warn!("Failed to clean up spill files in Drop: {}", e);
        }
    }
}

/// Statistics about spilling operations
#[derive(Debug, Clone)]
pub struct SpillStatistics {
    pub num_spills: usize,
    pub total_size_bytes: usize,
    pub total_rows: usize,
    pub average_compression_ratio: f64,
}

/// Iterator over spilled data for streaming reads
pub struct SpillIterator {
    spill: SpillFile,
    buffer: Option<Solution>,
    current_index: usize,
}

impl SpillIterator {
    fn new(spill: SpillFile) -> Result<Self> {
        Ok(Self {
            spill,
            buffer: None,
            current_index: 0,
        })
    }

    fn load_buffer(&mut self) -> Result<()> {
        let file = File::open(&self.spill.path)?;
        let mut reader = BufReader::new(file);

        let data = if self.spill.compressed {
            use flate2::read::GzDecoder;
            let decoder = GzDecoder::new(reader);
            let mut decompressed_reader = BufReader::new(decoder);
            let serializable: SerializableSolution = serde_json::from_reader(&mut decompressed_reader)?;
            Solution::from(serializable)
        } else {
            let serializable: SerializableSolution = serde_json::from_reader(&mut reader)?;
            Solution::from(serializable)
        };

        self.buffer = Some(data);
        self.current_index = 0;
        Ok(())
    }
}

impl Iterator for SpillIterator {
    type Item = Result<Binding>;

    fn next(&mut self) -> Option<Self::Item> {
        // Load buffer on first access
        if self.buffer.is_none() {
            if let Err(e) = self.load_buffer() {
                return Some(Err(e));
            }
        }

        if let Some(ref buffer) = self.buffer {
            if self.current_index < buffer.len() {
                let binding = buffer[self.current_index].clone();
                self.current_index += 1;
                Some(Ok(binding))
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Variable;

    fn create_test_solution(size: usize) -> Solution {
        let mut solution = Solution::new();
        for i in 0..size {
            let mut binding = Binding::new();
            binding.insert(
                Variable::new(&format!("x{}", i)).expect("valid variable name"),
                crate::algebra::Term::Iri(
                    oxirs_core::model::NamedNode::new(&format!("http://example.org/item{}", i))
                        .expect("valid IRI"),
                ),
            );
            solution.push(binding);
        }
        solution
    }

    #[test]
    fn test_spill_manager_creation() {
        let config = SpillConfig::default();
        let manager = SpillManager::new(config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_spill_and_read() {
        let config = SpillConfig::default();
        let mut manager = SpillManager::new(config).expect("Failed to create manager");

        let test_data = create_test_solution(100);
        let spill_id = manager.spill(&test_data).expect("Failed to spill");

        let read_data = manager.read_spill(spill_id).expect("Failed to read spill");
        assert_eq!(test_data.len(), read_data.len());

        manager.cleanup(spill_id).expect("Failed to cleanup");
    }

    #[test]
    fn test_spill_statistics() {
        let config = SpillConfig::default();
        let mut manager = SpillManager::new(config).expect("Failed to create manager");

        let test_data = create_test_solution(50);
        let _spill_id = manager.spill(&test_data).expect("Failed to spill");

        let stats = manager.statistics();
        assert_eq!(stats.num_spills, 1);
        assert_eq!(stats.total_rows, 50);

        manager.cleanup_all().expect("Failed to cleanup");
    }
}
