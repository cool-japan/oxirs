//! # Bitmap Index Implementation with Compression
//!
//! High-performance bitmap index implementation with multiple compression algorithms
//! optimized for analytical queries, range operations, and set operations on large datasets.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

/// Bitmap compression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BitmapCompression {
    /// Run-Length Encoding - good for sparse bitmaps
    RLE,
    /// Word-Aligned Hybrid (WAH) compression
    WAH,
    /// Enhanced Word-Aligned Hybrid (EWAH)
    EWAH,
    /// Roaring Bitmaps - excellent for general use
    Roaring,
    /// No compression (raw bitmap)
    None,
}

impl Default for BitmapCompression {
    fn default() -> Self {
        BitmapCompression::Roaring
    }
}

/// Compressed bitmap representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedBitmap {
    /// Compression algorithm used
    compression: BitmapCompression,
    /// Compressed data
    data: Vec<u8>,
    /// Original bitmap size in bits
    size: usize,
    /// Cardinality (number of set bits)
    cardinality: usize,
    /// Metadata for compression
    metadata: HashMap<String, String>,
}

impl CompressedBitmap {
    /// Create a new compressed bitmap from a bit vector
    pub fn new(bits: &[bool], compression: BitmapCompression) -> Result<Self> {
        let data = match compression {
            BitmapCompression::RLE => Self::compress_rle(bits)?,
            BitmapCompression::WAH => Self::compress_wah(bits)?,
            BitmapCompression::EWAH => Self::compress_ewah(bits)?,
            BitmapCompression::Roaring => Self::compress_roaring(bits)?,
            BitmapCompression::None => Self::compress_none(bits)?,
        };

        let cardinality = bits.iter().filter(|&&b| b).count();

        Ok(Self {
            compression,
            data,
            size: bits.len(),
            cardinality,
            metadata: HashMap::new(),
        })
    }

    /// Create from raw compressed data
    pub fn from_compressed(
        data: Vec<u8>,
        compression: BitmapCompression,
        size: usize,
        cardinality: usize,
    ) -> Self {
        Self {
            compression,
            data,
            size,
            cardinality,
            metadata: HashMap::new(),
        }
    }

    /// Decompress the bitmap to a bit vector
    pub fn decompress(&self) -> Result<Vec<bool>> {
        match self.compression {
            BitmapCompression::RLE => Self::decompress_rle(&self.data, self.size),
            BitmapCompression::WAH => Self::decompress_wah(&self.data, self.size),
            BitmapCompression::EWAH => Self::decompress_ewah(&self.data, self.size),
            BitmapCompression::Roaring => Self::decompress_roaring(&self.data, self.size),
            BitmapCompression::None => Self::decompress_none(&self.data, self.size),
        }
    }

    /// RLE compression
    fn compress_rle(bits: &[bool]) -> Result<Vec<u8>> {
        let mut result = Vec::new();
        if bits.is_empty() {
            return Ok(result);
        }

        let mut current_bit = bits[0];
        let mut run_length = 1u32;

        for &bit in &bits[1..] {
            if bit == current_bit && run_length < u32::MAX {
                run_length += 1;
            } else {
                // Encode current run: 1 bit for value + 31 bits for length
                let encoded = if current_bit {
                    0x80000000u32 | run_length
                } else {
                    run_length
                };
                result.extend_from_slice(&encoded.to_le_bytes());

                current_bit = bit;
                run_length = 1;
            }
        }

        // Encode final run
        let encoded = if current_bit {
            0x80000000u32 | run_length
        } else {
            run_length
        };
        result.extend_from_slice(&encoded.to_le_bytes());

        Ok(result)
    }

    /// RLE decompression
    fn decompress_rle(data: &[u8], expected_size: usize) -> Result<Vec<bool>> {
        let mut result = Vec::new();
        let mut i = 0;

        while i + 4 <= data.len() && result.len() < expected_size {
            let encoded = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
            let bit_value = (encoded & 0x80000000) != 0;
            let run_length = (encoded & 0x7FFFFFFF) as usize;

            for _ in 0..run_length.min(expected_size - result.len()) {
                result.push(bit_value);
            }

            i += 4;
        }

        // Pad with false if needed
        while result.len() < expected_size {
            result.push(false);
        }

        Ok(result)
    }

    /// WAH compression (simplified implementation)
    fn compress_wah(bits: &[bool]) -> Result<Vec<u8>> {
        let mut result = Vec::new();
        let word_size = 31; // 31 bits per word, 1 bit for type
        let mut i = 0;

        while i < bits.len() {
            let end = (i + word_size).min(bits.len());
            let chunk = &bits[i..end];

            // Check if this is a fill word (all 0s or all 1s)
            let all_zeros = chunk.iter().all(|&b| !b);
            let all_ones = chunk.iter().all(|&b| b);

            if all_zeros || all_ones {
                // Count consecutive fill words
                let mut fill_count = 1u32;
                let mut next_i = end;

                while next_i < bits.len() {
                    let next_end = (next_i + word_size).min(bits.len());
                    let next_chunk = &bits[next_i..next_end];

                    if (all_zeros && next_chunk.iter().all(|&b| !b))
                        || (all_ones && next_chunk.iter().all(|&b| b))
                    {
                        fill_count += 1;
                        next_i = next_end;
                    } else {
                        break;
                    }
                }

                // Encode fill word: 1 bit type + 1 bit value + 30 bits count
                let encoded = 0x80000000u32
                    | (if all_ones { 0x40000000u32 } else { 0u32 })
                    | (fill_count & 0x3FFFFFFF);
                result.extend_from_slice(&encoded.to_le_bytes());

                i = next_i;
            } else {
                // Literal word: pack bits
                let mut word = 0u32;
                for (j, &bit) in chunk.iter().enumerate() {
                    if bit {
                        word |= 1u32 << j;
                    }
                }
                // Type bit is 0 for literal
                result.extend_from_slice(&word.to_le_bytes());
                i = end;
            }
        }

        Ok(result)
    }

    /// WAH decompression
    fn decompress_wah(data: &[u8], expected_size: usize) -> Result<Vec<bool>> {
        let mut result = Vec::new();
        let mut i = 0;

        while i + 4 <= data.len() && result.len() < expected_size {
            let word = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);

            if (word & 0x80000000) != 0 {
                // Fill word
                let bit_value = (word & 0x40000000) != 0;
                let count = (word & 0x3FFFFFFF) as usize;

                for _ in 0..(count * 31).min(expected_size - result.len()) {
                    result.push(bit_value);
                }
            } else {
                // Literal word
                for bit_pos in 0..31 {
                    if result.len() >= expected_size {
                        break;
                    }
                    result.push((word & (1u32 << bit_pos)) != 0);
                }
            }

            i += 4;
        }

        // Pad with false if needed
        while result.len() < expected_size {
            result.push(false);
        }

        Ok(result)
    }

    /// EWAH compression (Enhanced WAH)
    fn compress_ewah(bits: &[bool]) -> Result<Vec<u8>> {
        // For simplicity, use WAH compression with metadata
        let mut data = Self::compress_wah(bits)?;

        // Add EWAH header (simplified)
        let header = (bits.len() as u32).to_le_bytes();
        let mut result = header.to_vec();
        result.append(&mut data);

        Ok(result)
    }

    /// EWAH decompression
    fn decompress_ewah(data: &[u8], expected_size: usize) -> Result<Vec<bool>> {
        if data.len() < 4 {
            return Err(anyhow!("EWAH data too short"));
        }

        // Skip header and use WAH decompression
        Self::decompress_wah(&data[4..], expected_size)
    }

    /// Roaring bitmap compression (simplified bit-packed representation)
    fn compress_roaring(bits: &[bool]) -> Result<Vec<u8>> {
        let mut result = Vec::new();

        // Group bits into 64-bit chunks and store set bit positions
        let chunk_size = 64;
        let mut chunk_count = 0u32;

        for chunk_start in (0..bits.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(bits.len());
            let chunk = &bits[chunk_start..chunk_end];

            let mut positions = Vec::new();
            for (i, &bit) in chunk.iter().enumerate() {
                if bit {
                    positions.push(i as u16);
                }
            }

            if !positions.is_empty() {
                // Store chunk header: chunk_id + position_count
                result.extend_from_slice(&(chunk_count as u32).to_le_bytes());
                result.extend_from_slice(&(positions.len() as u32).to_le_bytes());

                // Store positions
                for pos in positions {
                    result.extend_from_slice(&pos.to_le_bytes());
                }
            }

            chunk_count += 1;
        }

        Ok(result)
    }

    /// Roaring bitmap decompression
    fn decompress_roaring(data: &[u8], expected_size: usize) -> Result<Vec<bool>> {
        let mut result = vec![false; expected_size];
        let mut i = 0;

        while i + 8 <= data.len() {
            let chunk_id =
                u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
            let pos_count =
                u32::from_le_bytes([data[i + 4], data[i + 5], data[i + 6], data[i + 7]]) as usize;
            i += 8;

            let chunk_start = chunk_id * 64;

            for _ in 0..pos_count {
                if i + 2 <= data.len() {
                    let pos = u16::from_le_bytes([data[i], data[i + 1]]) as usize;
                    let abs_pos = chunk_start + pos;
                    if abs_pos < result.len() {
                        result[abs_pos] = true;
                    }
                    i += 2;
                } else {
                    break;
                }
            }
        }

        Ok(result)
    }

    /// No compression (store raw bits)
    fn compress_none(bits: &[bool]) -> Result<Vec<u8>> {
        let mut result = Vec::new();
        let mut byte = 0u8;
        let mut bit_count = 0;

        for &bit in bits {
            if bit {
                byte |= 1u8 << bit_count;
            }
            bit_count += 1;

            if bit_count == 8 {
                result.push(byte);
                byte = 0;
                bit_count = 0;
            }
        }

        // Handle remaining bits
        if bit_count > 0 {
            result.push(byte);
        }

        Ok(result)
    }

    /// No compression decompression
    fn decompress_none(data: &[u8], expected_size: usize) -> Result<Vec<bool>> {
        let mut result = Vec::new();

        for &byte in data {
            for bit_pos in 0..8 {
                if result.len() >= expected_size {
                    break;
                }
                result.push((byte & (1u8 << bit_pos)) != 0);
            }
        }

        // Pad or truncate to expected size
        result.resize(expected_size, false);
        Ok(result)
    }

    /// Perform AND operation with another bitmap
    pub fn and(&self, other: &CompressedBitmap) -> Result<CompressedBitmap> {
        let bits1 = self.decompress()?;
        let bits2 = other.decompress()?;

        let max_len = bits1.len().max(bits2.len());
        let mut result_bits = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let bit1 = bits1.get(i).copied().unwrap_or(false);
            let bit2 = bits2.get(i).copied().unwrap_or(false);
            result_bits.push(bit1 && bit2);
        }

        CompressedBitmap::new(&result_bits, self.compression)
    }

    /// Perform OR operation with another bitmap
    pub fn or(&self, other: &CompressedBitmap) -> Result<CompressedBitmap> {
        let bits1 = self.decompress()?;
        let bits2 = other.decompress()?;

        let max_len = bits1.len().max(bits2.len());
        let mut result_bits = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let bit1 = bits1.get(i).copied().unwrap_or(false);
            let bit2 = bits2.get(i).copied().unwrap_or(false);
            result_bits.push(bit1 || bit2);
        }

        CompressedBitmap::new(&result_bits, self.compression)
    }

    /// Perform NOT operation
    pub fn not(&self) -> Result<CompressedBitmap> {
        let bits = self.decompress()?;
        let result_bits: Vec<bool> = bits.iter().map(|&b| !b).collect();
        CompressedBitmap::new(&result_bits, self.compression)
    }

    /// Perform XOR operation with another bitmap
    pub fn xor(&self, other: &CompressedBitmap) -> Result<CompressedBitmap> {
        let bits1 = self.decompress()?;
        let bits2 = other.decompress()?;

        let max_len = bits1.len().max(bits2.len());
        let mut result_bits = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let bit1 = bits1.get(i).copied().unwrap_or(false);
            let bit2 = bits2.get(i).copied().unwrap_or(false);
            result_bits.push(bit1 != bit2);
        }

        CompressedBitmap::new(&result_bits, self.compression)
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        let uncompressed_size = (self.size + 7) / 8; // Bytes needed for raw bitmap
        if uncompressed_size == 0 {
            return 1.0;
        }
        self.data.len() as f64 / uncompressed_size as f64
    }

    /// Get cardinality (number of set bits)
    pub fn cardinality(&self) -> usize {
        self.cardinality
    }

    /// Get bitmap size in bits
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get compressed data size in bytes
    pub fn compressed_size(&self) -> usize {
        self.data.len()
    }

    /// Check if bitmap is empty (no set bits)
    pub fn is_empty(&self) -> bool {
        self.cardinality == 0
    }

    /// Extract range of bits as a new bitmap
    pub fn range(&self, start: usize, end: usize) -> Result<CompressedBitmap> {
        let bits = self.decompress()?;
        let end = end.min(bits.len());

        if start >= bits.len() || start >= end {
            return CompressedBitmap::new(&[], self.compression);
        }

        let range_bits = &bits[start..end];
        CompressedBitmap::new(range_bits, self.compression)
    }

    /// Count set bits in a range
    pub fn count_range(&self, start: usize, end: usize) -> Result<usize> {
        let range_bitmap = self.range(start, end)?;
        Ok(range_bitmap.cardinality())
    }

    /// Find the position of the nth set bit (0-indexed)
    pub fn select(&self, n: usize) -> Result<Option<usize>> {
        if n >= self.cardinality {
            return Ok(None);
        }

        let bits = self.decompress()?;
        let mut count = 0;

        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                if count == n {
                    return Ok(Some(i));
                }
                count += 1;
            }
        }

        Ok(None)
    }

    /// Count set bits up to position (exclusive)
    pub fn rank(&self, position: usize) -> Result<usize> {
        let bits = self.decompress()?;
        let end = position.min(bits.len());

        Ok(bits[..end].iter().filter(|&&b| b).count())
    }
}

/// Bitmap index configuration
#[derive(Debug, Clone)]
pub struct BitmapIndexConfig {
    /// Default compression algorithm
    pub compression: BitmapCompression,
    /// Enable automatic compression selection
    pub auto_compression: bool,
    /// Maximum bitmap size before splitting
    pub max_bitmap_size: usize,
    /// Cache size for frequently accessed bitmaps
    pub cache_size: usize,
}

impl Default for BitmapIndexConfig {
    fn default() -> Self {
        Self {
            compression: BitmapCompression::Roaring,
            auto_compression: true,
            max_bitmap_size: 1024 * 1024, // 1M bits
            cache_size: 1024 * 1024 * 10, // 10MB cache
        }
    }
}

/// Bitmap index statistics
#[derive(Debug, Clone, Default, Serialize)]
pub struct BitmapIndexStats {
    pub total_bitmaps: usize,
    pub total_size_bytes: usize,
    pub total_compressed_bytes: usize,
    pub avg_compression_ratio: f64,
    pub avg_cardinality: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub operation_count: u64,
}

/// Bitmap index for efficient set operations
pub struct BitmapIndex<K>
where
    K: Clone + Debug + Ord + Hash + Eq,
{
    /// Map from keys to compressed bitmaps
    bitmaps: Arc<RwLock<BTreeMap<K, CompressedBitmap>>>,
    /// Configuration
    config: BitmapIndexConfig,
    /// Statistics
    stats: BitmapIndexStats,
    /// Cache for frequently accessed bitmaps
    cache: Arc<RwLock<HashMap<K, Vec<bool>>>>,
}

impl<K> BitmapIndex<K>
where
    K: Clone + Debug + Ord + Hash + Eq,
{
    /// Create a new bitmap index
    pub fn new() -> Self {
        Self::with_config(BitmapIndexConfig::default())
    }

    /// Create a new bitmap index with custom configuration
    pub fn with_config(config: BitmapIndexConfig) -> Self {
        Self {
            bitmaps: Arc::new(RwLock::new(BTreeMap::new())),
            config,
            stats: BitmapIndexStats::default(),
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set a bitmap for a key
    pub fn set_bitmap(&mut self, key: K, bits: &[bool]) -> Result<()> {
        let compression = if self.config.auto_compression {
            self.select_optimal_compression(bits)
        } else {
            self.config.compression
        };

        let compressed = CompressedBitmap::new(bits, compression)?;

        {
            let mut bitmaps = self.bitmaps.write().unwrap();
            bitmaps.insert(key.clone(), compressed);
        }

        // Update cache
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(key, bits.to_vec());

            // Maintain cache size
            if cache.len() > self.config.cache_size / 1024 {
                // Remove oldest entries (simplified LRU)
                let keys_to_remove: Vec<_> = cache.keys().take(cache.len() / 2).cloned().collect();
                for k in keys_to_remove {
                    cache.remove(&k);
                }
            }
        }

        self.stats.total_bitmaps += 1;
        Ok(())
    }

    /// Get a bitmap for a key
    pub fn get_bitmap(&mut self, key: &K) -> Result<Option<Vec<bool>>> {
        self.stats.operation_count += 1;

        // Check cache first
        {
            let cache = self.cache.read().unwrap();
            if let Some(bits) = cache.get(key) {
                self.stats.cache_hits += 1;
                return Ok(Some(bits.clone()));
            }
        }

        self.stats.cache_misses += 1;

        // Load from storage
        let bitmaps = self.bitmaps.read().unwrap();
        if let Some(compressed) = bitmaps.get(key) {
            let bits = compressed.decompress()?;

            // Update cache
            {
                let mut cache = self.cache.write().unwrap();
                cache.insert(key.clone(), bits.clone());
            }

            Ok(Some(bits))
        } else {
            Ok(None)
        }
    }

    /// Perform AND operation on multiple bitmaps
    pub fn and_operation(&mut self, keys: &[K]) -> Result<Option<Vec<bool>>> {
        if keys.is_empty() {
            return Ok(None);
        }

        let bitmaps = self.bitmaps.read().unwrap();
        let mut result: Option<CompressedBitmap> = None;

        for key in keys {
            if let Some(bitmap) = bitmaps.get(key) {
                result = Some(match result {
                    Some(acc) => acc.and(bitmap)?,
                    None => bitmap.clone(),
                });
            } else {
                // If any bitmap is missing, result is empty
                return Ok(Some(vec![false; 0]));
            }
        }

        match result {
            Some(bitmap) => Ok(Some(bitmap.decompress()?)),
            None => Ok(None),
        }
    }

    /// Perform OR operation on multiple bitmaps
    pub fn or_operation(&mut self, keys: &[K]) -> Result<Option<Vec<bool>>> {
        if keys.is_empty() {
            return Ok(None);
        }

        let bitmaps = self.bitmaps.read().unwrap();
        let mut result: Option<CompressedBitmap> = None;

        for key in keys {
            if let Some(bitmap) = bitmaps.get(key) {
                result = Some(match result {
                    Some(acc) => acc.or(bitmap)?,
                    None => bitmap.clone(),
                });
            }
        }

        match result {
            Some(bitmap) => Ok(Some(bitmap.decompress()?)),
            None => Ok(None),
        }
    }

    /// Select optimal compression based on bitmap characteristics
    fn select_optimal_compression(&self, bits: &[bool]) -> BitmapCompression {
        let cardinality = bits.iter().filter(|&&b| b).count();
        let size = bits.len();
        let density = cardinality as f64 / size as f64;

        if density < 0.01 {
            // Very sparse - RLE is good
            BitmapCompression::RLE
        } else if density > 0.5 {
            // Dense - WAH might be better
            BitmapCompression::WAH
        } else {
            // Medium density - Roaring is generally good
            BitmapCompression::Roaring
        }
    }

    /// Remove a bitmap
    pub fn remove_bitmap(&mut self, key: &K) -> bool {
        let removed = {
            let mut bitmaps = self.bitmaps.write().unwrap();
            bitmaps.remove(key).is_some()
        };

        {
            let mut cache = self.cache.write().unwrap();
            cache.remove(key);
        }

        if removed {
            self.stats.total_bitmaps -= 1;
        }

        removed
    }

    /// Get all keys
    pub fn keys(&self) -> Vec<K> {
        let bitmaps = self.bitmaps.read().unwrap();
        bitmaps.keys().cloned().collect()
    }

    /// Get statistics
    pub fn get_stats(&mut self) -> BitmapIndexStats {
        let bitmaps = self.bitmaps.read().unwrap();

        let mut total_size = 0;
        let mut total_compressed = 0;
        let mut total_cardinality = 0;

        for bitmap in bitmaps.values() {
            total_size += bitmap.size();
            total_compressed += bitmap.compressed_size();
            total_cardinality += bitmap.cardinality();
        }

        self.stats.total_bitmaps = bitmaps.len();
        self.stats.total_size_bytes = total_size;
        self.stats.total_compressed_bytes = total_compressed;
        self.stats.avg_compression_ratio = if total_size > 0 {
            total_compressed as f64 / total_size as f64
        } else {
            1.0
        };
        self.stats.avg_cardinality = if bitmaps.len() > 0 {
            total_cardinality as f64 / bitmaps.len() as f64
        } else {
            0.0
        };

        self.stats.clone()
    }

    /// Clear all bitmaps
    pub fn clear(&mut self) {
        {
            let mut bitmaps = self.bitmaps.write().unwrap();
            bitmaps.clear();
        }

        {
            let mut cache = self.cache.write().unwrap();
            cache.clear();
        }

        self.stats = BitmapIndexStats::default();
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        let bitmaps = self.bitmaps.read().unwrap();
        bitmaps.is_empty()
    }

    /// Get the number of bitmaps
    pub fn len(&self) -> usize {
        let bitmaps = self.bitmaps.read().unwrap();
        bitmaps.len()
    }
}

impl<K> Default for BitmapIndex<K>
where
    K: Clone + Debug + Ord + Hash + Eq,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressed_bitmap_rle() {
        let bits = vec![true, true, true, false, false, false, false, true];
        let compressed = CompressedBitmap::new(&bits, BitmapCompression::RLE).unwrap();
        let decompressed = compressed.decompress().unwrap();
        assert_eq!(bits, decompressed);
    }

    #[test]
    fn test_compressed_bitmap_wah() {
        let bits = vec![true; 100]; // All ones
        let compressed = CompressedBitmap::new(&bits, BitmapCompression::WAH).unwrap();
        let decompressed = compressed.decompress().unwrap();
        assert_eq!(bits, decompressed);
    }

    #[test]
    fn test_compressed_bitmap_roaring() {
        let mut bits = vec![false; 1000];
        bits[10] = true;
        bits[100] = true;
        bits[500] = true;

        let compressed = CompressedBitmap::new(&bits, BitmapCompression::Roaring).unwrap();
        let decompressed = compressed.decompress().unwrap();
        assert_eq!(bits, decompressed);
    }

    #[test]
    fn test_bitmap_operations() {
        let bits1 = vec![true, false, true, false];
        let bits2 = vec![false, true, true, false];

        let bitmap1 = CompressedBitmap::new(&bits1, BitmapCompression::RLE).unwrap();
        let bitmap2 = CompressedBitmap::new(&bits2, BitmapCompression::RLE).unwrap();

        // Test AND
        let and_result = bitmap1.and(&bitmap2).unwrap().decompress().unwrap();
        assert_eq!(and_result, vec![false, false, true, false]);

        // Test OR
        let or_result = bitmap1.or(&bitmap2).unwrap().decompress().unwrap();
        assert_eq!(or_result, vec![true, true, true, false]);

        // Test XOR
        let xor_result = bitmap1.xor(&bitmap2).unwrap().decompress().unwrap();
        assert_eq!(xor_result, vec![true, true, false, false]);
    }

    #[test]
    fn test_bitmap_index_basic_operations() {
        let mut index = BitmapIndex::new();

        let bits1 = vec![true, false, true, false];
        let bits2 = vec![false, true, true, false];

        index.set_bitmap("key1".to_string(), &bits1).unwrap();
        index.set_bitmap("key2".to_string(), &bits2).unwrap();

        assert_eq!(index.get_bitmap(&"key1".to_string()).unwrap(), Some(bits1));
        assert_eq!(index.get_bitmap(&"key2".to_string()).unwrap(), Some(bits2));
        assert_eq!(index.get_bitmap(&"nonexistent".to_string()).unwrap(), None);
    }

    #[test]
    fn test_bitmap_index_operations() {
        let mut index = BitmapIndex::new();

        let bits1 = vec![true, false, true, false];
        let bits2 = vec![false, true, true, false];

        index.set_bitmap("key1".to_string(), &bits1).unwrap();
        index.set_bitmap("key2".to_string(), &bits2).unwrap();

        let keys = vec!["key1".to_string(), "key2".to_string()];

        // Test AND operation
        let and_result = index.and_operation(&keys).unwrap().unwrap();
        assert_eq!(and_result, vec![false, false, true, false]);

        // Test OR operation
        let or_result = index.or_operation(&keys).unwrap().unwrap();
        assert_eq!(or_result, vec![true, true, true, false]);
    }

    #[test]
    fn test_compression_ratios() {
        // Test sparse bitmap (should compress well)
        let mut sparse_bits = vec![false; 1000];
        sparse_bits[10] = true;
        sparse_bits[500] = true;

        let sparse_compressed =
            CompressedBitmap::new(&sparse_bits, BitmapCompression::RLE).unwrap();
        assert!(sparse_compressed.compression_ratio() < 1.0); // Should be smaller than uncompressed

        // Test dense bitmap
        let dense_bits = vec![true; 1000];
        let dense_compressed = CompressedBitmap::new(&dense_bits, BitmapCompression::RLE).unwrap();
        assert!(dense_compressed.compression_ratio() < 1.0); // Should be smaller than uncompressed

        // Verify compression actually happened
        assert!(sparse_compressed.compressed_size() < (sparse_bits.len() + 7) / 8);
    }

    #[test]
    fn test_bitmap_statistics() {
        let mut index = BitmapIndex::new();

        let bits = vec![true, false, true, false, true];
        index.set_bitmap("test".to_string(), &bits).unwrap();

        let stats = index.get_stats();
        assert_eq!(stats.total_bitmaps, 1);
        assert!(stats.avg_cardinality > 0.0);
    }
}
