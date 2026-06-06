//! Codec implementations: run-length encoding, delta encoding,
//! scalar quantization, product quantization, PCA compression.

use super::compression_types::VectorCompressor;
use crate::{Vector, VectorData, VectorError};
use std::io::{Read, Write};

// ─────────────────────────────────────────────────────────────────────────────
// Euclidean distance helper (used by ProductQuantizer)
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// ZstdCompressor
// ─────────────────────────────────────────────────────────────────────────────

/// Compressor backed by zstd (Pure Rust via oxiarc-zstd)
pub struct ZstdCompressor {
    pub(crate) level: i32,
}

impl ZstdCompressor {
    pub fn new(level: i32) -> Self {
        Self {
            level: level.clamp(1, 22),
        }
    }
}

impl VectorCompressor for ZstdCompressor {
    fn compress(&self, vector: &Vector) -> Result<Vec<u8>, VectorError> {
        let bytes = vector_to_bytes(vector)?;
        oxiarc_zstd::encode_all(&bytes, self.level)
            .map_err(|e| VectorError::CompressionError(e.to_string()))
    }

    fn decompress(&self, data: &[u8], dimensions: usize) -> Result<Vector, VectorError> {
        let decompressed = oxiarc_zstd::decode_all(data)
            .map_err(|e| VectorError::CompressionError(e.to_string()))?;
        bytes_to_vector(&decompressed, dimensions)
    }

    fn compression_ratio(&self) -> f32 {
        match self.level {
            1..=3 => 0.7,
            4..=9 => 0.5,
            10..=15 => 0.4,
            16..=22 => 0.3,
            _ => 1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ScalarQuantizer
// ─────────────────────────────────────────────────────────────────────────────

/// Uniform scalar quantization to N bits
pub struct ScalarQuantizer {
    pub(crate) bits: u8,
    pub(crate) min_val: f32,
    pub(crate) max_val: f32,
}

impl ScalarQuantizer {
    pub fn new(bits: u8) -> Self {
        Self {
            bits: bits.clamp(1, 16),
            min_val: 0.0,
            max_val: 1.0,
        }
    }

    pub fn with_range(bits: u8, min_val: f32, max_val: f32) -> Self {
        Self {
            bits: bits.clamp(1, 16),
            min_val,
            max_val,
        }
    }

    pub fn train(&mut self, vectors: &[Vector]) -> Result<(), VectorError> {
        if vectors.is_empty() {
            return Err(VectorError::InvalidDimensions(
                "No vectors to train on".to_string(),
            ));
        }

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        for vector in vectors {
            match &vector.values {
                VectorData::F32(v) => {
                    for &val in v {
                        min = min.min(val);
                        max = max.max(val);
                    }
                }
                VectorData::F64(v) => {
                    for &val in v {
                        min = min.min(val as f32);
                        max = max.max(val as f32);
                    }
                }
                _ => {}
            }
        }

        self.min_val = min;
        self.max_val = max;
        Ok(())
    }

    pub(crate) fn quantize_value(&self, value: f32) -> u16 {
        let normalized = ((value - self.min_val) / (self.max_val - self.min_val)).clamp(0.0, 1.0);
        let max_quant_val = (1u32 << self.bits) - 1;
        (normalized * max_quant_val as f32).round() as u16
    }

    pub(crate) fn dequantize_value(&self, quantized: u16) -> f32 {
        let max_quant_val = (1u32 << self.bits) - 1;
        let normalized = quantized as f32 / max_quant_val as f32;
        normalized * (self.max_val - self.min_val) + self.min_val
    }
}

impl VectorCompressor for ScalarQuantizer {
    fn compress(&self, vector: &Vector) -> Result<Vec<u8>, VectorError> {
        let values = match &vector.values {
            VectorData::F32(v) => v.clone(),
            VectorData::F64(v) => v.iter().map(|&x| x as f32).collect(),
            _ => {
                return Err(VectorError::UnsupportedOperation(
                    "Quantization only supports float vectors".to_string(),
                ))
            }
        };

        let mut compressed = Vec::new();

        compressed.write_all(&self.bits.to_le_bytes())?;
        compressed.write_all(&self.min_val.to_le_bytes())?;
        compressed.write_all(&self.max_val.to_le_bytes())?;

        if self.bits <= 8 {
            for val in values {
                let quantized = self.quantize_value(val) as u8;
                compressed.push(quantized);
            }
        } else {
            for val in values {
                let quantized = self.quantize_value(val);
                compressed.write_all(&quantized.to_le_bytes())?;
            }
        }

        Ok(compressed)
    }

    fn decompress(&self, data: &[u8], dimensions: usize) -> Result<Vector, VectorError> {
        let mut cursor = std::io::Cursor::new(data);

        let mut bits_buf = [0u8; 1];
        cursor.read_exact(&mut bits_buf)?;
        let bits = bits_buf[0];

        let mut min_buf = [0u8; 4];
        cursor.read_exact(&mut min_buf)?;
        let min_val = f32::from_le_bytes(min_buf);

        let mut max_buf = [0u8; 4];
        cursor.read_exact(&mut max_buf)?;
        let max_val = f32::from_le_bytes(max_buf);

        let quantizer = ScalarQuantizer {
            bits,
            min_val,
            max_val,
        };

        let mut values = Vec::with_capacity(dimensions);

        if bits <= 8 {
            let mut buf = [0u8; 1];
            for _ in 0..dimensions {
                cursor.read_exact(&mut buf)?;
                let quantized = buf[0] as u16;
                values.push(quantizer.dequantize_value(quantized));
            }
        } else {
            let mut buf = [0u8; 2];
            for _ in 0..dimensions {
                cursor.read_exact(&mut buf)?;
                let quantized = u16::from_le_bytes(buf);
                values.push(quantizer.dequantize_value(quantized));
            }
        }

        Ok(Vector::new(values))
    }

    fn compression_ratio(&self) -> f32 {
        self.bits as f32 / 32.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PcaCompressor
// ─────────────────────────────────────────────────────────────────────────────

/// PCA-based dimensionality-reduction compressor
pub struct PcaCompressor {
    pub(crate) components: usize,
    pub(crate) mean: Vec<f32>,
    pub(crate) components_matrix: Vec<Vec<f32>>,
    pub(crate) explained_variance_ratio: Vec<f32>,
}

impl PcaCompressor {
    pub fn new(components: usize) -> Self {
        Self {
            components,
            mean: Vec::new(),
            components_matrix: Vec::new(),
            explained_variance_ratio: Vec::new(),
        }
    }

    pub fn train(&mut self, vectors: &[Vector]) -> Result<(), VectorError> {
        if vectors.is_empty() {
            return Err(VectorError::InvalidDimensions(
                "No vectors to train on".to_string(),
            ));
        }

        let data: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| match &v.values {
                VectorData::F32(vals) => Ok(vals.clone()),
                VectorData::F64(vals) => Ok(vals.iter().map(|&x| x as f32).collect()),
                _ => Err(VectorError::UnsupportedOperation(
                    "PCA only supports float vectors".to_string(),
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let n_samples = data.len();
        let n_features = data[0].len();

        self.mean = vec![0.0; n_features];
        for sample in &data {
            for (i, &val) in sample.iter().enumerate() {
                self.mean[i] += val;
            }
        }
        for val in &mut self.mean {
            *val /= n_samples as f32;
        }

        use nalgebra::DMatrix;

        let training_data: Result<Vec<Vec<f32>>, _> = vectors
            .iter()
            .map(|v| match &v.values {
                VectorData::F32(vals) => Ok(vals.clone()),
                VectorData::F64(vals) => Ok(vals.iter().map(|&x| x as f32).collect()),
                _ => Err(VectorError::UnsupportedOperation(
                    "PCA only supports float vectors".to_string(),
                )),
            })
            .collect();

        let training_data = training_data?;
        let n_samples = training_data.len();
        if n_samples == 0 {
            return Err(VectorError::InvalidDimensions(
                "No training data provided for PCA".to_string(),
            ));
        }

        let mut data_matrix = DMatrix::<f32>::zeros(n_samples, n_features);
        for (i, sample) in training_data.iter().enumerate() {
            for (j, &val) in sample.iter().enumerate() {
                data_matrix[(i, j)] = val - self.mean[j];
            }
        }

        let covariance = data_matrix.transpose() * &data_matrix / (n_samples as f32 - 1.0);
        let svd = covariance.svd(true, true);
        self.components_matrix = Vec::with_capacity(self.components);

        if let Some(u) = svd.u {
            let num_components = self.components.min(u.ncols());
            let singular_values = &svd.singular_values;
            let total_variance: f32 = singular_values.iter().sum();
            let mut explained_variance = Vec::with_capacity(num_components);

            for i in 0..num_components {
                let component: Vec<f32> = u.column(i).iter().cloned().collect();
                self.components_matrix.push(component);

                let variance_ratio = singular_values[i] / total_variance;
                explained_variance.push(variance_ratio);
            }

            self.explained_variance_ratio = explained_variance;
        } else {
            return Err(VectorError::CompressionError(
                "SVD decomposition failed for PCA".to_string(),
            ));
        }

        Ok(())
    }

    fn project(&self, vector: &[f32]) -> Vec<f32> {
        let mut centered = vector.to_vec();
        for (i, val) in centered.iter_mut().enumerate() {
            *val -= self.mean.get(i).unwrap_or(&0.0);
        }

        let mut projected = vec![0.0; self.components];
        for (i, component) in self.components_matrix.iter().enumerate() {
            let mut dot = 0.0;
            for (j, &val) in centered.iter().enumerate() {
                dot += val * component.get(j).unwrap_or(&0.0);
            }
            projected[i] = dot;
        }

        projected
    }

    fn reconstruct(&self, projected: &[f32]) -> Vec<f32> {
        let n_features = self.mean.len();
        let mut reconstructed = self.mean.clone();

        for (i, &coeff) in projected.iter().enumerate() {
            if let Some(component) = self.components_matrix.get(i) {
                for (j, &comp_val) in component.iter().enumerate() {
                    if j < n_features {
                        reconstructed[j] += coeff * comp_val;
                    }
                }
            }
        }

        reconstructed
    }

    /// Get the explained variance ratio for each principal component
    pub fn explained_variance_ratio(&self) -> &[f32] {
        &self.explained_variance_ratio
    }

    /// Get the total explained variance for the selected components
    pub fn total_explained_variance(&self) -> f32 {
        self.explained_variance_ratio.iter().sum()
    }
}

impl VectorCompressor for PcaCompressor {
    fn compress(&self, vector: &Vector) -> Result<Vec<u8>, VectorError> {
        let values = match &vector.values {
            VectorData::F32(v) => v.clone(),
            VectorData::F64(v) => v.iter().map(|&x| x as f32).collect(),
            _ => {
                return Err(VectorError::UnsupportedOperation(
                    "PCA only supports float vectors".to_string(),
                ))
            }
        };

        let projected = self.project(&values);

        let mut compressed = Vec::new();
        compressed.write_all(&(self.components as u32).to_le_bytes())?;

        for val in projected {
            compressed.write_all(&val.to_le_bytes())?;
        }

        Ok(compressed)
    }

    fn decompress(&self, data: &[u8], _dimensions: usize) -> Result<Vector, VectorError> {
        let mut cursor = std::io::Cursor::new(data);

        let mut components_buf = [0u8; 4];
        cursor.read_exact(&mut components_buf)?;
        let components = u32::from_le_bytes(components_buf) as usize;

        let mut projected = Vec::with_capacity(components);
        let mut val_buf = [0u8; 4];

        for _ in 0..components {
            cursor.read_exact(&mut val_buf)?;
            projected.push(f32::from_le_bytes(val_buf));
        }

        let reconstructed = self.reconstruct(&projected);
        Ok(Vector::new(reconstructed))
    }

    fn compression_ratio(&self) -> f32 {
        if self.mean.is_empty() {
            1.0
        } else {
            self.components as f32 / self.mean.len() as f32
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProductQuantizer
// ─────────────────────────────────────────────────────────────────────────────

/// Product Quantization compressor for efficient vector compression
pub struct ProductQuantizer {
    pub(crate) subvectors: usize,
    pub(crate) codebook_size: usize,
    pub(crate) codebooks: Vec<Vec<Vec<f32>>>,
    pub(crate) subvector_dim: usize,
}

impl ProductQuantizer {
    pub fn new(subvectors: usize, codebook_size: usize) -> Self {
        Self {
            subvectors,
            codebook_size,
            codebooks: Vec::new(),
            subvector_dim: 0,
        }
    }

    pub fn train(&mut self, vectors: &[Vector]) -> Result<(), VectorError> {
        if vectors.is_empty() {
            return Err(VectorError::InvalidDimensions(
                "No training data provided for Product Quantization".to_string(),
            ));
        }

        let vector_dim = vectors[0].dimensions;
        if vector_dim % self.subvectors != 0 {
            return Err(VectorError::InvalidDimensions(format!(
                "Vector dimension {} is not divisible by number of subvectors {}",
                vector_dim, self.subvectors
            )));
        }

        self.subvector_dim = vector_dim / self.subvectors;
        self.codebooks = Vec::with_capacity(self.subvectors);

        let training_data: Result<Vec<Vec<f32>>, _> = vectors
            .iter()
            .map(|v| match &v.values {
                VectorData::F32(vals) => Ok(vals.clone()),
                VectorData::F64(vals) => Ok(vals.iter().map(|&x| x as f32).collect()),
                _ => Err(VectorError::UnsupportedOperation(
                    "Product quantization only supports float vectors".to_string(),
                )),
            })
            .collect();

        let training_data = training_data?;

        for subvec_idx in 0..self.subvectors {
            let start_dim = subvec_idx * self.subvector_dim;
            let end_dim = start_dim + self.subvector_dim;

            let subvectors: Vec<Vec<f32>> = training_data
                .iter()
                .map(|v| v[start_dim..end_dim].to_vec())
                .collect();

            let codebook = self.train_codebook(&subvectors)?;
            self.codebooks.push(codebook);
        }

        Ok(())
    }

    fn train_codebook(&self, subvectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, VectorError> {
        use scirs2_core::random::Random;
        let mut rng = Random::seed(42);

        if subvectors.is_empty() {
            return Err(VectorError::InvalidDimensions(
                "No subvectors to train codebook".to_string(),
            ));
        }

        let dim = subvectors[0].len();
        let mut centroids = Vec::with_capacity(self.codebook_size);

        for _ in 0..self.codebook_size {
            let mut centroid = vec![0.0; dim];
            for val in &mut centroid {
                *val = rng.gen_range(-1.0..1.0);
            }
            centroids.push(centroid);
        }

        for _ in 0..10 {
            let mut assignments = vec![0; subvectors.len()];

            for (i, subvec) in subvectors.iter().enumerate() {
                let mut best_dist = f32::INFINITY;
                let mut best_centroid = 0;

                for (j, centroid) in centroids.iter().enumerate() {
                    let dist = euclidean_distance(subvec, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_centroid = j;
                    }
                }
                assignments[i] = best_centroid;
            }

            for (j, centroid) in centroids.iter_mut().enumerate() {
                let assigned_points: Vec<&Vec<f32>> = subvectors
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| assignments[*i] == j)
                    .map(|(_, v)| v)
                    .collect();

                if !assigned_points.is_empty() {
                    for (d, centroid_val) in centroid.iter_mut().enumerate() {
                        *centroid_val = assigned_points.iter().map(|p| p[d]).sum::<f32>()
                            / assigned_points.len() as f32;
                    }
                }
            }
        }

        Ok(centroids)
    }

    pub(crate) fn quantize_vector(&self, vector: &[f32]) -> Result<Vec<u8>, VectorError> {
        if vector.len() != self.subvectors * self.subvector_dim {
            return Err(VectorError::InvalidDimensions(format!(
                "Vector dimension {} doesn't match expected {}",
                vector.len(),
                self.subvectors * self.subvector_dim
            )));
        }

        let mut codes = Vec::with_capacity(self.subvectors);

        for subvec_idx in 0..self.subvectors {
            let start_dim = subvec_idx * self.subvector_dim;
            let end_dim = start_dim + self.subvector_dim;
            let subvector = &vector[start_dim..end_dim];

            let codebook = &self.codebooks[subvec_idx];
            let mut best_dist = f32::INFINITY;
            let mut best_code = 0u8;

            for (code, centroid) in codebook.iter().enumerate() {
                let dist = euclidean_distance(subvector, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_code = code as u8;
                }
            }

            codes.push(best_code);
        }

        Ok(codes)
    }

    pub(crate) fn dequantize_codes(&self, codes: &[u8]) -> Result<Vec<f32>, VectorError> {
        if codes.len() != self.subvectors {
            return Err(VectorError::InvalidDimensions(format!(
                "Code length {} doesn't match expected {}",
                codes.len(),
                self.subvectors
            )));
        }

        let mut reconstructed = Vec::with_capacity(self.subvectors * self.subvector_dim);

        for (subvec_idx, &code) in codes.iter().enumerate() {
            let codebook = &self.codebooks[subvec_idx];
            if (code as usize) < codebook.len() {
                reconstructed.extend_from_slice(&codebook[code as usize]);
            } else {
                return Err(VectorError::InvalidDimensions(format!(
                    "Invalid code {} for codebook of size {}",
                    code,
                    codebook.len()
                )));
            }
        }

        Ok(reconstructed)
    }
}

impl VectorCompressor for ProductQuantizer {
    fn compress(&self, vector: &Vector) -> Result<Vec<u8>, VectorError> {
        let values = match &vector.values {
            VectorData::F32(v) => v.clone(),
            VectorData::F64(v) => v.iter().map(|&x| x as f32).collect(),
            _ => {
                return Err(VectorError::UnsupportedOperation(
                    "Product quantization only supports float vectors".to_string(),
                ))
            }
        };

        let codes = self.quantize_vector(&values)?;

        let mut compressed = Vec::new();
        compressed.write_all(&(self.subvectors as u32).to_le_bytes())?;
        compressed.write_all(&(self.codebook_size as u32).to_le_bytes())?;
        compressed.write_all(&(self.subvector_dim as u32).to_le_bytes())?;
        compressed.extend_from_slice(&codes);

        Ok(compressed)
    }

    fn decompress(&self, data: &[u8], _dimensions: usize) -> Result<Vector, VectorError> {
        if data.len() < 12 {
            return Err(VectorError::InvalidData(
                "Invalid compressed data format".to_string(),
            ));
        }

        let subvectors = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let codebook_size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let subvector_dim = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;

        if subvectors != self.subvectors
            || codebook_size != self.codebook_size
            || subvector_dim != self.subvector_dim
        {
            return Err(VectorError::InvalidData(
                "Metadata mismatch in compressed data".to_string(),
            ));
        }

        let codes = &data[12..];
        if codes.len() != subvectors {
            return Err(VectorError::InvalidData("Invalid code length".to_string()));
        }

        let values = self.dequantize_codes(codes)?;
        Ok(Vector::new(values))
    }

    fn compression_ratio(&self) -> f32 {
        (8.0 * self.subvectors as f32) / (32.0 * self.subvectors as f32 * self.subvector_dim as f32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NoOpCompressor (identity — used as fallback)
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) struct NoOpCompressor;

impl VectorCompressor for NoOpCompressor {
    fn compress(&self, vector: &Vector) -> Result<Vec<u8>, VectorError> {
        vector_to_bytes(vector)
    }

    fn decompress(&self, data: &[u8], dimensions: usize) -> Result<Vector, VectorError> {
        bytes_to_vector(data, dimensions)
    }

    fn compression_ratio(&self) -> f32 {
        1.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Byte-level helpers shared by several codecs
// ─────────────────────────────────────────────────────────────────────────────

pub fn vector_to_bytes(vector: &Vector) -> Result<Vec<u8>, VectorError> {
    let mut bytes = Vec::new();

    let type_byte = match &vector.values {
        VectorData::F32(_) => 0u8,
        VectorData::F64(_) => 1u8,
        VectorData::F16(_) => 2u8,
        VectorData::I8(_) => 3u8,
        VectorData::Binary(_) => 4u8,
    };
    bytes.push(type_byte);

    match &vector.values {
        VectorData::F32(v) => {
            for val in v {
                bytes.write_all(&val.to_le_bytes())?;
            }
        }
        VectorData::F64(v) => {
            for val in v {
                bytes.write_all(&val.to_le_bytes())?;
            }
        }
        VectorData::F16(v) => {
            for val in v {
                bytes.write_all(&val.to_le_bytes())?;
            }
        }
        VectorData::I8(v) => {
            for &val in v {
                bytes.push(val as u8);
            }
        }
        VectorData::Binary(v) => {
            bytes.extend_from_slice(v);
        }
    }

    Ok(bytes)
}

pub fn bytes_to_vector(data: &[u8], dimensions: usize) -> Result<Vector, VectorError> {
    if data.is_empty() {
        return Err(VectorError::InvalidDimensions("Empty data".to_string()));
    }

    let type_byte = data[0];
    let data = &data[1..];

    match type_byte {
        0 => {
            let mut values = Vec::with_capacity(dimensions);
            let mut cursor = std::io::Cursor::new(data);
            let mut buf = [0u8; 4];

            for _ in 0..dimensions {
                cursor.read_exact(&mut buf)?;
                values.push(f32::from_le_bytes(buf));
            }
            Ok(Vector::new(values))
        }
        1 => {
            let mut values = Vec::with_capacity(dimensions);
            let mut cursor = std::io::Cursor::new(data);
            let mut buf = [0u8; 8];

            for _ in 0..dimensions {
                cursor.read_exact(&mut buf)?;
                values.push(f64::from_le_bytes(buf));
            }
            Ok(Vector::f64(values))
        }
        2 => {
            let mut values = Vec::with_capacity(dimensions);
            let mut cursor = std::io::Cursor::new(data);
            let mut buf = [0u8; 2];

            for _ in 0..dimensions {
                cursor.read_exact(&mut buf)?;
                values.push(u16::from_le_bytes(buf));
            }
            Ok(Vector::f16(values))
        }
        3 => Ok(Vector::i8(
            data[..dimensions].iter().map(|&b| b as i8).collect(),
        )),
        4 => Ok(Vector::binary(data[..dimensions].to_vec())),
        _ => Err(VectorError::InvalidData(format!(
            "Unknown vector type: {type_byte}"
        ))),
    }
}
