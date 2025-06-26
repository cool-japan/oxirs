use crate::{Vector, VectorData, VectorError};
use std::io::{Read, Write};
use zstd;

#[derive(Debug, Clone)]
pub enum CompressionMethod {
    None,
    Zstd { level: i32 },
    Quantization { bits: u8 },
    ProductQuantization { subvectors: usize, codebook_size: usize },
    Pca { components: usize },
}

impl Default for CompressionMethod {
    fn default() -> Self {
        CompressionMethod::None
    }
}

pub trait VectorCompressor: Send + Sync {
    fn compress(&self, vector: &Vector) -> Result<Vec<u8>, VectorError>;
    fn decompress(&self, data: &[u8], dimensions: usize) -> Result<Vector, VectorError>;
    fn compression_ratio(&self) -> f32;
}

pub struct ZstdCompressor {
    level: i32,
}

impl ZstdCompressor {
    pub fn new(level: i32) -> Self {
        Self { level: level.clamp(1, 22) }
    }
}

impl VectorCompressor for ZstdCompressor {
    fn compress(&self, vector: &Vector) -> Result<Vec<u8>, VectorError> {
        let bytes = vector_to_bytes(vector)?;
        zstd::encode_all(&bytes[..], self.level)
            .map_err(|e| VectorError::CompressionError(e.to_string()))
    }

    fn decompress(&self, data: &[u8], dimensions: usize) -> Result<Vector, VectorError> {
        let decompressed = zstd::decode_all(data)
            .map_err(|e| VectorError::CompressionError(e.to_string()))?;
        bytes_to_vector(&decompressed, dimensions)
    }

    fn compression_ratio(&self) -> f32 {
        // Typical compression ratio for float data with zstd
        match self.level {
            1..=3 => 0.7,
            4..=9 => 0.5,
            10..=15 => 0.4,
            16..=22 => 0.3,
            _ => 1.0,
        }
    }
}

pub struct ScalarQuantizer {
    bits: u8,
    min_val: f32,
    max_val: f32,
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
            return Err(VectorError::InvalidDimensions("No vectors to train on".to_string()));
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

    fn quantize_value(&self, value: f32) -> u16 {
        let normalized = ((value - self.min_val) / (self.max_val - self.min_val))
            .clamp(0.0, 1.0);
        let max_quant_val = (1u32 << self.bits) - 1;
        (normalized * max_quant_val as f32).round() as u16
    }

    fn dequantize_value(&self, quantized: u16) -> f32 {
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
            _ => return Err(VectorError::UnsupportedOperation(
                "Quantization only supports float vectors".to_string()
            )),
        };

        let mut compressed = Vec::new();
        
        // Write header: bits, min_val, max_val
        compressed.write_all(&self.bits.to_le_bytes())?;
        compressed.write_all(&self.min_val.to_le_bytes())?;
        compressed.write_all(&self.max_val.to_le_bytes())?;
        
        // Quantize and pack values
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
        
        // Read header
        let mut bits_buf = [0u8; 1];
        cursor.read_exact(&mut bits_buf)?;
        let bits = bits_buf[0];
        
        let mut min_buf = [0u8; 4];
        cursor.read_exact(&mut min_buf)?;
        let min_val = f32::from_le_bytes(min_buf);
        
        let mut max_buf = [0u8; 4];
        cursor.read_exact(&mut max_buf)?;
        let max_val = f32::from_le_bytes(max_buf);
        
        // Create temporary quantizer with loaded params
        let quantizer = ScalarQuantizer {
            bits,
            min_val,
            max_val,
        };
        
        // Dequantize values
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
        // Original: 32 bits per float, compressed: self.bits per float
        self.bits as f32 / 32.0
    }
}

pub struct PcaCompressor {
    components: usize,
    mean: Vec<f32>,
    components_matrix: Vec<Vec<f32>>,
}

impl PcaCompressor {
    pub fn new(components: usize) -> Self {
        Self {
            components,
            mean: Vec::new(),
            components_matrix: Vec::new(),
        }
    }

    pub fn train(&mut self, vectors: &[Vector]) -> Result<(), VectorError> {
        if vectors.is_empty() {
            return Err(VectorError::InvalidDimensions("No vectors to train on".to_string()));
        }

        // Convert all vectors to f32
        let data: Vec<Vec<f32>> = vectors.iter()
            .map(|v| match &v.values {
                VectorData::F32(vals) => Ok(vals.clone()),
                VectorData::F64(vals) => Ok(vals.iter().map(|&x| x as f32).collect()),
                _ => Err(VectorError::UnsupportedOperation(
                    "PCA only supports float vectors".to_string()
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let n_samples = data.len();
        let n_features = data[0].len();

        // Compute mean
        self.mean = vec![0.0; n_features];
        for sample in &data {
            for (i, &val) in sample.iter().enumerate() {
                self.mean[i] += val;
            }
        }
        for val in &mut self.mean {
            *val /= n_samples as f32;
        }

        // Center data
        let mut centered = data.clone();
        for sample in &mut centered {
            for (i, val) in sample.iter_mut().enumerate() {
                *val -= self.mean[i];
            }
        }

        // Simplified PCA: use random projection for now
        // TODO: Implement proper SVD-based PCA
        self.components_matrix = Vec::with_capacity(self.components);
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        
        for _ in 0..self.components {
            let mut component = vec![0.0; n_features];
            let mut norm = 0.0f32;
            
            for val in &mut component {
                *val = rng.gen_range(-1.0..1.0);
                norm += *val * *val;
            }
            
            // Normalize
            norm = norm.sqrt();
            for val in &mut component {
                *val /= norm;
            }
            
            self.components_matrix.push(component);
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
}

impl VectorCompressor for PcaCompressor {
    fn compress(&self, vector: &Vector) -> Result<Vec<u8>, VectorError> {
        let values = match &vector.values {
            VectorData::F32(v) => v.clone(),
            VectorData::F64(v) => v.iter().map(|&x| x as f32).collect(),
            _ => return Err(VectorError::UnsupportedOperation(
                "PCA only supports float vectors".to_string()
            )),
        };

        let projected = self.project(&values);
        
        let mut compressed = Vec::new();
        // Write header with components count
        compressed.write_all(&(self.components as u32).to_le_bytes())?;
        
        // Write projected values
        for val in projected {
            compressed.write_all(&val.to_le_bytes())?;
        }

        Ok(compressed)
    }

    fn decompress(&self, data: &[u8], _dimensions: usize) -> Result<Vector, VectorError> {
        let mut cursor = std::io::Cursor::new(data);
        
        // Read header
        let mut components_buf = [0u8; 4];
        cursor.read_exact(&mut components_buf)?;
        let components = u32::from_le_bytes(components_buf) as usize;
        
        // Read projected values
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
        // Compression ratio depends on dimensionality reduction
        if self.mean.is_empty() {
            1.0
        } else {
            self.components as f32 / self.mean.len() as f32
        }
    }
}

pub fn create_compressor(method: &CompressionMethod) -> Box<dyn VectorCompressor> {
    match method {
        CompressionMethod::None => Box::new(NoOpCompressor),
        CompressionMethod::Zstd { level } => Box::new(ZstdCompressor::new(*level)),
        CompressionMethod::Quantization { bits } => Box::new(ScalarQuantizer::new(*bits)),
        CompressionMethod::Pca { components } => Box::new(PcaCompressor::new(*components)),
        CompressionMethod::ProductQuantization { .. } => {
            // TODO: Implement product quantization
            Box::new(NoOpCompressor)
        }
    }
}

struct NoOpCompressor;

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

fn vector_to_bytes(vector: &Vector) -> Result<Vec<u8>, VectorError> {
    let mut bytes = Vec::new();
    
    // Write type indicator
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

fn bytes_to_vector(data: &[u8], dimensions: usize) -> Result<Vector, VectorError> {
    if data.is_empty() {
        return Err(VectorError::InvalidDimensions("Empty data".to_string()));
    }

    let type_byte = data[0];
    let data = &data[1..];

    match type_byte {
        0 => { // F32
            let mut values = Vec::with_capacity(dimensions);
            let mut cursor = std::io::Cursor::new(data);
            let mut buf = [0u8; 4];
            
            for _ in 0..dimensions {
                cursor.read_exact(&mut buf)?;
                values.push(f32::from_le_bytes(buf));
            }
            Ok(Vector::new(values))
        }
        1 => { // F64
            let mut values = Vec::with_capacity(dimensions);
            let mut cursor = std::io::Cursor::new(data);
            let mut buf = [0u8; 8];
            
            for _ in 0..dimensions {
                cursor.read_exact(&mut buf)?;
                values.push(f64::from_le_bytes(buf));
            }
            Ok(Vector::f64(values))
        }
        2 => { // F16
            let mut values = Vec::with_capacity(dimensions);
            let mut cursor = std::io::Cursor::new(data);
            let mut buf = [0u8; 2];
            
            for _ in 0..dimensions {
                cursor.read_exact(&mut buf)?;
                values.push(u16::from_le_bytes(buf));
            }
            Ok(Vector::f16(values))
        }
        3 => { // I8
            Ok(Vector::i8(data[..dimensions].iter().map(|&b| b as i8).collect()))
        }
        4 => { // Binary
            Ok(Vector::binary(data[..dimensions].to_vec()))
        }
        _ => Err(VectorError::InvalidData(format!("Unknown vector type: {}", type_byte))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zstd_compression() {
        let vector = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let compressor = ZstdCompressor::new(3);
        
        let compressed = compressor.compress(&vector).unwrap();
        let decompressed = compressor.decompress(&compressed, 5).unwrap();
        
        let orig = vector.as_f32();
        let dec = decompressed.as_f32();
        assert_eq!(orig.len(), dec.len());
        for (a, b) in orig.iter().zip(dec.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_scalar_quantization() {
        let vector = Vector::new(vec![0.1, 0.5, 0.9, 0.3, 0.7]);
        let mut quantizer = ScalarQuantizer::new(8);
        quantizer.train(&[vector.clone()]).unwrap();
        
        let compressed = quantizer.compress(&vector).unwrap();
        let decompressed = quantizer.decompress(&compressed, 5).unwrap();
        
        // Check compression ratio
        assert!(compressed.len() < 20); // Should be much smaller than original
        
        let orig = vector.as_f32();
        let dec = decompressed.as_f32();
        assert_eq!(orig.len(), dec.len());
        // With 8-bit quantization, expect some loss
        for (a, b) in orig.iter().zip(dec.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn test_pca_compression() {
        let vectors = vec![
            Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            Vector::new(vec![2.0, 3.0, 4.0, 5.0, 6.0]),
            Vector::new(vec![3.0, 4.0, 5.0, 6.0, 7.0]),
        ];
        
        let mut pca = PcaCompressor::new(3);
        pca.train(&vectors).unwrap();
        
        let compressed = pca.compress(&vectors[0]).unwrap();
        let decompressed = pca.decompress(&compressed, 5).unwrap();
        
        let dec = decompressed.as_f32();
        assert_eq!(dec.len(), 5);
    }
}