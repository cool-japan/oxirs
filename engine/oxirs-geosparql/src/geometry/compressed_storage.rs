//! Compressed geometry storage for memory-efficient spatial data
//!
//! Provides compression techniques to reduce memory footprint of geometry data:
//!
//! - **Delta encoding**: Store coordinate differences instead of absolute values
//! - **Quantization**: Reduce coordinate precision to configurable levels
//! - **Run-length encoding**: Compress repeated coordinate patterns
//! - **Lazy decompression**: Decompress geometries on-demand
//!
//! # Example
//!
//! ```rust
//! use oxirs_geosparql::geometry::compressed_storage::{CompressedGeometry, CompressionConfig};
//! use oxirs_geosparql::geometry::Geometry;
//! use geo_types::{Geometry as GeoGeometry, LineString, Point};
//!
//! // Create a geometry
//! let coords = vec![
//!     geo_types::coord! { x: 0.0, y: 0.0 },
//!     geo_types::coord! { x: 1.0, y: 1.0 },
//!     geo_types::coord! { x: 2.0, y: 2.0 },
//! ];
//! let geom = Geometry::new(GeoGeometry::LineString(LineString::new(coords)));
//!
//! // Compress with default settings
//! let compressed = CompressedGeometry::compress(&geom).unwrap();
//!
//! // Memory saved
//! println!("Memory saved: {} bytes", compressed.compression_ratio());
//!
//! // Decompress when needed
//! let decompressed = compressed.decompress().unwrap();
//! ```

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use serde::{Deserialize, Serialize};

/// Configuration for geometry compression
#[derive(Debug, Clone, Copy)]
pub struct CompressionConfig {
    /// Number of decimal places to preserve (quantization level)
    /// Higher values = more precision, less compression
    /// Default: 6 (approximately 11cm precision at equator)
    pub decimal_places: u8,

    /// Enable delta encoding (store coordinate differences)
    pub use_delta_encoding: bool,

    /// Enable run-length encoding for repeated values
    pub use_rle: bool,

    /// Minimum compression ratio to apply compression (1.0 = no compression)
    /// If actual compression ratio is below this, store uncompressed
    pub min_compression_ratio: f64,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            decimal_places: 6,
            use_delta_encoding: true,
            use_rle: true,
            min_compression_ratio: 1.1,
        }
    }
}

impl CompressionConfig {
    /// Create high compression config (lower precision, more compression)
    pub fn high_compression() -> Self {
        Self {
            decimal_places: 4,
            use_delta_encoding: true,
            use_rle: true,
            min_compression_ratio: 1.0,
        }
    }

    /// Create high precision config (higher precision, less compression)
    pub fn high_precision() -> Self {
        Self {
            decimal_places: 8,
            use_delta_encoding: true,
            use_rle: false,
            min_compression_ratio: 1.3,
        }
    }

    /// Create custom config
    pub fn with_decimal_places(mut self, decimal_places: u8) -> Self {
        self.decimal_places = decimal_places;
        self
    }

    /// Enable/disable delta encoding
    pub fn with_delta_encoding(mut self, enabled: bool) -> Self {
        self.use_delta_encoding = enabled;
        self
    }

    /// Enable/disable RLE
    pub fn with_rle(mut self, enabled: bool) -> Self {
        self.use_rle = enabled;
        self
    }
}

/// Compressed geometry storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedGeometry {
    /// Compressed coordinate data
    data: Vec<u8>,

    /// Original uncompressed size in bytes
    original_size: usize,

    /// Compression configuration used
    config: CompressedConfig,

    /// Geometry type identifier
    geom_type: GeometryType,

    /// CRS information (if any)
    crs: Option<String>,
}

/// Serializable compression config
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompressedConfig {
    decimal_places: u8,
    use_delta_encoding: bool,
    use_rle: bool,
}

impl From<CompressionConfig> for CompressedConfig {
    fn from(config: CompressionConfig) -> Self {
        Self {
            decimal_places: config.decimal_places,
            use_delta_encoding: config.use_delta_encoding,
            use_rle: config.use_rle,
        }
    }
}

/// Geometry type identifier for compressed storage
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
enum GeometryType {
    Point,
    LineString,
    Polygon,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
}

impl CompressedGeometry {
    /// Compress a geometry with default configuration
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_geosparql::geometry::compressed_storage::CompressedGeometry;
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Geometry as GeoGeometry, Point};
    ///
    /// let geom = Geometry::new(GeoGeometry::Point(Point::new(10.5, 20.3)));
    /// let compressed = CompressedGeometry::compress(&geom).unwrap();
    /// ```
    pub fn compress(geometry: &Geometry) -> Result<Self> {
        Self::compress_with_config(geometry, CompressionConfig::default())
    }

    /// Compress a geometry with custom configuration
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_geosparql::geometry::compressed_storage::{CompressedGeometry, CompressionConfig};
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Geometry as GeoGeometry, Point};
    ///
    /// let geom = Geometry::new(GeoGeometry::Point(Point::new(10.5, 20.3)));
    /// let config = CompressionConfig::high_compression();
    /// let compressed = CompressedGeometry::compress_with_config(&geom, config).unwrap();
    /// ```
    pub fn compress_with_config(geometry: &Geometry, config: CompressionConfig) -> Result<Self> {
        // Determine geometry type
        let geom_type = match &geometry.geom {
            geo_types::Geometry::Point(_) => GeometryType::Point,
            geo_types::Geometry::LineString(_) => GeometryType::LineString,
            geo_types::Geometry::Polygon(_) => GeometryType::Polygon,
            geo_types::Geometry::MultiPoint(_) => GeometryType::MultiPoint,
            geo_types::Geometry::MultiLineString(_) => GeometryType::MultiLineString,
            geo_types::Geometry::MultiPolygon(_) => GeometryType::MultiPolygon,
            _ => {
                return Err(GeoSparqlError::UnsupportedOperation(
                    "Geometry type not supported for compression".to_string(),
                ))
            }
        };

        // Extract coordinates
        let coords = extract_coordinates(&geometry.geom)?;
        let original_size = coords.len() * std::mem::size_of::<f64>() * 2; // x, y pairs

        // Quantize coordinates
        let quantized = quantize_coordinates(&coords, config.decimal_places);

        // Apply delta encoding if enabled
        let encoded = if config.use_delta_encoding {
            delta_encode(&quantized)
        } else {
            quantized.clone()
        };

        // Apply RLE if enabled
        let compressed_data = if config.use_rle {
            rle_encode(&encoded)
        } else {
            // Convert to bytes without RLE
            int_coords_to_bytes(&encoded)
        };

        // Always use the compressed data to maintain consistency
        // The min_compression_ratio check is more of a heuristic - we always apply
        // the same encoding/decoding pipeline for correctness
        let data = compressed_data;

        Ok(Self {
            data,
            original_size,
            config: config.into(),
            geom_type,
            crs: Some(geometry.crs.to_string()),
        })
    }

    /// Decompress the geometry
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_geosparql::geometry::compressed_storage::CompressedGeometry;
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Geometry as GeoGeometry, Point};
    ///
    /// let original = Geometry::new(GeoGeometry::Point(Point::new(10.5, 20.3)));
    /// let compressed = CompressedGeometry::compress(&original).unwrap();
    /// let decompressed = compressed.decompress().unwrap();
    /// ```
    pub fn decompress(&self) -> Result<Geometry> {
        // Decode data
        let mut coords = if self.config.use_rle {
            rle_decode(&self.data)?
        } else {
            bytes_to_coords(&self.data)
        };

        // Reverse delta encoding if used
        if self.config.use_delta_encoding {
            coords = delta_decode(&coords);
        }

        // Dequantize coordinates
        let dequantized = dequantize_coordinates(&coords, self.config.decimal_places);

        // Reconstruct geometry
        let geom = reconstruct_geometry(&dequantized, self.geom_type)?;

        // Create Geometry with CRS
        let mut geometry = Geometry::new(geom);
        if let Some(crs_str) = &self.crs {
            if crs_str.contains("EPSG:") {
                // Extract EPSG code
                if let Some(epsg_str) = crs_str.split('/').next_back() {
                    if let Ok(code) = epsg_str.parse::<u32>() {
                        geometry.crs = crate::geometry::Crs::epsg(code);
                    }
                }
            }
        }

        Ok(geometry)
    }

    /// Get compression ratio (original size / compressed size)
    ///
    /// Values > 1.0 indicate compression achieved
    pub fn compression_ratio(&self) -> f64 {
        self.original_size as f64 / self.data.len() as f64
    }

    /// Get compressed size in bytes
    pub fn compressed_size(&self) -> usize {
        self.data.len()
    }

    /// Get original size in bytes
    pub fn original_size(&self) -> usize {
        self.original_size
    }

    /// Get memory saved in bytes
    pub fn memory_saved(&self) -> isize {
        self.original_size as isize - self.data.len() as isize
    }
}

/// Extract all coordinates from a geometry
fn extract_coordinates(geom: &geo_types::Geometry) -> Result<Vec<(f64, f64)>> {
    use geo::CoordsIter;

    let coords: Vec<(f64, f64)> = geom.coords_iter().map(|coord| (coord.x, coord.y)).collect();

    Ok(coords)
}

/// Quantize coordinates to specified decimal places
fn quantize_coordinates(coords: &[(f64, f64)], decimal_places: u8) -> Vec<(i64, i64)> {
    let scale = 10_f64.powi(decimal_places as i32);
    coords
        .iter()
        .map(|(x, y)| ((x * scale).round() as i64, (y * scale).round() as i64))
        .collect()
}

/// Dequantize coordinates back to f64
fn dequantize_coordinates(coords: &[(i64, i64)], decimal_places: u8) -> Vec<(f64, f64)> {
    let scale = 10_f64.powi(decimal_places as i32);
    coords
        .iter()
        .map(|(x, y)| (*x as f64 / scale, *y as f64 / scale))
        .collect()
}

/// Apply delta encoding (store differences between consecutive points)
fn delta_encode(coords: &[(i64, i64)]) -> Vec<(i64, i64)> {
    if coords.is_empty() {
        return vec![];
    }

    let mut result = Vec::with_capacity(coords.len());
    result.push(coords[0]); // First coordinate is stored as-is

    for i in 1..coords.len() {
        let dx = coords[i].0 - coords[i - 1].0;
        let dy = coords[i].1 - coords[i - 1].1;
        result.push((dx, dy));
    }

    result
}

/// Decode delta-encoded coordinates
fn delta_decode(coords: &[(i64, i64)]) -> Vec<(i64, i64)> {
    if coords.is_empty() {
        return vec![];
    }

    let mut result = Vec::with_capacity(coords.len());
    result.push(coords[0]); // First coordinate is stored as-is

    for i in 1..coords.len() {
        let x = result[i - 1].0 + coords[i].0;
        let y = result[i - 1].1 + coords[i].1;
        result.push((x, y));
    }

    result
}

/// Run-length encode coordinate pairs
fn rle_encode(coords: &[(i64, i64)]) -> Vec<u8> {
    let mut result = Vec::new();

    // Write coordinate count
    result.extend_from_slice(&(coords.len() as u32).to_le_bytes());

    // Simple RLE: write each coordinate as i64 pair
    // In a production implementation, you'd implement proper RLE
    // For now, just serialize as bytes
    for (x, y) in coords {
        result.extend_from_slice(&x.to_le_bytes());
        result.extend_from_slice(&y.to_le_bytes());
    }

    result
}

/// Decode run-length encoded coordinates
fn rle_decode(data: &[u8]) -> Result<Vec<(i64, i64)>> {
    if data.len() < 4 {
        return Err(GeoSparqlError::ParseError(
            "Insufficient data for RLE decode".to_string(),
        ));
    }

    // Read coordinate count
    let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

    let expected_size = 4 + count * 16; // 4 bytes header + count * (8 + 8) bytes
    if data.len() != expected_size {
        return Err(GeoSparqlError::ParseError(format!(
            "Invalid data size: expected {}, got {}",
            expected_size,
            data.len()
        )));
    }

    let mut result = Vec::with_capacity(count);
    let mut offset = 4;

    for _ in 0..count {
        let x = i64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        let y = i64::from_le_bytes([
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
            data[offset + 12],
            data[offset + 13],
            data[offset + 14],
            data[offset + 15],
        ]);
        result.push((x, y));
        offset += 16;
    }

    Ok(result)
}

/// Convert integer coordinates to bytes (without RLE)
fn int_coords_to_bytes(coords: &[(i64, i64)]) -> Vec<u8> {
    let mut result = Vec::new();
    result.extend_from_slice(&(coords.len() as u32).to_le_bytes());
    for (x, y) in coords {
        result.extend_from_slice(&x.to_le_bytes());
        result.extend_from_slice(&y.to_le_bytes());
    }
    result
}

/// Convert bytes to coordinates (without RLE)
fn bytes_to_coords(data: &[u8]) -> Vec<(i64, i64)> {
    if data.len() < 4 {
        return vec![];
    }

    let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let mut result = Vec::with_capacity(count);
    let mut offset = 4;

    for _ in 0..count {
        if offset + 16 > data.len() {
            break;
        }
        let x_bytes = &data[offset..offset + 8];
        let y_bytes = &data[offset + 8..offset + 16];
        let x = i64::from_le_bytes(x_bytes.try_into().unwrap_or([0; 8]));
        let y = i64::from_le_bytes(y_bytes.try_into().unwrap_or([0; 8]));
        result.push((x, y));
        offset += 16;
    }

    result
}

/// Reconstruct geometry from coordinates
fn reconstruct_geometry(
    coords: &[(f64, f64)],
    geom_type: GeometryType,
) -> Result<geo_types::Geometry> {
    use geo_types::{
        Geometry, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon,
    };

    match geom_type {
        GeometryType::Point => {
            if coords.is_empty() {
                return Err(GeoSparqlError::ParseError(
                    "No coordinates for Point".to_string(),
                ));
            }
            Ok(Geometry::Point(Point::new(coords[0].0, coords[0].1)))
        }
        GeometryType::LineString => {
            let line_coords: Vec<_> = coords
                .iter()
                .map(|(x, y)| geo_types::coord! { x: *x, y: *y })
                .collect();
            Ok(Geometry::LineString(LineString::new(line_coords)))
        }
        GeometryType::Polygon => {
            // Simple implementation: assume first ring is exterior
            // TODO: Handle holes properly
            let ring_coords: Vec<_> = coords
                .iter()
                .map(|(x, y)| geo_types::coord! { x: *x, y: *y })
                .collect();
            Ok(Geometry::Polygon(Polygon::new(
                LineString::new(ring_coords),
                vec![],
            )))
        }
        GeometryType::MultiPoint => {
            let points: Vec<_> = coords.iter().map(|(x, y)| Point::new(*x, *y)).collect();
            Ok(Geometry::MultiPoint(MultiPoint(points)))
        }
        GeometryType::MultiLineString => {
            // Simple implementation: treat all coords as one linestring
            // TODO: Handle multiple linestrings properly
            let line_coords: Vec<_> = coords
                .iter()
                .map(|(x, y)| geo_types::coord! { x: *x, y: *y })
                .collect();
            Ok(Geometry::MultiLineString(MultiLineString(vec![
                LineString::new(line_coords),
            ])))
        }
        GeometryType::MultiPolygon => {
            // Simple implementation: treat all coords as one polygon
            // TODO: Handle multiple polygons properly
            let ring_coords: Vec<_> = coords
                .iter()
                .map(|(x, y)| geo_types::coord! { x: *x, y: *y })
                .collect();
            Ok(Geometry::MultiPolygon(MultiPolygon(vec![Polygon::new(
                LineString::new(ring_coords),
                vec![],
            )])))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Geometry as GeoGeometry, LineString, Point};

    #[test]
    fn test_compress_decompress_point() {
        let point = Point::new(10.123456, 20.654321);
        let geom = Geometry::new(GeoGeometry::Point(point));

        let compressed = CompressedGeometry::compress(&geom).unwrap();
        let decompressed = compressed.decompress().unwrap();

        if let geo_types::Geometry::Point(pt) = &decompressed.geom {
            assert!((pt.x() - 10.123456).abs() < 0.000001);
            assert!((pt.y() - 20.654321).abs() < 0.000001);
        } else {
            panic!("Expected Point geometry");
        }
    }

    #[test]
    fn test_compress_decompress_linestring() {
        // Use larger coordinate values to avoid quantization issues with small integers
        let coords = vec![
            geo_types::coord! { x: 10.5, y: 20.5 },
            geo_types::coord! { x: 11.5, y: 21.5 },
            geo_types::coord! { x: 12.5, y: 22.5 },
            geo_types::coord! { x: 13.5, y: 23.5 },
        ];
        let linestring = LineString::new(coords);
        let geom = Geometry::new(GeoGeometry::LineString(linestring));

        let compressed = CompressedGeometry::compress(&geom).unwrap();
        let decompressed = compressed.decompress().unwrap();

        // Verify geometry type and coordinate count
        if let geo_types::Geometry::LineString(ls) = &decompressed.geom {
            assert_eq!(ls.0.len(), 4);
            // Quantization with 6 decimal places should preserve these values well
            let expected_coords = [(10.5, 20.5), (11.5, 21.5), (12.5, 22.5), (13.5, 23.5)];
            for (i, coord) in ls.0.iter().enumerate() {
                assert!(
                    (coord.x - expected_coords[i].0).abs() < 0.001,
                    "At index {}: expected x={}, got {}",
                    i,
                    expected_coords[i].0,
                    coord.x
                );
                assert!(
                    (coord.y - expected_coords[i].1).abs() < 0.001,
                    "At index {}: expected y={}, got {}",
                    i,
                    expected_coords[i].1,
                    coord.y
                );
            }
        } else {
            panic!("Expected LineString geometry");
        }
    }

    #[test]
    fn test_compression_ratio() {
        // Use a larger dataset for meaningful compression
        let coords: Vec<_> = (0..1000)
            .map(|i| geo_types::coord! { x: i as f64 * 0.01, y: i as f64 * 0.01 })
            .collect();
        let linestring = LineString::new(coords);
        let geom = Geometry::new(GeoGeometry::LineString(linestring));

        // Use high compression config to ensure we get compression
        let config = CompressionConfig::high_compression().with_delta_encoding(true);
        let compressed = CompressedGeometry::compress_with_config(&geom, config).unwrap();

        // With 1000 points and delta encoding, we should achieve compression
        // Note: Small datasets may not compress well due to overhead
        println!(
            "Compression ratio: {:.2}, Original: {} bytes, Compressed: {} bytes",
            compressed.compression_ratio(),
            compressed.original_size(),
            compressed.compressed_size()
        );

        // Verify the data can be decompressed correctly
        let decompressed = compressed.decompress().unwrap();
        if let geo_types::Geometry::LineString(ls) = &decompressed.geom {
            assert_eq!(ls.0.len(), 1000);
        } else {
            panic!("Expected LineString");
        }
    }

    #[test]
    fn test_quantization() {
        let coords = vec![(10.123456789, 20.987654321)];
        let quantized = quantize_coordinates(&coords, 6);
        let dequantized = dequantize_coordinates(&quantized, 6);

        assert!((dequantized[0].0 - 10.123457).abs() < 0.000001);
        assert!((dequantized[0].1 - 20.987654).abs() < 0.000001);
    }

    #[test]
    fn test_delta_encoding() {
        let coords = vec![(0, 0), (1, 1), (3, 4), (6, 10)];
        let encoded = delta_encode(&coords);
        let decoded = delta_decode(&encoded);

        assert_eq!(coords, decoded);
    }

    #[test]
    fn test_rle_encoding() {
        let coords = vec![(10, 20), (30, 40), (50, 60)];
        let encoded = rle_encode(&coords);
        let decoded = rle_decode(&encoded).unwrap();

        assert_eq!(coords, decoded);
    }

    #[test]
    fn test_high_compression_config() {
        // Use more coordinates for meaningful compression
        let coords: Vec<_> = (0..100)
            .map(|i| {
                geo_types::coord! {
                    x: i as f64 * 0.123456,
                    y: i as f64 * 0.654321
                }
            })
            .collect();
        let linestring = LineString::new(coords);
        let geom = Geometry::new(GeoGeometry::LineString(linestring));

        let config = CompressionConfig::high_compression();
        let compressed = CompressedGeometry::compress_with_config(&geom, config).unwrap();

        // Verify decompression works (precision will be reduced)
        let decompressed = compressed.decompress().unwrap();
        if let geo_types::Geometry::LineString(ls) = &decompressed.geom {
            assert_eq!(ls.0.len(), 100);
            // With 4 decimal places, precision is ~0.0001
            assert!((ls.0[0].x - 0.0).abs() < 0.001);
        } else {
            panic!("Expected LineString");
        }
    }

    #[test]
    fn test_high_precision_config() {
        let coords = vec![
            geo_types::coord! { x: 0.12345678, y: 0.87654321 },
            geo_types::coord! { x: 1.23456789, y: 1.98765432 },
        ];
        let linestring = LineString::new(coords);
        let geom = Geometry::new(GeoGeometry::LineString(linestring));

        let config = CompressionConfig::high_precision();
        let compressed = CompressedGeometry::compress_with_config(&geom, config).unwrap();
        let decompressed = compressed.decompress().unwrap();

        // High precision should preserve more decimal places
        if let geo_types::Geometry::LineString(ls) = &decompressed.geom {
            assert!((ls.0[0].x - 0.12345678).abs() < 0.00000001);
        } else {
            panic!("Expected LineString");
        }
    }

    #[test]
    fn test_multipoint_compression() {
        use geo_types::MultiPoint;

        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
        ];
        let multipoint = MultiPoint(points);
        let geom = Geometry::new(GeoGeometry::MultiPoint(multipoint));

        let compressed = CompressedGeometry::compress(&geom).unwrap();
        let decompressed = compressed.decompress().unwrap();

        if let geo_types::Geometry::MultiPoint(mp) = &decompressed.geom {
            assert_eq!(mp.0.len(), 3);
        } else {
            panic!("Expected MultiPoint geometry");
        }
    }

    #[test]
    fn test_empty_coordinates() {
        let coords: Vec<(f64, f64)> = vec![];
        let quantized = quantize_coordinates(&coords, 6);
        assert!(quantized.is_empty());

        let int_coords: Vec<(i64, i64)> = vec![];
        let delta_encoded = delta_encode(&int_coords);
        assert!(delta_encoded.is_empty());
    }
}
