//! Streaming geometry processing for memory-efficient large dataset handling
//!
//! Provides lazy evaluation and streaming operations for processing large geometry
//! collections without loading everything into memory at once.
//!
//! # Features
//!
//! - **Lazy evaluation**: Process geometries on-demand
//! - **Memory-efficient**: Stream through large datasets
//! - **Composable operations**: Chain transformations together
//! - **Parallel streaming**: Leverage multiple cores for processing
//! - **Backpressure handling**: Control memory usage during streaming
//!
//! # Example
//!
//! ```rust
//! use oxirs_geosparql::geometry::streaming::{GeometryStream, StreamProcessor};
//! use oxirs_geosparql::geometry::Geometry;
//! use geo_types::{Geometry as GeoGeometry, Point};
//!
//! // Create a stream from a vector
//! let geometries = vec![
//!     Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))),
//!     Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))),
//! ];
//!
//! let stream = GeometryStream::from_vec(geometries);
//!
//! // Process with transformations
//! let processor = StreamProcessor::new()
//!     .filter(|g| {
//!         if let geo_types::Geometry::Point(pt) = &g.geom {
//!             pt.x() > 0.0
//!         } else {
//!             false
//!         }
//!     })
//!     .buffer(100);
//!
//! let results: Vec<_> = processor.process(stream).filter_map(|r| r.ok()).collect();
//! assert_eq!(results.len(), 1);
//! ```

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Type alias for filter predicates
type FilterPredicate = Arc<dyn Fn(&Geometry) -> bool + Send + Sync>;

/// Type alias for transformation functions
type TransformFunction = Arc<dyn Fn(Geometry) -> Result<Geometry> + Send + Sync>;

/// A streaming source of geometries
pub struct GeometryStream<I>
where
    I: Iterator<Item = Geometry>,
{
    source: I,
    buffer_size: usize,
}

impl<I> GeometryStream<I>
where
    I: Iterator<Item = Geometry>,
{
    /// Create a stream from an iterator
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_geosparql::geometry::streaming::GeometryStream;
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Geometry as GeoGeometry, Point};
    ///
    /// let geometries = vec![
    ///     Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))),
    /// ];
    /// let stream = GeometryStream::from_iterator(geometries.into_iter());
    /// ```
    pub fn from_iterator(iter: I) -> Self {
        Self {
            source: iter,
            buffer_size: 1000, // Default buffer size
        }
    }

    /// Set the buffer size for batched operations
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Consume the stream and return the inner iterator
    pub fn into_inner(self) -> I {
        self.source
    }

    /// Collect buffered chunks of geometries
    pub fn chunks(self, chunk_size: usize) -> ChunkedStream<I> {
        ChunkedStream {
            source: self.source,
            chunk_size,
        }
    }
}

impl GeometryStream<std::vec::IntoIter<Geometry>> {
    /// Create a stream from a vector
    pub fn from_vec(geometries: Vec<Geometry>) -> Self {
        Self::from_iterator(geometries.into_iter())
    }
}

/// A chunked stream that yields batches of geometries
pub struct ChunkedStream<I>
where
    I: Iterator<Item = Geometry>,
{
    source: I,
    chunk_size: usize,
}

impl<I> Iterator for ChunkedStream<I>
where
    I: Iterator<Item = Geometry>,
{
    type Item = Vec<Geometry>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut chunk = Vec::with_capacity(self.chunk_size);

        for _ in 0..self.chunk_size {
            match self.source.next() {
                Some(geom) => chunk.push(geom),
                None => break,
            }
        }

        if chunk.is_empty() {
            None
        } else {
            Some(chunk)
        }
    }
}

/// Stream processor for applying operations to geometry streams
pub struct StreamProcessor {
    filters: Vec<FilterPredicate>,
    transforms: Vec<TransformFunction>,
    buffer_size: usize,
}

impl Default for StreamProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamProcessor {
    /// Create a new stream processor
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
            transforms: Vec::new(),
            buffer_size: 1000,
        }
    }

    /// Add a filter predicate
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_geosparql::geometry::streaming::StreamProcessor;
    ///
    /// let processor = StreamProcessor::new()
    ///     .filter(|g| {
    ///         if let geo_types::Geometry::Point(pt) = &g.geom {
    ///             pt.x() > 0.0
    ///         } else {
    ///             true
    ///         }
    ///     });
    /// ```
    pub fn filter<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&Geometry) -> bool + Send + Sync + 'static,
    {
        self.filters.push(Arc::new(predicate));
        self
    }

    /// Add a transformation function
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_geosparql::geometry::streaming::StreamProcessor;
    /// use oxirs_geosparql::geometry::Geometry;
    ///
    /// let processor = StreamProcessor::new()
    ///     .transform(|mut g| {
    ///         // Transform the geometry (e.g., apply CRS transformation)
    ///         Ok(g)
    ///     });
    /// ```
    pub fn transform<F>(mut self, transform: F) -> Self
    where
        F: Fn(Geometry) -> Result<Geometry> + Send + Sync + 'static,
    {
        self.transforms.push(Arc::new(transform));
        self
    }

    /// Set buffer size for batched operations
    pub fn buffer(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Process a geometry stream
    pub fn process<I>(self, stream: GeometryStream<I>) -> ProcessedStream<I>
    where
        I: Iterator<Item = Geometry>,
    {
        ProcessedStream {
            source: stream.source,
            filters: self.filters,
            transforms: self.transforms,
        }
    }

    /// Process a geometry stream in parallel (requires 'parallel' feature)
    #[cfg(feature = "parallel")]
    pub fn process_parallel<I>(
        self,
        stream: GeometryStream<I>,
    ) -> impl Iterator<Item = Result<Geometry>>
    where
        I: Iterator<Item = Geometry> + Send,
    {
        let filters = self.filters;
        let transforms = self.transforms;

        stream.chunks(self.buffer_size).flat_map(move |chunk| {
            let filters = filters.clone();
            let transforms = transforms.clone();

            chunk
                .into_par_iter()
                .filter(move |geom| filters.iter().all(|f| f(geom)))
                .map(move |mut geom| {
                    for transform in &transforms {
                        geom = transform(geom)?;
                    }
                    Ok(geom)
                })
                .collect::<Vec<_>>()
                .into_iter()
        })
    }
}

/// A processed geometry stream
pub struct ProcessedStream<I>
where
    I: Iterator<Item = Geometry>,
{
    source: I,
    filters: Vec<FilterPredicate>,
    transforms: Vec<TransformFunction>,
}

impl<I> Iterator for ProcessedStream<I>
where
    I: Iterator<Item = Geometry>,
{
    type Item = Result<Geometry>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let geom = self.source.next()?;

            // Apply filters
            if !self.filters.iter().all(|f| f(&geom)) {
                continue;
            }

            // Apply transformations
            let mut result = geom;
            for transform in &self.transforms {
                match transform(result) {
                    Ok(g) => result = g,
                    Err(e) => return Some(Err(e)),
                }
            }

            return Some(Ok(result));
        }
    }
}

/// Lazy geometry reader for file-based streaming
pub struct LazyGeometryReader {
    #[allow(dead_code)] // Will be used when file streaming is implemented
    file_path: String,
    format: GeometryFormat,
    buffer_size: usize,
}

/// Supported geometry formats for streaming
#[derive(Debug, Clone, Copy)]
pub enum GeometryFormat {
    /// WKT format
    Wkt,
    /// GeoJSON format
    #[cfg(feature = "geojson-support")]
    GeoJson,
    /// Shapefile format
    #[cfg(feature = "shapefile-support")]
    Shapefile,
    /// FlatGeobuf format
    #[cfg(feature = "flatgeobuf-support")]
    FlatGeobuf,
}

impl LazyGeometryReader {
    /// Create a new lazy reader
    pub fn new(file_path: impl Into<String>, format: GeometryFormat) -> Self {
        Self {
            file_path: file_path.into(),
            format,
            buffer_size: 1000,
        }
    }

    /// Set buffer size
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Create a geometry stream from the file
    ///
    /// This is a placeholder - actual implementation would depend on the format
    pub fn stream(self) -> Result<GeometryStream<Box<dyn Iterator<Item = Geometry>>>> {
        match self.format {
            GeometryFormat::Wkt => Err(GeoSparqlError::UnsupportedOperation(
                "WKT streaming not yet implemented".to_string(),
            )),
            #[cfg(feature = "geojson-support")]
            GeometryFormat::GeoJson => Err(GeoSparqlError::UnsupportedOperation(
                "GeoJSON streaming not yet implemented".to_string(),
            )),
            #[cfg(feature = "shapefile-support")]
            GeometryFormat::Shapefile => Err(GeoSparqlError::UnsupportedOperation(
                "Shapefile streaming not yet implemented".to_string(),
            )),
            #[cfg(feature = "flatgeobuf-support")]
            GeometryFormat::FlatGeobuf => Err(GeoSparqlError::UnsupportedOperation(
                "FlatGeobuf streaming not yet implemented".to_string(),
            )),
        }
    }
}

/// Statistics collector for streaming operations
#[derive(Debug, Clone, Default)]
pub struct StreamStatistics {
    /// Total geometries processed
    pub total: usize,
    /// Number of geometries filtered out
    pub filtered: usize,
    /// Number of geometries that passed filters
    pub passed: usize,
    /// Number of transformation errors
    pub errors: usize,
}

impl StreamStatistics {
    /// Create new statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a processed geometry
    pub fn record_processed(&mut self) {
        self.total += 1;
        self.passed += 1;
    }

    /// Record a filtered geometry
    pub fn record_filtered(&mut self) {
        self.total += 1;
        self.filtered += 1;
    }

    /// Record an error
    pub fn record_error(&mut self) {
        self.errors += 1;
    }

    /// Get pass rate (0.0 to 1.0)
    pub fn pass_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.passed as f64 / self.total as f64
        }
    }

    /// Get error rate (0.0 to 1.0)
    pub fn error_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.errors as f64 / self.total as f64
        }
    }
}

/// A stream processor with statistics tracking
pub struct MonitoredStreamProcessor {
    processor: StreamProcessor,
    stats: Arc<parking_lot::Mutex<StreamStatistics>>,
}

impl MonitoredStreamProcessor {
    /// Create a new monitored processor
    pub fn new(processor: StreamProcessor) -> Self {
        Self {
            processor,
            stats: Arc::new(parking_lot::Mutex::new(StreamStatistics::new())),
        }
    }

    /// Process a stream with statistics tracking
    ///
    /// Returns the processed stream and a handle to get statistics
    pub fn process<I>(
        self,
        stream: GeometryStream<I>,
    ) -> (
        MonitoredProcessedStream<I>,
        Arc<parking_lot::Mutex<StreamStatistics>>,
    )
    where
        I: Iterator<Item = Geometry>,
    {
        let stats = self.stats.clone();
        let processed = self.processor.process(stream);
        (
            MonitoredProcessedStream {
                stream: processed,
                stats: self.stats,
            },
            stats,
        )
    }

    /// Get current statistics
    pub fn statistics(&self) -> StreamStatistics {
        self.stats.lock().clone()
    }
}

/// A processed stream with statistics tracking
pub struct MonitoredProcessedStream<I>
where
    I: Iterator<Item = Geometry>,
{
    stream: ProcessedStream<I>,
    stats: Arc<parking_lot::Mutex<StreamStatistics>>,
}

impl<I> Iterator for MonitoredProcessedStream<I>
where
    I: Iterator<Item = Geometry>,
{
    type Item = Result<Geometry>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.stream.next() {
            Some(Ok(geom)) => {
                self.stats.lock().record_processed();
                Some(Ok(geom))
            }
            Some(Err(e)) => {
                self.stats.lock().record_error();
                Some(Err(e))
            }
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Geometry as GeoGeometry, Point};

    #[test]
    fn test_geometry_stream_from_vec() {
        let geometries = vec![
            Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))),
            Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))),
            Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0))),
        ];

        let stream = GeometryStream::from_vec(geometries);
        let collected: Vec<_> = stream.into_inner().collect();
        assert_eq!(collected.len(), 3);
    }

    #[test]
    fn test_chunked_stream() {
        let geometries = vec![
            Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))),
            Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))),
            Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0))),
            Geometry::new(GeoGeometry::Point(Point::new(3.0, 3.0))),
            Geometry::new(GeoGeometry::Point(Point::new(4.0, 4.0))),
        ];

        let stream = GeometryStream::from_vec(geometries);
        let chunks: Vec<_> = stream.chunks(2).collect();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 2);
        assert_eq!(chunks[1].len(), 2);
        assert_eq!(chunks[2].len(), 1);
    }

    #[test]
    fn test_stream_processor_filter() {
        let geometries = vec![
            Geometry::new(GeoGeometry::Point(Point::new(-1.0, 0.0))),
            Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))),
            Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0))),
        ];

        let stream = GeometryStream::from_vec(geometries);
        let processor = StreamProcessor::new().filter(|g| {
            if let geo_types::Geometry::Point(pt) = &g.geom {
                pt.x() > 0.0
            } else {
                false
            }
        });

        let results: Vec<_> = processor.process(stream).collect();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[test]
    fn test_stream_processor_transform() {
        let geometries = vec![
            Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))),
            Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0))),
        ];

        let stream = GeometryStream::from_vec(geometries);
        let processor = StreamProcessor::new().transform(|g| {
            // Simple identity transform
            Ok(g)
        });

        let results: Vec<_> = processor.process(stream).collect();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[test]
    fn test_stream_statistics() {
        let mut stats = StreamStatistics::new();

        stats.record_processed();
        stats.record_processed();
        stats.record_filtered();
        stats.record_error();

        assert_eq!(stats.total, 3);
        assert_eq!(stats.passed, 2);
        assert_eq!(stats.filtered, 1);
        assert_eq!(stats.errors, 1);

        assert!((stats.pass_rate() - 0.666).abs() < 0.01);
        assert!((stats.error_rate() - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_monitored_stream_processor() {
        let geometries = vec![
            Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))),
            Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0))),
            Geometry::new(GeoGeometry::Point(Point::new(3.0, 3.0))),
        ];

        let stream = GeometryStream::from_vec(geometries);
        let processor = StreamProcessor::new();
        let monitored = MonitoredStreamProcessor::new(processor);

        let (processed, stats_ref) = monitored.process(stream);
        let results: Vec<_> = processed.collect();

        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_ok()));

        // Verify statistics
        let stats = stats_ref.lock();
        assert_eq!(stats.passed, 3);
    }

    #[test]
    fn test_buffer_size_configuration() {
        let geometries = vec![Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0)))];

        let stream = GeometryStream::from_vec(geometries).with_buffer_size(500);
        assert_eq!(stream.buffer_size, 500);
    }

    #[test]
    fn test_processor_chaining() {
        let geometries = vec![
            Geometry::new(GeoGeometry::Point(Point::new(-1.0, 0.0))),
            Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))),
            Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0))),
            Geometry::new(GeoGeometry::Point(Point::new(3.0, 3.0))),
        ];

        let stream = GeometryStream::from_vec(geometries);
        let processor = StreamProcessor::new()
            .filter(|g| {
                if let geo_types::Geometry::Point(pt) = &g.geom {
                    pt.x() > 0.0
                } else {
                    false
                }
            })
            .filter(|g| {
                if let geo_types::Geometry::Point(pt) = &g.geom {
                    pt.x() < 3.0
                } else {
                    false
                }
            })
            .buffer(100);

        let results: Vec<_> = processor.process(stream).collect();
        assert_eq!(results.len(), 2); // Only 1.0 and 2.0 pass both filters
    }

    #[test]
    fn test_empty_stream() {
        let geometries: Vec<Geometry> = vec![];
        let stream = GeometryStream::from_vec(geometries);
        let processor = StreamProcessor::new();

        let results: Vec<_> = processor.process(stream).collect();
        assert_eq!(results.len(), 0);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_processing() {
        let geometries: Vec<_> = (0..100)
            .map(|i| Geometry::new(GeoGeometry::Point(Point::new(i as f64, i as f64))))
            .collect();

        let stream = GeometryStream::from_vec(geometries);
        let processor = StreamProcessor::new()
            .filter(|g| {
                if let geo_types::Geometry::Point(pt) = &g.geom {
                    pt.x() < 50.0
                } else {
                    false
                }
            })
            .buffer(10);

        let results: Vec<_> = processor.process_parallel(stream).collect();
        assert_eq!(results.len(), 50);
        assert!(results.iter().all(|r| r.is_ok()));
    }
}
