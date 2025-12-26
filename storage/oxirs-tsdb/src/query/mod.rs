//! Time-series query engine
//!
//! Provides efficient query operations over time-series data:
//!
//! - **Range queries** - Query data within time bounds
//! - **Aggregations** - AVG, MIN, MAX, SUM, COUNT
//! - **Window functions** - Moving averages, rolling min/max
//! - **Resampling** - Time bucketing and downsampling
//! - **Interpolation** - Fill missing values
//!
//! # Example
//!
//! ```
//! use oxirs_tsdb::{QueryEngine, Aggregation, TimeChunk, DataPoint};
//! use chrono::{Utc, Duration};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create query engine
//! let mut engine = QueryEngine::new();
//!
//! // Add sample data
//! let start_time = Utc::now();
//! let points = vec![DataPoint::new(start_time, 22.5)];
//! let chunk = TimeChunk::new(1, start_time, Duration::hours(2), points)?;
//! engine.add_chunk(chunk);
//!
//! // Query with aggregation
//! let start = start_time - Duration::hours(1);
//! let end = start_time + Duration::hours(1);
//! let result = engine.query()
//!     .series(1)
//!     .time_range(start, end)
//!     .aggregate(Aggregation::Avg)
//!     .execute()?;
//! # Ok(())
//! # }
//! ```

pub mod aggregate;
pub mod engine;
pub mod interpolate;
pub mod range;
pub mod resample;
pub mod window;

// Re-exports
pub use aggregate::{Aggregation, AggregationResult};
pub use engine::{QueryBuilder, QueryEngine, QueryResult};
pub use interpolate::{InterpolateMethod, Interpolator};
pub use range::{RangeQuery, TimeRange};
pub use resample::{ResampleBucket, Resampler};
pub use window::{WindowFunction, WindowSpec};
