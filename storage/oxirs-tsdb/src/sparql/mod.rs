//! SPARQL temporal extensions for time-series queries
//!
//! This module provides custom SPARQL functions for time-series operations.
//!
//! ## Temporal Functions
//!
//! ### Window Aggregations
//!
//! ```sparql
//! PREFIX ts: <http://oxirs.org/ts#>
//!
//! SELECT ?sensor (ts:window(?value, 600, "AVG") AS ?avg)
//! WHERE {
//!   ?sensor :value ?value ;
//!           :timestamp ?time .
//! }
//! ```
//!
//! ### Resampling
//!
//! ```sparql
//! PREFIX ts: <http://oxirs.org/ts#>
//!
//! SELECT (ts:resample(?timestamp, "1h") AS ?hour) (AVG(?value) AS ?avg)
//! WHERE {
//!   ?sensor :value ?value ;
//!           :timestamp ?timestamp .
//! }
//! GROUP BY (ts:resample(?timestamp, "1h"))
//! ```
//!
//! ### Interpolation
//!
//! ```sparql
//! PREFIX ts: <http://oxirs.org/ts#>
//!
//! SELECT ?sensor (ts:interpolate(?timestamp, ?value, "linear") AS ?interpolated)
//! WHERE {
//!   ?sensor :value ?value ;
//!           :timestamp ?timestamp .
//! }
//! ```

pub mod extensions;
pub mod router;

pub use extensions::{
    interpolate_function, register_temporal_functions, resample_function, window_function,
    TemporalFunctionRegistry, TemporalValue,
};
pub use router::{QueryRouter, RoutingDecision};
