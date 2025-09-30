//! Predictive Analytics with Forecasting Models and Recommendation Systems
//!
//! This module implements advanced predictive analytics capabilities including
//! time series forecasting, trend prediction, and intelligent recommendation systems.

pub mod config;
pub mod core;
pub mod forecasting;
pub mod recommendation;
pub mod time_series;
pub mod types;

// Re-export main public interface
pub use config::*;
pub use core::PredictiveAnalyticsEngine;
pub use forecasting::ForecastingModels;
pub use recommendation::RecommendationEngine;
pub use time_series::{TimeSeriesProcessor, TrendAnalyzer};
pub use types::*;
