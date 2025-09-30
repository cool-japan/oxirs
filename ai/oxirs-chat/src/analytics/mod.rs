//! Analytics module for oxirs-chat
//!
//! This module provides comprehensive analytics, insights, pattern recognition,
//! and conversation intelligence to help understand user behavior, optimize
//! responses, and improve the chat experience.

pub mod anomaly_detector;
pub mod pattern_detector;
pub mod types;

pub use anomaly_detector::*;
pub use pattern_detector::*;
pub use types::*;
