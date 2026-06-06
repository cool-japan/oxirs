//! # Visual Stream Designer and Debugger
//!
//! Comprehensive visual tool for designing, debugging, and optimizing stream processing pipelines.
//! Provides a graph-based interface for building complex stream processing flows with real-time
//! debugging, performance profiling, and automatic optimization suggestions.
//!
//! ## Features
//! - Visual pipeline designer with drag-and-drop interface
//! - Real-time debugging with event visualization
//! - Performance profiling and bottleneck detection
//! - Automatic pipeline validation and optimization
//! - Export/import pipeline definitions (JSON, YAML, DOT)
//! - Integration with existing stream processing operators
//! - Live monitoring and metrics dashboard
//! - Breakpoint support for debugging
//! - Time-travel debugging for historical analysis
//!
//! ## Module layout
//! - `visual_designer_types`   — all data types (structs, enums, config)
//! - `visual_designer_engine`  — `VisualStreamDesigner`, `PipelineValidator`, `PipelineOptimizer`
//! - `visual_designer_tests`   — integration tests

// Re-export everything from sibling modules so downstream code keeps working.
pub use crate::visual_designer_engine::*;
pub use crate::visual_designer_types::*;
