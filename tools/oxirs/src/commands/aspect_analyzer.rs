//! Aspect analyzer — thin facade re-exporting from sub-modules.
//!
//! Split into:
//! - `aspect_analyzer_types` — XSD mappings, conversion helpers
//! - `aspect_analyzer_runner` — `run()` and subcommand dispatch
//! - `aspect_analyzer_formats` — per-format output handlers
//! - `aspect_analyzer_tests` — unit tests
pub use crate::commands::aspect_analyzer_formats::*;
pub use crate::commands::aspect_analyzer_runner::run;
pub use crate::commands::aspect_analyzer_types::*;
