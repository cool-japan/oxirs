//! Rich content support for oxirs-chat.
//!
//! This module provides support for rich content types including:
//! - Code snippets with syntax highlighting
//! - SPARQL query blocks with validation
//! - Graph visualizations
//! - Table outputs
//! - Image / audio / video attachments
//! - File uploads
//! - Interactive widgets, dashboards, charts, timelines, maps, and 3D
//!   scenes
//!
//! The implementation is split across sibling modules; this file is a
//! thin facade that re-exports the public API:
//!
//! - [`crate::rich_content_types`] — value definitions for every block
//!   variant ([`RichContent`], graph nodes/edges, tables, charts, …) and
//!   the shared [`ChatError`] / [`ChatResult`] aliases.
//! - [`crate::rich_content_renderer`] — [`RichMessage`] composer with
//!   Markdown / HTML rendering plus the [`RichContentProcessor`] pass.
//! - [`crate::rich_content_tests`] — `#[cfg(test)]` unit tests for the
//!   above.

pub use crate::rich_content_renderer::{RichContentProcessor, RichMessage};
pub use crate::rich_content_types::*;
