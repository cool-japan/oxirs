//! Advanced content processing for multiple document formats
//!
//! This module provides comprehensive document parsing and content extraction
//! capabilities for PDF, HTML, XML, office documents, and multimedia content.
//!
//! This module is only available when the `content-processing` feature is enabled.

#[cfg(feature = "content-processing")]
use anyhow::Result;
#[cfg(feature = "content-processing")]
use std::collections::HashMap;

// Re-export handlers
mod data_handlers;
mod multimedia_handlers;
mod office_handlers;
mod pdf_handler;
mod text_handlers;
mod types;

#[cfg(feature = "content-processing")]
pub use data_handlers::*;
#[cfg(feature = "content-processing")]
pub use multimedia_handlers::*;
#[cfg(feature = "content-processing")]
pub use office_handlers::*;
#[cfg(feature = "content-processing")]
pub use pdf_handler::*;
#[cfg(feature = "content-processing")]
pub use text_handlers::*;
#[cfg(feature = "content-processing")]
pub use types::*;

/// Advanced content processor
#[cfg(feature = "content-processing")]
pub struct ContentProcessor {
    config: ContentExtractionConfig,
    format_handlers: HashMap<DocumentFormat, Box<dyn FormatHandler>>,
}

/// Trait for format-specific content handlers
#[cfg(feature = "content-processing")]
pub trait FormatHandler: Send + Sync {
    /// Extract content from document bytes
    fn extract_content(
        &self,
        data: &[u8],
        config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent>;

    /// Check if this handler can process the given data
    fn can_handle(&self, data: &[u8]) -> bool;

    /// Get supported file extensions
    fn supported_extensions(&self) -> Vec<&'static str>;
}
