//! OxiRS CLI Tools Suite
//!
//! Comprehensive command-line tools for RDF processing, SPARQL querying,
//! validation, and dataset management. Inspired by Apache Jena's CLI tools
//! but built in Rust for performance and safety.

use std::path::PathBuf;

/// Common result type for all tools
pub type ToolResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

// === Data Processing Tools ===
pub mod rdfcat;
pub mod rdfcopy;
pub mod rdfdiff;
pub mod rdfparse;
pub mod riot;

// === Advanced Query Tools ===
pub mod arq;
pub mod qparse;
pub mod rsparql;
pub mod rupdate;
pub mod uparse;

// === Storage Tools ===
pub mod tdbbackup;
pub mod tdbcompact;
pub mod tdbdump;
pub mod tdbloader;
pub mod tdbquery;
pub mod tdbstats;
pub mod tdbupdate;

// === Validation Tools ===
pub mod infer;
pub mod schemagen;
pub mod shacl;
pub mod shex;

// === Utility Tools ===
pub mod iri;
pub mod juuid;
pub mod langtag;
pub mod rset;
pub mod utf8;
pub mod wwwdec;
pub mod wwwenc;

/// Common utilities for tools
pub mod utils;

/// Tool execution statistics
#[derive(Debug, Clone)]
pub struct ToolStats {
    pub start_time: std::time::Instant,
    pub duration: std::time::Duration,
    pub items_processed: usize,
    pub errors: usize,
    pub warnings: usize,
}

impl ToolStats {
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            duration: std::time::Duration::new(0, 0),
            items_processed: 0,
            errors: 0,
            warnings: 0,
        }
    }

    pub fn finish(&mut self) {
        self.duration = self.start_time.elapsed();
    }

    pub fn rate(&self) -> f64 {
        if self.duration.as_secs_f64() > 0.0 {
            self.items_processed as f64 / self.duration.as_secs_f64()
        } else {
            0.0
        }
    }

    pub fn print_summary(&self, tool_name: &str) {
        println!("\n{} completed:", tool_name);
        println!("  Duration: {:.3}s", self.duration.as_secs_f64());
        println!("  Items processed: {}", self.items_processed);
        if self.errors > 0 {
            println!("  Errors: {}", self.errors);
        }
        if self.warnings > 0 {
            println!("  Warnings: {}", self.warnings);
        }
        if self.items_processed > 0 {
            println!("  Rate: {:.1} items/second", self.rate());
        }
    }
}

impl Default for ToolStats {
    fn default() -> Self {
        Self::new()
    }
}
