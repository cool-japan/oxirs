//! SHACL Language Server Protocol (LSP) Binary
//!
//! This binary provides IDE integration for SHACL shape authoring.
//!
//! # Features
//!
//! - Real-time validation diagnostics
//! - Code completion for SHACL vocabulary
//! - Hover documentation
//! - Go-to-definition for shape references
//! - Find references
//! - Semantic syntax highlighting
//!
//! # Usage
//!
//! Start the LSP server:
//! ```bash
//! cargo run --bin shacl_lsp --features lsp
//! ```
//!
//! The server communicates via stdin/stdout using the Language Server Protocol.
//!
//! # IDE Configuration
//!
//! ## VS Code
//!
//! Add to `.vscode/settings.json`:
//! ```json
//! {
//!   "rdf.lsp.enabled": true,
//!   "rdf.lsp.serverPath": "/path/to/oxirs-shacl/target/release/shacl_lsp"
//! }
//! ```
//!
//! ## IntelliJ IDEA
//!
//! Install the LSP Support plugin and configure the server path.

use anyhow::Result;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use oxirs_shacl::lsp::ShaclLanguageServer;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for logging
    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    tracing::info!("OxiRS SHACL Language Server v{}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Protocol: Language Server Protocol 3.17");
    tracing::info!("Transport: stdio");

    // Start the LSP server
    ShaclLanguageServer::start().await?;

    Ok(())
}
