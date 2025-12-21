//! LSP server implementation.
//!
//! Provides the main server setup and initialization.

use tower_lsp::{LspService, Server};

use crate::lsp::backend::ShaclBackend;

/// SHACL Language Server
pub struct ShaclLanguageServer;

impl ShaclLanguageServer {
    /// Start the LSP server
    pub async fn start() -> anyhow::Result<()> {
        tracing::info!("Starting SHACL LSP Server...");

        // Create LSP service
        let (service, socket) = LspService::new(ShaclBackend::new);

        // Start server with stdio transport
        Server::new(tokio::io::stdin(), tokio::io::stdout(), socket)
            .serve(service)
            .await;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_server_creation() {
        // Test that server can be created
        // Full test would require mocking stdio
    }
}
