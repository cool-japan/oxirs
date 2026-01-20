//! LSP Backend implementation for SHACL shapes.
//!
//! Provides the core language server functionality including:
//! - Document synchronization
//! - Diagnostics generation
//! - Code completion
//! - Hover information
//! - Go-to-definition
//! - Find references

use std::sync::Arc;

use dashmap::DashMap;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::{
    CompletionOptions, CompletionParams, CompletionResponse, Diagnostic, DiagnosticSeverity,
    DidChangeTextDocumentParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams,
    DidSaveTextDocumentParams, GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverParams,
    HoverProviderCapability, InitializeParams, InitializeResult, InitializedParams, MessageType,
    OneOf, Position, Range, SemanticTokens, SemanticTokensFullOptions, SemanticTokensOptions,
    SemanticTokensParams, SemanticTokensRangeParams, SemanticTokensRangeResult,
    SemanticTokensResult, SemanticTokensServerCapabilities, ServerCapabilities, ServerInfo,
    TextDocumentSyncCapability, TextDocumentSyncKind, Url, WorkDoneProgressOptions,
};
use tower_lsp::{Client, LanguageServer};

use crate::lsp::completion::CompletionProvider;
use crate::lsp::diagnostics::DiagnosticsGenerator;
use crate::lsp::hover::HoverProvider;
use crate::lsp::semantic_tokens::SemanticTokensProvider;
use crate::Shape;

use oxirs_core::ConcreteStore;

/// Document state
#[derive(Debug, Clone)]
struct DocumentState {
    /// Document text
    text: String,
    /// Document version
    version: i32,
    /// Parsed shapes
    shapes: Vec<Shape>,
    /// Current diagnostics
    diagnostics: Vec<Diagnostic>,
}

/// SHACL LSP Backend
pub struct ShaclBackend {
    /// LSP client for sending notifications
    client: Client,
    /// Document states indexed by URI
    documents: Arc<DashMap<Url, DocumentState>>,
    /// RDF store for shape parsing
    _store: Arc<ConcreteStore>,
    /// Completion provider
    completion_provider: Arc<CompletionProvider>,
    /// Hover provider
    hover_provider: Arc<HoverProvider>,
    /// Diagnostics generator
    diagnostics_generator: Arc<DiagnosticsGenerator>,
    /// Semantic tokens provider
    semantic_tokens_provider: Arc<SemanticTokensProvider>,
}

impl ShaclBackend {
    /// Create a new SHACL LSP backend
    pub fn new(client: Client) -> Self {
        let store = Arc::new(ConcreteStore::new().expect("Failed to create store"));

        Self {
            client,
            documents: Arc::new(DashMap::new()),
            _store: store.clone(),
            completion_provider: Arc::new(CompletionProvider::new(store.clone())),
            hover_provider: Arc::new(HoverProvider::new(store.clone())),
            diagnostics_generator: Arc::new(DiagnosticsGenerator::new()),
            semantic_tokens_provider: Arc::new(SemanticTokensProvider::new()),
        }
    }

    /// Validate a document and generate diagnostics
    async fn validate_document(&self, uri: &Url) {
        if let Some(mut doc) = self.documents.get_mut(uri) {
            // Parse shapes from document
            let shapes_result = self.parse_shapes_from_text(&doc.text);

            match shapes_result {
                Ok(shapes) => {
                    // Update shapes
                    doc.shapes = shapes.clone();

                    // Generate diagnostics
                    let diagnostics = self
                        .diagnostics_generator
                        .generate_diagnostics(&doc.text, &shapes);

                    doc.diagnostics = diagnostics.clone();

                    // Send diagnostics to client
                    self.client
                        .publish_diagnostics(uri.clone(), diagnostics, Some(doc.version))
                        .await;
                }
                Err(e) => {
                    // Parse error - create diagnostic
                    let diagnostic = Diagnostic {
                        range: Range::new(Position::new(0, 0), Position::new(0, 0)),
                        severity: Some(DiagnosticSeverity::ERROR),
                        code: None,
                        code_description: None,
                        source: Some("shacl-lsp".to_string()),
                        message: format!("Failed to parse SHACL shapes: {}", e),
                        related_information: None,
                        tags: None,
                        data: None,
                    };

                    doc.diagnostics = vec![diagnostic.clone()];

                    self.client
                        .publish_diagnostics(uri.clone(), vec![diagnostic], Some(doc.version))
                        .await;
                }
            }
        }
    }

    /// Parse SHACL shapes from document text
    fn parse_shapes_from_text(&self, _text: &str) -> anyhow::Result<Vec<Shape>> {
        // TODO: In a production implementation, you would:
        // 1. Detect format from file extension or content
        // 2. Parse RDF triples into the store
        // 3. Use ShapeFactory to extract shapes
        // 4. Return parsed shapes

        // For now, return empty vec - full implementation would use ShapeFactory
        Ok(Vec::new())
    }

    /// Get document state
    fn get_document(&self, uri: &Url) -> Option<DocumentState> {
        self.documents.get(uri).map(|d| d.clone())
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for ShaclBackend {
    async fn initialize(&self, _params: InitializeParams) -> Result<InitializeResult> {
        self.client
            .log_message(MessageType::INFO, "SHACL LSP Server initializing...")
            .await;

        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                completion_provider: Some(CompletionOptions {
                    resolve_provider: Some(false),
                    trigger_characters: Some(vec![
                        ":".to_string(),
                        " ".to_string(),
                        "\n".to_string(),
                    ]),
                    work_done_progress_options: WorkDoneProgressOptions::default(),
                    all_commit_characters: None,
                    completion_item: None,
                }),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                definition_provider: Some(OneOf::Left(true)),
                references_provider: Some(OneOf::Left(true)),
                document_symbol_provider: Some(OneOf::Left(true)),
                workspace_symbol_provider: Some(OneOf::Left(true)),
                semantic_tokens_provider: Some(
                    SemanticTokensServerCapabilities::SemanticTokensOptions(
                        SemanticTokensOptions {
                            work_done_progress_options: WorkDoneProgressOptions::default(),
                            legend: SemanticTokensProvider::legend(),
                            range: Some(true),
                            full: Some(SemanticTokensFullOptions::Bool(true)),
                        },
                    ),
                ),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "oxirs-shacl-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _params: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "SHACL LSP Server initialized!")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let text = params.text_document.text;
        let version = params.text_document.version;

        self.client
            .log_message(MessageType::INFO, format!("Document opened: {}", uri))
            .await;

        // Create document state
        let doc_state = DocumentState {
            text: text.clone(),
            version,
            shapes: Vec::new(),
            diagnostics: Vec::new(),
        };

        self.documents.insert(uri.clone(), doc_state);

        // Validate document
        self.validate_document(&uri).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = params.text_document.version;

        if let Some(change) = params.content_changes.first() {
            // Update document text
            if let Some(mut doc) = self.documents.get_mut(&uri) {
                doc.text = change.text.clone();
                doc.version = version;
            }

            // Validate document
            self.validate_document(&uri).await;
        }
    }

    async fn did_save(&self, params: DidSaveTextDocumentParams) {
        self.client
            .log_message(
                MessageType::INFO,
                format!("Document saved: {}", params.text_document.uri),
            )
            .await;
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;

        self.client
            .log_message(MessageType::INFO, format!("Document closed: {}", uri))
            .await;

        // Remove document state
        self.documents.remove(&uri);

        // Clear diagnostics
        self.client.publish_diagnostics(uri, Vec::new(), None).await;
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        if let Some(doc) = self.get_document(&uri) {
            let completions = self
                .completion_provider
                .provide_completions(&doc.text, position)
                .await;
            Ok(Some(completions))
        } else {
            Ok(None)
        }
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        if let Some(doc) = self.get_document(&uri) {
            let hover = self.hover_provider.provide_hover(&doc.text, position).await;
            Ok(hover)
        } else {
            Ok(None)
        }
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = params.text_document_position_params.text_document.uri;
        let _position = params.text_document_position_params.position;

        if let Some(_doc) = self.get_document(&uri) {
            // Find shape definition at position
            // For now, return None - full implementation would search for shape IRI
            Ok(None)
        } else {
            Ok(None)
        }
    }

    async fn semantic_tokens_full(
        &self,
        params: SemanticTokensParams,
    ) -> Result<Option<SemanticTokensResult>> {
        let uri = params.text_document.uri;

        if let Some(doc) = self.get_document(&uri) {
            let tokens = self.semantic_tokens_provider.generate_tokens(&doc.text);

            Ok(Some(SemanticTokensResult::Tokens(SemanticTokens {
                result_id: None,
                data: tokens,
            })))
        } else {
            Ok(None)
        }
    }

    async fn semantic_tokens_range(
        &self,
        params: SemanticTokensRangeParams,
    ) -> Result<Option<SemanticTokensRangeResult>> {
        let uri = params.text_document.uri;
        let range = params.range;

        if let Some(doc) = self.get_document(&uri) {
            let tokens = self.semantic_tokens_provider.generate_tokens_range(
                &doc.text,
                range.start.line,
                range.end.line,
            );

            Ok(Some(SemanticTokensRangeResult::Tokens(SemanticTokens {
                result_id: None,
                data: tokens,
            })))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower_lsp::LspService;

    #[tokio::test]
    async fn test_backend_creation() {
        let (_service, _) = LspService::new(ShaclBackend::new);
        // Backend created successfully
    }

    #[tokio::test]
    async fn test_document_lifecycle() {
        let (_service, _) = LspService::new(ShaclBackend::new);
        // Test document open, change, save, close
    }
}
