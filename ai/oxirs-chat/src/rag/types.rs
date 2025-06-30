//! Core types and data structures for the RAG system

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// A document in the RAG system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagDocument {
    /// Unique document identifier
    pub id: String,
    /// Document content (text)
    pub content: String,
    /// Document metadata
    pub metadata: HashMap<String, String>,
    /// Document embedding vector (if available)
    pub embedding: Option<Vec<f32>>,
    /// Document timestamp
    pub timestamp: DateTime<Utc>,
    /// Source of the document
    pub source: String,
}

impl RagDocument {
    /// Create a new RAG document
    pub fn new(id: String, content: String, source: String) -> Self {
        Self {
            id,
            content,
            source,
            metadata: HashMap::new(),
            embedding: None,
            timestamp: Utc::now(),
        }
    }

    /// Add metadata to the document
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set the embedding vector
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Get the length of the document content
    pub fn content_length(&self) -> usize {
        self.content.len()
    }

    /// Check if document has embedding
    pub fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }
}

/// Search result from the RAG system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Retrieved document
    pub document: RagDocument,
    /// Relevance score
    pub score: f64,
    /// Factors contributing to relevance
    pub relevance_factors: Vec<String>,
}

impl SearchResult {
    /// Create a new search result
    pub fn new(document: RagDocument, score: f64) -> Self {
        Self {
            document,
            score,
            relevance_factors: Vec::new(),
        }
    }

    /// Add a relevance factor
    pub fn add_relevance_factor(mut self, factor: String) -> Self {
        self.relevance_factors.push(factor);
        self
    }
}

/// Query context for RAG retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryContext {
    /// User ID (for personalization)
    pub user_id: Option<String>,
    /// Session ID
    pub session_id: String,
    /// Previous messages in conversation
    pub conversation_history: Vec<ConversationMessage>,
    /// Domain or topic constraints
    pub domain_constraints: Vec<String>,
    /// Preferred response format
    pub response_format: ResponseFormat,
    /// Maximum response length
    pub max_response_length: usize,
    /// Query intent classification
    pub query_intent: QueryIntent,
}

impl QueryContext {
    /// Create a new query context
    pub fn new(session_id: String) -> Self {
        Self {
            user_id: None,
            session_id,
            conversation_history: Vec::new(),
            domain_constraints: Vec::new(),
            response_format: ResponseFormat::Text,
            max_response_length: 4000,
            query_intent: QueryIntent::Information,
        }
    }

    /// Add a message to conversation history
    pub fn add_message(mut self, message: ConversationMessage) -> Self {
        self.conversation_history.push(message);
        self
    }

    /// Set domain constraints
    pub fn with_domain_constraints(mut self, constraints: Vec<String>) -> Self {
        self.domain_constraints = constraints;
        self
    }

    /// Set query intent
    pub fn with_intent(mut self, intent: QueryIntent) -> Self {
        self.query_intent = intent;
        self
    }
}

/// Conversation message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    /// Message role (user, assistant, system)
    pub role: MessageRole,
    /// Message content
    pub content: String,
    /// Message timestamp
    pub timestamp: DateTime<Utc>,
}

/// Message role enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

/// Response format preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseFormat {
    Text,
    Structured,
    Code,
    Table,
    List,
}

/// Query intent classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryIntent {
    Information,
    Navigation,
    Transaction,
    Comparison,
    Explanation,
    Discovery,
}

/// Assembled context from RAG retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssembledContext {
    /// Retrieved documents
    pub documents: Vec<SearchResult>,
    /// Synthesized context text
    pub context_text: String,
    /// Context metadata
    pub metadata: ContextMetadata,
    /// Assembly statistics
    pub stats: AssemblyStats,
}

impl AssembledContext {
    /// Create a new assembled context
    pub fn new(documents: Vec<SearchResult>, context_text: String) -> Self {
        Self {
            documents,
            context_text,
            metadata: ContextMetadata::default(),
            stats: AssemblyStats::default(),
        }
    }

    /// Get the total number of documents
    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    /// Get the context length in characters
    pub fn context_length(&self) -> usize {
        self.context_text.len()
    }

    /// Get the average relevance score
    pub fn average_relevance_score(&self) -> f64 {
        if self.documents.is_empty() {
            0.0
        } else {
            self.documents.iter().map(|d| d.score).sum::<f64>() / self.documents.len() as f64
        }
    }
}

/// Context metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMetadata {
    /// Assembly timestamp
    pub assembled_at: DateTime<Utc>,
    /// Source diversity (number of different sources)
    pub source_diversity: usize,
    /// Topic coverage
    pub topic_coverage: Vec<String>,
    /// Confidence score
    pub confidence_score: f64,
}

impl Default for ContextMetadata {
    fn default() -> Self {
        Self {
            assembled_at: Utc::now(),
            source_diversity: 0,
            topic_coverage: Vec::new(),
            confidence_score: 0.0,
        }
    }
}

/// Assembly statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblyStats {
    /// Time taken to assemble context
    pub assembly_time: Duration,
    /// Number of documents processed
    pub documents_processed: usize,
    /// Number of documents selected
    pub documents_selected: usize,
    /// Total tokens in context
    pub total_tokens: usize,
    /// Retrieval method used
    pub retrieval_method: String,
}

impl Default for AssemblyStats {
    fn default() -> Self {
        Self {
            assembly_time: Duration::from_millis(0),
            documents_processed: 0,
            documents_selected: 0,
            total_tokens: 0,
            retrieval_method: "default".to_string(),
        }
    }
}

/// Retrieval configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Maximum number of documents to retrieve
    pub max_documents: usize,
    /// Similarity threshold
    pub similarity_threshold: f64,
    /// Enable re-ranking
    pub enable_reranking: bool,
    /// Re-ranking model
    pub reranking_model: Option<String>,
    /// Enable temporal filtering
    pub enable_temporal_filtering: bool,
    /// Temporal window (for filtering by recency)
    pub temporal_window: Option<Duration>,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            max_documents: 20,
            similarity_threshold: 0.7,
            enable_reranking: true,
            reranking_model: None,
            enable_temporal_filtering: false,
            temporal_window: None,
        }
    }
}

/// Context assembly configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblyConfig {
    /// Maximum context length in tokens
    pub max_context_tokens: usize,
    /// Context overlap for chunking
    pub context_overlap: usize,
    /// Prioritize recent documents
    pub prioritize_recent: bool,
    /// Enable diversity optimization
    pub enable_diversity: bool,
    /// Diversity threshold
    pub diversity_threshold: f64,
}

impl Default for AssemblyConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: 4000,
            context_overlap: 200,
            prioritize_recent: true,
            enable_diversity: true,
            diversity_threshold: 0.8,
        }
    }
}