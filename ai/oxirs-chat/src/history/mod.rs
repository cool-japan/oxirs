//! Conversation History Management
//!
//! Provides persistent storage, full-text search, automatic summarization,
//! and export capabilities for conversation histories.
//!
//! # Features
//! - File-based JSON persistent storage
//! - Full-text search through past conversations
//! - Automatic summarization of old conversations to save space
//! - Export to Markdown/JSON formats
//! - `HistoryIndex` for fast retrieval

use crate::{Message, MessageRole};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};
use thiserror::Error;
use tracing::{debug, info, warn};

/// Error types for history management
#[derive(Debug, Error)]
pub enum HistoryError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("Conversation not found: {0}")]
    NotFound(String),
    #[error("History index error: {0}")]
    IndexError(String),
    #[error("Export error: {0}")]
    ExportError(String),
}

/// Result type for history operations
pub type HistoryResult<T> = Result<T, HistoryError>;

/// A single conversation entry stored in history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationEntry {
    /// Unique identifier for the conversation
    pub id: String,
    /// Session title (auto-generated or user-defined)
    pub title: String,
    /// All messages in the conversation
    pub messages: Vec<Message>,
    /// When the conversation started
    pub created_at: DateTime<Utc>,
    /// When the conversation was last modified
    pub updated_at: DateTime<Utc>,
    /// Number of messages
    pub message_count: usize,
    /// Total character count for sizing decisions
    pub total_chars: usize,
    /// Automatically generated summary (populated when conversation is old)
    pub summary: Option<ConversationSummary>,
    /// User-defined tags for organization
    pub tags: Vec<String>,
    /// Whether the full message history has been replaced by summary
    pub is_summarized: bool,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ConversationEntry {
    /// Create a new conversation entry
    pub fn new(id: String, messages: Vec<Message>) -> Self {
        let now = Utc::now();
        let title = Self::generate_title(&messages);
        let total_chars: usize = messages.iter().map(|m| m.content.len()).sum();
        let message_count = messages.len();
        Self {
            id,
            title,
            messages,
            created_at: now,
            updated_at: now,
            message_count,
            total_chars,
            summary: None,
            tags: Vec::new(),
            is_summarized: false,
            metadata: HashMap::new(),
        }
    }

    /// Generate a title from the first user message
    fn generate_title(messages: &[Message]) -> String {
        messages
            .iter()
            .find(|m| m.role == MessageRole::User)
            .map(|m| {
                let text = m.content.to_text();
                let truncated = if text.len() > 60 {
                    format!("{}...", &text[..57])
                } else {
                    text.to_string()
                };
                truncated
            })
            .unwrap_or_else(|| "Untitled Conversation".to_string())
    }

    /// Get searchable text from this conversation
    pub fn searchable_text(&self) -> String {
        let mut parts = vec![self.title.clone()];
        // Add tags
        parts.extend(self.tags.clone());
        // Add summary if present
        if let Some(summary) = &self.summary {
            parts.push(summary.text.clone());
            parts.extend(summary.key_points.clone());
        }
        // Add message content (up to 500 chars per message to limit memory)
        if !self.is_summarized {
            for msg in &self.messages {
                let text = msg.content.to_text();
                let chunk = if text.len() > 500 { &text[..500] } else { text };
                parts.push(chunk.to_string());
            }
        }
        parts.join(" ").to_lowercase()
    }
}

/// An automatically generated summary of a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSummary {
    /// Full summary text
    pub text: String,
    /// Key points extracted from the conversation
    pub key_points: Vec<String>,
    /// Main topics discussed
    pub topics: Vec<String>,
    /// When the summary was generated
    pub generated_at: DateTime<Utc>,
    /// Original message count before summarization
    pub original_message_count: usize,
    /// Original char count before summarization
    pub original_char_count: usize,
    /// Compression ratio achieved
    pub compression_ratio: f64,
}

/// Index entry for fast retrieval without loading full conversations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryIndexEntry {
    pub id: String,
    pub title: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub message_count: usize,
    pub total_chars: usize,
    pub tags: Vec<String>,
    pub is_summarized: bool,
    /// Pre-computed search tokens for fast full-text search
    pub search_tokens: Vec<String>,
}

impl HistoryIndexEntry {
    /// Build from a conversation entry
    pub fn from_entry(entry: &ConversationEntry) -> Self {
        let search_text = entry.searchable_text();
        let search_tokens = tokenize(&search_text);
        Self {
            id: entry.id.clone(),
            title: entry.title.clone(),
            created_at: entry.created_at,
            updated_at: entry.updated_at,
            message_count: entry.message_count,
            total_chars: entry.total_chars,
            tags: entry.tags.clone(),
            is_summarized: entry.is_summarized,
            search_tokens,
        }
    }
}

/// Fast search index over conversation history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryIndex {
    /// All index entries, keyed by conversation ID
    pub entries: HashMap<String, HistoryIndexEntry>,
    /// Inverted index: token -> list of conversation IDs
    inverted_index: HashMap<String, Vec<String>>,
    /// Last time the index was rebuilt
    pub last_updated: DateTime<Utc>,
}

impl HistoryIndex {
    /// Create an empty index
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            inverted_index: HashMap::new(),
            last_updated: Utc::now(),
        }
    }

    /// Add or update an entry in the index
    pub fn upsert(&mut self, entry: &ConversationEntry) {
        let index_entry = HistoryIndexEntry::from_entry(entry);

        // Remove old tokens if updating
        if let Some(old_entry) = self.entries.get(&entry.id) {
            for token in &old_entry.search_tokens {
                if let Some(ids) = self.inverted_index.get_mut(token) {
                    ids.retain(|id| id != &entry.id);
                }
            }
        }

        // Add new tokens
        for token in &index_entry.search_tokens {
            self.inverted_index
                .entry(token.clone())
                .or_default()
                .push(entry.id.clone());
        }

        self.entries.insert(entry.id.clone(), index_entry);
        self.last_updated = Utc::now();
    }

    /// Remove an entry from the index
    pub fn remove(&mut self, id: &str) {
        if let Some(old_entry) = self.entries.remove(id) {
            for token in &old_entry.search_tokens {
                if let Some(ids) = self.inverted_index.get_mut(token) {
                    ids.retain(|eid| eid != id);
                }
            }
        }
    }

    /// Full-text search returning ranked results
    pub fn search(&self, query: &str) -> Vec<SearchResult> {
        let query_tokens = tokenize(&query.to_lowercase());
        if query_tokens.is_empty() {
            return Vec::new();
        }

        // Score each conversation by token matches
        let mut scores: HashMap<&str, f64> = HashMap::new();
        for token in &query_tokens {
            if let Some(ids) = self.inverted_index.get(token) {
                for id in ids {
                    let score = scores.entry(id.as_str()).or_insert(0.0);
                    // Simple TF-IDF-like scoring: boost for rarer tokens
                    let idf = 1.0 + (self.entries.len() as f64 / (ids.len() as f64 + 1.0)).ln();
                    *score += idf;
                }
            }
        }

        let mut results: Vec<SearchResult> = scores
            .into_iter()
            .filter_map(|(id, score)| {
                self.entries.get(id).map(|entry| SearchResult {
                    id: id.to_string(),
                    title: entry.title.clone(),
                    score,
                    created_at: entry.created_at,
                    updated_at: entry.updated_at,
                    message_count: entry.message_count,
                    tags: entry.tags.clone(),
                    snippet: String::new(),
                })
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// List all entries sorted by updated_at descending
    pub fn list_recent(&self, limit: usize) -> Vec<&HistoryIndexEntry> {
        let mut entries: Vec<&HistoryIndexEntry> = self.entries.values().collect();
        entries.sort_by_key(|item| std::cmp::Reverse(item.updated_at));
        entries.into_iter().take(limit).collect()
    }
}

impl Default for HistoryIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// A search result with relevance score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub title: String,
    pub score: f64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub message_count: usize,
    pub tags: Vec<String>,
    /// Extracted snippet showing context around the match
    pub snippet: String,
}

/// Configuration for conversation history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryConfig {
    /// Root directory for history storage
    pub storage_dir: PathBuf,
    /// Maximum number of conversations to keep (0 = unlimited)
    pub max_conversations: usize,
    /// Summarize conversations older than this many days
    pub summarize_after_days: i64,
    /// Summarize conversations with more than this many messages
    pub summarize_after_messages: usize,
    /// Maximum characters before summarizing
    pub summarize_after_chars: usize,
    /// Auto-delete conversations older than this many days (0 = never)
    pub auto_delete_after_days: i64,
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            storage_dir: PathBuf::from("data/history"),
            max_conversations: 10_000,
            summarize_after_days: 30,
            summarize_after_messages: 100,
            summarize_after_chars: 50_000,
            auto_delete_after_days: 0, // Never auto-delete
        }
    }
}

/// Manages persistent conversation history with search and export capabilities
pub struct ConversationHistory {
    config: HistoryConfig,
    index: HistoryIndex,
    index_path: PathBuf,
}

impl ConversationHistory {
    /// Create a new ConversationHistory, loading the existing index if present
    pub fn new(config: HistoryConfig) -> HistoryResult<Self> {
        fs::create_dir_all(&config.storage_dir)?;

        let index_path = config.storage_dir.join("index.json");
        let index = if index_path.exists() {
            let data = fs::read_to_string(&index_path)?;
            serde_json::from_str(&data)?
        } else {
            HistoryIndex::new()
        };

        Ok(Self {
            config,
            index,
            index_path,
        })
    }

    /// Create with a temporary directory for testing
    pub fn with_temp_dir(dir: PathBuf) -> HistoryResult<Self> {
        let config = HistoryConfig {
            storage_dir: dir,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Save a conversation to persistent storage
    pub fn save(&mut self, entry: ConversationEntry) -> HistoryResult<()> {
        let conversation_path = self.conversation_path(&entry.id);
        let data = serde_json::to_string_pretty(&entry)?;
        fs::write(&conversation_path, data)?;
        self.index.upsert(&entry);
        self.save_index()?;
        debug!("Saved conversation {} to history", entry.id);
        Ok(())
    }

    /// Load a specific conversation by ID
    pub fn load(&self, id: &str) -> HistoryResult<ConversationEntry> {
        let path = self.conversation_path(id);
        if !path.exists() {
            return Err(HistoryError::NotFound(id.to_string()));
        }
        let data = fs::read_to_string(&path)?;
        let entry = serde_json::from_str(&data)?;
        Ok(entry)
    }

    /// Delete a conversation by ID
    pub fn delete(&mut self, id: &str) -> HistoryResult<()> {
        let path = self.conversation_path(id);
        if path.exists() {
            fs::remove_file(&path)?;
        }
        self.index.remove(id);
        self.save_index()?;
        info!("Deleted conversation {} from history", id);
        Ok(())
    }

    /// Full-text search through conversation history
    pub fn search(&self, query: &str) -> Vec<SearchResult> {
        self.index.search(query)
    }

    /// List recent conversations
    pub fn list_recent(&self, limit: usize) -> Vec<&HistoryIndexEntry> {
        self.index.list_recent(limit)
    }

    /// Automatically summarize old conversations to save space
    pub fn run_summarization(&mut self) -> HistoryResult<usize> {
        let now = Utc::now();
        let mut summarized_count = 0;

        // Collect IDs that need summarization
        let ids_to_summarize: Vec<String> = self
            .index
            .entries
            .values()
            .filter(|entry| {
                if entry.is_summarized {
                    return false;
                }
                let age_days = (now - entry.created_at).num_days();
                age_days >= self.config.summarize_after_days
                    || entry.message_count >= self.config.summarize_after_messages
                    || entry.total_chars >= self.config.summarize_after_chars
            })
            .map(|e| e.id.clone())
            .collect();

        for id in ids_to_summarize {
            match self.load(&id) {
                Ok(mut entry) => {
                    let summary = generate_summary(&entry);
                    let original_message_count = entry.messages.len();
                    let original_char_count = entry.total_chars;

                    entry.summary = Some(summary);
                    entry.is_summarized = true;
                    // Replace messages with a single summary marker
                    entry.messages.clear();
                    entry.total_chars = entry.summary.as_ref().map(|s| s.text.len()).unwrap_or(0);
                    entry.message_count = original_message_count;

                    info!(
                        "Summarized conversation {} ({} messages, {} chars -> {} chars)",
                        id, original_message_count, original_char_count, entry.total_chars
                    );

                    if let Err(e) = self.save(entry) {
                        warn!("Failed to save summarized conversation {}: {}", id, e);
                    } else {
                        summarized_count += 1;
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to load conversation {} for summarization: {}",
                        id, e
                    );
                }
            }
        }

        Ok(summarized_count)
    }

    /// Export a conversation to Markdown format
    pub fn export_markdown(&self, id: &str) -> HistoryResult<String> {
        let entry = self.load(id)?;
        let mut output = String::new();

        output.push_str(&format!("# {}\n\n", entry.title));
        output.push_str(&format!(
            "_Started: {}_\n\n",
            entry.created_at.format("%Y-%m-%d %H:%M UTC")
        ));

        if !entry.tags.is_empty() {
            output.push_str(&format!("**Tags:** {}\n\n", entry.tags.join(", ")));
        }

        if let Some(summary) = &entry.summary {
            output.push_str("## Summary\n\n");
            output.push_str(&summary.text);
            output.push_str("\n\n");

            if !summary.key_points.is_empty() {
                output.push_str("### Key Points\n\n");
                for point in &summary.key_points {
                    output.push_str(&format!("- {}\n", point));
                }
                output.push('\n');
            }

            if entry.is_summarized {
                output.push_str(
                    "_Note: Full message history has been replaced by this summary._\n\n",
                );
                return Ok(output);
            }
        }

        output.push_str("## Conversation\n\n");
        for msg in &entry.messages {
            let role_label = match msg.role {
                MessageRole::User => "**User**",
                MessageRole::Assistant => "**Assistant**",
                MessageRole::System => "**System**",
                MessageRole::Function => "**Function**",
            };
            output.push_str(&format!(
                "### {} _{}_\n\n",
                role_label,
                msg.timestamp.format("%H:%M")
            ));
            output.push_str(msg.content.to_text());
            output.push_str("\n\n---\n\n");
        }

        Ok(output)
    }

    /// Export a conversation to JSON format
    pub fn export_json(&self, id: &str) -> HistoryResult<String> {
        let entry = self.load(id)?;
        let json = serde_json::to_string_pretty(&entry)?;
        Ok(json)
    }

    /// Export a conversation to JSONL format (for fine-tuning)
    pub fn export_jsonl(&self, id: &str) -> HistoryResult<String> {
        let entry = self.load(id)?;
        let mut output = String::new();

        for (i, msg) in entry.messages.iter().enumerate() {
            let record = serde_json::json!({
                "conversation_id": entry.id,
                "message_index": i,
                "role": format!("{:?}", msg.role).to_lowercase(),
                "content": msg.content.to_text(),
                "timestamp": msg.timestamp,
            });
            output.push_str(&serde_json::to_string(&record)?);
            output.push('\n');
        }
        Ok(output)
    }

    /// Export all conversations to a single JSON file
    pub fn export_all_json(&self, output_path: &Path) -> HistoryResult<usize> {
        let entries: Vec<HistoryIndexEntry> = self.index.entries.values().cloned().collect();
        let mut conversations = Vec::new();

        for entry in &entries {
            match self.load(&entry.id) {
                Ok(conv) => conversations.push(conv),
                Err(e) => warn!("Failed to load {} for export: {}", entry.id, e),
            }
        }

        let count = conversations.len();
        let json = serde_json::to_string_pretty(&conversations)?;
        fs::write(output_path, json)?;
        info!("Exported {} conversations to {:?}", count, output_path);
        Ok(count)
    }

    /// Get the index for external inspection
    pub fn index(&self) -> &HistoryIndex {
        &self.index
    }

    /// Get the count of stored conversations
    pub fn conversation_count(&self) -> usize {
        self.index.entries.len()
    }

    /// Enforce maximum conversation limit by deleting oldest entries
    pub fn enforce_limit(&mut self) -> HistoryResult<usize> {
        let max = self.config.max_conversations;
        if max == 0 || self.index.entries.len() <= max {
            return Ok(0);
        }

        let excess = self.index.entries.len() - max;
        let oldest_ids: Vec<String> = {
            let mut entries: Vec<_> = self.index.entries.values().collect();
            entries.sort_by_key(|item| item.updated_at);
            entries
                .into_iter()
                .take(excess)
                .map(|e| e.id.clone())
                .collect()
        };

        let mut deleted = 0;
        for id in oldest_ids {
            if let Err(e) = self.delete(&id) {
                warn!("Failed to delete old conversation {}: {}", id, e);
            } else {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    // --- Private helpers ---

    fn conversation_path(&self, id: &str) -> PathBuf {
        // Use first 2 chars as subdirectory to avoid flat directory with millions of files
        let subdir = if id.len() >= 2 { &id[..2] } else { "xx" };
        let dir = self.config.storage_dir.join(subdir);
        // Create subdir if needed (ignore error - may already exist)
        let _ = fs::create_dir_all(&dir);
        dir.join(format!("{}.json", id))
    }

    fn save_index(&self) -> HistoryResult<()> {
        let data = serde_json::to_string_pretty(&self.index)?;
        fs::write(&self.index_path, data)?;
        Ok(())
    }
}

/// Tokenize text into search tokens
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 3)
        .map(|t| t.to_lowercase())
        .collect()
}

/// Generate a summary from a conversation entry using extractive summarization
fn generate_summary(entry: &ConversationEntry) -> ConversationSummary {
    let original_char_count = entry.total_chars;
    let original_message_count = entry.messages.len();

    // Collect all text content
    let all_text: Vec<String> = entry
        .messages
        .iter()
        .map(|m| m.content.to_text().to_string())
        .collect();

    // Extract key sentences using a simple frequency-based approach
    let key_points = extract_key_points(&all_text, 5);

    // Build topics from frequent words
    let topics = extract_topics(&all_text, 5);

    // Build summary text from key points
    let text = if key_points.is_empty() {
        format!(
            "Conversation with {} messages covering: {}",
            original_message_count,
            topics.join(", ")
        )
    } else {
        key_points.join(" ")
    };

    let compression_ratio = if original_char_count > 0 {
        text.len() as f64 / original_char_count as f64
    } else {
        1.0
    };

    ConversationSummary {
        text,
        key_points,
        topics,
        generated_at: Utc::now(),
        original_message_count,
        original_char_count,
        compression_ratio,
    }
}

/// Extract key sentences using sentence scoring
fn extract_key_points(texts: &[String], max_points: usize) -> Vec<String> {
    // Build word frequency map
    let mut word_freq: HashMap<String, usize> = HashMap::new();
    for text in texts {
        for word in text.split_whitespace() {
            let cleaned = word
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if cleaned.len() >= 4 {
                *word_freq.entry(cleaned).or_insert(0) += 1;
            }
        }
    }

    // Score sentences
    let mut scored_sentences: Vec<(f64, String)> = Vec::new();
    for text in texts {
        for sentence in text.split(['.', '!', '?']) {
            let sentence = sentence.trim();
            if sentence.len() < 20 || sentence.len() > 300 {
                continue;
            }
            let mut score = 0.0f64;
            for word in sentence.split_whitespace() {
                let cleaned = word
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase();
                if let Some(&freq) = word_freq.get(&cleaned) {
                    score += freq as f64;
                }
            }
            // Normalize by sentence length
            let word_count = sentence.split_whitespace().count();
            if word_count > 0 {
                score /= word_count as f64;
            }
            scored_sentences.push((score, sentence.to_string()));
        }
    }

    scored_sentences.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored_sentences
        .into_iter()
        .take(max_points)
        .map(|(_, s)| s)
        .collect()
}

/// Extract main topics as frequent significant words
fn extract_topics(texts: &[String], max_topics: usize) -> Vec<String> {
    // Common English stop words to filter out
    let stop_words = [
        "the", "and", "for", "that", "this", "with", "have", "from", "they", "will", "would",
        "could", "should", "what", "when", "where", "which", "there", "their", "been", "were",
        "into",
    ];

    let mut word_freq: HashMap<String, usize> = HashMap::new();
    for text in texts {
        for word in text.split_whitespace() {
            let cleaned = word
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if cleaned.len() >= 4 && !stop_words.contains(&cleaned.as_str()) {
                *word_freq.entry(cleaned).or_insert(0) += 1;
            }
        }
    }

    let mut freq_list: Vec<(String, usize)> = word_freq
        .into_iter()
        .filter(|(_, freq)| *freq >= 2)
        .collect();

    freq_list.sort_by_key(|item| std::cmp::Reverse(item.1));
    freq_list
        .into_iter()
        .take(max_topics)
        .map(|(word, _)| word)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MessageContent, MessageRole};
    use std::env;

    fn make_message(role: MessageRole, content: &str) -> Message {
        Message {
            id: uuid::Uuid::new_v4().to_string(),
            role,
            content: MessageContent::Text(content.to_string()),
            timestamp: Utc::now(),
            metadata: None,
            thread_id: None,
            parent_message_id: None,
            token_count: None,
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements: Vec::new(),
        }
    }

    fn make_conversation(id: &str) -> ConversationEntry {
        let messages = vec![
            make_message(MessageRole::User, "How does SPARQL work?"),
            make_message(
                MessageRole::Assistant,
                "SPARQL is a query language for RDF data. It allows you to query semantic web data.",
            ),
            make_message(MessageRole::User, "Can you show me an example query?"),
            make_message(
                MessageRole::Assistant,
                "Sure! Here is a simple SPARQL SELECT query: SELECT ?subject WHERE { ?subject a owl:Class }",
            ),
        ];
        ConversationEntry::new(id.to_string(), messages)
    }

    #[test]
    fn test_conversation_entry_creation() {
        let entry = make_conversation("test-001");
        assert_eq!(entry.id, "test-001");
        assert!(!entry.title.is_empty());
        assert_eq!(entry.message_count, 4);
        assert!(!entry.is_summarized);
    }

    #[test]
    fn test_history_index_search() {
        let mut index = HistoryIndex::new();
        let entry = make_conversation("conv-001");
        index.upsert(&entry);

        let results = index.search("sparql query");
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "conv-001");
    }

    #[test]
    fn test_history_index_remove() {
        let mut index = HistoryIndex::new();
        let entry = make_conversation("conv-remove");
        index.upsert(&entry);
        assert!(index.entries.contains_key("conv-remove"));

        index.remove("conv-remove");
        assert!(!index.entries.contains_key("conv-remove"));

        let results = index.search("sparql");
        assert!(results.iter().all(|r| r.id != "conv-remove"));
    }

    #[test]
    fn test_save_and_load() {
        let dir = env::temp_dir().join(format!("oxirs_history_test_{}", uuid::Uuid::new_v4()));
        let mut history = ConversationHistory::with_temp_dir(dir.clone()).expect("create history");

        let entry = make_conversation("save-test-001");
        history.save(entry.clone()).expect("save");

        let loaded = history.load("save-test-001").expect("load");
        assert_eq!(loaded.id, "save-test-001");
        assert_eq!(loaded.message_count, 4);

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_search_after_save() {
        let dir = env::temp_dir().join(format!("oxirs_history_search_{}", uuid::Uuid::new_v4()));
        let mut history = ConversationHistory::with_temp_dir(dir.clone()).expect("create history");

        history
            .save(make_conversation("search-test-001"))
            .expect("save");

        let results = history.search("sparql query");
        assert!(!results.is_empty());

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_export_markdown() {
        let dir = env::temp_dir().join(format!("oxirs_history_md_{}", uuid::Uuid::new_v4()));
        let mut history = ConversationHistory::with_temp_dir(dir.clone()).expect("create history");

        let entry = make_conversation("md-export-001");
        history.save(entry).expect("save");

        let markdown = history.export_markdown("md-export-001").expect("export");
        assert!(markdown.contains("## Conversation"));
        assert!(markdown.contains("**User**"));
        assert!(markdown.contains("SPARQL"));

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_export_json() {
        let dir = env::temp_dir().join(format!("oxirs_history_json_{}", uuid::Uuid::new_v4()));
        let mut history = ConversationHistory::with_temp_dir(dir.clone()).expect("create history");

        let entry = make_conversation("json-export-001");
        history.save(entry).expect("save");

        let json = history.export_json("json-export-001").expect("export");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        assert_eq!(parsed["id"], "json-export-001");

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_summarization() {
        let entry = make_conversation("sum-test");
        let summary = generate_summary(&entry);
        assert!(!summary.text.is_empty());
        assert!(summary.original_message_count > 0);
    }

    #[test]
    fn test_list_recent() {
        let mut index = HistoryIndex::new();
        index.upsert(&make_conversation("conv-a"));
        index.upsert(&make_conversation("conv-b"));
        index.upsert(&make_conversation("conv-c"));

        let recent = index.list_recent(2);
        assert_eq!(recent.len(), 2);
    }

    #[test]
    fn test_delete() {
        let dir = env::temp_dir().join(format!("oxirs_history_del_{}", uuid::Uuid::new_v4()));
        let mut history = ConversationHistory::with_temp_dir(dir.clone()).expect("create history");

        history
            .save(make_conversation("del-test-001"))
            .expect("save");
        assert_eq!(history.conversation_count(), 1);

        history.delete("del-test-001").expect("delete");
        assert_eq!(history.conversation_count(), 0);

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_export_jsonl() {
        let dir = env::temp_dir().join(format!("oxirs_history_jsonl_{}", uuid::Uuid::new_v4()));
        let mut history = ConversationHistory::with_temp_dir(dir.clone()).expect("create history");

        let entry = make_conversation("jsonl-001");
        history.save(entry).expect("save");

        let jsonl = history.export_jsonl("jsonl-001").expect("export");
        let lines: Vec<_> = jsonl.lines().filter(|l| !l.is_empty()).collect();
        assert!(!lines.is_empty());
        // Each line should be valid JSON
        for line in &lines {
            serde_json::from_str::<serde_json::Value>(line).expect("valid json line");
        }

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }
}
