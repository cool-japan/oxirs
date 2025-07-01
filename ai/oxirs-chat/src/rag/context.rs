//! Context assembly and optimization for the RAG system
//!
//! Handles intelligent context assembly, optimization, and formatting
//! for knowledge graph-based conversational AI responses.

use super::types::*;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Context assembler for RAG results
pub struct ContextAssembler {
    config: AssemblyConfig,
    optimizer: ContextOptimizer,
    formatter: ContextFormatter,
}

impl ContextAssembler {
    /// Create a new context assembler
    pub fn new(rag_config: &super::RAGConfig) -> Self {
        let config = AssemblyConfig {
            max_context_tokens: rag_config.max_context_length,
            context_overlap: rag_config.context_overlap,
            prioritize_recent: true,
            enable_diversity: true,
            diversity_threshold: 0.8,
        };

        Self {
            optimizer: ContextOptimizer::new(&config),
            formatter: ContextFormatter::new(&config),
            config,
        }
    }

    /// Assemble context from search results
    pub async fn assemble(
        &self,
        results: &[SearchResult],
        query_context: &QueryContext,
    ) -> Result<AssembledContext> {
        let start_time = Instant::now();

        // Stage 1: Optimize and filter results
        let optimized_results = self.optimizer.optimize(results, query_context).await?;

        // Stage 2: Format context text
        let context_text = self
            .formatter
            .format(&optimized_results, query_context)
            .await?;

        // Stage 3: Generate metadata
        let metadata = self.generate_metadata(&optimized_results, query_context);

        // Stage 4: Calculate statistics
        let stats = AssemblyStats {
            assembly_time: start_time.elapsed(),
            documents_processed: results.len(),
            documents_selected: optimized_results.len(),
            total_tokens: self.estimate_token_count(&context_text),
            retrieval_method: "multi_stage_rag".to_string(),
        };

        let assembled_context = AssembledContext {
            documents: optimized_results,
            context_text,
            metadata,
            stats,
        };

        info!(
            "Context assembled: {} documents, {} tokens",
            assembled_context.document_count(),
            assembled_context.stats.total_tokens
        );

        Ok(assembled_context)
    }

    /// Update configuration
    pub fn update_config(&mut self, rag_config: &super::RAGConfig) {
        self.config.max_context_tokens = rag_config.max_context_length;
        self.config.context_overlap = rag_config.context_overlap;
        self.optimizer.update_config(&self.config);
        self.formatter.update_config(&self.config);
    }

    /// Generate context metadata
    fn generate_metadata(
        &self,
        results: &[SearchResult],
        _context: &QueryContext,
    ) -> ContextMetadata {
        let sources: HashSet<String> = results.iter().map(|r| r.document.source.clone()).collect();

        let topics: Vec<String> = results
            .iter()
            .flat_map(|r| self.extract_topics(&r.document.content))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let confidence = if results.is_empty() {
            0.0
        } else {
            results.iter().map(|r| r.score).sum::<f64>() / results.len() as f64
        };

        ContextMetadata {
            assembled_at: chrono::Utc::now(),
            source_diversity: sources.len(),
            topic_coverage: topics,
            confidence_score: confidence,
        }
    }

    /// Extract topics from content (simplified implementation)
    fn extract_topics(&self, content: &str) -> Vec<String> {
        // Simple keyword extraction - in production would use NLP
        content
            .split_whitespace()
            .filter(|word| word.len() > 5)
            .take(3)
            .map(|s| s.to_lowercase())
            .collect()
    }

    /// Estimate token count (simplified)
    fn estimate_token_count(&self, text: &str) -> usize {
        // Rough estimation: ~4 characters per token
        text.len() / 4
    }
}

/// Context optimizer for improving relevance and diversity
pub struct ContextOptimizer {
    config: AssemblyConfig,
    diversity_calculator: DiversityCalculator,
}

impl ContextOptimizer {
    pub fn new(config: &AssemblyConfig) -> Self {
        Self {
            config: config.clone(),
            diversity_calculator: DiversityCalculator::new(),
        }
    }

    /// Optimize search results for context assembly
    pub async fn optimize(
        &self,
        results: &[SearchResult],
        context: &QueryContext,
    ) -> Result<Vec<SearchResult>> {
        let mut optimized = results.to_vec();

        // Sort by score initially
        optimized.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply diversity optimization if enabled
        if self.config.enable_diversity {
            optimized = self.apply_diversity_optimization(optimized).await?;
        }

        // Apply temporal prioritization if enabled
        if self.config.prioritize_recent {
            optimized = self.apply_temporal_prioritization(optimized);
        }

        // Apply context-specific filtering
        optimized = self.apply_context_filtering(optimized, context);

        // Ensure we don't exceed token limits (rough estimation)
        optimized = self.apply_length_constraints(optimized);

        debug!(
            "Optimized {} results to {} results",
            results.len(),
            optimized.len()
        );

        Ok(optimized)
    }

    /// Update configuration
    pub fn update_config(&mut self, config: &AssemblyConfig) {
        self.config = config.clone();
    }

    /// Apply diversity optimization to avoid redundant content
    async fn apply_diversity_optimization(
        &self,
        mut results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>> {
        let mut diverse_results = Vec::new();
        let mut selected_content = Vec::new();

        for result in results {
            let is_diverse = selected_content.is_empty()
                || self
                    .diversity_calculator
                    .calculate_diversity(&result.document.content, &selected_content)
                    >= self.config.diversity_threshold;

            if is_diverse {
                selected_content.push(result.document.content.clone());
                diverse_results.push(result);
            }
        }

        Ok(diverse_results)
    }

    /// Apply temporal prioritization (more recent documents preferred)
    fn apply_temporal_prioritization(&self, mut results: Vec<SearchResult>) -> Vec<SearchResult> {
        results.sort_by(|a, b| {
            // Combine recency and relevance score
            let recency_weight = 0.3;
            let relevance_weight = 0.7;

            let now = chrono::Utc::now();
            let a_age = (now - a.document.timestamp).num_hours() as f64;
            let b_age = (now - b.document.timestamp).num_hours() as f64;

            // Newer documents have lower age, so we invert
            let a_recency_score = 1.0 / (1.0 + a_age / 24.0); // Normalize by days
            let b_recency_score = 1.0 / (1.0 + b_age / 24.0);

            let a_combined = relevance_weight * a.score + recency_weight * a_recency_score;
            let b_combined = relevance_weight * b.score + recency_weight * b_recency_score;

            b_combined
                .partial_cmp(&a_combined)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Apply context-specific filtering
    fn apply_context_filtering(
        &self,
        mut results: Vec<SearchResult>,
        context: &QueryContext,
    ) -> Vec<SearchResult> {
        // Filter based on response format preferences
        match context.response_format {
            ResponseFormat::Code => {
                results.retain(|r| {
                    r.document.content.contains("```")
                        || r.document.content.contains("function")
                        || r.document.content.contains("class")
                });
            }
            ResponseFormat::Table => {
                results.retain(|r| {
                    r.document.content.contains("|")
                        || r.document.content.contains("table")
                        || r.document.metadata.contains_key("format")
                            && r.document.metadata["format"] == "table"
                });
            }
            _ => {
                // No specific filtering for other formats
            }
        }

        results
    }

    /// Apply length constraints to fit within token limits
    fn apply_length_constraints(&self, mut results: Vec<SearchResult>) -> Vec<SearchResult> {
        let mut total_tokens = 0;
        let mut constrained_results = Vec::new();

        for result in results {
            let estimated_tokens = result.document.content.len() / 4; // Rough estimation

            if total_tokens + estimated_tokens <= self.config.max_context_tokens {
                total_tokens += estimated_tokens;
                constrained_results.push(result);
            } else {
                break;
            }
        }

        constrained_results
    }
}

/// Diversity calculator for content analysis
pub struct DiversityCalculator {
    stopwords: HashSet<String>,
}

impl DiversityCalculator {
    pub fn new() -> Self {
        let stopwords = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Self { stopwords }
    }

    /// Calculate diversity score between new content and existing content
    pub fn calculate_diversity(&self, new_content: &str, existing_content: &[String]) -> f64 {
        if existing_content.is_empty() {
            return 1.0;
        }

        let new_words = self.extract_significant_words(new_content);
        let mut total_overlap = 0.0;
        let mut total_comparisons = 0;

        for existing in existing_content {
            let existing_words = self.extract_significant_words(existing);
            let overlap = self.calculate_word_overlap(&new_words, &existing_words);
            total_overlap += overlap;
            total_comparisons += 1;
        }

        let average_overlap = total_overlap / total_comparisons as f64;
        1.0 - average_overlap // Higher diversity = lower overlap
    }

    /// Extract significant words (excluding stopwords)
    fn extract_significant_words(&self, content: &str) -> HashSet<String> {
        content
            .split_whitespace()
            .map(|word| {
                word.to_lowercase()
                    .trim_matches(|c: char| !c.is_alphabetic())
                    .to_string()
            })
            .filter(|word| !self.stopwords.contains(word) && word.len() > 2)
            .collect()
    }

    /// Calculate word overlap between two sets
    fn calculate_word_overlap(&self, words1: &HashSet<String>, words2: &HashSet<String>) -> f64 {
        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let intersection_size = words1.intersection(words2).count();
        let union_size = words1.union(words2).count();

        intersection_size as f64 / union_size as f64
    }
}

/// Context formatter for generating human-readable context
pub struct ContextFormatter {
    config: AssemblyConfig,
}

impl ContextFormatter {
    pub fn new(config: &AssemblyConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Format search results into context text
    pub async fn format(&self, results: &[SearchResult], context: &QueryContext) -> Result<String> {
        match context.response_format {
            ResponseFormat::Structured => self.format_structured(results),
            ResponseFormat::Code => self.format_code_focused(results),
            ResponseFormat::Table => self.format_table(results),
            ResponseFormat::List => self.format_list(results),
            _ => self.format_natural_text(results),
        }
    }

    /// Update configuration
    pub fn update_config(&mut self, config: &AssemblyConfig) {
        self.config = config.clone();
    }

    /// Format as natural flowing text
    fn format_natural_text(&self, results: &[SearchResult]) -> Result<String> {
        let mut context = String::new();

        for (i, result) in results.iter().enumerate() {
            if i > 0 {
                context.push_str("\n\n");
            }

            context.push_str(&format!(
                "From {}: {}",
                result.document.source, result.document.content
            ));
        }

        Ok(context)
    }

    /// Format as structured sections
    fn format_structured(&self, results: &[SearchResult]) -> Result<String> {
        let mut context = String::new();

        for (i, result) in results.iter().enumerate() {
            context.push_str(&format!(
                "## Section {} (Source: {}, Relevance: {:.2})\n{}\n\n",
                i + 1,
                result.document.source,
                result.score,
                result.document.content
            ));
        }

        Ok(context)
    }

    /// Format with focus on code content
    fn format_code_focused(&self, results: &[SearchResult]) -> Result<String> {
        let mut context = String::new();

        for result in results {
            if result.document.content.contains("```")
                || result.document.content.contains("function")
                || result.document.content.contains("class")
            {
                context.push_str(&format!(
                    "Code example from {}:\n{}\n\n",
                    result.document.source, result.document.content
                ));
            }
        }

        if context.is_empty() {
            context = "No specific code examples found in the retrieved content.".to_string();
        }

        Ok(context)
    }

    /// Format as a table
    fn format_table(&self, results: &[SearchResult]) -> Result<String> {
        let mut context =
            String::from("| Source | Relevance | Content |\n|--------|-----------|----------|\n");

        for result in results {
            let truncated_content = if result.document.content.len() > 100 {
                format!("{}...", &result.document.content[..100])
            } else {
                result.document.content.clone()
            };

            context.push_str(&format!(
                "| {} | {:.2} | {} |\n",
                result.document.source,
                result.score,
                truncated_content.replace("|", "\\|").replace("\n", " ")
            ));
        }

        Ok(context)
    }

    /// Format as a list
    fn format_list(&self, results: &[SearchResult]) -> Result<String> {
        let mut context = String::new();

        for (i, result) in results.iter().enumerate() {
            context.push_str(&format!(
                "{}. **{}** (Relevance: {:.2})\n   {}\n\n",
                i + 1,
                result.document.source,
                result.score,
                result.document.content
            ));
        }

        Ok(context)
    }
}

impl Default for DiversityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_result(id: &str, content: &str, score: f64) -> SearchResult {
        SearchResult::new(
            RagDocument::new(id.to_string(), content.to_string(), "test".to_string()),
            score,
        )
    }

    #[test]
    fn test_diversity_calculator() {
        let calculator = DiversityCalculator::new();

        let content1 = "machine learning algorithms";
        let content2 = "deep learning neural networks";
        let content3 = "machine learning algorithms"; // identical

        let existing = vec![content1.to_string()];

        let diversity1 = calculator.calculate_diversity(content2, &existing);
        let diversity2 = calculator.calculate_diversity(content3, &existing);

        assert!(diversity1 > diversity2);
    }

    #[test]
    fn test_context_formatter_natural_text() {
        let formatter = ContextFormatter::new(&AssemblyConfig::default());
        let results = vec![
            create_test_result("doc1", "Content 1", 0.9),
            create_test_result("doc2", "Content 2", 0.8),
        ];

        let formatted = formatter.format_natural_text(&results).unwrap();
        assert!(formatted.contains("Content 1"));
        assert!(formatted.contains("Content 2"));
    }

    #[test]
    fn test_context_formatter_structured() {
        let formatter = ContextFormatter::new(&AssemblyConfig::default());
        let results = vec![create_test_result("doc1", "Content 1", 0.9)];

        let formatted = formatter.format_structured(&results).unwrap();
        assert!(formatted.contains("## Section 1"));
        assert!(formatted.contains("Relevance: 0.90"));
    }

    #[test]
    fn test_context_assembler_creation() {
        let rag_config = super::super::RAGConfig::default();
        let assembler = ContextAssembler::new(&rag_config);
        assert_eq!(
            assembler.config.max_context_tokens,
            rag_config.max_context_length
        );
    }
}
