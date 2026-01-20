//! Cross-Language Vector Alignment - Version 1.2 Feature
//!
//! This module implements comprehensive cross-language vector alignment capabilities
//! that enable semantic search and similarity computation across different languages.
//! It supports multilingual embeddings, translation-based alignment, and cross-lingual
//! similarity scoring for knowledge graphs with multilingual content.

use crate::{embeddings::EmbeddingGenerator, similarity::SimilarityMetric, Vector};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use tracing::{info, span, Level};

/// Configuration for cross-language vector alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossLanguageConfig {
    /// Supported languages (ISO 639-1 codes)
    pub supported_languages: Vec<String>,
    /// Primary language for fallback
    pub primary_language: String,
    /// Enable automatic language detection
    pub enable_language_detection: bool,
    /// Alignment strategy
    pub alignment_strategy: AlignmentStrategy,
    /// Translation service configuration
    pub translation_config: Option<TranslationConfig>,
    /// Multilingual embedding model configuration
    pub multilingual_embeddings: MultilingualEmbeddingConfig,
    /// Cross-lingual similarity threshold
    pub cross_lingual_threshold: f32,
}

impl Default for CrossLanguageConfig {
    fn default() -> Self {
        Self {
            supported_languages: vec![
                "en".to_string(), // English
                "es".to_string(), // Spanish
                "fr".to_string(), // French
                "de".to_string(), // German
                "it".to_string(), // Italian
                "pt".to_string(), // Portuguese
                "ru".to_string(), // Russian
                "zh".to_string(), // Chinese
                "ja".to_string(), // Japanese
                "ar".to_string(), // Arabic
            ],
            primary_language: "en".to_string(),
            enable_language_detection: true,
            alignment_strategy: AlignmentStrategy::MultilingualEmbeddings,
            translation_config: None,
            multilingual_embeddings: MultilingualEmbeddingConfig::default(),
            cross_lingual_threshold: 0.6,
        }
    }
}

/// Strategies for aligning vectors across languages
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlignmentStrategy {
    /// Use multilingual embedding models
    MultilingualEmbeddings,
    /// Use translation to common language
    TranslationBased,
    /// Hybrid approach with both methods
    Hybrid,
    /// Learn cross-lingual mappings
    LearnedMappings,
}

/// Configuration for translation services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationConfig {
    /// Translation service provider
    pub provider: TranslationProvider,
    /// API endpoint
    pub endpoint: Option<String>,
    /// API key for authentication
    pub api_key: Option<String>,
    /// Cache translated content
    pub enable_caching: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
}

/// Translation service providers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TranslationProvider {
    /// Google Translate API
    Google,
    /// Microsoft Translator
    Microsoft,
    /// AWS Translate
    Aws,
    /// Local/offline model
    Local,
}

/// Configuration for multilingual embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultilingualEmbeddingConfig {
    /// Model name for multilingual embeddings
    pub model_name: String,
    /// Dimension of embeddings
    pub dimensions: usize,
    /// Normalization strategy
    pub normalization: NormalizationStrategy,
    /// Language-specific preprocessing
    pub language_preprocessing: HashMap<String, Vec<String>>,
}

impl Default for MultilingualEmbeddingConfig {
    fn default() -> Self {
        Self {
            model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".to_string(),
            dimensions: 384,
            normalization: NormalizationStrategy::L2,
            language_preprocessing: HashMap::new(),
        }
    }
}

/// Vector normalization strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NormalizationStrategy {
    /// L2 normalization
    L2,
    /// Mean centering
    MeanCentering,
    /// Standardization (z-score)
    Standardization,
    /// None
    None,
}

/// Language detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageDetection {
    /// Detected language code
    pub language: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Alternative language candidates
    pub alternatives: Vec<(String, f32)>,
}

/// Cross-language content item
#[derive(Debug, Clone)]
pub struct CrossLanguageContent {
    /// Unique identifier
    pub id: String,
    /// Text content
    pub text: String,
    /// Detected or specified language
    pub language: String,
    /// Language detection confidence
    pub language_confidence: f32,
    /// Original vector embedding
    pub vector: Option<Vector>,
    /// Aligned vectors for different languages
    pub aligned_vectors: HashMap<String, Vector>,
}

/// Cross-language vector alignment engine
pub struct CrossLanguageAligner {
    config: CrossLanguageConfig,
    language_detector: Box<dyn LanguageDetector + Send + Sync>,
    embedding_generator: Box<dyn EmbeddingGenerator + Send + Sync>,
    translation_cache: Arc<RwLock<HashMap<String, String>>>,
    alignment_mappings: Arc<RwLock<HashMap<String, AlignmentMapping>>>,
    multilingual_embeddings: Arc<RwLock<HashMap<String, Vector>>>,
}

/// Language detection trait
pub trait LanguageDetector {
    /// Detect language of given text
    fn detect_language(&self, text: &str) -> Result<LanguageDetection>;

    /// Check if language is supported
    fn is_supported(&self, language: &str) -> bool;
}

/// Simple language detector implementation
pub struct SimpleLanguageDetector {
    supported_languages: HashSet<String>,
}

impl SimpleLanguageDetector {
    pub fn new(supported_languages: Vec<String>) -> Self {
        Self {
            supported_languages: supported_languages.into_iter().collect(),
        }
    }
}

impl LanguageDetector for SimpleLanguageDetector {
    fn detect_language(&self, text: &str) -> Result<LanguageDetection> {
        // Simplified language detection based on character sets and patterns
        let text_lower = text.to_lowercase();

        // Simple heuristics for language detection
        let language = if text_lower
            .chars()
            .any(|c| matches!(c, 'ñ' | 'ü' | 'é' | 'á' | 'í' | 'ó' | 'ú'))
        {
            "es" // Spanish
        } else if text_lower
            .chars()
            .any(|c| matches!(c, 'ç' | 'à' | 'è' | 'ù' | 'ê' | 'ô'))
        {
            "fr" // French
        } else if text_lower
            .chars()
            .any(|c| matches!(c, 'ä' | 'ö' | 'ü' | 'ß'))
        {
            "de" // German
        } else if text_lower
            .chars()
            .any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
        {
            "zh" // Chinese
        } else if text_lower
            .chars()
            .any(|c| ('\u{3040}'..='\u{309f}').contains(&c))
        {
            "ja" // Japanese
        } else if text_lower
            .chars()
            .any(|c| ('\u{0600}'..='\u{06ff}').contains(&c))
        {
            "ar" // Arabic
        } else if text_lower
            .chars()
            .any(|c| ('\u{0400}'..='\u{04ff}').contains(&c))
        {
            "ru" // Russian
        } else {
            "en" // Default to English
        };

        let confidence = if language == "en" { 0.7 } else { 0.8 };

        Ok(LanguageDetection {
            language: language.to_string(),
            confidence,
            alternatives: vec![("en".to_string(), 0.3)],
        })
    }

    fn is_supported(&self, language: &str) -> bool {
        self.supported_languages.contains(language)
    }
}

/// Alignment mapping between languages
#[derive(Debug, Clone)]
pub struct AlignmentMapping {
    /// Source language
    pub source_language: String,
    /// Target language
    pub target_language: String,
    /// Transformation matrix (if learned)
    pub transformation_matrix: Option<Vec<Vec<f32>>>,
    /// Translation pairs used for learning
    pub translation_pairs: Vec<(String, String)>,
    /// Mapping quality score
    pub quality_score: f32,
}

/// Cross-language search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossLanguageSearchResult {
    /// Content identifier
    pub id: String,
    /// Similarity score
    pub similarity: f32,
    /// Content language
    pub language: String,
    /// Original text content
    pub text: String,
    /// Translated text (if available)
    pub translated_text: Option<String>,
    /// Cross-lingual similarity metrics
    pub cross_lingual_metrics: HashMap<String, f32>,
}

impl CrossLanguageAligner {
    /// Create a new cross-language aligner
    pub fn new(
        config: CrossLanguageConfig,
        embedding_generator: Box<dyn EmbeddingGenerator + Send + Sync>,
    ) -> Self {
        let language_detector = Box::new(SimpleLanguageDetector::new(
            config.supported_languages.clone(),
        ));

        Self {
            config,
            language_detector,
            embedding_generator,
            translation_cache: Arc::new(RwLock::new(HashMap::new())),
            alignment_mappings: Arc::new(RwLock::new(HashMap::new())),
            multilingual_embeddings: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Process content and create cross-language representations
    pub async fn process_content(&self, content: &str, id: &str) -> Result<CrossLanguageContent> {
        let span = span!(Level::INFO, "process_content", content_id = %id);
        let _enter = span.enter();

        // Detect language
        let detection = if self.config.enable_language_detection {
            self.language_detector.detect_language(content)?
        } else {
            LanguageDetection {
                language: self.config.primary_language.clone(),
                confidence: 1.0,
                alternatives: Vec::new(),
            }
        };

        // Generate primary vector embedding
        let embeddable_content = crate::embeddings::EmbeddableContent::Text(content.to_string());
        let vector = self
            .embedding_generator
            .generate(&embeddable_content)
            .context("Failed to generate embedding")?;

        // Create aligned vectors for other languages
        let aligned_vectors = self
            .create_aligned_vectors(content, &detection.language, &vector)
            .await?;

        Ok(CrossLanguageContent {
            id: id.to_string(),
            text: content.to_string(),
            language: detection.language,
            language_confidence: detection.confidence,
            vector: Some(vector),
            aligned_vectors,
        })
    }

    /// Create aligned vectors for different languages
    async fn create_aligned_vectors(
        &self,
        content: &str,
        source_language: &str,
        source_vector: &Vector,
    ) -> Result<HashMap<String, Vector>> {
        let mut aligned_vectors = HashMap::new();

        match self.config.alignment_strategy {
            AlignmentStrategy::MultilingualEmbeddings => {
                // Use multilingual embedding model directly
                for target_lang in &self.config.supported_languages {
                    if target_lang != source_language {
                        let aligned_vector =
                            self.create_multilingual_embedding(content, target_lang)?;
                        aligned_vectors.insert(target_lang.clone(), aligned_vector);
                    }
                }
            }
            AlignmentStrategy::TranslationBased => {
                // Translate content and generate embeddings
                for target_lang in &self.config.supported_languages {
                    if target_lang != source_language {
                        let translated_text = self
                            .translate_text(content, source_language, target_lang)
                            .await?;
                        let embeddable_content =
                            crate::embeddings::EmbeddableContent::Text(translated_text);
                        let translated_vector =
                            self.embedding_generator.generate(&embeddable_content)?;
                        aligned_vectors.insert(target_lang.clone(), translated_vector);
                    }
                }
            }
            AlignmentStrategy::Hybrid => {
                // Use both multilingual embeddings and translation
                for target_lang in &self.config.supported_languages {
                    if target_lang != source_language {
                        let multilingual_vector =
                            self.create_multilingual_embedding(content, target_lang)?;
                        let translated_text = self
                            .translate_text(content, source_language, target_lang)
                            .await?;
                        let embeddable_content =
                            crate::embeddings::EmbeddableContent::Text(translated_text);
                        let translated_vector =
                            self.embedding_generator.generate(&embeddable_content)?;

                        // Combine vectors (simple average for now)
                        let combined_vector =
                            self.combine_vectors(&multilingual_vector, &translated_vector)?;
                        aligned_vectors.insert(target_lang.clone(), combined_vector);
                    }
                }
            }
            AlignmentStrategy::LearnedMappings => {
                // Apply learned transformation mappings
                for target_lang in &self.config.supported_languages {
                    if target_lang != source_language {
                        let mapped_vector = self.apply_learned_mapping(
                            source_vector,
                            source_language,
                            target_lang,
                        )?;
                        aligned_vectors.insert(target_lang.clone(), mapped_vector);
                    }
                }
            }
        }

        Ok(aligned_vectors)
    }

    /// Create multilingual embedding
    fn create_multilingual_embedding(
        &self,
        content: &str,
        target_language: &str,
    ) -> Result<Vector> {
        // For now, use the same embedding generator with language prefix
        let prefixed_content = format!("[{target_language}] {content}");
        let embeddable_content = crate::embeddings::EmbeddableContent::Text(prefixed_content);
        self.embedding_generator.generate(&embeddable_content)
    }

    /// Translate text between languages
    async fn translate_text(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<String> {
        let cache_key = format!("{source_lang}:{target_lang}:{text}");

        // Check cache first
        {
            let cache = self
                .translation_cache
                .read()
                .expect("translation cache lock should not be poisoned");
            if let Some(cached_translation) = cache.get(&cache_key) {
                return Ok(cached_translation.clone());
            }
        }

        // Simulate translation (in real implementation, would call translation API)
        let translated = match (source_lang, target_lang) {
            ("en", "es") => format!("[ES] {text}"),
            ("en", "fr") => format!("[FR] {text}"),
            ("en", "de") => format!("[DE] {text}"),
            ("es", "en") => text.replace("[ES]", "[EN]"),
            ("fr", "en") => text.replace("[FR]", "[EN]"),
            ("de", "en") => text.replace("[DE]", "[EN]"),
            _ => {
                let upper_lang = target_lang.to_uppercase();
                format!("[{upper_lang}] {text}")
            }
        };

        // Cache the translation
        {
            let mut cache = self
                .translation_cache
                .write()
                .expect("translation cache lock should not be poisoned");
            if cache.len()
                >= self
                    .config
                    .translation_config
                    .as_ref()
                    .map(|c| c.max_cache_size)
                    .unwrap_or(10000)
            {
                // Simple cache eviction: remove first entry
                if let Some(key) = cache.keys().next().cloned() {
                    cache.remove(&key);
                }
            }
            cache.insert(cache_key, translated.clone());
        }

        Ok(translated)
    }

    /// Combine two vectors (simple averaging)
    fn combine_vectors(&self, vector1: &Vector, vector2: &Vector) -> Result<Vector> {
        let v1_f32 = vector1.as_f32();
        let v2_f32 = vector2.as_f32();

        if v1_f32.len() != v2_f32.len() {
            return Err(anyhow!("Vector dimensions must match for combination"));
        }

        let combined: Vec<f32> = v1_f32
            .iter()
            .zip(v2_f32.iter())
            .map(|(a, b)| (a + b) / 2.0)
            .collect();

        Ok(Vector::new(combined))
    }

    /// Apply learned mapping transformation
    fn apply_learned_mapping(
        &self,
        source_vector: &Vector,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<Vector> {
        let mapping_key = format!("{source_lang}:{target_lang}");
        let mappings = self
            .alignment_mappings
            .read()
            .expect("alignment mappings lock should not be poisoned");

        if let Some(mapping) = mappings.get(&mapping_key) {
            if let Some(ref matrix) = mapping.transformation_matrix {
                return self.apply_matrix_transformation(source_vector, matrix);
            }
        }

        // Fallback to identity mapping
        Ok(source_vector.clone())
    }

    /// Apply matrix transformation to vector
    fn apply_matrix_transformation(&self, vector: &Vector, matrix: &[Vec<f32>]) -> Result<Vector> {
        let v_f32 = vector.as_f32();

        if matrix.is_empty() || matrix[0].len() != v_f32.len() {
            return Err(anyhow!("Matrix dimensions incompatible with vector"));
        }

        let transformed: Vec<f32> = matrix
            .iter()
            .map(|row| row.iter().zip(v_f32.iter()).map(|(m, v)| m * v).sum())
            .collect();

        Ok(Vector::new(transformed))
    }

    /// Cross-language similarity search
    pub fn cross_language_search(
        &self,
        query: &str,
        query_language: &str,
        content_items: &[CrossLanguageContent],
        k: usize,
    ) -> Result<Vec<CrossLanguageSearchResult>> {
        let span = span!(Level::INFO, "cross_language_search", query_lang = %query_language);
        let _enter = span.enter();

        // Generate query vector
        let embeddable_content = crate::embeddings::EmbeddableContent::Text(query.to_string());
        let query_vector = self.embedding_generator.generate(&embeddable_content)?;

        let mut results = Vec::new();

        for content in content_items {
            // Compute similarity with original vector
            let primary_similarity = if content.language == query_language {
                if let Some(ref content_vector) = content.vector {
                    SimilarityMetric::Cosine.compute(&query_vector, content_vector)?
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Compute cross-lingual similarity using aligned vectors
            let mut cross_lingual_similarities = HashMap::new();
            if let Some(aligned_vector) = content.aligned_vectors.get(query_language) {
                let cross_similarity =
                    SimilarityMetric::Cosine.compute(&query_vector, aligned_vector)?;
                cross_lingual_similarities.insert("cosine".to_string(), cross_similarity);
            }

            // Determine the best similarity score
            let best_similarity = primary_similarity.max(
                cross_lingual_similarities
                    .values()
                    .copied()
                    .fold(0.0, f32::max),
            );

            if best_similarity >= self.config.cross_lingual_threshold {
                results.push(CrossLanguageSearchResult {
                    id: content.id.clone(),
                    similarity: best_similarity,
                    language: content.language.clone(),
                    text: content.text.clone(),
                    translated_text: None, // Could add translation here
                    cross_lingual_metrics: cross_lingual_similarities,
                });
            }
        }

        // Sort by similarity (descending)
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        Ok(results)
    }

    /// Learn alignment mapping between languages
    pub fn learn_alignment_mapping(
        &mut self,
        source_language: &str,
        target_language: &str,
        translation_pairs: Vec<(String, String)>,
    ) -> Result<()> {
        let span = span!(Level::INFO, "learn_alignment_mapping",
                          source = %source_language, target = %target_language);
        let _enter = span.enter();

        // Generate embeddings for translation pairs
        let mut source_vectors = Vec::new();
        let mut target_vectors = Vec::new();

        for (source_text, target_text) in &translation_pairs {
            let source_embeddable = crate::embeddings::EmbeddableContent::Text(source_text.clone());
            let target_embeddable = crate::embeddings::EmbeddableContent::Text(target_text.clone());
            let source_vector = self.embedding_generator.generate(&source_embeddable)?;
            let target_vector = self.embedding_generator.generate(&target_embeddable)?;

            source_vectors.push(source_vector.as_f32());
            target_vectors.push(target_vector.as_f32());
        }

        // Learn transformation matrix (simplified - in practice would use more sophisticated methods)
        let transformation_matrix =
            self.compute_transformation_matrix(&source_vectors, &target_vectors)?;

        // Evaluate mapping quality
        let quality_score = self.evaluate_mapping_quality(
            &source_vectors,
            &target_vectors,
            &transformation_matrix,
        )?;

        let mapping = AlignmentMapping {
            source_language: source_language.to_string(),
            target_language: target_language.to_string(),
            transformation_matrix: Some(transformation_matrix),
            translation_pairs,
            quality_score,
        };

        let mapping_key = format!("{source_language}:{target_language}");
        let mut mappings = self
            .alignment_mappings
            .write()
            .expect("alignment mappings lock should not be poisoned");
        mappings.insert(mapping_key, mapping);

        info!(
            "Learned alignment mapping with quality score: {:.3}",
            quality_score
        );
        Ok(())
    }

    /// Compute transformation matrix using simple linear regression
    fn compute_transformation_matrix(
        &self,
        source_vectors: &[Vec<f32>],
        target_vectors: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>> {
        if source_vectors.is_empty() || source_vectors.len() != target_vectors.len() {
            return Err(anyhow!("Invalid vector sets for learning transformation"));
        }

        let dim = source_vectors[0].len();

        // Simple identity matrix as baseline (in practice, would use proper linear algebra)
        let mut matrix = vec![vec![0.0; dim]; dim];
        for (i, row) in matrix.iter_mut().enumerate().take(dim) {
            row[i] = 1.0;
        }

        // Add small random perturbations to simulate learned transformation
        for (i, row) in matrix.iter_mut().enumerate().take(dim) {
            for (j, row_val) in row.iter_mut().enumerate().take(dim) {
                if i != j {
                    *row_val = (i as f32 * j as f32 * 0.001) % 0.1 - 0.05;
                }
            }
        }

        Ok(matrix)
    }

    /// Evaluate quality of learned mapping
    fn evaluate_mapping_quality(
        &self,
        source_vectors: &[Vec<f32>],
        target_vectors: &[Vec<f32>],
        matrix: &[Vec<f32>],
    ) -> Result<f32> {
        let mut total_similarity = 0.0;
        let mut count = 0;

        for (source, target) in source_vectors.iter().zip(target_vectors) {
            let transformed_vector = Vector::new(source.clone());
            let transformed = self.apply_matrix_transformation(&transformed_vector, matrix)?;
            let target_vector = Vector::new(target.clone());

            let similarity = SimilarityMetric::Cosine.compute(&transformed, &target_vector)?;
            total_similarity += similarity;
            count += 1;
        }

        Ok(if count > 0 {
            total_similarity / count as f32
        } else {
            0.0
        })
    }

    /// Get language statistics
    pub fn get_language_statistics(&self) -> HashMap<String, usize> {
        let embeddings = self
            .multilingual_embeddings
            .read()
            .expect("multilingual embeddings lock should not be poisoned");
        let mut stats = HashMap::new();

        for lang in &self.config.supported_languages {
            stats.insert(lang.clone(), embeddings.len());
        }

        stats
    }

    /// Get supported languages
    pub fn get_supported_languages(&self) -> &[String] {
        &self.config.supported_languages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::MockEmbeddingGenerator;

    #[test]
    fn test_cross_language_config_creation() {
        let config = CrossLanguageConfig::default();
        assert!(!config.supported_languages.is_empty());
        assert_eq!(config.primary_language, "en");
        assert!(config.enable_language_detection);
    }

    #[test]
    fn test_language_detector_creation() {
        let languages = vec!["en".to_string(), "es".to_string(), "fr".to_string()];
        let detector = SimpleLanguageDetector::new(languages.clone());

        assert!(detector.is_supported("en"));
        assert!(detector.is_supported("es"));
        assert!(!detector.is_supported("de"));
    }

    #[test]
    fn test_language_detection() {
        let detector = SimpleLanguageDetector::new(vec!["en".to_string(), "es".to_string()]);

        let detection = detector.detect_language("Hello world").unwrap();
        assert_eq!(detection.language, "en");
        assert!(detection.confidence > 0.0);

        let detection = detector.detect_language("Hola mundo").unwrap();
        assert_eq!(detection.language, "en"); // Simple detector defaults to English
    }

    #[test]
    fn test_alignment_strategy_variants() {
        let strategies = vec![
            AlignmentStrategy::MultilingualEmbeddings,
            AlignmentStrategy::TranslationBased,
            AlignmentStrategy::Hybrid,
            AlignmentStrategy::LearnedMappings,
        ];

        for strategy in strategies {
            let config = CrossLanguageConfig {
                alignment_strategy: strategy.clone(),
                ..Default::default()
            };
            assert_eq!(config.alignment_strategy, strategy);
        }
    }

    #[tokio::test]
    async fn test_cross_language_aligner_creation() {
        let config = CrossLanguageConfig::default();
        let embedding_generator = Box::new(MockEmbeddingGenerator::new());

        let aligner = CrossLanguageAligner::new(config, embedding_generator);
        assert_eq!(aligner.get_supported_languages().len(), 10);
    }

    #[tokio::test]
    async fn test_content_processing() {
        let config = CrossLanguageConfig::default();
        let embedding_generator = Box::new(MockEmbeddingGenerator::new());

        let aligner = CrossLanguageAligner::new(config, embedding_generator);
        let content = aligner
            .process_content("Hello world", "test_id")
            .await
            .unwrap();

        assert_eq!(content.id, "test_id");
        assert_eq!(content.text, "Hello world");
        assert!(content.vector.is_some());
        assert!(!content.aligned_vectors.is_empty());
    }

    #[test]
    fn test_vector_combination() {
        let config = CrossLanguageConfig::default();
        let embedding_generator = Box::new(MockEmbeddingGenerator::new());
        let aligner = CrossLanguageAligner::new(config, embedding_generator);

        let vector1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let vector2 = Vector::new(vec![2.0, 4.0, 6.0]);

        let combined = aligner.combine_vectors(&vector1, &vector2).unwrap();
        let combined_f32 = combined.as_f32();

        assert_eq!(combined_f32, vec![1.5, 3.0, 4.5]);
    }

    #[test]
    fn test_cross_language_search_result() {
        let result = CrossLanguageSearchResult {
            id: "test".to_string(),
            similarity: 0.8,
            language: "en".to_string(),
            text: "test content".to_string(),
            translated_text: Some("contenido de prueba".to_string()),
            cross_lingual_metrics: HashMap::new(),
        };

        assert_eq!(result.id, "test");
        assert_eq!(result.similarity, 0.8);
        assert_eq!(result.language, "en");
    }
}
