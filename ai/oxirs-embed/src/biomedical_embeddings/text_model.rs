//! Module for biomedical embeddings

use crate::{ModelConfig, ModelStats, TrainingStats};
use anyhow::Result;
use chrono::Utc;
use scirs2_core::ndarray_ext::Array1;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Specialized text embedding models for domain-specific applications
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpecializedTextModel {
    /// SciBERT for scientific literature
    SciBERT,
    /// CodeBERT for code and programming
    CodeBERT,
    /// BioBERT for biomedical literature
    BioBERT,
    /// LegalBERT for legal documents
    LegalBERT,
    /// FinBERT for financial texts
    FinBERT,
    /// ClinicalBERT for clinical notes
    ClinicalBERT,
    /// ChemBERT for chemical compounds
    ChemBERT,
}

impl SpecializedTextModel {
    /// Get the model name for loading pre-trained weights
    pub fn model_name(&self) -> &'static str {
        match self {
            SpecializedTextModel::SciBERT => "allenai/scibert_scivocab_uncased",
            SpecializedTextModel::CodeBERT => "microsoft/codebert-base",
            SpecializedTextModel::BioBERT => "dmis-lab/biobert-base-cased-v1.2",
            SpecializedTextModel::LegalBERT => "nlpaueb/legal-bert-base-uncased",
            SpecializedTextModel::FinBERT => "ProsusAI/finbert",
            SpecializedTextModel::ClinicalBERT => "emilyalsentzer/Bio_ClinicalBERT",
            SpecializedTextModel::ChemBERT => "seyonec/ChemBERTa-zinc-base-v1",
        }
    }

    /// Get the vocabulary size for the model
    pub fn vocab_size(&self) -> usize {
        match self {
            SpecializedTextModel::SciBERT => 31090,
            SpecializedTextModel::CodeBERT => 50265,
            SpecializedTextModel::BioBERT => 28996,
            SpecializedTextModel::LegalBERT => 30522,
            SpecializedTextModel::FinBERT => 30522,
            SpecializedTextModel::ClinicalBERT => 28996,
            SpecializedTextModel::ChemBERT => 600,
        }
    }

    /// Get the default embedding dimension
    pub fn embedding_dim(&self) -> usize {
        match self {
            SpecializedTextModel::SciBERT => 768,
            SpecializedTextModel::CodeBERT => 768,
            SpecializedTextModel::BioBERT => 768,
            SpecializedTextModel::LegalBERT => 768,
            SpecializedTextModel::FinBERT => 768,
            SpecializedTextModel::ClinicalBERT => 768,
            SpecializedTextModel::ChemBERT => 384,
        }
    }

    /// Get the maximum sequence length
    pub fn max_sequence_length(&self) -> usize {
        match self {
            SpecializedTextModel::SciBERT => 512,
            SpecializedTextModel::CodeBERT => 512,
            SpecializedTextModel::BioBERT => 512,
            SpecializedTextModel::LegalBERT => 512,
            SpecializedTextModel::FinBERT => 512,
            SpecializedTextModel::ClinicalBERT => 512,
            SpecializedTextModel::ChemBERT => 512,
        }
    }

    /// Get domain-specific preprocessing rules
    pub fn get_preprocessing_rules(&self) -> Vec<PreprocessingRule> {
        match self {
            SpecializedTextModel::SciBERT => vec![
                PreprocessingRule::NormalizeScientificNotation,
                PreprocessingRule::ExpandAbbreviations,
                PreprocessingRule::HandleChemicalFormulas,
                PreprocessingRule::PreserveCitations,
            ],
            SpecializedTextModel::CodeBERT => vec![
                PreprocessingRule::PreserveCodeTokens,
                PreprocessingRule::HandleCamelCase,
                PreprocessingRule::NormalizeWhitespace,
                PreprocessingRule::PreservePunctuation,
            ],
            SpecializedTextModel::BioBERT => vec![
                PreprocessingRule::NormalizeMedicalTerms,
                PreprocessingRule::HandleGeneNames,
                PreprocessingRule::ExpandMedicalAbbreviations,
                PreprocessingRule::PreserveDosages,
            ],
            SpecializedTextModel::LegalBERT => vec![
                PreprocessingRule::PreserveLegalCitations,
                PreprocessingRule::HandleLegalTerms,
                PreprocessingRule::NormalizeCaseReferences,
            ],
            SpecializedTextModel::FinBERT => vec![
                PreprocessingRule::NormalizeFinancialTerms,
                PreprocessingRule::HandleCurrencySymbols,
                PreprocessingRule::PreservePercentages,
            ],
            SpecializedTextModel::ClinicalBERT => vec![
                PreprocessingRule::NormalizeMedicalTerms,
                PreprocessingRule::HandleMedicalAbbreviations,
                PreprocessingRule::PreserveDosages,
                PreprocessingRule::NormalizeTimestamps,
            ],
            SpecializedTextModel::ChemBERT => vec![
                PreprocessingRule::HandleChemicalFormulas,
                PreprocessingRule::PreserveMolecularStructures,
                PreprocessingRule::NormalizeChemicalNames,
            ],
        }
    }
}

/// Preprocessing rules for specialized text models
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PreprocessingRule {
    /// Normalize scientific notation (e.g., 1.23e-4)
    NormalizeScientificNotation,
    /// Expand domain-specific abbreviations
    ExpandAbbreviations,
    /// Handle chemical formulas and compounds
    HandleChemicalFormulas,
    /// Preserve citation formats
    PreserveCitations,
    /// Preserve code tokens and keywords
    PreserveCodeTokens,
    /// Handle camelCase and snake_case
    HandleCamelCase,
    /// Normalize whitespace patterns
    NormalizeWhitespace,
    /// Preserve punctuation in code
    PreservePunctuation,
    /// Normalize medical terminology
    NormalizeMedicalTerms,
    /// Handle gene and protein names
    HandleGeneNames,
    /// Expand medical abbreviations
    ExpandMedicalAbbreviations,
    /// Preserve dosage information
    PreserveDosages,
    /// Preserve legal citations
    PreserveLegalCitations,
    /// Handle legal terminology
    HandleLegalTerms,
    /// Normalize case references
    NormalizeCaseReferences,
    /// Normalize financial terms
    NormalizeFinancialTerms,
    /// Handle currency symbols
    HandleCurrencySymbols,
    /// Preserve percentage values
    PreservePercentages,
    /// Handle medical abbreviations
    HandleMedicalAbbreviations,
    /// Normalize timestamps
    NormalizeTimestamps,
    /// Preserve molecular structures
    PreserveMolecularStructures,
    /// Normalize chemical names
    NormalizeChemicalNames,
}

/// Configuration for specialized text embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecializedTextConfig {
    pub model_type: SpecializedTextModel,
    pub base_config: ModelConfig,
    /// Fine-tuning configuration
    pub fine_tune_config: FineTuningConfig,
    /// Preprocessing configuration
    pub preprocessing_enabled: bool,
    /// Domain-specific vocabulary augmentation
    pub vocab_augmentation: bool,
    /// Use domain-specific pre-training
    pub domain_pretraining: bool,
}

/// Fine-tuning configuration for specialized models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningConfig {
    /// Learning rate for fine-tuning
    pub learning_rate: f64,
    /// Number of fine-tuning epochs
    pub epochs: usize,
    /// Freeze base model layers
    pub freeze_base_layers: bool,
    /// Number of layers to freeze
    pub frozen_layers: usize,
    /// Use gradual unfreezing
    pub gradual_unfreezing: bool,
    /// Discriminative fine-tuning rates
    pub discriminative_rates: Vec<f64>,
}

impl Default for FineTuningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 2e-5,
            epochs: 3,
            freeze_base_layers: false,
            frozen_layers: 0,
            gradual_unfreezing: false,
            discriminative_rates: vec![],
        }
    }
}

impl Default for SpecializedTextConfig {
    fn default() -> Self {
        Self {
            model_type: SpecializedTextModel::BioBERT,
            base_config: ModelConfig::default(),
            fine_tune_config: FineTuningConfig::default(),
            preprocessing_enabled: true,
            vocab_augmentation: false,
            domain_pretraining: false,
        }
    }
}

/// Specialized text embedding processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecializedTextEmbedding {
    pub config: SpecializedTextConfig,
    pub model_id: Uuid,
    /// Text embeddings cache
    pub text_embeddings: HashMap<String, Array1<f32>>,
    /// Domain-specific vocabulary
    pub domain_vocab: HashSet<String>,
    /// Preprocessing pipeline
    pub preprocessing_rules: Vec<PreprocessingRule>,
    /// Training statistics
    pub training_stats: TrainingStats,
    /// Model statistics
    pub model_stats: ModelStats,
    pub is_trained: bool,
}

impl SpecializedTextEmbedding {
    /// Create new specialized text embedding model
    pub fn new(config: SpecializedTextConfig) -> Self {
        let model_id = Uuid::new_v4();
        let now = Utc::now();
        let preprocessing_rules = config.model_type.get_preprocessing_rules();

        Self {
            model_id,
            text_embeddings: HashMap::new(),
            domain_vocab: HashSet::new(),
            preprocessing_rules,
            training_stats: TrainingStats::default(),
            model_stats: ModelStats {
                num_entities: 0,
                num_relations: 0,
                num_triples: 0,
                dimensions: config.model_type.embedding_dim(),
                is_trained: false,
                model_type: format!("SpecializedText_{:?}", config.model_type),
                creation_time: now,
                last_training_time: None,
            },
            is_trained: false,
            config,
        }
    }

    /// Create SciBERT configuration
    pub fn scibert_config() -> SpecializedTextConfig {
        SpecializedTextConfig {
            model_type: SpecializedTextModel::SciBERT,
            base_config: ModelConfig::default().with_dimensions(768),
            fine_tune_config: FineTuningConfig::default(),
            preprocessing_enabled: true,
            vocab_augmentation: true,
            domain_pretraining: true,
        }
    }

    /// Create CodeBERT configuration
    pub fn codebert_config() -> SpecializedTextConfig {
        SpecializedTextConfig {
            model_type: SpecializedTextModel::CodeBERT,
            base_config: ModelConfig::default().with_dimensions(768),
            fine_tune_config: FineTuningConfig::default(),
            preprocessing_enabled: true,
            vocab_augmentation: false,
            domain_pretraining: true,
        }
    }

    /// Create BioBERT configuration
    pub fn biobert_config() -> SpecializedTextConfig {
        SpecializedTextConfig {
            model_type: SpecializedTextModel::BioBERT,
            base_config: ModelConfig::default().with_dimensions(768),
            fine_tune_config: FineTuningConfig {
                learning_rate: 1e-5,
                epochs: 5,
                freeze_base_layers: true,
                frozen_layers: 6,
                gradual_unfreezing: true,
                discriminative_rates: vec![1e-6, 5e-6, 1e-5, 2e-5],
            },
            preprocessing_enabled: true,
            vocab_augmentation: true,
            domain_pretraining: true,
        }
    }

    /// Preprocess text according to domain-specific rules
    pub fn preprocess_text(&self, text: &str) -> Result<String> {
        if !self.config.preprocessing_enabled {
            return Ok(text.to_string());
        }

        let mut processed = text.to_string();

        for rule in &self.preprocessing_rules {
            processed = self.apply_preprocessing_rule(&processed, rule)?;
        }

        Ok(processed)
    }

    /// Apply a specific preprocessing rule
    fn apply_preprocessing_rule(&self, text: &str, rule: &PreprocessingRule) -> Result<String> {
        match rule {
            PreprocessingRule::NormalizeScientificNotation => {
                // Convert scientific notation to normalized form (simplified)
                Ok(text
                    .replace("E+", "e+")
                    .replace("E-", "e-")
                    .replace("E", "e"))
            }
            PreprocessingRule::HandleChemicalFormulas => {
                // Preserve chemical formulas by adding special tokens (simplified)
                Ok(text.replace("H2O", "[CHEM]H2O[/CHEM]"))
            }
            PreprocessingRule::HandleCamelCase => {
                // Split camelCase into separate tokens (simplified)
                let mut result = String::new();
                let mut chars = text.chars().peekable();
                while let Some(c) = chars.next() {
                    result.push(c);
                    if c.is_lowercase() && chars.peek().is_some_and(|&next| next.is_uppercase()) {
                        result.push(' ');
                    }
                }
                Ok(result)
            }
            PreprocessingRule::NormalizeMedicalTerms => {
                // Normalize common medical abbreviations
                let mut result = text.to_string();
                let replacements = [
                    ("mg/kg", "milligrams per kilogram"),
                    ("q.d.", "once daily"),
                    ("b.i.d.", "twice daily"),
                    ("t.i.d.", "three times daily"),
                    ("q.i.d.", "four times daily"),
                ];

                for (abbrev, expansion) in &replacements {
                    result = result.replace(abbrev, expansion);
                }
                Ok(result)
            }
            PreprocessingRule::HandleGeneNames => {
                // Standardize gene name formatting (simplified)
                Ok(text
                    .replace("BRCA1", "[GENE]BRCA1[/GENE]")
                    .replace("TP53", "[GENE]TP53[/GENE]"))
            }
            PreprocessingRule::PreserveCodeTokens => {
                // Preserve code-like tokens (simplified)
                Ok(text.replace("function", "[CODE]function[/CODE]"))
            }
            _ => {
                // Placeholder for other rules - would implement in production
                Ok(text.to_string())
            }
        }
    }

    /// Generate embedding for text using specialized model
    pub async fn encode_text(&mut self, text: &str) -> Result<Array1<f32>> {
        // Preprocess the text
        let processed_text = self.preprocess_text(text)?;

        // Check cache first
        if let Some(cached_embedding) = self.text_embeddings.get(&processed_text) {
            return Ok(cached_embedding.clone());
        }

        // Generate embedding using domain-specific model
        let embedding = self.generate_specialized_embedding(&processed_text).await?;

        // Cache the result
        self.text_embeddings
            .insert(processed_text, embedding.clone());

        Ok(embedding)
    }

    /// Generate specialized embedding for the specific domain
    async fn generate_specialized_embedding(&self, text: &str) -> Result<Array1<f32>> {
        // In a real implementation, this would use the actual pre-trained model
        // For now, simulate domain-specific embeddings with enhanced features

        let embedding_dim = self.config.model_type.embedding_dim();
        let mut embedding = vec![0.0; embedding_dim];

        // Domain-specific feature extraction
        match self.config.model_type {
            SpecializedTextModel::SciBERT => {
                // Scientific text features: citations, formulas, terminology
                embedding[0] = if text.contains("et al.") { 1.0 } else { 0.0 };
                embedding[1] = if text.contains("figure") || text.contains("table") {
                    1.0
                } else {
                    0.0
                };
                embedding[2] = text.matches(char::is_numeric).count() as f32 / text.len() as f32;
            }
            SpecializedTextModel::CodeBERT => {
                // Code features: keywords, operators, structures
                embedding[0] = if text.contains("function") || text.contains("def") {
                    1.0
                } else {
                    0.0
                };
                embedding[1] = if text.contains("class") || text.contains("struct") {
                    1.0
                } else {
                    0.0
                };
                embedding[2] =
                    text.matches(|c: char| "{}()[]".contains(c)).count() as f32 / text.len() as f32;
            }
            SpecializedTextModel::BioBERT => {
                // Biomedical features: genes, proteins, diseases
                embedding[0] = if text.contains("protein") || text.contains("gene") {
                    1.0
                } else {
                    0.0
                };
                embedding[1] = if text.contains("disease") || text.contains("syndrome") {
                    1.0
                } else {
                    0.0
                };
                embedding[2] = if text.contains("mg") || text.contains("dose") {
                    1.0
                } else {
                    0.0
                };
            }
            _ => {
                // Generic specialized features
                embedding[0] = text.len() as f32 / 1000.0; // Length normalization
                embedding[1] = text.split_whitespace().count() as f32 / text.len() as f32;
                // Word density
            }
        }

        // Fill remaining dimensions with text-based features
        for (i, item) in embedding.iter_mut().enumerate().take(embedding_dim).skip(3) {
            let byte_val = text.as_bytes().get(i % text.len()).copied().unwrap_or(0) as f32;
            *item = (byte_val / 255.0 - 0.5) * 2.0; // Normalize to [-1, 1]
        }

        // Apply domain-specific transformations
        if self.config.domain_pretraining {
            for val in &mut embedding {
                *val *= 1.2; // Amplify features for domain-pretrained models
            }
        }

        Ok(Array1::from_vec(embedding))
    }

    /// Fine-tune the model on domain-specific data
    pub async fn fine_tune(&mut self, training_texts: Vec<String>) -> Result<TrainingStats> {
        let start_time = std::time::Instant::now();
        let epochs = self.config.fine_tune_config.epochs;

        let mut loss_history = Vec::new();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for text in &training_texts {
                // Generate embedding and compute loss
                let embedding = self.encode_text(text).await?;

                // Simplified fine-tuning loss computation
                let target_variance = 0.1; // Target embedding variance
                let actual_variance = embedding.var(0.0);
                let loss = (actual_variance - target_variance).powi(2);
                epoch_loss += loss;
            }

            epoch_loss /= training_texts.len() as f32;
            loss_history.push(epoch_loss as f64);

            if epoch % 10 == 0 {
                println!("Fine-tuning epoch {epoch}: Loss = {epoch_loss:.6}");
            }
        }

        let training_time = start_time.elapsed().as_secs_f64();

        self.training_stats = TrainingStats {
            epochs_completed: epochs,
            final_loss: loss_history.last().copied().unwrap_or(0.0),
            training_time_seconds: training_time,
            convergence_achieved: loss_history.last().is_some_and(|&loss| loss < 0.01),
            loss_history,
        };

        self.is_trained = true;
        self.model_stats.is_trained = true;
        self.model_stats.last_training_time = Some(Utc::now());

        Ok(self.training_stats.clone())
    }

    /// Get model statistics
    pub fn get_stats(&self) -> ModelStats {
        self.model_stats.clone()
    }

    /// Clear cached embeddings
    pub fn clear_cache(&mut self) {
        self.text_embeddings.clear();
    }
}

// Simplified regex-like functionality for preprocessing
#[allow(dead_code)]
mod regex {
    #[allow(dead_code)]
    pub struct Regex(String);

    impl Regex {
        #[allow(dead_code)]
        pub fn new(pattern: &str) -> Result<Self, &'static str> {
            Ok(Regex(pattern.to_string()))
        }

        #[allow(dead_code)]
        pub fn replace_all<'a, F>(&self, text: &'a str, _rep: F) -> std::borrow::Cow<'a, str>
        where
            F: Fn(&str) -> String,
        {
            // Simplified regex replacement for demo - just return original text
            std::borrow::Cow::Borrowed(text)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biomedical_embeddings::types::{
        BiomedicalEmbedding, BiomedicalEmbeddingConfig, BiomedicalEntityType,
    };

    #[test]
    fn test_biomedical_entity_type_from_iri() {
        assert_eq!(
            BiomedicalEntityType::from_iri("http://example.org/gene/BRCA1"),
            Some(BiomedicalEntityType::Gene)
        );
        assert_eq!(
            BiomedicalEntityType::from_iri("http://example.org/disease/cancer"),
            Some(BiomedicalEntityType::Disease)
        );
        assert_eq!(
            BiomedicalEntityType::from_iri("http://example.org/drug/aspirin"),
            Some(BiomedicalEntityType::Drug)
        );
    }

    #[test]
    fn test_biomedical_config_default() {
        let config = BiomedicalEmbeddingConfig::default();
        assert_eq!(config.gene_disease_weight, 2.0);
        assert_eq!(config.drug_target_weight, 1.5);
        assert!(config.use_sequence_similarity);
        assert_eq!(config.species_filter, Some("Homo sapiens".to_string()));
    }

    #[test]
    fn test_biomedical_embedding_creation() {
        let config = BiomedicalEmbeddingConfig::default();
        let model = BiomedicalEmbedding::new(config);

        assert_eq!(model.model_type(), "BiomedicalEmbedding");
        assert!(!model.is_trained());
        assert_eq!(model.gene_embeddings.len(), 0);
    }

    #[test]
    fn test_gene_disease_association() {
        let mut model = BiomedicalEmbedding::new(BiomedicalEmbeddingConfig::default());

        model.add_gene_disease_association("BRCA1", "breast_cancer", 0.8);

        assert_eq!(
            model
                .features
                .gene_disease_associations
                .get(&("BRCA1".to_string(), "breast_cancer".to_string())),
            Some(&0.8)
        );
    }

    #[test]
    fn test_drug_target_interaction() {
        let mut model = BiomedicalEmbedding::new(BiomedicalEmbeddingConfig::default());

        model.add_drug_target_interaction("aspirin", "COX1", 0.9);

        assert_eq!(
            model
                .features
                .drug_target_affinities
                .get(&("aspirin".to_string(), "COX1".to_string())),
            Some(&0.9)
        );
    }

    #[test]
    fn test_specialized_text_model_properties() {
        let scibert = SpecializedTextModel::SciBERT;
        assert_eq!(scibert.model_name(), "allenai/scibert_scivocab_uncased");
        assert_eq!(scibert.vocab_size(), 31090);
        assert_eq!(scibert.embedding_dim(), 768);
        assert_eq!(scibert.max_sequence_length(), 512);

        let codebert = SpecializedTextModel::CodeBERT;
        assert_eq!(codebert.model_name(), "microsoft/codebert-base");
        assert_eq!(codebert.vocab_size(), 50265);

        let biobert = SpecializedTextModel::BioBERT;
        assert_eq!(biobert.model_name(), "dmis-lab/biobert-base-cased-v1.2");
        assert_eq!(biobert.vocab_size(), 28996);
    }

    #[test]
    fn test_specialized_text_preprocessing_rules() {
        let scibert = SpecializedTextModel::SciBERT;
        let rules = scibert.get_preprocessing_rules();
        assert!(rules.contains(&PreprocessingRule::NormalizeScientificNotation));
        assert!(rules.contains(&PreprocessingRule::HandleChemicalFormulas));

        let codebert = SpecializedTextModel::CodeBERT;
        let rules = codebert.get_preprocessing_rules();
        assert!(rules.contains(&PreprocessingRule::PreserveCodeTokens));
        assert!(rules.contains(&PreprocessingRule::HandleCamelCase));

        let biobert = SpecializedTextModel::BioBERT;
        let rules = biobert.get_preprocessing_rules();
        assert!(rules.contains(&PreprocessingRule::NormalizeMedicalTerms));
        assert!(rules.contains(&PreprocessingRule::HandleGeneNames));
    }

    #[test]
    fn test_specialized_text_config_factory_methods() {
        let scibert_config = SpecializedTextEmbedding::scibert_config();
        assert_eq!(scibert_config.model_type, SpecializedTextModel::SciBERT);
        assert_eq!(scibert_config.base_config.dimensions, 768);
        assert!(scibert_config.preprocessing_enabled);
        assert!(scibert_config.vocab_augmentation);
        assert!(scibert_config.domain_pretraining);

        let codebert_config = SpecializedTextEmbedding::codebert_config();
        assert_eq!(codebert_config.model_type, SpecializedTextModel::CodeBERT);
        assert!(!codebert_config.vocab_augmentation);

        let biobert_config = SpecializedTextEmbedding::biobert_config();
        assert_eq!(biobert_config.model_type, SpecializedTextModel::BioBERT);
        assert!(biobert_config.fine_tune_config.freeze_base_layers);
        assert_eq!(biobert_config.fine_tune_config.frozen_layers, 6);
        assert!(biobert_config.fine_tune_config.gradual_unfreezing);
    }

    #[test]
    fn test_specialized_text_embedding_creation() {
        let config = SpecializedTextEmbedding::scibert_config();
        let model = SpecializedTextEmbedding::new(config);

        assert!(model.model_stats.model_type.contains("SciBERT"));
        assert_eq!(model.model_stats.dimensions, 768);
        assert!(!model.is_trained);
        assert_eq!(model.text_embeddings.len(), 0);
        assert_eq!(model.preprocessing_rules.len(), 4); // SciBERT has 4 rules
    }

    #[test]
    fn test_preprocessing_medical_terms() {
        let config = SpecializedTextEmbedding::biobert_config();
        let model = SpecializedTextEmbedding::new(config);

        let text = "Patient takes 100 mg/kg b.i.d. for treatment";
        let processed = model.preprocess_text(text).unwrap();

        // Should expand medical abbreviations
        assert!(processed.contains("milligrams per kilogram"));
        assert!(processed.contains("twice daily"));
    }

    #[test]
    fn test_preprocessing_disabled() {
        let mut config = SpecializedTextEmbedding::biobert_config();
        config.preprocessing_enabled = false;
        let model = SpecializedTextEmbedding::new(config);

        let text = "Patient takes 100 mg/kg b.i.d. for treatment";
        let processed = model.preprocess_text(text).unwrap();

        // Should be unchanged when preprocessing is disabled
        assert_eq!(processed, text);
    }

    #[tokio::test]
    async fn test_specialized_text_encoding() {
        let config = SpecializedTextEmbedding::scibert_config();
        let mut model = SpecializedTextEmbedding::new(config);

        let text = "The protein folding study shows significant results with p < 0.001";
        let embedding = model.encode_text(text).await.unwrap();

        assert_eq!(embedding.len(), 768);

        // Test caching - second call should return cached result
        let embedding2 = model.encode_text(text).await.unwrap();
        assert_eq!(embedding.to_vec(), embedding2.to_vec());
        assert_eq!(model.text_embeddings.len(), 1);
    }

    #[tokio::test]
    async fn test_domain_specific_features() {
        // Test SciBERT features
        let config = SpecializedTextEmbedding::scibert_config();
        let mut model = SpecializedTextEmbedding::new(config);

        let scientific_text = "The study by Smith et al. shows figure 1 demonstrates the results";
        let embedding = model.encode_text(scientific_text).await.unwrap();

        // Should detect scientific features (citations, figures)
        // Values are amplified by 1.2 due to domain pretraining
        assert_eq!(embedding[0], 1.2); // et al. detected, amplified
        assert_eq!(embedding[1], 1.2); // figure detected, amplified

        // Test CodeBERT features
        let config = SpecializedTextEmbedding::codebert_config();
        let mut model = SpecializedTextEmbedding::new(config);

        let code_text = "function calculateSum() { return a + b; }";
        let embedding = model.encode_text(code_text).await.unwrap();

        // Should detect code features (amplified by domain pretraining)
        assert_eq!(embedding[0], 1.2); // function detected, amplified
        assert!(embedding[2] > 0.0); // brackets detected (text-based features)

        // Test BioBERT features
        let config = SpecializedTextEmbedding::biobert_config();
        let mut model = SpecializedTextEmbedding::new(config);

        let biomedical_text =
            "The protein expression correlates with cancer disease progression, dose 100mg";
        let embedding = model.encode_text(biomedical_text).await.unwrap();

        // Should detect biomedical features (amplified by domain pretraining)
        assert_eq!(embedding[0], 1.2); // protein detected, amplified
        assert_eq!(embedding[1], 1.2); // disease detected, amplified
        assert_eq!(embedding[2], 1.2); // mg detected, amplified
    }

    #[tokio::test]
    async fn test_fine_tuning() {
        let config = SpecializedTextEmbedding::biobert_config();
        let mut model = SpecializedTextEmbedding::new(config);

        let training_texts = vec![
            "Gene expression analysis in cancer cells".to_string(),
            "Protein folding mechanisms in disease".to_string(),
            "Drug interaction with target proteins".to_string(),
        ];

        let stats = model.fine_tune(training_texts).await.unwrap();

        assert!(model.is_trained);
        assert_eq!(stats.epochs_completed, 5); // BioBERT config has 5 epochs
        assert!(stats.training_time_seconds > 0.0);
        assert!(!stats.loss_history.is_empty());
        assert!(model.model_stats.is_trained);
        assert!(model.model_stats.last_training_time.is_some());
    }
}
