//! Type definitions and configurations for transformer models

use crate::ModelConfig;
use scirs2_core::ndarray_ext::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of transformer model to use
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TransformerType {
    BERT,
    RoBERTa,
    SentenceBERT,
    SciBERT,
    BioBERT,
    CodeBERT,
    LegalBERT,
    NewsBERT,
    SocialMediaBERT,
    MBert,
    XLMR,
}

impl TransformerType {
    pub fn model_name(&self) -> &'static str {
        match self {
            TransformerType::BERT => "bert-base-uncased",
            TransformerType::RoBERTa => "roberta-base",
            TransformerType::SentenceBERT => "sentence-transformers/all-MiniLM-L6-v2",
            TransformerType::SciBERT => "allenai/scibert_scivocab_uncased",
            TransformerType::BioBERT => "dmis-lab/biobert-v1.1",
            TransformerType::CodeBERT => "microsoft/codebert-base",
            TransformerType::LegalBERT => "nlpaueb/legal-bert-base-uncased",
            TransformerType::NewsBERT => "dkleczek/bert-base-polish-uncased-v1",
            TransformerType::SocialMediaBERT => "vinai/bertweet-base",
            TransformerType::MBert => "bert-base-multilingual-cased",
            TransformerType::XLMR => "xlm-roberta-base",
        }
    }

    pub fn default_dimensions(&self) -> usize {
        match self {
            TransformerType::SentenceBERT => 384,
            _ => 768,
        }
    }
}

/// Pooling strategy for sentence embeddings
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum PoolingStrategy {
    /// Simple mean pooling
    Mean,
    /// Max pooling
    Max,
    /// CLS token pooling (for BERT-like models)
    CLS,
    /// Mean of first and last tokens
    MeanFirstLast,
    /// Weighted mean based on attention
    AttentionWeighted,
}

/// Configuration for transformer-based models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub base_config: ModelConfig,
    pub transformer_type: TransformerType,
    pub max_sequence_length: usize,
    pub use_pooling: bool,
    pub pooling_strategy: PoolingStrategy,
    pub fine_tune: bool,
    pub learning_rate_schedule: String,
    pub warmup_steps: usize,
    pub gradient_accumulation_steps: usize,
    pub normalize_embeddings: bool,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            base_config: ModelConfig::default(),
            transformer_type: TransformerType::SentenceBERT,
            max_sequence_length: 512,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Mean,
            fine_tune: false,
            learning_rate_schedule: "linear".to_string(),
            warmup_steps: 1000,
            gradient_accumulation_steps: 1,
            normalize_embeddings: true,
        }
    }
}

impl TransformerConfig {
    pub fn bert_config(dimensions: usize) -> Self {
        Self {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::BERT,
            pooling_strategy: PoolingStrategy::CLS,
            ..Default::default()
        }
    }

    pub fn sentence_bert_config(dimensions: usize) -> Self {
        Self {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::SentenceBERT,
            pooling_strategy: PoolingStrategy::Mean,
            normalize_embeddings: true,
            ..Default::default()
        }
    }

    pub fn domain_specific_config(
        transformer_type: TransformerType,
        dimensions: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type,
            max_sequence_length: max_seq_len,
            fine_tune: true,
            ..Default::default()
        }
    }
}

/// Model weights for transformer embeddings
#[derive(Debug, Clone)]
pub struct ModelWeights {
    pub embeddings: Array2<f32>,
    pub attention_weights: Option<Array2<f32>>,
    pub layer_norms: Option<Vec<Array2<f32>>>,
}

impl ModelWeights {
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            embeddings: Array2::zeros((vocab_size, hidden_size)),
            attention_weights: None,
            layer_norms: None,
        }
    }

    pub fn with_attention(vocab_size: usize, hidden_size: usize, num_heads: usize) -> Self {
        Self {
            embeddings: Array2::zeros((vocab_size, hidden_size)),
            attention_weights: Some(Array2::zeros((num_heads, hidden_size))),
            layer_norms: None,
        }
    }
}

/// Evaluation metrics for transformer embeddings
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingEvaluationMetrics {
    pub coherence_score: f32,
    pub diversity_score: f32,
    pub cluster_quality: f32,
    pub semantic_consistency: f32,
    pub triple_accuracy: f32,
    pub mean_cosine_similarity: f32,
    pub embedding_variance: f32,
    pub domain_adaptation_score: f32,
}

/// Training statistics for monitoring
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransformerTrainingStats {
    pub contrastive_loss: f32,
    pub reconstruction_loss: f32,
    pub regularization_loss: f32,
    pub gradient_norm: f32,
    pub learning_rate: f32,
    pub epoch: usize,
    pub batch_processed: usize,
}

/// Attention visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionVisualization {
    pub token_attention: Vec<Vec<f32>>,
    pub head_attention: Vec<Vec<f32>>,
    pub layer_attention: Vec<Vec<f32>>,
    pub input_tokens: Vec<String>,
}

/// Transformer-based embedding model
#[derive(Debug)]
pub struct TransformerEmbedding {
    /// Transformer configuration
    pub config: TransformerConfig,
    /// Model weights
    pub weights: ModelWeights,
    /// Whether the model has been trained
    pub is_trained: bool,
    /// Training statistics
    pub training_stats: TransformerTrainingStats,
}

impl TransformerEmbedding {
    /// Create a new transformer embedding model
    pub fn new(config: TransformerConfig) -> Self {
        let vocab_size = 10000; // Default vocabulary size
        let weights = ModelWeights::new(vocab_size, config.base_config.dimensions);

        Self {
            config,
            weights,
            is_trained: false,
            training_stats: TransformerTrainingStats::default(),
        }
    }

    /// Create a transformer embedding with attention
    pub fn with_attention(config: TransformerConfig, num_heads: usize) -> Self {
        let vocab_size = 10000; // Default vocabulary size
        let weights =
            ModelWeights::with_attention(vocab_size, config.base_config.dimensions, num_heads);

        Self {
            config,
            weights,
            is_trained: false,
            training_stats: TransformerTrainingStats::default(),
        }
    }

    /// Create a transformer embedding with sentence BERT configuration
    pub fn sentence_bert_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig::sentence_bert_config(dimensions)
    }

    /// Generate embeddings for input text
    pub fn generate_embedding(&self, text: &str) -> Result<Array2<f32>, anyhow::Error> {
        // Simple tokenization and embedding lookup
        let tokens = text.split_whitespace().collect::<Vec<_>>();
        let mut embeddings = Vec::new();

        for (i, _token) in tokens.iter().enumerate() {
            if i < self.weights.embeddings.nrows() {
                let embedding = self.weights.embeddings.row(i).to_owned();
                embeddings.push(embedding);
            }
        }

        if embeddings.is_empty() {
            return Err(anyhow::anyhow!("No embeddings generated"));
        }

        // Stack embeddings into matrix
        let num_tokens = embeddings.len();
        let embedding_dim = embeddings[0].len();
        let mut result = Array2::zeros((num_tokens, embedding_dim));

        for (i, embedding) in embeddings.iter().enumerate() {
            for (j, &value) in embedding.iter().enumerate() {
                result[[i, j]] = value;
            }
        }

        Ok(result)
    }
}

/// Domain-specific preprocessing rules
#[derive(Debug, Clone)]
pub struct DomainPreprocessingRules {
    pub abbreviation_expansions: HashMap<String, String>,
    pub domain_specific_patterns: Vec<(String, String)>,
    pub tokenization_rules: Vec<String>,
}

impl DomainPreprocessingRules {
    pub fn scientific() -> Self {
        let mut abbreviations = HashMap::new();
        abbreviations.insert("DNA".to_string(), "deoxyribonucleic acid".to_string());
        abbreviations.insert("RNA".to_string(), "ribonucleic acid".to_string());
        abbreviations.insert("ATP".to_string(), "adenosine triphosphate".to_string());
        abbreviations.insert("GDP".to_string(), "guanosine diphosphate".to_string());
        abbreviations.insert("GTP".to_string(), "guanosine triphosphate".to_string());
        abbreviations.insert("Co2".to_string(), "carbon dioxide".to_string());
        abbreviations.insert("H2O".to_string(), "water".to_string());
        abbreviations.insert("NaCl".to_string(), "sodium chloride".to_string());

        Self {
            abbreviation_expansions: abbreviations,
            domain_specific_patterns: vec![
                (r"(\d+)°C".to_string(), "$1 degrees celsius".to_string()),
                (
                    r"(\d+)mg/ml".to_string(),
                    "$1 milligrams per milliliter".to_string(),
                ),
                (r"pH(\d+)".to_string(), "pH level $1".to_string()),
            ],
            tokenization_rules: vec!["preserve_chemical_formulas".to_string()],
        }
    }

    pub fn biomedical() -> Self {
        let mut abbreviations = HashMap::new();
        abbreviations.insert("p53".to_string(), "tumor protein p53".to_string());
        abbreviations.insert("BRCA1".to_string(), "breast cancer gene 1".to_string());
        abbreviations.insert("BRCA2".to_string(), "breast cancer gene 2".to_string());
        abbreviations.insert(
            "TNF-α".to_string(),
            "tumor necrosis factor alpha".to_string(),
        );
        abbreviations.insert("mRNA".to_string(), "messenger ribonucleic acid".to_string());
        abbreviations.insert("tRNA".to_string(), "transfer ribonucleic acid".to_string());
        abbreviations.insert("CNS".to_string(), "central nervous system".to_string());
        abbreviations.insert("PNS".to_string(), "peripheral nervous system".to_string());

        Self {
            abbreviation_expansions: abbreviations,
            domain_specific_patterns: vec![
                (r"([A-Z]+)\d+".to_string(), "$1 protein".to_string()),
                (
                    r"(\w+)-mutation".to_string(),
                    "$1 genetic mutation".to_string(),
                ),
            ],
            tokenization_rules: vec!["preserve_gene_names".to_string()],
        }
    }

    pub fn legal() -> Self {
        let mut abbreviations = HashMap::new();
        abbreviations.insert("USC".to_string(), "United States Code".to_string());
        abbreviations.insert("CFR".to_string(), "Code of Federal Regulations".to_string());
        abbreviations.insert(
            "plaintiff".to_string(),
            "party bringing lawsuit".to_string(),
        );
        abbreviations.insert("defendant".to_string(), "party being sued".to_string());
        abbreviations.insert("tort".to_string(), "civil wrong".to_string());
        abbreviations.insert("v.".to_string(), "versus".to_string());

        Self {
            abbreviation_expansions: abbreviations,
            domain_specific_patterns: vec![
                (r"§(\d+)".to_string(), "section $1".to_string()),
                (
                    r"(\w+)\s+v\.\s+(\w+)".to_string(),
                    "$1 versus $2".to_string(),
                ),
            ],
            tokenization_rules: vec!["preserve_case_citations".to_string()],
        }
    }

    pub fn news() -> Self {
        let mut abbreviations = HashMap::new();
        abbreviations.insert("CEO".to_string(), "chief executive officer".to_string());
        abbreviations.insert("CFO".to_string(), "chief financial officer".to_string());
        abbreviations.insert("IPO".to_string(), "initial public offering".to_string());
        abbreviations.insert(
            "SEC".to_string(),
            "Securities and Exchange Commission".to_string(),
        );
        abbreviations.insert("GDP".to_string(), "gross domestic product".to_string());
        abbreviations.insert("CPI".to_string(), "consumer price index".to_string());

        Self {
            abbreviation_expansions: abbreviations,
            domain_specific_patterns: vec![
                (r"Q(\d)".to_string(), "quarter $1".to_string()),
                (r"(\d+)%".to_string(), "$1 percent".to_string()),
            ],
            tokenization_rules: vec!["preserve_financial_terms".to_string()],
        }
    }

    pub fn social_media() -> Self {
        let mut abbreviations = HashMap::new();
        abbreviations.insert("lol".to_string(), "laugh out loud".to_string());
        abbreviations.insert("omg".to_string(), "oh my god".to_string());
        abbreviations.insert("btw".to_string(), "by the way".to_string());
        abbreviations.insert("fyi".to_string(), "for your information".to_string());
        abbreviations.insert("imo".to_string(), "in my opinion".to_string());
        abbreviations.insert("tbh".to_string(), "to be honest".to_string());
        abbreviations.insert("smh".to_string(), "shaking my head".to_string());

        Self {
            abbreviation_expansions: abbreviations,
            domain_specific_patterns: vec![
                (r"#(\w+)".to_string(), "hashtag $1".to_string()),
                (r"@(\w+)".to_string(), "mention $1".to_string()),
                (r":\)".to_string(), "happy".to_string()),
                (r":\(".to_string(), "sad".to_string()),
                (r":D".to_string(), "very happy".to_string()),
                (r";\)".to_string(), "winking".to_string()),
            ],
            tokenization_rules: vec![
                "preserve_hashtags".to_string(),
                "preserve_mentions".to_string(),
            ],
        }
    }
}
