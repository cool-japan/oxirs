//! Type definitions for multi-modal validation

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use url::Url;

use oxirs_shacl::ValidationViolation;

/// Types of content that can be validated
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ContentType {
    /// Text content (plain text, markup, etc.)
    Text,
    /// Image content (JPEG, PNG, SVG, etc.)
    Image,
    /// Audio content (MP3, WAV, etc.)
    Audio,
    /// Video content (MP4, AVI, etc.)
    Video,
    /// Document content (PDF, Word, etc.)
    Document,
    /// Composite content with multiple types
    Composite,
}

/// Configuration for multi-modal validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalConfig {
    /// Enable text validation
    pub enable_text_validation: bool,
    /// Enable image validation
    pub enable_image_validation: bool,
    /// Enable audio validation
    pub enable_audio_validation: bool,
    /// Enable video validation
    pub enable_video_validation: bool,
    /// Enable document validation
    pub enable_document_validation: bool,
    /// Enable semantic analysis
    pub enable_semantic_analysis: bool,
    /// Enable cross-modal validation
    pub enable_cross_modal_validation: bool,
    /// Maximum content size in bytes
    pub max_content_size: usize,
    /// Cache expiration time
    pub cache_expiration: Duration,
    /// Quality threshold for acceptance
    pub quality_threshold: f64,
    /// Confidence threshold for acceptance
    pub confidence_threshold: f64,
}

impl Default for MultiModalConfig {
    fn default() -> Self {
        Self {
            enable_text_validation: true,
            enable_image_validation: true,
            enable_audio_validation: true,
            enable_video_validation: true,
            enable_document_validation: true,
            enable_semantic_analysis: true,
            enable_cross_modal_validation: true,
            max_content_size: 100 * 1024 * 1024, // 100MB
            cache_expiration: Duration::from_secs(3600),
            quality_threshold: 0.7,
            confidence_threshold: 0.8,
        }
    }
}

/// Reference to multi-modal content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalContentRef {
    /// Unique identifier for the content
    pub id: String,
    /// Type of content
    pub content_type: ContentType,
    /// Source URL or path
    pub source_url: Option<Url>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Multi-modal content data
#[derive(Debug, Clone)]
pub struct MultiModalContent {
    /// Unique identifier
    pub id: String,
    /// Type of content
    pub content_type: ContentType,
    /// Raw content data
    pub data: Vec<u8>,
    /// Content metadata
    pub metadata: ContentMetadata,
    /// Source URL if applicable
    pub source_url: Option<Url>,
    /// Timestamp when content was loaded
    pub timestamp: SystemTime,
}

/// Metadata for content
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct ContentMetadata {
    /// MIME type
    pub mime_type: Option<String>,
    /// File size in bytes
    pub size: Option<usize>,
    /// Creation timestamp
    pub created_at: Option<SystemTime>,
    /// Last modified timestamp
    pub modified_at: Option<SystemTime>,
    /// Author information
    pub author: Option<String>,
    /// Content title
    pub title: Option<String>,
    /// Content description
    pub description: Option<String>,
    /// Content language
    pub language: Option<String>,
    /// Content encoding
    pub encoding: Option<String>,
    /// Additional properties
    pub properties: HashMap<String, String>,
}


/// Result of content analysis
#[derive(Debug, Clone)]
pub struct ContentAnalysis {
    /// Content identifier
    pub content_id: String,
    /// Content type
    pub content_type: ContentType,
    /// Content metadata
    pub metadata: ContentMetadata,
    /// Text analysis results
    pub text_analysis: Option<TextAnalysis>,
    /// Image analysis results
    pub image_analysis: Option<ImageAnalysis>,
    /// Audio analysis results
    pub audio_analysis: Option<AudioAnalysis>,
    /// Video analysis results
    pub video_analysis: Option<VideoAnalysis>,
    /// Document analysis results
    pub document_analysis: Option<DocumentAnalysis>,
    /// Semantic analysis results
    pub semantic_analysis: Option<SemanticAnalysis>,
    /// Overall quality score
    pub quality_score: f64,
    /// Analysis confidence
    pub confidence: f64,
    /// Processing time
    pub processing_time: Duration,
    /// Analysis timestamp
    pub timestamp: SystemTime,
}

/// Text analysis results
#[derive(Debug, Clone)]
pub struct TextAnalysis {
    /// Validation results from text validators
    pub validation_results: HashMap<String, ValidationResult>,
    /// Detected language
    pub language: Option<String>,
    /// Sentiment analysis result
    pub sentiment: Option<SentimentResult>,
    /// Extracted entities
    pub entities: Vec<EntityResult>,
    /// Detected topics
    pub topics: Vec<TopicResult>,
    /// Readability score
    pub readability_score: f64,
    /// Complexity score
    pub complexity_score: f64,
}

/// Image analysis results
#[derive(Debug, Clone)]
pub struct ImageAnalysis {
    /// Validation results from image validators
    pub validation_results: HashMap<String, ValidationResult>,
    /// Image format
    pub format: Option<String>,
    /// Image dimensions
    pub dimensions: Option<(u32, u32)>,
    /// Color profile
    pub color_profile: Option<String>,
    /// Detected objects
    pub detected_objects: Vec<ObjectDetectionResult>,
    /// Detected faces
    pub faces: Vec<FaceDetectionResult>,
    /// Scene description
    pub scene_description: Option<String>,
}

/// Audio analysis results
#[derive(Debug, Clone)]
pub struct AudioAnalysis {
    /// Validation results from audio validators
    pub validation_results: HashMap<String, ValidationResult>,
    /// Audio format
    pub format: Option<String>,
    /// Duration
    pub duration: Option<Duration>,
    /// Sample rate
    pub sample_rate: Option<u32>,
    /// Number of channels
    pub channels: Option<u32>,
    /// Speech transcription
    pub transcription: Option<String>,
    /// Music features
    pub music_features: Option<MusicFeatures>,
}

/// Video analysis results
#[derive(Debug, Clone)]
pub struct VideoAnalysis {
    /// Validation results from video validators
    pub validation_results: HashMap<String, ValidationResult>,
    /// Video format
    pub format: Option<String>,
    /// Duration
    pub duration: Option<Duration>,
    /// Frame rate
    pub frame_rate: Option<f64>,
    /// Resolution
    pub resolution: Option<(u32, u32)>,
    /// Detected scenes
    pub scenes: Vec<SceneResult>,
    /// Motion vectors
    pub motion_vectors: Vec<MotionVector>,
}

/// Document analysis results
#[derive(Debug, Clone)]
pub struct DocumentAnalysis {
    /// Validation results from document validators
    pub validation_results: HashMap<String, ValidationResult>,
    /// Document format
    pub format: Option<String>,
    /// Number of pages
    pub page_count: Option<u32>,
    /// Extracted text content
    pub text_content: Option<String>,
    /// Document structure
    pub structure: Option<DocumentStructure>,
    /// Document metadata
    pub metadata: HashMap<String, String>,
}

/// Semantic analysis results
#[derive(Debug, Clone)]
pub struct SemanticAnalysis {
    /// Analysis results from semantic analyzers
    pub analysis_results: HashMap<String, AnalysisResult>,
    /// Extracted concepts
    pub concepts: Vec<ConceptResult>,
    /// Extracted relations
    pub relations: Vec<RelationResult>,
    /// Knowledge graph representation
    pub knowledge_graph: Option<KnowledgeGraph>,
    /// Semantic similarity score
    pub semantic_similarity: f64,
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Confidence score
    pub confidence: f64,
    /// Error message if validation failed
    pub error_message: Option<String>,
    /// Additional details
    pub details: HashMap<String, String>,
}

/// Analysis result
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Analysis score
    pub score: f64,
    /// Confidence in analysis
    pub confidence: f64,
    /// Analysis details
    pub details: HashMap<String, String>,
}

/// Sentiment analysis result
#[derive(Debug, Clone)]
pub struct SentimentResult {
    /// Sentiment polarity (-1.0 to 1.0)
    pub polarity: f64,
    /// Sentiment confidence
    pub confidence: f64,
    /// Sentiment label
    pub label: String,
}

/// Entity extraction result
#[derive(Debug, Clone)]
pub struct EntityResult {
    /// Entity text
    pub text: String,
    /// Entity type
    pub entity_type: String,
    /// Start position
    pub start: usize,
    /// End position
    pub end: usize,
    /// Confidence score
    pub confidence: f64,
}

/// Topic detection result
#[derive(Debug, Clone)]
pub struct TopicResult {
    /// Topic label
    pub label: String,
    /// Topic probability
    pub probability: f64,
    /// Topic keywords
    pub keywords: Vec<String>,
}

/// Object detection result
#[derive(Debug, Clone)]
pub struct ObjectDetectionResult {
    /// Object class
    pub class: String,
    /// Confidence score
    pub confidence: f64,
    /// Bounding box
    pub bbox: (f64, f64, f64, f64),
}

/// Face detection result
#[derive(Debug, Clone)]
pub struct FaceDetectionResult {
    /// Confidence score
    pub confidence: f64,
    /// Bounding box
    pub bbox: (f64, f64, f64, f64),
    /// Facial landmarks
    pub landmarks: Vec<(f64, f64)>,
}

/// Music features
#[derive(Debug, Clone)]
pub struct MusicFeatures {
    /// Tempo (BPM)
    pub tempo: Option<f64>,
    /// Key signature
    pub key: Option<String>,
    /// Time signature
    pub time_signature: Option<String>,
    /// Mood
    pub mood: Option<String>,
    /// Genre
    pub genre: Option<String>,
}

/// Scene analysis result
#[derive(Debug, Clone)]
pub struct SceneResult {
    /// Scene timestamp
    pub timestamp: Duration,
    /// Scene description
    pub description: String,
    /// Scene confidence
    pub confidence: f64,
}

/// Motion vector
#[derive(Debug, Clone)]
pub struct MotionVector {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
    /// Direction
    pub direction: f64,
    /// Magnitude
    pub magnitude: f64,
}

/// Document structure
#[derive(Debug, Clone)]
pub struct DocumentStructure {
    /// Document title
    pub title: Option<String>,
    /// Document sections
    pub sections: Vec<DocumentSection>,
    /// Table of contents
    pub toc: Vec<TocEntry>,
}

/// Document section
#[derive(Debug, Clone)]
pub struct DocumentSection {
    /// Section title
    pub title: String,
    /// Section level
    pub level: u32,
    /// Section content
    pub content: String,
}

/// Table of contents entry
#[derive(Debug, Clone)]
pub struct TocEntry {
    /// Entry title
    pub title: String,
    /// Entry level
    pub level: u32,
    /// Page number
    pub page: Option<u32>,
}

/// Concept result
#[derive(Debug, Clone)]
pub struct ConceptResult {
    /// Concept name
    pub name: String,
    /// Concept URI
    pub uri: Option<String>,
    /// Concept type
    pub concept_type: String,
    /// Confidence score
    pub confidence: f64,
}

/// Relation result
#[derive(Debug, Clone)]
pub struct RelationResult {
    /// Subject concept
    pub subject: ConceptResult,
    /// Predicate
    pub predicate: String,
    /// Object concept
    pub object: ConceptResult,
    /// Confidence score
    pub confidence: f64,
}

/// Knowledge graph representation
#[derive(Debug, Clone)]
pub struct KnowledgeGraph {
    /// Graph nodes
    pub nodes: Vec<ConceptResult>,
    /// Graph edges
    pub edges: Vec<RelationResult>,
}

/// Cached content
#[derive(Debug, Clone)]
pub struct CachedContent {
    /// Content data
    pub content: MultiModalContent,
    /// Cache timestamp
    pub timestamp: SystemTime,
    /// Access count
    pub access_count: u32,
}

/// Multi-modal validation report
#[derive(Debug, Clone)]
pub struct MultiModalValidationReport {
    /// Validation violations
    pub violations: Vec<ValidationViolation>,
    /// Content analyses
    pub content_analyses: Vec<ContentAnalysis>,
    /// Semantic insights
    pub semantic_insights: Vec<SemanticInsight>,
    /// Cross-modal insights
    pub cross_modal_insights: Vec<CrossModalInsight>,
    /// Validation statistics
    pub validation_statistics: ValidationStatistics,
    /// Recommendations
    pub recommendations: Vec<ValidationRecommendation>,
}

/// Semantic insight
#[derive(Debug, Clone)]
pub struct SemanticInsight {
    /// Insight type
    pub insight_type: String,
    /// Insight description
    pub description: String,
    /// Confidence score
    pub confidence: f64,
    /// Related concepts
    pub related_concepts: Vec<String>,
}

/// Cross-modal insight
#[derive(Debug, Clone)]
pub struct CrossModalInsight {
    /// Insight description
    pub description: String,
    /// Involved content types
    pub content_types: Vec<ContentType>,
    /// Confidence score
    pub confidence: f64,
    /// Correlation strength
    pub correlation_strength: f64,
}

/// Validation statistics
#[derive(Debug, Clone)]
pub struct ValidationStatistics {
    /// Total number of validations
    pub total_validations: u64,
    /// Number of successful validations
    pub successful_validations: u64,
    /// Number of failed validations
    pub failed_validations: u64,
    /// Average processing time
    pub average_processing_time: Duration,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// Validation recommendation
#[derive(Debug, Clone)]
pub struct ValidationRecommendation {
    /// Recommendation type
    pub recommendation_type: String,
    /// Recommendation description
    pub description: String,
    /// Priority level
    pub priority: u32,
    /// Confidence score
    pub confidence: f64,
}

impl fmt::Display for ContentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContentType::Text => write!(f, "text"),
            ContentType::Image => write!(f, "image"),
            ContentType::Audio => write!(f, "audio"),
            ContentType::Video => write!(f, "video"),
            ContentType::Document => write!(f, "document"),
            ContentType::Composite => write!(f, "composite"),
        }
    }
}

impl fmt::Display for MultiModalConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MultiModalConfig {{ text: {}, image: {}, audio: {}, video: {}, document: {}, semantic: {}, cross_modal: {} }}",
               self.enable_text_validation,
               self.enable_image_validation,
               self.enable_audio_validation,
               self.enable_video_validation,
               self.enable_document_validation,
               self.enable_semantic_analysis,
               self.enable_cross_modal_validation)
    }
}
