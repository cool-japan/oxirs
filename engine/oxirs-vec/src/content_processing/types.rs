//! Data types and structures for content processing
//!
//! This module contains all the core data types, enums, and structures
//! used throughout the content processing system.

use crate::{embeddings::EmbeddableContent, Vector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Document format types supported by the content processor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DocumentFormat {
    Pdf,
    Html,
    Xml,
    Markdown,
    PlainText,
    Docx,
    Pptx,
    Xlsx,
    Rtf,
    Epub,
    Json,
    Csv,
    // Multimedia formats
    Image,
    Audio,
    Video,
    Unknown,
}

/// Content extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentExtractionConfig {
    /// Extract text content
    pub extract_text: bool,
    /// Extract metadata
    pub extract_metadata: bool,
    /// Extract images
    pub extract_images: bool,
    /// Extract tables
    pub extract_tables: bool,
    /// Extract links
    pub extract_links: bool,
    /// Maximum content length to extract
    pub max_content_length: usize,
    /// Preserve document structure
    pub preserve_structure: bool,
    /// Extract page/section information
    pub extract_page_info: bool,
    /// Language detection
    pub detect_language: bool,
    /// Content chunking strategy
    pub chunking_strategy: ChunkingStrategy,
    /// Extract multimedia features (image analysis, audio analysis, etc.)
    pub extract_multimedia_features: bool,
    /// Generate image embeddings using computer vision models
    pub generate_image_embeddings: bool,
    /// Extract audio features and generate embeddings
    pub extract_audio_features: bool,
    /// Extract video keyframes and generate embeddings
    pub extract_video_features: bool,
    /// Maximum image processing resolution
    pub max_image_resolution: Option<(u32, u32)>,
}

impl Default for ContentExtractionConfig {
    fn default() -> Self {
        Self {
            extract_text: true,
            extract_metadata: true,
            extract_images: false,
            extract_tables: true,
            extract_links: true,
            max_content_length: 1_000_000, // 1MB
            preserve_structure: true,
            extract_page_info: true,
            detect_language: true,
            chunking_strategy: ChunkingStrategy::Paragraph,
            extract_multimedia_features: false,
            generate_image_embeddings: false,
            extract_audio_features: false,
            extract_video_features: false,
            max_image_resolution: Some((1920, 1080)), // Full HD max
        }
    }
}

/// Content chunking strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// Split by paragraphs
    Paragraph,
    /// Split by sentences
    Sentence,
    /// Split by fixed token count
    FixedTokens(usize),
    /// Split by semantic sections
    Semantic,
    /// Split by pages/slides
    Page,
    /// Custom regex pattern
    Custom(String),
}

/// Extracted document content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedContent {
    /// Document format
    pub format: DocumentFormat,
    /// Raw text content
    pub text: String,
    /// Document metadata
    pub metadata: HashMap<String, String>,
    /// Extracted images (base64 encoded)
    pub images: Vec<ExtractedImage>,
    /// Extracted tables
    pub tables: Vec<ExtractedTable>,
    /// Extracted links
    pub links: Vec<ExtractedLink>,
    /// Document structure information
    pub structure: DocumentStructure,
    /// Content chunks for embedding
    pub chunks: Vec<ContentChunk>,
    /// Detected language
    pub language: Option<String>,
    /// Processing statistics
    pub processing_stats: ProcessingStats,
    /// Extracted audio content
    pub audio_content: Vec<ExtractedAudio>,
    /// Extracted video content
    pub video_content: Vec<ExtractedVideo>,
    /// Cross-modal embeddings (combining text, image, audio, video)
    pub cross_modal_embeddings: Vec<CrossModalEmbedding>,
}

/// Extracted image information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedImage {
    /// Image data (base64 encoded)
    pub data: String,
    /// Image format (JPEG, PNG, etc.)
    pub format: String,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Alternative text
    pub alt_text: Option<String>,
    /// Caption
    pub caption: Option<String>,
    /// Page/location information
    pub location: ContentLocation,
    /// Extracted visual features (SIFT, HOG, color histograms, etc.)
    pub visual_features: Option<ImageFeatures>,
    /// Generated embedding vector
    pub embedding: Option<Vector>,
    /// Object detection results
    pub detected_objects: Vec<DetectedObject>,
    /// Image classification labels with confidence scores
    pub classification_labels: Vec<ClassificationLabel>,
}

/// Image feature extraction results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageFeatures {
    /// Color histogram features
    pub color_histogram: Option<Vec<f32>>,
    /// Texture features (LBP, GLCM, etc.)
    pub texture_features: Option<Vec<f32>>,
    /// Edge features
    pub edge_features: Option<Vec<f32>>,
    /// SIFT keypoints and descriptors
    pub sift_features: Option<Vec<f32>>,
    /// CNN features from pre-trained models
    pub cnn_features: Option<Vec<f32>>,
    /// Dominant colors
    pub dominant_colors: Vec<(u8, u8, u8)>, // RGB tuples
    /// Image complexity metrics
    pub complexity_metrics: ImageComplexityMetrics,
}

/// Object detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedObject {
    /// Object class label
    pub label: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Bounding box coordinates (x, y, width, height)
    pub bbox: (u32, u32, u32, u32),
}

/// Classification label with confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationLabel {
    /// Class label
    pub label: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
}

/// Image complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageComplexityMetrics {
    /// Edge density (0.0 to 1.0)
    pub edge_density: f32,
    /// Color diversity (0.0 to 1.0)
    pub color_diversity: f32,
    /// Texture complexity (0.0 to 1.0)
    pub texture_complexity: f32,
    /// Information entropy
    pub entropy: f32,
}

/// Extracted audio information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedAudio {
    /// Audio data (base64 encoded)
    pub data: String,
    /// Audio format (MP3, WAV, etc.)
    pub format: String,
    /// Duration in seconds
    pub duration: f32,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Extracted audio features
    pub audio_features: Option<AudioFeatures>,
    /// Generated embedding vector
    pub embedding: Option<Vector>,
    /// Transcribed text (if available)
    pub transcription: Option<String>,
    /// Music analysis (if music content)
    pub music_analysis: Option<MusicAnalysis>,
    /// Speech analysis (if speech content)
    pub speech_analysis: Option<SpeechAnalysis>,
}

/// Audio feature extraction results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFeatures {
    /// Mel-frequency cepstral coefficients
    pub mfcc: Option<Vec<f32>>,
    /// Spectral features (centroid, rolloff, etc.)
    pub spectral_features: Option<Vec<f32>>,
    /// Rhythm and tempo features
    pub rhythm_features: Option<Vec<f32>>,
    /// Harmonic features
    pub harmonic_features: Option<Vec<f32>>,
    /// Zero-crossing rate
    pub zero_crossing_rate: f32,
    /// Energy and loudness metrics
    pub energy_metrics: AudioEnergyMetrics,
}

/// Audio energy and loudness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEnergyMetrics {
    /// RMS energy
    pub rms_energy: f32,
    /// Peak amplitude
    pub peak_amplitude: f32,
    /// Average loudness (LUFS)
    pub average_loudness: f32,
    /// Dynamic range
    pub dynamic_range: f32,
}

/// Music analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicAnalysis {
    /// Detected tempo (BPM)
    pub tempo: Option<f32>,
    /// Key signature
    pub key: Option<String>,
    /// Time signature
    pub time_signature: Option<String>,
    /// Genre classification
    pub genre: Option<String>,
    /// Mood/valence (-1.0 to 1.0)
    pub valence: Option<f32>,
    /// Energy level (0.0 to 1.0)
    pub energy: Option<f32>,
}

/// Speech analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechAnalysis {
    /// Detected language
    pub language: Option<String>,
    /// Speaker gender (if detectable)
    pub speaker_gender: Option<String>,
    /// Speaker emotion
    pub emotion: Option<String>,
    /// Speech rate (words per minute)
    pub speech_rate: Option<f32>,
    /// Pitch statistics
    pub pitch_stats: Option<PitchStatistics>,
}

/// Pitch statistics for speech analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchStatistics {
    /// Mean pitch (Hz)
    pub mean_pitch: f32,
    /// Pitch standard deviation
    pub pitch_std: f32,
    /// Pitch range (max - min)
    pub pitch_range: f32,
}

/// Extracted video information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedVideo {
    /// Video data (base64 encoded or file path)
    pub data: String,
    /// Video format (MP4, AVI, etc.)
    pub format: String,
    /// Duration in seconds
    pub duration: f32,
    /// Frame rate (fps)
    pub frame_rate: f32,
    /// Video resolution (width, height)
    pub resolution: (u32, u32),
    /// Extracted keyframes
    pub keyframes: Vec<VideoKeyframe>,
    /// Generated embedding vector
    pub embedding: Option<Vector>,
    /// Audio track analysis
    pub audio_analysis: Option<ExtractedAudio>,
    /// Video analysis results
    pub video_analysis: Option<VideoAnalysis>,
}

/// Video keyframe information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoKeyframe {
    /// Timestamp in seconds
    pub timestamp: f32,
    /// Frame image data
    pub image: ExtractedImage,
    /// Scene change score (0.0 to 1.0)
    pub scene_change_score: f32,
}

/// Video analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoAnalysis {
    /// Detected scenes with timestamps
    pub scenes: Vec<VideoScene>,
    /// Motion analysis
    pub motion_analysis: Option<MotionAnalysis>,
    /// Visual activity level
    pub activity_level: f32,
    /// Color characteristics over time
    pub color_timeline: Vec<(f32, Vec<(u8, u8, u8)>)>, // (timestamp, dominant_colors)
}

/// Video scene detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoScene {
    /// Scene start time in seconds
    pub start_time: f32,
    /// Scene end time in seconds
    pub end_time: f32,
    /// Scene description/label
    pub description: Option<String>,
    /// Representative keyframe
    pub representative_frame: Option<ExtractedImage>,
}

/// Motion analysis for video
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionAnalysis {
    /// Average motion magnitude
    pub average_motion: f32,
    /// Motion variance
    pub motion_variance: f32,
    /// Camera motion type (pan, tilt, zoom, etc.)
    pub camera_motion: Option<String>,
    /// Object motion tracking
    pub object_motion: Vec<ObjectMotion>,
}

/// Object motion tracking result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectMotion {
    /// Object identifier
    pub object_id: String,
    /// Motion trajectory (time, x, y)
    pub trajectory: Vec<(f32, f32, f32)>,
    /// Motion speed (pixels per second)
    pub speed: f32,
}

/// Extracted table information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedTable {
    /// Table headers
    pub headers: Vec<String>,
    /// Table rows
    pub rows: Vec<Vec<String>>,
    /// Table caption
    pub caption: Option<String>,
    /// Location in document
    pub location: ContentLocation,
}

/// Extracted link information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedLink {
    /// Link URL
    pub url: String,
    /// Link text
    pub text: String,
    /// Link title
    pub title: Option<String>,
    /// Location in document
    pub location: ContentLocation,
}

/// Document structure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentStructure {
    /// Document title
    pub title: Option<String>,
    /// Headings hierarchy
    pub headings: Vec<Heading>,
    /// Page count
    pub page_count: usize,
    /// Section count
    pub section_count: usize,
    /// Table of contents
    pub table_of_contents: Vec<TocEntry>,
}

/// Heading information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heading {
    /// Heading level (1-6)
    pub level: usize,
    /// Heading text
    pub text: String,
    /// Location in document
    pub location: ContentLocation,
}

/// Table of contents entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TocEntry {
    /// Section title
    pub title: String,
    /// Section level
    pub level: usize,
    /// Page number
    pub page: Option<usize>,
    /// Location reference
    pub location: ContentLocation,
}

/// Content chunk for embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentChunk {
    /// Chunk text content
    pub text: String,
    /// Chunk type
    pub chunk_type: ChunkType,
    /// Location in document
    pub location: ContentLocation,
    /// Associated metadata
    pub metadata: HashMap<String, String>,
    /// Embedding vector (if computed)
    pub embedding: Option<Vector>,
}

/// Content chunk types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkType {
    Paragraph,
    Heading,
    Table,
    List,
    Quote,
    Code,
    Caption,
    Footnote,
    Header,
    Footer,
}

/// Content location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentLocation {
    /// Page number (1-indexed)
    pub page: Option<usize>,
    /// Section number
    pub section: Option<usize>,
    /// Character offset in document
    pub char_offset: Option<usize>,
    /// Line number
    pub line: Option<usize>,
    /// Column number
    pub column: Option<usize>,
}

/// Cross-modal embedding that combines multiple modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalEmbedding {
    /// Combined embedding vector
    pub embedding: Vector,
    /// Modalities included in this embedding
    pub modalities: Vec<Modality>,
    /// Fusion strategy used
    pub fusion_strategy: FusionStrategy,
    /// Confidence score for the embedding quality
    pub confidence: f32,
    /// Associated content identifiers
    pub content_ids: Vec<String>,
}

/// Modality types for cross-modal processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
}

/// Fusion strategies for combining modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Simple concatenation of features
    Concatenation,
    /// Weighted average of embeddings
    WeightedAverage(Vec<f32>), // weights for each modality
    /// Attention-based fusion
    Attention,
    /// Late fusion with score combination
    LateFusion,
    /// Multi-layer perceptron fusion
    MlpFusion,
    /// Transformer-based fusion
    TransformerFusion,
}

/// Processing statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProcessingStats {
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Total characters extracted
    pub total_chars: usize,
    /// Total words extracted
    pub total_words: usize,
    /// Number of images found
    pub image_count: usize,
    /// Number of tables found
    pub table_count: usize,
    /// Number of links found
    pub link_count: usize,
    /// Number of chunks created
    pub chunk_count: usize,
    /// Number of audio files processed
    pub audio_count: usize,
    /// Number of video files processed
    pub video_count: usize,
    /// Number of cross-modal embeddings generated
    pub cross_modal_embedding_count: usize,
    /// Total time spent on image processing (ms)
    pub image_processing_time_ms: u64,
    /// Total time spent on audio processing (ms)
    pub audio_processing_time_ms: u64,
    /// Total time spent on video processing (ms)
    pub video_processing_time_ms: u64,
    /// Processing warnings
    pub warnings: Vec<String>,
}