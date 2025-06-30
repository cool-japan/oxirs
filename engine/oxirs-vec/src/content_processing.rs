//! Advanced content processing for multiple document formats
//!
//! This module provides comprehensive document parsing and content extraction
//! capabilities for PDF, HTML, XML, office documents, and multimedia content.
//!
//! This module is only available when the `content-processing` feature is enabled.

#[cfg(feature = "content-processing")]
use crate::{embeddings::EmbeddableContent, Vector};
#[cfg(feature = "content-processing")]
use anyhow::{anyhow, Result};
#[cfg(feature = "content-processing")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "content-processing")]
use std::collections::HashMap;
#[cfg(feature = "content-processing")]
use std::path::Path;

/// Document format types supported by the content processor
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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

#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationLabel {
    /// Class label
    pub label: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
}

/// Image complexity metrics
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
}

/// Fusion strategies for combining modalities
#[cfg(feature = "content-processing")]
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
#[cfg(feature = "content-processing")]
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

/// Advanced content processor
#[cfg(feature = "content-processing")]
pub struct ContentProcessor {
    config: ContentExtractionConfig,
    format_handlers: HashMap<DocumentFormat, Box<dyn FormatHandler>>,
}

/// Trait for format-specific content handlers
#[cfg(feature = "content-processing")]
pub trait FormatHandler: Send + Sync {
    /// Extract content from document bytes
    fn extract_content(
        &self,
        data: &[u8],
        config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent>;

    /// Check if this handler can process the given data
    fn can_handle(&self, data: &[u8]) -> bool;

    /// Get supported file extensions
    fn supported_extensions(&self) -> Vec<&'static str>;
}

#[cfg(feature = "content-processing")]
impl ContentProcessor {
    /// Create a new content processor with default configuration
    pub fn new() -> Self {
        let mut processor = Self {
            config: ContentExtractionConfig::default(),
            format_handlers: HashMap::new(),
        };

        // Register default format handlers
        processor.register_default_handlers();
        processor
    }

    /// Create content processor with custom configuration
    pub fn with_config(config: ContentExtractionConfig) -> Self {
        let mut processor = Self {
            config,
            format_handlers: HashMap::new(),
        };

        processor.register_default_handlers();
        processor
    }

    /// Register a custom format handler
    pub fn register_handler(&mut self, format: DocumentFormat, handler: Box<dyn FormatHandler>) {
        self.format_handlers.insert(format, handler);
    }

    /// Process document from file path
    pub fn process_file<P: AsRef<Path>>(&self, path: P) -> Result<ExtractedContent> {
        let path = path.as_ref();
        let data = std::fs::read(path)?;
        let format = self.detect_format(&data, Some(path))?;
        self.process_document(&data, format)
    }

    /// Process document from raw bytes
    pub fn process_document(
        &self,
        data: &[u8],
        format: DocumentFormat,
    ) -> Result<ExtractedContent> {
        let start_time = std::time::Instant::now();

        let handler = self
            .format_handlers
            .get(&format)
            .ok_or_else(|| anyhow!("No handler available for format: {:?}", format))?;

        if !handler.can_handle(data) {
            return Err(anyhow!("Handler cannot process this document"));
        }

        let mut content = handler.extract_content(data, &self.config)?;

        // Post-process content
        self.post_process_content(&mut content)?;

        // Update processing stats
        content.processing_stats.processing_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(content)
    }

    /// Detect document format from content and file extension
    pub fn detect_format(&self, data: &[u8], path: Option<&Path>) -> Result<DocumentFormat> {
        // Check file extension first
        if let Some(path) = path {
            if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
                match extension.to_lowercase().as_str() {
                    "pdf" => return Ok(DocumentFormat::Pdf),
                    "html" | "htm" => return Ok(DocumentFormat::Html),
                    "xml" => return Ok(DocumentFormat::Xml),
                    "md" | "markdown" => return Ok(DocumentFormat::Markdown),
                    "txt" => return Ok(DocumentFormat::PlainText),
                    "docx" => return Ok(DocumentFormat::Docx),
                    "pptx" => return Ok(DocumentFormat::Pptx),
                    "xlsx" => return Ok(DocumentFormat::Xlsx),
                    "rtf" => return Ok(DocumentFormat::Rtf),
                    "epub" => return Ok(DocumentFormat::Epub),
                    "json" => return Ok(DocumentFormat::Json),
                    "csv" => return Ok(DocumentFormat::Csv),
                    // Image formats
                    "jpg" | "jpeg" | "png" | "gif" | "bmp" | "tiff" | "tif" | "webp" | "svg" => {
                        return Ok(DocumentFormat::Image)
                    }
                    // Audio formats
                    "mp3" | "wav" | "flac" | "ogg" | "aac" | "m4a" | "opus" | "wma" => {
                        return Ok(DocumentFormat::Audio)
                    }
                    // Video formats
                    "mp4" | "avi" | "mkv" | "mov" | "wmv" | "flv" | "webm" | "m4v" | "3gp" => {
                        return Ok(DocumentFormat::Video)
                    }
                    _ => {}
                }
            }
        }

        // Fallback to content-based detection
        self.detect_format_by_content(data)
    }

    /// Detect format by analyzing content
    fn detect_format_by_content(&self, data: &[u8]) -> Result<DocumentFormat> {
        if data.len() < 4 {
            return Ok(DocumentFormat::PlainText);
        }

        // Check magic bytes
        let magic = &data[0..4];

        match magic {
            [0x25, 0x50, 0x44, 0x46] => Ok(DocumentFormat::Pdf), // %PDF
            [0x50, 0x4B, 0x03, 0x04] | [0x50, 0x4B, 0x05, 0x06] => {
                // ZIP file - could be Office document
                if self.is_office_document(data) {
                    // Would need more sophisticated detection here
                    Ok(DocumentFormat::Docx)
                } else {
                    Ok(DocumentFormat::Unknown)
                }
            }
            // Image formats
            [0xFF, 0xD8, 0xFF, _] => Ok(DocumentFormat::Image), // JPEG
            [0x89, 0x50, 0x4E, 0x47] => Ok(DocumentFormat::Image), // PNG
            [0x47, 0x49, 0x46, 0x38] => Ok(DocumentFormat::Image), // GIF
            [0x42, 0x4D, _, _] => Ok(DocumentFormat::Image), // BMP
            [0x49, 0x49, 0x2A, 0x00] | [0x4D, 0x4D, 0x00, 0x2A] => Ok(DocumentFormat::Image), // TIFF
            [0x52, 0x49, 0x46, 0x46] => {
                // RIFF format - could be WebP, AVI, or WAV
                if data.len() >= 12 && &data[8..12] == b"WEBP" {
                    Ok(DocumentFormat::Image) // WebP
                } else if data.len() >= 12 && &data[8..12] == b"AVI " {
                    Ok(DocumentFormat::Video) // AVI
                } else if data.len() >= 12 && &data[8..12] == b"WAVE" {
                    Ok(DocumentFormat::Audio) // WAV
                } else {
                    Ok(DocumentFormat::Unknown)
                }
            }
            // Audio formats
            [0x49, 0x44, 0x33, _] => Ok(DocumentFormat::Audio), // MP3 with ID3 tag
            [0xFF, 0xFB, _, _] | [0xFF, 0xF3, _, _] | [0xFF, 0xF2, _, _] => Ok(DocumentFormat::Audio), // MP3
            [0x66, 0x4C, 0x61, 0x43] => Ok(DocumentFormat::Audio), // FLAC
            [0x4F, 0x67, 0x67, 0x53] => Ok(DocumentFormat::Audio), // OGG
            // Video formats
            [0x00, 0x00, 0x00, 0x18] | [0x00, 0x00, 0x00, 0x20] => {
                // MP4/M4V (check ftyp box)
                if data.len() >= 8 && &data[4..8] == b"ftyp" {
                    Ok(DocumentFormat::Video)
                } else {
                    Ok(DocumentFormat::Unknown)
                }
            }
            [0x1A, 0x45, 0xDF, 0xA3] => Ok(DocumentFormat::Video), // Matroska/WebM
            [0x46, 0x4C, 0x56, 0x01] => Ok(DocumentFormat::Video), // FLV
            _ => {
                // Check for text-based formats
                if data.starts_with(b"<!DOCTYPE") || data.starts_with(b"<html") {
                    Ok(DocumentFormat::Html)
                } else if data.starts_with(b"<?xml") {
                    Ok(DocumentFormat::Xml)
                } else if data.starts_with(b"{") || data.starts_with(b"[") {
                    Ok(DocumentFormat::Json)
                } else if data.starts_with(b"<svg") {
                    Ok(DocumentFormat::Image) // SVG
                } else {
                    // Default to plain text for now
                    Ok(DocumentFormat::PlainText)
                }
            }
        }
    }

    /// Check if ZIP file contains Office document
    fn is_office_document(&self, _data: &[u8]) -> bool {
        // Simplified check - in practice, would examine ZIP contents
        // for Office-specific files like word/document.xml, ppt/presentation.xml, etc.
        false
    }

    /// Register default format handlers
    fn register_default_handlers(&mut self) {
        self.register_handler(DocumentFormat::PlainText, Box::new(PlainTextHandler));
        self.register_handler(DocumentFormat::Html, Box::new(HtmlHandler));
        self.register_handler(DocumentFormat::Xml, Box::new(XmlHandler));
        self.register_handler(DocumentFormat::Markdown, Box::new(MarkdownHandler));
        self.register_handler(DocumentFormat::Json, Box::new(JsonHandler));
        self.register_handler(DocumentFormat::Csv, Box::new(CsvHandler));

        // Register advanced format handlers
        self.register_handler(DocumentFormat::Pdf, Box::new(PdfHandler));
        self.register_handler(DocumentFormat::Docx, Box::new(DocxHandler));
        self.register_handler(DocumentFormat::Pptx, Box::new(PptxHandler));
        self.register_handler(DocumentFormat::Xlsx, Box::new(XlsxHandler));

        // Register multimedia format handlers
        #[cfg(feature = "images")]
        self.register_handler(DocumentFormat::Image, Box::new(ImageHandler));
        self.register_handler(DocumentFormat::Audio, Box::new(AudioHandler));
        self.register_handler(DocumentFormat::Video, Box::new(VideoHandler));
    }

    /// Post-process extracted content
    fn post_process_content(&self, content: &mut ExtractedContent) -> Result<()> {
        // Language detection
        if self.config.detect_language && content.language.is_none() {
            content.language = self.detect_language(&content.text);
        }

        // Content chunking
        if content.chunks.is_empty() {
            content.chunks = self.create_chunks(&content.text, &content.structure)?;
        }

        // Process multimedia features if enabled
        if self.config.extract_multimedia_features {
            self.process_multimedia_features(content)?;
        }

        // Generate cross-modal embeddings if enabled
        if self.config.generate_image_embeddings
            || self.config.extract_audio_features
            || self.config.extract_video_features
        {
            self.generate_cross_modal_embeddings(content)?;
        }

        // Update statistics
        content.processing_stats.total_chars = content.text.len();
        content.processing_stats.total_words = content.text.split_whitespace().count();
        content.processing_stats.chunk_count = content.chunks.len();
        content.processing_stats.image_count = content.images.len();
        content.processing_stats.table_count = content.tables.len();
        content.processing_stats.link_count = content.links.len();
        content.processing_stats.audio_count = content.audio_content.len();
        content.processing_stats.video_count = content.video_content.len();
        content.processing_stats.cross_modal_embedding_count = content.cross_modal_embeddings.len();

        Ok(())
    }

    /// Process multimedia features for all content
    fn process_multimedia_features(&self, content: &mut ExtractedContent) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Process image features
        if self.config.generate_image_embeddings {
            for image in &mut content.images {
                if let Err(e) = self.process_image_features(image) {
                    content.processing_stats.warnings.push(format!(
                        "Failed to process image features: {}",
                        e
                    ));
                }
            }
        }

        content.processing_stats.image_processing_time_ms += start_time.elapsed().as_millis() as u64;

        // Process audio features
        if self.config.extract_audio_features {
            let audio_start = std::time::Instant::now();
            for audio in &mut content.audio_content {
                if let Err(e) = self.process_audio_features(audio) {
                    content.processing_stats.warnings.push(format!(
                        "Failed to process audio features: {}",
                        e
                    ));
                }
            }
            content.processing_stats.audio_processing_time_ms += audio_start.elapsed().as_millis() as u64;
        }

        // Process video features
        if self.config.extract_video_features {
            let video_start = std::time::Instant::now();
            for video in &mut content.video_content {
                if let Err(e) = self.process_video_features(video) {
                    content.processing_stats.warnings.push(format!(
                        "Failed to process video features: {}",
                        e
                    ));
                }
            }
            content.processing_stats.video_processing_time_ms += video_start.elapsed().as_millis() as u64;
        }

        Ok(())
    }

    /// Process image features and generate embeddings
    fn process_image_features(&self, image: &mut ExtractedImage) -> Result<()> {
        #[cfg(feature = "images")]
        {
            // Decode base64 image data
            let image_data = base64::decode(&image.data)
                .map_err(|e| anyhow!("Failed to decode image data: {}", e))?;

            // Load image using image crate
            let img = image::load_from_memory(&image_data)
                .map_err(|e| anyhow!("Failed to load image: {}", e))?;

            // Resize if needed
            let img = if let Some((max_width, max_height)) = self.config.max_image_resolution {
                if img.width() > max_width || img.height() > max_height {
                    img.resize(max_width, max_height, image::imageops::FilterType::Lanczos3)
                } else {
                    img
                }
            } else {
                img
            };

            // Extract visual features
            let visual_features = self.extract_image_features(&img)?;
            image.visual_features = Some(visual_features);

            // Generate image embedding (simplified implementation)
            let embedding = self.generate_image_embedding(&img)?;
            image.embedding = Some(embedding);

            // Basic object detection (placeholder)
            image.detected_objects = self.detect_objects(&img)?;

            // Basic image classification (placeholder)
            image.classification_labels = self.classify_image(&img)?;
        }

        #[cfg(not(feature = "images"))]
        {
            return Err(anyhow!("Image processing requires 'images' feature"));
        }

        Ok(())
    }

    /// Extract basic image features
    #[cfg(feature = "images")]
    fn extract_image_features(&self, img: &image::DynamicImage) -> Result<ImageFeatures> {
        let rgb_img = img.to_rgb8();
        let (width, height) = (rgb_img.width(), rgb_img.height());

        // Extract color histogram
        let color_histogram = self.extract_color_histogram(&rgb_img)?;

        // Extract dominant colors
        let dominant_colors = self.extract_dominant_colors(&rgb_img);

        // Calculate complexity metrics
        let complexity_metrics = self.calculate_image_complexity(&rgb_img)?;

        Ok(ImageFeatures {
            color_histogram: Some(color_histogram),
            texture_features: None, // TODO: Implement texture features
            edge_features: None,    // TODO: Implement edge features
            sift_features: None,    // TODO: Implement SIFT features
            cnn_features: None,     // TODO: Implement CNN features
            dominant_colors,
            complexity_metrics,
        })
    }

    /// Extract color histogram from RGB image
    #[cfg(feature = "images")]
    fn extract_color_histogram(&self, img: &image::RgbImage) -> Result<Vec<f32>> {
        const BINS: usize = 64; // 4 bins per channel (4^3 = 64 total bins)
        let mut histogram = vec![0.0; BINS];

        for pixel in img.pixels() {
            let r_bin = (pixel[0] as usize * 4) / 256;
            let g_bin = (pixel[1] as usize * 4) / 256;
            let b_bin = (pixel[2] as usize * 4) / 256;
            let bin_index = r_bin * 16 + g_bin * 4 + b_bin;
            histogram[bin_index.min(BINS - 1)] += 1.0;
        }

        // Normalize histogram
        let total_pixels = (img.width() * img.height()) as f32;
        for value in &mut histogram {
            *value /= total_pixels;
        }

        Ok(histogram)
    }

    /// Extract dominant colors from image
    #[cfg(feature = "images")]
    fn extract_dominant_colors(&self, img: &image::RgbImage) -> Vec<(u8, u8, u8)> {
        // Simple k-means clustering for dominant colors (simplified implementation)
        let mut color_counts: std::collections::HashMap<(u8, u8, u8), u32> = std::collections::HashMap::new();

        // Sample every 10th pixel for performance
        for (x, y, pixel) in img.enumerate_pixels() {
            if x % 10 == 0 && y % 10 == 0 {
                // Quantize to reduce color space
                let quantized = (
                    (pixel[0] / 32) * 32,
                    (pixel[1] / 32) * 32,
                    (pixel[2] / 32) * 32,
                );
                *color_counts.entry(quantized).or_insert(0) += 1;
            }
        }

        // Get top 5 most common colors
        let mut colors: Vec<_> = color_counts.into_iter().collect();
        colors.sort_by(|a, b| b.1.cmp(&a.1));
        colors.into_iter().take(5).map(|(color, _)| color).collect()
    }

    /// Calculate image complexity metrics
    #[cfg(feature = "images")]
    fn calculate_image_complexity(&self, img: &image::RgbImage) -> Result<ImageComplexityMetrics> {
        let (width, height) = (img.width() as usize, img.height() as usize);

        // Calculate edge density using simple sobel operator
        let mut edge_count = 0;
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let center = img.get_pixel(x as u32, y as u32);
                let right = img.get_pixel((x + 1) as u32, y as u32);
                let bottom = img.get_pixel(x as u32, (y + 1) as u32);

                let dx = ((right[0] as i32 - center[0] as i32).abs()
                    + (right[1] as i32 - center[1] as i32).abs()
                    + (right[2] as i32 - center[2] as i32).abs()) as f32
                    / 3.0;

                let dy = ((bottom[0] as i32 - center[0] as i32).abs()
                    + (bottom[1] as i32 - center[1] as i32).abs()
                    + (bottom[2] as i32 - center[2] as i32).abs()) as f32
                    / 3.0;

                if (dx * dx + dy * dy).sqrt() > 30.0 {
                    edge_count += 1;
                }
            }
        }

        let edge_density = edge_count as f32 / ((width - 2) * (height - 2)) as f32;

        // Calculate color diversity (number of unique colors / total possible)
        let mut unique_colors = std::collections::HashSet::new();
        for pixel in img.pixels() {
            unique_colors.insert((pixel[0], pixel[1], pixel[2]));
        }
        let color_diversity = (unique_colors.len() as f32 / (width * height) as f32).min(1.0);

        // Simple texture complexity (variance in local neighborhoods)
        let mut texture_variance = 0.0;
        let window_size = 5;
        for y in window_size..height - window_size {
            for x in window_size..width - window_size {
                let mut values = Vec::new();
                for dy in -(window_size as i32)..(window_size as i32) {
                    for dx in -(window_size as i32)..(window_size as i32) {
                        let pixel = img.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32);
                        let gray = (pixel[0] as f32 * 0.299 + pixel[1] as f32 * 0.587 + pixel[2] as f32 * 0.114);
                        values.push(gray);
                    }
                }
                let mean = values.iter().sum::<f32>() / values.len() as f32;
                let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
                texture_variance += variance;
            }
        }
        texture_variance /= ((width - 2 * window_size) * (height - 2 * window_size)) as f32;
        let texture_complexity = (texture_variance / 10000.0).min(1.0); // Normalize to 0-1

        // Calculate entropy (simplified)
        let entropy = -(unique_colors.len() as f32).log2() / 16.0; // Normalize roughly

        Ok(ImageComplexityMetrics {
            edge_density,
            color_diversity,
            texture_complexity,
            entropy,
        })
    }

    /// Generate image embedding (placeholder implementation)
    #[cfg(feature = "images")]
    fn generate_image_embedding(&self, img: &image::DynamicImage) -> Result<Vector> {
        // Simplified embedding generation - resize to fixed size and flatten
        let resized = img.resize(32, 32, image::imageops::FilterType::Lanczos3);
        let rgb = resized.to_rgb8();
        
        let mut embedding = Vec::new();
        for pixel in rgb.pixels() {
            embedding.push(pixel[0] as f32 / 255.0);
            embedding.push(pixel[1] as f32 / 255.0);
            embedding.push(pixel[2] as f32 / 255.0);
        }

        // Pad or truncate to standard embedding size
        embedding.resize(384, 0.0); // Standard embedding size

        Ok(Vector::new(embedding))
    }

    /// Detect objects in image (placeholder implementation)
    #[cfg(feature = "images")]
    fn detect_objects(&self, _img: &image::DynamicImage) -> Result<Vec<DetectedObject>> {
        // Placeholder: In practice, would use YOLO, R-CNN, etc.
        Ok(vec![])
    }

    /// Classify image (placeholder implementation)
    #[cfg(feature = "images")]
    fn classify_image(&self, _img: &image::DynamicImage) -> Result<Vec<ClassificationLabel>> {
        // Placeholder: In practice, would use CNN models like ResNet, EfficientNet, etc.
        Ok(vec![])
    }

    /// Process audio features (placeholder implementation)
    fn process_audio_features(&self, audio: &mut ExtractedAudio) -> Result<()> {
        // Placeholder: Audio feature extraction would require audio processing libraries
        // like FFTW, librosa equivalent, etc.
        audio.audio_features = Some(AudioFeatures {
            mfcc: None,
            spectral_features: None,
            rhythm_features: None,
            harmonic_features: None,
            zero_crossing_rate: 0.0,
            energy_metrics: AudioEnergyMetrics {
                rms_energy: 0.0,
                peak_amplitude: 0.0,
                average_loudness: 0.0,
                dynamic_range: 0.0,
            },
        });
        Ok(())
    }

    /// Process video features (placeholder implementation)
    fn process_video_features(&self, video: &mut ExtractedVideo) -> Result<()> {
        // Placeholder: Video feature extraction would require video processing libraries
        // like FFmpeg bindings, OpenCV, etc.
        video.video_analysis = Some(VideoAnalysis {
            scenes: vec![],
            motion_analysis: None,
            activity_level: 0.0,
            color_timeline: vec![],
        });
        Ok(())
    }

    /// Generate cross-modal embeddings combining multiple modalities
    fn generate_cross_modal_embeddings(&self, content: &mut ExtractedContent) -> Result<()> {
        let mut embeddings = Vec::new();

        // Simple cross-modal embedding: combine text and image embeddings
        if !content.text.is_empty() && !content.images.is_empty() {
            for image in &content.images {
                if let Some(image_embedding) = &image.embedding {
                    let text_embedding = self.generate_simple_text_embedding(&content.text);
                    let combined = self.combine_embeddings(&text_embedding, image_embedding, FusionStrategy::WeightedAverage(vec![0.5, 0.5]))?;
                    
                    embeddings.push(CrossModalEmbedding {
                        embedding: combined,
                        modalities: vec![Modality::Text, Modality::Image],
                        fusion_strategy: FusionStrategy::WeightedAverage(vec![0.5, 0.5]),
                        confidence: 0.8, // Placeholder confidence
                        content_ids: vec!["text".to_string(), format!("image_{}", embeddings.len())],
                    });
                }
            }
        }

        content.cross_modal_embeddings = embeddings;
        Ok(())
    }

    /// Generate simple text embedding (placeholder)
    fn generate_simple_text_embedding(&self, text: &str) -> Vector {
        // Simple bag-of-words embedding (placeholder)
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut embedding = vec![0.0; 384];
        
        // Simple hash-based embedding
        for (i, word) in words.iter().enumerate().take(384) {
            let hash = word.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
            embedding[i] = (hash as f32) / (u32::MAX as f32);
        }
        
        Vector::new(embedding)
    }

    /// Combine embeddings using specified fusion strategy
    fn combine_embeddings(&self, emb1: &Vector, emb2: &Vector, strategy: FusionStrategy) -> Result<Vector> {
        let vec1 = emb1.as_f32();
        let vec2 = emb2.as_f32();
        
        let combined = match strategy {
            FusionStrategy::Concatenation => {
                let mut result = vec1.clone();
                result.extend(vec2);
                result
            }
            FusionStrategy::WeightedAverage(weights) => {
                if weights.len() != 2 {
                    return Err(anyhow!("WeightedAverage requires exactly 2 weights"));
                }
                vec1.iter().zip(vec2.iter())
                    .map(|(a, b)| a * weights[0] + b * weights[1])
                    .collect()
            }
            _ => {
                // Default to simple average for other strategies
                vec1.iter().zip(vec2.iter())
                    .map(|(a, b)| (a + b) / 2.0)
                    .collect()
            }
        };
        
        Ok(Vector::new(combined))
    }

    /// Detect language of text content
    fn detect_language(&self, text: &str) -> Option<String> {
        // Simplified language detection
        // In practice, would use a proper language detection library
        if text.len() < 100 {
            return None;
        }

        // Basic heuristics for common languages
        let english_words = [
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        ];
        let spanish_words = [
            "el", "la", "de", "que", "y", "a", "en", "un", "es", "se", "no", "te",
        ];
        let french_words = [
            "le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir", "que", "pour",
        ];

        let lowercase_text = text.to_lowercase();
        let words: Vec<&str> = lowercase_text.split_whitespace().take(100).collect();

        let english_count = words.iter().filter(|w| english_words.contains(w)).count();
        let spanish_count = words.iter().filter(|w| spanish_words.contains(w)).count();
        let french_count = words.iter().filter(|w| french_words.contains(w)).count();

        let max_count = [english_count, spanish_count, french_count]
            .into_iter()
            .max()
            .unwrap();

        if max_count >= words.len() / 10 {
            if english_count == max_count {
                Some("en".to_string())
            } else if spanish_count == max_count {
                Some("es".to_string())
            } else {
                Some("fr".to_string())
            }
        } else {
            None
        }
    }

    /// Create content chunks based on strategy
    fn create_chunks(
        &self,
        text: &str,
        structure: &DocumentStructure,
    ) -> Result<Vec<ContentChunk>> {
        let mut chunks = Vec::new();

        match &self.config.chunking_strategy {
            ChunkingStrategy::Paragraph => {
                for (i, paragraph) in text.split("\n\n").enumerate() {
                    if !paragraph.trim().is_empty() {
                        chunks.push(ContentChunk {
                            text: paragraph.trim().to_string(),
                            chunk_type: ChunkType::Paragraph,
                            location: ContentLocation {
                                page: None,
                                section: None,
                                char_offset: None,
                                line: Some(i),
                                column: None,
                            },
                            metadata: HashMap::new(),
                            embedding: None,
                        });
                    }
                }
            }
            ChunkingStrategy::Sentence => {
                // Simple sentence splitting
                for (i, sentence) in text.split('.').enumerate() {
                    let sentence = sentence.trim();
                    if !sentence.is_empty() && sentence.len() > 10 {
                        chunks.push(ContentChunk {
                            text: format!("{}.", sentence),
                            chunk_type: ChunkType::Paragraph,
                            location: ContentLocation {
                                page: None,
                                section: None,
                                char_offset: None,
                                line: Some(i),
                                column: None,
                            },
                            metadata: HashMap::new(),
                            embedding: None,
                        });
                    }
                }
            }
            ChunkingStrategy::FixedTokens(token_count) => {
                let words: Vec<&str> = text.split_whitespace().collect();
                for (i, chunk_words) in words.chunks(*token_count).enumerate() {
                    let chunk_text = chunk_words.join(" ");
                    chunks.push(ContentChunk {
                        text: chunk_text,
                        chunk_type: ChunkType::Paragraph,
                        location: ContentLocation {
                            page: None,
                            section: None,
                            char_offset: Some(i * token_count),
                            line: None,
                            column: None,
                        },
                        metadata: HashMap::new(),
                        embedding: None,
                    });
                }
            }
            ChunkingStrategy::Semantic => {
                // For now, fallback to paragraph chunking
                // In practice, would use more sophisticated semantic segmentation
                return self.create_chunks(text, structure);
            }
            ChunkingStrategy::Page => {
                // Use document structure if available
                if structure.page_count > 0 {
                    // Would need page boundaries from document structure
                    // For now, split into equal parts
                    let words_per_page =
                        text.split_whitespace().count() / structure.page_count.max(1);
                    let words: Vec<&str> = text.split_whitespace().collect();

                    for (page_num, page_words) in words.chunks(words_per_page).enumerate() {
                        let page_text = page_words.join(" ");
                        chunks.push(ContentChunk {
                            text: page_text,
                            chunk_type: ChunkType::Paragraph,
                            location: ContentLocation {
                                page: Some(page_num + 1),
                                section: None,
                                char_offset: None,
                                line: None,
                                column: None,
                            },
                            metadata: HashMap::new(),
                            embedding: None,
                        });
                    }
                } else {
                    // Fallback to paragraph chunking
                    return self.create_chunks(text, structure);
                }
            }
            ChunkingStrategy::Custom(pattern) => {
                // Use regex pattern for splitting
                match regex::Regex::new(pattern) {
                    Ok(re) => {
                        for (i, chunk_text) in re.split(text).enumerate() {
                            if !chunk_text.trim().is_empty() {
                                chunks.push(ContentChunk {
                                    text: chunk_text.trim().to_string(),
                                    chunk_type: ChunkType::Paragraph,
                                    location: ContentLocation {
                                        page: None,
                                        section: None,
                                        char_offset: None,
                                        line: Some(i),
                                        column: None,
                                    },
                                    metadata: HashMap::new(),
                                    embedding: None,
                                });
                            }
                        }
                    }
                    Err(_) => {
                        // Fallback to paragraph chunking if regex is invalid
                        return self.create_chunks(text, structure);
                    }
                }
            }
        }

        Ok(chunks)
    }

    /// Convert extracted content to embeddable content
    pub fn to_embeddable_content(&self, content: &ExtractedContent) -> Vec<EmbeddableContent> {
        let mut embeddable_content = Vec::new();

        // Main document content
        if !content.text.is_empty() {
            embeddable_content.push(EmbeddableContent::Text(content.text.clone()));
        }

        // Individual chunks
        for chunk in &content.chunks {
            embeddable_content.push(EmbeddableContent::Text(chunk.text.clone()));
        }

        // Table content
        for table in &content.tables {
            let table_text = format!(
                "Table: {} | Headers: {} | Data: {}",
                table.caption.as_deref().unwrap_or(""),
                table.headers.join(", "),
                table
                    .rows
                    .iter()
                    .map(|row| row.join(" | "))
                    .collect::<Vec<_>>()
                    .join(" || ")
            );
            embeddable_content.push(EmbeddableContent::Text(table_text));
        }

        // Image content
        for image in &content.images {
            let image_text = format!(
                "Image: {} format, {}x{} pixels{}{}",
                image.format,
                image.width,
                image.height,
                image.alt_text.as_ref().map(|alt| format!(", alt: {}", alt)).unwrap_or_default(),
                image.caption.as_ref().map(|cap| format!(", caption: {}", cap)).unwrap_or_default()
            );
            embeddable_content.push(EmbeddableContent::Text(image_text));

            // Include detected object information
            if !image.detected_objects.is_empty() {
                let objects_text = format!(
                    "Detected objects: {}",
                    image.detected_objects
                        .iter()
                        .map(|obj| format!("{} ({:.2})", obj.label, obj.confidence))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                embeddable_content.push(EmbeddableContent::Text(objects_text));
            }

            // Include classification labels
            if !image.classification_labels.is_empty() {
                let labels_text = format!(
                    "Image labels: {}",
                    image.classification_labels
                        .iter()
                        .map(|label| format!("{} ({:.2})", label.label, label.confidence))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                embeddable_content.push(EmbeddableContent::Text(labels_text));
            }
        }

        // Audio content
        for audio in &content.audio_content {
            let audio_text = format!(
                "Audio: {} format, {:.1}s duration, {}Hz sample rate, {} channels",
                audio.format, audio.duration, audio.sample_rate, audio.channels
            );
            embeddable_content.push(EmbeddableContent::Text(audio_text));

            // Include transcription if available
            if let Some(transcription) = &audio.transcription {
                embeddable_content.push(EmbeddableContent::Text(format!("Transcription: {}", transcription)));
            }

            // Include music analysis if available
            if let Some(music) = &audio.music_analysis {
                let music_text = format!(
                    "Music analysis: {}{}{}",
                    music.tempo.map(|t| format!("tempo {}bpm", t)).unwrap_or_default(),
                    music.key.as_ref().map(|k| format!(", key {}", k)).unwrap_or_default(),
                    music.genre.as_ref().map(|g| format!(", genre {}", g)).unwrap_or_default()
                );
                embeddable_content.push(EmbeddableContent::Text(music_text));
            }

            // Include speech analysis if available
            if let Some(speech) = &audio.speech_analysis {
                let speech_text = format!(
                    "Speech analysis: {}{}{}",
                    speech.language.as_ref().map(|l| format!("language {}", l)).unwrap_or_default(),
                    speech.emotion.as_ref().map(|e| format!(", emotion {}", e)).unwrap_or_default(),
                    speech.speech_rate.map(|r| format!(", rate {:.1} wpm", r)).unwrap_or_default()
                );
                embeddable_content.push(EmbeddableContent::Text(speech_text));
            }
        }

        // Video content
        for video in &content.video_content {
            let video_text = format!(
                "Video: {} format, {:.1}s duration, {:.1}fps, {}x{} resolution",
                video.format, video.duration, video.frame_rate, video.resolution.0, video.resolution.1
            );
            embeddable_content.push(EmbeddableContent::Text(video_text));

            // Include keyframe information
            if !video.keyframes.is_empty() {
                let keyframes_text = format!(
                    "Keyframes: {} scenes at timestamps {}",
                    video.keyframes.len(),
                    video.keyframes
                        .iter()
                        .map(|kf| format!("{:.1}s", kf.timestamp))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                embeddable_content.push(EmbeddableContent::Text(keyframes_text));
            }

            // Include scene information if available
            if let Some(analysis) = &video.video_analysis {
                if !analysis.scenes.is_empty() {
                    let scenes_text = format!(
                        "Video scenes: {}",
                        analysis.scenes
                            .iter()
                            .enumerate()
                            .map(|(i, scene)| format!(
                                "scene {} ({:.1}s-{:.1}s){}",
                                i + 1,
                                scene.start_time,
                                scene.end_time,
                                scene.description.as_ref().map(|d| format!(": {}", d)).unwrap_or_default()
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                    embeddable_content.push(EmbeddableContent::Text(scenes_text));
                }
            }
        }

        // Cross-modal embeddings
        for cross_modal in &content.cross_modal_embeddings {
            let modalities_text = format!(
                "Cross-modal content combining: {}",
                cross_modal.modalities
                    .iter()
                    .map(|m| match m {
                        Modality::Text => "text",
                        Modality::Image => "image",
                        Modality::Audio => "audio",
                        Modality::Video => "video",
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            embeddable_content.push(EmbeddableContent::Text(modalities_text));
        }

        embeddable_content
    }
}

#[cfg(feature = "content-processing")]
impl Default for ContentProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// Format-specific handlers

/// Plain text handler
#[cfg(feature = "content-processing")]
struct PlainTextHandler;

#[cfg(feature = "content-processing")]
impl FormatHandler for PlainTextHandler {
    fn extract_content(
        &self,
        data: &[u8],
        _config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let text = String::from_utf8_lossy(data).to_string();

        Ok(ExtractedContent {
            format: DocumentFormat::PlainText,
            text,
            metadata: HashMap::new(),
            images: Vec::new(),
            tables: Vec::new(),
            links: Vec::new(),
            structure: DocumentStructure {
                title: None,
                headings: Vec::new(),
                page_count: 1,
                section_count: 1,
                table_of_contents: Vec::new(),
            },
            chunks: Vec::new(),
            language: None,
            processing_stats: ProcessingStats::default(),
            audio_content: Vec::new(),
            video_content: Vec::new(),
            cross_modal_embeddings: Vec::new(),
        })
    }

    fn can_handle(&self, data: &[u8]) -> bool {
        // Check if data is valid UTF-8
        String::from_utf8(data.to_vec()).is_ok()
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["txt", "text"]
    }
}

/// HTML handler
#[cfg(feature = "content-processing")]
struct HtmlHandler;

#[cfg(feature = "content-processing")]
impl FormatHandler for HtmlHandler {
    fn extract_content(
        &self,
        data: &[u8],
        config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let html = String::from_utf8_lossy(data);

        // Simple HTML text extraction (in practice, would use a proper HTML parser)
        let text = self.extract_text_from_html(&html);
        let headings = self.extract_headings(&html);
        let links = if config.extract_links {
            self.extract_links(&html)
        } else {
            Vec::new()
        };

        let metadata = self.extract_metadata(&html);
        let title = metadata.get("title").cloned();

        Ok(ExtractedContent {
            format: DocumentFormat::Html,
            text,
            metadata,
            images: Vec::new(), // Would implement image extraction
            tables: Vec::new(), // Would implement table extraction
            links,
            structure: DocumentStructure {
                title,
                headings,
                page_count: 1,
                section_count: 1,
                table_of_contents: Vec::new(),
            },
            chunks: Vec::new(),
            language: None,
            processing_stats: ProcessingStats::default(),
            audio_content: Vec::new(),
            video_content: Vec::new(),
            cross_modal_embeddings: Vec::new(),
        })
    }

    fn can_handle(&self, data: &[u8]) -> bool {
        let content = String::from_utf8_lossy(data);
        content.contains("<html") || content.contains("<!DOCTYPE")
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["html", "htm"]
    }
}

#[cfg(feature = "content-processing")]
impl HtmlHandler {
    fn extract_text_from_html(&self, html: &str) -> String {
        // Very basic HTML text extraction
        // In practice, would use html5ever or similar
        let mut text = html.to_string();

        // Remove script and style elements
        text = regex::Regex::new(r"<script[^>]*>.*?</script>")
            .unwrap()
            .replace_all(&text, "")
            .to_string();
        text = regex::Regex::new(r"<style[^>]*>.*?</style>")
            .unwrap()
            .replace_all(&text, "")
            .to_string();

        // Remove HTML tags
        text = regex::Regex::new(r"<[^>]*>")
            .unwrap()
            .replace_all(&text, " ")
            .to_string();

        // Clean up whitespace
        text = regex::Regex::new(r"\s+")
            .unwrap()
            .replace_all(&text, " ")
            .to_string();

        text.trim().to_string()
    }

    fn extract_headings(&self, html: &str) -> Vec<Heading> {
        let mut headings = Vec::new();

        for level in 1..=6 {
            let pattern = format!(r"<h{}[^>]*>(.*?)</h{}>", level, level);
            if let Ok(re) = regex::Regex::new(&pattern) {
                for (i, capture) in re.captures_iter(html).enumerate() {
                    if let Some(heading_text) = capture.get(1) {
                        let text = regex::Regex::new(r"<[^>]*>")
                            .unwrap()
                            .replace_all(heading_text.as_str(), "")
                            .trim()
                            .to_string();

                        headings.push(Heading {
                            level,
                            text,
                            location: ContentLocation {
                                page: None,
                                section: Some(i),
                                char_offset: None,
                                line: None,
                                column: None,
                            },
                        });
                    }
                }
            }
        }

        headings
    }

    fn extract_links(&self, html: &str) -> Vec<ExtractedLink> {
        let mut links = Vec::new();

        if let Ok(re) = regex::Regex::new(r#"<a[^>]*href=["']([^"']*)["'][^>]*>(.*?)</a>"#) {
            for (i, capture) in re.captures_iter(html).enumerate() {
                if let (Some(url), Some(text)) = (capture.get(1), capture.get(2)) {
                    let link_text = regex::Regex::new(r"<[^>]*>")
                        .unwrap()
                        .replace_all(text.as_str(), "")
                        .trim()
                        .to_string();

                    links.push(ExtractedLink {
                        url: url.as_str().to_string(),
                        text: link_text,
                        title: None,
                        location: ContentLocation {
                            page: None,
                            section: None,
                            char_offset: None,
                            line: Some(i),
                            column: None,
                        },
                    });
                }
            }
        }

        links
    }

    fn extract_metadata(&self, html: &str) -> HashMap<String, String> {
        let mut metadata = HashMap::new();

        // Extract title
        if let Ok(re) = regex::Regex::new(r"<title[^>]*>(.*?)</title>") {
            if let Some(capture) = re.captures(html) {
                if let Some(title) = capture.get(1) {
                    metadata.insert("title".to_string(), title.as_str().trim().to_string());
                }
            }
        }

        // Extract meta tags
        if let Ok(re) = regex::Regex::new(
            r#"<meta[^>]*name=["']([^"']*)["'][^>]*content=["']([^"']*)["'][^>]*>"#,
        ) {
            for capture in re.captures_iter(html) {
                if let (Some(name), Some(content)) = (capture.get(1), capture.get(2)) {
                    metadata.insert(name.as_str().to_string(), content.as_str().to_string());
                }
            }
        }

        metadata
    }
}

/// XML handler
#[cfg(feature = "content-processing")]
struct XmlHandler;

#[cfg(feature = "content-processing")]
impl FormatHandler for XmlHandler {
    fn extract_content(
        &self,
        data: &[u8],
        _config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let xml = String::from_utf8_lossy(data);

        // Basic XML text extraction
        let text = regex::Regex::new(r"<[^>]*>")
            .unwrap()
            .replace_all(&xml, " ")
            .to_string();
        let text = regex::Regex::new(r"\s+")
            .unwrap()
            .replace_all(&text, " ")
            .trim()
            .to_string();

        Ok(ExtractedContent {
            format: DocumentFormat::Xml,
            text,
            metadata: HashMap::new(),
            images: Vec::new(),
            tables: Vec::new(),
            links: Vec::new(),
            structure: DocumentStructure {
                title: None,
                headings: Vec::new(),
                page_count: 1,
                section_count: 1,
                table_of_contents: Vec::new(),
            },
            chunks: Vec::new(),
            language: None,
            processing_stats: ProcessingStats::default(),
            audio_content: Vec::new(),
            video_content: Vec::new(),
            cross_modal_embeddings: Vec::new(),
        })
    }

    fn can_handle(&self, data: &[u8]) -> bool {
        let content = String::from_utf8_lossy(data);
        content.starts_with("<?xml") || content.contains("<") && content.contains(">")
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["xml"]
    }
}

/// Markdown handler
#[cfg(feature = "content-processing")]
struct MarkdownHandler;

#[cfg(feature = "content-processing")]
impl FormatHandler for MarkdownHandler {
    fn extract_content(
        &self,
        data: &[u8],
        _config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let markdown = String::from_utf8_lossy(data).to_string();

        // Extract headings
        let headings = self.extract_markdown_headings(&markdown);

        // Convert markdown to plain text (basic conversion)
        let text = self.markdown_to_text(&markdown);

        Ok(ExtractedContent {
            format: DocumentFormat::Markdown,
            text,
            metadata: HashMap::new(),
            images: Vec::new(),
            tables: Vec::new(),
            links: Vec::new(),
            structure: DocumentStructure {
                title: headings.first().map(|h| h.text.clone()),
                headings,
                page_count: 1,
                section_count: 1,
                table_of_contents: Vec::new(),
            },
            chunks: Vec::new(),
            language: None,
            processing_stats: ProcessingStats::default(),
            audio_content: Vec::new(),
            video_content: Vec::new(),
            cross_modal_embeddings: Vec::new(),
        })
    }

    fn can_handle(&self, data: &[u8]) -> bool {
        let content = String::from_utf8_lossy(data);
        // Check for common markdown patterns
        content.contains("# ")
            || content.contains("## ")
            || content.contains("**")
            || content.contains("*")
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["md", "markdown"]
    }
}

#[cfg(feature = "content-processing")]
impl MarkdownHandler {
    fn extract_markdown_headings(&self, markdown: &str) -> Vec<Heading> {
        let mut headings = Vec::new();

        for (line_num, line) in markdown.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with('#') {
                let level = trimmed.chars().take_while(|&c| c == '#').count();
                if level <= 6 {
                    let text = trimmed.trim_start_matches('#').trim().to_string();
                    if !text.is_empty() {
                        headings.push(Heading {
                            level,
                            text,
                            location: ContentLocation {
                                page: None,
                                section: None,
                                char_offset: None,
                                line: Some(line_num + 1),
                                column: None,
                            },
                        });
                    }
                }
            }
        }

        headings
    }

    fn markdown_to_text(&self, markdown: &str) -> String {
        let mut text = markdown.to_string();

        // Remove markdown formatting
        text = regex::Regex::new(r"#+\s*")
            .unwrap()
            .replace_all(&text, "")
            .to_string();
        text = regex::Regex::new(r"\*\*(.*?)\*\*")
            .unwrap()
            .replace_all(&text, "$1")
            .to_string();
        text = regex::Regex::new(r"\*(.*?)\*")
            .unwrap()
            .replace_all(&text, "$1")
            .to_string();
        text = regex::Regex::new(r"`(.*?)`")
            .unwrap()
            .replace_all(&text, "$1")
            .to_string();
        text = regex::Regex::new(r"\[(.*?)\]\(.*?\)")
            .unwrap()
            .replace_all(&text, "$1")
            .to_string();

        // Clean up whitespace
        text = regex::Regex::new(r"\n\s*\n")
            .unwrap()
            .replace_all(&text, "\n\n")
            .to_string();

        text.trim().to_string()
    }
}

/// JSON handler
#[cfg(feature = "content-processing")]
struct JsonHandler;

#[cfg(feature = "content-processing")]
impl FormatHandler for JsonHandler {
    fn extract_content(
        &self,
        data: &[u8],
        _config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let json_str = String::from_utf8_lossy(data);

        // Parse JSON and extract text content
        let text = match serde_json::from_str::<serde_json::Value>(&json_str) {
            Ok(value) => self.extract_text_from_json(&value),
            Err(_) => json_str.to_string(),
        };

        Ok(ExtractedContent {
            format: DocumentFormat::Json,
            text,
            metadata: HashMap::new(),
            images: Vec::new(),
            tables: Vec::new(),
            links: Vec::new(),
            structure: DocumentStructure {
                title: None,
                headings: Vec::new(),
                page_count: 1,
                section_count: 1,
                table_of_contents: Vec::new(),
            },
            chunks: Vec::new(),
            language: None,
            processing_stats: ProcessingStats::default(),
            audio_content: Vec::new(),
            video_content: Vec::new(),
            cross_modal_embeddings: Vec::new(),
        })
    }

    fn can_handle(&self, data: &[u8]) -> bool {
        let content = String::from_utf8_lossy(data);
        serde_json::from_str::<serde_json::Value>(&content).is_ok()
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["json"]
    }
}

#[cfg(feature = "content-processing")]
impl JsonHandler {
    fn extract_text_from_json(&self, value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(arr) => arr
                .iter()
                .map(|v| self.extract_text_from_json(v))
                .collect::<Vec<_>>()
                .join(" "),
            serde_json::Value::Object(obj) => obj
                .values()
                .map(|v| self.extract_text_from_json(v))
                .collect::<Vec<_>>()
                .join(" "),
            _ => value.to_string(),
        }
    }
}

/// CSV handler
#[cfg(feature = "content-processing")]
struct CsvHandler;

#[cfg(feature = "content-processing")]
impl FormatHandler for CsvHandler {
    fn extract_content(
        &self,
        data: &[u8],
        _config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let csv_str = String::from_utf8_lossy(data);

        // Parse CSV and extract content
        let (text, tables) = self.parse_csv(&csv_str);

        Ok(ExtractedContent {
            format: DocumentFormat::Csv,
            text,
            metadata: HashMap::new(),
            images: Vec::new(),
            tables,
            links: Vec::new(),
            structure: DocumentStructure {
                title: None,
                headings: Vec::new(),
                page_count: 1,
                section_count: 1,
                table_of_contents: Vec::new(),
            },
            chunks: Vec::new(),
            language: None,
            processing_stats: ProcessingStats::default(),
            audio_content: Vec::new(),
            video_content: Vec::new(),
            cross_modal_embeddings: Vec::new(),
        })
    }

    fn can_handle(&self, data: &[u8]) -> bool {
        let content = String::from_utf8_lossy(data);
        // Basic CSV detection
        content.contains(',') || content.contains(';')
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["csv"]
    }
}

#[cfg(feature = "content-processing")]
impl CsvHandler {
    fn parse_csv(&self, csv_str: &str) -> (String, Vec<ExtractedTable>) {
        let lines: Vec<&str> = csv_str.lines().collect();
        if lines.is_empty() {
            return (String::new(), Vec::new());
        }

        // Simple CSV parsing (in practice, would use csv crate)
        let headers: Vec<String> = lines[0].split(',').map(|s| s.trim().to_string()).collect();
        let mut rows = Vec::new();

        for line in lines.iter().skip(1) {
            let row: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
            if row.len() == headers.len() {
                rows.push(row);
            }
        }

        // Create text representation
        let mut text_parts = vec![headers.join(" | ")];
        for row in &rows {
            text_parts.push(row.join(" | "));
        }
        let text = text_parts.join("\n");

        // Create table structure
        let table = ExtractedTable {
            headers,
            rows,
            caption: None,
            location: ContentLocation {
                page: Some(1),
                section: None,
                char_offset: None,
                line: None,
                column: None,
            },
        };

        (text, vec![table])
    }
}

/// Fallback handler for unsupported formats
#[cfg(feature = "content-processing")]
struct FallbackHandler(String);

#[cfg(feature = "content-processing")]
impl FormatHandler for FallbackHandler {
    fn extract_content(
        &self,
        data: &[u8],
        _config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        // Basic text extraction attempt
        let text = if let Ok(utf8_text) = String::from_utf8(data.to_vec()) {
            utf8_text
        } else {
            format!(
                "Binary content ({} bytes) - {} format not fully supported",
                data.len(),
                self.0
            )
        };

        Ok(ExtractedContent {
            format: DocumentFormat::Unknown,
            text,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("format".to_string(), self.0.clone());
                meta.insert("size".to_string(), data.len().to_string());
                meta
            },
            images: Vec::new(),
            tables: Vec::new(),
            links: Vec::new(),
            structure: DocumentStructure {
                title: None,
                headings: Vec::new(),
                page_count: 1,
                section_count: 1,
                table_of_contents: Vec::new(),
            },
            chunks: Vec::new(),
            language: None,
            processing_stats: ProcessingStats::default(),
            audio_content: Vec::new(),
            video_content: Vec::new(),
            cross_modal_embeddings: Vec::new(),
        })
    }

    fn can_handle(&self, _data: &[u8]) -> bool {
        true // Fallback can handle anything
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec![]
    }
}

/// PDF document handler
#[cfg(feature = "content-processing")]
struct PdfHandler;

#[cfg(feature = "content-processing")]
impl FormatHandler for PdfHandler {
    fn extract_content(
        &self,
        data: &[u8],
        config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let text = match pdf_extract::extract_text_from_mem(data) {
            Ok(extracted_text) => {
                if extracted_text.trim().is_empty() {
                    return Err(anyhow!("No text content found in PDF"));
                }
                extracted_text
            }
            Err(e) => {
                return Err(anyhow!("Failed to extract text from PDF: {}", e));
            }
        };

        // Enhanced metadata extraction
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "PDF".to_string());
        metadata.insert("size".to_string(), data.len().to_string());
        metadata.insert("extraction_method".to_string(), "pdf-extract".to_string());
        
        // Try to extract metadata from PDF header
        if let Some(pdf_metadata) = self.extract_pdf_metadata(data) {
            for (key, value) in pdf_metadata {
                metadata.insert(key, value);
            }
        }

        // Enhanced page count estimation
        let estimated_pages = text.matches("\x0C").count().max(1); // Form feed character

        // Extract structural elements
        let headings = self.extract_pdf_headings(&text);
        let tables = if config.extract_tables {
            self.extract_pdf_tables(&text)
        } else {
            Vec::new()
        };
        
        let links = if config.extract_links {
            self.extract_pdf_links(&text)
        } else {
            Vec::new()
        };

        // Enhanced table of contents generation
        let toc = self.generate_table_of_contents(&headings);

        // Extract images (basic implementation)
        let images = if config.extract_images {
            match self.extract_pdf_images(data, config) {
                Ok(imgs) => imgs,
                Err(_) => Vec::new(), // Fail silently for now
            }
        } else {
            Vec::new()
        };

        Ok(ExtractedContent {
            format: DocumentFormat::Pdf,
            text: text.trim().to_string(),
            metadata,
            images,
            tables,
            links,
            structure: DocumentStructure {
                title: self.extract_pdf_title(&text),
                headings: headings.clone(),
                page_count: estimated_pages,
                section_count: headings.len().max(1),
                table_of_contents: toc,
            },
            chunks: Vec::new(),
            language: None,
            processing_stats: ProcessingStats::default(),
            audio_content: Vec::new(),
            video_content: Vec::new(),
            cross_modal_embeddings: Vec::new(),
        })
    }

    fn can_handle(&self, data: &[u8]) -> bool {
        data.len() >= 4 && &data[0..4] == b"%PDF"
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["pdf"]
    }
}

#[cfg(feature = "content-processing")]
impl PdfHandler {
    fn extract_pdf_title(&self, text: &str) -> Option<String> {
        // Look for title-like patterns at the beginning of the document
        let lines: Vec<&str> = text.lines().take(10).collect();
        for line in lines {
            let trimmed = line.trim();
            if trimmed.len() > 5
                && trimmed.len() < 100
                && !trimmed.contains("http")
                && !trimmed.contains("www")
            {
                // Simple heuristic: first substantial line might be title
                return Some(trimmed.to_string());
            }
        }
        None
    }

    fn extract_pdf_headings(&self, text: &str) -> Vec<Heading> {
        let mut headings = Vec::new();

        // Simple heuristic: lines that are shorter, have capital words, and are followed by text
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.len() > 5 && trimmed.len() < 80 {
                // Check if it looks like a heading
                let words: Vec<&str> = trimmed.split_whitespace().collect();
                let capitalized_words = words
                    .iter()
                    .filter(|w| w.chars().next().map_or(false, |c| c.is_uppercase()))
                    .count();

                if capitalized_words >= words.len() / 2 && words.len() <= 10 {
                    headings.push(Heading {
                        level: 1, // Simple assumption - would need better analysis
                        text: trimmed.to_string(),
                        location: ContentLocation {
                            page: None,
                            section: None,
                            char_offset: None,
                            line: Some(i + 1),
                            column: None,
                        },
                    });
                }
            }
        }

        headings
    }

    /// Extract tables from PDF text using pattern recognition
    fn extract_pdf_tables(&self, text: &str) -> Vec<ExtractedTable> {
        let mut tables = Vec::new();
        let lines: Vec<&str> = text.lines().collect();
        
        let mut current_table: Vec<Vec<String>> = Vec::new();
        let mut in_table = false;
        
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            
            // Detect table-like patterns (multiple columns separated by spaces/tabs)
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            
            if parts.len() >= 2 && parts.len() <= 8 {
                // Check if this looks like a table row
                let has_numbers = parts.iter().any(|p| p.parse::<f64>().is_ok());
                let consistent_spacing = trimmed.contains('\t') || 
                    trimmed.matches("  ").count() >= 2;
                
                if has_numbers || consistent_spacing {
                    if !in_table {
                        in_table = true;
                        current_table.clear();
                    }
                    
                    let row: Vec<String> = parts.iter().map(|s| s.to_string()).collect();
                    current_table.push(row);
                } else if in_table && current_table.len() >= 2 {
                    // End of table detected
                    tables.push(ExtractedTable {
                        headers: if current_table.len() > 1 { 
                            current_table[0].clone() 
                        } else { 
                            Vec::new()
                        },
                        rows: current_table[1..].to_vec(),
                        caption: None,
                        location: ContentLocation {
                            page: None,
                            section: None,
                            char_offset: None,
                            line: Some(i + 1),
                            column: None,
                        },
                    });
                    
                    in_table = false;
                    current_table.clear();
                }
            } else if in_table {
                // Non-table line encountered, end current table
                if current_table.len() >= 2 {
                    tables.push(ExtractedTable {
                        headers: if current_table.len() > 1 { 
                            current_table[0].clone() 
                        } else { 
                            Vec::new()
                        },
                        rows: current_table[1..].to_vec(),
                        caption: None,
                        location: ContentLocation {
                            page: None,
                            section: None,
                            char_offset: None,
                            line: Some(i + 1),
                            column: None,
                        },
                    });
                }
                
                in_table = false;
                current_table.clear();
            }
        }
        
        // Handle table at end of document
        if in_table && current_table.len() >= 2 {
            tables.push(ExtractedTable {
                headers: if current_table.len() > 1 { 
                    current_table[0].clone() 
                } else { 
                    Vec::new()
                },
                rows: current_table[1..].to_vec(),
                caption: None,
                location: ContentLocation {
                    page: None,
                    section: None,
                    char_offset: None,
                    line: Some(lines.len()),
                    column: None,
                },
            });
        }
        
        tables
    }

    /// Extract links from PDF text
    fn extract_pdf_links(&self, text: &str) -> Vec<ExtractedLink> {
        let mut links = Vec::new();
        
        // Regular expressions for different link types
        let url_regex = regex::Regex::new(r"https?://[^\s\)]+").unwrap();
        let email_regex = regex::Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap();
        
        // Find HTTP/HTTPS URLs
        for mat in url_regex.find_iter(text) {
            let url = mat.as_str().trim_end_matches(&['.', ',', ')', ']', '}'][..]);
            links.push(ExtractedLink {
                url: url.to_string(),
                text: url.to_string(),
                title: Some("external".to_string()),
                location: ContentLocation {
                    page: None,
                    section: None,
                    char_offset: Some(mat.start()),
                    line: None,
                    column: None,
                },
            });
        }
        
        // Find email addresses
        for mat in email_regex.find_iter(text) {
            let email = mat.as_str();
            links.push(ExtractedLink {
                url: format!("mailto:{}", email),
                text: email.to_string(),
                title: Some("email".to_string()),
                location: ContentLocation {
                    page: None,
                    section: None,
                    char_offset: Some(mat.start()),
                    line: None,
                    column: None,
                },
            });
        }
        
        links
    }

    /// Extract basic metadata from PDF bytes
    fn extract_pdf_metadata(&self, data: &[u8]) -> Option<HashMap<String, String>> {
        let mut metadata = HashMap::new();
        
        // Simple PDF metadata extraction (looking for Info dictionary patterns)
        let content = String::from_utf8_lossy(data).into_owned();
        
        // Look for title
        if let Some(title_match) = regex::Regex::new(r"/Title\s*\(\s*([^)]+)\s*\)")
            .unwrap()
            .captures(&content) {
            if let Some(title) = title_match.get(1) {
                metadata.insert("title".to_string(), title.as_str().to_string());
            }
        }
        
        // Look for author
        if let Some(author_match) = regex::Regex::new(r"/Author\s*\(\s*([^)]+)\s*\)")
            .unwrap()
            .captures(&content) {
            if let Some(author) = author_match.get(1) {
                metadata.insert("author".to_string(), author.as_str().to_string());
            }
        }
        
        // Look for subject
        if let Some(subject_match) = regex::Regex::new(r"/Subject\s*\(\s*([^)]+)\s*\)")
            .unwrap()
            .captures(&content) {
            if let Some(subject) = subject_match.get(1) {
                metadata.insert("subject".to_string(), subject.as_str().to_string());
            }
        }
        
        // Look for creation date
        if let Some(date_match) = regex::Regex::new(r"/CreationDate\s*\(\s*([^)]+)\s*\)")
            .unwrap()
            .captures(&content) {
            if let Some(date) = date_match.get(1) {
                metadata.insert("creation_date".to_string(), date.as_str().to_string());
            }
        }
        
        if metadata.is_empty() {
            None
        } else {
            Some(metadata)
        }
    }

    /// Extract images from PDF (basic implementation)
    fn extract_pdf_images(&self, _data: &[u8], config: &ContentExtractionConfig) -> Result<Vec<ExtractedImage>> {
        if config.extract_images {
            // This is a placeholder implementation
            // In a real scenario, you'd use a PDF library that can extract embedded images
            // For now, just return empty vector
            Ok(Vec::new())
        } else {
            Ok(Vec::new())
        }
    }

    /// Generate table of contents from headings
    fn generate_table_of_contents(&self, headings: &[Heading]) -> Vec<TocEntry> {
        headings.iter().map(|heading| TocEntry {
            title: heading.text.clone(),
            level: heading.level,
            page: heading.location.page,
            location: heading.location.clone(),
        }).collect()
    }
}

/// Base handler for Office documents (DOCX, PPTX, XLSX)
#[cfg(feature = "content-processing")]
trait OfficeDocumentHandler {
    fn extract_from_zip(&self, data: &[u8], main_xml_path: &str) -> Result<String> {
        let cursor = std::io::Cursor::new(data);
        let mut archive = zip::ZipArchive::new(cursor)
            .map_err(|e| anyhow!("Failed to open ZIP archive: {}", e))?;

        // Try to find the main content file
        let file = archive
            .by_name(main_xml_path)
            .map_err(|e| anyhow!("Main content file not found: {}", e))?;

        let content =
            std::io::read_to_string(file).map_err(|e| anyhow!("Failed to read content: {}", e))?;

        self.extract_text_from_xml(&content)
    }

    fn extract_text_from_xml(&self, xml: &str) -> Result<String> {
        let mut reader = quick_xml::Reader::from_str(xml);
        let mut buf = Vec::new();
        let mut text_content = Vec::new();
        let mut in_text = false;

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(quick_xml::events::Event::Start(ref e)) => {
                    match e.name().as_ref() {
                        b"w:t" | b"a:t" | b"c" => in_text = true, // Word text, PowerPoint text, Excel cell
                        _ => {}
                    }
                }
                Ok(quick_xml::events::Event::End(ref e)) => match e.name().as_ref() {
                    b"w:t" | b"a:t" | b"c" => in_text = false,
                    _ => {}
                },
                Ok(quick_xml::events::Event::Text(e)) if in_text => {
                    let text = e.unescape().unwrap_or_default();
                    text_content.push(text.to_string());
                }
                Ok(quick_xml::events::Event::Eof) => break,
                Err(e) => return Err(anyhow!("XML parsing error: {}", e)),
                _ => {}
            }
            buf.clear();
        }

        Ok(text_content.join(" "))
    }

    fn extract_metadata_from_zip(&self, data: &[u8]) -> HashMap<String, String> {
        let mut metadata = HashMap::new();

        let cursor = std::io::Cursor::new(data);
        if let Ok(mut archive) = zip::ZipArchive::new(cursor) {
            // Try to read core properties
            if let Ok(file) = archive.by_name("docProps/core.xml") {
                if let Ok(content) = std::io::read_to_string(file) {
                    // Parse core properties XML
                    let mut reader = quick_xml::Reader::from_str(&content);
                    let mut buf = Vec::new();
                    let mut current_element = String::new();

                    loop {
                        match reader.read_event_into(&mut buf) {
                            Ok(quick_xml::events::Event::Start(ref e)) => {
                                current_element =
                                    String::from_utf8_lossy(e.name().as_ref()).to_string();
                            }
                            Ok(quick_xml::events::Event::Text(e)) => {
                                let text = e.unescape().unwrap_or_default();
                                match current_element.as_str() {
                                    "dc:title" => {
                                        metadata.insert("title".to_string(), text.to_string());
                                    }
                                    "dc:creator" => {
                                        metadata.insert("author".to_string(), text.to_string());
                                    }
                                    "dc:subject" => {
                                        metadata.insert("subject".to_string(), text.to_string());
                                    }
                                    "dc:description" => {
                                        metadata
                                            .insert("description".to_string(), text.to_string());
                                    }
                                    _ => {}
                                }
                            }
                            Ok(quick_xml::events::Event::Eof) => break,
                            _ => {}
                        }
                        buf.clear();
                    }
                }
            }
        }

        metadata.insert("size".to_string(), data.len().to_string());
        metadata
    }
}

/// DOCX document handler
#[cfg(feature = "content-processing")]
struct DocxHandler;

#[cfg(feature = "content-processing")]
impl OfficeDocumentHandler for DocxHandler {}

#[cfg(feature = "content-processing")]
impl FormatHandler for DocxHandler {
    fn extract_content(
        &self,
        data: &[u8],
        _config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let text = self.extract_from_zip(data, "word/document.xml")?;
        let metadata = self.extract_metadata_from_zip(data);
        let title = metadata.get("title").cloned();

        // Extract headings (would need style analysis for proper heading detection)
        let headings = self.extract_docx_headings(&text);

        Ok(ExtractedContent {
            format: DocumentFormat::Docx,
            text,
            metadata,
            images: Vec::new(), // Would require parsing word/media folder
            tables: Vec::new(), // Would require parsing table XML structures
            links: Vec::new(),  // Would require parsing hyperlink relationships
            structure: DocumentStructure {
                title,
                headings,
                page_count: 1, // Would need to analyze page breaks
                section_count: 1,
                table_of_contents: Vec::new(),
            },
            chunks: Vec::new(),
            language: None,
            processing_stats: ProcessingStats::default(),
            audio_content: Vec::new(),
            video_content: Vec::new(),
            cross_modal_embeddings: Vec::new(),
        })
    }

    fn can_handle(&self, data: &[u8]) -> bool {
        if data.len() < 4 {
            return false;
        }

        // Check for ZIP signature
        if &data[0..4] != &[0x50, 0x4B, 0x03, 0x04] && &data[0..4] != &[0x50, 0x4B, 0x05, 0x06] {
            return false;
        }

        // Check if it contains DOCX-specific files
        let cursor = std::io::Cursor::new(data);
        if let Ok(mut archive) = zip::ZipArchive::new(cursor) {
            archive.by_name("word/document.xml").is_ok()
                && archive.by_name("[Content_Types].xml").is_ok()
        } else {
            false
        }
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["docx"]
    }
}

#[cfg(feature = "content-processing")]
impl DocxHandler {
    fn extract_docx_headings(&self, text: &str) -> Vec<Heading> {
        let mut headings = Vec::new();

        // Simple heuristic for headings in extracted text
        for (i, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.len() > 3 && trimmed.len() < 100 {
                // Check if line looks like a heading
                let words: Vec<&str> = trimmed.split_whitespace().collect();
                if words.len() <= 8 && words.len() > 0 {
                    let first_char = trimmed.chars().next().unwrap_or(' ');
                    if first_char.is_uppercase() {
                        headings.push(Heading {
                            level: 1, // Would need style information for proper level detection
                            text: trimmed.to_string(),
                            location: ContentLocation {
                                page: None,
                                section: None,
                                char_offset: None,
                                line: Some(i + 1),
                                column: None,
                            },
                        });
                    }
                }
            }
        }

        headings
    }
}

/// PPTX document handler
#[cfg(feature = "content-processing")]
struct PptxHandler;

#[cfg(feature = "content-processing")]
impl OfficeDocumentHandler for PptxHandler {}

#[cfg(feature = "content-processing")]
impl FormatHandler for PptxHandler {
    fn extract_content(
        &self,
        data: &[u8],
        _config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        // Extract text from all slides
        let mut all_text = Vec::new();
        let cursor = std::io::Cursor::new(data);
        let mut archive = zip::ZipArchive::new(cursor)
            .map_err(|e| anyhow!("Failed to open PPTX archive: {}", e))?;

        // Find all slide files
        let file_names: Vec<String> = (0..archive.len())
            .filter_map(|i| {
                archive
                    .by_index(i)
                    .ok()
                    .map(|file| file.name().to_string())
                    .filter(|name| name.starts_with("ppt/slides/slide") && name.ends_with(".xml"))
            })
            .collect();

        for slide_name in file_names {
            if let Ok(file) = archive.by_name(&slide_name) {
                if let Ok(content) = std::io::read_to_string(file) {
                    if let Ok(slide_text) = self.extract_text_from_xml(&content) {
                        all_text.push(slide_text);
                    }
                }
            }
        }

        let text = all_text.join("\n\n");
        let metadata = self.extract_metadata_from_zip(data);
        let title = metadata.get("title").cloned();

        Ok(ExtractedContent {
            format: DocumentFormat::Pptx,
            text,
            metadata,
            images: Vec::new(),
            tables: Vec::new(),
            links: Vec::new(),
            structure: DocumentStructure {
                title,
                headings: Vec::new(), // Would extract slide titles as headings
                page_count: all_text.len(), // Each slide is a "page"
                section_count: all_text.len(),
                table_of_contents: Vec::new(),
            },
            chunks: Vec::new(),
            language: None,
            processing_stats: ProcessingStats::default(),
            audio_content: Vec::new(),
            video_content: Vec::new(),
            cross_modal_embeddings: Vec::new(),
        })
    }

    fn can_handle(&self, data: &[u8]) -> bool {
        if data.len() < 4 {
            return false;
        }

        // Check for ZIP signature
        if &data[0..4] != &[0x50, 0x4B, 0x03, 0x04] && &data[0..4] != &[0x50, 0x4B, 0x05, 0x06] {
            return false;
        }

        // Check if it contains PPTX-specific files
        let cursor = std::io::Cursor::new(data);
        if let Ok(mut archive) = zip::ZipArchive::new(cursor) {
            archive.by_name("ppt/presentation.xml").is_ok()
                && archive.by_name("[Content_Types].xml").is_ok()
        } else {
            false
        }
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["pptx"]
    }
}

/// XLSX document handler
#[cfg(feature = "content-processing")]
struct XlsxHandler;

#[cfg(feature = "content-processing")]
impl OfficeDocumentHandler for XlsxHandler {}

#[cfg(feature = "content-processing")]
impl FormatHandler for XlsxHandler {
    fn extract_content(
        &self,
        data: &[u8],
        config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let cursor = std::io::Cursor::new(data);
        let mut archive = zip::ZipArchive::new(cursor)
            .map_err(|e| anyhow!("Failed to open XLSX archive: {}", e))?;

        // Extract shared strings first
        let shared_strings = self.extract_shared_strings(&mut archive)?;

        // Extract worksheet content
        let (text, tables) = self.extract_worksheets(&mut archive, &shared_strings, config)?;
        let metadata = self.extract_metadata_from_zip(data);
        let title = metadata.get("title").cloned();

        Ok(ExtractedContent {
            format: DocumentFormat::Xlsx,
            text,
            metadata,
            images: Vec::new(),
            tables,
            links: Vec::new(),
            structure: DocumentStructure {
                title,
                headings: Vec::new(),
                page_count: 1,
                section_count: 1,
                table_of_contents: Vec::new(),
            },
            chunks: Vec::new(),
            language: None,
            processing_stats: ProcessingStats::default(),
            audio_content: Vec::new(),
            video_content: Vec::new(),
            cross_modal_embeddings: Vec::new(),
        })
    }

    fn can_handle(&self, data: &[u8]) -> bool {
        if data.len() < 4 {
            return false;
        }

        // Check for ZIP signature
        if &data[0..4] != &[0x50, 0x4B, 0x03, 0x04] && &data[0..4] != &[0x50, 0x4B, 0x05, 0x06] {
            return false;
        }

        // Check if it contains XLSX-specific files
        let cursor = std::io::Cursor::new(data);
        if let Ok(mut archive) = zip::ZipArchive::new(cursor) {
            archive.by_name("xl/workbook.xml").is_ok()
                && archive.by_name("[Content_Types].xml").is_ok()
        } else {
            false
        }
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["xlsx"]
    }
}

#[cfg(feature = "content-processing")]
impl XlsxHandler {
    fn extract_shared_strings(
        &self,
        archive: &mut zip::ZipArchive<std::io::Cursor<&[u8]>>,
    ) -> Result<Vec<String>> {
        let mut shared_strings = Vec::new();

        if let Ok(file) = archive.by_name("xl/sharedStrings.xml") {
            let content = std::io::read_to_string(file)
                .map_err(|e| anyhow!("Failed to read shared strings: {}", e))?;

            let mut reader = quick_xml::Reader::from_str(&content);
            let mut buf = Vec::new();
            let mut in_text = false;
            let mut current_string = String::new();

            loop {
                match reader.read_event_into(&mut buf) {
                    Ok(quick_xml::events::Event::Start(ref e)) => {
                        if e.name().as_ref() == b"t" {
                            in_text = true;
                            current_string.clear();
                        }
                    }
                    Ok(quick_xml::events::Event::End(ref e)) => {
                        if e.name().as_ref() == b"t" {
                            in_text = false;
                        } else if e.name().as_ref() == b"si" {
                            shared_strings.push(current_string.clone());
                            current_string.clear();
                        }
                    }
                    Ok(quick_xml::events::Event::Text(e)) if in_text => {
                        let text = e.unescape().unwrap_or_default();
                        current_string.push_str(&text);
                    }
                    Ok(quick_xml::events::Event::Eof) => break,
                    Err(e) => return Err(anyhow!("XML parsing error: {}", e)),
                    _ => {}
                }
                buf.clear();
            }
        }

        Ok(shared_strings)
    }

    fn extract_worksheets(
        &self,
        archive: &mut zip::ZipArchive<std::io::Cursor<&[u8]>>,
        shared_strings: &[String],
        config: &ContentExtractionConfig,
    ) -> Result<(String, Vec<ExtractedTable>)> {
        let mut all_text = Vec::new();
        let mut tables = Vec::new();

        // Find all worksheet files
        let file_names: Vec<String> = (0..archive.len())
            .filter_map(|i| {
                archive
                    .by_index(i)
                    .ok()
                    .map(|file| file.name().to_string())
                    .filter(|name| {
                        name.starts_with("xl/worksheets/sheet") && name.ends_with(".xml")
                    })
            })
            .collect();

        for (sheet_index, sheet_name) in file_names.iter().enumerate() {
            if let Ok(file) = archive.by_name(sheet_name) {
                if let Ok(content) = std::io::read_to_string(file) {
                    let (sheet_text, sheet_table) =
                        self.extract_sheet_content(&content, shared_strings)?;
                    all_text.push(sheet_text);

                    if config.extract_tables && !sheet_table.rows.is_empty() {
                        let mut table = sheet_table;
                        table.caption = Some(format!("Sheet {}", sheet_index + 1));
                        tables.push(table);
                    }
                }
            }
        }

        Ok((all_text.join("\n\n"), tables))
    }

    fn extract_sheet_content(
        &self,
        xml: &str,
        shared_strings: &[String],
    ) -> Result<(String, ExtractedTable)> {
        let mut reader = quick_xml::Reader::from_str(xml);
        let mut buf = Vec::new();
        let mut cells = Vec::new();
        let mut current_cell = (0, 0, String::new()); // (row, col, value)
        let mut in_value = false;
        let mut cell_type_owned = String::from("str"); // Default to string
        let mut row_index = 0;
        let mut col_index = 0;

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(quick_xml::events::Event::Start(ref e)) => {
                    match e.name().as_ref() {
                        b"c" => {
                            // Cell
                            // Parse cell reference and type
                            for attr in e.attributes() {
                                if let Ok(attr) = attr {
                                    match attr.key.as_ref() {
                                        b"r" => {
                                            // Parse cell reference like "A1", "B2", etc.
                                            let cell_ref = String::from_utf8_lossy(&attr.value);
                                            (col_index, row_index) =
                                                self.parse_cell_reference(&cell_ref);
                                        }
                                        b"t" => {
                                            cell_type_owned =
                                                String::from_utf8_lossy(&attr.value).to_string();
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                        b"v" => {
                            // Cell value
                            in_value = true;
                            current_cell = (row_index, col_index, String::new());
                        }
                        _ => {}
                    }
                }
                Ok(quick_xml::events::Event::End(ref e)) => {
                    match e.name().as_ref() {
                        b"c" => {
                            if !current_cell.2.is_empty() {
                                cells.push(current_cell.clone());
                            }
                            // Reset for next cell
                            cell_type_owned = String::from("str");
                        }
                        b"v" => {
                            in_value = false;
                        }
                        _ => {}
                    }
                }
                Ok(quick_xml::events::Event::Text(e)) if in_value => {
                    let text = e.unescape().unwrap_or_default();
                    if cell_type_owned == "s" {
                        // Shared string reference
                        if let Ok(index) = text.parse::<usize>() {
                            if index < shared_strings.len() {
                                current_cell.2 = shared_strings[index].clone();
                            }
                        }
                    } else {
                        current_cell.2 = text.to_string();
                    }
                }
                Ok(quick_xml::events::Event::Eof) => break,
                Err(e) => return Err(anyhow!("XML parsing error: {}", e)),
                _ => {}
            }
            buf.clear();
        }

        // Convert cells to table format
        let (text, table) = self.cells_to_table(cells);
        Ok((text, table))
    }

    fn parse_cell_reference(&self, cell_ref: &str) -> (usize, usize) {
        let mut col = 0;
        let mut row = 0;
        let mut i = 0;

        // Parse column letters
        for ch in cell_ref.chars() {
            if ch.is_alphabetic() {
                col = col * 26 + (ch.to_ascii_uppercase() as u8 - b'A') as usize + 1;
                i += 1;
            } else {
                break;
            }
        }

        // Parse row number
        if let Ok(row_num) = cell_ref[i..].parse::<usize>() {
            row = row_num;
        }

        (col.saturating_sub(1), row.saturating_sub(1))
    }

    fn cells_to_table(&self, cells: Vec<(usize, usize, String)>) -> (String, ExtractedTable) {
        if cells.is_empty() {
            return (
                String::new(),
                ExtractedTable {
                    headers: Vec::new(),
                    rows: Vec::new(),
                    caption: None,
                    location: ContentLocation {
                        page: Some(1),
                        section: None,
                        char_offset: None,
                        line: None,
                        column: None,
                    },
                },
            );
        }

        // Find dimensions
        let max_row = cells.iter().map(|(r, _, _)| *r).max().unwrap_or(0);
        let max_col = cells.iter().map(|(_, c, _)| *c).max().unwrap_or(0);

        // Create grid
        let mut grid = vec![vec![String::new(); max_col + 1]; max_row + 1];
        for (row, col, value) in cells {
            if row <= max_row && col <= max_col {
                grid[row][col] = value;
            }
        }

        // Extract headers (first row) and data rows
        let headers = if !grid.is_empty() {
            grid[0].clone()
        } else {
            Vec::new()
        };

        let rows = if grid.len() > 1 {
            grid[1..].to_vec()
        } else {
            Vec::new()
        };

        // Create text representation
        let mut text_parts = Vec::new();
        for row in &grid {
            let row_text = row
                .iter()
                .filter(|cell| !cell.is_empty())
                .cloned()
                .collect::<Vec<_>>()
                .join(" | ");
            if !row_text.is_empty() {
                text_parts.push(row_text);
            }
        }
        let text = text_parts.join("\n");

        let table = ExtractedTable {
            headers,
            rows,
            caption: None,
            location: ContentLocation {
                page: Some(1),
                section: None,
                char_offset: None,
                line: None,
                column: None,
            },
        };

        (text, table)
    }
}

/// Image document handler
#[cfg(all(feature = "content-processing", feature = "images"))]
struct ImageHandler;

#[cfg(all(feature = "content-processing", feature = "images"))]
impl FormatHandler for ImageHandler {
    fn extract_content(
        &self,
        data: &[u8],
        config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let start_time = std::time::Instant::now();
        let mut warnings = Vec::new();

        // Load image using image crate
        let img = image::load_from_memory(data)
            .map_err(|e| anyhow!("Failed to load image: {}", e))?;

        let (width, height) = (img.width(), img.height());
        let format = self.detect_image_format(data);

        // Resize image if it exceeds max resolution
        let processed_img = if let Some((max_w, max_h)) = config.max_image_resolution {
            if width > max_w || height > max_h {
                let ratio = (max_w as f32 / width as f32).min(max_h as f32 / height as f32);
                let new_width = (width as f32 * ratio) as u32;
                let new_height = (height as f32 * ratio) as u32;
                
                warnings.push(format!("Image resized from {}x{} to {}x{}", width, height, new_width, new_height));
                img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3)
            } else {
                img
            }
        } else {
            img
        };

        // Create base64 encoded data
        let image_data = base64::encode(data);

        // Extract advanced visual features
        let visual_features = if config.extract_multimedia_features {
            Some(self.extract_visual_features(&processed_img)?)
        } else {
            None
        };

        // Perform object detection (placeholder implementation)
        let detected_objects = if config.extract_multimedia_features {
            self.detect_objects(&processed_img)
        } else {
            Vec::new()
        };

        // Perform image classification (placeholder implementation)
        let classification_labels = if config.extract_multimedia_features {
            self.classify_image(&processed_img)
        } else {
            Vec::new()
        };

        // Generate image embedding if requested
        let embedding = if config.generate_image_embeddings {
            Some(self.generate_image_embedding(&processed_img)?)
        } else {
            None
        };

        let mut extracted_image = ExtractedImage {
            data: image_data,
            format: format.clone(),
            width: processed_img.width(),
            height: processed_img.height(),
            alt_text: self.generate_alt_text(&classification_labels, &detected_objects),
            caption: self.generate_caption(&processed_img, &classification_labels),
            location: ContentLocation {
                page: Some(1),
                section: None,
                char_offset: None,
                line: None,
                column: None,
            },
            visual_features,
            embedding,
            detected_objects,
            classification_labels,
        };

        // Generate enhanced text description
        let text = self.generate_descriptive_text(&extracted_image);

        let processing_time = start_time.elapsed().as_millis() as u64;

        let mut content = ExtractedContent {
            format: DocumentFormat::Image,
            text,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("width".to_string(), width.to_string());
                meta.insert("height".to_string(), height.to_string());
                meta.insert("format".to_string(), format);
                meta.insert("size".to_string(), data.len().to_string());
                meta.insert("processed_width".to_string(), processed_img.width().to_string());
                meta.insert("processed_height".to_string(), processed_img.height().to_string());
                
                if let Some(ref features) = extracted_image.visual_features {
                    meta.insert("dominant_colors".to_string(), 
                        format!("{:?}", features.dominant_colors));
                    meta.insert("edge_density".to_string(), 
                        features.complexity_metrics.edge_density.to_string());
                    meta.insert("color_diversity".to_string(), 
                        features.complexity_metrics.color_diversity.to_string());
                }
                
                meta
            },
            images: vec![extracted_image],
            tables: Vec::new(),
            links: Vec::new(),
            structure: DocumentStructure {
                title: Some(format!("Image ({}x{})", width, height)),
                headings: Vec::new(),
                page_count: 1,
                section_count: 1,
                table_of_contents: Vec::new(),
            },
            chunks: Vec::new(),
            language: None,
            processing_stats: ProcessingStats {
                processing_time_ms: processing_time,
                total_chars: 0,
                total_words: 0,
                image_count: 1,
                table_count: 0,
                link_count: 0,
                chunk_count: 0,
                audio_count: 0,
                video_count: 0,
                cross_modal_embedding_count: 0,
                image_processing_time_ms: processing_time,
                audio_processing_time_ms: 0,
                video_processing_time_ms: 0,
                warnings,
            },
            audio_content: Vec::new(),
            video_content: Vec::new(),
            cross_modal_embeddings: Vec::new(),
        };

        Ok(content)
    }

    fn can_handle(&self, data: &[u8]) -> bool {
        image::load_from_memory(data).is_ok()
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "webp"]
    }
}

#[cfg(all(feature = "content-processing", feature = "images"))]
impl ImageHandler {
    fn detect_image_format(&self, data: &[u8]) -> String {
        if data.len() < 4 {
            return "unknown".to_string();
        }

        match &data[0..4] {
            [0xFF, 0xD8, 0xFF, _] => "JPEG".to_string(),
            [0x89, 0x50, 0x4E, 0x47] => "PNG".to_string(),
            [0x47, 0x49, 0x46, 0x38] => "GIF".to_string(),
            [0x42, 0x4D, _, _] => "BMP".to_string(),
            [0x49, 0x49, 0x2A, 0x00] | [0x4D, 0x4D, 0x00, 0x2A] => "TIFF".to_string(),
            [0x52, 0x49, 0x46, 0x46] => {
                if data.len() >= 12 && &data[8..12] == b"WEBP" {
                    "WebP".to_string()
                } else {
                    "unknown".to_string()
                }
            }
            _ => "unknown".to_string(),
        }
    }

    /// Extract visual features from image
    fn extract_visual_features(&self, img: &image::DynamicImage) -> Result<ImageFeatures> {
        let rgb_img = img.to_rgb8();
        
        // Extract color histogram
        let color_histogram = self.extract_color_histogram(&rgb_img);
        
        // Calculate dominant colors
        let dominant_colors = self.extract_dominant_colors(&rgb_img);
        
        // Calculate complexity metrics
        let complexity_metrics = self.calculate_complexity_metrics(&rgb_img);
        
        Ok(ImageFeatures {
            color_histogram: Some(color_histogram),
            texture_features: None, // Placeholder - would implement LBP, GLCM, etc.
            edge_features: None,    // Placeholder - would implement edge detection
            sift_features: None,    // Placeholder - would implement SIFT
            cnn_features: None,     // Placeholder - would use pre-trained CNN
            dominant_colors,
            complexity_metrics,
        })
    }

    /// Extract color histogram from RGB image
    fn extract_color_histogram(&self, img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>) -> Vec<f32> {
        let mut histogram = vec![0u32; 256]; // Simple grayscale histogram
        
        for pixel in img.pixels() {
            // Convert RGB to grayscale
            let gray = (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32) as u8;
            histogram[gray as usize] += 1;
        }
        
        // Normalize histogram
        let total_pixels = img.width() * img.height();
        histogram.into_iter().map(|count| count as f32 / total_pixels as f32).collect()
    }

    /// Extract dominant colors using simple clustering
    fn extract_dominant_colors(&self, img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>) -> Vec<(u8, u8, u8)> {
        let mut color_counts: HashMap<(u8, u8, u8), u32> = HashMap::new();
        
        // Sample every 10th pixel to avoid processing entire image
        for (x, y, pixel) in img.enumerate_pixels() {
            if x % 10 == 0 && y % 10 == 0 {
                // Quantize colors to reduce noise
                let quantized = (
                    (pixel[0] / 32) * 32,
                    (pixel[1] / 32) * 32,
                    (pixel[2] / 32) * 32,
                );
                *color_counts.entry(quantized).or_insert(0) += 1;
            }
        }
        
        // Get top 5 most frequent colors
        let mut colors: Vec<_> = color_counts.into_iter().collect();
        colors.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        colors.into_iter().take(5).map(|(color, _)| color).collect()
    }

    /// Calculate image complexity metrics
    fn calculate_complexity_metrics(&self, img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>) -> ImageComplexityMetrics {
        let (width, height) = img.dimensions();
        
        // Simple edge density calculation using Sobel-like operation
        let mut edge_count = 0u32;
        for y in 1..height-1 {
            for x in 1..width-1 {
                let center = img.get_pixel(x, y);
                let right = img.get_pixel(x+1, y);
                let bottom = img.get_pixel(x, y+1);
                
                // Simple gradient magnitude
                let grad_x = (right[0] as i32 - center[0] as i32).abs() +
                           (right[1] as i32 - center[1] as i32).abs() +
                           (right[2] as i32 - center[2] as i32).abs();
                let grad_y = (bottom[0] as i32 - center[0] as i32).abs() +
                           (bottom[1] as i32 - center[1] as i32).abs() +
                           (bottom[2] as i32 - center[2] as i32).abs();
                
                if grad_x + grad_y > 30 { // Threshold for edge detection
                    edge_count += 1;
                }
            }
        }
        
        let edge_density = edge_count as f32 / ((width-2) * (height-2)) as f32;
        
        // Calculate color diversity (unique colors / total pixels)
        let mut unique_colors = std::collections::HashSet::new();
        for pixel in img.pixels() {
            unique_colors.insert((pixel[0], pixel[1], pixel[2]));
        }
        let color_diversity = unique_colors.len() as f32 / (width * height) as f32;
        
        ImageComplexityMetrics {
            edge_density,
            color_diversity,
            texture_complexity: 0.5, // Placeholder
            entropy: 0.5, // Placeholder - would calculate Shannon entropy
        }
    }

    /// Detect objects in image (placeholder implementation)
    fn detect_objects(&self, _img: &image::DynamicImage) -> Vec<DetectedObject> {
        // This would use a real object detection model like YOLO, R-CNN, etc.
        // For now, return empty vector as placeholder
        Vec::new()
    }

    /// Classify image content (placeholder implementation)
    fn classify_image(&self, _img: &image::DynamicImage) -> Vec<ClassificationLabel> {
        // This would use a real image classification model like ResNet, EfficientNet, etc.
        // For now, return placeholder classifications based on image properties
        vec![
            ClassificationLabel {
                label: "photograph".to_string(),
                confidence: 0.8,
            },
            ClassificationLabel {
                label: "color_image".to_string(),
                confidence: 0.9,
            }
        ]
    }

    /// Generate image embedding (placeholder implementation)
    fn generate_image_embedding(&self, _img: &image::DynamicImage) -> Result<Vector> {
        // This would use a pre-trained model like ResNet, CLIP, etc. to generate embeddings
        // For now, generate a random 512-dimensional vector as placeholder
        use crate::utils;
        Ok(utils::random_vector(512, Some(42)))
    }

    /// Generate alt text for accessibility
    fn generate_alt_text(&self, labels: &[ClassificationLabel], objects: &[DetectedObject]) -> Option<String> {
        if labels.is_empty() && objects.is_empty() {
            return None;
        }
        
        let mut parts = Vec::new();
        
        // Add top classification labels
        for label in labels.iter().take(2) {
            if label.confidence > 0.5 {
                parts.push(label.label.clone());
            }
        }
        
        // Add detected objects
        for object in objects.iter().take(3) {
            if object.confidence > 0.7 {
                parts.push(object.label.clone());
            }
        }
        
        if parts.is_empty() {
            None
        } else {
            Some(format!("Image containing: {}", parts.join(", ")))
        }
    }

    /// Generate image caption
    fn generate_caption(&self, img: &image::DynamicImage, labels: &[ClassificationLabel]) -> Option<String> {
        let (width, height) = (img.width(), img.height());
        
        let size_desc = match (width, height) {
            (w, h) if w > 1920 || h > 1080 => "high resolution",
            (w, h) if w < 640 || h < 480 => "low resolution",
            _ => "standard resolution",
        };
        
        let content_desc = if let Some(label) = labels.first() {
            &label.label
        } else {
            "image"
        };
        
        Some(format!("{} {} ({}x{})", size_desc, content_desc, width, height))
    }

    /// Generate descriptive text for the image
    fn generate_descriptive_text(&self, extracted_image: &ExtractedImage) -> String {
        let mut parts = Vec::new();
        
        // Basic image info
        parts.push(format!("Image: {} format, {}x{} pixels", 
            extracted_image.format, extracted_image.width, extracted_image.height));
        
        // Add alt text if available
        if let Some(ref alt_text) = extracted_image.alt_text {
            parts.push(alt_text.clone());
        }
        
        // Add caption if available
        if let Some(ref caption) = extracted_image.caption {
            parts.push(caption.clone());
        }
        
        // Add classification labels
        if !extracted_image.classification_labels.is_empty() {
            let labels: Vec<String> = extracted_image.classification_labels
                .iter()
                .filter(|l| l.confidence > 0.5)
                .map(|l| format!("{} ({:.1}%)", l.label, l.confidence * 100.0))
                .collect();
            if !labels.is_empty() {
                parts.push(format!("Classifications: {}", labels.join(", ")));
            }
        }
        
        // Add detected objects
        if !extracted_image.detected_objects.is_empty() {
            let objects: Vec<String> = extracted_image.detected_objects
                .iter()
                .filter(|o| o.confidence > 0.7)
                .map(|o| format!("{} ({:.1}%)", o.label, o.confidence * 100.0))
                .collect();
            if !objects.is_empty() {
                parts.push(format!("Detected objects: {}", objects.join(", ")));
            }
        }
        
        // Add complexity info
        if let Some(ref features) = extracted_image.visual_features {
            parts.push(format!("Visual complexity: edge density {:.2}, color diversity {:.2}",
                features.complexity_metrics.edge_density,
                features.complexity_metrics.color_diversity));
            
            if !features.dominant_colors.is_empty() {
                let color_desc: Vec<String> = features.dominant_colors
                    .iter()
                    .map(|(r, g, b)| format!("rgb({},{},{})", r, g, b))
                    .collect();
                parts.push(format!("Dominant colors: {}", color_desc.join(", ")));
            }
        }
        
        parts.join("\n")
    }
}

/// Audio document handler
#[cfg(feature = "content-processing")]
struct AudioHandler;

#[cfg(feature = "content-processing")]
impl FormatHandler for AudioHandler {
    fn extract_content(
        &self,
        data: &[u8],
        _config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let format = self.detect_audio_format(data);
        let audio_data = base64::encode(data);

        // Basic audio metadata extraction (simplified)
        let (duration, sample_rate, channels) = self.extract_basic_audio_info(data);

        let extracted_audio = ExtractedAudio {
            data: audio_data,
            format: format.clone(),
            duration,
            sample_rate,
            channels,
            audio_features: None,
            embedding: None,
            transcription: None,
            music_analysis: None,
            speech_analysis: None,
        };

        let text = format!(
            "Audio: {} format, {:.1}s duration, {}Hz sample rate, {} channels",
            format, duration, sample_rate, channels
        );

        let content = ExtractedContent {
            format: DocumentFormat::Audio,
            text,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("format".to_string(), format);
                meta.insert("duration".to_string(), duration.to_string());
                meta.insert("sample_rate".to_string(), sample_rate.to_string());
                meta.insert("channels".to_string(), channels.to_string());
                meta.insert("size".to_string(), data.len().to_string());
                meta
            },
            images: Vec::new(),
            tables: Vec::new(),
            links: Vec::new(),
            structure: DocumentStructure {
                title: Some(format!("Audio ({:.1}s)", duration)),
                headings: Vec::new(),
                page_count: 1,
                section_count: 1,
                table_of_contents: Vec::new(),
            },
            chunks: Vec::new(),
            language: None,
            processing_stats: ProcessingStats::default(),
            audio_content: vec![extracted_audio],
            video_content: Vec::new(),
            cross_modal_embeddings: Vec::new(),
        };

        Ok(content)
    }

    fn can_handle(&self, data: &[u8]) -> bool {
        self.detect_audio_format(data) != "unknown"
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["mp3", "wav", "flac", "ogg", "aac", "m4a", "opus", "wma"]
    }
}

#[cfg(feature = "content-processing")]
impl AudioHandler {
    fn detect_audio_format(&self, data: &[u8]) -> String {
        if data.len() < 4 {
            return "unknown".to_string();
        }

        match &data[0..4] {
            [0x49, 0x44, 0x33, _] => "MP3".to_string(), // ID3 tag
            [0xFF, 0xFB, _, _] | [0xFF, 0xF3, _, _] | [0xFF, 0xF2, _, _] => "MP3".to_string(),
            [0x66, 0x4C, 0x61, 0x43] => "FLAC".to_string(),
            [0x4F, 0x67, 0x67, 0x53] => "OGG".to_string(),
            [0x52, 0x49, 0x46, 0x46] => {
                if data.len() >= 12 && &data[8..12] == b"WAVE" {
                    "WAV".to_string()
                } else {
                    "unknown".to_string()
                }
            }
            _ => "unknown".to_string(),
        }
    }

    fn extract_basic_audio_info(&self, data: &[u8]) -> (f32, u32, u16) {
        // Simplified audio info extraction - in practice would use proper audio libraries
        // For now, return reasonable defaults
        let duration = (data.len() as f32) / 44100.0 / 2.0 / 2.0; // Rough estimate
        let sample_rate = 44100; // Common default
        let channels = 2; // Stereo default

        (duration, sample_rate, channels)
    }
}

/// Video document handler
#[cfg(feature = "content-processing")]
struct VideoHandler;

#[cfg(feature = "content-processing")]
impl FormatHandler for VideoHandler {
    fn extract_content(
        &self,
        data: &[u8],
        _config: &ContentExtractionConfig,
    ) -> Result<ExtractedContent> {
        let format = self.detect_video_format(data);
        let video_data = base64::encode(data);

        // Basic video metadata extraction (simplified)
        let (duration, frame_rate, resolution) = self.extract_basic_video_info(data);

        let extracted_video = ExtractedVideo {
            data: video_data,
            format: format.clone(),
            duration,
            frame_rate,
            resolution,
            keyframes: vec![],
            embedding: None,
            audio_analysis: None,
            video_analysis: None,
        };

        let text = format!(
            "Video: {} format, {:.1}s duration, {:.1}fps, {}x{} resolution",
            format, duration, frame_rate, resolution.0, resolution.1
        );

        let content = ExtractedContent {
            format: DocumentFormat::Video,
            text,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("format".to_string(), format);
                meta.insert("duration".to_string(), duration.to_string());
                meta.insert("frame_rate".to_string(), frame_rate.to_string());
                meta.insert("width".to_string(), resolution.0.to_string());
                meta.insert("height".to_string(), resolution.1.to_string());
                meta.insert("size".to_string(), data.len().to_string());
                meta
            },
            images: Vec::new(),
            tables: Vec::new(),
            links: Vec::new(),
            structure: DocumentStructure {
                title: Some(format!("Video ({:.1}s)", duration)),
                headings: Vec::new(),
                page_count: 1,
                section_count: 1,
                table_of_contents: Vec::new(),
            },
            chunks: Vec::new(),
            language: None,
            processing_stats: ProcessingStats::default(),
            audio_content: Vec::new(),
            video_content: vec![extracted_video],
            cross_modal_embeddings: Vec::new(),
        };

        Ok(content)
    }

    fn can_handle(&self, data: &[u8]) -> bool {
        self.detect_video_format(data) != "unknown"
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["mp4", "avi", "mkv", "mov", "wmv", "flv", "webm", "m4v", "3gp"]
    }
}

#[cfg(feature = "content-processing")]
impl VideoHandler {
    fn detect_video_format(&self, data: &[u8]) -> String {
        if data.len() < 8 {
            return "unknown".to_string();
        }

        match &data[0..4] {
            [0x00, 0x00, 0x00, 0x18] | [0x00, 0x00, 0x00, 0x20] => {
                if &data[4..8] == b"ftyp" {
                    "MP4".to_string()
                } else {
                    "unknown".to_string()
                }
            }
            [0x1A, 0x45, 0xDF, 0xA3] => "Matroska/WebM".to_string(),
            [0x46, 0x4C, 0x56, 0x01] => "FLV".to_string(),
            [0x52, 0x49, 0x46, 0x46] => {
                if data.len() >= 12 && &data[8..12] == b"AVI " {
                    "AVI".to_string()
                } else {
                    "unknown".to_string()
                }
            }
            _ => "unknown".to_string(),
        }
    }

    fn extract_basic_video_info(&self, data: &[u8]) -> (f32, f32, (u32, u32)) {
        // Simplified video info extraction - in practice would use FFmpeg or similar
        // For now, return reasonable defaults
        let duration = (data.len() as f32) / (1024.0 * 1024.0) * 10.0; // Rough estimate
        let frame_rate = 30.0; // Common default
        let resolution = (1920, 1080); // HD default

        (duration, frame_rate, resolution)
    }
}

#[cfg(all(test, feature = "content-processing"))]
mod tests {
    use super::*;

    #[test]
    fn test_plain_text_processing() {
        let processor = ContentProcessor::new();
        let content = b"This is a simple text document.\n\nIt has multiple paragraphs.";

        let result = processor
            .process_document(content, DocumentFormat::PlainText)
            .unwrap();

        assert_eq!(result.format, DocumentFormat::PlainText);
        assert!(result.text.contains("simple text document"));
        assert!(result.processing_stats.total_chars > 0);
    }

    #[test]
    fn test_html_processing() {
        let processor = ContentProcessor::new();
        let html = b"<html><head><title>Test Page</title></head><body><h1>Main Heading</h1><p>Test content</p></body></html>";

        let result = processor
            .process_document(html, DocumentFormat::Html)
            .unwrap();

        assert_eq!(result.format, DocumentFormat::Html);
        assert!(result.text.contains("Test content"));
        assert_eq!(result.metadata.get("title"), Some(&"Test Page".to_string()));
        assert!(!result.structure.headings.is_empty());
    }

    #[test]
    fn test_markdown_processing() {
        let processor = ContentProcessor::new();
        let markdown = b"# Main Title\n\nThis is a **bold** text with *italic* formatting.\n\n## Subtitle\n\nMore content here.";

        let result = processor
            .process_document(markdown, DocumentFormat::Markdown)
            .unwrap();

        assert_eq!(result.format, DocumentFormat::Markdown);
        assert!(result.text.contains("bold"));
        assert!(!result.structure.headings.is_empty());
        assert_eq!(result.structure.headings[0].level, 1);
    }

    #[test]
    fn test_format_detection() {
        let processor = ContentProcessor::new();

        // Test PDF magic bytes
        let pdf_data = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog";
        assert_eq!(
            processor.detect_format_by_content(pdf_data).unwrap(),
            DocumentFormat::Pdf
        );

        // Test HTML detection
        let html_data = b"<!DOCTYPE html><html><head>";
        assert_eq!(
            processor.detect_format_by_content(html_data).unwrap(),
            DocumentFormat::Html
        );

        // Test JSON detection
        let json_data = b"{\"key\": \"value\", \"array\": [1, 2, 3]}";
        assert_eq!(
            processor.detect_format_by_content(json_data).unwrap(),
            DocumentFormat::Json
        );
    }

    #[test]
    fn test_chunking_strategies() {
        let processor = ContentProcessor::with_config(ContentExtractionConfig {
            chunking_strategy: ChunkingStrategy::FixedTokens(5),
            ..ContentExtractionConfig::default()
        });

        let text = "This is a test document with multiple words that should be chunked properly.";
        let structure = DocumentStructure {
            title: None,
            headings: Vec::new(),
            page_count: 1,
            section_count: 1,
            table_of_contents: Vec::new(),
        };

        let chunks = processor.create_chunks(text, &structure).unwrap();
        assert!(!chunks.is_empty());

        // Each chunk should have approximately 5 words
        for chunk in chunks {
            let word_count = chunk.text.split_whitespace().count();
            assert!(word_count <= 6); // Allow for some variation
        }
    }

    #[test]
    fn test_language_detection() {
        let processor = ContentProcessor::new();

        // Test English detection
        let english_text = "The quick brown fox jumps over the lazy dog and runs to the forest";
        let lang = processor.detect_language(english_text);
        assert_eq!(lang, Some("en".to_string()));

        // Test Spanish detection
        let spanish_text = "El gato come pescado y bebe agua en la casa durante el día";
        let lang = processor.detect_language(spanish_text);
        assert_eq!(lang, Some("es".to_string()));
    }

    #[test]
    fn test_csv_processing() {
        let processor = ContentProcessor::new();
        let csv_data = b"Name,Age,City\nJohn,25,New York\nJane,30,Los Angeles";

        let result = processor
            .process_document(csv_data, DocumentFormat::Csv)
            .unwrap();

        assert_eq!(result.format, DocumentFormat::Csv);
        assert_eq!(result.tables.len(), 1);
        assert_eq!(result.tables[0].headers, vec!["Name", "Age", "City"]);
        assert_eq!(result.tables[0].rows.len(), 2);
    }
}
