//! Audio content validators

use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Duration;

use super::traits::*;
use super::types::*;
use crate::Result;

/// Audio format validator
#[derive(Debug)]
pub struct AudioFormatValidator {
    supported_formats: Vec<String>,
    max_file_size: usize,
}

impl Default for AudioFormatValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFormatValidator {
    pub fn new() -> Self {
        Self {
            supported_formats: vec![
                "mp3".to_string(),
                "wav".to_string(),
                "ogg".to_string(),
                "flac".to_string(),
                "aac".to_string(),
                "m4a".to_string(),
            ],
            max_file_size: 100 * 1024 * 1024, // 100MB
        }
    }

    pub fn with_formats(formats: Vec<String>) -> Self {
        Self {
            supported_formats: formats,
            max_file_size: 100 * 1024 * 1024,
        }
    }

    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.max_file_size = max_size;
        self
    }
}

#[async_trait]
impl AudioValidator for AudioFormatValidator {
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>> {
        if !self.supports_content(content) {
            return Ok(None);
        }

        let format = self.detect_audio_format(&content.data);
        let size_valid = content.data.len() <= self.max_file_size;
        let format_valid = format.is_some()
            && self
                .supported_formats
                .contains(&format.as_ref().unwrap().to_lowercase());

        let is_valid = format_valid && size_valid;

        let mut details = HashMap::new();
        details.insert(
            "detected_format".to_string(),
            format.clone().unwrap_or("unknown".to_string()),
        );
        details.insert(
            "supported_formats".to_string(),
            self.supported_formats.join(", "),
        );
        details.insert("file_size".to_string(), content.data.len().to_string());
        details.insert("max_file_size".to_string(), self.max_file_size.to_string());

        let confidence = if is_valid { 0.95 } else { 0.1 };

        let mut issues = Vec::new();
        if !format_valid {
            issues.push(format!(
                "Unsupported audio format: {}",
                format.unwrap_or("unknown".to_string())
            ));
        }
        if !size_valid {
            issues.push(format!(
                "File size exceeds limit: {} > {}",
                content.data.len(),
                self.max_file_size
            ));
        }

        let error_message = if issues.is_empty() {
            None
        } else {
            Some(issues.join("; "))
        };

        Ok(Some(ValidationResult {
            is_valid,
            confidence,
            error_message,
            details,
        }))
    }

    fn name(&self) -> &str {
        "audio_format"
    }

    fn description(&self) -> &str {
        "Validates audio file format and basic structure"
    }
}

impl AudioFormatValidator {
    fn detect_audio_format(&self, data: &[u8]) -> Option<String> {
        if data.len() < 12 {
            return None;
        }

        // Check file signatures (magic numbers)
        match &data[0..4] {
            [0x49, 0x44, 0x33, ..] => Some("mp3".to_string()), // ID3 tag
            [0xFF, 0xFB, ..] | [0xFF, 0xFA, ..] => Some("mp3".to_string()), // MP3 frame header
            [0x52, 0x49, 0x46, 0x46] if data.len() >= 12 => {
                // RIFF header, check for WAVE
                if &data[8..12] == b"WAVE" {
                    Some("wav".to_string())
                } else {
                    None
                }
            }
            [0x4F, 0x67, 0x67, 0x53] => Some("ogg".to_string()), // OggS
            [0x66, 0x4C, 0x61, 0x43] => Some("flac".to_string()), // fLaC
            _ => {
                // Check for AAC/M4A (more complex detection needed)
                if data.len() >= 8 && &data[4..8] == b"ftyp" {
                    Some("m4a".to_string())
                } else {
                    None
                }
            }
        }
    }
}

/// Speech recognition validator
#[derive(Debug)]
pub struct SpeechRecognitionValidator {
    min_confidence: f64,
    target_language: Option<String>,
}

impl Default for SpeechRecognitionValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeechRecognitionValidator {
    pub fn new() -> Self {
        Self {
            min_confidence: 0.5,
            target_language: None,
        }
    }

    pub fn with_confidence_threshold(mut self, threshold: f64) -> Self {
        self.min_confidence = threshold;
        self
    }

    pub fn with_target_language(mut self, language: String) -> Self {
        self.target_language = Some(language);
        self
    }
}

#[async_trait]
impl AudioValidator for SpeechRecognitionValidator {
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>> {
        if !self.supports_content(content) {
            return Ok(None);
        }

        let transcription = self.transcribe_audio(&content.data);
        let is_valid = transcription.confidence >= self.min_confidence;

        let mut details = HashMap::new();
        details.insert("transcription".to_string(), transcription.text.clone());
        details.insert(
            "confidence".to_string(),
            transcription.confidence.to_string(),
        );
        details.insert("language".to_string(), transcription.language.clone());
        details.insert(
            "duration".to_string(),
            format!("{:.2}s", transcription.duration.as_secs_f64()),
        );

        if let Some(target_lang) = &self.target_language {
            details.insert("target_language".to_string(), target_lang.clone());
            let language_match = transcription.language == *target_lang;
            details.insert("language_match".to_string(), language_match.to_string());
        }

        let confidence = transcription.confidence;
        let error_message = if is_valid {
            None
        } else {
            Some(format!(
                "Speech recognition confidence too low: {:.2} < {:.2}",
                transcription.confidence, self.min_confidence
            ))
        };

        Ok(Some(ValidationResult {
            is_valid,
            confidence,
            error_message,
            details,
        }))
    }

    fn name(&self) -> &str {
        "speech_recognition"
    }

    fn description(&self) -> &str {
        "Performs speech recognition on audio content"
    }
}

impl SpeechRecognitionValidator {
    fn transcribe_audio(&self, _data: &[u8]) -> SpeechRecognitionResult {
        // Simplified speech recognition (would use proper ASR in real implementation)
        SpeechRecognitionResult {
            text: "Hello, this is a test transcription".to_string(),
            confidence: 0.85,
            language: "en".to_string(),
            duration: Duration::from_secs(5),
            word_timestamps: vec![
                (
                    "Hello",
                    Duration::from_millis(0),
                    Duration::from_millis(500),
                ),
                (
                    "this",
                    Duration::from_millis(500),
                    Duration::from_millis(750),
                ),
                ("is", Duration::from_millis(750), Duration::from_millis(900)),
                ("a", Duration::from_millis(900), Duration::from_millis(1000)),
                (
                    "test",
                    Duration::from_millis(1000),
                    Duration::from_millis(1300),
                ),
                (
                    "transcription",
                    Duration::from_millis(1300),
                    Duration::from_millis(2000),
                ),
            ]
            .into_iter()
            .map(|(w, s, e)| (w.to_string(), s, e))
            .collect(),
        }
    }
}

/// Music analysis validator
#[derive(Debug)]
pub struct MusicAnalysisValidator {
    analyze_tempo: bool,
    analyze_key: bool,
    analyze_genre: bool,
}

impl Default for MusicAnalysisValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl MusicAnalysisValidator {
    pub fn new() -> Self {
        Self {
            analyze_tempo: true,
            analyze_key: true,
            analyze_genre: true,
        }
    }

    pub fn with_analysis_options(
        analyze_tempo: bool,
        analyze_key: bool,
        analyze_genre: bool,
    ) -> Self {
        Self {
            analyze_tempo,
            analyze_key,
            analyze_genre,
        }
    }
}

#[async_trait]
impl AudioValidator for MusicAnalysisValidator {
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>> {
        if !self.supports_content(content) {
            return Ok(None);
        }

        let analysis = self.analyze_music(&content.data);
        let is_valid = true; // Music analysis is informational

        let mut details = HashMap::new();

        if self.analyze_tempo {
            if let Some(tempo) = analysis.tempo {
                details.insert("tempo_bpm".to_string(), tempo.to_string());
            }
        }

        if self.analyze_key {
            if let Some(key) = &analysis.key {
                details.insert("key_signature".to_string(), key.clone());
            }
        }

        if self.analyze_genre {
            if let Some(genre) = &analysis.genre {
                details.insert("genre".to_string(), genre.clone());
            }
        }

        if let Some(mood) = &analysis.mood {
            details.insert("mood".to_string(), mood.clone());
        }

        if let Some(time_sig) = &analysis.time_signature {
            details.insert("time_signature".to_string(), time_sig.clone());
        }

        Ok(Some(ValidationResult {
            is_valid,
            confidence: 0.7,
            error_message: None,
            details,
        }))
    }

    fn name(&self) -> &str {
        "music_analysis"
    }

    fn description(&self) -> &str {
        "Analyzes musical properties of audio content"
    }
}

impl MusicAnalysisValidator {
    fn analyze_music(&self, _data: &[u8]) -> MusicFeatures {
        // Simplified music analysis (would use proper audio analysis in real implementation)
        MusicFeatures {
            tempo: Some(120.0),
            key: Some("C major".to_string()),
            time_signature: Some("4/4".to_string()),
            mood: Some("upbeat".to_string()),
            genre: Some("pop".to_string()),
        }
    }
}

/// Audio quality validator
#[derive(Debug)]
pub struct AudioQualityValidator {
    min_sample_rate: u32,
    max_sample_rate: u32,
    min_bitrate: u32,
    min_duration: Duration,
    max_duration: Duration,
}

impl Default for AudioQualityValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioQualityValidator {
    pub fn new() -> Self {
        Self {
            min_sample_rate: 8000,   // 8 kHz
            max_sample_rate: 192000, // 192 kHz
            min_bitrate: 32000,      // 32 kbps
            min_duration: Duration::from_millis(100),
            max_duration: Duration::from_secs(3600), // 1 hour
        }
    }

    pub fn with_sample_rate_limits(mut self, min_rate: u32, max_rate: u32) -> Self {
        self.min_sample_rate = min_rate;
        self.max_sample_rate = max_rate;
        self
    }

    pub fn with_duration_limits(mut self, min_duration: Duration, max_duration: Duration) -> Self {
        self.min_duration = min_duration;
        self.max_duration = max_duration;
        self
    }
}

#[async_trait]
impl AudioValidator for AudioQualityValidator {
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>> {
        if !self.supports_content(content) {
            return Ok(None);
        }

        let quality_info = self.analyze_quality(&content.data);

        let mut is_valid = true;
        let mut issues = Vec::new();
        let mut details = HashMap::new();

        // Check sample rate
        if let Some(sample_rate) = quality_info.sample_rate {
            details.insert("sample_rate".to_string(), sample_rate.to_string());
            if sample_rate < self.min_sample_rate {
                is_valid = false;
                issues.push(format!(
                    "Sample rate too low: {} < {}",
                    sample_rate, self.min_sample_rate
                ));
            }
            if sample_rate > self.max_sample_rate {
                is_valid = false;
                issues.push(format!(
                    "Sample rate too high: {} > {}",
                    sample_rate, self.max_sample_rate
                ));
            }
        }

        // Check bitrate
        if let Some(bitrate) = quality_info.bitrate {
            details.insert("bitrate".to_string(), bitrate.to_string());
            if bitrate < self.min_bitrate {
                is_valid = false;
                issues.push(format!(
                    "Bitrate too low: {} < {}",
                    bitrate, self.min_bitrate
                ));
            }
        }

        // Check duration
        if let Some(duration) = quality_info.duration {
            details.insert(
                "duration".to_string(),
                format!("{:.2}s", duration.as_secs_f64()),
            );
            if duration < self.min_duration {
                is_valid = false;
                issues.push(format!(
                    "Duration too short: {:.2}s < {:.2}s",
                    duration.as_secs_f64(),
                    self.min_duration.as_secs_f64()
                ));
            }
            if duration > self.max_duration {
                is_valid = false;
                issues.push(format!(
                    "Duration too long: {:.2}s > {:.2}s",
                    duration.as_secs_f64(),
                    self.max_duration.as_secs_f64()
                ));
            }
        }

        // Check channels
        if let Some(channels) = quality_info.channels {
            details.insert("channels".to_string(), channels.to_string());
        }

        let confidence = if is_valid { 0.9 } else { 0.3 };
        let error_message = if issues.is_empty() {
            None
        } else {
            Some(issues.join("; "))
        };

        Ok(Some(ValidationResult {
            is_valid,
            confidence,
            error_message,
            details,
        }))
    }

    fn name(&self) -> &str {
        "audio_quality"
    }

    fn description(&self) -> &str {
        "Validates audio quality parameters"
    }
}

impl AudioQualityValidator {
    fn analyze_quality(&self, _data: &[u8]) -> AudioQualityInfo {
        // Simplified audio quality analysis (would parse actual audio headers in real implementation)
        AudioQualityInfo {
            sample_rate: Some(44100),
            bitrate: Some(128000),
            channels: Some(2),
            duration: Some(Duration::from_secs(30)),
            format: Some("mp3".to_string()),
        }
    }
}

/// Speech recognition result
#[derive(Debug, Clone)]
struct SpeechRecognitionResult {
    pub text: String,
    pub confidence: f64,
    pub language: String,
    pub duration: Duration,
    pub word_timestamps: Vec<(String, Duration, Duration)>,
}

/// Audio quality information
#[derive(Debug, Clone)]
struct AudioQualityInfo {
    pub sample_rate: Option<u32>,
    pub bitrate: Option<u32>,
    pub channels: Option<u32>,
    pub duration: Option<Duration>,
    pub format: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    fn create_test_audio_content(data: &[u8]) -> MultiModalContent {
        MultiModalContent {
            id: "test_audio".to_string(),
            content_type: ContentType::Audio,
            data: data.to_vec(),
            metadata: ContentMetadata::default(),
            source_url: None,
            timestamp: SystemTime::now(),
        }
    }

    #[tokio::test]
    async fn test_audio_format_validator_mp3() {
        let validator = AudioFormatValidator::new();
        // MP3 ID3 tag signature with proper length (12+ bytes)
        let mp3_data = vec![
            0x49, 0x44, 0x33, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let content = create_test_audio_content(&mp3_data);

        let result = validator.validate(&content).await.unwrap().unwrap();
        assert!(result.is_valid);
        assert_eq!(result.details.get("detected_format").unwrap(), "mp3");
    }

    #[tokio::test]
    async fn test_audio_format_validator_wav() {
        let validator = AudioFormatValidator::new();
        // WAV RIFF header
        let mut wav_data = vec![0x52, 0x49, 0x46, 0x46]; // RIFF
        wav_data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // file size
        wav_data.extend_from_slice(&[0x57, 0x41, 0x56, 0x45]); // WAVE
        let content = create_test_audio_content(&wav_data);

        let result = validator.validate(&content).await.unwrap().unwrap();
        assert!(result.is_valid);
        assert_eq!(result.details.get("detected_format").unwrap(), "wav");
    }

    #[tokio::test]
    async fn test_speech_recognition_validator() {
        let validator = SpeechRecognitionValidator::new();
        let content = create_test_audio_content(&[0x49, 0x44, 0x33]); // Minimal MP3

        let result = validator.validate(&content).await.unwrap().unwrap();
        assert!(result.is_valid);
        assert!(result.details.contains_key("transcription"));
        assert!(result.details.contains_key("confidence"));
    }

    #[tokio::test]
    async fn test_music_analysis_validator() {
        let validator = MusicAnalysisValidator::new();
        let content = create_test_audio_content(&[0x49, 0x44, 0x33]); // Minimal MP3

        let result = validator.validate(&content).await.unwrap().unwrap();
        assert!(result.is_valid);
        assert!(result.details.contains_key("tempo_bpm"));
        assert!(result.details.contains_key("key_signature"));
    }

    #[tokio::test]
    async fn test_audio_quality_validator() {
        let validator = AudioQualityValidator::new();
        let content = create_test_audio_content(&[0x49, 0x44, 0x33]); // Minimal MP3

        let result = validator.validate(&content).await.unwrap().unwrap();
        assert!(result.is_valid);
        assert!(result.details.contains_key("sample_rate"));
        assert!(result.details.contains_key("bitrate"));
    }
}
