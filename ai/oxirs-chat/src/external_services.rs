//! External Services Integration for OxiRS Chat
//!
//! This module provides integration with external services including:
//! - Knowledge base APIs
//! - Search engines  
//! - Fact-checking services
//! - Translation services
//! - Speech recognition
//! - Text-to-speech

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tracing::warn;

/// Configuration for external services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalServicesConfig {
    pub knowledge_base_apis: Vec<KnowledgeBaseConfig>,
    pub search_engines: Vec<SearchEngineConfig>,
    pub fact_checkers: Vec<FactCheckerConfig>,
    pub translation_services: Vec<TranslationConfig>,
    pub speech_services: Vec<SpeechConfig>,
    pub timeout_duration: Duration,
    pub retry_attempts: usize,
    pub rate_limit_per_minute: u32,
}

impl Default for ExternalServicesConfig {
    fn default() -> Self {
        Self {
            knowledge_base_apis: Vec::new(),
            search_engines: Vec::new(),
            fact_checkers: Vec::new(),
            translation_services: Vec::new(),
            speech_services: Vec::new(),
            timeout_duration: Duration::from_secs(30),
            retry_attempts: 3,
            rate_limit_per_minute: 100,
        }
    }
}

/// Knowledge base API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBaseConfig {
    pub name: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub headers: HashMap<String, String>,
    pub enabled: bool,
    pub priority: u32,
}

/// Search engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchEngineConfig {
    pub name: String,
    pub api_url: String,
    pub api_key: Option<String>,
    pub search_type: SearchType,
    pub max_results: usize,
    pub enabled: bool,
}

/// Search engine types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchType {
    Web,
    Academic,
    News,
    Images,
    Videos,
    Knowledge,
}

/// Fact checker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactCheckerConfig {
    pub name: String,
    pub api_url: String,
    pub api_key: Option<String>,
    pub confidence_threshold: f32,
    pub enabled: bool,
}

/// Translation service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationConfig {
    pub name: String,
    pub api_url: String,
    pub api_key: Option<String>,
    pub supported_languages: Vec<String>,
    pub enabled: bool,
}

/// Speech service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechConfig {
    pub name: String,
    pub speech_to_text_url: Option<String>,
    pub text_to_speech_url: Option<String>,
    pub api_key: Option<String>,
    pub supported_languages: Vec<String>,
    pub enabled: bool,
    pub voice_models: Vec<VoiceModel>,
    pub real_time_streaming: bool,
    pub noise_cancellation: bool,
    pub speech_enhancement: bool,
    pub speaker_diarization: bool,
    pub emotion_detection: bool,
    pub custom_vocabulary: Vec<String>,
}

/// Voice model configuration for TTS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceModel {
    pub id: String,
    pub name: String,
    pub language: String,
    pub gender: VoiceGender,
    pub age_range: VoiceAgeRange,
    pub voice_type: VoiceType,
    pub sample_rate: u32,
    pub neural_voice: bool,
}

/// Voice gender options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoiceGender {
    Male,
    Female,
    Neutral,
}

/// Voice age range options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoiceAgeRange {
    Child,
    Young,
    Adult,
    Senior,
}

/// Voice type characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoiceType {
    Conversational,
    News,
    Narrative,
    Assistant,
    Customer,
    Educational,
    Emotional,
}

/// Advanced speech processing options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechProcessingOptions {
    pub enable_punctuation: bool,
    pub enable_profanity_filter: bool,
    pub enable_automatic_formatting: bool,
    pub enable_word_timestamps: bool,
    pub enable_confidence_scores: bool,
    pub enable_speaker_identification: bool,
    pub language_detection_mode: LanguageDetectionMode,
    pub audio_quality_enhancement: bool,
}

/// Language detection modes for speech
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LanguageDetectionMode {
    None,
    Automatic,
    Constrained(Vec<String>),
}

/// External services integration manager
pub struct ExternalServicesManager {
    config: ExternalServicesConfig,
    client: reqwest::Client,
    rate_limiter: RateLimiter,
}

impl ExternalServicesManager {
    /// Create a new external services manager
    pub fn new(config: ExternalServicesConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout_duration)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            rate_limiter: RateLimiter::new(config.rate_limit_per_minute),
            config,
            client,
        }
    }

    /// Search knowledge base APIs
    pub async fn search_knowledge_bases(&self, query: &str) -> Result<Vec<KnowledgeResult>> {
        let mut results = Vec::new();

        for kb_config in &self.config.knowledge_base_apis {
            if !kb_config.enabled {
                continue;
            }

            self.rate_limiter.check_limit().await?;

            match self.query_knowledge_base(kb_config, query).await {
                Ok(mut kb_results) => {
                    results.append(&mut kb_results);
                }
                Err(e) => {
                    warn!("Knowledge base {} failed: {}", kb_config.name, e);
                }
            }
        }

        // Sort by priority and relevance
        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Search external search engines
    pub async fn search_engines(
        &self,
        query: &str,
        search_type: SearchType,
    ) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();

        for engine_config in &self.config.search_engines {
            if !engine_config.enabled || engine_config.search_type != search_type {
                continue;
            }

            self.rate_limiter.check_limit().await?;

            match self.query_search_engine(engine_config, query).await {
                Ok(mut search_results) => {
                    results.append(&mut search_results);
                }
                Err(e) => {
                    warn!("Search engine {} failed: {}", engine_config.name, e);
                }
            }
        }

        Ok(results)
    }

    /// Fact-check a claim
    pub async fn fact_check(&self, claim: &str) -> Result<Vec<FactCheckResult>> {
        let mut results = Vec::new();

        for fc_config in &self.config.fact_checkers {
            if !fc_config.enabled {
                continue;
            }

            self.rate_limiter.check_limit().await?;

            match self.check_fact(fc_config, claim).await {
                Ok(fact_result) => {
                    if fact_result.confidence >= fc_config.confidence_threshold {
                        results.push(fact_result);
                    }
                }
                Err(e) => {
                    warn!("Fact checker {} failed: {}", fc_config.name, e);
                }
            }
        }

        Ok(results)
    }

    /// Translate text with automatic language detection
    pub async fn translate(&self, text: &str, target_language: &str) -> Result<TranslationResult> {
        // First detect the source language
        let source_language = self.detect_language(text).await?;
        self.translate_with_source(text, &source_language.language_code, target_language)
            .await
    }

    /// Translate text with known source language
    pub async fn translate_with_source(
        &self,
        text: &str,
        source_language: &str,
        target_language: &str,
    ) -> Result<TranslationResult> {
        for trans_config in &self.config.translation_services {
            if !trans_config.enabled
                || !trans_config
                    .supported_languages
                    .contains(&target_language.to_string())
                || !trans_config
                    .supported_languages
                    .contains(&source_language.to_string())
            {
                continue;
            }

            self.rate_limiter.check_limit().await?;

            match self
                .translate_text(trans_config, text, target_language)
                .await
            {
                Ok(result) => return Ok(result),
                Err(e) => {
                    warn!("Translation service {} failed: {}", trans_config.name, e);
                }
            }
        }

        Err(anyhow!(
            "No translation service available for {} -> {}",
            source_language,
            target_language
        ))
    }

    /// Detect the language of input text
    pub async fn detect_language(&self, text: &str) -> Result<LanguageDetectionResult> {
        for trans_config in &self.config.translation_services {
            if !trans_config.enabled {
                continue;
            }

            self.rate_limiter.check_limit().await?;

            // Use specific language detection for this service
            match self.detect_language_for_service(trans_config, text).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    warn!(
                        "Language detection service {} failed: {}",
                        trans_config.name, e
                    );
                }
            }
        }

        Err(anyhow!("No language detection service available"))
    }

    /// Batch translate multiple texts
    pub async fn batch_translate(
        &self,
        texts: &[String],
        target_language: &str,
    ) -> Result<Vec<TranslationResult>> {
        let mut results = Vec::new();

        for text in texts {
            match self.translate(text, target_language).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    warn!("Failed to translate text '{}': {}", text, e);
                    // Continue with other texts even if one fails
                }
            }
        }

        Ok(results)
    }

    /// Get supported language pairs for translation
    pub fn get_supported_language_pairs(&self) -> Vec<LanguagePair> {
        let mut pairs = Vec::new();

        for trans_config in &self.config.translation_services {
            if !trans_config.enabled {
                continue;
            }

            // Generate all possible pairs from supported languages
            for source in &trans_config.supported_languages {
                for target in &trans_config.supported_languages {
                    if source != target {
                        pairs.push(LanguagePair {
                            source_language: source.clone(),
                            target_language: target.clone(),
                            service: trans_config.name.clone(),
                        });
                    }
                }
            }
        }

        pairs
    }

    /// Check if a specific language pair is supported
    pub fn is_language_pair_supported(&self, source: &str, target: &str) -> bool {
        for trans_config in &self.config.translation_services {
            if trans_config.enabled
                && trans_config
                    .supported_languages
                    .contains(&source.to_string())
                && trans_config
                    .supported_languages
                    .contains(&target.to_string())
            {
                return true;
            }
        }
        false
    }

    /// Convert speech to text with advanced processing options
    pub async fn speech_to_text(&self, audio_data: &[u8], language: &str) -> Result<SpeechResult> {
        let options = SpeechProcessingOptions {
            enable_punctuation: true,
            enable_profanity_filter: false,
            enable_automatic_formatting: true,
            enable_word_timestamps: true,
            enable_confidence_scores: true,
            enable_speaker_identification: false,
            language_detection_mode: LanguageDetectionMode::None,
            audio_quality_enhancement: true,
        };

        self.speech_to_text_with_options(audio_data, language, &options)
            .await
    }

    /// Convert speech to text with custom processing options
    pub async fn speech_to_text_with_options(
        &self,
        audio_data: &[u8],
        language: &str,
        options: &SpeechProcessingOptions,
    ) -> Result<SpeechResult> {
        for speech_config in &self.config.speech_services {
            if !speech_config.enabled
                || speech_config.speech_to_text_url.is_none()
                || !speech_config
                    .supported_languages
                    .contains(&language.to_string())
            {
                continue;
            }

            self.rate_limiter.check_limit().await?;

            match self
                .convert_speech_to_text(speech_config, audio_data, language)
                .await
            {
                Ok(result) => return Ok(result),
                Err(e) => {
                    warn!(
                        "Speech-to-text service {} failed: {}",
                        speech_config.name, e
                    );
                }
            }
        }

        Err(anyhow!(
            "No speech-to-text service available for language: {}",
            language
        ))
    }

    /// Real-time streaming speech recognition
    pub async fn streaming_speech_to_text(
        &self,
        audio_stream: tokio::sync::mpsc::Receiver<Vec<u8>>,
        language: &str,
        options: &SpeechProcessingOptions,
    ) -> Result<tokio::sync::mpsc::Receiver<PartialSpeechResult>> {
        // Find a speech service with real-time streaming support
        for speech_config in &self.config.speech_services {
            if !speech_config.enabled
                || !speech_config.real_time_streaming
                || speech_config.speech_to_text_url.is_none()
                || !speech_config
                    .supported_languages
                    .contains(&language.to_string())
            {
                continue;
            }

            // Streaming speech connection not yet implemented
            return Err(anyhow!("Streaming speech-to-text not yet implemented"));
        }

        Err(anyhow!(
            "No streaming speech-to-text service available for language: {}",
            language
        ))
    }

    /// Convert text to speech with default voice
    pub async fn text_to_speech(&self, text: &str, language: &str) -> Result<Vec<u8>> {
        // Use first available voice model for the language
        for speech_config in &self.config.speech_services {
            if let Some(voice_model) = speech_config
                .voice_models
                .iter()
                .find(|v| v.language == language)
            {
                return self
                    .text_to_speech_with_voice(text, language, voice_model)
                    .await;
            }
        }

        // Fallback to basic TTS if no voice models configured
        self.text_to_speech_basic(text, language).await
    }

    /// Convert text to speech with specific voice model
    pub async fn text_to_speech_with_voice(
        &self,
        text: &str,
        language: &str,
        voice_model: &VoiceModel,
    ) -> Result<Vec<u8>> {
        for speech_config in &self.config.speech_services {
            if !speech_config.enabled
                || speech_config.text_to_speech_url.is_none()
                || !speech_config
                    .voice_models
                    .iter()
                    .any(|v| v.id == voice_model.id)
            {
                continue;
            }

            self.rate_limiter.check_limit().await?;

            match self
                .convert_text_to_speech(speech_config, text, &voice_model.language)
                .await
            {
                Ok(audio_data) => return Ok(audio_data),
                Err(e) => {
                    warn!(
                        "Text-to-speech service {} failed: {}",
                        speech_config.name, e
                    );
                }
            }
        }

        Err(anyhow!(
            "No text-to-speech service available for voice: {}",
            voice_model.name
        ))
    }

    /// Get available voice models for a language
    pub fn get_voice_models(&self, language: &str) -> Vec<&VoiceModel> {
        let mut models = Vec::new();
        for speech_config in &self.config.speech_services {
            if speech_config.enabled {
                models.extend(
                    speech_config
                        .voice_models
                        .iter()
                        .filter(|v| v.language == language),
                );
            }
        }
        models
    }

    /// Get recommended voice model based on preferences
    pub fn get_recommended_voice(
        &self,
        language: &str,
        gender: Option<VoiceGender>,
        voice_type: Option<VoiceType>,
        neural_preferred: bool,
    ) -> Option<&VoiceModel> {
        let mut candidates: Vec<&VoiceModel> = self.get_voice_models(language);

        // Filter by gender if specified
        if let Some(preferred_gender) = gender {
            candidates.retain(|v| {
                std::mem::discriminant(&v.gender) == std::mem::discriminant(&preferred_gender)
            });
        }

        // Filter by voice type if specified
        if let Some(preferred_type) = voice_type {
            candidates.retain(|v| {
                std::mem::discriminant(&v.voice_type) == std::mem::discriminant(&preferred_type)
            });
        }

        // Prefer neural voices if requested
        if neural_preferred {
            let neural_voices: Vec<&VoiceModel> = candidates
                .iter()
                .filter(|v| v.neural_voice)
                .copied()
                .collect();
            if !neural_voices.is_empty() {
                candidates = neural_voices;
            }
        }

        // Return the first candidate or None if no matches
        candidates.first().copied()
    }

    /// Basic text-to-speech fallback method
    async fn text_to_speech_basic(&self, text: &str, language: &str) -> Result<Vec<u8>> {
        for speech_config in &self.config.speech_services {
            if !speech_config.enabled
                || speech_config.text_to_speech_url.is_none()
                || !speech_config
                    .supported_languages
                    .contains(&language.to_string())
            {
                continue;
            }

            self.rate_limiter.check_limit().await?;

            match self
                .convert_text_to_speech(speech_config, text, language)
                .await
            {
                Ok(audio_data) => return Ok(audio_data),
                Err(e) => {
                    warn!(
                        "Text-to-speech service {} failed: {}",
                        speech_config.name, e
                    );
                }
            }
        }

        Err(anyhow!(
            "No text-to-speech service available for language: {}",
            language
        ))
    }

    // Private implementation methods
    async fn query_knowledge_base(
        &self,
        config: &KnowledgeBaseConfig,
        query: &str,
    ) -> Result<Vec<KnowledgeResult>> {
        let mut request = self
            .client
            .get(format!("{}/search", config.base_url))
            .query(&[("q", query)]);

        if let Some(api_key) = &config.api_key {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        for (key, value) in &config.headers {
            request = request.header(key, value);
        }

        let response = request.send().await?;
        let kb_response: KnowledgeBaseResponse = response.json().await?;

        Ok(kb_response
            .results
            .into_iter()
            .map(|r| KnowledgeResult {
                source: config.name.clone(),
                title: r.title,
                content: r.content,
                url: r.url,
                relevance_score: r.score,
                metadata: r.metadata,
            })
            .collect())
    }

    async fn query_search_engine(
        &self,
        config: &SearchEngineConfig,
        query: &str,
    ) -> Result<Vec<SearchResult>> {
        let mut request = self
            .client
            .get(&config.api_url)
            .query(&[("q", query), ("count", &config.max_results.to_string())]);

        if let Some(api_key) = &config.api_key {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        let response = request.send().await?;
        let search_response: SearchEngineResponse = response.json().await?;

        Ok(search_response
            .results
            .into_iter()
            .map(|r| SearchResult {
                engine: config.name.clone(),
                title: r.title,
                description: r.description,
                url: r.url,
                score: r.relevance,
                search_type: config.search_type.clone(),
            })
            .collect())
    }

    async fn check_fact(&self, config: &FactCheckerConfig, claim: &str) -> Result<FactCheckResult> {
        let mut request = self
            .client
            .post(format!("{}/check", config.api_url))
            .json(&serde_json::json!({"claim": claim}));

        if let Some(api_key) = &config.api_key {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        let response = request.send().await?;
        let fact_response: FactCheckerResponse = response.json().await?;

        Ok(FactCheckResult {
            checker: config.name.clone(),
            claim: claim.to_string(),
            verdict: fact_response.verdict,
            confidence: fact_response.confidence,
            explanation: fact_response.explanation,
            sources: fact_response.sources,
        })
    }

    async fn translate_text(
        &self,
        config: &TranslationConfig,
        text: &str,
        target_language: &str,
    ) -> Result<TranslationResult> {
        let mut request = self
            .client
            .post(format!("{}/translate", config.api_url))
            .json(&serde_json::json!({
                "text": text,
                "target": target_language
            }));

        if let Some(api_key) = &config.api_key {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        let response = request.send().await?;
        let trans_response: TranslationResponse = response.json().await?;

        Ok(TranslationResult {
            service: config.name.clone(),
            original_text: text.to_string(),
            translated_text: trans_response.translated_text,
            source_language: trans_response.detected_language.clone(),
            target_language: target_language.to_string(),
            confidence: trans_response.confidence,
            detected_language: Some(trans_response.detected_language),
            alternative_translations: Vec::new(),
            processing_time_ms: 0,
        })
    }

    async fn convert_speech_to_text(
        &self,
        config: &SpeechConfig,
        audio_data: &[u8],
        language: &str,
    ) -> Result<SpeechResult> {
        let url = config.speech_to_text_url.as_ref().unwrap();

        let form = reqwest::multipart::Form::new()
            .part(
                "audio",
                reqwest::multipart::Part::bytes(audio_data.to_vec()),
            )
            .text("language", language.to_string());

        let mut request = self.client.post(url).multipart(form);

        if let Some(api_key) = &config.api_key {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        let response = request.send().await?;
        let speech_response: SpeechToTextResponse = response.json().await?;

        Ok(SpeechResult {
            service: config.name.clone(),
            text: speech_response.text,
            confidence: speech_response.confidence,
            language: language.to_string(),
            word_timestamps: None,
            speaker_info: None,
            emotion_analysis: None,
            processing_time_ms: 0,
        })
    }

    async fn detect_language_for_service(
        &self,
        config: &TranslationConfig,
        text: &str,
    ) -> Result<LanguageDetectionResult> {
        let mut request = self
            .client
            .post(format!("{}/detect", config.api_url))
            .json(&serde_json::json!({"text": text}));

        if let Some(api_key) = &config.api_key {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        let response = request.send().await?;
        let detection_response: serde_json::Value = response.json().await?;

        Ok(LanguageDetectionResult {
            service: config.name.clone(),
            language_code: detection_response["language_code"]
                .as_str()
                .unwrap_or("unknown")
                .to_string(),
            language_name: detection_response["language_name"]
                .as_str()
                .unwrap_or("Unknown")
                .to_string(),
            confidence: detection_response["confidence"].as_f64().unwrap_or(0.0) as f32,
            alternative_languages: Vec::new(),
            text_sample: text.chars().take(50).collect(),
        })
    }

    async fn convert_text_to_speech(
        &self,
        config: &SpeechConfig,
        text: &str,
        language: &str,
    ) -> Result<Vec<u8>> {
        let url = config.text_to_speech_url.as_ref().unwrap();

        let mut request = self.client.post(url).json(&serde_json::json!({
            "text": text,
            "language": language
        }));

        if let Some(api_key) = &config.api_key {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        let response = request.send().await?;
        let audio_data = response.bytes().await?;

        Ok(audio_data.to_vec())
    }
}

/// Rate limiter for API calls
struct RateLimiter {
    limit_per_minute: u32,
    calls_made: std::sync::Arc<std::sync::Mutex<u32>>,
    last_reset: std::sync::Arc<std::sync::Mutex<std::time::Instant>>,
}

impl RateLimiter {
    fn new(limit_per_minute: u32) -> Self {
        Self {
            limit_per_minute,
            calls_made: std::sync::Arc::new(std::sync::Mutex::new(0)),
            last_reset: std::sync::Arc::new(std::sync::Mutex::new(std::time::Instant::now())),
        }
    }

    async fn check_limit(&self) -> Result<()> {
        let now = std::time::Instant::now();

        {
            let mut last_reset = self.last_reset.lock().unwrap();
            if now.duration_since(*last_reset) >= Duration::from_secs(60) {
                *last_reset = now;
                *self.calls_made.lock().unwrap() = 0;
            }
        }

        let mut calls = self.calls_made.lock().unwrap();
        if *calls >= self.limit_per_minute {
            return Err(anyhow!("Rate limit exceeded"));
        }

        *calls += 1;
        Ok(())
    }
}

// Result types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeResult {
    pub source: String,
    pub title: String,
    pub content: String,
    pub url: Option<String>,
    pub relevance_score: f32,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub engine: String,
    pub title: String,
    pub description: String,
    pub url: String,
    pub score: f32,
    pub search_type: SearchType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactCheckResult {
    pub checker: String,
    pub claim: String,
    pub verdict: FactVerdict,
    pub confidence: f32,
    pub explanation: String,
    pub sources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactVerdict {
    True,
    False,
    PartiallyTrue,
    Misleading,
    Unverifiable,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationResult {
    pub service: String,
    pub original_text: String,
    pub translated_text: String,
    pub source_language: String,
    pub target_language: String,
    pub confidence: f32,
    pub detected_language: Option<String>,
    pub alternative_translations: Vec<String>,
    pub processing_time_ms: u64,
}

/// Language detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageDetectionResult {
    pub service: String,
    pub language_code: String,
    pub language_name: String,
    pub confidence: f32,
    pub alternative_languages: Vec<LanguageCandidate>,
    pub text_sample: String,
}

/// Alternative language candidate from detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageCandidate {
    pub language_code: String,
    pub language_name: String,
    pub confidence: f32,
}

/// Language pair for translation capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguagePair {
    pub source_language: String,
    pub target_language: String,
    pub service: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechResult {
    pub service: String,
    pub text: String,
    pub confidence: f32,
    pub language: String,
    pub word_timestamps: Option<Vec<WordTimestamp>>,
    pub speaker_info: Option<Vec<SpeakerInfo>>,
    pub emotion_analysis: Option<EmotionAnalysis>,
    pub processing_time_ms: u64,
}

/// Enhanced result type for partial speech recognition (streaming)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialSpeechResult {
    pub service: String,
    pub partial_text: String,
    pub is_final: bool,
    pub confidence: f32,
    pub language: String,
    pub timestamp: std::time::SystemTime,
}

/// Word-level timestamp information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTimestamp {
    pub word: String,
    pub start_time_ms: u64,
    pub end_time_ms: u64,
    pub confidence: f32,
}

/// Speaker identification information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerInfo {
    pub speaker_id: String,
    pub start_time_ms: u64,
    pub end_time_ms: u64,
    pub confidence: f32,
    pub speaker_characteristics: Option<SpeakerCharacteristics>,
}

/// Speaker voice characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerCharacteristics {
    pub estimated_gender: Option<VoiceGender>,
    pub estimated_age_range: Option<VoiceAgeRange>,
    pub voice_energy: f32,
    pub speaking_rate: f32,
}

/// Emotion analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionAnalysis {
    pub primary_emotion: Emotion,
    pub emotion_scores: HashMap<Emotion, f32>,
    pub valence: f32, // Positive/negative sentiment
    pub arousal: f32, // Energy level
    pub confidence: f32,
}

/// Detected emotions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Emotion {
    Joy,
    Sadness,
    Anger,
    Fear,
    Surprise,
    Disgust,
    Neutral,
    Excitement,
    Frustration,
    Confusion,
}

// Internal response types for external APIs
#[derive(Debug, Deserialize)]
struct KnowledgeBaseResponse {
    results: Vec<KnowledgeBaseResult>,
}

#[derive(Debug, Deserialize)]
struct KnowledgeBaseResult {
    title: String,
    content: String,
    url: Option<String>,
    score: f32,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct SearchEngineResponse {
    results: Vec<SearchEngineResult>,
}

#[derive(Debug, Deserialize)]
struct SearchEngineResult {
    title: String,
    description: String,
    url: String,
    relevance: f32,
}

#[derive(Debug, Deserialize)]
struct FactCheckerResponse {
    verdict: FactVerdict,
    confidence: f32,
    explanation: String,
    sources: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct TranslationResponse {
    translated_text: String,
    detected_language: String,
    confidence: f32,
}

#[derive(Debug, Deserialize)]
struct SpeechToTextResponse {
    text: String,
    confidence: f32,
}

// Implement PartialEq for SearchType to enable comparison
impl PartialEq for SearchType {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (SearchType::Web, SearchType::Web)
                | (SearchType::Academic, SearchType::Academic)
                | (SearchType::News, SearchType::News)
                | (SearchType::Images, SearchType::Images)
                | (SearchType::Videos, SearchType::Videos)
                | (SearchType::Knowledge, SearchType::Knowledge)
        )
    }
}
