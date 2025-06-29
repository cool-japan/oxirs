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
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

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

    /// Translate text
    pub async fn translate(&self, text: &str, target_language: &str) -> Result<TranslationResult> {
        for trans_config in &self.config.translation_services {
            if !trans_config.enabled
                || !trans_config
                    .supported_languages
                    .contains(&target_language.to_string())
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
            "No translation service available for language: {}",
            target_language
        ))
    }

    /// Convert speech to text
    pub async fn speech_to_text(&self, audio_data: &[u8], language: &str) -> Result<SpeechResult> {
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

    /// Convert text to speech
    pub async fn text_to_speech(&self, text: &str, language: &str) -> Result<Vec<u8>> {
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
            .get(&format!("{}/search", config.base_url))
            .query(&[("q", query)]);

        if let Some(api_key) = &config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
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
            request = request.header("Authorization", format!("Bearer {}", api_key));
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
            .post(&format!("{}/check", config.api_url))
            .json(&serde_json::json!({"claim": claim}));

        if let Some(api_key) = &config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
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
            .post(&format!("{}/translate", config.api_url))
            .json(&serde_json::json!({
                "text": text,
                "target": target_language
            }));

        if let Some(api_key) = &config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = request.send().await?;
        let trans_response: TranslationResponse = response.json().await?;

        Ok(TranslationResult {
            service: config.name.clone(),
            original_text: text.to_string(),
            translated_text: trans_response.translated_text,
            source_language: trans_response.detected_language,
            target_language: target_language.to_string(),
            confidence: trans_response.confidence,
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
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = request.send().await?;
        let speech_response: SpeechToTextResponse = response.json().await?;

        Ok(SpeechResult {
            service: config.name.clone(),
            text: speech_response.text,
            confidence: speech_response.confidence,
            language: language.to_string(),
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
            request = request.header("Authorization", format!("Bearer {}", api_key));
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechResult {
    pub service: String,
    pub text: String,
    pub confidence: f32,
    pub language: String,
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
