//! Voice Interface Module
//!
//! Provides Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities for voice interactions
//! with the chat system. Supports multiple providers and streaming audio.

use anyhow::{Context, Result};
use async_openai::{
    config::OpenAIConfig,
    types::{
        AudioResponseFormat, CreateSpeechRequest, CreateTranscriptionRequestArgs, SpeechModel,
        SpeechResponseFormat, Voice,
    },
    Client,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

/// Voice interface manager
pub struct VoiceInterface {
    config: VoiceConfig,
    stt_provider: Arc<Mutex<dyn SpeechToTextProvider>>,
    tts_provider: Arc<Mutex<dyn TextToSpeechProvider>>,
}

/// Voice interface configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConfig {
    /// Enable speech-to-text
    pub enable_stt: bool,
    /// Enable text-to-speech
    pub enable_tts: bool,
    /// STT provider selection
    pub stt_provider: SttProviderType,
    /// TTS provider selection
    pub tts_provider: TtsProviderType,
    /// Audio sample rate (Hz)
    pub sample_rate: u32,
    /// Audio channels (1 = mono, 2 = stereo)
    pub channels: u16,
    /// Maximum audio duration (seconds)
    pub max_duration_secs: u64,
    /// Language code (e.g., "en-US", "ja-JP")
    pub language: String,
    /// Voice/speaker selection for TTS
    pub voice: String,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            enable_stt: true,
            enable_tts: true,
            stt_provider: SttProviderType::OpenAI,
            tts_provider: TtsProviderType::OpenAI,
            sample_rate: 16000,
            channels: 1,
            max_duration_secs: 300, // 5 minutes
            language: "en-US".to_string(),
            voice: "alloy".to_string(),
        }
    }
}

/// STT provider types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SttProviderType {
    /// OpenAI Whisper API
    OpenAI,
    /// Google Speech-to-Text
    Google,
    /// Azure Speech Services
    Azure,
    /// Local Whisper model
    LocalWhisper,
}

/// TTS provider types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TtsProviderType {
    /// OpenAI TTS API
    OpenAI,
    /// Google Text-to-Speech
    Google,
    /// Azure Speech Services
    Azure,
    /// Local TTS engine
    LocalEngine,
}

/// Speech-to-text provider trait
#[async_trait::async_trait]
pub trait SpeechToTextProvider: Send + Sync {
    /// Transcribe audio file to text
    async fn transcribe(&self, audio_data: &[u8], _config: &VoiceConfig) -> Result<SttResult>;

    /// Transcribe audio stream to text (real-time)
    async fn transcribe_stream(
        &self,
        audio_stream: tokio::sync::mpsc::Receiver<Vec<u8>>,
        config: &VoiceConfig,
    ) -> Result<tokio::sync::mpsc::Receiver<SttStreamResult>>;
}

/// Text-to-speech provider trait
#[async_trait::async_trait]
pub trait TextToSpeechProvider: Send + Sync {
    /// Synthesize text to audio
    async fn synthesize(&self, text: &str, _config: &VoiceConfig) -> Result<TtsResult>;

    /// Synthesize text to audio stream (real-time)
    async fn synthesize_stream(
        &self,
        _text: &str,
        _config: &VoiceConfig,
    ) -> Result<tokio::sync::mpsc::Receiver<Vec<u8>>>;
}

/// Speech-to-text result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttResult {
    /// Transcribed text
    pub text: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Language detected
    pub language: Option<String>,
    /// Processing duration (milliseconds)
    pub duration_ms: u64,
    /// Word-level timestamps
    pub word_timestamps: Vec<WordTimestamp>,
}

/// Streaming STT result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttStreamResult {
    /// Partial or final transcribed text
    pub text: String,
    /// Is this a final result (vs. partial)
    pub is_final: bool,
    /// Confidence score
    pub confidence: f32,
}

/// Word-level timestamp information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTimestamp {
    pub word: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

/// Text-to-speech result
#[derive(Debug, Clone)]
pub struct TtsResult {
    /// Audio data
    pub audio_data: Vec<u8>,
    /// Audio format
    pub format: AudioFormat,
    /// Duration (milliseconds)
    pub duration_ms: u64,
}

/// Audio format
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AudioFormat {
    Wav,
    Mp3,
    Opus,
    Pcm,
}

impl VoiceInterface {
    /// Create a new voice interface
    pub fn new(config: VoiceConfig) -> Self {
        let stt_provider: Arc<Mutex<dyn SpeechToTextProvider>> = match config.stt_provider {
            SttProviderType::OpenAI => Arc::new(Mutex::new(OpenAISttProvider::new(config.clone()))),
            SttProviderType::Google => Arc::new(Mutex::new(GoogleSttProvider::new(config.clone()))),
            SttProviderType::Azure => Arc::new(Mutex::new(AzureSttProvider::new(config.clone()))),
            SttProviderType::LocalWhisper => {
                Arc::new(Mutex::new(LocalWhisperProvider::new(config.clone())))
            }
        };

        let tts_provider: Arc<Mutex<dyn TextToSpeechProvider>> = match config.tts_provider {
            TtsProviderType::OpenAI => Arc::new(Mutex::new(OpenAITtsProvider::new(config.clone()))),
            TtsProviderType::Google => Arc::new(Mutex::new(GoogleTtsProvider::new(config.clone()))),
            TtsProviderType::Azure => Arc::new(Mutex::new(AzureTtsProvider::new(config.clone()))),
            TtsProviderType::LocalEngine => {
                Arc::new(Mutex::new(LocalTtsEngine::new(config.clone())))
            }
        };

        Self {
            config,
            stt_provider,
            tts_provider,
        }
    }

    /// Transcribe audio to text
    pub async fn transcribe(&self, audio_data: &[u8]) -> Result<SttResult> {
        if !self.config.enable_stt {
            anyhow::bail!("Speech-to-text is disabled");
        }

        let provider = self.stt_provider.lock().await;
        provider.transcribe(audio_data, &self.config).await
    }

    /// Synthesize text to speech
    pub async fn synthesize(&self, text: &str) -> Result<TtsResult> {
        if !self.config.enable_tts {
            anyhow::bail!("Text-to-speech is disabled");
        }

        let provider = self.tts_provider.lock().await;
        provider.synthesize(text, &self.config).await
    }
}

// ========== Provider Implementations ==========

/// OpenAI STT Provider (Whisper API)
struct OpenAISttProvider {
    config: VoiceConfig,
    client: Client<OpenAIConfig>,
}

impl OpenAISttProvider {
    fn new(config: VoiceConfig) -> Self {
        let client = Client::new();
        Self { config, client }
    }
}

#[async_trait::async_trait]
impl SpeechToTextProvider for OpenAISttProvider {
    async fn transcribe(&self, audio_data: &[u8], config: &VoiceConfig) -> Result<SttResult> {
        info!(
            "Transcribing audio with OpenAI Whisper (size: {} bytes)",
            audio_data.len()
        );

        let start_time = std::time::Instant::now();

        // Create transcription request using OpenAI Whisper
        let request = CreateTranscriptionRequestArgs::default()
            .file(async_openai::types::AudioInput {
                source: async_openai::types::InputSource::Bytes {
                    filename: "audio.mp3".to_string(),
                    bytes: audio_data.to_vec().into(),
                },
            })
            .model("whisper-1")
            .language(&config.language[..2]) // Convert "en-US" to "en"
            .response_format(AudioResponseFormat::VerboseJson)
            .build()
            .context("Failed to build transcription request")?;

        // Call OpenAI Whisper API
        let response = self
            .client
            .audio()
            .transcribe(request)
            .await
            .context("Failed to transcribe audio with OpenAI Whisper")?;

        let duration_ms = start_time.elapsed().as_millis() as u64;

        debug!(
            "OpenAI Whisper transcription completed: '{}' (duration: {}ms)",
            response.text, duration_ms
        );

        // Extract word timestamps if available (Whisper verbose JSON includes them)
        let word_timestamps = vec![]; // OpenAI Whisper API doesn't provide word-level timestamps in the current API

        Ok(SttResult {
            text: response.text,
            confidence: 0.95, // OpenAI doesn't provide confidence scores
            language: Some(config.language.clone()),
            duration_ms,
            word_timestamps,
        })
    }

    async fn transcribe_stream(
        &self,
        mut audio_stream: tokio::sync::mpsc::Receiver<Vec<u8>>,
        config: &VoiceConfig,
    ) -> Result<tokio::sync::mpsc::Receiver<SttStreamResult>> {
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        let client = self.client.clone();
        let language = config.language.clone();

        // Spawn background task for streaming transcription
        // Note: OpenAI Whisper doesn't natively support streaming, so we accumulate chunks
        tokio::spawn(async move {
            let mut accumulated_audio = Vec::new();

            while let Some(audio_chunk) = audio_stream.recv().await {
                accumulated_audio.extend_from_slice(&audio_chunk);

                // Process accumulated audio every 5 seconds worth of data (approx)
                // Assuming 16kHz, 1 channel, 16-bit = 32KB per second
                if accumulated_audio.len() >= 160_000 {
                    // Create transcription request
                    match CreateTranscriptionRequestArgs::default()
                        .file(async_openai::types::AudioInput {
                            source: async_openai::types::InputSource::Bytes {
                                filename: "audio_chunk.mp3".to_string(),
                                bytes: accumulated_audio.clone().into(),
                            },
                        })
                        .model("whisper-1")
                        .language(&language[..2])
                        .response_format(AudioResponseFormat::Json)
                        .build()
                    {
                        Ok(request) => {
                            if let Ok(response) = client.audio().transcribe(request).await {
                                let _ = tx
                                    .send(SttStreamResult {
                                        text: response.text,
                                        is_final: false,
                                        confidence: 0.95,
                                    })
                                    .await;
                            }
                        }
                        Err(e) => {
                            warn!("Failed to create transcription request: {}", e);
                        }
                    }

                    // Clear accumulated audio after processing
                    accumulated_audio.clear();
                }
            }

            // Process any remaining audio
            if !accumulated_audio.is_empty() {
                if let Ok(request) = CreateTranscriptionRequestArgs::default()
                    .file(async_openai::types::AudioInput {
                        source: async_openai::types::InputSource::Bytes {
                            filename: "audio_final.mp3".to_string(),
                            bytes: accumulated_audio.into(),
                        },
                    })
                    .model("whisper-1")
                    .language(&language[..2])
                    .response_format(AudioResponseFormat::Json)
                    .build()
                {
                    if let Ok(response) = client.audio().transcribe(request).await {
                        let _ = tx
                            .send(SttStreamResult {
                                text: response.text,
                                is_final: true,
                                confidence: 0.95,
                            })
                            .await;
                    }
                }
            }
        });

        Ok(rx)
    }
}

/// Google STT Provider (placeholder)
struct GoogleSttProvider {
    config: VoiceConfig,
}

impl GoogleSttProvider {
    fn new(config: VoiceConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl SpeechToTextProvider for GoogleSttProvider {
    async fn transcribe(&self, _audio_data: &[u8], _config: &VoiceConfig) -> Result<SttResult> {
        warn!("Google STT integration not yet implemented");
        Ok(SttResult {
            text: "[Google STT placeholder]".to_string(),
            confidence: 0.90,
            language: Some("en-US".to_string()),
            duration_ms: 1000,
            word_timestamps: vec![],
        })
    }

    async fn transcribe_stream(
        &self,
        _audio_stream: tokio::sync::mpsc::Receiver<Vec<u8>>,
        _config: &VoiceConfig,
    ) -> Result<tokio::sync::mpsc::Receiver<SttStreamResult>> {
        let (_tx, rx) = tokio::sync::mpsc::channel(100);
        Ok(rx)
    }
}

/// Azure STT Provider (placeholder)
struct AzureSttProvider {
    config: VoiceConfig,
}

impl AzureSttProvider {
    fn new(config: VoiceConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl SpeechToTextProvider for AzureSttProvider {
    async fn transcribe(&self, _audio_data: &[u8], _config: &VoiceConfig) -> Result<SttResult> {
        warn!("Azure STT integration not yet implemented");
        Ok(SttResult {
            text: "[Azure STT placeholder]".to_string(),
            confidence: 0.92,
            language: Some("en-US".to_string()),
            duration_ms: 1000,
            word_timestamps: vec![],
        })
    }

    async fn transcribe_stream(
        &self,
        _audio_stream: tokio::sync::mpsc::Receiver<Vec<u8>>,
        _config: &VoiceConfig,
    ) -> Result<tokio::sync::mpsc::Receiver<SttStreamResult>> {
        let (_tx, rx) = tokio::sync::mpsc::channel(100);
        Ok(rx)
    }
}

/// Local Whisper Provider (placeholder)
struct LocalWhisperProvider {
    config: VoiceConfig,
}

impl LocalWhisperProvider {
    fn new(config: VoiceConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl SpeechToTextProvider for LocalWhisperProvider {
    async fn transcribe(&self, _audio_data: &[u8], _config: &VoiceConfig) -> Result<SttResult> {
        warn!("Local Whisper integration not yet implemented");
        Ok(SttResult {
            text: "[Local Whisper placeholder]".to_string(),
            confidence: 0.88,
            language: Some("en-US".to_string()),
            duration_ms: 1000,
            word_timestamps: vec![],
        })
    }

    async fn transcribe_stream(
        &self,
        _audio_stream: tokio::sync::mpsc::Receiver<Vec<u8>>,
        _config: &VoiceConfig,
    ) -> Result<tokio::sync::mpsc::Receiver<SttStreamResult>> {
        let (_tx, rx) = tokio::sync::mpsc::channel(100);
        Ok(rx)
    }
}

/// OpenAI TTS Provider
struct OpenAITtsProvider {
    config: VoiceConfig,
    client: Client<OpenAIConfig>,
}

impl OpenAITtsProvider {
    fn new(config: VoiceConfig) -> Self {
        let client = Client::new();
        Self { config, client }
    }
}

#[async_trait::async_trait]
impl TextToSpeechProvider for OpenAITtsProvider {
    async fn synthesize(&self, text: &str, config: &VoiceConfig) -> Result<TtsResult> {
        info!(
            "Synthesizing speech with OpenAI TTS (text length: {} chars)",
            text.len()
        );

        let start_time = std::time::Instant::now();

        // Map voice string to OpenAI Voice enum
        let voice = match config.voice.as_str() {
            "alloy" => Voice::Alloy,
            "echo" => Voice::Echo,
            "fable" => Voice::Fable,
            "onyx" => Voice::Onyx,
            "nova" => Voice::Nova,
            "shimmer" => Voice::Shimmer,
            _ => Voice::Alloy, // Default fallback
        };

        // Create TTS request
        let request = CreateSpeechRequest {
            model: SpeechModel::Tts1,
            input: text.to_string(),
            voice,
            response_format: Some(SpeechResponseFormat::Mp3),
            speed: Some(1.0),
        };

        // Call OpenAI TTS API
        let response = self
            .client
            .audio()
            .speech(request)
            .await
            .context("Failed to synthesize speech with OpenAI TTS")?;

        let duration_ms = start_time.elapsed().as_millis() as u64;

        // Read audio bytes from response
        let audio_data = response.bytes.to_vec();

        debug!(
            "OpenAI TTS synthesis completed: {} bytes (duration: {}ms)",
            audio_data.len(),
            duration_ms
        );

        Ok(TtsResult {
            audio_data,
            format: AudioFormat::Mp3,
            duration_ms,
        })
    }

    async fn synthesize_stream(
        &self,
        text: &str,
        config: &VoiceConfig,
    ) -> Result<tokio::sync::mpsc::Receiver<Vec<u8>>> {
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        let client = self.client.clone();
        let text = text.to_string();
        let voice_str = config.voice.clone();

        // Spawn background task for streaming TTS
        tokio::spawn(async move {
            // Map voice string to OpenAI Voice enum
            let voice = match voice_str.as_str() {
                "alloy" => Voice::Alloy,
                "echo" => Voice::Echo,
                "fable" => Voice::Fable,
                "onyx" => Voice::Onyx,
                "nova" => Voice::Nova,
                "shimmer" => Voice::Shimmer,
                _ => Voice::Alloy,
            };

            // For streaming, we split text into sentences and synthesize each separately
            let sentences: Vec<&str> = text
                .split(['.', '!', '?'])
                .filter(|s| !s.trim().is_empty())
                .collect();

            for sentence in sentences {
                let request = CreateSpeechRequest {
                    model: SpeechModel::Tts1,
                    input: sentence.trim().to_string(),
                    voice: voice.clone(),
                    response_format: Some(SpeechResponseFormat::Mp3),
                    speed: Some(1.0),
                };

                match client.audio().speech(request).await {
                    Ok(response) => {
                        let audio_chunk = response.bytes.to_vec();
                        if tx.send(audio_chunk).await.is_err() {
                            break; // Receiver dropped
                        }
                    }
                    Err(e) => {
                        warn!("Failed to synthesize sentence in streaming mode: {}", e);
                        break;
                    }
                }
            }
        });

        Ok(rx)
    }
}

/// Google TTS Provider (placeholder)
struct GoogleTtsProvider {
    config: VoiceConfig,
}

impl GoogleTtsProvider {
    fn new(config: VoiceConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl TextToSpeechProvider for GoogleTtsProvider {
    async fn synthesize(&self, text: &str, _config: &VoiceConfig) -> Result<TtsResult> {
        warn!("Google TTS integration not yet implemented");
        Ok(TtsResult {
            audio_data: vec![],
            format: AudioFormat::Mp3,
            duration_ms: (text.len() as u64) * 100,
        })
    }

    async fn synthesize_stream(
        &self,
        _text: &str,
        _config: &VoiceConfig,
    ) -> Result<tokio::sync::mpsc::Receiver<Vec<u8>>> {
        let (_tx, rx) = tokio::sync::mpsc::channel(100);
        Ok(rx)
    }
}

/// Azure TTS Provider (placeholder)
struct AzureTtsProvider {
    config: VoiceConfig,
}

impl AzureTtsProvider {
    fn new(config: VoiceConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl TextToSpeechProvider for AzureTtsProvider {
    async fn synthesize(&self, text: &str, _config: &VoiceConfig) -> Result<TtsResult> {
        warn!("Azure TTS integration not yet implemented");
        Ok(TtsResult {
            audio_data: vec![],
            format: AudioFormat::Wav,
            duration_ms: (text.len() as u64) * 100,
        })
    }

    async fn synthesize_stream(
        &self,
        _text: &str,
        _config: &VoiceConfig,
    ) -> Result<tokio::sync::mpsc::Receiver<Vec<u8>>> {
        let (_tx, rx) = tokio::sync::mpsc::channel(100);
        Ok(rx)
    }
}

/// Local TTS Engine (placeholder)
struct LocalTtsEngine {
    config: VoiceConfig,
}

impl LocalTtsEngine {
    fn new(config: VoiceConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl TextToSpeechProvider for LocalTtsEngine {
    async fn synthesize(&self, text: &str, _config: &VoiceConfig) -> Result<TtsResult> {
        warn!("Local TTS engine not yet implemented");
        Ok(TtsResult {
            audio_data: vec![],
            format: AudioFormat::Wav,
            duration_ms: (text.len() as u64) * 100,
        })
    }

    async fn synthesize_stream(
        &self,
        _text: &str,
        _config: &VoiceConfig,
    ) -> Result<tokio::sync::mpsc::Receiver<Vec<u8>>> {
        let (_tx, rx) = tokio::sync::mpsc::channel(100);
        Ok(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_voice_interface_creation() {
        let config = VoiceConfig::default();
        let interface = VoiceInterface::new(config);
        assert!(interface.config.enable_stt);
        assert!(interface.config.enable_tts);
    }

    #[tokio::test]
    async fn test_transcribe_disabled() {
        let config = VoiceConfig {
            enable_stt: false,
            ..Default::default()
        };

        let interface = VoiceInterface::new(config);

        let audio_data = vec![0u8; 1000];
        let result = interface.transcribe(&audio_data).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("disabled"));
    }

    #[tokio::test]
    async fn test_synthesize_disabled() {
        let config = VoiceConfig {
            enable_tts: false,
            ..Default::default()
        };

        let interface = VoiceInterface::new(config);

        let text = "Hello, world!";
        let result = interface.synthesize(text).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("disabled"));
    }

    #[test]
    fn test_voice_config_custom() {
        let config = VoiceConfig {
            enable_stt: true,
            enable_tts: false,
            stt_provider: SttProviderType::Google,
            tts_provider: TtsProviderType::Azure,
            sample_rate: 48000,
            channels: 2,
            max_duration_secs: 600,
            language: "ja-JP".to_string(),
            voice: "echo".to_string(),
        };

        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.channels, 2);
        assert_eq!(config.language, "ja-JP");
        assert_eq!(config.voice, "echo");
        assert!(!config.enable_tts);
    }

    #[test]
    fn test_audio_format_variants() {
        assert!(matches!(AudioFormat::Wav, AudioFormat::Wav));
        assert!(matches!(AudioFormat::Mp3, AudioFormat::Mp3));
    }

    #[test]
    fn test_stt_provider_serialization() {
        let provider = SttProviderType::OpenAI;
        let serialized = serde_json::to_string(&provider).unwrap();
        assert_eq!(serialized, "\"open_a_i\""); // Snake case serialization

        let deserialized: SttProviderType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, SttProviderType::OpenAI);
    }

    #[test]
    fn test_tts_provider_serialization() {
        let provider = TtsProviderType::Google;
        let serialized = serde_json::to_string(&provider).unwrap();
        assert_eq!(serialized, "\"google\"");

        let deserialized: TtsProviderType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, TtsProviderType::Google);
    }
}
