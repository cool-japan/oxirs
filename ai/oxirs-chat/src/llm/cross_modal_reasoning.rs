use crate::llm::manager::LLMManager;
use crate::llm::types::{ChatMessage, ChatRole, LLMRequest, Priority, Usage, UseCase};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalInput {
    pub text: Option<String>,
    pub images: Vec<ImageInput>,
    pub structured_data: Option<StructuredData>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageInput {
    pub data: Vec<u8>,
    pub format: ImageFormat,
    pub description: Option<String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImageFormat {
    Jpeg,
    Png,
    Gif,
    Webp,
    Svg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredData {
    pub format: DataFormat,
    pub data: String,
    pub schema: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataFormat {
    Json,
    Xml,
    Csv,
    Rdf,
    Sparql,
    GraphQL,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalResponse {
    pub reasoning_chain: Vec<ReasoningStep>,
    pub final_answer: String,
    pub confidence: f32,
    pub modality_contributions: HashMap<String, f32>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub step_id: String,
    pub description: String,
    pub modality: ReasoningModality,
    pub input_references: Vec<String>,
    pub output: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningModality {
    Text,
    Vision,
    Structured,
    Multimodal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalConfig {
    pub enable_vision: bool,
    pub enable_structured_reasoning: bool,
    pub max_images: usize,
    pub max_structured_size: usize,
    pub reasoning_depth: usize,
    pub confidence_threshold: f32,
    pub fusion_strategy: FusionStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    EarlyFusion,
    LateFusion,
    HybridFusion,
    AdaptiveFusion,
}

impl Default for CrossModalConfig {
    fn default() -> Self {
        Self {
            enable_vision: true,
            enable_structured_reasoning: true,
            max_images: 10,
            max_structured_size: 1024 * 1024, // 1MB
            reasoning_depth: 5,
            confidence_threshold: 0.7,
            fusion_strategy: FusionStrategy::HybridFusion,
        }
    }
}

pub struct CrossModalReasoning {
    config: CrossModalConfig,
    llm_manager: Arc<RwLock<LLMManager>>,
    vision_processor: Arc<RwLock<VisionProcessor>>,
    structured_processor: Arc<RwLock<StructuredProcessor>>,
    fusion_engine: Arc<RwLock<FusionEngine>>,
    reasoning_history: Arc<RwLock<Vec<CrossModalResponse>>>,
}

impl CrossModalReasoning {
    pub fn new(config: CrossModalConfig, llm_manager: Arc<RwLock<LLMManager>>) -> Self {
        Self {
            config,
            llm_manager,
            vision_processor: Arc::new(RwLock::new(VisionProcessor::new())),
            structured_processor: Arc::new(RwLock::new(StructuredProcessor::new())),
            fusion_engine: Arc::new(RwLock::new(FusionEngine::new())),
            reasoning_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn reason(
        &self,
        input: CrossModalInput,
        query: &str,
    ) -> Result<CrossModalResponse, Box<dyn std::error::Error + Send + Sync>> {
        let mut reasoning_steps = Vec::new();
        let mut modality_contributions = HashMap::new();
        let mut total_usage = Usage::default();

        // Step 1: Process text input
        if let Some(text) = &input.text {
            let text_result = self.process_text(text, query).await?;
            reasoning_steps.push(ReasoningStep {
                step_id: format!("text_{}", reasoning_steps.len()),
                description: "Text understanding and analysis".to_string(),
                modality: ReasoningModality::Text,
                input_references: vec!["text_input".to_string()],
                output: text_result.content.clone(),
                confidence: text_result.confidence,
            });
            modality_contributions.insert("text".to_string(), text_result.confidence);
            total_usage.prompt_tokens += text_result.usage.prompt_tokens;
            total_usage.completion_tokens += text_result.usage.completion_tokens;
        }

        // Step 2: Process images
        if !input.images.is_empty() && self.config.enable_vision {
            let vision_result = self.process_images(&input.images, query).await?;
            reasoning_steps.push(ReasoningStep {
                step_id: format!("vision_{}", reasoning_steps.len()),
                description: "Visual content analysis".to_string(),
                modality: ReasoningModality::Vision,
                input_references: (0..input.images.len())
                    .map(|i| format!("image_{i}"))
                    .collect(),
                output: vision_result.content.clone(),
                confidence: vision_result.confidence,
            });
            modality_contributions.insert("vision".to_string(), vision_result.confidence);
            total_usage.prompt_tokens += vision_result.usage.prompt_tokens;
            total_usage.completion_tokens += vision_result.usage.completion_tokens;
        }

        // Step 3: Process structured data
        if let Some(structured) = &input.structured_data {
            if self.config.enable_structured_reasoning {
                let structured_result = self.process_structured_data(structured, query).await?;
                reasoning_steps.push(ReasoningStep {
                    step_id: format!("structured_{}", reasoning_steps.len()),
                    description: "Structured data analysis".to_string(),
                    modality: ReasoningModality::Structured,
                    input_references: vec!["structured_data".to_string()],
                    output: structured_result.content.clone(),
                    confidence: structured_result.confidence,
                });
                modality_contributions
                    .insert("structured".to_string(), structured_result.confidence);
                total_usage.prompt_tokens += structured_result.usage.prompt_tokens;
                total_usage.completion_tokens += structured_result.usage.completion_tokens;
            }
        }

        // Step 4: Fusion and final reasoning
        let fusion_result = self.fuse_modalities(&reasoning_steps, query).await?;
        reasoning_steps.push(ReasoningStep {
            step_id: format!("fusion_{}", reasoning_steps.len()),
            description: "Cross-modal reasoning and synthesis".to_string(),
            modality: ReasoningModality::Multimodal,
            input_references: reasoning_steps.iter().map(|s| s.step_id.clone()).collect(),
            output: fusion_result.content.clone(),
            confidence: fusion_result.confidence,
        });

        total_usage.prompt_tokens += fusion_result.usage.prompt_tokens;
        total_usage.completion_tokens += fusion_result.usage.completion_tokens;

        let response = CrossModalResponse {
            reasoning_chain: reasoning_steps,
            final_answer: fusion_result.content,
            confidence: fusion_result.confidence,
            modality_contributions,
            usage: total_usage,
        };

        // Store in history
        self.reasoning_history.write().await.push(response.clone());

        Ok(response)
    }

    async fn process_text(
        &self,
        text: &str,
        query: &str,
    ) -> Result<ModalityResult, Box<dyn std::error::Error + Send + Sync>> {
        let prompt = format!(
            "Analyze the following text in the context of the query: '{query}'\n\nText: {text}\n\nProvide detailed analysis and relevant insights:"
        );

        let request = LLMRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: prompt,
                metadata: None,
            }],
            max_tokens: Some(1000),
            temperature: 0.7,
            system_prompt: Some(
                "You are an expert text analyst. Provide thorough, accurate analysis.".to_string(),
            ),
            use_case: UseCase::Analysis,
            priority: Priority::Normal,
            timeout: None,
        };

        let response = self
            .llm_manager
            .write()
            .await
            .generate_response(request)
            .await?;

        Ok(ModalityResult {
            content: response.content,
            confidence: 0.8, // Text analysis typically has high confidence
            usage: response.usage,
        })
    }

    async fn process_images(
        &self,
        images: &[ImageInput],
        query: &str,
    ) -> Result<ModalityResult, Box<dyn std::error::Error + Send + Sync>> {
        let mut vision_processor = self.vision_processor.write().await;
        let vision_analysis = vision_processor.analyze_images(images).await?;

        let prompt = format!(
            "Analyze the following visual content in the context of the query: '{query}'\n\nVision Analysis: {vision_analysis}\n\nProvide detailed visual insights:"
        );

        let request = LLMRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: prompt,
                metadata: None,
            }],
            max_tokens: Some(1000),
            temperature: 0.7,
            system_prompt: Some(
                "You are an expert visual analyst. Provide thorough, accurate visual analysis."
                    .to_string(),
            ),
            use_case: UseCase::Analysis,
            priority: Priority::Normal,
            timeout: None,
        };

        let response = self
            .llm_manager
            .write()
            .await
            .generate_response(request)
            .await?;

        Ok(ModalityResult {
            content: response.content,
            confidence: 0.75, // Vision analysis confidence varies
            usage: response.usage,
        })
    }

    async fn process_structured_data(
        &self,
        data: &StructuredData,
        query: &str,
    ) -> Result<ModalityResult, Box<dyn std::error::Error + Send + Sync>> {
        let mut structured_processor = self.structured_processor.write().await;
        let structured_analysis = structured_processor.analyze_data(data).await?;

        let prompt = format!(
            "Analyze the following structured data in the context of the query: '{}'\n\nData Format: {:?}\nData Analysis: {}\n\nProvide detailed structural insights:",
            query, data.format, structured_analysis
        );

        let request = LLMRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: prompt,
                metadata: None,
            }],
            max_tokens: Some(1000),
            temperature: 0.7,
            system_prompt: Some(
                "You are an expert data analyst. Provide thorough, accurate structural analysis."
                    .to_string(),
            ),
            use_case: UseCase::Analysis,
            priority: Priority::Normal,
            timeout: None,
        };

        let response = self
            .llm_manager
            .write()
            .await
            .generate_response(request)
            .await?;

        Ok(ModalityResult {
            content: response.content,
            confidence: 0.85, // Structured data analysis typically has high confidence
            usage: response.usage,
        })
    }

    async fn fuse_modalities(
        &self,
        steps: &[ReasoningStep],
        query: &str,
    ) -> Result<ModalityResult, Box<dyn std::error::Error + Send + Sync>> {
        let mut fusion_engine = self.fusion_engine.write().await;
        let fusion_context = fusion_engine
            .create_fusion_context(steps, &self.config.fusion_strategy)
            .await?;

        let prompt = format!(
            "Given the following multi-modal analysis results for the query: '{query}', provide a comprehensive synthesized answer:\n\n{fusion_context}\n\nSynthesis:"
        );

        let request = LLMRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: prompt,
                metadata: None,
            }],
            max_tokens: Some(1500),
            temperature: 0.6,
            system_prompt: Some("You are an expert cross-modal reasoner. Synthesize information from multiple modalities to provide comprehensive, accurate answers.".to_string()),
            use_case: UseCase::ComplexReasoning,
            priority: Priority::Normal,
            timeout: None,
        };

        let response = self
            .llm_manager
            .write()
            .await
            .generate_response(request)
            .await?;

        // Calculate fusion confidence based on input confidences
        let avg_confidence = steps.iter().map(|s| s.confidence).sum::<f32>() / steps.len() as f32;

        Ok(ModalityResult {
            content: response.content,
            confidence: avg_confidence,
            usage: response.usage,
        })
    }

    pub async fn get_reasoning_history(&self) -> Vec<CrossModalResponse> {
        self.reasoning_history.read().await.clone()
    }

    pub async fn clear_history(&self) {
        self.reasoning_history.write().await.clear();
    }

    pub async fn get_stats(&self) -> CrossModalStats {
        let history = self.reasoning_history.read().await;
        let total_requests = history.len();
        let avg_confidence = if total_requests > 0 {
            history.iter().map(|r| r.confidence).sum::<f32>() / total_requests as f32
        } else {
            0.0
        };

        let modality_usage = history.iter().fold(HashMap::new(), |mut acc, response| {
            for (modality, contribution) in &response.modality_contributions {
                *acc.entry(modality.clone()).or_insert(0.0) += contribution;
            }
            acc
        });

        CrossModalStats {
            total_requests,
            avg_confidence,
            modality_usage,
            avg_reasoning_steps: if total_requests > 0 {
                history
                    .iter()
                    .map(|r| r.reasoning_chain.len())
                    .sum::<usize>() as f32
                    / total_requests as f32
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug, Clone)]
struct ModalityResult {
    content: String,
    confidence: f32,
    usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalStats {
    pub total_requests: usize,
    pub avg_confidence: f32,
    pub modality_usage: HashMap<String, f32>,
    pub avg_reasoning_steps: f32,
}

// Vision Processing Component
pub struct VisionProcessor {
    supported_formats: Vec<ImageFormat>,
}

impl Default for VisionProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl VisionProcessor {
    pub fn new() -> Self {
        Self {
            supported_formats: vec![
                ImageFormat::Jpeg,
                ImageFormat::Png,
                ImageFormat::Gif,
                ImageFormat::Webp,
                ImageFormat::Svg,
            ],
        }
    }

    pub async fn analyze_images(
        &mut self,
        images: &[ImageInput],
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut analyses = Vec::new();

        for (i, image) in images.iter().enumerate() {
            if !self.supported_formats.contains(&image.format) {
                return Err(format!("Unsupported image format: {:?}", image.format).into());
            }

            let analysis = self.analyze_single_image(image).await?;
            analyses.push(format!("Image {}: {}", i + 1, analysis));
        }

        Ok(analyses.join("\n"))
    }

    async fn analyze_single_image(
        &self,
        image: &ImageInput,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Simulate image analysis
        let size_analysis = format!("Image size: {} bytes", image.data.len());
        let format_analysis = format!("Format: {:?}", image.format);
        let description = image
            .description
            .as_ref()
            .map(|d| format!("Description: {d}"))
            .unwrap_or_default();

        Ok(format!("{size_analysis}, {format_analysis}, {description}"))
    }
}

// Structured Data Processing Component
pub struct StructuredProcessor {
    supported_formats: Vec<DataFormat>,
}

impl Default for StructuredProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl StructuredProcessor {
    pub fn new() -> Self {
        Self {
            supported_formats: vec![
                DataFormat::Json,
                DataFormat::Xml,
                DataFormat::Csv,
                DataFormat::Rdf,
                DataFormat::Sparql,
                DataFormat::GraphQL,
            ],
        }
    }

    pub async fn analyze_data(
        &mut self,
        data: &StructuredData,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        if !self.supported_formats.contains(&data.format) {
            return Err(format!("Unsupported data format: {:?}", data.format).into());
        }

        match data.format {
            DataFormat::Json => self.analyze_json(&data.data).await,
            DataFormat::Xml => self.analyze_xml(&data.data).await,
            DataFormat::Csv => self.analyze_csv(&data.data).await,
            DataFormat::Rdf => self.analyze_rdf(&data.data).await,
            DataFormat::Sparql => self.analyze_sparql(&data.data).await,
            DataFormat::GraphQL => self.analyze_graphql(&data.data).await,
        }
    }

    async fn analyze_json(
        &self,
        data: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Parse and analyze JSON structure
        let parsed: serde_json::Value = serde_json::from_str(data)?;
        let analysis = format!(
            "JSON structure with {} top-level keys",
            self.count_json_keys(&parsed)
        );
        Ok(analysis)
    }

    async fn analyze_xml(
        &self,
        data: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Simple XML analysis
        let element_count = data.matches('<').count() / 2; // Rough estimate
        Ok(format!(
            "XML document with approximately {element_count} elements"
        ))
    }

    async fn analyze_csv(
        &self,
        data: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let lines: Vec<&str> = data.lines().collect();
        let rows = lines.len();
        let columns = lines
            .first()
            .map(|line| line.split(',').count())
            .unwrap_or(0);
        Ok(format!("CSV data with {rows} rows and {columns} columns"))
    }

    async fn analyze_rdf(
        &self,
        data: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let triple_count = data
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
            .count();
        Ok(format!(
            "RDF data with approximately {triple_count} triples"
        ))
    }

    async fn analyze_sparql(
        &self,
        data: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let query_type = if data.to_uppercase().contains("SELECT") {
            "SELECT"
        } else if data.to_uppercase().contains("CONSTRUCT") {
            "CONSTRUCT"
        } else if data.to_uppercase().contains("ASK") {
            "ASK"
        } else if data.to_uppercase().contains("DESCRIBE") {
            "DESCRIBE"
        } else {
            "UNKNOWN"
        };
        Ok(format!("SPARQL {query_type} query"))
    }

    async fn analyze_graphql(
        &self,
        data: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let operation_type = if data.contains("query") {
            "query"
        } else if data.contains("mutation") {
            "mutation"
        } else if data.contains("subscription") {
            "subscription"
        } else {
            "unknown"
        };
        Ok(format!("GraphQL {operation_type} operation"))
    }

    fn count_json_keys(&self, value: &serde_json::Value) -> usize {
        match value {
            serde_json::Value::Object(obj) => obj.len(),
            _ => 0,
        }
    }
}

// Fusion Engine Component
pub struct FusionEngine {
    strategies: HashMap<String, Box<dyn FusionStrategyTrait + Send + Sync>>,
}

impl Default for FusionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl FusionEngine {
    pub fn new() -> Self {
        let mut strategies = HashMap::new();
        strategies.insert(
            "early".to_string(),
            Box::new(EarlyFusionStrategy) as Box<dyn FusionStrategyTrait + Send + Sync>,
        );
        strategies.insert(
            "late".to_string(),
            Box::new(LateFusionStrategy) as Box<dyn FusionStrategyTrait + Send + Sync>,
        );
        strategies.insert(
            "hybrid".to_string(),
            Box::new(HybridFusionStrategy) as Box<dyn FusionStrategyTrait + Send + Sync>,
        );
        strategies.insert(
            "adaptive".to_string(),
            Box::new(AdaptiveFusionStrategy) as Box<dyn FusionStrategyTrait + Send + Sync>,
        );

        Self { strategies }
    }

    pub async fn create_fusion_context(
        &mut self,
        steps: &[ReasoningStep],
        strategy: &FusionStrategy,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let strategy_name = match strategy {
            FusionStrategy::EarlyFusion => "early",
            FusionStrategy::LateFusion => "late",
            FusionStrategy::HybridFusion => "hybrid",
            FusionStrategy::AdaptiveFusion => "adaptive",
        };

        if let Some(fusion_strategy) = self.strategies.get(strategy_name) {
            fusion_strategy.fuse(steps).await
        } else {
            Err(format!("Unknown fusion strategy: {strategy_name}").into())
        }
    }
}

#[async_trait::async_trait]
pub trait FusionStrategyTrait {
    async fn fuse(
        &self,
        steps: &[ReasoningStep],
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>>;
}

pub struct EarlyFusionStrategy;

#[async_trait::async_trait]
impl FusionStrategyTrait for EarlyFusionStrategy {
    async fn fuse(
        &self,
        steps: &[ReasoningStep],
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut context = String::new();
        context.push_str("Early Fusion Analysis:\n");

        for step in steps {
            context.push_str(&format!(
                "- {}: {} (confidence: {:.2})\n",
                step.description, step.output, step.confidence
            ));
        }

        Ok(context)
    }
}

pub struct LateFusionStrategy;

#[async_trait::async_trait]
impl FusionStrategyTrait for LateFusionStrategy {
    async fn fuse(
        &self,
        steps: &[ReasoningStep],
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut context = String::new();
        context.push_str("Late Fusion Analysis:\n");

        // Group by modality
        let mut modality_groups: HashMap<String, Vec<&ReasoningStep>> = HashMap::new();
        for step in steps {
            let modality_key = format!("{:?}", step.modality);
            modality_groups.entry(modality_key).or_default().push(step);
        }

        for (modality, modality_steps) in modality_groups {
            context.push_str(&format!("\n{modality} Modality:\n"));
            for step in modality_steps {
                context.push_str(&format!("  - {}: {}\n", step.description, step.output));
            }
        }

        Ok(context)
    }
}

pub struct HybridFusionStrategy;

#[async_trait::async_trait]
impl FusionStrategyTrait for HybridFusionStrategy {
    async fn fuse(
        &self,
        steps: &[ReasoningStep],
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut context = String::new();
        context.push_str("Hybrid Fusion Analysis:\n");

        // Combine early and late fusion approaches
        let early_fusion = EarlyFusionStrategy;
        let late_fusion = LateFusionStrategy;

        let early_result = early_fusion.fuse(steps).await?;
        let late_result = late_fusion.fuse(steps).await?;

        context.push_str(&early_result);
        context.push('\n');
        context.push_str(&late_result);

        Ok(context)
    }
}

pub struct AdaptiveFusionStrategy;

#[async_trait::async_trait]
impl FusionStrategyTrait for AdaptiveFusionStrategy {
    async fn fuse(
        &self,
        steps: &[ReasoningStep],
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut context = String::new();
        context.push_str("Adaptive Fusion Analysis:\n");

        // Adapt strategy based on step confidence and modality distribution
        let avg_confidence = steps.iter().map(|s| s.confidence).sum::<f32>() / steps.len() as f32;
        let modality_count = steps
            .iter()
            .map(|s| format!("{:?}", s.modality))
            .collect::<std::collections::HashSet<_>>()
            .len();

        if avg_confidence > 0.8 && modality_count > 2 {
            // High confidence, multiple modalities - use hybrid approach
            let hybrid_fusion = HybridFusionStrategy;
            hybrid_fusion.fuse(steps).await
        } else if modality_count > 2 {
            // Multiple modalities, lower confidence - use late fusion
            let late_fusion = LateFusionStrategy;
            late_fusion.fuse(steps).await
        } else {
            // Single or few modalities - use early fusion
            let early_fusion = EarlyFusionStrategy;
            early_fusion.fuse(steps).await
        }
    }
}
