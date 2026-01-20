//! Personalized chat module with adaptive responses
//!
//! This module provides intelligent response personalization including:
//! - User modeling and expertise level detection
//! - Content adaptation based on user preferences
//! - Communication style adjustment
//! - Multi-modal response generation
//! - Accessibility features

use crate::types::Message;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::time::Duration;

/// User profile for personalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub user_id: String,
    pub expertise_level: ExpertiseLevel,
    pub communication_style: CommunicationStyle,
    pub preferred_formats: Vec<ResponseFormat>,
    pub interests: Vec<String>,
    pub accessibility_needs: AccessibilityNeeds,
    pub learning_preferences: LearningPreferences,
    pub interaction_history: InteractionHistory,
    pub language_preferences: LanguagePreferences,
}

/// User expertise levels
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpertiseLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
    Domain(String), // Expert in specific domain
}

/// Communication style preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationStyle {
    pub formality: FormalityLevel,
    pub detail_level: DetailLevel,
    pub explanation_style: ExplanationStyle,
    pub pace: InteractionPace,
    pub feedback_preference: FeedbackStyle,
}

/// Formality levels
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FormalityLevel {
    Casual,
    Professional,
    Academic,
    Technical,
}

/// Detail levels for responses
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DetailLevel {
    Brief,
    Moderate,
    Detailed,
    Comprehensive,
}

/// Explanation styles
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExplanationStyle {
    StepByStep,
    Conceptual,
    ExampleDriven,
    Analytical,
    Visual,
}

/// Interaction pace preferences
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InteractionPace {
    Quick,
    Moderate,
    Thorough,
    Exploratory,
}

/// Feedback style preferences
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeedbackStyle {
    Immediate,
    Confirmatory,
    Suggestive,
    Minimal,
}

/// Preferred response formats
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResponseFormat {
    Text,
    StructuredText,
    BulletPoints,
    Tables,
    Graphs,
    Code,
    Interactive,
    Audio,
    Video,
}

/// Accessibility needs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilityNeeds {
    pub visual_impairment: bool,
    pub hearing_impairment: bool,
    pub motor_impairment: bool,
    pub cognitive_assistance: bool,
    pub screen_reader_compatible: bool,
    pub high_contrast: bool,
    pub font_size_preference: Option<FontSize>,
}

/// Font size preferences
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FontSize {
    Small,
    Medium,
    Large,
    ExtraLarge,
}

/// Learning preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningPreferences {
    pub learning_style: LearningStyle,
    pub preferred_examples: ExampleType,
    pub scaffolding_level: ScaffoldingLevel,
    pub practice_preference: bool,
}

/// Learning styles
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LearningStyle {
    Visual,
    Auditory,
    Kinesthetic,
    Reading,
    Multimodal,
}

/// Example types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExampleType {
    RealWorld,
    Abstract,
    Historical,
    Current,
    Hypothetical,
}

/// Scaffolding levels
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScaffoldingLevel {
    Minimal,
    Moderate,
    High,
    Adaptive,
}

/// User interaction history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionHistory {
    pub total_sessions: u32,
    pub successful_queries: u32,
    pub failed_queries: u32,
    pub common_topics: Vec<String>,
    pub preferred_query_types: Vec<String>,
    pub avg_session_duration: Duration,
    pub last_active: chrono::DateTime<chrono::Utc>,
    pub satisfaction_scores: Vec<f32>,
}

/// Language preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguagePreferences {
    pub primary_language: String,
    pub secondary_languages: Vec<String>,
    pub technical_terminology: bool,
    pub localization: LocalizationSettings,
}

/// Localization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalizationSettings {
    pub timezone: String,
    pub date_format: String,
    pub number_format: String,
    pub currency: String,
}

/// Personalized response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizedResponse {
    pub content: String,
    pub format: ResponseFormat,
    pub confidence: f32,
    pub adaptation_rationale: String,
    pub accessibility_features: Vec<String>,
    pub suggestions: Vec<String>,
}

/// Chat personalization engine
pub struct ChatPersonalizer {
    user_profiles: HashMap<String, UserProfile>,
    expertise_detector: ExpertiseDetector,
    content_adapter: ContentAdapter,
    accessibility_enhancer: AccessibilityEnhancer,
}

impl ChatPersonalizer {
    /// Create a new chat personalizer
    pub fn new() -> Self {
        Self {
            user_profiles: HashMap::new(),
            expertise_detector: ExpertiseDetector::new(),
            content_adapter: ContentAdapter::new(),
            accessibility_enhancer: AccessibilityEnhancer::new(),
        }
    }

    /// Generate personalized response
    pub async fn personalize_response(
        &mut self,
        user_id: &str,
        query: &str,
        base_response: &str,
        context: &[Message],
    ) -> Result<PersonalizedResponse> {
        let profile = self.get_or_create_user_profile(user_id).await?;

        // Update user model based on interaction
        self.update_user_model(user_id, query, context).await?;

        // Adapt content based on user preferences
        let adapted_content = self
            .content_adapter
            .adapt_content(base_response, &profile, query)
            .await?;

        // Apply accessibility enhancements
        let accessible_content = self
            .accessibility_enhancer
            .enhance_content(&adapted_content, &profile.accessibility_needs)
            .await?;

        // Select appropriate format
        let format = self.select_optimal_format(&profile, query).await?;

        // Calculate confidence in personalization
        let confidence = self.calculate_personalization_confidence(&profile).await?;

        // Generate adaptation rationale
        let rationale = self.generate_adaptation_rationale(&profile, query).await?;

        // Generate suggestions for better interaction
        let suggestions = self.generate_interaction_suggestions(&profile).await?;

        Ok(PersonalizedResponse {
            content: accessible_content,
            format,
            confidence,
            adaptation_rationale: rationale,
            accessibility_features: self.get_applied_accessibility_features(&profile),
            suggestions,
        })
    }

    /// Get or create user profile
    async fn get_or_create_user_profile(&mut self, user_id: &str) -> Result<UserProfile> {
        if let Some(profile) = self.user_profiles.get(user_id) {
            Ok(profile.clone())
        } else {
            let new_profile = UserProfile {
                user_id: user_id.to_string(),
                expertise_level: ExpertiseLevel::Beginner,
                communication_style: CommunicationStyle {
                    formality: FormalityLevel::Professional,
                    detail_level: DetailLevel::Moderate,
                    explanation_style: ExplanationStyle::Conceptual,
                    pace: InteractionPace::Moderate,
                    feedback_preference: FeedbackStyle::Confirmatory,
                },
                preferred_formats: vec![ResponseFormat::Text, ResponseFormat::StructuredText],
                interests: Vec::new(),
                accessibility_needs: AccessibilityNeeds {
                    visual_impairment: false,
                    hearing_impairment: false,
                    motor_impairment: false,
                    cognitive_assistance: false,
                    screen_reader_compatible: false,
                    high_contrast: false,
                    font_size_preference: None,
                },
                learning_preferences: LearningPreferences {
                    learning_style: LearningStyle::Multimodal,
                    preferred_examples: ExampleType::RealWorld,
                    scaffolding_level: ScaffoldingLevel::Adaptive,
                    practice_preference: true,
                },
                interaction_history: InteractionHistory {
                    total_sessions: 0,
                    successful_queries: 0,
                    failed_queries: 0,
                    common_topics: Vec::new(),
                    preferred_query_types: Vec::new(),
                    avg_session_duration: Duration::from_secs(0),
                    last_active: chrono::Utc::now(),
                    satisfaction_scores: Vec::new(),
                },
                language_preferences: LanguagePreferences {
                    primary_language: "en".to_string(),
                    secondary_languages: Vec::new(),
                    technical_terminology: true,
                    localization: LocalizationSettings {
                        timezone: "UTC".to_string(),
                        date_format: "ISO8601".to_string(),
                        number_format: "en-US".to_string(),
                        currency: "USD".to_string(),
                    },
                },
            };

            self.user_profiles
                .insert(user_id.to_string(), new_profile.clone());
            Ok(new_profile)
        }
    }

    /// Update user model based on interaction
    async fn update_user_model(
        &mut self,
        user_id: &str,
        query: &str,
        context: &[Message],
    ) -> Result<()> {
        // Extract topics first to avoid borrowing conflicts
        let topics = self.extract_topics_from_query(query).await?;
        let detected_expertise = self
            .expertise_detector
            .detect_expertise_level(query, context)
            .await?;

        if let Some(profile) = self.user_profiles.get_mut(user_id) {
            // Update expertise level based on query complexity
            profile.expertise_level = detected_expertise;

            // Update interaction history
            profile.interaction_history.total_sessions += 1;
            profile.interaction_history.last_active = chrono::Utc::now();

            // Update topics and interests
            for topic in topics {
                if !profile.interests.contains(&topic) {
                    profile.interests.push(topic);
                }
            }
        }

        // Update common topics after releasing the mutable borrow
        if self.user_profiles.contains_key(user_id) {
            self.update_common_topics_for_user(user_id, query).await?;
        }

        Ok(())
    }

    /// Select optimal response format
    async fn select_optimal_format(
        &self,
        profile: &UserProfile,
        query: &str,
    ) -> Result<ResponseFormat> {
        // Analyze query intent to determine best format
        if query.to_lowercase().contains("table") || query.to_lowercase().contains("compare") {
            Ok(ResponseFormat::Tables)
        } else if query.to_lowercase().contains("code") || query.to_lowercase().contains("sparql") {
            Ok(ResponseFormat::Code)
        } else if query.to_lowercase().contains("steps") || query.to_lowercase().contains("how to")
        {
            Ok(ResponseFormat::BulletPoints)
        } else {
            // Use user's preferred format
            Ok(profile
                .preferred_formats
                .first()
                .unwrap_or(&ResponseFormat::Text)
                .clone())
        }
    }

    /// Calculate confidence in personalization
    async fn calculate_personalization_confidence(&self, profile: &UserProfile) -> Result<f32> {
        let mut confidence = 0.5; // Base confidence

        // Increase confidence based on interaction history
        if profile.interaction_history.total_sessions > 10 {
            confidence += 0.2;
        }

        // Increase confidence if we have satisfaction scores
        if !profile.interaction_history.satisfaction_scores.is_empty() {
            let avg_satisfaction: f32 = profile
                .interaction_history
                .satisfaction_scores
                .iter()
                .sum::<f32>()
                / profile.interaction_history.satisfaction_scores.len() as f32;
            confidence += (avg_satisfaction - 0.5) * 0.3;
        }

        // Increase confidence if we know user's expertise level
        match profile.expertise_level {
            ExpertiseLevel::Domain(_) => confidence += 0.2,
            ExpertiseLevel::Expert => confidence += 0.15,
            ExpertiseLevel::Advanced => confidence += 0.1,
            _ => {}
        }

        Ok(confidence.min(1.0))
    }

    /// Generate adaptation rationale
    async fn generate_adaptation_rationale(
        &self,
        profile: &UserProfile,
        _query: &str,
    ) -> Result<String> {
        let mut rationale = String::new();

        rationale.push_str(&format!(
            "Response adapted for {:?} expertise level",
            profile.expertise_level
        ));

        rationale.push_str(&format!(
            " with {:?} detail level",
            profile.communication_style.detail_level
        ));

        if profile.accessibility_needs.screen_reader_compatible {
            rationale.push_str(" and screen reader compatibility");
        }

        Ok(rationale)
    }

    /// Generate interaction suggestions
    async fn generate_interaction_suggestions(&self, profile: &UserProfile) -> Result<Vec<String>> {
        let mut suggestions = Vec::new();

        // Suggest based on expertise level
        match profile.expertise_level {
            ExpertiseLevel::Beginner => {
                suggestions.push("Try asking for step-by-step explanations".to_string());
                suggestions.push("Request examples to better understand concepts".to_string());
            }
            ExpertiseLevel::Expert => {
                suggestions.push("You can request more technical details".to_string());
                suggestions.push("Consider exploring advanced features".to_string());
            }
            _ => {}
        }

        // Suggest based on learning preferences
        if profile.learning_preferences.practice_preference {
            suggestions.push("Try practice questions to reinforce learning".to_string());
        }

        // Suggest based on preferred formats
        if profile
            .preferred_formats
            .contains(&ResponseFormat::Interactive)
        {
            suggestions.push("Request interactive demonstrations when available".to_string());
        }

        Ok(suggestions)
    }

    /// Get applied accessibility features
    fn get_applied_accessibility_features(&self, profile: &UserProfile) -> Vec<String> {
        let mut features = Vec::new();

        if profile.accessibility_needs.screen_reader_compatible {
            features.push("Screen reader optimized".to_string());
        }

        if profile.accessibility_needs.high_contrast {
            features.push("High contrast formatting".to_string());
        }

        if profile.accessibility_needs.cognitive_assistance {
            features.push("Simplified language".to_string());
        }

        features
    }

    // Helper methods
    async fn extract_topics_from_query(&self, query: &str) -> Result<Vec<String>> {
        // Simple keyword extraction - in practice would use NLP
        let keywords = vec!["sparql", "rdf", "ontology", "graph", "query", "data"];
        let query_lower = query.to_lowercase();

        Ok(keywords
            .into_iter()
            .filter(|&keyword| query_lower.contains(keyword))
            .map(|s| s.to_string())
            .collect())
    }

    async fn update_common_topics(&self, profile: &mut UserProfile, query: &str) -> Result<()> {
        let topics = self.extract_topics_from_query(query).await?;
        for topic in topics {
            if !profile.interaction_history.common_topics.contains(&topic) {
                profile.interaction_history.common_topics.push(topic);
            }
        }
        Ok(())
    }

    async fn update_common_topics_for_user(&mut self, user_id: &str, query: &str) -> Result<()> {
        let topics = self.extract_topics_from_query(query).await?;
        if let Some(profile) = self.user_profiles.get_mut(user_id) {
            for topic in topics {
                if !profile.interaction_history.common_topics.contains(&topic) {
                    profile.interaction_history.common_topics.push(topic);
                }
            }
        }
        Ok(())
    }
}

/// Expertise detection engine
pub struct ExpertiseDetector {
    complexity_keywords: HashMap<ExpertiseLevel, Vec<String>>,
}

impl ExpertiseDetector {
    fn new() -> Self {
        let mut complexity_keywords = HashMap::new();

        complexity_keywords.insert(
            ExpertiseLevel::Beginner,
            vec![
                "what is",
                "how to",
                "explain",
                "basic",
                "simple",
                "introduction",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
        );

        complexity_keywords.insert(
            ExpertiseLevel::Advanced,
            vec![
                "optimize",
                "performance",
                "advanced",
                "complex",
                "algorithm",
                "architecture",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
        );

        complexity_keywords.insert(
            ExpertiseLevel::Expert,
            vec![
                "ontology",
                "reasoning",
                "inference",
                "shacl",
                "federation",
                "distributed",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
        );

        Self {
            complexity_keywords,
        }
    }

    async fn detect_expertise_level(
        &self,
        query: &str,
        context: &[Message],
    ) -> Result<ExpertiseLevel> {
        let query_lower = query.to_lowercase();

        // Check for expert-level keywords
        if let Some(keywords) = self.complexity_keywords.get(&ExpertiseLevel::Expert) {
            for keyword in keywords {
                if query_lower.contains(keyword) {
                    return Ok(ExpertiseLevel::Expert);
                }
            }
        }

        // Check for advanced-level keywords
        if let Some(keywords) = self.complexity_keywords.get(&ExpertiseLevel::Advanced) {
            for keyword in keywords {
                if query_lower.contains(keyword) {
                    return Ok(ExpertiseLevel::Advanced);
                }
            }
        }

        // Check for beginner-level keywords
        if let Some(keywords) = self.complexity_keywords.get(&ExpertiseLevel::Beginner) {
            for keyword in keywords {
                if query_lower.contains(keyword) {
                    return Ok(ExpertiseLevel::Beginner);
                }
            }
        }

        // Analyze conversation context for expertise indicators
        let context_expertise = self.analyze_context_expertise(context).await?;

        Ok(context_expertise)
    }

    async fn analyze_context_expertise(&self, context: &[Message]) -> Result<ExpertiseLevel> {
        let mut expert_indicators = 0;
        let mut beginner_indicators = 0;

        for message in context {
            if let crate::types::MessageContent::Text(text) = &message.content {
                let text_lower = text.to_lowercase();

                if text_lower.contains("complex") || text_lower.contains("advanced") {
                    expert_indicators += 1;
                }
                if text_lower.contains("simple") || text_lower.contains("basic") {
                    beginner_indicators += 1;
                }
            }
        }

        if expert_indicators > beginner_indicators {
            Ok(ExpertiseLevel::Advanced)
        } else if beginner_indicators > expert_indicators {
            Ok(ExpertiseLevel::Beginner)
        } else {
            Ok(ExpertiseLevel::Intermediate)
        }
    }
}

/// Content adaptation engine
pub struct ContentAdapter {}

impl ContentAdapter {
    fn new() -> Self {
        Self {}
    }

    async fn adapt_content(
        &self,
        base_response: &str,
        profile: &UserProfile,
        _query: &str,
    ) -> Result<String> {
        let mut adapted_response = base_response.to_string();

        // Adapt based on detail level
        adapted_response = self
            .adjust_detail_level(&adapted_response, &profile.communication_style.detail_level)
            .await?;

        // Adapt based on formality
        adapted_response = self
            .adjust_formality(&adapted_response, &profile.communication_style.formality)
            .await?;

        // Adapt based on explanation style
        adapted_response = self
            .adjust_explanation_style(
                &adapted_response,
                &profile.communication_style.explanation_style,
            )
            .await?;

        // Adapt based on expertise level
        adapted_response = self
            .adjust_for_expertise(&adapted_response, &profile.expertise_level)
            .await?;

        Ok(adapted_response)
    }

    async fn adjust_detail_level(
        &self,
        content: &str,
        detail_level: &DetailLevel,
    ) -> Result<String> {
        match detail_level {
            DetailLevel::Brief => {
                // Summarize content to key points
                Ok(format!(
                    "**Summary**: {}",
                    content.chars().take(200).collect::<String>()
                ))
            }
            DetailLevel::Comprehensive => {
                // Add additional context and explanations
                Ok(format!("{content}\n\n**Additional Context**: This response provides comprehensive information on the topic. For more specific details, feel free to ask follow-up questions."))
            }
            _ => Ok(content.to_string()),
        }
    }

    async fn adjust_formality(&self, content: &str, formality: &FormalityLevel) -> Result<String> {
        match formality {
            FormalityLevel::Casual => {
                // Make language more conversational
                Ok(content
                    .replace("Furthermore", "Also")
                    .replace("Therefore", "So"))
            }
            FormalityLevel::Academic => {
                // Add more formal academic language
                Ok(format!(
                    "In accordance with semantic web principles, {content}"
                ))
            }
            _ => Ok(content.to_string()),
        }
    }

    async fn adjust_explanation_style(
        &self,
        content: &str,
        style: &ExplanationStyle,
    ) -> Result<String> {
        match style {
            ExplanationStyle::StepByStep => {
                // Structure as numbered steps
                let lines: Vec<&str> = content.split('.').collect();
                let mut stepped = String::new();
                for (i, line) in lines.iter().enumerate() {
                    if !line.trim().is_empty() {
                        stepped.push_str(&format!("{}. {}\n", i + 1, line.trim()));
                    }
                }
                Ok(stepped)
            }
            ExplanationStyle::ExampleDriven => {
                // Add examples
                Ok(format!("{content}\n\n**Example**: For instance, when querying for person names, you might use: SELECT ?name WHERE {{ ?person foaf:name ?name }}"))
            }
            _ => Ok(content.to_string()),
        }
    }

    async fn adjust_for_expertise(
        &self,
        content: &str,
        expertise: &ExpertiseLevel,
    ) -> Result<String> {
        match expertise {
            ExpertiseLevel::Beginner => {
                // Simplify technical terms
                Ok(content
                    .replace("SPARQL", "SPARQL (the query language for semantic data)")
                    .replace("RDF", "RDF (Resource Description Framework)")
                    .replace(
                        "ontology",
                        "ontology (a formal representation of knowledge)",
                    ))
            }
            ExpertiseLevel::Expert => {
                // Add technical depth
                Ok(format!("{content}\n\n**Technical Note**: For advanced optimization, consider using query federation and distributed processing techniques."))
            }
            _ => Ok(content.to_string()),
        }
    }
}

/// Accessibility enhancement engine
pub struct AccessibilityEnhancer {}

impl AccessibilityEnhancer {
    fn new() -> Self {
        Self {}
    }

    async fn enhance_content(
        &self,
        content: &str,
        accessibility_needs: &AccessibilityNeeds,
    ) -> Result<String> {
        let mut enhanced_content = content.to_string();

        if accessibility_needs.screen_reader_compatible {
            enhanced_content = self.make_screen_reader_friendly(&enhanced_content).await?;
        }

        if accessibility_needs.cognitive_assistance {
            enhanced_content = self
                .simplify_for_cognitive_assistance(&enhanced_content)
                .await?;
        }

        if accessibility_needs.visual_impairment {
            enhanced_content = self
                .enhance_for_visual_impairment(&enhanced_content)
                .await?;
        }

        Ok(enhanced_content)
    }

    async fn make_screen_reader_friendly(&self, content: &str) -> Result<String> {
        // Add structure markers and alt text descriptions
        let mut enhanced = content.to_string();

        // Add heading markers
        enhanced = enhanced.replace("##", "[Heading]");
        enhanced = enhanced.replace("**", "[Emphasis]");

        Ok(enhanced)
    }

    async fn simplify_for_cognitive_assistance(&self, content: &str) -> Result<String> {
        // Break into shorter sentences and simpler language
        let sentences: Vec<&str> = content.split('.').collect();
        let simplified: Vec<String> = sentences
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| format!("{s}."))
            .collect();

        Ok(simplified.join("\n\n"))
    }

    async fn enhance_for_visual_impairment(&self, content: &str) -> Result<String> {
        // Add descriptive text for visual elements
        Ok(format!("[Text content] {content}"))
    }
}

impl Default for ChatPersonalizer {
    fn default() -> Self {
        Self::new()
    }
}
