//! Advanced Emotional State
//!
//! Implements emotional state tracking with valence/arousal dynamics.

use anyhow::Result;
use std::collections::VecDeque;

use super::super::*;

#[derive(Debug, Clone)]
pub struct AdvancedEmotionalState {
    // Core emotional dimensions (public for access)
    // Core emotional dimensions
    pub valence: f64, // -1.0 (negative) to 1.0 (positive)
    arousal: f64,     // 0.0 (calm) to 1.0 (excited)
    dominance: f64,   // 0.0 (submissive) to 1.0 (dominant)

    // Advanced emotional features
    emotional_momentum: f64,
    emotional_complexity: f64,
    emotional_stability: f64,
    emotional_history: VecDeque<EmotionalSnapshot>,

    // Emotional regulation
    regulation_strategies: Vec<RegulationStrategy>,
    current_regulation: Option<RegulationStrategy>,
}

impl AdvancedEmotionalState {
    pub fn neutral() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.5,
            dominance: 0.5,
            emotional_momentum: 0.0,
            emotional_complexity: 0.0,
            emotional_stability: 1.0,
            emotional_history: VecDeque::with_capacity(20),
            regulation_strategies: vec![
                RegulationStrategy::Reappraisal,
                RegulationStrategy::Suppression,
                RegulationStrategy::Distraction,
            ],
            current_regulation: None,
        }
    }

    pub fn process_emotional_content(
        &mut self,
        query: &str,
        context: &AssembledContext,
    ) -> Result<EmotionalResponse> {
        // Analyze emotional content in query
        let query_emotion = self.analyze_query_emotion(query)?;

        // Analyze context emotional valence
        let context_emotion = self.analyze_context_emotion(context)?;

        // Update emotional state
        self.update_emotional_state(&query_emotion, &context_emotion)?;

        // Apply emotional regulation if needed
        self.apply_emotional_regulation()?;

        // Record emotional snapshot
        self.record_emotional_snapshot()?;

        Ok(EmotionalResponse {
            current_valence: self.valence,
            current_arousal: self.arousal,
            current_dominance: self.dominance,
            emotional_intensity: self.calculate_emotional_intensity()?,
            emotional_coherence: self.calculate_emotional_coherence()?,
            regulation_applied: self.current_regulation,
        })
    }

    pub fn get_current_state(&self) -> Result<EmotionalStateSnapshot> {
        Ok(EmotionalStateSnapshot {
            valence: self.valence,
            arousal: self.arousal,
            dominance: self.dominance,
            momentum: self.emotional_momentum,
            complexity: self.emotional_complexity,
            stability: self.emotional_stability,
        })
    }

    pub fn get_stability_score(&self) -> Result<f64> {
        Ok(self.emotional_stability)
    }

    pub fn calculate_resonance(&self, query: &str) -> f64 {
        // Calculate emotional resonance based on query content and current state
        let query_lower = query.to_lowercase();

        // Check for emotional keywords
        let emotional_words = [
            "feel",
            "feeling",
            "emotion",
            "emotional",
            "mood",
            "happy",
            "sad",
            "angry",
            "excited",
            "worried",
            "anxious",
            "calm",
            "stressed",
        ];

        let emotion_score = emotional_words
            .iter()
            .map(|&word| if query_lower.contains(word) { 0.2 } else { 0.0 })
            .sum::<f64>();

        // Factor in current emotional state
        let state_resonance = (self.valence.abs() + self.arousal + self.dominance) / 3.0;

        // Combine scores and normalize
        (emotion_score + state_resonance * 0.5).min(1.0)
    }

    fn analyze_query_emotion(&self, query: &str) -> Result<EmotionalAnalysis> {
        // Emotional word analysis
        let positive_words = [
            "happy",
            "excited",
            "pleased",
            "satisfied",
            "confident",
            "optimistic",
        ];
        let negative_words = [
            "sad",
            "angry",
            "frustrated",
            "worried",
            "anxious",
            "disappointed",
        ];
        let high_arousal_words = ["excited", "thrilled", "panicked", "furious", "ecstatic"];
        let low_arousal_words = ["calm", "peaceful", "relaxed", "serene", "tranquil"];

        let query_lower = query.to_lowercase();

        let positive_count = positive_words
            .iter()
            .filter(|w| query_lower.contains(*w))
            .count();
        let negative_count = negative_words
            .iter()
            .filter(|w| query_lower.contains(*w))
            .count();
        let high_arousal_count = high_arousal_words
            .iter()
            .filter(|w| query_lower.contains(*w))
            .count();
        let low_arousal_count = low_arousal_words
            .iter()
            .filter(|w| query_lower.contains(*w))
            .count();

        let valence_score = if positive_count + negative_count > 0 {
            (positive_count as f64 - negative_count as f64)
                / (positive_count + negative_count) as f64
        } else {
            0.0
        };

        let arousal_score = if high_arousal_count + low_arousal_count > 0 {
            (high_arousal_count as f64 - low_arousal_count as f64)
                / (high_arousal_count + low_arousal_count) as f64
                * 0.5
                + 0.5
        } else {
            0.5
        };

        Ok(EmotionalAnalysis {
            valence: valence_score,
            arousal: arousal_score,
            dominance: 0.5, // Default neutral dominance
            confidence: ((positive_count + negative_count + high_arousal_count + low_arousal_count)
                as f64
                / 4.0)
                .min(1.0),
        })
    }

    fn analyze_context_emotion(&self, context: &AssembledContext) -> Result<EmotionalAnalysis> {
        // Analyze emotional content in context
        let mut total_valence = 0.0;
        let mut total_arousal = 0.5;
        let mut confidence = 0.0;

        // Analyze semantic results for emotional content
        if !context.semantic_results.is_empty() {
            for result in &context.semantic_results {
                let content = result.triple.object().to_string();
                let emotion = self.analyze_query_emotion(&content)?;
                total_valence += emotion.valence * result.score as f64;
                total_arousal += emotion.arousal * result.score as f64;
                confidence += emotion.confidence * result.score as f64;
            }

            let result_count = context.semantic_results.len() as f64;
            total_valence /= result_count;
            total_arousal /= result_count;
            confidence /= result_count;
        }

        Ok(EmotionalAnalysis {
            valence: total_valence,
            arousal: total_arousal,
            dominance: 0.5,
            confidence,
        })
    }

    fn update_emotional_state(
        &mut self,
        query_emotion: &EmotionalAnalysis,
        context_emotion: &EmotionalAnalysis,
    ) -> Result<()> {
        // Calculate momentum
        let valence_change =
            (query_emotion.valence * 0.6 + context_emotion.valence * 0.4) - self.valence;
        let arousal_change =
            (query_emotion.arousal * 0.6 + context_emotion.arousal * 0.4) - self.arousal;

        self.emotional_momentum = (valence_change.abs() + arousal_change.abs()) / 2.0;

        // Update emotional state with dampening
        let dampening_factor = 0.3;
        self.valence += valence_change * dampening_factor;
        self.arousal += arousal_change * dampening_factor;

        // Clamp values
        self.valence = self.valence.clamp(-1.0, 1.0);
        self.arousal = self.arousal.clamp(0.0, 1.0);

        // Update emotional complexity and stability
        self.emotional_complexity = self.calculate_emotional_complexity()?;
        self.update_emotional_stability()?;

        Ok(())
    }

    fn apply_emotional_regulation(&mut self) -> Result<()> {
        // Apply regulation if emotional intensity is too high
        let intensity = self.calculate_emotional_intensity()?;

        if intensity > 0.8 {
            // Choose regulation strategy
            let strategy = if self.valence < -0.5 {
                RegulationStrategy::Reappraisal
            } else if self.arousal > 0.8 {
                RegulationStrategy::Suppression
            } else {
                RegulationStrategy::Distraction
            };

            // Apply regulation
            match strategy {
                RegulationStrategy::Reappraisal => {
                    self.valence *= 0.8; // Reduce negative valence
                    self.arousal *= 0.9; // Slightly reduce arousal
                }
                RegulationStrategy::Suppression => {
                    self.arousal *= 0.7; // Significantly reduce arousal
                }
                RegulationStrategy::Distraction => {
                    self.valence *= 0.9; // Slightly reduce valence
                    self.arousal *= 0.85; // Moderately reduce arousal
                }
            }

            self.current_regulation = Some(strategy);
        } else {
            self.current_regulation = None;
        }

        Ok(())
    }

    fn calculate_emotional_intensity(&self) -> Result<f64> {
        Ok((self.valence.abs() + self.arousal).min(1.0))
    }

    fn calculate_emotional_coherence(&self) -> Result<f64> {
        if self.emotional_history.len() < 2 {
            return Ok(1.0);
        }

        // Calculate coherence as inverse of emotional volatility
        let recent_snapshots: Vec<_> = self.emotional_history.iter().rev().take(5).collect();
        if recent_snapshots.len() < 2 {
            return Ok(1.0);
        }

        let mut volatility = 0.0;
        for i in 1..recent_snapshots.len() {
            let diff = ((recent_snapshots[i - 1].valence - recent_snapshots[i].valence).abs()
                + (recent_snapshots[i - 1].arousal - recent_snapshots[i].arousal).abs())
                / 2.0;
            volatility += diff;
        }

        let avg_volatility = volatility / (recent_snapshots.len() - 1) as f64;
        Ok((1.0 - avg_volatility).max(0.0))
    }

    fn calculate_emotional_complexity(&self) -> Result<f64> {
        // Complexity based on how much emotional state deviates from neutral
        let valence_complexity = self.valence.abs();
        let arousal_complexity = (self.arousal - 0.5).abs() * 2.0;
        let dominance_complexity = (self.dominance - 0.5).abs() * 2.0;

        Ok((valence_complexity + arousal_complexity + dominance_complexity) / 3.0)
    }

    fn update_emotional_stability(&mut self) -> Result<()> {
        // Stability decreases with high momentum and complexity
        let momentum_impact = (1.0 - self.emotional_momentum).max(0.0);
        let complexity_impact = (1.0 - self.emotional_complexity).max(0.0);

        self.emotional_stability = (momentum_impact + complexity_impact) / 2.0;
        Ok(())
    }

    fn record_emotional_snapshot(&mut self) -> Result<()> {
        let snapshot = EmotionalSnapshot {
            timestamp: Utc::now(),
            valence: self.valence,
            arousal: self.arousal,
            dominance: self.dominance,
            intensity: self.calculate_emotional_intensity()?,
            complexity: self.emotional_complexity,
        };

        self.emotional_history.push_back(snapshot);

        if self.emotional_history.len() > 20 {
            self.emotional_history.pop_front();
        }

        Ok(())
    }
}
