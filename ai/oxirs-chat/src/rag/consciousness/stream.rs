//! Consciousness Stream
//!
//! Experience tracking and consciousness stream management.

use std::collections::VecDeque;

use super::super::*;

#[derive(Debug, Clone)]
pub struct ConsciousnessStream {
    experiences: VecDeque<ConsciousnessExperience>,
    stream_coherence: f64,
    temporal_binding: TemporalBinding,
}

impl Default for ConsciousnessStream {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsciousnessStream {
    pub fn new() -> Self {
        Self {
            experiences: VecDeque::with_capacity(100),
            stream_coherence: 1.0,
            temporal_binding: TemporalBinding::new(),
        }
    }

    pub fn add_experience(
        &mut self,
        query: &str,
        context: &AssembledContext,
        neural_activation: &NeuralActivation,
    ) -> Result<()> {
        let experience = ConsciousnessExperience {
            timestamp: Utc::now(),
            query_content: query.to_string(),
            context_summary: format!("Context with {} results", context.semantic_results.len()),
            neural_activation: neural_activation.overall_activation,
            consciousness_level: neural_activation.consciousness_relevance,
        };

        self.experiences.push_back(experience);

        if self.experiences.len() > 100 {
            self.experiences.pop_front();
        }

        self.update_stream_coherence()?;
        Ok(())
    }

    pub fn get_awareness_influence(&self) -> Result<f64> {
        if self.experiences.is_empty() {
            return Ok(0.5);
        }

        let recent_experiences: Vec<_> = self.experiences.iter().rev().take(5).collect();
        let avg_consciousness = recent_experiences
            .iter()
            .map(|e| e.consciousness_level)
            .sum::<f64>()
            / recent_experiences.len() as f64;

        Ok(avg_consciousness)
    }

    pub fn calculate_coherence(&self) -> Result<f64> {
        Ok(self.stream_coherence)
    }

    pub fn get_recent_context(&self, count: usize) -> Vec<ConsciousnessExperience> {
        self.experiences.iter().rev().take(count).cloned().collect()
    }

    fn update_stream_coherence(&mut self) -> Result<()> {
        if self.experiences.len() < 2 {
            self.stream_coherence = 1.0;
            return Ok(());
        }

        let recent: Vec<_> = self.experiences.iter().rev().take(10).collect();
        if recent.len() < 2 {
            self.stream_coherence = 1.0;
            return Ok(());
        }

        let mut coherence_sum = 0.0;
        for i in 1..recent.len() {
            let coherence = self.calculate_experience_coherence(recent[i - 1], recent[i])?;
            coherence_sum += coherence;
        }

        self.stream_coherence = coherence_sum / (recent.len() - 1) as f64;
        Ok(())
    }

    fn calculate_experience_coherence(
        &self,
        exp1: &ConsciousnessExperience,
        exp2: &ConsciousnessExperience,
    ) -> Result<f64> {
        let time_diff = (exp1.timestamp - exp2.timestamp).num_seconds().abs() as f64;
        let time_coherence = (1.0 / (1.0 + time_diff / 60.0)).max(0.1); // Decay over minutes

        let neural_diff = (exp1.neural_activation - exp2.neural_activation).abs();
        let neural_coherence = 1.0 - neural_diff;

        let consciousness_diff = (exp1.consciousness_level - exp2.consciousness_level).abs();
        let consciousness_coherence = 1.0 - consciousness_diff;

        Ok((time_coherence + neural_coherence + consciousness_coherence) / 3.0)
    }
}
