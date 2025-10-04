//! Dream State Processor
//!
//! Memory consolidation during dream-like states.

use anyhow::Result;
use std::collections::VecDeque;

use super::super::*;

pub struct DreamStateProcessor {
    is_dreaming: bool,
    dream_intensity: f64,
    memory_fragments: Vec<MemoryFragment>,
    dream_scenarios: VecDeque<DreamScenario>,
    consolidation_metrics: ConsolidationMetrics,
    creative_insights: Vec<CreativeInsight>,
}

#[derive(Debug, Clone)]
pub struct MemoryFragment {
    content: String,
    emotional_weight: f64,
    temporal_marker: std::time::Instant,
    consolidation_priority: f64,
    associations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DreamScenario {
    narrative: String,
    participating_memories: Vec<usize>, // indices into memory_fragments
    emotional_tone: EmotionalTone,
    insight_potential: f64,
    symbolic_elements: Vec<String>,
}

impl Default for DreamStateProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl DreamStateProcessor {
    pub fn new() -> Self {
        Self {
            is_dreaming: false,
            dream_intensity: 0.0,
            memory_fragments: Vec::new(),
            dream_scenarios: VecDeque::new(),
            consolidation_metrics: ConsolidationMetrics::new(),
            creative_insights: Vec::new(),
        }
    }

    /// Enter dream state for memory consolidation
    pub fn enter_dream_state(&mut self, idle_duration: Duration) -> Result<()> {
        if idle_duration < Duration::from_secs(30) {
            return Ok(()); // Not enough idle time for dreaming
        }

        self.is_dreaming = true;
        self.dream_intensity = (idle_duration.as_secs() as f64 / 300.0).min(1.0); // Max intensity after 5 minutes

        debug!(
            "Entering dream state with intensity: {:.2}",
            self.dream_intensity
        );

        // Initiate memory consolidation
        self.consolidate_memories()?;

        // Generate dream scenarios
        self.generate_dream_scenarios()?;

        // Extract creative insights
        self.extract_creative_insights()?;

        Ok(())
    }

    /// Consolidate memories during dream state
    fn consolidate_memories(&mut self) -> Result<()> {
        // Sort memory fragments by consolidation priority
        self.memory_fragments.sort_by(|a, b| {
            b.consolidation_priority
                .partial_cmp(&a.consolidation_priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let consolidation_count =
            (self.memory_fragments.len() as f64 * self.dream_intensity * 0.3) as usize;

        // Pre-compute similar fragments for each content
        let similar_fragments: Vec<Option<String>> = self
            .memory_fragments
            .iter()
            .take(consolidation_count)
            .map(|f| {
                self.find_similar_memory(&f.content)
                    .map(|sim| sim.content.clone())
            })
            .collect();

        for (i, fragment) in self
            .memory_fragments
            .iter_mut()
            .take(consolidation_count)
            .enumerate()
        {
            // Strengthen important memories
            fragment.consolidation_priority *= 1.1;

            // Create new associations
            if let Some(similar_content) = &similar_fragments[i] {
                fragment.associations.push(similar_content.clone());
            }
        }

        self.consolidation_metrics
            .update(consolidation_count, self.dream_intensity);
        debug!("Consolidated {} memory fragments", consolidation_count);

        Ok(())
    }

    /// Generate creative dream scenarios
    fn generate_dream_scenarios(&mut self) -> Result<()> {
        let scenario_count = (self.dream_intensity * 5.0) as usize;

        for _ in 0..scenario_count {
            // Select random memory fragments
            let count = fastrand::usize(2..=5.min(self.memory_fragments.len()));
            let mut participating_memories = Vec::new();
            for _ in 0..count {
                let idx = fastrand::usize(..self.memory_fragments.len());
                if !participating_memories.contains(&idx) {
                    participating_memories.push(idx);
                }
            }

            let scenario = DreamScenario {
                narrative: self.weave_narrative(&participating_memories)?,
                participating_memories: participating_memories.clone(),
                emotional_tone: self.determine_emotional_tone(&participating_memories)?,
                insight_potential: fastrand::f64(),
                symbolic_elements: self.extract_symbolic_elements(&participating_memories)?,
            };

            self.dream_scenarios.push_back(scenario);
        }

        // Keep scenarios bounded
        while self.dream_scenarios.len() > 20 {
            self.dream_scenarios.pop_front();
        }

        debug!("Generated {} dream scenarios", scenario_count);
        Ok(())
    }

    /// Extract creative insights from dream processing
    fn extract_creative_insights(&mut self) -> Result<()> {
        for scenario in &self.dream_scenarios {
            if scenario.insight_potential > 0.7 {
                let insight = CreativeInsight {
                    insight_content: format!("Dream insight: {}", scenario.narrative),
                    novelty_score: scenario.insight_potential,
                    relevance_score: 0.8, // Default relevance for dream insights
                    confidence: scenario.insight_potential * 0.9, // High confidence for high potential insights
                };

                self.creative_insights.push(insight);
            }
        }

        debug!(
            "Extracted {} creative insights",
            self.creative_insights.len()
        );
        Ok(())
    }

    /// Find similar memory to given content
    fn find_similar_memory(&self, content: &str) -> Option<&MemoryFragment> {
        self.memory_fragments.iter().find(|fragment| {
            fragment
                .content
                .to_lowercase()
                .contains(&content.to_lowercase())
                || content
                    .to_lowercase()
                    .contains(&fragment.content.to_lowercase())
        })
    }

    /// Weave narrative from memory fragments
    fn weave_narrative(&self, memory_indices: &[usize]) -> Result<String> {
        let contents: Vec<String> = memory_indices
            .iter()
            .filter_map(|&idx| self.memory_fragments.get(idx))
            .map(|fragment| fragment.content.clone())
            .collect();

        if contents.is_empty() {
            return Ok("Empty dream narrative".to_string());
        }

        Ok(format!("Dream narrative: {}", contents.join(" -> ")))
    }

    /// Determine emotional tone of memories
    fn determine_emotional_tone(&self, memory_indices: &[usize]) -> Result<EmotionalTone> {
        let positive_keywords = ["happy", "joy", "love", "success", "achievement"];
        let negative_keywords = ["sad", "anger", "fear", "failure", "loss"];

        let mut positive_score = 0.0;
        let mut negative_score = 0.0;

        for &idx in memory_indices {
            if let Some(fragment) = self.memory_fragments.get(idx) {
                let content_lower = fragment.content.to_lowercase();
                positive_score += positive_keywords
                    .iter()
                    .filter(|&keyword| content_lower.contains(keyword))
                    .count() as f64;
                negative_score += negative_keywords
                    .iter()
                    .filter(|&keyword| content_lower.contains(keyword))
                    .count() as f64;
            }
        }

        if positive_score > negative_score * 1.2 {
            Ok(EmotionalTone::Positive)
        } else if negative_score > positive_score * 1.2 {
            Ok(EmotionalTone::Negative)
        } else if positive_score > 0.0 && negative_score > 0.0 {
            Ok(EmotionalTone::Mixed {
                positive_weight: positive_score / (positive_score + negative_score),
                negative_weight: negative_score / (positive_score + negative_score),
            })
        } else {
            Ok(EmotionalTone::Neutral)
        }
    }

    /// Extract symbolic elements from memories
    fn extract_symbolic_elements(&self, memory_indices: &[usize]) -> Result<Vec<String>> {
        let mut elements = Vec::new();
        let symbolic_keywords = [
            "light", "dark", "water", "fire", "earth", "air", "journey", "path", "door", "key",
        ];

        for &idx in memory_indices {
            if let Some(fragment) = self.memory_fragments.get(idx) {
                let content_lower = fragment.content.to_lowercase();
                for &keyword in &symbolic_keywords {
                    if content_lower.contains(keyword) && !elements.contains(&keyword.to_string()) {
                        elements.push(keyword.to_string());
                    }
                }
            }
        }

        if elements.is_empty() {
            elements.push("abstract".to_string());
        }

        Ok(elements)
    }
}
