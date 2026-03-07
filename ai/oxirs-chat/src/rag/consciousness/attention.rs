//! Attention Mechanism
//!
//! Implements weighted focus distribution and attention allocation.

use anyhow::Result;
use std::collections::{HashMap, VecDeque};

use super::super::*;

#[derive(Debug, Clone)]
pub struct AttentionMechanism {
    attention_weights: HashMap<String, f64>,
    focus_history: VecDeque<AttentionSnapshot>,
    attention_decay_rate: f64,
    max_attention_points: usize,
}

impl AttentionMechanism {
    pub fn new() -> Result<Self> {
        Ok(Self {
            attention_weights: HashMap::new(),
            focus_history: VecDeque::with_capacity(50),
            attention_decay_rate: 0.95,
            max_attention_points: 20,
        })
    }

    pub fn allocate_attention(
        &mut self,
        query: &str,
        context: &AssembledContext,
        neural_activation: &NeuralActivation,
    ) -> Result<AttentionAllocation> {
        // Extract attention targets from query and context
        let mut targets = self.extract_attention_targets(query, context)?;

        // Apply neural activation influence
        for target in &mut targets {
            if let Some(neural_weight) = neural_activation.get_concept_activation(&target.concept) {
                target.base_weight *= 1.0 + neural_weight;
            }
        }

        // Normalize weights
        let total_weight: f64 = targets.iter().map(|t| t.base_weight).sum();
        if total_weight > 0.0 {
            for target in &mut targets {
                target.normalized_weight = target.base_weight / total_weight;
            }
        }

        // Update internal attention state
        self.update_attention_state(&targets)?;

        Ok(AttentionAllocation {
            targets,
            total_attention_units: self.max_attention_points,
            allocation_entropy: self.calculate_entropy_internal()?,
            temporal_stability: self.calculate_temporal_stability()?,
        })
    }

    pub fn get_current_weights(&self) -> Result<HashMap<String, f64>> {
        Ok(self.attention_weights.clone())
    }

    pub fn get_health_score(&self) -> Result<f64> {
        let entropy = self.calculate_entropy_internal()?;
        let stability = self.calculate_temporal_stability()?;
        Ok((entropy + stability) / 2.0)
    }

    pub fn calculate_attention_entropy(&self) -> Result<f64> {
        self.calculate_entropy_internal()
    }

    fn extract_attention_targets(
        &self,
        query: &str,
        context: &AssembledContext,
    ) -> Result<Vec<AttentionTarget>> {
        let mut targets = Vec::new();

        // Extract from query
        let words: Vec<&str> = query.split_whitespace().collect();
        for word in words {
            if word.len() > 3 {
                targets.push(AttentionTarget {
                    concept: word.to_lowercase(),
                    base_weight: 1.0,
                    normalized_weight: 0.0,
                    attention_type: AttentionType::Lexical,
                });
            }
        }

        // Extract from context entities
        for entity in &context.extracted_entities {
            targets.push(AttentionTarget {
                concept: entity.text.clone(),
                base_weight: entity.confidence as f64 * 2.0,
                normalized_weight: 0.0,
                attention_type: AttentionType::Entity,
            });
        }

        // Limit and sort by weight
        targets.sort_by(|a, b| {
            b.base_weight
                .partial_cmp(&a.base_weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        targets.truncate(self.max_attention_points);

        Ok(targets)
    }

    fn update_attention_state(&mut self, targets: &[AttentionTarget]) -> Result<()> {
        // Decay existing weights
        for weight in self.attention_weights.values_mut() {
            *weight *= self.attention_decay_rate;
        }

        // Add new attention
        for target in targets {
            let entry = self
                .attention_weights
                .entry(target.concept.clone())
                .or_insert(0.0);
            *entry += target.normalized_weight;
        }

        // Remove very low weights
        self.attention_weights
            .retain(|_, &mut weight| weight > 0.01);

        Ok(())
    }

    fn calculate_entropy_internal(&self) -> Result<f64> {
        if self.attention_weights.is_empty() {
            return Ok(0.0);
        }

        let total: f64 = self.attention_weights.values().sum();
        if total <= 0.0 {
            return Ok(0.0);
        }

        let entropy = self
            .attention_weights
            .values()
            .map(|&weight| {
                let p = weight / total;
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        Ok(entropy / (self.attention_weights.len() as f64).log2())
    }

    fn calculate_temporal_stability(&self) -> Result<f64> {
        if self.focus_history.len() < 2 {
            return Ok(1.0);
        }

        // Calculate stability as inverse of variance in attention patterns
        let recent_snapshots: Vec<_> = self.focus_history.iter().rev().take(5).collect();
        if recent_snapshots.len() < 2 {
            return Ok(1.0);
        }

        // Simple stability metric: how much attention patterns change over time
        let mut changes = 0.0;
        for i in 1..recent_snapshots.len() {
            changes +=
                self.calculate_snapshot_difference(recent_snapshots[i - 1], recent_snapshots[i])?;
        }

        let avg_change = changes / (recent_snapshots.len() - 1) as f64;
        Ok((1.0 - avg_change).max(0.0))
    }

    fn calculate_snapshot_difference(
        &self,
        snap1: &AttentionSnapshot,
        snap2: &AttentionSnapshot,
    ) -> Result<f64> {
        // Simple difference calculation
        let diff = (snap1.primary_focus_strength - snap2.primary_focus_strength).abs()
            + (snap1.attention_dispersion - snap2.attention_dispersion).abs();
        Ok(diff / 2.0)
    }
}
