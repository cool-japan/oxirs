//! Consciousness State Machine
//!
//! Dynamic state transitions for adaptive consciousness behavior.

use anyhow::Result;
use std::collections::{HashMap, VecDeque};

use super::super::*;

pub struct ConsciousnessStateMachine {
    current_state: ConsciousnessState,
    state_history: VecDeque<StateTransition>,
    transition_rules: HashMap<ConsciousnessState, Vec<TransitionRule>>,
    state_metrics: StateMetrics,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConsciousnessState {
    /// Relaxed state for routine processing
    Baseline,
    /// Heightened awareness for complex queries
    Focused,
    /// Deep contemplation for philosophical questions
    Contemplative,
    /// Creative mode for synthesis and innovation
    Creative,
    /// Analytical mode for logical reasoning
    Analytical,
    /// Empathetic mode for emotional understanding
    Empathetic,
    /// Memory consolidation during idle periods
    Consolidating,
    /// Integration of multiple perspectives
    Integrative,
}

#[derive(Debug, Clone)]
pub struct StateTransition {
    from_state: ConsciousnessState,
    to_state: ConsciousnessState,
    trigger: TransitionTrigger,
    timestamp: std::time::Instant,
    success_probability: f64,
}

#[derive(Debug, Clone)]
pub enum TransitionTrigger {
    QueryComplexity(f64),
    EmotionalIntensity(f64),
    MemoryPressure(f64),
    AttentionShift(String),
    TimeBased(Duration),
    ExternalStimulus(String),
}

impl ConsciousnessStateMachine {
    pub fn new() -> Result<Self> {
        let mut transition_rules = HashMap::new();

        // Define state transition rules
        transition_rules.insert(
            ConsciousnessState::Baseline,
            vec![
                TransitionRule::new(
                    ConsciousnessState::Focused,
                    TransitionCondition::QueryComplexity(0.7),
                    0.8,
                ),
                TransitionRule::new(
                    ConsciousnessState::Creative,
                    TransitionCondition::KeywordPresence(vec!["creative", "innovative", "design"]),
                    0.7,
                ),
                TransitionRule::new(
                    ConsciousnessState::Empathetic,
                    TransitionCondition::EmotionalContent(0.6),
                    0.75,
                ),
            ],
        );

        transition_rules.insert(
            ConsciousnessState::Focused,
            vec![
                TransitionRule::new(
                    ConsciousnessState::Analytical,
                    TransitionCondition::LogicalPattern,
                    0.8,
                ),
                TransitionRule::new(
                    ConsciousnessState::Contemplative,
                    TransitionCondition::PhilosophicalContent,
                    0.7,
                ),
                TransitionRule::new(
                    ConsciousnessState::Baseline,
                    TransitionCondition::TimeElapsed(Duration::from_secs(10 * 60)),
                    0.6,
                ),
            ],
        );

        Ok(Self {
            current_state: ConsciousnessState::Baseline,
            state_history: VecDeque::new(),
            transition_rules,
            state_metrics: StateMetrics::new(),
        })
    }

    /// Evaluate and potentially transition consciousness state
    pub fn evaluate_transition(
        &mut self,
        query: &str,
        context: &AssembledContext,
    ) -> Result<Option<StateTransition>> {
        // Clone current state to avoid holding a borrow
        let current_state = self.current_state.clone();

        // Get and clone rules before the mutable borrow in the loop
        let rules_option = self.transition_rules.get(&current_state).cloned();
        let current_rules = rules_option
            .ok_or_else(|| anyhow::anyhow!("No transition rules for current state"))?;

        for rule in current_rules {
            if self.evaluate_condition(&rule.condition, query, context)? {
                let transition = StateTransition {
                    from_state: self.current_state.clone(),
                    to_state: rule.target_state.clone(),
                    trigger: self.identify_trigger(query, context)?,
                    timestamp: std::time::Instant::now(),
                    success_probability: rule.probability,
                };

                // Apply state transition
                self.current_state = rule.target_state.clone();
                self.state_history.push_back(transition.clone());

                // Keep history bounded
                if self.state_history.len() > 50 {
                    self.state_history.pop_front();
                }

                debug!(
                    "Consciousness state transition: {:?} -> {:?}",
                    transition.from_state, transition.to_state
                );

                return Ok(Some(transition));
            }
        }

        Ok(None)
    }

    /// Get state-specific processing parameters
    pub fn get_state_parameters(&self) -> StateProcessingParameters {
        match self.current_state {
            ConsciousnessState::Baseline => StateProcessingParameters {
                attention_focus: 0.5,
                emotional_sensitivity: 0.5,
                creativity_boost: 0.3,
                analytical_depth: 0.5,
                memory_consolidation: 0.2,
            },
            ConsciousnessState::Focused => StateProcessingParameters {
                attention_focus: 0.9,
                emotional_sensitivity: 0.3,
                creativity_boost: 0.4,
                analytical_depth: 0.8,
                memory_consolidation: 0.3,
            },
            ConsciousnessState::Creative => StateProcessingParameters {
                attention_focus: 0.6,
                emotional_sensitivity: 0.7,
                creativity_boost: 0.9,
                analytical_depth: 0.4,
                memory_consolidation: 0.5,
            },
            ConsciousnessState::Empathetic => StateProcessingParameters {
                attention_focus: 0.7,
                emotional_sensitivity: 0.9,
                creativity_boost: 0.6,
                analytical_depth: 0.5,
                memory_consolidation: 0.4,
            },
            // Add more state parameters as needed
            _ => StateProcessingParameters::default(),
        }
    }

    /// Evaluate transition condition
    fn evaluate_condition(
        &mut self,
        condition: &TransitionCondition,
        query: &str,
        _context: &AssembledContext,
    ) -> Result<bool> {
        match condition {
            TransitionCondition::QueryComplexity(threshold) => {
                let complexity = self.calculate_query_complexity(query)?;
                Ok(complexity >= *threshold)
            }
            TransitionCondition::EmotionalContent(threshold) => {
                let emotional_score = self.calculate_emotional_content(query)?;
                Ok(emotional_score >= *threshold)
            }
            TransitionCondition::KeywordPresence(keywords) => {
                let query_lower = query.to_lowercase();
                Ok(keywords.iter().any(|keyword| query_lower.contains(keyword)))
            }
            TransitionCondition::LogicalPattern => Ok(query.contains("because")
                || query.contains("therefore")
                || query.contains("thus")),
            TransitionCondition::PhilosophicalContent => {
                let philosophical_keywords = [
                    "meaning",
                    "purpose",
                    "existence",
                    "consciousness",
                    "reality",
                ];
                let query_lower = query.to_lowercase();
                Ok(philosophical_keywords
                    .iter()
                    .any(|&keyword| query_lower.contains(keyword)))
            }
            TransitionCondition::TimeElapsed(_duration) => {
                // For simplicity, always return false for time-based conditions
                Ok(false)
            }
        }
    }

    /// Identify the trigger for state transition
    fn identify_trigger(
        &self,
        query: &str,
        _context: &AssembledContext,
    ) -> Result<TransitionTrigger> {
        // Simple trigger identification based on query content
        if query.len() > 100 {
            Ok(TransitionTrigger::QueryComplexity(
                query.len() as f64 / 100.0,
            ))
        } else if query.contains('?') {
            Ok(TransitionTrigger::AttentionShift(
                "Question pattern detected".to_string(),
            ))
        } else {
            Ok(TransitionTrigger::ExternalStimulus(
                "General content trigger".to_string(),
            ))
        }
    }

    /// Calculate query complexity
    fn calculate_query_complexity(&self, query: &str) -> Result<f64> {
        let word_count = query.split_whitespace().count();
        let unique_words = query
            .split_whitespace()
            .collect::<std::collections::HashSet<_>>()
            .len();
        let complexity = (word_count as f64 * 0.1 + unique_words as f64 * 0.2).min(1.0);
        Ok(complexity)
    }

    /// Calculate emotional content score
    fn calculate_emotional_content(&self, query: &str) -> Result<f64> {
        let emotional_keywords = [
            "happy",
            "sad",
            "angry",
            "excited",
            "disappointed",
            "love",
            "hate",
        ];
        let query_lower = query.to_lowercase();
        let emotional_count = emotional_keywords
            .iter()
            .filter(|&keyword| query_lower.contains(keyword))
            .count();
        Ok((emotional_count as f64 / emotional_keywords.len() as f64).min(1.0))
    }
}

/// Consolidation metrics for memory processing
#[derive(Debug, Clone)]
pub struct ConsolidationMetrics {
    pub consolidation_rate: f64,
    pub memory_retention: f64,
    pub insight_generation_rate: f64,
}

impl Default for ConsolidationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsolidationMetrics {
    pub fn new() -> Self {
        Self {
            consolidation_rate: 0.5,
            memory_retention: 0.8,
            insight_generation_rate: 0.3,
        }
    }

    pub fn update(&mut self, consolidation_count: usize, dream_intensity: f64) {
        // Update consolidation metrics based on processing results
        self.consolidation_rate = (consolidation_count as f64 / 10.0).min(1.0);
        self.memory_retention = (self.memory_retention + dream_intensity * 0.1).min(1.0);
        self.insight_generation_rate = (dream_intensity * 0.5).min(1.0);
    }
}

#[derive(Clone)]
pub struct TransitionRule {
    target_state: ConsciousnessState,
    condition: TransitionCondition,
    probability: f64,
}

impl TransitionRule {
    pub fn new(
        target_state: ConsciousnessState,
        condition: TransitionCondition,
        probability: f64,
    ) -> Self {
        Self {
            target_state,
            condition,
            probability,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TransitionCondition {
    QueryComplexity(f64),
    EmotionalContent(f64),
    KeywordPresence(Vec<&'static str>),
    LogicalPattern,
    PhilosophicalContent,
    TimeElapsed(Duration),
}

#[derive(Debug, Clone)]
pub struct StateProcessingParameters {
    pub attention_focus: f64,
    pub emotional_sensitivity: f64,
    pub creativity_boost: f64,
    pub analytical_depth: f64,
    pub memory_consolidation: f64,
}

impl Default for StateProcessingParameters {
    fn default() -> Self {
        Self {
            attention_focus: 0.5,
            emotional_sensitivity: 0.5,
            creativity_boost: 0.5,
            analytical_depth: 0.5,
            memory_consolidation: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StateMetrics {
    state_transitions: usize,
    average_state_duration: Duration,
    most_frequent_state: ConsciousnessState,
    state_effectiveness: HashMap<ConsciousnessState, f64>,
}

impl Default for StateMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl StateMetrics {
    pub fn new() -> Self {
        Self {
            state_transitions: 0,
            average_state_duration: Duration::from_secs(0),
            most_frequent_state: ConsciousnessState::Baseline,
            state_effectiveness: HashMap::new(),
        }
    }
}
