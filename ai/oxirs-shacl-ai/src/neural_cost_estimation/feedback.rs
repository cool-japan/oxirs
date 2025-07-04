//! Real-time feedback processing for neural cost estimation

use oxirs_core::query::algebra::AlgebraTriplePattern;

use super::{core::QueryExecutionContext, types::*};
use crate::{Result, ShaclAiError};

/// Real-time feedback processor
#[derive(Debug)]
pub struct RealTimeFeedbackProcessor {
    /// Feedback history
    feedback_history: Vec<FeedbackRecord>,

    /// Adaptation parameters
    adaptation_rate: f64,

    /// Performance metrics
    metrics: FeedbackMetrics,
}

/// Feedback record
#[derive(Debug, Clone)]
pub struct FeedbackRecord {
    pub patterns: Vec<AlgebraTriplePattern>,
    pub context: QueryExecutionContext,
    pub actual_cost: f64,
    pub predicted_cost: f64,
    pub error: f64,
    pub timestamp: std::time::SystemTime,
}

/// Feedback metrics
#[derive(Debug, Clone)]
pub struct FeedbackMetrics {
    pub total_feedback_count: usize,
    pub average_error: f64,
    pub error_trend: f64,
    pub adaptation_count: usize,
}

impl RealTimeFeedbackProcessor {
    pub fn new() -> Self {
        Self {
            feedback_history: Vec::new(),
            adaptation_rate: 0.01,
            metrics: FeedbackMetrics {
                total_feedback_count: 0,
                average_error: 0.0,
                error_trend: 0.0,
                adaptation_count: 0,
            },
        }
    }

    /// Process feedback from actual performance
    pub fn process_feedback(
        &mut self,
        patterns: &[AlgebraTriplePattern],
        context: &QueryExecutionContext,
        actual_cost: f64,
        prediction: &CostPrediction,
    ) -> Result<()> {
        let error = (prediction.estimated_cost - actual_cost).abs() / actual_cost.max(1.0);

        let feedback_record = FeedbackRecord {
            patterns: patterns.to_vec(),
            context: context.clone(),
            actual_cost,
            predicted_cost: prediction.estimated_cost,
            error,
            timestamp: std::time::SystemTime::now(),
        };

        self.feedback_history.push(feedback_record);

        // Update metrics
        self.update_metrics();

        // Trigger adaptation if needed
        if self.should_adapt() {
            self.adapt_parameters()?;
        }

        // Keep only recent feedback
        if self.feedback_history.len() > 1000 {
            self.feedback_history.drain(0..100);
        }

        Ok(())
    }

    fn update_metrics(&mut self) {
        self.metrics.total_feedback_count = self.feedback_history.len();

        if !self.feedback_history.is_empty() {
            self.metrics.average_error = self.feedback_history.iter().map(|r| r.error).sum::<f64>()
                / self.feedback_history.len() as f64;

            // Calculate error trend (simplified)
            if self.feedback_history.len() >= 10 {
                let recent_error: f64 = self
                    .feedback_history
                    .iter()
                    .rev()
                    .take(5)
                    .map(|r| r.error)
                    .sum::<f64>()
                    / 5.0;

                let older_error: f64 = self
                    .feedback_history
                    .iter()
                    .rev()
                    .skip(5)
                    .take(5)
                    .map(|r| r.error)
                    .sum::<f64>()
                    / 5.0;

                self.metrics.error_trend = recent_error - older_error;
            }
        }
    }

    fn should_adapt(&self) -> bool {
        // Adapt if error is increasing or above threshold
        self.metrics.average_error > 0.2 || self.metrics.error_trend > 0.05
    }

    fn adapt_parameters(&mut self) -> Result<()> {
        // Simple adaptation logic
        if self.metrics.error_trend > 0.0 {
            self.adaptation_rate *= 1.1; // Increase learning rate
        } else {
            self.adaptation_rate *= 0.95; // Decrease learning rate
        }

        self.adaptation_rate = self.adaptation_rate.clamp(0.001, 0.1);
        self.metrics.adaptation_count += 1;

        Ok(())
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> &FeedbackMetrics {
        &self.metrics
    }

    /// Get adaptation rate
    pub fn get_adaptation_rate(&self) -> f64 {
        self.adaptation_rate
    }
}
