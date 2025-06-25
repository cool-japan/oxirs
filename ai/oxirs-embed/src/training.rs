//! Training utilities and advanced optimizers for embedding models

use crate::{EmbeddingModel, ModelConfig, TrainingStats};
use anyhow::Result;
use ndarray::Array2;
use std::collections::VecDeque;
use std::time::Instant;
use tracing::{debug, info};

/// Advanced training scheduler with various optimization strategies
pub struct TrainingScheduler {
    pub config: TrainingConfig,
    pub optimizer: OptimizerType,
    pub scheduler: LearningRateScheduler,
    pub early_stopping: Option<EarlyStopping>,
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub max_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub validation_freq: usize,
    pub checkpoint_freq: usize,
    pub log_freq: usize,
    pub use_early_stopping: bool,
    pub patience: usize,
    pub min_delta: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_epochs: 1000,
            batch_size: 1024,
            learning_rate: 0.01,
            validation_freq: 10,
            checkpoint_freq: 100,
            log_freq: 10,
            use_early_stopping: true,
            patience: 50,
            min_delta: 1e-6,
        }
    }
}

/// Optimizer types
#[derive(Debug, Clone)]
pub enum OptimizerType {
    SGD,
    Adam { beta1: f64, beta2: f64, epsilon: f64 },
    AdaGrad { epsilon: f64 },
    RMSprop { alpha: f64, epsilon: f64 },
}

impl Default for OptimizerType {
    fn default() -> Self {
        OptimizerType::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// Learning rate scheduler
#[derive(Debug, Clone)]
pub enum LearningRateScheduler {
    Constant,
    ExponentialDecay { decay_rate: f64, decay_steps: usize },
    StepDecay { step_size: usize, gamma: f64 },
    CosineAnnealing { t_max: usize, eta_min: f64 },
    ReduceOnPlateau { factor: f64, patience: usize, threshold: f64 },
}

impl Default for LearningRateScheduler {
    fn default() -> Self {
        LearningRateScheduler::ExponentialDecay {
            decay_rate: 0.96,
            decay_steps: 100,
        }
    }
}

impl LearningRateScheduler {
    pub fn get_lr(&self, epoch: usize, base_lr: f64, current_loss: Option<f64>) -> f64 {
        match self {
            LearningRateScheduler::Constant => base_lr,
            LearningRateScheduler::ExponentialDecay { decay_rate, decay_steps } => {
                base_lr * decay_rate.powf(epoch as f64 / *decay_steps as f64)
            }
            LearningRateScheduler::StepDecay { step_size, gamma } => {
                base_lr * gamma.powf((epoch / step_size) as f64)
            }
            LearningRateScheduler::CosineAnnealing { t_max, eta_min } => {
                eta_min + (base_lr - eta_min) * (1.0 + (std::f64::consts::PI * epoch as f64 / *t_max as f64).cos()) / 2.0
            }
            LearningRateScheduler::ReduceOnPlateau { .. } => {
                // This would require state tracking, simplified for now
                base_lr
            }
        }
    }
}

/// Early stopping implementation
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    patience: usize,
    min_delta: f64,
    best_loss: f64,
    wait_count: usize,
    stopped: bool,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: f64::INFINITY,
            wait_count: 0,
            stopped: false,
        }
    }
    
    pub fn update(&mut self, current_loss: f64) -> bool {
        if current_loss < self.best_loss - self.min_delta {
            self.best_loss = current_loss;
            self.wait_count = 0;
        } else {
            self.wait_count += 1;
            if self.wait_count > self.patience {
                self.stopped = true;
            }
        }
        
        self.stopped
    }
    
    pub fn should_stop(&self) -> bool {
        self.stopped
    }
}

/// Adam optimizer state
#[derive(Debug, Clone)]
pub struct AdamOptimizer {
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: usize, // time step
    m: Option<Array2<f64>>, // first moment
    v: Option<Array2<f64>>, // second moment
}

impl AdamOptimizer {
    pub fn new(beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            beta1,
            beta2,
            epsilon,
            t: 0,
            m: None,
            v: None,
        }
    }
    
    pub fn update(&mut self, params: &mut Array2<f64>, grads: &Array2<f64>, lr: f64) {
        self.t += 1;
        
        // Initialize moments if needed
        if self.m.is_none() {
            self.m = Some(Array2::zeros(params.raw_dim()));
            self.v = Some(Array2::zeros(params.raw_dim()));
        }
        
        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();
        
        // Update biased first moment estimate
        *m = &*m * self.beta1 + grads * (1.0 - self.beta1);
        
        // Update biased second raw moment estimate
        *v = &*v * self.beta2 + &(grads * grads) * (1.0 - self.beta2);
        
        // Compute bias-corrected first moment estimate
        let m_hat = &*m / (1.0 - self.beta1.powi(self.t as i32));
        
        // Compute bias-corrected second raw moment estimate
        let v_hat = &*v / (1.0 - self.beta2.powi(self.t as i32));
        
        // Update parameters
        *params = &*params - &(&m_hat / (&v_hat.mapv(|x| x.sqrt()) + self.epsilon)) * lr;
    }
}

/// Training metrics tracker
#[derive(Debug, Clone)]
pub struct MetricsTracker {
    pub losses: Vec<f64>,
    pub learning_rates: Vec<f64>,
    pub epochs: Vec<usize>,
    pub validation_losses: Vec<f64>,
    pub training_times: Vec<f64>,
}

impl MetricsTracker {
    pub fn new() -> Self {
        Self {
            losses: Vec::new(),
            learning_rates: Vec::new(),
            epochs: Vec::new(),
            validation_losses: Vec::new(),
            training_times: Vec::new(),
        }
    }
    
    pub fn record_epoch(&mut self, epoch: usize, loss: f64, lr: f64, training_time: f64) {
        self.epochs.push(epoch);
        self.losses.push(loss);
        self.learning_rates.push(lr);
        self.training_times.push(training_time);
    }
    
    pub fn record_validation(&mut self, val_loss: f64) {
        self.validation_losses.push(val_loss);
    }
    
    pub fn get_smoothed_loss(&self, window_size: usize) -> Vec<f64> {
        if self.losses.len() < window_size {
            return self.losses.clone();
        }
        
        let mut smoothed = Vec::new();
        let mut window: VecDeque<f64> = VecDeque::new();
        
        for &loss in &self.losses {
            window.push_back(loss);
            if window.len() > window_size {
                window.pop_front();
            }
            
            let avg = window.iter().sum::<f64>() / window.len() as f64;
            smoothed.push(avg);
        }
        
        smoothed
    }
}

impl Default for MetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced trainer with full optimization capabilities
pub struct AdvancedTrainer {
    config: TrainingConfig,
    optimizer: OptimizerType,
    scheduler: LearningRateScheduler,
    early_stopping: Option<EarlyStopping>,
    metrics: MetricsTracker,
}

impl AdvancedTrainer {
    pub fn new(config: TrainingConfig) -> Self {
        let early_stopping = if config.use_early_stopping {
            Some(EarlyStopping::new(config.patience, config.min_delta))
        } else {
            None
        };
        
        Self {
            config,
            optimizer: OptimizerType::default(),
            scheduler: LearningRateScheduler::default(),
            early_stopping,
            metrics: MetricsTracker::new(),
        }
    }
    
    pub fn with_optimizer(mut self, optimizer: OptimizerType) -> Self {
        self.optimizer = optimizer;
        self
    }
    
    pub fn with_scheduler(mut self, scheduler: LearningRateScheduler) -> Self {
        self.scheduler = scheduler;
        self
    }
    
    pub async fn train(&mut self, model: &mut dyn EmbeddingModel) -> Result<TrainingStats> {
        let start_time = Instant::now();
        info!("Starting advanced training with {} epochs", self.config.max_epochs);
        
        for epoch in 0..self.config.max_epochs {
            let epoch_start = Instant::now();
            
            // Get current learning rate
            let current_lr = self.scheduler.get_lr(epoch, self.config.learning_rate, None);
            
            // Train one epoch
            let epoch_stats = model.train(Some(1)).await?;
            let epoch_loss = epoch_stats.final_loss;
            let epoch_time = epoch_start.elapsed().as_secs_f64();
            
            // Record metrics
            self.metrics.record_epoch(epoch, epoch_loss, current_lr, epoch_time);
            
            // Log progress
            if epoch % self.config.log_freq == 0 {
                debug!("Epoch {}: loss = {:.6}, lr = {:.6}, time = {:.3}s", 
                       epoch, epoch_loss, current_lr, epoch_time);
            }
            
            // Check early stopping
            if let Some(ref mut early_stop) = self.early_stopping {
                if early_stop.update(epoch_loss) {
                    info!("Early stopping triggered at epoch {}", epoch);
                    break;
                }
            }
            
            // Simple convergence check
            if epoch > 10 && epoch_loss < 1e-8 {
                info!("Converged at epoch {} with loss {:.6}", epoch, epoch_loss);
                break;
            }
        }
        
        let training_time = start_time.elapsed().as_secs_f64();
        let final_loss = self.metrics.losses.last().copied().unwrap_or(0.0);
        
        Ok(TrainingStats {
            epochs_completed: self.metrics.epochs.len(),
            final_loss,
            training_time_seconds: training_time,
            convergence_achieved: final_loss < 1e-6,
            loss_history: self.metrics.losses.clone(),
        })
    }
    
    pub fn get_metrics(&self) -> &MetricsTracker {
        &self.metrics
    }
}

/// Validation utilities
pub struct ValidationSuite {
    pub test_triples: Vec<(String, String, String)>,
    pub validation_freq: usize,
}

impl ValidationSuite {
    pub fn new(test_triples: Vec<(String, String, String)>, validation_freq: usize) -> Self {
        Self {
            test_triples,
            validation_freq,
        }
    }
    
    pub fn evaluate_model(&self, model: &dyn EmbeddingModel) -> Result<ValidationMetrics> {
        let mut total_score = 0.0;
        let mut valid_predictions = 0;
        
        for (subject, predicate, object) in &self.test_triples {
            if let Ok(score) = model.score_triple(subject, predicate, object) {
                total_score += score;
                valid_predictions += 1;
            }
        }
        
        let avg_score = if valid_predictions > 0 {
            total_score / valid_predictions as f64
        } else {
            0.0
        };
        
        Ok(ValidationMetrics {
            average_score: avg_score,
            num_evaluated: valid_predictions,
            num_total: self.test_triples.len(),
        })
    }
}

/// Validation metrics
#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    pub average_score: f64,
    pub num_evaluated: usize,
    pub num_total: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_learning_rate_scheduler() {
        let scheduler = LearningRateScheduler::ExponentialDecay {
            decay_rate: 0.9,
            decay_steps: 10,
        };
        
        let lr0 = scheduler.get_lr(0, 0.1, None);
        let lr10 = scheduler.get_lr(10, 0.1, None);
        let lr20 = scheduler.get_lr(20, 0.1, None);
        
        assert!((lr0 - 0.1).abs() < 1e-10);
        assert!(lr10 < lr0);
        assert!(lr20 < lr10);
    }
    
    #[test]
    fn test_early_stopping() {
        let mut early_stop = EarlyStopping::new(3, 0.01);
        
        assert!(!early_stop.update(1.0));
        assert!(!early_stop.update(0.5));
        assert!(!early_stop.update(0.51));
        assert!(!early_stop.update(0.52));
        assert!(!early_stop.update(0.53));
        assert!(early_stop.update(0.54)); // Should stop now
    }
    
    #[test]
    fn test_metrics_tracker() {
        let mut tracker = MetricsTracker::new();
        
        tracker.record_epoch(0, 1.0, 0.01, 1.5);
        tracker.record_epoch(1, 0.5, 0.009, 1.4);
        tracker.record_epoch(2, 0.3, 0.008, 1.3);
        
        assert_eq!(tracker.losses.len(), 3);
        assert_eq!(tracker.epochs.len(), 3);
        
        let smoothed = tracker.get_smoothed_loss(2);
        assert_eq!(smoothed.len(), 3);
    }
}