//! Model selection and hyperparameter optimization
//!
//! This module provides tools for selecting the best ML models and optimizing
//! their hyperparameters for shape learning tasks.

use super::{
    ModelError, ModelMetrics, ModelParams, ShapeLearningModel, ShapeTrainingData,
    GraphData, LearnedShape,
};
use rand::SeedableRng;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::Rng;

/// Model selector for finding the best model and hyperparameters
#[derive(Debug)]
pub struct ModelSelector {
    config: ModelSelectionConfig,
    search_history: Vec<SearchResult>,
    best_result: Option<BestModel>,
}

/// Model selection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSelectionConfig {
    pub search_method: SearchMethod,
    pub evaluation_metric: EvaluationMetric,
    pub cv_folds: usize,
    pub n_iter: usize,
    pub random_state: Option<u64>,
    pub early_stopping: bool,
    pub patience: usize,
    pub parallel_jobs: usize,
}

/// Search methods for hyperparameter optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchMethod {
    GridSearch,
    RandomSearch,
    BayesianOptimization,
    EvolutionaryAlgorithm,
    Hyperband,
}

/// Evaluation metrics for model selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvaluationMetric {
    Accuracy,
    F1Score,
    AUC,
    Precision,
    Recall,
    CustomMetric(String),
}

/// Search result for a single model configuration
#[derive(Debug, Clone)]
struct SearchResult {
    model_name: String,
    params: ModelParams,
    cv_scores: Vec<f64>,
    mean_score: f64,
    std_score: f64,
    training_time: std::time::Duration,
}

/// Best model information
#[derive(Debug, Clone)]
pub struct BestModel {
    pub model_name: String,
    pub params: ModelParams,
    pub score: f64,
    pub metrics: ModelMetrics,
    pub feature_importance: HashMap<String, f64>,
}

/// Parameter space definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSpace {
    pub continuous_params: HashMap<String, ContinuousParam>,
    pub discrete_params: HashMap<String, DiscreteParam>,
    pub categorical_params: HashMap<String, CategoricalParam>,
}

/// Continuous parameter range
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousParam {
    pub min: f64,
    pub max: f64,
    pub scale: ScaleType,
}

/// Discrete parameter range
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteParam {
    pub min: i32,
    pub max: i32,
    pub step: i32,
}

/// Categorical parameter choices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalParam {
    pub choices: Vec<String>,
}

/// Scale types for continuous parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScaleType {
    Linear,
    Log,
    Exponential,
}

/// Cross-validation splitter
#[derive(Debug)]
struct CrossValidator {
    n_folds: usize,
    shuffle: bool,
    random_state: Option<u64>,
}

/// Bayesian optimization state
#[derive(Debug)]
struct BayesianOptimizer {
    acquisition_function: AcquisitionFunction,
    surrogate_model: SurrogateModel,
    explored_points: Vec<(ModelParams, f64)>,
}

/// Acquisition functions for Bayesian optimization
#[derive(Debug, Clone)]
enum AcquisitionFunction {
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound(f64),
    ThompsonSampling,
}

/// Surrogate models for Bayesian optimization
#[derive(Debug)]
enum SurrogateModel {
    GaussianProcess,
    RandomForest,
    TPE, // Tree-structured Parzen Estimator
}

impl ModelSelector {
    /// Create a new model selector
    pub fn new(config: ModelSelectionConfig) -> Self {
        Self {
            config,
            search_history: Vec::new(),
            best_result: None,
        }
    }

    /// Select the best model from a set of candidates
    pub fn select_best_model(
        &mut self,
        models: Vec<Box<dyn ShapeLearningModel>>,
        param_spaces: HashMap<String, ParameterSpace>,
        training_data: &ShapeTrainingData,
    ) -> Result<BestModel, ModelError> {
        tracing::info!(
            "Starting model selection with {} models using {:?}",
            models.len(),
            self.config.search_method
        );

        match self.config.search_method {
            SearchMethod::GridSearch => {
                self.grid_search(models, param_spaces, training_data)
            }
            SearchMethod::RandomSearch => {
                self.random_search(models, param_spaces, training_data)
            }
            SearchMethod::BayesianOptimization => {
                self.bayesian_optimization(models, param_spaces, training_data)
            }
            SearchMethod::EvolutionaryAlgorithm => {
                self.evolutionary_search(models, param_spaces, training_data)
            }
            SearchMethod::Hyperband => {
                self.hyperband_search(models, param_spaces, training_data)
            }
        }
    }

    /// Grid search over parameter space
    fn grid_search(
        &mut self,
        mut models: Vec<Box<dyn ShapeLearningModel>>,
        param_spaces: HashMap<String, ParameterSpace>,
        training_data: &ShapeTrainingData,
    ) -> Result<BestModel, ModelError> {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_model = None;

        for (model_idx, model) in models.iter_mut().enumerate() {
            let model_name = format!("model_{}", model_idx);
            let param_space = param_spaces.get(&model_name)
                .ok_or_else(|| ModelError::InvalidParams("No parameter space defined".to_string()))?;

            let param_grid = self.generate_param_grid(param_space);

            for params in param_grid {
                model.set_params(params.clone())?;
                
                let cv_scores = self.cross_validate(model.as_mut(), training_data)?;
                let mean_score = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;
                let std_score = self.calculate_std(&cv_scores, mean_score);

                let result = SearchResult {
                    model_name: model_name.clone(),
                    params: params.clone(),
                    cv_scores,
                    mean_score,
                    std_score,
                    training_time: std::time::Duration::from_secs(0),
                };

                self.search_history.push(result);

                if mean_score > best_score {
                    best_score = mean_score;
                    
                    // Train on full dataset
                    let metrics = model.train(training_data)?;
                    
                    best_model = Some(BestModel {
                        model_name: model_name.clone(),
                        params,
                        score: mean_score,
                        metrics,
                        feature_importance: HashMap::new(),
                    });
                }
            }
        }

        self.best_result = best_model.clone();
        best_model.ok_or_else(|| ModelError::TrainingError("No model found".to_string()))
    }

    /// Random search over parameter space
    fn random_search(
        &mut self,
        mut models: Vec<Box<dyn ShapeLearningModel>>,
        param_spaces: HashMap<String, ParameterSpace>,
        training_data: &ShapeTrainingData,
    ) -> Result<BestModel, ModelError> {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_model = None;
        let mut rng = rand::thread_rng();

        for _ in 0..self.config.n_iter {
            // Randomly select a model
            let model_idx = rng.gen_range(0..models.len());
            let model = &mut models[model_idx];
            let model_name = format!("model_{}", model_idx);

            let param_space = param_spaces.get(&model_name)
                .ok_or_else(|| ModelError::InvalidParams("No parameter space defined".to_string()))?;

            // Sample random parameters
            let params = self.sample_random_params(param_space);
            model.set_params(params.clone())?;

            let cv_scores = self.cross_validate(model.as_mut(), training_data)?;
            let mean_score = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;
            let std_score = self.calculate_std(&cv_scores, mean_score);

            let result = SearchResult {
                model_name: model_name.clone(),
                params: params.clone(),
                cv_scores,
                mean_score,
                std_score,
                training_time: std::time::Duration::from_secs(0),
            };

            self.search_history.push(result);

            if mean_score > best_score {
                best_score = mean_score;
                
                let metrics = model.train(training_data)?;
                
                best_model = Some(BestModel {
                    model_name: model_name.clone(),
                    params,
                    score: mean_score,
                    metrics,
                    feature_importance: HashMap::new(),
                });
            }
        }

        self.best_result = best_model.clone();
        best_model.ok_or_else(|| ModelError::TrainingError("No model found".to_string()))
    }

    /// Bayesian optimization
    fn bayesian_optimization(
        &mut self,
        mut models: Vec<Box<dyn ShapeLearningModel>>,
        param_spaces: HashMap<String, ParameterSpace>,
        training_data: &ShapeTrainingData,
    ) -> Result<BestModel, ModelError> {
        let mut optimizer = BayesianOptimizer::new(
            AcquisitionFunction::ExpectedImprovement,
            SurrogateModel::GaussianProcess,
        );

        let mut best_score = f64::NEG_INFINITY;
        let mut best_model = None;

        // Initial random exploration
        for _ in 0..5 {
            let model_idx = 0; // Simplified - use first model
            let model = &mut models[model_idx];
            let model_name = format!("model_{}", model_idx);

            let param_space = param_spaces.get(&model_name)
                .ok_or_else(|| ModelError::InvalidParams("No parameter space defined".to_string()))?;

            let params = self.sample_random_params(param_space);
            model.set_params(params.clone())?;

            let cv_scores = self.cross_validate(model.as_mut(), training_data)?;
            let score = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;

            optimizer.add_observation(params.clone(), score);

            if score > best_score {
                best_score = score;
                let metrics = model.train(training_data)?;
                
                best_model = Some(BestModel {
                    model_name: model_name.clone(),
                    params,
                    score,
                    metrics,
                    feature_importance: HashMap::new(),
                });
            }
        }

        // Bayesian optimization loop
        for _ in 0..self.config.n_iter - 5 {
            let model_idx = 0;
            let model = &mut models[model_idx];
            let model_name = format!("model_{}", model_idx);

            let param_space = param_spaces.get(&model_name)
                .ok_or_else(|| ModelError::InvalidParams("No parameter space defined".to_string()))?;

            // Get next point to evaluate from optimizer
            let params = optimizer.suggest_next(param_space);
            model.set_params(params.clone())?;

            let cv_scores = self.cross_validate(model.as_mut(), training_data)?;
            let score = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;

            optimizer.add_observation(params.clone(), score);

            if score > best_score {
                best_score = score;
                let metrics = model.train(training_data)?;
                
                best_model = Some(BestModel {
                    model_name: model_name.clone(),
                    params,
                    score,
                    metrics,
                    feature_importance: HashMap::new(),
                });
            }
        }

        self.best_result = best_model.clone();
        best_model.ok_or_else(|| ModelError::TrainingError("No model found".to_string()))
    }

    /// Evolutionary algorithm search
    fn evolutionary_search(
        &mut self,
        mut models: Vec<Box<dyn ShapeLearningModel>>,
        param_spaces: HashMap<String, ParameterSpace>,
        training_data: &ShapeTrainingData,
    ) -> Result<BestModel, ModelError> {
        // Simplified evolutionary algorithm
        let population_size = 20;
        let mut population = Vec::new();

        // Initialize population
        for _ in 0..population_size {
            let model_idx = 0; // Simplified
            let model_name = format!("model_{}", model_idx);
            let param_space = param_spaces.get(&model_name)
                .ok_or_else(|| ModelError::InvalidParams("No parameter space defined".to_string()))?;
            
            let params = self.sample_random_params(param_space);
            population.push((model_idx, params));
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_model = None;

        for generation in 0..self.config.n_iter / population_size {
            let mut scores = Vec::new();

            // Evaluate population
            for (model_idx, params) in &population {
                let model = &mut models[*model_idx];
                model.set_params(params.clone())?;

                let cv_scores = self.cross_validate(model.as_mut(), training_data)?;
                let score = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;
                scores.push(score);

                if score > best_score {
                    best_score = score;
                    let metrics = model.train(training_data)?;
                    
                    best_model = Some(BestModel {
                        model_name: format!("model_{}", model_idx),
                        params: params.clone(),
                        score,
                        metrics,
                        feature_importance: HashMap::new(),
                    });
                }
            }

            // Evolution step (simplified)
            if generation < self.config.n_iter / population_size - 1 {
                population = self.evolve_population(population, scores, &param_spaces);
            }
        }

        self.best_result = best_model.clone();
        best_model.ok_or_else(|| ModelError::TrainingError("No model found".to_string()))
    }

    /// Hyperband search
    fn hyperband_search(
        &mut self,
        models: Vec<Box<dyn ShapeLearningModel>>,
        param_spaces: HashMap<String, ParameterSpace>,
        training_data: &ShapeTrainingData,
    ) -> Result<BestModel, ModelError> {
        // Simplified Hyperband implementation
        self.random_search(models, param_spaces, training_data)
    }

    /// Cross-validate a model
    fn cross_validate(
        &self,
        model: &mut dyn ShapeLearningModel,
        data: &ShapeTrainingData,
    ) -> Result<Vec<f64>, ModelError> {
        let cv = CrossValidator::new(self.config.cv_folds, true, self.config.random_state);
        let folds = cv.split(data)?;
        let mut scores = Vec::new();

        for (train_data, val_data) in folds {
            model.train(&train_data)?;
            let metrics = model.evaluate(&val_data)?;
            let score = self.get_metric_score(&metrics);
            scores.push(score);
        }

        Ok(scores)
    }

    /// Get score from metrics based on evaluation metric
    fn get_metric_score(&self, metrics: &ModelMetrics) -> f64 {
        match self.config.evaluation_metric {
            EvaluationMetric::Accuracy => metrics.accuracy,
            EvaluationMetric::F1Score => metrics.f1_score,
            EvaluationMetric::AUC => metrics.auc_roc,
            EvaluationMetric::Precision => metrics.precision,
            EvaluationMetric::Recall => metrics.recall,
            EvaluationMetric::CustomMetric(_) => {
                // Custom metric implementation would go here
                metrics.f1_score
            }
        }
    }

    /// Generate parameter grid for grid search
    fn generate_param_grid(&self, param_space: &ParameterSpace) -> Vec<ModelParams> {
        let mut grid = vec![ModelParams::default()];

        // Generate all combinations (simplified)
        for _ in 0..10 {
            grid.push(self.sample_random_params(param_space));
        }

        grid
    }

    /// Sample random parameters from parameter space
    fn sample_random_params(&self, param_space: &ParameterSpace) -> ModelParams {
        let mut rng = rand::thread_rng();
        let mut params = ModelParams::default();

        // Sample continuous parameters
        for (name, cont_param) in &param_space.continuous_params {
            let value = match cont_param.scale {
                ScaleType::Linear => {
                    rng.gen_range(cont_param.min..cont_param.max)
                }
                ScaleType::Log => {
                    let log_min = cont_param.min.ln();
                    let log_max = cont_param.max.ln();
                    rng.gen_range(log_min..log_max).exp()
                }
                ScaleType::Exponential => {
                    let exp_min = cont_param.min.exp();
                    let exp_max = cont_param.max.exp();
                    rng.gen_range(exp_min..exp_max).ln()
                }
            };

            match name.as_str() {
                "learning_rate" => params.learning_rate = value,
                _ => {
                    params.model_specific.insert(name.clone(), serde_json::json!(value));
                }
            }
        }

        // Sample discrete parameters
        for (name, disc_param) in &param_space.discrete_params {
            let value = rng.gen_range(disc_param.min..=disc_param.max);
            
            match name.as_str() {
                "batch_size" => params.batch_size = value as usize,
                "num_epochs" => params.num_epochs = value as usize,
                _ => {
                    params.model_specific.insert(name.clone(), serde_json::json!(value));
                }
            }
        }

        params
    }

    /// Calculate standard deviation
    fn calculate_std(&self, values: &[f64], mean: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let variance = values.iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;

        variance.sqrt()
    }

    /// Evolve population for evolutionary algorithm
    fn evolve_population(
        &self,
        population: Vec<(usize, ModelParams)>,
        scores: Vec<f64>,
        _param_spaces: &HashMap<String, ParameterSpace>,
    ) -> Vec<(usize, ModelParams)> {
        // Simplified evolution - just keep best and mutate
        let mut indexed_scores: Vec<(usize, f64)> = scores.iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut new_population = Vec::new();
        
        // Keep top half
        for i in 0..population.len() / 2 {
            let idx = indexed_scores[i].0;
            new_population.push(population[idx].clone());
        }

        // Mutate to fill rest
        let mut rng = rand::thread_rng();
        for i in 0..population.len() / 2 {
            let idx = indexed_scores[i].0;
            let (model_idx, mut params) = population[idx].clone();
            
            // Simple mutation
            params.learning_rate *= rng.gen_range(0.8..1.2);
            params.batch_size = (params.batch_size as f64 * rng.gen_range(0.8..1.2)) as usize;
            
            new_population.push((model_idx, params));
        }

        new_population
    }

    /// Get search history
    pub fn get_search_history(&self) -> &[SearchResult] {
        &self.search_history
    }

    /// Get best model
    pub fn get_best_model(&self) -> Option<&BestModel> {
        self.best_result.as_ref()
    }
}

impl CrossValidator {
    fn new(n_folds: usize, shuffle: bool, random_state: Option<u64>) -> Self {
        Self {
            n_folds,
            shuffle,
            random_state,
        }
    }

    fn split(&self, data: &ShapeTrainingData) -> Result<Vec<(ShapeTrainingData, ShapeTrainingData)>, ModelError> {
        let n_samples = data.graph_features.len();
        let fold_size = n_samples / self.n_folds;
        
        let mut indices: Vec<usize> = (0..n_samples).collect();
        
        if self.shuffle {
            use rand::seq::SliceRandom;
            use rand::SeedableRng;
            let mut rng = match self.random_state {
                Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
                None => rand::rngs::StdRng::from_entropy(),
            };
            indices.shuffle(&mut rng);
        }

        let mut folds = Vec::new();
        
        for fold in 0..self.n_folds {
            let start = fold * fold_size;
            let end = if fold == self.n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            let val_indices: Vec<usize> = indices[start..end].to_vec();
            let train_indices: Vec<usize> = indices[..start].iter()
                .chain(indices[end..].iter())
                .cloned()
                .collect();

            let train_data = ShapeTrainingData {
                graph_features: train_indices.iter()
                    .map(|&i| data.graph_features[i].clone())
                    .collect(),
                shape_labels: train_indices.iter()
                    .map(|&i| data.shape_labels[i].clone())
                    .collect(),
                metadata: data.metadata.clone(),
            };

            let val_data = ShapeTrainingData {
                graph_features: val_indices.iter()
                    .map(|&i| data.graph_features[i].clone())
                    .collect(),
                shape_labels: val_indices.iter()
                    .map(|&i| data.shape_labels[i].clone())
                    .collect(),
                metadata: data.metadata.clone(),
            };

            folds.push((train_data, val_data));
        }

        Ok(folds)
    }
}

impl BayesianOptimizer {
    fn new(acquisition_function: AcquisitionFunction, surrogate_model: SurrogateModel) -> Self {
        Self {
            acquisition_function,
            surrogate_model,
            explored_points: Vec::new(),
        }
    }

    fn add_observation(&mut self, params: ModelParams, score: f64) {
        self.explored_points.push((params, score));
    }

    fn suggest_next(&self, param_space: &ParameterSpace) -> ModelParams {
        // Simplified suggestion - would use surrogate model in real implementation
        let mut rng = rand::thread_rng();
        let mut params = ModelParams::default();

        for (name, cont_param) in &param_space.continuous_params {
            let value = rng.gen_range(cont_param.min..cont_param.max);
            match name.as_str() {
                "learning_rate" => params.learning_rate = value,
                _ => {
                    params.model_specific.insert(name.clone(), serde_json::json!(value));
                }
            }
        }

        params
    }
}

impl Default for ModelSelectionConfig {
    fn default() -> Self {
        Self {
            search_method: SearchMethod::RandomSearch,
            evaluation_metric: EvaluationMetric::F1Score,
            cv_folds: 5,
            n_iter: 50,
            random_state: Some(42),
            early_stopping: true,
            patience: 10,
            parallel_jobs: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_selector_creation() {
        let config = ModelSelectionConfig::default();
        let selector = ModelSelector::new(config);
        assert!(selector.search_history.is_empty());
        assert!(selector.best_result.is_none());
    }
}