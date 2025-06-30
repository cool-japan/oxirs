//! Advanced explainer implementations (SHAP, LIME, Integrated Gradients)

use std::collections::HashMap;

use super::types::*;
use crate::{Result, ShaclAiError};

/// SHAP (SHapley Additive exPlanations) explainer
#[derive(Debug)]
pub struct SHAPExplainer {
    baseline_values: Vec<f64>,
    max_coalitions: usize,
    exact_calculation: bool,
}

impl SHAPExplainer {
    pub fn new(baseline_values: Vec<f64>) -> Self {
        Self {
            baseline_values,
            max_coalitions: 1000,
            exact_calculation: false,
        }
    }

    pub fn with_config(
        baseline_values: Vec<f64>,
        max_coalitions: usize,
        exact_calculation: bool,
    ) -> Self {
        Self {
            baseline_values,
            max_coalitions,
            exact_calculation,
        }
    }

    /// Calculate SHAP values for given input
    pub async fn calculate_shap_values(
        &self,
        input_features: &[f64],
        prediction_function: &dyn PredictionFunction,
    ) -> Result<Vec<f64>> {
        if self.exact_calculation {
            self.exact_shap_values(input_features, prediction_function).await
        } else {
            self.approximate_shap_values(input_features, prediction_function).await
        }
    }

    async fn exact_shap_values(
        &self,
        input_features: &[f64],
        prediction_function: &dyn PredictionFunction,
    ) -> Result<Vec<f64>> {
        let num_features = input_features.len();
        let mut shap_values = vec![0.0; num_features];

        // Generate all possible coalitions (subsets)
        for feature_idx in 0..num_features {
            let mut marginal_contributions = Vec::new();

            // Iterate through all possible coalitions
            for coalition_mask in 0u32..(1u32 << num_features) {
                if (coalition_mask & (1 << feature_idx)) != 0 {
                    continue; // Skip coalitions that already include this feature
                }

                let coalition_size = coalition_mask.count_ones() as usize;
                let weight = self.shapley_weight(coalition_size, num_features);

                // Coalition without feature
                let mut coalition_without = input_features.to_vec();
                for i in 0..num_features {
                    if (coalition_mask & (1 << i)) == 0 {
                        coalition_without[i] = self.baseline_values[i];
                    }
                }

                // Coalition with feature
                let mut coalition_with = coalition_without.clone();
                coalition_with[feature_idx] = input_features[feature_idx];

                let prediction_without = prediction_function.predict(&coalition_without).await?;
                let prediction_with = prediction_function.predict(&coalition_with).await?;

                let marginal_contribution = (prediction_with - prediction_without) * weight;
                marginal_contributions.push(marginal_contribution);
            }

            shap_values[feature_idx] = marginal_contributions.iter().sum();
        }

        Ok(shap_values)
    }

    async fn approximate_shap_values(
        &self,
        input_features: &[f64],
        prediction_function: &dyn PredictionFunction,
    ) -> Result<Vec<f64>> {
        let num_features = input_features.len();
        let mut shap_values = vec![0.0; num_features];

        for feature_idx in 0..num_features {
            let mut total_contribution = 0.0;
            let mut total_weight = 0.0;

            // Sample coalitions instead of exhaustive enumeration
            for _ in 0..self.max_coalitions {
                let coalition_size = fastrand::usize(0..num_features);
                let weight = self.shapley_weight(coalition_size, num_features);

                // Generate random coalition of specified size
                let mut coalition_indices: Vec<usize> = (0..num_features).collect();
                coalition_indices.shuffle(&mut fastrand::Rng::new());
                let coalition_indices = &coalition_indices[..coalition_size];

                // Create coalition without and with the feature
                let mut coalition_without = self.baseline_values.clone();
                let mut coalition_with = self.baseline_values.clone();

                for &idx in coalition_indices {
                    if idx != feature_idx {
                        coalition_without[idx] = input_features[idx];
                        coalition_with[idx] = input_features[idx];
                    }
                }
                coalition_with[feature_idx] = input_features[feature_idx];

                let prediction_without = prediction_function.predict(&coalition_without).await?;
                let prediction_with = prediction_function.predict(&coalition_with).await?;

                let marginal_contribution = (prediction_with - prediction_without) * weight;
                total_contribution += marginal_contribution;
                total_weight += weight;
            }

            shap_values[feature_idx] = if total_weight > 0.0 {
                total_contribution / total_weight
            } else {
                0.0
            };
        }

        Ok(shap_values)
    }

    fn shapley_weight(&self, coalition_size: usize, total_features: usize) -> f64 {
        let n = total_features as f64;
        let s = coalition_size as f64;
        1.0 / (n * binomial_coefficient(total_features - 1, coalition_size))
    }

    async fn calculate_baseline(
        &self,
        input_features: &[f64],
        strategy: BaselineStrategy,
    ) -> Result<Vec<f64>> {
        match strategy {
            BaselineStrategy::Zero => Ok(vec![0.0; input_features.len()]),
            BaselineStrategy::Mean => {
                let mean = input_features.iter().sum::<f64>() / input_features.len() as f64;
                Ok(vec![mean; input_features.len()])
            }
            BaselineStrategy::Random => {
                let mut baseline = Vec::new();
                for _ in 0..input_features.len() {
                    baseline.push(fastrand::f64() * 2.0 - 1.0); // Range: -1.0 to 1.0
                }
                Ok(baseline)
            }
            BaselineStrategy::Custom(values) => Ok(values),
        }
    }
}

/// LIME (Local Interpretable Model-agnostic Explanations) explainer
#[derive(Debug)]
pub struct LIMEExplainer {
    num_samples: usize,
    num_features: usize,
    perturbation_std: f64,
    kernel_width: f64,
    surrogate_model: SurrogateModel,
}

impl LIMEExplainer {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_samples: 1000,
            num_features,
            perturbation_std: 0.1,
            kernel_width: 0.75,
            surrogate_model: SurrogateModel::LinearRegression,
        }
    }

    pub fn with_config(
        num_features: usize,
        num_samples: usize,
        perturbation_std: f64,
        kernel_width: f64,
        surrogate_model: SurrogateModel,
    ) -> Self {
        Self {
            num_samples,
            num_features,
            perturbation_std,
            kernel_width,
            surrogate_model,
        }
    }

    /// Generate LIME explanation for given input
    pub async fn explain(
        &self,
        input_features: &[f64],
        prediction_function: &dyn PredictionFunction,
    ) -> Result<Vec<f64>> {
        // Generate perturbations around the input
        let perturbations = self.generate_perturbations(input_features);
        
        // Get predictions for all perturbations
        let mut predictions = Vec::new();
        for perturbation in &perturbations {
            let pred = prediction_function.predict(perturbation).await?;
            predictions.push(pred);
        }

        // Calculate weights based on distance to original input
        let weights = self.calculate_weights(input_features, &perturbations);

        // Fit surrogate model
        let coefficients = self.fit_surrogate_model(&perturbations, &predictions, &weights)?;

        Ok(coefficients)
    }

    fn generate_perturbations(&self, input_features: &[f64]) -> Vec<Vec<f64>> {
        let mut perturbations = Vec::new();
        perturbations.push(input_features.to_vec()); // Include original

        for _ in 1..self.num_samples {
            let mut perturbation = Vec::new();
            for &feature in input_features {
                let noise = fastrand::f64() * self.perturbation_std * 2.0 - self.perturbation_std;
                perturbation.push(feature + noise);
            }
            perturbations.push(perturbation);
        }

        perturbations
    }

    fn calculate_weights(&self, original: &[f64], perturbations: &[Vec<f64>]) -> Vec<f64> {
        perturbations
            .iter()
            .map(|perturbation| {
                let distance = euclidean_distance(original, perturbation);
                (-distance.powi(2) / self.kernel_width.powi(2)).exp()
            })
            .collect()
    }

    fn fit_surrogate_model(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        weights: &[f64],
    ) -> Result<Vec<f64>> {
        match self.surrogate_model {
            SurrogateModel::LinearRegression => self.fit_weighted_linear_regression(x, y, weights),
            SurrogateModel::DecisionTree => self.fit_decision_tree(x, y, weights),
            SurrogateModel::SimpleNN => self.fit_simple_nn(x, y, weights),
        }
    }

    fn fit_weighted_linear_regression(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        weights: &[f64],
    ) -> Result<Vec<f64>> {
        // Simplified weighted linear regression
        // In practice, you'd use a proper linear algebra library
        let n_features = x[0].len();
        let mut coefficients = vec![0.0; n_features + 1]; // +1 for intercept

        // Simple gradient descent approach (simplified)
        let learning_rate = 0.01;
        let iterations = 1000;

        for _ in 0..iterations {
            let mut gradients = vec![0.0; n_features + 1];

            for (i, (xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
                let weight = weights[i];
                
                // Calculate prediction
                let mut prediction = coefficients[0]; // intercept
                for (j, &xij) in xi.iter().enumerate() {
                    prediction += coefficients[j + 1] * xij;
                }

                let error = prediction - yi;

                // Update gradients
                gradients[0] += weight * error; // intercept gradient
                for (j, &xij) in xi.iter().enumerate() {
                    gradients[j + 1] += weight * error * xij;
                }
            }

            // Update coefficients
            for (j, grad) in gradients.iter().enumerate() {
                coefficients[j] -= learning_rate * grad / x.len() as f64;
            }
        }

        Ok(coefficients[1..].to_vec()) // Return feature coefficients (exclude intercept)
    }

    fn fit_decision_tree(
        &self,
        _x: &[Vec<f64>],
        _y: &[f64],
        _weights: &[f64],
    ) -> Result<Vec<f64>> {
        // Simplified decision tree feature importance
        // In practice, you'd implement a proper decision tree
        let mut importance = vec![0.0; self.num_features];
        for i in 0..self.num_features {
            importance[i] = fastrand::f64(); // Random importance for now
        }
        Ok(importance)
    }

    fn fit_simple_nn(
        &self,
        _x: &[Vec<f64>],
        _y: &[f64],
        _weights: &[f64],
    ) -> Result<Vec<f64>> {
        // Simplified neural network feature importance
        // In practice, you'd implement a proper neural network
        let mut importance = vec![0.0; self.num_features];
        for i in 0..self.num_features {
            importance[i] = fastrand::f64(); // Random importance for now
        }
        Ok(importance)
    }
}

/// Integrated Gradients explainer
#[derive(Debug)]
pub struct IntegratedGradientsExplainer {
    steps: usize,
    baseline_strategy: BaselineStrategy,
    apply_noise_tunnel: bool,
    noise_tunnel_samples: usize,
}

impl IntegratedGradientsExplainer {
    pub fn new(steps: usize, baseline_strategy: BaselineStrategy) -> Self {
        Self {
            steps,
            baseline_strategy,
            apply_noise_tunnel: false,
            noise_tunnel_samples: 10,
        }
    }

    pub fn with_noise_tunnel(mut self, samples: usize) -> Self {
        self.apply_noise_tunnel = true;
        self.noise_tunnel_samples = samples;
        self
    }

    /// Calculate integrated gradients for given input
    pub async fn calculate_integrated_gradients(
        &self,
        input: &[f64],
        baseline: &[f64],
        gradient_function: &dyn GradientFunction,
    ) -> Result<Vec<f64>> {
        if self.apply_noise_tunnel {
            self.apply_noise_tunnel(input, baseline, gradient_function).await
        } else {
            self.calculate_basic_integrated_gradients(input, baseline, gradient_function).await
        }
    }

    async fn calculate_basic_integrated_gradients(
        &self,
        input: &[f64],
        baseline: &[f64],
        gradient_function: &dyn GradientFunction,
    ) -> Result<Vec<f64>> {
        // Generate interpolated inputs
        let interpolated_inputs = self.generate_interpolated_inputs(baseline, input);
        
        // Calculate gradients for each interpolated input
        let mut gradients = Vec::new();
        for interpolated_input in &interpolated_inputs {
            let gradient = gradient_function.compute_gradient(interpolated_input).await?;
            gradients.push(gradient);
        }

        // Integrate gradients
        let integrated = self.integrate_gradients(&gradients, input, baseline);
        
        // Check convergence if needed
        let convergence = self.check_convergence(&integrated, input, baseline, gradient_function).await?;
        if convergence > 0.1 {
            tracing::warn!("Integrated gradients convergence check failed: {}", convergence);
        }

        Ok(integrated)
    }

    fn generate_interpolated_inputs(&self, baseline: &[f64], input: &[f64]) -> Vec<Vec<f64>> {
        let mut interpolated = Vec::new();
        
        for i in 0..=self.steps {
            let alpha = i as f64 / self.steps as f64;
            let interpolated_input: Vec<f64> = baseline
                .iter()
                .zip(input.iter())
                .map(|(&b, &x)| b + alpha * (x - b))
                .collect();
            interpolated.push(interpolated_input);
        }

        interpolated
    }

    fn integrate_gradients(&self, gradients: &[Vec<f64>], input: &[f64], baseline: &[f64]) -> Vec<f64> {
        let input_diff: Vec<f64> = input
            .iter()
            .zip(baseline.iter())
            .map(|(&x, &b)| x - b)
            .collect();

        let mut integrated = vec![0.0; input.len()];

        // Trapezoidal integration
        for i in 0..gradients.len() - 1 {
            for j in 0..input.len() {
                let avg_gradient = (gradients[i][j] + gradients[i + 1][j]) / 2.0;
                integrated[j] += avg_gradient * input_diff[j] / self.steps as f64;
            }
        }

        integrated
    }

    async fn apply_noise_tunnel(
        &self,
        input: &[f64],
        baseline: &[f64],
        gradient_function: &dyn GradientFunction,
    ) -> Result<Vec<f64>> {
        let mut accumulated_attributions = vec![0.0; input.len()];

        for _ in 0..self.noise_tunnel_samples {
            // Add small random noise to input
            let noisy_input: Vec<f64> = input
                .iter()
                .map(|x| x + fastrand::f64() * 0.01 - 0.005)
                .collect();

            // Calculate integrated gradients for noisy input
            let interpolated_inputs = self.generate_interpolated_inputs(baseline, &noisy_input);
            let mut gradients = Vec::new();

            for interpolated_input in &interpolated_inputs {
                let gradient = gradient_function
                    .compute_gradient(interpolated_input)
                    .await?;
                gradients.push(gradient);
            }

            let sample_attributions = self.integrate_gradients(&gradients, &noisy_input, baseline);

            for (acc, sample) in accumulated_attributions
                .iter_mut()
                .zip(sample_attributions.iter())
            {
                *acc += sample;
            }
        }

        // Average over all samples
        for attribution in &mut accumulated_attributions {
            *attribution /= self.noise_tunnel_samples as f64;
        }

        Ok(accumulated_attributions)
    }

    async fn check_convergence(
        &self,
        integrated_gradients: &[f64],
        input: &[f64],
        baseline: &[f64],
        gradient_function: &dyn GradientFunction,
    ) -> Result<f64> {
        // Check if sum of attributions equals difference in predictions
        let baseline_prediction = gradient_function.predict(baseline).await?;
        let input_prediction = gradient_function.predict(input).await?;

        let prediction_diff = input_prediction - baseline_prediction;
        let attribution_sum: f64 = integrated_gradients.iter().sum();

        Ok((prediction_diff - attribution_sum).abs())
    }
}

/// Trait for prediction functions used by explainers
pub trait PredictionFunction: Send + Sync {
    async fn predict(&self, input: &[f64]) -> Result<f64>;
}

/// Trait for gradient functions used by Integrated Gradients
pub trait GradientFunction: Send + Sync {
    async fn compute_gradient(&self, input: &[f64]) -> Result<Vec<f64>>;
    async fn predict(&self, input: &[f64]) -> Result<f64>;
}

// Utility functions

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn binomial_coefficient(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }

    let k = k.min(n - k); // Take advantage of symmetry
    let mut result = 1.0;

    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }

    result
}

trait VecShuffle<T> {
    fn shuffle(&mut self, rng: &mut fastrand::Rng);
}

impl<T> VecShuffle<T> for Vec<T> {
    fn shuffle(&mut self, rng: &mut fastrand::Rng) {
        for i in (1..self.len()).rev() {
            let j = rng.usize(0..=i);
            self.swap(i, j);
        }
    }
}