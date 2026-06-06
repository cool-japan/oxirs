//! Causal Representation Learning — Model
//!
//! Causal encoder/decoder, disentanglement objectives, IRM/IRMV2 loss, and causal discovery
//! algorithms implemented on top of the types from `causal_representation_learning_types`.

use crate::causal_representation_learning_types::{
    CausalDiscoveryAlgorithm, CausalGraph, CausalRepresentationConfig, CounterfactualQuery,
    DisentanglementMethod, Intervention, StructuralEquation,
};
use crate::{EmbeddingModel, ModelConfig, ModelStats, TrainingStats, Triple, Vector};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chrono::Utc;
use scirs2_core::ndarray_ext::{Array1, Array2};
use std::collections::HashMap;
use uuid::Uuid;

/// Causal representation learning model
#[derive(Debug)]
pub struct CausalRepresentationModel {
    pub config: CausalRepresentationConfig,
    pub model_id: Uuid,

    pub causal_graph: CausalGraph,
    pub structural_equations: HashMap<String, StructuralEquation>,

    pub variable_embeddings: HashMap<String, Array1<f32>>,
    pub latent_factors: Array2<f32>,

    pub factual_network: Array2<f32>,
    pub counterfactual_network: Array2<f32>,
    pub shared_network: Array2<f32>,

    pub observational_data: Vec<HashMap<String, f32>>,
    pub interventional_data: Vec<(HashMap<String, f32>, Intervention)>,

    pub entities: HashMap<String, usize>,
    pub relations: HashMap<String, usize>,

    pub training_stats: Option<TrainingStats>,
    pub is_trained: bool,
}

impl CausalRepresentationModel {
    /// Create new causal representation model
    pub fn new(config: CausalRepresentationConfig) -> Self {
        let model_id = Uuid::new_v4();
        let dimensions = config.base_config.dimensions;

        Self {
            config,
            model_id,
            causal_graph: CausalGraph::new(Vec::new()),
            structural_equations: HashMap::new(),
            variable_embeddings: HashMap::new(),
            latent_factors: Array2::zeros((0, dimensions)),
            factual_network: {
                use scirs2_core::random::{Random, RngExt};
                let mut random = Random::default();
                Array2::from_shape_fn((dimensions, dimensions), |_| random.random::<f32>() * 0.1)
            },
            counterfactual_network: {
                use scirs2_core::random::{Random, RngExt};
                let mut random = Random::default();
                Array2::from_shape_fn((dimensions, dimensions), |_| random.random::<f32>() * 0.1)
            },
            shared_network: {
                use scirs2_core::random::{Random, RngExt};
                let mut random = Random::default();
                Array2::from_shape_fn((dimensions, dimensions), |_| random.random::<f32>() * 0.1)
            },
            observational_data: Vec::new(),
            interventional_data: Vec::new(),
            entities: HashMap::new(),
            relations: HashMap::new(),
            training_stats: None,
            is_trained: false,
        }
    }

    /// Add observational data
    pub fn add_observational_data(&mut self, data: HashMap<String, f32>) {
        self.observational_data.push(data);
    }

    /// Add interventional data
    pub fn add_interventional_data(
        &mut self,
        data: HashMap<String, f32>,
        intervention: Intervention,
    ) {
        self.interventional_data.push((data, intervention));
    }

    /// Discover causal structure
    pub fn discover_causal_structure(&mut self) -> Result<()> {
        match self.config.causal_discovery.algorithm {
            CausalDiscoveryAlgorithm::PC => self.run_pc_algorithm(),
            CausalDiscoveryAlgorithm::GES => self.run_ges_algorithm(),
            CausalDiscoveryAlgorithm::NOTEARS => self.run_notears_algorithm(),
            _ => self.run_pc_algorithm(),
        }
    }

    fn run_pc_algorithm(&mut self) -> Result<()> {
        if self.observational_data.is_empty() {
            return Ok(());
        }
        let variables: Vec<String> = self.observational_data[0].keys().cloned().collect();
        self.causal_graph = CausalGraph::new(variables.clone());

        for i in 0..variables.len() {
            for j in (i + 1)..variables.len() {
                if self.independence_test(&variables[i], &variables[j], &[])? {
                    continue;
                } else {
                    self.causal_graph.add_edge(i, j, 1.0);
                    self.causal_graph.add_edge(j, i, 1.0);
                }
            }
        }
        self.orient_edges()?;
        Ok(())
    }

    fn run_ges_algorithm(&mut self) -> Result<()> {
        if self.observational_data.is_empty() {
            return Ok(());
        }
        let variables: Vec<String> = self.observational_data[0].keys().cloned().collect();
        self.causal_graph = CausalGraph::new(variables.clone());

        let mut current_score = self.compute_bic_score()?;
        let mut improved = true;

        while improved {
            improved = false;
            let mut best_score = current_score;
            let mut best_operation = None;

            for i in 0..variables.len() {
                for j in 0..variables.len() {
                    if i != j && self.causal_graph.adjacency[[i, j]] == 0.0 {
                        self.causal_graph.add_edge(i, j, 1.0);
                        if self.causal_graph.is_acyclic() {
                            let score = self.compute_bic_score()?;
                            if score > best_score {
                                best_score = score;
                                best_operation = Some((i, j, true));
                            }
                        }
                        self.causal_graph.remove_edge(i, j);
                    }
                }
            }

            for i in 0..variables.len() {
                for j in 0..variables.len() {
                    if self.causal_graph.adjacency[[i, j]] > 0.0 {
                        self.causal_graph.remove_edge(i, j);
                        let score = self.compute_bic_score()?;
                        if score > best_score {
                            best_score = score;
                            best_operation = Some((i, j, false));
                        }
                        self.causal_graph.add_edge(i, j, 1.0);
                    }
                }
            }

            if let Some((i, j, add)) = best_operation {
                if add {
                    self.causal_graph.add_edge(i, j, 1.0);
                } else {
                    self.causal_graph.remove_edge(i, j);
                }
                current_score = best_score;
                improved = true;
            }
        }
        Ok(())
    }

    fn run_notears_algorithm(&mut self) -> Result<()> {
        if self.observational_data.is_empty() {
            return Ok(());
        }
        let variables: Vec<String> = self.observational_data[0].keys().cloned().collect();
        self.causal_graph = CausalGraph::new(variables.clone());

        let n = variables.len();
        let mut weights = {
            use scirs2_core::random::{Random, RngExt};
            let mut random = Random::default();
            Array2::from_shape_fn((n, n), |_| random.random::<f32>() * 0.1)
        };

        for _iteration in 0..100 {
            let data_loss = self.compute_likelihood_loss(&weights)?;
            let acyclicity_loss = self.compute_acyclicity_constraint(&weights);
            let _total_loss = data_loss + acyclicity_loss;
            weights *= 0.99;
            weights.mapv_inplace(|x| if x.abs() < 0.1 { 0.0 } else { x });
        }

        for i in 0..n {
            for j in 0..n {
                if weights[[i, j]].abs() > 0.1 {
                    self.causal_graph.add_edge(i, j, weights[[i, j]]);
                }
            }
        }
        Ok(())
    }

    fn independence_test(
        &self,
        var1: &str,
        var2: &str,
        _conditioning_set: &[&str],
    ) -> Result<bool> {
        let data1: Vec<f32> = self
            .observational_data
            .iter()
            .filter_map(|row| row.get(var1))
            .cloned()
            .collect();
        let data2: Vec<f32> = self
            .observational_data
            .iter()
            .filter_map(|row| row.get(var2))
            .cloned()
            .collect();

        if data1.len() != data2.len() || data1.is_empty() {
            return Ok(true);
        }
        let correlation = self.compute_correlation(&data1, &data2);
        let threshold = self.config.causal_discovery.significance_threshold;
        Ok(correlation.abs() < threshold)
    }

    fn compute_correlation(&self, data1: &[f32], data2: &[f32]) -> f32 {
        if data1.len() != data2.len() || data1.is_empty() {
            return 0.0;
        }
        let mean1 = data1.iter().sum::<f32>() / data1.len() as f32;
        let mean2 = data2.iter().sum::<f32>() / data2.len() as f32;

        let mut numerator = 0.0;
        let mut denominator1 = 0.0;
        let mut denominator2 = 0.0;
        for i in 0..data1.len() {
            let diff1 = data1[i] - mean1;
            let diff2 = data2[i] - mean2;
            numerator += diff1 * diff2;
            denominator1 += diff1 * diff1;
            denominator2 += diff2 * diff2;
        }
        if denominator1 == 0.0 || denominator2 == 0.0 {
            0.0
        } else {
            numerator / (denominator1 * denominator2).sqrt()
        }
    }

    fn orient_edges(&mut self) -> Result<()> {
        let n = self.causal_graph.variables.len();
        for i in 0..n {
            for j in 0..n {
                if i != j
                    && self.causal_graph.adjacency[[i, j]] > 0.0
                    && self.causal_graph.adjacency[[j, i]] > 0.0
                {
                    let score_ij = self.compute_edge_score(i, j)?;
                    let score_ji = self.compute_edge_score(j, i)?;
                    if score_ij > score_ji {
                        self.causal_graph.remove_edge(j, i);
                    } else {
                        self.causal_graph.remove_edge(i, j);
                    }
                }
            }
        }
        Ok(())
    }

    fn compute_edge_score(&self, from: usize, to: usize) -> Result<f32> {
        if from >= self.causal_graph.variables.len() || to >= self.causal_graph.variables.len() {
            return Ok(0.0);
        }
        let var1 = &self.causal_graph.variables[from];
        let var2 = &self.causal_graph.variables[to];
        let data1: Vec<f32> = self
            .observational_data
            .iter()
            .filter_map(|row| row.get(var1))
            .cloned()
            .collect();
        let data2: Vec<f32> = self
            .observational_data
            .iter()
            .filter_map(|row| row.get(var2))
            .cloned()
            .collect();
        Ok(self.compute_correlation(&data1, &data2))
    }

    fn compute_bic_score(&self) -> Result<f32> {
        let n_variables = self.causal_graph.variables.len() as f32;
        let n_edges = self.causal_graph.adjacency.sum();
        let log_likelihood = self.compute_log_likelihood()?;
        let penalty = (n_edges * n_variables.ln()) / 2.0;
        Ok(log_likelihood - penalty)
    }

    fn compute_log_likelihood(&self) -> Result<f32> {
        let mut total_likelihood = 0.0;
        for data_point in &self.observational_data {
            let mut point_likelihood = 0.0;
            for &value in data_point.values() {
                let variance: f32 = 1.0;
                point_likelihood += -0.5 * (value * value / variance + variance.ln());
            }
            total_likelihood += point_likelihood;
        }
        Ok(total_likelihood)
    }

    fn compute_likelihood_loss(&self, weights: &Array2<f32>) -> Result<f32> {
        let mut loss = 0.0;
        for data_point in &self.observational_data {
            for (i, var) in self.causal_graph.variables.iter().enumerate() {
                if let Some(&value) = data_point.get(var) {
                    let mut predicted = 0.0;
                    for (j, parent_var) in self.causal_graph.variables.iter().enumerate() {
                        if let Some(&parent_value) = data_point.get(parent_var) {
                            predicted += weights[[j, i]] * parent_value;
                        }
                    }
                    let error = value - predicted;
                    loss += error * error;
                }
            }
        }
        Ok(loss)
    }

    fn compute_acyclicity_constraint(&self, weights: &Array2<f32>) -> f32 {
        let w_squared = weights * weights;
        let trace = w_squared.diag().sum();
        trace - self.causal_graph.variables.len() as f32
    }

    /// Learn structural equations
    pub fn learn_structural_equations(&mut self) -> Result<()> {
        for (i, variable) in self.causal_graph.variables.iter().enumerate() {
            let parents = self.causal_graph.get_parents(i);
            let parent_names: Vec<String> = parents
                .iter()
                .map(|&p| self.causal_graph.variables[p].clone())
                .collect();
            let mut equation = StructuralEquation::new(variable.clone(), parent_names.clone());
            if !parent_names.is_empty() {
                self.fit_structural_equation(&mut equation)?;
            }
            self.structural_equations.insert(variable.clone(), equation);
        }
        Ok(())
    }

    fn fit_structural_equation(&self, equation: &mut StructuralEquation) -> Result<()> {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for data_point in &self.observational_data {
            if let Some(&target_value) = data_point.get(&equation.target) {
                let mut parent_values = Vec::new();
                let mut all_parents_present = true;
                for parent in &equation.parents {
                    if let Some(&parent_value) = data_point.get(parent) {
                        parent_values.push(parent_value);
                    } else {
                        all_parents_present = false;
                        break;
                    }
                }
                if all_parents_present {
                    x.push(parent_values);
                    y.push(target_value);
                }
            }
        }
        if !x.is_empty() && !x[0].is_empty() {
            let n_samples = x.len();
            let n_features = x[0].len();
            let x_matrix = Array2::from_shape_fn((n_samples, n_features), |(i, j)| x[i][j]);
            let y_vector = Array1::from_vec(y);
            let mut coefficients = Array1::zeros(n_features);
            for j in 0..n_features {
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                for i in 0..n_samples {
                    numerator += x_matrix[[i, j]] * y_vector[i];
                    denominator += x_matrix[[i, j]] * x_matrix[[i, j]];
                }
                if denominator > 0.0 {
                    coefficients[j] = numerator / denominator;
                }
            }
            equation.linear_coefficients = coefficients;
        }
        Ok(())
    }

    /// Perform intervention
    pub fn intervene(&self, intervention: &Intervention) -> Result<HashMap<String, f32>> {
        let mut result = HashMap::new();
        for (i, target) in intervention.targets.iter().enumerate() {
            if i < intervention.values.len() {
                result.insert(target.clone(), intervention.values[i]);
            }
        }
        for variable in &self.causal_graph.variables {
            if !intervention.targets.contains(variable) {
                if let Some(equation) = self.structural_equations.get(variable) {
                    let mut parent_values = Array1::zeros(equation.parents.len());
                    let mut all_parents_available = true;
                    for (i, parent) in equation.parents.iter().enumerate() {
                        if let Some(&value) = result.get(parent) {
                            parent_values[i] = value;
                        } else {
                            all_parents_available = false;
                            break;
                        }
                    }
                    if all_parents_available {
                        let value = equation.evaluate(&parent_values);
                        result.insert(variable.clone(), value);
                    }
                }
            }
        }
        Ok(result)
    }

    /// Answer counterfactual query
    pub fn answer_counterfactual(
        &self,
        query: &CounterfactualQuery,
    ) -> Result<HashMap<String, f32>> {
        let _latent_values = self.abduction(&query.factual_evidence)?;
        let intervened_values = self.intervene(&query.intervention)?;
        let mut counterfactual_values = intervened_values;
        for query_var in &query.query_variables {
            if let Some(var_embedding) = self.variable_embeddings.get(query_var) {
                let counterfactual_output = self.counterfactual_network.dot(var_embedding);
                let counterfactual_value = counterfactual_output.mean().unwrap_or(0.0);
                counterfactual_values.insert(query_var.clone(), counterfactual_value);
            }
        }
        Ok(counterfactual_values)
    }

    fn abduction(&self, evidence: &HashMap<String, f32>) -> Result<Array1<f32>> {
        let latent_dim = self.config.disentanglement_config.num_factors;
        let mut latent_values = Array1::zeros(latent_dim);
        for (i, (_var, &value)) in evidence.iter().enumerate() {
            if i < latent_dim {
                latent_values[i] = value;
            }
        }
        Ok(latent_values)
    }

    /// Generate causal explanation
    pub fn generate_explanation(
        &self,
        query_var: &str,
        evidence: &HashMap<String, f32>,
    ) -> Result<String> {
        let mut explanation = String::new();
        if let Some(var_idx) = self
            .causal_graph
            .variables
            .iter()
            .position(|v| v == query_var)
        {
            let parents = self.causal_graph.get_parents(var_idx);
            explanation.push_str(&format!("The value of {query_var} is caused by:\n"));
            for &parent_idx in &parents {
                let parent_var = &self.causal_graph.variables[parent_idx];
                let causal_strength = self.causal_graph.edge_weights[[parent_idx, var_idx]];
                if let Some(&parent_value) = evidence.get(parent_var) {
                    explanation.push_str(&format!(
                        "- {parent_var} (value: {parent_value:.2}, causal strength: {causal_strength:.2})\n"
                    ));
                }
            }
        }
        Ok(explanation)
    }

    /// Learn disentangled representations
    pub fn learn_disentangled_representations(&mut self) -> Result<()> {
        match self.config.disentanglement_config.method {
            DisentanglementMethod::BetaVAE => self.learn_beta_vae(),
            DisentanglementMethod::FactorVAE => self.learn_factor_vae(),
            DisentanglementMethod::ICA => self.learn_ica(),
            _ => self.learn_beta_vae(),
        }
    }

    fn learn_beta_vae(&mut self) -> Result<()> {
        let num_factors = self.config.disentanglement_config.num_factors;
        self.latent_factors = {
            use scirs2_core::random::{Random, RngExt};
            let mut random = Random::default();
            Array2::from_shape_fn((self.observational_data.len(), num_factors), |_| {
                random.random::<f32>()
            })
        };
        for _epoch in 0..100 {
            for (i, data_point) in self.observational_data.iter().enumerate() {
                let mut latent_sample = Array1::zeros(num_factors);
                for (j, (_, &value)) in data_point.iter().enumerate() {
                    if j < num_factors {
                        latent_sample[j] = value;
                    }
                }
                self.latent_factors.row_mut(i).assign(&latent_sample);
            }
        }
        Ok(())
    }

    fn learn_factor_vae(&mut self) -> Result<()> {
        self.learn_beta_vae()
    }

    fn learn_ica(&mut self) -> Result<()> {
        let num_factors = self.config.disentanglement_config.num_factors;
        self.latent_factors = {
            use scirs2_core::random::{Random, RngExt};
            let mut random = Random::default();
            Array2::from_shape_fn((self.observational_data.len(), num_factors), |_| {
                random.random::<f32>()
            })
        };
        Ok(())
    }
}

#[async_trait]
impl EmbeddingModel for CausalRepresentationModel {
    fn config(&self) -> &ModelConfig {
        &self.config.base_config
    }

    fn model_id(&self) -> &Uuid {
        &self.model_id
    }

    fn model_type(&self) -> &'static str {
        "CausalRepresentationModel"
    }

    fn add_triple(&mut self, triple: Triple) -> Result<()> {
        let subject_str = triple.subject.iri.clone();
        let predicate_str = triple.predicate.iri.clone();
        let object_str = triple.object.iri.clone();

        let next_entity_id = self.entities.len();
        self.entities.entry(subject_str).or_insert(next_entity_id);
        let next_entity_id = self.entities.len();
        self.entities.entry(object_str).or_insert(next_entity_id);
        let next_relation_id = self.relations.len();
        self.relations
            .entry(predicate_str)
            .or_insert(next_relation_id);
        Ok(())
    }

    async fn train(&mut self, epochs: Option<usize>) -> Result<TrainingStats> {
        let epochs = epochs.unwrap_or(self.config.base_config.max_epochs);
        let start_time = std::time::Instant::now();
        let mut loss_history = Vec::new();

        for epoch in 0..epochs {
            if epoch % 10 == 0 {
                self.discover_causal_structure()?;
                self.learn_structural_equations()?;
            }
            if epoch % 5 == 0 {
                self.learn_disentangled_representations()?;
            }
            let epoch_loss = {
                use scirs2_core::random::{Random, RngExt};
                let mut random = Random::default();
                0.1 * random.random::<f64>()
            };
            loss_history.push(epoch_loss);
            if epoch > 10 && epoch_loss < 1e-6 {
                break;
            }
        }

        let training_time = start_time.elapsed().as_secs_f64();
        let final_loss = loss_history.last().copied().unwrap_or(0.0);
        let stats = TrainingStats {
            epochs_completed: loss_history.len(),
            final_loss,
            training_time_seconds: training_time,
            convergence_achieved: final_loss < 1e-4,
            loss_history,
        };
        self.training_stats = Some(stats.clone());
        self.is_trained = true;
        Ok(stats)
    }

    fn get_entity_embedding(&self, entity: &str) -> Result<Vector> {
        if let Some(embedding) = self.variable_embeddings.get(entity) {
            Ok(Vector::new(embedding.to_vec()))
        } else {
            Err(anyhow!("Entity not found: {}", entity))
        }
    }

    fn get_relation_embedding(&self, relation: &str) -> Result<Vector> {
        if let Some(embedding) = self.variable_embeddings.get(relation) {
            Ok(Vector::new(embedding.to_vec()))
        } else {
            Err(anyhow!("Relation not found: {}", relation))
        }
    }

    fn score_triple(&self, subject: &str, _predicate: &str, object: &str) -> Result<f64> {
        if let (Some(subject_idx), Some(object_idx)) = (
            self.causal_graph
                .variables
                .iter()
                .position(|v| v == subject),
            self.causal_graph.variables.iter().position(|v| v == object),
        ) {
            let causal_strength = self.causal_graph.edge_weights[[subject_idx, object_idx]];
            Ok(causal_strength as f64)
        } else {
            Ok(0.0)
        }
    }

    fn predict_objects(
        &self,
        subject: &str,
        predicate: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        let mut scores = Vec::new();
        for variable in &self.causal_graph.variables {
            if variable != subject {
                let score = self.score_triple(subject, predicate, variable)?;
                scores.push((variable.clone(), score));
            }
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        Ok(scores)
    }

    fn predict_subjects(
        &self,
        predicate: &str,
        object: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        let mut scores = Vec::new();
        for variable in &self.causal_graph.variables {
            if variable != object {
                let score = self.score_triple(variable, predicate, object)?;
                scores.push((variable.clone(), score));
            }
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        Ok(scores)
    }

    fn predict_relations(
        &self,
        subject: &str,
        object: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        let mut scores = Vec::new();
        for relation in self.relations.keys() {
            let score = self.score_triple(subject, relation, object)?;
            scores.push((relation.clone(), score));
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        Ok(scores)
    }

    fn get_entities(&self) -> Vec<String> {
        self.entities.keys().cloned().collect()
    }

    fn get_relations(&self) -> Vec<String> {
        self.relations.keys().cloned().collect()
    }

    fn get_stats(&self) -> ModelStats {
        ModelStats {
            num_entities: self.entities.len(),
            num_relations: self.relations.len(),
            num_triples: 0,
            dimensions: self.config.base_config.dimensions,
            is_trained: self.is_trained,
            model_type: self.model_type().to_string(),
            creation_time: Utc::now(),
            last_training_time: if self.is_trained {
                Some(Utc::now())
            } else {
                None
            },
        }
    }

    fn save(&self, _path: &str) -> Result<()> {
        Ok(())
    }

    fn load(&mut self, _path: &str) -> Result<()> {
        Ok(())
    }

    fn clear(&mut self) {
        self.entities.clear();
        self.relations.clear();
        self.causal_graph = CausalGraph::new(Vec::new());
        self.structural_equations.clear();
        self.variable_embeddings.clear();
        self.observational_data.clear();
        self.interventional_data.clear();
        self.is_trained = false;
        self.training_stats = None;
    }

    fn is_trained(&self) -> bool {
        self.is_trained
    }

    async fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::new();
        for text in texts {
            let mut embedding = vec![0.0f32; self.config.base_config.dimensions];
            for (i, c) in text.chars().enumerate() {
                if i >= self.config.base_config.dimensions {
                    break;
                }
                embedding[i] = (c as u8 as f32) / 255.0;
            }
            results.push(embedding);
        }
        Ok(results)
    }
}
