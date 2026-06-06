use crate::continual_learning_types::{
    ArchitectureAdaptation, BoundaryDetection, ContinualLearningModel, MemoryEntry,
    MemoryUpdateStrategy, RegularizationMethod, ReplayMethod, TaskDetection,
};
use anyhow::Result;
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::{Random, RngExt};

impl ContinualLearningModel {
    pub(crate) fn add_to_memory(
        &mut self,
        data: Array1<f32>,
        target: Array1<f32>,
        task_id: String,
    ) -> Result<()> {
        let mut random = Random::default();
        let entry = MemoryEntry::new(data, target, task_id);

        match self.config.memory_config.update_strategy {
            MemoryUpdateStrategy::FIFO => {
                if self.episodic_memory.len() >= self.config.memory_config.memory_capacity {
                    self.episodic_memory.pop_front();
                }
                self.episodic_memory.push_back(entry);
            }
            MemoryUpdateStrategy::Random => {
                if self.episodic_memory.len() >= self.config.memory_config.memory_capacity {
                    let idx = random.random_range(0..self.episodic_memory.len());
                    self.episodic_memory.remove(idx);
                }
                self.episodic_memory.push_back(entry);
            }
            MemoryUpdateStrategy::ReservoirSampling => {
                if self.episodic_memory.len() < self.config.memory_config.memory_capacity {
                    self.episodic_memory.push_back(entry);
                } else {
                    let k = self.episodic_memory.len();
                    let j = random.random_range(0..self.examples_seen + 1);
                    if j < k {
                        self.episodic_memory[j] = entry;
                    }
                }
            }
            MemoryUpdateStrategy::ImportanceBased => {
                self.add_by_importance(entry)?;
            }
            _ => {
                self.episodic_memory.push_back(entry);
            }
        }

        Ok(())
    }

    pub(crate) fn add_by_importance(&mut self, entry: MemoryEntry) -> Result<()> {
        if self.episodic_memory.len() < self.config.memory_config.memory_capacity {
            self.episodic_memory.push_back(entry);
        } else {
            let mut min_importance = f32::INFINITY;
            let mut min_idx = 0;

            for (i, existing_entry) in self.episodic_memory.iter().enumerate() {
                if existing_entry.importance < min_importance {
                    min_importance = existing_entry.importance;
                    min_idx = i;
                }
            }

            if entry.importance > min_importance {
                self.episodic_memory[min_idx] = entry;
            }
        }

        Ok(())
    }

    pub(crate) fn detect_task_boundary(&self, data: &Array1<f32>) -> Result<bool> {
        match self.config.task_config.boundary_detection {
            BoundaryDetection::ChangePoint => self.detect_change_point(data),
            BoundaryDetection::DistributionShift => self.detect_distribution_shift(data),
            BoundaryDetection::LossBased => self.detect_loss_change(data),
            BoundaryDetection::GradientBased => self.detect_gradient_change(data),
        }
    }

    fn detect_change_point(&self, _data: &Array1<f32>) -> Result<bool> {
        if self.examples_seen % 1000 == 0 && self.examples_seen > 0 {
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn detect_distribution_shift(&self, data: &Array1<f32>) -> Result<bool> {
        if self.episodic_memory.is_empty() {
            return Ok(false);
        }

        let recent_count = 100.min(self.episodic_memory.len());
        let mut total_distance = 0.0;

        for i in 0..recent_count {
            let idx = self.episodic_memory.len() - 1 - i;
            let recent_data = &self.episodic_memory[idx].data;
            let distance = self.euclidean_distance(data, recent_data);
            total_distance += distance;
        }

        let average_distance = total_distance / recent_count as f32;
        let threshold = 2.0;

        Ok(average_distance > threshold)
    }

    fn detect_loss_change(&self, _data: &Array1<f32>) -> Result<bool> {
        Ok(false)
    }

    fn detect_gradient_change(&self, _data: &Array1<f32>) -> Result<bool> {
        Ok(false)
    }

    pub(crate) fn apply_regularization(&self, mut gradients: Array2<f32>) -> Result<Array2<f32>> {
        for method in &self.config.regularization_config.methods {
            match method {
                RegularizationMethod::EWC => {
                    gradients = self.apply_ewc_regularization(gradients)?;
                }
                RegularizationMethod::SynapticIntelligence => {
                    gradients = self.apply_si_regularization(gradients)?;
                }
                RegularizationMethod::LwF => {
                    gradients = self.apply_lwf_regularization(gradients)?;
                }
                _ => {}
            }
        }

        Ok(gradients)
    }

    fn apply_ewc_regularization(&self, mut gradients: Array2<f32>) -> Result<Array2<f32>> {
        let lambda = self.config.regularization_config.ewc_config.lambda;

        for ewc_state in &self.ewc_states {
            let penalty = &ewc_state.fisher_information
                * (&self.embeddings - &ewc_state.optimal_parameters)
                * lambda
                * ewc_state.importance;

            let rows_to_update = gradients.nrows().min(penalty.nrows());
            let cols_to_update = gradients.ncols().min(penalty.ncols());

            for i in 0..rows_to_update {
                for j in 0..cols_to_update {
                    gradients[[i, j]] -= penalty[[i, j]];
                }
            }
        }

        Ok(gradients)
    }

    fn apply_si_regularization(&self, mut gradients: Array2<f32>) -> Result<Array2<f32>> {
        let c = self.config.regularization_config.si_config.c;

        if !self.synaptic_importance.is_empty() {
            let penalty = &self.synaptic_importance * c;

            let rows_to_update = gradients.nrows().min(penalty.nrows());
            let cols_to_update = gradients.ncols().min(penalty.ncols());

            for i in 0..rows_to_update {
                for j in 0..cols_to_update {
                    gradients[[i, j]] -= penalty[[i, j]];
                }
            }
        }

        Ok(gradients)
    }

    fn apply_lwf_regularization(&self, gradients: Array2<f32>) -> Result<Array2<f32>> {
        Ok(gradients)
    }

    pub(crate) fn compute_ewc_state(&mut self) -> Result<()> {
        use crate::continual_learning_types::EWCState;

        if let Some(ref current_task) = self.current_task {
            let mut fisher_information = Array2::zeros(self.embeddings.dim());

            for entry in &self.episodic_memory {
                if entry.task_id == current_task.task_id {
                    let gradients = self.compute_gradients(&entry.data, &entry.target)?;

                    let rows_to_update = gradients.nrows().min(fisher_information.nrows());
                    let cols_to_update = gradients.ncols().min(fisher_information.ncols());

                    for i in 0..rows_to_update {
                        for j in 0..cols_to_update {
                            fisher_information[[i, j]] += gradients[[i, j]] * gradients[[i, j]];
                        }
                    }
                }
            }

            let task_examples = self
                .episodic_memory
                .iter()
                .filter(|entry| entry.task_id == current_task.task_id)
                .count() as f32;

            if task_examples > 0.0 {
                fisher_information /= task_examples;
            }

            let ewc_state = EWCState {
                fisher_information,
                optimal_parameters: self.embeddings.clone(),
                task_id: current_task.task_id.clone(),
                importance: 1.0,
            };

            self.ewc_states.push(ewc_state);
        }

        Ok(())
    }

    pub(crate) fn add_network_column(&mut self) -> Result<()> {
        let dimensions = self.config.base_config.dimensions;
        let mut random = Random::default();
        let new_column =
            Array2::from_shape_fn((dimensions, dimensions), |_| random.random::<f32>() * 0.1);
        self.network_columns.push(new_column);

        if self.network_columns.len() > 1 {
            let lateral_connection = Array2::from_shape_fn((dimensions, dimensions), |_| {
                random.random::<f32>()
                    * self
                        .config
                        .architecture_config
                        .progressive_config
                        .lateral_strength
            });
            self.lateral_connections.push(lateral_connection);
        }

        Ok(())
    }

    pub(crate) async fn experience_replay(&mut self) -> Result<()> {
        if self.episodic_memory.is_empty() {
            return Ok(());
        }

        let mut random = Random::default();
        let replay_batch_size = (self.config.replay_config.replay_ratio * 32.0) as usize;
        let batch_size = replay_batch_size.min(self.episodic_memory.len());

        for _ in 0..batch_size {
            let idx = random.random_range(0..self.episodic_memory.len());

            let (data, target) = {
                let entry = &self.episodic_memory[idx];
                (entry.data.clone(), entry.target.clone())
            };

            self.episodic_memory[idx].access_count += 1;

            let gradients = self.compute_gradients(&data, &target)?;
            let regularized_gradients = self.apply_regularization(gradients)?;
            self.update_parameters(regularized_gradients)?;
        }

        Ok(())
    }

    pub(crate) async fn generative_replay(&mut self) -> Result<()> {
        if let Some(ref generator) = self.generator {
            let _replay_batch_size = (self.config.replay_config.replay_ratio * 32.0) as usize;
            let _generator_clone = generator.clone();
        }

        if let Some(generator) = self.generator.clone() {
            let replay_batch_size = (self.config.replay_config.replay_ratio * 32.0) as usize;

            for _ in 0..replay_batch_size {
                let mut random = Random::default();
                let noise = Array1::from_shape_fn(generator.ncols(), |_| random.random::<f32>());
                let generated_data = generator.dot(&noise);
                let generated_target = generated_data.mapv(|x| x.tanh());

                let gradients = self.compute_gradients(&generated_data, &generated_target)?;
                let regularized_gradients = self.apply_regularization(gradients)?;
                self.update_parameters(regularized_gradients)?;
            }
        }

        Ok(())
    }

    pub(crate) fn detect_is_automatic(&self) -> bool {
        matches!(
            self.config.task_config.detection_method,
            TaskDetection::Automatic
        )
    }

    pub(crate) fn is_progressive(&self) -> bool {
        matches!(
            self.config.architecture_config.adaptation_method,
            ArchitectureAdaptation::Progressive
        )
    }

    pub(crate) fn should_use_ewc(&self) -> bool {
        self.config
            .regularization_config
            .methods
            .contains(&RegularizationMethod::EWC)
    }

    pub(crate) fn should_use_si(&self) -> bool {
        self.config
            .regularization_config
            .methods
            .contains(&RegularizationMethod::SynapticIntelligence)
    }

    pub(crate) fn should_replay_experience(&self) -> bool {
        self.config
            .replay_config
            .methods
            .contains(&ReplayMethod::ExperienceReplay)
    }

    pub(crate) fn should_replay_generative(&self) -> bool {
        self.config
            .replay_config
            .methods
            .contains(&ReplayMethod::GenerativeReplay)
    }
}
