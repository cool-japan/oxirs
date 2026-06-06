//! Evolutionary NAS — Tests

#[cfg(test)]
mod tests {
    use crate::evolutionary_nas_eval::should_stop_early;
    use crate::evolutionary_nas_evolution::EvolutionaryNAS;
    use crate::evolutionary_nas_types::{
        ArchitectureCandidate, ArchitectureGenome, EvolutionaryConfig, FitnessScores,
        GlobalParameters, HardwareMetrics, PerformanceMetrics,
    };
    use uuid::Uuid;

    #[tokio::test]
    async fn test_evolutionary_nas_creation() {
        let config = EvolutionaryConfig::default();
        let nas = EvolutionaryNAS::new(config);
        assert!(nas.is_ok());
    }

    #[tokio::test]
    async fn test_population_initialization() {
        let config = EvolutionaryConfig {
            population_size: 10,
            ..Default::default()
        };
        let mut nas = EvolutionaryNAS::new(config).expect("should succeed");

        let result = nas.initialize_population().await;
        assert!(result.is_ok());

        let population = nas.population.read().await;
        assert_eq!(population.len(), 10);
    }

    #[tokio::test]
    async fn test_genome_distance_calculation() {
        let config = EvolutionaryConfig::default();
        let mut nas = EvolutionaryNAS::new(config).expect("should succeed");

        let candidate1 = nas.generate_random_candidate(0).expect("should succeed");
        let candidate2 = nas.generate_random_candidate(1).expect("should succeed");

        let distance = nas.calculate_genome_distance(&candidate1.genome, &candidate2.genome);
        assert!(distance.is_ok());
        assert!(distance.expect("should succeed") >= 0.0);
    }

    #[tokio::test]
    async fn test_fitness_calculation() {
        let config = EvolutionaryConfig::default();
        let nas = EvolutionaryNAS::new(config).expect("should succeed");

        let candidate = ArchitectureCandidate {
            id: Uuid::new_v4(),
            genome: ArchitectureGenome {
                nodes: Vec::new(),
                connections: Vec::new(),
                global_params: GlobalParameters::default(),
                modules: Vec::new(),
            },
            fitness: FitnessScores::default(),
            performance: Some(PerformanceMetrics {
                training_accuracy: 0.85,
                validation_accuracy: 0.82,
                test_accuracy: None,
                training_time: 300.0,
                inference_time_ms: 2.5,
                memory_usage_mb: 500.0,
                energy_consumption: Some(50.0),
                model_size: 1_000_000,
                flops: 5_000_000,
            }),
            generation: 0,
            parents: Vec::new(),
            novelty_score: 0.5,
            hardware_metrics: HardwareMetrics::default(),
        };

        let fitness = nas.calculate_fitness_scores(&candidate);
        assert!(fitness.is_ok());
        assert!(fitness.expect("should succeed").overall_fitness > 0.0);
    }

    #[tokio::test]
    async fn test_tournament_selection() {
        let config = EvolutionaryConfig::default();
        let nas = EvolutionaryNAS::new(config).expect("should succeed");

        let mut population = Vec::new();
        for i in 0..10 {
            let candidate = ArchitectureCandidate {
                id: Uuid::new_v4(),
                genome: ArchitectureGenome {
                    nodes: Vec::new(),
                    connections: Vec::new(),
                    global_params: GlobalParameters::default(),
                    modules: Vec::new(),
                },
                fitness: FitnessScores {
                    overall_fitness: i as f32 * 0.1,
                    ..Default::default()
                },
                performance: None,
                generation: 0,
                parents: Vec::new(),
                novelty_score: 0.0,
                hardware_metrics: HardwareMetrics::default(),
            };
            population.push(candidate);
        }

        let selected = nas.tournament_selection(&population);
        assert!(selected.is_ok());
        assert!(selected.expect("should succeed") < population.len());
    }

    #[test]
    fn test_early_stopping_triggered() {
        let history = vec![0.8, 0.80001, 0.80002, 0.80001, 0.80003];
        assert!(should_stop_early(&history, 4, 0.001));
    }

    #[test]
    fn test_early_stopping_not_triggered() {
        let history = vec![0.5, 0.6, 0.7, 0.8, 0.9];
        assert!(!should_stop_early(&history, 4, 0.001));
    }
}
