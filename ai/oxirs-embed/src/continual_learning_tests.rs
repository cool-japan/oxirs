use crate::continual_learning_types::{
    ContinualLearningConfig, ContinualLearningModel, EWCState, MemoryConfig, MemoryEntry,
    MemoryType, MemoryUpdateStrategy, TaskInfo,
};
use crate::ModelConfig;
use scirs2_core::ndarray_ext::Array1;
use scirs2_core::ndarray_ext::Array2;
use scirs2_core::random::{Random, RngExt};

#[test]
fn test_continual_learning_config_default() {
    let config = ContinualLearningConfig::default();
    assert!(matches!(
        config.memory_config.memory_type,
        MemoryType::EpisodicMemory
    ));
    assert_eq!(config.memory_config.memory_capacity, 10000);
}

#[test]
fn test_task_info_creation() {
    let task = TaskInfo::new("task1".to_string(), "classification".to_string());
    assert_eq!(task.task_id, "task1");
    assert_eq!(task.task_type, "classification");
    assert_eq!(task.examples_seen, 0);
}

#[test]
fn test_memory_entry_creation() {
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let target = Array1::from_vec(vec![0.0, 1.0]);
    let entry = MemoryEntry::new(data, target, "task1".to_string());

    assert_eq!(entry.task_id, "task1");
    assert_eq!(entry.importance, 1.0);
    assert_eq!(entry.access_count, 0);
}

#[test]
fn test_continual_learning_model_creation() {
    let config = ContinualLearningConfig::default();
    let model = ContinualLearningModel::new(config);

    assert_eq!(model.entities.len(), 0);
    assert_eq!(model.examples_seen, 0);
    assert!(model.current_task.is_none());
}

#[tokio::test]
async fn test_task_management() {
    let config = ContinualLearningConfig::default();
    let mut model = ContinualLearningModel::new(config);

    model
        .start_task("task1".to_string(), "test".to_string())
        .expect("should succeed");
    assert!(model.current_task.is_some());
    assert_eq!(
        model.current_task.as_ref().expect("should succeed").task_id,
        "task1"
    );

    model
        .start_task("task2".to_string(), "test".to_string())
        .expect("should succeed");
    assert_eq!(model.task_history.len(), 1);
    assert_eq!(
        model.current_task.as_ref().expect("should succeed").task_id,
        "task2"
    );
}

#[tokio::test]
async fn test_add_example() {
    let config = ContinualLearningConfig {
        base_config: ModelConfig {
            dimensions: 3,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut model = ContinualLearningModel::new(config);

    model
        .start_task("task1".to_string(), "test".to_string())
        .expect("should succeed");

    let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let target = Array1::from_vec(vec![1.0, 2.0, 3.0]);

    model
        .add_example(data, target, Some("task1".to_string()))
        .await
        .expect("should succeed");

    assert_eq!(model.examples_seen, 1);
    assert_eq!(model.episodic_memory.len(), 1);
    assert_eq!(
        model
            .current_task
            .as_ref()
            .expect("should succeed")
            .examples_seen,
        1
    );
}

#[tokio::test]
async fn test_memory_management() {
    let config = ContinualLearningConfig {
        memory_config: MemoryConfig {
            memory_capacity: 3,
            update_strategy: MemoryUpdateStrategy::FIFO,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut model = ContinualLearningModel::new(config);
    model
        .start_task("task1".to_string(), "test".to_string())
        .expect("should succeed");

    for i in 0..5 {
        let data = Array1::from_vec(vec![i as f32]);
        let target = Array1::from_vec(vec![i as f32]);
        model
            .add_example(data, target, Some("task1".to_string()))
            .await
            .expect("should succeed");
    }

    assert_eq!(model.episodic_memory.len(), 3);
}

#[tokio::test]
async fn test_continual_training() {
    let config = ContinualLearningConfig {
        base_config: ModelConfig {
            dimensions: 3,
            max_epochs: 10,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut model = ContinualLearningModel::new(config);

    model
        .start_task("initial_task".to_string(), "training".to_string())
        .expect("should succeed");

    use crate::EmbeddingModel;
    let stats = model.train(Some(10)).await.expect("should succeed");
    assert_eq!(stats.epochs_completed, 10);
    assert!(model.is_trained());
    assert!(!model.task_history.is_empty());
}

#[test]
fn test_forgetting_evaluation() {
    let config = ContinualLearningConfig::default();
    let model = ContinualLearningModel::new(config);

    let forgetting = model.evaluate_forgetting();
    assert_eq!(forgetting, 0.0);
}

#[test]
fn test_ewc_state_creation() {
    let mut random = Random::default();
    let fisher = Array2::from_shape_fn((5, 5), |_| random.random::<f32>());
    let params = Array2::from_shape_fn((5, 5), |_| random.random::<f32>());

    let ewc_state = EWCState {
        fisher_information: fisher,
        optimal_parameters: params,
        task_id: "task1".to_string(),
        importance: 1.0,
    };

    assert_eq!(ewc_state.task_id, "task1");
    assert_eq!(ewc_state.importance, 1.0);
}
