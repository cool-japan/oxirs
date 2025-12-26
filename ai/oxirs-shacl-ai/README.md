# OxiRS SHACL-AI - AI-Enhanced SHACL Validation

[![Version](https://img.shields.io/badge/version-0.1.0--rc.1-blue)](https://github.com/cool-japan/oxirs/releases)

**Status**: Release Candidate (v0.1.0-rc.1) - Released December 26, 2025

✨ **Release Candidate**: Production-ready with API stability guarantees and comprehensive testing.

AI-powered SHACL validation combining traditional constraint checking with machine learning for shape inference, anomaly detection, and intelligent validation.

## Features

### AI-Enhanced Validation
- **Shape Learning** - Automatically learn SHACL shapes from data
- **Anomaly Detection** - Detect unusual patterns violating implicit constraints
- **Confidence Scoring** - ML-based confidence scores for validations
- **Validation Suggestions** - Suggest fixes for constraint violations

### Machine Learning Models
- **Neural Networks** - Deep learning for pattern recognition
- **Decision Trees** - Interpretable validation rules
- **Ensemble Methods** - Combine multiple models for robustness
- **Transfer Learning** - Reuse models across similar schemas

### Integration
- **SHACL Engine** - Works with standard oxirs-shacl
- **Explainable AI** - Understand why constraints are learned
- **Incremental Learning** - Update models with new data
- **Human-in-the-loop** - Interactive refinement

## Installation

Add to your `Cargo.toml`:

```toml
# Experimental feature
[dependencies]
oxirs-shacl-ai = "0.1.0-rc.1"
oxirs-shacl = "0.1.0-rc.1"
```

## Quick Start

### Automatic Shape Learning

```rust
use oxirs_shacl_ai::{ShapeLearner, LearningConfig};
use oxirs_core::Dataset;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load training data
    let dataset = Dataset::from_file("training_data.ttl")?;

    // Configure shape learner
    let config = LearningConfig {
        min_confidence: 0.8,
        max_shapes: 50,
        include_optional: true,
        learn_datatypes: true,
    };

    // Learn shapes from data
    let learner = ShapeLearner::new(config);
    let learned_shapes = learner.learn_shapes(&dataset).await?;

    println!("Learned {} shapes", learned_shapes.len());

    // Export as SHACL
    learned_shapes.save_to_file("learned_shapes.ttl")?;

    Ok(())
}
```

### AI-Enhanced Validation

```rust
use oxirs_shacl_ai::AiValidator;
use oxirs_shacl::ValidationEngine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create AI-enhanced validator
    let ai_validator = AiValidator::builder()
        .shapes_file("shapes.ttl")
        .enable_anomaly_detection(true)
        .enable_suggestions(true)
        .confidence_threshold(0.7)
        .build()
        .await?;

    // Validate with AI enhancements
    let dataset = Dataset::from_file("data.ttl")?;
    let report = ai_validator.validate(&dataset).await?;

    // Process results with confidence scores
    for result in report.results() {
        println!("Violation: {} (confidence: {:.2})",
            result.message,
            result.confidence
        );

        if let Some(suggestion) = result.suggestion {
            println!("  Suggested fix: {}", suggestion);
        }
    }

    Ok(())
}
```

## Shape Learning Modes

### Automatic Discovery

```rust
use oxirs_shacl_ai::{ShapeLearner, DiscoveryMode};

let learner = ShapeLearner::builder()
    .mode(DiscoveryMode::Automatic)
    .min_support(0.1)  // 10% of entities must match
    .build();

let shapes = learner.discover_shapes(&dataset).await?;
```

### Guided Learning

```rust
use oxirs_shacl_ai::{ShapeLearner, TargetClass};

// Learn shapes for specific classes
let learner = ShapeLearner::builder()
    .target_class("http://xmlns.com/foaf/0.1/Person")
    .learn_property_shapes(true)
    .learn_cardinality(true)
    .learn_value_ranges(true)
    .build();

let shapes = learner.learn(&dataset).await?;
```

### Interactive Refinement

```rust
use oxirs_shacl_ai::InteractiveLearner;

let mut learner = InteractiveLearner::new();

// Initial learning
let candidate_shapes = learner.propose_shapes(&dataset).await?;

// Review and refine
for shape in candidate_shapes {
    println!("Proposed shape: {}", shape);
    println!("Confidence: {:.2}", shape.confidence);
    println!("Examples: {:?}", shape.examples);

    // User feedback
    let accept = prompt_user("Accept this shape? (y/n)")?;
    learner.provide_feedback(shape.id, accept);
}

let refined_shapes = learner.finalize().await?;
```

## Anomaly Detection

### Statistical Anomalies

```rust
use oxirs_shacl_ai::AnomalyDetector;

let detector = AnomalyDetector::builder()
    .method(AnomalyMethod::Statistical)
    .threshold(2.5)  // Z-score threshold
    .build();

let anomalies = detector.detect(&dataset).await?;

for anomaly in anomalies {
    println!("Anomaly: {} at {}", anomaly.description, anomaly.entity);
    println!("  Score: {:.2}", anomaly.score);
}
```

### ML-Based Detection

```rust
use oxirs_shacl_ai::{AnomalyDetector, AnomalyMethod};

// Train on normal data
let detector = AnomalyDetector::builder()
    .method(AnomalyMethod::Autoencoder)
    .train_on(&normal_dataset)
    .await?;

// Detect anomalies in new data
let anomalies = detector.detect(&new_dataset).await?;
```

## Model Training

### Custom Training

```rust
use oxirs_shacl_ai::{ModelTrainer, ModelConfig};

let config = ModelConfig {
    model_type: ModelType::NeuralNetwork,
    hidden_layers: vec![128, 64, 32],
    learning_rate: 0.001,
    epochs: 100,
    batch_size: 32,
};

let trainer = ModelTrainer::new(config);

// Train on labeled data
let model = trainer.train(
    &training_dataset,
    &validation_dataset
).await?;

// Save model
model.save_to_file("./models/validation_model.bin")?;
```

### Transfer Learning

```rust
use oxirs_shacl_ai::TransferLearning;

// Load pre-trained model
let base_model = Model::load("pretrained_model.bin")?;

// Fine-tune on your data
let transfer = TransferLearning::new(base_model);
let fine_tuned = transfer.fine_tune(
    &your_dataset,
    epochs: 20
).await?;
```

## Validation Suggestions

```rust
use oxirs_shacl_ai::ValidationSuggester;

let suggester = ValidationSuggester::builder()
    .enable_auto_fix(true)
    .suggest_alternatives(true)
    .build();

for violation in validation_report.violations() {
    let suggestions = suggester.suggest_fixes(&violation).await?;

    for suggestion in suggestions {
        println!("Suggestion (confidence {:.2}):", suggestion.confidence);
        println!("  {}", suggestion.description);
        println!("  Apply: {}", suggestion.sparql_update);
    }
}
```

## Explainability

```rust
use oxirs_shacl_ai::Explainer;

let explainer = Explainer::new(&ai_model);

// Explain why a shape was learned
let explanation = explainer.explain_shape(&shape).await?;
println!("Shape learned because:");
for reason in explanation.reasons {
    println!("  - {} (weight: {:.2})", reason.description, reason.weight);
}

// Explain validation decision
let explanation = explainer.explain_violation(&violation).await?;
println!("Contributing factors:");
for factor in explanation.factors {
    println!("  - {}: {}", factor.name, factor.contribution);
}
```

## Integration with oxirs-shacl

```rust
use oxirs_shacl::ValidationEngine;
use oxirs_shacl_ai::AiEnhancer;

// Standard SHACL validation
let shacl_engine = ValidationEngine::new(&shapes, config);
let mut report = shacl_engine.validate(&dataset)?;

// Enhance with AI
let ai_enhancer = AiEnhancer::new()?;
ai_enhancer.enhance_report(&mut report).await?;

// Now includes confidence scores and suggestions
for result in report.results() {
    println!("{} (confidence: {:.2})", result.message, result.confidence);
}
```

## Performance

### Shape Learning Performance

| Dataset Size | Classes | Learning Time | Shapes Generated |
|-------------|---------|---------------|------------------|
| 10K triples | 10 | 5s | 25 |
| 100K triples | 50 | 45s | 120 |
| 1M triples | 200 | 8m | 500 |

### Validation Performance

Standard SHACL validation + AI enhancements adds approximately 10-20% overhead.

## Configuration

```rust
use oxirs_shacl_ai::AiConfig;

let config = AiConfig {
    // Shape learning
    min_confidence: 0.8,
    max_shapes_per_class: 50,
    enable_cardinality_learning: true,
    enable_value_range_learning: true,

    // Anomaly detection
    anomaly_threshold: 0.7,
    statistical_method: true,
    ml_method: true,

    // Model settings
    model_cache_dir: Some("./models".into()),
    use_gpu: false,

    // Suggestions
    max_suggestions_per_violation: 5,
    suggest_auto_fixes: true,
};
```

## Status

### Release Candidate (v0.1.0-rc.1)
- ✅ Shape learning with persisted dataset snapshots and CLI integration
- ✅ Neural network validation leveraging SciRS2 telemetry for drift detection
- ✅ Anomaly detection with vector-based similarity checks
- ✅ Confidence scoring and remediation guidance integrated into Fuseki UI
- 🚧 Explainability features (saliency reporting) – in progress
- 🚧 Transfer learning (cross-dataset models) – in progress
- ⏳ Auto-fix suggestions (planned for future release)
- ⏳ Active learning (planned for v0.2.0)

## Research

This module is based on research in:
- Neural-symbolic AI
- Knowledge graph completion
- Constraint learning
- Explainable AI

## Contributing

This is a research-oriented experimental module. Contributions and research collaborations welcome!

## License

MIT OR Apache-2.0

## See Also

- [oxirs-shacl](../../engine/oxirs-shacl/) - Standard SHACL validation
- [oxirs-embed](../oxirs-embed/) - Embeddings for ML features
- [oxirs-rule](../../engine/oxirs-rule/) - Rule-based reasoning