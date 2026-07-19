# OxiRS SHACL-AI - AI-Enhanced SHACL Validation

[![Version](https://img.shields.io/badge/version-0.4.0-blue)](https://github.com/cool-japan/oxirs/releases)

**Status**: v0.4.0 - Released 2026-07-19

✨ **Production Release**: Production-ready with API stability guarantees and comprehensive testing.

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
[dependencies]
oxirs-shacl-ai = "0.3.2"
oxirs-shacl = "0.3.2"

# Optional: network-based LLM providers (OpenAI, Anthropic) for the `explainer` /
# `shape_nl_generator` modules. Default features stay 100% offline.
oxirs-shacl-ai = { version = "0.3.2", features = ["llm-network"] }
```

## Quick Start

### Automatic Shape Learning

```rust
use oxirs_shacl_ai::{LearningConfig, ShapeLearner};
use oxirs_core::RdfStore;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load an RDF store (see oxirs-core for parsing Turtle/N-Triples/etc. into it)
    let store = RdfStore::new()?;

    // Configure the shape learner
    let config = LearningConfig {
        min_confidence: 0.8,
        max_shapes: 50,
        enable_training: true,
        ..Default::default()
    };
    let mut learner = ShapeLearner::with_config(config);

    // Learn shapes from data
    let learned_shapes = learner.learn_shapes_from_store(&store, None)?;
    println!("Learned {} shapes", learned_shapes.len());

    Ok(())
}
```

### AI-Enhanced Validation

The main entry point is `ShaclAiAssistant`, which wraps shape learning, quality
assessment, and insight generation over a `Store`:

```rust
use oxirs_shacl_ai::ShaclAiAssistant;
use oxirs_core::Store;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    # let store: &dyn Store = unimplemented!();
    let mut assistant = ShaclAiAssistant::new();

    // 1. Learn shapes from existing data
    let shapes = assistant.learn_shapes(store, None)?;
    println!("Discovered {} data patterns", shapes.len());

    // 2. Assess current data quality against the learned shapes
    let quality_report = assistant.assess_quality(store, &shapes)?;
    println!("Overall quality score: {:.2}%", quality_report.overall_score * 100.0);

    // 3. Generate improvement recommendations
    let insights = assistant.generate_insights(store, &shapes, &[])?;
    for recommendation in &insights.recommendations {
        println!("Recommendation: {}", recommendation.description);
    }

    Ok(())
}
```

## Shape Learning Modes

### Automatic Discovery

`ShapeLearner` scans every RDF class it finds in the store and learns a shape per
class; `LearningConfig::min_support` controls how much evidence a pattern needs
before it becomes part of a shape:

```rust
use oxirs_shacl_ai::{LearningConfig, ShapeLearner};

let config = LearningConfig {
    min_support: 0.1, // 10% of entities must match a pattern
    max_shapes: 100,
    ..Default::default()
};
let mut learner = ShapeLearner::with_config(config);
let shapes = learner.learn_shapes_from_store(&store, None)?;

// For larger graphs, the parallel variant scales across classes
let shapes_parallel = learner.learn_shapes_from_store_parallel(&store, None)?;
```

### Interactive Refinement

`InteractiveLabelingInterface` implements human-in-the-loop annotation: queue
uncertain shapes/patterns as tasks, assign them to annotators, and collect labels
back:

```rust
use oxirs_shacl_ai::interactive_labeling::{
    Annotator, AnnotatorStats, InteractiveLabelingInterface, LabelingConfig,
};

let mut interface = InteractiveLabelingInterface::with_config(LabelingConfig::default());

interface.register_annotator(Annotator {
    id: "reviewer-1".to_string(),
    name: "Alice".to_string(),
    email: "alice@example.org".to_string(),
    expertise_level: 0.9,
    stats: AnnotatorStats {
        total_annotations: 0,
        avg_time_per_annotation: 0.0,
        agreement_rate: 0.0,
        quality_score: 0.0,
        validated_annotations: 0,
        rejected_annotations: 0,
    },
    is_active: true,
})?;

// `add_task`/`add_tasks_batch` enqueue AnnotationTask values built from
// low-confidence learner output; `get_next_task` hands the highest-priority
// task to an annotator, and `submit_annotation` records their decision.
if let Some(task) = interface.get_next_task("reviewer-1") {
    println!("Review task {}: confidence={:.2}", task.id, task.confidence);
}
```

## Anomaly Detection

### Statistical Anomalies

`AnomalyDetector::detect` takes a numeric feature vector (e.g. property-value
distributions, embedding norms, or other statistics extracted from your graph) —
not raw RDF — so plan on a feature-extraction step upstream:

```rust
use oxirs_shacl_ai::{AnomalyConfig, AnomalyDetector, DetectorType};
use scirs2_core::ndarray_ext::Array1;

let config = AnomalyConfig {
    detector_type: DetectorType::StatisticalOutlier,
    threshold: 2.5, // z-score-style threshold
    ..Default::default()
};
let detector = AnomalyDetector::new(config);

let data: Array1<f64> = extract_feature_vector(store);
let result = detector.detect(&data)?;

for anomaly in result.anomalies {
    println!("Anomaly: {}", anomaly.description);
    println!("  Affected: {:?}", anomaly.affected_entities);
    println!("  Score: {:.2}", anomaly.score.score);
}
```

### ML-Based Detection

```rust
use oxirs_shacl_ai::{AnomalyConfig, AnomalyDetector, DetectorType};

let config = AnomalyConfig {
    detector_type: DetectorType::Autoencoder,
    enable_drift_detection: true,
    enable_novelty_detection: true,
    ..Default::default()
};
let detector = AnomalyDetector::new(config);

let result = detector.detect(&data)?;
println!("{} anomalies found in {:?}", result.anomalies.len(), result.metrics);
```

## Model Training

### Custom Training

Training happens directly on `ShapeLearner` via `train_model`, which runs a
mini-batch gradient-descent classifier over feature/label pairs:

```rust
use oxirs_shacl_ai::learning::types::ShapeTrainingData;
use oxirs_shacl_ai::{LearningConfig, ShapeLearner};

let mut config = LearningConfig::default();
config.enable_training = true;
let mut learner = ShapeLearner::with_config(config);

let training_data = ShapeTrainingData {
    features: vec![vec![0.8, 0.2, 0.5], vec![0.3, 0.9, 0.1]],
    labels: vec!["valid".to_string(), "invalid".to_string()],
    validation_features: vec![],
    validation_labels: vec![],
};

let result = learner.train_model(&training_data)?;
println!(
    "Trained {} epochs, accuracy={:.2}%, loss={:.4}",
    result.epochs_trained,
    result.accuracy * 100.0,
    result.loss
);
```

### Transfer Learning

```rust
use oxirs_shacl_ai::{TransferLearner, TransferLearningConfig, TransferStrategy};
use oxirs_core::Store;
use oxirs_shacl::Shape;

let config = TransferLearningConfig {
    strategy: TransferStrategy::FineTuning,
    source_domain: "generic".to_string(),
    target_domain: "healthcare".to_string(),
    ..Default::default()
};
let mut transfer = TransferLearner::new(config)?;

// Adapt to a new domain using a handful of target-domain shape examples
# let target_store: &dyn Store = unimplemented!();
# let target_samples: &[Shape] = &[];
transfer.adapt_to_domain(target_store, target_samples, None)?;
println!("{:?}", transfer.get_stats());
```

## Validation Suggestions

`SuggestionEngine` turns `GeneratedConstraint`s (produced upstream by
`ConstraintGenerator`) into ranked, human-readable `ConstraintSuggestion`s:

```rust
use oxirs_shacl_ai::{GeneratedConstraint, SuggestionEngine};

// e.g. from `ConstraintGenerator::generate_for_class`/`generate_for_properties`
let generated_constraints: Vec<GeneratedConstraint> = generate_constraints_upstream();

let engine = SuggestionEngine::new().with_min_confidence(0.7);
let suggestions = engine.generate_suggestions(generated_constraints);

for suggestion in suggestions {
    println!(
        "[priority {}] {:?}: {}",
        suggestion.priority, suggestion.confidence, suggestion.recommendation
    );
}
```

## Explainability

`ConstraintExplainer` (module `explainer`, part of the `llm` provider stack — see
[`examples/explain_violation.rs`](examples/explain_violation.rs)) turns a raw SHACL
violation summary into a plain-English explanation via a pluggable `CompletionProvider`.
The default `LocalProvider` is fully offline (deterministic, no network/API key):

```rust
use oxirs_shacl_ai::{CompletionProvider, ConstraintExplainer, LocalProvider};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = Arc::new(LocalProvider::new());
    let explainer = ConstraintExplainer::new(provider.clone());

    let explanation = explainer
        .explain("sh:minCount violation: foaf:name must appear at least once on node <http://example.org/Alice>")
        .await?;
    println!("Explanation: {explanation}");

    Ok(())
}
```

With the optional `llm-network` feature, `OpenAiProvider`/`AnthropicProvider` can be
swapped in wherever a `CompletionProvider` is expected.

## Integration with oxirs-shacl

Run the deterministic `oxirs-shacl` engine and the AI-assisted pipeline side by side:

```rust
use oxirs_shacl::{ValidationConfig, ValidationEngine};
use oxirs_shacl_ai::ShaclAiAssistant;
use oxirs_core::Store;

# fn example(shapes: &indexmap::IndexMap<oxirs_shacl::ShapeId, oxirs_shacl::Shape>, store: &dyn Store) -> anyhow::Result<()> {
// Standard, deterministic SHACL validation
let mut engine = ValidationEngine::new(shapes, ValidationConfig::default());
let report = engine.validate_store(store)?;
println!("{} violations (deterministic engine)", report.violations().len());

// AI-assisted pass: shape learning + a data-quality score alongside it
let mut assistant = ShaclAiAssistant::new();
let learned_shapes = assistant.learn_shapes(store, None)?;
let quality_report = assistant.assess_quality(store, &learned_shapes)?;
println!("AI quality score: {:.2}%", quality_report.overall_score * 100.0);
# Ok(())
# }
```

## Performance

Run the criterion suite for up-to-date numbers on your own hardware:

```bash
cargo bench -p oxirs-shacl-ai --bench performance_benchmarks
cargo bench -p oxirs-shacl-ai --bench advanced_scirs2_benchmarks --features gpu
```

## Configuration

`ShaclAiConfig` aggregates one sub-config per subsystem (learning, quality,
prediction, optimization, patterns, analytics) plus global settings; every
piece implements `Default`:

```rust
use oxirs_shacl_ai::ShaclAiConfig;
use oxirs_shacl_ai::LearningConfig;

let config = ShaclAiConfig {
    learning: LearningConfig {
        min_confidence: 0.8,
        max_shapes: 50,
        enable_training: true,
        ..Default::default()
    },
    ..Default::default()
};

let mut assistant = ShaclAiAssistant::with_config(config);
```

## Status

### Production Release (v0.4.0)
- ✅ Shape learning (`ShapeLearner`) with store-parallel and pattern-driven discovery
- ✅ ML model certification suite (`certification` module — precision/recall/F1/MCC vs. the deterministic engine)
- ✅ Anomaly detection (statistical, autoencoder, drift, novelty, ensemble) via `AnomalyDetector`
- ✅ Constraint suggestions (`SuggestionEngine`) and LLM-backed violation explanations (`ConstraintExplainer`, offline by default)
- ✅ Transfer learning (`TransferLearner`) and model zoo (`model_zoo`, SHA256-verified shape-learning models)
- ✅ `optimization_engine` for validation strategy/performance optimization
- ✅ GPU acceleration behind the optional, Pure-Rust `gpu` feature (gates `advanced_scirs2_integration`)
- 🚧 Auto-fix application (suggestions are generated; automatic SPARQL Update application is not yet wired in)

## Research

This module is based on research in:
- Neural-symbolic AI
- Knowledge graph completion
- Constraint learning
- Explainable AI

## Contributing

This is a research-oriented experimental module. Contributions and research collaborations welcome!

## License

Apache-2.0

## See Also

- [oxirs-shacl](../../engine/oxirs-shacl/) - Standard SHACL validation
- [oxirs-embed](../oxirs-embed/) - Embeddings for ML features
- [oxirs-rule](../../engine/oxirs-rule/) - Rule-based reasoning