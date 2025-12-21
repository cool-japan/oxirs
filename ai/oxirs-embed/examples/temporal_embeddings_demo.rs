//! Temporal Embeddings Demo
//!
//! This example demonstrates how to use temporal embeddings to model
//! knowledge graphs that evolve over time.
//!
//! Run with:
//! ```bash
//! cargo run --example temporal_embeddings_demo
//! ```

use anyhow::Result;
use chrono::{Duration, Utc};
use oxirs_embed::{
    NamedNode, TemporalEmbeddingConfig, TemporalEmbeddingModel, TemporalGranularity, TemporalScope,
    TemporalTriple, Triple,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== OxiRS Temporal Embeddings Demo ===\n");

    // Step 1: Create temporal embedding model
    println!("1. Creating temporal embedding model...");
    let config = TemporalEmbeddingConfig {
        granularity: TemporalGranularity::Day,
        time_dim: 32,
        enable_decay: true,
        decay_rate: 0.9,
        enable_smoothing: true,
        smoothing_window: 7,
        enable_forecasting: true,
        forecast_horizon: 30,
        enable_event_detection: true,
        event_threshold: 0.5,
        ..Default::default()
    };

    let mut model = TemporalEmbeddingModel::new(config);
    println!("   Granularity: Daily");
    println!("   Time Dimensions: 32");
    println!("   Forecasting Enabled: Yes (30 days)");
    println!("   Event Detection: Enabled");

    // Step 2: Add temporal triples for employment history
    println!("\n2. Adding temporal employment history...");

    let now = Utc::now();

    // Alice worked at Company A from 2020 to 2022
    let triple1 = Triple::new(
        NamedNode::new("http://example.org/Alice")?,
        NamedNode::new("http://example.org/worksFor")?,
        NamedNode::new("http://example.org/CompanyA")?,
    );
    let temporal_triple1 = TemporalTriple::new(
        triple1,
        TemporalScope::Interval {
            start: now - Duration::days(1825), // ~5 years ago
            end: now - Duration::days(1095),   // ~3 years ago
        },
    );
    model.add_temporal_triple(temporal_triple1).await?;
    println!("   Alice worked at Company A (2020-2022)");

    // Alice works at Company B from 2022 to present
    let triple2 = Triple::new(
        NamedNode::new("http://example.org/Alice")?,
        NamedNode::new("http://example.org/worksFor")?,
        NamedNode::new("http://example.org/CompanyB")?,
    );
    let temporal_triple2 = TemporalTriple::new(
        triple2,
        TemporalScope::Interval {
            start: now - Duration::days(1095), // ~3 years ago
            end: now + Duration::days(365),    // 1 year future (expected)
        },
    );
    model.add_temporal_triple(temporal_triple2).await?;
    println!("   Alice works at Company B (2022-present)");

    // Bob joined Company B recently
    let triple3 = Triple::new(
        NamedNode::new("http://example.org/Bob")?,
        NamedNode::new("http://example.org/worksFor")?,
        NamedNode::new("http://example.org/CompanyB")?,
    );
    let temporal_triple3 = TemporalTriple::new(
        triple3,
        TemporalScope::Interval {
            start: now - Duration::days(180), // 6 months ago
            end: now + Duration::days(365),
        },
    );
    model.add_temporal_triple(temporal_triple3).await?;
    println!("   Bob joined Company B (6 months ago)");

    // Add periodic events - monthly meetings
    let triple4 = Triple::new(
        NamedNode::new("http://example.org/Alice")?,
        NamedNode::new("http://example.org/attendsMeeting")?,
        NamedNode::new("http://example.org/MonthlyReview")?,
    );
    let temporal_triple4 = TemporalTriple::new(
        triple4,
        TemporalScope::Periodic {
            start: now - Duration::days(365),
            period: Duration::days(30),
            count: Some(12), // 12 monthly meetings
        },
    );
    model.add_temporal_triple(temporal_triple4).await?;
    println!("   Alice attends monthly reviews (periodic)");

    // Step 3: Train temporal embeddings
    println!("\n3. Training temporal embeddings...");
    let training_stats = model.train_temporal(50).await?;
    println!("   Epochs: {}", training_stats.epochs_completed);
    println!("   Final Loss: {:.6}", training_stats.final_loss);
    println!(
        "   Training Time: {:.2}s",
        training_stats.training_time_seconds
    );
    println!(
        "   Convergence: {}",
        if training_stats.convergence_achieved {
            "Yes"
        } else {
            "No"
        }
    );

    // Step 4: Query temporal data
    println!("\n4. Querying temporal data...");

    // Query at different time points
    let past_date = now - Duration::days(1460); // 4 years ago
    let recent_date = now - Duration::days(30); // 1 month ago

    println!("\n   Triples valid 4 years ago:");
    let past_triples = model.query_at_time(&past_date).await;
    for triple in past_triples.iter().take(3) {
        let subject = triple
            .subject
            .iri
            .split('/')
            .next_back()
            .unwrap_or(&triple.subject.iri);
        let predicate = triple
            .predicate
            .iri
            .split('/')
            .next_back()
            .unwrap_or(&triple.predicate.iri);
        let object = triple
            .object
            .iri
            .split('/')
            .next_back()
            .unwrap_or(&triple.object.iri);
        println!("     {} -> {} -> {}", subject, predicate, object);
    }

    println!("\n   Triples valid 1 month ago:");
    let recent_triples = model.query_at_time(&recent_date).await;
    for triple in recent_triples.iter().take(3) {
        let subject = triple
            .subject
            .iri
            .split('/')
            .next_back()
            .unwrap_or(&triple.subject.iri);
        let predicate = triple
            .predicate
            .iri
            .split('/')
            .next_back()
            .unwrap_or(&triple.predicate.iri);
        let object = triple
            .object
            .iri
            .split('/')
            .next_back()
            .unwrap_or(&triple.object.iri);
        println!("     {} -> {} -> {}", subject, predicate, object);
    }

    // Step 5: Get entity embeddings at different times
    println!("\n5. Analyzing entity embeddings over time...");

    let entity = "http://example.org/Alice";

    match model.get_entity_embedding_at_time(entity, &past_date).await {
        Ok(embedding) => {
            println!("   Alice's embedding 4 years ago:");
            println!("     Dimensions: {}", embedding.dimensions);
            println!(
                "     First 5 values: {:?}",
                &embedding.values[..5.min(embedding.values.len())]
            );
        }
        Err(e) => println!("   Could not get past embedding: {}", e),
    }

    match model
        .get_entity_embedding_at_time(entity, &recent_date)
        .await
    {
        Ok(embedding) => {
            println!("\n   Alice's embedding 1 month ago:");
            println!("     Dimensions: {}", embedding.dimensions);
            println!(
                "     First 5 values: {:?}",
                &embedding.values[..5.min(embedding.values.len())]
            );
        }
        Err(e) => println!("   Could not get recent embedding: {}", e),
    }

    // Step 6: Forecast future embeddings
    println!("\n6. Forecasting future entity states...");

    match model.forecast(entity, 10).await {
        Ok(forecast) => {
            println!("   Forecasted 10 time steps ahead for Alice:");
            println!("   Number of predictions: {}", forecast.predictions.len());
            println!("   Target: {}", forecast.target);

            if !forecast.predictions.is_empty() {
                println!("   First prediction:");
                let first_pred = &forecast.predictions[0];
                println!("     Dimensions: {}", first_pred.dimensions);
                println!(
                    "     First 5 values: {:?}",
                    &first_pred.values[..5.min(first_pred.values.len())]
                );

                if !forecast.confidence_intervals.is_empty() {
                    let (lower, upper) = &forecast.confidence_intervals[0];
                    println!(
                        "     Confidence interval range: [{:.3}, {:.3}]",
                        lower.values[0], upper.values[0]
                    );
                }
            }
        }
        Err(e) => println!("   Forecasting not available: {}", e),
    }

    // Step 7: Detect temporal events
    println!("\n7. Detecting temporal events...");
    let events = model.detect_events(0.3).await?;

    if events.is_empty() {
        println!("   No significant events detected (threshold: 0.3)");
    } else {
        println!("   Detected {} temporal events:", events.len());
        for (i, event) in events.iter().take(3).enumerate() {
            println!("\n   Event {}:", i + 1);
            println!("     Type: {}", event.event_type);
            println!("     Significance: {:.3}", event.significance);
            println!("     Entities: {:?}", event.entities);
            if let Some(desc) = &event.description {
                println!("     Description: {}", desc);
            }
        }
    }

    // Step 8: Display temporal statistics
    println!("\n8. Temporal Statistics:");
    let stats = model.get_temporal_stats().await;

    println!("   Total Temporal Triples: {}", stats.num_temporal_triples);
    println!("   Unique Entities: {}", stats.num_entities);
    println!("   Unique Relations: {}", stats.num_relations);
    println!("   Time Points Tracked: {}", stats.num_time_points);
    println!("   Events Detected: {}", stats.num_events);

    if let (Some(start), Some(end)) = (stats.time_span_start, stats.time_span_end) {
        let duration = end.signed_duration_since(start);
        println!("   Time Span: {} days", duration.num_days());
    }

    println!("\n=== Demo Complete ===");
    println!("\nKey Takeaways:");
    println!("✓ Temporal embeddings capture how knowledge evolves over time");
    println!("✓ Query historical states of entities and relations");
    println!("✓ Forecast future entity states with confidence intervals");
    println!("✓ Detect significant temporal events automatically");
    println!("✓ Support multiple temporal granularities and scopes");

    Ok(())
}
