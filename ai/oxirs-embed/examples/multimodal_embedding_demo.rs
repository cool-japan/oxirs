//! Multi-modal embedding demonstration
//!
//! This example showcases the cross-modal alignment capabilities of OxiRS Embed,
//! including text-KG alignment, entity descriptions, and multilingual support.

use anyhow::Result;
use oxirs_embed::{
    AlignmentObjective, ContrastiveConfig, CrossDomainConfig, CrossModalConfig, EmbeddingModel,
    MultiModalEmbedding, NamedNode, Triple,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 OxiRS Embed Multi-Modal Demonstration");
    println!("==========================================\n");

    // Create multi-modal configuration
    let mut config = CrossModalConfig::default();
    config.alignment_objective = AlignmentObjective::ContrastiveLearning;
    config.contrastive_config = ContrastiveConfig {
        temperature: 0.05,
        negative_samples: 32,
        hard_negative_mining: true,
        margin: 0.3,
        use_info_nce: true,
    };
    config.cross_domain_config = CrossDomainConfig {
        enable_domain_adaptation: true,
        source_domains: vec!["scientific".to_string(), "general".to_string()],
        target_domains: vec!["biomedical".to_string(), "legal".to_string()],
        domain_adversarial: false,
        gradual_adaptation: true,
    };

    println!("📋 Configuration:");
    println!("  Text dimensions: {}", config.text_dim);
    println!("  KG dimensions: {}", config.kg_dim);
    println!("  Unified dimensions: {}", config.unified_dim);
    println!("  Alignment objective: {:?}", config.alignment_objective);
    println!("  Temperature: {}", config.contrastive_config.temperature);
    println!();

    // Create multi-modal embedding model
    let mut model = MultiModalEmbedding::new(config);
    println!(
        "✅ Created multi-modal embedding model with ID: {}",
        model.model_id()
    );

    // Add text-KG alignments
    println!("\n🔗 Adding Text-KG Alignments:");
    let alignments = vec![
        (
            "A person who conducts scientific research",
            "http://example.org/Scientist",
        ),
        (
            "A medical professional who treats patients",
            "http://example.org/Doctor",
        ),
        ("An educational instructor", "http://example.org/Teacher"),
        (
            "A legal professional who represents clients",
            "http://example.org/Lawyer",
        ),
        (
            "A person who develops software",
            "http://example.org/Developer",
        ),
    ];

    for (text, entity) in &alignments {
        model.add_text_kg_alignment(text, entity);
        println!("  📝 \"{}\" ↔ {}", text, entity);
    }

    // Add entity descriptions
    println!("\n📖 Adding Entity Descriptions:");
    let descriptions = vec![
        (
            "http://example.org/Scientist",
            "A researcher who uses scientific methods to study natural phenomena",
        ),
        (
            "http://example.org/Doctor",
            "A healthcare professional with medical degree who diagnoses and treats illnesses",
        ),
        (
            "http://example.org/Teacher",
            "An educator who instructs students in academic subjects",
        ),
        (
            "http://example.org/Lawyer",
            "A legal practitioner who represents clients in legal matters",
        ),
        (
            "http://example.org/Developer",
            "A software professional who creates computer programs and applications",
        ),
    ];

    for (entity, description) in &descriptions {
        model.add_entity_description(entity, description);
        println!("  🏷️  {} → \"{}\"", entity, description);
    }

    // Add property-text mappings
    println!("\n🔗 Adding Property-Text Mappings:");
    let properties = vec![
        ("http://example.org/worksAt", "is employed by"),
        ("http://example.org/specializes", "has expertise in"),
        ("http://example.org/collaboratesWith", "works together with"),
        ("http://example.org/supervises", "manages or oversees"),
    ];

    for (property, text) in &properties {
        model.add_property_text(property, text);
        println!("  🔀 {} → \"{}\"", property, text);
    }

    // Add multilingual mappings
    println!("\n🌍 Adding Multilingual Mappings:");
    let multilingual = vec![
        (
            "scientist",
            vec![
                "científico".to_string(),
                "scientifique".to_string(),
                "wissenschaftler".to_string(),
            ],
        ),
        (
            "doctor",
            vec![
                "médico".to_string(),
                "médecin".to_string(),
                "arzt".to_string(),
            ],
        ),
        (
            "teacher",
            vec![
                "profesor".to_string(),
                "enseignant".to_string(),
                "lehrer".to_string(),
            ],
        ),
    ];

    for (concept, translations) in &multilingual {
        model.add_multilingual_mapping(concept, translations.clone());
        println!("  🗣️  {} → {:?}", concept, translations);
    }

    // Add cross-domain mappings
    println!("\n🌐 Adding Cross-Domain Mappings:");
    let cross_domain = vec![
        ("scientific_researcher", "biomedical_researcher"),
        ("general_practitioner", "biomedical_specialist"),
        ("contract_lawyer", "legal_corporate_counsel"),
    ];

    for (source, target) in &cross_domain {
        model.add_cross_domain_mapping(source, target);
        println!("  ↔️  {} ↔ {}", source, target);
    }

    // Add some RDF triples for structured knowledge
    println!("\n📊 Adding RDF Triples:");
    let triples = vec![
        (
            "http://example.org/alice",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://example.org/Scientist",
        ),
        (
            "http://example.org/alice",
            "http://example.org/worksAt",
            "http://example.org/university",
        ),
        (
            "http://example.org/bob",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://example.org/Doctor",
        ),
        (
            "http://example.org/bob",
            "http://example.org/specializes",
            "http://example.org/cardiology",
        ),
    ];

    for (subject, predicate, object) in &triples {
        let triple = Triple::new(
            NamedNode::new(subject)?,
            NamedNode::new(predicate)?,
            NamedNode::new(object)?,
        );
        model.add_triple(triple)?;
        println!("  📦 {} --{}-> {}", subject, predicate, object);
    }

    println!("\n🧠 Training Multi-Modal Model...");
    let training_stats = model.train(Some(50)).await?;

    println!("✅ Training completed!");
    println!("  📈 Epochs: {}", training_stats.epochs_completed);
    println!("  ⏱️  Time: {:.2}s", training_stats.training_time_seconds);
    println!("  📉 Final loss: {:.6}", training_stats.final_loss);
    println!("  🎯 Converged: {}", training_stats.convergence_achieved);

    // Demonstrate unified embedding generation
    println!("\n🎨 Generating Unified Embeddings:");
    let unified_examples = vec![
        (
            "A brilliant researcher in artificial intelligence",
            "http://example.org/Scientist",
        ),
        ("An experienced heart surgeon", "http://example.org/Doctor"),
        (
            "A dedicated high school mathematics teacher",
            "http://example.org/Teacher",
        ),
    ];

    for (text, entity) in &unified_examples {
        let unified_embedding = model.generate_unified_embedding(text, entity).await?;
        println!(
            "  🎯 \"{}\" + {} → Embedding[{}D]",
            text,
            entity,
            unified_embedding.len()
        );
    }

    // Demonstrate zero-shot prediction
    println!("\n🔮 Zero-Shot Prediction:");
    let candidates = vec![
        "http://example.org/Scientist".to_string(),
        "http://example.org/Doctor".to_string(),
        "http://example.org/Teacher".to_string(),
        "http://example.org/Lawyer".to_string(),
        "http://example.org/Developer".to_string(),
    ];

    let queries = vec![
        "A person who studies quantum physics",
        "Someone who performs surgery",
        "An instructor of computer science",
        "A person who writes legal documents",
        "Someone who codes in Python",
    ];

    for query in &queries {
        let predictions = model.zero_shot_prediction(query, &candidates).await?;
        println!("\n  🔍 Query: \"{}\"", query);
        for (i, (entity, score)) in predictions.iter().take(3).enumerate() {
            println!("    {}. {} (score: {:.3})", i + 1, entity, score);
        }
    }

    // Demonstrate multilingual alignment
    println!("\n🌍 Multilingual Alignment:");
    for (concept, _) in &multilingual {
        let alignments = model.multilingual_alignment(concept).await?;
        if !alignments.is_empty() {
            println!("  🗣️  Concept: {}", concept);
            for (translation, score) in alignments {
                println!("    → {} (alignment: {:.3})", translation, score);
            }
        }
    }

    // Demonstrate cross-domain transfer
    println!("\n🌐 Cross-Domain Transfer:");
    let transfer_results = model
        .cross_domain_transfer("scientific", "biomedical")
        .await?;
    println!(
        "  🔄 Scientific → Biomedical transfer loss: {:.3}",
        transfer_results
    );

    let transfer_results = model.cross_domain_transfer("general", "legal").await?;
    println!(
        "  🔄 General → Legal transfer loss: {:.3}",
        transfer_results
    );

    // Show model statistics
    println!("\n📊 Multi-Modal Model Statistics:");
    let stats = model.get_multimodal_stats();
    println!("  📝 Text embeddings: {}", stats.num_text_embeddings);
    println!("  🧠 KG embeddings: {}", stats.num_kg_embeddings);
    println!("  🎯 Unified embeddings: {}", stats.num_unified_embeddings);
    println!("  🔗 Text-KG alignments: {}", stats.num_alignments);
    println!(
        "  📖 Entity descriptions: {}",
        stats.num_entity_descriptions
    );
    println!("  🔀 Property texts: {}", stats.num_property_texts);
    println!(
        "  🌍 Multilingual mappings: {}",
        stats.num_multilingual_mappings
    );
    println!(
        "  🌐 Cross-domain mappings: {}",
        stats.num_cross_domain_mappings
    );
    println!(
        "  📏 Dimensions: {}D text, {}D KG, {}D unified",
        stats.text_dim, stats.kg_dim, stats.unified_dim
    );

    // Demonstrate similarity searches
    println!("\n🔍 Similarity Search:");
    let query_text = "machine learning expert";
    if let Ok(text_embeddings) = model.encode(&[query_text.to_string()]).await {
        if let Some(query_embedding) = text_embeddings.first() {
            println!(
                "  📊 Generated embedding for \"{}\" with {} dimensions",
                query_text,
                query_embedding.len()
            );

            // Find most similar entities
            let entities = model.get_entities();
            if !entities.is_empty() {
                println!("  🎯 Found {} entities for comparison", entities.len());

                let predictions = model.predict_objects(
                    "http://example.org/alice",
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                    3,
                )?;
                println!("  🔮 Type predictions for Alice:");
                for (entity, score) in predictions {
                    println!("    → {} (confidence: {:.3})", entity, score);
                }
            }
        }
    }

    println!("\n🎉 Multi-Modal Embedding Demonstration Complete!");
    println!("✨ The model successfully demonstrates:");
    println!("   • Text-Knowledge Graph alignment");
    println!("   • Cross-modal contrastive learning");
    println!("   • Zero-shot prediction capabilities");
    println!("   • Multilingual concept alignment");
    println!("   • Cross-domain knowledge transfer");
    println!("   • Unified representation learning");

    Ok(())
}
