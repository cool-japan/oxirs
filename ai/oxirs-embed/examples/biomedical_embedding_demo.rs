//! Biomedical Knowledge Graph Embeddings Demo
//!
//! This example demonstrates how to use OxiRS Embed for biomedical knowledge graphs
//! including gene-disease associations, drug-target interactions, and specialized
//! text embeddings for biomedical literature.

use anyhow::Result;
use oxirs_embed::{
    BiomedicalEmbedding, BiomedicalEmbeddingConfig, BiomedicalEntityType, BiomedicalRelationType,
    EmbeddingModel, NamedNode, SpecializedTextEmbedding, Triple,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧬 OxiRS Biomedical Embeddings Demo");
    println!("=====================================\n");

    // 1. Biomedical Knowledge Graph Embeddings
    demo_biomedical_kg_embeddings().await?;

    // 2. Specialized Text Embeddings
    demo_specialized_text_embeddings().await?;

    // 3. Drug Discovery Use Case
    demo_drug_discovery().await?;

    Ok(())
}

/// Demonstrate biomedical knowledge graph embeddings
async fn demo_biomedical_kg_embeddings() -> Result<()> {
    println!("📊 1. Biomedical Knowledge Graph Embeddings");
    println!("───────────────────────────────────────────\n");

    // Create biomedical embedding model with specialized configuration
    let config = BiomedicalEmbeddingConfig {
        gene_disease_weight: 2.5,
        drug_target_weight: 2.0,
        pathway_weight: 1.5,
        protein_interaction_weight: 1.8,
        use_sequence_similarity: true,
        use_chemical_structure: true,
        species_filter: Some("Homo sapiens".to_string()),
        ..Default::default()
    };

    let mut model = BiomedicalEmbedding::new(config);

    // Add sample biomedical knowledge
    println!("🔬 Adding sample biomedical knowledge...");

    // Gene-disease associations
    let gene_disease_triples = [
        (
            "http://bio.org/gene/BRCA1",
            "http://bio.org/causes",
            "http://bio.org/disease/breast_cancer",
        ),
        (
            "http://bio.org/gene/TP53",
            "http://bio.org/associated_with",
            "http://bio.org/disease/cancer",
        ),
        (
            "http://bio.org/gene/APOE",
            "http://bio.org/predisposes_to",
            "http://bio.org/disease/alzheimer",
        ),
        (
            "http://bio.org/gene/HTT",
            "http://bio.org/causes",
            "http://bio.org/disease/huntington",
        ),
    ];

    // Drug-target interactions
    let drug_target_triples = vec![
        (
            "http://bio.org/drug/aspirin",
            "http://bio.org/targets",
            "http://bio.org/protein/COX1",
        ),
        (
            "http://bio.org/drug/ibuprofen",
            "http://bio.org/inhibits",
            "http://bio.org/protein/COX2",
        ),
        (
            "http://bio.org/drug/metformin",
            "http://bio.org/activates",
            "http://bio.org/protein/AMPK",
        ),
        (
            "http://bio.org/drug/warfarin",
            "http://bio.org/binds_to",
            "http://bio.org/protein/CYP2C9",
        ),
    ];

    // Pathway memberships
    let pathway_triples = vec![
        (
            "http://bio.org/gene/BRCA1",
            "http://bio.org/participates_in",
            "http://bio.org/pathway/dna_repair",
        ),
        (
            "http://bio.org/protein/COX1",
            "http://bio.org/participates_in",
            "http://bio.org/pathway/inflammation",
        ),
        (
            "http://bio.org/gene/APOE",
            "http://bio.org/participates_in",
            "http://bio.org/pathway/lipid_metabolism",
        ),
    ];

    // Add all triples to the model
    for (s, p, o) in gene_disease_triples
        .iter()
        .chain(&drug_target_triples)
        .chain(&pathway_triples)
    {
        let triple = Triple::new(NamedNode::new(s)?, NamedNode::new(p)?, NamedNode::new(o)?);
        model.add_triple(triple)?;
    }

    // Add quantitative biomedical features
    println!("📈 Adding quantitative biomedical features...");

    // Gene-disease association scores (from literature/databases)
    model.add_gene_disease_association(
        "http://bio.org/gene/BRCA1",
        "http://bio.org/disease/breast_cancer",
        0.95,
    );
    model.add_gene_disease_association(
        "http://bio.org/gene/TP53",
        "http://bio.org/disease/cancer",
        0.88,
    );
    model.add_gene_disease_association(
        "http://bio.org/gene/APOE",
        "http://bio.org/disease/alzheimer",
        0.72,
    );

    // Drug-target binding affinities (in nM)
    model.add_drug_target_interaction(
        "http://bio.org/drug/aspirin",
        "http://bio.org/protein/COX1",
        0.92,
    );
    model.add_drug_target_interaction(
        "http://bio.org/drug/ibuprofen",
        "http://bio.org/protein/COX2",
        0.85,
    );
    model.add_drug_target_interaction(
        "http://bio.org/drug/metformin",
        "http://bio.org/protein/AMPK",
        0.78,
    );

    // Pathway membership scores
    model.add_pathway_membership(
        "http://bio.org/gene/BRCA1",
        "http://bio.org/pathway/dna_repair",
        0.94,
    );
    model.add_pathway_membership(
        "http://bio.org/protein/COX1",
        "http://bio.org/pathway/inflammation",
        0.89,
    );

    // Protein-protein interactions
    model.add_protein_interaction(
        "http://bio.org/protein/COX1",
        "http://bio.org/protein/COX2",
        0.65,
    );

    // Train the model
    println!("🎯 Training biomedical embedding model...");
    let training_stats = model.train(Some(100)).await?;
    println!(
        "✅ Training completed in {:.2}s",
        training_stats.training_time_seconds
    );
    println!("   Final loss: {:.6}", training_stats.final_loss);
    println!(
        "   Convergence: {}",
        if training_stats.convergence_achieved {
            "✅"
        } else {
            "❌"
        }
    );

    // Demonstrate predictions
    println!("\n🔮 Making biomedical predictions...");

    // Predict diseases for BRCA1 gene
    if let Ok(predictions) = model.predict_gene_disease_associations("http://bio.org/gene/BRCA1", 3)
    {
        println!("\n🧬 Top diseases associated with BRCA1:");
        for (disease, score) in predictions {
            println!(
                "   {} → {:.3}",
                disease.split('/').next_back().unwrap_or(&disease),
                score
            );
        }
    }

    // Predict drug targets for aspirin
    if let Ok(predictions) = model.predict_drug_targets("http://bio.org/drug/aspirin", 3) {
        println!("\n💊 Top targets for aspirin:");
        for (target, score) in predictions {
            println!(
                "   {} → {:.3}",
                target.split('/').next_back().unwrap_or(&target),
                score
            );
        }
    }

    // Find entities related to DNA repair pathway
    if let Ok(predictions) = model.find_pathway_entities("http://bio.org/pathway/dna_repair", 3) {
        println!("\n🔗 Entities related to DNA repair pathway:");
        for (entity, score) in predictions {
            println!(
                "   {} → {:.3}",
                entity.split('/').next_back().unwrap_or(&entity),
                score
            );
        }
    }

    println!("\n");
    Ok(())
}

/// Demonstrate specialized text embeddings for biomedical literature
async fn demo_specialized_text_embeddings() -> Result<()> {
    println!("📚 2. Specialized Text Embeddings");
    println!("─────────────────────────────────\n");

    // Create different specialized text models
    let models = vec![
        (
            "SciBERT",
            SpecializedTextEmbedding::new(SpecializedTextEmbedding::scibert_config()),
        ),
        (
            "BioBERT",
            SpecializedTextEmbedding::new(SpecializedTextEmbedding::biobert_config()),
        ),
        (
            "CodeBERT",
            SpecializedTextEmbedding::new(SpecializedTextEmbedding::codebert_config()),
        ),
    ];

    let test_texts = vec![
        "The BRCA1 gene mutation increases breast cancer risk significantly.",
        "Patients receiving 100 mg/kg b.i.d. showed improved outcomes.",
        "The protein folding study demonstrates significant results with p < 0.001.",
        "function calculateDrugInteraction(drug, target) { return binding_affinity; }",
    ];

    for (model_name, mut model) in models {
        println!("🤖 Testing {model_name} Model:");

        for text in &test_texts {
            // Generate embedding for the text
            let embedding = model.encode_text(text).await?;
            println!("   Text: \"{}...\"", &text[..50.min(text.len())]);
            println!(
                "   Embedding dim: {}, norm: {:.3}",
                embedding.len(),
                embedding.iter().map(|x| x * x).sum::<f32>().sqrt()
            );
        }
        println!();
    }

    // Demonstrate fine-tuning on biomedical texts
    println!("🎯 Fine-tuning BioBERT on domain-specific texts...");
    let mut biobert = SpecializedTextEmbedding::new(SpecializedTextEmbedding::biobert_config());

    let biomedical_training_texts = vec![
        "Gene expression analysis reveals differential patterns in cancer cells".to_string(),
        "Protein-protein interactions regulate cellular signaling pathways".to_string(),
        "Drug metabolism pathways involve cytochrome P450 enzymes".to_string(),
        "Clinical trials demonstrate efficacy of targeted cancer therapies".to_string(),
        "Biomarker discovery enables personalized medicine approaches".to_string(),
    ];

    let fine_tune_stats = biobert.fine_tune(biomedical_training_texts).await?;
    println!(
        "✅ Fine-tuning completed in {:.2}s",
        fine_tune_stats.training_time_seconds
    );
    println!("   Final loss: {:.6}", fine_tune_stats.final_loss);
    println!("   Epochs: {}", fine_tune_stats.epochs_completed);

    println!();
    Ok(())
}

/// Demonstrate drug discovery use case
async fn demo_drug_discovery() -> Result<()> {
    println!("💊 3. Drug Discovery Use Case");
    println!("────────────────────────────\n");

    // Create specialized biomedical model for drug discovery
    let config = BiomedicalEmbeddingConfig {
        drug_target_weight: 3.0,
        gene_disease_weight: 2.0,
        pathway_weight: 1.5,
        use_chemical_structure: true,
        use_sequence_similarity: true,
        ..Default::default()
    };

    let mut discovery_model = BiomedicalEmbedding::new(config);

    println!("🔬 Building drug discovery knowledge base...");

    // Add comprehensive drug-target-disease network
    let knowledge_triples = vec![
        // Cardiovascular drugs
        (
            "http://drugs.org/atorvastatin",
            "http://bio.org/targets",
            "http://proteins.org/HMGCR",
        ),
        (
            "http://proteins.org/HMGCR",
            "http://bio.org/regulates",
            "http://pathways.org/cholesterol_synthesis",
        ),
        (
            "http://pathways.org/cholesterol_synthesis",
            "http://bio.org/affects",
            "http://diseases.org/hypercholesterolemia",
        ),
        // Cancer drugs
        (
            "http://drugs.org/trastuzumab",
            "http://bio.org/targets",
            "http://proteins.org/HER2",
        ),
        (
            "http://proteins.org/HER2",
            "http://bio.org/overexpressed_in",
            "http://diseases.org/breast_cancer",
        ),
        (
            "http://drugs.org/imatinib",
            "http://bio.org/inhibits",
            "http://proteins.org/BCR_ABL",
        ),
        // Neurological drugs
        (
            "http://drugs.org/donepezil",
            "http://bio.org/inhibits",
            "http://proteins.org/AChE",
        ),
        (
            "http://proteins.org/AChE",
            "http://bio.org/depleted_in",
            "http://diseases.org/alzheimer",
        ),
    ];

    for (s, p, o) in knowledge_triples {
        let triple = Triple::new(NamedNode::new(s)?, NamedNode::new(p)?, NamedNode::new(o)?);
        discovery_model.add_triple(triple)?;
    }

    // Add quantitative drug discovery data
    discovery_model.add_drug_target_interaction(
        "http://drugs.org/atorvastatin",
        "http://proteins.org/HMGCR",
        0.95,
    );
    discovery_model.add_drug_target_interaction(
        "http://drugs.org/trastuzumab",
        "http://proteins.org/HER2",
        0.98,
    );
    discovery_model.add_drug_target_interaction(
        "http://drugs.org/imatinib",
        "http://proteins.org/BCR_ABL",
        0.92,
    );
    discovery_model.add_drug_target_interaction(
        "http://drugs.org/donepezil",
        "http://proteins.org/AChE",
        0.87,
    );

    discovery_model.add_gene_disease_association(
        "http://proteins.org/HER2",
        "http://diseases.org/breast_cancer",
        0.94,
    );
    discovery_model.add_gene_disease_association(
        "http://proteins.org/BCR_ABL",
        "http://diseases.org/cml",
        0.98,
    );
    discovery_model.add_gene_disease_association(
        "http://proteins.org/AChE",
        "http://diseases.org/alzheimer",
        0.85,
    );

    // Train the discovery model
    println!("🎯 Training drug discovery model...");
    let stats = discovery_model.train(Some(150)).await?;
    println!(
        "✅ Training completed in {:.2}s with loss {:.6}",
        stats.training_time_seconds, stats.final_loss
    );

    // Demonstrate drug discovery predictions
    println!("\n🔮 Drug Discovery Predictions:");

    // Find potential targets for a new cardiovascular drug
    println!("\n💗 Cardiovascular Drug Targets:");
    if let Ok(targets) = discovery_model.predict_drug_targets("http://drugs.org/atorvastatin", 5) {
        for (target, score) in targets {
            let protein_name = target.split('/').next_back().unwrap_or(&target);
            println!("   {protein_name} → {score:.3}");
        }
    }

    // Find potential drugs for cancer treatment
    println!("\n🎯 Cancer Treatment Options:");
    if let Ok(drugs) =
        discovery_model.predict_subjects("http://bio.org/targets", "http://proteins.org/HER2", 3)
    {
        for (drug, score) in drugs {
            let drug_name = drug.split('/').next_back().unwrap_or(&drug);
            println!("   {drug_name} → {score:.3}");
        }
    }

    // Analyze model statistics
    let model_stats = discovery_model.get_stats();
    println!("\n📊 Discovery Model Statistics:");
    println!("   Entities: {}", model_stats.num_entities);
    println!("   Relations: {}", model_stats.num_relations);
    println!("   Triples: {}", model_stats.num_triples);
    println!("   Dimensions: {}", model_stats.dimensions);
    println!("   Model Type: {}", model_stats.model_type);

    println!("\n🎉 Drug discovery demo completed!");
    Ok(())
}

/// Helper function to display entity type information
#[allow(dead_code)]
fn display_entity_info(entity_iri: &str) -> String {
    if let Some(entity_type) = BiomedicalEntityType::from_iri(entity_iri) {
        format!(
            "{} ({})",
            entity_iri.split('/').next_back().unwrap_or(entity_iri),
            match entity_type {
                BiomedicalEntityType::Gene => "Gene",
                BiomedicalEntityType::Protein => "Protein",
                BiomedicalEntityType::Disease => "Disease",
                BiomedicalEntityType::Drug => "Drug",
                BiomedicalEntityType::Compound => "Compound",
                BiomedicalEntityType::Pathway => "Pathway",
                _ => "Other",
            }
        )
    } else {
        entity_iri
            .split('/')
            .next_back()
            .unwrap_or(entity_iri)
            .to_string()
    }
}

/// Helper function to display relation type information
#[allow(dead_code)]
fn display_relation_info(relation_iri: &str) -> String {
    if let Some(relation_type) = BiomedicalRelationType::from_iri(relation_iri) {
        format!(
            "{} ({})",
            relation_iri.split('/').next_back().unwrap_or(relation_iri),
            match relation_type {
                BiomedicalRelationType::CausesDisease => "Causes Disease",
                BiomedicalRelationType::TargetsProtein => "Targets Protein",
                BiomedicalRelationType::ParticipatesInPathway => "Participates in Pathway",
                BiomedicalRelationType::InteractsWith => "Interacts With",
                _ => "Other Relation",
            }
        )
    } else {
        relation_iri
            .split('/')
            .next_back()
            .unwrap_or(relation_iri)
            .to_string()
    }
}
