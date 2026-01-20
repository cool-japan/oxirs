//! Example: AAS to SAMM Conversion
//!
//! This example demonstrates how to use the AAS parser to convert
//! Asset Administration Shell files to SAMM Aspect Models.
//!
//! Run with:
//! ```bash
//! cargo run --example aas_conversion -- <input.aasx>
//! ```

use oxirs_samm::aas_parser;
use oxirs_samm::metamodel::ModelElement;
use oxirs_samm::serializer;
use std::env;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get input file from command line
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.aasx|input.json|input.xml>", args[0]);
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} AssetAdminShell.aasx", args[0]);
        std::process::exit(1);
    }

    let input_file = PathBuf::from(&args[1]);
    let output_dir = PathBuf::from("output");

    println!("AAS to SAMM Conversion Example");
    println!("================================\n");

    // Step 1: Parse AAS file
    println!("1. Parsing AAS file: {}", input_file.display());
    let env = aas_parser::parse_aas_file(&input_file).await?;
    println!("   ✓ Successfully parsed AAS file\n");

    // Step 2: List submodels
    println!("2. Found submodels:");
    let submodels = aas_parser::list_submodels(&env);
    for (idx, id, name, desc) in &submodels {
        println!("   [{}] {}", idx, name.as_ref().unwrap_or(id));
        println!("       ID: {}", id);
        if let Some(d) = desc {
            let short_desc = if d.len() > 60 {
                format!("{}...", &d[..57])
            } else {
                d.clone()
            };
            println!("       Description: {}", short_desc);
        }
    }
    println!();

    // Step 3: Convert to SAMM Aspects
    println!("3. Converting to SAMM Aspect Models...");
    let aspects = aas_parser::convert_to_aspects(&env, vec![])?;
    println!("   ✓ Converted {} Aspect Model(s)\n", aspects.len());

    // Step 4: Serialize to Turtle files
    println!("4. Generating Turtle (.ttl) files:");

    // Create output directory
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir)?;
    }

    for (idx, aspect) in aspects.iter().enumerate() {
        let filename = format!("{}.ttl", aspect.name());
        let filepath = output_dir.join(&filename);

        // Serialize to Turtle format
        serializer::serialize_aspect_to_file(aspect, &filepath).await?;

        println!("   [{}] {}", idx, aspect.name());
        println!("       URN: {}", aspect.metadata().urn);
        println!("       Properties: {}", aspect.properties().len());
        println!("       Operations: {}", aspect.operations().len());
        println!("       Output: {}", filepath.display());
        println!();
    }

    println!(
        "✓ Successfully generated {} Turtle file(s) in {}/",
        aspects.len(),
        output_dir.display()
    );

    // Step 5: Display sample Turtle content
    if !aspects.is_empty() {
        println!("\n5. Sample Turtle output (first aspect):");
        println!("   ---");
        let sample_ttl = serializer::serialize_aspect_to_string(&aspects[0])?;
        for (i, line) in sample_ttl.lines().take(15).enumerate() {
            println!("   {}", line);
            if i == 14 && sample_ttl.lines().count() > 15 {
                println!("   ... (truncated)");
            }
        }
        println!("   ---");
    }

    println!("\nConversion complete!");
    println!("Next steps:");
    println!(
        "  - Review generated .ttl files in {}/",
        output_dir.display()
    );
    println!("  - Use 'oxirs aspect <file>.ttl to <format>' to generate code");
    println!("  - Example: oxirs aspect output/Movement.ttl to graphql");

    Ok(())
}
