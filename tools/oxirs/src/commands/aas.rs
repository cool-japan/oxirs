//! AAS (Asset Administration Shell) command implementations (Java ESMF SDK compatible)
//!
//! This module provides commands for working with Asset Administration Shell (AAS) files
//! and converting them to SAMM Aspect Models, following the Java ESMF SDK CLI interface.

use crate::cli::CliResult;
use crate::AasAction;
use oxirs_samm::metamodel::ModelElement;
use std::path::PathBuf;

/// Run an AAS command (Java ESMF SDK compatible)
///
/// # Arguments
///
/// * `action` - The AAS action to perform (ToAspect or List)
///
/// # Returns
///
/// * `CliResult<()>` - Success or error result
pub async fn run(action: AasAction) -> CliResult<()> {
    match action {
        AasAction::ToAspect {
            file,
            output_directory,
            submodel_templates,
        } => to_aspect(file, output_directory, submodel_templates).await,
        AasAction::List { file } => list(file).await,
    }
}

/// Convert AAS Submodel Templates to SAMM Aspect Models
///
/// This command parses an AAS file (XML, JSON, or AASX format) and converts
/// the submodel templates to SAMM Aspect Models.
///
/// # Arguments
///
/// * `file` - Path to the AAS file (XML, JSON, or AASX)
/// * `output_directory` - Optional output directory for generated Aspect Models
/// * `submodel_templates` - Optional list of specific submodel template indices to convert
///
/// # Examples
///
/// ```bash
/// # Convert all submodel templates in an AASX file
/// oxirs aas AssetAdminShell.aasx to aspect
///
/// # Convert to a specific directory
/// oxirs aas AssetAdminShell.aasx to aspect -d output/
///
/// # Convert specific submodel templates (by index)
/// oxirs aas AssetAdminShell.aasx to aspect -s 1 -s 2 -d output/
/// ```
async fn to_aspect(
    file: PathBuf,
    output_directory: Option<PathBuf>,
    submodel_templates: Vec<usize>,
) -> CliResult<()> {
    // Determine file format from extension
    let format = detect_aas_format(&file)?;

    println!("Converting AAS file to Aspect Models...");
    println!("  File: {}", file.display());
    println!("  Format: {}", format);

    if let Some(ref dir) = output_directory {
        println!("  Output Directory: {}", dir.display());
    } else {
        println!("  Output Directory: (current directory)");
    }

    if !submodel_templates.is_empty() {
        println!("  Selected Submodels: {:?}", submodel_templates);
    } else {
        println!("  Selected Submodels: (all)");
    }

    // Check if file exists
    if !file.exists() {
        return Err(format!("AAS file not found: {}", file.display()).into());
    }

    // Parse the AAS file
    let env = oxirs_samm::aas_parser::parse_aas_file(&file)
        .await
        .map_err(|e| format!("Failed to parse AAS file: {}", e))?;

    println!("\n✓ Successfully parsed AAS file");
    println!("  Found {} submodel(s)", env.submodels.len());

    // Convert to Aspect Models
    let aspects = oxirs_samm::aas_parser::convert_to_aspects(&env, submodel_templates)
        .map_err(|e| format!("Failed to convert to Aspect Models: {}", e))?;

    println!("  Converted {} Aspect Model(s)", aspects.len());

    // Determine output directory
    let output_dir = output_directory
        .unwrap_or_else(|| std::env::current_dir().expect("failed to get current directory"));

    // Create output directory if it doesn't exist
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;
    }

    // Serialize each aspect to Turtle format and write to files
    println!("\nGenerating Aspect Model files:");
    for (idx, aspect) in aspects.iter().enumerate() {
        let filename = format!("{}.ttl", aspect.name());
        let filepath = output_dir.join(&filename);

        // Serialize to Turtle format
        oxirs_samm::serializer::serialize_aspect_to_file(aspect, &filepath)
            .await
            .map_err(|e| format!("Failed to serialize aspect {}: {}", aspect.name(), e))?;

        println!(
            "  {}. {} ({})",
            idx + 1,
            aspect.name(),
            aspect.metadata().urn
        );
        println!("     Properties: {}", aspect.properties().len());
        println!("     Operations: {}", aspect.operations().len());
        println!("     Output: {}", filepath.display());
    }

    println!(
        "\n✓ Successfully generated {} Turtle file(s) in {}",
        aspects.len(),
        output_dir.display()
    );

    Ok(())
}

/// List submodel templates in an AAS file
///
/// This command parses an AAS file and displays all available submodel templates
/// with their indices, which can be used with the `-s` option in `to aspect`.
///
/// # Arguments
///
/// * `file` - Path to the AAS file (XML, JSON, or AASX)
///
/// # Examples
///
/// ```bash
/// # List all submodel templates in an AASX file
/// oxirs aas AssetAdminShell.aasx list
/// ```
async fn list(file: PathBuf) -> CliResult<()> {
    // Determine file format from extension
    let format = detect_aas_format(&file)?;

    println!("Listing submodel templates in AAS file...");
    println!("  File: {}", file.display());
    println!("  Format: {}", format);
    println!();

    // Check if file exists
    if !file.exists() {
        return Err(format!("AAS file not found: {}", file.display()).into());
    }

    // Parse the AAS file
    let env = oxirs_samm::aas_parser::parse_aas_file(&file)
        .await
        .map_err(|e| format!("Failed to parse AAS file: {}", e))?;

    // List submodels
    let submodels = oxirs_samm::aas_parser::list_submodels(&env);

    println!("✓ Found {} submodel template(s):\n", submodels.len());

    if submodels.is_empty() {
        println!("  (No submodel templates found in this AAS file)");
    } else {
        for (idx, id, name, description) in submodels {
            println!("  [{}] {}", idx, name.as_ref().unwrap_or(&id));
            println!("      ID: {}", id);

            if let Some(desc) = description {
                // Truncate long descriptions
                let display_desc = if desc.len() > 80 {
                    format!("{}...", &desc[..77])
                } else {
                    desc
                };
                println!("      Description: {}", display_desc);
            }

            println!();
        }

        println!("Use `-s <index>` with `to-aspect` to convert specific submodels.");
        println!(
            "Example: oxirs aas to-aspect {} -s 0 -s 1 -d output/",
            file.display()
        );
    }

    Ok(())
}

/// Detect AAS file format from file extension
///
/// # Arguments
///
/// * `file` - Path to the AAS file
///
/// # Returns
///
/// * `CliResult<&'static str>` - File format ("XML", "JSON", or "AASX")
fn detect_aas_format(file: &std::path::Path) -> CliResult<&'static str> {
    let extension = file
        .extension()
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| format!("Unable to determine file format from: {}", file.display()))?;

    match extension.to_lowercase().as_str() {
        "xml" => Ok("XML"),
        "json" => Ok("JSON"),
        "aasx" => Ok("AASX"),
        other => Err(format!(
            "Unsupported AAS file format: .{}\n\
             Supported formats: .xml, .json, .aasx",
            other
        )
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_detect_aas_format_xml() {
        let file = PathBuf::from("test.xml");
        assert_eq!(detect_aas_format(&file).unwrap(), "XML");
    }

    #[test]
    fn test_detect_aas_format_json() {
        let file = PathBuf::from("test.json");
        assert_eq!(detect_aas_format(&file).unwrap(), "JSON");
    }

    #[test]
    fn test_detect_aas_format_aasx() {
        let file = PathBuf::from("test.aasx");
        assert_eq!(detect_aas_format(&file).unwrap(), "AASX");
    }

    #[test]
    fn test_detect_aas_format_case_insensitive() {
        let file = PathBuf::from("test.AASX");
        assert_eq!(detect_aas_format(&file).unwrap(), "AASX");
    }

    #[test]
    fn test_detect_aas_format_unsupported() {
        let file = PathBuf::from("test.txt");
        assert!(detect_aas_format(&file).is_err());
    }

    #[test]
    fn test_detect_aas_format_no_extension() {
        let file = PathBuf::from("test");
        assert!(detect_aas_format(&file).is_err());
    }
}
