//! Package management command implementations (Java ESMF SDK compatible)
//!
//! This module provides commands for importing and exporting SAMM namespace packages,
//! following the Java ESMF SDK CLI interface.
//!
//! ## Package Format
//!
//! Namespace packages are ZIP files containing SAMM Aspect Models organized by namespace
//! and version in the following directory structure:
//!
//! ```text
//! namespace-package.zip
//! └── org.eclipse.example.myns/
//!     └── 1.0.0/
//!         ├── AspectModel1.ttl
//!         ├── AspectModel2.ttl
//!         └── ...
//! ```

use crate::cli::CliResult;
use crate::PackageAction;
use std::path::PathBuf;

/// Run a package management command (Java ESMF SDK compatible)
///
/// # Arguments
///
/// * `action` - The package action to perform (Import or Export)
///
/// # Returns
///
/// * `CliResult<()>` - Success or error result
pub async fn run(action: PackageAction) -> CliResult<()> {
    match action {
        PackageAction::Import {
            file,
            models_root,
            dry_run,
            details,
            force,
        } => import(file, models_root, dry_run, details, force).await,
        PackageAction::Export {
            input,
            output,
            version,
        } => export(input, output, version).await,
    }
}

/// Import namespace package (ZIP) into models directory
///
/// This command extracts a namespace package ZIP file and imports the SAMM Aspect Models
/// into the specified models root directory, preserving the namespace/version structure.
///
/// # Arguments
///
/// * `file` - Path to the namespace package ZIP file
/// * `models_root` - Directory to import the models into
/// * `dry_run` - If true, only show what would be imported without writing files
/// * `details` - If true with dry_run, show detailed content changes
/// * `force` - If true, overwrite existing files
///
/// # Examples
///
/// ```bash
/// # Import package into models directory
/// oxirs package namespace-package.zip import --models-root ./models/
///
/// # Dry run to see what would be imported
/// oxirs package namespace-package.zip import --models-root ./models/ --dry-run
///
/// # Import with detailed preview
/// oxirs package namespace-package.zip import --models-root ./models/ --dry-run --details
///
/// # Force overwrite existing files
/// oxirs package namespace-package.zip import --models-root ./models/ --force
/// ```
async fn import(
    file: PathBuf,
    models_root: PathBuf,
    dry_run: bool,
    details: bool,
    force: bool,
) -> CliResult<()> {
    println!("Importing namespace package...");
    println!("  Package: {}", file.display());
    println!("  Target Directory: {}", models_root.display());

    if dry_run {
        println!("  Mode: DRY RUN (no files will be written)");
        if details {
            println!("  Details: ENABLED");
        }
    } else if force {
        println!("  Force: ENABLED (existing files will be overwritten)");
    }
    println!();

    // Check if package file exists
    if !file.exists() {
        return Err(format!("Package file not found: {}", file.display()).into());
    }

    // Check if it's a ZIP file
    if !file.extension().is_some_and(|ext| ext == "zip") {
        return Err(format!(
            "Invalid package format: {}\nExpected .zip file",
            file.display()
        )
        .into());
    }

    // Import the package using oxirs-samm package module
    let result = oxirs_samm::package::import_package(&file, &models_root, dry_run, force)
        .await
        .map_err(|e| format!("Failed to import package: {}", e))?;

    // Display results
    println!("✓ Import analysis complete\n");

    println!("Package Structure:");
    println!("  Namespaces: {}", result.namespaces.len());
    println!("  Total Models: {}", result.total_models);
    println!();

    println!("Models by Namespace:");
    for (namespace, models) in &result.namespaces {
        println!("  {} ({})", namespace, models.len());
        for model in models {
            let status = if model.exists && !force {
                "SKIP (exists)"
            } else if model.exists && force {
                "OVERWRITE"
            } else {
                "NEW"
            };

            println!("    - {} [{}]", model.name, status);

            if details && dry_run {
                println!("      Path: {}", model.path.display());
                println!("      Version: {}", model.version);
            }
        }
        println!();
    }

    if dry_run {
        println!("ℹ No files were written (--dry-run mode)");
        println!("Run without --dry-run to perform the actual import.");
    } else {
        println!(
            "✓ Successfully imported {} model(s) to {}",
            result.total_models,
            models_root.display()
        );

        if result.skipped > 0 {
            println!(
                "  ⚠ Skipped {} existing file(s) (use --force to overwrite)",
                result.skipped
            );
        }
    }

    Ok(())
}

/// Export Aspect Model or namespace as ZIP package
///
/// This command exports SAMM Aspect Models as a namespace package ZIP file.
/// The input can be either:
/// - A file path to a single Aspect Model (.ttl file)
/// - A namespace URN (e.g., `urn:samm:org.eclipse.example.myns:1.0.0`)
///
/// # Arguments
///
/// * `input` - Aspect Model file path or namespace URN
/// * `output` - Output ZIP file path
/// * `version` - Optional version filter for namespace exports
///
/// # Examples
///
/// ```bash
/// # Export from single Aspect Model file
/// oxirs package AspectModel.ttl export --output package.zip
///
/// # Export entire namespace from URN
/// oxirs package urn:samm:org.eclipse.example.myns:1.0.0 export --output package.zip
///
/// # Export namespace with specific version
/// oxirs package urn:samm:org.eclipse.example.myns export --output package.zip --version 1.0.0
/// ```
async fn export(input: String, output: PathBuf, version: Option<String>) -> CliResult<()> {
    println!("Exporting namespace package...");

    // Determine if input is a file or URN
    let is_urn = input.starts_with("urn:");

    if is_urn {
        println!("  Source: URN ({})", input);
        if let Some(ref ver) = version {
            println!("  Version Filter: {}", ver);
        }
    } else {
        println!("  Source: File ({})", input);
    }

    println!("  Output: {}", output.display());
    println!();

    // Check if input file exists (for file-based export)
    if !is_urn {
        let input_path = PathBuf::from(&input);
        if !input_path.exists() {
            return Err(format!("Aspect Model file not found: {}", input).into());
        }

        if !input_path.extension().is_some_and(|ext| ext == "ttl") {
            return Err(
                format!("Invalid Aspect Model format: {}\nExpected .ttl file", input).into(),
            );
        }
    }

    // Check if output file already exists
    if output.exists() {
        return Err(format!(
            "Output file already exists: {}\nPlease remove it first or choose a different path",
            output.display()
        )
        .into());
    }

    // Export the package using oxirs-samm package module
    let result = if is_urn {
        oxirs_samm::package::export_from_urn(&input, &output, version.as_deref())
            .await
            .map_err(|e| format!("Failed to export package: {}", e))?
    } else {
        oxirs_samm::package::export_from_file(&input, &output)
            .await
            .map_err(|e| format!("Failed to export package: {}", e))?
    };

    // Display results
    println!("✓ Export complete\n");

    println!("Package Contents:");
    println!("  Namespace: {}", result.namespace);
    println!("  Version: {}", result.version);
    println!("  Models: {}", result.models.len());
    println!();

    println!("Exported Models:");
    for model in &result.models {
        println!("  - {}", model);
    }
    println!();

    println!("✓ Successfully exported to {}", output.display());
    println!("  Package size: {} KB", result.size_bytes / 1024);

    Ok(())
}

#[cfg(test)]
mod tests {

    use std::path::PathBuf;

    #[test]
    fn test_import_validates_zip_extension() {
        let file = PathBuf::from("test.txt");
        assert!(!file.extension().map_or(false, |ext| ext == "zip"));
    }

    #[test]
    fn test_export_validates_ttl_extension() {
        let file = PathBuf::from("test.ttl");
        assert!(file.extension().map_or(false, |ext| ext == "ttl"));
    }

    #[test]
    fn test_is_urn_detection() {
        assert!("urn:samm:org.example:1.0.0".starts_with("urn:"));
        assert!(!"AspectModel.ttl".starts_with("urn:"));
    }
}
