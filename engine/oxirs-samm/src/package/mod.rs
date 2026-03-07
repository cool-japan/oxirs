//! Package management for SAMM namespace packages
//!
//! This module provides import and export functionality for SAMM namespace packages,
//! which are ZIP files containing Aspect Models organized by namespace and version.
//!
//! ## Package Format
//!
//! A namespace package follows this structure:
//!
//! ```text
//! namespace-package.zip
//! └── org.eclipse.example.myns/
//!     └── 1.0.0/
//!         ├── AspectModel1.ttl
//!         ├── AspectModel2.ttl
//!         └── ...
//! ```
//!
//! ## Features
//!
//! - Import ZIP packages into models directory
//! - Export models to ZIP packages
//! - Dry-run mode for previewing changes
//! - Force mode for overwriting existing files
//! - URN-based and file-based exports

use crate::error::{Result, SammError};
use crate::performance::profiling;
use oxiarc_archive::{ZipCompressionLevel, ZipReader, ZipWriter};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tokio::fs as async_fs;

/// Result of a package import operation
#[derive(Debug, Clone)]
pub struct ImportResult {
    /// Models organized by namespace
    pub namespaces: HashMap<String, Vec<ModelInfo>>,
    /// Total number of models
    pub total_models: usize,
    /// Number of models skipped (already exist)
    pub skipped: usize,
}

/// Information about a model in a package
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model name (without .ttl extension)
    pub name: String,
    /// Namespace
    pub namespace: String,
    /// Version
    pub version: String,
    /// Full path where model will be/is located
    pub path: PathBuf,
    /// Whether the file already exists
    pub exists: bool,
}

/// Result of a package export operation
#[derive(Debug, Clone)]
pub struct ExportResult {
    /// Namespace of the exported package
    pub namespace: String,
    /// Version of the exported package
    pub version: String,
    /// List of model names included
    pub models: Vec<String>,
    /// Total size in bytes
    pub size_bytes: u64,
}

/// Import a namespace package from a ZIP file
///
/// # Arguments
///
/// * `package_path` - Path to the ZIP file
/// * `models_root` - Target directory for models
/// * `dry_run` - If true, only analyze without writing files
/// * `force` - If true, overwrite existing files
///
/// # Returns
///
/// * `Result<ImportResult>` - Import result with statistics
///
/// # Examples
///
/// ```rust,no_run
/// use oxirs_samm::package;
/// use std::path::Path;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let result = package::import_package(
///     Path::new("namespace-package.zip"),
///     Path::new("./models"),
///     false,
///     false,
/// ).await?;
///
/// println!("Imported {} models", result.total_models);
/// # Ok(())
/// # }
/// ```
pub async fn import_package(
    package_path: &Path,
    models_root: &Path,
    dry_run: bool,
    force: bool,
) -> Result<ImportResult> {
    // Profile the entire import operation
    let (result, duration) = profiling::profile_async(
        "package_import",
        import_package_impl(package_path, models_root, dry_run, force),
    )
    .await;

    tracing::info!(
        "Package import completed in {:?}: {} models ({} skipped)",
        duration,
        result.as_ref().map(|r| r.total_models).unwrap_or(0),
        result.as_ref().map(|r| r.skipped).unwrap_or(0)
    );

    result
}

/// Internal implementation of import_package with profiling support
async fn import_package_impl(
    package_path: &Path,
    models_root: &Path,
    dry_run: bool,
    force: bool,
) -> Result<ImportResult> {
    // Open the ZIP file
    let file = std::fs::File::open(package_path)
        .map_err(|e| SammError::ParseError(format!("Failed to open package file: {}", e)))?;

    let mut archive = ZipReader::new(file)
        .map_err(|e| SammError::ParseError(format!("Invalid ZIP package: {}", e)))?;

    let mut namespaces: HashMap<String, Vec<ModelInfo>> = HashMap::new();
    let mut total_models = 0;
    let mut skipped = 0;

    // First, collect entry information (to avoid borrowing issues)
    let entries_to_process: Vec<_> = archive
        .entries()
        .iter()
        .filter(|entry| !entry.is_dir() && entry.name.ends_with(".ttl"))
        .cloned()
        .collect();

    // Process each entry
    for entry in entries_to_process {
        let file_path = entry.name.clone();

        // Parse the path: namespace/version/filename.ttl
        let parts: Vec<&str> = file_path.split('/').collect();
        if parts.len() < 3 {
            continue; // Invalid structure, skip
        }

        let namespace = parts[parts.len() - 3].to_string();
        let version = parts[parts.len() - 2].to_string();
        let filename = parts[parts.len() - 1];
        let model_name = filename.trim_end_matches(".ttl").to_string();

        // Construct target path
        let target_path = models_root.join(&namespace).join(&version).join(filename);

        let exists = target_path.exists();

        // Determine if we should skip this file
        if exists && !force {
            skipped += 1;
        }

        // Create model info
        let model_info = ModelInfo {
            name: model_name,
            namespace: namespace.clone(),
            version: version.clone(),
            path: target_path.clone(),
            exists,
        };

        // Add to namespace map
        namespaces
            .entry(namespace.clone())
            .or_default()
            .push(model_info);

        total_models += 1;

        // Write file if not in dry-run mode and (force or doesn't exist)
        if !dry_run && (force || !exists) {
            // Create parent directory
            if let Some(parent) = target_path.parent() {
                async_fs::create_dir_all(parent).await?;
            }

            // Read content from ZIP
            let content = archive.extract(&entry).map_err(|e| {
                SammError::ParseError(format!("Failed to extract file from ZIP: {}", e))
            })?;

            // Write to target path
            async_fs::write(&target_path, content).await?;
        }
    }

    Ok(ImportResult {
        namespaces,
        total_models,
        skipped,
    })
}

/// Export a single Aspect Model file to a namespace package
///
/// # Arguments
///
/// * `model_file` - Path to the Aspect Model TTL file
/// * `output_path` - Output ZIP file path
///
/// # Returns
///
/// * `Result<ExportResult>` - Export result with statistics
///
/// # Examples
///
/// ```rust,no_run
/// use oxirs_samm::package;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let result = package::export_from_file(
///     "AspectModel.ttl",
///     std::path::Path::new("package.zip"),
/// ).await?;
///
/// println!("Exported {} models", result.models.len());
/// # Ok(())
/// # }
/// ```
pub async fn export_from_file(model_file: &str, output_path: &Path) -> Result<ExportResult> {
    // Profile the entire export operation
    let (result, duration) = profiling::profile_async(
        "package_export_file",
        export_from_file_impl(model_file, output_path),
    )
    .await;

    tracing::info!(
        "Package export from file completed in {:?}: {}",
        duration,
        model_file
    );

    result
}

/// Internal implementation of export_from_file with profiling support
async fn export_from_file_impl(model_file: &str, output_path: &Path) -> Result<ExportResult> {
    let model_path = PathBuf::from(model_file);

    // Read the model file
    let content = async_fs::read_to_string(&model_path).await?;

    // Parse the file to extract namespace and version from URN
    let (namespace, version) = extract_namespace_from_content(&content)?;

    // Get model name from filename
    let model_name = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| SammError::ParseError("Invalid filename".to_string()))?
        .to_string();

    // Create ZIP file
    let zip_file = std::fs::File::create(output_path).map_err(SammError::Io)?;

    let mut zip = ZipWriter::new(zip_file);

    // Set compression level
    zip.set_compression(ZipCompressionLevel::Normal);

    // Add file to ZIP with proper structure: namespace/version/filename.ttl
    let zip_path = format!("{}/{}/{}.ttl", namespace, version, model_name);
    zip.add_file(&zip_path, content.as_bytes())
        .map_err(|e| SammError::ParseError(format!("Failed to add ZIP entry: {}", e)))?;

    let file = zip
        .into_inner()
        .map_err(|e| SammError::ParseError(format!("Failed to finalize ZIP: {}", e)))?;

    let size_bytes = file.metadata().map(|m| m.len()).unwrap_or(0);

    Ok(ExportResult {
        namespace,
        version,
        models: vec![model_name],
        size_bytes,
    })
}

/// Export models from a namespace URN to a package
///
/// # Arguments
///
/// * `urn` - Namespace URN (e.g., "urn:samm:org.eclipse.example.myns:1.0.0")
/// * `output_path` - Output ZIP file path
/// * `version_filter` - Optional version filter
///
/// # Returns
///
/// * `Result<ExportResult>` - Export result with statistics
///
/// # Examples
///
/// ```rust,no_run
/// use oxirs_samm::package;
/// use std::path::Path;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let result = package::export_from_urn(
///     "urn:samm:org.eclipse.example.myns:1.0.0",
///     Path::new("package.zip"),
///     None,
/// ).await?;
///
/// println!("Exported {} models", result.models.len());
/// # Ok(())
/// # }
/// ```
pub async fn export_from_urn(
    urn: &str,
    output_path: &Path,
    version_filter: Option<&str>,
) -> Result<ExportResult> {
    // Profile the entire export operation
    let (result, duration) = profiling::profile_async(
        "package_export_urn",
        export_from_urn_impl(urn, output_path, version_filter),
    )
    .await;

    tracing::info!(
        "Package export from URN completed in {:?}: {} ({} models)",
        duration,
        urn,
        result.as_ref().map(|r| r.models.len()).unwrap_or(0)
    );

    result
}

/// Internal implementation of export_from_urn with profiling support
async fn export_from_urn_impl(
    urn: &str,
    output_path: &Path,
    version_filter: Option<&str>,
) -> Result<ExportResult> {
    use tokio::fs as async_fs;

    // Parse URN to extract namespace and version
    let (namespace, version) = parse_urn(urn)?;

    // Use version filter if provided
    let final_version = version_filter.unwrap_or(&version);

    // Determine models root directory
    // Try these in order:
    // 1. Environment variable SAMM_MODELS_ROOT
    // 2. ./models if it exists
    // 3. Current directory
    let models_root = if let Ok(env_path) = std::env::var("SAMM_MODELS_ROOT") {
        PathBuf::from(env_path)
    } else if PathBuf::from("./models").exists() {
        PathBuf::from("./models")
    } else {
        std::env::current_dir().map_err(SammError::Io)?
    };

    // Construct path to namespace/version directory
    let namespace_dir = models_root.join(&namespace).join(final_version);

    // Check if directory exists
    if !namespace_dir.exists() {
        return Err(SammError::ParseError(format!(
            "Namespace directory not found: {} (models root: {})",
            namespace_dir.display(),
            models_root.display()
        )));
    }

    // Scan directory for .ttl files
    let mut entries = async_fs::read_dir(&namespace_dir)
        .await
        .map_err(SammError::Io)?;

    let mut model_files: Vec<PathBuf> = Vec::new();
    while let Some(entry) = entries.next_entry().await.map_err(SammError::Io)? {
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("ttl") {
            model_files.push(path);
        }
    }

    if model_files.is_empty() {
        return Err(SammError::ParseError(format!(
            "No .ttl files found in namespace directory: {}",
            namespace_dir.display()
        )));
    }

    // Create ZIP file
    let zip_file = std::fs::File::create(output_path).map_err(SammError::Io)?;

    let mut zip = ZipWriter::new(zip_file);

    // Set compression level
    zip.set_compression(ZipCompressionLevel::Normal);

    let mut model_names = Vec::new();

    // Add all model files to ZIP
    for model_file in &model_files {
        let content = async_fs::read_to_string(model_file)
            .await
            .map_err(SammError::Io)?;

        // Get model name from filename
        let model_name = model_file
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| SammError::ParseError("Invalid filename".to_string()))?;

        model_names.push(model_name.to_string());

        // Add file to ZIP with proper structure: namespace/version/filename.ttl
        let zip_path = format!("{}/{}/{}.ttl", namespace, final_version, model_name);
        zip.add_file(&zip_path, content.as_bytes())
            .map_err(|e| SammError::ParseError(format!("Failed to add ZIP entry: {}", e)))?;
    }

    let file = zip
        .into_inner()
        .map_err(|e| SammError::ParseError(format!("Failed to finalize ZIP: {}", e)))?;

    let size_bytes = file.metadata().map(|m| m.len()).unwrap_or(0);

    Ok(ExportResult {
        namespace,
        version: final_version.to_string(),
        models: model_names,
        size_bytes,
    })
}

/// Extract namespace and version from Turtle content
fn extract_namespace_from_content(content: &str) -> Result<(String, String)> {
    // Look for Aspect URN pattern first: <urn:samm:namespace:version#Element> a samm:Aspect
    // This avoids matching prefix declarations like @prefix samm: <urn:...>
    for line in content.lines() {
        // Skip prefix declarations
        if line.trim().starts_with("@prefix") {
            continue;
        }

        // Look for Aspect definitions: <urn:...> a samm:Aspect
        if line.contains("urn:samm:")
            && (line.contains("a samm:Aspect")
                || line.contains("a samm:Property")
                || line.contains("a samm:Operation"))
        {
            // Find URN pattern between < and >
            if let Some(start) = line.find("<urn:samm:") {
                let urn_part = &line[start + 1..]; // Skip the '<'
                if let Some(end) = urn_part.find('>') {
                    let urn = &urn_part[..end];
                    // Also need to strip the #Element part for parsing
                    let urn_without_element = if let Some(hash_pos) = urn.find('#') {
                        &urn[..hash_pos]
                    } else {
                        urn
                    };

                    if let Ok((namespace, version)) = parse_urn(urn_without_element) {
                        // Skip meta-model and characteristic namespaces
                        if !namespace.contains("esmf.samm") {
                            return Ok((namespace, version));
                        }
                    }
                }
            }
        }
    }

    // Fallback: look for any URN
    for line in content.lines() {
        if line.contains("urn:samm:") && !line.trim().starts_with("@prefix") {
            if let Some(start) = line.find("urn:samm:") {
                let urn_part = &line[start..];
                if let Some(end) = urn_part.find(['>', '#', ' ']) {
                    let urn = &urn_part[..end];
                    if let Ok((namespace, version)) = parse_urn(urn) {
                        if !namespace.contains("esmf.samm") {
                            return Ok((namespace, version));
                        }
                    }
                }
            }
        }
    }

    // Default fallback
    Ok(("org.example.default".to_string(), "1.0.0".to_string()))
}

/// Parse a SAMM URN into namespace and version components
///
/// Format: urn:samm:namespace:version#Element
fn parse_urn(urn: &str) -> Result<(String, String)> {
    // Remove "urn:samm:" prefix
    let without_prefix = urn
        .strip_prefix("urn:samm:")
        .ok_or_else(|| SammError::ParseError(format!("Invalid URN format: {}", urn)))?;

    // Split by '#' to remove element name if present
    let without_element = without_prefix.split('#').next().unwrap_or(without_prefix);

    // Split by ':' to get namespace and version
    let parts: Vec<&str> = without_element.rsplitn(2, ':').collect();

    if parts.len() == 2 {
        let version = parts[0].to_string();
        let namespace = parts[1].to_string();
        Ok((namespace, version))
    } else {
        // No version specified, use default
        Ok((without_element.to_string(), "1.0.0".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_urn_with_version() {
        let (namespace, version) = parse_urn("urn:samm:org.eclipse.example:1.0.0#Movement")
            .expect("parsing should succeed");
        assert_eq!(namespace, "org.eclipse.example");
        assert_eq!(version, "1.0.0");
    }

    #[test]
    fn test_parse_urn_without_element() {
        let (namespace, version) =
            parse_urn("urn:samm:org.eclipse.example:2.1.0").expect("parsing should succeed");
        assert_eq!(namespace, "org.eclipse.example");
        assert_eq!(version, "2.1.0");
    }

    #[test]
    fn test_parse_urn_without_version() {
        let (namespace, version) =
            parse_urn("urn:samm:org.eclipse.example").expect("parsing should succeed");
        assert_eq!(namespace, "org.eclipse.example");
        assert_eq!(version, "1.0.0"); // Default version
    }

    #[test]
    fn test_parse_urn_invalid() {
        assert!(parse_urn("invalid:urn").is_err());
        assert!(parse_urn("urn:other:namespace").is_err());
    }

    #[test]
    fn test_extract_namespace_from_content() {
        let content = r#"
@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.1.0#> .

<urn:samm:org.eclipse.example:1.0.0#Movement> a samm:Aspect .
"#;
        let (namespace, version) =
            extract_namespace_from_content(content).expect("operation should succeed");
        assert_eq!(namespace, "org.eclipse.example");
        assert_eq!(version, "1.0.0");
    }
}
