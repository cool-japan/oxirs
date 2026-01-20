//! AAS (Asset Administration Shell) parser module
//!
//! This module provides parsing capabilities for AAS files in XML, JSON, and AASX formats,
//! and conversion to SAMM Aspect Models.

pub mod aasx;
pub mod converter;
pub mod json;
pub mod models;
pub mod xml;

use crate::error::Result;
use crate::metamodel::Aspect;
use models::AasEnvironment;
use std::path::Path;

/// AAS file format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AasFormat {
    /// XML format
    Xml,
    /// JSON format
    Json,
    /// AASX (ZIP) format
    Aasx,
}

impl AasFormat {
    /// Detect format from file extension
    pub fn from_path(path: &Path) -> Result<Self> {
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| {
                crate::SammError::ParseError("Unable to determine file format".into())
            })?;

        match extension.to_lowercase().as_str() {
            "xml" => Ok(AasFormat::Xml),
            "json" => Ok(AasFormat::Json),
            "aasx" => Ok(AasFormat::Aasx),
            other => Err(crate::SammError::ParseError(format!(
                "Unsupported AAS file format: .{}",
                other
            ))),
        }
    }
}

/// Parse an AAS file and return the AAS environment
///
/// # Arguments
///
/// * `path` - Path to the AAS file (XML, JSON, or AASX)
///
/// # Returns
///
/// * `Result<AasEnvironment>` - Parsed AAS environment
///
/// # Examples
///
/// ```rust,no_run
/// use oxirs_samm::aas_parser::parse_aas_file;
/// use std::path::Path;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let env = parse_aas_file(Path::new("AssetAdminShell.aasx")).await?;
/// println!("Parsed {} submodels", env.submodels.len());
/// # Ok(())
/// # }
/// ```
pub async fn parse_aas_file(path: &Path) -> Result<AasEnvironment> {
    let format = AasFormat::from_path(path)?;

    match format {
        AasFormat::Xml => xml::parse_xml_file(path).await,
        AasFormat::Json => json::parse_json_file(path).await,
        AasFormat::Aasx => aasx::parse_aasx_file(path).await,
    }
}

/// Convert AAS environment to SAMM Aspect Models
///
/// # Arguments
///
/// * `env` - AAS environment
/// * `submodel_indices` - Optional list of submodel indices to convert (empty = all)
///
/// # Returns
///
/// * `Result<Vec<Aspect>>` - List of converted SAMM Aspect models
///
/// # Examples
///
/// ```rust,no_run
/// use oxirs_samm::aas_parser::{parse_aas_file, convert_to_aspects};
/// use std::path::Path;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let env = parse_aas_file(Path::new("AssetAdminShell.aasx")).await?;
/// let aspects = convert_to_aspects(&env, vec![])?;  // Convert all submodels
/// println!("Converted {} aspects", aspects.len());
/// # Ok(())
/// # }
/// ```
pub fn convert_to_aspects(
    env: &AasEnvironment,
    submodel_indices: Vec<usize>,
) -> Result<Vec<Aspect>> {
    converter::convert_environment_to_aspects(env, submodel_indices)
}

/// List submodel templates in an AAS environment
///
/// # Arguments
///
/// * `env` - AAS environment
///
/// # Returns
///
/// * List of (index, id, name, description) tuples
pub fn list_submodels(
    env: &AasEnvironment,
) -> Vec<(usize, String, Option<String>, Option<String>)> {
    env.submodels
        .iter()
        .enumerate()
        .map(|(idx, submodel)| {
            let id = submodel.id.clone();
            let name = submodel.id_short.clone();
            let description = submodel
                .description
                .as_ref()
                .and_then(|descs| descs.first().map(|lang_str| lang_str.text.clone()));
            (idx, id, name, description)
        })
        .collect()
}
