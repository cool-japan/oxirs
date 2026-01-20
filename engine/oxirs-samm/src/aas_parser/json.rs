//! JSON parser for AAS files
//!
//! This module provides parsing capabilities for AAS files in JSON format.

use super::models::AasEnvironment;
use crate::error::{Result, SammError};
use std::path::Path;
use tokio::fs;

/// Parse an AAS JSON file
///
/// # Arguments
///
/// * `path` - Path to the JSON file
///
/// # Returns
///
/// * `Result<AasEnvironment>` - Parsed AAS environment
pub async fn parse_json_file(path: &Path) -> Result<AasEnvironment> {
    // Read file contents
    let content = fs::read_to_string(path)
        .await
        .map_err(|e| SammError::ParseError(format!("Failed to read JSON file: {}", e)))?;

    parse_json_string(&content)
}

/// Parse AAS JSON from a string
///
/// # Arguments
///
/// * `json` - JSON content as string
///
/// # Returns
///
/// * `Result<AasEnvironment>` - Parsed AAS environment
pub fn parse_json_string(json: &str) -> Result<AasEnvironment> {
    serde_json::from_str(json)
        .map_err(|e| SammError::ParseError(format!("Failed to parse JSON: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_json() {
        let json = r#"{
            "assetAdministrationShells": [],
            "submodels": [],
            "conceptDescriptions": []
        }"#;

        let result = parse_json_string(json);
        assert!(result.is_ok());

        let env = result.unwrap();
        assert_eq!(env.asset_administration_shells.len(), 0);
        assert_eq!(env.submodels.len(), 0);
        assert_eq!(env.concept_descriptions.len(), 0);
    }

    #[test]
    fn test_parse_submodel() {
        let json = r#"{
            "assetAdministrationShells": [],
            "submodels": [{
                "id": "urn:submodel:example:1",
                "idShort": "ExampleSubmodel",
                "modelType": "Submodel",
                "submodelElements": []
            }],
            "conceptDescriptions": []
        }"#;

        let result = parse_json_string(json);
        assert!(result.is_ok());

        let env = result.unwrap();
        assert_eq!(env.submodels.len(), 1);
        assert_eq!(env.submodels[0].id, "urn:submodel:example:1");
        assert_eq!(
            env.submodels[0].id_short,
            Some("ExampleSubmodel".to_string())
        );
    }
}
