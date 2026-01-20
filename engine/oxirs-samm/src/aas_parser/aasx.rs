//! AASX (ZIP) parser for AAS files
//!
//! This module provides parsing capabilities for AASX package files.
//! AASX is a ZIP container format containing AAS data in XML or JSON format.

use super::models::AasEnvironment;
use super::{json, xml};
use crate::error::{Result, SammError};
use std::io::Read;
use std::path::Path;
use zip::ZipArchive;

/// Parse an AASX file (ZIP package)
///
/// # Arguments
///
/// * `path` - Path to the AASX file
///
/// # Returns
///
/// * `Result<AasEnvironment>` - Parsed AAS environment
///
/// # AASX Structure
///
/// An AASX package typically contains:
/// - `/aasx/aasx-origin` - Origin file
/// - `/aasx/xml/content.xml` or `/aasx/json/content.json` - AAS data
/// - `/aasx/thumbnails/` - Optional thumbnail images
pub async fn parse_aasx_file(path: &Path) -> Result<AasEnvironment> {
    // Read the ZIP file
    let file = std::fs::File::open(path)
        .map_err(|e| SammError::ParseError(format!("Failed to open AASX file: {}", e)))?;

    let mut archive = ZipArchive::new(file)
        .map_err(|e| SammError::ParseError(format!("Failed to read AASX as ZIP: {}", e)))?;

    // Try to find the content file (XML or JSON)
    // Common paths in AASX packages:
    // - aasx/xml/content.xml
    // - aasx/json/content.json
    // - xml/content.xml
    // - content.xml
    // - aasx-spec-3.0.xml

    let mut env: Option<AasEnvironment> = None;

    // First, try to find XML content
    let xml_paths = vec![
        "aasx/xml/content.xml",
        "xml/content.xml",
        "content.xml",
        "aasx-spec-3.0.xml",
    ];

    for xml_path in xml_paths {
        if let Ok(mut file) = archive.by_name(xml_path) {
            let mut content = String::new();
            file.read_to_string(&mut content).map_err(|e| {
                SammError::ParseError(format!("Failed to read XML from AASX: {}", e))
            })?;

            env = Some(xml::parse_xml_string(&content)?);
            break;
        }
    }

    // If XML not found, try JSON
    if env.is_none() {
        let json_paths = vec![
            "aasx/json/content.json",
            "json/content.json",
            "content.json",
        ];

        for json_path in json_paths {
            if let Ok(mut file) = archive.by_name(json_path) {
                let mut content = String::new();
                file.read_to_string(&mut content).map_err(|e| {
                    SammError::ParseError(format!("Failed to read JSON from AASX: {}", e))
                })?;

                env = Some(json::parse_json_string(&content)?);
                break;
            }
        }
    }

    env.ok_or_else(|| {
        SammError::ParseError(
            "No AAS content found in AASX package. Expected 'aasx/xml/content.xml' or 'aasx/json/content.json'".into()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_parse_aasx_with_json() {
        // Create a temporary AASX file (ZIP) with JSON content
        let mut temp_file = NamedTempFile::new().unwrap();

        // Create a minimal AAS JSON
        let json_content = r#"{
            "assetAdministrationShells": [],
            "submodels": [{
                "id": "urn:submodel:test:1",
                "idShort": "TestSubmodel",
                "modelType": "Submodel",
                "submodelElements": []
            }],
            "conceptDescriptions": []
        }"#;

        // Create ZIP archive
        {
            let file = temp_file.as_file_mut();
            let mut zip = zip::ZipWriter::new(file);

            let options = zip::write::FileOptions::<()>::default()
                .compression_method(zip::CompressionMethod::Deflated);

            // Add JSON content file
            zip.start_file("aasx/json/content.json", options).unwrap();
            zip.write_all(json_content.as_bytes()).unwrap();

            zip.finish().unwrap();
        }

        // Parse the AASX file
        let result = parse_aasx_file(temp_file.path()).await;
        assert!(result.is_ok());

        let env = result.unwrap();
        assert_eq!(env.submodels.len(), 1);
        assert_eq!(env.submodels[0].id, "urn:submodel:test:1");
    }
}
