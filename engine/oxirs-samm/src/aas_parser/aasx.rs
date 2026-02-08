//! AASX (ZIP) parser for AAS files
//!
//! This module provides parsing capabilities for AASX package files.
//! AASX is a ZIP container format containing AAS data in XML or JSON format.

use super::models::AasEnvironment;
use super::{json, xml};
use crate::error::{Result, SammError};
use oxiarc_archive::ZipReader;
use std::path::Path;

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

    let mut archive = ZipReader::new(file)
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
        if let Some(entry) = archive.entry_by_name(xml_path) {
            let entry = entry.clone();
            let data = archive.extract(&entry).map_err(|e| {
                SammError::ParseError(format!("Failed to read XML from AASX: {}", e))
            })?;

            let content = String::from_utf8(data)
                .map_err(|e| SammError::ParseError(format!("Invalid UTF-8 in XML: {}", e)))?;

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
            if let Some(entry) = archive.entry_by_name(json_path) {
                let entry = entry.clone();
                let data = archive.extract(&entry).map_err(|e| {
                    SammError::ParseError(format!("Failed to read JSON from AASX: {}", e))
                })?;

                let content = String::from_utf8(data)
                    .map_err(|e| SammError::ParseError(format!("Invalid UTF-8 in JSON: {}", e)))?;

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
    async fn test_basic_zip_round_trip() {
        // Test basic ZIP creation and reading
        let content = b"Hello, World!";

        // Create ZIP
        let mut zip = oxiarc_archive::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        zip.set_compression(oxiarc_archive::ZipCompressionLevel::Normal);
        zip.add_file("test.txt", content).unwrap();
        let cursor = zip.into_inner().unwrap();
        let zip_data = cursor.into_inner();

        eprintln!("Created ZIP with {} bytes", zip_data.len());

        // Read ZIP
        let mut reader = oxiarc_archive::ZipReader::new(std::io::Cursor::new(&zip_data)).unwrap();
        eprintln!("ZIP has {} entries", reader.entries().len());

        let entry = reader.entry_by_name("test.txt").unwrap().clone();
        let data = reader.extract(&entry).unwrap();

        assert_eq!(data, content);
    }

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

        // Create ZIP archive in memory first
        let zip_data = {
            let mut zip = oxiarc_archive::ZipWriter::new(std::io::Cursor::new(Vec::new()));

            // Use normal deflate compression (bug fixed in oxiarc-archive 0.3.0)
            zip.set_compression(oxiarc_archive::ZipCompressionLevel::Normal);

            // Add JSON content file
            zip.add_file("aasx/json/content.json", json_content.as_bytes())
                .unwrap();

            // Get the ZIP data
            let cursor = zip.into_inner().unwrap();
            cursor.into_inner()
        };

        // Write ZIP data to temp file
        temp_file.write_all(&zip_data).unwrap();
        temp_file.flush().unwrap();

        // Parse the AASX file
        let result = parse_aasx_file(temp_file.path()).await;
        assert!(result.is_ok(), "Failed to parse AASX: {:?}", result.err());

        let env = result.unwrap();
        assert_eq!(env.submodels.len(), 1);
        assert_eq!(env.submodels[0].id, "urn:submodel:test:1");
    }
}
