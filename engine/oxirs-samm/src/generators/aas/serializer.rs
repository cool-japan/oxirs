//! AAS serialization (JSON, XML, AASX)

use super::environment::{AssetAdministrationShell, Environment, Submodel, SubmodelElement};
use crate::error::SammError;
use quick_xml::se::to_string as xml_to_string;
use std::io::Write;
use std::path::Path;
use zip::write::{FileOptions, ZipWriter};
use zip::CompressionMethod;

#[cfg(feature = "aasx-thumbnails")]
use image::{imageops, DynamicImage, ImageFormat};

/// Serialize AAS Environment to JSON
pub fn serialize_json(env: &Environment) -> Result<String, SammError> {
    serde_json::to_string_pretty(env)
        .map_err(|e| SammError::Generation(format!("JSON serialization failed: {}", e)))
}

/// Serialize AAS Environment to XML
pub fn serialize_xml(env: &Environment) -> Result<String, SammError> {
    // Generate XML using quick-xml
    let mut xml = String::from(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    xml.push('\n');
    xml.push_str(r#"<environment xmlns="https://admin-shell.io/aas/3/0">"#);
    xml.push('\n');

    // Serialize Asset Administration Shells
    if let Some(shells) = &env.asset_administration_shells {
        xml.push_str("  <assetAdministrationShells>\n");
        for shell in shells {
            xml.push_str(&serialize_shell_to_xml(shell)?);
        }
        xml.push_str("  </assetAdministrationShells>\n");
    }

    // Serialize Submodels
    if let Some(submodels) = &env.submodels {
        xml.push_str("  <submodels>\n");
        for submodel in submodels {
            xml.push_str(&serialize_submodel_to_xml(submodel)?);
        }
        xml.push_str("  </submodels>\n");
    }

    // Serialize Concept Descriptions
    if let Some(concepts) = &env.concept_descriptions {
        xml.push_str("  <conceptDescriptions>\n");
        for concept in concepts {
            xml.push_str(&format!(
                "    <conceptDescription>\n      <id>{}</id>\n    </conceptDescription>\n",
                escape_xml(&concept.id)
            ));
        }
        xml.push_str("  </conceptDescriptions>\n");
    }

    xml.push_str("</environment>\n");
    Ok(xml)
}

/// Serialize AssetAdministrationShell to XML
fn serialize_shell_to_xml(shell: &AssetAdministrationShell) -> Result<String, SammError> {
    let mut xml = String::from("    <assetAdministrationShell>\n");
    xml.push_str(&format!("      <id>{}</id>\n", escape_xml(&shell.id)));

    if let Some(id_short) = &shell.id_short {
        xml.push_str(&format!(
            "      <idShort>{}</idShort>\n",
            escape_xml(id_short)
        ));
    }

    xml.push_str(&format!(
        "      <modelType>{}</modelType>\n",
        escape_xml(&shell.model_type)
    ));

    xml.push_str("      <assetInformation>\n");
    xml.push_str(&format!(
        "        <assetKind>{:?}</assetKind>\n",
        shell.asset_information.asset_kind
    ));
    if let Some(global_asset_id) = &shell.asset_information.global_asset_id {
        xml.push_str(&format!(
            "        <globalAssetId>{}</globalAssetId>\n",
            escape_xml(global_asset_id)
        ));
    }
    xml.push_str("      </assetInformation>\n");

    if let Some(submodels) = &shell.submodels {
        xml.push_str("      <submodels>\n");
        for submodel_ref in submodels {
            xml.push_str("        <reference>\n");
            xml.push_str(&format!(
                "          <type>{:?}</type>\n",
                submodel_ref.ref_type
            ));
            xml.push_str("          <keys>\n");
            for key in &submodel_ref.keys {
                xml.push_str("            <key>\n");
                xml.push_str(&format!("              <type>{:?}</type>\n", key.key_type));
                xml.push_str(&format!(
                    "              <value>{}</value>\n",
                    escape_xml(&key.value)
                ));
                xml.push_str("            </key>\n");
            }
            xml.push_str("          </keys>\n");
            xml.push_str("        </reference>\n");
        }
        xml.push_str("      </submodels>\n");
    }

    xml.push_str("    </assetAdministrationShell>\n");
    Ok(xml)
}

/// Serialize Submodel to XML
fn serialize_submodel_to_xml(submodel: &Submodel) -> Result<String, SammError> {
    let mut xml = String::from("    <submodel>\n");
    xml.push_str(&format!("      <id>{}</id>\n", escape_xml(&submodel.id)));

    if let Some(id_short) = &submodel.id_short {
        xml.push_str(&format!(
            "      <idShort>{}</idShort>\n",
            escape_xml(id_short)
        ));
    }

    xml.push_str(&format!(
        "      <modelType>{}</modelType>\n",
        escape_xml(&submodel.model_type)
    ));

    if let Some(kind) = &submodel.kind {
        xml.push_str(&format!("      <kind>{:?}</kind>\n", kind));
    }

    if let Some(elements) = &submodel.submodel_elements {
        xml.push_str("      <submodelElements>\n");
        for element in elements {
            xml.push_str(&serialize_element_to_xml(element)?);
        }
        xml.push_str("      </submodelElements>\n");
    }

    xml.push_str("    </submodel>\n");
    Ok(xml)
}

/// Serialize SubmodelElement to XML
fn serialize_element_to_xml(element: &SubmodelElement) -> Result<String, SammError> {
    match element {
        SubmodelElement::Property(prop) => {
            let mut xml = String::from("        <property>\n");
            if let Some(id_short) = &prop.id_short {
                xml.push_str(&format!(
                    "          <idShort>{}</idShort>\n",
                    escape_xml(id_short)
                ));
            }
            xml.push_str(&format!(
                "          <modelType>{}</modelType>\n",
                escape_xml(&prop.model_type)
            ));
            xml.push_str(&format!(
                "          <valueType>{}</valueType>\n",
                escape_xml(&prop.value_type)
            ));
            if let Some(value) = &prop.value {
                xml.push_str(&format!("          <value>{}</value>\n", escape_xml(value)));
            }
            xml.push_str("        </property>\n");
            Ok(xml)
        }
        SubmodelElement::Operation(op) => {
            let mut xml = String::from("        <operation>\n");
            if let Some(id_short) = &op.id_short {
                xml.push_str(&format!(
                    "          <idShort>{}</idShort>\n",
                    escape_xml(id_short)
                ));
            }
            xml.push_str(&format!(
                "          <modelType>{}</modelType>\n",
                escape_xml(&op.model_type)
            ));
            xml.push_str("        </operation>\n");
            Ok(xml)
        }
    }
}

/// Escape XML special characters
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// AASX generation options
#[derive(Debug, Clone)]
pub struct AasxOptions {
    /// Optional path to custom thumbnail image
    pub thumbnail_path: Option<std::path::PathBuf>,
    /// Thumbnail size (width, height) in pixels, default: (256, 256)
    pub thumbnail_size: (u32, u32),
}

impl Default for AasxOptions {
    fn default() -> Self {
        Self {
            thumbnail_path: None,
            thumbnail_size: (256, 256),
        }
    }
}

/// Serialize AAS Environment to AASX package (ZIP) with default options
pub fn serialize_aasx(env: &Environment) -> Result<Vec<u8>, SammError> {
    serialize_aasx_with_options(env, AasxOptions::default())
}

/// Serialize AAS Environment to AASX package (ZIP) with custom options
pub fn serialize_aasx_with_options(
    env: &Environment,
    aasx_options: AasxOptions,
) -> Result<Vec<u8>, SammError> {
    let mut zip_data = Vec::new();
    let mut zip = ZipWriter::new(std::io::Cursor::new(&mut zip_data));

    let file_options: FileOptions<()> = FileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .unix_permissions(0o644);

    // Add XML content
    let xml_content = serialize_xml(env)?;
    zip.start_file("aasx/xml/content.xml", file_options)
        .map_err(|e| SammError::Generation(format!("Failed to create AASX XML entry: {}", e)))?;
    zip.write_all(xml_content.as_bytes())
        .map_err(|e| SammError::Generation(format!("Failed to write AASX XML: {}", e)))?;

    // Add AASX manifest
    let manifest = create_aasx_manifest()?;
    zip.start_file("aasx/aasx-origin", file_options)
        .map_err(|e| SammError::Generation(format!("Failed to create manifest entry: {}", e)))?;
    zip.write_all(manifest.as_bytes())
        .map_err(|e| SammError::Generation(format!("Failed to write manifest: {}", e)))?;

    // Add thumbnail (custom or default)
    let thumbnail = if let Some(thumbnail_path) = &aasx_options.thumbnail_path {
        load_and_resize_thumbnail(thumbnail_path, aasx_options.thumbnail_size)?
    } else {
        create_thumbnail_placeholder()
    };

    zip.start_file("aasx/thumbnail.png", file_options)
        .map_err(|e| SammError::Generation(format!("Failed to create thumbnail entry: {}", e)))?;
    zip.write_all(&thumbnail)
        .map_err(|e| SammError::Generation(format!("Failed to write thumbnail: {}", e)))?;

    zip.finish()
        .map_err(|e| SammError::Generation(format!("Failed to finalize AASX: {}", e)))?;

    Ok(zip_data)
}

/// Create AASX manifest file
fn create_aasx_manifest() -> Result<String, SammError> {
    Ok(r#"<?xml version="1.0" encoding="UTF-8"?>
<aasx-origin xmlns="http://www.admin-shell.io/aasx/3/0">
  <origin>/aasx/xml/content.xml</origin>
</aasx-origin>"#
        .to_string())
}

/// Create placeholder thumbnail (1x1 PNG)
fn create_thumbnail_placeholder() -> Vec<u8> {
    // Minimal valid 1x1 transparent PNG
    vec![
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1 dimensions
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, // RGBA, deflate
        0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, // IDAT chunk
        0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00, // Data
        0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, // Checksum
        0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, // IEND chunk
        0x42, 0x60, 0x82,
    ]
}

/// Load and resize custom thumbnail image (requires "aasx-thumbnails" feature)
#[cfg(feature = "aasx-thumbnails")]
fn load_and_resize_thumbnail(path: &Path, size: (u32, u32)) -> Result<Vec<u8>, SammError> {
    // Load the image from file
    let img = image::open(path).map_err(|e| {
        SammError::Generation(format!("Failed to load thumbnail from {:?}: {}", path, e))
    })?;

    // Resize the image while preserving aspect ratio
    let resized = img.resize(size.0, size.1, imageops::FilterType::Lanczos3);

    // Encode to PNG
    let mut buffer = Vec::new();
    resized
        .write_to(&mut std::io::Cursor::new(&mut buffer), ImageFormat::Png)
        .map_err(|e| SammError::Generation(format!("Failed to encode thumbnail as PNG: {}", e)))?;

    Ok(buffer)
}

/// Load and resize custom thumbnail - fallback when feature is disabled
#[cfg(not(feature = "aasx-thumbnails"))]
fn load_and_resize_thumbnail(_path: &Path, _size: (u32, u32)) -> Result<Vec<u8>, SammError> {
    Err(SammError::Generation(
        "Custom thumbnails require the 'aasx-thumbnails' feature. \
         Rebuild with --features aasx-thumbnails"
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::aas::environment::*;

    #[test]
    fn test_serialize_json() {
        let env = Environment {
            asset_administration_shells: None,
            submodels: None,
            concept_descriptions: None,
        };

        let json = serialize_json(&env).unwrap();
        eprintln!("JSON output: {}", json); // Debug output
                                            // Check that JSON was generated successfully
        assert!(!json.is_empty());
        // JSON should be valid (starts with '{' and ends with '}')
        assert!(json.trim().starts_with('{'));
        assert!(json.trim().ends_with('}'));
    }
}
