//! Schema type detection for RDF generation
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use oxirs_core::format::{RdfFormat, RdfParser};
use oxirs_core::model::Object;
use oxirs_core::RdfTerm;
use std::error::Error;
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};

/// Map a schema file's extension to its RDF serialization format.
///
/// Shared by schema-type detection and every schema parser so the set of
/// supported extensions stays consistent. An unknown or absent extension is an
/// explicit error rather than a silent default.
pub(super) fn detect_rdf_format(path: &Path) -> Result<RdfFormat, Box<dyn Error>> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or("File has no extension")?;
    match extension.to_lowercase().as_str() {
        "ttl" => Ok(RdfFormat::Turtle),
        "rdf" | "xml" | "owl" => Ok(RdfFormat::RdfXml),
        "nt" => Ok(RdfFormat::NTriples),
        "nq" => Ok(RdfFormat::NQuads),
        "trig" => Ok(RdfFormat::TriG),
        "jsonld" | "json" => Ok(RdfFormat::JsonLd {
            profile: oxirs_core::format::JsonLdProfileSet::empty(),
        }),
        "n3" => Ok(RdfFormat::N3),
        ext => Err(format!("Unsupported file extension: {}", ext).into()),
    }
}

/// Detect the type of schema file (SHACL, RDFS, or OWL)
pub(super) fn detect_schema_type(
    schema_file: &PathBuf,
    ctx: &crate::cli::CliContext,
) -> Result<String, Box<dyn Error>> {
    let format = detect_rdf_format(schema_file)?;
    let file = fs::File::open(schema_file)?;
    let reader = BufReader::new(file);
    let parser = RdfParser::new(format);
    let mut has_shacl = false;
    let mut has_rdfs = false;
    let mut has_owl = false;
    let shacl_ns = "http://www.w3.org/ns/shacl#";
    let rdfs_ns = "http://www.w3.org/2000/01/rdf-schema#";
    let owl_ns = "http://www.w3.org/2002/07/owl#";
    let mut quad_count = 0;
    for quad in parser.for_reader(reader).flatten() {
        quad_count += 1;
        let pred_uri = quad.predicate().as_str();
        if pred_uri.starts_with(shacl_ns) {
            has_shacl = true;
        }
        if pred_uri.starts_with(rdfs_ns) {
            has_rdfs = true;
        }
        if pred_uri.starts_with(owl_ns) {
            has_owl = true;
        }
        if let Object::NamedNode(nn) = quad.object() {
            let obj_uri = nn.as_str();
            if obj_uri.starts_with(shacl_ns) {
                has_shacl = true;
            }
            if obj_uri.starts_with(rdfs_ns) {
                has_rdfs = true;
            }
            if obj_uri.starts_with(owl_ns) {
                has_owl = true;
            }
        }
        if quad_count > 100 {
            break;
        }
    }
    ctx.info(&format!(
        "Schema detection: SHACL={}, RDFS={}, OWL={}",
        has_shacl, has_rdfs, has_owl
    ));
    if has_shacl {
        Ok("SHACL".to_string())
    } else if has_owl {
        Ok("OWL".to_string())
    } else if has_rdfs {
        Ok("RDFS".to_string())
    } else {
        Err(
            "Unable to detect schema type. File must contain SHACL, RDFS, or OWL definitions."
                .into(),
        )
    }
}
