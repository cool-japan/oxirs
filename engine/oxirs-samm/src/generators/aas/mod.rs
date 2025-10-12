//! # AAS Generator Module
//!
//! Generates Asset Administration Shell (AAS) V3.0 compatible outputs from SAMM models.
//!
//! Supports:
//! - **XML**: AAS XML format (IDTA-01001-3-0)
//! - **JSON**: AAS JSON format
//! - **AASX**: AAS package (ZIP with XML/thumbnails)
//!
//! ## References
//! - [AAS Spec](https://industrialdigitaltwin.org/content-hub/aasspecifications)
//! - [admin-shell-io/aas-specs](https://github.com/admin-shell-io/aas-specs)

mod concept_description;
mod environment;
mod serializer;
mod submodel;
mod type_mapper;

pub use concept_description::*;
pub use environment::*;
pub use serializer::*;
pub use submodel::*;
pub use type_mapper::*;

use crate::error::SammError;
use crate::metamodel::Aspect;

/// AAS output format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AasFormat {
    /// AAS XML format
    Xml,
    /// AAS JSON format
    Json,
    /// AASX package (ZIP)
    Aasx,
}

/// Generate AAS output from SAMM Aspect
pub fn generate_aas(aspect: &Aspect, format: AasFormat) -> Result<String, SammError> {
    let environment = build_aas_environment(aspect)?;

    match format {
        AasFormat::Xml => serialize_xml(&environment),
        AasFormat::Json => serialize_json(&environment),
        AasFormat::Aasx => {
            // AASX is a ZIP containing XML + thumbnails
            let aasx_binary = serialize_aasx(&environment)?;
            // Return base64-encoded ZIP for text output
            use base64::Engine;
            let encoded = base64::engine::general_purpose::STANDARD.encode(&aasx_binary);
            Ok(format!("data:application/zip;base64,{}", encoded))
        }
    }
}
