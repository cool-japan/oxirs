//! SWRL Built-in Utility Functions
//!
//! Helper functions for extracting and converting SWRL argument values.

use super::super::types::SwrlArgument;
use anyhow::Result;

/// Extract numeric value from SWRL argument
pub(crate) fn extract_numeric_value(arg: &SwrlArgument) -> Result<f64> {
    match arg {
        SwrlArgument::Literal(value) => value
            .parse::<f64>()
            .map_err(|_| anyhow::anyhow!("Cannot parse '{}' as numeric value", value)),
        _ => Err(anyhow::anyhow!(
            "Expected literal numeric value, got {:?}",
            arg
        )),
    }
}

/// Extract string value from SWRL argument
pub(crate) fn extract_string_value(arg: &SwrlArgument) -> Result<String> {
    match arg {
        SwrlArgument::Literal(value) => Ok(value.clone()),
        SwrlArgument::Individual(value) => Ok(value.clone()),
        SwrlArgument::Variable(name) => Err(anyhow::anyhow!("Unbound variable: {}", name)),
    }
}

/// Extract the items of a list-valued SWRL argument.
///
/// The supported representation of a list in this (store-less) evaluation
/// context is a comma-encoded **literal** (e.g. `"a,b,c"`). If the argument is
/// instead an `Individual` — i.e. an IRI/blank node that heads a genuine
/// `rdf:List` (`rdf:first`/`rdf:rest`) in the triple store — we cannot resolve
/// it here, so we fail loud rather than silently splitting the IRI string on
/// commas and returning a wrong result. An unbound variable is likewise an
/// error.
pub(crate) fn extract_list_items(arg: &SwrlArgument) -> Result<Vec<String>> {
    match arg {
        SwrlArgument::Literal(value) => Ok(value.split(',').map(|s| s.to_string()).collect()),
        SwrlArgument::Individual(iri) => Err(anyhow::anyhow!(
            "SWRL list built-in received RDF list resource '{iri}': genuine rdf:List \
             (rdf:first/rdf:rest) traversal is not supported in this literal-list context; \
             supply a comma-encoded literal instead"
        )),
        SwrlArgument::Variable(name) => Err(anyhow::anyhow!("Unbound variable: {}", name)),
    }
}

/// Extract boolean value from SWRL argument
pub(crate) fn extract_boolean_value(arg: &SwrlArgument) -> Result<bool> {
    match arg {
        SwrlArgument::Literal(value) => match value.to_lowercase().as_str() {
            "true" | "1" => Ok(true),
            "false" | "0" => Ok(false),
            _ => Err(anyhow::anyhow!("Cannot parse '{}' as boolean value", value)),
        },
        _ => Err(anyhow::anyhow!(
            "Expected literal boolean value, got {:?}",
            arg
        )),
    }
}
