//! SPARQL query execution and processing for pattern mining

use std::collections::HashMap;
use tracing::{debug, warn};

use oxirs_core::rdf_store::QueryResults as CoreQueryResults;
use oxirs_core::Store;

use super::engine::FrequencyTables;
use super::types::QueryResults;
use crate::Result;

/// Execute SPARQL query against store
pub fn execute_sparql_query(
    store: &dyn Store,
    query: &str,
    _graph_name: Option<&str>,
) -> Result<QueryResults> {
    debug!("Executing SPARQL query: {}", query.trim());

    // Try to execute the query if store supports it
    match store.query(query) {
        Ok(oxirs_results) => {
            debug!("SPARQL query executed successfully");

            let mut bindings = Vec::new();

            // Convert OxirsQueryResults to our QueryResults format
            let variables = oxirs_results.variables();
            let results = oxirs_results.results();

            match results {
                CoreQueryResults::Bindings(variable_bindings) => {
                    for binding in variable_bindings {
                        let mut binding_map = HashMap::new();

                        for var in variables {
                            if let Some(value) = binding.get(var) {
                                binding_map.insert(var.clone(), value.to_string());
                            }
                        }

                        bindings.push(binding_map);
                    }
                }
                CoreQueryResults::Boolean(result) => {
                    // For ASK queries, convert boolean to binding
                    let mut binding_map = HashMap::new();
                    binding_map.insert("result".to_string(), result.to_string());
                    bindings.push(binding_map);
                }
                CoreQueryResults::Graph(quads) => {
                    // For CONSTRUCT/DESCRIBE queries, convert quads to bindings
                    for quad in quads {
                        let mut binding_map = HashMap::new();
                        binding_map.insert("subject".to_string(), quad.subject().to_string());
                        binding_map.insert("predicate".to_string(), quad.predicate().to_string());
                        binding_map.insert("object".to_string(), quad.object().to_string());
                        // Add graph name if it exists
                        binding_map.insert("graph".to_string(), quad.graph_name().to_string());
                        bindings.push(binding_map);
                    }
                }
            }

            debug!("Converted {} query results to bindings", bindings.len());
            Ok(QueryResults { bindings })
        }
        Err(e) => {
            warn!("SPARQL query execution failed: {}, using fallback", e);

            // Fallback to empty results - the calling code will handle fallback analysis
            Ok(QueryResults {
                bindings: Vec::new(),
            })
        }
    }
}

/// Process property frequency query results
pub fn process_property_frequency_results(
    frequency_tables: &mut FrequencyTables,
    results: QueryResults,
) -> Result<()> {
    debug!(
        "Processing property frequency results: {} bindings",
        results.bindings.len()
    );

    for binding in results.bindings {
        if let (Some(property), Some(count_str)) = (binding.get("property"), binding.get("count")) {
            if let Ok(count) = count_str.parse::<usize>() {
                frequency_tables.properties.insert(property.clone(), count);
            }
        }
    }

    Ok(())
}

/// Process class frequency query results
pub fn process_class_frequency_results(
    frequency_tables: &mut FrequencyTables,
    results: QueryResults,
) -> Result<()> {
    debug!(
        "Processing class frequency results: {} bindings",
        results.bindings.len()
    );

    for binding in results.bindings {
        if let (Some(class), Some(count_str)) = (binding.get("class"), binding.get("count")) {
            if let Ok(count) = count_str.parse::<usize>() {
                frequency_tables.classes.insert(class.clone(), count);
            }
        }
    }

    Ok(())
}

/// Process value pattern query results
pub fn process_value_pattern_results(
    frequency_tables: &mut FrequencyTables,
    results: QueryResults,
) -> Result<()> {
    debug!(
        "Processing value pattern results: {} bindings",
        results.bindings.len()
    );

    for binding in results.bindings {
        if let (Some(pattern), Some(count_str)) = (binding.get("pattern"), binding.get("count")) {
            if let Ok(count) = count_str.parse::<usize>() {
                frequency_tables
                    .value_patterns
                    .insert(pattern.clone(), count);
            }
        }
    }

    Ok(())
}

/// Process co-occurrence query results
pub fn process_co_occurrence_results(
    frequency_tables: &mut FrequencyTables,
    results: QueryResults,
) -> Result<()> {
    debug!(
        "Processing co-occurrence results: {} bindings",
        results.bindings.len()
    );

    for binding in results.bindings {
        if let (Some(prop1), Some(prop2), Some(count_str)) = (
            binding.get("prop1"),
            binding.get("prop2"),
            binding.get("count"),
        ) {
            if let Ok(count) = count_str.parse::<usize>() {
                frequency_tables
                    .co_occurrence
                    .insert((prop1.clone(), prop2.clone()), count);
            }
        }
    }

    Ok(())
}

/// Fallback property analysis using direct store access
pub fn fallback_property_analysis(
    frequency_tables: &mut FrequencyTables,
    store: &dyn Store,
    _graph_name: Option<&str>,
) -> Result<()> {
    debug!("Performing fallback property analysis");

    let mut property_counts = HashMap::new();

    // Manual property counting using store interface
    match store.quads() {
        Ok(quads) => {
            for quad in quads {
                let predicate_str = quad.predicate().to_string();
                *property_counts.entry(predicate_str).or_insert(0) += 1;
            }
        }
        Err(e) => {
            warn!("Failed to get quads from store: {}", e);
            return Err(crate::ShaclAiError::PatternRecognition(format!(
                "Store access failed: {e}"
            )));
        }
    }

    frequency_tables.properties = property_counts;
    debug!(
        "Fallback property analysis found {} properties",
        frequency_tables.properties.len()
    );

    Ok(())
}

/// Fallback class analysis using direct store access
pub fn fallback_class_analysis(
    frequency_tables: &mut FrequencyTables,
    store: &dyn Store,
    _graph_name: Option<&str>,
) -> Result<()> {
    debug!("Performing fallback class analysis");

    let mut class_counts = HashMap::new();
    let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

    // Manual class counting using store interface
    match store.quads() {
        Ok(quads) => {
            for quad in quads {
                if quad.predicate().to_string() == rdf_type {
                    let class_str = quad.object().to_string();
                    *class_counts.entry(class_str).or_insert(0) += 1;
                }
            }
        }
        Err(e) => {
            warn!("Failed to get quads from store: {}", e);
            return Err(crate::ShaclAiError::PatternRecognition(format!(
                "Store access failed: {e}"
            )));
        }
    }

    frequency_tables.classes = class_counts;
    debug!(
        "Fallback class analysis found {} classes",
        frequency_tables.classes.len()
    );

    Ok(())
}

/// Fallback value pattern analysis using direct store access
pub fn fallback_value_pattern_analysis(
    frequency_tables: &mut FrequencyTables,
    store: &dyn Store,
    _graph_name: Option<&str>,
) -> Result<()> {
    debug!("Performing fallback value pattern analysis");

    let mut pattern_counts = HashMap::new();

    // Manual value pattern analysis using store interface
    match store.quads() {
        Ok(quads) => {
            for quad in quads {
                let object_str = quad.object().to_string();

                // Extract patterns from object values
                if let Some(pattern) = extract_value_pattern(&object_str) {
                    *pattern_counts.entry(pattern).or_insert(0) += 1;
                }
            }
        }
        Err(e) => {
            warn!("Failed to get quads from store: {}", e);
            return Err(crate::ShaclAiError::PatternRecognition(format!(
                "Store access failed: {e}"
            )));
        }
    }

    // Filter patterns with sufficient support
    frequency_tables.value_patterns = pattern_counts
        .into_iter()
        .filter(|(_, count)| *count > 10)
        .collect();

    debug!(
        "Fallback value pattern analysis found {} patterns",
        frequency_tables.value_patterns.len()
    );

    Ok(())
}

/// Fallback co-occurrence analysis using direct store access
pub fn fallback_co_occurrence_analysis(
    frequency_tables: &mut FrequencyTables,
    store: &dyn Store,
    _graph_name: Option<&str>,
) -> Result<()> {
    debug!("Performing fallback co-occurrence analysis");

    let mut subject_properties: HashMap<String, Vec<String>> = HashMap::new();

    // Collect properties per subject
    match store.quads() {
        Ok(quads) => {
            for quad in quads {
                let subject_str = quad.subject().to_string();
                let predicate_str = quad.predicate().to_string();

                subject_properties
                    .entry(subject_str)
                    .or_default()
                    .push(predicate_str);
            }
        }
        Err(e) => {
            warn!("Failed to get quads from store: {}", e);
            return Err(crate::ShaclAiError::PatternRecognition(format!(
                "Store access failed: {e}"
            )));
        }
    }

    // Calculate co-occurrences
    let mut co_occurrence_counts = HashMap::new();
    for properties in subject_properties.values() {
        for i in 0..properties.len() {
            for j in (i + 1)..properties.len() {
                let prop1 = &properties[i];
                let prop2 = &properties[j];

                // Ensure consistent ordering
                let key = if prop1 < prop2 {
                    (prop1.clone(), prop2.clone())
                } else {
                    (prop2.clone(), prop1.clone())
                };

                *co_occurrence_counts.entry(key).or_insert(0) += 1;
            }
        }
    }

    // Filter co-occurrences with sufficient support
    frequency_tables.co_occurrence = co_occurrence_counts
        .into_iter()
        .filter(|(_, count)| *count > 5)
        .collect();

    debug!(
        "Fallback co-occurrence analysis found {} co-occurrences",
        frequency_tables.co_occurrence.len()
    );

    Ok(())
}

/// Extract value pattern from object string
fn extract_value_pattern(object_str: &str) -> Option<String> {
    // Simple pattern extraction - in practice this would be more sophisticated
    if object_str.contains("http://") || object_str.contains("https://") {
        return Some("URI".to_string());
    }

    if object_str.chars().all(|c| c.is_ascii_digit()) {
        return Some("INTEGER".to_string());
    }

    if object_str.parse::<f64>().is_ok() {
        return Some("DECIMAL".to_string());
    }

    if object_str.len() > 100 {
        return Some("LONG_TEXT".to_string());
    }

    if object_str.len() < 10 {
        return Some("SHORT_TEXT".to_string());
    }

    Some("TEXT".to_string())
}
