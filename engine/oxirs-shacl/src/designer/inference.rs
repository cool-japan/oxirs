//! Shape Inference from Data
//!
//! Provides capabilities to infer SHACL shapes from sample RDF data,
//! analyzing patterns, datatypes, and cardinalities.

use super::{ConstraintSpec, PropertyDesign, PropertyHint, ShapeDesign};
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Shape inference engine
#[derive(Debug)]
pub struct ShapeInferenceEngine {
    /// Inference configuration
    config: InferenceConfig,
    /// Statistics collector
    stats: InferenceStatistics,
}

/// Configuration for shape inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Minimum occurrences to consider a property
    pub min_property_occurrences: usize,
    /// Threshold for required property (0.0 - 1.0)
    pub required_threshold: f64,
    /// Threshold for unique property (0.0 - 1.0)
    pub unique_threshold: f64,
    /// Maximum distinct values to collect
    pub max_distinct_values: usize,
    /// Infer datatypes
    pub infer_datatypes: bool,
    /// Infer patterns
    pub infer_patterns: bool,
    /// Infer value ranges
    pub infer_ranges: bool,
    /// Sample size for large datasets (0 = all)
    pub sample_size: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            min_property_occurrences: 2,
            required_threshold: 0.95,
            unique_threshold: 0.99,
            max_distinct_values: 100,
            infer_datatypes: true,
            infer_patterns: true,
            infer_ranges: true,
            sample_size: 0,
        }
    }
}

/// Statistics collected during inference
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct InferenceStatistics {
    /// Total subjects analyzed
    pub total_subjects: usize,
    /// Total triples analyzed
    pub total_triples: usize,
    /// Properties discovered
    pub properties_discovered: usize,
    /// Types discovered
    pub types_discovered: usize,
    /// Inference duration in milliseconds
    pub duration_ms: u64,
}

/// Inferred property information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredProperty {
    /// Property IRI
    pub property: String,
    /// Occurrence count
    pub occurrences: usize,
    /// Unique subjects count
    pub unique_subjects: usize,
    /// Average values per subject
    pub avg_values_per_subject: f64,
    /// Max values per subject
    pub max_values_per_subject: usize,
    /// Min values per subject
    pub min_values_per_subject: usize,
    /// Distinct values (if within limit)
    pub distinct_values: Option<Vec<String>>,
    /// Inferred datatype
    pub datatype: Option<String>,
    /// Value statistics
    pub value_stats: ValueStatistics,
    /// Suggested constraints
    pub suggested_constraints: Vec<ConstraintSpec>,
    /// Suggested hints
    pub suggested_hints: Vec<PropertyHint>,
}

/// Property statistics for suggestion generation
#[derive(Debug, Clone)]
struct PropertyStats<'a> {
    unique_subjects: usize,
    total_subjects: usize,
    avg_values: f64,
    max_values: usize,
    datatype: &'a Option<String>,
    value_stats: &'a ValueStatistics,
}

/// Value statistics for a property
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValueStatistics {
    /// Min string length
    pub min_length: Option<usize>,
    /// Max string length
    pub max_length: Option<usize>,
    /// Average string length
    pub avg_length: Option<f64>,
    /// Min numeric value
    pub min_value: Option<f64>,
    /// Max numeric value
    pub max_value: Option<f64>,
    /// Average numeric value
    pub avg_value: Option<f64>,
    /// Common pattern (if detected)
    pub common_pattern: Option<String>,
    /// Null/empty percentage
    pub null_percentage: f64,
}

/// Inferred class/type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredType {
    /// Type IRI
    pub type_iri: String,
    /// Instance count
    pub instance_count: usize,
    /// Properties used by this type
    pub properties: Vec<String>,
}

/// Result of shape inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    /// Inferred types
    pub types: Vec<InferredType>,
    /// Inferred properties
    pub properties: HashMap<String, InferredProperty>,
    /// Generated shapes
    pub shapes: Vec<ShapeDesign>,
    /// Statistics
    pub statistics: InferenceStatistics,
    /// Warnings/notes
    pub warnings: Vec<String>,
}

/// Sample data for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleData {
    /// Subject IRI
    pub subject: String,
    /// RDF type(s)
    pub types: Vec<String>,
    /// Property-value pairs
    pub properties: HashMap<String, Vec<SampleValue>>,
}

/// Sample value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleValue {
    /// Value as string
    pub value: String,
    /// Datatype if literal
    pub datatype: Option<String>,
    /// Language tag if literal
    pub language: Option<String>,
    /// Whether this is an IRI
    pub is_iri: bool,
}

impl ShapeInferenceEngine {
    /// Create a new inference engine
    pub fn new() -> Self {
        Self {
            config: InferenceConfig::default(),
            stats: InferenceStatistics::default(),
        }
    }

    /// Create with configuration
    pub fn with_config(config: InferenceConfig) -> Self {
        Self {
            config,
            stats: InferenceStatistics::default(),
        }
    }

    /// Infer shapes from sample data
    pub fn infer_from_samples(&mut self, samples: Vec<SampleData>) -> Result<InferenceResult> {
        let start = std::time::Instant::now();

        let mut result = InferenceResult {
            types: Vec::new(),
            properties: HashMap::new(),
            shapes: Vec::new(),
            statistics: InferenceStatistics::default(),
            warnings: Vec::new(),
        };

        // Apply sampling if configured
        let samples = if self.config.sample_size > 0 && samples.len() > self.config.sample_size {
            result.warnings.push(format!(
                "Using sample of {} from {} total subjects",
                self.config.sample_size,
                samples.len()
            ));
            samples.into_iter().take(self.config.sample_size).collect()
        } else {
            samples
        };

        result.statistics.total_subjects = samples.len();

        // Collect type information
        let mut type_instances: HashMap<String, Vec<String>> = HashMap::new();
        let mut type_properties: HashMap<String, HashSet<String>> = HashMap::new();

        // Collect property statistics
        let mut property_values: HashMap<String, Vec<(String, SampleValue)>> = HashMap::new();
        let mut property_subjects: HashMap<String, HashSet<String>> = HashMap::new();

        for sample in &samples {
            // Track types
            for type_iri in &sample.types {
                type_instances
                    .entry(type_iri.clone())
                    .or_default()
                    .push(sample.subject.clone());
            }

            // Track properties
            for (prop, values) in &sample.properties {
                result.statistics.total_triples += values.len();

                property_subjects
                    .entry(prop.clone())
                    .or_default()
                    .insert(sample.subject.clone());

                for value in values {
                    property_values
                        .entry(prop.clone())
                        .or_default()
                        .push((sample.subject.clone(), value.clone()));
                }

                // Track which properties appear with which types
                for type_iri in &sample.types {
                    type_properties
                        .entry(type_iri.clone())
                        .or_default()
                        .insert(prop.clone());
                }
            }
        }

        // Build inferred types
        for (type_iri, instances) in &type_instances {
            let properties = type_properties
                .get(type_iri)
                .map(|s| s.iter().cloned().collect())
                .unwrap_or_default();

            result.types.push(InferredType {
                type_iri: type_iri.clone(),
                instance_count: instances.len(),
                properties,
            });
        }
        result.statistics.types_discovered = result.types.len();

        // Build inferred properties
        for (prop, values) in &property_values {
            if values.len() < self.config.min_property_occurrences {
                continue;
            }

            let subjects = property_subjects.get(prop).cloned().unwrap_or_default();
            let unique_subjects = subjects.len();

            // Calculate values per subject
            let mut values_per_subject: HashMap<String, usize> = HashMap::new();
            for (subj, _) in values {
                *values_per_subject.entry(subj.clone()).or_default() += 1;
            }

            let total_values: usize = values_per_subject.values().sum();
            let avg_values = total_values as f64 / unique_subjects as f64;
            let max_values = *values_per_subject.values().max().unwrap_or(&0);
            let min_values = *values_per_subject.values().min().unwrap_or(&0);

            // Collect distinct values
            let distinct: HashSet<_> = values.iter().map(|(_, v)| v.value.clone()).collect();
            let distinct_values = if distinct.len() <= self.config.max_distinct_values {
                Some(distinct.into_iter().collect())
            } else {
                None
            };

            // Infer datatype
            let datatype = self.infer_datatype(values);

            // Calculate value statistics
            let value_stats = self.calculate_value_stats(values);

            // Generate suggestions
            let prop_stats = PropertyStats {
                unique_subjects,
                total_subjects: result.statistics.total_subjects,
                avg_values,
                max_values,
                datatype: &datatype,
                value_stats: &value_stats,
            };
            let (suggested_constraints, suggested_hints) =
                self.generate_suggestions(prop, &prop_stats);

            result.properties.insert(
                prop.clone(),
                InferredProperty {
                    property: prop.clone(),
                    occurrences: values.len(),
                    unique_subjects,
                    avg_values_per_subject: avg_values,
                    max_values_per_subject: max_values,
                    min_values_per_subject: min_values,
                    distinct_values,
                    datatype,
                    value_stats,
                    suggested_constraints,
                    suggested_hints,
                },
            );
        }
        result.statistics.properties_discovered = result.properties.len();

        // Generate shapes for each type
        for inferred_type in &result.types {
            let shape = self.create_shape_from_type(inferred_type, &result.properties);
            result.shapes.push(shape);
        }

        result.statistics.duration_ms = start.elapsed().as_millis() as u64;
        self.stats = result.statistics.clone();

        Ok(result)
    }

    fn infer_datatype(&self, values: &[(String, SampleValue)]) -> Option<String> {
        if !self.config.infer_datatypes {
            return None;
        }

        let mut datatype_counts: HashMap<String, usize> = HashMap::new();

        for (_, value) in values {
            if let Some(dt) = &value.datatype {
                *datatype_counts.entry(dt.clone()).or_default() += 1;
            } else if value.is_iri {
                *datatype_counts.entry("IRI".to_string()).or_default() += 1;
            } else {
                // Try to infer from value
                let inferred = self.infer_datatype_from_value(&value.value);
                *datatype_counts.entry(inferred).or_default() += 1;
            }
        }

        // Return most common datatype
        datatype_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(dt, _)| dt)
    }

    fn infer_datatype_from_value(&self, value: &str) -> String {
        // Check for integer
        if value.parse::<i64>().is_ok() {
            return "http://www.w3.org/2001/XMLSchema#integer".to_string();
        }

        // Check for decimal/float
        if value.parse::<f64>().is_ok() {
            return "http://www.w3.org/2001/XMLSchema#decimal".to_string();
        }

        // Check for boolean
        if value == "true" || value == "false" {
            return "http://www.w3.org/2001/XMLSchema#boolean".to_string();
        }

        // Check for date patterns
        if value.len() == 10
            && value.chars().nth(4) == Some('-')
            && value.chars().nth(7) == Some('-')
        {
            return "http://www.w3.org/2001/XMLSchema#date".to_string();
        }

        // Check for dateTime patterns
        if value.contains('T') && value.len() >= 19 {
            return "http://www.w3.org/2001/XMLSchema#dateTime".to_string();
        }

        // Default to string
        "http://www.w3.org/2001/XMLSchema#string".to_string()
    }

    fn calculate_value_stats(&self, values: &[(String, SampleValue)]) -> ValueStatistics {
        let mut stats = ValueStatistics::default();

        if values.is_empty() {
            return stats;
        }

        // String length statistics
        let lengths: Vec<usize> = values.iter().map(|(_, v)| v.value.len()).collect();
        stats.min_length = lengths.iter().min().copied();
        stats.max_length = lengths.iter().max().copied();
        stats.avg_length = Some(lengths.iter().sum::<usize>() as f64 / lengths.len() as f64);

        // Numeric statistics (if applicable)
        let numbers: Vec<f64> = values
            .iter()
            .filter_map(|(_, v)| v.value.parse::<f64>().ok())
            .collect();

        if !numbers.is_empty() && numbers.len() as f64 / values.len() as f64 > 0.5 {
            stats.min_value = numbers.iter().copied().fold(None, |min, x| match min {
                None => Some(x),
                Some(m) => Some(if x < m { x } else { m }),
            });
            stats.max_value = numbers.iter().copied().fold(None, |max, x| match max {
                None => Some(x),
                Some(m) => Some(if x > m { x } else { m }),
            });
            stats.avg_value = Some(numbers.iter().sum::<f64>() / numbers.len() as f64);
        }

        // Null/empty percentage
        let empty_count = values.iter().filter(|(_, v)| v.value.is_empty()).count();
        stats.null_percentage = empty_count as f64 / values.len() as f64;

        // Pattern detection (simple)
        if self.config.infer_patterns {
            stats.common_pattern = self.detect_common_pattern(values);
        }

        stats
    }

    fn detect_common_pattern(&self, values: &[(String, SampleValue)]) -> Option<String> {
        // Check for email pattern
        let email_regex =
            regex::Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").ok()?;
        let email_match_count = values
            .iter()
            .filter(|(_, v)| email_regex.is_match(&v.value))
            .count();
        if email_match_count as f64 / values.len() as f64 > 0.9 {
            return Some("email".to_string());
        }

        // Check for URL pattern
        let url_regex = regex::Regex::new(r"^https?://[^\s]+$").ok()?;
        let url_match_count = values
            .iter()
            .filter(|(_, v)| url_regex.is_match(&v.value))
            .count();
        if url_match_count as f64 / values.len() as f64 > 0.9 {
            return Some("url".to_string());
        }

        // Check for UUID pattern
        let uuid_regex = regex::Regex::new(
            r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        )
        .ok()?;
        let uuid_match_count = values
            .iter()
            .filter(|(_, v)| uuid_regex.is_match(&v.value))
            .count();
        if uuid_match_count as f64 / values.len() as f64 > 0.9 {
            return Some("uuid".to_string());
        }

        None
    }

    fn generate_suggestions(
        &self,
        _property: &str,
        stats: &PropertyStats<'_>,
    ) -> (Vec<ConstraintSpec>, Vec<PropertyHint>) {
        let mut constraints = Vec::new();
        let mut hints = Vec::new();

        // Check if property is required
        let coverage = stats.unique_subjects as f64 / stats.total_subjects as f64;
        if coverage >= self.config.required_threshold {
            constraints.push(ConstraintSpec::MinCount(1));
            hints.push(PropertyHint::Required);
        }

        // Check if property is unique/single-valued
        if stats.avg_values <= 1.0 && stats.max_values == 1 {
            constraints.push(ConstraintSpec::MaxCount(1));
            if coverage >= self.config.unique_threshold {
                hints.push(PropertyHint::Unique);
            }
        } else if stats.max_values > 1 {
            hints.push(PropertyHint::MultiValued);
        }

        // Add datatype constraint
        if let Some(dt) = stats.datatype {
            if dt != "IRI" {
                constraints.push(ConstraintSpec::Datatype(dt.clone()));

                // Add type hint
                if dt.contains("string") {
                    hints.push(PropertyHint::String);
                } else if dt.contains("integer") {
                    hints.push(PropertyHint::Integer);
                } else if dt.contains("decimal") || dt.contains("float") {
                    hints.push(PropertyHint::Decimal);
                } else if dt.contains("date") && !dt.contains("Time") {
                    hints.push(PropertyHint::Date);
                } else if dt.contains("dateTime") {
                    hints.push(PropertyHint::DateTime);
                } else if dt.contains("boolean") {
                    hints.push(PropertyHint::Boolean);
                }
            } else {
                hints.push(PropertyHint::IRI);
            }
        }

        // Add range constraints for numeric values
        if self.config.infer_ranges {
            if let (Some(min), Some(_max)) =
                (stats.value_stats.min_value, stats.value_stats.max_value)
            {
                if min >= 0.0 {
                    constraints.push(ConstraintSpec::MinInclusive(0.0));
                }
            }
        }

        // Add pattern hints
        if let Some(pattern) = &stats.value_stats.common_pattern {
            match pattern.as_str() {
                "email" => hints.push(PropertyHint::Email),
                "url" => hints.push(PropertyHint::URL),
                _ => {}
            }
        }

        (constraints, hints)
    }

    fn create_shape_from_type(
        &self,
        inferred_type: &InferredType,
        properties: &HashMap<String, InferredProperty>,
    ) -> ShapeDesign {
        // Extract local name from type IRI
        let local_name = inferred_type
            .type_iri
            .rsplit(&['#', '/'][..])
            .next()
            .unwrap_or(&inferred_type.type_iri);

        let shape_id = format!("{}Shape", local_name);
        let mut design = ShapeDesign::new(&shape_id)
            .with_label(format!("{} Shape", local_name))
            .with_description(format!(
                "Inferred shape for {} instances of {}",
                inferred_type.instance_count, inferred_type.type_iri
            ));

        design.add_target_class(inferred_type.type_iri.clone());

        // Add properties
        for prop_iri in &inferred_type.properties {
            if let Some(prop_info) = properties.get(prop_iri) {
                let mut prop_design = PropertyDesign::new(prop_iri.clone());

                for hint in &prop_info.suggested_hints {
                    prop_design.hints.insert(*hint);
                }

                for constraint in &prop_info.suggested_constraints {
                    prop_design.constraints.push(constraint.clone());
                }

                design.add_property(prop_design);
            }
        }

        design
    }

    /// Get statistics from last inference
    pub fn statistics(&self) -> &InferenceStatistics {
        &self.stats
    }
}

impl Default for ShapeInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_sample_data() -> Vec<SampleData> {
        vec![
            SampleData {
                subject: "ex:person1".to_string(),
                types: vec!["foaf:Person".to_string()],
                properties: {
                    let mut props = HashMap::new();
                    props.insert(
                        "foaf:name".to_string(),
                        vec![SampleValue {
                            value: "John Doe".to_string(),
                            datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
                            language: None,
                            is_iri: false,
                        }],
                    );
                    props.insert(
                        "foaf:age".to_string(),
                        vec![SampleValue {
                            value: "30".to_string(),
                            datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
                            language: None,
                            is_iri: false,
                        }],
                    );
                    props.insert(
                        "foaf:mbox".to_string(),
                        vec![SampleValue {
                            value: "john@example.com".to_string(),
                            datatype: None,
                            language: None,
                            is_iri: false,
                        }],
                    );
                    props
                },
            },
            SampleData {
                subject: "ex:person2".to_string(),
                types: vec!["foaf:Person".to_string()],
                properties: {
                    let mut props = HashMap::new();
                    props.insert(
                        "foaf:name".to_string(),
                        vec![SampleValue {
                            value: "Jane Smith".to_string(),
                            datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
                            language: None,
                            is_iri: false,
                        }],
                    );
                    props.insert(
                        "foaf:age".to_string(),
                        vec![SampleValue {
                            value: "25".to_string(),
                            datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
                            language: None,
                            is_iri: false,
                        }],
                    );
                    props.insert(
                        "foaf:mbox".to_string(),
                        vec![SampleValue {
                            value: "jane@example.org".to_string(),
                            datatype: None,
                            language: None,
                            is_iri: false,
                        }],
                    );
                    props
                },
            },
        ]
    }

    #[test]
    fn test_inference_engine() {
        let mut engine = ShapeInferenceEngine::new();
        let samples = create_sample_data();

        let result = engine.infer_from_samples(samples).unwrap();

        assert_eq!(result.statistics.total_subjects, 2);
        assert!(!result.types.is_empty());
        assert!(!result.properties.is_empty());
    }

    #[test]
    fn test_type_inference() {
        let mut engine = ShapeInferenceEngine::new();
        let samples = create_sample_data();

        let result = engine.infer_from_samples(samples).unwrap();

        let person_type = result.types.iter().find(|t| t.type_iri == "foaf:Person");
        assert!(person_type.is_some());
        assert_eq!(person_type.unwrap().instance_count, 2);
    }

    #[test]
    fn test_property_inference() {
        let mut engine = ShapeInferenceEngine::new();
        let samples = create_sample_data();

        let result = engine.infer_from_samples(samples).unwrap();

        let name_prop = result.properties.get("foaf:name");
        assert!(name_prop.is_some());

        let name_prop = name_prop.unwrap();
        assert_eq!(name_prop.occurrences, 2);
        assert!(name_prop.suggested_hints.contains(&PropertyHint::Required));
    }

    #[test]
    fn test_shape_generation() {
        let mut engine = ShapeInferenceEngine::new();
        let samples = create_sample_data();

        let result = engine.infer_from_samples(samples).unwrap();

        assert!(!result.shapes.is_empty());

        let person_shape = result.shapes.iter().find(|s| s.id.contains("Person"));
        assert!(person_shape.is_some());
    }

    #[test]
    fn test_datatype_inference_from_value() {
        let engine = ShapeInferenceEngine::new();

        assert!(engine.infer_datatype_from_value("42").contains("integer"));
        assert!(engine.infer_datatype_from_value("3.14").contains("decimal"));
        assert!(engine.infer_datatype_from_value("true").contains("boolean"));
        assert!(engine
            .infer_datatype_from_value("2025-01-15")
            .contains("date"));
        assert!(engine.infer_datatype_from_value("hello").contains("string"));
    }

    #[test]
    fn test_pattern_detection() {
        let engine = ShapeInferenceEngine::new();

        let email_values = vec![
            (
                "s1".to_string(),
                SampleValue {
                    value: "test@example.com".to_string(),
                    datatype: None,
                    language: None,
                    is_iri: false,
                },
            ),
            (
                "s2".to_string(),
                SampleValue {
                    value: "user@domain.org".to_string(),
                    datatype: None,
                    language: None,
                    is_iri: false,
                },
            ),
        ];

        let pattern = engine.detect_common_pattern(&email_values);
        assert_eq!(pattern, Some("email".to_string()));
    }
}
