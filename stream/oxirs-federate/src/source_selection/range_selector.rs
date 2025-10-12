//! Range-based selector implementation
//!
//! Selects services based on numeric, string, and temporal ranges

use crate::source_selection::types::*;
use crate::ServiceRegistry;
use anyhow::Result;
use chrono::{DateTime, Datelike, Utc};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

impl Default for RangeBasedSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl RangeBasedSelector {
    pub fn new() -> Self {
        Self {
            range_indices: Arc::new(RwLock::new(HashMap::new())),
            temporal_indices: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn select_by_ranges(
        &self,
        constraints: &[RangeConstraint],
        _registry: &ServiceRegistry,
    ) -> Result<HashMap<String, f64>> {
        let mut matches = HashMap::new();
        let range_indices = self.range_indices.read().await;
        let temporal_indices = self.temporal_indices.read().await;

        for (service_endpoint, range_index) in range_indices.iter() {
            let mut match_score = 0.0;
            let mut total_constraints = 0.0;

            for constraint in constraints {
                total_constraints += 1.0;

                match constraint.data_type {
                    RangeDataType::Integer | RangeDataType::Float => {
                        if let Some(numeric_range) =
                            range_index.numeric_ranges.get(&constraint.field)
                        {
                            if self.range_overlaps_numeric(constraint, numeric_range) {
                                match_score += 1.0;
                            }
                        }
                    }
                    RangeDataType::DateTime => {
                        if let Some(temporal_index) = temporal_indices.get(service_endpoint) {
                            if let Some(datetime_range) =
                                temporal_index.datetime_ranges.get(&constraint.field)
                            {
                                if self.range_overlaps_datetime(constraint, datetime_range) {
                                    match_score += 1.0;
                                }
                            }
                        }
                    }
                    RangeDataType::String => {
                        if let Some(string_range) = range_index.string_ranges.get(&constraint.field)
                        {
                            if self.range_overlaps_string(constraint, string_range) {
                                match_score += 1.0;
                            }
                        }
                    }
                    RangeDataType::Uri => {
                        if let Some(uri_pattern) = range_index.uri_patterns.get(&constraint.field) {
                            if self.matches_uri_pattern(constraint, uri_pattern) {
                                match_score += 1.0;
                            }
                        }
                    }
                }
            }

            if total_constraints > 0.0 {
                let final_score = match_score / total_constraints;
                if final_score > 0.0 {
                    matches.insert(service_endpoint.clone(), final_score);
                }
            }
        }

        Ok(matches)
    }

    pub async fn update_indices(
        &self,
        service_endpoint: &str,
        triples: &[(String, String, String)],
    ) -> Result<()> {
        // Analyze numeric, string, and URI ranges from the triples
        let mut numeric_ranges = HashMap::new();
        let mut string_ranges = HashMap::new();
        let mut uri_patterns = HashMap::new();
        let mut datetime_ranges = HashMap::new();
        let mut year_ranges = HashMap::new();

        // Group triples by predicate for analysis
        let mut predicate_values: HashMap<String, Vec<String>> = HashMap::new();
        for (_, p, o) in triples {
            predicate_values
                .entry(p.clone())
                .or_default()
                .push(o.clone());
        }

        // Analyze each predicate's values
        let mut temporal_patterns = HashMap::new();
        for (predicate, values) in predicate_values {
            self.analyze_numeric_range(&predicate, &values, &mut numeric_ranges);
            self.analyze_string_range(&predicate, &values, &mut string_ranges);
            self.analyze_uri_patterns(&predicate, &values, &mut uri_patterns);
            self.analyze_datetime_range(&predicate, &values, &mut datetime_ranges);
            self.analyze_year_range(&predicate, &values, &mut year_ranges);
            self.analyze_temporal_patterns(&predicate, &values, &mut temporal_patterns);
        }

        let range_index = ServiceRangeIndex {
            numeric_ranges,
            string_ranges,
            uri_patterns,
            last_updated: Utc::now(),
        };

        let temporal_index = ServiceTemporalIndex {
            datetime_ranges,
            year_ranges,
            temporal_patterns,
            last_updated: Utc::now(),
        };

        self.range_indices
            .write()
            .await
            .insert(service_endpoint.to_string(), range_index);

        self.temporal_indices
            .write()
            .await
            .insert(service_endpoint.to_string(), temporal_index);
        Ok(())
    }

    fn range_overlaps_numeric(&self, constraint: &RangeConstraint, range: &NumericRange) -> bool {
        // Simplified overlap detection
        if let (Some(min_str), Some(max_str)) = (&constraint.min_value, &constraint.max_value) {
            if let (Ok(min_val), Ok(max_val)) = (min_str.parse::<f64>(), max_str.parse::<f64>()) {
                return !(max_val < range.min_value || min_val > range.max_value);
            }
        }
        true // Default to potential match if parsing fails
    }

    fn range_overlaps_datetime(&self, constraint: &RangeConstraint, range: &DateTimeRange) -> bool {
        if let (Some(min_str), Some(max_str)) = (&constraint.min_value, &constraint.max_value) {
            if let (Ok(min_dt), Ok(max_dt)) = (
                DateTime::parse_from_rfc3339(min_str).map(|dt| dt.with_timezone(&Utc)),
                DateTime::parse_from_rfc3339(max_str).map(|dt| dt.with_timezone(&Utc)),
            ) {
                return !(max_dt < range.earliest || min_dt > range.latest);
            }
        }
        true // Default to potential match if parsing fails
    }

    fn range_overlaps_string(&self, constraint: &RangeConstraint, range: &StringRange) -> bool {
        if let (Some(min_str), Some(max_str)) = (&constraint.min_value, &constraint.max_value) {
            let min_len = min_str.len();
            let max_len = max_str.len();

            // Check if length ranges overlap
            if min_len > range.max_length || max_len < range.min_length {
                return false;
            }

            // Check prefix matching
            for prefix in &range.common_prefixes {
                if min_str.starts_with(prefix) || max_str.starts_with(prefix) {
                    return true;
                }
            }
        }
        true // Default to potential match
    }

    fn matches_uri_pattern(&self, constraint: &RangeConstraint, pattern: &UriPattern) -> bool {
        if let Some(value) = &constraint.min_value {
            // Check if the constraint value matches any of the URI patterns
            for base_uri in &pattern.base_uris {
                if value.starts_with(base_uri) {
                    return true;
                }
            }

            for namespace in &pattern.namespace_prefixes {
                if value.starts_with(namespace) {
                    return true;
                }
            }
        }
        true // Default to potential match
    }

    /// Analyze numeric ranges in predicate values
    fn analyze_numeric_range(
        &self,
        predicate: &str,
        values: &[String],
        numeric_ranges: &mut HashMap<String, NumericRange>,
    ) {
        let mut numeric_values = Vec::new();

        for value in values {
            if let Ok(num) = value.parse::<f64>() {
                numeric_values.push(num);
            }
        }

        if !numeric_values.is_empty() {
            let min_value = numeric_values
                .iter()
                .cloned()
                .fold(f64::INFINITY, |a, b| a.min(b));
            let max_value = numeric_values
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, |a, b| a.max(b));

            // Take up to 10 sample values
            let mut sample_values = numeric_values.clone();
            sample_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sample_values.truncate(10);

            numeric_ranges.insert(
                predicate.to_string(),
                NumericRange {
                    min_value,
                    max_value,
                    count: numeric_values.len() as u64,
                    sample_values,
                },
            );
        }
    }

    /// Analyze string ranges in predicate values
    fn analyze_string_range(
        &self,
        predicate: &str,
        values: &[String],
        string_ranges: &mut HashMap<String, StringRange>,
    ) {
        if values.is_empty() {
            return;
        }

        let min_length = values.iter().map(|s| s.len()).min().unwrap_or(0);
        let max_length = values.iter().map(|s| s.len()).max().unwrap_or(0);

        // Find common prefixes (at least 3 chars and appears in >20% of values)
        let mut prefix_counts: HashMap<String, usize> = HashMap::new();
        for value in values {
            for len in 3..=value.len().min(10) {
                let prefix = &value[..len];
                *prefix_counts.entry(prefix.to_string()).or_insert(0) += 1;
            }
        }

        let threshold = (values.len() as f64 * 0.2) as usize;
        let common_prefixes: Vec<String> = prefix_counts
            .into_iter()
            .filter(|(_, count)| *count >= threshold)
            .map(|(prefix, _)| prefix)
            .collect();

        // Take up to 10 sample values
        let mut sample_values: Vec<String> = values.iter().take(10).cloned().collect();
        sample_values.sort();

        string_ranges.insert(
            predicate.to_string(),
            StringRange {
                min_length,
                max_length,
                common_prefixes,
                sample_values,
            },
        );
    }

    /// Analyze URI patterns in predicate values
    fn analyze_uri_patterns(
        &self,
        predicate: &str,
        values: &[String],
        uri_patterns: &mut HashMap<String, UriPattern>,
    ) {
        let mut base_uris = HashSet::new();
        let mut path_patterns = HashSet::new();
        let mut namespace_prefixes = HashSet::new();

        for value in values {
            if value.starts_with("http://") || value.starts_with("https://") {
                // Extract base URI
                if let Some(pos) = value[8..].find('/').map(|p| p + 8) {
                    base_uris.insert(value[..pos].to_string());

                    // Extract path pattern (remove specific IDs/numbers)
                    let path = &value[pos..];
                    let generalized_path = self.generalize_path(path);
                    path_patterns.insert(generalized_path);
                } else {
                    base_uris.insert(value.clone());
                }

                // Extract namespace (everything before the last '/' or '#')
                if let Some(pos) = value.rfind(&['/', '#'][..]) {
                    namespace_prefixes.insert(value[..=pos].to_string());
                }
            }
        }

        if !base_uris.is_empty() {
            uri_patterns.insert(
                predicate.to_string(),
                UriPattern {
                    base_uris: base_uris.into_iter().collect(),
                    path_patterns: path_patterns.into_iter().collect(),
                    namespace_prefixes: namespace_prefixes.into_iter().collect(),
                },
            );
        }
    }

    /// Analyze datetime ranges in predicate values
    fn analyze_datetime_range(
        &self,
        predicate: &str,
        values: &[String],
        datetime_ranges: &mut HashMap<String, DateTimeRange>,
    ) {
        let mut datetime_values = Vec::new();

        for value in values {
            if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
                datetime_values.push(dt.with_timezone(&Utc));
            }
        }

        if !datetime_values.is_empty() {
            let earliest = *datetime_values.iter().min().unwrap();
            let latest = *datetime_values.iter().max().unwrap();

            // Determine granularity based on the range
            let range_duration = latest.signed_duration_since(earliest);
            let granularity = if range_duration.num_days() > 365 {
                TemporalGranularity::Year
            } else if range_duration.num_days() > 30 {
                TemporalGranularity::Month
            } else if range_duration.num_days() > 1 {
                TemporalGranularity::Day
            } else {
                TemporalGranularity::Hour
            };

            datetime_ranges.insert(
                predicate.to_string(),
                DateTimeRange {
                    earliest,
                    latest,
                    count: datetime_values.len() as u64,
                    granularity,
                },
            );
        }
    }

    fn analyze_temporal_patterns(
        &self,
        predicate: &str,
        values: &[String],
        temporal_patterns: &mut HashMap<String, TemporalPattern>,
    ) {
        let mut datetime_values = Vec::new();

        // Parse datetime values
        for value in values {
            if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
                datetime_values.push(dt.with_timezone(&Utc));
            }
        }

        // Need at least 3 values to detect patterns
        if datetime_values.len() < 3 {
            return;
        }

        // Sort datetime values
        datetime_values.sort();

        // Calculate intervals between consecutive timestamps
        let mut intervals = Vec::new();
        for i in 1..datetime_values.len() {
            let interval = datetime_values[i].signed_duration_since(datetime_values[i - 1]);
            intervals.push(interval.num_seconds().unsigned_abs());
        }

        let pattern_type = self.detect_temporal_pattern_type(&intervals);
        let confidence = self.calculate_pattern_confidence(&intervals, &pattern_type);

        if confidence > 0.5 {
            // Only store patterns with reasonable confidence
            temporal_patterns.insert(
                predicate.to_string(),
                TemporalPattern {
                    pattern_type,
                    frequency: intervals.len() as u64,
                    confidence,
                },
            );
        }
    }

    fn detect_temporal_pattern_type(&self, intervals: &[u64]) -> TemporalPatternType {
        if intervals.is_empty() {
            return TemporalPatternType::Random;
        }

        // Calculate coefficient of variation
        let mean = intervals.iter().sum::<u64>() as f64 / intervals.len() as f64;
        let variance = intervals
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / intervals.len() as f64;
        let std_dev = variance.sqrt();
        let coefficient_of_variation = std_dev / mean;

        // Sequential: intervals are generally increasing
        let is_sequential =
            intervals.windows(2).filter(|w| w[1] > w[0]).count() > intervals.len() / 2;

        // Periodic: low coefficient of variation (regular intervals)
        let is_periodic = coefficient_of_variation < 0.3;

        // Clustered: high variation with some very small intervals
        let min_interval = *intervals.iter().min().unwrap();
        let max_interval = *intervals.iter().max().unwrap();
        let is_clustered = max_interval > min_interval * 10 && coefficient_of_variation > 1.0;

        if is_sequential {
            TemporalPatternType::Sequential
        } else if is_periodic {
            TemporalPatternType::Periodic
        } else if is_clustered {
            TemporalPatternType::Clustered
        } else {
            TemporalPatternType::Random
        }
    }

    fn calculate_pattern_confidence(
        &self,
        intervals: &[u64],
        pattern_type: &TemporalPatternType,
    ) -> f64 {
        if intervals.is_empty() {
            return 0.0;
        }

        let mean = intervals.iter().sum::<u64>() as f64 / intervals.len() as f64;
        let variance = intervals
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / intervals.len() as f64;
        let std_dev = variance.sqrt();
        let coefficient_of_variation = std_dev / mean;

        match pattern_type {
            TemporalPatternType::Sequential => {
                let increasing_count = intervals.windows(2).filter(|w| w[1] > w[0]).count();
                increasing_count as f64 / (intervals.len() - 1) as f64
            }
            TemporalPatternType::Periodic => {
                // Higher confidence for more regular intervals
                1.0 - coefficient_of_variation.min(1.0)
            }
            TemporalPatternType::Clustered => {
                // Confidence based on interval variation
                coefficient_of_variation.min(2.0) / 2.0
            }
            TemporalPatternType::Random => {
                // Low confidence for random patterns
                0.3
            }
        }
    }

    /// Analyze year ranges in predicate values
    fn analyze_year_range(
        &self,
        predicate: &str,
        values: &[String],
        year_ranges: &mut HashMap<String, YearRange>,
    ) {
        let mut year_distribution = HashMap::new();

        for value in values {
            // Try to extract year from various formats
            if let Some(year) = self.extract_year(value) {
                *year_distribution.entry(year).or_insert(0) += 1;
            }
        }

        if !year_distribution.is_empty() {
            let earliest_year = *year_distribution.keys().min().unwrap();
            let latest_year = *year_distribution.keys().max().unwrap();

            year_ranges.insert(
                predicate.to_string(),
                YearRange {
                    earliest_year,
                    latest_year,
                    year_distribution,
                },
            );
        }
    }

    /// Generalize path by replacing numbers with placeholders
    fn generalize_path(&self, path: &str) -> String {
        use regex::Regex;
        let re = Regex::new(r"\d+").unwrap();
        re.replace_all(path, "{id}").to_string()
    }

    /// Extract year from various date formats
    fn extract_year(&self, value: &str) -> Option<i32> {
        // Try ISO date format first
        if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
            return Some(dt.year());
        }

        // Try just year as number
        if let Ok(year) = value.parse::<i32>() {
            if (1900..=2100).contains(&year) {
                return Some(year);
            }
        }

        // Try to find 4-digit year in the string
        use regex::Regex;
        let re = Regex::new(r"\b(19|20)\d{2}\b").unwrap();
        if let Some(captures) = re.find(value) {
            if let Ok(year) = captures.as_str().parse::<i32>() {
                return Some(year);
            }
        }

        None
    }
}
