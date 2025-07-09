//! Shape Library System
//!
//! This module provides a comprehensive library system for managing,
//! discovering, and organizing SHACL shapes and patterns.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::{shape::Shape as AiShape, Result};
use oxirs_shacl::ShapeId;

use super::ShapePattern;

/// Shape library for organizing and discovering shapes
#[derive(Debug)]
pub struct ShapeLibrary {
    pub pattern_repository: PatternRepository,
    pub shape_catalog: HashMap<ShapeId, ShapeEntry>,
    pub categories: HashMap<String, Category>,
    pub search_index: SearchIndex,
}

/// Pattern repository for managing reusable patterns
#[derive(Debug)]
pub struct PatternRepository {
    pub patterns: HashMap<String, ShapePattern>,
    pub pattern_collections: HashMap<String, PatternCollection>,
    pub pattern_metadata: HashMap<String, PatternMetadata>,
}

/// Shape entry in the library
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeEntry {
    pub shape_id: ShapeId,
    pub name: String,
    pub description: String,
    pub category: String,
    pub tags: HashSet<String>,
    pub version: String,
    pub author: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub usage_count: u64,
    pub rating: f64,
    pub complexity_score: f64,
    pub dependencies: Vec<ShapeId>,
    pub related_shapes: Vec<ShapeId>,
    pub examples: Vec<UsageExample>,
}

/// Category for organizing shapes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Category {
    pub category_id: String,
    pub name: String,
    pub description: String,
    pub parent_category: Option<String>,
    pub subcategories: Vec<String>,
    pub shape_count: usize,
    pub metadata: HashMap<String, String>,
}

/// Search index for fast shape discovery
#[derive(Debug)]
pub struct SearchIndex {
    pub text_index: HashMap<String, Vec<ShapeId>>,
    pub tag_index: HashMap<String, Vec<ShapeId>>,
    pub category_index: HashMap<String, Vec<ShapeId>>,
    pub complexity_index: HashMap<ComplexityRange, Vec<ShapeId>>,
    pub semantic_index: HashMap<String, Vec<ShapeId>>,
}

/// Usage example for shapes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageExample {
    pub example_id: String,
    pub title: String,
    pub description: String,
    pub code_sample: String,
    pub expected_result: String,
    pub difficulty_level: DifficultyLevel,
    pub prerequisites: Vec<String>,
}

/// Difficulty levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DifficultyLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

/// Complexity range for indexing
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum ComplexityRange {
    VeryLow,  // 0.0 - 2.0
    Low,      // 2.0 - 4.0
    Medium,   // 4.0 - 6.0
    High,     // 6.0 - 8.0
    VeryHigh, // 8.0 - 10.0
}

/// Pattern collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternCollection {
    pub collection_id: String,
    pub name: String,
    pub description: String,
    pub patterns: Vec<String>,
    pub curator: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub is_public: bool,
    pub download_count: u64,
}

/// Pattern metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMetadata {
    pub pattern_id: String,
    pub quality_score: f64,
    pub validation_status: ValidationStatus,
    pub performance_metrics: PerformanceMetrics,
    pub compatibility_info: CompatibilityInfo,
    pub review_comments: Vec<ReviewComment>,
}

/// Validation status for patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    NotValidated,
    InValidation,
    Validated,
    ValidationFailed,
    Deprecated,
}

/// Performance metrics for patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub average_validation_time_ms: f64,
    pub memory_usage_kb: f64,
    pub cpu_utilization: f64,
    pub throughput_operations_per_second: f64,
    pub error_rate: f64,
}

/// Compatibility information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityInfo {
    pub supported_versions: Vec<String>,
    pub deprecated_features: Vec<String>,
    pub migration_notes: Vec<String>,
    pub breaking_changes: Vec<String>,
}

/// Review comment for patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewComment {
    pub comment_id: String,
    pub reviewer: String,
    pub comment: String,
    pub rating: u8, // 1-5 stars
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub helpful_votes: u32,
    pub tags: Vec<String>,
}

/// Search query for shapes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeSearchQuery {
    pub text_query: Option<String>,
    pub tags: Vec<String>,
    pub categories: Vec<String>,
    pub complexity_range: Option<(f64, f64)>,
    pub author: Option<String>,
    pub min_rating: Option<f64>,
    pub created_after: Option<chrono::DateTime<chrono::Utc>>,
    pub created_before: Option<chrono::DateTime<chrono::Utc>>,
    pub has_examples: Option<bool>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub sort_by: Option<SortOption>,
}

/// Sort options for search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortOption {
    Relevance,
    Rating,
    UsageCount,
    CreatedDate,
    UpdatedDate,
    Complexity,
    Name,
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeSearchResult {
    pub shape_id: ShapeId,
    pub relevance_score: f64,
    pub snippet: String,
    pub highlighted_fields: HashMap<String, String>,
}

impl ShapeLibrary {
    pub fn new() -> Self {
        Self {
            pattern_repository: PatternRepository::new(),
            shape_catalog: HashMap::new(),
            categories: HashMap::new(),
            search_index: SearchIndex::new(),
        }
    }

    pub fn pattern_repository(&self) -> &PatternRepository {
        &self.pattern_repository
    }

    pub fn search_shapes(&self, query: &ShapeSearchQuery) -> Result<Vec<ShapeSearchResult>> {
        // This would implement comprehensive search logic
        // For now, return empty results
        Ok(Vec::new())
    }

    pub fn add_shape(&mut self, shape: &AiShape, entry: ShapeEntry) -> Result<()> {
        // Add shape to catalog
        self.shape_catalog
            .insert(shape.id().into(), entry.clone());

        // Update search indices
        self.update_search_indices(&entry)?;

        Ok(())
    }

    pub fn get_shape(&self, shape_id: &ShapeId) -> Option<&ShapeEntry> {
        self.shape_catalog.get(shape_id)
    }

    pub fn get_shapes_by_category(&self, category: &str) -> Vec<&ShapeEntry> {
        self.shape_catalog
            .values()
            .filter(|entry| entry.category == category)
            .collect()
    }

    pub fn get_popular_shapes(&self, limit: usize) -> Vec<&ShapeEntry> {
        let mut shapes: Vec<&ShapeEntry> = self.shape_catalog.values().collect();
        shapes.sort_by(|a, b| b.usage_count.cmp(&a.usage_count));
        shapes.into_iter().take(limit).collect()
    }

    pub fn get_highly_rated_shapes(&self, min_rating: f64, limit: usize) -> Vec<&ShapeEntry> {
        let mut shapes: Vec<&ShapeEntry> = self
            .shape_catalog
            .values()
            .filter(|entry| entry.rating >= min_rating)
            .collect();
        shapes.sort_by(|a, b| b.rating.partial_cmp(&a.rating).unwrap());
        shapes.into_iter().take(limit).collect()
    }

    fn update_search_indices(&mut self, entry: &ShapeEntry) -> Result<()> {
        // Update text index with name and description
        let words: Vec<String> = entry
            .name
            .split_whitespace()
            .chain(entry.description.split_whitespace())
            .map(|s| s.to_lowercase())
            .collect();

        for word in words {
            self.search_index
                .text_index
                .entry(word)
                .or_default()
                .push(entry.shape_id.clone());
        }

        // Update tag index
        for tag in &entry.tags {
            self.search_index
                .tag_index
                .entry(tag.clone())
                .or_default()
                .push(entry.shape_id.clone());
        }

        // Update category index
        self.search_index
            .category_index
            .entry(entry.category.clone())
            .or_default()
            .push(entry.shape_id.clone());

        // Update complexity index
        let complexity_range = match entry.complexity_score {
            score if score < 2.0 => ComplexityRange::VeryLow,
            score if score < 4.0 => ComplexityRange::Low,
            score if score < 6.0 => ComplexityRange::Medium,
            score if score < 8.0 => ComplexityRange::High,
            _ => ComplexityRange::VeryHigh,
        };

        self.search_index
            .complexity_index
            .entry(complexity_range)
            .or_default()
            .push(entry.shape_id.clone());

        Ok(())
    }
}

impl Default for PatternRepository {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternRepository {
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            pattern_collections: HashMap::new(),
            pattern_metadata: HashMap::new(),
        }
    }

    pub fn add_pattern(&mut self, pattern: ShapePattern) -> Result<()> {
        let pattern_id = pattern.pattern_id.clone();
        self.patterns.insert(pattern_id.clone(), pattern);

        // Initialize metadata
        self.pattern_metadata.insert(
            pattern_id.clone(),
            PatternMetadata {
                pattern_id: pattern_id.clone(),
                quality_score: 0.0,
                validation_status: ValidationStatus::NotValidated,
                performance_metrics: PerformanceMetrics {
                    average_validation_time_ms: 0.0,
                    memory_usage_kb: 0.0,
                    cpu_utilization: 0.0,
                    throughput_operations_per_second: 0.0,
                    error_rate: 0.0,
                },
                compatibility_info: CompatibilityInfo {
                    supported_versions: Vec::new(),
                    deprecated_features: Vec::new(),
                    migration_notes: Vec::new(),
                    breaking_changes: Vec::new(),
                },
                review_comments: Vec::new(),
            },
        );

        Ok(())
    }

    pub fn get_pattern(&self, pattern_id: &str) -> Option<&ShapePattern> {
        self.patterns.get(pattern_id)
    }

    pub fn get_patterns_by_type(&self, pattern_type: &str) -> Vec<&ShapePattern> {
        self.patterns
            .values()
            .filter(|pattern| format!("{:?}", pattern.pattern_type) == pattern_type)
            .collect()
    }
}

impl Default for SearchIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchIndex {
    pub fn new() -> Self {
        Self {
            text_index: HashMap::new(),
            tag_index: HashMap::new(),
            category_index: HashMap::new(),
            complexity_index: HashMap::new(),
            semantic_index: HashMap::new(),
        }
    }
}

impl ToString for ComplexityRange {
    fn to_string(&self) -> String {
        match self {
            ComplexityRange::VeryLow => "very_low".to_string(),
            ComplexityRange::Low => "low".to_string(),
            ComplexityRange::Medium => "medium".to_string(),
            ComplexityRange::High => "high".to_string(),
            ComplexityRange::VeryHigh => "very_high".to_string(),
        }
    }
}

impl Default for ShapeLibrary {
    fn default() -> Self {
        Self::new()
    }
}
