//! Shape Quality Metrics
//!
//! This module provides comprehensive quality metrics for SHACL shapes, enabling
//! developers to understand and improve the quality of their shape definitions.
//!
//! # Metrics Categories
//!
//! - **Complexity**: Measures the structural complexity of shapes
//! - **Maintainability**: Assesses how easy shapes are to maintain
//! - **Performance**: Predicts validation performance
//! - **Coverage**: Analyzes data coverage and completeness
//! - **Security**: Evaluates security-related aspects
//! - **Best Practices**: Checks compliance with best practices

use crate::{Constraint, Shape, ShapeId, ShapeType};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Configuration for shape quality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAnalysisConfig {
    /// Enable complexity analysis
    pub analyze_complexity: bool,
    /// Enable maintainability scoring
    pub analyze_maintainability: bool,
    /// Enable performance prediction
    pub predict_performance: bool,
    /// Enable coverage analysis
    pub analyze_coverage: bool,
    /// Enable security assessment
    pub analyze_security: bool,
    /// Enable best practices checking
    pub check_best_practices: bool,
    /// Maximum acceptable complexity score
    pub max_complexity_threshold: f64,
    /// Minimum acceptable maintainability score
    pub min_maintainability_threshold: f64,
}

impl Default for QualityAnalysisConfig {
    fn default() -> Self {
        Self {
            analyze_complexity: true,
            analyze_maintainability: true,
            predict_performance: true,
            analyze_coverage: true,
            analyze_security: true,
            check_best_practices: true,
            max_complexity_threshold: 50.0,
            min_maintainability_threshold: 50.0,
        }
    }
}

/// Shape quality metrics analyzer
pub struct ShapeQualityAnalyzer {
    /// Configuration
    config: QualityAnalysisConfig,
    /// Historical metrics for comparison
    history: HashMap<ShapeId, Vec<ShapeQualityReport>>,
}

impl ShapeQualityAnalyzer {
    /// Create a new shape quality analyzer
    pub fn new(config: QualityAnalysisConfig) -> Self {
        Self {
            config,
            history: HashMap::new(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(QualityAnalysisConfig::default())
    }

    /// Analyze a single shape and generate quality report
    pub fn analyze_shape(&mut self, shape: &Shape) -> ShapeQualityReport {
        let complexity = if self.config.analyze_complexity {
            Some(self.compute_complexity_metrics(shape))
        } else {
            None
        };

        let maintainability = if self.config.analyze_maintainability {
            Some(self.compute_maintainability_metrics(shape))
        } else {
            None
        };

        let performance = if self.config.predict_performance {
            Some(self.predict_performance_metrics(shape))
        } else {
            None
        };

        let coverage = if self.config.analyze_coverage {
            Some(self.compute_coverage_metrics(shape))
        } else {
            None
        };

        let security = if self.config.analyze_security {
            Some(self.compute_security_metrics(shape))
        } else {
            None
        };

        let best_practices = if self.config.check_best_practices {
            self.check_best_practices(shape)
        } else {
            Vec::new()
        };

        // Calculate overall score
        let overall_score = self.calculate_overall_score(&complexity, &maintainability, &security);

        // Generate recommendations before moving values
        let recommendations = self.generate_recommendations(
            &complexity.clone().unwrap_or_default(),
            &maintainability.clone().unwrap_or_default(),
            &best_practices,
        );

        let report = ShapeQualityReport {
            shape_id: shape.id.clone(),
            overall_score,
            complexity,
            maintainability,
            performance,
            coverage,
            security,
            best_practices,
            recommendations,
        };

        // Store in history
        self.history
            .entry(shape.id.clone())
            .or_default()
            .push(report.clone());

        report
    }

    /// Analyze multiple shapes
    pub fn analyze_shapes(&mut self, shapes: &[Shape]) -> Vec<ShapeQualityReport> {
        shapes.iter().map(|s| self.analyze_shape(s)).collect()
    }

    /// Compute complexity metrics for a shape
    fn compute_complexity_metrics(&self, shape: &Shape) -> ComplexityMetrics {
        let constraint_count = shape.constraints.len();
        let target_count = shape.targets.len();

        // Calculate cyclomatic complexity (based on logical constraints)
        let cyclomatic_complexity = self.calculate_cyclomatic_complexity(shape);

        // Calculate nesting depth (for recursive/nested shapes)
        let nesting_depth = self.calculate_nesting_depth(shape);

        // Calculate weighted complexity
        let constraint_weights = self.calculate_constraint_weights(shape);
        let weighted_complexity: f64 = constraint_weights.values().sum();

        // Calculate halstead metrics (simplified)
        let halstead_volume = self.calculate_halstead_volume(shape);

        // Overall complexity score (0-100, higher = more complex)
        let complexity_score = (constraint_count as f64 * 2.0
            + cyclomatic_complexity as f64 * 5.0
            + nesting_depth as f64 * 10.0
            + weighted_complexity * 0.5
            + halstead_volume * 0.1)
            .min(100.0);

        ComplexityMetrics {
            complexity_score,
            constraint_count,
            target_count,
            cyclomatic_complexity,
            nesting_depth,
            weighted_complexity,
            halstead_volume,
            constraint_distribution: self.analyze_constraint_distribution(shape),
        }
    }

    /// Calculate cyclomatic complexity
    fn calculate_cyclomatic_complexity(&self, shape: &Shape) -> u32 {
        let mut complexity = 1; // Base complexity

        for constraint in shape.constraints.values() {
            match constraint {
                Constraint::And(_) => complexity += 1,
                Constraint::Or(_) => complexity += 1,
                Constraint::Not(_) => complexity += 1,
                Constraint::Xone(_) => complexity += 2, // Exclusive OR is more complex
                Constraint::QualifiedValueShape(_) => complexity += 2,
                _ => {}
            }
        }

        complexity
    }

    /// Calculate nesting depth
    fn calculate_nesting_depth(&self, shape: &Shape) -> u32 {
        let mut max_depth = 0;

        for constraint in shape.constraints.values() {
            let depth = self.get_constraint_depth(constraint);
            if depth > max_depth {
                max_depth = depth;
            }
        }

        max_depth
    }

    /// Get depth of a constraint (recursive)
    fn get_constraint_depth(&self, constraint: &Constraint) -> u32 {
        match constraint {
            Constraint::And(c) => 1 + c.shapes.len() as u32 / 2,
            Constraint::Or(c) => 1 + c.shapes.len() as u32 / 2,
            Constraint::Not(_) => 2,
            Constraint::Node(_) => 1,
            _ => 0,
        }
    }

    /// Calculate constraint weights
    fn calculate_constraint_weights(&self, shape: &Shape) -> HashMap<String, f64> {
        let mut weights = HashMap::new();

        for (id, constraint) in &shape.constraints {
            let weight = match constraint {
                // Simple constraints
                Constraint::MinCount(_) | Constraint::MaxCount(_) => 1.0,
                Constraint::Datatype(_) => 1.5,
                Constraint::NodeKind(_) => 1.5,

                // String constraints
                Constraint::MinLength(_) | Constraint::MaxLength(_) => 2.0,
                Constraint::Pattern(_) => 3.0, // Regex is expensive

                // Numeric constraints
                Constraint::MinInclusive(_)
                | Constraint::MaxInclusive(_)
                | Constraint::MinExclusive(_)
                | Constraint::MaxExclusive(_) => 2.0,

                // Logical constraints
                Constraint::And(c) => 3.0 + c.shapes.len() as f64,
                Constraint::Or(c) => 3.0 + c.shapes.len() as f64,
                Constraint::Not(_) => 4.0,
                Constraint::Xone(c) => 5.0 + c.shapes.len() as f64,

                // Reference constraints
                Constraint::Node(_) => 4.0,

                // Qualified constraints
                Constraint::QualifiedValueShape(_) => 6.0,

                // Property pair constraints
                Constraint::Equals(_)
                | Constraint::Disjoint(_)
                | Constraint::LessThan(_)
                | Constraint::LessThanOrEquals(_) => 5.0,

                // Set constraints
                Constraint::In(c) => 2.0 + (c.values.len() as f64).log2().max(1.0),
                Constraint::LanguageIn(c) => 2.0 + (c.languages.len() as f64).log2().max(1.0),

                // Other constraints
                _ => 2.0,
            };

            weights.insert(id.as_str().to_string(), weight);
        }

        weights
    }

    /// Calculate Halstead volume (simplified)
    fn calculate_halstead_volume(&self, shape: &Shape) -> f64 {
        let n1 = shape.constraints.len() as f64; // Operators (constraints)
        let n2 = shape.targets.len() as f64 + 1.0; // Operands (targets + shape itself)

        let vocabulary = n1 + n2;
        let length = n1 * 2.0 + n2;

        if vocabulary > 0.0 {
            length * vocabulary.log2()
        } else {
            0.0
        }
    }

    /// Analyze constraint distribution
    fn analyze_constraint_distribution(&self, shape: &Shape) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();

        for constraint in shape.constraints.values() {
            let category = self.categorize_constraint(constraint);
            *distribution.entry(category).or_insert(0) += 1;
        }

        distribution
    }

    /// Categorize a constraint
    fn categorize_constraint(&self, constraint: &Constraint) -> String {
        match constraint {
            Constraint::MinCount(_) | Constraint::MaxCount(_) => "cardinality".to_string(),
            Constraint::Datatype(_) | Constraint::NodeKind(_) => "type".to_string(),
            Constraint::MinLength(_)
            | Constraint::MaxLength(_)
            | Constraint::Pattern(_)
            | Constraint::LanguageIn(_)
            | Constraint::UniqueLang(_) => "string".to_string(),
            Constraint::MinInclusive(_)
            | Constraint::MaxInclusive(_)
            | Constraint::MinExclusive(_)
            | Constraint::MaxExclusive(_) => "numeric".to_string(),
            Constraint::And(_) | Constraint::Or(_) | Constraint::Not(_) | Constraint::Xone(_) => {
                "logical".to_string()
            }
            Constraint::Node(_) => "reference".to_string(),
            Constraint::QualifiedValueShape(_) => "qualified".to_string(),
            Constraint::Equals(_)
            | Constraint::Disjoint(_)
            | Constraint::LessThan(_)
            | Constraint::LessThanOrEquals(_) => "comparison".to_string(),
            Constraint::In(_) | Constraint::HasValue(_) => "value".to_string(),
            Constraint::Class(_) => "class".to_string(),
            Constraint::Closed(_) => "structure".to_string(),
            _ => "other".to_string(),
        }
    }

    /// Compute maintainability metrics
    fn compute_maintainability_metrics(&self, shape: &Shape) -> MaintainabilityMetrics {
        // Documentation score (based on labels and descriptions)
        let documentation_score = self.calculate_documentation_score(shape);

        // Modularity score (based on shape reuse and inheritance)
        let modularity_score = self.calculate_modularity_score(shape);

        // Naming quality
        let naming_score = self.calculate_naming_score(shape);

        // Testability score
        let testability_score = self.calculate_testability_score(shape);

        // Calculate overall maintainability index (0-100)
        let maintainability_index = (documentation_score * 0.25
            + modularity_score * 0.25
            + naming_score * 0.25
            + testability_score * 0.25)
            .clamp(0.0, 100.0);

        MaintainabilityMetrics {
            maintainability_index,
            documentation_score,
            modularity_score,
            naming_score,
            testability_score,
            technical_debt_ratio: self.calculate_technical_debt(shape),
        }
    }

    /// Calculate documentation score
    fn calculate_documentation_score(&self, shape: &Shape) -> f64 {
        let mut score: f64 = 0.0;

        // Label presence
        if shape.label.is_some() {
            score += 30.0;
        }

        // Description presence and quality
        if let Some(desc) = &shape.description {
            score += 30.0;
            // Bonus for longer descriptions
            if desc.len() > 50 {
                score += 10.0;
            }
            if desc.len() > 100 {
                score += 10.0;
            }
        }

        // Messages for constraints
        if !shape.messages.is_empty() {
            score += 20.0;
        }

        score.min(100.0)
    }

    /// Calculate modularity score
    fn calculate_modularity_score(&self, shape: &Shape) -> f64 {
        let mut score = 50.0; // Base score

        // Bonus for using inheritance
        if !shape.extends.is_empty() {
            score += 20.0;
        }

        // Penalty for too many constraints (suggests need for decomposition)
        if shape.constraints.len() > 10 {
            score -= (shape.constraints.len() - 10) as f64 * 2.0;
        }

        // Penalty for too many targets (suggests shape is too broad)
        if shape.targets.len() > 5 {
            score -= (shape.targets.len() - 5) as f64 * 3.0;
        }

        score.clamp(0.0, 100.0)
    }

    /// Calculate naming score
    fn calculate_naming_score(&self, shape: &Shape) -> f64 {
        let mut score: f64 = 50.0;

        let shape_id = shape.id.as_str();

        // Check for descriptive name
        if shape_id.contains(':') {
            score += 10.0; // Has namespace prefix
        }

        // Check name length (not too short, not too long)
        let name_part = shape_id.split(':').next_back().unwrap_or(shape_id);
        if name_part.len() >= 3 && name_part.len() <= 50 {
            score += 20.0;
        }

        // Check for common naming patterns
        if name_part.ends_with("Shape")
            || name_part.ends_with("Constraint")
            || name_part.ends_with("Validator")
        {
            score += 20.0;
        }

        score.min(100.0)
    }

    /// Calculate testability score
    fn calculate_testability_score(&self, shape: &Shape) -> f64 {
        let mut score = 50.0;

        // Fewer constraints = easier to test
        let constraint_penalty = (shape.constraints.len() as f64 / 5.0).min(30.0);
        score -= constraint_penalty;

        // Simple constraint types are easier to test
        let simple_constraints = shape.constraints.values().filter(|c| {
            matches!(
                c,
                Constraint::MinCount(_)
                    | Constraint::MaxCount(_)
                    | Constraint::Datatype(_)
                    | Constraint::NodeKind(_)
            )
        });
        let simple_ratio =
            simple_constraints.count() as f64 / shape.constraints.len().max(1) as f64;
        score += simple_ratio * 30.0;

        // Targets make testing easier (clear focus nodes)
        if !shape.targets.is_empty() {
            score += 20.0;
        }

        score.clamp(0.0, 100.0)
    }

    /// Calculate technical debt ratio
    fn calculate_technical_debt(&self, shape: &Shape) -> f64 {
        let mut debt = 0.0;

        // Missing documentation
        if shape.label.is_none() {
            debt += 5.0;
        }
        if shape.description.is_none() {
            debt += 5.0;
        }

        // Too many constraints (suggests refactoring needed)
        if shape.constraints.len() > 15 {
            debt += (shape.constraints.len() - 15) as f64 * 2.0;
        }

        // Deeply nested logical constraints
        for constraint in shape.constraints.values() {
            let depth = self.get_constraint_depth(constraint);
            if depth > 3 {
                debt += depth as f64 * 3.0;
            }
        }

        debt
    }

    /// Predict performance metrics
    fn predict_performance_metrics(&self, shape: &Shape) -> PerformanceMetrics {
        // Estimate validation time per node
        let base_time_ms = 0.1; // Base overhead

        let constraint_time: f64 = shape
            .constraints
            .values()
            .map(|c| self.estimate_constraint_time(c))
            .sum();

        let estimated_time_per_node_ms = base_time_ms + constraint_time;

        // Estimate memory usage
        let estimated_memory_bytes =
            shape.constraints.len() * 100 + shape.targets.len() * 50 + 1000; // Base + per item

        // Scalability rating (how well it scales with data size)
        let scalability_rating = self.calculate_scalability_rating(shape);

        // Cacheability (how well results can be cached)
        let cacheability_score = self.calculate_cacheability_score(shape);

        // Parallelizability
        let parallelizability_score = self.calculate_parallelizability_score(shape);

        PerformanceMetrics {
            estimated_time_per_node_ms,
            estimated_memory_bytes,
            scalability_rating,
            cacheability_score,
            parallelizability_score,
            bottleneck_constraints: self.identify_bottlenecks(shape),
        }
    }

    /// Estimate time for a constraint
    fn estimate_constraint_time(&self, constraint: &Constraint) -> f64 {
        match constraint {
            Constraint::MinCount(_) | Constraint::MaxCount(_) => 0.05,
            Constraint::Datatype(_) | Constraint::NodeKind(_) => 0.1,
            Constraint::MinLength(_) | Constraint::MaxLength(_) => 0.2,
            Constraint::Pattern(_) => 1.0, // Regex is expensive
            Constraint::MinInclusive(_)
            | Constraint::MaxInclusive(_)
            | Constraint::MinExclusive(_)
            | Constraint::MaxExclusive(_) => 0.15,
            Constraint::And(c) => 0.5 + c.shapes.len() as f64 * 0.5,
            Constraint::Or(c) => 0.5 + c.shapes.len() as f64 * 0.5,
            Constraint::Not(_) => 0.5,
            Constraint::Xone(c) => 1.0 + c.shapes.len() as f64 * 0.7,
            Constraint::Node(_) => 2.0, // Requires nested validation
            Constraint::In(c) => 0.1 + c.values.len() as f64 * 0.01,
            Constraint::QualifiedValueShape(_) => 3.0,
            _ => 0.3,
        }
    }

    /// Calculate scalability rating
    fn calculate_scalability_rating(&self, shape: &Shape) -> ScalabilityRating {
        let complexity = shape.constraints.len() as f64
            + shape
                .constraints
                .values()
                .filter(|c| matches!(c, Constraint::Node(_) | Constraint::QualifiedValueShape(_)))
                .count() as f64
                * 5.0;

        if complexity < 5.0 {
            ScalabilityRating::Excellent
        } else if complexity < 15.0 {
            ScalabilityRating::Good
        } else if complexity < 30.0 {
            ScalabilityRating::Moderate
        } else {
            ScalabilityRating::Poor
        }
    }

    /// Calculate cacheability score
    fn calculate_cacheability_score(&self, shape: &Shape) -> f64 {
        let mut score: f64 = 100.0;

        // Constraints that depend on other values are less cacheable
        for constraint in shape.constraints.values() {
            match constraint {
                Constraint::Equals(_)
                | Constraint::Disjoint(_)
                | Constraint::LessThan(_)
                | Constraint::LessThanOrEquals(_) => score -= 15.0,
                Constraint::Node(_) => score -= 10.0,
                _ => {}
            }
        }

        score.max(0.0)
    }

    /// Calculate parallelizability score
    fn calculate_parallelizability_score(&self, shape: &Shape) -> f64 {
        let mut score: f64 = 100.0;

        // Constraints with dependencies reduce parallelizability
        for constraint in shape.constraints.values() {
            match constraint {
                Constraint::Node(_) => score -= 10.0,
                Constraint::Equals(_)
                | Constraint::Disjoint(_)
                | Constraint::LessThan(_)
                | Constraint::LessThanOrEquals(_) => score -= 5.0,
                _ => {}
            }
        }

        score.max(0.0)
    }

    /// Identify performance bottlenecks
    fn identify_bottlenecks(&self, shape: &Shape) -> Vec<String> {
        let mut bottlenecks = Vec::new();

        for (id, constraint) in &shape.constraints {
            let time = self.estimate_constraint_time(constraint);
            if time > 1.0 {
                bottlenecks.push(format!("{}: {:.2}ms estimated", id.as_str(), time));
            }
        }

        bottlenecks
    }

    /// Compute coverage metrics
    fn compute_coverage_metrics(&self, shape: &Shape) -> CoverageMetrics {
        // Analyze target coverage
        let target_specificity = if shape.targets.is_empty() {
            0.0
        } else {
            100.0 / shape.targets.len() as f64
        };

        // Constraint coverage (variety of constraint types)
        let categories: HashSet<_> = shape
            .constraints
            .values()
            .map(|c| self.categorize_constraint(c))
            .collect();
        let constraint_diversity = (categories.len() as f64 / 10.0 * 100.0).min(100.0);

        // Property completeness
        let has_cardinality = shape
            .constraints
            .values()
            .any(|c| matches!(c, Constraint::MinCount(_) | Constraint::MaxCount(_)));
        let has_type = shape
            .constraints
            .values()
            .any(|c| matches!(c, Constraint::Datatype(_) | Constraint::NodeKind(_)));

        let property_completeness = if has_cardinality && has_type {
            100.0
        } else if has_cardinality || has_type {
            60.0
        } else {
            30.0
        };

        CoverageMetrics {
            target_specificity,
            constraint_diversity,
            property_completeness,
            semantic_coverage: self.estimate_semantic_coverage(shape),
        }
    }

    /// Estimate semantic coverage
    fn estimate_semantic_coverage(&self, shape: &Shape) -> f64 {
        let mut score = 0.0;

        // Check for essential constraint types
        let has_cardinality = shape
            .constraints
            .values()
            .any(|c| matches!(c, Constraint::MinCount(_) | Constraint::MaxCount(_)));
        let has_datatype = shape
            .constraints
            .values()
            .any(|c| matches!(c, Constraint::Datatype(_)));
        let has_value_constraints = shape.constraints.values().any(|c| {
            matches!(
                c,
                Constraint::In(_)
                    | Constraint::MinInclusive(_)
                    | Constraint::MaxInclusive(_)
                    | Constraint::Pattern(_)
            )
        });

        if has_cardinality {
            score += 30.0;
        }
        if has_datatype {
            score += 30.0;
        }
        if has_value_constraints {
            score += 20.0;
        }
        if !shape.targets.is_empty() {
            score += 20.0;
        }

        score
    }

    /// Compute security metrics
    fn compute_security_metrics(&self, shape: &Shape) -> SecurityMetrics {
        let mut vulnerabilities = Vec::new();
        let mut risk_score: f64 = 0.0;

        // Check for security-related issues
        for (id, constraint) in &shape.constraints {
            match constraint {
                Constraint::Pattern(c) => {
                    // Check for potentially dangerous regex patterns
                    let pattern = &c.pattern;
                    if pattern.contains(".*.*.*")
                        || pattern.contains("(.+)+")
                        || pattern.contains("(.*)+")
                    {
                        vulnerabilities.push(SecurityVulnerability {
                            constraint_id: id.as_str().to_string(),
                            severity: VulnerabilitySeverity::High,
                            description: "Potentially catastrophic backtracking regex pattern"
                                .to_string(),
                            recommendation: "Simplify regex pattern to avoid ReDoS attacks"
                                .to_string(),
                        });
                        risk_score += 30.0;
                    }
                }
                Constraint::In(c) if c.values.len() > 1000 => {
                    vulnerabilities.push(SecurityVulnerability {
                        constraint_id: id.as_str().to_string(),
                        severity: VulnerabilitySeverity::Medium,
                        description: "Large sh:in list may cause performance issues".to_string(),
                        recommendation: "Consider using a class or datatype constraint instead"
                            .to_string(),
                    });
                    risk_score += 10.0;
                }
                _ => {}
            }
        }

        // Check for closed world assumption
        let uses_closed = shape
            .constraints
            .values()
            .any(|c| matches!(c, Constraint::Closed(_)));

        SecurityMetrics {
            risk_score: risk_score.min(100.0),
            vulnerabilities,
            uses_closed_constraint: uses_closed,
            data_isolation_score: if uses_closed { 80.0 } else { 50.0 },
        }
    }

    /// Check best practices compliance
    fn check_best_practices(&self, shape: &Shape) -> Vec<BestPracticeViolation> {
        let mut violations = Vec::new();

        // BP1: Shape should have a label
        if shape.label.is_none() {
            violations.push(BestPracticeViolation {
                code: "BP001".to_string(),
                severity: BestPracticeSeverity::Warning,
                message: "Shape is missing a label (rdfs:label)".to_string(),
                recommendation: "Add a human-readable label to improve discoverability".to_string(),
            });
        }

        // BP2: Shape should have a description
        if shape.description.is_none() {
            violations.push(BestPracticeViolation {
                code: "BP002".to_string(),
                severity: BestPracticeSeverity::Info,
                message: "Shape is missing a description (rdfs:comment)".to_string(),
                recommendation: "Add a description explaining the purpose of this shape"
                    .to_string(),
            });
        }

        // BP3: Shape should have targets
        if shape.targets.is_empty() {
            violations.push(BestPracticeViolation {
                code: "BP003".to_string(),
                severity: BestPracticeSeverity::Warning,
                message: "Shape has no targets defined".to_string(),
                recommendation:
                    "Define targets to specify which nodes should be validated against this shape"
                        .to_string(),
            });
        }

        // BP4: Avoid too many constraints
        if shape.constraints.len() > 20 {
            violations.push(BestPracticeViolation {
                code: "BP004".to_string(),
                severity: BestPracticeSeverity::Warning,
                message: format!(
                    "Shape has {} constraints (recommended maximum is 20)",
                    shape.constraints.len()
                ),
                recommendation: "Consider decomposing this shape into smaller, reusable shapes"
                    .to_string(),
            });
        }

        // BP5: Property shapes should have a path
        if matches!(shape.shape_type, ShapeType::PropertyShape) && shape.path.is_none() {
            violations.push(BestPracticeViolation {
                code: "BP005".to_string(),
                severity: BestPracticeSeverity::Error,
                message: "Property shape is missing a path (sh:path)".to_string(),
                recommendation: "Define the property path that this shape validates".to_string(),
            });
        }

        // BP6: Avoid duplicate constraint types
        let constraint_types: Vec<_> = shape
            .constraints
            .values()
            .map(std::mem::discriminant)
            .collect();
        let unique_types: HashSet<_> = constraint_types.iter().collect();
        if constraint_types.len() > unique_types.len() * 2 {
            violations.push(BestPracticeViolation {
                code: "BP006".to_string(),
                severity: BestPracticeSeverity::Info,
                message: "Shape contains many constraints of the same type".to_string(),
                recommendation: "Consider consolidating similar constraints".to_string(),
            });
        }

        violations
    }

    /// Generate recommendations based on metrics
    fn generate_recommendations(
        &self,
        complexity: &ComplexityMetrics,
        maintainability: &MaintainabilityMetrics,
        best_practices: &[BestPracticeViolation],
    ) -> Vec<QualityRecommendation> {
        let mut recommendations = Vec::new();

        // Complexity recommendations
        if complexity.complexity_score > self.config.max_complexity_threshold {
            recommendations.push(QualityRecommendation {
                priority: RecommendationPriority::High,
                category: QualityCategory::Complexity,
                title: "High Shape Complexity".to_string(),
                description: format!(
                    "Complexity score ({:.1}) exceeds threshold ({:.1})",
                    complexity.complexity_score, self.config.max_complexity_threshold
                ),
                actions: vec![
                    "Break down into smaller, focused shapes".to_string(),
                    "Use shape inheritance to share common constraints".to_string(),
                    "Simplify nested logical constraints".to_string(),
                ],
            });
        }

        // Maintainability recommendations
        if maintainability.maintainability_index < self.config.min_maintainability_threshold {
            recommendations.push(QualityRecommendation {
                priority: RecommendationPriority::Medium,
                category: QualityCategory::Maintainability,
                title: "Low Maintainability Score".to_string(),
                description: format!(
                    "Maintainability index ({:.1}) is below threshold ({:.1})",
                    maintainability.maintainability_index,
                    self.config.min_maintainability_threshold
                ),
                actions: vec![
                    "Add documentation (labels, descriptions)".to_string(),
                    "Improve naming conventions".to_string(),
                    "Reduce constraint count through decomposition".to_string(),
                ],
            });
        }

        // Technical debt recommendations
        if maintainability.technical_debt_ratio > 20.0 {
            recommendations.push(QualityRecommendation {
                priority: RecommendationPriority::Medium,
                category: QualityCategory::TechnicalDebt,
                title: "Technical Debt Detected".to_string(),
                description: format!(
                    "Technical debt ratio ({:.1}) suggests refactoring is needed",
                    maintainability.technical_debt_ratio
                ),
                actions: vec![
                    "Address missing documentation".to_string(),
                    "Refactor complex constraint combinations".to_string(),
                    "Improve test coverage".to_string(),
                ],
            });
        }

        // Best practice recommendations
        let errors: Vec<_> = best_practices
            .iter()
            .filter(|v| v.severity == BestPracticeSeverity::Error)
            .collect();
        if !errors.is_empty() {
            recommendations.push(QualityRecommendation {
                priority: RecommendationPriority::Critical,
                category: QualityCategory::BestPractices,
                title: "Best Practice Errors".to_string(),
                description: format!("{} best practice error(s) found", errors.len()),
                actions: errors.iter().map(|e| e.recommendation.clone()).collect(),
            });
        }

        recommendations
    }

    /// Calculate overall quality score
    fn calculate_overall_score(
        &self,
        complexity: &Option<ComplexityMetrics>,
        maintainability: &Option<MaintainabilityMetrics>,
        security: &Option<SecurityMetrics>,
    ) -> f64 {
        let mut score = 0.0;
        let mut factors = 0.0;

        if let Some(c) = complexity {
            // Lower complexity is better
            let complexity_contribution = (100.0 - c.complexity_score).clamp(0.0, 100.0);
            score += complexity_contribution;
            factors += 1.0;
        }

        if let Some(m) = maintainability {
            score += m.maintainability_index.clamp(0.0, 100.0);
            factors += 1.0;
        }

        if let Some(s) = security {
            // Lower risk is better
            score += (100.0 - s.risk_score).clamp(0.0, 100.0);
            factors += 1.0;
        }

        if factors > 0.0 {
            (score / factors).clamp(0.0, 100.0)
        } else {
            50.0 // Neutral score if no metrics
        }
    }

    /// Get historical trend for a shape
    pub fn get_quality_trend(&self, shape_id: &ShapeId) -> Option<Vec<f64>> {
        self.history
            .get(shape_id)
            .map(|reports| reports.iter().map(|r| r.overall_score).collect())
    }

    /// Compare two shapes
    pub fn compare_shapes(&mut self, shape1: &Shape, shape2: &Shape) -> ShapeComparison {
        let report1 = self.analyze_shape(shape1);
        let report2 = self.analyze_shape(shape2);

        ShapeComparison {
            shape1_id: shape1.id.clone(),
            shape2_id: shape2.id.clone(),
            score_difference: report2.overall_score - report1.overall_score,
            complexity_difference: report2
                .complexity
                .as_ref()
                .map(|c| c.complexity_score)
                .unwrap_or(0.0)
                - report1
                    .complexity
                    .as_ref()
                    .map(|c| c.complexity_score)
                    .unwrap_or(0.0),
            maintainability_difference: report2
                .maintainability
                .as_ref()
                .map(|m| m.maintainability_index)
                .unwrap_or(0.0)
                - report1
                    .maintainability
                    .as_ref()
                    .map(|m| m.maintainability_index)
                    .unwrap_or(0.0),
            report1,
            report2,
        }
    }
}

impl Default for ShapeQualityAnalyzer {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Shape quality report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeQualityReport {
    /// Shape ID
    pub shape_id: ShapeId,
    /// Overall quality score (0-100)
    pub overall_score: f64,
    /// Complexity metrics
    pub complexity: Option<ComplexityMetrics>,
    /// Maintainability metrics
    pub maintainability: Option<MaintainabilityMetrics>,
    /// Performance predictions
    pub performance: Option<PerformanceMetrics>,
    /// Coverage metrics
    pub coverage: Option<CoverageMetrics>,
    /// Security metrics
    pub security: Option<SecurityMetrics>,
    /// Best practice violations
    pub best_practices: Vec<BestPracticeViolation>,
    /// Quality recommendations
    pub recommendations: Vec<QualityRecommendation>,
}

/// Complexity metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    /// Overall complexity score (0-100)
    pub complexity_score: f64,
    /// Number of constraints
    pub constraint_count: usize,
    /// Number of targets
    pub target_count: usize,
    /// Cyclomatic complexity
    pub cyclomatic_complexity: u32,
    /// Maximum nesting depth
    pub nesting_depth: u32,
    /// Weighted complexity
    pub weighted_complexity: f64,
    /// Halstead volume
    pub halstead_volume: f64,
    /// Constraint distribution by category
    pub constraint_distribution: HashMap<String, usize>,
}

/// Maintainability metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MaintainabilityMetrics {
    /// Maintainability index (0-100)
    pub maintainability_index: f64,
    /// Documentation completeness score
    pub documentation_score: f64,
    /// Modularity score
    pub modularity_score: f64,
    /// Naming quality score
    pub naming_score: f64,
    /// Testability score
    pub testability_score: f64,
    /// Technical debt ratio
    pub technical_debt_ratio: f64,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Estimated validation time per node (milliseconds)
    pub estimated_time_per_node_ms: f64,
    /// Estimated memory usage (bytes)
    pub estimated_memory_bytes: usize,
    /// Scalability rating
    pub scalability_rating: ScalabilityRating,
    /// Cacheability score (0-100)
    pub cacheability_score: f64,
    /// Parallelizability score (0-100)
    pub parallelizability_score: f64,
    /// Identified performance bottlenecks
    pub bottleneck_constraints: Vec<String>,
}

/// Scalability rating
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalabilityRating {
    Excellent,
    Good,
    Moderate,
    Poor,
}

/// Coverage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageMetrics {
    /// Target specificity (higher = more specific)
    pub target_specificity: f64,
    /// Constraint diversity (variety of constraint types)
    pub constraint_diversity: f64,
    /// Property completeness
    pub property_completeness: f64,
    /// Semantic coverage score
    pub semantic_coverage: f64,
}

/// Security metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetrics {
    /// Overall security risk score (0-100, lower is better)
    pub risk_score: f64,
    /// Identified vulnerabilities
    pub vulnerabilities: Vec<SecurityVulnerability>,
    /// Uses sh:closed constraint
    pub uses_closed_constraint: bool,
    /// Data isolation score
    pub data_isolation_score: f64,
}

/// Security vulnerability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityVulnerability {
    /// Constraint ID
    pub constraint_id: String,
    /// Severity
    pub severity: VulnerabilitySeverity,
    /// Description
    pub description: String,
    /// Recommendation
    pub recommendation: String,
}

/// Vulnerability severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VulnerabilitySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Best practice violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestPracticeViolation {
    /// Violation code
    pub code: String,
    /// Severity
    pub severity: BestPracticeSeverity,
    /// Message
    pub message: String,
    /// Recommendation
    pub recommendation: String,
}

/// Best practice severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BestPracticeSeverity {
    Info,
    Warning,
    Error,
}

/// Quality recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRecommendation {
    /// Priority
    pub priority: RecommendationPriority,
    /// Category
    pub category: QualityCategory,
    /// Title
    pub title: String,
    /// Description
    pub description: String,
    /// Suggested actions
    pub actions: Vec<String>,
}

/// Recommendation priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Quality category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityCategory {
    Complexity,
    Maintainability,
    Performance,
    Security,
    BestPractices,
    TechnicalDebt,
}

/// Shape comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeComparison {
    /// First shape ID
    pub shape1_id: ShapeId,
    /// Second shape ID
    pub shape2_id: ShapeId,
    /// Overall score difference (positive = shape2 is better)
    pub score_difference: f64,
    /// Complexity difference
    pub complexity_difference: f64,
    /// Maintainability difference
    pub maintainability_difference: f64,
    /// First shape report
    pub report1: ShapeQualityReport,
    /// Second shape report
    pub report2: ShapeQualityReport,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_shape() -> Shape {
        let mut shape = Shape::new(ShapeId::new("ex:TestShape"), ShapeType::NodeShape);
        shape.label = Some("Test Shape".to_string());
        shape.description = Some("A shape for testing quality metrics".to_string());
        shape
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = ShapeQualityAnalyzer::default_config();
        assert!(analyzer.config.analyze_complexity);
        assert!(analyzer.config.analyze_maintainability);
    }

    #[test]
    fn test_basic_quality_analysis() {
        let mut analyzer = ShapeQualityAnalyzer::default_config();
        let shape = create_test_shape();

        let report = analyzer.analyze_shape(&shape);

        assert!(report.overall_score > 0.0);
        assert!(report.overall_score <= 100.0);
        assert!(report.complexity.is_some());
        assert!(report.maintainability.is_some());
    }

    #[test]
    fn test_complexity_metrics() {
        let analyzer = ShapeQualityAnalyzer::default_config();
        let shape = create_test_shape();

        let metrics = analyzer.compute_complexity_metrics(&shape);

        assert_eq!(metrics.constraint_count, 0);
        assert!(metrics.complexity_score >= 0.0);
    }

    #[test]
    fn test_maintainability_metrics() {
        let analyzer = ShapeQualityAnalyzer::default_config();
        let shape = create_test_shape();

        let metrics = analyzer.compute_maintainability_metrics(&shape);

        // Should have good documentation score since we have label and description
        assert!(metrics.documentation_score > 50.0);
        assert!(metrics.maintainability_index > 0.0);
    }

    #[test]
    fn test_best_practices_check() {
        let analyzer = ShapeQualityAnalyzer::default_config();

        // Shape without label
        let mut shape = Shape::new(ShapeId::new("ex:TestShape"), ShapeType::NodeShape);
        let violations = analyzer.check_best_practices(&shape);

        assert!(!violations.is_empty());
        assert!(violations.iter().any(|v| v.code == "BP001"));

        // Shape with label
        shape.label = Some("Test".to_string());
        let violations = analyzer.check_best_practices(&shape);
        assert!(!violations.iter().any(|v| v.code == "BP001"));
    }

    #[test]
    fn test_performance_prediction() {
        let analyzer = ShapeQualityAnalyzer::default_config();
        let shape = create_test_shape();

        let metrics = analyzer.predict_performance_metrics(&shape);

        assert!(metrics.estimated_time_per_node_ms > 0.0);
        assert!(metrics.estimated_memory_bytes > 0);
    }

    #[test]
    fn test_security_metrics() {
        let analyzer = ShapeQualityAnalyzer::default_config();
        let shape = create_test_shape();

        let metrics = analyzer.compute_security_metrics(&shape);

        // Clean shape should have low risk
        assert!(metrics.risk_score < 50.0);
    }

    #[test]
    fn test_quality_trend() {
        let mut analyzer = ShapeQualityAnalyzer::default_config();
        let shape = create_test_shape();

        // Analyze multiple times
        analyzer.analyze_shape(&shape);
        analyzer.analyze_shape(&shape);

        let trend = analyzer.get_quality_trend(&shape.id);
        assert!(trend.is_some());
        assert_eq!(trend.expect("operation should succeed").len(), 2);
    }

    #[test]
    fn test_shape_comparison() {
        let mut analyzer = ShapeQualityAnalyzer::default_config();

        let shape1 = create_test_shape();
        let mut shape2 = Shape::new(ShapeId::new("ex:TestShape2"), ShapeType::NodeShape);
        shape2.label = Some("Shape 2".to_string());

        let comparison = analyzer.compare_shapes(&shape1, &shape2);

        assert_eq!(comparison.shape1_id, shape1.id);
        assert_eq!(comparison.shape2_id, shape2.id);
    }

    #[test]
    fn test_configuration() {
        let config = QualityAnalysisConfig {
            analyze_complexity: false,
            ..Default::default()
        };

        let mut analyzer = ShapeQualityAnalyzer::new(config);
        let shape = create_test_shape();

        let report = analyzer.analyze_shape(&shape);

        assert!(report.complexity.is_none());
        assert!(report.maintainability.is_some());
    }
}
