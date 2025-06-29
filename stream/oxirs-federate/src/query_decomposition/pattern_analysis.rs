//! Pattern analysis algorithms for query optimization
//!
//! This module provides sophisticated pattern analysis including selectivity estimation,
//! join pattern detection, and predicate affinity analysis.

use std::collections::{HashMap, HashSet};

use crate::{
    FederatedService, ServiceCapability,
    planner::TriplePattern,
};

use super::types::*;

impl QueryDecomposer {
    /// Find best service for a specific predicate
    pub fn find_best_service_for_predicate<'a>(
        &self,
        predicate: &str,
        services: &[&'a FederatedService],
    ) -> Option<&'a FederatedService> {
        let mut best_service = None;
        let mut best_score = 0.0;

        for service in services {
            let score = self.calculate_predicate_affinity(predicate, service);
            if score > best_score {
                best_score = score;
                best_service = Some(*service);
            }
        }

        best_service
    }

    /// Calculate affinity score between predicate and service
    pub fn calculate_predicate_affinity(&self, predicate: &str, service: &FederatedService) -> f64 {
        let mut score = 1.0;

        // Check predicate namespace matches
        if predicate.contains("://") {
            let namespace = predicate.split('/').take(3).collect::<Vec<_>>().join("/");
            if service.endpoint.contains(&namespace) {
                score += 2.0;
            }
        }

        // Check known vocabulary matches
        if predicate.contains("foaf:") && service.name.to_lowercase().contains("foaf") {
            score += 1.5;
        }
        if predicate.contains("geo:")
            && service
                .capabilities
                .contains(&ServiceCapability::Geospatial)
        {
            score += 2.0;
        }

        score
    }

    /// Extract variables from a pattern
    pub fn extract_pattern_variables(&self, pattern: &TriplePattern) -> HashSet<String> {
        let mut vars = HashSet::new();

        if pattern.subject.starts_with('?') {
            vars.insert(pattern.subject.clone());
        }
        if pattern.predicate.starts_with('?') {
            vars.insert(pattern.predicate.clone());
        }
        if pattern.object.starts_with('?') {
            vars.insert(pattern.object.clone());
        }

        vars
    }

    /// Estimate result size for patterns
    pub fn estimate_result_size(
        &self,
        service: &FederatedService,
        patterns: &[(usize, TriplePattern)],
    ) -> u64 {
        // Simple estimation based on pattern complexity
        let base_size = 1000;
        let pattern_factor = patterns.len() as u64;
        let selectivity = self.estimate_pattern_selectivity(patterns);

        (base_size * pattern_factor as u64 * selectivity as u64).max(1)
    }

    /// Estimate selectivity of patterns
    pub fn estimate_pattern_selectivity(&self, patterns: &[(usize, TriplePattern)]) -> f64 {
        let mut selectivity = 1.0;

        for (_, pattern) in patterns {
            let var_count = [&pattern.subject, &pattern.predicate, &pattern.object]
                .iter()
                .filter(|p| p.starts_with('?'))
                .count();

            selectivity *= match var_count {
                0 => 0.001, // All constants - very selective
                1 => 0.01,  // One variable
                2 => 0.1,   // Two variables
                3 => 1.0,   // All variables - least selective
                _ => 1.0,
            };
        }

        selectivity
    }

    /// Check if component represents a star join pattern
    pub fn is_star_join_pattern(&self, component: &QueryComponent) -> bool {
        if component.patterns.len() < 3 {
            return false;
        }

        // Count variable frequencies
        let mut var_counts = HashMap::new();
        for (_, pattern) in &component.patterns {
            for var_ref in [&pattern.subject, &pattern.predicate, &pattern.object] {
                if var_ref.starts_with('?') {
                    *var_counts.entry(var_ref.clone()).or_insert(0) += 1;
                }
            }
        }

        // Check if there's a central variable that appears in most patterns
        let max_count = var_counts.values().max().unwrap_or(&0);
        let total_patterns = component.patterns.len();
        
        *max_count >= (total_patterns * 2 / 3) // Central variable in at least 2/3 of patterns
    }

    /// Analyze join patterns in the component
    pub fn analyze_join_patterns(&self, patterns: &[(usize, TriplePattern)]) -> JoinPatternAnalysis {
        let mut analysis = JoinPatternAnalysis::new();
        
        // Build variable-pattern mapping
        let mut var_to_patterns: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, pattern) in patterns {
            for var_ref in [&pattern.subject, &pattern.predicate, &pattern.object] {
                if var_ref.starts_with('?') {
                    var_to_patterns.entry(var_ref.clone()).or_default().push(*idx);
                }
            }
        }
        
        // Analyze join structure
        for (var, pattern_indices) in var_to_patterns {
            if pattern_indices.len() > 1 {
                analysis.join_variables.insert(var.clone(), pattern_indices.clone());
                
                if pattern_indices.len() > analysis.max_join_degree {
                    analysis.max_join_degree = pattern_indices.len();
                    analysis.central_variable = Some(var);
                }
            }
        }
        
        // Determine join pattern type
        analysis.pattern_type = if analysis.max_join_degree >= patterns.len() * 2 / 3 {
            JoinPattern::Star
        } else if analysis.join_variables.len() == patterns.len() - 1 {
            JoinPattern::Chain
        } else {
            JoinPattern::Complex
        };
        
        analysis
    }

    /// Group patterns by join relationships
    pub fn group_patterns_by_joins(
        &self,
        patterns: &[(usize, TriplePattern)],
        join_analysis: &JoinPatternAnalysis,
    ) -> Vec<Vec<(usize, TriplePattern)>> {
        let mut groups = Vec::new();
        let mut assigned = HashSet::new();
        
        // Group patterns that share join variables
        for (var, pattern_indices) in &join_analysis.join_variables {
            if pattern_indices.len() >= 2 {
                let mut group = Vec::new();
                for &idx in pattern_indices {
                    if !assigned.contains(&idx) {
                        if let Some(pattern) = patterns.iter().find(|(i, _)| *i == idx) {
                            group.push(pattern.clone());
                            assigned.insert(idx);
                        }
                    }
                }
                if !group.is_empty() {
                    groups.push(group);
                }
            }
        }
        
        // Add remaining patterns as individual groups
        for pattern in patterns {
            if !assigned.contains(&pattern.0) {
                groups.push(vec![pattern.clone()]);
            }
        }
        
        groups
    }

    /// Calculate pattern selectivity with advanced heuristics
    pub fn calculate_pattern_selectivity(&self, pattern: &TriplePattern) -> PatternSelectivity {
        let mut variable_count = 0;
        let mut constant_count = 0;
        
        for term in [&pattern.subject, &pattern.predicate, &pattern.object] {
            if term.starts_with('?') {
                variable_count += 1;
            } else {
                constant_count += 1;
            }
        }
        
        // Calculate selectivity score based on variable/constant ratio
        let selectivity_score = match (variable_count, constant_count) {
            (0, 3) => 0.001, // All constants - very selective
            (1, 2) => 0.01,  // One variable, two constants
            (2, 1) => 0.1,   // Two variables, one constant
            (3, 0) => 1.0,   // All variables - least selective
            _ => 0.5,        // Default
        };
        
        PatternSelectivity {
            pattern_index: 0, // Would be set by caller
            selectivity_score,
            variable_count,
            constant_count,
        }
    }

    /// Analyze service affinity for patterns
    pub fn analyze_service_affinity(
        &self,
        patterns: &[(usize, TriplePattern)],
        services: &[&FederatedService],
    ) -> Vec<ServiceAffinity> {
        let mut affinities = Vec::new();
        
        for service in services {
            let mut affinity_score = 0.0;
            let mut predicate_matches = Vec::new();
            let mut capability_matches = Vec::new();
            
            // Check predicate matches
            for (_, pattern) in patterns {
                let predicate_affinity = self.calculate_predicate_affinity(&pattern.predicate, service);
                if predicate_affinity > 1.0 {
                    affinity_score += predicate_affinity;
                    predicate_matches.push(pattern.predicate.clone());
                }
            }
            
            // Check capability matches
            for capability in &service.capabilities {
                match capability {
                    ServiceCapability::FullTextSearch => {
                        capability_matches.push("full_text_search".to_string());
                        affinity_score += 0.5;
                    }
                    ServiceCapability::Geospatial => {
                        capability_matches.push("geospatial".to_string());
                        affinity_score += 0.5;
                    }
                    _ => {}
                }
            }
            
            affinities.push(ServiceAffinity {
                service_id: service.id.clone(),
                affinity_score,
                predicate_matches,
                capability_matches,
            });
        }
        
        // Sort by affinity score (descending)
        affinities.sort_by(|a, b| b.affinity_score.partial_cmp(&a.affinity_score).unwrap());
        
        affinities
    }
}

/// Analysis of join patterns in a query component
#[derive(Debug, Clone)]
pub struct JoinPatternAnalysis {
    pub join_variables: HashMap<String, Vec<usize>>,
    pub pattern_type: JoinPattern,
    pub max_join_degree: usize,
    pub central_variable: Option<String>,
}

impl JoinPatternAnalysis {
    pub fn new() -> Self {
        Self {
            join_variables: HashMap::new(),
            pattern_type: JoinPattern::Complex,
            max_join_degree: 0,
            central_variable: None,
        }
    }
}