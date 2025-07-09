//! Pattern mining algorithms and analysis functions

use tracing::debug;

use oxirs_core::Store;

use super::types::*;
use super::patterns::*;
use super::engine::FrequencyTables;
use crate::Result;

/// Discover frequent itemsets using Apriori algorithm
pub fn discover_frequent_itemsets(
    frequency_tables: &FrequencyTables,
    config: &AdvancedPatternMiningConfig,
) -> Result<Vec<Vec<String>>> {
    debug!("Discovering frequent itemsets with min_support: {}", config.min_support);

    let mut frequent_itemsets = Vec::new();
    let total_transactions = calculate_total_transactions(frequency_tables);
    let min_support_count = (config.min_support * total_transactions as f64) as usize;

    // Generate 1-itemsets
    let mut current_itemsets = generate_1_itemsets(frequency_tables, min_support_count);
    let mut k = 1;

    while !current_itemsets.is_empty() && k < config.max_pattern_length {
        debug!("Found {} frequent {}-itemsets", current_itemsets.len(), k);
        frequent_itemsets.extend(current_itemsets.clone());

        // Generate candidate (k+1)-itemsets
        let candidates = generate_candidate_itemsets(&current_itemsets, k + 1);
        
        // Prune candidates that don't meet support threshold
        current_itemsets = candidates
            .into_iter()
            .filter(|itemset| {
                let support = calculate_itemset_support(itemset, frequency_tables);
                support >= min_support_count
            })
            .collect();

        k += 1;
    }

    debug!("Total frequent itemsets discovered: {}", frequent_itemsets.len());
    Ok(frequent_itemsets)
}

/// Generate association rules from frequent itemsets
pub fn generate_association_rules(
    frequent_itemsets: &[Vec<String>],
    frequency_tables: &FrequencyTables,
    config: &AdvancedPatternMiningConfig,
) -> Result<Vec<AdvancedPattern>> {
    debug!("Generating association rules from {} itemsets", frequent_itemsets.len());

    let mut patterns = Vec::new();
    let total_transactions = calculate_total_transactions(frequency_tables);

    for itemset in frequent_itemsets.iter().filter(|is| is.len() >= 2) {
        // Generate all possible rules from this itemset
        for i in 1..itemset.len() {
            let combinations = generate_combinations(itemset, i);
            
            for antecedent in combinations {
                let consequent: Vec<String> = itemset
                    .iter()
                    .filter(|item| !antecedent.contains(item))
                    .cloned()
                    .collect();

                if consequent.is_empty() {
                    continue;
                }

                // Calculate metrics
                let support_count = calculate_itemset_support(itemset, frequency_tables);
                let antecedent_support = calculate_itemset_support(&antecedent, frequency_tables);
                let consequent_support = calculate_itemset_support(&consequent, frequency_tables);

                let support_ratio = support_count as f64 / total_transactions as f64;
                let confidence = if antecedent_support > 0 {
                    support_count as f64 / antecedent_support as f64
                } else {
                    0.0
                };

                let lift = if consequent_support > 0 {
                    (support_count as f64 * total_transactions as f64) 
                        / (antecedent_support as f64 * consequent_support as f64)
                } else {
                    0.0
                };

                let conviction = if confidence < 1.0 {
                    (1.0 - (consequent_support as f64 / total_transactions as f64)) / (1.0 - confidence)
                } else {
                    f64::INFINITY
                };

                if confidence >= config.min_confidence {
                    let pattern_items = create_pattern_items(&antecedent, &consequent);
                    let quality_score = calculate_quality_score(support_ratio, confidence, lift, conviction);

                    let pattern = AdvancedPattern {
                        items: pattern_items,
                        support_count,
                        support_ratio,
                        confidence,
                        lift,
                        conviction,
                        quality_score,
                        pattern_type: classify_pattern_type(&antecedent, &consequent),
                        temporal_info: None,
                        hierarchy_level: 0,
                        suggested_constraints: Vec::new(),
                    };

                    patterns.push(pattern);
                }
            }
        }
    }

    debug!("Generated {} association rules", patterns.len());
    Ok(patterns)
}

/// Enhance patterns with temporal analysis
pub fn enhance_with_temporal_analysis(
    patterns: &mut [AdvancedPattern],
    store: &dyn Store,
    graph_name: Option<&str>,
    config: &AdvancedPatternMiningConfig,
) -> Result<()> {
    debug!("Enhancing {} patterns with temporal analysis", patterns.len());

    for pattern in patterns.iter_mut() {
        if let Some(temporal_info) = analyze_temporal_characteristics(pattern, store, graph_name)? {
            pattern.temporal_info = Some(temporal_info);
            if pattern.pattern_type == PatternType::Structural {
                pattern.pattern_type = PatternType::Temporal;
            } else if pattern.pattern_type != PatternType::Temporal {
                pattern.pattern_type = PatternType::Mixed;
            }
        }
    }

    Ok(())
}

/// Analyze hierarchical patterns
pub fn analyze_hierarchical_patterns(
    patterns: &mut [AdvancedPattern],
    frequency_tables: &FrequencyTables,
    config: &AdvancedPatternMiningConfig,
) -> Result<()> {
    debug!("Analyzing hierarchical patterns for {} patterns", patterns.len());

    for pattern in patterns.iter_mut() {
        pattern.hierarchy_level = calculate_hierarchy_level(pattern, frequency_tables);
        
        if pattern.hierarchy_level > 0 {
            // Adjust quality score for hierarchical patterns
            pattern.quality_score *= 1.0 + (pattern.hierarchy_level as f64 * 0.1);
        }
    }

    Ok(())
}

/// Generate SHACL constraint suggestions
pub fn generate_constraint_suggestions(
    patterns: &mut [AdvancedPattern],
    config: &AdvancedPatternMiningConfig,
) -> Result<()> {
    debug!("Generating SHACL constraint suggestions for {} patterns", patterns.len());

    for pattern in patterns.iter_mut() {
        pattern.suggested_constraints = create_constraint_suggestions(pattern, config);
    }

    Ok(())
}

// Helper functions

fn calculate_total_transactions(frequency_tables: &FrequencyTables) -> usize {
    frequency_tables.properties.values().sum::<usize>().max(
        frequency_tables.classes.values().sum::<usize>().max(
            frequency_tables.value_patterns.values().sum()
        )
    )
}

fn generate_1_itemsets(frequency_tables: &FrequencyTables, min_support: usize) -> Vec<Vec<String>> {
    let mut itemsets = Vec::new();

    // Add frequent properties
    for (prop, &count) in &frequency_tables.properties {
        if count >= min_support {
            itemsets.push(vec![format!("prop:{}", prop)]);
        }
    }

    // Add frequent classes
    for (class, &count) in &frequency_tables.classes {
        if count >= min_support {
            itemsets.push(vec![format!("class:{}", class)]);
        }
    }

    // Add frequent value patterns
    for (pattern, &count) in &frequency_tables.value_patterns {
        if count >= min_support {
            itemsets.push(vec![format!("value:{}", pattern)]);
        }
    }

    itemsets
}

fn generate_candidate_itemsets(frequent_itemsets: &[Vec<String>], k: usize) -> Vec<Vec<String>> {
    let mut candidates = Vec::new();

    for i in 0..frequent_itemsets.len() {
        for j in (i + 1)..frequent_itemsets.len() {
            let itemset1 = &frequent_itemsets[i];
            let itemset2 = &frequent_itemsets[j];

            // Check if first k-2 items are the same
            if k > 2 && itemset1[..k-2] != itemset2[..k-2] {
                continue;
            }

            // Merge itemsets
            let mut candidate = itemset1.clone();
            for item in itemset2 {
                if !candidate.contains(item) {
                    candidate.push(item.clone());
                }
            }

            if candidate.len() == k {
                candidate.sort();
                if !candidates.contains(&candidate) {
                    candidates.push(candidate);
                }
            }
        }
    }

    candidates
}

fn calculate_itemset_support(itemset: &[String], frequency_tables: &FrequencyTables) -> usize {
    // Simplified support calculation - in practice this would be more sophisticated
    itemset.iter().map(|item| {
        if item.starts_with("prop:") {
            frequency_tables.properties.get(&item[5..]).copied().unwrap_or(0)
        } else if item.starts_with("class:") {
            frequency_tables.classes.get(&item[6..]).copied().unwrap_or(0)
        } else if item.starts_with("value:") {
            frequency_tables.value_patterns.get(&item[6..]).copied().unwrap_or(0)
        } else {
            0
        }
    }).min().unwrap_or(0)
}

fn generate_combinations(items: &[String], size: usize) -> Vec<Vec<String>> {
    if size == 0 || size > items.len() {
        return vec![];
    }
    
    if size == 1 {
        return items.iter().map(|item| vec![item.clone()]).collect();
    }

    let mut combinations = Vec::new();
    for i in 0..=items.len() - size {
        let first = items[i].clone();
        let remaining = &items[i + 1..];
        
        for mut combo in generate_combinations(remaining, size - 1) {
            combo.insert(0, first.clone());
            combinations.push(combo);
        }
    }

    combinations
}

fn create_pattern_items(antecedent: &[String], consequent: &[String]) -> Vec<PatternItem> {
    let mut items = Vec::new();

    for item in antecedent {
        items.push(PatternItem {
            item_type: classify_item_type(item),
            identifier: item.clone(),
            role: ItemRole::Predicate,
            frequency: 1.0, // Simplified - would calculate actual frequency
        });
    }

    for item in consequent {
        items.push(PatternItem {
            item_type: classify_item_type(item),
            identifier: item.clone(),
            role: ItemRole::Object,
            frequency: 1.0,
        });
    }

    items
}

fn classify_item_type(item: &str) -> PatternItemType {
    if item.starts_with("prop:") {
        PatternItemType::Property
    } else if item.starts_with("class:") {
        PatternItemType::Class
    } else if item.starts_with("value:") {
        PatternItemType::ValuePattern
    } else {
        PatternItemType::Property // Default
    }
}

fn classify_pattern_type(antecedent: &[String], consequent: &[String]) -> PatternType {
    let all_items: Vec<_> = antecedent.iter().chain(consequent.iter()).collect();
    
    let has_properties = all_items.iter().any(|item| item.starts_with("prop:"));
    let has_classes = all_items.iter().any(|item| item.starts_with("class:"));
    let has_values = all_items.iter().any(|item| item.starts_with("value:"));

    match (has_properties, has_classes, has_values) {
        (true, false, false) => PatternType::Structural,
        (false, false, true) => PatternType::Value,
        (false, true, false) => PatternType::Structural,
        _ => PatternType::Mixed,
    }
}

fn calculate_quality_score(support: f64, confidence: f64, lift: f64, conviction: f64) -> f64 {
    // Weighted combination of metrics
    let support_weight = 0.2;
    let confidence_weight = 0.4;
    let lift_weight = 0.3;
    let conviction_weight = 0.1;

    let normalized_lift = (lift - 1.0).max(0.0).min(2.0) / 2.0;
    let normalized_conviction = conviction.min(10.0) / 10.0;

    support * support_weight
        + confidence * confidence_weight
        + normalized_lift * lift_weight
        + normalized_conviction * conviction_weight
}

fn analyze_temporal_characteristics(
    pattern: &AdvancedPattern,
    store: &dyn Store,
    graph_name: Option<&str>,
) -> Result<Option<TemporalPatternInfo>> {
    // Simplified temporal analysis - would be more sophisticated in practice
    Ok(Some(TemporalPatternInfo {
        frequency: 1.0,
        seasonality: vec![],
        trend: TrendDirection::Stable,
        stability_score: 0.8,
    }))
}

fn calculate_hierarchy_level(pattern: &AdvancedPattern, frequency_tables: &FrequencyTables) -> usize {
    // Simplified hierarchy calculation
    pattern.items.len().saturating_sub(2)
}

fn create_constraint_suggestions(
    pattern: &AdvancedPattern,
    config: &AdvancedPatternMiningConfig,
) -> Vec<SuggestedConstraint> {
    let mut suggestions = Vec::new();

    // Generate suggestions based on pattern characteristics
    for item in &pattern.items {
        if item.item_type == PatternItemType::Property {
            suggestions.push(SuggestedConstraint {
                constraint_type: ConstraintType::MinCount,
                path: item.identifier.clone(),
                parameters: [("minCount".to_string(), "1".to_string())].into_iter().collect(),
                confidence: pattern.confidence,
                coverage: pattern.support_ratio,
                severity: if pattern.quality_score > 0.9 {
                    oxirs_shacl::Severity::Violation
                } else {
                    oxirs_shacl::Severity::Warning
                },
            });
        }
    }

    suggestions
}