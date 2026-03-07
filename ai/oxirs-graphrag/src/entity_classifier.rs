//! Entity type classification for knowledge graph nodes.
//!
//! Uses rule-based heuristics (pattern matching, suffix detection, numeric checks)
//! plus user-defined classification rules to assign entity types and confidence scores.

/// The set of recognised entity types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Event,
    Product,
    Concept,
    Date,
    Number,
    Unknown,
}

impl EntityType {
    /// Return a human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            EntityType::Person => "Person",
            EntityType::Organization => "Organization",
            EntityType::Location => "Location",
            EntityType::Event => "Event",
            EntityType::Product => "Product",
            EntityType::Concept => "Concept",
            EntityType::Date => "Date",
            EntityType::Number => "Number",
            EntityType::Unknown => "Unknown",
        }
    }
}

/// A single named feature with a numeric value, used to explain classification.
#[derive(Debug, Clone)]
pub struct ClassificationFeature {
    pub name: String,
    pub value: f64,
}

/// The result of classifying a single entity text string.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub entity_text: String,
    pub predicted_type: EntityType,
    pub confidence: f64,
    pub features: Vec<ClassificationFeature>,
}

/// A user-defined rule: if `pattern` is found in the entity text (case-insensitive),
/// score `entity_type` with an additional `confidence_boost`.
#[derive(Debug, Clone)]
pub struct ClassificationRule {
    pub pattern: String,
    pub entity_type: EntityType,
    pub confidence_boost: f64,
}

/// Month name constants used for Date detection.
const MONTH_NAMES: &[&str] = &[
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
];

/// Common location suffixes.
const LOCATION_SUFFIXES: &[&str] = &[
    "city", "river", "mountain", "street", "avenue", "lake", "island", "valley",
];

/// Common organisation suffixes.
const ORG_SUFFIXES: &[&str] = &["inc", "corp", "ltd", "gmbh", "llc", "plc", "ag", "bv", "sa"];

/// Base confidence for any match — additional boosts are applied on top.
const BASE_CONFIDENCE: f64 = 0.5;

/// Entity classifier using heuristic rules and optional user-defined rules.
pub struct EntityClassifier {
    rules: Vec<ClassificationRule>,
}

impl EntityClassifier {
    /// Create a new classifier with no user rules.
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a user-defined classification rule.
    pub fn add_rule(&mut self, rule: ClassificationRule) {
        self.rules.push(rule);
    }

    /// Return the number of user-defined rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Classify a single entity text string.
    pub fn classify(&self, text: &str) -> ClassificationResult {
        let lower = text.to_lowercase();
        let mut features: Vec<ClassificationFeature> = Vec::new();

        // Collect (EntityType, confidence) candidates from all heuristics
        let mut candidates: Vec<(EntityType, f64)> = Vec::new();

        // 1. Pure numeric check → Number
        if text
            .trim()
            .chars()
            .all(|c| c.is_ascii_digit() || c == '.' || c == '-')
            && !text.trim().is_empty()
        {
            features.push(ClassificationFeature {
                name: "is_numeric".to_string(),
                value: 1.0,
            });
            candidates.push((EntityType::Number, BASE_CONFIDENCE + 0.4));
        }

        // 2. Date detection: contains digits AND a month name
        let has_digits = text.chars().any(|c| c.is_ascii_digit());
        let has_month = MONTH_NAMES.iter().any(|&m| lower.contains(m));
        if has_digits && has_month {
            features.push(ClassificationFeature {
                name: "has_month_name".to_string(),
                value: 1.0,
            });
            candidates.push((EntityType::Date, BASE_CONFIDENCE + 0.35));
        }

        // 3. Location suffix
        if let Some(suffix) = LOCATION_SUFFIXES.iter().find(|&&s| lower.ends_with(s)) {
            features.push(ClassificationFeature {
                name: format!("location_suffix:{suffix}"),
                value: 1.0,
            });
            candidates.push((EntityType::Location, BASE_CONFIDENCE + 0.3));
        }

        // 4. Organisation suffix (check last word)
        let last_word_lower = lower
            .split_whitespace()
            .last()
            .unwrap_or("")
            .trim_end_matches('.');
        if ORG_SUFFIXES.contains(&last_word_lower) {
            features.push(ClassificationFeature {
                name: format!("org_suffix:{last_word_lower}"),
                value: 1.0,
            });
            candidates.push((EntityType::Organization, BASE_CONFIDENCE + 0.35));
        }

        // 5. Starts with uppercase + no spaces + ≤ 20 chars → possible Person/Org
        let starts_upper = text
            .chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false);
        let no_spaces = !text.contains(' ');
        let short = text.len() <= 20;
        if starts_upper && no_spaces && short {
            features.push(ClassificationFeature {
                name: "capitalized_single_token".to_string(),
                value: 1.0,
            });
            // Slight lean towards Person if it looks like a name; otherwise Concept
            candidates.push((EntityType::Person, BASE_CONFIDENCE + 0.1));
        }

        // 6. User-defined rules
        for rule in &self.rules {
            let pattern_lower = rule.pattern.to_lowercase();
            if lower.contains(&pattern_lower) {
                features.push(ClassificationFeature {
                    name: format!("rule_match:{}", rule.pattern),
                    value: rule.confidence_boost,
                });
                let conf = (BASE_CONFIDENCE + rule.confidence_boost).clamp(0.0, 1.0);
                candidates.push((rule.entity_type.clone(), conf));
            }
        }

        // Choose the candidate with the highest confidence
        let (predicted_type, confidence) = candidates
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((EntityType::Unknown, BASE_CONFIDENCE));

        ClassificationResult {
            entity_text: text.to_string(),
            predicted_type,
            confidence: confidence.clamp(0.0, 1.0),
            features,
        }
    }

    /// Classify a batch of entity text strings.
    pub fn classify_batch(&self, texts: &[&str]) -> Vec<ClassificationResult> {
        texts.iter().map(|&t| self.classify(t)).collect()
    }
}

impl Default for EntityClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classifier() -> EntityClassifier {
        EntityClassifier::new()
    }

    // --- Entity type label ---

    #[test]
    fn test_entity_type_labels() {
        assert_eq!(EntityType::Person.label(), "Person");
        assert_eq!(EntityType::Organization.label(), "Organization");
        assert_eq!(EntityType::Location.label(), "Location");
        assert_eq!(EntityType::Date.label(), "Date");
        assert_eq!(EntityType::Number.label(), "Number");
        assert_eq!(EntityType::Unknown.label(), "Unknown");
    }

    // --- Number detection ---

    #[test]
    fn test_classify_integer() {
        let c = classifier();
        let r = c.classify("42");
        assert_eq!(r.predicted_type, EntityType::Number);
    }

    #[test]
    fn test_classify_float() {
        let c = classifier();
        let r = c.classify("3.14");
        assert_eq!(r.predicted_type, EntityType::Number);
    }

    #[test]
    fn test_classify_negative_number() {
        let c = classifier();
        let r = c.classify("-7");
        assert_eq!(r.predicted_type, EntityType::Number);
    }

    // --- Date detection ---

    #[test]
    fn test_classify_date_with_month_name() {
        let c = classifier();
        let r = c.classify("January 2024");
        assert_eq!(r.predicted_type, EntityType::Date);
    }

    #[test]
    fn test_classify_date_abbreviated_month() {
        let c = classifier();
        let r = c.classify("15 Mar 2025");
        assert_eq!(r.predicted_type, EntityType::Date);
    }

    // --- Location detection ---

    #[test]
    fn test_classify_location_city() {
        let c = classifier();
        let r = c.classify("New York City");
        // ends with "city"
        assert_eq!(r.predicted_type, EntityType::Location);
    }

    #[test]
    fn test_classify_location_river() {
        let c = classifier();
        let r = c.classify("Amazon River");
        assert_eq!(r.predicted_type, EntityType::Location);
    }

    #[test]
    fn test_classify_location_mountain() {
        let c = classifier();
        let r = c.classify("Mount Everest Mountain");
        assert_eq!(r.predicted_type, EntityType::Location);
    }

    #[test]
    fn test_classify_location_street() {
        let c = classifier();
        let r = c.classify("Baker Street");
        assert_eq!(r.predicted_type, EntityType::Location);
    }

    // --- Organisation detection ---

    #[test]
    fn test_classify_org_inc() {
        let c = classifier();
        let r = c.classify("Acme Corp");
        assert_eq!(r.predicted_type, EntityType::Organization);
    }

    #[test]
    fn test_classify_org_ltd() {
        let c = classifier();
        let r = c.classify("Widgets Ltd");
        assert_eq!(r.predicted_type, EntityType::Organization);
    }

    #[test]
    fn test_classify_org_gmbh() {
        let c = classifier();
        let r = c.classify("Muller GmbH");
        assert_eq!(r.predicted_type, EntityType::Organization);
    }

    // --- Person detection (capitalised single token) ---

    #[test]
    fn test_classify_person_single_capitalized() {
        let c = classifier();
        let r = c.classify("Alice");
        // Should be Person (capitalized single short token)
        assert_eq!(r.predicted_type, EntityType::Person);
    }

    #[test]
    fn test_classify_person_confidence_positive() {
        let c = classifier();
        let r = c.classify("Bob");
        assert!(r.confidence > 0.0);
    }

    // --- Unknown ---

    #[test]
    fn test_classify_unknown_generic_phrase() {
        let c = classifier();
        let r = c.classify("the semantic web is interesting");
        // None of the heuristics fire strongly; falls to Unknown
        // (no uppercase start, no number, no month+digits, no suffix)
        let _ = r; // just ensure no panic
    }

    // --- Confidence bounds ---

    #[test]
    fn test_confidence_always_in_range() {
        let c = classifier();
        let texts = [
            "Alice",
            "42",
            "January 2024",
            "Acme Corp",
            "Baker Street",
            "foo",
            "",
        ];
        for text in &texts {
            let r = c.classify(text);
            assert!(
                r.confidence >= 0.0 && r.confidence <= 1.0,
                "Confidence out of range for '{text}': {}",
                r.confidence
            );
        }
    }

    // --- Features populated ---

    #[test]
    fn test_features_populated_for_number() {
        let c = classifier();
        let r = c.classify("100");
        assert!(!r.features.is_empty());
    }

    // --- Custom rule tests ---

    #[test]
    fn test_add_custom_rule_count() {
        let mut c = classifier();
        assert_eq!(c.rule_count(), 0);
        c.add_rule(ClassificationRule {
            pattern: "summit".to_string(),
            entity_type: EntityType::Event,
            confidence_boost: 0.4,
        });
        assert_eq!(c.rule_count(), 1);
    }

    #[test]
    fn test_custom_rule_fires() {
        let mut c = classifier();
        c.add_rule(ClassificationRule {
            pattern: "summit".to_string(),
            entity_type: EntityType::Event,
            confidence_boost: 0.4,
        });
        let r = c.classify("G7 Summit 2025");
        assert_eq!(r.predicted_type, EntityType::Event);
    }

    #[test]
    fn test_custom_rule_confidence_boosted() {
        let mut c = classifier();
        c.add_rule(ClassificationRule {
            pattern: "widget".to_string(),
            entity_type: EntityType::Product,
            confidence_boost: 0.3,
        });
        let r = c.classify("Super Widget Pro");
        assert!(r.confidence >= BASE_CONFIDENCE + 0.3 - 1e-9);
    }

    #[test]
    fn test_custom_rule_case_insensitive() {
        let mut c = classifier();
        c.add_rule(ClassificationRule {
            pattern: "WIDGET".to_string(),
            entity_type: EntityType::Product,
            confidence_boost: 0.2,
        });
        let r = c.classify("widget maker");
        assert_eq!(r.predicted_type, EntityType::Product);
    }

    #[test]
    fn test_multiple_custom_rules_highest_wins() {
        let mut c = classifier();
        c.add_rule(ClassificationRule {
            pattern: "demo".to_string(),
            entity_type: EntityType::Event,
            confidence_boost: 0.2,
        });
        c.add_rule(ClassificationRule {
            pattern: "demo".to_string(),
            entity_type: EntityType::Concept,
            confidence_boost: 0.45,
        });
        let r = c.classify("demo system");
        // Concept has higher boost
        assert_eq!(r.predicted_type, EntityType::Concept);
    }

    // --- Batch processing ---

    #[test]
    fn test_classify_batch_count() {
        let c = classifier();
        let texts = ["Alice", "Acme Corp", "42", "Baker Street"];
        let results = c.classify_batch(&texts);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_classify_batch_empty() {
        let c = classifier();
        let results = c.classify_batch(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_classify_batch_single() {
        let c = classifier();
        let results = c.classify_batch(&["100"]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].predicted_type, EntityType::Number);
    }

    // --- Edge cases ---

    #[test]
    fn test_classify_empty_string() {
        let c = classifier();
        let r = c.classify("");
        // Empty string: no heuristics fire → Unknown
        let _ = r.predicted_type; // just ensure no panic
    }

    #[test]
    fn test_classify_whitespace_only() {
        let c = classifier();
        let r = c.classify("   ");
        let _ = r;
    }

    #[test]
    fn test_default_classifier() {
        let c = EntityClassifier::default();
        assert_eq!(c.rule_count(), 0);
    }
}
