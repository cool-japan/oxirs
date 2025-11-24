//! Constraint Recommendations Engine
//!
//! Provides intelligent recommendations for SHACL constraints based on
//! property names, domains, and common patterns.

use super::{ConstraintSpec, Domain, PropertyDesign, PropertyHint};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Recommendation engine for SHACL constraints
#[derive(Debug)]
pub struct RecommendationEngine {
    /// Property name patterns
    property_patterns: Vec<PropertyPattern>,
    /// Domain-specific rules
    domain_rules: HashMap<Domain, Vec<DomainRule>>,
    /// Constraint templates
    constraint_templates: HashMap<String, ConstraintTemplate>,
}

/// Property name pattern for matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyPattern {
    /// Pattern name
    pub name: String,
    /// Keywords to match (case-insensitive)
    pub keywords: Vec<String>,
    /// Suffix patterns (e.g., "Date", "Time")
    pub suffixes: Vec<String>,
    /// Prefix patterns (e.g., "has", "is")
    pub prefixes: Vec<String>,
    /// Recommended hints
    pub hints: Vec<PropertyHint>,
    /// Recommended constraints
    pub constraints: Vec<ConstraintSpec>,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
}

/// Domain-specific rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainRule {
    /// Rule name
    pub name: String,
    /// Description
    pub description: String,
    /// Property patterns to apply
    pub patterns: Vec<String>,
    /// Additional hints
    pub hints: Vec<PropertyHint>,
    /// Priority (higher = more important)
    pub priority: u8,
}

/// Constraint template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintTemplate {
    /// Template name
    pub name: String,
    /// Description
    pub description: String,
    /// Constraints to apply
    pub constraints: Vec<ConstraintSpec>,
    /// Example usage
    pub example: String,
}

/// Recommendation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Recommendation type
    pub rec_type: RecommendationType,
    /// Recommended hints
    pub hints: Vec<PropertyHint>,
    /// Recommended constraints
    pub constraints: Vec<ConstraintSpec>,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Explanation
    pub explanation: String,
    /// Source pattern/rule name
    pub source: String,
}

/// Recommendation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Based on property name
    PropertyName,
    /// Based on domain
    Domain,
    /// Based on common patterns
    CommonPattern,
    /// Based on similar properties
    SimilarProperty,
    /// Best practice
    BestPractice,
}

impl RecommendationEngine {
    /// Create a new recommendation engine with default rules
    pub fn new() -> Self {
        let mut engine = Self {
            property_patterns: Vec::new(),
            domain_rules: HashMap::new(),
            constraint_templates: HashMap::new(),
        };

        engine.load_default_patterns();
        engine.load_domain_rules();
        engine.load_constraint_templates();

        engine
    }

    fn load_default_patterns(&mut self) {
        // Name patterns
        self.property_patterns.push(PropertyPattern {
            name: "name".to_string(),
            keywords: vec!["name".to_string(), "label".to_string(), "title".to_string()],
            suffixes: vec!["Name".to_string(), "Label".to_string(), "Title".to_string()],
            prefixes: vec![],
            hints: vec![PropertyHint::Required, PropertyHint::String],
            constraints: vec![
                ConstraintSpec::MinCount(1),
                ConstraintSpec::Datatype("http://www.w3.org/2001/XMLSchema#string".to_string()),
            ],
            confidence: 0.9,
        });

        // Email patterns
        self.property_patterns.push(PropertyPattern {
            name: "email".to_string(),
            keywords: vec!["email".to_string(), "mail".to_string(), "mbox".to_string()],
            suffixes: vec!["Email".to_string(), "Mail".to_string()],
            prefixes: vec![],
            hints: vec![PropertyHint::Email, PropertyHint::String],
            constraints: vec![ConstraintSpec::Pattern(
                r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$".to_string(),
            )],
            confidence: 0.95,
        });

        // Phone patterns
        self.property_patterns.push(PropertyPattern {
            name: "phone".to_string(),
            keywords: vec![
                "phone".to_string(),
                "telephone".to_string(),
                "mobile".to_string(),
                "cell".to_string(),
                "fax".to_string(),
            ],
            suffixes: vec!["Phone".to_string(), "Tel".to_string()],
            prefixes: vec![],
            hints: vec![PropertyHint::Phone, PropertyHint::String],
            constraints: vec![ConstraintSpec::Pattern(r"^\+?[1-9]\d{1,14}$".to_string())],
            confidence: 0.85,
        });

        // URL patterns
        self.property_patterns.push(PropertyPattern {
            name: "url".to_string(),
            keywords: vec![
                "url".to_string(),
                "uri".to_string(),
                "link".to_string(),
                "homepage".to_string(),
                "website".to_string(),
            ],
            suffixes: vec!["Url".to_string(), "Uri".to_string(), "Link".to_string()],
            prefixes: vec![],
            hints: vec![PropertyHint::URL, PropertyHint::IRI],
            constraints: vec![ConstraintSpec::Pattern(
                r"^https?://[^\s/$.?#].[^\s]*$".to_string(),
            )],
            confidence: 0.9,
        });

        // Date patterns
        self.property_patterns.push(PropertyPattern {
            name: "date".to_string(),
            keywords: vec!["date".to_string(), "day".to_string()],
            suffixes: vec![
                "Date".to_string(),
                "Day".to_string(),
                "At".to_string(),
                "On".to_string(),
            ],
            prefixes: vec![],
            hints: vec![PropertyHint::Date],
            constraints: vec![ConstraintSpec::Datatype(
                "http://www.w3.org/2001/XMLSchema#date".to_string(),
            )],
            confidence: 0.85,
        });

        // DateTime patterns
        self.property_patterns.push(PropertyPattern {
            name: "datetime".to_string(),
            keywords: vec!["datetime".to_string(), "timestamp".to_string()],
            suffixes: vec!["Time".to_string(), "Timestamp".to_string()],
            prefixes: vec![
                "created".to_string(),
                "modified".to_string(),
                "updated".to_string(),
            ],
            hints: vec![PropertyHint::DateTime],
            constraints: vec![ConstraintSpec::Datatype(
                "http://www.w3.org/2001/XMLSchema#dateTime".to_string(),
            )],
            confidence: 0.85,
        });

        // Age/count patterns
        self.property_patterns.push(PropertyPattern {
            name: "age".to_string(),
            keywords: vec!["age".to_string(), "count".to_string(), "number".to_string()],
            suffixes: vec!["Age".to_string(), "Count".to_string(), "Number".to_string()],
            prefixes: vec![],
            hints: vec![PropertyHint::Integer],
            constraints: vec![
                ConstraintSpec::Datatype("http://www.w3.org/2001/XMLSchema#integer".to_string()),
                ConstraintSpec::MinInclusive(0.0),
            ],
            confidence: 0.8,
        });

        // Price/amount patterns
        self.property_patterns.push(PropertyPattern {
            name: "price".to_string(),
            keywords: vec![
                "price".to_string(),
                "cost".to_string(),
                "amount".to_string(),
                "total".to_string(),
                "fee".to_string(),
            ],
            suffixes: vec![
                "Price".to_string(),
                "Cost".to_string(),
                "Amount".to_string(),
            ],
            prefixes: vec![],
            hints: vec![PropertyHint::Decimal],
            constraints: vec![
                ConstraintSpec::Datatype("http://www.w3.org/2001/XMLSchema#decimal".to_string()),
                ConstraintSpec::MinInclusive(0.0),
            ],
            confidence: 0.85,
        });

        // ID patterns
        self.property_patterns.push(PropertyPattern {
            name: "id".to_string(),
            keywords: vec![
                "id".to_string(),
                "identifier".to_string(),
                "code".to_string(),
                "sku".to_string(),
            ],
            suffixes: vec!["Id".to_string(), "ID".to_string(), "Code".to_string()],
            prefixes: vec![],
            hints: vec![
                PropertyHint::Required,
                PropertyHint::Unique,
                PropertyHint::String,
            ],
            constraints: vec![ConstraintSpec::MinCount(1), ConstraintSpec::MaxCount(1)],
            confidence: 0.85,
        });

        // Description patterns
        self.property_patterns.push(PropertyPattern {
            name: "description".to_string(),
            keywords: vec![
                "description".to_string(),
                "desc".to_string(),
                "comment".to_string(),
                "note".to_string(),
                "bio".to_string(),
            ],
            suffixes: vec![
                "Description".to_string(),
                "Comment".to_string(),
                "Note".to_string(),
            ],
            prefixes: vec![],
            hints: vec![PropertyHint::String, PropertyHint::Optional],
            constraints: vec![ConstraintSpec::Datatype(
                "http://www.w3.org/2001/XMLSchema#string".to_string(),
            )],
            confidence: 0.8,
        });

        // Boolean patterns
        self.property_patterns.push(PropertyPattern {
            name: "boolean".to_string(),
            keywords: vec![
                "active".to_string(),
                "enabled".to_string(),
                "valid".to_string(),
                "verified".to_string(),
                "approved".to_string(),
            ],
            suffixes: vec!["Flag".to_string()],
            prefixes: vec![
                "is".to_string(),
                "has".to_string(),
                "can".to_string(),
                "should".to_string(),
            ],
            hints: vec![PropertyHint::Boolean],
            constraints: vec![ConstraintSpec::Datatype(
                "http://www.w3.org/2001/XMLSchema#boolean".to_string(),
            )],
            confidence: 0.9,
        });
    }

    fn load_domain_rules(&mut self) {
        // Identity domain
        self.domain_rules.insert(
            Domain::Identity,
            vec![
                DomainRule {
                    name: "person_name".to_string(),
                    description: "Person names should be required strings".to_string(),
                    patterns: vec!["name".to_string()],
                    hints: vec![PropertyHint::Required],
                    priority: 10,
                },
                DomainRule {
                    name: "birth_date".to_string(),
                    description: "Birth dates should use xsd:date".to_string(),
                    patterns: vec!["date".to_string()],
                    hints: vec![PropertyHint::Date],
                    priority: 8,
                },
            ],
        );

        // Commerce domain
        self.domain_rules.insert(
            Domain::Commerce,
            vec![
                DomainRule {
                    name: "product_sku".to_string(),
                    description: "SKUs should be unique identifiers".to_string(),
                    patterns: vec!["id".to_string()],
                    hints: vec![PropertyHint::Required, PropertyHint::Unique],
                    priority: 10,
                },
                DomainRule {
                    name: "product_price".to_string(),
                    description: "Prices should be non-negative decimals".to_string(),
                    patterns: vec!["price".to_string()],
                    hints: vec![PropertyHint::Required],
                    priority: 9,
                },
            ],
        );

        // Contact domain
        self.domain_rules.insert(
            Domain::Contact,
            vec![
                DomainRule {
                    name: "email_validation".to_string(),
                    description: "Emails should have pattern validation".to_string(),
                    patterns: vec!["email".to_string()],
                    hints: vec![PropertyHint::Email],
                    priority: 10,
                },
                DomainRule {
                    name: "phone_validation".to_string(),
                    description: "Phone numbers should have pattern validation".to_string(),
                    patterns: vec!["phone".to_string()],
                    hints: vec![PropertyHint::Phone],
                    priority: 9,
                },
            ],
        );

        // Scientific domain
        self.domain_rules.insert(
            Domain::Scientific,
            vec![DomainRule {
                name: "measurement_precision".to_string(),
                description: "Measurements should use decimal datatype".to_string(),
                patterns: vec!["price".to_string()], // Reuse decimal pattern
                hints: vec![PropertyHint::Decimal],
                priority: 8,
            }],
        );
    }

    fn load_constraint_templates(&mut self) {
        self.constraint_templates.insert(
            "required_string".to_string(),
            ConstraintTemplate {
                name: "Required String".to_string(),
                description: "A required string property with minimum length".to_string(),
                constraints: vec![
                    ConstraintSpec::MinCount(1),
                    ConstraintSpec::Datatype("http://www.w3.org/2001/XMLSchema#string".to_string()),
                    ConstraintSpec::MinLength(1),
                ],
                example: "foaf:name".to_string(),
            },
        );

        self.constraint_templates.insert(
            "optional_string".to_string(),
            ConstraintTemplate {
                name: "Optional String".to_string(),
                description: "An optional string property".to_string(),
                constraints: vec![
                    ConstraintSpec::MaxCount(1),
                    ConstraintSpec::Datatype("http://www.w3.org/2001/XMLSchema#string".to_string()),
                ],
                example: "schema:description".to_string(),
            },
        );

        self.constraint_templates.insert(
            "unique_id".to_string(),
            ConstraintTemplate {
                name: "Unique Identifier".to_string(),
                description: "A required, unique identifier".to_string(),
                constraints: vec![ConstraintSpec::MinCount(1), ConstraintSpec::MaxCount(1)],
                example: "schema:identifier".to_string(),
            },
        );

        self.constraint_templates.insert(
            "email".to_string(),
            ConstraintTemplate {
                name: "Email Address".to_string(),
                description: "Email with pattern validation".to_string(),
                constraints: vec![ConstraintSpec::Pattern(
                    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$".to_string(),
                )],
                example: "foaf:mbox".to_string(),
            },
        );
    }

    /// Get recommendations for a property name
    pub fn recommend_for_property(&self, property_name: &str) -> Vec<PropertyHint> {
        let recommendations = self.get_recommendations(property_name, None);
        recommendations.into_iter().flat_map(|r| r.hints).collect()
    }

    /// Get full recommendations for a property
    pub fn get_recommendations(
        &self,
        property_name: &str,
        domain: Option<Domain>,
    ) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        // Extract local name from prefixed or full IRI
        let local_name = self.extract_local_name(property_name);
        let normalized = local_name.to_lowercase();

        // Check property patterns
        for pattern in &self.property_patterns {
            let mut confidence = 0.0f32;
            let mut matched = false;

            // Check keywords
            for keyword in &pattern.keywords {
                if normalized.contains(keyword) {
                    confidence = confidence.max(pattern.confidence);
                    matched = true;
                }
            }

            // Check suffixes
            for suffix in &pattern.suffixes {
                if local_name.ends_with(suffix) {
                    confidence = confidence.max(pattern.confidence * 0.95);
                    matched = true;
                }
            }

            // Check prefixes
            for prefix in &pattern.prefixes {
                if normalized.starts_with(prefix) {
                    confidence = confidence.max(pattern.confidence * 0.9);
                    matched = true;
                }
            }

            if matched {
                recommendations.push(Recommendation {
                    rec_type: RecommendationType::PropertyName,
                    hints: pattern.hints.clone(),
                    constraints: pattern.constraints.clone(),
                    confidence,
                    explanation: format!(
                        "Property name '{}' matches pattern '{}'",
                        local_name, pattern.name
                    ),
                    source: pattern.name.clone(),
                });
            }
        }

        // Apply domain rules
        if let Some(domain) = domain {
            if let Some(rules) = self.domain_rules.get(&domain) {
                for rule in rules {
                    for pattern_name in &rule.patterns {
                        if recommendations.iter().any(|r| &r.source == pattern_name) {
                            recommendations.push(Recommendation {
                                rec_type: RecommendationType::Domain,
                                hints: rule.hints.clone(),
                                constraints: vec![],
                                confidence: 0.8,
                                explanation: rule.description.clone(),
                                source: rule.name.clone(),
                            });
                        }
                    }
                }
            }
        }

        // Sort by confidence
        recommendations.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        recommendations
    }

    /// Extract local name from IRI or prefixed name
    fn extract_local_name<'a>(&self, name: &'a str) -> &'a str {
        // Handle prefixed names (e.g., foaf:name)
        if let Some(pos) = name.rfind(':') {
            return &name[pos + 1..];
        }

        // Handle full IRIs (e.g., http://xmlns.com/foaf/0.1/name)
        if let Some(pos) = name.rfind('#') {
            return &name[pos + 1..];
        }

        if let Some(pos) = name.rfind('/') {
            return &name[pos + 1..];
        }

        name
    }

    /// Get constraint template by name
    pub fn get_template(&self, name: &str) -> Option<&ConstraintTemplate> {
        self.constraint_templates.get(name)
    }

    /// List all template names
    pub fn list_templates(&self) -> Vec<String> {
        self.constraint_templates.keys().cloned().collect()
    }

    /// Apply recommendations to a property design
    pub fn apply_recommendations(&self, property: &mut PropertyDesign, domain: Option<Domain>) {
        let recommendations = self.get_recommendations(&property.path, domain);

        for rec in recommendations {
            if rec.confidence >= 0.7 {
                for hint in rec.hints {
                    property.hints.insert(hint);
                }
                for constraint in rec.constraints {
                    if !property
                        .constraints
                        .iter()
                        .any(|c| std::mem::discriminant(c) == std::mem::discriminant(&constraint))
                    {
                        property.constraints.push(constraint);
                    }
                }
            }
        }
    }

    /// Suggest common property names for a domain
    pub fn suggest_properties(&self, domain: Domain) -> Vec<SuggestedProperty> {
        match domain {
            Domain::Identity => vec![
                SuggestedProperty {
                    path: "foaf:name".to_string(),
                    label: "Name".to_string(),
                    description: "Person's full name".to_string(),
                    hints: vec![PropertyHint::Required, PropertyHint::String],
                },
                SuggestedProperty {
                    path: "foaf:mbox".to_string(),
                    label: "Email".to_string(),
                    description: "Email address".to_string(),
                    hints: vec![PropertyHint::Email],
                },
                SuggestedProperty {
                    path: "foaf:birthday".to_string(),
                    label: "Birthday".to_string(),
                    description: "Birth date".to_string(),
                    hints: vec![PropertyHint::Date],
                },
            ],
            Domain::Commerce => vec![
                SuggestedProperty {
                    path: "schema:name".to_string(),
                    label: "Product Name".to_string(),
                    description: "Product name".to_string(),
                    hints: vec![PropertyHint::Required, PropertyHint::String],
                },
                SuggestedProperty {
                    path: "schema:sku".to_string(),
                    label: "SKU".to_string(),
                    description: "Stock keeping unit".to_string(),
                    hints: vec![PropertyHint::Required, PropertyHint::Unique],
                },
                SuggestedProperty {
                    path: "schema:price".to_string(),
                    label: "Price".to_string(),
                    description: "Product price".to_string(),
                    hints: vec![PropertyHint::Required, PropertyHint::Decimal],
                },
            ],
            Domain::Contact => vec![
                SuggestedProperty {
                    path: "schema:email".to_string(),
                    label: "Email".to_string(),
                    description: "Email address".to_string(),
                    hints: vec![PropertyHint::Email],
                },
                SuggestedProperty {
                    path: "schema:telephone".to_string(),
                    label: "Phone".to_string(),
                    description: "Phone number".to_string(),
                    hints: vec![PropertyHint::Phone],
                },
                SuggestedProperty {
                    path: "schema:address".to_string(),
                    label: "Address".to_string(),
                    description: "Postal address".to_string(),
                    hints: vec![PropertyHint::Reference],
                },
            ],
            _ => vec![],
        }
    }
}

impl Default for RecommendationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Suggested property for a domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedProperty {
    /// Property path
    pub path: String,
    /// Human-readable label
    pub label: String,
    /// Description
    pub description: String,
    /// Recommended hints
    pub hints: Vec<PropertyHint>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recommendation_engine() {
        let engine = RecommendationEngine::new();

        let hints = engine.recommend_for_property("foaf:name");
        assert!(hints.contains(&PropertyHint::String));
        assert!(hints.contains(&PropertyHint::Required));
    }

    #[test]
    fn test_email_recommendation() {
        let engine = RecommendationEngine::new();

        let hints = engine.recommend_for_property("email");
        assert!(hints.contains(&PropertyHint::Email));
    }

    #[test]
    fn test_date_recommendation() {
        let engine = RecommendationEngine::new();

        let hints = engine.recommend_for_property("birthDate");
        assert!(hints.contains(&PropertyHint::Date));
    }

    #[test]
    fn test_full_recommendations() {
        let engine = RecommendationEngine::new();

        let recs = engine.get_recommendations("foaf:mbox", Some(Domain::Contact));
        assert!(!recs.is_empty());

        // Should have pattern-based and domain-based recommendations
        let has_property_rec = recs
            .iter()
            .any(|r| r.rec_type == RecommendationType::PropertyName);
        assert!(has_property_rec);
    }

    #[test]
    fn test_apply_recommendations() {
        let engine = RecommendationEngine::new();

        let mut property = PropertyDesign::new("email");
        engine.apply_recommendations(&mut property, Some(Domain::Contact));

        assert!(property.hints.contains(&PropertyHint::Email));
    }

    #[test]
    fn test_suggest_properties() {
        let engine = RecommendationEngine::new();

        let suggestions = engine.suggest_properties(Domain::Identity);
        assert!(!suggestions.is_empty());

        let name_prop = suggestions.iter().find(|p| p.path.contains("name"));
        assert!(name_prop.is_some());
    }

    #[test]
    fn test_templates() {
        let engine = RecommendationEngine::new();

        let templates = engine.list_templates();
        assert!(!templates.is_empty());

        let email_template = engine.get_template("email");
        assert!(email_template.is_some());
    }
}
