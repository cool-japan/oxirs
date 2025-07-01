//! # Comprehensive OxiRS Rule Engine Tutorial
//!
//! This module provides an extensive tutorial covering advanced features, real-world applications,
//! and comprehensive API documentation for the OxiRS Rule Engine.

use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::cache::RuleCache;
use crate::debug::DebuggableRuleEngine;
use crate::{Rule, RuleAtom, RuleEngine, Term};

/// # Advanced Rule Engine Features
///
/// This section demonstrates sophisticated features of the OxiRS Rule Engine including
/// complex reasoning patterns, optimization techniques, and integration scenarios.

pub struct AdvancedFeatures;

impl AdvancedFeatures {
    /// Demonstrate complex reasoning with multiple rule types
    pub fn complex_reasoning_example() -> Result<Vec<RuleAtom>> {
        let mut engine = RuleEngine::new();

        // Transitivity rule for "knows" relationship
        let transitivity_rule = Rule {
            name: "knows_transitivity".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("knows".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("knows".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
                // Prevent reflexivity
                RuleAtom::NotEqual {
                    left: Term::Variable("X".to_string()),
                    right: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("knows".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        };

        // Trust propagation rule
        let trust_rule = Rule {
            name: "trust_propagation".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("trusts".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("recommends".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Z".to_string()),
                    predicate: Term::Constant("type".to_string()),
                    object: Term::Constant("Service".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("trusts".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        };

        // Social influence rule
        let influence_rule = Rule {
            name: "social_influence".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("knows".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("likes".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("influencer".to_string()),
                    object: Term::Constant("true".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("interested_in".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        };

        engine.add_rule(transitivity_rule);
        engine.add_rule(trust_rule);
        engine.add_rule(influence_rule);

        // Social network facts
        let facts = vec![
            // Direct relationships
            RuleAtom::Triple {
                subject: Term::Constant("alice".to_string()),
                predicate: Term::Constant("knows".to_string()),
                object: Term::Constant("bob".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("bob".to_string()),
                predicate: Term::Constant("knows".to_string()),
                object: Term::Constant("charlie".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("charlie".to_string()),
                predicate: Term::Constant("knows".to_string()),
                object: Term::Constant("diana".to_string()),
            },
            // Trust relationships
            RuleAtom::Triple {
                subject: Term::Constant("alice".to_string()),
                predicate: Term::Constant("trusts".to_string()),
                object: Term::Constant("bob".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("bob".to_string()),
                predicate: Term::Constant("recommends".to_string()),
                object: Term::Constant("netflix".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("netflix".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Service".to_string()),
            },
            // Influence
            RuleAtom::Triple {
                subject: Term::Constant("charlie".to_string()),
                predicate: Term::Constant("influencer".to_string()),
                object: Term::Constant("true".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("charlie".to_string()),
                predicate: Term::Constant("likes".to_string()),
                object: Term::Constant("sustainable_fashion".to_string()),
            },
        ];

        engine.forward_chain(&facts)
    }

    /// Demonstrate rule engine with temporal reasoning
    pub fn temporal_reasoning_example() -> Result<Vec<RuleAtom>> {
        let mut engine = RuleEngine::new();

        // Event sequence rule
        let sequence_rule = Rule {
            name: "event_sequence".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("Event1".to_string()),
                    predicate: Term::Constant("precedes".to_string()),
                    object: Term::Variable("Event2".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Event2".to_string()),
                    predicate: Term::Constant("precedes".to_string()),
                    object: Term::Variable("Event3".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Event1".to_string()),
                predicate: Term::Constant("eventually_precedes".to_string()),
                object: Term::Variable("Event3".to_string()),
            }],
        };

        // Temporal constraint rule
        let temporal_constraint_rule = Rule {
            name: "temporal_constraint".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("starts_at".to_string()),
                    object: Term::Variable("StartTime".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("duration".to_string()),
                    object: Term::Variable("Duration".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("ends_at".to_string()),
                object: Term::Function {
                    name: "add".to_string(),
                    args: vec![
                        Term::Variable("StartTime".to_string()),
                        Term::Variable("Duration".to_string()),
                    ],
                },
            }],
        };

        engine.add_rule(sequence_rule);
        engine.add_rule(temporal_constraint_rule);

        let facts = vec![
            // Event timeline
            RuleAtom::Triple {
                subject: Term::Constant("meeting1".to_string()),
                predicate: Term::Constant("precedes".to_string()),
                object: Term::Constant("meeting2".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("meeting2".to_string()),
                predicate: Term::Constant("precedes".to_string()),
                object: Term::Constant("meeting3".to_string()),
            },
            // Timing information
            RuleAtom::Triple {
                subject: Term::Constant("meeting1".to_string()),
                predicate: Term::Constant("starts_at".to_string()),
                object: Term::Constant("09:00".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("meeting1".to_string()),
                predicate: Term::Constant("duration".to_string()),
                object: Term::Constant("60".to_string()),
            },
        ];

        engine.forward_chain(&facts)
    }

    /// Demonstrate probabilistic reasoning with confidence scores
    pub fn probabilistic_reasoning_example() -> Result<Vec<RuleAtom>> {
        let mut engine = RuleEngine::new();

        // High confidence rule
        let high_confidence_rule = Rule {
            name: "weather_prediction_high".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("Location".to_string()),
                    predicate: Term::Constant("humidity".to_string()),
                    object: Term::Constant("high".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Location".to_string()),
                    predicate: Term::Constant("clouds".to_string()),
                    object: Term::Constant("dense".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Location".to_string()),
                    predicate: Term::Constant("wind_speed".to_string()),
                    object: Term::Constant("low".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Location".to_string()),
                predicate: Term::Constant("will_rain".to_string()),
                object: Term::Constant("high_confidence".to_string()),
            }],
        };

        // Medium confidence rule
        let medium_confidence_rule = Rule {
            name: "weather_prediction_medium".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("Location".to_string()),
                    predicate: Term::Constant("humidity".to_string()),
                    object: Term::Constant("medium".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Location".to_string()),
                    predicate: Term::Constant("clouds".to_string()),
                    object: Term::Constant("present".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Location".to_string()),
                predicate: Term::Constant("will_rain".to_string()),
                object: Term::Constant("medium_confidence".to_string()),
            }],
        };

        engine.add_rule(high_confidence_rule);
        engine.add_rule(medium_confidence_rule);

        let facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("seattle".to_string()),
                predicate: Term::Constant("humidity".to_string()),
                object: Term::Constant("high".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("seattle".to_string()),
                predicate: Term::Constant("clouds".to_string()),
                object: Term::Constant("dense".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("seattle".to_string()),
                predicate: Term::Constant("wind_speed".to_string()),
                object: Term::Constant("low".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("phoenix".to_string()),
                predicate: Term::Constant("humidity".to_string()),
                object: Term::Constant("medium".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("phoenix".to_string()),
                predicate: Term::Constant("clouds".to_string()),
                object: Term::Constant("present".to_string()),
            },
        ];

        engine.forward_chain(&facts)
    }
}

/// # Real-World Applications
///
/// This section provides comprehensive examples of applying the rule engine
/// to real-world scenarios and business problems.

pub struct RealWorldApplications;

impl RealWorldApplications {
    /// Healthcare diagnosis support system
    pub fn healthcare_diagnosis_system() -> Result<Vec<RuleAtom>> {
        let mut engine = RuleEngine::new();

        // Symptom-based diagnosis rules
        let flu_diagnosis = Rule {
            name: "flu_diagnosis".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("Patient".to_string()),
                    predicate: Term::Constant("has_symptom".to_string()),
                    object: Term::Constant("fever".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Patient".to_string()),
                    predicate: Term::Constant("has_symptom".to_string()),
                    object: Term::Constant("cough".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Patient".to_string()),
                    predicate: Term::Constant("has_symptom".to_string()),
                    object: Term::Constant("body_aches".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Patient".to_string()),
                predicate: Term::Constant("possible_diagnosis".to_string()),
                object: Term::Constant("influenza".to_string()),
            }],
        };

        let pneumonia_diagnosis = Rule {
            name: "pneumonia_diagnosis".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("Patient".to_string()),
                    predicate: Term::Constant("has_symptom".to_string()),
                    object: Term::Constant("fever".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Patient".to_string()),
                    predicate: Term::Constant("has_symptom".to_string()),
                    object: Term::Constant("chest_pain".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Patient".to_string()),
                    predicate: Term::Constant("has_symptom".to_string()),
                    object: Term::Constant("difficulty_breathing".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Patient".to_string()),
                predicate: Term::Constant("possible_diagnosis".to_string()),
                object: Term::Constant("pneumonia".to_string()),
            }],
        };

        // Treatment recommendation rules
        let antibiotic_recommendation = Rule {
            name: "antibiotic_recommendation".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("Patient".to_string()),
                    predicate: Term::Constant("possible_diagnosis".to_string()),
                    object: Term::Constant("pneumonia".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Patient".to_string()),
                    predicate: Term::Constant("age".to_string()),
                    object: Term::Variable("Age".to_string()),
                },
                RuleAtom::GreaterThan {
                    left: Term::Variable("Age".to_string()),
                    right: Term::Constant("18".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Patient".to_string()),
                predicate: Term::Constant("recommended_treatment".to_string()),
                object: Term::Constant("antibiotics".to_string()),
            }],
        };

        engine.add_rule(flu_diagnosis);
        engine.add_rule(pneumonia_diagnosis);
        engine.add_rule(antibiotic_recommendation);

        // Patient data
        let facts = vec![
            // Patient 1 - Flu symptoms
            RuleAtom::Triple {
                subject: Term::Constant("patient001".to_string()),
                predicate: Term::Constant("has_symptom".to_string()),
                object: Term::Constant("fever".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("patient001".to_string()),
                predicate: Term::Constant("has_symptom".to_string()),
                object: Term::Constant("cough".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("patient001".to_string()),
                predicate: Term::Constant("has_symptom".to_string()),
                object: Term::Constant("body_aches".to_string()),
            },
            // Patient 2 - Pneumonia symptoms
            RuleAtom::Triple {
                subject: Term::Constant("patient002".to_string()),
                predicate: Term::Constant("has_symptom".to_string()),
                object: Term::Constant("fever".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("patient002".to_string()),
                predicate: Term::Constant("has_symptom".to_string()),
                object: Term::Constant("chest_pain".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("patient002".to_string()),
                predicate: Term::Constant("has_symptom".to_string()),
                object: Term::Constant("difficulty_breathing".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("patient002".to_string()),
                predicate: Term::Constant("age".to_string()),
                object: Term::Constant("45".to_string()),
            },
        ];

        engine.forward_chain(&facts)
    }

    /// Financial fraud detection system
    pub fn fraud_detection_system() -> Result<Vec<RuleAtom>> {
        let mut engine = RuleEngine::new();

        // Suspicious transaction patterns
        let unusual_amount_rule = Rule {
            name: "unusual_amount".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("Transaction".to_string()),
                    predicate: Term::Constant("amount".to_string()),
                    object: Term::Variable("Amount".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Transaction".to_string()),
                    predicate: Term::Constant("account".to_string()),
                    object: Term::Variable("Account".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Account".to_string()),
                    predicate: Term::Constant("avg_transaction".to_string()),
                    object: Term::Variable("AvgAmount".to_string()),
                },
                RuleAtom::GreaterThan {
                    left: Term::Variable("Amount".to_string()),
                    right: Term::Function {
                        name: "multiply".to_string(),
                        args: vec![
                            Term::Variable("AvgAmount".to_string()),
                            Term::Constant("10".to_string()),
                        ],
                    },
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Transaction".to_string()),
                predicate: Term::Constant("risk_level".to_string()),
                object: Term::Constant("high".to_string()),
            }],
        };

        let frequent_transactions_rule = Rule {
            name: "frequent_transactions".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("Account".to_string()),
                    predicate: Term::Constant("transactions_today".to_string()),
                    object: Term::Variable("Count".to_string()),
                },
                RuleAtom::GreaterThan {
                    left: Term::Variable("Count".to_string()),
                    right: Term::Constant("20".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Account".to_string()),
                predicate: Term::Constant("suspicious_activity".to_string()),
                object: Term::Constant("high_frequency".to_string()),
            }],
        };

        let geographic_anomaly_rule = Rule {
            name: "geographic_anomaly".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("Transaction".to_string()),
                    predicate: Term::Constant("location".to_string()),
                    object: Term::Variable("Location".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Transaction".to_string()),
                    predicate: Term::Constant("account".to_string()),
                    object: Term::Variable("Account".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Account".to_string()),
                    predicate: Term::Constant("usual_location".to_string()),
                    object: Term::Variable("UsualLocation".to_string()),
                },
                RuleAtom::NotEqual {
                    left: Term::Variable("Location".to_string()),
                    right: Term::Variable("UsualLocation".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Transaction".to_string()),
                predicate: Term::Constant("geographic_risk".to_string()),
                object: Term::Constant("anomaly".to_string()),
            }],
        };

        engine.add_rule(unusual_amount_rule);
        engine.add_rule(frequent_transactions_rule);
        engine.add_rule(geographic_anomaly_rule);

        let facts = vec![
            // Transaction data
            RuleAtom::Triple {
                subject: Term::Constant("txn001".to_string()),
                predicate: Term::Constant("amount".to_string()),
                object: Term::Constant("50000".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("txn001".to_string()),
                predicate: Term::Constant("account".to_string()),
                object: Term::Constant("acc123".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("txn001".to_string()),
                predicate: Term::Constant("location".to_string()),
                object: Term::Constant("russia".to_string()),
            },
            // Account profile
            RuleAtom::Triple {
                subject: Term::Constant("acc123".to_string()),
                predicate: Term::Constant("avg_transaction".to_string()),
                object: Term::Constant("500".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("acc123".to_string()),
                predicate: Term::Constant("usual_location".to_string()),
                object: Term::Constant("usa".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("acc123".to_string()),
                predicate: Term::Constant("transactions_today".to_string()),
                object: Term::Constant("25".to_string()),
            },
        ];

        engine.forward_chain(&facts)
    }

    /// Supply chain management optimization
    pub fn supply_chain_optimization() -> Result<Vec<RuleAtom>> {
        let mut engine = RuleEngine::new();

        // Inventory management rules
        let reorder_rule = Rule {
            name: "reorder_point".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("Product".to_string()),
                    predicate: Term::Constant("current_stock".to_string()),
                    object: Term::Variable("Stock".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Product".to_string()),
                    predicate: Term::Constant("reorder_level".to_string()),
                    object: Term::Variable("ReorderLevel".to_string()),
                },
                RuleAtom::LessThan {
                    left: Term::Variable("Stock".to_string()),
                    right: Term::Variable("ReorderLevel".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Product".to_string()),
                predicate: Term::Constant("action_required".to_string()),
                object: Term::Constant("reorder".to_string()),
            }],
        };

        // Supplier selection rule
        let supplier_selection_rule = Rule {
            name: "supplier_selection".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("Product".to_string()),
                    predicate: Term::Constant("action_required".to_string()),
                    object: Term::Constant("reorder".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Supplier".to_string()),
                    predicate: Term::Constant("supplies".to_string()),
                    object: Term::Variable("Product".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Supplier".to_string()),
                    predicate: Term::Constant("rating".to_string()),
                    object: Term::Variable("Rating".to_string()),
                },
                RuleAtom::GreaterThan {
                    left: Term::Variable("Rating".to_string()),
                    right: Term::Constant("4.0".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Product".to_string()),
                predicate: Term::Constant("preferred_supplier".to_string()),
                object: Term::Variable("Supplier".to_string()),
            }],
        };

        // Urgent delivery rule
        let urgent_delivery_rule = Rule {
            name: "urgent_delivery".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("Product".to_string()),
                    predicate: Term::Constant("current_stock".to_string()),
                    object: Term::Variable("Stock".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Product".to_string()),
                    predicate: Term::Constant("critical_level".to_string()),
                    object: Term::Variable("CriticalLevel".to_string()),
                },
                RuleAtom::LessThan {
                    left: Term::Variable("Stock".to_string()),
                    right: Term::Variable("CriticalLevel".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Product".to_string()),
                predicate: Term::Constant("delivery_priority".to_string()),
                object: Term::Constant("urgent".to_string()),
            }],
        };

        engine.add_rule(reorder_rule);
        engine.add_rule(supplier_selection_rule);
        engine.add_rule(urgent_delivery_rule);

        let facts = vec![
            // Product inventory
            RuleAtom::Triple {
                subject: Term::Constant("laptop_battery".to_string()),
                predicate: Term::Constant("current_stock".to_string()),
                object: Term::Constant("15".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("laptop_battery".to_string()),
                predicate: Term::Constant("reorder_level".to_string()),
                object: Term::Constant("25".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("laptop_battery".to_string()),
                predicate: Term::Constant("critical_level".to_string()),
                object: Term::Constant("10".to_string()),
            },
            // Supplier information
            RuleAtom::Triple {
                subject: Term::Constant("supplier_a".to_string()),
                predicate: Term::Constant("supplies".to_string()),
                object: Term::Constant("laptop_battery".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("supplier_a".to_string()),
                predicate: Term::Constant("rating".to_string()),
                object: Term::Constant("4.5".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("supplier_b".to_string()),
                predicate: Term::Constant("supplies".to_string()),
                object: Term::Constant("laptop_battery".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("supplier_b".to_string()),
                predicate: Term::Constant("rating".to_string()),
                object: Term::Constant("3.8".to_string()),
            },
        ];

        engine.forward_chain(&facts)
    }

    /// Smart home automation system
    pub fn smart_home_automation() -> Result<Vec<RuleAtom>> {
        let mut engine = RuleEngine::new();

        // Energy saving rule
        let energy_saving_rule = Rule {
            name: "energy_saving".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("Room".to_string()),
                    predicate: Term::Constant("occupancy".to_string()),
                    object: Term::Constant("empty".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Room".to_string()),
                    predicate: Term::Constant("lights".to_string()),
                    object: Term::Constant("on".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Room".to_string()),
                predicate: Term::Constant("action".to_string()),
                object: Term::Constant("turn_off_lights".to_string()),
            }],
        };

        // Security rule
        let security_rule = Rule {
            name: "security_alert".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("Sensor".to_string()),
                    predicate: Term::Constant("type".to_string()),
                    object: Term::Constant("motion_sensor".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Sensor".to_string()),
                    predicate: Term::Constant("status".to_string()),
                    object: Term::Constant("triggered".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Constant("home".to_string()),
                    predicate: Term::Constant("security_mode".to_string()),
                    object: Term::Constant("armed".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Constant("security_system".to_string()),
                predicate: Term::Constant("alert".to_string()),
                object: Term::Constant("intrusion_detected".to_string()),
            }],
        };

        // Climate control rule
        let climate_rule = Rule {
            name: "climate_control".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("Room".to_string()),
                    predicate: Term::Constant("temperature".to_string()),
                    object: Term::Variable("Temperature".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Room".to_string()),
                    predicate: Term::Constant("target_temperature".to_string()),
                    object: Term::Variable("Target".to_string()),
                },
                RuleAtom::GreaterThan {
                    left: Term::Variable("Temperature".to_string()),
                    right: Term::Function {
                        name: "add".to_string(),
                        args: vec![
                            Term::Variable("Target".to_string()),
                            Term::Constant("2".to_string()),
                        ],
                    },
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("Room".to_string()),
                predicate: Term::Constant("action".to_string()),
                object: Term::Constant("increase_cooling".to_string()),
            }],
        };

        engine.add_rule(energy_saving_rule);
        engine.add_rule(security_rule);
        engine.add_rule(climate_rule);

        let facts = vec![
            // Room status
            RuleAtom::Triple {
                subject: Term::Constant("living_room".to_string()),
                predicate: Term::Constant("occupancy".to_string()),
                object: Term::Constant("empty".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("living_room".to_string()),
                predicate: Term::Constant("lights".to_string()),
                object: Term::Constant("on".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("bedroom".to_string()),
                predicate: Term::Constant("temperature".to_string()),
                object: Term::Constant("25".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("bedroom".to_string()),
                predicate: Term::Constant("target_temperature".to_string()),
                object: Term::Constant("21".to_string()),
            },
            // Security system
            RuleAtom::Triple {
                subject: Term::Constant("front_door_sensor".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("motion_sensor".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("front_door_sensor".to_string()),
                predicate: Term::Constant("status".to_string()),
                object: Term::Constant("triggered".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("home".to_string()),
                predicate: Term::Constant("security_mode".to_string()),
                object: Term::Constant("armed".to_string()),
            },
        ];

        engine.forward_chain(&facts)
    }
}

/// # Performance Optimization Guide
///
/// This section provides comprehensive guidance on optimizing rule engine performance
/// for production deployments.

pub struct PerformanceOptimization;

impl PerformanceOptimization {
    /// Demonstrate performance monitoring and optimization
    pub fn performance_monitoring_example() -> Result<()> {
        let mut debug_engine = DebuggableRuleEngine::new();
        debug_engine.enable_debugging(false);

        // Create a performance-heavy rule set
        let rules = Self::create_performance_test_rules();
        for rule in rules {
            debug_engine.engine.add_rule(rule);
        }

        let facts = Self::create_large_fact_set(1000);

        let start = Instant::now();
        let _result = debug_engine.debug_forward_chain(&facts)?;
        let duration = start.elapsed();

        // Analyze performance
        let metrics = debug_engine.get_metrics();
        println!("Performance Analysis:");
        println!("Total execution time: {:?}", duration);
        println!("Facts processed: {}", metrics.facts_processed);
        println!("Facts derived: {}", metrics.facts_derived);
        println!(
            "Cache hit rate: {:.2}%",
            (metrics.cache_hits as f64 / (metrics.cache_hits + metrics.cache_misses) as f64)
                * 100.0
        );

        // Identify bottlenecks
        let conflicts = debug_engine.get_conflicts();
        let performance_issues: Vec<_> = conflicts
            .iter()
            .filter(|c| {
                matches!(
                    c.conflict_type,
                    crate::debug::ConflictType::PerformanceBottleneck
                )
            })
            .collect();

        if !performance_issues.is_empty() {
            println!("Performance bottlenecks detected:");
            for issue in performance_issues {
                println!("  - {}", issue.resolution_suggestion);
            }
        }

        Ok(())
    }

    /// Demonstrate caching optimization
    pub fn caching_optimization_example() -> Result<()> {
        let cache = RuleCache::with_sizes(2000, 1000, 500, 800);
        let mut engine = RuleEngine::new();
        engine.set_cache(Some(cache));

        // Warm the cache
        let rules = Self::create_performance_test_rules();
        let facts = Self::create_large_fact_set(100);

        if let Some(cache) = engine.get_cache() {
            cache.warm_cache(&rules, &facts);
        }

        for rule in rules {
            engine.add_rule(rule);
        }

        // Measure performance with and without cache
        let start = Instant::now();
        let _result1 = engine.forward_chain(&facts)?;
        let with_cache = start.elapsed();

        // Clear cache and measure again
        if let Some(cache) = engine.get_cache() {
            cache.clear_all();
        }

        let start = Instant::now();
        let _result2 = engine.forward_chain(&facts)?;
        let without_cache = start.elapsed();

        println!("Cache Performance Analysis:");
        println!("With cache: {:?}", with_cache);
        println!("Without cache: {:?}", without_cache);
        println!(
            "Performance improvement: {:.2}x",
            without_cache.as_nanos() as f64 / with_cache.as_nanos() as f64
        );

        if let Some(cache) = engine.get_cache() {
            let stats = cache.get_statistics();
            println!("Final cache statistics:");
            println!(
                "  Rule cache hit rate: {:.2}%",
                stats.rule_cache.hit_rate * 100.0
            );
            println!(
                "  Derivation cache hit rate: {:.2}%",
                stats.derivation_cache.hit_rate * 100.0
            );
        }

        Ok(())
    }

    /// Create a set of rules for performance testing
    fn create_performance_test_rules() -> Vec<Rule> {
        vec![
            Rule {
                name: "transitive_closure".to_string(),
                body: vec![
                    RuleAtom::Triple {
                        subject: Term::Variable("X".to_string()),
                        predicate: Term::Constant("connected".to_string()),
                        object: Term::Variable("Y".to_string()),
                    },
                    RuleAtom::Triple {
                        subject: Term::Variable("Y".to_string()),
                        predicate: Term::Constant("connected".to_string()),
                        object: Term::Variable("Z".to_string()),
                    },
                ],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("reachable".to_string()),
                    object: Term::Variable("Z".to_string()),
                }],
            },
            Rule {
                name: "classification_rule".to_string(),
                body: vec![
                    RuleAtom::Triple {
                        subject: Term::Variable("X".to_string()),
                        predicate: Term::Constant("property1".to_string()),
                        object: Term::Constant("value1".to_string()),
                    },
                    RuleAtom::Triple {
                        subject: Term::Variable("X".to_string()),
                        predicate: Term::Constant("property2".to_string()),
                        object: Term::Constant("value2".to_string()),
                    },
                ],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("category".to_string()),
                    object: Term::Constant("special".to_string()),
                }],
            },
            Rule {
                name: "aggregation_rule".to_string(),
                body: vec![
                    RuleAtom::Triple {
                        subject: Term::Variable("Group".to_string()),
                        predicate: Term::Constant("member".to_string()),
                        object: Term::Variable("X".to_string()),
                    },
                    RuleAtom::Triple {
                        subject: Term::Variable("X".to_string()),
                        predicate: Term::Constant("value".to_string()),
                        object: Term::Variable("V".to_string()),
                    },
                ],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("Group".to_string()),
                    predicate: Term::Constant("total_value".to_string()),
                    object: Term::Function {
                        name: "sum".to_string(),
                        args: vec![Term::Variable("V".to_string())],
                    },
                }],
            },
        ]
    }

    /// Create a large set of facts for performance testing
    fn create_large_fact_set(size: usize) -> Vec<RuleAtom> {
        let mut facts = Vec::new();

        // Generate connected graph
        for i in 0..size {
            if i < size - 1 {
                facts.push(RuleAtom::Triple {
                    subject: Term::Constant(format!("node{}", i)),
                    predicate: Term::Constant("connected".to_string()),
                    object: Term::Constant(format!("node{}", i + 1)),
                });
            }

            // Add properties
            if i % 3 == 0 {
                facts.push(RuleAtom::Triple {
                    subject: Term::Constant(format!("node{}", i)),
                    predicate: Term::Constant("property1".to_string()),
                    object: Term::Constant("value1".to_string()),
                });
            }

            if i % 5 == 0 {
                facts.push(RuleAtom::Triple {
                    subject: Term::Constant(format!("node{}", i)),
                    predicate: Term::Constant("property2".to_string()),
                    object: Term::Constant("value2".to_string()),
                });
            }
        }

        facts
    }

    /// Performance optimization tips and techniques
    pub fn optimization_techniques() -> Vec<(&'static str, &'static str)> {
        vec![
            (
                "Rule Ordering",
                "Place most selective rules first to reduce search space",
            ),
            (
                "Index Usage",
                "Use appropriate indices for frequently accessed patterns",
            ),
            (
                "Caching Strategy",
                "Implement multi-level caching for rule results and derivations",
            ),
            (
                "Memory Management",
                "Monitor memory usage and implement garbage collection",
            ),
            (
                "Parallel Processing",
                "Use parallel execution for independent rule evaluations",
            ),
            (
                "Fact Ordering",
                "Order facts to maximize early constraint satisfaction",
            ),
            (
                "Rule Specialization",
                "Create specialized rules for common patterns",
            ),
            (
                "Batch Processing",
                "Process facts in batches to improve cache locality",
            ),
            (
                "Incremental Updates",
                "Use incremental reasoning for dynamic fact sets",
            ),
            (
                "Profiling",
                "Regular profiling to identify and address bottlenecks",
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_reasoning() {
        let result = AdvancedFeatures::complex_reasoning_example();
        assert!(result.is_ok());
        let facts = result.unwrap();
        assert!(!facts.is_empty());
    }

    #[test]
    fn test_temporal_reasoning() {
        let result = AdvancedFeatures::temporal_reasoning_example();
        assert!(result.is_ok());
        let facts = result.unwrap();
        assert!(!facts.is_empty());
    }

    #[test]
    fn test_healthcare_diagnosis() {
        let result = RealWorldApplications::healthcare_diagnosis_system();
        assert!(result.is_ok());
        let facts = result.unwrap();

        // Should have diagnosis results
        let diagnoses: Vec<_> = facts
            .iter()
            .filter(|fact| match fact {
                RuleAtom::Triple { predicate, .. } => {
                    predicate == &Term::Constant("possible_diagnosis".to_string())
                }
                _ => false,
            })
            .collect();

        assert!(!diagnoses.is_empty());
    }

    #[test]
    fn test_fraud_detection() {
        let result = RealWorldApplications::fraud_detection_system();
        assert!(result.is_ok());
        let facts = result.unwrap();

        // Should detect risks
        let risks: Vec<_> = facts
            .iter()
            .filter(|fact| match fact {
                RuleAtom::Triple { predicate, .. } => {
                    predicate == &Term::Constant("risk_level".to_string())
                        || predicate == &Term::Constant("suspicious_activity".to_string())
                        || predicate == &Term::Constant("geographic_risk".to_string())
                }
                _ => false,
            })
            .collect();

        assert!(!risks.is_empty());
    }

    #[test]
    fn test_supply_chain_optimization() {
        let result = RealWorldApplications::supply_chain_optimization();
        assert!(result.is_ok());
        let facts = result.unwrap();

        // Should have optimization actions
        let actions: Vec<_> = facts
            .iter()
            .filter(|fact| match fact {
                RuleAtom::Triple { predicate, .. } => {
                    predicate == &Term::Constant("action_required".to_string())
                        || predicate == &Term::Constant("preferred_supplier".to_string())
                        || predicate == &Term::Constant("delivery_priority".to_string())
                }
                _ => false,
            })
            .collect();

        assert!(!actions.is_empty());
    }

    #[test]
    fn test_smart_home_automation() {
        let result = RealWorldApplications::smart_home_automation();
        assert!(result.is_ok());
        let facts = result.unwrap();

        // Should have automation actions
        let actions: Vec<_> = facts
            .iter()
            .filter(|fact| match fact {
                RuleAtom::Triple { predicate, .. } => {
                    predicate == &Term::Constant("action".to_string())
                        || predicate == &Term::Constant("alert".to_string())
                }
                _ => false,
            })
            .collect();

        assert!(!actions.is_empty());
    }

    #[test]
    fn test_performance_monitoring() {
        let result = PerformanceOptimization::performance_monitoring_example();
        assert!(result.is_ok());
    }

    #[test]
    fn test_caching_optimization() {
        let result = PerformanceOptimization::caching_optimization_example();
        assert!(result.is_ok());
    }
}
