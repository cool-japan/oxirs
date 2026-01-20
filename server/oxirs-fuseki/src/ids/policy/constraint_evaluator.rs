//! ODRL Constraint Evaluator
//!
//! Evaluates constraints attached to ODRL rules (permissions, prohibitions, obligations).

use super::super::residency::Region;
use super::EvaluationContext;
use crate::ids::types::{IdsError, IdsResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::net::IpAddr;

/// ODRL Constraint
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Constraint {
    /// Temporal constraint (valid time period)
    Temporal {
        left_operand: TemporalOperand,
        operator: ComparisonOperator,
        right_operand: DateTime<Utc>,
    },

    /// Spatial constraint (geographic region)
    Spatial {
        allowed_regions: Vec<Region>,
        restriction_type: SpatialRestriction,
    },

    /// Purpose constraint (data usage purpose)
    Purpose { allowed_purposes: Vec<Purpose> },

    /// Count constraint (usage limit)
    Count {
        operator: ComparisonOperator,
        max_count: u64,
    },

    /// Connector constraint (specific IDS connectors)
    Connector { allowed_connector_ids: Vec<String> },

    /// Security level constraint
    SecurityLevel { min_level: u8 },

    /// Event constraint (trigger based on events)
    Event { event_type: String },

    /// Logical constraint (AND, OR, XOR)
    Logical {
        operator: LogicalOperator,
        operands: Vec<Box<Constraint>>,
    },
}

/// Temporal left operand
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TemporalOperand {
    /// Current date/time
    DateTime,

    /// Event time (e.g., when resource was accessed)
    EventDateTime,

    /// Policy issuance time
    PolicyDateTime,

    /// Elapsed time since event
    ElapsedTime,
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ComparisonOperator {
    /// Equal to
    Eq,

    /// Not equal to
    Neq,

    /// Less than
    Lt,

    /// Less than or equal
    Lteq,

    /// Greater than
    Gt,

    /// Greater than or equal
    Gteq,

    /// Is a member of set
    IsA,

    /// Has part
    HasPart,

    /// Is part of
    IsPartOf,

    /// Is all of
    IsAllOf,

    /// Is any of
    IsAnyOf,

    /// Is none of
    IsNoneOf,
}

/// Logical operators for combining constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum LogicalOperator {
    /// All operands must be true
    And,

    /// Any operand must be true
    Or,

    /// Exactly one operand must be true
    Xor,

    /// Operands must not be true
    AndNot,
}

/// Spatial restriction type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SpatialRestriction {
    /// Data must be within allowed regions
    Within,

    /// Data must not be in prohibited regions
    Outside,

    /// Data processing must occur in region
    ProcessingIn,

    /// Data storage must be in region
    StorageIn,
}

/// Purpose for data usage (ODRL + IDS extensions)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Purpose {
    /// Commercial use
    CommercialUse,

    /// Research use
    ResearchUse,

    /// Educational use
    EducationalUse,

    /// Quality improvement
    QualityImprovement,

    /// Service provisioning
    ServiceProvisioning,

    /// Analytics
    Analytics,

    /// Machine learning training
    MachineLearningTraining,

    /// Testing/evaluation
    Testing,

    /// Compliance/audit
    Compliance,

    /// Custom purpose
    Custom(String),
}

/// Constraint evaluation result
#[derive(Debug, Clone)]
pub struct ConstraintResult {
    /// Was the constraint satisfied?
    pub satisfied: bool,

    /// Constraint that was evaluated
    pub constraint: String,

    /// Reason if not satisfied
    pub reason: Option<String>,

    /// Trust score impact (0.0 - 1.0)
    pub trust_impact: f64,
}

/// Constraint Evaluator
pub struct ConstraintEvaluator;

impl ConstraintEvaluator {
    /// Create a new constraint evaluator
    pub fn new() -> Self {
        Self
    }

    /// Evaluate a list of constraints
    pub async fn evaluate_constraints(
        &self,
        constraints: &[Constraint],
        context: &EvaluationContext,
    ) -> IdsResult<Vec<ConstraintResult>> {
        let mut results = Vec::new();

        for constraint in constraints {
            results.push(self.evaluate_constraint(constraint, context).await?);
        }

        Ok(results)
    }

    /// Evaluate a single constraint
    pub async fn evaluate_constraint(
        &self,
        constraint: &Constraint,
        context: &EvaluationContext,
    ) -> IdsResult<ConstraintResult> {
        match constraint {
            Constraint::Temporal {
                left_operand,
                operator,
                right_operand,
            } => self.evaluate_temporal(*left_operand, *operator, *right_operand, context),

            Constraint::Spatial {
                allowed_regions,
                restriction_type,
            } => self.evaluate_spatial(allowed_regions, *restriction_type, context),

            Constraint::Purpose { allowed_purposes } => {
                self.evaluate_purpose(allowed_purposes, context)
            }

            Constraint::Count {
                operator,
                max_count,
            } => self.evaluate_count(*operator, *max_count, context),

            Constraint::Connector {
                allowed_connector_ids,
            } => self.evaluate_connector(allowed_connector_ids, context),

            Constraint::SecurityLevel { min_level } => {
                self.evaluate_security_level(*min_level, context)
            }

            Constraint::Event { event_type } => self.evaluate_event(event_type, context),

            Constraint::Logical { operator, operands } => {
                self.evaluate_logical(*operator, operands, context).await
            }
        }
    }

    /// Evaluate temporal constraint
    fn evaluate_temporal(
        &self,
        left_operand: TemporalOperand,
        operator: ComparisonOperator,
        right_operand: DateTime<Utc>,
        context: &EvaluationContext,
    ) -> IdsResult<ConstraintResult> {
        let left_value = match left_operand {
            TemporalOperand::DateTime | TemporalOperand::EventDateTime => context.timestamp,
            TemporalOperand::PolicyDateTime => Utc::now(),
            TemporalOperand::ElapsedTime => context.timestamp,
        };

        let satisfied = match operator {
            ComparisonOperator::Lt => left_value < right_operand,
            ComparisonOperator::Lteq => left_value <= right_operand,
            ComparisonOperator::Gt => left_value > right_operand,
            ComparisonOperator::Gteq => left_value >= right_operand,
            ComparisonOperator::Eq => left_value == right_operand,
            ComparisonOperator::Neq => left_value != right_operand,
            _ => false,
        };

        Ok(ConstraintResult {
            satisfied,
            constraint: format!(
                "temporal({:?} {:?} {})",
                left_operand, operator, right_operand
            ),
            reason: if !satisfied {
                Some(format!(
                    "Time constraint not met: {} {:?} {}",
                    left_value, operator, right_operand
                ))
            } else {
                None
            },
            trust_impact: if satisfied { 1.0 } else { 0.0 },
        })
    }

    /// Evaluate spatial constraint
    fn evaluate_spatial(
        &self,
        allowed_regions: &[Region],
        restriction_type: SpatialRestriction,
        context: &EvaluationContext,
    ) -> IdsResult<ConstraintResult> {
        // Get region from context (could be from IP geolocation or explicit)
        let current_region = context.region_code.as_deref();

        let satisfied = match restriction_type {
            SpatialRestriction::Within
            | SpatialRestriction::ProcessingIn
            | SpatialRestriction::StorageIn => {
                // Must be within allowed regions
                if let Some(region_code) = current_region {
                    allowed_regions
                        .iter()
                        .any(|r| r.jurisdiction.country_code == region_code)
                } else {
                    // No region info - conservative denial
                    false
                }
            }
            SpatialRestriction::Outside => {
                // Must NOT be in listed regions
                if let Some(region_code) = current_region {
                    !allowed_regions
                        .iter()
                        .any(|r| r.jurisdiction.country_code == region_code)
                } else {
                    // No region info - conservative denial
                    false
                }
            }
        };

        let region_names: Vec<_> = allowed_regions.iter().map(|r| r.name.as_str()).collect();
        Ok(ConstraintResult {
            satisfied,
            constraint: format!(
                "spatial({:?}, regions={:?})",
                restriction_type, region_names
            ),
            reason: if !satisfied {
                Some(format!(
                    "Region {} does not satisfy {:?} constraint for regions {:?}",
                    current_region.unwrap_or("unknown"),
                    restriction_type,
                    region_names
                ))
            } else {
                None
            },
            trust_impact: if satisfied { 1.0 } else { 0.0 },
        })
    }

    /// Evaluate purpose constraint
    fn evaluate_purpose(
        &self,
        allowed_purposes: &[Purpose],
        context: &EvaluationContext,
    ) -> IdsResult<ConstraintResult> {
        // Get purpose from context (direct field or metadata)
        let request_purpose = context
            .purpose
            .as_deref()
            .or_else(|| context.metadata.get("purpose").map(|s| s.as_str()));

        let satisfied = if let Some(purpose_str) = request_purpose {
            // Try to match against allowed purposes
            allowed_purposes.iter().any(|allowed| match allowed {
                Purpose::CommercialUse => purpose_str.to_lowercase().contains("commercial"),
                Purpose::ResearchUse => purpose_str.to_lowercase().contains("research"),
                Purpose::EducationalUse => purpose_str.to_lowercase().contains("education"),
                Purpose::QualityImprovement => purpose_str.to_lowercase().contains("quality"),
                Purpose::ServiceProvisioning => purpose_str.to_lowercase().contains("service"),
                Purpose::Analytics => {
                    purpose_str.to_lowercase().contains("analytics")
                        || purpose_str.to_lowercase().contains("analysis")
                }
                Purpose::MachineLearningTraining => {
                    purpose_str.to_lowercase().contains("ml")
                        || purpose_str.to_lowercase().contains("machine learning")
                        || purpose_str.to_lowercase().contains("training")
                }
                Purpose::Testing => purpose_str.to_lowercase().contains("test"),
                Purpose::Compliance => {
                    purpose_str.to_lowercase().contains("compliance")
                        || purpose_str.to_lowercase().contains("audit")
                }
                Purpose::Custom(custom) => purpose_str.to_lowercase() == custom.to_lowercase(),
            })
        } else {
            false
        };

        let purpose_names: Vec<_> = allowed_purposes
            .iter()
            .map(|p| format!("{:?}", p))
            .collect();
        Ok(ConstraintResult {
            satisfied,
            constraint: format!("purpose(allowed={:?})", purpose_names),
            reason: if !satisfied {
                Some(format!(
                    "Purpose '{}' not in allowed purposes: {:?}",
                    request_purpose.unwrap_or("not specified"),
                    purpose_names
                ))
            } else {
                None
            },
            trust_impact: if satisfied { 1.0 } else { 0.3 },
        })
    }

    /// Evaluate count constraint
    fn evaluate_count(
        &self,
        operator: ComparisonOperator,
        max_count: u64,
        context: &EvaluationContext,
    ) -> IdsResult<ConstraintResult> {
        // Get current usage count from context
        let current_count = context.usage_count.unwrap_or(0);

        let satisfied = match operator {
            ComparisonOperator::Lt => current_count < max_count,
            ComparisonOperator::Lteq => current_count <= max_count,
            ComparisonOperator::Gt => current_count > max_count,
            ComparisonOperator::Gteq => current_count >= max_count,
            ComparisonOperator::Eq => current_count == max_count,
            ComparisonOperator::Neq => current_count != max_count,
            _ => false, // Other operators not applicable to count
        };

        Ok(ConstraintResult {
            satisfied,
            constraint: format!(
                "count(current={}, {:?} {})",
                current_count, operator, max_count
            ),
            reason: if !satisfied {
                Some(format!(
                    "Usage count {} does not satisfy {:?} {}",
                    current_count, operator, max_count
                ))
            } else {
                None
            },
            trust_impact: if satisfied { 1.0 } else { 0.0 },
        })
    }

    /// Evaluate connector constraint
    fn evaluate_connector(
        &self,
        allowed_connector_ids: &[String],
        context: &EvaluationContext,
    ) -> IdsResult<ConstraintResult> {
        // Get connector ID from context (typically from DAPS token)
        let connector_id = context.connector_id.as_deref();

        let satisfied = if let Some(id) = connector_id {
            // Check if connector ID is in the allowed list
            allowed_connector_ids.iter().any(|allowed| {
                // Exact match or wildcard pattern matching
                if allowed.ends_with('*') {
                    let prefix = &allowed[..allowed.len() - 1];
                    id.starts_with(prefix)
                } else {
                    id == allowed
                }
            })
        } else {
            // No connector ID in context - deny by default
            false
        };

        Ok(ConstraintResult {
            satisfied,
            constraint: format!("connector(allowed={:?})", allowed_connector_ids),
            reason: if !satisfied {
                Some(format!(
                    "Connector '{}' not in allowed list: {:?}",
                    connector_id.unwrap_or("not specified"),
                    allowed_connector_ids
                ))
            } else {
                None
            },
            trust_impact: if satisfied { 1.0 } else { 0.0 },
        })
    }

    /// Evaluate security level constraint
    fn evaluate_security_level(
        &self,
        min_level: u8,
        context: &EvaluationContext,
    ) -> IdsResult<ConstraintResult> {
        let trust_level = context.trust_level.unwrap_or(0.0);
        let security_level = (trust_level * 10.0) as u8;

        let satisfied = security_level >= min_level;

        Ok(ConstraintResult {
            satisfied,
            constraint: format!("securityLevel(min={})", min_level),
            reason: if !satisfied {
                Some(format!(
                    "Security level {} below minimum {}",
                    security_level, min_level
                ))
            } else {
                None
            },
            trust_impact: if satisfied { 1.0 } else { 0.0 },
        })
    }

    /// Evaluate event constraint
    fn evaluate_event(
        &self,
        expected_event_type: &str,
        context: &EvaluationContext,
    ) -> IdsResult<ConstraintResult> {
        // Get event type from context
        let current_event = context.event_type.as_deref();

        let satisfied = if let Some(event) = current_event {
            // Check if current event matches expected event type
            // Support hierarchical event matching (e.g., "policyUsage" matches "policyUsage:firstUse")
            event == expected_event_type
                || event.starts_with(&format!("{}:", expected_event_type))
                || expected_event_type.starts_with(&format!("{}:", event))
        } else {
            // No event in context - check if constraint is for "any" event
            expected_event_type.eq_ignore_ascii_case("any")
        };

        Ok(ConstraintResult {
            satisfied,
            constraint: format!("event(expected={})", expected_event_type),
            reason: if !satisfied {
                Some(format!(
                    "Event '{}' does not match expected '{}'",
                    current_event.unwrap_or("none"),
                    expected_event_type
                ))
            } else {
                None
            },
            trust_impact: if satisfied { 1.0 } else { 0.5 },
        })
    }

    /// Evaluate logical constraint (AND, OR, XOR, ANDNOT)
    async fn evaluate_logical(
        &self,
        operator: LogicalOperator,
        operands: &[Box<Constraint>],
        context: &EvaluationContext,
    ) -> IdsResult<ConstraintResult> {
        let mut operand_results = Vec::new();

        for operand in operands {
            operand_results.push(Box::pin(self.evaluate_constraint(operand, context)).await?);
        }

        let satisfied = match operator {
            LogicalOperator::And => operand_results.iter().all(|r| r.satisfied),
            LogicalOperator::Or => operand_results.iter().any(|r| r.satisfied),
            LogicalOperator::Xor => operand_results.iter().filter(|r| r.satisfied).count() == 1,
            LogicalOperator::AndNot => operand_results.iter().all(|r| !r.satisfied),
        };

        let trust_impact = if satisfied {
            operand_results.iter().map(|r| r.trust_impact).sum::<f64>()
                / operand_results.len() as f64
        } else {
            0.0
        };

        Ok(ConstraintResult {
            satisfied,
            constraint: format!("logical({:?}, {} operands)", operator, operands.len()),
            reason: if !satisfied {
                Some(format!("Logical constraint {:?} not satisfied", operator))
            } else {
                None
            },
            trust_impact,
        })
    }
}

impl Default for ConstraintEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_temporal_constraint() {
        let evaluator = ConstraintEvaluator::new();
        let context = EvaluationContext::new();

        let future_time = Utc::now() + chrono::Duration::days(1);

        let result = evaluator
            .evaluate_temporal(
                TemporalOperand::DateTime,
                ComparisonOperator::Lt,
                future_time,
                &context,
            )
            .unwrap();

        assert!(result.satisfied);
    }

    #[tokio::test]
    async fn test_security_level_constraint() {
        let evaluator = ConstraintEvaluator::new();
        let context = EvaluationContext::new().with_trust_level(0.8);

        let result = evaluator.evaluate_security_level(5, &context).unwrap();

        assert!(result.satisfied);
    }

    #[tokio::test]
    async fn test_logical_and_constraint() {
        let evaluator = ConstraintEvaluator::new();
        let mut context = EvaluationContext::new();
        context
            .metadata
            .insert("purpose".to_string(), "research".to_string());

        let constraint = Constraint::Logical {
            operator: LogicalOperator::And,
            operands: vec![
                Box::new(Constraint::SecurityLevel { min_level: 0 }),
                Box::new(Constraint::Purpose {
                    allowed_purposes: vec![Purpose::ResearchUse],
                }),
            ],
        };

        let result = evaluator
            .evaluate_constraint(&constraint, &context)
            .await
            .unwrap();

        assert!(result.satisfied);
    }

    #[tokio::test]
    async fn test_count_constraint_satisfied() {
        let evaluator = ConstraintEvaluator::new();
        let context = EvaluationContext::new().with_usage_count(5);

        let result = evaluator
            .evaluate_count(ComparisonOperator::Lteq, 10, &context)
            .unwrap();

        assert!(result.satisfied);
        assert!(result.constraint.contains("current=5"));
    }

    #[tokio::test]
    async fn test_count_constraint_exceeded() {
        let evaluator = ConstraintEvaluator::new();
        let context = EvaluationContext::new().with_usage_count(15);

        let result = evaluator
            .evaluate_count(ComparisonOperator::Lteq, 10, &context)
            .unwrap();

        assert!(!result.satisfied);
        assert!(result.reason.is_some());
    }

    #[tokio::test]
    async fn test_connector_constraint_allowed() {
        let evaluator = ConstraintEvaluator::new();
        let context =
            EvaluationContext::new().with_connector_id("urn:ids:connector:acme-corp:connector-01");

        let allowed = vec![
            "urn:ids:connector:acme-corp:connector-01".to_string(),
            "urn:ids:connector:partner-inc:connector-02".to_string(),
        ];

        let result = evaluator.evaluate_connector(&allowed, &context).unwrap();

        assert!(result.satisfied);
    }

    #[tokio::test]
    async fn test_connector_constraint_wildcard() {
        let evaluator = ConstraintEvaluator::new();
        let context =
            EvaluationContext::new().with_connector_id("urn:ids:connector:acme-corp:connector-99");

        let allowed = vec!["urn:ids:connector:acme-corp:*".to_string()];

        let result = evaluator.evaluate_connector(&allowed, &context).unwrap();

        assert!(result.satisfied);
    }

    #[tokio::test]
    async fn test_connector_constraint_denied() {
        let evaluator = ConstraintEvaluator::new();
        let context =
            EvaluationContext::new().with_connector_id("urn:ids:connector:unknown:connector-01");

        let allowed = vec!["urn:ids:connector:acme-corp:*".to_string()];

        let result = evaluator.evaluate_connector(&allowed, &context).unwrap();

        assert!(!result.satisfied);
    }

    #[tokio::test]
    async fn test_event_constraint_matched() {
        let evaluator = ConstraintEvaluator::new();
        let context = EvaluationContext::new().with_event_type("policyUsage:firstUse");

        let result = evaluator.evaluate_event("policyUsage", &context).unwrap();

        assert!(result.satisfied);
    }

    #[tokio::test]
    async fn test_event_constraint_not_matched() {
        let evaluator = ConstraintEvaluator::new();
        let context = EvaluationContext::new().with_event_type("contractExpiry");

        let result = evaluator.evaluate_event("policyUsage", &context).unwrap();

        assert!(!result.satisfied);
    }

    #[tokio::test]
    async fn test_purpose_constraint_research() {
        let evaluator = ConstraintEvaluator::new();
        let context = EvaluationContext::new().with_purpose("research and development");

        let allowed = vec![Purpose::ResearchUse, Purpose::EducationalUse];

        let result = evaluator.evaluate_purpose(&allowed, &context).unwrap();

        assert!(result.satisfied);
    }

    #[tokio::test]
    async fn test_purpose_constraint_denied() {
        let evaluator = ConstraintEvaluator::new();
        let context = EvaluationContext::new().with_purpose("commercial advertising");

        let allowed = vec![Purpose::ResearchUse, Purpose::EducationalUse];

        let result = evaluator.evaluate_purpose(&allowed, &context).unwrap();

        assert!(!result.satisfied);
    }
}
