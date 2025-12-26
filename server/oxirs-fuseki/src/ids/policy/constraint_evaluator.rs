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
        _allowed_regions: &[Region],
        _restriction_type: SpatialRestriction,
        _context: &EvaluationContext,
    ) -> IdsResult<ConstraintResult> {
        // TODO: Implement geolocation-based evaluation
        Ok(ConstraintResult {
            satisfied: true,
            constraint: "spatial(...)".to_string(),
            reason: None,
            trust_impact: 1.0,
        })
    }

    /// Evaluate purpose constraint
    fn evaluate_purpose(
        &self,
        _allowed_purposes: &[Purpose],
        context: &EvaluationContext,
    ) -> IdsResult<ConstraintResult> {
        // Check if request metadata contains purpose
        let request_purpose = context.metadata.get("purpose");

        Ok(ConstraintResult {
            satisfied: request_purpose.is_some(),
            constraint: "purpose(...)".to_string(),
            reason: if request_purpose.is_none() {
                Some("Purpose not specified in request".to_string())
            } else {
                None
            },
            trust_impact: if request_purpose.is_some() { 1.0 } else { 0.5 },
        })
    }

    /// Evaluate count constraint
    fn evaluate_count(
        &self,
        _operator: ComparisonOperator,
        _max_count: u64,
        _context: &EvaluationContext,
    ) -> IdsResult<ConstraintResult> {
        // TODO: Track usage count per resource
        Ok(ConstraintResult {
            satisfied: true,
            constraint: "count(...)".to_string(),
            reason: None,
            trust_impact: 1.0,
        })
    }

    /// Evaluate connector constraint
    fn evaluate_connector(
        &self,
        _allowed_connector_ids: &[String],
        _context: &EvaluationContext,
    ) -> IdsResult<ConstraintResult> {
        // TODO: Check connector ID from context
        Ok(ConstraintResult {
            satisfied: true,
            constraint: "connector(...)".to_string(),
            reason: None,
            trust_impact: 1.0,
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
        _event_type: &str,
        _context: &EvaluationContext,
    ) -> IdsResult<ConstraintResult> {
        // TODO: Check event in context
        Ok(ConstraintResult {
            satisfied: true,
            constraint: "event(...)".to_string(),
            reason: None,
            trust_impact: 1.0,
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
}
