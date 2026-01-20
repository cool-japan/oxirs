//! ODRL Policy Engine
//!
//! Implementation of ODRL 2.2 (Open Digital Rights Language) for usage control.
//! <https://www.w3.org/TR/odrl-model/>

pub mod constraint_evaluator;
pub mod odrl_parser;
pub mod usage_control;

pub use constraint_evaluator::{Constraint, ConstraintEvaluator, ConstraintResult};
pub use odrl_parser::{Obligation, OdrlParser, OdrlPolicy, Permission, Prohibition};
pub use usage_control::{UsageController, UsageEvent};

use super::types::{IdsError, IdsResult, IdsUri, Party};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Policy engine for evaluating ODRL policies
pub struct PolicyEngine {
    /// Stored policies
    policies: Arc<RwLock<Vec<OdrlPolicy>>>,

    /// Usage controller
    usage_controller: Arc<UsageController>,

    /// Constraint evaluator
    constraint_evaluator: Arc<ConstraintEvaluator>,
}

impl Default for PolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl PolicyEngine {
    /// Create a new policy engine
    pub fn new() -> Self {
        Self {
            policies: Arc::new(RwLock::new(Vec::new())),
            usage_controller: Arc::new(UsageController::new()),
            constraint_evaluator: Arc::new(ConstraintEvaluator::new()),
        }
    }

    /// Add a policy to the engine
    pub async fn add_policy(&self, policy: OdrlPolicy) -> IdsResult<()> {
        let mut policies = self.policies.write().await;
        policies.push(policy);
        Ok(())
    }

    /// Get policy by UID
    pub async fn get_policy(&self, uid: &IdsUri) -> Option<OdrlPolicy> {
        let policies = self.policies.read().await;
        policies.iter().find(|p| &p.uid == uid).cloned()
    }

    /// Evaluate if an action is permitted
    pub async fn evaluate(
        &self,
        policy_uid: &IdsUri,
        action: &OdrlAction,
        context: &EvaluationContext,
    ) -> IdsResult<PolicyDecision> {
        let policy = self.get_policy(policy_uid).await.ok_or_else(|| {
            IdsError::PolicyViolation(format!("Policy not found: {}", policy_uid))
        })?;

        // Check permissions
        for permission in &policy.permissions {
            if permission.action == *action {
                // Evaluate constraints
                let constraint_results = self
                    .constraint_evaluator
                    .evaluate_constraints(&permission.constraints, context)
                    .await?;

                if constraint_results.iter().all(|r| r.satisfied) {
                    return Ok(PolicyDecision::Permit {
                        policy_uid: policy.uid.clone(),
                        duties: permission.duties.clone(),
                    });
                }
            }
        }

        // Check prohibitions
        for prohibition in &policy.prohibitions {
            if prohibition.action == *action {
                return Ok(PolicyDecision::Deny {
                    policy_uid: policy.uid.clone(),
                    reason: "Action is prohibited".to_string(),
                });
            }
        }

        // Default deny
        Ok(PolicyDecision::Deny {
            policy_uid: policy.uid.clone(),
            reason: "No matching permission found".to_string(),
        })
    }

    /// Record usage event
    pub async fn record_usage(&self, event: UsageEvent) -> IdsResult<()> {
        self.usage_controller.record_event(event).await
    }

    /// Get usage controller
    pub fn usage_controller(&self) -> Arc<UsageController> {
        Arc::clone(&self.usage_controller)
    }
}

/// ODRL Action types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum OdrlAction {
    /// Use the resource
    Use,

    /// Read/access the resource
    Read,

    /// Modify the resource
    Modify,

    /// Delete the resource
    Delete,

    /// Execute the resource
    Execute,

    /// Print the resource
    Print,

    /// Play the resource (media)
    Play,

    /// Display the resource
    Display,

    /// Distribute copies
    Distribute,

    /// Grant access to others
    GrantUse,

    /// Commercial use
    CommercialUse,

    /// Derive new works
    DerivativeWorks,

    /// Reproduce the resource
    Reproduce,

    /// Share the resource
    Share,

    /// Transfer to third party
    Transfer,

    /// Custom action
    Custom(String),
}

impl OdrlAction {
    /// Get the ODRL vocabulary URI
    pub fn to_uri(&self) -> String {
        let base = "http://www.w3.org/ns/odrl/2/";
        match self {
            OdrlAction::Use => format!("{}use", base),
            OdrlAction::Read => format!("{}read", base),
            OdrlAction::Modify => format!("{}modify", base),
            OdrlAction::Delete => format!("{}delete", base),
            OdrlAction::Execute => format!("{}execute", base),
            OdrlAction::Print => format!("{}print", base),
            OdrlAction::Play => format!("{}play", base),
            OdrlAction::Display => format!("{}display", base),
            OdrlAction::Distribute => format!("{}distribute", base),
            OdrlAction::GrantUse => format!("{}grantUse", base),
            OdrlAction::CommercialUse => format!("{}commercialize", base),
            OdrlAction::DerivativeWorks => format!("{}derive", base),
            OdrlAction::Reproduce => format!("{}reproduce", base),
            OdrlAction::Share => format!("{}share", base),
            OdrlAction::Transfer => format!("{}transfer", base),
            OdrlAction::Custom(s) => s.clone(),
        }
    }
}

/// Policy decision result
#[derive(Debug, Clone)]
pub enum PolicyDecision {
    /// Action is permitted
    Permit {
        policy_uid: IdsUri,
        duties: Vec<Duty>,
    },

    /// Action is denied
    Deny { policy_uid: IdsUri, reason: String },

    /// Not applicable (no policy found)
    NotApplicable,
}

impl PolicyDecision {
    /// Check if the decision is a permit
    pub fn is_permit(&self) -> bool {
        matches!(self, PolicyDecision::Permit { .. })
    }

    /// Check if the decision is a deny
    pub fn is_deny(&self) -> bool {
        matches!(self, PolicyDecision::Deny { .. })
    }

    /// Get duties if permitted
    pub fn duties(&self) -> Vec<Duty> {
        match self {
            PolicyDecision::Permit { duties, .. } => duties.clone(),
            _ => Vec::new(),
        }
    }
}

/// Duty that must be fulfilled
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Duty {
    /// Action that must be performed
    pub action: OdrlAction,

    /// Constraints on the duty
    pub constraints: Vec<Constraint>,
}

/// Evaluation context for policy decisions
#[derive(Debug, Clone)]
pub struct EvaluationContext {
    /// Current timestamp
    pub timestamp: DateTime<Utc>,

    /// Requesting party
    pub requestor: Option<Party>,

    /// Target resource
    pub resource: Option<IdsUri>,

    /// Request metadata
    pub metadata: std::collections::HashMap<String, String>,

    /// Source IP address
    pub source_ip: Option<std::net::IpAddr>,

    /// Gaia-X trust level (0.0 - 1.0)
    pub trust_level: Option<f64>,

    /// Connector ID (from DAPS token)
    pub connector_id: Option<String>,

    /// Event type for event-based constraints
    pub event_type: Option<String>,

    /// Current usage count for the resource
    pub usage_count: Option<u64>,

    /// Region code (ISO 3166-1 alpha-2)
    pub region_code: Option<String>,

    /// Purpose of data usage
    pub purpose: Option<String>,
}

impl EvaluationContext {
    /// Create a new evaluation context
    pub fn new() -> Self {
        Self {
            timestamp: Utc::now(),
            requestor: None,
            resource: None,
            metadata: std::collections::HashMap::new(),
            source_ip: None,
            trust_level: None,
            connector_id: None,
            event_type: None,
            usage_count: None,
            region_code: None,
            purpose: None,
        }
    }

    /// Set requestor
    pub fn with_requestor(mut self, party: Party) -> Self {
        self.requestor = Some(party);
        self
    }

    /// Set resource
    pub fn with_resource(mut self, resource: IdsUri) -> Self {
        self.resource = Some(resource);
        self
    }

    /// Set source IP
    pub fn with_source_ip(mut self, ip: std::net::IpAddr) -> Self {
        self.source_ip = Some(ip);
        self
    }

    /// Set trust level
    pub fn with_trust_level(mut self, level: f64) -> Self {
        self.trust_level = Some(level);
        self
    }

    /// Set connector ID
    pub fn with_connector_id(mut self, connector_id: impl Into<String>) -> Self {
        self.connector_id = Some(connector_id.into());
        self
    }

    /// Set event type
    pub fn with_event_type(mut self, event_type: impl Into<String>) -> Self {
        self.event_type = Some(event_type.into());
        self
    }

    /// Set usage count
    pub fn with_usage_count(mut self, count: u64) -> Self {
        self.usage_count = Some(count);
        self
    }

    /// Set region code
    pub fn with_region(mut self, region_code: impl Into<String>) -> Self {
        self.region_code = Some(region_code.into());
        self
    }

    /// Set purpose
    pub fn with_purpose(mut self, purpose: impl Into<String>) -> Self {
        self.purpose = Some(purpose.into());
        self
    }
}

impl Default for EvaluationContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::types::SecurityProfile;

    #[test]
    fn test_odrl_action_uri() {
        assert_eq!(
            OdrlAction::Read.to_uri(),
            "http://www.w3.org/ns/odrl/2/read"
        );
        assert_eq!(OdrlAction::Use.to_uri(), "http://www.w3.org/ns/odrl/2/use");
        assert_eq!(
            OdrlAction::CommercialUse.to_uri(),
            "http://www.w3.org/ns/odrl/2/commercialize"
        );
    }

    #[test]
    fn test_policy_decision() {
        let decision = PolicyDecision::Permit {
            policy_uid: IdsUri::new("https://example.org/policy/1").unwrap(),
            duties: Vec::new(),
        };

        assert!(decision.is_permit());
        assert!(!decision.is_deny());

        let decision = PolicyDecision::Deny {
            policy_uid: IdsUri::new("https://example.org/policy/1").unwrap(),
            reason: "Test denial".to_string(),
        };

        assert!(!decision.is_permit());
        assert!(decision.is_deny());
    }

    #[test]
    fn test_security_profile_ordering() {
        assert!(SecurityProfile::BaseSecurityProfile < SecurityProfile::TrustSecurityProfile);
        assert!(SecurityProfile::TrustSecurityProfile < SecurityProfile::TrustPlusSecurityProfile);
    }

    #[test]
    fn test_evaluation_context() {
        let ctx = EvaluationContext::new()
            .with_resource(IdsUri::new("https://example.org/data").unwrap())
            .with_trust_level(0.95);

        assert!(ctx.resource.is_some());
        assert_eq!(ctx.trust_level, Some(0.95));
    }
}
