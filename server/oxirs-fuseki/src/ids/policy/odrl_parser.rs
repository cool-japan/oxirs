//! ODRL 2.2 Policy Parser
//!
//! Parses and validates ODRL (Open Digital Rights Language) policies.
//! <https://www.w3.org/TR/odrl-model/>

use super::constraint_evaluator::Constraint;
use super::{Duty, OdrlAction};
use crate::ids::types::{IdsError, IdsResult, IdsUri, Party};
use serde::{Deserialize, Serialize};

/// ODRL Policy (Set, Offer, or Agreement)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OdrlPolicy {
    /// Policy unique identifier
    #[serde(rename = "@id")]
    pub uid: IdsUri,

    /// Policy type
    #[serde(rename = "@type")]
    pub policy_type: PolicyType,

    /// JSON-LD context
    #[serde(rename = "@context", skip_serializing_if = "Option::is_none")]
    pub context: Option<serde_json::Value>,

    /// Profile
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,

    /// Permissions granted by this policy
    #[serde(default)]
    pub permissions: Vec<Permission>,

    /// Prohibitions enforced by this policy
    #[serde(default)]
    pub prohibitions: Vec<Prohibition>,

    /// Obligations required by this policy
    #[serde(default)]
    pub obligations: Vec<Obligation>,

    /// Target assets for this policy
    #[serde(default)]
    pub targets: Vec<AssetTarget>,

    /// Policy assigner (provider)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assigner: Option<Party>,

    /// Policy assignee (consumer)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assignee: Option<Party>,

    /// Inheritance source
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inherits_from: Option<IdsUri>,

    /// Conflict resolution strategy
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conflict: Option<ConflictStrategy>,
}

/// ODRL Policy Type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyType {
    /// Set (template for creating Offers and Agreements)
    Set,

    /// Offer (proposal from assigner to assignee)
    Offer,

    /// Agreement (accepted contract between parties)
    Agreement,

    /// Privacy policy
    Privacy,

    /// Request (request from assignee to assigner)
    Request,

    /// Ticket (transferable permission)
    Ticket,

    /// Assertion (unilateral statement)
    Assertion,
}

/// ODRL Permission
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Permission {
    /// Unique identifier for this permission
    #[serde(rename = "@id", skip_serializing_if = "Option::is_none")]
    pub uid: Option<IdsUri>,

    /// Action permitted
    pub action: OdrlAction,

    /// Constraints limiting the permission
    #[serde(default)]
    pub constraints: Vec<Constraint>,

    /// Duties that must be fulfilled
    #[serde(default)]
    pub duties: Vec<Duty>,

    /// Target assets
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<AssetTarget>,

    /// Assignee (who can perform the action)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assignee: Option<Party>,

    /// Assigner (who grants the permission)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assigner: Option<Party>,
}

/// ODRL Prohibition
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Prohibition {
    /// Unique identifier
    #[serde(rename = "@id", skip_serializing_if = "Option::is_none")]
    pub uid: Option<IdsUri>,

    /// Action prohibited
    pub action: OdrlAction,

    /// Constraints specifying when prohibition applies
    #[serde(default)]
    pub constraints: Vec<Constraint>,

    /// Remedies available if prohibition is violated
    #[serde(default)]
    pub remedies: Vec<Duty>,

    /// Target assets
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<AssetTarget>,
}

/// ODRL Obligation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Obligation {
    /// Unique identifier
    #[serde(rename = "@id", skip_serializing_if = "Option::is_none")]
    pub uid: Option<IdsUri>,

    /// Action that must be performed
    pub action: OdrlAction,

    /// Constraints on when obligation must be fulfilled
    #[serde(default)]
    pub constraints: Vec<Constraint>,

    /// Consequences if obligation is not fulfilled
    #[serde(default)]
    pub consequences: Vec<Duty>,

    /// Compensations offered for fulfilling obligation
    #[serde(default)]
    pub compensations: Vec<Duty>,
}

/// Asset target (resource covered by policy)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AssetTarget {
    /// Asset identifier
    #[serde(rename = "@id")]
    pub uid: IdsUri,

    /// Asset type
    #[serde(rename = "@type", skip_serializing_if = "Option::is_none")]
    pub asset_type: Option<String>,

    /// Asset title
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    /// Asset description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ConflictStrategy {
    /// Prohibition takes precedence
    Perm,

    /// Permission takes precedence
    Prohibit,

    /// Invalid (conflict detected)
    Invalid,
}

/// ODRL Parser
pub struct OdrlParser;

impl OdrlParser {
    /// Parse ODRL policy from JSON-LD
    pub fn parse(json: &str) -> IdsResult<OdrlPolicy> {
        serde_json::from_str(json).map_err(|e| {
            IdsError::SerializationError(format!("Failed to parse ODRL policy: {}", e))
        })
    }

    /// Serialize ODRL policy to JSON-LD
    pub fn serialize(policy: &OdrlPolicy) -> IdsResult<String> {
        serde_json::to_string_pretty(policy).map_err(|e| {
            IdsError::SerializationError(format!("Failed to serialize ODRL policy: {}", e))
        })
    }

    /// Validate policy structure
    pub fn validate(policy: &OdrlPolicy) -> IdsResult<()> {
        // Check at least one rule (permission, prohibition, or obligation)
        if policy.permissions.is_empty()
            && policy.prohibitions.is_empty()
            && policy.obligations.is_empty()
        {
            return Err(IdsError::PolicyViolation(
                "Policy must have at least one permission, prohibition, or obligation".to_string(),
            ));
        }

        // Validate permissions
        for perm in &policy.permissions {
            if matches!(perm.action, OdrlAction::Custom(ref s) if s.is_empty()) {
                return Err(IdsError::PolicyViolation(
                    "Custom action must have a non-empty URI".to_string(),
                ));
            }
        }

        // Validate that Agreement type has both assigner and assignee
        if policy.policy_type == PolicyType::Agreement
            && (policy.assigner.is_none() || policy.assignee.is_none())
        {
            return Err(IdsError::PolicyViolation(
                "Agreement must have both assigner and assignee".to_string(),
            ));
        }

        Ok(())
    }

    /// Create a simple read permission policy
    pub fn create_read_permission(
        uid: IdsUri,
        assigner: Party,
        assignee: Party,
        target: AssetTarget,
    ) -> OdrlPolicy {
        OdrlPolicy {
            uid,
            policy_type: PolicyType::Agreement,
            context: Some(serde_json::json!("http://www.w3.org/ns/odrl.jsonld")),
            profile: None,
            permissions: vec![Permission {
                uid: None,
                action: OdrlAction::Read,
                constraints: Vec::new(),
                duties: Vec::new(),
                target: Some(target.clone()),
                assignee: Some(assignee.clone()),
                assigner: Some(assigner.clone()),
            }],
            prohibitions: Vec::new(),
            obligations: Vec::new(),
            targets: vec![target],
            assigner: Some(assigner),
            assignee: Some(assignee),
            inherits_from: None,
            conflict: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_policy() {
        let json = r#"{
            "@id": "https://example.org/policy/read-only",
            "@type": "Agreement",
            "permissions": [{
                "action": "read"
            }]
        }"#;

        let policy = OdrlParser::parse(json).unwrap();
        assert_eq!(policy.policy_type, PolicyType::Agreement);
        assert_eq!(policy.permissions.len(), 1);
        assert_eq!(policy.permissions[0].action, OdrlAction::Read);
    }

    #[test]
    fn test_validate_policy() {
        let policy = OdrlPolicy {
            uid: IdsUri::new("https://example.org/policy/1").unwrap(),
            policy_type: PolicyType::Set,
            context: None,
            profile: None,
            permissions: vec![Permission {
                uid: None,
                action: OdrlAction::Read,
                constraints: Vec::new(),
                duties: Vec::new(),
                target: None,
                assignee: None,
                assigner: None,
            }],
            prohibitions: Vec::new(),
            obligations: Vec::new(),
            targets: Vec::new(),
            assigner: None,
            assignee: None,
            inherits_from: None,
            conflict: None,
        };

        assert!(OdrlParser::validate(&policy).is_ok());
    }

    #[test]
    fn test_validate_empty_policy() {
        let policy = OdrlPolicy {
            uid: IdsUri::new("https://example.org/policy/empty").unwrap(),
            policy_type: PolicyType::Set,
            context: None,
            profile: None,
            permissions: Vec::new(),
            prohibitions: Vec::new(),
            obligations: Vec::new(),
            targets: Vec::new(),
            assigner: None,
            assignee: None,
            inherits_from: None,
            conflict: None,
        };

        assert!(OdrlParser::validate(&policy).is_err());
    }

    #[test]
    fn test_create_read_permission() {
        let assigner = Party {
            id: IdsUri::new("https://provider.example.org").unwrap(),
            name: "Data Provider".to_string(),
            organization: None,
            contact: None,
            gaiax_participant_id: None,
        };

        let assignee = Party {
            id: IdsUri::new("https://consumer.example.org").unwrap(),
            name: "Data Consumer".to_string(),
            organization: None,
            contact: None,
            gaiax_participant_id: None,
        };

        let target = AssetTarget {
            uid: IdsUri::new("https://example.org/data/dataset1").unwrap(),
            asset_type: None,
            title: Some("Test Dataset".to_string()),
            description: None,
        };

        let policy = OdrlParser::create_read_permission(
            IdsUri::new("https://example.org/policy/read").unwrap(),
            assigner,
            assignee,
            target,
        );

        assert_eq!(policy.policy_type, PolicyType::Agreement);
        assert_eq!(policy.permissions.len(), 1);
        assert_eq!(policy.permissions[0].action, OdrlAction::Read);
        assert!(policy.assigner.is_some());
        assert!(policy.assignee.is_some());
    }

    #[test]
    fn test_roundtrip_serialization() {
        let policy = OdrlPolicy {
            uid: IdsUri::new("https://example.org/policy/test").unwrap(),
            policy_type: PolicyType::Offer,
            context: None,
            profile: None,
            permissions: vec![Permission {
                uid: None,
                action: OdrlAction::Use,
                constraints: Vec::new(),
                duties: Vec::new(),
                target: None,
                assignee: None,
                assigner: None,
            }],
            prohibitions: Vec::new(),
            obligations: Vec::new(),
            targets: Vec::new(),
            assigner: None,
            assignee: None,
            inherits_from: None,
            conflict: None,
        };

        let json = OdrlParser::serialize(&policy).unwrap();
        let parsed = OdrlParser::parse(&json).unwrap();

        assert_eq!(parsed.uid, policy.uid);
        assert_eq!(parsed.policy_type, policy.policy_type);
        assert_eq!(parsed.permissions.len(), policy.permissions.len());
    }
}
