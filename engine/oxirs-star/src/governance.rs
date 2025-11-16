//! Meta-annotations for governance and policy enforcement
//!
//! This module provides governance capabilities for RDF-star annotations,
//! including access control, approval workflows, policy enforcement,
//! compliance tracking, and audit trails.

use crate::annotations::TripleAnnotation;
use crate::cryptographic_provenance::CryptoProvenanceManager;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors related to governance operations
#[derive(Error, Debug)]
pub enum GovernanceError {
    #[error("Access denied for user '{0}': {1}")]
    AccessDenied(String, String),

    #[error("Policy violation: {0}")]
    PolicyViolation(String),

    #[error("Approval required from: {0:?}")]
    ApprovalRequired(Vec<String>),

    #[error("Insufficient permissions: required {required:?}, has {actual:?}")]
    InsufficientPermissions {
        required: Vec<Permission>,
        actual: Vec<Permission>,
    },

    #[error("Compliance check failed: {0}")]
    ComplianceFailed(String),

    #[error("Invalid governance state: {0}")]
    InvalidState(String),
}

/// Permission types for governance
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    /// Can read annotations
    Read,
    /// Can create new annotations
    Create,
    /// Can modify existing annotations
    Update,
    /// Can delete annotations
    Delete,
    /// Can approve annotations
    Approve,
    /// Can set governance policies
    SetPolicy,
    /// Can audit annotations
    Audit,
    /// Can override policies (superuser)
    Override,
    /// Can export sensitive data
    Export,
    /// Can view provenance chains
    ViewProvenance,
}

/// Role with associated permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// Role name
    pub name: String,

    /// Description
    pub description: String,

    /// Permissions granted to this role
    pub permissions: HashSet<Permission>,

    /// Role hierarchy (inherits from these roles)
    pub inherits_from: Vec<String>,
}

impl Role {
    /// Create a new role
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            permissions: HashSet::new(),
            inherits_from: Vec::new(),
        }
    }

    /// Add a permission to this role
    pub fn add_permission(&mut self, permission: Permission) {
        self.permissions.insert(permission);
    }

    /// Check if role has a permission
    pub fn has_permission(&self, permission: &Permission) -> bool {
        self.permissions.contains(permission)
    }

    /// Inherit from another role
    pub fn inherit_from(&mut self, role_name: String) {
        self.inherits_from.push(role_name);
    }
}

/// User with roles and permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    /// User identifier
    pub id: String,

    /// User's assigned roles
    pub roles: HashSet<String>,

    /// Direct permissions (in addition to role permissions)
    pub direct_permissions: HashSet<Permission>,

    /// User groups membership
    pub groups: HashSet<String>,

    /// User metadata
    pub metadata: HashMap<String, String>,
}

impl User {
    /// Create a new user
    pub fn new(id: String) -> Self {
        Self {
            id,
            roles: HashSet::new(),
            direct_permissions: HashSet::new(),
            groups: HashSet::new(),
            metadata: HashMap::new(),
        }
    }

    /// Assign a role to the user
    pub fn assign_role(&mut self, role: String) {
        self.roles.insert(role);
    }

    /// Grant a direct permission
    pub fn grant_permission(&mut self, permission: Permission) {
        self.direct_permissions.insert(permission);
    }

    /// Add to a group
    pub fn add_to_group(&mut self, group: String) {
        self.groups.insert(group);
    }
}

/// Governance policy for annotations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernancePolicy {
    /// Policy name
    pub name: String,

    /// Description
    pub description: String,

    /// Required approvers (user IDs or roles)
    pub required_approvers: Vec<String>,

    /// Minimum number of approvals needed
    pub min_approvals: usize,

    /// Permissions required to create annotations under this policy
    pub creation_permissions: Vec<Permission>,

    /// Permissions required to modify annotations under this policy
    pub modification_permissions: Vec<Permission>,

    /// Whether cryptographic signatures are required
    pub require_signatures: bool,

    /// Mandatory metadata fields
    pub mandatory_fields: Vec<String>,

    /// Retention period (in days)
    pub retention_days: Option<u64>,

    /// Compliance tags required
    pub compliance_tags: Vec<String>,

    /// Auto-approval conditions
    pub auto_approve_conditions: Vec<String>,
}

impl GovernancePolicy {
    /// Create a new governance policy
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            required_approvers: Vec::new(),
            min_approvals: 0,
            creation_permissions: Vec::new(),
            modification_permissions: Vec::new(),
            require_signatures: false,
            mandatory_fields: Vec::new(),
            retention_days: None,
            compliance_tags: Vec::new(),
            auto_approve_conditions: Vec::new(),
        }
    }

    /// Add required approver
    pub fn add_approver(&mut self, approver: String) {
        self.required_approvers.push(approver);
    }

    /// Set minimum approvals
    pub fn set_min_approvals(&mut self, min: usize) {
        self.min_approvals = min;
    }

    /// Require signature
    pub fn require_signature(&mut self) {
        self.require_signatures = true;
    }
}

/// Approval record for governed annotations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRecord {
    /// Approver user ID
    pub approver: String,

    /// Approval timestamp
    pub timestamp: DateTime<Utc>,

    /// Approval decision (approved/rejected)
    pub decision: ApprovalDecision,

    /// Comments
    pub comments: Option<String>,

    /// Cryptographic signature of the approval
    pub signature: Option<String>,
}

/// Approval decision
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ApprovalDecision {
    Approved,
    Rejected,
    ConditionallyApproved { conditions: Vec<String> },
}

/// Governed annotation with policy enforcement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernedAnnotation {
    /// The underlying annotation
    pub annotation: TripleAnnotation,

    /// Applied governance policy
    pub policy: String,

    /// Approval records
    pub approvals: Vec<ApprovalRecord>,

    /// Governance state
    pub state: GovernanceState,

    /// Creator user ID
    pub creator: String,

    /// Last modified by
    pub last_modified_by: Option<String>,

    /// Compliance tags
    pub compliance_tags: HashSet<String>,

    /// Access control list
    pub acl: AccessControlList,

    /// Audit trail
    pub audit_trail: Vec<AuditEvent>,
}

/// Governance state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GovernanceState {
    /// Awaiting approval
    PendingApproval,
    /// Approved and active
    Approved,
    /// Rejected
    Rejected,
    /// Archived
    Archived,
    /// Under review
    UnderReview,
    /// Suspended (temporarily inactive)
    Suspended,
}

/// Access Control List for fine-grained permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlList {
    /// Owner user ID
    pub owner: String,

    /// Users with read access
    pub readers: HashSet<String>,

    /// Users with write access
    pub writers: HashSet<String>,

    /// Groups with access
    pub group_permissions: HashMap<String, Vec<Permission>>,
}

impl AccessControlList {
    /// Create a new ACL
    pub fn new(owner: String) -> Self {
        Self {
            owner,
            readers: HashSet::new(),
            writers: HashSet::new(),
            group_permissions: HashMap::new(),
        }
    }

    /// Grant read access
    pub fn grant_read(&mut self, user: String) {
        self.readers.insert(user);
    }

    /// Grant write access
    pub fn grant_write(&mut self, user: String) {
        self.writers.insert(user);
    }

    /// Check if user can read
    pub fn can_read(&self, user: &str) -> bool {
        user == self.owner || self.readers.contains(user) || self.writers.contains(user)
    }

    /// Check if user can write
    pub fn can_write(&self, user: &str) -> bool {
        user == self.owner || self.writers.contains(user)
    }
}

/// Audit event for tracking governance actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Event type
    pub event_type: AuditEventType,

    /// User who triggered the event
    pub user: String,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Event details
    pub details: HashMap<String, String>,

    /// IP address (if applicable)
    pub ip_address: Option<String>,
}

/// Audit event types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuditEventType {
    Created,
    Modified,
    Deleted,
    Approved,
    Rejected,
    Exported,
    AccessGranted,
    AccessRevoked,
    PolicyChanged,
    StateTransition,
}

/// Governance manager for enforcing policies
#[allow(dead_code)]
pub struct GovernanceManager {
    /// Registered policies
    policies: HashMap<String, GovernancePolicy>,

    /// Registered roles
    roles: HashMap<String, Role>,

    /// Registered users
    users: HashMap<String, User>,

    /// Governed annotations
    annotations: HashMap<String, GovernedAnnotation>,

    /// Cryptographic provenance manager
    crypto_manager: CryptoProvenanceManager,
}

impl GovernanceManager {
    /// Create a new governance manager
    pub fn new() -> Self {
        let mut manager = Self {
            policies: HashMap::new(),
            roles: HashMap::new(),
            users: HashMap::new(),
            annotations: HashMap::new(),
            crypto_manager: CryptoProvenanceManager::new(),
        };

        // Initialize default roles
        manager.initialize_default_roles();
        manager
    }

    /// Initialize default roles
    fn initialize_default_roles(&mut self) {
        // Admin role
        let mut admin = Role::new(
            "admin".to_string(),
            "Administrator with full access".to_string(),
        );
        admin.add_permission(Permission::Read);
        admin.add_permission(Permission::Create);
        admin.add_permission(Permission::Update);
        admin.add_permission(Permission::Delete);
        admin.add_permission(Permission::Approve);
        admin.add_permission(Permission::SetPolicy);
        admin.add_permission(Permission::Audit);
        admin.add_permission(Permission::Override);
        admin.add_permission(Permission::Export);
        admin.add_permission(Permission::ViewProvenance);
        self.register_role(admin);

        // Editor role
        let mut editor = Role::new(
            "editor".to_string(),
            "Can create and modify annotations".to_string(),
        );
        editor.add_permission(Permission::Read);
        editor.add_permission(Permission::Create);
        editor.add_permission(Permission::Update);
        editor.add_permission(Permission::ViewProvenance);
        self.register_role(editor);

        // Reviewer role
        let mut reviewer = Role::new(
            "reviewer".to_string(),
            "Can approve annotations".to_string(),
        );
        reviewer.add_permission(Permission::Read);
        reviewer.add_permission(Permission::Approve);
        reviewer.add_permission(Permission::ViewProvenance);
        self.register_role(reviewer);

        // Viewer role
        let mut viewer = Role::new("viewer".to_string(), "Read-only access".to_string());
        viewer.add_permission(Permission::Read);
        self.register_role(viewer);

        info!("Initialized default roles: admin, editor, reviewer, viewer");
    }

    /// Register a new role
    pub fn register_role(&mut self, role: Role) {
        debug!("Registering role: {}", role.name);
        self.roles.insert(role.name.clone(), role);
    }

    /// Register a new user
    pub fn register_user(&mut self, user: User) {
        debug!("Registering user: {}", user.id);
        self.users.insert(user.id.clone(), user);
    }

    /// Register a policy
    pub fn register_policy(&mut self, policy: GovernancePolicy) {
        info!("Registering governance policy: {}", policy.name);
        self.policies.insert(policy.name.clone(), policy);
    }

    /// Get all permissions for a user (including inherited from roles)
    pub fn get_user_permissions(&self, user_id: &str) -> HashSet<Permission> {
        let mut permissions = HashSet::new();

        if let Some(user) = self.users.get(user_id) {
            // Add direct permissions
            permissions.extend(user.direct_permissions.clone());

            // Add role permissions
            for role_name in &user.roles {
                if let Some(role) = self.roles.get(role_name) {
                    permissions.extend(role.permissions.clone());

                    // Handle role inheritance
                    for inherited_role_name in &role.inherits_from {
                        if let Some(inherited_role) = self.roles.get(inherited_role_name) {
                            permissions.extend(inherited_role.permissions.clone());
                        }
                    }
                }
            }
        }

        permissions
    }

    /// Check if user has required permissions
    pub fn check_permissions(
        &self,
        user_id: &str,
        required: &[Permission],
    ) -> Result<(), GovernanceError> {
        let user_permissions = self.get_user_permissions(user_id);

        // Check for Override permission (superuser)
        if user_permissions.contains(&Permission::Override) {
            return Ok(());
        }

        for perm in required {
            if !user_permissions.contains(perm) {
                return Err(GovernanceError::InsufficientPermissions {
                    required: required.to_vec(),
                    actual: user_permissions.into_iter().collect(),
                });
            }
        }

        Ok(())
    }

    /// Create a governed annotation
    pub fn create_annotation(
        &mut self,
        annotation_id: String,
        annotation: TripleAnnotation,
        policy_name: String,
        creator_id: String,
    ) -> Result<(), GovernanceError> {
        // Check if policy exists
        let policy = self
            .policies
            .get(&policy_name)
            .ok_or_else(|| {
                GovernanceError::InvalidState(format!("Policy '{}' not found", policy_name))
            })?
            .clone();

        // Check creator permissions
        self.check_permissions(&creator_id, &policy.creation_permissions)?;

        // Validate mandatory fields
        for field in &policy.mandatory_fields {
            match field.as_str() {
                "confidence" => {
                    if annotation.confidence.is_none() {
                        return Err(GovernanceError::PolicyViolation(
                            "Mandatory field 'confidence' missing".to_string(),
                        ));
                    }
                }
                "source" => {
                    if annotation.source.is_none() {
                        return Err(GovernanceError::PolicyViolation(
                            "Mandatory field 'source' missing".to_string(),
                        ));
                    }
                }
                _ => {}
            }
        }

        // Create ACL
        let acl = AccessControlList::new(creator_id.clone());

        // Create governed annotation
        let governed = GovernedAnnotation {
            annotation,
            policy: policy_name,
            approvals: Vec::new(),
            state: if policy.min_approvals > 0 {
                GovernanceState::PendingApproval
            } else {
                GovernanceState::Approved
            },
            creator: creator_id.clone(),
            last_modified_by: None,
            compliance_tags: policy.compliance_tags.iter().cloned().collect(),
            acl,
            audit_trail: vec![AuditEvent {
                event_type: AuditEventType::Created,
                user: creator_id,
                timestamp: Utc::now(),
                details: HashMap::new(),
                ip_address: None,
            }],
        };

        info!(
            "Created governed annotation '{}' under policy '{}'",
            annotation_id, governed.policy
        );
        self.annotations.insert(annotation_id, governed);

        Ok(())
    }

    /// Submit an approval for an annotation
    pub fn approve_annotation(
        &mut self,
        annotation_id: &str,
        approver_id: &str,
        decision: ApprovalDecision,
        comments: Option<String>,
    ) -> Result<(), GovernanceError> {
        // Check approver has Approve permission
        self.check_permissions(approver_id, &[Permission::Approve])?;

        let annotation = self.annotations.get_mut(annotation_id).ok_or_else(|| {
            GovernanceError::InvalidState(format!("Annotation '{}' not found", annotation_id))
        })?;

        // Add approval record
        let approval = ApprovalRecord {
            approver: approver_id.to_string(),
            timestamp: Utc::now(),
            decision: decision.clone(),
            comments,
            signature: None, // Can be added with cryptographic signatures
        };

        annotation.approvals.push(approval);

        // Add audit event
        annotation.audit_trail.push(AuditEvent {
            event_type: match decision {
                ApprovalDecision::Approved => AuditEventType::Approved,
                ApprovalDecision::Rejected => AuditEventType::Rejected,
                _ => AuditEventType::Modified,
            },
            user: approver_id.to_string(),
            timestamp: Utc::now(),
            details: HashMap::new(),
            ip_address: None,
        });

        // Check if we have enough approvals
        let policy = self.policies.get(&annotation.policy).unwrap();
        let approved_count = annotation
            .approvals
            .iter()
            .filter(|a| matches!(a.decision, ApprovalDecision::Approved))
            .count();

        if approved_count >= policy.min_approvals {
            annotation.state = GovernanceState::Approved;
            info!(
                "Annotation '{}' approved after {} approvals",
                annotation_id, approved_count
            );
        } else if matches!(decision, ApprovalDecision::Rejected) {
            annotation.state = GovernanceState::Rejected;
            warn!("Annotation '{}' rejected by {}", annotation_id, approver_id);
        }

        Ok(())
    }

    /// Get audit trail for an annotation
    pub fn get_audit_trail(&self, annotation_id: &str) -> Option<&Vec<AuditEvent>> {
        self.annotations.get(annotation_id).map(|a| &a.audit_trail)
    }

    /// Check compliance status
    pub fn check_compliance(&self, annotation_id: &str) -> Result<bool, GovernanceError> {
        let annotation = self.annotations.get(annotation_id).ok_or_else(|| {
            GovernanceError::InvalidState(format!("Annotation '{}' not found", annotation_id))
        })?;

        let policy = self.policies.get(&annotation.policy).ok_or_else(|| {
            GovernanceError::InvalidState(format!("Policy '{}' not found", annotation.policy))
        })?;

        // Check all required compliance tags are present
        for required_tag in &policy.compliance_tags {
            if !annotation.compliance_tags.contains(required_tag) {
                return Err(GovernanceError::ComplianceFailed(format!(
                    "Missing required compliance tag: {}",
                    required_tag
                )));
            }
        }

        Ok(true)
    }

    /// Generate compliance report
    pub fn generate_compliance_report(&self) -> ComplianceReport {
        let total_annotations = self.annotations.len();
        let mut compliant = 0;
        let mut non_compliant = 0;
        let mut violations = Vec::new();

        for id in self.annotations.keys() {
            match self.check_compliance(id) {
                Ok(_) => compliant += 1,
                Err(e) => {
                    non_compliant += 1;
                    violations.push((id.clone(), e.to_string()));
                }
            }
        }

        ComplianceReport {
            total_annotations,
            compliant,
            non_compliant,
            compliance_rate: if total_annotations > 0 {
                compliant as f64 / total_annotations as f64
            } else {
                1.0
            },
            violations,
            generated_at: Utc::now(),
        }
    }
}

impl Default for GovernanceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    pub total_annotations: usize,
    pub compliant: usize,
    pub non_compliant: usize,
    pub compliance_rate: f64,
    pub violations: Vec<(String, String)>,
    pub generated_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_permissions() {
        let mut role = Role::new("test_role".to_string(), "Test role".to_string());
        role.add_permission(Permission::Read);
        role.add_permission(Permission::Create);

        assert!(role.has_permission(&Permission::Read));
        assert!(role.has_permission(&Permission::Create));
        assert!(!role.has_permission(&Permission::Delete));
    }

    #[test]
    fn test_user_roles() {
        let mut user = User::new("user1".to_string());
        user.assign_role("editor".to_string());
        user.grant_permission(Permission::Export);

        assert!(user.roles.contains("editor"));
        assert!(user.direct_permissions.contains(&Permission::Export));
    }

    #[test]
    fn test_governance_manager_permissions() {
        let mut manager = GovernanceManager::new();

        let mut user = User::new("user1".to_string());
        user.assign_role("editor".to_string());
        manager.register_user(user);

        let perms = manager.get_user_permissions("user1");
        assert!(perms.contains(&Permission::Read));
        assert!(perms.contains(&Permission::Create));
        assert!(!perms.contains(&Permission::Delete));
    }

    #[test]
    fn test_create_governed_annotation() {
        let mut manager = GovernanceManager::new();

        // Create user
        let mut user = User::new("creator1".to_string());
        user.assign_role("editor".to_string());
        manager.register_user(user);

        // Create policy
        let mut policy =
            GovernancePolicy::new("test_policy".to_string(), "Test policy".to_string());
        policy.creation_permissions = vec![Permission::Create];
        manager.register_policy(policy);

        // Create annotation
        let annotation = TripleAnnotation::new();
        let result = manager.create_annotation(
            "anno1".to_string(),
            annotation,
            "test_policy".to_string(),
            "creator1".to_string(),
        );

        assert!(result.is_ok());
        assert!(manager.annotations.contains_key("anno1"));
    }

    #[test]
    fn test_approval_workflow() {
        let mut manager = GovernanceManager::new();

        // Create users
        let mut creator = User::new("creator1".to_string());
        creator.assign_role("editor".to_string());
        manager.register_user(creator);

        let mut approver = User::new("approver1".to_string());
        approver.assign_role("reviewer".to_string());
        manager.register_user(approver);

        // Create policy requiring approval
        let mut policy = GovernancePolicy::new(
            "approval_policy".to_string(),
            "Requires approval".to_string(),
        );
        policy.creation_permissions = vec![Permission::Create];
        policy.min_approvals = 1;
        manager.register_policy(policy);

        // Create annotation
        let annotation = TripleAnnotation::new();
        manager
            .create_annotation(
                "anno1".to_string(),
                annotation,
                "approval_policy".to_string(),
                "creator1".to_string(),
            )
            .unwrap();

        // Check initial state
        assert_eq!(
            manager.annotations.get("anno1").unwrap().state,
            GovernanceState::PendingApproval
        );

        // Approve
        manager
            .approve_annotation("anno1", "approver1", ApprovalDecision::Approved, None)
            .unwrap();

        // Check approved state
        assert_eq!(
            manager.annotations.get("anno1").unwrap().state,
            GovernanceState::Approved
        );
    }

    #[test]
    fn test_access_control() {
        let mut acl = AccessControlList::new("owner1".to_string());
        acl.grant_read("user1".to_string());
        acl.grant_write("user2".to_string());

        assert!(acl.can_read("owner1"));
        assert!(acl.can_write("owner1"));
        assert!(acl.can_read("user1"));
        assert!(!acl.can_write("user1"));
        assert!(acl.can_read("user2"));
        assert!(acl.can_write("user2"));
        assert!(!acl.can_read("user3"));
    }

    #[test]
    fn test_compliance_report() {
        let mut manager = GovernanceManager::new();

        let mut user = User::new("user1".to_string());
        user.assign_role("editor".to_string());
        manager.register_user(user);

        let mut policy = GovernancePolicy::new("policy1".to_string(), "Policy 1".to_string());
        policy.creation_permissions = vec![Permission::Create];
        policy.compliance_tags = vec!["gdpr".to_string()];
        manager.register_policy(policy);

        let annotation = TripleAnnotation::new();
        manager
            .create_annotation(
                "anno1".to_string(),
                annotation,
                "policy1".to_string(),
                "user1".to_string(),
            )
            .unwrap();

        let report = manager.generate_compliance_report();
        assert_eq!(report.total_annotations, 1);
        assert!(report.compliance_rate <= 1.0);
    }
}
