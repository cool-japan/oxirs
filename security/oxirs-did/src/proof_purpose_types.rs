use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProofPurposeKind {
    Authentication,
    AssertionMethod,
    KeyAgreement,
    CapabilityInvocation,
    CapabilityDelegation,
    Custom(String),
}

impl ProofPurposeKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Authentication => "authentication",
            Self::AssertionMethod => "assertionMethod",
            Self::KeyAgreement => "keyAgreement",
            Self::CapabilityInvocation => "capabilityInvocation",
            Self::CapabilityDelegation => "capabilityDelegation",
            Self::Custom(uri) => uri.as_str(),
        }
    }

    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "authentication" => Self::Authentication,
            "assertionMethod" => Self::AssertionMethod,
            "keyAgreement" => Self::KeyAgreement,
            "capabilityInvocation" => Self::CapabilityInvocation,
            "capabilityDelegation" => Self::CapabilityDelegation,
            other => Self::Custom(other.to_string()),
        }
    }

    pub fn is_standard(&self) -> bool {
        !matches!(self, Self::Custom(_))
    }
}

impl std::fmt::Display for ProofPurposeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone)]
pub struct DidRelationships {
    pub controller: String,
    relationships: HashMap<String, HashSet<String>>,
}

impl DidRelationships {
    pub fn new(controller: impl Into<String>) -> Self {
        Self {
            controller: controller.into(),
            relationships: HashMap::new(),
        }
    }

    pub fn add_authentication(&mut self, vm_id: impl Into<String>) {
        self.add(ProofPurposeKind::Authentication, vm_id);
    }

    pub fn add_assertion_method(&mut self, vm_id: impl Into<String>) {
        self.add(ProofPurposeKind::AssertionMethod, vm_id);
    }

    pub fn add_key_agreement(&mut self, vm_id: impl Into<String>) {
        self.add(ProofPurposeKind::KeyAgreement, vm_id);
    }

    pub fn add_capability_invocation(&mut self, vm_id: impl Into<String>) {
        self.add(ProofPurposeKind::CapabilityInvocation, vm_id);
    }

    pub fn add_capability_delegation(&mut self, vm_id: impl Into<String>) {
        self.add(ProofPurposeKind::CapabilityDelegation, vm_id);
    }

    pub fn add_custom(&mut self, purpose_uri: impl Into<String>, vm_id: impl Into<String>) {
        let purpose = ProofPurposeKind::Custom(purpose_uri.into());
        self.add(purpose, vm_id);
    }

    pub fn add(&mut self, purpose: ProofPurposeKind, vm_id: impl Into<String>) {
        self.relationships
            .entry(purpose.as_str().to_string())
            .or_default()
            .insert(vm_id.into());
    }

    pub fn is_authorised(&self, purpose: &ProofPurposeKind, vm_id: &str) -> bool {
        self.relationships
            .get(purpose.as_str())
            .is_some_and(|set| set.contains(vm_id))
    }

    pub fn methods_for(&self, purpose: &ProofPurposeKind) -> Vec<String> {
        self.relationships
            .get(purpose.as_str())
            .map_or_else(Vec::new, |set| {
                let mut v: Vec<String> = set.iter().cloned().collect();
                v.sort();
                v
            })
    }

    pub fn purposes(&self) -> Vec<String> {
        let mut v: Vec<String> = self.relationships.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn total_entries(&self) -> usize {
        self.relationships.values().map(|s| s.len()).sum()
    }

    pub fn remove(&mut self, purpose: &ProofPurposeKind, vm_id: &str) -> bool {
        if let Some(set) = self.relationships.get_mut(purpose.as_str()) {
            let removed = set.remove(vm_id);
            if set.is_empty() {
                self.relationships.remove(purpose.as_str());
            }
            removed
        } else {
            false
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProofEntry {
    pub purpose: ProofPurposeKind,
    pub verification_method: String,
    pub created_ms: u64,
    pub expires_ms: Option<u64>,
    pub challenge: Option<String>,
    pub domain: Option<String>,
    pub nonce: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ValidationContext {
    pub challenge: Option<String>,
    pub domain: Option<String>,
    pub current_time_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PurposeError {
    Unauthorised {
        purpose: String,
        verification_method: String,
    },
    Expired {
        expires_ms: u64,
        current_ms: u64,
    },
    FutureProof {
        created_ms: u64,
        current_ms: u64,
    },
    ChallengeMismatch {
        expected: String,
        actual: String,
    },
    ChallengeMissing,
    DomainMismatch {
        expected: String,
        actual: String,
    },
    UnknownCustomPurpose(String),
    ChainError {
        index: usize,
        reason: String,
    },
    EmptyChain,
}

impl std::fmt::Display for PurposeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unauthorised {
                purpose,
                verification_method,
            } => write!(
                f,
                "Verification method '{verification_method}' is not authorised for purpose '{purpose}'"
            ),
            Self::Expired {
                expires_ms,
                current_ms,
            } => write!(
                f,
                "Proof expired at {expires_ms} ms, current time is {current_ms} ms"
            ),
            Self::FutureProof {
                created_ms,
                current_ms,
            } => write!(
                f,
                "Proof created at {created_ms} ms is in the future (current: {current_ms} ms)"
            ),
            Self::ChallengeMismatch { expected, actual } => {
                write!(
                    f,
                    "Challenge mismatch: expected '{expected}', got '{actual}'"
                )
            }
            Self::ChallengeMissing => write!(f, "Authentication proof requires a challenge"),
            Self::DomainMismatch { expected, actual } => {
                write!(f, "Domain mismatch: expected '{expected}', got '{actual}'")
            }
            Self::UnknownCustomPurpose(uri) => {
                write!(f, "Unknown custom proof purpose: '{uri}'")
            }
            Self::ChainError { index, reason } => {
                write!(f, "Proof chain error at index {index}: {reason}")
            }
            Self::EmptyChain => write!(f, "Proof chain must contain at least one entry"),
        }
    }
}

impl std::error::Error for PurposeError {}

pub trait CustomPurposeValidator: Send + Sync {
    fn validate(
        &self,
        entry: &ProofEntry,
        relationships: &DidRelationships,
        context: &ValidationContext,
    ) -> Result<(), String>;
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub valid: bool,
    pub purpose: String,
    pub verification_method: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct ChainValidationResult {
    pub valid: bool,
    pub entries: Vec<ValidationResult>,
    pub chain_length: usize,
    pub error: Option<String>,
}
