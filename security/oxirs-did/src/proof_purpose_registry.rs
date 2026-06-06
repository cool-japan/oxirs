use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ProofPurposeRegistry {
    purposes: HashMap<String, PurposeMetadata>,
}

#[derive(Debug, Clone)]
pub struct PurposeMetadata {
    pub purpose_id: String,
    pub label: String,
    pub description: String,
    pub requires_challenge: bool,
    pub recommends_domain: bool,
}

impl Default for ProofPurposeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofPurposeRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            purposes: HashMap::new(),
        };

        registry.register(PurposeMetadata {
            purpose_id: "authentication".to_string(),
            label: "Authentication".to_string(),
            description: "Prove the identity of the DID subject".to_string(),
            requires_challenge: true,
            recommends_domain: true,
        });

        registry.register(PurposeMetadata {
            purpose_id: "assertionMethod".to_string(),
            label: "Assertion Method".to_string(),
            description: "Make verifiable claims about a subject".to_string(),
            requires_challenge: false,
            recommends_domain: false,
        });

        registry.register(PurposeMetadata {
            purpose_id: "keyAgreement".to_string(),
            label: "Key Agreement".to_string(),
            description: "Establish a shared secret for encrypted communication".to_string(),
            requires_challenge: false,
            recommends_domain: false,
        });

        registry.register(PurposeMetadata {
            purpose_id: "capabilityInvocation".to_string(),
            label: "Capability Invocation".to_string(),
            description: "Exercise an authorization capability".to_string(),
            requires_challenge: false,
            recommends_domain: true,
        });

        registry.register(PurposeMetadata {
            purpose_id: "capabilityDelegation".to_string(),
            label: "Capability Delegation".to_string(),
            description: "Delegate an authorization capability".to_string(),
            requires_challenge: false,
            recommends_domain: true,
        });

        registry
    }

    pub fn register(&mut self, metadata: PurposeMetadata) {
        self.purposes.insert(metadata.purpose_id.clone(), metadata);
    }

    pub fn get(&self, purpose_id: &str) -> Option<&PurposeMetadata> {
        self.purposes.get(purpose_id)
    }

    pub fn is_registered(&self, purpose_id: &str) -> bool {
        self.purposes.contains_key(purpose_id)
    }

    pub fn all_purpose_ids(&self) -> Vec<String> {
        let mut v: Vec<String> = self.purposes.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn count(&self) -> usize {
        self.purposes.len()
    }

    pub fn unregister(&mut self, purpose_id: &str) -> bool {
        self.purposes.remove(purpose_id).is_some()
    }
}
