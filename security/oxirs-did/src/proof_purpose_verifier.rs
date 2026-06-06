use std::collections::HashMap;

use crate::proof_purpose_types::{
    ChainValidationResult, CustomPurposeValidator, DidRelationships, ProofEntry, ProofPurposeKind,
    PurposeError, ValidationContext, ValidationResult,
};

pub struct ProofPurposeValidator {
    custom_validators: HashMap<String, Box<dyn CustomPurposeValidator>>,
    max_clock_skew_ms: u64,
}

impl Default for ProofPurposeValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofPurposeValidator {
    pub fn new() -> Self {
        Self {
            custom_validators: HashMap::new(),
            max_clock_skew_ms: 300_000,
        }
    }

    pub fn with_max_clock_skew_ms(mut self, ms: u64) -> Self {
        self.max_clock_skew_ms = ms;
        self
    }

    pub fn register_custom(
        &mut self,
        purpose_uri: impl Into<String>,
        validator: Box<dyn CustomPurposeValidator>,
    ) {
        self.custom_validators.insert(purpose_uri.into(), validator);
    }

    pub fn registered_custom_purposes(&self) -> Vec<String> {
        let mut v: Vec<String> = self.custom_validators.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn validate(
        &self,
        entry: &ProofEntry,
        relationships: &DidRelationships,
        context: &ValidationContext,
    ) -> Result<ValidationResult, PurposeError> {
        if !relationships.is_authorised(&entry.purpose, &entry.verification_method) {
            return Err(PurposeError::Unauthorised {
                purpose: entry.purpose.as_str().to_string(),
                verification_method: entry.verification_method.clone(),
            });
        }

        if let Some(expires_ms) = entry.expires_ms {
            if context.current_time_ms > expires_ms {
                return Err(PurposeError::Expired {
                    expires_ms,
                    current_ms: context.current_time_ms,
                });
            }
        }

        if entry.created_ms > context.current_time_ms + self.max_clock_skew_ms {
            return Err(PurposeError::FutureProof {
                created_ms: entry.created_ms,
                current_ms: context.current_time_ms,
            });
        }

        match &entry.purpose {
            ProofPurposeKind::Authentication => {
                self.validate_authentication(entry, context)?;
            }
            ProofPurposeKind::AssertionMethod => {
                self.validate_assertion_method(entry, context)?;
            }
            ProofPurposeKind::KeyAgreement => {
                self.validate_key_agreement(entry, context)?;
            }
            ProofPurposeKind::CapabilityInvocation => {
                self.validate_capability_invocation(entry, context)?;
            }
            ProofPurposeKind::CapabilityDelegation => {
                self.validate_capability_delegation(entry, context)?;
            }
            ProofPurposeKind::Custom(uri) => {
                self.validate_custom(uri, entry, relationships, context)?;
            }
        }

        Ok(ValidationResult {
            valid: true,
            purpose: entry.purpose.as_str().to_string(),
            verification_method: entry.verification_method.clone(),
            message: format!(
                "Proof purpose '{}' validated successfully for method '{}'",
                entry.purpose.as_str(),
                entry.verification_method
            ),
        })
    }

    fn validate_authentication(
        &self,
        entry: &ProofEntry,
        context: &ValidationContext,
    ) -> Result<(), PurposeError> {
        if let Some(ref expected) = context.challenge {
            match &entry.challenge {
                Some(actual) if actual == expected => {}
                Some(actual) => {
                    return Err(PurposeError::ChallengeMismatch {
                        expected: expected.clone(),
                        actual: actual.clone(),
                    });
                }
                None => {
                    return Err(PurposeError::ChallengeMissing);
                }
            }
        }
        self.check_domain(entry, context)?;
        Ok(())
    }

    fn validate_assertion_method(
        &self,
        entry: &ProofEntry,
        context: &ValidationContext,
    ) -> Result<(), PurposeError> {
        self.check_domain(entry, context)?;
        Ok(())
    }

    fn validate_key_agreement(
        &self,
        entry: &ProofEntry,
        context: &ValidationContext,
    ) -> Result<(), PurposeError> {
        self.check_domain(entry, context)?;
        Ok(())
    }

    fn validate_capability_invocation(
        &self,
        entry: &ProofEntry,
        context: &ValidationContext,
    ) -> Result<(), PurposeError> {
        self.check_domain(entry, context)?;
        Ok(())
    }

    fn validate_capability_delegation(
        &self,
        entry: &ProofEntry,
        context: &ValidationContext,
    ) -> Result<(), PurposeError> {
        self.check_domain(entry, context)?;
        Ok(())
    }

    fn validate_custom(
        &self,
        uri: &str,
        entry: &ProofEntry,
        relationships: &DidRelationships,
        context: &ValidationContext,
    ) -> Result<(), PurposeError> {
        if let Some(validator) = self.custom_validators.get(uri) {
            validator
                .validate(entry, relationships, context)
                .map_err(|reason| PurposeError::ChainError { index: 0, reason })
        } else {
            Err(PurposeError::UnknownCustomPurpose(uri.to_string()))
        }
    }

    fn check_domain(
        &self,
        entry: &ProofEntry,
        context: &ValidationContext,
    ) -> Result<(), PurposeError> {
        if let Some(ref expected) = context.domain {
            if let Some(ref actual) = entry.domain {
                if actual != expected {
                    return Err(PurposeError::DomainMismatch {
                        expected: expected.clone(),
                        actual: actual.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn validate_chain(
        &self,
        chain: &[ProofEntry],
        relationships: &DidRelationships,
        context: &ValidationContext,
    ) -> Result<ChainValidationResult, PurposeError> {
        if chain.is_empty() {
            return Err(PurposeError::EmptyChain);
        }

        let mut results = Vec::with_capacity(chain.len());
        let mut prev_created_ms: u64 = 0;

        for (i, entry) in chain.iter().enumerate() {
            if entry.created_ms < prev_created_ms {
                return Err(PurposeError::ChainError {
                    index: i,
                    reason: format!(
                        "Proof at index {} has created_ms={} which is before previous entry's created_ms={}",
                        i, entry.created_ms, prev_created_ms
                    ),
                });
            }
            prev_created_ms = entry.created_ms;

            match self.validate(entry, relationships, context) {
                Ok(result) => results.push(result),
                Err(e) => {
                    return Err(PurposeError::ChainError {
                        index: i,
                        reason: e.to_string(),
                    });
                }
            }
        }

        Ok(ChainValidationResult {
            valid: true,
            chain_length: chain.len(),
            entries: results,
            error: None,
        })
    }

    pub fn requires_challenge(purpose: &ProofPurposeKind) -> bool {
        matches!(purpose, ProofPurposeKind::Authentication)
    }

    pub fn describe_purpose(purpose: &ProofPurposeKind) -> &'static str {
        match purpose {
            ProofPurposeKind::Authentication => {
                "Prove the identity of the DID subject via challenge-response"
            }
            ProofPurposeKind::AssertionMethod => {
                "Make verifiable claims about a subject (e.g. issue credentials)"
            }
            ProofPurposeKind::KeyAgreement => {
                "Establish a shared secret for encrypted communication"
            }
            ProofPurposeKind::CapabilityInvocation => {
                "Exercise an authorization capability granted by a controller"
            }
            ProofPurposeKind::CapabilityDelegation => {
                "Delegate an authorization capability to another party"
            }
            ProofPurposeKind::Custom(_) => "User-defined proof purpose",
        }
    }

    pub fn relationship_name(purpose: &ProofPurposeKind) -> &str {
        purpose.as_str()
    }
}
