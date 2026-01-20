//! Regional Data Residency Policy
//!
//! Manages data placement according to legal jurisdictions

use crate::ids::types::{AdequacyStatus, IdsError, IdsResult, Jurisdiction};
use serde::{Deserialize, Serialize};

/// Geographic Region
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Region {
    /// ISO 3166-1 alpha-2 country code
    pub code: String,

    /// Region name
    pub name: String,

    /// Legal jurisdiction
    pub jurisdiction: Jurisdiction,

    /// GDPR adequacy decision status
    pub adequacy: AdequacyStatus,
}

impl Region {
    /// EU Member States
    pub fn eu_member(code: impl Into<String>, name: impl Into<String>) -> Self {
        let code_str = code.into();
        Self {
            code: code_str.clone(),
            name: name.into(),
            jurisdiction: Jurisdiction {
                country_code: code_str, // Use actual country code for EEA check
                legal_framework: vec!["GDPR".to_string()],
                data_protection_authority: None,
            },
            adequacy: AdequacyStatus::Adequate,
        }
    }

    /// Japan (with adequacy decision)
    pub fn japan() -> Self {
        Self {
            code: "JP".to_string(),
            name: "Japan".to_string(),
            jurisdiction: Jurisdiction {
                country_code: "JP".to_string(),
                legal_framework: vec!["APPI".to_string()],
                data_protection_authority: Some(
                    "Personal Information Protection Commission".to_string(),
                ),
            },
            adequacy: AdequacyStatus::Adequate,
        }
    }
}

/// Data Residency Policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidencyPolicy {
    /// Policy ID
    pub policy_id: String,

    /// Allowed regions for data storage
    pub allowed_regions: Vec<Region>,

    /// Prohibited regions
    pub prohibited_regions: Vec<Region>,

    /// Storage requirement
    pub storage_requirement: StorageRequirement,

    /// Transfer restrictions
    pub transfer_restrictions: Vec<TransferRestriction>,

    /// GDPR adequacy required
    pub gdpr_adequacy_required: bool,
}

/// Storage Requirement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageRequirement {
    /// Data must remain in specified regions
    MustRemainIn,

    /// Data should preferably be in regions
    PreferentiallyIn,

    /// Data must not be in prohibited regions
    MustNotBeIn,
}

/// Transfer Restriction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferRestriction {
    /// Source region
    pub from_region: Option<Region>,

    /// Destination region
    pub to_region: Region,

    /// Transfer allowed
    pub allowed: bool,

    /// Required safeguards
    pub required_safeguards: Vec<String>,
}

/// Residency Enforcer
pub struct ResidencyEnforcer {
    policies: Vec<ResidencyPolicy>,
}

impl ResidencyEnforcer {
    /// Create a new residency enforcer
    pub fn new() -> Self {
        Self {
            policies: Vec::new(),
        }
    }

    /// Add a residency policy
    pub fn add_policy(&mut self, policy: ResidencyPolicy) {
        self.policies.push(policy);
    }

    /// Check if data placement is allowed
    pub fn check_placement(&self, region: &Region) -> IdsResult<bool> {
        for policy in &self.policies {
            // Check prohibited regions
            if policy
                .prohibited_regions
                .iter()
                .any(|r| r.code == region.code)
            {
                return Err(IdsError::ResidencyViolation(format!(
                    "Data storage prohibited in region: {}",
                    region.name
                )));
            }

            // Check GDPR adequacy if required
            if policy.gdpr_adequacy_required && region.adequacy != AdequacyStatus::Adequate {
                return Err(IdsError::GdprViolation(format!(
                    "Region {} does not have GDPR adequacy decision",
                    region.name
                )));
            }
        }

        Ok(true)
    }

    /// Check if transfer is allowed
    pub fn check_transfer(&self, from: &Region, to: &Region) -> IdsResult<bool> {
        for policy in &self.policies {
            for restriction in &policy.transfer_restrictions {
                if restriction.to_region.code == to.code && !restriction.allowed {
                    return Err(IdsError::ResidencyViolation(format!(
                        "Data transfer to {} is not allowed",
                        to.name
                    )));
                }
            }

            // GDPR adequacy check for transfers
            if policy.gdpr_adequacy_required && to.adequacy != AdequacyStatus::Adequate {
                return Err(IdsError::GdprViolation(format!(
                    "Cannot transfer to {} without adequate protection",
                    to.name
                )));
            }
        }

        Ok(true)
    }
}

impl Default for ResidencyEnforcer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eu_region() {
        let region = Region::eu_member("DE", "Germany");
        assert_eq!(region.code, "DE");
        assert_eq!(region.adequacy, AdequacyStatus::Adequate);
    }

    #[test]
    fn test_japan_region() {
        let region = Region::japan();
        assert_eq!(region.code, "JP");
        assert_eq!(region.adequacy, AdequacyStatus::Adequate);
    }

    #[test]
    fn test_residency_enforcer() {
        let mut enforcer = ResidencyEnforcer::new();

        let policy = ResidencyPolicy {
            policy_id: "eu-only".to_string(),
            allowed_regions: vec![Region::eu_member("DE", "Germany")],
            prohibited_regions: Vec::new(),
            storage_requirement: StorageRequirement::MustRemainIn,
            transfer_restrictions: Vec::new(),
            gdpr_adequacy_required: true,
        };

        enforcer.add_policy(policy);

        let de = Region::eu_member("DE", "Germany");
        assert!(enforcer.check_placement(&de).is_ok());

        let us = Region {
            code: "US".to_string(),
            name: "United States".to_string(),
            jurisdiction: Jurisdiction {
                country_code: "US".to_string(),
                legal_framework: vec![],
                data_protection_authority: None,
            },
            adequacy: AdequacyStatus::NotAdequate,
        };

        assert!(enforcer.check_placement(&us).is_err());
    }
}
