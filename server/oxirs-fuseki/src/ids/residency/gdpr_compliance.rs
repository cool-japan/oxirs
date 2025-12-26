//! GDPR Compliance Checker
//!
//! Validates GDPR Article 44-49 compliance for international data transfers

use super::Region;
use crate::ids::types::{AdequacyStatus, IdsError, IdsResult};

/// GDPR Articles related to data transfer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdprArticle {
    /// Art. 45: Transfers based on adequacy decision
    Article45,

    /// Art. 46: Transfers with appropriate safeguards
    Article46,

    /// Art. 47: Binding corporate rules
    Article47,

    /// Art. 49: Derogations for specific situations
    Article49,
}

/// GDPR Compliance Checker
pub struct GdprComplianceChecker;

impl GdprComplianceChecker {
    /// Check if transfer complies with GDPR
    pub fn check_transfer_compliance(from: &Region, to: &Region) -> IdsResult<GdprArticle> {
        // Article 45: Adequacy decision
        if to.adequacy == AdequacyStatus::Adequate {
            return Ok(GdprArticle::Article45);
        }

        // Article 46: Appropriate safeguards (e.g., Standard Contractual Clauses)
        if from.jurisdiction.country_code == "EU" {
            // TODO: Check for SCCs, BCRs, or other safeguards
            return Ok(GdprArticle::Article46);
        }

        Err(IdsError::GdprViolation(format!(
            "No legal basis for transfer from {} to {}",
            from.name, to.name
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gdpr_adequacy_transfer() {
        let eu = Region::eu_member("DE", "Germany");
        let jp = Region::japan();

        let result = GdprComplianceChecker::check_transfer_compliance(&eu, &jp);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), GdprArticle::Article45);
    }
}
