//! GDPR Compliance Checker
//!
//! Validates GDPR Article 44-49 compliance for international data transfers.
//! <https://gdpr-info.eu/chapter-5/>

use super::Region;
use crate::ids::types::{AdequacyStatus, IdsError, IdsResult, IdsUri, Party};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// GDPR Articles related to data transfer (Chapter V)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

impl GdprArticle {
    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            GdprArticle::Article45 => "Adequacy Decision",
            GdprArticle::Article46 => "Appropriate Safeguards",
            GdprArticle::Article47 => "Binding Corporate Rules",
            GdprArticle::Article49 => "Derogations for Specific Situations",
        }
    }
}

/// Appropriate safeguards under Article 46
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Safeguard {
    /// Standard Contractual Clauses (Art. 46(2)(c))
    StandardContractualClauses {
        /// SCC reference ID (EU Commission decision identifier)
        decision_reference: String,
        /// Date SCCs were signed
        signed_date: DateTime<Utc>,
        /// Parties to the SCCs
        parties: Vec<Party>,
    },

    /// Binding Corporate Rules (Art. 46(2)(b))
    BindingCorporateRules {
        /// BCR approval reference
        approval_reference: String,
        /// Approving supervisory authority
        approving_authority: String,
        /// Date of approval
        approval_date: DateTime<Utc>,
    },

    /// Approved Code of Conduct (Art. 46(2)(e))
    CodeOfConduct {
        /// Code name/identifier
        code_name: String,
        /// Monitoring body
        monitoring_body: String,
        /// Commitment date
        commitment_date: DateTime<Utc>,
    },

    /// Approved Certification Mechanism (Art. 46(2)(f))
    Certification {
        /// Certification scheme name
        scheme_name: String,
        /// Certification body
        certification_body: String,
        /// Certificate ID
        certificate_id: String,
        /// Valid until
        valid_until: DateTime<Utc>,
    },

    /// Ad-hoc contractual clauses (Art. 46(3)(a))
    AdHocContractualClauses {
        /// Authorizing supervisory authority
        authorizing_authority: String,
        /// Authorization reference
        authorization_reference: String,
    },

    /// Provisions in administrative arrangements (Art. 46(3)(b))
    AdministrativeArrangements {
        /// Public authority/body
        public_body: String,
        /// Arrangement reference
        arrangement_reference: String,
    },
}

impl Safeguard {
    /// Check if the safeguard is currently valid
    pub fn is_valid(&self, now: DateTime<Utc>) -> bool {
        match self {
            Safeguard::Certification { valid_until, .. } => *valid_until > now,
            // Other safeguards don't have expiration dates built in
            _ => true,
        }
    }

    /// Get safeguard type name
    pub fn safeguard_type(&self) -> &'static str {
        match self {
            Safeguard::StandardContractualClauses { .. } => "Standard Contractual Clauses",
            Safeguard::BindingCorporateRules { .. } => "Binding Corporate Rules",
            Safeguard::CodeOfConduct { .. } => "Code of Conduct",
            Safeguard::Certification { .. } => "Certification",
            Safeguard::AdHocContractualClauses { .. } => "Ad-hoc Contractual Clauses",
            Safeguard::AdministrativeArrangements { .. } => "Administrative Arrangements",
        }
    }
}

/// Article 49 derogations for specific situations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Article49Derogation {
    /// Explicit consent (Art. 49(1)(a))
    ExplicitConsent {
        /// Data subject identifier
        data_subject: String,
        /// Consent timestamp
        consent_date: DateTime<Utc>,
        /// Consent is informed about risks
        informed_of_risks: bool,
    },

    /// Necessary for contract performance (Art. 49(1)(b))
    ContractPerformance {
        /// Contract reference
        contract_reference: String,
    },

    /// Necessary for contract in interest of data subject (Art. 49(1)(c))
    ContractInInterest {
        /// Contract reference
        contract_reference: String,
        /// Description of data subject's interest
        interest_description: String,
    },

    /// Important public interest (Art. 49(1)(d))
    PublicInterest {
        /// Legal basis in EU or Member State law
        legal_basis: String,
    },

    /// Legal claims (Art. 49(1)(e))
    LegalClaims {
        /// Nature of legal proceedings
        proceedings_type: String,
    },

    /// Vital interests (Art. 49(1)(f))
    VitalInterests {
        /// Description of vital interest
        interest_description: String,
    },

    /// Public register (Art. 49(1)(g))
    PublicRegister {
        /// Register name
        register_name: String,
    },

    /// Compelling legitimate interests (Art. 49(2))
    CompellingLegitimateInterests {
        /// Description of interests
        interest_description: String,
        /// Assessment that transfer is not repetitive
        not_repetitive: bool,
        /// Limited number of data subjects
        limited_data_subjects: bool,
        /// Appropriate safeguards considered
        safeguards_considered: bool,
    },
}

impl Article49Derogation {
    /// Check if the derogation is valid
    pub fn is_valid(&self) -> bool {
        match self {
            Article49Derogation::ExplicitConsent {
                informed_of_risks, ..
            } => *informed_of_risks,
            Article49Derogation::CompellingLegitimateInterests {
                not_repetitive,
                limited_data_subjects,
                safeguards_considered,
                ..
            } => *not_repetitive && *limited_data_subjects && *safeguards_considered,
            _ => true,
        }
    }

    /// Get derogation type name
    pub fn derogation_type(&self) -> &'static str {
        match self {
            Article49Derogation::ExplicitConsent { .. } => "Explicit Consent",
            Article49Derogation::ContractPerformance { .. } => "Contract Performance",
            Article49Derogation::ContractInInterest { .. } => "Contract in Interest",
            Article49Derogation::PublicInterest { .. } => "Public Interest",
            Article49Derogation::LegalClaims { .. } => "Legal Claims",
            Article49Derogation::VitalInterests { .. } => "Vital Interests",
            Article49Derogation::PublicRegister { .. } => "Public Register",
            Article49Derogation::CompellingLegitimateInterests { .. } => {
                "Compelling Legitimate Interests"
            }
        }
    }
}

/// Transfer compliance result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferComplianceResult {
    /// Is the transfer compliant?
    pub compliant: bool,
    /// GDPR Article providing legal basis
    pub article: Option<GdprArticle>,
    /// Specific safeguard or derogation used
    pub legal_basis_detail: String,
    /// Reasons for non-compliance (if any)
    pub non_compliance_reasons: Vec<String>,
    /// Recommendations for achieving compliance
    pub recommendations: Vec<String>,
}

/// Transfer record for audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferRecord {
    /// Unique record ID
    pub id: String,
    /// Timestamp of transfer
    pub timestamp: DateTime<Utc>,
    /// Source region
    pub from_region: String,
    /// Destination region
    pub to_region: String,
    /// Data categories transferred
    pub data_categories: Vec<String>,
    /// Legal basis for transfer
    pub legal_basis: GdprArticle,
    /// Specific safeguard or derogation
    pub safeguard_detail: Option<String>,
    /// Data controller
    pub controller: Party,
    /// Data processor (if applicable)
    pub processor: Option<Party>,
    /// Purpose of transfer
    pub purpose: String,
    /// Additional notes
    pub notes: Option<String>,
}

impl TransferRecord {
    /// Create a new transfer record
    pub fn new(
        from_region: impl Into<String>,
        to_region: impl Into<String>,
        legal_basis: GdprArticle,
        controller: Party,
        purpose: impl Into<String>,
    ) -> Self {
        Self {
            id: format!("transfer-{}", Utc::now().timestamp_nanos_opt().unwrap_or(0)),
            timestamp: Utc::now(),
            from_region: from_region.into(),
            to_region: to_region.into(),
            data_categories: Vec::new(),
            legal_basis,
            safeguard_detail: None,
            controller,
            processor: None,
            purpose: purpose.into(),
            notes: None,
        }
    }

    /// Add data category
    pub fn with_data_category(mut self, category: impl Into<String>) -> Self {
        self.data_categories.push(category.into());
        self
    }

    /// Add data categories
    pub fn with_data_categories(mut self, categories: Vec<String>) -> Self {
        self.data_categories.extend(categories);
        self
    }

    /// Set safeguard detail
    pub fn with_safeguard_detail(mut self, detail: impl Into<String>) -> Self {
        self.safeguard_detail = Some(detail.into());
        self
    }

    /// Set processor
    pub fn with_processor(mut self, processor: Party) -> Self {
        self.processor = Some(processor);
        self
    }

    /// Set notes
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }
}

/// Type alias for organization pair to safeguards mapping
type SafeguardsMap = HashMap<(String, String), Vec<Safeguard>>;

/// GDPR Compliance Checker
pub struct GdprComplianceChecker {
    /// Registered safeguards per organization pair
    safeguards: Arc<RwLock<SafeguardsMap>>,
    /// Registered BCRs per organization
    bcrs: Arc<RwLock<HashMap<String, Safeguard>>>,
    /// Transfer audit log
    transfer_log: Arc<RwLock<Vec<TransferRecord>>>,
}

impl Default for GdprComplianceChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl GdprComplianceChecker {
    /// Create a new GDPR compliance checker
    pub fn new() -> Self {
        Self {
            safeguards: Arc::new(RwLock::new(HashMap::new())),
            bcrs: Arc::new(RwLock::new(HashMap::new())),
            transfer_log: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a safeguard between two organizations
    pub async fn register_safeguard(
        &self,
        from_org: impl Into<String>,
        to_org: impl Into<String>,
        safeguard: Safeguard,
    ) {
        let mut safeguards = self.safeguards.write().await;
        let key = (from_org.into(), to_org.into());
        safeguards.entry(key).or_default().push(safeguard);
    }

    /// Register Binding Corporate Rules for an organization
    pub async fn register_bcr(&self, organization: impl Into<String>, bcr: Safeguard) {
        let mut bcrs = self.bcrs.write().await;
        bcrs.insert(organization.into(), bcr);
    }

    /// Check if transfer complies with GDPR
    pub async fn check_transfer_compliance(
        &self,
        from: &Region,
        to: &Region,
        from_org: Option<&str>,
        to_org: Option<&str>,
    ) -> IdsResult<TransferComplianceResult> {
        let now = Utc::now();
        let mut result = TransferComplianceResult {
            compliant: false,
            article: None,
            legal_basis_detail: String::new(),
            non_compliance_reasons: Vec::new(),
            recommendations: Vec::new(),
        };

        // Step 1: Check if destination is within EEA (no restriction)
        if self.is_eea_internal_transfer(from, to) {
            result.compliant = true;
            result.legal_basis_detail =
                "Internal EEA transfer - no Chapter V restrictions apply".to_string();
            return Ok(result);
        }

        // Step 2: Article 45 - Adequacy decision
        if to.adequacy == AdequacyStatus::Adequate {
            result.compliant = true;
            result.article = Some(GdprArticle::Article45);
            result.legal_basis_detail = format!("Adequacy decision in effect for {}", to.name);
            return Ok(result);
        }

        // Step 3: Article 46/47 - Appropriate safeguards
        if let Some(safeguard) = self.find_valid_safeguard(from_org, to_org, now).await {
            // Check if it's BCR (Article 47) or other safeguard (Article 46)
            let article = if matches!(safeguard, Safeguard::BindingCorporateRules { .. }) {
                GdprArticle::Article47
            } else {
                GdprArticle::Article46
            };

            result.compliant = true;
            result.article = Some(article);
            result.legal_basis_detail = format!(
                "{} in place: {}",
                safeguard.safeguard_type(),
                self.safeguard_summary(&safeguard)
            );
            return Ok(result);
        }

        // Step 4: No automatic legal basis found
        result
            .non_compliance_reasons
            .push(format!("No adequacy decision for {}", to.name));
        result
            .non_compliance_reasons
            .push("No appropriate safeguards (SCCs, BCRs, etc.) registered".to_string());

        // Add recommendations
        result
            .recommendations
            .push("Consider implementing Standard Contractual Clauses (Art. 46(2)(c))".to_string());
        result.recommendations.push(
            "Verify if Binding Corporate Rules apply within your corporate group".to_string(),
        );
        result
            .recommendations
            .push("Evaluate if Article 49 derogations apply for specific situations".to_string());

        Ok(result)
    }

    /// Check compliance with a specific Article 49 derogation
    pub fn check_derogation(&self, derogation: &Article49Derogation) -> TransferComplianceResult {
        let mut result = TransferComplianceResult {
            compliant: false,
            article: Some(GdprArticle::Article49),
            legal_basis_detail: String::new(),
            non_compliance_reasons: Vec::new(),
            recommendations: Vec::new(),
        };

        if derogation.is_valid() {
            result.compliant = true;
            result.legal_basis_detail =
                format!("Article 49 derogation: {}", derogation.derogation_type());
        } else {
            result.non_compliance_reasons.push(format!(
                "Article 49 derogation '{}' does not meet all requirements",
                derogation.derogation_type()
            ));

            match derogation {
                Article49Derogation::ExplicitConsent {
                    informed_of_risks, ..
                } => {
                    if !informed_of_risks {
                        result.recommendations.push(
                            "Data subject must be informed of risks before giving consent"
                                .to_string(),
                        );
                    }
                }
                Article49Derogation::CompellingLegitimateInterests {
                    not_repetitive,
                    limited_data_subjects,
                    safeguards_considered,
                    ..
                } => {
                    if !not_repetitive {
                        result
                            .recommendations
                            .push("Transfer must not be repetitive".to_string());
                    }
                    if !limited_data_subjects {
                        result.recommendations.push(
                            "Transfer must concern only a limited number of data subjects"
                                .to_string(),
                        );
                    }
                    if !safeguards_considered {
                        result.recommendations.push(
                            "Appropriate safeguards must be assessed and documented".to_string(),
                        );
                    }
                }
                _ => {}
            }
        }

        result
    }

    /// Record a data transfer for audit purposes
    pub async fn record_transfer(&self, record: TransferRecord) -> IdsResult<()> {
        let mut log = self.transfer_log.write().await;
        log.push(record);
        Ok(())
    }

    /// Get transfer records for a time period
    pub async fn get_transfer_records(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Vec<TransferRecord> {
        let log = self.transfer_log.read().await;
        log.iter()
            .filter(|r| r.timestamp >= from && r.timestamp <= to)
            .cloned()
            .collect()
    }

    /// Get transfer records for a specific region
    pub async fn get_transfers_to_region(&self, region_code: &str) -> Vec<TransferRecord> {
        let log = self.transfer_log.read().await;
        log.iter()
            .filter(|r| r.to_region == region_code)
            .cloned()
            .collect()
    }

    /// Check if both regions are within EEA
    fn is_eea_internal_transfer(&self, from: &Region, to: &Region) -> bool {
        let eea_countries = [
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
            "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
            // EEA non-EU members
            "IS", "LI", "NO",
        ];

        let from_eea = eea_countries.contains(&from.jurisdiction.country_code.as_str());
        let to_eea = eea_countries.contains(&to.jurisdiction.country_code.as_str());

        from_eea && to_eea
    }

    /// Find a valid safeguard for the transfer
    async fn find_valid_safeguard(
        &self,
        from_org: Option<&str>,
        to_org: Option<&str>,
        now: DateTime<Utc>,
    ) -> Option<Safeguard> {
        // Check BCRs first
        if let Some(to) = to_org {
            let bcrs = self.bcrs.read().await;
            if let Some(bcr) = bcrs.get(to) {
                if bcr.is_valid(now) {
                    return Some(bcr.clone());
                }
            }
        }

        // Check organization-specific safeguards
        if let (Some(from), Some(to)) = (from_org, to_org) {
            let safeguards = self.safeguards.read().await;
            let key = (from.to_string(), to.to_string());
            if let Some(org_safeguards) = safeguards.get(&key) {
                for safeguard in org_safeguards {
                    if safeguard.is_valid(now) {
                        return Some(safeguard.clone());
                    }
                }
            }
        }

        None
    }

    /// Generate a summary of a safeguard
    fn safeguard_summary(&self, safeguard: &Safeguard) -> String {
        match safeguard {
            Safeguard::StandardContractualClauses {
                decision_reference,
                signed_date,
                ..
            } => {
                format!(
                    "SCCs ref. {} (signed {})",
                    decision_reference,
                    signed_date.format("%Y-%m-%d")
                )
            }
            Safeguard::BindingCorporateRules {
                approval_reference,
                approving_authority,
                ..
            } => {
                format!(
                    "BCR ref. {} (approved by {})",
                    approval_reference, approving_authority
                )
            }
            Safeguard::CodeOfConduct {
                code_name,
                monitoring_body,
                ..
            } => {
                format!("CoC '{}' (monitored by {})", code_name, monitoring_body)
            }
            Safeguard::Certification {
                certificate_id,
                certification_body,
                valid_until,
                ..
            } => {
                format!(
                    "Cert. {} by {} (valid until {})",
                    certificate_id,
                    certification_body,
                    valid_until.format("%Y-%m-%d")
                )
            }
            Safeguard::AdHocContractualClauses {
                authorization_reference,
                ..
            } => {
                format!("Ad-hoc clauses ref. {}", authorization_reference)
            }
            Safeguard::AdministrativeArrangements {
                arrangement_reference,
                ..
            } => {
                format!("Admin. arrangement ref. {}", arrangement_reference)
            }
        }
    }

    /// Legacy method for backward compatibility
    pub fn check_transfer_compliance_simple(from: &Region, to: &Region) -> IdsResult<GdprArticle> {
        // Article 45: Adequacy decision
        if to.adequacy == AdequacyStatus::Adequate {
            return Ok(GdprArticle::Article45);
        }

        // Article 46: Appropriate safeguards (assume available for EU transfers)
        if from.jurisdiction.country_code == "EU"
            || Self::is_eu_member(&from.jurisdiction.country_code)
        {
            return Ok(GdprArticle::Article46);
        }

        Err(IdsError::GdprViolation(format!(
            "No legal basis for transfer from {} to {}",
            from.name, to.name
        )))
    }

    /// Check if country code is an EU member state
    fn is_eu_member(code: &str) -> bool {
        let eu_members = [
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
            "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
        ];
        eu_members.contains(&code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::types::IdsUri;

    fn test_party() -> Party {
        Party {
            id: IdsUri::new("https://example.org/party/1").expect("valid uri"),
            name: "Test Party".to_string(),
            legal_name: None,
            description: None,
            contact: None,
            gaiax_participant_id: None,
        }
    }

    #[test]
    fn test_gdpr_adequacy_transfer() {
        let eu = Region::eu_member("DE", "Germany");
        let jp = Region::japan();

        let result = GdprComplianceChecker::check_transfer_compliance_simple(&eu, &jp);
        assert!(result.is_ok());
        assert_eq!(result.expect("should be ok"), GdprArticle::Article45);
    }

    #[test]
    fn test_gdpr_article_description() {
        assert_eq!(GdprArticle::Article45.description(), "Adequacy Decision");
        assert_eq!(
            GdprArticle::Article46.description(),
            "Appropriate Safeguards"
        );
        assert_eq!(
            GdprArticle::Article47.description(),
            "Binding Corporate Rules"
        );
        assert_eq!(
            GdprArticle::Article49.description(),
            "Derogations for Specific Situations"
        );
    }

    #[tokio::test]
    async fn test_eea_internal_transfer() {
        let checker = GdprComplianceChecker::new();
        let germany = Region::eu_member("DE", "Germany");
        let france = Region::eu_member("FR", "France");

        let result = checker
            .check_transfer_compliance(&germany, &france, None, None)
            .await
            .expect("should succeed");

        assert!(result.compliant);
        assert!(result.legal_basis_detail.contains("Internal EEA"));
    }

    #[tokio::test]
    async fn test_sccs_safeguard() {
        let checker = GdprComplianceChecker::new();

        // Register SCCs between organizations
        let sccs = Safeguard::StandardContractualClauses {
            decision_reference: "2021/914".to_string(),
            signed_date: Utc::now(),
            parties: vec![test_party()],
        };
        checker.register_safeguard("org-eu", "org-us", sccs).await;

        // Check that SCCs are found
        let safeguard = checker
            .find_valid_safeguard(Some("org-eu"), Some("org-us"), Utc::now())
            .await;

        assert!(safeguard.is_some());
        assert_eq!(
            safeguard.expect("should exist").safeguard_type(),
            "Standard Contractual Clauses"
        );
    }

    #[tokio::test]
    async fn test_bcr_registration() {
        let checker = GdprComplianceChecker::new();

        let bcr = Safeguard::BindingCorporateRules {
            approval_reference: "BCR-2023-001".to_string(),
            approving_authority: "German DPA".to_string(),
            approval_date: Utc::now(),
        };
        checker.register_bcr("mega-corp", bcr).await;

        let safeguard = checker
            .find_valid_safeguard(Some("any"), Some("mega-corp"), Utc::now())
            .await;

        assert!(safeguard.is_some());
        assert_eq!(
            safeguard.expect("should exist").safeguard_type(),
            "Binding Corporate Rules"
        );
    }

    #[test]
    fn test_explicit_consent_derogation() {
        let checker = GdprComplianceChecker::new();

        let consent = Article49Derogation::ExplicitConsent {
            data_subject: "user@example.com".to_string(),
            consent_date: Utc::now(),
            informed_of_risks: true,
        };

        let result = checker.check_derogation(&consent);
        assert!(result.compliant);
        assert_eq!(result.article, Some(GdprArticle::Article49));
    }

    #[test]
    fn test_invalid_consent_derogation() {
        let checker = GdprComplianceChecker::new();

        let consent = Article49Derogation::ExplicitConsent {
            data_subject: "user@example.com".to_string(),
            consent_date: Utc::now(),
            informed_of_risks: false, // Not informed
        };

        let result = checker.check_derogation(&consent);
        assert!(!result.compliant);
        assert!(!result.recommendations.is_empty());
    }

    #[test]
    fn test_compelling_legitimate_interests() {
        let checker = GdprComplianceChecker::new();

        let derogation = Article49Derogation::CompellingLegitimateInterests {
            interest_description: "Emergency data recovery".to_string(),
            not_repetitive: true,
            limited_data_subjects: true,
            safeguards_considered: true,
        };

        let result = checker.check_derogation(&derogation);
        assert!(result.compliant);
    }

    #[tokio::test]
    async fn test_transfer_record() {
        let checker = GdprComplianceChecker::new();

        let record = TransferRecord::new(
            "DE",
            "US",
            GdprArticle::Article46,
            test_party(),
            "Data analytics",
        )
        .with_data_category("Personal data")
        .with_safeguard_detail("SCCs 2021/914");

        checker
            .record_transfer(record)
            .await
            .expect("should succeed");

        let records = checker.get_transfers_to_region("US").await;
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].purpose, "Data analytics");
    }

    #[test]
    fn test_certification_validity() {
        let valid_cert = Safeguard::Certification {
            scheme_name: "EU-US DPF".to_string(),
            certification_body: "Cert Body".to_string(),
            certificate_id: "CERT-001".to_string(),
            valid_until: Utc::now() + chrono::Duration::days(365),
        };

        let expired_cert = Safeguard::Certification {
            scheme_name: "EU-US DPF".to_string(),
            certification_body: "Cert Body".to_string(),
            certificate_id: "CERT-002".to_string(),
            valid_until: Utc::now() - chrono::Duration::days(1),
        };

        assert!(valid_cert.is_valid(Utc::now()));
        assert!(!expired_cert.is_valid(Utc::now()));
    }
}
