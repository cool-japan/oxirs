//! W3C Verifiable Credentials Support

use crate::ids::types::IdsUri;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiableCredential {
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    pub id: IdsUri,
    #[serde(rename = "type")]
    pub credential_type: Vec<String>,
    pub issuer: IdsUri,
    pub issuance_date: DateTime<Utc>,
    pub expiration_date: Option<DateTime<Utc>>,
    pub credential_subject: CredentialSubject,
    pub proof: Option<Proof>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialSubject {
    pub id: IdsUri,
    #[serde(flatten)]
    pub claims: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proof {
    #[serde(rename = "type")]
    pub proof_type: String,
    pub created: DateTime<Utc>,
    pub verification_method: String,
    pub proof_purpose: String,
    pub proof_value: String,
}
