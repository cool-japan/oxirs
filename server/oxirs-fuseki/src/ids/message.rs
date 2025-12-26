//! IDS Message Protocol
//!
//! IDS Multipart messages for connector communication

use super::types::{IdsUri, Party};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// IDS Message Header
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IdsMessageHeader {
    #[serde(rename = "@id")]
    pub id: IdsUri,

    #[serde(rename = "@type")]
    pub message_type: String,

    pub issued: DateTime<Utc>,

    pub issuer_connector: IdsUri,

    pub sender_agent: IdsUri,

    pub recipient_connector: Option<Vec<IdsUri>>,

    pub security_token: Option<String>,

    pub correlation_message: Option<IdsUri>,
}

/// IDS Request Message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMessage {
    pub header: IdsMessageHeader,
    pub payload: Option<serde_json::Value>,
}

/// IDS Response Message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMessage {
    pub header: IdsMessageHeader,
    pub payload: Option<serde_json::Value>,
}
