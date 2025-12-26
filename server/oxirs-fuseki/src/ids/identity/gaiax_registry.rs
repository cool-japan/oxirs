//! Gaia-X Trust Registry Integration

use crate::ids::types::{IdsUri, Party};

/// Gaia-X Registry Client
pub struct GaiaxRegistry {
    registry_url: String,
}

impl GaiaxRegistry {
    pub fn new(registry_url: impl Into<String>) -> Self {
        Self {
            registry_url: registry_url.into(),
        }
    }

    pub async fn verify_participant(
        &self,
        participant_id: &str,
    ) -> crate::ids::types::IdsResult<bool> {
        // TODO: Query Gaia-X registry
        Ok(true)
    }
}
