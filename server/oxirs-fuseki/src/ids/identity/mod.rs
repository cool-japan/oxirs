//! IDS Identity and Trust Management

pub mod daps;
pub mod gaiax_registry;
pub mod verifiable_credentials;

pub use daps::{DapsClient, DapsToken};
pub use gaiax_registry::GaiaxRegistry;
pub use verifiable_credentials::{CredentialSubject, VerifiableCredential};

use crate::ids::types::{IdsUri, Party};

/// Identity Provider for IDS
pub struct IdentityProvider {
    daps_client: DapsClient,
}

impl IdentityProvider {
    pub fn new(daps_url: impl Into<String>) -> Self {
        Self {
            daps_client: DapsClient::new(daps_url),
        }
    }

    pub async fn authenticate(&self, party: &Party) -> crate::ids::types::IdsResult<DapsToken> {
        self.daps_client.get_token(&party.id).await
    }
}
