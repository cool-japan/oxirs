//! DID Method implementations

#[cfg(feature = "did-ethr")]
pub mod did_ethr;
#[cfg(feature = "did-ion")]
pub mod did_ion;
pub mod did_key;
pub mod did_pkh;
#[cfg(feature = "did-web")]
pub mod did_web;

#[cfg(feature = "did-ethr")]
pub use did_ethr::{DidEthr, DidEthrMethod, EthNetwork};
#[cfg(feature = "did-ion")]
pub use did_ion::{
    DidIon, DidIonMethod, IonCreateOperation, IonDocument, IonKeyDescriptor, IonKeyPurpose,
    IonOperationType, IonService,
};
pub use did_key::DidKeyMethod;
pub use did_pkh::{ChainNamespace, DidPkh, DidPkhMethod};
#[cfg(feature = "did-web")]
pub use did_web::DidWebMethod;

use super::{Did, DidDocument};
use crate::DidResult;
use async_trait::async_trait;

/// Trait for DID method resolvers
#[async_trait]
pub trait DidMethod: Send + Sync {
    /// Get the method name (e.g., "key", "web")
    fn method_name(&self) -> &str;

    /// Resolve a DID to its DID Document
    async fn resolve(&self, did: &Did) -> DidResult<DidDocument>;

    /// Check if this method supports the given DID
    fn supports(&self, did: &Did) -> bool {
        did.method() == self.method_name()
    }
}
