//! DID Method implementations

pub mod did_key;
#[cfg(feature = "did-web")]
pub mod did_web;

pub use did_key::DidKeyMethod;
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
