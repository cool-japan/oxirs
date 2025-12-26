//! Verifiable Credentials module

pub mod credential;
pub mod issuer;
pub mod presentation;
pub mod verifier;

pub use credential::{CredentialSubject, VerifiableCredential};
pub use issuer::CredentialIssuer;
pub use presentation::VerifiablePresentation;
pub use verifier::CredentialVerifier;
