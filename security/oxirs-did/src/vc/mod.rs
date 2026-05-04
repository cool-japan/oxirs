//! Verifiable Credentials module

pub mod credential;
pub mod issuer;
pub mod jwt_vc;
pub mod presentation;
pub mod sd_jwt;
pub mod verifier;

pub use credential::{CredentialSubject, VerifiableCredential};
pub use issuer::CredentialIssuer;
pub use jwt_vc::{decode_jwt_vc, encode_vc_as_jwt, JwtVc, JwtVcHeader, JwtVcPayload};
pub use presentation::VerifiablePresentation;
pub use verifier::CredentialVerifier;
