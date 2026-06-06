//! SAML 2.0 Service Provider authentication support
//!
//! Implements the SAML 2.0 Web Browser SSO Profile for service provider (SP) initiated
//! and IdP-initiated authentication flows. XML parsing is performed with `quick-xml`
//! for pure-Rust, zero-C-dependency operation.
//!
//! ## Signature verification
//!
//! XML signature verification uses a simplified approach: the raw `ds:SignatureValue`
//! bytes are verified against the canonicalized `ds:SignedInfo` element using RSA-SHA256
//! via the `rsa` crate. This covers the most common real-world case (enveloped RSA-SHA256
//! signatures) but does **not** implement full W3C XMLDSig (no Exclusive C14N, no
//! Transform resolution, no reference URI dereferencing). A more complete implementation
//! would require an external C14N library. The limitation is logged at `warn!` level when
//! operating in a production context.
//!
//! ## Module layout
//!
//! This module is a thin facade (Round 32 refactor). The implementation lives in
//! sibling modules:
//! - [`saml_types`](super::saml_types): configuration, provider/session, and protocol structs
//! - [`saml_parser`](super::saml_parser): the [`SamlResponseParser`] XML parser
//! - [`saml_provider`](super::saml_provider): the [`SamlProvider`] flow implementation
//! - [`saml_helpers`](super::saml_helpers): XML escaping / DER decoding helpers

pub use super::saml_parser::*;
pub use super::saml_types::*;
