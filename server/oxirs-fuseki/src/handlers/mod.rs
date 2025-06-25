//! HTTP request handlers for SPARQL protocol and server management

pub mod sparql;
pub mod graph;
pub mod admin;
pub mod auth;
pub mod websocket;
pub mod oauth2;
pub mod saml;
pub mod mfa;

// Re-export commonly used handlers
pub use sparql::{query_handler, update_handler};
pub use admin::{ui_handler};
pub use websocket::{websocket_handler, SubscriptionManager};
pub use oauth2::{
    initiate_oauth2_flow, handle_oauth2_callback, refresh_oauth2_token,
    get_oauth2_user_info, validate_oauth2_token, get_oauth2_config, oauth2_discovery
};
pub use saml::{
    initiate_saml_sso, handle_saml_acs, handle_saml_slo, get_saml_metadata, initiate_saml_logout
};
pub use mfa::{
    enroll_mfa, create_mfa_challenge, verify_mfa, get_mfa_status, disable_mfa, regenerate_backup_codes
};