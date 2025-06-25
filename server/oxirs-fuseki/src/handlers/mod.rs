//! HTTP request handlers for SPARQL protocol and server management

pub mod admin;
pub mod auth;
pub mod graph;
pub mod mfa;
pub mod oauth2;
pub mod saml;
pub mod sparql;
pub mod websocket;

// Re-export commonly used handlers
pub use admin::ui_handler;
pub use mfa::{
    create_mfa_challenge, disable_mfa, enroll_mfa, get_mfa_status, regenerate_backup_codes,
    verify_mfa,
};
pub use oauth2::{
    get_oauth2_config, get_oauth2_user_info, handle_oauth2_callback, initiate_oauth2_flow,
    oauth2_discovery, refresh_oauth2_token, validate_oauth2_token,
};
pub use saml::{
    get_saml_metadata, handle_saml_acs, handle_saml_slo, initiate_saml_logout, initiate_saml_sso,
};
pub use sparql::{query_handler, update_handler};
pub use websocket::{websocket_handler, SubscriptionManager};
