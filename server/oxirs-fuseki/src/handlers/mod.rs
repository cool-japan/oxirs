//! HTTP request handlers for SPARQL protocol and server management

pub mod admin;
pub mod api_keys;
pub mod auth;
pub mod graph;
pub mod ldap;
pub mod mfa;
pub mod oauth2;
#[cfg(feature = "saml")]
pub mod saml;
pub mod sparql;
pub mod sparql_refactored;
pub mod websocket;

// Re-export commonly used handlers
pub use admin::ui_handler;
pub use api_keys::{
    create_api_key, get_api_key, get_api_key_usage, list_api_keys, revoke_api_key, update_api_key,
    validate_api_key_auth,
};
pub use ldap::{get_ldap_config, get_ldap_groups, ldap_login, test_ldap_connection};
pub use mfa::{
    create_mfa_challenge, disable_mfa, enroll_mfa, get_mfa_status, regenerate_backup_codes,
    verify_mfa,
};
pub use oauth2::{
    get_oauth2_config, get_oauth2_user_info, handle_oauth2_callback, initiate_oauth2_flow,
    oauth2_discovery, refresh_oauth2_token, validate_oauth2_token,
};
#[cfg(feature = "saml")]
pub use saml::{
    get_saml_metadata, handle_saml_acs, handle_saml_slo, initiate_saml_logout, initiate_saml_sso,
};
pub use sparql_refactored::{query_handler, query_handler_get, query_handler_post, update_handler};
pub use websocket::{websocket_handler, SubscriptionManager};
