//! HTTP request handlers for SPARQL protocol and server management

pub mod sparql;
pub mod graph;
pub mod admin;
pub mod auth;
pub mod websocket;
pub mod oauth2;

// Re-export commonly used handlers
pub use sparql::{query_handler, update_handler};
pub use admin::{ui_handler};
pub use websocket::{websocket_handler, SubscriptionManager};
pub use oauth2::{
    initiate_oauth2_flow, handle_oauth2_callback, refresh_oauth2_token,
    get_oauth2_user_info, validate_oauth2_token, get_oauth2_config, oauth2_discovery
};