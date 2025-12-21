//! HTTP request handlers for SPARQL protocol and server management

pub mod adaptive_execution_admin; // Adaptive execution monitoring endpoints (v0.1.0 Final)
pub mod admin;
pub mod api_keys;
pub mod auth;
pub mod dataset_stats; // Dataset Statistics
pub mod gpu_embeddings_admin; // GPU knowledge graph embeddings management (v0.1.0 Final - Session 20)
pub mod graph;
pub mod gsp; // Graph Store Protocol (W3C SPARQL 1.1)
pub mod ldap;
pub mod mfa;
pub mod oauth2;
pub mod patch; // RDF Patch
pub mod performance; // Beta.2 Performance Monitoring
pub mod prefixes; // Prefix Management
pub mod production; // RC.1 Production Management (load balancing, edge caching, CDN)
pub mod rebac; // ReBAC (Relationship-Based Access Control) API
pub mod request_log; // Request Logging
#[cfg(feature = "saml")]
pub mod saml;
pub mod shacl; // SHACL Validation
pub mod simd_admin; // SIMD triple matcher management (v0.1.0 Final - Session 20)
pub mod sparql;
pub mod sparql_refactored;
pub mod tasks; // Async Task Management
pub mod upload; // RDF Bulk Upload
pub mod validation; // Validation Services (Query, Update, IRI, Data, Language Tag)
pub mod websocket;

// Re-export commonly used handlers
pub use admin::{list_backups, reload_config, ui_handler};
pub use api_keys::{
    create_api_key, get_api_key, get_api_key_usage, list_api_keys, revoke_api_key, update_api_key,
    validate_api_key_auth,
};
pub use dataset_stats::{get_dataset_stats, get_server_stats};
pub use gsp::{
    handle_gsp_delete_server, handle_gsp_get_server, handle_gsp_head_server,
    handle_gsp_options_server, handle_gsp_post_server, handle_gsp_put_server,
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
pub use patch::handle_patch_server;
pub use prefixes::{
    add_prefix, delete_prefix, expand_prefix, get_prefix, list_prefixes, update_prefix, PrefixStore,
};
pub use rebac::{add_tuple, batch_check_permissions, check_permission, list_tuples, remove_tuple};
pub use request_log::{
    clear_logs, get_log_config, get_log_statistics, get_logs, update_log_config, RequestLogger,
};
#[cfg(feature = "saml")]
pub use saml::{
    get_saml_metadata, handle_saml_acs, handle_saml_slo, initiate_saml_logout, initiate_saml_sso,
};
pub use shacl::handle_shacl_validation_server;
pub use sparql_refactored::{query_handler, query_handler_get, query_handler_post, update_handler};
pub use tasks::{
    cancel_task, create_task, delete_task, get_task, get_task_statistics, list_tasks, TaskManager,
};
pub use upload::{handle_multipart_upload_server, handle_upload_server};
pub use validation::{
    validate_data, validate_iri, validate_iri_get, validate_langtag, validate_langtag_get,
    validate_query, validate_query_get, validate_update, validate_update_get,
};
pub use websocket::{websocket_handler, SubscriptionManager};
