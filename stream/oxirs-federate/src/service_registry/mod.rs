//! Auto-generated module structure

pub mod connectionconfig_traits;
pub mod functions;
pub mod graphqlcapabilities_traits;
pub mod performancestats_traits;
pub mod registryconfig_traits;
pub mod serviceregistry_accessors;
pub mod serviceregistry_accessors_1;
pub mod serviceregistry_accessors_2;
pub mod serviceregistry_accessors_3;
pub mod serviceregistry_accessors_4;
pub mod serviceregistry_accessors_5;
pub mod serviceregistry_accessors_6;
pub mod serviceregistry_check_methods;
pub mod serviceregistry_check_methods_1;
pub mod serviceregistry_detect_graphql_version_group;
pub mod serviceregistry_detect_sparql_capabilities_group;
pub mod serviceregistry_enable_extended_metadata_group;
pub mod serviceregistry_fetch_graphql_schema_group;
pub mod serviceregistry_introspect_graphql_service_group;
pub mod serviceregistry_new_group;
pub mod serviceregistry_populate_service_capabilities_group;
pub mod serviceregistry_remove_service_group;
pub mod serviceregistry_start_group;
pub mod serviceregistry_stop_group;
pub mod serviceregistry_test_methods;
pub mod serviceregistry_test_methods_1;
pub mod serviceregistry_traits;
pub mod serviceregistry_type;
pub mod serviceregistry_update_service_capabilities_group;
pub mod serviceregistry_validate_graphql_service_group;
pub mod serviceregistry_validate_sparql_endpoint_group;
pub mod sparqlcapabilities_traits;
pub mod types;

// Re-export only the main types that are used externally
pub use serviceregistry_type::*;
pub use types::*;
