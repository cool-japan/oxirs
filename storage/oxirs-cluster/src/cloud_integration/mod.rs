//! Auto-generated module structure

pub mod azureblobbackend_traits;
pub mod cloudprovider_traits;
pub mod disasterrecoveryconfig_traits;
pub mod elasticscalingconfig_traits;
pub mod elasticscalingmanager_accessors;
pub mod elasticscalingmanager_evaluate_scaling_group;
pub mod elasticscalingmanager_execute_scaling_group;
pub mod elasticscalingmanager_new_group;
pub mod elasticscalingmanager_predict_scaling_needs_group;
pub mod elasticscalingmanager_type;
pub mod elasticscalingmanager_update_metrics_group;
pub mod functions;
pub mod gcsbackend_traits;
pub mod s3backend_traits;
pub mod types;

// Re-export all types
pub use elasticscalingmanager_type::*;
pub use functions::*;
pub use types::*;
