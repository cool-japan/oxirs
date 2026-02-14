//! Auto-generated module structure

pub mod errorratestats_traits;
pub mod functions;
pub mod geocoordinates_traits;
pub mod regionconfig_traits;
pub mod regionmanager_accessors;
pub mod regionmanager_accessors_1;
pub mod regionmanager_accessors_2;
pub mod regionmanager_accessors_3;
pub mod regionmanager_accessors_4;
pub mod regionmanager_accessors_5;
pub mod regionmanager_accessors_6;
pub mod regionmanager_accessors_7;
pub mod regionmanager_calculate_distance_group;
pub mod regionmanager_compress_for_transfer_group;
pub mod regionmanager_decompress_from_transfer_group;
pub mod regionmanager_dijkstra_group;
pub mod regionmanager_disable_monitoring_group;
pub mod regionmanager_enable_monitoring_group;
pub mod regionmanager_initialize_group;
pub mod regionmanager_measure_latency_group;
pub mod regionmanager_monitor_latencies_group;
pub mod regionmanager_new_group;
pub mod regionmanager_perform_region_failover_group;
pub mod regionmanager_ping_node_group;
pub mod regionmanager_predicates;
pub mod regionmanager_queries;
pub mod regionmanager_register_node_group;
pub mod regionmanager_replicate_cross_region_group;
pub mod regionmanager_route_read_group;
pub mod regionmanager_schedule_reconciliation_group;
pub mod regionmanager_type;
pub mod regionperformancemetrics_traits;
pub mod routingstrategy_traits;
pub mod throughputstats_traits;
pub mod types;
pub mod vectorclock_traits;

// Re-export all types
pub use regionmanager_type::*;
pub use types::*;
