//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
use crate::backup::{BackupConfig, BackupManager};
#[cfg(feature = "hot-reload")]
use crate::config_reload::ConfigReloadManager;
use crate::ddos_protection::{DDoSProtectionConfig, DDoSProtectionManager};
use crate::disaster_recovery::{DisasterRecoveryConfig, DisasterRecoveryManager};
use crate::edge_caching::{EdgeCacheConfig, EdgeCacheManager};
use crate::http_protocol::{Http2Manager, Http3Manager, HttpProtocolConfig};
use crate::load_balancing::{LoadBalancer, LoadBalancerConfig};
use crate::performance_profiler::{PerformanceProfiler, ProfilerConfig};
use crate::realtime_notifications::NotificationManager;
use crate::recovery::{RecoveryConfig, RecoveryManager};
use crate::security_audit::{SecurityAuditConfig, SecurityAuditManager};
use crate::tls_rotation::CertificateRotation;
use crate::{
    adaptive_execution::{AdaptiveExecutionConfig, AdaptiveExecutionEngine},
    auth::AuthService,
    batch_execution::{BatchConfig, BatchExecutor},
    concurrent::{ConcurrencyConfig, ConcurrencyManager},
    config::{ServerConfig, TlsConfig},
    dataset_management::{DatasetConfig, DatasetManager},
    error::{FusekiError, FusekiResult},
    federation::{FederationConfig, FederationManager},
    handlers,
    memory_pool::{MemoryManager, MemoryPoolConfig},
    metrics::{MetricsService, RequestMetrics},
    optimization::QueryOptimizer,
    performance::PerformanceService,
    store::Store,
    streaming::{StreamingConfig, StreamingManager},
    streaming_results::{StreamConfig, StreamManager},
    tls::TlsManager,
    websocket::{SubscriptionManager, WebSocketConfig},
};
use axum::{
    extract::{Path, Query, Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Json, Response},
    routing::{delete, get, post},
    Router,
};
#[cfg(feature = "rate-limit")]
use governor::{Quota, RateLimiter};
use std::net::SocketAddr;
#[cfg(feature = "rate-limit")]
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::signal;
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

/// HTTP server runtime with comprehensive middleware and services
pub struct Runtime {
    addr: SocketAddr,
    store: Store,
    config: ServerConfig,
    auth_service: Option<AuthService>,
    metrics_service: Option<Arc<MetricsService>>,
    performance_service: Option<Arc<PerformanceService>>,
    query_optimizer: Option<Arc<QueryOptimizer>>,
    subscription_manager: Option<Arc<SubscriptionManager>>,
    federation_manager: Option<Arc<FederationManager>>,
    streaming_manager: Option<Arc<StreamingManager>>,
    concurrency_manager: Option<Arc<ConcurrencyManager>>,
    memory_manager: Option<Arc<MemoryManager>>,
    batch_executor: Option<Arc<BatchExecutor>>,
    stream_manager: Option<Arc<StreamManager>>,
    dataset_manager: Option<Arc<DatasetManager>>,
    security_auditor: Option<Arc<SecurityAuditManager>>,
    ddos_protector: Option<Arc<DDoSProtectionManager>>,
    load_balancer: Option<Arc<LoadBalancer>>,
    edge_cache_manager: Option<Arc<EdgeCacheManager>>,
    performance_profiler: Option<Arc<PerformanceProfiler>>,
    notification_manager: Option<Arc<NotificationManager>>,
    backup_manager: Option<Arc<BackupManager>>,
    recovery_manager: Option<Arc<RecoveryManager>>,
    disaster_recovery: Option<Arc<DisasterRecoveryManager>>,
    certificate_rotation: Option<Arc<CertificateRotation>>,
    http2_manager: Option<Arc<Http2Manager>>,
    http3_manager: Option<Arc<Http3Manager>>,
    adaptive_execution_engine: Option<Arc<AdaptiveExecutionEngine>>,
    rebac_manager: Option<Arc<dyn crate::auth::rebac::RebacEvaluator>>,
    ids_api_state: Option<Arc<crate::ids::IdsApiState>>,
    #[cfg(feature = "rate-limit")]
    rate_limiter: Option<Arc<governor::DefaultKeyedRateLimiter<String>>>,
    #[cfg(feature = "hot-reload")]
    config_watcher: Option<watch::Receiver<ServerConfig>>,
    #[cfg(feature = "hot-reload")]
    config_reload_manager: Option<Arc<parking_lot::Mutex<ConfigReloadManager>>>,
}
impl Runtime {
    /// Create a new runtime instance
    pub fn new(addr: SocketAddr, store: Store, config: ServerConfig) -> Self {
        Runtime {
            addr,
            store,
            config,
            auth_service: None,
            metrics_service: None,
            performance_service: None,
            query_optimizer: None,
            subscription_manager: None,
            federation_manager: None,
            streaming_manager: None,
            concurrency_manager: None,
            memory_manager: None,
            batch_executor: None,
            stream_manager: None,
            dataset_manager: None,
            security_auditor: None,
            ddos_protector: None,
            load_balancer: None,
            edge_cache_manager: None,
            performance_profiler: None,
            notification_manager: None,
            backup_manager: None,
            recovery_manager: None,
            disaster_recovery: None,
            certificate_rotation: None,
            http2_manager: None,
            http3_manager: None,
            adaptive_execution_engine: None,
            rebac_manager: None,
            ids_api_state: None,
            #[cfg(feature = "rate-limit")]
            rate_limiter: None,
            #[cfg(feature = "hot-reload")]
            config_watcher: None,
            #[cfg(feature = "hot-reload")]
            config_reload_manager: None,
        }
    }
    /// Initialize services based on configuration
    pub async fn initialize_services(&mut self) -> FusekiResult<()> {
        info!("Initializing server services...");
        if self.config.security.auth_required {
            info!("Initializing authentication service");
            let auth_service = AuthService::new(self.config.security.clone()).await?;
            self.auth_service = Some(auth_service);
        }
        info!("Initializing ReBAC manager");
        let rebac_manager = Arc::new(crate::auth::rebac::InMemoryRebacManager::new());
        self.rebac_manager = Some(rebac_manager as Arc<dyn crate::auth::rebac::RebacEvaluator>);
        if self.config.monitoring.metrics.enabled {
            info!("Initializing metrics service");
            let metrics_service = MetricsService::new(self.config.monitoring.clone())?;
            self.metrics_service = Some(Arc::new(metrics_service));
        }
        info!("Initializing performance optimization service");
        let performance_service = PerformanceService::new(self.config.performance.clone())?;
        self.performance_service = Some(Arc::new(performance_service));
        if self.config.performance.query_optimization.enabled {
            info!("Initializing advanced query optimizer");
            let query_optimizer = QueryOptimizer::new(self.config.performance.clone())?;
            self.query_optimizer = Some(Arc::new(query_optimizer));
        }
        info!("Initializing adaptive execution engine with SciRS2 integration");
        let adaptive_config = AdaptiveExecutionConfig {
            enable_adaptive_learning: true,
            min_sample_size: 10,
            confidence_level: 0.95,
            enable_cost_model_tuning: true,
            enable_ml_prediction: true,
            ga_population_size: 50,
            ga_max_generations: 100,
            enable_parallel_evaluation: true,
            parallel_workers: num_cpus::get(),
        };
        let adaptive_engine = AdaptiveExecutionEngine::new(adaptive_config)?;
        self.adaptive_execution_engine = Some(Arc::new(adaptive_engine));
        info!("Initializing WebSocket subscription manager");
        let ws_config = WebSocketConfig::default();
        let store = Arc::new(self.store.clone());
        let metrics = match self.metrics_service.clone() {
            Some(service) => service,
            None => {
                let metrics_service = MetricsService::new(self.config.monitoring.clone())?;
                Arc::new(metrics_service)
            }
        };
        let subscription_manager = SubscriptionManager::new(store, metrics, ws_config);
        let manager_clone = subscription_manager.clone();
        tokio::spawn(async move {
            manager_clone.start().await;
        });
        self.subscription_manager = Some(Arc::new(subscription_manager));
        info!("Initializing federation manager");
        let federation_config = self
            .config
            .federation
            .clone()
            .unwrap_or_else(FederationConfig::default);
        let federation_manager = FederationManager::new(federation_config);
        federation_manager.start().await?;
        self.federation_manager = Some(Arc::new(federation_manager));
        info!("Initializing streaming manager");
        let streaming_config = self
            .config
            .streaming
            .clone()
            .unwrap_or_else(StreamingConfig::default);
        let streaming_manager = StreamingManager::new(streaming_config);
        streaming_manager.initialize().await?;
        self.streaming_manager = Some(Arc::new(streaming_manager));
        info!("Initializing Beta.2 Memory Manager");
        let memory_config = MemoryPoolConfig {
            enabled: true,
            max_memory_bytes: 4_294_967_296,
            pressure_threshold: 0.85,
            query_context_pool_size: 500,
            result_buffer_pool_size: 200,
            small_buffer_size: 4 * 1024,
            medium_buffer_size: 64 * 1024,
            large_buffer_size: 1024 * 1024,
            chunk_size_bytes: 512 * 1024,
            enable_profiling: true,
            gc_interval_secs: 60,
        };
        let memory_manager = MemoryManager::new(memory_config)?;
        self.memory_manager = Some(memory_manager.clone());
        info!("Initializing Beta.2 Concurrency Manager");
        let concurrency_config = ConcurrencyConfig {
            max_global_concurrent: 200,
            max_per_dataset_concurrent: 50,
            max_per_user_concurrent: 10,
            enable_work_stealing: true,
            max_queue_size: 10_000,
            queue_timeout_secs: 300,
            enable_load_shedding: true,
            load_shedding_threshold: 0.9,
            worker_threads: num_cpus::get(),
            enable_fair_scheduling: true,
        };
        let concurrency_manager = ConcurrencyManager::new(concurrency_config);
        self.concurrency_manager = Some(concurrency_manager.clone());
        info!("Initializing Beta.2 Batch Executor");
        let batch_config = BatchConfig {
            enabled: true,
            max_batch_size: 100,
            min_batch_size: 10,
            max_wait_time_ms: 100,
            adaptive_sizing: true,
            max_parallel_batches: 4,
            analyze_dependencies: true,
            max_parallel_queries: 20,
        };
        let batch_executor = BatchExecutor::new(batch_config, Arc::new(self.store.clone()));
        self.batch_executor = Some(batch_executor.clone());
        info!("Initializing Beta.2 Stream Manager");
        let stream_config = StreamConfig {
            chunk_size: 64 * 1024,
            buffer_size: 16,
            adaptive_chunking: true,
            max_memory_per_stream: 16 * 1024 * 1024,
            compression: crate::streaming_results::Compression::None,
            compression_level: 6,
            backpressure_threshold: 0.8,
        };
        let stream_manager = StreamManager::new(stream_config, Some(memory_manager.clone()));
        self.stream_manager = Some(stream_manager);
        info!("Initializing Beta.2 Dataset Manager");
        let dataset_config = DatasetConfig {
            base_path: std::path::PathBuf::from("./data/datasets"),
            enable_versioning: true,
            max_snapshots: 10,
            auto_backup: false,
            backup_interval_secs: 3600,
            max_concurrent_ops: 5,
        };
        let dataset_manager = DatasetManager::new(dataset_config).await?;
        self.dataset_manager = Some(dataset_manager);
        info!("Beta.2 Performance & Scalability modules initialized successfully");
        info!("Initializing RC.1 Security Auditor");
        let audit_config = SecurityAuditConfig {
            enabled: true,
            vulnerability_scanning: true,
            scan_interval_hours: 24,
            owasp_checks: true,
            compliance_checks: true,
            max_log_entries: 10_000,
        };
        let security_auditor = SecurityAuditManager::new(audit_config);
        self.security_auditor = Some(Arc::new(security_auditor));
        info!("Initializing RC.1 DDoS Protector");
        let ddos_config = DDoSProtectionConfig {
            enabled: true,
            requests_per_second: 100,
            burst_size: 50,
            block_duration_secs: 600,
            auto_block: true,
            enable_challenge: false,
            max_connections_per_ip: 20,
            enable_traffic_analysis: true,
        };
        let ddos_protector = DDoSProtectionManager::new(ddos_config);
        self.ddos_protector = Some(Arc::new(ddos_protector));
        info!("Initializing RC.1 Load Balancer");
        let load_balancer_config = LoadBalancerConfig::default();
        let load_balancer = LoadBalancer::new(load_balancer_config);
        self.load_balancer = Some(Arc::new(load_balancer));
        info!("Initializing RC.1 Edge Cache Manager");
        let edge_cache_config = EdgeCacheConfig::default();
        let edge_cache_manager = EdgeCacheManager::new(edge_cache_config);
        self.edge_cache_manager = Some(Arc::new(edge_cache_manager));
        info!("Initializing RC.1 Performance Profiler");
        let profiler_config = ProfilerConfig {
            enabled: true,
            sampling_rate: 0.1,
            max_profiles: 10_000,
            detailed_tracing: true,
            metrics_retention_duration: Duration::from_secs(24 * 3600),
        };
        let performance_profiler = PerformanceProfiler::new(profiler_config);
        self.performance_profiler = Some(Arc::new(performance_profiler));
        info!("Initializing RC.1 Notification Manager");
        let notification_manager = NotificationManager::new();
        self.notification_manager = Some(Arc::new(notification_manager));
        info!("Initializing RC.1 Backup Manager");
        let backup_config = BackupConfig {
            enabled: true,
            interval_hours: 1,
            backup_dir: std::path::PathBuf::from("./data/backups"),
            max_backups: 30,
            compression: true,
            include_indexes: true,
            strategy: crate::backup::BackupStrategy::Full,
        };
        let store_arc = Arc::new(self.store.clone());
        let backup_manager = Arc::new(BackupManager::new(store_arc.clone(), backup_config));
        self.backup_manager = Some(backup_manager.clone());
        info!("Initializing RC.1 Recovery Manager");
        let recovery_config = RecoveryConfig {
            enabled: true,
            health_check_interval: Duration::from_secs(30),
            max_restart_attempts: 3,
            restart_backoff_multiplier: 2.0,
            memory_threshold_mb: 1024,
            connection_pool_recovery: true,
        };
        let recovery_manager = RecoveryManager::new(store_arc.clone(), recovery_config);
        self.recovery_manager = Some(Arc::new(recovery_manager));
        info!("Initializing RC.1 Disaster Recovery Manager with Health Monitoring");
        let disaster_recovery_config = DisasterRecoveryConfig {
            enabled: true,
            rpo_minutes: 60,
            rto_minutes: 15,
            auto_failover: true,
            replication_targets: vec![],
            health_check_interval_secs: 60,
            enable_recovery_testing: false,
            recovery_test_interval_days: 30,
        };
        let disaster_recovery = DisasterRecoveryManager::with_health_monitoring(
            store_arc.clone(),
            backup_manager.clone(),
            disaster_recovery_config,
        );
        self.disaster_recovery = Some(Arc::new(disaster_recovery));
        info!("Disaster Recovery Manager initialized with comprehensive health monitoring");
        if self.config.server.tls.is_some() {
            info!("TLS certificate rotation available (manual configuration required)");
            self.certificate_rotation = None;
        }
        info!("Initializing RC.1 HTTP/2 Manager");
        let http_protocol_config = HttpProtocolConfig {
            http2_enabled: self.config.http_protocol.http2_enabled,
            http3_enabled: self.config.http_protocol.http3_enabled,
            http2_initial_connection_window_size: self
                .config
                .http_protocol
                .http2_initial_connection_window_size,
            http2_initial_stream_window_size: self
                .config
                .http_protocol
                .http2_initial_stream_window_size,
            http2_max_concurrent_streams: self.config.http_protocol.http2_max_concurrent_streams,
            http2_max_frame_size: self.config.http_protocol.http2_max_frame_size,
            http2_keep_alive_interval: Duration::from_secs(
                self.config.http_protocol.http2_keep_alive_interval_secs,
            ),
            http2_keep_alive_timeout: Duration::from_secs(
                self.config.http_protocol.http2_keep_alive_timeout_secs,
            ),
            enable_server_push: self.config.http_protocol.enable_server_push,
            enable_header_compression: self.config.http_protocol.enable_header_compression,
        };
        let mut http2_manager = Http2Manager::new(http_protocol_config.clone());
        if self.config.http_protocol.sparql_optimized {
            http2_manager.optimize_for_sparql();
        }
        self.http2_manager = Some(Arc::new(http2_manager));
        info!(
            "HTTP/2 enabled: {}, SPARQL optimized: {}",
            http_protocol_config.http2_enabled, self.config.http_protocol.sparql_optimized
        );
        if http_protocol_config.http3_enabled {
            info!("Initializing RC.1 HTTP/3 Manager (experimental)");
            let http3_manager = Http3Manager::new(http_protocol_config.clone());
            self.http3_manager = Some(Arc::new(http3_manager));
        }
        info!("All RC.1 Production & Advanced modules initialized successfully");
        #[cfg(feature = "rate-limit")]
        {
            if let Some(rate_limit_config) = &self.config.performance.rate_limiting {
                info!(
                    "Initializing rate limiter: {} requests per minute",
                    rate_limit_config.requests_per_minute
                );
                let quota = Quota::per_minute(
                    NonZeroU32::new(rate_limit_config.requests_per_minute)
                        .expect("requests_per_minute should be non-zero"),
                );
                let limiter = RateLimiter::dashmap(quota);
                self.rate_limiter = Some(Arc::new(limiter));
            }
        }
        #[cfg(feature = "hot-reload")]
        {
            if let Some(config_file) = &self.config.server.config_file {
                info!(
                    "Initializing configuration hot-reload manager for {:?}",
                    config_file
                );
                let shared_config = Arc::new(tokio::sync::RwLock::new(self.config.clone()));
                match ConfigReloadManager::new(config_file.clone(), shared_config.clone()) {
                    Ok(mut manager) => {
                        if let Err(e) = manager.start_watching() {
                            warn!("Failed to start config file watching: {}", e);
                        } else {
                            info!("Configuration hot-reload is now active");
                            self.config_reload_manager =
                                Some(Arc::new(parking_lot::Mutex::new(manager)));
                        }
                    }
                    Err(e) => {
                        warn!("Failed to initialize config reload manager: {}", e);
                    }
                }
            } else {
                info!("Hot-reload feature is available but no config file path specified");
            }
        }
        info!("Initializing IDS Connector");
        let ids_config = crate::ids::IdsConnectorConfig::default();
        let ids_connector = Arc::new(crate::ids::IdsConnector::new(ids_config));
        let data_plane = Arc::new(crate::ids::DataPlaneManager::new(
            ids_connector.connector_id().clone(),
            ids_connector.policy_engine(),
            ids_connector.lineage_tracker(),
        ));
        let ids_api_state = Arc::new(crate::ids::IdsApiState::new(ids_connector, data_plane));
        self.ids_api_state = Some(ids_api_state);
        info!("IDS Connector initialized (IDSA Reference Architecture 4.x)");
        info!("Server services initialized successfully");
        Ok(())
    }
    /// Start the HTTP server with full middleware stack
    pub async fn run(mut self) -> FusekiResult<()> {
        self.initialize_services().await?;
        let addr = self.addr;
        let config = self.config.clone();
        let app_state = AppState {
            store: self.store.clone(),
            config: config.clone(),
            auth_service: self.auth_service.clone(),
            metrics_service: self.metrics_service.clone(),
            performance_service: self.performance_service.clone(),
            query_optimizer: self.query_optimizer.clone(),
            subscription_manager: self.subscription_manager.clone(),
            federation_manager: self.federation_manager.clone(),
            streaming_manager: self.streaming_manager.clone(),
            concurrency_manager: self.concurrency_manager.clone(),
            memory_manager: self.memory_manager.clone(),
            batch_executor: self.batch_executor.clone(),
            stream_manager: self.stream_manager.clone(),
            dataset_manager: self.dataset_manager.clone(),
            security_auditor: self.security_auditor.clone(),
            ddos_protector: self.ddos_protector.clone(),
            load_balancer: self.load_balancer.clone(),
            edge_cache_manager: self.edge_cache_manager.clone(),
            performance_profiler: self.performance_profiler.clone(),
            notification_manager: self.notification_manager.clone(),
            backup_manager: self.backup_manager.clone(),
            recovery_manager: self.recovery_manager.clone(),
            disaster_recovery: self.disaster_recovery.clone(),
            certificate_rotation: self.certificate_rotation.clone(),
            http2_manager: self.http2_manager.clone(),
            http3_manager: self.http3_manager.clone(),
            adaptive_execution_engine: self.adaptive_execution_engine.clone(),
            rebac_manager: self.rebac_manager.clone(),
            prefix_store: Arc::new(handlers::PrefixStore::new()),
            task_manager: Arc::new(handlers::TaskManager::new()),
            request_logger: Arc::new(handlers::RequestLogger::new()),
            startup_time: Instant::now(),
            system_monitor: Arc::new(parking_lot::Mutex::new(sysinfo::System::new_all())),
            #[cfg(feature = "rate-limit")]
            rate_limiter: self.rate_limiter.clone(),
        };
        if let Some(subscription_manager) = &self.subscription_manager {
            subscription_manager.start().await;
        }
        let app_state_arc = Arc::new(app_state);
        let app = self.build_app(app_state_arc).await?;
        info!("Starting OxiRS Fuseki server on {}", addr);
        info!("Server configuration: {:#?}", config.server);
        let graceful_shutdown =
            Self::create_graceful_shutdown(config.server.graceful_shutdown_timeout_secs);
        #[cfg(feature = "tls")]
        if let Some(tls_config) = &config.server.tls {
            info!("TLS enabled - starting HTTPS server");
            self.run_tls_server(addr, app, tls_config.clone(), graceful_shutdown)
                .await?;
        } else {
            info!("TLS disabled - starting HTTP server");
            self.run_http_server(addr, app, graceful_shutdown).await?;
        }
        #[cfg(not(feature = "tls"))]
        {
            if config.server.tls.is_some() {
                warn!("TLS configured but TLS feature not enabled. Starting HTTP server.");
            }
            self.run_http_server(addr, app, graceful_shutdown).await?;
        }
        info!("Server shutdown complete");
        Ok(())
    }
    /// Build the application with all routes and middleware
    async fn build_app(&self, state: Arc<AppState>) -> FusekiResult<Router> {
        let mut app = Router::new();
        app = app.route(
            "/sparql",
            get(handlers::query_handler_get).post(handlers::query_handler_post),
        );
        app = app.route(
            "/graph",
            get(handlers::handle_gsp_get_server)
                .head(handlers::handle_gsp_head_server)
                .put(handlers::handle_gsp_put_server)
                .post(handlers::handle_gsp_post_server)
                .delete(handlers::handle_gsp_delete_server)
                .options(handlers::handle_gsp_options_server),
        );
        app = app.route("/shacl", post(handlers::handle_shacl_validation_server));
        app = app.route("/upload", post(handlers::handle_upload_server));
        app = app.route("/patch", post(handlers::handle_patch_server));
        app = app
            .route(
                "/$/prefixes",
                get(prefix_list_handler).post(prefix_add_handler),
            )
            .route(
                "/$/prefixes/:prefix",
                get(prefix_get_handler)
                    .put(prefix_update_handler)
                    .delete(prefix_delete_handler),
            )
            .route("/$/prefixes/expand", post(prefix_expand_handler));
        app = app
            .route("/$/tasks", get(task_list_handler).post(task_create_handler))
            .route("/$/tasks/statistics", get(task_statistics_handler))
            .route(
                "/$/tasks/:id",
                get(task_get_handler).delete(task_delete_handler),
            )
            .route("/$/tasks/:id/cancel", post(task_cancel_handler));
        app = app
            .route("/$/logs", get(logs_get_handler).delete(logs_clear_handler))
            .route("/$/logs/statistics", get(logs_statistics_handler))
            .route(
                "/$/logs/config",
                get(logs_config_get_handler).put(logs_config_update_handler),
            );
        app = app
            .route("/$/stats", get(stats_server_handler))
            .route("/$/stats/:dataset", get(stats_dataset_handler));
        app = app
            .route(
                "/$/performance/stats",
                get(handlers::performance::get_performance_stats),
            )
            .route(
                "/$/performance/memory",
                get(handlers::performance::get_memory_stats),
            )
            .route(
                "/$/performance/concurrency",
                get(handlers::performance::get_concurrency_stats),
            )
            .route(
                "/$/performance/memory/gc",
                post(handlers::performance::trigger_gc),
            )
            .route(
                "/$/performance/health",
                get(handlers::performance::beta2_health_check),
            );
        app = app
            .route(
                "/$/profiler/report",
                get(handlers::performance::profiler_report_handler),
            )
            .route(
                "/$/profiler/query-stats",
                get(handlers::performance::profiler_query_stats_handler),
            )
            .route(
                "/$/profiler/reset",
                post(handlers::performance::profiler_reset_handler),
            );
        app = app
            .route(
                "/$/load-balancer/status",
                get(handlers::production::load_balancer_status),
            )
            .route(
                "/$/load-balancer/backends",
                get(handlers::production::list_backends).post(handlers::production::add_backend),
            )
            .route(
                "/$/load-balancer/backends/:id",
                axum::routing::delete(handlers::production::remove_backend),
            )
            .route(
                "/$/load-balancer/select",
                post(handlers::production::select_backend),
            );
        app = app
            .route(
                "/$/edge-cache/status",
                get(handlers::production::edge_cache_status),
            )
            .route(
                "/$/edge-cache/purge",
                post(handlers::production::purge_cache),
            )
            .route(
                "/$/edge-cache/headers",
                post(handlers::production::get_cache_headers),
            );
        app = app
            .route("/$/cdn/config", get(handlers::production::cdn_config))
            .route(
                "/static/*path",
                get(handlers::production::serve_static_asset),
            );
        app = app
            .route(
                "/$/security/audit/status",
                get(handlers::production::security_audit_status),
            )
            .route(
                "/$/security/audit/scan",
                post(handlers::production::trigger_security_scan),
            );
        app = app
            .route(
                "/$/security/ddos/status",
                get(handlers::production::ddos_status),
            )
            .route(
                "/$/security/ddos/manage-ip",
                post(handlers::production::manage_ip),
            );
        app = app
            .route(
                "/$/recovery/status",
                get(handlers::production::disaster_recovery_status),
            )
            .route(
                "/$/recovery/create-point",
                post(handlers::production::create_recovery_point),
            );
        app = app
            .route(
                "/$/api-keys",
                get(handlers::api_keys::list_api_keys).post(handlers::api_keys::create_api_key),
            )
            .route(
                "/$/api-keys/:key_id",
                get(handlers::api_keys::get_api_key)
                    .put(handlers::api_keys::update_api_key)
                    .delete(handlers::api_keys::revoke_api_key),
            )
            .route(
                "/$/api-keys/:key_id/usage",
                get(handlers::api_keys::get_api_key_usage),
            );
        app = app.route("/update", post(handlers::sparql::update_handler));
        app = app
            .route("/$/datasets", get(handlers::admin::list_datasets))
            .route(
                "/$/datasets/:name",
                get(handlers::admin::get_dataset)
                    .post(handlers::admin::create_dataset)
                    .delete(handlers::admin::delete_dataset),
            );
        info!("Enabling ReBAC management API routes");
        app = app
            .route("/$/rebac/check", post(handlers::check_permission))
            .route(
                "/$/rebac/batch-check",
                post(handlers::batch_check_permissions),
            )
            .route(
                "/$/rebac/tuples",
                get(handlers::list_tuples)
                    .post(handlers::add_tuple)
                    .delete(handlers::remove_tuple),
            );
        app = app
            .route("/$/ping", get(ping_handler))
            .route("/$/server", get(handlers::admin::server_info))
            .route("/$/stats", get(handlers::admin::server_stats))
            .route("/$/compact/:name", post(handlers::admin::compact_dataset))
            .route("/$/backup/:name", post(handlers::admin::backup_dataset))
            .route("/$/backups-list", get(handlers::list_backups))
            .route("/$/reload", post(handlers::reload_config));
        app = app
            .route(
                "/$/validate/query",
                get(handlers::validate_query_get).post(handlers::validate_query),
            )
            .route(
                "/$/validate/update",
                get(handlers::validate_update_get).post(handlers::validate_update),
            )
            .route(
                "/$/validate/iri",
                get(handlers::validate_iri_get).post(handlers::validate_iri),
            )
            .route("/$/validate/data", post(handlers::validate_data))
            .route(
                "/$/validate/langtag",
                get(handlers::validate_langtag_get).post(handlers::validate_langtag),
            );
        if self.config.security.auth_required {
            app = app
                .route("/$/login", post(handlers::auth::login_handler))
                .route("/$/logout", post(handlers::auth::logout_handler))
                .route("/$/user", get(handlers::auth::user_info_handler))
                .route("/$/users", get(handlers::auth::list_users_handler));
        }
        if self.config.security.oauth.is_some() {
            app = app
                .route(
                    "/auth/oauth2/authorize",
                    get(handlers::oauth2::initiate_oauth2_flow),
                )
                .route(
                    "/auth/oauth2/callback",
                    get(handlers::oauth2::handle_oauth2_callback),
                )
                .route(
                    "/auth/oauth2/refresh",
                    post(handlers::oauth2::refresh_oauth2_token),
                )
                .route(
                    "/auth/oauth2/userinfo",
                    get(handlers::oauth2::get_oauth2_user_info),
                )
                .route(
                    "/auth/oauth2/validate",
                    get(handlers::oauth2::validate_oauth2_token),
                )
                .route(
                    "/auth/oauth2/config",
                    get(handlers::oauth2::get_oauth2_config),
                )
                .route(
                    "/auth/oauth2/.well-known/openid_configuration",
                    get(handlers::oauth2::oauth2_discovery),
                );
        }
        #[cfg(feature = "ldap")]
        if self.config.security.ldap.is_some() {
            app = app
                .route("/auth/ldap/login", post(handlers::ldap_login))
                .route("/auth/ldap/test", get(handlers::test_ldap_connection))
                .route("/auth/ldap/groups", get(handlers::get_ldap_groups))
                .route("/auth/ldap/config", get(handlers::get_ldap_config));
        }
        if let Some(mfa_config) = &self.config.security.mfa {
            if mfa_config.enabled {
                app = app
                    .route("/auth/mfa/enroll", post(handlers::enroll_mfa))
                    .route(
                        "/auth/mfa/challenge/:type",
                        post(handlers::create_mfa_challenge),
                    )
                    .route("/auth/mfa/verify", post(handlers::verify_mfa))
                    .route("/auth/mfa/status", get(handlers::get_mfa_status))
                    .route("/auth/mfa/disable/:type", delete(handlers::disable_mfa))
                    .route(
                        "/auth/mfa/backup-codes",
                        post(handlers::regenerate_backup_codes),
                    );
            }
        }
        app = app
            .route("/health", get(crate::health::health_handler))
            .route("/health/live", get(crate::health::liveness_handler))
            .route("/health/ready", get(crate::health::readiness_handler));
        if state.metrics_service.is_some() {
            app = app.route("/metrics", get(handlers::production::metrics_handler));
        }
        app = app
            .route(
                "/$/performance",
                get(handlers::performance::get_performance_stats),
            )
            .route(
                "/$/performance/memory",
                get(handlers::performance::get_memory_stats),
            )
            .route(
                "/$/performance/concurrency",
                get(handlers::performance::get_concurrency_stats),
            )
            .route("/$/performance/gc", post(handlers::performance::trigger_gc))
            .route(
                "/$/performance/health",
                get(handlers::performance::beta2_health_check),
            );
        if state.performance_profiler.is_some() {
            app = app
                .route(
                    "/$/profiler/report",
                    get(handlers::performance::profiler_report_handler),
                )
                .route(
                    "/$/profiler/query-stats",
                    get(handlers::performance::profiler_query_stats_handler),
                )
                .route(
                    "/$/profiler/reset",
                    post(handlers::performance::profiler_reset_handler),
                );
        }
        if state.query_optimizer.is_some() {
            app = app
                .route(
                    "/$/optimization/stats",
                    get(handlers::performance::optimization_stats_handler),
                )
                .route(
                    "/$/optimization/plans",
                    get(handlers::performance::optimization_plans_handler),
                )
                .route(
                    "/$/optimization/cache",
                    delete(handlers::performance::clear_optimization_cache_handler),
                )
                .route(
                    "/$/optimization/database",
                    get(handlers::performance::database_statistics_handler),
                );
        }
        app = app
            .route("/$/ws", get(handlers::websocket_handler))
            .route("/$/subscribe", get(handlers::websocket_handler));
        info!("Enabling GraphQL API routes");
        app = app
            .route(
                "/graphql",
                post(crate::graphql_integration::graphql_handler),
            )
            .route(
                "/graphql/playground",
                get(crate::graphql_integration::graphql_playground),
            );
        app = app
            .route(
                "/ngsi-ld/v1/entities",
                get(handlers::ngsi_query_entities).post(handlers::ngsi_create_entity),
            )
            .route(
                "/ngsi-ld/v1/entities/:id",
                get(handlers::ngsi_get_entity).delete(handlers::ngsi_delete_entity),
            )
            .route(
                "/ngsi-ld/v1/entities/:id/attrs",
                post(handlers::ngsi_ld::append_entity_attrs_server)
                    .patch(handlers::ngsi_update_entity),
            )
            .route(
                "/ngsi-ld/v1/entities/:id/attrs/:attrId",
                delete(handlers::ngsi_ld::delete_entity_attr_server),
            )
            .route(
                "/ngsi-ld/v1/subscriptions",
                get(handlers::ngsi_list_subscriptions).post(handlers::ngsi_create_subscription),
            )
            .route(
                "/ngsi-ld/v1/subscriptions/:id",
                get(handlers::ngsi_get_subscription)
                    .patch(handlers::ngsi_update_subscription)
                    .delete(handlers::ngsi_delete_subscription),
            )
            .route(
                "/ngsi-ld/v1/entityOperations/create",
                post(handlers::ngsi_batch_create),
            )
            .route(
                "/ngsi-ld/v1/entityOperations/upsert",
                post(handlers::ngsi_batch_upsert),
            )
            .route(
                "/ngsi-ld/v1/entityOperations/update",
                post(handlers::ngsi_batch_update),
            )
            .route(
                "/ngsi-ld/v1/entityOperations/delete",
                post(handlers::ngsi_batch_delete),
            )
            .route(
                "/ngsi-ld/v1/temporal/entities",
                get(handlers::ngsi_query_temporal).post(handlers::ngsi_create_temporal),
            )
            .route(
                "/ngsi-ld/v1/temporal/entities/:id",
                get(handlers::ngsi_get_temporal).delete(handlers::ngsi_delete_temporal),
            );
        app = crate::rest_api_v2::register_routes(app);
        if let Some(ids_api_state) = &self.ids_api_state {
            let ids_router = crate::ids::ids_router(ids_api_state.clone());
            app = app.nest_service("/api/ids", ids_router);
            info!("IDS API mounted at /api/ids");
        }
        #[cfg(feature = "admin-ui")]
        if self.config.server.admin_ui {
            app = app.route("/", get(handlers::ui_handler));
        }
        app = self.apply_middleware_stack(app, state.clone()).await?;
        Ok(app.with_state(state))
    }
    /// Apply comprehensive middleware stack
    async fn apply_middleware_stack(
        &self,
        mut app: Router<Arc<AppState>>,
        state: Arc<AppState>,
    ) -> FusekiResult<Router<Arc<AppState>>> {
        use crate::middleware::{
            api_version, health_check_bypass, https_security_headers, request_correlation_id,
            request_timing, route_based_rbac, security_headers,
        };
        use tower_http::{
            cors::CorsLayer, request_id::SetRequestIdLayer, timeout::TimeoutLayer,
            trace::TraceLayer,
        };
        app = app.layer(axum::middleware::from_fn(health_check_bypass));
        if let Some(ddos_protector) = &state.ddos_protector {
            let ddos = ddos_protector.clone();
            app = app.layer(axum::middleware::from_fn(
                move |req: Request, next: Next| {
                    let ddos = ddos.clone();
                    async move {
                        let client_ip = req
                            .headers()
                            .get("x-forwarded-for")
                            .and_then(|v| v.to_str().ok())
                            .and_then(|s| s.split(',').next())
                            .and_then(|s| s.parse::<std::net::IpAddr>().ok())
                            .unwrap_or_else(|| "127.0.0.1".parse().expect("localhost IP is valid"));
                        match ddos.check_request(client_ip).await {
                            Ok(crate::ddos_protection::RequestDecision::Allow) => {
                                ddos.register_connection(client_ip).await;
                                let response = next.run(req).await;
                                ddos.unregister_connection(client_ip).await;
                                response
                            }
                            Ok(crate::ddos_protection::RequestDecision::RateLimit { .. })
                            | Ok(crate::ddos_protection::RequestDecision::Block { .. }) => {
                                axum::response::Response::builder()
                                    .status(axum::http::StatusCode::TOO_MANY_REQUESTS)
                                    .body(axum::body::Body::from("Rate limit exceeded"))
                                    .expect("response body build should succeed")
                            }
                            Ok(crate::ddos_protection::RequestDecision::Challenge { .. }) => {
                                axum::response::Response::builder()
                                    .status(axum::http::StatusCode::TOO_MANY_REQUESTS)
                                    .body(axum::body::Body::from("Please solve challenge"))
                                    .expect("response body build should succeed")
                            }
                            Err(_) => next.run(req).await,
                        }
                    }
                },
            ));
            info!("DDoS protection middleware enabled");
        }
        if let Some(security_auditor) = &state.security_auditor {
            let auditor = security_auditor.clone();
            app = app.layer(axum::middleware::from_fn(move |req, next| {
                let auditor = auditor.clone();
                crate::middleware::security_audit_middleware(auditor, req, next)
            }));
            info!("Security audit middleware enabled");
        }
        app = app.layer(axum::middleware::from_fn(security_headers));
        if self.config.server.tls.is_some() {
            app = app.layer(axum::middleware::from_fn(https_security_headers));
        }
        app = app.layer(axum::middleware::from_fn(request_correlation_id));
        app = app.layer(axum::middleware::from_fn(request_timing));
        app = app.layer(axum::middleware::from_fn(api_version));
        if self.config.security.auth_required {
            info!("RBAC middleware enabled - enforcing role-based access control");
            app = app.layer(axum::middleware::from_fn(route_based_rbac));
        } else {
            debug!("RBAC middleware disabled - authentication not required");
        }
        app = app.layer(SetRequestIdLayer::x_request_id(RequestIdGenerator));
        app = app.layer(TraceLayer::new_for_http());
        app = app.layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(self.config.server.request_timeout_secs),
        ));
        if self.config.server.cors {
            let cors = CorsLayer::new()
                .allow_origin(tower_http::cors::Any)
                .allow_methods([
                    axum::http::Method::GET,
                    axum::http::Method::POST,
                    axum::http::Method::PUT,
                    axum::http::Method::DELETE,
                    axum::http::Method::OPTIONS,
                ])
                .allow_headers(tower_http::cors::Any)
                .expose_headers([
                    axum::http::HeaderName::from_static("x-request-id"),
                    axum::http::HeaderName::from_static("x-response-time"),
                    axum::http::HeaderName::from_static("x-api-version"),
                ]);
            app = app.layer(cors);
        }
        info!("Middleware stack configured: security, tracing, timing, CORS");
        Ok(app)
    }
    /// Run HTTP server (without TLS)
    async fn run_http_server<F>(
        &self,
        addr: SocketAddr,
        app: Router,
        graceful_shutdown: F,
    ) -> FusekiResult<()>
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| FusekiError::internal(format!("Failed to bind to {addr}: {e}")))?;
        axum::serve(listener, app)
            .with_graceful_shutdown(graceful_shutdown)
            .await
            .map_err(|e| FusekiError::internal(format!("Server error: {e}")))?;
        Ok(())
    }
    /// Run HTTPS server with TLS
    #[cfg(feature = "tls")]
    async fn run_tls_server<F>(
        &self,
        addr: SocketAddr,
        app: Router,
        tls_config: TlsConfig,
        graceful_shutdown: F,
    ) -> FusekiResult<()>
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        use axum_server::tls_rustls::RustlsConfig;
        let tls_manager = TlsManager::new(tls_config.clone());
        tls_manager.validate()?;
        let rustls_config = tls_manager.build_server_config()?;
        let axum_tls_config = RustlsConfig::from_config(rustls_config);
        info!("TLS certificates loaded successfully");
        info!("Starting HTTPS server on https://{}", addr);
        let handle = axum_server::Handle::new();
        tokio::spawn({
            let handle = handle.clone();
            async move {
                graceful_shutdown.await;
                handle.graceful_shutdown(Some(Duration::from_secs(30)));
            }
        });
        axum_server::bind_rustls(addr, axum_tls_config)
            .handle(handle)
            .serve(app.into_make_service())
            .await
            .map_err(|e| FusekiError::internal(format!("TLS server error: {e}")))?;
        Ok(())
    }
    /// Graceful shutdown with configurable timeout
    async fn create_graceful_shutdown(graceful_shutdown_timeout_secs: u64) {
        let shutdown_timeout = Duration::from_secs(graceful_shutdown_timeout_secs);
        let ctrl_c = async {
            signal::ctrl_c()
                .await
                .expect("failed to install Ctrl+C handler");
        };
        #[cfg(unix)]
        let terminate = async {
            signal::unix::signal(signal::unix::SignalKind::terminate())
                .expect("failed to install signal handler")
                .recv()
                .await;
        };
        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();
        tokio::select! {
            _ = ctrl_c => { info!("Received Ctrl+C, initiating graceful shutdown"); }, _
            = terminate => { info!("Received SIGTERM, initiating graceful shutdown"); },
        }
        info!(
            "Allowing {}s for graceful shutdown...",
            shutdown_timeout.as_secs()
        );
        tokio::time::sleep(shutdown_timeout).await;
        warn!("Graceful shutdown timeout reached, forcing exit");
    }
}
/// Request UUID generator for request IDs
#[derive(Clone)]
pub struct RequestIdGenerator;
/// Application state shared across all handlers and middleware
#[derive(Clone)]
pub struct AppState {
    pub store: Store,
    pub config: ServerConfig,
    pub auth_service: Option<AuthService>,
    pub metrics_service: Option<Arc<MetricsService>>,
    pub performance_service: Option<Arc<PerformanceService>>,
    pub query_optimizer: Option<Arc<QueryOptimizer>>,
    pub subscription_manager: Option<Arc<SubscriptionManager>>,
    pub federation_manager: Option<Arc<FederationManager>>,
    pub streaming_manager: Option<Arc<StreamingManager>>,
    pub concurrency_manager: Option<Arc<ConcurrencyManager>>,
    pub memory_manager: Option<Arc<MemoryManager>>,
    pub batch_executor: Option<Arc<BatchExecutor>>,
    pub stream_manager: Option<Arc<StreamManager>>,
    pub dataset_manager: Option<Arc<DatasetManager>>,
    pub security_auditor: Option<Arc<SecurityAuditManager>>,
    pub ddos_protector: Option<Arc<DDoSProtectionManager>>,
    pub load_balancer: Option<Arc<LoadBalancer>>,
    pub edge_cache_manager: Option<Arc<EdgeCacheManager>>,
    pub performance_profiler: Option<Arc<PerformanceProfiler>>,
    pub notification_manager: Option<Arc<NotificationManager>>,
    pub backup_manager: Option<Arc<BackupManager>>,
    pub recovery_manager: Option<Arc<RecoveryManager>>,
    pub disaster_recovery: Option<Arc<DisasterRecoveryManager>>,
    pub certificate_rotation: Option<Arc<CertificateRotation>>,
    pub http2_manager: Option<Arc<Http2Manager>>,
    pub http3_manager: Option<Arc<Http3Manager>>,
    pub adaptive_execution_engine: Option<Arc<AdaptiveExecutionEngine>>,
    pub rebac_manager: Option<Arc<dyn crate::auth::rebac::RebacEvaluator>>,
    pub prefix_store: Arc<handlers::PrefixStore>,
    pub task_manager: Arc<handlers::TaskManager>,
    pub request_logger: Arc<handlers::RequestLogger>,
    pub startup_time: Instant,
    pub system_monitor: Arc<parking_lot::Mutex<sysinfo::System>>,
    #[cfg(feature = "rate-limit")]
    pub rate_limiter: Option<Arc<governor::DefaultKeyedRateLimiter<String>>>,
}
