//! # WebAssembly Edge Computing Module
//!
//! Ultra-low latency edge processing using WebAssembly with hot-swappable plugins,
//! distributed execution, and advanced resource management for next-generation streaming.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, info, warn};

#[cfg(feature = "wasm")]
use wasmtime::{Engine, Instance, Module, Store, TypedFunc};

use crate::event::StreamEvent;

// Sibling modules
pub mod wasm_edge_computing_runtime;
pub mod wasm_edge_computing_sandbox;

// Re-export runtime types
pub use wasm_edge_computing_runtime::{
    AllocationConstraints, AllocationEvent, AllocationPlan, CacheOptimizer, CachedModule,
    DependencyGraph, EdgeResourceOptimizer, ExecutionProfile, ModelType, NodeAssignment,
    OptimizationMetrics, OptimizationStrategy, PredictionEngine, PrefetchPredictor, PriorityLevel,
    ResourceAllocation, ResourceModel, ResourcePrediction, SeasonalityType, TemporalPattern,
    WasmIntelligentCache, WorkloadDescription, WorkloadFeatures,
};

// Re-export sandbox types
pub use wasm_edge_computing_sandbox::{
    AdaptivePolicy, AdaptiveSecuritySandbox, BehaviorAnomaly, BehaviorProfile, BehavioralAnalyzer,
    ExecutionBehavior, ImpactLevel, MemoryAccessPattern, NetworkActivityLevel, Priority,
    SecurityAssessment, SecurityMetrics, SecurityRecommendation, ThreatDetector, ThreatIndicator,
    ThreatSignature, ThreatType,
};

/// WASM edge processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmEdgeConfig {
    pub optimization_level: OptimizationLevel,
    pub resource_limits: WasmResourceLimits,
    pub enable_caching: bool,
    pub enable_jit: bool,
    pub security_sandbox: bool,
    pub allowed_imports: Vec<String>,
    // Optional legacy fields for backward compatibility
    #[serde(default)]
    pub max_concurrent_instances: usize,
    #[serde(default)]
    pub memory_limit_mb: u64,
    #[serde(default)]
    pub execution_timeout_ms: u64,
    #[serde(default)]
    pub enable_hot_reload: bool,
    #[serde(default)]
    pub edge_locations: Vec<EdgeLocation>,
}

/// WASM resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmResourceLimits {
    pub max_memory_bytes: u64,
    pub max_execution_time_ms: u64,
    pub max_stack_size_bytes: u64,
    pub max_table_elements: u32,
    pub enable_simd: bool,
    pub enable_threads: bool,
    // Legacy fields
    #[serde(default)]
    pub max_memory_pages: u32,
    #[serde(default)]
    pub max_instances: u32,
    #[serde(default)]
    pub max_tables: u32,
    #[serde(default)]
    pub max_memories: u32,
    #[serde(default)]
    pub max_globals: u32,
    #[serde(default)]
    pub max_functions: u32,
    #[serde(default)]
    pub max_imports: u32,
    #[serde(default)]
    pub max_exports: u32,
}

/// Edge computing location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeLocation {
    pub id: String,
    pub region: String,
    pub latency_ms: f64,
    pub capacity_factor: f64,
    pub available_resources: ResourceMetrics,
    pub specializations: Vec<ProcessingSpecialization>,
}

/// Processing specializations for edge nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingSpecialization {
    RdfProcessing,
    SparqlOptimization,
    GraphAnalytics,
    MachineLearning,
    Cryptography,
    ComputerVision,
    NaturalLanguage,
    QuantumSimulation,
}

/// WASM optimization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    Debug,
    Release,
    Maximum,
    Adaptive,
}

/// Resource usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub cpu_cores: u32,
    pub memory_mb: u64,
    pub storage_gb: u64,
    pub network_mbps: f64,
    pub gpu_units: u32,
    pub quantum_qubits: u32,
}

/// WASM plugin metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmPlugin {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub capabilities: Vec<PluginCapability>,
    pub wasm_bytes: Vec<u8>,
    pub schema: PluginSchema,
    pub performance_profile: PerformanceProfile,
    pub security_level: SecurityLevel,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Plugin capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginCapability {
    EventProcessing,
    DataTransformation,
    Filtering,
    Aggregation,
    Enrichment,
    Validation,
    Compression,
    Encryption,
    Analytics,
    MachineLearning,
}

/// Plugin input/output schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginSchema {
    pub input_types: Vec<String>,
    pub output_types: Vec<String>,
    pub configuration_schema: serde_json::Value,
    pub required_imports: Vec<String>,
    pub exported_functions: Vec<String>,
}

impl Default for PluginSchema {
    fn default() -> Self {
        Self {
            input_types: vec!["StreamEvent".to_string()],
            output_types: vec!["StreamEvent".to_string()],
            configuration_schema: serde_json::json!({}),
            required_imports: vec![],
            exported_functions: vec!["process_events".to_string()],
        }
    }
}

/// Performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    pub average_execution_time_us: u64,
    pub memory_usage_mb: f64,
    pub cpu_intensity: f64,
    pub throughput_events_per_second: u64,
    pub scalability_factor: f64,
}

impl Default for PerformanceProfile {
    fn default() -> Self {
        Self {
            average_execution_time_us: 100,
            memory_usage_mb: 1.0,
            cpu_intensity: 0.5,
            throughput_events_per_second: 1000,
            scalability_factor: 1.0,
        }
    }
}

/// Security levels for plugins
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SecurityLevel {
    Untrusted,
    BasicSandbox,
    Standard,
    Enhanced,
    TrustedVerified,
    CriticalSecurity,
    High,
}

/// WASM execution context
pub struct WasmExecutionContext {
    #[cfg(feature = "wasm")]
    pub engine: Engine,
    #[cfg(feature = "wasm")]
    pub store: Store<WasmState>,
    #[cfg(feature = "wasm")]
    pub instance: Instance,
    pub plugin_id: String,
    pub execution_count: u64,
    pub total_execution_time_us: u64,
    pub last_execution: DateTime<Utc>,
    pub resource_usage: ResourceMetrics,
}

impl std::fmt::Debug for WasmExecutionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WasmExecutionContext")
            .field("plugin_id", &self.plugin_id)
            .field("execution_count", &self.execution_count)
            .field("total_execution_time_us", &self.total_execution_time_us)
            .field("last_execution", &self.last_execution)
            .field("resource_usage", &self.resource_usage)
            .finish_non_exhaustive()
    }
}

/// WASM execution state
#[derive(Debug, Default)]
pub struct WasmState {
    pub event_count: u64,
    pub memory_allocations: u64,
    pub external_calls: u64,
    pub start_time: Option<DateTime<Utc>>,
}

/// Edge execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeExecutionResult {
    pub plugin_id: String,
    pub execution_id: String,
    pub input_events: Vec<StreamEvent>,
    pub output_events: Vec<StreamEvent>,
    pub execution_time_us: u64,
    pub memory_used_mb: f64,
    pub edge_location: String,
    pub success: bool,
    pub error_message: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// WASM processing result for single event processing
#[derive(Debug, Clone)]
pub struct WasmProcessingResult {
    pub output: Option<StreamEvent>,
    pub latency_ms: f64,
}

/// WASM processor statistics
#[derive(Debug, Clone)]
pub struct WasmProcessorStats {
    pub total_processed: u64,
    pub average_latency_ms: f64,
    pub active_plugins: usize,
}

/// Performance tracking for plugins
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_executions: u64,
    pub total_execution_time_us: u64,
    pub average_execution_time_us: f64,
    pub max_execution_time_us: u64,
    pub min_execution_time_us: u64,
    pub success_rate: f64,
    pub throughput_events_per_second: f64,
    pub memory_efficiency: f64,
    pub last_updated: DateTime<Utc>,
}

impl PerformanceMetrics {
    fn new() -> Self {
        Self {
            total_executions: 0,
            total_execution_time_us: 0,
            average_execution_time_us: 0.0,
            max_execution_time_us: 0,
            min_execution_time_us: 0,
            success_rate: 0.0,
            throughput_events_per_second: 0.0,
            memory_efficiency: 0.0,
            last_updated: Utc::now(),
        }
    }
}

/// Risk assessment levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Security manager for WASM execution
#[derive(Debug)]
pub struct SecurityManager {
    trusted_plugins: RwLock<HashMap<String, SecurityLevel>>,
    execution_policies: RwLock<HashMap<SecurityLevel, ExecutionPolicy>>,
    audit_log: RwLock<Vec<SecurityAuditEntry>>,
}

/// Execution policies based on security level
#[derive(Debug, Clone)]
pub struct ExecutionPolicy {
    pub max_memory_pages: u32,
    pub max_execution_time_ms: u64,
    pub allowed_imports: Vec<String>,
    pub network_access: bool,
    pub file_system_access: bool,
    pub crypto_operations: bool,
    pub inter_plugin_communication: bool,
}

/// Security audit entry
#[derive(Debug, Clone)]
pub struct SecurityAuditEntry {
    pub timestamp: DateTime<Utc>,
    pub plugin_id: String,
    pub action: String,
    pub risk_level: RiskLevel,
    pub details: String,
}

impl Default for SecurityManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SecurityManager {
    pub fn new() -> Self {
        let mut execution_policies = HashMap::new();

        execution_policies.insert(
            SecurityLevel::Untrusted,
            ExecutionPolicy {
                max_memory_pages: 1,
                max_execution_time_ms: 100,
                allowed_imports: vec![],
                network_access: false,
                file_system_access: false,
                crypto_operations: false,
                inter_plugin_communication: false,
            },
        );

        execution_policies.insert(
            SecurityLevel::TrustedVerified,
            ExecutionPolicy {
                max_memory_pages: 64,
                max_execution_time_ms: 5000,
                allowed_imports: vec!["env".to_string()],
                network_access: true,
                file_system_access: false,
                crypto_operations: true,
                inter_plugin_communication: true,
            },
        );

        Self {
            trusted_plugins: RwLock::new(HashMap::new()),
            execution_policies: RwLock::new(execution_policies),
            audit_log: RwLock::new(Vec::new()),
        }
    }

    pub async fn validate_plugin(&self, plugin: &WasmPlugin) -> Result<()> {
        self.validate_plugin_metadata(plugin).await?;
        self.scan_wasm_bytecode(&plugin.wasm_bytes).await?;
        self.check_plugin_reputation(&plugin.id).await?;
        Ok(())
    }

    async fn validate_plugin_metadata(&self, _plugin: &WasmPlugin) -> Result<()> {
        Ok(())
    }

    async fn scan_wasm_bytecode(&self, _wasm_bytes: &[u8]) -> Result<()> {
        Ok(())
    }

    async fn check_plugin_reputation(&self, _plugin_id: &str) -> Result<()> {
        Ok(())
    }
}

// Default implementations

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            cpu_cores: 4,
            memory_mb: 8192,
            storage_gb: 256,
            network_mbps: 1000.0,
            gpu_units: 0,
            quantum_qubits: 0,
        }
    }
}

impl Default for WasmEdgeConfig {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::Release,
            resource_limits: WasmResourceLimits::default(),
            enable_caching: true,
            enable_jit: true,
            security_sandbox: true,
            allowed_imports: vec!["env".to_string(), "wasi_snapshot_preview1".to_string()],
            max_concurrent_instances: 10,
            memory_limit_mb: 512,
            execution_timeout_ms: 5000,
            enable_hot_reload: true,
            edge_locations: vec![],
        }
    }
}

impl Default for WasmResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_bytes: 512 * 1024 * 1024, // 512 MB
            max_execution_time_ms: 5000,
            max_stack_size_bytes: 2 * 1024 * 1024, // 2 MB
            max_table_elements: 10000,
            enable_simd: true,
            enable_threads: false,
            max_memory_pages: 8192, // 512 MB / 64KB per page
            max_instances: 10,
            max_tables: 10,
            max_memories: 10,
            max_globals: 100,
            max_functions: 1000,
            max_imports: 100,
            max_exports: 100,
        }
    }
}

// ============================================================
// WasmEdgeProcessor
// ============================================================

/// Advanced WASM edge computing processor
pub struct WasmEdgeProcessor {
    config: WasmEdgeConfig,
    plugins: RwLock<HashMap<String, WasmPlugin>>,
    execution_contexts: RwLock<HashMap<String, Arc<RwLock<WasmExecutionContext>>>>,
    execution_semaphore: Semaphore,
    edge_registry: RwLock<HashMap<String, EdgeLocation>>,
    plugin_registry: RwLock<HashMap<String, WasmPlugin>>,
    performance_metrics: RwLock<HashMap<String, PerformanceMetrics>>,
    security_manager: SecurityManager,
    #[cfg(feature = "wasm")]
    wasm_engine: Engine,
}

impl WasmEdgeProcessor {
    /// Create new WASM edge processor
    pub fn new(config: WasmEdgeConfig) -> Result<Self> {
        #[cfg(feature = "wasm")]
        let wasm_engine = {
            let mut wasm_config = wasmtime::Config::new();
            wasm_config.debug_info(true);
            wasm_config.wasm_simd(true);
            wasm_config.wasm_bulk_memory(true);
            wasm_config.wasm_reference_types(true);
            wasm_config.wasm_multi_value(true);
            wasm_config.cranelift_opt_level(wasmtime::OptLevel::Speed);
            Engine::new(&wasm_config)?
        };

        #[cfg(not(feature = "wasm"))]
        let _wasm_engine = ();

        let execution_semaphore = Semaphore::new(config.max_concurrent_instances);
        let security_manager = SecurityManager::new();

        Ok(Self {
            config,
            plugins: RwLock::new(HashMap::new()),
            execution_contexts: RwLock::new(HashMap::new()),
            execution_semaphore,
            edge_registry: RwLock::new(HashMap::new()),
            plugin_registry: RwLock::new(HashMap::new()),
            performance_metrics: RwLock::new(HashMap::new()),
            security_manager,
            #[cfg(feature = "wasm")]
            wasm_engine,
        })
    }

    /// Register a WASM plugin
    pub async fn register_plugin(&self, plugin: WasmPlugin) -> Result<()> {
        self.security_manager.validate_plugin(&plugin).await?;

        let plugin_id = plugin.id.clone();

        #[cfg(feature = "wasm")]
        {
            let module = Module::new(&self.wasm_engine, &plugin.wasm_bytes)
                .map_err(|e| anyhow!("Failed to compile WASM module: {}", e))?;

            let mut store = Store::new(&self.wasm_engine, WasmState::default());
            let instance = Instance::new(&mut store, &module, &[])
                .map_err(|e| anyhow!("Failed to instantiate WASM module: {}", e))?;

            let context = WasmExecutionContext {
                engine: self.wasm_engine.clone(),
                store,
                instance,
                plugin_id: plugin_id.clone(),
                execution_count: 0,
                total_execution_time_us: 0,
                last_execution: Utc::now(),
                resource_usage: ResourceMetrics::default(),
            };

            self.execution_contexts
                .write()
                .await
                .insert(plugin_id.clone(), Arc::new(RwLock::new(context)));
        }

        self.plugins
            .write()
            .await
            .insert(plugin_id.clone(), plugin.clone());
        self.plugin_registry
            .write()
            .await
            .insert(plugin_id.clone(), plugin);

        self.performance_metrics
            .write()
            .await
            .insert(plugin_id.clone(), PerformanceMetrics::new());

        info!("Registered WASM plugin: {}", plugin_id);
        Ok(())
    }

    /// Execute plugin on edge location
    pub async fn execute_plugin(
        &self,
        plugin_id: &str,
        events: Vec<StreamEvent>,
        edge_location: Option<String>,
    ) -> Result<EdgeExecutionResult> {
        let _permit = self
            .execution_semaphore
            .acquire()
            .await
            .map_err(|_| anyhow!("Failed to acquire execution permit"))?;

        let start_time = std::time::Instant::now();
        let execution_id = uuid::Uuid::new_v4().to_string();

        let edge_id = if let Some(location) = edge_location {
            location
        } else {
            self.select_optimal_edge_location(plugin_id, &events)
                .await?
        };

        let plugin = {
            let plugins = self.plugins.read().await;
            plugins
                .get(plugin_id)
                .ok_or_else(|| anyhow!("Plugin not found: {}", plugin_id))?
                .clone()
        };

        let result = self
            .execute_plugin_internal(&plugin, events.clone(), &edge_id, &execution_id)
            .await;

        let execution_time = start_time.elapsed();

        self.update_performance_metrics(plugin_id, &result, execution_time.as_micros() as u64)
            .await;

        let execution_result = EdgeExecutionResult {
            plugin_id: plugin_id.to_string(),
            execution_id,
            input_events: events,
            output_events: result.as_ref().map(|r| r.clone()).unwrap_or_default(),
            execution_time_us: execution_time.as_micros() as u64,
            memory_used_mb: self.estimate_memory_usage(&plugin).await,
            edge_location: edge_id,
            success: result.is_ok(),
            error_message: result.as_ref().err().map(|e| e.to_string()),
            metadata: HashMap::new(),
        };

        match result {
            Ok(output_events) => {
                debug!(
                    "Plugin execution successful: {} events processed in {:?}",
                    output_events.len(),
                    execution_time
                );
                Ok(EdgeExecutionResult {
                    output_events,
                    ..execution_result
                })
            }
            Err(e) => {
                warn!("Plugin execution failed: {}", e);
                Ok(execution_result)
            }
        }
    }

    /// Internal plugin execution
    async fn execute_plugin_internal(
        &self,
        plugin: &WasmPlugin,
        events: Vec<StreamEvent>,
        _edge_id: &str,
        _execution_id: &str,
    ) -> Result<Vec<StreamEvent>> {
        #[cfg(not(feature = "wasm"))]
        let _ = plugin;

        #[cfg(feature = "wasm")]
        {
            let context_arc = {
                let contexts = self.execution_contexts.read().await;
                contexts
                    .get(&plugin.id)
                    .ok_or_else(|| {
                        anyhow!("Execution context not found for plugin: {}", plugin.id)
                    })?
                    .clone()
            };

            let mut context = context_arc.write().await;

            context.store.data_mut().start_time = Some(Utc::now());
            context.store.data_mut().event_count = 0;

            let process_events: TypedFunc<(i32, i32), i32> = {
                let WasmExecutionContext {
                    ref instance,
                    ref mut store,
                    ..
                } = *context;
                instance
                    .get_typed_func(store, "process_events")
                    .map_err(|e| anyhow!("Failed to get process_events function: {}", e))?
            };

            let input_json = serde_json::to_string(&events)?;
            let input_ptr = self.allocate_memory(&mut context, input_json.as_bytes())?;

            let output_ptr = process_events
                .call(&mut context.store, (input_ptr, input_json.len() as i32))
                .map_err(|e| anyhow!("WASM execution failed: {}", e))?;

            let output_data = self.read_memory(&mut context, output_ptr)?;
            let output_json = String::from_utf8(output_data)?;
            let output_events: Vec<StreamEvent> = serde_json::from_str(&output_json)?;

            context.execution_count += 1;
            context.last_execution = Utc::now();

            Ok(output_events)
        }

        #[cfg(not(feature = "wasm"))]
        {
            warn!("WASM feature disabled, returning input events unchanged");
            Ok(events)
        }
    }

    /// Select optimal edge location for execution
    async fn select_optimal_edge_location(
        &self,
        plugin_id: &str,
        events: &[StreamEvent],
    ) -> Result<String> {
        let edge_registry = self.edge_registry.read().await;

        if edge_registry.is_empty() {
            return Ok("default".to_string());
        }

        let mut best_edge = None;
        let mut best_score = f64::NEG_INFINITY;

        for (edge_id, edge_location) in edge_registry.iter() {
            let score = self
                .calculate_edge_score(plugin_id, events, edge_location)
                .await;
            if score > best_score {
                best_score = score;
                best_edge = Some(edge_id.clone());
            }
        }

        best_edge.ok_or_else(|| anyhow!("No suitable edge location found"))
    }

    /// Calculate edge location suitability score
    pub(crate) async fn calculate_edge_score(
        &self,
        _plugin_id: &str,
        _events: &[StreamEvent],
        edge: &EdgeLocation,
    ) -> f64 {
        let latency_score = 1.0 / (1.0 + edge.latency_ms / 100.0);
        let capacity_score = edge.capacity_factor;
        let resource_score = (edge.available_resources.cpu_cores as f64 / 32.0).min(1.0);

        latency_score * 0.4 + capacity_score * 0.3 + resource_score * 0.3
    }

    /// Allocate memory in WASM instance
    #[cfg(feature = "wasm")]
    fn allocate_memory(&self, context: &mut WasmExecutionContext, data: &[u8]) -> Result<i32> {
        let allocate: TypedFunc<i32, i32> = {
            let instance = &context.instance;
            let store = &mut context.store;
            instance
                .get_typed_func(store, "allocate")
                .map_err(|e| anyhow!("Failed to get allocate function: {}", e))?
        };

        let ptr = allocate
            .call(&mut context.store, data.len() as i32)
            .map_err(|e| anyhow!("Memory allocation failed: {}", e))?;

        let memory = {
            let instance = &context.instance;
            let store = &mut context.store;
            instance
                .get_memory(store, "memory")
                .ok_or_else(|| anyhow!("Failed to get WASM memory"))?
        };

        memory
            .write(&mut context.store, ptr as usize, data)
            .map_err(|e| anyhow!("Failed to write to WASM memory: {}", e))?;

        Ok(ptr)
    }

    /// Read memory from WASM instance
    #[cfg(feature = "wasm")]
    fn read_memory(&self, context: &mut WasmExecutionContext, ptr: i32) -> Result<Vec<u8>> {
        let memory = {
            let instance = &context.instance;
            let store = &mut context.store;
            instance
                .get_memory(store, "memory")
                .ok_or_else(|| anyhow!("Failed to get WASM memory"))?
        };

        let mut len_bytes = [0u8; 4];
        memory
            .read(&context.store, ptr as usize, &mut len_bytes)
            .map_err(|e| anyhow!("Failed to read length from WASM memory: {}", e))?;

        let len = u32::from_le_bytes(len_bytes) as usize;

        let mut data = vec![0u8; len];
        memory
            .read(&context.store, (ptr + 4) as usize, &mut data)
            .map_err(|e| anyhow!("Failed to read data from WASM memory: {}", e))?;

        Ok(data)
    }

    /// Estimate memory usage for plugin
    async fn estimate_memory_usage(&self, plugin: &WasmPlugin) -> f64 {
        let base_memory = plugin.wasm_bytes.len() as f64 / (1024.0 * 1024.0);
        let complexity_factor = plugin.capabilities.len() as f64 * 0.1;
        base_memory + complexity_factor
    }

    /// Update performance metrics
    async fn update_performance_metrics(
        &self,
        plugin_id: &str,
        result: &Result<Vec<StreamEvent>>,
        execution_time_us: u64,
    ) {
        let mut metrics_guard = self.performance_metrics.write().await;
        let metrics = metrics_guard
            .entry(plugin_id.to_string())
            .or_insert_with(PerformanceMetrics::new);

        metrics.total_executions += 1;
        metrics.total_execution_time_us += execution_time_us;
        metrics.average_execution_time_us =
            metrics.total_execution_time_us as f64 / metrics.total_executions as f64;

        if execution_time_us > metrics.max_execution_time_us {
            metrics.max_execution_time_us = execution_time_us;
        }

        if metrics.min_execution_time_us == 0 || execution_time_us < metrics.min_execution_time_us {
            metrics.min_execution_time_us = execution_time_us;
        }

        let success = result.is_ok();
        let success_count = if success { 1.0 } else { 0.0 };
        metrics.success_rate = (metrics.success_rate * (metrics.total_executions - 1) as f64
            + success_count)
            / metrics.total_executions as f64;

        metrics.last_updated = Utc::now();
    }

    /// Get plugin performance metrics
    pub async fn get_plugin_metrics(&self, plugin_id: &str) -> Option<PerformanceMetrics> {
        self.performance_metrics
            .read()
            .await
            .get(plugin_id)
            .cloned()
    }

    /// List all registered plugins
    pub async fn list_plugins(&self) -> Vec<WasmPlugin> {
        self.plugins.read().await.values().cloned().collect()
    }

    /// Hot reload plugin
    pub async fn hot_reload_plugin(&self, plugin_id: &str, new_wasm_bytes: Vec<u8>) -> Result<()> {
        if !self.config.enable_hot_reload {
            return Err(anyhow!("Hot reload is disabled"));
        }

        let mut plugins = self.plugins.write().await;
        if let Some(plugin) = plugins.get_mut(plugin_id) {
            plugin.wasm_bytes = new_wasm_bytes;
            plugin.updated_at = Utc::now();

            #[cfg(feature = "wasm")]
            {
                let module = Module::new(&self.wasm_engine, &plugin.wasm_bytes)?;
                let mut store = Store::new(&self.wasm_engine, WasmState::default());
                let instance = Instance::new(&mut store, &module, &[])?;

                let context = WasmExecutionContext {
                    engine: self.wasm_engine.clone(),
                    store,
                    instance,
                    plugin_id: plugin_id.to_string(),
                    execution_count: 0,
                    total_execution_time_us: 0,
                    last_execution: Utc::now(),
                    resource_usage: ResourceMetrics::default(),
                };

                self.execution_contexts
                    .write()
                    .await
                    .insert(plugin_id.to_string(), Arc::new(RwLock::new(context)));
            }

            info!("Hot reloaded plugin: {}", plugin_id);
            Ok(())
        } else {
            Err(anyhow!("Plugin not found: {}", plugin_id))
        }
    }

    /// Unregister plugin
    pub async fn unregister_plugin(&self, plugin_id: &str) -> Result<()> {
        self.plugins.write().await.remove(plugin_id);
        self.execution_contexts.write().await.remove(plugin_id);
        self.performance_metrics.write().await.remove(plugin_id);
        info!("Unregistered plugin: {}", plugin_id);
        Ok(())
    }

    /// Load plugin (alias for register_plugin for API compatibility)
    pub async fn load_plugin(&self, plugin: WasmPlugin) -> Result<()> {
        self.register_plugin(plugin).await
    }

    /// Process a single event using the processor
    pub async fn process(&mut self, event: StreamEvent) -> Result<WasmProcessingResult> {
        let plugin_id = {
            let plugins = self.plugins.read().await;
            plugins.keys().next().cloned()
        };

        if let Some(pid) = plugin_id {
            let result = self.execute_plugin(&pid, vec![event], None).await?;
            Ok(WasmProcessingResult {
                output: if result.output_events.is_empty() {
                    None
                } else {
                    Some(result.output_events[0].clone())
                },
                latency_ms: result.execution_time_us as f64 / 1000.0,
            })
        } else {
            Ok(WasmProcessingResult {
                output: None,
                latency_ms: 0.0,
            })
        }
    }

    /// Process event at a specific edge location
    pub async fn process_at_location(
        &self,
        event: StreamEvent,
        location: &EdgeLocation,
    ) -> Result<WasmProcessingResult> {
        let plugin_id = {
            let plugins = self.plugins.read().await;
            plugins.keys().next().cloned()
        };

        if let Some(pid) = plugin_id {
            let result = self
                .execute_plugin(&pid, vec![event], Some(location.id.clone()))
                .await?;
            Ok(WasmProcessingResult {
                output: if result.output_events.is_empty() {
                    None
                } else {
                    Some(result.output_events[0].clone())
                },
                latency_ms: result.execution_time_us as f64 / 1000.0,
            })
        } else {
            Ok(WasmProcessingResult {
                output: None,
                latency_ms: 0.0,
            })
        }
    }

    /// Hot-swap plugin (alias for hot_reload_plugin for API compatibility)
    pub async fn hot_swap_plugin(&self, old_plugin_id: &str, new_plugin: WasmPlugin) -> Result<()> {
        self.unregister_plugin(old_plugin_id).await?;
        self.register_plugin(new_plugin).await?;
        info!("Hot-swapped plugin {} with new version", old_plugin_id);
        Ok(())
    }

    /// Get processor statistics
    pub async fn get_stats(&self) -> WasmProcessorStats {
        let plugins = self.plugins.read().await;
        let metrics = self.performance_metrics.read().await;

        let total_processed = metrics.values().map(|m| m.total_executions).sum();

        let average_latency_ms = if metrics.is_empty() {
            0.0
        } else {
            metrics
                .values()
                .map(|m| m.average_execution_time_us / 1000.0)
                .sum::<f64>()
                / metrics.len() as f64
        };

        WasmProcessorStats {
            total_processed,
            average_latency_ms,
            active_plugins: plugins.len(),
        }
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wasm_edge_processor_creation() {
        let config = WasmEdgeConfig::default();
        let processor = WasmEdgeProcessor::new(config).unwrap();

        let plugins = processor.list_plugins().await;
        assert_eq!(plugins.len(), 0);
    }

    #[tokio::test]
    async fn test_edge_location_scoring() {
        let config = WasmEdgeConfig::default();
        let processor = WasmEdgeProcessor::new(config).unwrap();

        let edge = EdgeLocation {
            id: "test-edge".to_string(),
            region: "us-west".to_string(),
            latency_ms: 50.0,
            capacity_factor: 0.8,
            available_resources: ResourceMetrics::default(),
            specializations: vec![ProcessingSpecialization::RdfProcessing],
        };

        let score = processor
            .calculate_edge_score("test-plugin", &[], &edge)
            .await;
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_security_manager_creation() {
        let security_manager = SecurityManager::new();
        assert!(security_manager.execution_policies.try_read().is_ok());
    }
}
