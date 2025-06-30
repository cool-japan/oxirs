//! Enhanced SPARQL Function Library with Dynamic Registration and Advanced Security
//!
//! This module provides a comprehensive function library system for SPARQL constraints
//! with dynamic function registration, advanced security sandboxing, and plugin architecture.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

use oxirs_core::model::Term;

use crate::{Result, ShaclError};

/// Enhanced function library manager with dynamic registration and security
#[derive(Debug)]
pub struct SparqlFunctionLibrary {
    /// Registered custom functions
    functions: Arc<RwLock<HashMap<String, Arc<dyn DynamicFunction>>>>,
    /// Function libraries
    libraries: Arc<RwLock<HashMap<String, FunctionLibrary>>>,
    /// Security policies per function
    security_policies: Arc<RwLock<HashMap<String, FunctionSecurityPolicy>>>,
    /// Global security configuration
    global_security: SecurityConfig,
    /// Execution monitor for resource tracking
    execution_monitor: Arc<RwLock<ExecutionMonitor>>,
    /// Dependency resolver
    dependency_resolver: DependencyResolver,
    /// Version manager
    version_manager: VersionManager,
}

impl SparqlFunctionLibrary {
    /// Create a new function library with default configuration
    pub fn new() -> Self {
        Self {
            functions: Arc::new(RwLock::new(HashMap::new())),
            libraries: Arc::new(RwLock::new(HashMap::new())),
            security_policies: Arc::new(RwLock::new(HashMap::new())),
            global_security: SecurityConfig::default(),
            execution_monitor: Arc::new(RwLock::new(ExecutionMonitor::new())),
            dependency_resolver: DependencyResolver::new(),
            version_manager: VersionManager::new(),
        }
    }

    /// Register a custom function with security policy
    pub fn register_function(
        &mut self,
        function: Arc<dyn DynamicFunction>,
        security_policy: Option<FunctionSecurityPolicy>,
    ) -> Result<()> {
        let function_name = function.metadata().name.clone();

        // Validate function metadata
        self.validate_function_metadata(function.metadata())?;

        // Set security policy
        let policy = security_policy.unwrap_or_else(|| FunctionSecurityPolicy::default());
        self.validate_security_policy(&policy)?;

        // Register function
        {
            let mut functions = self.functions.write().unwrap();
            functions.insert(function_name.clone(), function);
        }

        // Register security policy
        {
            let mut policies = self.security_policies.write().unwrap();
            policies.insert(function_name.clone(), policy);
        }

        println!("Registered custom SPARQL function: {}", function_name);
        Ok(())
    }

    /// Unregister a function
    pub fn unregister_function(&mut self, function_name: &str) -> Result<()> {
        {
            let mut functions = self.functions.write().unwrap();
            functions.remove(function_name);
        }

        {
            let mut policies = self.security_policies.write().unwrap();
            policies.remove(function_name);
        }

        println!("Unregistered custom SPARQL function: {}", function_name);
        Ok(())
    }

    /// Execute a function with security sandboxing
    pub fn execute_function(
        &self,
        function_name: &str,
        args: &[Term],
        context: &ExecutionContext,
    ) -> Result<FunctionExecutionResult> {
        let start_time = Instant::now();

        // Get function and security policy
        let (function, policy) = {
            let functions = self.functions.read().unwrap();
            let policies = self.security_policies.read().unwrap();

            let function = functions
                .get(function_name)
                .ok_or_else(|| {
                    ShaclError::ValidationEngine(format!("Function not found: {}", function_name))
                })?
                .clone();

            let policy = policies
                .get(function_name)
                .cloned()
                .unwrap_or_else(|| FunctionSecurityPolicy::default());

            (function, policy)
        };

        // Create sandboxed execution environment
        let sandbox = FunctionSandbox::new(&policy, &self.global_security);

        // Execute function in sandbox
        let result = sandbox.execute(function.as_ref(), args, context)?;

        // Record execution metrics
        let execution_time = start_time.elapsed();
        {
            let mut monitor = self.execution_monitor.write().unwrap();
            monitor.record_execution(function_name, execution_time, result.memory_used);
        }

        Ok(result)
    }

    /// Load a function library from a plugin
    pub fn load_library(&mut self, library: FunctionLibrary) -> Result<()> {
        // Validate library metadata
        self.validate_library_metadata(&library.metadata)?;

        // Resolve dependencies
        self.dependency_resolver.resolve_dependencies(&library)?;

        // Check version compatibility
        self.version_manager.check_compatibility(&library)?;

        // Load all functions from the library
        for function in &library.functions {
            let security_policy = library
                .security_policies
                .get(&function.metadata().name)
                .cloned();
            self.register_function(function.clone(), security_policy)?;
        }

        // Register library
        {
            let mut libraries = self.libraries.write().unwrap();
            libraries.insert(library.metadata.name.clone(), library);
        }

        Ok(())
    }

    /// Unload a function library
    pub fn unload_library(&mut self, library_name: &str) -> Result<()> {
        // Get library
        let library = {
            let libraries = self.libraries.read().unwrap();
            libraries
                .get(library_name)
                .ok_or_else(|| {
                    ShaclError::ValidationEngine(format!("Library not found: {}", library_name))
                })?
                .clone()
        };

        // Unregister all functions from the library
        for function in &library.functions {
            self.unregister_function(&function.metadata().name)?;
        }

        // Remove library
        {
            let mut libraries = self.libraries.write().unwrap();
            libraries.remove(library_name);
        }

        Ok(())
    }

    /// Get list of available functions
    pub fn list_functions(&self) -> Vec<FunctionMetadata> {
        let functions = self.functions.read().unwrap();
        functions.values().map(|f| f.metadata().clone()).collect()
    }

    /// Get execution statistics
    pub fn get_execution_stats(&self) -> ExecutionStats {
        let monitor = self.execution_monitor.read().unwrap();
        monitor.get_stats()
    }

    /// Update global security configuration
    pub fn update_security_config(&mut self, config: SecurityConfig) {
        self.global_security = config;
    }

    /// Validate function metadata
    fn validate_function_metadata(&self, metadata: &FunctionMetadata) -> Result<()> {
        if metadata.name.is_empty() {
            return Err(ShaclError::ValidationEngine(
                "Function name cannot be empty".to_string(),
            ));
        }

        if !metadata
            .name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_')
        {
            return Err(ShaclError::ValidationEngine(
                "Function name contains invalid characters".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate security policy
    fn validate_security_policy(&self, policy: &FunctionSecurityPolicy) -> Result<()> {
        if policy.max_execution_time > self.global_security.max_function_execution_time {
            return Err(ShaclError::ValidationEngine(
                "Function execution time exceeds global limit".to_string(),
            ));
        }

        if policy.max_memory_per_call > self.global_security.max_function_memory {
            return Err(ShaclError::ValidationEngine(
                "Function memory limit exceeds global limit".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate library metadata
    fn validate_library_metadata(&self, metadata: &LibraryMetadata) -> Result<()> {
        if metadata.name.is_empty() {
            return Err(ShaclError::ValidationEngine(
                "Library name cannot be empty".to_string(),
            ));
        }

        // Check for version conflicts
        let libraries = self.libraries.read().unwrap();
        if let Some(existing) = libraries.get(&metadata.name) {
            if existing.metadata.version == metadata.version {
                return Err(ShaclError::ValidationEngine(format!(
                    "Library {} version {} already loaded",
                    metadata.name, metadata.version
                )));
            }
        }

        Ok(())
    }
}

/// Trait for dynamic SPARQL functions with enhanced metadata and security
pub trait DynamicFunction: Send + Sync {
    /// Execute the function with given arguments
    fn execute(&self, args: &[Term], context: &ExecutionContext) -> Result<Term>;

    /// Get function metadata
    fn metadata(&self) -> &FunctionMetadata;

    /// Get function security requirements
    fn security_policy(&self) -> FunctionSecurityPolicy {
        FunctionSecurityPolicy::default()
    }

    /// Validate function arguments
    fn validate_args(&self, args: &[Term]) -> Result<()> {
        let metadata = self.metadata();
        if args.len() < metadata.min_args || args.len() > metadata.max_args {
            return Err(ShaclError::ValidationEngine(format!(
                "Function {} expects {}-{} arguments, got {}",
                metadata.name,
                metadata.min_args,
                metadata.max_args,
                args.len()
            )));
        }
        Ok(())
    }
}

/// Enhanced function metadata with comprehensive information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionMetadata {
    /// Function name
    pub name: String,
    /// Function description
    pub description: String,
    /// Function version
    pub version: String,
    /// Function author
    pub author: String,
    /// Minimum number of arguments
    pub min_args: usize,
    /// Maximum number of arguments
    pub max_args: usize,
    /// Expected argument types
    pub arg_types: Vec<String>,
    /// Return type
    pub return_type: String,
    /// Function category
    pub category: FunctionCategory,
    /// Whether function is deterministic
    pub deterministic: bool,
    /// Whether function has side effects
    pub has_side_effects: bool,
    /// Required permissions
    pub required_permissions: Vec<Permission>,
    /// Function documentation URL
    pub documentation_url: Option<String>,
    /// Example usage
    pub examples: Vec<String>,
}

/// Function execution context with enhanced information
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Query execution context
    pub query_context: HashMap<String, Term>,
    /// Execution timestamp
    pub timestamp: SystemTime,
    /// User permissions
    pub permissions: HashSet<Permission>,
    /// Execution ID for tracking
    pub execution_id: String,
    /// Maximum execution time allowed
    pub max_execution_time: Duration,
    /// Maximum memory allowed
    pub max_memory: usize,
}

/// Function execution result with detailed information
#[derive(Debug, Clone)]
pub struct FunctionExecutionResult {
    /// Function result value
    pub result: Term,
    /// Execution time
    pub execution_time: Duration,
    /// Memory used during execution
    pub memory_used: usize,
    /// Whether execution was successful
    pub success: bool,
    /// Warning messages
    pub warnings: Vec<String>,
    /// Debug information
    pub debug_info: Option<String>,
}

/// Advanced security policy for individual functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSecurityPolicy {
    /// Maximum execution time per call
    pub max_execution_time: Duration,
    /// Maximum memory usage per call
    pub max_memory_per_call: usize,
    /// Maximum number of calls per minute
    pub max_calls_per_minute: u32,
    /// Allowed operations
    pub allowed_operations: Vec<Operation>,
    /// Sandbox level
    pub sandbox_level: SandboxLevel,
    /// Required permissions
    pub required_permissions: Vec<Permission>,
    /// Whether function can access network
    pub network_access: bool,
    /// Whether function can access filesystem
    pub filesystem_access: bool,
    /// Custom security rules
    pub custom_rules: HashMap<String, serde_json::Value>,
}

impl Default for FunctionSecurityPolicy {
    fn default() -> Self {
        Self {
            max_execution_time: Duration::from_secs(5),
            max_memory_per_call: 16 * 1024 * 1024, // 16MB
            max_calls_per_minute: 100,
            allowed_operations: vec![Operation::Read],
            sandbox_level: SandboxLevel::Strict,
            required_permissions: vec![],
            network_access: false,
            filesystem_access: false,
            custom_rules: HashMap::new(),
        }
    }
}

/// Function library containing multiple related functions
#[derive(Debug, Clone)]
pub struct FunctionLibrary {
    /// Library metadata
    pub metadata: LibraryMetadata,
    /// Functions in this library
    pub functions: Vec<Arc<dyn DynamicFunction>>,
    /// Security policies for functions
    pub security_policies: HashMap<String, FunctionSecurityPolicy>,
    /// Dependencies on other libraries
    pub dependencies: Vec<LibraryDependency>,
}

/// Library metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryMetadata {
    /// Library name
    pub name: String,
    /// Library description
    pub description: String,
    /// Library version
    pub version: String,
    /// Library author
    pub author: String,
    /// License information
    pub license: String,
    /// Library website
    pub website: Option<String>,
    /// Minimum SHACL engine version required
    pub min_engine_version: String,
}

/// Library dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryDependency {
    /// Dependency library name
    pub name: String,
    /// Required version range
    pub version_range: String,
    /// Whether dependency is optional
    pub optional: bool,
}

/// Function sandbox for secure execution
#[derive(Debug)]
pub struct FunctionSandbox {
    /// Security policy
    policy: FunctionSecurityPolicy,
    /// Global security config
    global_config: SecurityConfig,
    /// Resource monitor
    resource_monitor: ResourceMonitor,
}

impl FunctionSandbox {
    fn new(policy: &FunctionSecurityPolicy, global_config: &SecurityConfig) -> Self {
        Self {
            policy: policy.clone(),
            global_config: global_config.clone(),
            resource_monitor: ResourceMonitor::new(),
        }
    }

    fn execute(
        &self,
        function: &dyn DynamicFunction,
        args: &[Term],
        context: &ExecutionContext,
    ) -> Result<FunctionExecutionResult> {
        let start_time = Instant::now();
        let start_memory = self.resource_monitor.current_memory_usage();

        // Validate permissions
        self.validate_permissions(function, context)?;

        // Start monitoring
        let _monitor_guard = self.resource_monitor.start_monitoring(&self.policy)?;

        // Execute function
        let result = function.execute(args, context)?;

        // Calculate metrics
        let execution_time = start_time.elapsed();
        let memory_used = self
            .resource_monitor
            .current_memory_usage()
            .saturating_sub(start_memory);

        // Validate execution time
        if execution_time > self.policy.max_execution_time {
            return Err(ShaclError::ValidationEngine(format!(
                "Function execution time {} exceeded limit {}",
                execution_time.as_millis(),
                self.policy.max_execution_time.as_millis()
            )));
        }

        // Validate memory usage
        if memory_used > self.policy.max_memory_per_call {
            return Err(ShaclError::ValidationEngine(format!(
                "Function memory usage {} exceeded limit {}",
                memory_used, self.policy.max_memory_per_call
            )));
        }

        Ok(FunctionExecutionResult {
            result,
            execution_time,
            memory_used,
            success: true,
            warnings: vec![],
            debug_info: None,
        })
    }

    fn validate_permissions(
        &self,
        function: &dyn DynamicFunction,
        context: &ExecutionContext,
    ) -> Result<()> {
        let required_permissions = &function.security_policy().required_permissions;

        for permission in required_permissions {
            if !context.permissions.contains(permission) {
                return Err(ShaclError::ValidationEngine(format!(
                    "Missing required permission: {:?}",
                    permission
                )));
            }
        }

        Ok(())
    }
}

/// Resource monitor for tracking function execution
#[derive(Debug)]
pub struct ResourceMonitor {
    start_time: Option<Instant>,
    start_memory: usize,
}

impl ResourceMonitor {
    fn new() -> Self {
        Self {
            start_time: None,
            start_memory: 0,
        }
    }

    fn start_monitoring(&self, _policy: &FunctionSecurityPolicy) -> Result<MonitorGuard> {
        Ok(MonitorGuard::new())
    }

    fn current_memory_usage(&self) -> usize {
        // Simplified implementation - in practice would use actual memory monitoring
        std::mem::size_of::<Self>()
    }
}

/// Monitor guard for automatic resource cleanup
#[derive(Debug)]
pub struct MonitorGuard {
    _start_time: Instant,
}

impl MonitorGuard {
    fn new() -> Self {
        Self {
            _start_time: Instant::now(),
        }
    }
}

/// Execution monitor for collecting statistics
#[derive(Debug)]
pub struct ExecutionMonitor {
    executions: HashMap<String, Vec<ExecutionRecord>>,
    total_executions: usize,
    total_execution_time: Duration,
}

impl ExecutionMonitor {
    fn new() -> Self {
        Self {
            executions: HashMap::new(),
            total_executions: 0,
            total_execution_time: Duration::ZERO,
        }
    }

    fn record_execution(
        &mut self,
        function_name: &str,
        execution_time: Duration,
        memory_used: usize,
    ) {
        let record = ExecutionRecord {
            timestamp: SystemTime::now(),
            execution_time,
            memory_used,
        };

        self.executions
            .entry(function_name.to_string())
            .or_insert_with(Vec::new)
            .push(record);

        self.total_executions += 1;
        self.total_execution_time += execution_time;
    }

    fn get_stats(&self) -> ExecutionStats {
        ExecutionStats {
            total_executions: self.total_executions,
            total_execution_time: self.total_execution_time,
            average_execution_time: if self.total_executions > 0 {
                self.total_execution_time / self.total_executions as u32
            } else {
                Duration::ZERO
            },
            function_stats: self.executions.clone(),
        }
    }
}

/// Individual execution record
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    pub timestamp: SystemTime,
    pub execution_time: Duration,
    pub memory_used: usize,
}

/// Comprehensive execution statistics
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    pub total_executions: usize,
    pub total_execution_time: Duration,
    pub average_execution_time: Duration,
    pub function_stats: HashMap<String, Vec<ExecutionRecord>>,
}

/// Dependency resolver for function libraries
#[derive(Debug)]
pub struct DependencyResolver {
    resolved_dependencies: HashMap<String, String>,
}

impl DependencyResolver {
    fn new() -> Self {
        Self {
            resolved_dependencies: HashMap::new(),
        }
    }

    fn resolve_dependencies(&mut self, _library: &FunctionLibrary) -> Result<()> {
        // Simplified implementation - would implement actual dependency resolution
        Ok(())
    }
}

/// Version manager for library compatibility
#[derive(Debug)]
pub struct VersionManager {
    version_cache: HashMap<String, String>,
}

impl VersionManager {
    fn new() -> Self {
        Self {
            version_cache: HashMap::new(),
        }
    }

    fn check_compatibility(&self, _library: &FunctionLibrary) -> Result<()> {
        // Simplified implementation - would implement actual version checking
        Ok(())
    }
}

/// Global security configuration
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    pub max_function_execution_time: Duration,
    pub max_function_memory: usize,
    pub enable_network_access: bool,
    pub enable_filesystem_access: bool,
    pub allowed_function_categories: HashSet<FunctionCategory>,
    pub default_sandbox_level: SandboxLevel,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            max_function_execution_time: Duration::from_secs(30),
            max_function_memory: 128 * 1024 * 1024, // 128MB
            enable_network_access: false,
            enable_filesystem_access: false,
            allowed_function_categories: [FunctionCategory::String, FunctionCategory::Math]
                .iter()
                .cloned()
                .collect(),
            default_sandbox_level: SandboxLevel::Strict,
        }
    }
}

/// Function categories for organization
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FunctionCategory {
    String,
    Math,
    DateTime,
    Logic,
    Conversion,
    Hash,
    Custom,
    System,
    Network,
    Database,
}

/// Security sandbox levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SandboxLevel {
    /// No sandboxing
    None,
    /// Basic resource monitoring
    Basic,
    /// Strict resource limits and operation restrictions
    Strict,
    /// Maximum security with minimal allowed operations
    Maximum,
}

/// Allowed operations within sandbox
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operation {
    Read,
    Write,
    Network,
    FileSystem,
    System,
    Database,
}

/// Security permissions for function execution
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    ReadData,
    WriteData,
    NetworkAccess,
    FileSystemAccess,
    SystemAccess,
    AdminAccess,
}

impl Default for SparqlFunctionLibrary {
    fn default() -> Self {
        Self::new()
    }
}
