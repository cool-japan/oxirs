//! Execution Configuration
//!
//! This module contains configuration structures for query execution.

use crate::builtin::register_builtin_functions;
use crate::extensions::ExtensionRegistry;
use lazy_static::lazy_static;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Query execution context
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Maximum execution time
    pub timeout: Option<Duration>,
    /// Memory limit in bytes
    pub memory_limit: Option<usize>,
    /// Enable parallel execution
    pub parallel: bool,
    /// Parallel execution configuration
    pub parallel_config: ParallelConfig,
    /// Streaming configuration
    pub streaming: StreamingConfig,
    /// Statistics collection
    pub collect_stats: bool,
    /// Query complexity threshold for parallel execution
    pub parallel_threshold: usize,
    /// Enable query result caching
    pub enable_caching: bool,
    /// Extension registry for functions
    pub extension_registry: Arc<ExtensionRegistry>,
}

/// Parallel execution configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Maximum number of threads to use
    pub max_threads: usize,
    /// Enable work-stealing
    pub work_stealing: bool,
    /// Chunk size for parallel processing
    pub chunk_size: usize,
    /// Threshold for enabling parallel execution
    pub parallel_threshold: usize,
    /// Thread pool configuration
    pub thread_pool_config: ThreadPoolConfig,
    /// Enable NUMA-aware execution
    pub numa_aware: bool,
    /// Minimum work size for parallel execution
    pub min_parallel_work: usize,
    /// Enable adaptive parallelization
    pub adaptive: bool,
}

/// Thread pool configuration
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Thread stack size
    pub stack_size: Option<usize>,
    /// Thread priority
    pub thread_priority: Option<i32>,
    /// Enable thread affinity
    pub thread_affinity: bool,
}

/// Streaming configuration
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Buffer size for streaming results
    pub buffer_size: usize,
    /// Batch size for result processing
    pub batch_size: usize,
    /// Enable streaming mode
    pub enabled: bool,
}

// Global function registry
lazy_static! {
    static ref FUNCTION_REGISTRY: Arc<ExtensionRegistry> = {
        let registry = Arc::<ExtensionRegistry>::new(ExtensionRegistry::new());
        register_builtin_functions(&registry).expect("Failed to register built-in functions");
        registry
    };
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            timeout: Some(Duration::from_secs(300)), // 5 minutes default timeout
            memory_limit: Some(1024 * 1024 * 1024),  // 1GB default limit
            parallel: true,
            parallel_config: ParallelConfig::default(),
            streaming: StreamingConfig::default(),
            collect_stats: false,
            parallel_threshold: 1000,
            enable_caching: true,
            extension_registry: FUNCTION_REGISTRY.clone(),
        }
    }
}

impl ExecutionContext {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            max_threads: num_cpus,
            work_stealing: true,
            chunk_size: 1000,
            parallel_threshold: 10000,
            thread_pool_config: ThreadPoolConfig::default(),
            numa_aware: false,
            min_parallel_work: 100,
            adaptive: true,
        }
    }
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            stack_size: Some(8 * 1024 * 1024), // 8MB stack size
            thread_priority: None,
            thread_affinity: false,
        }
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10000,
            batch_size: 1000,
            enabled: false,
        }
    }
}
