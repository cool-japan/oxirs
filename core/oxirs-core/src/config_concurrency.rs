//! Concurrency configuration for OxiRS Core.

use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencyConfig {
    pub thread_pool: ThreadPoolConfig,
    pub locks: LockConfig,
    pub async_runtime: AsyncRuntimeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolConfig {
    pub worker_threads: usize,
    pub stack_size: usize,
    pub priority: ThreadPriority,
    pub work_stealing: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreadPriority {
    Low,
    Normal,
    High,
    Realtime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockConfig {
    pub default_type: LockType,
    pub timeout: Duration,
    pub enable_debugging: bool,
    pub deadlock_detection: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LockType {
    Mutex,
    RwLock,
    SpinLock,
    AtomicLock,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncRuntimeConfig {
    pub runtime_type: AsyncRuntimeType,
    pub enable_io: bool,
    pub enable_time: bool,
    pub worker_config: AsyncWorkerConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AsyncRuntimeType {
    CurrentThread,
    MultiThread,
    MultiThreadAlt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncWorkerConfig {
    pub core_threads: usize,
    pub max_threads: usize,
    pub keep_alive: Duration,
    pub thread_name_prefix: String,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            thread_pool: ThreadPoolConfig::default(),
            locks: LockConfig::default(),
            async_runtime: AsyncRuntimeConfig::default(),
        }
    }
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            worker_threads: 0,
            stack_size: 2 * 1024 * 1024,
            priority: ThreadPriority::Normal,
            work_stealing: true,
        }
    }
}

impl Default for LockConfig {
    fn default() -> Self {
        Self {
            default_type: LockType::RwLock,
            timeout: Duration::from_secs(30),
            enable_debugging: false,
            deadlock_detection: true,
        }
    }
}

impl Default for AsyncRuntimeConfig {
    fn default() -> Self {
        Self {
            runtime_type: AsyncRuntimeType::MultiThread,
            enable_io: true,
            enable_time: true,
            worker_config: AsyncWorkerConfig::default(),
        }
    }
}

impl Default for AsyncWorkerConfig {
    fn default() -> Self {
        Self {
            core_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            max_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                * 4,
            keep_alive: Duration::from_secs(60),
            thread_name_prefix: "oxirs-async".to_string(),
        }
    }
}
