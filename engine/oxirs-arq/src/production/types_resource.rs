use anyhow::{anyhow, Result};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::RwLock;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub current_usage: usize,
    pub peak_usage: usize,
    pub memory_limit: usize,
    pub active_queries: usize,
    pub pressure_percentage: f64,
}

#[derive(Debug, Clone)]
pub struct GlobalStatistics {
    pub uptime: Duration,
    pub total_queries: u64,
    pub total_timeouts: u64,
    pub total_errors: u64,
}

pub struct QueryResourceQuota {
    max_result_size: AtomicUsize,
    max_query_time: RwLock<Duration>,
    max_pattern_complexity: AtomicUsize,
    enforced: AtomicBool,
}

impl QueryResourceQuota {
    pub fn new(
        max_result_size: usize,
        max_query_time: Duration,
        max_pattern_complexity: usize,
    ) -> Self {
        Self {
            max_result_size: AtomicUsize::new(max_result_size),
            max_query_time: RwLock::new(max_query_time),
            max_pattern_complexity: AtomicUsize::new(max_pattern_complexity),
            enforced: AtomicBool::new(true),
        }
    }
    pub fn check_result_size(&self, size: usize) -> Result<()> {
        if !self.enforced.load(Ordering::Relaxed) {
            return Ok(());
        }
        let max = self.max_result_size.load(Ordering::Relaxed);
        if size > max {
            return Err(anyhow!("Result size {} exceeds quota of {}", size, max));
        }
        Ok(())
    }
    pub fn check_query_time(&self, elapsed: Duration) -> Result<()> {
        if !self.enforced.load(Ordering::Relaxed) {
            return Ok(());
        }
        let max = *self.max_query_time.read().expect("lock poisoned");
        if elapsed > max {
            return Err(anyhow!(
                "Query time {:?} exceeds quota of {:?}",
                elapsed,
                max
            ));
        }
        Ok(())
    }
    pub fn check_pattern_complexity(&self, complexity: usize) -> Result<()> {
        if !self.enforced.load(Ordering::Relaxed) {
            return Ok(());
        }
        let max = self.max_pattern_complexity.load(Ordering::Relaxed);
        if complexity > max {
            return Err(anyhow!(
                "Pattern complexity {} exceeds quota of {}",
                complexity,
                max
            ));
        }
        Ok(())
    }
    pub fn set_result_size_limit(&self, limit: usize) {
        self.max_result_size.store(limit, Ordering::Relaxed);
    }
    pub fn set_time_limit(&self, limit: Duration) {
        *self.max_query_time.write().expect("lock poisoned") = limit;
    }
    pub fn set_complexity_limit(&self, limit: usize) {
        self.max_pattern_complexity.store(limit, Ordering::Relaxed);
    }
    pub fn set_enforced(&self, enforced: bool) {
        self.enforced.store(enforced, Ordering::Relaxed);
    }
    pub fn is_enforced(&self) -> bool {
        self.enforced.load(Ordering::Relaxed)
    }
}
