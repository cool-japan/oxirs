//! CPU Usage Profiling Per Resolver
//!
//! Tracks CPU usage for individual GraphQL resolvers to identify
//! performance bottlenecks and optimize resolver execution.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::{CpuRefreshKind, Pid, ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System};
use tokio::sync::RwLock;

/// CPU profiler for resolvers
pub struct CpuProfiler {
    /// Resolver profiles
    profiles: Arc<RwLock<HashMap<String, ResolverCpuProfile>>>,
    /// Completed profiles
    completed: Arc<RwLock<Vec<CompletedCpuProfile>>>,
    /// System info
    system: Arc<RwLock<System>>,
    /// Configuration
    config: CpuProfilingConfig,
    /// Process ID
    pid: Pid,
}

/// CPU profiling configuration
#[derive(Debug, Clone)]
pub struct CpuProfilingConfig {
    /// Enable detailed profiling
    pub enable_detailed_profiling: bool,
    /// Sampling interval
    pub sampling_interval: Duration,
    /// Maximum profiles to store
    pub max_profiles: usize,
    /// Retention period
    pub retention_period: Duration,
    /// CPU threshold for warnings (percentage)
    pub warning_threshold_percent: f32,
}

impl Default for CpuProfilingConfig {
    fn default() -> Self {
        Self {
            enable_detailed_profiling: true,
            sampling_interval: Duration::from_millis(100),
            max_profiles: 1000,
            retention_period: Duration::from_secs(3600), // 1 hour
            warning_threshold_percent: 80.0,
        }
    }
}

/// Resolver CPU profile (active)
#[derive(Debug, Clone)]
pub struct ResolverCpuProfile {
    /// Resolver name
    pub resolver_name: String,
    /// Field path
    pub field_path: String,
    /// Start time
    pub start_time: Instant,
    /// Start timestamp
    pub start_timestamp: u64,
    /// CPU samples
    pub samples: Vec<CpuSample>,
    /// Initial CPU usage
    pub initial_cpu_percent: f32,
    /// Peak CPU usage
    pub peak_cpu_percent: f32,
}

impl ResolverCpuProfile {
    /// Create a new resolver CPU profile
    pub fn new(resolver_name: String, field_path: String, initial_cpu: f32) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            resolver_name,
            field_path,
            start_time: Instant::now(),
            start_timestamp: now,
            samples: Vec::new(),
            initial_cpu_percent: initial_cpu,
            peak_cpu_percent: initial_cpu,
        }
    }

    /// Add a CPU sample
    pub fn add_sample(&mut self, sample: CpuSample) {
        self.peak_cpu_percent = self.peak_cpu_percent.max(sample.cpu_percent);
        self.samples.push(sample);
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get average CPU usage
    pub fn average_cpu_percent(&self) -> f32 {
        if self.samples.is_empty() {
            self.initial_cpu_percent
        } else {
            let sum: f32 = self.samples.iter().map(|s| s.cpu_percent).sum();
            sum / self.samples.len() as f32
        }
    }
}

/// CPU sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuSample {
    /// Timestamp offset from start (milliseconds)
    pub offset_ms: u64,
    /// CPU usage percentage
    pub cpu_percent: f32,
    /// Number of threads
    pub thread_count: usize,
}

impl CpuSample {
    /// Create a new CPU sample
    pub fn new(offset_ms: u64, cpu_percent: f32, thread_count: usize) -> Self {
        Self {
            offset_ms,
            cpu_percent,
            thread_count,
        }
    }
}

/// Completed CPU profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedCpuProfile {
    /// Resolver name
    pub resolver_name: String,
    /// Field path
    pub field_path: String,
    /// Start time
    pub start_time: u64,
    /// End time
    pub end_time: u64,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Initial CPU percentage
    pub initial_cpu_percent: f32,
    /// Peak CPU percentage
    pub peak_cpu_percent: f32,
    /// Average CPU percentage
    pub average_cpu_percent: f32,
    /// Number of samples
    pub sample_count: usize,
    /// CPU samples
    pub samples: Vec<CpuSample>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl CompletedCpuProfile {
    /// Create from active profile
    pub fn from_profile(profile: ResolverCpuProfile) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let duration_ms = profile.elapsed().as_millis() as u64;
        let average_cpu_percent = profile.average_cpu_percent();
        let sample_count = profile.samples.len();

        Self {
            resolver_name: profile.resolver_name,
            field_path: profile.field_path,
            start_time: profile.start_timestamp,
            end_time: now,
            duration_ms,
            initial_cpu_percent: profile.initial_cpu_percent,
            peak_cpu_percent: profile.peak_cpu_percent,
            average_cpu_percent,
            sample_count,
            samples: profile.samples,
            metadata: HashMap::new(),
        }
    }

    /// Check if CPU usage was high
    pub fn is_high_cpu(&self, threshold: f32) -> bool {
        self.average_cpu_percent >= threshold
    }
}

impl CpuProfiler {
    /// Create a new CPU profiler
    pub fn new(config: CpuProfilingConfig) -> Self {
        let pid = Pid::from_u32(std::process::id());

        let refresh_kind = RefreshKind::nothing()
            .with_cpu(CpuRefreshKind::everything())
            .with_processes(ProcessRefreshKind::nothing().with_cpu());

        Self {
            profiles: Arc::new(RwLock::new(HashMap::new())),
            completed: Arc::new(RwLock::new(Vec::new())),
            system: Arc::new(RwLock::new(System::new_with_specifics(refresh_kind))),
            config,
            pid,
        }
    }

    /// Start profiling a resolver
    pub async fn start_profiling(
        &self,
        profile_id: String,
        resolver_name: String,
        field_path: String,
    ) -> Result<()> {
        // Refresh system info
        let mut system = self.system.write().await;
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[self.pid]),
            false,
            ProcessRefreshKind::nothing().with_cpu(),
        );

        // Get current CPU usage
        let cpu_percent = system
            .process(self.pid)
            .map(|p| p.cpu_usage())
            .unwrap_or(0.0);

        drop(system);

        let profile = ResolverCpuProfile::new(resolver_name, field_path, cpu_percent);

        let mut profiles = self.profiles.write().await;
        profiles.insert(profile_id, profile);

        Ok(())
    }

    /// Record a CPU sample
    pub async fn record_sample(&self, profile_id: &str) -> Result<()> {
        let mut system = self.system.write().await;
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[self.pid]),
            false,
            ProcessRefreshKind::nothing().with_cpu(),
        );

        let cpu_percent = system
            .process(self.pid)
            .map(|p| p.cpu_usage())
            .unwrap_or(0.0);

        let thread_count = system
            .process(self.pid)
            .map(|p| p.tasks().map(|t| t.len()).unwrap_or(1))
            .unwrap_or(1);

        drop(system);

        let mut profiles = self.profiles.write().await;

        if let Some(profile) = profiles.get_mut(profile_id) {
            let offset_ms = profile.elapsed().as_millis() as u64;
            let sample = CpuSample::new(offset_ms, cpu_percent, thread_count);
            profile.add_sample(sample);

            // Check warning threshold
            if cpu_percent > self.config.warning_threshold_percent {
                tracing::warn!(
                    profile_id = %profile_id,
                    resolver = %profile.resolver_name,
                    cpu_percent = cpu_percent,
                    threshold = self.config.warning_threshold_percent,
                    "Resolver exceeded CPU warning threshold"
                );
            }
        }

        Ok(())
    }

    /// Stop profiling and get completed profile
    pub async fn stop_profiling(&self, profile_id: &str) -> Result<CompletedCpuProfile> {
        let mut profiles = self.profiles.write().await;

        let profile = profiles
            .remove(profile_id)
            .ok_or_else(|| anyhow::anyhow!("Profile not found: {}", profile_id))?;

        let completed_profile = CompletedCpuProfile::from_profile(profile);

        // Store in completed
        let mut completed = self.completed.write().await;
        completed.push(completed_profile.clone());

        // Cleanup old profiles
        self.cleanup_old_profiles(&mut completed).await;

        Ok(completed_profile)
    }

    /// Get current profile
    pub async fn get_profile(&self, profile_id: &str) -> Option<ResolverCpuProfile> {
        let profiles = self.profiles.read().await;
        profiles.get(profile_id).cloned()
    }

    /// Get completed profiles
    pub async fn get_completed_profiles(&self) -> Vec<CompletedCpuProfile> {
        let completed = self.completed.read().await;
        completed.clone()
    }

    /// Get profiles by resolver
    pub async fn get_profiles_by_resolver(&self, resolver_name: &str) -> Vec<CompletedCpuProfile> {
        let completed = self.completed.read().await;
        completed
            .iter()
            .filter(|p| p.resolver_name == resolver_name)
            .cloned()
            .collect()
    }

    /// Get top CPU consumers
    pub async fn get_top_consumers(&self, limit: usize) -> Vec<CompletedCpuProfile> {
        let completed = self.completed.read().await;

        let mut sorted = completed.clone();
        sorted.sort_by(|a, b| {
            b.average_cpu_percent
                .partial_cmp(&a.average_cpu_percent)
                .unwrap()
        });
        sorted.truncate(limit);
        sorted
    }

    /// Get CPU statistics
    pub async fn get_statistics(&self) -> CpuStatistics {
        let profiles = self.profiles.read().await;
        let completed = self.completed.read().await;

        let active_profiles = profiles.len();
        let completed_profiles = completed.len();

        let mut total_cpu_percent = 0.0;
        let mut high_cpu_count = 0;

        for profile in completed.iter() {
            total_cpu_percent += profile.average_cpu_percent;
            if profile.is_high_cpu(self.config.warning_threshold_percent) {
                high_cpu_count += 1;
            }
        }

        let avg_cpu_percent = if completed_profiles > 0 {
            total_cpu_percent / completed_profiles as f32
        } else {
            0.0
        };

        CpuStatistics {
            active_profiles,
            completed_profiles,
            avg_cpu_percent,
            high_cpu_count,
        }
    }

    /// Cleanup old profiles
    async fn cleanup_old_profiles(&self, completed: &mut Vec<CompletedCpuProfile>) {
        let retention_secs = self.config.retention_period.as_secs();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        completed.retain(|p| now - p.end_time < retention_secs);

        // Also limit by count
        if completed.len() > self.config.max_profiles {
            let excess = completed.len() - self.config.max_profiles;
            completed.drain(0..excess);
        }
    }
}

impl Default for CpuProfiler {
    fn default() -> Self {
        Self::new(CpuProfilingConfig::default())
    }
}

/// CPU statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuStatistics {
    /// Active profiles
    pub active_profiles: usize,
    /// Completed profiles
    pub completed_profiles: usize,
    /// Average CPU percentage
    pub avg_cpu_percent: f32,
    /// High CPU count
    pub high_cpu_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolver_cpu_profile_creation() {
        let profile =
            ResolverCpuProfile::new("UserResolver".to_string(), "Query.user".to_string(), 10.0);

        assert_eq!(profile.resolver_name, "UserResolver");
        assert_eq!(profile.field_path, "Query.user");
        assert_eq!(profile.initial_cpu_percent, 10.0);
        assert_eq!(profile.peak_cpu_percent, 10.0);
        assert_eq!(profile.samples.len(), 0);
    }

    #[test]
    fn test_resolver_cpu_profile_add_sample() {
        let mut profile =
            ResolverCpuProfile::new("UserResolver".to_string(), "Query.user".to_string(), 10.0);

        profile.add_sample(CpuSample::new(100, 20.0, 4));
        profile.add_sample(CpuSample::new(200, 30.0, 4));

        assert_eq!(profile.samples.len(), 2);
        assert_eq!(profile.peak_cpu_percent, 30.0);
    }

    #[test]
    fn test_resolver_cpu_profile_average() {
        let mut profile =
            ResolverCpuProfile::new("UserResolver".to_string(), "Query.user".to_string(), 10.0);

        profile.add_sample(CpuSample::new(100, 20.0, 4));
        profile.add_sample(CpuSample::new(200, 30.0, 4));
        profile.add_sample(CpuSample::new(300, 40.0, 4));

        assert_eq!(profile.average_cpu_percent(), 30.0);
    }

    #[test]
    fn test_cpu_sample_creation() {
        let sample = CpuSample::new(100, 25.5, 4);

        assert_eq!(sample.offset_ms, 100);
        assert_eq!(sample.cpu_percent, 25.5);
        assert_eq!(sample.thread_count, 4);
    }

    #[test]
    fn test_completed_cpu_profile_from_profile() {
        let mut profile =
            ResolverCpuProfile::new("UserResolver".to_string(), "Query.user".to_string(), 10.0);

        profile.add_sample(CpuSample::new(100, 20.0, 4));
        profile.add_sample(CpuSample::new(200, 30.0, 4));

        let completed = CompletedCpuProfile::from_profile(profile);

        assert_eq!(completed.resolver_name, "UserResolver");
        assert_eq!(completed.field_path, "Query.user");
        assert_eq!(completed.initial_cpu_percent, 10.0);
        assert_eq!(completed.peak_cpu_percent, 30.0);
        assert_eq!(completed.average_cpu_percent, 25.0);
        assert_eq!(completed.sample_count, 2);
    }

    #[test]
    fn test_completed_cpu_profile_is_high_cpu() {
        let mut profile =
            ResolverCpuProfile::new("UserResolver".to_string(), "Query.user".to_string(), 10.0);

        profile.add_sample(CpuSample::new(100, 80.0, 4));
        profile.add_sample(CpuSample::new(200, 90.0, 4));

        let completed = CompletedCpuProfile::from_profile(profile);

        assert!(completed.is_high_cpu(80.0));
        assert!(!completed.is_high_cpu(90.0));
    }

    #[tokio::test]
    async fn test_cpu_profiler_start_stop() {
        let profiler = CpuProfiler::default();

        profiler
            .start_profiling(
                "profile-1".to_string(),
                "UserResolver".to_string(),
                "Query.user".to_string(),
            )
            .await
            .unwrap();

        let profile = profiler.get_profile("profile-1").await;
        assert!(profile.is_some());

        let completed = profiler.stop_profiling("profile-1").await.unwrap();
        assert_eq!(completed.resolver_name, "UserResolver");

        let profile = profiler.get_profile("profile-1").await;
        assert!(profile.is_none());
    }

    #[tokio::test]
    async fn test_cpu_profiler_record_sample() {
        let profiler = CpuProfiler::default();

        profiler
            .start_profiling(
                "profile-1".to_string(),
                "UserResolver".to_string(),
                "Query.user".to_string(),
            )
            .await
            .unwrap();

        profiler.record_sample("profile-1").await.unwrap();
        profiler.record_sample("profile-1").await.unwrap();

        let profile = profiler.get_profile("profile-1").await.unwrap();
        assert!(profile.samples.len() >= 2);
    }

    #[tokio::test]
    async fn test_cpu_profiler_completed_profiles() {
        let profiler = CpuProfiler::default();

        profiler
            .start_profiling(
                "profile-1".to_string(),
                "UserResolver".to_string(),
                "Query.user".to_string(),
            )
            .await
            .unwrap();

        profiler.stop_profiling("profile-1").await.unwrap();

        let completed = profiler.get_completed_profiles().await;
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].resolver_name, "UserResolver");
    }

    #[tokio::test]
    async fn test_cpu_profiler_profiles_by_resolver() {
        let profiler = CpuProfiler::default();

        profiler
            .start_profiling(
                "profile-1".to_string(),
                "UserResolver".to_string(),
                "Query.user".to_string(),
            )
            .await
            .unwrap();

        profiler.stop_profiling("profile-1").await.unwrap();

        profiler
            .start_profiling(
                "profile-2".to_string(),
                "PostResolver".to_string(),
                "Query.posts".to_string(),
            )
            .await
            .unwrap();

        profiler.stop_profiling("profile-2").await.unwrap();

        let profiles = profiler.get_profiles_by_resolver("UserResolver").await;
        assert_eq!(profiles.len(), 1);
        assert_eq!(profiles[0].resolver_name, "UserResolver");
    }

    #[tokio::test]
    async fn test_cpu_profiler_top_consumers() {
        let profiler = CpuProfiler::default();

        // Profile 1 - low CPU
        profiler
            .start_profiling(
                "profile-1".to_string(),
                "Resolver1".to_string(),
                "Query.low".to_string(),
            )
            .await
            .unwrap();
        profiler.stop_profiling("profile-1").await.unwrap();

        // Profile 2 - high CPU
        profiler
            .start_profiling(
                "profile-2".to_string(),
                "Resolver2".to_string(),
                "Query.high".to_string(),
            )
            .await
            .unwrap();
        profiler.stop_profiling("profile-2").await.unwrap();

        let top = profiler.get_top_consumers(1).await;
        assert_eq!(top.len(), 1);
    }

    #[tokio::test]
    async fn test_cpu_profiler_statistics() {
        let profiler = CpuProfiler::default();

        profiler
            .start_profiling(
                "profile-1".to_string(),
                "UserResolver".to_string(),
                "Query.user".to_string(),
            )
            .await
            .unwrap();

        profiler
            .start_profiling(
                "profile-2".to_string(),
                "PostResolver".to_string(),
                "Query.posts".to_string(),
            )
            .await
            .unwrap();

        profiler.stop_profiling("profile-1").await.unwrap();

        let stats = profiler.get_statistics().await;
        assert_eq!(stats.active_profiles, 1); // profile-2 still active
        assert_eq!(stats.completed_profiles, 1);
    }

    #[tokio::test]
    async fn test_cpu_profiler_not_found() {
        let profiler = CpuProfiler::default();

        let result = profiler.stop_profiling("nonexistent").await;
        assert!(result.is_err());

        let profile = profiler.get_profile("nonexistent").await;
        assert!(profile.is_none());
    }

    #[test]
    fn test_cpu_profiling_config() {
        let config = CpuProfilingConfig::default();

        assert!(config.enable_detailed_profiling);
        assert_eq!(config.max_profiles, 1000);
        assert_eq!(config.warning_threshold_percent, 80.0);
    }

    #[test]
    fn test_resolver_cpu_profile_no_samples_average() {
        let profile =
            ResolverCpuProfile::new("UserResolver".to_string(), "Query.user".to_string(), 15.0);

        // With no samples, average should be initial CPU
        assert_eq!(profile.average_cpu_percent(), 15.0);
    }

    #[test]
    fn test_completed_cpu_profile_metadata() {
        let profile =
            ResolverCpuProfile::new("UserResolver".to_string(), "Query.user".to_string(), 10.0);

        let mut completed = CompletedCpuProfile::from_profile(profile);

        completed
            .metadata
            .insert("user_id".to_string(), "123".to_string());
        completed
            .metadata
            .insert("query_type".to_string(), "query".to_string());

        assert_eq!(completed.metadata.get("user_id"), Some(&"123".to_string()));
        assert_eq!(
            completed.metadata.get("query_type"),
            Some(&"query".to_string())
        );
    }
}
