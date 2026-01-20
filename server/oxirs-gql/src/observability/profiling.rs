//! Continuous Profiling Integration
//!
//! Provides continuous profiling capabilities with pprof and flamegraph support
//! for identifying performance bottlenecks in GraphQL operations.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// Profiling configuration
#[derive(Debug, Clone)]
pub struct ProfilingConfig {
    /// Enable CPU profiling
    pub enable_cpu_profiling: bool,
    /// Enable memory profiling
    pub enable_memory_profiling: bool,
    /// Sampling frequency (Hz)
    pub sampling_frequency: u32,
    /// Maximum profile duration
    pub max_profile_duration: Duration,
    /// Profile retention period
    pub retention_period: Duration,
    /// Enable flamegraph generation
    pub enable_flamegraph: bool,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enable_cpu_profiling: true,
            enable_memory_profiling: true,
            sampling_frequency: 100, // 100 Hz
            max_profile_duration: Duration::from_secs(60),
            retention_period: Duration::from_secs(3600), // 1 hour
            enable_flamegraph: true,
        }
    }
}

/// Stack frame information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StackFrame {
    /// Function name
    pub function: String,
    /// File name
    pub file: Option<String>,
    /// Line number
    pub line: Option<u32>,
    /// Module path
    pub module: Option<String>,
}

impl StackFrame {
    /// Create a new stack frame
    pub fn new(function: String) -> Self {
        Self {
            function,
            file: None,
            line: None,
            module: None,
        }
    }

    /// Set file name
    pub fn with_file(mut self, file: String) -> Self {
        self.file = Some(file);
        self
    }

    /// Set line number
    pub fn with_line(mut self, line: u32) -> Self {
        self.line = Some(line);
        self
    }

    /// Set module path
    pub fn with_module(mut self, module: String) -> Self {
        self.module = Some(module);
        self
    }

    /// Get display name
    pub fn display_name(&self) -> String {
        if let (Some(file), Some(line)) = (&self.file, self.line) {
            format!("{} ({}:{})", self.function, file, line)
        } else {
            self.function.clone()
        }
    }
}

/// Stack trace (call stack)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StackTrace {
    /// Stack frames (bottom to top)
    pub frames: Vec<StackFrame>,
}

impl StackTrace {
    /// Create a new empty stack trace
    pub fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Add a frame to the stack
    pub fn push_frame(mut self, frame: StackFrame) -> Self {
        self.frames.push(frame);
        self
    }

    /// Get the depth of the stack
    pub fn depth(&self) -> usize {
        self.frames.len()
    }
}

impl Default for StackTrace {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for StackTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stack_str = self
            .frames
            .iter()
            .map(|frame| frame.display_name())
            .collect::<Vec<_>>()
            .join(";");
        write!(f, "{}", stack_str)
    }
}

/// Profile sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSample {
    /// Stack trace
    pub stack_trace: StackTrace,
    /// Sample count (number of times this stack was observed)
    pub count: u64,
    /// CPU time in nanoseconds
    pub cpu_time_ns: u64,
    /// Memory allocated in bytes
    pub memory_bytes: u64,
    /// Timestamp of first observation
    pub first_seen: u64,
    /// Timestamp of last observation
    pub last_seen: u64,
}

impl ProfileSample {
    /// Create a new profile sample
    pub fn new(stack_trace: StackTrace) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            stack_trace,
            count: 1,
            cpu_time_ns: 0,
            memory_bytes: 0,
            first_seen: now,
            last_seen: now,
        }
    }

    /// Merge another sample into this one
    pub fn merge(&mut self, other: &ProfileSample) {
        self.count += other.count;
        self.cpu_time_ns += other.cpu_time_ns;
        self.memory_bytes += other.memory_bytes;
        self.last_seen = self.last_seen.max(other.last_seen);
    }
}

/// Profile data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileData {
    /// Profile ID
    pub id: String,
    /// Profile type
    pub profile_type: ProfileType,
    /// Start time
    pub start_time: u64,
    /// End time
    pub end_time: u64,
    /// Samples
    pub samples: Vec<ProfileSample>,
    /// Total sample count
    pub total_samples: u64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl ProfileData {
    /// Create a new profile
    pub fn new(id: String, profile_type: ProfileType) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id,
            profile_type,
            start_time: now,
            end_time: now,
            samples: Vec::new(),
            total_samples: 0,
            metadata: HashMap::new(),
        }
    }

    /// Add a sample
    pub fn add_sample(&mut self, sample: ProfileSample) {
        self.total_samples += sample.count;
        self.samples.push(sample);
    }

    /// Finish profiling
    pub fn finish(&mut self) {
        self.end_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

    /// Get duration in seconds
    pub fn duration_secs(&self) -> u64 {
        self.end_time - self.start_time
    }

    /// Export as pprof format (simplified)
    pub fn to_pprof(&self) -> String {
        let mut output = String::new();

        // Header
        output.push_str(&format!("--- profile: {}\n", self.profile_type.as_str()));
        output.push_str(&format!("duration: {} seconds\n", self.duration_secs()));
        output.push_str(&format!("samples: {}\n", self.total_samples));
        output.push_str("---\n");

        // Samples
        for sample in &self.samples {
            output.push_str(&format!(
                "{} {} # {}\n",
                sample.count, sample.cpu_time_ns, sample.stack_trace
            ));
        }

        output
    }

    /// Export as flamegraph format
    pub fn to_flamegraph(&self) -> String {
        let mut output = String::new();

        for sample in &self.samples {
            output.push_str(&format!("{} {}\n", sample.stack_trace, sample.count));
        }

        output
    }
}

/// Profile type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfileType {
    /// CPU profiling
    Cpu,
    /// Memory profiling (allocations)
    Memory,
    /// Wall clock time profiling
    Wall,
}

impl ProfileType {
    /// Get the profile type as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            ProfileType::Cpu => "cpu",
            ProfileType::Memory => "memory",
            ProfileType::Wall => "wall",
        }
    }
}

/// Profiler for continuous profiling
pub struct Profiler {
    /// Configuration
    config: ProfilingConfig,
    /// Active profiles
    active_profiles: Arc<RwLock<HashMap<String, ProfileData>>>,
    /// Completed profiles
    completed_profiles: Arc<RwLock<VecDeque<ProfileData>>>,
    /// Start time
    start_time: Instant,
}

impl Profiler {
    /// Create a new profiler
    pub fn new(config: ProfilingConfig) -> Self {
        Self {
            config,
            active_profiles: Arc::new(RwLock::new(HashMap::new())),
            completed_profiles: Arc::new(RwLock::new(VecDeque::new())),
            start_time: Instant::now(),
        }
    }

    /// Start a new profiling session
    pub async fn start_profile(&self, id: String, profile_type: ProfileType) -> Result<()> {
        let mut profiles = self.active_profiles.write().await;

        if profiles.contains_key(&id) {
            anyhow::bail!("Profile with ID {} already exists", id);
        }

        let profile = ProfileData::new(id.clone(), profile_type);
        profiles.insert(id, profile);

        Ok(())
    }

    /// Record a sample
    pub async fn record_sample(&self, profile_id: &str, sample: ProfileSample) -> Result<()> {
        let mut profiles = self.active_profiles.write().await;

        let profile = profiles.get_mut(profile_id).context("Profile not found")?;

        profile.add_sample(sample);

        Ok(())
    }

    /// Stop a profiling session
    pub async fn stop_profile(&self, id: &str) -> Result<ProfileData> {
        let mut active = self.active_profiles.write().await;

        let mut profile = active.remove(id).context("Profile not found")?;

        profile.finish();

        // Add to completed profiles
        let mut completed = self.completed_profiles.write().await;
        completed.push_back(profile.clone());

        // Limit retention
        self.cleanup_old_profiles(&mut completed).await;

        Ok(profile)
    }

    /// Get active profile
    pub async fn get_active_profile(&self, id: &str) -> Option<ProfileData> {
        let profiles = self.active_profiles.read().await;
        profiles.get(id).cloned()
    }

    /// List all active profiles
    pub async fn list_active_profiles(&self) -> Vec<String> {
        let profiles = self.active_profiles.read().await;
        profiles.keys().cloned().collect()
    }

    /// Get completed profile
    pub async fn get_completed_profile(&self, id: &str) -> Option<ProfileData> {
        let profiles = self.completed_profiles.read().await;
        profiles.iter().find(|p| p.id == id).cloned()
    }

    /// List all completed profiles
    pub async fn list_completed_profiles(&self) -> Vec<ProfileData> {
        let profiles = self.completed_profiles.read().await;
        profiles.iter().cloned().collect()
    }

    /// Cleanup old profiles
    async fn cleanup_old_profiles(&self, completed: &mut VecDeque<ProfileData>) {
        let retention_secs = self.config.retention_period.as_secs();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        while let Some(front) = completed.front() {
            if now - front.end_time > retention_secs {
                completed.pop_front();
            } else {
                break;
            }
        }

        // Also limit by count (keep max 100 profiles)
        while completed.len() > 100 {
            completed.pop_front();
        }
    }

    /// Export profile to pprof format
    pub async fn export_pprof(&self, profile_id: &str) -> Result<String> {
        // Try active profiles first
        if let Some(profile) = self.get_active_profile(profile_id).await {
            return Ok(profile.to_pprof());
        }

        // Try completed profiles
        if let Some(profile) = self.get_completed_profile(profile_id).await {
            return Ok(profile.to_pprof());
        }

        anyhow::bail!("Profile not found")
    }

    /// Export profile to flamegraph format
    pub async fn export_flamegraph(&self, profile_id: &str) -> Result<String> {
        // Try active profiles first
        if let Some(profile) = self.get_active_profile(profile_id).await {
            return Ok(profile.to_flamegraph());
        }

        // Try completed profiles
        if let Some(profile) = self.get_completed_profile(profile_id).await {
            return Ok(profile.to_flamegraph());
        }

        anyhow::bail!("Profile not found")
    }

    /// Get profiler statistics
    pub async fn get_statistics(&self) -> ProfilerStatistics {
        let active = self.active_profiles.read().await;
        let completed = self.completed_profiles.read().await;

        let mut total_samples = 0;
        for profile in completed.iter() {
            total_samples += profile.total_samples;
        }

        ProfilerStatistics {
            active_profiles: active.len(),
            completed_profiles: completed.len(),
            total_samples,
            uptime_secs: self.start_time.elapsed().as_secs(),
        }
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new(ProfilingConfig::default())
    }
}

/// Profiler statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerStatistics {
    /// Number of active profiles
    pub active_profiles: usize,
    /// Number of completed profiles
    pub completed_profiles: usize,
    /// Total samples across all profiles
    pub total_samples: u64,
    /// Profiler uptime in seconds
    pub uptime_secs: u64,
}

/// Helper for building stack traces
pub struct StackTraceBuilder {
    frames: Vec<StackFrame>,
}

impl StackTraceBuilder {
    /// Create a new stack trace builder
    pub fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Add a frame
    pub fn push(mut self, frame: StackFrame) -> Self {
        self.frames.push(frame);
        self
    }

    /// Add a simple function frame
    pub fn push_function(mut self, function: String) -> Self {
        self.frames.push(StackFrame::new(function));
        self
    }

    /// Build the stack trace
    pub fn build(self) -> StackTrace {
        StackTrace {
            frames: self.frames,
        }
    }
}

impl Default for StackTraceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_frame_creation() {
        let frame = StackFrame::new("test_function".to_string())
            .with_file("test.rs".to_string())
            .with_line(42)
            .with_module("test_module".to_string());

        assert_eq!(frame.function, "test_function");
        assert_eq!(frame.file, Some("test.rs".to_string()));
        assert_eq!(frame.line, Some(42));
        assert_eq!(frame.module, Some("test_module".to_string()));
        assert_eq!(frame.display_name(), "test_function (test.rs:42)");
    }

    #[test]
    fn test_stack_frame_display_name() {
        let frame1 = StackFrame::new("func1".to_string());
        assert_eq!(frame1.display_name(), "func1");

        let frame2 = StackFrame::new("func2".to_string())
            .with_file("test.rs".to_string())
            .with_line(10);
        assert_eq!(frame2.display_name(), "func2 (test.rs:10)");
    }

    #[test]
    fn test_stack_trace() {
        let trace = StackTrace::new()
            .push_frame(StackFrame::new("main".to_string()))
            .push_frame(StackFrame::new("process".to_string()))
            .push_frame(StackFrame::new("execute".to_string()));

        assert_eq!(trace.depth(), 3);
        assert_eq!(trace.to_string(), "main;process;execute");
        assert_eq!(format!("{}", trace), "main;process;execute");
    }

    #[test]
    fn test_stack_trace_builder() {
        let trace = StackTraceBuilder::new()
            .push_function("main".to_string())
            .push_function("process".to_string())
            .push(
                StackFrame::new("execute".to_string())
                    .with_file("app.rs".to_string())
                    .with_line(100),
            )
            .build();

        assert_eq!(trace.depth(), 3);
        let trace_str = trace.to_string();
        assert!(trace_str.contains("execute (app.rs:100)"));
    }

    #[test]
    fn test_profile_sample() {
        let trace = StackTrace::new().push_frame(StackFrame::new("test".to_string()));

        let mut sample1 = ProfileSample::new(trace.clone());
        sample1.cpu_time_ns = 1000;
        sample1.memory_bytes = 2000;

        let mut sample2 = ProfileSample::new(trace.clone());
        sample2.cpu_time_ns = 500;
        sample2.memory_bytes = 1000;

        sample1.merge(&sample2);

        assert_eq!(sample1.count, 2);
        assert_eq!(sample1.cpu_time_ns, 1500);
        assert_eq!(sample1.memory_bytes, 3000);
    }

    #[test]
    fn test_profile_data() {
        let mut profile = ProfileData::new("test-profile".to_string(), ProfileType::Cpu);

        let trace = StackTrace::new().push_frame(StackFrame::new("test_fn".to_string()));
        let sample = ProfileSample::new(trace);

        profile.add_sample(sample);
        assert_eq!(profile.total_samples, 1);

        profile.finish();
        // Duration should be valid (>= 0 is implicit for u64)
        let _duration = profile.duration_secs();
    }

    #[test]
    fn test_profile_type() {
        assert_eq!(ProfileType::Cpu.as_str(), "cpu");
        assert_eq!(ProfileType::Memory.as_str(), "memory");
        assert_eq!(ProfileType::Wall.as_str(), "wall");
    }

    #[test]
    fn test_pprof_export() {
        let mut profile = ProfileData::new("test".to_string(), ProfileType::Cpu);

        let trace = StackTrace::new()
            .push_frame(StackFrame::new("main".to_string()))
            .push_frame(StackFrame::new("process".to_string()));

        let mut sample = ProfileSample::new(trace);
        sample.count = 100;
        sample.cpu_time_ns = 5000000;

        profile.add_sample(sample);

        let pprof = profile.to_pprof();
        assert!(pprof.contains("profile: cpu"));
        assert!(pprof.contains("samples: 100"));
        assert!(pprof.contains("main;process"));
    }

    #[test]
    fn test_flamegraph_export() {
        let mut profile = ProfileData::new("test".to_string(), ProfileType::Cpu);

        let trace = StackTrace::new()
            .push_frame(StackFrame::new("main".to_string()))
            .push_frame(StackFrame::new("execute".to_string()));

        let mut sample = ProfileSample::new(trace);
        sample.count = 50;

        profile.add_sample(sample);

        let flamegraph = profile.to_flamegraph();
        assert!(flamegraph.contains("main;execute 50"));
    }

    #[tokio::test]
    async fn test_profiler_start_stop() {
        let profiler = Profiler::default();

        profiler
            .start_profile("test-1".to_string(), ProfileType::Cpu)
            .await
            .unwrap();

        assert!(profiler.get_active_profile("test-1").await.is_some());

        let profile = profiler.stop_profile("test-1").await.unwrap();
        assert_eq!(profile.id, "test-1");

        assert!(profiler.get_active_profile("test-1").await.is_none());
        assert!(profiler.get_completed_profile("test-1").await.is_some());
    }

    #[tokio::test]
    async fn test_profiler_duplicate_id() {
        let profiler = Profiler::default();

        profiler
            .start_profile("test".to_string(), ProfileType::Cpu)
            .await
            .unwrap();

        let result = profiler
            .start_profile("test".to_string(), ProfileType::Cpu)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_profiler_record_sample() {
        let profiler = Profiler::default();

        profiler
            .start_profile("test".to_string(), ProfileType::Cpu)
            .await
            .unwrap();

        let trace = StackTrace::new().push_frame(StackFrame::new("test_fn".to_string()));
        let sample = ProfileSample::new(trace);

        profiler.record_sample("test", sample).await.unwrap();

        let profile = profiler.get_active_profile("test").await.unwrap();
        assert_eq!(profile.total_samples, 1);
    }

    #[tokio::test]
    async fn test_profiler_list_profiles() {
        let profiler = Profiler::default();

        profiler
            .start_profile("test-1".to_string(), ProfileType::Cpu)
            .await
            .unwrap();
        profiler
            .start_profile("test-2".to_string(), ProfileType::Memory)
            .await
            .unwrap();

        let active = profiler.list_active_profiles().await;
        assert_eq!(active.len(), 2);
        assert!(active.contains(&"test-1".to_string()));
        assert!(active.contains(&"test-2".to_string()));
    }

    #[tokio::test]
    async fn test_profiler_export_pprof() {
        let profiler = Profiler::default();

        profiler
            .start_profile("test".to_string(), ProfileType::Cpu)
            .await
            .unwrap();

        let trace = StackTrace::new().push_frame(StackFrame::new("test_fn".to_string()));
        let sample = ProfileSample::new(trace);
        profiler.record_sample("test", sample).await.unwrap();

        let pprof = profiler.export_pprof("test").await.unwrap();
        assert!(pprof.contains("profile: cpu"));
    }

    #[tokio::test]
    async fn test_profiler_export_flamegraph() {
        let profiler = Profiler::default();

        profiler
            .start_profile("test".to_string(), ProfileType::Cpu)
            .await
            .unwrap();

        let trace = StackTrace::new().push_frame(StackFrame::new("test_fn".to_string()));
        let sample = ProfileSample::new(trace);
        profiler.record_sample("test", sample).await.unwrap();

        let flamegraph = profiler.export_flamegraph("test").await.unwrap();
        assert!(flamegraph.contains("test_fn"));
    }

    #[tokio::test]
    async fn test_profiler_statistics() {
        let profiler = Profiler::default();

        profiler
            .start_profile("test-1".to_string(), ProfileType::Cpu)
            .await
            .unwrap();
        profiler
            .start_profile("test-2".to_string(), ProfileType::Memory)
            .await
            .unwrap();

        let stats = profiler.get_statistics().await;
        assert_eq!(stats.active_profiles, 2);
        assert_eq!(stats.completed_profiles, 0);

        profiler.stop_profile("test-1").await.unwrap();

        let stats = profiler.get_statistics().await;
        assert_eq!(stats.active_profiles, 1);
        assert_eq!(stats.completed_profiles, 1);
    }

    #[tokio::test]
    async fn test_profiler_not_found() {
        let profiler = Profiler::default();

        let result = profiler.stop_profile("nonexistent").await;
        assert!(result.is_err());

        let result = profiler.export_pprof("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_profiling_config() {
        let config = ProfilingConfig::default();

        assert!(config.enable_cpu_profiling);
        assert!(config.enable_memory_profiling);
        assert_eq!(config.sampling_frequency, 100);
        assert!(config.enable_flamegraph);
    }

    #[test]
    fn test_profile_data_metadata() {
        let mut profile = ProfileData::new("test".to_string(), ProfileType::Cpu);

        profile
            .metadata
            .insert("version".to_string(), "1.0".to_string());
        profile
            .metadata
            .insert("env".to_string(), "production".to_string());

        assert_eq!(profile.metadata.get("version"), Some(&"1.0".to_string()));
        assert_eq!(profile.metadata.get("env"), Some(&"production".to_string()));
    }

    #[tokio::test]
    async fn test_completed_profiles_list() {
        let profiler = Profiler::default();

        profiler
            .start_profile("test-1".to_string(), ProfileType::Cpu)
            .await
            .unwrap();
        profiler
            .start_profile("test-2".to_string(), ProfileType::Memory)
            .await
            .unwrap();

        profiler.stop_profile("test-1").await.unwrap();
        profiler.stop_profile("test-2").await.unwrap();

        let completed = profiler.list_completed_profiles().await;
        assert_eq!(completed.len(), 2);
    }
}
