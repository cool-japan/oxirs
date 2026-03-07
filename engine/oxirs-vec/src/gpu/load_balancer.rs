//! GPU load balancing for distributing index-building work across multiple devices.
//!
//! This module provides:
//! - `GpuLoadBalancer`: runtime tracking of per-device workloads and selection of the
//!   least-loaded device for a new task.
//! - `WorkloadDistributor`: static splitting of a large index job into per-device
//!   contiguous chunks.
//!
//! # Pure Rust Policy
//!
//! No CUDA runtime calls are made here.  All load-balancing logic is Pure Rust and
//! operates on abstract device descriptors (`SimpleGpuDevice`).

use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

// ============================================================
// Device descriptor
// ============================================================

/// Lightweight descriptor of a GPU device used for load balancing decisions.
///
/// This is intentionally separate from `crate::gpu::GpuDevice` (which carries
/// CUDA-specific fields) so that the load balancer remains 100% Pure Rust.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SimpleGpuDevice {
    /// Numeric device identifier (matches CUDA device ordinal when CUDA is enabled)
    pub id: u32,
    /// Human-readable name, e.g. "NVIDIA A100-80GB"
    pub name: String,
    /// Total GPU memory in megabytes
    pub memory_mb: u64,
    /// Number of CUDA streaming multiprocessors / compute units
    pub compute_units: u32,
}

impl SimpleGpuDevice {
    /// Create a new device descriptor.
    pub fn new(id: u32, name: impl Into<String>, memory_mb: u64, compute_units: u32) -> Self {
        Self {
            id,
            name: name.into(),
            memory_mb,
            compute_units,
        }
    }
}

// ============================================================
// Per-device state (internal)
// ============================================================

#[derive(Debug)]
struct DeviceState {
    device: SimpleGpuDevice,
    /// Currently allocated workload in megabytes
    current_workload_mb: u64,
}

impl DeviceState {
    fn new(device: SimpleGpuDevice) -> Self {
        Self {
            device,
            current_workload_mb: 0,
        }
    }

    /// Utilisation as a fraction [0.0, 1.0] of total device memory.
    fn utilization(&self) -> f64 {
        if self.device.memory_mb == 0 {
            return 0.0;
        }
        (self.current_workload_mb as f64 / self.device.memory_mb as f64).min(1.0)
    }
}

// ============================================================
// GpuLoadBalancer
// ============================================================

/// Distributes GPU work across multiple devices using a least-loaded strategy.
///
/// All mutating operations are thread-safe via an internal `Mutex`.
///
/// # Example
/// ```
/// use oxirs_vec::gpu::{GpuLoadBalancer, SimpleGpuDevice};
///
/// let balancer = GpuLoadBalancer::new();
/// balancer.register_device(SimpleGpuDevice::new(0, "GPU-0", 8192, 128));
/// balancer.register_device(SimpleGpuDevice::new(1, "GPU-1", 16384, 256));
///
/// if let Some(id) = balancer.select_device(512) {
///     balancer.record_workload(id, 512);
///     // ... do GPU work ...
///     balancer.release_workload(id, 512);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct GpuLoadBalancer {
    inner: Arc<Mutex<GpuLoadBalancerInner>>,
}

#[derive(Debug)]
struct GpuLoadBalancerInner {
    /// Ordered list of registered device IDs (insertion order)
    device_order: Vec<u32>,
    /// Per-device state keyed by device ID
    states: HashMap<u32, DeviceState>,
}

impl GpuLoadBalancerInner {
    fn new() -> Self {
        Self {
            device_order: Vec::new(),
            states: HashMap::new(),
        }
    }
}

impl Default for GpuLoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuLoadBalancer {
    /// Create an empty load balancer with no registered devices.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(GpuLoadBalancerInner::new())),
        }
    }

    /// Register a GPU device.  If a device with the same `id` already exists it is
    /// replaced (workload is reset to zero).
    pub fn register_device(&self, device: SimpleGpuDevice) {
        let mut g = self.inner.lock();
        let id = device.id;
        info!("Registering GPU device {} ({})", id, device.name);
        if !g.device_order.contains(&id) {
            g.device_order.push(id);
        }
        g.states.insert(id, DeviceState::new(device));
    }

    /// Remove a device from the balancer.
    pub fn unregister_device(&self, device_id: u32) {
        let mut g = self.inner.lock();
        g.device_order.retain(|&x| x != device_id);
        g.states.remove(&device_id);
        debug!("Unregistered GPU device {}", device_id);
    }

    /// Select the device best suited to handle `workload_mb` megabytes of new work.
    ///
    /// Returns the `id` of the device with the lowest current utilisation that has
    /// enough free memory to accept the workload, or `None` if no suitable device
    /// exists or no devices are registered.
    pub fn select_device(&self, workload_mb: u64) -> Option<u32> {
        let g = self.inner.lock();
        g.device_order
            .iter()
            .filter_map(|&id| g.states.get(&id).map(|s| (id, s)))
            .filter(|(_, s)| {
                s.device.memory_mb.saturating_sub(s.current_workload_mb) >= workload_mb
            })
            .min_by(|(_, a), (_, b)| {
                a.utilization()
                    .partial_cmp(&b.utilization())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| id)
    }

    /// Record `mb` megabytes of additional workload on `device_id`.
    ///
    /// Returns an error if `device_id` is not registered.
    pub fn record_workload(&self, device_id: u32, mb: u64) -> Result<()> {
        let mut g = self.inner.lock();
        let state = g
            .states
            .get_mut(&device_id)
            .ok_or_else(|| anyhow!("Device {} not registered", device_id))?;
        state.current_workload_mb += mb;
        debug!(
            "Device {}: workload {} MB (util {:.1}%)",
            device_id,
            state.current_workload_mb,
            state.utilization() * 100.0
        );
        Ok(())
    }

    /// Release `mb` megabytes of workload from `device_id`.
    ///
    /// Clamps to zero to prevent underflow.  Returns an error if the device is
    /// not registered.
    pub fn release_workload(&self, device_id: u32, mb: u64) -> Result<()> {
        let mut g = self.inner.lock();
        let state = g
            .states
            .get_mut(&device_id)
            .ok_or_else(|| anyhow!("Device {} not registered", device_id))?;
        state.current_workload_mb = state.current_workload_mb.saturating_sub(mb);
        debug!(
            "Device {}: released {} MB, now {} MB",
            device_id, mb, state.current_workload_mb
        );
        Ok(())
    }

    /// Utilisation of `device_id` as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns `None` if the device is not registered.
    pub fn utilization(&self, device_id: u32) -> Option<f64> {
        let g = self.inner.lock();
        g.states.get(&device_id).map(|s| s.utilization())
    }

    /// Sum of memory across all registered devices in MB.
    pub fn total_capacity_mb(&self) -> u64 {
        let g = self.inner.lock();
        g.states.values().map(|s| s.device.memory_mb).sum()
    }

    /// Number of registered devices.
    pub fn device_count(&self) -> usize {
        self.inner.lock().states.len()
    }

    /// Returns a snapshot of device IDs and their current utilisation.
    pub fn utilization_snapshot(&self) -> Vec<(u32, f64)> {
        let g = self.inner.lock();
        g.device_order
            .iter()
            .filter_map(|&id| g.states.get(&id).map(|s| (id, s.utilization())))
            .collect()
    }
}

// ============================================================
// WorkloadChunk
// ============================================================

/// A contiguous slice of a vector dataset assigned to a specific GPU.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadChunk {
    /// ID of the GPU device responsible for this chunk
    pub device_id: u32,
    /// Start index (inclusive) in the source vector array
    pub start_idx: usize,
    /// End index (exclusive) in the source vector array
    pub end_idx: usize,
}

impl WorkloadChunk {
    /// Number of vectors in this chunk.
    pub fn len(&self) -> usize {
        self.end_idx.saturating_sub(self.start_idx)
    }

    /// Returns `true` if the chunk covers no vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ============================================================
// WorkloadDistributor
// ============================================================

/// Splits a large vector index job across multiple GPU devices proportionally to
/// their memory capacity.
///
/// The distributor is stateless: call `distribute` as many times as needed
/// without side effects.
#[derive(Debug, Clone, Default)]
pub struct WorkloadDistributor;

impl WorkloadDistributor {
    /// Create a new distributor.
    pub fn new() -> Self {
        Self
    }

    /// Distribute `total_vectors` vectors across `devices` proportionally to each
    /// device's `memory_mb`.
    ///
    /// Returns one [`WorkloadChunk`] per device (in device order).  Devices with
    /// zero memory are skipped.  Returns an error if `devices` is empty or all
    /// devices have zero memory.
    ///
    /// The last chunk absorbs any rounding remainder so that every vector is
    /// covered exactly once.
    pub fn distribute(
        &self,
        total_vectors: usize,
        devices: &[SimpleGpuDevice],
    ) -> Result<Vec<WorkloadChunk>> {
        let eligible: Vec<&SimpleGpuDevice> = devices.iter().filter(|d| d.memory_mb > 0).collect();

        if eligible.is_empty() {
            return Err(anyhow!(
                "No eligible GPU devices (all have zero memory or list is empty)"
            ));
        }

        let total_mem: u64 = eligible.iter().map(|d| d.memory_mb).sum();

        let mut chunks: Vec<WorkloadChunk> = Vec::with_capacity(eligible.len());
        let mut assigned = 0usize;

        for (i, device) in eligible.iter().enumerate() {
            let start_idx = assigned;
            let end_idx = if i == eligible.len() - 1 {
                // Last device gets remaining vectors (absorbs rounding error)
                total_vectors
            } else {
                let fraction = device.memory_mb as f64 / total_mem as f64;
                let count = (total_vectors as f64 * fraction).round() as usize;
                (assigned + count).min(total_vectors)
            };

            chunks.push(WorkloadChunk {
                device_id: device.id,
                start_idx,
                end_idx,
            });
            assigned = end_idx;

            if assigned >= total_vectors {
                break;
            }
        }

        Ok(chunks)
    }

    /// Distribute evenly (round-robin, ignoring memory ratios).
    ///
    /// Useful when all devices are homogeneous.  Returns an error if `devices` is
    /// empty.
    pub fn distribute_even(
        &self,
        total_vectors: usize,
        devices: &[SimpleGpuDevice],
    ) -> Result<Vec<WorkloadChunk>> {
        if devices.is_empty() {
            return Err(anyhow!("Cannot distribute across zero devices"));
        }

        let n = devices.len();
        let base = total_vectors / n;
        let remainder = total_vectors % n;

        let mut chunks = Vec::with_capacity(n);
        let mut start = 0;

        for (i, device) in devices.iter().enumerate() {
            let extra = if i < remainder { 1 } else { 0 };
            let end = start + base + extra;
            chunks.push(WorkloadChunk {
                device_id: device.id,
                start_idx: start,
                end_idx: end,
            });
            start = end;
        }

        Ok(chunks)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_device(id: u32, mem_mb: u64) -> SimpleGpuDevice {
        SimpleGpuDevice::new(id, format!("GPU-{}", id), mem_mb, 128)
    }

    // ---- SimpleGpuDevice ----

    #[test]
    fn test_simple_gpu_device_fields() {
        let d = SimpleGpuDevice::new(0, "TestGPU", 8192, 128);
        assert_eq!(d.id, 0);
        assert_eq!(d.name, "TestGPU");
        assert_eq!(d.memory_mb, 8192);
        assert_eq!(d.compute_units, 128);
    }

    // ---- GpuLoadBalancer ----

    #[test]
    fn test_register_device_count() {
        let lb = GpuLoadBalancer::new();
        lb.register_device(make_device(0, 8192));
        lb.register_device(make_device(1, 16384));
        assert_eq!(lb.device_count(), 2);
    }

    #[test]
    fn test_total_capacity_mb() {
        let lb = GpuLoadBalancer::new();
        lb.register_device(make_device(0, 4096));
        lb.register_device(make_device(1, 8192));
        assert_eq!(lb.total_capacity_mb(), 12288);
    }

    #[test]
    fn test_select_device_empty_returns_none() {
        let lb = GpuLoadBalancer::new();
        assert!(lb.select_device(100).is_none());
    }

    #[test]
    fn test_select_device_single() {
        let lb = GpuLoadBalancer::new();
        lb.register_device(make_device(0, 8192));
        let sel = lb.select_device(100);
        assert_eq!(sel, Some(0));
    }

    #[test]
    fn test_select_device_insufficient_memory() {
        let lb = GpuLoadBalancer::new();
        lb.register_device(make_device(0, 100)); // only 100 MB
                                                 // Requesting 200 MB should yield None
        assert!(lb.select_device(200).is_none());
    }

    #[test]
    fn test_select_device_prefers_least_loaded() {
        let lb = GpuLoadBalancer::new();
        lb.register_device(make_device(0, 8192));
        lb.register_device(make_device(1, 8192));

        // Load device 0 heavily
        lb.record_workload(0, 7000).unwrap();

        // Device 1 should be selected
        let sel = lb.select_device(500);
        assert_eq!(sel, Some(1), "Should prefer the less-loaded device");
    }

    #[test]
    fn test_record_and_release_workload() {
        let lb = GpuLoadBalancer::new();
        lb.register_device(make_device(0, 8192));

        lb.record_workload(0, 2048).unwrap();
        let u1 = lb.utilization(0).unwrap();
        assert!(
            (u1 - 0.25).abs() < 1e-6,
            "Expected 25% utilisation, got {}",
            u1
        );

        lb.release_workload(0, 2048).unwrap();
        let u2 = lb.utilization(0).unwrap();
        assert!(u2 < 1e-9, "Expected 0% after release, got {}", u2);
    }

    #[test]
    fn test_release_clamps_to_zero() {
        let lb = GpuLoadBalancer::new();
        lb.register_device(make_device(0, 8192));
        lb.record_workload(0, 100).unwrap();
        // Release more than recorded — should not underflow
        lb.release_workload(0, 9999).unwrap();
        assert_eq!(lb.utilization(0).unwrap(), 0.0);
    }

    #[test]
    fn test_record_unknown_device_errors() {
        let lb = GpuLoadBalancer::new();
        assert!(lb.record_workload(99, 100).is_err());
    }

    #[test]
    fn test_release_unknown_device_errors() {
        let lb = GpuLoadBalancer::new();
        assert!(lb.release_workload(99, 100).is_err());
    }

    #[test]
    fn test_utilization_unknown_device_none() {
        let lb = GpuLoadBalancer::new();
        assert!(lb.utilization(42).is_none());
    }

    #[test]
    fn test_utilization_snapshot() {
        let lb = GpuLoadBalancer::new();
        lb.register_device(make_device(0, 8192));
        lb.register_device(make_device(1, 4096));
        lb.record_workload(0, 4096).unwrap();
        let snap = lb.utilization_snapshot();
        assert_eq!(snap.len(), 2);
        let u0 = snap
            .iter()
            .find(|(id, _)| *id == 0)
            .map(|(_, u)| *u)
            .unwrap();
        assert!((u0 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_unregister_device() {
        let lb = GpuLoadBalancer::new();
        lb.register_device(make_device(0, 8192));
        lb.register_device(make_device(1, 8192));
        lb.unregister_device(0);
        assert_eq!(lb.device_count(), 1);
        assert!(lb.utilization(0).is_none());
    }

    #[test]
    fn test_reregister_device_resets_workload() {
        let lb = GpuLoadBalancer::new();
        lb.register_device(make_device(0, 8192));
        lb.record_workload(0, 4096).unwrap();
        // Re-register same device — workload should reset
        lb.register_device(make_device(0, 8192));
        assert_eq!(lb.utilization(0).unwrap(), 0.0);
    }

    // ---- WorkloadChunk ----

    #[test]
    fn test_workload_chunk_len() {
        let chunk = WorkloadChunk {
            device_id: 0,
            start_idx: 10,
            end_idx: 50,
        };
        assert_eq!(chunk.len(), 40);
    }

    #[test]
    fn test_workload_chunk_is_empty() {
        let chunk = WorkloadChunk {
            device_id: 0,
            start_idx: 5,
            end_idx: 5,
        };
        assert!(chunk.is_empty());
    }

    // ---- WorkloadDistributor ----

    #[test]
    fn test_distribute_empty_devices_error() {
        let dist = WorkloadDistributor::new();
        assert!(dist.distribute(1000, &[]).is_err());
    }

    #[test]
    fn test_distribute_single_device() {
        let dist = WorkloadDistributor::new();
        let devices = vec![make_device(0, 8192)];
        let chunks = dist.distribute(1000, &devices).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_idx, 0);
        assert_eq!(chunks[0].end_idx, 1000);
    }

    #[test]
    fn test_distribute_covers_all_vectors() {
        let dist = WorkloadDistributor::new();
        let devices = vec![make_device(0, 4096), make_device(1, 8192)];
        let chunks = dist.distribute(900, &devices).unwrap();
        let covered: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(covered, 900, "All vectors must be covered");
    }

    #[test]
    fn test_distribute_proportional_to_memory() {
        let dist = WorkloadDistributor::new();
        // Device 0: 1 GB, device 1: 3 GB => 25 / 75 split
        let devices = vec![make_device(0, 1024), make_device(1, 3072)];
        let chunks = dist.distribute(1000, &devices).unwrap();
        assert_eq!(chunks.len(), 2);
        // Device 0 should get ~250 vectors
        let c0 = &chunks[0];
        let c1 = &chunks[1];
        assert!(
            c0.len() <= 300,
            "Device 0 should get ~25%, got {}",
            c0.len()
        );
        assert!(
            c1.len() >= 700,
            "Device 1 should get ~75%, got {}",
            c1.len()
        );
        assert_eq!(c0.start_idx, 0);
        assert_eq!(c1.end_idx, 1000);
    }

    #[test]
    fn test_distribute_skips_zero_memory_device() {
        let dist = WorkloadDistributor::new();
        let devices = vec![make_device(0, 0), make_device(1, 8192)];
        let chunks = dist.distribute(100, &devices).unwrap();
        // Device 0 is skipped; only device 1
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].device_id, 1);
    }

    #[test]
    fn test_distribute_even_basic() {
        let dist = WorkloadDistributor::new();
        let devices = vec![
            make_device(0, 4096),
            make_device(1, 4096),
            make_device(2, 4096),
        ];
        let chunks = dist.distribute_even(9, &devices).unwrap();
        assert_eq!(chunks.iter().map(|c| c.len()).sum::<usize>(), 9);
        for chunk in &chunks {
            assert_eq!(chunk.len(), 3);
        }
    }

    #[test]
    fn test_distribute_even_with_remainder() {
        let dist = WorkloadDistributor::new();
        let devices = vec![make_device(0, 4096), make_device(1, 4096)];
        let chunks = dist.distribute_even(7, &devices).unwrap();
        assert_eq!(chunks.iter().map(|c| c.len()).sum::<usize>(), 7);
        // First device gets 4, second gets 3
        assert_eq!(chunks[0].len(), 4);
        assert_eq!(chunks[1].len(), 3);
    }

    #[test]
    fn test_distribute_even_empty_devices_error() {
        let dist = WorkloadDistributor::new();
        assert!(dist.distribute_even(100, &[]).is_err());
    }
}
