//! Backpressure bridge between the cluster Raft log and upstream stream operators.
//!
//! The [`BackpressureBridge`] tracks the depth of the in-flight queue and
//! produces a coarse-grained [`BackpressureSignal`] (`Continue` / `Slow` /
//! `Stop`) that upstream operators poll. This keeps the integration loosely
//! coupled — the upstream code only needs to read a single atomic signal and
//! does not need to be aware of Raft mechanics.
//!
//! ## Hysteresis
//!
//! Two watermarks are configured:
//!
//! * `slow_high_watermark` — once queue depth crosses this, the bridge moves
//!   from `Continue` to `Slow`.
//! * `stop_high_watermark` — once queue depth crosses this, the bridge moves
//!   to `Stop`.
//!
//! Recovery uses `slow_low_watermark` / `stop_low_watermark` (each must be
//! `<=` its high counterpart) so that hysteresis prevents thrashing.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::{ClusterError, Result};

/// Coarse-grained signal that the upstream stream operators read to apply
/// backpressure. Encoded as a single [`u8`] for cheap atomic loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum BackpressureSignal {
    /// Cluster log can absorb more events at full speed.
    Continue = 0,
    /// Cluster log is approaching capacity. Upstream should slow down.
    Slow = 1,
    /// Cluster log is saturated. Upstream must stop sending.
    Stop = 2,
}

impl BackpressureSignal {
    /// Decodes a raw `u8` (atomic state) back into a signal.
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Slow,
            2 => Self::Stop,
            _ => Self::Continue,
        }
    }
}

impl Default for BackpressureSignal {
    fn default() -> Self {
        Self::Continue
    }
}

/// Configuration for [`BackpressureBridge`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackpressureConfig {
    /// Queue depth at which the bridge transitions `Continue → Slow`.
    pub slow_high_watermark: u64,
    /// Queue depth at which the bridge transitions back `Slow → Continue`.
    pub slow_low_watermark: u64,
    /// Queue depth at which the bridge transitions `Slow → Stop`.
    pub stop_high_watermark: u64,
    /// Queue depth at which the bridge transitions back `Stop → Slow`.
    pub stop_low_watermark: u64,
}

impl BackpressureConfig {
    /// Validate that watermarks form a sensible monotone sequence.
    pub fn validate(&self) -> Result<()> {
        if self.slow_low_watermark > self.slow_high_watermark {
            return Err(ClusterError::Config(
                "slow_low_watermark must be <= slow_high_watermark".into(),
            ));
        }
        if self.stop_low_watermark > self.stop_high_watermark {
            return Err(ClusterError::Config(
                "stop_low_watermark must be <= stop_high_watermark".into(),
            ));
        }
        if self.slow_high_watermark > self.stop_high_watermark {
            return Err(ClusterError::Config(
                "slow_high_watermark must be <= stop_high_watermark".into(),
            ));
        }
        Ok(())
    }
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            slow_high_watermark: 1_000,
            slow_low_watermark: 800,
            stop_high_watermark: 4_000,
            stop_low_watermark: 3_500,
        }
    }
}

/// Threshold-driven backpressure bridge between the cluster's in-flight log
/// queue and upstream stream operators.
#[derive(Debug, Clone)]
pub struct BackpressureBridge {
    inner: Arc<BridgeInner>,
}

#[derive(Debug)]
struct BridgeInner {
    config: BackpressureConfig,
    /// Current queue depth as observed by the cluster sink.
    depth: AtomicU64,
    /// Cached encoded [`BackpressureSignal`] for fast polls.
    signal: AtomicU8,
    /// Total number of times the signal transitioned (any direction).
    transitions: AtomicU64,
}

impl BackpressureBridge {
    /// Creates a new bridge from the given watermark configuration.
    pub fn new(config: BackpressureConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            inner: Arc::new(BridgeInner {
                config,
                depth: AtomicU64::new(0),
                signal: AtomicU8::new(BackpressureSignal::Continue as u8),
                transitions: AtomicU64::new(0),
            }),
        })
    }

    /// Returns the configured watermarks.
    pub fn config(&self) -> &BackpressureConfig {
        &self.inner.config
    }

    /// Returns the most recently observed queue depth.
    pub fn depth(&self) -> u64 {
        self.inner.depth.load(Ordering::Acquire)
    }

    /// Records a new queue depth and recomputes the [`BackpressureSignal`].
    /// Returns the (possibly updated) signal.
    pub fn observe(&self, depth: u64) -> BackpressureSignal {
        self.inner.depth.store(depth, Ordering::Release);
        self.recompute_signal(depth)
    }

    /// Increments the queue depth by `delta` (saturating at [`u64::MAX`]).
    pub fn add(&self, delta: u64) -> BackpressureSignal {
        let new_depth = self.inner.depth.fetch_add(delta, Ordering::AcqRel) + delta;
        self.recompute_signal(new_depth)
    }

    /// Decrements the queue depth by `delta` (saturating at zero).
    pub fn sub(&self, delta: u64) -> BackpressureSignal {
        let mut current = self.inner.depth.load(Ordering::Acquire);
        loop {
            let next = current.saturating_sub(delta);
            match self.inner.depth.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return self.recompute_signal(next),
                Err(actual) => current = actual,
            }
        }
    }

    /// Returns the current [`BackpressureSignal`] without mutating depth.
    pub fn signal(&self) -> BackpressureSignal {
        BackpressureSignal::from_u8(self.inner.signal.load(Ordering::Acquire))
    }

    /// Returns the running count of signal transitions.
    pub fn transitions(&self) -> u64 {
        self.inner.transitions.load(Ordering::Acquire)
    }

    /// Computes the new signal for a given depth and updates the cached value.
    fn recompute_signal(&self, depth: u64) -> BackpressureSignal {
        let cfg = &self.inner.config;
        let prev = BackpressureSignal::from_u8(self.inner.signal.load(Ordering::Acquire));

        let next = match prev {
            BackpressureSignal::Continue => {
                if depth >= cfg.stop_high_watermark {
                    BackpressureSignal::Stop
                } else if depth >= cfg.slow_high_watermark {
                    BackpressureSignal::Slow
                } else {
                    BackpressureSignal::Continue
                }
            }
            BackpressureSignal::Slow => {
                if depth >= cfg.stop_high_watermark {
                    BackpressureSignal::Stop
                } else if depth <= cfg.slow_low_watermark {
                    BackpressureSignal::Continue
                } else {
                    BackpressureSignal::Slow
                }
            }
            BackpressureSignal::Stop => {
                if depth <= cfg.stop_low_watermark {
                    if depth <= cfg.slow_low_watermark {
                        BackpressureSignal::Continue
                    } else {
                        BackpressureSignal::Slow
                    }
                } else {
                    BackpressureSignal::Stop
                }
            }
        };

        if next != prev {
            self.inner.signal.store(next as u8, Ordering::Release);
            self.inner.transitions.fetch_add(1, Ordering::AcqRel);
        }

        next
    }
}

impl Default for BackpressureBridge {
    fn default() -> Self {
        Self::new(BackpressureConfig::default()).expect("default config is always valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_bridge() -> BackpressureBridge {
        let config = BackpressureConfig {
            slow_low_watermark: 5,
            slow_high_watermark: 10,
            stop_low_watermark: 25,
            stop_high_watermark: 30,
        };
        BackpressureBridge::new(config).expect("valid")
    }

    #[test]
    fn config_validation_rejects_inverted_watermarks() {
        let cfg = BackpressureConfig {
            slow_low_watermark: 100,
            slow_high_watermark: 10,
            stop_low_watermark: 0,
            stop_high_watermark: 0,
        };
        assert!(BackpressureBridge::new(cfg).is_err());
    }

    #[test]
    fn config_validation_rejects_slow_above_stop() {
        let cfg = BackpressureConfig {
            slow_low_watermark: 0,
            slow_high_watermark: 100,
            stop_low_watermark: 0,
            stop_high_watermark: 50,
        };
        assert!(BackpressureBridge::new(cfg).is_err());
    }

    #[test]
    fn default_signal_is_continue() {
        let bridge = small_bridge();
        assert_eq!(bridge.signal(), BackpressureSignal::Continue);
    }

    #[test]
    fn observe_transitions_to_slow_then_stop() {
        let bridge = small_bridge();
        bridge.observe(0);
        assert_eq!(bridge.signal(), BackpressureSignal::Continue);
        bridge.observe(10);
        assert_eq!(bridge.signal(), BackpressureSignal::Slow);
        bridge.observe(30);
        assert_eq!(bridge.signal(), BackpressureSignal::Stop);
    }

    #[test]
    fn hysteresis_holds_in_slow_band() {
        let bridge = small_bridge();
        bridge.observe(10); // Slow
        bridge.observe(8); // Still Slow (above slow_low_watermark = 5)
        assert_eq!(bridge.signal(), BackpressureSignal::Slow);
        bridge.observe(4); // Drop below slow_low — Continue
        assert_eq!(bridge.signal(), BackpressureSignal::Continue);
    }

    #[test]
    fn stop_recovers_through_slow_band() {
        let bridge = small_bridge();
        bridge.observe(40); // Stop
        assert_eq!(bridge.signal(), BackpressureSignal::Stop);
        bridge.observe(28); // Above stop_low_watermark = 25 → still Stop.
        assert_eq!(bridge.signal(), BackpressureSignal::Stop);
        bridge.observe(20); // <= stop_low_watermark, but > slow_low → Slow.
        assert_eq!(bridge.signal(), BackpressureSignal::Slow);
        bridge.observe(2); // Continue.
        assert_eq!(bridge.signal(), BackpressureSignal::Continue);
    }

    #[test]
    fn add_and_sub_track_depth() {
        let bridge = small_bridge();
        assert_eq!(bridge.depth(), 0);
        let _ = bridge.add(7);
        assert_eq!(bridge.depth(), 7);
        let _ = bridge.add(20);
        assert_eq!(bridge.depth(), 27);
        let _ = bridge.sub(15);
        assert_eq!(bridge.depth(), 12);
    }

    #[test]
    fn sub_saturates_at_zero() {
        let bridge = small_bridge();
        let _ = bridge.sub(50);
        assert_eq!(bridge.depth(), 0);
    }

    #[test]
    fn transitions_counter_increases_only_when_signal_changes() {
        let bridge = small_bridge();
        bridge.observe(0); // already Continue, no transition
        bridge.observe(0);
        bridge.observe(0);
        assert_eq!(bridge.transitions(), 0);
        bridge.observe(15); // Continue → Slow (count = 1)
        bridge.observe(20); // still Slow (no change)
        bridge.observe(40); // Slow → Stop (count = 2)
        assert_eq!(bridge.transitions(), 2);
    }

    #[test]
    fn signal_round_trip_via_u8() {
        for s in [
            BackpressureSignal::Continue,
            BackpressureSignal::Slow,
            BackpressureSignal::Stop,
        ] {
            assert_eq!(BackpressureSignal::from_u8(s as u8), s);
        }
    }

    #[test]
    fn default_signal_value_is_continue() {
        assert_eq!(BackpressureSignal::default(), BackpressureSignal::Continue);
    }
}
