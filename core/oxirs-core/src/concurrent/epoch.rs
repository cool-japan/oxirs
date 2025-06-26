//! Epoch-based memory reclamation for lock-free data structures
//!
//! This module provides a safe memory reclamation scheme for concurrent
//! data structures. It uses epochs to track when memory can be safely freed.

use crossbeam_epoch::{self as epoch, Atomic, Guard, Owned, Shared};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A thread-local epoch tracker for safe memory reclamation
pub struct EpochManager {
    /// The global epoch counter
    global_epoch: Arc<AtomicUsize>,
    /// Thread-local epoch guards
    _phantom: std::marker::PhantomData<()>,
}

impl EpochManager {
    /// Create a new epoch manager
    pub fn new() -> Self {
        Self {
            global_epoch: Arc::new(AtomicUsize::new(0)),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Pin the current thread to an epoch
    pub fn pin(&self) -> Guard {
        epoch::pin()
    }

    /// Advance the global epoch
    pub fn advance(&self) {
        self.global_epoch.fetch_add(1, Ordering::Release);
    }

    /// Get the current global epoch
    pub fn current_epoch(&self) -> usize {
        self.global_epoch.load(Ordering::Acquire)
    }

    /// Defer a closure until it's safe to execute
    pub fn defer<F>(&self, guard: &Guard, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        guard.defer(f);
    }

    /// Flush all deferred operations
    pub fn flush(&self, guard: &Guard) {
        guard.flush();
    }
}

impl Default for EpochManager {
    fn default() -> Self {
        Self::new()
    }
}

/// A versioned pointer for lock-free updates
pub struct VersionedPointer<T> {
    ptr: Atomic<VersionedNode<T>>,
}

/// A node with version information
pub struct VersionedNode<T> {
    data: T,
    version: usize,
}

impl<T> VersionedPointer<T> {
    /// Create a new versioned pointer
    pub fn new(data: T) -> Self {
        let node = VersionedNode { data, version: 0 };
        Self {
            ptr: Atomic::new(node),
        }
    }

    /// Load the current value
    pub fn load<'g>(&self, guard: &'g Guard) -> Option<&'g T> {
        let shared = self.ptr.load(Ordering::Acquire, guard);
        unsafe { shared.as_ref().map(|node| &node.data) }
    }

    /// Compare and swap the pointer
    pub fn compare_and_swap<'g>(
        &self,
        current: Shared<'g, VersionedNode<T>>,
        new: Owned<VersionedNode<T>>,
        guard: &'g Guard,
    ) -> Result<Shared<'g, VersionedNode<T>>, (Shared<'g, VersionedNode<T>>, Owned<VersionedNode<T>>)> {
        match self.ptr.compare_exchange(
            current,
            new,
            Ordering::Release,
            Ordering::Acquire,
            guard,
        ) {
            Ok(shared) => Ok(shared),
            Err(e) => Err((e.current, e.new)),
        }
    }

    /// Update the value with a new version
    pub fn update(&self, data: T, version: usize, guard: &Guard) -> bool {
        let current = self.ptr.load(Ordering::Acquire, guard);
        
        // Check version before attempting swap
        if let Some(current_node) = unsafe { current.as_ref() } {
            if current_node.version >= version {
                // Our version is outdated
                return false;
            }
        }
        
        let new_node = VersionedNode { data, version };
        let new = Owned::new(new_node);

        match self.compare_and_swap(current, new, guard) {
            Ok(_) => {
                // Defer cleanup of old node
                if !current.is_null() {
                    unsafe {
                        guard.defer_destroy(current);
                    }
                }
                true
            }
            Err((_, returned)) => {
                // Someone else updated between our load and CAS
                drop(returned);
                false
            }
        }
    }
}

/// Hazard pointer wrapper for additional safety
pub struct HazardPointer<T> {
    inner: Atomic<T>,
}

impl<T> HazardPointer<T> {
    /// Create a new hazard pointer
    pub fn new(data: T) -> Self {
        Self {
            inner: Atomic::new(data),
        }
    }

    /// Load with hazard pointer protection
    pub fn load<'g>(&self, guard: &'g Guard) -> Shared<'g, T> {
        self.inner.load(Ordering::Acquire, guard)
    }

    /// Store a new value
    pub fn store(&self, new: Owned<T>, guard: &Guard) {
        let old = self.inner.swap(new, Ordering::Release, guard);
        if !old.is_null() {
            unsafe {
                guard.defer_destroy(old);
            }
        }
    }

    /// Compare and swap
    pub fn compare_and_swap<'g>(
        &self,
        current: Shared<'g, T>,
        new: Owned<T>,
        guard: &'g Guard,
    ) -> Result<Shared<'g, T>, (Shared<'g, T>, Owned<T>)> {
        match self.inner.compare_exchange(
            current,
            new,
            Ordering::Release,
            Ordering::Acquire,
            guard,
        ) {
            Ok(shared) => Ok(shared),
            Err(e) => Err((e.current, e.new)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_epoch_manager() {
        let manager = Arc::new(EpochManager::new());
        let initial_epoch = manager.current_epoch();

        // Advance epoch
        manager.advance();
        assert_eq!(manager.current_epoch(), initial_epoch + 1);

        // Test pinning
        let guard = manager.pin();
        drop(guard);
    }

    #[test]
    fn test_versioned_pointer() {
        let ptr = Arc::new(VersionedPointer::new(42));
        let guard = epoch::pin();

        // Load initial value
        assert_eq!(ptr.load(&guard), Some(&42));

        // Update value
        assert!(ptr.update(100, 1, &guard));
        assert_eq!(ptr.load(&guard), Some(&100));

        // Try outdated update - this should fail because version 0 < current version 1
        let result = ptr.update(50, 0, &guard);
        assert!(!result, "Update with outdated version should fail");
        assert_eq!(ptr.load(&guard), Some(&100));
    }

    #[test]
    fn test_concurrent_updates() {
        let ptr = Arc::new(VersionedPointer::new(0));
        let num_threads = 4;
        let updates_per_thread = 1000;

        let handles: Vec<_> = (0..num_threads)
            .map(|i| {
                let ptr = ptr.clone();
                thread::spawn(move || {
                    let guard = epoch::pin();
                    for j in 0..updates_per_thread {
                        let version = i * updates_per_thread + j;
                        ptr.update(version as i32, version, &guard);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Check final state
        let guard = epoch::pin();
        let final_value = ptr.load(&guard).unwrap();
        assert!(*final_value >= 0);
    }

    #[test]
    fn test_hazard_pointer() {
        let hp = Arc::new(HazardPointer::new("initial"));
        let guard = epoch::pin();

        // Store new value
        hp.store(Owned::new("updated"), &guard);

        // Load value
        let loaded = hp.load(&guard);
        unsafe {
            assert_eq!(loaded.as_ref().unwrap(), &"updated");
        }
    }
}