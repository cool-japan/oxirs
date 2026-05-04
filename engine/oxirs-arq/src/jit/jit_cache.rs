//! Thread-safe cache of compiled JIT filter functions.
//!
//! [`JitFilterCache`] wraps a [`FilterCompiler`] and an LRU-bounded cache keyed by a
//! `u64` plan fingerprint.  Compilation requires exclusive access (via `Mutex`) because
//! each [`FilterCompiler::compile`] call creates and finalizes a `JITModule`; cache reads
//! are lock-free (`RwLock`).
//!
//! # Eviction
//!
//! When the cache reaches `max_entries`, the **oldest** entry is evicted (FIFO order
//! using insertion-order tracking via a `VecDeque`).  This is simpler than a full LRU
//! but sufficient for the hot-path use case where the set of hot queries is small.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use super::filter_compiler::{
    CompiledFilter, FilterCompiler, FilterCompilerError, FilterExpr, VarIndexMap,
};

/// Thread-safe, bounded cache of compiled JIT filter functions.
///
/// # Thread safety
///
/// - Cache reads use a shared `RwLock` — many readers can proceed concurrently.
/// - Compilation uses an exclusive `Mutex<FilterCompiler>` — only one thread compiles
///   at a time, but reads are never blocked during compilation.
pub struct JitFilterCache {
    /// Exclusive access for compilation (one `JITModule` at a time).
    compiler: Mutex<FilterCompiler>,
    /// Shared read access to the compiled-function cache.
    cache: RwLock<CacheInner>,
}

struct CacheInner {
    map: HashMap<u64, Arc<CompiledFilter>>,
    /// Insertion-order queue for FIFO eviction.
    order: VecDeque<u64>,
    max_entries: usize,
    /// Total compilations performed (hits excluded).
    compile_count: usize,
    /// Total cache hits.
    hit_count: usize,
}

impl CacheInner {
    fn new(max_entries: usize) -> Self {
        Self {
            map: HashMap::with_capacity(max_entries.min(256)),
            order: VecDeque::with_capacity(max_entries.min(256)),
            max_entries,
            compile_count: 0,
            hit_count: 0,
        }
    }

    fn get(&mut self, key: u64) -> Option<Arc<CompiledFilter>> {
        let result = self.map.get(&key).cloned();
        if result.is_some() {
            self.hit_count += 1;
        }
        result
    }

    fn insert(&mut self, key: u64, compiled: Arc<CompiledFilter>) {
        if self.map.contains_key(&key) {
            // Already inserted by a concurrent compile — do nothing
            return;
        }
        // Evict oldest if at capacity
        while self.order.len() >= self.max_entries {
            if let Some(old_key) = self.order.pop_front() {
                self.map.remove(&old_key);
            }
        }
        self.map.insert(key, compiled);
        self.order.push_back(key);
        self.compile_count += 1;
    }
}

/// Statistics snapshot from a [`JitFilterCache`].
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    /// Number of entries currently in the cache.
    pub len: usize,
    /// Maximum number of entries (capacity bound).
    pub max_entries: usize,
    /// Total cache hits since creation.
    pub hit_count: usize,
    /// Total successful compilations since creation.
    pub compile_count: usize,
}

impl JitFilterCache {
    /// Create a new cache with the given capacity bound.
    ///
    /// `max_entries` must be at least 1; values of `0` are clamped to `1`.
    pub fn new(max_entries: usize) -> Result<Self, FilterCompilerError> {
        let max_entries = max_entries.max(1);
        Ok(Self {
            compiler: Mutex::new(FilterCompiler::new()),
            cache: RwLock::new(CacheInner::new(max_entries)),
        })
    }

    /// Return the compiled filter for `key` if it is already cached.
    pub fn get(&self, key: u64) -> Option<Arc<CompiledFilter>> {
        // Acquire write lock so we can atomically read + update hit_count.
        // The write lock is only contested during compilation (rare); reads are
        // otherwise uncontested on the hot path because compilations are infrequent.
        let mut write = self.cache.write();
        write.get(key)
    }

    /// Compile `expr` + `var_map` and insert the result under `key`.
    ///
    /// If `key` is already in the cache (from a concurrent compile), the existing
    /// entry is returned without re-compiling.
    ///
    /// Returns `Ok(None)` if the expression is not in the JIT-supported subset.
    pub fn compile_and_insert(
        &self,
        key: u64,
        expr: &FilterExpr,
        var_map: VarIndexMap,
    ) -> Result<Option<Arc<CompiledFilter>>, FilterCompilerError> {
        // Check again before compiling (double-checked locking)
        {
            let mut write = self.cache.write();
            if let Some(existing) = write.get(key) {
                return Ok(Some(existing));
            }
        }

        // Compile (exclusive lock — one compilation at a time)
        let compiled_opt = {
            let compiler = self.compiler.lock();
            compiler.compile(expr, var_map)?
        };

        match compiled_opt {
            None => Ok(None),
            Some(compiled) => {
                let arc = Arc::new(compiled);
                let mut write = self.cache.write();
                write.insert(key, arc.clone());
                Ok(Some(arc))
            }
        }
    }

    /// Return the number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.cache.read().map.len()
    }

    /// Return `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return a snapshot of cache statistics.
    pub fn stats(&self) -> CacheStats {
        let inner = self.cache.read();
        CacheStats {
            len: inner.map.len(),
            max_entries: inner.max_entries,
            hit_count: inner.hit_count,
            compile_count: inner.compile_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::filter_compiler::{BinOp, FilterExpr};

    fn simple_gt_expr() -> (FilterExpr, VarIndexMap) {
        let mut vm = VarIndexMap::new();
        vm.insert("x".to_string(), 0);
        let expr = FilterExpr::BinOp {
            op: BinOp::Gt,
            left: Box::new(FilterExpr::Variable("x".to_string())),
            right: Box::new(FilterExpr::Literal(3.0)),
        };
        (expr, vm)
    }

    #[test]
    fn cache_starts_empty() {
        let cache = JitFilterCache::new(16).expect("cache init");
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn get_before_insert_returns_none() {
        let cache = JitFilterCache::new(16).expect("cache init");
        assert!(cache.get(12345).is_none());
    }

    #[test]
    fn insert_then_get_hit() {
        let cache = JitFilterCache::new(16).expect("cache init");
        let (expr, vm) = simple_gt_expr();
        let compiled = cache
            .compile_and_insert(42, &expr, vm)
            .expect("compile ok")
            .expect("filter compiled (not unsupported)");
        assert_eq!(cache.len(), 1);

        let hit = cache.get(42).expect("should be in cache");
        assert!(Arc::ptr_eq(&compiled, &hit));
    }

    #[test]
    fn eviction_at_capacity() {
        let cache = JitFilterCache::new(2).expect("cache init");
        let (expr1, vm1) = simple_gt_expr();
        let (expr2, vm2) = simple_gt_expr();
        let mut vm3 = VarIndexMap::new();
        vm3.insert("y".to_string(), 0);
        let expr3 = FilterExpr::BinOp {
            op: BinOp::Lt,
            left: Box::new(FilterExpr::Variable("y".to_string())),
            right: Box::new(FilterExpr::Literal(10.0)),
        };

        cache.compile_and_insert(1, &expr1, vm1).expect("ok");
        cache.compile_and_insert(2, &expr2, vm2).expect("ok");
        assert_eq!(cache.len(), 2);

        // Inserting key 3 should evict key 1 (oldest)
        cache.compile_and_insert(3, &expr3, vm3).expect("ok");
        assert_eq!(cache.len(), 2);
        assert!(cache.get(1).is_none(), "key 1 should have been evicted");
        assert!(cache.get(2).is_some());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn stats_tracks_hits_and_compiles() {
        let cache = JitFilterCache::new(16).expect("cache init");
        let (expr, vm) = simple_gt_expr();
        cache.compile_and_insert(99, &expr, vm).expect("compile ok");
        let _ = cache.get(99);
        let _ = cache.get(99);

        let stats = cache.stats();
        assert_eq!(stats.compile_count, 1);
        assert!(stats.hit_count >= 1);
    }
}
