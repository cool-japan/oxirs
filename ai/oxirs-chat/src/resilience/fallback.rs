//! Fallback strategies for graceful degradation

use anyhow::{anyhow, Result};
use std::sync::Arc;

/// Fallback strategy for when primary operation fails
pub enum FallbackStrategy<T> {
    /// Return a default value
    DefaultValue(T),

    /// Try alternative operation
    Alternative(Arc<dyn Fn() -> Result<T> + Send + Sync>),

    /// Return cached value if available
    Cache {
        get_cached: Arc<dyn Fn() -> Option<T> + Send + Sync>,
    },

    /// Degrade to simpler operation
    Degraded {
        operation: Arc<dyn Fn() -> Result<T> + Send + Sync>,
    },

    /// Fail silently (return None)
    Silent,
}

/// Chain of fallback strategies
pub struct FallbackChain {
    strategies: Vec<Box<dyn Fn() -> Result<String> + Send + Sync>>,
}

impl FallbackChain {
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
        }
    }

    /// Add a fallback strategy
    pub fn add<F>(&mut self, strategy: F)
    where
        F: Fn() -> Result<String> + Send + Sync + 'static,
    {
        self.strategies.push(Box::new(strategy));
    }

    /// Execute fallback chain
    pub fn execute(&self) -> Result<String> {
        for strategy in &self.strategies {
            if let Ok(result) = strategy() {
                return Ok(result);
            }
        }
        Err(anyhow!("All fallback strategies failed"))
    }
}

/// Execute operation with fallback
pub async fn execute_with_fallback<T, F, Fb>(
    primary: F,
    fallback: Fb,
) -> Result<T>
where
    F: FnOnce() -> Result<T>,
    Fb: FnOnce() -> Result<T>,
{
    match primary() {
        Ok(result) => Ok(result),
        Err(_) => fallback(),
    }
}

/// Graceful degradation handler
pub struct GracefulDegradation<T> {
    primary: Arc<dyn Fn() -> Result<T> + Send + Sync>,
    fallbacks: Vec<Arc<dyn Fn() -> Result<T> + Send + Sync>>,
    current_strategy: usize,
}

impl<T> GracefulDegradation<T> {
    pub fn new<F>(primary: F) -> Self
    where
        F: Fn() -> Result<T> + Send + Sync + 'static,
    {
        Self {
            primary: Arc::new(primary),
            fallbacks: Vec::new(),
            current_strategy: 0,
        }
    }

    /// Add a fallback strategy
    pub fn add_fallback<F>(&mut self, fallback: F)
    where
        F: Fn() -> Result<T> + Send + Sync + 'static,
    {
        self.fallbacks.push(Arc::new(fallback));
    }

    /// Execute with graceful degradation
    pub fn execute(&mut self) -> Result<T> {
        // Try primary first
        if let Ok(result) = (self.primary)() {
            self.current_strategy = 0;
            return Ok(result);
        }

        // Try fallbacks in order
        for (idx, fallback) in self.fallbacks.iter().enumerate() {
            if let Ok(result) = fallback() {
                self.current_strategy = idx + 1;
                return Ok(result);
            }
        }

        Err(anyhow!("All strategies failed"))
    }

    /// Get current strategy index (0 = primary, >0 = fallback)
    pub fn current_strategy(&self) -> usize {
        self.current_strategy
    }

    /// Check if currently degraded
    pub fn is_degraded(&self) -> bool {
        self.current_strategy > 0
    }
}

/// Cached fallback with TTL
pub struct CachedFallback<T> {
    cache: Arc<std::sync::RwLock<Option<CacheEntry<T>>>>,
    ttl: std::time::Duration,
}

#[derive(Clone)]
struct CacheEntry<T> {
    value: T,
    timestamp: std::time::Instant,
}

impl<T: Clone> CachedFallback<T> {
    pub fn new(ttl: std::time::Duration) -> Self {
        Self {
            cache: Arc::new(std::sync::RwLock::new(None)),
            ttl,
        }
    }

    /// Update cache with new value
    pub fn update(&self, value: T) -> Result<()> {
        let mut cache = self.cache.write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;
        *cache = Some(CacheEntry {
            value,
            timestamp: std::time::Instant::now(),
        });
        Ok(())
    }

    /// Get cached value if not expired
    pub fn get(&self) -> Result<Option<T>> {
        let cache = self.cache.read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;

        if let Some(entry) = cache.as_ref() {
            if entry.timestamp.elapsed() < self.ttl {
                return Ok(Some(entry.value.clone()));
            }
        }

        Ok(None)
    }

    /// Execute with cached fallback
    pub fn execute_with_cache<F>(&self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        match operation() {
            Ok(value) => {
                self.update(value.clone())?;
                Ok(value)
            }
            Err(_) => {
                if let Some(cached) = self.get()? {
                    Ok(cached)
                } else {
                    Err(anyhow!("No cached value available"))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fallback_success_primary() {
        let result = execute_with_fallback(
            || Ok::<_, anyhow::Error>(42),
            || Ok(0),
        )
        .await;

        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_fallback_uses_fallback() {
        let result = execute_with_fallback(
            || Err::<i32, _>(anyhow!("Primary failed")),
            || Ok(99),
        )
        .await;

        assert_eq!(result.unwrap(), 99);
    }

    #[test]
    fn test_graceful_degradation() {
        let mut degradation = GracefulDegradation::new(|| Err::<i32, _>(anyhow!("Primary fails")));

        degradation.add_fallback(|| Err::<i32, _>(anyhow!("First fallback fails")));
        degradation.add_fallback(|| Ok(42));

        let result = degradation.execute();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert!(degradation.is_degraded());
        assert_eq!(degradation.current_strategy(), 2);
    }

    #[test]
    fn test_cached_fallback_fresh() {
        let cache = CachedFallback::new(std::time::Duration::from_secs(60));

        let result = cache.execute_with_cache(|| Ok::<_, anyhow::Error>(42));
        assert_eq!(result.unwrap(), 42);

        // Should return cached value
        let cached = cache.get().unwrap();
        assert_eq!(cached, Some(42));
    }

    #[test]
    fn test_cached_fallback_expired() {
        let cache = CachedFallback::new(std::time::Duration::from_millis(10));

        cache.update(42).unwrap();

        // Wait for cache to expire
        std::thread::sleep(std::time::Duration::from_millis(20));

        let cached = cache.get().unwrap();
        assert_eq!(cached, None);
    }

    #[test]
    fn test_cached_fallback_on_failure() {
        let cache = CachedFallback::new(std::time::Duration::from_secs(60));

        // First call succeeds and caches
        let result = cache.execute_with_cache(|| Ok::<_, anyhow::Error>(42));
        assert_eq!(result.unwrap(), 42);

        // Second call fails but returns cached value
        let result = cache.execute_with_cache(|| Err::<i32, _>(anyhow!("Failed")));
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_fallback_chain() {
        let mut chain = FallbackChain::new();

        chain.add(|| Err(anyhow!("First fails")));
        chain.add(|| Err(anyhow!("Second fails")));
        chain.add(|| Ok("Third succeeds".to_string()));

        let result = chain.execute();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Third succeeds");
    }
}
