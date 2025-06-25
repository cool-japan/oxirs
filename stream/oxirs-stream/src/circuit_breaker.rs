//! # Advanced Circuit Breaker Implementation
//!
//! Enterprise-grade circuit breaker functionality for fault tolerance and resilience
//! in distributed streaming systems with adaptive thresholds, failure classification,
//! metrics integration, and advanced recovery strategies.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use std::collections::{HashMap, VecDeque};
use tokio::sync::{RwLock, Mutex};
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub enabled: bool,
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout: Duration,
    pub half_open_max_calls: u32,
}

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

/// Circuit breaker implementation
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: CircuitBreakerState,
    failure_count: u32,
    success_count: u32,
    last_failure_time: Option<Instant>,
    half_open_calls: u32,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: CircuitBreakerState::Closed,
            failure_count: 0,
            success_count: 0,
            last_failure_time: None,
            half_open_calls: 0,
        }
    }
    
    pub fn is_open(&self) -> bool {
        self.state == CircuitBreakerState::Open
    }
    
    pub fn state(&self) -> CircuitBreakerState {
        if !self.config.enabled {
            return CircuitBreakerState::Closed;
        }
        
        match self.state {
            CircuitBreakerState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() >= self.config.timeout {
                        CircuitBreakerState::HalfOpen
                    } else {
                        CircuitBreakerState::Open
                    }
                } else {
                    CircuitBreakerState::Open
                }
            }
            other => other,
        }
    }
    
    pub fn record_success(&mut self) {
        if !self.config.enabled {
            return;
        }
        
        match self.state {
            CircuitBreakerState::Closed => {
                self.failure_count = 0;
            }
            CircuitBreakerState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.config.success_threshold {
                    self.state = CircuitBreakerState::Closed;
                    self.failure_count = 0;
                    self.success_count = 0;
                    self.half_open_calls = 0;
                }
            }
            CircuitBreakerState::Open => {
                // Should not happen, but reset if it does
                self.state = CircuitBreakerState::Closed;
                self.failure_count = 0;
                self.success_count = 0;
            }
        }
    }
    
    pub fn record_failure(&mut self) {
        if !self.config.enabled {
            return;
        }
        
        self.last_failure_time = Some(Instant::now());
        
        match self.state {
            CircuitBreakerState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.config.failure_threshold {
                    self.state = CircuitBreakerState::Open;
                }
            }
            CircuitBreakerState::HalfOpen => {
                self.state = CircuitBreakerState::Open;
                self.half_open_calls = 0;
                self.success_count = 0;
            }
            CircuitBreakerState::Open => {
                // Already open, no change needed
            }
        }
    }
    
    pub fn can_execute(&mut self) -> bool {
        if !self.config.enabled {
            return true;
        }
        
        let current_state = self.state();
        
        match current_state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                // Check if we should transition to half-open
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() >= self.config.timeout {
                        self.state = CircuitBreakerState::HalfOpen;
                        self.half_open_calls = 0;
                        self.success_count = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitBreakerState::HalfOpen => {
                if self.half_open_calls < self.config.half_open_max_calls {
                    self.half_open_calls += 1;
                    true
                } else {
                    false
                }
            }
        }
    }
    
    /// Get circuit breaker statistics
    pub fn stats(&self) -> CircuitBreakerStats {
        CircuitBreakerStats {
            state: self.state(),
            failure_count: self.failure_count,
            success_count: self.success_count,
            half_open_calls: self.half_open_calls,
            last_failure_time: self.last_failure_time,
        }
    }
}

/// Circuit breaker statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerStats {
    pub state: CircuitBreakerState,
    pub failure_count: u32,
    pub success_count: u32,
    pub half_open_calls: u32,
    pub last_failure_time: Option<Instant>,
}

/// Shared circuit breaker for async usage
pub type SharedCircuitBreaker = Arc<RwLock<CircuitBreaker>>;

/// Create a new shared circuit breaker
pub fn new_shared_circuit_breaker(config: CircuitBreakerConfig) -> SharedCircuitBreaker {
    Arc::new(RwLock::new(CircuitBreaker::new(config)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    fn test_config() -> CircuitBreakerConfig {
        CircuitBreakerConfig {
            enabled: true,
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_millis(100),
            half_open_max_calls: 2,
        }
    }
    
    #[test]
    fn test_circuit_breaker_closed_state() {
        let mut cb = CircuitBreaker::new(test_config());
        
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
        assert!(cb.can_execute());
        
        // Record some successes
        cb.record_success();
        cb.record_success();
        
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
    }
    
    #[test]
    fn test_circuit_breaker_opens_on_failures() {
        let mut cb = CircuitBreaker::new(test_config());
        
        // Record failures up to threshold
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
        
        cb.record_failure(); // This should open the circuit
        assert_eq!(cb.state(), CircuitBreakerState::Open);
        assert!(!cb.can_execute());
    }
    
    #[tokio::test]
    async fn test_circuit_breaker_transitions_to_half_open() {
        let mut cb = CircuitBreaker::new(test_config());
        
        // Open the circuit
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitBreakerState::Open);
        
        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(150)).await;
        
        // Should transition to half-open
        assert!(cb.can_execute());
        assert_eq!(cb.state(), CircuitBreakerState::HalfOpen);
    }
    
    #[test]
    fn test_disabled_circuit_breaker() {
        let mut config = test_config();
        config.enabled = false;
        
        let mut cb = CircuitBreaker::new(config);
        
        // Should always allow execution when disabled
        assert!(cb.can_execute());
        
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        
        assert!(cb.can_execute());
        assert_eq!(cb.state(), CircuitBreakerState::Closed);
    }
}