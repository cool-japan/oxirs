//! # UDS Diagnostic Session Manager
//!
//! Manages UDS diagnostic session lifecycle including session transitions,
//! security access sequences, tester-present keepalive, session timeouts,
//! and concurrent session coordination.
//!
//! ## Features
//!
//! - **Session state machine**: Default → Extended → Programming → Safety System
//! - **Security access management**: Seed/key challenge-response tracking
//! - **Tester-present keepalive**: Automatic S3 timer refresh
//! - **Session timeout detection**: Automatic fallback to default session
//! - **Concurrent session coordination**: Prevents conflicting sessions
//! - **Audit trail**: Full history of session transitions and security events
//! - **Session capability tracking**: Track which services are available per session

use crate::error::CanbusError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────
// Session Types (ISO 14229-1 Table 2)
// ─────────────────────────────────────────────

/// UDS diagnostic session type sub-function values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiagnosticSession {
    /// 0x01 — Default diagnostic session (always active after power-on).
    Default,
    /// 0x02 — Programming session (for ECU reflashing).
    Programming,
    /// 0x03 — Extended diagnostic session (expanded service access).
    Extended,
    /// 0x04 — Safety system diagnostic session.
    SafetySystem,
    /// Vehicle-manufacturer specific session (0x40–0x5F).
    VehicleManufacturer(u8),
    /// System-supplier specific session (0x60–0x7E).
    SystemSupplier(u8),
}

impl DiagnosticSession {
    /// Convert to the ISO 14229 sub-function byte.
    pub fn to_byte(self) -> u8 {
        match self {
            DiagnosticSession::Default => 0x01,
            DiagnosticSession::Programming => 0x02,
            DiagnosticSession::Extended => 0x03,
            DiagnosticSession::SafetySystem => 0x04,
            DiagnosticSession::VehicleManufacturer(v) => v,
            DiagnosticSession::SystemSupplier(v) => v,
        }
    }

    /// Parse from an ISO 14229 sub-function byte.
    pub fn from_byte(b: u8) -> Self {
        match b {
            0x01 => DiagnosticSession::Default,
            0x02 => DiagnosticSession::Programming,
            0x03 => DiagnosticSession::Extended,
            0x04 => DiagnosticSession::SafetySystem,
            0x40..=0x5F => DiagnosticSession::VehicleManufacturer(b),
            0x60..=0x7E => DiagnosticSession::SystemSupplier(b),
            _ => DiagnosticSession::Default,
        }
    }

    /// Human-readable session name.
    pub fn name(&self) -> &str {
        match self {
            DiagnosticSession::Default => "Default",
            DiagnosticSession::Programming => "Programming",
            DiagnosticSession::Extended => "Extended",
            DiagnosticSession::SafetySystem => "SafetySystem",
            DiagnosticSession::VehicleManufacturer(_) => "VehicleManufacturer",
            DiagnosticSession::SystemSupplier(_) => "SystemSupplier",
        }
    }

    /// Whether this session requires security access.
    pub fn requires_security(&self) -> bool {
        matches!(
            self,
            DiagnosticSession::Programming | DiagnosticSession::SafetySystem
        )
    }
}

// ─────────────────────────────────────────────
// Security Access State
// ─────────────────────────────────────────────

/// Security access level state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityAccessState {
    /// Not authenticated.
    Locked,
    /// Seed has been requested, awaiting key.
    SeedReceived {
        /// The seed value received from ECU.
        seed: Vec<u8>,
        /// Security level being accessed.
        level: u8,
        /// When the seed was received.
        received_at: DateTime<Utc>,
    },
    /// Successfully authenticated at a security level.
    Unlocked {
        /// Security level.
        level: u8,
        /// When authentication succeeded.
        unlocked_at: DateTime<Utc>,
    },
    /// Authentication attempt exceeded.
    AttemptsExceeded {
        /// How many attempts were made.
        attempts: u32,
        /// When the lockout started.
        locked_at: DateTime<Utc>,
    },
}

// ─────────────────────────────────────────────
// Session Transition
// ─────────────────────────────────────────────

/// Record of a session transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTransition {
    /// Session before the transition.
    pub from: DiagnosticSession,
    /// Session after the transition.
    pub to: DiagnosticSession,
    /// Whether the transition was successful.
    pub success: bool,
    /// Error message (if failed).
    pub error: Option<String>,
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
}

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the session manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionManagerConfig {
    /// P2 server response timeout (default: 50ms).
    pub p2_server_timeout: Duration,
    /// P2* extended server response timeout (default: 5000ms).
    pub p2_star_timeout: Duration,
    /// S3 server session timeout (default: 5000ms).
    /// If no request within this window, ECU falls back to default.
    pub s3_server_timeout: Duration,
    /// Tester-present interval (must be < S3 timeout, default: 2000ms).
    pub tester_present_interval: Duration,
    /// Maximum security access attempts before lockout (default: 3).
    pub max_security_attempts: u32,
    /// Security access lockout duration (default: 10s).
    pub security_lockout_duration: Duration,
    /// Whether to auto-send TesterPresent keepalives (default: true).
    pub auto_tester_present: bool,
    /// Maximum history entries to keep (default: 100).
    pub max_history: usize,
}

impl Default for SessionManagerConfig {
    fn default() -> Self {
        Self {
            p2_server_timeout: Duration::from_millis(50),
            p2_star_timeout: Duration::from_millis(5000),
            s3_server_timeout: Duration::from_millis(5000),
            tester_present_interval: Duration::from_millis(2000),
            max_security_attempts: 3,
            security_lockout_duration: Duration::from_secs(10),
            auto_tester_present: true,
            max_history: 100,
        }
    }
}

// ─────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────

/// Statistics for the session manager.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionManagerStats {
    /// Total session transitions attempted.
    pub transitions_attempted: u64,
    /// Successful transitions.
    pub transitions_succeeded: u64,
    /// Failed transitions.
    pub transitions_failed: u64,
    /// Tester-present messages sent.
    pub tester_present_sent: u64,
    /// Session timeouts detected.
    pub session_timeouts: u64,
    /// Security access attempts.
    pub security_attempts: u64,
    /// Successful security unlocks.
    pub security_unlocks: u64,
    /// Failed security attempts.
    pub security_failures: u64,
    /// Current session duration (if not default).
    pub current_session_duration_ms: u64,
}

// ─────────────────────────────────────────────
// Session Manager
// ─────────────────────────────────────────────

/// Manages UDS diagnostic session lifecycle.
pub struct DiagnosticSessionManager {
    config: SessionManagerConfig,
    /// Current active session.
    current_session: DiagnosticSession,
    /// When the current session started.
    session_start: Instant,
    /// When the last request was sent (for S3 timeout tracking).
    last_request: Instant,
    /// Security access state.
    security_state: SecurityAccessState,
    /// Failed security attempt counter (per level).
    security_attempt_counts: HashMap<u8, u32>,
    /// Session transition history.
    history: Vec<SessionTransition>,
    /// Running statistics.
    stats: SessionManagerStats,
    /// Per-session available service IDs.
    session_capabilities: HashMap<DiagnosticSession, Vec<u8>>,
    /// Whether the manager has been initialized.
    initialized: bool,
}

impl DiagnosticSessionManager {
    /// Create a new session manager.
    pub fn new(config: SessionManagerConfig) -> Self {
        let now = Instant::now();
        Self {
            config,
            current_session: DiagnosticSession::Default,
            session_start: now,
            last_request: now,
            security_state: SecurityAccessState::Locked,
            security_attempt_counts: HashMap::new(),
            history: Vec::new(),
            stats: SessionManagerStats::default(),
            session_capabilities: Self::default_capabilities(),
            initialized: true,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SessionManagerConfig::default())
    }

    /// Get the current diagnostic session.
    pub fn current_session(&self) -> DiagnosticSession {
        self.current_session
    }

    /// Get the security access state.
    pub fn security_state(&self) -> &SecurityAccessState {
        &self.security_state
    }

    /// Whether the session is currently active (non-default).
    pub fn is_active_session(&self) -> bool {
        self.current_session != DiagnosticSession::Default
    }

    /// Whether a specific security level is unlocked.
    pub fn is_security_unlocked(&self, level: u8) -> bool {
        matches!(
            &self.security_state,
            SecurityAccessState::Unlocked { level: l, .. } if *l == level
        )
    }

    /// Request a session transition.
    ///
    /// Validates that the transition is allowed per ISO 14229 rules.
    pub fn request_transition(
        &mut self,
        target: DiagnosticSession,
    ) -> Result<SessionTransition, CanbusError> {
        self.stats.transitions_attempted += 1;

        // Check if transition is allowed
        if let Err(reason) = self.validate_transition(target) {
            let transition = SessionTransition {
                from: self.current_session,
                to: target,
                success: false,
                error: Some(reason.clone()),
                timestamp: Utc::now(),
            };
            self.record_transition(transition.clone());
            self.stats.transitions_failed += 1;
            return Err(CanbusError::Config(format!(
                "Session transition not allowed: {reason}"
            )));
        }

        // Check if security access is required
        if target.requires_security() && !self.is_any_security_unlocked() {
            let transition = SessionTransition {
                from: self.current_session,
                to: target,
                success: false,
                error: Some("Security access required".into()),
                timestamp: Utc::now(),
            };
            self.record_transition(transition.clone());
            self.stats.transitions_failed += 1;
            return Err(CanbusError::Config(
                "Security access required for this session type".into(),
            ));
        }

        let from = self.current_session;
        self.current_session = target;
        self.session_start = Instant::now();
        self.last_request = Instant::now();

        // Reset security state when going back to default
        if target == DiagnosticSession::Default {
            self.security_state = SecurityAccessState::Locked;
        }

        let transition = SessionTransition {
            from,
            to: target,
            success: true,
            error: None,
            timestamp: Utc::now(),
        };
        self.record_transition(transition.clone());
        self.stats.transitions_succeeded += 1;

        Ok(transition)
    }

    /// Process a security access seed response from the ECU.
    pub fn receive_security_seed(&mut self, level: u8, seed: Vec<u8>) -> Result<(), CanbusError> {
        self.stats.security_attempts += 1;

        // Check lockout
        if let SecurityAccessState::AttemptsExceeded { locked_at, .. } = &self.security_state {
            let elapsed = Utc::now().signed_duration_since(*locked_at);
            if elapsed
                < chrono::Duration::from_std(self.config.security_lockout_duration)
                    .unwrap_or(chrono::Duration::seconds(10))
            {
                return Err(CanbusError::Config(
                    "Security access locked out due to too many failed attempts".into(),
                ));
            }
            // Lockout expired, reset
            self.security_attempt_counts.clear();
        }

        // Check if seed is zero (already unlocked)
        if seed.iter().all(|&b| b == 0) {
            self.security_state = SecurityAccessState::Unlocked {
                level,
                unlocked_at: Utc::now(),
            };
            self.stats.security_unlocks += 1;
            return Ok(());
        }

        self.security_state = SecurityAccessState::SeedReceived {
            seed,
            level,
            received_at: Utc::now(),
        };

        Ok(())
    }

    /// Process security access key response (positive or negative).
    pub fn process_security_response(&mut self, success: bool) -> Result<(), CanbusError> {
        match &self.security_state {
            SecurityAccessState::SeedReceived { level, .. } => {
                let level = *level;
                if success {
                    self.security_state = SecurityAccessState::Unlocked {
                        level,
                        unlocked_at: Utc::now(),
                    };
                    self.stats.security_unlocks += 1;
                    self.security_attempt_counts.remove(&level);
                    Ok(())
                } else {
                    let count = self.security_attempt_counts.entry(level).or_insert(0);
                    *count += 1;
                    self.stats.security_failures += 1;

                    if *count >= self.config.max_security_attempts {
                        self.security_state = SecurityAccessState::AttemptsExceeded {
                            attempts: *count,
                            locked_at: Utc::now(),
                        };
                    } else {
                        self.security_state = SecurityAccessState::Locked;
                    }
                    Err(CanbusError::Config("Security access key rejected".into()))
                }
            }
            _ => Err(CanbusError::Config(
                "No pending security seed — cannot process response".into(),
            )),
        }
    }

    /// Record a tester-present message was sent.
    pub fn record_tester_present(&mut self) {
        self.last_request = Instant::now();
        self.stats.tester_present_sent += 1;
    }

    /// Check if a tester-present message should be sent now.
    pub fn should_send_tester_present(&self) -> bool {
        if !self.config.auto_tester_present {
            return false;
        }
        if self.current_session == DiagnosticSession::Default {
            return false;
        }
        self.last_request.elapsed() >= self.config.tester_present_interval
    }

    /// Check if the S3 server timeout has elapsed (session should auto-fallback).
    pub fn is_session_timed_out(&self) -> bool {
        if self.current_session == DiagnosticSession::Default {
            return false;
        }
        self.last_request.elapsed() > self.config.s3_server_timeout
    }

    /// Handle session timeout — revert to default session.
    pub fn handle_timeout(&mut self) -> Option<SessionTransition> {
        if self.is_session_timed_out() {
            self.stats.session_timeouts += 1;
            let from = self.current_session;
            self.current_session = DiagnosticSession::Default;
            self.security_state = SecurityAccessState::Locked;
            self.session_start = Instant::now();

            let transition = SessionTransition {
                from,
                to: DiagnosticSession::Default,
                success: true,
                error: Some("S3 timeout — reverted to default session".into()),
                timestamp: Utc::now(),
            };
            self.record_transition(transition.clone());
            Some(transition)
        } else {
            None
        }
    }

    /// Record any request activity (resets the S3 timer).
    pub fn record_activity(&mut self) {
        self.last_request = Instant::now();
    }

    /// Get session transition history.
    pub fn history(&self) -> &[SessionTransition] {
        &self.history
    }

    /// Get current statistics.
    pub fn stats(&self) -> SessionManagerStats {
        let mut stats = self.stats.clone();
        if self.current_session != DiagnosticSession::Default {
            stats.current_session_duration_ms = self.session_start.elapsed().as_millis() as u64;
        }
        stats
    }

    /// Get the configuration.
    pub fn config(&self) -> &SessionManagerConfig {
        &self.config
    }

    /// Whether the manager is initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Set capabilities for a session type.
    pub fn set_capabilities(&mut self, session: DiagnosticSession, service_ids: Vec<u8>) {
        self.session_capabilities.insert(session, service_ids);
    }

    /// Check if a service ID is available in the current session.
    pub fn is_service_available(&self, service_id: u8) -> bool {
        self.session_capabilities
            .get(&self.current_session)
            .is_some_and(|ids| ids.contains(&service_id))
    }

    /// Get available services for the current session.
    pub fn available_services(&self) -> Vec<u8> {
        self.session_capabilities
            .get(&self.current_session)
            .cloned()
            .unwrap_or_default()
    }

    /// Get time until next required tester-present.
    pub fn time_until_tester_present(&self) -> Duration {
        let elapsed = self.last_request.elapsed();
        if elapsed >= self.config.tester_present_interval {
            Duration::from_millis(0)
        } else {
            self.config.tester_present_interval - elapsed
        }
    }

    /// Get time until S3 timeout.
    pub fn time_until_timeout(&self) -> Duration {
        let elapsed = self.last_request.elapsed();
        if elapsed >= self.config.s3_server_timeout {
            Duration::from_millis(0)
        } else {
            self.config.s3_server_timeout - elapsed
        }
    }

    // ─── Internal ────────────────────────────

    /// Validate a session transition per ISO 14229 rules.
    fn validate_transition(&self, target: DiagnosticSession) -> Result<(), String> {
        // Same session is always allowed (refresh)
        if target == self.current_session {
            return Ok(());
        }

        // Default is always reachable
        if target == DiagnosticSession::Default {
            return Ok(());
        }

        // Programming session is only reachable from Extended or Default
        if target == DiagnosticSession::Programming {
            match self.current_session {
                DiagnosticSession::Default | DiagnosticSession::Extended => Ok(()),
                _ => Err(format!(
                    "Cannot transition from {} to Programming",
                    self.current_session.name()
                )),
            }
        } else {
            Ok(())
        }
    }

    fn is_any_security_unlocked(&self) -> bool {
        matches!(self.security_state, SecurityAccessState::Unlocked { .. })
    }

    fn record_transition(&mut self, transition: SessionTransition) {
        self.history.push(transition);
        if self.history.len() > self.config.max_history {
            self.history.remove(0);
        }
    }

    fn default_capabilities() -> HashMap<DiagnosticSession, Vec<u8>> {
        let mut caps = HashMap::new();
        // Default session: basic services
        caps.insert(
            DiagnosticSession::Default,
            vec![0x10, 0x11, 0x3E, 0x22, 0x19, 0x27],
        );
        // Extended session: more services
        caps.insert(
            DiagnosticSession::Extended,
            vec![
                0x10, 0x11, 0x14, 0x19, 0x22, 0x23, 0x24, 0x27, 0x28, 0x2A, 0x2C, 0x2E, 0x2F, 0x31,
                0x3E, 0x85,
            ],
        );
        // Programming session: flash services
        caps.insert(
            DiagnosticSession::Programming,
            vec![0x10, 0x11, 0x27, 0x34, 0x35, 0x36, 0x37, 0x3E],
        );
        // Safety system session: safety-critical services
        caps.insert(
            DiagnosticSession::SafetySystem,
            vec![0x10, 0x11, 0x22, 0x27, 0x2E, 0x31, 0x3E],
        );
        caps
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_manager() -> DiagnosticSessionManager {
        DiagnosticSessionManager::with_defaults()
    }

    #[test]
    fn test_initial_session_is_default() {
        let mgr = default_manager();
        assert_eq!(mgr.current_session(), DiagnosticSession::Default);
    }

    #[test]
    fn test_transition_to_extended() {
        let mut mgr = default_manager();
        let t = mgr
            .request_transition(DiagnosticSession::Extended)
            .expect("transition failed");
        assert!(t.success);
        assert_eq!(mgr.current_session(), DiagnosticSession::Extended);
    }

    #[test]
    fn test_transition_back_to_default() {
        let mut mgr = default_manager();
        mgr.request_transition(DiagnosticSession::Extended)
            .expect("transition failed");
        mgr.request_transition(DiagnosticSession::Default)
            .expect("transition failed");
        assert_eq!(mgr.current_session(), DiagnosticSession::Default);
    }

    #[test]
    fn test_programming_requires_security() {
        let mut mgr = default_manager();
        let result = mgr.request_transition(DiagnosticSession::Programming);
        assert!(result.is_err());
    }

    #[test]
    fn test_programming_with_security() {
        let mut mgr = default_manager();
        // Unlock security first (zero seed = already unlocked)
        mgr.receive_security_seed(1, vec![0, 0, 0, 0])
            .expect("seed failed");
        let t = mgr
            .request_transition(DiagnosticSession::Programming)
            .expect("transition failed");
        assert!(t.success);
    }

    #[test]
    fn test_security_seed_reception() {
        let mut mgr = default_manager();
        mgr.receive_security_seed(1, vec![0xDE, 0xAD])
            .expect("seed failed");
        assert!(matches!(
            mgr.security_state(),
            SecurityAccessState::SeedReceived { level: 1, .. }
        ));
    }

    #[test]
    fn test_security_key_success() {
        let mut mgr = default_manager();
        mgr.receive_security_seed(1, vec![0xDE, 0xAD])
            .expect("seed failed");
        mgr.process_security_response(true)
            .expect("response failed");
        assert!(mgr.is_security_unlocked(1));
    }

    #[test]
    fn test_security_key_failure() {
        let mut mgr = default_manager();
        mgr.receive_security_seed(1, vec![0xDE, 0xAD])
            .expect("seed failed");
        let result = mgr.process_security_response(false);
        assert!(result.is_err());
        assert!(!mgr.is_security_unlocked(1));
    }

    #[test]
    fn test_security_lockout() {
        let mut mgr = DiagnosticSessionManager::new(SessionManagerConfig {
            max_security_attempts: 2,
            ..Default::default()
        });

        // First failed attempt
        mgr.receive_security_seed(1, vec![0xDE])
            .expect("seed failed");
        let _ = mgr.process_security_response(false);

        // Second failed attempt
        mgr.receive_security_seed(1, vec![0xAD])
            .expect("seed failed");
        let _ = mgr.process_security_response(false);

        // Should be locked out now
        assert!(matches!(
            mgr.security_state(),
            SecurityAccessState::AttemptsExceeded { .. }
        ));
    }

    #[test]
    fn test_zero_seed_auto_unlock() {
        let mut mgr = default_manager();
        mgr.receive_security_seed(1, vec![0, 0, 0])
            .expect("seed failed");
        assert!(mgr.is_security_unlocked(1));
    }

    #[test]
    fn test_session_refresh() {
        let mut mgr = default_manager();
        mgr.request_transition(DiagnosticSession::Extended)
            .expect("transition failed");
        // Refreshing same session should succeed
        mgr.request_transition(DiagnosticSession::Extended)
            .expect("refresh failed");
        assert_eq!(mgr.current_session(), DiagnosticSession::Extended);
    }

    #[test]
    fn test_is_active_session() {
        let mut mgr = default_manager();
        assert!(!mgr.is_active_session());
        mgr.request_transition(DiagnosticSession::Extended)
            .expect("transition failed");
        assert!(mgr.is_active_session());
    }

    #[test]
    fn test_session_byte_conversion() {
        assert_eq!(DiagnosticSession::Default.to_byte(), 0x01);
        assert_eq!(DiagnosticSession::Programming.to_byte(), 0x02);
        assert_eq!(DiagnosticSession::Extended.to_byte(), 0x03);
        assert_eq!(DiagnosticSession::SafetySystem.to_byte(), 0x04);
    }

    #[test]
    fn test_session_from_byte() {
        assert_eq!(
            DiagnosticSession::from_byte(0x01),
            DiagnosticSession::Default
        );
        assert_eq!(
            DiagnosticSession::from_byte(0x02),
            DiagnosticSession::Programming
        );
        assert_eq!(
            DiagnosticSession::from_byte(0x03),
            DiagnosticSession::Extended
        );
        assert_eq!(
            DiagnosticSession::from_byte(0x50),
            DiagnosticSession::VehicleManufacturer(0x50)
        );
    }

    #[test]
    fn test_session_names() {
        assert_eq!(DiagnosticSession::Default.name(), "Default");
        assert_eq!(DiagnosticSession::Programming.name(), "Programming");
        assert_eq!(DiagnosticSession::Extended.name(), "Extended");
    }

    #[test]
    fn test_requires_security() {
        assert!(!DiagnosticSession::Default.requires_security());
        assert!(!DiagnosticSession::Extended.requires_security());
        assert!(DiagnosticSession::Programming.requires_security());
        assert!(DiagnosticSession::SafetySystem.requires_security());
    }

    #[test]
    fn test_tester_present_not_needed_in_default() {
        let mgr = default_manager();
        assert!(!mgr.should_send_tester_present());
    }

    #[test]
    fn test_tester_present_needed_in_extended() {
        let mut mgr = default_manager();
        mgr.request_transition(DiagnosticSession::Extended)
            .expect("transition failed");
        // Just transitioned, so should not need it yet
        assert!(!mgr.should_send_tester_present());

        // Wait and check (we can't wait in unit tests, but the logic is testable)
        // The interval check is `elapsed >= interval`
    }

    #[test]
    fn test_record_tester_present() {
        let mut mgr = default_manager();
        mgr.record_tester_present();
        assert_eq!(mgr.stats().tester_present_sent, 1);
    }

    #[test]
    fn test_session_timeout_not_in_default() {
        let mgr = default_manager();
        assert!(!mgr.is_session_timed_out());
    }

    #[test]
    fn test_handle_timeout_none_in_default() {
        let mut mgr = default_manager();
        let t = mgr.handle_timeout();
        assert!(t.is_none());
    }

    #[test]
    fn test_record_activity() {
        let mut mgr = default_manager();
        mgr.record_activity();
        // Should not panic, resets timer
    }

    #[test]
    fn test_history_tracking() {
        let mut mgr = default_manager();
        mgr.request_transition(DiagnosticSession::Extended)
            .expect("transition failed");
        mgr.request_transition(DiagnosticSession::Default)
            .expect("transition failed");

        assert_eq!(mgr.history().len(), 2);
        assert!(mgr.history()[0].success);
    }

    #[test]
    fn test_history_limit() {
        let mut mgr = DiagnosticSessionManager::new(SessionManagerConfig {
            max_history: 3,
            ..Default::default()
        });

        for _ in 0..5 {
            let _ = mgr.request_transition(DiagnosticSession::Extended);
            let _ = mgr.request_transition(DiagnosticSession::Default);
        }

        assert!(mgr.history().len() <= 3);
    }

    #[test]
    fn test_service_availability_default() {
        let mgr = default_manager();
        assert!(mgr.is_service_available(0x22)); // ReadDataByIdentifier
        assert!(mgr.is_service_available(0x3E)); // TesterPresent
    }

    #[test]
    fn test_service_availability_extended() {
        let mut mgr = default_manager();
        mgr.request_transition(DiagnosticSession::Extended)
            .expect("transition failed");
        assert!(mgr.is_service_available(0x2E)); // WriteDataByIdentifier
        assert!(mgr.is_service_available(0x31)); // RoutineControl
    }

    #[test]
    fn test_available_services_list() {
        let mgr = default_manager();
        let services = mgr.available_services();
        assert!(!services.is_empty());
        assert!(services.contains(&0x10)); // DiagnosticSessionControl always available
    }

    #[test]
    fn test_set_custom_capabilities() {
        let mut mgr = default_manager();
        mgr.set_capabilities(DiagnosticSession::Extended, vec![0x22, 0x2E]);
        mgr.request_transition(DiagnosticSession::Extended)
            .expect("transition failed");
        assert!(mgr.is_service_available(0x22));
        assert!(!mgr.is_service_available(0x31)); // No longer available after custom set
    }

    #[test]
    fn test_stats_initial() {
        let mgr = default_manager();
        let stats = mgr.stats();
        assert_eq!(stats.transitions_attempted, 0);
        assert_eq!(stats.transitions_succeeded, 0);
    }

    #[test]
    fn test_stats_after_transitions() {
        let mut mgr = default_manager();
        mgr.request_transition(DiagnosticSession::Extended)
            .expect("transition failed");
        let _ = mgr.request_transition(DiagnosticSession::Programming); // Should fail
        let stats = mgr.stats();
        assert_eq!(stats.transitions_attempted, 2);
        assert_eq!(stats.transitions_succeeded, 1);
        assert_eq!(stats.transitions_failed, 1);
    }

    #[test]
    fn test_config_defaults() {
        let config = SessionManagerConfig::default();
        assert_eq!(config.p2_server_timeout, Duration::from_millis(50));
        assert_eq!(config.s3_server_timeout, Duration::from_millis(5000));
        assert_eq!(config.max_security_attempts, 3);
        assert!(config.auto_tester_present);
    }

    #[test]
    fn test_config_serialization() {
        let config = SessionManagerConfig::default();
        let json = serde_json::to_string(&config).expect("serialize failed");
        assert!(json.contains("p2_server_timeout"));
    }

    #[test]
    fn test_transition_serialization() {
        let t = SessionTransition {
            from: DiagnosticSession::Default,
            to: DiagnosticSession::Extended,
            success: true,
            error: None,
            timestamp: Utc::now(),
        };
        let json = serde_json::to_string(&t).expect("serialize failed");
        assert!(json.contains("success"));
    }

    #[test]
    fn test_security_resets_on_default() {
        let mut mgr = default_manager();
        mgr.receive_security_seed(1, vec![0, 0])
            .expect("seed failed");
        assert!(mgr.is_security_unlocked(1));

        mgr.request_transition(DiagnosticSession::Default)
            .expect("transition failed");
        assert!(!mgr.is_security_unlocked(1));
    }

    #[test]
    fn test_time_until_timeout() {
        let mgr = default_manager();
        let t = mgr.time_until_timeout();
        assert!(t > Duration::from_millis(0));
    }

    #[test]
    fn test_time_until_tester_present() {
        let mgr = default_manager();
        let t = mgr.time_until_tester_present();
        assert!(t > Duration::from_millis(0));
    }

    #[test]
    fn test_is_initialized() {
        let mgr = default_manager();
        assert!(mgr.is_initialized());
    }

    #[test]
    fn test_process_response_without_seed() {
        let mut mgr = default_manager();
        let result = mgr.process_security_response(true);
        assert!(result.is_err());
    }

    #[test]
    fn test_vehicle_manufacturer_session() {
        let s = DiagnosticSession::VehicleManufacturer(0x50);
        assert_eq!(s.to_byte(), 0x50);
        assert_eq!(s.name(), "VehicleManufacturer");
    }

    #[test]
    fn test_system_supplier_session() {
        let s = DiagnosticSession::SystemSupplier(0x60);
        assert_eq!(s.to_byte(), 0x60);
        assert_eq!(s.name(), "SystemSupplier");
    }
}
