//! CAN bus diagnostic monitoring (OBD-II / ISO 15765-2 simplified).
//!
//! Tracks active diagnostic codes, monitors PID thresholds, and decodes
//! common OBD-II PIDs from raw CAN payload bytes.

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// Severity level of a diagnostic code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagSeverity {
    Info,
    Warning,
    Critical,
}

/// An active diagnostic trouble code.
#[derive(Debug, Clone)]
pub struct DiagnosticCode {
    pub code: String,
    pub description: String,
    pub severity: DiagSeverity,
    pub first_seen_ms: u64,
    pub last_seen_ms: u64,
    pub occurrence_count: u32,
}

/// A raw OBD-II / ISO 15765-2 CAN frame for diagnostics.
#[derive(Debug, Clone)]
pub struct DiagnosticFrame {
    pub timestamp_ms: u64,
    pub service_id: u8,
    pub pid: u8,
    pub data: Vec<u8>,
}

/// A threshold for a monitored PID value.
#[derive(Debug, Clone)]
pub struct MonitorThreshold {
    pub pid: u8,
    pub min_value: f64,
    pub max_value: f64,
    pub description: String,
}

/// CAN bus diagnostic monitor.
pub struct DiagnosticMonitor {
    active_codes: HashMap<String, DiagnosticCode>,
    thresholds: Vec<MonitorThreshold>,
    processed_frames: u64,
}

impl Default for DiagnosticMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl DiagnosticMonitor {
    /// Create a new monitor with no thresholds or active codes.
    pub fn new() -> Self {
        Self {
            active_codes: HashMap::new(),
            thresholds: Vec::new(),
            processed_frames: 0,
        }
    }

    /// Register a monitoring threshold for a PID.
    pub fn add_threshold(&mut self, threshold: MonitorThreshold) {
        self.thresholds.push(threshold);
    }

    /// Process an incoming diagnostic frame.
    ///
    /// Increments the processed frame counter and, if a decoded value
    /// violates a registered threshold, raises or refreshes the corresponding
    /// diagnostic code.
    pub fn process_frame(&mut self, frame: &DiagnosticFrame) {
        self.processed_frames += 1;

        if let Some(value) = Self::decode_pid_value(frame.pid, &frame.data) {
            // Clone thresholds to avoid borrow conflict with self.active_codes
            let violated: Vec<(String, String, DiagSeverity)> = self
                .thresholds
                .iter()
                .filter(|t| t.pid == frame.pid && (value < t.min_value || value > t.max_value))
                .map(|t| {
                    let code = format!("P{:04X}", (frame.pid as u16) << 4 | 0x1);
                    let desc = format!(
                        "PID 0x{:02X} value {:.2} out of range [{:.2}, {:.2}]: {}",
                        frame.pid, value, t.min_value, t.max_value, t.description
                    );
                    let sev = if value < t.min_value * 0.5 || value > t.max_value * 2.0 {
                        DiagSeverity::Critical
                    } else {
                        DiagSeverity::Warning
                    };
                    (code, desc, sev)
                })
                .collect();

            for (code, desc, sev) in violated {
                let entry =
                    self.active_codes
                        .entry(code.clone())
                        .or_insert_with(|| DiagnosticCode {
                            code: code.clone(),
                            description: desc.clone(),
                            severity: sev,
                            first_seen_ms: frame.timestamp_ms,
                            last_seen_ms: frame.timestamp_ms,
                            occurrence_count: 0,
                        });
                entry.last_seen_ms = frame.timestamp_ms;
                entry.occurrence_count += 1;
            }
        }
    }

    /// Return references to all currently active diagnostic codes.
    pub fn active_codes(&self) -> Vec<&DiagnosticCode> {
        self.active_codes.values().collect()
    }

    /// Return codes matching the given severity level.
    pub fn codes_by_severity(&self, severity: DiagSeverity) -> Vec<&DiagnosticCode> {
        self.active_codes
            .values()
            .filter(|c| c.severity == severity)
            .collect()
    }

    /// Remove the code with the given identifier.
    ///
    /// Returns `true` if a code was removed, `false` if it was not present.
    pub fn clear_code(&mut self, code: &str) -> bool {
        self.active_codes.remove(code).is_some()
    }

    /// Clear all active diagnostic codes.
    pub fn clear_all(&mut self) {
        self.active_codes.clear();
    }

    /// Total number of frames processed since construction.
    pub fn processed_count(&self) -> u64 {
        self.processed_frames
    }

    /// Decode a common OBD-II PID value from raw data bytes.
    ///
    /// Supported PIDs:
    /// - `0x05` Engine coolant temperature: A - 40 (°C)
    /// - `0x0C` Engine RPM: (256*A + B) / 4
    /// - `0x0D` Vehicle speed: A (km/h)
    /// - `0x11` Throttle position: A * 100 / 255 (%)
    pub fn decode_pid_value(pid: u8, data: &[u8]) -> Option<f64> {
        match pid {
            // Engine coolant temperature: A - 40
            0x05 => data.first().map(|&a| a as f64 - 40.0),
            // Engine RPM: (256 * A + B) / 4
            0x0C => {
                if data.len() >= 2 {
                    Some((256.0 * data[0] as f64 + data[1] as f64) / 4.0)
                } else {
                    None
                }
            }
            // Vehicle speed
            0x0D => data.first().map(|&a| a as f64),
            // Throttle position: A * 100 / 255
            0x11 => data.first().map(|&a| a as f64 * 100.0 / 255.0),
            _ => None,
        }
    }

    /// Check whether the given decoded PID `value` violates any threshold
    /// and raise/refresh codes accordingly.  Returns references to all
    /// matching (newly raised or updated) codes.
    pub fn check_thresholds(
        &mut self,
        pid: u8,
        value: f64,
        timestamp_ms: u64,
    ) -> Vec<&DiagnosticCode> {
        let violated: Vec<(String, String, DiagSeverity)> = self
            .thresholds
            .iter()
            .filter(|t| t.pid == pid && (value < t.min_value || value > t.max_value))
            .map(|t| {
                let code = format!("T{:02X}_{}", pid, t.description.replace(' ', "_"));
                let desc = format!(
                    "PID 0x{:02X} = {:.2} violates threshold [{:.2}, {:.2}]",
                    pid, value, t.min_value, t.max_value
                );
                let sev = if value < t.min_value * 0.5 || value > t.max_value * 2.0 {
                    DiagSeverity::Critical
                } else {
                    DiagSeverity::Warning
                };
                (code, desc, sev)
            })
            .collect();

        let mut raised_codes = Vec::new();
        for (code, desc, sev) in violated {
            let entry = self
                .active_codes
                .entry(code.clone())
                .or_insert_with(|| DiagnosticCode {
                    code: code.clone(),
                    description: desc.clone(),
                    severity: sev,
                    first_seen_ms: timestamp_ms,
                    last_seen_ms: timestamp_ms,
                    occurrence_count: 0,
                });
            entry.last_seen_ms = timestamp_ms;
            entry.occurrence_count += 1;
            raised_codes.push(code.clone());
        }

        // Collect references in a second pass (to avoid borrowing conflicts)
        raised_codes
            .iter()
            .filter_map(|k| self.active_codes.get(k.as_str()))
            .collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rpm_frame(rpm_raw_a: u8, rpm_raw_b: u8, ts: u64) -> DiagnosticFrame {
        DiagnosticFrame {
            timestamp_ms: ts,
            service_id: 0x41,
            pid: 0x0C,
            data: vec![rpm_raw_a, rpm_raw_b],
        }
    }

    fn speed_frame(speed_km_h: u8, ts: u64) -> DiagnosticFrame {
        DiagnosticFrame {
            timestamp_ms: ts,
            service_id: 0x41,
            pid: 0x0D,
            data: vec![speed_km_h],
        }
    }

    fn coolant_frame(raw: u8, ts: u64) -> DiagnosticFrame {
        DiagnosticFrame {
            timestamp_ms: ts,
            service_id: 0x41,
            pid: 0x05,
            data: vec![raw],
        }
    }

    fn throttle_frame(raw: u8, ts: u64) -> DiagnosticFrame {
        DiagnosticFrame {
            timestamp_ms: ts,
            service_id: 0x41,
            pid: 0x11,
            data: vec![raw],
        }
    }

    // ── add_threshold ────────────────────────────────────────────────────────

    #[test]
    fn test_add_threshold_registers() {
        let mut m = DiagnosticMonitor::new();
        m.add_threshold(MonitorThreshold {
            pid: 0x0D,
            min_value: 0.0,
            max_value: 250.0,
            description: "speed".to_string(),
        });
        assert_eq!(m.thresholds.len(), 1);
    }

    #[test]
    fn test_add_multiple_thresholds() {
        let mut m = DiagnosticMonitor::new();
        for pid in [0x05, 0x0C, 0x0D, 0x11] {
            m.add_threshold(MonitorThreshold {
                pid,
                min_value: 0.0,
                max_value: 100.0,
                description: "test".to_string(),
            });
        }
        assert_eq!(m.thresholds.len(), 4);
    }

    // ── process_frame ────────────────────────────────────────────────────────

    #[test]
    fn test_process_frame_increments_count() {
        let mut m = DiagnosticMonitor::new();
        m.process_frame(&speed_frame(50, 0));
        assert_eq!(m.processed_count(), 1);
        m.process_frame(&speed_frame(60, 1));
        assert_eq!(m.processed_count(), 2);
    }

    #[test]
    fn test_process_frame_no_threshold_no_codes() {
        let mut m = DiagnosticMonitor::new();
        m.process_frame(&speed_frame(100, 0));
        assert!(m.active_codes().is_empty());
    }

    #[test]
    fn test_process_frame_within_threshold_no_code() {
        let mut m = DiagnosticMonitor::new();
        m.add_threshold(MonitorThreshold {
            pid: 0x0D,
            min_value: 0.0,
            max_value: 200.0,
            description: "speed ok".to_string(),
        });
        m.process_frame(&speed_frame(100, 0));
        assert!(m.active_codes().is_empty());
    }

    #[test]
    fn test_process_frame_violates_threshold_raises_code() {
        let mut m = DiagnosticMonitor::new();
        m.add_threshold(MonitorThreshold {
            pid: 0x0D,
            min_value: 0.0,
            max_value: 50.0,
            description: "speed limit".to_string(),
        });
        m.process_frame(&speed_frame(200, 100));
        let codes = m.active_codes();
        assert_eq!(codes.len(), 1);
    }

    // ── active_codes ─────────────────────────────────────────────────────────

    #[test]
    fn test_active_codes_empty_initially() {
        let m = DiagnosticMonitor::new();
        assert!(m.active_codes().is_empty());
    }

    #[test]
    fn test_active_codes_after_violation() {
        let mut m = DiagnosticMonitor::new();
        m.add_threshold(MonitorThreshold {
            pid: 0x0C,
            min_value: 0.0,
            max_value: 6000.0,
            description: "rpm".to_string(),
        });
        // A=0xFF B=0xFF → (256*255+255)/4 = 16383.75 > 6000
        m.process_frame(&rpm_frame(0xFF, 0xFF, 0));
        assert!(!m.active_codes().is_empty());
    }

    // ── clear_code ───────────────────────────────────────────────────────────

    #[test]
    fn test_clear_code_removes_entry() {
        let mut m = DiagnosticMonitor::new();
        m.add_threshold(MonitorThreshold {
            pid: 0x0D,
            min_value: 0.0,
            max_value: 10.0,
            description: "slow".to_string(),
        });
        m.process_frame(&speed_frame(200, 0));
        let code_id = m.active_codes()[0].code.clone();
        assert!(m.clear_code(&code_id));
        assert!(m.active_codes().is_empty());
    }

    #[test]
    fn test_clear_code_missing_returns_false() {
        let mut m = DiagnosticMonitor::new();
        assert!(!m.clear_code("P9999"));
    }

    // ── clear_all ────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_all_removes_everything() {
        let mut m = DiagnosticMonitor::new();
        m.add_threshold(MonitorThreshold {
            pid: 0x0D,
            min_value: 0.0,
            max_value: 1.0,
            description: "d".to_string(),
        });
        m.process_frame(&speed_frame(100, 0));
        m.process_frame(&speed_frame(200, 1));
        m.clear_all();
        assert!(m.active_codes().is_empty());
    }

    // ── codes_by_severity ────────────────────────────────────────────────────

    #[test]
    fn test_codes_by_severity_empty() {
        let m = DiagnosticMonitor::new();
        assert!(m.codes_by_severity(DiagSeverity::Warning).is_empty());
    }

    #[test]
    fn test_codes_by_severity_filters_correctly() {
        let mut m = DiagnosticMonitor::new();
        // Manually insert a Warning code
        m.active_codes.insert(
            "WARN1".to_string(),
            DiagnosticCode {
                code: "WARN1".to_string(),
                description: "warning".to_string(),
                severity: DiagSeverity::Warning,
                first_seen_ms: 0,
                last_seen_ms: 0,
                occurrence_count: 1,
            },
        );
        m.active_codes.insert(
            "CRIT1".to_string(),
            DiagnosticCode {
                code: "CRIT1".to_string(),
                description: "critical".to_string(),
                severity: DiagSeverity::Critical,
                first_seen_ms: 0,
                last_seen_ms: 0,
                occurrence_count: 1,
            },
        );
        assert_eq!(m.codes_by_severity(DiagSeverity::Warning).len(), 1);
        assert_eq!(m.codes_by_severity(DiagSeverity::Critical).len(), 1);
        assert!(m.codes_by_severity(DiagSeverity::Info).is_empty());
    }

    // ── decode_pid_value ─────────────────────────────────────────────────────

    #[test]
    fn test_decode_pid_rpm() {
        // (256*0 + 200) / 4 = 50 RPM
        let v = DiagnosticMonitor::decode_pid_value(0x0C, &[0, 200]).expect("should succeed");
        assert!((v - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_decode_pid_rpm_high() {
        // (256*30 + 0) / 4 = 1920 RPM
        let v = DiagnosticMonitor::decode_pid_value(0x0C, &[30, 0]).expect("should succeed");
        assert!((v - 1920.0).abs() < 0.01);
    }

    #[test]
    fn test_decode_pid_speed() {
        let v = DiagnosticMonitor::decode_pid_value(0x0D, &[120]).expect("should succeed");
        assert!((v - 120.0).abs() < 0.01);
    }

    #[test]
    fn test_decode_pid_coolant() {
        // A = 100 → 100 - 40 = 60 °C
        let v = DiagnosticMonitor::decode_pid_value(0x05, &[100]).expect("should succeed");
        assert!((v - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_decode_pid_coolant_min() {
        // A = 0 → 0 - 40 = -40 °C
        let v = DiagnosticMonitor::decode_pid_value(0x05, &[0]).expect("should succeed");
        assert!((v - (-40.0)).abs() < 0.01);
    }

    #[test]
    fn test_decode_pid_throttle() {
        // A = 255 → 100%
        let v = DiagnosticMonitor::decode_pid_value(0x11, &[255]).expect("should succeed");
        assert!((v - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_decode_pid_throttle_half() {
        // A = 127 → ~49.8%
        let v = DiagnosticMonitor::decode_pid_value(0x11, &[127]).expect("should succeed");
        assert!(v > 49.0 && v < 51.0);
    }

    #[test]
    fn test_decode_pid_unknown_returns_none() {
        let v = DiagnosticMonitor::decode_pid_value(0xFF, &[0, 0]);
        assert!(v.is_none());
    }

    #[test]
    fn test_decode_pid_rpm_short_data_returns_none() {
        let v = DiagnosticMonitor::decode_pid_value(0x0C, &[10]); // needs 2 bytes
        assert!(v.is_none());
    }

    #[test]
    fn test_decode_pid_speed_empty_returns_none() {
        let v = DiagnosticMonitor::decode_pid_value(0x0D, &[]);
        assert!(v.is_none());
    }

    // ── check_thresholds ─────────────────────────────────────────────────────

    #[test]
    fn test_check_thresholds_raises_code_on_violation() {
        let mut m = DiagnosticMonitor::new();
        m.add_threshold(MonitorThreshold {
            pid: 0x0D,
            min_value: 0.0,
            max_value: 120.0,
            description: "speed".to_string(),
        });
        let codes = m.check_thresholds(0x0D, 200.0, 500);
        assert_eq!(codes.len(), 1);
        assert!(!m.active_codes().is_empty());
    }

    #[test]
    fn test_check_thresholds_no_violation_no_codes() {
        let mut m = DiagnosticMonitor::new();
        m.add_threshold(MonitorThreshold {
            pid: 0x0D,
            min_value: 0.0,
            max_value: 200.0,
            description: "speed".to_string(),
        });
        let codes = m.check_thresholds(0x0D, 100.0, 0);
        assert!(codes.is_empty());
    }

    #[test]
    fn test_occurrence_count_increments_on_repeated_violation() {
        let mut m = DiagnosticMonitor::new();
        m.add_threshold(MonitorThreshold {
            pid: 0x0D,
            min_value: 0.0,
            max_value: 10.0,
            description: "speed".to_string(),
        });
        m.check_thresholds(0x0D, 200.0, 100);
        m.check_thresholds(0x0D, 200.0, 200);
        m.check_thresholds(0x0D, 200.0, 300);
        let codes = m.active_codes();
        assert_eq!(codes.len(), 1);
        assert!(codes[0].occurrence_count >= 3);
    }

    #[test]
    fn test_process_frame_occurrence_count() {
        let mut m = DiagnosticMonitor::new();
        m.add_threshold(MonitorThreshold {
            pid: 0x0C,
            min_value: 0.0,
            max_value: 6000.0,
            description: "rpm".to_string(),
        });
        m.process_frame(&rpm_frame(0xFF, 0xFF, 10));
        m.process_frame(&rpm_frame(0xFF, 0xFF, 20));
        let codes = m.active_codes();
        assert_eq!(codes.len(), 1);
        assert!(codes[0].occurrence_count >= 2);
    }

    #[test]
    fn test_processed_count_increases_regardless_of_threshold() {
        let mut m = DiagnosticMonitor::new();
        for i in 0..10u64 {
            m.process_frame(&coolant_frame(80, i));
        }
        assert_eq!(m.processed_count(), 10);
    }

    #[test]
    fn test_default_monitor() {
        let m = DiagnosticMonitor::default();
        assert_eq!(m.processed_count(), 0);
        assert!(m.active_codes().is_empty());
    }

    #[test]
    fn test_check_thresholds_returns_code_refs() {
        let mut m = DiagnosticMonitor::new();
        m.add_threshold(MonitorThreshold {
            pid: 0x0C,
            min_value: 500.0,
            max_value: 6000.0,
            description: "rpm range".to_string(),
        });
        // RPM below min
        let codes = m.check_thresholds(0x0C, 100.0, 0);
        assert!(!codes.is_empty());
        assert!(!codes[0].code.is_empty());
    }

    #[test]
    fn test_throttle_frame_decode() {
        let f = throttle_frame(128, 0);
        let v = DiagnosticMonitor::decode_pid_value(f.pid, &f.data).expect("should succeed");
        assert!(v > 40.0 && v < 60.0);
    }

    // ── additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_multiple_thresholds_same_pid() {
        let mut m = DiagnosticMonitor::new();
        m.add_threshold(MonitorThreshold {
            pid: 0x0D,
            min_value: 0.0,
            max_value: 100.0,
            description: "speed-low".to_string(),
        });
        m.add_threshold(MonitorThreshold {
            pid: 0x0D,
            min_value: 0.0,
            max_value: 50.0,
            description: "speed-strict".to_string(),
        });
        assert_eq!(m.thresholds.len(), 2);
    }

    #[test]
    fn test_process_frame_multiple_times_same_pid() {
        let mut m = DiagnosticMonitor::new();
        m.add_threshold(MonitorThreshold {
            pid: 0x0D,
            min_value: 0.0,
            max_value: 10.0,
            description: "speed".to_string(),
        });
        for i in 0..5 {
            m.process_frame(&speed_frame(200, i));
        }
        assert_eq!(m.processed_count(), 5);
    }

    #[test]
    fn test_decode_pid_rpm_zero() {
        // A=0, B=0 → (0+0)/4 = 0
        let v = DiagnosticMonitor::decode_pid_value(0x0C, &[0, 0]).expect("should succeed");
        assert!((v - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_decode_pid_coolant_max() {
        // A=255 → 255-40 = 215 °C
        let v = DiagnosticMonitor::decode_pid_value(0x05, &[255]).expect("should succeed");
        assert!((v - 215.0).abs() < 0.01);
    }

    #[test]
    fn test_clear_all_clears_all_codes() {
        let mut m = DiagnosticMonitor::new();
        for i in 0..3 {
            m.active_codes.insert(
                format!("CODE{}", i),
                DiagnosticCode {
                    code: format!("CODE{}", i),
                    description: "test".to_string(),
                    severity: DiagSeverity::Warning,
                    first_seen_ms: 0,
                    last_seen_ms: 0,
                    occurrence_count: 1,
                },
            );
        }
        assert_eq!(m.active_codes().len(), 3);
        m.clear_all();
        assert!(m.active_codes().is_empty());
    }

    #[test]
    fn test_info_severity_code() {
        let mut m = DiagnosticMonitor::new();
        m.active_codes.insert(
            "INFO1".to_string(),
            DiagnosticCode {
                code: "INFO1".to_string(),
                description: "info".to_string(),
                severity: DiagSeverity::Info,
                first_seen_ms: 0,
                last_seen_ms: 0,
                occurrence_count: 1,
            },
        );
        assert_eq!(m.codes_by_severity(DiagSeverity::Info).len(), 1);
    }

    #[test]
    fn test_clear_nonexistent_code_returns_false() {
        let mut m = DiagnosticMonitor::new();
        assert!(!m.clear_code("NOTEXIST"));
    }

    #[test]
    fn test_check_thresholds_multiple_thresholds_same_pid() {
        let mut m = DiagnosticMonitor::new();
        m.add_threshold(MonitorThreshold {
            pid: 0x0D,
            min_value: 0.0,
            max_value: 100.0,
            description: "t1".to_string(),
        });
        m.add_threshold(MonitorThreshold {
            pid: 0x0D,
            min_value: 0.0,
            max_value: 50.0,
            description: "t2".to_string(),
        });
        // Both thresholds violated
        let codes = m.check_thresholds(0x0D, 200.0, 0);
        assert_eq!(codes.len(), 2);
    }

    #[test]
    fn test_processed_count_starts_at_zero() {
        let m = DiagnosticMonitor::new();
        assert_eq!(m.processed_count(), 0);
    }

    #[test]
    fn test_decode_pid_throttle_zero() {
        let v = DiagnosticMonitor::decode_pid_value(0x11, &[0]).expect("should succeed");
        assert!(v.abs() < 0.01);
    }

    #[test]
    fn test_active_codes_returns_all() {
        let mut m = DiagnosticMonitor::new();
        for i in 0..5 {
            m.active_codes.insert(
                format!("C{}", i),
                DiagnosticCode {
                    code: format!("C{}", i),
                    description: "test".to_string(),
                    severity: DiagSeverity::Warning,
                    first_seen_ms: 0,
                    last_seen_ms: 0,
                    occurrence_count: 1,
                },
            );
        }
        assert_eq!(m.active_codes().len(), 5);
    }

    #[test]
    fn test_threshold_description_preserved() {
        let mut m = DiagnosticMonitor::new();
        let desc = "my custom description".to_string();
        m.add_threshold(MonitorThreshold {
            pid: 0x0D,
            min_value: 0.0,
            max_value: 100.0,
            description: desc.clone(),
        });
        assert_eq!(m.thresholds[0].description, desc);
    }

    #[test]
    fn test_diag_severity_eq() {
        assert_eq!(DiagSeverity::Info, DiagSeverity::Info);
        assert_eq!(DiagSeverity::Warning, DiagSeverity::Warning);
        assert_eq!(DiagSeverity::Critical, DiagSeverity::Critical);
        assert_ne!(DiagSeverity::Info, DiagSeverity::Critical);
    }
}
