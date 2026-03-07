//! Modbus device scanner and discovery.
//!
//! `DeviceScanner` simulates scanning a range of Modbus unit IDs and
//! recording which addresses respond. For testing without real hardware, use
//! `simulate_scan` with a list of responding unit IDs.

use std::collections::HashMap;

/// A candidate device detected during a scan.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceCandidate {
    pub address: u8,
    pub unit_id: u8,
    pub detected_at_ms: u64,
}

/// Information gathered about a specific Modbus device.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceInfo {
    pub unit_id: u8,
    pub holding_registers: u16,
    pub coil_count: u16,
    pub device_type: Option<String>,
}

impl DeviceInfo {
    /// Create a new `DeviceInfo`.
    pub fn new(
        unit_id: u8,
        holding_registers: u16,
        coil_count: u16,
        device_type: Option<&str>,
    ) -> Self {
        Self {
            unit_id,
            holding_registers,
            coil_count,
            device_type: device_type.map(String::from),
        }
    }
}

/// Configuration for a scan operation.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanConfig {
    pub start_address: u8,
    pub end_address: u8,
    pub timeout_ms: u64,
    pub retry_count: usize,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            start_address: 1,
            end_address: 247,
            timeout_ms: 200,
            retry_count: 1,
        }
    }
}

/// Result of a completed scan.
#[derive(Debug, Clone)]
pub struct ScanResult {
    pub found: Vec<DeviceCandidate>,
    pub total_scanned: usize,
    pub duration_ms: u64,
}

impl ScanResult {
    /// Number of devices found.
    pub fn found_count(&self) -> usize {
        self.found.len()
    }
}

/// Errors that can occur during scanning.
#[derive(Debug, Clone, PartialEq)]
pub enum ScanError {
    /// `start_address > end_address` or both are 0.
    InvalidRange,
    /// The operation exceeded the configured timeout.
    Timeout,
    /// The scan was aborted by the caller.
    Aborted,
}

impl std::fmt::Display for ScanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScanError::InvalidRange => write!(f, "invalid address range"),
            ScanError::Timeout => write!(f, "scan timed out"),
            ScanError::Aborted => write!(f, "scan aborted"),
        }
    }
}

/// Modbus device scanner.
pub struct DeviceScanner {
    config: ScanConfig,
    found_devices: Vec<DeviceInfo>,
    device_index: HashMap<u8, usize>,
}

impl DeviceScanner {
    /// Create a new scanner with the given configuration.
    pub fn new(config: ScanConfig) -> Self {
        Self {
            config,
            found_devices: Vec::new(),
            device_index: HashMap::new(),
        }
    }

    /// Simulate a scan: mark every unit ID in `responding_units` that falls
    /// within the configured address range as "found".
    ///
    /// Returns `Err(ScanError::InvalidRange)` if the config range is invalid.
    pub fn simulate_scan(
        &mut self,
        responding_units: &[u8],
        now_ms: u64,
    ) -> Result<ScanResult, ScanError> {
        if !Self::address_range_valid(&self.config) {
            return Err(ScanError::InvalidRange);
        }

        let start = self.config.start_address;
        let end = self.config.end_address;
        let total_scanned = (end as usize).saturating_sub(start as usize) + 1;

        let mut found = Vec::new();
        for &uid in responding_units {
            if uid >= start && uid <= end {
                found.push(DeviceCandidate {
                    address: uid,
                    unit_id: uid,
                    detected_at_ms: now_ms,
                });
            }
        }

        // Sort for deterministic output
        found.sort_by_key(|c| c.unit_id);

        // Simulate a duration proportional to range size
        let duration_ms = total_scanned as u64 * self.config.timeout_ms / 10;

        Ok(ScanResult {
            found,
            total_scanned,
            duration_ms,
        })
    }

    /// Manually add a known device.
    pub fn add_device(&mut self, info: DeviceInfo) {
        let uid = info.unit_id;
        if let Some(&idx) = self.device_index.get(&uid) {
            self.found_devices[idx] = info;
        } else {
            let idx = self.found_devices.len();
            self.device_index.insert(uid, idx);
            self.found_devices.push(info);
        }
    }

    /// Look up a stored device by its unit ID.
    pub fn get_device(&self, unit_id: u8) -> Option<&DeviceInfo> {
        self.device_index
            .get(&unit_id)
            .and_then(|&idx| self.found_devices.get(idx))
    }

    /// All stored devices.
    pub fn all_devices(&self) -> &[DeviceInfo] {
        &self.found_devices
    }

    /// Number of stored devices.
    pub fn device_count(&self) -> usize {
        self.found_devices.len()
    }

    /// Whether the scan configuration address range is valid.
    pub fn address_range_valid(config: &ScanConfig) -> bool {
        config.start_address <= config.end_address && config.end_address > 0
    }

    /// Borrow the current configuration.
    pub fn config(&self) -> &ScanConfig {
        &self.config
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_scanner() -> DeviceScanner {
        DeviceScanner::new(ScanConfig::default())
    }

    // --- ScanConfig / Default -------------------------------------------

    #[test]
    fn test_scan_config_default() {
        let cfg = ScanConfig::default();
        assert_eq!(cfg.start_address, 1);
        assert_eq!(cfg.end_address, 247);
        assert_eq!(cfg.timeout_ms, 200);
        assert_eq!(cfg.retry_count, 1);
    }

    #[test]
    fn test_address_range_valid_default() {
        assert!(DeviceScanner::address_range_valid(&ScanConfig::default()));
    }

    #[test]
    fn test_address_range_invalid_start_greater_end() {
        let cfg = ScanConfig {
            start_address: 10,
            end_address: 5,
            ..Default::default()
        };
        assert!(!DeviceScanner::address_range_valid(&cfg));
    }

    #[test]
    fn test_address_range_equal_endpoints() {
        let cfg = ScanConfig {
            start_address: 5,
            end_address: 5,
            ..Default::default()
        };
        assert!(DeviceScanner::address_range_valid(&cfg));
    }

    #[test]
    fn test_address_range_zero_end() {
        let cfg = ScanConfig {
            start_address: 0,
            end_address: 0,
            ..Default::default()
        };
        assert!(!DeviceScanner::address_range_valid(&cfg));
    }

    // --- DeviceScanner::new + basic ----------------------------------------

    #[test]
    fn test_scanner_new_empty() {
        let s = default_scanner();
        assert_eq!(s.device_count(), 0);
        assert!(s.all_devices().is_empty());
    }

    #[test]
    fn test_scanner_config_accessible() {
        let s = default_scanner();
        assert_eq!(s.config().start_address, 1);
    }

    // --- simulate_scan ---------------------------------------------------

    #[test]
    fn test_simulate_scan_no_responders() {
        let mut s = default_scanner();
        let result = s.simulate_scan(&[], 1000).expect("scan ok");
        assert_eq!(result.found_count(), 0);
        assert!(result.total_scanned > 0);
    }

    #[test]
    fn test_simulate_scan_some_responders() {
        let mut s = default_scanner();
        let result = s.simulate_scan(&[1, 5, 247], 1000).expect("scan ok");
        assert_eq!(result.found_count(), 3);
    }

    #[test]
    fn test_simulate_scan_out_of_range_ignored() {
        let cfg = ScanConfig {
            start_address: 10,
            end_address: 20,
            ..Default::default()
        };
        let mut s = DeviceScanner::new(cfg);
        let result = s.simulate_scan(&[1, 5, 10, 15, 25], 1000).expect("ok");
        // Only 10 and 15 are in [10, 20]
        assert_eq!(result.found_count(), 2);
    }

    #[test]
    fn test_simulate_scan_sorted_output() {
        let mut s = default_scanner();
        let result = s.simulate_scan(&[50, 10, 30], 1000).expect("ok");
        let ids: Vec<u8> = result.found.iter().map(|c| c.unit_id).collect();
        assert_eq!(ids, vec![10, 30, 50]);
    }

    #[test]
    fn test_simulate_scan_invalid_range_error() {
        let cfg = ScanConfig {
            start_address: 20,
            end_address: 10,
            ..Default::default()
        };
        let mut s = DeviceScanner::new(cfg);
        let err = s.simulate_scan(&[15], 1000).unwrap_err();
        assert_eq!(err, ScanError::InvalidRange);
    }

    #[test]
    fn test_simulate_scan_total_scanned() {
        let cfg = ScanConfig {
            start_address: 1,
            end_address: 10,
            ..Default::default()
        };
        let mut s = DeviceScanner::new(cfg);
        let result = s.simulate_scan(&[], 0).expect("ok");
        assert_eq!(result.total_scanned, 10);
    }

    #[test]
    fn test_simulate_scan_candidate_fields() {
        let mut s = default_scanner();
        let result = s.simulate_scan(&[7], 5000).expect("ok");
        let c = &result.found[0];
        assert_eq!(c.unit_id, 7);
        assert_eq!(c.address, 7);
        assert_eq!(c.detected_at_ms, 5000);
    }

    // --- DeviceInfo -------------------------------------------------------

    #[test]
    fn test_device_info_new() {
        let d = DeviceInfo::new(5, 100, 200, Some("PLC-X"));
        assert_eq!(d.unit_id, 5);
        assert_eq!(d.holding_registers, 100);
        assert_eq!(d.coil_count, 200);
        assert_eq!(d.device_type.as_deref(), Some("PLC-X"));
    }

    #[test]
    fn test_device_info_no_type() {
        let d = DeviceInfo::new(1, 0, 0, None);
        assert!(d.device_type.is_none());
    }

    // --- add_device / get_device -----------------------------------------

    #[test]
    fn test_add_and_get_device() {
        let mut s = default_scanner();
        s.add_device(DeviceInfo::new(3, 50, 10, Some("Sensor")));
        let d = s.get_device(3).expect("should exist");
        assert_eq!(d.unit_id, 3);
        assert_eq!(d.holding_registers, 50);
    }

    #[test]
    fn test_get_device_not_found() {
        let s = default_scanner();
        assert!(s.get_device(100).is_none());
    }

    #[test]
    fn test_add_multiple_devices() {
        let mut s = default_scanner();
        s.add_device(DeviceInfo::new(1, 10, 5, None));
        s.add_device(DeviceInfo::new(2, 20, 6, None));
        s.add_device(DeviceInfo::new(3, 30, 7, None));
        assert_eq!(s.device_count(), 3);
    }

    #[test]
    fn test_update_existing_device() {
        let mut s = default_scanner();
        s.add_device(DeviceInfo::new(5, 10, 5, Some("Old")));
        s.add_device(DeviceInfo::new(5, 99, 99, Some("New")));
        assert_eq!(s.device_count(), 1);
        let d = s.get_device(5).expect("exists");
        assert_eq!(d.holding_registers, 99);
        assert_eq!(d.device_type.as_deref(), Some("New"));
    }

    #[test]
    fn test_all_devices_returns_slice() {
        let mut s = default_scanner();
        s.add_device(DeviceInfo::new(1, 0, 0, None));
        s.add_device(DeviceInfo::new(2, 0, 0, None));
        assert_eq!(s.all_devices().len(), 2);
    }

    // --- ScanResult helpers -----------------------------------------------

    #[test]
    fn test_scan_result_found_count() {
        let r = ScanResult {
            found: vec![
                DeviceCandidate {
                    address: 1,
                    unit_id: 1,
                    detected_at_ms: 0,
                },
                DeviceCandidate {
                    address: 2,
                    unit_id: 2,
                    detected_at_ms: 0,
                },
            ],
            total_scanned: 10,
            duration_ms: 100,
        };
        assert_eq!(r.found_count(), 2);
    }

    // --- ScanError Display ------------------------------------------------

    #[test]
    fn test_scan_error_display() {
        assert_eq!(ScanError::InvalidRange.to_string(), "invalid address range");
        assert_eq!(ScanError::Timeout.to_string(), "scan timed out");
        assert_eq!(ScanError::Aborted.to_string(), "scan aborted");
    }

    // --- Edge cases -------------------------------------------------------

    #[test]
    fn test_scan_all_255_addresses() {
        let cfg = ScanConfig {
            start_address: 1,
            end_address: 255,
            ..Default::default()
        };
        let mut s = DeviceScanner::new(cfg);
        let result = s.simulate_scan(&[], 0).expect("ok");
        assert_eq!(result.total_scanned, 255);
    }

    #[test]
    fn test_address_range_valid_start_one_end_one() {
        let cfg = ScanConfig {
            start_address: 1,
            end_address: 1,
            ..Default::default()
        };
        assert!(DeviceScanner::address_range_valid(&cfg));
    }

    #[test]
    fn test_simulate_scan_boundary_units() {
        let cfg = ScanConfig {
            start_address: 1,
            end_address: 255,
            ..Default::default()
        };
        let mut s = DeviceScanner::new(cfg);
        let result = s.simulate_scan(&[1, 255], 0).expect("ok");
        assert_eq!(result.found_count(), 2);
    }

    #[test]
    fn test_device_info_clone() {
        let d = DeviceInfo::new(7, 10, 20, Some("A"));
        let c = d.clone();
        assert_eq!(d, c);
    }

    #[test]
    fn test_scan_config_clone() {
        let cfg = ScanConfig::default();
        let c = cfg.clone();
        assert_eq!(cfg, c);
    }
}
