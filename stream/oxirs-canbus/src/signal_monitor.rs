//! CAN signal monitoring with threshold alerting.
//!
//! Provides a registry of signal definitions, sample recording, physical-value
//! decoding (raw * scale + offset), range checking, and rate-of-change
//! computation.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Definition of a CAN signal extracted from a frame payload.
#[derive(Debug, Clone)]
pub struct SignalDef {
    /// Unique signal name (e.g. "EngineSpeed").
    pub name: String,
    /// CAN frame identifier that carries this signal.
    pub frame_id: u32,
    /// Bit position of the signal's LSB within the 8-byte frame payload.
    pub start_bit: u8,
    /// Number of bits that make up the raw value.
    pub length_bits: u8,
    /// Scale factor: `phys = raw * scale + offset`.
    pub scale: f64,
    /// Offset applied after scaling.
    pub offset: f64,
    /// Lower bound of the valid physical range.
    pub min_value: f64,
    /// Upper bound of the valid physical range.
    pub max_value: f64,
    /// Physical unit string (e.g. "rpm", "°C").
    pub unit: String,
}

/// Kind of alert that can be raised for a signal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlertKind {
    /// Physical value is outside `[min_value, max_value]`.
    OutOfRange,
    /// The value changed faster than an expected threshold.
    RapidChange,
    /// No sample has been received within the expected interval.
    MissingFrame,
}

/// An alert raised for a monitored signal.
#[derive(Debug, Clone)]
pub struct SignalAlert {
    /// Name of the signal that raised the alert.
    pub signal_name: String,
    /// What triggered the alert.
    pub kind: AlertKind,
    /// The physical value that caused the alert, if applicable.
    pub value: Option<f64>,
    /// Millisecond timestamp of the alert.
    pub timestamp_ms: u64,
}

/// A recorded sample for a signal.
#[derive(Debug, Clone)]
pub struct SignalSample {
    /// Millisecond timestamp of the sample.
    pub timestamp_ms: u64,
    /// Raw bit-extracted value from the CAN frame.
    pub raw_value: u64,
    /// Physical value computed as `raw * scale + offset`.
    pub phys_value: f64,
}

/// Monitors CAN signals and raises alerts for out-of-range or rapid changes.
pub struct SignalMonitor {
    signals: HashMap<String, SignalDef>,
    samples: HashMap<String, Vec<SignalSample>>,
    alerts: Vec<SignalAlert>,
    max_samples: usize,
}

impl SignalMonitor {
    /// Create a new `SignalMonitor`.
    ///
    /// `max_samples` limits how many samples are retained per signal (oldest
    /// are dropped first).
    pub fn new(max_samples: usize) -> Self {
        Self {
            signals: HashMap::new(),
            samples: HashMap::new(),
            alerts: Vec::new(),
            max_samples: max_samples.max(1),
        }
    }

    /// Register a signal definition.
    pub fn register_signal(&mut self, def: SignalDef) {
        let name = def.name.clone();
        self.signals.insert(name.clone(), def);
        self.samples.entry(name).or_default();
    }

    /// Decode a raw value to a physical value using a signal's scale and offset.
    pub fn decode_physical(def: &SignalDef, raw: u64) -> f64 {
        raw as f64 * def.scale + def.offset
    }

    /// Record a new raw sample for `signal_name` and return any alerts raised.
    ///
    /// The raw value is decoded to a physical value and compared against the
    /// signal's configured range.  If the previous sample exists, the rate of
    /// change is checked; values that change by more than one full range width
    /// per millisecond are flagged as `RapidChange`.
    ///
    /// Returns an empty `Vec` when no alerts are triggered.
    pub fn record(&mut self, signal_name: &str, raw: u64, now_ms: u64) -> Vec<SignalAlert> {
        let mut new_alerts = Vec::new();

        let def = match self.signals.get(signal_name) {
            Some(d) => d.clone(),
            None => return new_alerts,
        };

        let phys = Self::decode_physical(&def, raw);

        // --- range check ---
        if phys < def.min_value || phys > def.max_value {
            let alert = SignalAlert {
                signal_name: signal_name.to_string(),
                kind: AlertKind::OutOfRange,
                value: Some(phys),
                timestamp_ms: now_ms,
            };
            new_alerts.push(alert.clone());
            self.alerts.push(alert);
        }

        // --- rate-of-change check ---
        let range_width = (def.max_value - def.min_value).abs();
        if range_width > 0.0 {
            if let Some(last) = self.samples.get(signal_name).and_then(|s| s.last()) {
                let dt_ms = now_ms.saturating_sub(last.timestamp_ms);
                if dt_ms > 0 {
                    let delta = (phys - last.phys_value).abs();
                    let rate = delta / dt_ms as f64; // units / ms
                    if rate > range_width {
                        let alert = SignalAlert {
                            signal_name: signal_name.to_string(),
                            kind: AlertKind::RapidChange,
                            value: Some(phys),
                            timestamp_ms: now_ms,
                        };
                        new_alerts.push(alert.clone());
                        self.alerts.push(alert);
                    }
                }
            }
        }

        // --- store sample ---
        let sample_buf = self.samples.entry(signal_name.to_string()).or_default();
        if sample_buf.len() >= self.max_samples {
            sample_buf.remove(0);
        }
        sample_buf.push(SignalSample {
            timestamp_ms: now_ms,
            raw_value: raw,
            phys_value: phys,
        });

        new_alerts
    }

    /// Return the most recent sample for `signal_name`, or `None`.
    pub fn latest(&self, signal_name: &str) -> Option<&SignalSample> {
        self.samples.get(signal_name)?.last()
    }

    /// Return all stored samples for `signal_name` (oldest first).
    pub fn history(&self, signal_name: &str) -> &[SignalSample] {
        self.samples
            .get(signal_name)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Return all alerts that have been raised so far.
    pub fn all_alerts(&self) -> &[SignalAlert] {
        &self.alerts
    }

    /// Number of registered signals.
    pub fn signal_count(&self) -> usize {
        self.signals.len()
    }

    /// Estimate the instantaneous rate of change for `signal_name` in
    /// units-per-millisecond, computed from the two most recent samples.
    ///
    /// Returns `None` if fewer than two samples are available.
    pub fn rate_of_change(&self, signal_name: &str) -> Option<f64> {
        let samples = self.samples.get(signal_name)?;
        if samples.len() < 2 {
            return None;
        }
        let last = &samples[samples.len() - 1];
        let prev = &samples[samples.len() - 2];
        let dt_ms = last.timestamp_ms.saturating_sub(prev.timestamp_ms);
        if dt_ms == 0 {
            return None;
        }
        Some((last.phys_value - prev.phys_value) / dt_ms as f64)
    }

    /// Access the signal definitions map.
    pub fn signals(&self) -> &HashMap<String, SignalDef> {
        &self.signals
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_def(name: &str, min: f64, max: f64) -> SignalDef {
        SignalDef {
            name: name.to_string(),
            frame_id: 0x100,
            start_bit: 0,
            length_bits: 16,
            scale: 1.0,
            offset: 0.0,
            min_value: min,
            max_value: max,
            unit: "unit".to_string(),
        }
    }

    fn make_def_scaled(name: &str, scale: f64, offset: f64, min: f64, max: f64) -> SignalDef {
        SignalDef {
            name: name.to_string(),
            frame_id: 0x200,
            start_bit: 0,
            length_bits: 8,
            scale,
            offset,
            min_value: min,
            max_value: max,
            unit: "°C".to_string(),
        }
    }

    // --- new / register_signal ---

    #[test]
    fn test_new_empty() {
        let mon = SignalMonitor::new(100);
        assert_eq!(mon.signal_count(), 0);
        assert!(mon.all_alerts().is_empty());
    }

    #[test]
    fn test_register_signal_increments_count() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def("Speed", 0.0, 300.0));
        assert_eq!(mon.signal_count(), 1);
        mon.register_signal(make_def("Temp", -40.0, 120.0));
        assert_eq!(mon.signal_count(), 2);
    }

    #[test]
    fn test_register_signal_replaces_existing() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def("Temp", -40.0, 100.0));
        mon.register_signal(make_def("Temp", -50.0, 150.0));
        assert_eq!(mon.signal_count(), 1);
        let def = &mon.signals()["Temp"];
        assert_eq!(def.min_value, -50.0);
    }

    // --- decode_physical ---

    #[test]
    fn test_decode_physical_scale_offset() {
        let def = make_def_scaled("T", 0.5, -40.0, -40.0, 87.5);
        let phys = SignalMonitor::decode_physical(&def, 100);
        assert!((phys - (100.0 * 0.5 - 40.0)).abs() < 1e-9);
    }

    #[test]
    fn test_decode_physical_zero_raw() {
        let def = make_def_scaled("T", 1.0, 5.0, 0.0, 255.0);
        let phys = SignalMonitor::decode_physical(&def, 0);
        assert!((phys - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_decode_physical_identity() {
        let def = make_def("Speed", 0.0, 1000.0);
        let phys = SignalMonitor::decode_physical(&def, 250);
        assert!((phys - 250.0).abs() < 1e-9);
    }

    // --- record ---

    #[test]
    fn test_record_in_range_no_alerts() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def("Speed", 0.0, 300.0));
        let alerts = mon.record("Speed", 100, 1000);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_record_out_of_range_high_alert() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def("Speed", 0.0, 200.0));
        let alerts = mon.record("Speed", 250, 1000);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].kind, AlertKind::OutOfRange);
        assert_eq!(alerts[0].signal_name, "Speed");
    }

    #[test]
    fn test_record_out_of_range_low_alert() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def_scaled("Temp", 1.0, -40.0, -40.0, 100.0));
        // raw=0 → phys = -40; min is -40 so this is exactly at boundary (no alert)
        let alerts = mon.record("Temp", 0, 1000);
        assert!(alerts.is_empty());
        // phys = 0*1 + (-40) = -40 (= min, OK)
    }

    #[test]
    fn test_record_unknown_signal_returns_empty() {
        let mut mon = SignalMonitor::new(100);
        let alerts = mon.record("Unknown", 100, 1000);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_record_stores_sample() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def("Speed", 0.0, 300.0));
        mon.record("Speed", 120, 5000);
        let sample = mon.latest("Speed").expect("sample present");
        assert_eq!(sample.raw_value, 120);
        assert_eq!(sample.timestamp_ms, 5000);
    }

    #[test]
    fn test_record_multiple_samples_ordered() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def("Speed", 0.0, 300.0));
        mon.record("Speed", 10, 100);
        mon.record("Speed", 20, 200);
        mon.record("Speed", 30, 300);
        let hist = mon.history("Speed");
        assert_eq!(hist.len(), 3);
        assert_eq!(hist[0].raw_value, 10);
        assert_eq!(hist[2].raw_value, 30);
    }

    #[test]
    fn test_record_respects_max_samples() {
        let mut mon = SignalMonitor::new(3);
        mon.register_signal(make_def("Speed", 0.0, 300.0));
        for i in 0..6u64 {
            mon.record("Speed", i * 10, i * 100);
        }
        assert_eq!(mon.history("Speed").len(), 3);
        // Oldest should be gone; latest is the 6th (raw=50)
        assert_eq!(mon.latest("Speed").expect("present").raw_value, 50);
    }

    // --- latest / history ---

    #[test]
    fn test_latest_none_initially() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def("Speed", 0.0, 300.0));
        assert!(mon.latest("Speed").is_none());
    }

    #[test]
    fn test_latest_unknown_signal_none() {
        let mon = SignalMonitor::new(100);
        assert!(mon.latest("Nonexistent").is_none());
    }

    #[test]
    fn test_history_unknown_signal_empty_slice() {
        let mon = SignalMonitor::new(100);
        assert!(mon.history("Nonexistent").is_empty());
    }

    // --- all_alerts ---

    #[test]
    fn test_all_alerts_accumulates() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def("Speed", 0.0, 100.0));
        mon.record("Speed", 200, 1000); // out of range
        mon.record("Speed", 250, 2000); // out of range
        assert_eq!(mon.all_alerts().len(), 2);
    }

    #[test]
    fn test_all_alerts_empty_initially() {
        let mon = SignalMonitor::new(100);
        assert!(mon.all_alerts().is_empty());
    }

    // --- rate_of_change ---

    #[test]
    fn test_rate_of_change_none_with_one_sample() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def("Speed", 0.0, 300.0));
        mon.record("Speed", 100, 1000);
        assert!(mon.rate_of_change("Speed").is_none());
    }

    #[test]
    fn test_rate_of_change_computed() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def("Speed", 0.0, 300.0));
        mon.record("Speed", 100, 1000);
        mon.record("Speed", 110, 1010); // Δ = 10 over 10ms → 1.0 unit/ms
        let roc = mon.rate_of_change("Speed").expect("two samples");
        assert!((roc - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_rate_of_change_negative() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def("Temp", -50.0, 150.0));
        mon.record("Temp", 100, 0);
        mon.record("Temp", 80, 5); // decreasing
        let roc = mon.rate_of_change("Temp").expect("two samples");
        assert!(roc < 0.0);
    }

    #[test]
    fn test_rate_of_change_none_unknown_signal() {
        let mon = SignalMonitor::new(100);
        assert!(mon.rate_of_change("Unknown").is_none());
    }

    // --- rapid change alert ---

    #[test]
    fn test_rapid_change_alert_raised() {
        let mut mon = SignalMonitor::new(100);
        // range = 100-0 = 100; a change of 150 in 1ms = 150 units/ms > 100 → RapidChange
        mon.register_signal(make_def("Pressure", 0.0, 100.0));
        mon.record("Pressure", 10, 1000);
        let alerts = mon.record("Pressure", 160, 1001);
        let rapid: Vec<_> = alerts
            .iter()
            .filter(|a| a.kind == AlertKind::RapidChange)
            .collect();
        assert!(!rapid.is_empty());
    }

    #[test]
    fn test_no_rapid_change_on_first_sample() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def("Speed", 0.0, 300.0));
        let alerts = mon.record("Speed", 0, 1000);
        assert!(alerts.iter().all(|a| a.kind != AlertKind::RapidChange));
    }

    // --- combined scenarios ---

    #[test]
    fn test_out_of_range_and_rapid_change_simultaneously() {
        let mut mon = SignalMonitor::new(100);
        // range 0..10; any value >10 is OoR; change > 10/ms is rapid
        mon.register_signal(make_def("Tiny", 0.0, 10.0));
        mon.record("Tiny", 5, 1000);
        // raw=100 → phys=100 (OoR); Δ=95 over 1ms = 95 > 10 → also RapidChange
        let alerts = mon.record("Tiny", 100, 1001);
        let kinds: Vec<_> = alerts.iter().map(|a| &a.kind).collect();
        assert!(kinds.contains(&&AlertKind::OutOfRange));
        assert!(kinds.contains(&&AlertKind::RapidChange));
    }

    #[test]
    fn test_alert_timestamp_matches_record_time() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def("Speed", 0.0, 100.0));
        let alerts = mon.record("Speed", 999, 7777);
        assert_eq!(alerts[0].timestamp_ms, 7777);
    }

    #[test]
    fn test_alert_value_is_physical() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def_scaled("Temp", 0.5, -40.0, -40.0, 87.5));
        // raw = 260 → phys = 130 - 40 = 90 > 87.5 → OoR
        let alerts = mon.record("Temp", 260, 1000);
        let oor: Vec<_> = alerts
            .iter()
            .filter(|a| a.kind == AlertKind::OutOfRange)
            .collect();
        assert!(!oor.is_empty());
        let v = oor[0].value.expect("value present");
        assert!((v - (260.0 * 0.5 - 40.0)).abs() < 1e-6);
    }

    #[test]
    fn test_signal_count_after_multiple_registrations() {
        let mut mon = SignalMonitor::new(100);
        let sigs = ["Speed", "Temp", "Pressure", "Voltage", "Current"];
        for s in &sigs {
            mon.register_signal(make_def(s, 0.0, 100.0));
        }
        assert_eq!(mon.signal_count(), 5);
    }

    #[test]
    fn test_phys_value_stored_in_sample() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def_scaled("T", 2.0, 10.0, 0.0, 1000.0));
        mon.record("T", 50, 1000);
        let s = mon.latest("T").expect("sample");
        // phys = 50 * 2 + 10 = 110
        assert!((s.phys_value - 110.0).abs() < 1e-9);
    }

    #[test]
    fn test_history_empty_after_register_before_record() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def("Speed", 0.0, 300.0));
        assert!(mon.history("Speed").is_empty());
    }

    #[test]
    fn test_max_samples_one() {
        let mut mon = SignalMonitor::new(1);
        mon.register_signal(make_def("Speed", 0.0, 300.0));
        mon.record("Speed", 10, 100);
        mon.record("Speed", 20, 200);
        assert_eq!(mon.history("Speed").len(), 1);
        assert_eq!(mon.latest("Speed").unwrap().raw_value, 20);
    }

    // --- additional coverage ---

    #[test]
    fn test_decode_physical_negative_offset() {
        let def = make_def_scaled("T", 1.0, -100.0, -100.0, 155.0);
        let phys = SignalMonitor::decode_physical(&def, 200);
        assert!((phys - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_record_exact_min_boundary_no_alert() {
        let mut mon = SignalMonitor::new(10);
        mon.register_signal(make_def("V", 10.0, 100.0));
        let alerts = mon.record("V", 10, 1000);
        // phys = 10.0 == min_value (exactly at boundary → no OoR)
        assert!(alerts.iter().all(|a| a.kind != AlertKind::OutOfRange));
    }

    #[test]
    fn test_record_exact_max_boundary_no_alert() {
        let mut mon = SignalMonitor::new(10);
        mon.register_signal(make_def("V", 0.0, 100.0));
        let alerts = mon.record("V", 100, 1000);
        assert!(alerts.iter().all(|a| a.kind != AlertKind::OutOfRange));
    }

    #[test]
    fn test_record_below_min_alert() {
        let mut mon = SignalMonitor::new(10);
        // offset makes phys negative; use pure integer scale
        mon.register_signal(make_def_scaled("T", 1.0, 0.0, 10.0, 200.0));
        let alerts = mon.record("T", 5, 1000); // phys=5 < min=10
        assert!(alerts.iter().any(|a| a.kind == AlertKind::OutOfRange));
    }

    #[test]
    fn test_no_rapid_change_when_dt_zero() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def("Speed", 0.0, 100.0));
        mon.record("Speed", 10, 1000);
        // Same timestamp → dt = 0 → rate undefined → no RapidChange
        let alerts = mon.record("Speed", 90, 1000);
        assert!(alerts.iter().all(|a| a.kind != AlertKind::RapidChange));
    }

    #[test]
    fn test_all_alerts_reflects_accumulation_across_signals() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def("A", 0.0, 10.0));
        mon.register_signal(make_def("B", 0.0, 10.0));
        mon.record("A", 100, 1000); // OoR
        mon.record("B", 200, 2000); // OoR
        assert_eq!(mon.all_alerts().len(), 2);
    }

    #[test]
    fn test_signal_frame_id_stored() {
        let mut mon = SignalMonitor::new(10);
        let def = SignalDef {
            name: "RPM".to_string(),
            frame_id: 0x0CF,
            start_bit: 0,
            length_bits: 16,
            scale: 0.125,
            offset: 0.0,
            min_value: 0.0,
            max_value: 8031.875,
            unit: "rpm".to_string(),
        };
        mon.register_signal(def);
        assert_eq!(mon.signals()["RPM"].frame_id, 0x0CF);
    }

    #[test]
    fn test_rate_of_change_same_timestamps_returns_none() {
        let mut mon = SignalMonitor::new(100);
        mon.register_signal(make_def("Speed", 0.0, 300.0));
        mon.record("Speed", 10, 500);
        mon.record("Speed", 20, 500); // same ts
        assert!(mon.rate_of_change("Speed").is_none());
    }

    #[test]
    fn test_record_multiple_signals_independent() {
        let mut mon = SignalMonitor::new(5);
        mon.register_signal(make_def("A", 0.0, 100.0));
        mon.register_signal(make_def("B", 0.0, 100.0));
        mon.record("A", 50, 100);
        mon.record("B", 60, 200);
        assert_eq!(mon.history("A").len(), 1);
        assert_eq!(mon.history("B").len(), 1);
        assert_eq!(mon.latest("A").unwrap().raw_value, 50);
        assert_eq!(mon.latest("B").unwrap().raw_value, 60);
    }

    #[test]
    fn test_alert_kind_eq() {
        assert_eq!(AlertKind::OutOfRange, AlertKind::OutOfRange);
        assert_ne!(AlertKind::OutOfRange, AlertKind::RapidChange);
        assert_ne!(AlertKind::RapidChange, AlertKind::MissingFrame);
    }

    #[test]
    fn test_max_samples_zero_becomes_one() {
        // Constructing with 0 should silently become 1
        let mut mon = SignalMonitor::new(0);
        mon.register_signal(make_def("X", 0.0, 10.0));
        mon.record("X", 1, 100);
        mon.record("X", 2, 200);
        assert_eq!(mon.history("X").len(), 1);
    }

    #[test]
    fn test_record_preserves_raw_value_in_sample() {
        let mut mon = SignalMonitor::new(10);
        mon.register_signal(make_def("Fuel", 0.0, 255.0));
        mon.record("Fuel", 128, 9000);
        assert_eq!(mon.latest("Fuel").unwrap().raw_value, 128);
    }

    #[test]
    fn test_decode_physical_large_raw_value() {
        let def = SignalDef {
            name: "RPM".to_string(),
            frame_id: 0x0CF,
            start_bit: 0,
            length_bits: 16,
            scale: 0.125,
            offset: 0.0,
            min_value: 0.0,
            max_value: 8031.875,
            unit: "rpm".to_string(),
        };
        // raw = 64255 → phys = 64255 * 0.125 = 8031.875
        let phys = SignalMonitor::decode_physical(&def, 64255);
        assert!((phys - 8031.875).abs() < 1e-6);
    }

    #[test]
    fn test_record_out_of_range_alert_value_matches_physical() {
        let mut mon = SignalMonitor::new(10);
        // scale=2, offset=0, max=100
        let def = SignalDef {
            name: "Sig".to_string(),
            frame_id: 1,
            start_bit: 0,
            length_bits: 8,
            scale: 2.0,
            offset: 0.0,
            min_value: 0.0,
            max_value: 100.0,
            unit: "".to_string(),
        };
        mon.register_signal(def);
        // raw=60 → phys=120 > 100
        let alerts = mon.record("Sig", 60, 500);
        let oor: Vec<_> = alerts
            .iter()
            .filter(|a| a.kind == AlertKind::OutOfRange)
            .collect();
        assert!(!oor.is_empty());
        assert!((oor[0].value.unwrap() - 120.0).abs() < 1e-9);
    }
}
