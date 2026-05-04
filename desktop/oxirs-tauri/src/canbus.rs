use crate::error::AppResult;
use serde::{Deserialize, Serialize};

/// A decoded J1939 CAN frame for display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanFrameView {
    pub id: String,
    pub pgn: u32,
    pub sa: u8,
    pub da: Option<u8>,
    pub data_hex: String,
    pub length: u8,
    pub timestamp_us: u64,
    pub pgn_name: String,
    pub decoded_signals: Vec<SignalView>,
}

/// A decoded SPN signal from a PGN.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalView {
    pub spn: u32,
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub raw: u64,
}

/// Bus statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusStats {
    pub frame_count: u64,
    pub error_count: u64,
    /// Estimated bus load 0–100 %.
    pub bus_load_pct: f64,
    pub frames_per_sec: f64,
    pub uptime_secs: u64,
}

/// Filter configuration for the frame monitor.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FrameFilter {
    pub pgn: Option<u32>,
    pub sa: Option<u8>,
    pub min_timestamp_us: Option<u64>,
}

/// Return a snapshot of recent CAN frames (mock data for UI validation).
///
/// In production this would subscribe to the oxirs-canbus frame stream.
/// The mock returns representative J1939 PGNs so the UI has real data to display.
#[tauri::command]
pub fn get_frames(filter: Option<FrameFilter>, limit: Option<u32>) -> AppResult<Vec<CanFrameView>> {
    let limit = limit.unwrap_or(50) as usize;
    let filter = filter.unwrap_or_default();

    let frames = mock_frames();
    let filtered: Vec<CanFrameView> = frames
        .into_iter()
        .filter(|f| {
            if let Some(pgn) = filter.pgn {
                if f.pgn != pgn {
                    return false;
                }
            }
            if let Some(sa) = filter.sa {
                if f.sa != sa {
                    return false;
                }
            }
            if let Some(min_ts) = filter.min_timestamp_us {
                if f.timestamp_us < min_ts {
                    return false;
                }
            }
            true
        })
        .take(limit)
        .collect();

    Ok(filtered)
}

/// Return current bus statistics.
///
/// Mock stats; replace with real monitoring data from oxirs-canbus.
#[tauri::command]
pub fn get_bus_stats() -> AppResult<BusStats> {
    Ok(BusStats {
        frame_count: 18_432,
        error_count: 3,
        bus_load_pct: 23.4,
        frames_per_sec: 487.0,
        uptime_secs: 37,
    })
}

/// Look up a PGN by number and return its name and description.
#[tauri::command]
pub fn lookup_pgn(pgn: u32) -> Option<(String, String)> {
    pgn_database()
        .into_iter()
        .find(|(n, _, _)| *n == pgn)
        .map(|(_, name, desc)| (name.to_string(), desc.to_string()))
}

/// Return the full PGN database for the UI lookup table.
#[tauri::command]
pub fn get_pgn_database() -> Vec<(u32, String, String)> {
    pgn_database()
        .into_iter()
        .map(|(n, name, desc)| (n, name.to_string(), desc.to_string()))
        .collect()
}

/// Clear the frame buffer (no-op in mock mode).
#[tauri::command]
pub fn clear_frames() -> AppResult<()> {
    Ok(())
}

// ---------------------------------------------------------------------------
// Mock data helpers
// ---------------------------------------------------------------------------

fn pgn_database() -> Vec<(u32, &'static str, &'static str)> {
    vec![
        (
            61_444,
            "EEC1",
            "Electronic Engine Controller 1 — Engine speed, torque",
        ),
        (
            65_262,
            "ET1",
            "Engine Temperature 1 — Coolant temp, fuel temp",
        ),
        (
            65_263,
            "EFL/P1",
            "Engine Fluid Level/Pressure 1 — Oil pressure, level",
        ),
        (
            65_265,
            "CCVS",
            "Cruise Control/Vehicle Speed — Wheel speed, clutch",
        ),
        (
            65_270,
            "IC1",
            "Inlet/Exhaust Conditions 1 — Boost pressure, air temp",
        ),
        (
            65_271,
            "VEP1",
            "Vehicle Electrical Power 1 — Battery voltage, current",
        ),
        (
            61_445,
            "EEC2",
            "Electronic Engine Controller 2 — Throttle position",
        ),
        (
            65_129,
            "EBC2",
            "Electronic Brake Controller 2 — Wheel speed sensors",
        ),
        (
            65_215,
            "RQST",
            "Request PGN — Request for data transmission",
        ),
        (60_928, "AC", "Address Claimed — J1939 address claim"),
    ]
}

fn mock_frames() -> Vec<CanFrameView> {
    vec![
        CanFrameView {
            id: "f001".to_string(),
            pgn: 61_444,
            sa: 0x00,
            da: None,
            data_hex: "F0 00 FF FF FF 20 00 00".to_string(),
            length: 8,
            timestamp_us: 0,
            pgn_name: "EEC1 — Electronic Engine Controller 1".to_string(),
            decoded_signals: vec![
                SignalView {
                    spn: 190,
                    name: "Engine Speed".to_string(),
                    value: 1800.0,
                    unit: "rpm".to_string(),
                    raw: 7200,
                },
                SignalView {
                    spn: 91,
                    name: "Throttle Position".to_string(),
                    value: 0.0,
                    unit: "%".to_string(),
                    raw: 0,
                },
            ],
        },
        CanFrameView {
            id: "f002".to_string(),
            pgn: 65_262,
            sa: 0x00,
            da: None,
            data_hex: "62 FF FF FF FF FF FF FF".to_string(),
            length: 8,
            timestamp_us: 2_048,
            pgn_name: "ET1 — Engine Temperature 1".to_string(),
            decoded_signals: vec![SignalView {
                spn: 110,
                name: "Engine Coolant Temperature".to_string(),
                value: 88.0,
                unit: "\u{00b0}C".to_string(),
                raw: 186,
            }],
        },
        CanFrameView {
            id: "f003".to_string(),
            pgn: 65_271,
            sa: 0x17,
            da: None,
            data_hex: "00 C8 07 00 FF FF FF FF".to_string(),
            length: 8,
            timestamp_us: 4_096,
            pgn_name: "VEP1 — Vehicle Electrical Power 1".to_string(),
            decoded_signals: vec![SignalView {
                spn: 158,
                name: "Battery Voltage".to_string(),
                value: 13.8,
                unit: "V".to_string(),
                raw: 552,
            }],
        },
        CanFrameView {
            id: "f004".to_string(),
            pgn: 65_265,
            sa: 0x28,
            da: None,
            data_hex: "F2 2F 00 00 00 FF FF FF".to_string(),
            length: 8,
            timestamp_us: 6_144,
            pgn_name: "CCVS — Cruise Control/Vehicle Speed".to_string(),
            decoded_signals: vec![SignalView {
                spn: 84,
                name: "Wheel-Based Vehicle Speed".to_string(),
                value: 87.5,
                unit: "km/h".to_string(),
                raw: 3500,
            }],
        },
        CanFrameView {
            id: "f005".to_string(),
            pgn: 65_263,
            sa: 0x00,
            da: None,
            data_hex: "FF 28 FF FF FF FF FF FF".to_string(),
            length: 8,
            timestamp_us: 8_192,
            pgn_name: "EFL/P1 — Engine Fluid Level/Pressure 1".to_string(),
            decoded_signals: vec![SignalView {
                spn: 100,
                name: "Engine Oil Pressure".to_string(),
                value: 420.0,
                unit: "kPa".to_string(),
                raw: 40,
            }],
        },
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_frames_returns_mock_data() {
        let frames = get_frames(None, None).unwrap();
        assert!(!frames.is_empty());
    }

    #[test]
    fn test_get_frames_limit_respected() {
        let frames = get_frames(None, Some(2)).unwrap();
        assert!(frames.len() <= 2);
    }

    #[test]
    fn test_get_frames_filter_by_pgn() {
        let filter = FrameFilter {
            pgn: Some(61_444),
            sa: None,
            min_timestamp_us: None,
        };
        let frames = get_frames(Some(filter), None).unwrap();
        assert!(frames.iter().all(|f| f.pgn == 61_444));
    }

    #[test]
    fn test_get_frames_filter_by_sa() {
        let filter = FrameFilter {
            pgn: None,
            sa: Some(0x00),
            min_timestamp_us: None,
        };
        let frames = get_frames(Some(filter), None).unwrap();
        assert!(frames.iter().all(|f| f.sa == 0x00));
    }

    #[test]
    fn test_get_frames_filter_by_min_timestamp() {
        let filter = FrameFilter {
            pgn: None,
            sa: None,
            min_timestamp_us: Some(4_000),
        };
        let frames = get_frames(Some(filter), None).unwrap();
        assert!(frames.iter().all(|f| f.timestamp_us >= 4_000));
    }

    #[test]
    fn test_get_bus_stats() {
        let stats = get_bus_stats().unwrap();
        assert!(stats.frame_count > 0);
        assert!((0.0..=100.0).contains(&stats.bus_load_pct));
        assert!(stats.frames_per_sec > 0.0);
    }

    #[test]
    fn test_get_bus_stats_uptime_positive() {
        let stats = get_bus_stats().unwrap();
        assert!(stats.uptime_secs > 0);
    }

    #[test]
    fn test_lookup_pgn_known() {
        let r = lookup_pgn(61_444);
        assert!(r.is_some());
        let (name, _) = r.unwrap();
        assert_eq!(name, "EEC1");
    }

    #[test]
    fn test_lookup_pgn_unknown() {
        let r = lookup_pgn(0xDEAD_BEEF);
        assert!(r.is_none());
    }

    #[test]
    fn test_get_pgn_database_nonempty() {
        let db = get_pgn_database();
        assert!(db.len() >= 5);
    }

    #[test]
    fn test_get_pgn_database_contains_eec1() {
        let db = get_pgn_database();
        let found = db.iter().any(|(n, name, _)| *n == 61_444 && name == "EEC1");
        assert!(found);
    }

    #[test]
    fn test_clear_frames_ok() {
        assert!(clear_frames().is_ok());
    }

    #[test]
    fn test_frame_view_serialization() {
        let f = CanFrameView {
            id: "f1".to_string(),
            pgn: 0,
            sa: 0,
            da: None,
            data_hex: "FF".to_string(),
            length: 1,
            timestamp_us: 0,
            pgn_name: "test".to_string(),
            decoded_signals: vec![],
        };
        let json = serde_json::to_string(&f).unwrap();
        let back: CanFrameView = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "f1");
    }

    #[test]
    fn test_frame_view_with_da_serialization() {
        let f = CanFrameView {
            id: "f2".to_string(),
            pgn: 0,
            sa: 0,
            da: Some(0xFF),
            data_hex: "00".to_string(),
            length: 1,
            timestamp_us: 0,
            pgn_name: "peer".to_string(),
            decoded_signals: vec![],
        };
        let json = serde_json::to_string(&f).unwrap();
        let back: CanFrameView = serde_json::from_str(&json).unwrap();
        assert_eq!(back.da, Some(0xFF));
    }

    #[test]
    fn test_signal_view_serialization() {
        let s = SignalView {
            spn: 110,
            name: "Temp".to_string(),
            value: 88.0,
            unit: "\u{00b0}C".to_string(),
            raw: 186,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("Temp"));
        let back: SignalView = serde_json::from_str(&json).unwrap();
        assert_eq!(back.spn, 110);
        assert!((back.value - 88.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_decoded_signals_in_frames() {
        let frames = get_frames(None, None).unwrap();
        let eec1 = frames.iter().find(|f| f.pgn == 61_444).unwrap();
        assert!(!eec1.decoded_signals.is_empty());
        let rpm_signal = eec1.decoded_signals.iter().find(|s| s.spn == 190).unwrap();
        assert!(rpm_signal.value > 0.0);
    }

    #[test]
    fn test_bus_stats_error_count_low() {
        let stats = get_bus_stats().unwrap();
        // Mock data has very few errors.
        assert!(stats.error_count < 100);
    }

    #[test]
    fn test_frame_filter_default_is_no_filter() {
        let filter = FrameFilter::default();
        assert!(filter.pgn.is_none());
        assert!(filter.sa.is_none());
        assert!(filter.min_timestamp_us.is_none());
    }

    #[test]
    fn test_get_frames_combined_filter() {
        let filter = FrameFilter {
            pgn: Some(61_444),
            sa: Some(0x00),
            min_timestamp_us: None,
        };
        let frames = get_frames(Some(filter), None).unwrap();
        assert!(frames.iter().all(|f| f.pgn == 61_444 && f.sa == 0x00));
    }

    #[test]
    fn test_lookup_pgn_et1() {
        let r = lookup_pgn(65_262);
        assert!(r.is_some());
        let (name, desc) = r.unwrap();
        assert_eq!(name, "ET1");
        assert!(desc.contains("Temperature"));
    }

    #[test]
    fn test_pgn_database_all_entries_have_names() {
        let db = get_pgn_database();
        for (_, name, _) in &db {
            assert!(!name.is_empty());
        }
    }

    #[test]
    fn test_pgn_database_all_entries_have_descriptions() {
        let db = get_pgn_database();
        for (_, _, desc) in &db {
            assert!(!desc.is_empty());
        }
    }

    #[test]
    fn test_app_error_not_used_but_importable() {
        // Verify the AppError import compiles; clear_frames returns AppResult<()>.
        let _: AppResult<()> = Ok(());
    }
}
