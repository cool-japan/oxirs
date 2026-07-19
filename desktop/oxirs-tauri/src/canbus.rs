use crate::error::AppResult;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::time::Instant;

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

/// Response envelope for [`get_frames`]. The UI must key its rendering off
/// `source_configured`/`demo` rather than assuming `frames` is ever live
/// vehicle data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FramesResponse {
    pub frames: Vec<CanFrameView>,
    /// `true` when `frames` is synthetic demo data (explicit opt-in via
    /// `OXIRS_CANBUS_DEMO=1`), never live bus traffic.
    pub demo: bool,
    /// `true` when a real or demo CAN source is configured and healthy.
    pub source_configured: bool,
    /// Set when a live source was configured but is not currently reachable.
    pub error: Option<String>,
}

/// Response envelope for [`get_bus_stats`]. See [`FramesResponse`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusStatsResponse {
    pub stats: Option<BusStats>,
    pub demo: bool,
    pub source_configured: bool,
    pub error: Option<String>,
}

/// How the CAN frame monitor is currently sourced.
enum SourceMode {
    /// No `OXIRS_CANBUS_INTERFACE` / `OXIRS_CANBUS_DEMO` configured: no data
    /// is served, and the UI must render an explicit "no source" state.
    Unconfigured,
    /// Explicit opt-in demo mode (`OXIRS_CANBUS_DEMO=1`): synthetic but
    /// representative J1939 frames, always tagged `demo: true`.
    Demo,
    /// Live capture from a real CAN interface (`OXIRS_CANBUS_INTERFACE=can0`
    /// etc.), backed by `oxirs_canbus`'s SocketCAN client on Linux.
    Live { interface: String },
}

/// Shared runtime state for the CAN monitor: current source mode, a bounded
/// ring buffer of recently observed frames, and running counters.
struct CanbusRuntime {
    mode: SourceMode,
    frames: StdMutex<VecDeque<CanFrameView>>,
    frame_count: AtomicU64,
    error_count: AtomicU64,
    started_at: Instant,
    /// Populated when a configured live source fails to start or drops.
    last_error: StdMutex<Option<String>>,
}

/// Maximum number of frames retained in the in-memory ring buffer.
const MAX_BUFFERED_FRAMES: usize = 5_000;

/// User-facing message returned whenever no CAN source has been configured
/// (neither `OXIRS_CANBUS_INTERFACE` nor `OXIRS_CANBUS_DEMO=1` is set).
const NO_SOURCE_CONFIGURED_MESSAGE: &str =
    "no CAN source configured: set OXIRS_CANBUS_INTERFACE=<iface> for live capture \
     or OXIRS_CANBUS_DEMO=1 to explicitly opt into demo data";

impl CanbusRuntime {
    fn new() -> Self {
        let mode = if let Ok(interface) = std::env::var("OXIRS_CANBUS_INTERFACE") {
            SourceMode::Live { interface }
        } else if std::env::var("OXIRS_CANBUS_DEMO")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        {
            SourceMode::Demo
        } else {
            SourceMode::Unconfigured
        };

        Self {
            mode,
            frames: StdMutex::new(VecDeque::new()),
            frame_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            started_at: Instant::now(),
            last_error: StdMutex::new(None),
        }
    }

    fn lock_frames(&self) -> std::sync::MutexGuard<'_, VecDeque<CanFrameView>> {
        self.frames
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn lock_last_error(&self) -> std::sync::MutexGuard<'_, Option<String>> {
        self.last_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn set_error(&self, message: String) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
        *self.lock_last_error() = Some(message);
    }

    fn push_frame(&self, frame: CanFrameView) {
        let mut buf = self.lock_frames();
        if buf.len() >= MAX_BUFFERED_FRAMES {
            buf.pop_front();
        }
        buf.push_back(frame);
        self.frame_count.fetch_add(1, Ordering::Relaxed);
    }
}

fn runtime() -> Arc<CanbusRuntime> {
    static RUNTIME: OnceLock<Arc<CanbusRuntime>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            let rt = Arc::new(CanbusRuntime::new());
            start_source(&rt);
            rt
        })
        .clone()
}

fn start_source(rt: &Arc<CanbusRuntime>) {
    match &rt.mode {
        SourceMode::Live { interface } => spawn_live_collector(interface.clone(), rt.clone()),
        SourceMode::Demo => {
            for frame in mock_frames() {
                rt.push_frame(frame);
            }
        }
        SourceMode::Unconfigured => {}
    }
}

/// Live SocketCAN capture, backed by `oxirs_canbus`'s Linux-only client.
#[cfg(target_os = "linux")]
fn spawn_live_collector(interface: String, rt: Arc<CanbusRuntime>) {
    tauri::async_runtime::spawn(async move {
        let config = oxirs_canbus::CanbusConfig {
            interface: interface.clone(),
            j1939_enabled: true,
            ..Default::default()
        };

        let mut client = match oxirs_canbus::CanbusClient::new(config) {
            Ok(client) => client,
            Err(e) => {
                rt.set_error(format!("failed to initialize CAN client: {e}"));
                return;
            }
        };

        if let Err(e) = client.start().await {
            rt.set_error(format!("failed to start CAN interface {interface}: {e}"));
            return;
        }

        let mut processor = oxirs_canbus::J1939Processor::new();
        let registry = oxirs_canbus::PgnRegistry::with_standard_decoders();

        loop {
            match client.recv_frame().await {
                Some(frame) => {
                    if let Some(view) = decode_frame(&rt, &frame, &mut processor, &registry) {
                        rt.push_frame(view);
                    }
                }
                None => {
                    rt.set_error(format!(
                        "connection to CAN interface {interface} was closed"
                    ));
                    break;
                }
            }
        }
    });
}

#[cfg(target_os = "linux")]
fn decode_frame(
    rt: &CanbusRuntime,
    frame: &oxirs_canbus::CanFrame,
    processor: &mut oxirs_canbus::J1939Processor,
    registry: &oxirs_canbus::PgnRegistry,
) -> Option<CanFrameView> {
    let message = processor.process(frame)?;
    let pgn = message.header.pgn.value();
    let decoded = registry.decode(&message);

    let (pgn_name, decoded_signals) = match decoded {
        Some(d) => (
            format!("{} — {}", d.name, d.description),
            d.signals
                .iter()
                .filter(|s| s.valid)
                .map(|s| SignalView {
                    // The lightweight PGN decoders in oxirs-canbus report
                    // name/value/unit/raw but not the originating SPN number;
                    // 0 marks "not tracked" rather than fabricating one.
                    spn: 0,
                    name: s.name.to_string(),
                    value: s.value,
                    unit: s.unit.to_string(),
                    raw: s.raw_value,
                })
                .collect(),
        ),
        None => (format!("Unknown PGN 0x{pgn:05X}"), Vec::new()),
    };

    let seq = rt.frame_count.load(Ordering::Relaxed);
    Some(CanFrameView {
        id: format!("live_{seq}"),
        pgn,
        sa: message.header.source_address,
        da: message.header.destination_address,
        data_hex: hex_encode(&frame.data),
        length: frame.data.len() as u8,
        timestamp_us: rt.started_at.elapsed().as_micros() as u64,
        pgn_name,
        decoded_signals,
    })
}

#[cfg(target_os = "linux")]
fn hex_encode(data: &[u8]) -> String {
    data.iter()
        .map(|b| format!("{b:02X}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(not(target_os = "linux"))]
fn spawn_live_collector(interface: String, rt: Arc<CanbusRuntime>) {
    rt.set_error(format!(
        "live CAN capture requires Linux (socketcan); OXIRS_CANBUS_INTERFACE={interface} \
         cannot be honored on this platform"
    ));
}

/// Return a snapshot of recent CAN frames.
///
/// Serves live frames when `OXIRS_CANBUS_INTERFACE` is configured, synthetic
/// (explicitly `demo: true`) frames when `OXIRS_CANBUS_DEMO=1` is set, and an
/// empty, `source_configured: false` response otherwise — never fabricated
/// data presented as live.
#[tauri::command]
pub fn get_frames(filter: Option<FrameFilter>, limit: Option<u32>) -> AppResult<FramesResponse> {
    let limit = limit.unwrap_or(50) as usize;
    let filter = filter.unwrap_or_default();
    let rt = runtime();

    let (demo, source_configured, error) = match &rt.mode {
        SourceMode::Unconfigured => (false, false, Some(NO_SOURCE_CONFIGURED_MESSAGE.to_string())),
        SourceMode::Demo => (true, true, None),
        SourceMode::Live { interface } => {
            let error = rt.lock_last_error().clone();
            let configured = error.is_none();
            let error = error.map(|e| format!("live CAN source '{interface}': {e}"));
            (false, configured, error)
        }
    };

    let frames: Vec<CanFrameView> = rt
        .lock_frames()
        .iter()
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
        .cloned()
        .collect();

    Ok(FramesResponse {
        frames,
        demo,
        source_configured,
        error,
    })
}

/// Return current bus statistics.
///
/// `stats` is `None` whenever no source is configured or a configured live
/// source is currently unreachable — callers must not treat that as "zero
/// traffic".
#[tauri::command]
pub fn get_bus_stats() -> AppResult<BusStatsResponse> {
    let rt = runtime();
    let uptime_secs = rt.started_at.elapsed().as_secs().max(1);
    let frame_count = rt.frame_count.load(Ordering::Relaxed);
    let error_count = rt.error_count.load(Ordering::Relaxed);

    match &rt.mode {
        SourceMode::Unconfigured => Ok(BusStatsResponse {
            stats: None,
            demo: false,
            source_configured: false,
            error: Some(NO_SOURCE_CONFIGURED_MESSAGE.to_string()),
        }),
        SourceMode::Demo => Ok(BusStatsResponse {
            stats: Some(BusStats {
                frame_count,
                error_count,
                // Representative constant for the static demo frame set;
                // always paired with `demo: true` so the UI cannot mistake
                // it for a live bus-load measurement.
                bus_load_pct: 23.4,
                frames_per_sec: frame_count as f64 / uptime_secs as f64,
                uptime_secs,
            }),
            demo: true,
            source_configured: true,
            error: None,
        }),
        SourceMode::Live { interface } => {
            let error = rt.lock_last_error().clone();
            if let Some(e) = error {
                Ok(BusStatsResponse {
                    stats: None,
                    demo: false,
                    source_configured: false,
                    error: Some(format!("live CAN source '{interface}': {e}")),
                })
            } else {
                Ok(BusStatsResponse {
                    stats: Some(BusStats {
                        frame_count,
                        error_count,
                        // Real bus-load percentage requires bit-timing
                        // configuration this monitor does not track;
                        // reporting 0 rather than a fabricated estimate.
                        bus_load_pct: 0.0,
                        frames_per_sec: frame_count as f64 / uptime_secs as f64,
                        uptime_secs,
                    }),
                    demo: false,
                    source_configured: true,
                    error: None,
                })
            }
        }
    }
}

/// Look up a PGN by number and return its name and description. This is a
/// static reference table (the J1939 PGN registry), not vehicle telemetry.
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

/// Clear the in-memory frame buffer.
#[tauri::command]
pub fn clear_frames() -> AppResult<()> {
    let rt = runtime();
    rt.lock_frames().clear();
    rt.frame_count.store(0, Ordering::Relaxed);
    Ok(())
}

// ---------------------------------------------------------------------------
// Static PGN reference table
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

// ---------------------------------------------------------------------------
// Demo-mode data (only served when OXIRS_CANBUS_DEMO=1 is explicitly set)
// ---------------------------------------------------------------------------

fn mock_frames() -> Vec<CanFrameView> {
    vec![
        CanFrameView {
            id: "demo_f001".to_string(),
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
            id: "demo_f002".to_string(),
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
            id: "demo_f003".to_string(),
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
            id: "demo_f004".to_string(),
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
            id: "demo_f005".to_string(),
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

    // NOTE: `runtime()` is a process-wide `OnceLock`, so its `SourceMode` is
    // fixed by whatever `OXIRS_CANBUS_INTERFACE`/`OXIRS_CANBUS_DEMO` are set
    // to the first time any test in this binary touches it. This test binary
    // runs with neither set, so every test below observes `Unconfigured`.

    #[test]
    fn test_get_frames_unconfigured_returns_no_source() {
        let resp = get_frames(None, None).expect("get_frames");
        assert!(!resp.source_configured);
        assert!(!resp.demo);
        assert!(resp.frames.is_empty());
        assert!(
            resp.error.is_some(),
            "must surface an explicit no-source state"
        );
    }

    #[test]
    fn test_get_bus_stats_unconfigured_returns_none() {
        let resp = get_bus_stats().expect("get_bus_stats");
        assert!(!resp.source_configured);
        assert!(resp.stats.is_none());
        assert!(
            resp.error.is_some(),
            "must surface an explicit no-source state"
        );
    }

    #[test]
    fn test_clear_frames_ok() {
        assert!(clear_frames().is_ok());
    }

    #[test]
    fn test_lookup_pgn_known() {
        let r = lookup_pgn(61_444);
        assert!(r.is_some());
        let (name, _) = r.expect("some");
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
        let json = serde_json::to_string(&f).expect("serialize");
        let back: CanFrameView = serde_json::from_str(&json).expect("deserialize");
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
        let json = serde_json::to_string(&f).expect("serialize");
        let back: CanFrameView = serde_json::from_str(&json).expect("deserialize");
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
        let json = serde_json::to_string(&s).expect("serialize");
        assert!(json.contains("Temp"));
        let back: SignalView = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.spn, 110);
        assert!((back.value - 88.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frame_filter_default_is_no_filter() {
        let filter = FrameFilter::default();
        assert!(filter.pgn.is_none());
        assert!(filter.sa.is_none());
        assert!(filter.min_timestamp_us.is_none());
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
    fn test_lookup_pgn_et1() {
        let r = lookup_pgn(65_262);
        assert!(r.is_some());
        let (name, desc) = r.expect("some");
        assert_eq!(name, "ET1");
        assert!(desc.contains("Temperature"));
    }

    #[test]
    fn test_mock_frames_are_only_used_in_demo_mode_data() {
        // Demo data must never claim to be `demo_` frames served outside an
        // explicit demo/no-source response; this just verifies the fixture
        // itself is well-formed and clearly labeled.
        let frames = mock_frames();
        assert!(!frames.is_empty());
        assert!(frames.iter().all(|f| f.id.starts_with("demo_")));
    }

    #[test]
    fn test_frames_response_serialization_round_trip() {
        let resp = FramesResponse {
            frames: vec![],
            demo: true,
            source_configured: true,
            error: None,
        };
        let json = serde_json::to_string(&resp).expect("serialize");
        let back: FramesResponse = serde_json::from_str(&json).expect("deserialize");
        assert!(back.demo);
        assert!(back.source_configured);
    }

    #[test]
    fn test_bus_stats_response_serialization_round_trip() {
        let resp = BusStatsResponse {
            stats: None,
            demo: false,
            source_configured: false,
            error: Some("no CAN source configured".to_string()),
        };
        let json = serde_json::to_string(&resp).expect("serialize");
        let back: BusStatsResponse = serde_json::from_str(&json).expect("deserialize");
        assert!(!back.source_configured);
        assert_eq!(back.error.as_deref(), Some("no CAN source configured"));
    }
}
