//! OBD-II PID (Parameter ID) decoder for automotive diagnostics.
//!
//! Implements SAE J1979 Mode 01 current-data PID decoding.  Raw bytes from an
//! OBD-II ECU response are converted to typed physical values using the
//! standard formulas defined in SAE J1979 / ISO 15031-5.
//!
//! This module uses distinct type names prefixed with `Pid` / `Pids` to avoid
//! collisions with the broader `obd2` module already present in the crate.

use std::fmt;

// ── Error ────────────────────────────────────────────────────────────────────

/// Errors produced during OBD-II PID decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObdError {
    /// The PID byte does not match any known Mode 01 formula.
    UnknownPid(u8),
    /// Not enough data bytes were supplied for the requested PID.
    InsufficientData {
        /// Bytes required by the formula.
        needed: usize,
        /// Bytes actually provided.
        got: usize,
    },
    /// The supplied data is structurally correct but semantically invalid.
    InvalidData(String),
}

impl fmt::Display for ObdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownPid(pid) => write!(f, "unknown OBD-II PID: 0x{pid:02X}"),
            Self::InsufficientData { needed, got } => {
                write!(f, "insufficient data: need {needed} byte(s), got {got}")
            }
            Self::InvalidData(msg) => write!(f, "invalid OBD data: {msg}"),
        }
    }
}

impl std::error::Error for ObdError {}

// ── ObdPid ────────────────────────────────────────────────────────────────────

/// Common Mode 01 PID identifiers, named according to SAE J1979.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ObdPid {
    /// PID 0x0C — Engine RPM.
    EngineRpm,
    /// PID 0x0D — Vehicle speed in km/h.
    VehicleSpeed,
    /// PID 0x05 — Engine coolant temperature.
    CoolantTemp,
    /// PID 0x04 — Calculated engine load.
    EngineLoad,
    /// PID 0x11 — Throttle position.
    ThrottlePosition,
    /// PID 0x0A — Fuel pressure (gauge).
    FuelPressure,
    /// PID 0x0B — Intake manifold absolute pressure.
    IntakeManifoldPressure,
    /// PID 0x0E — Timing advance.
    TimingAdvance,
    /// PID 0x10 — MAF air flow rate.
    AirflowRate,
    /// PID 0x2F — Fuel tank level.
    FuelLevel,
    /// PID 0x14 — Oxygen sensor voltage (bank 1, sensor 1).
    OxygenSensorVoltage,
    /// Any PID not covered by the known list.
    Unknown(u8),
}

impl ObdPid {
    /// Convert a raw PID byte to an `ObdPid` variant.
    pub fn from_byte(pid: u8) -> Self {
        match pid {
            0x0C => Self::EngineRpm,
            0x0D => Self::VehicleSpeed,
            0x05 => Self::CoolantTemp,
            0x04 => Self::EngineLoad,
            0x11 => Self::ThrottlePosition,
            0x0A => Self::FuelPressure,
            0x0B => Self::IntakeManifoldPressure,
            0x0E => Self::TimingAdvance,
            0x10 => Self::AirflowRate,
            0x2F => Self::FuelLevel,
            0x14 => Self::OxygenSensorVoltage,
            other => Self::Unknown(other),
        }
    }

    /// Return the raw PID byte for this variant.
    pub fn to_byte(&self) -> u8 {
        match self {
            Self::EngineRpm => 0x0C,
            Self::VehicleSpeed => 0x0D,
            Self::CoolantTemp => 0x05,
            Self::EngineLoad => 0x04,
            Self::ThrottlePosition => 0x11,
            Self::FuelPressure => 0x0A,
            Self::IntakeManifoldPressure => 0x0B,
            Self::TimingAdvance => 0x0E,
            Self::AirflowRate => 0x10,
            Self::FuelLevel => 0x2F,
            Self::OxygenSensorVoltage => 0x14,
            Self::Unknown(b) => *b,
        }
    }
}

// ── ObdValue ─────────────────────────────────────────────────────────────────

/// Typed physical value decoded from a raw OBD-II response.
#[derive(Debug, Clone, PartialEq)]
pub enum ObdValue {
    /// Engine speed in revolutions per minute.
    Rpm(f64),
    /// Vehicle speed in kilometres per hour.
    SpeedKph(f64),
    /// Temperature in degrees Celsius.
    TempCelsius(f64),
    /// Dimensionless percentage (0 – 100 %).
    Percent(f64),
    /// Pressure in kilopascals.
    KilopascalPressure(f64),
    /// Mass flow rate in grams per second.
    GramsPerSecond(f64),
    /// Angle in degrees.
    Degrees(f64),
    /// Voltage in volts.
    Volts(f64),
    /// Raw byte sequence for unknown or undecodable PIDs.
    Raw(Vec<u8>),
}

// ── ObdResponse ───────────────────────────────────────────────────────────────

/// A fully decoded OBD-II Mode 01 response.
#[derive(Debug)]
pub struct ObdResponse {
    /// Decoded PID identifier.
    pub pid: ObdPid,
    /// Original raw bytes from the ECU (excluding service byte).
    pub raw: Vec<u8>,
    /// Physical value derived from the raw bytes.
    pub value: ObdValue,
}

// ── ObdDecoder ────────────────────────────────────────────────────────────────

/// Stateless OBD-II Mode 01 PID decoder.
pub struct ObdDecoder;

impl ObdDecoder {
    /// Decode a Mode 01 PID response.
    ///
    /// `pid_byte` is the raw PID byte (0x00 – 0xFF).
    /// `data` contains the data bytes from the ECU response, **not** including
    /// the service or PID bytes.
    ///
    /// # Errors
    ///
    /// Returns `ObdError::UnknownPid` if the PID has no known formula.
    /// Returns `ObdError::InsufficientData` if fewer bytes are provided than
    /// the formula requires.
    pub fn decode(pid_byte: u8, data: &[u8]) -> Result<ObdResponse, ObdError> {
        let pid = ObdPid::from_byte(pid_byte);
        let raw = data.to_vec();

        let value = match &pid {
            ObdPid::EngineRpm => {
                // Formula: (A*256 + B) / 4  [RPM]  — 2 bytes required.
                Self::require_bytes(data, 2)?;
                let a = data[0] as f64;
                let b = data[1] as f64;
                ObdValue::Rpm((a * 256.0 + b) / 4.0)
            }
            ObdPid::VehicleSpeed => {
                // Formula: A  [km/h]  — 1 byte required.
                Self::require_bytes(data, 1)?;
                ObdValue::SpeedKph(data[0] as f64)
            }
            ObdPid::CoolantTemp => {
                // Formula: A - 40  [°C]  — 1 byte required.
                Self::require_bytes(data, 1)?;
                ObdValue::TempCelsius(data[0] as f64 - 40.0)
            }
            ObdPid::EngineLoad => {
                // Formula: A * 100 / 255  [%]  — 1 byte required.
                Self::require_bytes(data, 1)?;
                ObdValue::Percent(data[0] as f64 * 100.0 / 255.0)
            }
            ObdPid::ThrottlePosition => {
                // Formula: A * 100 / 255  [%]  — 1 byte required.
                Self::require_bytes(data, 1)?;
                ObdValue::Percent(data[0] as f64 * 100.0 / 255.0)
            }
            ObdPid::FuelPressure => {
                // Formula: A * 3  [kPa]  — 1 byte required.
                Self::require_bytes(data, 1)?;
                ObdValue::KilopascalPressure(data[0] as f64 * 3.0)
            }
            ObdPid::IntakeManifoldPressure => {
                // Formula: A  [kPa]  — 1 byte required.
                Self::require_bytes(data, 1)?;
                ObdValue::KilopascalPressure(data[0] as f64)
            }
            ObdPid::TimingAdvance => {
                // Formula: (A / 2) - 64  [degrees]  — 1 byte required.
                Self::require_bytes(data, 1)?;
                ObdValue::Degrees(data[0] as f64 / 2.0 - 64.0)
            }
            ObdPid::AirflowRate => {
                // Formula: (A*256 + B) / 100  [g/s]  — 2 bytes required.
                Self::require_bytes(data, 2)?;
                let a = data[0] as f64;
                let b = data[1] as f64;
                ObdValue::GramsPerSecond((a * 256.0 + b) / 100.0)
            }
            ObdPid::FuelLevel => {
                // Formula: A * 100 / 255  [%]  — 1 byte required.
                Self::require_bytes(data, 1)?;
                ObdValue::Percent(data[0] as f64 * 100.0 / 255.0)
            }
            ObdPid::OxygenSensorVoltage => {
                // Formula: A * 0.005  [V]  — 1 byte required.
                Self::require_bytes(data, 1)?;
                ObdValue::Volts(data[0] as f64 * 0.005)
            }
            ObdPid::Unknown(b) => return Err(ObdError::UnknownPid(*b)),
        };

        Ok(ObdResponse { pid, raw, value })
    }

    /// Return the human-readable name for a PID.
    pub fn pid_name(pid: &ObdPid) -> &'static str {
        match pid {
            ObdPid::EngineRpm => "Engine RPM",
            ObdPid::VehicleSpeed => "Vehicle Speed",
            ObdPid::CoolantTemp => "Coolant Temperature",
            ObdPid::EngineLoad => "Engine Load",
            ObdPid::ThrottlePosition => "Throttle Position",
            ObdPid::FuelPressure => "Fuel Pressure",
            ObdPid::IntakeManifoldPressure => "Intake Manifold Pressure",
            ObdPid::TimingAdvance => "Timing Advance",
            ObdPid::AirflowRate => "MAF Air Flow Rate",
            ObdPid::FuelLevel => "Fuel Level",
            ObdPid::OxygenSensorVoltage => "O2 Sensor Voltage",
            ObdPid::Unknown(_) => "Unknown PID",
        }
    }

    /// Return the list of PID byte values that have a known decode formula.
    pub fn supported_pids() -> Vec<u8> {
        vec![
            0x04, // EngineLoad
            0x05, // CoolantTemp
            0x0A, // FuelPressure
            0x0B, // IntakeManifoldPressure
            0x0C, // EngineRpm
            0x0D, // VehicleSpeed
            0x0E, // TimingAdvance
            0x10, // AirflowRate
            0x11, // ThrottlePosition
            0x14, // OxygenSensorVoltage
            0x2F, // FuelLevel
        ]
    }

    // ── private helpers ────────────────────────────────────────────────────

    fn require_bytes(data: &[u8], needed: usize) -> Result<(), ObdError> {
        if data.len() < needed {
            Err(ObdError::InsufficientData {
                needed,
                got: data.len(),
            })
        } else {
            Ok(())
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ObdPid conversions ─────────────────────────────────────────────────

    #[test]
    fn test_pid_from_byte_engine_rpm() {
        assert_eq!(ObdPid::from_byte(0x0C), ObdPid::EngineRpm);
    }

    #[test]
    fn test_pid_from_byte_vehicle_speed() {
        assert_eq!(ObdPid::from_byte(0x0D), ObdPid::VehicleSpeed);
    }

    #[test]
    fn test_pid_from_byte_coolant_temp() {
        assert_eq!(ObdPid::from_byte(0x05), ObdPid::CoolantTemp);
    }

    #[test]
    fn test_pid_from_byte_engine_load() {
        assert_eq!(ObdPid::from_byte(0x04), ObdPid::EngineLoad);
    }

    #[test]
    fn test_pid_from_byte_throttle() {
        assert_eq!(ObdPid::from_byte(0x11), ObdPid::ThrottlePosition);
    }

    #[test]
    fn test_pid_from_byte_fuel_pressure() {
        assert_eq!(ObdPid::from_byte(0x0A), ObdPid::FuelPressure);
    }

    #[test]
    fn test_pid_from_byte_intake_map() {
        assert_eq!(ObdPid::from_byte(0x0B), ObdPid::IntakeManifoldPressure);
    }

    #[test]
    fn test_pid_from_byte_timing_advance() {
        assert_eq!(ObdPid::from_byte(0x0E), ObdPid::TimingAdvance);
    }

    #[test]
    fn test_pid_from_byte_airflow_rate() {
        assert_eq!(ObdPid::from_byte(0x10), ObdPid::AirflowRate);
    }

    #[test]
    fn test_pid_from_byte_fuel_level() {
        assert_eq!(ObdPid::from_byte(0x2F), ObdPid::FuelLevel);
    }

    #[test]
    fn test_pid_from_byte_o2_sensor() {
        assert_eq!(ObdPid::from_byte(0x14), ObdPid::OxygenSensorVoltage);
    }

    #[test]
    fn test_pid_from_byte_unknown() {
        assert_eq!(ObdPid::from_byte(0xFF), ObdPid::Unknown(0xFF));
    }

    #[test]
    fn test_pid_to_byte_round_trip() {
        let pids = ObdDecoder::supported_pids();
        for pid_byte in pids {
            let pid = ObdPid::from_byte(pid_byte);
            assert_eq!(pid.to_byte(), pid_byte);
        }
    }

    // ── Engine RPM decode ──────────────────────────────────────────────────

    #[test]
    fn test_decode_rpm_basic() {
        // (0x1A * 256 + 0xF0) / 4 = (6656 + 240) / 4 = 6896 / 4 = 1724
        let r = ObdDecoder::decode(0x0C, &[0x1A, 0xF0]).expect("ok");
        match r.value {
            ObdValue::Rpm(v) => assert!((v - 1724.0).abs() < 1e-9),
            _ => panic!("expected Rpm"),
        }
    }

    #[test]
    fn test_decode_rpm_zero() {
        let r = ObdDecoder::decode(0x0C, &[0x00, 0x00]).expect("ok");
        match r.value {
            ObdValue::Rpm(v) => assert_eq!(v, 0.0),
            _ => panic!("expected Rpm"),
        }
    }

    #[test]
    fn test_decode_rpm_max() {
        // (255 * 256 + 255) / 4 = 65535 / 4 = 16383.75
        let r = ObdDecoder::decode(0x0C, &[0xFF, 0xFF]).expect("ok");
        match r.value {
            ObdValue::Rpm(v) => assert!((v - 16383.75).abs() < 1e-9),
            _ => panic!("expected Rpm"),
        }
    }

    #[test]
    fn test_decode_rpm_insufficient_data() {
        let err = ObdDecoder::decode(0x0C, &[0x10]).unwrap_err();
        assert_eq!(err, ObdError::InsufficientData { needed: 2, got: 1 });
    }

    // ── Vehicle Speed ──────────────────────────────────────────────────────

    #[test]
    fn test_decode_speed_100kph() {
        let r = ObdDecoder::decode(0x0D, &[100]).expect("ok");
        match r.value {
            ObdValue::SpeedKph(v) => assert_eq!(v, 100.0),
            _ => panic!("expected SpeedKph"),
        }
    }

    #[test]
    fn test_decode_speed_zero() {
        let r = ObdDecoder::decode(0x0D, &[0]).expect("ok");
        match r.value {
            ObdValue::SpeedKph(v) => assert_eq!(v, 0.0),
            _ => panic!(),
        }
    }

    #[test]
    fn test_decode_speed_insufficient_data() {
        let err = ObdDecoder::decode(0x0D, &[]).unwrap_err();
        assert_eq!(err, ObdError::InsufficientData { needed: 1, got: 0 });
    }

    // ── Coolant Temperature ────────────────────────────────────────────────

    #[test]
    fn test_decode_coolant_temp_normal() {
        // A=0x68 (104) → 104 - 40 = 64 °C
        let r = ObdDecoder::decode(0x05, &[0x68]).expect("ok");
        match r.value {
            ObdValue::TempCelsius(v) => assert_eq!(v, 64.0),
            _ => panic!(),
        }
    }

    #[test]
    fn test_decode_coolant_temp_minus40() {
        // A=0 → 0 - 40 = -40 °C
        let r = ObdDecoder::decode(0x05, &[0x00]).expect("ok");
        match r.value {
            ObdValue::TempCelsius(v) => assert_eq!(v, -40.0),
            _ => panic!(),
        }
    }

    // ── Engine Load ────────────────────────────────────────────────────────

    #[test]
    fn test_decode_engine_load_100_percent() {
        // A=255 → 255 * 100 / 255 = 100.0 %
        let r = ObdDecoder::decode(0x04, &[0xFF]).expect("ok");
        match r.value {
            ObdValue::Percent(v) => assert!((v - 100.0).abs() < 1e-9),
            _ => panic!(),
        }
    }

    #[test]
    fn test_decode_engine_load_zero() {
        let r = ObdDecoder::decode(0x04, &[0x00]).expect("ok");
        match r.value {
            ObdValue::Percent(v) => assert_eq!(v, 0.0),
            _ => panic!(),
        }
    }

    // ── Throttle Position ──────────────────────────────────────────────────

    #[test]
    fn test_decode_throttle_half() {
        // A=127 → 127 * 100 / 255 ≈ 49.80 %
        let r = ObdDecoder::decode(0x11, &[127]).expect("ok");
        match r.value {
            ObdValue::Percent(v) => assert!((v - (127.0 * 100.0 / 255.0)).abs() < 1e-9),
            _ => panic!(),
        }
    }

    // ── Fuel Pressure ──────────────────────────────────────────────────────

    #[test]
    fn test_decode_fuel_pressure() {
        // A=50 → 50 * 3 = 150 kPa
        let r = ObdDecoder::decode(0x0A, &[50]).expect("ok");
        match r.value {
            ObdValue::KilopascalPressure(v) => assert_eq!(v, 150.0),
            _ => panic!(),
        }
    }

    // ── Intake Manifold Pressure ───────────────────────────────────────────

    #[test]
    fn test_decode_intake_map() {
        // A=101 → 101 kPa
        let r = ObdDecoder::decode(0x0B, &[101]).expect("ok");
        match r.value {
            ObdValue::KilopascalPressure(v) => assert_eq!(v, 101.0),
            _ => panic!(),
        }
    }

    // ── Timing Advance ─────────────────────────────────────────────────────

    #[test]
    fn test_decode_timing_advance_positive() {
        // A=128 → 128/2 - 64 = 0.0 degrees
        let r = ObdDecoder::decode(0x0E, &[128]).expect("ok");
        match r.value {
            ObdValue::Degrees(v) => assert_eq!(v, 0.0),
            _ => panic!(),
        }
    }

    #[test]
    fn test_decode_timing_advance_negative() {
        // A=0 → 0/2 - 64 = -64.0 degrees
        let r = ObdDecoder::decode(0x0E, &[0]).expect("ok");
        match r.value {
            ObdValue::Degrees(v) => assert_eq!(v, -64.0),
            _ => panic!(),
        }
    }

    #[test]
    fn test_decode_timing_advance_positive_64() {
        // A=255 → 255/2 - 64 = 127.5 - 64 = 63.5 degrees
        let r = ObdDecoder::decode(0x0E, &[255]).expect("ok");
        match r.value {
            ObdValue::Degrees(v) => assert!((v - 63.5).abs() < 1e-9),
            _ => panic!(),
        }
    }

    // ── MAF Airflow Rate ───────────────────────────────────────────────────

    #[test]
    fn test_decode_airflow_rate() {
        // (0x01, 0x90) → (256 + 144) / 100 = 400 / 100 = 4.0 g/s
        let r = ObdDecoder::decode(0x10, &[0x01, 0x90]).expect("ok");
        match r.value {
            ObdValue::GramsPerSecond(v) => assert!((v - 4.0).abs() < 1e-9),
            _ => panic!(),
        }
    }

    #[test]
    fn test_decode_airflow_rate_insufficient() {
        let err = ObdDecoder::decode(0x10, &[0x01]).unwrap_err();
        assert_eq!(err, ObdError::InsufficientData { needed: 2, got: 1 });
    }

    // ── Fuel Level ─────────────────────────────────────────────────────────

    #[test]
    fn test_decode_fuel_level_full() {
        let r = ObdDecoder::decode(0x2F, &[0xFF]).expect("ok");
        match r.value {
            ObdValue::Percent(v) => assert!((v - 100.0).abs() < 1e-9),
            _ => panic!(),
        }
    }

    #[test]
    fn test_decode_fuel_level_half() {
        // A=127 → ~49.80 %
        let r = ObdDecoder::decode(0x2F, &[127]).expect("ok");
        match r.value {
            ObdValue::Percent(v) => assert!((v - (127.0 * 100.0 / 255.0)).abs() < 1e-9),
            _ => panic!(),
        }
    }

    // ── O2 Sensor Voltage ──────────────────────────────────────────────────

    #[test]
    fn test_decode_o2_voltage() {
        // A=200 → 200 * 0.005 = 1.0 V
        let r = ObdDecoder::decode(0x14, &[200]).expect("ok");
        match r.value {
            ObdValue::Volts(v) => assert!((v - 1.0).abs() < 1e-9),
            _ => panic!(),
        }
    }

    #[test]
    fn test_decode_o2_voltage_zero() {
        let r = ObdDecoder::decode(0x14, &[0]).expect("ok");
        match r.value {
            ObdValue::Volts(v) => assert_eq!(v, 0.0),
            _ => panic!(),
        }
    }

    // ── Unknown PID ────────────────────────────────────────────────────────

    #[test]
    fn test_decode_unknown_pid_returns_error() {
        let err = ObdDecoder::decode(0x99, &[0x00]).unwrap_err();
        assert_eq!(err, ObdError::UnknownPid(0x99));
    }

    #[test]
    fn test_decode_pid_00_unknown() {
        // PID 0x00 is not in our decode list (supported-PIDs bitmap).
        let err = ObdDecoder::decode(0x00, &[0xFF, 0xFF, 0xFF, 0xFF]).unwrap_err();
        assert_eq!(err, ObdError::UnknownPid(0x00));
    }

    // ── pid_name ───────────────────────────────────────────────────────────

    #[test]
    fn test_pid_name_all_known() {
        let supported = ObdDecoder::supported_pids();
        for pid_byte in supported {
            let pid = ObdPid::from_byte(pid_byte);
            let name = ObdDecoder::pid_name(&pid);
            assert!(!name.is_empty());
            assert_ne!(name, "Unknown PID");
        }
    }

    #[test]
    fn test_pid_name_unknown() {
        let pid = ObdPid::Unknown(0xAB);
        assert_eq!(ObdDecoder::pid_name(&pid), "Unknown PID");
    }

    // ── supported_pids ─────────────────────────────────────────────────────

    #[test]
    fn test_supported_pids_list_non_empty() {
        let pids = ObdDecoder::supported_pids();
        assert!(!pids.is_empty());
    }

    #[test]
    fn test_supported_pids_includes_rpm() {
        assert!(ObdDecoder::supported_pids().contains(&0x0C));
    }

    #[test]
    fn test_supported_pids_includes_speed() {
        assert!(ObdDecoder::supported_pids().contains(&0x0D));
    }

    #[test]
    fn test_supported_pids_count() {
        // We support exactly 11 PIDs.
        assert_eq!(ObdDecoder::supported_pids().len(), 11);
    }

    #[test]
    fn test_all_supported_pids_decode_with_sufficient_data() {
        for pid_byte in ObdDecoder::supported_pids() {
            // Provide 4 bytes — more than any single-formula requires.
            let result = ObdDecoder::decode(pid_byte, &[0x64, 0x64, 0x00, 0x00]);
            assert!(result.is_ok(), "PID 0x{pid_byte:02X} failed to decode");
        }
    }

    // ── raw field ──────────────────────────────────────────────────────────

    #[test]
    fn test_response_raw_preserved() {
        let data = &[0x1A, 0xF0];
        let r = ObdDecoder::decode(0x0C, data).expect("ok");
        assert_eq!(r.raw, data.to_vec());
    }

    // ── ObdError Display ───────────────────────────────────────────────────

    #[test]
    fn test_error_unknown_pid_display() {
        let e = ObdError::UnknownPid(0x42);
        assert!(e.to_string().contains("0x42"));
    }

    #[test]
    fn test_error_insufficient_data_display() {
        let e = ObdError::InsufficientData { needed: 2, got: 1 };
        assert!(e.to_string().contains("2"));
        assert!(e.to_string().contains("1"));
    }

    #[test]
    fn test_error_invalid_data_display() {
        let e = ObdError::InvalidData("bad checksum".to_string());
        assert!(e.to_string().contains("bad checksum"));
    }
}
