//! OBD-II (On-Board Diagnostics) diagnostic support
//!
//! Implements SAE J1979 (OBD-II) and ISO 15031-5 protocols for passenger vehicle
//! diagnostics. Supports Mode 01 (current data), Mode 03 (stored DTCs), and
//! Mode 07 (pending DTCs).
//!
//! # OBD-II Architecture
//!
//! OBD-II uses a request/response protocol over CAN (ISO 15765-4):
//! - Request: 0x7DF (functional) or 0x7E0-0x7E7 (physical)
//! - Response: 0x7E8-0x7EF
//!
//! # Example
//!
//! ```
//! use oxirs_canbus::obd2::{ObdDecoder, ObdRequest, ObdResponse, ObdPid, ObdValue};
//!
//! let response = ObdResponse { pid: 0x0C, data: vec![0x1A, 0xF0] };
//! let value = response.decode();
//! // Engine RPM = (256*0x1A + 0xF0) / 4 = 1724 RPM
//! if let ObdValue::Rpm(rpm) = value {
//!     println!("Engine RPM: {:.1}", rpm);
//! }
//! ```

use std::collections::HashMap;

/// OBD-II Service / Mode identifiers (SAE J1979)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ObdService {
    /// Mode 01: Show current data
    CurrentData = 0x01,
    /// Mode 02: Show freeze frame data
    FreezeFrameData = 0x02,
    /// Mode 03: Show stored DTCs
    StoredDtcs = 0x03,
    /// Mode 04: Clear DTCs and reset MIL
    ClearDtcs = 0x04,
    /// Mode 05: Test results, oxygen sensors (non-CAN)
    OxygenSensorTests = 0x05,
    /// Mode 06: Test results, other components
    OnboardTests = 0x06,
    /// Mode 07: Show pending DTCs
    PendingDtcs = 0x07,
    /// Mode 08: Control operation of on-board components
    ControlComponents = 0x08,
    /// Mode 09: Request vehicle information
    VehicleInfo = 0x09,
    /// Mode 0A: Permanent DTCs (CARB)
    PermanentDtcs = 0x0A,
}

impl ObdService {
    /// Convert from raw u8 byte
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::CurrentData),
            0x02 => Some(Self::FreezeFrameData),
            0x03 => Some(Self::StoredDtcs),
            0x04 => Some(Self::ClearDtcs),
            0x05 => Some(Self::OxygenSensorTests),
            0x06 => Some(Self::OnboardTests),
            0x07 => Some(Self::PendingDtcs),
            0x08 => Some(Self::ControlComponents),
            0x09 => Some(Self::VehicleInfo),
            0x0A => Some(Self::PermanentDtcs),
            _ => None,
        }
    }
}

/// Known OBD-II PIDs with their decoding formulas (Mode 01)
///
/// Based on SAE J1979 / ISO 15031-5
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ObdPid {
    /// PID 0x00: Supported PIDs 01-20 (bitfield)
    SupportedPids0120 = 0x00,
    /// PID 0x04: Calculated engine load (A/2.55 %)
    EngineLoad = 0x04,
    /// PID 0x05: Engine coolant temperature (A - 40 °C)
    CoolantTemp = 0x05,
    /// PID 0x06: Short-term fuel trim, bank 1 ((A - 128) * 100/128 %)
    ShortFuelTrimBank1 = 0x06,
    /// PID 0x07: Long-term fuel trim, bank 1 ((A - 128) * 100/128 %)
    LongFuelTrimBank1 = 0x07,
    /// PID 0x0B: Intake manifold pressure (A kPa absolute)
    IntakePressure = 0x0B,
    /// PID 0x0C: Engine RPM ((256*A + B) / 4 RPM)
    EngineRpm = 0x0C,
    /// PID 0x0D: Vehicle speed (A km/h)
    VehicleSpeed = 0x0D,
    /// PID 0x0E: Timing advance ((A - 128) / 2 degrees)
    TimingAdvance = 0x0E,
    /// PID 0x0F: Intake air temperature (A - 40 °C)
    IntakeAirTemp = 0x0F,
    /// PID 0x10: MAF air flow rate ((256*A + B) / 100 g/s)
    MafFlowRate = 0x10,
    /// PID 0x11: Throttle position (A/2.55 %)
    ThrottlePosition = 0x11,
    /// PID 0x1C: OBD standard this vehicle conforms to
    ObdStandard = 0x1C,
    /// PID 0x1F: Run time since engine start (256*A + B seconds)
    RuntimeSinceStart = 0x1F,
    /// PID 0x21: Distance traveled with MIL on (256*A + B km)
    DistanceWithMil = 0x21,
    /// PID 0x2F: Fuel tank level input (A/2.55 %)
    FuelTankLevel = 0x2F,
    /// PID 0x33: Absolute barometric pressure (A kPa)
    BarometricPressure = 0x33,
    /// PID 0x46: Ambient air temperature (A - 40 °C)
    AmbientAirTemp = 0x46,
    /// PID 0x47: Absolute throttle position B (A/2.55 %)
    ThrottlePositionB = 0x47,
    /// PID 0x49: Accelerator pedal position D (A/2.55 %)
    AccelPedalPosD = 0x49,
    /// PID 0x4A: Accelerator pedal position E (A/2.55 %)
    AccelPedalPosE = 0x4A,
    /// PID 0x4C: Commanded throttle actuator (A/2.55 %)
    ThrottleActuator = 0x4C,
    /// PID 0x51: Fuel type
    FuelType = 0x51,
    /// PID 0x5C: Engine oil temperature (A - 40 °C)
    EngineOilTemp = 0x5C,
}

impl ObdPid {
    /// Convert from raw PID byte
    pub fn from_byte(pid: u8) -> Option<Self> {
        match pid {
            0x00 => Some(Self::SupportedPids0120),
            0x04 => Some(Self::EngineLoad),
            0x05 => Some(Self::CoolantTemp),
            0x06 => Some(Self::ShortFuelTrimBank1),
            0x07 => Some(Self::LongFuelTrimBank1),
            0x0B => Some(Self::IntakePressure),
            0x0C => Some(Self::EngineRpm),
            0x0D => Some(Self::VehicleSpeed),
            0x0E => Some(Self::TimingAdvance),
            0x0F => Some(Self::IntakeAirTemp),
            0x10 => Some(Self::MafFlowRate),
            0x11 => Some(Self::ThrottlePosition),
            0x1C => Some(Self::ObdStandard),
            0x1F => Some(Self::RuntimeSinceStart),
            0x21 => Some(Self::DistanceWithMil),
            0x2F => Some(Self::FuelTankLevel),
            0x33 => Some(Self::BarometricPressure),
            0x46 => Some(Self::AmbientAirTemp),
            0x47 => Some(Self::ThrottlePositionB),
            0x49 => Some(Self::AccelPedalPosD),
            0x4A => Some(Self::AccelPedalPosE),
            0x4C => Some(Self::ThrottleActuator),
            0x51 => Some(Self::FuelType),
            0x5C => Some(Self::EngineOilTemp),
            _ => None,
        }
    }

    /// Get human-readable name for this PID
    pub fn name(&self) -> &'static str {
        match self {
            Self::SupportedPids0120 => "Supported PIDs 01-20",
            Self::EngineLoad => "Calculated Engine Load",
            Self::CoolantTemp => "Engine Coolant Temperature",
            Self::ShortFuelTrimBank1 => "Short-Term Fuel Trim Bank 1",
            Self::LongFuelTrimBank1 => "Long-Term Fuel Trim Bank 1",
            Self::IntakePressure => "Intake Manifold Absolute Pressure",
            Self::EngineRpm => "Engine RPM",
            Self::VehicleSpeed => "Vehicle Speed",
            Self::TimingAdvance => "Timing Advance",
            Self::IntakeAirTemp => "Intake Air Temperature",
            Self::MafFlowRate => "MAF Air Flow Rate",
            Self::ThrottlePosition => "Throttle Position",
            Self::ObdStandard => "OBD Standard",
            Self::RuntimeSinceStart => "Run Time Since Engine Start",
            Self::DistanceWithMil => "Distance with MIL On",
            Self::FuelTankLevel => "Fuel Tank Level",
            Self::BarometricPressure => "Barometric Pressure",
            Self::AmbientAirTemp => "Ambient Air Temperature",
            Self::ThrottlePositionB => "Absolute Throttle Position B",
            Self::AccelPedalPosD => "Accelerator Pedal Position D",
            Self::AccelPedalPosE => "Accelerator Pedal Position E",
            Self::ThrottleActuator => "Commanded Throttle Actuator",
            Self::FuelType => "Fuel Type",
            Self::EngineOilTemp => "Engine Oil Temperature",
        }
    }

    /// Get unit string for this PID
    pub fn unit(&self) -> &'static str {
        match self {
            Self::SupportedPids0120 => "bitfield",
            Self::EngineLoad => "%",
            Self::CoolantTemp => "°C",
            Self::ShortFuelTrimBank1 | Self::LongFuelTrimBank1 => "%",
            Self::IntakePressure | Self::BarometricPressure => "kPa",
            Self::EngineRpm => "RPM",
            Self::VehicleSpeed => "km/h",
            Self::TimingAdvance => "° before TDC",
            Self::IntakeAirTemp | Self::AmbientAirTemp | Self::EngineOilTemp => "°C",
            Self::MafFlowRate => "g/s",
            Self::ThrottlePosition
            | Self::ThrottlePositionB
            | Self::AccelPedalPosD
            | Self::AccelPedalPosE
            | Self::ThrottleActuator => "%",
            Self::ObdStandard | Self::FuelType => "",
            Self::RuntimeSinceStart => "s",
            Self::DistanceWithMil => "km",
            Self::FuelTankLevel => "%",
        }
    }
}

/// Decoded OBD-II physical value
#[derive(Debug, Clone, PartialEq)]
pub enum ObdValue {
    /// Percentage value (0.0 - 100.0)
    Percent(f64),
    /// Temperature in degrees Celsius
    Temperature(f64),
    /// Speed in km/h
    Speed(f64),
    /// Engine speed in RPM
    Rpm(f64),
    /// Mass air flow in grams/second
    Grams(f64),
    /// Distance in kilometers
    Km(f64),
    /// Pressure in kPa
    Pressure(f64),
    /// Time in seconds
    Seconds(f64),
    /// OBD standard identifier
    Standard(u8),
    /// Raw uninterpreted bytes
    Raw(Vec<u8>),
}

impl ObdValue {
    /// Get a human-readable string representation
    pub fn to_display_string(&self, unit: &str) -> String {
        match self {
            Self::Percent(v) => format!("{:.2} {}", v, unit),
            Self::Temperature(v) => format!("{:.1} {}", v, unit),
            Self::Speed(v) => format!("{:.1} {}", v, unit),
            Self::Rpm(v) => format!("{:.0} {}", v, unit),
            Self::Grams(v) => format!("{:.2} {}", v, unit),
            Self::Km(v) => format!("{:.0} {}", v, unit),
            Self::Pressure(v) => format!("{:.1} {}", v, unit),
            Self::Seconds(v) => format!("{:.0} {}", v, unit),
            Self::Standard(v) => format!("Standard {}", v),
            Self::Raw(bytes) => {
                let hex: Vec<String> = bytes.iter().map(|b| format!("{:02X}", b)).collect();
                hex.join(" ")
            }
        }
    }
}

/// An OBD-II diagnostic request
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObdRequest {
    /// Service / Mode (e.g. 0x01 = current data)
    pub service: u8,
    /// Parameter ID
    pub pid: u8,
}

impl ObdRequest {
    /// Create a new OBD-II request
    pub fn new(service: u8, pid: u8) -> Self {
        Self { service, pid }
    }

    /// Create a Mode 01 (current data) request
    pub fn mode01(pid: u8) -> Self {
        Self {
            service: ObdService::CurrentData as u8,
            pid,
        }
    }

    /// Encode this request to ISO-TP / CAN data bytes
    /// Returns [length, service, pid] (3-byte payload)
    pub fn encode(&self) -> Vec<u8> {
        vec![0x02, self.service, self.pid]
    }
}

/// An OBD-II diagnostic response
#[derive(Debug, Clone, PartialEq)]
pub struct ObdResponse {
    /// Parameter ID that was requested
    pub pid: u8,
    /// Raw response data bytes (A, B, C, D)
    pub data: Vec<u8>,
}

impl ObdResponse {
    /// Create a new OBD-II response
    pub fn new(pid: u8, data: Vec<u8>) -> Self {
        Self { pid, data }
    }

    /// Parse from raw ISO-TP response bytes
    ///
    /// Format: [length, 0x41 (response = service + 0x40), pid, A, B, C, D, ...]
    pub fn from_bytes(raw: &[u8]) -> Option<Self> {
        // Minimum: length byte + 0x41 + PID + at least 1 data byte
        if raw.len() < 4 {
            return None;
        }
        // Response service byte should be request service + 0x40
        if raw[1] != 0x41 {
            return None;
        }
        let pid = raw[2];
        let data = raw[3..].to_vec();
        Some(Self { pid, data })
    }

    /// Decode the response data into a physical value
    pub fn decode(&self) -> ObdValue {
        let a = self.data.first().copied().unwrap_or(0);
        let b = self.data.get(1).copied().unwrap_or(0);

        match self.pid {
            // PID 0x00: Supported PIDs bitfield - return raw
            0x00 => ObdValue::Raw(self.data.clone()),

            // PID 0x04: Engine load = A / 2.55 %
            0x04 => ObdValue::Percent(a as f64 / 2.55),

            // PID 0x05: Coolant temperature = A - 40 °C
            0x05 => ObdValue::Temperature(a as f64 - 40.0),

            // PID 0x06: Short-term fuel trim = (A - 128) * 100/128 %
            0x06 => ObdValue::Percent((a as f64 - 128.0) * 100.0 / 128.0),

            // PID 0x07: Long-term fuel trim = (A - 128) * 100/128 %
            0x07 => ObdValue::Percent((a as f64 - 128.0) * 100.0 / 128.0),

            // PID 0x0B: Intake manifold pressure = A kPa
            0x0B => ObdValue::Pressure(a as f64),

            // PID 0x0C: Engine RPM = (256*A + B) / 4
            0x0C => ObdValue::Rpm((256.0 * a as f64 + b as f64) / 4.0),

            // PID 0x0D: Vehicle speed = A km/h
            0x0D => ObdValue::Speed(a as f64),

            // PID 0x0E: Timing advance = (A - 128) / 2 degrees
            0x0E => ObdValue::Raw(vec![((a as f64 - 128.0) / 2.0) as u8]),

            // PID 0x0F: Intake air temperature = A - 40 °C
            0x0F => ObdValue::Temperature(a as f64 - 40.0),

            // PID 0x10: MAF air flow rate = (256*A + B) / 100 g/s
            0x10 => ObdValue::Grams((256.0 * a as f64 + b as f64) / 100.0),

            // PID 0x11: Throttle position = A / 2.55 %
            0x11 => ObdValue::Percent(a as f64 / 2.55),

            // PID 0x1C: OBD standard
            0x1C => ObdValue::Standard(a),

            // PID 0x1F: Runtime since start = 256*A + B seconds
            0x1F => ObdValue::Seconds(256.0 * a as f64 + b as f64),

            // PID 0x21: Distance with MIL on = 256*A + B km
            0x21 => ObdValue::Km(256.0 * a as f64 + b as f64),

            // PID 0x2F: Fuel tank level = A / 2.55 %
            0x2F => ObdValue::Percent(a as f64 / 2.55),

            // PID 0x33: Barometric pressure = A kPa
            0x33 => ObdValue::Pressure(a as f64),

            // PID 0x46: Ambient air temperature = A - 40 °C
            0x46 => ObdValue::Temperature(a as f64 - 40.0),

            // PID 0x47: Absolute throttle position B = A / 2.55 %
            0x47 => ObdValue::Percent(a as f64 / 2.55),

            // PID 0x49: Accelerator pedal position D = A / 2.55 %
            0x49 => ObdValue::Percent(a as f64 / 2.55),

            // PID 0x4A: Accelerator pedal position E = A / 2.55 %
            0x4A => ObdValue::Percent(a as f64 / 2.55),

            // PID 0x4C: Commanded throttle actuator = A / 2.55 %
            0x4C => ObdValue::Percent(a as f64 / 2.55),

            // PID 0x51: Fuel type
            0x51 => ObdValue::Standard(a),

            // PID 0x5C: Engine oil temperature = A - 40 °C
            0x5C => ObdValue::Temperature(a as f64 - 40.0),

            // Unknown PID: return raw bytes
            _ => ObdValue::Raw(self.data.clone()),
        }
    }

    /// Check if this is a positive response (service 0x41 = 0x01 + 0x40)
    pub fn is_positive(&self) -> bool {
        // If we have data, it's a positive response
        !self.data.is_empty()
    }
}

/// OBD-II decoder — provides higher-level API over raw responses
#[derive(Debug, Default)]
pub struct ObdDecoder {
    /// Cache of known supported PIDs per ECU
    supported_pids: HashMap<u8, u32>,
}

impl ObdDecoder {
    /// Create a new OBD decoder
    pub fn new() -> Self {
        Self {
            supported_pids: HashMap::new(),
        }
    }

    /// Decode an OBD-II response to a physical value
    pub fn decode(&self, response: &ObdResponse) -> ObdValue {
        response.decode()
    }

    /// Process a PID 0x00 (supported PIDs) response and cache it
    pub fn process_supported_pids(&mut self, response: &ObdResponse) {
        if response.pid == 0x00 && response.data.len() >= 4 {
            let supported = ((response.data[0] as u32) << 24)
                | ((response.data[1] as u32) << 16)
                | ((response.data[2] as u32) << 8)
                | (response.data[3] as u32);
            self.supported_pids.insert(0x00, supported);
        }
    }

    /// Check if a PID is supported (based on cached PID 0x00 response)
    pub fn is_pid_supported(&self, pid: u8) -> Option<bool> {
        if pid == 0x00 || pid > 0x20 {
            return None;
        }
        self.supported_pids.get(&0x00).map(|&bits| {
            let bit_position = 0x20 - pid; // bit 31 = PID 0x01, bit 0 = PID 0x20
            (bits >> bit_position) & 1 == 1
        })
    }

    /// Decode and format a response to a human-readable string
    pub fn format_response(&self, response: &ObdResponse) -> String {
        let value = self.decode(response);
        let unit = ObdPid::from_byte(response.pid)
            .map(|p| p.unit())
            .unwrap_or("");
        let name = ObdPid::from_byte(response.pid)
            .map(|p| p.name())
            .unwrap_or("Unknown PID");
        format!("{}: {}", name, value.to_display_string(unit))
    }
}

/// DTC (Diagnostic Trouble Code) — SAE J2012 / ISO 15031-6
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dtc {
    /// DTC code string (e.g. "P0301")
    pub code: String,
    /// Human-readable description
    pub description: &'static str,
}

impl Dtc {
    /// Create a DTC with given code and description
    pub fn new(code: String, description: &'static str) -> Self {
        Self { code, description }
    }
}

/// DTC system type prefix
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtcSystem {
    /// P = Powertrain (engine, transmission)
    Powertrain,
    /// C = Chassis (brakes, suspension, steering)
    Chassis,
    /// B = Body (airbags, seat belts, HVAC)
    Body,
    /// U = Network (CAN bus, OBD modules)
    Network,
}

impl DtcSystem {
    /// Convert from the two high bits of the first DTC byte
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0x03 {
            0x00 => Self::Powertrain,
            0x01 => Self::Chassis,
            0x02 => Self::Body,
            0x03 => Self::Network,
            _ => unreachable!(),
        }
    }

    /// Get the letter prefix character
    pub fn prefix_char(&self) -> char {
        match self {
            Self::Powertrain => 'P',
            Self::Chassis => 'C',
            Self::Body => 'B',
            Self::Network => 'U',
        }
    }
}

/// OBD-II DTC decoder for Diagnostic Trouble Codes
///
/// Decodes Mode 03 (stored DTCs) and Mode 07 (pending DTCs) responses
#[derive(Debug, Default)]
pub struct DtcDecoder {
    /// Known DTC descriptions lookup table
    descriptions: HashMap<&'static str, &'static str>,
}

impl DtcDecoder {
    /// Create a new DTC decoder with common powertrain codes pre-loaded
    pub fn new() -> Self {
        let mut decoder = Self {
            descriptions: HashMap::new(),
        };
        decoder.load_common_codes();
        decoder
    }

    /// Load common P0xxx (generic powertrain) DTC descriptions
    fn load_common_codes(&mut self) {
        let codes: &[(&'static str, &'static str)] = &[
            ("P0000", "No fault"),
            ("P0100", "Mass or Volume Air Flow Circuit Malfunction"),
            ("P0101", "Mass or Volume Air Flow Circuit Range/Performance"),
            ("P0102", "Mass or Volume Air Flow Circuit Low Input"),
            ("P0103", "Mass or Volume Air Flow Circuit High Input"),
            ("P0110", "Intake Air Temperature Circuit Malfunction"),
            ("P0111", "Intake Air Temperature Circuit Range/Performance"),
            ("P0112", "Intake Air Temperature Circuit Low Input"),
            ("P0113", "Intake Air Temperature Circuit High Input"),
            ("P0115", "Engine Coolant Temperature Circuit Malfunction"),
            (
                "P0116",
                "Engine Coolant Temperature Circuit Range/Performance",
            ),
            ("P0117", "Engine Coolant Temperature Circuit Low Input"),
            ("P0118", "Engine Coolant Temperature Circuit High Input"),
            (
                "P0120",
                "Throttle/Pedal Position Sensor A Circuit Malfunction",
            ),
            (
                "P0121",
                "Throttle/Pedal Position Sensor A Range/Performance",
            ),
            ("P0122", "Throttle/Pedal Position Sensor A Low Input"),
            ("P0123", "Throttle/Pedal Position Sensor A High Input"),
            ("P0130", "O2 Sensor Circuit Malfunction (Bank 1, Sensor 1)"),
            ("P0131", "O2 Sensor Circuit Low Voltage (Bank 1, Sensor 1)"),
            ("P0132", "O2 Sensor Circuit High Voltage (Bank 1, Sensor 1)"),
            (
                "P0133",
                "O2 Sensor Circuit Slow Response (Bank 1, Sensor 1)",
            ),
            (
                "P0134",
                "O2 Sensor Circuit No Activity Detected (Bank 1, Sensor 1)",
            ),
            (
                "P0135",
                "O2 Sensor Heater Circuit Malfunction (Bank 1, Sensor 1)",
            ),
            ("P0171", "System Too Lean (Bank 1)"),
            ("P0172", "System Too Rich (Bank 1)"),
            ("P0174", "System Too Lean (Bank 2)"),
            ("P0175", "System Too Rich (Bank 2)"),
            ("P0200", "Injector Circuit Malfunction"),
            ("P0201", "Injector Circuit Malfunction - Cylinder 1"),
            ("P0202", "Injector Circuit Malfunction - Cylinder 2"),
            ("P0203", "Injector Circuit Malfunction - Cylinder 3"),
            ("P0204", "Injector Circuit Malfunction - Cylinder 4"),
            ("P0300", "Random/Multiple Cylinder Misfire Detected"),
            ("P0301", "Cylinder 1 Misfire Detected"),
            ("P0302", "Cylinder 2 Misfire Detected"),
            ("P0303", "Cylinder 3 Misfire Detected"),
            ("P0304", "Cylinder 4 Misfire Detected"),
            ("P0305", "Cylinder 5 Misfire Detected"),
            ("P0306", "Cylinder 6 Misfire Detected"),
            ("P0307", "Cylinder 7 Misfire Detected"),
            ("P0308", "Cylinder 8 Misfire Detected"),
            ("P0325", "Knock Sensor 1 Circuit Malfunction (Bank 1)"),
            ("P0327", "Knock Sensor 1 Circuit Low Input (Bank 1)"),
            ("P0328", "Knock Sensor 1 Circuit High Input (Bank 1)"),
            ("P0335", "Crankshaft Position Sensor A Circuit Malfunction"),
            ("P0336", "Crankshaft Position Sensor A Range/Performance"),
            (
                "P0340",
                "Camshaft Position Sensor Circuit Malfunction (Bank 1)",
            ),
            (
                "P0341",
                "Camshaft Position Sensor Range/Performance (Bank 1)",
            ),
            ("P0400", "Exhaust Gas Recirculation Flow Malfunction"),
            (
                "P0401",
                "Exhaust Gas Recirculation Flow Insufficient Detected",
            ),
            ("P0402", "Exhaust Gas Recirculation Flow Excessive Detected"),
            (
                "P0420",
                "Catalyst System Efficiency Below Threshold (Bank 1)",
            ),
            (
                "P0430",
                "Catalyst System Efficiency Below Threshold (Bank 2)",
            ),
            ("P0440", "Evaporative Emission Control System Malfunction"),
            (
                "P0441",
                "Evaporative Emission Control System Incorrect Purge Flow",
            ),
            (
                "P0442",
                "Evaporative Emission Control System Leak Detected (small)",
            ),
            (
                "P0443",
                "Evaporative Emission Control System Purge Control Valve Circuit",
            ),
            (
                "P0455",
                "Evaporative Emission Control System Leak Detected (gross)",
            ),
            ("P0500", "Vehicle Speed Sensor Malfunction"),
            ("P0505", "Idle Air Control System Malfunction"),
            ("P0506", "Idle Air Control System RPM Lower Than Expected"),
            ("P0507", "Idle Air Control System RPM Higher Than Expected"),
            ("P0600", "Serial Communication Link Malfunction"),
            ("P0601", "Internal Control Module Memory Check Sum Error"),
            ("P0602", "Control Module Programming Error"),
            ("P0603", "Internal Control Module Keep Alive Memory Error"),
            (
                "P0604",
                "Internal Control Module Random Access Memory Error",
            ),
            ("P0605", "Internal Control Module Read Only Memory Error"),
            ("P0700", "Transmission Control System Malfunction"),
            ("P0715", "Input/Turbine Speed Sensor Circuit Malfunction"),
            ("P0720", "Output Speed Sensor Circuit Malfunction"),
            ("P0730", "Incorrect Gear Ratio"),
            ("P0740", "Torque Converter Clutch Circuit Malfunction"),
            ("P0750", "Shift Solenoid A Malfunction"),
            ("P0755", "Shift Solenoid B Malfunction"),
            ("P0760", "Shift Solenoid C Malfunction"),
            ("P0765", "Shift Solenoid D Malfunction"),
            ("P0770", "Shift Solenoid E Malfunction"),
        ];

        for (code, desc) in codes {
            self.descriptions.insert(code, desc);
        }
    }

    /// Decode 2 raw DTC bytes to a DTC code string (e.g. "P0301")
    ///
    /// DTC encoding per SAE J2012 / ISO 15031-6:
    /// - Byte A bits 7-6: system (00=P, 01=C, 10=B, 11=U)
    /// - Byte A bits 5-4: subtype digit (0-3)
    /// - Byte A bits 3-0: first code digit (0-9, A-F as hex)
    /// - Byte B bits 7-4: second code digit
    /// - Byte B bits 3-0: third code digit
    pub fn decode_dtc_bytes(a: u8, b: u8) -> String {
        let system = DtcSystem::from_bits(a >> 6);
        let prefix = system.prefix_char();

        // Digit 1: bits 5-4 of byte A (0-3)
        let digit1 = (a >> 4) & 0x03;

        // Digits 2 and 3 from byte A lower nibble and byte B
        let digit2 = a & 0x0F;
        let digit3 = (b >> 4) & 0x0F;
        let digit4 = b & 0x0F;

        format!("{}{}{:X}{:X}{:X}", prefix, digit1, digit2, digit3, digit4)
    }

    /// Decode a DTC code string to a Dtc struct with description
    pub fn lookup(&self, code: &str) -> Dtc {
        let description = self
            .descriptions
            .get(code)
            .copied()
            .unwrap_or("Unknown fault code");
        Dtc {
            code: code.to_string(),
            description,
        }
    }

    /// Decode bytes and lookup description
    pub fn decode_bytes_to_dtc(&self, a: u8, b: u8) -> Option<Dtc> {
        // 0x0000 means no fault
        if a == 0x00 && b == 0x00 {
            return None;
        }
        let code = Self::decode_dtc_bytes(a, b);
        Some(self.lookup(&code))
    }

    /// Parse a Mode 03 response payload into a list of DTCs
    ///
    /// Format: [num_dtcs, A1, B1, A2, B2, ...]
    pub fn parse_mode03_response(&self, data: &[u8]) -> Vec<Dtc> {
        if data.is_empty() {
            return Vec::new();
        }
        let count = data[0] as usize;
        let mut dtcs = Vec::with_capacity(count);

        for i in 0..count {
            let offset = 1 + i * 2;
            if offset + 1 < data.len() {
                if let Some(dtc) = self.decode_bytes_to_dtc(data[offset], data[offset + 1]) {
                    dtcs.push(dtc);
                }
            }
        }

        dtcs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ObdPid tests ----

    #[test]
    fn test_obd_pid_from_byte_known() {
        assert_eq!(ObdPid::from_byte(0x04), Some(ObdPid::EngineLoad));
        assert_eq!(ObdPid::from_byte(0x05), Some(ObdPid::CoolantTemp));
        assert_eq!(ObdPid::from_byte(0x0C), Some(ObdPid::EngineRpm));
        assert_eq!(ObdPid::from_byte(0x0D), Some(ObdPid::VehicleSpeed));
        assert_eq!(ObdPid::from_byte(0x10), Some(ObdPid::MafFlowRate));
        assert_eq!(ObdPid::from_byte(0x11), Some(ObdPid::ThrottlePosition));
        assert_eq!(ObdPid::from_byte(0x2F), Some(ObdPid::FuelTankLevel));
        assert_eq!(ObdPid::from_byte(0x46), Some(ObdPid::AmbientAirTemp));
        assert_eq!(ObdPid::from_byte(0x49), Some(ObdPid::AccelPedalPosD));
    }

    #[test]
    fn test_obd_pid_from_byte_unknown() {
        assert_eq!(ObdPid::from_byte(0xFF), None);
        assert_eq!(ObdPid::from_byte(0xAA), None);
    }

    #[test]
    fn test_obd_pid_name_and_unit() {
        let pid = ObdPid::EngineRpm;
        assert_eq!(pid.name(), "Engine RPM");
        assert_eq!(pid.unit(), "RPM");

        let pid = ObdPid::CoolantTemp;
        assert_eq!(pid.unit(), "°C");

        let pid = ObdPid::VehicleSpeed;
        assert_eq!(pid.unit(), "km/h");
    }

    // ---- ObdRequest tests ----

    #[test]
    fn test_obd_request_new() {
        let req = ObdRequest::new(0x01, 0x0C);
        assert_eq!(req.service, 0x01);
        assert_eq!(req.pid, 0x0C);
    }

    #[test]
    fn test_obd_request_mode01() {
        let req = ObdRequest::mode01(0x04);
        assert_eq!(req.service, 0x01);
        assert_eq!(req.pid, 0x04);
    }

    #[test]
    fn test_obd_request_encode() {
        let req = ObdRequest::mode01(0x0C);
        let encoded = req.encode();
        assert_eq!(encoded, vec![0x02, 0x01, 0x0C]);
    }

    // ---- ObdResponse decode tests ----

    #[test]
    fn test_decode_engine_rpm() {
        // PID 0x0C: (256*A + B) / 4
        // A=0x1A (26), B=0xF0 (240)
        // RPM = (256*26 + 240) / 4 = (6656 + 240) / 4 = 6896 / 4 = 1724
        let resp = ObdResponse::new(0x0C, vec![0x1A, 0xF0]);
        let value = resp.decode();
        assert!(matches!(value, ObdValue::Rpm(rpm) if (rpm - 1724.0).abs() < 0.01));
    }

    #[test]
    fn test_decode_engine_rpm_zero() {
        let resp = ObdResponse::new(0x0C, vec![0x00, 0x00]);
        let value = resp.decode();
        assert!(matches!(value, ObdValue::Rpm(rpm) if rpm == 0.0));
    }

    #[test]
    fn test_decode_coolant_temperature() {
        // PID 0x05: A - 40 °C
        // A=0x69 (105): 105 - 40 = 65 °C
        let resp = ObdResponse::new(0x05, vec![0x69]);
        let value = resp.decode();
        assert!(matches!(value, ObdValue::Temperature(t) if (t - 65.0).abs() < 0.01));
    }

    #[test]
    fn test_decode_coolant_temp_minus_40() {
        // A=0x00: 0 - 40 = -40 °C (minimum)
        let resp = ObdResponse::new(0x05, vec![0x00]);
        let value = resp.decode();
        assert!(matches!(value, ObdValue::Temperature(t) if (t - (-40.0)).abs() < 0.01));
    }

    #[test]
    fn test_decode_vehicle_speed() {
        // PID 0x0D: A km/h
        // A=0x64 (100): 100 km/h
        let resp = ObdResponse::new(0x0D, vec![0x64]);
        let value = resp.decode();
        assert!(matches!(value, ObdValue::Speed(s) if (s - 100.0).abs() < 0.01));
    }

    #[test]
    fn test_decode_throttle_position() {
        // PID 0x11: A / 2.55 %
        // A=0x80 (128): 128 / 2.55 ≈ 50.2 %
        let resp = ObdResponse::new(0x11, vec![0x80]);
        let value = resp.decode();
        assert!(matches!(value, ObdValue::Percent(p) if (p - 128.0/2.55).abs() < 0.01));
    }

    #[test]
    fn test_decode_engine_load() {
        // PID 0x04: A / 2.55 %
        // A=0xFF (255): 255 / 2.55 = 100 %
        let resp = ObdResponse::new(0x04, vec![0xFF]);
        let value = resp.decode();
        assert!(matches!(value, ObdValue::Percent(p) if (p - 100.0).abs() < 0.02));
    }

    #[test]
    fn test_decode_maf_flow_rate() {
        // PID 0x10: (256*A + B) / 100 g/s
        // A=0x01 (1), B=0x90 (144): (256 + 144) / 100 = 4.0 g/s
        let resp = ObdResponse::new(0x10, vec![0x01, 0x90]);
        let value = resp.decode();
        assert!(matches!(value, ObdValue::Grams(g) if (g - 4.0).abs() < 0.01));
    }

    #[test]
    fn test_decode_intake_air_temp() {
        // PID 0x0F: A - 40 °C
        // A=0x3C (60): 60 - 40 = 20 °C
        let resp = ObdResponse::new(0x0F, vec![0x3C]);
        let value = resp.decode();
        assert!(matches!(value, ObdValue::Temperature(t) if (t - 20.0).abs() < 0.01));
    }

    #[test]
    fn test_decode_ambient_air_temp() {
        // PID 0x46: A - 40 °C
        // A=0x1E (30): 30 - 40 = -10 °C
        let resp = ObdResponse::new(0x46, vec![0x1E]);
        let value = resp.decode();
        assert!(matches!(value, ObdValue::Temperature(t) if (t - (-10.0)).abs() < 0.01));
    }

    #[test]
    fn test_decode_distance_with_mil() {
        // PID 0x21: 256*A + B km
        // A=0x01 (1), B=0x00 (0): 256 km
        let resp = ObdResponse::new(0x21, vec![0x01, 0x00]);
        let value = resp.decode();
        assert!(matches!(value, ObdValue::Km(km) if (km - 256.0).abs() < 0.01));
    }

    #[test]
    fn test_decode_fuel_tank_level() {
        // PID 0x2F: A / 2.55 %
        // A=0x7F (127): 127 / 2.55 ≈ 49.8 %
        let resp = ObdResponse::new(0x2F, vec![0x7F]);
        let value = resp.decode();
        assert!(matches!(value, ObdValue::Percent(p) if (p - 127.0/2.55).abs() < 0.01));
    }

    #[test]
    fn test_decode_accel_pedal_pos_d() {
        // PID 0x49: A / 2.55 %
        let resp = ObdResponse::new(0x49, vec![0xCC]);
        let value = resp.decode();
        assert!(matches!(value, ObdValue::Percent(p) if (p - 0xCC as f64 / 2.55).abs() < 0.01));
    }

    #[test]
    fn test_decode_unknown_pid_returns_raw() {
        let resp = ObdResponse::new(0xAB, vec![0x12, 0x34]);
        let value = resp.decode();
        assert!(matches!(value, ObdValue::Raw(ref v) if *v == vec![0x12, 0x34]));
    }

    #[test]
    fn test_decode_obd_standard() {
        // PID 0x1C: OBD standard = A (0x01 = OBD-II)
        let resp = ObdResponse::new(0x1C, vec![0x01]);
        let value = resp.decode();
        assert!(matches!(value, ObdValue::Standard(1)));
    }

    #[test]
    fn test_obd_response_from_bytes() {
        let raw = vec![0x04, 0x41, 0x0C, 0x1A, 0xF0];
        let resp = ObdResponse::from_bytes(&raw);
        assert!(resp.is_some());
        let resp = resp.expect("valid response");
        assert_eq!(resp.pid, 0x0C);
        assert_eq!(resp.data, vec![0x1A, 0xF0]);
    }

    #[test]
    fn test_obd_response_from_bytes_invalid() {
        // Wrong response byte
        let raw = vec![0x03, 0x42, 0x0C, 0x00];
        assert!(ObdResponse::from_bytes(&raw).is_none());
        // Too short
        let raw = vec![0x01, 0x41];
        assert!(ObdResponse::from_bytes(&raw).is_none());
    }

    // ---- ObdDecoder tests ----

    #[test]
    fn test_obd_decoder_decode() {
        let decoder = ObdDecoder::new();
        let resp = ObdResponse::new(0x0C, vec![0x1A, 0xF0]);
        let value = decoder.decode(&resp);
        assert!(matches!(value, ObdValue::Rpm(rpm) if (rpm - 1724.0).abs() < 0.01));
    }

    #[test]
    fn test_obd_decoder_supported_pids() {
        let mut decoder = ObdDecoder::new();
        // All PIDs supported (0xFFFFFFFF)
        let resp = ObdResponse::new(0x00, vec![0xFF, 0xFF, 0xFF, 0xFF]);
        decoder.process_supported_pids(&resp);

        assert_eq!(decoder.is_pid_supported(0x01), Some(true));
        assert_eq!(decoder.is_pid_supported(0x04), Some(true));
        assert_eq!(decoder.is_pid_supported(0x20), Some(true));
    }

    #[test]
    fn test_obd_decoder_supported_pids_partial() {
        let mut decoder = ObdDecoder::new();
        // Only bit 31 set = PID 0x01 supported
        let resp = ObdResponse::new(0x00, vec![0x80, 0x00, 0x00, 0x00]);
        decoder.process_supported_pids(&resp);

        assert_eq!(decoder.is_pid_supported(0x01), Some(true));
        assert_eq!(decoder.is_pid_supported(0x02), Some(false));
    }

    #[test]
    fn test_obd_decoder_format_response() {
        let decoder = ObdDecoder::new();
        let resp = ObdResponse::new(0x0D, vec![0x64]);
        let s = decoder.format_response(&resp);
        assert!(s.contains("Vehicle Speed"));
        assert!(s.contains("100"));
    }

    // ---- DTC tests ----

    #[test]
    fn test_dtc_decode_bytes_p0301() {
        // P0301: Cylinder 1 Misfire
        // Byte A: 0x03 (P, subtype 0, digit2 3), Byte B: 0x01 (digit3 0, digit4 1)
        let code = DtcDecoder::decode_dtc_bytes(0x03, 0x01);
        assert_eq!(code, "P0301");
    }

    #[test]
    fn test_dtc_decode_bytes_p0420() {
        // P0420: Catalyst System Efficiency
        // A=0x04, B=0x20
        let code = DtcDecoder::decode_dtc_bytes(0x04, 0x20);
        assert_eq!(code, "P0420");
    }

    #[test]
    fn test_dtc_decode_bytes_chassis() {
        // C0xxx: Chassis code (bits 7-6 of A = 01)
        // A=0x41 (01_000001), B=0x00
        let code = DtcDecoder::decode_dtc_bytes(0x41, 0x00);
        assert!(code.starts_with('C'));
    }

    #[test]
    fn test_dtc_decode_bytes_body() {
        // B0xxx: Body code (bits 7-6 of A = 10)
        // A=0x80 (10_000000), B=0x00
        let code = DtcDecoder::decode_dtc_bytes(0x80, 0x00);
        assert!(code.starts_with('B'));
    }

    #[test]
    fn test_dtc_decode_bytes_network() {
        // U0xxx: Network code (bits 7-6 of A = 11)
        // A=0xC0 (11_000000), B=0x00
        let code = DtcDecoder::decode_dtc_bytes(0xC0, 0x00);
        assert!(code.starts_with('U'));
    }

    #[test]
    fn test_dtc_decoder_lookup_known() {
        let decoder = DtcDecoder::new();
        let dtc = decoder.lookup("P0301");
        assert_eq!(dtc.code, "P0301");
        assert!(dtc.description.contains("Misfire") || dtc.description.contains("Cylinder 1"));
    }

    #[test]
    fn test_dtc_decoder_lookup_unknown() {
        let decoder = DtcDecoder::new();
        let dtc = decoder.lookup("P9999");
        assert_eq!(dtc.code, "P9999");
        assert_eq!(dtc.description, "Unknown fault code");
    }

    #[test]
    fn test_dtc_decode_bytes_to_dtc_no_fault() {
        let decoder = DtcDecoder::new();
        let result = decoder.decode_bytes_to_dtc(0x00, 0x00);
        assert!(result.is_none());
    }

    #[test]
    fn test_dtc_decode_bytes_to_dtc_with_fault() {
        let decoder = DtcDecoder::new();
        let result = decoder.decode_bytes_to_dtc(0x03, 0x01);
        assert!(result.is_some());
        let dtc = result.expect("valid DTC");
        assert_eq!(dtc.code, "P0301");
    }

    #[test]
    fn test_dtc_parse_mode03_response_empty() {
        let decoder = DtcDecoder::new();
        let dtcs = decoder.parse_mode03_response(&[]);
        assert!(dtcs.is_empty());
    }

    #[test]
    fn test_dtc_parse_mode03_response_two_dtcs() {
        let decoder = DtcDecoder::new();
        // 2 DTCs: P0301 and P0420
        let data = vec![0x02, 0x03, 0x01, 0x04, 0x20];
        let dtcs = decoder.parse_mode03_response(&data);
        assert_eq!(dtcs.len(), 2);
        assert_eq!(dtcs[0].code, "P0301");
        assert_eq!(dtcs[1].code, "P0420");
    }

    #[test]
    fn test_dtc_parse_mode03_response_skips_null() {
        let decoder = DtcDecoder::new();
        // 2 entries but second is 0x0000 (no fault)
        let data = vec![0x02, 0x03, 0x01, 0x00, 0x00];
        let dtcs = decoder.parse_mode03_response(&data);
        // Only P0301 should be returned; 0x0000 is skipped
        assert_eq!(dtcs.len(), 1);
        assert_eq!(dtcs[0].code, "P0301");
    }

    #[test]
    fn test_obd_value_to_display_string() {
        let v = ObdValue::Rpm(1500.0);
        let s = v.to_display_string("RPM");
        assert!(s.contains("1500"));
        assert!(s.contains("RPM"));

        let v = ObdValue::Temperature(-10.0);
        let s = v.to_display_string("°C");
        assert!(s.contains("-10.0"));

        let v = ObdValue::Raw(vec![0xDE, 0xAD]);
        let s = v.to_display_string("");
        assert!(s.contains("DE"));
        assert!(s.contains("AD"));
    }

    #[test]
    fn test_obd_service_from_byte() {
        assert_eq!(ObdService::from_byte(0x01), Some(ObdService::CurrentData));
        assert_eq!(ObdService::from_byte(0x03), Some(ObdService::StoredDtcs));
        assert_eq!(ObdService::from_byte(0xFF), None);
    }

    #[test]
    fn test_decode_runtime_since_start() {
        // PID 0x1F: 256*A + B seconds
        // A=0x00 (0), B=0x3C (60): 60 seconds
        let resp = ObdResponse::new(0x1F, vec![0x00, 0x3C]);
        let value = resp.decode();
        assert!(matches!(value, ObdValue::Seconds(s) if (s - 60.0).abs() < 0.01));
    }
}
