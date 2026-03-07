//! J1939 Diagnostic Messages (DM) support
//!
//! Implements SAE J1939/73 diagnostic messaging protocol including:
//!
//! - **DM1** – Active Diagnostic Trouble Codes (DTCs)
//! - **DM2** – Previously Active Diagnostic Trouble Codes
//! - **DM3** – Diagnostic Data Clear/Reset of Previously Active Faults
//! - **DM11** – Diagnostic Data Clear/Reset of Active Faults
//! - **DM13** – Stop Start Broadcast
//! - **DM15** – Memory Access Response
//! - **DM16** – Binary Data Transfer
//!
//! # Failure Mode Identifiers (FMI)
//!
//! FMI codes describe the *nature* of the fault rather than which component
//! failed. Each FMI is a 5-bit value (0–31) per SAE J1939/71.
//!
//! # Example
//!
//! ```rust
//! use oxirs_canbus::j1939::diagnostics::{DiagnosticTroubleCode, Dm1Message, LampStatus};
//!
//! // Build a DM1 message with one active DTC
//! let dtc = DiagnosticTroubleCode::new(520, 4); // SPN 520, FMI 4
//! let dm1 = Dm1Message {
//!     mil_lamp:  LampStatus::On,
//!     rsl_lamp:  LampStatus::Off,
//!     awl_lamp:  LampStatus::Off,
//!     pl_lamp:   LampStatus::Off,
//!     dtcs:      vec![dtc],
//! };
//! assert!(dm1.has_active_faults());
//! assert_eq!(dm1.fault_count(), 1);
//! let encoded = dm1.encode();
//! let decoded = Dm1Message::decode(&encoded).expect("decode should succeed");
//! assert_eq!(decoded.fault_count(), 1);
//! ```

use crate::error::CanbusError;

// ============================================================================
// Lamp Status
// ============================================================================

/// Status of a J1939 diagnostic lamp
///
/// Each lamp is encoded as a 2-bit field per SAE J1939/73 Table 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LampStatus {
    /// Lamp is off (0b00 or 0b11)
    Off = 0b00,
    /// Lamp is on solid (0b01)
    On = 0b01,
    /// Lamp flashes slowly (0b10)
    SlowFlash = 0b10,
    /// Lamp flashes fast (0b11 in flash byte, 0b01 in status byte combo)
    FastFlash = 0b11,
}

impl LampStatus {
    /// Decode a 2-bit lamp status from its raw field value.
    pub fn from_raw(raw: u8) -> Self {
        match raw & 0b11 {
            0b01 => LampStatus::On,
            0b10 => LampStatus::SlowFlash,
            0b11 => LampStatus::FastFlash,
            _ => LampStatus::Off,
        }
    }

    /// Return the raw 2-bit value.
    pub fn as_raw(self) -> u8 {
        self as u8
    }

    /// Return `true` if the lamp is illuminated or flashing.
    pub fn is_active(self) -> bool {
        !matches!(self, LampStatus::Off)
    }
}

impl std::fmt::Display for LampStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LampStatus::Off => write!(f, "Off"),
            LampStatus::On => write!(f, "On"),
            LampStatus::SlowFlash => write!(f, "SlowFlash"),
            LampStatus::FastFlash => write!(f, "FastFlash"),
        }
    }
}

// ============================================================================
// Diagnostic Trouble Code
// ============================================================================

/// A J1939 Diagnostic Trouble Code (DTC)
///
/// Each DTC is encoded in 4 bytes (32 bits) per SAE J1939/73:
///
/// ```text
/// Bits 31–19  : SPN (Suspect Parameter Number, 19 bits, some in byte 3 bits 7-5)
/// Bits 18–13  : reserved / SPN MSBs
/// Bits 12–8   : SPN lower 5 MSBs embedded in byte 2 (bits 7-3)
/// Bits 7–3    : FMI (5 bits, bits 4-0 of byte 3)
/// Bit  2      : CM  (Conversion Method)
/// Bits 1–0    : OC  (Occurrence Count, 7 bits across byte 3)
/// ```
///
/// Precise layout (little-endian byte order, byte 0 first):
/// - Byte 0: SPN bits 7-0
/// - Byte 1: SPN bits 15-8
/// - Byte 2: SPN bits 18-16 in bits 7-5; FMI bits 4-0 in bits 4-0
/// - Byte 3: CM in bit 7; OC in bits 6-0
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticTroubleCode {
    /// Suspect Parameter Number (19 bits, 0–524287)
    pub spn: u32,
    /// Failure Mode Identifier (5 bits, 0–31)
    pub fmi: u8,
    /// Occurrence count (7 bits, 0–126; 127 = not available)
    pub occurrence_count: u8,
    /// Conversion Method bit (false = counts as J1939-71, true = manufacturer specific)
    pub cm: bool,
}

impl DiagnosticTroubleCode {
    /// Create a new DTC with occurrence count 1 and CM=false.
    pub fn new(spn: u32, fmi: u8) -> Self {
        Self {
            spn: spn & 0x7FFFF, // 19-bit mask
            fmi: fmi & 0x1F,    // 5-bit mask
            occurrence_count: 1,
            cm: false,
        }
    }

    /// Create a DTC with all fields specified.
    ///
    /// `occurrence_count` values 0–126 indicate known counts; 127 is the SAE
    /// J1939 "not available / not reported" sentinel (OC field is 7 bits, max
    /// encodable value is 127).
    pub fn with_details(spn: u32, fmi: u8, occurrence_count: u8, cm: bool) -> Self {
        Self {
            spn: spn & 0x7FFFF,
            fmi: fmi & 0x1F,
            // Clamp to the 7-bit field maximum (127 = "not available" sentinel is valid)
            occurrence_count: occurrence_count & 0x7F,
            cm,
        }
    }

    /// Encode this DTC into 4 bytes per SAE J1939/73.
    pub fn encode(&self) -> [u8; 4] {
        let spn = self.spn & 0x7FFFF;
        let byte0 = (spn & 0xFF) as u8;
        let byte1 = ((spn >> 8) & 0xFF) as u8;
        // Byte 2: bits 7-5 = SPN[18-16], bits 4-0 = FMI
        let byte2 = (((spn >> 16) & 0x07) as u8) << 5 | (self.fmi & 0x1F);
        // Byte 3: bit 7 = CM, bits 6-0 = OC
        let byte3 = ((self.cm as u8) << 7) | (self.occurrence_count & 0x7F);
        [byte0, byte1, byte2, byte3]
    }

    /// Decode a DTC from 4 bytes.
    pub fn decode(bytes: &[u8; 4]) -> Result<Self, CanbusError> {
        let spn_low = bytes[0] as u32;
        let spn_mid = (bytes[1] as u32) << 8;
        let spn_high = ((bytes[2] as u32) >> 5) << 16;
        let spn = spn_low | spn_mid | spn_high;

        let fmi = bytes[2] & 0x1F;
        let cm = (bytes[3] >> 7) & 0x01 != 0;
        let occurrence_count = bytes[3] & 0x7F;

        if fmi > 31 {
            return Err(CanbusError::Config(format!(
                "Invalid FMI value: {} (max 31)",
                fmi
            )));
        }

        Ok(Self {
            spn,
            fmi,
            occurrence_count,
            cm,
        })
    }

    /// Human-readable description of this FMI per SAE J1939/71 Table SPN 1213.
    pub fn fmi_description(&self) -> &'static str {
        match self.fmi {
            0 => "Data Valid But Above Normal Operational Range (most severe)",
            1 => "Data Valid But Below Normal Operational Range (most severe)",
            2 => "Data Erratic, Intermittent Or Incorrect",
            3 => "Voltage Above Normal, Or Shorted To High Source",
            4 => "Voltage Below Normal, Or Shorted To Low Source",
            5 => "Current Below Normal Or Open Circuit",
            6 => "Current Above Normal Or Grounded Circuit",
            7 => "Mechanical System Not Responding Or Out Of Adjustment",
            8 => "Abnormal Frequency Or Pulse Width Or Period",
            9 => "Abnormal Update Rate",
            10 => "Abnormal Rate Of Change",
            11 => "Root Cause Not Known",
            12 => "Bad Intelligent Device Or Component",
            13 => "Out Of Calibration",
            14 => "Special Instructions",
            15 => "Data Valid But Above Normal Operating Range (least severe)",
            16 => "Data Valid But Above Normal Operating Range (moderate severity)",
            17 => "Data Valid But Below Normal Operating Range (least severe)",
            18 => "Data Valid But Below Normal Operating Range (moderate severity)",
            19 => "Received Network Data In Error",
            20 => "Data Drifted High",
            21 => "Data Drifted Low",
            22..=30 => "Reserved for SAE Assignment",
            31 => "Condition Exists",
            _ => "Unknown FMI",
        }
    }

    /// Return `true` when occurrence count < 127 (i.e., known to be active).
    pub fn is_active(&self) -> bool {
        self.occurrence_count < 127
    }

    /// Return a compact string representation, e.g. `"SPN520/FMI4"`.
    pub fn short_label(&self) -> String {
        format!("SPN{}/FMI{}", self.spn, self.fmi)
    }
}

impl std::fmt::Display for DiagnosticTroubleCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DTC[SPN={} FMI={} OC={} CM={}]: {}",
            self.spn,
            self.fmi,
            self.occurrence_count,
            self.cm,
            self.fmi_description()
        )
    }
}

// ============================================================================
// DM1 – Active Diagnostic Trouble Codes
// ============================================================================

/// J1939 DM1 – Active Diagnostic Trouble Codes
///
/// Transmitted by ECUs when one or more DTCs are active. The message
/// is sent at 1 Hz while active faults exist, or broadcast as a "no active
/// faults" message (0 DTCs, all lamps off) when clear.
///
/// Wire format (N = number of DTCs):
/// ```text
/// Byte 0   : lamp status byte (MIL, RSL, AWL, PL – 2 bits each)
/// Byte 1   : lamp flash byte  (same order, separate encoding)
/// Bytes 2+ : DTCs, 4 bytes each
/// ```
#[derive(Debug, Clone)]
pub struct Dm1Message {
    /// Malfunction Indicator Lamp (MIL / CEL)
    pub mil_lamp: LampStatus,
    /// Red Stop Lamp (RSL)
    pub rsl_lamp: LampStatus,
    /// Amber Warning Lamp (AWL)
    pub awl_lamp: LampStatus,
    /// Protect Lamp (PL)
    pub pl_lamp: LampStatus,
    /// Active diagnostic trouble codes (may be empty)
    pub dtcs: Vec<DiagnosticTroubleCode>,
}

impl Dm1Message {
    /// Create a DM1 message with no active faults and all lamps off.
    pub fn no_faults() -> Self {
        Self {
            mil_lamp: LampStatus::Off,
            rsl_lamp: LampStatus::Off,
            awl_lamp: LampStatus::Off,
            pl_lamp: LampStatus::Off,
            dtcs: Vec::new(),
        }
    }

    /// Return `true` when at least one DTC is present.
    pub fn has_active_faults(&self) -> bool {
        !self.dtcs.is_empty()
    }

    /// Number of active DTCs.
    pub fn fault_count(&self) -> usize {
        self.dtcs.len()
    }

    /// Encode the DM1 message to bytes.
    ///
    /// Output: 2 header bytes + 4 bytes per DTC (min 2 bytes for empty).
    pub fn encode(&self) -> Vec<u8> {
        // Lamp status byte: bits 7-6 = MIL, 5-4 = RSL, 3-2 = AWL, 1-0 = PL
        let status_byte = (self.mil_lamp.as_raw() << 6)
            | (self.rsl_lamp.as_raw() << 4)
            | (self.awl_lamp.as_raw() << 2)
            | self.pl_lamp.as_raw();

        // Flash byte: same layout, currently mirrors status (simplified)
        let flash_byte = status_byte;

        let mut out = vec![status_byte, flash_byte];
        for dtc in &self.dtcs {
            out.extend_from_slice(&dtc.encode());
        }
        out
    }

    /// Decode a DM1 (or DM2) message from raw bytes.
    pub fn decode(data: &[u8]) -> Result<Self, CanbusError> {
        if data.len() < 2 {
            return Err(CanbusError::Config(
                "DM1 message requires at least 2 bytes".to_string(),
            ));
        }

        let status_byte = data[0];
        let mil_lamp = LampStatus::from_raw((status_byte >> 6) & 0x03);
        let rsl_lamp = LampStatus::from_raw((status_byte >> 4) & 0x03);
        let awl_lamp = LampStatus::from_raw((status_byte >> 2) & 0x03);
        let pl_lamp = LampStatus::from_raw(status_byte & 0x03);

        // Bytes 2+ are DTCs (4 bytes each)
        let dtc_data = &data[2..];
        if dtc_data.len() % 4 != 0 {
            return Err(CanbusError::Config(format!(
                "DM1 DTC data length {} is not a multiple of 4",
                dtc_data.len()
            )));
        }

        let mut dtcs = Vec::with_capacity(dtc_data.len() / 4);
        for chunk in dtc_data.chunks_exact(4) {
            let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            let dtc = DiagnosticTroubleCode::decode(&bytes)?;
            dtcs.push(dtc);
        }

        Ok(Self {
            mil_lamp,
            rsl_lamp,
            awl_lamp,
            pl_lamp,
            dtcs,
        })
    }

    /// Check whether any lamp is currently active.
    pub fn any_lamp_active(&self) -> bool {
        self.mil_lamp.is_active()
            || self.rsl_lamp.is_active()
            || self.awl_lamp.is_active()
            || self.pl_lamp.is_active()
    }

    /// Return the highest-severity lamp that is active.
    ///
    /// Severity order (highest to lowest): RSL > MIL > AWL > PL.
    pub fn highest_severity_lamp(&self) -> Option<LampStatus> {
        if self.rsl_lamp.is_active() {
            return Some(self.rsl_lamp);
        }
        if self.mil_lamp.is_active() {
            return Some(self.mil_lamp);
        }
        if self.awl_lamp.is_active() {
            return Some(self.awl_lamp);
        }
        if self.pl_lamp.is_active() {
            return Some(self.pl_lamp);
        }
        None
    }
}

// ============================================================================
// DM2 – Previously Active Diagnostic Trouble Codes
// ============================================================================

/// J1939 DM2 – Previously Active Diagnostic Trouble Codes
///
/// Same wire format as DM1. Contains DTCs that were active during previous
/// ignition cycles but are not currently active.
pub type Dm2Message = Dm1Message;

// ============================================================================
// DM3 / DM11 – Clear Requests
// ============================================================================

/// J1939 DM3 – Diagnostic Data Clear/Reset of Previously Active Faults
///
/// Sent to request an ECU clear its DM2 history. The request is a single
/// PGN request frame directed at the target ECU's address.
#[derive(Debug, Clone)]
pub struct Dm3Request {
    /// Source address of the device sending the clear request
    pub source_address: u8,
    /// Target ECU address (use 0xFF for global broadcast)
    pub destination_address: u8,
}

impl Dm3Request {
    /// Create a new DM3 clear request.
    pub fn new(source_address: u8, destination_address: u8) -> Self {
        Self {
            source_address,
            destination_address,
        }
    }

    /// PGN for DM3 (per J1939/73)
    pub const PGN: u32 = 0xFECC; // 65228
}

/// J1939 DM11 – Diagnostic Data Clear/Reset of Active Faults
///
/// Sent to request an ECU clear its DM1 active faults.
#[derive(Debug, Clone)]
pub struct Dm11Request {
    /// Source address of the requesting device
    pub source_address: u8,
    /// Target ECU address (0xFF = global)
    pub destination_address: u8,
}

impl Dm11Request {
    /// Create a new DM11 clear request.
    pub fn new(source_address: u8, destination_address: u8) -> Self {
        Self {
            source_address,
            destination_address,
        }
    }

    /// PGN for DM11 (per J1939/73)
    pub const PGN: u32 = 0xFED3; // 65235
}

// ============================================================================
// DM13 – Stop Start Broadcast
// ============================================================================

/// Hold Signal values for DM13 broadcast control
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoldSignal {
    /// Restart broadcast (0b00)
    RestartBroadcast = 0b00,
    /// Hold broadcast suspended (0b01)
    HoldSuspended = 0b01,
    /// Reserved (0b10)
    Reserved = 0b10,
    /// Not defined (0b11)
    NotDefined = 0b11,
}

impl HoldSignal {
    /// Decode from raw 2-bit value.
    pub fn from_raw(raw: u8) -> Self {
        match raw & 0b11 {
            0b00 => HoldSignal::RestartBroadcast,
            0b01 => HoldSignal::HoldSuspended,
            0b10 => HoldSignal::Reserved,
            _ => HoldSignal::NotDefined,
        }
    }

    /// Return the raw 2-bit value.
    pub fn as_raw(self) -> u8 {
        self as u8
    }
}

/// J1939 DM13 – Stop Start Broadcast
///
/// Used to temporarily suspend or restart periodic J1939 broadcast messages
/// from one or more ECUs on the network. Useful during diagnostics when the
/// bus load must be reduced.
#[derive(Debug, Clone)]
pub struct Dm13Message {
    /// Hold signal for current data link (bits 1-0 of byte 0)
    pub current_data_link: HoldSignal,
    /// Hold signal for J1939 network 1 (bits 3-2 of byte 0)
    pub j1939_network1: HoldSignal,
    /// Hold signal for J1939 network 2 (bits 5-4 of byte 0)
    pub j1939_network2: HoldSignal,
    /// Hold signal for J1939 network 3 (bits 7-6 of byte 0)
    pub j1939_network3: HoldSignal,
    /// Time to hold (seconds, 0 = indefinite)
    pub hold_time_seconds: u16,
}

impl Dm13Message {
    /// PGN for DM13
    pub const PGN: u32 = 0xDF00; // 57088

    /// Create a "suspend all broadcasts" DM13 message.
    pub fn suspend_all(hold_time_seconds: u16) -> Self {
        Self {
            current_data_link: HoldSignal::HoldSuspended,
            j1939_network1: HoldSignal::HoldSuspended,
            j1939_network2: HoldSignal::HoldSuspended,
            j1939_network3: HoldSignal::HoldSuspended,
            hold_time_seconds,
        }
    }

    /// Create a "restart all broadcasts" DM13 message.
    pub fn restart_all() -> Self {
        Self {
            current_data_link: HoldSignal::RestartBroadcast,
            j1939_network1: HoldSignal::RestartBroadcast,
            j1939_network2: HoldSignal::RestartBroadcast,
            j1939_network3: HoldSignal::RestartBroadcast,
            hold_time_seconds: 0,
        }
    }

    /// Encode to 8 bytes.
    pub fn encode(&self) -> [u8; 8] {
        let byte0 = (self.j1939_network3.as_raw() << 6)
            | (self.j1939_network2.as_raw() << 4)
            | (self.j1939_network1.as_raw() << 2)
            | self.current_data_link.as_raw();
        let hold = self.hold_time_seconds.to_le_bytes();
        [byte0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, hold[0], hold[1]]
    }

    /// Decode from 8 bytes.
    pub fn decode(data: &[u8]) -> Result<Self, CanbusError> {
        if data.len() < 8 {
            return Err(CanbusError::Config(
                "DM13 message requires 8 bytes".to_string(),
            ));
        }
        let byte0 = data[0];
        let current_data_link = HoldSignal::from_raw(byte0 & 0x03);
        let j1939_network1 = HoldSignal::from_raw((byte0 >> 2) & 0x03);
        let j1939_network2 = HoldSignal::from_raw((byte0 >> 4) & 0x03);
        let j1939_network3 = HoldSignal::from_raw((byte0 >> 6) & 0x03);
        let hold_time_seconds = u16::from_le_bytes([data[6], data[7]]);
        Ok(Self {
            current_data_link,
            j1939_network1,
            j1939_network2,
            j1939_network3,
            hold_time_seconds,
        })
    }
}

// ============================================================================
// DTC Event – High-level diagnostic event wrapper
// ============================================================================

/// A diagnostic event combining a DTC with additional context
#[derive(Debug, Clone)]
pub struct DiagnosticEvent {
    /// The actual trouble code
    pub dtc: DiagnosticTroubleCode,
    /// Source ECU address
    pub source_address: u8,
    /// Whether this is from DM1 (active) or DM2 (previously active)
    pub is_active: bool,
    /// Timestamp when the event was received
    pub timestamp: std::time::SystemTime,
}

impl DiagnosticEvent {
    /// Create a new active diagnostic event.
    pub fn active(dtc: DiagnosticTroubleCode, source_address: u8) -> Self {
        Self {
            dtc,
            source_address,
            is_active: true,
            timestamp: std::time::SystemTime::now(),
        }
    }

    /// Create a previously-active diagnostic event.
    pub fn previously_active(dtc: DiagnosticTroubleCode, source_address: u8) -> Self {
        Self {
            dtc,
            source_address,
            is_active: false,
            timestamp: std::time::SystemTime::now(),
        }
    }

    /// Return the SPN as a well-known string if recognized.
    pub fn spn_description(&self) -> Option<&'static str> {
        known_spn_description(self.dtc.spn)
    }
}

/// Return a human-readable description for well-known SPNs.
pub fn known_spn_description(spn: u32) -> Option<&'static str> {
    match spn {
        27 => Some("Engine Exhaust Gas Recirculation 1 Valve Position"),
        51 => Some("Throttle Position"),
        84 => Some("Wheel-Based Vehicle Speed"),
        91 => Some("Accelerator Pedal Position 1"),
        92 => Some("Percent Load At Current Speed"),
        100 => Some("Engine Oil Pressure"),
        102 => Some("Engine Intake Manifold 1 Pressure"),
        105 => Some("Engine Intake Manifold 1 Temperature"),
        108 => Some("Barometric Pressure"),
        110 => Some("Engine Coolant Temperature"),
        157 => Some("Engine Injector Metering Rail 1 Pressure"),
        168 => Some("Battery Potential / Power Input 1"),
        174 => Some("Engine Fuel Temperature 1"),
        175 => Some("Engine Oil Temperature 1"),
        183 => Some("Engine Fuel Rate"),
        190 => Some("Engine Speed"),
        235 => Some("Total Engine Hours"),
        247 => Some("Total Engine Revolutions"),
        248 => Some("Total Vehicle Distance"),
        512 => Some("Driver's Demand Engine - Percent Torque"),
        513 => Some("Actual Engine - Percent Torque"),
        514 => Some("Nominal Friction - Percent Torque"),
        519 => Some("Engine's Desired Operating Speed Asymmetry Adjustment"),
        520 => Some("Source Address Of Controlling Device For Engine Control"),
        529 => Some("Transmission Output Shaft Speed"),
        544 => Some("Engine Exhaust Gas Temperature"),
        917 => Some("Engine Average Fuel Economy"),
        1081 => Some("Engine Wait To Start Lamp"),
        1569 => Some("Engine Derate Request"),
        1634 => Some("Engine Fuel Delivery Pressure"),
        _ => None,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lamp_status_encoding() {
        assert_eq!(LampStatus::Off.as_raw(), 0b00);
        assert_eq!(LampStatus::On.as_raw(), 0b01);
        assert_eq!(LampStatus::SlowFlash.as_raw(), 0b10);
        assert_eq!(LampStatus::FastFlash.as_raw(), 0b11);
    }

    #[test]
    fn test_lamp_status_from_raw() {
        assert_eq!(LampStatus::from_raw(0b00), LampStatus::Off);
        assert_eq!(LampStatus::from_raw(0b01), LampStatus::On);
        assert_eq!(LampStatus::from_raw(0b10), LampStatus::SlowFlash);
        assert_eq!(LampStatus::from_raw(0b11), LampStatus::FastFlash);
    }

    #[test]
    fn test_lamp_status_is_active() {
        assert!(!LampStatus::Off.is_active());
        assert!(LampStatus::On.is_active());
        assert!(LampStatus::SlowFlash.is_active());
        assert!(LampStatus::FastFlash.is_active());
    }

    #[test]
    fn test_dtc_encode_decode_roundtrip() {
        let dtc = DiagnosticTroubleCode::with_details(520, 4, 3, false);
        let encoded = dtc.encode();
        let decoded = DiagnosticTroubleCode::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded.spn, 520);
        assert_eq!(decoded.fmi, 4);
        assert_eq!(decoded.occurrence_count, 3);
        assert!(!decoded.cm);
    }

    #[test]
    fn test_dtc_encode_decode_high_spn() {
        // SPN 524287 is the max 19-bit value
        let dtc = DiagnosticTroubleCode::with_details(524287, 31, 126, true);
        let encoded = dtc.encode();
        let decoded = DiagnosticTroubleCode::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded.spn, 524287);
        assert_eq!(decoded.fmi, 31);
        assert_eq!(decoded.occurrence_count, 126);
        assert!(decoded.cm);
    }

    #[test]
    fn test_dtc_fmi_descriptions() {
        let dtc = DiagnosticTroubleCode::new(100, 3);
        assert!(dtc.fmi_description().contains("Voltage Above Normal"));

        let dtc2 = DiagnosticTroubleCode::new(100, 0);
        assert!(dtc2
            .fmi_description()
            .contains("Above Normal Operational Range"));
    }

    #[test]
    fn test_dtc_is_active() {
        let dtc = DiagnosticTroubleCode::new(100, 3);
        assert!(dtc.is_active());

        let inactive = DiagnosticTroubleCode::with_details(100, 3, 127, false);
        assert!(!inactive.is_active());
    }

    #[test]
    fn test_dtc_short_label() {
        let dtc = DiagnosticTroubleCode::new(520, 4);
        assert_eq!(dtc.short_label(), "SPN520/FMI4");
    }

    #[test]
    fn test_dm1_no_faults() {
        let dm1 = Dm1Message::no_faults();
        assert!(!dm1.has_active_faults());
        assert_eq!(dm1.fault_count(), 0);
        assert!(!dm1.any_lamp_active());

        let encoded = dm1.encode();
        assert_eq!(encoded.len(), 2);
        let decoded = Dm1Message::decode(&encoded).expect("decode should succeed");
        assert!(!decoded.has_active_faults());
    }

    #[test]
    fn test_dm1_with_dtcs() {
        let dm1 = Dm1Message {
            mil_lamp: LampStatus::On,
            rsl_lamp: LampStatus::Off,
            awl_lamp: LampStatus::SlowFlash,
            pl_lamp: LampStatus::Off,
            dtcs: vec![
                DiagnosticTroubleCode::new(190, 3),
                DiagnosticTroubleCode::new(100, 4),
            ],
        };

        assert!(dm1.has_active_faults());
        assert_eq!(dm1.fault_count(), 2);
        assert!(dm1.any_lamp_active());

        let encoded = dm1.encode();
        assert_eq!(encoded.len(), 10); // 2 header + 2 * 4

        let decoded = Dm1Message::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded.fault_count(), 2);
        assert_eq!(decoded.mil_lamp, LampStatus::On);
        assert_eq!(decoded.awl_lamp, LampStatus::SlowFlash);
    }

    #[test]
    fn test_dm1_highest_severity_lamp() {
        let dm1 = Dm1Message {
            mil_lamp: LampStatus::On,
            rsl_lamp: LampStatus::On,
            awl_lamp: LampStatus::Off,
            pl_lamp: LampStatus::Off,
            dtcs: vec![DiagnosticTroubleCode::new(190, 3)],
        };
        // RSL is highest priority
        assert_eq!(dm1.highest_severity_lamp(), Some(LampStatus::On));

        let dm1_mil_only = Dm1Message {
            mil_lamp: LampStatus::On,
            rsl_lamp: LampStatus::Off,
            awl_lamp: LampStatus::Off,
            pl_lamp: LampStatus::Off,
            dtcs: vec![DiagnosticTroubleCode::new(190, 3)],
        };
        assert_eq!(dm1_mil_only.highest_severity_lamp(), Some(LampStatus::On));

        let dm1_none = Dm1Message::no_faults();
        assert_eq!(dm1_none.highest_severity_lamp(), None);
    }

    #[test]
    fn test_dm1_decode_invalid_length() {
        let result = Dm1Message::decode(&[0x00]);
        assert!(result.is_err());
    }

    #[test]
    fn test_dm1_decode_invalid_dtc_alignment() {
        // 2 header + 3 byte DTC payload (not multiple of 4)
        let result = Dm1Message::decode(&[0x00, 0x00, 0xAA, 0xBB, 0xCC]);
        assert!(result.is_err());
    }

    #[test]
    fn test_dm13_suspend_restart() {
        let suspend = Dm13Message::suspend_all(30);
        let encoded = suspend.encode();
        let decoded = Dm13Message::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded.hold_time_seconds, 30);
        assert_eq!(decoded.current_data_link, HoldSignal::HoldSuspended);

        let restart = Dm13Message::restart_all();
        let enc2 = restart.encode();
        let dec2 = Dm13Message::decode(&enc2).expect("decode should succeed");
        assert_eq!(dec2.current_data_link, HoldSignal::RestartBroadcast);
        assert_eq!(dec2.hold_time_seconds, 0);
    }

    #[test]
    fn test_diagnostic_event() {
        let dtc = DiagnosticTroubleCode::new(190, 3);
        let event = DiagnosticEvent::active(dtc, 0x00);
        assert!(event.is_active);
        assert_eq!(event.source_address, 0x00);
        // SPN 190 = Engine Speed
        assert_eq!(event.spn_description(), Some("Engine Speed"));
    }

    #[test]
    fn test_known_spn_descriptions() {
        assert_eq!(
            known_spn_description(110),
            Some("Engine Coolant Temperature")
        );
        assert_eq!(known_spn_description(190), Some("Engine Speed"));
        assert_eq!(known_spn_description(84), Some("Wheel-Based Vehicle Speed"));
        assert_eq!(known_spn_description(99999), None);
    }

    #[test]
    fn test_dm2_type_alias() {
        // DM2 must have same interface as DM1
        let dm2: Dm2Message = Dm1Message {
            mil_lamp: LampStatus::Off,
            rsl_lamp: LampStatus::Off,
            awl_lamp: LampStatus::Off,
            pl_lamp: LampStatus::Off,
            dtcs: vec![DiagnosticTroubleCode::new(100, 1)],
        };
        assert_eq!(dm2.fault_count(), 1);
    }

    #[test]
    fn test_dm3_dm11_pgns() {
        assert_eq!(Dm3Request::PGN, 0xFECC);
        assert_eq!(Dm11Request::PGN, 0xFED3);
    }
}
