//! UDS (Unified Diagnostic Services) – ISO 14229 implementation
//!
//! Provides a complete UDS implementation over ISO-TP (ISO 15765-2) framing,
//! covering all standard service IDs, negative response codes, and an async
//! [`UdsClient`] for communicating with ECUs over a CAN channel.
//!
//! # Protocol Stack
//!
//! ```text
//! UDS Service layer  (this module)
//!        │
//!        ▼
//! ISO 15765-2 (ISO-TP) transport  ← IsoTpCodec
//!        │
//!        ▼
//! CAN 2.0 frames  (oxirs_canbus CanFrame)
//! ```
//!
//! # Example
//!
//! ```rust
//! use oxirs_canbus::uds::{UdsRequest, UdsServiceId};
//!
//! let req = UdsRequest::new(UdsServiceId::ReadDataByIdentifier)
//!     .with_data(vec![0xF1, 0x90]);
//! assert_eq!(req.service_id, UdsServiceId::ReadDataByIdentifier);
//! ```

pub mod session_manager;

use crate::error::CanbusError;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex;

// ============================================================================
// UDS Service Identifiers – ISO 14229-1 Table 1
// ============================================================================

/// UDS service identifiers as defined by ISO 14229-1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum UdsServiceId {
    /// 0x10 – DiagnosticSessionControl
    DiagnosticSessionControl = 0x10,
    /// 0x11 – ECUReset
    EcuReset = 0x11,
    /// 0x14 – ClearDiagnosticInformation
    ClearDiagnosticInformation = 0x14,
    /// 0x19 – ReadDTCInformation
    ReadDtcInformation = 0x19,
    /// 0x22 – ReadDataByIdentifier
    ReadDataByIdentifier = 0x22,
    /// 0x23 – ReadMemoryByAddress
    ReadMemoryByAddress = 0x23,
    /// 0x24 – ReadScalingDataByIdentifier
    ReadScalingDataByIdentifier = 0x24,
    /// 0x27 – SecurityAccess
    SecurityAccess = 0x27,
    /// 0x28 – CommunicationControl
    CommunicationControl = 0x28,
    /// 0x29 – Authentication (ISO 14229-1:2020)
    Authentication = 0x29,
    /// 0x2A – ReadDataByPeriodicIdentifier
    ReadDataByPeriodicIdentifier = 0x2A,
    /// 0x2C – DynamicallyDefineDataIdentifier
    DynamicallyDefineDataIdentifier = 0x2C,
    /// 0x2E – WriteDataByIdentifier
    WriteDataByIdentifier = 0x2E,
    /// 0x2F – InputOutputControlByIdentifier
    InputOutputControlByIdentifier = 0x2F,
    /// 0x31 – RoutineControl
    RoutineControl = 0x31,
    /// 0x34 – RequestDownload
    RequestDownload = 0x34,
    /// 0x35 – RequestUpload
    RequestUpload = 0x35,
    /// 0x36 – TransferData
    TransferData = 0x36,
    /// 0x37 – RequestTransferExit
    RequestTransferExit = 0x37,
    /// 0x38 – RequestFileTransfer
    RequestFileTransfer = 0x38,
    /// 0x3D – WriteMemoryByAddress
    WriteMemoryByAddress = 0x3D,
    /// 0x3E – TesterPresent
    TesterPresent = 0x3E,
    /// 0x7F – NegativeResponse
    NegativeResponse = 0x7F,
    /// 0x83 – AccessTimingParameter
    AccessTimingParameter = 0x83,
    /// 0x84 – SecuredDataTransmission
    SecuredDataTransmission = 0x84,
    /// 0x85 – ControlDTCSetting
    ControlDtcSetting = 0x85,
    /// 0x86 – ResponseOnEvent
    ResponseOnEvent = 0x86,
    /// 0x87 – LinkControl
    LinkControl = 0x87,
}

impl UdsServiceId {
    /// Attempt to parse a raw byte as a [`UdsServiceId`].
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0x10 => Some(Self::DiagnosticSessionControl),
            0x11 => Some(Self::EcuReset),
            0x14 => Some(Self::ClearDiagnosticInformation),
            0x19 => Some(Self::ReadDtcInformation),
            0x22 => Some(Self::ReadDataByIdentifier),
            0x23 => Some(Self::ReadMemoryByAddress),
            0x24 => Some(Self::ReadScalingDataByIdentifier),
            0x27 => Some(Self::SecurityAccess),
            0x28 => Some(Self::CommunicationControl),
            0x29 => Some(Self::Authentication),
            0x2A => Some(Self::ReadDataByPeriodicIdentifier),
            0x2C => Some(Self::DynamicallyDefineDataIdentifier),
            0x2E => Some(Self::WriteDataByIdentifier),
            0x2F => Some(Self::InputOutputControlByIdentifier),
            0x31 => Some(Self::RoutineControl),
            0x34 => Some(Self::RequestDownload),
            0x35 => Some(Self::RequestUpload),
            0x36 => Some(Self::TransferData),
            0x37 => Some(Self::RequestTransferExit),
            0x38 => Some(Self::RequestFileTransfer),
            0x3D => Some(Self::WriteMemoryByAddress),
            0x3E => Some(Self::TesterPresent),
            0x7F => Some(Self::NegativeResponse),
            0x83 => Some(Self::AccessTimingParameter),
            0x84 => Some(Self::SecuredDataTransmission),
            0x85 => Some(Self::ControlDtcSetting),
            0x86 => Some(Self::ResponseOnEvent),
            0x87 => Some(Self::LinkControl),
            _ => None,
        }
    }

    /// Return the positive response service ID (request SID | 0x40).
    pub fn positive_response_id(self) -> u8 {
        (self as u8) | 0x40
    }

    /// Human-readable service name.
    pub fn name(self) -> &'static str {
        match self {
            Self::DiagnosticSessionControl => "DiagnosticSessionControl",
            Self::EcuReset => "ECUReset",
            Self::ClearDiagnosticInformation => "ClearDiagnosticInformation",
            Self::ReadDtcInformation => "ReadDTCInformation",
            Self::ReadDataByIdentifier => "ReadDataByIdentifier",
            Self::ReadMemoryByAddress => "ReadMemoryByAddress",
            Self::ReadScalingDataByIdentifier => "ReadScalingDataByIdentifier",
            Self::SecurityAccess => "SecurityAccess",
            Self::CommunicationControl => "CommunicationControl",
            Self::Authentication => "Authentication",
            Self::ReadDataByPeriodicIdentifier => "ReadDataByPeriodicIdentifier",
            Self::DynamicallyDefineDataIdentifier => "DynamicallyDefineDataIdentifier",
            Self::WriteDataByIdentifier => "WriteDataByIdentifier",
            Self::InputOutputControlByIdentifier => "InputOutputControlByIdentifier",
            Self::RoutineControl => "RoutineControl",
            Self::RequestDownload => "RequestDownload",
            Self::RequestUpload => "RequestUpload",
            Self::TransferData => "TransferData",
            Self::RequestTransferExit => "RequestTransferExit",
            Self::RequestFileTransfer => "RequestFileTransfer",
            Self::WriteMemoryByAddress => "WriteMemoryByAddress",
            Self::TesterPresent => "TesterPresent",
            Self::NegativeResponse => "NegativeResponse",
            Self::AccessTimingParameter => "AccessTimingParameter",
            Self::SecuredDataTransmission => "SecuredDataTransmission",
            Self::ControlDtcSetting => "ControlDTCSetting",
            Self::ResponseOnEvent => "ResponseOnEvent",
            Self::LinkControl => "LinkControl",
        }
    }
}

impl std::fmt::Display for UdsServiceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{:02X} ({})", *self as u8, self.name())
    }
}

// ============================================================================
// Session Types
// ============================================================================

/// Diagnostic session types for DiagnosticSessionControl (0x10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SessionType {
    /// Default diagnostic session (0x01)
    Default = 0x01,
    /// Programming session (0x02)
    Programming = 0x02,
    /// Extended diagnostic session (0x03)
    ExtendedDiagnostic = 0x03,
    /// Safety system diagnostic session (0x04)
    SafetySystem = 0x04,
}

impl SessionType {
    /// Parse from raw byte.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::Default),
            0x02 => Some(Self::Programming),
            0x03 => Some(Self::ExtendedDiagnostic),
            0x04 => Some(Self::SafetySystem),
            _ => None,
        }
    }
}

// ============================================================================
// ECU Reset Types
// ============================================================================

/// Reset types for ECUReset (0x11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ResetType {
    /// Hard reset (0x01)
    Hard = 0x01,
    /// Key off / on reset (0x02)
    KeyOffOn = 0x02,
    /// Soft reset (0x03)
    Soft = 0x03,
    /// Enable rapid power shut down (0x04)
    EnableRapidPowerShutDown = 0x04,
    /// Disable rapid power shut down (0x05)
    DisableRapidPowerShutDown = 0x05,
}

impl ResetType {
    /// Parse from raw byte.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::Hard),
            0x02 => Some(Self::KeyOffOn),
            0x03 => Some(Self::Soft),
            0x04 => Some(Self::EnableRapidPowerShutDown),
            0x05 => Some(Self::DisableRapidPowerShutDown),
            _ => None,
        }
    }
}

// ============================================================================
// Negative Response Codes – ISO 14229-1 Table A.1
// ============================================================================

/// Negative Response Code (NRC) as defined by ISO 14229-1 Annex A.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum NegativeResponseCode {
    /// 0x10 – General reject
    GeneralReject = 0x10,
    /// 0x11 – Service not supported
    ServiceNotSupported = 0x11,
    /// 0x12 – Sub-function not supported
    SubFunctionNotSupported = 0x12,
    /// 0x13 – Incorrect message length or invalid format
    IncorrectMessageLengthOrInvalidFormat = 0x13,
    /// 0x14 – Response too long
    ResponseTooLong = 0x14,
    /// 0x21 – Busy; repeat request
    BusyRepeatRequest = 0x21,
    /// 0x22 – Conditions not correct
    ConditionsNotCorrect = 0x22,
    /// 0x24 – Request sequence error
    RequestSequenceError = 0x24,
    /// 0x25 – No response from sub-net component
    NoResponseFromSubNetComponent = 0x25,
    /// 0x26 – Failure prevents execution of requested action
    FailurePreventsExecutionOfRequestedAction = 0x26,
    /// 0x31 – Request out of range
    RequestOutOfRange = 0x31,
    /// 0x33 – Security access denied
    SecurityAccessDenied = 0x33,
    /// 0x34 – Authentication required
    AuthenticationRequired = 0x34,
    /// 0x35 – Invalid key
    InvalidKey = 0x35,
    /// 0x36 – Exceeded number of attempts
    ExceededNumberOfAttempts = 0x36,
    /// 0x37 – Required time delay not expired
    RequiredTimeDelayNotExpired = 0x37,
    /// 0x38 – Secure data transmission required
    SecureDataTransmissionRequired = 0x38,
    /// 0x39 – Secure data transmission not allowed
    SecureDataTransmissionNotAllowed = 0x39,
    /// 0x3A – Secure data verification failed
    SecureDataVerificationFailed = 0x3A,
    /// 0x70 – Upload/download not accepted
    UploadDownloadNotAccepted = 0x70,
    /// 0x71 – Transfer data suspended
    TransferDataSuspended = 0x71,
    /// 0x72 – General programming failure
    GeneralProgrammingFailure = 0x72,
    /// 0x73 – Wrong block sequence counter
    WrongBlockSequenceCounter = 0x73,
    /// 0x78 – Request correctly received, response pending
    RequestCorrectlyReceivedResponsePending = 0x78,
    /// 0x7E – Sub-function not supported in active session
    SubFunctionNotSupportedInActiveSession = 0x7E,
    /// 0x7F – Service not supported in active session
    ServiceNotSupportedInActiveSession = 0x7F,
    /// 0x81 – RPM too high
    RpmTooHigh = 0x81,
    /// 0x82 – RPM too low
    RpmTooLow = 0x82,
    /// 0x83 – Engine is running
    EngineIsRunning = 0x83,
    /// 0x84 – Engine is not running
    EngineIsNotRunning = 0x84,
    /// 0x85 – Engine run time too low
    EngineRunTimeTooLow = 0x85,
    /// 0x86 – Temperature too high
    TemperatureTooHigh = 0x86,
    /// 0x87 – Temperature too low
    TemperatureTooLow = 0x87,
    /// 0x88 – Vehicle speed too high
    VehicleSpeedTooHigh = 0x88,
    /// 0x89 – Vehicle speed too low
    VehicleSpeedTooLow = 0x89,
    /// 0x8A – Throttle/pedal too high
    ThrottlePedalTooHigh = 0x8A,
    /// 0x8B – Throttle/pedal too low
    ThrottlePedalTooLow = 0x8B,
    /// 0x8C – Transmission range not in neutral
    TransmissionRangeNotInNeutral = 0x8C,
    /// 0x8D – Transmission range not in gear
    TransmissionRangeNotInGear = 0x8D,
    /// 0x8F – Brake switch(es) not closed
    BrakeSwitchNotClosed = 0x8F,
    /// 0x90 – Shifter lever not in park
    ShifterLeverNotInPark = 0x90,
    /// 0x91 – Torque converter clutch locked
    TorqueConverterClutchLocked = 0x91,
    /// 0x92 – Voltage too high
    VoltageTooHigh = 0x92,
    /// 0x93 – Voltage too low
    VoltageTooLow = 0x93,
    /// Unknown NRC
    Unknown = 0xFF,
}

impl NegativeResponseCode {
    /// Parse from raw byte.
    pub fn from_byte(b: u8) -> Self {
        match b {
            0x10 => Self::GeneralReject,
            0x11 => Self::ServiceNotSupported,
            0x12 => Self::SubFunctionNotSupported,
            0x13 => Self::IncorrectMessageLengthOrInvalidFormat,
            0x14 => Self::ResponseTooLong,
            0x21 => Self::BusyRepeatRequest,
            0x22 => Self::ConditionsNotCorrect,
            0x24 => Self::RequestSequenceError,
            0x25 => Self::NoResponseFromSubNetComponent,
            0x26 => Self::FailurePreventsExecutionOfRequestedAction,
            0x31 => Self::RequestOutOfRange,
            0x33 => Self::SecurityAccessDenied,
            0x34 => Self::AuthenticationRequired,
            0x35 => Self::InvalidKey,
            0x36 => Self::ExceededNumberOfAttempts,
            0x37 => Self::RequiredTimeDelayNotExpired,
            0x38 => Self::SecureDataTransmissionRequired,
            0x39 => Self::SecureDataTransmissionNotAllowed,
            0x3A => Self::SecureDataVerificationFailed,
            0x70 => Self::UploadDownloadNotAccepted,
            0x71 => Self::TransferDataSuspended,
            0x72 => Self::GeneralProgrammingFailure,
            0x73 => Self::WrongBlockSequenceCounter,
            0x78 => Self::RequestCorrectlyReceivedResponsePending,
            0x7E => Self::SubFunctionNotSupportedInActiveSession,
            0x7F => Self::ServiceNotSupportedInActiveSession,
            0x81 => Self::RpmTooHigh,
            0x82 => Self::RpmTooLow,
            0x83 => Self::EngineIsRunning,
            0x84 => Self::EngineIsNotRunning,
            0x85 => Self::EngineRunTimeTooLow,
            0x86 => Self::TemperatureTooHigh,
            0x87 => Self::TemperatureTooLow,
            0x88 => Self::VehicleSpeedTooHigh,
            0x89 => Self::VehicleSpeedTooLow,
            0x8A => Self::ThrottlePedalTooHigh,
            0x8B => Self::ThrottlePedalTooLow,
            0x8C => Self::TransmissionRangeNotInNeutral,
            0x8D => Self::TransmissionRangeNotInGear,
            0x8F => Self::BrakeSwitchNotClosed,
            0x90 => Self::ShifterLeverNotInPark,
            0x91 => Self::TorqueConverterClutchLocked,
            0x92 => Self::VoltageTooHigh,
            0x93 => Self::VoltageTooLow,
            _ => Self::Unknown,
        }
    }

    /// Human-readable description.
    pub fn description(self) -> &'static str {
        match self {
            Self::GeneralReject => "General reject",
            Self::ServiceNotSupported => "Service not supported",
            Self::SubFunctionNotSupported => "Sub-function not supported",
            Self::IncorrectMessageLengthOrInvalidFormat => {
                "Incorrect message length or invalid format"
            }
            Self::ResponseTooLong => "Response too long",
            Self::BusyRepeatRequest => "Busy; repeat request",
            Self::ConditionsNotCorrect => "Conditions not correct",
            Self::RequestSequenceError => "Request sequence error",
            Self::NoResponseFromSubNetComponent => "No response from sub-net component",
            Self::FailurePreventsExecutionOfRequestedAction => {
                "Failure prevents execution of requested action"
            }
            Self::RequestOutOfRange => "Request out of range",
            Self::SecurityAccessDenied => "Security access denied",
            Self::AuthenticationRequired => "Authentication required",
            Self::InvalidKey => "Invalid key",
            Self::ExceededNumberOfAttempts => "Exceeded number of attempts",
            Self::RequiredTimeDelayNotExpired => "Required time delay not expired",
            Self::SecureDataTransmissionRequired => "Secure data transmission required",
            Self::SecureDataTransmissionNotAllowed => "Secure data transmission not allowed",
            Self::SecureDataVerificationFailed => "Secure data verification failed",
            Self::UploadDownloadNotAccepted => "Upload/download not accepted",
            Self::TransferDataSuspended => "Transfer data suspended",
            Self::GeneralProgrammingFailure => "General programming failure",
            Self::WrongBlockSequenceCounter => "Wrong block sequence counter",
            Self::RequestCorrectlyReceivedResponsePending => {
                "Request correctly received, response pending"
            }
            Self::SubFunctionNotSupportedInActiveSession => {
                "Sub-function not supported in active session"
            }
            Self::ServiceNotSupportedInActiveSession => "Service not supported in active session",
            Self::RpmTooHigh => "RPM too high",
            Self::RpmTooLow => "RPM too low",
            Self::EngineIsRunning => "Engine is running",
            Self::EngineIsNotRunning => "Engine is not running",
            Self::EngineRunTimeTooLow => "Engine run time too low",
            Self::TemperatureTooHigh => "Temperature too high",
            Self::TemperatureTooLow => "Temperature too low",
            Self::VehicleSpeedTooHigh => "Vehicle speed too high",
            Self::VehicleSpeedTooLow => "Vehicle speed too low",
            Self::ThrottlePedalTooHigh => "Throttle/pedal too high",
            Self::ThrottlePedalTooLow => "Throttle/pedal too low",
            Self::TransmissionRangeNotInNeutral => "Transmission range not in neutral",
            Self::TransmissionRangeNotInGear => "Transmission range not in gear",
            Self::BrakeSwitchNotClosed => "Brake switch(es) not closed",
            Self::ShifterLeverNotInPark => "Shifter lever not in park",
            Self::TorqueConverterClutchLocked => "Torque converter clutch locked",
            Self::VoltageTooHigh => "Voltage too high",
            Self::VoltageTooLow => "Voltage too low",
            Self::Unknown => "Unknown NRC",
        }
    }
}

impl std::fmt::Display for NegativeResponseCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NRC 0x{:02X}: {}", *self as u8, self.description())
    }
}

// ============================================================================
// UDS Request / Response
// ============================================================================

/// A UDS diagnostic request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UdsRequest {
    /// Service identifier.
    pub service_id: UdsServiceId,
    /// Optional sub-function byte (bit 7 = suppress positive response).
    pub sub_function: Option<u8>,
    /// Service-specific data payload.
    pub data: Vec<u8>,
}

impl UdsRequest {
    /// Create a new request for the given service.
    pub fn new(service_id: UdsServiceId) -> Self {
        Self {
            service_id,
            sub_function: None,
            data: Vec::new(),
        }
    }

    /// Attach a sub-function byte.
    pub fn with_sub_function(mut self, sf: u8) -> Self {
        self.sub_function = Some(sf);
        self
    }

    /// Attach a data payload.
    pub fn with_data(mut self, data: Vec<u8>) -> Self {
        self.data = data;
        self
    }

    /// Encode the request into a byte stream suitable for ISO-TP transmission.
    pub fn encode(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(1 + self.sub_function.is_some() as usize + self.data.len());
        out.push(self.service_id as u8);
        if let Some(sf) = self.sub_function {
            out.push(sf);
        }
        out.extend_from_slice(&self.data);
        out
    }

    /// Decode a UDS request from a raw byte slice.
    pub fn decode(bytes: &[u8], has_sub_function: bool) -> Result<Self, CanbusError> {
        if bytes.is_empty() {
            return Err(CanbusError::Config(
                "UDS request too short: no service ID".to_string(),
            ));
        }
        let service_id = UdsServiceId::from_byte(bytes[0]).ok_or_else(|| {
            CanbusError::Config(format!("Unknown UDS service ID: 0x{:02X}", bytes[0]))
        })?;

        if has_sub_function {
            if bytes.len() < 2 {
                return Err(CanbusError::Config(
                    "UDS request missing sub-function byte".to_string(),
                ));
            }
            Ok(Self {
                service_id,
                sub_function: Some(bytes[1]),
                data: bytes[2..].to_vec(),
            })
        } else {
            Ok(Self {
                service_id,
                sub_function: None,
                data: bytes[1..].to_vec(),
            })
        }
    }
}

/// A UDS diagnostic response (positive or negative).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UdsResponse {
    /// Positive response: echoed SID | 0x40, optional sub-function, data.
    Positive {
        /// The original request service ID.
        service_id: UdsServiceId,
        /// Optional sub-function echo.
        sub_function: Option<u8>,
        /// Response payload.
        data: Vec<u8>,
    },
    /// Negative response: service ID 0x7F, original SID, NRC.
    Negative {
        /// The requested service that was rejected.
        service_id: UdsServiceId,
        /// Negative response code.
        nrc: NegativeResponseCode,
    },
}

impl UdsResponse {
    /// Decode a UDS response from a raw byte slice.
    pub fn decode(bytes: &[u8]) -> Result<Self, CanbusError> {
        if bytes.is_empty() {
            return Err(CanbusError::Config("UDS response is empty".to_string()));
        }
        if bytes[0] == 0x7F {
            // Negative response: [0x7F, SID, NRC]
            if bytes.len() < 3 {
                return Err(CanbusError::Config(
                    "UDS negative response too short (expected 3 bytes)".to_string(),
                ));
            }
            let service_id = UdsServiceId::from_byte(bytes[1]).ok_or_else(|| {
                CanbusError::Config(format!("Unknown rejected SID: 0x{:02X}", bytes[1]))
            })?;
            let nrc = NegativeResponseCode::from_byte(bytes[2]);
            return Ok(Self::Negative { service_id, nrc });
        }

        // Positive response: first byte is SID | 0x40
        let raw_sid = bytes[0] & !0x40;
        let service_id = UdsServiceId::from_byte(raw_sid).ok_or_else(|| {
            CanbusError::Config(format!(
                "Unknown positive response SID: 0x{:02X} (raw 0x{:02X})",
                raw_sid, bytes[0]
            ))
        })?;

        // Sub-function services: DiagnosticSessionControl, ECUReset, SecurityAccess,
        // CommunicationControl, TesterPresent, RoutineControl
        let has_sf = matches!(
            service_id,
            UdsServiceId::DiagnosticSessionControl
                | UdsServiceId::EcuReset
                | UdsServiceId::SecurityAccess
                | UdsServiceId::CommunicationControl
                | UdsServiceId::TesterPresent
                | UdsServiceId::RoutineControl
        );

        if has_sf && bytes.len() >= 2 {
            Ok(Self::Positive {
                service_id,
                sub_function: Some(bytes[1]),
                data: bytes[2..].to_vec(),
            })
        } else {
            Ok(Self::Positive {
                service_id,
                sub_function: None,
                data: bytes[1..].to_vec(),
            })
        }
    }

    /// Encode the response to bytes.
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Self::Positive {
                service_id,
                sub_function,
                data,
            } => {
                let mut out = Vec::new();
                out.push((*service_id as u8) | 0x40);
                if let Some(sf) = sub_function {
                    out.push(*sf);
                }
                out.extend_from_slice(data);
                out
            }
            Self::Negative { service_id, nrc } => {
                vec![0x7F, *service_id as u8, *nrc as u8]
            }
        }
    }

    /// Return `true` if this is a positive response.
    pub fn is_positive(&self) -> bool {
        matches!(self, Self::Positive { .. })
    }

    /// Return `true` if this is a negative response with the given NRC.
    pub fn is_nrc(&self, code: NegativeResponseCode) -> bool {
        matches!(self, Self::Negative { nrc, .. } if *nrc == code)
    }
}

// ============================================================================
// ISO 15765-2 (ISO-TP) Codec
// ============================================================================

/// Maximum CAN frame payload for ISO-TP (classic CAN 2.0).
const ISOTP_MAX_FRAME_BYTES: usize = 8;
/// Maximum single-frame data length.
const ISOTP_SF_MAX_DL: usize = 7;
/// Maximum first-frame data length (12 bits → 4095 bytes total message).
const ISOTP_MAX_MSG_LEN: usize = 4095;

/// ISO-TP frame type discriminants (upper nibble of first byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IsoTpFrameType {
    SingleFrame = 0x0,
    FirstFrame = 0x1,
    ConsecutiveFrame = 0x2,
    FlowControl = 0x3,
}

impl IsoTpFrameType {
    fn from_nibble(n: u8) -> Option<Self> {
        match n {
            0x0 => Some(Self::SingleFrame),
            0x1 => Some(Self::FirstFrame),
            0x2 => Some(Self::ConsecutiveFrame),
            0x3 => Some(Self::FlowControl),
            _ => None,
        }
    }
}

/// Flow control flag values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowStatus {
    /// Continue to send
    ContinueToSend = 0x00,
    /// Wait
    Wait = 0x01,
    /// Overflow/abort
    Overflow = 0x02,
}

impl FlowStatus {
    fn from_byte(b: u8) -> Option<Self> {
        match b & 0x0F {
            0x00 => Some(Self::ContinueToSend),
            0x01 => Some(Self::Wait),
            0x02 => Some(Self::Overflow),
            _ => None,
        }
    }
}

/// A decoded ISO-TP layer frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IsoTpFrame {
    /// Single Frame – carries the complete message.
    SingleFrame {
        /// Data length (1–7 bytes).
        data_len: u8,
        /// Payload bytes.
        data: Vec<u8>,
    },
    /// First Frame – first segment of a multi-frame message.
    FirstFrame {
        /// Total message length (12 bits, max 4095).
        total_len: u16,
        /// First segment payload (6 bytes).
        data: Vec<u8>,
    },
    /// Consecutive Frame – subsequent segments.
    ConsecutiveFrame {
        /// Sequence number (0–15, wraps).
        sequence_number: u8,
        /// Segment payload (up to 7 bytes).
        data: Vec<u8>,
    },
    /// Flow Control – receiver informs sender about buffer availability.
    FlowControl {
        /// Flow status.
        flow_status: FlowStatus,
        /// Block size (0 = all remaining frames without pause).
        block_size: u8,
        /// Separation time minimum in ms (0x00–0x7F) or 100–900 µs (0xF1–0xF9).
        st_min: u8,
    },
}

impl IsoTpFrame {
    /// Encode a Single Frame from a payload ≤7 bytes.
    pub fn encode_single(payload: &[u8]) -> Result<Vec<u8>, CanbusError> {
        if payload.is_empty() {
            return Err(CanbusError::Config(
                "ISO-TP single frame payload cannot be empty".to_string(),
            ));
        }
        if payload.len() > ISOTP_SF_MAX_DL {
            return Err(CanbusError::FrameTooLarge(payload.len()));
        }
        let mut out = vec![0u8; ISOTP_MAX_FRAME_BYTES];
        out[0] = payload.len() as u8; // type nibble 0x0 | data_len
        out[1..1 + payload.len()].copy_from_slice(payload);
        Ok(out)
    }

    /// Encode a First Frame from a full message whose total length is `total_len`.
    /// Returns a single 8-byte CAN frame containing the FF header + 6 data bytes.
    pub fn encode_first(total_len: usize, data_segment: &[u8]) -> Result<Vec<u8>, CanbusError> {
        if total_len > ISOTP_MAX_MSG_LEN {
            return Err(CanbusError::FrameTooLarge(total_len));
        }
        if data_segment.len() > 6 {
            return Err(CanbusError::FrameTooLarge(data_segment.len()));
        }
        let tl = total_len as u16;
        let mut out = vec![0u8; ISOTP_MAX_FRAME_BYTES];
        out[0] = 0x10 | ((tl >> 8) as u8 & 0x0F);
        out[1] = (tl & 0xFF) as u8;
        let copy_len = data_segment.len().min(6);
        out[2..2 + copy_len].copy_from_slice(&data_segment[..copy_len]);
        Ok(out)
    }

    /// Encode a Consecutive Frame.
    pub fn encode_consecutive(seq: u8, data_segment: &[u8]) -> Result<Vec<u8>, CanbusError> {
        if data_segment.len() > 7 {
            return Err(CanbusError::FrameTooLarge(data_segment.len()));
        }
        let mut out = vec![0u8; ISOTP_MAX_FRAME_BYTES];
        out[0] = 0x20 | (seq & 0x0F);
        let copy_len = data_segment.len().min(7);
        out[1..1 + copy_len].copy_from_slice(&data_segment[..copy_len]);
        Ok(out)
    }

    /// Encode a Flow Control frame.
    pub fn encode_flow_control(flow_status: FlowStatus, block_size: u8, st_min: u8) -> Vec<u8> {
        let mut out = vec![0u8; ISOTP_MAX_FRAME_BYTES];
        out[0] = 0x30 | (flow_status as u8);
        out[1] = block_size;
        out[2] = st_min;
        out
    }

    /// Decode an ISO-TP frame from a raw 8-byte CAN payload.
    pub fn decode(raw: &[u8]) -> Result<Self, CanbusError> {
        if raw.is_empty() {
            return Err(CanbusError::Config("ISO-TP raw frame is empty".to_string()));
        }
        let frame_type_nibble = (raw[0] >> 4) & 0x0F;
        let ft = IsoTpFrameType::from_nibble(frame_type_nibble).ok_or_else(|| {
            CanbusError::Config(format!(
                "Unknown ISO-TP frame type nibble: 0x{:X}",
                frame_type_nibble
            ))
        })?;

        match ft {
            IsoTpFrameType::SingleFrame => {
                let dl = raw[0] & 0x0F;
                if dl == 0 {
                    return Err(CanbusError::Config(
                        "ISO-TP single frame data length is zero".to_string(),
                    ));
                }
                let end = (1 + dl as usize).min(raw.len());
                Ok(Self::SingleFrame {
                    data_len: dl,
                    data: raw[1..end].to_vec(),
                })
            }
            IsoTpFrameType::FirstFrame => {
                if raw.len() < 2 {
                    return Err(CanbusError::Config(
                        "ISO-TP first frame too short".to_string(),
                    ));
                }
                let total_len = (((raw[0] & 0x0F) as u16) << 8) | raw[1] as u16;
                let end = raw.len().min(8);
                Ok(Self::FirstFrame {
                    total_len,
                    data: raw[2..end].to_vec(),
                })
            }
            IsoTpFrameType::ConsecutiveFrame => {
                let seq = raw[0] & 0x0F;
                let end = raw.len().min(8);
                Ok(Self::ConsecutiveFrame {
                    sequence_number: seq,
                    data: raw[1..end].to_vec(),
                })
            }
            IsoTpFrameType::FlowControl => {
                if raw.len() < 3 {
                    return Err(CanbusError::Config(
                        "ISO-TP flow control frame too short".to_string(),
                    ));
                }
                let flow_status = FlowStatus::from_byte(raw[0]).ok_or_else(|| {
                    CanbusError::Config(format!(
                        "Invalid ISO-TP flow status: 0x{:02X}",
                        raw[0] & 0x0F
                    ))
                })?;
                Ok(Self::FlowControl {
                    flow_status,
                    block_size: raw[1],
                    st_min: raw[2],
                })
            }
        }
    }
}

// ============================================================================
// ISO-TP Codec – multi-frame reassembly
// ============================================================================

/// State for reassembling a multi-frame ISO-TP message.
#[derive(Debug)]
struct IsoTpReassemblyState {
    total_len: usize,
    expected_seq: u8,
    buffer: Vec<u8>,
}

/// ISO-TP codec capable of segmenting and reassembling multi-frame messages.
///
/// The codec is **not** async; callers are expected to drive the state machine
/// via [`IsoTpCodec::feed`] for receiving and [`IsoTpCodec::segment`] for
/// sending.
#[derive(Debug)]
pub struct IsoTpCodec {
    reassembly: Option<IsoTpReassemblyState>,
    /// Outbound segments waiting to be fetched by the caller.
    outbound: VecDeque<Vec<u8>>,
}

impl IsoTpCodec {
    /// Create a new codec instance.
    pub fn new() -> Self {
        Self {
            reassembly: None,
            outbound: VecDeque::new(),
        }
    }

    /// Feed an incoming ISO-TP CAN frame into the codec.
    ///
    /// Returns `Some(complete_message)` when the full multi-frame (or
    /// single-frame) message has been reassembled, `None` if more frames are
    /// needed, or an error on protocol violation.
    pub fn feed(&mut self, raw: &[u8]) -> Result<Option<Vec<u8>>, CanbusError> {
        let frame = IsoTpFrame::decode(raw)?;
        match frame {
            IsoTpFrame::SingleFrame { data_len, data } => {
                // Reset any in-progress reassembly.
                self.reassembly = None;
                let dl = data_len as usize;
                if dl > data.len() {
                    return Err(CanbusError::Config(format!(
                        "ISO-TP SF data_len {} > available {} bytes",
                        dl,
                        data.len()
                    )));
                }
                Ok(Some(data[..dl].to_vec()))
            }
            IsoTpFrame::FirstFrame { total_len, data } => {
                let tl = total_len as usize;
                if tl > ISOTP_MAX_MSG_LEN {
                    return Err(CanbusError::FrameTooLarge(tl));
                }
                let mut buf = Vec::with_capacity(tl);
                buf.extend_from_slice(&data);
                self.reassembly = Some(IsoTpReassemblyState {
                    total_len: tl,
                    expected_seq: 1,
                    buffer: buf,
                });
                // Caller should now send a FlowControl::ContinueToSend frame.
                Ok(None)
            }
            IsoTpFrame::ConsecutiveFrame {
                sequence_number,
                data,
            } => {
                let state = self.reassembly.as_mut().ok_or_else(|| {
                    CanbusError::Config(
                        "ISO-TP consecutive frame received without prior first frame".to_string(),
                    )
                })?;
                if sequence_number != state.expected_seq {
                    return Err(CanbusError::Config(format!(
                        "ISO-TP out-of-order CF: expected seq {} got {}",
                        state.expected_seq, sequence_number
                    )));
                }
                state.buffer.extend_from_slice(&data);
                state.expected_seq = (state.expected_seq + 1) & 0x0F;

                if state.buffer.len() >= state.total_len {
                    let msg = state.buffer[..state.total_len].to_vec();
                    self.reassembly = None;
                    Ok(Some(msg))
                } else {
                    Ok(None)
                }
            }
            IsoTpFrame::FlowControl { .. } => {
                // Flow control frames drive the sender side; ignore on receiver.
                Ok(None)
            }
        }
    }

    /// Segment a full UDS payload into ISO-TP CAN frames.
    ///
    /// If the payload fits in a single frame (≤7 bytes), one frame is
    /// produced. Otherwise a First Frame + Consecutive Frames are enqueued
    /// in `outbound`.
    pub fn segment(&mut self, payload: &[u8]) -> Result<(), CanbusError> {
        self.outbound.clear();
        if payload.len() <= ISOTP_SF_MAX_DL {
            self.outbound.push_back(IsoTpFrame::encode_single(payload)?);
        } else {
            if payload.len() > ISOTP_MAX_MSG_LEN {
                return Err(CanbusError::FrameTooLarge(payload.len()));
            }
            // First frame carries bytes 0..6
            let ff = IsoTpFrame::encode_first(payload.len(), &payload[..6.min(payload.len())])?;
            self.outbound.push_back(ff);

            // Consecutive frames
            let mut offset = 6usize.min(payload.len());
            let mut seq: u8 = 1;
            while offset < payload.len() {
                let end = (offset + 7).min(payload.len());
                let cf = IsoTpFrame::encode_consecutive(seq, &payload[offset..end])?;
                self.outbound.push_back(cf);
                offset = end;
                seq = (seq + 1) & 0x0F;
            }
        }
        Ok(())
    }

    /// Retrieve the next outbound frame (if any).
    pub fn next_frame(&mut self) -> Option<Vec<u8>> {
        self.outbound.pop_front()
    }

    /// Returns `true` when there are no pending outbound frames.
    pub fn is_idle(&self) -> bool {
        self.outbound.is_empty()
    }
}

impl Default for IsoTpCodec {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// UDS Frame (convenience wrapper)
// ============================================================================

/// A UDS frame ready for ISO-TP segmentation, or decoded from a CAN channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UdsFrame {
    /// Source CAN arbitration ID (11-bit or 29-bit).
    pub source_id: u32,
    /// Destination CAN arbitration ID.
    pub dest_id: u32,
    /// Fully assembled UDS payload bytes.
    pub payload: Vec<u8>,
}

impl UdsFrame {
    /// Create a new UDS frame.
    pub fn new(source_id: u32, dest_id: u32, payload: Vec<u8>) -> Self {
        Self {
            source_id,
            dest_id,
            payload,
        }
    }

    /// Parse the payload as a [`UdsRequest`].
    ///
    /// The `has_sub_function` flag controls whether the second byte is
    /// interpreted as a sub-function.
    pub fn as_request(&self, has_sub_function: bool) -> Result<UdsRequest, CanbusError> {
        UdsRequest::decode(&self.payload, has_sub_function)
    }

    /// Parse the payload as a [`UdsResponse`].
    pub fn as_response(&self) -> Result<UdsResponse, CanbusError> {
        UdsResponse::decode(&self.payload)
    }

    /// Segment this frame's payload using ISO-TP and return all CAN-layer bytes.
    pub fn segment(&self) -> Result<Vec<Vec<u8>>, CanbusError> {
        let mut codec = IsoTpCodec::new();
        codec.segment(&self.payload)?;
        let mut frames = Vec::new();
        while let Some(f) = codec.next_frame() {
            frames.push(f);
        }
        Ok(frames)
    }
}

// ============================================================================
// UDS Client
// ============================================================================

/// An async UDS client for sending service requests to an ECU over CAN.
///
/// The client uses a pair of CAN arbitration IDs (tester → ECU and
/// ECU → tester) and wraps them with ISO-TP framing.  Actual CAN I/O is
/// abstracted through a [`UdsTransport`] trait so that tests can inject a
/// loopback transport without needing real hardware.
pub trait UdsTransport: Send + Sync {
    /// Send a single 8-byte ISO-TP CAN frame to the ECU.
    fn send_frame(&self, can_id: u32, data: &[u8]) -> Result<(), CanbusError>;
    /// Receive the next available 8-byte ISO-TP CAN frame from the ECU.
    fn recv_frame(&self) -> Result<Option<Vec<u8>>, CanbusError>;
}

/// A simple loopback transport used in tests.
#[derive(Debug, Default)]
pub struct LoopbackTransport {
    queue: Arc<Mutex<VecDeque<Vec<u8>>>>,
}

impl LoopbackTransport {
    /// Create a new loopback transport.
    pub fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Inject raw bytes as if they arrived from the ECU.
    pub async fn inject(&self, data: Vec<u8>) {
        let mut q = self.queue.lock().await;
        q.push_back(data);
    }
}

impl UdsTransport for LoopbackTransport {
    fn send_frame(&self, _can_id: u32, _data: &[u8]) -> Result<(), CanbusError> {
        // Loopback: discard sent frames (for testing caller behaviour).
        Ok(())
    }

    fn recv_frame(&self) -> Result<Option<Vec<u8>>, CanbusError> {
        // Synchronous try-lock: works in unit tests where we pre-fill the queue.
        match self.queue.try_lock() {
            Ok(mut q) => Ok(q.pop_front()),
            Err(_) => Ok(None),
        }
    }
}

/// Async UDS client.
pub struct UdsClient<T: UdsTransport> {
    transport: T,
    request_id: u32,
    /// Response CAN ID (ECU → tester). Not used for filtering in this implementation
    /// but stored for introspection via [`UdsClient::response_id`].
    response_id: u32,
    codec: IsoTpCodec,
}

impl<T: UdsTransport> UdsClient<T> {
    /// Return the configured response (ECU → tester) CAN arbitration ID.
    pub fn response_can_id(&self) -> u32 {
        self.response_id
    }

    /// Return the configured request (tester → ECU) CAN arbitration ID.
    pub fn request_can_id(&self) -> u32 {
        self.request_id
    }

    /// Create a new client.
    ///
    /// * `transport`    – CAN transport implementation.
    /// * `request_id`   – CAN arbitration ID to use when sending (tester → ECU).
    /// * `response_id`  – CAN arbitration ID expected from ECU responses.
    pub fn new(transport: T, request_id: u32, response_id: u32) -> Self {
        Self {
            transport,
            request_id,
            response_id,
            codec: IsoTpCodec::new(),
        }
    }

    /// Send a raw UDS payload to the ECU (handles ISO-TP segmentation).
    fn send_request(&mut self, payload: &[u8]) -> Result<(), CanbusError> {
        self.codec.segment(payload)?;
        while let Some(frame) = self.codec.next_frame() {
            self.transport.send_frame(self.request_id, &frame)?;
        }
        Ok(())
    }

    /// Receive a complete UDS payload from the ECU (handles ISO-TP reassembly).
    ///
    /// Polls `recv_frame` until a complete message is assembled.
    /// Returns an error if no frames are available.
    fn recv_response(&mut self) -> Result<Vec<u8>, CanbusError> {
        loop {
            let raw = self
                .transport
                .recv_frame()?
                .ok_or_else(|| CanbusError::Config("No UDS response available".to_string()))?;
            if let Some(msg) = self.codec.feed(&raw)? {
                return Ok(msg);
            }
        }
    }

    /// Perform a complete UDS request → response exchange.
    pub fn exchange(&mut self, request: UdsRequest) -> Result<UdsResponse, CanbusError> {
        let payload = request.encode();
        self.send_request(&payload)?;
        let raw_resp = self.recv_response()?;
        let resp = UdsResponse::decode(&raw_resp)?;

        // Keep-alive: if NRC 0x78 (response pending) re-read.
        if resp.is_nrc(NegativeResponseCode::RequestCorrectlyReceivedResponsePending) {
            let raw2 = self.recv_response()?;
            return UdsResponse::decode(&raw2);
        }
        Ok(resp)
    }

    // -----------------------------------------------------------------------
    // High-level service helpers
    // -----------------------------------------------------------------------

    /// 0x10 – DiagnosticSessionControl.
    pub fn diagnostic_session_control(
        &mut self,
        session_type: SessionType,
    ) -> Result<(), CanbusError> {
        let req = UdsRequest::new(UdsServiceId::DiagnosticSessionControl)
            .with_sub_function(session_type as u8);
        let resp = self.exchange(req)?;
        match resp {
            UdsResponse::Positive { .. } => Ok(()),
            UdsResponse::Negative { nrc, .. } => Err(CanbusError::Config(format!(
                "DiagnosticSessionControl rejected: {}",
                nrc
            ))),
        }
    }

    /// 0x11 – ECUReset.
    pub fn ecu_reset(&mut self, reset_type: ResetType) -> Result<(), CanbusError> {
        let req = UdsRequest::new(UdsServiceId::EcuReset).with_sub_function(reset_type as u8);
        let resp = self.exchange(req)?;
        match resp {
            UdsResponse::Positive { .. } => Ok(()),
            UdsResponse::Negative { nrc, .. } => {
                Err(CanbusError::Config(format!("ECUReset rejected: {}", nrc)))
            }
        }
    }

    /// 0x22 – ReadDataByIdentifier.
    ///
    /// Returns the raw data record for the given 16-bit data identifier.
    pub fn read_data_by_id(&mut self, data_id: u16) -> Result<Vec<u8>, CanbusError> {
        let hi = (data_id >> 8) as u8;
        let lo = (data_id & 0xFF) as u8;
        let req = UdsRequest::new(UdsServiceId::ReadDataByIdentifier).with_data(vec![hi, lo]);
        let resp = self.exchange(req)?;
        match resp {
            UdsResponse::Positive { data, .. } => {
                // Response data: [hi, lo, record...]
                if data.len() < 2 {
                    return Err(CanbusError::Config(
                        "ReadDataByIdentifier response too short".to_string(),
                    ));
                }
                Ok(data[2..].to_vec())
            }
            UdsResponse::Negative { nrc, .. } => Err(CanbusError::Config(format!(
                "ReadDataByIdentifier 0x{:04X} rejected: {}",
                data_id, nrc
            ))),
        }
    }

    /// 0x2E – WriteDataByIdentifier.
    pub fn write_data_by_id(&mut self, data_id: u16, data: &[u8]) -> Result<(), CanbusError> {
        let hi = (data_id >> 8) as u8;
        let lo = (data_id & 0xFF) as u8;
        let mut payload = vec![hi, lo];
        payload.extend_from_slice(data);
        let req = UdsRequest::new(UdsServiceId::WriteDataByIdentifier).with_data(payload);
        let resp = self.exchange(req)?;
        match resp {
            UdsResponse::Positive { .. } => Ok(()),
            UdsResponse::Negative { nrc, .. } => Err(CanbusError::Config(format!(
                "WriteDataByIdentifier 0x{:04X} rejected: {}",
                data_id, nrc
            ))),
        }
    }

    /// 0x27 – SecurityAccess: request seed (level must be odd).
    pub fn security_access_seed(&mut self, level: u8) -> Result<Vec<u8>, CanbusError> {
        let req = UdsRequest::new(UdsServiceId::SecurityAccess).with_sub_function(level);
        let resp = self.exchange(req)?;
        match resp {
            UdsResponse::Positive { data, .. } => Ok(data),
            UdsResponse::Negative { nrc, .. } => Err(CanbusError::Config(format!(
                "SecurityAccess seed request rejected: {}",
                nrc
            ))),
        }
    }

    /// 0x27 – SecurityAccess: send key (level must be even = request level + 1).
    pub fn security_access_key(&mut self, level: u8, key: &[u8]) -> Result<(), CanbusError> {
        let req = UdsRequest::new(UdsServiceId::SecurityAccess)
            .with_sub_function(level)
            .with_data(key.to_vec());
        let resp = self.exchange(req)?;
        match resp {
            UdsResponse::Positive { .. } => Ok(()),
            UdsResponse::Negative { nrc, .. } => Err(CanbusError::Config(format!(
                "SecurityAccess key rejected: {}",
                nrc
            ))),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ISO-TP Single Frame -----------------------------------------------

    #[test]
    fn test_isotp_sf_encode_decode_round_trip() {
        let payload = [0x10u8, 0x03]; // DSC ExtendedDiagnostic
        let encoded = IsoTpFrame::encode_single(&payload).expect("encode SF");
        assert_eq!(encoded[0], 0x02, "SF: type nibble 0 | DL 2");
        assert_eq!(&encoded[1..3], &payload);

        let decoded = IsoTpFrame::decode(&encoded).expect("decode SF");
        match decoded {
            IsoTpFrame::SingleFrame { data_len, data } => {
                assert_eq!(data_len, 2);
                assert_eq!(&data[..2], &payload);
            }
            _ => panic!("expected SingleFrame"),
        }
    }

    #[test]
    fn test_isotp_sf_max_payload() {
        let payload = [0xAAu8; 7];
        let encoded = IsoTpFrame::encode_single(&payload).expect("encode max SF");
        assert_eq!(encoded[0], 0x07);
    }

    #[test]
    fn test_isotp_sf_too_large_rejected() {
        let payload = [0xBBu8; 8];
        assert!(IsoTpFrame::encode_single(&payload).is_err());
    }

    #[test]
    fn test_isotp_ff_encode_decode() {
        let first_6 = [0x22u8, 0xF1, 0x90, 0x00, 0x00, 0x00];
        let encoded = IsoTpFrame::encode_first(20, &first_6).expect("encode FF");
        assert_eq!(encoded[0], 0x10); // type=1, len_hi=0
        assert_eq!(encoded[1], 20); // len_lo=20
        assert_eq!(&encoded[2..8], &first_6);

        let decoded = IsoTpFrame::decode(&encoded).expect("decode FF");
        match decoded {
            IsoTpFrame::FirstFrame { total_len, data } => {
                assert_eq!(total_len, 20);
                assert_eq!(&data[..6], &first_6);
            }
            _ => panic!("expected FirstFrame"),
        }
    }

    #[test]
    fn test_isotp_cf_encode_decode() {
        let seg = [0x01u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
        let encoded = IsoTpFrame::encode_consecutive(1, &seg).expect("encode CF");
        assert_eq!(encoded[0], 0x21); // type=2 | seq=1
        assert_eq!(&encoded[1..8], &seg);

        let decoded = IsoTpFrame::decode(&encoded).expect("decode CF");
        match decoded {
            IsoTpFrame::ConsecutiveFrame {
                sequence_number,
                data,
            } => {
                assert_eq!(sequence_number, 1);
                assert_eq!(&data[..7], &seg);
            }
            _ => panic!("expected ConsecutiveFrame"),
        }
    }

    #[test]
    fn test_isotp_flow_control_encode_decode() {
        let encoded = IsoTpFrame::encode_flow_control(FlowStatus::ContinueToSend, 0, 25);
        assert_eq!(encoded[0], 0x30);
        assert_eq!(encoded[1], 0x00);
        assert_eq!(encoded[2], 25);

        let decoded = IsoTpFrame::decode(&encoded).expect("decode FC");
        match decoded {
            IsoTpFrame::FlowControl {
                flow_status,
                block_size,
                st_min,
            } => {
                assert_eq!(flow_status, FlowStatus::ContinueToSend);
                assert_eq!(block_size, 0);
                assert_eq!(st_min, 25);
            }
            _ => panic!("expected FlowControl"),
        }
    }

    // ---- IsoTpCodec reassembly ---------------------------------------------

    #[test]
    fn test_codec_single_frame_reassembly() {
        let mut codec = IsoTpCodec::new();
        let payload = [0x10u8, 0x03]; // DSC request
        let frame = IsoTpFrame::encode_single(&payload).expect("encode");
        let result = codec.feed(&frame).expect("feed");
        assert_eq!(result, Some(vec![0x10, 0x03]));
    }

    #[test]
    fn test_codec_multi_frame_reassembly() {
        let mut codec = IsoTpCodec::new();
        // Build a 13-byte message split across FF + 2×CF
        let msg: Vec<u8> = (0u8..13).collect();

        // First frame: total_len=13, first 6 bytes
        let ff = IsoTpFrame::encode_first(13, &msg[..6]).expect("FF");
        let r1 = codec.feed(&ff).expect("feed FF");
        assert!(r1.is_none(), "not complete after FF");

        // Consecutive frame 1: bytes 6..13 (7 bytes)
        let cf1 = IsoTpFrame::encode_consecutive(1, &msg[6..13]).expect("CF1");
        let r2 = codec.feed(&cf1).expect("feed CF1");
        assert_eq!(r2, Some(msg));
    }

    #[test]
    fn test_codec_segmentation() {
        let mut codec = IsoTpCodec::new();
        // 13 bytes: should produce FF + CF
        let payload: Vec<u8> = (0u8..13).collect();
        codec.segment(&payload).expect("segment");

        let mut frames = Vec::new();
        while let Some(f) = codec.next_frame() {
            frames.push(f);
        }
        assert_eq!(frames.len(), 2, "FF + 1 CF");

        // Decode them back
        let mut rx = IsoTpCodec::new();
        let r1 = rx.feed(&frames[0]).expect("feed FF");
        assert!(r1.is_none());
        let r2 = rx.feed(&frames[1]).expect("feed CF");
        assert_eq!(r2.as_deref(), Some(payload.as_slice()));
    }

    // ---- UDS Request / Response --------------------------------------------

    #[test]
    fn test_uds_request_encode_decode_dsc() {
        let req = UdsRequest::new(UdsServiceId::DiagnosticSessionControl)
            .with_sub_function(SessionType::ExtendedDiagnostic as u8);
        let encoded = req.encode();
        assert_eq!(encoded, vec![0x10, 0x03]);

        let decoded = UdsRequest::decode(&encoded, true).expect("decode DSC");
        assert_eq!(decoded.service_id, UdsServiceId::DiagnosticSessionControl);
        assert_eq!(decoded.sub_function, Some(0x03));
    }

    #[test]
    fn test_uds_request_encode_decode_rdbi() {
        let req = UdsRequest::new(UdsServiceId::ReadDataByIdentifier).with_data(vec![0xF1, 0x90]); // VIN data ID
        let encoded = req.encode();
        assert_eq!(encoded, vec![0x22, 0xF1, 0x90]);

        let decoded = UdsRequest::decode(&encoded, false).expect("decode RDBI");
        assert_eq!(decoded.service_id, UdsServiceId::ReadDataByIdentifier);
        assert_eq!(decoded.data, vec![0xF1, 0x90]);
    }

    #[test]
    fn test_uds_response_positive_decode() {
        // Positive DSC response
        let raw = vec![0x50u8, 0x03, 0x00, 0x19, 0x01, 0xF4];
        let resp = UdsResponse::decode(&raw).expect("decode positive DSC resp");
        match resp {
            UdsResponse::Positive {
                service_id,
                sub_function,
                data,
            } => {
                assert_eq!(service_id, UdsServiceId::DiagnosticSessionControl);
                assert_eq!(sub_function, Some(0x03));
                assert_eq!(data, vec![0x00, 0x19, 0x01, 0xF4]);
            }
            _ => panic!("expected positive"),
        }
    }

    #[test]
    fn test_uds_response_negative_decode() {
        let raw = vec![0x7Fu8, 0x22, 0x31]; // NRC RequestOutOfRange for RDBI
        let resp = UdsResponse::decode(&raw).expect("decode negative");
        match resp {
            UdsResponse::Negative { service_id, nrc } => {
                assert_eq!(service_id, UdsServiceId::ReadDataByIdentifier);
                assert_eq!(nrc, NegativeResponseCode::RequestOutOfRange);
            }
            _ => panic!("expected negative"),
        }
    }

    #[test]
    fn test_uds_response_encode_negative() {
        let resp = UdsResponse::Negative {
            service_id: UdsServiceId::SecurityAccess,
            nrc: NegativeResponseCode::InvalidKey,
        };
        let encoded = resp.encode();
        assert_eq!(encoded, vec![0x7F, 0x27, 0x35]);
    }

    // ---- NRC codes ---------------------------------------------------------

    #[test]
    fn test_nrc_round_trip_all_known() {
        let known: &[(u8, NegativeResponseCode)] = &[
            (0x10, NegativeResponseCode::GeneralReject),
            (0x11, NegativeResponseCode::ServiceNotSupported),
            (0x12, NegativeResponseCode::SubFunctionNotSupported),
            (
                0x13,
                NegativeResponseCode::IncorrectMessageLengthOrInvalidFormat,
            ),
            (0x22, NegativeResponseCode::ConditionsNotCorrect),
            (0x31, NegativeResponseCode::RequestOutOfRange),
            (0x33, NegativeResponseCode::SecurityAccessDenied),
            (0x35, NegativeResponseCode::InvalidKey),
            (
                0x78,
                NegativeResponseCode::RequestCorrectlyReceivedResponsePending,
            ),
            (
                0x7F,
                NegativeResponseCode::ServiceNotSupportedInActiveSession,
            ),
        ];
        for &(raw, expected) in known {
            let got = NegativeResponseCode::from_byte(raw);
            assert_eq!(got, expected, "NRC 0x{:02X}", raw);
            assert_eq!(got as u8, raw);
        }
    }

    #[test]
    fn test_nrc_description_non_empty() {
        for raw in 0u8..=0xFFu8 {
            let nrc = NegativeResponseCode::from_byte(raw);
            assert!(!nrc.description().is_empty());
        }
    }

    // ---- Service ID helpers ------------------------------------------------

    #[test]
    fn test_service_id_positive_response_ids() {
        assert_eq!(
            UdsServiceId::ReadDataByIdentifier.positive_response_id(),
            0x62
        );
        assert_eq!(
            UdsServiceId::DiagnosticSessionControl.positive_response_id(),
            0x50
        );
        assert_eq!(UdsServiceId::SecurityAccess.positive_response_id(), 0x67);
    }

    #[test]
    fn test_service_id_roundtrip() {
        let ids: &[u8] = &[
            0x10, 0x11, 0x14, 0x19, 0x22, 0x27, 0x28, 0x2E, 0x2F, 0x31, 0x34, 0x36, 0x37, 0x3E,
            0x7F,
        ];
        for &id in ids {
            let sid = UdsServiceId::from_byte(id);
            assert!(sid.is_some(), "SID 0x{:02X} should parse", id);
            assert_eq!(sid.expect("SID should parse") as u8, id);
        }
    }

    // ---- UDS Client loopback -----------------------------------------------

    #[test]
    fn test_uds_client_rdbi_loopback() {
        let transport = LoopbackTransport::new();

        // Pre-inject a positive RDBI response for DID 0xF190 (VIN):
        // Positive: [0x62, 0xF1, 0x90, 0x57, 0x30, 0x52]
        // This fits in a single ISO-TP frame: [0x06, 0x62, 0xF1, 0x90, 0x57, 0x30, 0x52, 0x00]
        let resp_payload = vec![0x62u8, 0xF1, 0x90, 0x57, 0x30, 0x52];
        let isotp_frame = IsoTpFrame::encode_single(&resp_payload).expect("encode resp");

        // We can't use async inject in a sync test; use try_lock approach
        {
            let mut q = transport.queue.try_lock().expect("lock");
            q.push_back(isotp_frame);
        }

        let mut client = UdsClient::new(transport, 0x7DF, 0x7E8);
        let record = client.read_data_by_id(0xF190).expect("RDBI");
        assert_eq!(record, vec![0x57, 0x30, 0x52]);
    }

    #[test]
    fn test_uds_client_security_access_loopback() {
        let transport = LoopbackTransport::new();

        // Inject seed response: [0x67, 0x01, 0xAA, 0xBB]
        let seed_resp = vec![0x67u8, 0x01, 0xAA, 0xBB];
        let seed_frame = IsoTpFrame::encode_single(&seed_resp).expect("encode seed resp");
        {
            let mut q = transport.queue.try_lock().expect("lock");
            q.push_back(seed_frame);
        }

        let mut client = UdsClient::new(transport, 0x7DF, 0x7E8);
        let seed = client.security_access_seed(0x01).expect("seed");
        // Sub-function is consumed; data is the remaining bytes after sf
        // Response: [0x67, sub_fn=0x01, 0xAA, 0xBB] → Positive { sf=0x01, data=[0xAA, 0xBB] }
        assert_eq!(seed, vec![0xAA, 0xBB]);
    }

    #[test]
    fn test_uds_client_negative_response() {
        let transport = LoopbackTransport::new();

        // Inject NRC SecurityAccessDenied
        let nrc_resp = vec![0x7Fu8, 0x27, 0x33];
        let nrc_frame = IsoTpFrame::encode_single(&nrc_resp).expect("encode nrc");
        {
            let mut q = transport.queue.try_lock().expect("lock");
            q.push_back(nrc_frame);
        }

        let mut client = UdsClient::new(transport, 0x7DF, 0x7E8);
        let result = client.security_access_seed(0x01);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("rejected") || err_msg.contains("denied") || err_msg.contains("NRC"),
            "error should mention rejection: {}",
            err_msg
        );
    }
}
