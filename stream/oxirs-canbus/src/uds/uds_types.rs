//! UDS service type definitions – ISO 14229-1
//!
//! Contains all UDS service identifiers, session types, reset types,
//! negative response codes, and request/response structs.

use crate::error::CanbusError;

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
        use crate::uds::uds_codec::IsoTpCodec;
        let mut codec = IsoTpCodec::new();
        codec.segment(&self.payload)?;
        let mut frames = Vec::new();
        while let Some(f) = codec.next_frame() {
            frames.push(f);
        }
        Ok(frames)
    }
}
