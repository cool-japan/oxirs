//! UDS service implementations – ISO 14229 high-level helpers.
//!
//! Provides the [`UdsTransport`] trait, the loopback transport for testing,
//! and the [`UdsClient`] for performing complete request/response exchanges
//! over ISO-TP.

use crate::error::CanbusError;
use crate::uds::uds_codec::IsoTpCodec;
use crate::uds::uds_types::{
    NegativeResponseCode, ResetType, SessionType, UdsRequest, UdsResponse, UdsServiceId,
};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex;

// ============================================================================
// Transport trait
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

// ============================================================================
// Loopback transport
// ============================================================================

/// A simple loopback transport used in tests.
#[derive(Debug, Default)]
pub struct LoopbackTransport {
    /// Internal queue of frames, accessible in tests via try_lock.
    pub queue: Arc<Mutex<VecDeque<Vec<u8>>>>,
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

// ============================================================================
// UDS Client
// ============================================================================

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
