//! ISO 15765-2 (ISO-TP) encoding and decoding for UDS PDUs.
//!
//! Provides frame-level encode/decode for all four ISO-TP frame types
//! (Single Frame, First Frame, Consecutive Frame, Flow Control) plus a
//! stateful codec that handles multi-frame message segmentation and
//! reassembly.

use crate::error::CanbusError;
use std::collections::VecDeque;

// ============================================================================
// Constants
// ============================================================================

/// Maximum CAN frame payload for ISO-TP (classic CAN 2.0).
const ISOTP_MAX_FRAME_BYTES: usize = 8;
/// Maximum single-frame data length.
const ISOTP_SF_MAX_DL: usize = 7;
/// Maximum first-frame data length (12 bits → 4095 bytes total message).
const ISOTP_MAX_MSG_LEN: usize = 4095;

// ============================================================================
// ISO-TP Frame Types
// ============================================================================

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

// ============================================================================
// Flow Control
// ============================================================================

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

// ============================================================================
// ISO-TP Frame
// ============================================================================

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
