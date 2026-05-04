//! J1939 source facade: trait definition and mock implementation.
//!
//! The [`J1939SourceFacade`] trait abstracts over the data source for J1939 CAN
//! frames, allowing real SocketCAN sources and mock sources to be used
//! interchangeably in the bridge. The [`MockJ1939Source`] implementation yields
//! pre-programmed frames from an in-memory queue.

use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// J1939Frame
// ─────────────────────────────────────────────────────────────────────────────

/// A minimal SAE J1939 CAN frame as consumed by the digital-twin bridge.
///
/// This is deliberately simpler than the full [`crate::protocol::J1939Message`]
/// to avoid coupling the bridge to the richer protocol layer.  Each frame carries
/// the decoded Parameter Group Number, the source address, and the raw 8-byte
/// data payload.
#[derive(Debug, Clone, PartialEq)]
pub struct J1939Frame {
    /// Decoded Parameter Group Number from the 29-bit CAN identifier.
    pub pgn: u32,
    /// Source address from the 29-bit CAN identifier.
    pub sa: u8,
    /// 8-byte CAN data field (J1939 always uses exactly 8 data bytes).
    pub data: [u8; 8],
}

impl J1939Frame {
    /// Construct a new frame.
    pub fn new(pgn: u32, sa: u8, data: [u8; 8]) -> Self {
        Self { pgn, sa, data }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that may be returned by a [`J1939SourceFacade`] implementation.
#[derive(Debug, thiserror::Error)]
pub enum J1939SourceError {
    /// The source has no more frames and will never produce another one.
    /// The bridge should treat this as a clean shutdown signal.
    #[error("J1939 source exhausted")]
    Exhausted,
    /// A transient or fatal I/O error from the underlying transport.
    #[error("J1939 source I/O error: {0}")]
    Io(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// J1939SourceFacade trait
// ─────────────────────────────────────────────────────────────────────────────

/// Asynchronous source of raw J1939 frames.
///
/// Implementors may wrap a Linux SocketCAN socket, a virtual loopback, a UDP
/// replay stream, or (for tests) an in-memory queue.
///
/// # Contract
///
/// - A call to `next_frame` may suspend until a frame is available.
/// - When no more frames will ever be produced, the implementation must return
///   [`J1939SourceError::Exhausted`].
/// - Transient errors should be returned as [`J1939SourceError::Io`]; the bridge
///   will propagate them and stop.
#[async_trait::async_trait]
pub trait J1939SourceFacade: Send + Sync {
    /// Await the next available J1939 frame.
    ///
    /// Returns `Ok(frame)` on success, `Err(J1939SourceError::Exhausted)` when
    /// the source is permanently empty, or `Err(J1939SourceError::Io(_))` on
    /// transient transport failures.
    async fn next_frame(&mut self) -> Result<J1939Frame, J1939SourceError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// MockJ1939Source
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory J1939 source that yields pre-programmed frames.
///
/// Frames are returned in FIFO order.  Once the queue is drained,
/// [`J1939SourceError::Exhausted`] is returned on every subsequent call.
///
/// # Example
///
/// ```
/// use oxirs_canbus::digital_twin::client::{J1939Frame, J1939SourceFacade, MockJ1939Source};
///
/// # #[tokio::main]
/// # async fn main() {
/// let frames = vec![
///     J1939Frame::new(65262, 0x00, [75, 0, 0, 0, 0, 0, 0, 0]),
/// ];
/// let mut src = MockJ1939Source::new(frames);
/// let frame = src.next_frame().await.expect("should yield a frame");
/// assert_eq!(frame.pgn, 65262);
/// # }
/// ```
pub struct MockJ1939Source {
    frames: VecDeque<J1939Frame>,
}

impl MockJ1939Source {
    /// Construct a new mock source from a vector of frames.
    pub fn new(frames: Vec<J1939Frame>) -> Self {
        Self {
            frames: frames.into(),
        }
    }

    /// Returns how many frames remain in the queue.
    pub fn remaining(&self) -> usize {
        self.frames.len()
    }
}

#[async_trait::async_trait]
impl J1939SourceFacade for MockJ1939Source {
    async fn next_frame(&mut self) -> Result<J1939Frame, J1939SourceError> {
        self.frames.pop_front().ok_or(J1939SourceError::Exhausted)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_source_yields_frames_in_order() {
        let frames = vec![
            J1939Frame::new(65262, 0x00, [75, 0, 0, 0, 0, 0, 0, 0]),
            J1939Frame::new(65265, 0x01, [0x00, 0x64, 0, 0, 0, 0, 0, 0]),
        ];
        let mut src = MockJ1939Source::new(frames.clone());
        let f1 = src.next_frame().await.expect("first frame");
        assert_eq!(f1.pgn, 65262);
        let f2 = src.next_frame().await.expect("second frame");
        assert_eq!(f2.pgn, 65265);
    }

    #[tokio::test]
    async fn mock_source_exhausted_after_all_frames() {
        let mut src = MockJ1939Source::new(vec![J1939Frame::new(61444, 0, [0u8; 8])]);
        let _ = src.next_frame().await.expect("first frame");
        let err = src.next_frame().await.expect_err("should be exhausted");
        assert!(matches!(err, J1939SourceError::Exhausted));
    }

    #[tokio::test]
    async fn mock_source_empty_from_start() {
        let mut src = MockJ1939Source::new(vec![]);
        let err = src
            .next_frame()
            .await
            .expect_err("should be exhausted immediately");
        assert!(matches!(err, J1939SourceError::Exhausted));
        assert_eq!(src.remaining(), 0);
    }

    #[test]
    fn j1939_frame_accessors() {
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let frame = J1939Frame::new(61444, 0x11, data);
        assert_eq!(frame.pgn, 61444);
        assert_eq!(frame.sa, 0x11);
        assert_eq!(frame.data, data);
    }

    #[test]
    fn j1939_frame_equality() {
        let a = J1939Frame::new(65262, 0, [0u8; 8]);
        let b = J1939Frame::new(65262, 0, [0u8; 8]);
        assert_eq!(a, b);
    }
}
