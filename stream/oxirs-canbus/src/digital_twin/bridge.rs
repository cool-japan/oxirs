//! J1939 ↔ DTDL bridge: the core event loop.
//!
//! [`J1939DtdlBridge`] connects a [`J1939SourceFacade`] (upstream) to a
//! [`DtdlSinkFacade`] (downstream), translating incoming J1939 frames into
//! DTDL property writes according to the [`BridgeConfig`] mapping table.
//!
//! # Lifecycle
//!
//! 1. Call [`J1939DtdlBridge::new`] with a config, source, sink, and
//!    [`CancellationToken`].
//! 2. Call [`J1939DtdlBridge::run`].  The bridge loops until:
//!    - The cancellation token is cancelled (clean shutdown → `Ok(())`).
//!    - The source returns [`J1939SourceError::Exhausted`] (clean end-of-stream
//!      → `Ok(())`).
//!    - The source returns a hard I/O error (`Err(BridgeError::Source(…))`).
//!    - The sink returns an error (`Err(BridgeError::Sink(…))`).
//!
//! # Frame processing
//!
//! For each received J1939 frame:
//!
//! 1. Look up the frame's PGN + SPN=0 in the mapper's read-direction table.
//! 2. Extract the raw byte from data[0] via [`extract_spn`].
//! 3. Convert to a [`TwinValue`] via [`spn_to_twin_value`] — skipping the write
//!    if the J1939 not-available indicator (0xFE / 0xFF) is present.
//! 4. Call [`DtdlSinkFacade::set_property`] with the mapped property name.
//!
//! Frames whose PGN has no entry in the mapping table are silently ignored.

use tokio_util::sync::CancellationToken;

use super::client::{J1939Frame, J1939SourceError, J1939SourceFacade};
use super::config::BridgeConfig;
use super::mapper::{extract_spn, spn_to_twin_value, PropertyMapper};
use super::sink::DtdlSinkFacade;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors returned by [`J1939DtdlBridge::run`].
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    /// An error from the upstream J1939 source (other than exhaustion, which is
    /// treated as a clean shutdown).
    #[error("J1939 source error: {0}")]
    Source(#[from] J1939SourceError),
    /// An error from the downstream DTDL sink.
    #[error("DTDL sink error: {0}")]
    Sink(String),
    /// A frame arrived for a PGN that has no mapping in the current config.
    ///
    /// The bridge does **not** return this error by default; it is available for
    /// callers that want strict mode.
    #[error("no mapping for PGN {0}")]
    NoMapping(u32),
}

// ─────────────────────────────────────────────────────────────────────────────
// J1939DtdlBridge
// ─────────────────────────────────────────────────────────────────────────────

/// J1939 ↔ DTDL bridge.
///
/// Generic over any `S: J1939SourceFacade` and `D: DtdlSinkFacade`, so it can
/// be driven by a real SocketCAN interface and an Azure Digital Twins client, or
/// by mocks in unit/integration tests.
pub struct J1939DtdlBridge<S, D>
where
    S: J1939SourceFacade,
    D: DtdlSinkFacade,
{
    config: BridgeConfig,
    source: S,
    sink: D,
    cancel: CancellationToken,
}

impl<S, D> J1939DtdlBridge<S, D>
where
    S: J1939SourceFacade,
    D: DtdlSinkFacade,
{
    /// Construct a new bridge.
    ///
    /// # Parameters
    ///
    /// - `config`: TOML-deserialised bridge configuration with PGN/SPN mappings.
    /// - `source`: upstream J1939 frame provider.
    /// - `sink`: downstream DTDL property store.
    /// - `cancel`: token used to signal a graceful shutdown from outside the
    ///   bridge's event loop.
    pub fn new(config: BridgeConfig, source: S, sink: D, cancel: CancellationToken) -> Self {
        Self {
            config,
            source,
            sink,
            cancel,
        }
    }

    /// Run the bridge event loop.
    ///
    /// Processes frames until the cancellation token fires, the source is
    /// exhausted, or an unrecoverable error occurs.
    ///
    /// # Errors
    ///
    /// - [`BridgeError::Source`] — the source returned a hard I/O error.
    /// - [`BridgeError::Sink`] — the sink returned an error on a property write.
    pub async fn run(&mut self) -> Result<(), BridgeError> {
        let mapper = PropertyMapper::new(self.config.mappings.clone());

        loop {
            tokio::select! {
                // Honour external cancellation first (biased select).
                biased;
                _ = self.cancel.cancelled() => {
                    return Ok(());
                }
                frame_result = self.source.next_frame() => {
                    match frame_result {
                        Ok(frame) => {
                            self.process_frame(&frame, &mapper).await?;
                        }
                        Err(J1939SourceError::Exhausted) => {
                            // Clean end of stream — not an error.
                            return Ok(());
                        }
                        Err(e) => {
                            return Err(BridgeError::Source(e));
                        }
                    }
                }
            }
        }
    }

    /// Translate a single J1939 frame and write to the DTDL sink.
    async fn process_frame(
        &self,
        frame: &J1939Frame,
        mapper: &PropertyMapper,
    ) -> Result<(), BridgeError> {
        // Use SPN=0 as the lookup key; the mapper treats the PGN as the primary
        // discriminator and SPN=0 is the "whole-PGN" sentinel in our simplified model.
        // For real J1939-71 precision, the SPN would be decoded from the data field
        // and the mapper would be queried with the actual SPN.
        let spn_sentinel = 0u32;

        if let Some(prop_name) = mapper.find_property(frame.pgn, spn_sentinel) {
            let spn_val = extract_spn(&frame.data, spn_sentinel);
            if let Some(twin_val) = spn_to_twin_value(spn_val) {
                self.sink
                    .set_property(prop_name, twin_val)
                    .await
                    .map_err(|e| BridgeError::Sink(e.to_string()))?;
            }
            // NotAvailable → skip write; no error.
        }
        // Unmapped PGN → silently ignored (lenient mode).
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digital_twin::client::{J1939Frame, MockJ1939Source};
    use crate::digital_twin::config::{BridgeConfig, MappingDirection, RegisterMapping};
    use crate::digital_twin::sink::MockDtdlSink;
    use oxirs_physics::digital_twin::twin_value::TwinValue;

    fn et1_config() -> BridgeConfig {
        BridgeConfig {
            poll_interval_ms: 10,
            mappings: vec![RegisterMapping {
                pgn: 65262,
                spn: 0,
                twin_property: "engine.coolant_temp_c".to_string(),
                direction: MappingDirection::Read,
            }],
        }
    }

    #[tokio::test]
    async fn bridge_processes_single_frame() {
        let frames = vec![J1939Frame::new(65262, 0, [75, 0, 0, 0, 0, 0, 0, 0])];
        let source = MockJ1939Source::new(frames);
        let sink = MockDtdlSink::new();
        let cancel = CancellationToken::new();

        let mut bridge = J1939DtdlBridge::new(et1_config(), source, sink.clone(), cancel);
        bridge.run().await.expect("bridge should complete");

        let val = sink
            .get("engine.coolant_temp_c")
            .expect("should be written");
        assert_eq!(val, TwinValue::Integer(75));
    }

    #[tokio::test]
    async fn bridge_processes_multiple_frames() {
        let config = BridgeConfig {
            poll_interval_ms: 10,
            mappings: vec![
                RegisterMapping {
                    pgn: 65262,
                    spn: 0,
                    twin_property: "engine.coolant_temp_c".to_string(),
                    direction: MappingDirection::Read,
                },
                RegisterMapping {
                    pgn: 65265,
                    spn: 0,
                    twin_property: "vehicle.speed_kmh".to_string(),
                    direction: MappingDirection::Read,
                },
            ],
        };

        let frames = vec![
            J1939Frame::new(65262, 0, [90, 0, 0, 0, 0, 0, 0, 0]),
            J1939Frame::new(65265, 0, [50, 0, 0, 0, 0, 0, 0, 0]),
        ];
        let source = MockJ1939Source::new(frames);
        let sink = MockDtdlSink::new();
        let cancel = CancellationToken::new();

        let mut bridge = J1939DtdlBridge::new(config, source, sink.clone(), cancel);
        bridge.run().await.expect("bridge should complete");

        assert_eq!(
            sink.get("engine.coolant_temp_c"),
            Some(TwinValue::Integer(90))
        );
        assert_eq!(sink.get("vehicle.speed_kmh"), Some(TwinValue::Integer(50)));
    }

    #[tokio::test]
    async fn bridge_cancels_on_empty_source() {
        let source = MockJ1939Source::new(vec![]);
        let sink = MockDtdlSink::new();
        let cancel = CancellationToken::new();

        let mut bridge = J1939DtdlBridge::new(et1_config(), source, sink, cancel);
        bridge
            .run()
            .await
            .expect("empty source should be clean shutdown");
    }

    #[tokio::test]
    async fn bridge_respects_cancellation_token() {
        let source = MockJ1939Source::new(vec![]);
        let sink = MockDtdlSink::new();
        let cancel = CancellationToken::new();
        cancel.cancel(); // pre-cancelled

        let mut bridge = J1939DtdlBridge::new(et1_config(), source, sink, cancel);
        bridge
            .run()
            .await
            .expect("pre-cancelled bridge should complete cleanly");
    }

    #[tokio::test]
    async fn bridge_skips_na_indicator_0xfe() {
        let frames = vec![J1939Frame::new(65262, 0, [0xFE, 0, 0, 0, 0, 0, 0, 0])];
        let source = MockJ1939Source::new(frames);
        let sink = MockDtdlSink::new();
        let cancel = CancellationToken::new();

        let mut bridge = J1939DtdlBridge::new(et1_config(), source, sink.clone(), cancel);
        bridge.run().await.expect("bridge should complete");

        // No property should have been written.
        assert!(sink.is_empty());
    }

    #[tokio::test]
    async fn bridge_skips_na_indicator_0xff() {
        let frames = vec![J1939Frame::new(65262, 0, [0xFF, 0, 0, 0, 0, 0, 0, 0])];
        let source = MockJ1939Source::new(frames);
        let sink = MockDtdlSink::new();
        let cancel = CancellationToken::new();

        let mut bridge = J1939DtdlBridge::new(et1_config(), source, sink.clone(), cancel);
        bridge.run().await.expect("bridge should complete");

        assert!(sink.is_empty());
    }

    #[tokio::test]
    async fn bridge_ignores_unmapped_pgn() {
        // PGN 99999 has no entry in et1_config()
        let frames = vec![J1939Frame::new(99999, 0, [42, 0, 0, 0, 0, 0, 0, 0])];
        let source = MockJ1939Source::new(frames);
        let sink = MockDtdlSink::new();
        let cancel = CancellationToken::new();

        let mut bridge = J1939DtdlBridge::new(et1_config(), source, sink.clone(), cancel);
        bridge.run().await.expect("bridge should complete");

        assert!(sink.is_empty());
    }

    #[tokio::test]
    async fn bridge_writes_last_value_when_pgn_repeats() {
        let frames = vec![
            J1939Frame::new(65262, 0, [60, 0, 0, 0, 0, 0, 0, 0]),
            J1939Frame::new(65262, 0, [80, 0, 0, 0, 0, 0, 0, 0]),
        ];
        let source = MockJ1939Source::new(frames);
        let sink = MockDtdlSink::new();
        let cancel = CancellationToken::new();

        let mut bridge = J1939DtdlBridge::new(et1_config(), source, sink.clone(), cancel);
        bridge.run().await.expect("bridge should complete");

        // The second write should overwrite the first.
        assert_eq!(
            sink.get("engine.coolant_temp_c"),
            Some(TwinValue::Integer(80))
        );
    }
}
