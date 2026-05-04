//! Modbus ↔ OPC UA bidirectional bridge.
//!
//! [`OpcuaModbusBridge`] runs as a tokio task. It:
//!
//! 1. Polls Modbus registers at `poll_interval` and publishes coerced values to
//!    the OPC UA address space.
//! 2. Subscribes to OPC UA client writes and forwards them to Modbus.
//! 3. Shuts down cleanly when the [`CancellationToken`] is cancelled.

use std::time::Duration;

use thiserror::Error;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use crate::opcua::client::{ModbusClientError, ModbusClientFacade};
use crate::opcua::config::BridgeConfig;
use crate::opcua::mapper::RegisterMapper;
use crate::opcua::server::{OpcuaError, OpcuaServerFacade};
use crate::opcua::type_coercion::{
    registers_to_value, value_to_registers, CoercionError, DataValue,
};

/// Errors that can occur while running the bridge.
#[derive(Debug, Error)]
pub enum BridgeError {
    /// The OPC UA server layer reported an error.
    #[error("OPC UA server error: {0}")]
    OpcuaServer(#[from] OpcuaError),

    /// The Modbus client layer reported an error.
    #[error("Modbus client error: {0}")]
    ModbusClient(#[from] ModbusClientError),

    /// A type coercion failed.
    #[error("coercion error for register {register}: {source}")]
    Coercion {
        register: u16,
        #[source]
        source: CoercionError,
    },

    /// The write-subscription channel was closed unexpectedly.
    #[error("OPC UA write channel closed unexpectedly")]
    WriteChannelClosed,
}

/// Bidirectional Modbus ↔ OPC UA bridge.
///
/// Generic over `S` (OPC UA server facade) and `C` (Modbus client facade) so
/// that tests can inject mocks without opening real sockets.
pub struct OpcuaModbusBridge<S: OpcuaServerFacade, C: ModbusClientFacade> {
    /// Maps between Modbus register addresses and OPC UA node IDs.
    pub mapper: RegisterMapper,
    /// OPC UA server facade.
    pub opcua_server: S,
    /// Modbus client facade.
    pub modbus_client: C,
    /// Cancellation token; signal this to shut down the bridge.
    pub cancel: CancellationToken,
    /// How long to wait between Modbus polling cycles.
    pub poll_interval: Duration,
}

impl<S: OpcuaServerFacade, C: ModbusClientFacade> OpcuaModbusBridge<S, C> {
    /// Create a new bridge from a [`BridgeConfig`].
    pub fn new(
        config: BridgeConfig,
        opcua_server: S,
        modbus_client: C,
        cancel: CancellationToken,
    ) -> Self {
        let poll_interval = Duration::from_millis(config.poll_interval_ms);
        let mapper = RegisterMapper::new(config);
        Self {
            mapper,
            opcua_server,
            modbus_client,
            cancel,
            poll_interval,
        }
    }

    /// Run the bridge until the cancellation token is signalled.
    ///
    /// # Errors
    /// Returns [`BridgeError`] if a non-recoverable error occurs.  Transient
    /// per-register errors are logged and skipped rather than propagated.
    pub async fn run(mut self) -> Result<(), BridgeError> {
        // Obtain the write-event receiver before entering the select! loop so
        // there are no borrow-conflicts on `self`.
        let mut write_rx = self.opcua_server.subscribe_writes().await?;

        let cancel = self.cancel.clone();

        tokio::select! {
            _ = cancel.cancelled() => {
                debug!("bridge: cancellation token fired, shutting down");
                Ok(())
            }
            result = Self::run_loops(
                &self.mapper,
                &self.opcua_server,
                &self.modbus_client,
                &mut write_rx,
                self.poll_interval,
                self.cancel.clone(),
            ) => result,
        }
    }

    /// Inner function that owns the poll + write loops.
    ///
    /// Extracted from `run` so the `tokio::select!` above stays clean.
    async fn run_loops(
        mapper: &RegisterMapper,
        opcua_server: &S,
        modbus_client: &C,
        write_rx: &mut mpsc::Receiver<(String, DataValue)>,
        poll_interval: Duration,
        cancel: CancellationToken,
    ) -> Result<(), BridgeError> {
        let mut interval = tokio::time::interval(poll_interval);
        // The first tick fires immediately; skip it to avoid a redundant poll
        // right after startup.
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    debug!("bridge inner loop: cancellation token fired");
                    return Ok(());
                }

                _ = interval.tick() => {
                    Self::poll_modbus_to_opcua(mapper, opcua_server, modbus_client).await;
                }

                maybe_write = write_rx.recv() => {
                    match maybe_write {
                        Some((node_id, value)) => {
                            if let Err(e) = Self::forward_write_to_modbus(
                                mapper, modbus_client, &node_id, value,
                            ).await {
                                warn!("bridge: forward write for {} failed: {}", node_id, e);
                                // Non-fatal: log and continue.
                            }
                        }
                        None => {
                            // Channel closed — the OPC UA server shut down.
                            return Err(BridgeError::WriteChannelClosed);
                        }
                    }
                }
            }
        }
    }

    /// Poll all readable Modbus registers and publish results to OPC UA.
    ///
    /// Per-register errors are logged and skipped; this function never returns
    /// an error so that one bad register does not kill the whole bridge.
    async fn poll_modbus_to_opcua(mapper: &RegisterMapper, opcua_server: &S, modbus_client: &C) {
        for mapping in mapper.all_readable() {
            let count = mapping.data_type.register_count() as u16;

            let regs = match modbus_client
                .read_holding_registers(mapping.modbus_register, count)
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    warn!(
                        "bridge: read register {} failed: {}",
                        mapping.modbus_register, e
                    );
                    continue;
                }
            };

            let value = match registers_to_value(&regs, &mapping.data_type) {
                Ok(v) => v,
                Err(e) => {
                    warn!(
                        "bridge: coercion for register {} failed: {}",
                        mapping.modbus_register, e
                    );
                    continue;
                }
            };

            if let Err(e) = opcua_server
                .publish_value(&mapping.opcua_node_id, &value)
                .await
            {
                warn!("bridge: publish to {} failed: {}", mapping.opcua_node_id, e);
            } else {
                debug!(
                    "bridge: register {} → {} = {:?}",
                    mapping.modbus_register, mapping.opcua_node_id, value
                );
            }
        }
    }

    /// Forward an OPC UA write event to the corresponding Modbus register.
    async fn forward_write_to_modbus(
        mapper: &RegisterMapper,
        modbus_client: &C,
        node_id: &str,
        value: DataValue,
    ) -> Result<(), BridgeError> {
        let mapping = match mapper.find_mapping_by_node(node_id) {
            Some(m) => m,
            None => {
                warn!("bridge: no mapping found for OPC UA node {}", node_id);
                return Ok(());
            }
        };

        // Only forward if the mapping allows writes to Modbus.
        if !mapping.direction.is_writable() {
            debug!(
                "bridge: ignoring write to {} (direction {:?})",
                node_id, mapping.direction
            );
            return Ok(());
        }

        let registers =
            value_to_registers(&value, &mapping.data_type).map_err(|e| BridgeError::Coercion {
                register: mapping.modbus_register,
                source: e,
            })?;

        modbus_client
            .write_registers(mapping.modbus_register, &registers)
            .await
            .map_err(BridgeError::ModbusClient)?;

        debug!(
            "bridge: OPC UA {} → register {} = {:?}",
            node_id, mapping.modbus_register, registers
        );
        Ok(())
    }

    /// Execute a single poll iteration (reads + publishes) without running the
    /// full event loop.  Useful for unit tests that want deterministic control.
    pub async fn poll_once(&self) {
        Self::poll_modbus_to_opcua(&self.mapper, &self.opcua_server, &self.modbus_client).await;
    }

    /// Forward a single OPC UA write event without running the full event loop.
    /// Useful for unit tests.
    pub async fn forward_write_once(
        &self,
        node_id: &str,
        value: DataValue,
    ) -> Result<(), BridgeError> {
        Self::forward_write_to_modbus(&self.mapper, &self.modbus_client, node_id, value).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opcua::client::MockModbusClient;
    use crate::opcua::config::{BridgeConfig, DataTypeSpec, Direction, RegisterMapping};
    use crate::opcua::server::MockOpcuaServer;

    fn sample_config() -> BridgeConfig {
        BridgeConfig {
            poll_interval_ms: 100,
            modbus_host: "127.0.0.1".to_owned(),
            modbus_port: 502,
            opcua_endpoint: None,
            mappings: vec![RegisterMapping {
                modbus_register: 100,
                opcua_node_id: "ns=2;s=Speed".to_owned(),
                data_type: DataTypeSpec::U16,
                direction: Direction::Read,
            }],
        }
    }

    #[tokio::test]
    async fn bridge_poll_publishes_value() {
        let config = sample_config();
        let opcua = MockOpcuaServer::new();
        let modbus = MockModbusClient::new();
        modbus.set_register(100, vec![42u16]);

        // We need the subscriber to exist before subscribe_writes is called.
        let cancel = CancellationToken::new();

        // Use poll_once directly for a deterministic test without the full loop.
        let bridge = OpcuaModbusBridge::new(config, opcua, modbus, cancel);
        bridge.poll_once().await;

        let published = bridge.opcua_server.get_published();
        assert_eq!(published.len(), 1);
        assert_eq!(published[0].0, "ns=2;s=Speed");
        assert_eq!(published[0].1, DataValue::U16(42));
    }
}
