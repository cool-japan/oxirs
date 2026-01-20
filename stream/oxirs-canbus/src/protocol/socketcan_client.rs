//! SocketCAN client for Linux CAN interface communication
//!
//! Provides async wrappers around the socketcan crate for CAN bus operations.
//! This module is only available on Linux systems.

use crate::config::CanbusConfig;
use crate::error::{CanbusError, CanbusResult};
use crate::protocol::frame::{CanFrame, CanId};
use socketcan::{
    CanAnyFrame, CanFdSocket, CanFilter as SocketCanFilter, CanFrame as SocketCanFrame, CanSocket,
    EmbeddedFrame, ExtendedId, Frame, Socket, SocketOptions, StandardId,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

/// SocketCAN client for Linux CAN bus communication
pub struct CanbusClient {
    /// Configuration
    config: CanbusConfig,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    /// Reader task handle
    reader_handle: Option<JoinHandle<()>>,
    /// Frame receiver channel
    frame_rx: Option<mpsc::Receiver<CanFrame>>,
    /// Frame sender channel (for transmitting)
    tx_socket: Option<CanSocket>,
}

impl CanbusClient {
    /// Create a new CAN bus client
    pub fn new(config: CanbusConfig) -> CanbusResult<Self> {
        Ok(Self {
            config,
            shutdown: Arc::new(AtomicBool::new(false)),
            reader_handle: None,
            frame_rx: None,
            tx_socket: None,
        })
    }

    /// Start the CAN bus client
    ///
    /// This opens the CAN interface and starts the background reader task.
    pub async fn start(&mut self) -> CanbusResult<()> {
        info!(interface = %self.config.interface, "Starting CANbus client");

        // Open the socket for reading
        let socket = Self::open_socket(&self.config.interface)?;

        // Apply filters if configured
        if !self.config.filters.is_empty() {
            let filters: Vec<SocketCanFilter> = self
                .config
                .filters
                .iter()
                .map(|f| SocketCanFilter::new(f.can_id, f.mask))
                .collect();
            socket
                .set_filters(&filters)
                .map_err(|e| CanbusError::Config(format!("Failed to set filters: {}", e)))?;
        }

        // Open a separate socket for transmission
        self.tx_socket = Some(Self::open_socket(&self.config.interface)?);

        // Create frame channel
        let (tx, rx) = mpsc::channel(1024);
        self.frame_rx = Some(rx);

        // Start background reader task
        let shutdown = self.shutdown.clone();
        let j1939_enabled = self.config.j1939_enabled;
        let interface = self.config.interface.clone();

        self.reader_handle = Some(tokio::task::spawn_blocking(move || {
            Self::read_loop(socket, tx, shutdown, j1939_enabled, interface);
        }));

        info!("CANbus client started successfully");
        Ok(())
    }

    /// Open a CAN socket
    fn open_socket(interface: &str) -> CanbusResult<CanSocket> {
        CanSocket::open(interface).map_err(|e| {
            if e.to_string().contains("No such device") {
                CanbusError::InterfaceNotFound(interface.to_string())
            } else {
                CanbusError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to open CAN socket: {}", e),
                ))
            }
        })
    }

    /// Background read loop
    fn read_loop(
        socket: CanSocket,
        tx: mpsc::Sender<CanFrame>,
        shutdown: Arc<AtomicBool>,
        j1939_enabled: bool,
        interface: String,
    ) {
        debug!(interface = %interface, "Starting CAN read loop");

        // Set socket timeout for periodic shutdown check
        if let Err(e) = socket.set_read_timeout(Duration::from_millis(100)) {
            error!("Failed to set socket timeout: {}", e);
            return;
        }

        while !shutdown.load(Ordering::SeqCst) {
            match socket.read_frame() {
                Ok(frame) => {
                    match Self::convert_socket_frame(&frame, j1939_enabled) {
                        Ok(can_frame) => {
                            if tx.blocking_send(can_frame).is_err() {
                                // Channel closed, exit loop
                                break;
                            }
                        }
                        Err(e) => {
                            warn!("Failed to convert CAN frame: {}", e);
                        }
                    }
                }
                Err(e) => {
                    // Check if it's a timeout (expected during shutdown checks)
                    let err_str = e.to_string();
                    if !err_str.contains("timed out")
                        && !err_str.contains("Resource temporarily unavailable")
                    {
                        error!("Error reading CAN frame: {}", e);
                    }
                }
            }
        }

        debug!(interface = %interface, "CAN read loop terminated");
    }

    /// Convert socketcan frame to our CanFrame type
    fn convert_socket_frame(
        frame: &SocketCanFrame,
        _j1939_enabled: bool,
    ) -> CanbusResult<CanFrame> {
        let id = if frame.is_extended() {
            CanId::extended(frame.raw_id())?
        } else {
            CanId::standard(frame.raw_id() as u16)?
        };

        let data = frame.data().to_vec();

        Ok(CanFrame {
            id,
            data,
            rtr: frame.is_remote_frame(),
            fd: false, // Regular CAN socket doesn't support FD
        })
    }

    /// Receive the next CAN frame
    pub async fn recv_frame(&mut self) -> Option<CanFrame> {
        self.frame_rx.as_mut()?.recv().await
    }

    /// Try to receive a CAN frame without blocking
    pub fn try_recv_frame(&mut self) -> Option<CanFrame> {
        self.frame_rx.as_mut()?.try_recv().ok()
    }

    /// Send a CAN frame
    pub fn send_frame(&self, frame: &CanFrame) -> CanbusResult<()> {
        let socket = self
            .tx_socket
            .as_ref()
            .ok_or_else(|| CanbusError::Config("Client not started".to_string()))?;

        let socket_frame = Self::convert_to_socket_frame(frame)?;
        socket
            .write_frame(&socket_frame)
            .map_err(|e| CanbusError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        Ok(())
    }

    /// Convert our CanFrame to socketcan frame
    fn convert_to_socket_frame(frame: &CanFrame) -> CanbusResult<SocketCanFrame> {
        let data = if frame.data.len() <= 8 {
            let mut arr = [0u8; 8];
            arr[..frame.data.len()].copy_from_slice(&frame.data);
            arr
        } else {
            return Err(CanbusError::FrameTooLarge(frame.data.len()));
        };

        match frame.id {
            CanId::Standard(id) => {
                let std_id =
                    StandardId::new(id).ok_or_else(|| CanbusError::InvalidCanId(id as u32))?;
                Ok(SocketCanFrame::new(std_id, &data[..frame.data.len()])
                    .ok_or_else(|| CanbusError::Config("Failed to create CAN frame".to_string()))?)
            }
            CanId::Extended(id) => {
                let ext_id = ExtendedId::new(id).ok_or_else(|| CanbusError::InvalidCanId(id))?;
                Ok(SocketCanFrame::new(ext_id, &data[..frame.data.len()])
                    .ok_or_else(|| CanbusError::Config("Failed to create CAN frame".to_string()))?)
            }
        }
    }

    /// Stop the CAN bus client
    pub async fn stop(&mut self) {
        info!("Stopping CANbus client");
        self.shutdown.store(true, Ordering::SeqCst);

        if let Some(handle) = self.reader_handle.take() {
            let _ = handle.await;
        }

        self.frame_rx = None;
        self.tx_socket = None;
        info!("CANbus client stopped");
    }

    /// Check if client is running
    pub fn is_running(&self) -> bool {
        self.reader_handle.is_some() && !self.shutdown.load(Ordering::SeqCst)
    }

    /// Get the configured interface name
    pub fn interface(&self) -> &str {
        &self.config.interface
    }

    /// Check if J1939 protocol handling is enabled
    pub fn j1939_enabled(&self) -> bool {
        self.config.j1939_enabled
    }
}

impl Drop for CanbusClient {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

/// CAN FD client for extended payload support (up to 64 bytes)
pub struct CanFdClient {
    /// Configuration
    config: CanbusConfig,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    /// Reader task handle
    reader_handle: Option<JoinHandle<()>>,
    /// Frame receiver channel
    frame_rx: Option<mpsc::Receiver<CanFrame>>,
}

impl CanFdClient {
    /// Create a new CAN FD client
    pub fn new(config: CanbusConfig) -> CanbusResult<Self> {
        Ok(Self {
            config,
            shutdown: Arc::new(AtomicBool::new(false)),
            reader_handle: None,
            frame_rx: None,
        })
    }

    /// Start the CAN FD client
    pub async fn start(&mut self) -> CanbusResult<()> {
        info!(interface = %self.config.interface, "Starting CAN FD client");

        let socket = CanFdSocket::open(&self.config.interface).map_err(|e| {
            if e.to_string().contains("No such device") {
                CanbusError::InterfaceNotFound(self.config.interface.clone())
            } else {
                CanbusError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to open CAN FD socket: {}", e),
                ))
            }
        })?;

        let (tx, rx) = mpsc::channel(1024);
        self.frame_rx = Some(rx);

        let shutdown = self.shutdown.clone();
        let interface = self.config.interface.clone();

        self.reader_handle = Some(tokio::task::spawn_blocking(move || {
            Self::read_loop(socket, tx, shutdown, interface);
        }));

        info!("CAN FD client started successfully");
        Ok(())
    }

    /// Background read loop for CAN FD
    fn read_loop(
        socket: CanFdSocket,
        tx: mpsc::Sender<CanFrame>,
        shutdown: Arc<AtomicBool>,
        interface: String,
    ) {
        debug!(interface = %interface, "Starting CAN FD read loop");

        if let Err(e) = socket.set_read_timeout(Duration::from_millis(100)) {
            error!("Failed to set socket timeout: {}", e);
            return;
        }

        while !shutdown.load(Ordering::SeqCst) {
            match socket.read_frame() {
                Ok(frame) => match Self::convert_any_frame(&frame) {
                    Ok(can_frame) => {
                        if tx.blocking_send(can_frame).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        warn!("Failed to convert CAN FD frame: {}", e);
                    }
                },
                Err(e) => {
                    let err_str = e.to_string();
                    if !err_str.contains("timed out")
                        && !err_str.contains("Resource temporarily unavailable")
                    {
                        error!("Error reading CAN FD frame: {}", e);
                    }
                }
            }
        }

        debug!(interface = %interface, "CAN FD read loop terminated");
    }

    /// Convert CanAnyFrame to our CanFrame type
    fn convert_any_frame(frame: &CanAnyFrame) -> CanbusResult<CanFrame> {
        match frame {
            CanAnyFrame::Normal(f) => {
                let id = if f.is_extended() {
                    CanId::extended(f.raw_id())?
                } else {
                    CanId::standard(f.raw_id() as u16)?
                };
                Ok(CanFrame {
                    id,
                    data: f.data().to_vec(),
                    rtr: f.is_remote_frame(),
                    fd: false,
                })
            }
            CanAnyFrame::Fd(f) => {
                let id = if f.is_extended() {
                    CanId::extended(f.raw_id())?
                } else {
                    CanId::standard(f.raw_id() as u16)?
                };
                Ok(CanFrame {
                    id,
                    data: f.data().to_vec(),
                    rtr: false, // CAN FD doesn't support RTR
                    fd: true,
                })
            }
            CanAnyFrame::Remote(f) => {
                let id = if f.is_extended() {
                    CanId::extended(f.raw_id())?
                } else {
                    CanId::standard(f.raw_id() as u16)?
                };
                Ok(CanFrame {
                    id,
                    data: Vec::new(), // Remote frames have no data
                    rtr: true,
                    fd: false,
                })
            }
            CanAnyFrame::Error(_) => Err(CanbusError::Config("Error frame received".to_string())),
        }
    }

    /// Receive the next CAN FD frame
    pub async fn recv_frame(&mut self) -> Option<CanFrame> {
        self.frame_rx.as_mut()?.recv().await
    }

    /// Stop the CAN FD client
    pub async fn stop(&mut self) {
        info!("Stopping CAN FD client");
        self.shutdown.store(true, Ordering::SeqCst);

        if let Some(handle) = self.reader_handle.take() {
            let _ = handle.await;
        }

        self.frame_rx = None;
        info!("CAN FD client stopped");
    }

    /// Check if client is running
    pub fn is_running(&self) -> bool {
        self.reader_handle.is_some() && !self.shutdown.load(Ordering::SeqCst)
    }
}

impl Drop for CanFdClient {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

/// Statistics for CAN bus communication
#[derive(Debug, Clone, Default)]
pub struct CanStatistics {
    /// Total frames received
    pub frames_received: u64,
    /// Total frames transmitted
    pub frames_transmitted: u64,
    /// Error frames received
    pub error_frames: u64,
    /// Receive overruns
    pub rx_overruns: u64,
    /// Transmit timeouts
    pub tx_timeouts: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CanFilter;

    #[test]
    fn test_can_filter_accept_all() {
        let filter = CanFilter::accept_all();
        assert_eq!(filter.can_id, 0);
        assert_eq!(filter.mask, 0);
    }

    #[test]
    fn test_can_filter_exact() {
        let filter = CanFilter::exact(0x7DF);
        assert_eq!(filter.can_id, 0x7DF);
        assert_eq!(filter.mask, 0x1FFFFFFF);
    }

    #[test]
    fn test_canbus_client_creation() {
        let config = CanbusConfig::default();
        let client = CanbusClient::new(config);
        assert!(client.is_ok());
    }

    #[test]
    fn test_can_statistics_default() {
        let stats = CanStatistics::default();
        assert_eq!(stats.frames_received, 0);
        assert_eq!(stats.frames_transmitted, 0);
        assert_eq!(stats.error_frames, 0);
    }

    #[test]
    fn test_convert_standard_frame() {
        let id = CanId::standard(0x123).unwrap();
        let frame = CanFrame::new(id, vec![0xDE, 0xAD, 0xBE, 0xEF]).unwrap();

        // Test conversion to socket frame
        let socket_frame = CanbusClient::convert_to_socket_frame(&frame);
        assert!(socket_frame.is_ok());
    }

    #[test]
    fn test_convert_extended_frame() {
        let id = CanId::extended(0x18FEF100).unwrap();
        let frame = CanFrame::new(id, vec![0x01, 0x02, 0x03, 0x04]).unwrap();

        // Test conversion to socket frame
        let socket_frame = CanbusClient::convert_to_socket_frame(&frame);
        assert!(socket_frame.is_ok());
    }

    #[test]
    fn test_frame_too_large_for_standard() {
        let id = CanId::standard(0x123).unwrap();
        let frame = CanFrame {
            id,
            data: vec![0; 10], // 10 bytes, too large for standard CAN
            rtr: false,
            fd: false,
        };

        let result = CanbusClient::convert_to_socket_frame(&frame);
        assert!(result.is_err());
    }
}
