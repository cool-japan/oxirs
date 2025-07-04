//! Type definitions for photonic computing

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Wavelength range for optical processing
#[derive(Debug, Clone)]
pub struct WavelengthRange {
    /// Minimum wavelength (nm)
    pub min_wavelength: f64,
    /// Maximum wavelength (nm)  
    pub max_wavelength: f64,
}

/// Photonic gate for quantum operations
#[derive(Debug, Clone)]
pub struct PhotonicGate {
    /// Gate identifier
    pub id: String,
    /// Gate type
    pub gate_type: GateType,
    /// Transmission coefficient
    pub transmission: f64,
    /// Reflection coefficient
    pub reflection: f64,
}

/// Types of photonic gates
#[derive(Debug, Clone)]
pub enum GateType {
    BeamSplitter,
    PhaseMod,
    Rotation,
    Entangler,
    Detector,
}

/// Optical processing state
#[derive(Debug, Clone)]
pub enum OpticalProcessingState {
    Idle,
    Processing,
    Entangled,
    Measuring,
    Error,
}

/// Connection type between optical components
#[derive(Debug, Clone)]
pub enum ConnectionType {
    /// Free space propagation
    FreeSpace,
    /// Fiber optic connection
    Fiber,
    /// Waveguide connection
    Waveguide,
    /// Evanescent coupling
    Evanescent,
}

/// Photonic qubit representation
#[derive(Debug, Clone)]
pub struct PhotonicQubit {
    /// Qubit identifier
    pub id: String,
    /// Polarization state
    pub polarization: PolarizationState,
    /// Photon number state
    pub photon_number: PhotonNumberState,
    /// Frequency/wavelength
    pub frequency: f64,
    /// Spatial mode
    pub spatial_mode: SpatialMode,
    /// Coherence properties
    pub coherence: CoherenceProperties,
}

/// Polarization state of photon
#[derive(Debug, Clone)]
pub enum PolarizationState {
    /// Horizontal polarization
    Horizontal,
    /// Vertical polarization
    Vertical,
    /// Diagonal polarization
    Diagonal,
    /// Anti-diagonal polarization
    AntiDiagonal,
    /// Left circular polarization
    LeftCircular,
    /// Right circular polarization
    RightCircular,
    /// Superposition of polarizations
    Superposition { amplitudes: Vec<f64>, phases: Vec<f64> },
}

/// Photon number state
#[derive(Debug, Clone)]
pub struct PhotonNumberState {
    /// Number of photons
    pub number: u32,
    /// Amplitude
    pub amplitude: f64,
    /// Phase
    pub phase: f64,
}

/// Spatial mode of photon
#[derive(Debug, Clone)]
pub struct SpatialMode {
    /// Mode identifier
    pub id: String,
    /// Transverse mode number
    pub transverse_mode: (u32, u32),
    /// Beam waist
    pub beam_waist: f64,
}

/// Coherence properties
#[derive(Debug, Clone)]
pub struct CoherenceProperties {
    /// Coherence length
    pub coherence_length: f64,
    /// Coherence time  
    pub coherence_time: f64,
    /// Spatial coherence
    pub spatial_coherence: f64,
    /// Temporal coherence
    pub temporal_coherence: f64,
}

/// Material properties for photonic components
#[derive(Debug, Clone)]
pub enum MaterialType {
    /// Silicon
    Silicon,
    /// Silicon dioxide
    SiliconDioxide,
    /// Gallium arsenide
    GalliumArsenide,
    /// Lithium niobate
    LithiumNiobate,
    /// Custom material
    Custom { name: String },
}

/// Nonlinear optical properties
#[derive(Debug, Clone)]
pub struct NonlinearProperties {
    /// Second-order susceptibility
    pub chi2: f64,
    /// Third-order susceptibility
    pub chi3: f64,
    /// Kerr coefficient
    pub kerr_coefficient: f64,
    /// Two-photon absorption
    pub two_photon_absorption: f64,
}

/// Type of optical junction
#[derive(Debug, Clone)]
pub enum JunctionType {
    /// Simple beam splitter
    BeamSplitter,
    /// Directional coupler
    DirectionalCoupler,
    /// Multi-mode interferometer
    MultiModeInterferometer,
    /// Wavelength division multiplexer
    WavelengthMux,
    /// Optical switch
    OpticalSwitch,
}

/// Network topology
#[derive(Debug, Clone)]
pub enum NetworkTopology {
    Star,
    Ring,
    Mesh,
    Tree,
    Custom,
}