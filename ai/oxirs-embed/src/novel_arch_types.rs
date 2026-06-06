//! Configuration and state types for novel embedding architectures
//!
//! Contains every config struct, enum, parameter container and runtime state
//! shared by the [`NovelArchitectureModel`](crate::novel_arch_impl::NovelArchitectureModel)
//! implementation: graph transformers, neural ODEs, hyperbolic embeddings,
//! geometric deep learning, quantum-inspired layers and continuous flows.

use crate::{ModelConfig, TrainingStats};
use scirs2_core::ndarray_ext::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Configuration for novel architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NovelArchitectureConfig {
    pub base_config: ModelConfig,
    /// Architecture type
    pub architecture: ArchitectureType,
    /// Specialized parameters per architecture
    pub architecture_params: ArchitectureParams,
    /// Training dynamics configuration
    pub dynamics_config: DynamicsConfig,
    /// Geometric learning settings
    pub geometric_config: GeometricConfig,
}

impl Default for NovelArchitectureConfig {
    fn default() -> Self {
        Self {
            base_config: ModelConfig::default(),
            architecture: ArchitectureType::GraphTransformer,
            architecture_params: ArchitectureParams::default(),
            dynamics_config: DynamicsConfig::default(),
            geometric_config: GeometricConfig::default(),
        }
    }
}

/// Types of novel architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchitectureType {
    /// Graph Transformer with structural attention
    GraphTransformer,
    /// Neural ODE for continuous dynamics
    NeuralODE,
    /// Hyperbolic embeddings for hierarchical structures
    HyperbolicEmbedding,
    /// Geometric deep learning on manifolds
    GeometricDeepLearning,
    /// Quantum-inspired embedding methods
    QuantumInspired,
    /// Continuous normalizing flows
    ContinuousNormalizingFlow,
}

/// Architecture-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ArchitectureParams {
    /// Graph Transformer parameters
    pub transformer_params: GraphTransformerParams,
    /// Neural ODE parameters
    pub ode_params: NeuralODEParams,
    /// Hyperbolic parameters
    pub hyperbolic_params: HyperbolicParams,
    /// Geometric parameters
    pub geometric_params: GeometricParams,
    /// Quantum parameters
    pub quantum_params: QuantumParams,
}

/// Graph Transformer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTransformerParams {
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Attention dimension
    pub attention_dim: usize,
    /// Feed-forward dimension
    pub ff_dim: usize,
    /// Structural encoding dimension
    pub structural_dim: usize,
    /// Use positional encoding
    pub use_positional_encoding: bool,
    /// Attention mechanism
    pub attention_mechanism: AttentionMechanism,
    /// Structural bias type
    pub structural_bias: StructuralBias,
}

impl Default for GraphTransformerParams {
    fn default() -> Self {
        Self {
            num_heads: 8,
            num_layers: 6,
            attention_dim: 512,
            ff_dim: 2048,
            structural_dim: 128,
            use_positional_encoding: true,
            attention_mechanism: AttentionMechanism::SparseAttention,
            structural_bias: StructuralBias::SpectralFeatures,
        }
    }
}

/// Attention mechanisms for Graph Transformers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionMechanism {
    /// Standard multi-head attention
    MultiHeadAttention,
    /// Sparse attention for large graphs
    SparseAttention,
    /// Linear attention for efficiency
    LinearAttention,
    /// Performer-style attention
    PerformerAttention,
    /// Graph-aware attention
    GraphAwareAttention,
}

/// Structural bias types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StructuralBias {
    /// Spectral features from graph Laplacian
    SpectralFeatures,
    /// Shortest path distances
    ShortestPath,
    /// Random walk features
    RandomWalk,
    /// Centrality measures
    CentralityMeasures,
    /// Graph motif features
    GraphMotifs,
}

/// Neural ODE configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralODEParams {
    /// ODE solver type
    pub solver_type: ODESolverType,
    /// Integration time steps
    pub time_steps: usize,
    /// Tolerance for adaptive solvers
    pub tolerance: f64,
    /// Hidden dimensions for ODE function
    pub hidden_dims: Vec<usize>,
    /// Activation function
    pub activation: ActivationType,
    /// Adjoint method for backprop
    pub use_adjoint: bool,
    /// Regularization type
    pub regularization: ODERegularization,
}

impl Default for NeuralODEParams {
    fn default() -> Self {
        Self {
            solver_type: ODESolverType::DormandPrince,
            time_steps: 100,
            tolerance: 1e-6,
            hidden_dims: vec![512, 256, 128],
            activation: ActivationType::Swish,
            use_adjoint: true,
            regularization: ODERegularization::None,
        }
    }
}

/// ODE solver types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ODESolverType {
    /// Euler method
    Euler,
    /// Runge-Kutta 4th order
    RungeKutta4,
    /// Dormand-Prince adaptive method
    DormandPrince,
    /// Adams-Bashforth
    AdamsBashforth,
    /// Implicit methods
    BackwardEuler,
}

/// ODE regularization techniques
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ODERegularization {
    None,
    /// Kinetic energy regularization
    KineticEnergy,
    /// Jacobian regularization
    JacobianFrobenius,
    /// Spectral normalization
    SpectralNormalization,
}

/// Activation types for neural networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    Swish,
    Mish,
    GELU,
    ELU,
    LeakyReLU,
    Tanh,
}

/// Hyperbolic embedding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperbolicParams {
    /// Hyperbolic manifold type
    pub manifold: HyperbolicManifold,
    /// Curvature parameter
    pub curvature: f64,
    /// Manifold dimension
    pub manifold_dim: usize,
    /// Optimization method on manifold
    pub optimizer: ManifoldOptimizer,
    /// Distance function
    pub distance_function: HyperbolicDistance,
    /// Initialization strategy
    pub initialization: HyperbolicInit,
}

impl Default for HyperbolicParams {
    fn default() -> Self {
        Self {
            manifold: HyperbolicManifold::Poincare,
            curvature: -1.0,
            manifold_dim: 128,
            optimizer: ManifoldOptimizer::RiemannianAdam,
            distance_function: HyperbolicDistance::Poincare,
            initialization: HyperbolicInit::RandomNormal,
        }
    }
}

/// Hyperbolic manifold types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HyperbolicManifold {
    /// Poincaré ball model
    Poincare,
    /// Klein model
    Klein,
    /// Hyperboloid model
    Hyperboloid,
    /// Upper half-space model
    UpperHalfSpace,
}

/// Manifold optimizers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ManifoldOptimizer {
    /// Riemannian SGD
    RiemannianSGD,
    /// Riemannian Adam
    RiemannianAdam,
    /// Riemannian AdaGrad
    RiemannianAdaGrad,
    /// Exponential map based
    ExponentialMap,
}

/// Hyperbolic distance functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HyperbolicDistance {
    /// Poincaré distance
    Poincare,
    /// Hyperbolic distance in hyperboloid model
    Hyperboloid,
    /// Geodesic distance
    Geodesic,
}

/// Hyperbolic initialization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HyperbolicInit {
    /// Random normal initialization
    RandomNormal,
    /// Wrapped normal distribution
    WrappedNormal,
    /// Uniform on hyperbolic space
    UniformHyperbolic,
    /// Tree-based initialization
    TreeBased,
}

/// Geometric deep learning parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricParams {
    /// Geometric space type
    pub space_type: GeometricSpace,
    /// Equivariance groups
    pub equivariance_groups: Vec<EquivarianceGroup>,
    /// Gauge equivariant layers
    pub use_gauge_equivariance: bool,
    /// Fiber bundle dimension
    pub fiber_dim: usize,
    /// Connection learning
    pub learn_connection: bool,
    /// Curvature regularization
    pub curvature_regularization: f64,
}

impl Default for GeometricParams {
    fn default() -> Self {
        Self {
            space_type: GeometricSpace::RiemannianManifold,
            equivariance_groups: vec![EquivarianceGroup::SO3, EquivarianceGroup::SE3],
            use_gauge_equivariance: true,
            fiber_dim: 64,
            learn_connection: true,
            curvature_regularization: 0.01,
        }
    }
}

/// Geometric space types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeometricSpace {
    /// Riemannian manifolds
    RiemannianManifold,
    /// Lie groups
    LieGroup,
    /// Fiber bundles
    FiberBundle,
    /// Homogeneous spaces
    HomogeneousSpace,
    /// Simplicial complexes
    SimplicialComplex,
}

/// Equivariance groups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EquivarianceGroup {
    /// Special orthogonal group SO(3)
    SO3,
    /// Special Euclidean group SE(3)
    SE3,
    /// General linear group GL(n)
    GLn,
    /// Symmetric group
    SymmetricGroup,
    /// Lorentz group
    LorentzGroup,
}

/// Quantum-inspired parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumParams {
    /// Number of qubits for quantum state
    pub num_qubits: usize,
    /// Quantum gate set
    pub gate_set: QuantumGateSet,
    /// Entanglement structure
    pub entanglement: EntanglementStructure,
    /// Measurement strategy
    pub measurement: QuantumMeasurement,
    /// Quantum noise model
    pub noise_model: QuantumNoise,
    /// Classical-quantum interface
    pub hybrid_layers: bool,
}

impl Default for QuantumParams {
    fn default() -> Self {
        Self {
            num_qubits: 10,
            gate_set: QuantumGateSet::Universal,
            entanglement: EntanglementStructure::Linear,
            measurement: QuantumMeasurement::Computational,
            noise_model: QuantumNoise::None,
            hybrid_layers: true,
        }
    }
}

/// Quantum gate sets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumGateSet {
    /// Universal gate set
    Universal,
    /// Clifford gates
    Clifford,
    /// Variational gates
    Variational,
    /// Adiabatic evolution
    Adiabatic,
}

/// Entanglement structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntanglementStructure {
    /// Linear entanglement
    Linear,
    /// All-to-all entanglement
    AllToAll,
    /// Tree entanglement
    Tree,
    /// Hardware-efficient
    HardwareEfficient,
}

/// Quantum measurement strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumMeasurement {
    /// Computational basis
    Computational,
    /// Pauli measurements
    Pauli,
    /// Quantum state tomography
    Tomography,
    /// Shadow measurements
    Shadow,
}

/// Quantum noise models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumNoise {
    None,
    /// Depolarizing noise
    Depolarizing,
    /// Amplitude damping
    AmplitudeDamping,
    /// Phase damping
    PhaseDamping,
    /// Realistic device noise
    DeviceNoise,
}

/// Dynamics configuration for continuous models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsConfig {
    /// Time evolution parameters
    pub time_evolution: TimeEvolution,
    /// Continuous flow type
    pub flow_type: FlowType,
    /// Integration scheme
    pub integration_scheme: IntegrationScheme,
    /// Stability constraints
    pub stability_constraints: StabilityConstraints,
}

impl Default for DynamicsConfig {
    fn default() -> Self {
        Self {
            time_evolution: TimeEvolution::default(),
            flow_type: FlowType::NormalizingFlow,
            integration_scheme: IntegrationScheme::AdaptiveRungeKutta,
            stability_constraints: StabilityConstraints::default(),
        }
    }
}

/// Time evolution parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeEvolution {
    /// Start time
    pub t_start: f64,
    /// End time
    pub t_end: f64,
    /// Time steps
    pub time_steps: usize,
    /// Adaptive time stepping
    pub adaptive: bool,
}

impl Default for TimeEvolution {
    fn default() -> Self {
        Self {
            t_start: 0.0,
            t_end: 1.0,
            time_steps: 100,
            adaptive: true,
        }
    }
}

/// Flow types for continuous models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlowType {
    /// Normalizing flows
    NormalizingFlow,
    /// Continuous normalizing flows
    ContinuousNormalizingFlow,
    /// Neural flows
    NeuralFlow,
    /// Hamiltonian flows
    HamiltonianFlow,
}

/// Integration schemes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrationScheme {
    /// Fixed-step Runge-Kutta
    FixedRungeKutta,
    /// Adaptive Runge-Kutta
    AdaptiveRungeKutta,
    /// Symplectic integrators
    SymplecticIntegrator,
    /// Implicit methods
    ImplicitMethods,
}

/// Stability constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityConstraints {
    /// Maximum eigenvalue
    pub max_eigenvalue: f64,
    /// Lyapunov regularization
    pub lyapunov_reg: f64,
    /// Spectral normalization
    pub spectral_norm: bool,
}

impl Default for StabilityConstraints {
    fn default() -> Self {
        Self {
            max_eigenvalue: 1.0,
            lyapunov_reg: 0.01,
            spectral_norm: true,
        }
    }
}

/// Geometric configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeometricConfig {
    /// Manifold learning parameters
    pub manifold_learning: ManifoldLearning,
    /// Curvature computation
    pub curvature_computation: CurvatureComputation,
    /// Parallel transport
    pub parallel_transport: ParallelTransport,
}

/// Manifold learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifoldLearning {
    /// Intrinsic dimension
    pub intrinsic_dim: usize,
    /// Neighborhood size
    pub neighborhood_size: usize,
    /// Embedding method
    pub embedding_method: ManifoldMethod,
}

impl Default for ManifoldLearning {
    fn default() -> Self {
        Self {
            intrinsic_dim: 64,
            neighborhood_size: 10,
            embedding_method: ManifoldMethod::Isomap,
        }
    }
}

/// Manifold embedding methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ManifoldMethod {
    /// Isomap
    Isomap,
    /// Locally Linear Embedding
    LLE,
    /// Laplacian Eigenmaps
    LaplacianEigenmaps,
    /// Diffusion Maps
    DiffusionMaps,
    /// t-SNE
    TSNE,
    /// UMAP
    UMAP,
}

/// Curvature computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureComputation {
    /// Curvature type
    pub curvature_type: CurvatureType,
    /// Computation method
    pub computation_method: CurvatureMethod,
    /// Regularization
    pub regularization: f64,
}

impl Default for CurvatureComputation {
    fn default() -> Self {
        Self {
            curvature_type: CurvatureType::Ricci,
            computation_method: CurvatureMethod::FormanRicci,
            regularization: 0.01,
        }
    }
}

/// Curvature types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CurvatureType {
    /// Gaussian curvature
    Gaussian,
    /// Mean curvature
    Mean,
    /// Ricci curvature
    Ricci,
    /// Scalar curvature
    Scalar,
    /// Sectional curvature
    Sectional,
}

/// Curvature computation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CurvatureMethod {
    /// Forman-Ricci curvature
    FormanRicci,
    /// Ollivier-Ricci curvature
    OllivierRicci,
    /// Discrete Gaussian curvature
    DiscreteGaussian,
    /// Graph-based methods
    GraphBased,
}

/// Parallel transport configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelTransport {
    /// Transport method
    pub method: TransportMethod,
    /// Path discretization
    pub path_steps: usize,
    /// Tolerance
    pub tolerance: f64,
}

impl Default for ParallelTransport {
    fn default() -> Self {
        Self {
            method: TransportMethod::SchildLadder,
            path_steps: 50,
            tolerance: 1e-6,
        }
    }
}

/// Parallel transport methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransportMethod {
    /// Schild's ladder
    SchildLadder,
    /// Pole ladder
    PoleLadder,
    /// Geodesic parallel transport
    GeodesicTransport,
    /// Discrete transport
    DiscreteTransport,
}

/// Novel architecture embedding model
#[derive(Debug, Clone)]
pub struct NovelArchitectureModel {
    pub config: NovelArchitectureConfig,
    pub model_id: Uuid,
    pub entities: HashMap<String, usize>,
    pub relations: HashMap<String, usize>,
    pub entity_embeddings: Array2<f64>,
    pub relation_embeddings: Array2<f64>,
    pub architecture_state: ArchitectureState,
    pub training_stats: Option<TrainingStats>,
    pub is_trained: bool,
}

/// Architecture-specific state
#[derive(Debug, Clone)]
pub struct ArchitectureState {
    /// Graph transformer state
    pub transformer_state: Option<GraphTransformerState>,
    /// Neural ODE state
    pub ode_state: Option<NeuralODEState>,
    /// Hyperbolic state
    pub hyperbolic_state: Option<HyperbolicState>,
    /// Geometric state
    pub geometric_state: Option<GeometricState>,
    /// Quantum state
    pub quantum_state: Option<QuantumState>,
}

/// Graph transformer state
#[derive(Debug, Clone)]
pub struct GraphTransformerState {
    /// Attention weights
    pub attention_weights: Array3<f64>,
    /// Layer outputs
    pub layer_outputs: Vec<Array2<f64>>,
    /// Structural features
    pub structural_features: Array2<f64>,
    /// Position encodings
    pub position_encodings: Option<Array2<f64>>,
}

/// Neural ODE state
#[derive(Debug, Clone)]
pub struct NeuralODEState {
    /// Current time
    pub current_time: f64,
    /// State trajectory
    pub trajectory: Vec<Array2<f64>>,
    /// ODE function parameters
    pub ode_params: Array2<f64>,
    /// Integration statistics
    pub integration_stats: IntegrationStats,
}

/// Integration statistics
#[derive(Debug, Clone)]
pub struct IntegrationStats {
    pub steps_taken: usize,
    pub function_evaluations: usize,
    pub jacobian_evaluations: usize,
    pub failed_steps: usize,
    pub final_error: f64,
}

/// Hyperbolic state
#[derive(Debug, Clone)]
pub struct HyperbolicState {
    /// Manifold embeddings
    pub manifold_embeddings: Array2<f64>,
    /// Curvature parameter
    pub curvature: f64,
    /// Tangent vectors
    pub tangent_vectors: Array2<f64>,
    /// Metric tensor
    pub metric_tensor: Array3<f64>,
}

/// Geometric state
#[derive(Debug, Clone)]
pub struct GeometricState {
    /// Connection coefficients
    pub connection: Array3<f64>,
    /// Curvature tensor
    pub curvature_tensor: Array3<f64>,
    /// Parallel transport maps
    pub transport_maps: HashMap<String, Array2<f64>>,
    /// Equivariance maps
    pub equivariance_maps: Vec<Array2<f64>>,
}

/// Quantum state
#[derive(Debug, Clone)]
pub struct QuantumState {
    /// Quantum state vector
    pub state_vector: Array1<f64>,
    /// Quantum gates
    pub gates: Vec<Array2<f64>>,
    /// Measurement outcomes
    pub measurements: Vec<f64>,
    /// Entanglement measures
    pub entanglement: f64,
}
