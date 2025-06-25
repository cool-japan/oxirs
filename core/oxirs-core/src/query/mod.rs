//! SPARQL query processing module

pub mod algebra;
pub mod parser;
pub mod plan;
pub mod exec;
pub mod optimizer;
pub mod gpu;
pub mod jit;
pub mod distributed;
pub mod wasm;
pub mod functions;
pub mod property_paths;

pub use algebra::*;
pub use parser::*;
pub use optimizer::{AIQueryOptimizer, MultiQueryOptimizer};
pub use gpu::{GpuQueryExecutor, GpuBackend};
pub use jit::{JitCompiler, JitConfig};
pub use distributed::{DistributedQueryEngine, FederatedEndpoint, DistributedConfig};
pub use wasm::{WasmQueryCompiler, WasmTarget, OptimizationLevel};