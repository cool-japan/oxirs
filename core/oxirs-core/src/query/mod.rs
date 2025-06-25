//! SPARQL query processing module

pub mod algebra;
pub mod distributed;
pub mod exec;
pub mod functions;
pub mod gpu;
pub mod jit;
pub mod optimizer;
pub mod parser;
pub mod plan;
pub mod property_paths;
pub mod wasm;

pub use algebra::*;
pub use distributed::{DistributedConfig, DistributedQueryEngine, FederatedEndpoint};
pub use gpu::{GpuBackend, GpuQueryExecutor};
pub use jit::{JitCompiler, JitConfig};
pub use optimizer::{AIQueryOptimizer, MultiQueryOptimizer};
pub use parser::*;
pub use wasm::{OptimizationLevel, WasmQueryCompiler, WasmTarget};
