//! Inference sub-modules for SHACL-AI

pub mod quantized;
pub mod realtime;

pub use quantized::{
    quantize_model, Activation, BatchedInferenceConfig, BatchedInferenceEngine,
    CalibrationCollector, FusedLinearKernel, InferenceEngineStats, InferenceRequest,
    QuantizationConfig, QuantizationParams, QuantizationSummary, QuantizedInferenceResult,
    QuantizedWeightMatrix,
};
pub use realtime::{
    InferencePipelineConfig, InferenceResult, PipelineStats, PredictedViolation,
    RealTimeInferencePipeline,
};
