//! Hybrid GNN+LLM architecture — phases b, c, and d.
//!
//! This module provides:
//! - [`LlmProvider`] — async trait for LLM completion back-ends.
//! - [`LocalProvider`] — deterministic canned-response mock for offline tests.
//! - [`SoftPromptProjector`] — learnable linear projection from GNN embedding space
//!   to the prompt-token dimension, trained while the GNN encoder is frozen.
//! - [`HybridLlmHead`] — end-to-end KGQA head that chains the frozen encoder,
//!   the projector, and an LLM provider.
//! - [`JointTrainer`] — joint training scaffold with freeze controls and
//!   `AlternateEpoch` / `Curriculum` schedules (phase c).
//! - [`LoraAdapter`] / [`LoraTrainer`] — low-rank adaptation adapter and
//!   fine-tuning scaffold for the GNN→LLM projection (phase d).

pub mod joint_trainer;
pub mod llm_head;
pub mod lora;
pub mod provider;
pub mod soft_prompt;

pub use joint_trainer::{EpochMetrics, JointTrainer, Schedule, TrainingHistory};
pub use llm_head::{HybridLlmHead, KgqaExample, LlmHeadHistory};
pub use lora::{LoraAdapter, LoraTrainer};
pub use provider::{
    Capabilities, CompletionRequest, CompletionResponse, LlmError, LlmProvider, LocalProvider,
};
pub use soft_prompt::SoftPromptProjector;
