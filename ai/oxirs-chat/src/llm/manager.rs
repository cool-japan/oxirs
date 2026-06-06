//! LLM Manager — facade module
//!
//! Re-exports the full public API from the split sub-modules:
//! - `manager_types`  — types, session, usage tracker, rate limiter, stats
//! - `manager_router` — provider routing, fallback chain, circuit-breaker helpers
//! - `manager_cache`  — response caching and token budget helpers
//! - `manager_impl`   — `LLMManager` and `EnhancedLLMManager` core structs
//! - `manager_tests`  — integration tests (private)

pub use crate::llm::manager_cache::*;
pub use crate::llm::manager_impl::{EnhancedLLMManager, LLMManager};
pub use crate::llm::manager_router::*;
pub use crate::llm::manager_types::*;
