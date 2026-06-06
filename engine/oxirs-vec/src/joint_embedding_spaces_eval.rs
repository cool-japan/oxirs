//! Joint Embedding Spaces — evaluation utilities.
//!
//! Cross-space retrieval accuracy, word translation accuracy,
//! neighborhood preservation metrics.
//!
//! The primary evaluation entry-point (`evaluate_retrieval`) is implemented on
//! `JointEmbeddingSpace` in joint_embedding_spaces_aligner.  Zero-shot
//! retrieval helpers live in joint_embedding_spaces_transfer.

pub use crate::joint_embedding_spaces_aligner::JointEmbeddingSpace;
pub use crate::joint_embedding_spaces_transfer::zero_shot_retrieval;
