//! Distributed training module facade.
//!
//! Re-exports the public surface of the split sibling modules so that
//! existing call sites continue to resolve symbols such as
//! [`ParameterVector`], [`DistributedTrainer`] and [`FederatedShapeTrainer`]
//! through `crate::training::distributed::*`.

pub use crate::training::distributed_coordinator::*;
pub use crate::training::distributed_types::*;
pub use crate::training::distributed_worker::*;
