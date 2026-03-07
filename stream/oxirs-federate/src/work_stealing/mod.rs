//! Work-stealing scheduler for federated sub-plan execution.
//!
//! Provides a `WorkStealer` that distributes `SubPlan` tasks across a fixed
//! pool of worker queues.  When a worker's queue is empty it steals from the
//! busiest peer, minimising idle time during unbalanced workloads.

pub mod work_stealer;

pub use work_stealer::{
    SubPlan, SubPlanId, SubPlanResult, WorkStealer, WorkStealerConfig, WorkerStats,
};
