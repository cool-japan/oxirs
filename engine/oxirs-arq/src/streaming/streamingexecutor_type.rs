//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

use super::functions::DataStream;
use super::types::{MemoryMonitor, SpillManager, StreamingConfig, StreamingStats};

/// Streaming execution engine for large datasets
pub struct StreamingExecutor {
    pub(super) config: StreamingConfig,
    pub(super) memory_monitor: MemoryMonitor,
    pub(super) spill_manager: Arc<Mutex<SpillManager>>,
    #[allow(dead_code)]
    pub(super) temp_dir: TempDir,
    pub(super) active_streams: HashMap<String, Box<dyn DataStream>>,
    pub(super) execution_stats: StreamingStats,
}
