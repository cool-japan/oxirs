//! Storage recovery and integrity reporting

/// Crash recovery report
#[derive(Debug, Clone)]
pub struct RecoveryReport {
    pub wal_recovered: bool,
    pub corrupted_files: Vec<String>,
    pub recovered_files: Vec<String>,
    pub log_inconsistencies: usize,
    pub state_machine_repaired: bool,
}

impl RecoveryReport {
    pub fn new() -> Self {
        Self {
            wal_recovered: false,
            corrupted_files: Vec::new(),
            recovered_files: Vec::new(),
            log_inconsistencies: 0,
            state_machine_repaired: false,
        }
    }
}

/// File corruption report
#[derive(Debug, Clone)]
pub struct CorruptionReport {
    pub corrupted_files: Vec<String>,
    pub recovered_files: Vec<String>,
}

impl CorruptionReport {
    pub fn new() -> Self {
        Self {
            corrupted_files: Vec::new(),
            recovered_files: Vec::new(),
        }
    }
}

/// Log consistency report
#[derive(Debug, Clone)]
pub struct LogConsistencyReport {
    pub is_consistent: bool,
    pub issues: Vec<LogInconsistency>,
}

impl LogConsistencyReport {
    pub fn new() -> Self {
        Self {
            is_consistent: true,
            issues: Vec::new(),
        }
    }
}

/// Log inconsistency types
#[derive(Debug, Clone)]
pub enum LogInconsistency {
    IndexGap {
        expected: u64,
        found: u64,
    },
    DuplicateIndex {
        index: u64,
    },
    InvalidCommitIndex {
        commit_index: u64,
        last_log_index: u64,
    },
}

/// State consistency report
#[derive(Debug, Clone)]
pub struct StateConsistencyReport {
    pub repaired: bool,
}

impl StateConsistencyReport {
    pub fn new() -> Self {
        Self { repaired: false }
    }
}
