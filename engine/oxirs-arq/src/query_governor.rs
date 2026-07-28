//! Runtime query resource governor for ARQ.
//!
//! Enforces CPU wall-time, result-row, and triple-scan budgets **during**
//! query execution (not just at admission time).  All atomic counters use
//! [`Ordering::Relaxed`] — this is intentional: enforcement is best-effort
//! and is not a security boundary.
//!
//! # Quick start
//!
//! ```rust
//! use oxirs_arq::query_governor::{ExecutionBudget, ResourceBudget};
//!
//! let budget = ExecutionBudget::new(ResourceBudget::for_sla_tier("silver"));
//! budget.record_triple_scan(1).expect("within scan limit");
//! budget.record_result_row().expect("within row limit");
//! budget.check_time().expect("within wall-time limit");
//! ```

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────
// ResourceBudget
// ─────────────────────────────────────────────────────────────

/// Resource limits for one query.  `None` means unlimited.
#[derive(Debug, Clone)]
pub struct ResourceBudget {
    /// Maximum elapsed wall-clock time allowed.
    pub max_wall_time: Option<Duration>,
    /// Maximum number of result rows that may be produced.
    pub max_result_rows: Option<u64>,
    /// Maximum number of triples that may be scanned from the store.
    pub max_triples_scanned: Option<u64>,
}

impl ResourceBudget {
    /// No limits on any dimension.
    #[must_use]
    pub fn unlimited() -> Self {
        Self {
            max_wall_time: None,
            max_result_rows: None,
            max_triples_scanned: None,
        }
    }

    /// Pre-built budgets keyed by SLA tier name.
    ///
    /// | Tier     | Wall-time | Result rows | Triples scanned |
    /// |----------|-----------|-------------|-----------------|
    /// | bronze   | 30 s      | 1 000       | 1 000 000       |
    /// | silver   | 60 s      | 10 000      | 5 000 000       |
    /// | gold     | 300 s     | 100 000     | 50 000 000      |
    /// | platinum | unlimited | unlimited   | unlimited       |
    ///
    /// Unknown tier names fall back to `unlimited()`.
    #[must_use]
    pub fn for_sla_tier(tier: &str) -> Self {
        match tier.to_ascii_lowercase().as_str() {
            "bronze" => Self {
                max_wall_time: Some(Duration::from_secs(30)),
                max_result_rows: Some(1_000),
                max_triples_scanned: Some(1_000_000),
            },
            "silver" => Self {
                max_wall_time: Some(Duration::from_secs(60)),
                max_result_rows: Some(10_000),
                max_triples_scanned: Some(5_000_000),
            },
            "gold" => Self {
                max_wall_time: Some(Duration::from_secs(300)),
                max_result_rows: Some(100_000),
                max_triples_scanned: Some(50_000_000),
            },
            _ => Self::unlimited(),
        }
    }
}

// ─────────────────────────────────────────────────────────────
// BudgetExceeded
// ─────────────────────────────────────────────────────────────

/// The specific limit that was exceeded.
#[derive(Debug, Clone, PartialEq)]
pub enum BudgetExceeded {
    /// Wall-clock time limit was exceeded.
    TimeoutExceeded {
        /// Milliseconds elapsed since query start.
        elapsed_ms: u64,
        /// Configured limit in milliseconds.
        limit_ms: u64,
    },
    /// Result-row limit was exceeded.
    ResultRowsExceeded {
        /// Number of rows produced so far.
        produced: u64,
        /// Configured row limit.
        limit: u64,
    },
    /// Triple-scan limit was exceeded.
    TriplesScannedExceeded {
        /// Number of triples scanned so far.
        scanned: u64,
        /// Configured scan limit.
        limit: u64,
    },
}

impl fmt::Display for BudgetExceeded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimeoutExceeded {
                elapsed_ms,
                limit_ms,
            } => write!(
                f,
                "query budget exceeded: wall time {elapsed_ms} ms > limit {limit_ms} ms"
            ),
            Self::ResultRowsExceeded { produced, limit } => write!(
                f,
                "query budget exceeded: result rows {produced} > limit {limit}"
            ),
            Self::TriplesScannedExceeded { scanned, limit } => write!(
                f,
                "query budget exceeded: triples scanned {scanned} > limit {limit}"
            ),
        }
    }
}

impl std::error::Error for BudgetExceeded {}

// ─────────────────────────────────────────────────────────────
// ExecutionBudget
// ─────────────────────────────────────────────────────────────

/// Live budget tracker for a single query execution.
///
/// Create via [`ExecutionBudget::new`] and share the returned [`Arc`] with
/// any code that participates in query evaluation:
///
/// - BGP iterators call [`ExecutionBudget::record_triple_scan`].
/// - Result-row emitters call [`ExecutionBudget::record_result_row`].
/// - Join / subquery boundaries call [`ExecutionBudget::check_time`].
///
/// All methods are `&self` so the `Arc` can be shared without a mutex.
pub struct ExecutionBudget {
    budget: ResourceBudget,
    start: Instant,
    triples_scanned: AtomicU64,
    result_rows: AtomicU64,
}

impl ExecutionBudget {
    /// Allocate a new `ExecutionBudget` and start the wall-clock timer.
    #[must_use]
    pub fn new(budget: ResourceBudget) -> Arc<Self> {
        Arc::new(Self {
            budget,
            start: Instant::now(),
            triples_scanned: AtomicU64::new(0),
            result_rows: AtomicU64::new(0),
        })
    }

    /// Record `n` triples scanned at a BGP iterator step, then verify all
    /// three budget limits.
    ///
    /// Call order: time check first (cheapest, no mutation) → fetch_add →
    /// post-add value compared against the configured limit.
    ///
    /// # Errors
    ///
    /// Returns `Err(BudgetExceeded::*)` when any limit is exceeded.
    pub fn record_triple_scan(&self, n: u64) -> Result<(), BudgetExceeded> {
        // 1. Time — cheapest, no mutation
        self.check_time()?;

        // 2. Accumulate and get post-add value
        let total = self
            .triples_scanned
            .fetch_add(n, Ordering::Relaxed)
            .saturating_add(n);

        // 3. Compare post-add value against limit
        if let Some(limit) = self.budget.max_triples_scanned {
            if total > limit {
                return Err(BudgetExceeded::TriplesScannedExceeded {
                    scanned: total,
                    limit,
                });
            }
        }

        // 4. Also surface any row-limit breach (consistent check point)
        let rows = self.result_rows.load(Ordering::Relaxed);
        if let Some(limit) = self.budget.max_result_rows {
            if rows > limit {
                return Err(BudgetExceeded::ResultRowsExceeded {
                    produced: rows,
                    limit,
                });
            }
        }

        Ok(())
    }

    /// Record one result row emitted, then check all budget limits.
    ///
    /// # Errors
    ///
    /// Returns `Err(BudgetExceeded::*)` when any limit is exceeded.
    pub fn record_result_row(&self) -> Result<(), BudgetExceeded> {
        // 1. Time
        self.check_time()?;

        // 2. Accumulate and get post-add value
        let total = self
            .result_rows
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);

        // 3. Compare post-add value against limit
        if let Some(limit) = self.budget.max_result_rows {
            if total > limit {
                return Err(BudgetExceeded::ResultRowsExceeded {
                    produced: total,
                    limit,
                });
            }
        }

        // 4. Also surface any triple-scan breach at this checkpoint. Row emission
        // and triple scanning interleave (a result-heavy query may pass the row
        // limit while already far over the scan limit, or vice versa); mirroring
        // `record_triple_scan`'s cross-check keeps every checkpoint able to abort
        // on whichever limit was breached first.
        let scanned = self.triples_scanned.load(Ordering::Relaxed);
        if let Some(limit) = self.budget.max_triples_scanned {
            if scanned > limit {
                return Err(BudgetExceeded::TriplesScannedExceeded { scanned, limit });
            }
        }

        Ok(())
    }

    /// Check only the wall-clock time limit — call this at join boundaries
    /// where no new data has been produced yet.
    ///
    /// # Errors
    ///
    /// Returns `Err(BudgetExceeded::TimeoutExceeded)` when the wall-time limit
    /// is exceeded.
    pub fn check_time(&self) -> Result<(), BudgetExceeded> {
        if let Some(limit) = self.budget.max_wall_time {
            let elapsed = self.start.elapsed();
            if elapsed > limit {
                return Err(BudgetExceeded::TimeoutExceeded {
                    elapsed_ms: elapsed.as_millis() as u64,
                    limit_ms: limit.as_millis() as u64,
                });
            }
        }
        Ok(())
    }

    /// Elapsed time since this budget was created.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Total triples scanned so far (snapshot).
    #[must_use]
    pub fn triples_scanned(&self) -> u64 {
        self.triples_scanned.load(Ordering::Relaxed)
    }

    /// Total result rows produced so far (snapshot).
    #[must_use]
    pub fn result_rows(&self) -> u64 {
        self.result_rows.load(Ordering::Relaxed)
    }

    /// Reference to the underlying [`ResourceBudget`] configuration.
    #[must_use]
    pub fn budget(&self) -> &ResourceBudget {
        &self.budget
    }
}

impl fmt::Debug for ExecutionBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutionBudget")
            .field("budget", &self.budget)
            .field("elapsed", &self.start.elapsed())
            .field(
                "triples_scanned",
                &self.triples_scanned.load(Ordering::Relaxed),
            )
            .field("result_rows", &self.result_rows.load(Ordering::Relaxed))
            .finish()
    }
}
