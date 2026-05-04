//! Algebra-level JIT plan cache — phase a.
//!
//! This module provides a bounded, LRU-evicting, thread-safe cache for
//! optimised SPARQL algebra plans.  Queries whose fingerprints match a cached
//! entry skip the optimizer entirely, reducing hot-query latency.
//!
//! ## Modules
//! - [`fingerprint`] — stable `u64` hash of an algebra tree with variable-name
//!   normalisation via [`seahash::SeaHasher`].
//! - [`eviction`] — [`LruEviction`] policy using a `VecDeque<u64>` access
//!   list.
//! - [`cache`] — [`PlanCache<V>`] — bounded, `parking_lot::RwLock`-guarded
//!   cache with schema-version invalidation.
//!
//! ## Quick start
//!
//! ```rust
//! use oxirs_arq::plan_cache::{PlanCache, compute_fingerprint};
//! use oxirs_arq::algebra::{Algebra, Term, Variable, TriplePattern};
//! use oxirs_core::model::NamedNode;
//!
//! // Build a trivial plan and fingerprint it.
//! let pred = Term::Iri(NamedNode::new_unchecked("http://example.org/p"));
//! let plan = Algebra::Bgp(vec![TriplePattern {
//!     subject:   Term::Variable(Variable::new("s").unwrap()),
//!     predicate: pred,
//!     object:    Term::Variable(Variable::new("o").unwrap()),
//! }]);
//!
//! let key = compute_fingerprint(&plan);
//!
//! let cache: PlanCache<Algebra> = PlanCache::new(1024);
//! cache.insert(key, plan.clone());
//! assert!(cache.get(key).is_some());
//! ```

pub mod cache;
pub mod eviction;
pub mod fingerprint;

pub use cache::{CacheStats, PlanCache};
pub use eviction::LruEviction;
pub use fingerprint::compute_fingerprint;
