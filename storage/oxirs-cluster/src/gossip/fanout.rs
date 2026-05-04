//! Gossip fanout policy for epidemic-protocol dissemination.
//!
//! Controls how many peers are contacted per gossip round.
//! The choice of fanout is fundamental to convergence speed and network load:
//!
//! - **`Unbounded`** — original behaviour; contacts every peer each round.
//!   Safe for tiny clusters (≤ 32 nodes) but O(N) traffic at scale.
//! - **`Sqrt`** — the epidemic-protocol default; contacts ⌊√N⌋ peers each
//!   round.  Guarantees O(log log N) convergence with O(N √N) total messages,
//!   which is far superior to O(N²) for Unbounded at large N.
//! - **`Bounded(k)`** — contacts exactly `min(k, N)` peers; useful when a
//!   specific fanout budget is required (e.g., hardware-constrained clusters
//!   or environments with strict QoS limits).
//!
//! # Choosing a fanout
//!
//! `GossipFanout::default_for(n)` returns `Sqrt` when `n > 32` and
//! `Unbounded` for smaller clusters where the overhead of sub-sampling
//! outweighs its benefits.

/// Controls how many peers are contacted per gossip round.
///
/// The variants cover the three standard operational regimes:
/// deterministic budget, epidemic-protocol default, and legacy full-fan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GossipFanout {
    /// Contact exactly `n` peers per round (random sample, capped at cluster size).
    Bounded(usize),
    /// Contact ⌊√N⌋ peers per round — epidemic-protocol default.
    ///
    /// Provides O(log log N) convergence with O(N √N) total messages.
    Sqrt,
    /// Contact all peers — original behaviour; only suitable for small clusters.
    Unbounded,
}

impl GossipFanout {
    /// Compute the actual fanout for a cluster of size `n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_cluster::gossip::fanout::GossipFanout;
    ///
    /// assert_eq!(GossipFanout::Bounded(5).resolve(1000), 5);
    /// assert_eq!(GossipFanout::Sqrt.resolve(100), 10);
    /// assert_eq!(GossipFanout::Unbounded.resolve(8), 8);
    /// ```
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    pub fn resolve(&self, n: usize) -> usize {
        match self {
            GossipFanout::Bounded(k) => (*k).min(n),
            GossipFanout::Sqrt => (n as f64).sqrt().floor() as usize,
            GossipFanout::Unbounded => n,
        }
    }

    /// Return the recommended fanout policy for a cluster of size `n`.
    ///
    /// Uses `Sqrt` for clusters larger than 32 nodes (epidemic-protocol
    /// default) and `Unbounded` for smaller clusters where the full-fan
    /// overhead is acceptable.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_cluster::gossip::fanout::GossipFanout;
    ///
    /// assert_eq!(GossipFanout::default_for(1000), GossipFanout::Sqrt);
    /// assert_eq!(GossipFanout::default_for(10), GossipFanout::Unbounded);
    /// assert_eq!(GossipFanout::default_for(32), GossipFanout::Unbounded);
    /// assert_eq!(GossipFanout::default_for(33), GossipFanout::Sqrt);
    /// ```
    pub fn default_for(n: usize) -> Self {
        if n > 32 {
            GossipFanout::Sqrt
        } else {
            GossipFanout::Unbounded
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_capped_at_cluster_size() {
        let f = GossipFanout::Bounded(5);
        assert_eq!(f.resolve(1000), 5);
        // Never exceeds N even when k > N
        assert_eq!(f.resolve(3), 3);
    }

    #[test]
    fn test_bounded_zero_cluster() {
        let f = GossipFanout::Bounded(5);
        assert_eq!(f.resolve(0), 0);
    }

    #[test]
    fn test_sqrt_100_nodes() {
        let f = GossipFanout::Sqrt;
        assert_eq!(f.resolve(100), 10);
    }

    #[test]
    fn test_sqrt_1000_nodes() {
        // floor(sqrt(1000)) = 31
        let f = GossipFanout::Sqrt;
        assert_eq!(f.resolve(1000), 31);
    }

    #[test]
    fn test_sqrt_zero() {
        let f = GossipFanout::Sqrt;
        assert_eq!(f.resolve(0), 0);
    }

    #[test]
    fn test_unbounded_returns_n() {
        let f = GossipFanout::Unbounded;
        assert_eq!(f.resolve(8), 8);
        assert_eq!(f.resolve(1000), 1000);
    }

    #[test]
    fn test_default_for_small_cluster() {
        assert_eq!(GossipFanout::default_for(10), GossipFanout::Unbounded);
        assert_eq!(GossipFanout::default_for(32), GossipFanout::Unbounded);
    }

    #[test]
    fn test_default_for_large_cluster() {
        assert_eq!(GossipFanout::default_for(33), GossipFanout::Sqrt);
        assert_eq!(GossipFanout::default_for(1000), GossipFanout::Sqrt);
    }

    #[test]
    fn test_eq_and_copy() {
        let a = GossipFanout::Bounded(3);
        let b = a;
        assert_eq!(a, b);
    }
}
