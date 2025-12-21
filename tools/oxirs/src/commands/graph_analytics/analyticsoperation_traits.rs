//! # AnalyticsOperation - Trait Implementations
//!
//! This module contains trait implementations for `AnalyticsOperation`.
//!
//! ## Implemented Traits
//!
//! - `FromStr`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::AnalyticsOperation;

impl std::str::FromStr for AnalyticsOperation {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "pagerank" | "pr" => Ok(Self::PageRank),
            "degree" | "degrees" | "dd" => Ok(Self::DegreeDistribution),
            "community" | "communities" | "cd" => Ok(Self::CommunityDetection),
            "paths" | "shortest" | "sp" => Ok(Self::ShortestPaths),
            "stats" | "statistics" => Ok(Self::GraphStats),
            "betweenness" | "bc" => Ok(Self::BetweennessCentrality),
            "closeness" | "cc" => Ok(Self::ClosenessCentrality),
            "eigenvector" | "ec" => Ok(Self::EigenvectorCentrality),
            "katz" | "kc" => Ok(Self::KatzCentrality),
            "hits" | "ha" => Ok(Self::HitsAlgorithm),
            "louvain" | "lc" => Ok(Self::LouvainCommunities),
            "kcore" | "k-core" | "decomposition" => Ok(Self::KCoreDecomposition),
            "triangles" | "triangle" | "tc" => Ok(Self::TriangleCounting),
            "diameter" | "radius" | "dr" => Ok(Self::DiameterRadius),
            "center" | "center-nodes" | "cn" => Ok(Self::CenterNodes),
            "motifs" | "extended-motifs" | "em" => Ok(Self::ExtendedMotifs),
            "coloring" | "color" | "vertex-coloring" => Ok(Self::GraphColoring),
            "matching" | "max-matching" | "bipartite" => Ok(Self::MaximumMatching),
            "flow" | "max-flow" | "min-cut" | "network-flow" => Ok(Self::NetworkFlow),
            _ => Err(anyhow::anyhow!("Unknown analytics operation: {}", s)),
        }
    }
}
