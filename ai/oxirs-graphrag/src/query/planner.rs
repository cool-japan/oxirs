//! Query execution planning

use crate::{config::GraphRAGConfig, GraphRAGResult};
use serde::{Deserialize, Serialize};

use super::parser::ParsedQuery;

/// Query execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    /// Stages in execution order
    pub stages: Vec<PlanStage>,
    /// Estimated cost (arbitrary units)
    pub estimated_cost: f64,
    /// Whether to use parallel execution
    pub parallel: bool,
}

/// A stage in the query plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStage {
    /// Stage name
    pub name: String,
    /// Stage type
    pub stage_type: StageType,
    /// Parameters for this stage
    pub params: StageParams,
    /// Dependencies on other stages
    pub depends_on: Vec<usize>,
}

/// Type of execution stage
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum StageType {
    /// Query embedding
    Embed,
    /// Vector similarity search
    VectorSearch,
    /// Keyword/BM25 search
    KeywordSearch,
    /// Result fusion
    Fusion,
    /// Graph expansion via SPARQL
    GraphExpansion,
    /// Community detection
    CommunityDetection,
    /// Context building
    ContextBuild,
    /// LLM generation
    Generation,
}

/// Parameters for a stage
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StageParams {
    /// Top-K for search stages
    pub top_k: Option<usize>,
    /// Threshold for filtering
    pub threshold: Option<f32>,
    /// SPARQL query template
    pub sparql_template: Option<String>,
    /// Maximum results
    pub max_results: Option<usize>,
    /// Other parameters
    #[serde(default)]
    pub extra: std::collections::HashMap<String, String>,
}

/// Query planner
pub struct QueryPlanner {
    config: GraphRAGConfig,
}

impl QueryPlanner {
    pub fn new(config: GraphRAGConfig) -> Self {
        Self { config }
    }

    /// Create execution plan for a parsed query
    pub fn plan(&self, parsed: &ParsedQuery) -> GraphRAGResult<QueryPlan> {
        let mut stages = Vec::new();
        let mut stage_idx = 0;

        // Stage 0: Embed query
        stages.push(PlanStage {
            name: "embed_query".to_string(),
            stage_type: StageType::Embed,
            params: StageParams::default(),
            depends_on: vec![],
        });
        let embed_stage = stage_idx;
        stage_idx += 1;

        // Stage 1: Vector search (parallel with keyword)
        stages.push(PlanStage {
            name: "vector_search".to_string(),
            stage_type: StageType::VectorSearch,
            params: StageParams {
                top_k: Some(self.config.top_k),
                threshold: Some(self.config.similarity_threshold),
                ..Default::default()
            },
            depends_on: vec![embed_stage],
        });
        let vector_stage = stage_idx;
        stage_idx += 1;

        // Stage 2: Keyword search (parallel with vector)
        stages.push(PlanStage {
            name: "keyword_search".to_string(),
            stage_type: StageType::KeywordSearch,
            params: StageParams {
                top_k: Some(self.config.top_k),
                extra: parsed
                    .keywords
                    .iter()
                    .enumerate()
                    .map(|(i, k)| (format!("keyword_{}", i), k.clone()))
                    .collect(),
                ..Default::default()
            },
            depends_on: vec![], // Can run in parallel with vector search
        });
        let keyword_stage = stage_idx;
        stage_idx += 1;

        // Stage 3: Fusion
        stages.push(PlanStage {
            name: "fusion".to_string(),
            stage_type: StageType::Fusion,
            params: StageParams {
                max_results: Some(self.config.max_seeds),
                ..Default::default()
            },
            depends_on: vec![vector_stage, keyword_stage],
        });
        let fusion_stage = stage_idx;
        stage_idx += 1;

        // Stage 4: Graph expansion
        stages.push(PlanStage {
            name: "graph_expansion".to_string(),
            stage_type: StageType::GraphExpansion,
            params: StageParams {
                max_results: Some(self.config.max_subgraph_size),
                extra: [("hops".to_string(), self.config.expansion_hops.to_string())]
                    .into_iter()
                    .collect(),
                ..Default::default()
            },
            depends_on: vec![fusion_stage],
        });
        let expansion_stage = stage_idx;
        stage_idx += 1;

        // Stage 5: Community detection (optional)
        let community_stage = if self.config.enable_communities {
            stages.push(PlanStage {
                name: "community_detection".to_string(),
                stage_type: StageType::CommunityDetection,
                params: StageParams {
                    extra: [(
                        "algorithm".to_string(),
                        format!("{:?}", self.config.community_algorithm),
                    )]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                },
                depends_on: vec![expansion_stage],
            });
            let idx = stage_idx;
            stage_idx += 1;
            Some(idx)
        } else {
            None
        };

        // Stage 6: Context building
        let context_deps = if let Some(comm_stage) = community_stage {
            vec![expansion_stage, comm_stage]
        } else {
            vec![expansion_stage]
        };
        stages.push(PlanStage {
            name: "context_build".to_string(),
            stage_type: StageType::ContextBuild,
            params: StageParams {
                max_results: Some(self.config.max_context_triples),
                ..Default::default()
            },
            depends_on: context_deps,
        });
        let context_stage = stage_idx;
        stage_idx += 1;

        // Stage 7: LLM generation
        stages.push(PlanStage {
            name: "generation".to_string(),
            stage_type: StageType::Generation,
            params: StageParams {
                extra: [
                    (
                        "temperature".to_string(),
                        self.config.temperature.to_string(),
                    ),
                    ("max_tokens".to_string(), self.config.max_tokens.to_string()),
                ]
                .into_iter()
                .collect(),
                ..Default::default()
            },
            depends_on: vec![context_stage],
        });
        let _generation_stage = stage_idx;

        // Calculate estimated cost
        let estimated_cost = self.estimate_cost(&stages);

        Ok(QueryPlan {
            stages,
            estimated_cost,
            parallel: true, // Vector and keyword search can run in parallel
        })
    }

    /// Estimate execution cost
    fn estimate_cost(&self, stages: &[PlanStage]) -> f64 {
        let mut cost = 0.0;

        for stage in stages {
            cost += match stage.stage_type {
                StageType::Embed => 10.0,
                StageType::VectorSearch => 50.0 * (stage.params.top_k.unwrap_or(20) as f64 / 20.0),
                StageType::KeywordSearch => 30.0,
                StageType::Fusion => 5.0,
                StageType::GraphExpansion => 100.0 * (self.config.expansion_hops as f64),
                StageType::CommunityDetection => 200.0,
                StageType::ContextBuild => 10.0,
                StageType::Generation => 500.0,
            };
        }

        cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::parser::QueryParser;

    #[test]
    fn test_plan_creation() {
        let config = GraphRAGConfig::default();
        let planner = QueryPlanner::new(config);
        let parser = QueryParser::new();

        let parsed = parser.parse("What are battery safety issues?").unwrap();
        let plan = planner.plan(&parsed).unwrap();

        assert!(!plan.stages.is_empty());
        assert!(plan.stages.iter().any(|s| s.stage_type == StageType::Embed));
        assert!(plan
            .stages
            .iter()
            .any(|s| s.stage_type == StageType::VectorSearch));
        assert!(plan
            .stages
            .iter()
            .any(|s| s.stage_type == StageType::Generation));
    }
}
