//! Enterprise Knowledge Graphs - Business Domain Embeddings
//!
//! This module provides specialized embeddings and analysis for enterprise knowledge graphs,
//! including product catalogs, organizational knowledge, employee skill embeddings, and
//! recommendation systems for business applications.
//!
//! This is a thin facade over the focused companion modules. The
//! [`EnterpriseKnowledgeAnalyzer`] struct is defined here so that its inherent
//! methods can be split across companion modules, while all domain types are
//! re-exported from:
//! - [`enterprise_knowledge_config`](crate::enterprise_knowledge_config): analyzer
//!   and recommendation configuration.
//! - [`enterprise_knowledge_product`](crate::enterprise_knowledge_product): product,
//!   category, and sales types.
//! - [`enterprise_knowledge_employee`](crate::enterprise_knowledge_employee): employee,
//!   skill, organizational, and project types.
//! - [`enterprise_knowledge_customer`](crate::enterprise_knowledge_customer): customer,
//!   purchase, preference, and recommendation types.
//! - [`enterprise_knowledge_engine`](crate::enterprise_knowledge_engine): recommendation
//!   engine, market analysis, and enterprise metrics types.
//!
//! The analyzer's public methods live in
//! [`enterprise_knowledge_analyzer`](crate::enterprise_knowledge_analyzer) and its private
//! helpers, background tasks, and metrics live in
//! [`enterprise_knowledge_analyzer_helpers`](crate::enterprise_knowledge_analyzer_helpers).

pub use crate::enterprise_knowledge_config::*;
pub use crate::enterprise_knowledge_customer::*;
pub use crate::enterprise_knowledge_employee::*;
pub use crate::enterprise_knowledge_engine::*;
pub use crate::enterprise_knowledge_product::*;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::task::JoinHandle;

/// Enterprise knowledge graph analyzer and embedding generator
pub struct EnterpriseKnowledgeAnalyzer {
    /// Product catalog embeddings
    pub(crate) product_embeddings: Arc<RwLock<HashMap<String, ProductEmbedding>>>,
    /// Employee embeddings
    pub(crate) employee_embeddings: Arc<RwLock<HashMap<String, EmployeeEmbedding>>>,
    /// Customer embeddings
    pub(crate) customer_embeddings: Arc<RwLock<HashMap<String, CustomerEmbedding>>>,
    /// Product categories and hierarchies
    pub(crate) category_hierarchy: Arc<RwLock<CategoryHierarchy>>,
    /// Organizational structure
    pub(crate) organizational_structure: Arc<RwLock<OrganizationalStructure>>,
    /// Recommendation engines
    pub(crate) recommendation_engines: Arc<RwLock<HashMap<String, RecommendationEngine>>>,
    /// Configuration
    pub(crate) config: EnterpriseConfig,
    /// Background analysis tasks
    pub(crate) analysis_tasks: Vec<JoinHandle<()>>,
}

impl EnterpriseKnowledgeAnalyzer {
    /// Create new enterprise knowledge analyzer
    pub fn new(config: EnterpriseConfig) -> Self {
        Self {
            product_embeddings: Arc::new(RwLock::new(HashMap::new())),
            employee_embeddings: Arc::new(RwLock::new(HashMap::new())),
            customer_embeddings: Arc::new(RwLock::new(HashMap::new())),
            category_hierarchy: Arc::new(RwLock::new(CategoryHierarchy {
                categories: HashMap::new(),
                parent_child: HashMap::new(),
                category_embeddings: HashMap::new(),
            })),
            organizational_structure: Arc::new(RwLock::new(OrganizationalStructure {
                departments: HashMap::new(),
                teams: HashMap::new(),
                reporting_structure: HashMap::new(),
                projects: HashMap::new(),
            })),
            recommendation_engines: Arc::new(RwLock::new(HashMap::new())),
            config,
            analysis_tasks: Vec::new(),
        }
    }
}
