//! Multi-Layer Memory System
//!
//! Implements working, episodic, and semantic memory layers.

use anyhow::Result;

use super::super::*;

#[derive(Debug, Clone)]
pub struct MultiLayerMemorySystem {
    working_memory: WorkingMemory,
    episodic_memory: EpisodicMemory,
    semantic_memory: SemanticMemory,
    memory_consolidation: MemoryConsolidation,
}

impl MultiLayerMemorySystem {
    pub fn new() -> Result<Self> {
        Ok(Self {
            working_memory: WorkingMemory::new()?,
            episodic_memory: EpisodicMemory::new()?,
            semantic_memory: SemanticMemory::new()?,
            memory_consolidation: MemoryConsolidation::new()?,
        })
    }

    pub fn process_and_integrate(
        &mut self,
        query: &str,
        context: &AssembledContext,
        attention: &AttentionAllocation,
    ) -> Result<MemoryIntegrationResult> {
        // Store in working memory
        let working_trace = self.working_memory.store_immediate(query, context)?;

        // Create episodic memory entry
        let episodic_entry = self
            .episodic_memory
            .create_episode(query, context, attention)?;

        // Update semantic associations
        let semantic_updates = self.semantic_memory.update_associations(query, context)?;

        // Perform memory consolidation
        let consolidation_result = self.memory_consolidation.consolidate(
            &working_trace,
            &episodic_entry,
            &semantic_updates,
        )?;

        Ok(MemoryIntegrationResult {
            integration_strength: consolidation_result.strength,
            confidence: consolidation_result.confidence,
            consciousness_relevance: consolidation_result.consciousness_correlation,
            working_memory_load: self.working_memory.get_load()?,
            episodic_coherence: self.episodic_memory.get_coherence()?,
            semantic_connectivity: self.semantic_memory.get_connectivity()?,
        })
    }

    pub fn get_memory_pressure(&self) -> Result<f64> {
        let working_pressure = self.working_memory.get_pressure()?;
        let episodic_pressure = self.episodic_memory.get_pressure()?;
        let semantic_pressure = self.semantic_memory.get_pressure()?;

        Ok((working_pressure + episodic_pressure + semantic_pressure) / 3.0)
    }

    pub fn get_health_score(&self) -> Result<f64> {
        let working_health = self.working_memory.get_health()?;
        let episodic_health = self.episodic_memory.get_health()?;
        let semantic_health = self.semantic_memory.get_health()?;

        Ok((working_health + episodic_health + semantic_health) / 3.0)
    }

    /// Store episodic memory entry
    pub fn store_episodic_memory(&mut self, query: &str, context_summary: &str) -> Result<()> {
        // Create a simple episodic memory entry
        self.episodic_memory
            .store_simple_entry(query, context_summary.to_string())?;
        Ok(())
    }
}
