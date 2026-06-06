//! [`EnhancedValuesProcessor`] — VALUES clause processing with optimization.
//!
//! Split out of the original `bind_values_enhanced` module (Round 32 refactor).
//! Contains the VALUES-specific entry points: clause extraction from query
//! text, parsing, value-set creation, join-strategy application, and the
//! `Default` impl.

use crate::error::FusekiResult;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::bind_values_enhanced_types::{
    AdvancedValuesOptimizer, EnhancedValuesProcessor, JoinStrategy, JoinStrategySelector,
    JoinStrategyType, ValueSet, ValueSetManager, ValuesClause, ValuesStatistics,
};

impl Default for EnhancedValuesProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl EnhancedValuesProcessor {
    pub fn new() -> Self {
        Self {
            values_optimizer: AdvancedValuesOptimizer::new(),
            value_set_manager: ValueSetManager::new(),
            join_strategy_selector: JoinStrategySelector::new(),
            statistics: Arc::new(RwLock::new(ValuesStatistics::default())),
        }
    }

    /// Process VALUES clauses with optimization
    pub async fn process_values_clauses(
        &self,
        query: &str,
        bindings: &mut Vec<HashMap<String, serde_json::Value>>,
    ) -> FusekiResult<()> {
        let values_clauses = self.extract_values_clauses(query)?;

        for values_clause in values_clauses {
            // Optimize the VALUES clause
            let optimized = self.values_optimizer.optimize(&values_clause)?;

            // Create value set
            let value_set = self.create_value_set(&optimized)?;

            // Store in manager
            let set_id = self.value_set_manager.store_value_set(value_set).await?;

            // Select join strategy
            let strategy = self
                .join_strategy_selector
                .select_strategy(&optimized, bindings)?;

            // Apply VALUES using selected strategy
            self.apply_values_with_strategy(bindings, &set_id, strategy)
                .await?;

            // Update statistics
            self.update_statistics(&optimized).await;
        }

        Ok(())
    }

    pub(crate) fn extract_values_clauses(&self, query: &str) -> FusekiResult<Vec<ValuesClause>> {
        let mut clauses = Vec::new();

        // Find VALUES clauses
        let query_lower = query.to_lowercase();
        let mut pos = 0;

        while let Some(values_pos) = query_lower[pos..].find("values") {
            let values_start = pos + values_pos;

            // Extract the VALUES clause
            if let Some(clause_end) = self.find_values_end(&query[values_start..]) {
                let clause_text = &query[values_start..values_start + clause_end];

                if let Ok(parsed) = self.parse_values_clause(clause_text) {
                    clauses.push(parsed);
                }
            }

            pos = values_start + 6;
        }

        Ok(clauses)
    }

    fn find_values_end(&self, text: &str) -> Option<usize> {
        let mut brace_count = 0;
        let mut found_opening = false;

        for (i, ch) in text.chars().enumerate() {
            match ch {
                '{' => {
                    brace_count += 1;
                    found_opening = true;
                }
                '}' => {
                    brace_count -= 1;
                    if found_opening && brace_count == 0 {
                        return Some(i + 1);
                    }
                }
                _ => {}
            }
        }

        None
    }

    fn parse_values_clause(&self, text: &str) -> FusekiResult<ValuesClause> {
        // Parse VALUES clause structure
        let mut variables = Vec::new();
        let mut rows = Vec::new();

        // Find the variable list
        if let Some(var_start) = text.find('(') {
            if let Some(var_end) = text[var_start..].find(')') {
                let var_text = &text[var_start + 1..var_start + var_end];

                // Extract variable names
                for var in var_text.split_whitespace() {
                    if var.starts_with('?') || var.starts_with('$') {
                        variables.push(var.to_string());
                    }
                }
            }
        }

        // Find the values block
        if let Some(block_start) = text.find('{') {
            if let Some(block_end) = text.rfind('}') {
                let block_text = &text[block_start + 1..block_end];

                // Parse rows - each row is in parentheses
                let mut current_row = Vec::new();
                let mut in_parens = false;
                let mut current_value = String::new();
                let mut in_quotes = false;

                for ch in block_text.chars() {
                    match ch {
                        '"' => {
                            in_quotes = !in_quotes;
                            current_value.push(ch);
                        }
                        '(' if !in_quotes => {
                            in_parens = true;
                            current_row.clear();
                        }
                        ')' if !in_quotes => {
                            if !current_value.trim().is_empty() {
                                current_row.push(current_value.trim().to_string());
                                current_value.clear();
                            }
                            if !current_row.is_empty() {
                                // Convert strings to JSON values
                                let json_row: Vec<serde_json::Value> = current_row
                                    .iter()
                                    .map(|s| {
                                        if s.starts_with('"') && s.ends_with('"') {
                                            // String literal
                                            serde_json::Value::String(s[1..s.len() - 1].to_string())
                                        } else if s.starts_with(':') {
                                            // IRI/namespace
                                            serde_json::Value::String(s.to_string())
                                        } else {
                                            // Try parsing as other types
                                            serde_json::from_str(s).unwrap_or_else(|_| {
                                                serde_json::Value::String(s.to_string())
                                            })
                                        }
                                    })
                                    .collect();
                                rows.push(json_row);
                            }
                            in_parens = false;
                        }
                        c if in_parens => {
                            if c.is_whitespace() && !in_quotes {
                                if !current_value.trim().is_empty() {
                                    current_row.push(current_value.trim().to_string());
                                    current_value.clear();
                                }
                            } else {
                                current_value.push(c);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(ValuesClause {
            variables,
            rows,
            is_inline: text.len() < 1000,
        })
    }

    fn create_value_set(&self, clause: &ValuesClause) -> FusekiResult<ValueSet> {
        let mut values = Vec::new();

        for row in &clause.rows {
            let mut binding = HashMap::new();
            for (i, var) in clause.variables.iter().enumerate() {
                if let Some(val) = row.get(i) {
                    binding.insert(var.clone(), val.clone());
                }
            }
            values.push(binding);
        }

        Ok(ValueSet {
            id: uuid::Uuid::new_v4().to_string(),
            values,
            variables: clause.variables.clone(),
            indexed: false,
            compressed: false,
        })
    }

    async fn apply_values_with_strategy(
        &self,
        bindings: &mut Vec<HashMap<String, serde_json::Value>>,
        set_id: &str,
        strategy: JoinStrategy,
    ) -> FusekiResult<()> {
        match strategy.strategy_type {
            JoinStrategyType::NestedLoop => self.apply_nested_loop_join(bindings, set_id).await,
            JoinStrategyType::HashJoin => self.apply_hash_join(bindings, set_id).await,
            JoinStrategyType::SortMergeJoin => self.apply_sort_merge_join(bindings, set_id).await,
            _ => self.apply_nested_loop_join(bindings, set_id).await,
        }
    }

    async fn apply_nested_loop_join(
        &self,
        bindings: &mut Vec<HashMap<String, serde_json::Value>>,
        set_id: &str,
    ) -> FusekiResult<()> {
        let value_set = self.value_set_manager.get_value_set(set_id).await?;

        let mut new_bindings = Vec::new();

        for binding in bindings.iter() {
            for value_row in &value_set.values {
                let mut combined = binding.clone();
                for (var, val) in value_row {
                    combined.insert(var.clone(), val.clone());
                }
                new_bindings.push(combined);
            }
        }

        *bindings = new_bindings;
        Ok(())
    }

    async fn apply_hash_join(
        &self,
        bindings: &mut Vec<HashMap<String, serde_json::Value>>,
        set_id: &str,
    ) -> FusekiResult<()> {
        // Implement hash join
        self.apply_nested_loop_join(bindings, set_id).await
    }

    async fn apply_sort_merge_join(
        &self,
        bindings: &mut Vec<HashMap<String, serde_json::Value>>,
        set_id: &str,
    ) -> FusekiResult<()> {
        // Implement sort-merge join
        self.apply_nested_loop_join(bindings, set_id).await
    }

    async fn update_statistics(&self, clause: &ValuesClause) {
        {
            let mut stats = self.statistics.write().await;
            stats.total_values_clauses += 1;
            stats.total_value_count += clause.rows.len() as u64;
        }
    }
}
