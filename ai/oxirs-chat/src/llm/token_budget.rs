//! Token Budget Management for LLM API Usage
//!
//! Enforces token limits per user/tenant to control costs and prevent abuse.

use anyhow::{anyhow, Result};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, SystemTime},
};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// User identifier
pub type UserId = String;

/// User budget configuration
#[derive(Debug, Clone)]
pub struct UserBudget {
    pub user_id: UserId,
    pub monthly_limit: u64,
    pub used_tokens: Arc<AtomicU64>,
    pub reset_date: SystemTime,
}

impl UserBudget {
    pub fn new(user_id: UserId, monthly_limit: u64) -> Self {
        let reset_date = Self::calculate_next_reset();
        Self {
            user_id,
            monthly_limit,
            used_tokens: Arc::new(AtomicU64::new(0)),
            reset_date,
        }
    }

    fn calculate_next_reset() -> SystemTime {
        // Reset on the 1st of next month
        let now = SystemTime::now();
        now + Duration::from_secs(30 * 24 * 3600) // Approximate 30 days
    }

    pub fn get_used_tokens(&self) -> u64 {
        self.used_tokens.load(Ordering::Relaxed)
    }

    pub fn get_remaining_tokens(&self) -> u64 {
        self.monthly_limit.saturating_sub(self.get_used_tokens())
    }

    pub fn is_exceeded(&self) -> bool {
        self.get_used_tokens() >= self.monthly_limit
    }

    pub fn needs_reset(&self) -> bool {
        SystemTime::now() >= self.reset_date
    }

    pub fn reset(&mut self) {
        self.used_tokens.store(0, Ordering::Relaxed);
        self.reset_date = Self::calculate_next_reset();
    }

    pub fn add_usage(&self, tokens: u64) {
        self.used_tokens.fetch_add(tokens, Ordering::Relaxed);
    }
}

/// Token budget configuration
#[derive(Debug, Clone)]
pub struct BudgetConfig {
    pub default_monthly_limit: u64,
    pub admin_monthly_limit: u64,
    pub reset_interval_days: u64,
    pub warning_threshold: f64, // Percentage (0.0 - 1.0)
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            default_monthly_limit: 100_000, // 100k tokens
            admin_monthly_limit: 1_000_000, // 1M tokens
            reset_interval_days: 30,
            warning_threshold: 0.8, // Warn at 80% usage
        }
    }
}

/// Token budget manager
pub struct TokenBudget {
    budgets: Arc<RwLock<HashMap<UserId, UserBudget>>>,
    config: BudgetConfig,
}

impl TokenBudget {
    pub fn new(config: BudgetConfig) -> Self {
        Self {
            budgets: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Check if user has sufficient budget
    pub async fn check_budget(&self, user_id: &UserId, tokens: u64) -> Result<()> {
        let budgets = self.budgets.read().await;

        if let Some(budget) = budgets.get(user_id) {
            // Check if reset needed
            if budget.needs_reset() {
                drop(budgets);
                self.reset_user_budget(user_id).await?;
                // After reset, re-check budget
                let budgets = self.budgets.read().await;
                if let Some(budget) = budgets.get(user_id) {
                    let remaining = budget.get_remaining_tokens();
                    if tokens > remaining {
                        return Err(anyhow!(
                            "Insufficient token budget. Requested: {}, Available: {}",
                            tokens,
                            remaining
                        ));
                    }
                }
                return Ok(());
            }

            let remaining = budget.get_remaining_tokens();
            if tokens > remaining {
                return Err(anyhow!(
                    "Insufficient token budget. Requested: {}, Available: {}",
                    tokens,
                    remaining
                ));
            }

            Ok(())
        } else {
            // Create budget for new user
            drop(budgets);
            self.create_user_budget(user_id.clone(), self.config.default_monthly_limit)
                .await?;
            Ok(())
        }
    }

    /// Record token usage
    pub async fn record_usage(&self, user_id: &UserId, tokens: u64) -> Result<()> {
        let budgets = self.budgets.read().await;

        if let Some(budget) = budgets.get(user_id) {
            budget.add_usage(tokens);

            let used = budget.get_used_tokens();
            let limit = budget.monthly_limit;
            let usage_percentage = used as f64 / limit as f64;

            // Warn if approaching limit
            if usage_percentage >= self.config.warning_threshold {
                warn!(
                    "User {} has used {:.1}% of monthly token budget ({}/{})",
                    user_id,
                    usage_percentage * 100.0,
                    used,
                    limit
                );
            }

            debug!(
                "Recorded {} tokens for user {}. Total: {}/{}",
                tokens, user_id, used, limit
            );

            Ok(())
        } else {
            Err(anyhow!("User budget not found: {}", user_id))
        }
    }

    /// Get remaining budget for user
    pub async fn get_remaining_budget(&self, user_id: &UserId) -> u64 {
        let budgets = self.budgets.read().await;
        budgets
            .get(user_id)
            .map(|b| b.get_remaining_tokens())
            .unwrap_or(0)
    }

    /// Get usage statistics for user
    pub async fn get_usage_stats(&self, user_id: &UserId) -> Option<UsageStats> {
        let budgets = self.budgets.read().await;
        budgets.get(user_id).map(|budget| UsageStats {
            user_id: user_id.clone(),
            used_tokens: budget.get_used_tokens(),
            monthly_limit: budget.monthly_limit,
            remaining_tokens: budget.get_remaining_tokens(),
            usage_percentage: budget.get_used_tokens() as f64 / budget.monthly_limit as f64,
            reset_date: budget.reset_date,
        })
    }

    /// Create budget for new user
    pub async fn create_user_budget(&self, user_id: UserId, monthly_limit: u64) -> Result<()> {
        let mut budgets = self.budgets.write().await;
        budgets.insert(
            user_id.clone(),
            UserBudget::new(user_id.clone(), monthly_limit),
        );
        info!(
            "Created budget for user {}: {} tokens/month",
            user_id, monthly_limit
        );
        Ok(())
    }

    /// Update user budget limit
    pub async fn update_budget_limit(&self, user_id: &UserId, new_limit: u64) -> Result<()> {
        let mut budgets = self.budgets.write().await;
        if let Some(budget) = budgets.get_mut(user_id) {
            budget.monthly_limit = new_limit;
            info!(
                "Updated budget limit for user {}: {} tokens/month",
                user_id, new_limit
            );
            Ok(())
        } else {
            Err(anyhow!("User budget not found: {}", user_id))
        }
    }

    /// Reset budget for a specific user
    pub async fn reset_user_budget(&self, user_id: &UserId) -> Result<()> {
        let mut budgets = self.budgets.write().await;
        if let Some(budget) = budgets.get_mut(user_id) {
            budget.reset();
            info!("Reset budget for user {}", user_id);
            Ok(())
        } else {
            Err(anyhow!("User budget not found: {}", user_id))
        }
    }

    /// Reset all budgets (monthly reset)
    pub async fn reset_all_budgets(&self) -> Result<usize> {
        let mut budgets = self.budgets.write().await;
        let mut count = 0;

        for (user_id, budget) in budgets.iter_mut() {
            if budget.needs_reset() {
                budget.reset();
                debug!("Reset budget for user {}", user_id);
                count += 1;
            }
        }

        info!("Reset {} user budgets", count);
        Ok(count)
    }

    /// Start periodic budget reset task
    pub async fn start_periodic_reset(&self) -> Result<()> {
        let budgets = Arc::clone(&self.budgets);
        let _config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(24 * 3600)); // Daily check

            loop {
                interval.tick().await;

                let mut budgets_guard = budgets.write().await;
                for (user_id, budget) in budgets_guard.iter_mut() {
                    if budget.needs_reset() {
                        budget.reset();
                        info!("Auto-reset budget for user {}", user_id);
                    }
                }
            }
        });

        info!("Started periodic budget reset task");
        Ok(())
    }

    /// Get all usage statistics
    pub async fn get_all_usage_stats(&self) -> Vec<UsageStats> {
        let budgets = self.budgets.read().await;
        budgets
            .iter()
            .map(|(user_id, budget)| UsageStats {
                user_id: user_id.clone(),
                used_tokens: budget.get_used_tokens(),
                monthly_limit: budget.monthly_limit,
                remaining_tokens: budget.get_remaining_tokens(),
                usage_percentage: budget.get_used_tokens() as f64 / budget.monthly_limit as f64,
                reset_date: budget.reset_date,
            })
            .collect()
    }

    /// Remove user budget
    pub async fn remove_user(&self, user_id: &UserId) -> Result<()> {
        let mut budgets = self.budgets.write().await;
        budgets.remove(user_id);
        info!("Removed budget for user {}", user_id);
        Ok(())
    }
}

/// Usage statistics for a user
#[derive(Debug, Clone)]
pub struct UsageStats {
    pub user_id: UserId,
    pub used_tokens: u64,
    pub monthly_limit: u64,
    pub remaining_tokens: u64,
    pub usage_percentage: f64,
    pub reset_date: SystemTime,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_budget_creation() {
        let config = BudgetConfig::default();
        let budget_manager = TokenBudget::new(config);

        budget_manager
            .create_user_budget("user1".to_string(), 10000)
            .await
            .unwrap();

        let remaining = budget_manager
            .get_remaining_budget(&"user1".to_string())
            .await;
        assert_eq!(remaining, 10000);
    }

    #[tokio::test]
    async fn test_budget_check_success() {
        let config = BudgetConfig::default();
        let budget_manager = TokenBudget::new(config);

        budget_manager
            .create_user_budget("user1".to_string(), 10000)
            .await
            .unwrap();

        // Should succeed
        let result = budget_manager
            .check_budget(&"user1".to_string(), 5000)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_budget_check_failure() {
        let config = BudgetConfig::default();
        let budget_manager = TokenBudget::new(config);

        budget_manager
            .create_user_budget("user1".to_string(), 10000)
            .await
            .unwrap();

        // Use all tokens
        budget_manager
            .record_usage(&"user1".to_string(), 10000)
            .await
            .unwrap();

        // Should fail
        let result = budget_manager.check_budget(&"user1".to_string(), 100).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_usage_recording() {
        let config = BudgetConfig::default();
        let budget_manager = TokenBudget::new(config);

        budget_manager
            .create_user_budget("user1".to_string(), 10000)
            .await
            .unwrap();

        budget_manager
            .record_usage(&"user1".to_string(), 3000)
            .await
            .unwrap();
        budget_manager
            .record_usage(&"user1".to_string(), 2000)
            .await
            .unwrap();

        let stats = budget_manager
            .get_usage_stats(&"user1".to_string())
            .await
            .unwrap();
        assert_eq!(stats.used_tokens, 5000);
        assert_eq!(stats.remaining_tokens, 5000);
        assert!((stats.usage_percentage - 0.5).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_budget_reset() {
        let config = BudgetConfig::default();
        let budget_manager = TokenBudget::new(config);

        budget_manager
            .create_user_budget("user1".to_string(), 10000)
            .await
            .unwrap();

        budget_manager
            .record_usage(&"user1".to_string(), 5000)
            .await
            .unwrap();

        // Reset budget
        budget_manager
            .reset_user_budget(&"user1".to_string())
            .await
            .unwrap();

        let stats = budget_manager
            .get_usage_stats(&"user1".to_string())
            .await
            .unwrap();
        assert_eq!(stats.used_tokens, 0);
        assert_eq!(stats.remaining_tokens, 10000);
    }

    #[tokio::test]
    async fn test_budget_update() {
        let config = BudgetConfig::default();
        let budget_manager = TokenBudget::new(config);

        budget_manager
            .create_user_budget("user1".to_string(), 10000)
            .await
            .unwrap();

        // Update limit
        budget_manager
            .update_budget_limit(&"user1".to_string(), 20000)
            .await
            .unwrap();

        let stats = budget_manager
            .get_usage_stats(&"user1".to_string())
            .await
            .unwrap();
        assert_eq!(stats.monthly_limit, 20000);
    }

    #[tokio::test]
    async fn test_auto_create_on_check() {
        let config = BudgetConfig::default();
        let budget_manager = TokenBudget::new(config);

        // Check budget for non-existent user (should auto-create)
        let result = budget_manager
            .check_budget(&"new_user".to_string(), 1000)
            .await;
        assert!(result.is_ok());

        // Verify budget was created
        let stats = budget_manager
            .get_usage_stats(&"new_user".to_string())
            .await
            .unwrap();
        assert_eq!(stats.monthly_limit, 100_000); // Default limit
    }

    #[tokio::test]
    async fn test_all_usage_stats() {
        let config = BudgetConfig::default();
        let budget_manager = TokenBudget::new(config);

        budget_manager
            .create_user_budget("user1".to_string(), 10000)
            .await
            .unwrap();
        budget_manager
            .create_user_budget("user2".to_string(), 20000)
            .await
            .unwrap();

        budget_manager
            .record_usage(&"user1".to_string(), 5000)
            .await
            .unwrap();
        budget_manager
            .record_usage(&"user2".to_string(), 10000)
            .await
            .unwrap();

        let all_stats = budget_manager.get_all_usage_stats().await;
        assert_eq!(all_stats.len(), 2);
    }
}
