//! Billing and usage metering for multi-tenancy

use crate::multi_tenancy::types::{MultiTenancyError, MultiTenancyResult, TenantOperation};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Billing period for charges
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BillingPeriod {
    Hourly,
    Daily,
    Monthly,
    Annual,
}

impl BillingPeriod {
    /// Get duration in seconds
    pub fn duration_secs(&self) -> i64 {
        match self {
            Self::Hourly => 3600,
            Self::Daily => 86400,
            Self::Monthly => 2592000, // 30 days
            Self::Annual => 31536000, // 365 days
        }
    }
}

/// Pricing model for billing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PricingModel {
    /// Pay per request
    PerRequest {
        /// Cost per request
        cost_per_request: f64,
    },
    /// Pay per vector stored
    PerVector {
        /// Cost per 1000 vectors per month
        cost_per_1k_vectors: f64,
    },
    /// Pay per storage GB
    PerStorage {
        /// Cost per GB per month
        cost_per_gb: f64,
    },
    /// Pay per compute unit
    PerComputeUnit {
        /// Cost per compute unit (query complexity weighted)
        cost_per_unit: f64,
    },
    /// Flat subscription
    Subscription {
        /// Monthly subscription fee
        monthly_fee: f64,
        /// Included requests
        included_requests: u64,
        /// Overage cost per request
        overage_cost: f64,
    },
    /// Custom pricing
    Custom {
        /// Base fee
        base_fee: f64,
        /// Operation costs
        operation_costs: HashMap<String, f64>,
    },
}

impl PricingModel {
    /// Calculate cost for an operation
    pub fn calculate_cost(&self, operation: TenantOperation, count: u64) -> f64 {
        match self {
            Self::PerRequest { cost_per_request } => *cost_per_request * count as f64,
            Self::PerComputeUnit { cost_per_unit } => {
                *cost_per_unit * operation.default_cost_weight() * count as f64
            }
            Self::Custom {
                operation_costs, ..
            } => {
                let op_cost = operation_costs
                    .get(operation.name())
                    .copied()
                    .unwrap_or(0.01);
                op_cost * count as f64
            }
            _ => 0.0, // Other models calculated differently
        }
    }
}

/// Usage record for billing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    /// Tenant ID
    pub tenant_id: String,
    /// Operation type
    pub operation: TenantOperation,
    /// Number of operations
    pub count: u64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Cost (computed)
    pub cost: f64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl UsageRecord {
    /// Create new usage record
    pub fn new(tenant_id: impl Into<String>, operation: TenantOperation, count: u64) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            operation,
            count,
            timestamp: Utc::now(),
            cost: 0.0,
            metadata: HashMap::new(),
        }
    }

    /// Calculate cost using pricing model
    pub fn calculate_cost(&mut self, pricing: &PricingModel) {
        self.cost = pricing.calculate_cost(self.operation, self.count);
    }
}

/// Billing metrics for a tenant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingMetrics {
    /// Tenant ID
    pub tenant_id: String,

    /// Current billing period start
    pub period_start: DateTime<Utc>,

    /// Current billing period end
    pub period_end: DateTime<Utc>,

    /// Total cost for current period
    pub total_cost: f64,

    /// Total requests in period
    pub total_requests: u64,

    /// Average request cost
    pub avg_request_cost: f64,

    /// Cost by operation type
    pub cost_by_operation: HashMap<String, f64>,

    /// Request count by operation
    pub requests_by_operation: HashMap<String, u64>,

    /// Peak daily cost
    pub peak_daily_cost: f64,

    /// Estimated monthly cost (projected)
    pub estimated_monthly_cost: f64,
}

impl BillingMetrics {
    /// Create new billing metrics
    pub fn new(tenant_id: impl Into<String>, period: BillingPeriod) -> Self {
        let now = Utc::now();
        let period_end = now + Duration::seconds(period.duration_secs());

        Self {
            tenant_id: tenant_id.into(),
            period_start: now,
            period_end,
            total_cost: 0.0,
            total_requests: 0,
            avg_request_cost: 0.0,
            cost_by_operation: HashMap::new(),
            requests_by_operation: HashMap::new(),
            peak_daily_cost: 0.0,
            estimated_monthly_cost: 0.0,
        }
    }

    /// Record usage
    pub fn record_usage(&mut self, record: &UsageRecord) {
        self.total_cost += record.cost;
        self.total_requests += record.count;

        let op_name = record.operation.name().to_string();
        *self.cost_by_operation.entry(op_name.clone()).or_insert(0.0) += record.cost;
        *self.requests_by_operation.entry(op_name).or_insert(0) += record.count;

        // Update average
        if self.total_requests > 0 {
            self.avg_request_cost = self.total_cost / self.total_requests as f64;
        }

        // Update estimated monthly cost
        let elapsed_secs = (Utc::now() - self.period_start).num_seconds() as f64;
        if elapsed_secs > 0.0 {
            let monthly_secs = 2592000.0; // 30 days
            self.estimated_monthly_cost = self.total_cost * (monthly_secs / elapsed_secs);
        }
    }

    /// Reset for new billing period
    pub fn reset(&mut self, period: BillingPeriod) {
        self.period_start = Utc::now();
        self.period_end = self.period_start + Duration::seconds(period.duration_secs());
        self.total_cost = 0.0;
        self.total_requests = 0;
        self.avg_request_cost = 0.0;
        self.cost_by_operation.clear();
        self.requests_by_operation.clear();
    }
}

/// Billing engine for multi-tenancy
pub struct BillingEngine {
    /// Pricing models by tenant
    pricing: Arc<Mutex<HashMap<String, PricingModel>>>,

    /// Usage records
    usage_history: Arc<Mutex<Vec<UsageRecord>>>,

    /// Current billing metrics by tenant
    metrics: Arc<Mutex<HashMap<String, BillingMetrics>>>,

    /// Billing period
    period: BillingPeriod,
}

impl BillingEngine {
    /// Create new billing engine
    pub fn new(period: BillingPeriod) -> Self {
        Self {
            pricing: Arc::new(Mutex::new(HashMap::new())),
            usage_history: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(Mutex::new(HashMap::new())),
            period,
        }
    }

    /// Set pricing model for tenant
    pub fn set_pricing(
        &self,
        tenant_id: impl Into<String>,
        pricing: PricingModel,
    ) -> MultiTenancyResult<()> {
        let tenant_id = tenant_id.into();

        self.pricing
            .lock()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?
            .insert(tenant_id.clone(), pricing);

        // Initialize metrics
        self.metrics
            .lock()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?
            .entry(tenant_id.clone())
            .or_insert_with(|| BillingMetrics::new(tenant_id, self.period));

        Ok(())
    }

    /// Record usage for tenant
    pub fn record_usage(
        &self,
        tenant_id: &str,
        operation: TenantOperation,
        count: u64,
    ) -> MultiTenancyResult<f64> {
        let mut record = UsageRecord::new(tenant_id, operation, count);

        // Calculate cost
        let pricing = self
            .pricing
            .lock()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?
            .get(tenant_id)
            .cloned()
            .ok_or_else(|| MultiTenancyError::BillingError {
                message: format!("No pricing model for tenant: {}", tenant_id),
            })?;

        record.calculate_cost(&pricing);
        let cost = record.cost;

        // Update metrics
        let mut metrics = self
            .metrics
            .lock()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?;

        metrics
            .entry(tenant_id.to_string())
            .or_insert_with(|| BillingMetrics::new(tenant_id, self.period))
            .record_usage(&record);

        // Store record
        self.usage_history
            .lock()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?
            .push(record);

        Ok(cost)
    }

    /// Get billing metrics for tenant
    pub fn get_metrics(&self, tenant_id: &str) -> MultiTenancyResult<BillingMetrics> {
        self.metrics
            .lock()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?
            .get(tenant_id)
            .cloned()
            .ok_or_else(|| MultiTenancyError::TenantNotFound {
                tenant_id: tenant_id.to_string(),
            })
    }

    /// Get usage history for tenant
    pub fn get_usage_history(
        &self,
        tenant_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MultiTenancyResult<Vec<UsageRecord>> {
        let history = self
            .usage_history
            .lock()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?;

        Ok(history
            .iter()
            .filter(|r| r.tenant_id == tenant_id && r.timestamp >= start && r.timestamp <= end)
            .cloned()
            .collect())
    }

    /// Reset billing period for tenant
    pub fn reset_period(&self, tenant_id: &str) -> MultiTenancyResult<()> {
        let mut metrics = self
            .metrics
            .lock()
            .map_err(|e| MultiTenancyError::InternalError {
                message: format!("Lock error: {}", e),
            })?;

        metrics
            .get_mut(tenant_id)
            .ok_or_else(|| MultiTenancyError::TenantNotFound {
                tenant_id: tenant_id.to_string(),
            })?
            .reset(self.period);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_billing_period() {
        assert_eq!(BillingPeriod::Hourly.duration_secs(), 3600);
        assert_eq!(BillingPeriod::Daily.duration_secs(), 86400);
        assert_eq!(BillingPeriod::Monthly.duration_secs(), 2592000);
    }

    #[test]
    fn test_pricing_models() {
        let model = PricingModel::PerRequest {
            cost_per_request: 0.01,
        };
        assert_eq!(
            model.calculate_cost(TenantOperation::VectorSearch, 100),
            1.0
        );

        let model = PricingModel::PerComputeUnit { cost_per_unit: 0.1 };
        let cost = model.calculate_cost(TenantOperation::IndexBuild, 1);
        assert!(cost > 0.0); // Should be weighted by operation complexity
    }

    #[test]
    fn test_usage_record() {
        let mut record = UsageRecord::new("tenant1", TenantOperation::VectorSearch, 100);
        assert_eq!(record.count, 100);
        assert_eq!(record.cost, 0.0);

        let pricing = PricingModel::PerRequest {
            cost_per_request: 0.01,
        };
        record.calculate_cost(&pricing);
        assert_eq!(record.cost, 1.0);
    }

    #[test]
    fn test_billing_metrics() {
        let mut metrics = BillingMetrics::new("tenant1", BillingPeriod::Daily);
        assert_eq!(metrics.total_cost, 0.0);
        assert_eq!(metrics.total_requests, 0);

        let mut record = UsageRecord::new("tenant1", TenantOperation::VectorSearch, 100);
        record.cost = 1.0;
        metrics.record_usage(&record);

        assert_eq!(metrics.total_cost, 1.0);
        assert_eq!(metrics.total_requests, 100);
        assert!((metrics.avg_request_cost - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_billing_engine() {
        let engine = BillingEngine::new(BillingPeriod::Daily);

        // Set pricing
        let pricing = PricingModel::PerRequest {
            cost_per_request: 0.01,
        };
        engine.set_pricing("tenant1", pricing).unwrap();

        // Record usage
        let cost = engine
            .record_usage("tenant1", TenantOperation::VectorSearch, 100)
            .unwrap();
        assert_eq!(cost, 1.0);

        // Get metrics
        let metrics = engine.get_metrics("tenant1").unwrap();
        assert_eq!(metrics.total_cost, 1.0);
        assert_eq!(metrics.total_requests, 100);

        // Record more usage
        engine
            .record_usage("tenant1", TenantOperation::VectorInsert, 50)
            .unwrap();

        let metrics = engine.get_metrics("tenant1").unwrap();
        assert_eq!(metrics.total_cost, 1.5);
        assert_eq!(metrics.total_requests, 150);
    }

    #[test]
    fn test_usage_history() {
        let engine = BillingEngine::new(BillingPeriod::Daily);

        let pricing = PricingModel::PerRequest {
            cost_per_request: 0.01,
        };
        engine.set_pricing("tenant1", pricing).unwrap();

        // Record some usage
        engine
            .record_usage("tenant1", TenantOperation::VectorSearch, 100)
            .unwrap();
        engine
            .record_usage("tenant1", TenantOperation::VectorInsert, 50)
            .unwrap();

        // Get history
        let start = Utc::now() - Duration::hours(1);
        let end = Utc::now() + Duration::hours(1);
        let history = engine.get_usage_history("tenant1", start, end).unwrap();

        assert_eq!(history.len(), 2);
        assert_eq!(history[0].count, 100);
        assert_eq!(history[1].count, 50);
    }

    #[test]
    fn test_subscription_pricing() {
        let pricing = PricingModel::Subscription {
            monthly_fee: 100.0,
            included_requests: 10000,
            overage_cost: 0.02,
        };

        // Subscription pricing is handled differently, just test structure
        match pricing {
            PricingModel::Subscription {
                monthly_fee,
                included_requests,
                overage_cost,
            } => {
                assert_eq!(monthly_fee, 100.0);
                assert_eq!(included_requests, 10000);
                assert_eq!(overage_cost, 0.02);
            }
            _ => panic!("Expected subscription pricing"),
        }
    }
}
