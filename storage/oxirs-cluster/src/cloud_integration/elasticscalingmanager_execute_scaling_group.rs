//! # ElasticScalingManager - execute_scaling_group Methods
//!
//! This module contains method implementations for `ElasticScalingManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::types::{CloudError, NodeInstance, ScalingDecision, ScalingEvent};

use super::elasticscalingmanager_type::ElasticScalingManager;

impl ElasticScalingManager {
    /// Execute scaling decision
    pub async fn execute_scaling(&self, decision: ScalingDecision) -> Result<(), CloudError> {
        let start = Instant::now();
        match &decision {
            ScalingDecision::ScaleUp {
                count,
                instance_type,
                use_spot,
                reason,
            } => {
                info!(
                    "Scaling up by {} nodes (type: {}, spot: {}) - {}",
                    count, instance_type, use_spot, reason
                );
                let mut nodes = self.current_nodes.write().await;
                for i in 0..*count {
                    let instance = NodeInstance {
                        instance_id: format!("i-{}", uuid::Uuid::new_v4()),
                        node_id: (nodes.len() + i as usize) as u64 + 1,
                        instance_type: instance_type.clone(),
                        is_spot: *use_spot,
                        launch_time: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .expect("system time should be after UNIX_EPOCH")
                            .as_secs(),
                        cpu_utilization: 0.0,
                        memory_utilization: 0.0,
                        provider: self.config.provider,
                        region: "us-east-1".to_string(),
                    };
                    nodes.push(instance);
                }
                *self.last_scaling_time.write().await = Instant::now();
            }
            ScalingDecision::ScaleDown {
                count,
                instance_ids,
                reason,
            } => {
                info!("Scaling down by {} nodes - {}", count, reason);
                let mut nodes = self.current_nodes.write().await;
                nodes.retain(|n| !instance_ids.contains(&n.instance_id));
                *self.last_scaling_time.write().await = Instant::now();
            }
            ScalingDecision::NoAction { reason } => {
                info!("No scaling action: {}", reason);
            }
        }
        let duration = start.elapsed().as_millis() as u64;
        let event = ScalingEvent {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time should be after UNIX_EPOCH")
                .as_secs(),
            decision,
            success: true,
            duration_ms: duration,
            error: None,
        };
        let mut events = self.scaling_events.write().await;
        events.push_back(event);
        while events.len() > 1000 {
            events.pop_front();
        }
        Ok(())
    }
}
