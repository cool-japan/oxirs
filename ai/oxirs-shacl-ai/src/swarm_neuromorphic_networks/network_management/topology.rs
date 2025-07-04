//! Network topology management

use super::super::config::SwarmNetworkConfig;

#[derive(Debug)]
pub struct NetworkTopologyManager;

impl NetworkTopologyManager {
    pub fn new(_config: &SwarmNetworkConfig) -> Self {
        Self
    }
}