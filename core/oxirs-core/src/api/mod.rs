//! # API Integration Layer
//! 
//! This module provides comprehensive API integration capabilities for OxiRS,
//! including GraphQL schema generation, REST API, gRPC support, and WebSocket streaming.

pub mod graphql_schema;
pub mod rest_api; 
pub mod grpc_service;
pub mod websocket_streaming;
pub mod kafka_integration;
pub mod openapi_spec;

use crate::error::Result;
use crate::model::{Dataset, Graph, Triple, Quad};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main API service coordinator that manages all API endpoints
pub struct ApiService {
    /// Core dataset for API operations
    dataset: Arc<RwLock<Dataset>>,
    /// GraphQL schema generator
    graphql_schema: graphql_schema::GraphQLSchemaGenerator,
    /// REST API handler  
    rest_api: rest_api::RestApiHandler,
    /// gRPC service
    grpc_service: grpc_service::GrpcService,
    /// WebSocket streaming manager
    websocket_streaming: websocket_streaming::WebSocketManager,
    /// Kafka integration
    kafka_integration: kafka_integration::KafkaIntegration,
}

impl ApiService {
    /// Create a new API service instance
    pub fn new(dataset: Arc<RwLock<Dataset>>) -> Result<Self> {
        Ok(Self {
            graphql_schema: graphql_schema::GraphQLSchemaGenerator::new(dataset.clone())?,
            rest_api: rest_api::RestApiHandler::new(dataset.clone())?,
            grpc_service: grpc_service::GrpcService::new(dataset.clone())?,
            websocket_streaming: websocket_streaming::WebSocketManager::new(dataset.clone())?,
            kafka_integration: kafka_integration::KafkaIntegration::new(dataset.clone())?,
            dataset,
        })
    }
    
    /// Start all API services
    pub async fn start(&self) -> Result<()> {
        // Start GraphQL endpoint
        self.graphql_schema.initialize().await?;
        
        // Start REST API server
        self.rest_api.start_server().await?;
        
        // Start gRPC server
        self.grpc_service.start_server().await?;
        
        // Initialize WebSocket streaming
        self.websocket_streaming.initialize().await?;
        
        // Connect to Kafka
        self.kafka_integration.connect().await?;
        
        Ok(())
    }
    
    /// Stop all API services gracefully
    pub async fn stop(&self) -> Result<()> {
        self.kafka_integration.disconnect().await?;
        self.websocket_streaming.shutdown().await?;
        self.grpc_service.stop_server().await?;
        self.rest_api.stop_server().await?;
        
        Ok(())
    }
    
    /// Get service health status
    pub async fn health_check(&self) -> ApiServiceHealth {
        ApiServiceHealth {
            graphql_status: self.graphql_schema.health_check().await,
            rest_api_status: self.rest_api.health_check().await,
            grpc_status: self.grpc_service.health_check().await,
            websocket_status: self.websocket_streaming.health_check().await,
            kafka_status: self.kafka_integration.health_check().await,
        }
    }
}

/// Health status for all API services
#[derive(Debug, Clone)]
pub struct ApiServiceHealth {
    pub graphql_status: ServiceStatus,
    pub rest_api_status: ServiceStatus,
    pub grpc_status: ServiceStatus,
    pub websocket_status: ServiceStatus,
    pub kafka_status: ServiceStatus,
}

/// Individual service status
#[derive(Debug, Clone, PartialEq)]
pub enum ServiceStatus {
    Healthy,
    Degraded(String),
    Unhealthy(String),
}

impl ApiServiceHealth {
    /// Check if all services are healthy
    pub fn is_healthy(&self) -> bool {
        matches!(self.graphql_status, ServiceStatus::Healthy) &&
        matches!(self.rest_api_status, ServiceStatus::Healthy) &&
        matches!(self.grpc_status, ServiceStatus::Healthy) &&
        matches!(self.websocket_status, ServiceStatus::Healthy) &&
        matches!(self.kafka_status, ServiceStatus::Healthy)
    }
}

/// Configuration for API services
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// GraphQL endpoint configuration
    pub graphql_config: graphql_schema::GraphQLConfig,
    /// REST API configuration
    pub rest_config: rest_api::RestConfig,
    /// gRPC service configuration
    pub grpc_config: grpc_service::GrpcConfig,
    /// WebSocket configuration
    pub websocket_config: websocket_streaming::WebSocketConfig,
    /// Kafka configuration
    pub kafka_config: kafka_integration::KafkaConfig,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            graphql_config: graphql_schema::GraphQLConfig::default(),
            rest_config: rest_api::RestConfig::default(),
            grpc_config: grpc_service::GrpcConfig::default(),
            websocket_config: websocket_streaming::WebSocketConfig::default(),
            kafka_config: kafka_integration::KafkaConfig::default(),
        }
    }
}