//! Advanced Visualization Tools for SHACL-AI
//!
//! This module provides comprehensive visualization capabilities for neural architectures,
//! quantum patterns, federated learning networks, and adaptive AI performance monitoring.
//!
//! The module is organized into several sub-modules:
//! - `core`: Core visualization engine and configuration types
//! - `collectors`: Data collectors for different visualization sources
//! - `renderers`: Visualization renderers for different output formats
//! - `exporters`: Export functionality for various file formats

pub mod collectors;
pub mod core;
pub mod exporters;
pub mod renderers;

// Re-export main types for easy access
pub use core::{
    ActiveVisualization, AdvancedVisualizationEngine, ArchitectureVisualizationType, AttentionFlow,
    ColorScheme, DashboardLayout, EntanglementVisualization, ExplorerNode, ExportFormat,
    ExportQuality, ExportResult, GraphEdge, GraphNode, HeatmapLabels, InteractiveControls,
    InterpretabilityExplanation, PerformanceDebugInfo, QuantumClassicalConnection,
    QuantumStateVisualization, QuantumVisualizationMode, RenderOptions, TimeSeriesData,
    VisualizationConfig, VisualizationData, VisualizationMetadata, VisualizationOutput,
    VisualizationType,
};

pub use collectors::{
    CollectorMetadata, DataCollector, FederatedNetworkCollector, NeuralArchitectureCollector,
    PerformanceMetricsCollector, QuantumPatternCollector,
};

pub use renderers::{
    Graph3DRenderer, HeatmapRenderer, InterpretabilityRenderer, NetworkTopologyRenderer,
    QuantumStateRenderer, RendererCapabilities, TimelineRenderer, VisualizationRenderer,
};

pub use exporters::{
    ExportManager, Exporter, HTMLExporter, JSONExporter, PDFExporter, PNGExporter, SVGExporter,
};

use crate::Result;

/// Create a default advanced visualization engine
pub fn create_default_engine() -> AdvancedVisualizationEngine {
    let config = VisualizationConfig::default();
    AdvancedVisualizationEngine::new(config)
}

/// Create an advanced visualization engine with custom configuration
pub fn create_engine_with_config(config: VisualizationConfig) -> AdvancedVisualizationEngine {
    AdvancedVisualizationEngine::new(config)
}

/// Register all default data collectors with an engine
pub async fn register_default_collectors(engine: &AdvancedVisualizationEngine) -> Result<()> {
    engine
        .register_data_collector(
            "neural_architecture".to_string(),
            Box::new(NeuralArchitectureCollector::new()),
        )
        .await?;

    engine
        .register_data_collector(
            "quantum_patterns".to_string(),
            Box::new(QuantumPatternCollector::new()),
        )
        .await?;

    engine
        .register_data_collector(
            "federated_network".to_string(),
            Box::new(FederatedNetworkCollector::new()),
        )
        .await?;

    engine
        .register_data_collector(
            "performance_metrics".to_string(),
            Box::new(PerformanceMetricsCollector::new()),
        )
        .await?;

    tracing::info!("Registered default data collectors");
    Ok(())
}

/// Register all default renderers with an engine
pub async fn register_default_renderers(engine: &AdvancedVisualizationEngine) -> Result<()> {
    engine
        .register_renderer("graph_3d".to_string(), Box::new(Graph3DRenderer::new()))
        .await?;

    engine
        .register_renderer("heatmap".to_string(), Box::new(HeatmapRenderer::new()))
        .await?;

    engine
        .register_renderer("timeline".to_string(), Box::new(TimelineRenderer::new()))
        .await?;

    engine
        .register_renderer(
            "network_topology".to_string(),
            Box::new(NetworkTopologyRenderer::new()),
        )
        .await?;

    engine
        .register_renderer(
            "quantum_state".to_string(),
            Box::new(QuantumStateRenderer::new()),
        )
        .await?;

    engine
        .register_renderer(
            "interpretability".to_string(),
            Box::new(InterpretabilityRenderer::new()),
        )
        .await?;

    tracing::info!("Registered default renderers");
    Ok(())
}

/// Create a fully configured visualization engine with all defaults
pub async fn create_full_engine() -> Result<AdvancedVisualizationEngine> {
    let engine = create_default_engine();
    register_default_collectors(&engine).await?;
    register_default_renderers(&engine).await?;
    Ok(engine)
}

/// Utility function to create a quick visualization
pub async fn quick_visualization(
    engine: &AdvancedVisualizationEngine,
    collector_name: &str,
    renderer_name: &str,
    _title: &str,
) -> Result<VisualizationOutput> {
    let options = RenderOptions {
        visualization_type: VisualizationType::NetworkGraph,
        color_scheme: ColorScheme::Viridis,
        interactive: true,
        animation: true,
    };

    let viz_id = format!("quick_viz_{}", chrono::Utc::now().timestamp());

    engine
        .create_visualization(
            viz_id,
            collector_name.to_string(),
            renderer_name.to_string(),
            options,
        )
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = create_default_engine();
        assert_eq!(engine.config().max_data_points, 10000);
        assert!(engine.config().enable_animations);
    }

    #[test]
    fn test_custom_config() {
        let config = VisualizationConfig {
            max_data_points: 5000,
            enable_animations: false,
            ..Default::default()
        };

        let engine = create_engine_with_config(config);
        assert_eq!(engine.config().max_data_points, 5000);
        assert!(!engine.config().enable_animations);
    }

    #[tokio::test]
    async fn test_collector_registration() {
        let engine = create_default_engine();
        let result = register_default_collectors(&engine).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_renderer_registration() {
        let engine = create_default_engine();
        let result = register_default_renderers(&engine).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_full_engine_creation() {
        let result = create_full_engine().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_color_schemes() {
        let schemes = [
            ColorScheme::Viridis,
            ColorScheme::Plasma,
            ColorScheme::Inferno,
            ColorScheme::Custom(vec!["#ff0000".to_string(), "#00ff00".to_string()]),
        ];

        assert_eq!(schemes.len(), 4);
    }

    #[test]
    fn test_visualization_types() {
        let types = [
            VisualizationType::NetworkGraph,
            VisualizationType::Heatmap,
            VisualizationType::Timeline,
            VisualizationType::QuantumClassicalHybrid,
        ];

        assert_eq!(types.len(), 4);
    }

    #[test]
    fn test_export_formats() {
        let formats = [
            ExportFormat::SVG,
            ExportFormat::PNG,
            ExportFormat::HTML,
            ExportFormat::PDF,
            ExportFormat::JSON,
        ];

        assert_eq!(formats.len(), 5);
    }

    #[test]
    fn test_render_options_default() {
        let options = RenderOptions::default();
        assert!(options.interactive);
        assert!(options.animation);
        assert!(matches!(options.color_scheme, ColorScheme::Viridis));
    }

    #[test]
    fn test_export_manager() {
        let manager = ExportManager::new();
        let formats = manager.available_formats();
        assert!(!formats.is_empty());
        assert!(formats.contains(&ExportFormat::SVG));
        assert!(formats.contains(&ExportFormat::PNG));
    }
}
