//! # Visual Designer — Tests

#[cfg(test)]
mod tests {
    use crate::visual_designer_engine::VisualStreamDesigner;
    use crate::visual_designer_types::{
        DebuggerConfig, ExportFormat, NodeType, Position, SinkType, SourceType,
        VisualDesignerConfig,
    };

    #[tokio::test]
    async fn test_create_pipeline() {
        let designer = VisualStreamDesigner::new(VisualDesignerConfig::default());
        let pipeline_id = designer
            .create_pipeline(
                "Test Pipeline".to_string(),
                Some("Test description".to_string()),
            )
            .await
            .unwrap();

        assert!(!pipeline_id.is_empty());

        let pipelines = designer.list_pipelines().await;
        assert_eq!(pipelines.len(), 1);
        assert_eq!(pipelines[0].name, "Test Pipeline");
    }

    #[tokio::test]
    async fn test_add_nodes() {
        let designer = VisualStreamDesigner::new(VisualDesignerConfig::default());
        let pipeline_id = designer
            .create_pipeline("Test".to_string(), None)
            .await
            .unwrap();

        let source_id = designer
            .add_node(
                &pipeline_id,
                "Source".to_string(),
                NodeType::Source(SourceType::Memory),
                Position {
                    x: 0.0,
                    y: 0.0,
                    z: None,
                },
            )
            .await
            .unwrap();

        let sink_id = designer
            .add_node(
                &pipeline_id,
                "Sink".to_string(),
                NodeType::Sink(SinkType::Memory),
                Position {
                    x: 100.0,
                    y: 0.0,
                    z: None,
                },
            )
            .await
            .unwrap();

        assert!(!source_id.is_empty());
        assert!(!sink_id.is_empty());

        let pipeline = designer.get_pipeline(&pipeline_id).await.unwrap();
        assert_eq!(pipeline.nodes.len(), 2);
    }

    #[tokio::test]
    async fn test_add_edge() {
        let designer = VisualStreamDesigner::new(VisualDesignerConfig::default());
        let pipeline_id = designer
            .create_pipeline("Test".to_string(), None)
            .await
            .unwrap();

        let source_id = designer
            .add_node(
                &pipeline_id,
                "Source".to_string(),
                NodeType::Source(SourceType::Memory),
                Position {
                    x: 0.0,
                    y: 0.0,
                    z: None,
                },
            )
            .await
            .unwrap();

        let sink_id = designer
            .add_node(
                &pipeline_id,
                "Sink".to_string(),
                NodeType::Sink(SinkType::Memory),
                Position {
                    x: 100.0,
                    y: 0.0,
                    z: None,
                },
            )
            .await
            .unwrap();

        let edge_id = designer
            .add_edge(
                &pipeline_id,
                source_id,
                "output".to_string(),
                sink_id,
                "input".to_string(),
            )
            .await
            .unwrap();

        assert!(!edge_id.is_empty());

        let pipeline = designer.get_pipeline(&pipeline_id).await.unwrap();
        assert_eq!(pipeline.edges.len(), 1);
    }

    #[tokio::test]
    async fn test_validate_pipeline() {
        let designer = VisualStreamDesigner::new(VisualDesignerConfig::default());
        let pipeline_id = designer
            .create_pipeline("Test".to_string(), None)
            .await
            .unwrap();

        let source_id = designer
            .add_node(
                &pipeline_id,
                "Source".to_string(),
                NodeType::Source(SourceType::Memory),
                Position {
                    x: 0.0,
                    y: 0.0,
                    z: None,
                },
            )
            .await
            .unwrap();

        let sink_id = designer
            .add_node(
                &pipeline_id,
                "Sink".to_string(),
                NodeType::Sink(SinkType::Memory),
                Position {
                    x: 100.0,
                    y: 0.0,
                    z: None,
                },
            )
            .await
            .unwrap();

        designer
            .add_edge(
                &pipeline_id,
                source_id,
                "output".to_string(),
                sink_id,
                "input".to_string(),
            )
            .await
            .unwrap();

        let validation = designer.validate_pipeline(&pipeline_id).await.unwrap();
        assert!(validation.is_valid);
        assert!(validation.errors.is_empty());
    }

    #[tokio::test]
    async fn test_export_import_json() {
        let designer = VisualStreamDesigner::new(VisualDesignerConfig::default());
        let pipeline_id = designer
            .create_pipeline("Test".to_string(), None)
            .await
            .unwrap();

        designer
            .add_node(
                &pipeline_id,
                "Source".to_string(),
                NodeType::Source(SourceType::Memory),
                Position {
                    x: 0.0,
                    y: 0.0,
                    z: None,
                },
            )
            .await
            .unwrap();

        let json = designer
            .export_pipeline(&pipeline_id, ExportFormat::Json)
            .await
            .unwrap();

        assert!(!json.is_empty());

        let new_id = designer
            .import_pipeline(&json, ExportFormat::Json)
            .await
            .unwrap();

        assert!(!new_id.is_empty());

        let imported = designer.get_pipeline(&new_id).await.unwrap();
        assert_eq!(imported.nodes.len(), 1);
    }

    #[tokio::test]
    async fn test_export_dot() {
        let designer = VisualStreamDesigner::new(VisualDesignerConfig::default());
        let pipeline_id = designer
            .create_pipeline("Test".to_string(), None)
            .await
            .unwrap();

        let source_id = designer
            .add_node(
                &pipeline_id,
                "Source".to_string(),
                NodeType::Source(SourceType::Memory),
                Position {
                    x: 0.0,
                    y: 0.0,
                    z: None,
                },
            )
            .await
            .unwrap();

        let sink_id = designer
            .add_node(
                &pipeline_id,
                "Sink".to_string(),
                NodeType::Sink(SinkType::Memory),
                Position {
                    x: 100.0,
                    y: 0.0,
                    z: None,
                },
            )
            .await
            .unwrap();

        designer
            .add_edge(
                &pipeline_id,
                source_id,
                "output".to_string(),
                sink_id,
                "input".to_string(),
            )
            .await
            .unwrap();

        let dot = designer
            .export_pipeline(&pipeline_id, ExportFormat::Dot)
            .await
            .unwrap();

        assert!(dot.contains("digraph"));
        assert!(dot.contains("Source"));
        assert!(dot.contains("Sink"));
        assert!(dot.contains("->"));
    }

    #[tokio::test]
    async fn test_create_debugger() {
        let designer = VisualStreamDesigner::new(VisualDesignerConfig::default());
        let pipeline_id = designer
            .create_pipeline("Test".to_string(), None)
            .await
            .unwrap();

        let debugger_id = designer
            .create_debugger(&pipeline_id, DebuggerConfig::default())
            .await
            .unwrap();

        assert!(!debugger_id.is_empty());

        let state = designer.get_debugger_state(&debugger_id).await.unwrap();
        assert!(!state.is_running);
        assert!(!state.is_paused);
    }

    #[tokio::test]
    async fn test_add_breakpoint() {
        let designer = VisualStreamDesigner::new(VisualDesignerConfig::default());
        let pipeline_id = designer
            .create_pipeline("Test".to_string(), None)
            .await
            .unwrap();

        let node_id = designer
            .add_node(
                &pipeline_id,
                "Filter".to_string(),
                NodeType::Filter,
                Position {
                    x: 0.0,
                    y: 0.0,
                    z: None,
                },
            )
            .await
            .unwrap();

        let debugger_id = designer
            .create_debugger(&pipeline_id, DebuggerConfig::default())
            .await
            .unwrap();

        let breakpoint_id = designer
            .add_breakpoint(&debugger_id, node_id, None)
            .await
            .unwrap();

        assert!(!breakpoint_id.is_empty());
    }

    #[tokio::test]
    async fn test_optimize_pipeline() {
        let designer = VisualStreamDesigner::new(VisualDesignerConfig::default());
        let pipeline_id = designer
            .create_pipeline("Test".to_string(), None)
            .await
            .unwrap();

        // Create a chain of nodes
        let mut prev_id = None;
        for i in 0..12 {
            let node_id = designer
                .add_node(
                    &pipeline_id,
                    format!("Node{}", i),
                    NodeType::Map,
                    Position {
                        x: i as f64 * 100.0,
                        y: 0.0,
                        z: None,
                    },
                )
                .await
                .unwrap();

            if let Some(prev) = prev_id {
                designer
                    .add_edge(
                        &pipeline_id,
                        prev,
                        "output".to_string(),
                        node_id.clone(),
                        "input".to_string(),
                    )
                    .await
                    .unwrap();
            }

            prev_id = Some(node_id);
        }

        let optimization = designer.optimize_pipeline(&pipeline_id).await.unwrap();
        assert!(!optimization.suggestions.is_empty());
    }

    #[tokio::test]
    async fn test_delete_pipeline() {
        let designer = VisualStreamDesigner::new(VisualDesignerConfig::default());
        let pipeline_id = designer
            .create_pipeline("Test".to_string(), None)
            .await
            .unwrap();

        assert_eq!(designer.list_pipelines().await.len(), 1);

        designer.delete_pipeline(&pipeline_id).await.unwrap();
        assert_eq!(designer.list_pipelines().await.len(), 0);
    }

    #[tokio::test]
    async fn test_export_svg_contains_svg_tag() {
        let designer = VisualStreamDesigner::new(VisualDesignerConfig::default());
        let pipeline_id = designer
            .create_pipeline("SVG Test Pipeline".to_string(), None)
            .await
            .unwrap();

        let source_id = designer
            .add_node(
                &pipeline_id,
                "Source".to_string(),
                NodeType::Source(SourceType::Memory),
                Position {
                    x: 0.0,
                    y: 0.0,
                    z: None,
                },
            )
            .await
            .unwrap();

        let sink_id = designer
            .add_node(
                &pipeline_id,
                "Sink".to_string(),
                NodeType::Sink(SinkType::Memory),
                Position {
                    x: 200.0,
                    y: 0.0,
                    z: None,
                },
            )
            .await
            .unwrap();

        designer
            .add_edge(
                &pipeline_id,
                source_id,
                "output".to_string(),
                sink_id,
                "input".to_string(),
            )
            .await
            .unwrap();

        let svg = designer
            .export_pipeline(&pipeline_id, ExportFormat::Svg)
            .await
            .expect("SVG export should succeed");

        assert!(
            svg.contains("<svg"),
            "SVG output must start with <svg element"
        );
        assert!(svg.contains("</svg>"), "SVG output must be closed");
        assert!(
            svg.contains("Source"),
            "SVG must contain node label 'Source'"
        );
        assert!(svg.contains("Sink"), "SVG must contain node label 'Sink'");
    }

    #[tokio::test]
    async fn test_export_png_returns_descriptive_error() {
        let designer = VisualStreamDesigner::new(VisualDesignerConfig::default());
        let pipeline_id = designer
            .create_pipeline("PNG Test".to_string(), None)
            .await
            .unwrap();

        let result = designer
            .export_pipeline(&pipeline_id, ExportFormat::Png)
            .await;

        assert!(
            result.is_err(),
            "PNG export should return an error per policy"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("PNG"),
            "Error message should mention PNG: {err_msg}"
        );
    }
}
