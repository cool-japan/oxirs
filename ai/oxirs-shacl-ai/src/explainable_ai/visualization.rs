//! Attention visualization and visual explanation components

use std::collections::HashMap;
use std::time::SystemTime;

use super::types::*;
use crate::{Result, ShaclAiError};

/// Attention visualization handler
#[derive(Debug)]
pub struct AttentionVisualizer {
    config: VisualizationConfig,
}

#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    pub output_format: OutputFormat,
    pub resolution: (u32, u32),
    pub color_scheme: ColorScheme,
    pub interactive: bool,
}

#[derive(Debug, Clone)]
pub enum OutputFormat {
    SVG,
    PNG,
    HTML,
    JSON,
}

#[derive(Debug, Clone)]
pub enum ColorScheme {
    Viridis,
    Plasma,
    Inferno,
    Grayscale,
    Custom(Vec<String>),
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            output_format: OutputFormat::SVG,
            resolution: (800, 600),
            color_scheme: ColorScheme::Viridis,
            interactive: true,
        }
    }
}

impl AttentionVisualizer {
    pub fn new() -> Self {
        Self {
            config: VisualizationConfig::default(),
        }
    }

    pub fn with_config(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Generate attention heatmap visualization
    pub async fn generate_attention_heatmap(
        &self,
        attention_weights: &[Vec<f64>],
        tokens: &[String],
    ) -> Result<VisualizationResult> {
        let mut heatmap_data = Vec::new();
        
        for (i, row) in attention_weights.iter().enumerate() {
            for (j, &weight) in row.iter().enumerate() {
                heatmap_data.push(HeatmapCell {
                    row: i,
                    col: j,
                    value: weight,
                    token_from: tokens.get(i).cloned().unwrap_or_default(),
                    token_to: tokens.get(j).cloned().unwrap_or_default(),
                });
            }
        }

        let visualization = match self.config.output_format {
            OutputFormat::SVG => self.generate_svg_heatmap(&heatmap_data).await?,
            OutputFormat::PNG => self.generate_png_heatmap(&heatmap_data).await?,
            OutputFormat::HTML => self.generate_html_heatmap(&heatmap_data).await?,
            OutputFormat::JSON => self.generate_json_heatmap(&heatmap_data).await?,
        };

        Ok(VisualizationResult {
            visualization_type: VisualizationType::AttentionHeatmap,
            content: visualization,
            metadata: VisualizationMetadata {
                format: self.config.output_format.clone(),
                dimensions: self.config.resolution,
                created_at: SystemTime::now(),
                interactive: self.config.interactive,
            },
        })
    }

    /// Generate feature importance bar chart
    pub async fn generate_feature_importance_chart(
        &self,
        importance_data: &FeatureImportanceAnalysis,
    ) -> Result<VisualizationResult> {
        let mut chart_data = Vec::new();
        
        for feature in &importance_data.features {
            chart_data.push(BarChartItem {
                label: feature.feature_name.clone(),
                value: feature.importance_score,
                confidence: Some(feature.confidence_interval),
                category: Some(feature.category.clone()),
            });
        }

        // Sort by importance score
        chart_data.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap_or(std::cmp::Ordering::Equal));

        let visualization = match self.config.output_format {
            OutputFormat::SVG => self.generate_svg_bar_chart(&chart_data).await?,
            OutputFormat::PNG => self.generate_png_bar_chart(&chart_data).await?,
            OutputFormat::HTML => self.generate_html_bar_chart(&chart_data).await?,
            OutputFormat::JSON => self.generate_json_bar_chart(&chart_data).await?,
        };

        Ok(VisualizationResult {
            visualization_type: VisualizationType::FeatureImportance,
            content: visualization,
            metadata: VisualizationMetadata {
                format: self.config.output_format.clone(),
                dimensions: self.config.resolution,
                created_at: SystemTime::now(),
                interactive: self.config.interactive,
            },
        })
    }

    /// Generate decision tree visualization
    pub async fn generate_decision_tree(
        &self,
        tree_data: &DecisionTreeData,
    ) -> Result<VisualizationResult> {
        let visualization = match self.config.output_format {
            OutputFormat::SVG => self.generate_svg_tree(tree_data).await?,
            OutputFormat::PNG => self.generate_png_tree(tree_data).await?,
            OutputFormat::HTML => self.generate_html_tree(tree_data).await?,
            OutputFormat::JSON => self.generate_json_tree(tree_data).await?,
        };

        Ok(VisualizationResult {
            visualization_type: VisualizationType::DecisionTree,
            content: visualization,
            metadata: VisualizationMetadata {
                format: self.config.output_format.clone(),
                dimensions: self.config.resolution,
                created_at: SystemTime::now(),
                interactive: self.config.interactive,
            },
        })
    }

    async fn generate_svg_heatmap(&self, data: &[HeatmapCell]) -> Result<String> {
        // Generate SVG heatmap - simplified implementation
        let mut svg = format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            self.config.resolution.0, self.config.resolution.1
        );

        let cell_size = 20;
        for cell in data {
            let color = self.value_to_color(cell.value);
            let x = cell.col * cell_size;
            let y = cell.row * cell_size;
            
            svg.push_str(&format!(
                r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" title="{:.3}"/>"#,
                x, y, cell_size, cell_size, color, cell.value
            ));
        }

        svg.push_str("</svg>");
        Ok(svg)
    }

    async fn generate_png_heatmap(&self, _data: &[HeatmapCell]) -> Result<String> {
        // PNG generation would require image processing library
        // Return placeholder for now
        Ok("PNG_DATA_PLACEHOLDER".to_string())
    }

    async fn generate_html_heatmap(&self, data: &[HeatmapCell]) -> Result<String> {
        let mut html = String::from(r#"
<!DOCTYPE html>
<html>
<head>
    <title>Attention Heatmap</title>
    <style>
        .heatmap { display: grid; gap: 2px; }
        .cell { width: 20px; height: 20px; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <div class="heatmap">
"#);

        for cell in data {
            let color = self.value_to_color(cell.value);
            html.push_str(&format!(
                r#"<div class="cell" style="background-color: {}" title="{:.3}"></div>"#,
                color, cell.value
            ));
        }

        html.push_str("</div></body></html>");
        Ok(html)
    }

    async fn generate_json_heatmap(&self, data: &[HeatmapCell]) -> Result<String> {
        let json_data = serde_json::json!({
            "type": "heatmap",
            "data": data,
            "config": {
                "width": self.config.resolution.0,
                "height": self.config.resolution.1,
                "colorScheme": format!("{:?}", self.config.color_scheme)
            }
        });
        Ok(json_data.to_string())
    }

    async fn generate_svg_bar_chart(&self, data: &[BarChartItem]) -> Result<String> {
        let mut svg = format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            self.config.resolution.0, self.config.resolution.1
        );

        let bar_height = 30;
        let bar_spacing = 40;
        let max_value = data.iter().map(|item| item.value).fold(0.0, f64::max);

        for (i, item) in data.iter().enumerate() {
            let bar_width = (item.value / max_value) * (self.config.resolution.0 as f64 * 0.8);
            let y = i * bar_spacing;
            
            svg.push_str(&format!(
                r#"<rect x="100" y="{}" width="{}" height="{}" fill="steelblue" title="{}: {:.3}"/>"#,
                y, bar_width as u32, bar_height, item.label, item.value
            ));
            
            svg.push_str(&format!(
                r#"<text x="10" y="{}" font-family="Arial" font-size="12">{}</text>"#,
                y + bar_height / 2, item.label
            ));
        }

        svg.push_str("</svg>");
        Ok(svg)
    }

    async fn generate_png_bar_chart(&self, _data: &[BarChartItem]) -> Result<String> {
        Ok("PNG_BAR_CHART_PLACEHOLDER".to_string())
    }

    async fn generate_html_bar_chart(&self, data: &[BarChartItem]) -> Result<String> {
        let mut html = String::from(r#"
<!DOCTYPE html>
<html>
<head>
    <title>Feature Importance</title>
    <style>
        .chart { margin: 20px; }
        .bar { margin: 5px 0; }
        .bar-label { display: inline-block; width: 150px; text-align: right; }
        .bar-visual { display: inline-block; background: steelblue; height: 20px; margin-left: 10px; }
    </style>
</head>
<body>
    <div class="chart">
"#);

        let max_value = data.iter().map(|item| item.value).fold(0.0, f64::max);
        
        for item in data {
            let width = (item.value / max_value) * 300.0;
            html.push_str(&format!(
                r#"<div class="bar">
                    <span class="bar-label">{}</span>
                    <div class="bar-visual" style="width: {}px;" title="{:.3}"></div>
                   </div>"#,
                item.label, width as u32, item.value
            ));
        }

        html.push_str("</div></body></html>");
        Ok(html)
    }

    async fn generate_json_bar_chart(&self, data: &[BarChartItem]) -> Result<String> {
        let json_data = serde_json::json!({
            "type": "bar_chart",
            "data": data,
            "config": {
                "width": self.config.resolution.0,
                "height": self.config.resolution.1
            }
        });
        Ok(json_data.to_string())
    }

    async fn generate_svg_tree(&self, tree_data: &DecisionTreeData) -> Result<String> {
        let mut svg = format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            self.config.resolution.0, self.config.resolution.1
        );

        // Simplified tree rendering
        self.render_tree_node(&mut svg, &tree_data.root, 400, 50, 0)?;
        
        svg.push_str("</svg>");
        Ok(svg)
    }

    fn render_tree_node(
        &self,
        svg: &mut String,
        node: &DecisionNode,
        x: u32,
        y: u32,
        depth: u32,
    ) -> Result<()> {
        // Draw node
        svg.push_str(&format!(
            r#"<circle cx="{}" cy="{}" r="20" fill="lightblue" stroke="black"/>"#,
            x, y
        ));
        
        // Draw node label
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-size="10">{}</text>"#,
            x, y + 5, node.feature.chars().take(3).collect::<String>()
        ));

        // Draw children (simplified - only showing structure)
        if depth < 3 {
            let child_y = y + 80;
            let left_x = x - 60;
            let right_x = x + 60;
            
            // Draw connections
            svg.push_str(&format!(
                r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black"/>"#,
                x, y + 20, left_x, child_y - 20
            ));
            svg.push_str(&format!(
                r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black"/>"#,
                x, y + 20, right_x, child_y - 20
            ));
            
            // Create dummy child nodes for demonstration
            let left_child = DecisionNode {
                feature: format!("L{}", depth),
                threshold: 0.5,
                value: Some(0.3),
                samples: 50,
                impurity: 0.2,
            };
            
            let right_child = DecisionNode {
                feature: format!("R{}", depth),
                threshold: 0.7,
                value: Some(0.8),
                samples: 30,
                impurity: 0.1,
            };
            
            self.render_tree_node(svg, &left_child, left_x, child_y, depth + 1)?;
            self.render_tree_node(svg, &right_child, right_x, child_y, depth + 1)?;
        }

        Ok(())
    }

    async fn generate_png_tree(&self, _tree_data: &DecisionTreeData) -> Result<String> {
        Ok("PNG_TREE_PLACEHOLDER".to_string())
    }

    async fn generate_html_tree(&self, tree_data: &DecisionTreeData) -> Result<String> {
        let html = format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>Decision Tree</title>
    <style>
        .tree {{ font-family: Arial, sans-serif; }}
        .node {{ border: 1px solid #ccc; padding: 10px; margin: 5px; background: #f9f9f9; }}
    </style>
</head>
<body>
    <div class="tree">
        <div class="node">
            <strong>Root: {}</strong><br>
            Threshold: {:.3}<br>
            Samples: {}<br>
            Impurity: {:.3}
        </div>
    </div>
</body>
</html>
"#, tree_data.root.feature, tree_data.root.threshold, tree_data.root.samples, tree_data.root.impurity);
        
        Ok(html)
    }

    async fn generate_json_tree(&self, tree_data: &DecisionTreeData) -> Result<String> {
        let json_data = serde_json::json!({
            "type": "decision_tree",
            "data": tree_data,
            "config": {
                "width": self.config.resolution.0,
                "height": self.config.resolution.1
            }
        });
        Ok(json_data.to_string())
    }

    fn value_to_color(&self, value: f64) -> String {
        match self.config.color_scheme {
            ColorScheme::Viridis => self.viridis_color(value),
            ColorScheme::Plasma => self.plasma_color(value),
            ColorScheme::Inferno => self.inferno_color(value),
            ColorScheme::Grayscale => format!("rgb({0},{0},{0})", (value * 255.0) as u8),
            ColorScheme::Custom(ref colors) => {
                let index = (value * (colors.len() - 1) as f64) as usize;
                colors.get(index).cloned().unwrap_or_else(|| "#000000".to_string())
            }
        }
    }

    fn viridis_color(&self, value: f64) -> String {
        // Simplified viridis color mapping
        let r = (68.0 + value * (253.0 - 68.0)) as u8;
        let g = (1.0 + value * (231.0 - 1.0)) as u8;
        let b = (84.0 + value * (37.0 - 84.0)) as u8;
        format!("rgb({},{},{})", r, g, b)
    }

    fn plasma_color(&self, value: f64) -> String {
        // Simplified plasma color mapping
        let r = (13.0 + value * (240.0 - 13.0)) as u8;
        let g = (8.0 + value * (50.0 - 8.0)) as u8;
        let b = (135.0 + value * (50.0 - 135.0)) as u8;
        format!("rgb({},{},{})", r, g, b)
    }

    fn inferno_color(&self, value: f64) -> String {
        // Simplified inferno color mapping
        let r = (0.0 + value * 255.0) as u8;
        let g = (0.0 + value * 210.0) as u8;
        let b = (4.0 + value * 50.0) as u8;
        format!("rgb({},{},{})", r, g, b)
    }
}

#[derive(Debug, Clone)]
pub struct HeatmapCell {
    pub row: usize,
    pub col: usize,
    pub value: f64,
    pub token_from: String,
    pub token_to: String,
}

#[derive(Debug, Clone)]
pub struct BarChartItem {
    pub label: String,
    pub value: f64,
    pub confidence: Option<(f64, f64)>,
    pub category: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DecisionTreeData {
    pub root: DecisionNode,
    pub max_depth: usize,
    pub total_samples: usize,
}

#[derive(Debug, Clone)]
pub struct DecisionNode {
    pub feature: String,
    pub threshold: f64,
    pub value: Option<f64>,
    pub samples: usize,
    pub impurity: f64,
}

#[derive(Debug, Clone)]
pub struct VisualizationResult {
    pub visualization_type: VisualizationType,
    pub content: String,
    pub metadata: VisualizationMetadata,
}

#[derive(Debug, Clone)]
pub enum VisualizationType {
    AttentionHeatmap,
    FeatureImportance,
    DecisionTree,
    CounterfactualExplanation,
    SHAPWaterfall,
    LIMEExplanation,
}

#[derive(Debug, Clone)]
pub struct VisualizationMetadata {
    pub format: OutputFormat,
    pub dimensions: (u32, u32),
    pub created_at: SystemTime,
    pub interactive: bool,
}