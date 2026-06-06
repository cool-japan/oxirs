//! Rich content types — the data model for the rich messaging layer.
//!
//! Sibling module of [`crate::rich_content`]. Defines [`RichContent`] and
//! its supporting structures: code themes, query stats, graph nodes and
//! edges, table cells, chart datasets, timeline events, maps, 3D scenes,
//! dashboards, and multimedia descriptors. Also exposes the shared
//! [`ChatError`] / [`ChatResult`] aliases used throughout the layer.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Chat-specific error type.
#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    #[error("Processing error: {0}")]
    Processing(String),
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Chat result type alias.
pub type ChatResult<T> = Result<T, ChatError>;

/// Rich content types supported in chat messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RichContent {
    /// Plain text content.
    Text(String),
    /// Code snippet with syntax highlighting.
    CodeSnippet {
        code: String,
        language: String,
        filename: Option<String>,
        line_numbers: bool,
        highlight_lines: Vec<usize>,
        theme: CodeTheme,
    },
    /// SPARQL query block with validation.
    SparqlQuery {
        query: String,
        valid: bool,
        error_message: Option<String>,
        execution_plan: Option<String>,
        performance_stats: Option<QueryPerformanceStats>,
    },
    /// Advanced graph visualization data.
    GraphVisualization {
        nodes: Vec<GraphNode>,
        edges: Vec<GraphEdge>,
        layout: GraphLayout,
        styling: GraphStyling,
        interactive_features: InteractiveFeatures,
        metadata: HashMap<String, String>,
    },
    /// Enhanced table output with advanced features.
    Table {
        headers: Vec<TableHeader>,
        rows: Vec<TableRow>,
        pagination: Option<TablePagination>,
        sorting: Option<TableSorting>,
        filtering: Option<TableFiltering>,
        styling: TableStyling,
        metadata: HashMap<String, String>,
    },
    /// Interactive chart visualization.
    Chart {
        chart_type: ChartType,
        data: ChartData,
        configuration: ChartConfiguration,
        styling: ChartStyling,
        interactive_features: ChartInteractivity,
    },
    /// Timeline visualization.
    Timeline {
        events: Vec<TimelineEvent>,
        range: TimelineRange,
        styling: TimelineStyling,
        zoom_levels: Vec<TimelineZoomLevel>,
        interactive: bool,
    },
    /// Geographic map visualization.
    Map {
        map_type: MapType,
        center: GeoCoordinate,
        zoom_level: u8,
        markers: Vec<MapMarker>,
        layers: Vec<MapLayer>,
        styling: MapStyling,
    },
    /// 3D visualization.
    ThreeDVisualization {
        objects: Vec<ThreeDObject>,
        camera: CameraSettings,
        lighting: LightingSettings,
        materials: Vec<Material>,
        animations: Vec<Animation>,
    },
    /// Data dashboard with multiple widgets.
    Dashboard {
        widgets: Vec<DashboardWidget>,
        layout: DashboardLayout,
        refresh_interval: Option<std::time::Duration>,
        real_time_updates: bool,
    },
    /// Image attachment with annotations.
    Image {
        url: String,
        alt_text: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        annotations: Vec<ImageAnnotation>,
        filters: Vec<ImageFilter>,
    },
    /// Audio content.
    Audio {
        url: String,
        duration: Option<std::time::Duration>,
        waveform: Option<Vec<f32>>,
        transcript: Option<String>,
    },
    /// Video content.
    Video {
        url: String,
        duration: Option<std::time::Duration>,
        thumbnail: Option<String>,
        subtitles: Vec<SubtitleTrack>,
        chapters: Vec<VideoChapter>,
    },
    /// File upload with preview.
    File {
        path: PathBuf,
        filename: String,
        mime_type: String,
        size: u64,
        preview: Option<FilePreview>,
        metadata: FileMetadata,
    },
    /// Interactive widget.
    Widget {
        widget_type: WidgetType,
        configuration: serde_json::Value,
        state: serde_json::Value,
        interactive: bool,
    },
}

// ---------------------------------------------------------------------
// Code visualization types
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum CodeTheme {
    #[default]
    Light,
    Dark,
    HighContrast,
    Monokai,
    Solarized,
    GitHub,
    VSCode,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryPerformanceStats {
    pub execution_time_ms: u64,
    pub result_count: usize,
    pub memory_used_mb: f64,
    pub query_plan_complexity: u32,
}

impl Default for QueryPerformanceStats {
    fn default() -> Self {
        Self {
            execution_time_ms: 0,
            result_count: 0,
            memory_used_mb: 0.0,
            query_plan_complexity: 0,
        }
    }
}

// ---------------------------------------------------------------------
// Graph visualization types
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub node_type: String,
    pub properties: HashMap<String, String>,
    pub position: Option<NodePosition>,
    pub styling: NodeStyling,
    pub size: f64,
    pub shape: NodeShape,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub label: String,
    pub edge_type: String,
    pub properties: HashMap<String, String>,
    pub styling: EdgeStyling,
    pub weight: f64,
    pub directed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodePosition {
    pub x: f64,
    pub y: f64,
    pub z: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeStyling {
    pub color: String,
    pub border_color: String,
    pub border_width: f64,
    pub opacity: f64,
    pub font_size: f64,
    pub font_color: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeShape {
    Circle,
    Square,
    Triangle,
    Diamond,
    Star,
    Hexagon,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeStyling {
    pub color: String,
    pub width: f64,
    pub opacity: f64,
    pub style: EdgeStyle,
    pub arrow_type: ArrowType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EdgeStyle {
    Solid,
    Dashed,
    Dotted,
    Curved,
    Straight,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ArrowType {
    None,
    Standard,
    Filled,
    Open,
    Diamond,
    Circle,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeSize {
    Small,
    Medium,
    Large,
    ExtraLarge,
    Custom(f64),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EdgeThickness {
    Thin,
    Medium,
    Thick,
    ExtraThick,
    Custom(f64),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum GraphLayout {
    #[default]
    ForceDirected,
    Circular,
    Hierarchical,
    Grid,
    Tree,
    Cluster,
    Custom {
        algorithm: String,
        parameters: HashMap<String, f64>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphStyling {
    pub background_color: String,
    pub grid_enabled: bool,
    pub physics_enabled: bool,
    pub clustering_enabled: bool,
    pub smooth_curves: bool,
    pub node_color: String,
    pub edge_color: String,
    pub node_size: NodeSize,
    pub edge_thickness: EdgeThickness,
    pub layout_algorithm: String,
    pub show_labels: bool,
}

impl Default for GraphStyling {
    fn default() -> Self {
        Self {
            background_color: "#ffffff".to_string(),
            grid_enabled: false,
            physics_enabled: true,
            clustering_enabled: false,
            smooth_curves: true,
            node_color: "#3498db".to_string(),
            edge_color: "#7f8c8d".to_string(),
            node_size: NodeSize::Medium,
            edge_thickness: EdgeThickness::Medium,
            layout_algorithm: "force".to_string(),
            show_labels: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InteractiveFeatures {
    pub pan_enabled: bool,
    pub zoom_enabled: bool,
    pub node_selection: bool,
    pub edge_selection: bool,
    pub hover_effects: bool,
    pub click_to_expand: bool,
    pub context_menu: bool,
}

impl Default for InteractiveFeatures {
    fn default() -> Self {
        Self {
            pan_enabled: true,
            zoom_enabled: true,
            node_selection: true,
            edge_selection: true,
            hover_effects: true,
            click_to_expand: true,
            context_menu: true,
        }
    }
}

// ---------------------------------------------------------------------
// Table types
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableHeader {
    pub name: String,
    pub data_type: TableDataType,
    pub sortable: bool,
    pub filterable: bool,
    pub width: Option<u32>,
    pub alignment: TextAlignment,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableRow {
    pub cells: Vec<TableCell>,
    pub metadata: HashMap<String, String>,
    pub styling: Option<RowStyling>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableCell {
    pub value: String,
    pub data_type: TableDataType,
    pub formatting: Option<CellFormatting>,
    pub link: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TableDataType {
    Text,
    Number,
    Date,
    Boolean,
    Image,
    Link,
    Currency,
    Percentage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TextAlignment {
    Left,
    Center,
    Right,
    Justify,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TablePagination {
    pub page_size: usize,
    pub current_page: usize,
    pub total_pages: usize,
    pub show_page_numbers: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableSorting {
    pub column_index: usize,
    pub direction: SortDirection,
    pub secondary_sorts: Vec<(usize, SortDirection)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SortDirection {
    Ascending,
    Descending,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableFiltering {
    pub filters: Vec<ColumnFilter>,
    pub global_filter: Option<String>,
    pub case_sensitive: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnFilter {
    pub column_index: usize,
    pub filter_type: FilterType,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilterType {
    Contains,
    Equals,
    StartsWith,
    EndsWith,
    GreaterThan,
    LessThan,
    Range(String, String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableStyling {
    pub theme: TableTheme,
    pub striped_rows: bool,
    pub border_style: BorderStyle,
    pub header_styling: HeaderStyling,
    pub cell_padding: u32,
}

impl Default for TableStyling {
    fn default() -> Self {
        Self {
            theme: TableTheme::Default,
            striped_rows: true,
            border_style: BorderStyle::Solid,
            header_styling: HeaderStyling::default(),
            cell_padding: 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TableTheme {
    Default,
    Minimal,
    Modern,
    Classic,
    Dark,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BorderStyle {
    None,
    Solid,
    Dashed,
    Dotted,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeaderStyling {
    pub background_color: String,
    pub text_color: String,
    pub font_weight: FontWeight,
    pub fixed_header: bool,
}

impl Default for HeaderStyling {
    fn default() -> Self {
        Self {
            background_color: "#f8f9fa".to_string(),
            text_color: "#212529".to_string(),
            font_weight: FontWeight::Bold,
            fixed_header: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FontWeight {
    Normal,
    Bold,
    Light,
    ExtraBold,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RowStyling {
    pub background_color: Option<String>,
    pub text_color: Option<String>,
    pub highlighted: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellFormatting {
    pub color: Option<String>,
    pub background_color: Option<String>,
    pub font_weight: Option<FontWeight>,
    pub italic: bool,
    pub underline: bool,
}

// ---------------------------------------------------------------------
// Chart types
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChartType {
    Line,
    Bar,
    Pie,
    Scatter,
    Area,
    Histogram,
    BoxPlot,
    Radar,
    Heatmap,
    Treemap,
    Sunburst,
    Sankey,
    Gantt,
    Candlestick,
    Bubble,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChartData {
    pub datasets: Vec<ChartDataset>,
    pub categories: Vec<String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChartDataset {
    pub label: String,
    pub data: Vec<f64>,
    pub color: String,
    pub secondary_data: Option<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChartConfiguration {
    pub title: Option<String>,
    pub width: u32,
    pub height: u32,
    pub responsive: bool,
    pub animations_enabled: bool,
    pub legend_position: LegendPosition,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LegendPosition {
    Top,
    Bottom,
    Left,
    Right,
    Hidden,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChartStyling {
    pub theme: ChartTheme,
    pub color_palette: Vec<String>,
    pub grid_lines: bool,
    pub background_color: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChartTheme {
    Default,
    Dark,
    Minimal,
    Colorful,
    Professional,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChartInteractivity {
    pub hover_enabled: bool,
    pub click_enabled: bool,
    pub zoom_enabled: bool,
    pub pan_enabled: bool,
    pub tooltip_enabled: bool,
    pub crossfilter_enabled: bool,
}

// ---------------------------------------------------------------------
// Timeline types
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelineEvent {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    pub category: String,
    pub importance: EventImportance,
    pub styling: EventStyling,
    pub attachments: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EventImportance {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventStyling {
    pub color: String,
    pub icon: Option<String>,
    pub size: EventSize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EventSize {
    Small,
    Medium,
    Large,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelineRange {
    pub start: chrono::DateTime<chrono::Utc>,
    pub end: chrono::DateTime<chrono::Utc>,
    pub default_view: TimelineView,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TimelineView {
    Day,
    Week,
    Month,
    Year,
    Decade,
    Custom(std::time::Duration),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelineStyling {
    pub orientation: TimelineOrientation,
    pub background_color: String,
    pub axis_color: String,
    pub show_grid: bool,
    pub compact_mode: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TimelineOrientation {
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelineZoomLevel {
    pub level: u8,
    pub unit: TimeUnit,
    pub granularity: std::time::Duration,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TimeUnit {
    Millisecond,
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Year,
}

// ---------------------------------------------------------------------
// Map types
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MapType {
    Road,
    Satellite,
    Hybrid,
    Terrain,
    OpenStreetMap,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoCoordinate {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapMarker {
    pub id: String,
    pub position: GeoCoordinate,
    pub label: String,
    pub description: Option<String>,
    pub icon: MarkerIcon,
    pub popup_content: Option<String>,
    pub clickable: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarkerIcon {
    pub icon_type: IconType,
    pub color: String,
    pub size: IconSize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IconType {
    Pin,
    Circle,
    Square,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IconSize {
    Small,
    Medium,
    Large,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapLayer {
    pub id: String,
    pub layer_type: LayerType,
    pub data_source: String,
    pub visible: bool,
    pub opacity: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LayerType {
    Markers,
    Heatmap,
    Polygons,
    Polylines,
    Circles,
    GeoJSON,
    Tiles,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapStyling {
    pub theme: MapTheme,
    pub show_controls: bool,
    pub show_scale: bool,
    pub show_attribution: bool,
    pub custom_style: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MapTheme {
    Default,
    Dark,
    Light,
    Satellite,
    Custom(String),
}

// ---------------------------------------------------------------------
// 3D Visualization types
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeDObject {
    pub id: String,
    pub object_type: ObjectType,
    pub position: Vector3D,
    pub rotation: Vector3D,
    pub scale: Vector3D,
    pub material_id: String,
    pub animation_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ObjectType {
    Cube,
    Sphere,
    Cylinder,
    Plane,
    Mesh,
    PointCloud,
    Model(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Vector3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraSettings {
    pub position: Vector3D,
    pub target: Vector3D,
    pub field_of_view: f64,
    pub near_plane: f64,
    pub far_plane: f64,
    pub camera_type: CameraType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CameraType {
    Perspective,
    Orthographic,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LightingSettings {
    pub ambient_light: LightSource,
    pub directional_lights: Vec<LightSource>,
    pub point_lights: Vec<LightSource>,
    pub spot_lights: Vec<LightSource>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LightSource {
    pub id: String,
    pub light_type: LightType,
    pub color: String,
    pub intensity: f64,
    pub position: Option<Vector3D>,
    pub direction: Option<Vector3D>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LightType {
    Ambient,
    Directional,
    Point,
    Spot,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Material {
    pub id: String,
    pub material_type: MaterialType,
    pub color: String,
    pub texture: Option<String>,
    pub metallic: f64,
    pub roughness: f64,
    pub opacity: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MaterialType {
    Basic,
    Phong,
    PhysicallyBased,
    Wireframe,
    Points,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Animation {
    pub id: String,
    pub target_object_id: String,
    pub animation_type: AnimationType,
    pub duration: std::time::Duration,
    pub loop_enabled: bool,
    pub keyframes: Vec<Keyframe>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnimationType {
    Position,
    Rotation,
    Scale,
    Color,
    Opacity,
    Morph,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Keyframe {
    pub time: f64,
    pub value: Vector3D,
    pub interpolation: InterpolationType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InterpolationType {
    Linear,
    Bezier,
    Step,
}

// ---------------------------------------------------------------------
// Dashboard types
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DashboardWidget {
    pub id: String,
    pub widget_type: DashboardWidgetType,
    pub title: String,
    pub position: WidgetPosition,
    pub size: WidgetSize,
    pub data_source: String,
    pub configuration: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DashboardWidgetType {
    Chart,
    Table,
    Map,
    Text,
    Metric,
    Image,
    IFrame,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WidgetPosition {
    pub x: u32,
    pub y: u32,
    pub z_index: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WidgetSize {
    pub width: u32,
    pub height: u32,
    pub min_width: u32,
    pub min_height: u32,
    pub resizable: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DashboardLayout {
    Grid,
    Flexible,
    Masonry,
    Custom,
}

// ---------------------------------------------------------------------
// Multimedia types
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageAnnotation {
    pub id: String,
    pub annotation_type: AnnotationType,
    pub position: AnnotationPosition,
    pub content: String,
    pub styling: AnnotationStyling,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnnotationType {
    Text,
    Arrow,
    Rectangle,
    Circle,
    Highlight,
    Blur,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotationPosition {
    pub x: f64,
    pub y: f64,
    pub width: Option<f64>,
    pub height: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotationStyling {
    pub color: String,
    pub background_color: Option<String>,
    pub border_width: f64,
    pub font_size: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImageFilter {
    Blur(f64),
    Brightness(f64),
    Contrast(f64),
    Saturation(f64),
    Sepia(f64),
    Grayscale,
    Invert,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubtitleTrack {
    pub language: String,
    pub label: String,
    pub url: String,
    pub default: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoChapter {
    pub title: String,
    pub start_time: std::time::Duration,
    pub end_time: std::time::Duration,
    pub thumbnail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilePreview {
    Text(String),
    Image(String),
    Audio(String),
    Video(String),
    Document(String),
    None,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileMetadata {
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub modified_at: chrono::DateTime<chrono::Utc>,
    pub author: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub custom_properties: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WidgetType {
    Button,
    Slider,
    Toggle,
    Input,
    Dropdown,
    DatePicker,
    ColorPicker,
    FileUpload,
    RangeSlider,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WidgetConfig {
    pub interactive: bool,
    pub disabled: bool,
    pub validation_rules: Vec<ValidationRule>,
    pub event_handlers: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationRule {
    pub rule_type: ValidationType,
    pub value: String,
    pub error_message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationType {
    Required,
    MinLength,
    MaxLength,
    Pattern,
    Range,
    Email,
    Url,
    Custom(String),
}
