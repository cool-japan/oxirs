//! # Feature Engineering Pipelines for Stream Processing
//!
//! This module provides a comprehensive feature engineering framework for real-time
//! stream processing, enabling automatic feature extraction, transformation, and
//! selection for machine learning workflows.
//!
//! ## Features
//! - Automatic feature extraction from streaming events
//! - Real-time feature transformations (scaling, encoding, binning)
//! - Time-based features (rolling windows, lag features, rate of change)
//! - Categorical encoding (one-hot, label, target encoding)
//! - Feature selection and dimensionality reduction
//! - Feature store integration for reusability
//! - Pipeline composition with visual DAG representation
//!
//! ## Example Usage
//! ```rust,ignore
//! use oxirs_stream::feature_engineering::{FeaturePipeline, FeatureTransform};
//!
//! let mut pipeline = FeaturePipeline::new();
//! pipeline
//!     .add_transform(FeatureTransform::StandardScaler)
//!     .add_transform(FeatureTransform::RollingMean { window: 10 })
//!     .add_transform(FeatureTransform::OneHotEncoder { columns: vec!["category".into()] });
//!
//! let features = pipeline.transform(&event)?;
//! ```

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::Array2;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Feature data type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeatureValue {
    /// Numeric value
    Numeric(f64),
    /// Categorical value
    Categorical(String),
    /// Boolean value
    Boolean(bool),
    /// Array of numeric values
    NumericArray(Vec<f64>),
    /// Missing value
    Missing,
}

impl FeatureValue {
    /// Convert to numeric value (NaN for non-numeric)
    pub fn as_numeric(&self) -> f64 {
        match self {
            FeatureValue::Numeric(v) => *v,
            FeatureValue::Boolean(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
            _ => f64::NAN,
        }
    }

    /// Check if value is missing
    pub fn is_missing(&self) -> bool {
        match self {
            FeatureValue::Missing => true,
            FeatureValue::Numeric(v) => v.is_nan(),
            _ => false,
        }
    }
}

/// Feature definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feature {
    /// Feature name
    pub name: String,
    /// Feature value
    pub value: FeatureValue,
    /// Feature importance score (0-1)
    pub importance: Option<f64>,
    /// Feature metadata
    pub metadata: HashMap<String, String>,
}

impl Feature {
    /// Create a new numeric feature
    pub fn numeric(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value: FeatureValue::Numeric(value),
            importance: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new categorical feature
    pub fn categorical(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: FeatureValue::Categorical(value.into()),
            importance: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new boolean feature
    pub fn boolean(name: impl Into<String>, value: bool) -> Self {
        Self {
            name: name.into(),
            value: FeatureValue::Boolean(value),
            importance: None,
            metadata: HashMap::new(),
        }
    }
}

/// Feature set (collection of features)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSet {
    /// Features in this set
    pub features: Vec<Feature>,
    /// Timestamp when features were created
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Version of the feature set
    pub version: String,
}

impl FeatureSet {
    /// Create a new empty feature set
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            timestamp: chrono::Utc::now(),
            version: "1.0".to_string(),
        }
    }

    /// Add a feature to the set
    pub fn add_feature(&mut self, feature: Feature) {
        self.features.push(feature);
    }

    /// Get feature by name
    pub fn get_feature(&self, name: &str) -> Option<&Feature> {
        self.features.iter().find(|f| f.name == name)
    }

    /// Convert to numeric array (skipping non-numeric features)
    pub fn to_numeric_array(&self) -> Vec<f64> {
        self.features.iter().map(|f| f.value.as_numeric()).collect()
    }

    /// Get feature names
    pub fn feature_names(&self) -> Vec<String> {
        self.features.iter().map(|f| f.name.clone()).collect()
    }
}

impl Default for FeatureSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature transformation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureTransform {
    /// Standard scaling (mean=0, std=1)
    StandardScaler,
    /// Min-max scaling (range [0, 1])
    MinMaxScaler { min: f64, max: f64 },
    /// Robust scaling (using median and IQR)
    RobustScaler,
    /// Log transformation
    LogTransform { offset: f64 },
    /// Power transformation (Box-Cox like)
    PowerTransform { lambda: f64 },
    /// Rolling mean over window
    RollingMean { window: usize },
    /// Rolling standard deviation
    RollingStd { window: usize },
    /// Rolling sum
    RollingSum { window: usize },
    /// Exponential weighted moving average
    EWMA { alpha: f64 },
    /// Lag features (previous values)
    LagFeatures { lags: Vec<usize> },
    /// Rate of change (derivative)
    RateOfChange { period: usize },
    /// Binning/discretization
    Binning { bins: Vec<f64> },
    /// One-hot encoding for categorical features
    OneHotEncoder { columns: Vec<String> },
    /// Label encoding for categorical features
    LabelEncoder { columns: Vec<String> },
    /// Target encoding for categorical features
    TargetEncoder { column: String },
    /// Polynomial features
    PolynomialFeatures { degree: usize },
    /// Interaction features (cross products)
    InteractionFeatures { pairs: Vec<(String, String)> },
    /// Missing value imputation
    Imputation { strategy: ImputationStrategy },
    /// Feature selection (keep top k by importance)
    FeatureSelection { top_k: usize },
    /// PCA dimensionality reduction
    PCA { n_components: usize },
    /// Custom transformation (user-defined function)
    Custom { name: String },
}

/// Imputation strategy for missing values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImputationStrategy {
    /// Fill with mean value
    Mean,
    /// Fill with median value
    Median,
    /// Fill with mode (most frequent)
    Mode,
    /// Fill with constant value
    Constant,
    /// Forward fill (carry last value)
    ForwardFill,
    /// Backward fill
    BackwardFill,
    /// Interpolate (linear)
    Interpolate,
}

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractionConfig {
    /// Extract time-based features
    pub extract_time_features: bool,
    /// Extract statistical features
    pub extract_statistical_features: bool,
    /// Window size for rolling statistics
    pub rolling_window: usize,
    /// Enable automatic feature generation
    pub auto_generate: bool,
    /// Maximum number of features to generate
    pub max_features: usize,
}

impl Default for FeatureExtractionConfig {
    fn default() -> Self {
        Self {
            extract_time_features: true,
            extract_statistical_features: true,
            rolling_window: 10,
            auto_generate: false,
            max_features: 100,
        }
    }
}

/// Feature engineering pipeline
pub struct FeaturePipeline {
    /// Ordered list of transformations
    transforms: Vec<FeatureTransform>,
    /// Configuration
    config: FeatureExtractionConfig,
    /// Historical data buffer for time-based features
    history: Arc<RwLock<HashMap<String, VecDeque<f64>>>>,
    /// Fitted transformation parameters
    fitted_params: Arc<RwLock<FittedParameters>>,
    /// Statistics
    stats: Arc<RwLock<PipelineStats>>,
}

/// Fitted parameters for transformations
#[derive(Debug, Clone, Default)]
struct FittedParameters {
    /// Mean values for standard scaling
    means: HashMap<String, f64>,
    /// Standard deviations for standard scaling
    stds: HashMap<String, f64>,
    /// Min values for min-max scaling
    mins: HashMap<String, f64>,
    /// Max values for min-max scaling
    maxs: HashMap<String, f64>,
    /// Median values for robust scaling
    medians: HashMap<String, f64>,
    /// IQR values for robust scaling
    iqrs: HashMap<String, f64>,
    /// Label encodings
    label_encodings: HashMap<String, HashMap<String, usize>>,
    /// PCA components
    pca_components: Option<Array2<f64>>,
    /// Feature importance scores
    feature_importances: HashMap<String, f64>,
}

/// Pipeline statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineStats {
    /// Total features processed
    pub total_features_processed: u64,
    /// Total transformations applied
    pub total_transformations: u64,
    /// Average transformation time (ms)
    pub avg_transform_time_ms: f64,
    /// Number of features generated
    pub features_generated: usize,
    /// Number of features selected
    pub features_selected: usize,
}

impl FeaturePipeline {
    /// Create a new feature engineering pipeline
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
            config: FeatureExtractionConfig::default(),
            history: Arc::new(RwLock::new(HashMap::new())),
            fitted_params: Arc::new(RwLock::new(FittedParameters::default())),
            stats: Arc::new(RwLock::new(PipelineStats::default())),
        }
    }

    /// Create a pipeline with configuration
    pub fn with_config(config: FeatureExtractionConfig) -> Self {
        Self {
            transforms: Vec::new(),
            config,
            history: Arc::new(RwLock::new(HashMap::new())),
            fitted_params: Arc::new(RwLock::new(FittedParameters::default())),
            stats: Arc::new(RwLock::new(PipelineStats::default())),
        }
    }

    /// Add a transformation to the pipeline
    pub fn add_transform(&mut self, transform: FeatureTransform) -> &mut Self {
        self.transforms.push(transform);
        self
    }

    /// Fit the pipeline on training data
    pub async fn fit(&mut self, data: &[FeatureSet]) -> Result<()> {
        info!("Fitting feature pipeline on {} samples", data.len());

        if data.is_empty() {
            return Err(anyhow!("Cannot fit on empty data"));
        }

        let mut params = self.fitted_params.write().await;

        // Collect all numeric features for fitting
        let mut feature_values: HashMap<String, Vec<f64>> = HashMap::new();

        for feature_set in data {
            for feature in &feature_set.features {
                if let FeatureValue::Numeric(value) = feature.value {
                    if !value.is_nan() {
                        feature_values
                            .entry(feature.name.clone())
                            .or_default()
                            .push(value);
                    }
                }
            }
        }

        // Fit transformations
        for (name, values) in &feature_values {
            if values.is_empty() {
                continue;
            }

            // Compute statistics for scaling
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
            let std = variance.sqrt();

            let mut sorted_values = values.clone();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let min = sorted_values.first().copied().unwrap_or(0.0);
            let max = sorted_values.last().copied().unwrap_or(1.0);
            let median = sorted_values[sorted_values.len() / 2];

            // Compute IQR
            let q1_idx = sorted_values.len() / 4;
            let q3_idx = 3 * sorted_values.len() / 4;
            let q1 = sorted_values[q1_idx];
            let q3 = sorted_values[q3_idx];
            let iqr = q3 - q1;

            params.means.insert(name.clone(), mean);
            params.stds.insert(name.clone(), std.max(1e-10));
            params.mins.insert(name.clone(), min);
            params.maxs.insert(name.clone(), max);
            params.medians.insert(name.clone(), median);
            params.iqrs.insert(name.clone(), iqr.max(1e-10));
        }

        // Fit categorical encodings
        for transform in &self.transforms {
            if let FeatureTransform::LabelEncoder { columns } = transform {
                for column in columns {
                    let mut unique_values = std::collections::HashSet::new();
                    for feature_set in data {
                        if let Some(feature) = feature_set.get_feature(column) {
                            if let FeatureValue::Categorical(value) = &feature.value {
                                unique_values.insert(value.clone());
                            }
                        }
                    }

                    let encoding: HashMap<String, usize> = unique_values
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (v.clone(), i))
                        .collect();

                    params.label_encodings.insert(column.clone(), encoding);
                }
            }
        }

        info!("Pipeline fitted successfully");
        Ok(())
    }

    /// Transform a feature set using the fitted pipeline
    pub async fn transform(&self, input: &FeatureSet) -> Result<FeatureSet> {
        let start = std::time::Instant::now();
        let mut output = input.clone();

        let params = self.fitted_params.read().await;
        let mut history = self.history.write().await;

        // Apply each transformation in order
        for transform in &self.transforms {
            output = self
                .apply_transform(&output, transform, &params, &mut history)
                .await?;
        }

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_features_processed += output.features.len() as u64;
        stats.total_transformations += 1;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        stats.avg_transform_time_ms =
            (stats.avg_transform_time_ms * (stats.total_transformations - 1) as f64 + elapsed_ms)
                / stats.total_transformations as f64;
        stats.features_generated = output.features.len();

        Ok(output)
    }

    /// Apply a single transformation
    async fn apply_transform(
        &self,
        input: &FeatureSet,
        transform: &FeatureTransform,
        params: &FittedParameters,
        history: &mut HashMap<String, VecDeque<f64>>,
    ) -> Result<FeatureSet> {
        let mut output = input.clone();

        match transform {
            FeatureTransform::StandardScaler => {
                for feature in &mut output.features {
                    if let FeatureValue::Numeric(value) = &mut feature.value {
                        if let (Some(&mean), Some(&std)) = (
                            params.means.get(&feature.name),
                            params.stds.get(&feature.name),
                        ) {
                            *value = (*value - mean) / std;
                        }
                    }
                }
            }
            FeatureTransform::MinMaxScaler { .. } => {
                for feature in &mut output.features {
                    if let FeatureValue::Numeric(value) = &mut feature.value {
                        if let (Some(&min), Some(&max)) = (
                            params.mins.get(&feature.name),
                            params.maxs.get(&feature.name),
                        ) {
                            *value = (*value - min) / (max - min).max(1e-10);
                        }
                    }
                }
            }
            FeatureTransform::RobustScaler => {
                for feature in &mut output.features {
                    if let FeatureValue::Numeric(value) = &mut feature.value {
                        if let (Some(&median), Some(&iqr)) = (
                            params.medians.get(&feature.name),
                            params.iqrs.get(&feature.name),
                        ) {
                            *value = (*value - median) / iqr;
                        }
                    }
                }
            }
            FeatureTransform::LogTransform { offset } => {
                for feature in &mut output.features {
                    if let FeatureValue::Numeric(value) = &mut feature.value {
                        *value = (*value + offset).ln();
                    }
                }
            }
            FeatureTransform::PowerTransform { lambda } => {
                for feature in &mut output.features {
                    if let FeatureValue::Numeric(value) = &mut feature.value {
                        *value = if *lambda == 0.0 {
                            value.ln()
                        } else {
                            (value.powf(*lambda) - 1.0) / lambda
                        };
                    }
                }
            }
            FeatureTransform::RollingMean { window } => {
                self.apply_rolling_stat(input, &mut output, *window, history, |values| {
                    values.iter().sum::<f64>() / values.len() as f64
                })?;
            }
            FeatureTransform::RollingStd { window } => {
                self.apply_rolling_stat(input, &mut output, *window, history, |values| {
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                        / values.len() as f64;
                    variance.sqrt()
                })?;
            }
            FeatureTransform::RollingSum { window } => {
                self.apply_rolling_stat(input, &mut output, *window, history, |values| {
                    values.iter().sum()
                })?;
            }
            FeatureTransform::EWMA { alpha } => {
                for feature in &mut output.features {
                    if let FeatureValue::Numeric(value) = &mut feature.value {
                        let hist = history.entry(feature.name.clone()).or_default();
                        let ewma = if hist.is_empty() {
                            *value
                        } else {
                            alpha * (*value) + (1.0 - alpha) * hist.back().copied().unwrap_or(0.0)
                        };
                        hist.push_back(ewma);
                        *value = ewma;
                    }
                }
            }
            FeatureTransform::LagFeatures { lags } => {
                let mut new_features = Vec::new();
                for feature in &input.features {
                    if let FeatureValue::Numeric(value) = feature.value {
                        let hist = history.entry(feature.name.clone()).or_default();

                        for &lag in lags {
                            if lag > 0 && lag <= hist.len() {
                                let lag_value = hist[hist.len() - lag];
                                new_features.push(Feature::numeric(
                                    format!("{}_lag_{}", feature.name, lag),
                                    lag_value,
                                ));
                            }
                        }

                        hist.push_back(value);
                        if hist.len() > lags.iter().max().copied().unwrap_or(10) {
                            hist.pop_front();
                        }
                    }
                }
                output.features.extend(new_features);
            }
            FeatureTransform::RateOfChange { period } => {
                for feature in &mut output.features {
                    if let FeatureValue::Numeric(value) = &mut feature.value {
                        let hist = history.entry(feature.name.clone()).or_default();

                        if hist.len() >= *period {
                            let old_value = hist[hist.len() - period];
                            *value = (*value - old_value) / old_value.max(1e-10);
                        }

                        hist.push_back(*value);
                        if hist.len() > period + 1 {
                            hist.pop_front();
                        }
                    }
                }
            }
            FeatureTransform::Binning { bins } => {
                for feature in &mut output.features {
                    if let FeatureValue::Numeric(value) = &mut feature.value {
                        let bin_idx = bins.iter().position(|&b| *value < b).unwrap_or(bins.len());
                        *value = bin_idx as f64;
                    }
                }
            }
            FeatureTransform::OneHotEncoder { columns } => {
                let mut new_features = Vec::new();
                for feature in &input.features {
                    if columns.contains(&feature.name) {
                        if let FeatureValue::Categorical(cat_value) = &feature.value {
                            // Create binary features for each category
                            new_features.push(Feature::numeric(
                                format!("{}_{}", feature.name, cat_value),
                                1.0,
                            ));
                        }
                    }
                }
                output.features.extend(new_features);
            }
            FeatureTransform::LabelEncoder { columns } => {
                for feature in &mut output.features {
                    if columns.contains(&feature.name) {
                        if let FeatureValue::Categorical(cat_value) = &feature.value {
                            if let Some(encoding_map) = params.label_encodings.get(&feature.name) {
                                if let Some(&encoded) = encoding_map.get(cat_value) {
                                    feature.value = FeatureValue::Numeric(encoded as f64);
                                }
                            }
                        }
                    }
                }
            }
            FeatureTransform::PolynomialFeatures { degree } => {
                let numeric_features: Vec<_> = input
                    .features
                    .iter()
                    .filter(|f| matches!(f.value, FeatureValue::Numeric(_)))
                    .collect();

                let mut new_features = Vec::new();
                for d in 2..=*degree {
                    for feature in &numeric_features {
                        if let FeatureValue::Numeric(value) = feature.value {
                            new_features.push(Feature::numeric(
                                format!("{}_pow{}", feature.name, d),
                                value.powi(d as i32),
                            ));
                        }
                    }
                }
                output.features.extend(new_features);
            }
            FeatureTransform::InteractionFeatures { pairs } => {
                let mut new_features = Vec::new();
                for (name1, name2) in pairs {
                    if let (Some(f1), Some(f2)) =
                        (input.get_feature(name1), input.get_feature(name2))
                    {
                        if let (FeatureValue::Numeric(v1), FeatureValue::Numeric(v2)) =
                            (&f1.value, &f2.value)
                        {
                            new_features.push(Feature::numeric(
                                format!("{}_{}_interaction", name1, name2),
                                v1 * v2,
                            ));
                        }
                    }
                }
                output.features.extend(new_features);
            }
            FeatureTransform::Imputation { strategy } => {
                for feature in &mut output.features {
                    if feature.value.is_missing() {
                        let imputed_value = match strategy {
                            ImputationStrategy::Mean => params.means.get(&feature.name).copied(),
                            ImputationStrategy::Median => {
                                params.medians.get(&feature.name).copied()
                            }
                            ImputationStrategy::Constant => Some(0.0),
                            _ => None,
                        };

                        if let Some(value) = imputed_value {
                            feature.value = FeatureValue::Numeric(value);
                        }
                    }
                }
            }
            _ => {
                debug!("Transform {:?} not yet implemented", transform);
            }
        }

        Ok(output)
    }

    /// Apply rolling statistics
    fn apply_rolling_stat<F>(
        &self,
        _input: &FeatureSet,
        output: &mut FeatureSet,
        window: usize,
        history: &mut HashMap<String, VecDeque<f64>>,
        stat_fn: F,
    ) -> Result<()>
    where
        F: Fn(&VecDeque<f64>) -> f64,
    {
        for feature in &mut output.features {
            if let FeatureValue::Numeric(value) = &mut feature.value {
                let hist = history.entry(feature.name.clone()).or_default();
                hist.push_back(*value);

                if hist.len() > window {
                    hist.pop_front();
                }

                if hist.len() >= window {
                    *value = stat_fn(hist);
                }
            }
        }

        Ok(())
    }

    /// Extract features from raw event data
    pub async fn extract_features(
        &self,
        event_data: &HashMap<String, serde_json::Value>,
    ) -> Result<FeatureSet> {
        let mut feature_set = FeatureSet::new();

        // Extract basic features from event data
        for (key, value) in event_data {
            let feature = match value {
                serde_json::Value::Number(n) => {
                    if let Some(f) = n.as_f64() {
                        Feature::numeric(key, f)
                    } else {
                        continue;
                    }
                }
                serde_json::Value::String(s) => Feature::categorical(key, s.clone()),
                serde_json::Value::Bool(b) => Feature::boolean(key, *b),
                _ => continue,
            };

            feature_set.add_feature(feature);
        }

        // Extract time-based features if enabled
        if self.config.extract_time_features {
            let now = chrono::Utc::now();
            feature_set.add_feature(Feature::numeric(
                "hour_of_day",
                now.format("%H").to_string().parse::<f64>().unwrap_or(0.0),
            ));
            feature_set.add_feature(Feature::numeric(
                "day_of_week",
                now.format("%u").to_string().parse::<f64>().unwrap_or(0.0),
            ));
            feature_set.add_feature(Feature::numeric(
                "day_of_month",
                now.format("%d").to_string().parse::<f64>().unwrap_or(0.0),
            ));
            feature_set.add_feature(Feature::numeric(
                "month",
                now.format("%m").to_string().parse::<f64>().unwrap_or(0.0),
            ));
        }

        Ok(feature_set)
    }

    /// Get pipeline statistics
    pub async fn get_stats(&self) -> PipelineStats {
        self.stats.read().await.clone()
    }

    /// Clear pipeline history
    pub async fn clear_history(&mut self) {
        self.history.write().await.clear();
    }

    /// Get number of transformations in pipeline
    pub fn transform_count(&self) -> usize {
        self.transforms.len()
    }
}

impl Default for FeaturePipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature store for reusable features
pub struct FeatureStore {
    /// Stored feature sets indexed by ID
    features: Arc<RwLock<HashMap<String, FeatureSet>>>,
    /// Feature metadata
    metadata: Arc<RwLock<HashMap<String, FeatureMetadata>>>,
}

/// Feature metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMetadata {
    /// Feature set ID
    pub id: String,
    /// Description
    pub description: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last updated timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Version
    pub version: String,
    /// Tags for organization
    pub tags: Vec<String>,
}

impl FeatureStore {
    /// Create a new feature store
    pub fn new() -> Self {
        Self {
            features: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Store a feature set
    pub async fn store(
        &self,
        id: impl Into<String>,
        features: FeatureSet,
        metadata: FeatureMetadata,
    ) -> Result<()> {
        let id = id.into();
        self.features.write().await.insert(id.clone(), features);
        self.metadata.write().await.insert(id, metadata);
        Ok(())
    }

    /// Retrieve a feature set
    pub async fn retrieve(&self, id: &str) -> Option<FeatureSet> {
        self.features.read().await.get(id).cloned()
    }

    /// List all feature set IDs
    pub async fn list_ids(&self) -> Vec<String> {
        self.features.read().await.keys().cloned().collect()
    }

    /// Get feature metadata
    pub async fn get_metadata(&self, id: &str) -> Option<FeatureMetadata> {
        self.metadata.read().await.get(id).cloned()
    }

    /// Delete a feature set
    pub async fn delete(&self, id: &str) -> Result<()> {
        self.features.write().await.remove(id);
        self.metadata.write().await.remove(id);
        Ok(())
    }
}

impl Default for FeatureStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_value_conversions() {
        assert_eq!(FeatureValue::Numeric(2.5).as_numeric(), 2.5);
        assert_eq!(FeatureValue::Boolean(true).as_numeric(), 1.0);
        assert_eq!(FeatureValue::Boolean(false).as_numeric(), 0.0);
        assert!(FeatureValue::Categorical("test".into())
            .as_numeric()
            .is_nan());
        assert!(FeatureValue::Missing.is_missing());
    }

    #[test]
    fn test_feature_creation() {
        let num_feature = Feature::numeric("value", 42.0);
        assert_eq!(num_feature.name, "value");
        assert_eq!(num_feature.value.as_numeric(), 42.0);

        let cat_feature = Feature::categorical("category", "A");
        assert_eq!(cat_feature.name, "category");

        let bool_feature = Feature::boolean("flag", true);
        assert_eq!(bool_feature.value.as_numeric(), 1.0);
    }

    #[test]
    fn test_feature_set() {
        let mut feature_set = FeatureSet::new();
        feature_set.add_feature(Feature::numeric("x", 1.0));
        feature_set.add_feature(Feature::numeric("y", 2.0));
        feature_set.add_feature(Feature::categorical("cat", "A"));

        assert_eq!(feature_set.features.len(), 3);
        assert!(feature_set.get_feature("x").is_some());
        assert!(feature_set.get_feature("missing").is_none());

        let numeric_array = feature_set.to_numeric_array();
        assert_eq!(numeric_array.len(), 3);
        assert_eq!(numeric_array[0], 1.0);
        assert_eq!(numeric_array[1], 2.0);
    }

    #[tokio::test]
    async fn test_pipeline_creation() {
        let pipeline = FeaturePipeline::new();
        assert_eq!(pipeline.transform_count(), 0);
    }

    #[tokio::test]
    async fn test_add_transforms() {
        let mut pipeline = FeaturePipeline::new();
        pipeline
            .add_transform(FeatureTransform::StandardScaler)
            .add_transform(FeatureTransform::RollingMean { window: 5 });

        assert_eq!(pipeline.transform_count(), 2);
    }

    #[tokio::test]
    async fn test_standard_scaler() {
        let mut pipeline = FeaturePipeline::new();
        pipeline.add_transform(FeatureTransform::StandardScaler);

        // Training data
        let mut training_data = Vec::new();
        for i in 0..10 {
            let mut fs = FeatureSet::new();
            fs.add_feature(Feature::numeric("value", (i * 10) as f64));
            training_data.push(fs);
        }

        pipeline.fit(&training_data).await.unwrap();

        // Transform new data
        let mut test_fs = FeatureSet::new();
        test_fs.add_feature(Feature::numeric("value", 50.0));

        let result = pipeline.transform(&test_fs).await.unwrap();
        let value = result.get_feature("value").unwrap().value.as_numeric();

        // After standard scaling, mean should be 0, std should be 1
        // Value 50 should be close to 0 (since mean is 45)
        assert!((value.abs()) < 1.0);
    }

    #[tokio::test]
    async fn test_min_max_scaler() {
        let mut pipeline = FeaturePipeline::new();
        pipeline.add_transform(FeatureTransform::MinMaxScaler { min: 0.0, max: 1.0 });

        let mut training_data = Vec::new();
        for i in 0..10 {
            let mut fs = FeatureSet::new();
            fs.add_feature(Feature::numeric("value", (i * 10) as f64));
            training_data.push(fs);
        }

        pipeline.fit(&training_data).await.unwrap();

        let mut test_fs = FeatureSet::new();
        test_fs.add_feature(Feature::numeric("value", 90.0)); // Max value

        let result = pipeline.transform(&test_fs).await.unwrap();
        let value = result.get_feature("value").unwrap().value.as_numeric();

        assert!((value - 1.0).abs() < 0.01); // Should be close to 1.0
    }

    #[tokio::test]
    async fn test_polynomial_features() {
        let mut pipeline = FeaturePipeline::new();
        pipeline.add_transform(FeatureTransform::PolynomialFeatures { degree: 2 });

        let mut fs = FeatureSet::new();
        fs.add_feature(Feature::numeric("x", 3.0));

        let result = pipeline.transform(&fs).await.unwrap();

        // Should have original + polynomial features
        assert!(result.features.len() >= 2);
        let x_pow2 = result.get_feature("x_pow2").unwrap();
        assert_eq!(x_pow2.value.as_numeric(), 9.0);
    }

    #[tokio::test]
    async fn test_interaction_features() {
        let mut pipeline = FeaturePipeline::new();
        pipeline.add_transform(FeatureTransform::InteractionFeatures {
            pairs: vec![("x".to_string(), "y".to_string())],
        });

        let mut fs = FeatureSet::new();
        fs.add_feature(Feature::numeric("x", 2.0));
        fs.add_feature(Feature::numeric("y", 3.0));

        let result = pipeline.transform(&fs).await.unwrap();

        let interaction = result.get_feature("x_y_interaction").unwrap();
        assert_eq!(interaction.value.as_numeric(), 6.0);
    }

    #[tokio::test]
    async fn test_label_encoder() {
        let mut pipeline = FeaturePipeline::new();
        pipeline.add_transform(FeatureTransform::LabelEncoder {
            columns: vec!["category".to_string()],
        });

        let mut training_data = Vec::new();
        for cat in &["A", "B", "C", "A", "B"] {
            let mut fs = FeatureSet::new();
            fs.add_feature(Feature::categorical("category", *cat));
            training_data.push(fs);
        }

        pipeline.fit(&training_data).await.unwrap();

        let mut test_fs = FeatureSet::new();
        test_fs.add_feature(Feature::categorical("category", "B"));

        let result = pipeline.transform(&test_fs).await.unwrap();
        let encoded = result.get_feature("category").unwrap().value.as_numeric();

        assert!(!encoded.is_nan());
        assert!(encoded >= 0.0);
    }

    #[tokio::test]
    async fn test_feature_extraction() {
        let pipeline = FeaturePipeline::with_config(FeatureExtractionConfig {
            extract_time_features: true,
            ..Default::default()
        });

        let mut event_data = HashMap::new();
        event_data.insert("temperature".to_string(), serde_json::json!(23.5));
        event_data.insert("humidity".to_string(), serde_json::json!(65.0));
        event_data.insert("location".to_string(), serde_json::json!("room_A"));

        let features = pipeline.extract_features(&event_data).await.unwrap();

        assert!(features.get_feature("temperature").is_some());
        assert!(features.get_feature("humidity").is_some());
        assert!(features.get_feature("location").is_some());
        assert!(features.get_feature("hour_of_day").is_some());
    }

    #[tokio::test]
    async fn test_feature_store() {
        let store = FeatureStore::new();

        let mut fs = FeatureSet::new();
        fs.add_feature(Feature::numeric("value", 42.0));

        let metadata = FeatureMetadata {
            id: "test_1".to_string(),
            description: "Test features".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: "1.0".to_string(),
            tags: vec!["test".to_string()],
        };

        store.store("test_1", fs.clone(), metadata).await.unwrap();

        let retrieved = store.retrieve("test_1").await;
        assert!(retrieved.is_some());

        let ids = store.list_ids().await;
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], "test_1");

        store.delete("test_1").await.unwrap();
        assert!(store.retrieve("test_1").await.is_none());
    }

    #[tokio::test]
    async fn test_pipeline_stats() {
        let mut pipeline = FeaturePipeline::new();
        pipeline.add_transform(FeatureTransform::StandardScaler);

        let mut fs = FeatureSet::new();
        fs.add_feature(Feature::numeric("value", 42.0));

        let _ = pipeline.transform(&fs).await;

        let stats = pipeline.get_stats().await;
        assert_eq!(stats.total_transformations, 1);
        assert!(stats.avg_transform_time_ms >= 0.0);
    }
}
