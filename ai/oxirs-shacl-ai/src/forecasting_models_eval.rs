//! Evaluation metrics and accuracy functions for forecasting models

use crate::forecasting_models_types::ModelAccuracyMetrics;
use crate::ShaclAiError;

/// Compute mean absolute error between predictions and actuals
pub fn compute_mae(predictions: &[f64], actuals: &[f64]) -> Result<f64, ShaclAiError> {
    if predictions.len() != actuals.len() {
        return Err(ShaclAiError::PredictiveAnalytics(
            "Predictions and actuals must have the same length".to_string(),
        ));
    }
    if predictions.is_empty() {
        return Err(ShaclAiError::PredictiveAnalytics(
            "Cannot compute MAE on empty data".to_string(),
        ));
    }
    let mae = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(&p, &a)| (p - a).abs())
        .sum::<f64>()
        / predictions.len() as f64;
    Ok(mae)
}

/// Compute mean squared error
pub fn compute_mse(predictions: &[f64], actuals: &[f64]) -> Result<f64, ShaclAiError> {
    if predictions.len() != actuals.len() {
        return Err(ShaclAiError::PredictiveAnalytics(
            "Predictions and actuals must have the same length".to_string(),
        ));
    }
    if predictions.is_empty() {
        return Err(ShaclAiError::PredictiveAnalytics(
            "Cannot compute MSE on empty data".to_string(),
        ));
    }
    let mse = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(&p, &a)| (p - a).powi(2))
        .sum::<f64>()
        / predictions.len() as f64;
    Ok(mse)
}

/// Compute root mean squared error
pub fn compute_rmse(predictions: &[f64], actuals: &[f64]) -> Result<f64, ShaclAiError> {
    let mse = compute_mse(predictions, actuals)?;
    Ok(mse.sqrt())
}

/// Compute mean absolute percentage error
pub fn compute_mape(predictions: &[f64], actuals: &[f64]) -> Result<f64, ShaclAiError> {
    if predictions.len() != actuals.len() {
        return Err(ShaclAiError::PredictiveAnalytics(
            "Predictions and actuals must have the same length".to_string(),
        ));
    }
    if predictions.is_empty() {
        return Err(ShaclAiError::PredictiveAnalytics(
            "Cannot compute MAPE on empty data".to_string(),
        ));
    }
    let mape = predictions
        .iter()
        .zip(actuals.iter())
        .filter(|(_, &a)| a != 0.0)
        .map(|(&p, &a)| ((p - a).abs() / a.abs()) * 100.0)
        .sum::<f64>()
        / predictions.len() as f64;
    Ok(mape)
}

/// Compute R-squared (coefficient of determination)
pub fn compute_r_squared(predictions: &[f64], actuals: &[f64]) -> Result<f64, ShaclAiError> {
    if predictions.len() != actuals.len() {
        return Err(ShaclAiError::PredictiveAnalytics(
            "Predictions and actuals must have the same length".to_string(),
        ));
    }
    if actuals.is_empty() {
        return Err(ShaclAiError::PredictiveAnalytics(
            "Cannot compute R² on empty data".to_string(),
        ));
    }
    let actual_mean: f64 = actuals.iter().sum::<f64>() / actuals.len() as f64;
    let ss_tot: f64 = actuals.iter().map(|&x| (x - actual_mean).powi(2)).sum();
    let ss_res: f64 = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(&p, &a)| (a - p).powi(2))
        .sum();

    if ss_tot > 0.0 {
        Ok(1.0 - (ss_res / ss_tot))
    } else {
        Ok(0.0)
    }
}

/// Compute all accuracy metrics at once
pub fn compute_all_metrics(
    predictions: &[f64],
    actuals: &[f64],
) -> Result<ModelAccuracyMetrics, ShaclAiError> {
    let mae = compute_mae(predictions, actuals)?;
    let mse = compute_mse(predictions, actuals)?;
    let rmse = compute_rmse(predictions, actuals)?;
    let mape = compute_mape(predictions, actuals)?;
    let r_squared = compute_r_squared(predictions, actuals)?;
    let accuracy_score = (100.0 - mape.min(100.0)) / 100.0;

    Ok(ModelAccuracyMetrics {
        mean_absolute_error: mae,
        mean_squared_error: mse,
        root_mean_squared_error: rmse,
        mean_absolute_percentage_error: mape,
        r_squared,
        accuracy_score,
    })
}

/// Directional accuracy: percentage of predictions with correct sign of change
pub fn compute_directional_accuracy(
    predictions: &[f64],
    actuals: &[f64],
) -> Result<f64, ShaclAiError> {
    if predictions.len() < 2 || actuals.len() < 2 {
        return Err(ShaclAiError::PredictiveAnalytics(
            "Need at least 2 data points for directional accuracy".to_string(),
        ));
    }
    if predictions.len() != actuals.len() {
        return Err(ShaclAiError::PredictiveAnalytics(
            "Predictions and actuals must have the same length".to_string(),
        ));
    }

    let correct_directions = predictions
        .windows(2)
        .zip(actuals.windows(2))
        .filter(|(pred_w, actual_w)| {
            let pred_dir = pred_w[1] - pred_w[0];
            let actual_dir = actual_w[1] - actual_w[0];
            pred_dir * actual_dir > 0.0
        })
        .count();

    let total_pairs = predictions.len() - 1;
    Ok(correct_directions as f64 / total_pairs as f64)
}

/// Theil's U statistic for comparing forecast to naive forecast
pub fn compute_theils_u(predictions: &[f64], actuals: &[f64]) -> Result<f64, ShaclAiError> {
    if predictions.len() != actuals.len() {
        return Err(ShaclAiError::PredictiveAnalytics(
            "Predictions and actuals must have the same length".to_string(),
        ));
    }
    if actuals.len() < 2 {
        return Err(ShaclAiError::PredictiveAnalytics(
            "Need at least 2 data points for Theil's U".to_string(),
        ));
    }

    // Numerator: RMSE of forecast
    let forecast_mse: f64 = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(&p, &a)| ((p - a) / a).powi(2))
        .sum::<f64>()
        / predictions.len() as f64;

    // Denominator: RMSE of naive (random walk) forecast
    let naive_mse: f64 = actuals
        .windows(2)
        .map(|w| ((w[1] - w[0]) / w[0]).powi(2))
        .sum::<f64>()
        / (actuals.len() - 1) as f64;

    if naive_mse > 0.0 {
        Ok((forecast_mse / naive_mse).sqrt())
    } else {
        Ok(f64::INFINITY)
    }
}
