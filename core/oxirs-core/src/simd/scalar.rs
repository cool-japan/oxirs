//! Scalar fallback implementations for SIMD operations

/// Add two f32 slices element-wise
pub fn add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// Subtract two f32 slices element-wise
pub fn sub_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// Multiply two f32 slices element-wise
pub fn mul_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}

/// Compute dot product of two f32 slices
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Compute cosine distance between two f32 slices
pub fn cosine_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_f32(a, b);
    let norm_a = norm_f32(a);
    let norm_b = norm_f32(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        1.0 - (dot / (norm_a * norm_b))
    }
}

/// Compute Euclidean distance between two f32 slices
pub fn euclidean_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Compute Manhattan distance between two f32 slices
pub fn manhattan_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

/// Compute L2 norm of f32 slice
pub fn norm_f32(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Sum all elements in f32 slice
pub fn sum_f32(a: &[f32]) -> f32 {
    a.iter().sum()
}

// f64 implementations

/// Add two f64 slices element-wise
pub fn add_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// Subtract two f64 slices element-wise
pub fn sub_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// Multiply two f64 slices element-wise
pub fn mul_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}

/// Compute dot product of two f64 slices
pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Compute cosine distance between two f64 slices
pub fn cosine_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    let dot = dot_f64(a, b);
    let norm_a = norm_f64(a);
    let norm_b = norm_f64(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        1.0 - (dot / (norm_a * norm_b))
    }
}

/// Compute Euclidean distance between two f64 slices
pub fn euclidean_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute Manhattan distance between two f64 slices
pub fn manhattan_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

/// Compute L2 norm of f64 slice
pub fn norm_f64(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Sum all elements in f64 slice
pub fn sum_f64(a: &[f64]) -> f64 {
    a.iter().sum()
}
