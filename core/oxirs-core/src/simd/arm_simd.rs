//! ARM NEON SIMD implementations for Apple Silicon and ARM processors
//!
//! This module provides optimized SIMD operations using ARM NEON instructions
//! for f32 and f64 operations, providing significant performance improvements
//! on ARM processors including Apple M1/M2/M3 chips.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Add two f32 slices using ARM NEON
///
/// # Safety
/// - Both slices `a` and `b` must have the same length
/// - Uses SIMD intrinsics that require proper memory alignment
#[inline]
pub unsafe fn add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return super::scalar::add_f32(a, b);
    }

    let len = a.len();
    let mut result = vec![0.0f32; len];
    let simd_len = len & !3; // Process 4 elements at a time

    for i in (0..simd_len).step_by(4) {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let sum = vaddq_f32(a_vec, b_vec);
        vst1q_f32(result.as_mut_ptr().add(i), sum);
    }

    // Handle remaining elements
    for i in simd_len..len {
        result[i] = a[i] + b[i];
    }

    result
}

/// Subtract two f32 slices using ARM NEON
///
/// # Safety
/// - Both slices `a` and `b` must have the same length
/// - Uses SIMD intrinsics that require proper memory alignment
#[inline]
pub unsafe fn sub_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return super::scalar::sub_f32(a, b);
    }

    let len = a.len();
    let mut result = vec![0.0f32; len];
    let simd_len = len & !3;

    for i in (0..simd_len).step_by(4) {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let diff = vsubq_f32(a_vec, b_vec);
        vst1q_f32(result.as_mut_ptr().add(i), diff);
    }

    for i in simd_len..len {
        result[i] = a[i] - b[i];
    }

    result
}

/// Multiply two f32 slices using ARM NEON
///
/// # Safety
/// - Both slices `a` and `b` must have the same length
/// - Uses SIMD intrinsics that require proper memory alignment
#[inline]
pub unsafe fn mul_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return super::scalar::mul_f32(a, b);
    }

    let len = a.len();
    let mut result = vec![0.0f32; len];
    let simd_len = len & !3;

    for i in (0..simd_len).step_by(4) {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let prod = vmulq_f32(a_vec, b_vec);
        vst1q_f32(result.as_mut_ptr().add(i), prod);
    }

    for i in simd_len..len {
        result[i] = a[i] * b[i];
    }

    result
}

/// Compute dot product using ARM NEON
///
/// # Safety
/// - Both slices `a` and `b` must have the same length
/// - Uses SIMD intrinsics that require proper memory alignment
#[inline]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return super::scalar::dot_f32(a, b);
    }

    let len = a.len();
    let simd_len = len & !3;
    let mut sum = vdupq_n_f32(0.0);

    for i in (0..simd_len).step_by(4) {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        // Use fused multiply-add for better precision and performance
        sum = vfmaq_f32(sum, a_vec, b_vec);
    }

    // Horizontal sum using vaddvq_f32 (available on ARMv8.1+)
    let mut result = vaddvq_f32(sum);

    // Handle remaining elements
    for i in simd_len..len {
        result += a[i] * b[i];
    }

    result
}

/// Compute cosine distance using ARM NEON
///
/// # Safety
/// - Both slices `a` and `b` must have the same length
/// - Uses SIMD intrinsics that require proper memory alignment
/// - Returns 1.0 if either vector has zero norm
#[inline]
pub unsafe fn cosine_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return super::scalar::cosine_distance_f32(a, b);
    }

    let len = a.len();
    let simd_len = len & !3;

    let mut dot_sum = vdupq_n_f32(0.0);
    let mut norm_a_sum = vdupq_n_f32(0.0);
    let mut norm_b_sum = vdupq_n_f32(0.0);

    for i in (0..simd_len).step_by(4) {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));

        // Compute dot product
        dot_sum = vfmaq_f32(dot_sum, a_vec, b_vec);

        // Compute norms
        norm_a_sum = vfmaq_f32(norm_a_sum, a_vec, a_vec);
        norm_b_sum = vfmaq_f32(norm_b_sum, b_vec, b_vec);
    }

    let dot = vaddvq_f32(dot_sum);
    let norm_a = vaddvq_f32(norm_a_sum);
    let norm_b = vaddvq_f32(norm_b_sum);

    let (mut dot_scalar, mut norm_a_scalar, mut norm_b_scalar) = (dot, norm_a, norm_b);
    for i in simd_len..len {
        dot_scalar += a[i] * b[i];
        norm_a_scalar += a[i] * a[i];
        norm_b_scalar += b[i] * b[i];
    }

    let norm_a_sqrt = norm_a_scalar.sqrt();
    let norm_b_sqrt = norm_b_scalar.sqrt();

    if norm_a_sqrt == 0.0 || norm_b_sqrt == 0.0 {
        1.0
    } else {
        1.0 - (dot_scalar / (norm_a_sqrt * norm_b_sqrt))
    }
}

/// Compute Euclidean distance using ARM NEON
///
/// # Safety
/// - Both slices `a` and `b` must have the same length
/// - Uses SIMD intrinsics that require proper memory alignment
#[inline]
pub unsafe fn euclidean_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return super::scalar::euclidean_distance_f32(a, b);
    }

    let len = a.len();
    let simd_len = len & !3;
    let mut sum = vdupq_n_f32(0.0);

    for i in (0..simd_len).step_by(4) {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));

        let diff = vsubq_f32(a_vec, b_vec);
        sum = vfmaq_f32(sum, diff, diff);
    }

    let mut result = vaddvq_f32(sum);

    for i in simd_len..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result.sqrt()
}

/// Compute Manhattan distance using ARM NEON
///
/// # Safety
/// - Both slices `a` and `b` must have the same length
/// - Uses SIMD intrinsics that require proper memory alignment
#[inline]
pub unsafe fn manhattan_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return super::scalar::manhattan_distance_f32(a, b);
    }

    let len = a.len();
    let simd_len = len & !3;
    let mut sum = vdupq_n_f32(0.0);

    for i in (0..simd_len).step_by(4) {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));

        let diff = vsubq_f32(a_vec, b_vec);
        let abs_diff = vabsq_f32(diff);
        sum = vaddq_f32(sum, abs_diff);
    }

    let mut result = vaddvq_f32(sum);

    for i in simd_len..len {
        result += (a[i] - b[i]).abs();
    }

    result
}

/// Compute L2 norm using ARM NEON
///
/// # Safety
/// - Uses SIMD intrinsics that require proper memory alignment
#[inline]
pub unsafe fn norm_f32(a: &[f32]) -> f32 {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return super::scalar::norm_f32(a);
    }

    let len = a.len();
    let simd_len = len & !3;
    let mut sum = vdupq_n_f32(0.0);

    for i in (0..simd_len).step_by(4) {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        sum = vfmaq_f32(sum, a_vec, a_vec);
    }

    let mut result = vaddvq_f32(sum);

    for &val in a.iter().skip(simd_len) {
        result += val * val;
    }

    result.sqrt()
}

/// Sum all elements using ARM NEON
///
/// # Safety
/// - Uses SIMD intrinsics that require proper memory alignment
#[inline]
pub unsafe fn sum_f32(a: &[f32]) -> f32 {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return super::scalar::sum_f32(a);
    }

    let len = a.len();
    let simd_len = len & !3;
    let mut sum = vdupq_n_f32(0.0);

    for i in (0..simd_len).step_by(4) {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        sum = vaddq_f32(sum, a_vec);
    }

    let mut result = vaddvq_f32(sum);

    for &val in a.iter().skip(simd_len) {
        result += val;
    }

    result
}

// f64 implementations using ARM NEON (2 elements at a time)

/// Add two f64 slices using ARM NEON
///
/// # Safety
/// - Both slices `a` and `b` must have the same length
/// - Uses SIMD intrinsics that require proper memory alignment
#[inline]
pub unsafe fn add_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return super::scalar::add_f64(a, b);
    }

    let len = a.len();
    let mut result = vec![0.0f64; len];
    let simd_len = len & !1; // Process 2 elements at a time

    for i in (0..simd_len).step_by(2) {
        let a_vec = vld1q_f64(a.as_ptr().add(i));
        let b_vec = vld1q_f64(b.as_ptr().add(i));
        let sum = vaddq_f64(a_vec, b_vec);
        vst1q_f64(result.as_mut_ptr().add(i), sum);
    }

    for i in simd_len..len {
        result[i] = a[i] + b[i];
    }

    result
}

/// Subtract two f64 slices using ARM NEON
///
/// # Safety
/// - Both slices `a` and `b` must have the same length
/// - Uses SIMD intrinsics that require proper memory alignment
#[inline]
pub unsafe fn sub_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return super::scalar::sub_f64(a, b);
    }

    let len = a.len();
    let mut result = vec![0.0f64; len];
    let simd_len = len & !1;

    for i in (0..simd_len).step_by(2) {
        let a_vec = vld1q_f64(a.as_ptr().add(i));
        let b_vec = vld1q_f64(b.as_ptr().add(i));
        let diff = vsubq_f64(a_vec, b_vec);
        vst1q_f64(result.as_mut_ptr().add(i), diff);
    }

    for i in simd_len..len {
        result[i] = a[i] - b[i];
    }

    result
}

/// Multiply two f64 slices using ARM NEON
///
/// # Safety
/// - Both slices `a` and `b` must have the same length
/// - Uses SIMD intrinsics that require proper memory alignment
#[inline]
pub unsafe fn mul_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return super::scalar::mul_f64(a, b);
    }

    let len = a.len();
    let mut result = vec![0.0f64; len];
    let simd_len = len & !1;

    for i in (0..simd_len).step_by(2) {
        let a_vec = vld1q_f64(a.as_ptr().add(i));
        let b_vec = vld1q_f64(b.as_ptr().add(i));
        let prod = vmulq_f64(a_vec, b_vec);
        vst1q_f64(result.as_mut_ptr().add(i), prod);
    }

    for i in simd_len..len {
        result[i] = a[i] * b[i];
    }

    result
}

/// Compute dot product for f64 using ARM NEON
///
/// # Safety
/// - Both slices `a` and `b` must have the same length
/// - Uses SIMD intrinsics that require proper memory alignment
#[inline]
pub unsafe fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return super::scalar::dot_f64(a, b);
    }

    let len = a.len();
    let simd_len = len & !1;
    let mut sum = vdupq_n_f64(0.0);

    for i in (0..simd_len).step_by(2) {
        let a_vec = vld1q_f64(a.as_ptr().add(i));
        let b_vec = vld1q_f64(b.as_ptr().add(i));
        sum = vfmaq_f64(sum, a_vec, b_vec);
    }

    let mut result = vaddvq_f64(sum);

    for i in simd_len..len {
        result += a[i] * b[i];
    }

    result
}

/// Compute cosine distance for f64 using ARM NEON
///
/// # Safety
/// - Both slices `a` and `b` must have the same length
/// - Uses SIMD intrinsics that require proper memory alignment
/// - Returns 1.0 if either vector has zero norm
#[inline]
pub unsafe fn cosine_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return super::scalar::cosine_distance_f64(a, b);
    }

    let dot = dot_f64(a, b);
    let norm_a = norm_f64(a);
    let norm_b = norm_f64(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        1.0 - (dot / (norm_a * norm_b))
    }
}

/// Compute Euclidean distance for f64 using ARM NEON
///
/// # Safety
/// - Both slices `a` and `b` must have the same length
/// - Uses SIMD intrinsics that require proper memory alignment
#[inline]
pub unsafe fn euclidean_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return super::scalar::euclidean_distance_f64(a, b);
    }

    let len = a.len();
    let simd_len = len & !1;
    let mut sum = vdupq_n_f64(0.0);

    for i in (0..simd_len).step_by(2) {
        let a_vec = vld1q_f64(a.as_ptr().add(i));
        let b_vec = vld1q_f64(b.as_ptr().add(i));

        let diff = vsubq_f64(a_vec, b_vec);
        sum = vfmaq_f64(sum, diff, diff);
    }

    let mut result = vaddvq_f64(sum);

    for i in simd_len..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result.sqrt()
}

/// Compute Manhattan distance for f64 using ARM NEON
///
/// # Safety
/// - Both slices `a` and `b` must have the same length
/// - Uses SIMD intrinsics that require proper memory alignment
#[inline]
pub unsafe fn manhattan_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return super::scalar::manhattan_distance_f64(a, b);
    }

    let len = a.len();
    let simd_len = len & !1;
    let mut sum = vdupq_n_f64(0.0);

    for i in (0..simd_len).step_by(2) {
        let a_vec = vld1q_f64(a.as_ptr().add(i));
        let b_vec = vld1q_f64(b.as_ptr().add(i));

        let diff = vsubq_f64(a_vec, b_vec);
        let abs_diff = vabsq_f64(diff);
        sum = vaddq_f64(sum, abs_diff);
    }

    let mut result = vaddvq_f64(sum);

    for i in simd_len..len {
        result += (a[i] - b[i]).abs();
    }

    result
}

/// Compute L2 norm for f64 using ARM NEON
///
/// # Safety
/// - Uses SIMD intrinsics that require proper memory alignment
#[inline]
pub unsafe fn norm_f64(a: &[f64]) -> f64 {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return super::scalar::norm_f64(a);
    }

    let len = a.len();
    let simd_len = len & !1;
    let mut sum = vdupq_n_f64(0.0);

    for i in (0..simd_len).step_by(2) {
        let a_vec = vld1q_f64(a.as_ptr().add(i));
        sum = vfmaq_f64(sum, a_vec, a_vec);
    }

    let mut result = vaddvq_f64(sum);

    for &val in a.iter().skip(simd_len) {
        result += val * val;
    }

    result.sqrt()
}

/// Sum all elements for f64 using ARM NEON
///
/// # Safety
/// - Uses SIMD intrinsics that require proper memory alignment
#[inline]
pub unsafe fn sum_f64(a: &[f64]) -> f64 {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return super::scalar::sum_f64(a);
    }

    let len = a.len();
    let simd_len = len & !1;
    let mut sum = vdupq_n_f64(0.0);

    for i in (0..simd_len).step_by(2) {
        let a_vec = vld1q_f64(a.as_ptr().add(i));
        sum = vaddq_f64(sum, a_vec);
    }

    let mut result = vaddvq_f64(sum);

    for &val in a.iter().skip(simd_len) {
        result += val;
    }

    result
}
