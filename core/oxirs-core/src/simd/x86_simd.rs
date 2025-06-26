//! x86/x86_64 SIMD implementations using AVX2

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Add two f32 slices using AVX2
#[inline]
pub unsafe fn add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    if !is_x86_feature_detected!("avx2") {
        return super::scalar::add_f32(a, b);
    }
    
    let len = a.len();
    let mut result = vec![0.0f32; len];
    let simd_len = len & !7;
    
    for i in (0..simd_len).step_by(8) {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let sum = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), sum);
    }
    
    for i in simd_len..len {
        result[i] = a[i] + b[i];
    }
    
    result
}

/// Subtract two f32 slices using AVX2
#[inline]
pub unsafe fn sub_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    if !is_x86_feature_detected!("avx2") {
        return super::scalar::sub_f32(a, b);
    }
    
    let len = a.len();
    let mut result = vec![0.0f32; len];
    let simd_len = len & !7;
    
    for i in (0..simd_len).step_by(8) {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(a_vec, b_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), diff);
    }
    
    for i in simd_len..len {
        result[i] = a[i] - b[i];
    }
    
    result
}

/// Multiply two f32 slices using AVX2
#[inline]
pub unsafe fn mul_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    if !is_x86_feature_detected!("avx2") {
        return super::scalar::mul_f32(a, b);
    }
    
    let len = a.len();
    let mut result = vec![0.0f32; len];
    let simd_len = len & !7;
    
    for i in (0..simd_len).step_by(8) {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let prod = _mm256_mul_ps(a_vec, b_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), prod);
    }
    
    for i in simd_len..len {
        result[i] = a[i] * b[i];
    }
    
    result
}

/// Compute dot product using AVX2
#[inline]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    if !is_x86_feature_detected!("avx2") {
        return super::scalar::dot_f32(a, b);
    }
    
    let len = a.len();
    let simd_len = len & !7;
    let mut sum = _mm256_setzero_ps();
    
    for i in (0..simd_len).step_by(8) {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let prod = _mm256_mul_ps(a_vec, b_vec);
        sum = _mm256_add_ps(sum, prod);
    }
    
    // Horizontal sum
    let mut result = horizontal_sum_avx2(sum);
    
    // Handle remaining elements
    for i in simd_len..len {
        result += a[i] * b[i];
    }
    
    result
}

/// Compute cosine distance using AVX2
#[inline]
pub unsafe fn cosine_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    if !is_x86_feature_detected!("avx2") {
        return super::scalar::cosine_distance_f32(a, b);
    }
    
    let len = a.len();
    let simd_len = len & !7;
    
    let mut dot_sum = _mm256_setzero_ps();
    let mut norm_a_sum = _mm256_setzero_ps();
    let mut norm_b_sum = _mm256_setzero_ps();
    
    for i in (0..simd_len).step_by(8) {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        
        let prod = _mm256_mul_ps(a_vec, b_vec);
        dot_sum = _mm256_add_ps(dot_sum, prod);
        
        let a_squared = _mm256_mul_ps(a_vec, a_vec);
        let b_squared = _mm256_mul_ps(b_vec, b_vec);
        norm_a_sum = _mm256_add_ps(norm_a_sum, a_squared);
        norm_b_sum = _mm256_add_ps(norm_b_sum, b_squared);
    }
    
    let dot = horizontal_sum_avx2(dot_sum);
    let norm_a = horizontal_sum_avx2(norm_a_sum);
    let norm_b = horizontal_sum_avx2(norm_b_sum);
    
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

/// Compute Euclidean distance using AVX2
#[inline]
pub unsafe fn euclidean_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    if !is_x86_feature_detected!("avx2") {
        return super::scalar::euclidean_distance_f32(a, b);
    }
    
    let len = a.len();
    let simd_len = len & !7;
    let mut sum = _mm256_setzero_ps();
    
    for i in (0..simd_len).step_by(8) {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        
        let diff = _mm256_sub_ps(a_vec, b_vec);
        let squared = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, squared);
    }
    
    let mut result = horizontal_sum_avx2(sum);
    
    for i in simd_len..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }
    
    result.sqrt()
}

/// Compute Manhattan distance using AVX2
#[inline]
pub unsafe fn manhattan_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    if !is_x86_feature_detected!("avx2") {
        return super::scalar::manhattan_distance_f32(a, b);
    }
    
    let len = a.len();
    let simd_len = len & !7;
    let mut sum = _mm256_setzero_ps();
    
    for i in (0..simd_len).step_by(8) {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        
        let diff = _mm256_sub_ps(a_vec, b_vec);
        // Absolute value using bit manipulation
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFFFFFF));
        let abs_diff = _mm256_and_ps(diff, abs_mask);
        sum = _mm256_add_ps(sum, abs_diff);
    }
    
    let mut result = horizontal_sum_avx2(sum);
    
    for i in simd_len..len {
        result += (a[i] - b[i]).abs();
    }
    
    result
}

/// Compute L2 norm using AVX2
#[inline]
pub unsafe fn norm_f32(a: &[f32]) -> f32 {
    if !is_x86_feature_detected!("avx2") {
        return super::scalar::norm_f32(a);
    }
    
    let len = a.len();
    let simd_len = len & !7;
    let mut sum = _mm256_setzero_ps();
    
    for i in (0..simd_len).step_by(8) {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let squared = _mm256_mul_ps(a_vec, a_vec);
        sum = _mm256_add_ps(sum, squared);
    }
    
    let mut result = horizontal_sum_avx2(sum);
    
    for i in simd_len..len {
        result += a[i] * a[i];
    }
    
    result.sqrt()
}

/// Sum all elements using AVX2
#[inline]
pub unsafe fn sum_f32(a: &[f32]) -> f32 {
    if !is_x86_feature_detected!("avx2") {
        return super::scalar::sum_f32(a);
    }
    
    let len = a.len();
    let simd_len = len & !7;
    let mut sum = _mm256_setzero_ps();
    
    for i in (0..simd_len).step_by(8) {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        sum = _mm256_add_ps(sum, a_vec);
    }
    
    let mut result = horizontal_sum_avx2(sum);
    
    for i in simd_len..len {
        result += a[i];
    }
    
    result
}

// f64 implementations (using AVX2 for 4 elements at a time)

/// Add two f64 slices using AVX2
#[inline]
pub unsafe fn add_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    if !is_x86_feature_detected!("avx2") {
        return super::scalar::add_f64(a, b);
    }
    
    let len = a.len();
    let mut result = vec![0.0f64; len];
    let simd_len = len & !3;
    
    for i in (0..simd_len).step_by(4) {
        let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
        let sum = _mm256_add_pd(a_vec, b_vec);
        _mm256_storeu_pd(result.as_mut_ptr().add(i), sum);
    }
    
    for i in simd_len..len {
        result[i] = a[i] + b[i];
    }
    
    result
}

/// Subtract two f64 slices using AVX2
#[inline]
pub unsafe fn sub_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    if !is_x86_feature_detected!("avx2") {
        return super::scalar::sub_f64(a, b);
    }
    
    let len = a.len();
    let mut result = vec![0.0f64; len];
    let simd_len = len & !3;
    
    for i in (0..simd_len).step_by(4) {
        let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
        let diff = _mm256_sub_pd(a_vec, b_vec);
        _mm256_storeu_pd(result.as_mut_ptr().add(i), diff);
    }
    
    for i in simd_len..len {
        result[i] = a[i] - b[i];
    }
    
    result
}

/// Multiply two f64 slices using AVX2
#[inline]
pub unsafe fn mul_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    if !is_x86_feature_detected!("avx2") {
        return super::scalar::mul_f64(a, b);
    }
    
    let len = a.len();
    let mut result = vec![0.0f64; len];
    let simd_len = len & !3;
    
    for i in (0..simd_len).step_by(4) {
        let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
        let prod = _mm256_mul_pd(a_vec, b_vec);
        _mm256_storeu_pd(result.as_mut_ptr().add(i), prod);
    }
    
    for i in simd_len..len {
        result[i] = a[i] * b[i];
    }
    
    result
}

/// Compute dot product for f64 using AVX2
#[inline]
pub unsafe fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    if !is_x86_feature_detected!("avx2") {
        return super::scalar::dot_f64(a, b);
    }
    
    let len = a.len();
    let simd_len = len & !3;
    let mut sum = _mm256_setzero_pd();
    
    for i in (0..simd_len).step_by(4) {
        let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
        let prod = _mm256_mul_pd(a_vec, b_vec);
        sum = _mm256_add_pd(sum, prod);
    }
    
    let mut result = horizontal_sum_avx2_f64(sum);
    
    for i in simd_len..len {
        result += a[i] * b[i];
    }
    
    result
}

/// Compute cosine distance for f64 using AVX2
#[inline]
pub unsafe fn cosine_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    if !is_x86_feature_detected!("avx2") {
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

/// Compute Euclidean distance for f64 using AVX2
#[inline]
pub unsafe fn euclidean_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    if !is_x86_feature_detected!("avx2") {
        return super::scalar::euclidean_distance_f64(a, b);
    }
    
    let len = a.len();
    let simd_len = len & !3;
    let mut sum = _mm256_setzero_pd();
    
    for i in (0..simd_len).step_by(4) {
        let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
        
        let diff = _mm256_sub_pd(a_vec, b_vec);
        let squared = _mm256_mul_pd(diff, diff);
        sum = _mm256_add_pd(sum, squared);
    }
    
    let mut result = horizontal_sum_avx2_f64(sum);
    
    for i in simd_len..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }
    
    result.sqrt()
}

/// Compute Manhattan distance for f64 using AVX2
#[inline]
pub unsafe fn manhattan_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    if !is_x86_feature_detected!("avx2") {
        return super::scalar::manhattan_distance_f64(a, b);
    }
    
    let len = a.len();
    let simd_len = len & !3;
    let mut sum = _mm256_setzero_pd();
    
    // Mask for absolute value
    let abs_mask = _mm256_set1_pd(f64::from_bits(0x7FFFFFFFFFFFFFFF));
    
    for i in (0..simd_len).step_by(4) {
        let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
        
        let diff = _mm256_sub_pd(a_vec, b_vec);
        let abs_diff = _mm256_and_pd(diff, abs_mask);
        sum = _mm256_add_pd(sum, abs_diff);
    }
    
    let mut result = horizontal_sum_avx2_f64(sum);
    
    for i in simd_len..len {
        result += (a[i] - b[i]).abs();
    }
    
    result
}

/// Compute L2 norm for f64 using AVX2
#[inline]
pub unsafe fn norm_f64(a: &[f64]) -> f64 {
    if !is_x86_feature_detected!("avx2") {
        return super::scalar::norm_f64(a);
    }
    
    let len = a.len();
    let simd_len = len & !3;
    let mut sum = _mm256_setzero_pd();
    
    for i in (0..simd_len).step_by(4) {
        let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
        let squared = _mm256_mul_pd(a_vec, a_vec);
        sum = _mm256_add_pd(sum, squared);
    }
    
    let mut result = horizontal_sum_avx2_f64(sum);
    
    for i in simd_len..len {
        result += a[i] * a[i];
    }
    
    result.sqrt()
}

/// Sum all elements for f64 using AVX2
#[inline]
pub unsafe fn sum_f64(a: &[f64]) -> f64 {
    if !is_x86_feature_detected!("avx2") {
        return super::scalar::sum_f64(a);
    }
    
    let len = a.len();
    let simd_len = len & !3;
    let mut sum = _mm256_setzero_pd();
    
    for i in (0..simd_len).step_by(4) {
        let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
        sum = _mm256_add_pd(sum, a_vec);
    }
    
    let mut result = horizontal_sum_avx2_f64(sum);
    
    for i in simd_len..len {
        result += a[i];
    }
    
    result
}

// Helper functions

/// Horizontal sum for AVX2 __m256 (f32) vectors
#[inline]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    let low = _mm256_castps256_ps128(v);
    let high = _mm256_extractf128_ps(v, 1);
    
    let sum128 = _mm_add_ps(low, high);
    
    let shuf = _mm_shuffle_ps(sum128, sum128, 0x0E);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    
    _mm_cvtss_f32(result)
}

/// Horizontal sum for AVX2 __m256d (f64) vectors
#[inline]
unsafe fn horizontal_sum_avx2_f64(v: __m256d) -> f64 {
    let low = _mm256_castpd256_pd128(v);
    let high = _mm256_extractf128_pd(v, 1);
    
    let sum128 = _mm_add_pd(low, high);
    
    let high64 = _mm_unpackhi_pd(sum128, sum128);
    let sum64 = _mm_add_sd(sum128, high64);
    
    _mm_cvtsd_f64(sum64)
}