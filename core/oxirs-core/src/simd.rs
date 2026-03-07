//! SIMD operations abstraction for OxiRS
//!
//! This module provides unified SIMD operations across the OxiRS ecosystem.
//! All SIMD operations must go through this module - direct SIMD usage in other modules is forbidden.

/// Unified SIMD operations trait
pub trait SimdOps {
    /// Add two slices element-wise
    fn add(a: &[Self], b: &[Self]) -> Vec<Self>
    where
        Self: Sized;

    /// Subtract two slices element-wise
    fn sub(a: &[Self], b: &[Self]) -> Vec<Self>
    where
        Self: Sized;

    /// Multiply two slices element-wise
    fn mul(a: &[Self], b: &[Self]) -> Vec<Self>
    where
        Self: Sized;

    /// Compute dot product
    fn dot(a: &[Self], b: &[Self]) -> Self
    where
        Self: Sized;

    /// Compute cosine distance (1 - cosine_similarity)
    fn cosine_distance(a: &[Self], b: &[Self]) -> Self
    where
        Self: Sized;

    /// Compute Euclidean distance
    fn euclidean_distance(a: &[Self], b: &[Self]) -> Self
    where
        Self: Sized;

    /// Compute Manhattan distance
    fn manhattan_distance(a: &[Self], b: &[Self]) -> Self
    where
        Self: Sized;

    /// Compute L2 norm
    fn norm(a: &[Self]) -> Self
    where
        Self: Sized;

    /// Sum all elements
    fn sum(a: &[Self]) -> Self
    where
        Self: Sized;

    /// Compute mean
    fn mean(a: &[Self]) -> Self
    where
        Self: Sized;
}

// Platform-specific SIMD implementations
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
mod x86_simd;

// ARM NEON SIMD support for Apple Silicon and ARM processors
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
mod arm_simd;

// Generic scalar fallback implementation
mod scalar;

// Export the appropriate implementation based on features and platform
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
pub use x86_simd::*;

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
pub use arm_simd::*;

#[cfg(not(feature = "simd"))]
pub use scalar::*;

// Fallback to scalar on unsupported platforms
#[cfg(all(
    feature = "simd",
    not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))
))]
pub use scalar::*;

/// SIMD implementation for f32
impl SimdOps for f32 {
    fn add(a: &[Self], b: &[Self]) -> Vec<Self> {
        debug_assert_eq!(a.len(), b.len());
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        unsafe {
            x86_simd::add_f32(a, b)
        }
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        unsafe {
            arm_simd::add_f32(a, b)
        }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"),
            all(target_arch = "aarch64", feature = "simd")
        )))]
        scalar::add_f32(a, b)
    }

    fn sub(a: &[Self], b: &[Self]) -> Vec<Self> {
        debug_assert_eq!(a.len(), b.len());
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        unsafe {
            x86_simd::sub_f32(a, b)
        }
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        unsafe {
            arm_simd::sub_f32(a, b)
        }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"),
            all(target_arch = "aarch64", feature = "simd")
        )))]
        scalar::sub_f32(a, b)
    }

    fn mul(a: &[Self], b: &[Self]) -> Vec<Self> {
        debug_assert_eq!(a.len(), b.len());
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        unsafe {
            x86_simd::mul_f32(a, b)
        }
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        unsafe {
            arm_simd::mul_f32(a, b)
        }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"),
            all(target_arch = "aarch64", feature = "simd")
        )))]
        scalar::mul_f32(a, b)
    }

    fn dot(a: &[Self], b: &[Self]) -> Self {
        debug_assert_eq!(a.len(), b.len());
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        unsafe {
            x86_simd::dot_f32(a, b)
        }
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        unsafe {
            arm_simd::dot_f32(a, b)
        }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"),
            all(target_arch = "aarch64", feature = "simd")
        )))]
        scalar::dot_f32(a, b)
    }

    fn cosine_distance(a: &[Self], b: &[Self]) -> Self {
        debug_assert_eq!(a.len(), b.len());
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        unsafe {
            x86_simd::cosine_distance_f32(a, b)
        }
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        unsafe {
            arm_simd::cosine_distance_f32(a, b)
        }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"),
            all(target_arch = "aarch64", feature = "simd")
        )))]
        scalar::cosine_distance_f32(a, b)
    }

    fn euclidean_distance(a: &[Self], b: &[Self]) -> Self {
        debug_assert_eq!(a.len(), b.len());
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        unsafe {
            x86_simd::euclidean_distance_f32(a, b)
        }
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        unsafe {
            arm_simd::euclidean_distance_f32(a, b)
        }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"),
            all(target_arch = "aarch64", feature = "simd")
        )))]
        scalar::euclidean_distance_f32(a, b)
    }

    fn manhattan_distance(a: &[Self], b: &[Self]) -> Self {
        debug_assert_eq!(a.len(), b.len());
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        unsafe {
            x86_simd::manhattan_distance_f32(a, b)
        }
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        unsafe {
            arm_simd::manhattan_distance_f32(a, b)
        }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"),
            all(target_arch = "aarch64", feature = "simd")
        )))]
        scalar::manhattan_distance_f32(a, b)
    }

    fn norm(a: &[Self]) -> Self {
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        unsafe {
            x86_simd::norm_f32(a)
        }
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        unsafe {
            arm_simd::norm_f32(a)
        }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"),
            all(target_arch = "aarch64", feature = "simd")
        )))]
        scalar::norm_f32(a)
    }

    fn sum(a: &[Self]) -> Self {
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        unsafe {
            x86_simd::sum_f32(a)
        }
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        unsafe {
            arm_simd::sum_f32(a)
        }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"),
            all(target_arch = "aarch64", feature = "simd")
        )))]
        scalar::sum_f32(a)
    }

    fn mean(a: &[Self]) -> Self {
        if a.is_empty() {
            return 0.0;
        }
        Self::sum(a) / a.len() as f32
    }
}

/// SIMD implementation for f64
impl SimdOps for f64 {
    fn add(a: &[Self], b: &[Self]) -> Vec<Self> {
        debug_assert_eq!(a.len(), b.len());
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        unsafe {
            x86_simd::add_f64(a, b)
        }
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        unsafe {
            arm_simd::add_f64(a, b)
        }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"),
            all(target_arch = "aarch64", feature = "simd")
        )))]
        scalar::add_f64(a, b)
    }

    fn sub(a: &[Self], b: &[Self]) -> Vec<Self> {
        debug_assert_eq!(a.len(), b.len());
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        unsafe {
            x86_simd::sub_f64(a, b)
        }
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        unsafe {
            arm_simd::sub_f64(a, b)
        }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"),
            all(target_arch = "aarch64", feature = "simd")
        )))]
        scalar::sub_f64(a, b)
    }

    fn mul(a: &[Self], b: &[Self]) -> Vec<Self> {
        debug_assert_eq!(a.len(), b.len());
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        unsafe {
            x86_simd::mul_f64(a, b)
        }
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        unsafe {
            arm_simd::mul_f64(a, b)
        }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"),
            all(target_arch = "aarch64", feature = "simd")
        )))]
        scalar::mul_f64(a, b)
    }

    fn dot(a: &[Self], b: &[Self]) -> Self {
        debug_assert_eq!(a.len(), b.len());
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        unsafe {
            x86_simd::dot_f64(a, b)
        }
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        unsafe {
            arm_simd::dot_f64(a, b)
        }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"),
            all(target_arch = "aarch64", feature = "simd")
        )))]
        scalar::dot_f64(a, b)
    }

    fn cosine_distance(a: &[Self], b: &[Self]) -> Self {
        debug_assert_eq!(a.len(), b.len());
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        unsafe {
            x86_simd::cosine_distance_f64(a, b)
        }
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        unsafe {
            arm_simd::cosine_distance_f64(a, b)
        }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"),
            all(target_arch = "aarch64", feature = "simd")
        )))]
        scalar::cosine_distance_f64(a, b)
    }

    fn euclidean_distance(a: &[Self], b: &[Self]) -> Self {
        debug_assert_eq!(a.len(), b.len());
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        unsafe {
            x86_simd::euclidean_distance_f64(a, b)
        }
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        unsafe {
            arm_simd::euclidean_distance_f64(a, b)
        }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"),
            all(target_arch = "aarch64", feature = "simd")
        )))]
        scalar::euclidean_distance_f64(a, b)
    }

    fn manhattan_distance(a: &[Self], b: &[Self]) -> Self {
        debug_assert_eq!(a.len(), b.len());
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        unsafe {
            x86_simd::manhattan_distance_f64(a, b)
        }
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        unsafe {
            arm_simd::manhattan_distance_f64(a, b)
        }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"),
            all(target_arch = "aarch64", feature = "simd")
        )))]
        scalar::manhattan_distance_f64(a, b)
    }

    fn norm(a: &[Self]) -> Self {
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        unsafe {
            x86_simd::norm_f64(a)
        }
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        unsafe {
            arm_simd::norm_f64(a)
        }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"),
            all(target_arch = "aarch64", feature = "simd")
        )))]
        scalar::norm_f64(a)
    }

    fn sum(a: &[Self]) -> Self {
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
        unsafe {
            x86_simd::sum_f64(a)
        }
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        unsafe {
            arm_simd::sum_f64(a)
        }
        #[cfg(not(any(
            all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"),
            all(target_arch = "aarch64", feature = "simd")
        )))]
        scalar::sum_f64(a)
    }

    fn mean(a: &[Self]) -> Self {
        if a.is_empty() {
            return 0.0;
        }
        Self::sum(a) / a.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON_F32: f32 = 1e-5;
    const EPSILON_F64: f64 = 1e-10;

    // --- f32 tests ---

    #[test]
    fn test_f32_dot_product_basic() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let result = f32::dot(&a, &b);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!(
            (result - 32.0_f32).abs() < EPSILON_F32,
            "Expected 32.0, got {result}"
        );
    }

    #[test]
    fn test_f32_dot_product_zeros() {
        let a = [0.0_f32; 8];
        let b = [1.0_f32; 8];
        let result = f32::dot(&a, &b);
        assert!(
            (result - 0.0_f32).abs() < EPSILON_F32,
            "Expected 0.0, got {result}"
        );
    }

    #[test]
    fn test_f32_cosine_distance_identical_vectors() {
        let a = [1.0_f32, 0.0, 0.0];
        let b = [1.0_f32, 0.0, 0.0];
        // cosine_distance = 1 - cosine_similarity = 1 - 1 = 0
        let result = f32::cosine_distance(&a, &b);
        assert!(
            result.abs() < EPSILON_F32,
            "Identical vectors should have cosine distance 0, got {result}"
        );
    }

    #[test]
    fn test_f32_cosine_distance_orthogonal_vectors() {
        let a = [1.0_f32, 0.0, 0.0];
        let b = [0.0_f32, 1.0, 0.0];
        // cosine_similarity = 0, so cosine_distance = 1
        let result = f32::cosine_distance(&a, &b);
        assert!(
            (result - 1.0_f32).abs() < EPSILON_F32,
            "Orthogonal vectors should have cosine distance 1, got {result}"
        );
    }

    #[test]
    fn test_f32_euclidean_distance() {
        let a = [0.0_f32, 0.0, 0.0];
        let b = [3.0_f32, 4.0, 0.0];
        // sqrt(9 + 16) = 5
        let result = f32::euclidean_distance(&a, &b);
        assert!(
            (result - 5.0_f32).abs() < EPSILON_F32,
            "Expected 5.0, got {result}"
        );
    }

    #[test]
    fn test_f32_manhattan_distance() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 6.0, 8.0];
        // |1-4| + |2-6| + |3-8| = 3 + 4 + 5 = 12
        let result = f32::manhattan_distance(&a, &b);
        assert!(
            (result - 12.0_f32).abs() < EPSILON_F32,
            "Expected 12.0, got {result}"
        );
    }

    #[test]
    fn test_f32_norm_unit_vector() {
        let a = [1.0_f32, 0.0, 0.0];
        let result = f32::norm(&a);
        assert!(
            (result - 1.0_f32).abs() < EPSILON_F32,
            "Unit vector norm should be 1.0, got {result}"
        );
    }

    #[test]
    fn test_f32_norm_3_4_5() {
        let a = [3.0_f32, 4.0, 0.0];
        let result = f32::norm(&a);
        assert!(
            (result - 5.0_f32).abs() < EPSILON_F32,
            "Expected norm 5.0, got {result}"
        );
    }

    #[test]
    fn test_f32_sum_and_mean() {
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let sum = f32::sum(&a);
        let mean = f32::mean(&a);
        assert!(
            (sum - 10.0_f32).abs() < EPSILON_F32,
            "Expected sum 10.0, got {sum}"
        );
        assert!(
            (mean - 2.5_f32).abs() < EPSILON_F32,
            "Expected mean 2.5, got {mean}"
        );
    }

    #[test]
    fn test_f32_mean_empty_slice() {
        let a: [f32; 0] = [];
        let result = f32::mean(&a);
        assert!(
            (result - 0.0_f32).abs() < EPSILON_F32,
            "Mean of empty slice should be 0.0, got {result}"
        );
    }

    #[test]
    fn test_f32_add_element_wise() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let result = f32::add(&a, &b);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 5.0_f32).abs() < EPSILON_F32);
        assert!((result[1] - 7.0_f32).abs() < EPSILON_F32);
        assert!((result[2] - 9.0_f32).abs() < EPSILON_F32);
    }

    #[test]
    fn test_f32_sub_element_wise() {
        let a = [5.0_f32, 7.0, 9.0];
        let b = [1.0_f32, 2.0, 3.0];
        let result = f32::sub(&a, &b);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 4.0_f32).abs() < EPSILON_F32);
        assert!((result[1] - 5.0_f32).abs() < EPSILON_F32);
        assert!((result[2] - 6.0_f32).abs() < EPSILON_F32);
    }

    #[test]
    fn test_f32_mul_element_wise() {
        let a = [2.0_f32, 3.0, 4.0];
        let b = [5.0_f32, 6.0, 7.0];
        let result = f32::mul(&a, &b);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 10.0_f32).abs() < EPSILON_F32);
        assert!((result[1] - 18.0_f32).abs() < EPSILON_F32);
        assert!((result[2] - 28.0_f32).abs() < EPSILON_F32);
    }

    // --- f64 tests ---

    #[test]
    fn test_f64_dot_product_basic() {
        let a = [1.0_f64, 2.0, 3.0];
        let b = [4.0_f64, 5.0, 6.0];
        let result = f64::dot(&a, &b);
        assert!(
            (result - 32.0_f64).abs() < EPSILON_F64,
            "Expected 32.0, got {result}"
        );
    }

    #[test]
    fn test_f64_euclidean_distance_zero() {
        let a = [1.0_f64, 2.0, 3.0];
        let b = [1.0_f64, 2.0, 3.0];
        let result = f64::euclidean_distance(&a, &b);
        assert!(
            result.abs() < EPSILON_F64,
            "Identical vectors should have distance 0, got {result}"
        );
    }

    #[test]
    fn test_f64_cosine_distance_opposite_vectors() {
        // Opposite vectors: a = [1,0,0], b = [-1,0,0]
        // cosine_similarity = -1, cosine_distance = 1 - (-1) = 2
        let a = [1.0_f64, 0.0, 0.0];
        let b = [-1.0_f64, 0.0, 0.0];
        let result = f64::cosine_distance(&a, &b);
        assert!(
            (result - 2.0_f64).abs() < EPSILON_F64,
            "Opposite vectors should have cosine distance 2.0, got {result}"
        );
    }

    #[test]
    fn test_f64_manhattan_distance_symmetry() {
        let a = [1.0_f64, 2.0, 3.0];
        let b = [4.0_f64, 6.0, 8.0];
        let d_ab = f64::manhattan_distance(&a, &b);
        let d_ba = f64::manhattan_distance(&b, &a);
        assert!(
            (d_ab - d_ba).abs() < EPSILON_F64,
            "Manhattan distance should be symmetric"
        );
    }

    #[test]
    fn test_f64_norm_of_standard_basis() {
        let a = [0.0_f64, 0.0, 1.0, 0.0];
        let result = f64::norm(&a);
        assert!(
            (result - 1.0_f64).abs() < EPSILON_F64,
            "Norm of standard basis vector should be 1.0, got {result}"
        );
    }

    #[test]
    fn test_f64_sum_large_slice() {
        // Verify sum of arithmetic series: 1+2+...+100 = 5050
        let a: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = f64::sum(&a);
        assert!(
            (result - 5050.0_f64).abs() < EPSILON_F64,
            "Expected 5050.0, got {result}"
        );
    }

    #[test]
    fn test_f64_mean_empty_slice() {
        let a: [f64; 0] = [];
        let result = f64::mean(&a);
        assert!(
            result.abs() < EPSILON_F64,
            "Mean of empty slice should be 0.0, got {result}"
        );
    }

    #[test]
    fn test_f64_add_sub_roundtrip() {
        let a = [3.0_f64, 1.0, 4.0, 1.0, 5.0];
        let b = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let added = f64::add(&a, &b);
        let subtracted = f64::sub(added.as_slice(), &b);
        for (orig, recovered) in a.iter().zip(subtracted.iter()) {
            assert!(
                (orig - recovered).abs() < EPSILON_F64,
                "Add-sub roundtrip failed: {orig} vs {recovered}"
            );
        }
    }

    // --- Cross-type consistency tests ---

    #[test]
    fn test_euclidean_and_manhattan_triangle_inequality() {
        // For any two vectors, Euclidean distance <= Manhattan distance
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 6.0, 8.0];
        let euclidean = f32::euclidean_distance(&a, &b);
        let manhattan = f32::manhattan_distance(&a, &b);
        assert!(
            euclidean <= manhattan + EPSILON_F32,
            "Euclidean should be <= Manhattan: {euclidean} vs {manhattan}"
        );
    }

    #[test]
    fn test_cosine_distance_range() {
        // cosine_distance should be in [0, 2] for real vectors
        let a = [1.0_f32, 0.5, 0.25];
        let b = [0.5_f32, 1.0, 2.0];
        let result = f32::cosine_distance(&a, &b);
        assert!(
            result >= 0.0,
            "Cosine distance should be non-negative, got {result}"
        );
        assert!(
            result <= 2.0 + EPSILON_F32,
            "Cosine distance should be <= 2.0, got {result}"
        );
    }
}
