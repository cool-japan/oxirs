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
