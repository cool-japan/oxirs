//! Random utility functions
//!
//! Temporary utilities for random number generation until scirs2-core beta.4
//! provides full distribution support in public API.

use scirs2_core::random::{Random, Rng};

/// Sample from a normal distribution with given mean and standard deviation
/// using Box-Muller transform
#[inline]
pub fn sample_normal<R: Rng>(rng: &mut R, mean: f32, std: f32) -> f32 {
    let u1: f32 = rng.gen::<f32>();
    let u2: f32 = rng.gen::<f32>();
    let z = (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * std::f32::consts::PI * u2).cos();
    mean + z * std
}

/// Sample from a uniform distribution with given bounds
#[inline]
pub fn sample_uniform<R: Rng>(rng: &mut R, low: f32, high: f32) -> f32 {
    low + rng.gen::<f32>() * (high - low)
}

/// Normal distribution sampler (temporary replacement for scirs2_core::random::distributions::Normal)
pub struct NormalSampler {
    mean: f32,
    std: f32,
}

impl NormalSampler {
    pub fn new(mean: f32, std: f32) -> Result<Self, String> {
        if std <= 0.0 {
            return Err(format!("Standard deviation must be positive, got {}", std));
        }
        Ok(Self { mean, std })
    }

    pub fn sample<R: Rng>(&self, rng: &mut R) -> f32 {
        sample_normal(rng, self.mean, self.std)
    }
}

/// Uniform distribution sampler
pub struct UniformSampler {
    low: f32,
    high: f32,
}

impl UniformSampler {
    pub fn new(low: f32, high: f32) -> Result<Self, String> {
        if low >= high {
            return Err(format!("Low must be less than high, got {} and {}", low, high));
        }
        Ok(Self { low, high })
    }

    pub fn sample<R: Rng>(&self, rng: &mut R) -> f32 {
        sample_uniform(rng, self.low, self.high)
    }
}