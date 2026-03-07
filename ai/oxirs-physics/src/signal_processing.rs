//! # Signal Processing Utilities (Time-Domain)
//!
//! Provides time-domain signal processing: moving average, EMA, Savitzky–Golay,
//! RMS, zero-crossings, autocorrelation, convolution, normalisation, threshold
//! detection, up/down-sampling, and SNR computation.

/// A sampled signal with an associated sample rate.
#[derive(Debug, Clone)]
pub struct Signal {
    /// Sample values.
    pub samples: Vec<f64>,
    /// Sample rate in Hz.
    pub sample_rate_hz: f64,
}

impl Signal {
    /// Create a new signal.
    pub fn new(samples: Vec<f64>, sample_rate_hz: f64) -> Self {
        Self {
            samples,
            sample_rate_hz,
        }
    }
}

/// FIR / IIR filter coefficients (Direct Form I / Direct Form II).
#[derive(Debug, Clone)]
pub struct FilterCoeffs {
    /// Numerator (feedforward) coefficients b₀, b₁, …
    pub b: Vec<f64>,
    /// Denominator (feedback) coefficients a₀, a₁, …
    /// a\[0\] is normalised to 1 in most representations.
    pub a: Vec<f64>,
}

impl FilterCoeffs {
    /// Create FIR filter coefficients (unity denominator).
    pub fn fir(b: Vec<f64>) -> Self {
        Self { b, a: vec![1.0] }
    }

    /// Create IIR filter coefficients.
    pub fn iir(b: Vec<f64>, a: Vec<f64>) -> Self {
        Self { b, a }
    }
}

/// Collection of time-domain signal processing routines.
pub struct SignalProcessing;

impl SignalProcessing {
    // -----------------------------------------------------------------
    // Basic statistics / properties
    // -----------------------------------------------------------------

    /// Compute Root-Mean-Square of `signal`.
    ///
    /// Returns 0.0 for empty input.
    pub fn rms(signal: &[f64]) -> f64 {
        if signal.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = signal.iter().map(|&x| x * x).sum();
        (sum_sq / signal.len() as f64).sqrt()
    }

    /// Peak-to-peak amplitude (max − min).
    ///
    /// Returns 0.0 for empty input.
    pub fn peak_to_peak(signal: &[f64]) -> f64 {
        if signal.is_empty() {
            return 0.0;
        }
        let max = signal
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min = signal
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        max - min
    }

    /// Count zero-crossings (sign changes between consecutive samples).
    pub fn zero_crossings(signal: &[f64]) -> usize {
        if signal.len() < 2 {
            return 0;
        }
        signal
            .windows(2)
            .filter(|w| w[0] * w[1] < 0.0)
            .count()
    }

    // -----------------------------------------------------------------
    // Smoothing / filtering
    // -----------------------------------------------------------------

    /// Symmetric moving average with `window` samples.
    ///
    /// Output length equals input length; edges are handled by shrinking the
    /// window symmetrically.
    pub fn moving_average(signal: &[f64], window: usize) -> Vec<f64> {
        if signal.is_empty() || window == 0 {
            return signal.to_vec();
        }
        let w = window.min(signal.len());
        signal
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let half = w / 2;
                let start = i.saturating_sub(half);
                let end = (i + half + 1).min(signal.len());
                let slice = &signal[start..end];
                slice.iter().sum::<f64>() / slice.len() as f64
            })
            .collect()
    }

    /// Exponential Moving Average (EMA) with smoothing factor `alpha` ∈ (0, 1].
    ///
    /// S[0] = x[0]; S[i] = alpha * x[i] + (1 − alpha) * S[i−1]
    pub fn exponential_smoothing(signal: &[f64], alpha: f64) -> Vec<f64> {
        if signal.is_empty() {
            return Vec::new();
        }
        let alpha = alpha.clamp(0.0, 1.0);
        let mut out = Vec::with_capacity(signal.len());
        let mut prev = signal[0];
        out.push(prev);
        for &x in signal.iter().skip(1) {
            prev = alpha * x + (1.0 - alpha) * prev;
            out.push(prev);
        }
        out
    }

    /// Simplified Savitzky–Golay smoothing for `poly_order` ≤ 2.
    ///
    /// Uses a symmetric least-squares polynomial fit of degree ≤ 2 over a
    /// sliding `window` (must be odd ≥ 3).  Falls back to moving average for
    /// `poly_order` == 1 or if `window` < 3.
    pub fn savitzky_golay(signal: &[f64], window: usize, poly_order: usize) -> Vec<f64> {
        if signal.is_empty() {
            return Vec::new();
        }
        // Ensure window is odd and at least 3
        let w = if window < 3 { 3 } else if window % 2 == 0 { window + 1 } else { window };
        let half = (w / 2) as isize;
        let n = signal.len();

        if poly_order <= 1 {
            return Self::moving_average(signal, w);
        }

        // poly_order == 2: compute quadratic fit coefficients via normal equations
        // For a symmetric window of size 2m+1, the smoothing weights for a
        // quadratic fit are known analytically.
        // w_i = (3m^2 + 3m - 1 - 5i²) / ((2m-1)(2m+1)(2m+3)/3)
        // where i ∈ {-m, …, m}.
        let m = half;
        let denom = ((2 * m - 1) * (2 * m + 1) * (2 * m + 3)) as f64 / 3.0;

        let weights: Vec<f64> = ((-m)..=m)
            .map(|i| {
                let numerator = (3 * m * m + 3 * m - 1 - 5 * i * i) as f64;
                numerator / denom
            })
            .collect();

        (0..n)
            .map(|idx| {
                let mut acc = 0.0;
                let mut wsum = 0.0;
                for (wi, offset) in (-m..=m).enumerate() {
                    let j = idx as isize + offset;
                    if j >= 0 && j < n as isize {
                        acc += weights[wi] * signal[j as usize];
                        wsum += weights[wi];
                    }
                }
                if wsum.abs() < f64::EPSILON { signal[idx] } else { acc / wsum }
            })
            .collect()
    }

    // -----------------------------------------------------------------
    // Correlation / convolution
    // -----------------------------------------------------------------

    /// Biased autocorrelation at lags 0 … `max_lag`.
    ///
    /// R[k] = (1/N) Σ_{n=0}^{N-k-1} x[n] · x[n+k]
    pub fn autocorrelation(signal: &[f64], max_lag: usize) -> Vec<f64> {
        let n = signal.len();
        if n == 0 {
            return Vec::new();
        }
        let lags = max_lag.min(n - 1) + 1;
        let mut result = Vec::with_capacity(lags);
        for k in 0..lags {
            let sum: f64 = (0..(n - k)).map(|i| signal[i] * signal[i + k]).sum();
            result.push(sum / n as f64);
        }
        result
    }

    /// Linear convolution of `signal` with `kernel`.
    ///
    /// Output length = `signal.len() + kernel.len() − 1`.
    pub fn convolve(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
        if signal.is_empty() || kernel.is_empty() {
            return Vec::new();
        }
        let out_len = signal.len() + kernel.len() - 1;
        let mut out = vec![0.0_f64; out_len];
        for (i, &s) in signal.iter().enumerate() {
            for (j, &k) in kernel.iter().enumerate() {
                out[i + j] += s * k;
            }
        }
        out
    }

    // -----------------------------------------------------------------
    // Normalisation & detection
    // -----------------------------------------------------------------

    /// Normalise signal to the range [−1, 1] using peak absolute value.
    ///
    /// Returns zeros for a constant-zero signal.
    pub fn normalize(signal: &[f64]) -> Vec<f64> {
        if signal.is_empty() {
            return Vec::new();
        }
        let peak = signal
            .iter()
            .cloned()
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        if peak < f64::EPSILON {
            return vec![0.0; signal.len()];
        }
        signal.iter().map(|&x| x / peak).collect()
    }

    /// Return indices where `|x[i]| > threshold`.
    pub fn threshold_detect(signal: &[f64], threshold: f64) -> Vec<usize> {
        signal
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| if x.abs() > threshold { Some(i) } else { None })
            .collect()
    }

    // -----------------------------------------------------------------
    // Resampling
    // -----------------------------------------------------------------

    /// Downsample by keeping every `factor`-th sample.
    ///
    /// `factor` must be ≥ 1; if 1 the input is returned unchanged.
    pub fn downsample(signal: &[f64], factor: usize) -> Vec<f64> {
        let factor = factor.max(1);
        signal
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| if i % factor == 0 { Some(x) } else { None })
            .collect()
    }

    /// Upsample by inserting `factor − 1` zeros between each sample.
    ///
    /// `factor` must be ≥ 1; if 1 the input is returned unchanged.
    pub fn upsample(signal: &[f64], factor: usize) -> Vec<f64> {
        let factor = factor.max(1);
        if factor == 1 {
            return signal.to_vec();
        }
        let mut out = Vec::with_capacity(signal.len() * factor);
        for &x in signal {
            out.push(x);
            for _ in 1..factor {
                out.push(0.0);
            }
        }
        out
    }

    // -----------------------------------------------------------------
    // Signal-to-Noise Ratio
    // -----------------------------------------------------------------

    /// Signal-to-Noise Ratio in dB: 10 · log₁₀(P_signal / P_noise).
    ///
    /// Returns `f64::INFINITY` when noise power is zero, and
    /// `f64::NEG_INFINITY` when signal power is zero.
    pub fn snr_db(signal: &[f64], noise: &[f64]) -> f64 {
        if signal.is_empty() || noise.is_empty() {
            return f64::NEG_INFINITY;
        }
        let p_signal: f64 = signal.iter().map(|&x| x * x).sum::<f64>() / signal.len() as f64;
        let p_noise: f64 = noise.iter().map(|&x| x * x).sum::<f64>() / noise.len() as f64;
        if p_noise < f64::EPSILON {
            return f64::INFINITY;
        }
        if p_signal < f64::EPSILON {
            return f64::NEG_INFINITY;
        }
        10.0 * (p_signal / p_noise).log10()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Signal struct
    // -------------------------------------------------------------------------
    #[test]
    fn test_signal_new() {
        let s = Signal::new(vec![1.0, 2.0, 3.0], 44100.0);
        assert_eq!(s.samples.len(), 3);
        assert_eq!(s.sample_rate_hz, 44100.0);
    }

    #[test]
    fn test_signal_empty() {
        let s = Signal::new(vec![], 8000.0);
        assert!(s.samples.is_empty());
    }

    // -------------------------------------------------------------------------
    // FilterCoeffs
    // -------------------------------------------------------------------------
    #[test]
    fn test_filter_coeffs_fir() {
        let f = FilterCoeffs::fir(vec![0.25, 0.5, 0.25]);
        assert_eq!(f.b.len(), 3);
        assert_eq!(f.a, vec![1.0]);
    }

    #[test]
    fn test_filter_coeffs_iir() {
        let f = FilterCoeffs::iir(vec![1.0, -1.0], vec![1.0, -0.9]);
        assert_eq!(f.b.len(), 2);
        assert_eq!(f.a.len(), 2);
    }

    // -------------------------------------------------------------------------
    // RMS
    // -------------------------------------------------------------------------
    #[test]
    fn test_rms_constant() {
        let sig = vec![3.0; 4];
        assert!((SignalProcessing::rms(&sig) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_rms_sinusoid() {
        // For a pure sine of amplitude A, RMS = A/√2 in the limit.
        let n = 1000;
        let sig: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin()).collect();
        let rms = SignalProcessing::rms(&sig);
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((rms - expected).abs() < 0.01, "rms={rms}, expected≈{expected}");
    }

    #[test]
    fn test_rms_empty() {
        assert_eq!(SignalProcessing::rms(&[]), 0.0);
    }

    #[test]
    fn test_rms_mixed() {
        // [-1, 1]: rms = 1
        let sig = vec![-1.0, 1.0];
        assert!((SignalProcessing::rms(&sig) - 1.0).abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // Peak-to-peak
    // -------------------------------------------------------------------------
    #[test]
    fn test_peak_to_peak_basic() {
        let sig = vec![1.0, -3.0, 2.0, 0.5];
        assert!((SignalProcessing::peak_to_peak(&sig) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_peak_to_peak_constant() {
        let sig = vec![4.0; 5];
        assert!((SignalProcessing::peak_to_peak(&sig)).abs() < 1e-10);
    }

    #[test]
    fn test_peak_to_peak_empty() {
        assert_eq!(SignalProcessing::peak_to_peak(&[]), 0.0);
    }

    #[test]
    fn test_peak_to_peak_negative() {
        let sig = vec![-5.0, -1.0];
        assert!((SignalProcessing::peak_to_peak(&sig) - 4.0).abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // Zero crossings
    // -------------------------------------------------------------------------
    #[test]
    fn test_zero_crossings_square_wave() {
        let sig = vec![1.0, -1.0, 1.0, -1.0];
        assert_eq!(SignalProcessing::zero_crossings(&sig), 3);
    }

    #[test]
    fn test_zero_crossings_none() {
        let sig = vec![1.0, 2.0, 3.0];
        assert_eq!(SignalProcessing::zero_crossings(&sig), 0);
    }

    #[test]
    fn test_zero_crossings_empty() {
        assert_eq!(SignalProcessing::zero_crossings(&[]), 0);
    }

    #[test]
    fn test_zero_crossings_single() {
        assert_eq!(SignalProcessing::zero_crossings(&[1.0]), 0);
    }

    // -------------------------------------------------------------------------
    // Moving average
    // -------------------------------------------------------------------------
    #[test]
    fn test_moving_average_constant() {
        let sig = vec![5.0; 10];
        let out = SignalProcessing::moving_average(&sig, 3);
        assert_eq!(out.len(), 10);
        for v in &out {
            assert!((v - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_moving_average_impulse() {
        let sig = vec![0.0, 0.0, 6.0, 0.0, 0.0];
        let out = SignalProcessing::moving_average(&sig, 3);
        assert_eq!(out.len(), 5);
        // The middle point should be reduced toward 2.0
        assert!((out[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_moving_average_window_1() {
        let sig = vec![1.0, 2.0, 3.0];
        let out = SignalProcessing::moving_average(&sig, 1);
        for (a, b) in sig.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    // -------------------------------------------------------------------------
    // EMA
    // -------------------------------------------------------------------------
    #[test]
    fn test_exponential_smoothing_alpha_1() {
        // With alpha = 1 the output equals input
        let sig = vec![1.0, 3.0, -1.0, 2.0];
        let out = SignalProcessing::exponential_smoothing(&sig, 1.0);
        for (a, b) in sig.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_exponential_smoothing_convergence() {
        // With alpha close to 0 the output barely changes from the first value
        let sig = vec![0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0];
        let out = SignalProcessing::exponential_smoothing(&sig, 0.01);
        // Last value still close to 0
        assert!(out.last().copied().unwrap_or(0.0) < 10.0);
    }

    #[test]
    fn test_exponential_smoothing_empty() {
        assert!(SignalProcessing::exponential_smoothing(&[], 0.5).is_empty());
    }

    // -------------------------------------------------------------------------
    // Savitzky–Golay
    // -------------------------------------------------------------------------
    #[test]
    fn test_savitzky_golay_constant() {
        let sig = vec![3.0_f64; 15];
        let out = SignalProcessing::savitzky_golay(&sig, 5, 2);
        assert_eq!(out.len(), 15);
        for v in &out {
            assert!((v - 3.0).abs() < 1e-6, "expected 3.0 got {v}");
        }
    }

    #[test]
    fn test_savitzky_golay_poly1_falls_back() {
        let sig = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sg = SignalProcessing::savitzky_golay(&sig, 3, 1);
        let ma = SignalProcessing::moving_average(&sig, 3);
        for (a, b) in sg.iter().zip(ma.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_savitzky_golay_length_preserved() {
        let sig: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let out = SignalProcessing::savitzky_golay(&sig, 7, 2);
        assert_eq!(out.len(), 20);
    }

    // -------------------------------------------------------------------------
    // Autocorrelation
    // -------------------------------------------------------------------------
    #[test]
    fn test_autocorrelation_lag0_equals_variance() {
        let sig = vec![1.0, -1.0, 1.0, -1.0];
        let ac = SignalProcessing::autocorrelation(&sig, 0);
        assert_eq!(ac.len(), 1);
        // R[0] = mean of x[i]^2 = 1.0
        assert!((ac[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_autocorrelation_length() {
        let sig = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ac = SignalProcessing::autocorrelation(&sig, 3);
        assert_eq!(ac.len(), 4); // lags 0..=3
    }

    #[test]
    fn test_autocorrelation_empty() {
        assert!(SignalProcessing::autocorrelation(&[], 5).is_empty());
    }

    #[test]
    fn test_autocorrelation_constant_signal() {
        let sig = vec![2.0; 8];
        let ac = SignalProcessing::autocorrelation(&sig, 4);
        // R[k] = 4 * (8-k)/8 but biased: R[0] = 4.0
        assert!((ac[0] - 4.0).abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // Convolve
    // -------------------------------------------------------------------------
    #[test]
    fn test_convolve_identity_kernel() {
        let sig = vec![1.0, 2.0, 3.0];
        let kernel = vec![1.0];
        let out = SignalProcessing::convolve(&sig, &kernel);
        assert_eq!(out, sig);
    }

    #[test]
    fn test_convolve_length() {
        let sig = vec![1.0; 5];
        let kernel = vec![1.0; 3];
        let out = SignalProcessing::convolve(&sig, &kernel);
        assert_eq!(out.len(), 7); // 5 + 3 - 1
    }

    #[test]
    fn test_convolve_box_filter() {
        // Convolving [1,2,3] with [1,1] yields [1,3,5,3]
        let sig = vec![1.0, 2.0, 3.0];
        let kernel = vec![1.0, 1.0];
        let out = SignalProcessing::convolve(&sig, &kernel);
        let expected = vec![1.0, 3.0, 5.0, 3.0];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_convolve_empty_kernel() {
        let out = SignalProcessing::convolve(&[1.0, 2.0], &[]);
        assert!(out.is_empty());
    }

    // -------------------------------------------------------------------------
    // Normalize
    // -------------------------------------------------------------------------
    #[test]
    fn test_normalize_basic() {
        let sig = vec![2.0, -4.0, 1.0];
        let out = SignalProcessing::normalize(&sig);
        // Peak abs = 4 → normalised peak should be ±1
        let max_abs = out.iter().cloned().map(f64::abs).fold(0.0_f64, f64::max);
        assert!((max_abs - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zero_signal() {
        let sig = vec![0.0; 5];
        let out = SignalProcessing::normalize(&sig);
        for v in out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_normalize_range() {
        let sig: Vec<f64> = (-10..=10).map(|x| x as f64).collect();
        let out = SignalProcessing::normalize(&sig);
        for v in &out {
            assert!(*v >= -1.0 - 1e-10 && *v <= 1.0 + 1e-10);
        }
    }

    // -------------------------------------------------------------------------
    // Threshold detect
    // -------------------------------------------------------------------------
    #[test]
    fn test_threshold_detect_basic() {
        let sig = vec![0.5, 1.5, -0.3, -2.0, 0.1];
        let indices = SignalProcessing::threshold_detect(&sig, 1.0);
        assert_eq!(indices, vec![1, 3]);
    }

    #[test]
    fn test_threshold_detect_all_pass() {
        let sig = vec![5.0; 4];
        let indices = SignalProcessing::threshold_detect(&sig, 1.0);
        assert_eq!(indices.len(), 4);
    }

    #[test]
    fn test_threshold_detect_none_pass() {
        let sig = vec![0.1, 0.2, 0.3];
        let indices = SignalProcessing::threshold_detect(&sig, 1.0);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_threshold_detect_empty() {
        let indices = SignalProcessing::threshold_detect(&[], 0.5);
        assert!(indices.is_empty());
    }

    // -------------------------------------------------------------------------
    // Downsample
    // -------------------------------------------------------------------------
    #[test]
    fn test_downsample_factor_2() {
        let sig = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let out = SignalProcessing::downsample(&sig, 2);
        assert_eq!(out, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_downsample_factor_1() {
        let sig = vec![1.0, 2.0, 3.0];
        let out = SignalProcessing::downsample(&sig, 1);
        assert_eq!(out, sig);
    }

    #[test]
    fn test_downsample_factor_0_treated_as_1() {
        let sig = vec![1.0, 2.0, 3.0];
        let out = SignalProcessing::downsample(&sig, 0);
        assert_eq!(out, sig);
    }

    #[test]
    fn test_downsample_larger_than_len() {
        let sig = vec![7.0, 8.0, 9.0];
        let out = SignalProcessing::downsample(&sig, 10);
        assert_eq!(out, vec![7.0]);
    }

    // -------------------------------------------------------------------------
    // Upsample
    // -------------------------------------------------------------------------
    #[test]
    fn test_upsample_factor_2() {
        let sig = vec![1.0, 2.0, 3.0];
        let out = SignalProcessing::upsample(&sig, 2);
        assert_eq!(out, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_upsample_factor_1() {
        let sig = vec![1.0, 2.0, 3.0];
        let out = SignalProcessing::upsample(&sig, 1);
        assert_eq!(out, sig);
    }

    #[test]
    fn test_upsample_empty() {
        assert!(SignalProcessing::upsample(&[], 3).is_empty());
    }

    #[test]
    fn test_upsample_length() {
        let sig = vec![1.0; 5];
        let out = SignalProcessing::upsample(&sig, 4);
        assert_eq!(out.len(), 20);
    }

    // -------------------------------------------------------------------------
    // SNR
    // -------------------------------------------------------------------------
    #[test]
    fn test_snr_db_perfect_signal() {
        let sig = vec![1.0; 100];
        let noise = vec![0.0; 100];
        assert_eq!(SignalProcessing::snr_db(&sig, &noise), f64::INFINITY);
    }

    #[test]
    fn test_snr_db_zero_signal() {
        let sig = vec![0.0; 10];
        let noise = vec![1.0; 10];
        assert_eq!(SignalProcessing::snr_db(&sig, &noise), f64::NEG_INFINITY);
    }

    #[test]
    fn test_snr_db_equal_power() {
        let sig = vec![1.0; 10];
        let noise = vec![1.0; 10];
        let snr = SignalProcessing::snr_db(&sig, &noise);
        assert!((snr - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_snr_db_10x_signal() {
        // P_signal = 100, P_noise = 1 → SNR = 10 dB (power ratio)
        let sig = vec![10.0; 100];
        let noise = vec![1.0; 100];
        let snr = SignalProcessing::snr_db(&sig, &noise);
        assert!((snr - 20.0).abs() < 1e-6, "snr={snr}"); // 10*log10(100) = 20
    }

    #[test]
    fn test_snr_db_empty_inputs() {
        assert_eq!(
            SignalProcessing::snr_db(&[], &[1.0]),
            f64::NEG_INFINITY
        );
        assert_eq!(
            SignalProcessing::snr_db(&[1.0], &[]),
            f64::NEG_INFINITY
        );
    }

    // -------------------------------------------------------------------------
    // Additional integration tests
    // -------------------------------------------------------------------------
    #[test]
    fn test_downsample_then_upsample_length() {
        let sig: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let down = SignalProcessing::downsample(&sig, 3); // len = 4
        let up = SignalProcessing::upsample(&down, 3);   // len = 12
        assert_eq!(up.len(), 12);
        // Every 3rd sample should be the original downsampled value
        for (i, &d) in down.iter().enumerate() {
            assert!((up[i * 3] - d).abs() < 1e-10);
        }
    }

    #[test]
    fn test_normalize_then_rms() {
        let sig = vec![1.0, 2.0, -3.0, 4.0];
        let norm = SignalProcessing::normalize(&sig);
        // After normalisation peak = 1; RMS should be < 1
        let rms = SignalProcessing::rms(&norm);
        assert!(rms <= 1.0 + 1e-10);
    }

    #[test]
    fn test_moving_average_reduces_noise() {
        // A noisy constant signal should smooth to approximately the constant
        let sig: Vec<f64> = (0..100)
            .map(|i| 5.0 + if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let smoothed = SignalProcessing::moving_average(&sig, 10);
        // Middle values should be close to 5.0
        let mid = &smoothed[20..80];
        let mean = mid.iter().sum::<f64>() / mid.len() as f64;
        assert!((mean - 5.0).abs() < 0.5, "mean={mean}");
    }

    #[test]
    fn test_threshold_detect_symmetric() {
        let sig = vec![-2.0, 0.5, 2.0, -0.5, 3.0];
        let indices = SignalProcessing::threshold_detect(&sig, 1.5);
        assert_eq!(indices, vec![0, 2, 4]);
    }

    #[test]
    fn test_zero_crossings_exact_zero() {
        // Exact zero should not count as a crossing between positive
        let sig = vec![1.0, 0.0, 1.0];
        // 1.0 * 0.0 = 0.0 which is not < 0.0
        assert_eq!(SignalProcessing::zero_crossings(&sig), 0);
    }

    #[test]
    fn test_autocorrelation_max_lag_equals_len_minus_one() {
        let sig = vec![1.0, 2.0, 3.0];
        let ac = SignalProcessing::autocorrelation(&sig, 10); // clipped to len-1 = 2
        assert_eq!(ac.len(), 3); // lags 0, 1, 2
    }

    #[test]
    fn test_convolve_commutative_approx() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.5, -0.5];
        let ab = SignalProcessing::convolve(&a, &b);
        let ba = SignalProcessing::convolve(&b, &a);
        assert_eq!(ab.len(), ba.len());
        for (x, y) in ab.iter().zip(ba.iter()) {
            assert!((x - y).abs() < 1e-10);
        }
    }

    #[test]
    fn test_snr_db_positive() {
        let sig: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 10.0).collect();
        let noise: Vec<f64> = (0..100).map(|i| (i as f64 * 0.3).cos() * 0.1).collect();
        let snr = SignalProcessing::snr_db(&sig, &noise);
        assert!(snr > 0.0, "Expected positive SNR, got {snr}");
    }

    #[test]
    fn test_exponential_smoothing_monotone() {
        // Smoothed version of step function should be monotonically non-decreasing
        let sig: Vec<f64> = (0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }).collect();
        let out = SignalProcessing::exponential_smoothing(&sig, 0.3);
        for w in out.windows(2).skip(10) {
            assert!(w[1] >= w[0] - 1e-10, "non-monotone: {} > {}", w[0], w[1]);
        }
    }
}
