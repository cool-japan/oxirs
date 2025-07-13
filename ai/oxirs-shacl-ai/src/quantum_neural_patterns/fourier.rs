//! Quantum Fourier Transform for pattern analysis

use nalgebra::DMatrix;
use num_complex::Complex64;
use std::f64::consts::PI;

use super::core::QuantumPattern;
use crate::{Result, ShaclAiError};

/// Quantum Fourier Transform for Pattern Analysis
#[derive(Debug, Clone)]
pub struct QuantumFourierTransform {
    /// Number of qubits
    num_qubits: usize,
    /// Transform matrix
    transform_matrix: DMatrix<Complex64>,
}

impl QuantumFourierTransform {
    /// Create a new quantum Fourier transform
    pub fn new(num_qubits: usize) -> Self {
        let n = 2_usize.pow(num_qubits as u32);
        let mut transform_matrix = DMatrix::zeros(n, n);

        // Construct QFT matrix
        let omega = Complex64::new(0.0, 2.0 * PI / n as f64).exp();
        for i in 0..n {
            for j in 0..n {
                transform_matrix[(i, j)] =
                    omega.powf(i as f64 * j as f64) / Complex64::new((n as f64).sqrt(), 0.0);
            }
        }

        Self {
            num_qubits,
            transform_matrix,
        }
    }

    /// Apply quantum Fourier transform to pattern
    pub fn apply_qft(&self, pattern: &mut QuantumPattern) -> Result<()> {
        if pattern.quantum_state.amplitudes.len() != self.transform_matrix.nrows() {
            return Err(ShaclAiError::PatternRecognition(
                "Pattern size doesn't match QFT dimensions".to_string(),
            ));
        }

        // Convert to nalgebra vector
        let input_vector = nalgebra::DVector::from_vec(pattern.quantum_state.amplitudes.clone());

        // Apply QFT transformation
        let output_vector = &self.transform_matrix * input_vector;

        // Update pattern amplitudes
        pattern.quantum_state.amplitudes = output_vector.as_slice().to_vec();

        Ok(())
    }

    /// Apply inverse quantum Fourier transform
    pub fn apply_inverse_qft(&self, pattern: &mut QuantumPattern) -> Result<()> {
        // Inverse QFT is complex conjugate transpose
        let inverse_matrix = self.transform_matrix.adjoint();

        let input_vector = nalgebra::DVector::from_vec(pattern.quantum_state.amplitudes.clone());
        let output_vector = inverse_matrix * input_vector;

        pattern.quantum_state.amplitudes = output_vector.as_slice().to_vec();

        Ok(())
    }

    /// Extract frequency components from pattern
    pub fn analyze_frequency_spectrum(&self, pattern: &QuantumPattern) -> Result<Vec<f64>> {
        let spectrum: Vec<f64> = pattern
            .quantum_state
            .amplitudes
            .iter()
            .map(|amp| amp.norm_sqr())
            .collect();

        Ok(spectrum)
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get transform matrix dimensions
    pub fn matrix_size(&self) -> usize {
        self.transform_matrix.nrows()
    }
}
