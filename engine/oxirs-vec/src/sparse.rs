use crate::{Vector, VectorData, VectorError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sparse vector representation using a hash map for efficient storage
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseVector {
    /// Non-zero values indexed by their position
    pub values: HashMap<usize, f32>,
    /// Total dimensions of the vector
    pub dimensions: usize,
    /// Optional metadata
    pub metadata: Option<HashMap<String, String>>,
}

impl SparseVector {
    /// Create a new sparse vector from indices and values
    pub fn new(
        indices: Vec<usize>,
        values: Vec<f32>,
        dimensions: usize,
    ) -> Result<Self, VectorError> {
        if indices.len() != values.len() {
            return Err(VectorError::InvalidDimensions(
                "Indices and values must have same length".to_string(),
            ));
        }

        if let Some(&max_idx) = indices.iter().max() {
            if max_idx >= dimensions {
                return Err(VectorError::InvalidDimensions(format!(
                    "Index {} exceeds dimensions {}",
                    max_idx, dimensions
                )));
            }
        }

        let mut sparse_values = HashMap::new();
        for (idx, val) in indices.into_iter().zip(values.into_iter()) {
            if val != 0.0 {
                // Only store non-zero values
                sparse_values.insert(idx, val);
            }
        }

        Ok(Self {
            values: sparse_values,
            dimensions,
            metadata: None,
        })
    }

    /// Create sparse vector from dense vector
    pub fn from_dense(dense: &Vector) -> Self {
        let values = dense.as_f32();
        let mut sparse_values = HashMap::new();

        for (idx, &val) in values.iter().enumerate() {
            if val.abs() > f32::EPSILON {
                // Only store non-zero values
                sparse_values.insert(idx, val);
            }
        }

        Self {
            values: sparse_values,
            dimensions: dense.dimensions,
            metadata: dense.metadata.clone(),
        }
    }

    /// Convert to dense vector
    pub fn to_dense(&self) -> Vector {
        let mut values = vec![0.0; self.dimensions];

        for (&idx, &val) in &self.values {
            if idx < self.dimensions {
                values[idx] = val;
            }
        }

        let mut vec = Vector::new(values);
        vec.metadata = self.metadata.clone();
        vec
    }

    /// Get value at index
    pub fn get(&self, index: usize) -> f32 {
        self.values.get(&index).copied().unwrap_or(0.0)
    }

    /// Set value at index
    pub fn set(&mut self, index: usize, value: f32) -> Result<(), VectorError> {
        if index >= self.dimensions {
            return Err(VectorError::InvalidDimensions(format!(
                "Index {} exceeds dimensions {}",
                index, self.dimensions
            )));
        }

        if value.abs() > f32::EPSILON {
            self.values.insert(index, value);
        } else {
            self.values.remove(&index);
        }

        Ok(())
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparsity ratio (percentage of zero elements)
    pub fn sparsity(&self) -> f32 {
        let non_zero = self.nnz() as f32;
        let total = self.dimensions as f32;
        (total - non_zero) / total
    }

    /// Dot product with another sparse vector
    pub fn dot(&self, other: &SparseVector) -> Result<f32, VectorError> {
        if self.dimensions != other.dimensions {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: other.dimensions,
            });
        }

        let mut sum = 0.0;

        // Only iterate over the smaller set of indices
        if self.values.len() <= other.values.len() {
            for (&idx, &val) in &self.values {
                if let Some(&other_val) = other.values.get(&idx) {
                    sum += val * other_val;
                }
            }
        } else {
            for (&idx, &val) in &other.values {
                if let Some(&self_val) = self.values.get(&idx) {
                    sum += val * self_val;
                }
            }
        }

        Ok(sum)
    }

    /// Compute cosine similarity with another sparse vector
    pub fn cosine_similarity(&self, other: &SparseVector) -> Result<f32, VectorError> {
        let dot = self.dot(other)?;
        let self_norm = self.l2_norm();
        let other_norm = other.l2_norm();

        if self_norm == 0.0 || other_norm == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot / (self_norm * other_norm))
        }
    }

    /// Compute L2 norm
    pub fn l2_norm(&self) -> f32 {
        self.values.values().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Compute L1 norm
    pub fn l1_norm(&self) -> f32 {
        self.values.values().map(|v| v.abs()).sum()
    }

    /// Add another sparse vector
    pub fn add(&self, other: &SparseVector) -> Result<SparseVector, VectorError> {
        if self.dimensions != other.dimensions {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: other.dimensions,
            });
        }

        let mut result = self.clone();

        for (&idx, &val) in &other.values {
            let new_val = result.get(idx) + val;
            result.set(idx, new_val)?;
        }

        Ok(result)
    }

    /// Subtract another sparse vector
    pub fn subtract(&self, other: &SparseVector) -> Result<SparseVector, VectorError> {
        if self.dimensions != other.dimensions {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: other.dimensions,
            });
        }

        let mut result = self.clone();

        for (&idx, &val) in &other.values {
            let new_val = result.get(idx) - val;
            result.set(idx, new_val)?;
        }

        Ok(result)
    }

    /// Scale by scalar
    pub fn scale(&self, scalar: f32) -> SparseVector {
        let mut result = self.clone();

        for val in result.values.values_mut() {
            *val *= scalar;
        }

        result
    }

    /// Normalize to unit length
    pub fn normalize(&self) -> SparseVector {
        let norm = self.l2_norm();
        if norm > 0.0 {
            self.scale(1.0 / norm)
        } else {
            self.clone()
        }
    }
}

/// Compressed Sparse Row (CSR) format for efficient batch operations
#[derive(Debug, Clone, PartialEq)]
pub struct CSRMatrix {
    /// Non-zero values
    pub values: Vec<f32>,
    /// Column indices for each value
    pub col_indices: Vec<usize>,
    /// Row pointers (start index of each row in values/col_indices)
    pub row_ptrs: Vec<usize>,
    /// Shape of the matrix (rows, cols)
    pub shape: (usize, usize),
}

impl CSRMatrix {
    /// Create CSR matrix from sparse vectors
    pub fn from_sparse_vectors(vectors: &[SparseVector]) -> Result<Self, VectorError> {
        if vectors.is_empty() {
            return Ok(Self {
                values: Vec::new(),
                col_indices: Vec::new(),
                row_ptrs: vec![0],
                shape: (0, 0),
            });
        }

        let num_rows = vectors.len();
        let num_cols = vectors[0].dimensions;

        // Verify all vectors have same dimensions
        for (i, vec) in vectors.iter().enumerate() {
            if vec.dimensions != num_cols {
                return Err(VectorError::InvalidDimensions(format!(
                    "Vector {} has {} dimensions, expected {}",
                    i, vec.dimensions, num_cols
                )));
            }
        }

        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];

        for vec in vectors {
            // Sort by column index for CSR format
            let mut sorted_entries: Vec<_> = vec.values.iter().collect();
            sorted_entries.sort_by_key(|(&idx, _)| idx);

            for (&idx, &val) in sorted_entries {
                values.push(val);
                col_indices.push(idx);
            }

            row_ptrs.push(values.len());
        }

        Ok(Self {
            values,
            col_indices,
            row_ptrs,
            shape: (num_rows, num_cols),
        })
    }

    /// Get a specific row as sparse vector
    pub fn get_row(&self, row: usize) -> Option<SparseVector> {
        if row >= self.shape.0 {
            return None;
        }

        let start = self.row_ptrs[row];
        let end = self.row_ptrs[row + 1];

        let mut values = HashMap::new();
        for i in start..end {
            values.insert(self.col_indices[i], self.values[i]);
        }

        Some(SparseVector {
            values,
            dimensions: self.shape.1,
            metadata: None,
        })
    }

    /// Matrix-vector multiplication
    pub fn multiply_vector(&self, vector: &SparseVector) -> Result<Vec<f32>, VectorError> {
        if self.shape.1 != vector.dimensions {
            return Err(VectorError::DimensionMismatch {
                expected: self.shape.1,
                actual: vector.dimensions,
            });
        }

        let mut result = vec![0.0; self.shape.0];

        for row in 0..self.shape.0 {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];

            let mut sum = 0.0;
            for i in start..end {
                let col = self.col_indices[i];
                if let Some(&vec_val) = vector.values.get(&col) {
                    sum += self.values[i] * vec_val;
                }
            }
            result[row] = sum;
        }

        Ok(result)
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.values.len() * std::mem::size_of::<f32>()
            + self.col_indices.len() * std::mem::size_of::<usize>()
            + self.row_ptrs.len() * std::mem::size_of::<usize>()
    }

    /// Get sparsity of the matrix
    pub fn sparsity(&self) -> f32 {
        let total_elements = self.shape.0 * self.shape.1;
        let non_zero = self.values.len();
        (total_elements - non_zero) as f32 / total_elements as f32
    }
}

/// Coordinate (COO) format for easy construction
#[derive(Debug, Clone, PartialEq)]
pub struct COOMatrix {
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f32>,
    pub shape: (usize, usize),
}

impl COOMatrix {
    /// Create empty COO matrix
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            shape: (rows, cols),
        }
    }

    /// Add a value to the matrix
    pub fn add_value(&mut self, row: usize, col: usize, value: f32) -> Result<(), VectorError> {
        if row >= self.shape.0 || col >= self.shape.1 {
            return Err(VectorError::InvalidDimensions(format!(
                "Index ({}, {}) out of bounds for shape {:?}",
                row, col, self.shape
            )));
        }

        if value.abs() > f32::EPSILON {
            self.row_indices.push(row);
            self.col_indices.push(col);
            self.values.push(value);
        }

        Ok(())
    }

    /// Convert to CSR format
    pub fn to_csr(&self) -> CSRMatrix {
        // Sort by row, then column
        let mut entries: Vec<_> = (0..self.values.len())
            .map(|i| (self.row_indices[i], self.col_indices[i], self.values[i]))
            .collect();
        entries.sort_by_key(|&(r, c, _)| (r, c));

        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0];

        let mut current_row = 0;
        for (row, col, val) in entries {
            while current_row < row {
                row_ptrs.push(values.len());
                current_row += 1;
            }
            values.push(val);
            col_indices.push(col);
        }

        while current_row < self.shape.0 {
            row_ptrs.push(values.len());
            current_row += 1;
        }

        CSRMatrix {
            values,
            col_indices,
            row_ptrs,
            shape: self.shape,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_vector_creation() {
        let indices = vec![0, 3, 7];
        let values = vec![1.0, 2.0, 3.0];
        let sparse = SparseVector::new(indices, values, 10).unwrap();

        assert_eq!(sparse.get(0), 1.0);
        assert_eq!(sparse.get(3), 2.0);
        assert_eq!(sparse.get(7), 3.0);
        assert_eq!(sparse.get(5), 0.0);
        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.dimensions, 10);
    }

    #[test]
    fn test_sparse_dense_conversion() {
        let dense = Vector::new(vec![0.0, 1.0, 0.0, 2.0, 0.0]);
        let sparse = SparseVector::from_dense(&dense);

        assert_eq!(sparse.nnz(), 2);
        assert_eq!(sparse.get(1), 1.0);
        assert_eq!(sparse.get(3), 2.0);

        let dense_back = sparse.to_dense();
        assert_eq!(dense_back.as_f32(), vec![0.0, 1.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn test_sparse_operations() {
        let sparse1 = SparseVector::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0], 5).unwrap();
        let sparse2 = SparseVector::new(vec![1, 2, 3], vec![4.0, 5.0, 6.0], 5).unwrap();

        // Dot product
        let dot = sparse1.dot(&sparse2).unwrap();
        assert_eq!(dot, 10.0); // Only index 2 overlaps: 2.0 * 5.0 = 10.0

        // Addition
        let sum = sparse1.add(&sparse2).unwrap();
        assert_eq!(sum.get(0), 1.0);
        assert_eq!(sum.get(1), 4.0);
        assert_eq!(sum.get(2), 7.0);
        assert_eq!(sum.get(3), 6.0);
        assert_eq!(sum.get(4), 3.0);

        // Scaling
        let scaled = sparse1.scale(2.0);
        assert_eq!(scaled.get(0), 2.0);
        assert_eq!(scaled.get(2), 4.0);
        assert_eq!(scaled.get(4), 6.0);
    }

    #[test]
    fn test_csr_matrix() {
        let vectors = vec![
            SparseVector::new(vec![0, 2], vec![1.0, 2.0], 4).unwrap(),
            SparseVector::new(vec![1, 3], vec![3.0, 4.0], 4).unwrap(),
            SparseVector::new(vec![0, 1, 2], vec![5.0, 6.0, 7.0], 4).unwrap(),
        ];

        let csr = CSRMatrix::from_sparse_vectors(&vectors).unwrap();

        assert_eq!(csr.shape, (3, 4));
        assert_eq!(csr.values.len(), 7);
        assert_eq!(csr.row_ptrs, vec![0, 2, 4, 7]);

        // Test row extraction
        let row1 = csr.get_row(1).unwrap();
        assert_eq!(row1.get(1), 3.0);
        assert_eq!(row1.get(3), 4.0);
    }

    #[test]
    fn test_coo_to_csr() {
        let mut coo = COOMatrix::new(3, 3);
        coo.add_value(0, 0, 1.0).unwrap();
        coo.add_value(0, 2, 2.0).unwrap();
        coo.add_value(1, 1, 3.0).unwrap();
        coo.add_value(2, 0, 4.0).unwrap();
        coo.add_value(2, 2, 5.0).unwrap();

        let csr = coo.to_csr();
        assert_eq!(csr.values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(csr.col_indices, vec![0, 2, 1, 0, 2]);
        assert_eq!(csr.row_ptrs, vec![0, 2, 3, 5]);
    }
}
