//! # TDB Assembler
//!
//! Assembly language and grammar for TDB operations with TDB2 parity.
//! Provides low-level operation assembly and disassembly for efficient storage operations.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::triple_store::TripleKey;
use crate::nodes::NodeId;

/// TDB assembler for low-level storage operations
#[derive(Debug)]
pub struct Assembler {
    /// Operation code mapping
    opcodes: HashMap<OperationType, u8>,
    /// Reverse opcode mapping
    reverse_opcodes: HashMap<u8, OperationType>,
    /// Assembly statistics
    stats: AssemblerStats,
}

impl Assembler {
    /// Create a new assembler instance
    pub fn new() -> Self {
        let mut opcodes = HashMap::new();
        opcodes.insert(OperationType::Load, 0x01);
        opcodes.insert(OperationType::Store, 0x02);
        opcodes.insert(OperationType::Index, 0x03);
        opcodes.insert(OperationType::Query, 0x04);
        opcodes.insert(OperationType::TripleInsert, 0x10);
        opcodes.insert(OperationType::TripleDelete, 0x11);
        opcodes.insert(OperationType::TripleQuery, 0x12);
        opcodes.insert(OperationType::IndexScan, 0x20);
        opcodes.insert(OperationType::PageRead, 0x30);
        opcodes.insert(OperationType::PageWrite, 0x31);
        opcodes.insert(OperationType::Checkpoint, 0x40);
        opcodes.insert(OperationType::Transaction, 0x41);

        let reverse_opcodes = opcodes.iter().map(|(&op, &code)| (code, op)).collect();

        Self {
            opcodes,
            reverse_opcodes,
            stats: AssemblerStats::default(),
        }
    }
    
    /// Assemble TDB operations into bytecode
    pub fn assemble(&mut self, operations: &[Operation]) -> Result<AssemblyResult> {
        let mut bytecode = Vec::new();
        let mut operation_count = 0;

        // Write header
        bytecode.extend_from_slice(b"TDB2"); // Magic number
        bytecode.extend_from_slice(&(operations.len() as u32).to_le_bytes());

        for operation in operations {
            let op_bytecode = self.assemble_operation(operation)?;
            bytecode.extend_from_slice(&op_bytecode);
            operation_count += 1;
        }

        // Calculate checksum
        let checksum = self.calculate_checksum(&bytecode);
        bytecode.extend_from_slice(&checksum.to_le_bytes());

        // Update stats
        self.stats.operations_assembled += operation_count;
        self.stats.bytes_assembled += bytecode.len() as u64;

        let metadata = AssemblyMetadata {
            operations_count: operation_count,
            size: bytecode.len(),
            checksum,
            compression_ratio: 1.0, // No compression for now
        };

        Ok(AssemblyResult { bytecode, metadata })
    }
    
    /// Disassemble bytecode to operations
    pub fn disassemble(&mut self, bytecode: &[u8]) -> Result<Vec<Operation>> {
        if bytecode.len() < 16 {
            return Err(anyhow!("Bytecode too short"));
        }

        // Check magic number
        if &bytecode[0..4] != b"TDB2" {
            return Err(anyhow!("Invalid magic number"));
        }

        let operation_count = u32::from_le_bytes([
            bytecode[4], bytecode[5], bytecode[6], bytecode[7]
        ]) as usize;

        let mut operations = Vec::with_capacity(operation_count);
        let mut offset = 8;

        for _ in 0..operation_count {
            let (operation, consumed) = self.disassemble_operation(&bytecode[offset..])?;
            operations.push(operation);
            offset += consumed;
        }

        // Verify checksum
        let expected_checksum = u64::from_le_bytes([
            bytecode[offset], bytecode[offset + 1], bytecode[offset + 2], bytecode[offset + 3],
            bytecode[offset + 4], bytecode[offset + 5], bytecode[offset + 6], bytecode[offset + 7],
        ]);

        let actual_checksum = self.calculate_checksum(&bytecode[0..offset]);
        if actual_checksum != expected_checksum {
            return Err(anyhow!("Checksum mismatch"));
        }

        // Update stats
        self.stats.operations_disassembled += operations.len() as u64;

        Ok(operations)
    }

    /// Get assembler statistics
    pub fn get_stats(&self) -> &AssemblerStats {
        &self.stats
    }

    /// Reset assembler statistics
    pub fn reset_stats(&mut self) {
        self.stats = AssemblerStats::default();
    }

    // Private helper methods

    fn assemble_operation(&self, operation: &Operation) -> Result<Vec<u8>> {
        let mut bytecode = Vec::new();

        match operation {
            Operation::Load { address, size } => {
                bytecode.push(*self.opcodes.get(&OperationType::Load).unwrap());
                bytecode.extend_from_slice(&address.to_le_bytes());
                bytecode.extend_from_slice(&size.to_le_bytes());
            }

            Operation::Store { address, data } => {
                bytecode.push(*self.opcodes.get(&OperationType::Store).unwrap());
                bytecode.extend_from_slice(&address.to_le_bytes());
                bytecode.extend_from_slice(&(data.len() as u32).to_le_bytes());
                bytecode.extend_from_slice(data);
            }

            Operation::Index { key, value } => {
                bytecode.push(*self.opcodes.get(&OperationType::Index).unwrap());
                let key_bytes = key.as_bytes();
                let value_bytes = value.as_bytes();
                bytecode.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
                bytecode.extend_from_slice(key_bytes);
                bytecode.extend_from_slice(&(value_bytes.len() as u32).to_le_bytes());
                bytecode.extend_from_slice(value_bytes);
            }

            Operation::Query { pattern } => {
                bytecode.push(*self.opcodes.get(&OperationType::Query).unwrap());
                let pattern_bytes = pattern.as_bytes();
                bytecode.extend_from_slice(&(pattern_bytes.len() as u32).to_le_bytes());
                bytecode.extend_from_slice(pattern_bytes);
            }

            Operation::TripleInsert { subject, predicate, object } => {
                bytecode.push(*self.opcodes.get(&OperationType::TripleInsert).unwrap());
                bytecode.extend_from_slice(&subject.to_le_bytes());
                bytecode.extend_from_slice(&predicate.to_le_bytes());
                bytecode.extend_from_slice(&object.to_le_bytes());
            }

            Operation::TripleDelete { subject, predicate, object } => {
                bytecode.push(*self.opcodes.get(&OperationType::TripleDelete).unwrap());
                bytecode.extend_from_slice(&subject.to_le_bytes());
                bytecode.extend_from_slice(&predicate.to_le_bytes());
                bytecode.extend_from_slice(&object.to_le_bytes());
            }

            Operation::TripleQuery { subject, predicate, object } => {
                bytecode.push(*self.opcodes.get(&OperationType::TripleQuery).unwrap());
                bytecode.extend_from_slice(&subject.unwrap_or(0).to_le_bytes());
                bytecode.extend_from_slice(&predicate.unwrap_or(0).to_le_bytes());
                bytecode.extend_from_slice(&object.unwrap_or(0).to_le_bytes());
            }

            Operation::IndexScan { index_type, start_key, end_key } => {
                bytecode.push(*self.opcodes.get(&OperationType::IndexScan).unwrap());
                bytecode.push(*index_type as u8);
                bytecode.extend_from_slice(&start_key.to_bytes());
                bytecode.extend_from_slice(&end_key.to_bytes());
            }

            Operation::PageRead { page_id } => {
                bytecode.push(*self.opcodes.get(&OperationType::PageRead).unwrap());
                bytecode.extend_from_slice(&page_id.to_le_bytes());
            }

            Operation::PageWrite { page_id, data } => {
                bytecode.push(*self.opcodes.get(&OperationType::PageWrite).unwrap());
                bytecode.extend_from_slice(&page_id.to_le_bytes());
                bytecode.extend_from_slice(&(data.len() as u32).to_le_bytes());
                bytecode.extend_from_slice(data);
            }

            Operation::Checkpoint => {
                bytecode.push(*self.opcodes.get(&OperationType::Checkpoint).unwrap());
            }

            Operation::Transaction { transaction_id, operation_type } => {
                bytecode.push(*self.opcodes.get(&OperationType::Transaction).unwrap());
                bytecode.extend_from_slice(&transaction_id.to_le_bytes());
                bytecode.push(*operation_type as u8);
            }
        }

        Ok(bytecode)
    }

    fn disassemble_operation(&self, bytecode: &[u8]) -> Result<(Operation, usize)> {
        if bytecode.is_empty() {
            return Err(anyhow!("Empty bytecode"));
        }

        let opcode = bytecode[0];
        let op_type = self.reverse_opcodes.get(&opcode)
            .ok_or_else(|| anyhow!("Unknown opcode: {}", opcode))?;

        let mut offset = 1;

        let operation = match op_type {
            OperationType::Load => {
                let address = u64::from_le_bytes([
                    bytecode[offset], bytecode[offset + 1], bytecode[offset + 2], bytecode[offset + 3],
                    bytecode[offset + 4], bytecode[offset + 5], bytecode[offset + 6], bytecode[offset + 7],
                ]);
                offset += 8;
                let size = u32::from_le_bytes([
                    bytecode[offset], bytecode[offset + 1], bytecode[offset + 2], bytecode[offset + 3],
                ]);
                offset += 4;
                Operation::Load { address, size }
            }

            OperationType::Store => {
                let address = u64::from_le_bytes([
                    bytecode[offset], bytecode[offset + 1], bytecode[offset + 2], bytecode[offset + 3],
                    bytecode[offset + 4], bytecode[offset + 5], bytecode[offset + 6], bytecode[offset + 7],
                ]);
                offset += 8;
                let data_len = u32::from_le_bytes([
                    bytecode[offset], bytecode[offset + 1], bytecode[offset + 2], bytecode[offset + 3],
                ]) as usize;
                offset += 4;
                let data = bytecode[offset..offset + data_len].to_vec();
                offset += data_len;
                Operation::Store { address, data }
            }

            OperationType::TripleInsert => {
                let subject = u64::from_le_bytes([
                    bytecode[offset], bytecode[offset + 1], bytecode[offset + 2], bytecode[offset + 3],
                    bytecode[offset + 4], bytecode[offset + 5], bytecode[offset + 6], bytecode[offset + 7],
                ]);
                offset += 8;
                let predicate = u64::from_le_bytes([
                    bytecode[offset], bytecode[offset + 1], bytecode[offset + 2], bytecode[offset + 3],
                    bytecode[offset + 4], bytecode[offset + 5], bytecode[offset + 6], bytecode[offset + 7],
                ]);
                offset += 8;
                let object = u64::from_le_bytes([
                    bytecode[offset], bytecode[offset + 1], bytecode[offset + 2], bytecode[offset + 3],
                    bytecode[offset + 4], bytecode[offset + 5], bytecode[offset + 6], bytecode[offset + 7],
                ]);
                offset += 8;
                Operation::TripleInsert { subject, predicate, object }
            }

            _ => {
                return Err(anyhow!("Disassembly not implemented for opcode: {}", opcode));
            }
        };

        Ok((operation, offset))
    }

    fn calculate_checksum(&self, data: &[u8]) -> u64 {
        let mut checksum = 0u64;
        for byte in data {
            checksum = checksum.wrapping_add(*byte as u64);
            checksum = checksum.wrapping_mul(1103515245).wrapping_add(12345);
        }
        checksum
    }
}

impl Default for Assembler {
    fn default() -> Self {
        Self::new()
    }
}

/// TDB operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationType {
    Load,
    Store,
    Index,
    Query,
    TripleInsert,
    TripleDelete,
    TripleQuery,
    IndexScan,
    PageRead,
    PageWrite,
    Checkpoint,
    Transaction,
}

/// TDB operations for assembly
#[derive(Debug, Clone)]
pub enum Operation {
    /// Load operation
    Load { address: u64, size: u32 },
    /// Store operation
    Store { address: u64, data: Vec<u8> },
    /// Index operation
    Index { key: String, value: String },
    /// Query operation
    Query { pattern: String },
    /// Triple insert operation
    TripleInsert { subject: NodeId, predicate: NodeId, object: NodeId },
    /// Triple delete operation
    TripleDelete { subject: NodeId, predicate: NodeId, object: NodeId },
    /// Triple query operation
    TripleQuery { subject: Option<NodeId>, predicate: Option<NodeId>, object: Option<NodeId> },
    /// Index scan operation
    IndexScan { index_type: IndexScanType, start_key: TripleKey, end_key: TripleKey },
    /// Page read operation
    PageRead { page_id: u64 },
    /// Page write operation
    PageWrite { page_id: u64, data: Vec<u8> },
    /// Checkpoint operation
    Checkpoint,
    /// Transaction operation
    Transaction { transaction_id: u64, operation_type: TransactionOperationType },
}

/// Index scan types
#[derive(Debug, Clone, Copy)]
pub enum IndexScanType {
    SPO = 0,
    POS = 1,
    OSP = 2,
    SOP = 3,
    PSO = 4,
    OPS = 5,
}

/// Transaction operation types
#[derive(Debug, Clone, Copy)]
pub enum TransactionOperationType {
    Begin = 0,
    Commit = 1,
    Abort = 2,
}

/// Assembly result
#[derive(Debug)]
pub struct AssemblyResult {
    pub bytecode: Vec<u8>,
    pub metadata: AssemblyMetadata,
}

/// Assembly metadata
#[derive(Debug)]
pub struct AssemblyMetadata {
    pub operations_count: usize,
    pub size: usize,
    pub checksum: u64,
    pub compression_ratio: f32,
}

/// Assembler statistics
#[derive(Debug, Default, Serialize)]
pub struct AssemblerStats {
    pub operations_assembled: u64,
    pub operations_disassembled: u64,
    pub bytes_assembled: u64,
    pub assembly_errors: u64,
    pub disassembly_errors: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assembler_basic_operations() {
        let mut assembler = Assembler::new();
        
        let operations = vec![
            Operation::Load { address: 0x1000, size: 4096 },
            Operation::TripleInsert { subject: 1, predicate: 2, object: 3 },
            Operation::Checkpoint,
        ];
        
        let result = assembler.assemble(&operations).unwrap();
        assert!(!result.bytecode.is_empty());
        assert_eq!(result.metadata.operations_count, 3);
        
        let disassembled = assembler.disassemble(&result.bytecode).unwrap();
        assert_eq!(disassembled.len(), 3);
    }

    #[test]
    fn test_assembler_checksum() {
        let mut assembler = Assembler::new();
        
        let operations = vec![Operation::Checkpoint];
        let result = assembler.assemble(&operations).unwrap();
        
        // Corrupt the bytecode
        let mut corrupted = result.bytecode.clone();
        corrupted[8] = corrupted[8].wrapping_add(1);
        
        // Should fail checksum validation
        assert!(assembler.disassemble(&corrupted).is_err());
    }
}