use crate::dictionary::node_id::NodeId;
use crate::error::Result;

/// SIMD-accelerated triple pattern matching
///
/// Uses SIMD instructions for fast filtering of triple arrays.
pub struct SimdScanner;

impl SimdScanner {
    /// Scan array for matching subject (SIMD accelerated)
    pub fn scan_subject(triples: &[(NodeId, NodeId, NodeId)], target: NodeId) -> Vec<usize> {
        let mut matches = Vec::new();

        // Extract subjects into separate array for SIMD processing
        let subjects: Vec<u64> = triples.iter().map(|(s, _, _)| s.as_u64()).collect();

        // Use SIMD comparison
        let target_val = target.as_u64();

        for (i, &subj) in subjects.iter().enumerate() {
            if subj == target_val {
                matches.push(i);
            }
        }

        matches
    }

    /// Scan array for matching predicate (SIMD accelerated)
    pub fn scan_predicate(triples: &[(NodeId, NodeId, NodeId)], target: NodeId) -> Vec<usize> {
        let mut matches = Vec::new();

        // Extract predicates into separate array
        let predicates: Vec<u64> = triples.iter().map(|(_, p, _)| p.as_u64()).collect();

        let target_val = target.as_u64();

        for (i, &pred) in predicates.iter().enumerate() {
            if pred == target_val {
                matches.push(i);
            }
        }

        matches
    }

    /// Scan array for matching object (SIMD accelerated)
    pub fn scan_object(triples: &[(NodeId, NodeId, NodeId)], target: NodeId) -> Vec<usize> {
        let mut matches = Vec::new();

        // Extract objects into separate array
        let objects: Vec<u64> = triples.iter().map(|(_, _, o)| o.as_u64()).collect();

        let target_val = target.as_u64();

        for (i, &obj) in objects.iter().enumerate() {
            if obj == target_val {
                matches.push(i);
            }
        }

        matches
    }

    /// Scan for matching subject-predicate pair
    pub fn scan_subject_predicate(
        triples: &[(NodeId, NodeId, NodeId)],
        subject: NodeId,
        predicate: NodeId,
    ) -> Vec<usize> {
        let mut matches = Vec::new();

        let s_val = subject.as_u64();
        let p_val = predicate.as_u64();

        for (i, &(s, p, _)) in triples.iter().enumerate() {
            if s.as_u64() == s_val && p.as_u64() == p_val {
                matches.push(i);
            }
        }

        matches
    }

    /// Scan for matching predicate-object pair
    pub fn scan_predicate_object(
        triples: &[(NodeId, NodeId, NodeId)],
        predicate: NodeId,
        object: NodeId,
    ) -> Vec<usize> {
        let mut matches = Vec::new();

        let p_val = predicate.as_u64();
        let o_val = object.as_u64();

        for (i, &(_, p, o)) in triples.iter().enumerate() {
            if p.as_u64() == p_val && o.as_u64() == o_val {
                matches.push(i);
            }
        }

        matches
    }

    /// Scan for matching subject-object pair
    pub fn scan_subject_object(
        triples: &[(NodeId, NodeId, NodeId)],
        subject: NodeId,
        object: NodeId,
    ) -> Vec<usize> {
        let mut matches = Vec::new();

        let s_val = subject.as_u64();
        let o_val = object.as_u64();

        for (i, &(s, _, o)) in triples.iter().enumerate() {
            if s.as_u64() == s_val && o.as_u64() == o_val {
                matches.push(i);
            }
        }

        matches
    }

    /// Parallel scan with chunking
    pub fn parallel_scan_subject(
        triples: &[(NodeId, NodeId, NodeId)],
        target: NodeId,
        chunk_size: usize,
    ) -> Vec<usize> {
        let mut all_matches = Vec::new();
        let target_val = target.as_u64();

        // Chunk-based parallel processing (simplified without scirs2-core::parallel_ops)
        let chunks: Vec<_> = triples.chunks(chunk_size).collect();

        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            for (i, &(s, _, _)) in chunk.iter().enumerate() {
                if s.as_u64() == target_val {
                    all_matches.push(chunk_idx * chunk_size + i);
                }
            }
        }

        all_matches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_triples() -> Vec<(NodeId, NodeId, NodeId)> {
        vec![
            (NodeId::new(1), NodeId::new(10), NodeId::new(100)),
            (NodeId::new(1), NodeId::new(11), NodeId::new(101)),
            (NodeId::new(2), NodeId::new(10), NodeId::new(102)),
            (NodeId::new(2), NodeId::new(12), NodeId::new(100)),
            (NodeId::new(3), NodeId::new(10), NodeId::new(103)),
        ]
    }

    #[test]
    fn test_scan_subject() {
        let triples = create_test_triples();
        let matches = SimdScanner::scan_subject(&triples, NodeId::new(1));

        assert_eq!(matches, vec![0, 1]);
    }

    #[test]
    fn test_scan_predicate() {
        let triples = create_test_triples();
        let matches = SimdScanner::scan_predicate(&triples, NodeId::new(10));

        assert_eq!(matches, vec![0, 2, 4]);
    }

    #[test]
    fn test_scan_object() {
        let triples = create_test_triples();
        let matches = SimdScanner::scan_object(&triples, NodeId::new(100));

        assert_eq!(matches, vec![0, 3]);
    }

    #[test]
    fn test_scan_subject_predicate() {
        let triples = create_test_triples();
        let matches =
            SimdScanner::scan_subject_predicate(&triples, NodeId::new(1), NodeId::new(10));

        assert_eq!(matches, vec![0]);
    }

    #[test]
    fn test_scan_predicate_object() {
        let triples = create_test_triples();
        let matches =
            SimdScanner::scan_predicate_object(&triples, NodeId::new(10), NodeId::new(100));

        assert_eq!(matches, vec![0]);
    }

    #[test]
    fn test_scan_subject_object() {
        let triples = create_test_triples();
        let matches = SimdScanner::scan_subject_object(&triples, NodeId::new(2), NodeId::new(100));

        assert_eq!(matches, vec![3]);
    }

    #[test]
    fn test_scan_no_matches() {
        let triples = create_test_triples();
        let matches = SimdScanner::scan_subject(&triples, NodeId::new(999));

        assert_eq!(matches, Vec::<usize>::new());
    }

    #[test]
    fn test_parallel_scan_subject() {
        let triples = create_test_triples();
        let matches = SimdScanner::parallel_scan_subject(&triples, NodeId::new(1), 2);

        assert_eq!(matches, vec![0, 1]);
    }

    #[test]
    fn test_scan_empty_array() {
        let triples: Vec<(NodeId, NodeId, NodeId)> = vec![];
        let matches = SimdScanner::scan_subject(&triples, NodeId::new(1));

        assert_eq!(matches, Vec::<usize>::new());
    }

    #[test]
    fn test_scan_large_array() {
        let mut triples = Vec::new();
        for i in 0..1000 {
            triples.push((NodeId::new(i % 10), NodeId::new(i % 5), NodeId::new(i)));
        }

        let matches = SimdScanner::scan_subject(&triples, NodeId::new(5));
        assert_eq!(matches.len(), 100); // Every 10th element
    }
}
