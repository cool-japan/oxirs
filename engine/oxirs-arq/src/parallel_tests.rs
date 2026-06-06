#[cfg(test)]
mod tests {
    use crate::algebra::{Literal, Term, Variable};
    use crate::executor::ParallelConfig;
    use crate::parallel_executor::{ParallelQueryExecutor, WorkStealingQueue};
    use std::collections::HashMap;

    #[test]
    fn test_parallel_executor_creation() {
        let config = ParallelConfig::default();
        let executor = ParallelQueryExecutor::new(config).unwrap();
        assert!(executor.config.max_threads > 0);
    }

    #[test]
    fn test_work_stealing_queue() {
        let queue: WorkStealingQueue<i32> = WorkStealingQueue::new(4);

        // Push to different queues
        queue.push(0, 1);
        queue.push(1, 2);
        queue.push(2, 3);

        // Steal from queue 3 (empty) - should steal from queue 0 first (value 1)
        assert_eq!(queue.steal(3), Some(1)); // Should steal from another queue
    }

    #[test]
    fn test_parallel_distinct() {
        let config = ParallelConfig::default();
        let executor = ParallelQueryExecutor::new(config).unwrap();

        let mut solution = vec![];
        for i in 0..100 {
            let mut binding = HashMap::new();
            binding.insert(
                Variable::new("x").unwrap(),
                Term::Literal(Literal {
                    value: (i % 10).to_string(),
                    language: None,
                    datatype: None,
                }),
            );
            solution.push(binding);
        }

        let distinct = executor.parallel_distinct(solution);
        assert_eq!(distinct.len(), 10);
    }
}
