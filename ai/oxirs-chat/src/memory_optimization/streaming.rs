//! Streaming processing for large datasets

use anyhow::Result;
use std::marker::PhantomData;

/// Stream processor for memory-efficient data processing
pub struct StreamProcessor<T> {
    chunk_size: usize,
    _phantom: PhantomData<T>,
}

impl<T> StreamProcessor<T> {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            _phantom: PhantomData,
        }
    }

    /// Process data in chunks
    pub fn process_chunks<F, R>(&self, data: Vec<T>, mut processor: F) -> Result<Vec<R>>
    where
        F: FnMut(&[T]) -> Result<Vec<R>>,
    {
        let mut results = Vec::new();

        for chunk in data.chunks(self.chunk_size) {
            let chunk_results = processor(chunk)?;
            results.extend(chunk_results);
        }

        Ok(results)
    }

    /// Process data with streaming iterator
    pub fn process_stream<I, F, R>(&self, iterator: I, mut processor: F) -> Result<Vec<R>>
    where
        I: Iterator<Item = T>,
        F: FnMut(Vec<T>) -> Result<Vec<R>>,
    {
        let mut results = Vec::new();
        let mut buffer = Vec::with_capacity(self.chunk_size);

        for item in iterator {
            buffer.push(item);

            if buffer.len() >= self.chunk_size {
                let chunk_results = processor(buffer)?;
                results.extend(chunk_results);
                buffer = Vec::with_capacity(self.chunk_size);
            }
        }

        // Process remaining items
        if !buffer.is_empty() {
            let chunk_results = processor(buffer)?;
            results.extend(chunk_results);
        }

        Ok(results)
    }
}

/// Chunk processor for embeddings
pub struct ChunkProcessor {
    max_memory_mb: usize,
    estimated_item_size: usize,
}

impl ChunkProcessor {
    pub fn new(max_memory_mb: usize, estimated_item_size: usize) -> Self {
        Self {
            max_memory_mb,
            estimated_item_size,
        }
    }

    /// Calculate optimal chunk size based on memory constraints
    pub fn optimal_chunk_size(&self) -> usize {
        let max_bytes = self.max_memory_mb * 1024 * 1024;
        let chunk_size = max_bytes / self.estimated_item_size;
        chunk_size.max(1) // At least 1 item per chunk
    }

    /// Process embeddings in memory-efficient chunks
    pub fn process_embeddings<F>(
        &self,
        texts: Vec<String>,
        mut embed_fn: F,
    ) -> Result<Vec<Vec<f32>>>
    where
        F: FnMut(&[String]) -> Result<Vec<Vec<f32>>>,
    {
        let chunk_size = self.optimal_chunk_size();
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(chunk_size) {
            let chunk_embeddings = embed_fn(chunk)?;
            all_embeddings.extend(chunk_embeddings);
        }

        Ok(all_embeddings)
    }
}

/// Streaming aggregator for reducing memory footprint
pub struct StreamingAggregator<T> {
    state: T,
}

impl<T> StreamingAggregator<T> {
    pub fn new(initial_state: T) -> Self {
        Self {
            state: initial_state,
        }
    }

    /// Aggregate stream of data without storing all items
    pub fn aggregate<I, F>(&mut self, iterator: I, mut aggregator: F) -> &T
    where
        I: Iterator,
        F: FnMut(&mut T, I::Item),
    {
        for item in iterator {
            aggregator(&mut self.state, item);
        }
        &self.state
    }

    pub fn state(&self) -> &T {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_processor_chunks() {
        let processor = StreamProcessor::new(10);
        let data: Vec<i32> = (0..25).collect();

        let results = processor
            .process_chunks(data, |chunk| Ok(chunk.iter().map(|x| x * 2).collect()))
            .unwrap();

        assert_eq!(results.len(), 25);
        assert_eq!(results[0], 0);
        assert_eq!(results[24], 48);
    }

    #[test]
    fn test_stream_processor_iterator() {
        let processor = StreamProcessor::new(5);
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        let results = processor
            .process_stream(data.into_iter(), |chunk| {
                Ok(chunk.iter().map(|x| x * 2).collect())
            })
            .unwrap();

        assert_eq!(results.len(), 10);
        assert_eq!(results[0], 2);
        assert_eq!(results[9], 20);
    }

    #[test]
    fn test_chunk_processor_optimal_size() {
        // Assume 100MB max, 1KB per item
        let processor = ChunkProcessor::new(100, 1024);
        let chunk_size = processor.optimal_chunk_size();

        assert_eq!(chunk_size, 102400); // 100MB / 1KB = 102400 items
    }

    #[test]
    fn test_streaming_aggregator() {
        let mut aggregator = StreamingAggregator::new(0i32);

        let data = vec![1, 2, 3, 4, 5];
        let result = aggregator.aggregate(data.into_iter(), |state, item| {
            *state += item;
        });

        assert_eq!(*result, 15);
    }

    #[test]
    fn test_chunk_processor_embeddings() {
        let processor = ChunkProcessor::new(10, 1000);
        let texts: Vec<String> = (0..50).map(|i| format!("text_{}", i)).collect();

        let result = processor.process_embeddings(texts, |chunk| {
            // Simulate embedding generation
            Ok(chunk
                .iter()
                .map(|_| vec![1.0, 2.0, 3.0])
                .collect())
        });

        assert!(result.is_ok());
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 50);
    }
}
