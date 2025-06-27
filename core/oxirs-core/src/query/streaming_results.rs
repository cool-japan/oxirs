//! Streaming result sets for large query results with minimal memory overhead

use crate::model::{Term, Variable, Triple, Quad, NamedNode, BlankNode, Literal};
use crate::OxirsError;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tokio::sync::mpsc;
use futures::stream::{Stream, StreamExt};
use std::pin::Pin;
use std::task::{Context, Poll};
use std::collections::HashMap;
use parking_lot::RwLock;
use crossbeam::channel;
use std::time::{Duration, Instant};

/// Configuration for streaming result sets
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Buffer size for internal channels
    pub buffer_size: usize,
    /// Maximum memory usage in bytes
    pub max_memory: usize,
    /// Enable progress tracking
    pub track_progress: bool,
    /// Backpressure threshold (0.0 - 1.0)
    pub backpressure_threshold: f64,
    /// Timeout for blocking operations
    pub timeout: Option<Duration>,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1024,
            max_memory: 100 * 1024 * 1024, // 100MB
            track_progress: true,
            backpressure_threshold: 0.8,
            timeout: Some(Duration::from_secs(30)), // 30 seconds
        }
    }
}

/// Progress information for streaming queries
#[derive(Debug, Clone)]
pub struct StreamingProgress {
    /// Total results processed
    pub processed: usize,
    /// Estimated total results (if known)
    pub estimated_total: Option<usize>,
    /// Current memory usage
    pub memory_used: usize,
    /// Query start time
    pub start_time: Instant,
    /// Is query still running
    pub is_running: bool,
}

/// A single solution (row) in a SELECT query result
#[derive(Debug, Clone)]
pub struct Solution {
    /// Variable bindings for this solution
    bindings: HashMap<Variable, Option<Term>>,
    /// Metadata about this solution
    metadata: SolutionMetadata,
}

#[derive(Debug, Clone, Default)]
pub struct SolutionMetadata {
    /// Source of this solution (for federated queries)
    pub source: Option<String>,
    /// Confidence score (for fuzzy queries)
    pub confidence: Option<f64>,
    /// Solution timestamp
    pub timestamp: Option<u64>,
}

impl Solution {
    pub fn new(bindings: HashMap<Variable, Option<Term>>) -> Self {
        Self {
            bindings,
            metadata: SolutionMetadata::default(),
        }
    }

    pub fn with_metadata(
        bindings: HashMap<Variable, Option<Term>>,
        metadata: SolutionMetadata,
    ) -> Self {
        Self { bindings, metadata }
    }

    pub fn get(&self, var: &Variable) -> Option<&Term> {
        self.bindings.get(var).and_then(|opt| opt.as_ref())
    }

    pub fn contains(&self, var: &Variable) -> bool {
        self.bindings.contains_key(var)
    }

    pub fn variables(&self) -> impl Iterator<Item = &Variable> {
        self.bindings.keys()
    }

    pub fn values(&self) -> impl Iterator<Item = &Term> {
        self.bindings.values().filter_map(|opt| opt.as_ref())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Variable, Option<&Term>)> {
        self.bindings.iter().map(|(k, v)| (k, v.as_ref()))
    }
}

/// Streaming iterator for SELECT query results
pub struct SelectResults {
    /// Variables in the result set
    variables: Arc<Vec<Variable>>,
    /// Channel receiver for solutions
    receiver: channel::Receiver<Result<Solution, OxirsError>>,
    /// Progress tracker
    progress: Arc<RwLock<StreamingProgress>>,
    /// Cancellation token
    cancel_token: Arc<AtomicBool>,
    /// Current buffer for batch operations
    buffer: Vec<Solution>,
    /// Configuration
    config: StreamingConfig,
}

impl SelectResults {
    pub fn new(
        variables: Vec<Variable>,
        receiver: channel::Receiver<Result<Solution, OxirsError>>,
        config: StreamingConfig,
    ) -> Self {
        let progress = Arc::new(RwLock::new(StreamingProgress {
            processed: 0,
            estimated_total: None,
            memory_used: 0,
            start_time: Instant::now(),
            is_running: true,
        }));

        Self {
            variables: Arc::new(variables),
            receiver,
            progress,
            cancel_token: Arc::new(AtomicBool::new(false)),
            buffer: Vec::with_capacity(config.buffer_size),
            config,
        }
    }

    /// Get the variables in the result set
    pub fn variables(&self) -> &[Variable] {
        &self.variables
    }

    /// Get current progress information
    pub fn progress(&self) -> StreamingProgress {
        self.progress.read().clone()
    }

    /// Cancel the query execution
    pub fn cancel(&self) {
        self.cancel_token.store(true, Ordering::Relaxed);
    }

    /// Check if query was cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.load(Ordering::Relaxed)
    }

    /// Try to get the next solution without blocking
    pub fn try_next(&mut self) -> Result<Option<Solution>, OxirsError> {
        if self.is_cancelled() {
            return Ok(None);
        }

        match self.receiver.try_recv() {
            Ok(Ok(solution)) => {
                self.update_progress(1);
                Ok(Some(solution))
            }
            Ok(Err(e)) => Err(e),
            Err(channel::TryRecvError::Empty) => Ok(None),
            Err(channel::TryRecvError::Disconnected) => {
                self.mark_completed();
                Ok(None)
            }
        }
    }

    /// Get the next solution, blocking if necessary
    pub fn next(&mut self) -> Result<Option<Solution>, OxirsError> {
        if self.is_cancelled() {
            return Ok(None);
        }

        if let Some(timeout) = self.config.timeout {
            match self.receiver.recv_timeout(timeout) {
                Ok(Ok(solution)) => {
                    self.update_progress(1);
                    Ok(Some(solution))
                }
                Ok(Err(e)) => Err(e),
                Err(channel::RecvTimeoutError::Timeout) => {
                    Err(OxirsError::Query("Query timeout".to_string()))
                }
                Err(channel::RecvTimeoutError::Disconnected) => {
                    self.mark_completed();
                    Ok(None)
                }
            }
        } else {
            // No timeout - block indefinitely
            match self.receiver.recv() {
                Ok(Ok(solution)) => {
                    self.update_progress(1);
                    Ok(Some(solution))
                }
                Ok(Err(e)) => Err(e),
                Err(channel::RecvError) => {
                    self.mark_completed();
                    Ok(None)
                }
            }
        }
    }

    /// Collect next batch of solutions
    pub fn next_batch(&mut self, max_size: usize) -> Result<Vec<Solution>, OxirsError> {
        self.buffer.clear();
        
        for _ in 0..max_size {
            match self.try_next()? {
                Some(solution) => self.buffer.push(solution),
                None => break,
            }
        }

        Ok(std::mem::take(&mut self.buffer))
    }

    /// Skip n solutions
    pub fn skip_results(&mut self, n: usize) -> Result<(), OxirsError> {
        for _ in 0..n {
            if self.next()?.is_none() {
                break;
            }
        }
        Ok(())
    }

    /// Take up to n solutions
    pub fn take_results(&mut self, n: usize) -> Result<Vec<Solution>, OxirsError> {
        let mut results = Vec::with_capacity(n.min(self.config.buffer_size));
        
        for _ in 0..n {
            match self.next()? {
                Some(solution) => results.push(solution),
                None => break,
            }
        }

        Ok(results)
    }

    /// Convert to a stream for async iteration
    pub fn into_stream(self) -> impl Stream<Item = Result<Solution, OxirsError>> {
        SelectResultStream::new(self)
    }

    fn update_progress(&self, count: usize) {
        let mut progress = self.progress.write();
        progress.processed += count;
        // Update memory usage estimate (rough approximation)
        progress.memory_used = progress.processed * std::mem::size_of::<Solution>();
    }

    fn mark_completed(&self) {
        let mut progress = self.progress.write();
        progress.is_running = false;
    }
}

impl Iterator for SelectResults {
    type Item = Result<Solution, OxirsError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next() {
            Ok(Some(solution)) => Some(Ok(solution)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Async stream wrapper for SelectResults
struct SelectResultStream {
    results: SelectResults,
    receiver: Option<mpsc::UnboundedReceiver<Result<Solution, OxirsError>>>,
}

impl SelectResultStream {
    fn new(results: SelectResults) -> Self {
        Self {
            results,
            receiver: None,
        }
    }
}

impl Stream for SelectResultStream {
    type Item = Result<Solution, OxirsError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Try non-blocking first
        match self.results.try_next() {
            Ok(Some(solution)) => return Poll::Ready(Some(Ok(solution))),
            Ok(None) => {
                // Would block, need to wait
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }
            Err(e) => return Poll::Ready(Some(Err(e))),
        }
    }
}

/// Streaming iterator for CONSTRUCT query results
pub struct ConstructResults {
    receiver: channel::Receiver<Result<Triple, OxirsError>>,
    progress: Arc<RwLock<StreamingProgress>>,
    cancel_token: Arc<AtomicBool>,
    config: StreamingConfig,
}

impl ConstructResults {
    pub fn new(
        receiver: channel::Receiver<Result<Triple, OxirsError>>,
        config: StreamingConfig,
    ) -> Self {
        let progress = Arc::new(RwLock::new(StreamingProgress {
            processed: 0,
            estimated_total: None,
            memory_used: 0,
            start_time: Instant::now(),
            is_running: true,
        }));

        Self {
            receiver,
            progress,
            cancel_token: Arc::new(AtomicBool::new(false)),
            config,
        }
    }

    pub fn progress(&self) -> StreamingProgress {
        self.progress.read().clone()
    }

    pub fn cancel(&self) {
        self.cancel_token.store(true, Ordering::Relaxed);
    }

    pub fn next(&mut self) -> Result<Option<Triple>, OxirsError> {
        if self.cancel_token.load(Ordering::Relaxed) {
            return Ok(None);
        }

        if let Some(timeout) = self.config.timeout {
            match self.receiver.recv_timeout(timeout) {
                Ok(Ok(triple)) => {
                    self.update_progress(1);
                    Ok(Some(triple))
                }
                Ok(Err(e)) => Err(e),
                Err(channel::RecvTimeoutError::Timeout) => {
                    Err(OxirsError::Query("Query timeout".to_string()))
                }
                Err(channel::RecvTimeoutError::Disconnected) => {
                    self.mark_completed();
                    Ok(None)
                }
            }
        } else {
            // No timeout - block indefinitely
            match self.receiver.recv() {
                Ok(Ok(triple)) => {
                    self.update_progress(1);
                    Ok(Some(triple))
                }
                Ok(Err(e)) => Err(e),
                Err(channel::RecvError) => {
                    self.mark_completed();
                    Ok(None)
                }
            }
        }
    }

    pub fn collect_batch(&mut self, max_size: usize) -> Result<Vec<Triple>, OxirsError> {
        let mut batch = Vec::with_capacity(max_size.min(self.config.buffer_size));
        
        for _ in 0..max_size {
            match self.next()? {
                Some(triple) => batch.push(triple),
                None => break,
            }
        }

        Ok(batch)
    }

    fn update_progress(&self, count: usize) {
        let mut progress = self.progress.write();
        progress.processed += count;
        progress.memory_used = progress.processed * std::mem::size_of::<Triple>();
    }

    fn mark_completed(&self) {
        let mut progress = self.progress.write();
        progress.is_running = false;
    }
}

impl Iterator for ConstructResults {
    type Item = Result<Triple, OxirsError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next() {
            Ok(Some(triple)) => Some(Ok(triple)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Streaming query results
pub enum StreamingQueryResults {
    /// SELECT query results
    Select(SelectResults),
    /// ASK query results
    Ask(bool),
    /// CONSTRUCT query results  
    Construct(ConstructResults),
    /// DESCRIBE query results
    Describe(ConstructResults),
}

impl StreamingQueryResults {
    /// Check if results are from a SELECT query
    pub fn is_select(&self) -> bool {
        matches!(self, Self::Select(_))
    }

    /// Check if results are from an ASK query
    pub fn is_ask(&self) -> bool {
        matches!(self, Self::Ask(_))
    }

    /// Check if results are from a CONSTRUCT query
    pub fn is_construct(&self) -> bool {
        matches!(self, Self::Construct(_))
    }

    /// Get SELECT results if applicable
    pub fn as_select(&mut self) -> Option<&mut SelectResults> {
        match self {
            Self::Select(results) => Some(results),
            _ => None,
        }
    }

    /// Get ASK result if applicable
    pub fn as_ask(&self) -> Option<bool> {
        match self {
            Self::Ask(result) => Some(*result),
            _ => None,
        }
    }

    /// Get CONSTRUCT results if applicable
    pub fn as_construct(&mut self) -> Option<&mut ConstructResults> {
        match self {
            Self::Construct(results) => Some(results),
            _ => None,
        }
    }

    /// Cancel the query execution
    pub fn cancel(&self) {
        match self {
            Self::Select(results) => results.cancel(),
            Self::Construct(results) => results.cancel(),
            Self::Describe(results) => results.cancel(),
            Self::Ask(_) => {} // ASK queries complete immediately
        }
    }

    /// Get progress information
    pub fn progress(&self) -> Option<StreamingProgress> {
        match self {
            Self::Select(results) => Some(results.progress()),
            Self::Construct(results) => Some(results.progress()),
            Self::Describe(results) => Some(results.progress()),
            Self::Ask(_) => None,
        }
    }
}

/// Builder for creating streaming result sets
pub struct StreamingResultBuilder {
    config: StreamingConfig,
}

impl StreamingResultBuilder {
    pub fn new() -> Self {
        Self {
            config: StreamingConfig::default(),
        }
    }

    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size;
        self
    }

    pub fn with_max_memory(mut self, bytes: usize) -> Self {
        self.config.max_memory = bytes;
        self
    }

    pub fn with_progress_tracking(mut self, enable: bool) -> Self {
        self.config.track_progress = enable;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = Some(timeout);
        self
    }

    pub fn build_select(
        self,
        variables: Vec<Variable>,
    ) -> (SelectResults, channel::Sender<Result<Solution, OxirsError>>) {
        let (tx, rx) = channel::bounded(self.config.buffer_size);
        let results = SelectResults::new(variables, rx, self.config);
        (results, tx)
    }

    pub fn build_construct(self) -> (ConstructResults, channel::Sender<Result<Triple, OxirsError>>) {
        let (tx, rx) = channel::bounded(self.config.buffer_size);
        let results = ConstructResults::new(rx, self.config);
        (results, tx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solution_creation() {
        let mut bindings = HashMap::new();
        let var = Variable::new("x").unwrap();
        let term = Term::NamedNode(NamedNode::new("http://example.org/test").unwrap());
        bindings.insert(var.clone(), Some(term.clone()));

        let solution = Solution::new(bindings);
        assert_eq!(solution.get(&var), Some(&term));
        assert!(solution.contains(&var));
    }

    #[test]
    fn test_streaming_select_results() {
        let builder = StreamingResultBuilder::new()
            .with_buffer_size(10);
        
        let variables = vec![Variable::new("x").unwrap()];
        let (mut results, sender) = builder.build_select(variables.clone());

        // Send some solutions
        for i in 0..5 {
            let mut bindings = HashMap::new();
            let term = Term::Literal(Literal::new(&i.to_string()));
            bindings.insert(variables[0].clone(), Some(term));
            sender.send(Ok(Solution::new(bindings))).unwrap();
        }
        drop(sender);

        // Process results one by one to track progress
        let mut collected = Vec::new();
        while let Ok(Some(solution)) = results.next() {
            collected.push(solution);
        }
        
        assert_eq!(collected.len(), 5);
        assert_eq!(results.progress().processed, 5);
    }

    #[test]
    fn test_batch_operations() {
        let builder = StreamingResultBuilder::new();
        let variables = vec![Variable::new("x").unwrap()];
        let (mut results, sender) = builder.build_select(variables.clone());

        // Send 20 solutions
        for i in 0..20 {
            let mut bindings = HashMap::new();
            let term = Term::Literal(Literal::new(&i.to_string()));
            bindings.insert(variables[0].clone(), Some(term));
            sender.send(Ok(Solution::new(bindings))).unwrap();
        }
        drop(sender);

        // Take batch of 10
        let batch = results.next_batch(10).unwrap();
        assert_eq!(batch.len(), 10);

        // Skip 5 using our skip method
        results.skip_results(5).unwrap();

        // Take remaining using our take method
        let remaining = results.take_results(10).unwrap();
        assert_eq!(remaining.len(), 5);
    }

    #[test]
    fn test_cancellation() {
        let builder = StreamingResultBuilder::new();
        let variables = vec![Variable::new("x").unwrap()];
        let (mut results, _sender) = builder.build_select(variables);

        results.cancel();
        assert!(results.is_cancelled());
        assert!(results.next().unwrap().is_none());
    }
}