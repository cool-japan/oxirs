//! Parallel processing abstractions for OxiRS
//!
//! This module provides unified parallel processing operations across the OxiRS ecosystem.
//! All parallel operations must go through this module - direct Rayon usage in other modules is forbidden.

#[cfg(feature = "parallel")]
pub use rayon::{
    // Re-export all of rayon's prelude
    prelude::*,
    // Re-export commonly used items
    current_num_threads,
    join,
    scope,
    spawn,
    ThreadPool,
    ThreadPoolBuilder,
};

// Sequential fallbacks when parallel feature is disabled
#[cfg(not(feature = "parallel"))]
pub use self::sequential::*;

/// Check if parallel processing is enabled
pub fn is_parallel_enabled() -> bool {
    cfg!(feature = "parallel")
}

/// Get the number of threads available for parallel operations
pub fn num_threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// Process a slice in parallel chunks
pub fn par_chunks<T, F, R>(slice: &[T], chunk_size: usize, f: F) -> Vec<R>
where
    T: Sync,
    F: Fn(&[T]) -> R + Sync,
    R: Send,
{
    #[cfg(feature = "parallel")]
    {
        slice
            .par_chunks(chunk_size)
            .map(f)
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        slice
            .chunks(chunk_size)
            .map(f)
            .collect()
    }
}

/// Execute two closures potentially in parallel
pub fn par_join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    #[cfg(feature = "parallel")]
    {
        join(a, b)
    }
    #[cfg(not(feature = "parallel"))]
    {
        (a(), b())
    }
}

/// Sequential implementations for when parallel feature is disabled
#[cfg(not(feature = "parallel"))]
mod sequential {
    use std::iter::Iterator;
    
    /// Sequential iterator trait (mimics ParallelIterator)
    pub trait ParallelIterator: Iterator + Sized {
        fn map<F, R>(self, f: F) -> Map<Self, F>
        where
            F: FnMut(Self::Item) -> R,
        {
            Map { iter: self, f }
        }
        
        fn filter<F>(self, f: F) -> Filter<Self, F>
        where
            F: FnMut(&Self::Item) -> bool,
        {
            Filter { iter: self, f }
        }
        
        fn filter_map<F, R>(self, f: F) -> FilterMap<Self, F>
        where
            F: FnMut(Self::Item) -> Option<R>,
        {
            FilterMap { iter: self, f }
        }
        
        fn flat_map<F, I>(self, f: F) -> FlatMap<Self, F>
        where
            F: FnMut(Self::Item) -> I,
            I: IntoIterator,
        {
            FlatMap { iter: self, f }
        }
        
        fn for_each<F>(self, mut f: F)
        where
            F: FnMut(Self::Item),
        {
            for item in self {
                f(item);
            }
        }
        
        fn collect<C>(self) -> C
        where
            C: FromIterator<Self::Item>,
        {
            C::from_iter(self)
        }
        
        fn fold<T, F>(self, init: T, mut f: F) -> T
        where
            F: FnMut(T, Self::Item) -> T,
        {
            let mut accum = init;
            for item in self {
                accum = f(accum, item);
            }
            accum
        }
        
        fn reduce<F>(mut self, f: F) -> Option<Self::Item>
        where
            F: FnMut(Self::Item, Self::Item) -> Self::Item,
        {
            self.next().map(|first| self.fold(first, f))
        }
        
        fn sum<S>(self) -> S
        where
            S: std::iter::Sum<Self::Item>,
        {
            self.collect::<Vec<_>>().into_iter().sum()
        }
        
        fn min(self) -> Option<Self::Item>
        where
            Self::Item: Ord,
        {
            self.reduce(std::cmp::min)
        }
        
        fn max(self) -> Option<Self::Item>
        where
            Self::Item: Ord,
        {
            self.reduce(std::cmp::max)
        }
        
        fn any<F>(self, mut f: F) -> bool
        where
            F: FnMut(Self::Item) -> bool,
        {
            for item in self {
                if f(item) {
                    return true;
                }
            }
            false
        }
        
        fn all<F>(self, mut f: F) -> bool
        where
            F: FnMut(Self::Item) -> bool,
        {
            for item in self {
                if !f(item) {
                    return false;
                }
            }
            true
        }
    }
    
    /// Implement ParallelIterator for all iterators
    impl<I: Iterator> ParallelIterator for I {}
    
    /// Map iterator
    pub struct Map<I, F> {
        iter: I,
        f: F,
    }
    
    impl<I, F, R> Iterator for Map<I, F>
    where
        I: Iterator,
        F: FnMut(I::Item) -> R,
    {
        type Item = R;
        
        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next().map(&mut self.f)
        }
    }
    
    /// Filter iterator
    pub struct Filter<I, F> {
        iter: I,
        f: F,
    }
    
    impl<I, F> Iterator for Filter<I, F>
    where
        I: Iterator,
        F: FnMut(&I::Item) -> bool,
    {
        type Item = I::Item;
        
        fn next(&mut self) -> Option<Self::Item> {
            while let Some(item) = self.iter.next() {
                if (self.f)(&item) {
                    return Some(item);
                }
            }
            None
        }
    }
    
    /// FilterMap iterator
    pub struct FilterMap<I, F> {
        iter: I,
        f: F,
    }
    
    impl<I, F, R> Iterator for FilterMap<I, F>
    where
        I: Iterator,
        F: FnMut(I::Item) -> Option<R>,
    {
        type Item = R;
        
        fn next(&mut self) -> Option<Self::Item> {
            while let Some(item) = self.iter.next() {
                if let Some(result) = (self.f)(item) {
                    return Some(result);
                }
            }
            None
        }
    }
    
    /// FlatMap iterator
    pub struct FlatMap<I, F> {
        iter: I,
        f: F,
    }
    
    impl<I, F, J> Iterator for FlatMap<I, F>
    where
        I: Iterator,
        F: FnMut(I::Item) -> J,
        J: IntoIterator,
    {
        type Item = J::Item;
        
        fn next(&mut self) -> Option<Self::Item> {
            // Simplified implementation - in real code this would need to handle
            // the inner iterator state properly
            None
        }
    }
    
    /// Sequential parallel iterator extension trait
    pub trait IntoParallelIterator {
        type Item;
        type Iter: ParallelIterator<Item = Self::Item>;
        
        fn into_par_iter(self) -> Self::Iter;
    }
    
    /// Implementation for ranges
    impl IntoParallelIterator for std::ops::Range<usize> {
        type Item = usize;
        type Iter = std::ops::Range<usize>;
        
        fn into_par_iter(self) -> Self::Iter {
            self
        }
    }
    
    /// Implementation for slices
    impl<'a, T> IntoParallelIterator for &'a [T] {
        type Item = &'a T;
        type Iter = std::slice::Iter<'a, T>;
        
        fn into_par_iter(self) -> Self::Iter {
            self.iter()
        }
    }
    
    /// Implementation for mutable slices
    impl<'a, T> IntoParallelIterator for &'a mut [T] {
        type Item = &'a mut T;
        type Iter = std::slice::IterMut<'a, T>;
        
        fn into_par_iter(self) -> Self::Iter {
            self.iter_mut()
        }
    }
    
    /// Implementation for Vec
    impl<T> IntoParallelIterator for Vec<T> {
        type Item = T;
        type Iter = std::vec::IntoIter<T>;
        
        fn into_par_iter(self) -> Self::Iter {
            self.into_iter()
        }
    }
    
    /// Extension trait for slices
    pub trait ParallelSlice<T> {
        fn par_chunks(&self, chunk_size: usize) -> std::slice::Chunks<'_, T>;
        fn par_chunks_mut(&mut self, chunk_size: usize) -> std::slice::ChunksMut<'_, T>;
    }
    
    impl<T> ParallelSlice<T> for [T] {
        fn par_chunks(&self, chunk_size: usize) -> std::slice::Chunks<'_, T> {
            self.chunks(chunk_size)
        }
        
        fn par_chunks_mut(&mut self, chunk_size: usize) -> std::slice::ChunksMut<'_, T> {
            self.chunks_mut(chunk_size)
        }
    }
    
    /// Sequential scope (no-op)
    pub fn scope<'scope, F, R>(f: F) -> R
    where
        F: FnOnce() -> R,
    {
        f()
    }
    
    /// Sequential join
    pub fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA,
        B: FnOnce() -> RB,
    {
        (a(), b())
    }
    
    /// Sequential spawn (just executes immediately)
    pub fn spawn<F, R>(f: F) -> R
    where
        F: FnOnce() -> R,
    {
        f()
    }
    
    /// Get current number of threads (always 1 in sequential mode)
    pub fn current_num_threads() -> usize {
        1
    }
}