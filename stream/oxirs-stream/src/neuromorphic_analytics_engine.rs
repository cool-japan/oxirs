//! Neuromorphic Analytics Engine
//!
//! Spiking neural network engine: Leaky Integrate-and-Fire (LIF) neuron dynamics,
//! spike generation, reservoir computing, and the main analytics processing pipeline.
//!
//! This module re-exports from [`crate::neuromorphic_analytics_network`], which
//! contains the full implementation.

pub use crate::neuromorphic_analytics_network::{
    NeuralStateMachine, NeuralStateMachines, NeuromorphicAnalytics, NeuromorphicMemory,
    PopulationDynamics, SpikeNeuralNetwork, SynapticPlasticity, TemporalPatternRecognizer,
};
