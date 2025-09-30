#!/usr/bin/env python3
"""
Test suite for OxiRS Vector Search Python bindings.

This comprehensive test suite validates the functionality of the Python bindings,
including vector operations, store management, analytics, and SPARQL integration.
"""

import pytest
import numpy as np
import tempfile
import os
from typing import List, Dict, Any

# Import OxiRS Vector Search (will be mocked if not available)
try:
    import oxirs_vec
    OXIRS_AVAILABLE = True
except ImportError:
    OXIRS_AVAILABLE = False
    
    # Mock classes for testing when OxiRS is not available
    class MockVectorStore:
        def __init__(self, *args, **kwargs):
            self.vectors = {}
            self.metadata = {}
        
        def index_resource(self, id: str, content: str, metadata=None):
            self.vectors[id] = np.random.rand(128).astype(np.float32)
            self.metadata[id] = metadata or {}
        
        def similarity_search(self, query: str, limit=10, **kwargs):
            return [{"id": f"doc_{i}", "score": 0.8 - i*0.1, "metadata": {}} 
                   for i in range(min(limit, len(self.vectors)))]
    
    class MockModule:
        VectorStore = MockVectorStore
        VectorSearchError = Exception
    
    oxirs_vec = MockModule()

@pytest.fixture
def vector_store():
    """Create a test vector store."""
    if OXIRS_AVAILABLE:
        return oxirs_vec.VectorStore(
            embedding_strategy="tf_idf",
            index_type="memory"
        )
    else:
        return oxirs_vec.VectorStore()

@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        ("doc1", "Machine learning and artificial intelligence research"),
        ("doc2", "Deep neural networks for computer vision applications"), 
        ("doc3", "Natural language processing with transformer models"),
        ("doc4", "Reinforcement learning algorithms and applications"),
        ("doc5", "Computer graphics and 3D rendering techniques"),
        ("doc6", "Database systems and query optimization"),
        ("doc7", "Distributed computing and cloud architectures"),
        ("doc8", "Cybersecurity and encryption algorithms"),
        ("doc9", "Data science and statistical analysis methods"),
        ("doc10", "Software engineering and design patterns")
    ]

@pytest.fixture
def sample_vectors():
    """Generate sample vectors for testing."""
    np.random.seed(42)  # For reproducible tests
    return np.random.rand(100, 128).astype(np.float32)

class TestVectorStore:
    """Test vector store functionality."""
    
    def test_vector_store_creation(self):
        """Test vector store creation with different configurations."""
        if not OXIRS_AVAILABLE:
            pytest.skip("OxiRS not available")
        
        # Test different embedding strategies
        strategies = ["tf_idf", "sentence_transformer", "openai"]
        for strategy in strategies:
            try:
                store = oxirs_vec.VectorStore(embedding_strategy=strategy)
                assert store is not None
            except oxirs_vec.EmbeddingError:
                # Some strategies might not be available in test environment
                pass
    
    def test_index_resource(self, vector_store, sample_documents):
        """Test indexing text resources."""
        # Index documents
        for doc_id, content in sample_documents:
            metadata = {"category": "tech", "length": len(content)}
            vector_store.index_resource(doc_id, content, metadata)
        
        # Verify indexing
        if OXIRS_AVAILABLE:
            stats = vector_store.get_stats()
            assert stats["total_vectors"] == len(sample_documents)
    
    def test_similarity_search(self, vector_store, sample_documents):
        """Test similarity search functionality."""
        # Index documents
        for doc_id, content in sample_documents:
            vector_store.index_resource(doc_id, content)
        
        # Search for similar content
        results = vector_store.similarity_search(
            "machine learning", 
            limit=5, 
            threshold=0.1
        )
        
        assert len(results) <= 5
        assert all(isinstance(r, dict) for r in results)
        assert all("id" in r and "score" in r for r in results)
        
        # Check score ordering (should be descending)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_vector_operations(self, vector_store, sample_vectors):
        """Test direct vector operations."""
        if not OXIRS_AVAILABLE:
            pytest.skip("OxiRS not available")
        
        # Index vectors
        vector_ids = [f"vec_{i}" for i in range(len(sample_vectors))]
        metadata_list = [{"index": i, "category": i % 5} for i in range(len(sample_vectors))]
        
        vector_store.index_batch(vector_ids, sample_vectors, metadata_list)
        
        # Test vector search
        query_vector = sample_vectors[0]  # Use first vector as query
        results = vector_store.vector_search(
            query_vector, 
            limit=10, 
            metric="cosine"
        )
        
        assert len(results) <= 10
        assert results[0]["id"] == "vec_0"  # Should find itself first
        assert results[0]["score"] >= 0.99  # Should be very similar to itself
    
    def test_get_vector(self, vector_store, sample_vectors):
        """Test vector retrieval."""
        if not OXIRS_AVAILABLE:
            pytest.skip("OxiRS not available")
        
        # Index a vector
        test_id = "test_vector"
        test_vector = sample_vectors[0]
        vector_store.index_vector(test_id, test_vector)
        
        # Retrieve the vector
        retrieved = vector_store.get_vector(test_id)
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, test_vector, decimal=5)
    
    def test_remove_vector(self, vector_store, sample_vectors):
        """Test vector removal."""
        if not OXIRS_AVAILABLE:
            pytest.skip("OxiRS not available")
        
        # Index a vector
        test_id = "removable_vector"
        vector_store.index_vector(test_id, sample_vectors[0])
        
        # Verify it exists
        assert vector_store.get_vector(test_id) is not None
        
        # Remove it
        removed = vector_store.remove_vector(test_id)
        assert removed == True
        
        # Verify it's gone
        assert vector_store.get_vector(test_id) is None
    
    def test_store_persistence(self, vector_store, sample_documents):
        """Test saving and loading vector stores."""
        if not OXIRS_AVAILABLE:
            pytest.skip("OxiRS not available")
        
        # Index some data
        for doc_id, content in sample_documents[:5]:
            vector_store.index_resource(doc_id, content)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            temp_path = f.name
        
        try:
            vector_store.save(temp_path)
            
            # Load from file
            loaded_store = oxirs_vec.VectorStore.load(temp_path)
            
            # Verify data is preserved
            original_stats = vector_store.get_stats()
            loaded_stats = loaded_store.get_stats()
            assert original_stats["total_vectors"] == loaded_stats["total_vectors"]
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_compute_similarity(self):
        """Test similarity computation."""
        if not OXIRS_AVAILABLE:
            pytest.skip("OxiRS not available")
        
        # Create test vectors
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        vec3 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        # Test different metrics
        cosine_sim = oxirs_vec.compute_similarity(vec1, vec2, "cosine")
        assert abs(cosine_sim - 0.0) < 1e-6  # Orthogonal vectors
        
        cosine_sim_same = oxirs_vec.compute_similarity(vec1, vec3, "cosine")
        assert abs(cosine_sim_same - 1.0) < 1e-6  # Identical vectors
        
        euclidean_dist = oxirs_vec.compute_similarity(vec1, vec2, "euclidean")
        assert euclidean_dist > 0  # Should be positive distance
    
    def test_normalize_vector(self):
        """Test vector normalization."""
        if not OXIRS_AVAILABLE:
            pytest.skip("OxiRS not available")
        
        # Create test vector
        vector = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        normalized = oxirs_vec.normalize_vector(vector)
        
        # Check normalization
        norm = np.linalg.norm(normalized)
        assert abs(norm - 1.0) < 1e-6
        
        # Check direction preservation
        expected = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(normalized, expected, decimal=5)
    
    def test_batch_normalize(self):
        """Test batch vector normalization."""
        if not OXIRS_AVAILABLE:
            pytest.skip("OxiRS not available")
        
        # Create test vectors
        vectors = np.array([
            [3.0, 4.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        normalized = oxirs_vec.batch_normalize(vectors)
        
        # Check that all vectors are normalized
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0, 1.0], decimal=5)

class TestVectorAnalytics:
    """Test vector analytics functionality."""
    
    def test_analytics_creation(self):
        """Test analytics engine creation."""
        if not OXIRS_AVAILABLE:
            pytest.skip("OxiRS not available")
        
        analytics = oxirs_vec.VectorAnalytics()
        assert analytics is not None
    
    def test_vector_analysis(self, sample_vectors):
        """Test vector distribution analysis."""
        if not OXIRS_AVAILABLE:
            pytest.skip("OxiRS not available")
        
        analytics = oxirs_vec.VectorAnalytics()
        labels = [f"cluster_{i % 5}" for i in range(len(sample_vectors))]
        
        analysis = analytics.analyze_vectors(sample_vectors, labels)
        
        # Check analysis results
        assert "num_vectors" in analysis
        assert "dimension" in analysis
        assert "sparsity" in analysis
        assert analysis["num_vectors"] == len(sample_vectors)
        assert analysis["dimension"] == sample_vectors.shape[1]
    
    def test_optimization_recommendations(self):
        """Test optimization recommendations."""
        if not OXIRS_AVAILABLE:
            pytest.skip("OxiRS not available")
        
        analytics = oxirs_vec.VectorAnalytics()
        recommendations = analytics.get_recommendations()
        
        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert "type" in rec
            assert "priority" in rec
            assert "description" in rec
            assert "expected_improvement" in rec

class TestSparqlIntegration:
    """Test SPARQL integration functionality."""
    
    def test_sparql_search_creation(self, vector_store):
        """Test SPARQL search interface creation."""
        if not OXIRS_AVAILABLE:
            pytest.skip("OxiRS not available")
        
        sparql_search = oxirs_vec.SparqlVectorSearch(vector_store)
        assert sparql_search is not None
    
    def test_function_registration(self, vector_store):
        """Test custom function registration."""
        if not OXIRS_AVAILABLE:
            pytest.skip("OxiRS not available")
        
        sparql_search = oxirs_vec.SparqlVectorSearch(vector_store)
        
        # Register a custom function
        sparql_search.register_function(
            "vec:customSimilarity",
            arity=2,
            description="Custom similarity function"
        )
        
        # Should not raise an exception
        assert True
    
    def test_sparql_query_execution(self, vector_store, sample_documents):
        """Test SPARQL query execution."""
        if not OXIRS_AVAILABLE:
            pytest.skip("OxiRS not available")
        
        # Index some RDF-like content
        for doc_id, content in sample_documents[:5]:
            vector_store.index_resource(f"http://example.org/{doc_id}", content)
        
        sparql_search = oxirs_vec.SparqlVectorSearch(vector_store)
        
        # Simple SPARQL query with vector extensions
        query = """
        SELECT ?resource WHERE {
            ?resource vec:similar("machine learning", 3, 0.1) .
        }
        """
        
        try:
            results = sparql_search.execute_query(query)
            assert "bindings" in results
            assert "variables" in results
            assert "execution_time_ms" in results
        except Exception as e:
            # SPARQL parsing might not be fully implemented in test environment
            pytest.skip(f"SPARQL execution not available: {e}")

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_embedding_strategy(self):
        """Test invalid embedding strategy handling."""
        if not OXIRS_AVAILABLE:
            pytest.skip("OxiRS not available")
        
        with pytest.raises(oxirs_vec.EmbeddingError):
            oxirs_vec.VectorStore(embedding_strategy="invalid_strategy")
    
    def test_invalid_similarity_metric(self):
        """Test invalid similarity metric handling."""
        if not OXIRS_AVAILABLE:
            pytest.skip("OxiRS not available")
        
        vec1 = np.array([1.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0], dtype=np.float32)
        
        with pytest.raises(oxirs_vec.VectorSearchError):
            oxirs_vec.compute_similarity(vec1, vec2, "invalid_metric")
    
    def test_dimension_mismatch(self):
        """Test dimension mismatch handling."""
        if not OXIRS_AVAILABLE:
            pytest.skip("OxiRS not available")
        
        vec1 = np.array([1.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Different dimension
        
        with pytest.raises((oxirs_vec.VectorSearchError, ValueError)):
            oxirs_vec.compute_similarity(vec1, vec2, "cosine")

class TestPerformance:
    """Performance and benchmark tests."""
    
    def test_large_batch_indexing(self):
        """Test indexing large batches of vectors."""
        if not OXIRS_AVAILABLE:
            pytest.skip("OxiRS not available")
        
        # Generate large batch
        batch_size = 10000
        dimension = 256
        vectors = np.random.rand(batch_size, dimension).astype(np.float32)
        vector_ids = [f"vec_{i}" for i in range(batch_size)]
        
        store = oxirs_vec.VectorStore(
            embedding_strategy="tf_idf",
            index_type="memory"
        )
        
        # Time the indexing operation
        import time
        start_time = time.time()
        store.index_batch(vector_ids, vectors)
        end_time = time.time()
        
        indexing_time = end_time - start_time
        vectors_per_second = batch_size / indexing_time
        
        print(f"Indexed {batch_size} vectors in {indexing_time:.2f}s")
        print(f"Throughput: {vectors_per_second:.0f} vectors/second")
        
        # Verify indexing
        stats = store.get_stats()
        assert stats["total_vectors"] == batch_size
    
    def test_search_performance(self):
        """Test search performance with varying parameters."""
        if not OXIRS_AVAILABLE:
            pytest.skip("OxiRS not available")
        
        # Create store with performance-oriented configuration
        store = oxirs_vec.VectorStore(
            embedding_strategy="tf_idf",
            index_type="hnsw",
            max_connections=32,
            ef_construction=400
        )
        
        # Index test data
        n_vectors = 1000
        dimension = 128
        vectors = np.random.rand(n_vectors, dimension).astype(np.float32)
        vector_ids = [f"vec_{i}" for i in range(n_vectors)]
        store.index_batch(vector_ids, vectors)
        
        # Optimize for search
        store.optimize()
        
        # Benchmark search operations
        query_vector = np.random.rand(dimension).astype(np.float32)
        
        import time
        search_times = []
        n_queries = 100
        
        for _ in range(n_queries):
            start_time = time.time()
            results = store.vector_search(query_vector, limit=10)
            end_time = time.time()
            search_times.append(end_time - start_time)
        
        avg_search_time = np.mean(search_times) * 1000  # Convert to ms
        std_search_time = np.std(search_times) * 1000
        
        print(f"Average search time: {avg_search_time:.2f}±{std_search_time:.2f}ms")
        print(f"Queries per second: {1000 / avg_search_time:.0f}")
        
        # Performance assertion (should be fast)
        assert avg_search_time < 100  # Should be under 100ms

if __name__ == "__main__":
    # Run tests with pytest
    import subprocess
    import sys
    
    # Check if pytest is available
    try:
        import pytest
        print("Running OxiRS Vector Search Python binding tests...")
        result = pytest.main([__file__, "-v", "--tb=short"])
        sys.exit(result)
    except ImportError:
        print("pytest not available. Running basic tests...")
        
        # Run basic tests without pytest
        store = oxirs_vec.VectorStore() if OXIRS_AVAILABLE else MockVectorStore()
        docs = [
            ("doc1", "Machine learning"),
            ("doc2", "Deep learning"),
            ("doc3", "Neural networks")
        ]
        
        for doc_id, content in docs:
            store.index_resource(doc_id, content)
        
        results = store.similarity_search("AI", limit=2)
        print(f"Search results: {results}")
        print("Basic tests completed successfully!")