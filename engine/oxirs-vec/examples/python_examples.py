#!/usr/bin/env python3
"""
OxiRS Vector Search - Python Examples

This file demonstrates the Python bindings for OxiRS Vector Search,
showcasing integration with NumPy, pandas, and scikit-learn.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Import OxiRS Vector Search Python bindings
# Note: This assumes the module is compiled and available
try:
    import oxirs_vec
    OXIRS_AVAILABLE = True
except ImportError:
    OXIRS_AVAILABLE = False
    print("OxiRS Vector Search Python bindings not available. Install with 'pip install oxirs-vec'")

def basic_vector_operations():
    """Demonstrate basic vector operations using OxiRS bindings."""
    if not OXIRS_AVAILABLE:
        return
    
    print("=== Basic Vector Operations ===")
    
    # Create sample vectors
    vec1 = np.random.rand(128).astype(np.float32)
    vec2 = np.random.rand(128).astype(np.float32)
    
    # Compute similarity using OxiRS
    similarity = oxirs_vec.compute_similarity(vec1, vec2, "cosine")
    print(f"Cosine similarity: {similarity:.4f}")
    
    # Normalize vectors
    normalized_vec1 = oxirs_vec.normalize_vector(vec1)
    print(f"Original vector norm: {np.linalg.norm(vec1):.4f}")
    print(f"Normalized vector norm: {np.linalg.norm(normalized_vec1):.4f}")
    
    # Batch normalization
    batch_vectors = np.random.rand(100, 128).astype(np.float32)
    normalized_batch = oxirs_vec.batch_normalize(batch_vectors)
    print(f"Batch shape: {batch_vectors.shape} -> {normalized_batch.shape}")

def vector_store_demo():
    """Demonstrate vector store functionality."""
    if not OXIRS_AVAILABLE:
        return
    
    print("\n=== Vector Store Demo ===")
    
    # Create vector store with sentence transformer embeddings
    store = oxirs_vec.VectorStore(
        embedding_strategy="sentence_transformer",
        index_type="hnsw",
        max_connections=16,
        ef_construction=200
    )
    
    # Index some documents
    documents = [
        ("doc1", "Artificial intelligence and machine learning"),
        ("doc2", "Deep learning neural networks"),
        ("doc3", "Natural language processing"),
        ("doc4", "Computer vision and image recognition"),
        ("doc5", "Reinforcement learning algorithms"),
        ("doc6", "Data science and analytics"),
        ("doc7", "Big data processing frameworks"),
        ("doc8", "Cloud computing infrastructure"),
        ("doc9", "Cybersecurity and encryption"),
        ("doc10", "Quantum computing research")
    ]
    
    for doc_id, content in documents:
        metadata = {"category": "tech", "length": len(content)}
        store.index_resource(doc_id, content, metadata)
    
    print(f"Indexed {len(documents)} documents")
    
    # Search for similar content
    query = "machine learning and AI"
    results = store.similarity_search(query, limit=5, threshold=0.3)
    
    print(f"\nSearch results for '{query}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['id']} (score: {result['score']:.4f})")
        print(f"   Content: {result.get('content', 'N/A')}")
        print(f"   Metadata: {result['metadata']}")
    
    # Get store statistics
    stats = store.get_stats()
    print(f"\nStore statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return store

def numpy_integration_demo():
    """Demonstrate NumPy integration with vector operations."""
    if not OXIRS_AVAILABLE:
        return
    
    print("\n=== NumPy Integration Demo ===")
    
    # Create vector store
    store = oxirs_vec.VectorStore(
        embedding_strategy="tf_idf",
        index_type="memory"
    )
    
    # Generate synthetic vectors using NumPy
    n_vectors = 1000
    dimension = 256
    vectors = np.random.rand(n_vectors, dimension).astype(np.float32)
    
    # Normalize vectors
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Index vectors with metadata
    vector_ids = [f"vec_{i:04d}" for i in range(n_vectors)]
    metadata_list = [{"cluster": i % 10, "index": i} for i in range(n_vectors)]
    
    store.index_batch(vector_ids, vectors, metadata_list)
    print(f"Indexed {n_vectors} vectors with dimension {dimension}")
    
    # Perform vector search
    query_vector = np.random.rand(dimension).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)
    
    results = store.vector_search(query_vector, limit=10, metric="cosine")
    print(f"\nTop 10 similar vectors:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['id']} (score: {result['score']:.4f})")
    
    return vectors, store

def analytics_demo():
    """Demonstrate vector analytics capabilities."""
    if not OXIRS_AVAILABLE:
        return
    
    print("\n=== Analytics Demo ===")
    
    # Create analytics engine
    analytics = oxirs_vec.VectorAnalytics()
    
    # Generate sample data with clusters
    X, y = make_classification(
        n_samples=1000,
        n_features=128,
        n_informative=50,
        n_redundant=10,
        n_clusters_per_class=2,
        random_state=42
    )
    
    X = X.astype(np.float32)
    labels = [f"class_{label}" for label in y]
    
    # Analyze vector distribution
    analysis = analytics.analyze_vectors(X, labels)
    print("Vector analysis results:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # Get optimization recommendations
    recommendations = analytics.get_recommendations()
    print(f"\nOptimization recommendations ({len(recommendations)} items):")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['type']} (Priority: {rec['priority']})")
        print(f"   {rec['description']}")
        print(f"   Expected improvement: {rec['expected_improvement']:.2f}%")

def sparql_integration_demo():
    """Demonstrate SPARQL integration."""
    if not OXIRS_AVAILABLE:
        return
    
    print("\n=== SPARQL Integration Demo ===")
    
    # Create vector store
    store = oxirs_vec.VectorStore(
        embedding_strategy="sentence_transformer",
        index_type="hnsw"
    )
    
    # Index some RDF-like content
    rdf_documents = [
        ("http://example.org/person/alice", "Alice is a computer scientist specializing in AI"),
        ("http://example.org/person/bob", "Bob works in machine learning research"),
        ("http://example.org/paper/paper1", "Neural networks for natural language processing"),
        ("http://example.org/paper/paper2", "Deep learning applications in computer vision"),
        ("http://example.org/concept/ai", "Artificial intelligence and automated reasoning"),
    ]
    
    for uri, description in rdf_documents:
        store.index_resource(uri, description, {"type": "rdf_resource"})
    
    # Create SPARQL vector search interface
    sparql_search = oxirs_vec.SparqlVectorSearch(store)
    
    # Register custom vector function
    sparql_search.register_function(
        "vec:semanticSimilarity",
        arity=3,
        description="Compute semantic similarity between two resources"
    )
    
    # Example SPARQL query with vector extensions
    query = """
    SELECT ?resource ?similarity WHERE {
        ?resource vec:similar("artificial intelligence research", 5, 0.3) .
        BIND(vec:similarity(?resource, "http://example.org/concept/ai") AS ?similarity)
    }
    ORDER BY DESC(?similarity)
    """
    
    try:
        results = sparql_search.execute_query(query)
        print("SPARQL query results:")
        print(f"Variables: {results['variables']}")
        print(f"Execution time: {results['execution_time_ms']} ms")
        print("Bindings:")
        for binding in results['bindings']:
            print(f"  {binding}")
    except Exception as e:
        print(f"SPARQL query execution failed: {e}")

def performance_benchmark():
    """Benchmark vector operations performance."""
    if not OXIRS_AVAILABLE:
        return
    
    print("\n=== Performance Benchmark ===")
    
    import time
    
    # Test different index types
    index_types = ["memory", "hnsw"]
    dimensions = [128, 256, 512]
    
    results = []
    
    for index_type in index_types:
        for dim in dimensions:
            print(f"\nTesting {index_type} index with {dim} dimensions...")
            
            # Create store
            store = oxirs_vec.VectorStore(
                embedding_strategy="tf_idf",
                index_type=index_type
            )
            
            # Generate test data
            n_vectors = 1000
            vectors = np.random.rand(n_vectors, dim).astype(np.float32)
            vector_ids = [f"vec_{i}" for i in range(n_vectors)]
            
            # Benchmark indexing
            start_time = time.time()
            store.index_batch(vector_ids, vectors)
            index_time = time.time() - start_time
            
            # Optimize index
            start_time = time.time()
            store.optimize()
            optimize_time = time.time() - start_time
            
            # Benchmark search
            query_vector = np.random.rand(dim).astype(np.float32)
            search_times = []
            
            for _ in range(100):
                start_time = time.time()
                store.vector_search(query_vector, limit=10)
                search_times.append(time.time() - start_time)
            
            avg_search_time = np.mean(search_times) * 1000  # Convert to ms
            
            result = {
                'index_type': index_type,
                'dimension': dim,
                'index_time_s': index_time,
                'optimize_time_s': optimize_time,
                'avg_search_time_ms': avg_search_time,
                'search_std_ms': np.std(search_times) * 1000
            }
            results.append(result)
            
            print(f"  Indexing: {index_time:.3f}s")
            print(f"  Optimization: {optimize_time:.3f}s")
            print(f"  Search: {avg_search_time:.3f}±{result['search_std_ms']:.3f}ms")
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    print("\nPerformance Summary:")
    print(df.to_string(index=False))
    
    return df

def visualization_demo(vectors=None):
    """Create visualizations of vector data."""
    if not OXIRS_AVAILABLE or vectors is None:
        return
    
    print("\n=== Visualization Demo ===")
    
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        # Reduce dimensionality for visualization
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors[:500])  # Use subset for speed
        
        # Create visualization
        plt.figure(figsize=(12, 5))
        
        # PCA plot
        plt.subplot(1, 2, 1)
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6)
        plt.title('Vector Distribution (PCA)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        # Similarity heatmap (sample)
        sample_vectors = vectors[:20]
        similarity_matrix = cosine_similarity(sample_vectors)
        
        plt.subplot(1, 2, 2)
        sns.heatmap(similarity_matrix, annot=False, cmap='viridis')
        plt.title('Cosine Similarity Heatmap')
        
        plt.tight_layout()
        plt.savefig('/tmp/oxirs_vector_analysis.png', dpi=150, bbox_inches='tight')
        print("Visualization saved to /tmp/oxirs_vector_analysis.png")
        
    except ImportError:
        print("Visualization requires scikit-learn and matplotlib")

def main():
    """Run all demo functions."""
    print("OxiRS Vector Search - Python Integration Demo")
    print("=" * 50)
    
    if not OXIRS_AVAILABLE:
        print("Python bindings not available. Please compile with --features python")
        return
    
    # Run demos
    basic_vector_operations()
    store = vector_store_demo()
    vectors, _ = numpy_integration_demo()
    analytics_demo()
    sparql_integration_demo()
    
    # Performance benchmark
    perf_results = performance_benchmark()
    
    # Visualization
    if vectors is not None:
        visualization_demo(vectors)
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    
    # Save example data
    if store is not None:
        store.save("/tmp/oxirs_demo_store.bin")
        print("Demo store saved to /tmp/oxirs_demo_store.bin")

if __name__ == "__main__":
    main()