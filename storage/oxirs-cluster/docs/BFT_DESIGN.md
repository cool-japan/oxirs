# Byzantine Fault Tolerance Design for OxiRS Cluster

## Overview

This document describes the Byzantine Fault Tolerance (BFT) implementation for OxiRS cluster, providing protection against malicious nodes in untrusted environments.

## Architecture

### Core Components

1. **BFT Consensus Module** (`bft.rs`)
   - Implements PBFT (Practical Byzantine Fault Tolerance) consensus
   - Supports f Byzantine nodes in a cluster of n ≥ 3f + 1 nodes
   - Three-phase commit protocol: Pre-prepare, Prepare, Commit
   - View change mechanism for leader failures

2. **BFT Network Layer** (`bft_network.rs`)
   - Authenticated message passing with cryptographic signatures
   - Message ordering and duplicate detection
   - Byzantine node detection and isolation
   - Secure channels with Ed25519 signatures

3. **BFT Consensus Integration** (`bft_consensus.rs`)
   - Bridge between BFT and existing cluster infrastructure
   - Implements ConsensusManager trait for compatibility
   - Manages node registration and key distribution

## Key Features

### 1. Byzantine Fault Tolerance
- Tolerates up to f = (n-1)/3 Byzantine nodes
- Requires n ≥ 3f + 1 nodes for consensus
- Automatic Byzantine node detection and isolation

### 2. Cryptographic Security
- Ed25519 digital signatures for message authentication
- SHA-256 message digests for integrity
- Timestamp-based freshness checks (5-minute window)

### 3. Consensus Protocol
- **Pre-prepare Phase**: Primary broadcasts request
- **Prepare Phase**: Nodes verify and prepare request  
- **Commit Phase**: Nodes commit after 2f prepares
- **Reply Phase**: Send result to client after 2f+1 commits

### 4. View Change Protocol
- Automatic view change on primary timeout
- Preserves prepared requests across views
- Ensures liveness despite Byzantine primary

### 5. Message Types
```rust
pub enum BftMessage {
    Request { client_id, operation, timestamp, signature },
    PrePrepare { view, sequence, digest, request, primary_signature },
    Prepare { view, sequence, digest, node_id, signature },
    Commit { view, sequence, digest, node_id, signature },
    Reply { view, timestamp, client_id, node_id, result, signature },
    ViewChange { new_view, node_id, prepared_messages, signature },
    NewView { view, view_changes, pre_prepares, primary_signature },
    Checkpoint { sequence, digest, node_id, signature },
}
```

## Configuration

### BFT Configuration
```rust
pub struct BftConfig {
    pub min_nodes: usize,        // Minimum nodes for consensus (3f+1)
    pub max_faulty: usize,       // Maximum Byzantine nodes (f)
    pub view_timeout: Duration,  // Timeout for view changes
    pub auth_timeout: Duration,  // Message authentication timeout
    pub enable_signatures: bool, // Enable cryptographic signatures
    pub enable_ordering: bool,   // Enable message ordering checks
}
```

### Enabling BFT
```rust
let config = NodeConfig::new(node_id, address)
    .with_bft(true);  // Enable Byzantine fault tolerance

let node = ClusterNode::new(config).await?;
```

## Byzantine Node Detection

The system tracks invalid behavior and isolates Byzantine nodes:

1. **Invalid Signatures**: Nodes sending incorrectly signed messages
2. **Protocol Violations**: Nodes violating consensus protocol rules
3. **Timing Attacks**: Nodes sending excessively old messages
4. **Equivocation**: Nodes sending conflicting messages

### Detection Thresholds
- 5 invalid messages: Node marked as suspected
- 10 invalid messages: Node marked as Byzantine and isolated

## Security Considerations

### 1. Key Management
- Each node generates an Ed25519 keypair
- Public keys must be distributed securely
- Private keys must be protected

### 2. Network Security
- All messages are cryptographically signed
- Message freshness enforced with timestamps
- Duplicate messages detected and rejected

### 3. Storage Security
- Cryptographic proofs for stored data (planned)
- Merkle trees for data integrity (planned)
- Tamper detection mechanisms (planned)

## Performance Characteristics

### Message Complexity
- Normal operation: O(n²) messages per request
- View change: O(n³) messages worst case

### Latency
- 3 network round trips in normal case
- Additional latency for view changes

### Throughput
- Limited by message processing overhead
- Scales inversely with cluster size

## Usage Example

```rust
use oxirs_cluster::{ClusterNode, NodeConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Create BFT-enabled cluster node
    let config = NodeConfig::new(1, "127.0.0.1:8080".parse()?)
        .with_bft(true)
        .add_peer(2)
        .add_peer(3)
        .add_peer(4);
    
    let mut node = ClusterNode::new(config).await?;
    
    // Register peer public keys
    node.register_peer(2, peer2_public_key, "127.0.0.1:8081").await?;
    node.register_peer(3, peer3_public_key, "127.0.0.1:8082").await?;
    node.register_peer(4, peer4_public_key, "127.0.0.1:8083").await?;
    
    // Start the node
    node.start().await?;
    
    // Operations are now Byzantine fault-tolerant
    node.insert_triple(
        "<http://example.org/subject>",
        "<http://example.org/predicate>",
        "\"object\""
    ).await?;
    
    Ok(())
}
```

## Future Enhancements

1. **Optimizations**
   - Speculative execution
   - Batching for improved throughput
   - Adaptive timeout mechanisms

2. **Advanced Features**
   - Proactive recovery
   - State transfer optimization
   - Dynamic membership changes

3. **Integration**
   - Blockchain anchoring for audit logs
   - HSM support for key management
   - Zero-knowledge proofs for privacy

## Testing

The BFT implementation includes comprehensive tests:
- Unit tests for consensus logic
- Integration tests for network layer
- Byzantine fault injection tests
- Performance benchmarks

Run tests with:
```bash
cargo test -p oxirs-cluster --features bft
```

## References

1. Castro, M., & Liskov, B. (1999). Practical Byzantine fault tolerance
2. Lamport, L., Shostak, R., & Pease, M. (1982). The Byzantine Generals Problem
3. Ed25519: High-speed high-security signatures