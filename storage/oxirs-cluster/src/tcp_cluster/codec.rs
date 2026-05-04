//! Length-prefixed message framing for the TCP cluster harness.
//!
//! Wire format: `[u32 BE length][JSON body]`
//!
//! The same framing pattern used by the existing cluster transport in
//! `server/oxirs-fuseki/src/clustering/node.rs`, so operators already
//! familiar with the production code can reason about the format without
//! additional documentation.

use std::io;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

// ─────────────────────────────────────────────────────────────────────────────
// Message vocabulary
// ─────────────────────────────────────────────────────────────────────────────

/// Messages exchanged between TCP cluster nodes.
///
/// Each variant maps to a distinct cluster primitive:
/// - **Gossip** — epidemic-protocol key-value propagation
/// - **Ping** / **Pong** — heartbeat pair for liveness checking
/// - **Replicate** / **ReplicateAck** — log replication stubs that prove
///   the replication path works over real sockets without full Raft state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ClusterMessage {
    /// Gossip: propagate a key-value state entry (last-write-wins).
    Gossip {
        /// Node ID of the sender.
        sender_id: String,
        /// Key being gossiped.
        key: String,
        /// Opaque u64 value.
        value: u64,
        /// Monotonically-increasing version; higher wins on conflict.
        version: u64,
    },
    /// Heartbeat ping — the receiver should reply with a matching `Pong`.
    Ping {
        /// Node ID of the sender.
        sender_id: String,
        /// Sequence number echoed back in the matching `Pong`.
        seq: u64,
    },
    /// Heartbeat pong — reply to `Ping`.
    Pong {
        /// Node ID of the replying node.
        sender_id: String,
        /// Echoed sequence number from the matching `Ping`.
        seq: u64,
    },
    /// Replication stub: leader proposes a log entry.
    Replicate {
        /// Node ID of the leader.
        leader_id: String,
        /// Log index (1-based).
        index: u64,
        /// Raft term of this entry.
        term: u64,
        /// CRC32 checksum of the (simulated) entry payload.
        checksum: u64,
    },
    /// Acknowledgment from a follower for a `Replicate` RPC.
    ReplicateAck {
        /// Node ID of the acknowledging follower.
        follower_id: String,
        /// Log index being acknowledged.
        index: u64,
        /// `true` if the entry was accepted, `false` if rejected.
        success: bool,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Codec
// ─────────────────────────────────────────────────────────────────────────────

/// Zero-size type providing stateless read/write helpers for [`ClusterMessage`].
///
/// Wire format: `[u32 big-endian byte count][serde_json UTF-8 body]`
pub struct MessageCodec;

impl MessageCodec {
    /// Write one [`ClusterMessage`] to `writer`.
    ///
    /// The bytes on the wire are:
    /// 1. 4-byte big-endian unsigned length of the JSON body
    /// 2. the JSON body (UTF-8)
    ///
    /// # Errors
    ///
    /// Returns `Err` if serialization fails or the underlying write fails.
    pub async fn write<W>(writer: &mut W, msg: &ClusterMessage) -> io::Result<()>
    where
        W: AsyncWriteExt + Unpin,
    {
        let body =
            serde_json::to_vec(msg).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let len = u32::try_from(body.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "message body exceeds u32::MAX bytes",
            )
        })?;

        writer.write_all(&len.to_be_bytes()).await?;
        writer.write_all(&body).await?;
        writer.flush().await?;
        Ok(())
    }

    /// Read one [`ClusterMessage`] from `reader`.
    ///
    /// Reads the 4-byte length prefix, allocates a buffer of that size, reads
    /// the body, then deserialises.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the read fails or the body is not valid JSON.
    pub async fn read<R>(reader: &mut R) -> io::Result<ClusterMessage>
    where
        R: AsyncReadExt + Unpin,
    {
        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf) as usize;

        let mut body = vec![0u8; len];
        reader.read_exact(&mut body).await?;

        serde_json::from_slice(&body).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::duplex;

    #[tokio::test]
    async fn test_roundtrip_gossip() {
        let msg = ClusterMessage::Gossip {
            sender_id: "node-1".to_owned(),
            key: "alpha".to_owned(),
            value: 99,
            version: 7,
        };
        let (mut client, mut server) = duplex(1024);
        MessageCodec::write(&mut client, &msg).await.expect("write");
        let received = MessageCodec::read(&mut server).await.expect("read");
        assert_eq!(msg, received);
    }

    #[tokio::test]
    async fn test_roundtrip_ping() {
        let msg = ClusterMessage::Ping {
            sender_id: "node-2".to_owned(),
            seq: 42,
        };
        let (mut client, mut server) = duplex(1024);
        MessageCodec::write(&mut client, &msg).await.expect("write");
        let received = MessageCodec::read(&mut server).await.expect("read");
        assert_eq!(msg, received);
    }

    #[tokio::test]
    async fn test_roundtrip_replicate() {
        let msg = ClusterMessage::Replicate {
            leader_id: "leader".to_owned(),
            index: 100,
            term: 3,
            checksum: 0xDEAD_BEEF,
        };
        let (mut client, mut server) = duplex(1024);
        MessageCodec::write(&mut client, &msg).await.expect("write");
        let received = MessageCodec::read(&mut server).await.expect("read");
        assert_eq!(msg, received);
    }

    #[tokio::test]
    async fn test_multiple_messages_in_sequence() {
        let msgs = vec![
            ClusterMessage::Ping {
                sender_id: "a".to_owned(),
                seq: 1,
            },
            ClusterMessage::Pong {
                sender_id: "b".to_owned(),
                seq: 1,
            },
            ClusterMessage::ReplicateAck {
                follower_id: "c".to_owned(),
                index: 5,
                success: true,
            },
        ];
        let (mut client, mut server) = duplex(4096);
        for msg in &msgs {
            MessageCodec::write(&mut client, msg).await.expect("write");
        }
        // drop client write-half to signal EOF, but read from the duplex side
        for expected in &msgs {
            let received = MessageCodec::read(&mut server).await.expect("read");
            assert_eq!(expected, &received);
        }
    }
}
