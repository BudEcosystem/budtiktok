//! BudTikTok IPC - High-performance inter-process communication
//!
//! This crate provides zero-copy IPC mechanisms for distributed tokenization:
//! - Shared memory with lock-free ring buffers
//! - Unix domain sockets
//! - TCP transport for cross-machine communication
//!
//! # Performance
//!
//! - Shared memory: ~500ns per message
//! - Unix socket: ~2μs per message
//! - TCP (localhost): ~10μs per message
//!
//! # Example
//!
//! ```rust,ignore
//! use budtiktok_ipc::{SharedMemoryChannel, IpcMessage};
//!
//! // Server side
//! let server = SharedMemoryChannel::create("budtiktok-ipc", 64 * 1024 * 1024)?;
//!
//! // Client side
//! let client = SharedMemoryChannel::open("budtiktok-ipc")?;
//! client.send(IpcMessage::TokenizeRequest { text: "Hello" })?;
//! ```

#![deny(clippy::all)]
#![warn(clippy::pedantic)]

pub mod channel;
pub mod message;
pub mod serialization;
pub mod shared_memory;
pub mod socket;
pub mod transport;

pub use channel::{IpcChannel, IpcError};
pub use message::{
    IpcMessage, PreTokenizedRequest, PreTokenizedRequestBuilder,
    EmbeddingResponse, BatchMetadata, ResponseMetadata,
};
pub use serialization::{
    Serializer, SerializationFormat, SerializationError,
    MessageHeader, BatchSerializer, PreTokenizedView, serialize_raw,
    SchemaVersion, Versioned, VersionedSerializer,
};
pub use shared_memory::SharedMemoryChannel;
