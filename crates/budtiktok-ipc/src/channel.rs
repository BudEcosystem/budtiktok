//! IPC channel abstraction

use thiserror::Error;

/// IPC error type
#[derive(Error, Debug)]
pub enum IpcError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Send failed: {0}")]
    SendFailed(String),
    #[error("Receive failed: {0}")]
    ReceiveFailed(String),
    #[error("Channel closed")]
    ChannelClosed,
    #[error("Timeout")]
    Timeout,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// IPC channel trait
pub trait IpcChannel: Send + Sync {
    /// Send a message
    fn send(&self, data: &[u8]) -> Result<(), IpcError>;

    /// Receive a message
    fn recv(&self) -> Result<Vec<u8>, IpcError>;

    /// Close the channel
    fn close(&self) -> Result<(), IpcError>;
}
