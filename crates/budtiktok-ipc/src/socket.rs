//! Unix socket IPC implementation

use crate::channel::{IpcChannel, IpcError};

/// Unix socket channel
pub struct UnixSocketChannel {
    path: String,
}

impl UnixSocketChannel {
    /// Create a new Unix socket server
    pub fn bind(path: &str) -> Result<Self, IpcError> {
        Ok(Self {
            path: path.to_string(),
        })
    }

    /// Connect to a Unix socket server
    pub fn connect(path: &str) -> Result<Self, IpcError> {
        Ok(Self {
            path: path.to_string(),
        })
    }
}

impl IpcChannel for UnixSocketChannel {
    fn send(&self, _data: &[u8]) -> Result<(), IpcError> {
        // TODO: Implement Unix socket send
        Err(IpcError::SendFailed("Not implemented".into()))
    }

    fn recv(&self) -> Result<Vec<u8>, IpcError> {
        // TODO: Implement Unix socket receive
        Err(IpcError::ReceiveFailed("Not implemented".into()))
    }

    fn close(&self) -> Result<(), IpcError> {
        Ok(())
    }
}
