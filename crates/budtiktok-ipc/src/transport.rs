//! Transport layer abstraction

/// Transport type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportType {
    SharedMemory,
    UnixSocket,
    Tcp,
}

/// Transport configuration
#[derive(Debug, Clone)]
pub struct TransportConfig {
    pub transport_type: TransportType,
    pub address: String,
    pub timeout_ms: u64,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            transport_type: TransportType::SharedMemory,
            address: "budtiktok-ipc".to_string(),
            timeout_ms: 5000,
        }
    }
}
