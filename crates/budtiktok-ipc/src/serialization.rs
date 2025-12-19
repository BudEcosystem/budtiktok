//! Efficient serialization formats for IPC messages
//!
//! This module provides multiple serialization formats optimized for different use cases:
//!
//! - **Bincode**: Fast binary serialization, good balance of speed and size
//! - **JSON**: Human-readable, good for debugging and interop
//! - **Raw**: Zero-copy for performance-critical paths (shared memory)
//!
//! # Performance Characteristics
//!
//! | Format  | Serialize | Deserialize | Size   | Zero-copy |
//! |---------|-----------|-------------|--------|-----------|
//! | Bincode | ~100ns    | ~150ns      | Small  | No        |
//! | JSON    | ~1μs      | ~2μs        | Large  | No        |
//! | Raw     | ~10ns     | ~10ns       | Exact  | Yes       |
//!
//! # Usage
//!
//! ```rust,ignore
//! use budtiktok_ipc::{PreTokenizedRequest, Serializer, SerializationFormat};
//!
//! let request = PreTokenizedRequest::new(1);
//! let serializer = Serializer::new(SerializationFormat::Bincode);
//!
//! let bytes = serializer.serialize(&request)?;
//! let decoded: PreTokenizedRequest = serializer.deserialize(&bytes)?;
//! ```

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use thiserror::Error;

/// Serialization error types
#[derive(Debug, Error)]
pub enum SerializationError {
    #[error("Bincode error: {0}")]
    Bincode(#[from] bincode::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error("Buffer too small: need {needed}, have {available}")]
    BufferTooSmall { needed: usize, available: usize },

    #[error("Invalid header")]
    InvalidHeader,

    #[error("Version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u8, actual: u8 },

    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u8),

    #[error("Migration error: {0}")]
    MigrationError(String),
}

/// Serialization format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SerializationFormat {
    /// Bincode binary format (default)
    #[default]
    Bincode,
    /// JSON text format
    Json,
    /// Raw format with minimal framing
    Raw,
}

impl SerializationFormat {
    /// Get the format identifier byte
    pub fn id(&self) -> u8 {
        match self {
            SerializationFormat::Bincode => 0x01,
            SerializationFormat::Json => 0x02,
            SerializationFormat::Raw => 0x03,
        }
    }

    /// Parse format from identifier byte
    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0x01 => Some(SerializationFormat::Bincode),
            0x02 => Some(SerializationFormat::Json),
            0x03 => Some(SerializationFormat::Raw),
            _ => None,
        }
    }
}

/// Message header for framed messages
///
/// Layout: [magic(2)][version(1)][format(1)][length(4)]
#[derive(Debug, Clone, Copy)]
pub struct MessageHeader {
    /// Magic bytes for validation
    pub magic: [u8; 2],
    /// Protocol version
    pub version: u8,
    /// Serialization format
    pub format: SerializationFormat,
    /// Payload length in bytes
    pub length: u32,
}

impl MessageHeader {
    /// Magic bytes identifying BudTikTok IPC messages ("BT")
    pub const MAGIC: [u8; 2] = [0xBD, 0x54];

    /// Current protocol version
    pub const VERSION: u8 = 1;

    /// Minimum supported protocol version for backward compatibility
    pub const MIN_SUPPORTED_VERSION: u8 = 1;

    /// Header size in bytes
    pub const SIZE: usize = 8;

    /// Create a new header
    pub fn new(format: SerializationFormat, length: u32) -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            format,
            length,
        }
    }

    /// Create a header with a specific version (for testing/migration)
    pub fn with_version(format: SerializationFormat, length: u32, version: u8) -> Self {
        Self {
            magic: Self::MAGIC,
            version,
            format,
            length,
        }
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0] = self.magic[0];
        buf[1] = self.magic[1];
        buf[2] = self.version;
        buf[3] = self.format.id();
        buf[4..8].copy_from_slice(&self.length.to_le_bytes());
        buf
    }

    /// Parse header from bytes with backward compatibility
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SerializationError> {
        Self::from_bytes_with_options(bytes, true)
    }

    /// Parse header from bytes with version strictness option
    pub fn from_bytes_with_options(bytes: &[u8], allow_older_versions: bool) -> Result<Self, SerializationError> {
        if bytes.len() < Self::SIZE {
            return Err(SerializationError::BufferTooSmall {
                needed: Self::SIZE,
                available: bytes.len(),
            });
        }

        let magic = [bytes[0], bytes[1]];
        if magic != Self::MAGIC {
            return Err(SerializationError::InvalidHeader);
        }

        let version = bytes[2];

        // Version compatibility check
        if version > Self::VERSION {
            // Future version - not supported
            return Err(SerializationError::UnsupportedVersion(version));
        }

        if version < Self::MIN_SUPPORTED_VERSION {
            // Too old - not supported
            return Err(SerializationError::UnsupportedVersion(version));
        }

        if !allow_older_versions && version != Self::VERSION {
            return Err(SerializationError::VersionMismatch {
                expected: Self::VERSION,
                actual: version,
            });
        }

        let format = SerializationFormat::from_id(bytes[3])
            .ok_or_else(|| SerializationError::InvalidFormat(format!("Unknown format: {}", bytes[3])))?;

        let length = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

        Ok(Self {
            magic,
            version,
            format,
            length,
        })
    }

    /// Check if this message requires migration
    pub fn needs_migration(&self) -> bool {
        self.version < Self::VERSION
    }

    /// Check if this version is supported
    pub fn is_supported(&self) -> bool {
        self.version >= Self::MIN_SUPPORTED_VERSION && self.version <= Self::VERSION
    }
}

// =============================================================================
// Schema Versioning (7.1.3)
// =============================================================================

/// Schema version information for migration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchemaVersion {
    pub major: u8,
    pub minor: u8,
}

impl SchemaVersion {
    pub const V1_0: Self = Self { major: 1, minor: 0 };
    pub const CURRENT: Self = Self::V1_0;

    pub fn from_version_byte(version: u8) -> Self {
        // Version byte encoding: upper 4 bits = major, lower 4 bits = minor
        Self {
            major: version >> 4,
            minor: version & 0x0F,
        }
    }

    pub fn to_version_byte(&self) -> u8 {
        (self.major << 4) | (self.minor & 0x0F)
    }

    pub fn is_compatible_with(&self, other: &Self) -> bool {
        // Major version must match, minor can be equal or lower
        self.major == other.major && self.minor <= other.minor
    }
}

/// Trait for types that support schema versioning and migration
pub trait Versioned: Sized {
    /// Get the schema version this type was serialized with
    fn schema_version() -> SchemaVersion;

    /// Migrate from an older version if needed
    fn migrate(data: &[u8], from_version: SchemaVersion) -> Result<Self, SerializationError>
    where
        Self: for<'de> Deserialize<'de>,
    {
        if from_version == Self::schema_version() {
            // No migration needed
            bincode::deserialize(data).map_err(SerializationError::from)
        } else {
            // Version mismatch - subclasses should implement specific migration
            Err(SerializationError::MigrationError(format!(
                "No migration path from {:?} to {:?}",
                from_version,
                Self::schema_version()
            )))
        }
    }
}

/// Versioned serializer with automatic migration support
#[derive(Debug, Clone)]
pub struct VersionedSerializer {
    format: SerializationFormat,
    strict_version: bool,
}

impl VersionedSerializer {
    /// Create a new versioned serializer
    pub fn new(format: SerializationFormat) -> Self {
        Self {
            format,
            strict_version: false,
        }
    }

    /// Create a strict versioned serializer that rejects older versions
    pub fn strict(format: SerializationFormat) -> Self {
        Self {
            format,
            strict_version: true,
        }
    }

    /// Serialize with version prefix
    pub fn serialize<T: Serialize>(&self, value: &T) -> Result<Vec<u8>, SerializationError> {
        let payload = match self.format {
            SerializationFormat::Bincode => bincode::serialize(value)?,
            SerializationFormat::Json => serde_json::to_vec(value)?,
            SerializationFormat::Raw => bincode::serialize(value)?,
        };

        let header = MessageHeader::new(self.format, payload.len() as u32);
        let mut result = Vec::with_capacity(MessageHeader::SIZE + payload.len());
        result.extend_from_slice(&header.to_bytes());
        result.extend_from_slice(&payload);
        Ok(result)
    }

    /// Deserialize with version checking and optional migration
    pub fn deserialize<T>(&self, bytes: &[u8]) -> Result<T, SerializationError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let header = MessageHeader::from_bytes_with_options(bytes, !self.strict_version)?;
        let payload = &bytes[MessageHeader::SIZE..MessageHeader::SIZE + header.length as usize];

        match header.format {
            SerializationFormat::Bincode => Ok(bincode::deserialize(payload)?),
            SerializationFormat::Json => Ok(serde_json::from_slice(payload)?),
            SerializationFormat::Raw => Ok(bincode::deserialize(payload)?),
        }
    }

    /// Deserialize with automatic migration for versioned types
    pub fn deserialize_versioned<T>(&self, bytes: &[u8]) -> Result<T, SerializationError>
    where
        T: Versioned + for<'de> Deserialize<'de>,
    {
        let header = MessageHeader::from_bytes_with_options(bytes, !self.strict_version)?;
        let payload = &bytes[MessageHeader::SIZE..MessageHeader::SIZE + header.length as usize];

        if header.needs_migration() {
            let from_version = SchemaVersion::from_version_byte(header.version);
            T::migrate(payload, from_version)
        } else {
            match header.format {
                SerializationFormat::Bincode => Ok(bincode::deserialize(payload)?),
                SerializationFormat::Json => Ok(serde_json::from_slice(payload)?),
                SerializationFormat::Raw => Ok(bincode::deserialize(payload)?),
            }
        }
    }

    /// Get version info from serialized bytes without deserializing
    pub fn get_version(bytes: &[u8]) -> Result<u8, SerializationError> {
        if bytes.len() < MessageHeader::SIZE {
            return Err(SerializationError::BufferTooSmall {
                needed: MessageHeader::SIZE,
                available: bytes.len(),
            });
        }
        Ok(bytes[2])
    }

    /// Check if serialized data is compatible with current version
    pub fn is_compatible(bytes: &[u8]) -> Result<bool, SerializationError> {
        let version = Self::get_version(bytes)?;
        Ok(version >= MessageHeader::MIN_SUPPORTED_VERSION && version <= MessageHeader::VERSION)
    }
}

/// Serializer for IPC messages
#[derive(Debug, Clone)]
pub struct Serializer {
    format: SerializationFormat,
    include_header: bool,
}

impl Serializer {
    /// Create a new serializer with the specified format
    pub fn new(format: SerializationFormat) -> Self {
        Self {
            format,
            include_header: true,
        }
    }

    /// Create a serializer without message headers (for raw use)
    pub fn without_header(format: SerializationFormat) -> Self {
        Self {
            format,
            include_header: false,
        }
    }

    /// Get the serialization format
    pub fn format(&self) -> SerializationFormat {
        self.format
    }

    /// Serialize a value to bytes
    pub fn serialize<T: Serialize>(&self, value: &T) -> Result<Vec<u8>, SerializationError> {
        let payload = match self.format {
            SerializationFormat::Bincode => bincode::serialize(value)?,
            SerializationFormat::Json => serde_json::to_vec(value)?,
            SerializationFormat::Raw => bincode::serialize(value)?, // Fallback to bincode
        };

        if self.include_header {
            let header = MessageHeader::new(self.format, payload.len() as u32);
            let mut result = Vec::with_capacity(MessageHeader::SIZE + payload.len());
            result.extend_from_slice(&header.to_bytes());
            result.extend_from_slice(&payload);
            Ok(result)
        } else {
            Ok(payload)
        }
    }

    /// Serialize a value into a pre-allocated buffer
    pub fn serialize_into<T: Serialize, W: Write>(
        &self,
        value: &T,
        writer: &mut W,
    ) -> Result<usize, SerializationError> {
        let payload = match self.format {
            SerializationFormat::Bincode => bincode::serialize(value)?,
            SerializationFormat::Json => serde_json::to_vec(value)?,
            SerializationFormat::Raw => bincode::serialize(value)?,
        };

        let mut written = 0;
        if self.include_header {
            let header = MessageHeader::new(self.format, payload.len() as u32);
            writer.write_all(&header.to_bytes())?;
            written += MessageHeader::SIZE;
        }
        writer.write_all(&payload)?;
        written += payload.len();

        Ok(written)
    }

    /// Deserialize a value from bytes
    pub fn deserialize<T: for<'de> Deserialize<'de>>(
        &self,
        bytes: &[u8],
    ) -> Result<T, SerializationError> {
        let payload = if self.include_header {
            let header = MessageHeader::from_bytes(bytes)?;
            &bytes[MessageHeader::SIZE..MessageHeader::SIZE + header.length as usize]
        } else {
            bytes
        };

        match self.format {
            SerializationFormat::Bincode => Ok(bincode::deserialize(payload)?),
            SerializationFormat::Json => Ok(serde_json::from_slice(payload)?),
            SerializationFormat::Raw => Ok(bincode::deserialize(payload)?),
        }
    }

    /// Deserialize a value from a reader
    pub fn deserialize_from<T: for<'de> Deserialize<'de>, R: Read>(
        &self,
        reader: &mut R,
    ) -> Result<T, SerializationError> {
        if self.include_header {
            let mut header_bytes = [0u8; MessageHeader::SIZE];
            reader.read_exact(&mut header_bytes)?;
            let header = MessageHeader::from_bytes(&header_bytes)?;

            let mut payload = vec![0u8; header.length as usize];
            reader.read_exact(&mut payload)?;

            match header.format {
                SerializationFormat::Bincode => Ok(bincode::deserialize(&payload)?),
                SerializationFormat::Json => Ok(serde_json::from_slice(&payload)?),
                SerializationFormat::Raw => Ok(bincode::deserialize(&payload)?),
            }
        } else {
            match self.format {
                SerializationFormat::Bincode => Ok(bincode::deserialize_from(reader)?),
                SerializationFormat::Json => Ok(serde_json::from_reader(reader)?),
                SerializationFormat::Raw => Ok(bincode::deserialize_from(reader)?),
            }
        }
    }

    /// Estimate serialized size for a PreTokenizedRequest
    pub fn estimate_size(request: &super::PreTokenizedRequest) -> usize {
        // Bincode overhead is minimal, estimate based on data size
        let data_size = 8  // batch_id
            + request.input_ids.len() * 4  // input_ids
            + request.attention_mask.len()  // attention_mask
            + request.token_type_ids.as_ref().map(|t| t.len()).unwrap_or(0)  // token_type_ids
            + request.sequence_offsets.len() * 4  // sequence_offsets
            + request.original_lengths.len() * 4  // original_lengths
            + 64;  // metadata overhead estimate

        MessageHeader::SIZE + data_size
    }
}

impl Default for Serializer {
    fn default() -> Self {
        Self::new(SerializationFormat::Bincode)
    }
}

/// Batch serializer for multiple messages
pub struct BatchSerializer {
    serializer: Serializer,
    buffer: Vec<u8>,
}

impl BatchSerializer {
    /// Create a new batch serializer
    pub fn new(format: SerializationFormat) -> Self {
        Self {
            serializer: Serializer::new(format),
            buffer: Vec::with_capacity(64 * 1024), // 64KB default
        }
    }

    /// Create with pre-allocated capacity
    pub fn with_capacity(format: SerializationFormat, capacity: usize) -> Self {
        Self {
            serializer: Serializer::new(format),
            buffer: Vec::with_capacity(capacity),
        }
    }

    /// Serialize multiple messages into a batch
    pub fn serialize_batch<T: Serialize>(&mut self, messages: &[T]) -> Result<&[u8], SerializationError> {
        self.buffer.clear();

        // Write batch header: [count(4)]
        let count = messages.len() as u32;
        self.buffer.extend_from_slice(&count.to_le_bytes());

        // Serialize each message
        for msg in messages {
            self.serializer.serialize_into(msg, &mut self.buffer)?;
        }

        Ok(&self.buffer)
    }

    /// Deserialize a batch of messages
    pub fn deserialize_batch<T: for<'de> Deserialize<'de>>(
        &self,
        bytes: &[u8],
    ) -> Result<Vec<T>, SerializationError> {
        if bytes.len() < 4 {
            return Err(SerializationError::BufferTooSmall {
                needed: 4,
                available: bytes.len(),
            });
        }

        let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let mut results = Vec::with_capacity(count);
        let mut offset = 4;

        for _ in 0..count {
            let header = MessageHeader::from_bytes(&bytes[offset..])?;
            let msg_end = offset + MessageHeader::SIZE + header.length as usize;
            let msg: T = self.serializer.deserialize(&bytes[offset..msg_end])?;
            results.push(msg);
            offset = msg_end;
        }

        Ok(results)
    }

    /// Clear the internal buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Get the buffer capacity
    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }
}

/// Zero-copy view into serialized PreTokenizedRequest data
///
/// This provides direct access to the underlying data without deserialization,
/// useful for shared memory scenarios.
pub struct PreTokenizedView<'a> {
    data: &'a [u8],
    header: RawHeader,
}

/// Raw header for zero-copy access
#[derive(Debug, Clone, Copy)]
struct RawHeader {
    batch_id: u64,
    num_sequences: u32,
    total_tokens: u32,
    input_ids_offset: u32,
    attention_mask_offset: u32,
    sequence_offsets_offset: u32,
}

impl<'a> PreTokenizedView<'a> {
    /// Raw format magic
    const MAGIC: u32 = 0x50544B42; // "BTKP"

    /// Create a view from raw bytes (unsafe - caller must ensure valid layout)
    pub fn from_bytes(data: &'a [u8]) -> Result<Self, SerializationError> {
        if data.len() < 32 {
            return Err(SerializationError::BufferTooSmall {
                needed: 32,
                available: data.len(),
            });
        }

        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic != Self::MAGIC {
            return Err(SerializationError::InvalidHeader);
        }

        let header = RawHeader {
            batch_id: u64::from_le_bytes(data[4..12].try_into().unwrap()),
            num_sequences: u32::from_le_bytes(data[12..16].try_into().unwrap()),
            total_tokens: u32::from_le_bytes(data[16..20].try_into().unwrap()),
            input_ids_offset: u32::from_le_bytes(data[20..24].try_into().unwrap()),
            attention_mask_offset: u32::from_le_bytes(data[24..28].try_into().unwrap()),
            sequence_offsets_offset: u32::from_le_bytes(data[28..32].try_into().unwrap()),
        };

        Ok(Self { data, header })
    }

    /// Get the batch ID
    pub fn batch_id(&self) -> u64 {
        self.header.batch_id
    }

    /// Get the number of sequences
    pub fn num_sequences(&self) -> usize {
        self.header.num_sequences as usize
    }

    /// Get the total number of tokens
    pub fn total_tokens(&self) -> usize {
        self.header.total_tokens as usize
    }

    /// Get direct view of input IDs
    ///
    /// # Panics
    /// Panics if the data is not properly aligned for u32 access or if
    /// the buffer bounds are invalid.
    pub fn input_ids(&self) -> &[u32] {
        let start = self.header.input_ids_offset as usize;
        let end = self.header.attention_mask_offset as usize;

        // Bounds check
        assert!(
            start <= self.data.len() && end <= self.data.len() && start <= end,
            "Invalid input_ids bounds: start={}, end={}, data_len={}",
            start,
            end,
            self.data.len()
        );

        let byte_slice = &self.data[start..end];

        // Alignment check: pointer must be aligned to u32 (4 bytes)
        let ptr = byte_slice.as_ptr();
        assert!(
            ptr.align_offset(std::mem::align_of::<u32>()) == 0,
            "input_ids data is not properly aligned for u32 access"
        );

        // Length check: must be a multiple of 4
        assert!(
            byte_slice.len() % 4 == 0,
            "input_ids byte length {} is not a multiple of 4",
            byte_slice.len()
        );

        // SAFETY: We've verified:
        // 1. The slice bounds are valid
        // 2. The pointer is properly aligned for u32
        // 3. The length is a multiple of 4
        unsafe {
            std::slice::from_raw_parts(ptr as *const u32, byte_slice.len() / 4)
        }
    }

    /// Get direct view of attention mask
    pub fn attention_mask(&self) -> &[u8] {
        let start = self.header.attention_mask_offset as usize;
        let end = self.header.sequence_offsets_offset as usize;
        &self.data[start..end]
    }

    /// Get direct view of sequence offsets
    ///
    /// # Panics
    /// Panics if the data is not properly aligned for u32 access or if
    /// the buffer bounds are invalid.
    pub fn sequence_offsets(&self) -> &[u32] {
        let start = self.header.sequence_offsets_offset as usize;
        let num_offsets = (self.header.num_sequences as usize)
            .checked_add(1)
            .expect("sequence offset count overflow");
        let byte_len = num_offsets
            .checked_mul(4)
            .expect("sequence offsets byte length overflow");
        let end = start
            .checked_add(byte_len)
            .expect("sequence offsets end offset overflow");

        // Bounds check
        assert!(
            start <= self.data.len() && end <= self.data.len(),
            "Invalid sequence_offsets bounds: start={}, end={}, data_len={}",
            start,
            end,
            self.data.len()
        );

        let byte_slice = &self.data[start..end];

        // Alignment check
        let ptr = byte_slice.as_ptr();
        assert!(
            ptr.align_offset(std::mem::align_of::<u32>()) == 0,
            "sequence_offsets data is not properly aligned for u32 access"
        );

        // SAFETY: We've verified bounds, alignment, and length
        unsafe { std::slice::from_raw_parts(ptr as *const u32, num_offsets) }
    }
}

/// Serialize a PreTokenizedRequest to raw format for zero-copy access
///
/// # Panics
/// Panics if the buffer size calculation overflows.
pub fn serialize_raw(request: &super::PreTokenizedRequest) -> Vec<u8> {
    let num_sequences = request.num_sequences() as u32;
    let total_tokens = request.total_tokens() as u32;

    // Calculate offsets with overflow checking
    let header_size = 32u32;
    let input_ids_offset = header_size;

    // input_ids_size = total_tokens * 4 (checked)
    let input_ids_size = total_tokens
        .checked_mul(4)
        .expect("input_ids size overflow: total_tokens * 4");

    // attention_mask_offset = input_ids_offset + input_ids_size (checked)
    let attention_mask_offset = input_ids_offset
        .checked_add(input_ids_size)
        .expect("attention_mask offset overflow");

    let attention_mask_size = total_tokens;

    // sequence_offsets_offset = attention_mask_offset + attention_mask_size (checked)
    let sequence_offsets_offset = attention_mask_offset
        .checked_add(attention_mask_size)
        .expect("sequence_offsets offset overflow");

    // sequence_offsets_size = (num_sequences + 1) * 4 (checked)
    let num_offsets = num_sequences
        .checked_add(1)
        .expect("num_offsets overflow");
    let sequence_offsets_size = num_offsets
        .checked_mul(4)
        .expect("sequence_offsets size overflow");

    // total_size (checked)
    let total_size = sequence_offsets_offset
        .checked_add(sequence_offsets_size)
        .expect("total buffer size overflow") as usize;
    let mut buffer = vec![0u8; total_size];

    // Write header
    buffer[0..4].copy_from_slice(&PreTokenizedView::MAGIC.to_le_bytes());
    buffer[4..12].copy_from_slice(&request.batch_id.to_le_bytes());
    buffer[12..16].copy_from_slice(&num_sequences.to_le_bytes());
    buffer[16..20].copy_from_slice(&total_tokens.to_le_bytes());
    buffer[20..24].copy_from_slice(&input_ids_offset.to_le_bytes());
    buffer[24..28].copy_from_slice(&attention_mask_offset.to_le_bytes());
    buffer[28..32].copy_from_slice(&sequence_offsets_offset.to_le_bytes());

    // Write input_ids
    let ids_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            request.input_ids.as_ptr() as *const u8,
            request.input_ids.len() * 4,
        )
    };
    buffer[input_ids_offset as usize..attention_mask_offset as usize].copy_from_slice(ids_bytes);

    // Write attention_mask
    buffer[attention_mask_offset as usize..sequence_offsets_offset as usize]
        .copy_from_slice(&request.attention_mask);

    // Write sequence_offsets
    let offsets_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            request.sequence_offsets.as_ptr() as *const u8,
            request.sequence_offsets.len() * 4,
        )
    };
    buffer[sequence_offsets_offset as usize..].copy_from_slice(offsets_bytes);

    buffer
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PreTokenizedRequest;

    #[test]
    fn test_message_header() {
        let header = MessageHeader::new(SerializationFormat::Bincode, 1234);
        let bytes = header.to_bytes();
        let parsed = MessageHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.magic, MessageHeader::MAGIC);
        assert_eq!(parsed.version, MessageHeader::VERSION);
        assert_eq!(parsed.format, SerializationFormat::Bincode);
        assert_eq!(parsed.length, 1234);
    }

    #[test]
    fn test_serializer_bincode() {
        let serializer = Serializer::new(SerializationFormat::Bincode);

        let mut request = PreTokenizedRequest::new(42);
        request.add_sequence(&[101, 2054, 102], &[1, 1, 1], 5);

        let bytes = serializer.serialize(&request).unwrap();
        let decoded: PreTokenizedRequest = serializer.deserialize(&bytes).unwrap();

        assert_eq!(decoded.batch_id, 42);
        assert_eq!(decoded.num_sequences(), 1);
        assert_eq!(decoded.sequence_ids(0), Some(&[101u32, 2054, 102][..]));
    }

    #[test]
    fn test_serializer_json() {
        let serializer = Serializer::new(SerializationFormat::Json);

        let mut request = PreTokenizedRequest::new(42);
        request.add_sequence(&[101, 2054, 102], &[1, 1, 1], 5);

        let bytes = serializer.serialize(&request).unwrap();
        let decoded: PreTokenizedRequest = serializer.deserialize(&bytes).unwrap();

        assert_eq!(decoded.batch_id, 42);
    }

    #[test]
    fn test_serializer_without_header() {
        let serializer = Serializer::without_header(SerializationFormat::Bincode);

        let mut request = PreTokenizedRequest::new(42);
        request.add_sequence(&[101, 2054, 102], &[1, 1, 1], 5);

        let bytes = serializer.serialize(&request).unwrap();
        // Should not have header
        assert!(bytes.len() < 100); // Much smaller than with header overhead

        let decoded: PreTokenizedRequest = serializer.deserialize(&bytes).unwrap();
        assert_eq!(decoded.batch_id, 42);
    }

    #[test]
    fn test_batch_serializer() {
        let mut batch = BatchSerializer::new(SerializationFormat::Bincode);

        let requests: Vec<PreTokenizedRequest> = (0..5)
            .map(|i| {
                let mut req = PreTokenizedRequest::new(i);
                req.add_sequence(&[101, 102], &[1, 1], 2);
                req
            })
            .collect();

        let bytes = batch.serialize_batch(&requests).unwrap().to_vec();
        let decoded: Vec<PreTokenizedRequest> = batch.deserialize_batch(&bytes).unwrap();

        assert_eq!(decoded.len(), 5);
        for (i, req) in decoded.iter().enumerate() {
            assert_eq!(req.batch_id, i as u64);
        }
    }

    #[test]
    fn test_raw_serialization() {
        let mut request = PreTokenizedRequest::new(42);
        request.add_sequence(&[101, 2054, 102], &[1, 1, 1], 5);
        request.add_sequence(&[101, 3176, 102], &[1, 1, 1], 6);

        let raw = serialize_raw(&request);
        let view = PreTokenizedView::from_bytes(&raw).unwrap();

        assert_eq!(view.batch_id(), 42);
        assert_eq!(view.num_sequences(), 2);
        assert_eq!(view.total_tokens(), 6);
        assert_eq!(view.input_ids(), &[101, 2054, 102, 101, 3176, 102]);
        assert_eq!(view.attention_mask(), &[1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_serialization_format_id() {
        assert_eq!(SerializationFormat::Bincode.id(), 0x01);
        assert_eq!(SerializationFormat::Json.id(), 0x02);
        assert_eq!(SerializationFormat::Raw.id(), 0x03);

        assert_eq!(SerializationFormat::from_id(0x01), Some(SerializationFormat::Bincode));
        assert_eq!(SerializationFormat::from_id(0x02), Some(SerializationFormat::Json));
        assert_eq!(SerializationFormat::from_id(0xFF), None);
    }

    #[test]
    fn test_size_estimate() {
        let mut request = PreTokenizedRequest::new(1);
        request.add_sequence(&[101, 2054, 102], &[1, 1, 1], 5);

        let estimated = Serializer::estimate_size(&request);
        let serializer = Serializer::new(SerializationFormat::Bincode);
        let actual = serializer.serialize(&request).unwrap().len();

        // Estimate should be within 50% of actual
        assert!(estimated > actual / 2);
        assert!(estimated < actual * 2);
    }
}
