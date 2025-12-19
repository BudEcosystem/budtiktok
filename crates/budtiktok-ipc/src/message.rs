//! IPC message definitions
//!
//! This module defines the message types used for inter-process communication
//! between the BudTikTok tokenizer and LatentBud embedding server.
//!
//! # Message Types
//!
//! - `IpcMessage`: General IPC messages for tokenization requests
//! - `PreTokenizedRequest`: Pre-tokenized batch for embedding inference
//! - `EmbeddingResponse`: Embedding vectors response
//!
//! # Serialization
//!
//! Messages support both serde (JSON/bincode) and zero-copy (rkyv) serialization
//! for maximum performance in different scenarios.

use serde::{Deserialize, Serialize};

/// IPC message type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IpcMessage {
    /// Request to tokenize text
    TokenizeRequest {
        id: u64,
        text: String,
        add_special_tokens: bool,
    },

    /// Response with tokenization result
    TokenizeResponse {
        id: u64,
        ids: Vec<u32>,
        tokens: Vec<String>,
    },

    /// Batch tokenize request
    BatchTokenizeRequest {
        id: u64,
        texts: Vec<String>,
        add_special_tokens: bool,
    },

    /// Batch tokenize response
    BatchTokenizeResponse {
        id: u64,
        results: Vec<Vec<u32>>,
    },

    /// Pre-tokenized batch request for embedding
    PreTokenizedBatch(PreTokenizedRequest),

    /// Embedding response
    EmbeddingBatch(EmbeddingResponse),

    /// Error response
    Error {
        id: u64,
        message: String,
    },

    /// Ping for health check
    Ping { id: u64 },

    /// Pong response
    Pong { id: u64 },

    /// Shutdown request
    Shutdown,
}

impl IpcMessage {
    /// Get the message ID
    pub fn id(&self) -> Option<u64> {
        match self {
            IpcMessage::TokenizeRequest { id, .. } => Some(*id),
            IpcMessage::TokenizeResponse { id, .. } => Some(*id),
            IpcMessage::BatchTokenizeRequest { id, .. } => Some(*id),
            IpcMessage::BatchTokenizeResponse { id, .. } => Some(*id),
            IpcMessage::PreTokenizedBatch(req) => Some(req.batch_id),
            IpcMessage::EmbeddingBatch(resp) => Some(resp.batch_id),
            IpcMessage::Error { id, .. } => Some(*id),
            IpcMessage::Ping { id } => Some(*id),
            IpcMessage::Pong { id } => Some(*id),
            IpcMessage::Shutdown => None,
        }
    }
}

/// Pre-tokenized request for embedding inference
///
/// This structure is optimized for the LatentBud token-budget batching system.
/// It contains pre-tokenized sequences ready for embedding model inference.
///
/// # Memory Layout
///
/// The data is stored in a compact format:
/// - `input_ids`: Concatenated token IDs for all sequences
/// - `attention_mask`: Concatenated attention masks
/// - `sequence_offsets`: Start offset of each sequence in the concatenated arrays
///
/// This allows efficient memory transfer to GPU and supports variable-length sequences.
///
/// # Example
///
/// ```rust,ignore
/// let request = PreTokenizedRequest {
///     batch_id: 1,
///     input_ids: vec![101, 2054, 102, 101, 3176, 102],  // Two sequences
///     attention_mask: vec![1, 1, 1, 1, 1, 1],
///     sequence_offsets: vec![0, 3, 6],  // seq0: [0..3], seq1: [3..6]
///     original_lengths: vec![1, 1],  // Original text lengths for tracking
///     metadata: BatchMetadata::default(),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PreTokenizedRequest {
    /// Unique batch identifier for request/response matching
    pub batch_id: u64,

    /// Concatenated token IDs for all sequences
    /// Layout: [seq0_tokens..., seq1_tokens..., ...]
    pub input_ids: Vec<u32>,

    /// Attention mask (1 for real tokens, 0 for padding)
    /// Same layout as input_ids
    pub attention_mask: Vec<u8>,

    /// Token type IDs for segment embeddings (BERT-style)
    /// None if not used by the model
    pub token_type_ids: Option<Vec<u8>>,

    /// Byte offsets marking the start of each sequence in the concatenated arrays
    /// Length = num_sequences + 1 (last element is total length)
    pub sequence_offsets: Vec<u32>,

    /// Original text lengths (in chars) for each sequence
    /// Used for tracking and debugging
    pub original_lengths: Vec<u32>,

    /// Request metadata
    pub metadata: BatchMetadata,
}

impl PreTokenizedRequest {
    /// Create a new empty request
    pub fn new(batch_id: u64) -> Self {
        Self {
            batch_id,
            input_ids: Vec::new(),
            attention_mask: Vec::new(),
            token_type_ids: None,
            sequence_offsets: vec![0],
            original_lengths: Vec::new(),
            metadata: BatchMetadata::default(),
        }
    }

    /// Create a request with pre-allocated capacity
    pub fn with_capacity(batch_id: u64, num_sequences: usize, total_tokens: usize) -> Self {
        Self {
            batch_id,
            input_ids: Vec::with_capacity(total_tokens),
            attention_mask: Vec::with_capacity(total_tokens),
            token_type_ids: None,
            sequence_offsets: Vec::with_capacity(num_sequences + 1),
            original_lengths: Vec::with_capacity(num_sequences),
            metadata: BatchMetadata::default(),
        }
    }

    /// Add a sequence to the batch
    pub fn add_sequence(&mut self, ids: &[u32], attention: &[u8], original_len: u32) {
        self.input_ids.extend_from_slice(ids);
        self.attention_mask.extend_from_slice(attention);
        self.sequence_offsets.push(self.input_ids.len() as u32);
        self.original_lengths.push(original_len);
    }

    /// Add a sequence with token type IDs
    pub fn add_sequence_with_types(
        &mut self,
        ids: &[u32],
        attention: &[u8],
        types: &[u8],
        original_len: u32,
    ) {
        self.input_ids.extend_from_slice(ids);
        self.attention_mask.extend_from_slice(attention);

        let type_ids = self.token_type_ids.get_or_insert_with(Vec::new);
        type_ids.extend_from_slice(types);

        self.sequence_offsets.push(self.input_ids.len() as u32);
        self.original_lengths.push(original_len);
    }

    /// Get the number of sequences in the batch
    pub fn num_sequences(&self) -> usize {
        self.original_lengths.len()
    }

    /// Get the total number of tokens in the batch
    pub fn total_tokens(&self) -> usize {
        self.input_ids.len()
    }

    /// Get the token IDs for a specific sequence
    pub fn sequence_ids(&self, idx: usize) -> Option<&[u32]> {
        if idx >= self.num_sequences() {
            return None;
        }
        let start = self.sequence_offsets[idx] as usize;
        let end = self.sequence_offsets[idx + 1] as usize;
        Some(&self.input_ids[start..end])
    }

    /// Get the attention mask for a specific sequence
    pub fn sequence_attention(&self, idx: usize) -> Option<&[u8]> {
        if idx >= self.num_sequences() {
            return None;
        }
        let start = self.sequence_offsets[idx] as usize;
        let end = self.sequence_offsets[idx + 1] as usize;
        Some(&self.attention_mask[start..end])
    }

    /// Get the length of a specific sequence
    pub fn sequence_length(&self, idx: usize) -> Option<usize> {
        if idx >= self.num_sequences() {
            return None;
        }
        let start = self.sequence_offsets[idx] as usize;
        let end = self.sequence_offsets[idx + 1] as usize;
        Some(end - start)
    }

    /// Calculate the padded token count for token-budget batching
    ///
    /// For longest-first batching, we calculate:
    /// sum(max_len_so_far * 1) for each sequence added
    pub fn padded_token_count(&self) -> usize {
        if self.num_sequences() == 0 {
            return 0;
        }

        let mut max_len = 0usize;
        let mut total = 0usize;

        for i in 0..self.num_sequences() {
            let len = self.sequence_length(i).unwrap_or(0);
            max_len = max_len.max(len);
            total += max_len;
        }

        total
    }

    /// Validate the request structure
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.sequence_offsets.is_empty() {
            return Err("sequence_offsets cannot be empty");
        }

        if self.sequence_offsets.len() != self.original_lengths.len() + 1 {
            return Err("sequence_offsets length mismatch");
        }

        let expected_len = *self.sequence_offsets.last().unwrap() as usize;
        if self.input_ids.len() != expected_len {
            return Err("input_ids length mismatch");
        }

        if self.attention_mask.len() != expected_len {
            return Err("attention_mask length mismatch");
        }

        if let Some(ref types) = self.token_type_ids {
            if types.len() != expected_len {
                return Err("token_type_ids length mismatch");
            }
        }

        Ok(())
    }

    /// Estimate serialized size in bytes
    pub fn estimated_size(&self) -> usize {
        let base = 8 + 4 * 3; // batch_id + 3 vec headers
        let ids = self.input_ids.len() * 4;
        let mask = self.attention_mask.len();
        let types = self.token_type_ids.as_ref().map(|t| t.len()).unwrap_or(0);
        let offsets = self.sequence_offsets.len() * 4;
        let lengths = self.original_lengths.len() * 4;

        base + ids + mask + types + offsets + lengths + 64 // metadata estimate
    }
}

/// Batch metadata for tracking and debugging
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BatchMetadata {
    /// Model identifier
    pub model_id: Option<String>,

    /// Maximum sequence length (for padding reference)
    pub max_length: u32,

    /// Whether sequences are padded to max_length
    pub is_padded: bool,

    /// Truncation applied
    pub truncated: bool,

    /// Priority level (higher = more urgent)
    pub priority: u8,

    /// Timestamp when batch was created (unix millis)
    pub created_at: u64,

    /// Client identifier
    pub client_id: Option<String>,

    /// Custom metadata fields
    pub custom: Option<std::collections::HashMap<String, String>>,
}

impl BatchMetadata {
    /// Create metadata with current timestamp
    pub fn now() -> Self {
        Self {
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            ..Default::default()
        }
    }
}

/// Embedding response for a pre-tokenized batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// Batch ID matching the request
    pub batch_id: u64,

    /// Embedding vectors (flattened)
    /// Layout: [seq0_emb..., seq1_emb..., ...]
    /// Each embedding has `embedding_dim` elements
    pub embeddings: Vec<f32>,

    /// Embedding dimension
    pub embedding_dim: u32,

    /// Number of sequences in response
    pub num_sequences: u32,

    /// Response metadata
    pub metadata: ResponseMetadata,
}

impl EmbeddingResponse {
    /// Create a new embedding response
    pub fn new(batch_id: u64, embeddings: Vec<f32>, embedding_dim: u32) -> Self {
        let num_sequences = if embedding_dim > 0 {
            embeddings.len() as u32 / embedding_dim
        } else {
            0
        };

        Self {
            batch_id,
            embeddings,
            embedding_dim,
            num_sequences,
            metadata: ResponseMetadata::default(),
        }
    }

    /// Get the embedding for a specific sequence
    pub fn get_embedding(&self, idx: usize) -> Option<&[f32]> {
        if idx >= self.num_sequences as usize {
            return None;
        }
        let start = idx * self.embedding_dim as usize;
        let end = start + self.embedding_dim as usize;
        if end > self.embeddings.len() {
            return None;
        }
        Some(&self.embeddings[start..end])
    }

    /// Iterate over all embeddings
    pub fn iter_embeddings(&self) -> impl Iterator<Item = &[f32]> {
        (0..self.num_sequences as usize).filter_map(|i| self.get_embedding(i))
    }

    /// Validate response structure
    pub fn validate(&self) -> Result<(), &'static str> {
        let expected = self.num_sequences as usize * self.embedding_dim as usize;
        if self.embeddings.len() != expected {
            return Err("embeddings length mismatch");
        }
        Ok(())
    }
}

/// Response metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResponseMetadata {
    /// Processing time in microseconds
    pub processing_time_us: u64,

    /// Model used for inference
    pub model_id: Option<String>,

    /// GPU device used
    pub device: Option<String>,

    /// Inference batch size (may differ from request if rebatched)
    pub inference_batch_size: u32,

    /// Warning messages
    pub warnings: Vec<String>,
}

/// Builder for PreTokenizedRequest
pub struct PreTokenizedRequestBuilder {
    request: PreTokenizedRequest,
}

impl PreTokenizedRequestBuilder {
    /// Create a new builder
    pub fn new(batch_id: u64) -> Self {
        Self {
            request: PreTokenizedRequest::new(batch_id),
        }
    }

    /// Set model ID
    pub fn model(mut self, model_id: impl Into<String>) -> Self {
        self.request.metadata.model_id = Some(model_id.into());
        self
    }

    /// Set client ID
    pub fn client(mut self, client_id: impl Into<String>) -> Self {
        self.request.metadata.client_id = Some(client_id.into());
        self
    }

    /// Set priority
    pub fn priority(mut self, priority: u8) -> Self {
        self.request.metadata.priority = priority;
        self
    }

    /// Add a sequence
    pub fn add_sequence(mut self, ids: &[u32], original_len: u32) -> Self {
        let attention: Vec<u8> = vec![1; ids.len()];
        self.request.add_sequence(ids, &attention, original_len);
        self
    }

    /// Add a sequence with custom attention mask
    pub fn add_sequence_with_attention(
        mut self,
        ids: &[u32],
        attention: &[u8],
        original_len: u32,
    ) -> Self {
        self.request.add_sequence(ids, attention, original_len);
        self
    }

    /// Build the request
    pub fn build(mut self) -> PreTokenizedRequest {
        self.request.metadata.created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.request
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pre_tokenized_request_new() {
        let req = PreTokenizedRequest::new(1);
        assert_eq!(req.batch_id, 1);
        assert_eq!(req.num_sequences(), 0);
        assert_eq!(req.total_tokens(), 0);
    }

    #[test]
    fn test_pre_tokenized_request_add_sequence() {
        let mut req = PreTokenizedRequest::new(1);
        req.add_sequence(&[101, 2054, 102], &[1, 1, 1], 5);
        req.add_sequence(&[101, 3176, 4523, 102], &[1, 1, 1, 1], 8);

        assert_eq!(req.num_sequences(), 2);
        assert_eq!(req.total_tokens(), 7);
        assert_eq!(req.sequence_ids(0), Some(&[101u32, 2054, 102][..]));
        assert_eq!(req.sequence_ids(1), Some(&[101u32, 3176, 4523, 102][..]));
        assert_eq!(req.sequence_length(0), Some(3));
        assert_eq!(req.sequence_length(1), Some(4));
    }

    #[test]
    fn test_pre_tokenized_request_validate() {
        let mut req = PreTokenizedRequest::new(1);
        req.add_sequence(&[101, 102], &[1, 1], 2);

        assert!(req.validate().is_ok());

        // Corrupt the data
        req.input_ids.push(999);
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_pre_tokenized_request_padded_count() {
        let mut req = PreTokenizedRequest::new(1);
        // Add sequences in decreasing order (longest-first)
        req.add_sequence(&[1, 2, 3, 4, 5], &[1, 1, 1, 1, 1], 10);
        req.add_sequence(&[1, 2, 3], &[1, 1, 1], 6);
        req.add_sequence(&[1, 2], &[1, 1], 4);

        // Padded count: 5 + 5 + 5 = 15 (all pad to max=5)
        assert_eq!(req.padded_token_count(), 15);
    }

    #[test]
    fn test_embedding_response() {
        let embeddings = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 sequences, dim=3
        let resp = EmbeddingResponse::new(1, embeddings, 3);

        assert_eq!(resp.num_sequences, 2);
        assert_eq!(resp.get_embedding(0), Some(&[1.0f32, 2.0, 3.0][..]));
        assert_eq!(resp.get_embedding(1), Some(&[4.0f32, 5.0, 6.0][..]));
        assert!(resp.validate().is_ok());
    }

    #[test]
    fn test_builder() {
        let req = PreTokenizedRequestBuilder::new(42)
            .model("bge-small-en-v1.5")
            .client("test-client")
            .priority(10)
            .add_sequence(&[101, 2054, 102], 5)
            .add_sequence(&[101, 3176, 102], 6)
            .build();

        assert_eq!(req.batch_id, 42);
        assert_eq!(req.num_sequences(), 2);
        assert_eq!(req.metadata.model_id, Some("bge-small-en-v1.5".to_string()));
        assert_eq!(req.metadata.client_id, Some("test-client".to_string()));
        assert_eq!(req.metadata.priority, 10);
    }

    #[test]
    fn test_ipc_message_id() {
        let msg = IpcMessage::PreTokenizedBatch(PreTokenizedRequest::new(123));
        assert_eq!(msg.id(), Some(123));

        let embeddings = vec![1.0, 2.0, 3.0];
        let msg = IpcMessage::EmbeddingBatch(EmbeddingResponse::new(456, embeddings, 3));
        assert_eq!(msg.id(), Some(456));
    }
}
