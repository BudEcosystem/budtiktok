//! Token Budget Router (7.2.1, 7.2.2)
//!
//! Routes and batches pre-tokenized requests based on token budget,
//! compatible with LatentBud's token-budget continuous batching system.
//!
//! # Features
//!
//! - Token-budget batching: Groups requests by total padded tokens
//! - Longest-first sorting: Prioritizes longer sequences for better packing
//! - Timeout-based flushing: Ensures latency SLAs even with partial batches
//! - Async/await compatible: Works with Tokio runtime

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot};

/// Configuration for the token budget router
#[derive(Debug, Clone)]
pub struct TokenBudgetConfig {
    /// Maximum tokens per batch (default: 16384)
    pub max_batch_tokens: usize,
    /// Maximum sequences per batch (default: 64)
    pub max_batch_size: usize,
    /// Flush timeout in milliseconds (default: 10)
    pub flush_timeout_ms: u64,
    /// Minimum batch fill ratio before timeout flush (default: 0.5)
    pub min_fill_ratio: f32,
    /// Enable longest-first sorting (default: true)
    pub longest_first: bool,
}

impl Default for TokenBudgetConfig {
    fn default() -> Self {
        Self {
            max_batch_tokens: 16384,
            max_batch_size: 64,
            flush_timeout_ms: 10,
            min_fill_ratio: 0.5,
            longest_first: true,
        }
    }
}

/// A pending request with token information
#[derive(Debug)]
pub struct PendingRequest {
    /// Request ID
    pub id: u64,
    /// Token IDs
    pub token_ids: Vec<u32>,
    /// Attention mask
    pub attention_mask: Vec<u8>,
    /// Original text length (for sorting)
    pub original_length: u32,
    /// Priority (higher = more urgent)
    pub priority: u8,
    /// Time request was queued
    pub queued_at: Instant,
    /// Response channel
    pub response_tx: Option<oneshot::Sender<BatchResult>>,
}

impl PendingRequest {
    /// Get padded length (token count)
    pub fn padded_length(&self) -> usize {
        self.token_ids.len()
    }
}

/// Ordering for longest-first priority queue
impl PartialEq for PendingRequest {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for PendingRequest {}

impl PartialOrd for PendingRequest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PendingRequest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap, pops the "greatest" element first
        // We want: higher priority first, then longer sequences, then earlier queued
        match self.priority.cmp(&other.priority) {
            std::cmp::Ordering::Equal => {}
            ord => return ord, // Higher priority = greater = comes first
        }
        match self.padded_length().cmp(&other.padded_length()) {
            std::cmp::Ordering::Equal => {}
            ord => return ord, // Longer length = greater = comes first
        }
        // For queued time, earlier should come first, so reverse
        other.queued_at.cmp(&self.queued_at) // Earlier queued = greater = comes first
    }
}

/// Result of batch processing
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Request ID
    pub request_id: u64,
    /// Embedding vectors (if successful)
    pub embeddings: Option<Vec<f32>>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Processing time in microseconds
    pub processing_time_us: u64,
}

/// A formed batch ready for processing
#[derive(Debug)]
pub struct TokenBudgetBatch {
    /// Batch ID
    pub batch_id: u64,
    /// Requests in this batch
    pub requests: Vec<PendingRequest>,
    /// Total padded tokens in batch
    pub total_tokens: usize,
    /// Maximum sequence length in batch
    pub max_seq_length: usize,
    /// Time batch was formed
    pub formed_at: Instant,
}

impl TokenBudgetBatch {
    /// Get the number of sequences in the batch
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Calculate batch efficiency (actual tokens / padded tokens)
    pub fn efficiency(&self) -> f32 {
        if self.total_tokens == 0 {
            return 0.0;
        }
        let actual: usize = self.requests.iter().map(|r| r.original_length as usize).sum();
        actual as f32 / self.total_tokens as f32
    }
}

/// Token budget router state
struct RouterState {
    /// Pending requests queue (priority queue for longest-first)
    pending: BinaryHeap<PendingRequest>,
    /// Current pending token count
    pending_tokens: usize,
    /// Time of last batch formation
    last_batch_time: Instant,
}

/// Token Budget Router
///
/// Routes and batches requests based on token budget.
/// Compatible with LatentBud's token-budget continuous batching.
pub struct TokenBudgetRouter {
    config: TokenBudgetConfig,
    state: Mutex<RouterState>,
    batch_counter: AtomicU64,

    // Statistics
    batches_formed: AtomicU64,
    requests_processed: AtomicU64,
    tokens_processed: AtomicU64,
    timeout_flushes: AtomicU64,
}

impl TokenBudgetRouter {
    /// Create a new token budget router
    pub fn new(config: TokenBudgetConfig) -> Self {
        Self {
            config,
            state: Mutex::new(RouterState {
                pending: BinaryHeap::new(),
                pending_tokens: 0,
                last_batch_time: Instant::now(),
            }),
            batch_counter: AtomicU64::new(0),
            batches_formed: AtomicU64::new(0),
            requests_processed: AtomicU64::new(0),
            tokens_processed: AtomicU64::new(0),
            timeout_flushes: AtomicU64::new(0),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(TokenBudgetConfig::default())
    }

    /// Submit a request to the router
    pub fn submit(&self, request: PendingRequest) -> Option<TokenBudgetBatch> {
        let mut state = self.state.lock();
        let tokens = request.padded_length();

        state.pending.push(request);
        state.pending_tokens += tokens;

        // Check if we should form a batch
        self.maybe_form_batch(&mut state, false)
    }

    /// Check for timeout-based flush
    pub fn check_timeout(&self) -> Option<TokenBudgetBatch> {
        let mut state = self.state.lock();

        if state.pending.is_empty() {
            return None;
        }

        let elapsed = state.last_batch_time.elapsed();
        if elapsed.as_millis() >= self.config.flush_timeout_ms as u128 {
            // Check minimum fill ratio
            let fill_ratio = state.pending_tokens as f32 / self.config.max_batch_tokens as f32;
            if fill_ratio >= self.config.min_fill_ratio || state.pending.len() > 0 {
                self.timeout_flushes.fetch_add(1, Ordering::Relaxed);
                return self.form_batch(&mut state);
            }
        }

        None
    }

    /// Force flush all pending requests
    pub fn flush(&self) -> Option<TokenBudgetBatch> {
        let mut state = self.state.lock();
        if state.pending.is_empty() {
            return None;
        }
        self.form_batch(&mut state)
    }

    /// Maybe form a batch if conditions are met
    fn maybe_form_batch(&self, state: &mut RouterState, force: bool) -> Option<TokenBudgetBatch> {
        // Check if we have enough tokens or sequences
        let should_batch = force
            || state.pending_tokens >= self.config.max_batch_tokens
            || state.pending.len() >= self.config.max_batch_size;

        if should_batch {
            self.form_batch(state)
        } else {
            None
        }
    }

    /// Form a batch from pending requests
    fn form_batch(&self, state: &mut RouterState) -> Option<TokenBudgetBatch> {
        if state.pending.is_empty() {
            return None;
        }

        let batch_id = self.batch_counter.fetch_add(1, Ordering::Relaxed);
        let mut requests = Vec::with_capacity(self.config.max_batch_size.min(state.pending.len()));
        let mut total_tokens = 0;
        let mut max_seq_length = 0;

        // Greedily fill batch respecting token budget
        while let Some(req) = state.pending.peek() {
            let req_tokens = req.padded_length();

            // Check if adding this request would exceed budget
            if !requests.is_empty() {
                // Calculate new padded tokens (all sequences padded to new max length)
                let new_max = max_seq_length.max(req_tokens);
                let new_total = (requests.len() + 1) * new_max;

                if new_total > self.config.max_batch_tokens {
                    break;
                }
                if requests.len() >= self.config.max_batch_size {
                    break;
                }
            }

            // Add to batch
            let req = state.pending.pop().unwrap();
            let req_tokens = req.padded_length();
            max_seq_length = max_seq_length.max(req_tokens);
            requests.push(req);

            // Recalculate total tokens with padding
            total_tokens = requests.len() * max_seq_length;
        }

        // Update state
        state.pending_tokens = state.pending.iter().map(|r| r.padded_length()).sum();
        state.last_batch_time = Instant::now();

        // Update statistics
        self.batches_formed.fetch_add(1, Ordering::Relaxed);
        self.requests_processed.fetch_add(requests.len() as u64, Ordering::Relaxed);
        self.tokens_processed.fetch_add(total_tokens as u64, Ordering::Relaxed);

        Some(TokenBudgetBatch {
            batch_id,
            requests,
            total_tokens,
            max_seq_length,
            formed_at: Instant::now(),
        })
    }

    /// Get current queue depth
    pub fn queue_depth(&self) -> usize {
        self.state.lock().pending.len()
    }

    /// Get pending token count
    pub fn pending_tokens(&self) -> usize {
        self.state.lock().pending_tokens
    }

    /// Get router statistics
    pub fn stats(&self) -> TokenBudgetStats {
        let state = self.state.lock();
        TokenBudgetStats {
            batches_formed: self.batches_formed.load(Ordering::Relaxed),
            requests_processed: self.requests_processed.load(Ordering::Relaxed),
            tokens_processed: self.tokens_processed.load(Ordering::Relaxed),
            timeout_flushes: self.timeout_flushes.load(Ordering::Relaxed),
            queue_depth: state.pending.len(),
            pending_tokens: state.pending_tokens,
        }
    }
}

/// Router statistics
#[derive(Debug, Clone)]
pub struct TokenBudgetStats {
    pub batches_formed: u64,
    pub requests_processed: u64,
    pub tokens_processed: u64,
    pub timeout_flushes: u64,
    pub queue_depth: usize,
    pub pending_tokens: usize,
}

// =============================================================================
// Async Token Budget Router (7.2.2)
// =============================================================================

/// Message for async router communication
enum RouterMessage {
    Submit(PendingRequest),
    Flush,
    Shutdown,
}

/// Async token budget router with timeout-based flushing
pub struct AsyncTokenBudgetRouter {
    router: Arc<TokenBudgetRouter>,
    tx: mpsc::Sender<RouterMessage>,
}

impl AsyncTokenBudgetRouter {
    /// Create and start an async token budget router
    pub fn new(config: TokenBudgetConfig) -> (Self, mpsc::Receiver<TokenBudgetBatch>) {
        let router = Arc::new(TokenBudgetRouter::new(config.clone()));
        let (msg_tx, mut msg_rx) = mpsc::channel::<RouterMessage>(1000);
        let (batch_tx, batch_rx) = mpsc::channel::<TokenBudgetBatch>(100);

        let router_clone = Arc::clone(&router);
        let timeout_ms = config.flush_timeout_ms;

        // Spawn background task for timeout checking and message processing
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(timeout_ms / 2));

            loop {
                tokio::select! {
                    // Process incoming messages
                    msg = msg_rx.recv() => {
                        match msg {
                            Some(RouterMessage::Submit(request)) => {
                                if let Some(batch) = router_clone.submit(request) {
                                    let _ = batch_tx.send(batch).await;
                                }
                            }
                            Some(RouterMessage::Flush) => {
                                if let Some(batch) = router_clone.flush() {
                                    let _ = batch_tx.send(batch).await;
                                }
                            }
                            Some(RouterMessage::Shutdown) | None => {
                                // Final flush before shutdown
                                if let Some(batch) = router_clone.flush() {
                                    let _ = batch_tx.send(batch).await;
                                }
                                break;
                            }
                        }
                    }
                    // Check for timeout-based flush
                    _ = interval.tick() => {
                        if let Some(batch) = router_clone.check_timeout() {
                            let _ = batch_tx.send(batch).await;
                        }
                    }
                }
            }
        });

        (Self { router, tx: msg_tx }, batch_rx)
    }

    /// Submit a request asynchronously
    pub async fn submit(&self, request: PendingRequest) -> Result<(), mpsc::error::SendError<RouterMessage>> {
        self.tx.send(RouterMessage::Submit(request)).await
            .map_err(|_| mpsc::error::SendError(RouterMessage::Flush))
    }

    /// Force flush pending requests
    pub async fn flush(&self) -> Result<(), mpsc::error::SendError<RouterMessage>> {
        self.tx.send(RouterMessage::Flush).await
            .map_err(|_| mpsc::error::SendError(RouterMessage::Flush))
    }

    /// Shutdown the router
    pub async fn shutdown(&self) -> Result<(), mpsc::error::SendError<RouterMessage>> {
        self.tx.send(RouterMessage::Shutdown).await
            .map_err(|_| mpsc::error::SendError(RouterMessage::Flush))
    }

    /// Get router statistics
    pub fn stats(&self) -> TokenBudgetStats {
        self.router.stats()
    }

    /// Get queue depth
    pub fn queue_depth(&self) -> usize {
        self.router.queue_depth()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_request(id: u64, length: usize) -> PendingRequest {
        PendingRequest {
            id,
            token_ids: vec![1; length],
            attention_mask: vec![1; length],
            original_length: length as u32,
            priority: 0,
            queued_at: Instant::now(),
            response_tx: None,
        }
    }

    #[test]
    fn test_token_budget_config_default() {
        let config = TokenBudgetConfig::default();
        assert_eq!(config.max_batch_tokens, 16384);
        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.flush_timeout_ms, 10);
    }

    #[test]
    fn test_router_submit_and_batch() {
        let config = TokenBudgetConfig {
            max_batch_tokens: 100,
            max_batch_size: 4,
            ..Default::default()
        };
        let router = TokenBudgetRouter::new(config);

        // Submit requests until batch forms
        for i in 0..3 {
            let result = router.submit(create_request(i, 20));
            assert!(result.is_none()); // Not enough for batch
        }

        // This should trigger a batch (4 * 20 = 80 tokens, plus padding)
        let batch = router.submit(create_request(3, 30));
        assert!(batch.is_some());

        let batch = batch.unwrap();
        assert!(batch.len() <= 4);
        assert!(batch.total_tokens <= 100);
    }

    #[test]
    fn test_router_longest_first_ordering() {
        let config = TokenBudgetConfig {
            max_batch_tokens: 200,
            max_batch_size: 10,
            ..Default::default()
        };
        let router = TokenBudgetRouter::new(config);

        // Submit in random order
        router.submit(create_request(1, 10));
        router.submit(create_request(2, 50));
        router.submit(create_request(3, 30));
        router.submit(create_request(4, 40));
        router.submit(create_request(5, 20));

        let batch = router.flush().unwrap();

        // Longest should come first
        assert_eq!(batch.requests[0].padded_length(), 50);
    }

    #[test]
    fn test_router_flush() {
        let config = TokenBudgetConfig {
            max_batch_tokens: 1000,
            max_batch_size: 100,
            ..Default::default()
        };
        let router = TokenBudgetRouter::new(config);

        // Submit a few requests
        router.submit(create_request(1, 20));
        router.submit(create_request(2, 30));

        // Flush should return batch even if not full
        let batch = router.flush();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 2);
    }

    #[test]
    fn test_router_stats() {
        let router = TokenBudgetRouter::with_defaults();

        let stats = router.stats();
        assert_eq!(stats.batches_formed, 0);
        assert_eq!(stats.queue_depth, 0);
    }

    #[test]
    fn test_batch_efficiency() {
        let mut batch = TokenBudgetBatch {
            batch_id: 1,
            requests: vec![],
            total_tokens: 100,
            max_seq_length: 50,
            formed_at: Instant::now(),
        };

        // Empty batch
        assert_eq!(batch.efficiency(), 0.0);

        // Add requests with different original lengths
        batch.requests.push(PendingRequest {
            id: 1,
            token_ids: vec![1; 50],
            attention_mask: vec![1; 50],
            original_length: 40, // Shorter than padded
            priority: 0,
            queued_at: Instant::now(),
            response_tx: None,
        });
        batch.requests.push(PendingRequest {
            id: 2,
            token_ids: vec![1; 50],
            attention_mask: vec![1; 50],
            original_length: 30, // Even shorter
            priority: 0,
            queued_at: Instant::now(),
            response_tx: None,
        });

        // Efficiency = (40 + 30) / 100 = 0.7
        assert!((batch.efficiency() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_pending_request_ordering() {
        let r1 = create_request(1, 50);
        let r2 = create_request(2, 100);
        let r3 = create_request(3, 50);

        // r2 (longer) should be "greater" so it comes first in max-heap
        assert!(r2 > r1);
        assert!(r2 > r3);
    }
}
