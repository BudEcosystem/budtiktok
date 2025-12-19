//! Unigram tokenization algorithm
//!
//! Implementation of the Unigram (SentencePiece) algorithm that uses
//! dynamic programming to find the most likely segmentation.
//! Includes support for:
//! - Viterbi decoding (optimal segmentation)
//! - Byte fallback for unknown characters
//! - N-best decoding (top-N segmentations)
//! - Stochastic sampling

use ahash::AHashMap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::Path;

use crate::encoding::Encoding;
use crate::error::{Error, Result};
use crate::tokenizer::Tokenizer;
use crate::trie::Trie;
use crate::vocab::Vocabulary;

// ============================================================================
// Core Data Structures
// ============================================================================

/// Unigram model piece with score
#[derive(Debug, Clone)]
pub struct UnigramPiece {
    /// Token string
    pub token: String,
    /// Log probability score
    pub score: f64,
}

/// Unigram tokenizer configuration
#[derive(Debug, Clone)]
pub struct UnigramConfig {
    /// Unknown token
    pub unk_token: String,
    /// Unknown token ID
    pub unk_id: u32,
    /// BOS token (optional)
    pub bos_token: Option<String>,
    /// EOS token (optional)
    pub eos_token: Option<String>,
    /// Byte fallback for unknown characters
    pub byte_fallback: bool,
}

impl Default for UnigramConfig {
    fn default() -> Self {
        Self {
            unk_token: "<unk>".to_string(),
            unk_id: 0,
            bos_token: None,
            eos_token: None,
            byte_fallback: false,
        }
    }
}

/// Lattice node for Viterbi decoding
#[derive(Debug, Clone)]
struct LatticeNode {
    /// Start position in text (byte offset)
    start: usize,
    /// End position in text (byte offset)
    end: usize,
    /// Token ID
    token_id: u32,
    /// Score (log probability)
    score: f64,
    /// Best previous node index
    prev: Option<usize>,
    /// Cumulative score
    best_score: f64,
}

// ============================================================================
// Byte Fallback (3.2.2)
// ============================================================================

/// Generate byte fallback token for a given byte value
#[inline]
fn byte_token(b: u8) -> String {
    format!("<0x{:02X}>", b)
}

/// Pre-computed byte token IDs
pub struct ByteTokenIds {
    /// Maps byte value to token ID
    ids: [Option<u32>; 256],
}

impl ByteTokenIds {
    /// Create byte token ID lookup from vocabulary
    pub fn from_vocab(vocab: &Vocabulary) -> Self {
        let mut ids = [None; 256];
        for b in 0u8..=255 {
            let token = byte_token(b);
            ids[b as usize] = vocab.token_to_id(&token);
        }
        Self { ids }
    }

    /// Get token ID for a byte
    #[inline]
    pub fn get(&self, b: u8) -> Option<u32> {
        self.ids[b as usize]
    }

    /// Check if byte fallback is supported
    pub fn is_supported(&self) -> bool {
        // Check if at least some byte tokens exist
        self.ids.iter().filter(|id| id.is_some()).count() > 0
    }
}

// ============================================================================
// N-best Decoding (3.2.4)
// ============================================================================

/// Hypothesis for N-best search
#[derive(Clone)]
struct Hypothesis {
    /// Position in text (byte offset)
    position: usize,
    /// Token IDs collected so far
    tokens: Vec<u32>,
    /// Cumulative score
    score: f64,
}

impl Eq for Hypothesis {}

impl PartialEq for Hypothesis {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Ord for Hypothesis {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher score = higher priority (max heap)
        self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Hypothesis {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ============================================================================
// Unigram Tokenizer
// ============================================================================

/// Unigram tokenizer implementation
pub struct UnigramTokenizer {
    vocabulary: Vocabulary,
    pieces: Vec<UnigramPiece>,
    config: UnigramConfig,
    /// Trie for efficient piece lookup
    trie: Trie,
    /// Score for each token (indexed by vocab index in trie)
    scores: AHashMap<u32, f64>,
    /// Minimum score (for unknown penalty)
    min_score: f64,
    /// Byte token IDs for fallback
    byte_tokens: ByteTokenIds,
}

impl UnigramTokenizer {
    /// Create a new Unigram tokenizer
    pub fn new(vocabulary: Vocabulary, pieces: Vec<UnigramPiece>, config: UnigramConfig) -> Self {
        let min_score = pieces
            .iter()
            .map(|p| p.score)
            .fold(f64::INFINITY, f64::min);

        // Build trie from pieces
        let mut trie_builder = crate::trie::TrieBuilder::new();
        let mut scores = AHashMap::with_capacity(pieces.len());

        for piece in &pieces {
            if let Some(id) = vocabulary.token_to_id(&piece.token) {
                trie_builder.insert(&piece.token, id);
                scores.insert(id, piece.score);
            }
        }

        let trie = trie_builder.build();
        let byte_tokens = ByteTokenIds::from_vocab(&vocabulary);

        Self {
            vocabulary,
            pieces,
            config,
            trie,
            scores,
            min_score,
            byte_tokens,
        }
    }

    /// Create from a SentencePiece model file
    pub fn from_file(_path: impl AsRef<Path>) -> Result<Self> {
        Err(Error::vocab_load("SentencePiece loading not yet implemented"))
    }

    /// Get score for a token ID
    #[inline]
    fn get_score(&self, token_id: u32) -> f64 {
        self.scores.get(&token_id).copied().unwrap_or(self.min_score - 10.0)
    }

    /// Build lattice for Viterbi decoding
    fn build_lattice(&self, text: &str) -> Vec<Vec<LatticeNode>> {
        let bytes = text.as_bytes();
        let n = bytes.len();

        // lattice[i] contains all nodes ending at position i
        let mut lattice: Vec<Vec<LatticeNode>> = vec![Vec::new(); n + 1];

        // Add BOS node
        lattice[0].push(LatticeNode {
            start: 0,
            end: 0,
            token_id: 0,
            score: 0.0,
            prev: None,
            best_score: 0.0,
        });

        for start in 0..n {
            if lattice[start].is_empty() {
                continue;
            }

            // Find best score to reach this position
            let best_prev_score = lattice[start]
                .iter()
                .map(|node| node.best_score)
                .fold(f64::NEG_INFINITY, f64::max);

            let prev_idx = lattice[start].len().saturating_sub(1);

            // Use trie for efficient prefix enumeration
            for (token_len, token_id) in self.trie.common_prefix_search(bytes, start) {
                let end = start + token_len;
                let score = self.get_score(token_id);

                lattice[end].push(LatticeNode {
                    start,
                    end,
                    token_id,
                    score,
                    prev: Some(prev_idx),
                    best_score: best_prev_score + score,
                });
            }

            // Handle byte fallback if enabled and no regular token found
            if self.config.byte_fallback && !lattice[start + 1].iter().any(|n| n.start == start) {
                let byte_val = bytes[start];
                if let Some(byte_id) = self.byte_tokens.get(byte_val) {
                    let score = self.min_score - 5.0; // Penalty for byte fallback
                    lattice[start + 1].push(LatticeNode {
                        start,
                        end: start + 1,
                        token_id: byte_id,
                        score,
                        prev: Some(prev_idx),
                        best_score: best_prev_score + score,
                    });
                }
            }

            // Handle unknown character (single char fallback) if no token found
            if !lattice[start + 1].iter().any(|n| n.start == start) {
                // Find next char boundary
                let mut char_end = start + 1;
                while char_end < n && (bytes[char_end] & 0xC0) == 0x80 {
                    char_end += 1;
                }

                let unknown_score = self.min_score - 10.0;
                lattice[char_end].push(LatticeNode {
                    start,
                    end: char_end,
                    token_id: self.config.unk_id,
                    score: unknown_score,
                    prev: Some(prev_idx),
                    best_score: best_prev_score + unknown_score,
                });
            }
        }

        lattice
    }

    /// Find best path through lattice using Viterbi
    fn viterbi(&self, lattice: &[Vec<LatticeNode>]) -> Vec<(u32, usize, usize)> {
        let n = lattice.len() - 1;
        if lattice[n].is_empty() {
            return vec![];
        }

        // Find best final node
        let mut best_final_idx = 0;
        let mut best_final_score = f64::NEG_INFINITY;

        for (idx, node) in lattice[n].iter().enumerate() {
            if node.best_score > best_final_score {
                best_final_score = node.best_score;
                best_final_idx = idx;
            }
        }

        // Backtrace
        let mut result = Vec::new();
        let mut current_pos = n;
        let mut current_idx = best_final_idx;

        while current_pos > 0 {
            let node = &lattice[current_pos][current_idx];
            if node.start != node.end {
                result.push((node.token_id, node.start, node.end));
            }

            current_pos = node.start;
            if let Some(prev_idx) = node.prev {
                current_idx = prev_idx.min(lattice[current_pos].len().saturating_sub(1));
            } else {
                break;
            }
        }

        result.reverse();
        result
    }

    /// Encode text using Viterbi algorithm (optimal segmentation)
    pub fn encode_viterbi(&self, text: &str) -> Vec<u32> {
        let lattice = self.build_lattice(text);
        self.viterbi(&lattice)
            .into_iter()
            .map(|(id, _, _)| id)
            .collect()
    }

    /// N-best decoding using A* search (3.2.4)
    pub fn encode_nbest(&self, text: &str, n: usize) -> Vec<Vec<u32>> {
        if n == 0 {
            return vec![];
        }

        let bytes = text.as_bytes();
        let text_len = bytes.len();

        // Priority queue for A* search
        let mut agenda: BinaryHeap<Hypothesis> = BinaryHeap::new();
        let mut completed: Vec<Vec<u32>> = Vec::with_capacity(n);

        // Start with empty hypothesis
        agenda.push(Hypothesis {
            position: 0,
            tokens: Vec::new(),
            score: 0.0,
        });

        // Max agenda size to prevent memory explosion
        let max_agenda = n * 1000;

        while let Some(hyp) = agenda.pop() {
            if hyp.position == text_len {
                completed.push(hyp.tokens);
                if completed.len() >= n {
                    break;
                }
                continue;
            }

            // Expand hypothesis
            for (token_len, token_id) in self.trie.common_prefix_search(bytes, hyp.position) {
                let new_pos = hyp.position + token_len;
                let score = self.get_score(token_id);

                let mut new_tokens = hyp.tokens.clone();
                new_tokens.push(token_id);

                agenda.push(Hypothesis {
                    position: new_pos,
                    tokens: new_tokens,
                    score: hyp.score + score,
                });
            }

            // Byte fallback
            if self.config.byte_fallback {
                let byte_val = bytes[hyp.position];
                if let Some(byte_id) = self.byte_tokens.get(byte_val) {
                    let score = self.min_score - 5.0;
                    let mut new_tokens = hyp.tokens.clone();
                    new_tokens.push(byte_id);

                    agenda.push(Hypothesis {
                        position: hyp.position + 1,
                        tokens: new_tokens,
                        score: hyp.score + score,
                    });
                }
            }

            // Limit agenda size
            if agenda.len() > max_agenda {
                // Keep only top candidates
                let mut temp: Vec<_> = agenda.drain().collect();
                temp.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
                temp.truncate(max_agenda / 2);
                agenda.extend(temp);
            }
        }

        completed
    }

    /// Stochastic sampling using forward-filtering backward-sampling (3.2.5)
    pub fn sample(&self, text: &str, temperature: f64) -> Vec<u32> {
        if temperature <= 0.0 {
            // Temperature 0 means deterministic (Viterbi)
            return self.encode_viterbi(text);
        }

        let bytes = text.as_bytes();
        let n = bytes.len();

        // Forward pass: compute alpha (log-sum-exp scores)
        let mut alpha: Vec<f64> = vec![f64::NEG_INFINITY; n + 1];
        alpha[0] = 0.0;

        // Store all possible transitions for backward sampling
        let mut transitions: Vec<Vec<(usize, u32, f64)>> = vec![Vec::new(); n + 1];

        for i in 0..n {
            if alpha[i] == f64::NEG_INFINITY {
                continue;
            }

            for (token_len, token_id) in self.trie.common_prefix_search(bytes, i) {
                let j = i + token_len;
                let score = self.get_score(token_id) / temperature;
                let new_score = alpha[i] + score;

                // Log-sum-exp update
                alpha[j] = log_sum_exp(alpha[j], new_score);
                transitions[j].push((i, token_id, score));
            }

            // Byte fallback
            if self.config.byte_fallback {
                let byte_val = bytes[i];
                if let Some(byte_id) = self.byte_tokens.get(byte_val) {
                    let score = (self.min_score - 5.0) / temperature;
                    let new_score = alpha[i] + score;
                    alpha[i + 1] = log_sum_exp(alpha[i + 1], new_score);
                    transitions[i + 1].push((i, byte_id, score));
                }
            }
        }

        // Backward sampling
        let mut tokens = Vec::new();
        let mut pos = n;

        while pos > 0 {
            let trans = &transitions[pos];
            if trans.is_empty() {
                // Fallback to unknown
                tokens.push(self.config.unk_id);
                pos = pos.saturating_sub(1);
                continue;
            }

            // Compute sampling probabilities
            let total = alpha[pos];
            let mut probs: Vec<f64> = trans
                .iter()
                .map(|(prev, _, score)| (alpha[*prev] + score - total).exp())
                .collect();

            // Normalize
            let sum: f64 = probs.iter().sum();
            if sum > 0.0 {
                for p in &mut probs {
                    *p /= sum;
                }
            } else {
                // Uniform fallback
                let uniform = 1.0 / probs.len() as f64;
                for p in &mut probs {
                    *p = uniform;
                }
            }

            // Sample
            let r: f64 = fastrand::f64();
            let mut cumsum = 0.0;
            let mut selected = 0;

            for (idx, p) in probs.iter().enumerate() {
                cumsum += p;
                if r < cumsum {
                    selected = idx;
                    break;
                }
            }

            let (prev_pos, token_id, _) = trans[selected];
            tokens.push(token_id);
            pos = prev_pos;
        }

        tokens.reverse();
        tokens
    }
}

/// Log-sum-exp for numerical stability
#[inline]
fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        b
    } else if b == f64::NEG_INFINITY {
        a
    } else if a > b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}

impl Tokenizer for UnigramTokenizer {
    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding> {
        let lattice = self.build_lattice(text);
        let tokens = self.viterbi(&lattice);

        let mut encoding = Encoding::with_capacity(tokens.len() + 2);

        // Add BOS if configured
        if add_special_tokens {
            if let Some(ref bos) = self.config.bos_token {
                if let Some(bos_id) = self.vocabulary.token_to_id(bos) {
                    encoding.push(bos_id, bos.clone(), (0, 0), None, Some(0), true);
                }
            }
        }

        // Add tokens
        for (word_idx, (token_id, start, end)) in tokens.iter().enumerate() {
            let token = self
                .vocabulary
                .id_to_token(*token_id)
                .unwrap_or(&self.config.unk_token)
                .to_string();

            encoding.push(
                *token_id,
                token,
                (*start, *end),
                Some(word_idx as u32),
                Some(0),
                false,
            );
        }

        // Add EOS if configured
        if add_special_tokens {
            if let Some(ref eos) = self.config.eos_token {
                if let Some(eos_id) = self.vocabulary.token_to_id(eos) {
                    let last_offset = encoding.get_offsets().last().map(|(_, e)| *e).unwrap_or(0);
                    encoding.push(
                        eos_id,
                        eos.clone(),
                        (last_offset, last_offset),
                        None,
                        Some(0),
                        true,
                    );
                }
            }
        }

        Ok(encoding)
    }

    fn encode_pair(
        &self,
        text: &str,
        text_pair: &str,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        let mut encoding = self.encode(text, false)?;
        let encoding_pair = self.encode(text_pair, false)?;

        encoding.merge(encoding_pair, true);

        if add_special_tokens {
            // Add BOS at start and EOS at end
        }

        Ok(encoding)
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut result = String::new();

        for &id in ids {
            if let Some(token) = self.vocabulary.id_to_token(id) {
                if skip_special_tokens && (token.starts_with('<') && token.ends_with('>')) {
                    continue;
                }

                // SentencePiece uses ▁ for word boundaries
                let token = token.replace('▁', " ");
                result.push_str(&token);
            }
        }

        // Trim leading space from ▁ replacement
        Ok(result.trim_start().to_string())
    }

    fn save(&self, _path: &Path) -> Result<()> {
        Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Saving not yet implemented",
        )))
    }
}

// ============================================================================
// Builder
// ============================================================================

/// Builder for Unigram tokenizer
pub struct UnigramBuilder {
    vocabulary: Option<Vocabulary>,
    pieces: Vec<UnigramPiece>,
    config: UnigramConfig,
}

impl UnigramBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            vocabulary: None,
            pieces: Vec::new(),
            config: UnigramConfig::default(),
        }
    }

    /// Set vocabulary
    pub fn vocabulary(mut self, vocab: Vocabulary) -> Self {
        self.vocabulary = Some(vocab);
        self
    }

    /// Add a piece
    pub fn add_piece(mut self, token: &str, score: f64) -> Self {
        self.pieces.push(UnigramPiece {
            token: token.to_string(),
            score,
        });
        self
    }

    /// Add multiple pieces
    pub fn add_pieces(mut self, pieces: impl IntoIterator<Item = (String, f64)>) -> Self {
        for (token, score) in pieces {
            self.pieces.push(UnigramPiece { token, score });
        }
        self
    }

    /// Set configuration
    pub fn config(mut self, config: UnigramConfig) -> Self {
        self.config = config;
        self
    }

    /// Enable byte fallback
    pub fn byte_fallback(mut self, enabled: bool) -> Self {
        self.config.byte_fallback = enabled;
        self
    }

    /// Build the tokenizer
    pub fn build(self) -> Result<UnigramTokenizer> {
        let vocabulary = self
            .vocabulary
            .ok_or_else(|| Error::invalid_config("Vocabulary is required"))?;

        Ok(UnigramTokenizer::new(vocabulary, self.pieces, self.config))
    }
}

impl Default for UnigramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::VocabularyBuilder;

    fn create_test_tokenizer() -> UnigramTokenizer {
        let vocab = VocabularyBuilder::new()
            .add_tokens(["<unk>", "▁hello", "▁world", "▁", "lo", "he", "l", "o", "h", "e", "w", "r", "d"])
            .unk_token("<unk>")
            .build();

        let pieces = vec![
            UnigramPiece { token: "<unk>".to_string(), score: -10.0 },
            UnigramPiece { token: "▁hello".to_string(), score: -2.0 },
            UnigramPiece { token: "▁world".to_string(), score: -2.0 },
            UnigramPiece { token: "▁".to_string(), score: -1.0 },
            UnigramPiece { token: "lo".to_string(), score: -3.0 },
            UnigramPiece { token: "he".to_string(), score: -3.0 },
            UnigramPiece { token: "l".to_string(), score: -4.0 },
            UnigramPiece { token: "o".to_string(), score: -4.0 },
            UnigramPiece { token: "h".to_string(), score: -4.0 },
            UnigramPiece { token: "e".to_string(), score: -4.0 },
            UnigramPiece { token: "w".to_string(), score: -4.0 },
            UnigramPiece { token: "r".to_string(), score: -4.0 },
            UnigramPiece { token: "d".to_string(), score: -4.0 },
        ];

        UnigramTokenizer::new(vocab, pieces, UnigramConfig::default())
    }

    #[test]
    fn test_basic_tokenization() {
        let tokenizer = create_test_tokenizer();
        let encoding = tokenizer.encode("hello world", false).unwrap();

        assert!(!encoding.is_empty());
    }

    #[test]
    fn test_viterbi() {
        let tokenizer = create_test_tokenizer();
        let tokens = tokenizer.encode_viterbi("hello");

        // Should find optimal segmentation
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_nbest() {
        let tokenizer = create_test_tokenizer();
        let nbest = tokenizer.encode_nbest("hello", 3);

        // Should return up to 3 segmentations
        assert!(!nbest.is_empty());
        assert!(nbest.len() <= 3);
    }

    #[test]
    fn test_sample() {
        let tokenizer = create_test_tokenizer();

        // With temperature 0, should be deterministic (same as Viterbi)
        let sample_t0 = tokenizer.sample("hello", 0.0);
        let viterbi = tokenizer.encode_viterbi("hello");
        assert_eq!(sample_t0, viterbi);

        // With temperature > 0, should produce valid tokenization
        let sample_t1 = tokenizer.sample("hello", 1.0);
        assert!(!sample_t1.is_empty());
    }

    #[test]
    fn test_byte_token() {
        assert_eq!(byte_token(0x00), "<0x00>");
        assert_eq!(byte_token(0xFF), "<0xFF>");
        assert_eq!(byte_token(0x41), "<0x41>"); // 'A'
    }

    #[test]
    fn test_builder() {
        let vocab = VocabularyBuilder::new()
            .add_tokens(["<unk>", "hello", "world"])
            .unk_token("<unk>")
            .build();

        let tokenizer = UnigramBuilder::new()
            .vocabulary(vocab)
            .add_piece("hello", -1.0)
            .add_piece("world", -2.0)
            .byte_fallback(false)
            .build()
            .unwrap();

        assert!(!tokenizer.pieces.is_empty());
    }

    #[test]
    fn test_log_sum_exp() {
        // log(e^0 + e^0) = log(2) ≈ 0.693
        let result = log_sum_exp(0.0, 0.0);
        assert!((result - 0.693).abs() < 0.01);

        // log(e^(-inf) + e^0) = 0
        let result = log_sum_exp(f64::NEG_INFINITY, 0.0);
        assert!((result - 0.0).abs() < 0.001);
    }
}
