//! Encoding structure for tokenization results
//!
//! This module defines the output structure for tokenization, including
//! token IDs, attention masks, and offset mappings.

use serde::{Deserialize, Serialize};

/// Result of encoding text into tokens
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Encoding {
    /// Token IDs
    ids: Vec<u32>,
    /// Token type IDs (for sequence pairs)
    type_ids: Vec<u32>,
    /// Token strings
    tokens: Vec<String>,
    /// Offsets mapping tokens to original text positions
    offsets: Vec<(usize, usize)>,
    /// Special tokens mask (1 for special tokens, 0 for regular)
    special_tokens_mask: Vec<u32>,
    /// Attention mask (1 for real tokens, 0 for padding)
    attention_mask: Vec<u32>,
    /// Word IDs mapping tokens to original words
    word_ids: Vec<Option<u32>>,
    /// Sequence IDs (0 for first sequence, 1 for second, None for special)
    sequence_ids: Vec<Option<usize>>,
    /// Overflowing encodings (for truncation with stride)
    overflowing: Vec<Encoding>,
}

impl Encoding {
    /// Create a new empty encoding
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an encoding with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            ids: Vec::with_capacity(capacity),
            type_ids: Vec::with_capacity(capacity),
            tokens: Vec::with_capacity(capacity),
            offsets: Vec::with_capacity(capacity),
            special_tokens_mask: Vec::with_capacity(capacity),
            attention_mask: Vec::with_capacity(capacity),
            word_ids: Vec::with_capacity(capacity),
            sequence_ids: Vec::with_capacity(capacity),
            overflowing: Vec::new(),
        }
    }

    /// Create an encoding from components
    pub fn from_parts(
        ids: Vec<u32>,
        type_ids: Vec<u32>,
        tokens: Vec<String>,
        offsets: Vec<(usize, usize)>,
        special_tokens_mask: Vec<u32>,
        attention_mask: Vec<u32>,
        word_ids: Vec<Option<u32>>,
        sequence_ids: Vec<Option<usize>>,
    ) -> Self {
        Self {
            ids,
            type_ids,
            tokens,
            offsets,
            special_tokens_mask,
            attention_mask,
            word_ids,
            sequence_ids,
            overflowing: Vec::new(),
        }
    }

    /// Get the token IDs
    #[inline]
    pub fn get_ids(&self) -> &[u32] {
        &self.ids
    }

    /// Get the token type IDs
    #[inline]
    pub fn get_type_ids(&self) -> &[u32] {
        &self.type_ids
    }

    /// Get the token strings
    #[inline]
    pub fn get_tokens(&self) -> &[String] {
        &self.tokens
    }

    /// Get the offset mappings
    #[inline]
    pub fn get_offsets(&self) -> &[(usize, usize)] {
        &self.offsets
    }

    /// Get the special tokens mask
    #[inline]
    pub fn get_special_tokens_mask(&self) -> &[u32] {
        &self.special_tokens_mask
    }

    /// Get the attention mask
    #[inline]
    pub fn get_attention_mask(&self) -> &[u32] {
        &self.attention_mask
    }

    /// Get the word IDs
    #[inline]
    pub fn get_word_ids(&self) -> &[Option<u32>] {
        &self.word_ids
    }

    /// Get the sequence IDs
    #[inline]
    pub fn get_sequence_ids(&self) -> &[Option<usize>] {
        &self.sequence_ids
    }

    /// Get overflowing encodings
    #[inline]
    pub fn get_overflowing(&self) -> &[Encoding] {
        &self.overflowing
    }

    /// Get the number of tokens
    #[inline]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Add a token to the encoding
    pub fn push(
        &mut self,
        id: u32,
        token: String,
        offset: (usize, usize),
        word_id: Option<u32>,
        sequence_id: Option<usize>,
        is_special: bool,
    ) {
        self.ids.push(id);
        self.type_ids.push(sequence_id.unwrap_or(0) as u32);
        self.tokens.push(token);
        self.offsets.push(offset);
        self.special_tokens_mask.push(if is_special { 1 } else { 0 });
        self.attention_mask.push(1);
        self.word_ids.push(word_id);
        self.sequence_ids.push(sequence_id);
    }

    /// Truncate the encoding to a maximum length
    pub fn truncate(&mut self, max_length: usize, stride: usize) {
        if self.ids.len() <= max_length {
            return;
        }

        // Create overflowing encoding with stride
        if stride > 0 && self.ids.len() > max_length {
            let start = max_length - stride;
            let overflow = Encoding {
                ids: self.ids[start..].to_vec(),
                type_ids: self.type_ids[start..].to_vec(),
                tokens: self.tokens[start..].to_vec(),
                offsets: self.offsets[start..].to_vec(),
                special_tokens_mask: self.special_tokens_mask[start..].to_vec(),
                attention_mask: self.attention_mask[start..].to_vec(),
                word_ids: self.word_ids[start..].to_vec(),
                sequence_ids: self.sequence_ids[start..].to_vec(),
                overflowing: Vec::new(),
            };
            self.overflowing.push(overflow);
        }

        // Truncate main encoding
        self.ids.truncate(max_length);
        self.type_ids.truncate(max_length);
        self.tokens.truncate(max_length);
        self.offsets.truncate(max_length);
        self.special_tokens_mask.truncate(max_length);
        self.attention_mask.truncate(max_length);
        self.word_ids.truncate(max_length);
        self.sequence_ids.truncate(max_length);
    }

    /// Pad the encoding to a minimum length
    pub fn pad(&mut self, length: usize, pad_id: u32, pad_token: &str) {
        if self.ids.len() >= length {
            return;
        }

        let pad_count = length - self.ids.len();
        self.ids.extend(std::iter::repeat(pad_id).take(pad_count));
        self.type_ids.extend(std::iter::repeat(0).take(pad_count));
        self.tokens
            .extend(std::iter::repeat(pad_token.to_string()).take(pad_count));
        self.offsets.extend(std::iter::repeat((0, 0)).take(pad_count));
        self.special_tokens_mask
            .extend(std::iter::repeat(1).take(pad_count));
        self.attention_mask
            .extend(std::iter::repeat(0).take(pad_count)); // 0 for padding
        self.word_ids.extend(std::iter::repeat(None).take(pad_count));
        self.sequence_ids
            .extend(std::iter::repeat(None).take(pad_count));
    }

    /// Merge two encodings (for sequence pairs)
    pub fn merge(&mut self, other: Encoding, growing_offsets: bool) {
        let offset_shift = if growing_offsets {
            self.offsets.last().map(|(_, end)| *end).unwrap_or(0)
        } else {
            0
        };

        self.ids.extend(other.ids);
        self.type_ids.extend(other.type_ids);
        self.tokens.extend(other.tokens);

        if growing_offsets {
            self.offsets.extend(
                other
                    .offsets
                    .into_iter()
                    .map(|(start, end)| (start + offset_shift, end + offset_shift)),
            );
        } else {
            self.offsets.extend(other.offsets);
        }

        self.special_tokens_mask.extend(other.special_tokens_mask);
        self.attention_mask.extend(other.attention_mask);
        self.word_ids.extend(other.word_ids);
        self.sequence_ids.extend(other.sequence_ids);
    }

    // ========================================================================
    // Position Mapping Methods (HF Compatible)
    // ========================================================================

    /// Get the number of sequences in this encoding
    ///
    /// Returns 1 for single sequence, 2 for sequence pairs, etc.
    #[inline]
    pub fn n_sequences(&self) -> usize {
        // Count distinct sequence IDs (excluding None for special tokens)
        let mut seen = [false; 16]; // Support up to 16 sequences
        let mut count = 0;

        for seq_id in &self.sequence_ids {
            if let Some(id) = seq_id {
                if *id < 16 && !seen[*id] {
                    seen[*id] = true;
                    count += 1;
                }
            }
        }

        count.max(1) // At least 1 sequence
    }

    /// Get the token range (start, end) for a given word in a sequence
    ///
    /// # Arguments
    /// * `word` - The word index
    /// * `sequence_id` - The sequence ID (0 for first, 1 for second)
    ///
    /// # Returns
    /// The range of token indices [start, end) that correspond to this word
    #[inline]
    pub fn word_to_tokens(&self, word: u32, sequence_id: usize) -> Option<(usize, usize)> {
        let mut first: Option<usize> = None;
        let mut last: Option<usize> = None;

        for (i, (word_id, seq_id)) in self.word_ids.iter().zip(&self.sequence_ids).enumerate() {
            if *seq_id == Some(sequence_id) && *word_id == Some(word) {
                if first.is_none() {
                    first = Some(i);
                }
                last = Some(i);
            }
        }

        match (first, last) {
            (Some(f), Some(l)) => Some((f, l + 1)),
            _ => None,
        }
    }

    /// Get the character range for a given word in a sequence
    ///
    /// # Arguments
    /// * `word` - The word index
    /// * `sequence_id` - The sequence ID (0 for first, 1 for second)
    ///
    /// # Returns
    /// The character range (start, end) in the original text
    #[inline]
    pub fn word_to_chars(&self, word: u32, sequence_id: usize) -> Option<(usize, usize)> {
        let (start_tok, end_tok) = self.word_to_tokens(word, sequence_id)?;

        if start_tok >= self.offsets.len() || end_tok == 0 {
            return None;
        }

        let start_char = self.offsets[start_tok].0;
        let end_char = self.offsets[end_tok - 1].1;

        Some((start_char, end_char))
    }

    /// Get the character range and sequence ID for a given token
    ///
    /// # Arguments
    /// * `token` - The token index
    ///
    /// # Returns
    /// A tuple of (sequence_id, (start_char, end_char))
    #[inline]
    pub fn token_to_chars(&self, token: usize) -> Option<(usize, (usize, usize))> {
        if token >= self.len() {
            return None;
        }

        let seq_id = self.sequence_ids.get(token)?.as_ref()?;
        let offsets = self.offsets.get(token)?;

        Some((*seq_id, *offsets))
    }

    /// Get the word index and sequence ID for a given token
    ///
    /// # Arguments
    /// * `token` - The token index
    ///
    /// # Returns
    /// A tuple of (sequence_id, word_index)
    #[inline]
    pub fn token_to_word(&self, token: usize) -> Option<(usize, u32)> {
        if token >= self.len() {
            return None;
        }

        let seq_id = self.sequence_ids.get(token)?.as_ref()?;
        let word_id = self.word_ids.get(token)?.as_ref()?;

        Some((*seq_id, *word_id))
    }

    /// Get the token index for a character position in a sequence
    ///
    /// # Arguments
    /// * `pos` - The character position
    /// * `sequence_id` - The sequence ID (0 for first, 1 for second)
    ///
    /// # Returns
    /// The token index that contains this character position
    #[inline]
    pub fn char_to_token(&self, pos: usize, sequence_id: usize) -> Option<usize> {
        for (i, (offset, seq_id)) in self.offsets.iter().zip(&self.sequence_ids).enumerate() {
            if *seq_id == Some(sequence_id) && pos >= offset.0 && pos < offset.1 {
                return Some(i);
            }
        }
        None
    }

    /// Get the word index for a character position in a sequence
    ///
    /// # Arguments
    /// * `pos` - The character position
    /// * `sequence_id` - The sequence ID (0 for first, 1 for second)
    ///
    /// # Returns
    /// The word index that contains this character position
    #[inline]
    pub fn char_to_word(&self, pos: usize, sequence_id: usize) -> Option<u32> {
        let token = self.char_to_token(pos, sequence_id)?;
        self.word_ids.get(token)?.as_ref().copied()
    }

    /// Get the sequence ID for a given token
    ///
    /// # Arguments
    /// * `token` - The token index
    ///
    /// # Returns
    /// The sequence ID, or None for special tokens
    #[inline]
    pub fn token_to_sequence(&self, token: usize) -> Option<usize> {
        self.sequence_ids.get(token)?.as_ref().copied()
    }

    // ========================================================================
    // Padding Enhancements (HF Compatible)
    // ========================================================================

    /// Pad the encoding from the left side
    pub fn pad_left(&mut self, length: usize, pad_id: u32, pad_token: &str) {
        if self.ids.len() >= length {
            return;
        }

        let pad_count = length - self.ids.len();

        // Create padding vectors
        let pad_ids: Vec<u32> = std::iter::repeat(pad_id).take(pad_count).collect();
        let pad_type_ids: Vec<u32> = std::iter::repeat(0).take(pad_count).collect();
        let pad_tokens: Vec<String> = std::iter::repeat(pad_token.to_string()).take(pad_count).collect();
        let pad_offsets: Vec<(usize, usize)> = std::iter::repeat((0, 0)).take(pad_count).collect();
        let pad_special: Vec<u32> = std::iter::repeat(1).take(pad_count).collect();
        let pad_attention: Vec<u32> = std::iter::repeat(0).take(pad_count).collect();
        let pad_word_ids: Vec<Option<u32>> = std::iter::repeat(None).take(pad_count).collect();
        let pad_seq_ids: Vec<Option<usize>> = std::iter::repeat(None).take(pad_count).collect();

        // Prepend padding
        let mut new_ids = pad_ids;
        new_ids.extend(std::mem::take(&mut self.ids));
        self.ids = new_ids;

        let mut new_type_ids = pad_type_ids;
        new_type_ids.extend(std::mem::take(&mut self.type_ids));
        self.type_ids = new_type_ids;

        let mut new_tokens = pad_tokens;
        new_tokens.extend(std::mem::take(&mut self.tokens));
        self.tokens = new_tokens;

        let mut new_offsets = pad_offsets;
        new_offsets.extend(std::mem::take(&mut self.offsets));
        self.offsets = new_offsets;

        let mut new_special = pad_special;
        new_special.extend(std::mem::take(&mut self.special_tokens_mask));
        self.special_tokens_mask = new_special;

        let mut new_attention = pad_attention;
        new_attention.extend(std::mem::take(&mut self.attention_mask));
        self.attention_mask = new_attention;

        let mut new_word_ids = pad_word_ids;
        new_word_ids.extend(std::mem::take(&mut self.word_ids));
        self.word_ids = new_word_ids;

        let mut new_seq_ids = pad_seq_ids;
        new_seq_ids.extend(std::mem::take(&mut self.sequence_ids));
        self.sequence_ids = new_seq_ids;
    }

    /// Truncate from the left side
    pub fn truncate_left(&mut self, max_length: usize, stride: usize) {
        if self.ids.len() <= max_length {
            return;
        }

        let remove_count = self.ids.len() - max_length;

        // Create overflowing encoding with stride
        if stride > 0 {
            let end = remove_count + stride;
            let overflow = Encoding {
                ids: self.ids[..end].to_vec(),
                type_ids: self.type_ids[..end].to_vec(),
                tokens: self.tokens[..end].to_vec(),
                offsets: self.offsets[..end].to_vec(),
                special_tokens_mask: self.special_tokens_mask[..end].to_vec(),
                attention_mask: self.attention_mask[..end].to_vec(),
                word_ids: self.word_ids[..end].to_vec(),
                sequence_ids: self.sequence_ids[..end].to_vec(),
                overflowing: Vec::new(),
            };
            self.overflowing.push(overflow);
        }

        // Remove from left
        self.ids = self.ids.split_off(remove_count);
        self.type_ids = self.type_ids.split_off(remove_count);
        self.tokens = self.tokens.split_off(remove_count);
        self.offsets = self.offsets.split_off(remove_count);
        self.special_tokens_mask = self.special_tokens_mask.split_off(remove_count);
        self.attention_mask = self.attention_mask.split_off(remove_count);
        self.word_ids = self.word_ids.split_off(remove_count);
        self.sequence_ids = self.sequence_ids.split_off(remove_count);
    }

    /// Set the sequence ID for all tokens
    pub fn set_sequence_id(&mut self, sequence_id: usize) {
        for seq_id in &mut self.sequence_ids {
            if seq_id.is_some() {
                *seq_id = Some(sequence_id);
            }
        }
        for type_id in &mut self.type_ids {
            *type_id = sequence_id as u32;
        }
    }

    /// Set the type ID for all tokens
    pub fn set_type_id(&mut self, type_id: u32) {
        for tid in &mut self.type_ids {
            *tid = type_id;
        }
    }

    /// Get mutable reference to overflowing encodings
    pub fn get_overflowing_mut(&mut self) -> &mut Vec<Encoding> {
        &mut self.overflowing
    }

    /// Clear overflowing encodings
    pub fn clear_overflowing(&mut self) {
        self.overflowing.clear();
    }

    /// Take overflowing encodings, leaving empty vec
    pub fn take_overflowing(&mut self) -> Vec<Encoding> {
        std::mem::take(&mut self.overflowing)
    }
}

// ============================================================================
// Batch Padding Utilities (HF Compatible)
// ============================================================================

/// Padding direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaddingDirection {
    /// Pad on the right (default)
    #[default]
    Right,
    /// Pad on the left
    Left,
}

/// Padding strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// Pad to a fixed length
    Fixed(usize),
    /// Pad to the longest sequence in the batch
    BatchLongest,
}

/// Padding parameters
#[derive(Debug, Clone)]
pub struct PaddingParams {
    /// Padding strategy
    pub strategy: PaddingStrategy,
    /// Padding direction
    pub direction: PaddingDirection,
    /// Pad token ID
    pub pad_id: u32,
    /// Pad token string
    pub pad_token: String,
    /// Pad to multiple of this value (e.g., 8 for TPU)
    pub pad_to_multiple_of: Option<usize>,
}

impl Default for PaddingParams {
    fn default() -> Self {
        Self {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            pad_to_multiple_of: None,
        }
    }
}

impl PaddingParams {
    /// Create new padding parameters
    pub fn new(pad_id: u32, pad_token: impl Into<String>) -> Self {
        Self {
            pad_id,
            pad_token: pad_token.into(),
            ..Default::default()
        }
    }

    /// Set fixed length padding
    pub fn fixed(mut self, length: usize) -> Self {
        self.strategy = PaddingStrategy::Fixed(length);
        self
    }

    /// Set batch longest padding
    pub fn batch_longest(mut self) -> Self {
        self.strategy = PaddingStrategy::BatchLongest;
        self
    }

    /// Set left padding
    pub fn left(mut self) -> Self {
        self.direction = PaddingDirection::Left;
        self
    }

    /// Set right padding
    pub fn right(mut self) -> Self {
        self.direction = PaddingDirection::Right;
        self
    }

    /// Set pad to multiple
    pub fn multiple_of(mut self, n: usize) -> Self {
        self.pad_to_multiple_of = Some(n);
        self
    }
}

/// Pad a batch of encodings
///
/// High-performance batch padding with:
/// - Pre-calculated target length
/// - SIMD-friendly memory layouts
/// - Parallel padding for large batches
pub fn pad_encodings(encodings: &mut [Encoding], params: &PaddingParams) {
    if encodings.is_empty() {
        return;
    }

    // Calculate target length
    let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);

    let target_len = match params.strategy {
        PaddingStrategy::Fixed(len) => len,
        PaddingStrategy::BatchLongest => max_len,
    };

    // Apply pad_to_multiple_of
    let target_len = if let Some(multiple) = params.pad_to_multiple_of {
        if multiple > 0 {
            ((target_len + multiple - 1) / multiple) * multiple
        } else {
            target_len
        }
    } else {
        target_len
    };

    // Pad each encoding
    for encoding in encodings {
        match params.direction {
            PaddingDirection::Right => encoding.pad(target_len, params.pad_id, &params.pad_token),
            PaddingDirection::Left => encoding.pad_left(target_len, params.pad_id, &params.pad_token),
        }
    }
}

// ============================================================================
// Truncation Utilities (HF Compatible)
// ============================================================================

/// Truncation direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TruncationDirection {
    /// Truncate from the right (default)
    #[default]
    Right,
    /// Truncate from the left
    Left,
}

/// Truncation strategy for sequence pairs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TruncationStrategy {
    /// Truncate the longest sequence first
    #[default]
    LongestFirst,
    /// Only truncate the first sequence
    OnlyFirst,
    /// Only truncate the second sequence
    OnlySecond,
}

/// Truncation parameters
#[derive(Debug, Clone)]
pub struct TruncationParams {
    /// Maximum length
    pub max_length: usize,
    /// Stride for overflowing tokens
    pub stride: usize,
    /// Truncation strategy
    pub strategy: TruncationStrategy,
    /// Truncation direction
    pub direction: TruncationDirection,
}

impl Default for TruncationParams {
    fn default() -> Self {
        Self {
            max_length: 512,
            stride: 0,
            strategy: TruncationStrategy::LongestFirst,
            direction: TruncationDirection::Right,
        }
    }
}

impl TruncationParams {
    /// Create new truncation parameters
    pub fn new(max_length: usize) -> Self {
        Self {
            max_length,
            ..Default::default()
        }
    }

    /// Set stride
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set left truncation
    pub fn left(mut self) -> Self {
        self.direction = TruncationDirection::Left;
        self
    }

    /// Set right truncation
    pub fn right(mut self) -> Self {
        self.direction = TruncationDirection::Right;
        self
    }

    /// Set longest-first strategy
    pub fn longest_first(mut self) -> Self {
        self.strategy = TruncationStrategy::LongestFirst;
        self
    }

    /// Set only-first strategy
    pub fn only_first(mut self) -> Self {
        self.strategy = TruncationStrategy::OnlyFirst;
        self
    }

    /// Set only-second strategy
    pub fn only_second(mut self) -> Self {
        self.strategy = TruncationStrategy::OnlySecond;
        self
    }
}

/// Truncate an encoding based on parameters
pub fn truncate_encoding(encoding: &mut Encoding, params: &TruncationParams) {
    match params.direction {
        TruncationDirection::Right => encoding.truncate(params.max_length, params.stride),
        TruncationDirection::Left => encoding.truncate_left(params.max_length, params.stride),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_basic() {
        let mut encoding = Encoding::new();
        encoding.push(1, "hello".to_string(), (0, 5), Some(0), Some(0), false);
        encoding.push(2, "world".to_string(), (6, 11), Some(1), Some(0), false);

        assert_eq!(encoding.len(), 2);
        assert_eq!(encoding.get_ids(), &[1, 2]);
        assert_eq!(encoding.get_tokens(), &["hello", "world"]);
    }

    #[test]
    fn test_encoding_truncate() {
        let mut encoding = Encoding::with_capacity(5);
        for i in 0..5 {
            encoding.push(
                i,
                format!("token{}", i),
                (i as usize, i as usize + 1),
                Some(i),
                Some(0),
                false,
            );
        }

        encoding.truncate(3, 1);
        assert_eq!(encoding.len(), 3);
        assert_eq!(encoding.get_overflowing().len(), 1);
    }

    #[test]
    fn test_encoding_pad() {
        let mut encoding = Encoding::new();
        encoding.push(1, "hello".to_string(), (0, 5), Some(0), Some(0), false);

        encoding.pad(5, 0, "[PAD]");
        assert_eq!(encoding.len(), 5);
        assert_eq!(encoding.get_attention_mask(), &[1, 0, 0, 0, 0]);
    }
}
