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
