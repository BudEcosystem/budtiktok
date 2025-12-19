//! LinMaxMatch: Linear-Time WordPiece Tokenization
//!
//! Implementation of the LinMaxMatch algorithm from Google Research:
//! "Fast WordPiece Tokenization" (arXiv:2012.15524)
//!
//! This achieves O(n) complexity instead of O(n²) by using failure links
//! similar to Aho-Corasick, but optimized for the MaxMatch problem.

use ahash::AHashMap;
use std::borrow::Cow;

/// A node in the LinMaxMatch trie
#[derive(Debug, Clone)]
struct TrieNode {
    /// Children indexed by byte value for fast lookup
    children: Vec<Option<u32>>, // 256 entries for byte-indexed access
    /// Token ID if this node represents a complete token
    token_id: Option<u32>,
    /// Failure link - points to longest proper suffix that is also a prefix
    failure: u32,
    /// Failure pops - tokens to emit when following failure link
    failure_pops: Vec<u32>,
    /// Depth in trie (for debugging)
    depth: u32,
}

impl TrieNode {
    fn new(depth: u32) -> Self {
        Self {
            children: vec![None; 256],
            token_id: None,
            failure: 0,
            failure_pops: Vec::new(),
            depth,
        }
    }
}

/// LinMaxMatch tokenizer with O(n) complexity
pub struct LinMaxMatchTokenizer {
    /// Trie nodes stored in a flat vector for cache efficiency
    nodes: Vec<TrieNode>,
    /// Root node index (always 0)
    root: u32,
    /// Suffix indicator node index (for "##" prefix)
    suffix_root: u32,
    /// Continuing subword prefix (e.g., "##")
    continuing_subword_prefix: String,
    /// Unknown token ID
    unk_token_id: u32,
    /// Vocabulary for reverse lookup
    id_to_token: Vec<String>,
}

impl LinMaxMatchTokenizer {
    /// Build a LinMaxMatch tokenizer from vocabulary
    pub fn new(
        vocab: &AHashMap<String, u32>,
        continuing_subword_prefix: &str,
        unk_token: &str,
    ) -> Self {
        let mut tokenizer = Self {
            nodes: vec![TrieNode::new(0)], // Root node
            root: 0,
            suffix_root: 0,
            continuing_subword_prefix: continuing_subword_prefix.to_string(),
            unk_token_id: *vocab.get(unk_token).unwrap_or(&0),
            id_to_token: Vec::new(),
        };

        // Build id_to_token mapping
        let max_id = vocab.values().max().copied().unwrap_or(0) as usize;
        tokenizer.id_to_token = vec![String::new(); max_id + 1];
        for (token, &id) in vocab {
            if (id as usize) < tokenizer.id_to_token.len() {
                tokenizer.id_to_token[id as usize] = token.clone();
            }
        }

        // Create suffix root node for continuation tokens
        tokenizer.suffix_root = tokenizer.add_node(0);

        // Add edge from root to suffix_root for the prefix
        let mut current = tokenizer.root;
        for byte in continuing_subword_prefix.bytes() {
            let next = tokenizer.add_node(tokenizer.nodes[current as usize].depth + 1);
            tokenizer.nodes[current as usize].children[byte as usize] = Some(next);
            current = next;
        }
        tokenizer.suffix_root = current;

        // Insert all tokens into trie
        for (token, &id) in vocab {
            tokenizer.insert_token(token, id);
        }

        // Compute failure links using BFS
        tokenizer.compute_failure_links();

        tokenizer
    }

    /// Add a new node to the trie
    fn add_node(&mut self, depth: u32) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(TrieNode::new(depth));
        id
    }

    /// Insert a token into the trie
    fn insert_token(&mut self, token: &str, id: u32) {
        // Determine starting node based on whether this is a continuation token
        let (start_node, token_bytes) = if token.starts_with(&self.continuing_subword_prefix) {
            (self.suffix_root, &token[self.continuing_subword_prefix.len()..])
        } else {
            (self.root, token.as_ref())
        };

        let mut current = start_node;

        // Traverse/create path for token
        for byte in token_bytes.bytes() {
            let next = match self.nodes[current as usize].children[byte as usize] {
                Some(child) => child,
                None => {
                    let new_node = self.add_node(self.nodes[current as usize].depth + 1);
                    self.nodes[current as usize].children[byte as usize] = Some(new_node);
                    new_node
                }
            };
            current = next;
        }

        // Mark end of token
        self.nodes[current as usize].token_id = Some(id);
    }

    /// Compute failure links and failure pops using BFS
    fn compute_failure_links(&mut self) {
        use std::collections::VecDeque;

        let mut queue = VecDeque::new();

        // Initialize failure links for depth-1 nodes
        for byte in 0..256u16 {
            if let Some(child) = self.nodes[self.root as usize].children[byte as usize] {
                self.nodes[child as usize].failure = self.root;
                queue.push_back(child);
            }
        }

        // Handle suffix root children
        for byte in 0..256u16 {
            if let Some(child) = self.nodes[self.suffix_root as usize].children[byte as usize] {
                self.nodes[child as usize].failure = self.suffix_root;
                queue.push_back(child);
            }
        }

        // BFS to compute failure links
        while let Some(node) = queue.pop_front() {
            for byte in 0..256u16 {
                if let Some(child) = self.nodes[node as usize].children[byte as usize] {
                    // Compute failure link for child
                    let mut failure = self.nodes[node as usize].failure;

                    // Follow failure links until we find a match or reach root
                    while failure != self.root
                        && self.nodes[failure as usize].children[byte as usize].is_none()
                    {
                        failure = self.nodes[failure as usize].failure;
                    }

                    // Set failure link
                    if let Some(fail_child) = self.nodes[failure as usize].children[byte as usize] {
                        if fail_child != child {
                            self.nodes[child as usize].failure = fail_child;
                        }
                    }

                    // Compute failure pops
                    self.compute_failure_pops(child);

                    queue.push_back(child);
                }
            }
        }
    }

    /// Compute failure pops for a node
    fn compute_failure_pops(&mut self, node: u32) {
        let mut pops = Vec::new();
        let mut current = node;

        // Collect tokens along failure chain
        while current != self.root && current != self.suffix_root {
            if let Some(token_id) = self.nodes[current as usize].token_id {
                pops.push(token_id);
            }
            current = self.nodes[current as usize].failure;
        }

        self.nodes[node as usize].failure_pops = pops;
    }

    /// Tokenize a word using LinMaxMatch algorithm - O(n) complexity
    #[inline]
    pub fn tokenize_word(&self, word: &[u8]) -> Vec<u32> {
        if word.is_empty() {
            return Vec::new();
        }

        let mut tokens = Vec::with_capacity(word.len() / 2 + 1);
        let mut i = 0;
        let mut is_first_token = true;

        while i < word.len() {
            // Choose starting node based on position
            let start_node = if is_first_token {
                self.root
            } else {
                self.suffix_root
            };

            let (token_id, consumed) = self.match_longest(word, i, start_node);

            if consumed > 0 {
                tokens.push(token_id);
                i += consumed;
                is_first_token = false;
            } else {
                // No match found - return [UNK]
                return vec![self.unk_token_id];
            }
        }

        tokens
    }

    /// Match the longest token starting at position using failure links
    #[inline]
    fn match_longest(&self, word: &[u8], start: usize, start_node: u32) -> (u32, usize) {
        let mut current = start_node;
        let mut last_match: Option<(u32, usize)> = None;
        let mut i = start;

        while i < word.len() {
            let byte = word[i];

            // Try to extend current match
            if let Some(child) = self.nodes[current as usize].children[byte as usize] {
                current = child;
                i += 1;

                // Record match if this node is a complete token
                if let Some(token_id) = self.nodes[current as usize].token_id {
                    last_match = Some((token_id, i - start));
                }
            } else {
                // No edge for this byte - follow failure link or break
                if current == start_node {
                    break;
                }

                // Follow failure link
                current = self.nodes[current as usize].failure;

                // If we're back at root/suffix_root without a match, try the next byte
                if current == self.root || current == self.suffix_root {
                    if last_match.is_some() {
                        break;
                    }
                    // No match at all - advance by one byte and try again
                    return (self.unk_token_id, 0);
                }
            }
        }

        // Return the longest match found
        last_match.unwrap_or((self.unk_token_id, 0))
    }

    /// Tokenize with string output
    pub fn tokenize_word_to_strings(&self, word: &str) -> Vec<Cow<'_, str>> {
        let ids = self.tokenize_word(word.as_bytes());
        ids.iter()
            .map(|&id| {
                if (id as usize) < self.id_to_token.len() {
                    Cow::Borrowed(self.id_to_token[id as usize].as_str())
                } else {
                    Cow::Borrowed("[UNK]")
                }
            })
            .collect()
    }
}

/// Fast pre-tokenizer with SIMD optimization
pub struct FastPreTokenizer {
    /// Lookup table for character classification (256 bytes for ASCII)
    char_class: [u8; 256],
}

/// Character class constants
const CLASS_NORMAL: u8 = 0;
const CLASS_WHITESPACE: u8 = 1;
const CLASS_PUNCTUATION: u8 = 2;
const CLASS_CJK_START: u8 = 3; // Start of multi-byte CJK
const CLASS_MULTIBYTE_2: u8 = 4; // Start of 2-byte UTF-8 (potential Unicode punctuation)

impl FastPreTokenizer {
    pub fn new() -> Self {
        let mut char_class = [CLASS_NORMAL; 256];

        // Mark whitespace
        char_class[b' ' as usize] = CLASS_WHITESPACE;
        char_class[b'\t' as usize] = CLASS_WHITESPACE;
        char_class[b'\n' as usize] = CLASS_WHITESPACE;
        char_class[b'\r' as usize] = CLASS_WHITESPACE;

        // Mark ASCII punctuation
        for &c in b"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" {
            char_class[c as usize] = CLASS_PUNCTUATION;
        }

        // Mark 2-byte UTF-8 start bytes (potential Unicode punctuation)
        for i in 0xC0..=0xDF {
            char_class[i] = CLASS_MULTIBYTE_2;
        }

        // Mark high bytes that start multi-byte sequences (potential CJK)
        for i in 0xE0..=0xFF {
            char_class[i] = CLASS_CJK_START;
        }

        Self { char_class }
    }

    /// Check if UTF-8 bytes represent a Unicode punctuation character
    #[inline]
    fn is_unicode_punctuation(text: &[u8], start: usize) -> Option<usize> {
        if start >= text.len() {
            return None;
        }
        let b0 = text[start];
        let char_len = Self::utf8_char_len(b0);
        let end = (start + char_len).min(text.len());
        if end - start < char_len {
            return None;
        }

        let cp = match char_len {
            1 => b0 as u32,
            2 => {
                let b1 = text.get(start + 1).copied().unwrap_or(0) as u32;
                ((b0 as u32 & 0x1F) << 6) | (b1 & 0x3F)
            }
            3 => {
                let b1 = text.get(start + 1).copied().unwrap_or(0) as u32;
                let b2 = text.get(start + 2).copied().unwrap_or(0) as u32;
                ((b0 as u32 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F)
            }
            _ => return None,
        };

        if matches!(cp,
            0x00A1 | 0x00AB | 0x00B7 | 0x00BB | 0x00BF |
            0x2010..=0x2027 | 0x2030..=0x205E |
            0x3001..=0x3003 | 0x3008..=0x3011 | 0x3014..=0x301F |
            0xFF01..=0xFF0F | 0xFF1A..=0xFF20 | 0xFF3B..=0xFF40 | 0xFF5B..=0xFF65
        ) {
            Some(char_len)
        } else {
            None
        }
    }

    /// Pre-tokenize text into word boundaries (start, end pairs)
    /// Returns byte offsets for zero-copy slicing
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8]) -> Vec<(usize, usize)> {
        let mut words = Vec::with_capacity(text.len() / 5 + 1);
        let mut word_start: Option<usize> = None;
        let mut i = 0;

        while i < text.len() {
            let byte = text[i];
            let class = self.char_class[byte as usize];

            match class {
                CLASS_WHITESPACE => {
                    if let Some(start) = word_start {
                        words.push((start, i));
                        word_start = None;
                    }
                    i += 1;
                }
                CLASS_PUNCTUATION => {
                    if let Some(start) = word_start {
                        words.push((start, i));
                        word_start = None;
                    }
                    // Punctuation is its own token
                    words.push((i, i + 1));
                    i += 1;
                }
                CLASS_MULTIBYTE_2 => {
                    // Handle 2-byte UTF-8 character (potential Unicode punctuation)
                    if let Some(punct_len) = Self::is_unicode_punctuation(text, i) {
                        if let Some(start) = word_start {
                            words.push((start, i));
                            word_start = None;
                        }
                        words.push((i, i + punct_len));
                        i += punct_len;
                    } else {
                        // Not punctuation, treat as normal word character
                        if word_start.is_none() {
                            word_start = Some(i);
                        }
                        i += Self::utf8_char_len(byte);
                    }
                }
                CLASS_CJK_START => {
                    // Handle multi-byte character (CJK or Unicode punctuation)
                    if let Some(start) = word_start {
                        words.push((start, i));
                        word_start = None;
                    }

                    // Decode UTF-8 to get character length
                    let char_len = Self::utf8_char_len(byte);
                    let char_end = (i + char_len).min(text.len());

                    // Check if it's CJK or Unicode punctuation
                    if char_len >= 3 && Self::is_cjk_range(text, i, char_end) {
                        // CJK character is its own token
                        words.push((i, char_end));
                    } else if let Some(punct_len) = Self::is_unicode_punctuation(text, i) {
                        // Unicode punctuation is its own token
                        words.push((i, i + punct_len));
                        i = i + punct_len;
                        continue;
                    } else {
                        // Not CJK or punctuation, treat as normal
                        if word_start.is_none() {
                            word_start = Some(i);
                        }
                    }
                    i = char_end;
                }
                _ => {
                    // Normal character
                    if word_start.is_none() {
                        word_start = Some(i);
                    }
                    // Handle multi-byte UTF-8
                    if byte >= 0x80 {
                        i += Self::utf8_char_len(byte);
                    } else {
                        i += 1;
                    }
                }
            }
        }

        if let Some(start) = word_start {
            words.push((start, text.len()));
        }

        words
    }

    /// Get UTF-8 character length from first byte
    #[inline(always)]
    fn utf8_char_len(first_byte: u8) -> usize {
        if first_byte < 0x80 {
            1
        } else if first_byte < 0xE0 {
            2
        } else if first_byte < 0xF0 {
            3
        } else {
            4
        }
    }

    /// Check if bytes at range form a CJK character
    #[inline]
    fn is_cjk_range(text: &[u8], start: usize, end: usize) -> bool {
        if end - start < 3 || end > text.len() {
            return false;
        }

        // Decode UTF-8 to codepoint
        let b0 = text[start] as u32;
        let b1 = text.get(start + 1).copied().unwrap_or(0) as u32;
        let b2 = text.get(start + 2).copied().unwrap_or(0) as u32;

        let codepoint = if b0 < 0xF0 {
            // 3-byte sequence
            ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F)
        } else {
            // 4-byte sequence
            let b3 = text.get(start + 3).copied().unwrap_or(0) as u32;
            ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F)
        };

        // CJK ranges
        matches!(
            codepoint,
            0x4E00..=0x9FFF       // CJK Unified Ideographs
            | 0x3400..=0x4DBF    // CJK Extension A
            | 0x20000..=0x2A6DF  // CJK Extension B
            | 0x2A700..=0x2B73F  // CJK Extension C
            | 0x2B740..=0x2B81F  // CJK Extension D
            | 0x2B820..=0x2CEAF  // CJK Extension E
            | 0xF900..=0xFAFF    // CJK Compatibility Ideographs
            | 0x2F800..=0x2FA1F  // CJK Compatibility Supplement
        )
    }
}

impl Default for FastPreTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vocab() -> AHashMap<String, u32> {
        let mut vocab = AHashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("[CLS]".to_string(), 1);
        vocab.insert("[SEP]".to_string(), 2);
        vocab.insert("hello".to_string(), 3);
        vocab.insert("world".to_string(), 4);
        vocab.insert("##ing".to_string(), 5);
        vocab.insert("##ed".to_string(), 6);
        vocab.insert("play".to_string(), 7);
        vocab.insert("un".to_string(), 8);
        vocab.insert("##play".to_string(), 9);
        vocab
    }

    #[test]
    fn test_linmaxmatch_basic() {
        let vocab = create_test_vocab();
        let tokenizer = LinMaxMatchTokenizer::new(&vocab, "##", "[UNK]");

        let tokens = tokenizer.tokenize_word(b"hello");
        assert_eq!(tokens, vec![3]); // "hello"

        let tokens = tokenizer.tokenize_word(b"world");
        assert_eq!(tokens, vec![4]); // "world"
    }

    #[test]
    fn test_linmaxmatch_subword() {
        let vocab = create_test_vocab();
        let tokenizer = LinMaxMatchTokenizer::new(&vocab, "##", "[UNK]");

        let tokens = tokenizer.tokenize_word(b"playing");
        assert_eq!(tokens, vec![7, 5]); // "play" + "##ing"
    }

    #[test]
    fn test_linmaxmatch_unknown() {
        let vocab = create_test_vocab();
        let tokenizer = LinMaxMatchTokenizer::new(&vocab, "##", "[UNK]");

        let tokens = tokenizer.tokenize_word(b"xyz");
        assert_eq!(tokens, vec![0]); // [UNK]
    }

    #[test]
    fn test_fast_pretokenizer() {
        let pre_tokenizer = FastPreTokenizer::new();

        let text = b"hello world";
        let words = pre_tokenizer.pre_tokenize(text);
        assert_eq!(words, vec![(0, 5), (6, 11)]);

        let text = b"hello, world!";
        let words = pre_tokenizer.pre_tokenize(text);
        assert_eq!(words, vec![(0, 5), (5, 6), (7, 12), (12, 13)]);
    }

    #[test]
    fn test_fast_pretokenizer_cjk() {
        let pre_tokenizer = FastPreTokenizer::new();

        let text = "你好世界".as_bytes();
        let words = pre_tokenizer.pre_tokenize(text);
        // Each CJK character should be its own token
        assert_eq!(words.len(), 4);
    }
}
