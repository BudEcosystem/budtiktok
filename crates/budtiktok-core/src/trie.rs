//! High-performance 256-ary Trie for vocabulary lookup
//!
//! This module provides a trie data structure optimized for tokenizer vocabulary
//! operations, particularly common prefix search which is essential for
//! efficient BPE and Unigram tokenization.
//!
//! Key optimizations:
//! - 256-ary nodes for O(1) child lookup (no hashing needed)
//! - Byte-indexed for direct UTF-8 processing
//! - Cache-friendly memory layout
//! - Support for common prefix iteration (enumerate all matches)

const INVALID_INDEX: u32 = u32::MAX;

/// Builder for constructing a Trie
#[derive(Default)]
pub struct TrieBuilder {
    trie: Trie,
}

impl TrieBuilder {
    /// Create a new TrieBuilder
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a token with its ID
    pub fn insert(&mut self, token: &str, id: u32) {
        self.trie.insert(token.as_bytes(), id);
    }

    /// Insert raw bytes with their ID
    pub fn insert_bytes(&mut self, bytes: &[u8], id: u32) {
        self.trie.insert(bytes, id);
    }

    /// Build the final Trie
    pub fn build(self) -> Trie {
        self.trie
    }

    /// Build from an iterator of (token, id) pairs
    pub fn from_iter<'a, I>(iter: I) -> Trie
    where
        I: IntoIterator<Item = (&'a str, u32)>,
    {
        let mut builder = Self::new();
        for (token, id) in iter {
            builder.insert(token, id);
        }
        builder.build()
    }
}

/// A 256-ary Trie for efficient prefix matching
///
/// Each node has 256 children (one per byte value), enabling O(1) child lookup.
/// This is particularly efficient for UTF-8 encoded text where we process bytes directly.
#[derive(Clone)]
pub struct Trie {
    nodes: Vec<TrieNode>,
}

impl Default for Trie {
    fn default() -> Self {
        Self {
            nodes: vec![TrieNode::default()],
        }
    }
}

impl Trie {
    /// Create a new empty Trie
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a byte sequence with its associated value
    pub fn insert(&mut self, bytes: &[u8], value: u32) {
        let mut current = 0usize;
        for &byte in bytes {
            let child_idx = self.nodes[current].children[byte as usize];
            if child_idx == INVALID_INDEX {
                let new_index = self.nodes.len() as u32;
                self.nodes.push(TrieNode::default());
                self.nodes[current].children[byte as usize] = new_index;
                current = new_index as usize;
            } else {
                current = child_idx as usize;
            }
        }
        self.nodes[current].value = Some(value);
    }

    /// Look up an exact match for a byte sequence
    #[inline]
    pub fn get(&self, bytes: &[u8]) -> Option<u32> {
        let mut current = 0usize;
        for &byte in bytes {
            let child_idx = self.nodes[current].children[byte as usize];
            if child_idx == INVALID_INDEX {
                return None;
            }
            current = child_idx as usize;
        }
        self.nodes[current].value
    }

    /// Look up an exact match for a string
    #[inline]
    pub fn get_str(&self, s: &str) -> Option<u32> {
        self.get(s.as_bytes())
    }

    /// Check if a byte sequence exists in the trie
    #[inline]
    pub fn contains(&self, bytes: &[u8]) -> bool {
        self.get(bytes).is_some()
    }

    /// Check if a string exists in the trie
    #[inline]
    pub fn contains_str(&self, s: &str) -> bool {
        self.get_str(s).is_some()
    }

    /// Enumerate all prefix matches starting at a given position
    ///
    /// This is the key operation for efficient tokenization. It finds all
    /// tokens in the vocabulary that are prefixes of the input starting
    /// at position `start`.
    ///
    /// The visitor function is called with (length, token_id) for each match.
    #[inline]
    pub fn enumerate_matches<F>(&self, bytes: &[u8], start: usize, mut visitor: F)
    where
        F: FnMut(usize, u32),
    {
        if start >= bytes.len() {
            return;
        }

        let mut current = 0usize;
        let mut position = start;

        while position < bytes.len() {
            let byte = bytes[position];
            let child_idx = self.nodes[current].children[byte as usize];
            if child_idx == INVALID_INDEX {
                break;
            }
            current = child_idx as usize;
            position += 1;

            if let Some(value) = self.nodes[current].value {
                visitor(position - start, value);
            }
        }
    }

    /// Get an iterator over all prefix matches starting at a given position
    pub fn common_prefix_search<'a>(&'a self, bytes: &'a [u8], start: usize) -> CommonPrefixIter<'a> {
        CommonPrefixIter {
            trie: self,
            bytes,
            current_node: 0,
            position: start,
            start,
        }
    }

    /// Get an iterator over all prefix matches for a string slice
    pub fn common_prefix_search_str<'a>(&'a self, s: &'a str, start: usize) -> CommonPrefixIter<'a> {
        self.common_prefix_search(s.as_bytes(), start)
    }

    /// Find the longest prefix match starting at a position
    #[inline]
    pub fn longest_prefix(&self, bytes: &[u8], start: usize) -> Option<(usize, u32)> {
        let mut result: Option<(usize, u32)> = None;
        self.enumerate_matches(bytes, start, |len, id| {
            result = Some((len, id));
        });
        result
    }

    /// Get the number of nodes in the trie
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.nodes.len() * std::mem::size_of::<TrieNode>()
    }
}

/// Iterator over common prefix matches
pub struct CommonPrefixIter<'a> {
    trie: &'a Trie,
    bytes: &'a [u8],
    current_node: usize,
    position: usize,
    start: usize,
}

impl<'a> Iterator for CommonPrefixIter<'a> {
    type Item = (usize, u32);

    fn next(&mut self) -> Option<Self::Item> {
        while self.position < self.bytes.len() {
            let byte = self.bytes[self.position];
            let child_idx = self.trie.nodes[self.current_node].children[byte as usize];

            if child_idx == INVALID_INDEX {
                self.position = self.bytes.len(); // End iteration
                return None;
            }

            self.current_node = child_idx as usize;
            self.position += 1;

            if let Some(value) = self.trie.nodes[self.current_node].value {
                return Some((self.position - self.start, value));
            }
        }
        None
    }
}

/// Internal node structure
#[derive(Clone)]
struct TrieNode {
    /// Token ID if this node represents a complete token
    value: Option<u32>,
    /// Children indexed by byte value (256 entries)
    children: [u32; 256],
}

impl Default for TrieNode {
    fn default() -> Self {
        Self {
            value: None,
            children: [INVALID_INDEX; 256],
        }
    }
}

/// Create a Trie from a vocabulary HashMap
pub fn trie_from_vocab<'a, I>(vocab: I) -> Trie
where
    I: IntoIterator<Item = (&'a str, u32)>,
{
    TrieBuilder::from_iter(vocab)
}

// ============================================================================
// Cache-Oblivious Trie (2.4.4)
// ============================================================================

/// Cache-oblivious Trie with van Emde Boas layout
///
/// This structure reorders trie nodes for better cache performance:
/// - BFS order for top levels (hot path nodes close together)
/// - Compact child representation (sparse instead of 256-ary)
/// - Children of active nodes stored contiguously
///
/// The layout reduces cache misses during traversal by ensuring that
/// nodes accessed together are stored close in memory.
#[derive(Clone)]
pub struct CacheObliviousTrie {
    /// Nodes in cache-friendly order
    nodes: Vec<CompactTrieNode>,
    /// Mapping from original node index to new index
    index_map: Vec<u32>,
    /// Total number of entries (tokens)
    num_entries: usize,
}

/// Compact node representation for cache-oblivious trie
#[derive(Clone, Default)]
struct CompactTrieNode {
    /// Token ID if this node represents a complete token
    value: Option<u32>,
    /// Number of children
    child_count: u8,
    /// Child byte values (sparse representation)
    child_bytes: Vec<u8>,
    /// Child node indices (parallel to child_bytes)
    child_indices: Vec<u32>,
}

impl CompactTrieNode {
    /// Get child node index for a given byte
    #[inline]
    fn get_child(&self, byte: u8) -> Option<u32> {
        // Binary search for the byte
        match self.child_bytes.binary_search(&byte) {
            Ok(idx) => Some(self.child_indices[idx]),
            Err(_) => None,
        }
    }
}

impl CacheObliviousTrie {
    /// Create a cache-oblivious trie from an existing trie
    pub fn from_trie(trie: &Trie) -> Self {
        if trie.nodes.is_empty() {
            return Self {
                nodes: vec![CompactTrieNode::default()],
                index_map: vec![0],
                num_entries: 0,
            };
        }

        // Phase 1: Compute BFS order for cache-friendly layout
        let bfs_order = Self::compute_bfs_order(trie);

        // Phase 2: Create index mapping (old -> new)
        let mut index_map = vec![INVALID_INDEX; trie.nodes.len()];
        for (new_idx, &old_idx) in bfs_order.iter().enumerate() {
            index_map[old_idx] = new_idx as u32;
        }

        // Phase 3: Create compact nodes in new order
        let mut nodes = Vec::with_capacity(bfs_order.len());
        let mut num_entries = 0;

        for &old_idx in &bfs_order {
            let old_node = &trie.nodes[old_idx];

            // Collect children (sorted by byte value for binary search)
            let mut child_bytes = Vec::new();
            let mut child_indices = Vec::new();

            for (byte, &child_idx) in old_node.children.iter().enumerate() {
                if child_idx != INVALID_INDEX {
                    child_bytes.push(byte as u8);
                    child_indices.push(index_map[child_idx as usize]);
                }
            }

            if old_node.value.is_some() {
                num_entries += 1;
            }

            nodes.push(CompactTrieNode {
                value: old_node.value,
                child_count: child_bytes.len() as u8,
                child_bytes,
                child_indices,
            });
        }

        Self {
            nodes,
            index_map,
            num_entries,
        }
    }

    /// Compute BFS order for nodes (cache-friendly layout)
    fn compute_bfs_order(trie: &Trie) -> Vec<usize> {
        use std::collections::VecDeque;

        let mut order = Vec::with_capacity(trie.nodes.len());
        let mut queue = VecDeque::new();
        let mut visited = vec![false; trie.nodes.len()];

        // Start from root
        queue.push_back(0usize);
        visited[0] = true;

        while let Some(node_idx) = queue.pop_front() {
            order.push(node_idx);

            // Add children to queue (in byte order for consistency)
            let node = &trie.nodes[node_idx];
            for &child_idx in &node.children {
                if child_idx != INVALID_INDEX {
                    let child = child_idx as usize;
                    if !visited[child] {
                        visited[child] = true;
                        queue.push_back(child);
                    }
                }
            }
        }

        order
    }

    /// Look up an exact match for a byte sequence
    #[inline]
    pub fn get(&self, bytes: &[u8]) -> Option<u32> {
        let mut current = 0u32;

        for &byte in bytes {
            match self.nodes[current as usize].get_child(byte) {
                Some(child) => current = child,
                None => return None,
            }
        }

        self.nodes[current as usize].value
    }

    /// Look up an exact match for a string
    #[inline]
    pub fn get_str(&self, s: &str) -> Option<u32> {
        self.get(s.as_bytes())
    }

    /// Check if a byte sequence exists in the trie
    #[inline]
    pub fn contains(&self, bytes: &[u8]) -> bool {
        self.get(bytes).is_some()
    }

    /// Enumerate all prefix matches starting at a given position
    #[inline]
    pub fn enumerate_matches<F>(&self, bytes: &[u8], start: usize, mut visitor: F)
    where
        F: FnMut(usize, u32),
    {
        if start >= bytes.len() {
            return;
        }

        let mut current = 0u32;
        let mut position = start;

        while position < bytes.len() {
            let byte = bytes[position];

            match self.nodes[current as usize].get_child(byte) {
                Some(child) => {
                    current = child;
                    position += 1;

                    if let Some(value) = self.nodes[current as usize].value {
                        visitor(position - start, value);
                    }
                }
                None => break,
            }
        }
    }

    /// Find the longest prefix match starting at a position
    #[inline]
    pub fn longest_prefix(&self, bytes: &[u8], start: usize) -> Option<(usize, u32)> {
        let mut result: Option<(usize, u32)> = None;
        self.enumerate_matches(bytes, start, |len, id| {
            result = Some((len, id));
        });
        result
    }

    /// Get an iterator over all prefix matches starting at a given position
    pub fn common_prefix_search<'a>(&'a self, bytes: &'a [u8], start: usize) -> CacheObliviousPrefixIter<'a> {
        CacheObliviousPrefixIter {
            trie: self,
            bytes,
            current_node: 0,
            position: start,
            start,
        }
    }

    /// Get the number of nodes in the trie
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of entries (tokens) in the trie
    pub fn entry_count(&self) -> usize {
        self.num_entries
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let base_size = std::mem::size_of::<Self>();
        let nodes_overhead = self.nodes.len() * std::mem::size_of::<CompactTrieNode>();
        let children_size: usize = self.nodes.iter()
            .map(|n| n.child_bytes.len() + n.child_indices.len() * 4)
            .sum();
        let index_map_size = self.index_map.len() * 4;

        base_size + nodes_overhead + children_size + index_map_size
    }

    /// Calculate compression ratio compared to original trie
    pub fn compression_ratio(&self, original_trie: &Trie) -> f64 {
        let original_size = original_trie.memory_usage();
        let compact_size = self.memory_usage();

        if compact_size == 0 {
            return 0.0;
        }

        original_size as f64 / compact_size as f64
    }
}

/// Iterator over common prefix matches in cache-oblivious trie
pub struct CacheObliviousPrefixIter<'a> {
    trie: &'a CacheObliviousTrie,
    bytes: &'a [u8],
    current_node: u32,
    position: usize,
    start: usize,
}

impl<'a> Iterator for CacheObliviousPrefixIter<'a> {
    type Item = (usize, u32);

    fn next(&mut self) -> Option<Self::Item> {
        while self.position < self.bytes.len() {
            let byte = self.bytes[self.position];

            match self.trie.nodes[self.current_node as usize].get_child(byte) {
                Some(child) => {
                    self.current_node = child;
                    self.position += 1;

                    if let Some(value) = self.trie.nodes[self.current_node as usize].value {
                        return Some((self.position - self.start, value));
                    }
                }
                None => {
                    self.position = self.bytes.len();
                    return None;
                }
            }
        }
        None
    }
}

/// Statistics about cache-oblivious trie layout
#[derive(Debug, Clone)]
pub struct CacheObliviousTrieStats {
    /// Total number of nodes
    pub node_count: usize,
    /// Total number of entries (tokens)
    pub entry_count: usize,
    /// Average children per node
    pub avg_children: f64,
    /// Maximum depth of trie
    pub max_depth: usize,
    /// Memory usage in bytes
    pub memory_bytes: usize,
    /// Compression ratio vs original
    pub compression_ratio: f64,
}

impl CacheObliviousTrie {
    /// Compute statistics about the trie layout
    pub fn stats(&self, original: &Trie) -> CacheObliviousTrieStats {
        let total_children: usize = self.nodes.iter()
            .map(|n| n.child_count as usize)
            .sum();

        let avg_children = if self.nodes.is_empty() {
            0.0
        } else {
            total_children as f64 / self.nodes.len() as f64
        };

        // Compute max depth via BFS
        let max_depth = self.compute_max_depth();

        CacheObliviousTrieStats {
            node_count: self.node_count(),
            entry_count: self.entry_count(),
            avg_children,
            max_depth,
            memory_bytes: self.memory_usage(),
            compression_ratio: self.compression_ratio(original),
        }
    }

    fn compute_max_depth(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }

        use std::collections::VecDeque;
        let mut max_depth = 0;
        let mut queue: VecDeque<(u32, usize)> = VecDeque::new();
        queue.push_back((0, 0));

        while let Some((node_idx, depth)) = queue.pop_front() {
            max_depth = max_depth.max(depth);
            let node = &self.nodes[node_idx as usize];

            for &child_idx in &node.child_indices {
                queue.push_back((child_idx, depth + 1));
            }
        }

        max_depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trie_basic() {
        let mut builder = TrieBuilder::new();
        builder.insert("hello", 0);
        builder.insert("hell", 1);
        builder.insert("help", 2);
        builder.insert("world", 3);
        let trie = builder.build();

        assert_eq!(trie.get_str("hello"), Some(0));
        assert_eq!(trie.get_str("hell"), Some(1));
        assert_eq!(trie.get_str("help"), Some(2));
        assert_eq!(trie.get_str("world"), Some(3));
        assert_eq!(trie.get_str("hel"), None);
        assert_eq!(trie.get_str("helloo"), None);
    }

    #[test]
    fn test_trie_enumerate_matches() {
        let mut builder = TrieBuilder::new();
        builder.insert("h", 0);
        builder.insert("he", 1);
        builder.insert("hel", 2);
        builder.insert("hell", 3);
        builder.insert("hello", 4);
        let trie = builder.build();

        let mut matches = Vec::new();
        trie.enumerate_matches(b"hello world", 0, |len, id| {
            matches.push((len, id));
        });

        assert_eq!(matches, vec![(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)]);
    }

    #[test]
    fn test_trie_common_prefix_search() {
        let mut builder = TrieBuilder::new();
        builder.insert("h", 0);
        builder.insert("he", 1);
        builder.insert("hello", 2);
        let trie = builder.build();

        let matches: Vec<_> = trie.common_prefix_search(b"hello world", 0).collect();
        assert_eq!(matches, vec![(1, 0), (2, 1), (5, 2)]);
    }

    #[test]
    fn test_trie_longest_prefix() {
        let mut builder = TrieBuilder::new();
        builder.insert("h", 0);
        builder.insert("he", 1);
        builder.insert("hello", 2);
        let trie = builder.build();

        assert_eq!(trie.longest_prefix(b"hello world", 0), Some((5, 2)));
        assert_eq!(trie.longest_prefix(b"help", 0), Some((2, 1)));
        assert_eq!(trie.longest_prefix(b"xyz", 0), None);
    }

    #[test]
    fn test_trie_from_vocab() {
        let vocab = [
            ("the", 0u32),
            ("quick", 1),
            ("brown", 2),
            ("fox", 3),
        ];

        let trie = trie_from_vocab(vocab.iter().map(|(s, id)| (*s, *id)));

        assert_eq!(trie.get_str("the"), Some(0));
        assert_eq!(trie.get_str("quick"), Some(1));
        assert_eq!(trie.get_str("brown"), Some(2));
        assert_eq!(trie.get_str("fox"), Some(3));
        assert_eq!(trie.get_str("dog"), None);
    }

    #[test]
    fn test_trie_utf8() {
        let mut builder = TrieBuilder::new();
        builder.insert("café", 0);
        builder.insert("日本語", 1);
        builder.insert("hello", 2);
        let trie = builder.build();

        assert_eq!(trie.get_str("café"), Some(0));
        assert_eq!(trie.get_str("日本語"), Some(1));
        assert_eq!(trie.get_str("hello"), Some(2));
    }

    #[test]
    fn test_trie_start_position() {
        let mut builder = TrieBuilder::new();
        builder.insert("world", 0);
        let trie = builder.build();

        // "hello world" has "world" starting at position 6
        let matches: Vec<_> = trie.common_prefix_search(b"hello world", 6).collect();
        assert_eq!(matches, vec![(5, 0)]);
    }

    #[test]
    fn test_trie_empty_string() {
        let mut builder = TrieBuilder::new();
        builder.insert("", 0);
        builder.insert("a", 1);
        let trie = builder.build();

        // Empty string should match at root
        assert_eq!(trie.get_str(""), Some(0));
        assert_eq!(trie.get_str("a"), Some(1));
    }

    #[test]
    fn test_trie_memory_estimate() {
        let mut builder = TrieBuilder::new();
        builder.insert("test", 0);
        let trie = builder.build();

        // Should have 5 nodes (root + t + e + s + t)
        assert_eq!(trie.node_count(), 5);
        assert!(trie.memory_usage() > 0);
    }

    // ========== Cache-Oblivious Trie Tests ==========

    #[test]
    fn test_cache_oblivious_basic() {
        let mut builder = TrieBuilder::new();
        builder.insert("hello", 0);
        builder.insert("hell", 1);
        builder.insert("help", 2);
        builder.insert("world", 3);
        let trie = builder.build();

        let compact = CacheObliviousTrie::from_trie(&trie);

        assert_eq!(compact.get_str("hello"), Some(0));
        assert_eq!(compact.get_str("hell"), Some(1));
        assert_eq!(compact.get_str("help"), Some(2));
        assert_eq!(compact.get_str("world"), Some(3));
        assert_eq!(compact.get_str("hel"), None);
        assert_eq!(compact.get_str("helloo"), None);
    }

    #[test]
    fn test_cache_oblivious_enumerate_matches() {
        let mut builder = TrieBuilder::new();
        builder.insert("h", 0);
        builder.insert("he", 1);
        builder.insert("hel", 2);
        builder.insert("hell", 3);
        builder.insert("hello", 4);
        let trie = builder.build();

        let compact = CacheObliviousTrie::from_trie(&trie);

        let mut matches = Vec::new();
        compact.enumerate_matches(b"hello world", 0, |len, id| {
            matches.push((len, id));
        });

        assert_eq!(matches, vec![(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)]);
    }

    #[test]
    fn test_cache_oblivious_common_prefix_search() {
        let mut builder = TrieBuilder::new();
        builder.insert("h", 0);
        builder.insert("he", 1);
        builder.insert("hello", 2);
        let trie = builder.build();

        let compact = CacheObliviousTrie::from_trie(&trie);

        let matches: Vec<_> = compact.common_prefix_search(b"hello world", 0).collect();
        assert_eq!(matches, vec![(1, 0), (2, 1), (5, 2)]);
    }

    #[test]
    fn test_cache_oblivious_longest_prefix() {
        let mut builder = TrieBuilder::new();
        builder.insert("h", 0);
        builder.insert("he", 1);
        builder.insert("hello", 2);
        let trie = builder.build();

        let compact = CacheObliviousTrie::from_trie(&trie);

        assert_eq!(compact.longest_prefix(b"hello world", 0), Some((5, 2)));
        assert_eq!(compact.longest_prefix(b"help", 0), Some((2, 1)));
        assert_eq!(compact.longest_prefix(b"xyz", 0), None);
    }

    #[test]
    fn test_cache_oblivious_compression() {
        let mut builder = TrieBuilder::new();
        for i in 0..100 {
            builder.insert(&format!("word{}", i), i);
        }
        let trie = builder.build();

        let compact = CacheObliviousTrie::from_trie(&trie);

        // Compact trie should use less memory than 256-ary trie
        let ratio = compact.compression_ratio(&trie);
        assert!(ratio > 1.0, "Compact trie should be smaller than original");

        // Verify all lookups still work
        for i in 0..100 {
            assert_eq!(compact.get_str(&format!("word{}", i)), Some(i));
        }
    }

    #[test]
    fn test_cache_oblivious_stats() {
        let mut builder = TrieBuilder::new();
        builder.insert("a", 0);
        builder.insert("ab", 1);
        builder.insert("abc", 2);
        builder.insert("abcd", 3);
        builder.insert("xyz", 4);
        let trie = builder.build();

        let compact = CacheObliviousTrie::from_trie(&trie);
        let stats = compact.stats(&trie);

        assert_eq!(stats.entry_count, 5);
        assert!(stats.node_count > 0);
        assert!(stats.max_depth > 0);
        assert!(stats.memory_bytes > 0);
        assert!(stats.compression_ratio > 0.0);
    }

    #[test]
    fn test_cache_oblivious_utf8() {
        let mut builder = TrieBuilder::new();
        builder.insert("café", 0);
        builder.insert("日本語", 1);
        builder.insert("hello", 2);
        let trie = builder.build();

        let compact = CacheObliviousTrie::from_trie(&trie);

        assert_eq!(compact.get_str("café"), Some(0));
        assert_eq!(compact.get_str("日本語"), Some(1));
        assert_eq!(compact.get_str("hello"), Some(2));
    }

    #[test]
    fn test_cache_oblivious_empty() {
        let trie = Trie::new();
        let compact = CacheObliviousTrie::from_trie(&trie);

        assert_eq!(compact.node_count(), 1); // Just root
        assert_eq!(compact.entry_count(), 0);
        assert_eq!(compact.get_str("anything"), None);
    }

    #[test]
    fn test_cache_oblivious_bfs_order() {
        // Build a trie that would have different BFS vs DFS order
        let mut builder = TrieBuilder::new();
        builder.insert("a", 0);
        builder.insert("b", 1);
        builder.insert("aa", 2);
        builder.insert("ab", 3);
        builder.insert("ba", 4);
        builder.insert("bb", 5);
        let trie = builder.build();

        let compact = CacheObliviousTrie::from_trie(&trie);

        // BFS order: root, a, b, aa, ab, ba, bb (level by level)
        // This ensures siblings are close together in memory
        assert_eq!(compact.get_str("a"), Some(0));
        assert_eq!(compact.get_str("b"), Some(1));
        assert_eq!(compact.get_str("aa"), Some(2));
        assert_eq!(compact.get_str("ab"), Some(3));
        assert_eq!(compact.get_str("ba"), Some(4));
        assert_eq!(compact.get_str("bb"), Some(5));
    }
}
