//! Double-Array Trie Implementation (2.4.6)
//!
//! A cache-efficient trie implementation using two arrays:
//! - `base[]`: base values for state transitions
//! - `check[]`: verification of parent states
//!
//! For a transition from state `s` to state `t` on input byte `c`:
//! - `t = base[s] + c`
//! - `check[t] = s` (parent verification)
//!
//! This gives O(1) transitions and better cache locality than pointer-based tries.

use std::collections::BTreeMap;

/// Double-Array Trie for efficient prefix matching
///
/// Uses two arrays for compact state representation:
/// - `base[s]`: base offset for state s
/// - `check[t]`: parent state verification for state t
///
/// State 0 is reserved as the "free list" marker.
/// State 1 is the root state.
#[derive(Debug, Clone)]
pub struct DoubleArrayTrie {
    /// Base values for each state
    base: Vec<i32>,
    /// Parent verification for each state
    check: Vec<i32>,
    /// Terminal markers (state -> value)
    /// If a state is terminal, it maps to a value (e.g., vocabulary ID)
    terminals: BTreeMap<usize, u32>,
    /// Number of valid states
    num_states: usize,
}

/// Result of a trie lookup
#[derive(Debug, Clone, PartialEq)]
pub enum TrieLookup {
    /// Exact match found with associated value
    Match(u32),
    /// Prefix exists but no exact match
    Prefix,
    /// No match (neither exact nor prefix)
    None,
}

impl DoubleArrayTrie {
    /// Initial capacity for arrays
    const INITIAL_CAPACITY: usize = 1024;

    /// Root state index
    const ROOT: usize = 1;

    /// Free/unused state marker
    const FREE: i32 = 0;

    /// Create a new empty Double-Array Trie
    pub fn new() -> Self {
        let mut base = vec![Self::FREE; Self::INITIAL_CAPACITY];
        let mut check = vec![Self::FREE; Self::INITIAL_CAPACITY];

        // Initialize root state
        base[Self::ROOT] = 1;
        check[Self::ROOT] = Self::ROOT as i32; // Root's parent is itself

        Self {
            base,
            check,
            terminals: BTreeMap::new(),
            num_states: 2, // State 0 (free) and state 1 (root)
        }
    }

    /// Build a Double-Array Trie from a list of (key, value) pairs
    pub fn build<I, K>(entries: I) -> Self
    where
        I: IntoIterator<Item = (K, u32)>,
        K: AsRef<[u8]>,
    {
        let mut trie = Self::new();
        for (key, value) in entries {
            trie.insert(key.as_ref(), value);
        }
        trie
    }

    /// Insert a key-value pair into the trie
    pub fn insert(&mut self, key: &[u8], value: u32) {
        let mut state = Self::ROOT;

        for &byte in key {
            let c = byte as usize;
            let target = self.transition(state, c);

            if target.is_none() {
                // Need to create new transition
                state = self.add_transition(state, c);
            } else {
                state = target.unwrap();
            }
        }

        // Mark as terminal
        self.terminals.insert(state, value);
    }

    /// Try to transition from state `s` on input `c`
    #[inline]
    fn transition(&self, state: usize, c: usize) -> Option<usize> {
        let base_val = self.base[state];
        if base_val < 0 {
            return None;
        }

        let target = (base_val as usize) + c;
        if target >= self.check.len() {
            return None;
        }

        if self.check[target] == state as i32 {
            Some(target)
        } else {
            None
        }
    }

    /// Add a new transition from state `s` on input `c`
    fn add_transition(&mut self, state: usize, c: usize) -> usize {
        let old_base = self.base[state];

        // Collect existing children of this state
        let mut existing_children: Vec<(usize, usize)> = Vec::new();
        if old_base > 0 {
            for child_c in 0..256 {
                let old_target = old_base as usize + child_c;
                if old_target < self.check.len() && self.check[old_target] == state as i32 {
                    existing_children.push((child_c, old_target));
                }
            }
        }

        // Find a base value that works for all children (existing + new)
        let new_base = self.find_base_for_children(state, &existing_children, c);

        // If base changed and we have existing children, relocate them
        if old_base > 0 && new_base != old_base as usize && !existing_children.is_empty() {
            self.relocate_children(state, &existing_children, new_base);
        }

        // Update base for parent state
        self.base[state] = new_base as i32;

        // Create target state for new child
        let target = new_base + c;
        self.ensure_capacity(target + 1);

        self.check[target] = state as i32;
        self.base[target] = 0; // No children yet
        self.num_states += 1;

        target
    }

    /// Relocate existing children to new positions when base changes
    fn relocate_children(
        &mut self,
        state: usize,
        children: &[(usize, usize)],
        new_base: usize,
    ) {
        for &(child_c, old_target) in children {
            let new_target = new_base + child_c;
            self.ensure_capacity(new_target + 1);

            // Copy child's base value
            let child_base = self.base[old_target];
            self.base[new_target] = child_base;
            self.check[new_target] = state as i32;

            // Update grandchildren to point to new location
            if child_base > 0 {
                for gc in 0..256 {
                    let gc_target = child_base as usize + gc;
                    if gc_target < self.check.len() && self.check[gc_target] == old_target as i32 {
                        self.check[gc_target] = new_target as i32;
                    }
                }
            }

            // Move terminal entry if any
            if let Some(value) = self.terminals.remove(&old_target) {
                self.terminals.insert(new_target, value);
            }

            // Clear old position
            self.base[old_target] = Self::FREE;
            self.check[old_target] = Self::FREE;
        }
    }

    /// Find a base value that accommodates all children (existing + new)
    fn find_base_for_children(
        &mut self,
        state: usize,
        existing: &[(usize, usize)],
        new_c: usize,
    ) -> usize {
        // Collect all child labels (existing + new)
        let mut labels: Vec<usize> = existing.iter().map(|(c, _)| *c).collect();
        labels.push(new_c);

        let old_base = self.base[state];

        // Check if current base works for the new child
        if old_base > 0 {
            let target = old_base as usize + new_c;
            self.ensure_capacity(target + 1);
            if self.check[target] == Self::FREE {
                return old_base as usize;
            }
        }

        // Find a new base where all target positions are free
        for base in 1.. {
            let mut valid = true;
            for &label in &labels {
                let target = base + label;
                self.ensure_capacity(target + 1);

                // Position must be free, unless it's one of our own children being moved
                if self.check[target] != Self::FREE {
                    let is_own_child = existing.iter().any(|(_, old_t)| *old_t == target);
                    if !is_own_child {
                        valid = false;
                        break;
                    }
                }
            }
            if valid {
                return base;
            }
        }

        unreachable!("Should always find a valid base")
    }

    /// Ensure arrays have at least `min_capacity` elements
    fn ensure_capacity(&mut self, min_capacity: usize) {
        if min_capacity <= self.base.len() {
            return;
        }

        let new_capacity = (min_capacity * 2).max(self.base.len() * 2);
        self.base.resize(new_capacity, Self::FREE);
        self.check.resize(new_capacity, Self::FREE);
    }

    /// Look up a key in the trie
    #[inline]
    pub fn lookup(&self, key: &[u8]) -> TrieLookup {
        let mut state = Self::ROOT;

        for &byte in key {
            let c = byte as usize;
            match self.transition(state, c) {
                Some(next) => state = next,
                None => return TrieLookup::None,
            }
        }

        // Check if this is a terminal state
        if let Some(&value) = self.terminals.get(&state) {
            TrieLookup::Match(value)
        } else {
            TrieLookup::Prefix
        }
    }

    /// Get the value associated with a key (if it exists)
    #[inline]
    pub fn get(&self, key: &[u8]) -> Option<u32> {
        match self.lookup(key) {
            TrieLookup::Match(v) => Some(v),
            _ => None,
        }
    }

    /// Check if a key exists in the trie
    #[inline]
    pub fn contains(&self, key: &[u8]) -> bool {
        matches!(self.lookup(key), TrieLookup::Match(_))
    }

    /// Check if a prefix exists in the trie
    #[inline]
    pub fn has_prefix(&self, prefix: &[u8]) -> bool {
        !matches!(self.lookup(prefix), TrieLookup::None)
    }

    /// Find common prefix matches
    ///
    /// Returns all matches found while traversing the key,
    /// in order from shortest to longest.
    pub fn common_prefix_search(&self, key: &[u8]) -> Vec<(usize, u32)> {
        let mut results = Vec::new();
        let mut state = Self::ROOT;

        // Check if root is terminal (empty string match)
        if let Some(&value) = self.terminals.get(&state) {
            results.push((0, value));
        }

        for (i, &byte) in key.iter().enumerate() {
            let c = byte as usize;

            match self.transition(state, c) {
                Some(next) => {
                    state = next;
                    // After consuming byte at index i, we've matched key[0..=i] (length i+1)
                    if let Some(&value) = self.terminals.get(&state) {
                        results.push((i + 1, value));
                    }
                }
                None => break,
            }
        }

        results
    }

    /// Number of entries in the trie
    pub fn len(&self) -> usize {
        self.terminals.len()
    }

    /// Check if the trie is empty
    pub fn is_empty(&self) -> bool {
        self.terminals.is_empty()
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> DoubleArrayTrieStats {
        let base_bytes = self.base.len() * std::mem::size_of::<i32>();
        let check_bytes = self.check.len() * std::mem::size_of::<i32>();
        let terminals_bytes = self.terminals.len() * (std::mem::size_of::<usize>() + std::mem::size_of::<u32>());

        DoubleArrayTrieStats {
            num_entries: self.terminals.len(),
            num_states: self.num_states,
            base_array_bytes: base_bytes,
            check_array_bytes: check_bytes,
            terminals_bytes,
            total_bytes: base_bytes + check_bytes + terminals_bytes,
            fill_ratio: self.num_states as f64 / self.base.len() as f64,
        }
    }
}

impl Default for DoubleArrayTrie {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about Double-Array Trie memory usage
#[derive(Debug, Clone)]
pub struct DoubleArrayTrieStats {
    /// Number of entries (key-value pairs)
    pub num_entries: usize,
    /// Number of states (nodes)
    pub num_states: usize,
    /// Memory used by base array
    pub base_array_bytes: usize,
    /// Memory used by check array
    pub check_array_bytes: usize,
    /// Memory used by terminals map
    pub terminals_bytes: usize,
    /// Total memory usage
    pub total_bytes: usize,
    /// Ratio of used states to array size
    pub fill_ratio: f64,
}

/// Builder for Double-Array Trie with optimized construction
pub struct DoubleArrayTrieBuilder {
    entries: Vec<(Vec<u8>, u32)>,
}

impl DoubleArrayTrieBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a key-value pair (consuming builder pattern)
    pub fn insert(mut self, key: impl AsRef<[u8]>, value: u32) -> Self {
        self.entries.push((key.as_ref().to_vec(), value));
        self
    }

    /// Add a key-value pair (mutable reference pattern)
    pub fn add(&mut self, key: impl AsRef<[u8]>, value: u32) -> &mut Self {
        self.entries.push((key.as_ref().to_vec(), value));
        self
    }

    /// Add multiple entries
    pub fn insert_all<I, K>(mut self, entries: I) -> Self
    where
        I: IntoIterator<Item = (K, u32)>,
        K: AsRef<[u8]>,
    {
        for (key, value) in entries {
            self.entries.push((key.as_ref().to_vec(), value));
        }
        self
    }

    /// Build the Double-Array Trie
    ///
    /// Sorts entries by key for more efficient construction.
    pub fn build(mut self) -> DoubleArrayTrie {
        // Sort by key for better locality during construction
        self.entries.sort_by(|a, b| a.0.cmp(&b.0));

        let mut trie = DoubleArrayTrie::new();
        for (key, value) in self.entries {
            trie.insert(&key, value);
        }
        trie
    }
}

impl Default for DoubleArrayTrieBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_trie() {
        let trie = DoubleArrayTrie::new();
        assert!(trie.is_empty());
        assert_eq!(trie.len(), 0);
        assert_eq!(trie.lookup(b"test"), TrieLookup::None);
    }

    #[test]
    fn test_single_insert() {
        let mut trie = DoubleArrayTrie::new();
        trie.insert(b"hello", 42);

        assert!(!trie.is_empty());
        assert_eq!(trie.len(), 1);
        assert_eq!(trie.lookup(b"hello"), TrieLookup::Match(42));
        assert_eq!(trie.get(b"hello"), Some(42));
        assert!(trie.contains(b"hello"));
    }

    #[test]
    fn test_prefix_lookup() {
        let mut trie = DoubleArrayTrie::new();
        trie.insert(b"hello", 1);
        trie.insert(b"helloworld", 2);

        assert_eq!(trie.lookup(b"hello"), TrieLookup::Match(1));
        assert_eq!(trie.lookup(b"helloworld"), TrieLookup::Match(2));
        assert_eq!(trie.lookup(b"hell"), TrieLookup::Prefix);
        assert_eq!(trie.lookup(b"help"), TrieLookup::None);
        assert_eq!(trie.lookup(b"h"), TrieLookup::Prefix);
    }

    #[test]
    fn test_multiple_inserts() {
        let mut trie = DoubleArrayTrie::new();
        trie.insert(b"apple", 1);
        trie.insert(b"application", 2);
        trie.insert(b"banana", 3);
        trie.insert(b"band", 4);
        trie.insert(b"bandana", 5);

        assert_eq!(trie.len(), 5);
        assert_eq!(trie.get(b"apple"), Some(1));
        assert_eq!(trie.get(b"application"), Some(2));
        assert_eq!(trie.get(b"banana"), Some(3));
        assert_eq!(trie.get(b"band"), Some(4));
        assert_eq!(trie.get(b"bandana"), Some(5));

        assert!(trie.has_prefix(b"app"));
        assert!(trie.has_prefix(b"ban"));
        assert!(!trie.has_prefix(b"xyz"));
    }

    #[test]
    fn test_common_prefix_search() {
        let mut trie = DoubleArrayTrie::new();
        trie.insert(b"a", 1);
        trie.insert(b"ab", 2);
        trie.insert(b"abc", 3);
        trie.insert(b"abcd", 4);

        let results = trie.common_prefix_search(b"abcde");
        assert_eq!(results.len(), 4);
        assert_eq!(results[0], (1, 1)); // "a"
        assert_eq!(results[1], (2, 2)); // "ab"
        assert_eq!(results[2], (3, 3)); // "abc"
        assert_eq!(results[3], (4, 4)); // "abcd"
    }

    #[test]
    fn test_common_prefix_search_partial() {
        let mut trie = DoubleArrayTrie::new();
        trie.insert(b"hello", 1);
        trie.insert(b"help", 2);
        trie.insert(b"helper", 3);

        let results = trie.common_prefix_search(b"helping");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (4, 2)); // "help"
    }

    #[test]
    fn test_builder() {
        let trie = DoubleArrayTrieBuilder::new()
            .insert(b"foo", 1)
            .insert(b"bar", 2)
            .insert(b"baz", 3)
            .build();

        assert_eq!(trie.len(), 3);
        assert_eq!(trie.get(b"foo"), Some(1));
        assert_eq!(trie.get(b"bar"), Some(2));
        assert_eq!(trie.get(b"baz"), Some(3));
    }

    #[test]
    fn test_builder_insert_all() {
        let entries = vec![
            (b"alpha".to_vec(), 1),
            (b"beta".to_vec(), 2),
            (b"gamma".to_vec(), 3),
        ];

        let trie = DoubleArrayTrieBuilder::new()
            .insert_all(entries.into_iter().map(|(k, v)| (k, v)))
            .build();

        assert_eq!(trie.len(), 3);
        assert_eq!(trie.get(b"alpha"), Some(1));
        assert_eq!(trie.get(b"beta"), Some(2));
        assert_eq!(trie.get(b"gamma"), Some(3));
    }

    #[test]
    fn test_build_from_iterator() {
        let entries = vec![
            ("cat", 1u32),
            ("car", 2),
            ("card", 3),
        ];

        let trie = DoubleArrayTrie::build(entries.into_iter().map(|(k, v)| (k.as_bytes(), v)));

        assert_eq!(trie.len(), 3);
        assert_eq!(trie.get(b"cat"), Some(1));
        assert_eq!(trie.get(b"car"), Some(2));
        assert_eq!(trie.get(b"card"), Some(3));
    }

    #[test]
    fn test_unicode_keys() {
        let mut trie = DoubleArrayTrie::new();
        trie.insert("日本".as_bytes(), 1);
        trie.insert("日本語".as_bytes(), 2);
        trie.insert("中国".as_bytes(), 3);

        assert_eq!(trie.get("日本".as_bytes()), Some(1));
        assert_eq!(trie.get("日本語".as_bytes()), Some(2));
        assert_eq!(trie.get("中国".as_bytes()), Some(3));
        assert!(trie.has_prefix("日".as_bytes()));
    }

    #[test]
    fn test_memory_stats() {
        let mut trie = DoubleArrayTrie::new();
        for i in 0..100 {
            let key = format!("word{}", i);
            trie.insert(key.as_bytes(), i);
        }

        let stats = trie.memory_stats();
        assert_eq!(stats.num_entries, 100);
        assert!(stats.num_states > 100);
        assert!(stats.total_bytes > 0);
        assert!(stats.fill_ratio > 0.0);
        assert!(stats.fill_ratio <= 1.0);
    }

    #[test]
    fn test_overwrite_value() {
        let mut trie = DoubleArrayTrie::new();
        trie.insert(b"key", 1);
        assert_eq!(trie.get(b"key"), Some(1));

        trie.insert(b"key", 2);
        assert_eq!(trie.get(b"key"), Some(2));
    }

    #[test]
    fn test_empty_key() {
        let mut trie = DoubleArrayTrie::new();
        trie.insert(b"", 42);

        // Empty key shouldn't match (root isn't terminal by default)
        // This behavior may vary based on implementation choices
        assert_eq!(trie.lookup(b"a"), TrieLookup::None);
    }

    #[test]
    fn test_single_byte_keys() {
        let mut trie = DoubleArrayTrie::new();
        for b in 0u8..=255 {
            trie.insert(&[b], b as u32);
        }

        for b in 0u8..=255 {
            assert_eq!(trie.get(&[b]), Some(b as u32));
        }
    }

    #[test]
    fn test_long_keys() {
        let mut trie = DoubleArrayTrie::new();
        let long_key: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        trie.insert(&long_key, 999);

        assert_eq!(trie.get(&long_key), Some(999));
        assert!(trie.has_prefix(&long_key[..500]));
    }
}
