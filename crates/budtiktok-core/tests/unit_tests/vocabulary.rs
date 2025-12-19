//! Vocabulary Unit Tests (9.1.4)
//!
//! Tests for vocabulary data structures including:
//! - tokenizer.json loading
//! - Token to ID lookup
//! - ID to token lookup
//! - Trie prefix search
//! - Aho-Corasick matching
//! - Double-array trie

use budtiktok_core::{Trie, TrieBuilder};
use budtiktok_core::{DoubleArrayTrie, DoubleArrayTrieBuilder};
use budtiktok_core::{SpecialTokenMatcher, SpecialTokenMatcherBuilder, AddedToken};
use ahash::AHashMap;

// =============================================================================
// Test Helper: Simple Vocab wrapper for test purposes
// =============================================================================

/// Simple vocabulary for testing - wraps a HashMap
struct Vocab {
    token_to_id: AHashMap<String, u32>,
    id_to_token: Vec<Option<String>>,
}

impl Vocab {
    fn new() -> Self {
        Self {
            token_to_id: AHashMap::new(),
            id_to_token: Vec::new(),
        }
    }

    fn insert(&mut self, token: String, id: u32) {
        self.token_to_id.insert(token.clone(), id);
        let id_usize = id as usize;
        if id_usize >= self.id_to_token.len() {
            self.id_to_token.resize(id_usize + 1, None);
        }
        self.id_to_token[id_usize] = Some(token);
    }

    fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).and_then(|s| s.as_deref())
    }

    fn len(&self) -> usize {
        self.token_to_id.len()
    }

    fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }
}

impl std::iter::FromIterator<(String, u32)> for Vocab {
    fn from_iter<I: IntoIterator<Item = (String, u32)>>(iter: I) -> Self {
        let mut vocab = Vocab::new();
        for (token, id) in iter {
            vocab.insert(token, id);
        }
        vocab
    }
}

// =============================================================================
// Vocabulary Basic Tests
// =============================================================================

#[test]
fn test_vocab_new() {
    let vocab = Vocab::new();
    assert!(vocab.is_empty());
    assert_eq!(vocab.len(), 0);
}

#[test]
fn test_vocab_insert_and_lookup() {
    let mut vocab = Vocab::new();

    vocab.insert("hello".to_string(), 0);
    vocab.insert("world".to_string(), 1);

    assert_eq!(vocab.get_id("hello"), Some(0));
    assert_eq!(vocab.get_id("world"), Some(1));
    assert_eq!(vocab.get_id("missing"), None);
}

#[test]
fn test_vocab_reverse_lookup() {
    let mut vocab = Vocab::new();

    vocab.insert("hello".to_string(), 0);
    vocab.insert("world".to_string(), 1);

    assert_eq!(vocab.get_token(0), Some("hello"));
    assert_eq!(vocab.get_token(1), Some("world"));
    assert_eq!(vocab.get_token(999), None);
}

#[test]
fn test_vocab_from_iter() {
    let tokens = vec![
        ("hello".to_string(), 0u32),
        ("world".to_string(), 1),
        ("!".to_string(), 2),
    ];

    let vocab: Vocab = tokens.into_iter().collect();

    assert_eq!(vocab.len(), 3);
    assert_eq!(vocab.get_id("hello"), Some(0));
    assert_eq!(vocab.get_token(2), Some("!"));
}

#[test]
fn test_vocab_special_tokens() {
    let mut vocab = Vocab::new();

    vocab.insert("[PAD]".to_string(), 0);
    vocab.insert("[UNK]".to_string(), 100);
    vocab.insert("[CLS]".to_string(), 101);
    vocab.insert("[SEP]".to_string(), 102);
    vocab.insert("[MASK]".to_string(), 103);

    assert_eq!(vocab.get_id("[PAD]"), Some(0));
    assert_eq!(vocab.get_id("[UNK]"), Some(100));
    assert_eq!(vocab.get_id("[CLS]"), Some(101));
}

#[test]
fn test_vocab_unicode_tokens() {
    let mut vocab = Vocab::new();

    vocab.insert("日本語".to_string(), 0);
    vocab.insert("한글".to_string(), 1);
    vocab.insert("emoji😀".to_string(), 2);

    assert_eq!(vocab.get_id("日本語"), Some(0));
    assert_eq!(vocab.get_id("한글"), Some(1));
    assert_eq!(vocab.get_token(2), Some("emoji😀"));
}

#[test]
fn test_vocab_continuation_tokens() {
    let mut vocab = Vocab::new();

    // WordPiece style continuation tokens
    vocab.insert("##ing".to_string(), 0);
    vocab.insert("##ed".to_string(), 1);
    vocab.insert("##s".to_string(), 2);

    assert_eq!(vocab.get_id("##ing"), Some(0));
    assert_eq!(vocab.get_id("##ed"), Some(1));
}

// =============================================================================
// Trie Tests
// =============================================================================

#[test]
fn test_trie_insert_and_lookup() {
    let mut builder = TrieBuilder::new();

    builder.insert("hello", 0);
    builder.insert("help", 1);
    builder.insert("world", 2);

    let trie = builder.build();

    assert_eq!(trie.get("hello"), Some(0));
    assert_eq!(trie.get("help"), Some(1));
    assert_eq!(trie.get("world"), Some(2));
    assert_eq!(trie.get("missing"), None);
}

#[test]
fn test_trie_prefix_search() {
    let mut builder = TrieBuilder::new();

    builder.insert("hello", 0);
    builder.insert("help", 1);
    builder.insert("helper", 2);
    builder.insert("helping", 3);

    let trie = builder.build();

    // Common prefix search on "helping"
    let matches: Vec<_> = trie.common_prefix_search("helping").collect();

    // Should find "help" and "helping"
    assert!(matches.iter().any(|(_, id)| *id == 1)); // help
    assert!(matches.iter().any(|(_, id)| *id == 3)); // helping
}

#[test]
fn test_trie_prefix_only() {
    let mut builder = TrieBuilder::new();

    builder.insert("abc", 0);
    builder.insert("abcd", 1);
    builder.insert("abcde", 2);

    let trie = builder.build();

    let matches: Vec<_> = trie.common_prefix_search("abcdef").collect();

    // Should find all three
    assert_eq!(matches.len(), 3);
    assert!(matches.iter().any(|(len, _)| *len == 3)); // "abc"
    assert!(matches.iter().any(|(len, _)| *len == 4)); // "abcd"
    assert!(matches.iter().any(|(len, _)| *len == 5)); // "abcde"
}

#[test]
fn test_trie_no_match() {
    let mut builder = TrieBuilder::new();

    builder.insert("hello", 0);

    let trie = builder.build();

    let matches: Vec<_> = trie.common_prefix_search("world").collect();
    assert!(matches.is_empty());
}

#[test]
fn test_trie_empty() {
    let builder = TrieBuilder::new();
    let trie = builder.build();

    assert_eq!(trie.get("anything"), None);
    let matches: Vec<_> = trie.common_prefix_search("anything").collect();
    assert!(matches.is_empty());
}

#[test]
fn test_trie_single_char() {
    let mut builder = TrieBuilder::new();

    builder.insert("a", 0);
    builder.insert("b", 1);

    let trie = builder.build();

    assert_eq!(trie.get("a"), Some(0));
    assert_eq!(trie.get("b"), Some(1));
}

#[test]
fn test_trie_unicode() {
    let mut builder = TrieBuilder::new();

    builder.insert("日本", 0);
    builder.insert("日本語", 1);

    let trie = builder.build();

    assert_eq!(trie.get("日本"), Some(0));
    assert_eq!(trie.get("日本語"), Some(1));

    let matches: Vec<_> = trie.common_prefix_search("日本語入力").collect();
    assert!(matches.iter().any(|(_, id)| *id == 0)); // 日本
    assert!(matches.iter().any(|(_, id)| *id == 1)); // 日本語
}

// =============================================================================
// Double Array Trie Tests
// =============================================================================

#[test]
fn test_double_array_trie_basic() {
    let mut builder = DoubleArrayTrieBuilder::new();

    builder.insert("hello", 0);
    builder.insert("world", 1);

    let trie = builder.build();

    assert_eq!(trie.get("hello"), Some(0));
    assert_eq!(trie.get("world"), Some(1));
    assert_eq!(trie.get("missing"), None);
}

#[test]
fn test_double_array_trie_prefix() {
    let mut builder = DoubleArrayTrieBuilder::new();

    builder.insert("test", 0);
    builder.insert("testing", 1);
    builder.insert("tested", 2);

    let trie = builder.build();

    let matches: Vec<_> = trie.common_prefix_search("testing123").collect();

    assert!(matches.iter().any(|(_, id)| *id == 0)); // test
    assert!(matches.iter().any(|(_, id)| *id == 1)); // testing
}

#[test]
fn test_double_array_trie_stats() {
    let mut builder = DoubleArrayTrieBuilder::new();

    for i in 0..100u32 {
        builder.insert(&format!("token{}", i), i);
    }

    let trie = builder.build();
    let stats = trie.stats();

    assert!(stats.num_entries >= 100);
    assert!(stats.base_size > 0);
    assert!(stats.check_size > 0);
}

#[test]
fn test_double_array_trie_consistency() {
    // Both tries should give same results
    let tokens = vec!["hello", "help", "helper", "world", "work", "working"];

    let mut trie_builder = TrieBuilder::new();
    let mut da_builder = DoubleArrayTrieBuilder::new();

    for (i, token) in tokens.iter().enumerate() {
        trie_builder.insert(token, i as u32);
        da_builder.insert(token, i as u32);
    }

    let trie = trie_builder.build();
    let da_trie = da_builder.build();

    // Same lookups
    for (i, token) in tokens.iter().enumerate() {
        assert_eq!(trie.get(token), Some(i as u32));
        assert_eq!(da_trie.get(token), Some(i as u32));
    }
}

// =============================================================================
// Special Token Matcher Tests
// =============================================================================

#[test]
fn test_special_token_matcher_basic() {
    let mut builder = SpecialTokenMatcherBuilder::new();
    builder.add_token("[CLS]", 101);
    builder.add_token("[SEP]", 102);
    builder.add_token("[PAD]", 0);

    let matcher = builder.build();
    let matches: Vec<_> = matcher.find_matches("Hello [CLS] World [SEP]").collect();

    // Should find [CLS] and [SEP]
    assert!(matches.iter().any(|m| m.token_id == 101));
    assert!(matches.iter().any(|m| m.token_id == 102));
}

#[test]
fn test_special_token_matcher_leftmost_longest() {
    let mut builder = SpecialTokenMatcherBuilder::new();
    builder.add_token("[MASK]", 103);
    builder.add_token("[MASK123]", 104);

    let matcher = builder.build();
    let matches: Vec<_> = matcher.find_matches("Test [MASK123] here").collect();

    // Should match the longer one
    assert!(matches.iter().any(|m| m.token_id == 104));
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_special_token_matcher_overlapping() {
    let mut builder = SpecialTokenMatcherBuilder::new();
    builder.add_token("<|start|>", 1);
    builder.add_token("<|end|>", 2);

    let matcher = builder.build();
    let matches: Vec<_> = matcher.find_matches("<|start|>text<|end|>").collect();

    assert_eq!(matches.len(), 2);
}

#[test]
fn test_special_token_matcher_no_matches() {
    let mut builder = SpecialTokenMatcherBuilder::new();
    builder.add_token("[CLS]", 101);

    let matcher = builder.build();
    let matches: Vec<_> = matcher.find_matches("Hello World").collect();
    assert!(matches.is_empty());
}

#[test]
fn test_special_token_matcher_empty() {
    let builder = SpecialTokenMatcherBuilder::new();
    let matcher = builder.build();

    let matches: Vec<_> = matcher.find_matches("[CLS] test [SEP]").collect();
    assert!(matches.is_empty());
}

// =============================================================================
// Added Token Tests
// =============================================================================

#[test]
fn test_added_token_creation() {
    let token = AddedToken::new("[CUSTOM]", 999);
    assert_eq!(token.content, "[CUSTOM]");
    assert_eq!(token.id, 999);
}

#[test]
fn test_added_token_with_options() {
    let token = AddedToken::new("[CUSTOM]", 999)
        .single_word(true)
        .lstrip(true)
        .rstrip(true)
        .normalized(false)
        .special(true);

    assert!(token.single_word);
    assert!(token.lstrip);
    assert!(token.rstrip);
    assert!(!token.normalized);
    assert!(token.special);
}

// =============================================================================
// Integration Tests
// =============================================================================

#[test]
fn test_vocab_with_trie() {
    let tokens = vec![
        ("hello", 0u32),
        ("help", 1),
        ("helper", 2),
        ("world", 3),
    ];

    let mut vocab = Vocab::new();
    let mut trie_builder = TrieBuilder::new();

    for (token, id) in &tokens {
        vocab.insert(token.to_string(), *id);
        trie_builder.insert(token, *id);
    }

    let trie = trie_builder.build();

    // Vocab and trie should agree
    for (token, id) in &tokens {
        assert_eq!(vocab.get_id(token), Some(*id));
        assert_eq!(trie.get(token), Some(*id));
    }
}

#[test]
fn test_vocab_iteration() {
    let mut vocab = Vocab::new();

    vocab.insert("a".to_string(), 0);
    vocab.insert("b".to_string(), 1);
    vocab.insert("c".to_string(), 2);

    // Should be able to iterate over all tokens
    let mut count = 0;
    for id in 0..3 {
        if vocab.get_token(id).is_some() {
            count += 1;
        }
    }
    assert_eq!(count, 3);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_vocab_empty_token() {
    let mut vocab = Vocab::new();
    vocab.insert("".to_string(), 0);
    assert_eq!(vocab.get_id(""), Some(0));
}

#[test]
fn test_vocab_whitespace_token() {
    let mut vocab = Vocab::new();
    vocab.insert(" ".to_string(), 0);
    vocab.insert("\t".to_string(), 1);
    vocab.insert("\n".to_string(), 2);

    assert_eq!(vocab.get_id(" "), Some(0));
    assert_eq!(vocab.get_id("\t"), Some(1));
    assert_eq!(vocab.get_id("\n"), Some(2));
}

#[test]
fn test_trie_overlapping_tokens() {
    let mut builder = TrieBuilder::new();

    builder.insert("a", 0);
    builder.insert("ab", 1);
    builder.insert("abc", 2);
    builder.insert("abcd", 3);
    builder.insert("abcde", 4);

    let trie = builder.build();

    // All should be findable
    for i in 0..5 {
        let token = &"abcde"[..i + 1];
        assert_eq!(trie.get(token), Some(i as u32), "Failed for '{}'", token);
    }

    // Prefix search should find all
    let matches: Vec<_> = trie.common_prefix_search("abcdef").collect();
    assert_eq!(matches.len(), 5);
}

#[test]
fn test_vocab_large() {
    let mut vocab = Vocab::new();

    // Insert 10000 tokens
    for i in 0..10000u32 {
        vocab.insert(format!("token_{}", i), i);
    }

    assert_eq!(vocab.len(), 10000);

    // Random lookups should work
    assert_eq!(vocab.get_id("token_0"), Some(0));
    assert_eq!(vocab.get_id("token_5000"), Some(5000));
    assert_eq!(vocab.get_id("token_9999"), Some(9999));
}
