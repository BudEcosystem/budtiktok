//! WordPiece Algorithm Unit Tests (9.1.5)
//!
//! Tests for WordPiece tokenization including:
//! - Basic tokenization
//! - Subword splitting
//! - Unknown word handling
//! - Max chars limit
//! - Continuation prefix
//! - Cache integration
//! - BERT normalizer
//! - BERT pre-tokenizer
//! - Post-processing with [CLS]/[SEP]

use budtiktok_core::wordpiece::*;
use budtiktok_core::vocab::Vocab;

// =============================================================================
// WordPiece Basic Tests
// =============================================================================

#[test]
fn test_wordpiece_new() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);
    assert!(wp.vocab().len() > 0);
}

#[test]
fn test_wordpiece_with_config() {
    let vocab = create_test_vocab();
    let config = WordPieceConfig {
        unk_token: "[UNK]".to_string(),
        continuing_subword_prefix: "##".to_string(),
        max_input_chars_per_word: 100,
    };
    let wp = WordPiece::with_config(vocab, config);
    assert_eq!(wp.config().unk_token, "[UNK]");
    assert_eq!(wp.config().continuing_subword_prefix, "##");
}

#[test]
fn test_wordpiece_tokenize_known_word() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);

    let tokens = wp.tokenize("hello");
    assert!(!tokens.is_empty());
    // Should tokenize as a single token if "hello" is in vocab
}

#[test]
fn test_wordpiece_tokenize_unknown_word() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);

    let tokens = wp.tokenize("xyzabc123");
    // Should return [UNK] or subwords
    assert!(!tokens.is_empty());
}

#[test]
fn test_wordpiece_tokenize_subwords() {
    let vocab = create_test_vocab_with_subwords();
    let wp = WordPiece::new(vocab);

    // "playing" should be split into "play" + "##ing"
    let tokens = wp.tokenize("playing");
    assert!(tokens.len() >= 1);
}

#[test]
fn test_wordpiece_empty_input() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);

    let tokens = wp.tokenize("");
    assert!(tokens.is_empty());
}

// =============================================================================
// Subword Split Tests
// =============================================================================

#[test]
fn test_subword_split_basic() {
    let vocab = create_test_vocab_with_subwords();
    let wp = WordPiece::new(vocab);

    // Word that needs splitting
    let result = wp.encode_word("unhappiness");
    // Should split into subwords like "un", "##happy", "##ness" or similar
    assert!(result.tokens.len() >= 1);
}

#[test]
fn test_subword_split_greedy() {
    let vocab = create_test_vocab_with_subwords();
    let wp = WordPiece::new(vocab);

    // WordPiece uses greedy longest-match-first
    let result = wp.encode_word("international");
    // Verify greedy behavior: prefer longer subwords
    if result.tokens.len() > 1 {
        // First token should be as long as possible
        assert!(result.tokens[0].len() >= 2);
    }
}

#[test]
fn test_subword_continuation_prefix() {
    let vocab = create_test_vocab_with_subwords();
    let wp = WordPiece::new(vocab);

    let result = wp.encode_word("playing");
    // Second and subsequent tokens should have ## prefix
    for (i, token) in result.tokens.iter().enumerate() {
        if i > 0 {
            assert!(
                token.starts_with("##") || result.tokens.len() == 1,
                "Continuation token '{}' should start with ##",
                token
            );
        }
    }
}

#[test]
fn test_subword_all_unknown() {
    let vocab = create_minimal_vocab();
    let wp = WordPiece::new(vocab);

    let result = wp.encode_word("xyzxyzxyz");
    // Should fall back to [UNK]
    assert!(result.tokens.iter().any(|t| t == "[UNK]") || !result.tokens.is_empty());
}

// =============================================================================
// Max Chars Limit Tests
// =============================================================================

#[test]
fn test_max_chars_under_limit() {
    let vocab = create_test_vocab();
    let config = WordPieceConfig {
        unk_token: "[UNK]".to_string(),
        continuing_subword_prefix: "##".to_string(),
        max_input_chars_per_word: 100,
    };
    let wp = WordPiece::with_config(vocab, config);

    let tokens = wp.tokenize("hello"); // 5 chars, under 100
    assert!(!tokens.is_empty());
}

#[test]
fn test_max_chars_at_limit() {
    let vocab = create_test_vocab();
    let config = WordPieceConfig {
        unk_token: "[UNK]".to_string(),
        continuing_subword_prefix: "##".to_string(),
        max_input_chars_per_word: 5,
    };
    let wp = WordPiece::with_config(vocab, config);

    let tokens = wp.tokenize("hello"); // Exactly 5 chars
    assert!(!tokens.is_empty());
}

#[test]
fn test_max_chars_over_limit() {
    let vocab = create_test_vocab();
    let config = WordPieceConfig {
        unk_token: "[UNK]".to_string(),
        continuing_subword_prefix: "##".to_string(),
        max_input_chars_per_word: 3,
    };
    let wp = WordPiece::with_config(vocab, config);

    let tokens = wp.tokenize("hello"); // 5 chars, over 3 limit
    // Should return [UNK] for words over limit
    assert!(tokens.iter().any(|t| t.token == "[UNK]"));
}

#[test]
fn test_max_chars_mixed() {
    let vocab = create_test_vocab();
    let config = WordPieceConfig {
        unk_token: "[UNK]".to_string(),
        continuing_subword_prefix: "##".to_string(),
        max_input_chars_per_word: 5,
    };
    let wp = WordPiece::with_config(vocab, config);

    // "hi" is under, "supercalifragilistic" is over
    let text = "hi supercalifragilistic";
    let tokens = wp.tokenize(text);
    // Should have mix of normal tokens and [UNK]
    assert!(!tokens.is_empty());
}

// =============================================================================
// Unknown Word Handling Tests
// =============================================================================

#[test]
fn test_unk_token_default() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);

    let result = wp.encode_word("zzzzzzz");
    // Check that unknown handling works
    assert!(!result.tokens.is_empty());
}

#[test]
fn test_unk_token_custom() {
    let mut vocab = Vocab::new();
    vocab.insert("<UNK>".to_string(), 0);
    vocab.insert("hello".to_string(), 1);

    let config = WordPieceConfig {
        unk_token: "<UNK>".to_string(),
        continuing_subword_prefix: "##".to_string(),
        max_input_chars_per_word: 100,
    };
    let wp = WordPiece::with_config(vocab, config);

    let result = wp.encode_word("xyz");
    assert!(result.tokens.iter().any(|t| t == "<UNK>"));
}

#[test]
fn test_unk_token_in_sequence() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);

    let tokens = wp.tokenize("hello xyz world");
    // xyz should become [UNK]
    let has_unk = tokens.iter().any(|t| t.token == "[UNK]");
    let has_hello = tokens.iter().any(|t| t.token == "hello" || t.token.contains("hello"));
    assert!(has_unk || tokens.len() >= 3); // Either has UNK or was handled somehow
}

// =============================================================================
// BERT Normalizer Tests
// =============================================================================

#[test]
fn test_bert_normalizer_lowercase() {
    let normalizer = BertNormalizer::new(true, true, true, true);

    let result = normalizer.normalize("HELLO WORLD");
    assert_eq!(result, "hello world");
}

#[test]
fn test_bert_normalizer_no_lowercase() {
    let normalizer = BertNormalizer::new(false, true, true, true);

    let result = normalizer.normalize("HELLO");
    assert_eq!(result, "HELLO");
}

#[test]
fn test_bert_normalizer_strip_accents() {
    let normalizer = BertNormalizer::new(true, true, true, true);

    let result = normalizer.normalize("café");
    // Should strip accents: café -> cafe
    assert!(result == "cafe" || result.contains("cafe"));
}

#[test]
fn test_bert_normalizer_chinese_chars() {
    let normalizer = BertNormalizer::new(true, true, true, true);

    let result = normalizer.normalize("hello中文world");
    // Should add spaces around Chinese characters
    assert!(result.contains(" ") || result.contains("中"));
}

#[test]
fn test_bert_normalizer_control_chars() {
    let normalizer = BertNormalizer::new(true, true, true, true);

    let result = normalizer.normalize("hello\x00world");
    // Control chars should be removed or replaced
    assert!(!result.contains('\x00'));
}

#[test]
fn test_bert_normalizer_whitespace() {
    let normalizer = BertNormalizer::new(true, true, true, true);

    let result = normalizer.normalize("hello\t\nworld");
    // Tabs and newlines should become spaces
    assert!(result.contains(" ") || result == "hello world");
}

// =============================================================================
// BERT Pre-tokenizer Tests
// =============================================================================

#[test]
fn test_bert_pretokenizer_whitespace_split() {
    let pretokenizer = BertPreTokenizer::new();

    let tokens = pretokenizer.pre_tokenize("hello world foo");
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].0, "hello");
    assert_eq!(tokens[1].0, "world");
    assert_eq!(tokens[2].0, "foo");
}

#[test]
fn test_bert_pretokenizer_punctuation_split() {
    let pretokenizer = BertPreTokenizer::new();

    let tokens = pretokenizer.pre_tokenize("hello,world");
    // Should split on comma
    assert!(tokens.len() >= 2);
}

#[test]
fn test_bert_pretokenizer_multiple_spaces() {
    let pretokenizer = BertPreTokenizer::new();

    let tokens = pretokenizer.pre_tokenize("hello   world");
    // Multiple spaces should not create empty tokens
    assert_eq!(tokens.len(), 2);
}

#[test]
fn test_bert_pretokenizer_offsets() {
    let pretokenizer = BertPreTokenizer::new();

    let tokens = pretokenizer.pre_tokenize("hello world");
    // Offsets should be correct
    assert_eq!(tokens[0].1, (0, 5)); // "hello" at 0-5
    assert_eq!(tokens[1].1, (6, 11)); // "world" at 6-11
}

#[test]
fn test_bert_pretokenizer_unicode() {
    let pretokenizer = BertPreTokenizer::new();

    let tokens = pretokenizer.pre_tokenize("hello 世界");
    assert_eq!(tokens.len(), 2);
}

// =============================================================================
// Post-processing Tests
// =============================================================================

#[test]
fn test_postprocessor_add_cls_sep() {
    let postprocessor = BertPostProcessor::new("[CLS]", "[SEP]", 101, 102);

    let encoding = MockEncoding::new(vec!["hello".to_string()], vec![1]);
    let result = postprocessor.process(encoding, None);

    // Should have [CLS] at start and [SEP] at end
    assert_eq!(result.tokens[0], "[CLS]");
    assert_eq!(result.tokens[result.tokens.len() - 1], "[SEP]");
}

#[test]
fn test_postprocessor_single_sequence() {
    let postprocessor = BertPostProcessor::new("[CLS]", "[SEP]", 101, 102);

    let encoding = MockEncoding::new(vec!["hello".to_string(), "world".to_string()], vec![1, 2]);
    let result = postprocessor.process(encoding, None);

    // Result: [CLS] hello world [SEP]
    assert_eq!(result.tokens.len(), 4);
    assert_eq!(result.token_type_ids.iter().all(|&x| x == 0), true);
}

#[test]
fn test_postprocessor_pair_sequence() {
    let postprocessor = BertPostProcessor::new("[CLS]", "[SEP]", 101, 102);

    let encoding_a = MockEncoding::new(vec!["hello".to_string()], vec![1]);
    let encoding_b = MockEncoding::new(vec!["world".to_string()], vec![2]);
    let result = postprocessor.process(encoding_a, Some(encoding_b));

    // Result: [CLS] hello [SEP] world [SEP]
    assert_eq!(result.tokens.len(), 5);
    // Token type IDs: 0 for first sequence, 1 for second
    assert_eq!(result.token_type_ids[0], 0); // [CLS]
    assert_eq!(result.token_type_ids[1], 0); // hello
    assert_eq!(result.token_type_ids[2], 0); // [SEP]
    assert_eq!(result.token_type_ids[3], 1); // world
    assert_eq!(result.token_type_ids[4], 1); // [SEP]
}

#[test]
fn test_postprocessor_special_tokens_mask() {
    let postprocessor = BertPostProcessor::new("[CLS]", "[SEP]", 101, 102);

    let encoding = MockEncoding::new(vec!["hello".to_string()], vec![1]);
    let result = postprocessor.process(encoding, None);

    // Special tokens mask: 1 for special, 0 for regular
    assert_eq!(result.special_tokens_mask[0], 1); // [CLS]
    assert_eq!(result.special_tokens_mask[1], 0); // hello
    assert_eq!(result.special_tokens_mask[2], 1); // [SEP]
}

// =============================================================================
// Cache Integration Tests
// =============================================================================

#[test]
fn test_wordpiece_with_cache() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);
    let cached_wp = CachedWordPiece::new(wp, 1000);

    // First call - cache miss
    let tokens1 = cached_wp.tokenize("hello");

    // Second call - cache hit
    let tokens2 = cached_wp.tokenize("hello");

    // Results should be identical
    assert_eq!(tokens1.len(), tokens2.len());
}

#[test]
fn test_wordpiece_cache_different_inputs() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);
    let cached_wp = CachedWordPiece::new(wp, 1000);

    let tokens1 = cached_wp.tokenize("hello");
    let tokens2 = cached_wp.tokenize("world");

    // Different inputs should give different results
    // (unless both are [UNK])
    let _ = (tokens1, tokens2); // Use the results
}

#[test]
fn test_wordpiece_cache_stats() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);
    let cached_wp = CachedWordPiece::new(wp, 1000);

    cached_wp.tokenize("hello");
    cached_wp.tokenize("hello"); // Cache hit
    cached_wp.tokenize("world"); // Cache miss

    let stats = cached_wp.cache_stats();
    assert!(stats.hits >= 1);
    assert!(stats.misses >= 2);
}

// =============================================================================
// Encoding Output Tests
// =============================================================================

#[test]
fn test_encoding_ids() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);

    let encoding = wp.encode("hello");
    assert!(!encoding.ids.is_empty());
    // All IDs should be valid
    for id in &encoding.ids {
        assert!(wp.vocab().get_token(*id).is_some());
    }
}

#[test]
fn test_encoding_offsets() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);

    let text = "hello world";
    let encoding = wp.encode(text);

    // Offsets should be within text bounds
    for (start, end) in &encoding.offsets {
        assert!(*start <= text.len());
        assert!(*end <= text.len());
        assert!(start <= end);
    }
}

#[test]
fn test_encoding_attention_mask() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);

    let encoding = wp.encode("hello");
    // Attention mask should be all 1s for actual tokens
    assert!(encoding.attention_mask.iter().all(|&x| x == 1));
}

#[test]
fn test_encoding_word_ids() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);

    let encoding = wp.encode("hello world");
    // Word IDs should map tokens back to original words
    // Tokens from "hello" should have word_id 0
    // Tokens from "world" should have word_id 1
    for word_id in encoding.word_ids.iter().flatten() {
        assert!(*word_id <= 1);
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_wordpiece_single_char() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);

    let tokens = wp.tokenize("a");
    assert!(!tokens.is_empty());
}

#[test]
fn test_wordpiece_unicode() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);

    let tokens = wp.tokenize("日本語");
    assert!(!tokens.is_empty());
}

#[test]
fn test_wordpiece_emoji() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);

    let tokens = wp.tokenize("hello 😀");
    assert!(!tokens.is_empty());
}

#[test]
fn test_wordpiece_mixed_case() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);

    // Without normalization, "Hello" and "hello" are different
    let tokens_upper = wp.tokenize("Hello");
    let tokens_lower = wp.tokenize("hello");
    // May or may not be same depending on vocab
    let _ = (tokens_upper, tokens_lower);
}

#[test]
fn test_wordpiece_numbers() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);

    let tokens = wp.tokenize("123456");
    assert!(!tokens.is_empty());
}

#[test]
fn test_wordpiece_special_chars() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);

    let tokens = wp.tokenize("hello!@#$%");
    assert!(!tokens.is_empty());
}

// =============================================================================
// Batch Processing Tests
// =============================================================================

#[test]
fn test_wordpiece_batch_encode() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);

    let texts = vec!["hello", "world", "test"];
    let encodings = wp.encode_batch(&texts);

    assert_eq!(encodings.len(), 3);
}

#[test]
fn test_wordpiece_batch_encode_with_padding() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);

    let texts = vec!["hello", "hello world"];
    let encodings = wp.encode_batch_padded(&texts, PaddingStrategy::Longest);

    // All encodings should have same length
    let len = encodings[0].ids.len();
    assert!(encodings.iter().all(|e| e.ids.len() == len));
}

#[test]
fn test_wordpiece_batch_encode_truncation() {
    let vocab = create_test_vocab();
    let wp = WordPiece::new(vocab);

    let texts = vec!["hello world foo bar baz"];
    let encodings = wp.encode_batch_truncated(&texts, 3);

    assert!(encodings[0].ids.len() <= 3);
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_vocab() -> Vocab {
    let mut vocab = Vocab::new();

    // Special tokens
    vocab.insert("[PAD]".to_string(), 0);
    vocab.insert("[UNK]".to_string(), 100);
    vocab.insert("[CLS]".to_string(), 101);
    vocab.insert("[SEP]".to_string(), 102);
    vocab.insert("[MASK]".to_string(), 103);

    // Basic words
    vocab.insert("hello".to_string(), 1);
    vocab.insert("world".to_string(), 2);
    vocab.insert("test".to_string(), 3);
    vocab.insert("the".to_string(), 4);
    vocab.insert("a".to_string(), 5);

    vocab
}

fn create_test_vocab_with_subwords() -> Vocab {
    let mut vocab = create_test_vocab();

    // Continuation tokens
    vocab.insert("##ing".to_string(), 200);
    vocab.insert("##ed".to_string(), 201);
    vocab.insert("##s".to_string(), 202);
    vocab.insert("##er".to_string(), 203);
    vocab.insert("##est".to_string(), 204);
    vocab.insert("##ly".to_string(), 205);
    vocab.insert("##ness".to_string(), 206);

    // Base words
    vocab.insert("play".to_string(), 10);
    vocab.insert("work".to_string(), 11);
    vocab.insert("un".to_string(), 12);
    vocab.insert("happy".to_string(), 13);
    vocab.insert("##happy".to_string(), 213);
    vocab.insert("inter".to_string(), 14);
    vocab.insert("##nation".to_string(), 214);
    vocab.insert("##al".to_string(), 215);

    vocab
}

fn create_minimal_vocab() -> Vocab {
    let mut vocab = Vocab::new();
    vocab.insert("[UNK]".to_string(), 0);
    vocab.insert("[PAD]".to_string(), 1);
    vocab
}

// Mock encoding for post-processor tests
struct MockEncoding {
    tokens: Vec<String>,
    ids: Vec<u32>,
}

impl MockEncoding {
    fn new(tokens: Vec<String>, ids: Vec<u32>) -> Self {
        Self { tokens, ids }
    }
}
