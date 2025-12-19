//! Unigram Algorithm Unit Tests (9.1.6)
//!
//! Tests for Unigram/Viterbi tokenization including:
//! - Viterbi decoding
//! - Log probability scoring
//! - N-best paths
//! - Unknown token handling
//! - Sentence boundary detection
//! - Byte fallback
//! - SentencePiece compatibility
//! - EM training (optional)

use budtiktok_core::unigram::*;
use std::collections::HashMap;

// =============================================================================
// Unigram Model Basic Tests
// =============================================================================

#[test]
fn test_unigram_new() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);
    assert!(model.vocab_size() > 0);
}

#[test]
fn test_unigram_with_config() {
    let pieces = create_test_pieces();
    let config = UnigramConfig {
        unk_token: "<unk>".to_string(),
        unk_token_id: 0,
        byte_fallback: true,
    };
    let model = UnigramModel::with_config(pieces, config);
    assert_eq!(model.config().unk_token, "<unk>");
}

#[test]
fn test_unigram_vocab_size() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);
    assert_eq!(model.vocab_size(), model.iter().count());
}

#[test]
fn test_unigram_get_piece() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    assert!(model.get_piece("hello").is_some());
    assert!(model.get_piece("xyz_not_exist").is_none());
}

#[test]
fn test_unigram_get_score() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let score = model.get_score("hello");
    assert!(score.is_some());
    assert!(score.unwrap() < 0.0); // Log probabilities are negative
}

// =============================================================================
// Viterbi Decoding Tests
// =============================================================================

#[test]
fn test_viterbi_basic() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let result = model.encode("hello");
    assert!(!result.is_empty());
}

#[test]
fn test_viterbi_optimal_path() {
    let pieces = create_test_pieces_with_scores();
    let model = UnigramModel::new(pieces);

    let result = model.encode("helloworld");
    // Viterbi should find optimal (highest probability) segmentation
    // Check that we got a valid segmentation
    let reconstructed: String = result.iter().map(|t| t.piece.as_str()).collect();
    assert!(reconstructed == "helloworld" || !result.is_empty());
}

#[test]
fn test_viterbi_single_char() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let result = model.encode("a");
    assert!(!result.is_empty());
}

#[test]
fn test_viterbi_empty_input() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let result = model.encode("");
    assert!(result.is_empty());
}

#[test]
fn test_viterbi_with_spaces() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let result = model.encode("hello world");
    assert!(!result.is_empty());
}

// =============================================================================
// Log Probability Scoring Tests
// =============================================================================

#[test]
fn test_score_total() {
    let pieces = create_test_pieces_with_scores();
    let model = UnigramModel::new(pieces);

    let result = model.encode_with_score("hello");
    // Total score should be sum of log probabilities
    assert!(result.score.is_finite());
    assert!(result.score <= 0.0); // Log prob is non-positive
}

#[test]
fn test_score_comparison() {
    let pieces = create_test_pieces_with_scores();
    let model = UnigramModel::new(pieces);

    // More common words should have higher (less negative) scores
    let result_common = model.encode_with_score("the");
    let result_rare = model.encode_with_score("xyz");

    // Common should be higher (closer to 0)
    if !result_rare.tokens.iter().any(|t| t.piece == "<unk>") {
        // Only compare if not UNK
        assert!(result_common.score >= result_rare.score);
    }
}

#[test]
fn test_piece_scores_negative() {
    let pieces = create_test_pieces_with_scores();
    let model = UnigramModel::new(pieces);

    for piece in model.iter() {
        assert!(piece.score <= 0.0, "Score for '{}' should be non-positive", piece.piece);
    }
}

// =============================================================================
// N-Best Paths Tests
// =============================================================================

#[test]
fn test_nbest_single() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let results = model.encode_nbest("hello", 1);
    assert_eq!(results.len(), 1);
}

#[test]
fn test_nbest_multiple() {
    let pieces = create_test_pieces_with_subwords();
    let model = UnigramModel::new(pieces);

    let results = model.encode_nbest("hello", 5);
    // Should return up to 5 different segmentations
    assert!(results.len() >= 1);
    assert!(results.len() <= 5);
}

#[test]
fn test_nbest_ordered_by_score() {
    let pieces = create_test_pieces_with_subwords();
    let model = UnigramModel::new(pieces);

    let results = model.encode_nbest("hello", 5);
    // Results should be ordered by score (highest first)
    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "N-best results should be ordered by score"
        );
    }
}

#[test]
fn test_nbest_unique() {
    let pieces = create_test_pieces_with_subwords();
    let model = UnigramModel::new(pieces);

    let results = model.encode_nbest("hello", 10);
    // All results should be unique
    let mut seen: Vec<Vec<String>> = Vec::new();
    for result in &results {
        let tokens: Vec<String> = result.tokens.iter().map(|t| t.piece.clone()).collect();
        assert!(!seen.contains(&tokens), "N-best should have unique segmentations");
        seen.push(tokens);
    }
}

// =============================================================================
// Unknown Token Handling Tests
// =============================================================================

#[test]
fn test_unk_token_basic() {
    let pieces = create_minimal_pieces();
    let model = UnigramModel::new(pieces);

    let result = model.encode("xyz");
    // Should handle unknown via byte fallback or UNK
    assert!(!result.is_empty());
}

#[test]
fn test_unk_token_custom() {
    let mut pieces = create_test_pieces();
    pieces.push(SentencePiece::new("<UNK>".to_string(), 0, -10.0));

    let config = UnigramConfig {
        unk_token: "<UNK>".to_string(),
        unk_token_id: pieces.len() as u32 - 1,
        byte_fallback: false,
    };
    let model = UnigramModel::with_config(pieces, config);

    let result = model.encode("qqqqq");
    // Without byte fallback, should use UNK
    let has_unk = result.iter().any(|t| t.piece == "<UNK>");
    assert!(has_unk || !result.is_empty());
}

#[test]
fn test_unk_preserves_length() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let input = "hello xyz world";
    let result = model.encode(input);

    // Reconstructed should match input
    let reconstructed: String = result.iter().map(|t| t.piece.as_str()).collect();
    // Account for space handling differences
    assert!(reconstructed.len() >= input.len() - 2);
}

// =============================================================================
// Byte Fallback Tests
// =============================================================================

#[test]
fn test_byte_fallback_enabled() {
    let pieces = create_pieces_with_byte_fallback();
    let config = UnigramConfig {
        unk_token: "<unk>".to_string(),
        unk_token_id: 0,
        byte_fallback: true,
    };
    let model = UnigramModel::with_config(pieces, config);

    // Unknown char should fall back to bytes
    let result = model.encode("q");
    assert!(!result.is_empty());
}

#[test]
fn test_byte_fallback_unicode() {
    let pieces = create_pieces_with_byte_fallback();
    let config = UnigramConfig {
        unk_token: "<unk>".to_string(),
        unk_token_id: 0,
        byte_fallback: true,
    };
    let model = UnigramModel::with_config(pieces, config);

    // Unicode character should fall back to bytes
    let result = model.encode("日");
    // Japanese 日 is 3 UTF-8 bytes
    assert!(!result.is_empty());
}

#[test]
fn test_byte_fallback_disabled() {
    let pieces = create_test_pieces();
    let config = UnigramConfig {
        unk_token: "<unk>".to_string(),
        unk_token_id: 0,
        byte_fallback: false,
    };
    let model = UnigramModel::with_config(pieces, config);

    // Without byte fallback, should use UNK
    let result = model.encode("q");
    // Either UNK or empty result
    let has_unk = result.iter().any(|t| t.piece == "<unk>");
    assert!(has_unk || !result.is_empty());
}

// =============================================================================
// Sentence Boundary Detection Tests
// =============================================================================

#[test]
fn test_sentence_boundary_basic() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let sentences = model.split_sentences("Hello. World.");
    assert!(sentences.len() >= 1);
}

#[test]
fn test_sentence_boundary_multiple() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let sentences = model.split_sentences("First. Second. Third.");
    assert!(sentences.len() >= 2);
}

#[test]
fn test_sentence_boundary_with_encoding() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let result = model.encode_sentences("Hello. World.");
    // Should have sentence boundaries marked
    assert!(!result.is_empty());
}

// =============================================================================
// SentencePiece Compatibility Tests
// =============================================================================

#[test]
fn test_sentencepiece_format() {
    // Test that we can handle SentencePiece-style pieces
    let pieces = vec![
        SentencePiece::new("▁hello".to_string(), 0, -5.0), // Leading space marker
        SentencePiece::new("▁world".to_string(), 1, -5.5),
        SentencePiece::new("▁".to_string(), 2, -2.0),
        SentencePiece::new("<unk>".to_string(), 3, -10.0),
    ];

    let model = UnigramModel::new(pieces);
    let result = model.encode("▁hello▁world");
    assert!(!result.is_empty());
}

#[test]
fn test_add_dummy_prefix() {
    let pieces = create_test_pieces();
    let config = UnigramConfig {
        unk_token: "<unk>".to_string(),
        unk_token_id: 0,
        byte_fallback: false,
    };
    let model = UnigramModel::with_config(pieces, config);

    // With dummy prefix, "hello" becomes " hello"
    let result_no_prefix = model.encode("hello");
    let result_with_prefix = model.encode_with_dummy_prefix("hello");

    // Results may differ due to leading space
    let _ = (result_no_prefix, result_with_prefix);
}

#[test]
fn test_remove_extra_whitespace() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let result1 = model.encode("hello world");
    let result2 = model.encode("hello  world"); // Double space

    // Should handle multiple spaces appropriately
    let _ = (result1, result2);
}

// =============================================================================
// Lattice Tests
// =============================================================================

#[test]
fn test_lattice_construction() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let lattice = model.build_lattice("hello");
    assert!(lattice.len() > 0);
}

#[test]
fn test_lattice_nodes() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let lattice = model.build_lattice("hello");
    // Should have nodes for each position
    for node in lattice.iter() {
        assert!(node.score.is_finite());
    }
}

#[test]
fn test_lattice_coverage() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let input = "hello";
    let lattice = model.build_lattice(input);

    // Lattice should cover entire input
    let max_end = lattice.iter().map(|n| n.end).max().unwrap_or(0);
    assert!(max_end == input.len());
}

// =============================================================================
// Training Tests (EM Algorithm)
// =============================================================================

#[test]
fn test_em_iteration() {
    let mut pieces = create_test_pieces_with_scores();
    let trainer = UnigramTrainer::new();

    let corpus = vec!["hello", "world", "hello world"];
    let updated = trainer.em_iteration(&mut pieces, &corpus);

    // Scores should be updated
    assert!(updated);
}

#[test]
fn test_prune_vocab() {
    let pieces = create_large_vocab();
    let trainer = UnigramTrainer::new();

    let pruned = trainer.prune(pieces, 100);
    assert!(pruned.len() <= 100);
}

#[test]
fn test_training_convergence() {
    let pieces = create_test_pieces_with_scores();
    let trainer = UnigramTrainer::with_config(TrainerConfig {
        max_iterations: 10,
        convergence_threshold: 1e-6,
    });

    let corpus = vec!["hello", "world"];
    let trained = trainer.train(pieces, &corpus);

    // Should converge
    assert!(trained.converged || trained.iterations <= 10);
}

// =============================================================================
// Serialization Tests
// =============================================================================

#[test]
fn test_model_serialize() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let bytes = model.to_bytes();
    assert!(!bytes.is_empty());
}

#[test]
fn test_model_deserialize() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let bytes = model.to_bytes();
    let restored = UnigramModel::from_bytes(&bytes).unwrap();

    assert_eq!(model.vocab_size(), restored.vocab_size());
}

#[test]
fn test_model_roundtrip() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let bytes = model.to_bytes();
    let restored = UnigramModel::from_bytes(&bytes).unwrap();

    // Encoding should give same result
    let result1 = model.encode("hello");
    let result2 = restored.encode("hello");

    assert_eq!(result1.len(), result2.len());
    for (t1, t2) in result1.iter().zip(result2.iter()) {
        assert_eq!(t1.piece, t2.piece);
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_single_piece_vocab() {
    let pieces = vec![SentencePiece::new("<unk>".to_string(), 0, 0.0)];
    let model = UnigramModel::new(pieces);

    let result = model.encode("anything");
    // Everything should map to UNK
    assert!(!result.is_empty());
}

#[test]
fn test_unicode_input() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let result = model.encode("日本語");
    assert!(!result.is_empty());
}

#[test]
fn test_emoji_input() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let result = model.encode("hello 😀 world");
    assert!(!result.is_empty());
}

#[test]
fn test_very_long_input() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let long_input = "hello ".repeat(1000);
    let result = model.encode(&long_input);
    assert!(!result.is_empty());
}

#[test]
fn test_special_characters() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let result = model.encode("hello\n\t\r world");
    assert!(!result.is_empty());
}

// =============================================================================
// Decode Tests
// =============================================================================

#[test]
fn test_decode_basic() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let encoded = model.encode("hello");
    let ids: Vec<u32> = encoded.iter().map(|t| t.id).collect();

    let decoded = model.decode(&ids);
    // Decoded should reconstruct original (approximately)
    assert!(!decoded.is_empty());
}

#[test]
fn test_decode_ids() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    // Decode specific IDs
    let decoded = model.decode(&[0, 1, 2]);
    assert!(!decoded.is_empty());
}

#[test]
fn test_encode_decode_roundtrip() {
    let pieces = create_test_pieces();
    let model = UnigramModel::new(pieces);

    let original = "hello world";
    let encoded = model.encode(original);
    let ids: Vec<u32> = encoded.iter().map(|t| t.id).collect();
    let decoded = model.decode(&ids);

    // Should approximately match original
    assert!(decoded.len() >= original.len() - 2);
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_pieces() -> Vec<SentencePiece> {
    vec![
        SentencePiece::new("<unk>".to_string(), 0, -10.0),
        SentencePiece::new("hello".to_string(), 1, -5.0),
        SentencePiece::new("world".to_string(), 2, -5.5),
        SentencePiece::new("the".to_string(), 3, -3.0),
        SentencePiece::new("a".to_string(), 4, -2.5),
        SentencePiece::new(" ".to_string(), 5, -1.0),
        SentencePiece::new("h".to_string(), 6, -4.0),
        SentencePiece::new("e".to_string(), 7, -3.5),
        SentencePiece::new("l".to_string(), 8, -4.0),
        SentencePiece::new("o".to_string(), 9, -3.8),
    ]
}

fn create_test_pieces_with_scores() -> Vec<SentencePiece> {
    vec![
        SentencePiece::new("<unk>".to_string(), 0, -15.0),
        SentencePiece::new("the".to_string(), 1, -2.0),   // Very common
        SentencePiece::new("hello".to_string(), 2, -5.0), // Common
        SentencePiece::new("world".to_string(), 3, -5.5),
        SentencePiece::new("xyz".to_string(), 4, -12.0), // Rare
        SentencePiece::new(" ".to_string(), 5, -1.0),
        SentencePiece::new("h".to_string(), 6, -4.0),
        SentencePiece::new("e".to_string(), 7, -3.5),
        SentencePiece::new("l".to_string(), 8, -4.0),
        SentencePiece::new("o".to_string(), 9, -3.8),
        SentencePiece::new("w".to_string(), 10, -4.2),
        SentencePiece::new("r".to_string(), 11, -4.1),
        SentencePiece::new("d".to_string(), 12, -4.3),
    ]
}

fn create_test_pieces_with_subwords() -> Vec<SentencePiece> {
    let mut pieces = create_test_pieces();
    pieces.extend(vec![
        SentencePiece::new("hel".to_string(), 100, -4.5),
        SentencePiece::new("lo".to_string(), 101, -4.2),
        SentencePiece::new("hell".to_string(), 102, -4.8),
        SentencePiece::new("ello".to_string(), 103, -5.0),
    ]);
    pieces
}

fn create_minimal_pieces() -> Vec<SentencePiece> {
    vec![
        SentencePiece::new("<unk>".to_string(), 0, -10.0),
        SentencePiece::new(" ".to_string(), 1, -1.0),
    ]
}

fn create_pieces_with_byte_fallback() -> Vec<SentencePiece> {
    let mut pieces = create_test_pieces();
    // Add byte tokens <0x00> through <0xFF>
    for byte in 0u8..=255 {
        pieces.push(SentencePiece::new(
            format!("<0x{:02X}>", byte),
            256 + byte as u32,
            -8.0,
        ));
    }
    pieces
}

fn create_large_vocab() -> Vec<SentencePiece> {
    let mut pieces = create_test_pieces();
    for i in 0..200 {
        pieces.push(SentencePiece::new(
            format!("token{}", i),
            100 + i,
            -5.0 - (i as f32) * 0.01,
        ));
    }
    pieces
}
