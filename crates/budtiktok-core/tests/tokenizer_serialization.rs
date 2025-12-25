use budtiktok_core::tokenizer::Tokenizer;
use budtiktok_core::wordpiece_hyper::{HyperWordPieceTokenizer, HyperConfig};
use budtiktok_core::unigram_fast::{UnigramFast, UnigramFastConfig};
use budtiktok_core::bpe_linear::{BpeTokenizer, BpeConfig, MergeRule};
use budtiktok_core::vocab::{Vocabulary, SpecialTokens};
use std::collections::HashMap;
use ahash::AHashMap;
use tempfile::tempdir;

#[test]
fn test_hyper_wordpiece_serialization_roundtrip() {
    let mut vocab_map = AHashMap::new();
    vocab_map.insert("[PAD]".to_string(), 0);
    vocab_map.insert("[UNK]".to_string(), 1);
    vocab_map.insert("[CLS]".to_string(), 2);
    vocab_map.insert("[SEP]".to_string(), 3);
    vocab_map.insert("hello".to_string(), 4);
    vocab_map.insert("world".to_string(), 5);

    let special_tokens = SpecialTokens {
        pad_token: Some("[PAD]".to_string()),
        unk_token: Some("[UNK]".to_string()),
        cls_token: Some("[CLS]".to_string()),
        sep_token: Some("[SEP]".to_string()),
        ..Default::default()
    };

    let vocab = Vocabulary::new(vocab_map, special_tokens);
    let config = HyperConfig::default();
    let tokenizer = HyperWordPieceTokenizer::new(vocab, config);

    let dir = tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    tokenizer.save(&path).unwrap();

    let loaded = HyperWordPieceTokenizer::from_pretrained(&path).unwrap();

    assert_eq!(tokenizer.vocab_size(), loaded.vocab_size());
    assert_eq!(tokenizer.token_to_id("hello"), loaded.token_to_id("hello"));
    assert_eq!(tokenizer.id_to_token(4), loaded.id_to_token(4));
}

#[test]
fn test_unigram_fast_serialization_roundtrip() {
    let mut vocab_map = AHashMap::new();
    vocab_map.insert("<unk>".to_string(), 0);
    vocab_map.insert("hello".to_string(), 1);
    vocab_map.insert("world".to_string(), 2);

    let special_tokens = SpecialTokens {
        unk_token: Some("<unk>".to_string()),
        ..Default::default()
    };

    let vocab = Vocabulary::new(vocab_map, special_tokens);
    let pieces = vec![
        budtiktok_core::unigram::UnigramPiece { token: "<unk>".to_string(), score: 0.0 },
        budtiktok_core::unigram::UnigramPiece { token: "hello".to_string(), score: -1.0 },
        budtiktok_core::unigram::UnigramPiece { token: "world".to_string(), score: -2.0 },
    ];
    let config = UnigramFastConfig::default();
    let tokenizer = UnigramFast::new(vocab, pieces, config);

    let dir = tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    tokenizer.save(&path).unwrap();

    let loaded = UnigramFast::from_pretrained(&path).unwrap();

    assert_eq!(tokenizer.vocab_size(), loaded.vocab_size());
    assert_eq!(tokenizer.token_to_id("hello"), loaded.token_to_id("hello"));
}

#[test]
fn test_bpe_serialization_roundtrip() {
    let mut vocab_map = AHashMap::new();
    vocab_map.insert("<unk>".to_string(), 0);
    vocab_map.insert("h".to_string(), 1);
    vocab_map.insert("e".to_string(), 2);
    vocab_map.insert("l".to_string(), 3);
    vocab_map.insert("o".to_string(), 4);
    vocab_map.insert("he".to_string(), 5);
    vocab_map.insert("hel".to_string(), 6);
    vocab_map.insert("hello".to_string(), 7);

    let special_tokens = SpecialTokens {
        unk_token: Some("<unk>".to_string()),
        ..Default::default()
    };

    let vocab = Vocabulary::new(vocab_map, special_tokens);
    let merges = vec![
        MergeRule { first: "h".to_string(), second: "e".to_string(), result: "he".to_string(), priority: 0 },
        MergeRule { first: "he".to_string(), second: "l".to_string(), result: "hel".to_string(), priority: 1 },
    ];
    let config = BpeConfig::default();
    let tokenizer = BpeTokenizer::new(vocab, merges, config);

    let dir = tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    tokenizer.save(&path).unwrap();

    let loaded = BpeTokenizer::from_pretrained(&path).unwrap();

    assert_eq!(tokenizer.vocab_size(), loaded.vocab_size());
    assert_eq!(tokenizer.token_to_id("he"), loaded.token_to_id("he"));
}
