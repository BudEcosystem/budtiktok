//! Fuzzing Target: Tokenizer Input (9.5.1)
//!
//! Fuzzes arbitrary UTF-8 strings through the tokenization pipeline.
//! Ensures no panics on any valid or malformed input.
//!
//! Run with: cargo +nightly fuzz run fuzz_tokenize

#![no_main]

use libfuzzer_sys::fuzz_target;
use budtiktok_core::unicode::{is_ascii_fast, normalize, NormalizationForm};
use budtiktok_core::unicode::{is_whitespace, is_punctuation, is_cjk_character, is_control};
use budtiktok_core::unicode::get_category_flags;
use budtiktok_core::Encoding;

fuzz_target!(|data: &[u8]| {
    // Only test valid UTF-8 strings
    if let Ok(text) = std::str::from_utf8(data) {
        fuzz_text(text);
    }

    // Also test the raw bytes for functions that accept bytes
    fuzz_bytes(data);
});

/// Fuzz text-based operations
fn fuzz_text(text: &str) {
    // 1. Test ASCII detection
    // Should never panic regardless of input
    let _ = is_ascii_fast(text);

    // 2. Test normalization forms
    // Should never panic regardless of input
    let _ = normalize(text, NormalizationForm::NFC);
    let _ = normalize(text, NormalizationForm::NFD);
    let _ = normalize(text, NormalizationForm::NFKC);
    let _ = normalize(text, NormalizationForm::NFKD);

    // 3. Test character classification on each character
    // Should never panic for any Unicode character
    for ch in text.chars() {
        let _ = is_whitespace(ch);
        let _ = is_punctuation(ch);
        let _ = is_cjk_character(ch);
        let _ = is_control(ch);
        let _ = get_category_flags(ch);
    }

    // 4. Test encoding operations
    // Should never panic
    let mut encoding = Encoding::new();
    for (i, token) in text.split_whitespace().enumerate().take(100) {
        encoding.push(
            i as u32,
            token.to_string(),
            (0, token.len()),
            Some(i as u32),
            Some(0),
            false,
        );
    }

    // Test truncation at various lengths
    if encoding.len() > 0 {
        let _ = encoding.clone();
        let mut truncated = encoding.clone();
        truncated.truncate(encoding.len() / 2, 0);
    }

    // Test padding
    let mut padded = encoding.clone();
    padded.pad(encoding.len() + 10, 0, "[PAD]");
}

/// Fuzz byte-based operations
fn fuzz_bytes(data: &[u8]) {
    use budtiktok_core::swar::{
        has_zero_byte, has_byte, broadcast_byte, has_non_ascii,
        is_all_ascii_unrolled, to_lowercase_branchless, to_uppercase_branchless,
        is_ascii_whitespace_branchless, is_ascii_alphanumeric_branchless,
    };

    // Test SWAR operations on u64 chunks
    for chunk in data.chunks(8) {
        if chunk.len() == 8 {
            let word = u64::from_ne_bytes(chunk.try_into().unwrap());

            // These should never panic
            let _ = has_zero_byte(word);
            let _ = has_non_ascii(word);

            // Test with various byte targets
            for target in [0u8, 32, 65, 127, 255] {
                let _ = has_byte(word, target);
            }
        }
    }

    // Test broadcast
    for &byte in data.iter().take(10) {
        let _ = broadcast_byte(byte);
    }

    // Test branchless operations on individual bytes
    for &byte in data.iter().take(256) {
        let _ = to_lowercase_branchless(byte);
        let _ = to_uppercase_branchless(byte);
        let _ = is_ascii_whitespace_branchless(byte);
        let _ = is_ascii_alphanumeric_branchless(byte);
    }

    // Test unrolled ASCII check
    let _ = is_all_ascii_unrolled(data);
}
