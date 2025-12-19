//! Fuzzing Target: Unicode Processing
//!
//! Fuzzes Unicode normalization, character classification, and related
//! operations to ensure no panics on any valid Unicode input.
//!
//! Run with: cargo +nightly fuzz run fuzz_unicode

#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use budtiktok_core::unicode::{
    is_ascii_fast, normalize, NormalizationForm,
    is_whitespace, is_punctuation, is_cjk_character, is_control,
    get_category_flags,
};

/// Structured input for more targeted fuzzing
#[derive(Arbitrary, Debug)]
struct UnicodeInput {
    text: String,
    form: u8,
    operations: Vec<Operation>,
}

#[derive(Arbitrary, Debug)]
enum Operation {
    Normalize,
    CheckAscii,
    ClassifyChars,
    GetFlags,
}

fuzz_target!(|input: UnicodeInput| {
    let text = &input.text;

    // Limit input size to prevent memory exhaustion
    if text.len() > 100_000 {
        return;
    }

    // Run requested operations
    for op in &input.operations {
        match op {
            Operation::Normalize => {
                let form = match input.form % 4 {
                    0 => NormalizationForm::NFC,
                    1 => NormalizationForm::NFD,
                    2 => NormalizationForm::NFKC,
                    _ => NormalizationForm::NFKD,
                };
                let _ = normalize(text, form);
            }
            Operation::CheckAscii => {
                let _ = is_ascii_fast(text);
            }
            Operation::ClassifyChars => {
                for ch in text.chars().take(10000) {
                    let _ = is_whitespace(ch);
                    let _ = is_punctuation(ch);
                    let _ = is_cjk_character(ch);
                    let _ = is_control(ch);
                }
            }
            Operation::GetFlags => {
                for ch in text.chars().take(10000) {
                    let flags = get_category_flags(ch);
                    // Exercise all flag accessors
                    let _ = flags.is_letter();
                    let _ = flags.is_number();
                    let _ = flags.is_punctuation();
                    let _ = flags.is_symbol();
                    let _ = flags.is_separator();
                    let _ = flags.is_other();
                }
            }
        }
    }
});
