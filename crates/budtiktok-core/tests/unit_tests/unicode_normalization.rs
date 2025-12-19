//! Unicode Normalization Unit Tests (9.1.1)
//!
//! Tests for Unicode normalization including:
//! - ASCII fast path detection
//! - NFC quick check
//! - Full NFC/NFD normalization
//! - Hangul decomposition
//! - Streaming normalization

use budtiktok_core::unicode::*;

// =============================================================================
// ASCII Fast Path Tests
// =============================================================================

#[test]
fn test_is_ascii_fast_pure_ascii() {
    assert!(is_ascii_fast("hello"));
    assert!(is_ascii_fast("Hello World!"));
    assert!(is_ascii_fast("0123456789"));
    assert!(is_ascii_fast("!@#$%^&*()"));
    assert!(is_ascii_fast(""));
    assert!(is_ascii_fast(" "));
    assert!(is_ascii_fast("\t\n\r"));
}

#[test]
fn test_is_ascii_fast_with_high_bytes() {
    assert!(!is_ascii_fast("héllo"));
    assert!(!is_ascii_fast("café"));
    assert!(!is_ascii_fast("日本語"));
    assert!(!is_ascii_fast("hello\u{0080}")); // First non-ASCII
    assert!(!is_ascii_fast("\u{00FF}"));
}

#[test]
fn test_is_ascii_fast_boundary() {
    // Test at u64 chunk boundaries
    assert!(is_ascii_fast("12345678")); // Exactly 8 bytes
    assert!(is_ascii_fast("1234567890123456")); // Exactly 16 bytes
    assert!(is_ascii_fast("1234567")); // 7 bytes (remainder)
    assert!(is_ascii_fast("123456789")); // 9 bytes (8 + 1)
}

#[test]
fn test_is_ascii_fast_long_string() {
    let long_ascii = "a".repeat(10000);
    assert!(is_ascii_fast(&long_ascii));

    let mut with_unicode = long_ascii.clone();
    with_unicode.push('é');
    assert!(!is_ascii_fast(&with_unicode));
}

// =============================================================================
// Normalization Tests
// =============================================================================

#[test]
fn test_normalize_nfc() {
    // ASCII unchanged
    let result = normalize("hello", NormalizationForm::NFC);
    assert_eq!(result, "hello");
}

#[test]
fn test_normalize_nfd() {
    let result = normalize("hello", NormalizationForm::NFD);
    assert_eq!(result, "hello");
}

#[test]
fn test_normalize_none() {
    let result = normalize("hello", NormalizationForm::None);
    assert_eq!(result, "hello");
}

#[test]
fn test_normalize_with_fast_path_ascii() {
    use std::borrow::Cow;

    let result = normalize_with_fast_path("hello", NormalizationForm::NFC);
    // ASCII should be borrowed (no allocation)
    assert!(matches!(result, Cow::Borrowed(_)));
    assert_eq!(result, "hello");
}

#[test]
fn test_normalize_with_fast_path_unicode() {
    use std::borrow::Cow;

    let result = normalize_with_fast_path("héllo", NormalizationForm::NFC);
    // Unicode may require normalization (owned)
    assert!(!result.is_empty());
}

// =============================================================================
// Whitespace Tests
// =============================================================================

#[test]
fn test_is_whitespace_ascii() {
    assert!(is_whitespace(' '));
    assert!(is_whitespace('\t'));
    assert!(is_whitespace('\n'));
    assert!(is_whitespace('\r'));
}

#[test]
fn test_is_whitespace_not() {
    assert!(!is_whitespace('a'));
    assert!(!is_whitespace('Z'));
    assert!(!is_whitespace('0'));
    assert!(!is_whitespace('!'));
}

#[test]
fn test_is_whitespace_unicode() {
    assert!(is_whitespace('\u{00A0}')); // No-break space
}

// =============================================================================
// Punctuation Tests
// =============================================================================

#[test]
fn test_is_punctuation_basic() {
    assert!(is_punctuation('!'));
    assert!(is_punctuation('.'));
    assert!(is_punctuation(','));
    assert!(is_punctuation('?'));
    assert!(is_punctuation(':'));
    assert!(is_punctuation(';'));
}

#[test]
fn test_is_punctuation_not() {
    assert!(!is_punctuation('a'));
    assert!(!is_punctuation('Z'));
    assert!(!is_punctuation('5'));
    assert!(!is_punctuation(' '));
}

// =============================================================================
// CJK Character Tests
// =============================================================================

#[test]
fn test_is_cjk_chinese() {
    assert!(is_cjk_character('中'));
    assert!(is_cjk_character('文'));
    assert!(is_cjk_character('日'));
    assert!(is_cjk_character('本'));
}

#[test]
fn test_is_cjk_not() {
    assert!(!is_cjk_character('a'));
    assert!(!is_cjk_character('Z'));
    assert!(!is_cjk_character('5'));
    assert!(!is_cjk_character(' '));
    // Hiragana/Katakana are not CJK ideographs
    assert!(!is_cjk_character('あ'));
    assert!(!is_cjk_character('ア'));
}

// =============================================================================
// Control Character Tests
// =============================================================================

#[test]
fn test_is_control_basic() {
    assert!(is_control('\x00')); // NUL
    assert!(is_control('\x01')); // SOH
    assert!(is_control('\x1F')); // Unit separator
    assert!(is_control('\x7F')); // DEL
}

#[test]
fn test_is_control_not() {
    assert!(!is_control('a'));
    assert!(!is_control(' '));
    // Tab, LF, CR are typically excluded from "control" in tokenization context
    assert!(!is_control('\t'));
    assert!(!is_control('\n'));
    assert!(!is_control('\r'));
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_empty_string() {
    assert!(is_ascii_fast(""));
    assert_eq!(normalize("", NormalizationForm::NFC), "");
}

#[test]
fn test_single_character() {
    assert!(is_ascii_fast("a"));
    assert!(!is_ascii_fast("é"));
}

#[test]
fn test_null_character() {
    assert!(is_ascii_fast("\0"));
    assert!(is_ascii_fast("a\0b"));
}
