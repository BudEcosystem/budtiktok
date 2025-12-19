//! Unicode Categories Unit Tests (9.1.2)
//!
//! Tests for Unicode category detection including:
//! - ASCII category lookup
//! - Punctuation detection
//! - CJK detection
//! - Emoji detection
//! - Category flags operations

use budtiktok_core::unicode::*;

// =============================================================================
// Category Flags Tests
// =============================================================================

#[test]
fn test_category_flags_letter() {
    let flags = CategoryFlags::new(CategoryFlags::LETTER_UPPERCASE);
    assert!(flags.is_letter());

    let flags = CategoryFlags::new(CategoryFlags::LETTER_LOWERCASE);
    assert!(flags.is_letter());

    let flags = CategoryFlags::new(CategoryFlags::LETTER);
    assert!(flags.is_letter());
}

#[test]
fn test_category_flags_bitwise() {
    // Combining flags should work
    let combined = CategoryFlags::LETTER_UPPERCASE | CategoryFlags::LETTER_LOWERCASE;
    let flags = CategoryFlags::new(combined);
    assert!(flags.is_letter());
}

#[test]
fn test_category_flags_number() {
    let flags = CategoryFlags::new(CategoryFlags::NUMBER_DECIMAL);
    assert!(flags.is_number());
    assert!(!flags.is_letter());
}

#[test]
fn test_category_flags_punctuation() {
    let flags = CategoryFlags::new(CategoryFlags::PUNCTUATION_OTHER);
    assert!(flags.is_punctuation());
    assert!(!flags.is_letter());
}

// =============================================================================
// Get Category Flags Tests
// =============================================================================

#[test]
fn test_get_category_flags_ascii_letter() {
    let flags = get_category_flags('a');
    assert!(flags.is_letter());
    assert!(!flags.contains(CategoryFlags::LETTER_UPPERCASE));

    let flags = get_category_flags('Z');
    assert!(flags.is_letter());
    assert!(flags.contains(CategoryFlags::LETTER_UPPERCASE));
}

#[test]
fn test_get_category_flags_digit() {
    let flags = get_category_flags('5');
    assert!(flags.is_number());
    assert!(!flags.is_letter());
}

#[test]
fn test_get_category_flags_space() {
    let flags = get_category_flags(' ');
    assert!(flags.is_separator());
}

#[test]
fn test_get_category_flags_punctuation() {
    let flags = get_category_flags('!');
    assert!(flags.is_punctuation());

    let flags = get_category_flags('.');
    assert!(flags.is_punctuation());
}

// =============================================================================
// Punctuation Tests
// =============================================================================

#[test]
fn test_punctuation_basic() {
    // Note: $, %, +, <, =, >, ^, |, ~, `, # are symbols, not punctuation in Unicode
    // Standard ASCII punctuation (Connector, Dash, Open, Close, Other)
    let punct_chars = [
        '!', '"', '&', '\'', '(', ')', '*', ',', '-', '.', '/', ':', ';',
        '?', '@', '[', '\\', ']', '_', '{', '}',
    ];

    for ch in punct_chars {
        assert!(is_punctuation(ch), "'{}' should be punctuation", ch);
    }
}

#[test]
fn test_punctuation_unicode() {
    // Common Unicode punctuation
    let unicode_punct = [
        '。', '、', '「', '」', '『', '』', '【', '】', '〈', '〉', '《', '》', '〔', '〕', '〖',
        '〗', '〘', '〙', '〚', '〛',
    ];

    for ch in unicode_punct {
        assert!(is_punctuation(ch), "'{}' should be punctuation", ch);
    }
}

#[test]
fn test_not_punctuation() {
    assert!(!is_punctuation('a'));
    assert!(!is_punctuation('Z'));
    assert!(!is_punctuation('5'));
    assert!(!is_punctuation(' '));
}

// =============================================================================
// CJK Detection Tests
// =============================================================================

#[test]
fn test_cjk_chinese() {
    let chinese_chars = ['中', '文', '字', '国', '人', '大', '小', '上', '下'];
    for ch in chinese_chars {
        assert!(is_cjk_character(ch), "'{}' should be CJK", ch);
    }
}

#[test]
fn test_cjk_japanese_kanji() {
    // Japanese kanji (same as Chinese)
    let kanji = ['日', '本', '語', '漢', '字'];
    for ch in kanji {
        assert!(is_cjk_character(ch), "'{}' should be CJK", ch);
    }
}

#[test]
fn test_not_cjk() {
    // Japanese hiragana/katakana are NOT CJK ideographs
    assert!(!is_cjk_character('あ'), "Hiragana should not be CJK");
    assert!(!is_cjk_character('ア'), "Katakana should not be CJK");

    // Korean hangul is NOT CJK
    assert!(!is_cjk_character('한'), "Hangul should not be CJK");

    // ASCII is not CJK
    assert!(!is_cjk_character('A'));
    assert!(!is_cjk_character('5'));
}

// =============================================================================
// Whitespace Tests
// =============================================================================

#[test]
fn test_is_whitespace_comprehensive() {
    // Standard whitespace
    assert!(is_whitespace(' '));
    assert!(is_whitespace('\t'));
    assert!(is_whitespace('\n'));
    assert!(is_whitespace('\r'));

    // Unicode whitespace
    assert!(is_whitespace('\u{00A0}')); // No-break space
}

// =============================================================================
// Control Character Tests
// =============================================================================

#[test]
fn test_control_chars() {
    assert!(is_control('\x00')); // NUL
    assert!(is_control('\x1F')); // Unit separator
    assert!(is_control('\x7F')); // DEL
}

#[test]
fn test_control_flags() {
    let flags = get_category_flags('\x00');
    assert!(flags.contains(CategoryFlags::OTHER_CONTROL));
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_category_null_char() {
    let flags = get_category_flags('\0');
    assert!(flags.contains(CategoryFlags::OTHER_CONTROL));
}

#[test]
fn test_category_lookup_consistent() {
    // Multiple lookups should return same result
    let ch = '中';
    let cat1 = get_category_flags(ch);
    let cat2 = get_category_flags(ch);
    assert_eq!(cat1.bits(), cat2.bits());
}
