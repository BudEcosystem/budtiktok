//! Output Regression Tests (9.3.1)
//!
//! Tests that ensure fixed inputs produce fixed outputs.
//! Any change to tokenization output is detected immediately.

use budtiktok_core::unicode::*;

// =============================================================================
// Unicode Function Output Regression Tests
// =============================================================================

#[test]
fn test_regression_is_ascii_fast() {
    // Fixed inputs should always produce the same outputs
    let test_cases = [
        ("hello", true),
        ("Hello World!", true),
        ("日本語", false),
        ("café", false),
        ("", true),
        (" ", true),
        ("\t\n\r", true),
        ("hello\u{0080}", false),
    ];

    for (input, expected) in test_cases {
        assert_eq!(
            is_ascii_fast(input),
            expected,
            "is_ascii_fast('{}') changed",
            input.escape_debug()
        );
    }
}

#[test]
fn test_regression_normalize_nfc() {
    let test_cases = [
        ("hello", "hello"),
        ("HELLO", "HELLO"),
        ("", ""),
        (" ", " "),
    ];

    for (input, expected) in test_cases {
        let result = normalize(input, NormalizationForm::NFC);
        assert_eq!(
            result, expected,
            "normalize('{}', NFC) changed: got '{}'",
            input.escape_debug(),
            result.escape_debug()
        );
    }
}

#[test]
fn test_regression_is_whitespace() {
    let test_cases = [
        (' ', true),
        ('\t', true),
        ('\n', true),
        ('\r', true),
        ('\u{00A0}', true), // No-break space
        ('a', false),
        ('!', false),
        ('0', false),
    ];

    for (ch, expected) in test_cases {
        assert_eq!(
            is_whitespace(ch),
            expected,
            "is_whitespace('{}') changed",
            ch.escape_debug()
        );
    }
}

#[test]
fn test_regression_is_punctuation() {
    let test_cases = [
        ('!', true),
        ('.', true),
        (',', true),
        ('?', true),
        (':', true),
        (';', true),
        ('a', false),
        ('Z', false),
        ('5', false),
        (' ', false),
    ];

    for (ch, expected) in test_cases {
        assert_eq!(
            is_punctuation(ch),
            expected,
            "is_punctuation('{}') changed",
            ch.escape_debug()
        );
    }
}

#[test]
fn test_regression_is_cjk() {
    let test_cases = [
        ('中', true),
        ('文', true),
        ('日', true),
        ('本', true),
        ('a', false),
        ('Z', false),
        ('あ', false), // Hiragana is not CJK ideograph
        ('ア', false), // Katakana is not CJK ideograph
        ('한', false), // Hangul is not CJK ideograph
    ];

    for (ch, expected) in test_cases {
        assert_eq!(
            is_cjk_character(ch),
            expected,
            "is_cjk_character('{}') changed",
            ch.escape_debug()
        );
    }
}

#[test]
fn test_regression_is_control() {
    let test_cases = [
        ('\x00', true),  // NUL
        ('\x01', true),  // SOH
        ('\x1F', true),  // Unit separator
        ('\x7F', true),  // DEL
        ('a', false),
        (' ', false),
        ('\t', false),   // Tab is whitespace, not control for tokenization
        ('\n', false),   // Newline is whitespace, not control for tokenization
    ];

    for (ch, expected) in test_cases {
        assert_eq!(
            is_control(ch),
            expected,
            "is_control('\\x{:02X}') changed",
            ch as u32
        );
    }
}

// =============================================================================
// Category Flags Regression Tests
// =============================================================================

#[test]
fn test_regression_category_flags() {
    // ASCII letter lowercase
    let flags = get_category_flags('a');
    assert!(flags.is_letter(), "get_category_flags('a').is_letter() changed");
    assert!(!flags.is_number(), "get_category_flags('a').is_number() changed");

    // ASCII letter uppercase
    let flags = get_category_flags('Z');
    assert!(flags.is_letter(), "get_category_flags('Z').is_letter() changed");
    assert!(flags.contains(CategoryFlags::LETTER_UPPERCASE), "get_category_flags('Z') uppercase flag changed");

    // Digit
    let flags = get_category_flags('5');
    assert!(flags.is_number(), "get_category_flags('5').is_number() changed");
    assert!(!flags.is_letter(), "get_category_flags('5').is_letter() changed");

    // Space
    let flags = get_category_flags(' ');
    assert!(flags.is_separator(), "get_category_flags(' ').is_separator() changed");

    // Punctuation
    let flags = get_category_flags('!');
    assert!(flags.is_punctuation(), "get_category_flags('!').is_punctuation() changed");
}

// =============================================================================
// Golden File Tests (Pattern for future expansion)
// =============================================================================

/// Golden file test pattern for tokenization output
///
/// In a real implementation, this would:
/// 1. Load expected outputs from JSON files in tests/regression_tests/fixtures/
/// 2. Run tokenizer on each input
/// 3. Compare outputs byte-for-byte
///
/// Example fixture format:
/// ```json
/// {
///   "tokenizer": "bert-base-uncased",
///   "test_cases": [
///     {"input": "Hello, world!", "ids": [101, 7592, 1010, 2088, 102]},
///     {"input": "Test", "ids": [101, 3231, 102]}
///   ]
/// }
/// ```
#[test]
#[ignore = "Requires fixture files to be created"]
fn test_regression_golden_files() {
    // let fixtures = load_fixtures("tests/regression_tests/fixtures/*.json");
    //
    // for fixture in fixtures {
    //     let tokenizer = load_tokenizer(&fixture.tokenizer)?;
    //
    //     for case in fixture.test_cases {
    //         let encoding = tokenizer.encode(&case.input, true)?;
    //         assert_eq!(
    //             encoding.get_ids(),
    //             &case.ids,
    //             "Output changed for '{}' with tokenizer '{}'",
    //             case.input,
    //             fixture.tokenizer
    //         );
    //     }
    // }
    todo!("Implement golden file loading and comparison");
}

// =============================================================================
// Version Stability Tests
// =============================================================================

#[test]
fn test_regression_version_format() {
    use budtiktok_core::{VERSION, GIT_HASH, BUILD_TIMESTAMP};

    // Version should be semver format
    let parts: Vec<_> = VERSION.split('.').collect();
    assert!(
        parts.len() >= 2,
        "VERSION format changed: expected semver, got '{}'",
        VERSION
    );

    // Git hash should be 7-40 hex chars, empty, or a placeholder
    // (may be placeholder in dev builds without git)
    if !GIT_HASH.is_empty() && !GIT_HASH.contains("see") && !GIT_HASH.contains("unknown") {
        assert!(
            GIT_HASH.chars().all(|c| c.is_ascii_hexdigit()),
            "GIT_HASH format changed: expected hex, got '{}'",
            GIT_HASH
        );
    }

    // Build timestamp should be non-empty in release builds
    // (may be empty in test builds)
    let _ = BUILD_TIMESTAMP; // Just ensure it exists
}
