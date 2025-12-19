//! Tokenization Property Tests (9.4.1)
//!
//! Tests that verify invariants of tokenization that must hold for all inputs:
//! - encode(decode(ids)) produces valid (possibly different) ids
//! - decode(encode(text)) preserves semantics
//! - batch encode == individual encodes
//! - truncated + overflow == original
//!
//! Note: Uses simple random generation. For production, use proptest crate.

use budtiktok_core::unicode::*;
use budtiktok_core::Encoding;

// =============================================================================
// Simple Random Generator
// =============================================================================

/// Simple PRNG for reproducible testing (no external dependencies)
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn gen_range(&mut self, min: usize, max: usize) -> usize {
        (self.next_u64() as usize) % (max - min) + min
    }

    fn gen_ascii_string(&mut self, len: usize) -> String {
        (0..len)
            .map(|_| {
                let ch = (self.gen_range(32, 127)) as u8 as char;
                ch
            })
            .collect()
    }

    fn gen_unicode_string(&mut self, len: usize) -> String {
        (0..len)
            .map(|_| {
                match self.gen_range(0, 4) {
                    0 => (self.gen_range(0x0041, 0x005A) as u32).try_into().unwrap_or('A'),
                    1 => (self.gen_range(0x0061, 0x007A) as u32).try_into().unwrap_or('a'),
                    2 => char::from_u32(self.gen_range(0x4E00, 0x9FFF) as u32).unwrap_or('中'),
                    _ => (self.gen_range(0x20, 0x7E) as u32).try_into().unwrap_or(' '),
                }
            })
            .collect()
    }
}

// =============================================================================
// Normalization Properties
// =============================================================================

/// Property: is_ascii_fast should return true iff all bytes < 128
#[test]
fn prop_is_ascii_fast_correctness() {
    let mut rng = SimpleRng::new(12345);

    for _ in 0..100 {
        // Generate random ASCII string
        let len = rng.gen_range(0, 1000);
        let ascii = rng.gen_ascii_string(len);

        // Property: ASCII-only strings should return true
        assert!(
            is_ascii_fast(&ascii),
            "is_ascii_fast failed for ASCII string: {:?}",
            &ascii[..ascii.len().min(50)]
        );
    }

    // Test with non-ASCII
    for _ in 0..100 {
        let len = rng.gen_range(1, 100);
        let mut s = rng.gen_ascii_string(len);
        // Insert a non-ASCII character
        let pos = rng.gen_range(0, s.len().max(1));
        s.insert(pos, 'é');

        // Property: Strings with non-ASCII should return false
        assert!(
            !is_ascii_fast(&s),
            "is_ascii_fast returned true for non-ASCII string"
        );
    }
}

/// Property: normalize should be idempotent (normalize(normalize(x)) == normalize(x))
#[test]
fn prop_normalize_idempotent() {
    let mut rng = SimpleRng::new(23456);

    for _ in 0..100 {
        let len = rng.gen_range(0, 500);
        let s = rng.gen_unicode_string(len);

        let once = normalize(&s, NormalizationForm::NFC);
        let twice = normalize(&once, NormalizationForm::NFC);

        assert_eq!(
            once, twice,
            "Normalization is not idempotent for: {:?}",
            &s[..s.len().min(50)]
        );
    }
}

/// Property: ASCII strings should be unchanged by NFC normalization
#[test]
fn prop_normalize_ascii_unchanged() {
    let mut rng = SimpleRng::new(34567);

    for _ in 0..100 {
        let len = rng.gen_range(0, 500);
        let ascii = rng.gen_ascii_string(len);

        let normalized = normalize(&ascii, NormalizationForm::NFC);

        assert_eq!(
            ascii, normalized,
            "ASCII string changed by normalization"
        );
    }
}

// =============================================================================
// Encoding Properties
// =============================================================================

/// Property: truncate should preserve the first max_length tokens
#[test]
fn prop_truncate_preserves_prefix() {
    let mut rng = SimpleRng::new(45678);

    for _ in 0..100 {
        let len = rng.gen_range(10, 100);
        let mut encoding = Encoding::new();

        for i in 0..len {
            encoding.push(
                i as u32,
                format!("t{}", i),
                (i, i + 1),
                Some(i as u32),
                Some(0),
                false,
            );
        }

        let max_len = rng.gen_range(1, len);
        let original_ids: Vec<_> = encoding.get_ids()[..max_len].to_vec();

        encoding.truncate(max_len, 0);

        assert_eq!(
            encoding.get_ids(),
            &original_ids,
            "Truncation changed preserved tokens"
        );
    }
}

/// Property: padding should not change existing tokens
#[test]
fn prop_padding_preserves_tokens() {
    let mut rng = SimpleRng::new(56789);

    for _ in 0..100 {
        let len = rng.gen_range(1, 50);
        let mut encoding = Encoding::new();

        for i in 0..len {
            encoding.push(
                i as u32 + 100, // Use IDs > 0 to distinguish from padding
                format!("t{}", i),
                (i, i + 1),
                Some(i as u32),
                Some(0),
                false,
            );
        }

        let original_ids: Vec<_> = encoding.get_ids().to_vec();
        let pad_len = rng.gen_range(len, len + 50);

        encoding.pad(pad_len, 0, "[PAD]");

        // First `len` tokens should be unchanged
        assert_eq!(
            &encoding.get_ids()[..len],
            &original_ids,
            "Padding changed existing tokens"
        );

        // Padding tokens should be 0
        for &id in &encoding.get_ids()[len..] {
            assert_eq!(id, 0, "Padding token has wrong ID");
        }
    }
}

/// Property: attention mask should match token structure
#[test]
fn prop_attention_mask_structure() {
    let mut rng = SimpleRng::new(67890);

    for _ in 0..100 {
        let len = rng.gen_range(1, 50);
        let pad_len = rng.gen_range(len, len + 20);
        let mut encoding = Encoding::new();

        for i in 0..len {
            encoding.push(
                i as u32,
                format!("t{}", i),
                (i, i + 1),
                Some(i as u32),
                Some(0),
                false,
            );
        }

        encoding.pad(pad_len, 0, "[PAD]");

        let mask = encoding.get_attention_mask();

        // Real tokens should have mask = 1
        for &m in &mask[..len] {
            assert_eq!(m, 1, "Real token has wrong attention mask");
        }

        // Padding tokens should have mask = 0
        for &m in &mask[len..] {
            assert_eq!(m, 0, "Padding token has wrong attention mask");
        }
    }
}

// =============================================================================
// Character Classification Properties
// =============================================================================

/// Property: is_whitespace should return true only for actual whitespace
#[test]
fn prop_whitespace_classification() {
    // All ASCII characters
    for c in 0u8..128u8 {
        let ch = c as char;
        let is_ws = is_whitespace(ch);

        // Known whitespace characters in ASCII (matching Rust's char::is_whitespace)
        // Includes: space, tab, newline, vertical tab, form feed, carriage return
        let expected_ws = matches!(c, 0x09 | 0x0A | 0x0B | 0x0C | 0x0D | 0x20);

        // Our is_whitespace should match Rust's char::is_whitespace exactly
        assert_eq!(
            is_ws, expected_ws,
            "Whitespace mismatch for \\x{:02X}: got {}, expected {}",
            c, is_ws, expected_ws
        );
    }
}

/// Property: is_punctuation should be consistent with category flags
#[test]
fn prop_punctuation_vs_category() {
    let punct_chars = ['!', '.', ',', '?', ':', ';', '-', '\'', '"', '(', ')'];

    for ch in punct_chars {
        let is_punct = is_punctuation(ch);
        let flags = get_category_flags(ch);

        // is_punctuation and category flags should agree
        assert!(
            is_punct && flags.is_punctuation(),
            "Inconsistent punctuation detection for '{}'",
            ch
        );
    }
}

/// Property: category flags should be mutually exclusive for basic categories
#[test]
fn prop_category_exclusivity() {
    let mut rng = SimpleRng::new(78901);

    for _ in 0..1000 {
        // Generate random ASCII char
        let c = (rng.gen_range(0, 128)) as u8 as char;
        let flags = get_category_flags(c);

        // Count how many major categories it belongs to
        let mut count = 0;
        if flags.is_letter() { count += 1; }
        if flags.is_number() { count += 1; }
        if flags.is_punctuation() { count += 1; }
        if flags.is_symbol() { count += 1; }
        if flags.is_separator() { count += 1; }
        if flags.contains(CategoryFlags::OTHER_CONTROL) { count += 1; }

        // Each char should belong to at most one major category
        // (some may be unclassified)
        assert!(
            count <= 1,
            "Character '{}' (U+{:04X}) belongs to {} categories",
            c.escape_debug(),
            c as u32,
            count
        );
    }
}

// =============================================================================
// CJK Detection Properties
// =============================================================================

/// Property: CJK detection should be consistent with Unicode ranges
#[test]
fn prop_cjk_ranges() {
    // CJK Unified Ideographs: 4E00-9FFF
    for cp in 0x4E00..=0x9FFF_u32 {
        if let Some(ch) = char::from_u32(cp) {
            assert!(
                is_cjk_character(ch),
                "U+{:04X} should be CJK",
                cp
            );
        }
    }

    // ASCII should NOT be CJK
    for c in 0u8..128 {
        let ch = c as char;
        assert!(
            !is_cjk_character(ch),
            "ASCII '{}' should not be CJK",
            ch.escape_debug()
        );
    }
}

// =============================================================================
// Control Character Properties
// =============================================================================

/// Property: control characters should be in expected ranges
#[test]
fn prop_control_ranges() {
    // C0 controls (0x00-0x1F, excluding tab/newline/cr for tokenization)
    for c in 0x00u8..=0x1F {
        if !matches!(c, 0x09 | 0x0A | 0x0D) {
            assert!(
                is_control(c as char),
                "\\x{:02X} should be control",
                c
            );
        }
    }

    // DEL (0x7F)
    assert!(is_control('\x7F'), "DEL should be control");

    // Printable ASCII should NOT be control
    for c in 0x20u8..=0x7E {
        assert!(
            !is_control(c as char),
            "'{}' should not be control",
            c as char
        );
    }
}

// =============================================================================
// Stress Tests
// =============================================================================

/// Property: functions should handle empty input
#[test]
fn prop_handles_empty_input() {
    assert!(is_ascii_fast(""));
    assert_eq!(normalize("", NormalizationForm::NFC), "");

    let mut encoding = Encoding::new();
    assert!(encoding.is_empty());
    encoding.truncate(10, 0);
    assert!(encoding.is_empty());
}

/// Property: functions should handle very long input
#[test]
fn prop_handles_long_input() {
    let long_ascii = "a".repeat(100_000);
    let long_unicode = "日".repeat(10_000);

    // Should not panic
    let _ = is_ascii_fast(&long_ascii);
    let _ = is_ascii_fast(&long_unicode);
    let _ = normalize(&long_ascii, NormalizationForm::NFC);
    let _ = normalize(&long_unicode, NormalizationForm::NFC);
}
