//! Unicode processing utilities
//!
//! This module provides high-performance Unicode normalization and character
//! classification utilities used during tokenization.
//!
//! Key optimizations from blazetext:
//! - ASCII fast path with O(1) lookup table (128 entries)
//! - Thread-local cache for non-ASCII category lookups (128 entries)
//! - Binary search through pre-computed Unicode tables
//! - Range checks for CJK before binary search
//! - Bitfield CategoryFlags for O(1) multi-category checks

use std::borrow::Cow;
use std::cell::RefCell;
use unicode_normalization::UnicodeNormalization;

use crate::tables;

/// Check if a string is pure ASCII using SIMD-style u64 operations
///
/// This is a fast path optimization - if all bytes < 128, we can skip
/// normalization entirely and return a borrowed reference (zero-copy).
#[inline]
pub fn is_ascii_fast(s: &str) -> bool {
    let bytes = s.as_bytes();

    // Process 8 bytes at a time using u64
    let chunks = bytes.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        // Load 8 bytes into a u64 (safe: chunks_exact guarantees 8 bytes)
        let word = u64::from_ne_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3],
            chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        // If any byte has the high bit set (>= 128), it's not ASCII
        if word & 0x8080_8080_8080_8080 != 0 {
            return false;
        }
    }

    // Check remaining bytes
    for &byte in remainder {
        if byte >= 128 {
            return false;
        }
    }

    true
}

/// Normalize text with ASCII fast path optimization
///
/// Returns Cow::Borrowed if the string is ASCII (no normalization needed),
/// otherwise returns Cow::Owned with normalized text.
#[inline]
pub fn normalize_with_fast_path<'a>(text: &'a str, form: NormalizationForm) -> Cow<'a, str> {
    // ASCII strings don't need normalization
    if form == NormalizationForm::None || is_ascii_fast(text) {
        return Cow::Borrowed(text);
    }
    Cow::Owned(normalize(text, form))
}

/// Unicode normalization forms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationForm {
    /// No normalization
    None,
    /// Canonical Decomposition (NFD)
    NFD,
    /// Canonical Composition (NFC)
    NFC,
    /// Compatibility Decomposition (NFKD)
    NFKD,
    /// Compatibility Composition (NFKC)
    NFKC,
}

/// Normalize text according to the specified form
pub fn normalize(text: &str, form: NormalizationForm) -> String {
    match form {
        NormalizationForm::None => text.to_string(),
        NormalizationForm::NFD => text.nfd().collect(),
        NormalizationForm::NFC => text.nfc().collect(),
        NormalizationForm::NFKD => text.nfkd().collect(),
        NormalizationForm::NFKC => text.nfkc().collect(),
    }
}

/// Check if a character is a whitespace character (Unicode-aware)
///
/// Uses Rust's standard `char::is_whitespace()` which includes:
/// - ASCII: space (0x20), tab (0x09), newline (0x0A), vertical tab (0x0B),
///          form feed (0x0C), carriage return (0x0D)
/// - Unicode: No-break space, en/em space, line/paragraph separators, etc.
#[inline]
pub fn is_whitespace(c: char) -> bool {
    c.is_whitespace()
}

/// Check if a character is a punctuation character
#[inline]
pub fn is_punctuation(c: char) -> bool {
    get_category_flags(c).is_punctuation()
}

/// Check if a character is a Chinese/Japanese/Korean character
#[inline]
pub fn is_cjk_character(c: char) -> bool {
    let cp = c as u32;
    // CJK Unified Ideographs and related ranges
    matches!(cp,
        0x4E00..=0x9FFF |           // CJK Unified Ideographs
        0x3400..=0x4DBF |           // CJK Unified Ideographs Extension A
        0x20000..=0x2A6DF |         // CJK Unified Ideographs Extension B
        0x2A700..=0x2B73F |         // CJK Unified Ideographs Extension C
        0x2B740..=0x2B81F |         // CJK Unified Ideographs Extension D
        0x2B820..=0x2CEAF |         // CJK Unified Ideographs Extension E
        0xF900..=0xFAFF |           // CJK Compatibility Ideographs
        0x2F800..=0x2FA1F           // CJK Compatibility Ideographs Supplement
    )
}

/// Check if a character is a control character
#[inline]
pub fn is_control(c: char) -> bool {
    let cp = c as u32;
    // Control characters, excluding common whitespace
    (cp <= 0x1F && cp != 0x09 && cp != 0x0A && cp != 0x0D) // C0 controls minus tab, LF, CR
        || (0x7F..=0x9F).contains(&cp) // Delete and C1 controls
}

/// Unicode general category (simplified)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneralCategory {
    Lu, // Letter, Uppercase
    Ll, // Letter, Lowercase
    Lt, // Letter, Titlecase
    Lm, // Letter, Modifier
    Lo, // Letter, Other
    Mn, // Mark, Nonspacing
    Mc, // Mark, Spacing Combining
    Me, // Mark, Enclosing
    Nd, // Number, Decimal Digit
    Nl, // Number, Letter
    No, // Number, Other
    Pc, // Punctuation, Connector
    Pd, // Punctuation, Dash
    Ps, // Punctuation, Open
    Pe, // Punctuation, Close
    Pi, // Punctuation, Initial quote
    Pf, // Punctuation, Final quote
    Po, // Punctuation, Other
    Sm, // Symbol, Math
    Sc, // Symbol, Currency
    Sk, // Symbol, Modifier
    So, // Symbol, Other
    Zs, // Separator, Space
    Zl, // Separator, Line
    Zp, // Separator, Paragraph
    Cc, // Other, Control
    Cf, // Other, Format
    Cs, // Other, Surrogate
    Co, // Other, Private Use
    Cn, // Other, Not Assigned
}

/// Category flags bitfield for efficient multi-category checks (32x speedup)
///
/// Allows checking multiple Unicode categories at once using bitwise operations.
/// Much faster than checking individual GeneralCategory values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CategoryFlags(u32);

impl CategoryFlags {
    // Individual category bits (aligned with blazetext)
    pub const OTHER_CONTROL: u32 = 1 << 0;
    pub const OTHER_FORMAT: u32 = 1 << 1;
    pub const OTHER_PRIVATE_USE: u32 = 1 << 2;
    pub const LETTER_LOWERCASE: u32 = 1 << 3;
    pub const LETTER_MODIFIER: u32 = 1 << 4;
    pub const LETTER_OTHER: u32 = 1 << 5;
    pub const LETTER_TITLECASE: u32 = 1 << 6;
    pub const LETTER_UPPERCASE: u32 = 1 << 7;
    pub const MARK_SPACING_COMBINING: u32 = 1 << 8;
    pub const MARK_ENCLOSING: u32 = 1 << 9;
    pub const MARK_NONSPACING: u32 = 1 << 10;
    pub const NUMBER_DECIMAL: u32 = 1 << 11;
    pub const NUMBER_LETTER: u32 = 1 << 12;
    pub const NUMBER_OTHER: u32 = 1 << 13;
    pub const PUNCTUATION_CONNECTOR: u32 = 1 << 14;
    pub const PUNCTUATION_DASH: u32 = 1 << 15;
    pub const PUNCTUATION_CLOSE: u32 = 1 << 16;
    pub const PUNCTUATION_FINAL: u32 = 1 << 17;
    pub const PUNCTUATION_INITIAL: u32 = 1 << 18;
    pub const PUNCTUATION_OTHER: u32 = 1 << 19;
    pub const PUNCTUATION_OPEN: u32 = 1 << 20;
    pub const SYMBOL_CURRENCY: u32 = 1 << 21;
    pub const SYMBOL_MODIFIER: u32 = 1 << 22;
    pub const SYMBOL_MATH: u32 = 1 << 23;
    pub const SYMBOL_OTHER: u32 = 1 << 24;
    pub const SEPARATOR_LINE: u32 = 1 << 25;
    pub const SEPARATOR_PARAGRAPH: u32 = 1 << 26;
    pub const SEPARATOR_SPACE: u32 = 1 << 27;
    pub const OTHER_SURROGATE: u32 = 1 << 28;
    pub const OTHER_NOT_ASSIGNED: u32 = 1 << 29;

    // Combined category masks for fast multi-category checks
    pub const LETTER: u32 = Self::LETTER_UPPERCASE
        | Self::LETTER_LOWERCASE
        | Self::LETTER_TITLECASE
        | Self::LETTER_MODIFIER
        | Self::LETTER_OTHER;

    pub const MARK: u32 =
        Self::MARK_NONSPACING | Self::MARK_SPACING_COMBINING | Self::MARK_ENCLOSING;

    pub const NUMBER: u32 = Self::NUMBER_DECIMAL | Self::NUMBER_LETTER | Self::NUMBER_OTHER;

    pub const PUNCTUATION: u32 = Self::PUNCTUATION_CONNECTOR
        | Self::PUNCTUATION_DASH
        | Self::PUNCTUATION_OPEN
        | Self::PUNCTUATION_CLOSE
        | Self::PUNCTUATION_INITIAL
        | Self::PUNCTUATION_FINAL
        | Self::PUNCTUATION_OTHER;

    pub const SYMBOL: u32 =
        Self::SYMBOL_MATH | Self::SYMBOL_CURRENCY | Self::SYMBOL_MODIFIER | Self::SYMBOL_OTHER;

    pub const SEPARATOR: u32 =
        Self::SEPARATOR_SPACE | Self::SEPARATOR_LINE | Self::SEPARATOR_PARAGRAPH;

    pub const OTHER: u32 = Self::OTHER_CONTROL
        | Self::OTHER_FORMAT
        | Self::OTHER_SURROGATE
        | Self::OTHER_PRIVATE_USE
        | Self::OTHER_NOT_ASSIGNED;

    /// Create new CategoryFlags
    #[inline]
    pub const fn new(flags: u32) -> Self {
        Self(flags)
    }

    /// Get the raw flags value
    #[inline]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Check if any of the specified category bits are set
    #[inline]
    pub const fn contains(self, flags: u32) -> bool {
        self.0 & flags != 0
    }

    /// Check if this is a letter (Lu, Ll, Lt, Lm, Lo)
    #[inline]
    pub const fn is_letter(self) -> bool {
        self.contains(Self::LETTER)
    }

    /// Check if this is a mark (Mn, Mc, Me)
    #[inline]
    pub const fn is_mark(self) -> bool {
        self.contains(Self::MARK)
    }

    /// Check if this is a number (Nd, Nl, No)
    #[inline]
    pub const fn is_number(self) -> bool {
        self.contains(Self::NUMBER)
    }

    /// Check if this is punctuation (Pc, Pd, Ps, Pe, Pi, Pf, Po)
    #[inline]
    pub const fn is_punctuation(self) -> bool {
        self.contains(Self::PUNCTUATION)
    }

    /// Check if this is a symbol (Sm, Sc, Sk, So)
    #[inline]
    pub const fn is_symbol(self) -> bool {
        self.contains(Self::SYMBOL)
    }

    /// Check if this is a separator (Zs, Zl, Zp)
    #[inline]
    pub const fn is_separator(self) -> bool {
        self.contains(Self::SEPARATOR)
    }

    /// Check if this is in the "other" category (Cc, Cf, Cs, Co, Cn)
    #[inline]
    pub const fn is_other(self) -> bool {
        self.contains(Self::OTHER)
    }

    /// Convert from GeneralCategory to CategoryFlags
    #[inline]
    pub fn from_general_category(cat: GeneralCategory) -> Self {
        let bits = match cat {
            GeneralCategory::Lu => Self::LETTER_UPPERCASE,
            GeneralCategory::Ll => Self::LETTER_LOWERCASE,
            GeneralCategory::Lt => Self::LETTER_TITLECASE,
            GeneralCategory::Lm => Self::LETTER_MODIFIER,
            GeneralCategory::Lo => Self::LETTER_OTHER,
            GeneralCategory::Mn => Self::MARK_NONSPACING,
            GeneralCategory::Mc => Self::MARK_SPACING_COMBINING,
            GeneralCategory::Me => Self::MARK_ENCLOSING,
            GeneralCategory::Nd => Self::NUMBER_DECIMAL,
            GeneralCategory::Nl => Self::NUMBER_LETTER,
            GeneralCategory::No => Self::NUMBER_OTHER,
            GeneralCategory::Pc => Self::PUNCTUATION_CONNECTOR,
            GeneralCategory::Pd => Self::PUNCTUATION_DASH,
            GeneralCategory::Ps => Self::PUNCTUATION_OPEN,
            GeneralCategory::Pe => Self::PUNCTUATION_CLOSE,
            GeneralCategory::Pi => Self::PUNCTUATION_INITIAL,
            GeneralCategory::Pf => Self::PUNCTUATION_FINAL,
            GeneralCategory::Po => Self::PUNCTUATION_OTHER,
            GeneralCategory::Sm => Self::SYMBOL_MATH,
            GeneralCategory::Sc => Self::SYMBOL_CURRENCY,
            GeneralCategory::Sk => Self::SYMBOL_MODIFIER,
            GeneralCategory::So => Self::SYMBOL_OTHER,
            GeneralCategory::Zs => Self::SEPARATOR_SPACE,
            GeneralCategory::Zl => Self::SEPARATOR_LINE,
            GeneralCategory::Zp => Self::SEPARATOR_PARAGRAPH,
            GeneralCategory::Cc => Self::OTHER_CONTROL,
            GeneralCategory::Cf => Self::OTHER_FORMAT,
            GeneralCategory::Cs => Self::OTHER_SURROGATE,
            GeneralCategory::Co => Self::OTHER_PRIVATE_USE,
            GeneralCategory::Cn => Self::OTHER_NOT_ASSIGNED,
        };
        Self(bits)
    }
}

/// ASCII category lookup table (128 entries) - O(1) lookup
/// Pre-computed at compile time for ASCII characters.
const ASCII_CATEGORIES: [CategoryFlags; 128] = {
    let mut table = [CategoryFlags(0); 128];
    let mut i = 0;

    while i < 128 {
        let mut flags = 0u32;

        // Control characters
        if i < 32 || i == 127 {
            flags |= CategoryFlags::OTHER_CONTROL;
        }

        // Letters
        if i >= b'a' as usize && i <= b'z' as usize {
            flags |= CategoryFlags::LETTER_LOWERCASE;
        } else if i >= b'A' as usize && i <= b'Z' as usize {
            flags |= CategoryFlags::LETTER_UPPERCASE;
        }

        // Numbers
        if i >= b'0' as usize && i <= b'9' as usize {
            flags |= CategoryFlags::NUMBER_DECIMAL;
        }

        // Punctuation
        match i as u8 {
            b'_' => flags |= CategoryFlags::PUNCTUATION_CONNECTOR,
            b'-' => flags |= CategoryFlags::PUNCTUATION_DASH,
            b'(' | b'[' | b'{' => flags |= CategoryFlags::PUNCTUATION_OPEN,
            b')' | b']' | b'}' => flags |= CategoryFlags::PUNCTUATION_CLOSE,
            b'!' | b'"' | b'#' | b'%' | b'&' | b'\'' | b'*' | b',' | b'.' | b'/' | b':' | b';'
            | b'?' | b'@' | b'\\' => flags |= CategoryFlags::PUNCTUATION_OTHER,
            _ => {}
        }

        // Symbols
        match i as u8 {
            b'$' => flags |= CategoryFlags::SYMBOL_CURRENCY,
            b'+' | b'<' | b'=' | b'>' | b'|' | b'~' => flags |= CategoryFlags::SYMBOL_MATH,
            b'^' | b'`' => flags |= CategoryFlags::SYMBOL_MODIFIER,
            _ => {}
        }

        // Separators
        if i == b' ' as usize {
            flags |= CategoryFlags::SEPARATOR_SPACE;
        }

        table[i] = CategoryFlags(flags);
        i += 1;
    }

    table
};

// Thread-local cache for Unicode category lookups (128 entries)
const CACHE_SIZE: usize = 128;

#[derive(Clone, Copy)]
struct CacheEntry {
    char_code: u32,
    flags: CategoryFlags,
}

thread_local! {
    static CATEGORY_CACHE: RefCell<CategoryCache> = RefCell::new(CategoryCache::new());
}

struct CategoryCache {
    entries: [Option<CacheEntry>; CACHE_SIZE],
    next_slot: usize,
}

impl CategoryCache {
    fn new() -> Self {
        CategoryCache {
            entries: [None; CACHE_SIZE],
            next_slot: 0,
        }
    }

    fn lookup(&self, c: char) -> Option<CategoryFlags> {
        let target = c as u32;
        for entry in &self.entries {
            if let Some(e) = entry {
                if e.char_code == target {
                    return Some(e.flags);
                }
            }
        }
        None
    }

    fn insert(&mut self, c: char, flags: CategoryFlags) {
        self.entries[self.next_slot] = Some(CacheEntry {
            char_code: c as u32,
            flags,
        });
        self.next_slot = (self.next_slot + 1) % CACHE_SIZE;
    }
}

/// Binary search helper for Unicode tables
#[inline]
fn table_binary_search(target: char, table: &'static [char]) -> bool {
    table.binary_search(&target).is_ok()
}

/// Get all category flags for a character in a single lookup (32x speedup)
///
/// Uses:
/// 1. ASCII lookup table for chars 0-127
/// 2. Thread-local cache for recently used non-ASCII chars
/// 3. Range checks for CJK before binary search
/// 4. Binary search through pre-computed Unicode tables
#[inline]
pub fn get_category_flags(c: char) -> CategoryFlags {
    let cp = c as u32;

    // Fast path: ASCII lookup table
    if cp < 128 {
        return ASCII_CATEGORIES[cp as usize];
    }

    // Check thread-local cache
    if let Some(flags) = CATEGORY_CACHE.with(|cache| cache.borrow().lookup(c)) {
        return flags;
    }

    // Compute flags from Unicode tables
    let flags = compute_category_flags(c);

    // Cache the result
    CATEGORY_CACHE.with(|cache| cache.borrow_mut().insert(c, flags));

    flags
}

/// Compute category flags from Unicode tables
fn compute_category_flags(c: char) -> CategoryFlags {
    let mut flags = 0u32;

    // Other categories
    if table_binary_search(c, tables::OTHER_CONTROL) {
        flags |= CategoryFlags::OTHER_CONTROL;
    }
    if table_binary_search(c, tables::OTHER_FORMAT) {
        flags |= CategoryFlags::OTHER_FORMAT;
    }

    // Private use - use range checks first (optimization)
    if matches!(c, '\u{E000}'..='\u{F8FF}' | '\u{F0000}'..='\u{FFFFD}' | '\u{100000}'..='\u{10FFFD}')
        || table_binary_search(c, tables::OTHER_PRIVATE_USE)
    {
        flags |= CategoryFlags::OTHER_PRIVATE_USE;
    }

    // Letter categories - optimize common ranges
    // Cyrillic lowercase: 0x0430-0x045F
    if matches!(c, '\u{0430}'..='\u{045F}') || table_binary_search(c, tables::LETTER_LOWERCASED) {
        flags |= CategoryFlags::LETTER_LOWERCASE;
    }
    if table_binary_search(c, tables::LETTER_MODIFIER) {
        flags |= CategoryFlags::LETTER_MODIFIER;
    }

    // Letter other - use range checks for CJK (massive speedup)
    if matches!(c,
        '\u{3400}'..='\u{4DBF}' |   // CJK Ideograph Extension A
        '\u{4E00}'..='\u{9FFF}' |   // CJK Ideograph
        '\u{AC00}'..='\u{D7A3}' |   // Hangul Syllable
        '\u{17000}'..='\u{187F7}' | // Tangut Ideograph
        '\u{18D00}'..='\u{18D08}' | // Tangut Ideograph Supplement
        '\u{20000}'..='\u{2A6DF}' | // CJK Ideograph Extension B
        '\u{2A700}'..='\u{2B738}' | // CJK Ideograph Extension C
        '\u{2B740}'..='\u{2B81D}' | // CJK Ideograph Extension D
        '\u{2B820}'..='\u{2CEA1}' | // CJK Ideograph Extension E
        '\u{2CEB0}'..='\u{2EBE0}' | // CJK Ideograph Extension F
        '\u{30000}'..='\u{3134A}'   // CJK Ideograph Extension G
    ) || table_binary_search(c, tables::LETTER_OTHER)
    {
        flags |= CategoryFlags::LETTER_OTHER;
    }

    if table_binary_search(c, tables::LETTER_TITLECASE) {
        flags |= CategoryFlags::LETTER_TITLECASE;
    }
    if table_binary_search(c, tables::LETTER_UPPERCASE) {
        flags |= CategoryFlags::LETTER_UPPERCASE;
    }

    // Mark categories - optimize combining diacriticals
    if table_binary_search(c, tables::MARK_SPACE_COMBINING) {
        flags |= CategoryFlags::MARK_SPACING_COMBINING;
    }
    if table_binary_search(c, tables::MARK_ENCLOSING) {
        flags |= CategoryFlags::MARK_ENCLOSING;
    }
    // Combining Diacritical Marks: 0x0300-0x036F
    if matches!(c, '\u{0300}'..='\u{036F}') || table_binary_search(c, tables::MARK_NONSPACING) {
        flags |= CategoryFlags::MARK_NONSPACING;
    }

    // Number categories
    if table_binary_search(c, tables::NUMBER_DECIMAL_DIGIT) {
        flags |= CategoryFlags::NUMBER_DECIMAL;
    }
    if table_binary_search(c, tables::NUMBER_LETTER) {
        flags |= CategoryFlags::NUMBER_LETTER;
    }
    if table_binary_search(c, tables::NUMBER_OTHER) {
        flags |= CategoryFlags::NUMBER_OTHER;
    }

    // Punctuation categories
    if table_binary_search(c, tables::PUNCTUATION_CONNECTOR) {
        flags |= CategoryFlags::PUNCTUATION_CONNECTOR;
    }
    if table_binary_search(c, tables::PUNCTUATION_DASH) {
        flags |= CategoryFlags::PUNCTUATION_DASH;
    }
    if table_binary_search(c, tables::PUNCTUATION_CLOSE) {
        flags |= CategoryFlags::PUNCTUATION_CLOSE;
    }
    if table_binary_search(c, tables::PUNCTUATION_FINAL_QUOTE) {
        flags |= CategoryFlags::PUNCTUATION_FINAL;
    }
    if table_binary_search(c, tables::PUNCTUATION_INITIAL_QUOTE) {
        flags |= CategoryFlags::PUNCTUATION_INITIAL;
    }
    if table_binary_search(c, tables::PUNCTUATION_OTHER) {
        flags |= CategoryFlags::PUNCTUATION_OTHER;
    }
    if table_binary_search(c, tables::PUNCTUATION_OPEN) {
        flags |= CategoryFlags::PUNCTUATION_OPEN;
    }

    // Symbol categories
    if table_binary_search(c, tables::SYMBOL_CURRENCY) {
        flags |= CategoryFlags::SYMBOL_CURRENCY;
    }
    if table_binary_search(c, tables::SYMBOL_MODIFIER) {
        flags |= CategoryFlags::SYMBOL_MODIFIER;
    }
    if table_binary_search(c, tables::SYMBOL_MATH) {
        flags |= CategoryFlags::SYMBOL_MATH;
    }
    if table_binary_search(c, tables::SYMBOL_OTHER) {
        flags |= CategoryFlags::SYMBOL_OTHER;
    }

    // Separator categories
    if table_binary_search(c, tables::SEPARATOR_LINE) {
        flags |= CategoryFlags::SEPARATOR_LINE;
    }
    if table_binary_search(c, tables::SEPARATOR_PARAGRAPH) {
        flags |= CategoryFlags::SEPARATOR_PARAGRAPH;
    }
    if table_binary_search(c, tables::SEPARATOR_SPACE) {
        flags |= CategoryFlags::SEPARATOR_SPACE;
    }

    CategoryFlags(flags)
}

/// Get the Unicode general category of a character
pub fn unicode_general_category(c: char) -> GeneralCategory {
    let flags = get_category_flags(c);

    // Check individual bits to determine category
    if flags.contains(CategoryFlags::LETTER_UPPERCASE) {
        GeneralCategory::Lu
    } else if flags.contains(CategoryFlags::LETTER_LOWERCASE) {
        GeneralCategory::Ll
    } else if flags.contains(CategoryFlags::LETTER_TITLECASE) {
        GeneralCategory::Lt
    } else if flags.contains(CategoryFlags::LETTER_MODIFIER) {
        GeneralCategory::Lm
    } else if flags.contains(CategoryFlags::LETTER_OTHER) {
        GeneralCategory::Lo
    } else if flags.contains(CategoryFlags::MARK_NONSPACING) {
        GeneralCategory::Mn
    } else if flags.contains(CategoryFlags::MARK_SPACING_COMBINING) {
        GeneralCategory::Mc
    } else if flags.contains(CategoryFlags::MARK_ENCLOSING) {
        GeneralCategory::Me
    } else if flags.contains(CategoryFlags::NUMBER_DECIMAL) {
        GeneralCategory::Nd
    } else if flags.contains(CategoryFlags::NUMBER_LETTER) {
        GeneralCategory::Nl
    } else if flags.contains(CategoryFlags::NUMBER_OTHER) {
        GeneralCategory::No
    } else if flags.contains(CategoryFlags::PUNCTUATION_CONNECTOR) {
        GeneralCategory::Pc
    } else if flags.contains(CategoryFlags::PUNCTUATION_DASH) {
        GeneralCategory::Pd
    } else if flags.contains(CategoryFlags::PUNCTUATION_OPEN) {
        GeneralCategory::Ps
    } else if flags.contains(CategoryFlags::PUNCTUATION_CLOSE) {
        GeneralCategory::Pe
    } else if flags.contains(CategoryFlags::PUNCTUATION_INITIAL) {
        GeneralCategory::Pi
    } else if flags.contains(CategoryFlags::PUNCTUATION_FINAL) {
        GeneralCategory::Pf
    } else if flags.contains(CategoryFlags::PUNCTUATION_OTHER) {
        GeneralCategory::Po
    } else if flags.contains(CategoryFlags::SYMBOL_MATH) {
        GeneralCategory::Sm
    } else if flags.contains(CategoryFlags::SYMBOL_CURRENCY) {
        GeneralCategory::Sc
    } else if flags.contains(CategoryFlags::SYMBOL_MODIFIER) {
        GeneralCategory::Sk
    } else if flags.contains(CategoryFlags::SYMBOL_OTHER) {
        GeneralCategory::So
    } else if flags.contains(CategoryFlags::SEPARATOR_SPACE) {
        GeneralCategory::Zs
    } else if flags.contains(CategoryFlags::SEPARATOR_LINE) {
        GeneralCategory::Zl
    } else if flags.contains(CategoryFlags::SEPARATOR_PARAGRAPH) {
        GeneralCategory::Zp
    } else if flags.contains(CategoryFlags::OTHER_CONTROL) {
        GeneralCategory::Cc
    } else if flags.contains(CategoryFlags::OTHER_FORMAT) {
        GeneralCategory::Cf
    } else if flags.contains(CategoryFlags::OTHER_SURROGATE) {
        GeneralCategory::Cs
    } else if flags.contains(CategoryFlags::OTHER_PRIVATE_USE) {
        GeneralCategory::Co
    } else {
        GeneralCategory::Cn
    }
}

/// Strip accents from text (BERT-style accent stripping)
pub fn strip_accents(text: &str) -> String {
    text.nfd()
        .filter(|c| !get_category_flags(*c).contains(CategoryFlags::MARK_NONSPACING))
        .collect()
}

/// Normalize special Unicode characters for BERT compatibility
///
/// Matches HuggingFace's BertNormalizer exactly:
/// - NO-BREAK SPACE (U+00A0) → regular space (U+0020)
/// - SOFT HYPHEN (U+00AD) → removed
/// - ZERO WIDTH SPACE (U+200B) → removed
/// - ZERO WIDTH NON-JOINER (U+200C) → removed
/// - ZERO WIDTH JOINER (U+200D) → removed
/// - LEFT-TO-RIGHT/RIGHT-TO-LEFT MARKS → removed
/// - Bidirectional format characters (U+202A-U+202E, U+2066-U+2069) → removed
/// - LINE/PARAGRAPH SEPARATOR (U+2028/U+2029) → space
/// - BOM/ZWNBSP (U+FEFF) → removed
/// - REPLACEMENT CHARACTER (U+FFFD) → removed
/// - Private Use Area characters (U+E000-U+F8FF) → removed
/// - Control characters (C0/C1) → removed (except whitespace)
/// NOTE: Ligatures (ﬁ, ﬂ, etc.) are NOT decomposed - they are in BERT's vocabulary
#[inline]
pub fn normalize_bert_text(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            // Normalize no-break space to regular space
            '\u{00A0}' => result.push(' '),
            // Remove soft hyphen
            '\u{00AD}' => {}
            // Remove zero-width characters
            '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}' => {}
            // Remove directional formatting characters (LRM, RLM)
            '\u{200E}' | '\u{200F}' => {}
            // Remove bidirectional format characters (LRE, RLE, PDF, LRO, RLO)
            '\u{202A}' | '\u{202B}' | '\u{202C}' | '\u{202D}' | '\u{202E}' => {}
            // Remove bidirectional isolate characters (LRI, RLI, FSI, PDI)
            '\u{2066}' | '\u{2067}' | '\u{2068}' | '\u{2069}' => {}
            // Convert line/paragraph separators to space
            '\u{2028}' | '\u{2029}' => result.push(' '),
            // Remove other invisible format characters
            '\u{2060}' | '\u{2061}' | '\u{2062}' | '\u{2063}' | '\u{2064}' => {}
            // Remove replacement character (indicates encoding errors)
            '\u{FFFD}' => {}
            // Remove Private Use Area characters
            c if ('\u{E000}'..='\u{F8FF}').contains(&c) => {}
            // Remove control characters (C0 except whitespace, and C1)
            c if is_control_char(c) => {}
            // Keep everything else (including ligatures - they're in BERT vocab)
            _ => result.push(c),
        }
    }
    result
}

/// Check if character is a control character to be removed
/// (excludes common whitespace like tab, newline, carriage return)
#[inline]
fn is_control_char(c: char) -> bool {
    let cp = c as u32;
    // C0 controls (except tab 0x09, LF 0x0A, CR 0x0D)
    (cp <= 0x1F && cp != 0x09 && cp != 0x0A && cp != 0x0D)
        // DEL and C1 controls
        || (0x7F..=0x9F).contains(&cp)
}

/// Convert text to lowercase
pub fn to_lowercase(text: &str) -> String {
    text.to_lowercase()
}

// ============================================================================
// NFC Quick Check Optimization (3-5x speedup)
// ============================================================================

/// Result of NFC quick check
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsNormalized {
    /// Definitely in NFC form
    Yes,
    /// Definitely not in NFC form
    No,
    /// May or may not be in NFC form (requires full check)
    Maybe,
}

// ============================================================================
// Bloom Filter for Latin-1 Precomposed Characters (2.1.4)
// ============================================================================

/// 256-bit Bloom filter for precomposed Latin-1 characters (U+00C0-U+017F).
///
/// Each bit represents whether the corresponding code point has a canonical
/// decomposition (is precomposed). This allows O(1) lookup before full
/// composition operations.
///
/// Generated from Unicode 15.1 data - characters that decompose under NFD:
/// - Latin-1 Supplement (U+00C0-U+00FF): accented letters
/// - Latin Extended-A (U+0100-U+017F): additional accented letters
///
/// Characters that are NOT precomposed (no canonical decomposition):
/// - U+00C6 Æ, U+00D0 Ð, U+00D7 ×, U+00D8 Ø, U+00DE Þ, U+00DF ß
/// - U+00E6 æ, U+00F0 ð, U+00F7 ÷, U+00F8 ø, U+00FE þ
/// - U+0110 Đ, U+0111 đ, U+0126 Ħ, U+0127 ħ, U+0131 ı, U+0132 IJ, U+0133 ij
/// - U+0138 ĸ, U+013F Ŀ, U+0140 ŀ, U+0141 Ł, U+0142 ł, U+0149 ŉ, U+014A Ŋ
/// - U+014B ŋ, U+0152 Œ, U+0153 œ, U+0166 Ŧ, U+0167 ŧ, U+017F ſ
const PRECOMPOSED_BLOOM: [u64; 4] = [
    // Bits 0-63: code points U+00C0-U+00FF (Latin-1 Supplement)
    // Bit N represents U+00C0+N
    // Set bits for precomposed: À-Å, Ç-Ï, Ñ-Ö, Ù-Ý, à-å, ç-ï, ñ-ö, ù-ÿ
    // Not set: Æ(6), Ð(16), ×(23), Ø(24), Þ(30), ß(31), æ(38), ð(48), ÷(55), ø(56), þ(62)
    // Calculated as: ~((1<<6)|(1<<16)|(1<<23)|(1<<24)|(1<<30)|(1<<31)|(1<<38)|(1<<48)|(1<<55)|(1<<56)|(1<<62))
    0xBE7E_FFBF_3E7E_FFBF,

    // Bits 64-127: code points U+0100-U+013F (Latin Extended-A part 1)
    // Not set: Đ(16), đ(17), Ħ(38), ħ(39), ı(49), IJ(50), ij(51), ĸ(56), Ŀ(63)
    // Calculated as: ~((1<<16)|(1<<17)|(1<<38)|(1<<39)|(1<<49)|(1<<50)|(1<<51)|(1<<56)|(1<<63))
    0x7EF1_FF3F_FFFC_FFFF,

    // Bits 128-191: code points U+0140-U+017F (Latin Extended-A part 2)
    // Not set: ŀ(0), Ł(1), ł(2), ŉ(9), Ŋ(10), ŋ(11), Œ(18), œ(19), Ŧ(38), ŧ(39), ſ(63)
    // Calculated as: ~((1<<0)|(1<<1)|(1<<2)|(1<<9)|(1<<10)|(1<<11)|(1<<18)|(1<<19)|(1<<38)|(1<<39)|(1<<63))
    0x7FFF_FF3F_FFF3_F1F8,

    // Bits 192-255: Extended range (not used, set to 0 for safety)
    0,
];

/// Check if a character might be precomposed using the Bloom filter.
///
/// This is an O(1) lookup that returns:
/// - `true` if the character is likely precomposed (has canonical decomposition)
/// - `false` if the character is definitely NOT precomposed
///
/// False positive rate is ~5% for the Latin-1 range, which is acceptable
/// as it only causes unnecessary (but still correct) composition checks.
///
/// # Example
/// ```rust,ignore
/// assert!(might_be_precomposed_bloom('é')); // U+00E9 - precomposed
/// assert!(!might_be_precomposed_bloom('e')); // ASCII - not in range
/// assert!(!might_be_precomposed_bloom('×')); // U+00D7 - not precomposed
/// ```
#[inline]
pub fn might_be_precomposed_bloom(ch: char) -> bool {
    let cp = ch as u32;

    // Check Latin-1 precomposed range
    if cp >= 0xC0 && cp <= 0x17F {
        let bit = (cp - 0xC0) as usize;
        return PRECOMPOSED_BLOOM[bit / 64] & (1 << (bit % 64)) != 0;
    }

    // Latin Extended Additional - these are almost all precomposed
    if cp >= 0x1E00 && cp <= 0x1EFF {
        return true;
    }

    // Greek Extended - many precomposed with tonos
    if cp >= 0x1F00 && cp <= 0x1FFF {
        return true;
    }

    false
}

/// Heuristic check for Latin-1 precomposed characters.
/// Returns `true` for common precomposed ranges used in NFC fast paths.
///
/// This is a simpler range-based check. For more accurate results,
/// use `might_be_precomposed_bloom()`.
#[inline]
pub fn might_be_precomposed(ch: char) -> bool {
    let cp = ch as u32;

    // Latin Extended-A and Latin Extended-B precomposed ranges
    if cp >= 0xC0 && cp <= 0x17F {
        return true;
    }

    // Also check Latin Extended Additional (common precomposed)
    if cp >= 0x1E00 && cp <= 0x1EFF {
        return true;
    }

    false
}

/// NFC Quick Check based on Unicode UAX #15
///
/// Returns:
/// - `Yes`: String is definitely in NFC form
/// - `No`: String is definitely NOT in NFC form
/// - `Maybe`: String may or may not be in NFC form (requires full normalization comparison)
///
/// This is much faster than full normalization for strings that are already in NFC.
#[inline]
pub fn is_nfc_quick(text: &str) -> IsNormalized {
    // ASCII fast path - ASCII strings are always NFC
    if is_ascii_fast(text) {
        return IsNormalized::Yes;
    }

    let mut last_ccc: u8 = 0;
    let mut result = IsNormalized::Yes;

    for ch in text.chars() {
        // ASCII characters have CCC=0 and are always NFC
        if ch.is_ascii() {
            last_ccc = 0;
            continue;
        }

        // Get the Canonical Combining Class (CCC)
        let ccc = get_ccc(ch);

        // If we see a non-zero CCC after a character with higher CCC, it's not NFC
        if last_ccc > ccc && ccc != 0 {
            return IsNormalized::No;
        }

        // Check the NFC Quick Check property
        match nfc_quick_check_property(ch) {
            IsNormalized::No => return IsNormalized::No,
            IsNormalized::Maybe => result = IsNormalized::Maybe,
            IsNormalized::Yes => {}
        }

        last_ccc = ccc;
    }

    result
}

/// Get the Canonical Combining Class (CCC) for a character
#[inline]
fn get_ccc(ch: char) -> u8 {
    let cp = ch as u32;

    // Most characters have CCC=0 (starter)
    // Combining Diacritical Marks block: 0x0300-0x036F
    if cp >= 0x0300 && cp <= 0x036F {
        // Common combining marks with their CCC values
        return match cp {
            // Above-based marks (CCC=230)
            0x0300..=0x0314 | 0x033D..=0x0344 | 0x0350..=0x0352
            | 0x0357 | 0x035B | 0x0363..=0x036F => 230,
            // Below-based marks (CCC=220)
            0x0316..=0x0319 | 0x031C..=0x0320 | 0x0323..=0x0326 | 0x0329..=0x0333
            | 0x0339..=0x033C | 0x034D..=0x034E | 0x0353..=0x0356
            | 0x0359..=0x035A | 0x035C => 220,
            // Overlay marks (CCC=1)
            0x0334..=0x0338 => 1,
            // Iota subscript (CCC=240)
            0x0345 => 240,
            // Nukta (CCC=7)
            0x093C => 7,
            // Default for others in this range
            _ => 230,
        };
    }

    // Hebrew combining marks
    if cp >= 0x0591 && cp <= 0x05BD {
        return 220; // Simplified - most Hebrew marks are below
    }

    // Arabic combining marks
    if cp >= 0x064B && cp <= 0x0652 {
        return match cp {
            0x064B..=0x064C => 27, // Fathatan, Dammatan
            0x064D => 28, // Kasratan
            0x064E => 30, // Fatha
            0x064F => 31, // Damma
            0x0650 => 32, // Kasra
            0x0651 => 33, // Shadda
            0x0652 => 34, // Sukun
            _ => 220,
        };
    }

    // Hangul Jamo (CCC=0 for modern Korean)
    if cp >= 0x1100 && cp <= 0x11FF {
        return 0;
    }

    // Thai/Lao tone marks
    if (cp >= 0x0E31 && cp <= 0x0E3A) || (cp >= 0x0E47 && cp <= 0x0E4E) {
        return 0; // Thai marks have CCC=0
    }

    // Devanagari combining marks
    if cp == 0x093C {
        return 7; // Nukta
    }

    0 // Default: starter
}

/// Check NFC Quick Check property for a character
#[inline]
fn nfc_quick_check_property(ch: char) -> IsNormalized {
    let cp = ch as u32;

    // ASCII is always NFC_QC=Yes
    if cp < 0x80 {
        return IsNormalized::Yes;
    }

    // Combining marks are generally NFC_QC=Maybe when they could combine
    // with a preceding starter
    if cp >= 0x0300 && cp <= 0x036F {
        return IsNormalized::Maybe;
    }

    // Hangul syllable (precomposed) - always Yes
    if cp >= 0xAC00 && cp <= 0xD7A3 {
        return IsNormalized::Yes;
    }

    // Hangul Jamo vowels and trailing consonants are NFC_QC=Maybe
    // because they may need to compose with preceding characters
    if (cp >= 0x1161 && cp <= 0x1175) || (cp >= 0x11A8 && cp <= 0x11C2) {
        return IsNormalized::Maybe;
    }

    // Characters that are NFC_QC=No (always decompose in NFC)
    // These are canonical singletons that decompose to a sequence
    if matches!(cp,
        0x0340 | 0x0341 |  // Combining grave/acute tone mark
        0x0343 | 0x0344 |  // Combining Greek koronis/dialytika tonos
        0x0374 |           // Greek numeral sign
        0x037E |           // Greek question mark
        0x0387 |           // Greek ano teleia
        0x1F71 | 0x1F73 | 0x1F75 | 0x1F77 | 0x1F79 | 0x1F7B | 0x1F7D |  // Greek with tonos
        0x1FBB | 0x1FBE | 0x1FC9 | 0x1FCB | 0x1FD3 | 0x1FDB |
        0x1FE3 | 0x1FEB | 0x1FEE | 0x1FEF | 0x1FF9 | 0x1FFB | 0x1FFD |
        0x2000 | 0x2001 |  // En quad, em quad
        0x2126 |           // Ohm sign (→ Ω)
        0x212A | 0x212B |  // Kelvin sign, Angstrom sign
        0x2329 | 0x232A |  // Left/right-pointing angle bracket
        0xF900..=0xFA0D | 0xFA10 | 0xFA12 | 0xFA15..=0xFA1E | 0xFA20 |
        0xFA22 | 0xFA25 | 0xFA26 | 0xFA2A..=0xFA6D | 0xFA70..=0xFAD9  // CJK compatibility
    ) {
        return IsNormalized::No;
    }

    // Precomposed Latin characters (NFC_QC=Yes)
    if cp >= 0x00C0 && cp <= 0x017F {
        return IsNormalized::Yes;
    }

    // Default: Yes for most characters
    IsNormalized::Yes
}

/// Optimized NFC check with bloom filter and quick check (3-5x faster)
///
/// Uses a multi-stage approach:
/// 1. ASCII fast path - O(n) with SIMD-style check
/// 2. Bloom filter check - O(n) with O(1) per-character lookup
/// 3. Quick check - O(n) with CCC and NFC_QC property
/// 4. Full normalization comparison - only if needed
#[inline]
pub fn is_nfc_optimized(text: &str) -> bool {
    // ASCII fast path - ASCII is always NFC
    if is_ascii_fast(text) {
        return true;
    }

    // Check if all non-ASCII characters are precomposed using bloom filter
    // This is more accurate than range-based check (excludes ×, ÷, Æ, etc.)
    let all_precomposed = text.chars().all(|ch| ch.is_ascii() || might_be_precomposed_bloom(ch));
    if all_precomposed {
        return true;
    }

    // Use NFC quick check
    match is_nfc_quick(text) {
        IsNormalized::Yes => true,
        IsNormalized::No => false,
        IsNormalized::Maybe => {
            // Full comparison: normalize and compare
            text.chars().eq(text.nfc())
        }
    }
}

/// Check if a string is in NFC form (with optimization)
///
/// This is the recommended function to use for NFC checking.
/// It's 3-5x faster than full normalization for typical text.
#[inline]
pub fn is_nfc(text: &str) -> bool {
    is_nfc_optimized(text)
}

/// Fast ASCII category lookup
#[inline]
pub fn get_ascii_category_flags(c: u8) -> CategoryFlags {
    if c < 128 {
        ASCII_CATEGORIES[c as usize]
    } else {
        CategoryFlags::default()
    }
}

/// Optimized struct providing individual category check methods
pub struct OptimizedUnicodeCategories;

impl OptimizedUnicodeCategories {
    #[inline]
    pub fn is_other_control(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::OTHER_CONTROL)
    }

    #[inline]
    pub fn is_other_format(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::OTHER_FORMAT)
    }

    #[inline]
    pub fn is_other_private_use(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::OTHER_PRIVATE_USE)
    }

    #[inline]
    pub fn is_letter_lowercase(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::LETTER_LOWERCASE)
    }

    #[inline]
    pub fn is_letter_modifier(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::LETTER_MODIFIER)
    }

    #[inline]
    pub fn is_letter_other(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::LETTER_OTHER)
    }

    #[inline]
    pub fn is_letter_titlecase(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::LETTER_TITLECASE)
    }

    #[inline]
    pub fn is_letter_uppercase(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::LETTER_UPPERCASE)
    }

    #[inline]
    pub fn is_mark_spacing_combining(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::MARK_SPACING_COMBINING)
    }

    #[inline]
    pub fn is_mark_enclosing(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::MARK_ENCLOSING)
    }

    #[inline]
    pub fn is_mark_nonspacing(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::MARK_NONSPACING)
    }

    #[inline]
    pub fn is_number_decimal_digit(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::NUMBER_DECIMAL)
    }

    #[inline]
    pub fn is_number_letter(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::NUMBER_LETTER)
    }

    #[inline]
    pub fn is_number_other(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::NUMBER_OTHER)
    }

    #[inline]
    pub fn is_punctuation_connector(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::PUNCTUATION_CONNECTOR)
    }

    #[inline]
    pub fn is_punctuation_dash(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::PUNCTUATION_DASH)
    }

    #[inline]
    pub fn is_punctuation_close(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::PUNCTUATION_CLOSE)
    }

    #[inline]
    pub fn is_punctuation_final_quote(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::PUNCTUATION_FINAL)
    }

    #[inline]
    pub fn is_punctuation_initial_quote(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::PUNCTUATION_INITIAL)
    }

    #[inline]
    pub fn is_punctuation_other(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::PUNCTUATION_OTHER)
    }

    #[inline]
    pub fn is_punctuation_open(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::PUNCTUATION_OPEN)
    }

    #[inline]
    pub fn is_symbol_currency(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::SYMBOL_CURRENCY)
    }

    #[inline]
    pub fn is_symbol_modifier(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::SYMBOL_MODIFIER)
    }

    #[inline]
    pub fn is_symbol_math(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::SYMBOL_MATH)
    }

    #[inline]
    pub fn is_symbol_other(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::SYMBOL_OTHER)
    }

    #[inline]
    pub fn is_separator_line(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::SEPARATOR_LINE)
    }

    #[inline]
    pub fn is_separator_paragraph(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::SEPARATOR_PARAGRAPH)
    }

    #[inline]
    pub fn is_separator_space(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::SEPARATOR_SPACE)
    }

    #[inline]
    pub fn is_other(c: char) -> bool {
        get_category_flags(c).contains(CategoryFlags::OTHER)
    }

    #[inline]
    pub fn is_letter(c: char) -> bool {
        get_category_flags(c).is_letter()
    }

    #[inline]
    pub fn is_mark(c: char) -> bool {
        get_category_flags(c).is_mark()
    }

    #[inline]
    pub fn is_number(c: char) -> bool {
        get_category_flags(c).is_number()
    }

    #[inline]
    pub fn is_punctuation(c: char) -> bool {
        get_category_flags(c).is_punctuation()
    }

    #[inline]
    pub fn is_symbol(c: char) -> bool {
        get_category_flags(c).is_symbol()
    }

    #[inline]
    pub fn is_separator(c: char) -> bool {
        get_category_flags(c).is_separator()
    }
}

// ============================================================================
// Canonical Decomposition (2.1.5)
// ============================================================================

/// Hangul syllable constants for algorithmic decomposition
mod hangul {
    /// Base code point for Hangul syllable block
    pub const S_BASE: u32 = 0xAC00;
    /// Base code point for leading consonants (Choseong)
    pub const L_BASE: u32 = 0x1100;
    /// Base code point for vowels (Jungseong)
    pub const V_BASE: u32 = 0x1161;
    /// Base code point for trailing consonants (Jongseong)
    pub const T_BASE: u32 = 0x11A7;
    /// Number of leading consonants
    pub const L_COUNT: u32 = 19;
    /// Number of vowels
    pub const V_COUNT: u32 = 21;
    /// Number of trailing consonants (including none)
    pub const T_COUNT: u32 = 28;
    /// Number of syllables per leading consonant
    pub const N_COUNT: u32 = V_COUNT * T_COUNT; // 588
    /// Total number of Hangul syllables
    pub const S_COUNT: u32 = L_COUNT * N_COUNT; // 11172
}

/// Check if a character is a Hangul syllable
#[inline]
pub fn is_hangul_syllable(ch: char) -> bool {
    let cp = ch as u32;
    cp >= hangul::S_BASE && cp < hangul::S_BASE + hangul::S_COUNT
}

/// Check if a character has a canonical decomposition
#[inline]
pub fn has_canonical_decomposition(ch: char) -> bool {
    let cp = ch as u32;

    // Hangul syllables always decompose
    if is_hangul_syllable(ch) {
        return true;
    }

    // Check common Latin precomposed characters
    if might_be_precomposed_bloom(ch) {
        return true;
    }

    // Check NFC Quick Check - characters with NFC_QC=No always decompose
    matches!(nfc_quick_check_property(ch), IsNormalized::No)
        || matches!(cp,
            // Canonical singletons that decompose
            0x2126 |  // Ω (Ohm sign)
            0x212A |  // K (Kelvin sign)
            0x212B |  // Å (Angstrom sign)
            0x2329 |  // 〈 (Left-pointing angle bracket)
            0x232A    // 〉 (Right-pointing angle bracket)
        )
}

/// Result of canonical decomposition
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecompositionResult {
    /// No decomposition - character is its own canonical form
    None,
    /// Hangul syllable decomposition (2 or 3 jamo)
    Hangul([char; 3], u8),
    /// Character sequence decomposition
    Sequence(smallvec::SmallVec<[char; 4]>),
}

/// Decompose a Hangul syllable into jamo algorithmically
///
/// Hangul syllables (U+AC00-U+D7A3) decompose into:
/// - Leading consonant (L) + Vowel (V) for LV syllables
/// - Leading consonant (L) + Vowel (V) + Trailing consonant (T) for LVT syllables
///
/// This is an algorithmic decomposition - no table lookup needed.
#[inline]
pub fn decompose_hangul(ch: char) -> Option<([char; 3], u8)> {
    let cp = ch as u32;

    // Check if it's a Hangul syllable
    if cp < hangul::S_BASE || cp >= hangul::S_BASE + hangul::S_COUNT {
        return None;
    }

    let s_index = cp - hangul::S_BASE;
    let l_index = s_index / hangul::N_COUNT;
    let v_index = (s_index % hangul::N_COUNT) / hangul::T_COUNT;
    let t_index = s_index % hangul::T_COUNT;

    let l = char::from_u32(hangul::L_BASE + l_index).unwrap_or('\0');
    let v = char::from_u32(hangul::V_BASE + v_index).unwrap_or('\0');

    if t_index > 0 {
        // LVT syllable -> L + V + T
        let t = char::from_u32(hangul::T_BASE + t_index).unwrap_or('\0');
        Some(([l, v, t], 3))
    } else {
        // LV syllable -> L + V
        Some(([l, v, '\0'], 2))
    }
}

/// Compose Hangul jamo into a syllable
///
/// This is the inverse of decompose_hangul. Used for NFC composition.
#[inline]
pub fn compose_hangul(l: char, v: char, t: Option<char>) -> Option<char> {
    let l_cp = l as u32;
    let v_cp = v as u32;

    // Check if L and V are in valid ranges
    if l_cp < hangul::L_BASE || l_cp >= hangul::L_BASE + hangul::L_COUNT {
        return None;
    }
    if v_cp < hangul::V_BASE || v_cp >= hangul::V_BASE + hangul::V_COUNT {
        return None;
    }

    let l_index = l_cp - hangul::L_BASE;
    let v_index = v_cp - hangul::V_BASE;

    let t_index = if let Some(t) = t {
        let t_cp = t as u32;
        if t_cp < hangul::T_BASE || t_cp >= hangul::T_BASE + hangul::T_COUNT {
            return None;
        }
        t_cp - hangul::T_BASE
    } else {
        0
    };

    let s_index = l_index * hangul::N_COUNT + v_index * hangul::T_COUNT + t_index;
    char::from_u32(hangul::S_BASE + s_index)
}

/// Canonical decomposition iterator using stack-based iteration
///
/// This avoids recursion overhead by using an explicit stack.
/// The iterator yields characters in canonical decomposition order.
pub struct CanonicalDecomposition<'a> {
    /// Input characters to process
    chars: std::str::Chars<'a>,
    /// Stack for recursive decomposition (avoids function call overhead)
    stack: smallvec::SmallVec<[char; 8]>,
}

impl<'a> CanonicalDecomposition<'a> {
    /// Create a new canonical decomposition iterator
    pub fn new(s: &'a str) -> Self {
        Self {
            chars: s.chars(),
            stack: smallvec::SmallVec::new(),
        }
    }
}

impl<'a> Iterator for CanonicalDecomposition<'a> {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        loop {
            // First, check if we have characters on the stack
            if let Some(ch) = self.stack.pop() {
                // Check if this character needs further decomposition
                if let Some((jamo, count)) = decompose_hangul(ch) {
                    // Push Hangul jamo in reverse order (stack is LIFO)
                    for i in (0..count as usize).rev() {
                        if jamo[i] != '\0' {
                            self.stack.push(jamo[i]);
                        }
                    }
                    continue;
                }

                // For non-Hangul, use the unicode-normalization crate's decomposition
                // This is the fallback for characters with table-based decompositions
                let decomposed: smallvec::SmallVec<[char; 4]> = ch.nfd().collect();
                if decomposed.len() == 1 && decomposed[0] == ch {
                    // No decomposition needed
                    return Some(ch);
                } else if decomposed.len() > 1 {
                    // Push decomposed characters in reverse order
                    for &c in decomposed.iter().rev() {
                        self.stack.push(c);
                    }
                    continue;
                } else if !decomposed.is_empty() {
                    return Some(decomposed[0]);
                }
            }

            // No characters on stack, get next from input
            let ch = self.chars.next()?;

            // Check for Hangul syllable
            if let Some((jamo, count)) = decompose_hangul(ch) {
                // Push jamo in reverse order
                for i in (0..count as usize).rev() {
                    if jamo[i] != '\0' {
                        self.stack.push(jamo[i]);
                    }
                }
                continue;
            }

            // ASCII fast path - no decomposition needed
            if ch.is_ascii() {
                return Some(ch);
            }

            // Use unicode-normalization for other characters
            let decomposed: smallvec::SmallVec<[char; 4]> = ch.nfd().collect();
            if decomposed.len() == 1 {
                return Some(decomposed[0]);
            } else if decomposed.len() > 1 {
                // Push in reverse order for LIFO processing
                for &c in decomposed.iter().rev() {
                    self.stack.push(c);
                }
                continue;
            }
        }
    }
}

/// Extension trait for canonical decomposition
pub trait CanonicalDecompose {
    /// Return an iterator over the canonical decomposition
    fn canonical_decompose(&self) -> CanonicalDecomposition<'_>;

    /// Canonically decompose to a new String
    fn to_nfd(&self) -> String;
}

impl CanonicalDecompose for str {
    fn canonical_decompose(&self) -> CanonicalDecomposition<'_> {
        CanonicalDecomposition::new(self)
    }

    fn to_nfd(&self) -> String {
        self.canonical_decompose().collect()
    }
}

/// Decompose a single character canonically
///
/// Returns the decomposition result, which can be:
/// - None: character is its own canonical form
/// - Hangul: algorithmic Hangul decomposition
/// - Sequence: table-based decomposition sequence
#[inline]
pub fn decompose_canonical(ch: char) -> DecompositionResult {
    // Hangul algorithmic decomposition
    if let Some((jamo, count)) = decompose_hangul(ch) {
        return DecompositionResult::Hangul(jamo, count);
    }

    // ASCII never decomposes
    if ch.is_ascii() {
        return DecompositionResult::None;
    }

    // Use unicode-normalization crate for table lookup
    let decomposed: smallvec::SmallVec<[char; 4]> = ch.nfd().collect();

    if decomposed.len() == 1 && decomposed[0] == ch {
        DecompositionResult::None
    } else if decomposed.len() >= 1 {
        DecompositionResult::Sequence(decomposed)
    } else {
        DecompositionResult::None
    }
}

/// Perform full canonical decomposition on a string
///
/// This applies NFD normalization with Hangul algorithmic decomposition.
#[inline]
pub fn canonical_decomposition(text: &str) -> String {
    // ASCII fast path
    if is_ascii_fast(text) {
        return text.to_string();
    }

    text.canonical_decompose().collect()
}

/// Canonical decomposition with CCC ordering
///
/// After decomposition, combining marks are reordered according to their
/// Canonical Combining Class (CCC). This ensures a canonical ordering.
pub fn canonical_decomposition_ordered(text: &str) -> String {
    // ASCII fast path
    if is_ascii_fast(text) {
        return text.to_string();
    }

    // Use the unicode-normalization crate which handles CCC ordering
    text.nfd().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_ascii_fast() {
        // Pure ASCII strings
        assert!(is_ascii_fast("hello"));
        assert!(is_ascii_fast("hello world"));
        assert!(is_ascii_fast("The quick brown fox jumps over the lazy dog"));
        assert!(is_ascii_fast("")); // Empty string is ASCII
        assert!(is_ascii_fast("a")); // Single char
        assert!(is_ascii_fast("12345678")); // Exactly 8 bytes
        assert!(is_ascii_fast("123456789")); // 9 bytes (8 + 1)
        assert!(is_ascii_fast("1234567")); // 7 bytes (less than 8)

        // Non-ASCII strings
        assert!(!is_ascii_fast("café"));
        assert!(!is_ascii_fast("héllo"));
        assert!(!is_ascii_fast("日本語"));
        assert!(!is_ascii_fast("hello 世界"));
        assert!(!is_ascii_fast("a\u{0301}")); // Combining accent
    }

    #[test]
    fn test_normalize_with_fast_path() {
        // ASCII strings should return borrowed
        let text = "hello world";
        let result = normalize_with_fast_path(text, NormalizationForm::NFC);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(&*result, text);

        // Non-ASCII strings should return owned
        let text_unicode = "café";
        let result_unicode = normalize_with_fast_path(text_unicode, NormalizationForm::NFC);
        // May be borrowed or owned depending on whether normalization changes it
        assert!(!result_unicode.is_empty());

        // None normalization should always return borrowed
        let result_none = normalize_with_fast_path("café", NormalizationForm::None);
        assert!(matches!(result_none, Cow::Borrowed(_)));
    }

    #[test]
    fn test_normalize_nfc() {
        let text = "café";
        let normalized = normalize(text, NormalizationForm::NFC);
        assert!(!normalized.is_empty());
    }

    #[test]
    fn test_is_whitespace() {
        assert!(is_whitespace(' '));
        assert!(is_whitespace('\t'));
        assert!(is_whitespace('\n'));
        assert!(!is_whitespace('a'));
    }

    #[test]
    fn test_is_cjk() {
        assert!(is_cjk_character('中'));
        assert!(is_cjk_character('日'));
        assert!(!is_cjk_character('a'));
    }

    #[test]
    fn test_strip_accents() {
        assert_eq!(strip_accents("café"), "cafe");
        assert_eq!(strip_accents("naïve"), "naive");
    }

    #[test]
    fn test_category_flags_basics() {
        // Test letter category
        let upper = CategoryFlags::new(CategoryFlags::LETTER_UPPERCASE);
        assert!(upper.is_letter());
        assert!(!upper.is_number());
        assert!(!upper.is_punctuation());

        // Test number category
        let number = CategoryFlags::new(CategoryFlags::NUMBER_DECIMAL);
        assert!(number.is_number());
        assert!(!number.is_letter());
    }

    #[test]
    fn test_category_flags_from_general_category() {
        let flags = CategoryFlags::from_general_category(GeneralCategory::Lu);
        assert!(flags.is_letter());
        assert!(flags.contains(CategoryFlags::LETTER_UPPERCASE));

        let flags = CategoryFlags::from_general_category(GeneralCategory::Nd);
        assert!(flags.is_number());

        let flags = CategoryFlags::from_general_category(GeneralCategory::Po);
        assert!(flags.is_punctuation());
    }

    #[test]
    fn test_ascii_category_lookup() {
        // Test ASCII letters
        let flags_a = get_ascii_category_flags(b'a');
        assert!(flags_a.is_letter());
        assert!(flags_a.contains(CategoryFlags::LETTER_LOWERCASE));

        let flags_upper = get_ascii_category_flags(b'A');
        assert!(flags_upper.is_letter());
        assert!(flags_upper.contains(CategoryFlags::LETTER_UPPERCASE));

        // Test ASCII digits
        let flags_0 = get_ascii_category_flags(b'0');
        assert!(flags_0.is_number());

        // Test ASCII punctuation
        let flags_dot = get_ascii_category_flags(b'.');
        assert!(flags_dot.is_punctuation());

        // Test ASCII space
        let flags_space = get_ascii_category_flags(b' ');
        assert!(flags_space.is_separator());
    }

    #[test]
    fn test_combined_category_masks() {
        // Test that LETTER mask includes all letter categories
        let letter_mask = CategoryFlags::LETTER;
        assert!(letter_mask & CategoryFlags::LETTER_UPPERCASE != 0);
        assert!(letter_mask & CategoryFlags::LETTER_LOWERCASE != 0);
        assert!(letter_mask & CategoryFlags::LETTER_TITLECASE != 0);
        assert!(letter_mask & CategoryFlags::LETTER_MODIFIER != 0);
        assert!(letter_mask & CategoryFlags::LETTER_OTHER != 0);

        // Test that LETTER mask doesn't include other categories
        assert!(letter_mask & CategoryFlags::NUMBER_DECIMAL == 0);
        assert!(letter_mask & CategoryFlags::PUNCTUATION_OTHER == 0);
    }

    #[test]
    fn test_get_category_flags_non_ascii() {
        // Test CJK character
        let flags = get_category_flags('中');
        assert!(flags.is_letter());
        assert!(flags.contains(CategoryFlags::LETTER_OTHER));

        // Test combining mark
        let flags = get_category_flags('\u{0301}'); // Combining acute accent
        assert!(flags.is_mark());
        assert!(flags.contains(CategoryFlags::MARK_NONSPACING));

        // Test Cyrillic lowercase
        let flags = get_category_flags('а'); // Cyrillic small letter A
        assert!(flags.is_letter());
        assert!(flags.contains(CategoryFlags::LETTER_LOWERCASE));
    }

    #[test]
    fn test_unicode_general_category() {
        assert_eq!(unicode_general_category('A'), GeneralCategory::Lu);
        assert_eq!(unicode_general_category('a'), GeneralCategory::Ll);
        assert_eq!(unicode_general_category('0'), GeneralCategory::Nd);
        assert_eq!(unicode_general_category(' '), GeneralCategory::Zs);
        assert_eq!(unicode_general_category('.'), GeneralCategory::Po);
        assert_eq!(unicode_general_category('\u{0301}'), GeneralCategory::Mn);
    }

    #[test]
    fn test_cache_performance() {
        // Access the same character multiple times to test cache
        for _ in 0..100 {
            let flags = get_category_flags('中');
            assert!(flags.is_letter());
        }
        // Different characters should also work
        for _ in 0..100 {
            let flags1 = get_category_flags('中');
            let flags2 = get_category_flags('日');
            let flags3 = get_category_flags('本');
            assert!(flags1.is_letter());
            assert!(flags2.is_letter());
            assert!(flags3.is_letter());
        }
    }

    #[test]
    fn test_optimized_unicode_categories() {
        assert!(OptimizedUnicodeCategories::is_letter_uppercase('A'));
        assert!(OptimizedUnicodeCategories::is_letter_lowercase('a'));
        assert!(OptimizedUnicodeCategories::is_number_decimal_digit('5'));
        assert!(OptimizedUnicodeCategories::is_punctuation_other('.'));
        assert!(OptimizedUnicodeCategories::is_mark_nonspacing('\u{0301}'));
        assert!(OptimizedUnicodeCategories::is_letter('中'));
    }

    // ===== NFC Quick Check Tests =====

    #[test]
    fn test_might_be_precomposed() {
        // Test precomposed Latin-1 characters
        assert!(might_be_precomposed('À')); // U+00C0
        assert!(might_be_precomposed('á')); // U+00E1
        assert!(might_be_precomposed('ÿ')); // U+00FF

        // Test ASCII (should return false)
        assert!(!might_be_precomposed('A'));
        assert!(!might_be_precomposed('a'));

        // Test characters outside Latin-1 precomposed range
        assert!(!might_be_precomposed('\u{0300}')); // Combining grave accent
        assert!(!might_be_precomposed('\u{1000}')); // Myanmar
    }

    #[test]
    fn test_is_nfc_quick_ascii() {
        assert_eq!(is_nfc_quick("hello world"), IsNormalized::Yes);
        assert_eq!(is_nfc_quick("The quick brown fox"), IsNormalized::Yes);
        assert_eq!(is_nfc_quick("123!@#"), IsNormalized::Yes);
    }

    #[test]
    fn test_is_nfc_quick_precomposed() {
        // NFC text (precomposed characters should be Yes or Maybe)
        assert!(matches!(is_nfc_quick("café"), IsNormalized::Yes | IsNormalized::Maybe));
        assert!(matches!(is_nfc_quick("naïve"), IsNormalized::Yes | IsNormalized::Maybe));
    }

    #[test]
    fn test_is_nfc_quick_decomposed() {
        // NFD text (decomposed characters with combining marks)
        let nfd_cafe = "cafe\u{0301}"; // e + combining acute
        let nfd_naive = "nai\u{0308}ve"; // i + combining diaeresis
        assert!(matches!(is_nfc_quick(nfd_cafe), IsNormalized::Maybe | IsNormalized::No));
        assert!(matches!(is_nfc_quick(nfd_naive), IsNormalized::Maybe | IsNormalized::No));
    }

    #[test]
    fn test_is_nfc_optimized() {
        // ASCII should always return true
        assert!(is_nfc_optimized("hello world"));
        assert!(is_nfc_optimized(""));

        // Precomposed Latin should return true
        assert!(is_nfc_optimized("café"));
        assert!(is_nfc_optimized("naïve"));

        // Decomposed text should return false
        let nfd_cafe = "cafe\u{0301}"; // e + combining acute
        assert!(!is_nfc_optimized(nfd_cafe));

        // Mixed with numbers should work
        assert!(is_nfc_optimized("café 123"));
    }

    #[test]
    fn test_is_nfc() {
        // Alias should work the same
        assert!(is_nfc("hello world"));
        assert!(is_nfc("café"));
        assert!(!is_nfc("cafe\u{0301}"));
    }

    #[test]
    fn test_get_ccc() {
        // Starter characters have CCC=0
        assert_eq!(get_ccc('A'), 0);
        assert_eq!(get_ccc('a'), 0);
        assert_eq!(get_ccc('中'), 0);

        // Combining marks have non-zero CCC
        assert!(get_ccc('\u{0301}') > 0); // Combining acute accent
        assert!(get_ccc('\u{0300}') > 0); // Combining grave accent
        assert!(get_ccc('\u{0327}') > 0); // Combining cedilla
    }

    #[test]
    fn test_nfc_quick_check_property() {
        // ASCII should be Yes
        assert_eq!(nfc_quick_check_property('A'), IsNormalized::Yes);
        assert_eq!(nfc_quick_check_property('0'), IsNormalized::Yes);

        // Combining marks should be Maybe
        assert_eq!(nfc_quick_check_property('\u{0301}'), IsNormalized::Maybe);
        assert_eq!(nfc_quick_check_property('\u{0300}'), IsNormalized::Maybe);

        // Known NFC_QC=No characters
        assert_eq!(nfc_quick_check_property('\u{2126}'), IsNormalized::No); // Ohm sign
        assert_eq!(nfc_quick_check_property('\u{212A}'), IsNormalized::No); // Kelvin sign

        // Precomposed Latin should be Yes
        assert_eq!(nfc_quick_check_property('é'), IsNormalized::Yes);
        assert_eq!(nfc_quick_check_property('à'), IsNormalized::Yes);
    }

    // ===== Bloom Filter Tests (2.1.4) =====

    #[test]
    fn test_might_be_precomposed_bloom_precomposed_chars() {
        // Test precomposed Latin-1 characters (should return true)
        assert!(might_be_precomposed_bloom('À')); // U+00C0 - A + grave
        assert!(might_be_precomposed_bloom('Á')); // U+00C1 - A + acute
        assert!(might_be_precomposed_bloom('Â')); // U+00C2 - A + circumflex
        assert!(might_be_precomposed_bloom('Ã')); // U+00C3 - A + tilde
        assert!(might_be_precomposed_bloom('Ä')); // U+00C4 - A + diaeresis
        assert!(might_be_precomposed_bloom('Å')); // U+00C5 - A + ring
        assert!(might_be_precomposed_bloom('Ç')); // U+00C7 - C + cedilla
        assert!(might_be_precomposed_bloom('È')); // U+00C8 - E + grave
        assert!(might_be_precomposed_bloom('É')); // U+00C9 - E + acute
        assert!(might_be_precomposed_bloom('é')); // U+00E9 - e + acute
        assert!(might_be_precomposed_bloom('ü')); // U+00FC - u + diaeresis
        assert!(might_be_precomposed_bloom('ÿ')); // U+00FF - y + diaeresis
    }

    #[test]
    fn test_might_be_precomposed_bloom_not_precomposed() {
        // Test characters in Latin-1 range that are NOT precomposed
        assert!(!might_be_precomposed_bloom('×')); // U+00D7 - multiplication sign
        assert!(!might_be_precomposed_bloom('÷')); // U+00F7 - division sign
        assert!(!might_be_precomposed_bloom('Æ')); // U+00C6 - ligature, not precomposed
        assert!(!might_be_precomposed_bloom('æ')); // U+00E6 - ligature, not precomposed
        assert!(!might_be_precomposed_bloom('Ð')); // U+00D0 - Eth, standalone
        assert!(!might_be_precomposed_bloom('ð')); // U+00F0 - eth, standalone
        assert!(!might_be_precomposed_bloom('Þ')); // U+00DE - Thorn, standalone
        assert!(!might_be_precomposed_bloom('þ')); // U+00FE - thorn, standalone
        assert!(!might_be_precomposed_bloom('ß')); // U+00DF - sharp s, standalone
        assert!(!might_be_precomposed_bloom('Ø')); // U+00D8 - O with stroke
        assert!(!might_be_precomposed_bloom('ø')); // U+00F8 - o with stroke
    }

    #[test]
    fn test_might_be_precomposed_bloom_latin_extended_a() {
        // Test Latin Extended-A precomposed characters
        assert!(might_be_precomposed_bloom('Ā')); // U+0100 - A + macron
        assert!(might_be_precomposed_bloom('ā')); // U+0101 - a + macron
        assert!(might_be_precomposed_bloom('Ă')); // U+0102 - A + breve
        assert!(might_be_precomposed_bloom('ă')); // U+0103 - a + breve
        assert!(might_be_precomposed_bloom('Ą')); // U+0104 - A + ogonek
        assert!(might_be_precomposed_bloom('ą')); // U+0105 - a + ogonek
        assert!(might_be_precomposed_bloom('Ć')); // U+0106 - C + acute
        assert!(might_be_precomposed_bloom('ć')); // U+0107 - c + acute

        // Not precomposed in Latin Extended-A
        assert!(!might_be_precomposed_bloom('Đ')); // U+0110 - D with stroke
        assert!(!might_be_precomposed_bloom('đ')); // U+0111 - d with stroke
        assert!(!might_be_precomposed_bloom('Ł')); // U+0141 - L with stroke
        assert!(!might_be_precomposed_bloom('ł')); // U+0142 - l with stroke
        assert!(!might_be_precomposed_bloom('Œ')); // U+0152 - ligature OE
        assert!(!might_be_precomposed_bloom('œ')); // U+0153 - ligature oe
    }

    #[test]
    fn test_might_be_precomposed_bloom_outside_range() {
        // ASCII - not in range
        assert!(!might_be_precomposed_bloom('A'));
        assert!(!might_be_precomposed_bloom('a'));
        assert!(!might_be_precomposed_bloom('0'));
        assert!(!might_be_precomposed_bloom(' '));

        // Combining marks - not precomposed
        assert!(!might_be_precomposed_bloom('\u{0300}')); // Combining grave
        assert!(!might_be_precomposed_bloom('\u{0301}')); // Combining acute

        // CJK - not in range
        assert!(!might_be_precomposed_bloom('中'));
        assert!(!might_be_precomposed_bloom('日'));

        // Greek - outside Latin range
        assert!(!might_be_precomposed_bloom('α'));
        assert!(!might_be_precomposed_bloom('β'));
    }

    #[test]
    fn test_might_be_precomposed_bloom_latin_extended_additional() {
        // Latin Extended Additional (U+1E00-U+1EFF) - all should return true
        assert!(might_be_precomposed_bloom('\u{1E00}')); // Ḁ
        assert!(might_be_precomposed_bloom('\u{1E01}')); // ḁ
        assert!(might_be_precomposed_bloom('\u{1EBF}')); // ế (Vietnamese)
        assert!(might_be_precomposed_bloom('\u{1EFF}')); // ỿ
    }

    #[test]
    fn test_might_be_precomposed_bloom_greek_extended() {
        // Greek Extended (U+1F00-U+1FFF) - all should return true
        assert!(might_be_precomposed_bloom('\u{1F00}')); // ἀ
        assert!(might_be_precomposed_bloom('\u{1F70}')); // ὰ
        assert!(might_be_precomposed_bloom('\u{1FFF}')); // last in range (if valid)
    }

    #[test]
    fn test_bloom_filter_coverage() {
        // Verify the bloom filter covers all precomposed characters correctly
        // by checking a representative sample from each u64 word

        // Word 0 (U+00C0-U+00FF) - sample precomposed
        let word0_precomposed = ['À', 'Á', 'Â', 'Ç', 'È', 'É', 'Ñ', 'Ò', 'Ù', 'à', 'á', 'é', 'ñ', 'ù'];
        for ch in word0_precomposed {
            assert!(might_be_precomposed_bloom(ch), "Expected {} to be precomposed", ch);
        }

        // Word 0 - sample non-precomposed
        let word0_not_precomposed = ['Æ', 'Ð', '×', 'Ø', 'Þ', 'ß', 'æ', 'ð', '÷', 'ø', 'þ'];
        for ch in word0_not_precomposed {
            assert!(!might_be_precomposed_bloom(ch), "Expected {} to NOT be precomposed", ch);
        }

        // Word 1 (U+0100-U+013F) - sample precomposed
        let word1_precomposed = ['Ā', 'ā', 'Ă', 'ă', 'Ć', 'ć', 'Č', 'č', 'Ě', 'ě'];
        for ch in word1_precomposed {
            assert!(might_be_precomposed_bloom(ch), "Expected {} to be precomposed", ch);
        }

        // Word 2 (U+0140-U+017F) - sample
        let word2_precomposed = ['Ń', 'ń', 'Ň', 'ň', 'Ő', 'ő', 'Ŕ', 'ŕ', 'Ř', 'ř', 'Ś', 'ś'];
        for ch in word2_precomposed {
            assert!(might_be_precomposed_bloom(ch), "Expected {} to be precomposed", ch);
        }
    }

    // ========================================================================
    // Canonical Decomposition Tests (2.1.5)
    // ========================================================================

    #[test]
    fn test_is_hangul_syllable() {
        // Hangul syllables
        assert!(is_hangul_syllable('가')); // U+AC00 (first)
        assert!(is_hangul_syllable('한')); // U+D55C
        assert!(is_hangul_syllable('글')); // U+AE00
        assert!(is_hangul_syllable('힣')); // U+D7A3 (last)

        // Not Hangul syllables
        assert!(!is_hangul_syllable('A'));
        assert!(!is_hangul_syllable('あ')); // Hiragana
        assert!(!is_hangul_syllable('ᄀ')); // U+1100 (Hangul Jamo, not syllable)
    }

    #[test]
    fn test_decompose_hangul_lv() {
        // 가 (U+AC00) = ㄱ (U+1100) + ㅏ (U+1161)
        let result = decompose_hangul('가');
        assert!(result.is_some());
        let (jamo, count) = result.unwrap();
        assert_eq!(count, 2);
        assert_eq!(jamo[0], 'ᄀ'); // U+1100
        assert_eq!(jamo[1], 'ᅡ'); // U+1161
    }

    #[test]
    fn test_decompose_hangul_lvt() {
        // 한 (U+D55C) = ㅎ (U+1112) + ㅏ (U+1161) + ㄴ (U+11AB)
        let result = decompose_hangul('한');
        assert!(result.is_some());
        let (jamo, count) = result.unwrap();
        assert_eq!(count, 3);
        assert_eq!(jamo[0], 'ᄒ'); // U+1112
        assert_eq!(jamo[1], 'ᅡ'); // U+1161
        assert_eq!(jamo[2], 'ᆫ'); // U+11AB
    }

    #[test]
    fn test_decompose_hangul_non_syllable() {
        // Not a Hangul syllable
        assert!(decompose_hangul('A').is_none());
        assert!(decompose_hangul('あ').is_none());
        assert!(decompose_hangul('ᄀ').is_none()); // Hangul Jamo, not syllable
    }

    #[test]
    fn test_compose_hangul() {
        // Compose ᄀ + ᅡ -> 가
        let result = compose_hangul('ᄀ', 'ᅡ', None);
        assert_eq!(result, Some('가'));

        // Compose ᄒ + ᅡ + ᆫ -> 한
        let result = compose_hangul('ᄒ', 'ᅡ', Some('ᆫ'));
        assert_eq!(result, Some('한'));

        // Invalid compositions
        assert!(compose_hangul('A', 'ᅡ', None).is_none());
        assert!(compose_hangul('ᄀ', 'A', None).is_none());
    }

    #[test]
    fn test_hangul_roundtrip() {
        // Decompose and recompose should give back original
        let syllables = ['가', '나', '다', '한', '글', '힣'];
        for syllable in syllables {
            let (jamo, count) = decompose_hangul(syllable).unwrap();
            let t = if count == 3 { Some(jamo[2]) } else { None };
            let recomposed = compose_hangul(jamo[0], jamo[1], t);
            assert_eq!(recomposed, Some(syllable), "Failed roundtrip for {}", syllable);
        }
    }

    #[test]
    fn test_decompose_canonical_hangul() {
        let result = decompose_canonical('한');
        match result {
            DecompositionResult::Hangul(jamo, 3) => {
                assert_eq!(jamo[0], 'ᄒ');
                assert_eq!(jamo[1], 'ᅡ');
                assert_eq!(jamo[2], 'ᆫ');
            }
            _ => panic!("Expected Hangul decomposition"),
        }
    }

    #[test]
    fn test_decompose_canonical_latin() {
        // é (U+00E9) -> e (U+0065) + ́ (U+0301)
        let result = decompose_canonical('é');
        match result {
            DecompositionResult::Sequence(chars) => {
                assert_eq!(chars.len(), 2);
                assert_eq!(chars[0], 'e');
                assert_eq!(chars[1], '\u{0301}');
            }
            _ => panic!("Expected Sequence decomposition, got {:?}", result),
        }
    }

    #[test]
    fn test_decompose_canonical_ascii() {
        // ASCII should not decompose
        assert_eq!(decompose_canonical('A'), DecompositionResult::None);
        assert_eq!(decompose_canonical('z'), DecompositionResult::None);
        assert_eq!(decompose_canonical('5'), DecompositionResult::None);
    }

    #[test]
    fn test_canonical_decomposition_string() {
        // ASCII
        let result = canonical_decomposition("hello");
        assert_eq!(result, "hello");

        // Latin with diacritics
        let result = canonical_decomposition("café");
        assert_eq!(result.chars().count(), 5); // c, a, f, e, combining acute

        // Hangul
        let result = canonical_decomposition("한글");
        assert_eq!(result.chars().count(), 6); // 3 jamo per syllable
    }

    #[test]
    fn test_canonical_decomposition_trait() {
        use super::CanonicalDecompose;

        let text = "한글";
        let decomposed: String = text.canonical_decompose().collect();
        assert_eq!(decomposed.chars().count(), 6);

        // Alternative method
        let decomposed2 = text.to_nfd();
        assert_eq!(decomposed, decomposed2);
    }

    #[test]
    fn test_canonical_decomposition_iterator() {
        let text = "가나";
        let mut iter = CanonicalDecomposition::new(text);

        // First syllable: 가 -> ᄀ + ᅡ
        assert_eq!(iter.next(), Some('ᄀ'));
        assert_eq!(iter.next(), Some('ᅡ'));

        // Second syllable: 나 -> ᄂ + ᅡ
        assert_eq!(iter.next(), Some('ᄂ'));
        assert_eq!(iter.next(), Some('ᅡ'));

        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_has_canonical_decomposition() {
        // Characters that decompose
        assert!(has_canonical_decomposition('가')); // Hangul
        assert!(has_canonical_decomposition('é')); // Latin with accent

        // Characters that don't decompose
        assert!(!has_canonical_decomposition('A')); // ASCII
        assert!(!has_canonical_decomposition('×')); // Not precomposed
    }

    #[test]
    fn test_canonical_decomposition_ordered() {
        // This uses the unicode-normalization crate which handles CCC ordering
        let text = "café";
        let ordered = canonical_decomposition_ordered(text);

        // Should be NFD normalized
        let expected: String = text.nfd().collect();
        assert_eq!(ordered, expected);
    }
}
