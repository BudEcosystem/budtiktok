# HuggingFace Tokenizers Complete Feature Reference

This document provides a comprehensive catalog of all features, components, and capabilities in the HuggingFace Tokenizers library (Rust implementation, version 0.15.2).

## Table of Contents

1. [Core Architecture](#core-architecture)
2. [Tokenization Models](#tokenization-models)
3. [Normalizers](#normalizers)
4. [Pre-Tokenizers](#pre-tokenizers)
5. [Post-Processors](#post-processors)
6. [Decoders](#decoders)
7. [Encoding Features](#encoding-features)
8. [Added Vocabulary](#added-vocabulary)
9. [Padding & Truncation](#padding--truncation)
10. [Training](#training)
11. [Serialization](#serialization)
12. [Python Bindings](#python-bindings)

---

## Core Architecture

### Main Tokenizer Pipeline
The HuggingFace tokenizer follows a modular pipeline:

```
Input Text
    ↓
[Normalizer] - Text normalization (NFC, lowercase, etc.)
    ↓
[Pre-Tokenizer] - Split into words/tokens
    ↓
[Model] - Apply tokenization algorithm (BPE, WordPiece, etc.)
    ↓
[Post-Processor] - Add special tokens
    ↓
Encoding Output
```

### Core Traits

| Trait | Description | Methods |
|-------|-------------|---------|
| `Model` | Tokenization algorithm | `tokenize()`, `token_to_id()`, `id_to_token()`, `get_vocab()`, `get_vocab_size()`, `save()` |
| `Normalizer` | Text normalization | `normalize()` |
| `PreTokenizer` | Text splitting | `pre_tokenize()` |
| `PostProcessor` | Special token handling | `process()`, `added_tokens()` |
| `Decoder` | Token to text conversion | `decode()`, `decode_chain()` |
| `Trainer` | Vocabulary training | `train()`, `process_tokens()` |

---

## Tokenization Models

### 1. BPE (Byte-Pair Encoding)
**File:** `models/bpe/mod.rs`

| Feature | Description | Status |
|---------|-------------|--------|
| `vocab` | Token to ID mapping | Required |
| `merges` | Merge rules (priority-ordered) | Required |
| `cache` | Token cache for faster encoding | Optional |
| `dropout` | BPE dropout for training robustness | Optional (0.0-1.0) |
| `unk_token` | Unknown token fallback | Optional |
| `continuing_subword_prefix` | Prefix for continuation tokens (e.g., "") | Optional |
| `end_of_word_suffix` | Suffix for word-end tokens (e.g., "</w>") | Optional |
| `fuse_unk` | Merge consecutive unknowns | Default: false |
| `byte_fallback` | Fall back to byte tokens for unknown chars | Default: false |

**Special Variants:**
- Standard BPE (GPT-2 style)
- Byte-Level BPE (maps all bytes to printable chars)
- Character-Level BPE

### 2. WordPiece
**File:** `models/wordpiece/mod.rs`

| Feature | Description | Status |
|---------|-------------|--------|
| `vocab` | Token to ID mapping | Required |
| `unk_token` | Unknown token (default: "[UNK]") | Required |
| `continuing_subword_prefix` | Continuation prefix (default: "##") | Required |
| `max_input_chars_per_word` | Max chars before treating as unknown | Default: 100 |

**Algorithm:** Greedy longest-match-first tokenization.

### 3. WordLevel
**File:** `models/wordlevel/mod.rs`

| Feature | Description | Status |
|---------|-------------|--------|
| `vocab` | Token to ID mapping | Required |
| `unk_token` | Unknown token fallback | Required |

**Algorithm:** Simple word-to-ID lookup (no subword tokenization).

### 4. Unigram
**File:** `models/unigram/mod.rs`

| Feature | Description | Status |
|---------|-------------|--------|
| `vocab` | List of (token, score) pairs | Required |
| `unk_id` | Unknown token ID | Required |
| `byte_fallback` | Use byte tokens for unknown chars | Default: false |
| `min_score` | Minimum score for token consideration | Computed |

**Algorithm:** Viterbi-based maximum likelihood tokenization using log probabilities.

**Special Features:**
- SentencePiece compatibility
- Byte-fallback for unknown characters
- Score-based token selection

---

## Normalizers

**File:** `normalizers/mod.rs`

### Individual Normalizers

| Normalizer | Description | Parameters |
|------------|-------------|------------|
| `BertNormalizer` | BERT-style normalization | `clean_text`, `handle_chinese_chars`, `strip_accents`, `lowercase` |
| `NFD` | Unicode NFD normalization | None |
| `NFC` | Unicode NFC normalization | None |
| `NFKD` | Unicode NFKD normalization | None |
| `NFKC` | Unicode NFKC normalization | None |
| `Lowercase` | Convert to lowercase | None |
| `Strip` | Strip whitespace | `left`, `right` |
| `StripAccents` | Remove accent marks | None |
| `Replace` | Pattern replacement | `pattern`, `content` |
| `Prepend` | Prepend string | `prepend` |
| `Precompiled` | Precompiled normalization table | `precompiled_charsmap` |

### Composite Normalizers

| Normalizer | Description |
|------------|-------------|
| `Sequence` | Apply multiple normalizers in order |

### BertNormalizer Options

```rust
pub struct BertNormalizerOptions {
    clean_text: bool,           // Remove control chars, normalize whitespace
    handle_chinese_chars: bool, // Add spaces around CJK characters
    strip_accents: Option<bool>, // Strip accent marks
    lowercase: bool,            // Convert to lowercase
}
```

---

## Pre-Tokenizers

**File:** `pre_tokenizers/mod.rs`

### Individual Pre-Tokenizers

| Pre-Tokenizer | Description | Parameters |
|---------------|-------------|------------|
| `Whitespace` | Split on whitespace (using regex `\w+\|[^\w\s]+`) | None |
| `WhitespaceSplit` | Split on whitespace only | None |
| `BertPreTokenizer` | BERT-style splitting | None |
| `Metaspace` | SentencePiece-style | `replacement` (▁), `prepend_scheme`, `split` |
| `ByteLevel` | GPT-2 byte-level | `add_prefix_space`, `trim_offsets`, `use_regex` |
| `CharDelimiterSplit` | Split on character | `delimiter` |
| `Split` | Regex-based splitting | `pattern`, `behavior`, `invert` |
| `Punctuation` | Split on punctuation | `behavior` |
| `Digits` | Isolate digits | `individual_digits` |
| `UnicodeScripts` | Split on Unicode script changes | None |

### Composite Pre-Tokenizers

| Pre-Tokenizer | Description |
|---------------|-------------|
| `Sequence` | Apply multiple pre-tokenizers in order |

### Split Behavior Enum

```rust
pub enum SplitDelimiterBehavior {
    Removed,          // Remove delimiter
    Isolated,         // Keep delimiter as separate token
    MergedWithPrevious, // Merge with previous token
    MergedWithNext,   // Merge with next token
    Contiguous,       // Keep contiguous delimiters together
}
```

### Metaspace PrependScheme

```rust
pub enum PrependScheme {
    First,  // Prepend only to first word
    Never,  // Never prepend
    Always, // Always prepend
}
```

### ByteLevel Pre-Tokenizer

Uses GPT-2's byte-to-char mapping for all 256 bytes:
- Bytes 0x21-0x7E and 0xA1-0xFF map directly
- Other bytes map to U+0100+

---

## Post-Processors

**File:** `processors/mod.rs`

### Individual Post-Processors

| Post-Processor | Description | Format |
|----------------|-------------|--------|
| `BertProcessing` | BERT-style | `[CLS] A [SEP]` / `[CLS] A [SEP] B [SEP]` |
| `RobertaProcessing` | RoBERTa-style | `<s> A </s>` / `<s> A </s></s> B </s>` |
| `ByteLevel` | GPT-2 byte-level | Offset trimming |
| `TemplateProcessing` | Custom templates | User-defined |

### TemplateProcessing

Allows custom special token templates using a DSL:

```python
TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B [SEP]",
    special_tokens=[
        ("[CLS]", 101),
        ("[SEP]", 102),
    ]
)
```

Template syntax:
- `$A` - First sequence
- `$B` - Second sequence (pair only)
- `$0`, `$1` - Type IDs
- `TOKEN:N` - Token with type ID N

### Composite Post-Processors

| Post-Processor | Description |
|----------------|-------------|
| `Sequence` | Apply multiple post-processors in order |

---

## Decoders

**File:** `decoders/mod.rs`

### Individual Decoders

| Decoder | Description | Parameters |
|---------|-------------|------------|
| `BPEDecoder` | Standard BPE | `suffix` (default: "</w>") |
| `ByteLevel` | GPT-2 byte-level | None |
| `WordPiece` | WordPiece-style | `prefix` (default: "##"), `cleanup` |
| `Metaspace` | SentencePiece-style | `replacement` (▁), `prepend_scheme` |
| `Fuse` | Fuse all tokens | None |
| `Strip` | Strip characters | `content`, `left`, `right` |
| `Replace` | Pattern replacement | `pattern`, `content` |
| `ByteFallback` | Decode byte tokens | None |
| `CTC` | Connectionist Temporal Classification | `pad_token`, `word_delimiter_token`, `cleanup` |

### Composite Decoders

| Decoder | Description |
|---------|-------------|
| `Sequence` | Apply multiple decoders in order |

---

## Encoding Features

**File:** `tokenizer/encoding.rs`

### Encoding Structure

```rust
pub struct Encoding {
    ids: Vec<u32>,                    // Token IDs
    type_ids: Vec<u32>,               // Segment/Type IDs (0/1 for BERT)
    tokens: Vec<String>,              // Token strings
    offsets: Vec<(usize, usize)>,     // Character offsets in original text
    special_tokens_mask: Vec<u32>,    // 1 for special tokens, 0 otherwise
    attention_mask: Vec<u32>,         // 1 for real tokens, 0 for padding
    word_ids: Vec<Option<u32>>,       // Word index for each token
    sequence_ids: Vec<Option<usize>>, // Sequence ID (None for special)
    overflowing: Vec<Encoding>,       // Overflow encodings from truncation
}
```

### Encoding Methods

| Method | Description |
|--------|-------------|
| `get_ids()` | Get token IDs |
| `get_type_ids()` | Get segment/type IDs |
| `get_tokens()` | Get token strings |
| `get_offsets()` | Get character offsets |
| `get_special_tokens_mask()` | Get special token mask |
| `get_attention_mask()` | Get attention mask |
| `get_word_ids()` | Get word indices |
| `get_sequence_ids()` | Get sequence IDs |
| `get_overflowing()` | Get overflow encodings |
| `len()` | Number of tokens |
| `is_empty()` | Check if empty |
| `truncate()` | Truncate to max length |
| `pad()` | Pad to length |
| `merge()` | Merge with another encoding |
| `char_to_token()` | Map char position to token |
| `char_to_word()` | Map char position to word |
| `token_to_chars()` | Map token to char range |
| `token_to_word()` | Map token to word index |
| `word_to_chars()` | Map word to char range |
| `word_to_tokens()` | Map word to token range |

---

## Added Vocabulary

**File:** `tokenizer/added_vocabulary.rs`

### AddedToken Structure

```rust
pub struct AddedToken {
    content: String,
    single_word: bool,    // Match only at word boundaries
    lstrip: bool,         // Strip left whitespace
    rstrip: bool,         // Strip right whitespace
    normalized: bool,     // Apply normalizer before matching
    special: bool,        // Is this a special token
}
```

### AddedVocabulary Features

| Feature | Description |
|---------|-------------|
| Special token matching | Efficient Aho-Corasick matching |
| Word boundary matching | `single_word` option |
| Whitespace handling | `lstrip`, `rstrip` options |
| Normalization control | `normalized` option |
| Leftmost-longest matching | Prefer longer matches |
| Split-pattern integration | Split around special tokens |

### Vocabulary Management Methods

| Method | Description |
|--------|-------------|
| `add_tokens()` | Add regular tokens |
| `add_special_tokens()` | Add special tokens |
| `get_vocab()` | Get full vocabulary |
| `get_added_tokens_decoder()` | Get added tokens by ID |
| `is_special_token()` | Check if token is special |
| `extract_and_normalize()` | Split text around special tokens |

---

## Padding & Truncation

**File:** `utils/padding.rs`, `utils/truncation.rs`

### Padding Configuration

```rust
pub struct PaddingParams {
    strategy: PaddingStrategy,    // BatchLongest, Fixed(usize)
    direction: PaddingDirection,  // Left, Right
    pad_to_multiple_of: Option<usize>,
    pad_id: u32,
    pad_type_id: u32,
    pad_token: String,
}

pub enum PaddingStrategy {
    BatchLongest,
    Fixed(usize),
}

pub enum PaddingDirection {
    Left,
    Right,
}
```

### Truncation Configuration

```rust
pub struct TruncationParams {
    direction: TruncationDirection,
    max_length: usize,
    strategy: TruncationStrategy,
    stride: usize,
}

pub enum TruncationDirection {
    Left,
    Right,
}

pub enum TruncationStrategy {
    LongestFirst,
    OnlyFirst,
    OnlySecond,
}
```

### Truncation Features

| Feature | Description |
|---------|-------------|
| `max_length` | Maximum sequence length |
| `stride` | Overlap for overflow sequences |
| `strategy` | Which sequence to truncate for pairs |
| `direction` | Truncate from left or right |

---

## Training

**File:** `models/*/trainer.rs`

### Trainer Types

| Trainer | For Model | Key Parameters |
|---------|-----------|----------------|
| `BpeTrainer` | BPE | `vocab_size`, `min_frequency`, `special_tokens`, `limit_alphabet`, `initial_alphabet`, `show_progress` |
| `WordPieceTrainer` | WordPiece | `vocab_size`, `min_frequency`, `special_tokens`, `limit_alphabet`, `initial_alphabet`, `continuing_subword_prefix` |
| `WordLevelTrainer` | WordLevel | `vocab_size`, `min_frequency`, `special_tokens`, `show_progress` |
| `UnigramTrainer` | Unigram | `vocab_size`, `special_tokens`, `shrinking_factor`, `unk_token`, `max_piece_length`, `n_sub_iterations`, `seed_size` |

### BpeTrainer Parameters

```rust
pub struct BpeTrainer {
    min_frequency: u32,           // Minimum token frequency
    vocab_size: usize,            // Target vocabulary size
    show_progress: bool,          // Show progress bar
    special_tokens: Vec<AddedToken>,
    limit_alphabet: Option<usize>, // Limit initial alphabet size
    initial_alphabet: HashSet<char>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    max_token_length: Option<usize>,
}
```

### UnigramTrainer Parameters

```rust
pub struct UnigramTrainer {
    vocab_size: u32,
    n_sub_iterations: u32,        // EM iterations
    shrinking_factor: f64,        // Vocabulary reduction factor
    special_tokens: Vec<AddedToken>,
    initial_alphabet: HashSet<char>,
    unk_token: Option<String>,
    max_piece_length: usize,
    seed_size: usize,             // Initial seed vocabulary size
}
```

---

## Serialization

### JSON Format

The standard `tokenizer.json` format includes:

```json
{
    "version": "1.0",
    "truncation": { /* TruncationParams */ },
    "padding": { /* PaddingParams */ },
    "added_tokens": [ /* AddedToken[] */ ],
    "normalizer": { /* Normalizer config */ },
    "pre_tokenizer": { /* PreTokenizer config */ },
    "post_processor": { /* PostProcessor config */ },
    "decoder": { /* Decoder config */ },
    "model": {
        "type": "BPE|WordPiece|WordLevel|Unigram",
        /* Model-specific config */
    }
}
```

### Methods

| Method | Description |
|--------|-------------|
| `from_str()` | Load from JSON string |
| `from_file()` | Load from file path |
| `from_pretrained()` | Load from HuggingFace Hub |
| `to_string()` | Serialize to JSON string |
| `save()` | Save to file |

---

## Python Bindings

**File:** `python/` (via PyO3)

### Python API

```python
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

# Create tokenizer
tokenizer = Tokenizer(models.BPE())

# Configure components
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),
    normalizers.Lowercase(),
])
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Encode
encoding = tokenizer.encode("Hello world")
print(encoding.ids)
print(encoding.tokens)
print(encoding.offsets)

# Batch encode
encodings = tokenizer.encode_batch(["Hello", "World"])

# Decode
text = tokenizer.decode(encoding.ids)

# Train
from tokenizers.trainers import BpeTrainer
trainer = BpeTrainer(vocab_size=30000, special_tokens=["[UNK]", "[CLS]", "[SEP]"])
tokenizer.train(["data.txt"], trainer)

# Save/Load
tokenizer.save("tokenizer.json")
tokenizer = Tokenizer.from_file("tokenizer.json")
```

### Key Python Features

| Feature | Description |
|---------|-------------|
| `encode()` | Encode single text |
| `encode_batch()` | Encode multiple texts |
| `decode()` | Decode token IDs |
| `decode_batch()` | Decode multiple sequences |
| `train()` | Train from files |
| `train_from_iterator()` | Train from iterator |
| `save()` / `from_file()` | Serialization |
| `from_pretrained()` | Load from HuggingFace Hub |
| `enable_padding()` | Enable padding |
| `enable_truncation()` | Enable truncation |
| `add_tokens()` | Add tokens |
| `add_special_tokens()` | Add special tokens |
| `get_vocab()` | Get vocabulary |
| `get_vocab_size()` | Get vocabulary size |
| `token_to_id()` | Token to ID lookup |
| `id_to_token()` | ID to token lookup |

---

## Additional Features

### Offset Tracking
- Character-level offset tracking through entire pipeline
- Byte-to-char conversion for byte-level models
- Word-to-token mapping

### Parallelism
- Built-in Rayon parallelism for batch operations
- `encode_batch_par()` for parallel encoding
- Thread-safe tokenizer

### Caching
- Token cache in BPE model
- Vocabulary cache for fast lookups

### Error Handling
- `Result<T, Error>` types
- Descriptive error messages
- Graceful fallbacks (unk tokens)

---

## Version History

| Version | Key Changes |
|---------|-------------|
| 0.15.x | Current stable, Python 3.12 support |
| 0.14.x | Added Sequence post-processor |
| 0.13.x | Improved byte-level handling |
| 0.12.x | Added ByteFallback decoder |
| 0.11.x | Template processing improvements |
| 0.10.x | Performance improvements |
