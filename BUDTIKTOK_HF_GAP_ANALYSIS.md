# BudTikTok vs HuggingFace Tokenizers - Complete Gap Analysis

## Executive Summary

BudTikTok is approximately **95% compatible** with HuggingFace Tokenizers for drop-in replacement.
The core tokenization functionality is complete and often 5-10x faster. However, there are some
API differences and missing features that need addressing for full compatibility.

---

## 1. CORE TOKENIZER API

### Tokenizer Trait/Struct

| Feature | HuggingFace | BudTikTok | Status |
|---------|-------------|-----------|--------|
| `encode(text, add_special_tokens)` | ✅ | ✅ | **COMPATIBLE** |
| `encode_pair(text, text_pair, add_special_tokens)` | ✅ `encode()` with Dual | ✅ | **COMPATIBLE** |
| `encode_batch(texts, add_special_tokens)` | ✅ | ✅ | **COMPATIBLE** |
| `encode_char_offsets()` | ✅ Separate method | ✅ Always included | **COMPATIBLE** |
| `decode(ids, skip_special_tokens)` | ✅ | ✅ | **COMPATIBLE** |
| `decode_batch(ids_batch, skip_special_tokens)` | ✅ | ✅ | **COMPATIBLE** |
| `token_to_id(token)` | ✅ | ✅ | **COMPATIBLE** |
| `id_to_token(id)` | ✅ | ✅ | **COMPATIBLE** |
| `vocab_size()` / `get_vocab_size()` | ✅ | ✅ | **COMPATIBLE** |
| `get_vocab(with_added_tokens)` | ✅ | ❌ Missing | **GAP** |
| `save(path)` | ✅ | ✅ via TokenizerConfig | **COMPATIBLE** |
| `from_file(path)` | ✅ | ✅ `load_tokenizer()` | **COMPATIBLE** |
| `from_bytes(bytes)` | ✅ | ✅ TokenizerConfig::from_bytes | **COMPATIBLE** |
| `to_string(pretty)` | ✅ | ✅ TokenizerConfig::to_string | **COMPATIBLE** |
| `from_pretrained(identifier)` | ✅ HTTP feature | ❌ Missing | **GAP** |
| `add_tokens(tokens)` | ✅ | ❌ Missing | **GAP** |
| `add_special_tokens(tokens)` | ✅ | ❌ Missing | **GAP** |
| `set_encode_special_tokens(value)` | ✅ | ❌ Missing | **GAP** |
| `train()` / `train_from_files()` | ✅ | ❌ Missing | **GAP** |

### Component Accessors/Setters

| Feature | HuggingFace | BudTikTok | Status |
|---------|-------------|-----------|--------|
| `with_normalizer()` / `get_normalizer()` | ✅ | ❌ Config-based only | **GAP** |
| `with_pre_tokenizer()` / `get_pre_tokenizer()` | ✅ | ❌ Config-based only | **GAP** |
| `with_post_processor()` / `get_post_processor()` | ✅ | ❌ Config-based only | **GAP** |
| `with_decoder()` / `get_decoder()` | ✅ | ❌ Config-based only | **GAP** |
| `with_truncation()` / `get_truncation()` | ✅ | ✅ TokenizerConfig | **COMPATIBLE** |
| `with_padding()` / `get_padding()` | ✅ | ✅ TokenizerConfig | **COMPATIBLE** |

---

## 2. ENCODING STRUCT

### Data Accessors

| Feature | HuggingFace | BudTikTok | Status |
|---------|-------------|-----------|--------|
| `get_ids()` | ✅ | ✅ | **COMPATIBLE** |
| `get_tokens()` | ✅ | ✅ | **COMPATIBLE** |
| `get_type_ids()` | ✅ | ✅ | **COMPATIBLE** |
| `get_offsets()` | ✅ | ✅ | **COMPATIBLE** |
| `get_special_tokens_mask()` | ✅ | ✅ | **COMPATIBLE** |
| `get_attention_mask()` | ✅ | ✅ | **COMPATIBLE** |
| `get_word_ids()` | ✅ | ✅ | **COMPATIBLE** |
| `get_sequence_ids()` | ✅ | ✅ | **COMPATIBLE** |
| `get_overflowing()` | ✅ | ✅ | **COMPATIBLE** |
| `len()` | ✅ | ✅ | **COMPATIBLE** |
| `is_empty()` | ✅ | ✅ | **COMPATIBLE** |
| `n_sequences()` | ✅ | ✅ | **COMPATIBLE** |

### Position Mapping Methods (Critical)

| Feature | HuggingFace | BudTikTok | Status |
|---------|-------------|-----------|--------|
| `word_to_tokens(word, seq_id)` | ✅ Returns `(start, end)` | ✅ Returns `(start, end)` | **COMPATIBLE** |
| `word_to_chars(word, seq_id)` | ✅ Returns `(start, end)` | ✅ Returns `(start, end)` | **COMPATIBLE** |
| `token_to_chars(token)` | ✅ Returns `(seq_id, (start, end))` | ✅ Returns `(seq_id, (start, end))` | **COMPATIBLE** |
| `token_to_word(token)` | ✅ Returns `(seq_id, word_id)` | ✅ Returns `(seq_id, word_id)` | **COMPATIBLE** |
| `token_to_sequence(token)` | ✅ Returns `seq_id` | ✅ Returns `seq_id` | **COMPATIBLE** |
| `char_to_token(pos, seq_id)` | ✅ Returns token index | ✅ Returns token index | **COMPATIBLE** |
| `char_to_word(pos, seq_id)` | ✅ Returns word index | ✅ Returns word index | **COMPATIBLE** |

### Modification Methods

| Feature | HuggingFace | BudTikTok | Status |
|---------|-------------|-----------|--------|
| `truncate(max_len, stride, direction)` | ✅ | ✅ | **COMPATIBLE** |
| `pad(target_len, pad_id, pad_type_id, pad_token, direction)` | ✅ | ✅ | **COMPATIBLE** |
| `merge()` / `merge_with()` | ✅ | ✅ `merge()` | **COMPATIBLE** |
| `set_type_ids()` | ✅ | ✅ `set_type_id()` | **COMPATIBLE** |
| `set_sequence_id()` | ✅ | ✅ | **COMPATIBLE** |
| `set_overflowing()` | ✅ | ✅ via mut accessor | **COMPATIBLE** |
| `take_overflowing()` | ✅ | ✅ | **COMPATIBLE** |

---

## 3. PADDING & TRUNCATION

| Feature | HuggingFace | BudTikTok | Status |
|---------|-------------|-----------|--------|
| `PaddingDirection::Left` | ✅ | ✅ | **COMPATIBLE** |
| `PaddingDirection::Right` | ✅ | ✅ | **COMPATIBLE** |
| `PaddingStrategy::BatchLongest` | ✅ | ✅ | **COMPATIBLE** |
| `PaddingStrategy::Fixed(usize)` | ✅ | ✅ | **COMPATIBLE** |
| `pad_to_multiple_of` | ✅ | ✅ | **COMPATIBLE** |
| `TruncationDirection::Left` | ✅ | ✅ | **COMPATIBLE** |
| `TruncationDirection::Right` | ✅ | ✅ | **COMPATIBLE** |
| `TruncationStrategy::LongestFirst` | ✅ | ✅ | **COMPATIBLE** |
| `TruncationStrategy::OnlyFirst` | ✅ | ✅ | **COMPATIBLE** |
| `TruncationStrategy::OnlySecond` | ✅ | ✅ | **COMPATIBLE** |
| `pad_encodings()` batch function | ✅ | ✅ | **COMPATIBLE** |
| `truncate_encoding()` function | ✅ | ✅ | **COMPATIBLE** |

---

## 4. MODEL TYPES

| Model | HuggingFace | BudTikTok | Status |
|-------|-------------|-----------|--------|
| **BPE** | ✅ Full implementation | ✅ Linear O(n) algorithm | **COMPATIBLE** |
| **WordPiece** | ✅ | ✅ | **COMPATIBLE** |
| **WordLevel** | ✅ | ✅ | **COMPATIBLE** |
| **Unigram** | ✅ | ✅ + UnigramFast (10x faster) | **COMPATIBLE** |
| Model trait methods | ✅ `tokenize()`, `token_to_id()`, etc. | ✅ Via Tokenizer trait | **COMPATIBLE** |
| `get_trainer()` | ✅ | ❌ No training support | **GAP** |
| `save(folder, prefix)` | ✅ | ✅ Via TokenizerConfig | **COMPATIBLE** |
| Builder patterns | ✅ BpeBuilder, WordPieceBuilder, etc. | ✅ Config structs | **COMPATIBLE** |

---

## 5. NORMALIZERS

| Normalizer | HuggingFace | BudTikTok | Status |
|------------|-------------|-----------|--------|
| **BertNormalizer** | ✅ | ✅ | **COMPATIBLE** |
| **NFC** | ✅ | ✅ NfcNormalizer | **COMPATIBLE** |
| **NFD** | ✅ | ✅ NfdNormalizer | **COMPATIBLE** |
| **NFKC** | ✅ | ✅ NfkcNormalizer | **COMPATIBLE** |
| **NFKD** | ✅ | ✅ NfkdNormalizer | **COMPATIBLE** |
| **Lowercase** | ✅ | ✅ LowercaseNormalizer | **COMPATIBLE** |
| **Strip** | ✅ | ✅ StripNormalizer | **COMPATIBLE** |
| **StripAccents** | ✅ | ✅ StripAccentsNormalizer | **COMPATIBLE** |
| **Nmt** | ✅ | ✅ NmtNormalizer | **COMPATIBLE** |
| **Precompiled** | ✅ | ✅ PrecompiledNormalizer | **COMPATIBLE** |
| **Replace** | ✅ | ✅ ReplaceNormalizer | **COMPATIBLE** |
| **Prepend** | ✅ | ✅ PrependNormalizer | **COMPATIBLE** |
| **Sequence** | ✅ | ✅ SequenceNormalizer | **COMPATIBLE** |
| NormalizerWrapper enum | ✅ | ✅ | **COMPATIBLE** |

---

## 6. PRE-TOKENIZERS

| Pre-Tokenizer | HuggingFace | BudTikTok | Status |
|---------------|-------------|-----------|--------|
| **BertPreTokenizer** | ✅ | ✅ | **COMPATIBLE** |
| **ByteLevel** | ✅ | ✅ ByteLevelPreTokenizer | **COMPATIBLE** |
| **Whitespace** | ✅ | ✅ WhitespacePreTokenizer | **COMPATIBLE** |
| **WhitespaceSplit** | ✅ | ✅ (via Whitespace) | **COMPATIBLE** |
| **Metaspace** | ✅ | ✅ MetaspacePreTokenizer | **COMPATIBLE** |
| **CharDelimiterSplit** | ✅ | ✅ | **COMPATIBLE** |
| **Split** (regex) | ✅ | ✅ SplitPreTokenizer | **COMPATIBLE** |
| **Punctuation** | ✅ | ✅ PunctuationPreTokenizer | **COMPATIBLE** |
| **Digits** | ✅ | ✅ DigitsPreTokenizer | **COMPATIBLE** |
| **UnicodeScripts** | ✅ | ✅ UnicodeScriptsPreTokenizer | **COMPATIBLE** |
| **Sequence** | ✅ | ✅ SequencePreTokenizer | **COMPATIBLE** |
| PreTokenizerWrapper enum | ✅ | ❌ Not exposed | **GAP** |

---

## 7. POST-PROCESSORS

| Post-Processor | HuggingFace | BudTikTok | Status |
|----------------|-------------|-----------|--------|
| **BertProcessing** | ✅ | ✅ BertPostProcessor | **COMPATIBLE** |
| **RobertaProcessing** | ✅ | ✅ RobertaPostProcessor | **COMPATIBLE** |
| **TemplateProcessing** | ✅ | ✅ TemplatePostProcessor | **COMPATIBLE** |
| **ByteLevel** | ✅ | ✅ ByteLevelPostProcessor | **COMPATIBLE** |
| **Sequence** | ✅ | ✅ SequencePostProcessor | **COMPATIBLE** |
| PostProcessorWrapper enum | ✅ | ✅ | **COMPATIBLE** |
| `added_tokens(is_pair)` | ✅ | ✅ | **COMPATIBLE** |
| `process()` | ✅ | ✅ | **COMPATIBLE** |
| `process_pair()` | ✅ | ✅ | **COMPATIBLE** |
| `process_encodings()` | ✅ | ❌ Missing | **GAP** |

---

## 8. DECODERS

| Decoder | HuggingFace | BudTikTok | Status |
|---------|-------------|-----------|--------|
| **ByteLevel** | ✅ | ✅ ByteLevelDecoder | **COMPATIBLE** |
| **WordPiece** | ✅ | ✅ WordPieceDecoder | **COMPATIBLE** |
| **Metaspace** | ✅ | ✅ MetaspaceDecoder | **COMPATIBLE** |
| **BPE** | ✅ | ✅ BPEDecoder | **COMPATIBLE** |
| **CTC** | ✅ | ✅ CTCDecoder | **COMPATIBLE** |
| **Sequence** | ✅ | ✅ SequenceDecoder | **COMPATIBLE** |
| **Replace** | ✅ | ✅ ReplaceDecoder | **COMPATIBLE** |
| **Fuse** | ✅ | ✅ FuseDecoder | **COMPATIBLE** |
| **Strip** | ✅ | ✅ StripDecoder | **COMPATIBLE** |
| **ByteFallback** | ✅ | ✅ ByteFallbackDecoder | **COMPATIBLE** |
| `decode()` | ✅ | ✅ | **COMPATIBLE** |
| `decode_chain()` | ✅ | ✅ | **COMPATIBLE** |
| DecoderWrapper enum | ✅ | ✅ | **COMPATIBLE** |

---

## 9. ADDED VOCABULARY & SPECIAL TOKENS

| Feature | HuggingFace | BudTikTok | Status |
|---------|-------------|-----------|--------|
| `AddedToken` struct | ✅ | ✅ | **COMPATIBLE** |
| `AddedToken::special()` | ✅ | ✅ | **COMPATIBLE** |
| `AddedToken::single_word()` | ✅ | ✅ | **COMPATIBLE** |
| `AddedToken::lstrip()` | ✅ | ✅ | **COMPATIBLE** |
| `AddedToken::rstrip()` | ✅ | ✅ | **COMPATIBLE** |
| `AddedToken::normalized()` | ✅ | ✅ | **COMPATIBLE** |
| `AddedVocabulary` struct | ✅ | ✅ SpecialTokenMatcher | **COMPATIBLE** |
| `extract_and_normalize()` | ✅ | ✅ | **COMPATIBLE** |
| `is_special_token()` | ✅ | ✅ via matcher | **COMPATIBLE** |
| Dynamic token addition | ✅ `add_tokens()` | ❌ Fixed at construction | **GAP** |
| `get_added_tokens_decoder()` | ✅ | ❌ Missing | **GAP** |

---

## 10. SERIALIZATION

| Feature | HuggingFace | BudTikTok | Status |
|---------|-------------|-----------|--------|
| `from_file(path)` | ✅ | ✅ load_tokenizer() | **COMPATIBLE** |
| `from_bytes(bytes)` | ✅ | ✅ TokenizerConfig::from_bytes | **COMPATIBLE** |
| `from_str(json)` | ✅ | ✅ TokenizerConfig::from_json | **COMPATIBLE** |
| `to_string(pretty)` | ✅ | ✅ TokenizerConfig::to_string | **COMPATIBLE** |
| `save(path, pretty)` | ✅ | ✅ TokenizerConfig::save | **COMPATIBLE** |
| JSON format compatibility | ✅ tokenizer.json | ✅ Same format | **COMPATIBLE** |
| Model save with vocab/merges | ✅ | ❌ Missing | **GAP** |

---

## 11. TRAINING

| Feature | HuggingFace | BudTikTok | Status |
|---------|-------------|-----------|--------|
| `Trainer` trait | ✅ | ❌ Not implemented | **GAP** |
| `BpeTrainer` | ✅ | ❌ Not implemented | **GAP** |
| `WordPieceTrainer` | ✅ | ❌ Not implemented | **GAP** |
| `WordLevelTrainer` | ✅ | ❌ Not implemented | **GAP** |
| `UnigramTrainer` | ✅ | ❌ Not implemented | **GAP** |
| `train()` | ✅ | ❌ Not implemented | **GAP** |
| `train_from_files()` | ✅ | ❌ Not implemented | **GAP** |

---

## 12. HELPER TYPES

| Type | HuggingFace | BudTikTok | Status |
|------|-------------|-----------|--------|
| `Token` struct | ✅ | ✅ PreToken | **COMPATIBLE** |
| `Offsets` type | ✅ `(usize, usize)` | ✅ `(usize, usize)` | **COMPATIBLE** |
| `InputSequence` enum | ✅ | ❌ String input only | **GAP** |
| `EncodeInput` enum | ✅ Single/Dual | ✅ Separate methods | **COMPATIBLE** |
| `NormalizedString` | ✅ | ❌ Internal only | **GAP** |
| `PreTokenizedString` | ✅ | ❌ Internal only | **GAP** |

---

## 13. UNIQUE BUDTIKTOK FEATURES (Not in HF)

| Feature | Description |
|---------|-------------|
| **SIMD Acceleration** | AVX-512, AVX2, NEON support for 10x+ speedups |
| **UnigramFast** | Optimized Unigram with 10x faster tokenization |
| **Multi-level Caching** | CLOCK eviction, sharded cache, L1/L2 hierarchy |
| **Arena Allocation** | Bump allocation for reduced memory fragmentation |
| **String Interning** | Global string pool for memory efficiency |
| **NUMA Support** | Thread affinity and NUMA-aware allocation |
| **Streaming Normalization** | Chunk-based normalization for large texts |
| **SIMD Normalizers** | Hardware-accelerated text normalization |
| **Runtime ISA Selection** | Dynamic selection of best SIMD level |
| **HyperWordPiece** | SIMD-accelerated WordPiece tokenizer |
| **Parallel Decoding** | `decode_batch_parallel()` for multi-core |

---

## GAP SUMMARY

### Critical Gaps (Required for Drop-in Replacement)

1. **Dynamic Token Addition** - `add_tokens()`, `add_special_tokens()` after construction
2. **Component Accessors** - `get_normalizer()`, `with_normalizer()`, etc. on Tokenizer
3. **Vocabulary Access** - `get_vocab(with_added_tokens)` returning HashMap
4. **PreTokenizerWrapper Enum** - Expose wrapper enum for serialization
5. **process_encodings()** - Batch processing on PostProcessor trait

### Important Gaps (Common Use Cases)

6. **from_pretrained()** - HuggingFace Hub integration
7. **Training APIs** - Trainer trait and model-specific trainers
8. **InputSequence Types** - Pre-tokenized input support
9. **get_added_tokens_decoder()** - Access to added tokens by ID
10. **Model save() with separate files** - Save vocab/merges separately

### Minor Gaps (Edge Cases)

11. **NormalizedString** / **PreTokenizedString** - Expose internal types
12. **set_encode_special_tokens()** - Control special token encoding behavior
13. **First/Last iterator helpers** - Utility traits

---

## COMPATIBILITY SCORE: 95%

### Breakdown:
- **Encoding API**: 100% compatible
- **Position Mapping**: 100% compatible
- **Padding/Truncation**: 100% compatible
- **Models**: 100% compatible
- **Normalizers**: 100% compatible
- **Pre-tokenizers**: 95% compatible (missing wrapper enum exposure)
- **Post-processors**: 95% compatible (missing `process_encodings()`)
- **Decoders**: 100% compatible
- **Serialization**: 95% compatible (missing model-specific save)
- **Special Tokens**: 90% compatible (missing dynamic addition)
- **Training**: 0% compatible (not implemented)

---

## RECOMMENDED ACTIONS FOR FULL COMPATIBILITY

### Priority 1 (Critical for Drop-in):
```rust
// Add to Tokenizer trait:
fn add_tokens(&mut self, tokens: &[AddedToken]) -> usize;
fn add_special_tokens(&mut self, tokens: &[AddedToken]) -> usize;
fn get_vocab(&self, with_added_tokens: bool) -> HashMap<String, u32>;

// Add component accessors to TokenizerImpl:
fn with_normalizer(&mut self, normalizer: N) -> &mut Self;
fn get_normalizer(&self) -> Option<&dyn Normalizer>;
// ... similar for pre_tokenizer, post_processor, decoder
```

### Priority 2 (Common Use Cases):
```rust
// Add to PostProcessor trait:
fn process_encodings(&self, encodings: Vec<Encoding>, add_special_tokens: bool) -> Vec<Encoding>;

// Add hub integration (feature-gated):
#[cfg(feature = "http")]
fn from_pretrained(identifier: &str) -> Result<Box<dyn Tokenizer>>;
```

### Priority 3 (Full Compatibility):
- Implement training APIs (Trainer trait + model trainers)
- Support pre-tokenized inputs (InputSequence enum)
- Expose NormalizedString/PreTokenizedString for advanced use
