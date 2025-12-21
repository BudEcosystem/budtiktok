# BudTikTok HF-Compatible Architecture Design

## Executive Summary

This document outlines the optimal architecture to bridge the critical gaps between BudTikTok and HuggingFace tokenizers while maintaining BudTikTok's 5-10x performance advantage.

---

## Current Architecture Comparison

### HuggingFace Pattern
```rust
// Generic struct with 5 type parameters
struct TokenizerImpl<M, N, PT, PP, D> {
    normalizer: Option<N>,
    pre_tokenizer: Option<PT>,
    model: M,
    post_processor: Option<PP>,
    decoder: Option<D>,
    added_vocabulary: AddedVocabulary,  // Dynamic token management
    truncation: Option<TruncationParams>,
    padding: Option<PaddingParams>,
}

// Concrete type alias for serialization
type Tokenizer = TokenizerImpl<ModelWrapper, NormalizerWrapper, ...>;
```

### Current BudTikTok Pattern
```rust
// Simple trait, concrete implementations
trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding>;
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String>;
    // ... minimal methods
}

// Each tokenizer embeds its own logic
struct WordPieceTokenizer {
    vocabulary: Vocabulary,
    config: WordPieceConfig,
    cache: TokenCache,
    // Normalizer, pre-tokenizer logic is EMBEDDED, not injected
}
```

---

## Proposed Hybrid Architecture

The optimal solution uses a **Pipeline Wrapper** pattern that:
1. Preserves existing fast tokenizers as the "model" component
2. Adds composable pipeline stages around them
3. Maintains performance through lazy evaluation and caching

### Core Design

```
┌─────────────────────────────────────────────────────────────────┐
│                      TokenizerPipeline                          │
│  ┌───────────┐  ┌──────────────┐  ┌───────┐  ┌──────────────┐  │
│  │Normalizer │→ │PreTokenizer  │→ │ Model │→ │PostProcessor │  │
│  │ (Option)  │  │  (Option)    │  │       │  │  (Option)    │  │
│  └───────────┘  └──────────────┘  └───────┘  └──────────────┘  │
│                                       ↓                         │
│  ┌─────────────────┐            ┌─────────┐                     │
│  │ AddedVocabulary │←───────────│ Decoder │                     │
│  │ (Aho-Corasick)  │            │(Option) │                     │
│  └─────────────────┘            └─────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Strategy

### Phase 1: Core Infrastructure

#### 1.1 TokenizerPipeline Struct

```rust
// src/pipeline.rs

use std::sync::Arc;
use parking_lot::RwLock;

/// High-performance tokenizer pipeline with HF-compatible API
pub struct TokenizerPipeline {
    /// The core tokenization model (WordPiece, BPE, Unigram, etc.)
    model: Arc<dyn TokenizerModel>,

    /// Optional normalizer (applied before tokenization)
    normalizer: Option<Arc<dyn Normalizer>>,

    /// Optional pre-tokenizer (splits text before model)
    pre_tokenizer: Option<Arc<dyn PreTokenizer>>,

    /// Optional post-processor (adds special tokens)
    post_processor: Option<Arc<dyn PostProcessor>>,

    /// Optional decoder (converts IDs back to text)
    decoder: Option<Arc<dyn Decoder>>,

    /// Dynamic vocabulary for added tokens (thread-safe)
    added_vocabulary: RwLock<AddedVocabulary>,

    /// Truncation configuration
    truncation: Option<TruncationParams>,

    /// Padding configuration
    padding: Option<PaddingParams>,
}
```

**Key Design Decisions:**
- `Arc<dyn Trait>` for shared ownership and thread safety
- `RwLock<AddedVocabulary>` for dynamic token addition (read-heavy workload)
- Optional components via `Option<T>`
- Separate `TokenizerModel` trait from `Tokenizer` trait

#### 1.2 TokenizerModel Trait (New)

```rust
/// Core tokenization model (the "M" in HF's TokenizerImpl<M, N, PT, PP, D>)
pub trait TokenizerModel: Send + Sync {
    /// Tokenize a single pre-processed segment
    fn tokenize(&self, text: &str) -> Result<Vec<Token>>;

    /// Get token ID for a token string
    fn token_to_id(&self, token: &str) -> Option<u32>;

    /// Get token string for an ID
    fn id_to_token(&self, id: u32) -> Option<&str>;

    /// Get vocabulary size (base, without added tokens)
    fn vocab_size(&self) -> usize;

    /// Get the full vocabulary
    fn get_vocab(&self) -> &Vocabulary;
}

pub struct Token {
    pub id: u32,
    pub value: String,
    pub offsets: (usize, usize),
}
```

**Implementations:**
- `WordPieceModel` - Wraps existing `WordPieceTokenizer`
- `BpeModel` - Wraps existing `BpeTokenizer`
- `UnigramModel` - Wraps existing `UnigramTokenizer`
- `WordLevelModel` - Wraps existing `WordLevelTokenizer`

#### 1.3 AddedVocabulary (Enhanced)

```rust
/// Dynamic vocabulary for user-added tokens
/// Based on HF's AddedVocabulary with BudTikTok optimizations
pub struct AddedVocabulary {
    /// Token content -> ID mapping
    added_tokens_map: AHashMap<String, u32>,

    /// ID -> AddedToken mapping (for decode)
    added_tokens_map_r: AHashMap<u32, AddedToken>,

    /// Regular added tokens (ordered)
    added_tokens: Vec<AddedToken>,

    /// Special tokens (ordered)
    special_tokens: Vec<AddedToken>,

    /// O(1) special token lookup
    special_tokens_set: AHashSet<String>,

    /// Aho-Corasick for non-normalized tokens
    split_trie: Option<(AhoCorasick, Vec<u32>)>,

    /// Aho-Corasick for normalized tokens
    split_normalized_trie: Option<(AhoCorasick, Vec<u32>)>,

    /// Whether to encode special tokens
    encode_special_tokens: bool,

    /// Next available ID for new tokens
    next_id: u32,
}

impl AddedVocabulary {
    /// Add tokens to the vocabulary (HF-compatible)
    pub fn add_tokens<N: Normalizer>(
        &mut self,
        tokens: &[AddedToken],
        model: &dyn TokenizerModel,
        normalizer: Option<&N>,
    ) -> usize {
        // 1. Validate and deduplicate tokens
        // 2. Assign IDs (after model vocab or after existing added)
        // 3. Rebuild Aho-Corasick automata
        // 4. Return count of newly added tokens
    }

    /// Add special tokens (HF-compatible)
    pub fn add_special_tokens<N: Normalizer>(
        &mut self,
        tokens: &[AddedToken],
        model: &dyn TokenizerModel,
        normalizer: Option<&N>,
    ) -> usize {
        // Same as add_tokens but marks as special
    }

    /// Extract and normalize text, identifying added tokens
    /// Two-pass: non-normalized first, then normalized
    pub fn extract_and_normalize<N: Normalizer>(
        &self,
        text: &str,
        normalizer: Option<&N>,
    ) -> Vec<ExtractedSegment> {
        // 1. Find all non-normalized token matches
        // 2. For non-matched segments, apply normalizer
        // 3. Find normalized token matches in normalized segments
        // 4. Return segments with token info
    }
}
```

### Phase 2: Pipeline Methods

#### 2.1 Core Encode/Decode

```rust
impl TokenizerPipeline {
    /// Encode text (HF-compatible signature)
    pub fn encode<'s, E>(&self, input: E, add_special_tokens: bool) -> Result<Encoding>
    where
        E: Into<EncodeInput<'s>>,
    {
        let input = input.into();

        // 1. Extract added tokens and normalize (single pass with Aho-Corasick)
        let added_vocab = self.added_vocabulary.read();
        let segments = added_vocab.extract_and_normalize(
            input.text(),
            self.normalizer.as_deref(),
        );

        // 2. Pre-tokenize non-added segments
        let pre_tokens = if let Some(ref pt) = self.pre_tokenizer {
            segments.flat_map(|seg| {
                if seg.is_added_token {
                    vec![PreToken::from_added(seg)]
                } else {
                    pt.pre_tokenize(&seg.text)
                }
            }).collect()
        } else {
            segments.into_iter().map(PreToken::from_segment).collect()
        };

        // 3. Tokenize each pre-token with model
        let mut encoding = Encoding::with_capacity(pre_tokens.len() * 2);
        for pre_token in pre_tokens {
            if let Some(added_id) = pre_token.added_token_id {
                // Added token - use directly
                encoding.push(added_id, pre_token.text, pre_token.offsets, ...);
            } else {
                // Regular token - use model
                let tokens = self.model.tokenize(&pre_token.text)?;
                for token in tokens {
                    encoding.push(token.id, token.value, ...);
                }
            }
        }

        // 4. Post-process (add special tokens like [CLS], [SEP])
        if add_special_tokens {
            if let Some(ref pp) = self.post_processor {
                encoding = pp.process(encoding, true);
            }
        }

        // 5. Apply truncation if configured
        if let Some(ref trunc) = self.truncation {
            truncate_encoding(&mut encoding, trunc);
        }

        Ok(encoding)
    }

    /// Batch encode with optional parallelization
    pub fn encode_batch<'s, E>(
        &self,
        inputs: Vec<E>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        // Use rayon for parallel encoding
        let mut encodings: Vec<Encoding> = inputs
            .into_par_iter()
            .map(|input| self.encode(input, add_special_tokens))
            .collect::<Result<Vec<_>>>()?;

        // Apply padding after batch (needs all lengths)
        if let Some(ref pad) = self.padding {
            pad_encodings(&mut encodings, pad)?;
        }

        Ok(encodings)
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let added_vocab = self.added_vocabulary.read();

        // 1. Convert IDs to tokens
        let tokens: Vec<String> = ids.iter().map(|&id| {
            // Check added vocabulary first
            if let Some(added) = added_vocab.id_to_token(id) {
                if skip_special_tokens && added.special {
                    return String::new(); // Skip
                }
                return added.content.clone();
            }
            // Fall back to model vocabulary
            self.model.id_to_token(id)
                .map(|s| s.to_string())
                .unwrap_or_default()
        }).collect();

        // 2. Apply decoder chain
        if let Some(ref decoder) = self.decoder {
            decoder.decode(tokens)
        } else {
            Ok(tokens.join(""))
        }
    }
}
```

#### 2.2 Component Accessors/Setters (HF-Compatible)

```rust
impl TokenizerPipeline {
    // === Normalizer ===
    pub fn with_normalizer(&mut self, normalizer: impl Into<NormalizerWrapper>) -> &mut Self {
        self.normalizer = Some(Arc::new(normalizer.into()));
        self
    }

    pub fn get_normalizer(&self) -> Option<&dyn Normalizer> {
        self.normalizer.as_deref()
    }

    // === PreTokenizer ===
    pub fn with_pre_tokenizer(&mut self, pt: impl Into<PreTokenizerWrapper>) -> &mut Self {
        self.pre_tokenizer = Some(Arc::new(pt.into()));
        self
    }

    pub fn get_pre_tokenizer(&self) -> Option<&dyn PreTokenizer> {
        self.pre_tokenizer.as_deref()
    }

    // === PostProcessor ===
    pub fn with_post_processor(&mut self, pp: impl Into<PostProcessorWrapper>) -> &mut Self {
        self.post_processor = Some(Arc::new(pp.into()));
        self
    }

    pub fn get_post_processor(&self) -> Option<&dyn PostProcessor> {
        self.post_processor.as_deref()
    }

    // === Decoder ===
    pub fn with_decoder(&mut self, decoder: impl Into<DecoderWrapper>) -> &mut Self {
        self.decoder = Some(Arc::new(decoder.into()));
        self
    }

    pub fn get_decoder(&self) -> Option<&dyn Decoder> {
        self.decoder.as_deref()
    }

    // === Truncation/Padding ===
    pub fn with_truncation(&mut self, params: Option<TruncationParams>) -> &mut Self {
        self.truncation = params;
        self
    }

    pub fn with_padding(&mut self, params: Option<PaddingParams>) -> &mut Self {
        self.padding = params;
        self
    }
}
```

#### 2.3 Dynamic Token Addition

```rust
impl TokenizerPipeline {
    /// Add tokens to vocabulary (returns count of newly added)
    pub fn add_tokens(&mut self, tokens: &[AddedToken]) -> usize {
        let mut added_vocab = self.added_vocabulary.write();
        added_vocab.add_tokens(
            tokens,
            self.model.as_ref(),
            self.normalizer.as_deref(),
        )
    }

    /// Add special tokens (returns count of newly added)
    pub fn add_special_tokens(&mut self, tokens: &[AddedToken]) -> usize {
        let mut added_vocab = self.added_vocabulary.write();
        added_vocab.add_special_tokens(
            tokens,
            self.model.as_ref(),
            self.normalizer.as_deref(),
        )
    }

    /// Get vocabulary with optional added tokens
    pub fn get_vocab(&self, with_added_tokens: bool) -> HashMap<String, u32> {
        let mut vocab = self.model.get_vocab().to_hashmap();

        if with_added_tokens {
            let added = self.added_vocabulary.read();
            for (token, id) in added.iter() {
                vocab.insert(token.to_string(), id);
            }
        }

        vocab
    }

    /// Get total vocabulary size
    pub fn vocab_size(&self, with_added_tokens: bool) -> usize {
        let base = self.model.vocab_size();
        if with_added_tokens {
            base + self.added_vocabulary.read().len()
        } else {
            base
        }
    }
}
```

### Phase 3: Wrapper Enums for Serialization

#### 3.1 ModelWrapper

```rust
/// Wrapper enum for all model types (enables serialization)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ModelWrapper {
    #[serde(rename = "WordPiece")]
    WordPiece(WordPieceModel),

    #[serde(rename = "BPE")]
    BPE(BpeModel),

    #[serde(rename = "Unigram")]
    Unigram(UnigramModel),

    #[serde(rename = "WordLevel")]
    WordLevel(WordLevelModel),
}

impl TokenizerModel for ModelWrapper {
    fn tokenize(&self, text: &str) -> Result<Vec<Token>> {
        match self {
            Self::WordPiece(m) => m.tokenize(text),
            Self::BPE(m) => m.tokenize(text),
            Self::Unigram(m) => m.tokenize(text),
            Self::WordLevel(m) => m.tokenize(text),
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::WordPiece(m) => m.token_to_id(token),
            Self::BPE(m) => m.token_to_id(token),
            Self::Unigram(m) => m.token_to_id(token),
            Self::WordLevel(m) => m.token_to_id(token),
        }
    }

    // ... other trait methods
}
```

#### 3.2 Updated NormalizerWrapper, PreTokenizerWrapper, etc.

These already exist in BudTikTok. Need to ensure they implement the dispatch pattern consistently.

### Phase 4: Serialization

#### 4.1 Pipeline Serialization

```rust
impl Serialize for TokenizerPipeline {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Tokenizer", 9)?;
        state.serialize_field("version", "1.0")?;
        state.serialize_field("truncation", &self.truncation)?;
        state.serialize_field("padding", &self.padding)?;
        state.serialize_field("added_tokens", &*self.added_vocabulary.read())?;
        state.serialize_field("normalizer", &self.normalizer)?;
        state.serialize_field("pre_tokenizer", &self.pre_tokenizer)?;
        state.serialize_field("post_processor", &self.post_processor)?;
        state.serialize_field("decoder", &self.decoder)?;
        state.serialize_field("model", &self.model)?;
        state.end()
    }
}
```

### Phase 5: from_pretrained (Optional, Feature-Gated)

```rust
#[cfg(feature = "http")]
impl TokenizerPipeline {
    /// Load tokenizer from HuggingFace Hub
    pub fn from_pretrained(
        identifier: &str,
        params: Option<FromPretrainedParams>,
    ) -> Result<Self> {
        // 1. Download tokenizer.json from HF Hub
        let path = download_tokenizer(identifier, params)?;

        // 2. Load from file
        Self::from_file(path)
    }
}

#[cfg(feature = "http")]
fn download_tokenizer(
    identifier: &str,
    params: Option<FromPretrainedParams>,
) -> Result<PathBuf> {
    use hf_hub::{api::sync::Api, Repo, RepoType};

    let api = Api::new()?;
    let repo = Repo::new(identifier.to_string(), RepoType::Model);
    let api = api.repo(repo);

    Ok(api.get("tokenizer.json")?)
}
```

---

## Performance Considerations

### 1. Lazy Component Evaluation

```rust
// Instead of applying normalizer eagerly, use lazy evaluation
struct NormalizedText<'a> {
    original: &'a str,
    normalized: OnceCell<Cow<'a, str>>,
    normalizer: Option<&'a dyn Normalizer>,
}

impl<'a> NormalizedText<'a> {
    fn get(&self) -> &str {
        self.normalized.get_or_init(|| {
            match self.normalizer {
                Some(n) => n.normalize(self.original),
                None => Cow::Borrowed(self.original),
            }
        })
    }
}
```

### 2. RwLock for Added Vocabulary

```rust
// Read-heavy workload - RwLock is optimal
// Writes only happen during add_tokens(), reads during every encode()
added_vocabulary: RwLock<AddedVocabulary>

// Fast path: most encodes don't need write lock
pub fn encode(&self, ...) {
    let vocab = self.added_vocabulary.read(); // Non-blocking for concurrent reads
    // ...
}

pub fn add_tokens(&mut self, ...) {
    let mut vocab = self.added_vocabulary.write(); // Exclusive access
    // Rebuild Aho-Corasick only when tokens change
}
```

### 3. Arc for Shared Components

```rust
// Components are immutable after construction, so Arc is sufficient
// No Mutex needed for components themselves
normalizer: Option<Arc<dyn Normalizer>>

// Clone is cheap (just ref count increment)
let tokenizer_clone = tokenizer.clone(); // O(1)
```

### 4. Preserve Existing Optimizations

```rust
// Model implementations keep their optimizations:
// - CLOCK cache in WordPiece
// - Linear O(n) algorithm in BPE
// - SIMD in pre-tokenizers
// - Static lookup tables in decoders

// Pipeline just wraps, doesn't replace the fast paths
impl WordPieceModel {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        // Uses existing cached tokenization
        self.inner.tokenize_to_tokens(text)
    }
}
```

---

## Migration Path

### Step 1: Add New Types (Non-Breaking)
- Add `TokenizerPipeline` struct
- Add `TokenizerModel` trait
- Add `AddedVocabulary` enhanced version
- Keep existing `Tokenizer` trait and implementations

### Step 2: Implement Wrappers
- `WordPieceModel` wraps `WordPieceTokenizer`
- `BpeModel` wraps `BpeTokenizer`
- etc.

### Step 3: Update Loader
- `load_tokenizer()` returns `TokenizerPipeline` (implements `Tokenizer` trait)
- Old code still works via trait

### Step 4: Deprecate (Future)
- Mark old direct tokenizer usage as deprecated
- Recommend `TokenizerPipeline` for new code

---

## API Comparison

### Before (Current BudTikTok)
```rust
let tokenizer = load_tokenizer("tokenizer.json")?;
let encoding = tokenizer.encode("Hello world", true)?;
// Cannot add tokens dynamically
// Cannot access/modify components
```

### After (With Pipeline)
```rust
let mut tokenizer = TokenizerPipeline::from_file("tokenizer.json")?;

// Dynamic token addition
tokenizer.add_special_tokens(&[
    AddedToken::from("<custom>", true),
]);

// Component access/modification
tokenizer.with_normalizer(NfkcNormalizer);
tokenizer.with_post_processor(BertPostProcessor::new(...));

// Same encoding API
let encoding = tokenizer.encode("Hello world", true)?;

// Get vocab with added tokens
let vocab = tokenizer.get_vocab(true);
```

---

## File Structure

```
src/
├── pipeline/
│   ├── mod.rs              # TokenizerPipeline struct
│   ├── added_vocabulary.rs  # AddedVocabulary with dual Aho-Corasick
│   ├── model.rs            # TokenizerModel trait + wrappers
│   ├── serialization.rs    # Serialize/Deserialize impl
│   └── from_pretrained.rs  # Hub integration (feature-gated)
├── tokenizer.rs            # Existing Tokenizer trait (unchanged)
├── wordpiece.rs            # Existing (unchanged)
├── bpe_linear.rs           # Existing (unchanged)
├── unigram.rs              # Existing (unchanged)
└── ...
```

---

## Summary

This design achieves:

1. **Full HF Compatibility** - Same API surface, same JSON format
2. **Performance Preservation** - Existing optimizations untouched
3. **Extensibility** - Easy to add new models, normalizers, etc.
4. **Thread Safety** - RwLock for dynamic tokens, Arc for components
5. **Backward Compatibility** - Old code continues to work
6. **Minimal Overhead** - Thin wrapper pattern, lazy evaluation
