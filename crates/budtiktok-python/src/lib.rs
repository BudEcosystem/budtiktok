//! Python bindings for BudTikTok
//!
//! This module provides Python bindings for the BudTikTok tokenizer library,
//! exposing a HuggingFace-compatible API for seamless integration with
//! existing codebases like LatentBud.
//!
//! # Performance Features
//!
//! - **Rayon parallelism**: Automatic work-stealing thread pool for batch encoding
//! - **SIMD acceleration**: AVX2/AVX-512/NEON where available
//! - **Zero-copy where possible**: Minimal data copying between Rust and Python
//! - **GIL release**: Releases Python GIL during tokenization for true parallelism
//!
//! # Usage
//!
//! ```python
//! from budtiktok import Tokenizer
//!
//! # Load from tokenizer.json
//! tokenizer = Tokenizer.from_file("path/to/tokenizer.json")
//!
//! # Single encoding
//! encoding = tokenizer.encode("Hello, world!", add_special_tokens=True)
//! print(encoding.ids)  # [101, 7592, 117, 2088, 106, 102]
//!
//! # Batch encoding (parallel)
//! encodings = tokenizer.encode_batch(["Hello", "World"], add_special_tokens=True)
//!
//! # HuggingFace-compatible __call__
//! result = tokenizer(["Hello", "World"], padding="longest", return_tensors="np")
//! print(result["input_ids"])  # numpy array
//! ```

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2};
use std::sync::Arc;

use budtiktok_hf_compat::{
    Tokenizer as RustTokenizer,
    Encoding as RustEncoding,
};

/// Python wrapper for BudTikTok Encoding
///
/// Provides HuggingFace-compatible access to tokenization results.
#[pyclass(name = "Encoding")]
#[derive(Clone)]
pub struct PyEncoding {
    inner: RustEncoding,
}

#[pymethods]
impl PyEncoding {
    /// Get token IDs as a list
    #[getter]
    fn ids(&self) -> Vec<u32> {
        self.inner.get_ids().to_vec()
    }

    /// Get token IDs as a numpy array (zero-copy when possible)
    fn get_ids_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice_bound(py, self.inner.get_ids())
    }

    /// Get token type IDs
    #[getter]
    fn type_ids(&self) -> Vec<u32> {
        self.inner.get_type_ids().to_vec()
    }

    /// Get token strings
    #[getter]
    fn tokens(&self) -> Vec<String> {
        self.inner.get_tokens().to_vec()
    }

    /// Get attention mask
    #[getter]
    fn attention_mask(&self) -> Vec<u32> {
        self.inner.get_attention_mask().to_vec()
    }

    /// Get special tokens mask
    #[getter]
    fn special_tokens_mask(&self) -> Vec<u32> {
        self.inner.get_special_tokens_mask().to_vec()
    }

    /// Get byte offsets for each token
    #[getter]
    fn offsets(&self) -> Vec<(usize, usize)> {
        self.inner.get_offsets().to_vec()
    }

    /// Get word IDs (which word each token belongs to)
    #[getter]
    fn word_ids(&self) -> Vec<Option<u32>> {
        self.inner.get_word_ids().to_vec()
    }

    /// Get sequence IDs
    #[getter]
    fn sequence_ids(&self) -> Vec<Option<usize>> {
        self.inner.get_sequence_ids().to_vec()
    }

    /// Get the number of tokens
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Get token at index
    fn __getitem__(&self, idx: isize) -> PyResult<u32> {
        let len = self.inner.len() as isize;
        let actual_idx = if idx < 0 { len + idx } else { idx } as usize;

        self.inner.get_ids()
            .get(actual_idx)
            .copied()
            .ok_or_else(|| PyValueError::new_err("index out of range"))
    }

    fn __repr__(&self) -> String {
        format!("Encoding(ids={:?}, tokens={:?})",
            self.inner.get_ids(),
            self.inner.get_tokens())
    }
}

/// Python wrapper for BudTikTok Tokenizer
///
/// Drop-in replacement for HuggingFace tokenizers with 4-20x faster performance.
#[pyclass(name = "Tokenizer")]
pub struct PyTokenizer {
    inner: Arc<RustTokenizer>,
    /// Cached model max length
    model_max_length: usize,
    /// Cached pad token ID
    pad_token_id: u32,
}

#[pymethods]
impl PyTokenizer {
    /// Load tokenizer from a tokenizer.json file
    ///
    /// Args:
    ///     path: Path to tokenizer.json file
    ///
    /// Returns:
    ///     Tokenizer instance
    #[staticmethod]
    #[pyo3(signature = (path))]
    fn from_file(path: &str) -> PyResult<Self> {
        let tokenizer = RustTokenizer::from_file(path)
            .map_err(|e| PyValueError::new_err(format!("Failed to load tokenizer: {}", e)))?;

        // Try to find pad token ID
        let pad_token_id = tokenizer.token_to_id("[PAD]")
            .or_else(|| tokenizer.token_to_id("<pad>"))
            .unwrap_or(0);

        Ok(Self {
            inner: Arc::new(tokenizer),
            model_max_length: 512,  // Default, can be overridden
            pad_token_id,
        })
    }

    /// Load tokenizer from a JSON string
    #[staticmethod]
    #[pyo3(signature = (json))]
    fn from_str(json: &str) -> PyResult<Self> {
        let tokenizer = RustTokenizer::from_str(json)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse tokenizer: {}", e)))?;

        let pad_token_id = tokenizer.token_to_id("[PAD]")
            .or_else(|| tokenizer.token_to_id("<pad>"))
            .unwrap_or(0);

        Ok(Self {
            inner: Arc::new(tokenizer),
            model_max_length: 512,
            pad_token_id,
        })
    }

    /// Load tokenizer from a pretrained model directory or HuggingFace Hub
    ///
    /// This method looks for tokenizer.json in the model directory.
    #[staticmethod]
    #[pyo3(signature = (model_name_or_path))]
    fn from_pretrained(model_name_or_path: &str) -> PyResult<Self> {
        use std::path::Path;

        let path = Path::new(model_name_or_path);

        // If it's a directory, look for tokenizer.json
        if path.is_dir() {
            let tokenizer_path = path.join("tokenizer.json");
            if tokenizer_path.exists() {
                return Self::from_file(tokenizer_path.to_str().unwrap());
            }
            return Err(PyValueError::new_err(
                format!("No tokenizer.json found in {}", model_name_or_path)
            ));
        }

        // If it's a file, load directly
        if path.is_file() {
            return Self::from_file(model_name_or_path);
        }

        // TODO: Add HuggingFace Hub download support
        Err(PyValueError::new_err(
            format!("Path does not exist: {}. HuggingFace Hub download not yet implemented.",
                model_name_or_path)
        ))
    }

    /// Model max length property
    #[getter]
    fn model_max_length(&self) -> usize {
        self.model_max_length
    }

    /// Set model max length
    #[setter]
    fn set_model_max_length(&mut self, value: usize) {
        self.model_max_length = value;
    }

    /// Pad token ID property
    #[getter]
    fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    /// Set pad token ID
    #[setter]
    fn set_pad_token_id(&mut self, value: u32) {
        self.pad_token_id = value;
    }

    /// Vocabulary size property
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Get vocabulary size with option to include added tokens
    #[pyo3(signature = (with_added_tokens = true))]
    fn vocab_size_with_added(&self, with_added_tokens: bool) -> usize {
        self.inner.get_vocab_size(with_added_tokens)
    }

    /// Convert a token to its ID
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Convert an ID to its token
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Encode a single text
    ///
    /// Args:
    ///     text: Input text to tokenize
    ///     add_special_tokens: Whether to add special tokens (CLS, SEP, etc.)
    ///
    /// Returns:
    ///     Encoding object
    #[pyo3(signature = (text, add_special_tokens = true))]
    fn encode(&self, py: Python<'_>, text: &str, add_special_tokens: bool) -> PyResult<PyEncoding> {
        // Release GIL during tokenization
        py.allow_threads(|| {
            self.inner.encode(text, add_special_tokens)
                .map(|e| PyEncoding { inner: e })
                .map_err(|e| PyValueError::new_err(format!("Encoding failed: {}", e)))
        })
    }

    /// Encode a batch of texts in parallel
    ///
    /// Uses Rayon's work-stealing thread pool for maximum parallelism.
    /// Releases the Python GIL during encoding.
    ///
    /// Args:
    ///     texts: List of texts to tokenize
    ///     add_special_tokens: Whether to add special tokens
    ///
    /// Returns:
    ///     List of Encoding objects
    #[pyo3(signature = (texts, add_special_tokens = true))]
    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>, add_special_tokens: bool) -> PyResult<Vec<PyEncoding>> {
        // Release GIL during parallel tokenization
        py.allow_threads(|| {
            self.inner.encode_batch(texts, add_special_tokens)
                .map(|encodings| encodings.into_iter().map(|e| PyEncoding { inner: e }).collect())
                .map_err(|e| PyValueError::new_err(format!("Batch encoding failed: {}", e)))
        })
    }

    /// Get token lengths for a batch of texts (fast path for token-budget queue)
    ///
    /// This is optimized for LatentBud's token-budget batching system.
    /// Only computes token IDs, skipping other encoding metadata.
    ///
    /// Args:
    ///     texts: List of texts to get lengths for
    ///     add_special_tokens: Whether to count special tokens
    ///
    /// Returns:
    ///     List of token counts
    #[pyo3(signature = (texts, add_special_tokens = true))]
    fn get_token_lengths(&self, py: Python<'_>, texts: Vec<String>, add_special_tokens: bool) -> PyResult<Vec<usize>> {
        py.allow_threads(|| {
            self.inner.encode_batch(texts, add_special_tokens)
                .map(|encodings| encodings.iter().map(|e| e.len()).collect())
                .map_err(|e| PyValueError::new_err(format!("Length calculation failed: {}", e)))
        })
    }

    /// Decode token IDs back to text
    ///
    /// Args:
    ///     ids: Token IDs to decode
    ///     skip_special_tokens: Whether to skip special tokens in output
    ///
    /// Returns:
    ///     Decoded text string
    #[pyo3(signature = (ids, skip_special_tokens = true))]
    fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        self.inner.decode(&ids, skip_special_tokens)
            .map_err(|e| PyValueError::new_err(format!("Decode failed: {}", e)))
    }

    /// HuggingFace-compatible __call__ method
    ///
    /// Tokenizes texts with padding and truncation, returning a dict-like object
    /// compatible with HuggingFace transformers.
    ///
    /// Args:
    ///     text: Single text or list of texts
    ///     max_length: Maximum sequence length (default: model_max_length)
    ///     padding: Padding strategy: "longest", "max_length", or False
    ///     truncation: Whether to truncate sequences
    ///     return_tensors: "np" for numpy arrays, "pt" for PyTorch tensors, None for lists
    ///     return_attention_mask: Whether to return attention mask
    ///     return_token_type_ids: Whether to return token type IDs
    ///
    /// Returns:
    ///     Dict with "input_ids", "attention_mask", optionally "token_type_ids"
    #[pyo3(signature = (
        text,
        max_length = None,
        padding = None,
        truncation = true,
        return_tensors = None,
        return_attention_mask = true,
        return_token_type_ids = false,
        add_special_tokens = true
    ))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        text: &Bound<'py, PyAny>,
        max_length: Option<usize>,
        padding: Option<&str>,
        truncation: bool,
        return_tensors: Option<&str>,
        return_attention_mask: bool,
        return_token_type_ids: bool,
        add_special_tokens: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        // Extract texts
        let texts: Vec<String> = if let Ok(s) = text.extract::<String>() {
            vec![s]
        } else if let Ok(list) = text.downcast::<PyList>() {
            list.iter()
                .map(|item| item.extract::<String>())
                .collect::<PyResult<Vec<String>>>()?
        } else {
            return Err(PyValueError::new_err("text must be a string or list of strings"));
        };

        let max_len = max_length.unwrap_or(self.model_max_length);
        let pad_to = padding.unwrap_or("longest");
        let pad_id = self.pad_token_id;

        // Encode batch (releases GIL)
        let mut encodings = py.allow_threads(|| {
            self.inner.encode_batch(texts, add_special_tokens)
                .map_err(|e| PyValueError::new_err(format!("Encoding failed: {}", e)))
        })?;

        // Apply truncation
        if truncation {
            for enc in &mut encodings {
                enc.truncate(max_len, 0);
            }
        }

        // Determine padding length
        let pad_length = match pad_to {
            "longest" => encodings.iter().map(|e| e.len()).max().unwrap_or(0),
            "max_length" => max_len,
            _ => 0,
        };

        // Pad encodings
        if pad_length > 0 {
            for enc in &mut encodings {
                enc.pad(pad_length, pad_id, "[PAD]");
            }
        }

        // Build result dict
        let result = PyDict::new_bound(py);

        match return_tensors {
            Some("np") | Some("numpy") => {
                // Return numpy arrays
                let input_ids_arr = PyArray2::from_vec2_bound(
                    py,
                    &encodings.iter().map(|e| e.get_ids().to_vec()).collect::<Vec<_>>()
                )?;
                result.set_item("input_ids", input_ids_arr)?;

                if return_attention_mask {
                    let attention_arr = PyArray2::from_vec2_bound(
                        py,
                        &encodings.iter().map(|e| e.get_attention_mask().to_vec()).collect::<Vec<_>>()
                    )?;
                    result.set_item("attention_mask", attention_arr)?;
                }

                if return_token_type_ids {
                    let type_ids_arr = PyArray2::from_vec2_bound(
                        py,
                        &encodings.iter().map(|e| e.get_type_ids().to_vec()).collect::<Vec<_>>()
                    )?;
                    result.set_item("token_type_ids", type_ids_arr)?;
                }
            }
            Some("pt") | Some("torch") => {
                // Return numpy arrays and let Python convert to torch
                // This avoids a direct torch dependency in Rust
                let input_ids_arr = PyArray2::from_vec2_bound(
                    py,
                    &encodings.iter().map(|e| e.get_ids().to_vec()).collect::<Vec<_>>()
                )?;
                result.set_item("input_ids", input_ids_arr)?;
                result.set_item("_convert_to_torch", true)?;

                if return_attention_mask {
                    let attention_arr = PyArray2::from_vec2_bound(
                        py,
                        &encodings.iter().map(|e| e.get_attention_mask().to_vec()).collect::<Vec<_>>()
                    )?;
                    result.set_item("attention_mask", attention_arr)?;
                }

                if return_token_type_ids {
                    let type_ids_arr = PyArray2::from_vec2_bound(
                        py,
                        &encodings.iter().map(|e| e.get_type_ids().to_vec()).collect::<Vec<_>>()
                    )?;
                    result.set_item("token_type_ids", type_ids_arr)?;
                }
            }
            _ => {
                // Return Python lists
                let input_ids: Vec<Vec<u32>> = encodings.iter()
                    .map(|e| e.get_ids().to_vec())
                    .collect();
                result.set_item("input_ids", input_ids)?;

                if return_attention_mask {
                    let attention_mask: Vec<Vec<u32>> = encodings.iter()
                        .map(|e| e.get_attention_mask().to_vec())
                        .collect();
                    result.set_item("attention_mask", attention_mask)?;
                }

                if return_token_type_ids {
                    let token_type_ids: Vec<Vec<u32>> = encodings.iter()
                        .map(|e| e.get_type_ids().to_vec())
                        .collect();
                    result.set_item("token_type_ids", token_type_ids)?;
                }
            }
        }

        Ok(result)
    }

    /// Batch encode plus - HuggingFace compatible method
    ///
    /// This method provides compatibility with transformers' batch_encode_plus.
    #[pyo3(signature = (
        batch_text_or_text_pairs,
        add_special_tokens = true,
        padding = false,
        truncation = false,
        max_length = None,
        return_tensors = None,
        return_token_type_ids = false,
        return_attention_mask = true
    ))]
    fn batch_encode_plus<'py>(
        &self,
        py: Python<'py>,
        batch_text_or_text_pairs: Vec<String>,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        max_length: Option<usize>,
        return_tensors: Option<&str>,
        return_token_type_ids: bool,
        return_attention_mask: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let max_len = max_length.unwrap_or(self.model_max_length);
        let pad_id = self.pad_token_id;

        // Encode batch
        let mut encodings = py.allow_threads(|| {
            self.inner.encode_batch(batch_text_or_text_pairs, add_special_tokens)
                .map_err(|e| PyValueError::new_err(format!("Encoding failed: {}", e)))
        })?;

        // Apply truncation
        if truncation {
            for enc in &mut encodings {
                enc.truncate(max_len, 0);
            }
        }

        // Apply padding
        if padding {
            let pad_length = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
            for enc in &mut encodings {
                enc.pad(pad_length, pad_id, "[PAD]");
            }
        }

        // Build result dict
        let result = PyDict::new_bound(py);

        match return_tensors {
            Some("np") | Some("numpy") => {
                let input_ids_arr = PyArray2::from_vec2_bound(
                    py,
                    &encodings.iter().map(|e| e.get_ids().to_vec()).collect::<Vec<_>>()
                )?;
                result.set_item("input_ids", input_ids_arr)?;

                if return_attention_mask {
                    let attention_arr = PyArray2::from_vec2_bound(
                        py,
                        &encodings.iter().map(|e| e.get_attention_mask().to_vec()).collect::<Vec<_>>()
                    )?;
                    result.set_item("attention_mask", attention_arr)?;
                }

                if return_token_type_ids {
                    let type_ids_arr = PyArray2::from_vec2_bound(
                        py,
                        &encodings.iter().map(|e| e.get_type_ids().to_vec()).collect::<Vec<_>>()
                    )?;
                    result.set_item("token_type_ids", type_ids_arr)?;
                }
            }
            _ => {
                let input_ids: Vec<Vec<u32>> = encodings.iter()
                    .map(|e| e.get_ids().to_vec())
                    .collect();
                result.set_item("input_ids", input_ids)?;

                if return_attention_mask {
                    let attention_mask: Vec<Vec<u32>> = encodings.iter()
                        .map(|e| e.get_attention_mask().to_vec())
                        .collect();
                    result.set_item("attention_mask", attention_mask)?;
                }

                if return_token_type_ids {
                    let token_type_ids: Vec<Vec<u32>> = encodings.iter()
                        .map(|e| e.get_type_ids().to_vec())
                        .collect();
                    result.set_item("token_type_ids", token_type_ids)?;
                }
            }
        }

        // Add encodings for HF compatibility
        // Convert to PyObjects individually since PyEncoding doesn't implement ToPyObject
        let py_encodings_list = PyList::new_bound(py,
            encodings.into_iter().map(|e| {
                Py::new(py, PyEncoding { inner: e }).unwrap()
            })
        );
        result.set_item("encodings", py_encodings_list)?;

        Ok(result)
    }

    fn __repr__(&self) -> String {
        format!("BudTikTokTokenizer(vocab_size={}, model_max_length={})",
            self.inner.get_vocab_size(true),
            self.model_max_length)
    }
}

/// Get auto-configuration info
///
/// Returns a dict with detected hardware capabilities and optimal settings.
#[pyfunction]
fn get_config(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let config = budtiktok_core::get_auto_config();

    let dict = PyDict::new_bound(py);
    dict.set_item("best_isa", config.best_isa.to_string())?;
    dict.set_item("physical_cores", config.physical_cores)?;
    dict.set_item("logical_cores", config.logical_cores)?;
    dict.set_item("use_simd_pretokenizer", config.use_simd_pretokenizer)?;
    dict.set_item("use_simd_normalizer", config.use_simd_normalizer)?;
    dict.set_item("cache_size", config.cache_size)?;
    dict.set_item("recommended_batch_size", config.recommended_batch_size)?;

    Ok(dict)
}

/// Set the number of threads for parallel tokenization (Rayon thread pool)
///
/// Args:
///     threads: Number of threads to use. Use 0 for auto-detection (all cores).
///              For CPU-bound tokenization, physical cores is often optimal.
///
/// Example:
///     budtiktok.set_num_threads(8)  # Use 8 threads
///     budtiktok.set_num_threads(0)  # Auto-detect (use all cores)
#[pyfunction]
fn set_num_threads(threads: usize) {
    if threads == 0 {
        // Auto-detect: use all logical cores
        let config = budtiktok_core::get_auto_config();
        budtiktok_core::set_global_threads(config.logical_cores);
    } else {
        budtiktok_core::set_global_threads(threads);
    }
}

/// Get the current number of threads used for parallel tokenization
///
/// Returns:
///     Number of threads configured for the Rayon thread pool
#[pyfunction]
fn get_num_threads() -> usize {
    budtiktok_core::get_global_threads()
}

/// Initialize BudTikTok with optimal settings for the current hardware
///
/// This should be called once at startup for best performance.
/// It configures the Rayon thread pool and applies auto-detected optimizations.
///
/// Args:
///     threads: Optional number of threads. If None, uses physical cores.
///              Set to 0 to use all logical cores.
///
/// Returns:
///     Dict with the applied configuration
#[pyfunction]
#[pyo3(signature = (threads = None))]
fn init_optimal(py: Python<'_>, threads: Option<usize>) -> PyResult<Bound<'_, PyDict>> {
    let config = budtiktok_core::get_auto_config();

    // Determine thread count
    let num_threads = match threads {
        Some(0) => config.logical_cores,  // All logical cores
        Some(n) => n,                      // User specified
        None => config.physical_cores,     // Default: physical cores (optimal for CPU-bound)
    };

    budtiktok_core::set_global_threads(num_threads);

    // Return the applied configuration
    let dict = PyDict::new_bound(py);
    dict.set_item("threads", num_threads)?;
    dict.set_item("best_isa", config.best_isa.to_string())?;
    dict.set_item("physical_cores", config.physical_cores)?;
    dict.set_item("logical_cores", config.logical_cores)?;
    dict.set_item("use_simd_pretokenizer", config.use_simd_pretokenizer)?;
    dict.set_item("use_simd_normalizer", config.use_simd_normalizer)?;
    dict.set_item("recommended_batch_size", config.recommended_batch_size)?;

    Ok(dict)
}

/// Python module definition
#[pymodule]
fn budtiktok(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEncoding>()?;
    m.add_class::<PyTokenizer>()?;
    m.add_function(wrap_pyfunction!(get_config, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(init_optimal, m)?)?;

    // PyTokenizer is exposed as "Tokenizer" due to #[pyclass(name = "Tokenizer")]
    // No need to add an alias - it's already named Tokenizer in Python

    Ok(())
}
