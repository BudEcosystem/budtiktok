//! Post-processor types compatible with HuggingFace tokenizers
//!
//! These types provide compatibility with HF tokenizers' post-processor API.
//! Currently, these are stub implementations - full post-processor support would
//! require additional work to match HF's TemplateProcessing behavior.

use serde::{Deserialize, Serialize};

/// Post-processor wrapper enum - compatible with tokenizers::PostProcessorWrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PostProcessorWrapper {
    /// Template-based post-processor
    Template(TemplateProcessing),
    /// Sequence of post-processors
    Sequence(Sequence),
}

/// Template processing configuration
///
/// This is a stub implementation. Full template processing would require
/// parsing template strings and applying them during post-processing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemplateProcessing {
    single: Option<String>,
    pair: Option<String>,
    special_tokens: Vec<(String, u32)>,
}

impl TemplateProcessing {
    /// Create a builder for TemplateProcessing
    pub fn builder() -> TemplateProcessingBuilder {
        TemplateProcessingBuilder::default()
    }
}

/// Builder for TemplateProcessing
#[derive(Debug, Clone, Default)]
pub struct TemplateProcessingBuilder {
    single: Option<String>,
    pair: Option<String>,
    special_tokens: Vec<(String, u32)>,
}

impl TemplateProcessingBuilder {
    /// Set the single sequence template
    pub fn try_single(mut self, template: &str) -> Result<Self, String> {
        self.single = Some(template.to_string());
        Ok(self)
    }

    /// Set the pair sequence template
    pub fn try_pair(mut self, template: &str) -> Result<Self, String> {
        self.pair = Some(template.to_string());
        Ok(self)
    }

    /// Set special tokens
    pub fn special_tokens(mut self, tokens: Vec<(&str, u32)>) -> Self {
        self.special_tokens = tokens.into_iter()
            .map(|(s, id)| (s.to_string(), id))
            .collect();
        self
    }

    /// Build the TemplateProcessing
    pub fn build(self) -> Result<TemplateProcessing, String> {
        Ok(TemplateProcessing {
            single: self.single,
            pair: self.pair,
            special_tokens: self.special_tokens,
        })
    }
}

/// Sequence of post-processors
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Sequence {
    processors: Vec<PostProcessorWrapper>,
}

impl Sequence {
    /// Create a new sequence from post-processors
    pub fn new(processors: Vec<PostProcessorWrapper>) -> Self {
        Self { processors }
    }
}

// Conversion implementations
impl From<TemplateProcessing> for PostProcessorWrapper {
    fn from(tp: TemplateProcessing) -> Self {
        PostProcessorWrapper::Template(tp)
    }
}

impl From<Sequence> for PostProcessorWrapper {
    fn from(seq: Sequence) -> Self {
        PostProcessorWrapper::Sequence(seq)
    }
}
