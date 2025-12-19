//! Post-processing for tokenization output
//!
//! Post-processors modify the tokenization output after the main tokenization.
//! Common operations include:
//! - Adding special tokens ([CLS], [SEP], [BOS], [EOS])
//! - Setting token type IDs for sentence pairs
//! - Truncation handling

use crate::encoding::Encoding;

/// Trait for post-processors
pub trait PostProcessor: Send + Sync {
    /// Process a single encoding (add special tokens, set type IDs, etc.)
    fn process(&self, encoding: Encoding, add_special_tokens: bool) -> Encoding;

    /// Process a pair of encodings (for sentence pair tasks)
    fn process_pair(
        &self,
        encoding: Encoding,
        pair: Encoding,
        add_special_tokens: bool,
    ) -> Encoding;

    /// Get the number of special tokens added by this processor for single sequences
    fn added_tokens(&self, is_pair: bool) -> usize;
}

/// Special token information
#[derive(Debug, Clone)]
pub struct SpecialToken {
    /// The token string
    pub token: String,
    /// The token ID
    pub id: u32,
    /// The token type ID (0 or 1 for BERT)
    pub type_id: usize,
}

impl SpecialToken {
    pub fn new(token: impl Into<String>, id: u32) -> Self {
        Self {
            token: token.into(),
            id,
            type_id: 0,
        }
    }

    pub fn with_type_id(mut self, type_id: usize) -> Self {
        self.type_id = type_id;
        self
    }
}

/// BERT-style post-processor
///
/// Adds [CLS] at the start and [SEP] at the end.
/// For pairs: [CLS] A [SEP] B [SEP]
/// Sets type_ids to 0 for first sequence, 1 for second.
#[derive(Debug, Clone)]
pub struct BertPostProcessor {
    /// [CLS] token
    pub cls: SpecialToken,
    /// [SEP] token
    pub sep: SpecialToken,
}

impl BertPostProcessor {
    /// Create a new BERT post-processor
    pub fn new(cls: SpecialToken, sep: SpecialToken) -> Self {
        Self { cls, sep }
    }

    /// Create with default tokens
    pub fn with_ids(cls_id: u32, sep_id: u32) -> Self {
        Self {
            cls: SpecialToken::new("[CLS]", cls_id),
            sep: SpecialToken::new("[SEP]", sep_id),
        }
    }
}

impl PostProcessor for BertPostProcessor {
    fn process(&self, encoding: Encoding, add_special_tokens: bool) -> Encoding {
        if !add_special_tokens {
            return encoding;
        }

        let mut result = Encoding::with_capacity(encoding.len() + 2);

        // Add [CLS]
        result.push(
            self.cls.id,
            self.cls.token.clone(),
            (0, 0),
            None,
            Some(self.cls.type_id),
            true,
        );

        // Add original tokens with type_id = 0
        for i in 0..encoding.len() {
            result.push(
                encoding.get_ids()[i],
                encoding.get_tokens()[i].clone(),
                encoding.get_offsets()[i],
                encoding.get_word_ids()[i],
                Some(0),
                encoding.get_special_tokens_mask()[i] == 1,
            );
        }

        // Add [SEP]
        let last_offset = encoding.get_offsets().last().map(|(_, e)| *e).unwrap_or(0);
        result.push(
            self.sep.id,
            self.sep.token.clone(),
            (last_offset, last_offset),
            None,
            Some(0),
            true,
        );

        result
    }

    fn process_pair(
        &self,
        encoding: Encoding,
        pair: Encoding,
        add_special_tokens: bool,
    ) -> Encoding {
        if !add_special_tokens {
            // Just merge without special tokens
            let mut result = encoding;
            result.merge(pair, false);
            return result;
        }

        let mut result = Encoding::with_capacity(encoding.len() + pair.len() + 3);

        // Add [CLS]
        result.push(
            self.cls.id,
            self.cls.token.clone(),
            (0, 0),
            None,
            Some(0),
            true,
        );

        // Add first sequence tokens with type_id = 0
        for i in 0..encoding.len() {
            result.push(
                encoding.get_ids()[i],
                encoding.get_tokens()[i].clone(),
                encoding.get_offsets()[i],
                encoding.get_word_ids()[i],
                Some(0),
                encoding.get_special_tokens_mask()[i] == 1,
            );
        }

        // Add first [SEP]
        let first_last_offset = encoding.get_offsets().last().map(|(_, e)| *e).unwrap_or(0);
        result.push(
            self.sep.id,
            self.sep.token.clone(),
            (first_last_offset, first_last_offset),
            None,
            Some(0),
            true,
        );

        // Add second sequence tokens with type_id = 1
        for i in 0..pair.len() {
            result.push(
                pair.get_ids()[i],
                pair.get_tokens()[i].clone(),
                pair.get_offsets()[i],
                pair.get_word_ids()[i],
                Some(1),
                pair.get_special_tokens_mask()[i] == 1,
            );
        }

        // Add second [SEP]
        let second_last_offset = pair.get_offsets().last().map(|(_, e)| *e).unwrap_or(0);
        result.push(
            self.sep.id,
            self.sep.token.clone(),
            (second_last_offset, second_last_offset),
            None,
            Some(1),
            true,
        );

        result
    }

    fn added_tokens(&self, is_pair: bool) -> usize {
        if is_pair {
            3 // [CLS] + [SEP] + [SEP]
        } else {
            2 // [CLS] + [SEP]
        }
    }
}

/// RoBERTa-style post-processor
///
/// Same as BERT but uses <s> and </s> tokens.
/// For pairs: <s> A </s></s> B </s>
#[derive(Debug, Clone)]
pub struct RobertaPostProcessor {
    /// <s> token
    pub bos: SpecialToken,
    /// </s> token
    pub eos: SpecialToken,
    /// Whether to add prefix space
    pub add_prefix_space: bool,
    /// Whether to trim offsets
    pub trim_offsets: bool,
}

impl RobertaPostProcessor {
    pub fn new(bos: SpecialToken, eos: SpecialToken) -> Self {
        Self {
            bos,
            eos,
            add_prefix_space: true,
            trim_offsets: true,
        }
    }

    pub fn with_ids(bos_id: u32, eos_id: u32) -> Self {
        Self {
            bos: SpecialToken::new("<s>", bos_id),
            eos: SpecialToken::new("</s>", eos_id),
            add_prefix_space: true,
            trim_offsets: true,
        }
    }
}

impl PostProcessor for RobertaPostProcessor {
    fn process(&self, encoding: Encoding, add_special_tokens: bool) -> Encoding {
        if !add_special_tokens {
            return encoding;
        }

        let mut result = Encoding::with_capacity(encoding.len() + 2);

        // Add <s>
        result.push(
            self.bos.id,
            self.bos.token.clone(),
            (0, 0),
            None,
            Some(0),
            true,
        );

        // Add original tokens
        for i in 0..encoding.len() {
            result.push(
                encoding.get_ids()[i],
                encoding.get_tokens()[i].clone(),
                encoding.get_offsets()[i],
                encoding.get_word_ids()[i],
                Some(0),
                encoding.get_special_tokens_mask()[i] == 1,
            );
        }

        // Add </s>
        let last_offset = encoding.get_offsets().last().map(|(_, e)| *e).unwrap_or(0);
        result.push(
            self.eos.id,
            self.eos.token.clone(),
            (last_offset, last_offset),
            None,
            Some(0),
            true,
        );

        result
    }

    fn process_pair(
        &self,
        encoding: Encoding,
        pair: Encoding,
        add_special_tokens: bool,
    ) -> Encoding {
        if !add_special_tokens {
            let mut result = encoding;
            result.merge(pair, false);
            return result;
        }

        let mut result = Encoding::with_capacity(encoding.len() + pair.len() + 4);

        // Add <s>
        result.push(
            self.bos.id,
            self.bos.token.clone(),
            (0, 0),
            None,
            Some(0),
            true,
        );

        // Add first sequence
        for i in 0..encoding.len() {
            result.push(
                encoding.get_ids()[i],
                encoding.get_tokens()[i].clone(),
                encoding.get_offsets()[i],
                encoding.get_word_ids()[i],
                Some(0),
                encoding.get_special_tokens_mask()[i] == 1,
            );
        }

        // Add </s></s> between sequences
        let first_last_offset = encoding.get_offsets().last().map(|(_, e)| *e).unwrap_or(0);
        result.push(
            self.eos.id,
            self.eos.token.clone(),
            (first_last_offset, first_last_offset),
            None,
            Some(0),
            true,
        );
        result.push(
            self.eos.id,
            self.eos.token.clone(),
            (first_last_offset, first_last_offset),
            None,
            Some(0),
            true,
        );

        // Add second sequence
        for i in 0..pair.len() {
            result.push(
                pair.get_ids()[i],
                pair.get_tokens()[i].clone(),
                pair.get_offsets()[i],
                pair.get_word_ids()[i],
                Some(0), // RoBERTa doesn't use type_ids
                pair.get_special_tokens_mask()[i] == 1,
            );
        }

        // Add final </s>
        let second_last_offset = pair.get_offsets().last().map(|(_, e)| *e).unwrap_or(0);
        result.push(
            self.eos.id,
            self.eos.token.clone(),
            (second_last_offset, second_last_offset),
            None,
            Some(0),
            true,
        );

        result
    }

    fn added_tokens(&self, is_pair: bool) -> usize {
        if is_pair {
            4 // <s> + </s> + </s> + </s>
        } else {
            2 // <s> + </s>
        }
    }
}

/// Template-based post-processor
///
/// Allows defining custom templates for special token insertion.
#[derive(Debug, Clone)]
pub struct TemplatePostProcessor {
    /// Template for single sequences
    pub single: Vec<TemplatePart>,
    /// Template for pair sequences
    pub pair: Vec<TemplatePart>,
    /// Special tokens map
    pub special_tokens: Vec<SpecialToken>,
}

/// Part of a template
#[derive(Debug, Clone)]
pub enum TemplatePart {
    /// Reference to a special token by name
    SpecialToken { name: String, type_id: usize },
    /// Placeholder for sequence A
    SequenceA { type_id: usize },
    /// Placeholder for sequence B
    SequenceB { type_id: usize },
}

impl TemplatePostProcessor {
    pub fn new(single: Vec<TemplatePart>, pair: Vec<TemplatePart>, special_tokens: Vec<SpecialToken>) -> Self {
        Self {
            single,
            pair,
            special_tokens,
        }
    }

    fn get_special_token(&self, name: &str) -> Option<&SpecialToken> {
        self.special_tokens.iter().find(|t| t.token == name)
    }
}

impl PostProcessor for TemplatePostProcessor {
    fn process(&self, encoding: Encoding, add_special_tokens: bool) -> Encoding {
        if !add_special_tokens || self.single.is_empty() {
            return encoding;
        }

        let mut result = Encoding::new();

        for part in &self.single {
            match part {
                TemplatePart::SpecialToken { name, type_id } => {
                    if let Some(token) = self.get_special_token(name) {
                        let last_offset = result.get_offsets().last().map(|(_, e)| *e).unwrap_or(0);
                        result.push(
                            token.id,
                            token.token.clone(),
                            (last_offset, last_offset),
                            None,
                            Some(*type_id),
                            true,
                        );
                    }
                }
                TemplatePart::SequenceA { type_id } => {
                    for i in 0..encoding.len() {
                        result.push(
                            encoding.get_ids()[i],
                            encoding.get_tokens()[i].clone(),
                            encoding.get_offsets()[i],
                            encoding.get_word_ids()[i],
                            Some(*type_id),
                            encoding.get_special_tokens_mask()[i] == 1,
                        );
                    }
                }
                TemplatePart::SequenceB { .. } => {
                    // Ignore sequence B in single mode
                }
            }
        }

        result
    }

    fn process_pair(
        &self,
        encoding: Encoding,
        pair: Encoding,
        add_special_tokens: bool,
    ) -> Encoding {
        if !add_special_tokens || self.pair.is_empty() {
            let mut result = encoding;
            result.merge(pair, false);
            return result;
        }

        let mut result = Encoding::new();

        for part in &self.pair {
            match part {
                TemplatePart::SpecialToken { name, type_id } => {
                    if let Some(token) = self.get_special_token(name) {
                        let last_offset = result.get_offsets().last().map(|(_, e)| *e).unwrap_or(0);
                        result.push(
                            token.id,
                            token.token.clone(),
                            (last_offset, last_offset),
                            None,
                            Some(*type_id),
                            true,
                        );
                    }
                }
                TemplatePart::SequenceA { type_id } => {
                    for i in 0..encoding.len() {
                        result.push(
                            encoding.get_ids()[i],
                            encoding.get_tokens()[i].clone(),
                            encoding.get_offsets()[i],
                            encoding.get_word_ids()[i],
                            Some(*type_id),
                            encoding.get_special_tokens_mask()[i] == 1,
                        );
                    }
                }
                TemplatePart::SequenceB { type_id } => {
                    for i in 0..pair.len() {
                        result.push(
                            pair.get_ids()[i],
                            pair.get_tokens()[i].clone(),
                            pair.get_offsets()[i],
                            pair.get_word_ids()[i],
                            Some(*type_id),
                            pair.get_special_tokens_mask()[i] == 1,
                        );
                    }
                }
            }
        }

        result
    }

    fn added_tokens(&self, is_pair: bool) -> usize {
        let template = if is_pair { &self.pair } else { &self.single };
        template
            .iter()
            .filter(|p| matches!(p, TemplatePart::SpecialToken { .. }))
            .count()
    }
}

/// ByteLevel post-processor
///
/// Used with byte-level BPE (GPT-2 style).
#[derive(Debug, Clone, Default)]
pub struct ByteLevelPostProcessor {
    /// Whether to trim offsets to actual content
    pub trim_offsets: bool,
}

impl ByteLevelPostProcessor {
    pub fn new() -> Self {
        Self { trim_offsets: true }
    }
}

impl PostProcessor for ByteLevelPostProcessor {
    fn process(&self, encoding: Encoding, _add_special_tokens: bool) -> Encoding {
        // Byte-level post-processor typically doesn't add special tokens
        // but may adjust offsets
        encoding
    }

    fn process_pair(
        &self,
        encoding: Encoding,
        pair: Encoding,
        _add_special_tokens: bool,
    ) -> Encoding {
        let mut result = encoding;
        result.merge(pair, false);
        result
    }

    fn added_tokens(&self, _is_pair: bool) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_encoding(tokens: &[(&str, u32)]) -> Encoding {
        let mut encoding = Encoding::new();
        let mut offset = 0;
        for (i, (token, id)) in tokens.iter().enumerate() {
            let len = token.len();
            encoding.push(
                *id,
                token.to_string(),
                (offset, offset + len),
                Some(i as u32),
                Some(0),
                false,
            );
            offset += len + 1; // +1 for space
        }
        encoding
    }

    #[test]
    fn test_bert_postprocessor_single() {
        let processor = BertPostProcessor::with_ids(101, 102);
        let encoding = create_test_encoding(&[("hello", 1000), ("world", 1001)]);

        let result = processor.process(encoding, true);

        assert_eq!(result.len(), 4); // [CLS] hello world [SEP]
        assert_eq!(result.get_ids()[0], 101); // [CLS]
        assert_eq!(result.get_ids()[1], 1000); // hello
        assert_eq!(result.get_ids()[2], 1001); // world
        assert_eq!(result.get_ids()[3], 102); // [SEP]

        // Check type IDs
        assert_eq!(result.get_type_ids()[0], 0);
        assert_eq!(result.get_type_ids()[1], 0);
        assert_eq!(result.get_type_ids()[2], 0);
        assert_eq!(result.get_type_ids()[3], 0);

        // Check special tokens mask
        assert_eq!(result.get_special_tokens_mask()[0], 1);
        assert_eq!(result.get_special_tokens_mask()[1], 0);
        assert_eq!(result.get_special_tokens_mask()[2], 0);
        assert_eq!(result.get_special_tokens_mask()[3], 1);
    }

    #[test]
    fn test_bert_postprocessor_pair() {
        let processor = BertPostProcessor::with_ids(101, 102);
        let encoding1 = create_test_encoding(&[("hello", 1000)]);
        let encoding2 = create_test_encoding(&[("world", 1001)]);

        let result = processor.process_pair(encoding1, encoding2, true);

        assert_eq!(result.len(), 5); // [CLS] hello [SEP] world [SEP]
        assert_eq!(result.get_ids()[0], 101); // [CLS]
        assert_eq!(result.get_ids()[1], 1000); // hello
        assert_eq!(result.get_ids()[2], 102); // [SEP]
        assert_eq!(result.get_ids()[3], 1001); // world
        assert_eq!(result.get_ids()[4], 102); // [SEP]

        // Check type IDs
        assert_eq!(result.get_type_ids()[0], 0); // [CLS] -> type 0
        assert_eq!(result.get_type_ids()[1], 0); // hello -> type 0
        assert_eq!(result.get_type_ids()[2], 0); // [SEP] -> type 0
        assert_eq!(result.get_type_ids()[3], 1); // world -> type 1
        assert_eq!(result.get_type_ids()[4], 1); // [SEP] -> type 1
    }

    #[test]
    fn test_bert_postprocessor_no_special_tokens() {
        let processor = BertPostProcessor::with_ids(101, 102);
        let encoding = create_test_encoding(&[("hello", 1000), ("world", 1001)]);

        let result = processor.process(encoding.clone(), false);

        // Should be unchanged
        assert_eq!(result.len(), encoding.len());
        assert_eq!(result.get_ids(), encoding.get_ids());
    }

    #[test]
    fn test_bert_postprocessor_added_tokens() {
        let processor = BertPostProcessor::with_ids(101, 102);

        assert_eq!(processor.added_tokens(false), 2);
        assert_eq!(processor.added_tokens(true), 3);
    }

    #[test]
    fn test_roberta_postprocessor_single() {
        let processor = RobertaPostProcessor::with_ids(0, 2);
        let encoding = create_test_encoding(&[("hello", 1000)]);

        let result = processor.process(encoding, true);

        assert_eq!(result.len(), 3); // <s> hello </s>
        assert_eq!(result.get_ids()[0], 0); // <s>
        assert_eq!(result.get_ids()[1], 1000); // hello
        assert_eq!(result.get_ids()[2], 2); // </s>
    }

    #[test]
    fn test_roberta_postprocessor_pair() {
        let processor = RobertaPostProcessor::with_ids(0, 2);
        let encoding1 = create_test_encoding(&[("hello", 1000)]);
        let encoding2 = create_test_encoding(&[("world", 1001)]);

        let result = processor.process_pair(encoding1, encoding2, true);

        assert_eq!(result.len(), 6); // <s> hello </s></s> world </s>
        assert_eq!(result.get_ids()[0], 0); // <s>
        assert_eq!(result.get_ids()[1], 1000); // hello
        assert_eq!(result.get_ids()[2], 2); // </s>
        assert_eq!(result.get_ids()[3], 2); // </s>
        assert_eq!(result.get_ids()[4], 1001); // world
        assert_eq!(result.get_ids()[5], 2); // </s>
    }

    #[test]
    fn test_template_postprocessor() {
        let processor = TemplatePostProcessor::new(
            vec![
                TemplatePart::SpecialToken { name: "[CLS]".to_string(), type_id: 0 },
                TemplatePart::SequenceA { type_id: 0 },
                TemplatePart::SpecialToken { name: "[SEP]".to_string(), type_id: 0 },
            ],
            vec![
                TemplatePart::SpecialToken { name: "[CLS]".to_string(), type_id: 0 },
                TemplatePart::SequenceA { type_id: 0 },
                TemplatePart::SpecialToken { name: "[SEP]".to_string(), type_id: 0 },
                TemplatePart::SequenceB { type_id: 1 },
                TemplatePart::SpecialToken { name: "[SEP]".to_string(), type_id: 1 },
            ],
            vec![
                SpecialToken::new("[CLS]", 101),
                SpecialToken::new("[SEP]", 102),
            ],
        );

        let encoding = create_test_encoding(&[("hello", 1000)]);
        let result = processor.process(encoding, true);

        assert_eq!(result.len(), 3);
        assert_eq!(result.get_ids()[0], 101);
        assert_eq!(result.get_ids()[1], 1000);
        assert_eq!(result.get_ids()[2], 102);
    }
}
