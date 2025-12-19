#!/usr/bin/env python3
"""Test handling of U+FFFD replacement character."""

from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
backend = tokenizer.backend_tokenizer
normalizer = backend.normalizer

test_cases = [
    "state\uFFFDs",  # replacement char
    "Rond\uFFFDnia",
    "\uFFFD",  # standalone
    "hello\uFFFDworld",
    "1080×1920",  # multiplication sign
    "×",  # standalone mult sign
]

print("HF Normalizer behavior with special chars:\n")
for text in test_cases:
    if normalizer:
        normalized = normalizer.normalize_str(text)
    else:
        normalized = text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_strs = tokenizer.convert_ids_to_tokens(tokens)
    print(f"Input: {repr(text):25} -> Normalized: {repr(normalized):25} -> Tokens: {token_strs}")

# Check Unicode category
import unicodedata
print(f"\nU+FFFD category: {unicodedata.category(chr(0xFFFD))} ({unicodedata.name(chr(0xFFFD), 'UNKNOWN')})")
print(f"U+00D7 category: {unicodedata.category(chr(0x00D7))} ({unicodedata.name(chr(0x00D7), 'UNKNOWN')})")
