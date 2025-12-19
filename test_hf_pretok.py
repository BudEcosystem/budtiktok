#!/usr/bin/env python3
"""Test HuggingFace pre-tokenizer behavior with different characters."""

from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
backend = tokenizer.backend_tokenizer
pre_tokenizer = backend.pre_tokenizer

test_cases = [
    "$100",
    "£150",
    "€50",
    "test$100",
    "test£150",
    "100$test",
    "a+b",
    "a-b",
    "a*b",
    "a/b",
    "a=b",
    "a<b",
    "a>b",
    "a@b",
    "a#b",
    "a%b",
    "a^b",
    "a&b",
    "test.com",
    "hello,world",
    "word;word",
    "word:word",
]

print("HuggingFace Pre-tokenizer behavior:\n")
for text in test_cases:
    pre_tokens = pre_tokenizer.pre_tokenize_str(text)
    tokens = [t for t, span in pre_tokens]
    print(f"{text:20} -> {tokens}")

# Also check what Unicode categories are split on
print("\n\nChecking specific Unicode categories:")
import unicodedata
chars_to_check = ['$', '£', '€', '+', '-', '*', '/', '=', '<', '>', '@', '#', '%', '^', '&', '.', ',', ';', ':', '!', '?']
for c in chars_to_check:
    cat = unicodedata.category(c)
    name = unicodedata.name(c, 'UNKNOWN')
    print(f"  {c!r:5} U+{ord(c):04X} | Category: {cat} | {name}")
