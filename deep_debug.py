#!/usr/bin/env python3
"""Deep debug to find exact tokenization differences."""

import json
from transformers import BertTokenizerFast

DATA_PATH = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/openwebtext_1gb.jsonl"

def load_documents(path, limit=100):
    documents = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            try:
                data = json.loads(line)
                if 'text' in data:
                    documents.append(data['text'])
            except json.JSONDecodeError:
                continue
    return documents

def main():
    print("Loading HuggingFace BERT tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Get the backend tokenizer to access normalizer and pre-tokenizer
    backend = tokenizer.backend_tokenizer

    documents = load_documents(DATA_PATH, 50)

    # Test specific problematic characters
    print("\n=== Testing Specific Characters ===\n")

    test_cases = [
        ("en-dash", "word–word"),  # U+2013
        ("em-dash", "word—word"),  # U+2014
        ("ellipsis", "word…word"),  # U+2026
        ("curly apos", "it's"),     # U+2019
        ("straight apos", "it's"),  # U+0027
        ("nbspace", "word\u00a0word"),  # U+00A0
        ("mixed", "test–one…two's"),
    ]

    for name, text in test_cases:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_strs = tokenizer.convert_ids_to_tokens(tokens)

        # Also get normalized form
        normalized = backend.normalizer.normalize_str(text) if backend.normalizer else text

        print(f"{name:15} | Input: {repr(text):25} | Normalized: {repr(normalized):25} | Tokens: {token_strs}")

    # Now let's look at document 23 which has HF with MORE tokens (diff=9)
    print("\n\n=== Analyzing Document 23 (HF has 9 more tokens) ===\n")

    doc = documents[23]
    hf_tokens = tokenizer.encode(doc, add_special_tokens=False)
    hf_token_strs = tokenizer.convert_ids_to_tokens(hf_tokens)

    print(f"HF token count: {len(hf_tokens)}")
    print(f"\nFirst 200 chars: {repr(doc[:200])}")

    # Find unusual characters
    print(f"\nUnusual characters in document:")
    for i, c in enumerate(doc):
        if ord(c) > 127 or c in '–—…''""':
            context_start = max(0, i-10)
            context_end = min(len(doc), i+10)
            print(f"  pos {i}: U+{ord(c):04X} {repr(c)} context: ...{repr(doc[context_start:context_end])}...")
            if i > 500:
                print("  ... (truncated)")
                break

    # Check what HF's pre-tokenizer does
    print("\n\n=== HF Pre-tokenizer behavior ===\n")

    # The pre-tokenizer splits text into words
    pre_tokenizer = backend.pre_tokenizer
    if pre_tokenizer:
        test_text = "Hello–world…test's example"
        pre_tokens = pre_tokenizer.pre_tokenize_str(test_text)
        print(f"Input: {repr(test_text)}")
        print(f"Pre-tokenized: {pre_tokens}")

    # Check normalizer in detail
    print("\n\n=== HF Normalizer details ===\n")
    normalizer = backend.normalizer
    if normalizer:
        print(f"Normalizer type: {type(normalizer)}")
        # Test various Unicode characters
        chars_to_test = [
            '\u2013',  # EN DASH
            '\u2014',  # EM DASH
            '\u2026',  # HORIZONTAL ELLIPSIS
            '\u2018',  # LEFT SINGLE QUOTATION MARK
            '\u2019',  # RIGHT SINGLE QUOTATION MARK
            '\u201C',  # LEFT DOUBLE QUOTATION MARK
            '\u201D',  # RIGHT DOUBLE QUOTATION MARK
            '\u00A0',  # NO-BREAK SPACE
            '\u200B',  # ZERO WIDTH SPACE
            '\u00AD',  # SOFT HYPHEN
        ]
        for c in chars_to_test:
            normalized = normalizer.normalize_str(c)
            print(f"  U+{ord(c):04X} {repr(c):10} -> {repr(normalized)}")

if __name__ == "__main__":
    main()
