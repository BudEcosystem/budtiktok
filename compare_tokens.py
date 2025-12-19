#!/usr/bin/env python3
"""Compare exact token outputs between HuggingFace and BudTikTok."""

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

    # Load BudTikTok results
    with open("/home/bud/Desktop/latentbud/budtiktok/wordpiece_budtiktok_results.json") as f:
        bt_results = json.load(f)

    documents = load_documents(DATA_PATH, 50)

    # Check document 3 (HF=1427, BT=1429, diff=-2)
    doc_idx = 3
    doc = documents[doc_idx]

    print(f"\n=== Analyzing Document {doc_idx} ===")
    print(f"Document length: {len(doc)} chars")

    # Get HF tokens
    hf_tokens = tokenizer.encode(doc, add_special_tokens=False)
    hf_token_strs = tokenizer.convert_ids_to_tokens(hf_tokens)
    bt_tokens = bt_results["first_10_docs_tokens"][doc_idx]

    print(f"HF token count: {len(hf_tokens)}")
    print(f"BT token count: {len(bt_tokens)}")

    # Find differences
    print(f"\n--- Finding first divergence ---")
    min_len = min(len(hf_token_strs), len(bt_tokens))

    # We need to convert BT tokens (IDs) to strings
    # For now, let's compare token counts by segment

    # Let's check what HF's normalizer does
    backend = tokenizer.backend_tokenizer
    normalizer = backend.normalizer

    # Test normalization on specific characters
    test_chars = [
        '\u2019',  # RIGHT SINGLE QUOTATION MARK
        '\u2018',  # LEFT SINGLE QUOTATION MARK
        '\u201c',  # LEFT DOUBLE QUOTATION MARK
        '\u201d',  # RIGHT DOUBLE QUOTATION MARK
        '\u0027',  # APOSTROPHE
        '\u00a0',  # NO-BREAK SPACE
        '\u2013',  # EN DASH
        '\u2014',  # EM DASH
    ]

    print(f"\n--- Unicode Character Normalization ---")
    for c in test_chars:
        if normalizer:
            normalized = normalizer.normalize_str(c)
            print(f"U+{ord(c):04X} {repr(c)} -> {repr(normalized)}")

    # Check for unusual characters in the document
    print(f"\n--- Characters in doc that might differ ---")
    unusual = {}
    for i, c in enumerate(doc):
        if ord(c) > 127:
            if c not in unusual:
                unusual[c] = []
            unusual[c].append(i)

    for c, positions in list(unusual.items())[:10]:
        print(f"  U+{ord(c):04X} {repr(c)} at positions: {positions[:5]}{'...' if len(positions) > 5 else ''}")

    # Try to find which token is different
    print(f"\n--- Comparing token by token (first 100) ---")
    # Convert HF tokens to see pattern
    for i in range(min(100, len(hf_token_strs))):
        hf_tok = hf_token_strs[i]
        bt_id = bt_tokens[i] if i < len(bt_tokens) else None
        if i < 50:
            print(f"{i:3d}: HF={hf_tok!r:20s} BT_ID={bt_id}")

if __name__ == "__main__":
    main()
