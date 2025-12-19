#!/usr/bin/env python3
"""Debug WordPiece tokenization differences between HuggingFace and BudTikTok."""

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

    documents = load_documents(DATA_PATH, 100)

    # Check specific mismatching documents
    mismatching_docs = [3, 16, 23, 26, 32]

    for doc_idx in mismatching_docs:
        if doc_idx >= len(documents):
            continue

        doc = documents[doc_idx]
        hf_tokens = tokenizer.encode(doc, add_special_tokens=False)
        bt_token_count = bt_results["token_counts"][doc_idx]

        print(f"\n{'='*80}")
        print(f"Document {doc_idx}: HF={len(hf_tokens)} tokens, BT={bt_token_count} tokens, diff={len(hf_tokens)-bt_token_count}")
        print(f"{'='*80}")

        # Show first 500 chars of document
        print(f"\nDocument preview (first 500 chars):")
        print(repr(doc[:500]))

        # Tokenize with HF
        hf_decoded = tokenizer.decode(hf_tokens)
        print(f"\nHF decoded (first 500 chars): {repr(hf_decoded[:500])}")

        # Show HF tokens
        hf_token_strs = tokenizer.convert_ids_to_tokens(hf_tokens[:50])
        print(f"\nFirst 50 HF tokens: {hf_token_strs}")

        if doc_idx in [3, 16]:  # Focus on smaller differences
            # Let's look for specific patterns
            print("\n--- Checking for Unicode issues ---")
            for i, c in enumerate(doc[:200]):
                if ord(c) > 127:
                    print(f"  Non-ASCII at pos {i}: {repr(c)} (U+{ord(c):04X})")

            # Check accents
            import unicodedata
            accented = [c for c in doc if unicodedata.combining(c) or
                       (ord(c) > 127 and unicodedata.category(c) == 'Lm')]
            if accented:
                print(f"  Found {len(accented)} accent/modifier chars: {accented[:20]}")

if __name__ == "__main__":
    main()
