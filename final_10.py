#!/usr/bin/env python3
"""Investigate final 10 mismatching documents."""

import json
from transformers import BertTokenizerFast

DATA_PATH = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/openwebtext_1gb.jsonl"

def load_documents(path, limit=10000):
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
    backend = tokenizer.backend_tokenizer
    normalizer = backend.normalizer

    documents = load_documents(DATA_PATH, 10000)

    mismatching = [2420, 3974, 3996, 4330, 4455]

    for doc_idx in mismatching:
        if doc_idx >= len(documents):
            continue

        doc = documents[doc_idx]
        hf_tokens = tokenizer.encode(doc, add_special_tokens=False)

        print(f"\n{'='*80}")
        print(f"Document {doc_idx}: HF={len(hf_tokens)} tokens")
        print(f"{'='*80}")

        # Find unusual characters
        print("\nUnusual characters (non-ASCII):")
        unusual_count = 0
        for i, c in enumerate(doc):
            if ord(c) > 127:
                # Get context
                start = max(0, i-10)
                end = min(len(doc), i+20)
                print(f"  pos {i}: U+{ord(c):04X} {repr(c):10} | {repr(doc[start:end])}")
                unusual_count += 1
                if unusual_count >= 15:
                    print("  ... (more)")
                    break

        # Test normalization of problematic parts
        if unusual_count > 0:
            print("\nNormalization test:")
            for i, c in enumerate(doc):
                if ord(c) > 127:
                    start = max(0, i-5)
                    end = min(len(doc), i+10)
                    segment = doc[start:end]
                    if normalizer:
                        normalized = normalizer.normalize_str(segment)
                        if segment != normalized:
                            print(f"  {repr(segment)} -> {repr(normalized)}")
                    if i > 500:
                        break

if __name__ == "__main__":
    main()
